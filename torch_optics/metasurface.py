from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint


class _TiO2SirenSurrogate(nn.Module):
    """Standalone copy of the trained 3 -> 512 -> 512 -> 6 SIREN."""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(3, 512),
            nn.Linear(512, 512),
        ])
        self.last_layer = nn.Linear(512, 6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.sin(30.0 * self.layers[0](x))
        x = torch.sin(self.layers[1](x))
        return self.last_layer(x)


def _load_checkpoint_state(path: Path) -> Tuple[Dict[str, torch.Tensor], Optional[int]]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")

    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported TiO2 checkpoint payload in '{path}'")
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, dict):
        raise ValueError(f"TiO2 checkpoint has no state_dict in '{path}'")

    remapped = {}
    for name, tensor in state.items():
        if name.startswith("model."):
            name = name[len("model."):]
        remapped[name] = tensor
    return remapped, checkpoint.get("epoch")


class TiO2ScalarMetasurface(nn.Module):
    """Scalar x/y-polarized TiO2 nanofin transmission from a frozen FDTD surrogate.

    The trainable quantities are two global geometry maps. The surrogate parameters
    stay frozen, while autograd remains enabled through the surrogate so task losses
    can update the length and width maps.
    """

    LENGTH_BOUNDS_M = (80e-9, 300e-9)
    WIDTH_BOUNDS_M = (80e-9, 300e-9)
    WAVELENGTH_BOUNDS_M = (400e-9, 700e-9)
    VALID_POLARIZATIONS = {"x", "y"}

    def __init__(
        self,
        checkpoint_path: str,
        spatial_size: int = 128,
        wave_lengths: Optional[torch.Tensor] = None,
        trainable_geometry: bool = True,
        polarization: str = "x",
        geometry_seed: int = 123,
        init_logit_range: float = 1.0,
        mlp_chunk_size: int = 16384,
        use_activation_checkpoint: bool = True,
        clamp_amplitude: bool = True,
        cache_frozen: bool = True,
        phase_eps: float = 1e-8,
    ):
        super().__init__()
        checkpoint_file = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint_file.is_file():
            raise FileNotFoundError(
                f"TiO2 metasurface checkpoint not found: {checkpoint_file}"
            )
        if spatial_size < 1:
            raise ValueError(f"spatial_size must be positive, got {spatial_size}")
        if mlp_chunk_size < 1:
            raise ValueError(f"mlp_chunk_size must be positive, got {mlp_chunk_size}")
        if phase_eps <= 0:
            raise ValueError(f"phase_eps must be positive, got {phase_eps}")
        if init_logit_range < 0:
            raise ValueError(
                f"init_logit_range must be non-negative, got {init_logit_range}"
            )
        polarization = str(polarization).lower()
        if polarization not in self.VALID_POLARIZATIONS:
            raise ValueError(
                f"polarization must be one of {self.VALID_POLARIZATIONS}, "
                f"got '{polarization}'"
            )

        if wave_lengths is None:
            wave_lengths = torch.linspace(420e-9, 660e-9, 25, dtype=torch.float32)
        wave_lengths = torch.as_tensor(wave_lengths, dtype=torch.float32)
        if wave_lengths.ndim != 1 or wave_lengths.numel() == 0:
            raise ValueError(
                "wave_lengths must be a non-empty one-dimensional tensor, "
                f"got {tuple(wave_lengths.shape)}"
            )
        wl_lo, wl_hi = self.WAVELENGTH_BOUNDS_M
        if torch.any(wave_lengths < wl_lo) or torch.any(wave_lengths > wl_hi):
            raise ValueError(
                f"All wavelengths must lie in [{wl_lo:g}, {wl_hi:g}] m"
            )

        self.spatial_size = int(spatial_size)
        self.trainable_geometry = bool(trainable_geometry)
        self.polarization = polarization
        self.mlp_chunk_size = int(mlp_chunk_size)
        self.use_activation_checkpoint = bool(use_activation_checkpoint)
        self.clamp_amplitude = bool(clamp_amplitude)
        self.cache_frozen = bool(cache_frozen)
        self.phase_eps = float(phase_eps)
        self.checkpoint_path = str(checkpoint_file)

        self.register_buffer("wave_lengths", wave_lengths)
        self.register_buffer(
            "length_bounds_m", torch.tensor(self.LENGTH_BOUNDS_M, dtype=torch.float32)
        )
        self.register_buffer(
            "width_bounds_m", torch.tensor(self.WIDTH_BOUNDS_M, dtype=torch.float32)
        )
        self.register_buffer(
            "wavelength_bounds_m",
            torch.tensor(self.WAVELENGTH_BOUNDS_M, dtype=torch.float32),
        )

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(geometry_seed))
        shape = (self.spatial_size, self.spatial_size)
        length_raw = torch.empty(shape, dtype=torch.float32).uniform_(
            -init_logit_range, init_logit_range, generator=generator
        )
        width_raw = torch.empty(shape, dtype=torch.float32).uniform_(
            -init_logit_range, init_logit_range, generator=generator
        )
        self.length_raw = nn.Parameter(
            length_raw, requires_grad=self.trainable_geometry
        )
        self.width_raw = nn.Parameter(
            width_raw, requires_grad=self.trainable_geometry
        )

        self.surrogate = _TiO2SirenSurrogate()
        state, checkpoint_epoch = _load_checkpoint_state(checkpoint_file)
        self.surrogate.load_state_dict(state, strict=True)
        for parameter in self.surrogate.parameters():
            parameter.requires_grad_(False)
        self.surrogate.eval()
        self.checkpoint_epoch = checkpoint_epoch

        self._cached_transmission: Optional[torch.Tensor] = None
        self._latest_transmission: Optional[torch.Tensor] = None

    def train(self, mode: bool = True):
        super().train(mode)
        # The surrogate represents fixed FDTD physics and is never trainable.
        self.surrogate.eval()
        return self

    def _apply(self, fn):
        self.clear_cache()
        return super()._apply(fn)

    def _load_from_state_dict(self, *args, **kwargs):
        self.clear_cache()
        return super()._load_from_state_dict(*args, **kwargs)

    def clear_cache(self):
        self._cached_transmission = None
        self._latest_transmission = None

    @staticmethod
    def _bounded(raw: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
        return bounds[0] + (bounds[1] - bounds[0]) * torch.sigmoid(raw)

    def geometry_m(self) -> Tuple[torch.Tensor, torch.Tensor]:
        length = self._bounded(self.length_raw, self.length_bounds_m)
        width = self._bounded(self.width_raw, self.width_bounds_m)
        return length, width

    def geometry_nm(self) -> Tuple[torch.Tensor, torch.Tensor]:
        length, width = self.geometry_m()
        return length * 1e9, width * 1e9

    def design_named_parameters(self) -> Iterable[Tuple[str, nn.Parameter]]:
        yield "metasurface_length_raw", self.length_raw
        yield "metasurface_width_raw", self.width_raw

    def _surrogate_forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = []
        use_checkpoint = (
            self.use_activation_checkpoint
            and torch.is_grad_enabled()
            and inputs.requires_grad
        )
        for chunk in inputs.split(self.mlp_chunk_size, dim=0):
            if use_checkpoint:
                output = activation_checkpoint(
                    self.surrogate, chunk, use_reentrant=False
                )
            else:
                output = self.surrogate(chunk)
            outputs.append(output)
        return torch.cat(outputs, dim=0)

    def _compute_transmission(self) -> torch.Tensor:
        length, width = self.geometry_m()
        num_wavelengths = int(self.wave_lengths.numel())
        height = width_px = self.spatial_size

        length_norm = (
            (length - self.length_bounds_m[0])
            / (self.length_bounds_m[1] - self.length_bounds_m[0])
        )
        width_norm = (
            (width - self.width_bounds_m[0])
            / (self.width_bounds_m[1] - self.width_bounds_m[0])
        )
        wavelength_norm = (
            (self.wave_lengths - self.wavelength_bounds_m[0])
            / (self.wavelength_bounds_m[1] - self.wavelength_bounds_m[0])
        )

        inputs = torch.stack(
            [
                length_norm.unsqueeze(0).expand(num_wavelengths, -1, -1),
                width_norm.unsqueeze(0).expand(num_wavelengths, -1, -1),
                wavelength_norm[:, None, None].expand(-1, height, width_px),
            ],
            dim=-1,
        ).reshape(-1, 3)
        raw = self._surrogate_forward(inputs).reshape(
            num_wavelengths, height, width_px, 6
        )

        offset = 0 if self.polarization == "x" else 3
        amplitude = raw[..., offset]
        if self.clamp_amplitude:
            amplitude = amplitude.clamp(0.0, 1.0)
        phase_real = 2.0 * raw[..., offset + 1] - 1.0
        phase_imag = 2.0 * raw[..., offset + 2] - 1.0
        phase_norm = torch.sqrt(
            phase_real.square() + phase_imag.square() + self.phase_eps
        )
        phase_real = phase_real / phase_norm
        phase_imag = phase_imag / phase_norm
        return torch.complex(amplitude * phase_real, amplitude * phase_imag)

    def complex_transmission(self) -> torch.Tensor:
        frozen_geometry = not (
            self.length_raw.requires_grad or self.width_raw.requires_grad
        )
        if self.cache_frozen and frozen_geometry:
            if self._cached_transmission is None:
                self._cached_transmission = self._compute_transmission().detach()
            transmission = self._cached_transmission
        else:
            transmission = self._compute_transmission()
        self._latest_transmission = transmission.detach()
        return transmission

    def forward(
        self,
        x: torch.Tensor,
        transmission: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"TiO2ScalarMetasurface expects [B,C,H,W], got {tuple(x.shape)}"
            )
        expected = (
            int(self.wave_lengths.numel()),
            self.spatial_size,
            self.spatial_size,
        )
        if tuple(x.shape[1:]) != expected:
            raise ValueError(
                f"TiO2ScalarMetasurface expects channel/spatial shape {expected}, "
                f"got {tuple(x.shape[1:])}"
            )
        if transmission is None:
            transmission = self.complex_transmission()
        if tuple(transmission.shape) != expected:
            raise ValueError(
                f"transmission must have shape {expected}, got {tuple(transmission.shape)}"
            )
        return x.to(torch.complex64) * transmission.unsqueeze(0)

    @torch.no_grad()
    def diagnostics(self) -> Dict[str, float]:
        length_nm, width_nm = self.geometry_nm()
        transmission = self._latest_transmission
        if transmission is None:
            transmission = self.complex_transmission()
        amplitude = transmission.abs()
        return {
            "length_min_nm": float(length_nm.min().item()),
            "length_max_nm": float(length_nm.max().item()),
            "length_mean_nm": float(length_nm.mean().item()),
            "width_min_nm": float(width_nm.min().item()),
            "width_max_nm": float(width_nm.max().item()),
            "width_mean_nm": float(width_nm.mean().item()),
            "amplitude_min": float(amplitude.min().item()),
            "amplitude_max": float(amplitude.max().item()),
            "amplitude_mean": float(amplitude.mean().item()),
            "transmission_power_mean": float(amplitude.square().mean().item()),
        }
