import math as m
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_optics.utils_fft import centered_fft2, centered_ifft2


class PropagationLayer(nn.Module):
    def __init__(
        self,
        Mp: int = 300,
        L: float = 1.0,
        zi: float = 2.0,
        wave_lengths: Optional[torch.Tensor] = None,
        trainable_z: bool = True,
        padding_factor: int = 1,
    ):
        super().__init__()
        self.Mp = int(Mp)
        self.L = float(L)
        if isinstance(padding_factor, bool) or not isinstance(padding_factor, int):
            raise TypeError(
                f"padding_factor must be an integer >= 1, got {padding_factor!r}"
            )
        if padding_factor < 1:
            raise ValueError(f"padding_factor must be >= 1, got {padding_factor}")
        self.padding_factor = padding_factor
        self.work_Mp = self.Mp * self.padding_factor
        # Grow N and L together so the original spatial pitch dx=L/N is
        # unchanged while the larger calculation window suppresses FFT
        # periodic wrap-around.
        self.work_L = self.L * self.padding_factor

        if wave_lengths is None:
            wave_lengths = torch.from_numpy(
                np.linspace(420, 660, 25).astype(np.float32) * 1e-9
            )
        else:
            wave_lengths = torch.as_tensor(wave_lengths, dtype=torch.float32)
        self.register_buffer("wave_lengths", wave_lengths)

        z_init = torch.tensor([float(zi)], dtype=torch.float32)
        if trainable_z:
            self.z = nn.Parameter(z_init)
        else:
            self.register_buffer("z", z_init)
        # Fixed-distance kernels are expensive at padded sizes. Keep a lazy
        # per-process cache outside registered buffers: DDP would otherwise
        # broadcast every padded kernel. Trainable z must rebuild the kernel
        # on every forward so its gradient remains valid.
        self._fixed_kernel_cache: Optional[torch.Tensor] = None

    def _build_kernel(self, device: torch.device) -> torch.Tensor:
        dx = self.work_L / self.work_Mp
        fx = torch.linspace(
            -1.0 / (2.0 * dx),
            1.0 / (2.0 * dx) - 1.0 / self.work_L,
            self.work_Mp,
            device=device,
            dtype=torch.float32,
        )
        ffx, ffy = torch.meshgrid(fx, fx, indexing="xy")
        freq2 = (ffx**2 + ffy**2)[None, :, :]

        # TensorFlow NonNeg constraint parity.
        z_eff = torch.clamp(self.z, min=0.0).to(device=device, dtype=torch.complex64)
        lambdas = self.wave_lengths.to(device=device, dtype=torch.float32)
        kernel = torch.exp(
            (-1j * m.pi * lambdas[:, None, None] * z_eff) * freq2.to(torch.complex64)
        )
        return torch.fft.fftshift(kernel, dim=(-2, -1)).unsqueeze(0)

    def _kernel(self, device: torch.device) -> torch.Tensor:
        if self.z.requires_grad:
            return self._build_kernel(device)
        expected_shape = (
            1,
            int(self.wave_lengths.numel()),
            self.work_Mp,
            self.work_Mp,
        )
        cached = self._fixed_kernel_cache
        if (
            cached is None
            or tuple(cached.shape) != expected_shape
            or cached.device != device
        ):
            self._fixed_kernel_cache = self._build_kernel(device).detach()
        return self._fixed_kernel_cache

    def _apply(self, fn):
        # The cache is deliberately not a registered buffer, so discard it
        # before device/dtype transforms and rebuild lazily.
        self._fixed_kernel_cache = None
        return super()._apply(fn)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # A fixed z may change when loading a checkpoint into an already-used
        # module. Its old lazy kernel must not survive that state change.
        self._fixed_kernel_cache = None
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _pad_to_work_grid(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        if self.padding_factor == 1:
            return x, (0, 0)
        pad_h = self.work_Mp - x.shape[-2]
        pad_w = self.work_Mp - x.shape[-1]
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        return F.pad(x, (left, right, top, bottom)), (top, left)

    def _crop_from_work_grid(
        self,
        x: torch.Tensor,
        crop_offset: tuple[int, int],
    ) -> torch.Tensor:
        if self.padding_factor == 1:
            return x
        top, left = crop_offset
        return x[..., top : top + self.Mp, left : left + self.Mp]

    def _propagate_work_grid(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        if x.ndim != 4:
            raise ValueError(
                f"PropagationLayer expects 4D tensor [B, C, H, W], got {tuple(x.shape)}"
            )

        b, c, h, w = x.shape
        if h != self.Mp or w != self.Mp:
            raise ValueError(
                f"PropagationLayer expects spatial size {self.Mp}x{self.Mp}, got {h}x{w}"
            )
        if c != int(self.wave_lengths.numel()):
            raise ValueError(
                f"PropagationLayer expects {self.wave_lengths.numel()} bands, got {c}"
            )

        x_work, crop_offset = self._pad_to_work_grid(x)
        x_complex = x_work.to(torch.complex64)
        u1f = centered_fft2(x_complex, dim=(-2, -1))
        u2f = u1f * self._kernel(x.device)
        output_work = centered_ifft2(u2f, dim=(-2, -1))
        return output_work, crop_offset

    def forward_work_grid(self, x: torch.Tensor) -> torch.Tensor:
        """Propagate on the padded grid without discarding its outer support.

        ``forward`` retains the historical API and center-crops back to
        ``Mp``.  PSF generation sometimes needs the complete padded sensor
        field so it can normalize against all propagated energy before
        selecting a finite convolution kernel; this method exposes that field
        while sharing the exact same validation, kernel, and FFT path.
        """
        output_work, _ = self._propagate_work_grid(x)
        return output_work

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_work, crop_offset = self._propagate_work_grid(x)
        return self._crop_from_work_grid(output_work, crop_offset)

    def adjoint(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the discrete adjoint of the padded-and-cropped propagation.

        This is not a negative-distance approximation. It exactly reverses
        the linear algebra used by :meth:`forward`: the sensor crop is
        zero-embedded on the work grid, the transfer kernel is conjugated,
        and the result is cropped back to the DOE grid. It is useful for
        phase-conjugate initialization and inverse-problem diagnostics.
        """
        if x.ndim != 4:
            raise ValueError(
                "PropagationLayer.adjoint expects 4D tensor [B,C,H,W], "
                f"got {tuple(x.shape)}"
            )
        if x.shape[-2:] != (self.Mp, self.Mp):
            raise ValueError(
                f"PropagationLayer.adjoint expects {self.Mp}x{self.Mp}, "
                f"got {tuple(x.shape[-2:])}"
            )
        if x.shape[1] != int(self.wave_lengths.numel()):
            raise ValueError(
                "PropagationLayer.adjoint wavelength count mismatch: "
                f"expected {self.wave_lengths.numel()}, got {x.shape[1]}"
            )

        x_work, crop_offset = self._pad_to_work_grid(x)
        spectrum = centered_fft2(x_work.to(torch.complex64), dim=(-2, -1))
        backpropagated = centered_ifft2(
            spectrum * torch.conj(self._kernel(x.device)), dim=(-2, -1)
        )
        return self._crop_from_work_grid(backpropagated, crop_offset)


class PadoFresnelPropagationLayer(nn.Module):
    """PADO-compatible spatial-domain Fresnel linear convolution.

    This layer intentionally mirrors the discretization used by PADO's
    ``Propagator('Fresnel')`` with ``linear=True``: the input is symmetrically
    zero-padded to twice its native size, convolved with a sampled positive-
    phase Fresnel impulse response, and cropped back to the native grid.
    It is kept separate from :class:`PropagationLayer` so existing optical
    forward modes retain their historical transfer-function semantics.
    """

    def __init__(
        self,
        Mp: int,
        L: float,
        zi: float,
        wave_lengths: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.Mp = int(Mp)
        self.L = float(L)
        if self.Mp < 1 or self.L <= 0.0:
            raise ValueError("Mp and L must be positive")
        if self.Mp % 2 != 0:
            raise ValueError("PADO-compatible linear padding requires even Mp")
        self.padding_factor = 2
        self.work_Mp = 2 * self.Mp
        self.work_L = 2.0 * self.L

        if wave_lengths is None:
            wave_lengths = torch.from_numpy(
                np.linspace(420, 660, 25).astype(np.float32) * 1e-9
            )
        else:
            wave_lengths = torch.as_tensor(wave_lengths, dtype=torch.float32)
        self.register_buffer("wave_lengths", wave_lengths)
        self.register_buffer("z", torch.tensor([float(zi)], dtype=torch.float32))
        self._fixed_kernel_fft_cache: Optional[torch.Tensor] = None

    @staticmethod
    def _fft2(x: torch.Tensor) -> torch.Tensor:
        shifted = torch.fft.ifftshift(x, dim=(-2, -1))
        return torch.fft.fftshift(
            torch.fft.fft2(shifted, dim=(-2, -1)), dim=(-2, -1)
        )

    @staticmethod
    def _ifft2(x: torch.Tensor) -> torch.Tensor:
        shifted = torch.fft.ifftshift(x, dim=(-2, -1))
        return torch.fft.fftshift(
            torch.fft.ifft2(shifted, dim=(-2, -1)), dim=(-2, -1)
        )

    def _build_kernel_fft(self, device: torch.device) -> torch.Tensor:
        pitch = self.L / self.Mp
        coordinates = torch.arange(
            -self.Mp,
            self.Mp,
            device=device,
            dtype=torch.float32,
        )
        yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
        radius_squared = (xx * pitch).square() + (yy * pitch).square()
        wavelengths = self.wave_lengths.to(device=device, dtype=torch.float32)
        z = self.z.to(device=device, dtype=torch.float32).clamp_min(0.0)
        if not bool(torch.all(z > 0.0)):
            raise ValueError("PADO Fresnel propagation distance must be positive")

        phase = (
            (2.0 * m.pi / wavelengths[:, None, None])
            * radius_squared[None]
            / (2.0 * z)
        )
        # PADO wraps to [-pi, pi] before taking the complex exponential.
        phase = torch.remainder(phase, 2.0 * m.pi)
        phase = torch.where(phase > m.pi, phase - 2.0 * m.pi, phase)
        amplitude = torch.ones_like(phase) / z / wavelengths[:, None, None]
        kernel = amplitude.to(torch.complex64) * torch.exp(
            1j * phase.to(torch.complex64)
        )
        kernel = kernel / kernel.abs().sum(dim=(-2, -1), keepdim=True)
        return self._fft2(kernel).unsqueeze(0)

    def _kernel_fft(self, device: torch.device) -> torch.Tensor:
        expected = (
            1,
            int(self.wave_lengths.numel()),
            self.work_Mp,
            self.work_Mp,
        )
        cached = self._fixed_kernel_fft_cache
        if cached is None or cached.device != device or tuple(cached.shape) != expected:
            self._fixed_kernel_fft_cache = self._build_kernel_fft(device).detach()
        return self._fixed_kernel_fft_cache

    def _apply(self, fn):
        self._fixed_kernel_fft_cache = None
        return super()._apply(fn)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        self._fixed_kernel_fft_cache = None
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _validate(self, x: torch.Tensor) -> None:
        if x.ndim != 4:
            raise ValueError(
                "PadoFresnelPropagationLayer expects [B,C,H,W], "
                f"got {tuple(x.shape)}"
            )
        if x.shape[-2:] != (self.Mp, self.Mp):
            raise ValueError(
                f"Expected spatial size {self.Mp}x{self.Mp}, "
                f"got {tuple(x.shape[-2:])}"
            )
        if x.shape[1] != int(self.wave_lengths.numel()):
            raise ValueError(
                f"Expected {self.wave_lengths.numel()} wavelengths, "
                f"got {x.shape[1]}"
            )

    def forward_work_grid(self, x: torch.Tensor) -> torch.Tensor:
        self._validate(x)
        half = self.Mp // 2
        padded = F.pad(x.to(torch.complex64), (half, half, half, half))
        return self._ifft2(self._fft2(padded) * self._kernel_fft(x.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.forward_work_grid(x)
        half = self.Mp // 2
        return output[..., half:-half, half:-half]
