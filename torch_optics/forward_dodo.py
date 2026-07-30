from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_optics.doe import DOELayer, DOEFreeLayer
from torch_optics.propagation import PropagationLayer
from torch_optics.sensing import SensingLayer


def _tensor_stats(t: torch.Tensor) -> dict:
    """Finite/min/max/mean/std for any tensor (works with complex)."""
    t_real = t.real if t.is_complex() else t
    finite = bool(torch.isfinite(t_real).all().item())
    return {
        'finite': finite,
        'min': float(t_real.min().item()),
        'max': float(t_real.max().item()),
        'mean': float(t_real.mean().item()),
        'std': float(t_real.std().item()),
    }


def _tensor_stats_real(t: torch.Tensor) -> dict:
    """Stats using abs() for complex tensors (magnitude-based)."""
    if t.is_complex():
        t_mag = torch.abs(t)
    else:
        t_mag = t
    finite = bool(torch.isfinite(t_mag).all().item())
    return {
        'finite': finite,
        'min': float(t_mag.min().item()),
        'max': float(t_mag.max().item()),
        'mean': float(t_mag.mean().item()),
        'std': float(t_mag.std().item()),
        'has_nonfinite': bool((~torch.isfinite(t_mag)).any().item()),
    }


def _normalize_once(y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    y_max = torch.max(y)
    if not torch.isfinite(y_max) or y_max <= 0:
        return y
    return y / (y_max + eps)


def _radiance_to_field_amplitude(radiance: torch.Tensor) -> torch.Tensor:
    """Convert non-negative spectral radiance/intensity to field amplitude."""
    return torch.sqrt(torch.clamp(radiance, min=0.0))


_VALID_FORMATS = {"nchw", "nhwc"}
_DEPTH_LAYERING_MODES = {"hard_depth", "hard_meter", "soft_diopter"}
_IMAGE_FORMATION_MODES = {"whole_field", "psf_convolution"}
_PSF_LAYER_MASK_MODES = {"current", "baek_hard"}
_PSF_BOUNDARY_MODES = {"linear_zero", "circular"}
_PSF_OPTICS_VERSIONS = {"legacy", "consistent_grid_v1"}


def _next_fast_fft_length(target: int) -> int:
    """Return the smallest 5-smooth FFT length greater than or equal to target."""
    target = int(target)
    if target < 1:
        raise ValueError(f"FFT length must be >= 1, got {target}")
    candidate = target
    while True:
        remainder = candidate
        for factor in (2, 3, 5):
            while remainder % factor == 0:
                remainder //= factor
        if remainder == 1:
            return candidate
        candidate += 1


class SoftDiopterBinner(nn.Module):
    def __init__(
        self,
        z_min: float,
        z_max: float,
        num_layers: int,
        eps: float = 1e-8,
        bandwidth_scale: float = 1.0,
    ):
        super().__init__()
        if z_min <= 0:
            raise ValueError(f"z_min must be > 0, got {z_min}")
        if z_max <= z_min:
            raise ValueError(f"z_max must be > z_min, got z_min={z_min}, z_max={z_max}")
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if eps <= 0:
            raise ValueError(f"eps must be > 0, got {eps}")
        if bandwidth_scale <= 0:
            raise ValueError(f"bandwidth_scale must be > 0, got {bandwidth_scale}")

        self.z_min = float(z_min)
        self.z_max = float(z_max)
        self.num_layers = int(num_layers)
        self.eps = float(eps)
        self.bandwidth_scale = float(bandwidth_scale)

        u_min = 1.0 / self.z_max
        u_max = 1.0 / self.z_min
        centers_u = torch.linspace(u_min, u_max, self.num_layers, dtype=torch.float32)
        z_centers = 1.0 / centers_u
        if self.num_layers > 1:
            du = centers_u[1] - centers_u[0]
        else:
            du = torch.tensor(1.0, dtype=torch.float32)

        self.register_buffer("centers_u", centers_u)
        self.register_buffer("z_centers", z_centers)
        self.register_buffer("du", du)

    def forward(
        self,
        depth: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        return_debug: bool = False,
    ):
        if depth.ndim != 4 or depth.shape[1] != 1:
            raise ValueError(f"depth must have shape [B,1,H,W], got {tuple(depth.shape)}")

        b, _, h, w = depth.shape
        calc_dtype = torch.float32
        depth_f = depth.to(dtype=calc_dtype)
        finite_positive = torch.isfinite(depth_f) & (depth_f > 0)
        if valid_mask is None:
            valid = finite_positive.to(dtype=calc_dtype)
        else:
            if valid_mask.ndim != 4 or valid_mask.shape[1] != 1:
                raise ValueError(f"valid_mask must have shape [B,1,H,W], got {tuple(valid_mask.shape)}")
            valid = (finite_positive & (valid_mask > 0)).to(dtype=calc_dtype)

        if self.num_layers == 1:
            weights = torch.ones((b, 1, h, w), device=depth.device, dtype=calc_dtype) * valid
            debug = {"weight_sum": weights.sum(dim=1, keepdim=True)} if return_debug else None
            if return_debug:
                return weights.to(dtype=depth.dtype), self.z_centers.to(depth.device, depth.dtype), debug
            return weights.to(dtype=depth.dtype), self.z_centers.to(depth.device, depth.dtype)

        z_safe = torch.where(
            finite_positive,
            depth_f.clamp(min=self.z_min, max=self.z_max),
            torch.full_like(depth_f, self.z_min),
        )
        u = (1.0 / z_safe).clamp(min=1.0 / self.z_max, max=1.0 / self.z_min)

        centers_u = self.centers_u.to(device=depth.device, dtype=calc_dtype).view(1, self.num_layers, 1, 1)
        bandwidth = (self.du.to(device=depth.device, dtype=calc_dtype) * self.bandwidth_scale).view(1, 1, 1, 1)
        raw_w = torch.relu(1.0 - torch.abs(u - centers_u) / bandwidth)
        raw_w = raw_w * valid
        weights = raw_w / (raw_w.sum(dim=1, keepdim=True) + self.eps)
        weights = weights * valid

        z_centers = self.z_centers.to(device=depth.device, dtype=depth.dtype)
        weights = weights.to(dtype=depth.dtype)
        if return_debug:
            debug = {
                "weight_sum": weights.sum(dim=1, keepdim=True),
                "centers_u": self.centers_u.to(device=depth.device, dtype=depth.dtype),
                "depth_centers": z_centers,
            }
            return weights, z_centers, debug
        return weights, z_centers


class DoDoForwardModel(nn.Module):
    def __init__(
        self,
        input_size: Tuple[int, int, int] = (128, 128, 25),
        doe_type_a: str = "Zeros",
        train_c: bool = True,
        free: bool = False,
        n_terms: int = 150,
        input_format: str = "nchw",
        output_format: str = "nchw",
        assets_dir: str = "torch_optics/assets",
        sensing_normalize_mode: str = "global",
        use_second_doe: bool = True,
        sensor_measurement: str = "amplitude",
        skip_prop2: bool = False,
    ):
        super().__init__()
        self.skip_prop2 = bool(skip_prop2)
        self.input_size = input_size
        self.input_format = input_format.lower()
        self.output_format = output_format.lower()
        self.use_second_doe = use_second_doe

        mss = 128
        minput = 128

        self.prop1 = PropagationLayer(Mp=minput, L=0.01, zi=0.06, trainable_z=False)
        if free:
            self.doe1 = DOEFreeLayer(
                Mdoe=mss,
                Mesce=minput,
                n_terms=n_terms,
                doe_type=doe_type_a,
                trainable=train_c,
                assets_dir=assets_dir,
                phase_scale_mode="legacy_free",
            )
        else:
            self.doe1 = DOELayer(
                Mdoe=mss,
                Mesce=minput,
                doe_type=doe_type_a,
                trainable=train_c,
                assets_dir=assets_dir,
                phase_scale_mode="legacy_doe",
            )
        self.prop2 = PropagationLayer(Mp=mss, L=0.006, zi=0.05, trainable_z=False)
        self.doe2 = DOELayer(
            Mdoe=mss,
            Mesce=mss,
            doe_type="Spiral",
            trainable=False,
            assets_dir=assets_dir,
            phase_scale_mode="legacy_doe",
        )
        self.prop3 = PropagationLayer(Mp=mss, L=0.0048, zi=0.01, trainable_z=False)
        self.sensing = SensingLayer(Ms=mss, assets_dir=assets_dir, normalize=True, normalize_mode=sensing_normalize_mode,
                                     sensor_measurement=sensor_measurement)

    def _to_nchw(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_format == "nchw":
            return x
        if self.input_format == "nhwc":
            return x.permute(0, 3, 1, 2).contiguous()
        raise ValueError("input_format must be 'nchw' or 'nhwc'")

    def _from_nchw(self, y: torch.Tensor) -> torch.Tensor:
        if self.output_format == "nchw":
            return y
        if self.output_format == "nhwc":
            return y.permute(0, 2, 3, 1).contiguous()
        raise ValueError("output_format must be 'nchw' or 'nhwc'")

    def clamp_parameters_(self):
        if hasattr(self.doe1, "clamp_parameters_"):
            self.doe1.clamp_parameters_()
        if hasattr(self.doe2, "clamp_parameters_"):
            self.doe2.clamp_parameters_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._to_nchw(x)
        x = x.to(torch.float32)

        x = self.prop1(x)
        x = self.doe1(x)
        if not self.skip_prop2:
            x = self.prop2(x)
        if self.use_second_doe:
            x = self.doe2(x)
        x = self.prop3(x)
        y = self.sensing(x)
        return self._from_nchw(y)


class DepthAwareDoDoForwardModel(nn.Module):
    def __init__(
        self,
        depth_min: float = 0.4,
        depth_max: float = 2.0,
        num_depth_layers: int = 8,
        use_second_doe: bool = False,
        doe_type_a: str = "Zeros",
        train_c: bool = True,
        free: bool = False,
        n_terms: int = 150,
        zernike_basis_path: Optional[str] = None,
        input_format: str = "nhwc",
        output_format: str = "nhwc",
        assets_dir: str = "torch_optics/assets",
        measurement_norm_mode: str = "legacy_max",
        measurement_norm_scale: float = 1.0,
        sensing_mode: str = "rgb",
        measurement_channels: int = 3,
        depth_layering_mode: str = "hard_depth",
        soft_diopter_eps: float = 1e-8,
        soft_diopter_bandwidth_scale: float = 1.0,
        sensor_measurement: str = "amplitude",
        skip_prop2: bool = False,
        image_formation_mode: str = "whole_field",
        psf_layer_mask_mode: str = "baek_hard",
        psf_mask_blur_sigma: float = 1.0,
        psf_boundary_mode: str = "linear_zero",
        psf_depth_chunk_size: int = 1,
        prop1_padding_factor: int = 1,
        *,
        doe_basis_mode: str = "legacy_raw12",
        doe_basis_rank: int = 9,
        doe_basis_rank_rtol: float = 1e-4,
        doe_basis_rms_m: float = 3e-6,
        doe_coeff_norm_limit: float = 1.0,
        doe_init_coeff_norm: float = 0.2,
        psf_optics_version: str = "legacy",
    ):
        super().__init__()
        self.skip_prop2 = bool(skip_prop2)
        psf_optics_version = str(psf_optics_version).lower()
        if psf_optics_version not in _PSF_OPTICS_VERSIONS:
            raise ValueError(
                "psf_optics_version must be one of "
                f"{_PSF_OPTICS_VERSIONS}, got '{psf_optics_version}'"
            )
        self.psf_optics_version = psf_optics_version
        self.optical_grid_size = 128
        self.optical_grid_length_m = 0.01
        if self.psf_optics_version == "consistent_grid_v1":
            self.sensor_padding_factor = 2
            self.psf_kernel_size = 129
            self.psf_energy_reference = "full_field"
        else:
            self.sensor_padding_factor = 1
            self.psf_kernel_size = 128
            self.psf_energy_reference = "crop"
        if self.psf_optics_version == "consistent_grid_v1":
            if not self.skip_prop2:
                raise ValueError(
                    "consistent_grid_v1 currently requires skip_prop2=True; "
                    "the intermediate propagation chain has not been "
                    "validated for this PSF model"
                )
            if use_second_doe:
                raise ValueError(
                    "consistent_grid_v1 currently requires "
                    "use_second_doe=False"
                )
        if isinstance(prop1_padding_factor, bool) or not isinstance(
            prop1_padding_factor, int
        ):
            raise TypeError(
                "prop1_padding_factor must be an integer >= 1, "
                f"got {prop1_padding_factor!r}"
            )
        if prop1_padding_factor < 1:
            raise ValueError(
                f"prop1_padding_factor must be >= 1, got {prop1_padding_factor}"
            )
        self.prop1_padding_factor = prop1_padding_factor
        if depth_min >= depth_max:
            raise ValueError(f"depth_min ({depth_min}) must be < depth_max ({depth_max})")
        if num_depth_layers < 1:
            raise ValueError(f"num_depth_layers must be >= 1, got {num_depth_layers}")
        fmt_in = input_format.lower()
        fmt_out = output_format.lower()
        if fmt_in not in _VALID_FORMATS:
            raise ValueError(f"input_format must be one of {_VALID_FORMATS}, got '{input_format}'")
        if fmt_out not in _VALID_FORMATS:
            raise ValueError(f"output_format must be one of {_VALID_FORMATS}, got '{output_format}'")
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.num_depth_layers = num_depth_layers
        self.use_second_doe = use_second_doe
        self.input_format = fmt_in
        self.output_format = fmt_out

        image_formation_mode = image_formation_mode.lower()
        if image_formation_mode not in _IMAGE_FORMATION_MODES:
            raise ValueError(
                f"image_formation_mode must be one of {_IMAGE_FORMATION_MODES}, "
                f"got '{image_formation_mode}'")
        if (
            self.psf_optics_version == "consistent_grid_v1"
            and image_formation_mode != "psf_convolution"
        ):
            raise ValueError(
                "consistent_grid_v1 is defined only for "
                "image_formation_mode='psf_convolution'"
            )
        psf_layer_mask_mode = psf_layer_mask_mode.lower()
        if psf_layer_mask_mode not in _PSF_LAYER_MASK_MODES:
            raise ValueError(
                f"psf_layer_mask_mode must be one of {_PSF_LAYER_MASK_MODES}, "
                f"got '{psf_layer_mask_mode}'")
        psf_boundary_mode = psf_boundary_mode.lower()
        if psf_boundary_mode not in _PSF_BOUNDARY_MODES:
            raise ValueError(
                f"psf_boundary_mode must be one of {_PSF_BOUNDARY_MODES}, "
                f"got '{psf_boundary_mode}'")
        if (
            self.psf_optics_version == "consistent_grid_v1"
            and psf_boundary_mode != "linear_zero"
        ):
            raise ValueError(
                "consistent_grid_v1 requires "
                "psf_boundary_mode='linear_zero': its centered 129x129 PSF "
                "must not be truncated to the 128x128 scene FFT grid"
            )
        psf_mask_blur_sigma = float(psf_mask_blur_sigma)
        if psf_mask_blur_sigma < 0:
            raise ValueError(
                f"psf_mask_blur_sigma must be >= 0, got {psf_mask_blur_sigma}")
        if isinstance(psf_depth_chunk_size, bool) or not isinstance(
            psf_depth_chunk_size, int
        ):
            raise TypeError(
                "psf_depth_chunk_size must be an integer >= 1, "
                f"got {psf_depth_chunk_size!r}"
            )
        if psf_depth_chunk_size < 1:
            raise ValueError(
                "psf_depth_chunk_size must be >= 1, "
                f"got {psf_depth_chunk_size}")
        if image_formation_mode == "psf_convolution" and sensor_measurement.lower() != "intensity":
            raise ValueError(
                "psf_convolution forms an incoherent intensity image and therefore requires "
                "sensor_measurement='intensity'")
        if free and doe_basis_mode != "legacy_raw12":
            raise ValueError(
                "doe_basis_mode applies only to the legacy 12-term DOE; "
                "set free=False (or --dodo_zernike_mode legacy12) to use "
                f"'{doe_basis_mode}'"
            )
        self.image_formation_mode = image_formation_mode
        self.psf_layer_mask_mode = psf_layer_mask_mode
        self.psf_mask_blur_sigma = psf_mask_blur_sigma
        self.psf_boundary_mode = psf_boundary_mode
        self.psf_depth_chunk_size = psf_depth_chunk_size
        depth_layering_mode = depth_layering_mode.lower()
        if depth_layering_mode not in _DEPTH_LAYERING_MODES:
            raise ValueError(
                f"depth_layering_mode must be one of {_DEPTH_LAYERING_MODES}, got '{depth_layering_mode}'")
        self.depth_layering_mode = depth_layering_mode
        if measurement_norm_mode not in ("legacy_max", "none", "per_sample_max", "fixed_scale"):
            raise ValueError(
                f"measurement_norm_mode must be one of legacy_max/none/per_sample_max/fixed_scale, "
                f"got '{measurement_norm_mode}'")
        self.measurement_norm_mode = measurement_norm_mode
        measurement_norm_scale = float(measurement_norm_scale)
        if measurement_norm_mode == "fixed_scale" and measurement_norm_scale <= 0.0:
            raise ValueError("measurement_norm_scale must be > 0 when measurement_norm_mode='fixed_scale'")
        self.register_buffer(
            "measurement_norm_scale",
            torch.tensor(max(measurement_norm_scale, 1e-8), dtype=torch.float32),
            persistent=False,
        )

        mss = 128
        minput = 128

        # Compute bin edges and centers
        edges = torch.linspace(depth_min, depth_max, num_depth_layers + 1)
        if depth_layering_mode == "soft_diopter":
            self.diopter_binner = SoftDiopterBinner(
                z_min=depth_min,
                z_max=depth_max,
                num_layers=num_depth_layers,
                eps=soft_diopter_eps,
                bandwidth_scale=soft_diopter_bandwidth_scale,
            )
            z_centers = self.diopter_binner.z_centers.detach().clone()
        else:
            self.diopter_binner = None
            z_centers = 0.5 * (edges[:-1] + edges[1:])
        self.register_buffer("bin_edges", edges)
        self.register_buffer("z_centers", z_centers)

        # One prop1 per depth bin (fixed zi = bin center)
        self.prop1_layers = nn.ModuleList([
            PropagationLayer(
                Mp=minput,
                L=self.optical_grid_length_m,
                zi=float(z_centers[k]),
                trainable_z=False,
                padding_factor=self.prop1_padding_factor,
            )
            for k in range(num_depth_layers)
        ])

        if free:
            self.doe1 = DOEFreeLayer(
                Mdoe=mss, Mesce=minput, n_terms=n_terms,
                doe_type=doe_type_a, trainable=train_c,
                assets_dir=assets_dir, basis_path=zernike_basis_path,
                phase_scale_mode="legacy_free",
                use_pupil_mask=(
                    self.psf_optics_version == "consistent_grid_v1"
                ),
            )
        else:
            self.doe1 = DOELayer(
                Mdoe=mss, Mesce=minput, doe_type=doe_type_a,
                trainable=train_c, assets_dir=assets_dir,
                phase_scale_mode="legacy_doe",
                use_pupil_mask=(
                    self.psf_optics_version == "consistent_grid_v1"
                ),
                basis_mode=doe_basis_mode,
                basis_rank=doe_basis_rank,
                basis_rank_rtol=doe_basis_rank_rtol,
                basis_rms_m=doe_basis_rms_m,
                coeff_norm_limit=doe_coeff_norm_limit,
                init_coeff_norm=doe_init_coeff_norm,
            )

        intermediate_length = (
            self.optical_grid_length_m
            if self.psf_optics_version == "consistent_grid_v1"
            else 0.006
        )
        sensor_length = (
            self.optical_grid_length_m
            if self.psf_optics_version == "consistent_grid_v1"
            else 0.0048
        )
        self.prop2 = PropagationLayer(
            Mp=mss,
            L=intermediate_length,
            zi=0.05,
            trainable_z=False,
        )
        self.doe2 = DOELayer(
            Mdoe=mss, Mesce=mss, doe_type="Spiral", trainable=False,
            assets_dir=assets_dir, phase_scale_mode="legacy_doe",
            use_pupil_mask=(
                self.psf_optics_version == "consistent_grid_v1"
            ),
        )
        self.prop3 = PropagationLayer(
            Mp=mss,
            L=sensor_length,
            zi=0.01,
            trainable_z=False,
            padding_factor=self.sensor_padding_factor,
        )
        self.sensing_unnorm = SensingLayer(Ms=mss, assets_dir=assets_dir, normalize=False,
                                            sensing_mode=sensing_mode,
                                            measurement_channels=measurement_channels,
                                            sensor_measurement=sensor_measurement)
        if self.psf_optics_version == "consistent_grid_v1":
            self._assert_consistent_optical_sampling()
        # Frozen-optics inference/Stage-B training can reuse this bank.  It is
        # intentionally a plain attribute, not a persistent buffer, so the new
        # image-formation mode does not change checkpoint state-dict keys.
        self._cached_psf_bank = None
        self._cached_psf_key = None
        self._cached_psf_fft_bank = None
        self._cached_psf_fft_key = None
        # The center impulse and Prop1 layers are fixed. Cache their output
        # independently from the trainable DOE-dependent PSF bank.
        self._cached_prop1_field_bank = None
        self._cached_prop1_field_key = None
        self._last_psf_capture_fraction = None

    def _assert_consistent_optical_sampling(self):
        """Verify that every represented plane uses one physical sample pitch."""
        if self.psf_optics_version != "consistent_grid_v1":
            return

        expected_size = self.optical_grid_size
        expected_length = self.optical_grid_length_m
        expected_pitch = expected_length / expected_size
        propagation_layers = [
            *self.prop1_layers,
            self.prop2,
            self.prop3,
        ]
        for layer in propagation_layers:
            if layer.Mp != expected_size:
                raise ValueError(
                    "consistent_grid_v1 requires every propagation plane to "
                    f"use Mp={expected_size}, got {layer.Mp}"
                )
            pitch = layer.L / layer.Mp
            if abs(pitch - expected_pitch) > 1e-12:
                raise ValueError(
                    "consistent_grid_v1 requires equal sampling pitch across "
                    f"all planes: expected {expected_pitch:g}m, got {pitch:g}m"
                )

        for name, doe in (("doe1", self.doe1), ("doe2", self.doe2)):
            if doe.Mesce != expected_size or doe.Mdoe != expected_size:
                raise ValueError(
                    "consistent_grid_v1 requires the DOE sampling grid to "
                    f"remain {expected_size}x{expected_size}; {name} has "
                    f"Mesce={doe.Mesce}, Mdoe={doe.Mdoe}"
                )
            if not doe.use_pupil_mask:
                raise ValueError(
                    f"consistent_grid_v1 requires {name} to apply its pupil "
                    "as an amplitude-domain validity mask"
                )

        if self.prop3.padding_factor != 2:
            raise ValueError(
                "consistent_grid_v1 requires Prop3 padding_factor=2"
            )
        if self.prop3.work_Mp != 256 or abs(self.prop3.work_L - 0.02) > 1e-12:
            raise ValueError(
                "consistent_grid_v1 requires the Prop3 work grid to be "
                "256x256 over 0.02m"
            )
        if self.psf_kernel_size != 129:
            raise ValueError(
                "consistent_grid_v1 requires a centered 129x129 PSF kernel"
            )

    def _psf_config_signature(self):
        return (
            self.psf_optics_version,
            self.optical_grid_size,
            self.optical_grid_length_m,
            self.sensor_padding_factor,
            self.psf_kernel_size,
            self.prop1_padding_factor,
            tuple(
                (
                    layer.Mp,
                    layer.L,
                    layer.padding_factor,
                    layer.work_Mp,
                    layer.work_L,
                )
                for layer in (*self.prop1_layers, self.prop2, self.prop3)
            ),
            self.doe1.Mesce,
            self.doe1.Mdoe,
            self.doe1.use_pupil_mask,
            self.doe2.Mesce,
            self.doe2.Mdoe,
            self.doe2.use_pupil_mask,
        )

    @property
    def psf_capture_fraction(self) -> Optional[torch.Tensor]:
        """Energy fraction retained by the latest consistent-grid PSF crop."""
        return self._last_psf_capture_fraction

    def clamp_parameters_(self):
        if hasattr(self.doe1, "clamp_parameters_"):
            self.doe1.clamp_parameters_()
        if self.use_second_doe and hasattr(self.doe2, "clamp_parameters_"):
            self.doe2.clamp_parameters_()
        self.clear_psf_cache()

    def clear_psf_cache(self):
        self._cached_psf_bank = None
        self._cached_psf_key = None
        self._cached_psf_fft_bank = None
        self._cached_psf_fft_key = None
        self._last_psf_capture_fraction = None

    def clear_static_optics_cache(self):
        self.clear_psf_cache()
        self._cached_prop1_field_bank = None
        self._cached_prop1_field_key = None

    def _apply(self, fn):
        self.clear_static_optics_cache()
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
        self.clear_static_optics_cache()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        self.clear_static_optics_cache()

    def _to_nchw(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_format == "nhwc":
            return x.permute(0, 3, 1, 2).contiguous()
        return x

    def _from_nchw(self, y: torch.Tensor) -> torch.Tensor:
        if self.output_format == "nhwc":
            return y.permute(0, 2, 3, 1).contiguous()
        return y

    def _prepare_inputs(
        self,
        spectral: torch.Tensor,
        depth: torch.Tensor,
        valid_mask: Optional[torch.Tensor],
    ):
        if spectral.ndim != 4:
            raise ValueError(f"spectral must be 4D (B,H,W,C) or (B,C,H,W), got {spectral.ndim}D")
        if depth.ndim not in (3, 4):
            raise ValueError(f"depth must be 3D (B,H,W) or 4D (B,1,H,W), got {depth.ndim}D")

        spectral = self._to_nchw(spectral).to(torch.float32)
        if depth.ndim == 3:
            depth = depth.unsqueeze(1)
        depth = depth.to(device=spectral.device, dtype=torch.float32)

        batch_s, channels, height_s, width_s = spectral.shape
        batch_d, depth_channels, height_d, width_d = depth.shape
        if depth_channels != 1:
            raise ValueError(f"depth must have one channel, got shape {tuple(depth.shape)}")
        if batch_s != batch_d:
            raise ValueError(f"spectral batch size ({batch_s}) != depth batch size ({batch_d})")
        if height_s != height_d or width_s != width_d:
            raise ValueError(
                f"spectral spatial size ({height_s}x{width_s}) != "
                f"depth spatial size ({height_d}x{width_d})")
        expected_bands = int(self.prop3.wave_lengths.numel())
        if channels != expected_bands:
            raise ValueError(f"spectral must have {expected_bands} bands, got {channels}")

        if valid_mask is not None:
            if valid_mask.ndim == 3:
                valid_mask = valid_mask.unsqueeze(1)
            if valid_mask.ndim != 4 or valid_mask.shape[1] != 1:
                raise ValueError(
                    f"valid_mask must be 3D [B,H,W] or 4D [B,1,H,W], "
                    f"got {tuple(valid_mask.shape)}")
            if valid_mask.shape[0] != batch_s or valid_mask.shape[-2:] != (height_s, width_s):
                raise ValueError(
                    f"valid_mask shape {tuple(valid_mask.shape)} is incompatible with "
                    f"spectral/depth shape batch={batch_s}, spatial={height_s}x{width_s}")
            valid_mask = valid_mask.to(device=spectral.device, dtype=torch.float32)

        return spectral, depth, valid_mask

    def _current_depth_weights(
        self,
        depth: torch.Tensor,
        valid_mask: Optional[torch.Tensor],
        return_debug: bool,
    ):
        if self.depth_layering_mode == "soft_diopter":
            result = self.diopter_binner(
                depth,
                valid_mask=valid_mask,
                return_debug=return_debug,
            )
            if return_debug:
                weights, _, debug = result
                return weights, debug
            weights, _ = result
            return weights, None

        # Preserve the legacy hard-mode behavior exactly: valid_mask is not
        # applied here because the caller historically masked spectral input.
        depth_clamped = torch.clamp(depth, self.depth_min, self.depth_max)
        layer_weights = []
        for k in range(self.num_depth_layers):
            lo = self.bin_edges[k]
            hi = self.bin_edges[k + 1]
            if k < self.num_depth_layers - 1:
                layer_weight = ((depth_clamped >= lo) & (depth_clamped < hi)).to(torch.float32)
            else:
                layer_weight = ((depth_clamped >= lo) & (depth_clamped <= hi)).to(torch.float32)
            layer_weights.append(layer_weight)
        weights = torch.cat(layer_weights, dim=1)
        debug = {"weight_sum": weights.sum(dim=1, keepdim=True)} if return_debug else None
        return weights, debug

    def _gaussian_blur_layer_weights(self, weights: torch.Tensor) -> torch.Tensor:
        sigma = self.psf_mask_blur_sigma
        if sigma <= 0:
            return weights
        radius = max(1, int(3.0 * sigma + 0.5))
        coords = torch.arange(-radius, radius + 1, device=weights.device, dtype=weights.dtype)
        kernel_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = torch.outer(kernel_1d, kernel_1d).view(1, 1, 2 * radius + 1, 2 * radius + 1)

        batch, layers, height, width = weights.shape
        flattened = weights.reshape(batch * layers, 1, height, width)
        flattened = F.pad(flattened, (radius, radius, radius, radius), mode="replicate")
        blurred = F.conv2d(flattened, kernel_2d)
        return blurred.reshape(batch, layers, height, width)

    def _baek_depth_weights(
        self,
        depth: torch.Tensor,
        valid_mask: Optional[torch.Tensor],
        return_debug: bool,
    ):
        finite_positive = torch.isfinite(depth) & (depth > 0)
        if valid_mask is not None:
            valid = finite_positive & (valid_mask > 0)
        else:
            valid = finite_positive

        depth_safe = torch.where(
            finite_positive,
            depth.clamp(min=self.depth_min, max=self.depth_max),
            torch.full_like(depth, self.depth_min),
        )
        if self.depth_layering_mode == "soft_diopter":
            inverse_depth = 1.0 / depth_safe
            centers_u = self.diopter_binner.centers_u.to(
                device=depth.device, dtype=depth.dtype).view(1, self.num_depth_layers, 1, 1)
            layer_index = torch.argmin(torch.abs(inverse_depth - centers_u), dim=1)
        else:
            layer_index = torch.bucketize(
                depth_safe[:, 0], self.bin_edges[1:-1].to(depth.device))

        weights = F.one_hot(layer_index, num_classes=self.num_depth_layers)
        weights = weights.permute(0, 3, 1, 2).to(dtype=depth.dtype)
        weights = weights * valid.to(dtype=depth.dtype)
        weights = self._gaussian_blur_layer_weights(weights)
        weights = weights * valid.to(dtype=depth.dtype)
        weight_sum = weights.sum(dim=1, keepdim=True)
        weights = torch.where(
            valid,
            weights / weight_sum.clamp_min(1e-8),
            torch.zeros_like(weights),
        )
        debug = {"weight_sum": weights.sum(dim=1, keepdim=True)} if return_debug else None
        return weights, debug

    def _psf_depth_weights(
        self,
        depth: torch.Tensor,
        valid_mask: Optional[torch.Tensor],
        return_debug: bool,
    ):
        if self.psf_layer_mask_mode == "baek_hard":
            return self._baek_depth_weights(depth, valid_mask, return_debug)
        weights, debug = self._current_depth_weights(depth, valid_mask, return_debug)
        finite_positive = torch.isfinite(depth) & (depth > 0)
        valid = finite_positive if valid_mask is None else (finite_positive & (valid_mask > 0))
        if self.psf_mask_blur_sigma > 0:
            weights = self._gaussian_blur_layer_weights(weights)
        weights = weights * valid.to(dtype=weights.dtype)
        weights = torch.where(
            valid,
            weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8),
            torch.zeros_like(weights),
        )
        if return_debug:
            debug = {"weight_sum": weights.sum(dim=1, keepdim=True)}
        return weights, debug

    def _propagate_to_sensor(self, field: torch.Tensor, depth_index: int) -> torch.Tensor:
        field = self.prop1_layers[depth_index](field)
        return self._propagate_after_prop1(field)

    def _propagate_after_prop1(
        self,
        field: torch.Tensor,
        *,
        return_sensor_work_grid: bool = False,
    ) -> torch.Tensor:
        field = self.doe1(field)
        if not self.skip_prop2:
            field = self.prop2(field)
        if self.use_second_doe:
            field = self.doe2(field)
        if return_sensor_work_grid:
            return self.prop3.forward_work_grid(field)
        return self.prop3(field)

    def _optics_are_frozen(self) -> bool:
        return not any(parameter.requires_grad for parameter in self.parameters())

    def _optics_parameter_signature(self):
        return tuple(
            (parameter.data_ptr(), int(parameter._version))
            for parameter in self.parameters()
        )

    def _prop1_impulse_field_bank(
        self,
        height: int,
        width: int,
        device: torch.device,
    ) -> torch.Tensor:
        cache_key = (
            device.type,
            device.index,
            height,
            width,
            self._psf_config_signature(),
        )
        prop1_is_frozen = not any(
            parameter.requires_grad
            for layer in self.prop1_layers
            for parameter in layer.parameters()
        )
        if (
            prop1_is_frozen
            and self._cached_prop1_field_key == cache_key
            and self._cached_prop1_field_bank is not None
        ):
            return self._cached_prop1_field_bank

        bands = int(self.prop3.wave_lengths.numel())
        impulse = torch.zeros(
            (1, bands, height, width),
            device=device,
            dtype=torch.float32,
        )
        impulse[:, :, height // 2, width // 2] = 1.0

        if prop1_is_frozen:
            with torch.no_grad():
                field_bank = torch.cat(
                    [layer(impulse) for layer in self.prop1_layers],
                    dim=0,
                ).detach()
            self._cached_prop1_field_bank = field_bank
            self._cached_prop1_field_key = cache_key
            return self._cached_prop1_field_bank

        return torch.cat(
            [layer(impulse) for layer in self.prop1_layers],
            dim=0,
        )

    def _generate_psf_bank(
        self,
        height: int,
        width: int,
        device: torch.device,
        use_cache: bool = True,
    ) -> torch.Tensor:
        expected_size = self.prop1_layers[0].Mp
        if height != expected_size or width != expected_size:
            raise ValueError(
                f"PSF convolution expects spatial size {expected_size}x{expected_size}, "
                f"got {height}x{width}")

        cache_key = (
            device.type,
            device.index,
            height,
            width,
            self._psf_config_signature(),
            self._optics_parameter_signature(),
        )
        # Stage-A validation runs without gradients while DOE parameters remain
        # unchanged. Reuse one detached PSF bank across validation batches;
        # training forwards still rebuild a live autograd graph.
        can_cache = bool(
            use_cache
            and (self._optics_are_frozen() or not torch.is_grad_enabled())
        )
        if can_cache and self._cached_psf_key == cache_key and self._cached_psf_bank is not None:
            return self._cached_psf_bank

        # Depth is represented as the batch dimension, so DOE phase generation
        # and all downstream propagations execute once as batched operations.
        prop1_fields = self._prop1_impulse_field_bank(height, width, device)
        if self.psf_optics_version == "consistent_grid_v1":
            self._assert_consistent_optical_sampling()
            sensor_field = self._propagate_after_prop1(
                prop1_fields,
                return_sensor_work_grid=True,
            )
            full_intensity = torch.abs(sensor_field).to(torch.float32).square()
            normalized_full = full_intensity / full_intensity.sum(
                dim=(-2, -1), keepdim=True
            ).clamp_min(1e-8)

            full_height, full_width = normalized_full.shape[-2:]
            half_kernel = self.psf_kernel_size // 2
            crop_top = full_height // 2 - half_kernel
            crop_left = full_width // 2 - half_kernel
            crop_bottom = crop_top + self.psf_kernel_size
            crop_right = crop_left + self.psf_kernel_size
            if (
                crop_top < 0
                or crop_left < 0
                or crop_bottom > full_height
                or crop_right > full_width
            ):
                raise ValueError(
                    f"Cannot center-crop a {self.psf_kernel_size}x"
                    f"{self.psf_kernel_size} PSF from the "
                    f"{full_height}x{full_width} sensor work grid"
                )
            # Deliberately do not normalize this crop again. Its sum is the
            # fraction of the complete propagated energy represented by the
            # finite convolution kernel.
            psf_bank = normalized_full[
                ...,
                crop_top:crop_bottom,
                crop_left:crop_right,
            ]
            self._last_psf_capture_fraction = psf_bank.detach().sum(
                dim=(-2, -1)
            )
        else:
            sensor_field = self._propagate_after_prop1(prop1_fields)
            psf_bank = torch.abs(sensor_field).to(torch.float32).square()
            psf_bank = psf_bank / psf_bank.sum(
                dim=(-2, -1), keepdim=True
            ).clamp_min(1e-8)
            self._last_psf_capture_fraction = None

        if can_cache:
            self._cached_psf_bank = psf_bank.detach()
            self._cached_psf_key = cache_key
            return self._cached_psf_bank
        return psf_bank

    def psf_bank(
        self,
        spatial_size: Tuple[int, int] = (128, 128),
        device: Optional[torch.device] = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        if device is None:
            device = self.z_centers.device
        return self._generate_psf_bank(
            int(spatial_size[0]), int(spatial_size[1]), torch.device(device), use_cache=use_cache)

    def _psf_frequency_bank(
        self,
        psf_bank: torch.Tensor,
        fft_size: Tuple[int, int],
    ) -> torch.Tensor:
        cache_key = (
            psf_bank.data_ptr(),
            int(psf_bank._version),
            psf_bank.device.type,
            psf_bank.device.index,
            psf_bank.dtype,
            tuple(fft_size),
            self.psf_boundary_mode,
            self._psf_config_signature(),
        )
        can_cache = not psf_bank.requires_grad
        if (
            can_cache
            and self._cached_psf_fft_key == cache_key
            and self._cached_psf_fft_bank is not None
        ):
            return self._cached_psf_fft_bank

        kernels = psf_bank
        if self.psf_boundary_mode == "circular":
            kernels = torch.fft.ifftshift(kernels, dim=(-2, -1))
        frequency_bank = torch.fft.rfft2(
            kernels, s=fft_size, dim=(-2, -1))
        if can_cache:
            self._cached_psf_fft_bank = frequency_bank.detach()
            self._cached_psf_fft_key = cache_key
            return self._cached_psf_fft_bank
        return frequency_bank

    def _sensor_response_matrix(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        sensing = self.sensing_unnorm
        if sensing.sensing_mode == "rgb":
            response = torch.stack([sensing.sensor_r, sensing.sensor_g, sensing.sensor_b], dim=0)
        else:
            response = sensing.response.transpose(0, 1)
        return response.to(device=device, dtype=dtype)

    def _apply_measurement_norm(self, y_sum: torch.Tensor) -> torch.Tensor:
        if self.measurement_norm_mode == "none":
            return y_sum
        if self.measurement_norm_mode == "per_sample_max":
            y_max = torch.amax(y_sum, dim=(1, 2, 3), keepdim=True)
            return y_sum / (y_max + 1e-8)
        if self.measurement_norm_mode == "fixed_scale":
            return torch.clamp(
                y_sum / (self.measurement_norm_scale.to(y_sum.device) + 1e-8), 0.0, 1.0)
        return _normalize_once(y_sum)

    def _forward_whole_field(
        self,
        spectral: torch.Tensor,
        weights: torch.Tensor,
        binner_debug: Optional[dict],
        debug_stages: bool,
    ) -> torch.Tensor:
        y_sum = None
        stage_diag = [] if debug_stages else None
        if debug_stages:
            stage_diag.append(("image_formation_mode", {"mode": "whole_field"}))
            stage_diag.append(("depth_layering_mode", {"mode": self.depth_layering_mode}))
            if binner_debug is not None:
                stage_diag.append(("depth_weight_sum", _tensor_stats(binner_debug["weight_sum"].detach())))

        for k in range(self.num_depth_layers):
            layer_weight = weights[:, k:k + 1].to(dtype=spectral.dtype)
            # Dataset values are spectral radiance/intensity, while coherent
            # propagation operates on field amplitude. Apply the depth weight
            # in the intensity domain first, then take sqrt, so
            # |x_k|^2 = spectral * layer_weight. This is especially important
            # for soft depth weights, which would otherwise be squared by the
            # intensity sensor.
            layer_radiance = spectral * layer_weight
            x_k = _radiance_to_field_amplitude(layer_radiance)
            if debug_stages and k == 0:
                stage_diag.append(("input_masked", _tensor_stats(x_k)))

            x_k = self.prop1_layers[k](x_k)
            if debug_stages and k == 0:
                stage_diag.append(("prop1", _tensor_stats_real(x_k)))
            x_k = self.doe1(x_k)
            if debug_stages and k == 0:
                stage_diag.append(("doe1", _tensor_stats_real(x_k)))
            if not self.skip_prop2:
                x_k = self.prop2(x_k)
            if debug_stages and k == 0 and not self.skip_prop2:
                stage_diag.append(("prop2", _tensor_stats_real(x_k)))
            if self.use_second_doe:
                x_k = self.doe2(x_k)
                if debug_stages and k == 0:
                    stage_diag.append(("doe2", _tensor_stats_real(x_k)))
            x_k = self.prop3(x_k)
            if debug_stages and k == 0:
                stage_diag.append(("prop3", _tensor_stats_real(x_k)))

            y_k = self.sensing_unnorm(x_k)
            if debug_stages and k == 0:
                stage_diag.append(("sensing", _tensor_stats_real(y_k)))
            y_sum = y_k if y_sum is None else y_sum + y_k

        if debug_stages:
            stage_diag.append(("y_sum_before_norm", _tensor_stats_real(y_sum)))
        y = self._apply_measurement_norm(y_sum)
        if debug_stages:
            stage_diag.append(("y_after_norm", _tensor_stats_real(y)))
            self._last_stage_diag = stage_diag
        return y

    def _forward_psf_convolution(
        self,
        spectral: torch.Tensor,
        weights: torch.Tensor,
        binner_debug: Optional[dict],
        debug_stages: bool,
        return_psf: bool = False,
        output_size: Optional[Tuple[int, int]] = None,
    ):
        batch, _, height, width = spectral.shape
        if output_size is None:
            output_height, output_width = height, width
        else:
            output_height, output_width = map(int, output_size)
            if (
                output_height < 1
                or output_width < 1
                or output_height > height
                or output_width > width
            ):
                raise ValueError(
                    "PSF convolution output_size must be positive and no "
                    f"larger than the {height}x{width} input, got "
                    f"{output_height}x{output_width}")
        output_top = (height - output_height) // 2
        output_left = (width - output_width) // 2
        output_weights = weights[
            ...,
            output_top:output_top + output_height,
            output_left:output_left + output_width,
        ]
        # Unlike the coherent whole-field path, this path is already an
        # incoherent intensity model: spectral radiance is convolved directly
        # with normalized intensity PSFs. Do not apply the field-amplitude
        # square root here.
        # The optical input grid remains 128x128. Legacy mode also returns a
        # 128x128 kernel, whereas consistent_grid_v1 retains a centered
        # 129x129 kernel from its complete padded sensor field. Scene tiles may
        # be larger (for example, a 256x256 tile with a 64-pixel halo).
        optical_input_size = int(self.prop1_layers[0].Mp)
        psf_bank = self._generate_psf_bank(
            optical_input_size,
            optical_input_size,
            spectral.device,
            use_cache=True,
        )
        response = self._sensor_response_matrix(spectral.device, spectral.dtype)
        y_sum = torch.zeros(
            (batch, response.shape[0], output_height, output_width),
            device=spectral.device,
            dtype=spectral.dtype,
        )

        if self.psf_boundary_mode == "linear_zero":
            kernel_height, kernel_width = psf_bank.shape[-2:]
            use_overlap_save = (
                output_height < height or output_width < width)
            if use_overlap_save:
                # Only the center output tile enters the decoder. Overlap-save
                # needs output+kernel-1 samples, reducing halo64 FFTs from
                # 384x384 to 256x256 without changing that center tile.
                fft_size = (
                    _next_fast_fft_length(
                        output_height + kernel_height - 1),
                    _next_fast_fft_length(
                        output_width + kernel_width - 1),
                )
                block_top = (
                    output_top + kernel_height // 2
                    - (kernel_height - 1)
                )
                block_left = (
                    output_left + kernel_width // 2
                    - (kernel_width - 1)
                )
                source_top = max(block_top, 0)
                source_left = max(block_left, 0)
                source_bottom = min(block_top + fft_size[0], height)
                source_right = min(block_left + fft_size[1], width)
                spectral_block = spectral[
                    ...,
                    source_top:source_bottom,
                    source_left:source_right,
                ]
                pad_top = source_top - block_top
                pad_left = source_left - block_left
                pad_bottom = (
                    fft_size[0] - pad_top - spectral_block.shape[-2])
                pad_right = (
                    fft_size[1] - pad_left - spectral_block.shape[-1])
                spectral_block = F.pad(
                    spectral_block,
                    (pad_left, pad_right, pad_top, pad_bottom),
                )
                spectral_fft = torch.fft.rfft2(
                    spectral_block, dim=(-2, -1))
                convolution_crop_top = kernel_height - 1
                convolution_crop_left = kernel_width - 1
            else:
                full_height = height + kernel_height - 1
                full_width = width + kernel_width - 1
                # Any FFT grid at least as large as the full linear-convolution
                # support is mathematically equivalent. Avoid prime sizes such
                # as 383x383 by using a fast 5-smooth grid.
                fft_size = (
                    _next_fast_fft_length(full_height),
                    _next_fast_fft_length(full_width),
                )
                spectral_fft = torch.fft.rfft2(
                    spectral, s=fft_size, dim=(-2, -1))
                convolution_crop_top = kernel_height // 2
                convolution_crop_left = kernel_width // 2
        else:
            fft_size = (height, width)
            spectral_fft = torch.fft.rfft2(spectral, dim=(-2, -1))
        cached_psf_fft_bank = (
            self._psf_frequency_bank(psf_bank, fft_size)
            if not psf_bank.requires_grad
            else None
        )
        response_complex = response.to(dtype=spectral_fft.dtype)

        stage_diag = [] if debug_stages else None
        if debug_stages:
            stage_diag.append(("image_formation_mode", {"mode": "psf_convolution"}))
            stage_diag.append(("depth_layering_mode", {"mode": self.depth_layering_mode}))
            stage_diag.append(("psf_layer_mask_mode", {"mode": self.psf_layer_mask_mode}))
            stage_diag.append(("psf_boundary_mode", {"mode": self.psf_boundary_mode}))
            if binner_debug is not None:
                stage_diag.append(("depth_weight_sum", _tensor_stats(binner_debug["weight_sum"].detach())))
            stage_diag.append(("psf_bank", _tensor_stats(psf_bank.detach())))
            stage_diag.append((
                "psf_energy",
                _tensor_stats(psf_bank.detach().sum(dim=(-2, -1))),
            ))
            if self._last_psf_capture_fraction is not None:
                stage_diag.append((
                    "psf_capture_fraction",
                    _tensor_stats(self._last_psf_capture_fraction),
                ))

        chunk_size = min(self.psf_depth_chunk_size, self.num_depth_layers)
        for chunk_start in range(0, self.num_depth_layers, chunk_size):
            chunk_end = min(
                chunk_start + chunk_size, self.num_depth_layers)
            if cached_psf_fft_bank is not None:
                psf_fft_chunk = cached_psf_fft_bank[
                    chunk_start:chunk_end]
            else:
                kernels = psf_bank[chunk_start:chunk_end]
                if self.psf_boundary_mode == "circular":
                    kernels = torch.fft.ifftshift(
                        kernels, dim=(-2, -1))
                psf_fft_chunk = torch.fft.rfft2(
                    kernels, s=fft_size, dim=(-2, -1))
            mixed_fft = torch.einsum(
                "bcxy,kcxy,oc->bkoxy",
                spectral_fft,
                psf_fft_chunk,
                response_complex,
            )
            full = torch.fft.irfft2(
                mixed_fft, s=fft_size, dim=(-2, -1))
            if self.psf_boundary_mode == "linear_zero":
                blurred_chunk = full[
                    ...,
                    convolution_crop_top:
                    convolution_crop_top + output_height,
                    convolution_crop_left:
                    convolution_crop_left + output_width,
                ]
            else:
                blurred_chunk = full[
                    ...,
                    output_top:output_top + output_height,
                    output_left:output_left + output_width,
                ]

            # Preserve the historical layer summation order while amortizing
            # FFT launch overhead across a small depth chunk.
            for local_index, k in enumerate(range(chunk_start, chunk_end)):
                blurred_sensor = blurred_chunk[:, local_index]
                # Baek et al. Eq. (3): depth occupancy is applied after each
                # wavelength-dependent PSF convolution.
                layered_sensor = blurred_sensor * output_weights[
                    :, k:k + 1
                ].to(dtype=blurred_sensor.dtype)
                y_sum = y_sum + layered_sensor
                if debug_stages and k == 0:
                    stage_diag.append((
                        "blurred_sensor", _tensor_stats(blurred_sensor)))
                    stage_diag.append((
                        "layered_sensor", _tensor_stats(layered_sensor)))

        if debug_stages:
            stage_diag.append(("y_sum_before_norm", _tensor_stats_real(y_sum)))
        y = self._apply_measurement_norm(y_sum)
        if debug_stages:
            stage_diag.append(("y_after_norm", _tensor_stats_real(y)))
            self._last_stage_diag = stage_diag
        if return_psf:
            return y, psf_bank
        return y

    def forward(
        self,
        spectral: torch.Tensor,
        depth: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        debug_stages: bool = False,
        return_psf: bool = False,
        output_size: Optional[Tuple[int, int]] = None,
    ):
        spectral, depth, valid_mask = self._prepare_inputs(spectral, depth, valid_mask)
        psf_bank = None
        if self.image_formation_mode == "psf_convolution":
            weights, binner_debug = self._psf_depth_weights(
                depth, valid_mask, return_debug=debug_stages)
            y, psf_bank = self._forward_psf_convolution(
                spectral,
                weights,
                binner_debug,
                debug_stages,
                return_psf=True,
                output_size=output_size,
            )
        else:
            weights, binner_debug = self._current_depth_weights(
                depth, valid_mask, return_debug=debug_stages)
            y = self._forward_whole_field(
                spectral, weights, binner_debug, debug_stages)
        y = self._from_nchw(y)
        if return_psf:
            return y, psf_bank
        return y


def Forward_DM_Spiral_Depth(
    depth_min=0.4,
    depth_max=2.0,
    num_depth_layers=8,
    use_second_doe=False,
    DOE_typeA="Zeros",
    Train_c=True,
    assets_dir="torch_optics/assets",
    measurement_norm_mode="legacy_max",
    measurement_norm_scale=1.0,
    sensing_mode="rgb",
    measurement_channels=3,
    depth_layering_mode="hard_depth",
    soft_diopter_eps=1e-8,
    soft_diopter_bandwidth_scale=1.0,
    sensor_measurement="amplitude",
    skip_prop2=False,
    image_formation_mode="whole_field",
    psf_layer_mask_mode="baek_hard",
    psf_mask_blur_sigma=1.0,
    psf_boundary_mode="linear_zero",
    psf_depth_chunk_size=1,
    prop1_padding_factor=1,
    *,
    doe_basis_mode="legacy_raw12",
    doe_basis_rank=9,
    doe_basis_rank_rtol=1e-4,
    doe_basis_rms_m=3e-6,
    doe_coeff_norm_limit=1.0,
    doe_init_coeff_norm=0.2,
    psf_optics_version="legacy",
):
    return DepthAwareDoDoForwardModel(
        depth_min=depth_min,
        depth_max=depth_max,
        num_depth_layers=num_depth_layers,
        use_second_doe=use_second_doe,
        doe_type_a=DOE_typeA,
        train_c=Train_c,
        free=False,
        input_format="nhwc",
        output_format="nhwc",
        assets_dir=assets_dir,
        measurement_norm_mode=measurement_norm_mode,
        measurement_norm_scale=measurement_norm_scale,
        sensing_mode=sensing_mode,
        measurement_channels=measurement_channels,
        depth_layering_mode=depth_layering_mode,
        soft_diopter_eps=soft_diopter_eps,
        soft_diopter_bandwidth_scale=soft_diopter_bandwidth_scale,
        sensor_measurement=sensor_measurement,
        skip_prop2=skip_prop2,
        prop1_padding_factor=prop1_padding_factor,
        image_formation_mode=image_formation_mode,
        psf_layer_mask_mode=psf_layer_mask_mode,
        psf_mask_blur_sigma=psf_mask_blur_sigma,
        psf_boundary_mode=psf_boundary_mode,
        psf_depth_chunk_size=psf_depth_chunk_size,
        doe_basis_mode=doe_basis_mode,
        doe_basis_rank=doe_basis_rank,
        doe_basis_rank_rtol=doe_basis_rank_rtol,
        doe_basis_rms_m=doe_basis_rms_m,
        doe_coeff_norm_limit=doe_coeff_norm_limit,
        doe_init_coeff_norm=doe_init_coeff_norm,
        psf_optics_version=psf_optics_version,
    )


def Forward_DM_Spiral(
    input_size=(128, 128, 25),
    DOE_typeA="Zeros",
    name="Forward_Model",
    Train_c=True,
    assets_dir="torch_optics/assets",
    use_second_doe=True,
):
    _ = name
    return DoDoForwardModel(
        input_size=input_size,
        doe_type_a=DOE_typeA,
        train_c=Train_c,
        free=False,
        input_format="nhwc",
        output_format="nhwc",
        assets_dir=assets_dir,
        use_second_doe=use_second_doe,
    )


def Forward_DM_Spiral_Free(
    input_size=(128, 128, 25),
    Nterms=150,
    DOE_typeA="Zeros",
    name="Forward_Model",
    Train_c=True,
    assets_dir="torch_optics/assets",
    use_second_doe=True,
):
    _ = name
    return DoDoForwardModel(
        input_size=input_size,
        doe_type_a=DOE_typeA,
        train_c=Train_c,
        free=True,
        n_terms=Nterms,
        input_format="nhwc",
        output_format="nhwc",
        assets_dir=assets_dir,
        use_second_doe=use_second_doe,
    )
