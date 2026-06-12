from typing import Optional, Tuple

import torch
import torch.nn as nn

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


_VALID_FORMATS = {"nchw", "nhwc"}
_DEPTH_LAYERING_MODES = {"hard_depth", "hard_meter", "soft_diopter"}


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
        skip_prop2: bool = True,
    ):
        super().__init__()
        self.skip_prop2 = skip_prop2
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
        skip_prop2: bool = True,
    ):
        super().__init__()
        self.skip_prop2 = skip_prop2
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
            PropagationLayer(Mp=minput, L=0.01, zi=float(z_centers[k]), trainable_z=False)
            for k in range(num_depth_layers)
        ])

        if free:
            self.doe1 = DOEFreeLayer(
                Mdoe=mss, Mesce=minput, n_terms=n_terms,
                doe_type=doe_type_a, trainable=train_c,
                assets_dir=assets_dir, phase_scale_mode="legacy_free",
            )
        else:
            self.doe1 = DOELayer(
                Mdoe=mss, Mesce=minput, doe_type=doe_type_a,
                trainable=train_c, assets_dir=assets_dir,
                phase_scale_mode="legacy_doe",
            )

        self.prop2 = PropagationLayer(Mp=mss, L=0.006, zi=0.05, trainable_z=False)
        self.doe2 = DOELayer(
            Mdoe=mss, Mesce=mss, doe_type="Spiral", trainable=False,
            assets_dir=assets_dir, phase_scale_mode="legacy_doe",
        )
        self.prop3 = PropagationLayer(Mp=mss, L=0.0048, zi=0.01, trainable_z=False)
        self.sensing_unnorm = SensingLayer(Ms=mss, assets_dir=assets_dir, normalize=False,
                                            sensing_mode=sensing_mode,
                                            measurement_channels=measurement_channels,
                                            sensor_measurement=sensor_measurement)

    def clamp_parameters_(self):
        if hasattr(self.doe1, "clamp_parameters_"):
            self.doe1.clamp_parameters_()
        if self.use_second_doe and hasattr(self.doe2, "clamp_parameters_"):
            self.doe2.clamp_parameters_()

    def _to_nchw(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_format == "nhwc":
            return x.permute(0, 3, 1, 2).contiguous()
        return x

    def _from_nchw(self, y: torch.Tensor) -> torch.Tensor:
        if self.output_format == "nhwc":
            return y.permute(0, 2, 3, 1).contiguous()
        return y

    def forward(
        self,
        spectral: torch.Tensor,
        depth: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        debug_stages: bool = False,
    ) -> torch.Tensor:
        # Input validation
        if spectral.ndim != 4:
            raise ValueError(f"spectral must be 4D (B,H,W,C) or (B,C,H,W), got {spectral.ndim}D")
        if depth.ndim not in (3, 4):
            raise ValueError(f"depth must be 3D (B,H,W) or 4D (B,1,H,W), got {depth.ndim}D")

        spectral = self._to_nchw(spectral).to(torch.float32)  # (B, C, H, W)

        # Normalize depth shape to (B, 1, H, W)
        if depth.ndim == 3:
            depth = depth.unsqueeze(1)
        depth = depth.to(torch.float32)

        B_s, _, H_s, W_s = spectral.shape
        B_d, _, H_d, W_d = depth.shape
        if B_s != B_d:
            raise ValueError(f"spectral batch size ({B_s}) != depth batch size ({B_d})")
        if H_s != H_d or W_s != W_d:
            raise ValueError(f"spectral spatial size ({H_s}x{W_s}) != depth spatial size ({H_d}x{W_d})")

        if valid_mask is not None:
            if valid_mask.ndim == 3:
                valid_mask = valid_mask.unsqueeze(1)
            if valid_mask.ndim != 4 or valid_mask.shape[1] != 1:
                raise ValueError(f"valid_mask must be 3D [B,H,W] or 4D [B,1,H,W], got {tuple(valid_mask.shape)}")
            if valid_mask.shape[0] != B_s or valid_mask.shape[-2:] != (H_s, W_s):
                raise ValueError(
                    f"valid_mask shape {tuple(valid_mask.shape)} is incompatible with "
                    f"spectral/depth shape batch={B_s}, spatial={H_s}x{W_s}")
            valid_mask = valid_mask.to(device=depth.device)

        if self.depth_layering_mode == "soft_diopter":
            result = self.diopter_binner(
                depth,
                valid_mask=valid_mask,
                return_debug=debug_stages,
            )
            if debug_stages:
                weights, z_centers, binner_debug = result
            else:
                weights, z_centers = result
                binner_debug = None
            _ = z_centers
        else:
            # Clamp out-of-range depths to nearest meter-space bin for legacy hard modes.
            depth = torch.clamp(depth, self.depth_min, self.depth_max)
            edges = self.bin_edges  # (K+1,)
            weights = None
            binner_debug = None

        y_sum = None
        stage_diag = [] if debug_stages else None

        if debug_stages:
            stage_diag.append(("depth_layering_mode", {"mode": self.depth_layering_mode}))
            if binner_debug is not None:
                stage_diag.append(("depth_weight_sum", _tensor_stats(binner_debug["weight_sum"].detach())))

        for k in range(self.num_depth_layers):
            if self.depth_layering_mode == "soft_diopter":
                layer_weight = weights[:, k:k + 1, :, :].to(dtype=spectral.dtype)
            else:
                lo = edges[k]
                hi = edges[k + 1]
                if k < self.num_depth_layers - 1:
                    layer_weight = ((depth >= lo) & (depth < hi)).to(torch.float32)
                else:
                    layer_weight = ((depth >= lo) & (depth <= hi)).to(torch.float32)

            x_k = spectral * layer_weight  # broadcast (B,C,H,W) * (B,1,H,W)

            if debug_stages and k == 0:
                stage_diag.append(('input_masked', _tensor_stats(x_k)))

            x_k = self.prop1_layers[k](x_k)
            if debug_stages and k == 0:
                stage_diag.append(('prop1', _tensor_stats_real(x_k)))

            x_k = self.doe1(x_k)
            if debug_stages and k == 0:
                stage_diag.append(('doe1', _tensor_stats_real(x_k)))

            x_k = self.prop2(x_k)
            if debug_stages and k == 0:
                stage_diag.append(('prop2', _tensor_stats_real(x_k)))

            if self.use_second_doe:
                x_k = self.doe2(x_k)
                if debug_stages and k == 0:
                    stage_diag.append(('doe2', _tensor_stats_real(x_k)))

            x_k = self.prop3(x_k)
            if debug_stages and k == 0:
                stage_diag.append(('prop3', _tensor_stats_real(x_k)))

            y_k = self.sensing_unnorm(x_k)  # unnormalized (B, 3, H, W)
            if debug_stages and k == 0:
                stage_diag.append(('sensing', _tensor_stats_real(y_k)))

            y_sum = y_k if y_sum is None else y_sum + y_k

        if debug_stages:
            stage_diag.append(('y_sum_before_norm', _tensor_stats_real(y_sum)))

        if self.measurement_norm_mode == "none":
            y = y_sum
        elif self.measurement_norm_mode == "per_sample_max":
            b = y_sum.shape[0]
            y_flat = y_sum.view(b, -1)
            y_max = y_flat.max(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
            y = y_sum / (y_max + 1e-8)
        elif self.measurement_norm_mode == "fixed_scale":
            y = torch.clamp(y_sum / (self.measurement_norm_scale.to(y_sum.device) + 1e-8), 0.0, 1.0)
        else:
            y = _normalize_once(y_sum)

        if debug_stages:
            stage_diag.append(('y_after_norm', _tensor_stats_real(y)))
            self._last_stage_diag = stage_diag

        return self._from_nchw(y)


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
