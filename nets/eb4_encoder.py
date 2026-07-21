"""EB4 physics-guided dual-task Mamba encoder blocks.

The module is deliberately self-contained: it is only used at encoder levels
E2--E4 and does not change the E1 CNN, bottleneck, or either CNN decoder.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None


def build_sensor_projectors(sensor_response: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the row-space and null-space projectors of a [3, L] response.

    The calculation is performed in float64 for numerical stability and the
    returned tensors use the input floating-point dtype.  For the null-space
    projector ``N``, ``sensor_response @ N`` is zero up to floating-point error.
    """

    if sensor_response.ndim != 2:
        raise ValueError(
            f"sensor_response must be two-dimensional, got {tuple(sensor_response.shape)}"
        )
    if sensor_response.shape[0] != 3:
        raise ValueError(
            f"EB4 expects a three-channel RGB response, got {tuple(sensor_response.shape)}"
        )
    if not torch.is_floating_point(sensor_response):
        sensor_response = sensor_response.float()

    response64 = sensor_response.detach().to(dtype=torch.float64)
    row_projector64 = torch.linalg.pinv(response64) @ response64
    row_projector64 = 0.5 * (row_projector64 + row_projector64.transpose(0, 1))
    null_projector64 = torch.eye(
        response64.shape[1], device=response64.device, dtype=response64.dtype
    ) - row_projector64
    output_dtype = sensor_response.dtype
    return row_projector64.to(output_dtype), null_projector64.to(output_dtype)


def _group_norm(channels: int) -> nn.GroupNorm:
    groups = min(8, channels)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class LocalDetailBranch(nn.Module):
    """Depthwise local enhancement branch for edges and fine structures."""

    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            _group_norm(channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class FourDirectionSpatialMamba(nn.Module):
    """Shared-weight horizontal/vertical bidirectional spatial Mamba."""

    def __init__(self, channels: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        if Mamba is None:
            raise ImportError("EB4 requires mamba-ssm. Install it with: pip install mamba-ssm")

        self.scan_channels = max(16, channels // 4)
        self.input_projection = nn.Conv2d(channels, self.scan_channels, kernel_size=1)
        self.norm = nn.LayerNorm(self.scan_channels)
        self.mamba = Mamba(
            d_model=self.scan_channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.output_projection = nn.Conv2d(self.scan_channels * 4, channels, kernel_size=1)

    def _scan(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.mamba(self.norm(sequence.contiguous()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature = self.input_projection(x)
        batch, channels, height, width = feature.shape

        row_sequence = feature.flatten(2).transpose(1, 2)
        row_forward = self._scan(row_sequence)
        row_backward = torch.flip(
            self._scan(torch.flip(row_sequence, dims=(1,))), dims=(1,)
        )
        row_forward = row_forward.transpose(1, 2).reshape(batch, channels, height, width)
        row_backward = row_backward.transpose(1, 2).reshape(batch, channels, height, width)

        column_feature = feature.transpose(2, 3).contiguous()
        column_sequence = column_feature.flatten(2).transpose(1, 2)
        column_forward = self._scan(column_sequence)
        column_backward = torch.flip(
            self._scan(torch.flip(column_sequence, dims=(1,))), dims=(1,)
        )
        column_forward = (
            column_forward.transpose(1, 2)
            .reshape(batch, channels, width, height)
            .transpose(2, 3)
            .contiguous()
        )
        column_backward = (
            column_backward.transpose(1, 2)
            .reshape(batch, channels, width, height)
            .transpose(2, 3)
            .contiguous()
        )

        directions = torch.cat(
            [row_forward, row_backward, column_forward, column_backward], dim=1
        )
        return self.output_projection(directions)


class PhysicsSpectralMamba(nn.Module):
    """Wavelength-token Mamba with an RGB-response null-space correction."""

    def __init__(
        self,
        channels: int,
        sensor_response: torch.Tensor,
        wavelengths: torch.Tensor,
        spectral_dim: int,
        d_state: int = 8,
        d_conv: int = 3,
        expand: int = 2,
    ):
        super().__init__()
        if Mamba is None:
            raise ImportError("EB4 requires mamba-ssm. Install it with: pip install mamba-ssm")

        if wavelengths.ndim != 1:
            raise ValueError(f"wavelengths must be one-dimensional, got {wavelengths.shape}")
        if sensor_response.shape[1] != wavelengths.numel():
            raise ValueError(
                "sensor_response and wavelengths disagree on the spectral channel count: "
                f"{sensor_response.shape[1]} versus {wavelengths.numel()}"
            )

        row_projector, null_projector = build_sensor_projectors(sensor_response)
        self.spectral_channels = int(wavelengths.numel())
        self.spectral_dim = int(spectral_dim)
        self.register_buffer("sensor_response", sensor_response.detach().float(), persistent=False)
        self.register_buffer("row_projector", row_projector.detach().float(), persistent=False)
        self.register_buffer("null_projector", null_projector.detach().float(), persistent=False)
        wavelengths = wavelengths.detach().float()
        wavelength_span = wavelengths.max() - wavelengths.min()
        if float(wavelength_span) <= 0.0:
            raise ValueError("EB4 wavelengths must span more than one value")
        normalized_wavelengths = 2.0 * (
            wavelengths - wavelengths.min()
        ) / wavelength_span - 1.0
        self.register_buffer("wavelengths", normalized_wavelengths, persistent=False)

        self.latent_correction = nn.Conv2d(channels, self.spectral_channels, kernel_size=1)
        self.amplitude_embedding = nn.Linear(1, self.spectral_dim)
        self.wavelength_embedding = nn.Sequential(
            nn.Linear(1, self.spectral_dim),
            nn.SiLU(),
            nn.Linear(self.spectral_dim, self.spectral_dim),
        )
        self.response_embedding = nn.Linear(3, self.spectral_dim)
        self.norm = nn.LayerNorm(self.spectral_dim)
        self.mamba = Mamba(
            d_model=self.spectral_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.band_readout = nn.Linear(self.spectral_dim, 1)
        self.output_projection = nn.Conv2d(self.spectral_channels, channels, kernel_size=1)

    def forward(
        self, feature: torch.Tensor, spectral_prior: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if spectral_prior.ndim != 4 or spectral_prior.shape[1] != self.spectral_channels:
            raise ValueError(
                "EB4 spectral prior must have shape [B, L, H, W] with "
                f"L={self.spectral_channels}, got {tuple(spectral_prior.shape)}"
            )
        if spectral_prior.shape[-2:] != feature.shape[-2:]:
            raise ValueError(
                f"EB4 feature/prior spatial mismatch: {feature.shape[-2:]} versus "
                f"{spectral_prior.shape[-2:]}"
            )

        correction_proposal = self.latent_correction(feature)
        null_projector = self.null_projector.to(
            device=correction_proposal.device, dtype=correction_proposal.dtype
        )
        row_projector = self.row_projector.to(
            device=correction_proposal.device, dtype=correction_proposal.dtype
        )
        null_correction = torch.einsum(
            "ij,bjhw->bihw", null_projector, correction_proposal
        )
        physics_residual = torch.einsum(
            "ij,bjhw->bihw", row_projector, correction_proposal
        ).abs().mean(dim=1, keepdim=True)
        measurement_preserving_proxy = spectral_prior + null_correction

        batch, bands, height, width = measurement_preserving_proxy.shape
        amplitude_tokens = (
            measurement_preserving_proxy.permute(0, 2, 3, 1)
            .reshape(batch * height * width, bands, 1)
        )
        dtype = amplitude_tokens.dtype
        device = amplitude_tokens.device
        wavelengths = self.wavelengths.to(device=device, dtype=dtype).view(1, bands, 1)
        response = (
            self.sensor_response.to(device=device, dtype=dtype)
            .transpose(0, 1)
            .unsqueeze(0)
        )
        tokens = (
            self.amplitude_embedding(amplitude_tokens)
            + self.wavelength_embedding(wavelengths)
            + self.response_embedding(response)
        )
        normalized_tokens = self.norm(tokens)
        forward_tokens = self.mamba(normalized_tokens)
        backward_tokens = torch.flip(
            self.mamba(torch.flip(normalized_tokens, dims=(1,)).contiguous()), dims=(1,)
        )
        spectral_tokens = 0.5 * (forward_tokens + backward_tokens)
        spectral_update = self.band_readout(spectral_tokens).squeeze(-1)
        spectral_update = spectral_update.reshape(batch, height, width, bands)
        spectral_update = spectral_update.permute(0, 3, 1, 2).contiguous()
        # Re-project the sequence-model update as well.  Consequently every
        # learned spectral correction, before the C-channel feature lift, lies
        # in ker(Omega) and cannot change the RGB response of the ridge prior.
        spectral_update = torch.einsum(
            "ij,bjhw->bihw", null_projector, spectral_update
        )
        reconstructed_bands = measurement_preserving_proxy + spectral_update
        return self.output_projection(reconstructed_bands), physics_residual

    def projector_error(self) -> torch.Tensor:
        response = self.sensor_response
        return torch.linalg.matrix_norm(response @ self.null_projector)


class BranchGate(nn.Module):
    """Predict spatial weights for local, spatial, and spectral branches."""

    def __init__(self, channels: int):
        super().__init__()
        hidden_channels = max(16, channels // 4)
        self.body = nn.Sequential(
            nn.Conv2d(channels + 4, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 3, kernel_size=1),
        )

    @property
    def output_layer(self) -> nn.Conv2d:
        return self.body[-1]

    def forward(
        self, feature: torch.Tensor, rgb_measurement: torch.Tensor, physics_residual: torch.Tensor
    ) -> torch.Tensor:
        gate_input = torch.cat([feature, rgb_measurement, physics_residual], dim=1)
        return torch.softmax(self.body(gate_input), dim=1)


class EB4EncoderBlock(nn.Module):
    """Complete EB4 block used only at E2--E4.

    It returns a common feature for the next encoder level and two task-specific
    skip features for the unchanged depth and hyperspectral decoders.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        sensor_response: torch.Tensor,
        wavelengths: torch.Tensor,
        spectral_dim: int,
        layer_scale_init: float = 1e-3,
    ):
        super().__init__()
        self.channel_projection = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.local_branch = LocalDetailBranch(out_channels)
        self.spatial_branch = FourDirectionSpatialMamba(out_channels)
        self.spectral_branch = PhysicsSpectralMamba(
            channels=out_channels,
            sensor_response=sensor_response,
            wavelengths=wavelengths,
            spectral_dim=spectral_dim,
        )
        self.common_gate = BranchGate(out_channels)
        self.depth_gate = BranchGate(out_channels)
        self.hs_gate = BranchGate(out_channels)
        self.common_scale = nn.Parameter(
            torch.full((1, out_channels, 1, 1), float(layer_scale_init))
        )
        self.depth_scale = nn.Parameter(
            torch.full((1, out_channels, 1, 1), float(layer_scale_init))
        )
        self.hs_scale = nn.Parameter(
            torch.full((1, out_channels, 1, 1), float(layer_scale_init))
        )
        self.downsample = nn.MaxPool2d(2)
        self._last_gate_means: Dict[str, torch.Tensor] = {}

    @staticmethod
    def _fuse(
        weights: torch.Tensor,
        local_feature: torch.Tensor,
        spatial_feature: torch.Tensor,
        spectral_feature: torch.Tensor,
    ) -> torch.Tensor:
        return (
            weights[:, 0:1] * local_feature
            + weights[:, 1:2] * spatial_feature
            + weights[:, 2:3] * spectral_feature
        )

    def reset_stable_initialization(self, layer_scale_init: float = 1e-3) -> None:
        """Start all three gates equally while retaining trainable residual paths."""

        for gate in (self.common_gate, self.depth_gate, self.hs_gate):
            nn.init.zeros_(gate.output_layer.weight)
            nn.init.zeros_(gate.output_layer.bias)
        with torch.no_grad():
            self.common_scale.fill_(layer_scale_init)
            self.depth_scale.fill_(layer_scale_init)
            self.hs_scale.fill_(layer_scale_init)

    def forward(
        self,
        x: torch.Tensor,
        rgb_measurement: torch.Tensor,
        spectral_prior: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        feature = self.channel_projection(x)
        spatial_size = feature.shape[-2:]
        if rgb_measurement.shape[-2:] != spatial_size:
            rgb_measurement = F.interpolate(rgb_measurement, size=spatial_size, mode="area")
        if spectral_prior.shape[-2:] != spatial_size:
            spectral_prior = F.interpolate(spectral_prior, size=spatial_size, mode="area")
        rgb_measurement = rgb_measurement.to(device=feature.device, dtype=feature.dtype)
        spectral_prior = spectral_prior.to(device=feature.device, dtype=feature.dtype)

        local_feature = self.local_branch(feature)
        spatial_feature = self.spatial_branch(feature)
        spectral_feature, physics_residual = self.spectral_branch(feature, spectral_prior)

        common_weights = self.common_gate(feature, rgb_measurement, physics_residual)
        depth_weights = self.depth_gate(feature, rgb_measurement, physics_residual)
        hs_weights = self.hs_gate(feature, rgb_measurement, physics_residual)
        common_feature = feature + self.common_scale * self._fuse(
            common_weights, local_feature, spatial_feature, spectral_feature
        )
        depth_skip = feature + self.depth_scale * self._fuse(
            depth_weights, local_feature, spatial_feature, spectral_feature
        )
        hs_skip = feature + self.hs_scale * self._fuse(
            hs_weights, local_feature, spatial_feature, spectral_feature
        )

        self._last_gate_means = {
            "common": common_weights.detach().mean(dim=(0, 2, 3)),
            "depth": depth_weights.detach().mean(dim=(0, 2, 3)),
            "hs": hs_weights.detach().mean(dim=(0, 2, 3)),
        }
        return depth_skip, hs_skip, self.downsample(common_feature), physics_residual

    def diagnostics(self) -> Dict[str, torch.Tensor]:
        diagnostics = {
            "sensor_null_error": self.spectral_branch.projector_error().detach(),
        }
        diagnostics.update(self._last_gate_means)
        return diagnostics
