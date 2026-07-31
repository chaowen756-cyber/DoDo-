"""Reusable objective and physical-budget helpers for DOE-only PSF search."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

from util.psf_regularization import (
    multiscale_psf_energy_concentration_loss,
    psf_mtf_floor_loss,
    sensor_weighted_depth_psf_separation_loss,
    sensor_weighted_spectral_psf_separation_loss,
)


@dataclass(frozen=True)
class DOEPreoptimizationWeights:
    """Relative weights for the dimensionless PSF feasibility objectives."""

    fisher: float = 1.0
    mtf: float = 20.0
    optical_spectral_separation: float = 5.0
    optical_depth_separation: float = 2.0
    sensor_spectral_separation: float = 0.0
    sensor_depth_separation: float = 0.5
    energy_guard: float = 0.10


@dataclass(frozen=True)
class DOEPreoptimizationTargets:
    """Conservative targets shared by rank-9 and higher-capacity searches."""

    fisher_ridge: float = 1e-8
    fisher_loss_scale: float = 1e-7
    fisher_spatial_crlb_weight: float = 0.10
    fisher_depth_crlb_weight: float = 1.0
    fisher_wavelength_crlb_weight: float = 1.0
    mtf_min_frequency: float = 0.02
    mtf_max_frequency: float = 0.15
    mtf_at_005: float = 0.12
    mtf_at_010: float = 0.05
    mtf_at_015: float = 0.025
    spectral_margin: float = 0.90
    depth_margin: float = 0.90
    optical_spectral_offsets: Tuple[int, ...] = (1, 2, 4)
    optical_depth_offsets: Tuple[int, ...] = (1,)
    energy_radii: Tuple[float, float] = (16.0, 24.0)
    energy_outside_budgets: Tuple[float, float] = (0.75, 0.55)


def _require_nonnegative_weights(weights: DOEPreoptimizationWeights) -> None:
    for name, value in vars(weights).items():
        if float(value) < 0.0:
            raise ValueError(f"{name} weight must be >= 0, got {value}")


def initialize_doe_height_(
    doe,
    *,
    target_rms_m: float,
    generator: torch.Generator,
    zero_piston: bool = True,
) -> None:
    """Initialize any linear Zernike DOE at a common physical pupil RMS."""
    target_rms_m = float(target_rms_m)
    if target_rms_m <= 0.0:
        raise ValueError("target_rms_m must be > 0")
    coefficients = getattr(doe, "zernike_coeffs", None)
    if (
        not isinstance(coefficients, torch.nn.Parameter)
        or not coefficients.requires_grad
    ):
        raise ValueError("DOE must expose trainable zernike_coeffs")
    with torch.no_grad():
        coefficients.normal_(generator=generator)
        if zero_piston and coefficients.numel() > 0:
            source_indices = getattr(doe, "zernike_source_indices", None)
            if isinstance(source_indices, torch.Tensor) and source_indices.numel() > 0:
                piston = torch.nonzero(source_indices == 0, as_tuple=False)
                if piston.numel() > 0:
                    coefficients[int(piston[0, 0])] = 0.0
            else:
                # Poppy/free Zernike volumes use the first term as piston.
                coefficients[0] = 0.0
        current_rms = doe.pupil_rms(doe.heightmap())
        if not torch.isfinite(current_rms) or current_rms <= 0.0:
            raise FloatingPointError("random DOE initialization has invalid height RMS")
        coefficients.mul_(target_rms_m / current_rms)


def doe_physical_stats(doe) -> Dict[str, torch.Tensor]:
    """Return detached physical diagnostics that are comparable across bases."""
    height = doe.heightmap()
    pupil = doe.spiral_p.to(device=height.device) > 0.5
    selected = height[pupil].to(torch.float32)
    coefficients = doe.zernike_coeffs
    stats = {
        "doe/height_rms_m": doe.pupil_rms(height).detach(),
        "doe/height_min_m": selected.min().detach(),
        "doe/height_max_m": selected.max().detach(),
        "doe/coeff_norm": coefficients.detach().norm(),
    }
    if hasattr(doe, "wrapped_heightmap"):
        wrapped_height = doe.wrapped_heightmap()
        wrapped_selected = wrapped_height[pupil].to(torch.float32)
        stats.update(
            {
                "doe/wrapped_height_rms_m": doe.pupil_rms(wrapped_height).detach(),
                "doe/wrapped_height_min_m": wrapped_selected.min().detach(),
                "doe/wrapped_height_max_m": wrapped_selected.max().detach(),
                "doe/unwrapped_phase_rms_rad": doe.phase_rms().detach(),
            }
        )
    return stats


def initialize_phase_conjugate_doe_(
    model,
    *,
    reference_depth_m: float,
    reference_wavelength_m: float,
    phase_noise_std_rad: float,
    generator: torch.Generator,
) -> Dict[str, float]:
    """Initialize a pixel-phase DOE by discrete adjoint phase conjugation.

    The carrier maximizes the center-pixel complex amplitude for the closest
    available depth/wavelength sample under the exact current Prop1/Prop3
    operators. No continuous focal-length or sensor-pitch approximation is
    introduced. A small in-pupil phase perturbation makes repeated seeds a
    meaningful non-convexity check without destroying the focusing carrier.
    """
    doe = model.doe1
    if not hasattr(doe, "wrapped_heightmap"):
        raise ValueError("phase-conjugate initialization requires pixel-phase DOE")
    if not model.skip_prop2 or model.use_second_doe:
        raise ValueError(
            "phase-conjugate initialization currently requires direct DOE-to-Prop3 "
            "propagation"
        )
    reference_depth_m = float(reference_depth_m)
    reference_wavelength_m = float(reference_wavelength_m)
    phase_noise_std_rad = float(phase_noise_std_rad)
    if reference_depth_m <= 0.0 or reference_wavelength_m <= 0.0:
        raise ValueError("reference depth and wavelength must be > 0")
    if phase_noise_std_rad < 0.0:
        raise ValueError("phase_noise_std_rad must be >= 0")

    coefficients = doe.zernike_coeffs
    device = coefficients.device
    depth_index = int(
        torch.argmin(torch.abs(model.z_centers.to(device=device) - reference_depth_m))
    )
    wavelengths = model.prop3.wave_lengths.to(device=device)
    wavelength_index = int(
        torch.argmin(torch.abs(wavelengths - reference_wavelength_m))
    )
    height = width = int(doe.Mdoe)
    with torch.no_grad():
        incoming_bank = model._prop1_impulse_field_bank(height, width, device)
        incoming = incoming_bank[depth_index, wavelength_index]
        baseline_sensor = model.prop3(doe(incoming_bank[depth_index : depth_index + 1]))
        baseline_center = (
            baseline_sensor[0, wavelength_index, height // 2, width // 2].abs().square()
        )
        target = torch.zeros(
            (1, int(wavelengths.numel()), height, width),
            device=device,
            dtype=torch.complex64,
        )
        target[:, wavelength_index, height // 2, width // 2] = 1.0
        backpropagated = model.prop3.adjoint(target)[0, wavelength_index]
        carrier = torch.angle(backpropagated) - torch.angle(incoming)
        pupil = doe.spiral_p.to(device=device) > 0.5
        carrier = torch.where(pupil, carrier, torch.zeros_like(carrier))
        if phase_noise_std_rad > 0.0:
            noise = torch.randn(
                carrier.shape,
                device=device,
                dtype=carrier.dtype,
                generator=generator,
            )
            carrier = carrier + phase_noise_std_rad * noise * pupil
        coefficients.copy_(carrier)
        focused_sensor = model.prop3(doe(incoming_bank[depth_index : depth_index + 1]))
        focused_center = (
            focused_sensor[0, wavelength_index, height // 2, width // 2].abs().square()
        )

    return {
        "reference_depth_index": depth_index,
        "reference_depth_m": float(model.z_centers[depth_index]),
        "reference_wavelength_index": wavelength_index,
        "reference_wavelength_m": float(wavelengths[wavelength_index]),
        "center_intensity_before": float(baseline_center),
        "center_intensity_after": float(focused_center),
        "center_intensity_gain": float(
            focused_center / baseline_center.clamp_min(1e-20)
        ),
    }


def rms_constrained_optimizer_step_(
    doe,
    optimizer: torch.optim.Optimizer,
    *,
    maximum_rms_m: float,
    boundary_fraction: float = 0.999,
) -> Dict[str, float]:
    """Take an optimizer step inside a physical pupil-RMS ball.

    Once the DOE reaches the boundary, the outward component of the proposed
    coefficient update is removed in the local RMS-normal direction.  A tiny
    radial retraction remains as a numerical safeguard for the second-order
    radius increase of a tangent step.  This avoids repeatedly shrinking a
    strongly outward Adam proposal and preserves motion along the boundary.
    """
    maximum_rms_m = float(maximum_rms_m)
    boundary_fraction = float(boundary_fraction)
    if maximum_rms_m <= 0.0:
        raise ValueError("maximum_rms_m must be > 0")
    if not 0.0 < boundary_fraction <= 1.0:
        raise ValueError("boundary_fraction must lie in (0,1]")
    coefficients = getattr(doe, "zernike_coeffs", None)
    if not isinstance(coefficients, torch.nn.Parameter):
        raise ValueError("DOE must expose trainable zernike_coeffs")

    before = coefficients.detach().clone()
    rms_before_tensor = doe.pupil_rms(doe.heightmap())
    rms_before = float(rms_before_tensor.detach())
    normal = None
    if rms_before >= boundary_fraction * maximum_rms_m:
        normal = torch.autograd.grad(
            rms_before_tensor,
            coefficients,
            retain_graph=False,
            create_graph=False,
        )[0].detach()

    optimizer.step()
    tangent_correction = False
    retracted = False
    retraction_scale = 1.0
    with torch.no_grad():
        if normal is not None:
            normal_norm_sq = normal.square().sum().clamp_min(1e-30)
            proposed_delta = coefficients - before
            outward_delta = (proposed_delta * normal).sum()
            if outward_delta > 0.0:
                proposed_delta.sub_(outward_delta / normal_norm_sq * normal)
                coefficients.copy_(before + proposed_delta)
                tangent_correction = True

        rms_after = float(doe.pupil_rms(doe.heightmap()).detach())
        if rms_after > maximum_rms_m:
            retraction_scale = maximum_rms_m / max(rms_after, 1e-30)
            coefficients.mul_(retraction_scale)
            retracted = True
            rms_after = float(doe.pupil_rms(doe.heightmap()).detach())

    return {
        "rms_before_m": rms_before,
        "rms_after_m": rms_after,
        "tangent_correction": float(tangent_correction),
        "retracted": float(retracted),
        "retraction_scale": retraction_scale,
    }


def optical_psf_shape_separation_loss(
    psf_bank: torch.Tensor,
    *,
    dimension: str,
    margin: float = 0.90,
    offsets: Sequence[int] = (1,),
    hard_fraction: float = 0.20,
    hard_weight: float = 0.50,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Separate normalized monochromatic PSF shapes without sensor response.

    Removing total intensity and RGB spectral response makes this a direct
    test of whether the DOE itself changes the spatial PSF with wavelength or
    depth.  Centroid shifts remain part of the signature because they are a
    valid coding cue in the convolutional measurement.
    """
    if psf_bank.ndim != 4:
        raise ValueError(
            "psf_bank must have shape [depth,wavelength,H,W], "
            f"got {tuple(psf_bank.shape)}"
        )
    if dimension not in ("spectral", "depth"):
        raise ValueError("dimension must be 'spectral' or 'depth'")
    if not -1.0 <= float(margin) <= 1.0:
        raise ValueError("margin must lie in [-1,1]")
    if not torch.isfinite(psf_bank).all():
        raise ValueError("PSF bank must be finite")

    comparison_dim = 1 if dimension == "spectral" else 0
    comparison_size = psf_bank.shape[comparison_dim]
    signatures = F.normalize(
        psf_bank.to(torch.float32).flatten(start_dim=2),
        p=2,
        dim=-1,
        eps=1e-12,
    )
    cosine_groups = []
    offset_means = {}
    for raw_offset in offsets:
        offset = int(raw_offset)
        if offset < 1 or offset >= comparison_size:
            raise ValueError(
                f"{dimension} offset must be in [1,{comparison_size - 1}], "
                f"got {offset}"
            )
        leading = [slice(None)] * signatures.ndim
        trailing = [slice(None)] * signatures.ndim
        leading[comparison_dim] = slice(None, -offset)
        trailing[comparison_dim] = slice(offset, None)
        cosine = (signatures[tuple(leading)] * signatures[tuple(trailing)]).sum(dim=-1)
        cosine_groups.append(cosine.flatten())
        offset_means[f"offset_{offset}_cosine_mean"] = cosine.mean().detach()

    if not cosine_groups:
        raise ValueError("at least one separation offset is required")
    all_cosine = torch.cat(cosine_groups)
    violation = F.relu(all_cosine - float(margin))
    loss = violation.mean()
    if hard_weight > 0.0:
        count = max(1, int(round(violation.numel() * float(hard_fraction))))
        count = min(count, violation.numel())
        loss = (
            loss
            + float(hard_weight)
            * torch.topk(violation, count, largest=True).values.mean()
        )

    with torch.no_grad():
        stats = {
            "cosine_mean": all_cosine.mean().detach(),
            "cosine_p90": torch.quantile(all_cosine.detach(), 0.9),
            "cosine_max": all_cosine.max().detach(),
            "active_fraction": (violation > 0).to(all_cosine.dtype).mean(),
            **offset_means,
        }
    return loss, stats


def load_preoptimized_doe_(
    doe,
    checkpoint_path: Union[str, Path],
    *,
    map_location: Union[str, torch.device] = "cpu",
) -> Dict:
    """Load only DOE coefficients from a feasibility-search checkpoint."""
    checkpoint = torch.load(Path(checkpoint_path), map_location=map_location)
    if checkpoint.get("format") != "doe_psf_preoptimization_v1":
        raise ValueError("unsupported DOE preoptimization checkpoint format")
    source = checkpoint.get("doe_coefficients")
    target = getattr(doe, "zernike_coeffs", None)
    if not isinstance(source, torch.Tensor) or not isinstance(
        target, torch.nn.Parameter
    ):
        raise ValueError("checkpoint or target DOE has no Zernike coefficients")
    if tuple(source.shape) != tuple(target.shape):
        raise ValueError(
            "DOE coefficient shape mismatch: "
            f"checkpoint {tuple(source.shape)} vs target {tuple(target.shape)}"
        )
    with torch.no_grad():
        target.copy_(source.to(device=target.device, dtype=target.dtype))
    return checkpoint


def _unit_grid_finite_difference(value: torch.Tensor, dim: int) -> torch.Tensor:
    """Differentiate on a unit-spaced discrete grid with one-sided edges."""
    dim = dim % value.ndim
    if value.shape[dim] < 2:
        raise ValueError("finite differences require at least two samples")
    result = torch.empty_like(value)
    index = [slice(None)] * value.ndim

    first = index.copy()
    first[dim] = 0
    second = index.copy()
    second[dim] = 1
    result[tuple(first)] = value[tuple(second)] - value[tuple(first)]

    last = index.copy()
    last[dim] = -1
    penultimate = index.copy()
    penultimate[dim] = -2
    result[tuple(last)] = value[tuple(last)] - value[tuple(penultimate)]

    if value.shape[dim] > 2:
        middle = index.copy()
        middle[dim] = slice(1, -1)
        following = index.copy()
        following[dim] = slice(2, None)
        preceding = index.copy()
        preceding[dim] = slice(None, -2)
        result[tuple(middle)] = 0.5 * (
            value[tuple(following)] - value[tuple(preceding)]
        )
    return result


def psf_fisher_a_optimality_loss(
    psf_bank: torch.Tensor,
    sensor_response: torch.Tensor,
    *,
    ridge: float = 1e-8,
    loss_scale: float = 1e-7,
    parameter_weights: Tuple[float, float, float, float] = (0.10, 0.10, 1.0, 1.0),
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Task-weighted A-optimal Fisher loss for monochromatic RGB PSFs.

    This is the discrete counterpart of the DOE initialization objective in
    Baek et al., ICCV 2021.  The four source coordinates are x pixel, y pixel,
    one depth-bin step, and one wavelength-bin step.  Expressing every
    derivative per sampling bin makes the A-optimal trace independent of the
    arbitrary choice of metre versus nanometre units.

    The inverse is always taken over the complete x/y/depth/wavelength Fisher
    matrix, so x/y remain nuisance variables even when their CRLB weights are
    small or zero.  The PSFs deliberately retain their captured-energy scale:
    normalizing each PSF here would incorrectly make a low-throughput design
    look informative.
    """
    if psf_bank.ndim != 4:
        raise ValueError(
            "psf_bank must have shape [depth,wavelength,H,W], "
            f"got {tuple(psf_bank.shape)}"
        )
    if psf_bank.shape[0] < 2 or psf_bank.shape[1] < 2:
        raise ValueError("Fisher loss requires at least two depths and wavelengths")
    if sensor_response.ndim != 2 or sensor_response.shape[1] != psf_bank.shape[1]:
        raise ValueError("sensor_response must have shape [sensor_channel,wavelength]")
    ridge = float(ridge)
    loss_scale = float(loss_scale)
    if ridge <= 0.0 or loss_scale <= 0.0:
        raise ValueError("Fisher ridge and loss_scale must both be > 0")
    parameter_weights = tuple(float(value) for value in parameter_weights)
    if len(parameter_weights) != 4:
        raise ValueError("Fisher parameter_weights must contain x/y/depth/wavelength")
    if any(value < 0.0 for value in parameter_weights):
        raise ValueError("Fisher parameter weights must be >= 0")
    parameter_weight_sum = sum(parameter_weights)
    if parameter_weight_sum <= 0.0:
        raise ValueError("at least one Fisher parameter weight must be > 0")
    if not torch.isfinite(psf_bank).all() or not torch.isfinite(sensor_response).all():
        raise ValueError("PSF bank and sensor response must be finite")

    psf = psf_bank.to(torch.float32)
    response = sensor_response.to(device=psf.device, dtype=psf.dtype).transpose(0, 1)
    response_power = response.square().sum(dim=-1)
    dx = _unit_grid_finite_difference(psf, -1)
    dy = _unit_grid_finite_difference(psf, -2)
    dz = _unit_grid_finite_difference(psf, 0)

    # Wavelength differentiation must include the RGB spectral response, not
    # only the optical PSF, because the camera observes their product.
    observed = psf.unsqueeze(2) * response[None, :, :, None, None]
    dlambda = _unit_grid_finite_difference(observed, 1)
    spatial_depth_derivatives = (dx, dy, dz)

    entries = [[None for _ in range(4)] for _ in range(4)]
    for row, first in enumerate(spatial_depth_derivatives):
        for column, second in enumerate(spatial_depth_derivatives):
            entries[row][column] = (first * second).sum(dim=(-2, -1)) * response_power[
                None, :
            ]
        cross = torch.einsum("dlhw,dlchw,lc->dl", first, dlambda, response)
        entries[row][3] = cross
        entries[3][row] = cross
    entries[3][3] = dlambda.square().sum(dim=(2, 3, 4))
    fisher = torch.stack([torch.stack(row, dim=-1) for row in entries], dim=-2)

    identity = torch.eye(4, device=psf.device, dtype=psf.dtype)
    regularized = fisher + ridge * identity
    inverse = torch.linalg.solve(regularized, identity.expand_as(regularized))
    crlb_diagonal = inverse.diagonal(dim1=-2, dim2=-1)
    full_a_optimality = crlb_diagonal.sum(dim=-1)
    task_a_optimality = crlb_diagonal[..., 2:].sum(dim=-1)
    weight_tensor = torch.tensor(parameter_weights, device=psf.device, dtype=psf.dtype)
    # Keep the numerical scale identical to the historical full trace when all
    # four weights equal one, and comparable when users reweight parameters.
    normalized_weights = 4.0 * weight_tensor / parameter_weight_sum
    weighted_a_optimality = (crlb_diagonal * normalized_weights).sum(dim=-1)
    loss = loss_scale * weighted_a_optimality.mean()

    with torch.no_grad():
        eigenvalues = torch.linalg.eigvalsh(fisher.detach()).clamp_min(0.0)
        minimum = eigenvalues[..., 0]
        maximum = eigenvalues[..., -1]
        condition = (maximum + ridge) / (minimum + ridge)
        stats = {
            "a_optimality_mean": full_a_optimality.mean().detach(),
            "a_optimality_p90": torch.quantile(full_a_optimality.flatten(), 0.9),
            "a_optimality_max": full_a_optimality.max().detach(),
            "task_a_optimality_mean": task_a_optimality.mean().detach(),
            "task_a_optimality_p90": torch.quantile(task_a_optimality.flatten(), 0.9),
            "weighted_a_optimality_mean": weighted_a_optimality.mean().detach(),
            "weighted_a_optimality_p90": torch.quantile(
                weighted_a_optimality.flatten(), 0.9
            ),
            "crlb_x_mean": crlb_diagonal[..., 0].mean().detach(),
            "crlb_y_mean": crlb_diagonal[..., 1].mean().detach(),
            "crlb_depth_mean": crlb_diagonal[..., 2].mean().detach(),
            "crlb_wavelength_mean": crlb_diagonal[..., 3].mean().detach(),
            "minimum_eigenvalue_mean": minimum.mean().detach(),
            "minimum_eigenvalue_p10": torch.quantile(minimum.flatten(), 0.1),
            "minimum_eigenvalue_min": minimum.min().detach(),
            "condition_mean": condition.mean().detach(),
            "condition_p90": torch.quantile(condition.flatten(), 0.9),
            "trace_mean": eigenvalues.sum(dim=-1).mean().detach(),
        }
    return loss, stats


def doe_preoptimization_objective(
    psf_bank: torch.Tensor,
    sensor_response: torch.Tensor,
    *,
    energy_reference: str,
    weights: DOEPreoptimizationWeights = DOEPreoptimizationWeights(),
    targets: DOEPreoptimizationTargets = DOEPreoptimizationTargets(),
    separation_scale: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Optimize spatial bandwidth and wavelength/depth coding without a CNN.

    ``separation_scale`` supports a short warm-up in which the MTF guard is
    active immediately and the two discrimination losses enter gradually.
    """
    _require_nonnegative_weights(weights)
    separation_scale = float(separation_scale)
    if not 0.0 <= separation_scale <= 1.0:
        raise ValueError("separation_scale must lie in [0,1]")

    fisher_loss, fisher_stats = psf_fisher_a_optimality_loss(
        psf_bank,
        sensor_response,
        ridge=targets.fisher_ridge,
        loss_scale=targets.fisher_loss_scale,
        parameter_weights=(
            targets.fisher_spatial_crlb_weight,
            targets.fisher_spatial_crlb_weight,
            targets.fisher_depth_crlb_weight,
            targets.fisher_wavelength_crlb_weight,
        ),
    )
    mtf_loss, mtf_stats = psf_mtf_floor_loss(
        psf_bank,
        min_frequency=targets.mtf_min_frequency,
        max_frequency=targets.mtf_max_frequency,
        mtf_at_005=targets.mtf_at_005,
        mtf_at_010=targets.mtf_at_010,
        mtf_at_015=targets.mtf_at_015,
        worst_fraction=0.10,
        worst_weight=0.25,
    )
    optical_spectral_loss, optical_spectral_stats = optical_psf_shape_separation_loss(
        psf_bank,
        dimension="spectral",
        margin=targets.spectral_margin,
        offsets=targets.optical_spectral_offsets,
        hard_fraction=0.20,
        hard_weight=0.50,
    )
    optical_depth_loss, optical_depth_stats = optical_psf_shape_separation_loss(
        psf_bank,
        dimension="depth",
        margin=targets.depth_margin,
        offsets=targets.optical_depth_offsets,
        hard_fraction=0.20,
        hard_weight=0.50,
    )
    (
        sensor_spectral_loss,
        sensor_spectral_stats,
    ) = sensor_weighted_spectral_psf_separation_loss(
        psf_bank,
        sensor_response,
        margin=targets.spectral_margin,
        offsets=(1,),
        hard_fraction=0.20,
        hard_weight=0.50,
    )
    sensor_depth_loss, sensor_depth_stats = sensor_weighted_depth_psf_separation_loss(
        psf_bank,
        sensor_response,
        margin=targets.depth_margin,
        hard_fraction=0.20,
        hard_weight=0.50,
        energy_reference=energy_reference,
    )
    energy_loss, energy_stats = multiscale_psf_energy_concentration_loss(
        psf_bank,
        radii=targets.energy_radii,
        outside_budgets=targets.energy_outside_budgets,
        scale_weights=(1.0, 0.5),
        softness=1.5,
        cvar_fraction=0.10,
        cvar_weight=0.10,
        penalty_power=1.0,
        energy_reference=energy_reference,
    )

    weighted_fisher = float(weights.fisher) * fisher_loss
    weighted_mtf = float(weights.mtf) * mtf_loss
    full_weighted_optical_spectral = (
        float(weights.optical_spectral_separation) * optical_spectral_loss
    )
    full_weighted_optical_depth = (
        float(weights.optical_depth_separation) * optical_depth_loss
    )
    full_weighted_sensor_spectral = (
        float(weights.sensor_spectral_separation) * sensor_spectral_loss
    )
    full_weighted_sensor_depth = (
        float(weights.sensor_depth_separation) * sensor_depth_loss
    )
    weighted_energy = float(weights.energy_guard) * energy_loss
    full_separation = (
        full_weighted_optical_spectral
        + full_weighted_optical_depth
        + full_weighted_sensor_spectral
        + full_weighted_sensor_depth
    )
    full_total = weighted_fisher + weighted_mtf + full_separation + weighted_energy
    total = full_total - full_separation + separation_scale * full_separation

    metrics: Dict[str, torch.Tensor] = {
        "loss/total": total.detach(),
        "loss/train_total": total.detach(),
        "loss/full_total": full_total.detach(),
        "loss/fisher_a_optimality": fisher_loss.detach(),
        "loss/mtf": mtf_loss.detach(),
        "loss/optical_spectral_separation": optical_spectral_loss.detach(),
        "loss/optical_depth_separation": optical_depth_loss.detach(),
        "loss/sensor_spectral_separation": sensor_spectral_loss.detach(),
        "loss/sensor_depth_separation": sensor_depth_loss.detach(),
        "loss/energy_guard": energy_loss.detach(),
        "weighted/fisher_a_optimality": weighted_fisher.detach(),
        "weighted/mtf": weighted_mtf.detach(),
        "weighted/optical_spectral_separation": (
            separation_scale * full_weighted_optical_spectral
        ).detach(),
        "weighted/optical_depth_separation": (
            separation_scale * full_weighted_optical_depth
        ).detach(),
        "weighted/sensor_spectral_separation": (
            separation_scale * full_weighted_sensor_spectral
        ).detach(),
        "weighted/sensor_depth_separation": (
            separation_scale * full_weighted_sensor_depth
        ).detach(),
        "weighted_full/optical_spectral_separation": (
            full_weighted_optical_spectral.detach()
        ),
        "weighted_full/optical_depth_separation": (
            full_weighted_optical_depth.detach()
        ),
        "weighted_full/sensor_spectral_separation": (
            full_weighted_sensor_spectral.detach()
        ),
        "weighted_full/sensor_depth_separation": (full_weighted_sensor_depth.detach()),
        "weighted/energy_guard": weighted_energy.detach(),
        "schedule/separation_scale": torch.tensor(
            separation_scale, device=psf_bank.device, dtype=psf_bank.dtype
        ),
        "mtf/005_mean": mtf_stats["mtf_005_mean"],
        "mtf/005_p10": mtf_stats["mtf_005_p10"],
        "mtf/010_mean": mtf_stats["mtf_010_mean"],
        "mtf/010_p10": mtf_stats["mtf_010_p10"],
        "mtf/020_mean": mtf_stats["mtf_020_mean"],
        "optical_spectral/cosine_mean": optical_spectral_stats["cosine_mean"],
        "optical_spectral/cosine_p90": optical_spectral_stats["cosine_p90"],
        "optical_spectral/cosine_max": optical_spectral_stats["cosine_max"],
        "optical_spectral/adjacent_cosine_mean": (
            optical_spectral_stats["offset_1_cosine_mean"]
        ),
        "optical_depth/cosine_mean": optical_depth_stats["cosine_mean"],
        "optical_depth/cosine_p90": optical_depth_stats["cosine_p90"],
        "optical_depth/cosine_max": optical_depth_stats["cosine_max"],
        "optical_depth/adjacent_cosine_mean": (
            optical_depth_stats["offset_1_cosine_mean"]
        ),
        "spectral/adjacent_cosine_mean": sensor_spectral_stats["adjacent_cosine_mean"],
        "spectral/adjacent_cosine_p90": sensor_spectral_stats["adjacent_cosine_p90"],
        "spectral/adjacent_cosine_max": sensor_spectral_stats["adjacent_cosine_max"],
        "depth/adjacent_cosine_mean": sensor_depth_stats["adjacent_cosine_mean"],
        "depth/adjacent_cosine_p90": sensor_depth_stats["adjacent_cosine_p90"],
        "depth/adjacent_cosine_max": sensor_depth_stats["adjacent_cosine_max"],
    }
    for name, value in optical_spectral_stats.items():
        if name.startswith("offset_"):
            metrics[f"optical_spectral/{name}"] = value
    for name, value in optical_depth_stats.items():
        if name.startswith("offset_"):
            metrics[f"optical_depth/{name}"] = value
    for name, value in fisher_stats.items():
        metrics[f"fisher/{name}"] = value
    for name in (
        "captured_mean",
        "captured_min",
        "missing_mean",
        "r50_mean",
        "r80_mean",
        "r90_mean",
        "r90_p90",
        "r90_max",
    ):
        if name in energy_stats:
            metrics[f"energy/{name}"] = energy_stats[name]
    return total, metrics
