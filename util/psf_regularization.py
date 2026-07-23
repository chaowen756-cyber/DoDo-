"""Differentiable regularizers and physical diagnostics for intensity PSFs."""

from typing import Dict, Sequence, Tuple

import torch
import torch.nn.functional as F


def epoch_warmup_weight(
    target_weight: float,
    current_epoch: int,
    warmup_epochs: int,
) -> float:
    """Return a linear epoch warm-up weight with epoch zero at zero."""
    target_weight = float(target_weight)
    current_epoch = int(current_epoch)
    warmup_epochs = int(warmup_epochs)
    if target_weight < 0:
        raise ValueError(f"target_weight must be >= 0, got {target_weight}")
    if current_epoch < 0:
        raise ValueError(f"current_epoch must be >= 0, got {current_epoch}")
    if warmup_epochs < 0:
        raise ValueError(f"warmup_epochs must be >= 0, got {warmup_epochs}")
    if warmup_epochs == 0:
        return target_weight
    scale = min(float(current_epoch) / float(warmup_epochs), 1.0)
    return target_weight * scale


def epoch_tightening_value(
    initial_value: float,
    target_value: float,
    current_epoch: int,
    tightening_epochs: int,
) -> float:
    """Linearly tighten a constraint from an initial to a target value."""
    current_epoch = int(current_epoch)
    tightening_epochs = int(tightening_epochs)
    if current_epoch < 0 or tightening_epochs < 0:
        raise ValueError("current_epoch and tightening_epochs must be >= 0")
    if tightening_epochs == 0:
        return float(target_value)
    fraction = min(float(current_epoch) / float(tightening_epochs), 1.0)
    return float(initial_value) + fraction * (
        float(target_value) - float(initial_value))


def _validate_and_normalize_psf(psf_bank: torch.Tensor) -> torch.Tensor:
    if psf_bank.ndim != 4:
        raise ValueError(
            "psf_bank must have shape [depth, wavelength, H, W], "
            f"got {tuple(psf_bank.shape)}")
    if not torch.isfinite(psf_bank).all():
        raise ValueError("psf_bank contains NaN or Inf")
    psf = psf_bank.to(dtype=torch.float32)
    if torch.any(psf < 0):
        raise ValueError("psf_bank must contain non-negative intensity values")
    eps = torch.finfo(psf.dtype).eps
    return psf / psf.sum(dim=(-2, -1), keepdim=True).clamp_min(eps)


def _radial_distance(psf: torch.Tensor) -> torch.Tensor:
    height, width = psf.shape[-2:]
    yy = torch.arange(height, device=psf.device, dtype=psf.dtype) - height // 2
    xx = torch.arange(width, device=psf.device, dtype=psf.dtype) - width // 2
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
    return torch.sqrt(grid_x.square() + grid_y.square())


def _outside_energy(
    psf: torch.Tensor,
    radial_distance: torch.Tensor,
    radius: float,
    softness: float,
) -> torch.Tensor:
    if softness == 0:
        outside_mask = (radial_distance > radius).to(dtype=psf.dtype)
    else:
        outside_mask = torch.sigmoid(
            (radial_distance - float(radius)) / float(softness))
    return (psf * outside_mask).sum(dim=(-2, -1))


def _top_fraction_mean(values: torch.Tensor, fraction: float) -> torch.Tensor:
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"top fraction must be in (0,1], got {fraction}")
    flat = values.flatten()
    count = max(1, int(round(flat.numel() * float(fraction))))
    return torch.topk(flat, k=count, largest=True, sorted=False).values.mean()


@torch.no_grad()
def _encircled_energy_radius_stats(
    psf: torch.Tensor,
    radial_distance: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    order = torch.argsort(radial_distance.flatten())
    sorted_radius = radial_distance.flatten()[order]
    cumulative = psf.flatten(start_dim=-2)[..., order].cumsum(dim=-1)
    stats: Dict[str, torch.Tensor] = {}
    for fraction, label in ((0.50, "r50"), (0.80, "r80"), (0.90, "r90")):
        index = (cumulative < fraction).sum(dim=-1).clamp_max(
            sorted_radius.numel() - 1)
        radius = sorted_radius[index]
        stats[f"{label}_mean"] = radius.mean().detach()
        stats[f"{label}_p90"] = torch.quantile(
            radius.detach().flatten(), 0.9)
        stats[f"{label}_max"] = radius.max().detach()
    return stats


def psf_energy_concentration_loss(
    psf_bank: torch.Tensor,
    radius: float = 16.0,
    outside_budget: float = 0.5,
    softness: float = 1.5,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Penalize normalized PSF energy exceeding a radial outside budget.

    Args:
        psf_bank: Non-negative intensity PSFs with shape [depth, wavelength, H, W].
        radius: Target radius in sensor pixels.
        outside_budget: Allowed fraction of energy outside ``radius``.
        softness: Logistic transition width in pixels. A value of zero selects
            a hard radial mask.

    Returns:
        The mean squared hinge penalty and detached diagnostic statistics.
    """
    if radius <= 0:
        raise ValueError(f"radius must be > 0, got {radius}")
    if not 0.0 <= outside_budget <= 1.0:
        raise ValueError(
            f"outside_budget must be in [0, 1], got {outside_budget}"
        )
    if softness < 0:
        raise ValueError(f"softness must be >= 0, got {softness}")
    psf = _validate_and_normalize_psf(psf_bank)
    radial_distance = _radial_distance(psf)
    outside_energy = _outside_energy(
        psf, radial_distance, radius, softness)
    violation = F.relu(outside_energy - outside_budget)
    loss = violation.square().mean()

    with torch.no_grad():
        stats = {
            "outside_mean": outside_energy.mean().detach(),
            "outside_p90": torch.quantile(outside_energy.detach().flatten(), 0.9),
            "outside_max": outside_energy.max().detach(),
            "inside_mean": (1.0 - outside_energy.mean()).detach(),
            "active_fraction": (violation > 0).to(psf.dtype).mean().detach(),
        }
    return loss, stats


def multiscale_psf_energy_concentration_loss(
    psf_bank: torch.Tensor,
    radii: Sequence[float] = (16.0, 24.0),
    outside_budgets: Sequence[float] = (0.20, 0.05),
    scale_weights: Sequence[float] = (1.0, 0.5),
    softness: float = 1.5,
    cvar_fraction: float = 0.10,
    cvar_weight: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Constrain the PSF core and tail, including the worst PSF subset."""
    if not (len(radii) == len(outside_budgets) == len(scale_weights)):
        raise ValueError("radii, outside_budgets and scale_weights must match")
    if len(radii) == 0:
        raise ValueError("at least one energy radius is required")
    if softness < 0:
        raise ValueError("softness must be >= 0")
    if cvar_weight < 0:
        raise ValueError("cvar_weight must be >= 0")

    psf = _validate_and_normalize_psf(psf_bank)
    radial_distance = _radial_distance(psf)
    loss = psf.sum() * 0.0
    stats: Dict[str, torch.Tensor] = {}
    primary_violation = None
    primary_outside = None
    for index, (radius, budget, weight) in enumerate(zip(
            radii, outside_budgets, scale_weights)):
        radius = float(radius)
        budget = float(budget)
        weight = float(weight)
        if radius <= 0 or not 0.0 <= budget <= 1.0 or weight < 0:
            raise ValueError(
                f"invalid energy scale radius={radius}, budget={budget}, "
                f"weight={weight}")
        outside = _outside_energy(psf, radial_distance, radius, softness)
        violation = F.relu(outside - budget)
        loss = loss + weight * violation.square().mean()
        key = f"r{int(round(radius))}"
        with torch.no_grad():
            stats[f"{key}_outside_mean"] = outside.mean().detach()
            stats[f"{key}_outside_p90"] = torch.quantile(
                outside.detach().flatten(), 0.9)
            stats[f"{key}_outside_max"] = outside.max().detach()
            stats[f"{key}_inside_mean"] = (1.0 - outside.mean()).detach()
            stats[f"{key}_active_fraction"] = (
                violation > 0).to(psf.dtype).mean().detach()
        if index == 0:
            primary_violation = violation
            primary_outside = outside

    if cvar_weight > 0:
        cvar_loss = _top_fraction_mean(
            primary_violation.square(), cvar_fraction)
        loss = loss + float(cvar_weight) * cvar_loss
    else:
        cvar_loss = loss.detach() * 0.0
    with torch.no_grad():
        stats["cvar_loss"] = cvar_loss.detach()
        stats["outside_mean"] = primary_outside.mean().detach()
        stats["outside_p90"] = torch.quantile(
            primary_outside.detach().flatten(), 0.9)
        stats["outside_max"] = primary_outside.max().detach()
        stats["inside_mean"] = (1.0 - primary_outside.mean()).detach()
        stats["active_fraction"] = (
            primary_violation > 0).to(psf.dtype).mean().detach()
        stats.update(_encircled_energy_radius_stats(
            psf.detach(), radial_distance.detach()))
    return loss, stats


def psf_mtf_floor_loss(
    psf_bank: torch.Tensor,
    min_frequency: float = 0.02,
    max_frequency: float = 0.15,
    mtf_at_005: float = 0.12,
    mtf_at_010: float = 0.05,
    mtf_at_015: float = 0.025,
    worst_fraction: float = 0.10,
    worst_weight: float = 0.25,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Penalize low/mid-frequency MTF values below a conservative floor."""
    if not 0.0 <= min_frequency < max_frequency <= 0.5:
        raise ValueError("MTF frequency band must lie in [0,0.5]")
    for value in (mtf_at_005, mtf_at_010, mtf_at_015):
        if not 0.0 <= value <= 1.0:
            raise ValueError("MTF targets must be in [0,1]")
    psf = _validate_and_normalize_psf(psf_bank)
    height, width = psf.shape[-2:]
    fy = torch.fft.fftshift(torch.fft.fftfreq(
        height, device=psf.device, dtype=psf.dtype))
    fx = torch.fft.fftshift(torch.fft.fftfreq(
        width, device=psf.device, dtype=psf.dtype))
    grid_y, grid_x = torch.meshgrid(fy, fx, indexing="ij")
    frequency = torch.sqrt(grid_x.square() + grid_y.square())
    mtf = torch.abs(torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(psf, dim=(-2, -1))),
        dim=(-2, -1),
    ))
    dc = mtf[..., height // 2, width // 2].unsqueeze(-1).unsqueeze(-1)
    mtf = mtf / dc.clamp_min(torch.finfo(psf.dtype).eps)

    target = torch.where(
        frequency <= 0.05,
        torch.full_like(frequency, float(mtf_at_005)),
        torch.where(
            frequency <= 0.10,
            float(mtf_at_005)
            + (frequency - 0.05) / 0.05
            * (float(mtf_at_010) - float(mtf_at_005)),
            float(mtf_at_010)
            + (frequency - 0.10) / 0.05
            * (float(mtf_at_015) - float(mtf_at_010)),
        ),
    )
    band = (
        (frequency >= float(min_frequency))
        & (frequency <= float(max_frequency))
    )
    violation = F.relu(target[band] - mtf[..., band])
    per_psf = violation.square().mean(dim=-1)
    loss = per_psf.mean()
    if worst_weight > 0:
        loss = loss + float(worst_weight) * _top_fraction_mean(
            per_psf, worst_fraction)

    with torch.no_grad():
        stats: Dict[str, torch.Tensor] = {}
        bin_half_width = 1.0 / min(height, width)
        for frequency_value, label in (
            (0.05, "005"), (0.10, "010"), (0.20, "020")
        ):
            annulus = torch.abs(frequency - frequency_value) <= bin_half_width
            per_psf_value = mtf[..., annulus].mean(dim=-1)
            stats[f"mtf_{label}_mean"] = per_psf_value.mean().detach()
            stats[f"mtf_{label}_p10"] = torch.quantile(
                per_psf_value.detach().flatten(), 0.1)
        stats["active_fraction"] = (
            per_psf > 0).to(psf.dtype).mean().detach()
    return loss, stats


def sensor_weighted_spectral_psf_separation_loss(
    psf_bank: torch.Tensor,
    sensor_response: torch.Tensor,
    margin: float = 0.95,
    offsets: Sequence[int] = (1,),
    hard_fraction: float = 0.0,
    hard_weight: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Penalize overly similar adjacent-wavelength sensor PSF signatures.

    Each wavelength signature contains all sensor channels and spatial PSF
    samples.  This matches the information that survives spectral collapse in
    the RGB measurement instead of optimizing bare PSFs that the sensor never
    observes directly.
    """
    if psf_bank.ndim != 4:
        raise ValueError(
            "psf_bank must have shape [depth, wavelength, H, W], "
            f"got {tuple(psf_bank.shape)}")
    if sensor_response.ndim != 2:
        raise ValueError(
            "sensor_response must have shape [sensor_channel, wavelength], "
            f"got {tuple(sensor_response.shape)}")
    if sensor_response.shape[1] != psf_bank.shape[1]:
        raise ValueError(
            f"sensor wavelengths ({sensor_response.shape[1]}) do not match "
            f"PSF wavelengths ({psf_bank.shape[1]})")
    if psf_bank.shape[1] < 2:
        raise ValueError("spectral PSF separation requires at least two wavelengths")
    if not -1.0 <= margin <= 1.0:
        raise ValueError(f"margin must be in [-1, 1], got {margin}")
    if not torch.isfinite(psf_bank).all() or not torch.isfinite(sensor_response).all():
        raise ValueError("PSF bank and sensor response must be finite")

    psf = psf_bank.to(dtype=torch.float32)
    response = sensor_response.to(device=psf.device, dtype=psf.dtype)
    # [D, L, S, H, W], where S is the number of sensor channels.
    signatures = psf.unsqueeze(2) * response.t()[None, :, :, None, None]
    signatures = signatures.flatten(start_dim=2)
    signatures = F.normalize(signatures, p=2, dim=-1, eps=1e-12)
    response_strength = torch.linalg.vector_norm(response, dim=0)
    cosine_groups = []
    violation_groups = []
    weight_groups = []
    for offset in offsets:
        offset = int(offset)
        if offset < 1 or offset >= psf.shape[1]:
            raise ValueError(
                f"spectral offset must be in [1,{psf.shape[1] - 1}], got {offset}")
        cosine = (
            signatures[:, :-offset] * signatures[:, offset:]
        ).sum(dim=-1)
        pair_weight = 0.5 * (
            response_strength[:-offset] + response_strength[offset:])
        pair_weight = pair_weight / pair_weight.mean().clamp_min(1e-12)
        pair_weight = pair_weight.unsqueeze(0).expand_as(cosine)
        violation = F.relu(cosine - float(margin))
        cosine_groups.append(cosine)
        violation_groups.append(violation)
        weight_groups.append(pair_weight)
    all_cosine = torch.cat([value.flatten() for value in cosine_groups])
    all_violation = torch.cat([value.flatten() for value in violation_groups])
    all_weights = torch.cat([value.flatten() for value in weight_groups])
    loss = (all_violation * all_weights).sum() / all_weights.sum().clamp_min(1e-12)
    if hard_weight > 0:
        loss = loss + float(hard_weight) * _top_fraction_mean(
            all_violation, hard_fraction)

    with torch.no_grad():
        stats = {
            "adjacent_cosine_mean": all_cosine.mean().detach(),
            "adjacent_cosine_p90": torch.quantile(
                all_cosine.detach().flatten(), 0.9),
            "adjacent_cosine_max": all_cosine.max().detach(),
            "active_fraction": (
                all_violation > 0).to(psf.dtype).mean().detach(),
        }
    return loss, stats


def sensor_weighted_depth_psf_separation_loss(
    psf_bank: torch.Tensor,
    sensor_response: torch.Tensor,
    margin: float = 0.90,
    hard_fraction: float = 0.20,
    hard_weight: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Separate adjacent-depth PSFs as they appear in the RGB measurement."""
    psf = _validate_and_normalize_psf(psf_bank)
    response = sensor_response.to(device=psf.device, dtype=psf.dtype)
    if response.ndim != 2 or response.shape[1] != psf.shape[1]:
        raise ValueError(
            "sensor_response must have shape [sensor_channel, wavelength]")
    if psf.shape[0] < 2:
        raise ValueError("depth PSF separation requires at least two depths")
    effective = torch.einsum("sl,dlhw->dshw", response, psf)
    signatures = F.normalize(
        effective.flatten(start_dim=1), p=2, dim=-1, eps=1e-12)
    cosine = (signatures[:-1] * signatures[1:]).sum(dim=-1)
    violation = F.relu(cosine - float(margin))
    loss = violation.mean()
    if hard_weight > 0:
        loss = loss + float(hard_weight) * _top_fraction_mean(
            violation, hard_fraction)
    with torch.no_grad():
        stats = {
            "adjacent_cosine_mean": cosine.mean().detach(),
            "adjacent_cosine_p90": torch.quantile(
                cosine.detach().flatten(), 0.9),
            "adjacent_cosine_max": cosine.max().detach(),
            "active_fraction": (
                violation > 0).to(psf.dtype).mean().detach(),
        }
    return loss, stats


def zernike_order_weighted_l2(
    coefficients: torch.Tensor,
    protected_terms: int = 15,
) -> torch.Tensor:
    """Weakly discourage unnecessary high-order Zernike coefficients."""
    if coefficients.ndim != 1:
        raise ValueError("Zernike coefficients must be a 1-D tensor")
    protected_terms = max(0, int(protected_terms))
    if protected_terms >= coefficients.numel():
        return coefficients.sum() * 0.0
    indices = torch.arange(
        coefficients.numel(), device=coefficients.device,
        dtype=coefficients.dtype)
    # Sequential Zernike bases contain n+1 modes at radial order n.
    radial_order = torch.ceil(
        (-3.0 + torch.sqrt(9.0 + 8.0 * indices)) / 2.0
    ).clamp_min(0.0)
    weights = (radial_order / radial_order.max().clamp_min(1.0)).square()
    weights[:protected_terms] = 0.0
    return (weights * coefficients.square()).mean()
