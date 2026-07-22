"""Differentiable regularizers for normalized intensity PSF banks."""

from typing import Dict, Tuple

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
    if psf_bank.ndim != 4:
        raise ValueError(
            "psf_bank must have shape [depth, wavelength, H, W], "
            f"got {tuple(psf_bank.shape)}"
        )
    if radius <= 0:
        raise ValueError(f"radius must be > 0, got {radius}")
    if not 0.0 <= outside_budget <= 1.0:
        raise ValueError(
            f"outside_budget must be in [0, 1], got {outside_budget}"
        )
    if softness < 0:
        raise ValueError(f"softness must be >= 0, got {softness}")
    if not torch.isfinite(psf_bank).all():
        raise ValueError("psf_bank contains NaN or Inf")

    psf = psf_bank.to(dtype=torch.float32)
    if torch.any(psf < 0):
        raise ValueError("psf_bank must contain non-negative intensity values")

    height, width = psf.shape[-2:]
    yy = torch.arange(height, device=psf.device, dtype=psf.dtype) - height // 2
    xx = torch.arange(width, device=psf.device, dtype=psf.dtype) - width // 2
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
    radial_distance = torch.sqrt(grid_x.square() + grid_y.square())

    if softness == 0:
        outside_mask = (radial_distance > radius).to(dtype=psf.dtype)
    else:
        outside_mask = torch.sigmoid((radial_distance - radius) / softness)

    # Normalize each intensity PSF independently. This makes the regularizer a
    # fractional-energy constraint and prevents its scale from depending on
    # wavelength, depth, or numerical throughput.
    eps = torch.finfo(psf.dtype).eps
    psf = psf / psf.sum(dim=(-2, -1), keepdim=True).clamp_min(eps)
    outside_energy = (psf * outside_mask).sum(dim=(-2, -1))
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
