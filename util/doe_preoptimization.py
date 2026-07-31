"""Reusable objective and physical-budget helpers for DOE-only PSF search."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Union

import torch

from util.psf_regularization import (
    multiscale_psf_energy_concentration_loss,
    psf_mtf_floor_loss,
    sensor_weighted_depth_psf_separation_loss,
    sensor_weighted_spectral_psf_separation_loss,
)


@dataclass(frozen=True)
class DOEPreoptimizationWeights:
    """Relative weights for the dimensionless PSF feasibility objectives."""

    mtf: float = 20.0
    spectral_separation: float = 1.0
    depth_separation: float = 1.0
    energy_guard: float = 0.10


@dataclass(frozen=True)
class DOEPreoptimizationTargets:
    """Conservative targets shared by rank-9 and higher-capacity searches."""

    mtf_min_frequency: float = 0.02
    mtf_max_frequency: float = 0.15
    mtf_at_005: float = 0.12
    mtf_at_010: float = 0.05
    mtf_at_015: float = 0.025
    spectral_margin: float = 0.90
    depth_margin: float = 0.90
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
    if not isinstance(coefficients, torch.nn.Parameter) or not coefficients.requires_grad:
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
    return {
        "doe/height_rms_m": doe.pupil_rms(height).detach(),
        "doe/height_min_m": selected.min().detach(),
        "doe/height_max_m": selected.max().detach(),
        "doe/coeff_norm": coefficients.detach().norm(),
    }


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
    spectral_loss, spectral_stats = sensor_weighted_spectral_psf_separation_loss(
        psf_bank,
        sensor_response,
        margin=targets.spectral_margin,
        offsets=(1,),
        hard_fraction=0.20,
        hard_weight=0.50,
    )
    depth_loss, depth_stats = sensor_weighted_depth_psf_separation_loss(
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

    weighted_mtf = float(weights.mtf) * mtf_loss
    weighted_spectral = (
        separation_scale * float(weights.spectral_separation) * spectral_loss
    )
    weighted_depth = (
        separation_scale * float(weights.depth_separation) * depth_loss
    )
    weighted_energy = float(weights.energy_guard) * energy_loss
    total = weighted_mtf + weighted_spectral + weighted_depth + weighted_energy

    metrics: Dict[str, torch.Tensor] = {
        "loss/total": total.detach(),
        "loss/mtf": mtf_loss.detach(),
        "loss/spectral_separation": spectral_loss.detach(),
        "loss/depth_separation": depth_loss.detach(),
        "loss/energy_guard": energy_loss.detach(),
        "weighted/mtf": weighted_mtf.detach(),
        "weighted/spectral_separation": weighted_spectral.detach(),
        "weighted/depth_separation": weighted_depth.detach(),
        "weighted/energy_guard": weighted_energy.detach(),
        "schedule/separation_scale": torch.tensor(
            separation_scale, device=psf_bank.device, dtype=psf_bank.dtype
        ),
        "mtf/005_mean": mtf_stats["mtf_005_mean"],
        "mtf/005_p10": mtf_stats["mtf_005_p10"],
        "mtf/010_mean": mtf_stats["mtf_010_mean"],
        "mtf/010_p10": mtf_stats["mtf_010_p10"],
        "mtf/020_mean": mtf_stats["mtf_020_mean"],
        "spectral/adjacent_cosine_mean": spectral_stats["adjacent_cosine_mean"],
        "spectral/adjacent_cosine_p90": spectral_stats["adjacent_cosine_p90"],
        "spectral/adjacent_cosine_max": spectral_stats["adjacent_cosine_max"],
        "depth/adjacent_cosine_mean": depth_stats["adjacent_cosine_mean"],
        "depth/adjacent_cosine_p90": depth_stats["adjacent_cosine_p90"],
        "depth/adjacent_cosine_max": depth_stats["adjacent_cosine_max"],
    }
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
