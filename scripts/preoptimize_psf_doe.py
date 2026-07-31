#!/usr/bin/env python
"""DOE-only feasibility search for the current consistent-grid PSF model.

The reconstruction network and dataset are deliberately absent.  Rank-9 and
free-150 Zernike DOEs receive the same pupil-height RMS budget and are judged
by point-source Fisher information, sensor-visible PSF bandwidth,
wavelength/depth separation, and a loose energy-spread guard.
"""

import argparse
import json
import math
import shlex
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from torch_optics.forward_dodo import DepthAwareDoDoForwardModel
from util.doe_preoptimization import (
    DOEPreoptimizationTargets,
    DOEPreoptimizationWeights,
    doe_physical_stats,
    doe_preoptimization_objective,
    initialize_doe_height_,
    rms_constrained_optimizer_step_,
)


MODE_SPECS = {
    "rank9": {"free": False, "n_terms": 12, "basis_mode": "orthogonal_rms"},
    "free150": {"free": True, "n_terms": 150, "basis_mode": "legacy_raw12"},
}


def _float_metrics(metrics: Dict[str, torch.Tensor]) -> Dict[str, float]:
    return {name: float(value.detach().cpu().item()) for name, value in metrics.items()}


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)


def _sensor_response(model, device: torch.device) -> torch.Tensor:
    sensing = model.sensing_unnorm
    return torch.stack(
        [sensing.sensor_r, sensing.sensor_g, sensing.sensor_b], dim=0
    ).to(device=device, dtype=torch.float32)


def build_model(args, mode: str, device: torch.device):
    spec = MODE_SPECS[mode]
    model = DepthAwareDoDoForwardModel(
        depth_min=args.depth_min,
        depth_max=args.depth_max,
        num_depth_layers=args.depth_layers,
        use_second_doe=False,
        doe_type_a="New",
        train_c=True,
        free=spec["free"],
        n_terms=spec["n_terms"],
        input_format="nchw",
        output_format="nchw",
        measurement_norm_mode="none",
        sensing_mode="rgb",
        measurement_channels=3,
        depth_layering_mode="soft_diopter",
        sensor_measurement="intensity",
        skip_prop2=True,
        prop1_padding_factor=2,
        image_formation_mode="psf_convolution",
        psf_layer_mask_mode="baek_hard",
        psf_mask_blur_sigma=1.0,
        psf_boundary_mode="linear_zero",
        psf_depth_chunk_size=4,
        doe_basis_mode=spec["basis_mode"],
        doe_basis_rank=9,
        doe_basis_rank_rtol=1e-4,
        doe_basis_rms_m=3e-6,
        doe_coeff_norm_limit=1.0,
        doe_init_coeff_norm=0.2,
        psf_optics_version="consistent_grid_v1",
    ).to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.doe1.zernike_coeffs.requires_grad_(True)
    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if len(trainable) != 1 or trainable[0][1] is not model.doe1.zernike_coeffs:
        raise RuntimeError(
            "DOE preoptimization must expose exactly doe1.zernike_coeffs as "
            f"trainable, got {[name for name, _ in trainable]}"
        )
    return model


def _save_heightmap(doe, output_dir: Path) -> None:
    height_um = doe.heightmap().detach().cpu().numpy() * 1e6
    np.save(output_dir / "best_heightmap_m.npy", height_um * 1e-6)
    np.save(
        output_dir / "best_coefficients.npy",
        doe.zernike_coeffs.detach().cpu().numpy(),
    )
    pupil = doe.spiral_p.detach().cpu().numpy() > 0.5
    display = np.where(pupil, height_um, np.nan)
    plt.figure(figsize=(5.2, 4.4))
    image = plt.imshow(display, cmap="coolwarm")
    plt.colorbar(image, fraction=0.046, pad=0.04, label="height (μm)")
    plt.title("Best DOE height map")
    plt.tight_layout()
    plt.savefig(output_dir / "best_heightmap.png", dpi=180)
    plt.close()


def _save_psf_montage(psf_bank: torch.Tensor, output_dir: Path) -> None:
    psf = psf_bank.detach().cpu().float().numpy()
    depth_indices = sorted({0, psf.shape[0] // 2, psf.shape[0] - 1})
    wavelength_indices = sorted({0, psf.shape[1] // 2, psf.shape[1] - 1})
    figure, axes = plt.subplots(
        len(depth_indices), len(wavelength_indices), figsize=(8.5, 8.0)
    )
    axes = np.asarray(axes).reshape(len(depth_indices), len(wavelength_indices))
    for row, depth_index in enumerate(depth_indices):
        for column, wavelength_index in enumerate(wavelength_indices):
            value = psf[depth_index, wavelength_index]
            log_value = np.log10(value / max(float(value.max()), 1e-20) + 1e-6)
            axes[row, column].imshow(log_value, cmap="magma", vmin=-6.0, vmax=0.0)
            axes[row, column].set_title(f"depth#{depth_index}, λ#{wavelength_index}")
            axes[row, column].axis("off")
    figure.suptitle("Best PSF bank (log10 normalized intensity)")
    figure.tight_layout()
    figure.savefig(output_dir / "best_psf_montage.png", dpi=180)
    plt.close(figure)


def _separation_scale(step: int, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, float(step) / float(warmup_steps))


def _run_one(args, mode: str, seed: int, output_dir: Path) -> Dict:
    device = torch.device(args.device)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    model = build_model(args, mode, device)
    initialize_doe_height_(
        model.doe1,
        target_rms_m=args.initial_height_rms_um * 1e-6,
        generator=generator,
    )
    coefficients = model.doe1.zernike_coeffs
    optimizer = torch.optim.Adam([coefficients], lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.steps, 1),
        eta_min=args.lr * args.final_lr_ratio,
    )
    response = _sensor_response(model, device)
    weights = DOEPreoptimizationWeights(
        fisher=args.fisher_weight,
        mtf=args.mtf_weight,
        optical_spectral_separation=args.optical_spectral_weight,
        optical_depth_separation=args.optical_depth_weight,
        sensor_spectral_separation=args.sensor_spectral_weight,
        sensor_depth_separation=args.sensor_depth_weight,
        energy_guard=args.energy_weight,
    )
    targets = DOEPreoptimizationTargets(
        fisher_ridge=args.fisher_ridge,
        fisher_loss_scale=args.fisher_loss_scale,
        fisher_spatial_crlb_weight=args.fisher_spatial_crlb_weight,
        fisher_depth_crlb_weight=args.fisher_depth_crlb_weight,
        fisher_wavelength_crlb_weight=args.fisher_wavelength_crlb_weight,
        mtf_min_frequency=args.mtf_min_frequency,
        mtf_max_frequency=args.mtf_max_frequency,
        mtf_at_005=args.mtf_target_005,
        mtf_at_010=args.mtf_target_010,
        mtf_at_015=args.mtf_target_015,
        spectral_margin=args.spectral_margin,
        depth_margin=args.depth_margin,
        optical_spectral_offsets=tuple(args.optical_spectral_offsets),
        optical_depth_offsets=tuple(args.optical_depth_offsets),
        energy_radii=(args.energy_radius, args.energy_outer_radius),
        energy_outside_budgets=(
            args.energy_outside_budget,
            args.energy_outer_outside_budget,
        ),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        **vars(args),
        "mode": mode,
        "seed": seed,
        "weights": asdict(weights),
        "targets": asdict(targets),
        "branch": "DOE可编码性预优化实验",
    }
    config["output_dir"] = str(output_dir)
    _write_json(output_dir / "config.json", config)

    best_loss = math.inf
    best_step = -1
    best_coefficients = None
    best_metrics = None
    initial_metrics = None
    history_path = output_dir / "history.jsonl"
    retraction_count = 0
    tangent_correction_count = 0
    minimum_retraction_scale = 1.0

    with history_path.open("w", encoding="utf-8") as history_file:
        for step in range(args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            psf_bank = model.psf_bank(use_cache=False)
            separation_scale = _separation_scale(step, args.separation_warmup_steps)
            total_loss, metrics = doe_preoptimization_objective(
                psf_bank,
                response,
                energy_reference=model.psf_energy_reference,
                weights=weights,
                targets=targets,
                separation_scale=separation_scale,
            )
            metrics.update(doe_physical_stats(model.doe1))
            metrics_float = _float_metrics(metrics)
            metrics_float["step"] = step
            metrics_float["optimizer/lr"] = float(optimizer.param_groups[0]["lr"])
            metrics_float["constraint/retraction_count"] = retraction_count
            metrics_float[
                "constraint/tangent_correction_count"
            ] = tangent_correction_count
            metrics_float[
                "constraint/minimum_retraction_scale"
            ] = minimum_retraction_scale

            if initial_metrics is None:
                initial_metrics = dict(metrics_float)
            if metrics_float["loss/full_total"] < best_loss:
                best_loss = metrics_float["loss/full_total"]
                best_step = step
                best_coefficients = coefficients.detach().cpu().clone()
                best_metrics = dict(metrics_float)

            should_log = step == 0 or step == args.steps or step % args.log_every == 0
            if should_log:
                history_file.write(
                    json.dumps(metrics_float, ensure_ascii=False, sort_keys=True) + "\n"
                )
                history_file.flush()
                print(
                    f"[{mode} seed={seed} step={step:04d}] "
                    f"train/full={metrics_float['loss/train_total']:.6f}/"
                    f"{metrics_float['loss/full_total']:.6f} "
                    f"FisherTask={metrics_float['fisher/task_a_optimality_mean']:.3e} "
                    f"MTF005(p10/mean)={metrics_float['mtf/005_p10']:.4f}/"
                    f"{metrics_float['mtf/005_mean']:.4f} "
                    f"spec_cos={metrics_float['spectral/adjacent_cosine_mean']:.4f} "
                    f"opt_spec={metrics_float['optical_spectral/cosine_mean']:.4f} "
                    f"depth_cos={metrics_float['depth/adjacent_cosine_mean']:.4f} "
                    f"r90={metrics_float.get('energy/r90_mean', float('nan')):.2f} "
                    f"height_rms={metrics_float['doe/height_rms_m'] * 1e6:.3f}μm"
                )

            if step == args.steps:
                break
            if not torch.isfinite(total_loss):
                raise FloatingPointError(f"non-finite DOE objective at step {step}")
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [coefficients], args.gradient_clip_norm
            )
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite DOE gradient at step {step}")
            constraint_stats = rms_constrained_optimizer_step_(
                model.doe1,
                optimizer,
                maximum_rms_m=args.maximum_height_rms_um * 1e-6,
                boundary_fraction=args.rms_boundary_fraction,
            )
            retraction_count += int(constraint_stats["retracted"])
            tangent_correction_count += int(constraint_stats["tangent_correction"])
            minimum_retraction_scale = min(
                minimum_retraction_scale,
                constraint_stats["retraction_scale"],
            )
            scheduler.step()

    if best_coefficients is None or best_metrics is None:
        raise RuntimeError("DOE search produced no eligible best state")
    with torch.no_grad():
        coefficients.copy_(best_coefficients.to(device))
        best_psf = model.psf_bank(use_cache=False)

    checkpoint = {
        "format": "doe_psf_preoptimization_v1",
        "mode": mode,
        "seed": seed,
        "best_step": best_step,
        "config": config,
        "metrics": best_metrics,
        "camera_state_dict": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "doe_coefficients": best_coefficients,
        "doe_heightmap_m": model.doe1.heightmap().detach().cpu(),
    }
    torch.save(checkpoint, output_dir / "best_doe.pt")
    if args.save_psf_bank:
        torch.save(best_psf.detach().cpu(), output_dir / "best_psf_bank.pt")
    _save_heightmap(model.doe1, output_dir)
    _save_psf_montage(best_psf, output_dir)

    summary = {
        "mode": mode,
        "seed": seed,
        "best_step": best_step,
        "projection_count": retraction_count,
        "constraint": {
            "retraction_count": retraction_count,
            "tangent_correction_count": tangent_correction_count,
            "minimum_retraction_scale": minimum_retraction_scale,
        },
        "initial": initial_metrics,
        "best": best_metrics,
        "improvement": {
            "loss_full_total": initial_metrics["loss/full_total"]
            - best_metrics["loss/full_total"],
            "fisher_a_optimality": initial_metrics["fisher/a_optimality_mean"]
            - best_metrics["fisher/a_optimality_mean"],
            "fisher_task_a_optimality": initial_metrics["fisher/task_a_optimality_mean"]
            - best_metrics["fisher/task_a_optimality_mean"],
            "fisher_weighted_a_optimality": initial_metrics[
                "fisher/weighted_a_optimality_mean"
            ]
            - best_metrics["fisher/weighted_a_optimality_mean"],
            "fisher_minimum_eigenvalue": best_metrics["fisher/minimum_eigenvalue_mean"]
            - initial_metrics["fisher/minimum_eigenvalue_mean"],
            "mtf_005_p10": (
                best_metrics["mtf/005_p10"] - initial_metrics["mtf/005_p10"]
            ),
            "mtf_005_mean": (
                best_metrics["mtf/005_mean"] - initial_metrics["mtf/005_mean"]
            ),
            "spectral_cosine_mean": initial_metrics["spectral/adjacent_cosine_mean"]
            - best_metrics["spectral/adjacent_cosine_mean"],
            "optical_spectral_cosine_mean": initial_metrics[
                "optical_spectral/cosine_mean"
            ]
            - best_metrics["optical_spectral/cosine_mean"],
            "optical_spectral_adjacent_cosine_mean": initial_metrics[
                "optical_spectral/adjacent_cosine_mean"
            ]
            - best_metrics["optical_spectral/adjacent_cosine_mean"],
            "depth_cosine_mean": initial_metrics["depth/adjacent_cosine_mean"]
            - best_metrics["depth/adjacent_cosine_mean"],
            "optical_depth_cosine_mean": initial_metrics["optical_depth/cosine_mean"]
            - best_metrics["optical_depth/cosine_mean"],
        },
        "feasibility": {
            "fisher_a_optimality_improved": (
                best_metrics["fisher/a_optimality_mean"]
                < initial_metrics["fisher/a_optimality_mean"]
            ),
            "fisher_task_a_optimality_improved": (
                best_metrics["fisher/task_a_optimality_mean"]
                < initial_metrics["fisher/task_a_optimality_mean"]
            ),
            "mtf_floor_satisfied": best_metrics["loss/mtf"] <= 1e-6,
            "optical_spectral_margin_satisfied": (
                best_metrics["loss/optical_spectral_separation"] <= 1e-6
            ),
            "optical_depth_margin_satisfied": (
                best_metrics["loss/optical_depth_separation"] <= 1e-6
            ),
            "all_information_targets_satisfied": (
                best_metrics["fisher/task_a_optimality_mean"]
                < initial_metrics["fisher/task_a_optimality_mean"]
                and best_metrics["loss/mtf"] <= 1e-6
                and best_metrics["loss/optical_spectral_separation"] <= 1e-6
                and best_metrics["loss/optical_depth_separation"] <= 1e-6
            ),
        },
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def _parse_args(argv: Iterable[str] = None):
    parser = argparse.ArgumentParser(
        description="Freeze the CNN and search directly for an informative PSF DOE."
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--modes", nargs="+", choices=tuple(MODE_SPECS), default=["rank9", "free150"]
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[123])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--final_lr_ratio", type=float, default=0.05)
    parser.add_argument("--gradient_clip_norm", type=float, default=1.0)
    parser.add_argument("--separation_warmup_steps", type=int, default=100)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_psf_bank", action="store_true")

    parser.add_argument("--depth_min", type=float, default=0.4)
    parser.add_argument("--depth_max", type=float, default=2.0)
    parser.add_argument("--depth_layers", type=int, default=16)
    parser.add_argument("--initial_height_rms_um", type=float, default=0.6)
    parser.add_argument("--maximum_height_rms_um", type=float, default=3.0)
    parser.add_argument("--rms_boundary_fraction", type=float, default=0.999)

    parser.add_argument("--fisher_weight", type=float, default=1.0)
    parser.add_argument("--fisher_ridge", type=float, default=1e-8)
    parser.add_argument("--fisher_loss_scale", type=float, default=1e-7)
    parser.add_argument("--fisher_spatial_crlb_weight", type=float, default=0.10)
    parser.add_argument("--fisher_depth_crlb_weight", type=float, default=1.0)
    parser.add_argument("--fisher_wavelength_crlb_weight", type=float, default=1.0)
    parser.add_argument("--mtf_weight", type=float, default=20.0)
    parser.add_argument("--optical_spectral_weight", type=float, default=5.0)
    parser.add_argument("--optical_depth_weight", type=float, default=2.0)
    parser.add_argument("--sensor_spectral_weight", type=float, default=0.0)
    parser.add_argument("--sensor_depth_weight", type=float, default=0.5)
    parser.add_argument("--energy_weight", type=float, default=0.10)
    parser.add_argument("--mtf_min_frequency", type=float, default=0.02)
    parser.add_argument("--mtf_max_frequency", type=float, default=0.15)
    parser.add_argument("--mtf_target_005", type=float, default=0.12)
    parser.add_argument("--mtf_target_010", type=float, default=0.05)
    parser.add_argument("--mtf_target_015", type=float, default=0.025)
    parser.add_argument("--spectral_margin", type=float, default=0.90)
    parser.add_argument("--depth_margin", type=float, default=0.90)
    parser.add_argument(
        "--optical_spectral_offsets", nargs="+", type=int, default=[1, 2, 4]
    )
    parser.add_argument("--optical_depth_offsets", nargs="+", type=int, default=[1])
    parser.add_argument("--energy_radius", type=float, default=16.0)
    parser.add_argument("--energy_outer_radius", type=float, default=24.0)
    parser.add_argument("--energy_outside_budget", type=float, default=0.75)
    parser.add_argument("--energy_outer_outside_budget", type=float, default=0.55)
    args = parser.parse_args(argv)

    if args.steps < 1:
        parser.error("--steps must be >= 1")
    if args.depth_layers < 2:
        parser.error("--depth_layers must be >= 2 for depth separation")
    if args.log_every < 1:
        parser.error("--log_every must be >= 1")
    if args.separation_warmup_steps < 0:
        parser.error("--separation_warmup_steps must be >= 0")
    if args.separation_warmup_steps > args.steps:
        parser.error("--separation_warmup_steps cannot exceed --steps")
    if args.lr <= 0.0 or args.final_lr_ratio <= 0.0:
        parser.error("learning rate and final_lr_ratio must be > 0")
    objective_weights = (
        args.fisher_weight,
        args.mtf_weight,
        args.optical_spectral_weight,
        args.optical_depth_weight,
        args.sensor_spectral_weight,
        args.sensor_depth_weight,
        args.energy_weight,
    )
    if any(weight < 0.0 for weight in objective_weights):
        parser.error("all objective weights must be >= 0")
    if args.fisher_ridge <= 0.0 or args.fisher_loss_scale <= 0.0:
        parser.error("Fisher ridge and loss scale must be > 0")
    fisher_parameter_weights = (
        args.fisher_spatial_crlb_weight,
        args.fisher_depth_crlb_weight,
        args.fisher_wavelength_crlb_weight,
    )
    if any(value < 0.0 for value in fisher_parameter_weights):
        parser.error("Fisher CRLB weights must be >= 0")
    if args.fisher_spatial_crlb_weight * 2.0 + sum(fisher_parameter_weights[1:]) <= 0.0:
        parser.error("at least one Fisher CRLB weight must be > 0")
    if args.initial_height_rms_um > args.maximum_height_rms_um:
        parser.error("initial DOE height RMS cannot exceed the maximum")
    if not 0.0 < args.rms_boundary_fraction <= 1.0:
        parser.error("--rms_boundary_fraction must lie in (0,1]")
    if any(offset < 1 for offset in args.optical_spectral_offsets):
        parser.error("optical spectral offsets must all be >= 1")
    if any(
        offset < 1 or offset >= args.depth_layers
        for offset in args.optical_depth_offsets
    ):
        parser.error("optical depth offsets must lie in [1,depth_layers-1]")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error(f"CUDA device requested but unavailable: {args.device}")
    return args


def main(argv: Iterable[str] = None) -> int:
    command_args = list(argv) if argv is not None else list(sys.argv[1:])
    args = _parse_args(command_args)
    torch.set_float32_matmul_precision("high")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    command = shlex.join([sys.executable, str(Path(__file__).resolve()), *command_args])
    (args.output_dir / "command.txt").write_text(command + "\n", encoding="utf-8")
    summaries: List[Dict] = []
    for mode in args.modes:
        for seed in args.seeds:
            run_dir = args.output_dir / mode / f"seed_{seed}"
            summaries.append(_run_one(args, mode, seed, run_dir))
    _write_json(args.output_dir / "comparison.json", {"runs": summaries})
    print(f"DOE feasibility comparison written to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
