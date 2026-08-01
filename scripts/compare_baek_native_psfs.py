#!/usr/bin/env python3
"""Compare the deployed Baek-native PSFs against the original PADO pipeline.

The reference side executes the same public PADO API calls as
``e2e_HSD.ipynb``.  The candidate side executes this repository's
``doe_native_grid_v1`` implementation.  All 25 wavelengths and all 20
inverse-depth samples from the notebook are evaluated.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import pado
except ImportError as exc:  # pragma: no cover - exercised by the CLI
    raise SystemExit(
        "PADO is required on PYTHONPATH. Install it in a temporary target with "
        "`python -m pip install --no-deps --target /tmp/pado_pkg "
        "git+https://github.com/shwbaek/pado.git`."
    ) from exc

from torch_optics.forward_dodo import DepthAwareDoDoForwardModel


GRID_SIZE = 376
PITCH_M = 8e-6
FOCAL_LENGTH_M = 50e-3
CROP_96 = 96
CROP_129 = 129


def _center_crop(x: torch.Tensor, size: int) -> torch.Tensor:
    top = x.shape[-2] // 2 - size // 2
    left = x.shape[-1] // 2 - size // 2
    return x[..., top : top + size, left : left + size]


def _normalize(x: torch.Tensor) -> torch.Tensor:
    return x / x.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-20)


def _comparison(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    reference = _normalize(reference)
    candidate = _normalize(candidate)
    difference = candidate - reference
    ref_flat = reference.flatten()
    candidate_flat = candidate.flatten()
    cosine = F.cosine_similarity(ref_flat, candidate_flat, dim=0)
    nrmse = torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(
        reference
    ).clamp_min(1e-20)
    total_variation = 0.5 * difference.abs().sum()
    ref_peak = int(torch.argmax(ref_flat))
    candidate_peak = int(torch.argmax(candidate_flat))
    ref_y, ref_x = divmod(ref_peak, reference.shape[-1])
    candidate_y, candidate_x = divmod(candidate_peak, candidate.shape[-1])
    peak_shift = math.hypot(candidate_y - ref_y, candidate_x - ref_x)
    return {
        "cosine": float(cosine.item()),
        "nrmse": float(nrmse.item()),
        "tv": float(total_variation.item()),
        "peak_shift_px": float(peak_shift),
    }


def _complex_coherence(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    numerator = torch.abs(torch.sum(torch.conj(reference.flatten()) * candidate.flatten()))
    denominator = (
        torch.linalg.vector_norm(reference) * torch.linalg.vector_norm(candidate)
    ).clamp_min(1e-20)
    return float((numerator / denominator).item())


def _build_current_model(height_path: Path, depths: torch.Tensor, device: torch.device):
    model = DepthAwareDoDoForwardModel(
        depth_min=0.3,
        depth_max=2.0,
        num_depth_layers=20,
        use_second_doe=False,
        doe_type_a="Zeros",
        train_c=False,
        input_format="nchw",
        output_format="nchw",
        assets_dir=str(Path(__file__).resolve().parents[1] / "torch_optics/assets"),
        measurement_norm_mode="none",
        sensing_mode="identity",
        measurement_channels=25,
        depth_layering_mode="soft_diopter",
        sensor_measurement="intensity",
        skip_prop2=True,
        prop1_padding_factor=1,
        image_formation_mode="psf_convolution",
        psf_layer_mask_mode="baek_hard",
        psf_boundary_mode="linear_zero",
        doe_parameterization="fixed_height",
        doe_height_path=str(height_path),
        doe_height_pad_to_size=376,
        psf_optics_version="doe_native_grid_v1",
    ).to(device).eval()
    # Use the notebook order (near to far) rather than the binner's internal
    # descending order. This affects only this explicit PSF parity experiment.
    model.z_centers.copy_(depths.to(device=device, dtype=torch.float32))
    model.clear_static_optics_cache()
    return model


def _pado_scalar_source_bank(
    depths: torch.Tensor,
    wavelengths: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    coordinates = torch.arange(
        -GRID_SIZE // 2,
        GRID_SIZE // 2,
        device=device,
        dtype=torch.float32,
    ) * PITCH_M
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    radial_squared = xx.square() + yy.square()
    fields = []
    for depth in depths.tolist():
        radius = torch.sqrt(radial_squared + float(depth) ** 2)
        bands = []
        for wavelength in wavelengths.tolist():
            # Deliberately use Python float division, matching the notebook's
            # one-wavelength PADO Light object exactly.
            phase = (2 * torch.pi * radius / float(wavelength)) % (2 * torch.pi)
            bands.append(torch.exp(1j * phase))
        fields.append(torch.stack(bands, dim=0))
    return torch.stack(fields, dim=0)


def _generate_current_intensities(
    model,
    depths: torch.Tensor,
    wavelengths: torch.Tensor,
    device: torch.device,
    depth_chunk: int,
):
    with torch.no_grad():
        current_sources = model._prop1_impulse_field_bank(
            GRID_SIZE, GRID_SIZE, device
        )
        scalar_sources = _pado_scalar_source_bank(depths, wavelengths, device)
        source_coherence = torch.empty(
            (depths.numel(), wavelengths.numel()), dtype=torch.float32
        )
        for depth_index in range(depths.numel()):
            for wavelength_index in range(wavelengths.numel()):
                source_coherence[depth_index, wavelength_index] = _complex_coherence(
                    scalar_sources[depth_index, wavelength_index],
                    current_sources[depth_index, wavelength_index],
                )

        current_intensities = []
        scalar_source_intensities = []
        for start in range(0, depths.numel(), depth_chunk):
            end = min(start + depth_chunk, depths.numel())
            current_output = model._propagate_after_prop1(current_sources[start:end])
            scalar_output = model._propagate_after_prop1(scalar_sources[start:end])
            current_intensities.append(
                _normalize(current_output.abs().square()).cpu()
            )
            scalar_source_intensities.append(
                _normalize(scalar_output.abs().square()).cpu()
            )
        return (
            torch.cat(current_intensities, dim=0),
            torch.cat(scalar_source_intensities, dim=0),
            source_coherence,
        )


def _save_metric_heatmaps(rows: list[dict], output_path: Path) -> None:
    metrics = (
        ("full_nrmse", "Full-grid NRMSE", "magma"),
        ("full_cosine", "Full-grid cosine", "viridis"),
        ("aligned_full_nrmse", "NRMSE after PADO-scalar source", "magma"),
        ("capture129_delta", "129 crop capture: current - PADO", "coolwarm"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    for axis, (key, title, cmap) in zip(axes.flat, metrics):
        values = np.asarray([row[key] for row in rows], dtype=np.float64).reshape(25, 20).T
        if key == "capture129_delta":
            limit = max(float(np.abs(values).max()), 1e-8)
            image = axis.imshow(values, aspect="auto", cmap=cmap, vmin=-limit, vmax=limit)
        else:
            image = axis.imshow(values, aspect="auto", cmap=cmap)
        axis.set_title(title)
        axis.set_xlabel("wavelength index (420–660 nm)")
        axis.set_ylabel("depth index (0.3–2.0 m, inverse-depth spacing)")
        figure.colorbar(image, ax=axis, fraction=0.046)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _save_representative_figure(
    reference_96: torch.Tensor,
    current_96: torch.Tensor,
    depths: torch.Tensor,
    wavelengths: torch.Tensor,
    output_path: Path,
) -> None:
    wavelength_indices = [0, 6, 12, 18, 24]
    depth_indices = [0, 4, 9, 14, 19]
    figure, axes = plt.subplots(5, 15, figsize=(30, 10), constrained_layout=True)
    for row_index, depth_index in enumerate(depth_indices):
        for group_index, wavelength_index in enumerate(wavelength_indices):
            reference = reference_96[wavelength_index, depth_index]
            current = current_96[wavelength_index, depth_index]
            scale = max(float(reference.max()), float(current.max()), 1e-20)
            difference = torch.abs(current - reference)
            panels = (reference / scale, current / scale, difference / scale)
            for panel_index, panel in enumerate(panels):
                axis = axes[row_index, 3 * group_index + panel_index]
                axis.imshow(panel.numpy(), cmap="inferno", vmin=0.0, vmax=1.0)
                axis.set_xticks([])
                axis.set_yticks([])
                if row_index == 0:
                    label = ("PADO", "current", "abs diff")[panel_index]
                    axis.set_title(
                        f"{float(wavelengths[wavelength_index]) * 1e9:.0f}nm\n{label}",
                        fontsize=8,
                    )
                if group_index == 0 and panel_index == 0:
                    axis.set_ylabel(f"z={float(depths[depth_index]):.3f}m")
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _save_worst_case_figure(
    worst_row: dict,
    reference_96: torch.Tensor,
    current_96: torch.Tensor,
    aligned_96: torch.Tensor,
    output_path: Path,
) -> None:
    wavelength_index = int(worst_row["wavelength_index"])
    depth_index = int(worst_row["depth_index"])
    reference = reference_96[wavelength_index, depth_index]
    current = current_96[wavelength_index, depth_index]
    aligned = aligned_96[wavelength_index, depth_index]
    scale = max(
        float(reference.max()), float(current.max()), float(aligned.max()), 1e-20
    )
    panels = (
        (reference / scale, "PADO"),
        (current / scale, "current"),
        (aligned / scale, "current + PADO scalar source"),
        (torch.abs(current - reference) / scale, "|current - PADO|"),
        (torch.abs(aligned - reference) / scale, "|aligned - PADO|"),
    )
    figure, axes = plt.subplots(1, 5, figsize=(17, 3.6), constrained_layout=True)
    for axis, (panel, title) in zip(axes, panels):
        axis.imshow(panel.numpy(), cmap="inferno", vmin=0.0, vmax=1.0)
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(
        f"Worst full-grid NRMSE: {worst_row['wavelength_nm']:.0f} nm, "
        f"z={worst_row['depth_m']:.3f} m"
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _aggregate(rows: list[dict], key: str) -> dict:
    values = np.asarray([row[key] for row in rows], dtype=np.float64)
    return {
        "min": float(values.min()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--height_path",
        type=Path,
        default=repo_root
        / "e2e_HSD_learned_DOE_and_PSF_simulation/e2e_HSD_doe_height.pth",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=repo_root
        / "论文实验/PSF卷积/baek_native_psf_parity_20260801",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--depth_chunk", type=int, default=2)
    parser.add_argument("--pado_revision", default="unknown")
    args = parser.parse_args()

    if not args.height_path.is_file():
        raise FileNotFoundError(args.height_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    wavelengths = torch.linspace(420e-9, 660e-9, 25, dtype=torch.float32)
    depths = 1.0 / torch.linspace(1.0 / 0.3, 1.0 / 2.0, 20, dtype=torch.float32)

    model = _build_current_model(args.height_path, depths, device)
    current_full, aligned_full, source_coherence = _generate_current_intensities(
        model, depths, wavelengths, device, args.depth_chunk
    )

    height = torch.load(args.height_path, map_location=device, weights_only=True)
    height = F.pad(height.to(torch.float32), (0, 1, 0, 1))
    material = pado.material.Material("NOA61")
    propagator = pado.propagator.Propagator("Fresnel")
    dim = (1, 1, GRID_SIZE, GRID_SIZE)
    rows = []
    reference_crops_96 = torch.empty((25, 20, CROP_96, CROP_96))
    current_crops_96 = torch.empty_like(reference_crops_96)
    aligned_crops_96 = torch.empty_like(reference_crops_96)

    with torch.no_grad():
        for wavelength_index, wavelength_tensor in enumerate(wavelengths):
            wavelength = float(wavelength_tensor.item())
            for depth_index, depth_tensor in enumerate(depths):
                depth = float(depth_tensor.item())
                light = pado.light.Light(dim, PITCH_M, wavelength, device=str(device))
                light.set_spherical_light(depth)
                doe = pado.optical_element.DOE(
                    dim, PITCH_M, material, wavelength, str(device), height
                )
                light = doe.forward(light)
                aperture = pado.optical_element.Aperture(
                    dim,
                    PITCH_M,
                    GRID_SIZE * PITCH_M,
                    "circle",
                    wavelength,
                    str(device),
                )
                light = aperture.forward(light)
                reference = propagator.forward(light, FOCAL_LENGTH_M).get_intensity()[
                    0, 0
                ]
                reference = _normalize(reference)
                current = current_full[depth_index, wavelength_index].to(device)
                aligned = aligned_full[depth_index, wavelength_index].to(device)

                full_metric = _comparison(reference, current)
                aligned_metric = _comparison(reference, aligned)
                reference_96 = _center_crop(reference, CROP_96)
                current_96 = _center_crop(current, CROP_96)
                aligned_96 = _center_crop(aligned, CROP_96)
                reference_129 = _center_crop(reference, CROP_129)
                current_129 = _center_crop(current, CROP_129)
                crop96_metric = _comparison(reference_96, current_96)
                crop129_metric = _comparison(reference_129, current_129)

                reference_crops_96[wavelength_index, depth_index] = _normalize(
                    reference_96
                ).cpu()
                current_crops_96[wavelength_index, depth_index] = _normalize(
                    current_96
                ).cpu()
                aligned_crops_96[wavelength_index, depth_index] = _normalize(
                    aligned_96
                ).cpu()
                reference_capture96 = float(reference_96.sum().item())
                current_capture96 = float(current_96.sum().item())
                reference_capture129 = float(reference_129.sum().item())
                current_capture129 = float(current_129.sum().item())
                rows.append(
                    {
                        "wavelength_index": wavelength_index,
                        "wavelength_nm": wavelength * 1e9,
                        "depth_index": depth_index,
                        "depth_m": depth,
                        "source_complex_coherence": float(
                            source_coherence[depth_index, wavelength_index]
                        ),
                        "full_cosine": full_metric["cosine"],
                        "full_nrmse": full_metric["nrmse"],
                        "full_tv": full_metric["tv"],
                        "full_peak_shift_px": full_metric["peak_shift_px"],
                        "aligned_full_cosine": aligned_metric["cosine"],
                        "aligned_full_nrmse": aligned_metric["nrmse"],
                        "crop96_cosine": crop96_metric["cosine"],
                        "crop96_nrmse": crop96_metric["nrmse"],
                        "crop129_cosine": crop129_metric["cosine"],
                        "crop129_nrmse": crop129_metric["nrmse"],
                        "pado_capture96": reference_capture96,
                        "current_capture96": current_capture96,
                        "capture96_delta": current_capture96
                        - reference_capture96,
                        "pado_capture129": reference_capture129,
                        "current_capture129": current_capture129,
                        "capture129_delta": current_capture129
                        - reference_capture129,
                    }
                )

    metric_path = args.output_dir / "per_psf_metrics.csv"
    with metric_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    aggregate_keys = (
        "source_complex_coherence",
        "full_cosine",
        "full_nrmse",
        "full_tv",
        "full_peak_shift_px",
        "aligned_full_cosine",
        "aligned_full_nrmse",
        "crop96_cosine",
        "crop96_nrmse",
        "crop129_cosine",
        "crop129_nrmse",
        "pado_capture96",
        "current_capture96",
        "capture96_delta",
        "pado_capture129",
        "current_capture129",
        "capture129_delta",
    )
    worst_nrmse = max(rows, key=lambda row: row["full_nrmse"])
    worst_cosine = min(rows, key=lambda row: row["full_cosine"])
    summary = {
        "height_path": str(args.height_path.resolve()),
        "notebook": str(
            (args.height_path.parent / "e2e_HSD.ipynb").resolve()
        ),
        "pado_version": getattr(pado, "__version__", "unknown"),
        "pado_revision": args.pado_revision,
        "device": str(device),
        "grid": {"size": GRID_SIZE, "pitch_m": PITCH_M},
        "focal_length_m": FOCAL_LENGTH_M,
        "wavelength_count": 25,
        "depth_count": 20,
        "pair_count": len(rows),
        "aggregates": {key: _aggregate(rows, key) for key in aggregate_keys},
        "worst_full_nrmse": worst_nrmse,
        "worst_full_cosine": worst_cosine,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    torch.save(
        {
            "wavelengths_m": wavelengths,
            "depths_m": depths,
            "pado_normalized_96": reference_crops_96,
            "current_normalized_96": current_crops_96,
            "pado_scalar_source_normalized_96": aligned_crops_96,
        },
        args.output_dir / "normalized_psf_crops_96.pt",
    )
    _save_metric_heatmaps(rows, args.output_dir / "metric_heatmaps.png")
    _save_representative_figure(
        reference_crops_96,
        current_crops_96,
        depths,
        wavelengths,
        args.output_dir / "representative_psf_comparison.png",
    )
    _save_worst_case_figure(
        worst_nrmse,
        reference_crops_96,
        current_crops_96,
        aligned_crops_96,
        args.output_dir / "worst_case_psf_comparison.png",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
