import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.preoptimize_psf_doe import main
from torch_optics.doe import DOELayer
from util.doe_preoptimization import (
    DOEPreoptimizationTargets,
    DOEPreoptimizationWeights,
    doe_preoptimization_objective,
    initialize_doe_height_,
    load_preoptimized_doe_,
    optical_psf_shape_separation_loss,
    psf_fisher_a_optimality_loss,
    rms_constrained_optimizer_step_,
)


def test_common_height_rms_initialization_and_projection():
    doe = DOELayer(
        doe_type="New",
        trainable=True,
        basis_mode="orthogonal_rms",
        basis_rank=9,
        basis_rms_m=3e-6,
        init_coeff_norm=0.2,
    )
    generator = torch.Generator().manual_seed(7)

    initialize_doe_height_(doe, target_rms_m=1.5e-6, generator=generator)
    torch.testing.assert_close(
        doe.pupil_rms(doe.heightmap()),
        torch.tensor(1.5e-6),
        atol=1e-11,
        rtol=1e-5,
    )

    projected = doe.project_height_rms_(0.75e-6)
    assert projected
    torch.testing.assert_close(
        doe.pupil_rms(doe.heightmap()),
        torch.tensor(0.75e-6),
        atol=1e-11,
        rtol=1e-5,
    )
    assert not doe.project_height_rms_(0.75e-6)


def test_preoptimization_objective_is_finite_and_differentiable():
    torch.manual_seed(11)
    logits = torch.randn(3, 25, 33, 33, requires_grad=True)
    psf = torch.softmax(logits.flatten(start_dim=-2), dim=-1).reshape_as(logits)
    response = torch.rand(3, 25)
    weights = DOEPreoptimizationWeights(
        mtf=2.0,
        optical_spectral_separation=0.7,
        optical_depth_separation=0.4,
        sensor_spectral_separation=0.2,
        sensor_depth_separation=0.1,
        energy_guard=0.1,
    )
    targets = DOEPreoptimizationTargets(
        energy_radii=(4.0, 8.0),
        energy_outside_budgets=(0.8, 0.6),
    )

    loss, metrics = doe_preoptimization_objective(
        psf,
        response,
        energy_reference="crop",
        weights=weights,
        targets=targets,
        separation_scale=0.5,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.norm() > 0
    assert metrics["weighted/optical_spectral_separation"] == (
        0.5
        * weights.optical_spectral_separation
        * metrics["loss/optical_spectral_separation"]
    )
    assert metrics["weighted/optical_depth_separation"] == (
        0.5
        * weights.optical_depth_separation
        * metrics["loss/optical_depth_separation"]
    )
    assert metrics["loss/full_total"] > metrics["loss/train_total"]


def test_optical_shape_loss_is_scale_invariant_and_rewards_shape_change():
    identical = torch.ones(2, 5, 4, 4, requires_grad=True)
    scale = torch.linspace(0.5, 2.5, 5).reshape(1, 5, 1, 1)
    scaled_loss, _ = optical_psf_shape_separation_loss(
        identical * scale,
        dimension="spectral",
        margin=0.9,
        offsets=(1, 2),
    )
    reference_loss, reference_stats = optical_psf_shape_separation_loss(
        identical,
        dimension="spectral",
        margin=0.9,
        offsets=(1, 2),
    )

    distinct = torch.zeros(2, 5, 4, 4)
    for wavelength in range(5):
        distinct[:, wavelength, wavelength // 4, wavelength % 4] = 1.0
    distinct_loss, distinct_stats = optical_psf_shape_separation_loss(
        distinct,
        dimension="spectral",
        margin=0.9,
        offsets=(1, 2),
    )

    torch.testing.assert_close(scaled_loss, reference_loss)
    assert distinct_loss < reference_loss
    assert distinct_stats["cosine_mean"] < reference_stats["cosine_mean"]
    reference_loss.backward()
    assert identical.grad is not None
    assert torch.isfinite(identical.grad).all()


def test_rms_constrained_adam_removes_outward_boundary_step():
    doe = DOELayer(
        doe_type="New",
        trainable=True,
        basis_mode="orthogonal_rms",
        basis_rank=9,
        basis_rms_m=3e-6,
        init_coeff_norm=0.2,
    )
    generator = torch.Generator().manual_seed(31)
    maximum_rms = 0.75e-6
    initialize_doe_height_(doe, target_rms_m=maximum_rms, generator=generator)
    optimizer = torch.optim.Adam([doe.zernike_coeffs], lr=1e-2)

    # Minimizing negative RMS proposes a purely outward update.
    loss = -doe.pupil_rms(doe.heightmap())
    loss.backward()
    stats = rms_constrained_optimizer_step_(
        doe,
        optimizer,
        maximum_rms_m=maximum_rms,
    )

    assert stats["tangent_correction"] == 1.0
    assert stats["rms_after_m"] <= maximum_rms * (1.0 + 1e-5)
    assert stats["retraction_scale"] > 0.995


def test_fisher_a_optimality_preserves_signal_strength_and_gradients():
    torch.manual_seed(23)
    psf = torch.rand(3, 4, 9, 9, requires_grad=True)
    response = torch.rand(3, 4)

    loss, stats = psf_fisher_a_optimality_loss(
        psf, response, ridge=1e-6, loss_scale=1e-6
    )
    stronger_loss, _ = psf_fisher_a_optimality_loss(
        2.0 * psf, response, ridge=1e-6, loss_scale=1e-6
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert stronger_loss < loss
    assert stats["minimum_eigenvalue_mean"] >= 0
    assert psf.grad is not None
    assert torch.isfinite(psf.grad).all()
    assert psf.grad.norm() > 0


def test_fisher_task_weights_keep_full_nuisance_inverse():
    torch.manual_seed(29)
    psf = torch.rand(3, 4, 7, 7)
    response = torch.rand(3, 4)

    full_loss, full_stats = psf_fisher_a_optimality_loss(
        psf,
        response,
        ridge=1e-6,
        loss_scale=1e-6,
        parameter_weights=(1.0, 1.0, 1.0, 1.0),
    )
    task_loss, task_stats = psf_fisher_a_optimality_loss(
        psf,
        response,
        ridge=1e-6,
        loss_scale=1e-6,
        parameter_weights=(0.0, 0.0, 1.0, 1.0),
    )

    torch.testing.assert_close(
        full_stats["a_optimality_mean"], task_stats["a_optimality_mean"]
    )
    torch.testing.assert_close(full_loss, 1e-6 * full_stats["a_optimality_mean"])
    torch.testing.assert_close(task_loss, 2e-6 * task_stats["task_a_optimality_mean"])
    assert task_stats["crlb_depth_mean"] > 0
    assert task_stats["crlb_wavelength_mean"] > 0


def test_rank9_cpu_smoke_writes_reusable_best_doe(tmp_path):
    output_dir = tmp_path / "preopt"
    result = main(
        [
            "--output_dir",
            str(output_dir),
            "--modes",
            "rank9",
            "--seeds",
            "17",
            "--device",
            "cpu",
            "--steps",
            "1",
            "--depth_layers",
            "2",
            "--separation_warmup_steps",
            "0",
            "--log_every",
            "1",
        ]
    )

    run_dir = output_dir / "rank9" / "seed_17"
    assert result == 0
    assert (output_dir / "comparison.json").is_file()
    assert (run_dir / "best_doe.pt").is_file()
    assert (run_dir / "best_coefficients.npy").is_file()
    assert (run_dir / "best_heightmap_m.npy").is_file()
    assert (run_dir / "best_psf_montage.png").is_file()
    with (run_dir / "summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    assert summary["mode"] == "rank9"
    assert summary["best_step"] in (0, 1)
    checkpoint = torch.load(run_dir / "best_doe.pt", map_location="cpu")
    assert checkpoint["format"] == "doe_psf_preoptimization_v1"
    assert checkpoint["doe_coefficients"].shape == (9,)
    assert checkpoint["doe_heightmap_m"].shape == (128, 128)
    restored = DOELayer(
        doe_type="New",
        trainable=True,
        basis_mode="orthogonal_rms",
        basis_rank=9,
        basis_rms_m=3e-6,
        init_coeff_norm=0.2,
    )
    load_preoptimized_doe_(restored, run_dir / "best_doe.pt")
    torch.testing.assert_close(
        restored.zernike_coeffs.detach(), checkpoint["doe_coefficients"]
    )
