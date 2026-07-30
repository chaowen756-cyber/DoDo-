from types import SimpleNamespace

import pytest
import torch

from snapshotdepth_hs import SnapshotDepthHS
from util.psf_regularization import (
    delayed_epoch_tightening_value,
    delayed_epoch_warmup_weight,
    epoch_tightening_value,
    epoch_warmup_weight,
    multiscale_psf_energy_concentration_loss,
    psf_mtf_floor_loss,
    psf_energy_concentration_loss,
    sensor_weighted_depth_psf_separation_loss,
    sensor_weighted_spectral_psf_separation_loss,
    task_relative_regularizer_scale,
)


def test_epoch_warmup_matches_final_training_schedule():
    assert epoch_warmup_weight(0.02, current_epoch=0, warmup_epochs=2) == 0.0
    assert epoch_warmup_weight(0.02, current_epoch=1, warmup_epochs=2) == 0.01
    assert epoch_warmup_weight(0.02, current_epoch=2, warmup_epochs=2) == 0.02
    assert epoch_warmup_weight(0.02, current_epoch=20, warmup_epochs=2) == 0.02


def test_epoch_constraint_budget_tightens_without_zero_weight_window():
    assert epoch_tightening_value(0.35, 0.20, 0, 3) == pytest.approx(0.35)
    assert epoch_tightening_value(0.35, 0.20, 1, 3) == pytest.approx(0.30)
    assert epoch_tightening_value(0.35, 0.20, 3, 3) == pytest.approx(0.20)
    assert epoch_tightening_value(0.35, 0.20, 20, 3) == pytest.approx(0.20)


def test_delayed_regularizer_schedule_stabilizes_task_before_ramp():
    weights = [
        delayed_epoch_warmup_weight(0.02, epoch, start_epoch=5, warmup_epochs=5)
        for epoch in (0, 4, 5, 7, 9, 19)
    ]
    assert weights == pytest.approx([0.0, 0.0, 0.0, 0.008, 0.016, 0.02])

    budgets = [
        delayed_epoch_tightening_value(
            0.70, 0.60, epoch, start_epoch=5, tightening_epochs=14)
        for epoch in (0, 5, 12, 19)
    ]
    assert budgets == pytest.approx([0.70, 0.70, 0.65, 0.60])


def test_task_relative_regularizer_cap_is_detached_and_respected():
    regularizer = torch.tensor(0.05, requires_grad=True)
    task = torch.tensor(0.02, requires_grad=True)
    scale = task_relative_regularizer_scale(
        regularizer, task, max_ratio=0.15)
    capped = scale * regularizer

    assert scale.item() == pytest.approx(0.06)
    assert capped.item() == pytest.approx(0.003)
    assert capped.item() / task.item() == pytest.approx(0.15)
    assert not scale.requires_grad

    capped.backward()
    assert regularizer.grad.item() == pytest.approx(0.06)
    assert task.grad is None


@pytest.mark.parametrize(
    ("current_epoch", "expected_weight"),
    [(0, 0.0), (1, 0.01), (2, 0.02), (20, 0.02)],
)
def test_snapshot_model_reads_public_lightning_epoch_for_psf_warmup(
    current_epoch,
    expected_weight,
):
    model = SimpleNamespace(
        hparams=SimpleNamespace(
            dodo_psf_energy_weight=0.02,
            dodo_psf_energy_start_epoch=0,
            dodo_psf_energy_warmup_epochs=2,
            optimize_optics=True,
        ),
        optical_model_type="dodo_depth",
        camera=SimpleNamespace(image_formation_mode="psf_convolution"),
        current_epoch=current_epoch,
    )

    actual = SnapshotDepthHS._dodo_psf_energy_weight(model)

    assert actual == expected_weight


def test_snapshot_model_disables_psf_energy_loss_when_optics_are_frozen():
    model = SimpleNamespace(
        hparams=SimpleNamespace(
            dodo_psf_energy_weight=0.02,
            dodo_psf_energy_warmup_epochs=2,
            optimize_optics=False,
        ),
        optical_model_type="dodo_depth",
        camera=SimpleNamespace(image_formation_mode="psf_convolution"),
        current_epoch=20,
    )

    assert SnapshotDepthHS._dodo_psf_energy_weight(model) == 0.0


def test_centered_psf_satisfies_radius_16_budget():
    psf = torch.zeros((2, 3, 128, 128))
    psf[..., 64, 64] = 1.0

    loss, stats = psf_energy_concentration_loss(
        psf, radius=16.0, outside_budget=0.5, softness=1.5)

    torch.testing.assert_close(loss, torch.tensor(0.0), atol=0, rtol=0)
    assert stats["inside_mean"].item() > 0.999
    assert stats["active_fraction"].item() == 0.0


def test_outside_psf_has_expected_squared_hinge_penalty():
    psf = torch.zeros((1, 1, 128, 128))
    psf[..., 64, 100] = 1.0

    loss, stats = psf_energy_concentration_loss(
        psf, radius=16.0, outside_budget=0.5, softness=0.0)

    torch.testing.assert_close(loss, torch.tensor(0.25), atol=0, rtol=0)
    torch.testing.assert_close(stats["outside_mean"], torch.tensor(1.0), atol=0, rtol=0)
    assert stats["active_fraction"].item() == 1.0


def test_loss_uses_fractional_energy_and_mean_over_psfs():
    psf = torch.zeros((1, 2, 128, 128))
    psf[0, 0, 64, 64] = 7.0
    psf[0, 1, 64, 100] = 19.0

    loss, _ = psf_energy_concentration_loss(
        psf, radius=16.0, outside_budget=0.5, softness=0.0)

    # One kernel has zero loss and the other has (1 - 0.5)^2; average over two.
    torch.testing.assert_close(loss, torch.tensor(0.125), atol=0, rtol=0)


def test_loss_backpropagates_finite_nonzero_gradient():
    logits = torch.zeros((1, 1, 128 * 128), requires_grad=True)
    with torch.no_grad():
        logits[..., 64 * 128 + 100] = 5.0
    psf = torch.softmax(logits, dim=-1).reshape(1, 1, 128, 128)

    loss, _ = psf_energy_concentration_loss(
        psf, radius=16.0, outside_budget=0.5, softness=1.5)
    loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.norm().item() > 0


def test_full_field_energy_reference_counts_missing_crop_energy():
    captured = torch.tensor(0.5, requires_grad=True)
    psf = torch.zeros((1, 1, 129, 129))
    psf[..., 64, 64] = captured

    crop_loss, _ = multiscale_psf_energy_concentration_loss(
        psf,
        radii=(16.0,),
        outside_budgets=(0.0,),
        scale_weights=(1.0,),
        softness=0.0,
        cvar_weight=0.0,
        energy_reference="crop",
    )
    full_loss, stats = multiscale_psf_energy_concentration_loss(
        psf,
        radii=(16.0,),
        outside_budgets=(0.0,),
        scale_weights=(1.0,),
        softness=0.0,
        cvar_weight=0.0,
        energy_reference="full_field",
    )
    full_loss.backward()

    torch.testing.assert_close(crop_loss, torch.tensor(0.0))
    torch.testing.assert_close(full_loss, torch.tensor(0.25))
    torch.testing.assert_close(stats["outside_mean"], torch.tensor(0.5))
    torch.testing.assert_close(stats["captured_mean"], torch.tensor(0.5))
    torch.testing.assert_close(stats["missing_mean"], torch.tensor(0.5))
    assert captured.grad.item() == pytest.approx(-1.0)


def test_full_field_energy_gradient_pushes_energy_outside_crop_inward():
    logits = torch.zeros((1, 1, 17, 17), requires_grad=True)
    full_psf = torch.softmax(logits.flatten(-2), dim=-1).reshape_as(logits)
    cropped_psf = full_psf[..., 4:13, 4:13]

    loss, _ = multiscale_psf_energy_concentration_loss(
        cropped_psf,
        radii=(2.0,),
        outside_budgets=(0.0,),
        scale_weights=(1.0,),
        softness=0.0,
        cvar_weight=0.0,
        energy_reference="full_field",
    )
    loss.backward()

    assert logits.grad[..., 8, 8].item() < 0.0
    assert logits.grad[..., 0, 0].item() > 0.0
    assert torch.isfinite(logits.grad).all()


def test_full_field_energy_uses_odd_129_center_and_checks_radius():
    centered = torch.zeros((1, 1, 129, 129))
    centered[..., 64, 64] = 1.0
    boundary = torch.zeros_like(centered)
    boundary[..., 64, 80] = 1.0
    outside = torch.zeros_like(centered)
    outside[..., 64, 81] = 1.0

    def energy(psf):
        _, stats = psf_energy_concentration_loss(
            psf,
            radius=16.0,
            outside_budget=1.0,
            softness=0.0,
            energy_reference="full_field",
        )
        return stats["outside_mean"]

    torch.testing.assert_close(energy(centered), torch.tensor(0.0))
    torch.testing.assert_close(energy(boundary), torch.tensor(0.0))
    torch.testing.assert_close(energy(outside), torch.tensor(1.0))
    with pytest.raises(ValueError, match="circle must lie inside"):
        psf_energy_concentration_loss(
            centered,
            radius=65.0,
            energy_reference="full_field",
        )


def test_multiscale_energy_penalizes_core_and_tail_and_reports_radii():
    psf = torch.zeros((1, 2, 64, 64), requires_grad=True)
    with torch.no_grad():
        psf[0, 0, 32, 32] = 1.0
        psf[0, 1, 32, 60] = 1.0
    loss, stats = multiscale_psf_energy_concentration_loss(
        psf,
        radii=(8.0, 16.0),
        outside_budgets=(0.2, 0.05),
        softness=0.0,
        cvar_fraction=0.5,
        cvar_weight=0.5,
    )
    loss.backward()
    assert loss.item() > 0
    assert stats['r8_inside_mean'].item() == pytest.approx(0.5)
    assert stats['r90_max'].item() >= 28.0
    assert psf.grad is not None


def test_linear_energy_hinge_keeps_stronger_gradient_near_budget():
    logits_linear = torch.zeros((1, 1, 16 * 16), requires_grad=True)
    logits_squared = logits_linear.detach().clone().requires_grad_(True)
    with torch.no_grad():
        logits_linear[..., 0] = 0.2
        logits_squared[..., 0] = 0.2

    def compute(logits, penalty_power):
        psf = torch.softmax(logits, dim=-1).reshape(1, 1, 16, 16)
        loss, _ = multiscale_psf_energy_concentration_loss(
            psf,
            radii=(3.0,),
            outside_budgets=(0.80,),
            scale_weights=(1.0,),
            softness=0.0,
            cvar_weight=0.0,
            penalty_power=penalty_power,
        )
        loss.backward()
        return loss, logits.grad.norm()

    linear_loss, linear_grad = compute(logits_linear, 1.0)
    squared_loss, squared_grad = compute(logits_squared, 2.0)
    assert linear_loss.item() > 0
    assert squared_loss.item() > 0
    assert linear_grad.item() > squared_grad.item()


def test_mtf_floor_distinguishes_delta_from_broad_psf_and_backpropagates():
    delta = torch.zeros((1, 1, 64, 64))
    delta[..., 32, 32] = 1.0
    broad_logits = torch.zeros((1, 1, 64, 64), requires_grad=True)
    broad = torch.softmax(broad_logits.flatten(-2), dim=-1).reshape_as(broad_logits)
    delta_loss, delta_stats = psf_mtf_floor_loss(delta)
    broad_loss, broad_stats = psf_mtf_floor_loss(broad)
    broad_loss.backward()
    assert delta_loss.item() == pytest.approx(0.0, abs=1e-8)
    assert broad_loss.item() > delta_loss.item()
    assert delta_stats['mtf_005_mean'].item() > broad_stats['mtf_005_mean'].item()
    assert broad_logits.grad is not None
    assert torch.isfinite(broad_logits.grad).all()


def test_spectral_offsets_and_depth_hard_negatives_are_differentiable():
    psf = torch.rand((3, 4, 16, 16), requires_grad=True)
    response = torch.rand((3, 4))
    spectral_loss, spectral_stats = (
        sensor_weighted_spectral_psf_separation_loss(
            psf, response, margin=0.90, offsets=(1, 2),
            hard_fraction=0.2, hard_weight=0.5))
    depth_loss, depth_stats = sensor_weighted_depth_psf_separation_loss(
        psf, response, margin=0.90, hard_fraction=0.2, hard_weight=0.5)
    (spectral_loss + depth_loss).backward()
    assert psf.grad is not None
    assert torch.isfinite(psf.grad).all()
    assert 0.0 <= spectral_stats['active_fraction'].item() <= 1.0
    assert depth_stats['adjacent_cosine_p90'].item() <= 1.0 + 1e-6


def test_depth_separation_full_field_preserves_relative_spectral_capture():
    psf = torch.zeros((2, 2, 3, 3))
    psf[0, 0, 1, 1] = 0.9
    psf[0, 1, 1, 1] = 0.1
    psf[1, 0, 1, 1] = 0.1
    psf[1, 1, 1, 1] = 0.9
    response = torch.eye(2)

    crop_loss, crop_stats = sensor_weighted_depth_psf_separation_loss(
        psf,
        response,
        margin=0.5,
        hard_weight=0.0,
        energy_reference="crop",
    )
    full_loss, full_stats = sensor_weighted_depth_psf_separation_loss(
        psf,
        response,
        margin=0.5,
        hard_weight=0.0,
        energy_reference="full_field",
    )

    assert crop_stats["adjacent_cosine_mean"].item() == pytest.approx(1.0)
    assert crop_loss.item() == pytest.approx(0.5)
    assert full_stats["adjacent_cosine_mean"].item() == pytest.approx(
        0.2195122, abs=1e-6)
    assert full_loss.item() == pytest.approx(0.0)


def test_mtf_is_invariant_to_full_field_capture_scale():
    torch.manual_seed(41)
    psf = torch.rand((1, 2, 129, 129))

    full_loss, full_stats = psf_mtf_floor_loss(psf)
    scaled_loss, scaled_stats = psf_mtf_floor_loss(psf * 0.25)

    torch.testing.assert_close(full_loss, scaled_loss, atol=1e-7, rtol=1e-6)
    for key in ("mtf_005_mean", "mtf_010_mean", "mtf_020_mean"):
        torch.testing.assert_close(
            full_stats[key],
            scaled_stats[key],
            atol=1e-7,
            rtol=1e-6,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"radius": 0.0},
        {"outside_budget": -0.1},
        {"outside_budget": 1.1},
        {"softness": -1.0},
    ],
)
def test_invalid_regularizer_configuration_fails_fast(kwargs):
    with pytest.raises(ValueError):
        psf_energy_concentration_loss(torch.ones((1, 1, 8, 8)), **kwargs)
