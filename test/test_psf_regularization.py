from types import SimpleNamespace

import pytest
import torch

from snapshotdepth_hs import SnapshotDepthHS
from util.psf_regularization import (
    epoch_warmup_weight,
    psf_energy_concentration_loss,
)


def test_epoch_warmup_matches_final_training_schedule():
    assert epoch_warmup_weight(0.02, current_epoch=0, warmup_epochs=2) == 0.0
    assert epoch_warmup_weight(0.02, current_epoch=1, warmup_epochs=2) == 0.01
    assert epoch_warmup_weight(0.02, current_epoch=2, warmup_epochs=2) == 0.02
    assert epoch_warmup_weight(0.02, current_epoch=20, warmup_epochs=2) == 0.02


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
