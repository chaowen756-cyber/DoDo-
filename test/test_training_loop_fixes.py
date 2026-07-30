from argparse import ArgumentParser
from types import SimpleNamespace

import pytest
import torch

from snapshotdepth_hs import (
    SnapshotDepthHS,
    _VALIDATION_TOTAL_KEYS,
    _all_reduce_validation_totals,
)
from snapshotdepth_trainer_hs import (
    _checkpoint_psf_optics_version,
    _configure_trainer_callbacks,
    _ensure_checkpoint_psf_optics_version,
)
from torch_optics.doe import DOELayer


def _optimizer_hparams(*, optimize_optics):
    return SimpleNamespace(
        optimize_optics=optimize_optics,
        lr_decay_strategy="none",
        lr_warmup_steps=0,
        cnn_lr=1e-3,
        optics_lr=2.0,
        cnn_lr_decay_epochs=20,
        optics_lr_decay_epochs=10,
        loss_plot_every_n_steps=1,
    )


def test_adam_optimizer_step_does_not_execute_lightning_closure():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.Adam(
        [{"params": [parameter], "lr": 1e-3, "name": "cnn"}]
    )
    parameter.square().backward()
    closure_calls = []

    def closure():
        closure_calls.append(1)
        parameter.square().backward()
        return parameter.square()

    dummy = SimpleNamespace(
        hparams=_optimizer_hparams(optimize_optics=False),
        trainer=SimpleNamespace(current_epoch=0, global_step=0),
        optical_model_type="dodo_depth",
    )
    SnapshotDepthHS.optimizer_step(
        dummy,
        epoch=0,
        batch_idx=0,
        optimizer=optimizer,
        optimizer_closure=closure,
    )

    assert closure_calls == []


class _ToyDOE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.zernike_coeffs = torch.nn.Parameter(
            torch.tensor([0.2, 0.0], dtype=torch.float32))
        self.coeff_norm_limit = 1.0
        self.clamp_calls = 0

    def clamp_parameters_(self):
        self.clamp_calls += 1
        with torch.no_grad():
            norm = torch.linalg.vector_norm(self.zernike_coeffs)
            if norm > self.coeff_norm_limit:
                self.zernike_coeffs.mul_(self.coeff_norm_limit / norm)


class _ToyCamera(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.doe1 = _ToyDOE()

    def clamp_parameters_(self):
        self.doe1.clamp_parameters_()


def test_optimizer_step_records_raw_and_projected_doe_updates_once():
    camera = _ToyCamera()
    optimizer = torch.optim.Adam(
        [{
            "params": camera.parameters(),
            "lr": 2.0,
            "name": "optics",
        }]
    )
    (-camera.doe1.zernike_coeffs.sum()).backward()
    logged = {}
    dummy = SimpleNamespace(
        hparams=_optimizer_hparams(optimize_optics=True),
        trainer=SimpleNamespace(current_epoch=0, global_step=0),
        optical_model_type="dodo_depth",
        camera=camera,
        _clamp_hook_count=0,
        log=lambda name, value, **kwargs: logged.__setitem__(name, value),
    )

    SnapshotDepthHS.optimizer_step(
        dummy,
        epoch=0,
        batch_idx=0,
        optimizer=optimizer,
    )

    metrics = dummy._last_doe_metrics
    assert camera.doe1.clamp_calls == 1
    assert metrics["coeff_norm_raw"] > 1.0
    assert metrics["coeff_norm"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["clamp_hit"] == 1.0
    assert metrics["projection_correction_norm"] > 0.0
    assert metrics["raw_update_norm"] > metrics["effective_update_norm"]
    assert 0.0 < metrics["update_retention"] < 1.0
    assert "doe/update_retention" in logged


def test_validation_totals_use_float64_ddp_sum(monkeypatch):
    totals = {
        key: float(index + 1)
        for index, key in enumerate(_VALIDATION_TOTAL_KEYS)
    }
    remote = torch.arange(
        101,
        101 + len(_VALIDATION_TOTAL_KEYS),
        dtype=torch.float64,
    )
    calls = []

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def fake_all_reduce(values, op):
        calls.append((values.dtype, op))
        values.add_(remote.to(device=values.device))

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    reduced = _all_reduce_validation_totals(totals, torch.device("cpu"))

    assert calls == [(torch.float64, torch.distributed.ReduceOp.SUM)]
    for index, key in enumerate(_VALIDATION_TOTAL_KEYS):
        assert reduced[key] == totals[key] + remote[index].item()


def test_validation_epoch_metrics_are_computed_from_global_raw_totals():
    totals = {key: 0.0 for key in _VALIDATION_TOTAL_KEYS}
    totals.update({
        "depth_abs_sum": 2.0,
        "depth_sq_sum": 1.0,
        "depth_valid_count": 4.0,
        "metric_depth_abs_sum": 4.0,
        "metric_depth_batch_mae_sum": 2.0,
        "hs_abs_sum": 4.0,
        "hs_sq_sum": 2.0,
        "hs_valid_count": 8.0,
        "hs_full_sq_sum": 2.0,
        "hs_full_count": 8.0,
        "depth_tv_dx_sum": 1.0,
        "depth_tv_dx_count": 2.0,
        "depth_tv_dy_sum": 3.0,
        "depth_tv_dy_count": 6.0,
        "background_hs_abs_sum": 1.0,
        "background_hs_count": 4.0,
        "valid_batches": 2.0,
    })
    logged = {}
    dummy = SimpleNamespace(
        _val_totals=totals,
        device=torch.device("cpu"),
        hparams=SimpleNamespace(
            image_loss_weight=1.0,
            depth_loss_weight=0.1,
            depth_smooth_weight=0.2,
            background_hs_loss_weight=0.4,
        ),
        log=lambda name, value, **kwargs: logged.__setitem__(
            name, float(value)),
        _nonfinite_count=0,
        artifact_root=None,
        log_dir=None,
        _trainer_is_global_zero=lambda: True,
    )

    SnapshotDepthHS.validation_epoch_end(dummy, outputs=[])

    # HS L1=.5, depth MAE=.5, TV=.5, background L1=.25.
    expected_val_loss = 0.5 + 0.1 * 0.5 + 0.2 * 0.5 + 0.4 * 0.25
    assert logged["val_loss"] == pytest.approx(expected_val_loss)
    assert logged["validation/mae_depth_m"] == pytest.approx(1.0)
    assert logged["validation/hs_l1_masked"] == pytest.approx(0.5)


def test_non_global_rank_does_not_write_validation_artifacts(tmp_path):
    output_dir = tmp_path / "rank-one-artifacts"
    dummy = SimpleNamespace(
        _trainer_is_global_zero=lambda: False,
        artifact_root=str(output_dir),
        log_dir=None,
    )

    SnapshotDepthHS._save_validation_artifacts(
        dummy, extra={"val_loss": 1.0}, out_dir=str(output_dir))

    assert not output_dir.exists()


def test_orthogonal_rms_default_initializes_inside_constraint_ball():
    torch.manual_seed(123)
    doe = DOELayer(
        doe_type="New",
        trainable=True,
        basis_mode="orthogonal_rms",
    )

    torch.testing.assert_close(
        doe.zernike_coeffs.norm(),
        torch.tensor(0.2),
        atol=1e-6,
        rtol=0,
    )
    assert doe.coeff_norm_limit == 1.0


def test_training_cli_defaults_orthogonal_rms_init_norm_to_point_two():
    parser = SnapshotDepthHS.add_model_specific_args(ArgumentParser())
    args = parser.parse_args([])

    assert args.dodo_doe_init_coeff_norm == pytest.approx(0.2)
    assert args.dodo_doe_coeff_norm_limit == pytest.approx(1.0)
    assert args.dodo_psf_optics_version == "legacy"
    assert args.checkpoint_monitor == "val_loss"
    assert args.checkpoint_mode == "min"


def test_checkpoint_psf_version_guard_preserves_legacy_and_blocks_mismatch():
    historical = {"state_dict": {"decoder.weight": torch.ones(1)}}
    consistent = {
        "state_dict": {"decoder.weight": torch.ones(1)},
        "hyper_parameters": {
            "dodo_psf_optics_version": "consistent_grid_v1",
        },
    }
    raw_state_dict = {"decoder.weight": torch.ones(1)}

    assert _checkpoint_psf_optics_version(historical) == "legacy"
    assert (
        _ensure_checkpoint_psf_optics_version(
            consistent, "consistent_grid_v1")
        == "consistent_grid_v1"
    )
    for nested_metadata in (
        {"hparams": {
            "dodo_psf_optics_version": "consistent_grid_v1",
        }},
        {"hparams": SimpleNamespace(
            dodo_psf_optics_version="consistent_grid_v1",
        )},
    ):
        assert _checkpoint_psf_optics_version({
            "state_dict": {},
            "hyper_parameters": nested_metadata,
        }) == "consistent_grid_v1"
    with pytest.raises(ValueError, match="version mismatch"):
        _ensure_checkpoint_psf_optics_version(
            historical, "consistent_grid_v1")
    with pytest.raises(ValueError, match="version mismatch"):
        _ensure_checkpoint_psf_optics_version(consistent, "legacy")
    with pytest.raises(ValueError, match="Cannot determine"):
        _ensure_checkpoint_psf_optics_version(
            raw_state_dict, "consistent_grid_v1")
    assert _ensure_checkpoint_psf_optics_version(
        raw_state_dict,
        "consistent_grid_v1",
        allow_mismatch=True,
    ) is None


def test_explicit_checkpoint_callbacks_disable_lightning_default(tmp_path):
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint

    def make_checkpoint(name, monitor):
        try:
            return ModelCheckpoint(
                monitor=monitor,
                dirpath=str(tmp_path),
                filename=f"{name}-{{epoch:03d}}",
                save_top_k=1,
                mode="min",
            )
        except TypeError:
            return ModelCheckpoint(
                monitor=monitor,
                filepath=str(tmp_path / f"{name}-{{epoch:03d}}"),
                save_top_k=1,
                mode="min",
            )

    joint = make_checkpoint("joint-best", "val_loss")
    depth = make_checkpoint("depth-best", "validation/mae_depth_m")
    kwargs = {
        "logger": False,
        "max_epochs": 1,
        "progress_bar_refresh_rate": 0,
    }
    _configure_trainer_callbacks(kwargs, [joint, depth])
    trainer = Trainer(**kwargs)

    checkpoint_callbacks = [
        callback
        for callback in trainer.callbacks
        if isinstance(callback, ModelCheckpoint)
    ]
    assert checkpoint_callbacks == [joint, depth]
