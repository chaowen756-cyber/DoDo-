from types import MethodType, SimpleNamespace

import torch
import torch.nn.functional as F

from torch_optics.forward_dodo import DepthAwareDoDoForwardModel
from util.psf_regularization import multiscale_psf_energy_concentration_loss


def _make_consistent_model(
    *,
    num_depth_layers=1,
    sensing_mode="identity",
    measurement_channels=25,
    psf_depth_chunk_size=1,
    doe_type_a="Zeros",
    train_c=False,
    doe_basis_mode="legacy_raw12",
    doe_init_coeff_norm=1.0,
):
    return DepthAwareDoDoForwardModel(
        depth_min=0.4,
        depth_max=2.0,
        num_depth_layers=num_depth_layers,
        use_second_doe=False,
        doe_type_a=doe_type_a,
        train_c=train_c,
        input_format="nchw",
        output_format="nchw",
        assets_dir="torch_optics/assets",
        measurement_norm_mode="none",
        sensing_mode=sensing_mode,
        measurement_channels=measurement_channels,
        depth_layering_mode="soft_diopter",
        sensor_measurement="intensity",
        skip_prop2=True,
        prop1_padding_factor=1,
        image_formation_mode="psf_convolution",
        psf_layer_mask_mode="baek_hard",
        psf_mask_blur_sigma=0.0,
        psf_boundary_mode="linear_zero",
        psf_depth_chunk_size=psf_depth_chunk_size,
        psf_optics_version="consistent_grid_v1",
        doe_basis_mode=doe_basis_mode,
        doe_init_coeff_norm=doe_init_coeff_norm,
    )


def test_lightning_adam_batch_executes_one_forward_backward_and_step():
    # This integration test deliberately goes through Lightning 1.0.2's
    # training loop.  That version completes training_step + backward before
    # invoking optimizer_step, so forwarding its second-order closure to Adam
    # would double both counters for every batch.
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader, TensorDataset

    from snapshotdepth_hs import SnapshotDepthHS

    class CountingAdam(torch.optim.Adam):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.step_calls = 0

        def step(self, *args, **kwargs):
            self.step_calls += 1
            return super().step(*args, **kwargs)

    class CountingModule(pl.LightningModule):
        optimizer_step = SnapshotDepthHS.optimizer_step

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.5))
            self.forward_calls = 0
            self.backward_calls = 0
            self.optimizer_instance = None
            self.weight.register_hook(self._count_backward)
            self.hparams.lr_decay_strategy = "none"
            self.hparams.lr_warmup_steps = 0
            self.hparams.optics_lr = 1e-4
            self.hparams.cnn_lr = 1e-2
            self.hparams.optimize_optics = False
            self.hparams.loss_plot_every_n_steps = 50
            self.optical_model_type = None

        def _count_backward(self, gradient):
            self.backward_calls += 1
            return gradient

        def training_step(self, batch, batch_idx):
            del batch_idx
            self.forward_calls += 1
            input_value, target = batch
            return (self.weight * input_value - target).square().mean()

        def configure_optimizers(self):
            self.optimizer_instance = CountingAdam(
                [{
                    "params": [self.weight],
                    "lr": self.hparams.cnn_lr,
                    "name": "cnn",
                }]
            )
            return self.optimizer_instance

    dataset = TensorDataset(
        torch.tensor([[1.0], [2.0]]),
        torch.tensor([[2.0], [4.0]]),
    )
    model = CountingModule()
    trainer = pl.Trainer(
        max_epochs=1,
        limit_train_batches=2,
        logger=False,
        checkpoint_callback=False,
        progress_bar_refresh_rate=0,
        weights_summary=None,
        num_sanity_val_steps=0,
    )
    trainer.fit(
        model,
        train_dataloader=DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        ),
    )

    assert model.forward_calls == 2
    assert model.backward_calls == 2
    assert model.optimizer_instance.step_calls == 2
    assert trainer.global_step == 2


def test_consistent_grid_has_one_pitch_across_prop1_doe_and_prop3():
    model = _make_consistent_model()

    model._assert_consistent_optical_sampling()
    prop1 = model.prop1_layers[0]
    prop3 = model.prop3
    prop1_dx = prop1.L / prop1.Mp
    prop3_dx = prop3.L / prop3.Mp
    prop3_work_dx = prop3.work_L / prop3.work_Mp

    assert model.psf_optics_version == "consistent_grid_v1"
    assert model.psf_energy_reference == "full_field"
    assert prop1.Mp == model.doe1.Mesce == model.doe1.Mdoe == prop3.Mp == 128
    assert prop1_dx == prop3_dx == prop3_work_dx
    assert prop3.padding_factor == 2
    assert prop3.work_Mp == 256
    assert prop3.work_L == 0.02
    assert model.doe1.use_pupil_mask


def test_consistent_grid_psf_normalizes_full_grid_then_captures_center_129():
    model = _make_consistent_model().eval()

    with torch.no_grad():
        actual = model.psf_bank(use_cache=False)

        prop1_fields = model._prop1_impulse_field_bank(
            128,
            128,
            model.z_centers.device,
        )
        field_after_doe = model.doe1(prop1_fields)
        full_sensor_field = model.prop3.forward_work_grid(field_after_doe)
        full_intensity = full_sensor_field.abs().to(torch.float32).square()
        full_normalized = full_intensity / full_intensity.sum(
            dim=(-2, -1),
            keepdim=True,
        ).clamp_min(1e-8)

        full_center_y = full_normalized.shape[-2] // 2
        full_center_x = full_normalized.shape[-1] // 2
        kernel_radius = 129 // 2
        expected = full_normalized[
            ...,
            full_center_y - kernel_radius:
            full_center_y + kernel_radius + 1,
            full_center_x - kernel_radius:
            full_center_x + kernel_radius + 1,
        ]
        expected_capture = expected.sum(dim=(-2, -1))

    assert actual.shape == (1, 25, 129, 129)
    torch.testing.assert_close(
        full_normalized.sum(dim=(-2, -1)),
        torch.ones((1, 25)),
        atol=2e-6,
        rtol=0,
    )
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-5)
    torch.testing.assert_close(
        actual.sum(dim=(-2, -1)),
        expected_capture,
        atol=2e-6,
        rtol=0,
    )
    torch.testing.assert_close(
        model.psf_capture_fraction,
        expected_capture,
        atol=2e-6,
        rtol=0,
    )
    assert torch.all(expected_capture >= 0.995)
    assert torch.all(expected_capture <= 1.0 + 2e-6)

    _, energy_stats = multiscale_psf_energy_concentration_loss(
        actual,
        radii=(16.0, 24.0),
        outside_budgets=(1.0, 1.0),
        scale_weights=(1.0, 1.0),
        softness=1.5,
        cvar_weight=0.0,
        energy_reference=model.psf_energy_reference,
    )
    full_y = torch.arange(256, dtype=full_normalized.dtype) - 128
    full_x = torch.arange(256, dtype=full_normalized.dtype) - 128
    full_yy, full_xx = torch.meshgrid(full_y, full_x, indexing="ij")
    full_radius = torch.sqrt(full_xx.square() + full_yy.square())
    direct_r16_outside = (
        full_normalized
        * torch.sigmoid((full_radius - 16.0) / 1.5)
    ).sum(dim=(-2, -1))
    direct_r24_outside = (
        full_normalized
        * torch.sigmoid((full_radius - 24.0) / 1.5)
    ).sum(dim=(-2, -1))
    torch.testing.assert_close(
        energy_stats["r16_outside_mean"],
        direct_r16_outside.mean(),
        atol=2e-6,
        rtol=2e-5,
    )
    torch.testing.assert_close(
        energy_stats["r24_outside_mean"],
        direct_r24_outside.mean(),
        atol=2e-6,
        rtol=2e-5,
    )


def test_consistent_grid_doe_gradient_matches_centered_finite_difference():
    torch.manual_seed(79)
    model = _make_consistent_model(
        doe_type_a="New",
        train_c=True,
        doe_basis_mode="orthogonal_rms",
        doe_init_coeff_norm=0.25,
    )
    psf = model.psf_bank(use_cache=False)
    probe = torch.randn_like(psf)
    probe = probe - probe.mean(dim=(-2, -1), keepdim=True)
    objective = (psf * probe).sum()
    gradient = torch.autograd.grad(
        objective,
        model.doe1.zernike_coeffs,
    )[0]
    coefficient_index = int(torch.argmax(gradient.abs()).item())
    autograd_derivative = gradient[coefficient_index].detach()

    epsilon = 3e-3
    coefficients = model.doe1.zernike_coeffs
    original_value = coefficients[coefficient_index].detach().clone()
    with torch.no_grad():
        coefficients[coefficient_index].copy_(original_value + epsilon)
        plus = (model.psf_bank(use_cache=False) * probe).sum()
        coefficients[coefficient_index].copy_(original_value - epsilon)
        minus = (model.psf_bank(use_cache=False) * probe).sum()
        coefficients[coefficient_index].copy_(original_value)
    finite_difference = (plus - minus) / (2.0 * epsilon)

    denominator = torch.maximum(
        autograd_derivative.abs(),
        finite_difference.abs(),
    ).clamp_min(1e-8)
    relative_error = (
        (autograd_derivative - finite_difference).abs() / denominator
    )
    assert torch.isfinite(gradient).all()
    assert autograd_derivative.abs() > 1e-6
    assert relative_error < 0.02


def test_consistent_grid_full_field_energy_loss_reaches_doe_parameters():
    torch.manual_seed(83)
    model = _make_consistent_model(
        doe_type_a="New",
        train_c=True,
        doe_basis_mode="orthogonal_rms",
        doe_init_coeff_norm=0.2,
    )
    psf = model.psf_bank(use_cache=False)
    loss, stats = multiscale_psf_energy_concentration_loss(
        psf,
        radii=(16.0, 24.0),
        outside_budgets=(0.0, 0.0),
        scale_weights=(1.0, 0.5),
        softness=1.5,
        cvar_weight=0.0,
        energy_reference=model.psf_energy_reference,
    )
    gradient = torch.autograd.grad(
        loss,
        model.doe1.zernike_coeffs,
    )[0]

    assert loss.item() > 0.0
    assert torch.isfinite(gradient).all()
    assert gradient.norm().item() > 0.0
    torch.testing.assert_close(
        stats["captured_mean"],
        psf.detach().sum(dim=(-2, -1)).mean(),
        atol=2e-7,
        rtol=0,
    )


def test_snapshot_total_loss_uses_full_field_psf_energy_and_reaches_doe():
    from snapshotdepth_hs import SnapshotDepthHS

    model = _make_consistent_model(
        doe_type_a="New",
        train_c=True,
        doe_basis_mode="orthogonal_rms",
        doe_init_coeff_norm=0.2,
    )
    psf = model.psf_bank(use_cache=False)
    zeros_hs = torch.zeros((1, 25, 4, 4))
    zeros_depth = torch.zeros((1, 4, 4))
    outputs = SimpleNamespace(
        est_images=zeros_hs,
        est_depthmaps=zeros_depth,
        psf=psf,
    )

    def image_loss(est_images, target_images, mask):
        zero = (est_images - target_images).sum() * 0.0
        return zero, {
            "l1": zero,
            "mse": zero,
            "sam": zero,
            "gradient": zero,
        }

    dummy = SimpleNamespace(
        training=False,
        global_step=1,
        current_epoch=1,
        optical_model_type="dodo_depth",
        camera=model,
        image_lossfn=image_loss,
        hparams=SimpleNamespace(
            depth_loss_weight=0.0,
            image_loss_weight=0.0,
            depth_smooth_weight=0.0,
            metric_depth_loss_weight=0.0,
            background_hs_loss_weight=0.0,
            psf_loss_weight=0.0,
            optimize_optics=True,
            dodo_psf_energy_weight=1.0,
            dodo_psf_energy_start_epoch=0,
            dodo_psf_energy_warmup_epochs=0,
            dodo_psf_energy_outside_budget=0.0,
            dodo_psf_energy_outer_outside_budget=0.0,
            dodo_psf_energy_initial_outside_budget=0.0,
            dodo_psf_energy_initial_outer_outside_budget=0.0,
            dodo_psf_energy_cvar_weight=0.0,
            dodo_psf_mtf_weight=0.0,
            dodo_zernike_mode="legacy12",
            dodo_optical_regularizer_max_ratio=0.0,
        ),
    )
    dummy._dodo_psf_energy_weight = MethodType(
        SnapshotDepthHS._dodo_psf_energy_weight,
        dummy,
    )
    dummy._dodo_optical_weight = MethodType(
        SnapshotDepthHS._dodo_optical_weight,
        dummy,
    )

    total_loss, logs = (
        SnapshotDepthHS._SnapshotDepthHS__compute_loss(
            dummy,
            outputs,
            zeros_depth,
            zeros_hs,
            torch.ones_like(zeros_depth),
        )
    )
    total_loss.backward()

    assert total_loss.item() > 0.0
    assert logs["psf_loss_weighted"].item() > 0.0
    torch.testing.assert_close(
        logs["psf_energy_captured_mean"],
        psf.detach().sum(dim=(-2, -1)).mean(),
        atol=2e-7,
        rtol=0,
    )
    gradient = model.doe1.zernike_coeffs.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.norm().item() > 0.0


def test_centered_delta_in_odd_129_psf_has_no_convolution_shift(monkeypatch):
    model = _make_consistent_model().eval()
    artificial_psf = torch.zeros((1, 25, 129, 129))
    artificial_psf[:, :, 64, 64] = 1.0

    def fake_psf_bank(self, height, width, device, use_cache=True):
        assert (height, width) == (128, 128)
        return artificial_psf.to(device)

    monkeypatch.setattr(
        model,
        "_generate_psf_bank",
        MethodType(fake_psf_bank, model),
    )
    torch.manual_seed(71)
    spectral = torch.rand((1, 25, 256, 256))
    depth = torch.full((1, 1, 256, 256), float(model.z_centers[0]))

    with torch.no_grad():
        actual = model(spectral, depth, output_size=(128, 128))

    expected = spectral[..., 64:192, 64:192]
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_wavelength_resolved_convolution_preserves_all_25_band_routes(
    monkeypatch,
):
    """Each HS band must use its own oriented PSF without channel mixing."""
    model = _make_consistent_model(
        sensing_mode="identity",
        measurement_channels=25,
    ).eval()
    psf_bank = torch.zeros((1, 25, 5, 5))
    for band in range(25):
        psf_bank[0, band, band % 5, (3 * band) % 5] = 1.0

    def fake_psf_bank(self, height, width, device, use_cache=True):
        assert (height, width) == (128, 128)
        return psf_bank.to(device)

    monkeypatch.setattr(
        model,
        "_generate_psf_bank",
        MethodType(fake_psf_bank, model),
    )
    spectral = torch.zeros((1, 25, 15, 17))
    spectral[0, :, 7, 8] = torch.linspace(0.1, 2.5, 25)
    depth = torch.full(
        (1, 1, 15, 17),
        float(model.z_centers[0]),
    )

    with torch.no_grad():
        actual = model(spectral, depth)
        expected = F.conv2d(
            spectral,
            psf_bank[0].unsqueeze(1).flip(-2, -1),
            padding=(2, 2),
            groups=25,
        )

    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(
        actual.sum(dim=(-2, -1)),
        spectral.sum(dim=(-2, -1)),
        atol=2e-6,
        rtol=2e-5,
    )


def _direct_spatial_depth_convolution(
    model,
    spectral,
    weights,
    psf_bank,
    output_size,
):
    batch, channels, height, width = spectral.shape
    output_height, output_width = output_size
    output_top = (height - output_height) // 2
    output_left = (width - output_width) // 2
    response = model._sensor_response_matrix(
        spectral.device,
        spectral.dtype,
    )
    output = spectral.new_zeros(
        (batch, response.shape[0], output_height, output_width)
    )

    for depth_index in range(psf_bank.shape[0]):
        kernel = psf_bank[depth_index].unsqueeze(1).flip(-2, -1)
        blurred_spectral = F.conv2d(
            spectral,
            kernel,
            padding=(
                psf_bank.shape[-2] // 2,
                psf_bank.shape[-1] // 2,
            ),
            groups=channels,
        )
        blurred_sensor = torch.einsum(
            "bchw,oc->bohw",
            blurred_spectral,
            response,
        )
        blurred_sensor = blurred_sensor[
            ...,
            output_top:output_top + output_height,
            output_left:output_left + output_width,
        ]
        output = output + blurred_sensor * weights[
            :,
            depth_index:depth_index + 1,
            output_top:output_top + output_height,
            output_left:output_left + output_width,
        ]
    return output


def test_overlap_save_matches_direct_linear_convolution_output_and_gradients(
    monkeypatch,
):
    torch.manual_seed(73)
    model = _make_consistent_model(
        num_depth_layers=2,
        sensing_mode="rgb",
        measurement_channels=3,
        psf_depth_chunk_size=2,
    )
    output_size = (11, 13)

    optimized_spectral = torch.rand(
        (2, 25, 19, 21),
        requires_grad=True,
    )
    reference_spectral = (
        optimized_spectral.detach().clone().requires_grad_()
    )
    psf_seed = torch.rand((2, 25, 7, 5))
    psf_seed = psf_seed / psf_seed.sum(
        dim=(-2, -1),
        keepdim=True,
    )
    optimized_psf = psf_seed.detach().clone().requires_grad_()
    reference_psf = psf_seed.detach().clone().requires_grad_()
    weights = torch.softmax(torch.rand((2, 2, 19, 21)), dim=1)

    def fake_psf_bank(self, height, width, device, use_cache=True):
        assert (height, width) == (128, 128)
        return optimized_psf.to(device)

    monkeypatch.setattr(
        model,
        "_generate_psf_bank",
        MethodType(fake_psf_bank, model),
    )

    optimized = model._forward_psf_convolution(
        optimized_spectral,
        weights,
        binner_debug=None,
        debug_stages=False,
        output_size=output_size,
    )
    reference = _direct_spatial_depth_convolution(
        model,
        reference_spectral,
        weights,
        reference_psf,
        output_size,
    )
    torch.testing.assert_close(
        optimized,
        reference,
        atol=3e-6,
        rtol=3e-5,
    )

    probe = torch.linspace(
        0.25,
        1.25,
        optimized.numel(),
        dtype=optimized.dtype,
    ).reshape_as(optimized)
    optimized_gradients = torch.autograd.grad(
        (optimized * probe).mean(),
        (optimized_spectral, optimized_psf),
    )
    reference_gradients = torch.autograd.grad(
        (reference * probe).mean(),
        (reference_spectral, reference_psf),
    )
    for actual_gradient, expected_gradient in zip(
        optimized_gradients,
        reference_gradients,
    ):
        torch.testing.assert_close(
            actual_gradient,
            expected_gradient,
            atol=3e-7,
            rtol=5e-5,
        )
