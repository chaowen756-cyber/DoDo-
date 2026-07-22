import types

import pytest
import torch

from torch_optics.forward_dodo import DepthAwareDoDoForwardModel
from util.psf_regularization import psf_energy_concentration_loss


def _make_model(
    *,
    image_formation_mode="psf_convolution",
    train_c=False,
    doe_type_a="Zeros",
    num_depth_layers=1,
    sensing_mode="identity",
    measurement_channels=25,
    psf_layer_mask_mode="baek_hard",
    psf_mask_blur_sigma=1.0,
    psf_boundary_mode="linear_zero",
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
        image_formation_mode=image_formation_mode,
        psf_layer_mask_mode=psf_layer_mask_mode,
        psf_mask_blur_sigma=psf_mask_blur_sigma,
        psf_boundary_mode=psf_boundary_mode,
    )


@pytest.fixture(scope="module")
def frozen_psf_model():
    return _make_model().eval()


def test_psf_convolution_requires_intensity_measurement():
    with pytest.raises(ValueError, match="requires sensor_measurement='intensity'"):
        DepthAwareDoDoForwardModel(
            num_depth_layers=1,
            input_format="nchw",
            output_format="nchw",
            image_formation_mode="psf_convolution",
            sensor_measurement="amplitude",
        )


def test_image_formation_mode_does_not_change_state_dict_keys():
    whole_field = _make_model(image_formation_mode="whole_field")
    psf_convolution = _make_model(image_formation_mode="psf_convolution")

    assert set(whole_field.state_dict()) == set(psf_convolution.state_dict())


def test_psf_bank_is_finite_nonnegative_normalized_and_cached(frozen_psf_model):
    with torch.no_grad():
        first = frozen_psf_model.psf_bank(use_cache=True)
        second = frozen_psf_model.psf_bank(use_cache=True)

    assert first.shape == (1, 25, 128, 128)
    assert torch.isfinite(first).all()
    assert torch.all(first >= 0)
    torch.testing.assert_close(
        first.sum(dim=(-2, -1)),
        torch.ones((1, 25)),
        atol=2e-6,
        rtol=0,
    )
    assert first.data_ptr() == second.data_ptr()


def test_baek_depth_masks_sum_to_validity_after_gaussian_blur():
    model = _make_model(num_depth_layers=4, psf_mask_blur_sigma=1.0)
    depth = torch.linspace(0.4, 2.0, 128).view(1, 1, 1, 128).expand(1, 1, 128, 128)
    valid_mask = torch.ones_like(depth)
    valid_mask[..., :16, :16] = 0

    weights, debug = model._baek_depth_weights(depth, valid_mask, return_debug=True)
    weight_sum = weights.sum(dim=1, keepdim=True)

    assert weights.shape == (1, 4, 128, 128)
    torch.testing.assert_close(
        weight_sum[valid_mask > 0],
        torch.ones_like(weight_sum[valid_mask > 0]),
        atol=1e-6,
        rtol=0,
    )
    torch.testing.assert_close(
        weight_sum[valid_mask == 0],
        torch.zeros_like(weight_sum[valid_mask == 0]),
        atol=0,
        rtol=0,
    )
    assert torch.isfinite(debug["weight_sum"]).all()


def test_linear_fft_convolution_has_no_center_shift(monkeypatch):
    model = _make_model(psf_mask_blur_sigma=0.0).eval()
    artificial_psf = torch.zeros((1, 25, 128, 128))
    artificial_psf[:, :, 64, 64] = 1.0

    def fake_psf_bank(self, height, width, device, use_cache=True):
        assert (height, width) == (128, 128)
        return artificial_psf.to(device)

    monkeypatch.setattr(
        model,
        "_generate_psf_bank",
        types.MethodType(fake_psf_bank, model),
    )
    torch.manual_seed(7)
    spectral = torch.rand((1, 25, 128, 128))
    depth = torch.ones((1, 1, 128, 128))

    with torch.no_grad():
        output = model(spectral, depth)

    torch.testing.assert_close(output, spectral, atol=1e-5, rtol=1e-5)


def test_center_point_measurement_matches_generated_psf(frozen_psf_model):
    spectral = torch.zeros((1, 25, 128, 128))
    spectral[:, :, 64, 64] = 1.0
    depth = torch.ones((1, 1, 128, 128))

    with torch.no_grad():
        psf = frozen_psf_model.psf_bank(use_cache=True)
        output = frozen_psf_model(spectral, depth)

    torch.testing.assert_close(output[0], psf[0], atol=1e-7, rtol=1e-5)


def test_psf_forward_is_linear_without_measurement_normalization(frozen_psf_model):
    torch.manual_seed(11)
    first = torch.rand((1, 25, 128, 128)) * 0.25
    second = torch.rand((1, 25, 128, 128)) * 0.25
    depth = torch.ones((1, 1, 128, 128))

    with torch.no_grad():
        summed_input = frozen_psf_model(first + second, depth)
        summed_outputs = frozen_psf_model(first, depth) + frozen_psf_model(second, depth)

    torch.testing.assert_close(summed_input, summed_outputs, atol=2e-6, rtol=2e-5)


def test_forward_can_return_same_live_psf_bank_for_regularization():
    model = _make_model(train_c=True, doe_type_a="New").eval()
    spectral = torch.zeros((1, 25, 128, 128))
    spectral[:, :, 64, 64] = 1.0
    depth = torch.ones((1, 1, 128, 128))

    output, psf = model(spectral, depth, return_psf=True)
    loss, _ = psf_energy_concentration_loss(
        psf, radius=16.0, outside_budget=0.5, softness=1.5)
    loss.backward()

    torch.testing.assert_close(output[0], psf[0], atol=1e-7, rtol=1e-5)
    assert model.doe1.zernike_coeffs.grad is not None
    assert torch.isfinite(model.doe1.zernike_coeffs.grad).all()
    assert model.doe1.zernike_coeffs.grad.norm().item() > 0


def test_baek_equation_applies_depth_mask_after_convolution(monkeypatch):
    model = _make_model(
        num_depth_layers=2,
        psf_mask_blur_sigma=0.0,
    ).eval()
    artificial_psf = torch.zeros((2, 25, 128, 128))
    artificial_psf[0, :, 64, 67] = 1.0
    artificial_psf[1, :, 64, 64] = 1.0

    def fake_psf_bank(self, height, width, device, use_cache=True):
        assert (height, width) == (128, 128)
        return artificial_psf.to(device)

    monkeypatch.setattr(
        model,
        "_generate_psf_bank",
        types.MethodType(fake_psf_bank, model),
    )

    torch.manual_seed(17)
    spectral = torch.rand((1, 25, 128, 128))
    weights = torch.zeros((1, 2, 128, 128))
    weights[:, 0, :, :64] = 1.0
    weights[:, 1, :, 64:] = 1.0

    def linear_convolution_same(image, kernel):
        fft_size = (
            image.shape[-2] + kernel.shape[-2] - 1,
            image.shape[-1] + kernel.shape[-1] - 1,
        )
        result = torch.fft.irfft2(
            torch.fft.rfft2(image, s=fft_size)
            * torch.fft.rfft2(kernel, s=fft_size),
            s=fft_size,
        )
        start_y = kernel.shape[-2] // 2
        start_x = kernel.shape[-1] // 2
        return result[
            ..., start_y:start_y + image.shape[-2], start_x:start_x + image.shape[-1]
        ]

    with torch.no_grad():
        output = model._forward_psf_convolution(
            spectral, weights, binner_debug=None, debug_stages=False)
        post_mask_reference = sum(
            linear_convolution_same(spectral, artificial_psf[k])
            * weights[:, k:k + 1]
            for k in range(2)
        )
        pre_mask_reference = sum(
            linear_convolution_same(
                spectral * weights[:, k:k + 1], artificial_psf[k])
            for k in range(2)
        )

    torch.testing.assert_close(output, post_mask_reference, atol=2e-6, rtol=2e-5)
    assert (output - pre_mask_reference).abs().max().item() > 0.1


def test_sensor_response_matrix_matches_existing_sensing_layer():
    model = _make_model(sensing_mode="rgb", measurement_channels=3)
    torch.manual_seed(19)
    intensity = torch.rand((2, 25, 8, 8))
    response = model._sensor_response_matrix(intensity.device, intensity.dtype)
    direct = torch.einsum("oc,bchw->bohw", response, intensity)
    via_existing_layer = model.sensing_unnorm(torch.sqrt(intensity).to(torch.complex64))

    torch.testing.assert_close(direct, via_existing_layer, atol=2e-6, rtol=1e-6)


def test_whole_field_dispatch_matches_direct_legacy_path():
    model = _make_model(image_formation_mode="whole_field", num_depth_layers=2).eval()
    torch.manual_seed(23)
    spectral = torch.rand((1, 25, 128, 128))
    depth = torch.rand((1, 1, 128, 128)) * 1.6 + 0.4

    with torch.no_grad():
        prepared_spectral, prepared_depth, prepared_mask = model._prepare_inputs(
            spectral, depth, None)
        weights, debug = model._current_depth_weights(
            prepared_depth, prepared_mask, return_debug=False)
        direct = model._forward_whole_field(
            prepared_spectral, weights, debug, debug_stages=False)
        dispatched = model(spectral, depth)

    torch.testing.assert_close(dispatched, direct, atol=0, rtol=0)


def test_psf_convolution_propagates_finite_nonzero_doe_gradient():
    torch.manual_seed(29)
    model = _make_model(
        train_c=True,
        doe_type_a="New",
        num_depth_layers=1,
        sensing_mode="rgb",
        measurement_channels=3,
        psf_mask_blur_sigma=0.0,
    )
    spectral = torch.rand((1, 25, 128, 128))
    depth = torch.ones((1, 1, 128, 128))

    loss = model(spectral, depth).square().mean()
    loss.backward()

    gradient = model.doe1.zernike_coeffs.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.norm().item() > 0


def test_psf_debug_stages_report_operator_and_energy(frozen_psf_model):
    spectral = torch.rand((1, 25, 128, 128))
    depth = torch.ones((1, 1, 128, 128))

    with torch.no_grad():
        output = frozen_psf_model(spectral, depth, debug_stages=True)

    diagnostics = dict(frozen_psf_model._last_stage_diag)
    assert output.shape == (1, 25, 128, 128)
    assert diagnostics["image_formation_mode"]["mode"] == "psf_convolution"
    assert diagnostics["psf_boundary_mode"]["mode"] == "linear_zero"
    assert diagnostics["psf_energy"]["finite"]
    assert abs(diagnostics["psf_energy"]["mean"] - 1.0) < 2e-6
