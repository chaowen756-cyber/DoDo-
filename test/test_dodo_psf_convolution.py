import inspect
import types

import pytest
import torch

from torch_optics.doe import DOELayer, DOEFreeLayer, DOEFixedHeightLayer
from torch_optics.forward_dodo import (
    DepthAwareDoDoForwardModel,
    Forward_DM_Spiral_Depth,
    _next_fast_fft_length,
)
from torch_optics.propagation import PadoFresnelPropagationLayer
from util.psf_regularization import psf_energy_concentration_loss
from util.psf_regularization import sensor_weighted_spectral_psf_separation_loss


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
    psf_depth_chunk_size=1,
    free=False,
    n_terms=150,
    zernike_basis_path=None,
    prop1_padding_factor=1,
    doe_basis_mode="legacy_raw12",
    doe_basis_rank=9,
    doe_basis_rank_rtol=1e-4,
    doe_basis_rms_m=3e-6,
    doe_coeff_norm_limit=1.0,
    doe_init_coeff_norm=1.0,
    doe_parameterization="zernike",
    doe_height_path=None,
    doe_height_pad_to_size=0,
    doe_height_resize_mode="area",
    psf_optics_version="legacy",
):
    return DepthAwareDoDoForwardModel(
        depth_min=0.4,
        depth_max=2.0,
        num_depth_layers=num_depth_layers,
        use_second_doe=False,
        doe_type_a=doe_type_a,
        train_c=train_c,
        free=free,
        n_terms=n_terms,
        zernike_basis_path=zernike_basis_path,
        input_format="nchw",
        output_format="nchw",
        assets_dir="torch_optics/assets",
        measurement_norm_mode="none",
        sensing_mode=sensing_mode,
        measurement_channels=measurement_channels,
        depth_layering_mode="soft_diopter",
        sensor_measurement="intensity",
        skip_prop2=True,
        prop1_padding_factor=prop1_padding_factor,
        image_formation_mode=image_formation_mode,
        psf_layer_mask_mode=psf_layer_mask_mode,
        psf_mask_blur_sigma=psf_mask_blur_sigma,
        psf_boundary_mode=psf_boundary_mode,
        psf_depth_chunk_size=psf_depth_chunk_size,
        doe_basis_mode=doe_basis_mode,
        doe_basis_rank=doe_basis_rank,
        doe_basis_rank_rtol=doe_basis_rank_rtol,
        doe_basis_rms_m=doe_basis_rms_m,
        doe_coeff_norm_limit=doe_coeff_norm_limit,
        doe_init_coeff_norm=doe_init_coeff_norm,
        doe_parameterization=doe_parameterization,
        doe_height_path=doe_height_path,
        doe_height_pad_to_size=doe_height_pad_to_size,
        doe_height_resize_mode=doe_height_resize_mode,
        psf_optics_version=psf_optics_version,
    )


def _legacy_loop_psf_bank(model):
    height = width = model.prop1_layers[0].Mp
    bands = int(model.prop3.wave_lengths.numel())
    device = model.z_centers.device
    impulse = torch.zeros((1, bands, height, width), device=device)
    impulse[:, :, height // 2, width // 2] = 1.0
    psfs = []
    for depth_index in range(model.num_depth_layers):
        sensor_field = model._propagate_to_sensor(impulse, depth_index)
        psf = sensor_field.abs().to(torch.float32).square()
        psfs.append(
            (psf / psf.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8))[0]
        )
    return torch.stack(psfs, dim=0)


def _legacy_depth_loop_convolution(model, spectral, weights, psf_bank):
    batch, _, height, width = spectral.shape
    response = model._sensor_response_matrix(
        spectral.device, spectral.dtype)
    kernel_height, kernel_width = psf_bank.shape[-2:]
    fft_size = (
        _next_fast_fft_length(height + kernel_height - 1),
        _next_fast_fft_length(width + kernel_width - 1),
    )
    spectral_fft = torch.fft.rfft2(spectral, s=fft_size, dim=(-2, -1))
    response = response.to(dtype=spectral_fft.dtype)
    output = torch.zeros(
        (batch, response.shape[0], height, width),
        device=spectral.device,
        dtype=spectral.dtype,
    )
    for depth_index in range(model.num_depth_layers):
        psf_fft = torch.fft.rfft2(
            psf_bank[depth_index], s=fft_size, dim=(-2, -1))
        mixed_fft = torch.einsum(
            "bcxy,cxy,oc->boxy", spectral_fft, psf_fft, response)
        full = torch.fft.irfft2(
            mixed_fft, s=fft_size, dim=(-2, -1))
        blurred = full[
            ...,
            kernel_height // 2:kernel_height // 2 + height,
            kernel_width // 2:kernel_width // 2 + width,
        ]
        output = output + blurred * weights[:, depth_index:depth_index + 1]
    return output


def test_default_zernike_mode_remains_legacy_12_term_mat_basis():
    model = _make_model(train_c=True, doe_type_a="New")

    assert isinstance(model.doe1, DOELayer)
    assert model.doe1.zernike_basis.shape == (12, 128, 128)
    assert model.doe1.zernike_coeffs.shape == (12,)


def test_free_zernike_mode_loads_150_term_npy_basis():
    model = _make_model(
        train_c=True,
        doe_type_a="New",
        free=True,
        n_terms=150,
    )

    assert isinstance(model.doe1, DOEFreeLayer)
    assert model.doe1.zernike_basis.shape == (150, 128, 128)
    assert model.doe1.zernike_basis.dtype == torch.float32
    assert model.doe1.zernike_coeffs.shape == (150,)
    assert model.doe1.zernike_coeffs.requires_grad

    psf = model.psf_bank(use_cache=False)
    loss, _ = psf_energy_concentration_loss(
        psf, radius=16.0, outside_budget=0.5, softness=1.5)
    loss.backward()
    grad = model.doe1.zernike_coeffs.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert torch.count_nonzero(grad).item() == 150


def test_orthogonal_rms_mode_is_used_by_psf_model():
    torch.manual_seed(123)
    model = _make_model(
        train_c=True,
        doe_type_a="New",
        doe_basis_mode="orthogonal_rms",
    )

    assert isinstance(model.doe1, DOELayer)
    assert model.doe1.zernike_basis.shape == (9, 128, 128)
    assert model.doe1.zernike_coeffs.shape == (9,)
    psf = model.psf_bank(use_cache=False)
    assert torch.isfinite(psf).all()
    torch.testing.assert_close(
        psf.sum(dim=(-2, -1)),
        torch.ones((1, 25)),
        atol=2e-6,
        rtol=0,
    )
    loss, _ = psf_energy_concentration_loss(
        psf, radius=16.0, outside_budget=0.5, softness=1.5)
    loss.backward()
    gradient = model.doe1.zernike_coeffs.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient).item() == 9


def test_free_zernike_rejects_legacy12_orthogonal_mode():
    with pytest.raises(ValueError, match="legacy 12-term DOE"):
        _make_model(free=True, doe_basis_mode="orthogonal_rms")


def test_fixed_height_doe_reproduces_source_padding_and_is_frozen(tmp_path):
    source = torch.linspace(0.0, 1.0e-6, 127 * 127).reshape(1, 1, 127, 127)
    height_path = tmp_path / "height.pth"
    torch.save(source, height_path)

    doe = DOEFixedHeightLayer(
        heightmap_path=str(height_path),
        source_pad_to_size=128,
        resize_mode="area",
        use_pupil_mask=True,
    )

    expected = torch.nn.functional.pad(source[0, 0], (0, 1, 0, 1))
    assert doe.source_shape == (127, 127)
    torch.testing.assert_close(doe.heightmap(), expected)
    assert list(doe.parameters()) == []
    assert doe.zernike_coeffs is None


def test_fixed_height_doe_integrates_with_consistent_psf_forward(tmp_path):
    source = torch.linspace(0.0, 1.0e-6, 128 * 128).reshape(128, 128)
    height_path = tmp_path / "height.pth"
    torch.save(source, height_path)

    model = _make_model(
        train_c=False,
        doe_parameterization="fixed_height",
        doe_height_path=str(height_path),
        psf_optics_version="consistent_grid_v1",
    )
    assert isinstance(model.doe1, DOEFixedHeightLayer)
    assert model._optics_are_frozen()
    psf = model.psf_bank(use_cache=False)
    assert psf.shape == (1, 25, 129, 129)
    assert torch.isfinite(psf).all()
    captured = psf.sum(dim=(-2, -1))
    assert torch.all(captured <= 1.0 + 2e-6)
    assert torch.all(captured > 0.99)


def test_fixed_height_doe_rejects_optical_optimization(tmp_path):
    height_path = tmp_path / "height.pth"
    torch.save(torch.zeros(128, 128), height_path)

    with pytest.raises(ValueError, match="fixed_height DOE is frozen"):
        _make_model(
            train_c=True,
            doe_parameterization="fixed_height",
            doe_height_path=str(height_path),
        )


def test_baek_native_grid_uses_exact_height_and_pado_sampling(tmp_path):
    source = torch.linspace(0.0, 1.0e-6, 375 * 375).reshape(1, 1, 375, 375)
    height_path = tmp_path / "baek_height.pth"
    torch.save(source, height_path)

    model = _make_model(
        train_c=False,
        doe_parameterization="fixed_height",
        doe_height_path=str(height_path),
        doe_height_pad_to_size=376,
        psf_optics_version="doe_native_grid_v1",
    ).eval()

    assert model.optical_grid_size == 376
    assert model.optical_grid_length_m == pytest.approx(376 * 8e-6)
    assert model.optical_grid_length_m / model.optical_grid_size == pytest.approx(
        8e-6
    )
    assert isinstance(model.prop3, PadoFresnelPropagationLayer)
    assert model.prop3.z.item() == pytest.approx(50e-3)
    notebook_wavelengths = torch.linspace(
        420e-9, 660e-9, 25, dtype=torch.float32
    )
    torch.testing.assert_close(
        model.prop3.wave_lengths, notebook_wavelengths, atol=0, rtol=0
    )
    torch.testing.assert_close(
        model.doe1.wave_lengths, notebook_wavelengths, atol=0, rtol=0
    )
    for layer in model.prop1_layers:
        torch.testing.assert_close(
            layer.wave_lengths, notebook_wavelengths, atol=0, rtol=0
        )
    assert model.doe1.pupil_convention == "pado_integer_centered"
    assert model.doe1.spiral_p.sum().item() == 111007
    expected_height = torch.nn.functional.pad(source[0, 0], (0, 1, 0, 1))
    torch.testing.assert_close(model.doe1.heightmap(), expected_height)

    field = model._prop1_impulse_field_bank(376, 376, torch.device("cpu"))
    assert field.shape == (1, 25, 376, 376)
    coordinates = torch.arange(-188, 188, dtype=torch.float32) * 8e-6
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    radius_grid = torch.sqrt(
        xx.square() + yy.square() + float(model.z_centers[0]) ** 2
    )
    # PADO creates one Light object per wavelength and receives wavelength as
    # a Python scalar.  Comparing complete fields catches the float32
    # broadcast/modulo phase error that a single central sample misses.
    for wavelength_index, wavelength_value in enumerate(
        model.prop3.wave_lengths.detach().cpu().tolist()
    ):
        expected_field = torch.exp(
            1j
            * torch.remainder(
                (2.0 * torch.pi * radius_grid) / float(wavelength_value),
                2.0 * torch.pi,
            ).to(torch.complex64)
        )
        torch.testing.assert_close(
            field[0, wavelength_index], expected_field, atol=0, rtol=0
        )
    coordinate = -188 * 8e-6
    radius = (2.0 * coordinate**2 + float(model.z_centers[0]) ** 2) ** 0.5
    expected_corner = torch.exp(
        1j
        * torch.remainder(
            torch.tensor(2.0 * torch.pi * radius / 420e-9),
            torch.tensor(2.0 * torch.pi),
        ).to(torch.complex64)
    )
    torch.testing.assert_close(field[0, 0, 0, 0], expected_corner, atol=2e-5, rtol=2e-5)

    # Fixed-height optics must cache in training mode even while autograd is
    # globally enabled.  The associated frequency-domain kernel is cached too.
    model.train()
    assert model._optics_are_frozen()
    psf = model.psf_bank(use_cache=True)
    cached_psf = model.psf_bank(use_cache=True)
    assert not psf.requires_grad
    assert cached_psf.data_ptr() == psf.data_ptr()
    psf_fft = model._psf_frequency_bank(psf, (256, 256))
    cached_psf_fft = model._psf_frequency_bank(psf, (256, 256))
    assert cached_psf_fft.data_ptr() == psf_fft.data_ptr()
    assert psf.shape == (1, 25, 129, 129)
    assert torch.isfinite(psf).all()
    torch.testing.assert_close(
        psf.sum(dim=(-2, -1)), torch.ones((1, 25)), atol=2e-6, rtol=2e-6
    )
    assert torch.all(model.psf_capture_fraction > 0.0)
    assert torch.all(model.psf_capture_fraction < 1.0)
    assert list(model.doe1.parameters()) == []


def test_baek_native_grid_rejects_nonfixed_doe():
    with pytest.raises(ValueError, match="defined for doe_parameterization='fixed_height'"):
        _make_model(psf_optics_version="doe_native_grid_v1")


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


def test_legacy_optics_version_preserves_original_grid_and_checkpoint_keys():
    legacy = _make_model()
    consistent = _make_model(psf_optics_version="consistent_grid_v1")

    assert legacy.psf_optics_version == "legacy"
    assert legacy.prop1_layers[0].Mp == 128
    assert legacy.prop1_layers[0].L == 0.01
    assert legacy.prop2.L == 0.006
    assert legacy.prop3.Mp == 128
    assert legacy.prop3.L == 0.0048
    assert legacy.prop3.padding_factor == 1
    assert legacy.psf_kernel_size == 128
    assert legacy.psf_energy_reference == "crop"
    assert not legacy.doe1.use_pupil_mask
    assert legacy.psf_capture_fraction is None
    assert set(legacy.state_dict()) == set(consistent.state_dict())


def test_legacy_optics_version_matches_pre_version_numeric_signature():
    """Lock the legacy path to the fixed output from commit 5d598db."""
    model = _make_model(
        doe_type_a="New",
        num_depth_layers=2,
        prop1_padding_factor=2,
        doe_init_coeff_norm=1.0,
        psf_optics_version="legacy",
    ).eval()
    with torch.no_grad():
        model.doe1.zernike_coeffs.copy_(torch.linspace(-0.7, 0.8, 12))
        psf = model.psf_bank(use_cache=False).contiguous()
        spectral = torch.zeros((1, 25, 128, 128))
        spectral[:, :, 32, 47] = torch.linspace(0.1, 1.0, 25)
        depth = torch.full((1, 1, 128, 128), 1.1)
        output = model(spectral, depth).contiguous()

    def signature(tensor):
        flat = tensor.flatten().double()
        index = torch.arange(flat.numel(), dtype=torch.float64)
        probe = (
            torch.sin(index * 0.0012345)
            + 0.5 * torch.cos(index * 0.000371)
        )
        return torch.stack((
            tensor.double().square().sum(),
            (flat * probe).sum(),
        ))

    assert psf.shape == (2, 25, 128, 128)
    assert output.shape == (1, 25, 128, 128)
    torch.testing.assert_close(
        signature(psf),
        torch.tensor(
            [0.02176754705838957, -0.41372271047027076],
            dtype=torch.float64,
        ),
        atol=2e-8,
        rtol=2e-5,
    )
    torch.testing.assert_close(
        signature(output),
        torch.tensor(
            [0.0003747086780208601, 0.2607653568955358],
            dtype=torch.float64,
        ),
        atol=2e-8,
        rtol=2e-5,
    )


def test_prop1_padding_parameter_preserves_existing_positional_api():
    orthogonal_parameters = (
        "doe_basis_mode",
        "doe_basis_rank",
        "doe_basis_rank_rtol",
        "doe_basis_rms_m",
        "doe_coeff_norm_limit",
        "doe_init_coeff_norm",
        "psf_optics_version",
    )
    for model_factory in (
        DepthAwareDoDoForwardModel,
        Forward_DM_Spiral_Depth,
    ):
        parameters = inspect.signature(model_factory).parameters
        positional = [
            name
            for name, parameter in parameters.items()
            if parameter.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        assert positional[-1] == "prop1_padding_factor"
        assert all(
            parameters[name].kind == inspect.Parameter.KEYWORD_ONLY
            for name in orthogonal_parameters
        )


def test_consistent_grid_rejects_unvalidated_optical_chains():
    with pytest.raises(ValueError, match="defined only"):
        DepthAwareDoDoForwardModel(
            skip_prop2=True,
            psf_optics_version="consistent_grid_v1",
        )

    with pytest.raises(ValueError, match="requires skip_prop2=True"):
        DepthAwareDoDoForwardModel(
            image_formation_mode="psf_convolution",
            sensor_measurement="intensity",
            skip_prop2=False,
            psf_optics_version="consistent_grid_v1",
        )

    with pytest.raises(ValueError, match="requires use_second_doe=False"):
        DepthAwareDoDoForwardModel(
            use_second_doe=True,
            image_formation_mode="psf_convolution",
            sensor_measurement="intensity",
            skip_prop2=True,
            psf_optics_version="consistent_grid_v1",
        )

    with pytest.raises(ValueError, match="requires psf_boundary_mode='linear_zero'"):
        DepthAwareDoDoForwardModel(
            image_formation_mode="psf_convolution",
            sensor_measurement="intensity",
            skip_prop2=True,
            psf_boundary_mode="circular",
            psf_optics_version="consistent_grid_v1",
        )


def test_consistent_grid_uses_one_pitch_and_pupil_amplitude_mask():
    model = _make_model(psf_optics_version="consistent_grid_v1")

    expected_pitch = 0.01 / 128
    assert all(layer.Mp == 128 for layer in model.prop1_layers)
    assert all(
        layer.L / layer.Mp == expected_pitch
        for layer in model.prop1_layers
    )
    assert model.prop3.Mp == 128
    assert model.prop3.L / model.prop3.Mp == expected_pitch
    assert model.prop3.padding_factor == 2
    assert model.prop3.work_Mp == 256
    assert model.prop3.work_L == 0.02
    assert model.doe1.use_pupil_mask

    field = torch.ones((1, 25, 128, 128), dtype=torch.complex64)
    with torch.no_grad():
        modulated = model.doe1(field)
    pupil = model.doe1.spiral_p > 0.5
    torch.testing.assert_close(
        modulated[..., ~pupil],
        torch.zeros_like(modulated[..., ~pupil]),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        modulated[..., pupil].abs(),
        torch.ones_like(modulated[..., pupil].real),
        atol=1e-6,
        rtol=0,
    )

    model._assert_consistent_optical_sampling()
    model.prop3.L = 0.0048
    with pytest.raises(ValueError, match="equal sampling pitch"):
        model._assert_consistent_optical_sampling()


def test_consistent_grid_psf_normalizes_full_field_then_crops_129():
    model = _make_model(
        num_depth_layers=2,
        prop1_padding_factor=2,
        psf_optics_version="consistent_grid_v1",
    ).eval()

    with torch.no_grad():
        prop1_fields = model._prop1_impulse_field_bank(
            128, 128, torch.device("cpu"))
        sensor_field = model._propagate_after_prop1(
            prop1_fields,
            return_sensor_work_grid=True,
        )
        full_intensity = sensor_field.abs().square()
        normalized_full = full_intensity / full_intensity.sum(
            dim=(-2, -1), keepdim=True)
        expected = normalized_full[..., 64:193, 64:193]
        actual = model.psf_bank(use_cache=False)

    assert sensor_field.shape == (2, 25, 256, 256)
    assert actual.shape == (2, 25, 129, 129)
    torch.testing.assert_close(
        normalized_full.sum(dim=(-2, -1)),
        torch.ones((2, 25)),
        atol=2e-6,
        rtol=0,
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert model._last_psf_capture_fraction is not None
    torch.testing.assert_close(
        actual.sum(dim=(-2, -1)),
        model.psf_capture_fraction,
        atol=0,
        rtol=0,
    )
    assert torch.all(model.psf_capture_fraction > 0.99)
    assert torch.all(model.psf_capture_fraction <= 1.0 + 2e-6)


def test_consistent_grid_129_crop_uses_optical_center_not_floor_difference(
    monkeypatch,
):
    model = _make_model(psf_optics_version="consistent_grid_v1").eval()

    def fake_sensor_field(self, field, *, return_sensor_work_grid=False):
        assert return_sensor_work_grid
        result = torch.zeros(
            (self.num_depth_layers, 25, 256, 256),
            dtype=torch.complex64,
            device=field.device,
        )
        result[..., 128, 128] = 1.0
        return result

    monkeypatch.setattr(
        model,
        "_propagate_after_prop1",
        types.MethodType(fake_sensor_field, model),
    )
    with torch.no_grad():
        psf = model.psf_bank(use_cache=False)

    assert psf.shape == (1, 25, 129, 129)
    assert torch.count_nonzero(psf).item() == 25
    torch.testing.assert_close(
        psf[..., 64, 64],
        torch.ones((1, 25)),
        atol=0,
        rtol=0,
    )


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


def test_next_fast_fft_length_avoids_prime_linear_convolution_grids():
    assert _next_fast_fft_length(255) == 256
    assert _next_fast_fft_length(383) == 384
    assert _next_fast_fft_length(384) == 384
    with pytest.raises(ValueError, match="must be >= 1"):
        _next_fast_fft_length(0)


def test_depth_chunk_convolution_matches_legacy_output_and_gradients(monkeypatch):
    torch.manual_seed(100)
    model = _make_model(
        num_depth_layers=3,
        sensing_mode="rgb",
        measurement_channels=3,
        psf_depth_chunk_size=2,
    )
    optimized_spectral = torch.rand(
        (2, 25, 16, 16), requires_grad=True)
    reference_spectral = optimized_spectral.detach().clone().requires_grad_()
    optimized_psf = torch.rand(
        (3, 25, 8, 8), requires_grad=True)
    reference_psf = optimized_psf.detach().clone().requires_grad_()
    weights = torch.softmax(torch.rand((2, 3, 16, 16)), dim=1)

    def fake_psf_bank(self, height, width, device, use_cache=True):
        return optimized_psf

    monkeypatch.setattr(
        model,
        "_generate_psf_bank",
        types.MethodType(fake_psf_bank, model),
    )
    optimized = model._forward_psf_convolution(
        optimized_spectral,
        weights,
        binner_debug=None,
        debug_stages=False,
        output_size=(8, 8),
    )
    reference_full = _legacy_depth_loop_convolution(
        model, reference_spectral, weights, reference_psf)
    reference = reference_full[..., 4:12, 4:12]
    torch.testing.assert_close(
        optimized, reference, atol=2e-5, rtol=2e-5)

    probe = torch.linspace(
        0.5, 1.5, optimized.numel(), dtype=optimized.dtype
    ).reshape_as(optimized)
    (optimized * probe).mean().backward()
    (reference * probe).mean().backward()
    torch.testing.assert_close(
        optimized_spectral.grad,
        reference_spectral.grad,
        atol=2e-7,
        rtol=2e-5,
    )
    torch.testing.assert_close(
        optimized_psf.grad,
        reference_psf.grad,
        atol=2e-7,
        rtol=2e-5,
    )


def test_detached_psf_frequency_bank_is_reused_and_cleared():
    model = _make_model(num_depth_layers=2)
    psf = torch.rand((2, 25, 8, 8))
    first = model._psf_frequency_bank(psf, (24, 24))
    second = model._psf_frequency_bank(psf, (24, 24))

    assert first.data_ptr() == second.data_ptr()
    model.clear_psf_cache()
    assert model._cached_psf_fft_bank is None
    third = model._psf_frequency_bank(psf, (24, 24))
    assert third.data_ptr() != first.data_ptr()


def test_overlap_save_halo_center_matches_full_linear_convolution():
    torch.manual_seed(102)
    model = _make_model(
        num_depth_layers=1,
        sensing_mode="rgb",
        measurement_channels=3,
    ).eval()
    spectral = torch.rand((1, 25, 256, 256))
    depth = torch.ones((1, 1, 256, 256))

    with torch.no_grad():
        full = model(spectral, depth)
        center = model(
            spectral, depth, output_size=(128, 128))

    torch.testing.assert_close(
        center,
        full[..., 64:192, 64:192],
        atol=3e-6,
        rtol=3e-5,
    )


def test_batched_psf_generation_matches_legacy_loop_and_doe_gradient():
    torch.manual_seed(101)
    optimized = _make_model(
        train_c=True,
        doe_type_a="New",
        num_depth_layers=2,
        doe_basis_mode="orthogonal_rms",
    )
    reference = _make_model(
        train_c=True,
        doe_type_a="New",
        num_depth_layers=2,
        doe_basis_mode="orthogonal_rms",
    )
    reference.load_state_dict(optimized.state_dict(), strict=True)

    optimized_psf = optimized.psf_bank(use_cache=False)
    reference_psf = _legacy_loop_psf_bank(reference)
    torch.testing.assert_close(
        optimized_psf,
        reference_psf,
        atol=2e-7,
        rtol=2e-5,
    )

    probe = torch.linspace(
        0.25,
        1.25,
        optimized_psf.numel(),
        dtype=optimized_psf.dtype,
    ).reshape_as(optimized_psf)
    (optimized_psf * probe).mean().backward()
    (reference_psf * probe).mean().backward()
    torch.testing.assert_close(
        optimized.doe1.zernike_coeffs.grad,
        reference.doe1.zernike_coeffs.grad,
        atol=2e-8,
        rtol=2e-4,
    )


def test_trainable_doe_reuses_validation_psf_but_training_gets_live_graph():
    torch.manual_seed(103)
    model = _make_model(
        train_c=True,
        doe_type_a="New",
        num_depth_layers=2,
        doe_basis_mode="orthogonal_rms",
    )

    with torch.no_grad():
        first = model.psf_bank(use_cache=True)
        prop1_fields = model._cached_prop1_field_bank
        second = model.psf_bank(use_cache=True)

    assert first.data_ptr() == second.data_ptr()
    assert not first.requires_grad
    assert prop1_fields is not None
    assert model._cached_prop1_field_bank.data_ptr() == prop1_fields.data_ptr()

    with torch.no_grad():
        model.doe1.zernike_coeffs.add_(0.01)
        updated = model.psf_bank(use_cache=True)
    assert updated.data_ptr() != first.data_ptr()
    assert not torch.equal(updated, first)
    assert model._cached_prop1_field_bank.data_ptr() == prop1_fields.data_ptr()

    live = model.psf_bank(use_cache=True)
    assert live.requires_grad
    assert live.data_ptr() != first.data_ptr()
    live[..., 64, 64].sum().backward()
    assert model.doe1.zernike_coeffs.grad is not None
    assert torch.isfinite(model.doe1.zernike_coeffs.grad).all()


def test_optics_caches_are_not_checkpointed_and_load_clears_them():
    model = _make_model(num_depth_layers=2).eval()
    with torch.no_grad():
        model.psf_bank(use_cache=True)
        model._psf_frequency_bank(
            model._cached_psf_bank, (256, 256))

    assert model._cached_psf_bank is not None
    assert model._cached_psf_fft_bank is not None
    assert model._cached_prop1_field_bank is not None
    state = model.state_dict()
    assert all("cached_psf" not in key for key in state)
    assert all("cached_prop1" not in key for key in state)

    result = model.load_state_dict(state, strict=True)
    assert not result.missing_keys
    assert not result.unexpected_keys
    assert model._cached_psf_bank is None
    assert model._cached_psf_fft_bank is None
    assert model._cached_prop1_field_bank is None

    with torch.no_grad():
        model.psf_bank(use_cache=True)
        model._psf_frequency_bank(
            model._cached_psf_bank, (256, 256))
    assert model._cached_psf_bank is not None
    assert model._cached_psf_fft_bank is not None
    assert model._cached_prop1_field_bank is not None

    model._apply(lambda tensor: tensor)
    assert model._cached_psf_bank is None
    assert model._cached_psf_fft_bank is None
    assert model._cached_prop1_field_bank is None


def test_prop1_padding_generates_finite_normalized_intensity_psf():
    model = _make_model(prop1_padding_factor=2).eval()

    with torch.no_grad():
        psf = model.psf_bank(use_cache=False)

    assert model.prop1_layers[0].padding_factor == 2
    assert model.prop1_layers[0].work_Mp == 256
    assert model.prop1_layers[0].work_L == 0.02
    assert model.prop2.padding_factor == 1
    assert model.prop3.padding_factor == 1
    assert torch.isfinite(psf).all()
    assert torch.all(psf >= 0)
    torch.testing.assert_close(
        psf.sum(dim=(-2, -1)),
        torch.ones((1, 25)),
        atol=2e-6,
        rtol=0,
    )


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


def test_larger_scene_tile_uses_fixed_128_psf_and_preserves_center_crop(frozen_psf_model):
    spectral = torch.zeros((1, 25, 192, 192))
    spectral[:, :, 96, 96] = 1.0
    depth = torch.ones((1, 1, 192, 192))

    with torch.no_grad():
        psf = frozen_psf_model.psf_bank(use_cache=True)
        output = frozen_psf_model(spectral, depth)

    assert output.shape == (1, 25, 192, 192)
    torch.testing.assert_close(
        output[0, :, 32:160, 32:160], psf[0], atol=1e-7, rtol=1e-5)


def test_sensor_weighted_spectral_separation_penalizes_identical_neighbors():
    psf = torch.zeros((2, 3, 8, 8), requires_grad=True)
    with torch.no_grad():
        psf[:, :, 4, 4] = 1.0
    response = torch.ones((3, 3))

    loss, stats = sensor_weighted_spectral_psf_separation_loss(
        psf, response, margin=0.95)
    loss.backward()

    torch.testing.assert_close(loss.detach(), torch.tensor(0.05), atol=1e-6, rtol=0)
    torch.testing.assert_close(
        stats['adjacent_cosine_mean'], torch.tensor(1.0), atol=1e-6, rtol=0)
    assert psf.grad is not None
    assert torch.isfinite(psf.grad).all()


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
