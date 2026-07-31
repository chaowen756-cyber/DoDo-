import pytest
import torch
import torch.nn.functional as F

from torch_optics.forward_dodo import DepthAwareDoDoForwardModel
from torch_optics.propagation import PropagationLayer
from torch_optics.utils_fft import centered_fft2, centered_ifft2


def _center_pad(value: torch.Tensor, target_size: int) -> tuple[torch.Tensor, int]:
    source_size = value.shape[-1]
    offset = (target_size - source_size) // 2
    remainder = target_size - source_size - offset
    return F.pad(value, (offset, remainder, offset, remainder)), offset


def _center_crop(value: torch.Tensor, size: int, offset: int) -> torch.Tensor:
    return value[..., offset : offset + size, offset : offset + size]


def _legacy_propagation(
    value: torch.Tensor,
    *,
    length: float,
    distance: float,
    wavelengths: torch.Tensor,
) -> torch.Tensor:
    size = value.shape[-1]
    dx = length / size
    fx = torch.linspace(
        -1.0 / (2.0 * dx),
        1.0 / (2.0 * dx) - 1.0 / length,
        size,
        dtype=torch.float32,
    )
    ffx, ffy = torch.meshgrid(fx, fx, indexing="xy")
    freq2 = (ffx.square() + ffy.square())[None]
    kernel = torch.exp(
        (-1j * torch.pi * wavelengths[:, None, None] * distance)
        * freq2.to(torch.complex64)
    )
    kernel = torch.fft.fftshift(kernel, dim=(-2, -1)).unsqueeze(0)
    return centered_ifft2(centered_fft2(value.to(torch.complex64)) * kernel)


def test_padding_factor_one_matches_the_original_formula():
    torch.manual_seed(3)
    wavelengths = torch.tensor([500e-9, 600e-9], dtype=torch.float32)
    value = torch.rand(2, 2, 17, 17)
    layer = PropagationLayer(
        Mp=17,
        L=0.013,
        zi=0.23,
        wave_lengths=wavelengths,
        trainable_z=False,
    )

    expected = _legacy_propagation(
        value,
        length=0.013,
        distance=0.23,
        wavelengths=wavelengths,
    )
    actual = layer(value)

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_padding_factor_rejects_invalid_values():
    with pytest.raises(ValueError, match="must be >= 1"):
        PropagationLayer(padding_factor=0)
    with pytest.raises(TypeError, match="must be an integer"):
        PropagationLayer(padding_factor=1.5)


def test_padding_factor_preserves_sampling_and_matches_explicit_work_grid():
    torch.manual_seed(7)
    wavelengths = torch.tensor([540e-9], dtype=torch.float32)
    value = torch.rand(1, 1, 16, 16)

    padded = PropagationLayer(
        Mp=16,
        L=0.01,
        zi=0.7,
        wave_lengths=wavelengths,
        trainable_z=False,
        padding_factor=2,
    )
    explicit = PropagationLayer(
        Mp=32,
        L=0.02,
        zi=0.7,
        wave_lengths=wavelengths,
        trainable_z=False,
    )

    value_work, offset = _center_pad(value, 32)
    expected = _center_crop(explicit(value_work), 16, offset)
    actual = padded(value)

    assert padded.work_Mp == 32
    assert padded.work_L == 0.02
    assert padded.work_L / padded.work_Mp == padded.L / padded.Mp
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_zero_distance_padding_is_identity():
    torch.manual_seed(11)
    value = torch.rand(1, 2, 15, 15)
    layer = PropagationLayer(
        Mp=15,
        L=0.01,
        zi=0.0,
        wave_lengths=torch.tensor([500e-9, 600e-9]),
        trainable_z=False,
        padding_factor=2,
    )

    output = layer(value)

    torch.testing.assert_close(output.real, value, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(
        output.imag,
        torch.zeros_like(value),
        atol=2e-6,
        rtol=0,
    )


def test_full_work_grid_preserves_energy_and_crop_does_not_renormalize():
    size = 32
    factor = 2
    wavelengths = torch.tensor([660e-9], dtype=torch.float32)
    value = torch.zeros(1, 1, size, size)
    value[:, :, 8:24, -5:-1] = 1.0
    value_work, offset = _center_pad(value, size * factor)
    explicit = PropagationLayer(
        Mp=size * factor,
        L=0.02,
        zi=2.0,
        wave_lengths=wavelengths,
        trainable_z=False,
    )
    output_work = explicit(value_work)
    output_crop = _center_crop(output_work, size, offset)

    input_energy = value_work.abs().square().sum()
    work_energy = output_work.abs().square().sum()
    crop_energy = output_crop.abs().square().sum()

    torch.testing.assert_close(work_energy, input_energy, atol=1e-5, rtol=1e-5)
    assert crop_energy < work_energy


def test_forward_work_grid_exposes_the_same_padded_propagation_before_crop():
    torch.manual_seed(13)
    wavelengths = torch.tensor([500e-9, 620e-9], dtype=torch.float32)
    value = torch.rand(1, 2, 17, 17)
    layer = PropagationLayer(
        Mp=17,
        L=0.01,
        zi=0.04,
        wave_lengths=wavelengths,
        trainable_z=False,
        padding_factor=2,
    )

    work_output = layer.forward_work_grid(value)
    cropped_output = layer(value)
    crop_top = (layer.work_Mp - layer.Mp) // 2
    crop_left = (layer.work_Mp - layer.Mp) // 2

    assert work_output.shape == (1, 2, 34, 34)
    torch.testing.assert_close(
        cropped_output,
        work_output[
            ...,
            crop_top : crop_top + layer.Mp,
            crop_left : crop_left + layer.Mp,
        ],
        atol=0,
        rtol=0,
    )


def test_padding_reduces_opposite_edge_periodic_wraparound():
    size = 128
    wavelengths = torch.tensor([660e-9], dtype=torch.float32)
    value = torch.zeros(1, 1, size, size)
    value[:, :, 40:88, -8:-2] = 1.0

    outputs = {}
    for factor in (1, 2, 4):
        layer = PropagationLayer(
            Mp=size,
            L=0.01,
            zi=2.0,
            wave_lengths=wavelengths,
            trainable_z=False,
            padding_factor=factor,
        )
        outputs[factor] = layer(value)

    intensity_no_pad = outputs[1].abs().square()
    intensity_pad2 = outputs[2].abs().square()
    intensity_pad4 = outputs[4].abs().square()
    opposite_no_pad = intensity_no_pad[..., :16].sum() / intensity_no_pad.sum()
    opposite_pad2 = intensity_pad2[..., :16].sum() / intensity_pad2.sum()
    pad2_to_pad4_error = torch.linalg.vector_norm(
        intensity_pad2 - intensity_pad4
    ) / torch.linalg.vector_norm(intensity_pad4)

    assert opposite_no_pad > 0.1
    assert opposite_pad2 < 1e-3
    assert opposite_pad2 < opposite_no_pad * 0.01
    assert pad2_to_pad4_error < 1e-3


def test_pad2_converges_to_pad4_for_far_field_intensity_and_doe_gradient():
    size = 128
    wavelength = torch.tensor([660e-9], dtype=torch.float32)
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    window = torch.zeros(size, size)
    hann = torch.hann_window(64, periodic=False)
    window[32:96, 32:96] = hann[:, None] * hann[None, :]
    carrier = torch.exp(2j * torch.pi * (0.25 * xx + 0.175 * yy))
    value = (window * carrier)[None, None]

    outputs = {
        factor: PropagationLayer(
            Mp=size,
            L=0.01,
            zi=2.0,
            wave_lengths=wavelength,
            trainable_z=False,
            padding_factor=factor,
        )(value)
        for factor in (1, 2, 4)
    }
    reference_intensity = outputs[4].abs().square()
    intensity_errors = {
        factor: (
            torch.linalg.vector_norm(
                outputs[factor].abs().square() - reference_intensity
            )
            / torch.linalg.vector_norm(reference_intensity)
        )
        for factor in (1, 2)
    }

    # lambda*z*N/L^2 aliases at factor 1 and becomes Nyquist-safe at factor 2.
    sampling_ratio = float(wavelength.item() * 2.0 * size / (0.01**2))
    assert sampling_ratio > 1.0
    assert sampling_ratio / 2.0 < 1.0
    assert intensity_errors[1] > 0.1
    assert intensity_errors[2] < 1e-3

    doe_mode = (xx - (size - 1) / 2) / size
    sensor_weight = (0.3 + 1.4 * xx / size + 0.8 * yy / size)[None, None]
    gradients = {}
    sensor_propagation = PropagationLayer(
        Mp=size,
        L=0.0048,
        zi=0.01,
        wave_lengths=wavelength,
        trainable_z=False,
    )
    for factor in (1, 2, 4):
        coefficient = torch.tensor(0.37, requires_grad=True)
        coded = outputs[factor] * torch.exp(1j * coefficient * doe_mode)[None, None]
        sensor_field = sensor_propagation(coded)
        (sensor_field.abs().square() * sensor_weight).mean().backward()
        gradients[factor] = coefficient.grad.detach()

    assert gradients[1] * gradients[2] < 0
    torch.testing.assert_close(
        gradients[2],
        gradients[4],
        atol=1e-8,
        rtol=5e-3,
    )


def test_padding_cache_is_not_checkpointed_and_strict_load_stays_compatible():
    wavelengths = torch.tensor([540e-9], dtype=torch.float32)
    old_layer = PropagationLayer(
        Mp=16,
        L=0.01,
        zi=0.7,
        wave_lengths=wavelengths,
        trainable_z=False,
    )
    _ = old_layer(torch.rand(1, 1, 16, 16))
    state = old_layer.state_dict()

    padded_layer = PropagationLayer(
        Mp=16,
        L=0.01,
        zi=0.1,
        wave_lengths=wavelengths,
        trainable_z=False,
        padding_factor=2,
    )
    probe = torch.rand(1, 1, 16, 16)
    _ = padded_layer(probe)
    assert padded_layer._fixed_kernel_cache is not None
    assert padded_layer._fixed_kernel_cache.numel() > 0
    result = padded_layer.load_state_dict(state, strict=True)
    reference_layer = PropagationLayer(
        Mp=16,
        L=0.01,
        zi=float(old_layer.z.item()),
        wave_lengths=wavelengths,
        trainable_z=False,
        padding_factor=2,
    )

    assert not result.missing_keys
    assert not result.unexpected_keys
    assert all("kernel_cache" not in key for key in state)
    assert "_fixed_kernel_cache" not in dict(padded_layer.named_buffers())
    torch.testing.assert_close(padded_layer.z, old_layer.z)
    torch.testing.assert_close(
        padded_layer(probe),
        reference_layer(probe),
    )


def test_fixed_distance_fresnel_kernel_is_reused():
    layer = PropagationLayer(
        Mp=8,
        L=0.01,
        zi=0.7,
        wave_lengths=torch.tensor([540e-9]),
        trainable_z=False,
        padding_factor=2,
    )

    first = layer._kernel(torch.device("cpu"))
    second = layer._kernel(torch.device("cpu"))

    assert first.data_ptr() == second.data_ptr()
    assert layer._fixed_kernel_cache is first


def test_trainable_distance_fresnel_kernel_is_never_cached():
    layer = PropagationLayer(
        Mp=8,
        L=0.01,
        zi=0.7,
        wave_lengths=torch.tensor([540e-9]),
        trainable_z=True,
        padding_factor=2,
    )

    first = layer._kernel(torch.device("cpu"))
    second = layer._kernel(torch.device("cpu"))

    assert first.data_ptr() != second.data_ptr()
    assert first.grad_fn is not None
    assert second.grad_fn is not None
    assert layer._fixed_kernel_cache is None


def test_depth_aware_model_applies_padding_only_to_prop1():
    model = DepthAwareDoDoForwardModel(
        depth_min=0.4,
        depth_max=2.0,
        num_depth_layers=3,
        input_format="nchw",
        output_format="nchw",
        prop1_padding_factor=2,
    )

    assert all(layer.padding_factor == 2 for layer in model.prop1_layers)
    assert model.prop2.padding_factor == 1
    assert model.prop3.padding_factor == 1


def test_padded_depth_aware_forward_has_finite_nonzero_doe_gradient():
    torch.manual_seed(19)
    model = DepthAwareDoDoForwardModel(
        depth_min=0.4,
        depth_max=2.0,
        num_depth_layers=1,
        doe_type_a="New",
        train_c=True,
        input_format="nchw",
        output_format="nchw",
        depth_layering_mode="soft_diopter",
        measurement_norm_mode="none",
        sensor_measurement="intensity",
        skip_prop2=True,
        prop1_padding_factor=2,
    )
    spectral = torch.rand(1, 25, 128, 128)
    depth = torch.full((1, 1, 128, 128), 2.0)
    spatial_weight = torch.linspace(0.5, 1.5, 128).view(1, 1, 1, 128)

    measurement = model(spectral, depth)
    (measurement * spatial_weight).mean().backward()
    gradient = model.doe1.zernike_coeffs.grad

    assert measurement.shape == (1, 3, 128, 128)
    assert torch.isfinite(measurement).all()
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.norm() > 0


def test_padded_cropped_propagation_adjoint_satisfies_inner_product():
    torch.manual_seed(101)
    layer = PropagationLayer(
        Mp=16,
        L=0.01,
        zi=0.02,
        wave_lengths=torch.tensor([500e-9, 600e-9]),
        trainable_z=False,
        padding_factor=2,
    )
    source = torch.complex(torch.randn(1, 2, 16, 16), torch.randn(1, 2, 16, 16))
    sensor = torch.complex(torch.randn(1, 2, 16, 16), torch.randn(1, 2, 16, 16))

    forward_inner = torch.sum(torch.conj(sensor) * layer(source))
    adjoint_inner = torch.sum(torch.conj(layer.adjoint(sensor)) * source)

    torch.testing.assert_close(forward_inner, adjoint_inner, rtol=2e-5, atol=2e-5)
