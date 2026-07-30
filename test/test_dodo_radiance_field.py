import torch
import torch.nn as nn

from torch_optics.forward_dodo import (
    DepthAwareDoDoForwardModel,
    _radiance_to_field_amplitude,
)


class _CaptureIdentity(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_input = None

    def forward(self, value):
        self.last_input = value
        return value


class _IntensitySensor(nn.Module):
    def forward(self, field):
        return field.abs().square().sum(dim=1, keepdim=True)


def test_radiance_to_field_amplitude_round_trip():
    radiance = torch.tensor([0.0, 0.04, 0.25, 1.0], dtype=torch.float32)

    amplitude = _radiance_to_field_amplitude(radiance)

    torch.testing.assert_close(
        amplitude.square(),
        radiance,
        atol=1e-7,
        rtol=1e-6,
    )


def test_soft_layer_weights_preserve_radiometric_energy():
    radiance = torch.tensor([[[[0.64]]]], dtype=torch.float32)
    layer_weights = torch.tensor([0.5, 0.5], dtype=torch.float32)

    layer_intensities = torch.stack([
        _radiance_to_field_amplitude(radiance * weight).square()
        for weight in layer_weights
    ])

    torch.testing.assert_close(
        layer_intensities.sum(dim=0),
        radiance,
        atol=1e-7,
        rtol=1e-6,
    )


def test_radiance_to_field_amplitude_clamps_negative_roundoff():
    radiance = torch.tensor([-1e-7, 0.0, 0.25], dtype=torch.float32)

    amplitude = _radiance_to_field_amplitude(radiance)

    assert torch.isfinite(amplitude).all()
    torch.testing.assert_close(
        amplitude,
        torch.tensor([0.0, 0.0, 0.5], dtype=torch.float32),
    )


def test_depth_aware_whole_field_converts_each_layer_to_field_amplitude():
    model = DepthAwareDoDoForwardModel(
        depth_min=0.4,
        depth_max=2.0,
        num_depth_layers=2,
        input_format="nchw",
        output_format="nchw",
        depth_layering_mode="soft_diopter",
        measurement_norm_mode="none",
        skip_prop2=True,
        sensor_measurement="intensity",
        image_formation_mode="whole_field",
    )
    captures = nn.ModuleList([_CaptureIdentity(), _CaptureIdentity()])
    model.prop1_layers = captures
    model.doe1 = nn.Identity()
    model.prop3 = nn.Identity()
    model.prop3.register_buffer("wave_lengths", torch.empty(25))
    model.sensing_unnorm = _IntensitySensor()

    radiance = torch.full((1, 25, 4, 4), 0.64)
    centers_u = model.diopter_binner.centers_u
    midpoint_depth = float(1.0 / ((centers_u[0] + centers_u[1]) * 0.5))
    depth = torch.full((1, 1, 4, 4), midpoint_depth)

    model(radiance, depth)

    reconstructed_radiance = sum(
        capture.last_input.square()
        for capture in captures
    )
    torch.testing.assert_close(
        reconstructed_radiance,
        radiance,
        atol=1e-6,
        rtol=1e-6,
    )
