import types

import torch

from torch_optics.forward_dodo import DepthAwareDoDoForwardModel
from torch_optics.metasurface import TiO2ScalarMetasurface, _TiO2SirenSurrogate


def _write_test_checkpoint(tmp_path):
    torch.manual_seed(7)
    surrogate = _TiO2SirenSurrogate()
    with torch.no_grad():
        for parameter in surrogate.parameters():
            parameter.normal_(mean=0.0, std=0.01)
        surrogate.last_layer.bias.copy_(
            torch.tensor([0.55, 0.75, 0.55, 0.50, 0.70, 0.60])
        )
    state = {
        f"model.{name}": tensor
        for name, tensor in surrogate.state_dict().items()
    }
    path = tmp_path / "model.ckpt"
    torch.save({"epoch": 17, "state_dict": state}, path)
    return path


def test_frozen_surrogate_backpropagates_to_geometry(tmp_path):
    checkpoint = _write_test_checkpoint(tmp_path)
    layer = TiO2ScalarMetasurface(
        str(checkpoint),
        spatial_size=8,
        wave_lengths=torch.tensor([420e-9, 540e-9, 660e-9]),
        trainable_geometry=True,
        mlp_chunk_size=31,
        use_activation_checkpoint=True,
    )

    transmission = layer.complex_transmission()
    loss = transmission.real.mean() + transmission.imag.square().mean()
    loss.backward()

    assert transmission.shape == (3, 8, 8)
    assert torch.isfinite(transmission).all()
    assert transmission.abs().max() <= 1.0 + 1e-5
    assert all(not parameter.requires_grad for parameter in layer.surrogate.parameters())
    assert all(parameter.grad is None for parameter in layer.surrogate.parameters())
    assert layer.checkpoint_epoch == 17

    for _, parameter in layer.design_named_parameters():
        assert parameter.requires_grad
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.norm() > 0

    length_nm, width_nm = layer.geometry_nm()
    assert length_nm.min() >= 80.0
    assert length_nm.max() <= 300.0
    assert width_nm.min() >= 80.0
    assert width_nm.max() <= 300.0


def test_frozen_geometry_caches_transmission(tmp_path):
    checkpoint = _write_test_checkpoint(tmp_path)
    layer = TiO2ScalarMetasurface(
        str(checkpoint),
        spatial_size=4,
        wave_lengths=torch.tensor([420e-9, 660e-9]),
        trainable_geometry=False,
        cache_frozen=True,
        use_activation_checkpoint=False,
    )

    first = layer.complex_transmission()
    second = layer.complex_transmission()

    assert first.data_ptr() == second.data_ptr()
    assert not first.requires_grad
    assert all(not parameter.requires_grad for parameter in layer.parameters())


def test_depth_forward_builds_metasurface_transmission_once(tmp_path):
    checkpoint = _write_test_checkpoint(tmp_path)
    model = DepthAwareDoDoForwardModel(
        depth_min=0.4,
        depth_max=2.0,
        num_depth_layers=3,
        use_second_doe=False,
        train_c=True,
        input_format="nchw",
        output_format="nchw",
        measurement_norm_mode="none",
        depth_layering_mode="soft_diopter",
        sensor_measurement="intensity",
        skip_prop2=True,
        optical_element_type="tio2_metasurface",
        metasurface_checkpoint_path=str(checkpoint),
    )
    call_count = 0

    def fake_transmission(self):
        nonlocal call_count
        call_count += 1
        return torch.ones(25, 128, 128, dtype=torch.complex64)

    model.doe1.complex_transmission = types.MethodType(
        fake_transmission, model.doe1
    )
    spectral = torch.rand(1, 25, 128, 128)
    depth = torch.full((1, 1, 128, 128), 1.0)

    with torch.no_grad():
        measurement = model(spectral, depth)

    assert call_count == 1
    assert measurement.shape == (1, 3, 128, 128)
    assert torch.isfinite(measurement).all()
