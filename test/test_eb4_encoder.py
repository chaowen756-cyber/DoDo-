"""Structural, physics, and smoke tests for the E2--E4 EB4 encoder."""

import pytest
import torch

from nets.eb4_encoder import EB4EncoderBlock, build_sensor_projectors
from nets.mamba_helper import VSSBlock
from nets.mamba_unet import DoubleConv, MambaDualHeadUNet, MambaEncoderBlock


def _response():
    generator = torch.Generator().manual_seed(18)
    return torch.rand(3, 25, generator=generator)


def _build_backbone():
    return MambaDualHeadUNet(
        in_channels=32,
        out_hs_channels=25,
        scheme='hybrid',
        norm_type='group',
        depth_bins=16,
        encoder_variant='eb4',
        sensor_response=_response(),
        wavelengths=torch.linspace(420e-9, 660e-9, 25),
    )


def test_eb4_null_space_projection_preserves_rgb_response():
    response = _response()
    row_projector, null_projector = build_sensor_projectors(response)
    prior = torch.rand(2, 25, 7, 9)
    proposal = torch.randn_like(prior)
    corrected = prior + torch.einsum('ij,bjhw->bihw', null_projector, proposal)

    prior_rgb = torch.einsum('ci,bihw->bchw', response, prior)
    corrected_rgb = torch.einsum('ci,bihw->bchw', response, corrected)
    assert torch.allclose(prior_rgb, corrected_rgb, atol=2e-6, rtol=2e-6)
    assert torch.linalg.matrix_norm(response @ null_projector) < 1e-5
    assert torch.allclose(row_projector + null_projector, torch.eye(25), atol=1e-6)


def test_eb4_replaces_only_encoder_levels_e2_to_e4():
    model = _build_backbone()

    assert isinstance(model.encoders[0], MambaEncoderBlock)
    assert all(isinstance(block, EB4EncoderBlock) for block in model.encoders[1:])
    assert isinstance(model.bottleneck[1], VSSBlock)
    assert isinstance(model.conv_depth_4, DoubleConv)
    assert isinstance(model.conv_hs_4, DoubleConv)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='mamba-ssm forward requires CUDA')
def test_eb4_full_backbone_forward_backward_and_diagnostics():
    torch.manual_seed(19)
    device = torch.device('cuda')
    model = _build_backbone().to(device)
    model.init_depth_conditioning_identity()
    model.init_eb4_stable()

    feature = torch.randn(1, 32, 32, 32, device=device, requires_grad=True)
    rgb = torch.rand(1, 3, 32, 32, device=device)
    prior = torch.rand(1, 25, 32, 32, device=device)
    depth_logits, hs_logits = model(
        feature, rgb_measurement=rgb, spectral_prior=prior
    )

    assert depth_logits.shape == (1, 16, 32, 32)
    assert hs_logits.shape == (1, 25, 32, 32)
    assert torch.isfinite(depth_logits).all()
    assert torch.isfinite(hs_logits).all()

    (depth_logits.square().mean() + hs_logits.square().mean()).backward()
    assert feature.grad is not None and torch.isfinite(feature.grad).all()
    for block in model.encoders[1:]:
        assert block.depth_gate.output_layer.weight.grad is not None
        assert block.hs_gate.output_layer.weight.grad is not None
        assert block.spectral_branch.latent_correction.weight.grad is not None

    diagnostics = model.eb4_diagnostics()
    assert set(diagnostics) == {'e2', 'e3', 'e4'}
    for values in diagnostics.values():
        assert values['sensor_null_error'] < 1e-5
        for gate_name in ('common', 'depth', 'hs'):
            assert torch.allclose(
                values[gate_name].sum(),
                torch.tensor(1.0, device=values[gate_name].device),
                atol=1e-5,
            )
