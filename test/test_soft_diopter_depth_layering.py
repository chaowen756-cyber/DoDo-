import torch

from torch_optics.forward_dodo import DepthAwareDoDoForwardModel, SoftDiopterBinner


def test_soft_diopter_shape():
    binner = SoftDiopterBinner(z_min=0.4, z_max=2.0, num_layers=8)
    depth = torch.rand(2, 1, 64, 64) * 1.6 + 0.4

    weights, z_centers = binner(depth)

    assert weights.shape == (2, 8, 64, 64)
    assert z_centers.shape == (8,)


def test_soft_diopter_weight_sum_valid_invalid():
    binner = SoftDiopterBinner(z_min=0.4, z_max=2.0, num_layers=8)
    depth = torch.rand(2, 1, 16, 16) * 1.6 + 0.4
    valid_mask = torch.ones_like(depth)
    valid_mask[:, :, :4, :4] = 0
    depth[:, :, 4:8, 4:8] = float("nan")

    weights, _ = binner(depth, valid_mask=valid_mask)
    weight_sum = weights.sum(dim=1, keepdim=True)
    valid = torch.isfinite(depth) & (valid_mask > 0)

    torch.testing.assert_close(weight_sum[valid], torch.ones_like(weight_sum[valid]), atol=1e-5, rtol=0)
    torch.testing.assert_close(weight_sum[~valid], torch.zeros_like(weight_sum[~valid]), atol=1e-5, rtol=0)


def test_soft_diopter_boundary_direction():
    z_min = 0.4
    z_max = 2.0
    binner = SoftDiopterBinner(z_min=z_min, z_max=z_max, num_layers=8)

    far_weights, _ = binner(torch.full((1, 1, 4, 4), z_max))
    near_weights, _ = binner(torch.full((1, 1, 4, 4), z_min))

    assert torch.argmax(far_weights[0, :, 0, 0]).item() == 0
    assert torch.argmax(near_weights[0, :, 0, 0]).item() == 7


def test_soft_diopter_differentiability():
    binner = SoftDiopterBinner(z_min=0.4, z_max=2.0, num_layers=8)
    depth = (torch.rand(1, 1, 16, 16) * 1.2 + 0.6).requires_grad_(True)

    weights, _ = binner(depth)
    centers_u = binner.centers_u.view(1, binner.num_layers, 1, 1)
    expected_u = (weights * centers_u).sum(dim=1)
    loss = expected_u.mean()
    loss.backward()

    assert depth.grad is not None
    assert torch.isfinite(depth.grad).all()


def test_soft_diopter_forward_valid_mask_is_wired():
    model = DepthAwareDoDoForwardModel(
        depth_min=0.4,
        depth_max=2.0,
        num_depth_layers=4,
        input_format="nchw",
        output_format="nchw",
        depth_layering_mode="soft_diopter",
        measurement_norm_mode="none",
    )
    spectral = torch.rand(1, 25, 128, 128)
    depth = torch.rand(1, 1, 128, 128) * 1.6 + 0.4
    valid_mask = torch.ones_like(depth)
    valid_mask[:, :, :, 64:] = 0

    _ = model(spectral, depth, valid_mask=valid_mask, debug_stages=True)
    diag = dict(model._last_stage_diag)
    weight_sum_stats = diag["depth_weight_sum"]

    assert weight_sum_stats["finite"]
    assert weight_sum_stats["min"] == 0.0
    assert weight_sum_stats["max"] <= 1.0 + 1e-5


def test_depth_aware_forward_modes_shape():
    spectral = torch.rand(1, 25, 128, 128)
    depth = torch.rand(1, 1, 128, 128) * 1.6 + 0.4
    outputs = []

    for mode in ("hard_depth", "hard_meter", "soft_diopter"):
        model = DepthAwareDoDoForwardModel(
            depth_min=0.4,
            depth_max=2.0,
            num_depth_layers=2,
            input_format="nchw",
            output_format="nchw",
            depth_layering_mode=mode,
            measurement_norm_mode="none",
        )
        with torch.no_grad():
            outputs.append(model(spectral, depth))

    assert outputs[0].shape == outputs[1].shape == outputs[2].shape == (1, 3, 128, 128)


def test_soft_diopter_doe_grad_smoke():
    model = DepthAwareDoDoForwardModel(
        depth_min=0.4,
        depth_max=2.0,
        num_depth_layers=2,
        doe_type_a="New",
        train_c=True,
        input_format="nchw",
        output_format="nchw",
        depth_layering_mode="soft_diopter",
        measurement_norm_mode="none",
    )
    spectral = torch.rand(1, 25, 128, 128)
    depth = torch.rand(1, 1, 128, 128) * 1.6 + 0.4

    loss = model(spectral, depth).mean()
    loss.backward()

    grad = model.doe1.zernike_coeffs.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert grad.norm().item() > 0


def test_soft_diopter_forward_smoothness_smoke():
    model = DepthAwareDoDoForwardModel(
        depth_min=0.4,
        depth_max=2.0,
        num_depth_layers=8,
        input_format="nchw",
        output_format="nchw",
        depth_layering_mode="soft_diopter",
        measurement_norm_mode="none",
    )
    spectral = torch.rand(1, 25, 128, 128)
    z0 = float(model.z_centers[3].item())
    delta = 1e-3

    with torch.no_grad():
        y_left = model(spectral, torch.full((1, 1, 128, 128), z0 - delta))
        y_mid = model(spectral, torch.full((1, 1, 128, 128), z0))
        y_right = model(spectral, torch.full((1, 1, 128, 128), z0 + delta))

    left_step = (y_left - y_mid).abs().mean()
    right_step = (y_right - y_mid).abs().mean()
    scale = torch.stack([y_left.abs().mean(), y_mid.abs().mean(), y_right.abs().mean()]).max()

    assert torch.isfinite(left_step)
    assert torch.isfinite(right_step)
    assert left_step <= scale * 0.25 + 1e-6
    assert right_step <= scale * 0.25 + 1e-6
