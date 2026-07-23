import torch

from util.hs_loss import CombinedLoss


def test_combined_hs_loss_uses_all_requested_terms_and_mask():
    prediction = torch.zeros((1, 3, 4, 4), requires_grad=True)
    target = torch.zeros_like(prediction)
    target[..., 1:3, 1:3] = 1.0
    target[..., 1, 2] = 0.25
    mask = torch.zeros((1, 4, 4))
    mask[..., 1:3, 1:3] = 1.0
    loss_fn = CombinedLoss(
        l1_weight=1.0,
        mse_weight=0.5,
        sam_weight=0.02,
        gradient_weight=0.05,
    )
    loss, components = loss_fn(prediction, target, mask=mask)
    loss.backward()
    assert loss.item() > 0
    assert components['l1'].item() > 0
    assert components['mse'].item() > 0
    assert components['sam'].item() > 0
    assert components['gradient'].item() > 0
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_sam_weight_is_not_silently_forced_to_zero():
    prediction = torch.tensor([[[[1.0]], [[0.0]]]])
    target = torch.tensor([[[[0.0]], [[1.0]]]])
    loss_fn = CombinedLoss(l1_weight=0.0, sam_weight=0.5)
    loss, components = loss_fn(prediction, target)
    torch.testing.assert_close(
        loss, 0.5 * components['sam'], atol=1e-7, rtol=0)
    assert components['sam'].item() > 1.5
