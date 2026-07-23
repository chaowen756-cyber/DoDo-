"""Masked hyperspectral reconstruction losses."""

import torch
import torch.nn as nn


class SAMLoss(nn.Module):
    """Spectral-angle-mapper loss in radians."""

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = float(eps)

    def forward(self, y_pred, y_true, mask=None):
        y_pred_flat = y_pred.reshape(y_pred.shape[0], y_pred.shape[1], -1)
        y_true_flat = y_true.reshape(y_true.shape[0], y_true.shape[1], -1)
        dot_product = torch.sum(y_pred_flat * y_true_flat, dim=1)
        norm_pred = torch.linalg.norm(y_pred_flat, dim=1)
        norm_true = torch.linalg.norm(y_true_flat, dim=1)
        cos_angle = dot_product / (norm_pred * norm_true + self.eps)
        clamp_eps = max(
            self.eps, float(torch.finfo(cos_angle.dtype).eps) * 4.0)
        angle = torch.acos(torch.clamp(
            cos_angle, -1.0 + clamp_eps, 1.0 - clamp_eps))

        if mask is None:
            valid = torch.isfinite(angle)
        else:
            if mask.ndim == 4 and mask.shape[1] == 1:
                mask = mask.squeeze(1)
            if mask.ndim != 3:
                raise ValueError(
                    f"mask shape must be [B,H,W] or [B,1,H,W], got {mask.shape}")
            valid = (mask.reshape(mask.shape[0], -1) > 0.5) & torch.isfinite(angle)
        if valid.any():
            return angle[valid].mean()
        return y_pred.sum() * 0.0


def _prepare_mask(mask, prediction):
    if mask is None:
        return torch.ones(
            (prediction.shape[0], 1, prediction.shape[2], prediction.shape[3]),
            dtype=prediction.dtype,
            device=prediction.device,
        )
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError(
            f"mask shape must be [B,H,W] or [B,1,H,W], got {mask.shape}")
    return mask.to(dtype=prediction.dtype, device=prediction.device)


class CombinedLoss(nn.Module):
    """L1 + MSE + SAM + masked spatial-gradient hyperspectral loss."""

    def __init__(
        self,
        l1_weight=1.0,
        sam_weight=0.0,
        mse_weight=0.0,
        gradient_weight=0.0,
    ):
        super().__init__()
        self.l1_weight = float(l1_weight)
        self.sam_weight = float(sam_weight or 0.0)
        self.mse_weight = float(mse_weight)
        self.gradient_weight = float(gradient_weight)
        for name, value in (
            ("l1_weight", self.l1_weight),
            ("sam_weight", self.sam_weight),
            ("mse_weight", self.mse_weight),
            ("gradient_weight", self.gradient_weight),
        ):
            if value < 0.0:
                raise ValueError(f"{name} must be >= 0, got {value}")
        self.sam_loss = SAMLoss()
        print(
            "初始化高光谱损失: "
            f"L1={self.l1_weight:g}, MSE={self.mse_weight:g}, "
            f"SAM={self.sam_weight:g}, Gradient={self.gradient_weight:g}"
        )

    @staticmethod
    def _masked_mean(loss_map, mask, channels):
        return (loss_map * mask).sum() / (
            mask.sum() * int(channels) + torch.finfo(loss_map.dtype).eps)

    @staticmethod
    def _gradient_loss(y_pred, y_true, mask):
        pred_dx = y_pred[..., :, 1:] - y_pred[..., :, :-1]
        true_dx = y_true[..., :, 1:] - y_true[..., :, :-1]
        mask_dx = mask[..., :, 1:] * mask[..., :, :-1]
        pred_dy = y_pred[..., 1:, :] - y_pred[..., :-1, :]
        true_dy = y_true[..., 1:, :] - y_true[..., :-1, :]
        mask_dy = mask[..., 1:, :] * mask[..., :-1, :]
        channels = y_pred.shape[1]
        loss_x = CombinedLoss._masked_mean(
            torch.abs(pred_dx - true_dx), mask_dx, channels)
        loss_y = CombinedLoss._masked_mean(
            torch.abs(pred_dy - true_dy), mask_dy, channels)
        return 0.5 * (loss_x + loss_y)

    def forward(self, y_pred, y_true, mask=None):
        if y_pred.shape != y_true.shape:
            raise ValueError(
                f"prediction/target shapes differ: {y_pred.shape} vs {y_true.shape}")
        mask4d = _prepare_mask(mask, y_pred)
        channels = y_pred.shape[1]
        residual = y_pred - y_true
        loss_l1 = self._masked_mean(torch.abs(residual), mask4d, channels)
        loss_mse = self._masked_mean(residual.square(), mask4d, channels)
        loss_sam = (
            self.sam_loss(y_pred, y_true, mask=mask4d)
            if self.sam_weight > 0.0 else y_pred.sum() * 0.0
        )
        loss_gradient = (
            self._gradient_loss(y_pred, y_true, mask4d)
            if self.gradient_weight > 0.0 else y_pred.sum() * 0.0
        )
        total_loss = (
            self.l1_weight * loss_l1
            + self.mse_weight * loss_mse
            + self.sam_weight * loss_sam
            + self.gradient_weight * loss_gradient
        )
        return total_loss, {
            "l1": loss_l1,
            "mse": loss_mse,
            "sam": loss_sam,
            "gradient": loss_gradient,
        }
