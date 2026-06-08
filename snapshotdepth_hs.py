# snapshotdepth_hs.py (最终完整版)

import copy
from argparse import ArgumentParser
from collections import namedtuple
import numpy as np

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim
import torchvision.transforms
import torchvision.utils
# from pytorch_lightning.metrics.regression import MeanAbsoluteError, MeanSquaredError
# PL 1.5+: metrics 迁移到 torchmetrics
try:
    from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError
except ImportError:
    try:
        from torchmetrics import MeanAbsoluteError, MeanSquaredError
    except ImportError:
        # 最后兜底：老版本 PL
        from pytorch_lightning.metrics.regression import MeanAbsoluteError, MeanSquaredError

# 导入我们修改过的文件
from models.simple_model_mamba import SimpleModelHS as SimpleModel
from optics import hyperspectral_camera as camera
from torch_optics.forward_dodo import DepthAwareDoDoForwardModel
from util.hs_loss import CombinedLoss

# 导入项目原有的辅助工具
from solvers.image_reconstruction import apply_tikhonov_inverse
from util.fft import crop_psf, fftshift
from util.helper import crop_boundary, gray_to_rgb, imresize, ips_to_metric

SnapshotOutputs = namedtuple('SnapshotOutputs',
                             field_names=['captimgs', 'captimgs_linear',
                                          'est_images', 'est_depthmaps',
                                          'target_images', 'target_depthmaps',
                                          'psf'])


class SnapshotDepthHS(pl.LightningModule):

    def __init__(self, hparams, log_dir=None, artifact_root=None):
        super().__init__()
        # PL 1.9+: hparams 是只读属性，不能直接赋值
        self._hparams_local = copy.deepcopy(hparams)
        self.save_hyperparameters(self._hparams_local)
        self.__build_model()
        # PL 1.9+: Metric 必须作为 Module 属性注册，用 ModuleDict
        self.metrics = torch.nn.ModuleDict({
            'mae_depthmap': MeanAbsoluteError(),
            'mse_depthmap': MeanSquaredError(),
            'mae_image': MeanAbsoluteError(),
            'mse_image': MeanSquaredError(),
        })
        self.log_dir = log_dir
        self.artifact_root = artifact_root
        self._val_psnr_hs_sum = 0.0
        self._val_mae_depth_m_sum = 0.0
        self._val_mae_depth_sum = 0.0
        self._val_steps = 0
        self._val_skipped_steps = 0
        # Diagnostics state
        self._doe_diag_done = False
        self._nonfinite_count = 0
        self._clamp_hook_count = 0
        self._doe_grad_norms = []
        self._last_train_loss_logs = {}
        self._last_train_misc_logs = {}

    # =================================================================================
    # ## 以下是之前缺失的、从原始文件迁移过来的 PyTorch Lightning 核心方法 ##
    # =================================================================================

#     def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure=None, on_tpu=False,
#                        using_native_amp=False, using_lbfgs=False):
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx=0, optimizer_closure=None, **kwargs):
        if self.trainer.global_step < 54:
            lr_scale = min(1., float(self.trainer.global_step + 1) / 54.)
            for pg in optimizer.param_groups:
                if pg.get('name') == 'optics':
                    pg['lr'] = lr_scale * self.hparams.optics_lr
                else:
                    pg['lr'] = lr_scale * self.hparams.cnn_lr
        optimizer.step(closure=optimizer_closure)
        if self.hparams.optimize_optics and hasattr(self.camera, 'clamp_parameters_'):
            self.camera.clamp_parameters_()
            self._clamp_hook_count += 1
            if self._clamp_hook_count == 1:
                print('[doe_diag] clamp_parameters_() executed (first call)')

    def configure_optimizers(self):
        param_groups = []
        optics_params = list(self.camera.parameters())
        if self.hparams.optimize_optics and len(optics_params) > 0:
            param_groups.append({'params': optics_params, 'lr': self.hparams.optics_lr, 'name': 'optics'})
        param_groups.append({'params': self.decoder.parameters(), 'lr': self.hparams.cnn_lr, 'name': 'cnn'})
        optimizer = torch.optim.Adam(param_groups)

        # DOE param group identity diagnostics
        if self.hparams.optimize_optics and hasattr(self.camera, 'doe1') and hasattr(self.camera.doe1, 'zernike_coeffs'):
            zc = self.camera.doe1.zernike_coeffs
            if isinstance(zc, nn.Parameter):
                in_optics = any(zc is p for pg in optimizer.param_groups if pg.get('name') == 'optics' for p in pg['params'])
                print(f'[doe_diag] doe1.zernike_coeffs.requires_grad={zc.requires_grad}, '
                      f'in optics param group (by identity)={in_optics}')
        return optimizer

#     def training_step(self, samples, batch_idx):
#         target_images = samples['hs_image']
#         target_depthmaps = samples['depth_map']
# #         print(f"[原始数据检查]")
# #         print(f"  target_images:    min={target_images.min():.4f}, max={target_images.max():.4f}")
# #         print(f"  target_depthmaps: min={target_depthmaps.min():.4f}, max={target_depthmaps.max():.4f}")
# #         print(f"  target_depthmaps 是否全零: {(target_depthmaps == 0).all()}")
#         # 修复多余维度
#         if target_images.ndim == 5:
#             target_images = target_images.squeeze(1)
    
#         if target_depthmaps.ndim == 4:
#             target_depthmaps = target_depthmaps.squeeze(1)
    
#         if batch_idx == 0:
#             print(f"\n{'='*70}")
#             print(f"DEBUG training_step - 数据集直接输出:")
#             print(f"  target_images.shape:    {target_images.shape}")
#             print(f"  target_depthmaps.shape: {target_depthmaps.shape}")
#             print(f"  target_images.ndim:     {target_images.ndim}")
#             print(f"  target_depthmaps.ndim:  {target_depthmaps.ndim}")
#             print(f"{'='*70}\n")

#         # 在这个版本中，我们假设所有像素都是有效的，创建一个全1的置信度图
#         depth_conf = torch.ones_like(target_depthmaps)
#         depth_conf = crop_boundary(depth_conf, self.crop_width * 2)

#         outputs = self.forward(target_images, target_depthmaps, is_testing=torch.tensor(False))
# #         print(f"[forward 后检查]")
# #         print(f"  outputs.target_depthmaps: min={outputs.target_depthmaps.min():.4f}, max={outputs.target_depthmaps.max():.4f}")
# #         print(f"  outputs.target_depthmaps 是否全零: {(outputs.target_depthmaps == 0).all()}")
        
#         data_loss, loss_logs = self.__compute_loss(outputs, outputs.target_depthmaps, outputs.target_images, depth_conf)
#         loss_logs = {f'train_loss/{key}': val for key, val in loss_logs.items()}

#         misc_logs = {
#             'train_misc/est_depth_max': outputs.est_depthmaps.max(),
#             'train_misc/est_depth_min': outputs.est_depthmaps.min(),
#             'train_misc/est_image_max': outputs.est_images.max(),
#             'train_misc/est_image_min': outputs.est_images.min(),
#         }
#         if self.hparams.optimize_optics:
#             misc_logs.update({
#                 'optics/heightmap_max': self.camera.heightmap1d().max(),
#                 'optics/heightmap_min': self.camera.heightmap1d().min(),
#             })

#         logs = {**loss_logs, **misc_logs}

#         if not self.global_step % self.hparams.summary_track_train_every:
#             self.__log_images(outputs, outputs.target_images, outputs.target_depthmaps, 'train')

#         self.log_dict(logs)
#         return data_loss

# 2026.1.22 修改
    def training_step(self, samples, batch_idx):
        # 1. 获取数据，包括新增的 mask
        target_images = samples['hs_image']
        target_depthmaps = samples['depth_map']
        
        # [NEW] 获取 Dataset 返回的 mask
        # 形状通常是 [B, H, W] 或 [B, 1, H, W]，我们需要统一处理
        valid_mask = samples['mask'] 

        # 修复多余维度 (Squeeze logic)
        if target_images.ndim == 5:
            target_images = target_images.squeeze(1)
        if target_depthmaps.ndim == 4:
            target_depthmaps = target_depthmaps.squeeze(1)
        if valid_mask.ndim == 4:
            valid_mask = valid_mask.squeeze(1) # 确保 mask 是 [B, H, W]

        # 2. 合并 Mask 和 边界裁剪 (depth_conf)
        # 你原来的 depth_conf 是为了裁剪边缘效应
        boundary_mask = torch.ones_like(target_depthmaps)
        boundary_mask = crop_boundary(boundary_mask, self.crop_width * 2)
        valid_mask = crop_boundary(valid_mask, self.crop_width * 2)
        # [NEW] 最终的有效区域 = Dataset提供的物理Mask * 边界裁剪Mask
        # 这是一个二值 mask (0.0 或 1.0)
        final_mask = valid_mask * boundary_mask
        
        depth_metric = samples.get('depth_metric')
        if depth_metric is not None and depth_metric.ndim == 4:
            depth_metric = depth_metric.squeeze(1)

        outputs = self.forward(target_images, target_depthmaps, is_testing=torch.tensor(False),
                               depth_metric=depth_metric, valid_mask=valid_mask)

        # 3. 计算 Loss (传入 final_mask)
        data_loss, loss_logs = self.__compute_loss(outputs, outputs.target_depthmaps, outputs.target_images, final_mask)

        # --- DOE diagnostics (first training step only) ---
        if not self._doe_diag_done and self.hparams.optimize_optics and self.optical_model_type == 'dodo_depth':
            self._doe_diag_done = True
            if hasattr(self.camera, 'doe1') and hasattr(self.camera.doe1, 'zernike_coeffs'):
                zc = self.camera.doe1.zernike_coeffs
                print(f'[doe_diag] doe1.zernike_coeffs.requires_grad={zc.requires_grad}')
                # Register backward hook to capture grad stats
                def _make_doe_grad_hook():
                    def hook(grad):
                        if grad is not None:
                            gnorm = grad.norm().item()
                            gfinite = torch.isfinite(grad).all().item()
                            print(f'[doe_diag] doe1.zernike_coeffs.grad norm={gnorm:.6f}, finite={gfinite}')
                            self._doe_grad_norms.append(gnorm)
                        else:
                            print('[doe_diag] WARNING: doe1.zernike_coeffs.grad is None after backward')
                    return hook
                zc.register_hook(_make_doe_grad_hook())
                print(f'[doe_diag] registered backward hook on doe1.zernike_coeffs')
            # Verify optimizer param group membership by identity
            if hasattr(self.trainer, 'optimizers') and self.trainer.optimizers:
                opt = self.trainer.optimizers[0]
                found_optics = False
                for pg in opt.param_groups:
                    if pg.get('name') == 'optics':
                        found_optics = True
                        n_params = len(pg['params'])
                        print(f'[doe_diag] optics param group has {n_params} params')
                        break
                if not found_optics:
                    print('[doe_diag] WARNING: no optics param group found in optimizer')

        # --- Effect diagnostics (periodic, forward-pass only) ---
        if self.global_step % 50 == 0:
            with torch.no_grad():
                capt = outputs.captimgs
                tgt_std = 0.0
                est_std = 0.0
                logits_std = 0.0
                if final_mask.sum() > 0:
                    valid_tgt_depth = outputs.target_depthmaps[final_mask > 0.5]
                    valid_est_depth = outputs.est_depthmaps[final_mask > 0.5]
                    if valid_tgt_depth.numel() > 0:
                        tgt_std = valid_tgt_depth.std().item()
                        est_std = valid_est_depth.std().item()
                est_d = outputs.est_depthmaps
                eps = 1e-6
                est_d_clamped = est_d.clamp(eps, 1.0 - eps)
                depth_logits_approx = torch.log(est_d_clamped / (1.0 - est_d_clamped))
                if final_mask.sum() > 0:
                    logits_std = depth_logits_approx[final_mask > 0.5].std().item()

        # Store raw unprefixed loss values BEFORE key prefixing so
        # _save_validation_artifacts can find the correct keys.
        self._last_train_loss_logs = {k: v.detach() if hasattr(v, 'detach') else v
                                       for k, v in loss_logs.items()}

        # Logging
        loss_logs = {f'train_loss/{key}': val for key, val in loss_logs.items()}
        misc_logs = {
            'train_misc/est_depth_max': outputs.est_depthmaps.max(),
            'train_misc/est_depth_min': outputs.est_depthmaps.min(),
            'train_misc/est_depth_std': outputs.est_depthmaps.std(),
            'train_misc/est_image_max': outputs.est_images.max(),
            'train_misc/est_image_min': outputs.est_images.min(),
            'train_misc/nonfinite_count': float(self._nonfinite_count),
        }
        if self.global_step % 50 == 0:
            misc_logs.update({
                'diag/target_depth_std': tgt_std,
                'diag/est_depth_std': est_std,
                'diag/depth_logits_std': logits_std,
            })
            if final_mask.sum() > 0:
                misc_logs.update({
                    'diag/captimgs_min': capt.min(),
                    'diag/captimgs_max': capt.max(),
                    'diag/captimgs_mean': capt.mean(),
                    'diag/captimgs_std': capt.std(),
                })
        if self.hparams.optimize_optics and self.optical_model_type == 'legacy_camera':
             misc_logs.update({
                'optics/heightmap_max': self.camera.heightmap1d().max(),
                'optics/heightmap_min': self.camera.heightmap1d().min(),
            })

        logs = {**loss_logs, **misc_logs}

        # 按设定间隔记录训练图像，避免每个 step 都写 TensorBoard 导致训练变慢
        if (
            self.hparams.summary_track_train_every > 0
            and self.global_step % self.hparams.summary_track_train_every == 0
        ):
            self.__log_images(outputs, outputs.target_images, outputs.target_depthmaps, 'train', final_mask)

        # Store misc logs for metrics.json (loss logs already stored above)
        self._last_train_misc_logs = misc_logs.copy()

        self.log_dict(logs)
        return data_loss

    def on_after_backward(self):
        """Collect gradient norms after backward pass (reliable, not pre-backward)."""
        if self.global_step % 50 != 0:
            return
        grad_norms = {}
        # Decoder components
        for name, module in [
            ('input_adapter', self.decoder.input_adapter),
            ('backbone', self.decoder.backbone),
        ]:
            total_norm = 0.0
            n_params = 0
            for p in module.parameters():
                if p.grad is not None:
                    total_norm += p.grad.norm().item() ** 2
                    n_params += 1
            grad_norms[name] = (total_norm ** 0.5) if n_params > 0 else 0.0
        # HS head and depth head
        for head_name in ['hs_out', 'depth_out']:
            head = getattr(self.decoder.backbone, head_name, None)
            if head is not None:
                total_norm = 0.0
                for p in head.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.norm().item() ** 2
                grad_norms[f'{head_name}_head'] = total_norm ** 0.5
        # DOE zernike grad
        if hasattr(self, 'camera') and hasattr(self.camera, 'doe1') and hasattr(self.camera.doe1, 'zernike_coeffs'):
            zc = self.camera.doe1.zernike_coeffs
            if isinstance(zc, nn.Parameter) and zc.grad is not None:
                gnorm = zc.grad.norm().item()
                gfinite = torch.isfinite(zc.grad).all().item()
                grad_norms['doe_zernike'] = gnorm
                grad_norms['doe_zernike_finite'] = float(gfinite)
            else:
                grad_norms['doe_zernike'] = 0.0
                grad_norms['doe_zernike_finite'] = 0.0
        # Log all grad norms
        for k, v in grad_norms.items():
            self.log(f'diag/grad_{k}', v if isinstance(v, float) else float(v), on_step=True)
        # Persist latest grad norms for metrics.json
        self._last_grad_norms = grad_norms


    def on_validation_epoch_start(self) -> None:
        for metric in self.metrics.values():
            metric.reset()
            metric.to(self.device)
        self._val_psnr_hs_sum = 0.0
        self._val_mae_depth_m_sum = 0.0
        self._val_mae_depth_sum = 0.0
        self._val_steps = 0
        self._val_skipped_steps = 0

#     def validation_step(self, samples, batch_idx):
#         target_images = samples['hs_image']
#         target_depthmaps = samples['depth_map']
#         depth_conf = torch.ones_like(target_depthmaps)
#         depth_conf = crop_boundary(depth_conf, 2 * self.crop_width)

#         outputs = self.forward(target_images, target_depthmaps, is_testing=torch.tensor(False))

#         est_depthmaps = outputs.est_depthmaps * depth_conf
#         target_depthmaps_val = outputs.target_depthmaps * depth_conf

#         self.metrics['mae_depthmap'](est_depthmaps, target_depthmaps_val)
#         self.metrics['mse_depthmap'](est_depthmaps, target_depthmaps_val)
#         self.metrics['mae_image'](outputs.est_images, outputs.target_images)
#         self.metrics['mse_image'](outputs.est_images, outputs.target_images)

#         self.log('validation/mse_depthmap', self.metrics['mse_depthmap'], on_step=False, on_epoch=True)
#         self.log('validation/mae_depthmap', self.metrics['mae_depthmap'], on_step=False, on_epoch=True)
#         self.log('validation/mse_image', self.metrics['mse_image'], on_step=False, on_epoch=True)
#         self.log('validation/mae_image', self.metrics['mae_image'], on_step=False, on_epoch=True)

#         if batch_idx == 0:
#             self.__log_images(outputs, outputs.target_images, outputs.target_depthmaps, 'validation')
    def validation_step(self, samples, batch_idx):
        target_images = samples['hs_image']
        target_depthmaps = samples['depth_map']

        # [NEW] 获取并处理 Mask
        valid_mask = samples['mask']
        if valid_mask.ndim == 4: valid_mask = valid_mask.squeeze(1)

        # 边界裁剪
        boundary_mask = torch.ones_like(target_depthmaps)
        boundary_mask = crop_boundary(boundary_mask, 2 * self.crop_width)
        valid_mask = crop_boundary(valid_mask, 2 * self.crop_width)
        final_mask = valid_mask * boundary_mask

        depth_metric = samples.get('depth_metric')
        if depth_metric is not None and depth_metric.ndim == 4:
            depth_metric = depth_metric.squeeze(1)

        outputs = self.forward(target_images, target_depthmaps, is_testing=torch.tensor(False),
                               depth_metric=depth_metric, valid_mask=valid_mask)

        est = outputs.est_depthmaps
        tgt = outputs.target_depthmaps

        # 计算差异
        diff = torch.abs(est - tgt) * final_mask
        diff_sq = (est - tgt)**2 * final_mask

        # 计算平均值 (Sum / Count)
        num_valid = final_mask.sum() + 1e-6
        mae = diff.sum() / num_valid
        mse = diff_sq.sum() / num_valid

        # Log 手动计算的指标
        self.log('validation/mse_depthmap', mse, on_step=False, on_epoch=True)
        self.log('validation/mae_depthmap', mae, on_step=False, on_epoch=True)

        # Metric-depth MAE (meters)
        num_valid_px = final_mask.sum()
        if num_valid_px > 0:
            est_m = ips_to_metric(est.clamp(0, 1), self.hparams.min_depth, self.hparams.max_depth)
            tgt_m = ips_to_metric(tgt.clamp(0, 1), self.hparams.min_depth, self.hparams.max_depth)
            mae_depth_m = (torch.abs(est_m - tgt_m) * final_mask).sum() / num_valid_px
        else:
            mae_depth_m = torch.tensor(float('nan'), device=est.device)
        self.log('validation/mae_depth_m', mae_depth_m, on_step=False, on_epoch=True)

        # Masked HS PSNR (with shape check)
        est_images = outputs.est_images
        target_images_val = outputs.target_images
        if est_images.shape != target_images_val.shape:
            raise ValueError(
                f'est_images.shape={tuple(est_images.shape)} != '
                f'target_images.shape={tuple(target_images_val.shape)}'
            )
        mask4d = final_mask.unsqueeze(1)  # (B,1,H,W)
        n_valid_hs = mask4d.sum() * est_images.shape[1]
        if n_valid_hs > 0:
            mse_hs = ((est_images - target_images_val) ** 2 * mask4d).sum() / n_valid_hs
        else:
            mse_hs = torch.tensor(1e-10, device=est_images.device)
        psnr_hs_masked = 10 * torch.log10(1.0 / (mse_hs + 1e-10))
        self.log('validation/psnr_hs_masked', psnr_hs_masked, on_step=False, on_epoch=True)

        # Full-image PSNR (for reference)
        mse_hs_full = ((est_images - target_images_val) ** 2).mean()
        psnr_hs_full = 10 * torch.log10(1.0 / (mse_hs_full + 1e-10))
        self.log('validation/psnr_hs_full', psnr_hs_full, on_step=False, on_epoch=True)

        # Accumulate for epoch-end artifact saving — skip no-valid-pixel batches
        if num_valid_px > 0:
            self._val_psnr_hs_sum += psnr_hs_masked.item()
            if not torch.isnan(mae_depth_m):
                self._val_mae_depth_m_sum += mae_depth_m.item()
            self._val_mae_depth_sum += mae.item()
            self._val_steps += 1
        else:
            self._val_skipped_steps += 1

        # Save first valid batch outputs for artifact PNG generation
        if batch_idx == 0:
            self.__log_images(outputs, outputs.target_images, outputs.target_depthmaps, 'validation', final_mask)
            self._last_val_outputs = outputs
            self._last_val_mask = final_mask

    def on_test_epoch_start(self):
        self.on_validation_epoch_start()

    def test_step(self, samples, batch_idx):
        """Validation-only eval: delegates to validation_step."""
        return self.validation_step(samples, batch_idx)

    def test_epoch_end(self, outputs):
        """Validation-only eval: delegates to validation_epoch_end."""
        self.validation_epoch_end(outputs)

#     def validation_epoch_end(self, outputs):
#         val_loss = self.__combine_loss(self.metrics['mae_depthmap'].compute(),
#                                        self.metrics['mae_image'].compute(),
#                                        0.)
#         self.log('val_loss', val_loss)
#         mse_image = self.metrics['mse_image'].compute()
#         # PSNR = 10 * log10(MAX^2 / MSE)
#         # 假设图像范围是 [0, 1]，MAX = 1.0
#         psnr_image = 10 * torch.log10(1.0 / (mse_image + 1e-10))  # 加小值避免除零
#         self.log('validation/psnr_image', psnr_image)

    def validation_epoch_end(self, outputs):
        n = max(self._val_steps, 1)
        avg_psnr_hs = self._val_psnr_hs_sum / n
        avg_mae_depth_m = self._val_mae_depth_m_sum / n
        avg_mae_depth = self._val_mae_depth_sum / n
        val_loss = self.hparams.depth_loss_weight * avg_mae_depth + \
                   self.hparams.image_loss_weight * (1.0 / (10 ** (avg_psnr_hs / 10) + 1e-10))
        self.log('val_loss', torch.tensor(val_loss, device=self.device))
        extra = {
            'val_loss': val_loss,
            'validation/psnr_hs_masked': avg_psnr_hs,
            'validation/mae_depth_m': avg_mae_depth_m,
            'validation/mae_depthmap': avg_mae_depth,
            'validation/val_steps': self._val_steps,
            'validation/skipped_steps': self._val_skipped_steps,
            'train_misc/nonfinite_count': self._nonfinite_count,
        }
        last_grads = getattr(self, '_last_grad_norms', None)
        if last_grads:
            for k, v in last_grads.items():
                extra[f'diag/grad_{k}'] = v if isinstance(v, float) else float(v)
        # Save artifacts (prefer artifact_root over log_dir)
        out_dir = self.artifact_root or self.log_dir
        if out_dir:
            self._save_validation_artifacts(extra, out_dir)

    def _save_validation_artifacts(self, extra=None, out_dir=None):
        import json, os
        from torchvision.utils import save_image
        out_dir = out_dir or self.artifact_root or self.log_dir
        if not out_dir:
            return
        os.makedirs(out_dir, exist_ok=True)

        # Merge all metrics
        metrics = {}
        for k, v in self.trainer.callback_metrics.items():
            metrics[k] = v.item() if hasattr(v, 'item') else v
        if extra:
            metrics.update(extra)
        metrics['epoch'] = self.current_epoch
        metrics['global_step'] = self.global_step
        # Record eval metadata when validation-only mode is used
        eval_tag = getattr(self, '_eval_tag', None)
        if eval_tag:
            metrics['eval_tag'] = eval_tag
        init_ckpt = getattr(self.hparams, 'init_ckpt_path', '') or getattr(self.hparams, 'validate_only_ckpt', '')
        if init_ckpt:
            metrics['eval_ckpt_path'] = init_ckpt
        # Record train loss from stored last-step logs (not stale callback_metrics which may be zero)
        stored = getattr(self, '_last_train_loss_logs', None)
        stored_misc = getattr(self, '_last_train_misc_logs', None)
        for key_internal, key_export in [
            ('total_loss', 'train_loss/total_loss'),
            ('depth_loss', 'train_loss/depth_loss'),
            ('image_loss_total', 'train_loss/image_loss_total'),
            ('depth_smooth_loss', 'train_loss/depth_smooth_loss'),
            ('image_loss_l1', 'train_loss/image_loss_l1'),
            ('metric_depth_loss', 'train_loss/metric_depth_loss'),
            ('psf_loss', 'train_loss/psf_loss'),
            ('psf_out_of_fov_max', 'train_loss/psf_out_of_fov_max'),
            ('background_hs_loss', 'train_loss/background_hs_loss'),
        ]:
            if stored and key_internal in stored:
                val = stored[key_internal]
                metrics[key_export] = val.item() if hasattr(val, 'item') else float(val)
        # Cache misc from stored values too
        for misc_key in ('est_depth_max', 'est_depth_min', 'est_depth_std', 'est_image_max', 'est_image_min', 'nonfinite_count'):
            export_key = f'train_misc/{misc_key}'
            if stored_misc and misc_key in stored_misc:
                val = stored_misc[misc_key]
                metrics[export_key] = val.item() if hasattr(val, 'item') else float(val)

        with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        hp_path = os.path.join(out_dir, 'hparams.json')
        if not os.path.exists(hp_path):
            try:
                hp_dict = vars(self.hparams) if hasattr(self.hparams, '__dict__') else dict(self.hparams)
                with open(hp_path, 'w') as f:
                    json.dump({k: str(v) for k, v in hp_dict.items()}, f, indent=2)
            except Exception:
                pass

        # Save PNG quicklooks from last validation batch
        outputs = getattr(self, '_last_val_outputs', None)
        final_mask = getattr(self, '_last_val_mask', None)
        if outputs is not None and final_mask is not None:
            self._save_quicklook_pngs(outputs, final_mask, out_dir)

    def _save_quicklook_pngs(self, outputs, final_mask, out_dir):
        import os
        from torchvision.utils import save_image
        # Take first sample in batch
        capt = outputs.captimgs[0:1]  # (1, 3, H, W) for dodo_depth
        est_hs = outputs.est_images[0:1]  # (1, 25, H, W)
        gt_hs = outputs.target_images[0:1]  # (1, 25, H, W)
        est_d = outputs.est_depthmaps[0:1]  # (1, H, W) or (H, W)
        gt_d = outputs.target_depthmaps[0:1]
        mask = final_mask[0:1]

        if est_d.ndim == 2:
            est_d = est_d.unsqueeze(0).unsqueeze(0)
        elif est_d.ndim == 3:
            est_d = est_d.unsqueeze(0) if est_d.shape[0] != 1 else est_d.unsqueeze(1)
        if gt_d.ndim == 2:
            gt_d = gt_d.unsqueeze(0).unsqueeze(0)
        elif gt_d.ndim == 3:
            gt_d = gt_d.unsqueeze(0) if gt_d.shape[0] != 1 else gt_d.unsqueeze(1)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.ndim == 3:
            mask = mask.unsqueeze(0) if mask.shape[0] != 1 else mask.unsqueeze(1)

        # capt_rgb: measurement quicklook (use first 3 channels or pad)
        capt_ch = capt.shape[1]
        if capt_ch >= 3:
            capt_vis = capt[:, :3, :, :].clamp(0, 1)
        else:
            capt_vis = capt[:, :1, :, :].repeat(1, 3, 1, 1).clamp(0, 1)
        save_image(capt_vis, os.path.join(out_dir, 'capt_rgb.png'))

        # gt_hs_rgb / est_hs_rgb: use 3 representative channels
        n_ch = gt_hs.shape[1]
        vis_ch = [n_ch // 4, n_ch // 2, 3 * n_ch // 4]
        gt_hs_rgb = gt_hs[:, vis_ch, :, :].clamp(0, 1)
        est_hs_rgb = est_hs[:, vis_ch, :, :].clamp(0, 1)
        save_image(gt_hs_rgb, os.path.join(out_dir, 'gt_hs_rgb.png'))
        save_image(est_hs_rgb, os.path.join(out_dir, 'est_hs_rgb.png'))

        # gt_depth_m / est_depth_m: convert IPS to metric
        est_d_ips = est_d.clamp(0, 1)
        gt_d_ips = gt_d.clamp(0, 1)
        est_d_m = ips_to_metric(est_d_ips, self.hparams.min_depth, self.hparams.max_depth)
        gt_d_m = ips_to_metric(gt_d_ips, self.hparams.min_depth, self.hparams.max_depth)
        # Normalize to [0,1] for saving
        d_min = self.hparams.min_depth
        d_max = self.hparams.max_depth
        save_image((gt_d_m - d_min) / (d_max - d_min), os.path.join(out_dir, 'gt_depth_m.png'))
        save_image((est_d_m - d_min) / (d_max - d_min), os.path.join(out_dir, 'est_depth_m.png'))

        # depth_abs_error_m
        abs_err = (torch.abs(est_d_m - gt_d_m) * mask).clamp(0, d_max - d_min)
        save_image(abs_err / max((d_max - d_min), 1e-6), os.path.join(out_dir, 'depth_abs_error_m.png'))

    def __build_model(self):
        hparams = self.hparams
        self.crop_width = hparams.crop_width
        self.optical_model_type = getattr(hparams, 'optical_model', 'legacy_camera')

        # Auto-correct incompatible settings for dodo_depth
        if self.optical_model_type == 'dodo_depth':
            if getattr(hparams, 'preinverse', False):
                print('[dodo_depth] preinverse forced to False')
                hparams.preinverse = False
            if getattr(hparams, 'psf_loss_weight', 0.0) > 0:
                print('[dodo_depth] psf_loss_weight forced to 0.0')
                hparams.psf_loss_weight = 0.0

        if self.optical_model_type == 'dodo_depth':
            if hparams.image_sz != 128:
                raise ValueError(
                    f'dodo_depth requires image_sz=128, got {hparams.image_sz}. '
                    'Use --image_sz 128 --crop_width 0.'
                )
            if hparams.crop_width != 0:
                raise ValueError(
                    f'dodo_depth requires crop_width=0, got {hparams.crop_width}. '
                    'Use --image_sz 128 --crop_width 0.'
                )
            n_depth_layers = (getattr(hparams, 'dodo_depth_layers', None) or hparams.n_depths)
            if n_depth_layers < 1:
                raise ValueError(f'dodo_depth_layers must be >= 1, got {n_depth_layers}')
            use_second_doe = getattr(hparams, 'dodo_use_second_doe', False)
            dodo_doe_type = getattr(hparams, 'dodo_doe_type', 'Zeros')
            dodo_forward_norm = getattr(hparams, 'dodo_forward_norm', 'legacy_max')
            dodo_sensing_mode = getattr(hparams, 'dodo_sensing_mode', 'rgb')
            depth_layering_mode = getattr(hparams, 'depth_layering_mode', 'hard_depth')
            soft_diopter_eps = getattr(hparams, 'soft_diopter_eps', 1e-8)
            soft_diopter_bandwidth_scale = getattr(hparams, 'soft_diopter_bandwidth_scale', 1.0)
            dodo_sensor_measurement = getattr(hparams, 'dodo_sensor_measurement', 'amplitude')
            # Determine measurement_channels from sensing mode
            if dodo_sensing_mode == 'rgb':
                hparams.measurement_channels = 3
            elif dodo_sensing_mode == 'identity':
                hparams.measurement_channels = 25
            else:
                meas_ch = getattr(hparams, 'measurement_channels', None)
                if meas_ch is None or meas_ch <= 3:
                    hparams.measurement_channels = 8  # default for spectral_bins
                else:
                    hparams.measurement_channels = int(meas_ch)
            self.camera = DepthAwareDoDoForwardModel(
                depth_min=hparams.min_depth,
                depth_max=hparams.max_depth,
                num_depth_layers=n_depth_layers,
                use_second_doe=use_second_doe,
                doe_type_a=dodo_doe_type,
                train_c=hparams.optimize_optics,
                input_format='nchw',
                output_format='nchw',
                measurement_norm_mode=dodo_forward_norm,
                sensing_mode=dodo_sensing_mode,
                measurement_channels=int(hparams.measurement_channels),
                depth_layering_mode=depth_layering_mode,
                soft_diopter_eps=soft_diopter_eps,
                soft_diopter_bandwidth_scale=soft_diopter_bandwidth_scale,
                sensor_measurement=dodo_sensor_measurement,
            )
            print(f'[dodo_depth] doe_type_a={dodo_doe_type}, train_c={hparams.optimize_optics}, '
                  f'forward_norm={dodo_forward_norm}, '
                  f'depth_layering={depth_layering_mode}, '
                  f'sensor_measurement={dodo_sensor_measurement}, '
                  f'sensing={dodo_sensing_mode} ch={int(hparams.measurement_channels)}, '
                  f'doe1.zernike_coeffs.requires_grad='
                  f'{self.camera.doe1.zernike_coeffs.requires_grad}')
            # measurement_channels = 3 (RGB sensing output)
            if not hasattr(hparams, 'measurement_channels') or hparams.measurement_channels is None:
                hparams.measurement_channels = 3
            print(f'[dodo_depth] DepthAwareDoDoForwardModel: depth_layers={n_depth_layers}, '
                  f'measurement_channels={hparams.measurement_channels}, '
                  f'depth_layering_mode={depth_layering_mode}')
        else:
            mask_diameter = hparams.focal_length / hparams.f_number
            wavelengths = np.linspace(hparams.start_wl, hparams.end_wl, hparams.hs_channels)
            print(f"Initializing camera with {hparams.hs_channels} channels, "
                  f"from {hparams.start_wl * 1e9:.1f}nm to {hparams.end_wl * 1e9:.1f}nm")
            camera_recipe = {
                'wavelengths': wavelengths, 'min_depth': hparams.min_depth, 'max_depth': hparams.max_depth,
                'focal_depth': hparams.focal_depth, 'n_depths': hparams.n_depths,
                'image_size': hparams.image_sz + 4 * self.crop_width,
                'camera_pixel_pitch': hparams.camera_pixel_pitch,
                'focal_length': hparams.focal_length, 'mask_diameter': mask_diameter,
                'mask_size': hparams.mask_sz,
                'mask_upsample_factor': hparams.mask_upsample_factor,
                'diffraction_efficiency': hparams.diffraction_efficiency,
                'full_size': hparams.full_size,
                'use_virtual_lens_phase': getattr(hparams, 'use_virtual_lens_phase', True),
            }
            self.camera = camera.MixedCamera(**camera_recipe, requires_grad=hparams.optimize_optics)
            if not hasattr(hparams, 'measurement_channels') or hparams.measurement_channels is None:
                hparams.measurement_channels = hparams.hs_channels
            print(self.camera)

        # Decoder depth input (opt-in, default false)
        decoder_use_depth_input = getattr(hparams, 'decoder_use_depth_input', False)
        decoder_depth_input_mode = getattr(hparams, 'decoder_depth_input_mode', 'normalized_diopter')
        hparams.decoder_use_depth_input = bool(decoder_use_depth_input)
        hparams.decoder_depth_input_mode = str(decoder_depth_input_mode)
        hparams.decoder_in_channels = (int(hparams.measurement_channels) + 1
                                        if hparams.decoder_use_depth_input
                                        else int(hparams.measurement_channels))

        self.decoder = SimpleModel(hparams)
        decoder_norm = getattr(hparams, 'decoder_norm', 'batch')
        dodo_meas_norm = getattr(hparams, 'dodo_measurement_norm', 'none')
        print(f'[decoder] decoder_norm={decoder_norm}, dodo_measurement_norm={dodo_meas_norm}, '
              f'decoder_use_depth_input={hparams.decoder_use_depth_input}, '
              f'decoder_depth_input_mode={hparams.decoder_depth_input_mode}, '
              f'decoder_in_channels={hparams.decoder_in_channels}')
        self.image_lossfn = CombinedLoss(l1_weight=hparams.l1_loss_weight)
        self.depth_lossfn = torch.nn.L1Loss()

    def forward(self, images, depthmaps, is_testing, depth_metric=None, valid_mask=None):
        while images.ndim > 4:
            if images.shape[1] == 1:
                images = images.squeeze(1)
            else:
                break
        while depthmaps.ndim > 3:
            if depthmaps.shape[1] == 1:
                depthmaps = depthmaps.squeeze(1)
            else:
                break

        images_linear = images

        if self.optical_model_type == 'dodo_depth':
            # Apply valid mask to suppress invalid background spectral input
            if valid_mask is not None:
                if valid_mask.ndim == 3:
                    valid_mask = valid_mask.unsqueeze(1)  # (B,H,W) -> (B,1,H,W)
                images_linear = images_linear * valid_mask

            if depth_metric is None:
                raise ValueError(
                    'dodo_depth requires metric depth input. '
                    'Dataset must provide depth_metric (metric meters).'
                )

            # DepthAwareDoDoForwardModel: input_format='nchw', output_format='nchw'
            # output: (B, 3, H, W)
            captimgs = self.camera(images_linear, depth_metric, valid_mask=valid_mask)
            psf = None

            # NaN/Inf guard: dodo optical model can produce non-finite output for near-zero input
            if not torch.isfinite(captimgs).all():
                n_nonfinite = (~torch.isfinite(captimgs)).sum().item()
                mask_ratio = valid_mask.mean().item() if valid_mask is not None else float('nan')
                spec_sum = images_linear.sum().item()
                dmetric_min = depth_metric.min().item() if depth_metric is not None else float('nan')
                dmetric_max = depth_metric.max().item() if depth_metric is not None else float('nan')
                nonfinite_policy = getattr(self.hparams, 'dodo_nonfinite_policy', 'zero')
                if nonfinite_policy == 'fail':
                    raise RuntimeError(
                        f'[dodo_depth] Nonfinite captimgs detected at global_step={self.global_step}: '
                        f'{n_nonfinite} non-finite values '
                        f'(mask_ratio={mask_ratio:.4f}, input_spectral_sum={spec_sum:.4f}, '
                        f'depth_metric_min={dmetric_min:.4f}, depth_metric_max={dmetric_max:.4f}). '
                        f'Policy is "fail".'
                    )
                print(f'[dodo_depth] WARNING: {n_nonfinite} non-finite values in captimgs '
                      f'(mask_ratio={mask_ratio:.4f}, input_spectral_sum={spec_sum:.4f}, '
                      f'depth_metric_min={dmetric_min:.4f}, depth_metric_max={dmetric_max:.4f}, '
                      f'global_step={self.global_step}). Replacing with 0.')
                captimgs = torch.nan_to_num(captimgs, nan=0.0, posinf=0.0, neginf=0.0)
                self._nonfinite_count += 1

            # Diag: capture measurement before norm (opt-in, no training impact)
            if getattr(self, '_diag_capture', False):
                self._diag_capt_before = captimgs.detach().cpu()

            # dodo_measurement_norm: applied after NaN/Inf guard, before decoder
            captimgs_stats_before = {
                'min': captimgs.min().item(), 'max': captimgs.max().item(),
                'mean': captimgs.mean().item(), 'std': captimgs.std().item(),
            }
            meas_norm_mode = getattr(self.hparams, 'dodo_measurement_norm', 'none')
            # Opt-in inference-only norm override (does not affect training)
            norm_override = getattr(self, '_norm_override', None)
            if norm_override is not None:
                meas_norm_mode = norm_override
            if meas_norm_mode == 'per_sample_mean_std':
                b = captimgs.shape[0]
                captimgs_flat = captimgs.view(b, -1)
                mean = captimgs_flat.mean(dim=1, keepdim=True).view(b, 1, 1, 1)
                std = captimgs_flat.std(dim=1, keepdim=True).view(b, 1, 1, 1)
                captimgs = (captimgs - mean) / (std + 1e-6)
            elif meas_norm_mode == 'per_sample_minmax':
                b = captimgs.shape[0]
                captimgs_flat = captimgs.view(b, -1)
                vmin = captimgs_flat.min(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
                vmax = captimgs_flat.max(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
                captimgs = (captimgs - vmin) / (vmax - vmin + 1e-6)
            # Diag: capture measurement after norm (opt-in)
            if getattr(self, '_diag_capture', False):
                self._diag_capt_after = captimgs.detach().cpu()

            captimgs_stats_after = {
                'min': captimgs.min().item(), 'max': captimgs.max().item(),
                'mean': captimgs.mean().item(), 'std': captimgs.std().item(),
            }
            if meas_norm_mode != 'none' and self.global_step % 50 == 0:
                print(f'[meas_norm] mode={meas_norm_mode} '
                      f'before: min={captimgs_stats_before["min"]:.4f} max={captimgs_stats_before["max"]:.4f} '
                      f'mean={captimgs_stats_before["mean"]:.4f} std={captimgs_stats_before["std"]:.4f} | '
                      f'after: min={captimgs_stats_after["min"]:.4f} max={captimgs_stats_after["max"]:.4f} '
                      f'mean={captimgs_stats_after["mean"]:.4f} std={captimgs_stats_after["std"]:.4f}')

            noise_sigma = (self.hparams.noise_sigma_max - self.hparams.noise_sigma_min) * torch.rand(
                (captimgs.shape[0], 1, 1, 1), device=images.device) + self.hparams.noise_sigma_min
            captimgs = captimgs + noise_sigma * torch.randn(captimgs.shape, device=images.device, dtype=images.dtype)

            captimgs = crop_boundary(captimgs, self.crop_width)
            pinv_volumes = torch.zeros(
                captimgs.shape[0], self.hparams.hs_channels * self.hparams.n_depths,
                captimgs.shape[2], captimgs.shape[3], device=images.device
            )
        else:
            captimgs, target_volumes, psf = self.camera.forward(images_linear, depthmaps,
                                                                occlusion=self.hparams.occlusion,
                                                                is_training=self.training)
            psf_pure = self.camera.psf_at_camera(is_training=torch.tensor(False)).unsqueeze(0)
            noise_sigma = (self.hparams.noise_sigma_max - self.hparams.noise_sigma_min) * torch.rand(
                (captimgs.shape[0], 1, 1, 1), device=images.device) + self.hparams.noise_sigma_min
            captimgs = captimgs + noise_sigma * torch.randn(captimgs.shape, device=images.device, dtype=images.dtype)
            captimgs = crop_boundary(captimgs, self.crop_width)
            target_volumes = crop_boundary(target_volumes, self.crop_width)
            if self.hparams.preinverse:
                psf_cropped = crop_psf(psf_pure, captimgs.shape[-2:])
                pinv_volumes = apply_tikhonov_inverse(captimgs, psf_cropped, self.hparams.reg_tikhonov,
                                                      apply_edgetaper=True)
            else:
                pinv_volumes = torch.zeros_like(target_volumes)

        # Decoder depth input (opt-in, default false; only for non-preinverse path)
        if (getattr(self.hparams, 'decoder_use_depth_input', False)
                and not self.hparams.preinverse):
            mode = getattr(self.hparams, 'decoder_depth_input_mode', 'normalized_diopter')
            z_min = self.hparams.min_depth
            z_max = self.hparams.max_depth
            eps = 1e-8
            # Use metric depth when available, else convert from IPS
            if depth_metric is not None:
                depth_m = depth_metric
                if depth_m.ndim == 3:
                    depth_m = depth_m.unsqueeze(1)
            else:
                depth_ips = depthmaps.clamp(0, 1)
                if depth_ips.ndim == 3:
                    depth_ips = depth_ips.unsqueeze(1)
                depth_m = ips_to_metric(depth_ips, z_min, z_max)
                if depth_m.ndim == 3:
                    depth_m = depth_m.unsqueeze(1)

            depth_safe = depth_m.clamp(min=z_min, max=z_max)
            if mode == 'normalized_diopter':
                u = 1.0 / depth_safe
                u_min = 1.0 / z_max
                u_max = 1.0 / z_min
                depth_feature = (u - u_min) / (u_max - u_min + eps)
            else:  # normalized_z
                depth_feature = (depth_safe - z_min) / (z_max - z_min + eps)
            depth_feature = depth_feature.clamp(0, 1)

            # Ensure spatial dims match captimgs
            if depth_feature.shape[-2:] != captimgs.shape[-2:]:
                depth_feature = crop_boundary(depth_feature, self.crop_width)
            captimgs = torch.cat([captimgs, depth_feature.to(captimgs.dtype)], dim=1)

        model_outputs = self.decoder(captimgs=captimgs, pinv_volumes=pinv_volumes, images=images_linear,
                                     depthmaps=depthmaps)
        target_images = crop_boundary(images, 2 * self.crop_width)
        target_depthmaps = crop_boundary(depthmaps, 2 * self.crop_width)
        est_images = crop_boundary(model_outputs.est_images, self.crop_width)
        est_depthmaps = crop_boundary(model_outputs.est_depthmaps, self.crop_width)

        return SnapshotOutputs(
            target_images=target_images, target_depthmaps=target_depthmaps,
            captimgs=captimgs, captimgs_linear=captimgs, est_images=est_images,
            est_depthmaps=est_depthmaps, psf=psf,
        )

    def __combine_loss(self, depth_loss, image_loss, psf_loss):
        return self.hparams.depth_loss_weight * depth_loss + \
            self.hparams.image_loss_weight * image_loss + \
            self.hparams.psf_loss_weight * psf_loss
    
#     def __compute_loss(self, outputs, target_depthmaps, target_images, depth_conf):
#         est_images = outputs.est_images
#         est_depthmaps = outputs.est_depthmaps

#         # --- 1. 计算各损失分量 ---
#         depth_loss = self.depth_lossfn(est_depthmaps * depth_conf, target_depthmaps * depth_conf)
        
#         image_loss, image_l1, image_sam = self.image_lossfn(est_images, target_images) 
        
#         # --- 修复：在这里为 'psf_out_of_fov_max' 提供一个默认值 ---
#         psf_loss = torch.tensor(0.0, device=depth_loss.device) 
#         psf_out_of_fov_max = torch.tensor(0.0, device=depth_loss.device) # <--- 添加这一行
#         # --- 修复结束 ---
        
#         if self.hparams.psf_loss_weight > 0:
#             psf_out_of_fov_sum, psf_out_of_fov_max = self.camera.psf_out_of_fov_energy(self.hparams.psf_size)
#             psf_loss = psf_out_of_fov_sum / self.hparams.hs_channels
# #              psf_loss = psf_out_of_fov_sum / self.hparams.hs_channels

#         # --- 2. 计算加权后的损失 ---
#         weighted_depth_loss = self.hparams.depth_loss_weight * depth_loss
#         weighted_image_loss = self.hparams.image_loss_weight * image_loss
#         weighted_psf_loss = self.hparams.psf_loss_weight * psf_loss

#         total_loss = weighted_depth_loss + weighted_image_loss + weighted_psf_loss

#         # --- 3. 添加详细的调试信息 (关键！) ---
#         if self.training and self.global_step % 100 == 0:  # 每 5 步打印一次
#             print(f"\n==================== 损失分量分析 (Step {self.global_step}) ====================")
#             print(f"--- 1. 原始损失分量 (Unweighted) ---")
#             print(f"  Depth Loss (L1):       {depth_loss.item():.6f}")
#             print(f"  Image L1 Loss (Raw):   {image_l1.item():.6f}")
#             print(f"  Image SAM Loss (Raw):  {image_sam.item():.6f}")
#             print(f"  Image Loss (Combined): {image_loss.item():.6f}") # (L1*w_l1 + SAM*w_sam)
#             print(f"  PSF Loss (Normalized): {psf_loss.item():.6f}")
            
#             print(f"\n--- 2. 权重设置 (Weights) ---")
#             print(f"  Depth Weight: {self.hparams.depth_loss_weight}")
#             print(f"  Image Weight: {self.hparams.image_loss_weight}") # (e.g., 0.1)
#             print(f"  PSF Weight:   {self.hparams.psf_loss_weight}")   # (e.g., 0)

#             print(f"\n--- 3. 加权后损失分量 (Weighted) ---")
#             print(f"  Weighted Depth: {weighted_depth_loss.item():.6f}")
#             print(f"  Weighted Image: {weighted_image_loss.item():.6f}")
#             print(f"  Weighted PSF:   {weighted_psf_loss.item():.6f}")
            
#             print(f"\n--- 4. 最终总损失 ---")
#             print(f"  TOTAL LOSS:     {total_loss.item():.6f}")
#             image_contribution = weighted_image_loss / total_loss
#             depth_contribution = weighted_depth_loss / total_loss
#             psf_contribution = weighted_psf_loss / total_loss
#             print(f"Loss比例: image={image_contribution:.2%}, depth={depth_contribution:.2%}, psf={psf_contribution:.2%}")

            
#             print(f"\n--- 5. 数据范围检查 (关键！) ---")
#             print(f"  Target Depth:  min={target_depthmaps.min().item():.3f}, max={target_depthmaps.max().item():.3f}, mean={target_depthmaps.mean().item():.3f}")
#             print(f"  Est Depth:     min={est_depthmaps.min().item():.3f}, max={est_depthmaps.max().item():.3f}, mean={est_depthmaps.mean().item():.3f}")
#             print(f"  Target Images: min={target_images.min().item():.3f}, max={target_images.max().item():.3f}, mean={target_images.mean().item():.3f}")
#             print(f"  Est Images:    min={est_images.min().item():.3f}, max={est_images.max().item():.3f}, mean={est_images.mean().item():.3f}")
#             print("========================================================================\n")

#         return total_loss, {
#             'total_loss': total_loss, 'depth_loss': depth_loss, 'image_loss_total': image_loss,
#             'image_loss_l1': image_l1, 'psf_loss': psf_loss,
#             'psf_out_of_fov_max': psf_out_of_fov_max, # 这一行现在安全了
#         }
    def __compute_loss(self, outputs, target_depthmaps, target_images, final_mask):
        est_images = outputs.est_images
        est_depthmaps = outputs.est_depthmaps

        # --- 1. 计算 Masked Depth Loss (关键修改) ---
        # 不要直接调用 self.depth_lossfn，因为它内部可能是 mean reduction
        # 我们手动写，或者确认你的 lossfn 配置为 reduction='none'
        
        # 假设 self.depth_lossfn 是 L1Loss 或 MSELoss
        # 推荐：手动计算以确保万无一失
        
        # 绝对误差图
        diff = torch.abs(est_depthmaps - target_depthmaps)
        
        # 只保留 mask 区域的误差
        masked_diff = diff * final_mask
        
        # 归一化：除以有效像素数量，而不是总像素数量
        # +1e-6 防止除以零
        depth_loss = masked_diff.sum() / (final_mask.sum() + 1e-6)
        
        if self.training and self.global_step % 100 == 0:
            with torch.no_grad():
                # 只看 mask 区域的统计
                valid_gt = target_depthmaps[final_mask > 0.5]
                valid_est = est_depthmaps[final_mask > 0.5]
                
                if valid_gt.numel() > 0:
                    print(f"\n[Step {self.global_step}] IPS 深度预测诊断:")
                    print(f"  GT 深度 (IPS 归一化): min={valid_gt.min():.4f}, max={valid_gt.max():.4f}, "
                          f"mean={valid_gt.mean():.4f}, std={valid_gt.std():.4f}")
                    print(f"  预测深度 (IPS 归一化): min={valid_est.min():.4f}, max={valid_est.max():.4f}, "
                          f"mean={valid_est.mean():.4f}, std={valid_est.std():.4f}")
                    
                    # 物理深度反算（仅用于可视化）
                    # 反演公式：d = (d_min * d_max) / (d_max - (d_max - d_min) * d_norm)
                    def ips_to_physical(ips_norm, min_d=0.4, max_d=2.0):
                        # 避免除以零
                        safe_norm = torch.clamp(ips_norm, 1e-6, 1.0 - 1e-6)
                        return (max_d * min_d) / (max_d - (max_d - min_d) * safe_norm)
                    
                    gt_phys_mean = ips_to_physical(valid_gt.mean()).item()
                    est_phys_mean = ips_to_physical(valid_est.mean()).item()
                    
                    print(f"  GT 物理深度: 平均 ≈ {gt_phys_mean:.3f}m")
                    print(f"  预测物理深度: 平均 ≈ {est_phys_mean:.3f}m")
                    
                    # 检查预测的动态范围
                    gt_range = valid_gt.max() - valid_gt.min()
                    est_range = valid_est.max() - valid_est.min()
                    print(f"  动态范围: GT={gt_range:.4f} (IPS), EST={est_range:.4f} (IPS), "
                          f"比值={est_range/(gt_range+1e-6):.2%}")
                    
                    if est_range < 0.1 and gt_range > 0.3:
                        print(f"  ⚠️ 警告：预测深度动态范围过小！网络可能陷入常数输出。")
                        print(f"      => 检查是否：")
                        print(f"         1. 开启了 optimize_optics=True（DOE 优化）")
                        print(f"         2. Loss 权重设置是否合理（depth_loss_weight）")
                        print(f"         3. 数据中物体间深度差异是否足够大")
        
        
                    print(f"  预测深度 (mask内): min={valid_est.min():.4f}, max={valid_est.max():.4f}, "
                          f"mean={valid_est.mean():.4f}, std={valid_est.std():.4f}")
                    
                    # 检查预测的动态范围
                    gt_range = valid_gt.max() - valid_gt.min()
                    est_range = valid_est.max() - valid_est.min()
                    print(f"  动态范围: GT={gt_range:.4f}, EST={est_range:.4f}, "
                          f"比值={est_range/(gt_range+1e-6):.2%}")
                    
                    if est_range < 0.1 and gt_range > 0.3:
                        print(f"  ⚠️ 警告：预测深度动态范围过小！网络可能陷入常数输出。")
        # --- Image Loss ---
        # 仅在有效区域计算图像重建损失，避免大面积无效区域把误差“稀释”。
        image_loss, image_l1, image_sam = self.image_lossfn(est_images, target_images, mask=final_mask)

        # --- Depth Smoothness Loss (可选) ---
        # 抑制深度图内跟随颜色纹理出现的高频伪细节。
        depth_smooth_weight = float(getattr(self.hparams, 'depth_smooth_weight', 0.0))
        if depth_smooth_weight > 0.0:
            dx = torch.abs(est_depthmaps[:, :, 1:] - est_depthmaps[:, :, :-1])
            dy = torch.abs(est_depthmaps[:, 1:, :] - est_depthmaps[:, :-1, :])
            mask_dx = final_mask[:, :, 1:] * final_mask[:, :, :-1]
            mask_dy = final_mask[:, 1:, :] * final_mask[:, :-1, :]
            smooth_x = (dx * mask_dx).sum() / (mask_dx.sum() + 1e-6)
            smooth_y = (dy * mask_dy).sum() / (mask_dy.sum() + 1e-6)
            depth_smooth_loss = 0.5 * (smooth_x + smooth_y)
        else:
            depth_smooth_loss = torch.tensor(0.0, device=depth_loss.device)

        # --- Metric-Depth SmoothL1 Loss (opt-in) ---
        metric_depth_loss_weight = float(getattr(self.hparams, 'metric_depth_loss_weight', 0.0))
        metric_depth_loss = torch.tensor(0.0, device=depth_loss.device)
        if metric_depth_loss_weight > 0.0:
            # Convert IPS normalized depth to metric meters
            est_m = ips_to_metric(est_depthmaps.clamp(0, 1), self.hparams.min_depth, self.hparams.max_depth)
            tgt_m = ips_to_metric(target_depthmaps.clamp(0, 1), self.hparams.min_depth, self.hparams.max_depth)
            # SmoothL1 in meters, masked, normalized by depth range
            depth_range = self.hparams.max_depth - self.hparams.min_depth
            diff_m = torch.abs(est_m - tgt_m) / depth_range  # normalized residual
            smoothl1 = torch.where(
                diff_m < 1.0,
                0.5 * diff_m ** 2,
                diff_m - 0.5
            )
            num_valid = final_mask.sum() + 1e-6
            metric_depth_loss = (smoothl1 * final_mask).sum() / num_valid

        # --- PSF Loss ---
        psf_loss = torch.tensor(0.0, device=depth_loss.device)
        psf_out_of_fov_max = torch.tensor(0.0, device=depth_loss.device)

        if self.hparams.psf_loss_weight > 0 and self.optical_model_type == 'legacy_camera':
            psf_out_of_fov_sum, psf_out_of_fov_max = self.camera.psf_out_of_fov_energy(self.hparams.psf_size)
            psf_loss = psf_out_of_fov_sum / self.hparams.hs_channels

        # --- Background HS Loss (opt-in) ---
        bg_hs_loss_weight = float(getattr(self.hparams, 'background_hs_loss_weight', 0.0))
        bg_hs_loss = torch.tensor(0.0, device=depth_loss.device)
        if bg_hs_loss_weight > 0.0:
            bg_mask = (1.0 - final_mask.unsqueeze(1))  # [B,1,H,W], background=1
            if bg_mask.sum() > 0:
                target_images_bg = target_images * bg_mask
                est_images_bg = est_images * bg_mask
                bg_hs_loss = (torch.abs(est_images_bg - target_images_bg) * bg_mask).sum() / (bg_mask.sum() + 1e-6)

        # --- 2. 加权 ---
        weighted_depth_loss = self.hparams.depth_loss_weight * depth_loss
        weighted_image_loss = self.hparams.image_loss_weight * image_loss
        weighted_psf_loss = self.hparams.psf_loss_weight * psf_loss
        weighted_depth_smooth_loss = depth_smooth_weight * depth_smooth_loss
        weighted_metric_depth_loss = metric_depth_loss_weight * metric_depth_loss

        weighted_bg_hs_loss = bg_hs_loss_weight * bg_hs_loss

        total_loss = (weighted_depth_loss + weighted_image_loss + weighted_psf_loss +
                     weighted_depth_smooth_loss + weighted_metric_depth_loss +
                     weighted_bg_hs_loss)

        return total_loss, {
            'total_loss': total_loss,
            'depth_loss': depth_loss,
            'depth_smooth_loss': depth_smooth_loss,
            'metric_depth_loss': metric_depth_loss,
            'image_loss_total': image_loss,
            'image_loss_l1': image_l1,
            'psf_loss': psf_loss,
            'psf_out_of_fov_max': psf_out_of_fov_max,
            'background_hs_loss': bg_hs_loss,
        }
    
    @torch.no_grad()
    def __log_images(self, outputs, target_images, target_depthmaps, tag: str, final_mask):
        # 在你的 __log_images 函数中添加
        diff_map = torch.abs(outputs.est_depthmaps - outputs.target_depthmaps) * final_mask
        # 归一化以便显示
        diff_vis = diff_map / (diff_map.max() + 1e-6) 
        # =========== 【修复代码】 ===========
        # 3. 增加通道维度 (B, H, W) -> (B, 1, H, W)
        # TensorBoard 需要 4D 张量
        if diff_vis.ndim == 3:
            diff_vis = diff_vis.unsqueeze(1)
        # ==================================

        # 4. 现在 diff_vis 的形状是 (2, 1, 384, 384)，符合 NCHW
        self.logger.experiment.add_images(f'{tag}/diff_error', diff_vis, self.global_step)
        captimgs, est_images, est_depthmaps = outputs.captimgs, outputs.est_images, outputs.est_depthmaps
        summary_image_sz = self.hparams.summary_image_sz
        summary_max_images = min(self.hparams.summary_max_images, captimgs.shape[0])
        n_channels = target_images.shape[1]
        vis_channels = [n_channels // 4, n_channels // 2, 3 * n_channels // 4]
        # captimgs may have fewer channels (e.g. 3 for dodo_depth)
        capt_ch = captimgs.shape[1]
        if capt_ch >= 3:
            capt_vis_channels = [capt_ch // 4, capt_ch // 2, 3 * capt_ch // 4]
        else:
            capt_vis_channels = list(range(capt_ch))
        # ensure exactly 3 channels for concat
        captimgs_vis = captimgs[:, capt_vis_channels[:3], ...]
        if captimgs_vis.shape[1] < 3:
            captimgs_vis = captimgs_vis.repeat(1, 3 // captimgs_vis.shape[1] + 1, 1, 1)[:, :3, ...]
        target_images_vis = target_images[:, vis_channels, ...]
        est_images_vis = est_images[:, vis_channels, ...]
        target_depthmaps_4d = target_depthmaps.unsqueeze(1)  # (B, 1, H, W)
        est_depthmaps_4d = est_depthmaps.unsqueeze(1)        # (B, 1, H, W)
        captimgs_resized, target_images_resized, target_depthmaps_resized, est_images_resized, est_depthmaps_resized = [
        imresize(x, (summary_image_sz, summary_image_sz)) for x in
        [captimgs_vis, target_images_vis, target_depthmaps_4d, est_images_vis, est_depthmaps_4d]
        ]
        
        target_depthmaps = target_depthmaps_resized.squeeze(1)  # (B, 1, H, W) → (B, H, W)
        est_depthmaps = est_depthmaps_resized.squeeze(1)        # (B, 1, H, W) → (B, H, W)
        captimgs, target_images, est_images = \
            captimgs_resized, target_images_resized, est_images_resized
#        # ✅ 为深度图添加通道维度用于拼接
#         target_depthmaps, est_depthmaps = \
#             gray_to_rgb(1.0 - target_depthmaps), \
#             gray_to_rgb(1.0 - est_depthmaps)
        # --- 修改开始 ---
        
        # 1. 此时 target_depthmaps 是 (B, H, W)，先反转颜色 (1.0 - x)
        # 注意：不需要 gray_to_rgb，我们手动变 RGB
        td_inv = 1.0 - target_depthmaps
        ed_inv = 1.0 - est_depthmaps

        # 2. 强制转为 3 通道 (B, 3, H, W)
        # 无论之前是 3D 还是 4D，都统一处理
        if td_inv.dim() == 3:
             td_rgb = td_inv.unsqueeze(1).repeat(1, 3, 1, 1)
        elif td_inv.dim() == 4 and td_inv.shape[1] == 1:
             td_rgb = td_inv.repeat(1, 3, 1, 1)
        else:
             td_rgb = td_inv # 假设已经是3通道
             
        if ed_inv.dim() == 3:
             ed_rgb = ed_inv.unsqueeze(1).repeat(1, 3, 1, 1)
        elif ed_inv.dim() == 4 and ed_inv.shape[1] == 1:
             ed_rgb = ed_inv.repeat(1, 3, 1, 1)
        else:
             ed_rgb = ed_inv
#         summary = torch.cat([captimgs, target_images, est_images, target_depthmaps, est_depthmaps], dim=-2)[
#                   :summary_max_images]
        # 3. 拼接 (使用刚才定义的变量)
        summary = torch.cat([captimgs, target_images, est_images, td_rgb, ed_rgb], dim=-2)[
                  :summary_max_images]
        grid_summary = torchvision.utils.make_grid(summary, nrow=summary_max_images)
        self.logger.experiment.add_image(f'{tag}/summary', grid_summary, self.global_step)
        if (self.hparams.optimize_optics or self.global_step == 0) and self.optical_model_type == 'legacy_camera':
            psf = self.camera.psf_at_camera(size=(128, 128), is_training=torch.tensor(False))
            psf = self.camera.normalize_psf(psf)
            psf = fftshift(crop_psf(psf, 64), dims=(-1, -2))
            psf_vis = psf / psf.view(psf.shape[0], psf.shape[1], -1).max(dim=-1, keepdim=True)[0].unsqueeze(-1)
            heightmap = imresize(self.camera.heightmap()[None, None, ...],
                                 [self.hparams.summary_mask_sz, self.hparams.summary_mask_sz]).squeeze(0)
            heightmap = (heightmap - heightmap.min()) / (heightmap.max() - heightmap.min())
            grid_psf = torchvision.utils.make_grid(
                psf_vis[vis_channels, ::self.hparams.summary_depth_every].transpose(0, 1), nrow=len(vis_channels),
                pad_value=1, normalize=False)
            self.logger.experiment.add_image('optics/psf_normalized_per_depth', grid_psf, self.global_step)
            self.logger.experiment.add_image('optics/heightmap', heightmap, self.global_step)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--optical_model', type=str, default='legacy_camera',
                            choices=['legacy_camera', 'dodo_depth'],
                            help='光学前向模型选择')
        parser.add_argument('--measurement_channels', type=int, default=None,
                            help='光学测量输出通道数；None=自动推断')
        parser.add_argument('--dodo_depth_layers', type=int, default=None,
                            help='DoDo 深度分层数；None=使用 n_depths')
        parser.add_argument('--depth_layering_mode', type=str, default='hard_depth',
                            choices=['hard_depth', 'hard_meter', 'soft_diopter'],
                            help='DoDo depth layering mode')
        parser.add_argument('--soft_diopter_eps', type=float, default=1e-8,
                            help='Soft diopter weight normalization epsilon')
        parser.add_argument('--soft_diopter_bandwidth_scale', type=float, default=1.0,
                            help='Soft diopter triangular bandwidth multiplier')
        parser.add_argument('--dodo_doe_type', type=str, default='Zeros',
                            help='DoDo DOE 类型（Zeros=frozen, New=trainable Zernike）')
        parser.add_argument('--dodo_use_second_doe', dest='dodo_use_second_doe', action='store_true',
                            help='启用 DoDo 第二 DOE')
        parser.add_argument('--no-dodo_use_second_doe', dest='dodo_use_second_doe', action='store_false')
        parser.set_defaults(dodo_use_second_doe=False)
        parser.add_argument('--summary_max_images', type=int, default=4)
        parser.add_argument('--summary_image_sz', type=int, default=256)
        parser.add_argument('--summary_mask_sz', type=int, default=256)
        parser.add_argument('--summary_depth_every', type=int, default=1)
        parser.add_argument('--summary_track_train_every', type=int, default=4000)
        parser.add_argument('--cnn_lr', type=float, default=1e-4)
        parser.add_argument('--optics_lr', type=float, default=1e-9)
        parser.add_argument('--batch_sz', type=int, default=2)
        parser.add_argument('--num_workers', type=int, default=8)
        parser.add_argument('--randcrop', default=False, action='store_true')
        parser.add_argument('--augment', default=False, action='store_true')
        parser.add_argument('--patch_filter', dest='patch_filter', action='store_true',
                            help='训练时对随机裁剪 patch 做质量筛选（轻量版：仅在 depth/mask 上重采样判定）')
        parser.add_argument('--no-patch_filter', dest='patch_filter', action='store_false',
                            help='关闭训练 patch 质量筛选')
        parser.set_defaults(patch_filter=True)
        parser.add_argument('--min_valid_ratio', type=float, default=0.12,
                            help='patch 中有效像素占比下限（0~1）')
        parser.add_argument('--min_depth_range_ips', type=float, default=0.10,
                            help='patch 内有效区域 IPS 深度动态范围下限')
        parser.add_argument('--max_crop_retries', type=int, default=8,
                            help='随机裁剪失败后最多重采样次数（轻量判定，开销较小）')
        parser.add_argument('--patch_filter_stride', type=int, default=4,
                            help='patch筛选预检步长(>1更快，=1最严格)')
        parser.add_argument('--depth_loss_weight', type=float, default=1.0)
        parser.add_argument('--image_loss_weight', type=float, default=1.0)
        parser.add_argument('--psf_loss_weight', type=float, default=1.0)
        parser.add_argument('--depth_smooth_weight', type=float, default=0.01,
                    help='深度平滑正则权重（抑制颜色纹理串扰）')
        parser.add_argument('--metric_depth_loss_weight', type=float, default=0.0,
                    help='掩码内 metric-depth SmoothL1 损失权重（默认 0=关闭）')
        parser.add_argument('--psf_size', type=int, default=64)
        parser.add_argument('--l1_loss_weight', type=float, default=1.0)
        parser.add_argument('--sam_loss_weight', type=float, default=0.5)
        parser.add_argument('--image_sz', type=int, default=512)
        parser.add_argument('--n_depths', type=int, default=8)
        parser.add_argument('--min_depth', type=float, default=0.4)
        parser.add_argument('--max_depth', type=float, default=2.0)
        parser.add_argument('--crop_width', type=int, default=32)
        parser.add_argument('--reg_tikhonov', type=float, default=1.0)
        parser.add_argument('--model_base_ch', type=int, default=32)
        # [ARCH-MOD-20260403] 深度分支浅层 skip 解耦模式。
        # 可选: lowpass / drop / full
        parser.add_argument('--depth_shallow_skip_mode', type=str, default='lowpass',
                    choices=['lowpass', 'drop', 'full'],
                    help='深度头最浅层skip注入策略：lowpass(默认), drop, full')
        parser.add_argument('--preinverse', dest='preinverse', action='store_true')
        parser.add_argument('--no-preinverse', dest='preinverse', action='store_false')
        parser.set_defaults(preinverse=True)
        parser.add_argument('--camera_type', type=str, default='mixed')
        parser.add_argument('--mask_sz', type=int, default=8000)
        parser.add_argument('--focal_length', type=float, default=50e-3)
        parser.add_argument('--focal_depth', type=float, default=0.67)
        parser.add_argument('--use_virtual_lens_phase', dest='use_virtual_lens_phase', action='store_true',
                    help='在 pupil 处叠加基类“理想薄透镜”相位（传统成像/对焦建模）。')
        parser.add_argument('--no-use_virtual_lens_phase', dest='use_virtual_lens_phase', action='store_false',
                    help='关闭基类“理想薄透镜”相位（Baek-like：DOE 充当唯一主透镜时推荐）。')
        parser.set_defaults(use_virtual_lens_phase=True)
        parser.add_argument('--f_number', type=float, default=6.3)
        parser.add_argument('--camera_pixel_pitch', type=float, default=6.45e-6)
        parser.add_argument('--noise_sigma_min', type=float, default=0.001)
        parser.add_argument('--noise_sigma_max', type=float, default=0.005)
        parser.add_argument('--full_size', type=int, default=1920)
        parser.add_argument('--mask_upsample_factor', type=int, default=10)
        parser.add_argument('--diffraction_efficiency', type=float, default=0.7)
        parser.add_argument('--occlusion', dest='occlusion', action='store_true')
        parser.add_argument('--no-occlusion', dest='occlusion', action='store_false')
        parser.set_defaults(occlusion=True)
        parser.add_argument('--optimize_optics', dest='optimize_optics', action='store_true')
        parser.add_argument('--no-optimize_optics', dest='optimize_optics', action='store_false')
        parser.set_defaults(optimize_optics=False)
        parser.add_argument('--psfjitter', dest='psf_jitter', action='store_true')
        parser.add_argument('--no-psfjitter', dest='psf_jitter', action='store_false')
        parser.set_defaults(psf_jitter=True)
        parser.add_argument('--hs_channels', type=int, default=25, help='高光谱数据的通道数')
        parser.add_argument('--start_wl', type=float, default=420e-9, help='起始波长（米, 例如 420nm）')
        parser.add_argument('--end_wl', type=float, default=660e-9, help='结束波长（米, 例如 660nm）')
        parser.add_argument('--bayer', dest='bayer', action='store_true')
        parser.add_argument('--no-bayer', dest='bayer', action='store_false')
        parser.set_defaults(bayer=False)
        parser.add_argument('--checkpoint_monitor', type=str, default='validation/psnr_hs_masked')
        parser.add_argument('--checkpoint_mode', type=str, default='max')
        parser.add_argument('--artifact_root', type=str, default='',
                            help='单一实验产物根目录；所有 metrics/PNG/logs 保存于此')
        parser.add_argument('--require_artifact_root', dest='require_artifact_root', action='store_true',
                            help='要求 artifact_root 必须非空；无法解析时 fail-fast')
        parser.add_argument('--no-require_artifact_root', dest='require_artifact_root', action='store_false')
        parser.set_defaults(require_artifact_root=False)
        parser.add_argument('--decoder_norm', type=str, default='batch',
                            choices=['batch', 'group'],
                            help='Decoder 归一化类型（batch=BatchNorm, group=GroupNorm）')
        parser.add_argument('--dodo_measurement_norm', type=str, default='none',
                            choices=['none', 'per_sample_mean_std', 'per_sample_minmax'],
                            help='DoDo 测量归一化模式（none/per_sample_mean_std/per_sample_minmax）')
        parser.add_argument('--dodo_nonfinite_policy', type=str, default='zero',
                            choices=['zero', 'fail'],
                            help='DoDo 非有限测量策略（zero=替换为0继续, fail=抛异常停止）')
        parser.add_argument('--dodo_forward_norm', type=str, default='legacy_max',
                            choices=['legacy_max', 'none', 'per_sample_max'],
                            help='DoDo forward internal measurement norm mode')
        parser.add_argument('--background_hs_loss_weight', type=float, default=0.0,
                            help='Background HS L1 loss weight (opt-in, default 0)')
        parser.add_argument('--dodo_sensing_mode', type=str, default='rgb',
                            choices=['rgb', 'spectral_bins', 'identity'],
                            help='DoDo sensing mode')
        parser.add_argument('--dodo_sensor_measurement', type=str, default='amplitude',
                            choices=['amplitude', 'intensity'],
                            help='DoDo sensor measurement type (amplitude=abs(field), intensity=abs(field)^2)')
        parser.add_argument('--decoder_use_depth_input', dest='decoder_use_depth_input',
                            action='store_true',
                            help='Enable decoder depth input channel (concat normalized depth to captimgs)')
        parser.add_argument('--no-decoder_use_depth_input', dest='decoder_use_depth_input',
                            action='store_false',
                            help='Disable decoder depth input (default)')
        parser.set_defaults(decoder_use_depth_input=False)
        parser.add_argument('--decoder_depth_input_mode', type=str, default='normalized_diopter',
                            choices=['normalized_z', 'normalized_diopter'],
                            help='Depth normalization mode for decoder depth input')
        return parser
