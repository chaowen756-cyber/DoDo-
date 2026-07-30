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
from util.psf_regularization import (
    delayed_epoch_tightening_value,
    delayed_epoch_warmup_weight,
    multiscale_psf_energy_concentration_loss,
    psf_mtf_floor_loss,
    sensor_weighted_depth_psf_separation_loss,
    sensor_weighted_spectral_psf_separation_loss,
    task_relative_regularizer_scale,
    zernike_order_weighted_l2,
)

# 导入项目原有的辅助工具
from solvers.image_reconstruction import apply_tikhonov_inverse
from util.fft import crop_psf, fftshift
from util.helper import crop_boundary, gray_to_rgb, imresize, ips_to_metric

SnapshotOutputs = namedtuple('SnapshotOutputs',
                             field_names=['captimgs', 'captimgs_linear',
                                          'est_images', 'est_depthmaps',
                                          'target_images', 'target_depthmaps',
                                          'psf'])

_VALIDATION_TOTAL_KEYS = (
    'depth_abs_sum',
    'depth_sq_sum',
    'depth_valid_count',
    'metric_depth_abs_sum',
    'metric_depth_batch_mae_sum',
    'hs_abs_sum',
    'hs_sq_sum',
    'hs_valid_count',
    'hs_full_sq_sum',
    'hs_full_count',
    'depth_tv_dx_sum',
    'depth_tv_dx_count',
    'depth_tv_dy_sum',
    'depth_tv_dy_count',
    'background_hs_abs_sum',
    'background_hs_count',
    'valid_batches',
    'skipped_batches',
)


def _all_reduce_validation_totals(totals, device):
    """Sum additive validation statistics across all active DDP ranks."""
    values = torch.tensor(
        [float(totals[key]) for key in _VALIDATION_TOTAL_KEYS],
        dtype=torch.float64,
        device=device,
    )
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM)
    return {
        key: float(value)
        for key, value in zip(_VALIDATION_TOTAL_KEYS, values.detach().cpu().tolist())
    }


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
        self._val_totals = {
            key: 0.0 for key in _VALIDATION_TOTAL_KEYS
        }
        # Diagnostics state
        self._doe_diag_done = False
        self._nonfinite_count = 0
        self._clamp_hook_count = 0
        self._doe_grad_norms = []
        self._last_train_loss_logs = {}
        self._last_train_misc_logs = {}
        self._zernike_schedule_hook_handle = None
        self._register_zernike_gradient_schedule()

    def _register_zernike_gradient_schedule(self):
        """Freeze then gently release high-order free-Zernike coefficients."""
        if not bool(getattr(self.hparams, 'optimize_optics', False)):
            return
        if getattr(self.hparams, 'dodo_zernike_mode', 'legacy12') != 'free':
            return
        doe1 = getattr(getattr(self, 'camera', None), 'doe1', None)
        coefficients = getattr(doe1, 'zernike_coeffs', None)
        if not isinstance(coefficients, nn.Parameter) or not coefficients.requires_grad:
            return
        protected_terms = int(getattr(
            self.hparams, 'dodo_zernike_low_order_terms', 15))
        unlock_epoch = int(getattr(
            self.hparams, 'dodo_zernike_high_order_unlock_epoch', 0))
        high_order_lr_ratio = float(getattr(
            self.hparams, 'dodo_zernike_high_order_lr_ratio', 1.0))
        if protected_terms < 0 or protected_terms > coefficients.numel():
            raise ValueError(
                'dodo_zernike_low_order_terms must be within the coefficient '
                f'count, got {protected_terms}/{coefficients.numel()}')
        if unlock_epoch < 0:
            raise ValueError('dodo_zernike_high_order_unlock_epoch must be >= 0')
        if not 0.0 <= high_order_lr_ratio <= 1.0:
            raise ValueError(
                'dodo_zernike_high_order_lr_ratio must be in [0,1]')

        def scale_high_order_gradient(gradient):
            if gradient is None or protected_terms >= gradient.numel():
                return gradient
            scaled = gradient.clone()
            current_epoch = int(self.current_epoch)
            ratio = (
                0.0 if current_epoch < unlock_epoch else high_order_lr_ratio)
            scaled[protected_terms:] *= ratio
            return scaled

        self._zernike_schedule_hook_handle = coefficients.register_hook(
            scale_high_order_gradient)
        print(
            '[doe_schedule] high-order Zernike gradient schedule: '
            f'protected_terms={protected_terms}, unlock_epoch={unlock_epoch}, '
            f'lr_ratio_after_unlock={high_order_lr_ratio:g}')

    def _build_rgb_pinv_prior_matrix(self, ridge_lambda):
        sensing = getattr(self.camera, 'sensing_unnorm', None)
        if sensing is None or getattr(sensing, 'sensing_mode', None) != 'rgb':
            raise ValueError('decoder RGB pinv prior requires dodo_sensing_mode="rgb"')
        if not all(hasattr(sensing, name) for name in ('sensor_r', 'sensor_g', 'sensor_b')):
            raise ValueError('RGB sensor response buffers are missing; cannot build pinv prior')

        response = torch.stack([
            sensing.sensor_r.to(dtype=torch.float32),
            sensing.sensor_g.to(dtype=torch.float32),
            sensing.sensor_b.to(dtype=torch.float32),
        ], dim=0)  # [3, 25], matching SensingLayer RGB collapse.
        if response.shape != (3, int(self.hparams.hs_channels)):
            raise ValueError(
                f'RGB response shape {tuple(response.shape)} is incompatible with '
                f'hs_channels={self.hparams.hs_channels}'
            )

        ridge = float(ridge_lambda)
        if ridge < 0:
            raise ValueError(f'decoder_rgb_pinv_lambda must be >= 0, got {ridge}')
        eye = torch.eye(response.shape[0], dtype=response.dtype, device=response.device)
        gram = response @ response.t()
        return response.t() @ torch.linalg.solve(gram + ridge * eye, eye)  # [25, 3]

    def _rgb_pinv_prior_from_measurement(self, captimgs):
        if captimgs.ndim != 4 or captimgs.shape[1] != 3:
            raise ValueError(
                f'RGB pinv prior expects captimgs with shape [B,3,H,W], got {tuple(captimgs.shape)}'
            )

        rgb = captimgs
        if getattr(self.hparams, 'decoder_rgb_pinv_unscale_measurement', True):
            forward_norm = getattr(self.hparams, 'dodo_forward_norm', 'legacy_max')
            if forward_norm == 'fixed_scale':
                scale = float(getattr(self.hparams, 'dodo_forward_scale', 1.0) or 1.0)
                rgb = rgb * scale

        pinv = self.rgb_pinv_prior_matrix.to(device=rgb.device, dtype=rgb.dtype)
        prior = torch.einsum('cm,bmhw->bchw', pinv, rgb)
        prior = torch.clamp(prior, min=0.0)

        norm_mode = getattr(self.hparams, 'decoder_rgb_pinv_norm', 'per_sample_max')
        eps = 1e-8
        if norm_mode == 'none':
            return prior
        if norm_mode == 'per_sample_max':
            denom = torch.amax(prior, dim=(1, 2, 3), keepdim=True)
            return prior / (denom + eps)
        if norm_mode == 'per_sample_mean_std':
            mean = prior.mean(dim=(1, 2, 3), keepdim=True)
            std = prior.std(dim=(1, 2, 3), keepdim=True)
            return (prior - mean) / (std + eps)
        raise ValueError(f'Unknown decoder_rgb_pinv_norm={norm_mode}')

    # =================================================================================
    # ## 以下是之前缺失的、从原始文件迁移过来的 PyTorch Lightning 核心方法 ##
    # =================================================================================

#     def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure=None, on_tpu=False,
#                        using_native_amp=False, using_lbfgs=False):
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx=0, optimizer_closure=None, **kwargs):
        lr_decay_strategy = str(getattr(self.hparams, 'lr_decay_strategy', 'none'))
        current_epoch = int(getattr(self.trainer, 'current_epoch', epoch))
        completed_step = int(self.trainer.global_step) + 1
        monitor_every = max(
            1, int(getattr(self.hparams, 'loss_plot_every_n_steps', 50)))
        monitor_doe = (
            bool(getattr(self.hparams, 'optimize_optics', False))
            and getattr(self, 'optical_model_type', None) == 'dodo_depth'
            and (completed_step == 1 or completed_step % monitor_every == 0)
        )
        doe_coeffs = None
        coeffs_before = None
        if monitor_doe:
            doe_layer = getattr(getattr(self, 'camera', None), 'doe1', None)
            candidate = getattr(doe_layer, 'zernike_coeffs', None)
            if isinstance(candidate, nn.Parameter) and candidate.requires_grad:
                doe_coeffs = candidate
                coeffs_before = candidate.detach().clone()

        warmup_steps = int(getattr(self.hparams, 'lr_warmup_steps', 54))
        warmup_scale = 1.0
        if warmup_steps > 0 and self.trainer.global_step < warmup_steps:
            warmup_scale = min(1., float(self.trainer.global_step + 1) / float(warmup_steps))

        for pg in optimizer.param_groups:
            if pg.get('name') == 'optics':
                base_lr = self.hparams.optics_lr
                decay_epochs = int(getattr(self.hparams, 'optics_lr_decay_epochs', 10))
            else:
                base_lr = self.hparams.cnn_lr
                decay_epochs = int(getattr(self.hparams, 'cnn_lr_decay_epochs', 20))

            decay_scale = 1.0
            if lr_decay_strategy == 'baek' and decay_epochs > 0:
                decay_scale = 0.1 ** (current_epoch // decay_epochs)
            pg['lr'] = warmup_scale * decay_scale * base_lr

        # Lightning 1.0.2 has already run training_step_and_backward before
        # entering this hook. Adam/SGD must therefore not execute the supplied
        # closure, otherwise every optimizer update repeats forward/backward
        # and accumulates a second gradient. LBFGS is the only optimizer used
        # here that requires the closure.
        if bool(kwargs.get('using_native_amp', False)):
            self.trainer.scaler.step(optimizer)
        elif isinstance(optimizer, torch.optim.LBFGS):
            optimizer.step(closure=optimizer_closure)
        else:
            optimizer.step()

        coeffs_raw = None
        grad_norm = None
        if doe_coeffs is not None:
            with torch.no_grad():
                coeffs_raw = doe_coeffs.detach().clone()
                if doe_coeffs.grad is not None:
                    grad_norm = float(
                        torch.linalg.vector_norm(
                            doe_coeffs.grad.detach()).item())

        if self.hparams.optimize_optics and hasattr(self.camera, 'clamp_parameters_'):
            self.camera.clamp_parameters_()
            self._clamp_hook_count += 1
            if self._clamp_hook_count == 1:
                print('[doe_diag] clamp_parameters_() executed (first call)')

        if (
                doe_coeffs is not None
                and coeffs_before is not None
                and coeffs_raw is not None):
            with torch.no_grad():
                coeffs_after = doe_coeffs.detach()
                raw_delta = coeffs_raw - coeffs_before
                effective_delta = coeffs_after - coeffs_before
                projection_delta = coeffs_after - coeffs_raw
                raw_update_norm = torch.linalg.vector_norm(raw_delta)
                effective_update_norm = torch.linalg.vector_norm(
                    effective_delta)
                projection_correction_norm = torch.linalg.vector_norm(
                    projection_delta)
                if raw_update_norm > 1e-20:
                    update_retention = (
                        effective_update_norm / raw_update_norm)
                    projection_fraction = (
                        projection_correction_norm / raw_update_norm)
                else:
                    update_retention = raw_update_norm.new_tensor(1.0)
                    projection_fraction = raw_update_norm.new_tensor(0.0)
                coeff_norm_before = torch.linalg.vector_norm(coeffs_before)
                coeff_norm_raw = torch.linalg.vector_norm(coeffs_raw)
                coeff_norm = torch.linalg.vector_norm(coeffs_after)
                update_rel = (
                    effective_update_norm
                    / coeff_norm_before.clamp_min(1e-12)
                )
                clamp_hit = float(
                    projection_correction_norm.item() > 1e-12)
                optics_lr = next(
                    (
                        float(pg['lr']) for pg in optimizer.param_groups
                        if pg.get('name') == 'optics'
                    ),
                    0.0,
                )

            self._last_doe_metrics = {
                'step': completed_step,
                'grad_norm': grad_norm,
                'raw_update_norm': float(raw_update_norm.item()),
                'effective_update_norm': float(
                    effective_update_norm.item()),
                'projection_correction_norm': float(
                    projection_correction_norm.item()),
                'update_retention': float(update_retention.item()),
                'projection_fraction': float(projection_fraction.item()),
                'update_rel': float(update_rel.item()),
                'clamp_hit': clamp_hit,
                'coeff_norm_before': float(coeff_norm_before.item()),
                'coeff_norm_raw': float(coeff_norm_raw.item()),
                'coeff_norm': float(coeff_norm.item()),
                'optics_lr': optics_lr,
            }
            for key, value in self._last_doe_metrics.items():
                if key == 'step' or value is None:
                    continue
                self.log(
                    f'doe/{key}', float(value),
                    on_step=True, on_epoch=False)

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

        if 'aug_clip_ratio' in samples:
            augmentation_metrics = {
                'augmentation/clip_ratio': samples['aug_clip_ratio'].float().mean(),
                'augmentation/scale_half_ratio': (
                    samples['aug_scale_factor'].float() < 0.75
                ).float().mean(),
                'augmentation/depth_shift_abs_m': samples[
                    'aug_depth_shift_m'
                ].float().abs().mean(),
                'augmentation/illuminant_ratio': samples[
                    'aug_illuminant_applied'
                ].float().mean(),
                'augmentation/illuminant_requested_ratio': samples[
                    'aug_illuminant_requested'
                ].float().mean(),
                'augmentation/illuminant_fallback_ratio': samples[
                    'aug_illuminant_fallback'
                ].float().mean(),
                'augmentation/illuminant_attempts': samples[
                    'aug_illuminant_attempts'
                ].float().mean(),
                'augmentation/exposure': samples['aug_exposure'].float().mean(),
            }
            for metric_name, metric_value in augmentation_metrics.items():
                self.log(
                    metric_name,
                    metric_value,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=False,
                )

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

        optical_images = samples.get('hs_optical')
        optical_depth_metric = samples.get('depth_metric_optical')
        optical_valid_mask = samples.get('mask_optical')
        if optical_depth_metric is not None and optical_depth_metric.ndim == 4:
            optical_depth_metric = optical_depth_metric.squeeze(1)
        if optical_valid_mask is not None and optical_valid_mask.ndim == 4:
            optical_valid_mask = optical_valid_mask.squeeze(1)

        outputs = self.forward(target_images, target_depthmaps, is_testing=torch.tensor(False),
                               depth_metric=depth_metric, valid_mask=valid_mask,
                               optical_images=optical_images,
                               optical_depth_metric=optical_depth_metric,
                               optical_valid_mask=optical_valid_mask)

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
        if self.optical_model_type == 'dodo_depth':
            capture_fraction = getattr(
                self.camera, 'psf_capture_fraction', None)
            if capture_fraction is not None:
                misc_logs.update({
                    'optics/psf_capture_min': capture_fraction.min(),
                    'optics/psf_capture_mean': capture_fraction.mean(),
                    'optics/psf_capture_max': capture_fraction.max(),
                })
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
                if getattr(
                        self.hparams, 'dodo_zernike_mode', 'legacy12') == 'free':
                    low_terms = int(getattr(
                        self.hparams, 'dodo_zernike_low_order_terms', 15))
                else:
                    low_terms = zc.numel()
                grad_norms['doe_zernike_low_order'] = (
                    zc.grad[:low_terms].norm().item())
                grad_norms['doe_zernike_high_order'] = (
                    zc.grad[low_terms:].norm().item())
            else:
                grad_norms['doe_zernike'] = 0.0
                grad_norms['doe_zernike_finite'] = 0.0
                grad_norms['doe_zernike_low_order'] = 0.0
                grad_norms['doe_zernike_high_order'] = 0.0
        # Log all grad norms
        for k, v in grad_norms.items():
            self.log(f'diag/grad_{k}', v if isinstance(v, float) else float(v), on_step=True)
        # Persist latest grad norms for metrics.json
        self._last_grad_norms = grad_norms


    def _trainer_is_global_zero(self):
        trainer = getattr(self, 'trainer', None)
        if trainer is None:
            return True
        if hasattr(trainer, 'is_global_zero'):
            return bool(trainer.is_global_zero)
        return int(getattr(trainer, 'global_rank', 0) or 0) == 0

    def on_validation_epoch_start(self) -> None:
        for metric in self.metrics.values():
            metric.reset()
            metric.to(self.device)
        self._val_totals = {
            key: 0.0 for key in _VALIDATION_TOTAL_KEYS
        }

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

        optical_images = samples.get('hs_optical')
        optical_depth_metric = samples.get('depth_metric_optical')
        optical_valid_mask = samples.get('mask_optical')
        if optical_depth_metric is not None and optical_depth_metric.ndim == 4:
            optical_depth_metric = optical_depth_metric.squeeze(1)
        if optical_valid_mask is not None and optical_valid_mask.ndim == 4:
            optical_valid_mask = optical_valid_mask.squeeze(1)

        outputs = self.forward(target_images, target_depthmaps, is_testing=torch.tensor(False),
                               depth_metric=depth_metric, valid_mask=valid_mask,
                               optical_images=optical_images,
                               optical_depth_metric=optical_depth_metric,
                               optical_valid_mask=optical_valid_mask)

        est = outputs.est_depthmaps
        tgt = outputs.target_depthmaps

        # Accumulate raw additive statistics. Ratios and PSNR are deliberately
        # deferred until all DDP ranks have contributed at epoch end.
        diff = torch.abs(est - tgt) * final_mask
        diff_sq = (est - tgt)**2 * final_mask
        depth_abs_sum = diff.sum()
        depth_sq_sum = diff_sq.sum()
        num_valid_px = final_mask.sum()

        # Metric-depth MAE (meters)
        if num_valid_px > 0:
            est_m = ips_to_metric(est.clamp(0, 1), self.hparams.min_depth, self.hparams.max_depth)
            tgt_m = ips_to_metric(tgt.clamp(0, 1), self.hparams.min_depth, self.hparams.max_depth)
            abs_depth_err_m = (torch.abs(est_m - tgt_m) * final_mask).sum()
            mae_depth_m = abs_depth_err_m / num_valid_px
        else:
            abs_depth_err_m = torch.tensor(0.0, device=est.device)
            mae_depth_m = torch.tensor(float('nan'), device=est.device)

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
        hs_error = est_images - target_images_val
        hs_abs_sum = (torch.abs(hs_error) * mask4d).sum()
        hs_sq_sum = (hs_error.square() * mask4d).sum()

        # Full-image PSNR (for reference)
        hs_full_sq_sum = hs_error.square().sum()
        hs_full_count = hs_error.numel()

        # Baek-style depth TV on inverse-depth/IPS predictions, masked to valid neighbors.
        dx = torch.abs(est[:, :, 1:] - est[:, :, :-1])
        dy = torch.abs(est[:, 1:, :] - est[:, :-1, :])
        mask_dx = final_mask[:, :, 1:] * final_mask[:, :, :-1]
        mask_dy = final_mask[:, 1:, :] * final_mask[:, :-1, :]
        depth_tv_dx_sum = (dx * mask_dx).sum()
        depth_tv_dx_count = mask_dx.sum()
        depth_tv_dy_sum = (dy * mask_dy).sum()
        depth_tv_dy_count = mask_dy.sum()

        # Background HS L1 for full-image visual quality; opt-in via background_hs_loss_weight.
        bg_mask4d = (1.0 - final_mask).unsqueeze(1)
        n_bg_hs = bg_mask4d.sum() * est_images.shape[1]
        background_hs_abs_sum = (torch.abs(hs_error) * bg_mask4d).sum()

        batch_totals = {
            'depth_abs_sum': depth_abs_sum,
            'depth_sq_sum': depth_sq_sum,
            'depth_valid_count': num_valid_px,
            'metric_depth_abs_sum': abs_depth_err_m,
            'metric_depth_batch_mae_sum': (
                mae_depth_m if num_valid_px > 0
                else torch.zeros_like(abs_depth_err_m)
            ),
            'hs_abs_sum': hs_abs_sum,
            'hs_sq_sum': hs_sq_sum,
            'hs_valid_count': n_valid_hs,
            'hs_full_sq_sum': hs_full_sq_sum,
            'hs_full_count': hs_full_count,
            'depth_tv_dx_sum': depth_tv_dx_sum,
            'depth_tv_dx_count': depth_tv_dx_count,
            'depth_tv_dy_sum': depth_tv_dy_sum,
            'depth_tv_dy_count': depth_tv_dy_count,
            'background_hs_abs_sum': background_hs_abs_sum,
            'background_hs_count': n_bg_hs,
            'valid_batches': 1.0 if num_valid_px > 0 else 0.0,
            'skipped_batches': 0.0 if num_valid_px > 0 else 1.0,
        }
        for key, value in batch_totals.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().to(torch.float64).item()
            self._val_totals[key] += float(value)

        # Save first valid batch outputs for artifact PNG generation
        if batch_idx == 0 and self._trainer_is_global_zero():
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
        totals = _all_reduce_validation_totals(
            self._val_totals, device=self.device)
        depth_count = totals['depth_valid_count']
        hs_count = totals['hs_valid_count']
        if depth_count <= 0.0 or hs_count <= 0.0:
            raise RuntimeError(
                'Validation has no globally valid foreground pixels; '
                'cannot compute model-selection metrics.')

        avg_mae_depth = totals['depth_abs_sum'] / depth_count
        avg_mse_depth = totals['depth_sq_sum'] / depth_count
        avg_mae_depth_m = totals['metric_depth_abs_sum'] / depth_count
        valid_batches = totals['valid_batches']
        avg_mae_depth_m_batch = (
            totals['metric_depth_batch_mae_sum'] / valid_batches
            if valid_batches > 0.0 else float('nan')
        )
        avg_hs_l1_masked = totals['hs_abs_sum'] / hs_count
        mse_hs_masked = totals['hs_sq_sum'] / hs_count
        avg_psnr_hs = 10.0 * np.log10(
            1.0 / (mse_hs_masked + 1e-10))

        hs_full_count = totals['hs_full_count']
        mse_hs_full = (
            totals['hs_full_sq_sum'] / hs_full_count
            if hs_full_count > 0.0 else float('nan')
        )
        psnr_hs_full = (
            10.0 * np.log10(1.0 / (mse_hs_full + 1e-10))
            if np.isfinite(mse_hs_full) else float('nan')
        )
        depth_tv_dx = (
            totals['depth_tv_dx_sum'] / totals['depth_tv_dx_count']
            if totals['depth_tv_dx_count'] > 0.0 else 0.0
        )
        depth_tv_dy = (
            totals['depth_tv_dy_sum'] / totals['depth_tv_dy_count']
            if totals['depth_tv_dy_count'] > 0.0 else 0.0
        )
        avg_depth_tv = 0.5 * (depth_tv_dx + depth_tv_dy)
        avg_bg_hs_l1 = (
            totals['background_hs_abs_sum']
            / totals['background_hs_count']
            if totals['background_hs_count'] > 0.0 else 0.0
        )
        val_loss = (
            self.hparams.image_loss_weight * avg_hs_l1_masked +
            self.hparams.depth_loss_weight * avg_mae_depth +
            float(getattr(self.hparams, 'depth_smooth_weight', 0.0)) * avg_depth_tv +
            float(getattr(self.hparams, 'background_hs_loss_weight', 0.0)) * avg_bg_hs_l1
        )
        global_metrics = {
            'val_loss': val_loss,
            'validation/mae_depthmap': avg_mae_depth,
            'validation/mse_depthmap': avg_mse_depth,
            'validation/mae_depth_m': avg_mae_depth_m,
            'validation/mae_depth_m_batch_avg': avg_mae_depth_m_batch,
            'validation/hs_l1_masked': avg_hs_l1_masked,
            'validation/psnr_hs_masked': avg_psnr_hs,
            'validation/psnr_hs_full': psnr_hs_full,
            'validation/depth_tv': avg_depth_tv,
            'validation/hs_l1_background': avg_bg_hs_l1,
        }
        for name, value in global_metrics.items():
            self.log(
                name,
                torch.tensor(value, dtype=torch.float32, device=self.device),
                on_step=False,
                on_epoch=True,
                sync_dist=False,
            )
        extra = {
            **global_metrics,
            'validation/mae_depth_m_valid_px': depth_count,
            'validation/hs_valid_values': hs_count,
            'validation/val_steps': valid_batches,
            'validation/skipped_steps': totals['skipped_batches'],
            'train_misc/nonfinite_count': self._nonfinite_count,
        }
        last_grads = getattr(self, '_last_grad_norms', None)
        if last_grads:
            for k, v in last_grads.items():
                extra[f'diag/grad_{k}'] = v if isinstance(v, float) else float(v)
        last_doe_metrics = getattr(self, '_last_doe_metrics', None)
        if last_doe_metrics:
            for key, value in last_doe_metrics.items():
                if key == 'step' or value is None:
                    continue
                extra[f'doe/{key}'] = float(value)
        capture_fraction = getattr(
            getattr(self, 'camera', None), 'psf_capture_fraction', None)
        if capture_fraction is not None:
            capture_fraction = capture_fraction.detach()
            capture_metrics = {
                'optics/psf_capture_min': float(
                    capture_fraction.min().item()),
                'optics/psf_capture_mean': float(
                    capture_fraction.mean().item()),
                'optics/psf_capture_max': float(
                    capture_fraction.max().item()),
            }
            extra.update(capture_metrics)
            for name, value in capture_metrics.items():
                self.log(
                    name,
                    torch.tensor(
                        value, dtype=torch.float32, device=self.device),
                    on_step=False,
                    on_epoch=True,
                    sync_dist=False,
                )
        # Save artifacts (prefer artifact_root over log_dir)
        out_dir = self.artifact_root or self.log_dir
        if out_dir and self._trainer_is_global_zero():
            self._save_validation_artifacts(extra, out_dir)

    def _save_validation_artifacts(self, extra=None, out_dir=None):
        import json, os
        if not self._trainer_is_global_zero():
            return
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
            ('image_loss_mse', 'train_loss/image_loss_mse'),
            ('image_loss_sam', 'train_loss/image_loss_sam'),
            ('image_loss_gradient', 'train_loss/image_loss_gradient'),
            ('metric_depth_loss', 'train_loss/metric_depth_loss'),
            ('psf_loss', 'train_loss/psf_loss'),
            ('psf_loss_weight', 'train_loss/psf_loss_weight'),
            ('psf_loss_effective_weight',
             'train_loss/psf_loss_effective_weight'),
            ('psf_loss_weighted', 'train_loss/psf_loss_weighted'),
            ('psf_out_of_fov_max', 'train_loss/psf_out_of_fov_max'),
            ('psf_energy_outside_mean', 'train_loss/psf_energy_outside_mean'),
            ('psf_energy_outside_p90', 'train_loss/psf_energy_outside_p90'),
            ('psf_energy_inside_mean', 'train_loss/psf_energy_inside_mean'),
            ('psf_energy_active_fraction', 'train_loss/psf_energy_active_fraction'),
            ('psf_energy_captured_mean',
             'train_loss/psf_energy_captured_mean'),
            ('psf_energy_missing_mean',
             'train_loss/psf_energy_missing_mean'),
            ('psf_energy_outer_inside_mean',
             'train_loss/psf_energy_outer_inside_mean'),
            ('psf_energy_outer_outside_p90',
             'train_loss/psf_energy_outer_outside_p90'),
            ('psf_energy_r50_mean', 'train_loss/psf_energy_r50_mean'),
            ('psf_energy_r80_mean', 'train_loss/psf_energy_r80_mean'),
            ('psf_energy_r90_mean', 'train_loss/psf_energy_r90_mean'),
            ('psf_energy_r90_p90', 'train_loss/psf_energy_r90_p90'),
            ('psf_energy_r90_max', 'train_loss/psf_energy_r90_max'),
            ('psf_energy_r90_unresolved_fraction',
             'train_loss/psf_energy_r90_unresolved_fraction'),
            ('psf_energy_core_budget', 'train_loss/psf_energy_core_budget'),
            ('psf_energy_outer_budget', 'train_loss/psf_energy_outer_budget'),
            ('psf_mtf_loss', 'train_loss/psf_mtf_loss'),
            ('psf_mtf_effective_weight',
             'train_loss/psf_mtf_effective_weight'),
            ('psf_mtf_weighted', 'train_loss/psf_mtf_weighted'),
            ('psf_mtf_005_mean', 'train_loss/psf_mtf_005_mean'),
            ('psf_mtf_005_p10', 'train_loss/psf_mtf_005_p10'),
            ('psf_mtf_010_mean', 'train_loss/psf_mtf_010_mean'),
            ('psf_mtf_010_p10', 'train_loss/psf_mtf_010_p10'),
            ('psf_mtf_020_mean', 'train_loss/psf_mtf_020_mean'),
            ('psf_spectral_separation_loss',
             'train_loss/psf_spectral_separation_loss'),
            ('psf_spectral_separation_weight',
             'train_loss/psf_spectral_separation_weight'),
            ('psf_spectral_separation_effective_weight',
             'train_loss/psf_spectral_separation_effective_weight'),
            ('psf_spectral_separation_weighted',
             'train_loss/psf_spectral_separation_weighted'),
            ('psf_spectral_adjacent_cosine_mean',
             'train_loss/psf_spectral_adjacent_cosine_mean'),
            ('psf_spectral_adjacent_cosine_p90',
             'train_loss/psf_spectral_adjacent_cosine_p90'),
            ('psf_spectral_adjacent_cosine_max',
             'train_loss/psf_spectral_adjacent_cosine_max'),
            ('psf_spectral_active_fraction',
             'train_loss/psf_spectral_active_fraction'),
            ('psf_depth_separation_loss',
             'train_loss/psf_depth_separation_loss'),
            ('psf_depth_separation_effective_weight',
             'train_loss/psf_depth_separation_effective_weight'),
            ('psf_depth_separation_weighted',
             'train_loss/psf_depth_separation_weighted'),
            ('psf_depth_adjacent_cosine_mean',
             'train_loss/psf_depth_adjacent_cosine_mean'),
            ('psf_depth_adjacent_cosine_p90',
             'train_loss/psf_depth_adjacent_cosine_p90'),
            ('psf_depth_adjacent_cosine_max',
             'train_loss/psf_depth_adjacent_cosine_max'),
            ('zernike_high_order_loss', 'train_loss/zernike_high_order_loss'),
            ('zernike_high_order_effective_weight',
             'train_loss/zernike_high_order_effective_weight'),
            ('zernike_high_order_weighted',
             'train_loss/zernike_high_order_weighted'),
            ('zernike_low_order_norm', 'train_loss/zernike_low_order_norm'),
            ('zernike_high_order_norm', 'train_loss/zernike_high_order_norm'),
            ('task_loss_weighted', 'train_loss/task_loss_weighted'),
            ('optical_regularizer_raw',
             'train_loss/optical_regularizer_raw'),
            ('optical_regularizer_scale',
             'train_loss/optical_regularizer_scale'),
            ('optical_regularizer_weighted',
             'train_loss/optical_regularizer_weighted'),
            ('optical_regularizer_ratio',
             'train_loss/optical_regularizer_ratio'),
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
                print(
                    '[dodo_depth] legacy psf_loss_weight forced to 0.0; '
                    'DoDo PSF convolution uses dodo_psf_energy_weight instead')
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
            dodo_zernike_mode = getattr(hparams, 'dodo_zernike_mode', 'legacy12')
            dodo_zernike_terms = int(getattr(hparams, 'dodo_zernike_terms', 150))
            dodo_zernike_basis_path = getattr(
                hparams, 'dodo_zernike_basis_path', None)
            dodo_doe_basis_mode = getattr(
                hparams, 'dodo_doe_basis_mode', 'legacy_raw12')
            dodo_doe_basis_rank = int(
                getattr(hparams, 'dodo_doe_basis_rank', 9))
            dodo_doe_basis_rank_rtol = float(
                getattr(hparams, 'dodo_doe_basis_rank_rtol', 1e-4))
            dodo_doe_basis_rms_m = float(
                getattr(hparams, 'dodo_doe_basis_rms_m', 3e-6))
            dodo_doe_coeff_norm_limit = float(
                getattr(hparams, 'dodo_doe_coeff_norm_limit', 1.0))
            dodo_doe_init_coeff_norm = float(
                getattr(hparams, 'dodo_doe_init_coeff_norm', 0.2))
            if dodo_zernike_mode not in ('legacy12', 'free'):
                raise ValueError(
                    'dodo_zernike_mode must be legacy12 or free, got '
                    f'{dodo_zernike_mode!r}')
            if dodo_doe_basis_mode not in ('legacy_raw12', 'orthogonal_rms'):
                raise ValueError(
                    'dodo_doe_basis_mode must be legacy_raw12 or '
                    f'orthogonal_rms, got {dodo_doe_basis_mode!r}')
            if (dodo_zernike_mode == 'free'
                    and dodo_doe_basis_mode != 'legacy_raw12'):
                raise ValueError(
                    'orthogonal_rms applies only to '
                    '--dodo_zernike_mode legacy12')
            if dodo_zernike_terms < 1:
                raise ValueError(
                    f'dodo_zernike_terms must be >= 1, got {dodo_zernike_terms}')
            if dodo_zernike_basis_path and dodo_zernike_mode != 'free':
                raise ValueError(
                    'dodo_zernike_basis_path is only valid when '
                    '--dodo_zernike_mode free is selected')
            dodo_use_free_zernike = dodo_zernike_mode == 'free'
            dodo_forward_norm = getattr(hparams, 'dodo_forward_norm', 'legacy_max')
            dodo_forward_scale = float(getattr(hparams, 'dodo_forward_scale', 1.0))
            dodo_sensing_mode = getattr(hparams, 'dodo_sensing_mode', 'rgb')
            dodo_skip_prop2 = bool(getattr(hparams, 'dodo_skip_prop2', False))
            # Historical checkpoints do not contain this field and must retain
            # their original unpadded optical forward when loaded by new code.
            dodo_prop1_padding_factor = int(
                getattr(hparams, 'dodo_prop1_padding_factor', 1)
            )
            if dodo_prop1_padding_factor < 1:
                raise ValueError(
                    'dodo_prop1_padding_factor must be >= 1, '
                    f'got {dodo_prop1_padding_factor}'
                )
            depth_layering_mode = getattr(hparams, 'depth_layering_mode', 'hard_depth')
            soft_diopter_eps = getattr(hparams, 'soft_diopter_eps', 1e-8)
            soft_diopter_bandwidth_scale = getattr(hparams, 'soft_diopter_bandwidth_scale', 1.0)
            dodo_sensor_measurement = getattr(hparams, 'dodo_sensor_measurement', 'amplitude')
            dodo_image_formation = getattr(hparams, 'dodo_image_formation', 'whole_field')
            dodo_psf_optics_version = str(getattr(
                hparams, 'dodo_psf_optics_version', 'legacy'))
            dodo_psf_layer_mask = getattr(hparams, 'dodo_psf_layer_mask', 'baek_hard')
            dodo_psf_mask_blur_sigma = float(
                getattr(hparams, 'dodo_psf_mask_blur_sigma', 1.0))
            dodo_psf_boundary = getattr(hparams, 'dodo_psf_boundary', 'linear_zero')
            dodo_psf_depth_chunk_size = int(
                getattr(hparams, 'dodo_psf_depth_chunk_size', 1))
            if dodo_psf_depth_chunk_size < 1:
                raise ValueError(
                    'dodo_psf_depth_chunk_size must be >= 1, '
                    f'got {dodo_psf_depth_chunk_size}')
            dodo_psf_energy_weight = float(
                getattr(hparams, 'dodo_psf_energy_weight', 0.02))
            dodo_psf_energy_radius = float(
                getattr(hparams, 'dodo_psf_energy_radius', 16.0))
            dodo_psf_energy_outside_budget = float(
                getattr(hparams, 'dodo_psf_energy_outside_budget', 0.20))
            dodo_psf_energy_softness = float(
                getattr(hparams, 'dodo_psf_energy_softness', 1.5))
            dodo_psf_energy_warmup_epochs = int(
                getattr(hparams, 'dodo_psf_energy_warmup_epochs', 2))
            dodo_psf_energy_start_epoch = int(
                getattr(hparams, 'dodo_psf_energy_start_epoch', 0))
            dodo_psf_energy_outer_radius = float(getattr(
                hparams, 'dodo_psf_energy_outer_radius', 24.0))
            dodo_psf_energy_outer_outside_budget = float(getattr(
                hparams, 'dodo_psf_energy_outer_outside_budget', 0.05))
            dodo_psf_energy_initial_outside_budget = float(getattr(
                hparams, 'dodo_psf_energy_initial_outside_budget', 0.35))
            dodo_psf_energy_initial_outer_outside_budget = float(getattr(
                hparams,
                'dodo_psf_energy_initial_outer_outside_budget', 0.15))
            dodo_psf_energy_tightening_epochs = int(getattr(
                hparams, 'dodo_psf_energy_tightening_epochs', 3))
            dodo_psf_energy_tightening_start_epoch = int(getattr(
                hparams, 'dodo_psf_energy_tightening_start_epoch', 0))
            dodo_psf_energy_cvar_fraction = float(getattr(
                hparams, 'dodo_psf_energy_cvar_fraction', 0.10))
            dodo_psf_energy_cvar_weight = float(getattr(
                hparams, 'dodo_psf_energy_cvar_weight', 0.5))
            dodo_psf_energy_penalty_power = float(getattr(
                hparams, 'dodo_psf_energy_penalty_power', 2.0))
            # Missing fields identify historical checkpoints.  They retain the
            # old 128-only optical path and no spectral regularizer.  The
            # Historical checkpoints have no halo/separation fields and retain
            # their original zero-context behavior.
            dodo_optical_halo = int(getattr(hparams, 'dodo_optical_halo', 0))
            dodo_psf_spectral_separation_weight = float(getattr(
                hparams, 'dodo_psf_spectral_separation_weight', 0.0))
            dodo_psf_spectral_separation_margin = float(getattr(
                hparams, 'dodo_psf_spectral_separation_margin', 0.90))
            dodo_psf_spectral_separation_warmup_epochs = int(getattr(
                hparams, 'dodo_psf_spectral_separation_warmup_epochs', 2))
            dodo_psf_spectral_separation_start_epoch = int(getattr(
                hparams, 'dodo_psf_spectral_separation_start_epoch', 0))
            dodo_psf_spectral_hard_fraction = float(getattr(
                hparams, 'dodo_psf_spectral_hard_fraction', 0.20))
            dodo_psf_spectral_hard_weight = float(getattr(
                hparams, 'dodo_psf_spectral_hard_weight', 0.5))
            dodo_psf_depth_separation_weight = float(getattr(
                hparams, 'dodo_psf_depth_separation_weight', 0.0))
            dodo_psf_depth_separation_margin = float(getattr(
                hparams, 'dodo_psf_depth_separation_margin', 0.90))
            dodo_psf_depth_separation_start_epoch = int(getattr(
                hparams, 'dodo_psf_depth_separation_start_epoch', 0))
            dodo_psf_depth_separation_warmup_epochs = int(getattr(
                hparams, 'dodo_psf_depth_separation_warmup_epochs', 0))
            dodo_psf_mtf_weight = float(getattr(
                hparams, 'dodo_psf_mtf_weight', 0.0))
            dodo_psf_mtf_start_epoch = int(getattr(
                hparams, 'dodo_psf_mtf_start_epoch', 0))
            dodo_psf_mtf_warmup_epochs = int(getattr(
                hparams, 'dodo_psf_mtf_warmup_epochs', 0))
            dodo_optical_regularizer_max_ratio = float(getattr(
                hparams, 'dodo_optical_regularizer_max_ratio', 0.0))
            dodo_zernike_coefficient_limit = float(getattr(
                hparams, 'dodo_zernike_coefficient_limit', 1.0))
            if dodo_psf_energy_weight < 0:
                raise ValueError('dodo_psf_energy_weight must be >= 0')
            if dodo_psf_energy_radius <= 0:
                raise ValueError('dodo_psf_energy_radius must be > 0')
            if not 0.0 <= dodo_psf_energy_outside_budget <= 1.0:
                raise ValueError('dodo_psf_energy_outside_budget must be in [0, 1]')
            if dodo_psf_energy_softness < 0:
                raise ValueError('dodo_psf_energy_softness must be >= 0')
            if dodo_psf_energy_warmup_epochs < 0 or dodo_psf_energy_start_epoch < 0:
                raise ValueError('dodo_psf_energy_warmup_epochs must be >= 0')
            if dodo_psf_energy_outer_radius <= dodo_psf_energy_radius:
                raise ValueError(
                    'dodo_psf_energy_outer_radius must exceed the core radius')
            for name, value in (
                ('dodo_psf_energy_outer_outside_budget',
                 dodo_psf_energy_outer_outside_budget),
                ('dodo_psf_energy_initial_outside_budget',
                 dodo_psf_energy_initial_outside_budget),
                ('dodo_psf_energy_initial_outer_outside_budget',
                 dodo_psf_energy_initial_outer_outside_budget),
            ):
                if not 0.0 <= value <= 1.0:
                    raise ValueError(f'{name} must be in [0,1]')
            if (dodo_psf_energy_tightening_epochs < 0
                    or dodo_psf_energy_tightening_start_epoch < 0):
                raise ValueError(
                    'PSF energy tightening epochs/start epoch must be >= 0')
            if not 0.0 < dodo_psf_energy_cvar_fraction <= 1.0:
                raise ValueError(
                    'dodo_psf_energy_cvar_fraction must be in (0,1]')
            if dodo_psf_energy_cvar_weight < 0:
                raise ValueError('dodo_psf_energy_cvar_weight must be >= 0')
            if not 1.0 <= dodo_psf_energy_penalty_power <= 2.0:
                raise ValueError(
                    'dodo_psf_energy_penalty_power must be in [1,2]')
            if dodo_optical_halo < 0:
                raise ValueError('dodo_optical_halo must be >= 0')
            if dodo_optical_halo > 0 and dodo_image_formation != 'psf_convolution':
                raise ValueError(
                    'dodo_optical_halo is only supported with '
                    '--dodo_image_formation psf_convolution')
            if dodo_psf_spectral_separation_weight < 0:
                raise ValueError(
                    'dodo_psf_spectral_separation_weight must be >= 0')
            if not -1.0 <= dodo_psf_spectral_separation_margin <= 1.0:
                raise ValueError(
                    'dodo_psf_spectral_separation_margin must be in [-1, 1]')
            if (dodo_psf_spectral_separation_warmup_epochs < 0
                    or dodo_psf_spectral_separation_start_epoch < 0):
                raise ValueError(
                    'spectral separation start/warmup epochs must be >= 0')
            if not 0.0 < dodo_psf_spectral_hard_fraction <= 1.0:
                raise ValueError(
                    'dodo_psf_spectral_hard_fraction must be in (0,1]')
            if dodo_psf_spectral_hard_weight < 0:
                raise ValueError(
                    'dodo_psf_spectral_hard_weight must be >= 0')
            if dodo_psf_depth_separation_weight < 0:
                raise ValueError(
                    'dodo_psf_depth_separation_weight must be >= 0')
            if not -1.0 <= dodo_psf_depth_separation_margin <= 1.0:
                raise ValueError(
                    'dodo_psf_depth_separation_margin must be in [-1,1]')
            if (dodo_psf_depth_separation_start_epoch < 0
                    or dodo_psf_depth_separation_warmup_epochs < 0):
                raise ValueError(
                    'depth separation start/warmup epochs must be >= 0')
            if dodo_psf_mtf_weight < 0:
                raise ValueError('dodo_psf_mtf_weight must be >= 0')
            if dodo_psf_mtf_start_epoch < 0 or dodo_psf_mtf_warmup_epochs < 0:
                raise ValueError('MTF start/warmup epochs must be >= 0')
            if dodo_optical_regularizer_max_ratio < 0:
                raise ValueError(
                    'dodo_optical_regularizer_max_ratio must be >= 0')
            if dodo_zernike_coefficient_limit <= 0:
                raise ValueError(
                    'dodo_zernike_coefficient_limit must be > 0')
            if (dodo_psf_spectral_separation_weight > 0
                    and dodo_sensing_mode != 'rgb'):
                raise ValueError(
                    'The sensor-weighted spectral PSF separation loss currently '
                    'requires --dodo_sensing_mode rgb')
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
                free=dodo_use_free_zernike,
                n_terms=dodo_zernike_terms,
                zernike_basis_path=dodo_zernike_basis_path,
                input_format='nchw',
                output_format='nchw',
                measurement_norm_mode=dodo_forward_norm,
                measurement_norm_scale=dodo_forward_scale,
                sensing_mode=dodo_sensing_mode,
                measurement_channels=int(hparams.measurement_channels),
                depth_layering_mode=depth_layering_mode,
                soft_diopter_eps=soft_diopter_eps,
                soft_diopter_bandwidth_scale=soft_diopter_bandwidth_scale,
                sensor_measurement=dodo_sensor_measurement,
                skip_prop2=dodo_skip_prop2,
                prop1_padding_factor=dodo_prop1_padding_factor,
                image_formation_mode=dodo_image_formation,
                psf_layer_mask_mode=dodo_psf_layer_mask,
                psf_mask_blur_sigma=dodo_psf_mask_blur_sigma,
                psf_boundary_mode=dodo_psf_boundary,
                psf_depth_chunk_size=dodo_psf_depth_chunk_size,
                doe_basis_mode=dodo_doe_basis_mode,
                doe_basis_rank=dodo_doe_basis_rank,
                doe_basis_rank_rtol=dodo_doe_basis_rank_rtol,
                doe_basis_rms_m=dodo_doe_basis_rms_m,
                doe_coeff_norm_limit=dodo_doe_coeff_norm_limit,
                doe_init_coeff_norm=dodo_doe_init_coeff_norm,
                psf_optics_version=dodo_psf_optics_version,
            )
            if hasattr(self.camera.doe1, 'coefficient_limit'):
                self.camera.doe1.coefficient_limit = (
                    dodo_zernike_coefficient_limit)
            else:
                setattr(
                    self.camera.doe1, 'coefficient_limit',
                    dodo_zernike_coefficient_limit)
            if dodo_use_free_zernike:
                dodo_basis_summary = 'doe_basis_mode=n/a(free), '
            else:
                dodo_basis_summary = (
                    f'doe_basis_mode={dodo_doe_basis_mode}, '
                    f'doe_basis_rank={self.camera.doe1.zernike_basis.shape[0]}, '
                    f'doe_basis_rms={dodo_doe_basis_rms_m:.3g}m, '
                    f'doe_coeff_norm_limit={dodo_doe_coeff_norm_limit:g}, '
                    f'doe_init_coeff_norm={dodo_doe_init_coeff_norm:g}, '
                )
            print(f'[dodo_depth] doe_type_a={dodo_doe_type}, train_c={hparams.optimize_optics}, '
                  f'zernike_mode={dodo_zernike_mode}, '
                  f'zernike_terms={self.camera.doe1.zernike_basis.shape[0]}, '
                  f'zernike_basis={dodo_zernike_basis_path or "auto"}, '
                  f'{dodo_basis_summary}'
                  f'forward_norm={dodo_forward_norm}, '
                  f'forward_scale={dodo_forward_scale:g}, '
                  f'skip_prop2={dodo_skip_prop2}, '
                  f'prop1_padding_factor={dodo_prop1_padding_factor} '
                  f'(work_grid={self.camera.prop1_layers[0].work_Mp}, '
                  f'work_L={self.camera.prop1_layers[0].work_L:g}m), '
                  f'image_formation={dodo_image_formation}, '
                  f'psf_optics_version={dodo_psf_optics_version}, '
                  f'psf_kernel_size={self.camera.psf_kernel_size}, '
                  f'sensor_padding_factor={self.camera.sensor_padding_factor}, '
                  f'depth_layering={depth_layering_mode}, '
                  f'psf_layer_mask={dodo_psf_layer_mask}, '
                  f'psf_mask_sigma={dodo_psf_mask_blur_sigma:g}, '
                  f'psf_boundary={dodo_psf_boundary}, '
                  f'psf_depth_chunk_size={dodo_psf_depth_chunk_size}, '
                  f'optical_halo={dodo_optical_halo}px '
                  f'(context={hparams.image_sz + 2 * dodo_optical_halo}px), '
                  f'psf_energy=(weight={dodo_psf_energy_weight:g}, '
                  f'radius={dodo_psf_energy_radius:g}px, '
                  f'outside_budget={dodo_psf_energy_outside_budget:g}, '
                  f'outer_radius={dodo_psf_energy_outer_radius:g}px, '
                  f'outer_budget={dodo_psf_energy_outer_outside_budget:g}, '
                  f'start_epoch={dodo_psf_energy_start_epoch}, '
                  f'tightening_start='
                  f'{dodo_psf_energy_tightening_start_epoch}, '
                  f'tightening_epochs={dodo_psf_energy_tightening_epochs}, '
                  f'penalty_power={dodo_psf_energy_penalty_power:g}, '
                  f'softness={dodo_psf_energy_softness:g}px, '
                  f'warmup_epochs={dodo_psf_energy_warmup_epochs}), '
                  f'psf_spectral_separation=('
                  f'weight={dodo_psf_spectral_separation_weight:g}, '
                  f'margin={dodo_psf_spectral_separation_margin:g}, '
                  f'start_epoch={dodo_psf_spectral_separation_start_epoch}, '
                  f'warmup_epochs={dodo_psf_spectral_separation_warmup_epochs}), '
                  f'psf_depth_separation=(weight='
                  f'{dodo_psf_depth_separation_weight:g}, margin='
                  f'{dodo_psf_depth_separation_margin:g}, start_epoch='
                  f'{dodo_psf_depth_separation_start_epoch}, warmup_epochs='
                  f'{dodo_psf_depth_separation_warmup_epochs}), '
                  f'psf_mtf=(weight={dodo_psf_mtf_weight:g}, start_epoch='
                  f'{dodo_psf_mtf_start_epoch}, warmup_epochs='
                  f'{dodo_psf_mtf_warmup_epochs}), '
                  f'optical_regularizer_max_ratio='
                  f'{dodo_optical_regularizer_max_ratio:g}, '
                  f'zernike_coefficient_limit='
                  f'{dodo_zernike_coefficient_limit:g}, '
                  f'sensor_measurement={dodo_sensor_measurement}, '
                  f'sensing={dodo_sensing_mode} ch={int(hparams.measurement_channels)}, '
                  f'doe1.zernike_coeffs.requires_grad='
                  f'{self.camera.doe1.zernike_coeffs.requires_grad}')
            # measurement_channels = 3 (RGB sensing output)
            if not hasattr(hparams, 'measurement_channels') or hparams.measurement_channels is None:
                hparams.measurement_channels = 3
            print(f'[dodo_depth] DepthAwareDoDoForwardModel: depth_layers={n_depth_layers}, '
                  f'measurement_channels={hparams.measurement_channels}, '
                  f'depth_layering_mode={depth_layering_mode}, '
                  f'image_formation_mode={dodo_image_formation}')
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

        # Decoder depth input / RGB pseudo-inverse prior (opt-in, default false)
        decoder_use_depth_input = getattr(hparams, 'decoder_use_depth_input', False)
        decoder_depth_input_mode = getattr(hparams, 'decoder_depth_input_mode', 'normalized_diopter')
        decoder_use_rgb_pinv_prior = getattr(hparams, 'decoder_use_rgb_pinv_prior', False)
        decoder_rgb_pinv_lambda = getattr(hparams, 'decoder_rgb_pinv_lambda', 1e-3)
        decoder_rgb_pinv_norm = getattr(hparams, 'decoder_rgb_pinv_norm', 'per_sample_max')
        decoder_rgb_pinv_unscale = getattr(hparams, 'decoder_rgb_pinv_unscale_measurement', True)
        hs_residual_prior = getattr(hparams, 'hs_residual_prior', False)
        hs_residual_prior_eps = getattr(hparams, 'hs_residual_prior_eps', 1e-4)
        detach_depth_guidance_for_hs = getattr(hparams, 'detach_depth_guidance_for_hs', False)
        isolate_hs_decoder_gradients = getattr(hparams, 'isolate_hs_decoder_gradients', False)
        hparams.decoder_use_depth_input = bool(decoder_use_depth_input)
        hparams.decoder_depth_input_mode = str(decoder_depth_input_mode)
        hparams.decoder_use_rgb_pinv_prior = bool(decoder_use_rgb_pinv_prior)
        hparams.decoder_rgb_pinv_lambda = float(decoder_rgb_pinv_lambda)
        hparams.decoder_rgb_pinv_norm = str(decoder_rgb_pinv_norm)
        hparams.decoder_rgb_pinv_unscale_measurement = bool(decoder_rgb_pinv_unscale)
        hparams.hs_residual_prior = bool(hs_residual_prior)
        hparams.hs_residual_prior_eps = float(hs_residual_prior_eps)
        hparams.detach_depth_guidance_for_hs = bool(detach_depth_guidance_for_hs)
        hparams.isolate_hs_decoder_gradients = bool(isolate_hs_decoder_gradients)

        if hparams.hs_residual_prior and not hparams.decoder_use_rgb_pinv_prior:
            raise ValueError('hs_residual_prior requires decoder_use_rgb_pinv_prior')

        if hparams.decoder_use_rgb_pinv_prior:
            if self.optical_model_type != 'dodo_depth':
                raise ValueError('decoder_rgb_pinv_prior is only supported for optical_model=dodo_depth')
            if hparams.preinverse:
                raise ValueError('decoder_rgb_pinv_prior is only supported with --no-preinverse')
            if getattr(hparams, 'dodo_sensing_mode', 'rgb') != 'rgb' or int(hparams.measurement_channels) != 3:
                raise ValueError('decoder_rgb_pinv_prior requires dodo_sensing_mode=rgb and measurement_channels=3')
            if getattr(hparams, 'dodo_measurement_norm', 'none') != 'none':
                raise ValueError('decoder_rgb_pinv_prior requires dodo_measurement_norm=none')
            if hparams.decoder_rgb_pinv_norm not in ('none', 'per_sample_max', 'per_sample_mean_std'):
                raise ValueError(
                    'decoder_rgb_pinv_norm must be one of none/per_sample_max/per_sample_mean_std'
                )
            if hparams.hs_residual_prior and hparams.decoder_rgb_pinv_norm == 'per_sample_mean_std':
                raise ValueError(
                    'hs_residual_prior requires decoder_rgb_pinv_norm=per_sample_max or none; '
                    'per_sample_mean_std can produce negative prior values'
                )
            pinv_prior = self._build_rgb_pinv_prior_matrix(hparams.decoder_rgb_pinv_lambda)
            self.register_buffer('rgb_pinv_prior_matrix', pinv_prior, persistent=False)
        else:
            self.register_buffer('rgb_pinv_prior_matrix', torch.empty(0), persistent=False)

        decoder_extra_channels = 0
        if hparams.decoder_use_depth_input:
            decoder_extra_channels += 1
        if hparams.decoder_use_rgb_pinv_prior and not hparams.hs_residual_prior:
            decoder_extra_channels += int(hparams.hs_channels)
        hparams.decoder_in_channels = int(hparams.measurement_channels) + decoder_extra_channels

        self.decoder = SimpleModel(hparams)
        decoder_norm = getattr(hparams, 'decoder_norm', 'batch')
        dodo_meas_norm = getattr(hparams, 'dodo_measurement_norm', 'none')
        print(f'[decoder] decoder_norm={decoder_norm}, dodo_measurement_norm={dodo_meas_norm}, '
              f'decoder_use_depth_input={hparams.decoder_use_depth_input}, '
              f'decoder_depth_input_mode={hparams.decoder_depth_input_mode}, '
              f'decoder_use_rgb_pinv_prior={hparams.decoder_use_rgb_pinv_prior}, '
              f'hs_residual_prior={hparams.hs_residual_prior}, '
              f'hs_residual_prior_eps={hparams.hs_residual_prior_eps:g}, '
              f'detach_depth_guidance_for_hs={hparams.detach_depth_guidance_for_hs}, '
              f'isolate_hs_decoder_gradients={hparams.isolate_hs_decoder_gradients}, '
              f'decoder_rgb_pinv_lambda={hparams.decoder_rgb_pinv_lambda:g}, '
              f'decoder_rgb_pinv_norm={hparams.decoder_rgb_pinv_norm}, '
              f'decoder_rgb_pinv_unscale_measurement={hparams.decoder_rgb_pinv_unscale_measurement}, '
              f'decoder_in_channels={hparams.decoder_in_channels}')
        self.image_lossfn = CombinedLoss(
            l1_weight=hparams.l1_loss_weight,
            sam_weight=getattr(hparams, 'sam_loss_weight', 0.0),
            mse_weight=getattr(hparams, 'mse_loss_weight', 0.0),
            gradient_weight=getattr(hparams, 'spatial_gradient_loss_weight', 0.0),
        )
        self.depth_lossfn = torch.nn.L1Loss()

    @staticmethod
    def _center_crop_to_size(tensor, target_height, target_width):
        height, width = tensor.shape[-2:]
        if height < target_height or width < target_width:
            raise ValueError(
                f'Cannot center-crop {height}x{width} to '
                f'{target_height}x{target_width}')
        top = (height - target_height) // 2
        left = (width - target_width) // 2
        return tensor[..., top:top + target_height, left:left + target_width]

    def forward(
        self,
        images,
        depthmaps,
        is_testing,
        depth_metric=None,
        valid_mask=None,
        optical_images=None,
        optical_depth_metric=None,
        optical_valid_mask=None,
    ):
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
        rgb_pinv_prior = None

        if self.optical_model_type == 'dodo_depth':
            # The center 128x128 tensors remain the reconstruction targets.
            # Optional larger tensors supply real scene context only to the
            # optical convolution; the resulting measurement is cropped back
            # to the target size before entering the decoder.
            optical_images_linear = images if optical_images is None else optical_images
            optical_depth = depth_metric if optical_depth_metric is None else optical_depth_metric
            optical_mask = valid_mask if optical_valid_mask is None else optical_valid_mask

            # Apply target mask for decoder-side guidance inputs.
            if valid_mask is not None:
                if valid_mask.ndim == 3:
                    valid_mask = valid_mask.unsqueeze(1)  # (B,H,W) -> (B,1,H,W)
                images_linear = images_linear * valid_mask

            # Apply the context mask before optical image formation.
            if optical_mask is not None:
                if optical_mask.ndim == 3:
                    optical_mask = optical_mask.unsqueeze(1)
                optical_images_linear = optical_images_linear * optical_mask

            if optical_depth is None:
                raise ValueError(
                    'dodo_depth requires metric depth input. '
                    'Dataset must provide depth_metric (metric meters).'
                )
            target_height, target_width = images.shape[-2:]

            # DepthAwareDoDoForwardModel: input_format='nchw', output_format='nchw'
            # output: (B, 3, H, W)
            if self.camera.image_formation_mode == 'psf_convolution':
                captimgs, psf = self.camera(
                    optical_images_linear,
                    optical_depth,
                    valid_mask=optical_mask,
                    return_psf=True,
                    output_size=(target_height, target_width),
                )
            else:
                if optical_images is not None:
                    raise ValueError(
                        'Optical halo context is only supported by psf_convolution')
                captimgs = self.camera(images_linear, depth_metric, valid_mask=valid_mask)
                psf = None

            captimgs = self._center_crop_to_size(
                captimgs, target_height, target_width)

            # NaN/Inf guard: dodo optical model can produce non-finite output for near-zero input
            if not torch.isfinite(captimgs).all():
                n_nonfinite = (~torch.isfinite(captimgs)).sum().item()
                mask_ratio = optical_mask.mean().item() if optical_mask is not None else float('nan')
                spec_sum = optical_images_linear.sum().item()
                dmetric_min = optical_depth.min().item() if optical_depth is not None else float('nan')
                dmetric_max = optical_depth.max().item() if optical_depth is not None else float('nan')
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
            if getattr(self.hparams, 'decoder_use_rgb_pinv_prior', False):
                rgb_pinv_prior = self._rgb_pinv_prior_from_measurement(captimgs)
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

        if (getattr(self.hparams, 'decoder_use_rgb_pinv_prior', False)
                and not self.hparams.preinverse):
            if self.optical_model_type != 'dodo_depth':
                raise ValueError('decoder_rgb_pinv_prior is only supported for dodo_depth')
            if rgb_pinv_prior is None:
                rgb_pinv_prior = self._rgb_pinv_prior_from_measurement(captimgs[:, :3, :, :])
            if rgb_pinv_prior.shape[-2:] != captimgs.shape[-2:]:
                rgb_pinv_prior = crop_boundary(rgb_pinv_prior, self.crop_width)
            if not getattr(self.hparams, 'hs_residual_prior', False):
                captimgs = torch.cat([captimgs, rgb_pinv_prior.to(captimgs.dtype)], dim=1)

        model_outputs = self.decoder(captimgs=captimgs, pinv_volumes=pinv_volumes, images=images_linear,
                                     depthmaps=depthmaps,
                                     rgb_pinv_prior=rgb_pinv_prior)
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

    def _dodo_psf_energy_weight(self):
        """Delay and warm up the DOE energy regularizer during Stage A."""
        target_weight = float(getattr(self.hparams, 'dodo_psf_energy_weight', 0.02))
        if target_weight < 0.0:
            raise ValueError(f'dodo_psf_energy_weight must be >= 0, got {target_weight}')
        if target_weight == 0.0 or not bool(getattr(self.hparams, 'optimize_optics', False)):
            return 0.0
        if self.optical_model_type != 'dodo_depth':
            return 0.0
        if getattr(self.camera, 'image_formation_mode', None) != 'psf_convolution':
            return 0.0

        start_epoch = int(getattr(
            self.hparams, 'dodo_psf_energy_start_epoch', 0))
        warmup_epochs = int(getattr(self.hparams, 'dodo_psf_energy_warmup_epochs', 2))
        current_epoch = int(self.current_epoch)
        effective_weight = delayed_epoch_warmup_weight(
            target_weight, current_epoch, start_epoch, warmup_epochs)
        if current_epoch >= start_epoch + max(1, warmup_epochs) and effective_weight <= 0.0:
            raise RuntimeError(
                'DoDo PSF energy regularization is enabled during Stage A, '
                f'but its effective weight is {effective_weight} at epoch '
                f'{current_epoch}. Refusing to continue with a silently '
                'disabled optical regularizer.'
            )
        return effective_weight

    def _dodo_psf_spectral_separation_weight(self):
        """Delay and warm up the sensor-visible wavelength separation loss."""
        target_weight = float(getattr(
            self.hparams, 'dodo_psf_spectral_separation_weight', 0.0))
        if target_weight < 0.0:
            raise ValueError(
                'dodo_psf_spectral_separation_weight must be >= 0, got '
                f'{target_weight}')
        if target_weight == 0.0 or not bool(
                getattr(self.hparams, 'optimize_optics', False)):
            return 0.0
        if (self.optical_model_type != 'dodo_depth'
                or getattr(self.camera, 'image_formation_mode', None)
                != 'psf_convolution'):
            return 0.0
        start_epoch = int(getattr(
            self.hparams, 'dodo_psf_spectral_separation_start_epoch', 0))
        warmup_epochs = int(getattr(
            self.hparams, 'dodo_psf_spectral_separation_warmup_epochs', 2))
        return delayed_epoch_warmup_weight(
            target_weight, int(self.current_epoch), start_epoch, warmup_epochs)

    def _dodo_optical_weight(
            self, hparam_name, default, start_epoch_name=None,
            warmup_epochs_name=None):
        """Return a scheduled DOE-only weight, automatically disabled in Stage B."""
        weight = float(getattr(self.hparams, hparam_name, default))
        if weight < 0.0:
            raise ValueError(f'{hparam_name} must be >= 0, got {weight}')
        if weight == 0.0 or not bool(
                getattr(self.hparams, 'optimize_optics', False)):
            return 0.0
        if (self.optical_model_type != 'dodo_depth'
                or getattr(self.camera, 'image_formation_mode', None)
                != 'psf_convolution'):
            return 0.0
        if start_epoch_name is None and warmup_epochs_name is None:
            return weight
        start_epoch = int(getattr(self.hparams, start_epoch_name, 0))
        warmup_epochs = int(getattr(self.hparams, warmup_epochs_name, 0))
        return delayed_epoch_warmup_weight(
            weight, int(self.current_epoch), start_epoch, warmup_epochs)
    
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
        image_loss, image_components = self.image_lossfn(
            est_images, target_images, mask=final_mask)
        image_l1 = image_components['l1']
        image_mse = image_components['mse']
        image_sam = image_components['sam']
        image_gradient = image_components['gradient']

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

        # --- PSF physical losses ---
        zero = depth_loss * 0.0
        psf_loss = zero
        psf_out_of_fov_max = zero.detach()
        psf_energy_outside_mean = zero.detach()
        psf_energy_outside_p90 = zero.detach()
        psf_energy_inside_mean = zero.detach()
        psf_energy_active_fraction = zero.detach()
        psf_energy_captured_mean = zero.detach()
        psf_energy_missing_mean = zero.detach()
        psf_energy_outer_inside_mean = zero.detach()
        psf_energy_outer_outside_p90 = zero.detach()
        psf_energy_r50_mean = zero.detach()
        psf_energy_r80_mean = zero.detach()
        psf_energy_r90_mean = zero.detach()
        psf_energy_r90_p90 = zero.detach()
        psf_energy_r90_max = zero.detach()
        psf_energy_r90_unresolved_fraction = zero.detach()
        effective_psf_loss_weight = 0.0

        psf_mtf_loss = zero
        psf_mtf_005_mean = zero.detach()
        psf_mtf_005_p10 = zero.detach()
        psf_mtf_010_mean = zero.detach()
        psf_mtf_010_p10 = zero.detach()
        psf_mtf_020_mean = zero.detach()
        effective_psf_mtf_weight = 0.0

        psf_spectral_separation_loss = zero
        psf_spectral_adjacent_cosine_mean = zero.detach()
        psf_spectral_adjacent_cosine_p90 = zero.detach()
        psf_spectral_adjacent_cosine_max = zero.detach()
        psf_spectral_active_fraction = zero.detach()
        effective_psf_spectral_weight = 0.0

        psf_depth_separation_loss = zero
        psf_depth_adjacent_cosine_mean = zero.detach()
        psf_depth_adjacent_cosine_p90 = zero.detach()
        psf_depth_adjacent_cosine_max = zero.detach()
        effective_psf_depth_weight = 0.0

        zernike_high_order_loss = zero
        effective_zernike_high_order_weight = 0.0
        zernike_low_order_norm = zero.detach()
        zernike_high_order_norm = zero.detach()

        if self.hparams.psf_loss_weight > 0 and self.optical_model_type == 'legacy_camera':
            psf_out_of_fov_sum, psf_out_of_fov_max = self.camera.psf_out_of_fov_energy(self.hparams.psf_size)
            psf_loss = psf_out_of_fov_sum / self.hparams.hs_channels
            effective_psf_loss_weight = float(self.hparams.psf_loss_weight)
        elif (
            self.optical_model_type == 'dodo_depth'
            and getattr(self.camera, 'image_formation_mode', None) == 'psf_convolution'
        ):
            if outputs.psf is None:
                raise RuntimeError(
                    'PSF convolution must return its live PSF bank when the DoDo '
                    'energy regularizer is configured.'
                )
            tightening_start_epoch = int(getattr(
                self.hparams, 'dodo_psf_energy_tightening_start_epoch', 0))
            core_budget = delayed_epoch_tightening_value(
                float(getattr(
                    self.hparams,
                    'dodo_psf_energy_initial_outside_budget', 0.35)),
                float(getattr(
                    self.hparams,
                    'dodo_psf_energy_outside_budget', 0.20)),
                int(self.current_epoch),
                tightening_start_epoch,
                int(getattr(
                    self.hparams, 'dodo_psf_energy_tightening_epochs', 3)),
            )
            outer_budget = delayed_epoch_tightening_value(
                float(getattr(
                    self.hparams,
                    'dodo_psf_energy_initial_outer_outside_budget', 0.15)),
                float(getattr(
                    self.hparams,
                    'dodo_psf_energy_outer_outside_budget', 0.05)),
                int(self.current_epoch),
                tightening_start_epoch,
                int(getattr(
                    self.hparams, 'dodo_psf_energy_tightening_epochs', 3)),
            )
            core_radius = float(getattr(
                self.hparams, 'dodo_psf_energy_radius', 16.0))
            outer_radius = float(getattr(
                self.hparams, 'dodo_psf_energy_outer_radius', 24.0))
            psf_loss, psf_stats = multiscale_psf_energy_concentration_loss(
                outputs.psf,
                radii=(core_radius, outer_radius),
                outside_budgets=(core_budget, outer_budget),
                scale_weights=(1.0, 0.5),
                softness=float(getattr(
                    self.hparams, 'dodo_psf_energy_softness', 1.5)),
                cvar_fraction=float(getattr(
                    self.hparams, 'dodo_psf_energy_cvar_fraction', 0.10)),
                cvar_weight=float(getattr(
                    self.hparams, 'dodo_psf_energy_cvar_weight', 0.5)),
                penalty_power=float(getattr(
                    self.hparams, 'dodo_psf_energy_penalty_power', 2.0)),
                energy_reference=getattr(
                    self.camera, 'psf_energy_reference', 'crop'),
            )
            psf_energy_outside_mean = psf_stats['outside_mean']
            psf_energy_outside_p90 = psf_stats['outside_p90']
            psf_out_of_fov_max = psf_stats['outside_max']
            psf_energy_inside_mean = psf_stats['inside_mean']
            psf_energy_active_fraction = psf_stats['active_fraction']
            psf_energy_captured_mean = psf_stats['captured_mean']
            psf_energy_missing_mean = psf_stats['missing_mean']
            outer_key = f'r{int(round(outer_radius))}'
            psf_energy_outer_inside_mean = psf_stats[
                f'{outer_key}_inside_mean']
            psf_energy_outer_outside_p90 = psf_stats[
                f'{outer_key}_outside_p90']
            psf_energy_r50_mean = psf_stats['r50_mean']
            psf_energy_r80_mean = psf_stats['r80_mean']
            psf_energy_r90_mean = psf_stats['r90_mean']
            psf_energy_r90_p90 = psf_stats['r90_p90']
            psf_energy_r90_max = psf_stats['r90_max']
            psf_energy_r90_unresolved_fraction = psf_stats[
                'r90_unresolved_fraction']
            effective_psf_loss_weight = self._dodo_psf_energy_weight()

            psf_mtf_loss, mtf_stats = psf_mtf_floor_loss(
                outputs.psf,
                min_frequency=float(getattr(
                    self.hparams, 'dodo_psf_mtf_min_frequency', 0.02)),
                max_frequency=float(getattr(
                    self.hparams, 'dodo_psf_mtf_max_frequency', 0.15)),
                mtf_at_005=float(getattr(
                    self.hparams, 'dodo_psf_mtf_target_005', 0.12)),
                mtf_at_010=float(getattr(
                    self.hparams, 'dodo_psf_mtf_target_010', 0.05)),
                mtf_at_015=float(getattr(
                    self.hparams, 'dodo_psf_mtf_target_015', 0.025)),
            )
            psf_mtf_005_mean = mtf_stats['mtf_005_mean']
            psf_mtf_005_p10 = mtf_stats['mtf_005_p10']
            psf_mtf_010_mean = mtf_stats['mtf_010_mean']
            psf_mtf_010_p10 = mtf_stats['mtf_010_p10']
            psf_mtf_020_mean = mtf_stats['mtf_020_mean']
            effective_psf_mtf_weight = self._dodo_optical_weight(
                'dodo_psf_mtf_weight', 0.0,
                'dodo_psf_mtf_start_epoch',
                'dodo_psf_mtf_warmup_epochs')

            if getattr(self.camera.sensing_unnorm, 'sensing_mode', None) == 'rgb':
                sensor_response = self.camera._sensor_response_matrix(
                    outputs.psf.device, outputs.psf.dtype)
                (psf_spectral_separation_loss,
                 psf_spectral_stats) = (
                    sensor_weighted_spectral_psf_separation_loss(
                        outputs.psf,
                        sensor_response,
                        margin=float(getattr(
                            self.hparams,
                            'dodo_psf_spectral_separation_margin',
                            0.90)),
                        offsets=(1, 2),
                        hard_fraction=float(getattr(
                            self.hparams,
                            'dodo_psf_spectral_hard_fraction', 0.20)),
                        hard_weight=float(getattr(
                            self.hparams,
                            'dodo_psf_spectral_hard_weight', 0.5)),
                    )
                )
                psf_spectral_adjacent_cosine_mean = psf_spectral_stats[
                    'adjacent_cosine_mean']
                psf_spectral_adjacent_cosine_p90 = psf_spectral_stats[
                    'adjacent_cosine_p90']
                psf_spectral_adjacent_cosine_max = psf_spectral_stats[
                    'adjacent_cosine_max']
                psf_spectral_active_fraction = psf_spectral_stats[
                    'active_fraction']
                effective_psf_spectral_weight = (
                    self._dodo_psf_spectral_separation_weight())

                (psf_depth_separation_loss,
                 psf_depth_stats) = sensor_weighted_depth_psf_separation_loss(
                    outputs.psf,
                    sensor_response,
                    margin=float(getattr(
                        self.hparams,
                        'dodo_psf_depth_separation_margin', 0.90)),
                    hard_fraction=float(getattr(
                        self.hparams,
                        'dodo_psf_depth_hard_fraction', 0.20)),
                    hard_weight=float(getattr(
                        self.hparams,
                        'dodo_psf_depth_hard_weight', 0.5)),
                    energy_reference=getattr(
                        self.camera, 'psf_energy_reference', 'crop'),
                )
                psf_depth_adjacent_cosine_mean = psf_depth_stats[
                    'adjacent_cosine_mean']
                psf_depth_adjacent_cosine_p90 = psf_depth_stats[
                    'adjacent_cosine_p90']
                psf_depth_adjacent_cosine_max = psf_depth_stats[
                    'adjacent_cosine_max']
                effective_psf_depth_weight = self._dodo_optical_weight(
                    'dodo_psf_depth_separation_weight', 0.0,
                    'dodo_psf_depth_separation_start_epoch',
                    'dodo_psf_depth_separation_warmup_epochs')

            coefficients = getattr(self.camera.doe1, 'zernike_coeffs', None)
            if (isinstance(coefficients, torch.Tensor)
                    and getattr(
                        self.hparams, 'dodo_zernike_mode', 'legacy12') == 'free'):
                low_terms = int(getattr(
                    self.hparams, 'dodo_zernike_low_order_terms', 15))
                zernike_high_order_loss = zernike_order_weighted_l2(
                    coefficients, protected_terms=low_terms)
                effective_zernike_high_order_weight = self._dodo_optical_weight(
                    'dodo_zernike_high_order_weight', 0.0)
                with torch.no_grad():
                    zernike_low_order_norm = torch.linalg.vector_norm(
                        coefficients[:low_terms]).detach()
                    zernike_high_order_norm = torch.linalg.vector_norm(
                        coefficients[low_terms:]).detach()

        # --- Background HS Loss (opt-in) ---
        bg_hs_loss_weight = float(getattr(self.hparams, 'background_hs_loss_weight', 0.0))
        bg_hs_loss = torch.tensor(0.0, device=depth_loss.device)
        if bg_hs_loss_weight > 0.0:
            bg_mask = (1.0 - final_mask.unsqueeze(1))  # [B,1,H,W], background=1
            if bg_mask.sum() > 0:
                bg_hs_loss = (
                    torch.abs(est_images - target_images) * bg_mask
                ).sum() / (bg_mask.sum() * est_images.shape[1] + 1e-6)

        # --- 2. 加权 ---
        weighted_depth_loss = self.hparams.depth_loss_weight * depth_loss
        weighted_image_loss = self.hparams.image_loss_weight * image_loss
        weighted_psf_loss = effective_psf_loss_weight * psf_loss
        weighted_psf_spectral_loss = (
            effective_psf_spectral_weight * psf_spectral_separation_loss)
        weighted_psf_mtf_loss = effective_psf_mtf_weight * psf_mtf_loss
        weighted_psf_depth_loss = (
            effective_psf_depth_weight * psf_depth_separation_loss)
        weighted_zernike_high_order_loss = (
            effective_zernike_high_order_weight * zernike_high_order_loss)
        weighted_depth_smooth_loss = depth_smooth_weight * depth_smooth_loss
        weighted_metric_depth_loss = metric_depth_loss_weight * metric_depth_loss

        weighted_bg_hs_loss = bg_hs_loss_weight * bg_hs_loss

        task_loss = (
            weighted_depth_loss + weighted_image_loss
            + weighted_depth_smooth_loss + weighted_metric_depth_loss
            + weighted_bg_hs_loss
        )
        optical_regularizer_raw = (
            weighted_psf_loss + weighted_psf_spectral_loss
            + weighted_psf_mtf_loss + weighted_psf_depth_loss
            + weighted_zernike_high_order_loss
        )
        optical_regularizer_scale = task_relative_regularizer_scale(
            optical_regularizer_raw,
            task_loss,
            float(getattr(
                self.hparams, 'dodo_optical_regularizer_max_ratio', 0.0)),
        )
        weighted_psf_loss = weighted_psf_loss * optical_regularizer_scale
        weighted_psf_spectral_loss = (
            weighted_psf_spectral_loss * optical_regularizer_scale)
        weighted_psf_mtf_loss = (
            weighted_psf_mtf_loss * optical_regularizer_scale)
        weighted_psf_depth_loss = (
            weighted_psf_depth_loss * optical_regularizer_scale)
        weighted_zernike_high_order_loss = (
            weighted_zernike_high_order_loss * optical_regularizer_scale)
        optical_regularizer_weighted = (
            weighted_psf_loss + weighted_psf_spectral_loss
            + weighted_psf_mtf_loss + weighted_psf_depth_loss
            + weighted_zernike_high_order_loss
        )
        optical_regularizer_ratio = (
            optical_regularizer_weighted.detach()
            / task_loss.detach().abs().clamp_min(1e-12)
        )
        total_loss = task_loss + optical_regularizer_weighted

        return total_loss, {
            'total_loss': total_loss,
            'depth_loss': depth_loss,
            'depth_smooth_loss': depth_smooth_loss,
            'metric_depth_loss': metric_depth_loss,
            'image_loss_total': image_loss,
            'image_loss_l1': image_l1,
            'image_loss_mse': image_mse,
            'image_loss_sam': image_sam,
            'image_loss_gradient': image_gradient,
            'psf_loss': psf_loss,
            'psf_loss_weight': torch.tensor(
                effective_psf_loss_weight, device=depth_loss.device),
            'psf_loss_effective_weight': (
                optical_regularizer_scale
                * effective_psf_loss_weight),
            'psf_loss_weighted': weighted_psf_loss,
            'psf_out_of_fov_max': psf_out_of_fov_max,
            'psf_energy_outside_mean': psf_energy_outside_mean,
            'psf_energy_outside_p90': psf_energy_outside_p90,
            'psf_energy_inside_mean': psf_energy_inside_mean,
            'psf_energy_active_fraction': psf_energy_active_fraction,
            'psf_energy_captured_mean': psf_energy_captured_mean,
            'psf_energy_missing_mean': psf_energy_missing_mean,
            'psf_energy_outer_inside_mean': psf_energy_outer_inside_mean,
            'psf_energy_outer_outside_p90': psf_energy_outer_outside_p90,
            'psf_energy_r50_mean': psf_energy_r50_mean,
            'psf_energy_r80_mean': psf_energy_r80_mean,
            'psf_energy_r90_mean': psf_energy_r90_mean,
            'psf_energy_r90_p90': psf_energy_r90_p90,
            'psf_energy_r90_max': psf_energy_r90_max,
            'psf_energy_r90_unresolved_fraction': (
                psf_energy_r90_unresolved_fraction),
            'psf_energy_core_budget': torch.tensor(
                core_budget if self.optical_model_type == 'dodo_depth' else 0.0,
                device=depth_loss.device),
            'psf_energy_outer_budget': torch.tensor(
                outer_budget if self.optical_model_type == 'dodo_depth' else 0.0,
                device=depth_loss.device),
            'psf_mtf_loss': psf_mtf_loss,
            'psf_mtf_weight': torch.tensor(
                effective_psf_mtf_weight, device=depth_loss.device),
            'psf_mtf_effective_weight': (
                optical_regularizer_scale * effective_psf_mtf_weight),
            'psf_mtf_weighted': weighted_psf_mtf_loss,
            'psf_mtf_005_mean': psf_mtf_005_mean,
            'psf_mtf_005_p10': psf_mtf_005_p10,
            'psf_mtf_010_mean': psf_mtf_010_mean,
            'psf_mtf_010_p10': psf_mtf_010_p10,
            'psf_mtf_020_mean': psf_mtf_020_mean,
            'psf_spectral_separation_loss': psf_spectral_separation_loss,
            'psf_spectral_separation_weight': torch.tensor(
                effective_psf_spectral_weight, device=depth_loss.device),
            'psf_spectral_separation_effective_weight': (
                optical_regularizer_scale
                * effective_psf_spectral_weight),
            'psf_spectral_separation_weighted': weighted_psf_spectral_loss,
            'psf_spectral_adjacent_cosine_mean': (
                psf_spectral_adjacent_cosine_mean),
            'psf_spectral_adjacent_cosine_p90': (
                psf_spectral_adjacent_cosine_p90),
            'psf_spectral_adjacent_cosine_max': (
                psf_spectral_adjacent_cosine_max),
            'psf_spectral_active_fraction': psf_spectral_active_fraction,
            'psf_depth_separation_loss': psf_depth_separation_loss,
            'psf_depth_separation_weight': torch.tensor(
                effective_psf_depth_weight, device=depth_loss.device),
            'psf_depth_separation_effective_weight': (
                optical_regularizer_scale
                * effective_psf_depth_weight),
            'psf_depth_separation_weighted': weighted_psf_depth_loss,
            'psf_depth_adjacent_cosine_mean': (
                psf_depth_adjacent_cosine_mean),
            'psf_depth_adjacent_cosine_p90': (
                psf_depth_adjacent_cosine_p90),
            'psf_depth_adjacent_cosine_max': (
                psf_depth_adjacent_cosine_max),
            'zernike_high_order_loss': zernike_high_order_loss,
            'zernike_high_order_effective_weight': (
                optical_regularizer_scale
                * effective_zernike_high_order_weight),
            'zernike_high_order_weighted': weighted_zernike_high_order_loss,
            'zernike_low_order_norm': zernike_low_order_norm,
            'zernike_high_order_norm': zernike_high_order_norm,
            'task_loss_weighted': task_loss,
            'optical_regularizer_raw': optical_regularizer_raw,
            'optical_regularizer_scale': optical_regularizer_scale,
            'optical_regularizer_weighted': optical_regularizer_weighted,
            'optical_regularizer_ratio': optical_regularizer_ratio,
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
        parser.add_argument(
            '--dodo_prop1_padding_factor',
            type=int,
            default=1,
            help=(
                'Prop1 保持物理像素间距的计算窗倍数；'
                '2 表示将 128/0.01m 扩为 256/0.02m 后中心裁剪'
            ),
        )
        parser.add_argument('--depth_layering_mode', type=str, default='soft_diopter',
                            choices=['hard_depth', 'hard_meter', 'soft_diopter'],
                            help='DoDo depth layering mode')
        parser.add_argument('--soft_diopter_eps', type=float, default=1e-8,
                            help='Soft diopter weight normalization epsilon')
        parser.add_argument('--soft_diopter_bandwidth_scale', type=float, default=1.0,
                            help='Soft diopter triangular bandwidth multiplier')
        parser.add_argument('--dodo_image_formation', type=str, default='whole_field',
                            choices=['whole_field', 'psf_convolution'],
                            help='DoDo image formation: legacy whole-field propagation or Baek-style PSF convolution')
        parser.add_argument(
            '--dodo_psf_optics_version',
            type=str,
            default='legacy',
            choices=['legacy', 'consistent_grid_v1'],
            help=(
                'PSF 光学离散化版本；legacy 保持旧 checkpoint 语义，'
                'consistent_grid_v1 使用统一采样、完整传感器工作网格'
                '与中心 129x129 PSF'
            ),
        )
        parser.add_argument('--dodo_psf_layer_mask', type=str, default='baek_hard',
                            choices=['current', 'baek_hard'],
                            help='PSF path depth masks: current layering weights or Baek hard occupancy masks')
        parser.add_argument('--dodo_psf_mask_blur_sigma', type=float, default=1.0,
                            help='Gaussian sigma in pixels for PSF-path depth occupancy masks; 0 disables blur')
        parser.add_argument('--dodo_psf_boundary', type=str, default='linear_zero',
                            choices=['linear_zero', 'circular'],
                            help='PSF convolution boundary model; linear_zero avoids circular FFT wrap-around')
        parser.add_argument(
            '--dodo_psf_depth_chunk_size',
            type=int,
            default=1,
            help=(
                'Number of depth layers processed by each batched inverse FFT; '
                'larger values reduce launch overhead but use more peak memory'
            ),
        )
        parser.add_argument('--dodo_doe_type', type=str, default='Zeros',
                            help='DoDo DOE 类型（Zeros=frozen, New=trainable Zernike）')
        parser.add_argument('--dodo_zernike_mode', type=str, default='legacy12',
                            choices=['legacy12', 'free'],
                            help='Zernike basis: legacy12 loads the original 12-term MAT; '
                                 'free loads an N-term NPY basis')
        parser.add_argument(
            '--dodo_doe_basis_mode',
            type=str,
            default='legacy_raw12',
            choices=['legacy_raw12', 'orthogonal_rms'],
            help=(
                'legacy12 内部基底：legacy_raw12 保持旧 checkpoint；'
                'orthogonal_rms 去除近共线方向并统一 pupil 内物理 RMS'
            ),
        )
        parser.add_argument(
            '--dodo_doe_basis_rank',
            type=int,
            default=9,
            help='orthogonal_rms 从原始 12 项中保留的有效独立模式数',
        )
        parser.add_argument(
            '--dodo_doe_basis_rank_rtol',
            type=float,
            default=1e-4,
            help='orthogonal_rms 丢弃近共线模式的相对残差阈值',
        )
        parser.add_argument(
            '--dodo_doe_basis_rms_m',
            type=float,
            default=3e-6,
            help='orthogonal_rms 每个正交模式在 pupil 内的高度 RMS（米）',
        )
        parser.add_argument(
            '--dodo_doe_coeff_norm_limit',
            type=float,
            default=1.0,
            help='正交 DOE 系数向量的 L2 上限',
        )
        parser.add_argument(
            '--dodo_doe_init_coeff_norm',
            type=float,
            default=0.2,
            help='orthogonal_rms 模式的初始系数 L2 范数',
        )
        parser.add_argument('--dodo_zernike_terms', type=int, default=150,
                            help='Number of Zernike terms when --dodo_zernike_mode free is used')
        parser.add_argument('--dodo_zernike_basis_path', type=str, default=None,
                            help='Optional explicit NPY basis path for free mode; by default loads '
                                 'torch_optics/assets/zernike_volume1_128_Nterms_<N>.npy')
        parser.add_argument('--dodo_zernike_init_checkpoint', type=str, default='',
                            help='Legacy checkpoint whose DOE wavefront is projected into the active free basis.')
        parser.add_argument('--dodo_zernike_init_legacy_basis_path', type=str, default='',
                            help='Legacy MAT basis containing HmBase for wavefront projection.')
        parser.add_argument('--dodo_zernike_low_order_terms', type=int, default=15)
        parser.add_argument('--dodo_zernike_high_order_unlock_epoch', type=int, default=0)
        parser.add_argument('--dodo_zernike_high_order_lr_ratio', type=float, default=1.0)
        parser.add_argument('--dodo_zernike_high_order_weight', type=float, default=0.0)
        parser.add_argument('--dodo_zernike_coefficient_limit', type=float, default=1.0)
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
        parser.add_argument('--lr_decay_strategy', type=str, default='none',
                            choices=['none', 'baek'],
                            help='学习率衰减策略；baek=optics每10epoch*0.1，CNN每20epoch*0.1')
        parser.add_argument('--lr_warmup_steps', type=int, default=54,
                            help='线性warmup步数；0=关闭')
        parser.add_argument('--optics_lr_decay_epochs', type=int, default=10,
                            help='baek策略下optics学习率每隔多少epoch乘0.1')
        parser.add_argument('--cnn_lr_decay_epochs', type=int, default=20,
                            help='baek策略下CNN学习率每隔多少epoch乘0.1')
        parser.add_argument('--batch_sz', type=int, default=2)
        parser.add_argument('--num_workers', type=int, default=8)
        parser.add_argument('--randcrop', default=False, action='store_true')
        parser.add_argument('--augment', default=False, action='store_true')
        parser.add_argument(
            '--baek_augment',
            action='store_true',
            help='启用同步缩放/翻转、米制深度平移和29种CIE光源增强',
        )
        parser.add_argument(
            '--baek_scale_half_probability', type=float, default=0.30
        )
        parser.add_argument('--baek_depth_shift_m', type=float, default=0.20)
        parser.add_argument(
            '--baek_depth_shift_probability', type=float, default=0.50
        )
        parser.add_argument(
            '--baek_illuminant_probability', type=float, default=0.80
        )
        parser.add_argument('--baek_exposure_min', type=float, default=0.90)
        parser.add_argument('--baek_exposure_max', type=float, default=1.10)
        parser.add_argument('--baek_max_clip_ratio', type=float, default=0.001)
        parser.add_argument('--baek_illuminant_retries', type=int, default=8)
        parser.add_argument('--patch_filter', dest='patch_filter', action='store_true',
                            help='训练时对随机裁剪 patch 做质量筛选（轻量版：仅在 depth/mask 上重采样判定）')
        parser.add_argument('--no-patch_filter', dest='patch_filter', action='store_false',
                            help='关闭训练 patch 质量筛选')
        parser.set_defaults(patch_filter=True)
        parser.add_argument('--min_valid_ratio', type=float, default=0.12,
                            help='patch 中有效像素占比下限（0~1）')
        parser.add_argument('--min_depth_range_ips', type=float, default=0.10,
                            help='patch 内有效区域 IPS 深度动态范围下限')
        parser.add_argument('--min_center_valid_ratio', type=float, default=0.0,
                            help='patch 中心区域有效像素占比下限；0=不启用，加载候选池时默认沿用候选池meta')
        parser.add_argument('--max_crop_retries', type=int, default=8,
                            help='随机裁剪失败后最多重采样次数（轻量判定，开销较小）')
        parser.add_argument('--patch_filter_stride', type=int, default=4,
                            help='patch筛选预检步长(>1更快，=1最严格)')
        parser.add_argument('--patch_index_path', type=str, nargs='?', const='auto', default='',
                            help='离线高质量patch候选池 .npz 路径；只写参数不填值时自动使用 data_root/.patch_index 下的默认候选池')
        parser.add_argument('--train_patch_index_path', type=str, default='',
                            help='训练专用patch索引；为空时沿用--patch_index_path')
        parser.add_argument('--val_patch_index_path', type=str, default='',
                            help='验证专用patch索引；为空时沿用--patch_index_path')
        parser.add_argument('--patch_index_jitter', type=int, default=16,
                            help='候选池坐标在线随机扰动像素数；0=不扰动')
        parser.add_argument('--patch_index_hs_jitter', type=int, default=8,
                            help='HS高亮/复杂类别的最大坐标扰动；避免偏离光谱难例')
        parser.add_argument('--patch_index_strict', dest='patch_index_strict', action='store_true',
                            help='候选池坐标/jitter后仍按当前质量阈值复检')
        parser.add_argument('--no-patch_index_strict', dest='patch_index_strict', action='store_false')
        parser.set_defaults(patch_index_strict=True)
        parser.add_argument('--patch_index_weighted', dest='patch_index_weighted', action='store_true',
                            help='按候选池score加权采样；默认均匀采样')
        parser.add_argument('--no-patch_index_weighted', dest='patch_index_weighted', action='store_false')
        parser.set_defaults(patch_index_weighted=False)
        parser.add_argument('--patch_index_use_meta_thresholds', dest='patch_index_use_meta_thresholds',
                            action='store_true',
                            help='加载候选池后使用候选池meta中的质量阈值做在线复检')
        parser.add_argument('--no-patch_index_use_meta_thresholds', dest='patch_index_use_meta_thresholds',
                            action='store_false')
        parser.set_defaults(patch_index_use_meta_thresholds=True)
        parser.add_argument('--train_samples_per_epoch', type=int, default=0,
                            help='训练时每个epoch抽取的虚拟patch数；0=按真实scene数量')
        parser.add_argument('--train_patch_index_enumerate', dest='train_patch_index_enumerate',
                            action='store_true',
                            help='训练时按patch index固定窗口枚举；DataLoader可打乱顺序，但epoch覆盖候选池')
        parser.add_argument('--no-train_patch_index_enumerate', dest='train_patch_index_enumerate',
                            action='store_false')
        parser.set_defaults(train_patch_index_enumerate=False)
        parser.add_argument(
            '--train_patch_category_mix',
            type=str,
            default='',
            help=(
                '分层训练patch比例，例如 '
                'depth_hard=0.4,hs_bright=0.2,hs_complex=0.2,general=0.2；'
                '为空时保持旧索引采样行为'
            ),
        )
        parser.add_argument('--train_patch_category_seed', type=int, default=123,
                            help='每场景类别配额调度的固定随机种子')
        parser.add_argument('--baek_patch_epoch', dest='baek_patch_epoch', action='store_true',
                            help='Baek-style patch epoch：每个训练epoch按当前候选池规模抽取6143个patch')
        parser.add_argument('--no-baek_patch_epoch', dest='baek_patch_epoch', action='store_false')
        parser.set_defaults(baek_patch_epoch=False)
        parser.add_argument('--val_patch_eval', dest='val_patch_eval', action='store_true',
                            help='验证时使用固定patch候选池，而不是每个val scene的中心crop')
        parser.add_argument('--no-val_patch_eval', dest='val_patch_eval', action='store_false')
        parser.set_defaults(val_patch_eval=None)
        parser.add_argument('--val_samples_per_epoch', type=int, default=1024,
                            help='固定patch验证时每个epoch评估的patch数；0=使用全部匹配val候选')
        parser.add_argument('--depth_loss_weight', type=float, default=0.03)
        parser.add_argument('--image_loss_weight', type=float, default=1.0)
        parser.add_argument('--psf_loss_weight', type=float, default=1.0)
        parser.add_argument(
            '--dodo_psf_energy_weight', type=float, default=0.02,
            help=(
                'DoDo PSF-convolution energy-regularizer target weight. '
                'Effective weight is warmed up per dodo_psf_energy_warmup_epochs.'
            ),
        )
        parser.add_argument(
            '--dodo_psf_energy_radius', type=float, default=16.0,
            help='PSF concentration target radius in sensor pixels.',
        )
        parser.add_argument(
            '--dodo_psf_energy_outside_budget', type=float, default=0.20,
            help='Allowed normalized PSF energy fraction outside the target radius.',
        )
        parser.add_argument('--dodo_psf_energy_outer_radius', type=float, default=24.0)
        parser.add_argument(
            '--dodo_psf_energy_outer_outside_budget', type=float, default=0.05)
        parser.add_argument(
            '--dodo_psf_energy_initial_outside_budget', type=float, default=0.35)
        parser.add_argument(
            '--dodo_psf_energy_initial_outer_outside_budget',
            type=float, default=0.15)
        parser.add_argument(
            '--dodo_psf_energy_tightening_epochs', type=int, default=3)
        parser.add_argument(
            '--dodo_psf_energy_cvar_fraction', type=float, default=0.10)
        parser.add_argument(
            '--dodo_psf_energy_cvar_weight', type=float, default=0.5)
        parser.add_argument(
            '--dodo_psf_energy_penalty_power', type=float, default=2.0,
            help=(
                'Constraint-violation exponent in [1,2]. One keeps a '
                'non-vanishing gradient near the target; two is the legacy '
                'squared hinge.'
            ),
        )
        parser.add_argument(
            '--dodo_psf_energy_softness', type=float, default=1.5,
            help='Logistic radial-mask transition width in pixels; 0 selects a hard mask.',
        )
        parser.add_argument(
            '--dodo_psf_energy_warmup_epochs', type=int, default=2,
            help='Epochs used for PSF energy weight warm-up: epoch 0=0, epoch N=full.',
        )
        parser.add_argument(
            '--dodo_psf_energy_start_epoch', type=int, default=0,
            help='Zero-weight anchor epoch for the PSF energy-loss ramp.')
        parser.add_argument(
            '--dodo_psf_energy_tightening_start_epoch', type=int, default=0,
            help='Epoch at which PSF outside-energy budgets begin tightening.')
        parser.add_argument(
            '--dodo_optical_halo', type=int, default=0,
            help=(
                'Real scene context on each side of a 128 target patch. '
                '64 gives a 256x256 optical input and a center 128x128 target.'
            ),
        )
        parser.add_argument(
            '--dodo_psf_spectral_separation_weight', type=float, default=0.02,
            help='Target weight for RGB-visible adjacent-wavelength PSF separation.',
        )
        parser.add_argument(
            '--dodo_psf_spectral_separation_margin', type=float, default=0.90,
            help='Maximum desired adjacent-wavelength cosine similarity.',
        )
        parser.add_argument(
            '--dodo_psf_spectral_separation_warmup_epochs', type=int, default=2,
            help='Epochs used to linearly warm up the spectral PSF loss weight.',
        )
        parser.add_argument(
            '--dodo_psf_spectral_separation_start_epoch', type=int, default=0,
            help='Zero-weight anchor epoch for the spectral PSF-loss ramp.')
        parser.add_argument(
            '--dodo_psf_spectral_hard_fraction', type=float, default=0.20)
        parser.add_argument(
            '--dodo_psf_spectral_hard_weight', type=float, default=0.5)
        parser.add_argument(
            '--dodo_psf_depth_separation_weight', type=float, default=0.005)
        parser.add_argument(
            '--dodo_psf_depth_separation_margin', type=float, default=0.90)
        parser.add_argument(
            '--dodo_psf_depth_separation_start_epoch', type=int, default=0)
        parser.add_argument(
            '--dodo_psf_depth_separation_warmup_epochs', type=int, default=0)
        parser.add_argument(
            '--dodo_psf_depth_hard_fraction', type=float, default=0.20)
        parser.add_argument(
            '--dodo_psf_depth_hard_weight', type=float, default=0.5)
        parser.add_argument('--dodo_psf_mtf_weight', type=float, default=0.25)
        parser.add_argument('--dodo_psf_mtf_start_epoch', type=int, default=0)
        parser.add_argument('--dodo_psf_mtf_warmup_epochs', type=int, default=0)
        parser.add_argument('--dodo_psf_mtf_min_frequency', type=float, default=0.02)
        parser.add_argument('--dodo_psf_mtf_max_frequency', type=float, default=0.15)
        parser.add_argument('--dodo_psf_mtf_target_005', type=float, default=0.12)
        parser.add_argument('--dodo_psf_mtf_target_010', type=float, default=0.05)
        parser.add_argument('--dodo_psf_mtf_target_015', type=float, default=0.025)
        parser.add_argument(
            '--dodo_optical_regularizer_max_ratio', type=float, default=0.0,
            help=(
                'Cap all weighted DOE regularizers to this fraction of the '
                'weighted reconstruction/depth task loss. Zero keeps the '
                'historical uncapped behavior.'
            ),
        )
        parser.add_argument('--depth_smooth_weight', type=float, default=0.01,
                    help='深度平滑正则权重（抑制颜色纹理串扰）')
        parser.add_argument('--metric_depth_loss_weight', type=float, default=0.0,
                    help='掩码内 metric-depth SmoothL1 损失权重（默认 0=关闭）')
        parser.add_argument('--psf_size', type=int, default=64)
        parser.add_argument('--l1_loss_weight', type=float, default=1.0)
        parser.add_argument('--sam_loss_weight', type=float, default=0.0)
        parser.add_argument('--mse_loss_weight', type=float, default=0.0)
        parser.add_argument('--spatial_gradient_loss_weight', type=float, default=0.0)
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
        parser.add_argument('--hs_norm_mode', type=str, default='fixed_scale',
                            choices=['scene_max', 'fixed_scale'],
                            help='HS target normalization mode; fixed_scale uses --hs_norm_scale for train/val/infer')
        parser.add_argument('--hs_norm_scale', type=float, default=0.9367284796834017,
                            help='Fixed HS scale used when --hs_norm_mode=fixed_scale')
        parser.add_argument('--hs_sanity_threshold', type=float, default=10000.0,
                            help='Clip EXR outliers above this threshold before HS normalization')
        parser.add_argument('--start_wl', type=float, default=420e-9, help='起始波长（米, 例如 420nm）')
        parser.add_argument('--end_wl', type=float, default=660e-9, help='结束波长（米, 例如 660nm）')
        parser.add_argument('--bayer', dest='bayer', action='store_true')
        parser.add_argument('--no-bayer', dest='bayer', action='store_false')
        parser.set_defaults(bayer=False)
        parser.add_argument('--checkpoint_monitor', type=str, default='val_loss')
        parser.add_argument('--checkpoint_mode', type=str, default='min')
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
        parser.add_argument('--dodo_forward_norm', type=str, default='fixed_scale',
                            choices=['legacy_max', 'none', 'per_sample_max', 'fixed_scale'],
                            help='DoDo forward internal measurement norm mode')
        parser.add_argument('--dodo_forward_scale', type=float, default=3.7003112959862983,
                            help='Fixed DoDo sensor scale used when --dodo_forward_norm=fixed_scale')
        parser.add_argument('--dodo_skip_prop2', dest='dodo_skip_prop2', action='store_true',
                            help='Skip the prop2 propagation stage between doe1 and optional doe2')
        parser.add_argument('--no-dodo_skip_prop2', dest='dodo_skip_prop2', action='store_false',
                            help='Keep the prop2 propagation stage between doe1 and optional doe2')
        parser.set_defaults(dodo_skip_prop2=False)
        parser.add_argument('--background_hs_loss_weight', type=float, default=0.02,
                            help='Background HS L1 loss weight for full-image visual quality')
        parser.add_argument('--dodo_sensing_mode', type=str, default='rgb',
                            choices=['rgb', 'spectral_bins', 'identity'],
                            help='DoDo sensing mode')
        parser.add_argument('--dodo_sensor_measurement', type=str, default='amplitude',
                            choices=['amplitude', 'intensity'],
                            help='DoDo sensor measurement type (amplitude=abs(field), intensity=abs(field)^2)')
        parser.add_argument('--decoder_use_rgb_pinv_prior', dest='decoder_use_rgb_pinv_prior',
                            action='store_true',
                            help='Enable a 25-channel ridge pseudo-inverse prior X0 computed from RGB measurement')
        parser.add_argument('--no-decoder_use_rgb_pinv_prior', dest='decoder_use_rgb_pinv_prior',
                            action='store_false',
                            help='Disable RGB pseudo-inverse prior (default)')
        parser.set_defaults(decoder_use_rgb_pinv_prior=False)
        parser.add_argument('--decoder_rgb_pinv_lambda', type=float, default=1e-3,
                            help='Ridge lambda for RGB sensor-response pseudo-inverse prior')
        parser.add_argument('--decoder_rgb_pinv_norm', type=str, default='per_sample_max',
                            choices=['none', 'per_sample_max', 'per_sample_mean_std'],
                            help='Normalization applied to the 25-channel RGB pseudo-inverse prior')
        parser.add_argument('--hs_residual_prior', dest='hs_residual_prior',
                            action='store_true',
                            help='Use RGB pseudo-inverse prior as HS logit baseline and predict residual logits')
        parser.add_argument('--no-hs_residual_prior', dest='hs_residual_prior',
                            action='store_false',
                            help='Use the RGB pseudo-inverse prior in the legacy concat-input path')
        parser.set_defaults(hs_residual_prior=False)
        parser.add_argument('--hs_residual_prior_eps', type=float, default=1e-4,
                            help='Clamp epsilon before logit(prior) when --hs_residual_prior is enabled')
        parser.add_argument('--detach_depth_guidance_for_hs',
                            dest='detach_depth_guidance_for_hs', action='store_true',
                            help='Keep HS depth guidance values but detach them so HS loss does not update the depth decoder/head through guidance')
        parser.add_argument('--no-detach_depth_guidance_for_hs',
                            dest='detach_depth_guidance_for_hs', action='store_false',
                            help='Allow HS loss gradients to flow into the depth decoder through HS depth guidance')
        parser.set_defaults(detach_depth_guidance_for_hs=False)
        parser.add_argument('--isolate_hs_decoder_gradients',
                            dest='isolate_hs_decoder_gradients', action='store_true',
                            help='Detach shared encoder/bottleneck and depth-guidance inputs before the HS decoder so HS loss only updates HS decoder/guidance modules')
        parser.add_argument('--no-isolate_hs_decoder_gradients',
                            dest='isolate_hs_decoder_gradients', action='store_false',
                            help='Allow HS loss gradients to update shared encoder/bottleneck features through the HS decoder')
        parser.set_defaults(isolate_hs_decoder_gradients=False)
        parser.add_argument('--decoder_rgb_pinv_unscale_measurement',
                            dest='decoder_rgb_pinv_unscale_measurement', action='store_true',
                            help='Undo fixed_scale forward normalization before applying the RGB pseudo-inverse')
        parser.add_argument('--no-decoder_rgb_pinv_unscale_measurement',
                            dest='decoder_rgb_pinv_unscale_measurement', action='store_false',
                            help='Apply the RGB pseudo-inverse directly to normalized captimgs')
        parser.set_defaults(decoder_rgb_pinv_unscale_measurement=True)
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
