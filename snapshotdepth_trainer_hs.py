# snapshotdepth_trainer_hs.py

import os
import inspect
import json
import math
from argparse import ArgumentParser
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint,Callback
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

# 确保这些我们创建的文件与此脚本在同一项目路径下
from datasets.hyperspectral_dataset import HyperspectralDepthDataset
from snapshotdepth_hs import SnapshotDepthHS as SnapshotDepth
from util.log_manager import LogManager

seed_everything(123)

# DoDo
class DOEParameterClampCallback(Callback):
    """
    Mimic Keras constraint=MinMaxNorm(...) for DOE parameters.
    """

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, *args, **kwargs):
        for module in pl_module.modules():
            clamp_fn = getattr(module, "clamp_parameters_", None)
            if callable(clamp_fn):
                clamp_fn()


class BestMetricTracker(Callback):
    """Track best validation metrics without writing checkpoint files."""

    def __init__(self, metrics, output_path):
        self.metrics = metrics
        self.output_path = output_path
        self.best = {
            name: {
                'monitor': monitor,
                'mode': mode,
                'best_epoch': None,
                'best_score': None,
            }
            for name, monitor, mode in metrics
        }

    @staticmethod
    def _to_float(value):
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                return None
            value = value.detach().cpu().item()
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(value):
            return None
        return value

    @staticmethod
    def _is_better(value, best_value, mode):
        if best_value is None:
            return True
        if mode == 'min':
            return value < best_value
        return value > best_value

    def _write_summary(self):
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, 'w') as f:
            json.dump(self.best, f, indent=2)

    def _update_from_trainer(self, trainer):
        if getattr(trainer, 'sanity_checking', False):
            return

        callback_metrics = getattr(trainer, 'callback_metrics', {}) or {}
        epoch = int(getattr(trainer, 'current_epoch', 0))
        changed = False

        for name, monitor, mode in self.metrics:
            value = self._to_float(callback_metrics.get(monitor))
            if value is None:
                continue

            entry = self.best[name]
            if self._is_better(value, entry['best_score'], mode):
                entry['best_score'] = value
                entry['best_epoch'] = epoch
                changed = True
                print(f'[best-metric] {name}: epoch={epoch:03d}, {monitor}={value:.6g}')

        if changed:
            self._write_summary()

    def on_validation_epoch_end(self, trainer, pl_module):
        self._update_from_trainer(trainer)

    def on_validation_end(self, trainer, pl_module):
        self._update_from_trainer(trainer)


class LossCurvePlotter(Callback):
    """Write lightweight loss and DOE convergence curves during training."""

    def __init__(self, output_dir, every_n_steps=50):
        self.output_dir = output_dir
        self.every_n_steps = max(1, int(every_n_steps))
        self.train_points = []
        self.val_points = []
        self.doe_points = []
        self.png_path = os.path.join(output_dir, 'train_loss.png')
        self.json_path = os.path.join(output_dir, 'loss_history.json')
        self.doe_png_path = os.path.join(output_dir, 'doe_convergence.png')
        self.doe_json_path = os.path.join(output_dir, 'doe_history.json')

    @staticmethod
    def _to_float(value):
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                return None
            value = value.detach().cpu().item()
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(value):
            return None
        return value

    @staticmethod
    def _is_global_zero(trainer):
        if hasattr(trainer, 'is_global_zero'):
            return bool(trainer.is_global_zero)
        return int(getattr(trainer, 'global_rank', 0) or 0) == 0

    def _append_train_loss(self, trainer, pl_module):
        stored = getattr(pl_module, '_last_train_loss_logs', None) or {}
        value = self._to_float(stored.get('total_loss'))
        if value is None:
            metrics = getattr(trainer, 'callback_metrics', {}) or {}
            value = self._to_float(metrics.get('train_loss/total_loss'))
        if value is None:
            return
        step = int(getattr(trainer, 'global_step', 0) or 0)
        if self.train_points and self.train_points[-1]['step'] == step:
            self.train_points[-1]['loss'] = value
        else:
            self.train_points.append({'step': step, 'loss': value})

    def _append_val_loss(self, trainer):
        metrics = getattr(trainer, 'callback_metrics', {}) or {}
        value = self._to_float(metrics.get('val_loss'))
        if value is None:
            return
        step = int(getattr(trainer, 'global_step', 0) or 0)
        epoch = int(getattr(trainer, 'current_epoch', 0) or 0)
        if self.val_points and self.val_points[-1]['step'] == step:
            self.val_points[-1].update({'epoch': epoch, 'loss': value})
        else:
            self.val_points.append({'step': step, 'epoch': epoch, 'loss': value})

    def _append_doe_metrics(self, pl_module):
        stored = getattr(pl_module, '_last_doe_metrics', None) or {}
        if 'step' not in stored:
            return

        point = {'step': int(stored['step'])}
        for key in ('update_rel', 'grad_norm', 'coeff_norm'):
            value = self._to_float(stored.get(key))
            if value is not None:
                point[key] = value
        if len(point) == 1:
            return

        if self.doe_points and self.doe_points[-1]['step'] == point['step']:
            self.doe_points[-1].update(point)
        else:
            self.doe_points.append(point)

    def _write_history(self):
        os.makedirs(self.output_dir, exist_ok=True)
        history = {
            'train_loss/total_loss': self.train_points,
            'val_loss': self.val_points,
        }
        tmp_path = self.json_path + '.tmp'
        with open(tmp_path, 'w') as f:
            json.dump(history, f, indent=2)
        os.replace(tmp_path, self.json_path)

    def _write_doe_history(self):
        if not self.doe_points:
            return
        os.makedirs(self.output_dir, exist_ok=True)
        history = {
            f'doe/{key}': [
                {'step': point['step'], 'value': point[key]}
                for point in self.doe_points
                if key in point
            ]
            for key in ('update_rel', 'grad_norm', 'coeff_norm')
        }
        tmp_path = self.doe_json_path + '.tmp'
        with open(tmp_path, 'w') as f:
            json.dump(history, f, indent=2)
        os.replace(tmp_path, self.doe_json_path)

    def _write_plot(self, trainer):
        if not self.train_points and not self.val_points:
            return
        os.makedirs(self.output_dir, exist_ok=True)
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except Exception as exc:
            print(f'[loss-plot] matplotlib unavailable, skip train_loss.png: {exc}')
            return

        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=140)
        if self.train_points:
            xs = [p['step'] for p in self.train_points]
            ys = [p['loss'] for p in self.train_points]
            ax.plot(xs, ys, label='train_loss/total_loss', color='#1f77b4', linewidth=1.2)
        if self.val_points:
            xs = [p['step'] for p in self.val_points]
            ys = [p['loss'] for p in self.val_points]
            ax.plot(xs, ys, label='val_loss', color='#d62728', marker='o', linewidth=1.6)
        ax.set_xlabel('global step')
        ax.set_ylabel('loss')
        ax.set_title(f'Loss Curve (epoch {int(getattr(trainer, "current_epoch", 0) or 0)})')
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        tmp_path = self.png_path + '.tmp.png'
        fig.savefig(tmp_path)
        plt.close(fig)
        os.replace(tmp_path, self.png_path)

    def _write_doe_plot(self, trainer):
        if not self.doe_points:
            return
        os.makedirs(self.output_dir, exist_ok=True)
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except Exception as exc:
            print(f'[doe-plot] matplotlib unavailable, skip doe_convergence.png: {exc}')
            return

        fig, (ax_top, ax_bottom) = plt.subplots(
            2, 1, figsize=(8, 6.5), dpi=140, sharex=True
        )
        series = (
            ('update_rel', 'DOE relative update', '#1f77b4'),
            ('grad_norm', 'DOE gradient norm', '#d62728'),
        )
        for key, label, color in series:
            points = [point for point in self.doe_points if key in point]
            if points:
                ax_top.plot(
                    [point['step'] for point in points],
                    [point[key] for point in points],
                    label=label,
                    color=color,
                    linewidth=1.3,
                )
        ax_top.set_yscale('symlog', linthresh=1e-12)
        ax_top.set_ylabel('update / gradient')
        ax_top.grid(True, alpha=0.25)
        ax_top.legend()

        coeff_points = [point for point in self.doe_points if 'coeff_norm' in point]
        if coeff_points:
            ax_bottom.plot(
                [point['step'] for point in coeff_points],
                [point['coeff_norm'] for point in coeff_points],
                label='DOE coefficient norm',
                color='#2ca02c',
                linewidth=1.3,
            )
        ax_bottom.axhline(1.0, color='#7f7f7f', linestyle='--', linewidth=1.0,
                          label='clamp boundary')
        ax_bottom.set_xlabel('global step')
        ax_bottom.set_ylabel('coefficient norm')
        ax_bottom.grid(True, alpha=0.25)
        ax_bottom.legend()

        fig.suptitle(
            f'DOE Convergence (epoch {int(getattr(trainer, "current_epoch", 0) or 0)})'
        )
        fig.tight_layout()
        tmp_path = self.doe_png_path + '.tmp.png'
        fig.savefig(tmp_path)
        plt.close(fig)
        os.replace(tmp_path, self.doe_png_path)

    def _flush(self, trainer):
        self._write_history()
        self._write_plot(trainer)
        self._write_doe_history()
        self._write_doe_plot(trainer)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, *args, **kwargs):
        if not self._is_global_zero(trainer):
            return
        self._append_train_loss(trainer, pl_module)
        self._append_doe_metrics(pl_module)
        step = int(getattr(trainer, 'global_step', 0) or 0)
        if step == 0 or step % self.every_n_steps == 0:
            self._flush(trainer)

    def on_validation_end(self, trainer, pl_module):
        if getattr(trainer, 'sanity_checking', False):
            return
        if not self._is_global_zero(trainer):
            return
        self._append_val_loss(trainer)
        self._flush(trainer)

    def on_train_end(self, trainer, pl_module):
        if not self._is_global_zero(trainer):
            return
        self._flush(trainer)


def _patch_pl_ddp_sync_params_if_missing():
    """兼容某些 torch 版本移除了 DDP 私有方法 `_sync_params` 的情况。"""
    try:
        from pytorch_lightning.overrides.data_parallel import LightningDistributedDataParallel
        if not hasattr(LightningDistributedDataParallel, '_sync_params'):
            LightningDistributedDataParallel._sync_params = lambda self: None
            print('[Compat] Patched LightningDistributedDataParallel._sync_params')
    except Exception:
        pass


def _strip_deprecated_trainer_args(args):
    """移除/转换 PL 1.5+ 废弃的 Trainer 参数。
    
    1. checkpoint_callback / resume_from_checkpoint: 已移除
    2. accelerator='ddp' -> strategy='ddp' + accelerator='auto'
    3. gpus -> devices (PL 1.7+)
    """
    # accelerator='ddp' 转换
    if hasattr(args, 'accelerator') and getattr(args, 'accelerator', None) == 'ddp':
        args.strategy = 'ddp'
        args.accelerator = 'auto'
        print('[Compat] 已将 accelerator=ddp 转换为 strategy=ddp, accelerator=auto')
    
    # gpus -> devices (如果 PL 版本需要)
    if hasattr(args, 'gpus') and getattr(args, 'gpus', None) is not None:
        if not hasattr(args, 'devices') or getattr(args, 'devices', None) is None:
            args.devices = args.gpus
            print(f'[Compat] 已将 gpus={args.gpus} 复制到 devices')
    
    # 删除废弃参数
    for key in ("checkpoint_callback", "resume_from_checkpoint"):
        if hasattr(args, key):
            try:
                delattr(args, key)
            except Exception:
                setattr(args, key, None)


def _scene_folders_from_range(start, end):
    start = int(start)
    end = int(end)
    if start < 1 or end < start:
        raise ValueError(f'Invalid scene range: start={start}, end={end}')
    return [f'deploy {i}' for i in range(start, end + 1)]


def prepare_data(hparams):
    """
    为高光谱数据准备训练和验证的 DataLoader。
    """
    train_folders = _scene_folders_from_range(
        getattr(hparams, 'train_scene_start', 1),
        getattr(hparams, 'train_scene_end', 15),
    )
    val_folders = _scene_folders_from_range(
        getattr(hparams, 'val_scene_start', 16),
        getattr(hparams, 'val_scene_end', 18),
    )

    # For DoDo mode, train always uses randcrop (measurement fixed at 128x128, crop needed for valid patches)
    optical_model = getattr(hparams, 'optical_model', 'legacy_camera')
    train_randcrop = True if optical_model == 'dodo_depth' else hparams.randcrop
    if train_randcrop != getattr(hparams, 'randcrop', False):
        print(f'[data] train randcrop: hparams={hparams.randcrop} → actual={train_randcrop} '
              f'(forced by optical_model={optical_model})')

    print(f"训练集文件夹数量: {len(train_folders)}")
    print(f"验证集文件夹数量: {len(val_folders)}")
    print(f"[data] train scene folders: {train_folders[0]} ... {train_folders[-1]}")
    print(f"[data] val scene folders: {val_folders[0]} ... {val_folders[-1]}")

    patch_index_path = getattr(hparams, 'patch_index_path', '')
    if patch_index_path == 'auto':
        patch_index_path = os.path.join(
            hparams.data_root,
            '.patch_index',
            f'patch{hparams.image_sz}_stride32_valid20_range060_center10_v1.npz',
        )
        print(f'[data] auto patch_index_path: {patch_index_path}')
    train_patch_index_path = str(
        getattr(hparams, 'train_patch_index_path', '') or patch_index_path
    )
    val_patch_index_path = str(
        getattr(hparams, 'val_patch_index_path', '') or patch_index_path
    )
    if train_patch_index_path:
        print(f'[data] train patch index: {train_patch_index_path}')
    if val_patch_index_path:
        print(f'[data] val patch index: {val_patch_index_path}')

    train_samples_per_epoch = int(getattr(hparams, 'train_samples_per_epoch', 0) or 0)
    if getattr(hparams, 'baek_patch_epoch', False):
        if train_samples_per_epoch > 0 and train_samples_per_epoch != 6143:
            print(f'[data] --baek_patch_epoch overrides train_samples_per_epoch='
                  f'{train_samples_per_epoch} -> 6143')
        train_samples_per_epoch = 6143
        setattr(hparams, 'train_samples_per_epoch', train_samples_per_epoch)

    if train_samples_per_epoch > 0:
        batch_sz = max(1, int(getattr(hparams, 'batch_sz', 1)))
        steps_per_epoch = (train_samples_per_epoch + batch_sz - 1) // batch_sz
        print(f'[data] patch-epoch mode: train_samples_per_epoch={train_samples_per_epoch}, '
              f'batch_sz={batch_sz}, estimated_train_steps_per_epoch={steps_per_epoch}')

    val_patch_eval_arg = getattr(hparams, 'val_patch_eval', None)
    if val_patch_eval_arg is None:
        val_patch_eval = bool(getattr(hparams, 'baek_patch_epoch', False))
    else:
        val_patch_eval = bool(val_patch_eval_arg)
    val_samples_per_epoch = int(getattr(hparams, 'val_samples_per_epoch', 0) or 0)
    train_patch_index_enumerate = bool(
        getattr(hparams, 'train_patch_index_enumerate', False)
    )
    if train_patch_index_enumerate:
        print('[data] fixed training patch-index enumeration enabled')
    if val_patch_eval:
        print(f'[data] fixed validation patch-index mode enabled; '
              f'val_samples_per_epoch={val_samples_per_epoch or "all"}')

    train_dataset = HyperspectralDepthDataset(
        base_dir=hparams.data_root,
        scene_folders=train_folders,
        image_size=(hparams.image_sz, hparams.image_sz),
        hs_channels=hparams.hs_channels,
        is_training=True,
        randcrop=train_randcrop,
        augment=hparams.augment,
        min_depth=hparams.min_depth,
        max_depth=hparams.max_depth,
        use_exr_cache=hparams.use_exr_cache,
        exr_cache_dir=hparams.exr_cache_dir,
        patch_filter=getattr(hparams, 'patch_filter', True),
        min_valid_ratio=getattr(hparams, 'min_valid_ratio', 0.12),
        min_depth_range_ips=getattr(hparams, 'min_depth_range_ips', 0.10),
        max_crop_retries=getattr(hparams, 'max_crop_retries', 8),
        patch_filter_stride=getattr(hparams, 'patch_filter_stride', 4),
        patch_index_path=train_patch_index_path,
        patch_index_jitter=getattr(hparams, 'patch_index_jitter', 16),
        patch_index_strict=getattr(hparams, 'patch_index_strict', True),
        patch_index_weighted=getattr(hparams, 'patch_index_weighted', False),
        patch_index_use_meta_thresholds=getattr(hparams, 'patch_index_use_meta_thresholds', True),
        min_center_valid_ratio=getattr(hparams, 'min_center_valid_ratio', 0.0),
        samples_per_epoch=train_samples_per_epoch,
        enumerate_patch_index=train_patch_index_enumerate,
        patch_category_mix=getattr(hparams, 'train_patch_category_mix', ''),
        patch_category_seed=getattr(hparams, 'train_patch_category_seed', 123),
        patch_index_hs_jitter=getattr(hparams, 'patch_index_hs_jitter', 8),
        hs_norm_mode=getattr(hparams, 'hs_norm_mode', 'scene_max'),
        hs_norm_scale=getattr(hparams, 'hs_norm_scale', 0.0),
        hs_sanity_threshold=getattr(hparams, 'hs_sanity_threshold', 10000.0),
        baek_augment=getattr(hparams, 'baek_augment', False),
        baek_scale_half_probability=getattr(
            hparams, 'baek_scale_half_probability', 0.30
        ),
        baek_depth_shift_m=getattr(hparams, 'baek_depth_shift_m', 0.20),
        baek_depth_shift_probability=getattr(
            hparams, 'baek_depth_shift_probability', 0.50
        ),
        baek_illuminant_probability=getattr(
            hparams, 'baek_illuminant_probability', 0.80
        ),
        baek_exposure_min=getattr(hparams, 'baek_exposure_min', 0.90),
        baek_exposure_max=getattr(hparams, 'baek_exposure_max', 1.10),
        baek_max_clip_ratio=getattr(hparams, 'baek_max_clip_ratio', 0.001),
        baek_illuminant_retries=getattr(
            hparams, 'baek_illuminant_retries', 8
        ),
    )

    val_dataset = HyperspectralDepthDataset(
        base_dir=hparams.data_root,
        scene_folders=val_folders,
        image_size=(hparams.image_sz, hparams.image_sz),
        hs_channels=hparams.hs_channels,
        is_training=False,
        randcrop=False,
        augment=False,
        min_depth=hparams.min_depth,
        max_depth=hparams.max_depth,
        use_exr_cache=hparams.use_exr_cache,
        exr_cache_dir=hparams.exr_cache_dir,
        patch_filter=getattr(hparams, 'patch_filter', True),
        min_valid_ratio=getattr(hparams, 'min_valid_ratio', 0.12),
        min_depth_range_ips=getattr(hparams, 'min_depth_range_ips', 0.10),
        max_crop_retries=getattr(hparams, 'max_crop_retries', 8),
        patch_filter_stride=getattr(hparams, 'patch_filter_stride', 4),
        patch_index_path=val_patch_index_path if val_patch_eval else '',
        patch_index_jitter=0 if val_patch_eval else getattr(hparams, 'patch_index_jitter', 16),
        patch_index_strict=getattr(hparams, 'patch_index_strict', True),
        patch_index_weighted=getattr(hparams, 'patch_index_weighted', False),
        patch_index_use_meta_thresholds=getattr(hparams, 'patch_index_use_meta_thresholds', True),
        min_center_valid_ratio=getattr(hparams, 'min_center_valid_ratio', 0.0),
        samples_per_epoch=val_samples_per_epoch,
        eval_patch_index=val_patch_eval,
        hs_norm_mode=getattr(hparams, 'hs_norm_mode', 'scene_max'),
        hs_norm_scale=getattr(hparams, 'hs_norm_scale', 0.0),
        hs_sanity_threshold=getattr(hparams, 'hs_sanity_threshold', 10000.0),
    )

    train_dataloader = DataLoader(train_dataset, batch_size=hparams.batch_sz,
                                  num_workers=hparams.num_workers, shuffle=True, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=hparams.batch_sz,
                                num_workers=hparams.num_workers, shuffle=False, pin_memory=True)

    return train_dataloader, val_dataloader


def main(args):
    _patch_pl_ddp_sync_params_if_missing()
    _strip_deprecated_trainer_args(args)

    logger = TensorBoardLogger(args.default_root_dir, name=args.experiment_name)
    logmanager_callback = LogManager()

    # Determine artifact root FIRST: CLI → DODO_ARTIFACT_ROOT → EXP_ROOT → legacy fallback
    # Must happen before checkpoint creation so checkpoints land under artifact_dir/checkpoints/.
    import sys, subprocess
    raw_cli = getattr(args, 'artifact_root', '')
    require_root = getattr(args, 'require_artifact_root', False)

    print(f'[artifact] raw sys.argv ({len(sys.argv)} tokens): {" ".join(sys.argv[:6])}...')
    print(f'[artifact] CLI --artifact_root = {repr(raw_cli)}')
    print(f'[artifact] env DODO_ARTIFACT_ROOT = {repr(os.environ.get("DODO_ARTIFACT_ROOT", ""))}')
    print(f'[artifact] env EXP_ROOT = {repr(os.environ.get("EXP_ROOT", ""))}')
    print(f'[artifact] --require_artifact_root = {require_root}')

    if raw_cli:
        artifact_dir = raw_cli
    elif os.environ.get('DODO_ARTIFACT_ROOT', ''):
        artifact_dir = os.environ['DODO_ARTIFACT_ROOT']
        print(f'[artifact] resolved from DODO_ARTIFACT_ROOT')
    elif os.environ.get('EXP_ROOT', ''):
        artifact_dir = os.environ['EXP_ROOT']
        print(f'[artifact] resolved from EXP_ROOT')
    else:
        artifact_dir = os.path.join('infer_results', 'DoDo-change',
                                    args.experiment_name,
                                    f'version_{logger.version}')
        if require_root:
            raise ValueError(
                '--require_artifact_root is set but no artifact_root could be resolved. '
                'Set --artifact_root, DODO_ARTIFACT_ROOT, or EXP_ROOT.'
            )
        print(f'[artifact] WARNING: using legacy fallback path (no --artifact_root/DODO_ARTIFACT_ROOT/EXP_ROOT)')

    # Store resolved artifact_root back in args so hparams.json records it correctly
    args.artifact_root = artifact_dir

    logs_dir = os.path.join(artifact_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    ckpt_dir = os.path.join(artifact_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    # Save command.txt
    with open(os.path.join(artifact_dir, 'command.txt'), 'w') as f:
        f.write(' '.join(sys.argv) + '\n')
    # Save hparams.json (now with resolved artifact_root)
    try:
        with open(os.path.join(artifact_dir, 'hparams.json'), 'w') as f:
            json.dump({k: str(v) for k, v in vars(args).items()}, f, indent=2)
    except Exception:
        pass
    # Save git_status.txt
    try:
        git_status = subprocess.run(['git', 'status', '--porcelain'],
                                    capture_output=True, text=True, timeout=10,
                                    cwd=os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(artifact_dir, 'git_status.txt'), 'w') as f:
            f.write(git_status.stdout if git_status.stdout else '(clean or not a git repo)\n')
        git_revision = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            check=True,
        )
        with open(os.path.join(artifact_dir, 'git_commit.txt'), 'w') as f:
            f.write(git_revision.stdout.strip() + '\n')
    except Exception:
        with open(os.path.join(artifact_dir, 'git_status.txt'), 'w') as f:
            f.write('(git status unavailable)\n')
    print(f'[artifact] resolved root={artifact_dir}')
    print(f'[artifact] checkpoint dir={ckpt_dir}')

    # --- Checkpoint callbacks ---
    # Default policy: only joint-best writes a checkpoint. Auxiliary best metrics
    # are tracked in JSON to avoid duplicating large .ckpt files.
    # Use artifact_dir for checkpoint storage with stable filenames (no fake 0.0000 metric values).
    ckpt_monitor = getattr(args, 'checkpoint_monitor', 'validation/psnr_hs_masked')
    ckpt_mode = getattr(args, 'checkpoint_mode', 'max')
    try:
        checkpoint_callback = ModelCheckpoint(
            monitor=ckpt_monitor,
            dirpath=ckpt_dir,
            filename='joint-best-{epoch:03d}',
            save_top_k=1,
            mode=ckpt_mode,
            verbose=True,
        )
    except TypeError:
        checkpoint_callback = ModelCheckpoint(
            verbose=True,
            monitor=ckpt_monitor,
            filepath=os.path.join(ckpt_dir, 'joint-best-{epoch:03d}'),
            save_top_k=1,
            mode=ckpt_mode,
        )

    auxiliary_best_metrics = [
        ('depth-best', 'validation/mae_depth_m', 'min'),
        ('hs-best', 'validation/hs_l1_masked', 'min'),
    ]
    checkpoint_callbacks_for_compat = [checkpoint_callback]
    auxiliary_callbacks = []

    if getattr(args, 'save_aux_best_ckpts', False):
        for metric_name, metric_monitor, metric_mode in auxiliary_best_metrics:
            try:
                auxiliary_callback = ModelCheckpoint(
                    monitor=metric_monitor,
                    dirpath=ckpt_dir,
                    filename=f'{metric_name}-{{epoch:03d}}',
                    save_top_k=1,
                    mode=metric_mode,
                    verbose=True,
                )
            except TypeError:
                auxiliary_callback = ModelCheckpoint(
                    verbose=True,
                    monitor=metric_monitor,
                    filepath=os.path.join(ckpt_dir, f'{metric_name}-{{epoch:03d}}'),
                    save_top_k=1,
                    mode=metric_mode,
                )
            auxiliary_callbacks.append(auxiliary_callback)
            checkpoint_callbacks_for_compat.append(auxiliary_callback)
        print('[checkpoint] auxiliary best checkpoints enabled: depth-best, hs-best')
    else:
        best_metric_path = os.path.join(artifact_dir, 'best_metric_epochs.json')
        auxiliary_callbacks.append(BestMetricTracker(auxiliary_best_metrics, best_metric_path))
        print(f'[checkpoint] auxiliary best checkpoints disabled; tracking epochs in {best_metric_path}')

    model = SnapshotDepth(hparams=args, log_dir=logger.log_dir, artifact_root=artifact_dir)
    train_dataloader, val_dataloader = prepare_data(hparams=args)

    # --- Load initial checkpoint weights if requested (fresh optimizer) ---
    init_ckpt = getattr(args, 'init_ckpt_path', '') or getattr(args, 'validate_only_ckpt', '')
    if init_ckpt:
        print(f'[init] Loading checkpoint weights from {init_ckpt}')
        checkpoint = torch.load(init_ckpt, map_location='cpu')
        # Handle both PL checkpoint dict and raw state_dict
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f'[init] Missing keys ({len(missing)}): {missing[:8]}...')
        if unexpected:
            print(f'[init] Unexpected keys ({len(unexpected)}): {unexpected[:8]}...')
        if not missing and not unexpected:
            print('[init] Weights loaded with exact key match.')
        # Record checkpoint source
        args.init_ckpt_loaded = init_ckpt

    # 兼容不同 PL 版本的 Trainer 初始化参数：
    # 1) 老版本依赖 checkpoint_callback=... 来注入 save_function
    # 2) 新版本通常通过 callbacks=[...] 传入 checkpoint callback
    loss_plot_callback = LossCurvePlotter(
        artifact_dir,
        every_n_steps=getattr(args, 'loss_plot_every_n_steps', 50),
    )
    callbacks = [logmanager_callback, checkpoint_callback] + auxiliary_callbacks + [
        loss_plot_callback,
        DOEParameterClampCallback()
    ]
    trainer_init_params = inspect.signature(Trainer.__init__).parameters
    trainer_kwargs = dict(
        logger=logger,
        sync_batchnorm=True,
        benchmark=True,
    )

    if 'callbacks' in trainer_init_params:
        trainer_kwargs['callbacks'] = callbacks

    trainer = Trainer.from_argparse_args(args, **trainer_kwargs)

    # 老版本 Lightning 在某些路径不会自动注入 save_function，做兜底。
    for callback in checkpoint_callbacks_for_compat:
        if getattr(callback, 'save_function', None) is None:
            callback.save_function = trainer.save_checkpoint
            print(f'[Compat] Set {callback.__class__.__name__}.save_function = trainer.save_checkpoint')

    validate_only = getattr(args, 'validate_only_ckpt', '')
    if validate_only:
        # --- Validation-only mode: no training, no optimizers ---
        eval_tag = getattr(args, 'eval_tag', '') or 'eval'
        print(f'[eval] Validation-only mode. Checkpoint: {validate_only}')
        print(f'[eval] Tag: {eval_tag}')
        model._eval_tag = eval_tag

        # PL 1.0.2: no trainer.validate(). Use trainer.test() which calls test_step.
        test_sig = inspect.signature(trainer.test)
        test_kwargs = {}
        if 'test_dataloaders' in test_sig.parameters:
            test_kwargs['test_dataloaders'] = val_dataloader
        elif 'dataloaders' in test_sig.parameters:
            test_kwargs['dataloaders'] = val_dataloader
        else:
            test_kwargs['model'] = model

        trainer.test(model, **test_kwargs)
        print('[eval] Validation-only run complete.')
        return

    # 兼容不同 PL 版本的 fit 参数命名
    fit_params = inspect.signature(trainer.fit).parameters
    fit_kwargs = {}

    if 'train_dataloaders' in fit_params:
        fit_kwargs['train_dataloaders'] = train_dataloader
    else:
        fit_kwargs['train_dataloader'] = train_dataloader

    if 'val_dataloaders' in fit_params:
        fit_kwargs['val_dataloaders'] = val_dataloader
    elif 'val_dataloader' in fit_params:
        fit_kwargs['val_dataloader'] = val_dataloader

    trainer.fit(model, **fit_kwargs)




if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)

    parser.add_argument('--experiment_name', type=str, default='Hyperspectral_LearnedDepth')

    # --- Checkpoint eval / resume ---
    parser.add_argument('--validate_only_ckpt', type=str, default='',
                        help='仅验证模式：加载 checkpoint 并运行 validation，不训练')
    parser.add_argument('--eval_tag', type=str, default='',
                        help='验证/评估标签，记录到 metrics.json')
    parser.add_argument('--init_ckpt_path', type=str, default='',
                        help='从指定 checkpoint 初始化模型权重（optimizer 从零开始）')
    parser.add_argument('--save_aux_best_ckpts', dest='save_aux_best_ckpts', action='store_true',
                        help='保存 depth-best/hs-best 辅助 checkpoint；默认只记录 best epoch/score，不落盘 ckpt')
    parser.add_argument('--no-save_aux_best_ckpts', dest='save_aux_best_ckpts', action='store_false')
    parser.set_defaults(save_aux_best_ckpts=False)
    parser.add_argument('--loss_plot_every_n_steps', type=int, default=50,
                        help='每隔多少个 global step 刷新 loss 与 DOE 收敛曲线')

    # --- 核心修改点：动态计算默认的数据集路径 ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_path = os.path.join(script_dir, "Baek数据集")
    parser.add_argument('--data_root', type=str,
                        default=default_data_path,
                        help="包含 'deploy X' 文件夹的数据集根目录")
    parser.add_argument('--train_scene_start', type=int, default=1,
                        help='训练集起始 deploy 编号，默认 1')
    parser.add_argument('--train_scene_end', type=int, default=15,
                        help='训练集结束 deploy 编号，默认 15')
    parser.add_argument('--val_scene_start', type=int, default=16,
                        help='验证集起始 deploy 编号，默认 16')
    parser.add_argument('--val_scene_end', type=int, default=18,
                        help='验证集结束 deploy 编号，默认 18')

    parser.add_argument('--use_exr_cache', dest='use_exr_cache', action='store_true',
                        help='启用 EXR 原始读取缓存（严格等价：仅缓存 read_exr 输出）')
    parser.add_argument('--no-use_exr_cache', dest='use_exr_cache', action='store_false',
                        help='关闭 EXR 读取缓存')
    parser.set_defaults(use_exr_cache=True)
    parser.add_argument('--exr_cache_dir', type=str, default='',
                        help='EXR 缓存目录；留空则默认放在 data_root/.exr_cache_npy_v1')

    # 从 Trainer, Model 添加所有需要的参数
    parser = Trainer.add_argparse_args(parser)
    parser = SnapshotDepth.add_model_specific_args(parser)

    # 设置一些默认值
    parser.set_defaults(
        # gpus=1, # 在M1上运行时，最好从命令行指定
        default_root_dir=os.path.join(script_dir, 'data'),
        max_epochs=100,
    )

    args = parser.parse_args()
    args.default_root_dir = os.path.abspath(args.default_root_dir)

    print("-" * 50)
    print(f"数据根目录将使用: {args.data_root}")
    print(f"训练输出目录将使用: {args.default_root_dir}")
    print("-" * 50)

    main(args)
