# snapshotdepth_trainer_hs.py

import os
import inspect
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


def prepare_data(hparams):
    """
    为高光谱数据准备训练和验证的 DataLoader。
    """
    all_scene_folders = [f'deploy {i}' for i in range(1, 19)]
    train_folders = all_scene_folders[:15]
    val_folders = all_scene_folders[15:]

    # For DoDo mode, train always uses randcrop (measurement fixed at 128x128, crop needed for valid patches)
    optical_model = getattr(hparams, 'optical_model', 'legacy_camera')
    train_randcrop = True if optical_model == 'dodo_depth' else hparams.randcrop
    if train_randcrop != getattr(hparams, 'randcrop', False):
        print(f'[data] train randcrop: hparams={hparams.randcrop} → actual={train_randcrop} '
              f'(forced by optical_model={optical_model})')

    print(f"训练集文件夹数量: {len(train_folders)}")
    print(f"验证集文件夹数量: {len(val_folders)}")

    patch_index_path = getattr(hparams, 'patch_index_path', '')
    if patch_index_path == 'auto':
        patch_index_path = os.path.join(
            hparams.data_root,
            '.patch_index',
            f'patch{hparams.image_sz}_stride32_valid20_range060_center10_v1.npz',
        )
        print(f'[data] auto patch_index_path: {patch_index_path}')

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
        patch_index_path=patch_index_path,
        patch_index_jitter=getattr(hparams, 'patch_index_jitter', 16),
        patch_index_strict=getattr(hparams, 'patch_index_strict', True),
        patch_index_weighted=getattr(hparams, 'patch_index_weighted', False),
        patch_index_use_meta_thresholds=getattr(hparams, 'patch_index_use_meta_thresholds', True),
        min_center_valid_ratio=getattr(hparams, 'min_center_valid_ratio', 0.0),
        samples_per_epoch=train_samples_per_epoch,
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
        patch_index_path=patch_index_path if val_patch_eval else '',
        patch_index_jitter=0 if val_patch_eval else getattr(hparams, 'patch_index_jitter', 16),
        patch_index_strict=getattr(hparams, 'patch_index_strict', True),
        patch_index_weighted=getattr(hparams, 'patch_index_weighted', False),
        patch_index_use_meta_thresholds=getattr(hparams, 'patch_index_use_meta_thresholds', True),
        min_center_valid_ratio=getattr(hparams, 'min_center_valid_ratio', 0.0),
        samples_per_epoch=val_samples_per_epoch,
        eval_patch_index=val_patch_eval,
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
    import json, sys, subprocess
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
    except Exception:
        with open(os.path.join(artifact_dir, 'git_status.txt'), 'w') as f:
            f.write('(git status unavailable)\n')
    print(f'[artifact] resolved root={artifact_dir}')
    print(f'[artifact] checkpoint dir={ckpt_dir}')

    # --- Checkpoint callbacks (Joint-best + Depth-MAE-best) ---
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

    depth_ckpt_monitor = 'validation/mae_depth_m'
    try:
        depth_checkpoint_callback = ModelCheckpoint(
            monitor=depth_ckpt_monitor,
            dirpath=ckpt_dir,
            filename='depth-best-{epoch:03d}',
            save_top_k=1,
            mode='min',
            verbose=True,
        )
    except TypeError:
        depth_checkpoint_callback = ModelCheckpoint(
            verbose=True,
            monitor=depth_ckpt_monitor,
            filepath=os.path.join(ckpt_dir, 'depth-best-{epoch:03d}'),
            save_top_k=1,
            mode='min',
        )

    hs_ckpt_monitor = 'validation/hs_l1_masked'
    try:
        hs_checkpoint_callback = ModelCheckpoint(
            monitor=hs_ckpt_monitor,
            dirpath=ckpt_dir,
            filename='hs-best-{epoch:03d}',
            save_top_k=1,
            mode='min',
            verbose=True,
        )
    except TypeError:
        hs_checkpoint_callback = ModelCheckpoint(
            verbose=True,
            monitor=hs_ckpt_monitor,
            filepath=os.path.join(ckpt_dir, 'hs-best-{epoch:03d}'),
            save_top_k=1,
            mode='min',
        )

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
    callbacks = [logmanager_callback, checkpoint_callback, depth_checkpoint_callback,
                 hs_checkpoint_callback, DOEParameterClampCallback()]
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
    if getattr(checkpoint_callback, 'save_function', None) is None:
        checkpoint_callback.save_function = trainer.save_checkpoint
        print('[Compat] Set checkpoint_callback.save_function = trainer.save_checkpoint')
    if getattr(depth_checkpoint_callback, 'save_function', None) is None:
        depth_checkpoint_callback.save_function = trainer.save_checkpoint
        print('[Compat] Set depth_checkpoint_callback.save_function = trainer.save_checkpoint')
    if getattr(hs_checkpoint_callback, 'save_function', None) is None:
        hs_checkpoint_callback.save_function = trainer.save_checkpoint
        print('[Compat] Set hs_checkpoint_callback.save_function = trainer.save_checkpoint')

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

    # --- 核心修改点：动态计算默认的数据集路径 ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_path = os.path.join(script_dir, "Baek数据集")
    parser.add_argument('--data_root', type=str,
                        default=default_data_path,
                        help="包含 'deploy X' 文件夹的数据集根目录")

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
