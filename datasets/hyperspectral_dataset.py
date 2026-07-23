import os
import re
import hashlib
import math
import json
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple, List, Optional, Dict
from kornia.augmentation import CenterCrop
# 假设 transform 能够处理多通道输入 (C, H, W)，如果你的 GenericRandomTransform 
# 内部写死了只处理单通道深度图，可能需要微调，但通常 Kornia/TorchVision 都能处理多通道。
from .generic_transform import GenericRandomTransform
from .baek_augmentation import (
    CIE_ILLUMINANT_NAMES,
    apply_cie_illuminant,
    translate_metric_depth,
)
from util.helper import metric_to_ips

import OpenEXR
import Imath

STRATIFIED_CATEGORY_BITS = {
    'depth_hard': 1,
    'hs_bright': 2,
    'hs_complex': 4,
    'general': 8,
}


def parse_patch_category_mix(value: str) -> Dict[str, float]:
    text = str(value or '').strip()
    if not text:
        return {}
    result = {}
    for item in text.split(','):
        name, separator, ratio = item.strip().partition('=')
        if not separator:
            raise ValueError(
                f"Invalid patch category mix item {item!r}; expected name=ratio"
            )
        name = name.strip()
        if name not in STRATIFIED_CATEGORY_BITS:
            raise ValueError(
                f"Unknown patch category {name!r}; "
                f"expected one of {sorted(STRATIFIED_CATEGORY_BITS)}"
            )
        result[name] = float(ratio)
    if any(value < 0.0 for value in result.values()):
        raise ValueError('Patch category ratios must be non-negative')
    total = sum(result.values())
    if total <= 0.0:
        raise ValueError('Patch category ratios must sum to a positive value')
    return {name: ratio / total for name, ratio in result.items()}


def read_exr(file_path):
    """
    使用OpenEXR库读取.exr文件并返回一个NumPy数组。
    """
    if not OpenEXR.isOpenExrFile(file_path):
        raise IOError(f"文件不是一个有效的EXR文件: {file_path}")

    exr_file = OpenEXR.InputFile(file_path)
    header = exr_file.header()

    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    channels_info = header['channels']
    channel_names = sorted(channels_info.keys())
    
    if not channel_names:
        raise ValueError(f"EXR文件没有任何通道: {file_path}")

    first_channel_type = channels_info[channel_names[0]]
    if first_channel_type.type == Imath.PixelType(Imath.PixelType.FLOAT):
        dtype = np.float32
    elif first_channel_type.type == Imath.PixelType(Imath.PixelType.HALF):
        dtype = np.float16
    else:
        raise TypeError(f"不支持的EXR数据类型: {first_channel_type.type}")

    all_channels_bytes = exr_file.channels(channel_names)

    np_channels = []
    for i, name in enumerate(channel_names):
        channel_data = np.frombuffer(all_channels_bytes[i], dtype=dtype)
        channel_data = channel_data.reshape(height, width)
        np_channels.append(channel_data)

    image_np = np.stack(np_channels, axis=-1)
    
    # 稍微放宽全0检查，避免浮点数精度极低的情况误判，但通常全0就是0
    if np.allclose(image_np, 0):
        raise ValueError(f"❌ EXR文件 {file_path} 读取后全为零！")

    return image_np


def normalize_hs_image(
    hs_image: np.ndarray,
    norm_mode: str = 'scene_max',
    norm_scale: float = 0.0,
    sanity_threshold: float = 10000.0,
) -> np.ndarray:
    """Normalize HS data with the same contract for training and inference."""
    hs_image = hs_image.astype(np.float32, copy=False)
    min_hs = 0.0

    if np.max(hs_image) > sanity_threshold:
        valid_pixels = hs_image < sanity_threshold
        scene_max = float(np.max(hs_image[valid_pixels])) if np.any(valid_pixels) else 1.0
        hs_image = np.clip(hs_image, min_hs, scene_max)
    else:
        scene_max = float(np.max(hs_image))

    norm_mode = str(norm_mode or 'scene_max')
    if norm_mode == 'scene_max':
        scale = scene_max
    elif norm_mode == 'fixed_scale':
        scale = float(norm_scale)
        if scale <= 0.0:
            raise ValueError('hs_norm_mode=fixed_scale requires --hs_norm_scale > 0')
    else:
        raise ValueError(f"Unsupported hs_norm_mode: {norm_mode}")

    if scale > min_hs:
        hs_image = (hs_image - min_hs) / (scale - min_hs)
    if norm_mode == 'fixed_scale':
        hs_image = np.clip(hs_image, 0.0, 1.0)
    return hs_image.astype(np.float32, copy=False)


class HyperspectralDepthDataset(Dataset):
    def __init__(self, base_dir: str, scene_folders: List[str], image_size: Tuple[int, int], hs_channels: int,
                 is_training: bool = True, randcrop: bool = False, augment: bool = False,
                 min_depth: float = 0.4, max_depth: float = 2.0,
                 use_exr_cache: bool = False, exr_cache_dir: str = '',
                 patch_filter: bool = True, min_valid_ratio: float = 0.12,
                 min_depth_range_ips: float = 0.10, max_crop_retries: int = 8,
                 patch_filter_stride: int = 4,
                 patch_index_path: str = '', patch_index_jitter: int = 16,
                 patch_index_strict: bool = True, patch_index_weighted: bool = False,
                 patch_index_use_meta_thresholds: bool = True,
                 min_center_valid_ratio: float = 0.0,
                 samples_per_epoch: int = 0,
                 eval_patch_index: bool = False,
                 enumerate_patch_index: bool = False,
                 patch_category_mix: str = '',
                 patch_category_seed: int = 123,
                 patch_index_hs_jitter: int = 8,
                 hs_norm_mode: str = 'scene_max',
                 hs_norm_scale: float = 0.0,
                 hs_sanity_threshold: float = 10000.0,
                 baek_augment: bool = False,
                 baek_scale_half_probability: float = 0.30,
                 baek_depth_shift_m: float = 0.20,
                 baek_depth_shift_probability: float = 0.50,
                 baek_illuminant_probability: float = 0.80,
                 baek_exposure_min: float = 0.90,
                 baek_exposure_max: float = 1.10,
                 baek_max_clip_ratio: float = 0.001,
                 baek_illuminant_retries: int = 8,
                 optical_halo: int = 0):
        
        super().__init__()
        self.is_training = is_training

        if isinstance(image_size, int):
            image_size = (image_size, image_size)

        self.transform = GenericRandomTransform(image_size, randcrop, augment, hs_channels)
        self.centercrop = CenterCrop(image_size)
        self.hs_channels = int(hs_channels)

        self.min_depth = min_depth
        self.max_depth = max_depth

        self.image_size = tuple(image_size)
        self.randcrop = bool(randcrop)
        self.augment = bool(augment)

        self.patch_filter = bool(patch_filter and self.is_training and self.randcrop)
        self.min_valid_ratio = float(min_valid_ratio)
        self.min_depth_range_ips = float(min_depth_range_ips)
        self.min_center_valid_ratio = float(min_center_valid_ratio)
        self.max_crop_retries = max(1, int(max_crop_retries))
        self.patch_filter_stride = max(1, int(patch_filter_stride))
        self.patch_index_path = str(patch_index_path or '')
        self.patch_index_jitter = max(0, int(patch_index_jitter))
        self.patch_index_strict = bool(patch_index_strict)
        self.patch_index_weighted = bool(patch_index_weighted)
        self.patch_index_use_meta_thresholds = bool(patch_index_use_meta_thresholds)
        self.samples_per_epoch = max(0, int(samples_per_epoch or 0))
        self.eval_patch_index = bool(eval_patch_index and not self.is_training)
        self.enumerate_patch_index = bool(enumerate_patch_index and self.is_training)
        self.patch_category_mix = parse_patch_category_mix(patch_category_mix)
        self.patch_category_seed = int(patch_category_seed)
        self.patch_index_hs_jitter = max(0, int(patch_index_hs_jitter))
        self.hs_norm_mode = str(hs_norm_mode or 'scene_max')
        self.hs_norm_scale = float(hs_norm_scale or 0.0)
        self.hs_sanity_threshold = float(hs_sanity_threshold or 10000.0)
        self.baek_augment = bool(baek_augment and self.is_training)
        self.baek_scale_half_probability = float(baek_scale_half_probability)
        self.baek_depth_shift_m = float(baek_depth_shift_m)
        self.baek_depth_shift_probability = float(baek_depth_shift_probability)
        self.baek_illuminant_probability = float(baek_illuminant_probability)
        self.baek_exposure_min = float(baek_exposure_min)
        self.baek_exposure_max = float(baek_exposure_max)
        self.baek_max_clip_ratio = float(baek_max_clip_ratio)
        self.baek_illuminant_retries = max(1, int(baek_illuminant_retries))
        self.optical_halo = int(optical_halo)
        if self.optical_halo < 0:
            raise ValueError(f'optical_halo must be >= 0, got {self.optical_halo}')
        for name, probability in (
            ('baek_scale_half_probability', self.baek_scale_half_probability),
            ('baek_depth_shift_probability', self.baek_depth_shift_probability),
            ('baek_illuminant_probability', self.baek_illuminant_probability),
        ):
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f'{name} must be in [0, 1], got {probability}')
        if self.baek_depth_shift_m < 0.0:
            raise ValueError('baek_depth_shift_m must be non-negative')
        if not 0.0 < self.baek_exposure_min <= self.baek_exposure_max:
            raise ValueError('Baek exposure range must be positive and ordered')
        if not 0.0 <= self.baek_max_clip_ratio <= 1.0:
            raise ValueError('baek_max_clip_ratio must be in [0, 1]')
        if self.baek_augment:
            if self.hs_channels != 25:
                raise ValueError('Baek CIE augmentation requires hs_channels=25')
            if self.hs_norm_mode != 'fixed_scale' or self.hs_norm_scale <= 0.0:
                raise ValueError(
                    'Baek augmentation requires fixed global HS normalization scale'
                )
        self.patch_index_by_id: Dict[str, Dict[str, torch.Tensor]] = {}
        self.patch_index_meta: Dict = {}
        self.patch_index_windows: List[Tuple[str, int, int, bool]] = []
        self.patch_index_category_bits = dict(STRATIFIED_CATEGORY_BITS)
        self.patch_index_scene_thresholds: Dict[str, Dict[str, float]] = {}
        self.stratified_category_schedule: Dict[str, List[str]] = {}
        self._hs_scene_scale_cache: Dict[str, float] = {}

        if self.baek_augment:
            print(
                '[Baek Augment] enabled: scale={1.0,0.5}, depth_shift='
                f'{{-{self.baek_depth_shift_m:.2f},0,+{self.baek_depth_shift_m:.2f}}}m, '
                f'illuminants={len(CIE_ILLUMINANT_NAMES)}, exposure='
                f'[{self.baek_exposure_min:.2f},{self.baek_exposure_max:.2f}], '
                f'max_clip={self.baek_max_clip_ratio:.4f}'
            )

        self.use_exr_cache = use_exr_cache
        if not exr_cache_dir:
            exr_cache_dir = os.path.join(base_dir, '.exr_cache_npy_v1')
        self.exr_cache_dir = exr_cache_dir
        if self.use_exr_cache:
            os.makedirs(self.exr_cache_dir, exist_ok=True)
            mode = 'train' if self.is_training else 'val'
            print(f"[Dataset Cache] mode={mode}, dir={self.exr_cache_dir}")

        if self.patch_filter:
            print(
                f"[Patch Filter] enabled (fast), min_valid_ratio={self.min_valid_ratio:.3f}, "
                f"min_depth_range_ips={self.min_depth_range_ips:.3f}, retries={self.max_crop_retries}, "
                f"stride={self.patch_filter_stride}"
            )

        self.sample_pairs = []
        for folder_name in scene_folders:
            match = re.search(r'\d+', folder_name)
            if not match: continue
            scene_num = match.group(0).zfill(2)

            hs_path = os.path.join(base_dir, folder_name, f'scene{scene_num}_hs.exr')
            depth_path = os.path.join(base_dir, folder_name, f'scene{scene_num}_depth_map.exr')

            if os.path.exists(hs_path) and os.path.exists(depth_path):
                self.sample_pairs.append({'hs_path': hs_path, 'depth_path': depth_path, 'id': f'scene_{scene_num}'})
            else:
                # 仅在调试时打印，避免刷屏
                pass
        self.sample_by_id = {sample['id']: sample for sample in self.sample_pairs}

        if (
            self.patch_filter
            or self.eval_patch_index
            or self.enumerate_patch_index
            or (self.is_training and self.patch_index_path and self.patch_index_weighted)
            or (self.baek_augment and self.patch_index_path)
        ) and self.patch_index_path:
            self._load_patch_index(self.patch_index_path)
        if (self.is_training and self.optical_halo > 0
                and self.patch_index_by_id):
            indexed_halo = self.patch_index_meta.get('optical_halo')
            if indexed_halo is None or int(indexed_halo) != self.optical_halo:
                raise ValueError(
                    'Training with optical_halo requires a halo-safe patch index '
                    f'generated for the same halo. Requested halo={self.optical_halo}, '
                    f'index meta optical_halo={indexed_halo!r}.')
            if self.patch_index_jitter != 0:
                raise ValueError(
                    'patch_index_jitter must be 0 with a halo-safe index; jitter '
                    'would invalidate the verified train/validation exclusion.')
        if self.is_training and self.patch_category_mix:
            self._build_stratified_category_schedule()

        if self.eval_patch_index:
            print(
                f"[Dataset] fixed eval patch-index mode: "
                f"windows={len(self.patch_index_windows)}, "
                f"samples_per_epoch={self.samples_per_epoch or 'all'}"
            )
        if self.enumerate_patch_index:
            print(
                f"[Dataset] fixed train patch-index mode: "
                f"windows={len(self.patch_index_windows)}, "
                f"samples_per_epoch={self.samples_per_epoch or 'all'}"
            )

        if self.samples_per_epoch > 0:
            mode = 'train' if self.is_training else 'eval'
            print(
                f"[Dataset] virtual {mode} length enabled: "
                f"samples_per_epoch={self.samples_per_epoch}, scenes={len(self.sample_pairs)}"
            )

    def __len__(self):
        if self.samples_per_epoch > 0:
            return self.samples_per_epoch
        if self.enumerate_patch_index and self.patch_index_windows:
            return len(self.patch_index_windows)
        if self.eval_patch_index and self.patch_index_windows:
            return len(self.patch_index_windows)
        return len(self.sample_pairs)

    def _cache_file_path(self, exr_path: str) -> str:
        stat = os.stat(exr_path)
        key_src = f"{exr_path}|{stat.st_mtime_ns}|{stat.st_size}"
        key = hashlib.sha1(key_src.encode('utf-8')).hexdigest()
        return os.path.join(self.exr_cache_dir, f"{key}.npy")

    def _read_exr_with_cache(self, exr_path: str, mmap: bool = False) -> np.ndarray:
        if not self.use_exr_cache:
            return read_exr(exr_path)

        cache_path = self._cache_file_path(exr_path)
        if os.path.exists(cache_path):
            mmap_mode = 'r' if mmap else None
            return np.load(cache_path, allow_pickle=False, mmap_mode=mmap_mode)

        image_np = read_exr(exr_path)
        tmp_path = f"{cache_path}.tmp.{os.getpid()}.npy"
        np.save(tmp_path, image_np)
        try:
            os.replace(tmp_path, cache_path)
        except OSError:
            # 多进程下可能并发写入同一缓存文件，保留先写入者即可。
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        if mmap and os.path.exists(cache_path):
            return np.load(cache_path, allow_pickle=False, mmap_mode='r')
        return image_np

    def _load_patch_index(self, patch_index_path: str) -> None:
        if not os.path.exists(patch_index_path):
            raise FileNotFoundError(f"patch_index_path 不存在: {patch_index_path}")

        data = np.load(patch_index_path, allow_pickle=False)
        required = {'scene_ids', 'tops', 'lefts'}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"patch index 缺少字段: {sorted(missing)}")

        scene_ids = data['scene_ids'].astype(str)
        tops = data['tops'].astype(np.int64)
        lefts = data['lefts'].astype(np.int64)
        if not (len(scene_ids) == len(tops) == len(lefts)):
            raise ValueError("patch index 字段长度不一致")

        if 'quality_scores' in data.files:
            quality_scores = data['quality_scores'].astype(np.float32)
        elif 'scores' in data.files:
            quality_scores = data['scores'].astype(np.float32)
        else:
            quality_scores = np.ones(len(tops), dtype=np.float32)
        if 'sampling_weight' in data.files:
            sampling_weights = data['sampling_weight'].astype(np.float32)
        else:
            sampling_weights = quality_scores.copy()
        if 'scale_05_eligible' in data.files:
            scale_half_eligible = data['scale_05_eligible'].astype(bool)
        else:
            scale_half_eligible = np.zeros(len(tops), dtype=bool)
        if 'category_masks' in data.files:
            category_masks = data['category_masks'].astype(np.uint8)
        else:
            category_masks = np.zeros(len(tops), dtype=np.uint8)
        if 'category_names' in data.files and 'category_bits' in data.files:
            self.patch_index_category_bits = {
                str(name): int(bit)
                for name, bit in zip(data['category_names'], data['category_bits'])
            }

        if 'meta_json' in data.files:
            try:
                self.patch_index_meta = json.loads(str(data['meta_json'].item()))
            except Exception:
                self.patch_index_meta = {}
        for scene_stats in self.patch_index_meta.get('scene_stats', []):
            scene_id = str(scene_stats.get('scene_id', ''))
            if scene_id:
                self.patch_index_scene_thresholds[scene_id] = dict(
                    scene_stats.get('thresholds', {})
                )

        # 默认沿用候选池生成时的阈值，保证“填入候选池路径即可使用”的复检口径一致。
        if self.patch_index_use_meta_thresholds:
            meta_valid_ratio = float(self.patch_index_meta.get('min_valid_ratio', 0.0) or 0.0)
            meta_depth_range = float(self.patch_index_meta.get('min_depth_range_ips', 0.0) or 0.0)
            meta_center_ratio = float(self.patch_index_meta.get('min_center_valid_ratio', 0.0) or 0.0)
            if meta_valid_ratio > 0.0:
                self.min_valid_ratio = meta_valid_ratio
            if self.patch_index_meta.get('index_type') == 'stratified_training':
                self.min_depth_range_ips = meta_depth_range
            elif meta_depth_range > 0.0:
                self.min_depth_range_ips = meta_depth_range
            if meta_center_ratio > 0.0:
                self.min_center_valid_ratio = meta_center_ratio

        valid_sample_ids = {sample['id'] for sample in self.sample_pairs}
        loaded = 0
        for scene_id in sorted(set(scene_ids.tolist())):
            if scene_id not in valid_sample_ids:
                continue
            idx = np.nonzero(scene_ids == scene_id)[0]
            if idx.size == 0:
                continue
            scene_sampling_weights = np.maximum(sampling_weights[idx], 1e-6)
            self.patch_index_by_id[scene_id] = {
                'tops': torch.from_numpy(tops[idx].copy()).long(),
                'lefts': torch.from_numpy(lefts[idx].copy()).long(),
                'scores': torch.from_numpy(scene_sampling_weights.copy()).float(),
                'quality_scores': torch.from_numpy(
                    quality_scores[idx].copy()
                ).float(),
                'scale_half_eligible': torch.from_numpy(
                    scale_half_eligible[idx].copy()
                ).bool(),
                'category_masks': torch.from_numpy(category_masks[idx].copy()).to(torch.uint8),
            }
            if self.eval_patch_index or self.enumerate_patch_index:
                for i in idx.tolist():
                    self.patch_index_windows.append(
                        (
                            scene_id,
                            int(tops[i]),
                            int(lefts[i]),
                            bool(scale_half_eligible[i]),
                        )
                    )
            loaded += int(idx.size)

        print(
            f"[Patch Index] loaded {loaded} windows from {patch_index_path}; "
            f"matched_scenes={len(self.patch_index_by_id)}/{len(self.sample_pairs)}, "
            f"jitter={self.patch_index_jitter}, strict={self.patch_index_strict}, "
            f"weighted={self.patch_index_weighted}, "
            f"sampling_weight_field={'sampling_weight' in data.files}, "
            f"scale05_eligible={int(scale_half_eligible.sum())}, "
            f"use_meta_thresholds={self.patch_index_use_meta_thresholds}, "
            f"min_valid_ratio={self.min_valid_ratio:.3f}, "
            f"min_depth_range_ips={self.min_depth_range_ips:.3f}, "
            f"center_valid_ratio={self.min_center_valid_ratio:.3f}"
        )

    def _build_stratified_category_schedule(self) -> None:
        if not self.patch_category_mix:
            return
        if not self.patch_index_by_id:
            raise ValueError('Stratified patch sampling requires a loaded patch index')
        missing = [
            category
            for category in self.patch_category_mix
            if category not in self.patch_index_category_bits
        ]
        if missing:
            raise ValueError(f'Patch index is missing categories: {missing}')

        scene_count = max(1, len(self.sample_pairs))
        scene_sample_counts = {}
        scene_category_counts = {}
        for scene_offset, sample in enumerate(self.sample_pairs):
            scene_id = sample['id']
            sample_count = 0
            if self.samples_per_epoch > scene_offset:
                sample_count = 1 + (self.samples_per_epoch - 1 - scene_offset) // scene_count
            scene_sample_counts[scene_id] = sample_count
            if sample_count <= 0:
                scene_category_counts[scene_id] = {
                    name: 0 for name in self.patch_category_mix
                }
                continue

            exact = {
                name: ratio * sample_count
                for name, ratio in self.patch_category_mix.items()
            }
            counts = {name: int(math.floor(value)) for name, value in exact.items()}
            remainder = sample_count - sum(counts.values())
            order = sorted(
                exact,
                key=lambda name: (exact[name] - counts[name], name),
                reverse=True,
            )
            for name in order[:remainder]:
                counts[name] += 1
            scene_category_counts[scene_id] = counts

        global_exact = {
            name: ratio * self.samples_per_epoch
            for name, ratio in self.patch_category_mix.items()
        }
        global_targets = {
            name: int(math.floor(value)) for name, value in global_exact.items()
        }
        global_remainder = self.samples_per_epoch - sum(global_targets.values())
        global_order = sorted(
            global_exact,
            key=lambda name: (global_exact[name] - global_targets[name], name),
            reverse=True,
        )
        for name in global_order[:global_remainder]:
            global_targets[name] += 1

        current_totals = {
            name: sum(counts[name] for counts in scene_category_counts.values())
            for name in self.patch_category_mix
        }
        while current_totals != global_targets:
            under = next(
                name
                for name in self.patch_category_mix
                if current_totals[name] < global_targets[name]
            )
            over = next(
                name
                for name in self.patch_category_mix
                if current_totals[name] > global_targets[name]
            )
            candidates = [
                sample['id']
                for sample in self.sample_pairs
                if scene_category_counts[sample['id']][over] > 0
            ]
            if not candidates:
                raise RuntimeError(
                    f'Cannot rebalance patch categories from {over} to {under}'
                )
            scene_id = min(
                candidates,
                key=lambda item: (
                    scene_category_counts[item][under]
                    / max(scene_sample_counts[item], 1),
                    item,
                ),
            )
            scene_category_counts[scene_id][over] -= 1
            scene_category_counts[scene_id][under] += 1
            current_totals[over] -= 1
            current_totals[under] += 1

        for scene_offset, sample in enumerate(self.sample_pairs):
            scene_id = sample['id']
            sample_count = scene_sample_counts[scene_id]
            counts = scene_category_counts[scene_id]
            schedule = [
                name
                for name in self.patch_category_mix
                for _ in range(counts[name])
            ]
            rng = np.random.default_rng(self.patch_category_seed + scene_offset)
            rng.shuffle(schedule)
            self.stratified_category_schedule[scene_id] = schedule
            index = self.patch_index_by_id.get(scene_id)
            if index is None:
                raise ValueError(f'Patch index has no entries for {scene_id}')
            category_masks = index['category_masks']
            availability = {
                name: int(
                    torch.count_nonzero(
                        category_masks & int(self.patch_index_category_bits[name])
                    ).item()
                )
                for name in counts
            }
            empty = [name for name, count in availability.items() if count == 0]
            if empty:
                raise ValueError(f'{scene_id} has empty stratified categories: {empty}')
            print(
                f"[Patch Mix] {scene_id}: samples={sample_count}, "
                + ', '.join(f"{name}={counts[name]}" for name in counts)
                + '; pools='
                + ', '.join(f"{name}:{availability[name]}" for name in availability)
            )

    def _stratified_category_for_index(self, sample_id: str, idx: int) -> Optional[str]:
        schedule = self.stratified_category_schedule.get(sample_id)
        if not schedule:
            return None
        scene_count = max(1, len(self.sample_pairs))
        occurrence = int(idx) // scene_count
        return schedule[occurrence % len(schedule)]

    def _patch_quality_stats(self, depth_patch: torch.Tensor, mask_patch: torch.Tensor) -> Tuple[float, float, float]:
        valid = mask_patch > 0.5
        valid_ratio = float(valid.float().mean().item())
        if valid.sum() == 0:
            return valid_ratio, 0.0, 0.0

        h, w = valid.shape[-2], valid.shape[-1]
        center_h = max(1, h // 2)
        center_w = max(1, w // 2)
        center_top = max(0, (h - center_h) // 2)
        center_left = max(0, (w - center_w) // 2)
        center_valid = valid[
            ..., center_top:center_top + center_h, center_left:center_left + center_w
        ]
        center_valid_ratio = float(center_valid.float().mean().item())

        valid_depth = depth_patch[valid]
        depth_range = float((valid_depth.max() - valid_depth.min()).item())
        return valid_ratio, depth_range, center_valid_ratio

    def _sample_random_crop_window(self, full_h: int, full_w: int) -> Tuple[int, int]:
        crop_h, crop_w = self.image_size
        if full_h <= crop_h:
            top = 0
        else:
            top = int(torch.randint(0, full_h - crop_h + 1, (1,)).item())

        if full_w <= crop_w:
            left = 0
        else:
            left = int(torch.randint(0, full_w - crop_w + 1, (1,)).item())
        return top, left

    def _crop_window(self, x: torch.Tensor, top: int, left: int) -> torch.Tensor:
        crop_h, crop_w = self.image_size
        return x[..., top:top + crop_h, left:left + crop_w]

    def _apply_random_flips(self, x: torch.Tensor, do_vflip: bool, do_hflip: bool) -> torch.Tensor:
        if do_vflip:
            x = torch.flip(x, dims=[-2])
        if do_hflip:
            x = torch.flip(x, dims=[-1])
        return x

    def _normalize_hs_patch(
        self, hs_patch: np.ndarray, hs_image: np.ndarray, cache_key: str
    ) -> np.ndarray:
        norm_mode = str(self.hs_norm_mode or 'scene_max')
        if norm_mode != 'scene_max':
            return normalize_hs_image(
                hs_patch,
                norm_mode=norm_mode,
                norm_scale=self.hs_norm_scale,
                sanity_threshold=self.hs_sanity_threshold,
            )

        scale = self._hs_scene_scale_cache.get(cache_key)
        if scale is None:
            hs_for_scale = hs_image[:, :, :self.hs_channels]
            max_hs = float(np.max(hs_for_scale))
            if max_hs > self.hs_sanity_threshold:
                valid_pixels = hs_for_scale < self.hs_sanity_threshold
                scale = float(np.max(hs_for_scale[valid_pixels])) if np.any(valid_pixels) else 1.0
            else:
                scale = max_hs
            if scale <= 1e-8:
                scale = 1.0
            self._hs_scene_scale_cache[cache_key] = scale

        patch = hs_patch.astype(np.float32, copy=False)
        if scale < self.hs_sanity_threshold:
            patch = np.clip(patch, 0.0, scale)
        return patch / scale

    def _patch_tensors_from_arrays(
        self,
        hs_image: np.ndarray,
        depth_map: np.ndarray,
        top: int,
        left: int,
        hs_cache_key: str,
        spatial_scale: float = 1.0,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
        Dict[str, torch.Tensor],
    ]:
        crop_h, crop_w = self.image_size
        if spatial_scale not in (0.5, 1.0):
            raise ValueError(f'Unsupported Baek spatial scale: {spatial_scale}')
        context_h = crop_h + 2 * self.optical_halo
        context_w = crop_w + 2 * self.optical_halo
        source_h = int(round(context_h / spatial_scale))
        source_w = int(round(context_w / spatial_scale))
        target_center_y = top + crop_h // 2
        target_center_x = left + crop_w // 2
        source_top = target_center_y - source_h // 2
        source_left = target_center_x - source_w // 2

        def extract_with_zero_padding(array, window_top, window_left, height, width):
            array_h, array_w = array.shape[:2]
            src_top = max(0, window_top)
            src_left = max(0, window_left)
            src_bottom = min(array_h, window_top + height)
            src_right = min(array_w, window_left + width)
            out_shape = (height, width) + tuple(array.shape[2:])
            output = np.zeros(out_shape, dtype=np.float32)
            if src_bottom > src_top and src_right > src_left:
                dst_top = src_top - window_top
                dst_left = src_left - window_left
                output[
                    dst_top:dst_top + (src_bottom - src_top),
                    dst_left:dst_left + (src_right - src_left),
                    ...,
                ] = np.asarray(
                    array[src_top:src_bottom, src_left:src_right, ...],
                    dtype=np.float32,
                )
            return output

        hs_patch = extract_with_zero_padding(
            hs_image[:, :, :self.hs_channels],
            source_top,
            source_left,
            source_h,
            source_w,
        )

        depth_source = depth_map[:, :, 0] if depth_map.ndim == 3 else depth_map
        depth_patch = extract_with_zero_padding(
            depth_source, source_top, source_left, source_h, source_w)
        depth_patch = np.asarray(depth_patch, dtype=np.float32).copy() / 1000.0
        valid_mask = (depth_patch > self.min_depth - 1e-3).astype(np.float32)

        if self.baek_augment:
            hs_patch = np.array(hs_patch, dtype=np.float32, copy=True)
            finite_valid = np.isfinite(hs_patch) & (
                hs_patch < self.hs_sanity_threshold
            )
            safe_max = float(np.max(hs_patch[finite_valid])) if np.any(finite_valid) else 1.0
            hs_patch = np.nan_to_num(
                hs_patch,
                nan=0.0,
                posinf=safe_max,
                neginf=0.0,
            )
            hs_patch = np.clip(hs_patch, 0.0, safe_max)
            hs_tensor = torch.from_numpy(hs_patch).permute(2, 0, 1).unsqueeze(0)
        else:
            hs_patch = self._normalize_hs_patch(hs_patch, hs_image, hs_cache_key)
            hs_tensor = torch.from_numpy(hs_patch).permute(2, 0, 1).unsqueeze(0)

        depth_raw_tensor = torch.from_numpy(depth_patch).unsqueeze(0).unsqueeze(0)
        mask_tensor = torch.from_numpy(valid_mask).unsqueeze(0).unsqueeze(0)
        if spatial_scale != 1.0:
            hs_tensor = F.interpolate(
                hs_tensor.float(),
                size=(context_h, context_w),
                mode='bilinear',
                align_corners=False,
            )
            depth_raw_tensor = F.interpolate(
                depth_raw_tensor.float(), size=(context_h, context_w), mode='nearest'
            )
            mask_tensor = F.interpolate(
                mask_tensor.float(), size=(context_h, context_w), mode='nearest'
            )

        illuminant_index = -1
        illuminant_requested = False
        illuminant_attempts = 0
        exposure = 1.0
        clip_ratio = 0.0
        requested_shift = 0.0
        applied_shift = 0.0
        if self.baek_augment:
            exposure = self.baek_exposure_min + float(torch.rand(1).item()) * (
                self.baek_exposure_max - self.baek_exposure_min
            )
            hs_raw_tensor = hs_tensor.float()
            illuminant_requested = bool(
                torch.rand(1).item() < self.baek_illuminant_probability
            )
            hs_normalized = None
            if illuminant_requested:
                for attempt in range(self.baek_illuminant_retries):
                    candidate_index = int(
                        torch.randint(0, len(CIE_ILLUMINANT_NAMES), (1,)).item()
                    )
                    candidate = apply_cie_illuminant(
                        hs_raw_tensor, candidate_index, exposure
                    ) / self.hs_norm_scale
                    candidate_clip_ratio = float(
                        ((candidate < 0.0) | (candidate > 1.0))
                        .float()
                        .mean()
                        .item()
                    )
                    illuminant_attempts = attempt + 1
                    if candidate_clip_ratio <= self.baek_max_clip_ratio:
                        illuminant_index = candidate_index
                        hs_normalized = candidate
                        break
            if hs_normalized is None:
                hs_normalized = apply_cie_illuminant(
                    hs_raw_tensor, -1, exposure
                ) / self.hs_norm_scale
            clip_ratio = float(
                ((hs_normalized < 0.0) | (hs_normalized > 1.0))
                .float()
                .mean()
                .item()
            )
            hs_tensor = hs_normalized.clamp(0.0, 1.0)

            if torch.rand(1).item() < self.baek_depth_shift_probability:
                direction = -1.0 if torch.rand(1).item() < 0.5 else 1.0
                requested_shift = direction * self.baek_depth_shift_m

        valid_mask_bool = mask_tensor > 0.5
        depth_metric_tensor = torch.where(
            valid_mask_bool,
            depth_raw_tensor.clamp(self.min_depth, self.max_depth),
            torch.full_like(depth_raw_tensor, self.min_depth),
        )
        if self.baek_augment:
            depth_metric_tensor, applied_shift = translate_metric_depth(
                depth_metric_tensor,
                mask_tensor,
                requested_shift,
                self.min_depth,
                self.max_depth,
            )
        ips_depth = metric_to_ips(
            depth_metric_tensor, self.min_depth, self.max_depth
        )

        depth_tensor = torch.clamp(ips_depth, 0.0, 1.0).float()
        center_y = self.optical_halo
        center_x = self.optical_halo
        center_slice = (
            slice(None), slice(None),
            slice(center_y, center_y + crop_h),
            slice(center_x, center_x + crop_w),
        )
        hs_target = hs_tensor[center_slice]
        depth_target = depth_tensor[center_slice]
        depth_metric_target = depth_metric_tensor[center_slice]
        mask_target = mask_tensor[center_slice]
        augmentation = {
            'aug_scale_factor': torch.tensor(float(spatial_scale)),
            'aug_depth_shift_requested_m': torch.tensor(float(requested_shift)),
            'aug_depth_shift_m': torch.tensor(float(applied_shift)),
            'aug_illuminant_index': torch.tensor(float(illuminant_index)),
            'aug_illuminant_requested': torch.tensor(
                float(illuminant_requested)
            ),
            'aug_illuminant_applied': torch.tensor(float(illuminant_index >= 0)),
            'aug_illuminant_attempts': torch.tensor(float(illuminant_attempts)),
            'aug_illuminant_fallback': torch.tensor(
                float(illuminant_requested and illuminant_index < 0)
            ),
            'aug_exposure': torch.tensor(float(exposure)),
            'aug_clip_ratio': torch.tensor(float(clip_ratio)),
        }
        return (
            hs_target.float(),
            depth_target,
            depth_metric_target.float(),
            mask_target.float(),
            hs_tensor.float(),
            depth_tensor,
            depth_metric_tensor.float(),
            mask_tensor.float(),
            augmentation,
        )

    def _depth_mask_tensors_from_arrays(
        self, depth_map: np.ndarray, top: int, left: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        crop_h, crop_w = self.image_size
        if depth_map.ndim == 3:
            depth_patch = depth_map[top:top + crop_h, left:left + crop_w, 0]
        else:
            depth_patch = depth_map[top:top + crop_h, left:left + crop_w]
        depth_patch = np.asarray(depth_patch, dtype=np.float32) / 1000.0
        valid_mask = (depth_patch > self.min_depth - 1e-3).astype(np.float32)
        depth_raw_tensor = torch.from_numpy(depth_patch).float()
        valid_mask_bool = depth_raw_tensor >= self.min_depth - 1e-3
        depth_safe = torch.where(
            valid_mask_bool,
            depth_raw_tensor,
            torch.tensor(self.min_depth),
        )
        ips_depth = metric_to_ips(depth_safe, self.min_depth, self.max_depth)
        depth_tensor = torch.clamp(ips_depth, 0.0, 1.0).unsqueeze(0).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(valid_mask).unsqueeze(0).unsqueeze(0).float()
        return depth_tensor, mask_tensor

    def _patch_passes_stratified_category(
        self,
        sample_id: str,
        category: str,
        hs_patch: np.ndarray,
        depth_patch: torch.Tensor,
        mask_patch: torch.Tensor,
    ) -> bool:
        if not self._window_passes_quality(depth_patch, mask_patch, 0, 0):
            return False
        if category == 'general':
            return True

        valid = mask_patch[0, 0] > 0.5
        if not torch.any(valid):
            return False
        if category == 'depth_hard':
            threshold = float(
                self.patch_index_meta.get(
                    'depth_hard_min_ips', self.min_depth_range_ips
                )
            )
            depth_try = depth_patch[0, 0]
            valid_depth = depth_try[valid]
            return float((valid_depth.max() - valid_depth.min()).item()) >= threshold

        valid_np = valid.numpy()
        hs_valid = hs_patch[valid_np]
        if hs_valid.size == 0:
            return False
        thresholds = self.patch_index_scene_thresholds.get(sample_id, {})
        if category == 'hs_bright':
            bright_value = float(self.patch_index_meta.get('bright_value_threshold', 0.8))
            hs_rms = float(np.sqrt(np.mean(np.square(hs_valid))))
            bright_ratio = float(np.mean(np.max(hs_patch, axis=2) >= bright_value))
            return (
                hs_rms >= float(thresholds.get('hs_rms_bright', 0.0))
                or bright_ratio >= float(
                    self.patch_index_meta.get('absolute_bright_ratio', 1.0)
                )
            )
        if category == 'hs_complex':
            spectral_var = float(np.mean(np.var(hs_valid, axis=1)))
            intensity = np.mean(hs_patch, axis=2)
            gradient = np.zeros_like(intensity)
            gradient[:, 1:] += np.abs(intensity[:, 1:] - intensity[:, :-1])
            gradient[1:, :] += np.abs(intensity[1:, :] - intensity[:-1, :])
            spatial_gradient = float(np.mean(gradient[valid_np]))
            return (
                spectral_var >= float(thresholds.get('hs_spectral_var_complex', 0.0))
                or spatial_gradient >= float(
                    thresholds.get('hs_spatial_gradient_complex', 0.0)
                )
            )
        return True

    def _window_passes_quality(
        self, depth_base: torch.Tensor, mask_base: torch.Tensor, top: int, left: int
    ) -> bool:
        depth_try = self._crop_window(depth_base, top, left)
        mask_try = self._crop_window(mask_base, top, left)
        valid_ratio, depth_range, center_valid_ratio = self._patch_quality_stats(depth_try, mask_try)
        return (
            valid_ratio >= self.min_valid_ratio
            and depth_range >= self.min_depth_range_ips
            and center_valid_ratio >= self.min_center_valid_ratio
        )

    def _sample_patch_index_window(
        self,
        sample_id: str,
        full_h: int,
        full_w: int,
        category: Optional[str] = None,
        require_scale_half: bool = False,
    ) -> Optional[Tuple[int, int, int, int, Optional[str], bool]]:
        index = self.patch_index_by_id.get(sample_id)
        if not index:
            return None

        tops = index['tops']
        lefts = index['lefts']
        if tops.numel() == 0:
            return None

        candidate_indices = torch.arange(tops.numel())
        if category is not None:
            bit = int(self.patch_index_category_bits[category])
            candidate_indices = torch.nonzero(
                (index['category_masks'] & bit) != 0, as_tuple=False
            ).flatten()
            if candidate_indices.numel() == 0:
                raise RuntimeError(f'No {category} patch candidates for {sample_id}')

        use_scale_half = bool(require_scale_half)
        if use_scale_half:
            eligible = index['scale_half_eligible'][candidate_indices]
            candidate_indices = candidate_indices[eligible]
            if candidate_indices.numel() == 0:
                use_scale_half = False
                candidate_indices = torch.arange(tops.numel())
                if category is not None:
                    bit = int(self.patch_index_category_bits[category])
                    candidate_indices = torch.nonzero(
                        (index['category_masks'] & bit) != 0, as_tuple=False
                    ).flatten()

        if self.patch_index_weighted:
            probs = index['scores'][candidate_indices]
            local_pick = int(torch.multinomial(probs, 1).item())
        else:
            local_pick = int(torch.randint(0, candidate_indices.numel(), (1,)).item())
        pick = int(candidate_indices[local_pick].item())

        base_top = int(tops[pick].item())
        base_left = int(lefts[pick].item())
        top, left = base_top, base_left

        jitter = 0 if use_scale_half else self.patch_index_jitter
        if category in ('hs_bright', 'hs_complex'):
            jitter = min(jitter, self.patch_index_hs_jitter)
        if jitter > 0:
            top += int(torch.randint(-jitter, jitter + 1, (1,)).item())
            left += int(torch.randint(-jitter, jitter + 1, (1,)).item())

        crop_h, crop_w = self.image_size
        top = max(0, min(top, max(0, full_h - crop_h)))
        left = max(0, min(left, max(0, full_w - crop_w)))
        return top, left, base_top, base_left, category, use_scale_half

    def _window_passes_stratified_category(
        self,
        sample_id: str,
        category: str,
        hs_image: np.ndarray,
        depth_base: torch.Tensor,
        mask_base: torch.Tensor,
        top: int,
        left: int,
    ) -> bool:
        if not self._window_passes_quality(depth_base, mask_base, top, left):
            return False
        if category == 'general':
            return True

        depth_try = self._crop_window(depth_base, top, left)
        mask_try = self._crop_window(mask_base, top, left)
        valid = mask_try[0, 0] > 0.5
        if not torch.any(valid):
            return False
        if category == 'depth_hard':
            threshold = float(
                self.patch_index_meta.get(
                    'depth_hard_min_ips', self.min_depth_range_ips
                )
            )
            values = depth_try[0, 0][valid]
            return float(values.max() - values.min()) >= threshold

        crop_h, crop_w = self.image_size
        hs_patch = hs_image[top:top + crop_h, left:left + crop_w, :]
        hs_patch = normalize_hs_image(
            hs_patch,
            norm_mode=self.hs_norm_mode,
            norm_scale=self.hs_norm_scale,
            sanity_threshold=self.hs_sanity_threshold,
        )
        valid_np = valid.cpu().numpy()
        hs_valid = hs_patch[valid_np]
        thresholds = self.patch_index_scene_thresholds.get(sample_id, {})
        if category == 'hs_bright':
            hs_rms = float(np.sqrt(np.mean(hs_valid * hs_valid)))
            bright_value = float(self.patch_index_meta.get('bright_value_threshold', 0.8))
            bright_ratio = float(np.mean(hs_valid >= bright_value))
            return (
                hs_rms >= float(thresholds.get('hs_rms_bright', 0.0))
                or bright_ratio >= float(
                    self.patch_index_meta.get('absolute_bright_ratio', 1.0)
                )
            )
        if category == 'hs_complex':
            spectral_var = float(np.mean(np.var(hs_valid, axis=1)))
            intensity = np.mean(hs_patch, axis=2)
            gradient = np.zeros_like(intensity)
            gradient[:, 1:] += np.abs(intensity[:, 1:] - intensity[:, :-1])
            gradient[1:, :] += np.abs(intensity[1:, :] - intensity[:-1, :])
            spatial_gradient = float(np.mean(gradient[valid_np]))
            return (
                spectral_var
                >= float(thresholds.get('hs_spectral_var_complex', 0.0))
                or spatial_gradient
                >= float(thresholds.get('hs_spatial_gradient_complex', 0.0))
            )
        return True

    def _getitem_patch_first(
        self,
        idx: int,
        sample: Dict[str, str],
        eval_window: Optional[Tuple[int, int, bool]],
    ) -> Optional[Dict[str, torch.Tensor]]:
        if (
            self.is_training
            and not self.patch_filter
            and not self.baek_augment
            and not self.patch_index_by_id
        ):
            return None
        if str(self.hs_norm_mode or 'scene_max') not in ('fixed_scale', 'scene_max'):
            return None

        hs_path = sample['hs_path']
        depth_path = sample['depth_path']
        sample_id = sample['id']

        try:
            hs_image = self._read_exr_with_cache(hs_path, mmap=True)
            depth_map = self._read_exr_with_cache(depth_path, mmap=True)
        except Exception as e:
            raise IOError(f"无法读取文件: {sample_id} \n错误: {e}")

        if hs_image.ndim != 3:
            raise ValueError(f"高光谱图像维度异常: {sample_id}, shape={hs_image.shape}")
        if hs_image.shape[2] < self.hs_channels:
            raise ValueError(
                f"高光谱通道数不足: {sample_id}, got={hs_image.shape[2]}, required={self.hs_channels}"
            )

        full_h, full_w = depth_map.shape[:2]
        use_scale_half = False
        if self.is_training and eval_window is not None:
            top, left, scale_half_eligible = eval_window
            use_scale_half = bool(
                self.baek_augment
                and scale_half_eligible
                and torch.rand(1).item() < self.baek_scale_half_probability
            )
        elif self.is_training:
            category = self._stratified_category_for_index(sample_id, idx)
            request_scale_half = bool(
                self.baek_augment
                and torch.rand(1).item() < self.baek_scale_half_probability
            )
            indexed_window = self._sample_patch_index_window(
                sample_id,
                full_h,
                full_w,
                category=category,
                require_scale_half=request_scale_half,
            )
            if indexed_window is not None:
                (
                    top,
                    left,
                    base_top,
                    base_left,
                    category,
                    use_scale_half,
                ) = indexed_window
                use_retry_sampler = False
                if category is not None and self.patch_index_strict:
                    depth_try, mask_try = self._depth_mask_tensors_from_arrays(
                        depth_map, top, left
                    )
                    crop_h, crop_w = self.image_size
                    hs_try = np.asarray(
                        hs_image[top:top + crop_h, left:left + crop_w, :self.hs_channels],
                        dtype=np.float32,
                    )
                    if not self._patch_passes_stratified_category(
                        sample_id, category, hs_try, depth_try, mask_try
                    ):
                        top, left = base_top, base_left
                elif self.patch_index_strict:
                    depth_try, mask_try = self._depth_mask_tensors_from_arrays(
                        depth_map, top, left
                    )
                    if not self._window_passes_quality(depth_try, mask_try, 0, 0):
                        depth_base_try, mask_base_try = self._depth_mask_tensors_from_arrays(
                            depth_map, base_top, base_left
                        )
                        if self._window_passes_quality(depth_base_try, mask_base_try, 0, 0):
                            top, left = base_top, base_left
                        else:
                            top, left = self._sample_random_crop_window(full_h, full_w)
                            use_scale_half = False
                            use_retry_sampler = True
            else:
                top, left = self._sample_random_crop_window(full_h, full_w)
                use_scale_half = False
                use_retry_sampler = True

            if use_retry_sampler:
                for _ in range(self.max_crop_retries):
                    depth_try, mask_try = self._depth_mask_tensors_from_arrays(
                        depth_map, top, left
                    )
                    valid_ratio, depth_range, center_valid_ratio = self._patch_quality_stats(
                        depth_try, mask_try
                    )
                    if (
                        valid_ratio >= self.min_valid_ratio
                        and depth_range >= self.min_depth_range_ips
                        and center_valid_ratio >= self.min_center_valid_ratio
                    ):
                        break
                    top, left = self._sample_random_crop_window(full_h, full_w)
                use_scale_half = False
        else:
            if eval_window is not None:
                top, left = eval_window[:2]
            else:
                crop_h, crop_w = self.image_size
                top = max(0, (full_h - crop_h) // 2)
                left = max(0, (full_w - crop_w) // 2)

        (
            hs_tensor,
            depth_tensor,
            depth_metric_tensor,
            mask_tensor,
            hs_optical_tensor,
            depth_optical_tensor,
            depth_metric_optical_tensor,
            mask_optical_tensor,
            augmentation,
        ) = self._patch_tensors_from_arrays(
            hs_image,
            depth_map,
            top,
            left,
            self._cache_file_path(hs_path),
            spatial_scale=0.5 if use_scale_half else 1.0,
        )

        if self.is_training and (self.augment or self.baek_augment):
            do_vflip = bool(torch.rand(1).item() < 0.5)
            do_hflip = bool(torch.rand(1).item() < 0.5)
            hs_tensor = self._apply_random_flips(hs_tensor, do_vflip, do_hflip)
            depth_tensor = self._apply_random_flips(depth_tensor, do_vflip, do_hflip)
            depth_metric_tensor = self._apply_random_flips(depth_metric_tensor, do_vflip, do_hflip)
            mask_tensor = self._apply_random_flips(mask_tensor, do_vflip, do_hflip)
            hs_optical_tensor = self._apply_random_flips(
                hs_optical_tensor, do_vflip, do_hflip)
            depth_optical_tensor = self._apply_random_flips(
                depth_optical_tensor, do_vflip, do_hflip)
            depth_metric_optical_tensor = self._apply_random_flips(
                depth_metric_optical_tensor, do_vflip, do_hflip)
            mask_optical_tensor = self._apply_random_flips(
                mask_optical_tensor, do_vflip, do_hflip)

        item = {
            'id': sample_id,
            'hs_image': hs_tensor.squeeze(0),
            'depth_map': depth_tensor.squeeze(0).squeeze(0),
            'depth_metric': depth_metric_tensor.squeeze(0).squeeze(0),
            'mask': mask_tensor.squeeze(0).squeeze(0),
        }
        if self.baek_augment:
            item.update(augmentation)
        if self.optical_halo > 0:
            item.update({
                'hs_optical': hs_optical_tensor.squeeze(0),
                'depth_map_optical': depth_optical_tensor.squeeze(0).squeeze(0),
                'depth_metric_optical': (
                    depth_metric_optical_tensor.squeeze(0).squeeze(0)),
                'mask_optical': mask_optical_tensor.squeeze(0).squeeze(0),
            })
        return item

    def __getitem__(self, idx):
        eval_window = None
        if (self.eval_patch_index or self.enumerate_patch_index) and self.patch_index_windows:
            scene_id, top, left, scale_half_eligible = self.patch_index_windows[
                idx % len(self.patch_index_windows)
            ]
            sample = self.sample_by_id[scene_id]
            eval_window = (top, left, scale_half_eligible)
        else:
            if self.samples_per_epoch > 0:
                idx = idx % len(self.sample_pairs)
            sample = self.sample_pairs[idx]

        hs_path = sample['hs_path']
        depth_path = sample['depth_path']
        sample_id = sample['id']

        fast_item = self._getitem_patch_first(idx, sample, eval_window)
        if fast_item is not None:
            return fast_item
        if self.optical_halo > 0:
            raise RuntimeError(
                'optical_halo requires the patch-first dataset path; use fixed_scale or '
                'scene_max normalization with a patch index/cache-compatible configuration')
        
        try:
            hs_image = self._read_exr_with_cache(hs_path)
            depth_map = self._read_exr_with_cache(depth_path)
        except Exception as e:
            raise IOError(f"无法读取文件: {sample_id} \n错误: {e}")

        hs_image = hs_image.astype(np.float32)
        depth_map = depth_map.astype(np.float32)

        if hs_image.ndim != 3:
            raise ValueError(f"高光谱图像维度异常: {sample_id}, shape={hs_image.shape}")
        if hs_image.shape[2] < self.hs_channels:
            raise ValueError(
                f"高光谱通道数不足: {sample_id}, got={hs_image.shape[2]}, required={self.hs_channels}"
            )
        # 数据原始波段按 420nm -> 700nm 递增排列，这里只保留 420nm -> 660nm 的前 25 个波段。
        hs_image = hs_image[:, :, :self.hs_channels]

        # 确保深度图是 (H, W)
        if depth_map.ndim == 3:
            depth_map = depth_map.squeeze(-1)
        
        # --- 深度单位转换 (mm -> m) ---
        depth_map = depth_map / 1000.0

        # ============================================================
        # 步骤 A: 生成 Mask (在归一化造成负数之前)
        # ============================================================
        # 【IPS 迁移】背景是0，物体是0.4~2.0。
        # 任何低于 min_depth 的都是背景，标记为 0；有效物体标记为 1。
        # 这个 Mask 稍后会在 Loss 计算时用到，确保网络不学背景的虚假值。
        valid_mask = (depth_map > self.min_depth - 1e-3).astype(np.float32)

        # ============================================================
        # 步骤 B: 高光谱图像处理
        # ============================================================
        # fixed_scale 不依赖整张图的 max。对于明确裁剪窗口的路径，先保留 raw HS，
        # 后面只归一化实际送入模型的 patch，避免每个样本扫描整张高光谱图。
        defer_hs_norm = self.hs_norm_mode == 'fixed_scale' and (self.patch_filter or not self.is_training)
        if not defer_hs_norm:
            hs_image = normalize_hs_image(
                hs_image,
                norm_mode=self.hs_norm_mode,
                norm_scale=self.hs_norm_scale,
                sanity_threshold=self.hs_sanity_threshold,
            )
        
        # ============================================================
        # 步骤 C: 深度图归一化 (【IPS 体系】使用逆深度均匀化)
        # ============================================================
        # 【迁移到 IPS】线性深度分布在远场会导致 PSF 差异很小。
        # 改用逆深度 (Inverse Perspective Sampling) 使深度分辨率均匀。
        #
        # 数学：
        #   物理深度 d ∈ [0.4m, 2.0m]
        #   逆深度 d_inv = 1/d ∈ [0.5, 2.5]
        #   归一化: d_norm = (1/d - 1/d_max) / (1/d_min - 1/d_max)
        #        = (max_depth * d - max_depth * min_depth) / ((max_depth - min_depth) * d)
        #
        # 边界条件：
        #   d=min_depth(0.4m): d_norm → 0.0（最近处）
        #   d=max_depth(2.0m): d_norm → 1.0（最远处）
        #   背景 d=0: 设为 1.0（或任意值，Mask 会排除）
        #
        # 使用 util.helper.metric_to_ips() 函数进行转换
        # ============================================================
        
        # 将 depth_map 转为 torch.Tensor 以使用 metric_to_ips
        depth_tensor = torch.from_numpy(depth_map).float()

        # 创建背景 mask（depth < min_depth 的像素）
        valid_mask_bool = depth_tensor >= self.min_depth - 1e-3

        # depth_metric: 物理米制深度，背景 clamp 到 min_depth（配合 mask 抑制无效区域）
        depth_metric_tensor = torch.clamp(depth_tensor, self.min_depth, self.max_depth)

        # 对有效前景像素使用 metric_to_ips 函数进行 IPS 归一化
        depth_safe = torch.where(valid_mask_bool, depth_tensor, torch.tensor(self.min_depth))
        ips_depth_tensor = metric_to_ips(depth_safe, self.min_depth, self.max_depth)

        # 确保在 [0, 1] 范围内，并转回 numpy
        depth_map = torch.clamp(ips_depth_tensor, 0.0, 1.0).numpy()
        depth_metric_np = depth_metric_tensor.numpy()

        # ============================================================
        # 步骤 D: 转 Tensor 并处理 Transform
        # ============================================================
        # HS: [H, W, C] -> [C, H, W]
        hs_tensor = None if defer_hs_norm else torch.from_numpy(hs_image).permute(2, 0, 1).float()
        
        # Depth & Mask: [H, W] -> [1, H, W]
        depth_tensor = torch.from_numpy(depth_map).unsqueeze(0).float()
        depth_metric_tensor = torch.from_numpy(depth_metric_np).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(valid_mask).unsqueeze(0).float()

        # 增加 batch 维度以适配 transform: [1, C, H, W]
        if hs_tensor is not None:
            hs_tensor = hs_tensor.unsqueeze(0)
        depth_tensor = depth_tensor.unsqueeze(0)
        depth_metric_tensor = depth_metric_tensor.unsqueeze(0)
        mask_tensor = mask_tensor.unsqueeze(0)

        def hs_window_tensor(top: int, left: int) -> torch.Tensor:
            crop_h, crop_w = self.image_size
            hs_patch = hs_image[top:top + crop_h, left:left + crop_w, :]
            hs_patch = normalize_hs_image(
                hs_patch,
                norm_mode=self.hs_norm_mode,
                norm_scale=self.hs_norm_scale,
                sanity_threshold=self.hs_sanity_threshold,
            )
            return torch.from_numpy(hs_patch).permute(2, 0, 1).unsqueeze(0).float()

        if self.is_training:
            hs_base = hs_tensor
            depth_base = depth_tensor
            depth_metric_base = depth_metric_tensor
            mask_base = mask_tensor

            # 拼接 depth_ips / depth_metric / mask，确保几何变换一致
            depth_mask_cat_base = torch.cat([depth_base, depth_metric_base, mask_base], dim=1)

            if self.patch_filter:
                _, _, full_h, full_w = depth_base.shape
                crop_h, crop_w = self.image_size
                stride = self.patch_filter_stride

                if stride > 1:
                    depth_proxy = depth_base[:, :, ::stride, ::stride]
                    mask_proxy = mask_base[:, :, ::stride, ::stride]
                    crop_h_proxy = max(1, math.ceil(crop_h / stride))
                    crop_w_proxy = max(1, math.ceil(crop_w / stride))
                else:
                    depth_proxy = None
                    mask_proxy = None
                    crop_h_proxy = 0
                    crop_w_proxy = 0

                category = self._stratified_category_for_index(sample_id, idx)
                indexed_window = self._sample_patch_index_window(
                    sample_id, full_h, full_w, category=category
                )
                if indexed_window is not None:
                    top, left, base_top, base_left, category = indexed_window
                    use_retry_sampler = False
                    if category is not None and self.patch_index_strict:
                        shifted_ok = self._window_passes_stratified_category(
                            sample_id, category, hs_image, depth_base, mask_base, top, left
                        )
                        if not shifted_ok:
                            top, left = base_top, base_left
                    elif self.patch_index_strict and not self._window_passes_quality(
                        depth_base, mask_base, top, left
                    ):
                        if self._window_passes_quality(depth_base, mask_base, base_top, base_left):
                            top, left = base_top, base_left
                        else:
                            top, left = self._sample_random_crop_window(full_h, full_w)
                            use_retry_sampler = True
                    # 候选池窗口已离线通过质量筛选；strict=False 时直接使用，strict=True 时上面已复检。
                else:
                    top, left = self._sample_random_crop_window(full_h, full_w)
                    use_retry_sampler = True

                if use_retry_sampler:
                    for _ in range(self.max_crop_retries):
                        if depth_proxy is not None and mask_proxy is not None:
                            top_proxy = top // stride
                            left_proxy = left // stride
                            depth_try_proxy = depth_proxy[
                                ..., top_proxy:top_proxy + crop_h_proxy, left_proxy:left_proxy + crop_w_proxy
                            ]
                            mask_try_proxy = mask_proxy[
                                ..., top_proxy:top_proxy + crop_h_proxy, left_proxy:left_proxy + crop_w_proxy
                            ]
                            (
                                valid_ratio_proxy,
                                depth_range_proxy,
                                center_valid_ratio_proxy,
                            ) = self._patch_quality_stats(depth_try_proxy, mask_try_proxy)
                            if (
                                valid_ratio_proxy < self.min_valid_ratio
                                or depth_range_proxy < self.min_depth_range_ips
                                or center_valid_ratio_proxy < self.min_center_valid_ratio
                            ):
                                top, left = self._sample_random_crop_window(full_h, full_w)
                                continue

                        depth_try = self._crop_window(depth_base, top, left)
                        mask_try = self._crop_window(mask_base, top, left)
                        valid_ratio, depth_range, center_valid_ratio = self._patch_quality_stats(
                            depth_try, mask_try
                        )

                        if (
                            valid_ratio >= self.min_valid_ratio
                            and depth_range >= self.min_depth_range_ips
                            and center_valid_ratio >= self.min_center_valid_ratio
                        ):
                            break
                        top, left = self._sample_random_crop_window(full_h, full_w)

                hs_tensor = hs_window_tensor(top, left) if defer_hs_norm else self._crop_window(hs_base, top, left)
                depth_tensor = self._crop_window(depth_base, top, left)
                depth_metric_tensor = self._crop_window(depth_metric_base, top, left)
                mask_tensor = (self._crop_window(mask_base, top, left) > 0.5).float()

                if self.augment:
                    do_vflip = bool(torch.rand(1).item() < 0.5)
                    do_hflip = bool(torch.rand(1).item() < 0.5)
                    hs_tensor = self._apply_random_flips(hs_tensor, do_vflip, do_hflip)
                    depth_tensor = self._apply_random_flips(depth_tensor, do_vflip, do_hflip)
                    depth_metric_tensor = self._apply_random_flips(depth_metric_tensor, do_vflip, do_hflip)
                    mask_tensor = self._apply_random_flips(mask_tensor, do_vflip, do_hflip)
            else:
                hs_tensor, depth_mask_cat = self.transform(hs_base, depth_mask_cat_base)
                depth_tensor = depth_mask_cat[:, 0:1, :, :]
                depth_metric_tensor = depth_mask_cat[:, 1:2, :, :]
                mask_tensor = (depth_mask_cat[:, 2:3, :, :] > 0.5).float()

        else:
            if eval_window is not None:
                top, left = eval_window[:2]
                hs_tensor = hs_window_tensor(top, left) if defer_hs_norm else self._crop_window(hs_tensor, top, left)
                depth_tensor = self._crop_window(depth_tensor, top, left)
                depth_metric_tensor = self._crop_window(depth_metric_tensor, top, left)
                mask_tensor = self._crop_window(mask_tensor, top, left)
            else:
                if defer_hs_norm:
                    _, _, full_h, full_w = depth_tensor.shape
                    crop_h, crop_w = self.image_size
                    top = max(0, (full_h - crop_h) // 2)
                    left = max(0, (full_w - crop_w) // 2)
                    hs_tensor = hs_window_tensor(top, left)
                else:
                    hs_tensor = self.centercrop(hs_tensor)
                depth_tensor = self.centercrop(depth_tensor)
                depth_metric_tensor = self.centercrop(depth_metric_tensor)
                mask_tensor = self.centercrop(mask_tensor)

        # 移除 batch 维度
        hs_tensor = hs_tensor.squeeze(0)               # [C, H, W]
        depth_tensor = depth_tensor.squeeze(0)         # [1, H, W] -> [H, W]
        depth_metric_tensor = depth_metric_tensor.squeeze(0)  # [1, H, W] -> [H, W]
        mask_tensor = mask_tensor.squeeze(0)           # [1, H, W] -> [H, W]

        depth_tensor = depth_tensor.squeeze(0)
        depth_metric_tensor = depth_metric_tensor.squeeze(0)
        mask_tensor = mask_tensor.squeeze(0)

        return {
            'id': sample_id,
            'hs_image': hs_tensor,
            'depth_map': depth_tensor,
            'depth_metric': depth_metric_tensor,
            'mask': mask_tensor,
        }
