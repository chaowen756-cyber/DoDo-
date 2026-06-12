import argparse
import glob
import json
import os
import sys
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from infer_contect import load_model, read_exr, select_hs_bands
from datasets.hyperspectral_dataset import normalize_hs_image


def parse_args():
    parser = argparse.ArgumentParser(
        description='Estimate fixed HS and DoDo measurement scales from training patch distribution.'
    )
    parser.add_argument('--data_root', type=str, default='/root/autodl-tmp/Baek数据集')
    parser.add_argument(
        '--patch_index_path',
        type=str,
        default='/root/autodl-tmp/Baek数据集/.patch_index/patch128_stride32_valid20_range060_center10_v1.npz',
        help='Training patch-index .npz. Empty string falls back to grid patches.',
    )
    parser.add_argument('--scene_start', type=int, default=1)
    parser.add_argument('--scene_end', type=int, default=15)
    parser.add_argument('--hs_channels', type=int, default=25)
    parser.add_argument('--hs_percentile', type=float, default=99.9)
    parser.add_argument('--sensor_percentile', type=float, default=99.9)
    parser.add_argument('--min_depth', type=float, default=0.4)
    parser.add_argument('--max_depth', type=float, default=2.0)
    parser.add_argument('--patch_size', type=int, default=128)
    parser.add_argument('--patch_stride', type=int, default=256,
                        help='Fallback grid stride when no patch_index_path is provided.')
    parser.add_argument('--max_patches', type=int, default=0,
                        help='Maximum patch-index windows to use; 0 = all matching windows.')
    parser.add_argument('--max_hs_values', type=int, default=5000000,
                        help='Deterministic reservoir size for HS percentile estimation.')
    parser.add_argument('--max_sensor_values', type=int, default=5000000,
                        help='Deterministic reservoir size for sensor percentile estimation.')
    parser.add_argument('--sensor_batch_size', type=int, default=32)
    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='Checkpoint whose optical parameters are used for raw DoDo measurement scale.')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=123)
    return parser.parse_args()


def scene_id_to_no(scene_id: str) -> int:
    digits = ''.join(ch for ch in str(scene_id) if ch.isdigit())
    if not digits:
        raise ValueError(f'Cannot parse scene number from scene_id={scene_id!r}')
    return int(digits)


def scene_paths(data_root: str, scene_no: int) -> Tuple[str, str]:
    folder = os.path.join(data_root, f'deploy {scene_no}')
    hs_files = glob.glob(os.path.join(folder, '*_hs.exr'))
    if not hs_files:
        raise FileNotFoundError(f'No *_hs.exr in {folder}')
    hs_path = hs_files[0]
    depth_path = hs_path.replace('_hs.exr', '_depth_map.exr')
    if not os.path.exists(depth_path):
        raise FileNotFoundError(depth_path)
    return hs_path, depth_path


def load_scene(data_root: str, scene_no: int, hs_channels: int):
    hs_path, depth_path = scene_paths(data_root, scene_no)
    hs = select_hs_bands(read_exr(hs_path), hs_channels)
    depth = read_exr(depth_path)
    if depth.ndim == 3:
        depth = depth.squeeze(-1)
    return hs, depth / 1000.0


def load_patch_windows(args) -> List[Tuple[int, int, int]]:
    if args.patch_index_path:
        data = np.load(args.patch_index_path, allow_pickle=False)
        scene_ids = data['scene_ids'].astype(str)
        tops = data['tops'].astype(np.int64)
        lefts = data['lefts'].astype(np.int64)
        windows = []
        for scene_id, top, left in zip(scene_ids, tops, lefts):
            scene_no = scene_id_to_no(scene_id)
            if args.scene_start <= scene_no <= args.scene_end:
                windows.append((scene_no, int(top), int(left)))
        if args.max_patches > 0:
            windows = windows[:args.max_patches]
        if not windows:
            raise RuntimeError(f'No patch-index windows matched scenes {args.scene_start}-{args.scene_end}.')
        return windows

    windows = []
    for scene_no in range(args.scene_start, args.scene_end + 1):
        _, depth = load_scene(args.data_root, scene_no, args.hs_channels)
        h, w = depth.shape
        for top in range(0, max(1, h - args.patch_size + 1), args.patch_stride):
            for left in range(0, max(1, w - args.patch_size + 1), args.patch_stride):
                windows.append((scene_no, int(top), int(left)))
                if args.max_patches > 0 and len(windows) >= args.max_patches:
                    return windows
    return windows


class ValueSampler:
    """Chunk-wise deterministic sampler for large percentile estimates."""

    def __init__(self, max_values: int, seed: int):
        self.max_values = int(max_values)
        self.rng = np.random.default_rng(seed)
        self.values = []
        self.total_seen = 0

    def add(self, values: np.ndarray):
        values = np.asarray(values).reshape(-1)
        values = values[np.isfinite(values)]
        values = values[(values >= 0.0) & (values < 10000.0)]
        if values.size == 0:
            return
        self.total_seen += int(values.size)
        if self.max_values <= 0 or values.size <= self.max_values:
            self.values.append(values.astype(np.float32, copy=False))
        else:
            idx = self.rng.choice(values.size, size=self.max_values, replace=False)
            self.values.append(values[idx].astype(np.float32, copy=False))

    def finish(self):
        if not self.values:
            raise RuntimeError('No values collected for scale estimation.')
        values = np.concatenate(self.values)
        if self.max_values > 0 and values.size > self.max_values:
            idx = self.rng.choice(values.size, size=self.max_values, replace=False)
            values = values[idx]
        return values


def scene_cache_get(cache: Dict[int, Tuple[np.ndarray, np.ndarray]], args, scene_no: int):
    item = cache.get(scene_no)
    if item is None:
        item = load_scene(args.data_root, scene_no, args.hs_channels)
        cache[scene_no] = item
    return item


def estimate_hs_scale(args, windows):
    cache = {}
    sampler = ValueSampler(args.max_hs_values, args.seed)
    valid_patch_count = 0
    for scene_no, top, left in windows:
        hs, depth = scene_cache_get(cache, args, scene_no)
        hs_patch = hs[top:top + args.patch_size, left:left + args.patch_size, :]
        depth_patch = depth[top:top + args.patch_size, left:left + args.patch_size]
        if hs_patch.shape[:2] != (args.patch_size, args.patch_size):
            continue
        mask = depth_patch > (args.min_depth - 1e-3)
        if not np.any(mask):
            continue
        sampler.add(hs_patch[mask])
        valid_patch_count += 1
    values = sampler.finish()
    return {
        'scale': float(np.percentile(values, args.hs_percentile)),
        'sampled_values': int(values.size),
        'total_values_seen': int(sampler.total_seen),
        'valid_patches': int(valid_patch_count),
    }


@torch.no_grad()
def estimate_sensor_scale(args, windows, hs_scale):
    device = torch.device('cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    model = load_model(args.ckpt_path, device)
    model.eval()
    old_norm = model.camera.measurement_norm_mode
    model.camera.measurement_norm_mode = 'none'

    cache = {}
    sampler = ValueSampler(args.max_sensor_values, args.seed + 1)
    valid_patch_count = 0
    batch = []

    def flush_batch():
        nonlocal valid_patch_count, batch
        if not batch:
            return
        hs_batch = torch.stack([
            torch.from_numpy(h).permute(2, 0, 1).float() for h, _, _ in batch
        ], dim=0).to(device)
        depth_batch = torch.stack([
            torch.from_numpy(d).float() for _, d, _ in batch
        ], dim=0).to(device)
        mask_batch = torch.stack([
            torch.from_numpy(m).float() for _, _, m in batch
        ], dim=0).to(device)
        y_raw = model.camera(hs_batch, depth_batch, valid_mask=mask_batch)
        sampler.add(y_raw.detach().cpu().numpy())
        valid_patch_count += len(batch)
        batch = []

    for scene_no, top, left in windows:
        hs, depth = scene_cache_get(cache, args, scene_no)
        hs_patch = hs[top:top + args.patch_size, left:left + args.patch_size, :]
        depth_patch = depth[top:top + args.patch_size, left:left + args.patch_size]
        if hs_patch.shape[:2] != (args.patch_size, args.patch_size):
            continue
        mask = (depth_patch > (args.min_depth - 1e-3)).astype(np.float32)
        if mask.mean() < 0.2:
            continue
        hs_norm = normalize_hs_image(hs_patch, norm_mode='fixed_scale', norm_scale=hs_scale)
        depth_metric = np.clip(depth_patch, args.min_depth, args.max_depth)
        depth_metric[mask < 0.5] = args.min_depth
        batch.append((hs_norm, depth_metric.astype(np.float32), mask))
        if len(batch) >= max(1, args.sensor_batch_size):
            flush_batch()

    flush_batch()

    model.camera.measurement_norm_mode = old_norm
    values = sampler.finish()
    return {
        'scale': float(np.percentile(values, args.sensor_percentile)),
        'sampled_values': int(values.size),
        'total_values_seen': int(sampler.total_seen),
        'valid_patches': int(valid_patch_count),
    }


def main():
    args = parse_args()
    windows = load_patch_windows(args)
    hs = estimate_hs_scale(args, windows)
    sensor = estimate_sensor_scale(args, windows, hs['scale'])
    result = {
        'hs_norm_mode': 'fixed_scale',
        'hs_norm_scale': hs['scale'],
        'hs_percentile': args.hs_percentile,
        'hs_valid_patches': hs['valid_patches'],
        'hs_sampled_values': hs['sampled_values'],
        'hs_total_values_seen': hs['total_values_seen'],
        'dodo_forward_norm': 'fixed_scale',
        'dodo_forward_scale': sensor['scale'],
        'sensor_percentile': args.sensor_percentile,
        'sensor_valid_patches': sensor['valid_patches'],
        'sensor_sampled_values': sensor['sampled_values'],
        'sensor_total_values_seen': sensor['total_values_seen'],
        'patch_index_path': args.patch_index_path,
        'patch_windows_used': len(windows),
        'scene_range': [args.scene_start, args.scene_end],
        'ckpt_path': args.ckpt_path,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
