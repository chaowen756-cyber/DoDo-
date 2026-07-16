#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.patch_balance import (
    depth_histograms_ips,
    scale_half_eligibility,
    tempered_depth_sampling_weights,
)


def read_exr_depth(file_path: str) -> np.ndarray:
    try:
        import Imath
        import OpenEXR
    except ImportError as exc:
        raise RuntimeError(
            'OpenEXR is required when the depth cache is missing'
        ) from exc
    if not OpenEXR.isOpenExrFile(file_path):
        raise IOError(f'Not a valid EXR file: {file_path}')
    exr_file = OpenEXR.InputFile(file_path)
    header = exr_file.header()
    data_window = header['dataWindow']
    width = data_window.max.x - data_window.min.x + 1
    height = data_window.max.y - data_window.min.y + 1
    channel_name = sorted(header['channels'].keys())[0]
    channel = header['channels'][channel_name]
    if channel.type == Imath.PixelType(Imath.PixelType.FLOAT):
        dtype = np.float32
    elif channel.type == Imath.PixelType(Imath.PixelType.HALF):
        dtype = np.float16
    else:
        raise TypeError(f'Unsupported EXR depth type: {channel.type}')
    values = np.frombuffer(exr_file.channel(channel_name), dtype=dtype)
    return values.reshape(height, width)


def load_meta(data: np.lib.npyio.NpzFile) -> dict:
    if 'meta_json' not in data.files:
        return {}
    try:
        return json.loads(str(data['meta_json'].item()))
    except Exception:
        return {}


def cache_path(exr_path: Path, cache_dir: Path) -> Path:
    stat = exr_path.stat()
    key_source = f'{exr_path}|{stat.st_mtime_ns}|{stat.st_size}'
    key = hashlib.sha1(key_source.encode('utf-8')).hexdigest()
    return cache_dir / f'{key}.npy'


def load_depth_map(exr_path: Path, cache_dir: Path) -> np.ndarray:
    cached = cache_path(exr_path, cache_dir)
    if cached.exists():
        depth = np.load(cached, allow_pickle=False, mmap_mode='r')
    else:
        cache_dir.mkdir(parents=True, exist_ok=True)
        depth = read_exr_depth(str(exr_path))
        temporary = cached.with_name(f'{cached.name}.tmp.{os.getpid()}.npy')
        np.save(temporary, depth)
        os.replace(temporary, cached)
        depth = np.load(cached, allow_pickle=False, mmap_mode='r')
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    return np.asarray(depth, dtype=np.float32) / 1000.0


def scene_depth_path(data_root: Path, scene_id: str) -> Path:
    scene_number = int(scene_id.rsplit('_', 1)[-1])
    return data_root / f'deploy {scene_number}' / f'scene{scene_number:02d}_depth_map.exr'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Build scene-uniform, IPS-depth-balanced patch-index V2.'
    )
    parser.add_argument('--source_train_index', required=True)
    parser.add_argument('--val_index', required=True)
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--report', default='')
    parser.add_argument('--exr_cache_dir', default='')
    parser.add_argument('--patch_size', type=int, default=128)
    parser.add_argument('--bins', type=int, default=16)
    parser.add_argument('--min_depth', type=float, default=0.4)
    parser.add_argument('--max_depth', type=float, default=2.0)
    parser.add_argument('--target_exponent', type=float, default=0.5)
    parser.add_argument('--weight_min', type=float, default=0.25)
    parser.add_argument('--weight_max', type=float, default=4.0)
    parser.add_argument('--min_ess_ratio', type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bins != 16:
        raise ValueError('--bins must be 16 for depth_hist_ips_16 compatibility')
    data_root = Path(args.data_root).resolve()
    cache_dir = Path(
        args.exr_cache_dir or data_root / '.exr_cache_npy_v1'
    ).resolve()
    output_path = Path(args.output).resolve()
    report_path = Path(args.report).resolve() if args.report else output_path.with_suffix('.json')

    train_data = np.load(args.source_train_index, allow_pickle=False)
    val_data = np.load(args.val_index, allow_pickle=False)
    required = {'scene_ids', 'tops', 'lefts'}
    for label, data in (('train', train_data), ('val', val_data)):
        missing = required - set(data.files)
        if missing:
            raise ValueError(f'{label} index missing fields: {sorted(missing)}')

    scene_ids = train_data['scene_ids'].astype(str)
    tops = train_data['tops'].astype(np.int64)
    lefts = train_data['lefts'].astype(np.int64)
    val_scene_ids = val_data['scene_ids'].astype(str)
    val_tops = val_data['tops'].astype(np.int64)
    val_lefts = val_data['lefts'].astype(np.int64)
    histograms = np.zeros((len(scene_ids), args.bins), dtype=np.uint32)
    scale_eligible = np.zeros(len(scene_ids), dtype=bool)
    scene_diagnostics = {}

    for scene_id in sorted(set(scene_ids.tolist())):
        train_indices = np.flatnonzero(scene_ids == scene_id)
        val_indices = np.flatnonzero(val_scene_ids == scene_id)
        depth_path = scene_depth_path(data_root, scene_id)
        if not depth_path.exists():
            raise FileNotFoundError(depth_path)
        depth_m = load_depth_map(depth_path, cache_dir)
        histograms[train_indices] = depth_histograms_ips(
            depth_m,
            tops[train_indices],
            lefts[train_indices],
            args.patch_size,
            args.min_depth,
            args.max_depth,
            args.bins,
        )
        scale_eligible[train_indices] = scale_half_eligibility(
            tops[train_indices],
            lefts[train_indices],
            val_tops[val_indices],
            val_lefts[val_indices],
            depth_m.shape[0],
            depth_m.shape[1],
            args.patch_size,
        )
        scene_diagnostics[scene_id] = {
            'train_windows': int(len(train_indices)),
            'val_windows': int(len(val_indices)),
            'scale_05_eligible': int(scale_eligible[train_indices].sum()),
            'scale_05_eligible_ratio': float(
                scale_eligible[train_indices].mean()
            ),
        }
        print(
            f'[{scene_id}] train={len(train_indices)} val={len(val_indices)} '
            f'scale05={scale_eligible[train_indices].sum()}'
        )

    weights, balance_report = tempered_depth_sampling_weights(
        histograms,
        scene_ids,
        target_exponent=args.target_exponent,
        minimum=args.weight_min,
        maximum=args.weight_max,
    )
    if balance_report['ess_ratio'] < args.min_ess_ratio:
        raise RuntimeError(
            f"ESS ratio {balance_report['ess_ratio']:.4f} is below "
            f'{args.min_ess_ratio:.4f}'
        )
    for scene_id, scene_report in balance_report['scenes'].items():
        if scene_report['ess_ratio'] < args.min_ess_ratio:
            raise RuntimeError(
                f"{scene_id} ESS ratio {scene_report['ess_ratio']:.4f} is below "
                f'{args.min_ess_ratio:.4f}'
            )

    source_meta = load_meta(train_data)
    metadata = dict(source_meta)
    metadata.update(
        {
            'index_type': 'scene_uniform_ips_depth_balanced_v2',
            'source_train_index': str(Path(args.source_train_index).resolve()),
            'validation_index': str(Path(args.val_index).resolve()),
            'patch_size': args.patch_size,
            'depth_histogram_bins': args.bins,
            'depth_histogram_edges_ips': np.linspace(
                0.0, 1.0, args.bins + 1
            ).tolist(),
            'sampling_target': f'frequency^{args.target_exponent:g}',
            'sampling_weight_bounds': [args.weight_min, args.weight_max],
            'sampling_scene_normalized': True,
            'sampling_ess_ratio': balance_report['ess_ratio'],
            'scale_05_source_size': args.patch_size * 2,
            'scale_05_excludes_validation_union': True,
            'scale_05_eligible_count': int(scale_eligible.sum()),
        }
    )

    arrays = {
        name: train_data[name]
        for name in train_data.files
        if name != 'meta_json'
    }
    if 'quality_scores' not in arrays:
        arrays['quality_scores'] = (
            arrays['scores'].astype(np.float32)
            if 'scores' in arrays
            else np.ones(len(scene_ids), dtype=np.float32)
        )
    arrays['depth_hist_ips_16'] = histograms
    arrays['sampling_weight'] = weights
    arrays['scale_05_eligible'] = scale_eligible
    arrays['meta_json'] = np.asarray(json.dumps(metadata, ensure_ascii=False))

    report = {
        'output': str(output_path),
        'metadata': metadata,
        'balance': balance_report,
        'scenes': scene_diagnostics,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f'{output_path.name}.tmp.{os.getpid()}')
    with open(temporary_path, 'wb') as output_file:
        np.savez_compressed(output_file, **arrays)
    os.replace(temporary_path, output_path)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )
    print(
        f'[done] windows={len(scene_ids)} scale05={scale_eligible.sum()} '
        f"ESS={balance_report['ess_ratio']:.4f}"
    )
    print(f'[done] index: {output_path}')
    print(f'[done] report: {report_path}')


if __name__ == '__main__':
    main()
