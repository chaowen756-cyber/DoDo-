#!/usr/bin/env python
"""Round 23 low-foreground tile DoDo measurement-energy diagnostic.

Loads the DoDo-depth model/checkpoint and computes full / foreground-only /
background-only optical measurement energy on low-foreground 128x128 tiles.

Background-only policy:
  Background pixels have no valid depth. Following the same approach as the
  inference pipeline, bg depth is clamped to min_depth. HS input is multiplied
  by (1-fg_mask) to isolate background spectral content. The resulting
  measurement represents what the DoDo camera would produce from background-only
  input with clamped depth.
"""

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime
from glob import glob
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))


def _cache_key(file_path):
    stat = os.stat(file_path)
    key_src = f"{file_path}|{stat.st_mtime_ns}|{stat.st_size}"
    return hashlib.sha1(key_src.encode('utf-8')).hexdigest()


def load_camera_from_checkpoint(ckpt_path, device='cpu'):
    """Load DepthAwareDoDoForwardModel camera from a SnapshotDepthHS checkpoint."""
    from snapshotdepth_hs import SnapshotDepthHS

    # Load the model - suppress verbose logging
    model = SnapshotDepthHS.load_from_checkpoint(
        ckpt_path, map_location=device, strict=False,
    )
    model.eval()
    return model.camera, model.hparams


def compute_measurement_energy_3way(camera, camera_hp, hs_tile, depth_tile,
                                      valid_mask_tile, device='cpu'):
    """Compute full/fg/bg measurement energy from actual DoDo optical path.

    Returns dict with measurement energy stats for full / fg-only / bg-only.
    """
    # Ensure correct shapes for camera (nchw format expected by DepthAwareDoDoForwardModel)
    # hs_tile: (25, 128, 128), depth_tile: (1, 128, 128), valid_mask_tile: (1, 128, 128)
    hs_batch = hs_tile.unsqueeze(0).float().to(device)        # (1, 25, 128, 128)
    depth_batch = depth_tile.unsqueeze(0).float().to(device)  # (1, 1, 128, 128)
    vm_batch = valid_mask_tile.unsqueeze(0).float().to(device)  # (1, 1, 128, 128)

    fg_mask = (vm_batch > 0.5).float()
    bg_mask = 1.0 - fg_mask

    depth_min = getattr(camera_hp, 'min_depth', 0.4)
    depth_max = getattr(camera_hp, 'max_depth', 2.0)

    results = {}

    with torch.no_grad():
        # --- Full measurement ---
        meas_full = camera(hs_batch, depth_batch, valid_mask=vm_batch)
        results['measurement_full_energy'] = float(meas_full.abs().sum().item())

        # --- Foreground-only measurement ---
        hs_fg = hs_batch * fg_mask
        meas_fg = camera(hs_fg, depth_batch, valid_mask=vm_batch)
        results['measurement_fg_energy'] = float(meas_fg.abs().sum().item())

        # --- Background-only measurement ---
        # Policy: bg HS = hs * (1-fg_mask), bg depth clamped to min_depth.
        # This mirrors the inference pipeline: invalid-depth pixels get clamp(min_depth).
        depth_bg = depth_batch.clone()
        depth_bg[bg_mask < 0.5] = depth_min
        depth_bg = depth_bg.clamp(min=depth_min, max=depth_max)
        hs_bg = hs_batch * bg_mask
        meas_bg = camera(hs_bg, depth_bg, valid_mask=bg_mask)
        results['measurement_bg_energy'] = float(meas_bg.abs().sum().item())

    total = results['measurement_full_energy']
    fg_e = results['measurement_fg_energy']
    bg_e = results['measurement_bg_energy']
    results['measurement_fg_fraction'] = float(fg_e / (total + 1e-10))
    results['measurement_bg_fraction'] = float(bg_e / (total + 1e-10))
    results['measurement_fg_bg_ratio'] = float(fg_e / (bg_e + 1e-10))
    results['measurement_all_finite'] = bool(
        np.isfinite([total, fg_e, bg_e]).all()
    )

    return results


def scan_lowfg_tiles(hs_npy_path, depth_npy_path, cache_dir,
                     patch_size=128, min_depth=0.4):
    """Scan depth map for low-foreground tile coordinates using .npy cache."""
    depth_np = np.load(os.path.join(cache_dir,
                                     f'{_cache_key(depth_npy_path)}.npy'))
    if depth_np.ndim == 3 and depth_np.shape[2] == 1:
        depth_np = depth_np[:, :, 0]
    H, W = depth_np.shape

    valid_mask = (depth_np > min_depth - 1e-3)

    coords = []
    for y in range(0, H - patch_size + 1, patch_size):
        for x in range(0, W - patch_size + 1, patch_size):
            vm = valid_mask[y:y + patch_size, x:x + patch_size]
            valid_ratio = float(vm.mean())
            if valid_ratio < 0.1:
                coords.append((y, x, valid_ratio))
    coords.sort(key=lambda c: c[2])
    return coords, (H, W), valid_mask


def main():
    parser = argparse.ArgumentParser(description='Round 23 DoDo measurement-energy diagnostic')
    parser.add_argument('--data_root', type=str, default=None,
                        help='Dataset root')
    parser.add_argument('--deploys', type=str, nargs='+', default=['deploy 1', 'deploy 16'])
    parser.add_argument('--output_root', type=str, default=None)
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='DoDo checkpoint path (default: R12 depth-best)')
    parser.add_argument('--patch_size', type=int, default=128)
    parser.add_argument('--min_depth', type=float, default=0.4)
    parser.add_argument('--max_measure_tiles', type=int, default=5,
                        help='Max tiles per scene to run full measurement on')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for measurement computation')
    args = parser.parse_args()

    # Resolve paths
    if args.data_root is None:
        data_root = SCRIPT_DIR / 'Baek数据集'
    else:
        data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f'Data root not found: {data_root}')

    cache_dir = str(data_root / '.exr_cache_npy_v1')
    if not os.path.isdir(cache_dir):
        raise FileNotFoundError(f'EXR cache not found: {cache_dir}')

    if args.ckpt_path is None:
        ckpt_path = str(SCRIPT_DIR / 'infer_results' / 'DoDo-change'
                         / 'DoDo_depth_finite_joint_metricdepth_260_v1'
                         / '20260507_112631' / 'checkpoints'
                         / 'depth-best-epoch=226.ckpt')
    else:
        ckpt_path = args.ckpt_path
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

    if args.output_root is None:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_root = (SCRIPT_DIR / 'infer_results' / 'DoDo-change'
                       / f'round23_repair_lowfg_v1' / ts)
    else:
        output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f'Data root:   {data_root}')
    print(f'Checkpoint:  {ckpt_path}')
    print(f'Output:      {output_root}')

    # Determine device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print('[WARN] CUDA not available, falling back to CPU')

    # Load camera from checkpoint
    print('\nLoading DoDo camera from checkpoint...')
    camera, camera_hp = load_camera_from_checkpoint(ckpt_path, device='cpu')
    camera = camera.to(device)
    print(f'  optical_model:     {getattr(camera_hp, "optical_model", "N/A")}')
    print(f'  depth_layering:    {getattr(camera, "depth_layering_mode", "N/A")}')
    print(f'  sensor_measurement:{getattr(camera_hp, "dodo_sensor_measurement", "amplitude")}')
    print(f'  forward_norm:      {getattr(camera_hp, "dodo_forward_norm", "legacy_max")}')
    print(f'  measurement_norm:  {getattr(camera_hp, "dodo_measurement_norm", "per_sample_mean_std")}')

    # Build metadata
    model_meta = {
        'checkpoint_path': ckpt_path,
        'optical_model': str(getattr(camera_hp, 'optical_model', 'dodo_depth')),
        'sensor_measurement': str(getattr(camera_hp, 'dodo_sensor_measurement', 'amplitude')),
        'depth_layering_mode': str(getattr(camera, 'depth_layering_mode', 'hard_depth')),
        'dodo_forward_norm': str(getattr(camera_hp, 'dodo_forward_norm', 'legacy_max')),
        'dodo_measurement_norm': str(getattr(camera_hp, 'dodo_measurement_norm', 'per_sample_mean_std')),
        'background_policy': (
            'bg_hs = hs * (1-fg_mask); '
            'bg_depth = clamp(invalid_depth, min_depth, max_depth); '
            'bg_valid_mask = 1-fg_mask (treated as valid for bg-only synthesis)'
        ),
        'patch_size': args.patch_size,
        'min_depth': args.min_depth,
        'valid_ratio_threshold': 0.1,
    }

    all_tiles = []
    scene_summaries = []

    for deploy_name in args.deploys:
        deploy_dir = data_root / deploy_name
        if not deploy_dir.exists():
            print(f'\n[SKIP] Deploy not found: {deploy_dir}')
            continue

        hs_files = sorted(glob(str(deploy_dir / '*_hs.exr')))
        if not hs_files:
            print(f'\n[SKIP] No HS EXR in: {deploy_dir}')
            continue

        for hs_f in hs_files:
            scene_name = Path(hs_f).stem.replace('_hs', '')
            depth_f = hs_f.replace('_hs.exr', '_depth_map.exr')
            if not os.path.exists(depth_f):
                print(f'\n[SKIP] No depth file for: {hs_f}')
                continue

            print(f'\n=== {deploy_name}/{scene_name} ===')

            # Step 1: scan depth map for lowfg tiles (fast, no HS loading)
            coords, (H, W), valid_mask_full = scan_lowfg_tiles(
                hs_f, depth_f, cache_dir, args.patch_size, args.min_depth)
            total_tiles = (H // args.patch_size) * (W // args.patch_size)
            print(f'  Image: {H}x{W}, total tiles: {total_tiles}')
            print(f'  Found {len(coords)} tiles with valid_ratio < 0.1')

            if not coords:
                scene_summaries.append({
                    'deploy': deploy_name, 'scene': scene_name,
                    'total_lowfg_tiles': 0, 'total_tiles_scanned': total_tiles,
                    'note': 'No tile with valid_ratio < 0.1 found',
                })
                continue

            # Step 2: load HS tiles for measurement (only for selected tiles)
            hs_npy = np.load(os.path.join(cache_dir,
                                           f'{_cache_key(hs_f)}.npy'))
            depth_npy = np.load(os.path.join(cache_dir,
                                              f'{_cache_key(depth_f)}.npy'))
            if depth_npy.ndim == 3 and depth_npy.shape[2] == 1:
                depth_npy = depth_npy[:, :, 0]

            n_measure = min(args.max_measure_tiles, len(coords))
            print(f'  Running DoDo measurement on {n_measure} tiles...')

            for idx, (y, x, valid_ratio) in enumerate(coords[:n_measure]):
                # Extract tiles (use first 25 bands — HS cube may have 29 channels in cache)
                hs_data = hs_npy[y:y + args.patch_size, x:x + args.patch_size, :25].astype(np.float32)
                hs_tile = torch.from_numpy(hs_data).permute(2, 0, 1)  # (25, H, W)
                depth_tile = torch.from_numpy(
                    depth_npy[y:y + args.patch_size, x:x + args.patch_size].astype(np.float32)
                )
                if depth_tile.ndim == 2:
                    depth_tile = depth_tile.unsqueeze(0)  # (1, H, W)
                vm_tile = torch.from_numpy(
                    valid_mask_full[y:y + args.patch_size, x:x + args.patch_size].astype(np.float32)
                ).unsqueeze(0)  # (1, H, W)

                # Raw HS input energy (for comparison)
                hs_abs = hs_tile.abs()
                fg_mask = (vm_tile > 0.5).float()
                total_hs_e = float(hs_abs.sum().item())
                fg_hs_e = float((hs_abs * fg_mask).sum().item())
                bg_hs_e = float((hs_abs * (1 - fg_mask)).sum().item())

                # DoDo measurement energy
                meas = compute_measurement_energy_3way(
                    camera, camera_hp, hs_tile, depth_tile, vm_tile, device=device)

                row = {
                    'deploy': deploy_name,
                    'scene': scene_name,
                    'tile_y': y,
                    'tile_x': x,
                    'valid_ratio': valid_ratio,
                    # Raw HS input energy (diagnostic-only, NOT measurement)
                    'hs_full_energy': total_hs_e,
                    'hs_fg_energy': fg_hs_e,
                    'hs_bg_energy': bg_hs_e,
                    'hs_fg_fraction': float(fg_hs_e / (total_hs_e + 1e-10)),
                    'hs_bg_fraction': float(bg_hs_e / (total_hs_e + 1e-10)),
                    # DoDo optical measurement energy
                    **meas,
                }
                all_tiles.append(row)

                print(f'    tile ({y},{x}) vr={valid_ratio:.4f}: '
                      f'meas_full={meas["measurement_full_energy"]:.2e}, '
                      f'meas_fg_frac={meas["measurement_fg_fraction"]:.4f}, '
                      f'meas_bg_frac={meas["measurement_bg_fraction"]:.4f}, '
                      f'all_finite={meas["measurement_all_finite"]}')

            # Scene summary
            meas_tiles_scene = [t for t in all_tiles
                                if t.get('deploy') == deploy_name
                                and t.get('scene') == scene_name]
            if meas_tiles_scene:
                fg_fracs = [t['measurement_fg_fraction'] for t in meas_tiles_scene]
                bg_fracs = [t['measurement_bg_fraction'] for t in meas_tiles_scene]
                scene_summaries.append({
                    'deploy': deploy_name,
                    'scene': scene_name,
                    'total_lowfg_tiles': len(coords),
                    'total_tiles_scanned': total_tiles,
                    'measured_tiles': len(meas_tiles_scene),
                    'measurement_fg_fraction_mean': float(np.mean(fg_fracs)),
                    'measurement_fg_fraction_min': float(np.min(fg_fracs)),
                    'measurement_fg_fraction_max': float(np.max(fg_fracs)),
                    'measurement_bg_fraction_mean': float(np.mean(bg_fracs)),
                    'measurement_vs_hs_energy_ratio': float(
                        np.mean([t['measurement_full_energy'] / (t['hs_full_energy'] + 1e-10)
                                 for t in meas_tiles_scene])
                    ),
                })

            # Free memory
            del hs_npy, depth_npy

    # --- Write outputs ---
    csv_path = output_root / 'diag_lowfg_tiles.csv'
    if all_tiles:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(all_tiles[0].keys()))
            writer.writeheader()
            writer.writerows(all_tiles)
        print(f'\nCSV saved: {csv_path} ({len(all_tiles)} rows)')
    else:
        with open(csv_path, 'w') as f:
            f.write('No low-foreground tiles found.\n')
        print('\n[WARN] No data to write to CSV.')

    summary = {
        'description': 'Round 23 DoDo optical measurement-energy diagnostic (repaired)',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        **model_meta,
        'total_lowfg_tiles_found': len(all_tiles),
        'scene_summaries': scene_summaries,
    }
    summary_path = output_root / 'diag_lowfg_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str, ensure_ascii=False)
    print(f'Summary saved: {summary_path}')

    cmd_path = output_root / 'command.txt'
    with open(cmd_path, 'w') as f:
        f.write(' '.join(sys.argv) + '\n')

    print(f'\nArtifact root: {output_root}')
    print('Done.')


if __name__ == '__main__':
    main()
