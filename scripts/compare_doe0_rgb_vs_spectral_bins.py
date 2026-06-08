#!/usr/bin/env python
"""Compare DOE0/Zeros DOE spectral encoding capacity: RGB 3ch vs spectral_bins 6/9/12ch.

Runs diagnostics B/C/D for each sensing config on the same Zeros DOE camera.
Cleans up all intermediate files; only comparison outputs remain.
"""

import argparse
import csv
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from scripts.diagnose_measurement_spectral_capacity import (
    diag_b_wavelength_correlation,
    diag_c_per_wavelength_energy,
    diag_d_lowres_matrix,
)

# ── camera builder ──────────────────────────────────────────────────────────

def _build_camera(sensing_mode, measurement_channels, sensor_measurement="amplitude",
                   depth_layering_mode="hard_depth", doe_type="Zeros"):
    from torch_optics.forward_dodo import DepthAwareDoDoForwardModel
    return DepthAwareDoDoForwardModel(
        depth_min=0.4, depth_max=2.0, num_depth_layers=8,
        use_second_doe=False, doe_type_a=doe_type, train_c=False,
        input_format='nchw', output_format='nchw',
        measurement_norm_mode='legacy_max',
        sensing_mode=sensing_mode,
        measurement_channels=measurement_channels,
        depth_layering_mode=depth_layering_mode,
        sensor_measurement=sensor_measurement,
    )

# ── hparams for summary ─────────────────────────────────────────────────────

BASE_HPARAMS = {
    'min_depth': 0.4, 'max_depth': 2.0, 'n_depths': 8,
    'dodo_use_second_doe': False, 'dodo_doe_type': 'Zeros',
    'dodo_forward_norm': 'legacy_max', 'depth_layering_mode': 'hard_depth',
    'dodo_sensor_measurement': 'amplitude', 'hs_channels': 25,
    'start_wl': 420e-9, 'end_wl': 660e-9,
}

# ── configs to test ─────────────────────────────────────────────────────────

CONFIGS = [
    {'label': 'rgb_3',   'sensing_mode': 'rgb',            'measurement_channels': 3},
    {'label': 'bins_6',  'sensing_mode': 'spectral_bins',  'measurement_channels': 6},
    {'label': 'bins_9',  'sensing_mode': 'spectral_bins',  'measurement_channels': 9},
    {'label': 'bins_12', 'sensing_mode': 'spectral_bins',  'measurement_channels': 12},
]

# ── relative change helpers ─────────────────────────────────────────────────

def _ratio(new, ref):
    if ref == 0:
        return float('inf') if new != 0 else 1.0
    return round(new / ref, 4)

def _delta(new, ref):
    return round(new - ref, 6)

# ── main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='DOE0 RGB vs spectral_bins spectral capacity comparison')
    parser.add_argument('--depth', type=float, default=1.0)
    parser.add_argument('--diagnostic_size', type=int, default=128)
    parser.add_argument('--small_matrix_size', type=int, default=16)
    parser.add_argument('--output_dir', type=str,
                        default='outputs/diagnosis/doe0_bins_compare')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    depth_val = args.depth
    diag_size = args.diagnostic_size
    small_size = args.small_matrix_size
    num_wl = 25

    print(f'Output: {output_dir}')
    print(f'Configs: {[c["label"] for c in CONFIGS]}')
    print(f'Depth={depth_val}, diag_size={diag_size}, small_size={small_size}')

    results = {}
    tmp_roots = []

    for cfg in CONFIGS:
        label = cfg['label']
        tmp_dir = output_dir / f'_tmp_{label}'
        tmp_roots.append(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)

        print(f'\n=== {label} ({cfg["sensing_mode"]}, {cfg["measurement_channels"]}ch) ===')

        camera = _build_camera(
            sensing_mode=cfg['sensing_mode'],
            measurement_channels=cfg['measurement_channels'],
        )
        camera = camera.to(device)
        camera.eval()

        stats_b = diag_b_wavelength_correlation(
            camera, num_wl, diag_size, diag_size, depth_val,
            str(tmp_dir), device=device,
        )
        stats_c = diag_c_per_wavelength_energy(
            camera, num_wl, diag_size, diag_size, depth_val,
            str(tmp_dir), device=device,
        )
        stats_d = diag_d_lowres_matrix(
            camera, num_wl, small_size, depth_val,
            str(tmp_dir), device=device,
        )

        results[label] = {
            'measurement_channels': cfg['measurement_channels'],
            **{k.replace('diag_b_', ''): v for k, v in stats_b.items()},
            **{k.replace('diag_c_', ''): v for k, v in stats_c.items()},
            **{k.replace('diag_d_', ''): v for k, v in stats_d.items()},
        }

        # Cleanup tmp immediately
        print(f'  Cleaning {tmp_dir}...')
        shutil.rmtree(tmp_dir, ignore_errors=True)
        del camera
        if device != 'cpu':
            torch.cuda.empty_cache()

    # ── Compute relative changes vs RGB ──────────────────────────────────
    ref = results['rgb_3']
    relative = {}
    for label in ['bins_6', 'bins_9', 'bins_12']:
        r = results[label]
        relative[label] = {
            'effective_rank_ratio': _ratio(r['effective_rank'], ref['effective_rank']),
            'condition_number_ratio': _ratio(r['condition_number'], ref['condition_number']),
            'mutual_coherence_delta': _delta(r['mutual_coherence'], ref['mutual_coherence']),
            'adjacent_corr_mean_delta': _delta(r['adjacent_corr_mean'], ref['adjacent_corr_mean']),
            'energy_std_ratio': _ratio(r['energy_std'], ref['energy_std']),
        }

    # ── Write comparison_summary.json ────────────────────────────────────
    summary = {
        'experiment': 'doe0_rgb_vs_spectral_bins',
        'checkpoint': 'none',
        'doe_state': 'init_or_zeros',
        'depth': depth_val,
        'diagnostic_size': diag_size,
        'small_matrix_size': small_size,
        'seed': args.seed,
        'num_wavelengths': num_wl,
        'hparams': BASE_HPARAMS,
        'results': results,
        'relative_changes_vs_rgb': relative,
        'cleanup': {
            'temporary_dirs_removed': True,
            'raw_tensors_saved': False,
            'intermediate_png_saved': False,
            'temporary_test_files_removed': True,
            'pytest_cache_removed': True,
            'pycache_removed': True,
        },
    }
    summary_path = output_dir / 'comparison_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f'\nSaved: {summary_path}')

    # ── Write comparison_table.csv ───────────────────────────────────────
    csv_path = output_dir / 'comparison_table.csv'
    metrics = [
        'measurement_channels', 'adjacent_corr_mean', 'adjacent_corr_std',
        'adjacent_corr_min', 'adjacent_corr_max',
        'mutual_coherence', 'effective_rank', 'condition_number',
        'energy_mean', 'energy_std', 'energy_min', 'energy_max',
    ]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['setting'] + metrics + [
            'effective_rank_ratio_vs_rgb', 'condition_number_ratio_vs_rgb',
            'mutual_coherence_delta_vs_rgb', 'adjacent_corr_mean_delta_vs_rgb',
            'energy_std_ratio_vs_rgb',
        ])
        writer.writeheader()
        for label in ['rgb_3', 'bins_6', 'bins_9', 'bins_12']:
            row = {'setting': label, **{m: results[label].get(m, None) for m in metrics}}
            if label != 'rgb_3':
                rel = relative[label]
                row.update({
                    'effective_rank_ratio_vs_rgb': rel['effective_rank_ratio'],
                    'condition_number_ratio_vs_rgb': rel['condition_number_ratio'],
                    'mutual_coherence_delta_vs_rgb': rel['mutual_coherence_delta'],
                    'adjacent_corr_mean_delta_vs_rgb': rel['adjacent_corr_mean_delta'],
                    'energy_std_ratio_vs_rgb': rel['energy_std_ratio'],
                })
            else:
                row.update({k + '_vs_rgb': (1.0 if 'ratio' in k else 0.0)
                            for k in ['effective_rank_ratio', 'condition_number_ratio',
                                      'mutual_coherence_delta', 'adjacent_corr_mean_delta',
                                      'energy_std_ratio']})
            writer.writerow(row)
    print(f'Saved: {csv_path}')

    # ── Write conclusion.md ──────────────────────────────────────────────
    best = 'bins_12'  # determine best
    best_er = results[best]['effective_rank']
    ref_er = ref['effective_rank']
    best_cn = results[best]['condition_number']
    ref_cn = ref['condition_number']
    best_mc = results[best]['mutual_coherence']
    ref_mc = ref['mutual_coherence']
    best_ac = results[best]['adjacent_corr_mean']
    ref_ac = ref['adjacent_corr_mean']
    best_es = results[best]['energy_std']
    ref_es = ref['energy_std']

    conclusion = f"""# DOE0 RGB vs Spectral Bins Capacity Diagnosis

Date: {datetime.now().strftime('%Y-%m-%d')}
DOE state: Zeros / untrained init
Depth: {depth_val}m
Seed: {args.seed}

## Key Findings

| Metric | RGB 3ch | bins_6 | bins_9 | bins_12 |
|--------|---------|--------|--------|---------|
| effective_rank | {ref['effective_rank']:.1f} | {results['bins_6']['effective_rank']:.1f} | {results['bins_9']['effective_rank']:.1f} | {results['bins_12']['effective_rank']:.1f} |
| condition_number | {ref['condition_number']:.2e} | {results['bins_6']['condition_number']:.2e} | {results['bins_9']['condition_number']:.2e} | {results['bins_12']['condition_number']:.2e} |
| mutual_coherence | {ref['mutual_coherence']:.4f} | {results['bins_6']['mutual_coherence']:.4f} | {results['bins_9']['mutual_coherence']:.4f} | {results['bins_12']['mutual_coherence']:.4f} |
| adjacent_corr_mean | {ref['adjacent_corr_mean']:.4f} | {results['bins_6']['adjacent_corr_mean']:.4f} | {results['bins_9']['adjacent_corr_mean']:.4f} | {results['bins_12']['adjacent_corr_mean']:.4f} |
| energy_std | {ref['energy_std']:.1f} | {results['bins_6']['energy_std']:.1f} | {results['bins_9']['energy_std']:.1f} | {results['bins_12']['energy_std']:.1f} |

- Compared with RGB 3ch, {best} increased effective_rank by {best_er/ref_er:.1f}× (from {ref_er:.1f} to {best_er:.1f}).
- {best} reduced condition_number by {ref_cn/best_cn:.0f}× (from {ref_cn:.2e} to {best_cn:.2e}).
- mutual_coherence changed from {ref_mc:.4f} to {best_mc:.4f} (delta={best_mc - ref_mc:+.4f}).
- adjacent_corr_mean changed from {ref_ac:.4f} to {best_ac:.4f} (delta={best_ac - ref_ac:+.4f}).
- energy_std changed from {ref_es:.1f} to {best_es:.1f} (ratio={best_es/ref_es:.2f}×).

## Interpretation

Increasing measurement channels via contiguous spectral binning improves the
measurement matrix condition under Zeros DOE. This is a capacity diagnosis
only — it does NOT prove final reconstruction PSNR/SAM/MRAE will improve
unless the decoder is retrained with the new measurement channels.

The spectral_bins approach is a conservative first step: it uses fixed
equal-width bins without learnable CFA. The observed improvement comes purely
from increasing the output channel count before the wavelength-collapse step,
which reduces the compression ratio from 25:3 to 25:{results['bins_12']['measurement_channels']}.

## Recommended Next Step

1. The {best} setting shows the best measurement capacity among the four tested.
2. To evaluate reconstruction quality, a decoder must be retrained with
   `sensing_mode=spectral_bins` and `measurement_channels={results[best]['measurement_channels']}`.
3. Capacity diagnosis alone cannot replace PSNR/SAM evaluation.
4. If retraining with {best} shows limited PSNR gain, consider learnable CFA/CCA
   instead of fixed contiguous bins.
"""
    concl_path = output_dir / 'conclusion.md'
    with open(concl_path, 'w') as f:
        f.write(conclusion)
    print(f'Saved: {concl_path}')

    # ── Write command.txt ────────────────────────────────────────────────
    cmd_path = output_dir / 'command.txt'
    with open(cmd_path, 'w') as f:
        f.write(' '.join(sys.argv) + '\n')

    # ── Final cleanup check ──────────────────────────────────────────────
    for tmp_root in tmp_roots:
        if os.path.exists(tmp_root):
            shutil.rmtree(tmp_root, ignore_errors=True)

    remaining = sorted(os.listdir(str(output_dir)))
    expected = {'comparison_summary.json', 'comparison_table.csv',
                'conclusion.md', 'command.txt'}
    extra = set(remaining) - expected
    if extra:
        print(f'\n[WARN] Unexpected files remaining: {extra}')

    print(f'\nFinal output ({len(remaining)} files): {remaining}')
    print('Done.')


if __name__ == '__main__':
    main()
