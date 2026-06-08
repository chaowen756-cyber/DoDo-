#!/usr/bin/env python
"""Compare init vs trained spectral capacity diagnostic summaries."""

import argparse
import json
import sys
from pathlib import Path


def load_summary(path):
    with open(path) as f:
        return json.load(f)


def compare(init_path, trained_path, output_path):
    init = load_summary(init_path)
    trained = load_summary(trained_path)

    def _pct(a, b):
        if a == 0:
            return float('inf') if b != 0 else 0.0
        return round((b - a) / abs(a) * 100, 2)

    # Build comparison
    metrics = [
        ('adjacent_corr_mean', 'diag_b_adjacent_corr_mean', 'lower_better'),
        ('adjacent_corr_std',  'diag_b_adjacent_corr_std',  'lower_better'),
        ('adjacent_corr_min',  'diag_b_adjacent_corr_min',  'higher_better'),
        ('adjacent_corr_max',  'diag_b_adjacent_corr_max',  'lower_better'),
        ('energy_mean',        'diag_c_energy_mean',        'lower_better'),
        ('energy_std',         'diag_c_energy_std',         'lower_better'),
        ('energy_min',         'diag_c_energy_min',         'higher_better'),
        ('energy_max',         'diag_c_energy_max',         'lower_better'),
        ('effective_rank',     'diag_d_effective_rank',     'higher_better'),
        ('condition_number',   'diag_d_condition_number',   'lower_better'),
        ('mutual_coherence',   'diag_d_mutual_coherence',   'lower_better'),
    ]

    rows = []
    for label, key, direction in metrics:
        v_init = init.get(key, None)
        v_trained = trained.get(key, None)
        if v_init is None or v_trained is None:
            continue
        change_pct = _pct(v_init, v_trained)
        # Determine if change is favorable
        if direction == 'lower_better':
            favorable = v_trained < v_init
        else:
            favorable = v_trained > v_init

        rows.append({
            'metric': label,
            'init': v_init,
            'trained': v_trained,
            'change_pct': change_pct,
            'direction': direction,
            'favorable': favorable,
        })

    # Print table
    print(f"{'Metric':<24} {'Init':>12} {'Trained':>12} {'Change%':>9} {'Direction':>14} {'Better?':>8}")
    print("-" * 84)
    for r in rows:
        arrow = "✓" if r['favorable'] else "✗"
        print(f"{r['metric']:<24} {r['init']:>12.4f} {r['trained']:>12.4f} "
              f"{r['change_pct']:>+8.1f}% {r['direction']:>14} {arrow:>8}")

    # Summary
    n_better = sum(1 for r in rows if r['favorable'])
    n_total = len(rows)
    print(f"\nTrained DOE improves {n_better}/{n_total} metrics.")

    # Interpretation
    print("\n=== Interpretation ===")
    er_init = init.get('diag_d_effective_rank', 0)
    er_trained = trained.get('diag_d_effective_rank', 0)
    mc_init = init.get('diag_d_mutual_coherence', 1)
    mc_trained = trained.get('diag_d_mutual_coherence', 1)
    cn_init = init.get('diag_d_condition_number', 0)
    cn_trained = trained.get('diag_d_condition_number', 0)

    print(f"  Effective rank:     {er_init:.1f} → {er_trained:.1f} "
          f"({er_trained/er_init:.1f}× increase)")
    print(f"  Mutual coherence:   {mc_init:.4f} → {mc_trained:.4f} "
          f"({'improved' if mc_trained < mc_init else 'worsened'})")
    print(f"  Condition number:   {cn_init:.2e} → {cn_trained:.2e} "
          f"({cn_init/cn_trained:.0f}× better conditioned)")

    if mc_trained > 0.9:
        print(f"\n  ⚠  mutual_coherence still > 0.9 after training: "
              f"3-channel measurement remains severely compressed.")
    if er_trained < 100:
        print(f"  ⚠  effective_rank ({er_trained:.1f}) < column count (144): "
              f"measurement matrix still rank-deficient.")

    # Save
    compare_data = {
        'init_summary': init_path,
        'trained_summary': trained_path,
        'comparison_rows': rows,
        'n_metrics_improved': n_better,
        'n_metrics_total': n_total,
        'init_checkpoint': init.get('checkpoint', 'none'),
        'trained_checkpoint': trained.get('checkpoint', 'none'),
    }
    with open(output_path, 'w') as f:
        json.dump(compare_data, f, indent=2, default=str)
    print(f"\nComparison saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare spectral capacity diagnostic summaries')
    parser.add_argument('--init', type=str, required=True,
                        help='Path to init (Zeros DOE) summary.json')
    parser.add_argument('--trained', type=str, required=True,
                        help='Path to trained checkpoint summary.json')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for compare_summary.json')
    args = parser.parse_args()

    compare(args.init, args.trained, args.output)


if __name__ == '__main__':
    main()
