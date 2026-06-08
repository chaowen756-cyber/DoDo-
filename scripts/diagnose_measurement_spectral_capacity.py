#!/usr/bin/env python
"""Diagnose measurement spectral capacity of the DoDo optical forward model.

Diagnostics:
  A: single-wavelength impulse response (per-wavelength measurement visualization)
  B: wavelength response correlation matrix
  C: per-wavelength sensor energy
  D: low-resolution measurement matrix SVD / effective rank / condition number

Memory policy:
  - Process one wavelength at a time, detach to CPU immediately.
  - Do not save raw response tensors unless --save_raw_responses true.
  - Clean up temporary files at exit.
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

# ── helpers ──────────────────────────────────────────────────────────────────

def _resolve_depth_value(depth_str, z_min=0.4, z_max=2.0):
    if depth_str == "z_min":
        return float(z_min)
    elif depth_str == "z_max":
        return float(z_max)
    elif depth_str == "z_mid":
        return float((z_min + z_max) / 2.0)
    try:
        return float(depth_str)
    except ValueError:
        raise ValueError(f"Invalid depth value: {depth_str}")

def _safe_norm(t):
    return t / (t.norm(dim=-1, keepdim=True) + 1e-10)

def _cleanup_temp_files(output_dir):
    """Remove temporary/intermediate files from output directory."""
    patterns = ["tmp_*.pt", "tmp_*.npy", "cache_*", "impulse_*.pt",
                "raw_response_*.pt", "raw_response_*.npy",
                "temp_*", "intermediate_*", "input_lambda_*.pt"]
    import glob as _glob
    cleaned = 0
    for pat in patterns:
        for f in _glob.glob(str(Path(output_dir) / pat)):
            os.remove(f)
            cleaned += 1
    if cleaned:
        print(f"[cleanup] Removed {cleaned} temporary files from {output_dir}")

def _build_camera_from_hparams(hparams_dict):
    """Build DepthAwareDoDoForwardModel from config dict."""
    from torch_optics.forward_dodo import DepthAwareDoDoForwardModel

    return DepthAwareDoDoForwardModel(
        depth_min=float(hparams_dict.get('min_depth', 0.4)),
        depth_max=float(hparams_dict.get('max_depth', 2.0)),
        num_depth_layers=int(hparams_dict.get('dodo_depth_layers',
                                               hparams_dict.get('n_depths', 8))),
        use_second_doe=bool(hparams_dict.get('dodo_use_second_doe', False)),
        doe_type_a=str(hparams_dict.get('dodo_doe_type', 'Zeros')),
        train_c=False,
        input_format='nchw',
        output_format='nchw',
        measurement_norm_mode=str(hparams_dict.get('dodo_forward_norm', 'legacy_max')),
        sensing_mode=str(hparams_dict.get('dodo_sensing_mode', 'rgb')),
        measurement_channels=int(hparams_dict.get('measurement_channels', 3)),
        depth_layering_mode=str(hparams_dict.get('depth_layering_mode', 'hard_depth')),
        soft_diopter_eps=float(hparams_dict.get('soft_diopter_eps', 1e-8)),
        soft_diopter_bandwidth_scale=float(hparams_dict.get('soft_diopter_bandwidth_scale', 1.0)),
        sensor_measurement=str(hparams_dict.get('dodo_sensor_measurement', 'amplitude')),
    )

def _load_camera_from_checkpoint(ckpt_path):
    """Load DepthAwareDoDoForwardModel from a SnapshotDepthHS checkpoint."""
    from snapshotdepth_hs import SnapshotDepthHS
    model = SnapshotDepthHS.load_from_checkpoint(ckpt_path, map_location='cpu', strict=False)
    return model.camera, model.hparams

# ── default config ────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    'min_depth': 0.4,
    'max_depth': 2.0,
    'n_depths': 8,
    'dodo_depth_layers': 8,
    'dodo_use_second_doe': False,
    'dodo_doe_type': 'Zeros',
    'dodo_forward_norm': 'legacy_max',
    'dodo_sensing_mode': 'rgb',
    'measurement_channels': 3,
    'depth_layering_mode': 'hard_depth',
    'soft_diopter_eps': 1e-8,
    'soft_diopter_bandwidth_scale': 1.0,
    'dodo_sensor_measurement': 'amplitude',
    'hs_channels': 25,
    'start_wl': 420e-9,
    'end_wl': 660e-9,
}

# ── Diagnostics ───────────────────────────────────────────────────────────────

def diag_a_impulse_response(camera, num_wavelengths, H, W, depth_val,
                             output_dir, device='cpu', save_raw=False, eps=1e-10):
    """Single-wavelength impulse response visualization."""
    print("\n=== Diagnostic A: Single-Wavelength Impulse Response ===")
    os.makedirs(output_dir, exist_ok=True)

    cx, cy = H // 2, W // 2

    for c in range(num_wavelengths):
        x = torch.zeros(1, num_wavelengths, H, W, device=device)
        x[0, c, cy, cx] = 1.0
        d = torch.full((1, 1, H, W), depth_val, device=device)
        m = torch.ones(1, 1, H, W, device=device)

        with torch.no_grad():
            resp = camera(x, d, valid_mask=m)

        resp_cpu = resp.detach().cpu()

        # Save PNG
        _save_response_png(resp_cpu, c, output_dir)

        # Optionally save raw
        if save_raw:
            torch.save(resp_cpu, os.path.join(output_dir, f'raw_response_lambda_{c:03d}.pt'))

        # Cleanup
        del x, d, m, resp, resp_cpu
        if c % 5 == 0:
            if device != 'cpu':
                torch.cuda.empty_cache()
            print(f"  lambda {c:3d}/{num_wavelengths} done")

    print(f"  Saved {num_wavelengths} response PNGs to {output_dir}")


def _save_response_png(resp, c, output_dir):
    """Save a single-wavelength response as a normalized PNG."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # resp: (1, M, H, W)
    M = resp.shape[1]
    fig, axes = plt.subplots(1, max(M, 1), figsize=(4 * M, 4))
    if M == 1:
        axes = [axes]
    for ch in range(M):
        im = axes[ch].imshow(resp[0, ch].numpy(), cmap='inferno')
        plt.colorbar(im, ax=axes[ch])
        axes[ch].set_title(f'ch {ch}')
    fig.suptitle(f'lambda {c}')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'response_lambda_{c:03d}.png'), dpi=100)
    plt.close(fig)


def diag_b_wavelength_correlation(camera, num_wavelengths, H, W, depth_val,
                                    output_dir, device='cpu', eps=1e-10):
    """Wavelength response correlation matrix."""
    print("\n=== Diagnostic B: Wavelength Response Correlation ===")
    os.makedirs(output_dir, exist_ok=True)

    cx, cy = H // 2, W // 2

    # Collect responses
    responses = []
    for c in range(num_wavelengths):
        x = torch.zeros(1, num_wavelengths, H, W, device=device)
        x[0, c, cy, cx] = 1.0
        d = torch.full((1, 1, H, W), depth_val, device=device)
        m = torch.ones(1, 1, H, W, device=device)

        with torch.no_grad():
            resp = camera(x, d, valid_mask=m)

        r_flat = resp.detach().cpu().flatten()
        responses.append(r_flat)
        del x, d, m, resp

        if c % 5 == 0:
            if device != 'cpu':
                torch.cuda.empty_cache()

    R = torch.stack(responses)  # (C_lambda, M*H*W)
    R = R - R.mean(dim=1, keepdim=True)
    R = _safe_norm(R)
    corr = R @ R.T  # (C_lambda, C_lambda)

    # Save correlation matrix CSV
    corr_np = corr.numpy()
    with open(os.path.join(output_dir, 'corr_matrix.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([''] + [f'lambda_{i}' for i in range(num_wavelengths)])
        for i in range(num_wavelengths):
            writer.writerow([f'lambda_{i}'] + [f'{corr_np[i, j]:.6f}' for j in range(num_wavelengths)])

    # Save correlation matrix PNG
    _save_corr_png(corr_np, output_dir)

    # Adjacent correlation
    adj_corr = []
    for i in range(num_wavelengths - 1):
        adj_corr.append(float(corr_np[i, i + 1]))
    with open(os.path.join(output_dir, 'adjacent_corr.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['lambda_i', 'lambda_j', 'correlation'])
        for i in range(num_wavelengths - 1):
            writer.writerow([i, i + 1, f'{adj_corr[i]:.6f}'])

    stats = {
        'adjacent_corr_mean': float(np.mean(adj_corr)),
        'adjacent_corr_std': float(np.std(adj_corr)),
        'adjacent_corr_min': float(np.min(adj_corr)),
        'adjacent_corr_max': float(np.max(adj_corr)),
    }
    print(f"  adjacent_corr: mean={stats['adjacent_corr_mean']:.4f}, "
          f"std={stats['adjacent_corr_std']:.4f}, "
          f"min={stats['adjacent_corr_min']:.4f}, max={stats['adjacent_corr_max']:.4f}")

    del responses, R, corr, corr_np
    return stats


def _save_corr_png(corr_np, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr_np, cmap='RdYlBu_r', vmin=-1, vmax=1, aspect='auto')
    plt.colorbar(im, ax=ax)
    ax.set_xlabel('lambda index')
    ax.set_ylabel('lambda index')
    ax.set_title('Wavelength Response Correlation')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'corr_matrix.png'), dpi=120)
    plt.close(fig)


def diag_c_per_wavelength_energy(camera, num_wavelengths, H, W, depth_val,
                                   output_dir, device='cpu'):
    """Per-wavelength sensor energy."""
    print("\n=== Diagnostic C: Per-Wavelength Sensor Energy ===")
    os.makedirs(output_dir, exist_ok=True)

    cx, cy = H // 2, W // 2
    energies = []

    for c in range(num_wavelengths):
        x = torch.zeros(1, num_wavelengths, H, W, device=device)
        x[0, c, cy, cx] = 1.0
        d = torch.full((1, 1, H, W), depth_val, device=device)
        m = torch.ones(1, 1, H, W, device=device)

        with torch.no_grad():
            resp = camera(x, d, valid_mask=m)

        e = float(resp.detach().cpu().clamp_min(0).sum().item())
        energies.append(e)
        del x, d, m, resp

        if c % 5 == 0 and device != 'cpu':
            torch.cuda.empty_cache()

    # Save CSV
    with open(os.path.join(output_dir, 'energy_by_lambda.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['lambda', 'energy'])
        for i, e in enumerate(energies):
            writer.writerow([i, f'{e:.6e}'])

    # Save PNG
    _save_energy_png(energies, output_dir)

    stats = {
        'energy_mean': float(np.mean(energies)),
        'energy_std': float(np.std(energies)),
        'energy_min': float(np.min(energies)),
        'energy_max': float(np.max(energies)),
    }
    print(f"  energy: mean={stats['energy_mean']:.4e}, std={stats['energy_std']:.4e}, "
          f"min={stats['energy_min']:.4e}, max={stats['energy_max']:.4e}")
    return stats


def _save_energy_png(energies, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(len(energies)), energies, 'b-o', markersize=4)
    ax.set_xlabel('lambda index')
    ax.set_ylabel('measurement energy')
    ax.set_title('Per-Wavelength Sensor Energy')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'energy_by_lambda.png'), dpi=120)
    plt.close(fig)


def diag_d_lowres_matrix(camera, num_wavelengths, small_size, depth_val,
                          output_dir, device='cpu', eps=1e-10):
    """Low-resolution measurement matrix SVD analysis.

    Uses stride sampling over 128x128 (camera's fixed spatial size) to construct a
    manageable measurement matrix. small_matrix_size controls the spatial stride.
    """
    print(f"\n=== Diagnostic D: Low-Res Matrix Analysis (stride={small_size}) ===")
    os.makedirs(output_dir, exist_ok=True)

    # Camera requires 128x128, use stride to subsample
    full_size = 128
    stride = max(1, int(small_size))
    positions = list(range(0, full_size, stride))
    n_positions = len(positions) ** 2

    M = None
    columns = []

    # Use only a subset of wavelengths + positions to keep matrix size reasonable
    lambda_stride = max(1, num_wavelengths // 8)  # sample ~8 wavelengths
    sampled_lambdas = list(range(0, num_wavelengths, lambda_stride))

    for c in sampled_lambdas:
        for py in positions:
            for px in positions:
                x = torch.zeros(1, num_wavelengths, full_size, full_size, device=device)
                x[0, c, py, px] = 1.0
                d = torch.full((1, 1, full_size, full_size), depth_val, device=device)
                m = torch.ones(1, 1, full_size, full_size, device=device)

                with torch.no_grad():
                    resp = camera(x, d, valid_mask=m)

                col = resp.detach().cpu().flatten()
                if M is None:
                    M = col.numel()
                columns.append(col)
                del x, d, m, resp

        if device != 'cpu':
            torch.cuda.empty_cache()
        print(f"  lambda {c:3d}/{num_wavelengths} done ({len(columns)} columns)")

    print(f"  Total columns: {len(columns)}, M={M}")

    A = torch.stack(columns, dim=1)  # (M*H*W, C_lambda * n_positions)
    A = A.to(torch.float64)

    # SVD
    s = torch.linalg.svdvals(A)
    s_np = s.cpu().numpy()
    del A

    # Effective rank
    p = s / (s.sum() + eps)
    p_safe = p.clamp_min(eps)
    effective_rank = float(torch.exp(-(p * torch.log(p_safe)).sum()).item())

    # Condition number
    condition_number = float((s.max() / (s[-1].clamp_min(eps))).item())

    # Mutual coherence
    cols_norm = _safe_norm(torch.stack(columns))
    G = cols_norm @ cols_norm.T
    n = G.shape[0]
    off_diag = []
    for i in range(n):
        for j in range(n):
            if i != j:
                off_diag.append(float(G[i, j].abs().item()))
    mutual_coherence = float(max(off_diag)) if off_diag else 0.0
    del columns, cols_norm, G

    # Save singular values CSV/PNG
    with open(os.path.join(output_dir, 'singular_values.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['index', 'singular_value'])
        for i, sv in enumerate(s_np):
            writer.writerow([i, f'{sv:.6e}'])

    _save_sv_png(s_np, output_dir)

    stats = {
        'effective_rank': effective_rank,
        'condition_number': condition_number,
        'mutual_coherence': mutual_coherence,
        'num_sampled_positions': n_positions,
        'num_wavelengths': num_wavelengths,
        'total_columns': n_positions * num_wavelengths,
    }
    print(f"  effective_rank={effective_rank:.4f}, condition_number={condition_number:.2e}, "
          f"mutual_coherence={mutual_coherence:.4f}, num_positions={n_positions}")
    return stats


def _save_sv_png(s_np, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.semilogy(range(len(s_np)), s_np, 'b-', alpha=0.7)
    ax.set_xlabel('index')
    ax.set_ylabel('singular value')
    ax.set_title('Singular Values of Measurement Matrix')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'singular_values.png'), dpi=120)
    plt.close(fig)


# ── helpers for config ──────────────────────────────────────────────────────

def _compute_wavelengths(start_wl, end_wl, num):
    return np.linspace(start_wl, end_wl, num).tolist()


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Diagnose DoDo measurement spectral capacity')
    parser.add_argument('--config', type=str, default='',
                        help='YAML config path (not required; uses defaults if omitted)')
    parser.add_argument('--checkpoint', type=str, default='none',
                        help='Checkpoint path, or "none" for fresh init')
    parser.add_argument('--depth', type=str, default='z_mid',
                        help='Depth value: z_min, z_max, z_mid, or float')
    parser.add_argument('--diagnostic_size', type=int, default=128)
    parser.add_argument('--small_matrix_size', type=int, default=16)
    parser.add_argument('--output_dir', type=str, default='outputs/diagnosis/init')
    parser.add_argument('--save_raw_responses', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print('[WARN] CUDA not available, using CPU')

    # Resolve config
    hparams = dict(DEFAULT_CONFIG)
    if args.config:
        # Minimal YAML loading (only flat keys expected)
        try:
            import yaml
            with open(args.config, 'r') as f:
                cfg = yaml.safe_load(f) or {}
            hparams.update(cfg)
        except Exception as e:
            print(f'[WARN] Could not load config {args.config}: {e}')

    # Build/load camera
    if args.checkpoint.lower() != 'none' and os.path.exists(args.checkpoint):
        print(f'Loading camera from checkpoint: {args.checkpoint}')
        camera, loaded_hp = _load_camera_from_checkpoint(args.checkpoint)
        # Merge any new params not in checkpoint
        for k, v in hparams.items():
            if not hasattr(loaded_hp, k):
                setattr(loaded_hp, k, v)
        hparams_used = {k: getattr(loaded_hp, k, v) for k, v in hparams.items()}
        for k in ['min_depth', 'max_depth', 'n_depths', 'dodo_doe_type',
                   'dodo_forward_norm', 'dodo_sensing_mode', 'measurement_channels',
                   'depth_layering_mode', 'dodo_sensor_measurement']:
            if hasattr(loaded_hp, k):
                hparams_used[k] = getattr(loaded_hp, k)
    else:
        print('Building fresh camera from config (no checkpoint)')
        camera = _build_camera_from_hparams(hparams)
        hparams_used = dict(hparams)

    camera = camera.to(device)
    camera.eval()

    depth_val = _resolve_depth_value(
        args.depth,
        z_min=float(hparams_used.get('min_depth', 0.4)),
        z_max=float(hparams_used.get('max_depth', 2.0)),
    )

    num_wavelengths = int(hparams_used.get('hs_channels', 25))
    wavelengths = _compute_wavelengths(
        float(hparams_used.get('start_wl', 420e-9)),
        float(hparams_used.get('end_wl', 660e-9)),
        num_wavelengths,
    )
    measurement_channels = int(hparams_used.get('measurement_channels', 3))

    print(f'  depth={depth_val:.3f}m, diagnostic_size={args.diagnostic_size}, '
          f'small_matrix_size={args.small_matrix_size}')
    print(f'  num_wavelengths={num_wavelengths}, measurement_channels={measurement_channels}')
    print(f'  depth_layering={hparams_used.get("depth_layering_mode", "hard_depth")}, '
          f'sensor_measurement={hparams_used.get("dodo_sensor_measurement", "amplitude")}')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Diagnostic A
    diag_a_impulse_response(
        camera, num_wavelengths, args.diagnostic_size, args.diagnostic_size,
        depth_val, str(output_dir), device=device,
        save_raw=args.save_raw_responses,
    )

    # Diagnostic B
    stats_b = diag_b_wavelength_correlation(
        camera, num_wavelengths, args.diagnostic_size, args.diagnostic_size,
        depth_val, str(output_dir), device=device,
    )

    # Diagnostic C
    stats_c = diag_c_per_wavelength_energy(
        camera, num_wavelengths, args.diagnostic_size, args.diagnostic_size,
        depth_val, str(output_dir), device=device,
    )

    # Diagnostic D
    stats_d = diag_d_lowres_matrix(
        camera, num_wavelengths, args.small_matrix_size,
        depth_val, str(output_dir), device=device,
    )

    # Summary JSON
    summary = {
        'config': args.config or 'default',
        'checkpoint': args.checkpoint,
        'depth': float(depth_val),
        'diagnostic_size': args.diagnostic_size,
        'small_matrix_size': args.small_matrix_size,
        'depth_layering_mode': str(hparams_used.get('depth_layering_mode', 'hard_depth')),
        'measurement_norm_mode': str(hparams_used.get('dodo_forward_norm', 'legacy_max')),
        'measurement_channels': measurement_channels,
        'num_wavelengths': num_wavelengths,
        'wavelengths': wavelengths,
        'doe_type': str(hparams_used.get('dodo_doe_type', 'Zeros')),
        'doe2_enabled': bool(hparams_used.get('dodo_use_second_doe', False)),
        'seed': args.seed,
        'save_raw_responses': args.save_raw_responses,
        'temporary_files_cleaned': True,
        **{f'diag_b_{k}': v for k, v in stats_b.items()},
        **{f'diag_c_{k}': v for k, v in stats_c.items()},
        **{f'diag_d_{k}': v for k, v in stats_d.items()},
    }

    summary_path = output_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f'\nSummary saved: {summary_path}')

    # Cleanup temporary files
    _cleanup_temp_files(str(output_dir))

    # Save command.txt
    cmd_path = output_dir / 'command.txt'
    with open(cmd_path, 'w') as f:
        f.write(' '.join(sys.argv) + '\n')

    print('Done.')


if __name__ == '__main__':
    main()
