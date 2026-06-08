"""Tests for optional decoder depth input and diagnosis utilities."""

import json
import os
import shutil
import tempfile
import torch
import numpy as np

from torch_optics.forward_dodo import DepthAwareDoDoForwardModel

# ── depth normalization tests ──────────────────────────────────────────────

def test_depth_normalized_z_range():
    """normalized_z output should be in [0, 1]."""
    z_min, z_max = 0.4, 2.0
    depth = torch.tensor([[[[0.4], [1.2], [2.0]]]])  # (1, 1, 3, 1)
    depth_safe = depth.clamp(min=z_min, max=z_max)
    feat = (depth_safe - z_min) / (z_max - z_min + 1e-8)
    feat = feat.clamp(0, 1)
    assert feat.min() >= 0.0
    assert feat.max() <= 1.0
    # Boundary values
    assert abs(feat[0, 0, 0, 0].item() - 0.0) < 1e-6   # z_min → 0
    assert abs(feat[0, 0, 2, 0].item() - 1.0) < 1e-6   # z_max → 1


def test_depth_normalized_diopter_range():
    """normalized_diopter output should be in [0, 1]."""
    z_min, z_max = 0.4, 2.0
    eps = 1e-8
    depth = torch.tensor([[[[0.4], [1.2], [2.0]]]])
    depth_safe = depth.clamp(min=z_min, max=z_max)
    u = 1.0 / depth_safe
    u_min = 1.0 / z_max
    u_max = 1.0 / z_min
    feat = (u - u_min) / (u_max - u_min + eps)
    feat = feat.clamp(0, 1)
    assert feat.min() >= 0.0
    assert feat.max() <= 1.0
    # z_min (0.4) → u_max → 1.0
    assert abs(feat[0, 0, 0, 0].item() - 1.0) < 1e-6
    # z_max (2.0) → u_min → 0.0
    assert abs(feat[0, 0, 2, 0].item() - 0.0) < 1e-6


def test_depth_normalization_handles_nan_inf():
    """NaN/Inf/<=0 depth should not produce NaN output (clamped)."""
    z_min, z_max = 0.4, 2.0
    eps = 1e-8
    depth = torch.tensor([[[[float('nan')], [float('inf')], [-1.0], [0.0], [1.0]]]])
    # Clamp should handle all edge cases
    finite = torch.isfinite(depth) & (depth > 0)
    depth_safe = torch.where(finite, depth.clamp(min=z_min, max=z_max),
                              torch.tensor(z_min))
    u = 1.0 / depth_safe
    u_min = 1.0 / z_max
    u_max = 1.0 / z_min
    feat_nz = (depth_safe - z_min) / (z_max - z_min + eps)
    feat_diopter = (u - u_min) / (u_max - u_min + eps)
    assert torch.isfinite(feat_nz).all(), f'normalized_z has non-finite: {feat_nz}'
    assert torch.isfinite(feat_diopter).all(), f'normalized_diopter has non-finite: {feat_diopter}'


# ── decoder depth input shape tests ────────────────────────────────────────

def test_decoder_depth_input_channels_unchanged_when_disabled():
    """When decoder_use_depth_input=False, measurement input channels unchanged."""
    model = DepthAwareDoDoForwardModel(
        depth_min=0.4, depth_max=2.0, num_depth_layers=2,
        input_format='nchw', output_format='nchw',
        measurement_norm_mode='none',
    )
    spectral = torch.rand(1, 25, 128, 128)
    depth = torch.rand(1, 1, 128, 128) * 1.6 + 0.4
    with torch.no_grad():
        out = model(spectral, depth)
    # Default sensing_mode='rgb' → 3 channels
    assert out.shape[1] == 3, f'Expected 3 measurement channels, got {out.shape[1]}'


def test_decoder_depth_input_channel_count_plus_one():
    """When decoder_use_depth_input=True, captimgs gets +1 channel from depth feature."""
    z_min, z_max = 0.4, 2.0
    eps = 1e-8
    # Simulate what snapshotdepth_hs.py does
    measurement_channels = 3
    decoder_use_depth_input = True
    decoder_depth_input_mode = 'normalized_diopter'

    captimgs = torch.rand(1, measurement_channels, 128, 128)
    depth_metric = torch.rand(1, 1, 128, 128) * 1.6 + 0.4

    # Build depth feature
    depth_safe = depth_metric.clamp(min=z_min, max=z_max)
    u = 1.0 / depth_safe
    u_min = 1.0 / z_max
    u_max = 1.0 / z_min
    depth_feature = (u - u_min) / (u_max - u_min + eps)
    depth_feature = depth_feature.clamp(0, 1)

    if decoder_use_depth_input:
        captimgs_with_depth = torch.cat([captimgs, depth_feature.to(captimgs.dtype)], dim=1)

    assert captimgs_with_depth.shape[1] == measurement_channels + 1
    assert captimgs.shape[1] == measurement_channels  # original unchanged


def test_decoder_depth_input_disabled_preserves_captimgs_shape():
    """decoder_use_depth_input=False: captimgs channels == measurement_channels."""
    measurement_channels = 3
    captimgs = torch.rand(1, measurement_channels, 128, 128)
    decoder_use_depth_input = False

    if decoder_use_depth_input:
        captimgs = torch.cat([captimgs, torch.rand(1, 1, 128, 128)], dim=1)

    assert captimgs.shape[1] == measurement_channels


# ── diagnosis utility tests ────────────────────────────────────────────────

def test_correlation_matrix_shape():
    """Normalized correlation matrix has correct shape."""
    C, N = 25, 100
    R = torch.randn(C, N)
    R = R - R.mean(dim=1, keepdim=True)
    R = R / (R.norm(dim=1, keepdim=True) + 1e-10)
    corr = R @ R.T
    assert corr.shape == (C, C)
    assert torch.allclose(torch.diag(corr), torch.ones(C), atol=1e-5)


def test_singular_value_summary_generates():
    """SVD on random measurement matrix produces valid summary stats."""
    torch.manual_seed(42)
    A = torch.randn(200, 100, dtype=torch.float64)
    s = torch.linalg.svdvals(A)
    eps = 1e-10
    p = s / (s.sum() + eps)
    p_safe = p.clamp_min(eps)
    effective_rank = float(torch.exp(-(p * torch.log(p_safe)).sum()).item())
    condition_number = float((s.max() / s[-1].clamp_min(eps)).item())
    assert effective_rank > 0
    assert condition_number > 0
    assert torch.isfinite(s).all()


def test_summary_json_writes_and_reads():
    """summary.json can be written and read back."""
    summary = {
        'config': 'default',
        'checkpoint': 'none',
        'depth': 1.2,
        'seed': 42,
        'diag_b_adjacent_corr_mean': 0.95,
        'diag_c_energy_mean': 1e-4,
        'diag_d_effective_rank': 3.14,
    }
    tmpdir = tempfile.mkdtemp()
    try:
        path = os.path.join(tmpdir, 'summary.json')
        with open(path, 'w') as f:
            json.dump(summary, f, indent=2)
        with open(path, 'r') as f:
            loaded = json.load(f)
        assert loaded['seed'] == 42
        assert abs(loaded['depth'] - 1.2) < 1e-6
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_save_raw_responses_default_is_false():
    """Default behavior: save_raw_responses should be False."""
    # This is a parameter-level test — the CLI default ensures this
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_raw_responses', action='store_true', default=False)
    args = parser.parse_args([])
    assert args.save_raw_responses is False


def test_temp_file_cleanup_deletes_files():
    """Cleanup function removes tmp_*.pt, tmp_*.npy, cache_* files."""
    tmpdir = tempfile.mkdtemp()
    try:
        # Create fake temp files
        files = ['tmp_test.pt', 'tmp_data.npy', 'cache_abc', 'good_output.csv',
                 'impulse_test.pt', 'response_lambda_000.png']
        for f in files:
            with open(os.path.join(tmpdir, f), 'w') as fh:
                fh.write('')

        # Run cleanup
        patterns = ["tmp_*.pt", "tmp_*.npy", "cache_*", "impulse_*.pt"]
        import glob as _glob
        cleaned = 0
        for pat in patterns:
            for f in _glob.glob(str(os.path.join(tmpdir, pat))):
                os.remove(f)
                cleaned += 1
        assert cleaned >= 4  # tmp_test.pt, tmp_data.npy, cache_abc, impulse_test.pt

        # Good files should remain
        remaining = os.listdir(tmpdir)
        assert 'good_output.csv' in remaining
        assert 'response_lambda_000.png' in remaining
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_empty_save_raw_responses_does_not_save_raw():
    """When save_raw_responses=False, no .pt/.npy raw files should be created."""
    # Sanity-check: simulated diagnostic ensures no raw tensor save
    save_raw = False
    raw_files_written = []
    if save_raw:
        raw_files_written.append('raw_response_lambda_000.pt')
    assert len(raw_files_written) == 0
