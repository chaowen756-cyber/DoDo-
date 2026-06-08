"""Round 20 Diagnostics C-E only (lightweight, no full model load for D already done)."""
import os, json
import numpy as np
import torch
import sys; sys.path.insert(0, '/root/autodl-tmp')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = '/root/autodl-tmp/infer_results/DoDo-change/round20_forward_audit_v1/diagnostics'
device = torch.device('cuda')

# ============================================================
# DIAGNOSTIC C: Background Contribution
# ============================================================
print('\n=== DIAGNOSTIC C: Background Contribution ===')
from torch_optics.forward_dodo import DepthAwareDoDoForwardModel
from infer_contect import read_exr, select_hs_bands, normalize_hs_minmax
import glob

# Load scene
deploy1_files = sorted(glob.glob('/root/autodl-tmp/Baek数据集/deploy 1/*_hs.exr'))
hs_path = deploy1_files[0]; depth_path = hs_path.replace('_hs.exr', '_depth_map.exr')
hs_raw = read_exr(hs_path); hs_raw = select_hs_bands(hs_raw, 25)
depth_raw = read_exr(depth_path)
if depth_raw.ndim == 3: depth_raw = depth_raw.squeeze(-1)
depth_raw = depth_raw / 1000.0
hs_norm, _, _ = normalize_hs_minmax(hs_raw)
valid_mask = (depth_raw > 0.4 - 1e-3).astype(np.float32)
depth_metric = np.clip(depth_raw, 0.4, 2.0)

# Central 128x128 crop
h, w = depth_raw.shape
y0, x0 = max(0, h//2 - 64), max(0, w//2 - 64)
hs_crop = hs_norm[y0:y0+128, x0:x0+128, :]
dm_crop = depth_metric[y0:y0+128, x0:x0+128]
vm_crop = valid_mask[y0:y0+128, x0:x0+128]
vr = vm_crop.mean()

hs_t = torch.from_numpy(hs_crop).permute(2,0,1).unsqueeze(0).to(device)
dm_t = torch.from_numpy(dm_crop).unsqueeze(0).to(device)
vm_t = torch.from_numpy(vm_crop).unsqueeze(0).to(device)

# Build camera directly (skip full model)
camera = DepthAwareDoDoForwardModel(depth_min=0.4, depth_max=2.0, num_depth_layers=8,
    use_second_doe=False, doe_type_a='New', train_c=False,
    input_format='nchw', output_format='nchw',
    measurement_norm_mode='legacy_max', sensing_mode='rgb').to(device)
# Load pretrained weights into camera
ckpt = torch.load('/root/autodl-tmp/infer_results/DoDo-change/DoDo_depth_finite_joint_metricdepth_260_v1/20260507_112631/checkpoints/depth-best-epoch=226.ckpt',
                  map_location='cpu', weights_only=True)
cam_state = {k.replace('camera.', ''): v for k, v in ckpt['state_dict'].items() if k.startswith('camera.')}
camera.load_state_dict(cam_state, strict=False)
camera.eval()

with torch.no_grad():
    # Full measurement
    hs_in = hs_t * vm_t.unsqueeze(1)
    capt_full = camera(hs_in.permute(0,2,3,1), dm_t)
    # Background-only
    bg_mask = 1.0 - vm_t
    hs_bg = hs_t * bg_mask.unsqueeze(1)
    capt_bg = camera(hs_bg.permute(0,2,3,1), dm_t)
    # Zero-background measurement (only foreground)
    capt_fg = capt_full - capt_bg  # approximation

energy_full = capt_full.norm().item()
energy_bg = capt_bg.norm().item()
bg_fraction = energy_bg / max(energy_full, 1e-8)

print(f'  Valid ratio: {vr:.4f} ({vr*100:.1f}%)')
print(f'  Full measurement energy: {energy_full:.4f}')
print(f'  Background-only energy: {energy_bg:.4f}')
print(f'  Background fraction: {bg_fraction:.4f} ({bg_fraction*100:.1f}%)')
print(f'  Conclusion: {"BACKGROUND DOMINATES measurement" if bg_fraction > 0.3 else "Foreground dominates measurement"}')

diagC = {
    'valid_ratio': float(vr),
    'bg_measurement_energy_fraction': float(bg_fraction),
    'bg_depth_clamp_value_m': 0.4,
    'note': 'Background pixels clamped to min_depth=0.4m and still contribute to optical measurement.',
}
json.dump(diagC, open(os.path.join(OUT, 'diagC_background_contribution.json'), 'w'), indent=2)

# ============================================================
# DIAGNOSTIC D: Metric Inflation
# ============================================================
print('\n=== DIAGNOSTIC D: Metric Inflation ===')
r12_d1 = '/root/autodl-tmp/infer_results/DoDo-change/DoDo_depth_fullscene_smoke_deploy1_v1/20260507_145027'
agg = json.load(open(os.path.join(r12_d1, 'aggregate_metrics.json')))
r12_d16 = '/root/autodl-tmp/infer_results/DoDo-change/DoDo_depth_fullscene_smoke_deploy16_v1/20260507_144850'
agg16 = json.load(open(os.path.join(r12_d16, 'aggregate_metrics.json')))

print(f'  deploy1 masked PSNR: {agg["hs_psnr_masked_db_mean"]:.2f} dB')
print(f'  deploy1 full PSNR:   {agg["hs_psnr_full_db_mean"]:.2f} dB')
print(f'  Inflation gap:        {agg["hs_psnr_masked_db_mean"] - agg["hs_psnr_full_db_mean"]:.2f} dB')
print(f'  Valid pixel ratio:    {0.258:.3f} (only {0.258*100:.1f}% of image)')
print(f'  deploy16 masked PSNR: {agg16["hs_psnr_masked_db_mean"]:.2f} dB')
print(f'  deploy16 full PSNR:   {agg16["hs_psnr_full_db_mean"]:.2f} dB')
print(f'  deploy16 gap:         {agg16["hs_psnr_masked_db_mean"] - agg16["hs_psnr_full_db_mean"]:.2f} dB')

diagD = {
    'deploy1_psnr_masked': agg["hs_psnr_masked_db_mean"],
    'deploy1_psnr_full': agg["hs_psnr_full_db_mean"],
    'deploy1_inflation_db': agg["hs_psnr_masked_db_mean"] - agg["hs_psnr_full_db_mean"],
    'deploy1_valid_ratio': 0.258,
    'deploy16_psnr_masked': agg16["hs_psnr_masked_db_mean"],
    'deploy16_psnr_full': agg16["hs_psnr_full_db_mean"],
    'deploy16_inflation_db': agg16["hs_psnr_masked_db_mean"] - agg16["hs_psnr_full_db_mean"],
    'note': 'Masked PSNR is 8-10 dB higher than full PSNR, computed over only 26-37% valid pixels.',
}
json.dump(diagD, open(os.path.join(OUT, 'diagD_metric_inflation.json'), 'w'), indent=2)

# ============================================================
# DIAGNOSTIC E: GT-Depth Oracle
# ============================================================
print('\n=== DIAGNOSTIC E: GT-Depth Oracle ===')
print('  GT depth dependency confirmed in infer_contect.py:')
print('    line 368: depth_metric_raw = depth_gt_raw.copy()')
print('    line 370: depth_metric_raw[valid_mask < 0.5] = min_depth')
print('    line 439: model(..., depth_metric=dm_patch, valid_mask=vm_patch)')
print('  This passes GT metric depth into the optical forward model.')
print('  Real deployment has NO depth input — measurement must be formed from scene only.')

diagE = {
    'gt_depth_oracle': True,
    'code_locations': [
        'infer_contect.py:368-371 (depth_metric from GT)',
        'infer_contect.py:439 (depth_metric passed to model)',
        'snapshotdepth_hs.py:726-735 (depth_metric used in optical forward)',
    ],
    'note': 'Current full-scene inference is oracle simulation, not measurement-only deployment.',
}
json.dump(diagE, open(os.path.join(OUT, 'diagE_gt_depth_oracle.json'), 'w'), indent=2)

print('\n=== DIAGNOSTICS C-E COMPLETE ===')
for f in ['diagC_background_contribution.json','diagD_metric_inflation.json','diagE_gt_depth_oracle.json']:
    path = os.path.join(OUT, f)
    if os.path.exists(path):
        d = json.load(open(path))
        print(f'  {f}: {json.dumps(d, indent=None)}')
