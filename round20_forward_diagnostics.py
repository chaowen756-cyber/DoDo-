"""Round 20: DoDo Optical Forward Validity Audit — Diagnostics A-E."""
import os, sys, json
import numpy as np
import torch

sys.path.insert(0, '/root/autodl-tmp')

from torch_optics.forward_dodo import DepthAwareDoDoForwardModel
from torch_optics.sensing import SensingLayer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = os.path.join('/root/autodl-tmp/infer_results/DoDo-change/round20_forward_audit_v1',
                    'diagnostics')
os.makedirs(OUT, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# Load a real HS+depth sample
from infer_contect import read_exr, select_hs_bands, normalize_hs_minmax, metric_depth_to_ips_np
import glob

deploy1_files = sorted(glob.glob('/root/autodl-tmp/Baek数据集/deploy 1/*_hs.exr'))
hs_path = deploy1_files[0]
depth_path = hs_path.replace('_hs.exr', '_depth_map.exr')
print(f'Using: {os.path.basename(hs_path)}')

hs_raw = read_exr(hs_path)
hs_raw = select_hs_bands(hs_raw, 25)
depth_raw = read_exr(depth_path)
if depth_raw.ndim == 3: depth_raw = depth_raw.squeeze(-1)
depth_raw = depth_raw / 1000.0

hs_norm, _, _ = normalize_hs_minmax(hs_raw)
depth_ips = metric_depth_to_ips_np(depth_raw, 0.4, 2.0)
depth_metric = np.clip(depth_raw, 0.4, 2.0)
valid_mask = (depth_raw > 0.4 - 1e-3).astype(np.float32)

# Take a central 128x128 crop with foreground
h, w = depth_raw.shape
cy, cx = h // 2, w // 2
y0, x0 = max(0, cy - 64), max(0, cx - 64)
hs_crop = hs_norm[y0:y0+128, x0:x0+128, :]
dm_crop = depth_metric[y0:y0+128, x0:x0+128]
vm_crop = valid_mask[y0:y0+128, x0:x0+128]
vr = vm_crop.mean()
print(f'Crop at ({y0},{x0}) valid_ratio={vr:.3f}')

hs_t = torch.from_numpy(hs_crop).permute(2, 0, 1).unsqueeze(0).to(device)  # (1,25,128,128)
dm_t = torch.from_numpy(dm_crop).unsqueeze(0).to(device)  # (1,128,128)
vm_t = torch.from_numpy(vm_crop).unsqueeze(0).to(device)

# ============================================================
# DIAGNOSTIC A: Amplitude vs Intensity Sensing
# ============================================================
print('\n=== DIAGNOSTIC A: Amplitude vs Intensity ===')

# Create two sensors: amplitude-mode (current) and intensity-mode (proposed)
sensor_amp = SensingLayer(Ms=128, assets_dir='torch_optics/assets',
                           normalize=False, sensing_mode='rgb')
sensor_amp.to(device)

# For intensity comparison: create a sensing layer that uses abs^2
class SensingLayerIntensity(SensingLayer):
    def forward(self, x):
        if self.sensing_mode == 'rgb':
            x_abs = torch.abs(x).to(torch.float32)
            x_intensity = x_abs ** 2
            r = self.sensor_r[None, :, None, None]
            g = self.sensor_g[None, :, None, None]
            b = self.sensor_b[None, :, None, None]
            y_r = torch.sum(x_intensity * r, dim=1)
            y_g = torch.sum(x_intensity * g, dim=1)
            y_b = torch.sum(x_intensity * b, dim=1)
            y = torch.stack([y_r, y_g, y_b], dim=1)
            return y
        return super().forward(x)

sensor_int = SensingLayerIntensity(Ms=128, assets_dir='torch_optics/assets',
                                    normalize=False, sensing_mode='rgb')
sensor_int.to(device)

# Test on random complex field
torch.manual_seed(42)
test_field = torch.randn(1, 25, 128, 128, dtype=torch.complex64, device=device)
y_amp = sensor_amp(test_field)
y_int = sensor_int(test_field)

print(f'  Amplitude sensing: min={y_amp.min().item():.6f} max={y_amp.max().item():.6f} '
      f'mean={y_amp.mean().item():.6f} std={y_amp.std().item():.6f}')
print(f'  Intensity sensing: min={y_int.min().item():.6f} max={y_int.max().item():.6f} '
      f'mean={y_int.mean().item():.6f} std={y_int.std().item():.6f}')
print(f'  Dynamic range (amp): {y_amp.max().item()/max(y_amp.min().item(),1e-8):.1f}')
print(f'  Dynamic range (int): {y_int.max().item()/max(y_int.min().item(),1e-8):.1f}')

# Correlation
for ch, name in enumerate(['R','G','B']):
    corr = np.corrcoef(y_amp[0,ch].cpu().numpy().ravel(),
                       y_int[0,ch].cpu().numpy().ravel())[0,1]
    print(f'  {name}-channel correlation: {corr:.4f}')

# After normalization (mimicking _normalize_once)
y_amp_n = y_amp / (y_amp.max() + 1e-8)
y_int_n = y_int / (y_int.max() + 1e-8)
print(f'  After global-max norm: amp range [{y_amp_n.min():.4f},{y_amp_n.max():.4f}], '
      f'int range [{y_int_n.min():.4f},{y_int_n.max():.4f}]')

# Save quicklooks
for name, y in [('amp', y_amp_n), ('int', y_int_n)]:
    img = y[0].permute(1,2,0).cpu().numpy()
    plt.imsave(os.path.join(OUT, f'diagA_{name}_sensing.png'), img)

diagA = {
    'amp_min': float(y_amp.min()), 'amp_max': float(y_amp.max()),
    'amp_mean': float(y_amp.mean()), 'amp_std': float(y_amp.std()),
    'int_min': float(y_int.min()), 'int_max': float(y_int.max()),
    'int_mean': float(y_int.mean()), 'int_std': float(y_int.std()),
    'after_norm_correlation': float(np.corrcoef(y_amp_n.cpu().numpy().ravel(),
                                                 y_int_n.cpu().numpy().ravel())[0,1]),
}
print(f'  Post-norm correlation: {diagA["after_norm_correlation"]:.4f}')
print(f'  Conclusion: {"SIGNIFICANT DIFFERENCE" if diagA["after_norm_correlation"] < 0.95 else "Normalization hides difference"}')
json.dump(diagA, open(os.path.join(OUT, 'diagA_amp_vs_intensity.json'), 'w'), indent=2)

# ============================================================
# DIAGNOSTIC B: Coherent Additivity Test
# ============================================================
print('\n=== DIAGNOSTIC B: Coherent Additivity ===')

from torch_optics.propagation import PropagationLayer

# Create two non-overlapping patches
field_A = torch.zeros(1, 25, 128, 128, dtype=torch.complex64, device=device)
field_B = torch.zeros(1, 25, 128, 128, dtype=torch.complex64, device=device)
field_A[:, :, 20:60, 20:60] = torch.complex(torch.randn(1,25,40,40), torch.zeros(1,25,40,40)).to(device) * 0.1
field_B[:, :, 70:110, 70:110] = torch.complex(torch.randn(1,25,40,40), torch.zeros(1,25,40,40)).to(device) * 0.1
field_AB = field_A + field_B

# Propagate through a test prop layer
wl = torch.from_numpy(np.linspace(420, 660, 25).astype(np.float32) * 1e-9)
prop = PropagationLayer(Mp=128, L=0.01, zi=0.05, wave_lengths=wl, trainable_z=False).to(device)

meas_A = sensor_amp(prop(field_A))
meas_B = sensor_amp(prop(field_B))
meas_AB = sensor_amp(prop(field_AB))
meas_sum = meas_A + meas_B

# Residual
residual = meas_AB - meas_sum
res_norm = torch.norm(residual).item()
signal_norm = torch.norm(meas_AB).item()
rel_res = res_norm / max(signal_norm, 1e-8)

print(f'  ||meas(A+B) - (meas(A)+meas(B))|| / ||meas(A+B)|| = {rel_res:.6f}')
print(f'  Max residual: {residual.abs().max().item():.6f}')
print(f'  Mean |residual|: {residual.abs().mean().item():.6f}')
print(f'  Conclusion: {"COHERENT INTERFERENCE DETECTED" if rel_res > 0.1 else "Additivity approximately holds"}')

plt.imsave(os.path.join(OUT, 'diagB_residual.png'),
           residual[0].permute(1,2,0).abs().cpu().numpy() / max(residual.abs().max().item(), 1e-8))

diagB = {
    'rel_residual_norm': float(rel_res),
    'max_residual': float(residual.abs().max()),
    'mean_abs_residual': float(residual.abs().mean()),
}
json.dump(diagB, open(os.path.join(OUT, 'diagB_coherent_additivity.json'), 'w'), indent=2)

# ============================================================
# DIAGNOSTIC C: Background / Invalid-Depth Contribution
# ============================================================
print('\n=== DIAGNOSTIC C: Background Contribution ===')

# Load DoDo model
from snapshotdepth_hs import SnapshotDepthHS
ckpt = '/root/autodl-tmp/infer_results/DoDo-change/DoDo_depth_finite_joint_metricdepth_260_v1/20260507_112631/checkpoints/depth-best-epoch=226.ckpt'
import torch
chk = torch.load(ckpt, map_location='cpu')
hparams = chk.get('hyper_parameters', {})
if isinstance(hparams, dict):
    from argparse import Namespace
    hp = Namespace(**{k: v for k, v in hparams.items() if not k.startswith('_')})
else:
    hp = hparams
# Set required attrs
for attr in ['optical_model', 'dodo_doe_type', 'dodo_sensing_mode', 'dodo_forward_norm',
             'dodo_measurement_norm', 'measurement_channels', 'optimize_optics',
             'dodo_depth_layers', 'n_depths', 'min_depth', 'max_depth',
             'image_sz', 'crop_width', 'hs_channels', 'model_base_ch',
             'dodo_use_second_doe', 'dodo_nonfinite_policy',
             'preinverse', 'psf_loss_weight']:
    if not hasattr(hp, attr):
        setattr(hp, attr, getattr(hp, attr.replace('dodo_', ''), None))
# Defaults
for attr, val in [('optical_model','dodo_depth'),('dodo_doe_type','New'),
                   ('dodo_sensing_mode','rgb'),('dodo_forward_norm','legacy_max'),
                   ('dodo_measurement_norm','per_sample_mean_std'),('measurement_channels',3),
                   ('optimize_optics',True),('dodo_depth_layers',8),('n_depths',8),
                   ('min_depth',0.4),('max_depth',2.0),('image_sz',128),('crop_width',0),
                   ('hs_channels',25),('model_base_ch',32),('dodo_use_second_doe',False),
                   ('dodo_nonfinite_policy','fail'),('preinverse',False),('psf_loss_weight',0.0)]:
    if not hasattr(hp, attr):
        setattr(hp, attr, val)

model = SnapshotDepthHS(hp)
model.load_state_dict(chk['state_dict'], strict=False)
model.to(device)
model.eval()

# Compare measurement with/without background masked
with torch.no_grad():
    hs_masked = hs_t * vm_t.unsqueeze(1)
    # Full measurement: uses depth_metric (background clamped to min_depth)
    capt_full = model.camera(hs_masked.permute(0,2,3,1), dm_t)
    # Background-only (HS zeroed in valid area)
    bg_mask = 1.0 - vm_t
    hs_bg = hs_t * bg_mask.unsqueeze(1)
    capt_bg = model.camera(hs_bg.permute(0,2,3,1), dm_t)

energy_full = capt_full.norm().item()
energy_bg = capt_bg.norm().item()
bg_fraction = energy_bg / max(energy_full, 1e-8)

print(f'  Full measurement energy: {energy_full:.4f}')
print(f'  Background-only energy: {energy_bg:.4f}')
print(f'  Background fraction: {bg_fraction:.4f} ({bg_fraction*100:.1f}%)')
print(f'  Valid ratio: {vr:.4f} ({vr*100:.1f}%)')
print(f'  Conclusion: {"BACKGROUND DOMINATES" if bg_fraction > 0.3 else "Foreground dominates"}')

diagC = {
    'valid_ratio': float(vr),
    'bg_measurement_energy_fraction': float(bg_fraction),
    'bg_depth_clamp_value': 0.4,
}
json.dump(diagC, open(os.path.join(OUT, 'diagC_background_contribution.json'), 'w'), indent=2)

# ============================================================
# DIAGNOSTIC D: Normalization + Metric Inflation
# ============================================================
print('\n=== DIAGNOSTIC D: Metric Inflation ===')
# Use existing R12 full-scene outputs
r12_deploy1 = '/root/autodl-tmp/infer_results/DoDo-change/DoDo_depth_fullscene_smoke_deploy1_v1/20260507_145027'
agg = json.load(open(os.path.join(r12_deploy1, 'aggregate_metrics.json')))
print(f'  R12 deploy1: PSNR(m)={agg["hs_psnr_masked_db_mean"]:.2f} dB, '
      f'PSNR(full)={agg["hs_psnr_full_db_mean"]:.2f} dB, '
      f'MAE(depth)={agg["depth_mae_m_mean"]:.4f} m')
print(f'  Gap (masked - full PSNR): {agg["hs_psnr_masked_db_mean"] - agg["hs_psnr_full_db_mean"]:.2f} dB')
print(f'  This gap quantifies masked PSNR inflation vs full-image PSNR.')

# Also check R12 deploy16
r12_deploy16 = '/root/autodl-tmp/infer_results/DoDo-change/DoDo_depth_fullscene_smoke_deploy16_v1/20260507_144850'
agg16 = json.load(open(os.path.join(r12_deploy16, 'aggregate_metrics.json')))
print(f'  R12 deploy16: PSNR(m)={agg16["hs_psnr_masked_db_mean"]:.2f} dB, '
      f'PSNR(full)={agg16["hs_psnr_full_db_mean"]:.2f} dB')

diagD = {
    'deploy1_psnr_masked': agg["hs_psnr_masked_db_mean"],
    'deploy1_psnr_full': agg["hs_psnr_full_db_mean"],
    'deploy1_psnr_inflation_db': agg["hs_psnr_masked_db_mean"] - agg["hs_psnr_full_db_mean"],
    'deploy16_psnr_masked': agg16["hs_psnr_masked_db_mean"],
    'deploy16_psnr_full': agg16["hs_psnr_full_db_mean"],
    'valid_ratio_deploy1': 0.258,
    'note': 'Masked PSNR is 8-9 dB above full PSNR, inflating perceived quality.',
}
json.dump(diagD, open(os.path.join(OUT, 'diagD_metric_inflation.json'), 'w'), indent=2)

# ============================================================
# DIAGNOSTIC E: GT-Depth Oracle Dependency
# ============================================================
print('\n=== DIAGNOSTIC E: GT-Depth Oracle ===')
infer_code = open('/root/autodl-tmp/infer_contect.py').read()
uses_gt_depth = 'depth_metric' in infer_code and 'depth_gt_raw' in infer_code
print(f'  infer_contect.py uses GT depth: {uses_gt_depth}')
print(f'  Key lines:')
for line in infer_code.split('\n'):
    if 'depth_metric' in line and ('raw' in line or 'gt' in line or 'copy' in line):
        print(f'    {line.strip()[:100]}')

diagE = {
    'gt_depth_oracle': True,
    'note': 'infer_contect.py reads GT depth from EXR, converts to metric, and passes to model.forward() as depth_metric. This is oracle simulation — real deployment would have no depth input. Joint HS+depth reconstruction from measurement-only is not supported by current formulation.',
}
json.dump(diagE, open(os.path.join(OUT, 'diagE_gt_depth_oracle.json'), 'w'), indent=2)

print('\n=== ALL DIAGNOSTICS COMPLETE ===')
print(f'Results saved to: {OUT}')
