import os
import re
import glob
import argparse
from typing import Optional, Tuple

import numpy as np
import torch
import OpenEXR
import Imath
import matplotlib.pyplot as plt
from tqdm import tqdm

from snapshotdepth_hs import SnapshotDepthHS
from util.helper import metric_to_ips, ips_to_metric


def read_exr(file_path: str) -> np.ndarray:
    if not OpenEXR.isOpenExrFile(file_path):
        raise IOError(f"Invalid EXR file: {file_path}")

    exr_file = OpenEXR.InputFile(file_path)
    header = exr_file.header()
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    channels_info = header['channels']
    channel_names = sorted(channels_info.keys())
    if not channel_names:
        raise ValueError(f"No channels found in EXR: {file_path}")

    first_channel_type = channels_info[channel_names[0]]
    if first_channel_type.type == Imath.PixelType(Imath.PixelType.FLOAT):
        dtype = np.float32
    elif first_channel_type.type == Imath.PixelType(Imath.PixelType.HALF):
        dtype = np.float16
    else:
        raise TypeError(f"Unsupported EXR pixel type: {first_channel_type.type}")

    all_channels_bytes = exr_file.channels(channel_names)
    np_channels = []
    for i in range(len(channel_names)):
        channel_data = np.frombuffer(all_channels_bytes[i], dtype=dtype).reshape(height, width)
        np_channels.append(channel_data)

    image_np = np.stack(np_channels, axis=-1)
    return image_np.astype(np.float32)


def normalize_hs_minmax(data: np.ndarray) -> Tuple[np.ndarray, float, float]:
    d_min = float(data.min())
    d_max = float(data.max())
    if d_max == d_min:
        return np.zeros_like(data), d_min, d_max
    norm = (data - d_min) / (d_max - d_min)
    return norm, d_min, d_max


def metric_depth_to_ips_np(depth_m: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
    depth_tensor = torch.from_numpy(depth_m).float()
    valid_mask = depth_tensor >= (min_depth - 1e-3)
    depth_safe = torch.where(valid_mask, depth_tensor, torch.tensor(min_depth, dtype=depth_tensor.dtype))
    ips_depth = metric_to_ips(depth_safe, min_depth, max_depth)
    return torch.clamp(ips_depth, 0.0, 1.0).numpy()


def ips_to_metric_np(depth_ips: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
    depth_tensor = torch.from_numpy(depth_ips).float().clamp(0.0, 1.0)
    return ips_to_metric(depth_tensor, min_depth, max_depth).numpy()


def calculate_psnr(img1: np.ndarray, img2: np.ndarray, data_range: float = 1.0) -> float:
    mse = float(np.mean((img1 - img2) ** 2))
    if mse == 0.0:
        return 100.0
    return float(10.0 * np.log10((data_range ** 2) / mse))


def calculate_masked_psnr(gt: np.ndarray, pred: np.ndarray, valid_mask: np.ndarray, data_range: float = 1.0) -> float:
    m = valid_mask > 0.5
    if m.sum() == 0:
        return float('nan')
    if gt.ndim == 3:
        gt_valid = gt[m]
        pred_valid = pred[m]
    else:
        gt_valid = gt[m]
        pred_valid = pred[m]
    mse = float(np.mean((gt_valid - pred_valid) ** 2))
    if mse == 0.0:
        return 100.0
    return float(10.0 * np.log10((data_range ** 2) / mse))


def calculate_masked_sam(gt: np.ndarray, pred: np.ndarray, valid_mask: np.ndarray, eps: float = 1e-8) -> float:
    m = valid_mask > 0.5
    if m.sum() == 0:
        return float('nan')
    if gt.ndim != 3 or pred.ndim != 3:
        raise ValueError('SAM expects hyperspectral cubes with shape HxWxC.')

    gt_vec = gt[m]
    pred_vec = pred[m]
    gt_norm = np.linalg.norm(gt_vec, axis=1)
    pred_norm = np.linalg.norm(pred_vec, axis=1)
    valid_spec = (gt_norm > eps) & (pred_norm > eps)
    if not np.any(valid_spec):
        return float('nan')

    gt_vec = gt_vec[valid_spec]
    pred_vec = pred_vec[valid_spec]
    gt_norm = gt_norm[valid_spec]
    pred_norm = pred_norm[valid_spec]

    cos_angle = np.sum(gt_vec * pred_vec, axis=1) / (gt_norm * pred_norm + eps)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angles = np.arccos(cos_angle)
    return float(np.mean(angles))


def calculate_masked_mse(gt: np.ndarray, pred: np.ndarray, valid_mask: np.ndarray) -> float:
    m = valid_mask > 0.5
    if m.sum() == 0:
        return float('nan')
    return float(np.mean((gt[m] - pred[m]) ** 2))


def calculate_masked_mae(gt: np.ndarray, pred: np.ndarray, valid_mask: np.ndarray) -> float:
    m = valid_mask > 0.5
    if m.sum() == 0:
        return float('nan')
    return float(np.mean(np.abs(gt[m] - pred[m])))


def compute_depth_histogram(
    depth_map: np.ndarray,
    valid_mask: np.ndarray,
    depth_min: float,
    depth_max: float,
    num_bins: int = 8,
):
    valid_depths = depth_map[valid_mask > 0.5]
    if valid_depths.size == 0:
        raise ValueError("No valid pixels found in depth map.")

    bin_edges = np.linspace(depth_min, depth_max, num_bins + 1)
    counts, _ = np.histogram(valid_depths, bins=bin_edges)
    total = counts.sum()
    percentages = counts / max(total, 1) * 100.0
    return bin_edges, percentages, counts


def pick_rgb_band_indices(num_channels: int, start_wl: float, end_wl: float) -> Tuple[int, int, int]:
    # Prefer wavelengths close to R/G/B anchors for pseudo-RGB visualization.
    target_wls = np.array([650e-9, 550e-9, 450e-9], dtype=np.float64)
    wl_grid = np.linspace(start_wl, end_wl, num_channels)
    indices = [int(np.argmin(np.abs(wl_grid - t))) for t in target_wls]
    return (indices[0], indices[1], indices[2])


def visualize_hyperspectral(
    hs_tensor_chw: torch.Tensor,
    save_path: str,
    bands: Tuple[int, int, int],
    p_min: Optional[float] = None,
    p_max: Optional[float] = None,
):
    hs_np = hs_tensor_chw.cpu().numpy().transpose(1, 2, 0)
    b = [min(int(x), hs_np.shape[2] - 1) for x in bands]
    rgb = hs_np[..., b]

    if p_min is None:
        p_min = float(np.percentile(rgb, 1))
    if p_max is None:
        p_max = float(np.percentile(rgb, 99))
    if p_max - p_min < 1e-6:
        p_max = p_min + 1.0

    rgb = np.clip((rgb - p_min) / (p_max - p_min), 0, 1)
    plt.imsave(save_path, rgb)
    return float(p_min), float(p_max)


def visualize_depth(depth_data: np.ndarray, save_path: str, vmin: Optional[float], vmax: Optional[float]):
    plt.imsave(save_path, depth_data, cmap='inferno', vmin=vmin, vmax=vmax)


def get_cosine_mask(h: int, w: int, device: torch.device) -> torch.Tensor:
    idx_h = torch.linspace(0, np.pi, h, device=device)
    idx_w = torch.linspace(0, np.pi, w, device=device)
    mask_h = torch.sin(idx_h) ** 2
    mask_w = torch.sin(idx_w) ** 2
    return mask_h.unsqueeze(1) * mask_w.unsqueeze(0) + 1e-8


def resolve_path(path_str: str, script_dir: str) -> str:
    if os.path.isabs(path_str):
        return path_str
    if os.path.exists(path_str):
        return os.path.abspath(path_str)
    return os.path.abspath(os.path.join(script_dir, path_str))


def choose_checkpoint(ckpt_path: str) -> str:
    if os.path.isfile(ckpt_path):
        return ckpt_path

    if not os.path.isdir(ckpt_path):
        raise FileNotFoundError(f"Checkpoint path not found: {ckpt_path}")

    ckpt_files = glob.glob(os.path.join(ckpt_path, '*.ckpt'))
    if not ckpt_files:
        raise FileNotFoundError(f"No .ckpt files found in: {ckpt_path}")

    val_loss_items = []
    for p in ckpt_files:
        m = re.search(r"val_loss=([0-9]*\.?[0-9]+)", os.path.basename(p))
        if m:
            val_loss_items.append((float(m.group(1)), p))

    if val_loss_items:
        val_loss_items.sort(key=lambda x: x[0])
        chosen = val_loss_items[0][1]
        print(f"Checkpoint selected by min val_loss: {chosen}")
        return chosen

    ckpt_files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    chosen = ckpt_files[0]
    print(f"Checkpoint selected by latest mtime: {chosen}")
    return chosen


def load_model(real_ckpt_path: str, device: torch.device) -> SnapshotDepthHS:
    checkpoint = torch.load(real_ckpt_path, map_location='cpu')

    hparams = None
    if 'hyper_parameters' in checkpoint:
        hparams_dict = checkpoint['hyper_parameters']
        if 'hparams' in hparams_dict and isinstance(hparams_dict['hparams'], (dict, argparse.Namespace)):
            hparams_obj = hparams_dict['hparams']
            if isinstance(hparams_obj, dict):
                hparams = argparse.Namespace(**hparams_obj)
            else:
                hparams = hparams_obj
        elif isinstance(hparams_dict, dict):
            hparams = argparse.Namespace(**hparams_dict)

    load_errors = []
    model = None
    for kwargs in ([{'hparams': hparams}] if hparams is not None else []) + [{}]:
        try:
            model = SnapshotDepthHS.load_from_checkpoint(real_ckpt_path, **kwargs)
            break
        except Exception as e:
            load_errors.append(str(e))

    if model is None:
        raise RuntimeError("Failed to load checkpoint. Errors:\n" + "\n".join(load_errors))

    expected_hs_channels = 25
    expected_start_wl = 420e-9
    expected_end_wl = 660e-9
    ckpt_hs_channels = int(getattr(model.hparams, 'hs_channels', expected_hs_channels))
    ckpt_start_wl = float(getattr(model.hparams, 'start_wl', expected_start_wl))
    ckpt_end_wl = float(getattr(model.hparams, 'end_wl', expected_end_wl))
    if (
        ckpt_hs_channels != expected_hs_channels
        or abs(ckpt_start_wl - expected_start_wl) > 1e-12
        or abs(ckpt_end_wl - expected_end_wl) > 1e-12
    ):
        raise ValueError(
            "Checkpoint spectral config mismatch: "
            f"got hs_channels={ckpt_hs_channels}, "
            f"start_wl={ckpt_start_wl * 1e9:.1f}nm, "
            f"end_wl={ckpt_end_wl * 1e9:.1f}nm. "
            "Please use a checkpoint trained with 25 bands from 420nm to 660nm."
        )

    model.eval()
    model.to(device)

    # Ensure deterministic inference: disable synthetic sensor noise during inference.
    if hasattr(model, 'hparams'):
        if hasattr(model.hparams, 'noise_sigma_min'):
            model.hparams.noise_sigma_min = 0.0
        if hasattr(model.hparams, 'noise_sigma_max'):
            model.hparams.noise_sigma_max = 0.0

    return model


def select_hs_bands(hs_cube: np.ndarray, expected_channels: int) -> np.ndarray:
    if hs_cube.ndim != 3:
        raise ValueError(f"Expected hyperspectral cube with 3 dims, got shape={hs_cube.shape}")
    if hs_cube.shape[2] < expected_channels:
        raise ValueError(
            f"Hyperspectral channel count is too small: got={hs_cube.shape[2]}, expected={expected_channels}"
        )
    # 原始 EXR 通道按 420nm -> 700nm 递增排列，保留前 expected_channels 个波段即可得到 420nm -> 660nm。
    return hs_cube[:, :, :expected_channels]


@torch.no_grad()
def process_single_scene(
    model: SnapshotDepthHS,
    hs_path: str,
    depth_path: str,
    output_dir: str,
    patch_size: int,
    device: torch.device,
    min_depth: float,
    max_depth: float,
    rgb_bands: Tuple[int, int, int],
    diagnostic_dump: bool = False,
    stride_override: int = -1,
    min_tile_valid_ratio: float = 0.0,
    fill_skipped_tiles: str = 'zero',
):
    scene_name = os.path.splitext(os.path.basename(hs_path))[0].replace('_hs', '')
    print(f"\nProcessing scene: {scene_name} ...")

    hs_gt_raw = read_exr(hs_path)
    depth_gt_raw = read_exr(depth_path)
    expected_channels = int(getattr(model.hparams, 'hs_channels', hs_gt_raw.shape[2]))
    hs_gt_raw = select_hs_bands(hs_gt_raw, expected_channels)

    if depth_gt_raw.ndim == 3:
        depth_gt_raw = depth_gt_raw.squeeze(-1)

    depth_gt_raw = depth_gt_raw / 1000.0  # mm -> m
    valid_mask = (depth_gt_raw > (min_depth - 1e-3)).astype(np.float32)

    hs_norm, _, _ = normalize_hs_minmax(hs_gt_raw)
    depth_norm_ips = metric_depth_to_ips_np(depth_gt_raw, min_depth, max_depth)

    print(f"  [Data Info] GT Depth Range (Raw): [{depth_gt_raw.min():.4f}m, {depth_gt_raw.max():.4f}m]")
    print(f"  [Data Info] IPS Input Range:      [{depth_norm_ips.min():.4f}, {depth_norm_ips.max():.4f}]")

    hs_tensor = torch.from_numpy(hs_norm).permute(2, 0, 1).float()
    depth_tensor = torch.from_numpy(depth_norm_ips).float()

    # Determine optical model type
    optical_model = getattr(model.hparams, 'optical_model', 'legacy_camera')
    is_dodo = (optical_model == 'dodo_depth')
    print(f"  [Model] optical_model={optical_model}, is_dodo={is_dodo}")

    crop_width = int(model.hparams.crop_width)
    input_patch_size = int(patch_size)
    valid_size = input_patch_size - 4 * crop_width
    if valid_size <= 0:
        raise ValueError(
            f"Invalid patch setup: patch_size={input_patch_size}, crop_width={crop_width}, valid_size={valid_size}."
        )

    if is_dodo:
        # For dodo_depth: force crop_width=0, valid_size=128
        if crop_width != 0:
            crop_width = 0
            print(f"  [dodo] Override crop_width=0")
        if input_patch_size != 128:
            input_patch_size = 128
            valid_size = 128
            print(f"  [dodo] Override patch_size=128")

    # Build DoDo-specific tensors
    if is_dodo:
        # depth_metric: metric meters, background clamped to min_depth
        depth_metric_raw = depth_gt_raw.copy()
        depth_metric_raw[valid_mask < 0.5] = min_depth
        depth_metric_tensor = torch.from_numpy(depth_metric_raw).float()
        # valid_mask_tensor: binary foreground mask
        valid_mask_tensor = torch.from_numpy(valid_mask).float()

    stride = stride_override if stride_override > 0 else max(1, valid_size // 2)
    print(f"  [Stitching] Valid Patch: {valid_size}x{valid_size}")
    print(f"  [Stitching] Stride:      {stride}px")

    c, h, w = hs_tensor.shape
    est_hs_sum = torch.zeros((c, h, w), device=device)
    est_hs_weight = torch.zeros((c, h, w), device=device)
    est_depth_sum = torch.zeros((h, w), device=device)
    est_depth_weight = torch.zeros((h, w), device=device)

    patch_weight_mask = get_cosine_mask(valid_size, valid_size, device)

    pad_base = 2 * crop_width
    pad_buffer = input_patch_size

    def pad_tensor_4d(t, pad):
        return torch.nn.functional.pad(t.unsqueeze(0), pad, mode='reflect')

    hs_padded = pad_tensor_4d(hs_tensor,
        (pad_base + pad_buffer, pad_base + pad_buffer, pad_base + pad_buffer, pad_base + pad_buffer))
    depth_padded = pad_tensor_4d(depth_tensor.unsqueeze(0),
        (pad_base + pad_buffer, pad_base + pad_buffer, pad_base + pad_buffer, pad_base + pad_buffer))
    if is_dodo:
        dm_padded = pad_tensor_4d(depth_metric_tensor.unsqueeze(0),
            (pad_base + pad_buffer, pad_base + pad_buffer, pad_base + pad_buffer, pad_base + pad_buffer))
        vm_padded = pad_tensor_4d(valid_mask_tensor.unsqueeze(0),
            (pad_base + pad_buffer, pad_base + pad_buffer, pad_base + pad_buffer, pad_base + pad_buffer))

    # Per-tile measurement stats collection
    tile_meas_stats = []
    skipped_tiles = []
    total_tiles = 0
    skipped_tile_count = 0

    for y in tqdm(range(0, h, stride), desc='Inference'):
        for x in range(0, w, stride):
            total_tiles += 1
            py = y + pad_buffer
            px = x + pad_buffer

            # Pre-compute tile valid ratio for skip decision
            if is_dodo and min_tile_valid_ratio > 0:
                vm_check = vm_padded[0, 0, py:py + input_patch_size, px:px + input_patch_size]
                tile_vr = vm_check.mean().item()
                if tile_vr < min_tile_valid_ratio:
                    skipped_tile_count += 1
                    skipped_tiles.append({'y': y, 'x': x, 'valid_ratio': tile_vr,
                                          'py': py - pad_buffer, 'px': px - pad_buffer})
                    target_h = min(valid_size, h - y)
                    target_w = min(valid_size, w - x)
                    if fill_skipped_tiles == 'zero':
                        # zero-fill: leave sums unchanged (already zero)
                        pass
                    continue

            hs_patch = hs_padded[:, :, py:py + input_patch_size, px:px + input_patch_size].to(device)
            depth_patch = depth_padded[:, :, py:py + input_patch_size, px:px + input_patch_size].squeeze(1).to(device)

            if is_dodo:
                dm_patch = dm_padded[:, :, py:py + input_patch_size, px:px + input_patch_size].squeeze(1).to(device)
                vm_patch = vm_padded[:, :, py:py + input_patch_size, px:px + input_patch_size].squeeze(1).to(device)
                tile_valid_ratio = vm_patch.mean().item()
                hs_sum = hs_patch.sum().item()
                outputs = model(hs_patch, depth_patch, is_testing=torch.tensor(True, device=device),
                                depth_metric=dm_patch, valid_mask=vm_patch)
                # Per-tile measurement stats (after norm, from outputs.captimgs)
                capt = outputs.captimgs
                tile_meas_stats.append({
                    'y': y, 'x': x,
                    'valid_ratio': tile_valid_ratio,
                    'spectral_sum': hs_sum,
                    'capt_after_min': capt.min().item(),
                    'capt_after_max': capt.max().item(),
                    'capt_after_mean': capt.mean().item(),
                    'capt_after_std': capt.std().item(),
                })
            else:
                outputs = model(hs_patch, depth_patch, is_testing=torch.tensor(True, device=device))

            out_hs = outputs.est_images
            out_depth = outputs.est_depthmaps
            if out_depth.ndim == 4 and out_depth.shape[1] == 1:
                out_depth = out_depth.squeeze(1)

            target_h = min(valid_size, h - y)
            target_w = min(valid_size, w - x)
            if target_h <= 0 or target_w <= 0:
                continue

            mask_slice = patch_weight_mask[:target_h, :target_w]

            est_hs_sum[:, y:y + target_h, x:x + target_w] += out_hs[0, :, :target_h, :target_w] * mask_slice
            est_hs_weight[:, y:y + target_h, x:x + target_w] += mask_slice
            est_depth_sum[y:y + target_h, x:x + target_w] += out_depth[0, :target_h, :target_w] * mask_slice
            est_depth_weight[y:y + target_h, x:x + target_w] += mask_slice

    final_hs_norm = est_hs_sum / (est_hs_weight + 1e-8)
    final_depth_ips = est_depth_sum / (est_depth_weight + 1e-8)

    final_hs_norm = torch.nan_to_num(final_hs_norm, 0.0).cpu().numpy().transpose(1, 2, 0)
    final_depth_ips = torch.nan_to_num(final_depth_ips, 0.0).cpu().numpy()

    est_depth_real = ips_to_metric_np(final_depth_ips, min_depth, max_depth)

    num_bins = 8
    gt_bins, gt_pct, _ = compute_depth_histogram(depth_gt_raw, valid_mask, min_depth, max_depth, num_bins)
    _, pred_pct, _ = compute_depth_histogram(est_depth_real, valid_mask, min_depth, max_depth, num_bins)

    print('\n  [Depth Distribution Analysis]')
    print('  Range (m)      GT (%)     Pred (%)')
    print('  ----------------------------------')
    for i in range(num_bins):
        print(f"  {gt_bins[i]:.2f}-{gt_bins[i+1]:.2f}      {gt_pct[i]:6.2f}     {pred_pct[i]:6.2f}")

    print(f"  [Result Info] Pred Depth Range: [{est_depth_real.min():.4f}m, {est_depth_real.max():.4f}m]")

    mse_real = calculate_masked_mse(depth_gt_raw, est_depth_real, valid_mask)
    rmse_real = float(np.sqrt(mse_real)) if not np.isnan(mse_real) else float('nan')
    mae_real = calculate_masked_mae(depth_gt_raw, est_depth_real, valid_mask)
    psnr_hs_full = calculate_psnr(hs_norm, final_hs_norm)
    psnr_hs_masked = calculate_masked_psnr(hs_norm, final_hs_norm, valid_mask)
    sam_hs_masked = calculate_masked_sam(hs_norm, final_hs_norm, valid_mask)
    valid_ratio = float(valid_mask.sum() / valid_mask.size)
    pred_d_min = float(est_depth_real[valid_mask > 0.5].min()) if valid_mask.sum() > 0 else float('nan')
    pred_d_max = float(est_depth_real[valid_mask > 0.5].max()) if valid_mask.sum() > 0 else float('nan')
    pred_d_mean = float(est_depth_real[valid_mask > 0.5].mean()) if valid_mask.sum() > 0 else float('nan')
    pred_d_std = float(est_depth_real[valid_mask > 0.5].std()) if valid_mask.sum() > 0 else float('nan')

    print('  -> Metrics (Valid Region):')
    print(f"     Depth MSE:  {mse_real:.6f} (m^2)")
    print(f"     Depth RMSE: {rmse_real:.6f} (m)")
    print(f"     Depth MAE:  {mae_real:.6f} (m)")
    print(f"     HS PSNR(full):   {psnr_hs_full:.4f} dB")
    print(f"     HS PSNR(masked): {psnr_hs_masked:.4f} dB")
    print(f"     HS SAM(masked):  {sam_hs_masked:.6f} rad")
    print(f"     Valid ratio:     {valid_ratio:.4f}")
    print(f"     Pred Depth:      min={pred_d_min:.3f} max={pred_d_max:.3f} mean={pred_d_mean:.3f} std={pred_d_std:.3f}")

    scene_out_dir = os.path.join(output_dir, scene_name)
    os.makedirs(scene_out_dir, exist_ok=True)

    visualize_depth(est_depth_real, os.path.join(scene_out_dir, 'est_depth_fixed_scale.png'), vmin=min_depth, vmax=max_depth)
    visualize_depth(depth_gt_raw, os.path.join(scene_out_dir, 'gt_depth_fixed_scale.png'), vmin=min_depth, vmax=max_depth)
    visualize_depth(est_depth_real, os.path.join(scene_out_dir, 'est_depth_auto_scale.png'), vmin=None, vmax=None)

    gt_rgb = hs_norm[..., list(rgb_bands)]
    rgb_p_min = float(np.percentile(gt_rgb, 1))
    rgb_p_max = float(np.percentile(gt_rgb, 99))
    if rgb_p_max - rgb_p_min < 1e-6:
        rgb_p_max = rgb_p_min + 1.0

    visualize_hyperspectral(
        torch.from_numpy(final_hs_norm).permute(2, 0, 1),
        os.path.join(scene_out_dir, 'est_hs.png'),
        bands=rgb_bands,
        p_min=rgb_p_min,
        p_max=rgb_p_max,
    )
    visualize_hyperspectral(
        hs_tensor,
        os.path.join(scene_out_dir, 'gt_hs.png'),
        bands=rgb_bands,
        p_min=rgb_p_min,
        p_max=rgb_p_max,
    )

    with open(os.path.join(scene_out_dir, 'vis_info.txt'), 'w') as f:
        f.write(f"rgb_bands={rgb_bands}\n")
        f.write(f"rgb_percentile_min={rgb_p_min:.6f}\n")
        f.write(f"rgb_percentile_max={rgb_p_max:.6f}\n")

    # Compute skipped pixel ratio
    skipped_pixel_ratio = 0.0
    if total_tiles > 0 and skipped_tile_count > 0:
        skipped_pixel_ratio = skipped_tile_count / total_tiles * (valid_size * valid_size) / (h * w)

    # Diagnostic outputs
    scenario_metrics = {'scene_name': scene_name,
                        'total_tiles': total_tiles, 'skipped_tiles': skipped_tile_count,
                        'skipped_pixel_ratio': skipped_pixel_ratio}
    if diagnostic_dump:
        print('  [Diag] Saving diagnostic metrics...')
        sw = est_depth_weight.cpu().numpy()
        plt.imsave(os.path.join(scene_out_dir, 'stitch_weight_map.png'), sw, cmap='viridis')

        # Full scene metrics CSV
        with open(os.path.join(scene_out_dir, 'metrics_full_scene.csv'), 'w') as f:
            f.write('scene_name,psnr_full_db,psnr_masked_db,sam_masked_rad,depth_mae_m,depth_rmse_m,'
                    'valid_ratio,skipped_tiles,skipped_pixel_ratio,'
                    'pseudo_rgb_psnr_db,pseudo_rgb_mae,oracle_simulation\n')
            f.write(f'{scene_name},{psnr_hs_full:.4f},{psnr_hs_masked:.4f},{sam_hs_masked:.6f},'
                    f'{mae_real:.6f},{rmse_real:.6f},{valid_ratio:.4f},'
                    f'{skipped_tile_count},{skipped_pixel_ratio:.4f},'
                    f'{calculate_masked_psnr(hs_norm[..., list(rgb_bands)], final_hs_norm[..., list(rgb_bands)], valid_mask):.4f},'
                    f'{calculate_masked_mae(hs_norm[..., list(rgb_bands)], final_hs_norm[..., list(rgb_bands)], valid_mask):.6f},'
                    f'True\n')

        # Per-band metrics
        pb = compute_per_band_metrics(hs_norm, final_hs_norm, valid_mask)
        with open(os.path.join(scene_out_dir, 'metrics_per_band.csv'), 'w') as f:
            f.write('band,psnr_masked_db,mae_masked\n')
            for r in pb:
                f.write(f"{r['band']},{r['psnr_masked_db']:.4f},{r['mae_masked']:.6f}\n")

        # Region metrics CSV
        rm = compute_region_metrics(hs_norm, final_hs_norm, depth_gt_raw, est_depth_real, valid_mask, min_depth, max_depth)
        scenario_metrics['region'] = rm
        with open(os.path.join(scene_out_dir, 'metrics_regions.csv'), 'w') as f:
            keys = sorted(rm.keys())
            f.write(','.join(keys) + '\n')
            f.write(','.join(str(rm[k]) for k in keys) + '\n')

        # Depth baselines CSV
        db = compute_depth_baselines(depth_gt_raw, est_depth_real, valid_mask)
        scenario_metrics['depth_baselines'] = db
        with open(os.path.join(scene_out_dir, 'metrics_depth_baselines.csv'), 'w') as f:
            keys = sorted(db.keys())
            f.write(','.join(keys) + '\n')
            f.write(','.join(str(db[k]) for k in keys) + '\n')

        # Spectral quality CSV
        sq = compute_spectral_quality(hs_norm, final_hs_norm, valid_mask, rgb_bands)
        scenario_metrics['spectral'] = sq
        scenario_metrics['oracle_simulation'] = is_dodo
        if is_dodo:
            scenario_metrics['dodo_sensor_measurement'] = getattr(
                model.hparams, 'dodo_sensor_measurement', 'amplitude')
        with open(os.path.join(scene_out_dir, 'metrics_spectral_quality.csv'), 'w') as f:
            keys = sorted(sq.keys())
            f.write(','.join(keys) + '\n')
            f.write(','.join(str(sq[k]) for k in keys) + '\n')

        # Measurement stats CSV + JSON
        if is_dodo and tile_meas_stats:
            with open(os.path.join(scene_out_dir, 'metrics_measurement_tiles.csv'), 'w') as f:
                cols = ['y', 'x', 'valid_ratio', 'spectral_sum',
                        'capt_after_min', 'capt_after_max', 'capt_after_mean', 'capt_after_std']
                f.write(','.join(cols) + '\n')
                for ts in tile_meas_stats:
                    f.write(','.join(str(ts[c]) for c in cols) + '\n')

            after_stds = [ts['capt_after_std'] for ts in tile_meas_stats]
            after_means = [ts['capt_after_mean'] for ts in tile_meas_stats]
            after_maxs = [ts['capt_after_max'] for ts in tile_meas_stats]
            ms = {
                'num_tiles': len(tile_meas_stats),
                'after_norm_mean_std': float(np.mean(after_stds)),
                'after_norm_median_std': float(np.median(after_stds)),
                'after_norm_mean_mean': float(np.mean(after_means)),
                'after_norm_max_mean': float(np.mean(after_maxs)),
                'after_norm_zero_ratio': float(np.mean([1.0 if s < 1e-8 else 0.0 for s in after_stds])),
            }
            scenario_metrics['measurement_stats'] = ms
            with open(os.path.join(scene_out_dir, 'measurement_stats_summary.json'), 'w') as f:
                json.dump(ms, f, indent=2)

        # Skipped tiles log
        if skipped_tiles:
            with open(os.path.join(scene_out_dir, 'skipped_tiles.csv'), 'w') as f:
                f.write('y,x,valid_ratio\n')
                for st in skipped_tiles:
                    f.write(f"{st['y']},{st['x']},{st['valid_ratio']:.6f}\n")

        # Save diagnostic metrics JSON (comprehensive)
        import json
        with open(os.path.join(scene_out_dir, 'diagnostic_metrics.json'), 'w') as f:
            json.dump(scenario_metrics, f, indent=2, default=str)

    return scene_name, mse_real, rmse_real, mae_real, psnr_hs_full, psnr_hs_masked, sam_hs_masked, valid_ratio, pred_d_min, pred_d_max, pred_d_mean, pred_d_std, scenario_metrics


def compute_per_band_metrics(gt_hs, est_hs, valid_mask):
    """Compute per-band masked PSNR and MAE."""
    n_bands = gt_hs.shape[2]
    records = []
    for b in range(n_bands):
        gt_b = gt_hs[:, :, b]
        est_b = est_hs[:, :, b]
        psnr_b = calculate_masked_psnr(gt_b, est_b, valid_mask)
        mae_b = calculate_masked_mae(gt_b, est_b, valid_mask)
        records.append({'band': b, 'psnr_masked_db': psnr_b, 'mae_masked': mae_b})
    return records


def compute_region_metrics(gt_hs, est_hs, gt_d_m, est_d_m, valid_mask, depth_min, depth_max):
    """Compute metrics split by foreground/background/boundary/interior regions."""
    m = valid_mask > 0.5
    n_valid = m.sum()
    n_total = valid_mask.size
    if n_valid == 0:
        return {'foreground_ratio': 0.0, 'n_valid': 0, 'n_total': n_total}

    # Boundary: pixels within 8-pixel dilation of mask edges
    from scipy import ndimage
    eroded = ndimage.binary_erosion(m, iterations=8)
    boundary_mask = m & ~eroded
    interior_mask = eroded

    results = {
        'foreground_ratio': float(n_valid / n_total),
        'n_valid': int(n_valid),
        'n_total': n_total,
        'n_boundary': int(boundary_mask.sum()),
        'n_interior': int(interior_mask.sum()),
    }

    # HS metrics per region
    for label, region_m in [('fg', m), ('boundary', boundary_mask), ('interior', interior_mask)]:
        if region_m.sum() == 0:
            continue
        results[f'hs_psnr_masked_{label}'] = calculate_masked_psnr(gt_hs, est_hs, region_m)
        results[f'hs_sam_masked_{label}'] = calculate_masked_sam(gt_hs, est_hs, region_m)
        results[f'depth_mae_m_{label}'] = calculate_masked_mae(gt_d_m, est_d_m, region_m)

    # Background: !valid_mask
    bg_m = ~m
    if bg_m.sum() > 0:
        results['hs_psnr_full_bg'] = calculate_masked_psnr(gt_hs, est_hs, bg_m)
        results['depth_mae_m_bg'] = calculate_masked_mae(gt_d_m, est_d_m, bg_m)

    return results


def compute_depth_baselines(gt_d_m, est_d_m, valid_mask):
    """Compare model depth MAE/RMSE against constant-depth baselines."""
    m = valid_mask > 0.5
    if m.sum() == 0:
        return {}
    gt_valid = gt_d_m[m]
    est_valid = est_d_m[m]
    model_mae = float(np.mean(np.abs(gt_valid - est_valid)))
    model_rmse = float(np.sqrt(np.mean((gt_valid - est_valid)**2)))

    median_depth = float(np.median(gt_valid))
    mean_depth = float(np.mean(gt_valid))
    baseline_median_mae = float(np.mean(np.abs(gt_valid - median_depth)))
    baseline_mean_mae = float(np.mean(np.abs(gt_valid - mean_depth)))
    baseline_median_rmse = float(np.sqrt(np.mean((gt_valid - median_depth)**2)))
    baseline_mean_rmse = float(np.sqrt(np.mean((gt_valid - mean_depth)**2)))

    return {
        'model_mae_m': model_mae,
        'model_rmse_m': model_rmse,
        'baseline_median_mae_m': baseline_median_mae,
        'baseline_mean_mae_m': baseline_mean_mae,
        'baseline_median_rmse_m': baseline_median_rmse,
        'baseline_mean_rmse_m': baseline_mean_rmse,
        'mae_vs_median_ratio': model_mae / max(baseline_median_mae, 1e-6),
        'mae_vs_mean_ratio': model_mae / max(baseline_mean_mae, 1e-6),
        'gt_depth_median_m': median_depth,
        'gt_depth_mean_m': mean_depth,
        'gt_depth_std_m': float(np.std(gt_valid)),
    }


def compute_spectral_quality(gt_hs, est_hs, valid_mask, rgb_bands):
    """Compute SAM, spectral MAE, pseudo-RGB PSNR/MAE."""
    sam = calculate_masked_sam(gt_hs, est_hs, valid_mask)

    # Spectral MAE: per-pixel mean abs error across bands, then average over valid pixels
    m = valid_mask > 0.5
    spec_mae = float(np.mean(np.abs(gt_hs[m] - est_hs[m]))) if m.sum() > 0 else float('nan')

    # Pseudo-RGB: extract 3 representative bands
    b = [min(int(x), gt_hs.shape[2] - 1) for x in rgb_bands]
    gt_rgb = gt_hs[..., b]
    est_rgb = est_hs[..., b]
    rgb_psnr = calculate_masked_psnr(gt_rgb, est_rgb, valid_mask)
    rgb_mae = calculate_masked_mae(gt_rgb, est_rgb, valid_mask)

    return {
        'sam_rad': sam,
        'spectral_mae': spec_mae,
        'pseudo_rgb_psnr_db': rgb_psnr,
        'pseudo_rgb_mae': rgb_mae,
    }


def compute_captimgs_stats(before_norm, after_norm):
    """Compute measurement statistics before/after per-sample norm."""
    def stats(t):
        if t is None:
            return {}
        return {
            'min': float(t.min()), 'max': float(t.max()),
            'mean': float(t.mean()), 'std': float(t.std()),
        }
    return {'before_norm': stats(before_norm), 'after_norm': stats(after_norm)}


def save_diagnostic_quicklook(tensor_chw, path, vmin=None, vmax=None):
    """Save a quicklook PNG of a measurement tensor (3, H, W)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    img = tensor_chw.permute(1, 2, 0).numpy()
    # Handle >3 channel measurement: keep first 3 channels
    if img.shape[-1] > 3:
        img = img[..., :3]
    elif img.shape[-1] == 2:
        img = np.concatenate([img, img[..., :1]], axis=-1)
    if vmin is None:
        vmin = float(img.min())
    if vmax is None:
        vmax = float(img.max())
    if vmax - vmin < 1e-6:
        vmax = vmin + 1.0
    img = np.clip((img - vmin) / (vmax - vmin), 0, 1)
    plt.imsave(path, img)


@torch.no_grad()
def process_roi_direct(model, hs_tensor, depth_ips_tensor, depth_metric_tensor,
                        valid_mask_tensor, y, x, patch_size, device, min_depth, max_depth,
                        rgb_bands, out_dir, roi_name):
    """Direct inference on a single 128x128 ROI patch."""
    os.makedirs(out_dir, exist_ok=True)

    h, w = depth_ips_tensor.shape
    py = max(0, min(y, h - patch_size))
    px = max(0, min(x, w - patch_size))

    hs_patch = hs_tensor[:, py:py+patch_size, px:px+patch_size].unsqueeze(0).to(device)
    dp_patch = depth_ips_tensor[py:py+patch_size, px:px+patch_size].unsqueeze(0).to(device)
    dm_patch = depth_metric_tensor[py:py+patch_size, px:px+patch_size].unsqueeze(0).to(device)
    vm_patch = valid_mask_tensor[py:py+patch_size, px:px+patch_size].unsqueeze(0).to(device)

    is_dodo = (getattr(model.hparams, 'optical_model', 'legacy_camera') == 'dodo_depth')
    if is_dodo:
        outputs = model(hs_patch, dp_patch, is_testing=torch.tensor(True, device=device),
                        depth_metric=dm_patch, valid_mask=vm_patch)
    else:
        outputs = model(hs_patch, dp_patch, is_testing=torch.tensor(True, device=device))

    est_hs = outputs.est_images[0].cpu().numpy().transpose(1, 2, 0)
    est_depth_ips = outputs.est_depthmaps[0].cpu().numpy()
    if est_depth_ips.ndim == 3:
        est_depth_ips = est_depth_ips[0]
    est_depth_m = ips_to_metric_np(np.clip(est_depth_ips, 0, 1), min_depth, max_depth)

    # Ground truth crops
    gt_hs_crop = hs_tensor[:, py:py+patch_size, px:px+patch_size].cpu().numpy().transpose(1, 2, 0)
    gt_d_crop = depth_metric_tensor[py:py+patch_size, px:px+patch_size].cpu().numpy()
    vm_crop = valid_mask_tensor[py:py+patch_size, px:px+patch_size].cpu().numpy()

    # Metrics
    psnr_f = calculate_psnr(gt_hs_crop, est_hs)
    psnr_m = calculate_masked_psnr(gt_hs_crop, est_hs, vm_crop)
    sam_m = calculate_masked_sam(gt_hs_crop, est_hs, vm_crop)
    mae_d = calculate_masked_mae(gt_d_crop, est_depth_m, vm_crop)
    rmse_d = float(np.sqrt(calculate_masked_mse(gt_d_crop, est_depth_m, vm_crop))) if vm_crop.sum() > 0 else float('nan')
    valid_ratio = float(vm_crop.sum() / vm_crop.size)

    # Save quicklooks
    visualize_depth(est_depth_m, os.path.join(out_dir, f'{roi_name}_est_depth_fixed.png'), vmin=min_depth, vmax=max_depth)
    visualize_depth(gt_d_crop, os.path.join(out_dir, f'{roi_name}_gt_depth_fixed.png'), vmin=min_depth, vmax=max_depth)
    gt_rgb_crop = gt_hs_crop[..., list(rgb_bands)]
    p_min = float(np.percentile(gt_rgb_crop, 1))
    p_max = float(np.percentile(gt_rgb_crop, 99))
    visualize_hyperspectral(torch.from_numpy(est_hs).permute(2, 0, 1),
                            os.path.join(out_dir, f'{roi_name}_est_hs.png'),
                            bands=rgb_bands, p_min=p_min, p_max=p_max)
    visualize_hyperspectral(torch.from_numpy(gt_hs_crop).permute(2, 0, 1),
                            os.path.join(out_dir, f'{roi_name}_gt_hs.png'),
                            bands=rgb_bands, p_min=p_min, p_max=p_max)

    # Measurement diagnostic: enable capture on model
    model._diag_capture = True
    if is_dodo:
        _ = model(hs_patch, dp_patch, is_testing=torch.tensor(True, device=device),
                  depth_metric=dm_patch, valid_mask=vm_patch)
    capt_before = getattr(model, '_diag_capt_before', None)
    capt_after = getattr(model, '_diag_capt_after', None)
    if capt_before is not None:
        save_diagnostic_quicklook(capt_before[0], os.path.join(out_dir, f'{roi_name}_capt_before_norm.png'))
    if capt_after is not None:
        save_diagnostic_quicklook(capt_after[0], os.path.join(out_dir, f'{roi_name}_capt_after_norm.png'))
    cs = compute_captimgs_stats(capt_before, capt_after)
    model._diag_capture = False

    roi_metrics = {
        'roi_name': roi_name, 'y': py, 'x': px,
        'hs_psnr_full_db': psnr_f, 'hs_psnr_masked_db': psnr_m,
        'hs_sam_masked_rad': sam_m, 'depth_mae_m': mae_d, 'depth_rmse_m': rmse_d,
        'valid_ratio': valid_ratio,
    }
    roi_metrics.update({f'capt_{k}': v for k, v in cs.items()})

    # Per-band on ROI
    pb = compute_per_band_metrics(gt_hs_crop, est_hs, vm_crop)
    depth_bl = compute_depth_baselines(gt_d_crop, est_depth_m, vm_crop)
    spec_q = compute_spectral_quality(gt_hs_crop, est_hs, vm_crop, rgb_bands)
    region_m = compute_region_metrics(gt_hs_crop, est_hs, gt_d_crop, est_depth_m, vm_crop, min_depth, max_depth)

    print(f'  [ROI {roi_name}] PSNR(m)={psnr_m:.2f}dB, MAE(d)={mae_d:.3f}m, SAM={sam_m:.3f}rad, valid={valid_ratio:.3f}')
    return roi_metrics, pb, depth_bl, spec_q, region_m


def build_args() -> argparse.Namespace:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='Inference for SnapshotDepthHS model')

    parser.add_argument('--input_folder', type=str, default=os.path.join(script_dir, 'Baek数据集', 'deploy 16'))
    parser.add_argument(
        '--ckpt_path',
        type=str,
        default=os.path.join(script_dir, 'data', 'Hyperspectral_LearnedDepth', 'version_55', 'checkpoints'),
        help='checkpoint file or directory',
    )
    parser.add_argument('--output_dir', type=str, default=os.path.join(script_dir, 'infer_results'))
    parser.add_argument('--patch_size', type=int, default=512)
    parser.add_argument('--depth_min', type=float, default=0.4)
    parser.add_argument('--depth_max', type=float, default=2.0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--max_scenes', type=int, default=0,
                        help='Max scenes to process (0 = all)')
    parser.add_argument('--diagnostic_dump', action='store_true', default=False,
                        help='Save .npy arrays and extended diagnostic metrics')
    parser.add_argument('--stride', type=int, default=-1,
                        help='Tile stride override (-1 = auto = valid_size // 2)')
    parser.add_argument('--measurement_norm_override', type=str, default='checkpoint',
                        choices=['checkpoint', 'none', 'per_sample_mean_std', 'per_sample_minmax'],
                        help='Override dodo_measurement_norm for inference')
    parser.add_argument('--min_tile_valid_ratio', type=float, default=0.0,
                        help='Skip tiles with valid ratio below this threshold')
    parser.add_argument('--fill_skipped_tiles', type=str, default='zero',
                        choices=['zero'],  # gt_background, nearest not yet implemented
                        help='How to fill skipped tiles (zero only for now)')

    return parser.parse_args()


def main():
    args = build_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_folder = resolve_path(args.input_folder, script_dir)
    ckpt_path = resolve_path(args.ckpt_path, script_dir)
    output_dir = resolve_path(args.output_dir, script_dir)

    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f'Using device: {device}')
    print(f'Input folder: {input_folder}')
    print(f'Checkpoint path: {ckpt_path}')
    print(f'Output dir: {output_dir}')

    if not os.path.isdir(input_folder):
        raise FileNotFoundError(f'Input folder not found: {input_folder}')

    real_ckpt_path = choose_checkpoint(ckpt_path)
    print(f'Loading model from: {real_ckpt_path}')
    model = load_model(real_ckpt_path, device)

    is_dodo_model = (getattr(model.hparams, 'optical_model', 'legacy_camera') == 'dodo_depth')
    oracle_simulation = is_dodo_model  # only DoDo-depth uses GT depth for measurement synthesis

    hs_files = sorted(glob.glob(os.path.join(input_folder, '*_hs.exr')))
    if not hs_files:
        raise FileNotFoundError(f'No *_hs.exr files found in: {input_folder}')

    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, 'metrics_real.txt')
    with open(metrics_path, 'w') as f:
        f.write('scene,mse_m2,rmse_m,mae_m,hs_psnr_full_db,hs_psnr_masked_db,hs_sam_masked_rad,valid_ratio,pred_d_min_m,pred_d_max_m,pred_d_mean_m,pred_d_std_m,oracle_simulation\n')

    ckpt_min_depth = float(getattr(model.hparams, 'min_depth', args.depth_min))
    ckpt_max_depth = float(getattr(model.hparams, 'max_depth', args.depth_max))
    if abs(args.depth_min - ckpt_min_depth) > 1e-6 or abs(args.depth_max - ckpt_max_depth) > 1e-6:
        print(
            f"[Depth Range Override] Using checkpoint depth range [{ckpt_min_depth:.3f}, {ckpt_max_depth:.3f}] "
            f"instead of CLI [{args.depth_min:.3f}, {args.depth_max:.3f}]"
        )

    rgb_bands = pick_rgb_band_indices(
        int(getattr(model.hparams, 'hs_channels', 25)),
        float(getattr(model.hparams, 'start_wl', 420e-9)),
        float(getattr(model.hparams, 'end_wl', 660e-9)),
    )
    print(f'[Visualization] RGB bands selected: {rgb_bands}')

    # Apply measurement norm override for inference-only ablation
    if args.measurement_norm_override != 'checkpoint':
        setattr(model, '_norm_override',
                None if args.measurement_norm_override == 'none' else args.measurement_norm_override)
        print(f'[NormOverride] dodo_measurement_norm set to: {args.measurement_norm_override}')

    # Save command.txt
    with open(os.path.join(output_dir, 'command.txt'), 'w') as f:
        import sys
        f.write(' '.join(sys.argv) + '\n')

    print(f'Found {len(hs_files)} scenes to process.')
    if args.max_scenes > 0:
        hs_files = hs_files[:args.max_scenes]
        print(f'[max_scenes] Limited to {len(hs_files)} scenes.')

    all_results = []
    skipped = 0
    for hs_file in hs_files:
        depth_file = hs_file.replace('_hs.exr', '_depth_map.exr')
        if not os.path.exists(depth_file):
            print(f'Warning: depth map not found for {hs_file}, skipping.')
            skipped += 1
            continue

        try:
            results = process_single_scene(
                model, hs_file, depth_file, output_dir, args.patch_size,
                device, ckpt_min_depth, ckpt_max_depth, rgb_bands,
                diagnostic_dump=args.diagnostic_dump,
                stride_override=args.stride,
                min_tile_valid_ratio=args.min_tile_valid_ratio,
                fill_skipped_tiles=args.fill_skipped_tiles,
            )
            all_results.append(results)
        except Exception as e:
            print(f'ERROR processing {hs_file}: {e}')
            import traceback; traceback.print_exc()
            skipped += 1
            continue

        scene_name = results[0]
        mse_real, rmse_real, mae_real = results[1], results[2], results[3]
        psnr_hs_full, psnr_hs_masked, sam_hs_masked = results[4], results[5], results[6]
        valid_ratio, pred_d_min, pred_d_max, pred_d_mean, pred_d_std = results[7], results[8], results[9], results[10], results[11]
        scenario_metrics = results[12] if len(results) > 12 else {}
        with open(metrics_path, 'a') as f:
            f.write(
                f'{scene_name},{mse_real:.6f},{rmse_real:.6f},{mae_real:.6f},'
                f'{psnr_hs_full:.4f},{psnr_hs_masked:.4f},{sam_hs_masked:.6f},'
                f'{valid_ratio:.4f},{pred_d_min:.3f},{pred_d_max:.3f},{pred_d_mean:.3f},{pred_d_std:.3f},'
                f'{oracle_simulation}\n'
            )

    # Write aggregate metrics
    if all_results:
        mae_vals = [r[3] for r in all_results if not np.isnan(r[3])]
        rmse_vals = [r[2] for r in all_results if not np.isnan(r[2])]
        psnr_m_vals = [r[5] for r in all_results if not np.isnan(r[5])]
        psnr_f_vals = [r[4] for r in all_results if not np.isnan(r[4])]
        sam_vals = [r[6] for r in all_results if not np.isnan(r[6])]

        agg_path = os.path.join(output_dir, 'aggregate_metrics.json')
        agg = {
            'scene_count': len(all_results),
            'skipped_count': skipped,
            'oracle_simulation': oracle_simulation,
            'dodo_sensor_measurement': getattr(model.hparams, 'dodo_sensor_measurement', 'amplitude'),
            'depth_mae_m_mean': float(np.mean(mae_vals)) if mae_vals else float('nan'),
            'depth_mae_m_median': float(np.median(mae_vals)) if mae_vals else float('nan'),
            'depth_rmse_m_mean': float(np.mean(rmse_vals)) if rmse_vals else float('nan'),
            'depth_rmse_m_median': float(np.median(rmse_vals)) if rmse_vals else float('nan'),
            'hs_psnr_masked_db_mean': float(np.mean(psnr_m_vals)) if psnr_m_vals else float('nan'),
            'hs_psnr_masked_db_median': float(np.median(psnr_m_vals)) if psnr_m_vals else float('nan'),
            'hs_psnr_full_db_mean': float(np.mean(psnr_f_vals)) if psnr_f_vals else float('nan'),
            'hs_psnr_full_db_median': float(np.median(psnr_f_vals)) if psnr_f_vals else float('nan'),
            'hs_sam_masked_rad_mean': float(np.mean(sam_vals)) if sam_vals else float('nan'),
        }
        import json
        with open(agg_path, 'w') as f:
            json.dump(agg, f, indent=2)
        print(f'\nAggregate metrics saved to: {agg_path}')
        print(f'  Scenes: {agg["scene_count"]}, Skipped: {agg["skipped_count"]}')
        print(f'  Depth MAE:   mean={agg["depth_mae_m_mean"]:.4f}m, median={agg["depth_mae_m_median"]:.4f}m')
        print(f'  Depth RMSE:  mean={agg["depth_rmse_m_mean"]:.4f}m, median={agg["depth_rmse_m_median"]:.4f}m')
        print(f'  HS PSNR(m):  mean={agg["hs_psnr_masked_db_mean"]:.2f}dB, median={agg["hs_psnr_masked_db_median"]:.2f}dB')
    else:
        print('No successful scenes processed!')

    # ROI direct inference diagnostics
    if args.diagnostic_dump and len(hs_files) > 0:
        print('\n=== ROI Direct Inference Diagnostics ===')
        hs_file = hs_files[0]
        depth_file = hs_file.replace('_hs.exr', '_depth_map.exr')
        hs_gt_raw = read_exr(hs_file)
        depth_gt_raw = read_exr(depth_file)
        expected_channels = int(getattr(model.hparams, 'hs_channels', hs_gt_raw.shape[2]))
        hs_gt_raw = select_hs_bands(hs_gt_raw, expected_channels)
        if depth_gt_raw.ndim == 3:
            depth_gt_raw = depth_gt_raw.squeeze(-1)
        depth_gt_raw = depth_gt_raw / 1000.0
        valid_mask_roi = (depth_gt_raw > (ckpt_min_depth - 1e-3)).astype(np.float32)
        hs_norm_roi, _, _ = normalize_hs_minmax(hs_gt_raw)
        depth_ips_roi = metric_depth_to_ips_np(depth_gt_raw, ckpt_min_depth, ckpt_max_depth)
        hs_t_roi = torch.from_numpy(hs_norm_roi).permute(2, 0, 1).float()
        dp_t_roi = torch.from_numpy(depth_ips_roi).float()
        dm_t_roi = torch.from_numpy(np.clip(depth_gt_raw, ckpt_min_depth, ckpt_max_depth)).float()
        vm_t_roi = torch.from_numpy(valid_mask_roi).float()

        h, w = dp_t_roi.shape
        ps = args.patch_size
        roi_defs = []
        # Find a high-foreground patch (scan for max valid ratio)
        best_ratio = 0; best_y = best_x = 0
        for sy in range(0, h - ps, ps // 2):
            for sx in range(0, w - ps, ps // 2):
                r = vm_t_roi[sy:sy+ps, sx:sx+ps].mean().item()
                if r > best_ratio:
                    best_ratio = r; best_y = sy; best_x = sx
        roi_defs.append(('high_fg', best_y, best_x))
        # Boundary patch: near edge of valid mask (use mid-point)
        roi_defs.append(('boundary', h // 2 - ps // 2, w // 2 - ps // 2))
        # Low-foreground patch: find area with low valid ratio
        worst_ratio = 1.0; worst_y = worst_x = 0
        for sy in range(0, h - ps, ps // 2):
            for sx in range(0, w - ps, ps // 2):
                r = vm_t_roi[sy:sy+ps, sx:sx+ps].mean().item()
                if r < worst_ratio and r > 0.01:
                    worst_ratio = r; worst_y = sy; worst_x = sx
        roi_defs.append(('low_fg', worst_y, worst_x))

        roi_dir = os.path.join(output_dir, 'roi_diagnostics')
        os.makedirs(roi_dir, exist_ok=True)
        for roi_name, ry, rx in roi_defs:
            print(f'  ROI {roi_name} at ({ry},{rx}) valid_ratio={vm_t_roi[ry:ry+ps, rx:rx+ps].mean().item():.3f}')
            r_metrics, r_pb, r_db, r_sq, r_rm = process_roi_direct(
                model, hs_t_roi, dp_t_roi, dm_t_roi, vm_t_roi,
                ry, rx, ps, device, ckpt_min_depth, ckpt_max_depth,
                rgb_bands, roi_dir, roi_name)
            # Save ROI diagnostic metrics
            roi_all = {'roi_metrics': r_metrics, 'per_band': r_pb,
                       'depth_baselines': r_db, 'spectral': r_sq, 'region': r_rm}
            import json
            with open(os.path.join(roi_dir, f'{roi_name}_metrics.json'), 'w') as f:
                json.dump(roi_all, f, indent=2, default=str)

    print(f'Inference done. Metrics saved to: {metrics_path}')


if __name__ == '__main__':
    main()
