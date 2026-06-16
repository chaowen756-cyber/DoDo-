#!/usr/bin/env python
"""Evaluate brightness/gain-corrected PSNR on full-scene stitched predictions."""

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from infer_contect import (
    get_cosine_mask,
    ips_to_metric_np,
    load_model,
    metric_depth_to_ips_np,
    normalize_hs_for_hparams,
    read_exr,
    select_hs_bands,
)


def parse_scene_list(text: str) -> List[int]:
    scenes = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            scenes.extend(range(int(lo), int(hi) + 1))
        else:
            scenes.append(int(part))
    return sorted(dict.fromkeys(scenes))


def psnr_from_mse(mse: float, data_range: float = 1.0) -> float:
    if not np.isfinite(mse):
        return float("nan")
    if mse <= 0.0:
        return 100.0
    return float(10.0 * math.log10((data_range * data_range) / mse))


def masked_mse(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    m = mask > 0.5
    if m.sum() == 0:
        return float("nan")
    return float(np.mean((gt[m] - pred[m]) ** 2))


def masked_psnr(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    return psnr_from_mse(masked_mse(gt, pred, mask))


def spectral_angle(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray, eps: float = 1e-8) -> float:
    m = mask > 0.5
    if m.sum() == 0:
        return float("nan")
    g = gt[m]
    p = pred[m]
    gn = np.linalg.norm(g, axis=1)
    pn = np.linalg.norm(p, axis=1)
    keep = (gn > eps) & (pn > eps)
    if not np.any(keep):
        return float("nan")
    g = g[keep]
    p = p[keep]
    gn = gn[keep]
    pn = pn[keep]
    c = np.sum(g * p, axis=1) / (gn * pn + eps)
    return float(np.mean(np.arccos(np.clip(c, -1.0, 1.0))))


def scalar_gain(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    m = mask > 0.5
    if m.sum() == 0:
        return float("nan")
    g = gt[m].reshape(-1).astype(np.float64)
    p = pred[m].reshape(-1).astype(np.float64)
    den = float(np.dot(p, p))
    if den <= 1e-20:
        return 0.0
    return float(np.dot(g, p) / den)


def per_band_gain(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> np.ndarray:
    gains = []
    m = mask > 0.5
    for b in range(gt.shape[2]):
        g = gt[:, :, b][m].reshape(-1).astype(np.float64)
        p = pred[:, :, b][m].reshape(-1).astype(np.float64)
        den = float(np.dot(p, p))
        gains.append(float(np.dot(g, p) / den) if den > 1e-20 else 0.0)
    return np.asarray(gains, dtype=np.float32)


def per_band_psnr(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> Tuple[float, float, int]:
    values = []
    for b in range(gt.shape[2]):
        values.append(masked_psnr(gt[:, :, b], pred[:, :, b], mask))
    arr = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(arr)), float(np.nanmin(arr)), int(np.nanargmin(arr))


def per_pixel_psnr(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> np.ndarray:
    m = mask > 0.5
    mse = np.mean((gt[m] - pred[m]) ** 2, axis=1)
    return 10.0 * np.log10(1.0 / np.maximum(mse, 1e-12))


def quality_bins(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray, prefix: str) -> List[Dict]:
    psnr = per_pixel_psnr(gt, pred, mask)
    bins = [
        ("lt20", -np.inf, 20.0),
        ("20_25", 20.0, 25.0),
        ("25_30", 25.0, 30.0),
        ("30_35", 30.0, 35.0),
        ("ge35", 35.0, np.inf),
    ]
    rows = []
    total = max(int(psnr.size), 1)
    for label, lo, hi in bins:
        keep = (psnr >= lo) & (psnr < hi)
        rows.append({
            "quality_bin": label,
            f"{prefix}_pixels": int(np.sum(keep)),
            f"{prefix}_pixel_frac": float(np.sum(keep) / total),
        })
    return rows


def region_masks(valid_mask: np.ndarray) -> Dict[str, np.ndarray]:
    m = valid_mask > 0.5
    masks = {"foreground": m}
    try:
        from scipy import ndimage
        eroded = ndimage.binary_erosion(m, iterations=8)
        masks["interior"] = eroded
        masks["boundary8"] = m & ~eroded
    except Exception:
        masks["interior"] = m
        masks["boundary8"] = np.zeros_like(m, dtype=bool)
    return masks


def region_error_rows(scene: str, gt: np.ndarray, pred: np.ndarray, valid_mask: np.ndarray) -> List[Dict]:
    rows = []
    fg = valid_mask > 0.5
    total_fg_mse_sum = float(np.sum((gt[fg] - pred[fg]) ** 2)) if np.any(fg) else 0.0
    for name, rm in region_masks(valid_mask).items():
        if not np.any(rm):
            continue
        err2 = (gt[rm] - pred[rm]) ** 2
        mse = float(np.mean(err2))
        ps = per_pixel_psnr(gt, pred, rm.astype(np.float32))
        rows.append({
            "scene": scene,
            "region": name,
            "pixel_count": int(np.sum(rm)),
            "pixel_frac_of_fg": float(np.sum(rm) / max(np.sum(fg), 1)),
            "psnr_db": psnr_from_mse(mse),
            "mse": mse,
            "mse_share_of_fg": float(np.sum(err2) / max(total_fg_mse_sum, 1e-20)),
            "pixel_psnr_mean": float(np.mean(ps)),
            "pixel_psnr_p10": float(np.percentile(ps, 10.0)),
            "pixel_psnr_p50": float(np.percentile(ps, 50.0)),
            "pixel_psnr_p90": float(np.percentile(ps, 90.0)),
            "frac_pixel_psnr_lt20": float(np.mean(ps < 20.0)),
            "frac_pixel_psnr_lt25": float(np.mean(ps < 25.0)),
            "frac_pixel_psnr_ge30": float(np.mean(ps >= 30.0)),
        })
    return rows


def depth_bin_error_rows(
    scene: str,
    gt: np.ndarray,
    pred: np.ndarray,
    valid_mask: np.ndarray,
    depth_m: np.ndarray,
    min_depth: float,
    max_depth: float,
    num_bins: int = 8,
) -> List[Dict]:
    rows = []
    fg = valid_mask > 0.5
    total_fg_mse_sum = float(np.sum((gt[fg] - pred[fg]) ** 2)) if np.any(fg) else 0.0
    edges = np.linspace(min_depth, max_depth, num_bins + 1)
    for i in range(num_bins):
        lo, hi = float(edges[i]), float(edges[i + 1])
        if i == num_bins - 1:
            bm = fg & (depth_m >= lo) & (depth_m <= hi)
        else:
            bm = fg & (depth_m >= lo) & (depth_m < hi)
        if not np.any(bm):
            continue
        err2 = (gt[bm] - pred[bm]) ** 2
        ps = per_pixel_psnr(gt, pred, bm.astype(np.float32))
        rows.append({
            "scene": scene,
            "depth_bin": f"{lo:.2f}-{hi:.2f}m",
            "pixel_count": int(np.sum(bm)),
            "pixel_frac_of_fg": float(np.sum(bm) / max(np.sum(fg), 1)),
            "psnr_db": psnr_from_mse(float(np.mean(err2))),
            "mse_share_of_fg": float(np.sum(err2) / max(total_fg_mse_sum, 1e-20)),
            "frac_pixel_psnr_lt20": float(np.mean(ps < 20.0)),
            "frac_pixel_psnr_lt25": float(np.mean(ps < 25.0)),
            "frac_pixel_psnr_ge30": float(np.mean(ps >= 30.0)),
        })
    return rows


def scene_paths(data_root: str, scene_no: int) -> Tuple[str, str]:
    folder = os.path.join(data_root, f"deploy {scene_no}")
    hs_path = os.path.join(folder, f"scene{scene_no:02d}_hs.exr")
    depth_path = os.path.join(folder, f"scene{scene_no:02d}_depth_map.exr")
    if not os.path.exists(hs_path):
        raise FileNotFoundError(hs_path)
    if not os.path.exists(depth_path):
        raise FileNotFoundError(depth_path)
    return hs_path, depth_path


@torch.no_grad()
def stitched_predict(
    model,
    hs_norm: np.ndarray,
    depth_m: np.ndarray,
    valid_mask: np.ndarray,
    min_depth: float,
    max_depth: float,
    patch_size: int,
    stride: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    hs_tensor = torch.from_numpy(hs_norm).permute(2, 0, 1).float()
    depth_ips = metric_depth_to_ips_np(depth_m, min_depth, max_depth)
    depth_tensor = torch.from_numpy(depth_ips).float()
    depth_metric = depth_m.copy()
    depth_metric[valid_mask < 0.5] = min_depth
    depth_metric_tensor = torch.from_numpy(depth_metric).float()
    valid_mask_tensor = torch.from_numpy(valid_mask).float()

    input_patch_size = int(patch_size)
    crop_width = int(getattr(model.hparams, "crop_width", 0))
    is_dodo = getattr(model.hparams, "optical_model", "legacy_camera") == "dodo_depth"
    valid_size = input_patch_size - 4 * crop_width
    if is_dodo:
        crop_width = 0
        input_patch_size = 128
        valid_size = 128
    if valid_size <= 0:
        raise ValueError(f"Invalid patch_size={patch_size}, crop_width={crop_width}")
    stride = int(stride) if int(stride) > 0 else max(1, valid_size // 2)

    c, h, w = hs_tensor.shape
    est_hs_sum = torch.zeros((c, h, w), device=device)
    est_hs_weight = torch.zeros((c, h, w), device=device)
    est_depth_sum = torch.zeros((h, w), device=device)
    est_depth_weight = torch.zeros((h, w), device=device)
    patch_weight = get_cosine_mask(valid_size, valid_size, device)

    pad_base = 2 * crop_width
    pad_buffer = input_patch_size

    def pad4(t):
        return torch.nn.functional.pad(
            t.unsqueeze(0),
            (pad_base + pad_buffer, pad_base + pad_buffer, pad_base + pad_buffer, pad_base + pad_buffer),
            mode="reflect",
        )

    hs_padded = pad4(hs_tensor)
    depth_padded = pad4(depth_tensor.unsqueeze(0))
    dm_padded = pad4(depth_metric_tensor.unsqueeze(0))
    vm_padded = pad4(valid_mask_tensor.unsqueeze(0))

    y_starts = range(0, h, stride)
    x_starts = range(0, w, stride)
    for y in tqdm(y_starts, desc="tiles", leave=False):
        for x in x_starts:
            py = y + pad_buffer
            px = x + pad_buffer
            hs_patch = hs_padded[:, :, py:py + input_patch_size, px:px + input_patch_size].to(device)
            depth_patch = depth_padded[:, :, py:py + input_patch_size, px:px + input_patch_size].squeeze(1).to(device)
            if is_dodo:
                dm_patch = dm_padded[:, :, py:py + input_patch_size, px:px + input_patch_size].squeeze(1).to(device)
                vm_patch = vm_padded[:, :, py:py + input_patch_size, px:px + input_patch_size].squeeze(1).to(device)
                outputs = model(
                    hs_patch,
                    depth_patch,
                    is_testing=torch.tensor(True, device=device),
                    depth_metric=dm_patch,
                    valid_mask=vm_patch,
                )
            else:
                outputs = model(hs_patch, depth_patch, is_testing=torch.tensor(True, device=device))

            out_hs = outputs.est_images
            out_depth = outputs.est_depthmaps
            if out_depth.ndim == 4 and out_depth.shape[1] == 1:
                out_depth = out_depth.squeeze(1)

            out_y0 = max(y, 0)
            out_x0 = max(x, 0)
            out_y1 = min(y + valid_size, h)
            out_x1 = min(x + valid_size, w)
            th = out_y1 - out_y0
            tw = out_x1 - out_x0
            if th <= 0 or tw <= 0:
                continue
            src_y0 = out_y0 - y
            src_x0 = out_x0 - x
            ms = patch_weight[src_y0:src_y0 + th, src_x0:src_x0 + tw]
            est_hs_sum[:, out_y0:out_y1, out_x0:out_x1] += out_hs[0, :, src_y0:src_y0 + th, src_x0:src_x0 + tw] * ms
            est_hs_weight[:, out_y0:out_y1, out_x0:out_x1] += ms
            est_depth_sum[out_y0:out_y1, out_x0:out_x1] += out_depth[0, src_y0:src_y0 + th, src_x0:src_x0 + tw] * ms
            est_depth_weight[out_y0:out_y1, out_x0:out_x1] += ms

    pred_hs = (est_hs_sum / (est_hs_weight + 1e-8)).cpu().numpy().transpose(1, 2, 0)
    pred_depth_ips = (est_depth_sum / (est_depth_weight + 1e-8)).cpu().numpy()
    pred_hs = np.nan_to_num(pred_hs, nan=0.0, posinf=0.0, neginf=0.0)
    pred_depth_m = ips_to_metric_np(np.nan_to_num(pred_depth_ips, nan=0.0), min_depth, max_depth)
    return pred_hs.astype(np.float32, copy=False), pred_depth_m.astype(np.float32, copy=False)


def evaluate_scene(args, model, scene_no: int, device: torch.device) -> Tuple[Dict, List[Dict]]:
    hs_path, depth_path = scene_paths(args.data_root, scene_no)
    hs_raw = select_hs_bands(read_exr(hs_path), int(getattr(model.hparams, "hs_channels", 25)))
    depth_raw = read_exr(depth_path)
    if depth_raw.ndim == 3:
        depth_raw = depth_raw.squeeze(-1)
    depth_m = depth_raw.astype(np.float32, copy=False) / 1000.0
    min_depth = float(getattr(model.hparams, "min_depth", args.min_depth))
    max_depth = float(getattr(model.hparams, "max_depth", args.max_depth))
    valid_mask = (depth_m > (min_depth - 1e-3)).astype(np.float32)
    hs_norm = normalize_hs_for_hparams(hs_raw, model.hparams)

    pred, pred_depth_m = stitched_predict(
        model, hs_norm, depth_m, valid_mask, min_depth, max_depth,
        args.patch_size, args.stride, device,
    )

    raw_psnr = masked_psnr(hs_norm, pred, valid_mask)
    raw_pb_mean, raw_pb_min, raw_pb_argmin = per_band_psnr(hs_norm, pred, valid_mask)

    g_scalar = scalar_gain(hs_norm, pred, valid_mask)
    pred_scalar = pred * g_scalar
    scalar_psnr = masked_psnr(hs_norm, pred_scalar, valid_mask)
    scalar_clip_psnr = masked_psnr(hs_norm, np.clip(pred_scalar, 0.0, 1.0), valid_mask)
    scalar_pb_mean, scalar_pb_min, scalar_pb_argmin = per_band_psnr(hs_norm, pred_scalar, valid_mask)

    gains = per_band_gain(hs_norm, pred, valid_mask)
    pred_band = pred * gains.reshape(1, 1, -1)
    band_psnr = masked_psnr(hs_norm, pred_band, valid_mask)
    band_clip_psnr = masked_psnr(hs_norm, np.clip(pred_band, 0.0, 1.0), valid_mask)
    band_pb_mean, band_pb_min, band_pb_argmin = per_band_psnr(hs_norm, pred_band, valid_mask)

    m = valid_mask > 0.5
    depth_mae = float(np.mean(np.abs(depth_m[m] - pred_depth_m[m]))) if m.sum() else float("nan")

    row = {
        "scene": f"scene{scene_no:02d}",
        "valid_ratio": float(valid_mask.mean()),
        "raw_psnr_db": raw_psnr,
        "scalar_gain": g_scalar,
        "scalar_gain_psnr_db": scalar_psnr,
        "scalar_gain_clipped_psnr_db": scalar_clip_psnr,
        "per_band_gain_psnr_db": band_psnr,
        "per_band_gain_clipped_psnr_db": band_clip_psnr,
        "raw_mean_per_band_psnr_db": raw_pb_mean,
        "raw_min_per_band_psnr_db": raw_pb_min,
        "raw_min_band": raw_pb_argmin,
        "scalar_gain_mean_per_band_psnr_db": scalar_pb_mean,
        "scalar_gain_min_per_band_psnr_db": scalar_pb_min,
        "scalar_gain_min_band": scalar_pb_argmin,
        "per_band_gain_mean_per_band_psnr_db": band_pb_mean,
        "per_band_gain_min_per_band_psnr_db": band_pb_min,
        "per_band_gain_min_band": band_pb_argmin,
        "sam_raw_rad": spectral_angle(hs_norm, pred, valid_mask),
        "sam_scalar_gain_rad": spectral_angle(hs_norm, pred_scalar, valid_mask),
        "sam_per_band_gain_rad": spectral_angle(hs_norm, pred_band, valid_mask),
        "depth_mae_m": depth_mae,
        "per_band_gain_min": float(np.min(gains)),
        "per_band_gain_max": float(np.max(gains)),
        "per_band_gain_mean": float(np.mean(gains)),
    }
    band_mses = []
    for b in range(hs_norm.shape[2]):
        band_mses.append(masked_mse(hs_norm[:, :, b], pred[:, :, b], valid_mask))
    band_mses = np.asarray(band_mses, dtype=np.float64)
    total_band_mse = float(np.sum(band_mses))

    per_band_rows = []
    for b, gain in enumerate(gains.tolist()):
        per_band_rows.append({
            "scene": f"scene{scene_no:02d}",
            "band": b,
            "gain": gain,
            "raw_psnr_db": masked_psnr(hs_norm[:, :, b], pred[:, :, b], valid_mask),
            "gain_psnr_db": masked_psnr(hs_norm[:, :, b], pred_band[:, :, b], valid_mask),
            "raw_mse": float(band_mses[b]),
            "raw_mse_share": float(band_mses[b] / max(total_band_mse, 1e-20)),
            "raw_mse_rank": int(1 + np.argsort(-band_mses).tolist().index(b)),
        })
    region_rows = region_error_rows(f"scene{scene_no:02d}", hs_norm, pred, valid_mask)
    depth_rows = depth_bin_error_rows(
        f"scene{scene_no:02d}", hs_norm, pred, valid_mask, depth_m, min_depth, max_depth
    )
    pixel_rows = quality_bins(hs_norm, pred, valid_mask, "raw")
    for r in pixel_rows:
        r["scene"] = f"scene{scene_no:02d}"
    return row, per_band_rows, region_rows, depth_rows, pixel_rows


def write_csv(path: str, rows: List[Dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--data_root", default="/root/autodl-tmp/Baek数据集")
    parser.add_argument("--scenes", default="1,3,16")
    parser.add_argument("--output_dir", default="/root/autodl-tmp/experiments/scale_metric_patch_diagnosis/gain_corrected_fixed")
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--min_depth", type=float, default=0.4)
    parser.add_argument("--max_depth", type=float, default=2.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"device={device}")
    print(f"ckpt={args.ckpt_path}")
    model = load_model(args.ckpt_path, device)

    rows = []
    per_band_rows = []
    region_rows = []
    depth_rows = []
    pixel_rows = []
    for scene_no in parse_scene_list(args.scenes):
        print(f"\n[evaluate] scene{scene_no:02d}")
        row, pb, rr, dr, qr = evaluate_scene(args, model, scene_no, device)
        rows.append(row)
        per_band_rows.extend(pb)
        region_rows.extend(rr)
        depth_rows.extend(dr)
        pixel_rows.extend(qr)
        print(
            f"  raw={row['raw_psnr_db']:.3f} dB, "
            f"scalar_gain={row['scalar_gain_psnr_db']:.3f} dB "
            f"(gain={row['scalar_gain']:.4f}), "
            f"per_band_gain={row['per_band_gain_psnr_db']:.3f} dB, "
            f"SAM={row['sam_raw_rad']:.4f}"
        )

    write_csv(os.path.join(args.output_dir, "gain_corrected_summary.csv"), rows)
    write_csv(os.path.join(args.output_dir, "gain_corrected_per_band.csv"), per_band_rows)
    write_csv(os.path.join(args.output_dir, "region_error_summary.csv"), region_rows)
    write_csv(os.path.join(args.output_dir, "depth_bin_error_summary.csv"), depth_rows)
    write_csv(os.path.join(args.output_dir, "pixel_quality_bins.csv"), pixel_rows)
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump({"rows": rows}, f, indent=2)
    print(f"\nsaved: {args.output_dir}")


if __name__ == "__main__":
    main()
