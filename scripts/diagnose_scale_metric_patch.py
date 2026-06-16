#!/usr/bin/env python
"""Offline diagnostics for HS scale, PSNR aggregation, and patch-index coverage."""

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def cache_file_path(exr_path: str, cache_dir: str) -> str:
    stat = os.stat(exr_path)
    key_src = f"{exr_path}|{stat.st_mtime_ns}|{stat.st_size}"
    key = hashlib.sha1(key_src.encode("utf-8")).hexdigest()
    return os.path.join(cache_dir, f"{key}.npy")


def read_exr(file_path: str, cache_dir: str = "") -> np.ndarray:
    if cache_dir:
        cache_path = cache_file_path(file_path, cache_dir)
        if os.path.exists(cache_path):
            return np.load(cache_path, allow_pickle=False).astype(np.float32, copy=False)

    import Imath
    import OpenEXR

    if not OpenEXR.isOpenExrFile(file_path):
        raise IOError(f"Invalid EXR file: {file_path}")
    exr_file = OpenEXR.InputFile(file_path)
    header = exr_file.header()
    dw = header["dataWindow"]
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    channel_names = sorted(header["channels"].keys())
    if not channel_names:
        raise ValueError(f"No channels found in EXR: {file_path}")
    first_type = header["channels"][channel_names[0]].type
    if first_type == Imath.PixelType(Imath.PixelType.FLOAT):
        dtype = np.float32
    elif first_type == Imath.PixelType(Imath.PixelType.HALF):
        dtype = np.float16
    else:
        raise TypeError(f"Unsupported EXR pixel type: {first_type}")
    channels = exr_file.channels(channel_names)
    arr = [
        np.frombuffer(channels[i], dtype=dtype).reshape(height, width)
        for i in range(len(channel_names))
    ]
    return np.stack(arr, axis=-1).astype(np.float32, copy=False)


def select_hs_bands(hs_cube: np.ndarray, expected_channels: int) -> np.ndarray:
    if hs_cube.ndim != 3:
        raise ValueError(f"Expected HxWxC cube, got shape={hs_cube.shape}")
    if hs_cube.shape[2] < expected_channels:
        raise ValueError(
            f"Hyperspectral channel count too small: got={hs_cube.shape[2]}, "
            f"expected={expected_channels}"
        )
    return hs_cube[:, :, :expected_channels]


def parse_scene_list(text: str) -> List[int]:
    scenes = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            scenes.extend(range(int(a), int(b) + 1))
        else:
            scenes.append(int(part))
    return sorted(dict.fromkeys(scenes))


def scene_id(scene_no: int) -> str:
    return f"scene_{scene_no:02d}"


def scene_name(scene_no: int) -> str:
    return f"scene{scene_no:02d}"


def scene_paths(data_root: str, scene_no: int) -> Tuple[str, str]:
    folder = os.path.join(data_root, f"deploy {scene_no}")
    hs_path = os.path.join(folder, f"scene{scene_no:02d}_hs.exr")
    depth_path = os.path.join(folder, f"scene{scene_no:02d}_depth_map.exr")
    if not os.path.exists(hs_path):
        raise FileNotFoundError(hs_path)
    if not os.path.exists(depth_path):
        raise FileNotFoundError(depth_path)
    return hs_path, depth_path


def finite_hs_values(hs: np.ndarray, mask: np.ndarray, sanity_threshold: float) -> np.ndarray:
    vals = hs[mask > 0.5]
    vals = vals[np.isfinite(vals)]
    vals = vals[(vals >= 0.0) & (vals < sanity_threshold)]
    return vals.astype(np.float32, copy=False)


def psnr_from_mse(mse: float, data_range: float = 1.0) -> float:
    if mse <= 0.0:
        return 100.0
    return float(10.0 * math.log10((data_range * data_range) / mse))


def compute_scene_scale_stats(args, scene_no: int) -> Tuple[Dict, List[Dict]]:
    hs_path, depth_path = scene_paths(args.data_root, scene_no)
    hs = select_hs_bands(read_exr(hs_path, args.exr_cache_dir), args.hs_channels).astype(np.float32, copy=False)
    depth = read_exr(depth_path, args.exr_cache_dir)
    if depth.ndim == 3:
        depth = depth.squeeze(-1)
    depth_m = depth.astype(np.float32, copy=False) / 1000.0
    mask = (depth_m > (args.min_depth - args.valid_eps)).astype(np.float32)

    valid_vals = finite_hs_values(hs, mask, args.hs_sanity_threshold)
    if valid_vals.size == 0:
        raise RuntimeError(f"No valid HS values for scene {scene_no}.")

    scene_max = float(np.max(valid_vals))
    fixed = float(args.fixed_scale)
    scene_row = {
        "scene": scene_name(scene_no),
        "valid_ratio": float(mask.mean()),
        "hs_valid_min": float(np.min(valid_vals)),
        "hs_valid_max": scene_max,
        "hs_valid_mean": float(np.mean(valid_vals)),
        "hs_valid_std": float(np.std(valid_vals)),
        "hs_valid_p99": float(np.percentile(valid_vals, 99.0)),
        "hs_valid_p999": float(np.percentile(valid_vals, 99.9)),
        "hs_valid_p9999": float(np.percentile(valid_vals, 99.99)),
        "fixed_scale": fixed,
        "fixed_scale_over_scene_p999": fixed / max(float(np.percentile(valid_vals, 99.9)), 1e-12),
        "valid_clip_frac_fixed": float(np.mean(valid_vals > fixed)),
        "valid_near_zero_frac_fixed_norm": float(np.mean((valid_vals / max(fixed, 1e-12)) < 1e-3)),
        "scene_max_scale": scene_max,
    }

    band_rows = []
    valid_mask = mask > 0.5
    for b in range(hs.shape[2]):
        vals = hs[:, :, b][valid_mask]
        vals = vals[np.isfinite(vals)]
        vals = vals[(vals >= 0.0) & (vals < args.hs_sanity_threshold)]
        if vals.size == 0:
            continue
        fixed_norm = np.clip(vals / max(fixed, 1e-12), 0.0, 1.0)
        scene_norm = vals / max(scene_max, 1e-12)
        band_rows.append({
            "scene": scene_name(scene_no),
            "band": b,
            "raw_min": float(np.min(vals)),
            "raw_max": float(np.max(vals)),
            "raw_mean": float(np.mean(vals)),
            "raw_std": float(np.std(vals)),
            "raw_p99": float(np.percentile(vals, 99.0)),
            "raw_p999": float(np.percentile(vals, 99.9)),
            "fixed_norm_mean": float(np.mean(fixed_norm)),
            "fixed_norm_std": float(np.std(fixed_norm)),
            "scene_norm_mean": float(np.mean(scene_norm)),
            "scene_norm_std": float(np.std(scene_norm)),
            "clip_frac_fixed": float(np.mean(vals > fixed)),
        })
    return scene_row, band_rows


def read_csv_dict(path: str) -> List[Dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def parse_inference_root(spec: str) -> Tuple[str, str]:
    if "=" not in spec:
        label = Path(spec).name
        return label, spec
    label, path = spec.split("=", 1)
    return label.strip(), path.strip()


def load_metrics_real(root: str) -> Dict[str, Dict]:
    path = os.path.join(root, "metrics_real.txt")
    if not os.path.exists(path):
        return {}
    rows = read_csv_dict(path)
    return {row["scene"]: row for row in rows}


def load_scene_extra(root: str, scene: str) -> Dict:
    scene_dir = os.path.join(root, scene)
    extra = {}
    per_band_path = os.path.join(scene_dir, "metrics_per_band.csv")
    if os.path.exists(per_band_path):
        rows = read_csv_dict(per_band_path)
        psnrs = np.array([float(r["psnr_masked_db"]) for r in rows], dtype=np.float64)
        extra.update({
            "per_band_psnr_min": float(np.min(psnrs)),
            "per_band_psnr_mean": float(np.mean(psnrs)),
            "per_band_psnr_last5_mean": float(np.mean(psnrs[-5:])),
            "per_band_psnr_first5_mean": float(np.mean(psnrs[:5])),
            "per_band_psnr_argmin": int(np.argmin(psnrs)),
        })
    meas_path = os.path.join(scene_dir, "measurement_stats_summary.json")
    if os.path.exists(meas_path):
        with open(meas_path) as f:
            ms = json.load(f)
        for key in ("after_norm_mean_std", "after_norm_median_std", "after_norm_zero_ratio"):
            if key in ms:
                extra[f"measurement_{key}"] = ms[key]
    diag_path = os.path.join(scene_dir, "diagnostic_metrics.json")
    if os.path.exists(diag_path):
        with open(diag_path) as f:
            diag = json.load(f)
        extra["stitch_coverage_ratio"] = diag.get("stitch_coverage_ratio")
        extra["skipped_tiles"] = diag.get("skipped_tiles")
        region = diag.get("region") or {}
        for key in ("hs_psnr_masked_boundary", "hs_psnr_masked_interior", "hs_psnr_full_bg"):
            if key in region:
                extra[key] = region[key]
    return extra


def collect_inference_rows(specs: Iterable[str]) -> List[Dict]:
    rows = []
    for spec in specs:
        label, root = parse_inference_root(spec)
        metrics = load_metrics_real(root)
        for scene, row in metrics.items():
            out = {"label": label, "root": root, "scene": scene}
            for key, val in row.items():
                if key == "scene":
                    continue
                try:
                    out[key] = float(val)
                except ValueError:
                    out[key] = val
            out.update(load_scene_extra(root, scene))
            rows.append(out)
    return rows


def compute_patch_coverage(args) -> Tuple[List[Dict], Dict]:
    if not args.patch_index_path:
        return [], {}
    data = np.load(args.patch_index_path, allow_pickle=False)
    ids = data["scene_ids"].astype(str)
    scores = data["scores"].astype(np.float64) if "scores" in data.files else np.ones(len(ids), dtype=np.float64)
    counts = Counter(ids.tolist())
    train_scenes = [scene_id(i) for i in range(1, 16)]
    val_scenes = [scene_id(i) for i in range(16, 19)]
    samples_per_scene_per_epoch = args.train_samples_per_epoch / max(len(train_scenes), 1)

    rows = []
    for sid in sorted(counts, key=lambda s: int(re.sub(r"\D", "", s) or "0")):
        idx = np.nonzero(ids == sid)[0]
        n = len(idx)
        split = "train" if sid in train_scenes else "val" if sid in val_scenes else "other"
        draws_12 = samples_per_scene_per_epoch * args.coverage_epochs if split == "train" else 0.0
        draws_22 = samples_per_scene_per_epoch * args.coverage_epochs_extended if split == "train" else 0.0
        p_uniform = np.full(n, 1.0 / max(n, 1), dtype=np.float64)
        scene_scores = np.maximum(scores[idx], 1e-12)
        p_weighted = scene_scores / scene_scores.sum()

        def expected_unique(probs: np.ndarray, draws: float) -> float:
            if draws <= 0 or probs.size == 0:
                return 0.0
            return float(np.sum(1.0 - np.power(1.0 - probs, draws)))

        rows.append({
            "scene_id": sid,
            "split": split,
            "candidate_count": n,
            "score_min": float(scene_scores.min()),
            "score_max": float(scene_scores.max()),
            "score_cv": float(scene_scores.std() / max(scene_scores.mean(), 1e-12)),
            "draws_per_scene_at_epochs": float(draws_12),
            "expected_unique_uniform": expected_unique(p_uniform, draws_12),
            "expected_unique_uniform_frac": expected_unique(p_uniform, draws_12) / max(n, 1),
            "expected_unique_weighted": expected_unique(p_weighted, draws_12),
            "expected_unique_weighted_frac": expected_unique(p_weighted, draws_12) / max(n, 1),
            "draws_per_scene_extended": float(draws_22),
            "expected_unique_weighted_frac_extended": expected_unique(p_weighted, draws_22) / max(n, 1),
        })

    train_counts = [r["candidate_count"] for r in rows if r["split"] == "train"]
    summary = {
        "patch_index_path": args.patch_index_path,
        "total_candidates": int(len(ids)),
        "train_candidates": int(sum(train_counts)),
        "val_candidates": int(sum(r["candidate_count"] for r in rows if r["split"] == "val")),
        "train_samples_per_epoch": int(args.train_samples_per_epoch),
        "samples_per_train_scene_per_epoch": samples_per_scene_per_epoch,
        "min_train_candidates_per_scene": int(min(train_counts)) if train_counts else 0,
        "max_train_candidates_per_scene": int(max(train_counts)) if train_counts else 0,
    }
    return rows, summary


def write_csv(path: str, rows: List[Dict]) -> None:
    if not rows:
        return
    keys = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", default="/root/autodl-tmp/Baek数据集")
    parser.add_argument("--exr_cache_dir", default="/root/autodl-tmp/Baek数据集/.exr_cache_npy_v1")
    parser.add_argument("--scenes", default="1,3,8,16,17")
    parser.add_argument("--hs_channels", type=int, default=25)
    parser.add_argument("--fixed_scale", type=float, default=0.9367284796834017)
    parser.add_argument("--hs_sanity_threshold", type=float, default=10000.0)
    parser.add_argument("--min_depth", type=float, default=0.4)
    parser.add_argument("--valid_eps", type=float, default=1e-3)
    parser.add_argument("--patch_index_path", default="/root/autodl-tmp/Baek数据集/.patch_index/patch128_stride32_valid20_range060_center10_far140b20_vfar160b30_v1.npz")
    parser.add_argument("--train_samples_per_epoch", type=int, default=1530)
    parser.add_argument("--coverage_epochs", type=int, default=12)
    parser.add_argument("--coverage_epochs_extended", type=int, default=22)
    parser.add_argument("--inference_root", action="append", default=[],
                        help="label=/path/to/inference_root. Can be repeated.")
    parser.add_argument("--output_dir", default="/root/autodl-tmp/experiments/scale_metric_patch_diagnosis")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    scene_rows = []
    band_rows = []
    for scene_no in parse_scene_list(args.scenes):
        print(f"[scale] scene {scene_no}")
        scene_row, band = compute_scene_scale_stats(args, scene_no)
        scene_rows.append(scene_row)
        band_rows.extend(band)

    inference_rows = collect_inference_rows(args.inference_root)
    patch_rows, patch_summary = compute_patch_coverage(args)

    write_csv(os.path.join(args.output_dir, "scene_scale_stats.csv"), scene_rows)
    write_csv(os.path.join(args.output_dir, "band_scale_stats.csv"), band_rows)
    write_csv(os.path.join(args.output_dir, "inference_metrics_summary.csv"), inference_rows)
    write_csv(os.path.join(args.output_dir, "patch_index_coverage.csv"), patch_rows)

    summary = {
        "scene_scale_stats_csv": os.path.join(args.output_dir, "scene_scale_stats.csv"),
        "band_scale_stats_csv": os.path.join(args.output_dir, "band_scale_stats.csv"),
        "inference_metrics_summary_csv": os.path.join(args.output_dir, "inference_metrics_summary.csv"),
        "patch_index_coverage_csv": os.path.join(args.output_dir, "patch_index_coverage.csv"),
        "patch_summary": patch_summary,
        "inference_roots": args.inference_root,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
