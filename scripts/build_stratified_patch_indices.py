#!/usr/bin/env python
import argparse
import hashlib
import json
import os
import re
from typing import Dict, List, Tuple

import numpy as np


CATEGORY_BITS = {
    "depth_hard": 1,
    "hs_bright": 2,
    "hs_complex": 4,
    "general": 8,
}


def cache_file_path(exr_path: str, cache_dir: str) -> str:
    stat = os.stat(exr_path)
    key_src = f"{exr_path}|{stat.st_mtime_ns}|{stat.st_size}"
    key = hashlib.sha1(key_src.encode("utf-8")).hexdigest()
    return os.path.join(cache_dir, f"{key}.npy")


def read_exr(exr_path: str) -> np.ndarray:
    import Imath
    import OpenEXR

    if not OpenEXR.isOpenExrFile(exr_path):
        raise IOError(f"Not a valid EXR file: {exr_path}")
    exr_file = OpenEXR.InputFile(exr_path)
    header = exr_file.header()
    dw = header["dataWindow"]
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    channel_names = sorted(header["channels"].keys())
    channel_type = header["channels"][channel_names[0]].type
    if channel_type == Imath.PixelType(Imath.PixelType.FLOAT):
        dtype = np.float32
    elif channel_type == Imath.PixelType(Imath.PixelType.HALF):
        dtype = np.float16
    else:
        raise TypeError(f"Unsupported EXR channel type: {channel_type}")
    channels = [
        np.frombuffer(buf, dtype=dtype).reshape(height, width)
        for buf in exr_file.channels(channel_names)
    ]
    return np.stack(channels, axis=-1).astype(np.float32, copy=False)


def load_array(exr_path: str, cache_dir: str) -> np.ndarray:
    cache_path = cache_file_path(exr_path, cache_dir)
    if os.path.exists(cache_path):
        return np.load(cache_path, mmap_mode="r", allow_pickle=False)
    return read_exr(exr_path)


def integral_image(x: np.ndarray) -> np.ndarray:
    return np.pad(x.astype(np.float64, copy=False), ((1, 0), (1, 0))).cumsum(0).cumsum(1)


def box_sum(ii: np.ndarray, top: int, left: int, size: int) -> float:
    bottom = top + size
    right = left + size
    return float(ii[bottom, right] - ii[top, right] - ii[bottom, left] + ii[top, left])


def scan_positions(length: int, patch_size: int, stride: int) -> List[int]:
    positions = list(range(0, max(1, length - patch_size + 1), stride))
    last = max(0, length - patch_size)
    if not positions or positions[-1] != last:
        positions.append(last)
    return positions


def metric_to_ips(depth_m: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
    return (max_depth * depth_m - max_depth * min_depth) / (
        (max_depth - min_depth) * depth_m
    )


def scene_number(scene_id: str) -> int:
    match = re.search(r"\d+", str(scene_id))
    if not match:
        raise ValueError(f"Cannot parse scene number from {scene_id!r}")
    return int(match.group(0))


def spectral_maps(
    hs: np.ndarray,
    valid: np.ndarray,
    hs_channels: int,
    hs_scale: float,
    bright_threshold: float,
    chunk_rows: int = 192,
) -> Dict[str, np.ndarray]:
    height, width = valid.shape
    maps = {
        "mean_sq": np.zeros((height, width), dtype=np.float32),
        "spectral_var": np.zeros((height, width), dtype=np.float32),
        "bright_fraction": np.zeros((height, width), dtype=np.float32),
        "intensity": np.zeros((height, width), dtype=np.float32),
    }
    channels = min(int(hs_channels), int(hs.shape[2]))
    for start in range(0, height, chunk_rows):
        end = min(start + chunk_rows, height)
        block = np.asarray(hs[start:end, :, :channels], dtype=np.float32)
        block = np.clip(block / hs_scale, 0.0, 1.0)
        maps["mean_sq"][start:end] = np.mean(block * block, axis=2)
        maps["spectral_var"][start:end] = np.var(block, axis=2)
        maps["bright_fraction"][start:end] = np.mean(block >= bright_threshold, axis=2)
        maps["intensity"][start:end] = np.mean(block, axis=2)

    intensity = maps.pop("intensity")
    gradient = np.zeros_like(intensity)
    gradient[:, 1:] += np.abs(intensity[:, 1:] - intensity[:, :-1])
    gradient[1:, :] += np.abs(intensity[1:, :] - intensity[:-1, :])
    maps["spatial_gradient"] = gradient
    valid_f = valid.astype(np.float32, copy=False)
    for key in maps:
        maps[key] *= valid_f
    return maps


def build_scene_candidates(args, scene_no: int) -> Tuple[Dict[str, np.ndarray], Dict]:
    folder = os.path.join(args.base_dir, f"deploy {scene_no}")
    prefix = os.path.join(folder, f"scene{scene_no:02d}")
    hs = load_array(prefix + "_hs.exr", args.exr_cache_dir)
    depth = load_array(prefix + "_depth_map.exr", args.exr_cache_dir)
    if depth.ndim == 3:
        depth = np.squeeze(depth)
    depth_m = np.asarray(depth, dtype=np.float32) / 1000.0
    valid = np.isfinite(depth_m) & (depth_m > args.min_depth - args.valid_eps)
    valid_ii = integral_image(valid)

    depth_safe = np.where(valid, depth_m, args.min_depth)
    depth_ips = np.clip(
        metric_to_ips(depth_safe, args.min_depth, args.max_depth), 0.0, 1.0
    ).astype(np.float32)
    maps = spectral_maps(
        hs,
        valid,
        args.hs_channels,
        args.hs_norm_scale,
        args.bright_value_threshold,
    )
    map_integrals = {key: integral_image(value) for key, value in maps.items()}

    patch_area = args.patch_size * args.patch_size
    center_size = max(1, int(round(args.patch_size * args.center_fraction)))
    center_offset = (args.patch_size - center_size) // 2
    center_area = center_size * center_size
    rows = {
        "scene_ids": [],
        "tops": [],
        "lefts": [],
        "scores": [],
        "valid_ratios": [],
        "center_valid_ratios": [],
        "depth_ranges_ips": [],
        "hs_rms": [],
        "hs_spectral_var": [],
        "hs_spatial_gradient": [],
        "hs_bright_ratios": [],
    }

    for top in scan_positions(depth_m.shape[0], args.patch_size, args.stride):
        for left in scan_positions(depth_m.shape[1], args.patch_size, args.stride):
            valid_count = box_sum(valid_ii, top, left, args.patch_size)
            valid_ratio = valid_count / patch_area
            if valid_ratio < args.min_valid_ratio:
                continue
            center_valid = box_sum(
                valid_ii, top + center_offset, left + center_offset, center_size
            )
            center_valid_ratio = center_valid / center_area
            if center_valid_ratio < args.min_center_valid_ratio:
                continue

            patch_valid = valid[
                top:top + args.patch_size,
                left:left + args.patch_size,
            ]
            valid_depth = depth_ips[
                top:top + args.patch_size,
                left:left + args.patch_size,
            ][patch_valid]
            if valid_depth.size == 0:
                continue
            depth_range = float(valid_depth.max() - valid_depth.min())
            denom = max(valid_count, 1.0)
            mean_sq = box_sum(map_integrals["mean_sq"], top, left, args.patch_size) / denom
            hs_rms = float(np.sqrt(max(mean_sq, 0.0)))
            spectral_var = (
                box_sum(map_integrals["spectral_var"], top, left, args.patch_size) / denom
            )
            spatial_gradient = (
                box_sum(map_integrals["spatial_gradient"], top, left, args.patch_size) / denom
            )
            bright_ratio = (
                box_sum(map_integrals["bright_fraction"], top, left, args.patch_size) / denom
            )
            rows["scene_ids"].append(f"scene_{scene_no:02d}")
            rows["tops"].append(top)
            rows["lefts"].append(left)
            rows["valid_ratios"].append(valid_ratio)
            rows["center_valid_ratios"].append(center_valid_ratio)
            rows["depth_ranges_ips"].append(depth_range)
            rows["hs_rms"].append(hs_rms)
            rows["hs_spectral_var"].append(spectral_var)
            rows["hs_spatial_gradient"].append(spatial_gradient)
            rows["hs_bright_ratios"].append(bright_ratio)
            rows["scores"].append(valid_ratio * (1.0 + depth_range))

    arrays = {key: np.asarray(value) for key, value in rows.items()}
    if arrays["tops"].size == 0:
        raise RuntimeError(f"No common foreground candidates for scene {scene_no}")

    bright_rms_threshold = float(np.percentile(arrays["hs_rms"], args.hs_bright_percentile))
    spectral_var_threshold = float(
        np.percentile(arrays["hs_spectral_var"], args.hs_complex_percentile)
    )
    spatial_gradient_threshold = float(
        np.percentile(arrays["hs_spatial_gradient"], args.hs_complex_percentile)
    )
    category_masks = np.full(arrays["tops"].shape, CATEGORY_BITS["general"], dtype=np.uint8)
    category_masks[arrays["depth_ranges_ips"] >= args.depth_hard_min_ips] |= CATEGORY_BITS[
        "depth_hard"
    ]
    bright = (
        (arrays["hs_rms"] >= bright_rms_threshold)
        | (arrays["hs_bright_ratios"] >= args.absolute_bright_ratio)
    )
    category_masks[bright] |= CATEGORY_BITS["hs_bright"]
    complex_mask = (
        (arrays["hs_spectral_var"] >= spectral_var_threshold)
        | (arrays["hs_spatial_gradient"] >= spatial_gradient_threshold)
    )
    category_masks[complex_mask] |= CATEGORY_BITS["hs_complex"]
    arrays["category_masks"] = category_masks

    counts = {
        name: int(np.count_nonzero(category_masks & bit))
        for name, bit in CATEGORY_BITS.items()
    }
    stats = {
        "scene_id": f"scene_{scene_no:02d}",
        "common_candidates": int(arrays["tops"].size),
        "category_counts": counts,
        "thresholds": {
            "hs_rms_bright": bright_rms_threshold,
            "hs_spectral_var_complex": spectral_var_threshold,
            "hs_spatial_gradient_complex": spatial_gradient_threshold,
        },
    }
    return arrays, stats


def concatenate(rows: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    keys = rows[0].keys()
    return {key: np.concatenate([row[key] for row in rows]) for key in keys}


def save_index(path: str, arrays: Dict[str, np.ndarray], meta: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        scene_ids=arrays["scene_ids"].astype("U16"),
        tops=arrays["tops"].astype(np.int32),
        lefts=arrays["lefts"].astype(np.int32),
        scores=arrays["scores"].astype(np.float32),
        valid_ratios=arrays["valid_ratios"].astype(np.float32),
        center_valid_ratios=arrays["center_valid_ratios"].astype(np.float32),
        depth_ranges_ips=arrays["depth_ranges_ips"].astype(np.float32),
        hs_rms=arrays["hs_rms"].astype(np.float32),
        hs_spectral_var=arrays["hs_spectral_var"].astype(np.float32),
        hs_spatial_gradient=arrays["hs_spatial_gradient"].astype(np.float32),
        hs_bright_ratios=arrays["hs_bright_ratios"].astype(np.float32),
        category_masks=arrays["category_masks"].astype(np.uint8),
        category_names=np.asarray(list(CATEGORY_BITS), dtype="U24"),
        category_bits=np.asarray(list(CATEGORY_BITS.values()), dtype=np.uint8),
        meta_json=np.asarray(json.dumps(meta, ensure_ascii=False)),
    )
    with open(path + ".summary.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)


def build_balanced_validation(args) -> Tuple[Dict[str, np.ndarray], Dict]:
    source = np.load(args.validation_source_index, allow_pickle=False)
    source_ids = source["scene_ids"].astype(str)
    rng = np.random.default_rng(args.seed)
    selected = []
    counts = {}
    for scene_no in range(args.val_scene_start, args.val_scene_end + 1):
        scene_id = f"scene_{scene_no:02d}"
        candidates = np.flatnonzero(source_ids == scene_id)
        if candidates.size < args.val_patches_per_scene:
            raise RuntimeError(
                f"{scene_id} has {candidates.size} validation candidates, "
                f"needs {args.val_patches_per_scene}"
            )
        picks = np.sort(
            rng.choice(candidates, size=args.val_patches_per_scene, replace=False)
        )
        selected.append(picks)
        counts[scene_id] = int(picks.size)
    selected_idx = np.concatenate(selected)
    size = selected_idx.size
    arrays = {
        "scene_ids": source_ids[selected_idx],
        "tops": source["tops"][selected_idx],
        "lefts": source["lefts"][selected_idx],
        "scores": source["scores"][selected_idx] if "scores" in source else np.ones(size),
        "valid_ratios": (
            source["valid_ratios"][selected_idx]
            if "valid_ratios" in source
            else np.zeros(size)
        ),
        "center_valid_ratios": (
            source["center_valid_ratios"][selected_idx]
            if "center_valid_ratios" in source
            else np.zeros(size)
        ),
        "depth_ranges_ips": (
            source["depth_ranges_ips"][selected_idx]
            if "depth_ranges_ips" in source
            else np.zeros(size)
        ),
        "hs_rms": np.zeros(size),
        "hs_spectral_var": np.zeros(size),
        "hs_spatial_gradient": np.zeros(size),
        "hs_bright_ratios": np.zeros(size),
        "category_masks": np.full(size, CATEGORY_BITS["general"], dtype=np.uint8),
    }
    meta = {
        "version": 2,
        "index_type": "balanced_validation",
        "source_index": os.path.abspath(args.validation_source_index),
        "seed": args.seed,
        "patch_size": args.patch_size,
        "min_valid_ratio": args.min_valid_ratio,
        "min_center_valid_ratio": args.min_center_valid_ratio,
        "min_depth_range_ips": args.depth_hard_min_ips,
        "patches_per_scene": args.val_patches_per_scene,
        "scene_counts": counts,
    }
    return arrays, meta


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build stratified HS training and scene-balanced validation patch indices."
    )
    parser.add_argument("--base_dir", default="/root/autodl-tmp/Baek数据集")
    parser.add_argument("--exr_cache_dir", default="")
    parser.add_argument("--train_output", required=True)
    parser.add_argument("--val_output", required=True)
    parser.add_argument("--validation_source_index", required=True)
    parser.add_argument("--train_scene_start", type=int, default=1)
    parser.add_argument("--train_scene_end", type=int, default=15)
    parser.add_argument("--val_scene_start", type=int, default=16)
    parser.add_argument("--val_scene_end", type=int, default=18)
    parser.add_argument("--val_patches_per_scene", type=int, default=341)
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--min_depth", type=float, default=0.4)
    parser.add_argument("--max_depth", type=float, default=2.0)
    parser.add_argument("--valid_eps", type=float, default=1e-3)
    parser.add_argument("--min_valid_ratio", type=float, default=0.20)
    parser.add_argument("--center_fraction", type=float, default=0.5)
    parser.add_argument("--min_center_valid_ratio", type=float, default=0.10)
    parser.add_argument("--depth_hard_min_ips", type=float, default=0.06)
    parser.add_argument("--hs_channels", type=int, default=25)
    parser.add_argument("--hs_norm_scale", type=float, default=0.9367284796834017)
    parser.add_argument("--hs_bright_percentile", type=float, default=80.0)
    parser.add_argument("--hs_complex_percentile", type=float, default=80.0)
    parser.add_argument("--bright_value_threshold", type=float, default=0.8)
    parser.add_argument("--absolute_bright_ratio", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.base_dir = os.path.abspath(args.base_dir)
    args.exr_cache_dir = args.exr_cache_dir or os.path.join(
        args.base_dir, ".exr_cache_npy_v1"
    )
    args.train_output = os.path.abspath(args.train_output)
    args.val_output = os.path.abspath(args.val_output)
    args.validation_source_index = os.path.abspath(args.validation_source_index)
    return args


def main():
    args = parse_args()
    for path in (args.train_output, args.val_output):
        if os.path.exists(path) and not args.force:
            raise FileExistsError(f"Output exists: {path}; pass --force to overwrite")

    scene_rows = []
    scene_stats = []
    for scene_no in range(args.train_scene_start, args.train_scene_end + 1):
        arrays, stats = build_scene_candidates(args, scene_no)
        scene_rows.append(arrays)
        scene_stats.append(stats)
        counts = stats["category_counts"]
        print(
            f"[{stats['scene_id']}] common={stats['common_candidates']} "
            f"depth={counts['depth_hard']} bright={counts['hs_bright']} "
            f"complex={counts['hs_complex']} general={counts['general']}"
        )

    train_arrays = concatenate(scene_rows)
    train_meta = {
        "version": 2,
        "index_type": "stratified_training",
        "base_dir": args.base_dir,
        "patch_size": args.patch_size,
        "stride": args.stride,
        "min_depth": args.min_depth,
        "max_depth": args.max_depth,
        "valid_eps": args.valid_eps,
        "min_valid_ratio": args.min_valid_ratio,
        "min_center_valid_ratio": args.min_center_valid_ratio,
        "min_depth_range_ips": 0.0,
        "depth_hard_min_ips": args.depth_hard_min_ips,
        "hs_norm_scale": args.hs_norm_scale,
        "hs_bright_percentile": args.hs_bright_percentile,
        "hs_complex_percentile": args.hs_complex_percentile,
        "bright_value_threshold": args.bright_value_threshold,
        "absolute_bright_ratio": args.absolute_bright_ratio,
        "category_bits": CATEGORY_BITS,
        "scene_stats": scene_stats,
    }
    save_index(args.train_output, train_arrays, train_meta)

    val_arrays, val_meta = build_balanced_validation(args)
    save_index(args.val_output, val_arrays, val_meta)
    print(f"Saved train index: {args.train_output} ({len(train_arrays['tops'])} windows)")
    print(f"Saved val index:   {args.val_output} ({len(val_arrays['tops'])} windows)")


if __name__ == "__main__":
    main()
