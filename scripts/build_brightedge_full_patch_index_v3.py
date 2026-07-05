#!/usr/bin/env python
import argparse
import hashlib
import json
import os
from typing import Dict, List

import numpy as np


def cache_file_path(exr_path: str, cache_dir: str) -> str:
    stat = os.stat(exr_path)
    key_src = f"{exr_path}|{stat.st_mtime_ns}|{stat.st_size}"
    key = hashlib.sha1(key_src.encode("utf-8")).hexdigest()
    return os.path.join(cache_dir, f"{key}.npy")


def metric_to_ips(depth_m: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
    return (max_depth * depth_m - max_depth * min_depth) / (
        (max_depth - min_depth) * depth_m
    )


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


def load_scene(base_dir: str, cache_dir: str, scene_no: int):
    folder = os.path.join(base_dir, f"deploy {scene_no}")
    prefix = os.path.join(folder, f"scene{scene_no:02d}")
    hs_cache = cache_file_path(prefix + "_hs.exr", cache_dir)
    depth_cache = cache_file_path(prefix + "_depth_map.exr", cache_dir)
    if not os.path.exists(hs_cache):
        raise FileNotFoundError(hs_cache)
    if not os.path.exists(depth_cache):
        raise FileNotFoundError(depth_cache)
    return (
        np.load(hs_cache, mmap_mode="r", allow_pickle=False),
        np.load(depth_cache, mmap_mode="r", allow_pickle=False),
    )


def rgb_metrics(hs, depth, top, left, args):
    rows = slice(top, top + args.patch_size, args.rgb_metric_stride)
    cols = slice(left, left + args.patch_size, args.rgb_metric_stride)
    depth_patch = np.asarray(depth[rows, cols], dtype=np.float32)
    if np.nanmax(depth_patch) > 20.0:
        depth_patch = depth_patch / 1000.0
    valid = (
        np.isfinite(depth_patch)
        & (depth_patch > args.min_depth - args.valid_eps)
        & (depth_patch <= args.max_depth + args.valid_eps)
    )
    valid_count = int(np.count_nonzero(valid))
    if valid_count <= 0:
        return 0.0, 0.0
    channels = min(25, int(hs.shape[2]))
    rgb_indices = [idx for idx in (23, 13, 3) if idx < channels]
    if not rgb_indices:
        rgb_indices = list(range(min(3, channels)))
    rgb_sum = None
    for channel in rgb_indices:
        band = np.asarray(hs[rows, cols, channel], dtype=np.float32)
        rgb_sum = band if rgb_sum is None else rgb_sum + band
    luma = np.clip(rgb_sum / (len(rgb_indices) * args.hs_norm_scale), 0.0, 1.0)
    bright = luma >= args.rgb_bright_threshold
    gradient = np.zeros_like(luma, dtype=np.float32)
    gradient[:, 1:] += np.abs(luma[:, 1:] - luma[:, :-1])
    gradient[1:, :] += np.abs(luma[1:, :] - luma[:-1, :])
    bright_edge = bright & (gradient >= args.bright_edge_gradient_threshold)
    return (
        float(np.count_nonzero(bright & valid) / valid_count),
        float(np.count_nonzero(bright_edge & valid) / valid_count),
    )


def append_row(rows: Dict[str, list], scene_id, top, left, score, valid_ratio, center_valid_ratio, depth_range):
    rows["scene_ids"].append(scene_id)
    rows["tops"].append(int(top))
    rows["lefts"].append(int(left))
    rows["scores"].append(float(score))
    rows["valid_ratios"].append(float(valid_ratio))
    rows["center_valid_ratios"].append(float(center_valid_ratio))
    rows["depth_ranges_ips"].append(float(depth_range))
    rows["far_ratios"].append(0.0)
    rows["very_far_ratios"].append(0.0)


def build(args):
    source = np.load(args.input_index, allow_pickle=False)
    rows = {
        "scene_ids": source["scene_ids"].astype(str).tolist(),
        "tops": source["tops"].astype(np.int32).tolist(),
        "lefts": source["lefts"].astype(np.int32).tolist(),
        "scores": source["scores"].astype(np.float32).tolist(),
        "valid_ratios": source["valid_ratios"].astype(np.float32).tolist(),
        "center_valid_ratios": source["center_valid_ratios"].astype(np.float32).tolist(),
        "depth_ranges_ips": source["depth_ranges_ips"].astype(np.float32).tolist(),
        "far_ratios": (
            source["far_ratios"].astype(np.float32).tolist()
            if "far_ratios" in source.files
            else [0.0] * len(source["tops"])
        ),
        "very_far_ratios": (
            source["very_far_ratios"].astype(np.float32).tolist()
            if "very_far_ratios" in source.files
            else [0.0] * len(source["tops"])
        ),
    }
    existing = {
        (str(scene_id), int(top), int(left))
        for scene_id, top, left in zip(rows["scene_ids"], rows["tops"], rows["lefts"])
    }

    scene_stats = []
    for scene_no in range(args.scene_start, args.scene_end + 1):
        scene_id = f"scene_{scene_no:02d}"
        hs, depth = load_scene(args.base_dir, args.exr_cache_dir, scene_no)
        if depth.ndim == 3:
            depth = np.squeeze(depth)
        depth_m = np.asarray(depth, dtype=np.float32)
        if np.nanmax(depth_m) > 20.0:
            depth_m = depth_m / 1000.0
        valid = np.isfinite(depth_m) & (depth_m > args.min_depth - args.valid_eps)
        valid_ii = integral_image(valid)
        depth_safe = np.where(valid, depth_m, args.min_depth)
        depth_ips = np.clip(
            metric_to_ips(depth_safe, args.min_depth, args.max_depth), 0.0, 1.0
        )
        patch_area = args.patch_size * args.patch_size
        center_size = max(1, int(round(args.patch_size * args.center_fraction)))
        center_offset = (args.patch_size - center_size) // 2
        center_area = center_size * center_size

        added = 0
        bright_hits = 0
        edge_hits = 0
        for top in scan_positions(depth_m.shape[0], args.patch_size, args.stride):
            for left in scan_positions(depth_m.shape[1], args.patch_size, args.stride):
                key = (scene_id, int(top), int(left))
                if key in existing:
                    continue
                valid_count = box_sum(valid_ii, top, left, args.patch_size)
                valid_ratio = valid_count / patch_area
                if valid_ratio < args.min_valid_ratio:
                    continue
                center_valid = box_sum(
                    valid_ii,
                    top + center_offset,
                    left + center_offset,
                    center_size,
                )
                center_valid_ratio = center_valid / center_area
                if center_valid_ratio < args.min_center_valid_ratio:
                    continue

                bright_ratio, edge_ratio = rgb_metrics(hs, depth_m, top, left, args)
                bright_ok = bright_ratio >= args.rgb_bright_min_ratio
                edge_ok = edge_ratio >= args.bright_edge_min_ratio
                if not (bright_ok or edge_ok):
                    continue
                patch_valid = valid[top:top + args.patch_size, left:left + args.patch_size]
                valid_depth = depth_ips[top:top + args.patch_size, left:left + args.patch_size][patch_valid]
                if valid_depth.size == 0:
                    continue
                depth_range = float(valid_depth.max() - valid_depth.min())
                score = float(valid_ratio * (1.0 + depth_range))
                append_row(
                    rows,
                    scene_id,
                    top,
                    left,
                    score,
                    valid_ratio,
                    center_valid_ratio,
                    depth_range,
                )
                existing.add(key)
                added += 1
                bright_hits += int(bright_ok)
                edge_hits += int(edge_ok)
        scene_stats.append(
            {
                "scene_id": scene_id,
                "added_brightedge_candidates": added,
                "rgb_bright_hits": bright_hits,
                "bright_edge_hits": edge_hits,
            }
        )
        print(
            f"[{scene_id}] added={added} rgb_bright_hits={bright_hits} "
            f"bright_edge_hits={edge_hits}"
        )

    meta = {}
    if "meta_json" in source.files:
        try:
            meta = json.loads(str(source["meta_json"].item()))
        except Exception:
            meta = {}
    meta.update(
        {
            "version": 3,
            "bright_edge_v3_source_index": os.path.abspath(args.input_index),
            "rgb_bright_threshold": args.rgb_bright_threshold,
            "rgb_bright_min_ratio": args.rgb_bright_min_ratio,
            "bright_edge_gradient_threshold": args.bright_edge_gradient_threshold,
            "bright_edge_min_ratio": args.bright_edge_min_ratio,
            "rgb_metric_stride": args.rgb_metric_stride,
            "brightedge_scene_stats": scene_stats,
        }
    )
    os.makedirs(os.path.dirname(args.output_index), exist_ok=True)
    np.savez_compressed(
        args.output_index,
        scene_ids=np.asarray(rows["scene_ids"], dtype="U16"),
        tops=np.asarray(rows["tops"], dtype=np.int32),
        lefts=np.asarray(rows["lefts"], dtype=np.int32),
        scores=np.asarray(rows["scores"], dtype=np.float32),
        valid_ratios=np.asarray(rows["valid_ratios"], dtype=np.float32),
        center_valid_ratios=np.asarray(rows["center_valid_ratios"], dtype=np.float32),
        depth_ranges_ips=np.asarray(rows["depth_ranges_ips"], dtype=np.float32),
        far_ratios=np.asarray(rows["far_ratios"], dtype=np.float32),
        very_far_ratios=np.asarray(rows["very_far_ratios"], dtype=np.float32),
        meta_json=np.asarray(json.dumps(meta, ensure_ascii=False)),
    )
    with open(args.output_index + ".summary.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)
    print(f"Saved {args.output_index} ({len(rows['tops'])} windows)")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default="/root/autodl-tmp/Baek数据集")
    parser.add_argument("--exr_cache_dir", default="")
    parser.add_argument("--input_index", required=True)
    parser.add_argument("--output_index", required=True)
    parser.add_argument("--scene_start", type=int, default=1)
    parser.add_argument("--scene_end", type=int, default=18)
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--rgb_metric_stride", type=int, default=2)
    parser.add_argument("--min_depth", type=float, default=0.4)
    parser.add_argument("--max_depth", type=float, default=2.0)
    parser.add_argument("--valid_eps", type=float, default=1e-3)
    parser.add_argument("--min_valid_ratio", type=float, default=0.20)
    parser.add_argument("--center_fraction", type=float, default=0.5)
    parser.add_argument("--min_center_valid_ratio", type=float, default=0.10)
    parser.add_argument("--hs_norm_scale", type=float, default=0.9367284796834017)
    parser.add_argument("--rgb_bright_threshold", type=float, default=0.6)
    parser.add_argument("--rgb_bright_min_ratio", type=float, default=0.10)
    parser.add_argument("--bright_edge_gradient_threshold", type=float, default=0.03)
    parser.add_argument("--bright_edge_min_ratio", type=float, default=0.03)
    args = parser.parse_args()
    args.base_dir = os.path.abspath(args.base_dir)
    args.exr_cache_dir = args.exr_cache_dir or os.path.join(args.base_dir, ".exr_cache_npy_v1")
    args.input_index = os.path.abspath(args.input_index)
    args.output_index = os.path.abspath(args.output_index)
    return args


def main():
    build(parse_args())


if __name__ == "__main__":
    main()
