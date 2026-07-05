#!/usr/bin/env python
import argparse
import hashlib
import json
import os
import re
import shutil
from typing import Dict, Iterable, Tuple

import numpy as np


def cache_file_path(exr_path: str, cache_dir: str) -> str:
    stat = os.stat(exr_path)
    key_src = f"{exr_path}|{stat.st_mtime_ns}|{stat.st_size}"
    key = hashlib.sha1(key_src.encode("utf-8")).hexdigest()
    return os.path.join(cache_dir, f"{key}.npy")


def scene_number(scene_id: str) -> int:
    match = re.search(r"\d+", str(scene_id))
    if not match:
        raise ValueError(f"Cannot parse scene number from {scene_id!r}")
    return int(match.group(0))


def load_cached_scene(base_dir: str, cache_dir: str, scene_no: int) -> Tuple[np.ndarray, np.ndarray]:
    folder = os.path.join(base_dir, f"deploy {scene_no}")
    prefix = os.path.join(folder, f"scene{scene_no:02d}")
    hs_path = cache_file_path(prefix + "_hs.exr", cache_dir)
    depth_path = cache_file_path(prefix + "_depth_map.exr", cache_dir)
    if not os.path.exists(hs_path):
        raise FileNotFoundError(f"Missing HS cache: {hs_path}")
    if not os.path.exists(depth_path):
        raise FileNotFoundError(f"Missing depth cache: {depth_path}")
    return (
        np.load(hs_path, mmap_mode="r", allow_pickle=False),
        np.load(depth_path, mmap_mode="r", allow_pickle=False),
    )


def iter_scene_indices(scene_ids: np.ndarray) -> Iterable[Tuple[str, np.ndarray]]:
    for scene_id in sorted(set(scene_ids.tolist())):
        yield scene_id, np.flatnonzero(scene_ids == scene_id)


def compute_rgb_metrics_for_scene(
    hs: np.ndarray,
    depth: np.ndarray,
    tops: np.ndarray,
    lefts: np.ndarray,
    patch_size: int,
    stride: int,
    hs_norm_scale: float,
    rgb_bright_threshold: float,
    edge_gradient_threshold: float,
    min_depth: float,
    max_depth: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if depth.ndim == 3:
        depth = np.squeeze(depth)
    channels = min(25, int(hs.shape[2]))
    rgb_indices = [idx for idx in (23, 13, 3) if idx < channels]
    if not rgb_indices:
        rgb_indices = list(range(min(3, channels)))

    rgb_bright_ratios = np.zeros(tops.shape, dtype=np.float32)
    rgb_spatial_gradient = np.zeros(tops.shape, dtype=np.float32)
    rgb_bright_edge_ratios = np.zeros(tops.shape, dtype=np.float32)

    for out_i, (top, left) in enumerate(zip(tops, lefts)):
        top = int(top)
        left = int(left)
        rows = slice(top, top + patch_size, stride)
        cols = slice(left, left + patch_size, stride)

        depth_patch = np.asarray(depth[rows, cols], dtype=np.float32)
        if np.nanmax(depth_patch) > 20.0:
            depth_patch = depth_patch / 1000.0
        valid = (
            np.isfinite(depth_patch)
            & (depth_patch > min_depth - 1e-3)
            & (depth_patch <= max_depth + 1e-3)
        )
        valid_count = int(np.count_nonzero(valid))
        if valid_count <= 0:
            continue

        rgb_sum = None
        for channel in rgb_indices:
            band = np.asarray(hs[rows, cols, channel], dtype=np.float32)
            rgb_sum = band if rgb_sum is None else rgb_sum + band
        luma = np.clip(rgb_sum / (len(rgb_indices) * hs_norm_scale), 0.0, 1.0)
        bright = luma >= rgb_bright_threshold

        gradient = np.zeros_like(luma, dtype=np.float32)
        gradient[:, 1:] += np.abs(luma[:, 1:] - luma[:, :-1])
        gradient[1:, :] += np.abs(luma[1:, :] - luma[:-1, :])
        bright_edge = bright & (gradient >= edge_gradient_threshold)

        rgb_bright_ratios[out_i] = float(np.count_nonzero(bright & valid) / valid_count)
        rgb_spatial_gradient[out_i] = float(np.sum(gradient[valid]) / valid_count)
        rgb_bright_edge_ratios[out_i] = float(
            np.count_nonzero(bright_edge & valid) / valid_count
        )

    return rgb_bright_ratios, rgb_spatial_gradient, rgb_bright_edge_ratios


def load_meta(data: np.lib.npyio.NpzFile) -> Dict:
    if "meta_json" not in data.files:
        return {}
    try:
        return json.loads(str(data["meta_json"].item()))
    except Exception:
        return {}


def save_npz_like(path: str, arrays: Dict[str, np.ndarray], meta: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arrays = dict(arrays)
    arrays["meta_json"] = np.asarray(json.dumps(meta, ensure_ascii=False))
    np.savez_compressed(path, **arrays)
    with open(path + ".summary.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)


def build_train(args) -> None:
    data = np.load(args.input_train, allow_pickle=False)
    arrays = {name: data[name] for name in data.files if name != "meta_json"}
    meta = load_meta(data)
    scene_ids = arrays["scene_ids"].astype(str)
    tops = arrays["tops"].astype(np.int64)
    lefts = arrays["lefts"].astype(np.int64)
    category_names = arrays["category_names"].astype(str).tolist()
    category_bits = arrays["category_bits"].astype(np.uint8).tolist()
    bits = {name: int(bit) for name, bit in zip(category_names, category_bits)}
    required = {"hs_bright", "hs_complex", "general"}
    missing = required - set(bits)
    if missing:
        raise ValueError(f"Input patch index is missing categories: {sorted(missing)}")

    rgb_bright_all = np.zeros(tops.shape, dtype=np.float32)
    rgb_grad_all = np.zeros(tops.shape, dtype=np.float32)
    rgb_edge_all = np.zeros(tops.shape, dtype=np.float32)

    for scene_id, idx in iter_scene_indices(scene_ids):
        scene_no = scene_number(scene_id)
        hs, depth = load_cached_scene(args.base_dir, args.exr_cache_dir, scene_no)
        bright, grad, edge = compute_rgb_metrics_for_scene(
            hs,
            depth,
            tops[idx],
            lefts[idx],
            args.patch_size,
            args.rgb_metric_stride,
            args.hs_norm_scale,
            args.rgb_bright_threshold,
            args.bright_edge_gradient_threshold,
            args.min_depth,
            args.max_depth,
        )
        rgb_bright_all[idx] = bright
        rgb_grad_all[idx] = grad
        rgb_edge_all[idx] = edge
        print(
            f"[{scene_id}] windows={idx.size} "
            f"rgb_bright>=min={int(np.count_nonzero(bright >= args.rgb_bright_min_ratio))} "
            f"bright_edge>=min={int(np.count_nonzero(edge >= args.bright_edge_min_ratio))}"
        )

    category_masks = arrays["category_masks"].astype(np.uint8).copy()
    bright_mask = rgb_bright_all >= args.rgb_bright_min_ratio
    bright_edge_mask = rgb_edge_all >= args.bright_edge_min_ratio
    category_masks[bright_mask | bright_edge_mask] |= np.uint8(bits["hs_bright"])
    category_masks[bright_edge_mask] |= np.uint8(bits["hs_complex"])

    arrays["category_masks"] = category_masks
    arrays["rgb_bright_ratios"] = rgb_bright_all
    arrays["rgb_spatial_gradient"] = rgb_grad_all
    arrays["rgb_bright_edge_ratios"] = rgb_edge_all

    scene_stats = []
    for scene_id, idx in iter_scene_indices(scene_ids):
        masks = category_masks[idx]
        counts = {name: int(np.count_nonzero(masks & bit)) for name, bit in bits.items()}
        counts["rgb_bright_added"] = int(np.count_nonzero(bright_mask[idx]))
        counts["bright_edge_added"] = int(np.count_nonzero(bright_edge_mask[idx]))
        scene_stats.append(
            {
                "scene_id": scene_id,
                "common_candidates": int(idx.size),
                "category_counts": counts,
                "thresholds": {
                    "rgb_bright_threshold": args.rgb_bright_threshold,
                    "rgb_bright_min_ratio": args.rgb_bright_min_ratio,
                    "bright_edge_gradient_threshold": args.bright_edge_gradient_threshold,
                    "bright_edge_min_ratio": args.bright_edge_min_ratio,
                    "rgb_metric_stride": args.rgb_metric_stride,
                },
            }
        )

    meta.update(
        {
            "version": 3,
            "index_type": "stratified_training",
            "bright_edge_v3_source_index": os.path.abspath(args.input_train),
            "rgb_bright_threshold": args.rgb_bright_threshold,
            "rgb_bright_min_ratio": args.rgb_bright_min_ratio,
            "bright_edge_gradient_threshold": args.bright_edge_gradient_threshold,
            "bright_edge_min_ratio": args.bright_edge_min_ratio,
            "rgb_metric_stride": args.rgb_metric_stride,
            "scene_stats": scene_stats,
        }
    )
    save_npz_like(args.output_train, arrays, meta)
    print(f"Saved train index: {args.output_train} ({len(tops)} windows)")


def copy_validation(args) -> None:
    if not args.input_val or not args.output_val:
        return
    os.makedirs(os.path.dirname(args.output_val), exist_ok=True)
    shutil.copy2(args.input_val, args.output_val)
    summary_src = args.input_val + ".summary.json"
    summary_dst = args.output_val + ".summary.json"
    if os.path.exists(summary_src):
        with open(summary_src, "r", encoding="utf-8") as handle:
            meta = json.load(handle)
    else:
        meta = {}
    meta["version"] = max(int(meta.get("version", 0) or 0), 3)
    meta["bright_edge_v3_note"] = "Validation index copied unchanged from v2."
    with open(summary_dst, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)
    print(f"Copied val index: {args.output_val}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Post-process an existing stratified patch index by adding scene18-like "
            "pseudo-RGB bright/high-frequency candidates to existing hs_bright and "
            "hs_complex pools without removing original candidates."
        )
    )
    parser.add_argument("--base_dir", default="/root/autodl-tmp/Baek数据集")
    parser.add_argument("--exr_cache_dir", default="")
    parser.add_argument("--input_train", required=True)
    parser.add_argument("--output_train", required=True)
    parser.add_argument("--input_val", default="")
    parser.add_argument("--output_val", default="")
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--rgb_metric_stride", type=int, default=2)
    parser.add_argument("--hs_norm_scale", type=float, default=0.9367284796834017)
    parser.add_argument("--rgb_bright_threshold", type=float, default=0.6)
    parser.add_argument("--rgb_bright_min_ratio", type=float, default=0.10)
    parser.add_argument("--bright_edge_gradient_threshold", type=float, default=0.03)
    parser.add_argument("--bright_edge_min_ratio", type=float, default=0.03)
    parser.add_argument("--min_depth", type=float, default=0.4)
    parser.add_argument("--max_depth", type=float, default=2.0)
    args = parser.parse_args()
    args.base_dir = os.path.abspath(args.base_dir)
    args.exr_cache_dir = args.exr_cache_dir or os.path.join(
        args.base_dir, ".exr_cache_npy_v1"
    )
    args.input_train = os.path.abspath(args.input_train)
    args.output_train = os.path.abspath(args.output_train)
    if args.input_val:
        args.input_val = os.path.abspath(args.input_val)
    if args.output_val:
        args.output_val = os.path.abspath(args.output_val)
    return args


def main():
    args = parse_args()
    build_train(args)
    copy_validation(args)


if __name__ == "__main__":
    main()
