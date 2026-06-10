#!/usr/bin/env python
import argparse
import hashlib
import json
import os
import re
from typing import Dict, List, Tuple

import numpy as np


def cache_file_path(exr_path: str, cache_dir: str) -> str:
    stat = os.stat(exr_path)
    key_src = f"{exr_path}|{stat.st_mtime_ns}|{stat.st_size}"
    key = hashlib.sha1(key_src.encode("utf-8")).hexdigest()
    return os.path.join(cache_dir, f"{key}.npy")


def read_exr_depth(exr_path: str) -> np.ndarray:
    try:
        import OpenEXR
        import Imath
    except ImportError as exc:
        raise RuntimeError(
            "OpenEXR is not installed in this Python environment. "
            "Use --use_exr_cache with an existing cache, or run in the training environment."
        ) from exc

    if not OpenEXR.isOpenExrFile(exr_path):
        raise IOError(f"Not a valid EXR file: {exr_path}")

    exr_file = OpenEXR.InputFile(exr_path)
    header = exr_file.header()
    dw = header["dataWindow"]
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    channel_names = sorted(header["channels"].keys())
    if not channel_names:
        raise ValueError(f"No channels in EXR: {exr_path}")

    channel_type = header["channels"][channel_names[0]].type
    if channel_type == Imath.PixelType(Imath.PixelType.FLOAT):
        dtype = np.float32
    elif channel_type == Imath.PixelType(Imath.PixelType.HALF):
        dtype = np.float16
    else:
        raise TypeError(f"Unsupported EXR pixel type: {channel_type}")

    channel = exr_file.channel(channel_names[0])
    depth = np.frombuffer(channel, dtype=dtype).reshape(height, width)
    return depth.astype(np.float32, copy=False)


def load_depth(exr_path: str, cache_dir: str, use_exr_cache: bool) -> np.ndarray:
    if use_exr_cache:
        cache_path = cache_file_path(exr_path, cache_dir)
        if os.path.exists(cache_path):
            return np.load(cache_path, allow_pickle=False).astype(np.float32, copy=False)

    return read_exr_depth(exr_path)


def metric_to_ips_np(depth_m: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
    return (max_depth * depth_m - max_depth * min_depth) / (
        (max_depth - min_depth) * depth_m
    )


def integral_image(x: np.ndarray) -> np.ndarray:
    return np.pad(x, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)


def box_sum(ii: np.ndarray, top: int, left: int, height: int, width: int) -> float:
    bottom = top + height
    right = left + width
    return float(ii[bottom, right] - ii[top, right] - ii[bottom, left] + ii[top, left])


def scan_positions(length: int, patch: int, stride: int) -> List[int]:
    if length < patch:
        return [0]
    positions = list(range(0, length - patch + 1, stride))
    if positions[-1] != length - patch:
        positions.append(length - patch)
    return positions


def scene_sort_key(folder_name: str) -> int:
    match = re.search(r"\d+", folder_name)
    return int(match.group(0)) if match else 10**9


def discover_scene_folders(base_dir: str) -> List[str]:
    folders = [
        name for name in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, name)) and re.match(r"^deploy\s+\d+$", name)
    ]
    return sorted(folders, key=scene_sort_key)


def score_window(valid_ratio: float, center_valid_ratio: float, depth_range_ips: float,
                 min_depth_range_ips: float, far_ratio: float = 0.0,
                 very_far_ratio: float = 0.0, far_score_boost: float = 0.0,
                 very_far_score_boost: float = 0.0) -> float:
    range_scale = depth_range_ips / max(min_depth_range_ips, 1e-6)
    range_scale = min(range_scale, 2.5)
    base_score = valid_ratio * (0.5 + 0.5 * center_valid_ratio) * range_scale
    far_boost = 1.0 + far_score_boost * far_ratio + very_far_score_boost * very_far_ratio
    return float(base_score * max(far_boost, 1e-6))


def build_index_for_scene(
    depth_raw: np.ndarray,
    scene_id: str,
    patch_size: int,
    stride: int,
    min_depth: float,
    max_depth: float,
    valid_eps: float,
    min_valid_ratio: float,
    min_depth_range_ips: float,
    center_fraction: float,
    min_center_valid_ratio: float,
    far_depth_threshold: float,
    very_far_depth_threshold: float,
    far_score_boost: float,
    very_far_score_boost: float,
) -> Tuple[Dict[str, List], Dict]:
    if depth_raw.ndim == 3:
        depth_raw = np.squeeze(depth_raw)
    if depth_raw.ndim != 2:
        raise ValueError(f"Depth must be 2D after squeeze, got shape={depth_raw.shape}")

    height, width = depth_raw.shape
    depth_m = depth_raw.astype(np.float32, copy=False) / 1000.0

    valid = np.isfinite(depth_m) & (depth_m > min_depth - valid_eps)
    valid_f = valid.astype(np.float32, copy=False)
    valid_ii = integral_image(valid_f)

    depth_safe = np.where(valid, depth_m, min_depth).astype(np.float32, copy=False)
    depth_ips = metric_to_ips_np(depth_safe, min_depth, max_depth)
    depth_ips = np.clip(depth_ips, 0.0, 1.0).astype(np.float32, copy=False)

    patch_area = patch_size * patch_size
    center_size = max(1, int(round(patch_size * center_fraction)))
    center_size = min(center_size, patch_size)
    center_offset = (patch_size - center_size) // 2
    center_area = center_size * center_size

    ys = scan_positions(height, patch_size, stride)
    xs = scan_positions(width, patch_size, stride)

    out = {
        "scene_ids": [],
        "tops": [],
        "lefts": [],
        "scores": [],
        "valid_ratios": [],
        "center_valid_ratios": [],
        "depth_ranges_ips": [],
        "far_ratios": [],
        "very_far_ratios": [],
    }

    total_windows = 0
    pass_valid = 0
    pass_center = 0
    pass_range = 0

    for top in ys:
        for left in xs:
            total_windows += 1
            valid_ratio = box_sum(valid_ii, top, left, patch_size, patch_size) / patch_area
            if valid_ratio < min_valid_ratio:
                continue
            pass_valid += 1

            center_top = top + center_offset
            center_left = left + center_offset
            center_valid_ratio = (
                box_sum(valid_ii, center_top, center_left, center_size, center_size) / center_area
            )
            if center_valid_ratio < min_center_valid_ratio:
                continue
            pass_center += 1

            patch_valid = valid[top:top + patch_size, left:left + patch_size]
            patch_ips = depth_ips[top:top + patch_size, left:left + patch_size]
            valid_ips = patch_ips[patch_valid]
            if valid_ips.size == 0:
                continue

            depth_range_ips = float(valid_ips.max() - valid_ips.min())
            if depth_range_ips < min_depth_range_ips:
                continue
            pass_range += 1

            patch_depth = depth_m[top:top + patch_size, left:left + patch_size]
            valid_depth = patch_depth[patch_valid]
            far_ratio = float(np.mean(valid_depth >= far_depth_threshold))
            very_far_ratio = float(np.mean(valid_depth >= very_far_depth_threshold))

            out["scene_ids"].append(scene_id)
            out["tops"].append(top)
            out["lefts"].append(left)
            out["valid_ratios"].append(valid_ratio)
            out["center_valid_ratios"].append(center_valid_ratio)
            out["depth_ranges_ips"].append(depth_range_ips)
            out["far_ratios"].append(far_ratio)
            out["very_far_ratios"].append(very_far_ratio)
            out["scores"].append(
                score_window(
                    valid_ratio,
                    center_valid_ratio,
                    depth_range_ips,
                    min_depth_range_ips,
                    far_ratio=far_ratio,
                    very_far_ratio=very_far_ratio,
                    far_score_boost=far_score_boost,
                    very_far_score_boost=very_far_score_boost,
                )
            )

    valid_values = depth_m[valid]
    stats = {
        "scene_id": scene_id,
        "height": int(height),
        "width": int(width),
        "total_windows": int(total_windows),
        "candidates": int(len(out["tops"])),
        "pass_valid_ratio": int(pass_valid),
        "pass_center_valid_ratio": int(pass_center),
        "pass_depth_range": int(pass_range),
        "full_valid_ratio": float(valid.mean()),
        "valid_depth_min_m": float(valid_values.min()) if valid_values.size else None,
        "valid_depth_max_m": float(valid_values.max()) if valid_values.size else None,
    }
    return out, stats


def append_scene(dst: Dict[str, List], src: Dict[str, List]) -> None:
    for key, values in src.items():
        dst[key].extend(values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build high-quality 128x128 patch index for Baek depth EXR files.")
    parser.add_argument("--base_dir", type=str, default="/root/autodl-tmp/Baek数据集")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--min_depth", type=float, default=0.4)
    parser.add_argument("--max_depth", type=float, default=2.0)
    parser.add_argument("--valid_eps", type=float, default=1e-3)
    parser.add_argument("--min_valid_ratio", type=float, default=0.20)
    parser.add_argument("--min_depth_range_ips", type=float, default=0.06)
    parser.add_argument("--center_fraction", type=float, default=0.5)
    parser.add_argument("--min_center_valid_ratio", type=float, default=0.10)
    parser.add_argument("--far_depth_threshold", type=float, default=1.4)
    parser.add_argument("--very_far_depth_threshold", type=float, default=1.6)
    parser.add_argument("--far_score_boost", type=float, default=0.0)
    parser.add_argument("--very_far_score_boost", type=float, default=0.0)
    parser.add_argument("--use_exr_cache", action="store_true", default=True)
    parser.add_argument("--no-use_exr_cache", dest="use_exr_cache", action="store_false")
    parser.add_argument("--exr_cache_dir", type=str, default="")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.patch_size <= 0 or args.stride <= 0:
        raise ValueError("--patch_size and --stride must be positive")
    if not (0.0 <= args.min_valid_ratio <= 1.0):
        raise ValueError("--min_valid_ratio must be in [0, 1]")
    if not (0.0 <= args.min_center_valid_ratio <= 1.0):
        raise ValueError("--min_center_valid_ratio must be in [0, 1]")
    if not (0.0 < args.center_fraction <= 1.0):
        raise ValueError("--center_fraction must be in (0, 1]")

    base_dir = os.path.abspath(args.base_dir)
    cache_dir = args.exr_cache_dir or os.path.join(base_dir, ".exr_cache_npy_v1")
    output = args.output
    if not output:
        index_dir = os.path.join(base_dir, ".patch_index")
        output = os.path.join(
            index_dir,
            (
                f"patch{args.patch_size}_stride{args.stride}"
                f"_valid{int(round(args.min_valid_ratio * 100)):02d}"
                f"_range{int(round(args.min_depth_range_ips * 1000)):03d}"
                f"_center{int(round(args.min_center_valid_ratio * 100)):02d}"
                f"_far{int(round(args.far_depth_threshold * 100)):03d}"
                f"b{int(round(args.far_score_boost * 10)):02d}"
                f"_vfar{int(round(args.very_far_depth_threshold * 100)):03d}"
                f"b{int(round(args.very_far_score_boost * 10)):02d}_v1.npz"
            ),
        )
    output = os.path.abspath(output)
    os.makedirs(os.path.dirname(output), exist_ok=True)

    if os.path.exists(output) and not args.force:
        raise FileExistsError(f"Output already exists: {output}. Use --force to overwrite.")

    scene_folders = discover_scene_folders(base_dir)
    all_candidates = {
        "scene_ids": [],
        "tops": [],
        "lefts": [],
        "scores": [],
        "valid_ratios": [],
        "center_valid_ratios": [],
        "depth_ranges_ips": [],
        "far_ratios": [],
        "very_far_ratios": [],
    }
    scene_stats = []

    for folder_name in scene_folders:
        scene_num = scene_sort_key(folder_name)
        scene_id = f"scene_{scene_num:02d}"
        depth_path = os.path.join(base_dir, folder_name, f"scene{scene_num:02d}_depth_map.exr")
        if not os.path.exists(depth_path):
            print(f"[skip] missing depth: {depth_path}")
            continue

        depth = load_depth(depth_path, cache_dir, args.use_exr_cache)
        scene_candidates, stats = build_index_for_scene(
            depth,
            scene_id=scene_id,
            patch_size=args.patch_size,
            stride=args.stride,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            valid_eps=args.valid_eps,
            min_valid_ratio=args.min_valid_ratio,
            min_depth_range_ips=args.min_depth_range_ips,
            center_fraction=args.center_fraction,
            min_center_valid_ratio=args.min_center_valid_ratio,
            far_depth_threshold=args.far_depth_threshold,
            very_far_depth_threshold=args.very_far_depth_threshold,
            far_score_boost=args.far_score_boost,
            very_far_score_boost=args.very_far_score_boost,
        )
        append_scene(all_candidates, scene_candidates)
        scene_stats.append(stats)
        ratio = stats["candidates"] / max(stats["total_windows"], 1)
        print(
            f"[{scene_id}] size={stats['height']}x{stats['width']} "
            f"valid={stats['full_valid_ratio']:.3f} "
            f"candidates={stats['candidates']}/{stats['total_windows']} ({ratio:.3%})"
        )

    meta = {
        "version": 1,
        "base_dir": base_dir,
        "patch_size": args.patch_size,
        "stride": args.stride,
        "min_depth": args.min_depth,
        "max_depth": args.max_depth,
        "valid_eps": args.valid_eps,
        "min_valid_ratio": args.min_valid_ratio,
        "min_depth_range_ips": args.min_depth_range_ips,
        "center_fraction": args.center_fraction,
        "min_center_valid_ratio": args.min_center_valid_ratio,
        "far_depth_threshold": args.far_depth_threshold,
        "very_far_depth_threshold": args.very_far_depth_threshold,
        "far_score_boost": args.far_score_boost,
        "very_far_score_boost": args.very_far_score_boost,
        "valid_mask_rule": "finite(depth_m) and depth_m > min_depth - valid_eps",
        "depth_range_rule": "range(metric_to_ips(depth_m, min_depth, max_depth).clip(0, 1)) over valid pixels",
        "score_rule": (
            "base_score * (1 + far_score_boost * far_ratio "
            "+ very_far_score_boost * very_far_ratio); "
            "far ratios are measured over valid pixels only"
        ),
        "scene_stats": scene_stats,
    }

    np.savez_compressed(
        output,
        scene_ids=np.asarray(all_candidates["scene_ids"], dtype="U16"),
        tops=np.asarray(all_candidates["tops"], dtype=np.int32),
        lefts=np.asarray(all_candidates["lefts"], dtype=np.int32),
        scores=np.asarray(all_candidates["scores"], dtype=np.float32),
        valid_ratios=np.asarray(all_candidates["valid_ratios"], dtype=np.float32),
        center_valid_ratios=np.asarray(all_candidates["center_valid_ratios"], dtype=np.float32),
        depth_ranges_ips=np.asarray(all_candidates["depth_ranges_ips"], dtype=np.float32),
        far_ratios=np.asarray(all_candidates["far_ratios"], dtype=np.float32),
        very_far_ratios=np.asarray(all_candidates["very_far_ratios"], dtype=np.float32),
        meta_json=np.asarray(json.dumps(meta, ensure_ascii=False)),
    )

    summary_path = output + ".summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    total_windows = sum(s["total_windows"] for s in scene_stats)
    total_candidates = len(all_candidates["tops"])
    print("-" * 80)
    print(f"Saved patch index: {output}")
    print(f"Saved summary:     {summary_path}")
    print(f"Total candidates:  {total_candidates}/{total_windows} ({total_candidates / max(total_windows, 1):.3%})")


if __name__ == "__main__":
    main()
