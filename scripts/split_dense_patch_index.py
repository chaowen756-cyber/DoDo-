#!/usr/bin/env python
import argparse
import itertools
import json
import math
import os
from typing import Dict, Iterable

import numpy as np


def _load_meta(data: np.lib.npyio.NpzFile) -> Dict:
    if "meta_json" not in data.files:
        return {}
    try:
        return json.loads(str(data["meta_json"].item()))
    except Exception:
        return {}


def _scene_ids(start: int, end: int) -> Iterable[str]:
    for scene_no in range(int(start), int(end) + 1):
        yield f"scene_{scene_no:02d}"


def _val_count(total: int, fraction: float, mode: str) -> int:
    raw = total * fraction
    if mode == "floor":
        count = math.floor(raw)
    elif mode == "ceil":
        count = math.ceil(raw)
    else:
        count = round(raw)
    return min(max(int(count), 1), max(total - 1, 1))


def _scene_dimensions(source_meta: Dict, scene_id: str, tops, lefts, patch_size: int):
    for stats in source_meta.get("scene_stats", []):
        if str(stats.get("scene_id", "")) == scene_id:
            height = int(stats.get("height", 0) or 0)
            width = int(stats.get("width", 0) or 0)
            if height > 0 and width > 0:
                return height, width
    return int(np.max(tops) + patch_size), int(np.max(lefts) + patch_size)


def _cell_ids_for_candidates(
    tops,
    lefts,
    patch_size: int,
    height: int,
    width: int,
    rows: int,
    cols: int,
) -> np.ndarray:
    centers_y = tops.astype(np.float64) + patch_size * 0.5
    centers_x = lefts.astype(np.float64) + patch_size * 0.5
    row_ids = np.floor(centers_y / max(height, 1) * rows).astype(np.int64)
    col_ids = np.floor(centers_x / max(width, 1) * cols).astype(np.int64)
    row_ids = np.clip(row_ids, 0, rows - 1)
    col_ids = np.clip(col_ids, 0, cols - 1)
    return row_ids * cols + col_ids


def _choose_block_cells(
    cell_ids: np.ndarray,
    target_count: int,
    rows: int,
    cols: int,
    max_blocks: int,
    rng: np.random.Generator,
):
    populated_cells = sorted(int(cell) for cell in np.unique(cell_ids))
    cell_counts = {
        cell: int(np.count_nonzero(cell_ids == cell))
        for cell in populated_cells
    }
    max_blocks = max(1, min(int(max_blocks), len(populated_cells)))
    best = None
    for block_count in range(1, max_blocks + 1):
        for combo in itertools.combinations(populated_cells, block_count):
            count = sum(cell_counts[cell] for cell in combo)
            score = (
                abs(count - target_count),
                abs(block_count - max(1, round(rows * cols * 0.10))),
                rng.random(),
            )
            if best is None or score < best[0]:
                best = (score, combo, count)
    if best is None:
        raise RuntimeError("No populated block cells found for block split")
    return set(best[1]), int(best[2])


def _remove_train_overlapping_val(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    tops: np.ndarray,
    lefts: np.ndarray,
    patch_size: int,
    chunk_size: int = 256,
) -> np.ndarray:
    if train_idx.size == 0 or val_idx.size == 0:
        return train_idx

    train_top = tops[train_idx]
    train_left = lefts[train_idx]
    train_bottom = train_top + patch_size
    train_right = train_left + patch_size
    keep = np.ones(train_idx.shape[0], dtype=bool)

    for start in range(0, val_idx.size, chunk_size):
        chunk = val_idx[start:start + chunk_size]
        val_top = tops[chunk]
        val_left = lefts[chunk]
        val_bottom = val_top + patch_size
        val_right = val_left + patch_size
        overlap_y = (
            (train_top[:, None] < val_bottom[None, :])
            & (train_bottom[:, None] > val_top[None, :])
        )
        overlap_x = (
            (train_left[:, None] < val_right[None, :])
            & (train_right[:, None] > val_left[None, :])
        )
        keep &= ~(overlap_y & overlap_x).any(axis=1)
        if not keep.any():
            break

    return train_idx[keep]


def _subset_arrays(
    data: np.lib.npyio.NpzFile,
    indices: np.ndarray,
    meta: Dict,
) -> Dict[str, np.ndarray]:
    total = len(data["scene_ids"])
    arrays: Dict[str, np.ndarray] = {}
    for key in data.files:
        if key == "meta_json":
            continue
        value = data[key]
        if value.shape[:1] == (total,):
            arrays[key] = value[indices]
        else:
            arrays[key] = value
    arrays["meta_json"] = np.asarray(json.dumps(meta, ensure_ascii=False))
    return arrays


def _save_index(path: str, arrays: Dict[str, np.ndarray], meta: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **arrays)
    with open(path + ".summary.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)


def _scene_stats_subset(source_meta: Dict, selected_scene_ids) -> list:
    selected = set(selected_scene_ids)
    return [
        stats for stats in source_meta.get("scene_stats", [])
        if str(stats.get("scene_id", "")) in selected
    ]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Split an existing dense foreground patch index into per-scene "
            "train/validation subsets without changing candidate filters."
        )
    )
    parser.add_argument("--source_index", required=True)
    parser.add_argument("--train_output", required=True)
    parser.add_argument("--val_output", required=True)
    parser.add_argument("--scene_start", type=int, default=1)
    parser.add_argument("--scene_end", type=int, default=13)
    parser.add_argument("--heldout_scene_start", type=int, default=14)
    parser.add_argument("--heldout_scene_end", type=int, default=18)
    parser.add_argument("--val_fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--split_mode", choices=["random", "block"], default="random")
    parser.add_argument("--patch_size", type=int, default=0,
                        help="Patch size for block overlap checks; 0=read from source meta")
    parser.add_argument("--block_grid_rows", type=int, default=5)
    parser.add_argument("--block_grid_cols", type=int, default=5)
    parser.add_argument("--max_val_blocks_per_scene", type=int, default=4)
    parser.add_argument("--exclude_train_overlapping_val", action="store_true",
                        help="Drop train patches whose windows overlap any selected val patch")
    parser.add_argument(
        "--val_count_mode",
        choices=["round", "floor", "ceil"],
        default="round",
        help="How to convert per-scene val_fraction to an integer patch count.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.source_index = os.path.abspath(args.source_index)
    args.train_output = os.path.abspath(args.train_output)
    args.val_output = os.path.abspath(args.val_output)
    if not (0.0 < args.val_fraction < 1.0):
        raise ValueError("--val_fraction must be in (0, 1)")
    if args.scene_start < 1 or args.scene_end < args.scene_start:
        raise ValueError("--scene_start/--scene_end must form a valid positive range")
    for path in (args.train_output, args.val_output):
        if os.path.exists(path) and not args.force:
            raise FileExistsError(f"Output exists: {path}; pass --force to overwrite")
    return args


def main() -> None:
    args = parse_args()
    data = np.load(args.source_index, allow_pickle=False)
    required = {"scene_ids", "tops", "lefts"}
    missing = required - set(data.files)
    if missing:
        raise ValueError(f"Source index missing fields: {sorted(missing)}")

    source_meta = _load_meta(data)
    source_ids = data["scene_ids"].astype(str)
    patch_size = int(args.patch_size or source_meta.get("patch_size", 0) or 0)
    if patch_size <= 0:
        raise ValueError("--patch_size must be set when source meta lacks patch_size")
    rng = np.random.default_rng(args.seed)

    train_indices = []
    val_indices = []
    split_counts = {}
    selected_scene_ids = list(_scene_ids(args.scene_start, args.scene_end))

    for scene_id in selected_scene_ids:
        candidates = np.flatnonzero(source_ids == scene_id)
        if candidates.size < 2:
            raise RuntimeError(f"{scene_id} has too few candidates: {candidates.size}")
        shuffled = candidates.copy()
        rng.shuffle(shuffled)
        val_n = _val_count(candidates.size, args.val_fraction, args.val_count_mode)
        block_cells = []
        block_count = None
        if args.split_mode == "block":
            scene_tops = data["tops"][candidates].astype(np.int64)
            scene_lefts = data["lefts"][candidates].astype(np.int64)
            height, width = _scene_dimensions(
                source_meta, scene_id, scene_tops, scene_lefts, patch_size
            )
            cell_ids = _cell_ids_for_candidates(
                scene_tops,
                scene_lefts,
                patch_size,
                height,
                width,
                args.block_grid_rows,
                args.block_grid_cols,
            )
            selected_cells, block_count = _choose_block_cells(
                cell_ids,
                val_n,
                args.block_grid_rows,
                args.block_grid_cols,
                args.max_val_blocks_per_scene,
                rng,
            )
            block_cells = sorted(selected_cells)
            val_pick = np.sort(candidates[np.isin(cell_ids, block_cells)])
            train_pick = np.sort(candidates[~np.isin(cell_ids, block_cells)])
        else:
            val_pick = np.sort(shuffled[:val_n])
            train_pick = np.sort(shuffled[val_n:])

        train_before_overlap_filter = int(train_pick.size)
        if args.exclude_train_overlapping_val:
            train_pick = np.sort(
                _remove_train_overlapping_val(
                    train_pick,
                    val_pick,
                    data["tops"].astype(np.int64),
                    data["lefts"].astype(np.int64),
                    patch_size,
                )
            )
        val_indices.append(val_pick)
        train_indices.append(train_pick)
        split_counts[scene_id] = {
            "total": int(candidates.size),
            "train": int(train_pick.size),
            "val": int(val_pick.size),
            "target_val": int(val_n),
            "train_dropped_overlap": int(train_before_overlap_filter - train_pick.size),
            "selected_val_blocks": block_cells,
            "selected_val_candidates_from_blocks": block_count,
        }
        print(
            f"[{scene_id}] total={candidates.size} "
            f"train={train_pick.size} val={val_pick.size} target_val={val_n}"
        )

    train_idx = np.concatenate(train_indices)
    val_idx = np.concatenate(val_indices)

    common_meta = {
        **source_meta,
        "version": max(int(source_meta.get("version", 1) or 1), 1),
        "index_type": "dense_foreground_scene_split",
        "source_index": args.source_index,
        "split_scene_start": args.scene_start,
        "split_scene_end": args.scene_end,
        "split_scenes": selected_scene_ids,
        "val_sampling_seed": args.seed,
        "val_sampling_fraction": args.val_fraction,
        "val_count_mode": args.val_count_mode,
        "split_mode": args.split_mode,
        "patch_size": patch_size,
        "block_grid_rows": args.block_grid_rows,
        "block_grid_cols": args.block_grid_cols,
        "max_val_blocks_per_scene": args.max_val_blocks_per_scene,
        "exclude_train_overlapping_val": bool(args.exclude_train_overlapping_val),
        "scene_stats": _scene_stats_subset(source_meta, selected_scene_ids),
        "split_counts": split_counts,
        "held_out_test_scenes": [
            f"scene_{scene_no:02d}"
            for scene_no in range(args.heldout_scene_start, args.heldout_scene_end + 1)
        ],
    }

    train_meta = {
        **common_meta,
        "split_name": "train",
        "candidate_count": int(train_idx.size),
    }
    val_meta = {
        **common_meta,
        "split_name": "val",
        "candidate_count": int(val_idx.size),
    }

    _save_index(args.train_output, _subset_arrays(data, train_idx, train_meta), train_meta)
    _save_index(args.val_output, _subset_arrays(data, val_idx, val_meta), val_meta)

    print("-" * 80)
    print(f"Saved train index: {args.train_output} ({train_idx.size} windows)")
    print(f"Saved val index:   {args.val_output} ({val_idx.size} windows)")


if __name__ == "__main__":
    main()
