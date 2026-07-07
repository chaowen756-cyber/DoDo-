#!/usr/bin/env python
import argparse
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
        val_pick = np.sort(shuffled[:val_n])
        train_pick = np.sort(shuffled[val_n:])
        val_indices.append(val_pick)
        train_indices.append(train_pick)
        split_counts[scene_id] = {
            "total": int(candidates.size),
            "train": int(train_pick.size),
            "val": int(val_pick.size),
        }
        print(
            f"[{scene_id}] total={candidates.size} "
            f"train={train_pick.size} val={val_pick.size}"
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
