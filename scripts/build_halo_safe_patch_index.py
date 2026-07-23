#!/usr/bin/env python3
"""Build a halo-safe train index without mutating retained row metadata.

The center 128x128 window remains the reconstruction target.  A training row
is retained only if its scale-1 optical context does not overlap a validation
target.  ``scale_05_eligible`` is then recomputed for the larger source window
required by the same halo.  Every other row-aligned field is copied byte-for-
byte (same dtype and retained values) from the source training index.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def _rect_overlaps_any(
    top: int,
    left: int,
    size: int,
    other_tops: np.ndarray,
    other_lefts: np.ndarray,
    other_size: int,
) -> bool:
    return bool(np.any(
        (top < other_tops + other_size)
        & (top + size > other_tops)
        & (left < other_lefts + other_size)
        & (left + size > other_lefts)
    ))


def _scene_shapes(meta: Dict) -> Dict[str, Tuple[int, int]]:
    shapes = {}
    for row in meta.get('scene_stats', []):
        scene_id = str(row.get('scene_id', ''))
        if scene_id:
            shapes[scene_id] = (int(row['height']), int(row['width']))
    return shapes


def build_halo_safe_index(
    train_path: Path,
    val_path: Path,
    output_path: Path,
    halo: int,
) -> Dict[str, int]:
    if halo < 0:
        raise ValueError(f'halo must be >= 0, got {halo}')
    with np.load(train_path, allow_pickle=False) as source_npz:
        source = {key: source_npz[key] for key in source_npz.files}
    with np.load(val_path, allow_pickle=False) as val_npz:
        val = {key: val_npz[key] for key in val_npz.files}

    for required in ('scene_ids', 'tops', 'lefts', 'meta_json'):
        if required not in source:
            raise ValueError(f'train index missing {required!r}')
    for required in ('scene_ids', 'tops', 'lefts'):
        if required not in val:
            raise ValueError(f'validation index missing {required!r}')

    meta = json.loads(str(source['meta_json'].item()))
    patch_size = int(meta['patch_size'])
    context_size = patch_size + 2 * halo
    source_count = len(source['tops'])
    scene_shapes = _scene_shapes(meta)

    train_scenes = source['scene_ids'].astype(str)
    train_tops = source['tops'].astype(np.int64)
    train_lefts = source['lefts'].astype(np.int64)
    val_scenes = val['scene_ids'].astype(str)
    val_tops = val['tops'].astype(np.int64)
    val_lefts = val['lefts'].astype(np.int64)

    keep = np.ones(source_count, dtype=bool)
    for scene_id in np.unique(train_scenes):
        train_indices = np.flatnonzero(train_scenes == scene_id)
        val_indices = np.flatnonzero(val_scenes == scene_id)
        if scene_id not in scene_shapes:
            raise ValueError(f'missing image shape for {scene_id!r} in meta_json')
        height, width = scene_shapes[scene_id]
        vt = val_tops[val_indices] if len(val_indices) else np.empty(0, dtype=np.int64)
        vl = val_lefts[val_indices] if len(val_indices) else np.empty(0, dtype=np.int64)
        for index in train_indices:
            context_top = int(train_tops[index]) - halo
            context_left = int(train_lefts[index]) - halo
            context_in_bounds = (
                context_top >= 0
                and context_left >= 0
                and context_top + context_size <= height
                and context_left + context_size <= width
            )
            overlaps_validation = (
                len(val_indices) > 0
                and _rect_overlaps_any(
                    context_top, context_left, context_size,
                    vt, vl, patch_size)
            )
            keep[index] = context_in_bounds and not overlaps_validation

    retained_indices = np.flatnonzero(keep)
    retained_scenes = train_scenes[keep]
    retained_tops = train_tops[keep]
    retained_lefts = train_lefts[keep]

    # A 0.5-scale augmentation must extract twice the optical-context extent,
    # centered on the same 128 target.  This eligibility is intentionally
    # recomputed rather than inherited from the old 256-source-window index.
    scale_source_size = 2 * context_size
    scale_eligible = np.zeros(len(retained_indices), dtype=bool)
    for row, (scene_id, top, left) in enumerate(zip(
            retained_scenes, retained_tops, retained_lefts)):
        if scene_id not in scene_shapes:
            raise ValueError(f'missing image shape for {scene_id!r} in meta_json')
        height, width = scene_shapes[scene_id]
        source_top = int(top) + patch_size // 2 - scale_source_size // 2
        source_left = int(left) + patch_size // 2 - scale_source_size // 2
        if (source_top < 0 or source_left < 0
                or source_top + scale_source_size > height
                or source_left + scale_source_size > width):
            continue
        val_indices = np.flatnonzero(val_scenes == scene_id)
        if len(val_indices) and _rect_overlaps_any(
                source_top, source_left, scale_source_size,
                val_tops[val_indices], val_lefts[val_indices], patch_size):
            continue
        scale_eligible[row] = True

    output = {}
    for key, values in source.items():
        if key == 'meta_json':
            continue
        if values.ndim >= 1 and values.shape[0] == source_count:
            output[key] = values[keep]
        else:
            output[key] = values.copy()
    output['scale_05_eligible'] = scale_eligible

    meta.update({
        'index_type': f"{meta.get('index_type', 'train')}_halo_safe",
        'source_train_index_before_halo': str(train_path.resolve()),
        'validation_index_for_halo_exclusion': str(val_path.resolve()),
        'optical_halo': int(halo),
        'optical_context_size': int(context_size),
        'halo_excludes_validation_union': True,
        'halo_context_in_bounds': True,
        'candidate_count_before_halo': int(source_count),
        'candidate_count': int(len(retained_indices)),
        'halo_overlap_dropped_count': int(source_count - len(retained_indices)),
        'scale_05_source_size': int(scale_source_size),
        'scale_05_excludes_validation_union': True,
        'scale_05_eligible_count': int(scale_eligible.sum()),
    })
    output['meta_json'] = np.asarray(
        json.dumps(meta, ensure_ascii=False, sort_keys=True))

    # Rigorous invariant: only scale_05_eligible may change among retained
    # row-aligned fields.  Dtype, shape tail, and values must otherwise match.
    for key, source_values in source.items():
        if key in ('meta_json', 'scale_05_eligible'):
            continue
        if source_values.ndim >= 1 and source_values.shape[0] == source_count:
            expected = source_values[keep]
            actual = output[key]
            values_equal = np.array_equal(
                expected, actual, equal_nan=True
            ) if expected.dtype.kind in 'fc' else np.array_equal(expected, actual)
            if expected.dtype != actual.dtype or not values_equal:
                raise AssertionError(f'retained field changed unexpectedly: {key}')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(output_path.name + '.tmp.npz')
    np.savez_compressed(temp_path, **output)
    os.replace(temp_path, output_path)
    return {
        'source_count': source_count,
        'retained_count': len(retained_indices),
        'dropped_count': source_count - len(retained_indices),
        'scale_05_eligible_count': int(scale_eligible.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-index', type=Path, required=True)
    parser.add_argument('--val-index', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--halo', type=int, default=32)
    args = parser.parse_args()
    stats = build_halo_safe_index(
        args.train_index, args.val_index, args.output, args.halo)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f'wrote: {args.output.resolve()}')


if __name__ == '__main__':
    main()
