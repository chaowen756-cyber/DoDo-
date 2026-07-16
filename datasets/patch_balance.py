from typing import Dict, Tuple

import numpy as np


def metric_to_ips_numpy(
    depth_m: np.ndarray, min_depth: float, max_depth: float
) -> np.ndarray:
    return (
        max_depth * depth_m - max_depth * min_depth
    ) / ((max_depth - min_depth) * depth_m)


def rectangle_sums(
    integral: np.ndarray,
    tops: np.ndarray,
    lefts: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    bottoms = tops + int(height)
    rights = lefts + int(width)
    bottom_right = integral[bottoms, rights].astype(np.int64)
    top_right = integral[tops, rights].astype(np.int64)
    bottom_left = integral[bottoms, lefts].astype(np.int64)
    top_left = integral[tops, lefts].astype(np.int64)
    return (
        bottom_right - top_right - bottom_left + top_left
    )


def depth_histograms_ips(
    depth_m: np.ndarray,
    tops: np.ndarray,
    lefts: np.ndarray,
    patch_size: int,
    min_depth: float,
    max_depth: float,
    bins: int = 16,
) -> np.ndarray:
    depth_m = np.asarray(depth_m, dtype=np.float32)
    tops = np.asarray(tops, dtype=np.int64)
    lefts = np.asarray(lefts, dtype=np.int64)
    if depth_m.ndim != 2:
        raise ValueError(f'Expected depth map [H,W], got {depth_m.shape}')
    if np.any(tops < 0) or np.any(lefts < 0):
        raise ValueError('Patch coordinates must be non-negative')
    if np.any(tops + patch_size > depth_m.shape[0]) or np.any(
        lefts + patch_size > depth_m.shape[1]
    ):
        raise ValueError('Patch coordinate exceeds depth-map bounds')

    valid = np.isfinite(depth_m) & (depth_m >= min_depth - 1e-3)
    ips = np.zeros_like(depth_m, dtype=np.float32)
    ips[valid] = metric_to_ips_numpy(depth_m[valid], min_depth, max_depth)
    bin_ids = np.floor(np.clip(ips, 0.0, 1.0) * bins).astype(np.int16)
    bin_ids = np.minimum(bin_ids, bins - 1)

    histograms = np.zeros((len(tops), bins), dtype=np.uint32)
    integral = np.zeros(
        (depth_m.shape[0] + 1, depth_m.shape[1] + 1), dtype=np.uint32
    )
    for bin_index in range(bins):
        integral.fill(0)
        integral[1:, 1:] = valid & (bin_ids == bin_index)
        np.cumsum(integral, axis=0, dtype=np.uint32, out=integral)
        np.cumsum(integral, axis=1, dtype=np.uint32, out=integral)
        histograms[:, bin_index] = rectangle_sums(
            integral, tops, lefts, patch_size, patch_size
        )
    return histograms


def bounded_mean_normalize(
    raw_weights: np.ndarray, minimum: float, maximum: float
) -> np.ndarray:
    raw_weights = np.maximum(np.asarray(raw_weights, dtype=np.float64), 1e-12)
    if not 0.0 < minimum <= 1.0 <= maximum:
        raise ValueError('Weight bounds must contain 1.0')
    lower_scale = 0.0
    upper_scale = maximum / float(raw_weights.min())
    for _ in range(80):
        scale = (lower_scale + upper_scale) * 0.5
        mean_weight = np.clip(raw_weights * scale, minimum, maximum).mean()
        if mean_weight < 1.0:
            lower_scale = scale
        else:
            upper_scale = scale
    return np.clip(raw_weights * upper_scale, minimum, maximum).astype(np.float32)


def effective_sample_size(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=np.float64)
    return float(weights.sum() ** 2 / np.square(weights).sum())


def tempered_depth_sampling_weights(
    histograms: np.ndarray,
    scene_ids: np.ndarray,
    target_exponent: float = 0.5,
    minimum: float = 0.25,
    maximum: float = 4.0,
) -> Tuple[np.ndarray, Dict]:
    histograms = np.asarray(histograms, dtype=np.float64)
    scene_ids = np.asarray(scene_ids).astype(str)
    if histograms.ndim != 2 or histograms.shape[0] != len(scene_ids):
        raise ValueError('Histogram and scene-id dimensions do not match')
    if not 0.0 <= target_exponent <= 1.0:
        raise ValueError('target_exponent must be in [0, 1]')

    unique_scenes = sorted(set(scene_ids.tolist()))
    scene_uniform_scale = np.zeros(len(scene_ids), dtype=np.float64)
    for scene_id in unique_scenes:
        scene_mask = scene_ids == scene_id
        scene_uniform_scale[scene_mask] = 1.0 / int(scene_mask.sum())
    bin_counts = (histograms * scene_uniform_scale[:, None]).sum(axis=0)
    if bin_counts.sum() <= 0.0:
        raise ValueError('Cannot balance an index with no valid depth pixels')
    source_distribution = bin_counts / bin_counts.sum()
    target_distribution = np.power(source_distribution, target_exponent)
    target_distribution /= target_distribution.sum()
    bin_importance = np.divide(
        target_distribution,
        source_distribution,
        out=np.zeros_like(target_distribution),
        where=source_distribution > 0.0,
    )

    patch_totals = histograms.sum(axis=1, keepdims=True)
    patch_distributions = np.divide(
        histograms,
        patch_totals,
        out=np.zeros_like(histograms),
        where=patch_totals > 0.0,
    )
    raw_weights = patch_distributions @ bin_importance
    raw_weights[patch_totals[:, 0] <= 0.0] = 1.0

    weights = np.ones(len(scene_ids), dtype=np.float32)
    scene_report = {}
    for scene_id in unique_scenes:
        scene_mask = scene_ids == scene_id
        weights[scene_mask] = bounded_mean_normalize(
            raw_weights[scene_mask], minimum, maximum
        )
        scene_weights = weights[scene_mask]
        scene_report[scene_id] = {
            'count': int(scene_mask.sum()),
            'weight_min': float(scene_weights.min()),
            'weight_max': float(scene_weights.max()),
            'weight_mean': float(scene_weights.mean()),
            'ess': effective_sample_size(scene_weights),
            'ess_ratio': effective_sample_size(scene_weights) / int(scene_mask.sum()),
        }

    weighted_counts = np.zeros(histograms.shape[1], dtype=np.float64)
    for scene_id in unique_scenes:
        scene_mask = scene_ids == scene_id
        scene_weights = weights[scene_mask].astype(np.float64)
        weighted_counts += (
            histograms[scene_mask] * scene_weights[:, None]
        ).sum(axis=0) / scene_weights.sum()
    weighted_distribution = weighted_counts / weighted_counts.sum()
    report = {
        'source_distribution': source_distribution.tolist(),
        'target_distribution': target_distribution.tolist(),
        'weighted_distribution': weighted_distribution.tolist(),
        'bin_importance': bin_importance.tolist(),
        'weight_min': float(weights.min()),
        'weight_max': float(weights.max()),
        'weight_mean': float(weights.mean()),
        'ess': effective_sample_size(weights),
        'ess_ratio': effective_sample_size(weights) / len(weights),
        'scenes': scene_report,
    }
    return weights, report


def scale_half_eligibility(
    train_tops: np.ndarray,
    train_lefts: np.ndarray,
    val_tops: np.ndarray,
    val_lefts: np.ndarray,
    image_height: int,
    image_width: int,
    patch_size: int,
) -> np.ndarray:
    train_tops = np.asarray(train_tops, dtype=np.int64)
    train_lefts = np.asarray(train_lefts, dtype=np.int64)
    val_tops = np.asarray(val_tops, dtype=np.int64)
    val_lefts = np.asarray(val_lefts, dtype=np.int64)
    source_size = int(patch_size) * 2
    source_tops = train_tops - patch_size // 2
    source_lefts = train_lefts - patch_size // 2
    eligible = (
        (source_tops >= 0)
        & (source_lefts >= 0)
        & (source_tops + source_size <= image_height)
        & (source_lefts + source_size <= image_width)
    )
    if len(val_tops) == 0 or not np.any(eligible):
        return eligible

    eligible_indices = np.flatnonzero(eligible)
    for start in range(0, len(eligible_indices), 1024):
        chunk = eligible_indices[start:start + 1024]
        source_top = source_tops[chunk, None]
        source_left = source_lefts[chunk, None]
        overlaps = (
            (source_top < val_tops[None, :] + patch_size)
            & (source_top + source_size > val_tops[None, :])
            & (source_left < val_lefts[None, :] + patch_size)
            & (source_left + source_size > val_lefts[None, :])
        )
        eligible[chunk] &= ~np.any(overlaps, axis=1)
    return eligible
