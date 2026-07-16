import numpy as np
import pytest

from datasets.patch_balance import (
    depth_histograms_ips,
    scale_half_eligibility,
    tempered_depth_sampling_weights,
)


def test_depth_histograms_count_only_valid_pixels():
    depth = np.array(
        [
            [0.4, 0.4, 2.0, 2.0],
            [0.4, 0.0, 2.0, 2.0],
            [0.8, 0.8, 1.2, 1.2],
            [0.8, 0.8, 1.2, 0.0],
        ],
        dtype=np.float32,
    )
    histograms = depth_histograms_ips(
        depth,
        tops=np.array([0, 0, 2]),
        lefts=np.array([0, 2, 0]),
        patch_size=2,
        min_depth=0.4,
        max_depth=2.0,
        bins=16,
    )
    assert histograms.shape == (3, 16)
    assert histograms.sum(axis=1).tolist() == [3, 4, 4]
    assert histograms[0, 0] == 3
    assert histograms[1, 15] == 4


def test_tempered_weights_are_scene_normalized_bounded_and_effective():
    histograms = np.array(
        [
            [100, 0],
            [100, 0],
            [0, 100],
            [100, 0],
            [100, 0],
            [0, 100],
        ],
        dtype=np.float64,
    )
    scene_ids = np.array(['scene_01'] * 3 + ['scene_02'] * 3)
    weights, report = tempered_depth_sampling_weights(
        histograms,
        scene_ids,
        target_exponent=0.5,
        minimum=0.25,
        maximum=4.0,
    )
    assert weights.min() >= 0.25
    assert weights.max() <= 4.0
    assert weights[2] > weights[0]
    assert weights[5] > weights[3]
    assert weights[:3].mean() == pytest.approx(1.0, abs=1e-6)
    assert weights[3:].mean() == pytest.approx(1.0, abs=1e-6)
    assert report['ess_ratio'] > 0.5
    assert report['weighted_distribution'][1] > report['source_distribution'][1]


def test_scale_half_eligibility_rejects_bounds_and_validation_overlap():
    eligible = scale_half_eligibility(
        train_tops=np.array([64, 64, 0, 400]),
        train_lefts=np.array([64, 256, 64, 256]),
        val_tops=np.array([0]),
        val_lefts=np.array([0]),
        image_height=512,
        image_width=512,
        patch_size=128,
    )
    assert eligible.tolist() == [False, True, False, False]
