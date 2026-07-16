import numpy as np
import pytest
import torch

from datasets.baek_augmentation import (
    CIE_ILLUMINANT_NAMES,
    apply_cie_illuminant,
    cie_illuminants,
    translate_metric_depth,
)
from datasets.hyperspectral_dataset import HyperspectralDepthDataset
from util.helper import metric_to_ips


def test_cie_illuminant_set_matches_baek_configuration():
    illuminants = cie_illuminants()
    assert illuminants.shape == (29, 25)
    assert CIE_ILLUMINANT_NAMES[:8] == (
        'A', 'B', 'C', 'D50', 'D55', 'D65', 'D93', 'E'
    )
    assert torch.allclose(illuminants[7], torch.ones(25))


def test_flat_illuminant_only_changes_exposure():
    hs_image = torch.full((1, 25, 3, 4), 0.25)
    augmented = apply_cie_illuminant(hs_image, -1, 1.1)
    assert torch.allclose(augmented, torch.full_like(hs_image, 0.275))


def test_depth_translation_uses_one_feasible_global_shift():
    depth = torch.tensor([[[[0.45, 1.20, 1.90], [0.40, 0.40, 0.40]]]])
    mask = torch.tensor([[[[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]]])
    shifted, applied = translate_metric_depth(depth, mask, 0.20, 0.4, 2.0)
    assert applied == pytest.approx(0.10, abs=1e-6)
    assert torch.allclose(
        shifted[0, 0, 0], torch.tensor([0.55, 1.30, 2.00]), atol=1e-6
    )
    assert torch.all(shifted[mask == 0] == 0.4)


def test_half_scale_resizes_all_modalities_and_recomputes_ips():
    dataset = HyperspectralDepthDataset.__new__(HyperspectralDepthDataset)
    dataset.image_size = (2, 2)
    dataset.hs_channels = 25
    dataset.min_depth = 0.4
    dataset.max_depth = 2.0
    dataset.hs_norm_mode = 'fixed_scale'
    dataset.hs_norm_scale = 1.0
    dataset.hs_sanity_threshold = 10000.0
    dataset.baek_augment = True
    dataset.baek_illuminant_probability = 0.0
    dataset.baek_exposure_min = 1.0
    dataset.baek_exposure_max = 1.0
    dataset.baek_depth_shift_probability = 0.0
    dataset.baek_depth_shift_m = 0.2
    dataset.baek_max_clip_ratio = 0.001
    dataset.baek_illuminant_retries = 8

    base = np.arange(16, dtype=np.float32).reshape(4, 4) / 100.0
    hs_image = np.repeat(base[:, :, None], 25, axis=2)
    depth_m = np.array(
        [
            [0.4, 0.5, 0.6, 0.7],
            [0.8, 0.9, 1.0, 1.1],
            [1.2, 1.3, 1.4, 1.5],
            [1.6, 1.7, 1.8, 0.0],
        ],
        dtype=np.float32,
    )
    depth_mm = depth_m * 1000.0
    hs, ips, metric, mask, metadata = dataset._patch_tensors_from_arrays(
        hs_image,
        depth_mm,
        top=1,
        left=1,
        hs_cache_key='unused',
        spatial_scale=0.5,
    )

    assert hs.shape == (1, 25, 2, 2)
    assert metric.shape == mask.shape == ips.shape == (1, 1, 2, 2)
    assert metadata['aug_scale_factor'].item() == 0.5
    assert set(mask.unique().tolist()) <= {0.0, 1.0}
    expected_ips = metric_to_ips(metric, 0.4, 2.0).clamp(0.0, 1.0)
    assert torch.allclose(ips, expected_ips)
    assert torch.all(metric[mask == 0] == 0.4)


def test_validation_dataset_never_enables_baek_augmentation(tmp_path):
    dataset = HyperspectralDepthDataset(
        base_dir=str(tmp_path),
        scene_folders=[],
        image_size=(128, 128),
        hs_channels=25,
        is_training=False,
        hs_norm_mode='fixed_scale',
        hs_norm_scale=0.9685,
        baek_augment=True,
    )
    assert dataset.baek_augment is False
