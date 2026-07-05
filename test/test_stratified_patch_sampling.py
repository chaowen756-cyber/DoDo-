from collections import Counter

import torch

from datasets.hyperspectral_dataset import (
    HyperspectralDepthDataset,
    parse_patch_category_mix,
)


def test_stratified_schedule_matches_global_quota():
    dataset = HyperspectralDepthDataset.__new__(HyperspectralDepthDataset)
    dataset.patch_category_mix = parse_patch_category_mix(
        'depth_hard=0.4,hs_bright=0.2,hs_complex=0.2,general=0.2'
    )
    dataset.patch_category_seed = 123
    dataset.samples_per_epoch = 6143
    dataset.sample_pairs = [{'id': f'scene_{scene:02d}'} for scene in range(1, 16)]
    dataset.patch_index_category_bits = {
        'depth_hard': 1,
        'hs_bright': 2,
        'hs_complex': 4,
        'general': 8,
    }
    dataset.patch_index_by_id = {
        sample['id']: {
            'category_masks': torch.tensor([1, 2, 4, 8, 15], dtype=torch.uint8)
        }
        for sample in dataset.sample_pairs
    }
    dataset.stratified_category_schedule = {}

    dataset._build_stratified_category_schedule()

    counts = Counter()
    scene_counts = Counter()
    for idx in range(dataset.samples_per_epoch):
        sample_id = dataset.sample_pairs[idx % len(dataset.sample_pairs)]['id']
        counts[dataset._stratified_category_for_index(sample_id, idx)] += 1
        scene_counts[sample_id] += 1

    assert counts == {
        'depth_hard': 2457,
        'hs_bright': 1229,
        'hs_complex': 1229,
        'general': 1228,
    }
    assert set(scene_counts.values()) == {409, 410}
