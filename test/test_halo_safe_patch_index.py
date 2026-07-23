import json

import numpy as np

from scripts.build_halo_safe_patch_index import build_halo_safe_index


def test_halo_index_recomputes_scale_eligibility_and_preserves_other_fields(
        tmp_path):
    train_path = tmp_path / 'train.npz'
    val_path = tmp_path / 'val.npz'
    output_path = tmp_path / 'halo.npz'
    meta = {
        'patch_size': 2,
        'scene_stats': [
            {'scene_id': 'scene_01', 'height': 20, 'width': 20},
        ],
    }
    np.savez_compressed(
        train_path,
        scene_ids=np.asarray(['scene_01'] * 3),
        tops=np.asarray([0, 6, 12], dtype=np.int32),
        lefts=np.asarray([0, 6, 12], dtype=np.int32),
        scores=np.asarray([1.25, 2.5, 3.75], dtype=np.float32),
        category_masks=np.asarray([1, 2, 4], dtype=np.uint8),
        scale_05_eligible=np.asarray([True, True, False]),
        meta_json=np.asarray(json.dumps(meta)),
    )
    np.savez_compressed(
        val_path,
        scene_ids=np.asarray(['scene_01']),
        tops=np.asarray([2], dtype=np.int32),
        lefts=np.asarray([2], dtype=np.int32),
    )

    stats = build_halo_safe_index(
        train_path, val_path, output_path, halo=1)

    with np.load(output_path, allow_pickle=False) as result:
        assert result['tops'].tolist() == [6, 12]
        assert result['lefts'].tolist() == [6, 12]
        assert result['scores'].dtype == np.float32
        assert result['scores'].tolist() == [2.5, 3.75]
        assert result['category_masks'].tolist() == [2, 4]
        # halo context=4, so half-scale source=8.  The first retained source
        # still overlaps validation; the second is both in-bounds and disjoint.
        assert result['scale_05_eligible'].tolist() == [False, True]
    assert stats == {
        'source_count': 3,
        'retained_count': 2,
        'dropped_count': 1,
        'scale_05_eligible_count': 1,
    }


def test_halo_index_drops_scale_one_context_outside_scene(tmp_path):
    train_path = tmp_path / 'train.npz'
    val_path = tmp_path / 'val.npz'
    output_path = tmp_path / 'halo.npz'
    meta = {
        'patch_size': 4,
        'scene_stats': [
            {'scene_id': 'scene_01', 'height': 20, 'width': 20},
        ],
    }
    np.savez_compressed(
        train_path,
        scene_ids=np.asarray(['scene_01', 'scene_01']),
        tops=np.asarray([0, 8], dtype=np.int32),
        lefts=np.asarray([0, 8], dtype=np.int32),
        scores=np.asarray([1.0, 2.0], dtype=np.float32),
        scale_05_eligible=np.asarray([True, True]),
        meta_json=np.asarray(json.dumps(meta)),
    )
    np.savez_compressed(
        val_path,
        scene_ids=np.asarray([], dtype='<U8'),
        tops=np.asarray([], dtype=np.int32),
        lefts=np.asarray([], dtype=np.int32),
    )
    build_halo_safe_index(train_path, val_path, output_path, halo=2)
    with np.load(output_path, allow_pickle=False) as result:
        assert result['tops'].tolist() == [8]
        assert result['lefts'].tolist() == [8]
        assert result['scores'].tolist() == [2.0]
        saved_meta = json.loads(str(result['meta_json'].item()))
        assert saved_meta['halo_context_in_bounds'] is True
