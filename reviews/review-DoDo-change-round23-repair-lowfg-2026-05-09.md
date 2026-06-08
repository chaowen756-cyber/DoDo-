# Review: DoDo-change Round 23 Repair LowFG Measurement-Energy Diagnostic

Date: 2026-05-09

## Scope

Reviewed Claude's Round 23 repair summary and inspected:

- `round23_lowfg_diag.py`
- `infer_contect.py`
- `handoff/DoDo-change/CURRENT_CODE_CHANGES.md`
- `handoff/DoDo-change/EXPERIMENTS.md`
- `handoff/DoDo-change/implementation-notes.md`
- `handoff/DoDo-change/BEST_CHECKPOINT.md`

## Result

Accepted. The Round 23 repair resolves the previous blocker.

## Accepted Items

### 1. LowFG diagnostic now computes DoDo measurement energy

`round23_lowfg_diag.py` now loads a DoDo checkpoint via `SnapshotDepthHS.load_from_checkpoint(...)`, extracts `model.camera`, and computes actual optical measurements with `camera(...)`:

- full measurement: `camera(hs_batch, depth_batch, valid_mask=vm_batch)`;
- foreground-only: `camera(hs_fg, depth_batch, valid_mask=vm_batch)`;
- background-only: `camera(hs_bg, depth_bg, valid_mask=bg_mask)`.

The output rows include true measurement columns:

- `measurement_full_energy`
- `measurement_fg_energy`
- `measurement_bg_energy`
- `measurement_fg_fraction`
- `measurement_bg_fraction`

Raw HS energy columns are retained only as diagnostic reference, which is acceptable.

### 2. Background policy is explicit

The script documents and records the background policy:

- `bg_hs = hs * (1-fg_mask)`;
- invalid/background depth is clamped to `[min_depth, max_depth]`;
- `bg_valid_mask = 1-fg_mask` for background-only synthesis.

This is now sufficient for interpreting the diagnostic.

### 3. Oracle metadata cleanup is acceptable

`infer_contect.py` now gates `oracle_simulation` through `is_dodo_model` instead of writing an unconditional `True` in the main output path. Per-scene diagnostic metadata also records `dodo_sensor_measurement` for DoDo runs.

### 4. Validation and cleanup

Claude reported:

- `python -m py_compile round23_lowfg_diag.py infer_contect.py`: exit 0.
- `pytest -q test/test_soft_diopter_depth_layering.py`: 8/8 passed.
- `python round23_lowfg_diag.py --deploys "deploy 1" "deploy 16" --max_measure_tiles 5`: completed.
- Large preflight checkpoints were deleted while preserving summary artifacts.

The key diagnostic result is now meaningful: measured low-foreground pure-background tiles produce finite nonzero DoDo measurement energy, with `measurement_bg_fraction=1.0` for the sampled valid_ratio=0 tiles. This supports the current hypothesis that background tiles plus per-patch normalization contribute to full-scene white/background artifacts.

## Minor Notes

- `CURRENT_CODE_CHANGES.md` still labels the script location as `/root/autodl-tmp/round23_lowfg_diag.py`; in this checkout the file is at repository root `round23_lowfg_diag.py`. This is a documentation path nit, not a blocker.
- No checkpoint was promoted. Full-scene deployment remains not approved.

## Conclusion

Round 23 intensity sensing and LowFG measurement-energy diagnostic repair are complete enough to proceed to the next scoped change. Do not start long training solely from this result; the next work item should be planned explicitly, especially if it changes spectral modeling or spectral losses/metrics.
