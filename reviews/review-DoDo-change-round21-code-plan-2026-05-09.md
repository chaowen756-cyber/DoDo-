# Review: DoDo-change Round 21 Intensity-Sensing Code Plan

Date: 2026-05-09

## Scope

Reviewed:

- `handoff/DoDo-change/ROUND21_CODE_PLAN.md`
- `handoff/DoDo-change/CURRENT_CODE_CHANGES.md`
- `handoff/DoDo-change/implementation-notes.md` Section 21
- `handoff/DoDo-change/BEST_CHECKPOINT.md`
- `torch_optics/sensing.py`
- `torch_optics/forward_dodo.py`
- `snapshotdepth_hs.py`
- `infer_contect.py`

## Findings

### 1. Plan is approved for minimal implementation with amendments

Severity: Low

The plan correctly keeps legacy behavior as default with `amplitude`, adds opt-in `intensity`, covers `rgb` / `spectral_bins` / `identity`, and avoids long training. This is the right next step after Round 20.

### 2. Backward-compatibility test needs a stronger baseline

Severity: Medium

Comparing `DepthAwareDoDoForwardModel(..., sensor_measurement='amplitude')` against `DepthAwareDoDoForwardModel(...)` after the patch only compares two new-code paths. Round 22 should also compare `SensingLayer(..., sensor_measurement='amplitude')` against a manual old-formula implementation:

- RGB: `sum(abs(x) * sensor_{r,g,b})`
- spectral/identity: `sum(abs(x) * response)`

This proves default behavior remains compatible with old checkpoints.

### 3. Oracle label must be written to all full-scene output contracts

Severity: Medium

`infer_contect.py` currently writes multiple metrics outputs:

- top-level `metrics_real.txt`
- top-level `aggregate_metrics.json`
- per-scene `metrics_full_scene.csv`
- per-scene `diagnostic_metrics.json`

Round 22 must add `oracle_simulation=True` consistently to all of these when GT depth is used to synthesize DoDo-depth measurement. Adding it only to `metrics_full_scene.csv` is incomplete.

### 4. Diagnostic scripts must not hard-code `/root/autodl-tmp`

Severity: Medium

Round 20 scripts used absolute `/root/autodl-tmp` paths. Round 22's low-foreground diagnostic should resolve paths relative to the current workspace or accept CLI args for data root, checkpoint, deploy folders, and output root.

## Approval

Approved to implement Round 22 with the amendments above.

Round 22 must not run long training. It may run unit/smoke checks, finite forward smoke, one tiny 1-epoch preflight, and low-foreground diagnostics only.
