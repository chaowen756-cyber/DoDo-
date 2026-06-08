# Review: DoDo-change Round 23 Intensity-Sensing Minimal Implementation

Date: 2026-05-09

## Scope

Reviewed:

- `handoff/DoDo-change/CURRENT_CODE_CHANGES.md`
- `handoff/DoDo-change/EXPERIMENTS.md`
- `handoff/DoDo-change/implementation-notes.md` Section 23
- `handoff/DoDo-change/BEST_CHECKPOINT.md`
- `handoff/DoDo-change/ROUND21_CODE_PLAN.md`
- `reviews/review-DoDo-change-round21-code-plan-2026-05-09.md`
- `torch_optics/sensing.py`
- `torch_optics/forward_dodo.py`
- `snapshotdepth_hs.py`
- `infer_contect.py`
- `round23_lowfg_diag.py`

Claude reported these validation results:

- SensingLayer deterministic smoke: PASS, amplitude/intensity manual diffs `0.00e+00`.
- Backward compatibility: PASS, default vs explicit amplitude diff `0`.
- Round 22 regression: PASS, soft-diopter pytest `8/8`.
- DoDo forward smoke: PASS, intensity + hard/soft/valid_mask finite.
- Tiny 1-epoch preflight: PASS, exit 0, nonfinite 0, hparams records `dodo_sensor_measurement=intensity`.
- LowFG diagnostic: completed, deploy 1 found 491 low-valid tiles, deploy 16 found 406 low-valid tiles.

## Findings

### 1. LowFG diagnostic does not measure optical measurement energy

Severity: High

`round23_lowfg_diag.py` is documented and scheduled as a low-foreground **measurement energy** diagnostic: it should compute full / foreground-only / background-only measurement energy. However the implementation only loads the HS cube from cache and computes input spectral energy:

- `round23_lowfg_diag.py:63-76` loads HS data, creates foreground/background masks, and computes `tile.abs().sum()`, `(tile.abs() * fg_mask).sum()`, `(tile.abs() * bg_mask).sum()`.
- It does not load a checkpoint or instantiate `SnapshotDepthHS` / `DepthAwareDoDoForwardModel`.
- It does not run `camera(...)`, `SensingLayer`, `sensor_measurement`, propagation, DOE, or measurement normalization.

This means the diagnostic output is not the requested optical measurement-energy decomposition. It only shows that many low-valid tiles are mostly background in the input HS cube. That is useful as a tile-distribution statistic, but it does not answer the Round 20/Round 23 question about background contribution to DoDo measurement energy.

Required fix:

- Either rename/reclassify the current script and documentation as an HS input-energy / low-foreground tile census, not a measurement-energy diagnostic; or preferably implement the required measurement-energy diagnostic.
- For the required diagnostic, load the DoDo model/checkpoint or instantiate the optical camera and compute at least:
  - `measurement_full = camera(hs, depth_metric, valid_mask=mask)` or equivalent SnapshotDepthHS capture path;
  - `measurement_fg = camera(hs * mask, depth_metric, valid_mask=mask)`;
  - `measurement_bg = camera(hs * (1-mask), depth_metric, valid_mask=(1-mask or explicit background handling))`, with the chosen valid-mask policy documented;
  - energy statistics from `measurement_*` tensors, not raw HS tensors.
- Record `sensor_measurement`, `depth_layering_mode`, checkpoint path, and measurement normalization policy in `diag_lowfg_summary.json`.
- Re-run Step 5 and update `EXPERIMENTS.md` and Section 23.

Until this is fixed, Round 23's low-foreground diagnostic should not be treated as satisfying the measurement-energy requirement.

### 2. Oracle metadata is written unconditionally, not gated on DoDo oracle usage

Severity: Low

`infer_contect.py` now writes `oracle_simulation=True` to output contracts:

- `metrics_real.txt`: header and rows at `infer_contect.py:940` and `infer_contect.py:1003-1009`.
- `metrics_full_scene.csv`: `infer_contect.py:561-570`.
- `diagnostic_metrics.json`: `scenario_metrics['oracle_simulation'] = True` at `infer_contect.py:598`.
- `aggregate_metrics.json`: `oracle_simulation=True` at `infer_contect.py:1023`.

This covers the requested output contracts for DoDo-depth oracle simulation. The minor issue is that the flag is unconditional in `main()` / diagnostic output rather than explicitly tied to `optical_model == 'dodo_depth'` and GT depth being passed into measurement synthesis. If this script is also used for legacy camera simulation, the label may be imprecise.

Recommended fix:

- Define `oracle_simulation = (getattr(model.hparams, 'optical_model', 'legacy_camera') == 'dodo_depth')` in `main()` and pass it into `process_single_scene(...)`, or document why all current full-scene paths are simulation/oracle.
- Also include `dodo_sensor_measurement` in per-scene `diagnostic_metrics.json` when available, not only aggregate metadata.

This is not a blocker for the intensity-sensing implementation, but it should be cleaned up with the LowFG diagnostic fix.

## Accepted Items

### SensingLayer implementation

Accepted.

- `torch_optics/sensing.py:37` defines valid choices.
- `torch_optics/sensing.py:50-64` adds `sensor_measurement` with default `amplitude` and validation.
- `torch_optics/sensing.py:100-102` preserves old `torch.abs(x)` behavior for amplitude and applies `x_abs ** 2` for intensity.
- Existing `rgb`, `spectral_bins`, and `identity` branches share the same `x_abs` tensor, so all modes are covered.

### Forward-model passthrough

Accepted.

- `DoDoForwardModel.__init__` accepts `sensor_measurement` and passes it to `SensingLayer` at `torch_optics/forward_dodo.py:158-200`.
- `DepthAwareDoDoForwardModel.__init__` accepts and passes it at `torch_optics/forward_dodo.py:256-334`.
- `Forward_DM_Spiral_Depth(...)` accepts and passes it at `torch_optics/forward_dodo.py:494-513`.
- Default remains `amplitude`, so old construction paths remain valid.

### SnapshotDepthHS CLI / hparams path

Accepted.

- `snapshotdepth_hs.py:668` uses `getattr(..., 'amplitude')` for old checkpoint compatibility.
- `snapshotdepth_hs.py:680-695` passes `sensor_measurement` into `DepthAwareDoDoForwardModel`.
- `snapshotdepth_hs.py:1314-1316` adds `--dodo_sensor_measurement {amplitude,intensity}`.

### Runtime validation coverage

Accepted for code paths except the LowFG measurement-energy semantic issue.

The reported smoke tests and one-epoch preflight are sufficient to establish that the intensity path is finite and backward-compatible at a smoke-test level. They are not sufficient to justify long training or checkpoint promotion.

## Conclusion

Round 23's core opt-in intensity sensing implementation is accepted with one blocking diagnostic finding:

- Do not use the current `round23_lowfg_diag.py` output as evidence about optical measurement energy.
- Fix or reclassify the LowFG diagnostic before using it to drive training decisions.

No long training should start yet. The next action should be a small Round 23 repair focused on LowFG measurement-energy semantics and oracle metadata cleanup, followed by Codex re-review.
