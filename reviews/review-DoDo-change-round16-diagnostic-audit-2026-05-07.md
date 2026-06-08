# Review: DoDo-change Round 16 Diagnostic Audit

Date: 2026-05-07
Reviewer: Codex
Change: DoDo-change

## Findings

1. No blocking issue in the main diagnosis.

   Round 16 correctly reclassifies Round 15 as a failed full-scene result. The new evidence is decisive: deploy 1 has masked PSNR 30.02 dB but SAM 0.540 rad and pseudo-RGB PSNR 26.95 dB; depth MAE 0.453m is worse than the constant median-depth baseline 0.424m. deploy 16 is also worse than constant-depth baselines. The checkpoint must not be treated as full-scene usable.

2. High - `BEST_CHECKPOINT.md` still contains stale full-scene success language.

   The file now has a correct Round 16 warning, but the older bullets below it still say HS PSNR is preserved, depth degradation is expected, and deploy 16 is not a model failure. Those statements conflict with the warning and should be rewritten so future readers do not quote the obsolete Round 15 conclusion.

3. Medium - measurement normalization evidence is directionally correct but the stats capture is incomplete.

   `diagnostic_metrics.json` reports `captimgs_stats.before_norm` and `after_norm` as all zeros for both deploy 1 and deploy 16. That means the current capture likely records only the final/zero diagnostic patch or the wrong tensor, not aggregate measurement statistics over all tiles. The root cause ranking can still rely on metric/ROI evidence, but Round 17 should fix measurement-stat aggregation before making a final normalization claim.

4. Medium - reporting contract is partially implemented.

   Required arrays, `metrics_per_band.csv`, `diagnostic_metrics.json`, quicklooks, and ROI metrics exist. However, the requested separate `metrics_regions.csv`, `metrics_depth_baselines.csv`, and `metrics_spectral_quality.csv` are not present as standalone CSVs; their contents are embedded in JSON. This is acceptable for diagnosis but should be standardized before further comparisons.

5. Low - optical forward citation needs correction.

   Section 16 says `forward_dodo.py` uses `y_stack.sum(dim=1)`, but the current code accumulates per-depth responses with `y_sum = y_k if y_sum is None else y_sum + y_k`. The conclusion is still correct: depth-layer optical responses are summed into a 3-channel measurement. The citation should be corrected in the next notes update.

## Assessment

Round 16 establishes that the current checkpoint is only valid for filtered 128x128 crop evaluation, not full-scene deployment. The next useful step is not another long training run. First, run a controlled inference-only ablation to determine whether normalization/tiling/reporting fixes can improve full-scene behavior without retraining. If that fails, the evidence supports opening an architecture/training change: remove/rethink per-patch measurement normalization, include low-foreground/full-scene tiles, and use honest full-scene metrics as promotion gates.
