# Review: DoDo-change Round 11 Pareto Results

Date: 2026-05-07
Reviewer: Codex
Change: DoDo-change

## Findings

1. Medium - Checkpoint selection works, but checkpoint filenames are misleading.

   In [snapshotdepth_trainer_hs.py](/Users/chao.wen3/myserver/snapshotdepth_trainer_hs.py:132), the filename template replaces `/` with `_` while the monitored metrics are logged as `validation/psnr_hs_masked` and `validation/mae_depth_m`. The Round 11 logs show correct callback decisions, but saved filenames contain `validation_* = 0.0000`. This does not invalidate the experiments, but it is a reproducibility and operator-error risk before another long run. Next round should fix checkpoint filenames and preferably put checkpoints under the resolved artifact root.

2. Medium - 120 epoch PSNR is not enough evidence for an architecture/data ceiling.

   Section 11 shows all 120 epoch depth-focused runs around 24.5-24.9 dB, but R10 Main reached only about 25.5 dB near epoch 125 and then improved to 31.11 dB by epoch 259. The current evidence supports "depth-focused objectives slow or suppress early HS PSNR", not "120 epoch is the current upper bound". Candidate B should be extended only because its PSNR curve is still rising and its depth-best checkpoint is strong.

3. Low - Do not add depth-MAE early stopping yet.

   Candidate B depth MAE best is 0.242m at epoch 117, while last is 0.339m, but PSNR is still improving at epoch 119. Early stopping on depth MAE would likely stop before the HS reconstruction catches up. The correct mechanism for now is dual checkpointing plus explicit reporting of PSNR-best and depth-best metrics.

4. Low - Metric-depth loss does not need code-level rescaling before the next run.

   The implementation in [snapshotdepth_hs.py](/Users/chao.wen3/myserver/snapshotdepth_hs.py:967) normalizes meter residuals by depth range and applies SmoothL1 over valid pixels. Although the logged unweighted metric loss is smaller than IPS depth loss, Candidate B already improves best metric MAE from 0.307m to 0.242m at similar PSNR. Increasing `--metric_depth_loss_weight` to 2 or 5 is a reasonable later sweep, but doing it before the `metric_depth_loss_weight=1` long curve would risk unnecessary PSNR loss.

## Answers To The Round 11 Questions

1. PSNR bottleneck: no, the four 120 epoch runs do not prove a hard ceiling. R10 Main's +5.6 dB from epoch 125 to 259 was most likely dominated by longer optimization under the weaker depth objective. Round 11 candidates have similar 120 epoch PSNR because depth pressure is higher or more direct, but both curves were still rising.

2. Depth MAE best vs last: the gap is consistent with a late multi-objective trade-off and validation noise, not a training failure. Use depth-MAE-best checkpoint for depth reporting and PSNR-best checkpoint for HS reporting. Do not add depth early stopping before seeing the 260 epoch curve.

3. Metric-depth loss vs IPS depth loss: metric-space supervision is clearly more aligned with the target metric. Keep `depth_loss_weight=1` and `metric_depth_loss_weight=1` for the next long run. Only sweep `metric_depth_loss_weight=2` if the 260 epoch run preserves PSNR but depth-best MAE does not improve or is unstable.

4. DOE trainability: DOE gradients are finite and non-zero across runs. There is no evidence of DOE saturation at 120 epochs because PSNR is still climbing. Keep `--optics_lr 1e-6`; do not change optics LR in the next run.

5. Next experiment: approve one long Candidate B continuation as a fresh 260 epoch run, with checkpoint/artifact cleanup first. Gate extension analysis at epoch 180/200 and final 260.

6. Code changes: no training-objective code change is required. Required cleanup is reproducibility-only: checkpoint filename/dirpath handling, plus a 1 epoch preflight to confirm artifact and checkpoint outputs.

## Recommended Next Round

Run `DoDo_depth_finite_joint_metricdepth_260_v1` with:

```bash
--depth_loss_weight 1
--metric_depth_loss_weight 1
--max_epochs 260
--optics_lr 1e-6
--cnn_lr 5e-5
```

Approval criteria after the run:

- `nonfinite_count == 0`.
- PSNR-best reaches at least 28 dB, with target 30+ dB if the R10 late-epoch behavior repeats.
- Depth-MAE-best remains `<= 0.25m`; last depth MAE should be reported separately and should ideally stay `<= 0.35m`.
- `est_depth_std >= 0.02`.
- DOE and decoder gradients remain finite.
- Quicklooks show non-constant HS reconstruction and localized depth errors.

