# Review: DoDo-change Round 12 Metric-Depth 260 Epoch

Date: 2026-05-07
Reviewer: Codex
Change: DoDo-change

## Findings

1. High - The best current model is the depth-MAE-best checkpoint, not the final or PSNR-best checkpoint.

   Round 12 produced a strong Pareto point: depth-best at epoch 226 has `validation/mae_depth_m = 0.178m` and PSNR around 30.15 dB, while PSNR-best at epoch 259 has 31.10 dB but depth MAE regresses to 0.494m. The final checkpoint is not acceptable for depth-sensitive use. Treat `infer_results/DoDo-change/DoDo_depth_finite_joint_metricdepth_260_v1/20260507_112631/checkpoints/depth-best-epoch=226.ckpt` as the current candidate model until a better Pareto point is proven.

2. Medium - Depth-best metrics and quicklooks need a standalone checkpoint evaluation artifact.

   Section 12 reports the depth-best MAE from the checkpoint callback and estimates the depth-best PSNR from the training timeline. That is enough to identify the best epoch, but it is not enough for final reporting. Next round should add/run a validation-only checkpoint evaluation path for both `depth-best-epoch=226.ckpt` and `psnr-best-epoch=259.ckpt`, saving separate `metrics.json` and quicklook PNGs for each.

3. Medium - `metrics.json` train losses are still wrong in Round 12.

   `metrics.json` again contains `train_loss/* = 0.0`. The likely root cause is in [snapshotdepth_hs.py](/Users/chao.wen3/myserver/snapshotdepth_hs.py:293): `training_step` stores `loss_logs` after the keys have already been prefixed as `train_loss/...`, while `_save_validation_artifacts` later looks for unprefixed keys such as `total_loss` in [snapshotdepth_hs.py](/Users/chao.wen3/myserver/snapshotdepth_hs.py:518). This is not blocking model quality, but it is blocking reliable experiment accounting.

4. Low - The late-epoch MAE regression is now a confirmed optimization trade-off, not a stability failure.

   `nonfinite_count = 0`, DOE gradients are finite, and PSNR rises normally to 31.10 dB. The regression from 0.178m at epoch 226 to 0.494m at epoch 259 is therefore most likely HS/depth objective competition in the shared decoder/DOE optimization. Do not run another full 260 epoch from scratch before validating whether short continuation from epoch 226 can improve PSNR while preserving depth.

## Round 12 Assessment

Round 12 should be considered successful:

- Checkpoint reproducibility is fixed: checkpoint files are under `artifact_root/checkpoints/` and filenames no longer contain fake `0.0000` values.
- PSNR fully recovers to R10 level: 31.10 dB vs R10 31.11 dB.
- Depth-best MAE is the best so far: 0.178m, a 65% improvement over R10 Main's 0.516m final MAE.
- The remaining problem is checkpoint selection and late-epoch trade-off control, not DoDo numerical stability.

## Recommended Next Round

Do not start another long run yet. Run Round 13 as a consolidation and short fine-tune round:

1. Fix `metrics.json` train-loss persistence.
2. Add validation-only checkpoint evaluation and evaluate both R12 checkpoints.
3. If depth-best evaluation confirms PSNR >= 30.0 dB and MAE <= 0.20m, register epoch 226 as the current best model.
4. Optionally run at most two short 40 epoch fine-tunes initialized from depth-best:
   - Low-LR preservation: `metric_depth_loss_weight=1`, `cnn_lr=1e-5`, `optics_lr=2e-7`.
   - Stronger depth anchor: `metric_depth_loss_weight=2`, `cnn_lr=1e-5`, `optics_lr=2e-7`.

Approve a fine-tune only if it improves PSNR toward 31 dB while keeping depth MAE near the depth-best checkpoint. Otherwise keep epoch 226 as the best current result.

