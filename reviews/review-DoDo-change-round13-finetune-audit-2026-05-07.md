# Review: DoDo-change Round 13 Checkpoint Eval And Fine-Tunes

Date: 2026-05-07
Reviewer: Codex
Change: DoDo-change

## Findings

1. High - Section 13 misreports fine-tune depth-best MAE.

   Section 13 states that FT-A and FT-B have best depth MAE around 0.339/0.340 at epoch 0. The training logs show the opposite: both fine-tunes saved `depth-best-epoch=000.ckpt` with `validation/mae_depth_m` about 0.1805m. The 0.339/0.340 values are the final epoch metrics from `metrics.json`, not the depth-best checkpoint metrics. This changes the interpretation: fine-tuning does not destroy depth immediately at the first validation, but later PSNR improvement still trades depth away.

2. Medium - Fine-tune PSNR-best checkpoints likely fail the depth criterion, but need standalone eval artifacts.

   Logs show FT-A/FT-B PSNR-best at epoch 36 with PSNR about 31.02/31.03 dB, while final metrics show depth MAE around 0.339/0.340m. That suggests they do not beat the Pareto criterion, but the correct next step is to evaluate each saved fine-tune checkpoint with the validation-only path and compare same-checkpoint PSNR and depth MAE.

3. Low - Metrics persistence and validation-only eval are now fixed enough to rely on.

   The metrics persistence preflight wrote non-zero train losses, and R12 depth-best / PSNR-best checkpoint eval produced standalone metrics and quicklooks. These pieces are usable for final model selection.

## Corrected Interpretation

The current reliable Pareto anchor remains R12 `depth-best-epoch=226.ckpt`:

- R12 depth-best eval: PSNR 30.1345 dB, MAE 0.17836m.
- R12 PSNR-best eval: PSNR 31.0999 dB, MAE 0.49390m.

The fine-tunes need a same-checkpoint audit:

- FT depth-best checkpoints may preserve depth around 0.180m but probably add only about 0.025 dB PSNR.
- FT PSNR-best checkpoints likely recover PSNR around 31.02 dB but likely degrade depth to about 0.34m.

Unless the standalone fine-tune checkpoint eval proves a checkpoint with PSNR at least 30.43 dB and MAE <= 0.22m, keep R12 depth-best as the best DoDo-depth model.

## Recommended Next Round

Do not run more training. Run validation-only evaluation on the four fine-tune checkpoints, correct the Section 13 record in a new Section 14, and write a best-checkpoint handoff document that names the final selected checkpoint and metrics.

