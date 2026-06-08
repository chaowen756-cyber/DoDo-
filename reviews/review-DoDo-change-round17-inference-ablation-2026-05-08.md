# Review: DoDo-change Round 17 Inference-Only Ablation

Date: 2026-05-08
Reviewer: Codex
Change: DoDo-change

## Findings

1. No blocking issue in the Round 17 conclusion.

   Round 17 completed the required inference-only ablations and all acceptance criteria failed. The results are consistent across both deploy folders: disabling second-stage `dodo_measurement_norm` has no effect, minmax provides only small depth changes while spectral quality remains poor, and low-foreground tile skipping skips only about 0.1% of tiles with no useful quality improvement.

2. High - inference-time mitigation path is exhausted.

   deploy 1 best reported values remain far from acceptance: SAM about 0.540, pseudo-RGB PSNR about 26.40 dB, and `mae_vs_median` about 1.027. deploy 16 remains worse: SAM about 0.412 and `mae_vs_median` about 1.133. No setting approaches `SAM <= 0.30` or depth MAE 20% better than median-depth baseline.

3. High - the primary normalization target shifts from `snapshotdepth_hs.py` to `torch_optics/forward_dodo.py`.

   Round 17 shows `--measurement_norm_override none` and checkpoint mode are numerically identical. That means the second-stage `dodo_measurement_norm` is not the dominant cause during inference. The next experiment must make the internal `_normalize_once(y_sum)` behavior in `DepthAwareDoDoForwardModel.forward()` configurable and train a new candidate rather than trying more inference flags on the old checkpoint.

4. Medium - full-scene gates must become promotion gates.

   Prior checkpoint selection used crop metrics such as masked HS PSNR and depth MAE. Round 15-17 show those are insufficient. Any new training run must be judged by full-scene SAM, pseudo-RGB PSNR, depth-vs-median baseline, and visual quicklooks before approving long training.

## Assessment

Round 17 closes the inference-only branch. The next useful work is a small, controlled training/architecture round: expose the DoDo forward measurement normalization mode, add background-aware loss and honest validation metrics, run short candidates, then evaluate full-scene results before any long training.
