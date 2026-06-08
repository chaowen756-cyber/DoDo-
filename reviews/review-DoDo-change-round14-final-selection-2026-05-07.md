# Review: DoDo-change Round 14 Final Checkpoint Selection

Date: 2026-05-07
Reviewer: Codex
Change: DoDo-change

## Findings

1. No blocking findings in final checkpoint selection.

   Section 14 correctly fixes the Section 13 fine-tune interpretation. Same-checkpoint eval shows no fine-tune checkpoint satisfies the promotion rule. The final selected checkpoint in `BEST_CHECKPOINT.md` is correct:

   ```text
   infer_results/DoDo-change/DoDo_depth_finite_joint_metricdepth_260_v1/20260507_112631/checkpoints/depth-best-epoch=226.ckpt
   ```

2. Medium - Full-scene inference is not yet validated for DoDo-depth.

   The existing full-scene inference file `infer_contect.py` is designed as a tiled inference/evaluation script, but it is not currently compatible with the DoDo-depth path. It defaults to `patch_size=512`, while DoDo-depth requires 128x128, and it calls `model(hs_patch, depth_patch, ...)` without passing `depth_metric` and `valid_mask`, which will raise `ValueError` in `snapshotdepth_hs.py`.

3. Low - Collaboration log policy needs to be enforced now.

   The user requested that `implementation-notes.md` permanently keep only the latest 8 top-level sections. This rule has been added to `AGENT.md`, `ai-collab/CLAUDE.md`, and `ai-collab/CODEX.md`. The next Claude round should add Section 15 and prune old sections so only Sections 8-15 remain.

## Assessment

Round 14 closes the checkpoint-selection loop. The best depth-sensitive model is now fixed and documented. Further same-architecture weight/lr/loss sweeps are not justified by the current evidence.

The next useful step is external validity: run the chosen checkpoint on complete hyperspectral/depth scene pairs rather than only 128x128 validation crops, using tiled 128x128 DoDo-depth inference and full-scene metrics.

