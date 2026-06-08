# Codex Review: DoDo-change Round 10 Depth Trade-off

## Summary

Round 10 resolves the previous numerical-stability and trainability blockers. The main 260 epoch run reached `validation/psnr_hs_masked = 31.11 dB` with `nonfinite_count = 0`; DOE and decoder gradients were finite. The remaining issue is now a real multi-objective optimization problem: stronger depth supervision improves metric depth MAE but damages HS PSNR.

## Findings

### High: `artifact_root` contract failed in Round 10 launch

Round 10 artifacts were written to default paths:

- `infer_results/DoDo-change/DoDo_depth_finite_joint_260_v1/version_0/`
- `infer_results/DoDo-change/DoDo_depth_finite_joint_depthw5_v1/version_0/`

Both `command.txt` files omit `--artifact_root`, and both `hparams.json` files record `"artifact_root": ""`. Earlier runs using the same parser did record non-empty `artifact_root`, so this is not proven to be an argparse implementation bug. The likely failure is launch-command construction under `conda run` / shell wrapping.

Next round must still harden the code path:

- Print raw `sys.argv` and parsed `args.artifact_root` at startup.
- Accept `DODO_ARTIFACT_ROOT` or `EXP_ROOT` as fallback when CLI `--artifact_root` is empty.
- Add a fail-fast guard for managed experiments so a missing artifact root stops immediately instead of falling back to `version_0`.
- Run a 1-batch preflight before any long run and verify `hparams.json`, `command.txt`, train log, quicklooks, and `[artifact] root=...` all point to the timestamped root.

### High: Depth objective is misaligned with the acceptance metric

Training currently optimizes depth primarily in IPS space, while the acceptance metric is `validation/mae_depth_m`. IPS error and metric-meter error are not equivalent over `[0.4, 2.0]m`; a small IPS bias can map to a larger meter error depending on depth. The main run learned a narrow high-IPS depth range `[0.541, 0.996]` and landed at `0.516m` MAE despite excellent HS PSNR.

Add an optional masked metric-depth loss, preferably SmoothL1/Huber in meters, normalized by the depth range so its scale is predictable:

```text
loss_depth = ips_l1_weight * L1(depth_ips) + metric_depth_loss_weight * SmoothL1(depth_m / depth_range)
```

Keep it opt-in and default off to avoid changing legacy behavior.

### Medium: `depth_loss_weight=5` proves depth learnability but overpowers HS

The follow-up improved `validation/mae_depth_m` from `0.516m` to `0.326m` but reduced PSNR from `31.11 dB` to `24.53 dB`. Gradients also rose sharply: depth head `0.96 -> 6.68`, DOE `0.60 -> 6.10`, backbone `5.98 -> 54.27`.

This is not an architecture failure. It is a loss-balance conflict through the shared DOE/backbone. The next design should avoid a static high depth weight from epoch 0. Recommended options:

- `depth_loss_weight` sweep at smaller values: `2` and `3` before trying another `5`.
- A curriculum schedule: start with `depth_w=1`, then ramp to `3` or `5` after HS has reached useful structure.
- Metric-depth auxiliary loss with a modest weight instead of simply multiplying IPS loss.

### Medium: Checkpointing is PSNR-only

Both Round 10 checkpoints are selected by `validation/psnr_hs_masked`. That is insufficient for Pareto work because the best-depth epoch may not equal the best-PSNR epoch. Next round should save at least two checkpoint streams:

- PSNR-best: monitor `validation/psnr_hs_masked`, mode `max`.
- Depth-best: monitor `validation/mae_depth_m`, mode `min`.

Report both metrics for both checkpoints.

### Medium: Main-run final train loss in `metrics.json` is unreliable

Main run `metrics.json` records all `train_loss/*` as `0.0`, while Section 10 reports final train loss around `0.118` from TensorBoard/log-derived evidence. This was already a concern in earlier reviews and is still present. Fix metric persistence before drawing conclusions from final train losses.

## Recommended Next Step

Do not change the DoDo optical formula or `optics/`. First fix experiment reproducibility and metric persistence. Then run short Pareto experiments using either `depth_w=2/3` or a new metric-depth loss. Only after a candidate reaches a useful trade-off should Claude spend time on another 260 epoch run.

