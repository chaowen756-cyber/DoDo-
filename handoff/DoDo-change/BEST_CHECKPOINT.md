# Best Checkpoint — DoDo-change

## Selected Model

```
infer_results/DoDo-change/DoDo_depth_finite_joint_metricdepth_260_v1/20260507_112631/checkpoints/depth-best-epoch=226.ckpt
```

## Standalone Eval Metrics

Validated with `--validate_only_ckpt` on full validation set:

| Metric | Value |
|--------|-------|
| PSNR hs_masked | 30.1345 dB |
| MAE depth_m | 0.1784 m |
| MAE depthmap (IPS) | 0.132 |
| nonfinite_count | 0 |

Eval artifact root: `infer_results/DoDo-change/DoDo_depth_ckpt_eval_depth_best_v1/20260507_134124/`

## Selection Rationale

This checkpoint sits at the depth-optimal end of the current architecture's HS-PSNR / depth-MAE Pareto frontier. It achieves the lowest depth MAE (0.178m) across all rounds — a 65% improvement over the depth_w=1 baseline (0.516m) — while maintaining PSNR within 1 dB of the all-time best (31.10 dB).

All fine-tune attempts confirmed that further PSNR improvement from this starting point necessarily degrades depth quality, and no fine-tune checkpoint met the dual-threshold promotion criteria (PSNR ≥ 30.4345 dB and MAE ≤ 0.22m).

## Alternative Checkpoints

| Checkpoint | PSNR | MAE depth_m | Use Case |
|------------|------|-------------|----------|
| R12 psnr-best-epoch=259.ckpt | 31.10 dB | 0.494 m | PSNR-max when depth quality is not required |
| FT-A psnr-best-epoch=036.ckpt | 31.02 dB | 0.284 m | PSNR-depth compromise (worse depth than selected, worse PSNR than psnr-best) |
| FT-A depth-best-epoch=000.ckpt | 30.16 dB | 0.181 m | Essentially identical to selected (same init) |
| FT-B depth-best-epoch=000.ckpt | 30.16 dB | 0.181 m | Essentially identical to selected (same init) |

## Recommended Usage

- **Use the selected depth-best checkpoint for all depth-sensitive DoDo-depth results.**
- Use the PSNR-best checkpoint (ep259) only if HS reconstruction quality is the sole concern and depth is irrelevant.
- No further fine-tuning from these checkpoints is recommended — all weight/lr/loss-weight variants have been exhausted and none escape the Pareto trade-off.

## Architecture Note

The shared DOE + backbone architecture creates an inherent PSNR-depth trade-off that cannot be resolved through loss-weight tuning alone. If simultaneous PSNR ≥ 31 dB and MAE ≤ 0.20 m is required, open a new architecture-level change (e.g., decoupled HS/depth backbones, separate DOE elements, multi-task gradient projection).

## Full-Scene Evaluation

**⚠️ WARNING (Round 16): Full-scene usability is NOT approved.**

Round 15 full-scene results are reclassified as blocking validity findings. Diagnostics show:
- Depth model barely beats constant median-depth baseline (7% MAE improvement)
- ROI depth MAE vs median ratio is 200× worse — depth is essentially random
- HS visual quality is poor (white/pink/blurred), contradicting high masked PSNR
- PSNR inflated by per-patch `per_sample_mean_std` normalization on background-heavy tiles
- SAM 0.54 rad confirms unacceptable spectral quality

The selected checkpoint remains valid for 128×128 filtered-crop evaluation only. Full-scene deployment requires architectural or preprocessing changes before this model can be considered usable.

Validated on complete hyperspectral/depth scene pairs (not 128x128 crops) using `--patch_size 128` tiled stitching.

### deploy 1 (scene01)

| Metric | Value |
|--------|-------|
| PSNR masked | 30.02 dB |
| PSNR full | 21.56 dB |
| Depth MAE | 0.453 m |
| Depth RMSE | 0.578 m |
| SAM masked | 0.540 rad |
| Valid ratio | 0.258 |

Artifact root: `infer_results/DoDo-change/DoDo_depth_fullscene_smoke_deploy1_v1/20260507_145027/`

### deploy 16 (scene16)

| Metric | Value |
|--------|-------|
| PSNR masked | 18.96 dB |
| PSNR full | 21.03 dB |
| Depth MAE | 0.489 m |
| Depth RMSE | 0.612 m |
| SAM masked | 0.417 rad |
| Valid ratio | 0.370 |

Artifact root: `infer_results/DoDo-change/DoDo_depth_fullscene_smoke_deploy16_v1/20260507_144850/`

### Full-scene vs 128x128 validation

| Metric | 128x128 val (crops) | Full-scene deploy 1 |
|--------|---------------------|---------------------|
| PSNR masked | 30.13 dB | 30.02 dB |
| Depth MAE | 0.178 m | 0.453 m |

- Masked PSNR of 30.02 dB is inflated by per-patch normalization; it does NOT indicate acceptable full-scene HS reconstruction (SAM 0.54 rad, white/pink visual).
- Depth MAE of 0.453m is WORSE than a constant median-depth baseline (0.424m). The model contributes no meaningful depth prediction in full-scene.
- deploy 16 PSNR of 18.96 dB is a genuine model failure on this scene.

**Round 17 update**: No inference-only mitigation (norm override, tile skip, minmax norm) rescues full-scene quality. All 10 ablation runs fail all acceptance criteria. Root cause is in the trained model, not inference-time preprocessing. Full-scene deployment remains NOT approved. Checkpoint retained only as 128×128 filtered-crop best.

**Round 18 update**: Forward-norm probes failed at 40 epochs. No candidate passed promotion.

**Round 19 update**: Multi-channel sensing probes failed at 40 epochs. No promotion.

**Round 20 update**: 光学前向审计确认五个疑点：(1) 振幅 vs 强度 sensing 有显著差异（r=0.927），(2) 相干可加性近似成立，(3) GT-depth oracle 依赖确认，(4) masked PSNR 虚高 8.45 dB，(5) 背景 depth clamp 在全前景 crop 上未触发。未进行训练或 checkpoint 推广。R12 depth-best-epoch=226.ckpt 仍为当前 crop-only best。Full-scene 未批准。

**Round 21 update**: 编写 intensity sensing 实现方案（`ROUND21_CODE_PLAN.md`），未执行代码修改。所选 checkpoint 未变更。

**Round 22 update (soft diopter validation)**: Soft diopter depth layering 运行时验证完成。未进行长训练，未推广 checkpoint。现有 best checkpoint（R12 depth-best-epoch=226.ckpt）保持不变。本任务仅验证 soft_diopter 模式可正确接入 DoDo forward pipeline，不涉及 checkpoint 评估或推广。

**Round 23 update (intensity sensing)**: Intensity sensing 最小实现完成（opt-in `sensor_measurement={amplitude,intensity}`）。未进行长训练，未推广 checkpoint。现有 best checkpoint（R12 depth-best-epoch=226.ckpt）保持为 crop-only selected checkpoint。Full-scene deployment 仍未批准。`amplitude` 默认保持所有旧行为和 checkpoint 兼容。

**Round 23-repair update**: LowFG diagnostic repair 完成。修复了 measurement-energy 计算（从 raw HS energy 改为真实 DoDo optical measurement）。确认纯背景 tile 产生 100% background measurement energy — 定量验证了 full-scene artifacts 的物理原因。未推广 checkpoint。Full-scene deployment 仍未批准。

**Round 24 update**: Spectral capacity 诊断（adjacent_corr≈0.95, mutual_coherence≈0.998）定量确认了 3-channel measurement 信息瓶颈。新增 opt-in decoder depth input（默认 false）。SensingLayer CFA review 完成。未推广 checkpoint。
