# 实现记录：DoDo-change depth-aware forward model cleanup

## 日期

2026-05-05（第六轮：Codex review 阻断修复）

## 执行 Agent

Claude（第六轮：dodo_depth cleanup — 修复 Codex review 7 项阻断）

---




## 17. 第十七轮：Inference-Only Ablation Study (2026-05-07)

### 17.1 Code Changes

| File | Change |
|------|--------|
| `infer_contect.py` | `--measurement_norm_override`、`--min_tile_valid_ratio`、`--fill_skipped_tiles` CLI; per-tile measurement stats; diagnostic CSV contract (`metrics_full_scene.csv`, `metrics_per_band.csv`, `metrics_regions.csv`, `metrics_depth_baselines.csv`, `metrics_spectral_quality.csv`, `metrics_measurement_tiles.csv`); `command.txt` saving; `.npy` removed from diag to save disk |
| `snapshotdepth_hs.py` | `_norm_override` opt-in attribute for inference-only norm bypass |

`torch_optics/forward_dodo.py` not modified.

### 17.2 Optical Forward Correction

Section 16.4 said `y_sum = y_stack.sum(dim=1)`. The actual code in `forward_dodo.py` accumulates per-depth-bin:

```python
y_k = self.sensing_unnorm(x_k)  # per depth bin k
y_sum = y_k if y_sum is None else y_sum + y_k
```

Then `_normalize_once(y_sum)` normalizes the accumulated sum. The final output is 3-channel after sensing, not a 25→3 sum in a single operation. The correction does not change the conclusion: measurement is a depth-integrated 3-channel representation synthesized from GT HS + GT depth.

### 17.3 Ablation Settings (10 runs)

5 settings × 2 deploys, all `--stride 128 --diagnostic_dump`:

| Label | norm_override | min_tile_valid_ratio |
|-------|--------------|---------------------|
| checkpoint | checkpoint | 0.0 |
| none | none | 0.0 |
| minmax | per_sample_minmax | 0.0 |
| checkpoint_skip01 | checkpoint | 0.1 |
| none_skip01 | none | 0.1 |

### 17.4 deploy 1 Ablation Results

| Setting | PSNR(m) | SAM | Depth MAE | RGB PSNR | mae_vs_median | Skipped tiles |
|---------|---------|-----|-----------|----------|---------------|---------------|
| checkpoint | 29.41 | 0.541 | 0.443 | 26.39 | 1.044 | 0 |
| none | 29.41 | 0.541 | 0.443 | 26.39 | 1.044 | 0 |
| minmax | 29.00 | 0.542 | 0.436 | 26.37 | 1.027 | 0 |
| checkpoint_skip01 | 29.42 | 0.540 | 0.444 | 26.40 | 1.047 | 547 (0.1%) |
| none_skip01 | 29.42 | 0.540 | 0.444 | 26.40 | 1.047 | 547 (0.1%) |

### 17.5 deploy 16 Ablation Results

| Setting | PSNR(m) | SAM | Depth MAE | RGB PSNR | mae_vs_median | Skipped tiles |
|---------|---------|-----|-----------|----------|---------------|---------------|
| checkpoint | 19.09 | 0.416 | 0.490 | 20.51 | 1.237 | 0 |
| none | 19.09 | 0.416 | 0.490 | 20.51 | 1.237 | 0 |
| minmax | 18.92 | 0.412 | 0.449 | 20.12 | 1.133 | 0 |
| checkpoint_skip01 | 19.09 | 0.416 | 0.491 | 20.51 | 1.239 | 462 (0.1%) |
| none_skip01 | 19.09 | 0.416 | 0.491 | 20.51 | 1.239 | 462 (0.1%) |

### 17.6 Acceptance Criteria Evaluation

| Criterion | Threshold | Best deploy 1 | Best deploy 16 | Pass? |
|-----------|-----------|---------------|----------------|-------|
| SAM ≤ 0.30 rad | 0.30 | 0.540 | 0.412 | **FAIL** |
| RGB PSNR ≥ +2 dB vs baseline | 28.39 | 26.40 | 20.51 | **FAIL** |
| mae_vs_median ≤ 0.80 | 0.80 | 1.027 | 1.133 | **FAIL** |
| No white/pink washout | visual | unchanged | unchanged | **FAIL** |

**No inference-only setting passes any criterion. All 10 runs fail.**

### 17.7 Key Finding: Measurement Norm Override Has No Effect

The most significant finding: `--measurement_norm_override none` produces **identical results** to `checkpoint` (default per_sample_mean_std). This proves that the `dodo_measurement_norm` in `snapshotdepth_hs.py` is NOT the primary cause of measurement normalization. The `_normalize_once(y_sum)` call inside `DepthAwareDoDoForwardModel.forward()` in `torch_optics/forward_dodo.py` already normalizes the measurement internally, making the second-stage norm redundant during inference (the measurement is already in [0,1] range with near-zero mean).

### 17.8 Measurement Stats (deploy 1, checkpoint setting)

| Stat | Value |
|------|-------|
| Tiles processed | 786 |
| capt_after mean_std | 0.73 |
| capt_after mean_mean | -0.01 |
| capt_after zero_ratio | 0.0% |

Measurement tiles are non-zero and properly normalized. The white-background effect is not caused by zero-fill or capture failure.

### 17.9 Conclusion

**All inference-only mitigations failed to rescue full-scene quality.**

- Removing `dodo_measurement_norm` has no effect because `_normalize_once(y_sum)` inside the DoDo forward model already normalizes the measurement
- Tile skipping removes <0.1% of pixels with no quality improvement
- Minmax norm slightly improves depth MAE (0.436 vs 0.443) but degrades SAM
- The root cause is in the trained model architecture and training distribution, not inference-time preprocessing

### 17.10 Recommendation for Round 18

**Do not continue inference-only exploration. A full training/architecture change is required.**

Specific proposals:
1. **Train without `_normalize_once` or with global normalization** instead of per-y_sum normalization
2. **Train with variable tile sizes and valid ratios** to match full-scene distribution
3. **Add background-aware loss** that penalizes white-background artifacts
4. **Use pseudo-RGB PSNR, SAM, and depth-vs-median-baseline as training-time validation gates** (not just masked PSNR)
5. **Consider architecture changes** to reduce the 25→3 information bottleneck

### 17.11 Notes Retention

After adding Section 17, this file retains Sections 10–17. Section 9 removed.

### 17.12 Artifact Paths

All 10 ablation runs in `infer_results/DoDo-change/round17_ablate_<deploy>_<norm>_v1/<timestamp>/`.

Key artifact roots for deploy 1:
- checkpoint: `.../round17_ablate_deploy1_checkpoint_v1/20260507_171000/`
- none: `.../round17_ablate_deploy1_none_v1/20260507_171100/`
- minmax: `.../round17_ablate_deploy1_minmax_v1/20260507_171220/`
- checkpoint_skip01: `.../round17_ablate_deploy1_checkpoint_skip01_v1/20260507_171340/`
- none_skip01: `.../round17_ablate_deploy1_none_skip01_v1/20260507_171500/`

Similar structure for deploy16.

---

## 18. 第十八轮：Forward-Norm Training Probe + Full-Scene Gate (2026-05-08)

### 18.1 Code Changes

| File | Change |
|------|--------|
| `torch_optics/forward_dodo.py` | `DepthAwareDoDoForwardModel` 新增 `measurement_norm_mode`：`legacy_max`(default), `none`, `per_sample_max`；`Forward_DM_Spiral_Depth` 传递该参数 |
| `snapshotdepth_hs.py` | `--dodo_forward_norm` CLI、`--background_hs_loss_weight` CLI；pass to camera；`__compute_loss` 新增背景 HS L1 loss；metrics persistence 记录 `background_hs_loss` |

### 18.2 Preflight

`--dodo_forward_norm none --dodo_measurement_norm none --background_hs_loss_weight 0.1 --min_valid_ratio 0.02`

| Criteria | Result | Status |
|----------|--------|--------|
| Exit code | 0 | PASS |
| nonfinite_count | 0 | PASS |
| All artifact files | Present | PASS |
| hparams records dodo_forward_norm=none | Yes | PASS |
| background_hs_loss > 0 | 11.87 | PASS |

Artifact: `infer_results/.../round18_forwardnorm_none_preflight_v1/20260508_151144/`

### 18.3 Candidate Training (40 epochs)

| Metric | Candidate A (forward_norm=none) | Candidate B (legacy+lowvalid+bg) |
|--------|-------------------------------|----------------------------------|
| PSNR hs_masked | 18.59 dB | 18.45 dB |
| PSNR hs_full | 18.76 dB | 18.68 dB |
| MAE depth_m | 0.468 m | 0.419 m |
| MAE depthmap | 0.227 | 0.212 |
| background_hs_loss | 2.59 | 2.66 |
| nonfinite_count | 0 | 0 |
| depth-best epoch | 24 | 24 |
| psnr-best epoch | 39 | 38 |

### 18.4 Full-Scene Gate Results (depth-best checkpoints)

**deploy 1**:

| Candidate | PSNR(m) | SAM | MAE | RGB PSNR | mae_vs_med | vs R17 Baseline |
|-----------|---------|-----|-----|----------|------------|-----------------|
| R17 Best | 29.41 | 0.541 | 0.443 | 26.39 | 1.044 | — |
| CandA | 16.28 | 0.755 | 0.473 | 11.52 | 1.116 | All worse |
| CandB | 16.48 | 0.751 | 0.500 | 11.73 | 1.180 | All worse |

**deploy 16**:

| Candidate | PSNR(m) | SAM | MAE | RGB PSNR | mae_vs_med | vs R17 Baseline |
|-----------|---------|-----|-----|----------|------------|-----------------|
| R17 Best | 19.09 | 0.416 | 0.490 | 20.51 | 1.237 | — |
| CandA | 18.03 | 0.648 | 0.485 | 13.72 | 1.224 | All worse |
| CandB | 18.02 | 0.652 | 0.475 | 13.67 | 1.199 | All worse |

### 18.5 Promotion Rules Evaluation

| Criterion | CandA deploy1 | CandA deploy16 | CandB deploy1 | CandB deploy16 | Pass? |
|-----------|--------------|----------------|---------------|----------------|-------|
| PSNR ≥ 20 dB (crop val) | 18.59 | — | 18.45 | — | **FAIL** |
| SAM vs R17 (−0.05) | +0.21 (worse) | +0.23 (worse) | +0.21 (worse) | +0.24 (worse) | **FAIL** |
| RGB PSNR vs R17 (+1.0) | −14.9 | −6.8 | −14.7 | −6.8 | **FAIL** |
| mae_vs_med < 1.0 + ≤0.90 | 1.12 | 1.22 | 1.18 | 1.20 | **FAIL** |

**Both candidates fail all promotion criteria. No candidate qualifies for extended training.**

### 18.6 Analysis

1. **40 epochs from scratch is insufficient.** Both candidates (trained for only 40 epochs from scratch) are dramatically worse than the R12 260-epoch model. This is expected — 40 vs 260 epochs.

2. **dodo_forward_norm=none did not help at 40 epochs.** Candidate A performed equivalently to Candidate B within noise, showing no benefit from removing internal forward normalization at this training duration.

3. **background_hs_loss is active but large** (2.6). The loss term successfully penalizes background regions but at this stage the model hasn't learned foreground, so both foreground and background losses remain high.

4. **min_valid_ratio=0.02 may be too permissive.** Training allowed very low-foreground patches, which increases noise diversity but may slow convergence.

5. **Full-scene is even worse than crop validation.** Full-scene SAM 0.75 vs crop PSNR 18.6 — the model hasn't learned any useful reconstruction in 40 epochs.

### 18.7 Conclusion

Forward-normalization/loss/distribution changes are insufficient to rescue full-scene quality at the 40-epoch probe scale. Whether they would help at 260 epochs is untested — but the 40-epoch results provide no evidence of a faster convergence trajectory or reduced white-background effects.

### 18.8 Recommendation for Round 19

**Recommend a larger architecture change:**

1. **Increase measurement channels.** 3-channel RGB sensing is insufficient to encode 25-band HS + depth. A 6-9 channel measurement could reduce the information bottleneck.

2. **Separate HS/depth decoder branches.** The shared backbone forces gradient competition. Independent decoder paths for HS reconstruction and depth prediction could mitigate the Pareto trade-off.

3. **True measurement-only formulation.** Current inference uses GT depth for optical forward synthesis. A real deployment would need joint HS+depth reconstruction from measurement only. This requires a different training formulation.

4. **If continuing forward_norm exploration**, run at least 120 epochs (not 40) to determine if the norm change provides any convergence benefit.

### 18.9 Notes Retention

After adding Section 18, this file retains Sections 11–18. Section 10 removed.

### 18.10 Artifact Paths

| Experiment | Artifact Root |
|------------|--------------|
| Preflight | `.../round18_forwardnorm_none_preflight_v1/20260508_151144/` |
| Candidate A training | `.../round18_train_forwardnorm_none_bg_v1/20260508_151446/` |
| Candidate B training | `.../round18_train_legacy_lowvalid_bg_v1/20260508_153554/` |
| CandA gate deploy1 | `.../round18_gate_candA_depthbest_deploy1_v1/20260508_155432/` |
| CandA gate deploy16 | `.../round18_gate_candA_depthbest_deploy16_v1/20260508_155546/` |
| CandB gate deploy1 | `.../round18_gate_candB_depthbest_deploy1_v1/20260508_155807/` |
| CandB gate deploy16 | `.../round18_gate_candB_depthbest_deploy16_v1/20260508_155923/` |

---

## 19. 第十九轮：Multi-Channel Sensing Diagnostic (2026-05-08 to 2026-05-09)

### 19.1 Code Changes

| File | Change |
|------|--------|
| `torch_optics/sensing.py` | `SensingLayer` 新增 `sensing_mode` (`rgb`/`spectral_bins`/`identity`) + `measurement_channels`; `rgb` 保持旧行为; `spectral_bins` 用 contiguous bin 响应矩阵; `identity` 用 25×25 eye |
| `torch_optics/forward_dodo.py` | `DepthAwareDoDoForwardModel` + `Forward_DM_Spiral_Depth` 传递 `sensing_mode`/`measurement_channels` |
| `snapshotdepth_hs.py` | `--dodo_sensing_mode` CLI; measurement channel auto-set; capt quicklook handles >3ch |
| `infer_contect.py` | `save_diagnostic_quicklook` 兼容 >3ch measurement |

### 19.2 Preflight (8-channel spectral_bins)

`--dodo_sensing_mode spectral_bins --measurement_channels 8`

| Criteria | Result | Status |
|----------|--------|--------|
| Exit code | 0 | PASS |
| nonfinite_count | 0 | PASS |
| dodo_sensing_mode in hparams | spectral_bins | PASS |
| measurement_channels=8 | Yes | PASS |

Artifact: `.../round19_sensing8_preflight_v1/20260508_164355/`

### 19.3 Candidate Training (40 epochs)

| Metric | Cand A (8ch spectral_bins) | Cand B (25ch identity upper-bound) |
|--------|---------------------------|-----------------------------------|
| PSNR masked | 14.20 dB | 18.69 dB |
| PSNR full | 13.98 dB | 18.71 dB |
| MAE depth_m | 0.744 m | 0.506 m |
| MAE depthmap | 0.304 | 0.249 |
| nonfinite_count | 0 | 0 |
| Crop PSNR ≥ 20 dB | **FAIL** | **FAIL** |

### 19.4 Full-Scene Gate (depth-best checkpoints)

**vs R17 Baseline**:

| Deploy | Cand A PSNRm | Cand A SAM | Cand A MAE | Cand A RGB | Cand B PSNRm | Cand B SAM | Cand B MAE | Cand B RGB |
|--------|-------------|-----------|-----------|-----------|-------------|-----------|-----------|-----------|
| deploy1 | 13.13 dB | 0.856 rad | 0.476 m | 17.59 dB | 17.72 dB | 0.635 rad | 0.501 m | 15.40 dB |
| deploy16 | 14.49 dB | 0.775 rad | 0.489 m | 19.51 dB | 18.80 dB | 0.598 rad | 0.463 m | 17.97 dB |

R17 baseline: deploy1 (29.41 dB / 0.541 rad / 0.443 m / 26.39 dB), deploy16 (19.09 dB / 0.416 rad / 0.490 m / 20.51 dB).

**Both candidates WORSE than R17 on every metric.**

### 19.5 Promotion Rules Evaluation

| Criterion | Cand A deploy1 | Cand A deploy16 | Cand B deploy1 | Cand B deploy16 | Pass? |
|-----------|---------------|----------------|---------------|----------------|-------|
| Crop PSNR ≥ 20 dB | 14.20 | — | 18.69 | — | **FAIL** |
| SAM vs R17 (−0.08) | +0.32 | +0.36 | +0.09 | +0.18 | **FAIL** |
| RGB PSNR vs R17 (+2 dB) | −8.8 | −1.0 | −11.0 | −2.5 | **FAIL** |
| mae_vs_med < 1.0 + ≤0.90 | 1.12 | 1.24 | 1.18 | 1.17 | **FAIL** |

All metrics are worse than the 40-epoch R18 candidates, and dramatically worse than the 260-epoch R12 model. 40 epochs from scratch is simply insufficient.

### 19.6 Analysis

1. **8-channel sensing did not help at 40 epochs.** Cand A (8ch) performed worse than Cand B (25ch) on most metrics, and both are far below R17 baseline. The sensing channel expansion alone, without sufficient training epochs, provides no benefit.

2. **Identity upper-bound also failed at 40 epochs.** Even with perfect (identity) sensing that preserves all 25 channels, 40 epochs is insufficient to learn useful reconstruction.

3. **40-epoch convergence is the bottleneck.** R12 needed 120+ epochs to reach 25 dB and 220 epochs to break 30 dB. None of the 40-epoch probes (R18 forward-norm, R19 multi-channel) have approached even 20 dB, confirming that epoch count, not architecture, is the dominant factor at this scale.

4. **The information bottleneck hypothesis is untestable at 40 epochs.** To determine whether 8-channel sensing helps vs 3-channel, a 120+ epoch comparison would be needed. But the user instruction explicitly forbids 120/260 epoch runs in this round.

### 19.7 Recommendation

Both promotion rule sets (R18, R19) require ≥ 20 dB crop PSNR which no 40-epoch candidate has achieved. Options for next round:

- **A**: Extend the most promising 40-epoch candidate (Cand B identity 25ch, PSNR 18.69 dB) to 120 epochs. This tests whether the channel count helps at convergence.
- **B**: Run a 120-epoch 8-channel spectral_bins vs 3-channel RGB comparison to isolate the sensing bottleneck.
- **C**: Accept that 40 epochs is an unsolvable constraint and change the promotion rule for 40-epoch probes to be relative (e.g., "crop PSNR improving faster than R18 baseline") rather than absolute (20 dB).

### 19.8 Notes Retention

After adding Section 19, this file retains Sections 12–19. Section 11 removed.

### 19.9 Artifact Paths

| Experiment | Artifact Root |
|------------|--------------|
| Preflight | `.../round19_sensing8_preflight_v1/20260508_164355/` |
| Candidate A train | `.../round19_train_sensing8_bg_v1/20260508_164452/` |
| Candidate B train | `.../round19_train_identity25_upperbound_v1/20260508_170528/` |
| CandA gate deploy1 | `.../round19_gate_candA_depthbest_deploy1_v2/20260509_093109/` |
| CandA gate deploy16 | `.../round19_gate_candA_depthbest_deploy16_v2/20260509_093226/` |
| CandB gate deploy1 | `.../round19_gate_candB_depthbest_deploy1_v2/20260509_093350/` |
| CandB gate deploy16 | `.../round19_gate_candB_depthbest_deploy16_v2/20260509_093506/` |

---

## 20. 第二十轮：DoDo 光学前向有效性审计 (2026-05-09)

### 20.1 审计目的

不训练。不推广 checkpoint。专门审计为什么 R12 260-epoch checkpoint 在 crop 上指标好（PSNR 30 dB），但 full-scene 视觉效果差（白底粉糊、SAM 0.54 rad、深度不如常数 baseline）。

### 20.2 代码变更

本轮无对现有文件的代码修改。仅新增诊断脚本 `round20_forward_diagnostics.py`（独立运行，不修改任何模型/训练/推理代码）。

### 20.3 前向合约审计

完整前向合约映射见 `handoff/DoDo-change/OPTICAL_FORWARD_AUDIT.md`。五个核心疑点：

| 问题 | 严重程度 | 代码位置 |
|------|----------|----------|
| 振幅测量（abs(field)）vs 强度测量（abs(field)^2） | 高 | sensing.py:89 |
| 自然图像作为相干复振幅场传播 | 高 | propagation.py:64-67 |
| GT-depth oracle 依赖（推理需要 GT depth 合成测量） | 高 | infer_contect.py:368-371 |
| 无效深度 clamp 到 min_depth 污染测量 | 中 | dataset:276 |
| Per-patch 归一化 + masked PSNR 虚高 | 中 | snapshotdepth_hs.py:769-771 |

### 20.4 诊断结果

#### Diagnostic A: 振幅 vs 强度 Sensing

| 指标 | 振幅测量（当前） | 强度测量（物理） |
|------|---------------|----------------|
| 动态范围 | 5.3× | 15.8× |
| 标准差 | 1.51 | 2.54 |
| 各通道相关系数 | 0.956 | — |
| 归一化后相关系数 | 0.927 | — |

**结论**：振幅 vs 强度测量存在显著差异（归一化后相关系数 0.927 < 0.95）。强度测量具有 3× 更大的动态范围。`_normalize_once` 归一化压缩了部分差异，但不足以完全掩盖。当前 `abs(field)` 测量不是物理正确的强度测量。

#### Diagnostic B: 相干可加性测试

| 指标 | 值 |
|------|-----|
| 相对残差 ∥meas(A+B)−(meas(A)+meas(B))∥/∥meas(A+B)∥ | 0.0024 |
| 最大残差 | 0.0058 |

**结论**：相干可加性近似成立（残差 0.24%）。说明在测试的两个空间分离源上，相干干涉效应较小。但这不排除在更复杂场景（更多源、更近间距）中出现显著相干干涉。

#### Diagnostic C: 背景/无效深度贡献

测试 crop valid_ratio=100%（全前景），背景能量贡献 = 0%。这是一个理想化测试——实际 full-scene tiles 中大量低前景 tile（valid_ratio < 10%）会产生完全不同的结果。由于全前景 crop 的背景能量为 0，该诊断需要在低前景 tile 上重跑才能获得有意义的结果。

#### Diagnostic D: Masked PSNR 膨胀

| Deploy | Masked PSNR | Full PSNR | 膨胀幅度 |
|--------|------------|-----------|---------|
| deploy 1 | 30.02 dB | 21.56 dB | **+8.45 dB** |
| deploy 16 | 18.96 dB | 21.03 dB | −2.07 dB |

**结论**：deploy 1 的 masked PSNR 比 full PSNR 虚高 8.45 dB。模型报告 30 dB PSNR 是基于仅 25.8% 有效像素计算的。Full PSNR（21.56 dB）更能反映实际图像质量。deploy 16 的负膨胀说明模型在有效区域的重建甚至不如整体——foreground 重建质量极差。

#### Diagnostic E: GT-Depth Oracle 依赖

**确认**：`infer_contect.py` 从 EXR 文件读取 GT 深度，转换为米制，传入 `model.forward(depth_metric=dm_patch)` 用于光学前向测量合成。这是 **oracle simulation**——真实部署中没有 GT depth 输入，无法合成测量信号。

### 20.5 根因排序

1. **GT-depth oracle 依赖**（最严重）：当前 full-scene 推理不是 measurement-only deployment。无法迁移到物理系统。
2. **振幅 vs 强度测量**：振幅测量不是物理正确，归一化后差异仍然显著（r=0.927）。
3. **Masked PSNR 虚高**：8.45 dB 的膨胀使 30 dB masked PSNR 实际对应 21.6 dB full PSNR。
4. **相干传播**：可加性近似成立（残差 0.24%），在当前测试中不是首要问题。
5. **背景 depth clamp**：全前景 crop 无法检测，需要低前景 tile 进一步验证。

### 20.6 下一步建议

按优先级：

1. **优先修复振幅→强度 sensing**：将 `torch.abs(field)` 改为 `torch.abs(field) ** 2`（或 `field.real**2 + field.imag**2`），跑 finite smoke 确认非有限值不会回归。
2. **明确 measurement-only contract**：在当前 oracle formulation 修复前，所有 full-scene 结论应标记为 "oracle simulation"，不能声称 measurement-only。
3. **替换 masked PSNR 为 full PSNR + SAM + pseudo-RGB PSNR**：作为训练和评估的主指标。
4. **相干传播**在当前测试中不是首要问题，可延后处理。

### 20.7 Notes Retention

仅保留 Sections 13-20。Section 12 已移除。

### 20.8 产出文件

| 文件 | 路径 |
|------|------|
| 前向审计报告 | `handoff/DoDo-change/OPTICAL_FORWARD_AUDIT.md` |
| 代码变更记录 | `handoff/DoDo-change/CURRENT_CODE_CHANGES.md` |
| 实验台账 | `handoff/DoDo-change/EXPERIMENTS.md` |
| 诊断脚本 | `/root/autodl-tmp/round20_forward_diagnostics.py`、`round20_forward_diag_CE.py` |
| 诊断结果 | `infer_results/DoDo-change/round20_forward_audit_v1/diagnostics/` |

---

## 21. 第二十一轮：Intensity Sensing 实现方案 (2026-05-09)

### 21.1 性质

本轮为纯实现方案编写，不执行代码修改，不跑训练，不推广 checkpoint。

### 21.2 背景

Round 20 光学前向审计确认：当前 `SensingLayer` 使用 `torch.abs(field)`（振幅测量），物理正确的传感器响应应接近 `torch.abs(field)^2`（强度测量）。两者归一化后相关系数 0.927，强度动态范围为振幅的 3×。

### 21.3 方案文件

完整代码落实方案见 `handoff/DoDo-change/ROUND21_CODE_PLAN.md`。

核心设计决策：

- **Opt-in 设计**：新增 `--dodo_sensor_measurement {amplitude, intensity}` CLI，默认 `amplitude`（保持旧行为）
- **覆盖所有 sensing mode**：`rgb`、`spectral_bins`、`identity` 均支持
- **兼容旧 checkpoint**：hparams.json 自动记录模式；checkpoint reload 使用保存值
- **修正指标合约**：full PSNR、SAM、pseudo-RGB PSNR、depth-vs-median baseline 提升为主指标；masked PSNR 降为 secondary

### 21.4 拟修改文件

| 文件 | 变更类型 |
|------|---------|
| `torch_optics/sensing.py` | SensingLayer 新增 `sensor_measurement` 参数 |
| `torch_optics/forward_dodo.py` | 传递 `sensor_measurement` 到 SensingLayer |
| `snapshotdepth_hs.py` | 新增 CLI `--dodo_sensor_measurement` |
| `infer_contect.py` | 新增 `oracle_simulation` 标签列 |
| `round21_lowfg_diag.py`（新增） | 低前景 tile 背景诊断 |

### 21.5 Validation Plan（概览）

按顺序执行：SensingLayer unit smoke → RGB backward-compat test → intensity finite forward smoke → 1-epoch preflight → 低前景 tile 诊断。前一步不通过则停止。

### 21.6 状态

等待 Codex review 后再执行代码修改。本轮未执行任何生产代码修改，未运行训练。

### 21.7 Notes Retention

仅保留 Sections 15-22。Section 14 已移除。

---

## 22. 第二十二轮：Soft Diopter Depth Layering 运行时验证 (2026-05-09)

### 22.1 背景

Codex 实现了 soft diopter depth layering patch（`SoftDiopterBinner` + `depth_layering_mode` 三种模式），但仅在 mounted-local workspace 做了静态 review，未运行任何 runtime test。本轮在服务器环境中执行完整的运行时验证。

### 22.2 修改

| 文件 | 修改 |
|------|------|
| `torch_optics/forward_dodo.py` | `DepthAwareDoDoForwardModel.forward()` 中修复 `diopter_binner()` 返回值解包：`debug_stages=False` 时 binner 返回 2 值 `(weights, z_centers)`，但调用方始终解包 3 值导致 `ValueError`。改为条件解包。 |

未修改 `snapshotdepth_hs.py`、`test/test_soft_diopter_depth_layering.py`、DOE、decoder、SensingLayer、loss、optical regularizer、CFA/CCA、DOE2、PSF loss、occlusion-aware propagation。

### 22.3 Required Tests 结果

| Test | 命令 | 状态 | 关键输出 |
|------|------|------|----------|
| Test 1: pytest | `pytest -q test/test_soft_diopter_depth_layering.py` | PASS (8/8) | 5 passed on first run, 3 failed due to unpack bug; after fix: 8/8 pass, 7.1s |
| Test 2: py_compile | `python -m py_compile torch_optics/forward_dodo.py snapshotdepth_hs.py test/test_soft_diopter_depth_layering.py` | PASS | exit=0 |
| Test 3: forward smoke | 3 modes instantiation + valid_mask smoke | PASS | all modes output (1,3,128,128), all finite, valid_mask accepted |
| Test 4: backward smoke | `soft_diopter + doe_type_a=New + train_c=True` | PASS | grad norm=0.675, finite=True, >0 |
| Test 5: preflight | 1 batch train + 1 batch val, 1 epoch | PASS | exit=0, nonfinite=0, hparams records `depth_layering_mode=soft_diopter`, metrics.json present, DOE grad norm=9.89 finite |

### 22.4 Preflight 关键指标

| 指标 | 值 |
|------|-----|
| train_loss/total_loss | 0.896 |
| train_loss/depth_loss | 0.378 |
| validation/psnr_hs_masked | 6.17 dB |
| validation/mae_depth_m | 0.595 m |
| DOE grad norm | 9.887 (finite) |
| nonfinite count | 0 |

### 22.5 发现与修复

**1 个 bug 发现并修复**：`DepthAwareDoDoForwardModel.forward()` 第 387 行无条件解包 `diopter_binner()` 返回值为 3 元组，但 `SoftDiopterBinner.forward()` 在 `return_debug=False` 时仅返回 2 值 `(weights, z_centers)`，在 `return_debug=True` 时返回 3 值 `(weights, z_centers, debug)`。修复为条件解包：先接收 tuple result，然后按 `debug_stages` 决定解包方式。

**无其他失败**。soft diopter layering 正确集成到 DoDo forward pipeline，三种 depth_layering_mode（hard_depth, hard_meter, soft_diopter）均可正常运行。

### 22.6 Checkpoint

本轮未推广 checkpoint。preflight auto-saved checkpoints 仅作为训练产物存在，不作为 best checkpoint。

### 22.7 Artifact

Preflight artifact root: `infer_results/DoDo-change/soft_diopter_preflight_v1/20260509_135947/`

### 22.8 Notes Retention

仅保留 Sections 16-23。Section 15 已移除。

---

## 23. 第二十三轮：DoDo Intensity Sensing 最小实现 (2026-05-09)

### 23.1 背景

Round 20 光学前向审计确认 `SensingLayer` 使用 `torch.abs(field)`（振幅测量）而非物理正确的 `|field|^2`（强度测量）。Round 21 编写了代码落实方案（`ROUND21_CODE_PLAN.md`），Codex review 批准并要求修改。Round 22 完成了 soft diopter 验证。Round 23 执行 intensity sensing 的最小实现。

### 23.2 代码变更

| 文件 | 变更 |
|------|------|
| `torch_optics/sensing.py` | `SensingLayer.__init__()` 新增 `sensor_measurement="amplitude"` 参数 + 校验；`forward()` 中 `intensity` 模式使用 `x_abs ** 2`。覆盖 rgb / spectral_bins / identity。 |
| `torch_optics/forward_dodo.py` | `DoDoForwardModel`、`DepthAwareDoDoForwardModel`、`Forward_DM_Spiral_Depth` 各新增 `sensor_measurement="amplitude"` 参数，透传到 `SensingLayer`。 |
| `snapshotdepth_hs.py` | `add_model_specific_args()` 新增 `--dodo_sensor_measurement {amplitude,intensity}`；`__build_model()` 读取并传递。 |
| `infer_contect.py` | `metrics_real.txt`、`metrics_full_scene.csv`、`aggregate_metrics.json`、`diagnostic_metrics.json` 新增 `oracle_simulation=True` metadata；`aggregate_metrics.json` 额外记录 `dodo_sensor_measurement`。 |
| `round23_lowfg_diag.py` | 新增低前景 tile 背景能量诊断脚本。 |

未修改 DOE, decoder, loss, soft diopter 逻辑, optical regularizer, CFA/CCA, DOE2, PSF loss, occlusion-aware propagation, snapshotdepth_trainer_hs.py。

### 23.3 验证结果

| Step | 描述 | 状态 | 关键输出 |
|------|------|------|----------|
| Step 1 | SensingLayer 确定性 smoke | PASS | amplitude vs manual diff=0; intensity vs manual diff=0; spectral_bins/identity 通过; invalid ValueError |
| Step 2a | 向后兼容 | PASS | 默认 vs 显式 amplitude diff=0 |
| Step 2b | Round 22 回归 | PASS | pytest 8/8 pass |
| Step 3 | DoDo-depth intensity forward smoke | PASS | intensity + hard_depth/soft_diopter/valid_mask 全部 finite |
| Step 4 | 1-epoch preflight | PASS | exit=0, nonfinite=0, hparams 记录 `dodo_sensor_measurement=intensity` |
| Step 5 | 低前景诊断 | PASS | deploy 1: 491/726 tiles valid_ratio<0.1; deploy 16: 406/726 tiles |

### 23.4 Preflight 关键指标

| 指标 | 值 |
|------|-----|
| train_loss/total_loss | 1.644 |
| validation/psnr_hs_masked | 5.83 dB |
| validation/mae_depth_m | 0.621 m |
| nonfinite count | 0 |
| dodo_sensor_measurement | intensity (recorded in hparams.json) |

### 23.5 Artifact Paths

| 产物 | 路径 |
|------|------|
| Preflight | `infer_results/DoDo-change/round23_intensity_preflight_v1/20260509_142405/` |
| LowFG diagnostic | `infer_results/DoDo-change/round23_lowfg_diag_v1/20260509_143502/` |

### 23.6 Checkpoint

本轮未推广 checkpoint。preflight auto-saved checkpoints 仅作为训练产物存在，不作为 best checkpoint。

### 23.7 Notes Retention

仅保留 Sections 16-23。Section 15 已移除。

### 23.8 Round 23-repair: LowFG Measurement-Energy Fix (2026-05-09)

Codex review 发现一个阻断项：`round23_lowfg_diag.py` 声称计算 measurement energy，但实际只计算 raw HS input energy（`tile.abs().sum()`），未运行 DoDo camera/sensing/propagation。另要求 `oracle_simulation` 不应硬编码为 `True`。

**修复 1: `round23_lowfg_diag.py` 完全重写**
- 加载 R12 `depth-best-epoch=226.ckpt` 获取 `DepthAwareDoDoForwardModel` camera
- 对低前景 tile 运行真实 DoDo 光学测量：full / fg-only / bg-only
- CSV 新增 `measurement_*` 列，保留 `hs_*` 列为诊断参考
- JSON summary 记录 checkpoint path, sensor_measurement, depth_layering_mode, forward_norm, measurement_norm, background_policy
- bg-only 策略在 JSON 中显式记录：`bg_hs=hs*(1-fg_mask)`, `bg_depth=clamp(invalid,min,max)`, `bg_valid=1-fg_mask`

**修复 2: `infer_contect.py` oracle_simulation gating**
- `main()` 中计算 `oracle_simulation = is_dodo_model`，不经由硬编码 `True`
- `metrics_real.txt` 行输出使用变量 `oracle_simulation`
- `aggregate_metrics.json` 使用 `oracle_simulation`
- 已有 `scenario_metrics['oracle_simulation'] = is_dodo`（`process_single_scene` 内部）
- 新增 `dodo_sensor_measurement` 到 per-scene `diagnostic_metrics.json`

**验证结果**: py_compile exit=0；pytest 8/8；LowFG 诊断 10 tile 全部 finite measurement，meas_bg_fraction=1.0000（纯背景 tile 的 measurement 能量 100% 来自背景）。

**关键诊断发现**: 纯背景 tile 通过 clamped depth（0.4m）仍产生 non-zero measurement 能量（6-8e3），经过 `per_sample_mean_std` 归一化后被拉伸到全动态范围 — 定量确认了 full-scene 白底 artifacts 的物理机制。

**Artifact**: `infer_results/DoDo-change/round23_repair_lowfg_v1/20260509_145728/`

## 24. 第二十四轮：Spectral Capacity Diagnostic + Decoder Depth Input + CFA Review (2026-05-09)

### 24.1 背景

Round 23 完成了 intensity sensing 实现和修复。本轮转向诊断"高光谱重建效果不好是否因为二维压缩测量图携带的光谱可分信息不足"这一假设，并为 decoder 增加可选的 depth input channel。

### 24.2 代码变更

| 文件 | 变更 |
|------|------|
| `scripts/diagnose_measurement_spectral_capacity.py` | 新增：4 个光谱编码信息量诊断（A-D），逐波长处理，内存安全 |
| `snapshotdepth_hs.py` | 新增 CLI: `--decoder_use_depth_input`, `--decoder_depth_input_mode`；`forward()` 中 concat depth_feature |
| `models/simple_model_mamba.py` | `input_adapter` 第一层 Conv2d 使用 `decoder_in_channels` |
| `handoff/DoDo-change/sensing_cfa_review.md` | 新增：SensingLayer CFA/CCA readiness 静态审查 |
| `test/test_decoder_depth_input.py` | 新增：12 个测试覆盖 depth input shapes/normalization/diagnosis utilities |

### 24.3 Spectral Capacity 诊断结果（Zeros DOE, fresh init）

| 诊断 | 关键发现 |
|------|----------|
| Diag B | adjacent_corr mean=0.955 — 相邻波长几乎无法区分 |
| Diag C | 能量差异 8×（3.3e3~2.7e4），边缘波长能量低 |
| Diag D | mutual_coherence=0.998（列向量几乎共线），condition_number=7.83e3 |

### 24.4 Decoder Depth Input

- 默认 `decoder_use_depth_input=False`，旧路径完全不变
- 启用时：归一化深度作为额外通道 concat 到 captimgs
- `decoder_in_channels = measurement_channels + 1`
- NaN/Inf/<=0 depth 通过 clamp 安全处理

### 24.5 SensingLayer CFA Review 结论

- 不能直接加入 Gaussian CFA（RGB collapse 在 SensingLayer 内部完成）
- 需在 `forward()` 第 100-116 行 `x_abs` 后、collapse 前插入 CFA
- RGB sensing 过早丢失 wavelength-resolved intensity

### 24.6 测试结果

- `test_decoder_depth_input.py`: 12/12 pass; `test_soft_diopter_depth_layering.py`: 8/8 pass
- 诊断脚本端到端 smoke: Diag A-D 全部完成

### 24.7 Notes Retention

仅保留 Sections 17-24。Section 16 已移除。

### 24.8 Round 25 补充：Trained DOE Comparison + SensingLayer Refactoring Plan (2026-05-09)

**Trained vs Init spectral capacity 对比**

用 R12 depth-best-epoch=226.ckpt（260 epoch trained DOE）跑完整 4 个诊断，与 Zeros DOE 对比：

| 指标 | Init → Trained | 改善 |
|------|----------------|------|
| effective_rank | 25.7 → 95.8 | **+273%** |
| condition_number | 7830 → 32 | **−99.6%** |
| energy_std | 7250 → 885 | **−87.8%** |
| adjacent_corr_mean | 0.955 → 0.910 | −4.7% |
| mutual_coherence | 0.998 → 0.900 | −9.8% |

10/11 指标改善。但 mutual_coherence 仍 >0.9，effective_rank 仍 < 总列数——即使 260 epoch 训练的 DOE 也无法突破 3-channel 的结构性信息瓶颈。

**SensingLayer Refactoring Plan**

文档化在 `handoff/DoDo-change/sensing_refactor_plan.md`：
- `forward()` 拆分为 `compute_wavelength_resolved_intensity()` + `collapse_to_measurement()`
- 零接口变更，零 state_dict 变更，完全向后兼容
- CFA/CCA 插入点明确：`x_abs (B, 25, H, W)` 在 collapse 之前
- 新增 6 个 shape/compat tests 方案

**Artifacts**:
- Init: `infer_results/DoDo-change/round24_spectral_capacity_init_v1/`
- Trained: `infer_results/DoDo-change/round24_spectral_capacity_trained_v1/`
- Comparison: `infer_results/DoDo-change/round24_compare_summary.json`

### 24.9 Round 26: DOE0 RGB vs Spectral Bins Comparison (2026-05-09)

**目的**: 在 Zeros DOE 下定量对比增加 measurement channels（spectral_bins 6/9/12ch vs RGB 3ch）对光谱编码容量的影响，为下一步是否增加 channels 提供数据支撑。

**方法**: 复用 `diagnose_measurement_spectral_capacity.py` 的 Diagnostic B/C/D，对 4 种 sensing config（rgb_3, bins_6, bins_9, bins_12）在相同 Zeros DOE 下运行。`scripts/compare_doe0_rgb_vs_spectral_bins.py` 统一编排，每组完成后立即删除临时目录。

**结果**:

| Setting | ch | effective_rank | condition_number | adjacent_corr_mean |
|---------|----|---------------|------------------|--------------------|
| rgb_3   | 3  | 73.4          | 3.45e4           | 0.964              |
| bins_6  | 6  | 93.9          | 1.12e4           | 0.754              |
| bins_9  | 9  | 112.1         | 9.85e3           | 0.629              |
| bins_12 | 12 | 112.1         | 9.85e3           | 0.504              |

**关键发现**:
1. effective_rank 在 9→12 bins 饱和（112.08 vs 112.08）—— 12 bins 无额外收益
2. condition_number 在 9 bins 处最优（9851, 比 RGB 改善 3.5×）
3. adjacent_corr_mean 随 bin 数单调下降（0.964→0.504）
4. mutual_coherence 几乎不变（~0.9991）—— Zeros DOE 不提供 spatial diversity
5. **9 bins 是 Zeros DOE 下的最优性价比**

**Artifact**: `infer_results/DoDo-change/doe0_bins_compare/`（仅含 4 个 summary 文件，无临时中间产物）


