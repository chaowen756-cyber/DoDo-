# 实验台账：DoDo-change

> Claude 在每轮完成后追加一条记录。
> 本文件累加记录，不受 8-section 保留规则约束。

## 必填格式

```markdown
## EXP-YYYYMMDD-NN: 简短名称

- 日期:
- 轮次:
- 类型:
- 状态:
- Checkpoint / 权重路径:
- Artifact root:
- 推理结果路径:
- 关键指标:
- 相关代码/配置:

摘要:
一段话说明做了什么、为什么跑这个实验、有什么结论需要记住。
```

---

## EXP-20260508-01: round19_sensing8_preflight

- 日期: 2026-05-08
- 轮次: 19
- 类型: preflight
- 状态: 通过
- Checkpoint: 无（从头训练）
- Artifact root: `infer_results/DoDo-change/round19_sensing8_preflight_v1/20260508_164355/`
- 推理结果路径: 无
- 关键指标: exit=0, nonfinite=0, dodo_sensing_mode=spectral_bins, measurement_channels=8
- 相关代码: `torch_optics/sensing.py`, `torch_optics/forward_dodo.py`, `snapshotdepth_hs.py`

摘要:
8 通道 spectral_bins sensing preflight。验证 DepthAwareDoDoForwardModel + SpectralBins SensingLayer 正确输出 8 通道有限值测量。hparams 正确记录了 dodo_sensing_mode 和 measurement_channels。所有标准通过。

---

## EXP-20260508-02: round19_train_sensing8_bg (Candidate A)

- 日期: 2026-05-08
- 轮次: 19
- 类型: 40 epoch 训练候选
- 状态: 完成（promotion 失败）
- Checkpoint: `round19_train_sensing8_bg_v1/20260508_164452/checkpoints/depth-best-epoch=033.ckpt`
- Artifact root: `infer_results/DoDo-change/round19_train_sensing8_bg_v1/20260508_164452/`
- 推理结果路径: `round19_gate_candA_depthbest_deploy1_v2/20260509_093109/`, `round19_gate_candA_depthbest_deploy16_v2/20260509_093226/`
- 关键指标: crop PSNR=14.20 dB, depth MAE=0.744m, nonfinite=0; full-scene deploy1 SAM=0.856, deploy16 SAM=0.775
- 相关代码: `torch_optics/sensing.py`（spectral_bins 模式）

摘要:
Candidate A 使用 8 通道 spectral_bins sensing + background_hs_loss=0.1。从头训练 40 epochs，crop PSNR 未达到 20 dB 门槛（14.20 dB）。full-scene 所有指标均差于 R17 baseline。40 epochs 不足以让 8 通道架构收敛，尚不能否定 8 通道的潜在优势——需要 120+ epoch 对照才能判断。

---

## EXP-20260508-03: round19_train_identity25_upperbound (Candidate B)

- 日期: 2026-05-08
- 轮次: 19
- 类型: 40 epoch 训练候选（upper-bound 诊断）
- 状态: 完成（promotion 失败，仅作上限诊断）
- Checkpoint: `round19_train_identity25_upperbound_v1/20260508_170528/checkpoints/depth-best-epoch=032.ckpt`
- Artifact root: `infer_results/DoDo-change/round19_train_identity25_upperbound_v1/20260508_170528/`
- 推理结果路径: `round19_gate_candB_depthbest_deploy1_v2/20260509_093350/`, `round19_gate_candB_depthbest_deploy16_v2/20260509_093506/`
- 关键指标: crop PSNR=18.69 dB, depth MAE=0.506m, nonfinite=0; full-scene deploy1 SAM=0.635, deploy16 SAM=0.598
- 相关代码: `torch_optics/sensing.py`（identity 模式）

摘要:
Candidate B 使用 25 通道 identity sensing（直通，无信息损失），仅作上限诊断——不可作为物理模型。从头训练 40 epochs。crop PSNR (18.69 dB) 略好于 Cand A (14.20) 但仍未达 20 dB。是 R18/R19 所有候选中 40 epoch PSNR 最高的，说明通道数可能影响学习速度。identity checkpoint 不作为物理模型推广。

---

## EXP-20260509-01: round20_forward_audit_diagnostics

- 日期: 2026-05-09
- 轮次: 20
- 类型: 前向审计诊断
- 状态: 完成
- Checkpoint: `depth-best-epoch=226.ckpt`（仅用于诊断加载）
- Artifact root: `infer_results/DoDo-change/round20_forward_audit_v1/`
- 推理结果路径: 同 artifact root
- 关键指标: 见各 Diagnostic JSON
- 相关代码: `round20_forward_diagnostics.py`（新增诊断脚本）

摘要:
运行五个非训练光学前向诊断（Diagnostics A-E），审计 DoDo forward model 的核心物理假设：
- Diag A: 振幅 vs 强度 sensing 对比
- Diag B: 相干可加性测试（相干干涉 vs 非相干成像）
- Diag C: 背景/无效深度对测量能量的贡献
- Diag D: masked PSNR 膨胀量化
- Diag E: GT-depth oracle 依赖文档化
结论和建议见 OPTICAL_FORWARD_AUDIT.md 和 Section 20。

---

## EXP-20260509-02: soft_diopter_preflight_v1

- 日期: 2026-05-09
- 轮次: 22（soft diopter depth layering runtime validation）
- 类型: preflight（1 batch train + 1 batch val, 1 epoch）
- 状态: 通过
- Checkpoint: 无（preflight checkpoint 不推广）
- Artifact root: `infer_results/DoDo-change/soft_diopter_preflight_v1/20260509_135947/`
- 推理结果路径: 无
- 关键指标: exit=0, nonfinite=0, depth_layering_mode=soft_diopter (recorded in hparams.json), train_loss/total_loss=0.896, validation/psnr_hs_masked=6.17 dB, validation/mae_depth_m=0.595 m, doe1.zernike_coeffs.grad norm=9.89 (finite)
- 相关代码: `torch_optics/forward_dodo.py`（SoftDiopterBinner, unpack fix）, `snapshotdepth_hs.py`（depth_layering_mode CLI wiring）

摘要:
Soft diopter depth layering 运行时验证 preflight。修复了 `DepthAwareDoDoForwardModel.forward()` 中 `diopter_binner()` 返回值解包问题（debug_stages=False 时 binner 返回 2 值但调用方期望 3 值）后，所有 Required Tests 1-4 通过（pytest 8/8, py_compile, forward smoke 3 modes, backward grad smoke）。Required Test 5 1-epoch preflight 成功：soft_diopter 模式可完整运行 training + validation loop，DOE zernike 梯度非零且有限（9.89），无 nonfinite 测量失败。hparams.json 正确记录了 depth_layering_mode=soft_diopter。preflight checkpoint 未推广（非训练任务）。确认 soft diopter layering 正确接入 DoDo forward pipeline。

---

## EXP-20260509-03: round23_intensity_preflight_v1

- 日期: 2026-05-09
- 轮次: 23（intensity sensing minimal implementation）
- 类型: preflight（2 batch train + 2 batch val, 1 epoch, intensity mode）
- 状态: 通过
- Checkpoint: 无（preflight checkpoint 不推广）
- Artifact root: `infer_results/DoDo-change/round23_intensity_preflight_v1/20260509_142405/`
- 推理结果路径: 无
- 关键指标: exit=0, nonfinite_count=0, dodo_sensor_measurement=intensity (recorded in hparams.json), train_loss/total_loss=1.644, validation/psnr_hs_masked=5.83 dB, validation/mae_depth_m=0.621 m
- 相关代码: `torch_optics/sensing.py`（sensor_measurement 新增）, `torch_optics/forward_dodo.py`（透传）, `snapshotdepth_hs.py`（CLI wiring）

摘要:
Intensity sensing (`abs(field)^2`) 1-epoch preflight 验证。所有 5 个验证步骤通过：
- Step 1: SensingLayer 确定性 smoke — amplitude 与手动公式精确匹配（diff=0.00e+00），intensity 与手动公式精确匹配（diff=0.00e+00），intensity 不同于 amplitude（mean ratio 1.6×），spectral_bins 和 identity 模式均有限。
- Step 2: 向后兼容 — 默认构造与显式 amplitude 精确匹配（diff=0）；Round 22 回归测试 8/8 通过。
- Step 3: DoDo-depth forward smoke — intensity + hard_depth/soft_diopter/valid_mask 全部有限输出。
- Step 4: 1-epoch preflight — exit 0, 无 nonfinite, hparams 正确记录 sensor_measurement=intensity。
- Step 5: 低前景诊断 — deploy 1 发现 491 个 valid_ratio<0.1 tile, deploy 16 发现 406 个。
intensity 模式成功接入完整 train/val pipeline，向后兼容性确认。默认 amplitude 维持旧行为。

---

## EXP-20260509-04: round23_lowfg_diag_v1（原始版本 — HS input-energy 统计）

- 日期: 2026-05-09
- 轮次: 23
- 类型: 诊断（无训练）
- 状态: 通过但被 Codex review 判定为不满足 measurement-energy 诊断要求（仅计算 raw HS input energy）
- Artifact root: `infer_results/DoDo-change/round23_lowfg_diag_v1/20260509_143502/`
- 关键指标: deploy 1: 491/726 tiles valid_ratio<0.1; deploy 16: 406/726 tiles

摘要:
低前景 tile 原始 HS input-energy 统计。发现 56-68% 的 tile 有效像素 <10%，但仅计算 `hs.abs().sum()` 而非 DoDo optical measurement energy。已被 EXP-20260509-05 取代。

---

## EXP-20260509-05: round23_repair_lowfg_v1（修复版 — DoDo measurement-energy diagnostic）

- 日期: 2026-05-09
- 轮次: 23-repair（Codex review 修复）
- 类型: 诊断（无训练）
- 状态: 完成（Codex review 阻断项已修复）
- Checkpoint: R12 `depth-best-epoch=226.ckpt`（仅用于诊断加载，未推广）
- Artifact root: `infer_results/DoDo-change/round23_repair_lowfg_v1/20260509_145728/`
- 推理结果路径: 同 artifact root
- 关键指标:
  - 模型: dodo_depth, hard_depth, amplitude sensing, legacy_max forward norm, per_sample_mean_std measurement norm
  - deploy 1: 491 lowfg tiles, 5 measured — meas_bg_fraction=1.0000（全部 measurement energy 来自背景）
  - deploy 16: 406 lowfg tiles, 5 measured — meas_bg_fraction=1.0000
  - 所有 10 个 tile 的 measurement 均为 finite
  - measurement_vs_hs_energy_ratio: deploy1 ~0.93, deploy16 ~0.96
- 相关代码: `round23_lowfg_diag.py`（完全重写，加载 DoDo checkpoint 运行真实光学前向）

摘要:
修复了 Codex review 标记的阻断项。重写后的诊断加载 R12 depth-best checkpoint 的 `DepthAwareDoDoForwardModel` camera，对低前景 128×128 tile 运行真实的 DoDo 光学前向（深度分层 → 传播 → DOE → sensing → 测量），分别计算 full/fg-only/bg-only measurement energy。bg-only 策略：`bg_hs = hs * (1-fg_mask)`，bg depth clamp 到 min_depth，bg_valid_mask = 1-fg_mask，与推理 pipeline 一致。

**关键结果**: 纯背景 tile（valid_ratio=0）的 measurement energy 100% 来自背景 HS 内容 + clamped depth。这些 tile 在 per-patch `per_sample_mean_std` 归一化下会被拉伸到全动态范围，定量确认了 full-scene 白底/粉糊 artifacts 的物理原因。Measurement 能量 (6-8e3) 与 HS 输入能量 (6.7-7.9e3) 的比例约为 0.93-0.96，说明光学前向传输保留了大部分背景能量。

CSV 同时保留 raw HS energy 列和 measurement energy 列以便对比。JSON summary 完整记录 checkpoint path、sensor_measurement、depth_layering_mode、forward_norm、measurement_norm、background_policy。

---

## EXP-20260509-06: spectral_capacity_smoke

- 日期: 2026-05-09
- 轮次: 24（spectral capacity diagnostic）
- 类型: 诊断（无训练，smoke run）
- 状态: 完成
- Checkpoint: 无（fresh init, Zeros DOE, amplitude sensing）
- Artifact root: 临时 `/tmp/diag_smoke_test`（已清理）
- 推理结果路径: 无
- 关键指标:
  - Diag B: adjacent_corr mean=0.9547, std=0.0616, min=0.6877, max=0.9969
  - Diag C: energy mean=1.16e4, std=7.25e3, min=3.34e3, max=2.70e4
  - Diag D: effective_rank=25.69, condition_number=7.83e3, mutual_coherence=0.9977, 144 columns
- 相关代码: `scripts/diagnose_measurement_spectral_capacity.py`（新增）

摘要:
首次运行 spectral capacity 诊断脚本，使用 fresh Zeros DOE（无训练）+ amplitude sensing + hard_depth layering。关键发现：
1. 相邻波长 correlation 极高（mean=0.955），说明当前 3-channel measurement 对相邻波长几乎无法区分。
2. 各波长 energy 差异约 8×（3.3e3~2.7e4），边缘波长能量较低。
3. 测量矩阵 mutual_coherence=0.998（接近 1），列向量几乎共线——这是一个严重的信息瓶颈信号。
4. condition_number=7.83e3，矩阵 moderately ill-conditioned。
这些结果定量支持了"3-channel RGB measurement 携带的光谱可分信息严重不足"的假设。训练过的 DOE 可能会改善这些指标——需要通过 `--checkpoint <path>` 对比验证。

---

## EXP-20260509-07: round24_spectral_capacity_trained + comparison

- 日期: 2026-05-09
- 轮次: 25（trained spectral capacity + SensingLayer refactoring plan）
- 类型: 诊断（无训练，comparison）
- 状态: 完成
- Checkpoint: R12 `depth-best-epoch=226.ckpt`（仅用于诊断加载）
- Artifact root (init): `infer_results/DoDo-change/round24_spectral_capacity_init_v1/`
- Artifact root (trained): `infer_results/DoDo-change/round24_spectral_capacity_trained_v1/`
- Comparison: `infer_results/DoDo-change/round24_compare_summary.json`
- 关键指标 (init → trained):

| Metric | Zeros DOE | Trained DOE | Change |
|--------|-----------|-------------|--------|
| adjacent_corr_mean | 0.9547 | 0.9103 | −4.7% ✓ |
| adjacent_corr_max | 0.9969 | 0.9461 | −5.1% ✓ |
| energy_std | 7.25e3 | 8.85e2 | −87.8% ✓ |
| effective_rank | 25.7 | 95.8 | +273% ✓ |
| condition_number | 7.83e3 | 32.0 | −99.6% ✓ |
| mutual_coherence | 0.9977 | 0.9000 | −9.8% ✓ |

- 相关代码: `scripts/diagnose_measurement_spectral_capacity.py`, `scripts/compare_diagnosis.py`

摘要:
用 R12 训练过的 DOE（260 epochs）跑同样的 4 个 spectral capacity 诊断，与 Zeros DOE 对比。**训练显著改善了 10/11 个指标**：
- effective_rank 从 25.7 → 95.8（3.7×），说明 DOE 训练确实学会了给不同波长赋予不同的 diffractive encoding。
- condition_number 从 7830 → 32（244× better conditioned），反问题的数值稳定性大幅提升。
- energy 分布从极度不均（std=7250）变为相当均匀（std=885）。
- adjacent_corr 从 0.955 → 0.910，下降但仍有提升空间。

**但关键警告**：
- mutual_coherence 仍为 0.900（>0.9 阈值）：即使训练了 260 epochs 的 DOE，3-channel measurement 的列向量仍然高度相关。
- effective_rank=95.8 仍 < 总列数 144：测量矩阵仍然 rank-deficient。
- adjacent_corr 仍在 0.9+ 范围：相邻波长的 measurement response 区分度仍然不足。

**核心结论**：训练 DOE 能显著改善光谱编码（不是 loss 的问题），但 3-channel measurement 的信息瓶颈是结构性的——即使最优 DOE 也无法突破 3 channels 的容量上限。下一步应该增加 measurement channels（CFA/CCA 或 spectral_bins）而不是继续调整 loss 或 DOE 训练。

---

## EXP-20260509-08: doe0_rgb_vs_spectral_bins comparison

- 日期: 2026-05-09
- 轮次: 26（DOE0 RGB vs spectral_bins 对比）
- 类型: 诊断（Zeros DOE, 4 sensing configs）
- 状态: 完成
- Checkpoint: 无（Zeros DOE / fresh init）
- Artifact root: `infer_results/DoDo-change/doe0_bins_compare/`
- 关键指标:

| Setting | ch | effective_rank | condition_number | mutual_coherence | adjacent_corr_mean | energy_std |
|---------|----|----------------|------------------|------------------|--------------------|------------|
| rgb_3   | 3  | 73.4           | 3.45e4           | 0.9991           | 0.964              | 4672       |
| bins_6  | 6  | 93.9           | 1.12e4           | 0.9991           | 0.754              | 2850       |
| bins_9  | 9  | **112.1**      | **9.85e3**       | 0.9991           | 0.629              | 2850       |
| bins_12 | 12 | 112.1          | 9.85e3           | 0.9991           | **0.504**          | 2850       |

- 相关代码: `scripts/compare_doe0_rgb_vs_spectral_bins.py`

摘要:
在 Zeros DOE 下对比 4 种 sensing config（rgb 3ch, spectral_bins 6/9/12ch）的光谱编码容量。核心发现：
1. **effective_rank 随 channels 增加**：73→94→112，但在 9→12 bins 饱和（112.08 vs 112.08）。Zeros DOE 下 12 bins 不提供超过 9 bins 的额外容量。
2. **condition_number 显著改善**：34509→11187→9851→9851，9 bins 时达最优。
3. **adjacent_corr_mean 大幅下降**：0.964→0.754→0.629→0.504，spectral binning 天然降低了相邻波长 correlation（因为 bin 内多个波长被合并）。
4. **mutual_coherence 几乎不变**（~0.9991）：说明 Zeros DOE 不提供 spatial diversity——不同波长+位置的 measurement columns 仍然高度共线。
5. **9 bins 是最优性价比**：effective_rank 和 condition_number 在 9 bins 处饱和，12 bins 无额外收益。
6. 这是 capacity diagnosis only——不代表最终 PSNR/SAM。要评估重建质量需要重新训练 decoder。
