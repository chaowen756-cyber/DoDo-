# Experiments

## 2026-07-31 CPU smoke：rank-9 vs free-150，4 depth，10 step

- 类型：preflight/smoke。
- checkpoint：`/tmp/doe_preopt_rank9_free150_smoke/*/seed_123/best_doe.pt`。
- artifact root：`/tmp/doe_preopt_rank9_free150_smoke`。
- 推理结果：无。
- 状态：完成。
- 摘要：验证两种 DOE 均可独立反传、投影、保存和可视化。rank-9 的
  MTF@0.05 mean 从 0.0317 到 0.0321，free-150 从 0.0343 到 0.0355；该
  smoke 仅有4个深度层，不用于判定完整深度可编码性。

## 2026-07-31 CPU preflight：rank-9 vs free-150，16 depth，20 step

- 类型：短可行性对照。
- checkpoint：`/tmp/doe_preopt_full16_short/*/seed_123/best_doe.pt`。
- artifact root：`/tmp/doe_preopt_full16_short`。
- 推理结果：无。
- 状态：完成。
- 摘要：完整16层下 rank-9 的 MTF@0.05 mean 从 0.0297 到 0.0305、深度
  cosine 从 0.9665 到 0.9655；free-150 从 0.0320 到 0.0340、深度 cosine
  从 0.9647 到 0.9633。20步只验证方向和闭环，不能替代1000步GPU实验。

## 2026-07-31 CPU final smoke：free-150，2 depth，1 step

- 类型：最终接口/产物 smoke。
- checkpoint：`/tmp/doe_preopt_final_smoke/free150/seed_31/best_doe.pt`。
- artifact root：`/tmp/doe_preopt_final_smoke`。
- 推理结果：无。
- 状态：完成。
- 摘要：确认最终版本的 free-150 路径可构造、反传、更新和保存；
  MTF@0.05 mean 从 0.0397 到 0.0405。该实验只有2个深度层和1步，
  只作为接口 smoke，不用于可编码性结论。

## 2026-07-31 Fisher preflight：两种容量，16 depth，20 step

- 类型：Fisher 目标与学习率预检。
- checkpoint：`/tmp/doe_fisher_lr1e2_20step/*/seed_123/best_doe.pt`。
- artifact root：`/tmp/doe_fisher_lr1e2_20step`。
- 推理结果：无。
- 关键指标：rank-9 A-optimality `2.343e6 → 2.295e6`，MTF@0.05 mean
  `0.0297 → 0.0303`；free-150 A-optimality `2.020e6 → 1.646e6`，
  MTF@0.05 mean `0.0320 → 0.0329`。
- 状态：完成。
- 摘要：在相同0.6 μm初始化、3 μm上限和 `Adam lr=1e-2` 下，两种容量
  均稳定优化且未牺牲平均 MTF；free-150 的 Fisher 改善明显更快。该结果支持
  保留 `1e-2` 进入正式1000步实验，但20步仍不构成最终容量结论。

## 2026-07-31 初始化诊断：随机 vs Fresnel 投影

- 类型：初始化选择诊断。
- checkpoint：无。
- artifact root：无（只读即时诊断）。
- 推理结果：无。
- 关键指标：rank-9 对二次 Fresnel 高度的相对拟合误差约0.866；free-150
  在0.6 μm RMS下随机初始化 A-optimality 为 `2.020e6`，较好符号 Fresnel
  为 `2.218e6`。
- 状态：完成。
- 摘要：Fresnel 是原论文全相位自由度 Fisher 优化的合理起点，但在当前
  rank-9/free-150 公平容量对照中并未形成一致优势；采用固定 seed 随机初始化
  可以避免把基底拟合误差和初始化差异混入容量结论。

## 2026-07-31 Task-Fisher preflight：free-150，16 depth，20 step

- 类型：任务 CRLB 目标与学习率 preflight。
- checkpoint：`/tmp/doe_taskfisher_free150_20step/free150/seed_123/best_doe.pt`。
- artifact root：`/tmp/doe_taskfisher_free150_20step`。
- 推理结果：无。
- 关键指标：task A-optimality `6.635e5 → 5.979e5`，MTF@0.05 mean
  `0.0320 → 0.0332`，depth cosine `0.9647 → 0.9642`，高度 RMS
  `0.600 → 0.956 μm`。
- 状态：完成。
- 摘要：在完整16深度/25波长上，任务加权 Fisher 在 `lr=1e-2` 下稳定下降，
  且未牺牲平均 MTF 或能量范围。20步仅确认目标与梯度方向，可进入1000步双
  seed GPU 对照。

## 2026-07-31 Task-Fisher 正式实验：free-150 双 seed，1000 step

- 类型：DOE-only 正式可行性实验。
- checkpoint：
  `experiments/PSF卷积/DOE预优化/taskFisher_free150_seed{123,456}_1000step/free150/seed_*/best_doe.pt`。
- artifact root：上述两个 `taskFisher_*` 目录。
- 推理结果：无。
- 关键指标：seed123/456 task A-optimality 分别约下降25.2%/23.2%，深度
  CRLB下降37.6%/35.5%，波长CRLB只下降11.7%/10.1%；sensor-weighted
  spectral cosine 均未改善，MTF floor 均未满足。
- 状态：完成，未通过综合可行性判据。
- 摘要：双 seed 一致证明 free-150 能改善局部 Fisher，尤其是深度导数信息，
  但没有学出明显的波长 PSF 形状差异。日志中0–100步总损失上升来自 separation
  warm-up 的目标口径变化；最优点后只有约0.1%–0.2%反弹。该结果促使下一轮
  将 DOE optical-only 形状编码从 RGB response 信息中显式拆出。

## 2026-07-31 Optical-shape preflight：free-150，16 depth

- 类型：新目标、固定日志口径与学习率预检。
- checkpoint：`/tmp/doe_optical_shape_preflight_46f93ec/free150/seed_123/best_doe.pt`
  与 `/tmp/doe_optical_shape_boundary_preflight/free150/seed_456/best_doe.pt`。
- artifact root：上述两个 `/tmp` 目录。
- 推理结果：无。
- 关键指标：30步固定完整损失 `0.9423 → 0.9038`，optical spectral cosine
  `0.9621 → 0.9605`；200步固定完整损失 `0.9282 → 0.8749`，optical
  spectral cosine `0.9612 → 0.9584`，sensor spectral cosine
  `0.9737 → 0.9733`。
- 状态：完成。
- 摘要：`loss/train_total` 在 warm-up 中随新项加入而上升，但
  `loss/full_total` 从第一步开始持续下降，确认日志口径和目标梯度正确。该短跑
  未达到3 μm边界，只用于确认 optical-only 目标确实能推动单色 PSF 形状变化。

## 2026-07-31 RMS 边界压力测试：free-150，50 step

- 类型：约束优化压力测试。
- checkpoint：`/tmp/doe_optical_shape_boundary_hit_v2/free150/seed_789/best_doe.pt`。
- artifact root：`/tmp/doe_optical_shape_boundary_hit_v2`。
- 推理结果：无。
- 关键指标：从2.99 μm RMS初始化，50步完整损失 `0.9074 → 0.8918`；
  49次切向修正、50次安全回缩，最小回缩比例 `0.9895`。
- 状态：完成。
- 摘要：初版同时投影 Adam 一阶动量但未同步二阶动量，压力测试发现一次
  `0.0027` 异常回缩；删除错误的动量投影后重新测试，边界训练恢复稳定下降。
  正式实现只切向化实际候选更新，并保留轻量二阶安全回缩。

## 2026-07-31 Optical-shape 正式实验：free-150双seed，3000 step

- 类型：optical-only DOE正式可行性实验。
- checkpoint：
  `experiments/PSF卷积/DOE预优化/opticalShape_free150_seed{123,456}_1000step_commit-7d319b1/free150/seed_*/best_doe.pt`。
- artifact root：上述两个目录（目录名写1000step，但command确认实际均为3000步）。
- 推理结果：无。
- 关键指标：seed123/456相邻波长optical cosine最终均约0.98977；offset-4约
  0.9133；task Fisher约5.20e5/5.26e5；MTF@0.05 p10仍约0.0116/0.0112。
- 状态：完成，未通过综合可行性判据。
- 摘要：两个不同高度图收敛到几乎相同PSF指标，证明free-150平滑Zernike达到
  稳定表达上限，而非随机初始化或优化步数不足。深度和宽间隔波长稍有编码，
  相邻10 nm波段与最差空间MTF仍失败，因此不进入联合CNN训练。

## 2026-07-31 Pixel-phase参数化与学习率预检：16 depth，200 step

- 类型：高容量逐像素相位DOE与学习率选择。
- checkpoint：`/tmp/doe_pixelphase_lr{1e2,5e2,1e1}_200/.../best_doe.pt`及
  `/tmp/doe_pixelphase_wrapped_lr1e1_200/.../best_doe.pt`。
- artifact root：上述 `/tmp` 目录。
- 推理结果：无。
- 关键指标：未包裹模拟中lr 0.01/0.05/0.1的完整损失分别约0.827/0.786/0.779；
  最终采用物理包裹参与前向后，lr0.1在200步达到完整损失0.797、task Fisher
  3.66e5、MTF@0.05 mean/p10 0.0338/0.0111、optical spectral综合cosine
  0.9567。
- 状态：完成，方向显著优于free-150，选用lr0.1。
- 摘要：逐像素相位仅200步就明显超过free-150的3000步Fisher与形状分离；
  包裹后物理高度严格位于约0–0.98 μm单周期。PSF出现随波长变化的细纹，说明
  高空间频率/相位包裹表达能力是上一轮关键瓶颈。

## 2026-07-31 Pixel-phase邻近波段权重对照：200 step

- 类型：目标权重消融。
- checkpoint：`/tmp/doe_pixelphase_adjacent_lr1e1_200/pixelphase/seed_456/best_doe.pt`。
- artifact root：`/tmp/doe_pixelphase_adjacent_lr1e1_200`。
- 推理结果：无。
- 关键指标：仅使用offset1/2并把光谱权重从5升到10，相邻optical cosine由
  0.98997小幅降至0.98983，但task Fisher由3.66e5变差至3.71e5，MTF无明显收益。
- 状态：完成，不采用该配置。
- 摘要：过度强化相邻波段只能换来约1.3e-4额外cosine改善，性价比很低；正式
  实验保留offset1/2/4、光谱权重5的平衡目标。
