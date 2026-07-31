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
