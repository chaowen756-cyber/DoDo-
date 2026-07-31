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
