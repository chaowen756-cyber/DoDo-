# Review: DoDo-change Round 15 Full-Scene Validity

Date: 2026-05-07
Reviewer: Codex
Change: DoDo-change

## Findings

1. Blocking - Round 15 的“best checkpoint 已确认可用于 full-scene”结论不成立。

   用户观察和 quicklook 直接冲突于 Section 15.6/15.7 的结论。`gt_hs.png` 是正常彩色场景，但 `est_hs.png` 呈现白底、粉色糊状、颜色不可分；这不能被描述为 “excellent HS reconstruction”。`est_depth_fixed_scale.png` 虽有非零纹理和轮廓，但有明显 tile/DOE 纹理，且 depth MAE 约 0.45-0.49m，已经远差于 128x128 validation 的 0.178m。

2. Blocking - 当前 PSNR/MAE 不足以解释或验收 full-scene 结果。

   deploy 1 的 `hs_psnr_masked_db=30.02` 与可视化质量强烈不一致，说明至少需要审计 metric 计算、mask 范围、归一化方式、per-band error 和 SAM。Round 15 已有 SAM 0.540 rad，这是很差的谱角结果，不能用单个 masked PSNR 宣称 HS 重建好。

3. High - depth “non-constant” 判据过弱。

   Section 15 用 `pred_depth_std > 0.02` 判定 depth 有效，但预测图可见强烈周期性纹理，可能是 tiled stitching/DOE artifact，而不是物理深度区分。下一轮必须加入 constant-depth baseline、GT depth distribution、pred-vs-GT correlation、edge/boundary vs interior MAE，判断模型是否真的优于简单常数预测。

4. High - 光学 forward contract 需要重新写清楚。

   当前 DoDo-depth forward 是按 metric depth 分层，将 25 通道光谱经 depth-dependent optical propagation 压缩成 3 通道 measurement，这个方向符合 coded snapshot 重建思路。但在 `infer_contect.py` full-scene eval 中，measurement 是用 `GT HS + GT depth` 现场模拟出来的；如果真实部署时没有 GT depth，这不是 measurement-only 的真实联合重建 contract。需要明确当前实验到底是在验证“物理仿真可逆性”，还是“真实 measurement 到 HS/depth 的联合重建”。

5. High - measurement normalization 可能破坏了 depth/光谱可辨识信息。

   `DepthAwareDoDoForwardModel` 内部先 `_normalize_once(y_sum)`，`snapshotdepth_hs.py` 又按 `dodo_measurement_norm=per_sample_mean_std` 做每样本归一化。full-scene tiled inference 实际变成每个 128x128 patch 独立标准化，这会移除绝对亮度、背景比例和部分深度相关能量线索，并可能导致跨 tile 颜色/深度尺度不一致。下一轮必须保存并比较 norm 前后 measurement 分布。

## Assessment

Round 15 完成了 full-scene inference plumbing，但没有完成可靠验收。当前结果应标记为 blocking failure pending audit，而不是确认最终模型可用。

下一轮不应继续训练，也不应直接改架构。优先做 metric/preprocessing/forward contract 诊断，产出可复现证据：保存原始数组、per-band metrics、constant baselines、measurement stats、tile weight/stitching diagnostics，并解释训练 crop 指标与 full-scene 可视化差异的根因。
