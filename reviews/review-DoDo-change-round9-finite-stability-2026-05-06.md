# Review: DoDo-change 第九轮 finite stability 结果

## 结论

第九轮成功修复了 DoDo measurement 非有限根因。`_normalize_once(eps=0)` 导致的除零/NaN 风险已修复；finite smoke、finite preflight、foreground one-batch 和 20 epoch short sanity 均显示 `nonfinite_count == 0`。HS 重建从失败状态恢复到可训练状态。

当前可以恢复正式长训练，但必须使用 finite-safe 参数，并保持 `--dodo_nonfinite_policy fail`。不能回到旧的 `nan_to_num(..., 0)` 继续训练策略。

## 已完成

- `torch_optics/forward_dodo.py` 修复 `_normalize_once`：`eps=1e-8`，并在 `y_max` 非有限或 `<=0` 时不再除零。
- 增加 stage-wise finite diagnostics，synthetic foreground、zero input 和真实 foreground batch 均通过。
- `snapshotdepth_hs.py` 增加 `--dodo_nonfinite_policy fail`。
- backward 后 grad diagnostics 已迁移到 `on_after_backward()`。
- finite preflight 通过：`nonfinite_count=0`，captimgs finite/nonzero，DOE/decoder grad finite。
- foreground one-batch overfit 通过：PSNR +3.39 dB，depth 非常数，DOE grad finite。
- short sanity 20 epoch 通过关键修复验证：PSNR 从 5.62 到 15.03，`nonfinite_count=0`，HS/depth 不再常数化。

## Findings

### High: 可以恢复长训，但必须带 fail-fast nonfinite gate

第九轮证明 DoDo measurement 非有限问题已修复。下一轮可以恢复 260 epoch 正式训练，参数必须包含：

- `--decoder_norm group`
- `--dodo_measurement_norm per_sample_mean_std`
- `--dodo_nonfinite_policy fail`
- `--depth_smooth_weight 0`
- `--dodo_doe_type New`
- `--optimize_optics`

如果训练中出现任何 nonfinite captimgs，应立即失败并停止，不允许替换为 0 后继续训练。

### Medium: depth MAE 仍是主要风险

short sanity 的 HS PSNR 已明显改善，但 `validation/mae_depth_m = 0.933m`，仍然很差。虽然 depth 输出不再常数化，但深度精度尚未验收。

下一轮不能只汇报 PSNR；必须把 depth 作为同等重点记录：

- best/last `validation/mae_depth_m`
- valid-mask `est_depth_std`
- depth error quicklook
- depth 是否反向、饱和到 0/1、边界是否错位

如果 260 epoch 后 HS PSNR 改善而 depth MAE 仍高，应只做一个单变量 follow-up：提高 `depth_loss_weight`。

### Medium: final grad metrics 可能仍有 sampling/timing 噪声

short sanity 的最终 `metrics.json` 中一些 grad norm 为 0，但日志中 epoch 19 多个 step 的 DOE grad finite 且非零。下一轮应以日志和 metrics 双重记录为准，不要仅凭最后一次 metrics 中的 0 判断梯度缺失。

## 下一轮建议

运行 finite-safe 260 epoch joint training。完成后根据结果最多做一个 targeted follow-up：

- 如果 `nonfinite_count > 0` 或命令失败：停止，不做 follow-up，记录首个错误。
- 如果 HS PSNR 上升且 depth MAE 仍明显偏高：运行 `depth_loss_weight=5` 的单变量实验。
- 如果 depth 输出常数化：运行 one-batch foreground overfit depth-focused 诊断，不要继续长训。
- 如果 HS 和 depth 都稳定改善：停止等待 Codex review，不要自行继续更多实验。
