# Review: DoDo-change 第八轮 diagnostics + repair 结果

## 结论

第八轮完成了 artifact、diagnostics、GroupNorm、measurement normalization、baseline one-batch、repair one-batch 和 20 epoch sanity run。但当前结果不能验收，也不能恢复 260 epoch。

核心阻断已经从 decoder 训练配置转移为：`dodo_depth` 光学测量在大量有效输入上产生整张非有限值，随后被 `nan_to_num(..., 0)` 替换为全零 measurement，decoder 只能输出 bias 驱动的常数图。

下一轮必须先修 DoDo measurement 非有限根因，并用 `nonfinite_count == 0` 的前景样本 overfit 证明数据流可学习。禁止继续在全 NaN/全 0 measurement 上调 loss 或跑长训练。

## 已完成

- Artifact root、`command.txt`、`hparams.json`、`git_status.txt`、`logs/train.log` 和 6 张 quicklook PNG 已落地。
- `--decoder_norm group` 和 `--dodo_measurement_norm per_sample_mean_std` 已实现。
- `--artifact_root` 已实现。
- DOE optimizer membership、clamp hook、非有限计数和 quicklook 保存已有基础实现。
- Baseline one-batch 失败后执行了 targeted repair patch。
- Repair one-batch PSNR 从 6.171 到 10.924，说明 decoder 在某种单样本设置下有拟合能力。

## Findings

### High: 20 epoch sanity 仍完全失败，不能恢复长训

`infer_results/DoDo-change/DoDo_depth_repair_short20_v1/20260506_125152/metrics.json` 显示：

- `validation/psnr_hs_masked = 6.474`，只比 epoch 0 约高 0.024 dB。
- `train_misc/est_depth_std = 0.0`。
- `train_misc/est_depth_max == train_misc/est_depth_min == 0.5008958578`。
- `train_misc/est_image_max == train_misc/est_image_min == 0.4986894131`。
- `train_misc/nonfinite_count = 360`。
- `diag/captimgs_min == max == mean == std == 0.0`。

这说明 repair 没有解决多样本训练失败，输出仍是常数。

### High: DoDo forward 在有效前景输入上也整张非有限

短 sanity 日志不仅有全背景 batch，也有明显前景 batch，例如：

- `mask_ratio=0.2720, input_spectral_sum=1384.4119`
- `mask_ratio=0.5379, input_spectral_sum=19889.5234`
- `mask_ratio=1.0000, input_spectral_sum=25173.5332`

这些 batch 仍出现 `98304 non-finite values in captimgs`，即 B=2 时 3x128x128 全部测量值非有限。这不是单纯 patch_filter/全背景问题，而是 DoDo 光学前向数值稳定性问题。

### High: `nan_to_num` 兜底正在掩盖阻断并污染优化

`snapshotdepth_hs.py:683-695` 在 `captimgs` 非有限时把 NaN/Inf 替换为 0 并继续训练。短 sanity 中 replacement 后 `captimgs` 统计为全 0，导致 decoder 收到零输入，输出常数。

同时日志显示 DOE gradient 大量为 NaN：`doe1.zernike_coeffs.grad norm=nan, finite=False`。继续训练会污染 optimizer state，不能作为有效实验。

### High: `torch_optics.forward_dodo._normalize_once` 存在明显数值风险

`torch_optics/forward_dodo.py:11-12`:

```python
def _normalize_once(y: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
    return y / (torch.max(y) + eps)
```

`eps=0` 在 `max == 0` 或 `max == NaN` 时会直接产生 NaN。DepthAwareDoDoForwardModel 在所有 depth bins 累加后调用它：`torch_optics/forward_dodo.py:239`。

下一轮必须做 stage-wise finite smoke，定位非有限是在 propagation、DOE、sensing 还是 final normalization 出现；然后做最小数值稳定修复。

### Medium: 第八轮 gradient diagnostics 仍不可靠

`snapshotdepth_hs.py:233-312` 在 `training_step` 内读取参数 `.grad`，此时 backward 尚未发生，所以 metrics 中很多 `diag/grad_* = 0.0` 不是可靠结论。DOE hook 日志能捕捉 backward，但没有把 finite/nonfinite 梯度状态以结构化 metrics 持久化。

下一轮应使用 Lightning `on_after_backward` 或等价 hook 记录 decoder/DOE grad norm。

### Medium: hparams 与实际 train crop 行为不一致

`hparams.json` 记录 `randcrop=False`，但 `snapshotdepth_trainer_hs.py` 当前给 train dataset 传了 `randcrop=True`。这会误导实验复盘。下一轮需要让 CLI、hparams 和实际行为一致；如果 DoDo train 强制 randcrop，应明确记录并打印。

## 下一轮必须做

1. 修 DoDo measurement 非有限根因，不要继续只调 decoder。
2. 将 DoDo 非有限策略从“替换为 0 继续训练”改为本轮实验使用 `fail` 或至少 `skip_batch`；训练通过条件必须是 `nonfinite_count == 0`。
3. 修正 `_normalize_once` 的 `eps=0` 风险，并增加 per-sample safe normalization 或显式 finite checks。
4. 增加 stage-wise finite smoke：synthetic nonzero input 和真实前景 batch 都必须检查每个 DoDo stage 的 finite/min/max。
5. 修正 gradient diagnostics 到 backward 之后记录。
6. 运行真实前景 one-batch overfit，必须证明 batch `mask_ratio >= 0.2`、`input_spectral_sum > 0`、`captimgs` finite 且非零。
7. 只有在前景 one-batch overfit 和 20 epoch sanity 都 `nonfinite_count == 0` 且不常数化后，下一轮 Codex 才能考虑恢复 260 epoch。
