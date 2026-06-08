# Review: DoDo-change training readiness

## 结论

第六轮 cleanup 后，`dodo_depth` 端到端接入已经没有明显的结构性阻断：新 DoDo 光学模型能接入 `SnapshotDepthHS`，mamba decoder 的 3 通道 measurement 输入已修复，fast-dev run 已完成，旧 `optics/` 路径保留。

但在正式实验训练前还有实验闭环层面的缺口，需要 Claude 先补齐：

- 当前深度验证主要仍是 IPS 空间误差；用户目标需要米制 MAE。
- 当前缺少稳定保存验证重建结果、指标表和实验配置的机制，不利于迭代分析。
- `dodo_depth` 全零输入会产生 NaN，训练依赖 patch filter 避免全零批次；正式训练前应在 dodo 路径增加非有限 measurement 防护或显式诊断。
- 后续若启用 `optimize_optics=True`，需要确保 optimizer step 后调用 `clamp_parameters_()`，否则 DOE 参数投影只存在但不会被训练循环执行。

## 已完成

- `DepthAwareDoDoForwardModel` 已作为 `dodo_depth` 光学前向接入。
- Dataset 已提供 `depth_metric` 给光学前向，同时保留 `depth_map` 作为 IPS 深度监督。
- `models.simple_model_mamba.SimpleModelHS` 已支持 `measurement_channels=3`。
- `dodo_depth` 参数校验、PSF guard、日志 3 通道 measurement 越界、`dodo_depth_layers=None` 等阻断项已修复。
- 已完成 fast-dev run：1 train + 1 val batch 无崩溃。

## 进入训练前必须补齐

1. 增加并记录 `validation/mae_depth_m`：将 `est_depthmaps` 和 `target_depthmaps` 从 IPS `[0,1]` 转为 metric meters 后，在 `final_mask` 内计算 MAE。
2. 增加并记录稳定的 hyperspectral PSNR：至少记录 `validation/psnr_hs_masked`，建议同时记录 full-image PSNR。
3. 保存验证重建结果：每个实验至少保存若干 val 样本的 HS 快视图、GT/估计深度图、误差图、指标 JSON/CSV、训练命令和 hparams。
4. `dodo_depth` measurement 非有限防护：如果光学输出出现 NaN/Inf，记录诊断并用安全值处理，或抛出带 mask 有效比例的清晰错误。正式训练不应静默污染 loss。
5. DoDo 光学参数训练钩子：若 `optimize_optics=True` 且 camera 有 `clamp_parameters_()`，optimizer step 后必须调用。

## 实验策略

第一阶段先不要优化 DOE，先验证 reconstruction network 能从 DoDo 3 通道 measurement 学到可用重建。

推荐实验顺序：

1. `preflight`: 极小 batch 验证完整日志、指标、artifact 保存。
2. `baseline_clean`: `optimize_optics=False`、无噪声，判断上限和网络可学习性。
3. `baseline_noise`: 在 clean baseline 正常后启用默认噪声，判断鲁棒性。
4. `loss_balance`: 按 PSNR/MAE_m 的短板调整 `image_loss_weight`、`depth_loss_weight`、`depth_smooth_weight`，必要时新增 metric-depth loss。
5. `optics_finetune`: 只有 decoder baseline 收敛后才启用 `optimize_optics=True`，使用小 `optics_lr` 并强制 DOE clamp。

## 当前建议

下一轮 Claude 应先实现训练评估闭环，再跑 preflight 和一个 baseline clean 训练。训练过程中 Claude 只需等待实验结束；训练完成后分析指标和保存图像，再按 `TASK.md` 的决策树最多追加一个针对性实验，然后停止等待 Codex review。
