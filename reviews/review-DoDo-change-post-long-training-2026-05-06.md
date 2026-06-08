# Review: DoDo-change 第七轮长训练后验收

## 结论

上一轮完成了部分关键实现和一次有效的 DOE-optimized preflight，但没有完成原 stop condition：260 epoch 主实验提前停止在 epoch 62，未执行 one-batch overfit，也没有完整满足 artifact、日志、DOE 梯度验证和非有限诊断要求。

当前不建议直接继续 260 epoch。应先补齐可追溯 instrumentation，然后运行 one-batch overfit 来确认深度头能否学习非恒定深度。

用户追问后补充：原下一轮方案偏 diagnostics，没有强制包含“针对实验效果失败的代码修正”。这不够。60+ epoch 后光谱重建约 6.5 dB 且 depth 常数化，已经不是单纯日志问题；下一轮必须在复现实验失败后实施最小 repair patch，再用短实验验证效果。

## 已完成

- `--dodo_doe_type` CLI 已加入，`dodo_depth` 构造时会把 `doe_type_a` 传给 `DepthAwareDoDoForwardModel`：`snapshotdepth_hs.py:402-409`。
- Preflight 命令包含 `--dodo_doe_type New` 和 `--optimize_optics`，exit code 为 0；日志确认 `doe1.zernike_coeffs.requires_grad=True`。
- `validation/mae_depth_m` 已实现，使用 `ips_to_metric` 并按 `final_mask` 统计：`snapshotdepth_hs.py:285-293`。
- `validation/psnr_hs_masked` 已实现，按 masked valid pixels 和全部 HS channels 统计：`snapshotdepth_hs.py:295-305`。
- DoDo 测量输出的 NaN/Inf 已有拦截和 `torch.nan_to_num` 兜底：`snapshotdepth_hs.py:479-485`。
- `--checkpoint_monitor` / `--checkpoint_mode` 已加入，默认 monitor 为 `validation/psnr_hs_masked`、mode 为 `max`：`snapshotdepth_trainer_hs.py:123-134`、`snapshotdepth_hs.py:897-898`。
- `optimizer_step()` 中已有 `camera.clamp_parameters_()` 调用：`snapshotdepth_hs.py:77-79`。

## Findings

### High: 主实验未完成 260 epoch，且未执行 Phase 4

`implementation-notes.md` 记录主实验提前终止于 epoch 62，原因是用户要求提前停止。原任务要求 Phase 2 运行 260 epoch，Phase 4 在结果异常时最多执行一个针对性实验。当前深度输出常数化，符合 one-batch overfit 触发条件，但该实验尚未执行。

证据：
- `infer_results/DoDo-change/DoDo_depth_optics_joint_260_v1/version_0/metrics.json` 中 `epoch` 为 62。
- `train_misc/est_depth_max == train_misc/est_depth_min == 0.5038233995437622`，深度输出为常数。

### High: DOE 梯度与 optimizer membership 未被验证

当前日志只确认 `requires_grad=True`，但没有确认 backward 后 `camera.doe1.zernike_coeffs.grad is not None` 且 finite，也没有记录 `doe1.zernike_coeffs` 是否按 identity 出现在 optics param group 中。

代码层面目前只有构造时打印 `requires_grad`：`snapshotdepth_hs.py:413-415`。`training_step()` 没有 after-backward DOE grad 诊断：`snapshotdepth_hs.py:146-206`。

影响：无法证明 DOE 参数实际从 reconstruction loss 接收到梯度并参与优化。`requires_grad=True` 只是必要条件，不是训练成功证据。

### High: 主训练 stdout/stderr 日志为空

主实验记录的日志文件大小为 0：

`infer_results/DoDo-change/DoDo_depth_optics_joint_260_v1/20260506_100327/logs/train.log`

这违反了长训练输出规则，也导致异常、诊断、checkpoint 选择和 stopping context 无法从 `train.log` 复盘。下一轮必须修正启动方式，建议使用 `conda run --no-capture-output ... > "$EXP_ROOT/logs/train.log" 2>&1`，并确认日志非空后再长跑。

### Medium: Artifact contract 未完成

当前 artifact 保存只覆盖 `metrics.json` 和 `hparams.json`，启动脚本保存 `command.txt`：`snapshotdepth_hs.py:351-369`、`snapshotdepth_trainer_hs.py:144-155`。

缺失项：
- 没有 `git_status.txt`。
- 没有把实际 `logs/train.log` 放入同一个 version/timestamp artifact root。
- 没有保存 PNG quicklooks，例如 `capt_rgb.png`、`gt_hs_rgb.png`、`est_hs_rgb.png`、`gt_depth_m.png`、`est_depth_m.png`、`depth_abs_error_m.png`。
- 目录结构不是要求的 `infer_results/DoDo-change/<experiment_name>/<timestamp>/`，而是混用了 timestamp log dir 和 `version_0` artifact dir。

### Medium: 非有限诊断信息不足且未持久化

当前 guard 只打印 nonfinite 数量和 `mask_ratio`：`snapshotdepth_hs.py:480-484`。任务要求还要记录输入 spectral sum、depth range，并把 nonfinite 发生情况写入日志或 metrics。

影响：如果后续出现 NaN/Inf，只能知道发生了替换，无法定位是全零光谱、异常 mask 还是 depth range 导致。

### Medium: validation metric 边界条件仍不严谨

`validation/psnr_hs_masked` 没有显式检查 `est_images.shape == target_images.shape`：`snapshotdepth_hs.py:295-305`。此外，当 `num_valid_px == 0` 时，`mae_depth_m` 设为 NaN，但 epoch 聚合仍用 `_val_steps` 做分母，可能把无效 batch 稀释成误导性均值：`snapshotdepth_hs.py:313-336`。

### Medium: `metrics.json` 中 final train loss 不可靠

主实验 `metrics.json` 中所有 `train_loss/*` 为 0，但 `implementation-notes.md` 又从 TensorBoard 事件文件提取了非零 train loss。说明当前 artifact 的最终训练指标不可靠，不能作为实验结论来源。

### High: 下一轮需要针对失败效果做代码级 repair，不应只诊断

60+ epoch 后 HS 与 depth 都接近失败，说明当前 DoDo decoder 训练设置可能存在可学习性问题。需要优先验证并修正以下高风险点：

- 小 batch 训练仍大量使用 `BatchNorm2d`。`models/simple_model_mamba.py:47-53` 的 input adapter 和 `nets/mamba_unet.py:9-15` 的 `DoubleConv` 都使用 BatchNorm；当前 batch size 1-2，这会使统计噪声大，容易输出均值化。
- DoDo 只有 3 通道测量输入，且没有持久化记录 `captimgs` 的 min/max/std/spectral sum。若测量动态范围过窄或尺度漂移，decoder 容易学成常数。
- depth 输出经 sigmoid，初始自然落在 0.5 附近；当前主训练最终 `est_depth_min == est_depth_max ~= 0.5038`，需要记录 depth logits/grad/std，并验证 depth head 是否收到有效梯度。
- `depth_smooth_weight` 默认 0.01。对已经常数化的 depth head，下一轮 one-batch overfit 应先关掉 smooth，避免正则继续奖励平滑常数解。

下一轮允许并要求在 baseline one-batch 失败后做最小代码修正，而不是只写“建议优化”。

## 下一轮必须做

1. 先补齐 diagnostics/artifact/logging，不要直接重跑 260 epoch。
2. 用 DOE-optimized preflight 验证：日志非空、DOE grad finite、param group membership、clamp hook、nonfinite counters、quicklook PNG、git status 均可追溯。
3. 运行 baseline one-batch overfit 80 epoch，因为当前深度输出常数化。
4. 若 baseline one-batch 不能明显 overfit，必须实施 targeted repair patch：DoDo decoder 小 batch norm 改为可选 GroupNorm、增加 DoDo measurement normalization CLI、记录 decoder/DOE/depth head 梯度与 captimgs 动态范围，并用 `--depth_smooth_weight 0` 做 repair overfit。
5. 若 repair one-batch 成功，再运行一个短 sanity train/val，不超过 20 epoch；若仍失败，停止并给出根因结论，不要继续 260 epoch。
