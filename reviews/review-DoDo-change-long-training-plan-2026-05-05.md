# Review: DoDo-change long-training execution plan

## 结论

DoDo 光学模型已经接入端到端训练链路，下一步应进入真实训练实验。但用户明确要求本轮训练必须开启嵌入 DOE 优化，即 `--optimize_optics`，同时训练 DoDo DOE 与重建网络；训练轮数先设为 260 epoch。

由于训练可能持续数小时，Claude 不应持续读取训练终端输出。所有训练 stdout/stderr 必须重定向到日志文件，Claude 只需要：

- 启动训练；
- 记录 PID、命令、日志路径；
- 低频轮询进程是否结束；
- 训练结束后读取日志尾部、metrics、checkpoint 和保存图像进行分析。

## 相对上一版计划的调整

- 取消“正式 baseline 先冻结光学”的默认策略；保留冻结光学只作为失败诊断选项。
- 正式主实验使用 `--optimize_optics`。
- 推荐学习率：`--optics_lr 1e-6`，`--cnn_lr 5e-5`。
- 正式主实验 epoch：`--max_epochs 260`。
- 仍必须先跑短 preflight，验证 metric/artifact/log redirection/checkpoint 都正常。

## 仍需 Claude 先补齐

- `validation/psnr_hs_masked`
- `validation/mae_depth_m`
- 验证结果图片和 metrics artifact 保存
- DoDo measurement 非有限防护
- `optimize_optics=True` 后 `clamp_parameters_()` hook
- checkpoint monitor 推荐改为 `validation/psnr_hs_masked` / `max`

## 长训练执行要求

训练命令必须类似：

```bash
mkdir -p infer_results/DoDo-change/<exp>/logs
stdbuf -oL -eL python snapshotdepth_trainer_hs.py ... \
  > infer_results/DoDo-change/<exp>/logs/train.log 2>&1
```

Claude 不要在训练期间反复 `tail` 大日志。允许：

- 启动后确认日志文件创建；
- 每 20-30 分钟仅用 `ps -p <PID>` 或等价方式检查进程是否仍在运行；
- 若必须检查健康状态，只读取最后 50-100 行；
- 训练结束后再读取日志摘要和 metrics。

## 下一步建议

Claude 应按 `handoff/DoDo-change/TASK.md` 执行：

1. 完成训练评估基础设施。
2. 跑带日志重定向的 preflight。
3. 跑 260 epoch 的 `dodo_depth` + DOE 优化主实验。
4. 等待训练结束。
5. 分析 PSNR、米制 depth MAE、保存图像和训练日志。
6. 若主实验失败或指标明显异常，只做一个针对性修复/诊断实验，然后停止等待 Codex review。
