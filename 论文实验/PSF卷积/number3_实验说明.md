# PSF 卷积 number3 集中改进实验

执行：

```bash
cd /home/wenchao/autodl-tmp
bash 论文实验/PSF卷积/run_joint_training_gpu01.sh
```

脚本自动执行 Stage A 20 epoch、Stage A 场景 14–18 推理、Stage B
30 epoch、Stage B 场景 14–18 推理，使用物理 GPU 0、1。终端输出通过
`tee` 实时显示，同时保存在 `论文实验/PSF卷积/pipeline_logs/`。

## number3 相对 number2 的集中修改

- 使用 halo=64：256×256 光学上下文卷积后只保留中心 128×128；训练索引
  仅重新计算 `scale_05_eligible`，保留行的其他字段和值不变。
- r16/r24 多尺度 encircled-energy 约束，预算由 0.35/0.15 逐步收紧到
  0.20/0.05，并约束最差 10% PSF。
- 加入低中频 MTF floor、相邻及隔一波长 hard-negative 可分性、RGB
  有效相邻深度可分性。
- MTF loss 在真实 16×25 PSF 上完成量纲校准，权重为 0.25；初始加权
  贡献约 `9.8e-4`，避免沿用 0.005 时只有约 `2e-5` 而形同虚设。
- 把 number18e 旧 12 项 DOE 的实际波面最小二乘投影到 150 项基底；
  前 15 项先训练，高阶项第 5 个 epoch 后以 0.2 倍梯度释放。
- 高光谱损失为 `L1 + 0.5*MSE + 0.02*SAM + 0.05*spatial-gradient`，
  修复了旧代码中 SAM 权重被固定为 0 的问题。
- Stage B 不再隔离 HS 到共享编码器的梯度，并把 CNN 学习率降至 `5e-5`。
- 每轮记录 r16/r24 能量、r50/r80/r90、MTF、波长/深度 cosine 和
  Zernike 高低阶范数。Stage A 后自动打印物理门槛；默认只告警并继续
  一次性流程，设置 `ENFORCE_PHYSICAL_GATE=1` 可在未达标时终止。

halo=64 增大了光学上下文，单卡 batch 从 16 改为 8，并使用
`accumulate_grad_batches=2` 保持有效 batch 不变。
