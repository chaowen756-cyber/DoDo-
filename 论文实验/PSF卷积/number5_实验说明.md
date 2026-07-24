# Number5：延迟、限幅的 PSF 正则联合训练

Number5 根据 Number2 与 Number4 的直接对比修正联合优化。它保留
Zernike-150、halo=64、Baek 式线性零边界 PSF 卷积和 Number18
combined 数据方案，但撤回 Number4 中同时引入的激进优化设置。

## 训练基线

- Stage A：DOE 学习率恒定 `1e-5`，CNN 学习率恒定 `1e-4`，20 epochs。
- Stage B：固定 DOE，CNN 初始学习率 `5e-5`，每 20 epochs 衰减 0.1，
  共 30 epochs。
- 不加载整场传播 Number18e 的 DOE；150 个自由 Zernike 系数按
  PSF 卷积模型自身的默认值开始优化。
- 高频 Zernike 从 epoch 0 正常参与优化，不再额外锁定或施加 L2。
- 光谱重建损失恢复为 Number2 的 L1 主损失；关闭 Number4 新增的
  SAM、MSE 和空间梯度项，避免一次实验中同时改变过多目标。

## PSF 正则调度

- epoch 0–5：只由重建和深度任务塑造 DOE，不施加 PSF 正则。
- 能量集中：epoch 5–10 将权重从 0 线性升至 `0.02`。
- 能量预算从 epoch 5 开始，用 14 epochs 将半径 16 外能量预算
  从 `0.70` 收紧到 `0.60`，半径 24 外预算从 `0.45` 收紧到 `0.35`。
- CVaR 最差样本权重从 Number4 的 `0.5` 降至 `0.1`。
- 能量预算违反项由平方 hinge 改为线性 hinge（仅 Number5 配置启用）。
  这样接近预算边界时梯度不会快速衰减，再由 15% 总贡献上限防止其
  压过任务损失。
- 波长、深度分离从 epoch 10 开始，用 5 epochs 分别升至
  `0.005` 和 `0.002`；margin 分别为 `0.95`、`0.97`，只重点处理
  最相似的困难 PSF 对。
- 关闭独立 MTF loss。能量集中与任务损失已经约束空间传递，
  Number4 中 MTF 梯度与分离梯度存在冲突。
- 所有 DOE 正则的实际加权和最多为当前任务损失的 `15%`。缩放系数
  detach，不会通过抬高任务损失来放宽上限。

训练日志新增以下字段，用来区分“代码参与计算”和“物理指标改善”：

- `train_loss/psf_loss_effective_weight`
- `train_loss/optical_regularizer_raw`
- `train_loss/optical_regularizer_scale`
- `train_loss/optical_regularizer_weighted`
- `train_loss/optical_regularizer_ratio`

## 物理失效门控

Stage A 推理 14–18 场景后，脚本检查最终 PSF 能量、深度/波长余弦
相似度、正则实际权重和正则/任务损失比例。默认
`ENFORCE_PHYSICAL_GATE=1`：若正则仍没有带来要求的物理结果，脚本
保留 Stage A 训练和推理结果，但停止 Stage B，避免继续消耗算力。

运行：

```bash
bash /home/wenchao/autodl-tmp/论文实验/PSF卷积/run_joint_training_gpu23.sh
```
