# PSF 卷积 number4 学习率实验

执行：

```bash
cd /home/wenchao/autodl-tmp
bash 论文实验/PSF卷积/run_joint_training_gpu01.sh
```

脚本使用物理 GPU 2、3，自动执行 Stage A 20 epoch、Stage A 场景
14–18 推理、Stage B 30 epoch、Stage B 场景 14–18 推理。

## number4 相对 number3 的修改

- Stage A DOE 和 CNN 的初始学习率均为 `1e-4`。
- Stage A 使用 Baek 阶梯衰减，DOE 和 CNN 每 10 epoch 乘 `0.1`：
  epoch 0–9 为 `1e-4`，epoch 10–19 为 `1e-5`。
- Stage B DOE 按阶段定义保持冻结，学习率为 `0`；CNN 初始学习率改为
  `1e-4`，每 10 epoch 乘 `0.1`：epoch 0–9 为 `1e-4`，
  epoch 10–19 为 `1e-5`，epoch 20–29 为 `1e-6`。
- Stage A、Stage B 均保留前 54 个优化 step 的线性 warmup。
- 除学习率和衰减配置外，number3 的 Zernike-150、halo=64、PSF
  多物理正则、损失组成、数据索引和训练/推理流程保持不变。
