# DOE PSF Preoptimization Task

## 目标

在 `psf卷积` 的 `consistent_grid_v1` 前向上实现不依赖重建网络的 DOE 可编码性
预优化，对比当前 rank-9 与 free-150 Zernike 在相同 3 μm pupil-RMS 预算下可
达到的 MTF、波长分离和深度分离能力。

## 范围

- 新增独立 CLI，不修改联合训练默认行为。
- 复用现有 PSF MTF、传感器加权光谱/深度分离和能量损失。
- 保存可复用 DOE checkpoint、逐步指标和可视化。
- 为不同 DOE 基底增加统一物理高度 RMS 投影。

## 非目标

- 不修改 Prop1、Prop3、10 mm 一致网格、pupil 或 PSF 卷积。
- 不引入 50 mm/6.45 μm scaled Fresnel。
- 不加载数据集、网络或旧联合训练 checkpoint。
- 不在本 change 中自动启动长时 GPU 实验。

## 验收

- rank-9/free-150 均可稳定优化并输出相同格式 artifact。
- 两者初始化和最大高度 RMS 预算一致。
- DOE 梯度、目标函数、checkpoint 恢复和 CPU smoke 通过测试。
- 现有 DOE/PSF 测试不回归。
