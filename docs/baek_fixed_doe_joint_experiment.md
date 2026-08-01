# Baek 预训练 DOE 冻结联合训练实验

## 实验问题

直接加载 `e2e_HSD_doe_height.pth` 中已经联合优化的物理高度，保持其训练时的
DOE 采样与传播约定，冻结全部光学参数，只联合训练当前 HS/深度重建网络，观察
预训练 DOE 的编码能力能否迁移。

## DOE 优先的光学口径

新模式 `doe_native_grid_v1` 只作用于本实验；原有 `legacy` 和
`consistent_grid_v1` 数值路径不变。

- 高度：源 tensor `375x375`，严格按 notebook 在右侧和下侧补零到
  `376x376`；单位为米，不缩放、不插值、不重新包裹。
- DOE 网格：`376x376`、8 µm pitch、3.008 mm 全宽。
- 材料：NOA61；当前实现与 PADO 使用同一 Cauchy 色散公式。
- 入射场：在 DOE 面直接按 PADO `set_spherical_light` 生成当前深度中心的球面波，
  不再用离散场景点经 Prop1 近似。
- 光阑：使用 PADO 的整数中心、直径 3.008 mm 圆孔约定。
- DOE 后传播：PADO-compatible Fresnel linear convolution，传播 50 mm；输入对称
  补零到 `752x752` 后回裁到 `376x376`。
- PSF：取中心 `129x129` 供当前 halo64 linear-zero 场景卷积。记录该窗口相对
  `376x376` PADO 输出的能量占比，并把有限卷积核校准为单位和，避免整体曝光
  随裁剪比例漂移。
- 波长：420--660 nm，共 25 个波段。

当前场景 PSF 卷积、RGB sensing、重建网络、损失和训练数据口径均不变。深度仍
使用当前实验的 0.4--2.0 m、16 个 inverse-depth 中心；这不是逐项复现 Baek 的
20 个 0.3--2.0 m 可视化深度，而是把 DOE 放入当前任务设置中做受控迁移实验。

## 固定实验设置

- DOE：`fixed_height` buffer，无可训练 Parameter。
- 光学：`--no-optimize_optics`；误开优化会立即拒绝启动。
- 无第二 DOE，跳过当前 Prop2，光学正则全部为 0。
- 网络：HS loss 与 depth loss 联合反向，重建网络从头初始化。
- 训练：12 epoch，双卡 FP32，batch 8、梯度累积 2。

## 运行

默认使用物理 GPU 1 和 3：

```bash
cd /home/wenchao/autodl-tmp
bash scripts/run_baek_fixed_doe_joint.sh
```

只展开并检查完整命令：

```bash
cd /home/wenchao/autodl-tmp
DRY_RUN=1 bash scripts/run_baek_fixed_doe_joint.sh
```

覆盖 GPU 或实验名：

```bash
cd /home/wenchao/autodl-tmp
CUDA_DEVICES=0,2 \
EXPERIMENT_NAME=psfconv_baek_native_doe_frozen_joint_seed123 \
bash scripts/run_baek_fixed_doe_joint.sh
```

脚本拒绝覆盖已有 `artifacts/command.txt`；重跑时应设置新的实验名。

## 真实文件预检

源文件 SHA256：
`6d16ac1f32f6be4e7487110527161d87480feaf7dd2b31bbeee20e6bab1e9662`。

- height shape：`376x376`
- height min/mean/std/max：
  `0 / 3.875903e-7 / 3.381788e-7 / 1.023254e-6 m`
- 光学可训练参数：0
- PSF：`16x25x129x129`，全部 finite，每个有限核校准后求和约为 1
- 裁剪前能量占比 min/mean/max：
  `0.237906 / 0.361023 / 0.442250`

较低的中心窗口能量占比不是传播失败：Baek DOE 产生的是用于光谱/深度编码的扩展
PSF，源 notebook 本身只展示中心 `96x96`。本实验保留当前 halo64 所允许的
`129x129` 卷积核并显式记录这项截断，后续若需要可把更大 halo 作为独立消融，
不应和首次 DOE 迁移实验混在一起。

## 结果解释

主要对照应使用相同网络、数据、16-depth 设置和训练策略，只替换 DOE/对应物理
传播。若效果仍差，优先检查有限 PSF 支撑范围和“源 DOE 的联合训练网络与当前
网络不同”这两个迁移差异；不能仅据此否定高度图本身的可编码性。
