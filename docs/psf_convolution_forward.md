# Baek 式深度-波长 PSF 卷积前向

## 1. 目标与兼容边界

`dodo_depth` 现在支持两种成像模型：

- `whole_field`：原有实现。每个深度层的整幅高光谱图作为复场传播。
- `psf_convolution`：先生成每个深度和波长的强度 PSF，再按 Baek et al. 的分层卷积模型合成传感器图像。

默认值仍是 `whole_field`，所以旧 checkpoint 中缺少新参数时不会静默改变成像模型。当前分支的 Number18 启动脚本显式使用 `psf_convolution`。

以下接口保持不变：

- 输入高光谱：`[B, 25, 128, 128]`
- 输入米制深度：`[B, 1, 128, 128]`
- 输入有效 mask：`[B, 1, 128, 128]`
- RGB 测量输出：`[B, 3, 128, 128]`
- DOE 参数和 state-dict 键
- measurement normalization、noise、decoder、loss、训练和推理入口

## 2. 成像公式

对深度层 `k` 和波长 `lambda`，使用中心单位点源经过当前 DoDo 光路：

```text
delta -> prop1[k] -> doe1 -> optional prop2 -> optional doe2 -> prop3
```

强度 PSF 定义为：

```text
P[k, lambda] = abs(U[k, lambda]) ** 2
P[k, lambda] /= sum_xy(P[k, lambda])
```

Baek 式传感器图像为（见 [Baek et al., ICCV 2021, 式 (3)](https://openaccess.thecvf.com/content/ICCV2021/html/Baek_Single-Shot_Hyperspectral-Depth_Imaging_With_Learned_Diffractive_Optics_ICCV_2021_paper.html)）：

```text
J[c] = sum_lambda response[c, lambda]
       * sum_k M[k] * convolution(I[lambda], P[k, lambda]) + noise
```

这里的 `I` 是输入的高光谱辐亮度/光强，`P` 是归一化强度 PSF，
所以 PSF 路径直接做强度域线性卷积，不对场景 `I` 开平方。开平方
`sqrt(I * depth_weight)` 只属于 `whole_field` 相干整场传播的
“辐亮度转场振幅”步骤；若在 PSF 路径重复开平方，会破坏
`J(aI_1+bI_2)=aJ(I_1)+bJ(I_2)` 的非相干成像线性。

深度 mask 在卷积之后相乘，与 Baek et al. 主论文式 (3) 一致。实现先在频域把波长维按传感器响应压缩成 3 个 RGB 通道，再执行逆 FFT，从而避免构造 `[B,K,25,H,W]` 中间张量。

## 3. 深度 mask

`--dodo_psf_layer_mask baek_hard` 是当前分支的实验默认值：

1. 对 `soft_diopter` 中心，在逆深度空间选择最近层；
2. 生成二值 occupancy；
3. 使用 `--dodo_psf_mask_blur_sigma` 指定的 Gaussian 模糊；
4. 重新应用有效区域 mask；
5. 沿深度维归一化，使每个有效像素的权重和为 1。

`--dodo_psf_layer_mask current` 可复用原有 soft/hard depth weights，主要用于只比较成像算子的受控消融。

当前 Number18 配置仍使用 16 个逆深度均匀层，以保持 decoder 深度概率通道不变。若要复现论文的 7 个深度层，应作为单独实验设置 `dodo_depth_layers=7` 和 `n_depths=7`。

## 4. Prop1 等采样 padding

PSF 的中心点源传播可通过：

```text
--dodo_prop1_padding_factor 2
```

把每个 `prop1[k]` 的计算网格从 `128 / 0.01 m` 扩为
`256 / 0.02 m`，传播后中心裁回 128。采样数 \(N\) 与物理窗 \(L\)
同倍扩大，因此像素间距 \(\Delta x=L/N\) 不变，同时更大的计算窗会
抑制 Prop1 远场传播的周期环绕混叠。Prop2/Prop3 不做这项 padding。

核心接口、旧 checkpoint 和通用 PSF 训练脚本均默认 factor=1，以保持
原光学前向。新 padding 实验必须显式设置 factor=2，并确保 Stage A 与
Stage B 使用同一值。

### 4.1 原始 12 项 DOE 的 orthogonal RMS 模式

`legacy12` DOE 可选用：

```text
--dodo_zernike_mode legacy12
--dodo_doe_basis_mode orthogonal_rms
--dodo_doe_basis_rank 9
--dodo_doe_basis_rank_rtol 1e-4
--dodo_doe_basis_rms_m 3e-6
--dodo_doe_coeff_norm_limit 1.0
--dodo_doe_init_coeff_norm 1.0
```

该模式在 pupil 内对原始 12 项做两遍 Gram-Schmidt，并按相对残差
阈值去除近共线方向。默认保留原始索引
`[0,1,2,3,4,5,6,10,11]` 的 9 个有效模式。每个模式的 pupil
高度 RMS 都是 3 微米，因此总高度 RMS 为
`3 微米 * ||coeff||2`，系数向量用 L2 球约束，而不是逐项裁剪。

默认仍为 `legacy_raw12`，所以旧 12 维 checkpoint 和现有 free150
实验不变。orthogonal 9 维 checkpoint 不能直接与 raw12 checkpoint
互载，Stage A/B 必须使用相同的 mode 和 rank。free150 的高阶解锁、
高阶正则与诊断也只在 `--dodo_zernike_mode free` 下启用。

## 5. 卷积边界

默认 `--dodo_psf_boundary linear_zero`：

- FFT 尺寸为 `H+Kh-1, W+Kw-1`；
- 执行线性卷积；
- 按 PSF 中心裁回 `H,W`；
- 不产生 circular FFT 的跨边界环绕。

`circular` 仅保留作诊断和速度对照，不建议作为正式物理模型。

## 6. PSF 缓存与梯度

- DOE 可训练时，PSF 每个 forward 重新生成，梯度从 loss 经卷积、PSF 强度和传播链回到 `doe1.zernike_coeffs`。
- DOE 冻结时，PSF bank 自动缓存，用于 Stage B 和推理。
- Stage A 验证处于无梯度模式，DOE 在整段 validation loop 中不更新，因此同一轮验证只生成一次 detached PSF bank；恢复训练后仍重新生成 live autograd PSF。
- 固定中心点源经过各深度 Prop1 的复场独立缓存。Prop1 距离、波长和网格均冻结，这部分不需要梯度；缓存之后仍批量执行可训练 DOE、Prop3 和 PSF 归一化，DOE 梯度保持不变。
- 缓存不是 persistent buffer，不写入 checkpoint；`clamp_parameters_()` 会主动清空缓存。
- 固定传播距离的 padded Fresnel kernel 也按进程惰性缓存，但不注册为 buffer、不写入 checkpoint，并在设备迁移或载入 state dict 时失效重建。

线性卷积仍使用零填充 FFT，但工作尺寸会向上取最近的 5-smooth
`next-fast` 长度。例如 halo64 的 `256 + 128 - 1 = 383` 会使用
`384 x 384` FFT；halo0 的 `255 x 255` 会使用 `256 x 256` FFT。
只要 FFT 尺寸不小于完整线性卷积支撑，中心裁剪结果在数学上等价，
差异仅来自浮点 FFT 的舍入顺序。

## 7. 启动参数

正式 PSF 卷积实验至少需要：

```text
--optical_model dodo_depth
--dodo_image_formation psf_convolution
--depth_layering_mode soft_diopter
--dodo_psf_layer_mask baek_hard
--dodo_psf_mask_blur_sigma 1.0
--dodo_psf_boundary linear_zero
--dodo_prop1_padding_factor 2
--dodo_sensor_measurement intensity
```

可直接使用：

```bash
DODO_PROP1_PADDING_FACTOR=2 \
  bash scripts/run_number18_baek_balanced.sh stage-a-combined
```

若要运行 12 项来源的 orthogonal RMS + padding PSF 实验：

```bash
DODO_ZERNIKE_MODE=legacy12 \
DODO_DOE_BASIS_MODE=orthogonal_rms \
DODO_PROP1_PADDING_FACTOR=2 \
  bash scripts/run_number18_baek_balanced.sh stage-a-combined
```

新实验目录统一带 `psfconv_` 前缀，避免覆盖原 Number18 结果。

## 8. 归一化注意事项

旧 whole-field forward 的 `dodo_forward_scale` 不应复用于 PSF 卷积。建议先使用：

```text
--dodo_forward_norm none
--dodo_measurement_norm none
```

统计新测量分布后再标定新的 fixed scale。旧 decoder checkpoint 虽然结构上可加载，但其训练测量分布不同，正式 PSF 卷积实验应重新训练 decoder。

## 9. 验证

自动化测试位于 `test/test_dodo_psf_convolution.py`、
`test/test_doe_orthogonal_rms.py`、
`test/test_dodo_radiance_field.py` 和
`test/test_propagation_padding.py`，覆盖：

- 原始 12 项正交化后的有效 rank、等 RMS Gram 矩阵和 L2 约束；
- legacy raw12 checkpoint 严格加载兼容；
- PSF 非负、有限和单位能量；
- factor=1 与旧传播公式完全一致；
- factor=2 保持采样间距并抑制远场周期环绕；
- 固定距离 Fresnel 核复用与可训练距离禁用缓存；
- 383→384 next-fast FFT 与最小长度线性卷积的输出等价；
- 批量 PSF 生成与旧逐深度实现的输出、DOE 梯度等价；
- Stage A 验证 PSF 复用、DOE 更新失效与训练 live graph 隔离；
- Prop1 点源场缓存不进入 checkpoint，并在设备迁移或加载后失效；
- padded Prop1 生成的 PSF 仍非负、有限、单位能量；
- 冻结光学缓存；
- Gaussian depth occupancy 归一化；
- FFT 中心对齐；
- 中心点源与 PSF 一致；
- 非相干前向线性可加性；
- Baek 式“先卷积、后乘深度 mask”顺序；
- RGB sensor response 等价性；
- 旧 whole-field 分发回归；
- DOE 梯度非零且有限；
- 调试统计和输出形状。
