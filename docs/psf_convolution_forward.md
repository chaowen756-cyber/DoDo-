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

## 4. 卷积边界

默认 `--dodo_psf_boundary linear_zero`：

- FFT 尺寸为 `H+Kh-1, W+Kw-1`；
- 执行线性卷积；
- 按 PSF 中心裁回 `H,W`；
- 不产生 circular FFT 的跨边界环绕。

`circular` 仅保留作诊断和速度对照，不建议作为正式物理模型。

## 5. PSF 缓存与梯度

- DOE 可训练时，PSF 每个 forward 重新生成，梯度从 loss 经卷积、PSF 强度和传播链回到 `doe1.zernike_coeffs`。
- DOE 冻结时，PSF bank 自动缓存，用于 Stage B 和推理。
- 缓存不是 persistent buffer，不写入 checkpoint；`clamp_parameters_()` 会主动清空缓存。

## 6. 启动参数

正式 PSF 卷积实验至少需要：

```text
--optical_model dodo_depth
--dodo_image_formation psf_convolution
--depth_layering_mode soft_diopter
--dodo_psf_layer_mask baek_hard
--dodo_psf_mask_blur_sigma 1.0
--dodo_psf_boundary linear_zero
--dodo_sensor_measurement intensity
```

可直接使用：

```bash
bash scripts/run_number18_baek_balanced.sh stage-a-combined
```

新实验目录统一带 `psfconv_` 前缀，避免覆盖原 Number18 结果。

## 7. 归一化注意事项

旧 whole-field forward 的 `dodo_forward_scale` 不应复用于 PSF 卷积。建议先使用：

```text
--dodo_forward_norm none
--dodo_measurement_norm none
```

统计新测量分布后再标定新的 fixed scale。旧 decoder checkpoint 虽然结构上可加载，但其训练测量分布不同，正式 PSF 卷积实验应重新训练 decoder。

## 8. 验证

自动化测试位于 `test/test_dodo_psf_convolution.py`，覆盖：

- PSF 非负、有限和单位能量；
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
