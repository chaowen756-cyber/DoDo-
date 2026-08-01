# Baek 固定 DOE 接入后的 PSF 生成前向架构

## 先回答核心问题

是的，在相同的波长 `lambda`、物距 `z`、376x376/8 µm 网格和相同裁剪口径下，
只要把 DOE 前的球面波生成改成与 Baek/PADO 完全一致，当前模型生成的 PSF 就会
非常接近 `e2e_HSD.ipynb` 的结果。

这里的“第一段传播”需要准确理解为：

- 旧前向：在场景平面放一个离散 impulse，再通过 `Prop1` 传播到 DOE。
- Baek 前向：不执行这次离散传播，而是在 DOE 平面直接调用
  `Light.set_spherical_light(z)` 构造点光源球面波。
- 当前 `doe_native_grid_v1`：结构上已经绕过 `Prop1`，直接在 DOE 面生成球面波；
  但当前 25 波长 float32 向量化除法的数值语义还没有与 PADO 的逐波长 Python
  float 完全对齐。这是当前唯一需要修正的主要差异。

## 一致性实验到底比较了什么

实验包含三条前向，而不是只比较两张展示图。

| 路径 | DOE 前球面波 | DOE、圆孔、50 mm 传播 | 用途 |
|---|---|---|---|
| A. Baek/PADO reference | PADO 每次单独计算一个 `wvl_val` | PADO 原生 API | 参考答案 |
| B. 当前 native | 25 波长 float32 tensor 一次向量化 | 当前 PADO-compatible 实现 | 检查当前代码整体结果 |
| C. 根因隔离 | PADO 逐波长标量语义 | 与 B 完全相同 | 判断差异是否只来自球面波 |

结果为：

- B 对 A：full-grid NRMSE mean `0.150379`，max `1.106150`。
- C 对 A：full-grid NRMSE mean `4.79e-6`，max `1.62e-5`；cosine 基本为 1。

因此可以得出严格结论：

> 当前 DOE 高度、NOA61 相位、PADO 圆孔和 DOE 后 50 mm Fresnel 传播已经正确；
> 只要修正 DOE 前球面波的数值计算，当前前向在对应 `(lambda, z)` 上就能复现
> Baek notebook PSF。

## 修正后的 PSF 生成总架构

```text
波长 λ [25]
物距 z [D]
    │
    ▼
在 DOE 平面直接生成 PADO 球面波
U_source(z, λ, x, y) = exp(i · 2π · sqrt(x²+y²+z²) / λ)
单位振幅；不经过旧 Prop1
tensor: [D, 25, 376, 376], complex64
    │
    ▼
固定 Baek DOE 高度
原始 375×375 ──右/下补零──> 376×376
高度单位 m，不缩放、不插值、无可训练参数
    │
    ▼
NOA61 DOE 相位
U_doe = U_source · exp(i · 2π(n(λ)-1)h/λ)
    │
    ▼
PADO integer-centered circular aperture
376×376，pitch=8 µm，diameter=3.008 mm
    │
    ▼
跳过 Prop2；不使用第二片 DOE
    │
    ▼
PADO-compatible Fresnel linear convolution
传播距离 f=50 mm
376×376 输入 ──对称补零──> 752×752
与正相位 Fresnel impulse response 做 FFT 线性卷积
中心回裁为 376×376 complex field
    │
    ▼
传感器强度
I(z, λ) = |U_sensor|²
tensor: [D, 25, 376, 376], float32
    │
    ▼
完整 376×376 PSF 按能量归一化
每个 (z, λ) 的 full-grid sum = 1
    │
    ├── Baek notebook 展示：中心 96×96
    │
    └── 当前场景卷积：中心 129×129
          记录该 crop 的 full-grid capture fraction
          再将 129×129 有限核归一化到 sum = 1
          tensor: [D, 25, 129, 129]
```

## 每一层的物理与张量定义

### 1. 波长与深度

波长与 notebook 相同：

```text
λ = linspace(420 nm, 660 nm, 25)
```

需要区分两套深度配置：

```text
Baek notebook 对比：20 层，0.3–2.0 m，inverse-depth 均匀采样
当前训练脚本：       16 层，0.4–2.0 m，inverse-depth 均匀采样
```

这不影响同一个 `(lambda, z)` 下的 PSF 一致性，但两套完整 PSF bank 的深度数量和
深度中心并不相同。若要逐格复现 notebook 的 25x20 展示，必须使用 notebook 的
20 个深度；若接入当前训练，则保留当前任务的 16 个深度中心。

### 2. DOE 面球面波

PADO 使用：

```text
x,y = arange(-188, 188) × 8 µm
r    = sqrt(x² + y² + z²)
U    = exp(i × ((2πr/λ) mod 2π))
```

振幅设为 1，没有额外的 `1/r` 衰减。

为了数值上严格对齐 PADO，波长需要使用逐波长 Python float 语义，示意为：

```python
bands = []
for wavelength in wavelengths.tolist():
    phase = (2 * torch.pi * radius / float(wavelength)) % (2 * torch.pi)
    bands.append(torch.exp(1j * phase))
field = torch.stack(bands, dim=0)
```

不能直接把数百万至数千万弧度的 `2πr/λ` 用一个 float32 wavelength tensor
向量化后再取模；虽然数学公式相同，但 float32 舍入会改变取模后的空间相位。

### 3. 固定 DOE

高度文件：

```text
e2e_HSD_learned_DOE_and_PSF_simulation/e2e_HSD_doe_height.pth
```

处理方式严格复现 notebook：

```text
[1,1,375,375] --F.pad(0,1,0,1)--> [1,1,376,376]
```

DOE 相位为：

```text
phi_doe(λ,x,y) = 2π(n_NOA61(λ)-1)h(x,y)/λ
```

高度注册为 buffer，不是 `Parameter`，不会进入 optimizer。

### 4. 圆孔

采用 PADO 的整数中心坐标：

```text
x,y = -188,...,187
mask = x²+y² <= 188²
有效像素数 = 111007
```

### 5. DOE 后 Fresnel 传播

当前 `PadoFresnelPropagationLayer` 对齐 PADO `Fresnel, linear=True`：

```text
h_f(x,y,λ) = exp(i·k(x²+y²)/(2f)) / (fλ)
f = 50 mm
```

数值流程：

```text
376×376 field
  -> 四边各补 188 pixel
  -> 752×752 centered FFT
  -> 乘 Fresnel kernel FFT
  -> centered IFFT
  -> 中心裁回 376×376
```

### 6. 强度与有限 PSF 核

完整传感器强度：

```text
I = abs(U_sensor)²
I_full = I / sum(I over 376×376)
```

Baek notebook 只展示中心 96×96；当前场景卷积使用中心 129×129，因为 halo64
能够无截断容纳的最大奇数核为 `2×64+1=129`。

当前有限核处理：

```text
capture_fraction = sum(center_crop_129(I_full))
PSF_129 = center_crop_129(I_full) / capture_fraction
```

## PSF bank 如何进入当前图像形成模型

PSF 生成完成后，场景侧仍保持当前 PSF 卷积架构，不需要改成 376×376 场景图：

```text
HS scene [B,25,H,W]
depth map [B,1,H,W]
       │
       ├──> inverse-depth layer weights [B,D,H,W]
       │
       └──> 每个深度层使用 PSF_bank[d, λ, 129,129]
                 │
                 ▼
          linear-zero spectral convolution
                 │
                 ▼
          RGB spectral response collapse
                 │
                 ▼
          captured RGB [B,3,H,W]
```

也就是说，376×376 是“离线/缓存 PSF 生成的物理网格”，129×129 是“进入场景卷积
的有限核”，场景 patch 仍然可以是当前 128 target + halo64 的 256×256 输入。

## 当前代码位置

| 功能 | 文件/入口 |
|---|---|
| native 模式配置 | `torch_optics/forward_dodo.py::DepthAwareDoDoForwardModel` |
| DOE 面球面波 | `torch_optics/forward_dodo.py::_prop1_impulse_field_bank` |
| 固定高度 DOE | `torch_optics/doe.py::DOEFixedHeightLayer` |
| PADO Fresnel | `torch_optics/propagation.py::PadoFresnelPropagationLayer` |
| PSF 强度、裁剪和归一化 | `torch_optics/forward_dodo.py::_generate_psf_bank` |
| 独立 PADO 一致性实验 | `scripts/compare_baek_native_psfs.py` |

## 当前完成状态

| 项目 | 状态 |
|---|---|
| Baek 375→376 高度读取 | 已完成 |
| 8 µm / 376 网格 | 已完成 |
| NOA61 相位 | 已完成 |
| PADO integer-centered pupil | 已完成 |
| 50 mm PADO Fresnel linear convolution | 已完成 |
| 结构上绕过旧 Prop1、在 DOE 面直接生成球面波 | 已完成 |
| 球面波逐波长标量数值语义 | **尚待修正** |
| 25x20 PADO parity 脚本与基线 | 已完成 |
| 正式网络训练 | 未启动；应等待 parity 修正通过 |

## 修正后的验收标准

修改只应作用于 `doe_native_grid_v1` 的球面波生成，不改变其他光学模式。重新对比
全部 500 个 `(lambda,z)` 后，目标为：

```text
full-grid cosine 接近 1
mean NRMSE ≈ 5e-6
max NRMSE <= 2e-5
```

达到该标准后，可以认为当前固定 Baek DOE 的 PSF 生成前向已经在数值上复现
notebook，再决定是否将这些 PSF 用于当前 16-depth 重建网络实验。
