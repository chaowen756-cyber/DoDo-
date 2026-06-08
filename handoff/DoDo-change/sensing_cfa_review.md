# SensingLayer CFA/CCA Readiness — Static Review

Date: 2026-05-09

## Context

Reviewing whether `torch_optics/sensing.py:SensingLayer` can support future Gaussian CFA / learnable CFA integration without structural refactoring of the optical forward or loss pipeline.

## Current Code Structure (sensing.py, `forward()`)

```
1. x_abs = torch.abs(x).to(torch.float32)       ← wavelength-resolved intensity (B, 25, H, W)
2. if sensor_measurement == "intensity":
       x_abs = x_abs ** 2                       ← still (B, 25, H, W)
3. if sensing_mode == "rgb":
       r, g, b sensors: (25,) → broadcast        ← linear combination over lambda dim
       y_r = sum(x_abs * sensor_r, dim=1)        ← dim=1 collapse: 25 → scalar per pixel
       y = stack([y_r, y_g, y_b], dim=1)         ← (B, 3, H, W)
   else:  # spectral_bins or identity
       resp: (25, C)                             ← linear combination
       y = sum(x_abs_expanded * resp, dim=1)     ← dim=1 collapse: 25 → C per pixel
4. normalize (optional)
5. return y
```

## Review Questions

### A. 当前是否能直接加入 Gaussian CFA

**不能直接加入。** 当前的 RGB integration 在 `torch.sum(x_abs * sensor_{r,g,b}, dim=1)` 这一步立即将 25 个波长通道 collapse 成 3 个 RGB 通道。这个 collapse 发生在 SensingLayer 内部，调用方 (`DepthAwareDoDoForwardModel.forward()`) 无法访问 collapse 之前的 wavelength-resolved intensity。

如果要加入 Gaussian CFA（在 sensor plane 对每个像素应用不同的 wavelength-dependent transmittance pattern），需要在 `x_abs` (B, 25, H, W) 上进行 per-pixel wavelength filtering，而不是在 collapse 之后。

### B. 如果不能，需要重构 SensingLayer 的哪个位置

**需要重构的位置：`forward()` 第 104-116 行，即 `x_abs` 产生后、RGB/bins collapse 之前。**

具体来说，需要在以下两个步骤之间插入 CFA 层：

```python
# Line 100-102: wavelength-resolved intensity available here
x_abs = torch.abs(x).to(torch.float32)
if self.sensor_measurement == "intensity":
    x_abs = x_abs ** 2                         # ← (B, 25, H, W) — 完整波长信息

# ★ 此处应插入 CFA/CCA 编码 ★
# e.g., x_coded = self.cfa_layer(x_abs)        # (B, M_filter, H, W)

# Line 104-116: current collapse logic
if self.sensing_mode == "rgb":
    y = torch.stack([sum(x_abs * sensor_r, dim=1), ...], dim=1)  # ← collapse
else:
    y = torch.sum(x_abs_expanded * resp, dim=1)                  # ← collapse
```

重构方案建议：
- 新增 `cfa_mode` 参数（`none`, `gaussian`, `learnable`）
- 当 `cfa_mode != "none"` 时，在 collapse 之前执行 wavelength→filter 映射
- CFA 输出 `(B, M_filter, H, W)` 可直接作为测量传入 decoder，也可再经过 RGB rendering
- RGB sensing mode 作为 CFA 的一个特例（3 个固定 spectral filters）

### C. 当前 RGB sensing 是否已经过早丢失 wavelength-resolved intensity

**是的。** 在 `sensing_mode="rgb"` 下：

1. `torch.abs(x).to(torch.float32)` 产生 `(B, 25, H, W)` 的 wavelength-resolved intensity
2. 紧接着就在第 108-111 行通过 `sum(x_abs * sensor_{r,g,b}, dim=1)` 把 25→3 压缩了
3. 之后所有的操作都只在 3 个 RGB 通道上进行

这意味着 **decoder 接收到的测量已经丢失了波长维度的可分辨信息**。25 个波段的 HS 信息被不可逆地压缩为 3 个 RGB-like 通道。如果要实现 CFA/CCA（在 sensor 平面对不同像素编码不同波长响应），必须在 collapse 之前进行，而不是在 collapse 之后对 3 通道 RGB 做后处理。

对于 `sensing_mode="spectral_bins"` 和 `sensing_mode="identity"`，collapse 发生在 `torch.sum(x_abs_expanded * resp, dim=1)`。虽然输出通道数可以 >3，但仍然是通过线性组合（response matrix）一次性完成的，没有 per-pixel spatial variation——这与 CFA 的 per-pixel wavelength filtering 不同。

### D. 如果未来实现 CFA/CCA，建议新增哪些接口

建议在 `SensingLayer.__init__()` 中新增以下参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `cfa_mode` | `"none"` | `none` / `gaussian` / `learnable` |
| `cfa_channels` | `None` | CFA 输出通道数；`None` 时使用 `measurement_channels` |
| `cfa_gaussian_sigma` | `1.5` | Gaussian CFA 的 spectral bandwidth |
| `cfa_center_wavelengths` | `None` | 手动指定 CFA center wavelengths；`None` 时等距分布 |

在 `forward()` 中新增接口位置（collapse 之前）：

```python
# After x_abs computed, before rgb/spectral_bins/identity collapse
if self.cfa_mode != "none":
    x_abs = self._apply_cfa(x_abs)  # (B, 25, H, W) → (B, cfa_channels, H, W)
```

具体 CFA 实现：
- **Gaussian CFA**: 对每个 CFA channel k，用 `exp(-(lambda - lambda_k)^2 / (2*sigma^2))` 对 x_abs 的 wavelength 维度做加权求和
- **Learnable CFA**: 用可训练的 `nn.Parameter` 矩阵 `(cfa_channels, 25)` 替换固定 sensor response

### Summary

| Question | Answer |
|----------|--------|
| A. 能否直接加 Gaussian CFA | 不能。RGB collapse 在 SensingLayer 内部完成，无 wavelength-resolved 接口暴露 |
| B. 需要重构的位置 | `forward()` 第 100-116 行：`x_abs` 产生后、collapse 之前的区域 |
| C. RGB sensing 是否过早丢失波长信息 | 是的。25→3 的压缩发生在 SensingLayer forward 内部，不可逆 |
| D. 未来 CFA 建议接口 | `cfa_mode`, `cfa_channels`, `cfa_gaussian_sigma`, `cfa_center_wavelengths`；在 collapse 前插入 `_apply_cfa(x_abs)` |

### Recommendation

1. **在实现 CFA 之前**，先通过 `diagnose_measurement_spectral_capacity.py`（Diagnostics A-D）量化当前 3-channel measurement 的 spectral capacity loss，为 CFA channel count 决策提供数据支撑。
2. **CFA 实现应在 SensingLayer 内部**，不修改 optical forward (`forward_dodo.py`) 或 decoder。
3. **保留 RGB mode 作为 `cfa_mode="none"` 的特例**，确保向后兼容。
4. **Learnable CFA 的权重矩阵 shape 为 `(cfa_channels, 25)`**，可与 DOE zernike coefficients 一起优化。
