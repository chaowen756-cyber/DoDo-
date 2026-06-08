# SensingLayer 最小重构方案：暴露 wavelength-resolved intensity

Date: 2026-05-09

## 1. 当前 `x_abs` 的 shape 和语义

位置：`torch_optics/sensing.py`, `forward()`, 第 100-102 行。

```python
x_abs = torch.abs(x).to(torch.float32)          # (B, 25, H, W)
if self.sensor_measurement == "intensity":
    x_abs = x_abs ** 2                           # (B, 25, H, W)
```

**语义**：wavelength-resolved intensity。每个空间位置 (h,w) 有 25 个值，分别对应 420nm~660nm 均匀采样的 25 个波段的传感器平面复振幅的幅度（amplitude mode）或强度（intensity mode）。

**关键事实**：这是整个 forward pipeline 中唯一一个 `(B, 25, H, W)` 的 wavelength-resolved 张量。之后的所有操作都在 collapse 之后的 measurement 通道上进行。

## 2. RGB collapse 发生在哪一步

RGB collapse 发生在 `forward()` 第 104-111 行：

```python
if self.sensing_mode == "rgb":
    r = self.sensor_r[None, :, None, None]       # (1, 25, 1, 1)
    g = self.sensor_g[None, :, None, None]
    b = self.sensor_b[None, :, None, None]
    y_r = torch.sum(x_abs * r, dim=1)            # dim=1 collapse: 25 → 1
    y_g = torch.sum(x_abs * g, dim=1)
    y_b = torch.sum(x_abs * b, dim=1)
    y = torch.stack([y_r, y_g, y_b], dim=1)      # (B, 3, H, W)
```

`torch.sum(..., dim=1)` 将 25 个波长通道不可逆地压缩为 1 个标量（R/G/B）。3 个传感器响应函数（sensor_r/g/b，从 `Sensor_25_new3.mat` 加载）是固定的 25→1 线性组合权重。

对 `spectral_bins` 和 `identity` 模式，collapse 发生在第 113-116 行：

```python
resp = self.response[None, :, :, None, None]     # (1, 25, C, 1, 1)
x_expanded = x_abs.unsqueeze(2)                   # (B, 25, 1, H, W)
y = torch.sum(x_expanded * resp, dim=1)           # dim=1 collapse: 25 → C
```

区别仅在于 collapse 到的通道数不同（3 / 8 / 25），但机制相同：固定线性组合 `sum(x_abs * response, dim=1)`。

## 3. 如何拆成两个方法

### 3.1 `compute_wavelength_resolved_intensity()`

```python
def compute_wavelength_resolved_intensity(self, x: torch.Tensor) -> torch.Tensor:
    """Compute wavelength-resolved intensity from complex field.

    Args:
        x: (B, 25, H, W) complex field at sensor plane.

    Returns:
        x_abs: (B, 25, H, W) wavelength-resolved intensity in float32.
    """
    x_abs = torch.abs(x).to(torch.float32)
    if self.sensor_measurement == "intensity":
        x_abs = x_abs ** 2
    return x_abs
```

这是纯提取——把现有的第 100-102 行独立出来。不添加任何新逻辑。

### 3.2 `collapse_to_measurement()`

```python
def collapse_to_measurement(self, x_abs: torch.Tensor) -> torch.Tensor:
    """Collapse wavelength-resolved intensity to measurement channels.

    Args:
        x_abs: (B, 25, H, W) wavelength-resolved intensity in float32.

    Returns:
        y: (B, C, H, W) collapsed measurement in float32.
    """
    if self.sensing_mode == "rgb":
        r = self.sensor_r[None, :, None, None]
        g = self.sensor_g[None, :, None, None]
        b = self.sensor_b[None, :, None, None]
        y_r = torch.sum(x_abs * r, dim=1)
        y_g = torch.sum(x_abs * g, dim=1)
        y_b = torch.sum(x_abs * b, dim=1)
        y = torch.stack([y_r, y_g, y_b], dim=1)
    else:
        resp = self.response[None, :, :, None, None]
        x_expanded = x_abs.unsqueeze(2)
        y = torch.sum(x_expanded * resp, dim=1)

    if self.normalize:
        if self.normalize_mode == "global":
            y = y / (torch.max(y) + self.eps)
        elif self.normalize_mode == "per_sample":
            y = y / (torch.amax(y, dim=(1, 2, 3), keepdim=True) + self.eps)
    return y
```

## 4. 旧 forward 如何保持完全兼容

重构后的 `forward()` 保持对外接口完全不变：

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(...)
    if x.shape[1] != self.input_bands:
        raise ValueError(...)

    x_abs = self.compute_wavelength_resolved_intensity(x)  # (B, 25, H, W)

    # ★ 此处为 future CFA/CCA 插入点 ★
    # if self.cfa_mode != "none":
    #     x_abs = self._apply_cfa(x_abs)  # (B, 25, H, W) → (B, C_cfa, H, W)

    y = self.collapse_to_measurement(x_abs)                 # (B, C, H, W)
    return y
```

**兼容性保证**：

- 外部调用方（`DoDoForwardModel`, `DepthAwareDoDoForwardModel`）不感知任何变化——`forward()` 的输入/输出 signature 完全不变。
- 所有旧 checkpoint 的 state_dict 结构不变（没有新增 `nn.Parameter` 或 `nn.Module`）。
- `compute_wavelength_resolved_intensity` 和 `collapse_to_measurement` 只是内部方法拆分，不改变数值路径。
- 单元测试可以直接调用 `compute_wavelength_resolved_intensity(x)` 检查 `x_abs` 的 wavelength 维度和有限性。

## 5. 未来 spectral_bins / Gaussian CFA / learnable CFA 的插入位置

**统一插入点：`forward()` 中 `compute_wavelength_resolved_intensity()` 之后、`collapse_to_measurement()` 之前。**

即 `x_abs: (B, 25, H, W)` 张量位置。

### 5.1 spectral_bins（已存在，属于 collapse 方式之一）

当前通过 `self.sensing_mode == "spectral_bins"` 分支实现，使用固定的 contiguous equal-width bins 响应矩阵。重构后，spectral_bins 仍然在 `collapse_to_measurement` 中通过 `self.response` 实现——它是 collapse 的一种方式，不是 wavelength-resolved 阶段的变换。

### 5.2 Gaussian CFA

```python
# 伪代码，插入在 collapse_to_measurement 之前
x_abs: (B, 25, H, W)
cfa_centers = torch.linspace(lambda_min, lambda_max, cfa_channels)      # (K,)
cfa_sigma = gaussian_sigma
for k in range(cfa_channels):
    g_k = exp(-(lambda_indices - cfa_centers[k])**2 / (2 * cfa_sigma**2))  # (25,)
    cfa_response[k, :] = g_k
# cfa_response: (K, 25)
x_cfa = einsum('bchw,kc->bkhw', x_abs, cfa_response)                   # (B, K, H, W)
```

CFA 的 per-filter spectral response `(K, 25)` 作用在 wavelength 维度，输出 `(B, K, H, W)`。之后可以：
- 直接作为 measurement 输出给 decoder（不经过 RGB collapse）
- 或者再经过 RGB sensors 做可视化渲染

### 5.3 Learnable CFA

```python
# self.cfa_weights: nn.Parameter (K, 25), 初始化为 Gaussian pattern
x_cfa = einsum('bchw,kc->bkhw', x_abs, self.cfa_weights)               # (B, K, H, W)
```

learnable CFA 的权重矩阵 `(K, 25)` 与 DOE zernike coefficients 一起优化。

## 6. 需要新增的 shape tests

重构后需要的最小测试集：

### 6.1 wavelength-resolved intensity shape

```python
def test_compute_wavelength_resolved_intensity_shape():
    layer = SensingLayer(sensing_mode='rgb')
    x = torch.randn(1, 25, 128, 128, dtype=torch.complex64)
    x_abs = layer.compute_wavelength_resolved_intensity(x)
    assert x_abs.shape == (1, 25, 128, 128)
    assert x_abs.dtype == torch.float32
    assert torch.isfinite(x_abs).all()
```

### 6.2 amplitude vs intensity mode (backward compat)

```python
def test_compute_wavelength_resolved_intensity_amplitude():
    layer_amp = SensingLayer(sensor_measurement='amplitude')
    layer_int = SensingLayer(sensor_measurement='intensity')
    x = torch.randn(1, 25, 128, 128, dtype=torch.complex64)
    a = layer_amp.compute_wavelength_resolved_intensity(x)
    b = layer_int.compute_wavelength_resolved_intensity(x)
    assert torch.allclose(b, a ** 2, atol=1e-5)
```

### 6.3 forward() output unchanged after refactor

```python
def test_forward_unchanged_after_refactor():
    torch.manual_seed(42)
    x = torch.randn(1, 25, 128, 128, dtype=torch.complex64)
    y_old = old_sensing_layer(x).clone()
    y_new = new_sensing_layer(x).clone()
    torch.testing.assert_close(y_new, y_old, atol=1e-6, rtol=1e-6)
```

### 6.4 collapse output shape for all modes

```python
def test_collapse_to_measurement_shapes():
    for mode, expected_c in [('rgb', 3), ('spectral_bins', 8), ('identity', 25)]:
        layer = SensingLayer(sensing_mode=mode, measurement_channels=8,
                             sensor_measurement='amplitude', normalize=False)
        x_abs = torch.rand(1, 25, 64, 64)
        y = layer.collapse_to_measurement(x_abs)
        assert y.shape == (1, expected_c, 64, 64)
        assert torch.isfinite(y).all()
```

### 6.5 collapse normalization unchanged

```python
def test_collapse_normalization_unchanged():
    layer = SensingLayer(normalize=True, normalize_mode='global')
    x_abs = torch.rand(1, 25, 64, 64)
    y = layer.collapse_to_measurement(x_abs)
    assert y.max() <= 1.0 + 1e-6
```

### 6.6 CFA insertion point (future-proof)

```python
def test_cfa_insertion_point_shape():
    """Verify that a CFA-like transform at the insertion point produces valid shapes."""
    layer = SensingLayer(sensing_mode='rgb', normalize=False)
    x_abs = layer.compute_wavelength_resolved_intensity(
        torch.randn(1, 25, 64, 64, dtype=torch.complex64))
    # Simulate a K=6 CFA: (25,) → (6,) linear map per spatial position
    cfa_weights = torch.randn(6, 25)
    x_cfa = torch.einsum('bchw,kc->bkhw', x_abs, cfa_weights)  # (1, 6, 64, 64)
    assert x_cfa.shape == (1, 6, 64, 64)
    assert torch.isfinite(x_cfa).all()
```

## 7. 总结

| 项目 | 结论 |
|------|------|
| 重构范围 | `forward()` 拆分为 `compute_wavelength_resolved_intensity()` + `collapse_to_measurement()` |
| 旧 forward 兼容 | 完全兼容，signature 不变，state_dict 不变 |
| 新接口暴露 | `x_abs: (B, 25, H, W)` 在 collapse 前可被外部/子类访问 |
| CFA 插入位置 | `compute_...` 之后、`collapse_...` 之前 |
| 新增参数 | 不需要（完全内部重构） |
| 新增测试 | 6 个 shape/数值/compat tests |
| 风险 | 极低——纯提取方法，零逻辑变更 |
