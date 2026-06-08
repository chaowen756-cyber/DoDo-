# DoDo 光学前向模型审计

## 1. 前向合约映射

### 1.1 输入路径（full-scene 推理，`infer_contect.py`）

```
EXR 文件 → hs_norm (25通道, [0,1])
EXR 文件 → depth_gt_raw (物理米制深度)
         → valid_mask = (depth_gt_raw > min_depth - 1e-3)
         → depth_metric_raw = clamp(depth_gt_raw, min_depth, max_depth)
         → depth_ips (逆透视空间, [0,1])
```

**关键点**：`depth_metric_raw` 是从 GT depth 文件中读取的物理深度，直接用于光学前向模型。这是 **oracle/仿真** 操作——真实部署中深度未知。

### 1.2 `DepthAwareDoDoForwardModel.forward()`（训练 + 推理）

```
1. spectral: (B, 25, H, W) — dodo_depth 路径中已用 valid_mask 抑制背景
2. depth: (B, H, W) — 米制深度，背景 clamp 到 min_depth
3. 对每个深度 bin k (k = 0..num_depth_layers-1):
   a. depth_weight_k = soft indicator: 像素属于 bin k
   b. x_k_in = spectral * depth_weight_k  (B, 25, H, W)
   c. prop1_k(x_k_in)     → 复振幅场传播到 DOE1 平面
   d. doe1(field)          → 相位调制
   e. prop2(field)         → 传播到 DOE2 平面
   f. doe2(field)          → 相位调制
   g. prop3(field)         → 传播到 sensor 平面
   h. sensing_unnorm(field) → **torch.abs(field)** 加权求和 → 3/8/25 通道
   i. y_sum = y_sum + y_k  (跨深度累加)
4. _normalize_once(y_sum) 若 mode=legacy_max → y ∈ [0,1]
5. measurement_norm_mode: per_sample_mean_std 或 none → 最终 captimgs
```

**物理疑点**：
- x_k_in 以实数值（HS 强度代理）进入，以复振幅传播
- 跨深度干涉：y_sum = Σ y_k 累加的是振幅，而非强度
- 步骤 3h 测量 `abs(field)` 而非 `abs(field)^2`（物理强度）

### 1.3 Decoder 输入

```
captimgs (B, 3/8/25, H, W) → input_adapter → backbone → hs_head / depth_head
```

深度仅通过 **captimgs**（测量信号）进入 decoder，不作为独立输入。Decoder 必须单独从 captimgs 联合重建 25 通道 HS + 深度。

### 1.4 训练损失和 Mask

```
final_mask = valid_mask * boundary_mask  (crop_width 裁剪后)
foreground_loss = L1(est_hs, gt_hs) * final_mask
depth_loss = L1(est_depth, gt_d_ips) * final_mask
background_hs_loss = L1(est_hs, gt_hs) * (1-final_mask)  (可选)
```

Masked PSNR 仅在 `final_mask > 0.5` 上计算（通常占图像 25-37% 像素）。

## 2. 可疑问题（按物理影响排序）

### 问题 1：振幅测量 vs 强度测量（高）
- **代码**：`torch_optics/sensing.py:89` — `x_abs = torch.abs(x).to(torch.float32)`
- **物理预期**：光学传感器测量的是强度 = |E|²，而非振幅 |E|
- **影响**：振幅测量的动态范围约为强度测量 sqrt。经过 `_normalize_once` 归一化后两者在数值上可能相似，但底层物理不同。
- **验证**：Diagnostic A — 对比振幅 vs 强度测量的统计分布和相关性。

### 问题 2：自然图像作为相干场传播（高）
- **代码**：`torch_optics/propagation.py:64-67` — `x_complex = x.to(torch.complex64)` 然后 FFT 传播
- **物理预期**：自然场景是空间非相干的——各像点之间无固定相位关系
- **影响**：将自然 HS 图像视为相干波前会在 DOE/sensor 平面产生非物理干涉图案。两个空间分离的光源会相互干涉，这在非相干成像中不会发生。
- **验证**：Diagnostic B — 测试 meas(A+B) vs meas(A)+meas(B) 的可加性。

### 问题 3：GT-Depth Oracle 依赖（高）
- **代码**：`infer_contect.py:368-371` — `depth_metric_raw = depth_gt_raw.copy()` 传入 `model.forward()`
- **物理预期**：测量应由入射光线形成，深度未知
- **影响**：当前推理是 oracle simulation，不是真正的 measurement-only deployment。结果不可转移至物理系统。真实部署需要从 measurement 联合估计 HS 和 depth，缺少 GT depth 输入。
- **验证**：Diagnostic E — 文档化 oracle 依赖路径。

### 问题 4：无效深度 Clamp 污染测量（中）
- **代码**：`datasets/hyperspectral_dataset.py:276` — `depth_metric_tensor = torch.clamp(depth_tensor, self.min_depth, self.max_depth)`
- **影响**：背景像素（无物体）被赋值为最近深度平面（0.4m），仍然参与光学测量。前向模型的传播依赖深度——对无效像素赋予任意 clamp 值会在测量中产生人为背景贡献。
- **验证**：Diagnostic C — 量化背景像素对测量能量的贡献。

### 问题 5：Per-Patch 归一化 + Masked PSNR 虚高（中）
- **代码**：`snapshotdepth_hs.py:769-771` — 每个 128×128 tile 独立 `per_sample_mean_std` 归一化
- **影响**：低前景 tile 的少量有效像素被拉伸到全动态范围，造成白底效果。Masked PSNR 只在 25-37% 有效像素上计算，完全忽略背景崩塌。full PSNR 与 masked PSNR 差距约 8-9 dB。
- **验证**：Diagnostic D — 量化 PSNR 膨胀幅度。

## 3. 诊断计划

参见 `round20_forward_diagnostics.py` 可执行 Diagnostics A-E。结果保存至 `infer_results/DoDo-change/round20_forward_audit_v1/diagnostics/`。

## 4. 诊断结果汇总

### Diagnostic A: 振幅 vs 强度 Sensing

| 指标 | 振幅测量 | 强度测量 |
|------|---------|---------|
| 动态范围 | 5.3× | 15.8× |
| 标准差 | 1.51 | 2.54 |
| R/G/B 通道相关系数 | 0.956 | — |
| 归一化后全局相关系数 | 0.927 | — |

**结论**: 振幅 vs 强度存在显著差异（r<0.95）。强度测量动态范围是振幅的 3×。`_normalize_once` 归一化压缩了部分差异但不足以完全掩盖。当前 `abs(field)` 测量不是物理正确的强度测量。建议修改为 `abs(field)^2`。

### Diagnostic B: 相干可加性

| 指标 | 值 |
|------|-----|
| ∥meas(A+B)−Σmeas∥/∥meas(A+B)∥ | 0.0024 |
| 最大残差 | 0.0058 |

**结论**: 可加性近似成立。在此测试（两个空间分离的 40×40 源）上相干干涉可忽略。这不能排除更复杂场景或更近间距时出现显著干涉。当前不是首要问题。

### Diagnostic C: 背景贡献

| 指标 | 值 |
|------|-----|
| Crop valid_ratio | 100% |
| 背景能量占比 | 0% |

**结论**: 测试 crop 为全前景，背景贡献为 0。需要在低前景 tile（valid_ratio < 10%）上重新测试才能检测背景 clamp 效应。受时间限制，本轮先记录此限制。

### Diagnostic D: PSNR 膨胀

| Deploy | Masked PSNR | Full PSNR | 膨胀 |
|--------|------------|-----------|------|
| deploy 1 | 30.02 dB | 21.56 dB | **+8.45 dB** |
| deploy 16 | 18.96 dB | 21.03 dB | −2.07 dB |

**结论**: deploy 1 masked PSNR (30 dB) 仅在 25.8% 有效像素上计算，比 full PSNR 虚高 8.45 dB。Full PSNR (21.6 dB) 才是真实图像重建质量。deploy 16 负膨胀说明 foreground 重建比 background 更差。

### Diagnostic E: GT-Depth Oracle

**确认**: `infer_contect.py:368-371` 读取 GT depth → `depth_metric_raw` → 传入 `model.forward(depth_metric=...)` 合成 measurement。这是 oracle simulation，不是 measurement-only deployment。真实物理系统中没有 GT depth 输入。

### 根因最终排序

1. **GT-depth oracle**（阻断开销最大）：当前无法进行 measurement-only 推理
2. **振幅 vs 强度**（物理不正确）：修改为 `abs^2` 或 `|field|^2`
3. **Masked PSNR 虚高**（指标误导）：应替换为 full PSNR + SAM + pseudo-RGB PSNR
4. **相干传播**（当前测试中可忽略）：残差 0.24%
5. **背景 depth clamp**（需更多测试）：全前景 crop 上未触发
