# Baek notebook 与当前 native PSF 一致性实验

## 目的

在不训练重建网络的情况下，直接比较同一固定 DOE 高度在 Baek notebook/PADO
参考前向与当前 `doe_native_grid_v1` 下的 PSF。参考端实际调用 PADO API，不使用
当前仓库的兼容传播代码作为“参考”。

## 实验设置

- 高度：`e2e_HSD_doe_height.pth`，375x375 右/下补零到 376x376。
- 网格：376x376，8 µm；NOA61；圆孔直径 3.008 mm；传播 50 mm。
- 波长：420--660 nm，25 个。
- 深度：`1 / linspace(1/0.3, 1/2.0, 20)`，严格使用 notebook 的 20 点。
- 总计：500 个 wavelength-depth PSF。
- PADO：GitHub revision `12c57df2467127636a83415aa2a2f50ca6988840`，
  package version 1.0.1，临时安装到 `/tmp`。
- 比较范围：完整 376x376、notebook 的中心 96x96、当前部署的中心 129x129。
- 强度均先按完整网格单位和归一化；有限裁剪形状指标再各自单位和归一化。

复现实验：

```bash
python -m pip install --no-deps --target /tmp/pado_compare_git \
  'git+https://github.com/shwbaek/pado.git'

cd /home/wenchao/autodl-tmp
CUDA_VISIBLE_DEVICES=1 \
PYTHONPATH=/tmp/pado_compare_git \
/home/wenchao/conda_envs/ld_clean/bin/python \
  scripts/compare_baek_native_psfs.py \
  --device cuda:0 \
  --pado_revision 12c57df2467127636a83415aa2a2f50ca6988840
```

## 当前版本对 PADO 的结果

完整 376x376 PSF：

- cosine：mean `0.974696`，median `0.995426`，min `0.537816`。
- NRMSE：mean `0.150379`，median `0.097317`，P95 `0.464780`，max
  `1.106150`。
- total variation distance：mean `0.102704`，max `0.410970`。
- `62.6%` 的 PSF cosine >= 0.99，`87.2%` >= 0.95。
- `50.6%` 的 PSF NRMSE <= 0.1，`81.8%` <= 0.25。

中心裁剪没有消除形状偏差：

- 96x96 NRMSE mean/max：`0.146951 / 1.115253`。
- 129x129 NRMSE mean/max：`0.147419 / 1.120954`。

最差 full-grid NRMSE 位于 590 nm、2.0 m：cosine `0.573889`，NRMSE
`1.106150`。最低 cosine 位于 560 nm、2.0 m：cosine `0.537816`。误差整体随
深度上升：0.3 m 的跨波长 mean NRMSE 为 `0.056885`，2.0 m 为 `0.343761`。

裁剪能量比例本身几乎一致，但这不能证明 PSF 形状一致：

- PADO/current 96x96 mean capture：`0.313147 / 0.313203`。
- PADO/current 129x129 mean capture：`0.361708 / 0.361773`。
- 129 capture delta 范围：`-0.004481` 到 `0.004856`。

## 根因隔离

Baek notebook 为每个波长单独创建 `Light`，`wvl_val` 是 Python float；当前实现
一次向量化生成 25 个波长的球面波，并以 float32 wavelength tensor 做除法。
球面相位 `2*pi*r/lambda` 在 0.3--2.0 m 范围达到数百万至数千万弧度。float32
在这个量级的舍入误差会在相位取模后变成可见的 O(1) 相位误差，且距离越远越
严重。

证据：球面波 complex coherence 与最终 NRMSE 的相关系数为 `-0.7717`。保持
当前 DOE、pupil 和传播完全不变，只把输入球面波改为 PADO 的逐波长 Python 标量
计算语义后：

- full-grid cosine：mean `0.999999995`，min `0.999999762`。
- full-grid NRMSE：mean `4.7866e-6`，max `1.6164e-5`。

这说明 376x376 高度、NOA61 相位、PADO 圆孔和 50 mm Fresnel 传播实现已经对齐；
当前差异集中在 native 球面波的浮点计算语义，不是 DOE 高度或采样几何错误。

## 结论

当前版本不应立即开展正式训练。虽然平均 cosine 较高，仍有一批远距离/特定波长
PSF 发生实质形状偏差，而 capture fraction 会掩盖这种问题。下一步应先把 native
球面波生成改成与 PADO 数值稳定且等价的实现，再以本实验要求 500 个 PSF 的
max NRMSE 达到约 `1e-5` 量级，之后才讨论是否进行网络训练。

## 产物

目录：`论文实验/PSF卷积/baek_native_psf_parity_20260801/`

- `summary.json`：总体统计和最差案例。
- `per_psf_metrics.csv`：500 组逐项指标。
- `normalized_psf_crops_96.pt`：PADO/current/aligned 的归一化 96x96 PSF。
- `metric_heatmaps.png`：波长-深度误差热图。
- `representative_psf_comparison.png`：代表性 PSF 对比。
- `worst_case_psf_comparison.png`：最差案例及根因隔离结果。
