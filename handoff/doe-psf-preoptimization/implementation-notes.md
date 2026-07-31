# Implementation Notes

## 1. DOE-only 可行性入口

- 从 `psf卷积@06a6eb5` 新建分支 `DOE可编码性预优化实验`。
- 新入口不加载数据集和 CNN，只优化当前完整 PSF bank。
- rank-9 与 free-150 使用统一的 0.6 μm 初始化 RMS 和 3 μm 最大 RMS。
- 输出 DOE checkpoint、系数、高度、PSF、历史、摘要和可视化。
- CPU smoke 与16层短对照均完成；正式1000步实验需在可用GPU上执行。

## 2. Review 与验证

- 新增测试3项全部通过；原有 orthogonal-RMS/PSF 卷积回归37项全部通过。
- rank-9 与 free-150 最终 CLI smoke 均通过。
- `test/test_psf_regularization.py` 在当前 shell 因缺少 `pytorch_lightning`
  无法收集；本次直接调用的四个正则已由新增目标梯度测试覆盖。
- 当前 shell 的 NVIDIA 驱动不可用，因此没有在本轮冒充完成正式1000步GPU实验。

## 3. Fisher A-optimality 最后修改

- 依据同一 HS-D 任务的 Baek ICCV 2021 补充材料第6节，将单色点源
  `(x,y,depth,wavelength)` Fisher A-optimality 加入 DOE-only 目标。
- 保留 MTF 和能量护栏，防止 Fisher 优化通过牺牲空间带宽或 capture 取巧；
  原相邻光谱/深度 cosine 继续作为直接分离辅助项。
- 使用一个像素/深度层/波长层作为导数单位，并显式保留 RGB 响应与 PSF
  capture 强度。
- 16深度20步测试确认 `Adam lr=1e-2` 稳定；没有照搬联合训练的 `1e-4`。
- Fresnel 投影在当前 rank-9 基底中拟合较差，且在 free-150 中初始 Fisher
  不优，因此保留相同 seed、相同物理 RMS 的随机初始化。
- 最终新增测试4项、原有 DOE/PSF 回归37项全部通过。
