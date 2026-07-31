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
