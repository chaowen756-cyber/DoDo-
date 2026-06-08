# Review: DoDo-change optical forward parity

## Findings

1. **功能完全一致目标尚未完成：Sensing 归一化零输入行为不同。**
   `optics/Sensing_layer.py:65`-`67` 直接执行 `y_final / reduce_max(y_final)`，当输出全零时会产生 `NaN`；`torch_optics/sensing.py:68`-`70` 使用 `torch.max(y) + eps`，默认 `eps=1e-8`，全零输入会得到有限的 0。正常非零输入下公式基本一致，但如果目标是“两个版本功能完全一致”，这个默认行为必须改成 legacy exact，或至少由显式参数切换且默认对齐原版。

2. **可训练 `DOE` 的参数约束仍不一致。**
   原版 `DOE` 的 `zernike_coeffs` 形状是 `(1, 1, N)`，并在 `optics/DOE_layer_v4_128.py:53`-`56` 使用 `MinMaxNorm(axis=2)`，约束的是整条系数向量的 L2 norm；PyTorch 当前 `_BaseDOE.clamp_parameters_()` 在 `torch_optics/doe.py:47`-`50` 对每个元素独立 `clamp_(-1, 1)`。这会改变训练可达参数空间，影响 `DOE_type='New'` 的训练一致性。注意：`DOE_Free` 原版系数形状是 `(N, 1, 1)` 且 `axis=2`，逐元素 clamp 与其更接近；该问题主要针对 `DOELayer`。

3. **`DOE_Free` 缺失 Zernike 基底文件时的功能不一致。**
   原版在 `optics/DOE_layer_v4_128.py:122`-`127` 会用 `poppy.zernike.zernike_basis(...)` 自动生成并保存缺失的 `zernike_volume1_<Mdoe>_Nterms_<Nterms>.npy`；PyTorch 在 `torch_optics/doe.py:172`-`174` 直接抛出 `FileNotFoundError`。当前已有 150/200 项资产，所以默认路径不触发；但对原版功能覆盖来说仍是不一致。

4. **上一轮实现记录中关于 `DOE_type='New'` 随机初始化数量的结论错误。**
   `handoff/DoDo-change/implementation-notes.md` 写到原版随机初始化 11 个系数，但源码 `optics/DOE_layer_v4_128.py:40`-`51` 初始化索引 `0..11`，共 12 个；PyTorch `torch_optics/doe.py:116`-`118` 也是 12 个。因此这不是代码差异，需要修正文档，避免下一轮基于错误结论改坏代码。

## Completed

- 已完成公式级验证记录：传播核、DOE 相位公式、IdLens 色散公式、光谱传感求和、FFT shift 顺序、默认前向链路均已被上一轮记录覆盖。
- 已确认当前环境缺少 TensorFlow/Keras，因此上一轮未能做跨框架数值 parity 测试是合理限制。
- 已完成 PyTorch-only smoke check，并记录输出形状与有限性。

## Required Next Step

下一轮应由 Claude 修正剩余不一致并更新实现记录：

- 修正 `SensingLayer` 默认归一化行为，使默认路径与原版 `y / max(y)` 一致。
- 修正 `DOELayer` 的训练后参数投影语义，使其匹配原版 `MinMaxNorm(axis=2)` 的向量 L2 norm 上限。
- 为 `DOEFreeLayer` 恢复缺失基底文件自动生成逻辑，生成公式必须匹配原版 `1e-6 * poppy.zernike.zernike_basis(...)`。
- 修正 `implementation-notes.md` 中“11 个随机初始化系数”的错误结论。
- 不要修改原版 `optics/`。
