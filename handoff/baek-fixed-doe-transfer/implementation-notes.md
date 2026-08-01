# Implementation Notes

## 1. 2026-08-01 固定高度接入

新增冻结外部高度 DOE 表达、训练 CLI 与完整实验脚本。固定高度被注册为 buffer，
不会进入 optimizer；误用 `--optimize_optics` 会 fail fast。

## 2. 2026-08-01 改为 DOE 原生采样优先

在用户确认“采样不一致时以 DOE 优先”后，主实验从 128x128 归一化孔径形状迁移
改为 376x376、8 µm、NOA61、50 mm。进一步核对源 notebook/PADO 后，native
模式直接在 DOE 面生成球面波，并复现其整数中心圆孔和 Fresnel linear convolution，
而不是继续使用当前 Prop1 impulse/transfer-function 离散近似。

真实高度的 16x25 PSF 全 finite，光学可训练参数为 0。129x129 中心窗口相对
376x376 输出的能量占比为 0.237906--0.442250；卷积前把有限核校准为单位和，并
保留 capture fraction 日志。12 epoch 长训练未启动。
