# Baek Fixed DOE Transfer Task

## 目标

把 Baek 已训练的米制 DOE 高度放入当前 PSF 卷积模型；以 DOE 原生采样为准，冻结
光学，从头联合训练 HS/深度网络，并提供可复现实验命令。

## 约束

- 保持 375 到 376 的右/下补零、8 µm、NOA61、PADO 球面波/圆孔/Fresnel 和
  50 mm 传播约定。
- 原有 `legacy`/`consistent_grid_v1` 与场景卷积、sensing、网络、loss 不变。
- 不自动启动 12 epoch 长训练。

## 验收

- DOE 高度无插值地成为 376x376 buffer，零光学可训练参数。
- 误开光学优化时 fail fast。
- 16x25 个 129x129 PSF 全 finite，单位核和，记录裁剪能量占比。
- 提供完整双卡实验脚本、覆盖保护和 dry-run。
