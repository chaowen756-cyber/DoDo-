# Current Code Changes

## torch_optics/doe.py

- 新增 `DOEFixedHeightLayer`：读取米制 tensor/dict，高度校验、右/下补零、可选插值，
  以 buffer 保存且无 Parameter。
- 为 `doe_native_grid_v1` 增加 PADO integer-centered circular aperture；其他 DOE
  模式沿用原 pupil。

## torch_optics/propagation.py

- 新增独立的 `PadoFresnelPropagationLayer`，复现 PADO Fresnel `linear=True` 的
  2N 对称补零、正相位空间脉冲响应、FFT 线性卷积和中心回裁。
- 原 `PropagationLayer` 未改语义。

## torch_optics/forward_dodo.py

- 新增 `fixed_height` 参数化和 `doe_native_grid_v1`。
- native 模式固定 376x376、8 µm、50 mm；在 DOE 面直接构造 PADO 球面波，使用
  PADO pupil/propagator，输出中心 129x129 单位和 PSF，同时记录裁剪能量占比。
- native 模式只允许 fixed height、skip Prop2、无第二 DOE、linear-zero PSF 卷积。

## snapshotdepth_hs.py / scripts

- 训练 CLI 接入高度路径、补零尺寸、插值方式和新 optics version；固定高度强制
  `--no-optimize_optics`。
- `scripts/run_baek_fixed_doe_joint.sh` 提供双卡 12 epoch 完整命令、依赖预检、
  overwrite protection 和 dry-run。

## 风险边界

- 当前 halo64 只容纳 129x129 PSF，而 native PADO 输出为 376x376；当前中心核在
  16x25 PSF 上保留约 23.8%--44.2% 能量，已校准单位和并记录诊断。
- 长训练尚未启动。
