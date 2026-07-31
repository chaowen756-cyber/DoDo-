# Current Code Changes

## torch_optics/doe.py

- 区域：`_BaseDOE` 物理高度接口。
- 目的：让 rank-9 与 free-150 使用可比较的 pupil-height RMS 预算。
- 修改：统一提供 `heightmap()`、`pupil_rms()` 和 `project_height_rms_()`。
- 风险：仅新增接口；历史 `clamp_parameters_()` 行为不变。
- 验证：高度初始化/投影单元测试、原有 DOE 测试。

## util/doe_preoptimization.py

- 区域：DOE-only 目标、初始化、诊断和 checkpoint 加载。
- 目的：复用现有 PSF 正则并避免依赖训练网络。
- 修改：组合 MTF、传感器加权波长/深度分离、能量护栏；增加公共物理预算和恢复函数。
- 风险：多目标权重决定搜索方向；所有原始分量均独立落盘以便审计。
- 验证：合成 PSF 梯度测试和 checkpoint 恢复测试。

## scripts/preoptimize_psf_doe.py

- 区域：独立 CLI。
- 目的：运行 rank-9/free-150 同预算可行性实验。
- 修改：冻结非 DOE 参数，执行 Adam+RMS 投影，保存 checkpoint、历史、指标和图像。
- 风险：完整 16×25 PSF bank 在 CPU 较慢；正式实验应使用 GPU。
- 验证：CPU 双模式 smoke、完整16层短对照。

## test/test_doe_preoptimization.py

- 区域：新实验的单元和端到端 smoke。
- 目的：验证物理预算、目标梯度、artifact 与恢复闭环。
- 风险：端到端 CPU smoke 有少量运行时开销。
- 验证：目标测试文件全部通过。

## docs/doe_psf_preoptimization.md

- 区域：实验说明和命令。
- 目的：明确实验回答的问题、非目标、输出与判读标准。
- 风险：无运行时影响。
- 验证：命令参数由 CPU smoke 覆盖。
