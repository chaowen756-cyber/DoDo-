# Current Code Changes

## util/doe_preoptimization.py

- 区域：DOE-only 目标函数与指标。
- 目的：补齐 Baek ICCV 2021 同任务 DOE 初始化所用的 Fisher 信息目标。
- 修改：构造单色点源 RGB PSF 关于 x/y/深度/波长的离散 Fisher 矩阵，加入
  A-optimality、ridge、数值缩放和特征值/条件数指标；与原 MTF、cosine、能量
  护栏共同优化。
- 风险：Fisher 项增加显存和计算；A-optimality 对参数单位敏感，因此四个参数
  明确定义为一个采样 bin，并由测试锁定信号强度行为。
- 验证：Fisher 梯度/信号尺度测试、完整16深度20步预检、端到端 CLI smoke。

## scripts/preoptimize_psf_doe.py

- 区域：优化参数、日志和结果摘要。
- 目的：让正式 rank-9/free-150 实验显式记录和比较 Fisher 可辨识性。
- 修改：新增 Fisher weight/ridge/scale CLI；日志输出 A-optimality；summary 记录
  A-optimality 与最小特征值改善，并增加参数合法性检查。
- 风险：旧命令会采用新的默认 Fisher 项，符合本轮“最后修改”后的实验定义。
- 验证：CPU CLI smoke、参数解析和 artifact 测试。

## test/test_doe_preoptimization.py

- 区域：Fisher 目标回归。
- 目的：确保损失有限、可微，并保留物理信号强度而非逐 PSF 归一化。
- 修改：验证放大 PSF 强度会提高 Fisher 信息并降低 A-optimality loss。
- 风险：无生产运行时影响。
- 验证：本测试文件全部通过。

## docs/doe_psf_preoptimization.md

- 区域：实验依据、命令与判读。
- 目的：记录 Baek/D-Flat/dO 对照以及为何保留 `Adam lr=1e-2` 和随机初始化。
- 修改：加入 Fisher 原理、参数、参考资料及预检依据。
- 风险：无运行时影响。
- 验证：命令参数由 CLI smoke 覆盖。
