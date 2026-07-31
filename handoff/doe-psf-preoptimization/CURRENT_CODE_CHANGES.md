# Current Code Changes

## util/doe_preoptimization.py

- 区域：`psf_fisher_a_optimality_loss` 与预优化 targets。
- 目的：把优化重点从 x/y 空间定位转向 HS-D 任务真正需要的深度/波长估计。
- 修改：仍求完整4×4 Fisher 逆矩阵，将 x/y 作为 nuisance 参数保留；新增
  可配置 CRLB 权重，默认 `(x,y,z,lambda)=(0.1,0.1,1,1)`；同时输出完整、
  任务加权、纯 z/lambda A-optimality 和四个单项 CRLB。
- 风险：新默认值改变后续实验目标，但不影响已有 checkpoint；权重归一化保持
  损失量级可比较。
- 验证：完整/任务权重公式测试、梯度测试、16深度 free-150 预检。

## scripts/preoptimize_psf_doe.py

- 区域：CLI、日志、summary 与 feasibility。
- 目的：显式控制并审计任务 Fisher 权重。
- 修改：新增三项 CRLB 权重参数；日志改报 task A-optimality；摘要同时保存
  full/task/weighted 改善，成功判据使用 task A-optimality。
- 风险：旧命令未显式指定时会采用新的任务权重，符合本轮实验定义。
- 验证：CLI smoke、summary artifact 测试、参数合法性检查。

## test/test_doe_preoptimization.py

- 区域：Fisher nuisance/权重回归。
- 目的：防止“聚焦 z/lambda”被错误实现为删除 x/y Fisher 子空间。
- 修改：验证 full trace 不随损失权重改变，纯任务权重等于完整逆矩阵中
  depth/wavelength CRLB 的加权和。
- 风险：无生产运行时影响。
- 验证：新增测试文件5项全部通过。

## docs/doe_psf_preoptimization.md

- 区域：目标说明、参数和判读。
- 目的：记录任务 CRLB 的数学含义和正式命令。
- 修改：加入 `(0.1,0.1,1,1)` 权重、nuisance 处理和 task 指标。
- 风险：无运行时影响。
- 验证：文档参数由 CLI 覆盖。
