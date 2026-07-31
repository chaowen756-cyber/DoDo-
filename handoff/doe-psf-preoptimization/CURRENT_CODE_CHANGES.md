# Current Code Changes

## util/doe_preoptimization.py

- 区域：`DOEPreoptimizationWeights/Targets`、`optical_psf_shape_separation_loss`、
  `rms_constrained_optimizer_step_`、组合目标。
- 目的：只把 DOE 实际造成的单色 PSF 形状变化认作光谱/深度编码，并稳定处理
  3 μm RMS 活跃约束。
- 修改：新增波长1/2/4 offset和深度相邻层的 optical-only cosine；保留
  sensor-weighted 分离/Fisher；同时返回 warm-up train total 与固定 full total；
  候选 Adam 更新在边界移除向外法向分量后再安全回缩。
- 潜在风险：新目标比旧 sensor cosine 更严格且量级更大；通过独立权重显式控制。
  切向修正基于局部一阶 RMS 法向，仍需二阶回缩保证严格可行。
- 验证：尺度不变性/梯度单测、边界向外更新单测、2.99 μm GPU压力测试。

## scripts/preoptimize_psf_doe.py

- 区域：CLI、训练循环、日志、最佳状态选择与 summary。
- 目的：让实验日志可跨 warm-up 比较，并暴露新的光学编码与约束诊断。
- 修改：新增四类 separation 权重、optical offsets 和边界参数；控制台输出
  train/full loss及optical spectral cosine；最佳状态按固定 full loss选择；记录
  tangent/retraction计数、最小回缩比例和 optical feasibility。
- 潜在风险：历史 `--spectral_weight/--depth_weight` 不再是当前CLI参数；正式命令
  已完整显式指定新参数，旧实验 artifact 不受影响。
- 验证：CLI smoke、30/200步完整16深度GPU preflight、summary检查。

## test/test_doe_preoptimization.py

- 区域：目标函数与约束回归测试。
- 目的：防止仅靠波段强度变化骗过 optical 编码目标，防止 RMS 边界更新越界。
- 修改：验证光学形状损失对逐波段尺度不敏感、区分不同图样并可反传；验证固定
  full loss不随warm-up缩放；验证纯向外Adam更新被切向修正。
- 潜在风险：无生产运行时影响。
- 验证：本文件7项测试全部通过。

## docs 与 handoff

- 区域：实验说明、实现记录、实验台账和下一动作。
- 目的：记录双seed负结果、新目标含义、压力测试修复及正式命令判读口径。
- 修改：更新 `docs/doe_psf_preoptimization.md`、`EXPERIMENTS.md`、
  `implementation-notes.md`、`NEXT_ACTION.md`。
- 潜在风险：无运行时影响。
- 验证：文档参数名与CLI逐项核对。
