# Recent Collaboration History

> Keep only the most recent 5 rounds.

---

## Round 17

### Owner
Codex

### Change
DoDo-change

### Goal
Review inference-only ablation 结果，并决定是否进入训练/架构改动。

### Result
Round 17 的 10 个 inference-only ablation 全部失败：norm override 无效果，minmax 和 tile skip 无实质改善，两个 deploy 的 SAM、pseudo-RGB PSNR、depth-vs-median baseline 均未达标。下一轮转入短程训练/架构 probe：让 DoDo forward 内部归一化可配置，加入 background HS loss，并用 full-scene gate 决定是否允许后续长训。

---

## Round 18

### Owner
Codex

### Change
DoDo-change

### Goal
Review forward-normalization / background-loss training probe，并设计下一轮针对 full-scene 失败的方案。

### Result
Round 18 两个 40 epoch candidate 均失败：crop PSNR 低于 20 dB，full-scene SAM 和 pseudo-RGB PSNR 显著差于 R17 baseline，depth 仍差于 constant median-depth baseline。下一轮不延长同配置训练，改为 Round 19 multi-channel sensing diagnostic：新增 opt-in 8-channel spectral-bin sensing 和 25-channel identity upper-bound，验证 3-channel RGB sensing 是否是主要信息瓶颈。

---

## Round 19

### Owner
Codex

### Change
DoDo-change

### Goal
Review multi-channel sensing diagnostic，并决定是否继续训练还是转入 forward contract 审计。

### Result
Round 19 8-channel spectral-bin 和 25-channel identity 两个 40 epoch candidate 均失败，crop PSNR 分别为 14.20 dB 和 18.69 dB，full-scene 仍显著差于 R17/R12 baseline。结合 R12/R17 260 epoch 已出现“crop 指标好但 full-scene 实际结果不对”，下一轮不再做增量训练，改为光学前向模型正确性审计：检查 sensor intensity、coherent propagation、GT depth oracle、invalid/background depth 和 normalization/masked metric 膨胀。

---

## Round 20

### Owner
Codex

### Change
DoDo-change

### Goal
Review 光学前向有效性审计，并决定下一轮是否进入代码修正。

### Result
Round 20 确认 `abs(field)` sensing 与物理 `abs(field)^2` intensity 差异显著，masked PSNR 在 deploy 1 上虚高约 8.45 dB，GT-depth oracle 依赖仍阻断真实 measurement-only 部署。相干可加性在当前测试中不是首要问题；背景贡献诊断因使用 100% foreground crop 仍需补测。下一轮先不改生产代码，要求 Claude 写出 opt-in intensity sensing 和指标契约修正的具体代码落实方案，等待 Codex review 后再执行。

---

## Round 21

### Owner
Codex

### Change
DoDo-change

### Goal
Review intensity sensing 代码落实方案，并决定是否批准实现。

### Result
Round 21 方案通过 Codex review，批准进入 Round 22 最小实现。执行时必须保持 `amplitude` 默认兼容旧 checkpoint，新增 opt-in `intensity`，同时修正 full-scene oracle 标签和指标契约。Round 22 不允许长训，只能跑 deterministic smoke、finite forward、tiny 1-epoch preflight 和低前景诊断。
