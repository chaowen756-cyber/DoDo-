# TASK: DoDo-change 长时间训练-验证-迭代方案

## Codex Review Update: 2026-05-06

Review file:

- `reviews/review-DoDo-change-post-long-training-2026-05-06.md`
- `reviews/review-DoDo-change-round8-dodo-nonfinite-2026-05-06.md`
- `reviews/review-DoDo-change-round9-finite-stability-2026-05-06.md`
- `reviews/review-DoDo-change-round10-depth-tradeoff-2026-05-06.md`
- `reviews/review-DoDo-change-round11-pareto-2026-05-07.md`
- `reviews/review-DoDo-change-round12-depth-best-2026-05-07.md`
- `reviews/review-DoDo-change-round13-finetune-audit-2026-05-07.md`
- `reviews/review-DoDo-change-round14-final-selection-2026-05-07.md`
- `reviews/review-DoDo-change-round15-fullscene-validity-2026-05-07.md`
- `reviews/review-DoDo-change-round16-diagnostic-audit-2026-05-07.md`
- `reviews/review-DoDo-change-round17-inference-ablation-2026-05-08.md`
- `reviews/review-DoDo-change-round18-forwardnorm-probe-2026-05-08.md`
- `reviews/review-DoDo-change-round19-sensing-diagnostic-2026-05-09.md`
- `reviews/review-DoDo-change-round20-forward-audit-2026-05-09.md`
- `reviews/review-DoDo-change-round21-code-plan-2026-05-09.md`
- `reviews/review-DoDo-change-round22-soft-diopter-validation-2026-05-09.md`

### Round 21 Review Update

Claude 第二十一轮按要求只写了 intensity sensing 实现方案，没有修改生产代码、没有训练、没有推广 checkpoint。

已确认：

- `ROUND21_CODE_PLAN.md` 覆盖了 Scope、Code Changes、Validation Plan、Metric Contract、Risks、Stop Condition。
- 方案采用 opt-in `--dodo_sensor_measurement {amplitude,intensity}`，默认 `amplitude`，兼容旧 checkpoint。
- 方案要求 `intensity` 使用 `abs(field)^2`，并覆盖 `rgb`、`spectral_bins`、`identity`。
- 指标合约方向正确：full PSNR、SAM、pseudo-RGB PSNR、depth-vs-median baseline 作为 primary，masked PSNR 降为 secondary。

Codex 审查通过，但 Round 22 执行时必须修正三点：

- backward-compat test 不能只比较两个新构造对象，必须与旧公式的手工实现输出对齐。
- `oracle_simulation=True` 必须写入 `metrics_real.txt`、`aggregate_metrics.json`、per-scene `metrics_full_scene.csv` 和 `diagnostic_metrics.json`。
- 新低前景诊断脚本不得硬编码 `/root/autodl-tmp`，必须支持 CLI 或相对当前工作区路径。

### Round 20 Review Update

Claude 第二十轮完成光学前向有效性审计，产出 `OPTICAL_FORWARD_AUDIT.md`、Section 20、实验台账和独立诊断脚本。

已确认：

- 当前 `SensingLayer` 使用 `abs(field)` 做 sensor integration，而物理 sensor 应更接近 `abs(field)^2` intensity；诊断显示归一化后相关系数约 0.927，强度测量动态范围约为振幅测量 3 倍。
- 相干可加性测试在两个分离源上残差约 0.24%，当前不是首要问题，但不能完全排除复杂场景问题。
- deploy 1 masked PSNR 相对 full PSNR 虚高约 8.45 dB，说明 masked PSNR 不能作为 full-scene 主要成功指标。
- full-scene inference 依赖 GT depth 合成 measurement，当前是 oracle simulation，不是真实 measurement-only deployment。
- 背景/无效深度贡献诊断使用了 100% foreground crop，未覆盖低前景 tile，结论不完整。

Codex 结论：

- 下一步不训练，也不直接改生产代码。
- Round 21 先写具体代码落实方案，等待 Codex review 通过后再执行。
- 方案应聚焦 opt-in intensity sensing、full-scene 指标契约修正、低前景 tile 背景贡献诊断补齐、以及 oracle simulation 标注。
- `CURRENT_CODE_CHANGES.md` 需要修正表述：新增诊断脚本也是代码变更，只是没有修改生产代码。

### Round 19 Review Update

Claude 第十九轮完成 multi-channel sensing diagnostic，并跑了 8-channel `spectral_bins` 与 25-channel `identity` 两个 40 epoch candidate。

已确认：

- 8-channel candidate：crop PSNR 14.20 dB，deploy 1 SAM 0.856 rad，pseudo-RGB PSNR 17.59 dB，full-scene FAIL。
- 25-channel identity upper-bound：crop PSNR 18.69 dB，deploy 1 SAM 0.635 rad，pseudo-RGB PSNR 15.40 dB，full-scene FAIL。
- 两个 candidate 均明显差于 R17/R12 260-epoch baseline，不替换当前 R12 depth-best checkpoint。
- Round 19 没有证明 multi-channel sensing 无效，只证明 40 epoch from scratch 已成为主要混杂因素。
- 新增协作规则中的 `CURRENT_CODE_CHANGES.md` 与 `EXPERIMENTS.md` 未被实际维护，下一轮必须补齐实验台账，并严格维护当前轮代码修改台账。

Codex 结论：

- 不再继续新增 40 epoch from-scratch 架构分支。
- 也不应马上基于 Round 19 或 R12/R17 checkpoint 做增量 epoch 实验，因为 R12/R17 已证明“crop 指标好但 full-scene 实际结果不对”。
- 下一轮应暂停训练，转为光学前向模型正确性审计。
- 优先检查：`abs(field)` vs `abs(field)^2` sensor intensity、自然场景 coherent propagation 假设、GT depth 参与 measurement synthesis、invalid/background depth 处理、per-patch normalization 与 masked metric 膨胀。
- 只有确认并修正 forward contract 后，才值得重新设计训练实验。

### Round 18 Review Update

Claude 第十八轮完成 forward normalization / background loss 的短程训练 probe，并跑了 full-scene gate。

已确认：

- `dodo_forward_norm=none` 的 1 epoch preflight 通过，`nonfinite_count=0`。
- Candidate A (`forward_norm=none`) 与 Candidate B (`legacy+lowvalid+bg`) 都完成 40 epoch，但 crop validation PSNR 均低于 20 dB。
- 两个 candidate 的 full-scene 指标均显著差于 R17 baseline：deploy 1 SAM 约 0.75 rad，pseudo-RGB PSNR 约 11.5-11.7 dB。
- depth 仍未超过 constant median-depth baseline，`mae_vs_median` 仍大于 1。
- Round 18 不支持继续同配置延长到 120/260 epoch。

Codex 结论：

- forward normalization 和 background loss 不是当前 full-scene 失败的主瓶颈。
- 下一轮应直接诊断 3-channel RGB sensing 是否构成信息瓶颈。
- Round 19 需要新增 opt-in multi-channel sensing，并保留默认 RGB 行为不变。
- 先跑 8-channel spectral-bin production-like candidate，再跑 25-channel identity upper-bound diagnostic。
- 只有 8-channel candidate 明确改善 full-scene gate，才允许后续考虑长训。

### Round 17 Review Update

Claude 第十七轮完成 10 个 inference-only ablation，并修复 reporting contract。

已确认：

- `measurement_norm_override=none` 与 checkpoint 默认结果一致，说明 `snapshotdepth_hs.py` 的第二阶段 `dodo_measurement_norm` 不是 full-scene 失败主因。
- `per_sample_minmax` 仅带来极小 depth 变化，SAM 和视觉质量仍失败。
- `min_tile_valid_ratio=0.1` 跳过约 0.1% tile，几乎不改变质量。
- `deploy 1` 最好仍约 SAM 0.540、pseudo-RGB PSNR 26.40 dB、`mae_vs_median` 1.027。
- `deploy 16` 最好仍约 SAM 0.412、pseudo-RGB PSNR 20.51 dB、`mae_vs_median` 1.133。
- 所有 inference-only setting 均未满足 acceptance criteria。

Codex 结论：

- inference-only 路线已结束。
- 下一轮应进入小规模训练/架构 probe，但不能直接长训。
- 主要修改点转为 `torch_optics/forward_dodo.py` 内部 `_normalize_once(y_sum)`，需要改成可配置 forward normalization。
- 新训练必须加入 full-scene gate：SAM、pseudo-RGB PSNR、depth-vs-median baseline 和 quicklook，不再只依赖 crop masked PSNR。

下一轮目标：

1. 增加 `--dodo_forward_norm`，默认保持旧行为，支持至少 `legacy_max` 和 `none`。
2. 增加 `--background_hs_loss_weight`，惩罚 background white/pink artifact。
3. 先跑 1 epoch finite preflight。
4. preflight 通过后最多跑两个 40 epoch candidates。
5. 每个 candidate 必须跑 deploy 1 和 deploy 16 full-scene gate，只有明显改善才允许下一轮延长。

### Round 16 Review Update

Claude 第十六轮完成 full-scene validity audit，并确认 Round 15 full-scene 成功结论需要撤销。

已确认：

- `deploy 1`: masked PSNR 30.02 dB 但 SAM 0.540 rad、pseudo-RGB PSNR 26.95 dB、depth MAE 0.453m。
- `deploy 1` constant median-depth baseline MAE 是 0.424m，模型 depth 甚至差于常数中位数 baseline。
- `deploy 16`: masked PSNR 18.96 dB、SAM 0.417 rad、depth MAE 0.489m，也差于常数 baseline。
- 当前 checkpoint 只能作为 filtered 128x128 crop eval 的最佳模型，不能作为 full-scene usable checkpoint。
- full-scene eval 使用 GT depth 合成 measurement，是 oracle/simulation forward，不是真实 measurement-only deployment。

Codex review 发现：

- `BEST_CHECKPOINT.md` 已加 warning，但旧的 full-scene 成功描述仍残留，下一轮必须改掉。
- Round 16 的 `captimgs_stats` 全零，measurement normalization 根因方向合理，但统计捕获还需修正为全 tile 聚合。
- 部分诊断指标只写入 JSON，没有按独立 CSV contract 落盘。
- Section 16 的 optical forward 引用应修正为当前代码里的 `y_sum` 累积路径。

下一轮目标：

1. 不训练，不换 checkpoint。
2. 修复诊断 reporting contract 和 `BEST_CHECKPOINT.md` stale 文字。
3. 修正 measurement stats 捕获，按 tile 聚合 norm 前后统计。
4. 做 inference-only ablation：checkpoint norm、none、minmax、低 foreground tile skip 及组合。
5. 若 inference-only 方案无法同时改善 SAM/pseudo-RGB/depth-baseline，则 Round 18 转入训练/架构改动。

### Round 15 Review Update

Claude 第十五轮完成 full-scene inference plumbing，并分别跑了 `deploy 1` 和 `deploy 16`：

- `deploy 1`: masked PSNR 30.02 dB, depth MAE 0.453m, SAM 0.540 rad。
- `deploy 16`: masked PSNR 18.96 dB, depth MAE 0.489m, SAM 0.417 rad。

Codex 复查后认为 Round 15 的成功结论不成立：

- 用户观察和 quicklook 显示 `est_hs.png` 是白底粉色糊状，颜色分离失败。
- depth quicklook 有结构但带强烈 tile/DOE 纹理，且 MAE 已退化到约 0.45-0.49m。
- deploy 1 的 masked PSNR 30.02 dB 与视觉质量冲突，说明必须审计 metric、mask、归一化和 per-band/SAM 指标。
- 当前 DoDo-depth full-scene eval 使用 GT depth 参与 optical forward measurement synthesis，必须明确这是 oracle/simulation forward contract，而不等同真实 measurement-only 联合重建。

下一轮目标：

1. 不跑训练，不换 checkpoint。
2. 将 Round 15 full-scene usability 标记为 pending audit，不再宣称最佳 checkpoint 已通过 full-scene 验收。
3. 为 `infer_contect.py` 增加诊断输出：原始数组、per-band metrics、foreground/background/interior/boundary metrics、constant-depth baseline、stitch weight map、ROI patch direct-vs-stitched 比较。
4. 审计 `torch_optics/forward_dodo.py`、`snapshotdepth_hs.py`、`datasets/hyperspectral_dataset.py`、`infer_contect.py` 的光学 forward contract 和 measurement normalization。
5. 在 Section 16 给出“训练 crop 指标与 full-scene 可视化差异”的证据链和 Round 17 建议。

### Round 14 Review Update

Claude 第十四轮完成 fine-tune checkpoint audit、Section 13 勘误和 `BEST_CHECKPOINT.md`。

已确认：

- FT-A/FT-B depth-best checkpoint 约 30.159 dB / 0.1805m，几乎等同 R12 depth-best，没有实质提升。
- FT-A/FT-B psnr-best checkpoint 约 31.02 dB，但 MAE 约 0.284m，不满足 depth 阈值。
- 没有 fine-tune checkpoint 满足 promotion rule（PSNR >= 30.4345 dB 且 MAE <= 0.22m）。
- 最终 depth-sensitive best checkpoint 固化为 R12 `depth-best-epoch=226.ckpt`：
  `infer_results/DoDo-change/DoDo_depth_finite_joint_metricdepth_260_v1/20260507_112631/checkpoints/depth-best-epoch=226.ckpt`

Codex 新增协作规则：

- `implementation-notes.md` 以后永远只保留最新 8 个顶级编号 section。
- 已写入 `AGENT.md`、`ai-collab/CLAUDE.md`、`ai-collab/CODEX.md`。

下一轮目标：

1. 不再进行同架构训练或 fine-tune。
2. 修剪 `implementation-notes.md`，新增 Section 15 后只保留 Sections 8-15。
3. 改造现有全幅推理脚本 `infer_contect.py`，使其支持 DoDo-depth 128x128 tiled full-scene inference。
4. 使用最终 best checkpoint 对完整高光谱/深度数据对做推理评估，必须同时跑 `Baek数据集/deploy 16` 和 `Baek数据集/deploy 1`，分别保存两个 deploy 的 PSNR/MAE，并输出 combined comparison summary。

### Round 13 Review Update

Claude 第十三轮完成：

- `metrics.json train_loss/*=0.0` 修复，并通过 1 epoch preflight 验证非零 train loss。
- `--validate_only_ckpt` / `--eval_tag` checkpoint evaluation 模式。
- R12 depth-best / PSNR-best 的 standalone eval。
- 两个从 R12 depth-best 初始化的 40 epoch 低 LR fine-tune。

已确认结果：

- R12 depth-best eval：PSNR 30.1345 dB，MAE 0.17836m。
- R12 PSNR-best eval：PSNR 31.0999 dB，MAE 0.49390m。
- FT-A/FT-B PSNR-best：约 31.02/31.03 dB，但 last MAE 约 0.339/0.340m。

Codex 发现一个 Section 13 记录错误：

- Section 13 把 FT-A/FT-B 的 last MAE 0.339/0.340m 写成了 best MAE。
- 实际日志显示两个 fine-tune 都保存了 `depth-best-epoch=000.ckpt`，MAE 约 0.1805m。
- 正确结论不是“fine-tune 立即完全破坏 depth”，而是“fine-tune 的 depth-preserving checkpoint 几乎不提升 PSNR；PSNR-best checkpoint 会明显破坏 depth”。

下一轮目标：

1. 不再跑训练。
2. 用 validation-only 模式独立评估四个 fine-tune checkpoint：FT-A depth-best、FT-A PSNR-best、FT-B depth-best、FT-B PSNR-best。
3. 在 `implementation-notes.md` 新增 Section 14，修正 Section 13 的 fine-tune 解读。
4. 写 `handoff/DoDo-change/BEST_CHECKPOINT.md`，固化当前最佳模型选择。

### Round 12 Review Update

Claude 第十二轮完成 checkpoint reproducibility cleanup、1 epoch preflight 和 `metric_depth_loss_weight=1` 的 260 epoch 主实验：

- Checkpoint contract 已修复：checkpoint 落到 `artifact_root/checkpoints/`，文件名稳定为 `psnr-best-epoch=*.ckpt` / `depth-best-epoch=*.ckpt`，不再有 fake `0.0000`。
- R12 Main：PSNR-best 31.10 dB，完全复现 R10 Main 的 31.11 dB。
- R12 depth-MAE-best：epoch 226 达到 0.178m，是当前历史最佳 depth MAE。
- R12 last/PSNR-best：epoch 259 depth MAE 回退到 0.494m，说明 220+ epoch 后 HS PSNR 与 depth MAE 竞争明显加剧。
- `nonfinite_count=0`，DOE/decoder grad finite，问题不是数值稳定性。

Codex 结论：

1. 当前最佳模型应是 `depth-best-epoch=226.ckpt`，不是 final/PSNR-best checkpoint。
2. 下一轮不要直接再跑 260 epoch，也不要从头扫大权重。
3. 先补 standalone checkpoint evaluation：分别重载 depth-best 和 PSNR-best，生成独立 `metrics.json` 和 quicklook，以便把 depth-best 的 PSNR/MAE 从日志推断固化为可复现 artifact。
4. `metrics.json` 的 `train_loss/*=0.0` 仍未修好。根因很可能是 `training_step` 保存了已加 `train_loss/` 前缀的 key，但 `_save_validation_artifacts` 按未加前缀 key 读取。
5. 若 checkpoint evaluation 确认 depth-best 约 PSNR >=30 dB 且 MAE <=0.20m，则注册 epoch 226 为当前 best model。之后最多允许两个从 depth-best 初始化的 40 epoch 低学习率 fine-tune，不再从头长训。

### Round 11 Review Update

Claude 第十一轮完成 artifact contract、metrics persistence、双 checkpoint、metric-depth loss，并运行两个 120 epoch Pareto candidates：

- Candidate A (`depth_loss_weight=2`, `metric_depth_loss_weight=0`)：PSNR 24.91 dB，depth-MAE-best 0.307m，last MAE 0.353m，`nonfinite_count=0`。
- Candidate B (`depth_loss_weight=1`, `metric_depth_loss_weight=1`)：PSNR 24.72 dB，depth-MAE-best 0.242m，last MAE 0.339m，`nonfinite_count=0`。

Codex 结论：

1. 120 epoch PSNR 约 24.5-24.9 dB 不能证明架构/数据上限。R10 Main 在 epoch 125 约 25.5 dB，epoch 259 达 31.11 dB，说明后半程仍可能大幅提升。
2. Candidate B 是当前最优 Pareto 候选。metric-depth loss 与验收指标对齐，在相近 PSNR 下比 IPS `depth_w=2` 得到更好的 depth MAE。
3. 暂不增加 `metric_depth_loss_weight` 到 2 或 5，先跑 `metric_depth_loss_weight=1` 的 260 epoch 长曲线。
4. 暂不加 depth-MAE early stopping。PSNR 仍在上升，early stopping 可能过早停止；用 PSNR-best 和 depth-MAE-best 双 checkpoint 分别报告。
5. 下一轮先做 checkpoint/artifact reproducibility cleanup：checkpoint 文件名目前显示 `0.0000`，容易误导；建议让 checkpoint 落到 resolved artifact root 的 `checkpoints/` 下，并用稳定文件名前缀区分 PSNR-best 与 depth-best。

下一轮目标：

- 修复 checkpoint filename/dirpath reproducibility issue。
- 1 epoch preflight 验证 artifact root、checkpoint、metrics、quicklooks 都正确落盘。
- 运行 `DoDo_depth_finite_joint_metricdepth_260_v1`，参数保持 Candidate B，只把 `max_epochs` 改为 260。
- 完成后更新 `implementation-notes.md` Section 12 并停止等待 Codex review。

### Round 10 Review Update

Claude 第十轮完成 finite-safe 260 epoch 主实验和一个 `depth_loss_weight=5` follow-up：

- 主实验：`validation/psnr_hs_masked = 31.11 dB`，`validation/mae_depth_m = 0.516m`，`nonfinite_count = 0`。
- Follow-up：`validation/mae_depth_m = 0.326m`，但 `validation/psnr_hs_masked` 降至 `24.53 dB`。
- DOE 和 decoder 梯度 finite，深度头可训练性已证实。
- 当前问题不再是 nonfinite 或深度头不可训，而是 HS 重建与 metric-depth MAE 的 Pareto trade-off。

下一轮优先级：

1. 先修复 `artifact_root` contract。Round 10 的 `command.txt` 没有包含 `--artifact_root`，`hparams.json` 也显示空字符串，导致 artifacts 落回默认 `version_0`。必须增加代码兜底和 `--require_artifact_root` fail-fast，并做 1 epoch preflight 验证。
2. 修复 `metrics.json` 中 main run final `train_loss/* = 0.0` 的持久化问题。
3. 增加 PSNR-best 与 depth-best 双 checkpoint，避免只按 `validation/psnr_hs_masked` 选择模型。
4. 设计保守 depth objective 改进：优先新增 opt-in masked metric-depth SmoothL1 loss，或运行较温和的 `depth_loss_weight=2` 短实验，不要直接再次长训 `depth_w=5`。
5. 下一轮最多运行两个 120 epoch short Pareto candidates；只有候选达到 `mae_depth_m <= 0.35m` 且 `psnr_hs_masked >= 27 dB` 或曲线明显可延长时，才考虑新的 260 epoch。

### Round 9 Review Update

Claude 第九轮完成了 DoDo numerical stability fix：

- `_normalize_once(eps=0)` 已修复为 safe normalization。
- synthetic foreground、zero input、real foreground batch 的 stage-wise finite smoke 均通过。
- finite preflight 通过，`nonfinite_count=0`，DOE/decoder grad finite。
- foreground one-batch overfit 通过，PSNR +3.39 dB，depth 不再常数化。
- 20 epoch short sanity 通过关键趋势验证：PSNR 从 5.62 到 15.03，`nonfinite_count=0`，HS/depth 输出不再常数化。

下一轮可以恢复正式长训练，但必须使用 finite-safe 设置，并持续 fail-fast：

```bash
--decoder_norm group
--dodo_measurement_norm per_sample_mean_std
--dodo_nonfinite_policy fail
--depth_smooth_weight 0
--dodo_doe_type New
--optimize_optics
```

注意：depth 仍是主要风险。第九轮 short sanity 的 `validation/mae_depth_m` 仍约 0.93m；下一轮必须重点分析 depth MAE、depth error quicklook 和 depth 输出动态范围。如果 HS 改善但 depth 仍差，只允许一个 `depth_loss_weight` 单变量 follow-up。

#### Required next experiment: finite-safe 260 epoch

```bash
--experiment_name DoDo_depth_finite_joint_260_v1
--optical_model dodo_depth
--dodo_doe_type New
--image_sz 128
--crop_width 0
--no-preinverse
--psf_loss_weight 0
--optimize_optics
--optics_lr 1e-6
--cnn_lr 5e-5
--noise_sigma_min 0
--noise_sigma_max 0
--batch_sz 2
--num_workers 4
--max_epochs 260
--decoder_norm group
--dodo_measurement_norm per_sample_mean_std
--dodo_nonfinite_policy fail
--depth_smooth_weight 0
--min_valid_ratio 0.2
```

通过条件：

- exit code 0。
- `logs/train.log` 非空。
- `nonfinite_count == 0`。
- best/last `validation/psnr_hs_masked` 明显高于 20 epoch short sanity。
- depth 不常数化：valid-mask `est_depth_std >= 0.02`。
- 记录 best/last `validation/mae_depth_m`。
- quicklook 显示 HS 有结构且 depth error 不全图饱和。
- DOE 与 decoder grad finite。

#### Optional one targeted follow-up

只允许一个 follow-up，按实际主实验结果选择：

- 如果 `nonfinite_count > 0` 或命令失败：不要 follow-up，停止并记录首个错误。
- 如果 HS PSNR 改善但 `validation/mae_depth_m` 仍高于 0.35m：运行 `depth_loss_weight=5` 单变量实验。
- 如果 depth 输出再次常数化：运行 foreground one-batch depth-focused overfit 诊断，不跑长训。
- 如果 HS 和 depth 都稳定改善：不做 follow-up，停止等待 Codex review。

推荐 depth follow-up 参数：

```bash
--experiment_name DoDo_depth_finite_joint_depthw5_v1
--optical_model dodo_depth
--dodo_doe_type New
--image_sz 128
--crop_width 0
--no-preinverse
--psf_loss_weight 0
--optimize_optics
--optics_lr 1e-6
--cnn_lr 5e-5
--noise_sigma_min 0
--noise_sigma_max 0
--batch_sz 2
--num_workers 4
--max_epochs 120
--decoder_norm group
--dodo_measurement_norm per_sample_mean_std
--dodo_nonfinite_policy fail
--depth_smooth_weight 0
--depth_loss_weight 5
--min_valid_ratio 0.2
```

完成后更新 `implementation-notes.md` Section 10，记录命令、artifact、日志、best/last PSNR、best/last metric depth MAE、final train loss、nonfinite count、grad finite 状态、quicklook 定性分析和下一步建议。

### Round 8 Review Update

Claude 第八轮完成了 diagnostics、artifact、GroupNorm、measurement normalization、baseline one-batch、repair one-batch 和 short sanity。但 short sanity 仍失败：

- `validation/psnr_hs_masked ~= 6.47`，提升极小。
- `est_depth_std = 0`，深度仍常数化。
- `est_image_max == est_image_min`，HS 输出近常数。
- `nonfinite_count = 360`。
- `captimgs` 最终统计为全 0。

新的阻断结论：问题不只是全背景 one-batch，也不只是 decoder norm。日志显示即使 `mask_ratio > 0`、`input_spectral_sum > 0`，DoDo `captimgs` 仍会整张非有限，随后被 `nan_to_num(..., 0)` 变成零输入。下一轮必须先修 DoDo measurement 非有限根因。

#### 本轮新增硬性要求

- 不要继续 260 epoch。
- 不要继续在 `captimgs` 非有限后替换为 0 的状态下训练。
- 不要只调 `depth_loss_weight`、`image_loss_weight` 或 decoder。
- 先做 stage-wise finite smoke，定位 DoDo forward 哪一段产生 NaN/Inf。
- 修复 DoDo measurement normalization/finite policy 后，所有后续实验必须满足 `nonfinite_count == 0`。
- 前景 one-batch overfit 必须记录 `mask_ratio >= 0.2`、`input_spectral_sum > 0`、`captimgs` finite 且非零。

#### Required next code fixes

1. DoDo safe normalization。
   - 修复 `torch_optics.forward_dodo._normalize_once(eps=0)` 的除零风险。
   - 增加 eps，建议默认 `1e-8`。
   - 优先支持 per-sample normalization，避免 batch 中某个全零样本污染其它样本。
   - 如果 `y_sum` 在 normalization 前已非有限，必须记录 stage 并 fail，不要直接吞掉。
2. DoDo nonfinite policy。
   - 增加 CLI，例如 `--dodo_nonfinite_policy zero|fail|skip_batch`。
   - 默认可保持 `zero` 兼容旧行为，但下一轮所有实验必须用 `--dodo_nonfinite_policy fail` 或 `skip_batch`。
   - 本轮通过条件是 `nonfinite_count == 0`，不是“替换后继续”。
3. Stage-wise finite diagnostics。
   - 增加一个轻量 debug path 或脚本，检查 synthetic nonzero input 和真实 foreground batch。
   - 至少记录：after prop1、doe1、prop2、prop3、sensing、final normalize 的 finite、min、max、mean、std。
   - 如果某 stage 首次出现非有限，停止训练并记录。
4. Foreground sampling。
   - 让 hparams 与实际 crop 行为一致。当前 `hparams.randcrop=False` 但 train dataset 实际 `randcrop=True`，必须修正记录。
   - 为 DoDo overfit 增加 foreground batch guarantee：batch `mask_ratio >= 0.2`，`input_spectral_sum > 0`。
   - 如果 dataset 在 retries 后仍找不到合格 crop，不要 silent fallback 到全背景；应选择 best valid crop 或 fail with clear error。
5. Backward diagnostics。
   - 把 input_adapter、backbone、HS head、depth head、DOE zernike grad norm 记录移到 backward 之后，例如 Lightning `on_after_backward`。
   - 持久化 grad finite 状态到 metrics。

#### Required next experiments

Phase A finite smoke, no training:

- Synthetic foreground input：随机正 HS、depth=1m、mask=1。
- Real foreground batch：必须 `mask_ratio >= 0.2`，`input_spectral_sum > 0`。
- 两者都要求 DoDo output finite、非零、std > 0。

Phase B diagnostics preflight:

- 使用 `--dodo_nonfinite_policy fail`。
- 使用 `--decoder_norm group` 和 `--dodo_measurement_norm per_sample_mean_std`。
- 通过条件：exit code 0、`nonfinite_count == 0`、DOE/decoder grad finite。

Phase C foreground one-batch overfit:

- 必须使用真实前景 batch，不允许全背景。
- 必须记录 batch id、mask_ratio、spectral_sum、captimgs stats。
- 通过条件：PSNR 提升 >= 2 dB，`est_depth_std >= 0.02`，`nonfinite_count == 0`，grad finite。

Phase D short sanity 20 epoch:

- 仅在 Phase C 通过后运行。
- 通过条件：`nonfinite_count == 0`，HS/depth 不常数化，PSNR 有明显上升趋势。

完成 Phase A-D 后停止，等待 Codex review。

上一轮 Claude 完成了部分 Phase 0 和一次 DOE-optimized preflight，但没有完成原长训练 stop condition。

已完成：

- `--dodo_doe_type New` 已接入 `DepthAwareDoDoForwardModel(..., doe_type_a=...)`。
- DOE preflight 使用 `--optimize_optics` 和 `--dodo_doe_type New`，exit code 为 0。
- 日志确认 `doe1.zernike_coeffs.requires_grad=True`。
- `validation/mae_depth_m`、`validation/psnr_hs_masked` 已有基础实现。
- `checkpoint_monitor` / `checkpoint_mode` CLI 已有基础实现。
- DoDo `captimgs` 已有 NaN/Inf guard 和 `nan_to_num` 兜底。

未完成或需要修正：

- 主实验没有完成 260 epoch；用户提前停止，实际到 epoch 62。
- 当前深度输出常数化：`est_depth_max == est_depth_min ~= 0.5038`。
- Phase 4 one-batch overfit 尚未执行。
- 主实验 `logs/train.log` 为空，不满足长训练日志规则。
- Artifact contract 未完成：缺 `git_status.txt`、quicklook PNG、同一 artifact root 下的 `logs/train.log`。
- 没有验证 backward 后 `camera.doe1.zernike_coeffs.grad is not None` 且 finite。
- 没有按 identity 验证 `doe1.zernike_coeffs` 在 optics optimizer group 中。
- 没有持久化记录 clamp hook 是否执行。
- 非有限 guard 未记录 spectral sum、depth range，也未持久化 nonfinite counter。
- `validation/psnr_hs_masked` 缺 shape assert；`mae_depth_m` 的 no-valid-pixel epoch 聚合仍可能误导。
- `metrics.json` 中 final train loss 不可靠，主实验里 train loss 为 0，但 TensorBoard 事件文件中为非零。

本轮下一步不要直接恢复 260 epoch 主训练。先补齐可追溯 diagnostics 与 artifact 保存，再运行 one-batch overfit 判断深度头和数据流是否能在单 batch 上收敛。

用户追问后修正：原方案主要是 diagnostics，没有强制要求对 60+ epoch 后 HS/depth 基本失败的问题做代码级针对性修复。现在要求 Claude 在 baseline one-batch 失败后必须实施 targeted repair patch，并用短实验验证 repair 是否改善光谱与深度。

验收顺序：

1. 修复日志、artifact、DOE grad/param group/clamp、nonfinite diagnostics、metric 边界条件。
2. 重新运行 DOE-optimized preflight，并确认所有 diagnostics 出现在日志和 artifact 中。
3. 运行 one-batch overfit 80 epoch。
4. 如果 baseline one-batch 失败，实施 targeted repair patch，再运行 repair one-batch。
5. 如果 repair one-batch 成功，运行一个不超过 20 epoch 的短 sanity train/val；如果 repair one-batch 仍失败，停止并记录根因。
6. 更新 `implementation-notes.md` 后停止，等待 Codex review。

### Required fixes before next experiments

- 增加单一 artifact root，推荐 CLI：`--artifact_root "$EXP_ROOT"`。
- Artifact root 必须包含 `metrics.json`、`hparams.json`、`command.txt`、`git_status.txt`、`logs/train.log` 和 validation quicklook PNG。
- Required PNGs：`capt_rgb.png`、`gt_hs_rgb.png`、`est_hs_rgb.png`、`gt_depth_m.png`、`est_depth_m.png`、`depth_abs_error_m.png`。
- 所有实验使用非空日志，建议 `conda run -n ld_clean --no-capture-output python ... > "$EXP_ROOT/logs/train.log" 2>&1`。
- 记录 `doe1.zernike_coeffs.requires_grad=True`。
- 用 object identity 验证 `doe1.zernike_coeffs` 在 optimizer `optics` param group。
- backward 后记录 `doe1.zernike_coeffs.grad is not None`、grad finite、grad norm。
- 记录 `camera.clamp_parameters_()` 是否在 `--optimize_optics` 下执行。
- DoDo `captimgs` 非有限时记录 nonfinite count、mask ratio、input spectral sum、`depth_metric` min/max、global step，并持久化 counter。
- `validation/psnr_hs_masked` 前检查 `est_images.shape == target_images.shape`。
- `validation/mae_depth_m` 对 no-valid-pixel batch 做 skip 计数，不要污染 epoch 平均。

### Effect-targeted repair patch if baseline one-batch fails

Baseline one-batch 失败定义，满足任一条即可：

- 80 epoch 后 `validation/psnr_hs_masked` 提升小于 2 dB。
- `validation/mae_depth_m` 下降小于 30%。
- 有效 mask 内 `est_depth_std < 0.02`，仍接近常数深度。
- `est_images` 仍接近常数或动态范围明显小于 GT。
- decoder depth/HS head 梯度缺失或梯度极小。

若失败，Claude 必须做以下最小代码修正，目标是修复可学习性，而不是改光学公式：

1. 小 batch normalization repair。
   - 给 DoDo decoder 增加 CLI，例如 `--decoder_norm batch|group`。
   - 默认保持 `batch`，避免破坏 legacy 行为。
   - DoDo repair 实验必须使用 `--decoder_norm group`。
   - `models/simple_model_mamba.py` 的 DoDo input adapter 和 `nets/mamba_unet.py` 的 `DoubleConv` 需要支持 GroupNorm。
   - 不要修改 `optics/`。
2. DoDo measurement scaling repair。
   - 增加 CLI，例如 `--dodo_measurement_norm none|per_sample_mean_std|per_sample_minmax`。
   - 默认 `none`，保持旧行为。
   - Repair 实验优先用 `per_sample_mean_std`。
   - 标准化位置应在 DoDo `captimgs` NaN/Inf guard 之后、decoder 之前；不要改变光学模型内部。
   - 记录标准化前后的 `captimgs` min/max/mean/std。
3. Depth 常数化 diagnostics。
   - 记录有效 mask 内 `target_depth_std`、`est_depth_std`、`depth_logits_std`。
   - 记录 depth head、HS head、input adapter 的 grad norm。
   - 若 depth head grad 为 0/None，直接停止并记录为阻断。
4. Overfit repair loss 设置。
   - repair one-batch 必须显式使用 `--depth_smooth_weight 0`，先排除平滑正则奖励常数解。
   - 不要同时大幅改多个 loss 权重。若需要权重实验，只允许在 repair one-batch 后单独增加 `--depth_loss_weight 5`，且必须记录为单变量变更。

### Required next experiments

Phase A diagnostics preflight:

```bash
--experiment_name DoDo_depth_preflight_optics_diag_v1
--optical_model dodo_depth
--dodo_doe_type New
--image_sz 128
--crop_width 0
--no-preinverse
--psf_loss_weight 0
--optimize_optics
--optics_lr 1e-6
--cnn_lr 5e-5
--noise_sigma_min 0
--noise_sigma_max 0
--batch_sz 1
--num_workers 0
--max_epochs 1
--limit_train_batches 2
--limit_val_batches 2
```

Phase B baseline one-batch overfit:

```bash
--experiment_name DoDo_depth_overfit_onebatch_optics_v1
--optical_model dodo_depth
--dodo_doe_type New
--image_sz 128
--crop_width 0
--no-preinverse
--psf_loss_weight 0
--optimize_optics
--optics_lr 1e-6
--cnn_lr 5e-5
--noise_sigma_min 0
--noise_sigma_max 0
--batch_sz 1
--num_workers 0
--max_epochs 80
--limit_train_batches 1
--limit_val_batches 1
```

Phase C repair one-batch overfit, only if Phase B fails:

```bash
--experiment_name DoDo_depth_overfit_onebatch_repair_norm_v1
--optical_model dodo_depth
--dodo_doe_type New
--image_sz 128
--crop_width 0
--no-preinverse
--psf_loss_weight 0
--optimize_optics
--optics_lr 1e-6
--cnn_lr 5e-5
--noise_sigma_min 0
--noise_sigma_max 0
--batch_sz 1
--num_workers 0
--max_epochs 80
--limit_train_batches 1
--limit_val_batches 1
--decoder_norm group
--dodo_measurement_norm per_sample_mean_std
--depth_smooth_weight 0
```

Phase D short sanity train/val, only if Phase C succeeds:

```bash
--experiment_name DoDo_depth_repair_short20_v1
--optical_model dodo_depth
--dodo_doe_type New
--image_sz 128
--crop_width 0
--no-preinverse
--psf_loss_weight 0
--optimize_optics
--optics_lr 1e-6
--cnn_lr 5e-5
--noise_sigma_min 0
--noise_sigma_max 0
--batch_sz 2
--num_workers 4
--max_epochs 20
--decoder_norm group
--dodo_measurement_norm per_sample_mean_std
--depth_smooth_weight 0
```

Phase B/C/D 完成后停止。不要在本轮启动 260 epoch。

## 当前状态

新的 DoDo depth-aware 光学前向已经接入端到端重建网络：

- 光学模型：`torch_optics.forward_dodo.DepthAwareDoDoForwardModel`
- 训练入口：`snapshotdepth_trainer_hs.py`
- Lightning 模型：`snapshotdepth_hs.py`
- Dataset：`datasets/hyperspectral_dataset.py`
- 当前 decoder：`models.simple_model_mamba.SimpleModelHS`
- 旧光学路径：`legacy_camera` 保留，`optics/` 未修改

已验证：

- `dodo_depth` fast-dev run：1 train + 1 val batch 完成，无崩溃。
- `dodo_depth` 输出 `captimgs=(B,3,128,128)`，decoder 输出 `est_images=(B,25,128,128)` 和 `est_depthmaps=(B,128,128)`。
- `depth_metric` 用于光学前向，`depth_map` 保留为 IPS `[0,1]` 深度监督。

本轮用户要求：

- 正式训练必须开启嵌入 DOE 优化：使用 `--optimize_optics`。
- DoDo DOE 与重建网络同时训练。
- DoDo 第一片 DOE 必须是可训练 Zernike 参数，不能使用默认 frozen `Zeros`。
- 训练轮数先设为 `260` epoch。
- 训练过程终端输出必须写入日志文件，避免 Claude 在等待长训练时读取大量无用输出浪费 tokens。
- Claude 启动训练后只需等待实验结束，再进行结果验证、图像分析和下一步判断。

Review file:

- `reviews/review-DoDo-change-long-training-plan-2026-05-05.md`
- `reviews/review-DoDo-change-doe-trainability-2026-05-05.md`

## Critical Correction: 当前默认不会训练 zernike_coeffs

Codex 核对当前代码后确认：

- `snapshotdepth_hs.py` 构造 `DepthAwareDoDoForwardModel(...)` 时没有传 `doe_type_a`。
- `DepthAwareDoDoForwardModel` 默认 `doe_type_a="Zeros"`。
- `torch_optics.doe.DOELayer` 的 `Zeros` 分支将 `zernike_coeffs` 创建为 `requires_grad=False`。

因此，只有 `--optimize_optics` 不足以训练 zernike coefficients。`--psf_loss_weight 0` 不是问题；问题是 DOE 类型默认 frozen。

Claude 必须在任何长训练前修正：

1. 给 `dodo_depth` 增加 CLI 参数，例如：

```bash
--dodo_doe_type New
```

2. 在 `snapshotdepth_hs.py` 构造 `DepthAwareDoDoForwardModel` 时传入：

```python
doe_type_a=hparams.dodo_doe_type
```

3. 正式训练和 preflight 命令必须包含 `--dodo_doe_type New`。
4. preflight 必须打印/记录：

```text
camera.doe1.zernike_coeffs.requires_grad = True
camera.doe1.zernike_coeffs.grad is not None after backward
```

5. 如果 Claude 已经启动了缺少 `--dodo_doe_type New` 的训练，必须停止该训练，记录原因，然后按本 TASK 修正后重启。

## 总目标

建立可复现的长时间训练-验证-分析-迭代闭环，提高：

- 高光谱重建：主要指标 `PSNR`，默认使用 masked HS PSNR。
- 深度重建：主要指标 `MAE`，单位必须是 meters。

## 非目标

- 不重新设计 DoDo 光学物理公式。
- 不修改 `optics/`。
- 不提交 checkpoint、日志目录、训练图像、缓存或大文件到仓库。
- 不把 IPS depth MAE 当作最终深度指标。
- 不在训练过程中反复读取完整日志。

## Phase 0: 训练前必须补齐

Claude 必须先确认或实现以下内容。如果已实现，直接验证并记录。

### 0.1 米制深度 MAE

新增/确认 validation metric：

- `validation/mae_depth_m`
- 可选：`validation/mae_depth_ips`

计算：

```text
metric_est = ips_to_metric(clamp(est_depthmaps,0,1), min_depth, max_depth)
metric_gt = ips_to_metric(clamp(target_depthmaps,0,1), min_depth, max_depth)
mae_depth_m = mean(abs(metric_est - metric_gt) over final_mask)
```

要求：

- 使用 `util.helper.ips_to_metric`。
- 只在 `final_mask > 0.5` 的有效区域统计。
- 有效像素为 0 时跳过或记录诊断，不要产生误导性 0。

### 0.2 高光谱 PSNR

新增/确认 validation metric：

- `validation/psnr_hs_masked`
- 可选：`validation/psnr_hs_full`

计算：

```text
mse = mean((est_images - target_images)^2 over HS channels and valid pixels)
psnr = 10 * log10(1.0 / (mse + eps))
```

要求：

- 默认以 `validation/psnr_hs_masked` 作为主要指标。
- `est_images.shape == target_images.shape`，否则抛清晰错误。

### 0.3 保存验证重建结果

每个实验至少保存：

- `metrics.json` 或 `metrics.csv`
- `hparams.json`
- `command.txt`
- `git_status.txt` 或简短 dirty 状态
- `logs/train.log`
- 若干 val 样本快视图：
  - `capt_rgb.png`
  - `gt_hs_rgb.png`
  - `est_hs_rgb.png`
  - `gt_depth_m.png`
  - `est_depth_m.png`
  - `depth_abs_error_m.png`
  - 可选：`hs_abs_error_rgb.png`

推荐目录：

```text
infer_results/DoDo-change/<experiment_name>/<timestamp>/
```

不要保存完整大体积 HS cube，除非用户明确要求。

### 0.4 DoDo measurement 非有限防护

已知全零输入可能产生 NaN。

要求：

- 在 `dodo_depth` 路径检测 `captimgs` 是否 finite。
- 非 finite 时记录有效 mask ratio、输入 spectral sum、depth range。
- 训练时可用 `torch.nan_to_num(captimgs, nan=0.0, posinf=0.0, neginf=0.0)` 继续，但必须计数并写入日志/metrics。
- 不允许 NaN/Inf 静默进入 decoder/loss。

### 0.5 DoDo optics clamp hook

因为本轮正式训练开启 `--optimize_optics`，必须保证 DoDo DOE 参数投影实际执行。

要求：

- 确认 `self.camera.doe1.zernike_coeffs.requires_grad == True`。
- 确认 optimizer 的 `optics` param group 包含 `doe1.zernike_coeffs`。
- 在一次 backward 后确认 `doe1.zernike_coeffs.grad is not None` 且 finite。
- optimizer step 后，如果 `self.hparams.optimize_optics` 且 `self.camera` 有 `clamp_parameters_()`，调用它。
- 记录 clamp hook 是否执行。
- 如果能记录 DOE 参数 norm/range，写入 metrics 或日志；至少写入 implementation notes。

### 0.6 checkpoint monitor

建议新增或确认 CLI：

- `--checkpoint_monitor`
- `--checkpoint_mode`

本轮主实验推荐：

- monitor：`validation/psnr_hs_masked`
- mode：`max`

若暂不改 CLI，必须说明实际 best checkpoint 如何选择。

## Phase 1: 日志重定向 preflight

目的：确认指标、artifact 保存、NaN 防护、checkpoint monitor、日志重定向都可用。

命令必须将 stdout/stderr 写入日志文件。示例：

```bash
EXP_ROOT="infer_results/DoDo-change/DoDo_depth_preflight_optics_v1/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$EXP_ROOT/logs"
cat > "$EXP_ROOT/command.txt" <<'EOF'
python snapshotdepth_trainer_hs.py \
  --experiment_name DoDo_depth_preflight_optics_v1 \
  --optical_model dodo_depth \
  --dodo_doe_type New \
  --image_sz 128 \
  --crop_width 0 \
  --no-preinverse \
  --psf_loss_weight 0 \
  --optimize_optics \
  --optics_lr 1e-6 \
  --cnn_lr 5e-5 \
  --noise_sigma_min 0 \
  --noise_sigma_max 0 \
  --batch_sz 1 \
  --num_workers 0 \
  --max_epochs 1 \
  --limit_train_batches 2 \
  --limit_val_batches 2
EOF
stdbuf -oL -eL bash "$EXP_ROOT/command.txt" > "$EXP_ROOT/logs/train.log" 2>&1
```

如果 `bash "$EXP_ROOT/command.txt"` 不适配当前 shell，Claude 可直接执行 `python ... > "$EXP_ROOT/logs/train.log" 2>&1`，但必须保存实际命令。

Preflight 通过条件：

- 命令完成且 exit code 为 0。
- 日志文件存在。
- `doe1.zernike_coeffs.requires_grad=True`。
- `doe1.zernike_coeffs.grad` 在 backward 后非空且 finite。
- 日志/metrics 中出现 `validation/psnr_hs_masked` 和 `validation/mae_depth_m`。
- artifact 目录中有 metrics、hparams/command 和至少 1 个 val 样本图像。
- loss、PSNR、MAE_m 没有 NaN/Inf。
- `--optimize_optics` 下 camera 参数组存在，clamp hook 执行。

## Phase 2: 主实验，DOE + 重建网络共同训练 260 epoch

目的：正式训练嵌入 DoDo DOE 与重建网络，观察 PSNR 和米制深度 MAE。

推荐参数：

- `--optimize_optics`
- `--optics_lr 1e-6`
- `--cnn_lr 5e-5`
- `--max_epochs 260`
- `--batch_sz 2`
- `--num_workers 4`
- `--noise_sigma_min 0`
- `--noise_sigma_max 0`

说明：

- 第一轮主实验先用无噪声，减少变量，确认 DOE+decoder 联合优化上限。
- 如果显存不足，先把 `batch_sz` 降到 1，不要改其他变量。
- 如果 loss 爆炸或 NaN，停止并记录，不要继续加大学习率。

示例命令：

```bash
EXP_ROOT="infer_results/DoDo-change/DoDo_depth_optics_joint_260_v1/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$EXP_ROOT/logs"
cat > "$EXP_ROOT/command.txt" <<'EOF'
python snapshotdepth_trainer_hs.py \
  --experiment_name DoDo_depth_optics_joint_260_v1 \
  --optical_model dodo_depth \
  --dodo_doe_type New \
  --image_sz 128 \
  --crop_width 0 \
  --no-preinverse \
  --psf_loss_weight 0 \
  --optimize_optics \
  --optics_lr 1e-6 \
  --cnn_lr 5e-5 \
  --noise_sigma_min 0 \
  --noise_sigma_max 0 \
  --batch_sz 2 \
  --num_workers 4 \
  --max_epochs 260
EOF
stdbuf -oL -eL bash "$EXP_ROOT/command.txt" > "$EXP_ROOT/logs/train.log" 2>&1
```

训练等待规则：

- Claude 启动训练后，不要持续读取训练输出。
- 记录 PID、`EXP_ROOT`、`logs/train.log` 路径。
- 每 20-30 分钟最多检查一次进程是否结束，例如 `ps -p <PID>`。
- 如需健康检查，只读取日志最后 50-100 行。
- 不要读取完整 train.log。
- 训练结束后再读取日志尾部、metrics、checkpoint 和 artifact。

## Phase 3: 训练结束后的结果验证与分析

训练结束后 Claude 必须收集：

- exit code
- `EXP_ROOT`
- `logs/train.log`
- best checkpoint path
- best epoch
- best `validation/psnr_hs_masked`
- best/last `validation/mae_depth_m`
- final train loss
- 是否发生非 finite measurement
- `doe1.zernike_coeffs.requires_grad` 和 grad 检查结果
- clamp hook 是否执行
- 保存图像路径

图像分析至少包括：

- HS 重建是否模糊、偏色、动态范围不足。
- depth 是否常数化、边界错误、远近反转。
- `depth_abs_error_m.png` 误差集中在哪里。

## Phase 4: 最多一个针对性迭代实验

主实验结束后，Claude 最多运行一个针对性迭代实验，然后停止等待 Codex review。不要多变量同时变化。

选择规则：

1. 如果训练 loss 不下降或输出常数：
   - 运行 one-batch overfit，确认数据流和模型容量。
2. 如果出现 NaN/loss 爆炸：
   - 降低 `optics_lr` 到 `1e-7`，保持 `cnn_lr=5e-5`。
3. 如果 HS PSNR 低但 depth MAE_m 尚可：
   - 增加 `image_loss_weight` 或降低 `depth_loss_weight`，只改一个。
4. 如果 depth MAE_m 高但 HS PSNR 尚可：
   - 增加 `depth_loss_weight` 或加入 metric-depth loss，优先只增加 `depth_loss_weight`。
5. 如果 clean joint 训练正常：
   - 才考虑 noise robustness，`noise_sigma_min/max=0.0005/0.002`，不要直接跳最大噪声。

可选实验命令示例：

### Option A: one-batch overfit

```bash
python snapshotdepth_trainer_hs.py \
  --experiment_name DoDo_depth_overfit_onebatch_optics_v1 \
  --optical_model dodo_depth \
  --dodo_doe_type New \
  --image_sz 128 \
  --crop_width 0 \
  --no-preinverse \
  --psf_loss_weight 0 \
  --optimize_optics \
  --optics_lr 1e-6 \
  --cnn_lr 5e-5 \
  --noise_sigma_min 0 \
  --noise_sigma_max 0 \
  --batch_sz 1 \
  --num_workers 0 \
  --max_epochs 80 \
  --limit_train_batches 1 \
  --limit_val_batches 1
```

### Option B: lower optics lr

```bash
python snapshotdepth_trainer_hs.py \
  --experiment_name DoDo_depth_optics_joint_260_lr1e7_v1 \
  --optical_model dodo_depth \
  --dodo_doe_type New \
  --image_sz 128 \
  --crop_width 0 \
  --no-preinverse \
  --psf_loss_weight 0 \
  --optimize_optics \
  --optics_lr 1e-7 \
  --cnn_lr 5e-5 \
  --noise_sigma_min 0 \
  --noise_sigma_max 0 \
  --batch_sz 2 \
  --num_workers 4 \
  --max_epochs 260
```

### Option C: mild noise robustness

```bash
python snapshotdepth_trainer_hs.py \
  --experiment_name DoDo_depth_optics_joint_260_noise_mild_v1 \
  --optical_model dodo_depth \
  --dodo_doe_type New \
  --image_sz 128 \
  --crop_width 0 \
  --no-preinverse \
  --psf_loss_weight 0 \
  --optimize_optics \
  --optics_lr 1e-6 \
  --cnn_lr 5e-5 \
  --noise_sigma_min 0.0005 \
  --noise_sigma_max 0.002 \
  --batch_sz 2 \
  --num_workers 4 \
  --max_epochs 260
```

## Claude 输出要求

更新 `handoff/DoDo-change/implementation-notes.md`，包括：

- 本轮补齐的指标、artifact 保存、NaN 防护、clamp hook、checkpoint monitor。
- 实际运行命令。
- 日志文件路径。
- 每个实验结果表：
  - experiment name
  - epochs
  - best epoch
  - best `validation/psnr_hs_masked`
  - best/last `validation/mae_depth_m`
  - final train loss
  - checkpoint path
  - artifact path
  - log path
- 保存图像的定性分析。
- 如果训练失败：exit code、日志尾部关键错误、下一步最小修复建议。

## Stop Condition

Claude 完成以下内容后停止：

1. Phase 0 训练基础设施补齐。
2. Phase 1 preflight 通过。
3. Phase 2 `--optimize_optics` 260 epoch 主实验结束。
4. 按 Phase 3 完成分析。
5. 最多执行 1 个 Phase 4 针对性实验。
6. 更新 `handoff/DoDo-change/implementation-notes.md`。
7. 不要继续开更多实验，等待 Codex review。
