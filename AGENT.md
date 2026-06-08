# Repository Agent Protocol

本协议定义 Codex、Claude 和用户之间的仓库文件协作方式。
目标是让需求、实现、review 和下一步动作都落在仓库文件中，减少聊天上下文丢失和职责混乱。

## 核心原则

- Agent 不依赖口头记忆推进任务，必须读取仓库文件。
- 一次只推进一个明确的 change。
- 当前应该执行什么，以 `handoff/NEXT_ACTION.md` 为准。
- Codex 负责规划、任务拆分、Review。
- Claude 负责实现、记录实现结果。
- Claude 完成后必须停止，等待下一步指令。
- Codex Review 完成后必须写入 `reviews/` 或更新 `NEXT_ACTION.md`。

## 文件职责

### AGENTS.md
项目通用协作规则，Codex 和 Claude 都必须遵守。

### ai-collab/CODEX.md
Codex 的角色说明和行为约束。

### ai-collab/CLAUDE.md
Claude 的角色说明和行为约束。

### handoff/NEXT_ACTION.md
当前唯一调度入口。谁来执行、读什么、做什么、写什么，都在这里定义。

### handoff/<change-id>/TASK.md
某个 change 的任务拆分、目标、非目标、验收标准。

### handoff/<change-id>/implementation-notes.md
Claude 的实现记录，包括改了什么、测了什么、还有什么问题。

### handoff/<change-id>/CURRENT_CODE_CHANGES.md
Claude 当前一轮代码修改台账。
用途是让 Codex 能按文件和位置审查 Claude 的每一次代码改动。

规则：

- 只保留当前 Claude 执行轮的代码修改记录，不累计历史轮次。
- Claude 开始新一轮实现前必须重置该文件。
- Claude 每完成一组代码修改后，必须及时写入或更新该文件，而不是只在最终总结时补写。
- 每条记录必须包含：文件路径、函数/类/代码区域、修改目的、修改内容摘要、潜在风险、对应测试或验证方式。
- 如果本轮没有代码修改，Claude 必须明确写 `本轮无代码修改`。
- Codex review 时必须读取该文件，并把其中列出的修改点作为 review 主索引。

### handoff/<change-id>/EXPERIMENTS.md
实验台账，由 Claude 在每次实验或推理完成后追加维护。
用途是让用户和 Codex 能按实验轮次快速找到权重、推理结果和实验目的。

规则：

- 该文件累计保留，不按 8 个 section 修剪。
- 每完成一次 preflight、训练实验、checkpoint eval 或 full-scene inference，Claude 必须追加一条记录。
- 每条记录必须包含：实验 ID、日期、关联 round、实验类型、权重/checkpoint 保存路径、artifact root、推理结果路径、关键指标、状态。
- 每条记录必须包含一段自然语言总结，说明这次实验改了什么、为什么跑、结论是什么，便于之后只看描述就能回忆实验含义。
- 如果某次实验没有产生 checkpoint 或没有推理结果，必须显式写 `无`，不能留空。

## 记录保留规则

- `handoff/<change-id>/implementation-notes.md` 只保留最新 8 个顶级编号 section。
- Claude 新增一个 section 后，必须删除更早的旧 section，只保留最近 8 轮实现记录。
- Codex review 时如果发现该文件超过 8 个顶级编号 section，必须在 `NEXT_ACTION.md` 中要求 Claude 修剪。
- `handoff/<change-id>/CURRENT_CODE_CHANGES.md` 只保留当前一轮代码修改记录。
- `handoff/<change-id>/EXPERIMENTS.md` 是累计实验台账，不受 8-section 保留规则影响。

### reviews/review-xxx.md
Codex 的 Review 输出。
