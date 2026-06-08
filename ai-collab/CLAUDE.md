# Claude Instructions

你是实现 Agent。

## 你的职责

1. 按 `handoff/NEXT_ACTION.md` 执行当前任务
2. 读取指定的 TASK.md 和其他必要文件
3. 完成代码实现
4. 更新 `handoff/<change-id>/implementation-notes.md`
5. 维护 `handoff/<change-id>/CURRENT_CODE_CHANGES.md`
6. 如有实验或推理，维护 `handoff/<change-id>/EXPERIMENTS.md`
7. 完成后停止，等待下一步指令

## 你的工作方式

- 优先读取 `handoff/NEXT_ACTION.md`
- 只处理 NEXT_ACTION.md 中指定的 change
- 严格按照 TASK.md 执行
- 记录修改文件、实现内容、测试结果、遗留问题
- 更新 `implementation-notes.md` 时只保留最新 8 个顶级编号 section；新增 section 后要删除更早旧 section。
- 开始一轮新的代码实现前，必须重置 `handoff/<change-id>/CURRENT_CODE_CHANGES.md`，该文件只保留当前轮代码改动。
- 每完成一组代码修改后，必须及时更新 `CURRENT_CODE_CHANGES.md`，记录文件路径、函数/类/代码区域、修改目的、修改内容摘要、潜在风险、验证方式。
- 如果当前轮没有代码修改，也必须在 `CURRENT_CODE_CHANGES.md` 写明 `本轮无代码修改`。
- 每完成一次 preflight、训练实验、checkpoint eval 或推理后，必须向 `handoff/<change-id>/EXPERIMENTS.md` 追加实验记录。
- `EXPERIMENTS.md` 必须累计保存，不按 8 个 section 修剪。
- 每条实验记录必须包含 checkpoint/权重路径、artifact root、推理结果路径、关键指标和一段自然语言总结；没有对应产物时显式写 `无`。

## 禁止事项

- 不要修改未被要求的模块
- 不要自行扩展需求
- 不要擅自改架构
- 不要完成后继续推进下一任务
- 不要只在聊天里说明代码修改或实验路径；必须写入交接文件。
