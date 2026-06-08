# Handoff

本目录用于保存任务交接信息。

## 文件说明

### NEXT_ACTION.md
当前调度入口。
定义：
- 当前由谁执行
- 当前 change 是什么
- 需要读取哪些文件
- 要做什么
- 输出到哪里
- 何时停止

### <change-id>/TASK.md
该 change 的任务拆分文档。
内容包括：
- 背景
- 目标
- 非目标
- 修改范围
- 开发步骤
- 验收标准

### <change-id>/implementation-notes.md
Claude 的实现记录。
内容包括：
- 实现了什么
- 改了哪些文件
- 跑了哪些测试
- 剩余问题

只保留最近 8 个顶级编号 section。

### <change-id>/CURRENT_CODE_CHANGES.md
Claude 当前轮代码修改台账。
内容包括：
- 当前轮是否有代码修改
- 每个修改文件路径
- 对应函数/类/代码区域
- 修改目的
- 修改内容摘要
- 潜在风险
- 验证方式

该文件只保留当前一轮 Claude 代码修改。Claude 开始新一轮实现前必须重置，Codex review 时必须读取。

### <change-id>/EXPERIMENTS.md
累计实验台账。
内容包括：
- 实验 ID、日期、关联 round
- 实验类型：preflight、training、checkpoint eval、full-scene inference 等
- 权重/checkpoint 保存路径
- artifact root
- 推理结果路径
- 关键指标
- 实验状态
- 一段自然语言总结，说明实验修改、目的和结论

该文件由 Claude 每次实验或推理完成后追加维护，不按 8 个 section 修剪。
