# 更新日志

本文件记录项目的重要变更。

格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [0.1.0] - 2026-08-05

首个公开版本。此前已用 letcode 自举开发，本标签冻结该可用基线。

### 新增

- 基于 Ratatui 的 TUI（opencode 风格）与行命令式 CLI/REPL，共用同一套会话引擎
- 多 Provider 模型路由、推理力度控制，以及专家 Agent 的模型覆盖
- 权限模式：`safe` / `default` / `auto` / `yolo`（读取配置时仍兼容旧值 `solo`）
- `auto` 模式下的 sticky reviewer 专家；reviewer 子视图以请求/决策卡片展示审批
- 工具面：shell、文件系统、搜索、web fetch、git、工作流、skill、子代理等
- 支持处理器声明的并行工具调用；default/auto 的 Ask 矩阵支持会话级 AllowAlways
- 追加写入的 JSONL 会话 transcript，支持恢复、历史树，以及 TUI 内 undo/redo
- 上下文压缩、支持项的运行时配置热重载，以及可选的 Langfuse/OpenTelemetry 追踪
- 可选 TUI 主题；工具、todo、权限与子代理结果的结构化卡片展示
- Default/Auto 对高风险 shell 命令走审批（人工或 reviewer），不做命令级硬黑名单
- Agent 提示词与面向操作者的引导文案使用中文

[0.1.0]: https://github.com/letr007/letcode/releases/tag/v0.1.0
