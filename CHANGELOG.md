# Changelog

本文件记录项目的重要变更。

格式参考 [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)，
版本号遵循 [Semantic Versioning](https://semver.org/)。

## [0.1.0] - 2026-08-05

首个公开版本。

### Added

- Rust 终端 Agent：Ratatui TUI 与行命令式 CLI/REPL，共用同一会话引擎
- 多 Provider 配置（API Key / Base URL / 协议），模型展示名、工具调用、并行工具请求、推理等级与文本详细度
- 专家 Agent 独立模型路由；`@` 委托探索、修复、设计、检索、综合等专家，并支持子会话浏览与回到父会话
- 权限模式 `safe` / `default` / `auto` / `yolo`：读写与命令按模式自动放行、询问或全放行；`default` / `auto` 支持会话内「始终允许」；`auto` 由 sticky reviewer 专家完成审批，并在子视图以请求/决策卡片呈现
- 内置工具：shell、文件系统读写、搜索、web fetch、git、代码 AST、工作流 todo / 自动续跑、记忆召回、skill 与 MCP 工具发现与调用
- 工具并行策略可配置；shell 输出、diff、todo、权限与子代理结果以结构化卡片展示
- 会话以追加写入的 JSONL transcript 持久化，支持恢复、历史树浏览，以及 TUI 内 undo / redo 与上下文压缩
- 运行时配置热重载；可选 Langfuse / OpenTelemetry 追踪
- TUI 主题、工具输出展开、滚动条与 `/` 本地命令补全

[0.1.0]: https://github.com/letr007/letcode/releases/tag/v0.1.0
