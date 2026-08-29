# Session

`SessionEngine` 连接前端输入、会话转录、AgentRunner 和会话事件流，提供命令入口 `SessionEngineIngress` 和事件出口 `SessionTransportEvent`。

## 启动

`SessionEngine::start` 接收 Agent、当前 `TranscriptRecorder`、模型标签和 `SessionEngineConfig`，先读取当前转录的会话 ID 与标题，生成 `SessionEngineProjection`：

- `session_id`
- `session_title`
- `model_id`
- `model_label`
- `permission_mode_label`
- `fast_mode_enabled`
- `api_key_configured`

随后启动三类运行资源：

1. 监听 MCP 配置文件的 watcher。
2. 异步执行 MCP server discovery。
3. 异步运行 `run_engine_loop`，并把 Agent、转录、模型路由、重试配置、MCP 配置、子代理运行时和两个 channel 交给该循环。

返回值为 `(SessionEngine, SessionEngineProjection)`。调用方通过 `take_ingress` 取得命令入口，通过 `take_event_egress` 取得事件接收器。引擎保存执行任务、MCP discovery 任务、配置 watcher 和当前转录。

`join` 的关闭顺序是：停止配置 watcher，等待引擎任务，终止并等待尚未结束的 MCP discovery 任务，最后删除当前路径上的空会话文件。任务 join 或转录清理失败会返回错误。

## 命令

命令解析产生 `CommandIntent`；后端相关 intent 通过 `SessionCommand::from_command_intent` 转换为 `SessionCommand`。展示、帮助和本地 UI 设置等 intent 不会转换为会话命令。

会话命令：

| 命令 | 参数 | 作用 |
| --- | --- | --- |
| `SubmitPrompt` | `UserMessageSubmission` | 提交用户提示 |
| `DelegateSubagent` | agent 名称、任务 | 启动子代理委派 |
| `Compact` | 无 | 执行手动上下文压缩 |
| `ShowHistoryTree` | 无 | 请求会话历史树 |
| `Undo` / `Redo` | 无 | 在当前会话中导航历史分支 |
| `NavigateHistory` | 条目 ID | 导航到指定历史条目 |
| `ViewChild` | 子会话导航、可选锚点 | 查看子会话转录 |
| `ViewParent` | 无 | 查看父会话转录 |
| `SetPermissionMode` | 权限模式 | 设置会话权限模式 |
| `AnchoredToggle` | 无 | 切换 anchored 状态 |
| `SetModel` | 模型 ID | 设置主模型 |
| `SetExpertModel` | agent 名称、模型 ID | 设置指定 expert 模型 |
| `SetExpertAllowedModels` | agent 名称、模型 ID 列表 | 设置指定 expert 可用模型 |
| `ToggleFastMode` | 无 | 切换 Fast Mode |
| `SetReasoningEffort` | `off`、`none`、`minimal`、`low`、`medium`、`high` 或 `xhigh` | 设置推理强度 |
| `SetFakeClient` | fake client 或空值 | 设置或关闭 fake client |
| `ResumeSession` | 会话 ID 或前缀 | 恢复会话 |
| `NewSession` | 无 | 创建新会话 |
| `ToggleMcpServer` | server 名称 | 启用或关闭 MCP server |
| `Interrupt` | 无 | 请求中断当前操作 |

可用的 slash 命令由 `src/command.rs` 中的 `COMMANDS` 描述。包含 `/help`、`/exit`、`/quit`、`/permission`、`/language`、`/model`、`/anchored`、`/agents`、`/fast`、`/reasoning`、`/thoughts`、`/tool-output`、`/scrollbar`、`/panel`、`/theme`、`/fake`、`/compact`、`/tree`、`/undo`、`/redo`、`/resume`、`/new`、`/context`、`/mcp`、`/skill`、`/child`、`/children` 和 `/parent`；部分条目是别名或仅在帮助中显示。

`SessionEngineIngress::submit` 将 `SessionCommand` 转为内部 `SessionEngineCommand`，写入无界控制 channel。`Interrupt` 使用独立的中断控制信号，`shutdown` 使用关闭控制信号。控制 channel 保持 FIFO 顺序。

命令在活动 turn 中按以下方式处理：

- `SubmitPrompt` 进入提示队列。
- `ViewChild` 和 `ViewParent` 立即处理。
- 权限、模型、推理强度、fake client、Fast Mode、MCP 等设置进入延迟队列。
- `DelegateSubagent`、压缩、历史树、撤销、重做、历史导航、新建会话和恢复会话在活动 turn 中产生拒绝或相应提示。
- `Interrupt` 请求取消当前操作。

延迟设置按命令类型合并；同一设置在当前 turn 结束前的后一次值替换前一次值。turn 结束后刷新延迟命令。

## Turn

提示执行由 `AgentRunner::run_prompt` 进入 `run_prompt_with_options`，每次执行使用一个 turn continuation queue。带转录时，Runner 为 Agent 安装 runtime snapshot provider；provider 从当前 JSONL 读取记录，并依据当前 context branch 投影运行时快照。

带用户输入的 turn 按以下顺序处理：

1. 发出 `UserMessage` 传输事件。
2. 将用户消息写入 `TranscriptRecorder`，并发出 context projection 更新。
3. 如果需要生成会话标题，异步生成标题；生成成功后写入转录并发出 `SessionTitleUpdated`。
4. 调用 `run_stream_content_with_interactions_async`。
5. 对 AgentEvent 持久化可持久化内容，并将流式结果转换为传输事件。
6. 成功时发出 context projection 更新、`AssistantDone` 和 `Done`。
7. 失败时写入转录错误，发出 `Error` 和 `Done`。

流式 AgentEvent 的处理包括：

- assistant 文本增量转换为 `AssistantDelta`。
- reasoning 增量和完成事件转换为 `ReasoningDelta`、`ReasoningDone`。
- 工具等待、开始、输出增量、结束和批次结束转换为 `ToolPending`、`ToolStarted`、`ToolOutputDelta`、`ToolFinished`、`ToolBatchFinished`。
- token usage 转换为 `TokenUsage`；请求准备阶段的估算转换为 `PreparedTokenUsage`。
- 上下文压缩开始、预览增量、无进展、提交和失败转换为对应压缩事件。
- retry 生命周期转换为 `RetryScheduled` 和 `RetryStarted`。
- todo、自动继续、Fast Mode 和模型流问题转换为对应状态或诊断事件。
- 权限请求通过一次性响应 channel 暂停 turn；问题请求通过 `RunnerQuestionRequest` 暂停 turn，前端提交响应后继续。

后台子代理完成后，Engine 将结构化结果安装到 Agent，发出 `BackgroundSubagentCompleted`，写入内部 continuation，并把 `ContinueSession` 放回命令队列。

## 事件

### SessionEvent

`SessionEvent` 包含：

- `Tick`
- `UserMessage`
- `ReasoningDelta`、`ReasoningDone`
- `AssistantDelta`、`AssistantDone`
- `TokenUsage`
- `ToolPending`、`ToolCancelled`、`ToolStarted`、`ToolFinished`、`ToolOutputDelta`、`ToolBatchFinished`
- `TodoSnapshot`、`AutoContinueChanged`
- `PermissionRequested`、`PermissionResolved`
- `ProcessIssue`、`Notice`
- `CompactionStarted`、`CompactionPreviewDelta`、`CompactionCommitted`、`CompactionNoProgress`、`CompactionFailed`
- `RuntimeContextUpdated`、`ContextTreeUpdated`、`ContextViewUpdated`、`ContextDetailOpened`、`ContextSummaryUpdated`
- `SessionStarted`、`SessionResumed`、`SessionTokenUsage`、`ContextBranchChanged`
- `RetryScheduled`、`RetryStarted`、`Interrupted`、`Error`、`Done`、`Quit`

`SessionTransportEvent` 是 Engine 的内部事件流类型，携带交互句柄和原始恢复数据，包括：

- `PreparedTokenUsage`、设置变更事件、模型目录事件和 MCP server 目录事件。
- `PermissionRequested`、`QuestionRequested` 及其 child 版本，事件中带有一次性响应句柄。
- `ChildSessionEvent`，用于标记 child session 的事件来源。
- `SessionStarted`，包含 session ID、转录记录、运行时上下文和 expert 模型。
- `SessionResumed`，包含 session ID、branch ID、消息、转录记录、证据数量、模型、token usage、运行时上下文和 expert 模型。
- `ChildSessionViewed`、`ParentSessionViewed` 和 `SessionHistoryLoaded`。
- `SessionTitleUpdated`、`Interrupted`、`Error` 和 `Done`。

`SessionTransportEvent::session_event` 将可公开为 `SessionEvent` 的事件转换为对应值；交互句柄、MCP 目录、会话查看原始记录等传输专用事件不转换。

TUI 从事件出口消费 `SessionTransportEvent`。assistant 增量先进入 typewriter；其余事件在当前增量完成后应用。每个 TUI frame 最多接收并处理 256 个会话事件，且处理时间预算为 4 ms。

## 恢复

### 新会话

`bootstrap_new_transcript` 创建 `TranscriptRecorder`，生成 `{session_id}.jsonl`，并写入 `SessionStarted` 记录。`prepare_new_session_package` 随后读取新转录，使用 `main` 根 branch 投影空的运行时恢复快照，并生成 `RuntimeActiveContext`。

新会话安装前会准备 context scope、校验运行时快照、准备权限重置和可选的模型路由。提交安装时交换 live recorder，安装运行时快照和 turn 序列，应用 context scope 与模型路由，并清理之前的空会话文件。

### 恢复已有会话

`prepare_resume_package` 执行以下步骤：

1. 读取 `{session_id}.jsonl` 及其 fingerprint。
2. 根据记录和默认恢复 cursor 投影运行时恢复快照，并解析 child sessions。
3. 在 fingerprint 仍匹配且文件尾部没有未提交事务时，以 append-safe 方式打开转录。
4. 让 recorder 采用快照中的 branch。
5. 如果快照包含活动 turn，取消未完成的工具调用和子代理运行，写入 `TurnInterrupted`，然后重新读取记录并重新投影快照。
6. 返回包含 session ID、记录、快照和 recorder 的 `PreparedResume`。

恢复提交前会准备并校验模型路由、运行时快照、会话 token usage、context scope 和 Fast Mode 状态。提交时：

- 交换 live recorder。
- 应用恢复的模型路由、权限模式、推理强度和 context scope。
- 安装已校验的运行时快照。
- 恢复最大 turn ID。
- 清理被替换的空会话文件。

恢复成功后发出 `SessionResumed`，其中包含恢复后的消息、记录、branch、模型、token usage、运行时上下文和 expert 模型。恢复期间发生错误会通过 `Error` 传输事件报告。

历史分支导航使用新的 branch 记录 `ContextBranchCreated`、`ContextCheckout` 和 `HistoryNavigation`，投影目标 branch 后以 `SessionResumed` 通知前端。父会话和子会话查看使用 `ParentSessionViewed` 或 `ChildSessionViewed`，这两类查看事件携带原始记录和运行时上下文。

## 会话索引

会话目录中的每个 `*.jsonl` 文件以文件名 stem 作为 session ID。`src/transcript/session_index.rs` 在同一目录维护 `sessions-index.json`，格式版本为 `1`，索引条目包含：

- 文件大小和修改时间（`size`、`mtime_ms`）。
- 记录数量、首尾时间戳。
- 模型、标题、最近的用户摘要和 assistant 摘要。
- 是否包含有效内容。

列出会话时先读取索引。若条目的大小和修改时间匹配，则直接生成 `SessionSummary`；若索引缺失、版本不匹配或条目过期，则重新扫描对应 JSONL。扫描期间若文件发生变化，该条目不会被写入为有效缓存。索引会移除已不存在的会话，并按最近时间戳降序返回会话摘要。索引通过临时文件写入后替换正式文件。

`/resume` 使用 session ID 或前缀解析目标会话；解析和恢复由 `resolve_session_prefix`、`prepare_resume_package` 以及 Engine 的 `ResumeSession` 分支共同完成。

## 源码索引

- `src/session/mod.rs`
- `src/session/engine.rs`
- `src/session/engine/control.rs`
- `src/session/command.rs`
- `src/session/ports.rs`
- `src/session/runner.rs`
- `src/session/event.rs`
- `src/session/events.rs`
- `src/session/lifecycle.rs`
- `src/session/restore.rs`
- `src/session/coordinator.rs`
- `src/session/interrupt.rs`
- `src/session/settings.rs`
- `src/transcript/journal.rs`
- `src/transcript/recorder.rs`
- `src/transcript/transcript_projection.rs`
- `src/transcript/session_index.rs`
- `src/tui/runtime/session_command_adapter.rs`
- `src/tui/runtime.rs`
