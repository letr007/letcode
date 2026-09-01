# Session

`SessionEngine` 连接前端命令、会话 transcript、AgentRunner、resolved model route、子代理运行时和 session event stream。`SessionEngineIngress` 接收 frontend-neutral `SessionCommand`；`SessionTransportEvent` 向 TUI 或其它前端发布执行、恢复和交互状态。

## 启动与运行资源

`SessionEngine::start` 接收 Agent、当前 `TranscriptRecorder`、模型标签和 `SessionEngineConfig`，建立 `SessionEngineProjection`，然后启动：

1. 配置文件 watcher；
2. MCP server discovery；
3. 持有 Agent、recorder、runtime catalog、MCP 配置、subagent pool 和 channels 的 `run_engine_loop`。

调用方通过 `take_ingress` 提交命令，通过 `take_event_egress` 消费传输事件。`join` 先停止 watcher，再等待 engine 和 discovery 任务，最后清理当前路径上的空会话文件。

## 命令与活动 turn

命令解析先得到 `CommandIntent`，后端相关 intent 转换为 `SessionCommand`。`SubmitPrompt` 进入提示队列；`ViewChild`/`ViewParent` 可立即处理；权限、模型、reasoning、fake、Fast Mode 和 MCP 设置在活动 turn 中进入 deferred settings。后一次同类设置覆盖前一次设置，并在 turn 完成后应用。

`DelegateSubagent`、手动 compaction、历史导航、Undo/Redo、New/Resume 等会影响当前 turn 或 transcript 的命令，在活动 turn 中按各自 disposition 拒绝、延后或返回提示。`Interrupt` 通过独立控制信号请求取消当前操作。

## Route preparation 与 resolved authority

session 不把配置中的 `ModelRoute` 直接交给发送层。模型选择、恢复和配置 reload 先通过 route factory 准备 `PreparedPrimaryRoute`，验证 provider/model、protocol、metadata、retry 和 runtime catalog，再在提交点安装。

安装后的 `Agent::resolved_model_route` 是当前执行 authority：normal turn、compaction one-shot、child turn 和其它模型 helper 都使用它或由它派生的明确 resolved route。`active_protocol` 主要服务于状态和兼容判断；实际 request preparation 由 resolved route 的 `ProtocolBinding` 完成。

恢复时，`prepare_restored_model_route` 根据 transcript 中的 latest model 准备 restored route。若 latest model 与当前 provider/model 可解析，则提交完整 prepared route；若只能恢复模型标识，则保留 model-only 路径并由当前 Agent route 提供 runtime authority。提交前会验证 route、runtime snapshot、context scope、permission/reasoning 状态和 session token usage。

## 配置 reload 的当前语义

watcher 合并短时间内的重复通知后调用 `apply_config_reload`。reload 会重新加载并验证配置，更新 runtime catalog、model catalog、primary/expert route factories、API-key hints、allowed models、retry、compaction、tool timeout、tool parallelism 和 new-session defaults。

reload 的判断同时比较：

- resolved runtime catalog fingerprint；
- model/expert/default route maps；
- Agent retry、compaction、timeout 和 tool parallelism settings；
- 当前 active route 对应的 runtime fingerprint；
- 当前 provider 的 protocol 与 model metadata catalog；
- new-session default route。

没有 runtime delta 时保持现有 Agent 执行状态不变。当前 route 的 runtime 或 catalog 发生变化时，reload 先准备新 route，再重新安装当前 route。当前 route 若从配置 catalog 移除，已有 session 保留正在使用的 live route，并通过 notice 提示；它不会被静默切换到新默认模型。新 session default 的变化只影响后续新会话，不自动替换当前 route。配置无效或准备失败时，engine 保留原有运行状态并发出错误事件。

## Turn 执行

提示执行进入 `AgentRunner::run_prompt` 或带 continuation provider 的 `run_prompt_with_continuations`。带 recorder 的 runner 为 Agent 安装 runtime snapshot provider；provider 从当前 transcript 和 context branch 投影运行时快照。

一个 prompt turn 的 session 侧顺序是：

1. 发出 `UserMessage`；
2. 将 user content 写入 recorder，并发布 context projection；
3. 必要时生成并持久化 session title；
4. 调用 Agent 的 `run_stream_content_with_interactions_async`，由其进入 `run_resolved_turn_async`；
5. 将 AgentEvent 持久化为当前 transcript 事件，并转换为 stream events；
6. 成功时发布 context update、`AssistantDone` 和 `Done`；
7. 失败或中断时记录错误/中断状态，并发布 `Error` 和 `Done`。

assistant delta、reasoning、tool lifecycle、usage/prepared usage、compaction、retry、todo、permission、question、subagent 和模型流问题都通过 `SessionTransportEvent` 对外表达。permission/question 事件携带一次性 response handle；child event 标记其来源。

后台 subagent 完成后，engine 接收 `BackgroundSubagentCompleted`，将结构化结果安装为 parent-side evidence/continuation，并把继续处理的命令放回控制队列。

## Resume 与 session swap

`ResumeSession` 先解析 session ID 或前缀，再调用 `prepare_resume_package`。准备阶段：

1. 使用 resumable transcript reader 读取当前 JSONL 和 fingerprint；
2. 按 default cursor 解析 branch、leaf 和 runtime restore projection，并发现 child sessions；
3. 在 fingerprint 仍匹配、事务完整且 schema 可恢复时打开 append-safe `TranscriptRecorder`；
4. 将 recorder 采用到选定 branch，并准备 restored context scope；
5. 如果 snapshot 有 active turn，取消未完成 tool/subagent，记录 `TurnInterrupted`，重新读取并投影；
6. 返回包含 session ID、records、runtime snapshot 和 recorder 的 `PreparedResume`。

提交是一次受校验的 session swap：先验证恢复快照和 prepared route，再交换 live recorder，安装 runtime snapshot、turn sequence、context scope、permission/reasoning 状态和 resolved route，最后清理被替换的空会话文件。成功后发布 `SessionResumed`；失败不会用未校验的 recorder 或 route 覆盖 live session。

普通 resume 只接受当前 journal schema 和当前 assistant payload 形态。legacy/v1 transcript 可以被发现、扫描和列入 session index，但不能作为普通 session resume 或 append 的来源；其 decode-only 记录不会恢复 live Agent state。

## Branch、history 与 child view

runtime restore 根据显式 cursor、最近 checkout 或根 branch 解析当前内容路径。父路径和 branch-local records 合并后恢复 runtime frames、history、context nodes、evidence、summary artifacts 和 child summaries；其它 branch 的普通内容不会混入当前路径。

Undo、Redo 和 history navigation 通过 branch metadata 记录目标 branch/sequence，重新投影 snapshot，准备 route 和 context scope 后以 `SessionResumed` 形式通知前端。`ViewChild`、`ViewParent` 是查看 projection，不等同于把另一个 session 安装为当前 live session。

## 会话索引

`src/transcript/session_index.rs` 维护 `sessions-index.json`。索引使用文件 size 和 mtime 判断条目是否新鲜；缺失、版本不匹配或过期时只重新扫描对应 JSONL。扫描能够识别当前及可发现的 legacy/v1 transcript，并生成 record count、时间范围、model、title、最近 user/assistant 摘要和 content presence。索引只服务于发现和列表，不绕过 resume schema gate。

## 源码索引

- `src/session/engine.rs` — engine lifecycle、command loop、reload dispatch 和 transport events。
- `src/session/engine/config_reload.rs` — 配置 reload、runtime fingerprint 比较和 route reinstall 语义。
- `src/session/runner.rs` — prompt runner、AgentEvent 转换和 child event ingress。
- `src/session/restore.rs` — `prepare_resume_package`、restore projection 和 prepared route install。
- `src/session/lifecycle.rs` — session path、resumable records 和 append-safe recorder open。
- `src/session/settings.rs` — route preparation、authority install 和 session setting application。
- `src/transcript/journal.rs` — schema discovery/resume gate 和 fingerprint。
- `src/transcript/session_index.rs` — session discovery index。
