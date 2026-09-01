# Transcript

## 数据形态

Transcript 是按会话保存的 JSON Lines journal。主会话文件为 `<session_id>.jsonl`；child session 文件位于会话目录下的 `children/`。每一行是一个带 journal envelope 的 `TranscriptRecord`，记录 session ID、sequence、timestamp、branch scope、revision 和扁平化的 `TranscriptEvent`。

`context_branch_id` 省略时表示根 branch `main`。sequence 在整个 session 内单调递增，不随 branch checkout 重置。事务记录以 payload 和 commit envelope 保证一组事件原子可见；未通过 commit 校验的事务不会进入投影。

## 当前事件模型

`TranscriptEvent` 同时承载内容、运行状态、branch metadata、subagent 状态和观测：

- session/model：`SessionStarted`、`SessionTitle`、`ModelChanged`、`ReasoningEffortChanged`、`ExpertModelChanged`；
- turn 内容：`UserMessage`、`AssistantTurn`、`InternalContinuation`、tool lifecycle 和 tool execution summary；
- turn 状态：`TurnStarted`、`TurnFinalized`、`TurnInterrupted`、`AutoContinueChanged`、`AutoContinuationScheduled`、`TodoSnapshot`；
- permission/error/evidence：permission decision/mode、`Error`、`Evidence`、validation advisory；
- subagent/telemetry：`SubagentStarted`、`SubagentLifecycle`、`SubagentResult`、`LlmRequestTelemetry`；
- context：compaction、logical checkpoint、branch create/summary/checkout、history navigation、context node/view/summary metadata 和 folded output metadata。

事件判定方法决定它是否进入 session history、content projection 或 branch metadata。控制事件、索引事件和 telemetry 仍保留在 journal，但不一定显示为用户消息。

### AssistantTurn

当前 assistant 内容统一写入 `AssistantTurn`：

```rust
pub struct TranscriptAssistantTurn {
    pub text: Option<String>,
    pub reasoning_content: Option<String>,
    pub replay: Option<OpaqueReplayState>,
    pub calls: Vec<HistoryToolCall>,
}
```

无 tool call 的 assistant text、带 reasoning 的 assistant result、以及包含 tool calls 的 assistant result 都使用同一个事件形态。tool call batch 的 call ID、name 和 arguments 保留在 `calls` 中，provider-native replay state 保留在 `replay` 中。普通新写入不会生成分散的 assistant message 或 assistant tool-call payload。

## Journal schema

- `JOURNAL_SCHEMA_VERSION` 为 `2`；
- `LEGACY_JOURNAL_SCHEMA_VERSION` 为 `1`，仅用于兼容读取和发现；
- 普通 record envelope 使用 `schema_version`、event ID、scope、base/resulting revision、可选 transaction fields 以及扁平化 record；
- transaction commit 同样使用 `schema_version`，并校验 transaction ID、count、revision、payload length 和 digest；
- logical checkpoint 自带其 payload `schema_version`，因此该事件的外层 journal schema 使用独立的 `journal_schema_version`，避免重复 JSON key。

payload digest 是确定性的 FNV-1a 风格 64 位 corruption guard，不是密码学完整性证明。

## Recorder

`TranscriptRecorder` 保存 session path、`JournalSink`、已提交 sequence、health、当前 branch cursor、context scope 和 active-turn tracking。`create` 创建新的 v2 session 文件；普通 `append`、metadata append、事务 append 和各类 `record_*` helper 都通过当前 schema 写入。

assistant 相关 helper 包括 `record_assistant_message` 和 `record_assistant_tool_call_batch`，二者都追加 `TranscriptEvent::AssistantTurn`。写入流程先分配 sequence 并构造 envelope，再 `write_all`、`flush`，需要 durable confirmation 的记录还会 `sync_data`。任何写入失败都会将 recorder 标记为 poisoned，且不推进 sequence 或 active-turn state。

事务 append 会在内存中为整组事件分配连续 sequence，写入 payload 和 commit 后才推进 recorder 状态。读取侧只有完整、已提交且通过 envelope/transaction 校验的事件可见。

## 读取与发现

严格读取用于完整 journal；partial-tail 读取只允许识别最后一个没有换行符的未完成 JSON 行。`repair_partial_tail` 只在完整前缀可严格解析时截断不完整尾部。未提交 transaction tail 不会被当作可追加 frontier。

发现读取接受当前 v2、v1 envelope 和无 envelope 的 legacy record，因而 session index、history discovery 和审计工具仍可列出旧文件。发现能力不等于可恢复能力：`read_resumable_records_with_fingerprint` 会要求每条 journal/commit record 使用 schema 2，并拒绝 schema 2 下的 legacy assistant payload。

因此当前边界是：

- v2 transcript 可用于普通 resume、append 和继续写入；
- v1/legacy transcript 可发现、读取、扫描和列出；
- v1/legacy transcript 不可作为普通 resume 或 append 的来源；
- decode-only 的 `AssistantMessage`、`AssistantToolCallBatch` 只为读取旧形态保留，不是当前写入或 live Agent restore 的 assistant contract。

## Branch 与 cursor

`BranchIndex` 从完整记录建立根 branch `main`，并校验 branch create、base sequence、tip、checkout 和 summary。非根 branch 的可见路径是父 branch 在 base sequence 的路径前缀加 branch-local records；显式 `leaf_sequence` 固定内容截面，否则使用选定 branch 当前 tip。

checkout 选择 append/read 的活动 branch scope，不会把内容永久冻结在 checkout 时的 leaf。context compaction、logical checkpoint 和 navigation records 在实际 branch scope 下校验其 source span、retained items 和 journal frontier。

## 投影

`project_runtime_restore_snapshot` 先解析 branch cursor，再恢复 model、permission/reasoning/workflow、context tree、protocol frames、evidence、summary artifacts 和 child sessions。runtime snapshot 中的 protocol frames 是恢复后 Agent 的协议上下文来源，并保留 runtime frame identity。

`restore_history_projection` 将 user、`AssistantTurn`、tool calls 和 tool outputs 组合为 history items；compaction 用 summary 替换 retired prefix；checkpoint 生成 summary/continuation；中断会关闭 active tool/subagent 状态；尾部不完整或取消的 tool group 会被规范化。

`project_session_history_tree` 只选择 session-history events，按 branch path 建立可导航的历史节点。TUI timeline projection 则将 message、reasoning、tool、permission、todo、compaction 和 subagent 状态转换为可渲染项；标题、branch、checkout、telemetry 等 metadata 不直接生成 timeline item。

## Resume 读取边界

session resume 的读取链路是：

```text
session id
  -> read_resumable_records_with_fingerprint
  -> branch-aware runtime restore projection
  -> append-safe TranscriptRecorder open
  -> validate/install snapshot and resolved route
  -> live recorder swap
```

resume 会验证文件 fingerprint，拒绝未提交事务 frontier，并在恢复 active turn 时记录中断、取消未完成 tool/subagent 后重新投影。恢复成功后，后续新写入继续使用 schema 2 和 `AssistantTurn`。

## 源码索引

- `src/transcript/journal.rs` — schema 2、legacy discovery、resumable gate、transaction validation 和 fingerprint。
- `src/transcript/model.rs` — `TranscriptRecord`、`TranscriptAssistantTurn` 和 `TranscriptEvent`。
- `src/transcript/recorder.rs` — v2 append/transaction writer、assistant helpers 和 recorder health。
- `src/transcript/restore.rs` — history/runtime restore helpers。
- `src/transcript/transcript_projection.rs` — branch-aware runtime/history/session projections。
- `src/transcript/transcript_projection/history.rs` — assistant/tool history normalization。
- `src/transcript/session_index.rs` — discovery/list sidecar index。
- `src/session/restore.rs` — resumable session package and recorder swap。
