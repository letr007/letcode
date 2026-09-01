# Transcript

Transcript 是按会话保存的 append-only JSON Lines journal。主会话文件为 `<session_id>.jsonl`，子代理会话位于 `children/<child_session_id>.jsonl`。`sessions-index.json` 是列表和发现用的派生索引，不是会话事实来源。

当前写入使用 journal schema `2`。schema `1` envelope 可直接 resume：读取时会在内存中将 legacy assistant payload 规范化为 `AssistantTurn`，原文件不重写，后续追加使用 schema 2。无 envelope 的更早 legacy 记录仍只用于读取与发现。

## Record 结构

普通 journal 行由 `JournalRecordEnvelope`、`TranscriptRecord` 和 `TranscriptEvent` 扁平序列化为一个 JSON object：

```rust
pub struct JournalRecordEnvelope {
    schema_version: u32,
    event_id: String,
    scope: JournalScope,
    base_revision: u64,
    resulting_revision: u64,
    transaction_id: Option<String>,
    transaction_index: Option<usize>,
    transaction_count: Option<usize>,
    record: TranscriptRecord,
}

pub struct TranscriptRecord {
    session_id: String,
    sequence: u64,
    timestamp_ms: u128,
    context_branch_id: Option<String>,
    event: TranscriptEvent,
}
```

`record` 和 `event` 都使用 `serde(flatten)`；事件使用 `kind` 作为 snake_case tag。因此实际 JSON 不包含嵌套的 `record` 或 `event` 字段：

```json
{
  "schema_version": 2,
  "event_id": "session-01:2",
  "scope": "global",
  "base_revision": 1,
  "resulting_revision": 2,
  "session_id": "session-01",
  "sequence": 2,
  "timestamp_ms": 1788277074123,
  "kind": "user_message",
  "content": {
    "parts": [{ "kind": "text", "text": "检查当前修改" }]
  }
}
```

`event_id` 固定为 `<session_id>:<sequence>`。`sequence` 在整个 session 内严格递增，`resulting_revision` 等于当前 sequence，`base_revision` 是前一个 revision。

`scope` 只描述 journal envelope：record 带 `context_branch_id` 时为 `branch`，否则为 `global`。根 branch `main` 的普通内容不写 `context_branch_id`；branch topology、checkout 和其它全局 metadata 由事件 payload、关联 cursor 与 projection 规则解释，不能仅根据 `scope` 判断其内容归属。

## AssistantTurn

schema v2 使用统一的 assistant 历史单元：

```rust
pub struct TranscriptAssistantTurn {
    pub text: Option<String>,
    pub reasoning_content: Option<String>,
    pub replay: Option<OpaqueReplayState>,
    pub calls: Vec<HistoryToolCall>,
}
```

普通文本回复只写 `text`：

```json
{
  "schema_version": 2,
  "event_id": "session-01:7",
  "scope": "global",
  "base_revision": 6,
  "resulting_revision": 7,
  "session_id": "session-01",
  "sequence": 7,
  "timestamp_ms": 1788277079205,
  "kind": "assistant_turn",
  "text": "检查完成。"
}
```

带工具调用的 turn 在 `calls` 中保存 `call_id`、工具名和 `arguments_json`。provider-native replay state 写入 `replay`，其中包含 namespace、payload version、producer identity 和 opaque payload；恢复时只有兼容的 protocol binding 可以使用该 replay。

`reasoning_message` 仍是当前事件，用于 timeline/audit 展示及可选的 `duration_ms`。`assistant_turn.reasoning_content` 和 `replay` 则属于可恢复的 assistant history。旧的 `assistant_message` 与 `assistant_tool_call_batch` 用于解码 schema 1；resume reader 和 v2 recorder 都会将它们规范化为 `assistant_turn`，而 schema 2 文件若直接包含旧 assistant payload 仍视为无效。

## 事件模型

`TranscriptEvent` 同时承载会话内容和运行状态：

- session/model：session start/title、model、reasoning effort 和 expert route 变化；
- turn/history：user、assistant、continuation、turn lifecycle 和 reasoning；
- tool/permission：tool lifecycle、tool execution summary 和 permission decision；
- workflow：todo、auto continue 和 validation advisory；
- context：compaction、logical checkpoint、branch、checkout、history navigation 和 context metadata；
- subagent/evidence：child lifecycle、structured result 和 evidence；
- observation：LLM request telemetry、usage、cache 和错误。

事件是否进入 provider history、runtime restore、session history tree、TUI timeline、session index 或 job board，由各自 projection 决定。journal 中存在的 metadata 和 telemetry 不一定显示为聊天消息。

## Transaction 与持久化

需要原子可见的一组事件使用 transaction。每个 payload record 携带相同的 `transaction_id`、连续的 `transaction_index` 和一致的 `transaction_count`；payload 后必须跟随 `journal_entry = "transaction_commit"`，其中记录 base/resulting revision、payload byte length 和 digest。

读取器只释放通过 commit 校验的 transaction。完整但未 commit 的 transaction tail 对 projection 不可见，也不能作为 append-safe resume 的已确认 frontier。

单条写入执行 `write_all` 和 `flush`；影响恢复语义的事件还会执行 `sync_data`。transaction 一次写入全部 payload 和 commit 后执行 `flush`、`sync_data`。任一 I/O 失败都会将 recorder 标记为 poisoned，且不会推进 sequence 或 active-turn state。

`logical_checkpoint` payload 自带独立的 `schema_version`。为避免扁平 JSON 出现重复 key，该事件的 journal envelope 使用 `journal_schema_version = 2`；payload 的 `schema_version` 当前为 `1`。

## Branch 与投影

根 branch ID 为 `main`。`ContextBranchCreated` 记录 parent 和 `base_sequence`；非根 branch 的内容 record 携带 `context_branch_id`。branch 的可见内容由父路径在 base sequence 的前缀，加上本 branch 不超过目标 leaf 的内容组成。

`ContextCheckout` 选择活动 branch scope，但不会永久冻结内容 leaf；显式 `SessionContextCursor.leaf_sequence` 才表示固定截面。Undo、Redo 和 Navigate 通过 `HistoryNavigation` 持久化目标 sequence 与 redo stack，不修改已有记录。

主要投影包括：

- runtime restore：恢复 model、permission、workflow、protocol frames、context tree、evidence 和 child summaries；
- history projection：恢复 user、`AssistantTurn`、tool output、continuation、compaction 和 checkpoint；
- session history tree：按 branch path 生成导航节点；
- TUI timeline：恢复 message、reasoning、tool、permission、todo 和 subagent 状态；
- session index/job board：生成会话摘要和子代理运行状态。

## 读取与 Resume

严格读取校验 session identity、sequence/revision 连续性、event ID、scope 和 transaction。live partial-tail 读取只允许忽略最后一个未完成 JSON 行；`repair_partial_tail` 只在完整前缀可严格解析时补换行或截断无效尾部。

普通 resume 使用 `read_resumable_records_with_fingerprint`：

1. 接受 schema 1 或 2 envelope，并拒绝无 envelope legacy；
2. 校验 sequence/revision、transaction 和文件 fingerprint；
3. 将 schema 1 的 `assistant_message` / `assistant_tool_call_batch` 在内存中规范化为 `AssistantTurn`；无效的 legacy replay state 直接报错；
4. 投影选定 branch 的 runtime snapshot，并验证 restored route 和 context scope；
5. 打开 append-safe recorder，并在提交点替换 live session。

兼容过程不改写既有 JSONL。resume 后的新记录使用 schema 2，因此文件可以包含合法的 v1 前缀和 v2 后缀；下次 resume 会重复相同的内存规范化。无 envelope 的更早 legacy transcript 仍可用于发现、列表、摘要和只读审计，但不能普通 resume 或 append。

## 源码索引

- `src/transcript/journal.rs` — envelope、transaction、读取、schema gate 和 fingerprint。
- `src/transcript/model.rs` — `TranscriptRecord`、`TranscriptAssistantTurn` 和事件 payload。
- `src/transcript/recorder.rs` — append、transaction、assistant normalization 和 durability。
- `src/transcript/transcript_projection.rs` — branch-aware runtime/history/session projections。
- `src/transcript/transcript_projection/history.rs` — assistant/tool history normalization。
- `src/transcript/transcript_projection/session_tree.rs` — session history tree。
- `src/transcript/session_index.rs` — session discovery index。
- `src/session/restore.rs` — resume package、route validation 和 recorder swap。
- `src/protocol_frames.rs` — 可恢复的 protocol history item。
- `src/model_runtime/mod.rs` — opaque replay state 和 compatibility scope。
