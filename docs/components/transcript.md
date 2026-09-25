# Transcript

Transcript 是按会话保存的只追加写入（append-only）JSON Lines 日志。主会话日志保存在 `<session_id>.jsonl`，子代理日志保存在 `children/<child_session_id>.jsonl`。`sessions-index.json` 是专供列表展示与快速发现的派生索引；会话的真实数据全部保存在各会话的 JSONL 日志中。

当前系统采用日志规范版本 schema 2 写入新记录。系统能够直接恢复采用 schema 1 封套的历史日志：在内存中将旧版助手输出规范化为 `AssistantTurn`，原有文件内容保持不变，后续追加写入自动升级为 schema 2。不带封套的更早版本日志仅用于离线读取与检索展示。

## 记录结构与封套

普通的日志记录行由封套结构 `JournalRecordEnvelope`、记录元数据 `TranscriptRecord` 与具体事件 `TranscriptEvent` 扁平化序列化为一个 JSON 对象：

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

`record` 和 `event` 均通过 `serde(flatten)` 展开序列化，事件类型以 `kind` 字段作为标签。实际的 JSON 记录不包含嵌套的 `record` 或 `event` 外层键名：

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

`event_id` 固定由 `<session_id>:<sequence>` 拼接生成。`sequence` 在单个会话内严格单调递增，`resulting_revision` 等同于当前序列号，`base_revision` 对应上一条记录的版本号。

`scope` 字段标识封套层面的作用域：当记录包含 `context_branch_id` 时为 `branch`，否则为 `global`。默认主分支 `main` 的常规消息不单独标注分支 ID；分支拓扑、检出状态等元数据通过事件体、关联游标与投影规则解析。

## 助手历史单元

日志规范 schema 2 采用统一的数据结构保存助手回复：

```rust
pub struct TranscriptAssistantTurn {
    pub text: Option<String>,
    pub reasoning_content: Option<String>,
    pub replay: Option<OpaqueReplayState>,
    pub calls: Vec<HistoryToolCall>,
}
```

常规纯文本回复仅写入 `text` 字段：

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

包含工具调用的回合会在 `calls` 数组中记录调用 ID、工具名称与 JSON 参数。服务商私有的重放状态保存在 `replay` 中，记录命名空间、格式版本和负载内容；只有兼容的协议绑定才能在恢复时使用该重放状态。

`reasoning_message` 作为独立事件派发，用于时间线展示与耗时统计。`assistant_turn.reasoning_content` 和 `replay` 则作为助手历史的一部分持久化。旧版的 `assistant_message` 与 `assistant_tool_call_batch` 用于兼容解码旧日志；恢复读取器与 v2 记录器均会将它们转换为 `assistant_turn`。schema 2 格式的日志文件若包含旧版字段，会被判定为格式无效。

## 事件分类

`TranscriptEvent` 同时承载会话内容与系统运行状态：

- 会话与模型：会话启动与标题、模型变更、思考等级与专家路由调整；
- 回合与历史：用户输入、助手回复、流程继续、生命周期变迁与思考过程；
- 工具与权限：工具调用状态、执行摘要与权限审批结果；
- 流程管理：待办事项更新、自动继续状态与验证建议；
- 上下文演进：上下文压缩、逻辑检查点、分支创建、分支检出、历史导航与元数据更新；
- 子代理与证据：子任务生命周期、结构化结果与审计证据；
- 运行观测：LLM 请求遥测指标、Token 用量、缓存统计与系统异常。

各消费方按需投影所需事件，并非所有日志事件都会在聊天界面中展示为消息。

## 事务与落盘机制

需要原子写入的一组事件使用事务封装。每个记录行携带相同的 `transaction_id`、连续递增的 `transaction_index` 和一致的总数 `transaction_count`。数据写入完毕后，必须追加一条 `journal_entry = "transaction_commit"` 提交记录，记录基础版本、结束版本、数据总字节数与内容校验和。

读取器只解析已包含提交记录的完整事务。未完成提交的事务尾部对投影层不可见，也不能作为会话恢复的安全端点。

单条常规记录写入后执行 `write_all` 与 `flush`；影响恢复状态的关键事件会追加调用 `sync_data` 确保落盘。事务则在写入全部数据与提交标记后统一执行 `flush` 和 `sync_data`。

每个 `TranscriptRecorder` 在写入期间持有 `<session_id>.jsonl.lock` 独占文件锁。单个会话文件禁止多进程同时写入，只读读取不受锁影响。若发生底层 I/O 错误，记录器会被标记为损坏状态并释放文件锁，序列号与活动状态停止推进，保护文件尾部可被受控恢复。

`logical_checkpoint` 事件携带自身独立的版本号。为避免 JSON 展开后字段命名冲突，其封套字段声明为 `journal_schema_version = 2`，业务负载版本号声明为 `schema_version = 1`。

## 分支管理与状态投影

主分支固定命名为 `main`。`ContextBranchCreated` 事件记录父分支与派生时的基础序列号 `base_sequence`；分支上的内容记录携带 `context_branch_id`。分支的可见内容由派生点之前的父分支前缀与当前分支的记录拼接而成。

`ContextCheckout` 切换当前活动分支，不冻结具体的内容节点；显式指定 `SessionContextCursor.leaf_sequence` 时固定选定截面。撤销、重做与分支跳转统一通过 `HistoryNavigation` 事件记录目标序列号与重做栈，不修改磁盘上已存在的历史数据。

日志通过不同的投影器提供多种运行视图：

- 运行时恢复投影：恢复模型、权限、工作流、协议帧、上下文树、审计证据与子代理摘要；
- 对话历史投影：恢复用户消息、助手回复、工具输出、续接提示、压缩记录与检查点；
- 分支历史树：根据分支路径生成界面导航树；
- TUI 时间线：组装消息气泡、思考块、工具卡片、权限弹窗、待办列表与子代理状态；
- 会话索引与看板：生成全局会话摘要与子任务监控列表。

## 日志校验与会话恢复

系统执行严格的日志解析，校验会话一致性、序列号连续性、事件 ID 格式、作用域规则与事务完整性。读取遇到未完成的尾部时，只允许忽略最后一行未截断完整的 JSON；`repair_partial_tail` 仅在已有前缀完全合法时执行截断修复。

会话恢复通过 `read_resumable_records_with_fingerprint` 完成：

1. 验证日志包含合法的 schema 1 或 schema 2 封套，拒绝无封套的旧文件；
2. 校验序列号、事务连续性与文件指纹；
3. 将 schema 1 的旧版记录在内存中规范化为 `AssistantTurn`，重放状态损坏时报错；
4. 构建指定分支的运行时快照，验证恢复路由与上下文作用域；
5. 获取文件独占锁，重新核对指纹与事务边界，若文件被占用则拒绝恢复；
6. 构造安全追加模式的记录器，并在原子提交点替换当前活动会话。

恢复过程完全在内存中完成数据适配，不重写既有文件。恢复后的新记录使用 schema 2 追加写入，磁盘文件可由合法的旧版前缀与新版记录组合构成。无封套的更早日志仅供检索，不支持恢复追加。

## 源码索引

- `src/transcript/journal.rs`：实现封套定义、事务校验、读取门禁与文件指纹计算。
- `src/transcript/read.rs`：提供轻量读取接口，按需解析目标记录。
- `src/transcript/model.rs`：定义记录模型、助手单元与全部事件载荷。
- `src/transcript/recorder.rs`：负责日志追加、事务管理、数据规范化与落盘保证。
- `src/transcript/transcript_projection.rs`：实现支持分支的运行时投影与会话状态构建。
- `src/transcript/transcript_projection/history.rs`：规范化对话历史与工具调用记录。
- `src/transcript/transcript_projection/session_tree.rs`：生成会话分支历史导航树。
- `src/transcript/session_index.rs`：维护磁盘会话发现索引。
- `src/session/restore.rs`：校验恢复数据包、检查路由可用性并执行记录器切换。
- `src/protocol_frames.rs`：定义可用于状态恢复的协议历史项。
- `src/model_runtime/mod.rs`：声明不透明重放状态与协议兼容性作用域。
