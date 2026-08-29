# Transcript

## 数据形态

Transcript 是按会话保存的 JSON Lines 日志。会话文件位于会话目录下，文件名为 `<session_id>.jsonl`；子会话文件位于 `children/` 下的同名结构中。

每个业务记录是一个 `TranscriptRecord`：

```rust
pub struct TranscriptRecord {
    pub session_id: String,
    pub sequence: u64,
    pub timestamp_ms: u128,
    pub context_branch_id: Option<String>,
    pub event: TranscriptEvent,
}
```

`event` 使用 Serde flatten 写入记录对象，因此事件字段与记录元数据位于同一 JSON 对象中。`context_branch_id` 可省略；读取和分支投影会把省略值解释为根分支 `main`。`sequence` 是会话内单调递增的日志修订号，`timestamp_ms` 是记录时间戳。

`TranscriptEvent` 承载内容事件、运行时状态事件和上下文元数据：

- 会话和模型：`SessionStarted`、`SessionTitle`、`ModelChanged`、`ReasoningEffortChanged`、`ExpertModelChanged`；
- 对话和工具：`UserMessage`、`AssistantMessage`、`ReasoningMessage`、`AssistantToolCallBatch`、`ToolCallStarted`、`ToolCallFinished`、`ToolCallCancelled`、`InternalContinuation`；
- 回合与自动继续：`TurnStarted`、`TurnFinalized`、`TurnInterrupted`、`AutoContinueChanged`、`AutoContinuationScheduled`、`TodoSnapshot`；
- 权限和执行：`PermissionDecision`、`PermissionModeChanged`、`ToolExecutionSummary`、`Error`、`Evidence`；
- 子代理和观测：`SubagentStarted`、`SubagentResult`、`SubagentLifecycle`、`LlmRequestTelemetry`；
- 上下文：`ContextCompaction`、`LogicalCheckpoint`、`ContextBranchCreated`、`ContextBranchSummary`、`ContextCheckout`、`HistoryNavigation`、`ContextExperimentStarted`、`ContextExperimentReturned`、`ContextNodeCreated`、`ContextNodeLifecycle`、`ContextViewOperationMetadata`、`ContextSummaryArtifactMetadata`、`FoldedOutputMetadata`。

事件是否属于会话历史、会话内容或上下文分支元数据由 `TranscriptEvent` 的判定方法决定。历史树只选择可显示的历史事件；日志中的控制、索引和观测事件仍然保留在日志中，但不一定进入用户可见内容。

## Recorder

`TranscriptRecorder` 是追加写入入口。它保存：

- 当前会话 ID和 JSONL 路径；
- `JournalSink` 写入端；
- 已提交的 `sequence`；
- `RecorderHealth`（`Healthy` 或 `Poisoned`）；
- 当前上下文分支游标；
- `ContextScopeState`，其中包含活动上下文实验；
- 推理计时和每个分支的活动回合跟踪器。

### 创建和打开

`TranscriptRecorder::create(base_dir)` 创建目录，生成会话 ID，打开 `<session_id>.jsonl` 的 append 文件，并以 `sequence = 0`、健康状态和根分支回合跟踪器初始化。

`open` 和 `open_existing` 用于接管或恢复已有会话。打开流程先读取完整记录和文件指纹，然后重新读取文件内容并检查：

1. 文件长度与确定性内容摘要仍与读取阶段一致；
2. 文件末尾没有未提交事务；
3. 所有记录的 `session_id` 与目标会话一致；
4. 分支、上下文作用域和活动回合跟踪器可以从记录重建。

通过检查后，Recorder 的 `sequence` 设置为记录中的最大序号，写入端以 append 模式打开。这样恢复写入不会从过期的序号前沿继续追加。

### 单条追加

`append(event)` 使用当前时间戳进入以下路径：

```text
append
  -> append_with_timestamp
  -> append_with_timestamp_and_branch
  -> append_record
```

普通事件使用 `current_context_branch_id`；上下文分支元数据不附带分支 ID。根分支由 `None` 表示，非根分支使用显式 ID。`append_metadata` 明确以无分支作用域写入元数据。

`append_record` 为事件分配下一个序号，构造 `JournalRecordV1`，生成 `event_id = <session_id>:<sequence>`，计算 `base_revision` 和 `resulting_revision`，序列化后写入一行。每次写入都会调用 `write_all` 和 `flush`；需要持久化确认的事件还会调用 `sync_data`。上下文压缩通过 `append_durable_on_branch` 写入，且在写入前会根据当前记录验证压缩范围。

任何写入、flush 或同步失败都会把 Recorder 标记为 `Poisoned`，并且不会推进 `sequence` 或活动回合状态。Poisoned Recorder 后续追加会直接失败。

### 事务追加

`append_transaction` 接收 `(TranscriptEvent, Option<String>)` 列表。它先在内存中为全部事件分配连续序号，并生成同一个 `transaction_id`、事务索引和事务总数；随后追加一个提交记录：

```rust
JournalTransactionCommitV1 {
    schema_version,
    journal_entry: "transaction_commit",
    transaction_id,
    transaction_count,
    base_revision,
    resulting_revision,
    payload_length,
    payload_digest,
}
```

整个 payload、提交记录一次写入，然后执行 `flush` 和 `sync_data`。只有这些操作全部成功后，Recorder 才推进序号并逐条更新活动回合跟踪器。事务读取时，只有通过提交记录的 ID、数量、修订号、payload 长度和摘要校验的事务才会进入记录结果。

## Journal

Journal 定义：

- `JOURNAL_SCHEMA_VERSION` 为 `1`；
- `JournalScope` 为 `Global` 或 `Branch`；
- `JournalRecordV1` 包含 schema、事件 ID、作用域、修订号、可选事务字段以及扁平化的 `TranscriptRecord`；
- `FileJournalSink` 将 `JournalSink` 映射到 `std::fs::File` 的 `write_all`、`flush` 和 `sync_data`。

普通事件由 `serde_json` 序列化。`LogicalCheckpoint` 使用专门的序列化路径，把日志外层字段写入对象，避免与 checkpoint 自身的 `schema_version` 产生重复 JSON 键。

payload 摘要是确定性的 FNV-1a 风格 64 位摘要，用于检测事务 payload 是否发生变化，不提供密码学完整性保证。

### 读取模式

- `read_records` 和 `read_records_with_fingerprint` 使用严格解析；后者同时返回文件长度和内容摘要组成的 `TranscriptFileFingerprint`。
- `read_records_allow_partial_tail` 允许忽略文件中没有换行符、且位于最后的未完成 JSON 行。被忽略的尾部不会进入投影，并记录 debug 日志。
- `repair_partial_tail` 在确认最后一个完整换行前缀可以严格解析后，把文件截断到该前缀并同步文件数据。
- `scan_transcript_content` 不生成记录，而是验证日志事务结构并报告是否存在未提交事务尾部。

解析器逐行处理普通记录和提交记录。事务记录先进入 pending buffer，遇到有效提交记录后才批量释放；普通记录不能插入 pending 事务中。解析器还验证事务索引连续、事务计数一致、提交修订号连续，以及 payload 摘要一致。最终会执行日志条目校验，再返回 `TranscriptRecord`。

完整但没有提交标记的事务 payload不会被释放到读取结果中。`open_existing` 另外调用 `content_tail_is_uncommitted_transaction`，发现这种尾部时拒绝以追加模式打开，避免从不确定的持久化前沿继续写入。

## 分支和游标

根上下文分支 ID 是 `main`。`BranchIndex` 从完整记录建立：

- 预置根分支，父分支为空，基础序号和 tip 为 `0`；
- `ContextBranchCreated` 必须使用未出现过的 ID，父分支必须存在，`base_sequence` 必须能在父分支路径解析；新分支 tip 从 `base_sequence` 开始；
- 带分支作用域的普通记录更新对应分支 tip；没有 `context_branch_id` 的记录归入 `main`；
- `ContextCheckout` 记录当前 checkout 的分支和叶序号；`ContextBranchSummary` 的叶序号不能超过该分支 tip。

活动分支的解析顺序是显式的 `current_branch_id`、日志中最近的 checkout、根分支 `main`。`SessionContextCursor` 还可以带 `leaf_sequence`：没有显式叶序号时使用所选分支 tip；有显式叶序号时读取该分支在该序号处的截面。

`collect_branch_path_records` 对非根分支递归收集父分支在 `base_sequence` 处的路径，再追加当前分支不超过叶序号的记录。分支元数据不进入内容路径。一个分支的可见内容由父路径前缀和当前分支局部记录组成。

checkout 选择的是活动分支作用域；它本身不会把该分支永久冻结在 checkout 时的内容叶。需要固定内容截面时，调用方必须提供显式 `leaf_sequence`。

## 投影

投影先解析分支上下文，再按消费场景选择记录。

### Runtime restore 投影

`project_runtime_restore_snapshot` 的流程是：

1. 用 `SessionContextCursor` 解析分支和叶序号；
2. 从选定分支路径构造 `runtime_projection_records`；
3. 校验上下文压缩和逻辑 checkpoint 的投影事件；
4. 从选定记录恢复最新模型、权限模式、工作流、上下文树、协议帧、证据和其它运行时状态；
5. 从完整会话记录计算 `max_turn_id`，因为 turn ID 分配跨分支保持会话级连续；
6. 返回包含 `branch_id`、`leaf_sequence`、选定记录和运行时快照的 `RuntimeRestoreSnapshot`。

`runtime_projection_records` 还会在选定内容路径之外纳入与当前路径相关的上下文节点创建/生命周期、视图操作元数据和摘要来源元数据，但会按当前分支、checkout 作用域、叶序号和节点关联进行限制。它不会把其它分支的普通内容混入运行时上下文。

上下文压缩的校验范围以压缩提交前的 journal frontier 和实际选定分支为依据。根分支的内容保持 `None` 分支作用域；非根分支的压缩记录以实际分支 ID 写入。逻辑 checkpoint 的来源跨度、活动 segment 和 retained items 在投影阶段重新校验。

### 历史和对话投影

`restore_history_projection` 按记录顺序建立带来源跨度的历史项：

- 用户消息、助手文本、工具调用和工具输出形成历史项；
- `TurnStarted` 关联相邻的用户帧并建立活动回合；
- `ContextCompaction` 用 summary 替换已退休前缀，并保留压缩后的尾部；
- `LogicalCheckpoint` 生成 checkpoint summary 和 continuation 两个历史项；
- 中断或被中断的回合会关闭活动工具和等待状态；
- 末尾会规范化未完成或已取消的工具调用组。

`history_item_to_conversation_message` 将上下文摘要、用户消息、内部 continuation 和助手文本转换为对话消息；助手工具调用和工具输出不直接转换为 `ConversationMessage`。

`project_session_history_tree` 只处理 `is_session_history_entry()` 返回 true 的事件，生成 `SessionHistoryEntry`，并以每个分支最近的历史叶作为父节点；分支首次出现历史项时，通过父分支的 `base_sequence` 找到锚点，把分支历史接到父路径上。

### TUI 时间线投影

`timeline_from_transcript_records` 顺序遍历记录并更新 `Timeline`：

- 用户和助手消息恢复为消息项；
- reasoning 恢复为带可选持续时间的 reasoning 项；
- 工具开始、完成和取消恢复为工具项；
- 权限决定恢复为已解决的权限项；
- todo 和 auto-continue 恢复为对应状态项；
- 压缩恢复为带 summary 的压缩项；
- 中断会取消活动工具和前台子代理等待；
- 子代理开始事件注册子代理状态；
- 会话标题、分支、checkout、观测和其它元数据不产生时间线项。

TUI 的 `render_transcript` 消费当前活动时间线，并通过 `TranscriptRenderCache` 缓存每个时间线项的文档、行起点和行数。缓存按宽度、主题和时间线缓存 ID 失效；文档同时用于显示、复制和选择坐标映射。

## 恢复读取流程

恢复已有会话：

```text
session_id
  -> session_path(<session_id>.jsonl)
  -> read_records_with_fingerprint
  -> project_runtime_restore_snapshot_with_children
  -> TranscriptRecorder::open_existing
  -> validate / install runtime snapshot
  -> replace live recorder
```

`project_runtime_restore_snapshot_with_children` 先对父会话记录做一次运行时投影，再从选定记录中发现子会话摘要；存在子会话时，以这些摘要做第二次投影。默认恢复游标是 `{ branch_id: None, leaf_sequence: None }`，即从日志 checkout 或根分支解析活动分支，并读取该分支当前 tip。

恢复包 `PreparedResume` 同时携带会话 ID、原始记录、运行时恢复快照和已打开的 Recorder。安装阶段先验证快照自身的协议帧、历史、证据和模型信息，再准备上下文作用域及模型路由；验证成功后替换 live Recorder，安装运行时快照、turn 序号、上下文作用域、权限模式和恢复后的模型状态。

运行时快照中的协议帧是协议恢复的来源。恢复后的 Agent 从快照重建活动协议帧和历史项，并保留运行时 frame ID。
