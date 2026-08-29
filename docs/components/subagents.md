# Subagents

## 入口与专家目录

Subagents 通过 Agent 的子代理工具进入运行时。可用入口包括：

- `agent__explore`
- `agent__fixer`
- `agent__oracle`
- `agent__designer`
- `agent__librarian`
- `agent__general`
- `agent__jobs`
- `agent__status`
- `agent__wait`
- `agent__cancel`

专家目录提供七个模板，每个模板包含名称、用途、系统提示、工具范围、权限模式、写入和委派能力、超时、工具调用上限、输入约定及结果形状等字段。

| 专家 | 工具范围 | 权限模式 | 可写入 | 可继续委派 | 默认边界 |
| --- | --- | --- | --- | --- | --- |
| `explorer` | 只读探索 | `Default` | 否 | 否 | 无模板级超时或调用上限 |
| `fixer` | 完整工具范围 | `Default` | 是 | 否 | 无模板级超时或调用上限 |
| `oracle` | 只读探索 | `Default` | 否 | 否 | 无模板级超时或调用上限 |
| `designer` | 只读探索 | `Default` | 否 | 否 | 无模板级超时或调用上限 |
| `librarian` | 只读探索 | `Default` | 否 | 否 | 无模板级超时或调用上限 |
| `general` | 只读探索 | `Default` | 否 | 否 | 无模板级超时或调用上限 |
| `reviewer` | 只读探索 | `Yolo` | 否 | 否 | 30 秒、最多 2 次工具调用 |

子代理创建时，运行时根据模板构造 child agent。child agent 不暴露 `agent__*` 子代理工具，因此不能递归委派。`reviewer` 使用独立的自动审批配置；普通 child agent 可以继承父级权限模式，但不会继承父级 permission session 中已经授予的具体 grant。

## 输入归一化与任务边界

子代理输入字段包括：

- `task` 或 `objective`：至少提供一个非空值；`objective` 优先；
- `success_criteria`：成功标准字符串数组；
- `allowed_paths`：允许读取的路径范围；
- `forbidden_paths`：禁止访问的路径范围，优先于允许范围；
- `owned_paths`：该运行可写入并参与并发锁定的路径范围；
- `model`：单次模型路由覆盖值；
- `target_child_session_id`：要接管的已有 child session；
- `background`：是否后台运行，默认 `false`。

`model` 与 `target_child_session_id` 不能同时提供。`agent__fixer` 必须提供非空 `owned_paths`。模板中的超时和工具调用上限由运行时应用，不作为模型可见的委派参数。

归一化后的内容会组成 child prompt。除任务字段外，prompt 包含成功标准、路径范围、模型或接管目标、执行边界，以及固定的“不递归委派、保持给定范围、简洁报告结果”约定；`agent__explore` 还会收到只读模式标记。

## 路径访问与运行锁

路径边界有两层：

1. **child tool 访问授权**：输入路径根会规范化并 canonicalize。带范围时，读工具只能访问 `allowed_paths` 或 `owned_paths`，写工具只能访问 `owned_paths`；`forbidden_paths` 对读写都优先拒绝。受路径范围检查的工具包括 `fs__read`、`fs__list`、`search__rg`、AST 搜索与预览、`fs__write`、`fs__append`、`fs__mkdir` 和 `edit__apply_patch`。
2. **Pool 运行锁**：可写专家使用 canonicalized `owned_paths`；只读专家在未提供路径范围时锁定 workspace root，否则锁定 `allowed_paths` 与 `owned_paths` 的 canonicalized roots。

运行锁在创建 reservation 时检查当前 active slots：

- 读与读不冲突，可以并行；
- 读与写在路径集合重叠或一方位于另一方之下时冲突；
- 写与写在路径集合重叠或一方位于另一方之下时冲突。

写入完成后，Pool 从 child transcript 中收集文件变更路径，并检查实际变更是否位于已获取的 `owned_paths` 锁内。发现锁外变更时，结果被改为 `failed`，`failure_kind` 为 `logical`，并把锁外路径写入 blocker。路径锁属于 Pool 的运行期状态；child tool 的路径授权仍独立执行。

## 创建新运行

创建入口是 `SubagentPool::start_named_governed()`，普通同步执行由 `run_named_governed()` 直接调用它后等待完成。

创建步骤如下：

1. 按专家名称查找模板；
2. 从归一化输入计算有效超时、工具预算和模型路由；
3. 为输入建立路径范围，计算 `RunPathAccess`，并向 Pool 申请 reservation；
4. reservation 分配稳定的 `run_id`；Pool 还为 child 分配单调递增的 `pool_ordinal`，用于 job 列表和 TUI 排序；
5. 普通创建在 `children/` 下创建 child transcript，记录 `SessionStarted`、`SubagentLifecycle(running)` 和 `SubagentStarted` 事件；
6. 根据模板和路由创建 child agent，安装路径范围和 child transcript 的 context scope；
7. 父 transcript（如果存在）记录对应的 `SubagentStarted`；
8. reservation 激活，写入 child session 摘要和取消通道；运行进入 active pool slot。

`SubagentPool` 的 active slot 保存取消发送端、取消请求状态、child 摘要、运行路径访问模式以及接管目标。激活前的 reservation 也在 active map 中，因此取消 active runs 覆盖尚未完成 child 初始化的运行。

## Takeover

通过 `target_child_session_id` 接管已有 child session。目标必须属于当前 parent，并存在 child transcript；专家名称必须与本次调用的模板名称一致；状态必须是 terminal：`completed`、`failed`、`budget_exhausted`、`cancelled` 或 `timed_out`（兼容读取 `error`、`errored`）。运行中的 child 不能接管。

child transcript 必须包含可解析的最近模型路由。接管不能同时使用模型覆盖，并始终恢复记录的 provider/model route。运行时会按接管模式校验记录路由；没有记录路由、路由不在专家允许范围内或 provider/model 未配置都会失败。

成功接管时，Pool 追加写入原 child transcript，恢复 turn 序列和 context branch，记录新的 `SubagentLifecycle(running)` 与 `SubagentStarted`，并保留原有 `pool_ordinal`。接管产生新的 `run_id`，但继续使用被接管 child 的 `child_session_id` 和历史 transcript。

同一个 child session 不能同时存在两个 active takeover。父 transcript 中的 child 关系、child transcript 中的记录路由和 terminal 状态共同决定接管是否可用。

## Pool job 生命周期

`SubagentPool` 由共享的 `SubagentPoolState`、变更通知 `Notify` 和 ordinal issuer 组成。active 与 completed 运行都按稳定 `run_id` 索引。

生命周期状态包括：

1. **Reservation**：生成 `run_id`，检查路径锁和重复 takeover；slot 状态为 `starting`，取消通道尚未激活；
2. **Active**：child transcript、child agent 和 parent start 记录准备完成后，slot 写入 `running` 摘要并安装取消通道；
3. **Execution**：执行器运行 child prompt；Pool 同时监听取消通道，并按有效超时包裹执行；
4. **Terminal reconciliation**：执行输出映射为结果状态，检查写入范围和锁覆盖，写 child completion 与 parent result，最后把结果从 active map 移到 completed map；
5. **Completed lookup**：结果保留在 completed map 中，`agent__status`、`agent__wait` 和上层 job 查询可读取；transcript projection 也能在运行内存之外重建 job。

`SubagentStatus` 当前包含：`running`、`completed`、`failed`、`budget_exhausted`、`cancelled` 和 `timed_out`。child executor 错误通常映射为 `failed`；包含“too many tool calls”时映射为 `budget_exhausted`。超时映射为 `timed_out`，取消映射为 `cancelled`。

如果 `ActiveRunGuard` 在正常完成前被丢弃，Drop 实现会生成 cancelled runtime summary，尝试写入 child completion 和 parent result，并把该 summary 放入 completed map。正常完成由 `guard.complete()` 原子地移除 active slot、保存 completed result，并唤醒等待者。

## 前台、后台与等待

### 前台运行

`background` 为 `false` 时，`RunnerSubagentDelegate::run_named()` 等待 `complete_started_run()` 返回：

- 成功状态返回 `ToolResult::ok`；
- 其他 terminal 状态返回带数据的错误结果；
- 数据包含 `run_id`、`child_session_id`、`agent_name`、`status`、`failure_kind`、`summary`、`full_summary`、`structured_result` 和 `active: false`。

### 后台运行

`background` 为 `true` 时，运行时必须配置 `background_control_tx`；否则调用立即返回后台执行不可用错误。Delegate 启动 child 后使用 `tokio::spawn` 独立执行，并立即返回：

- `run_id`；
- `child_session_id`；
- `agent_name`；
- `status: "running"`；
- 启动摘要；
- `active: true`；
- `background: true`。

后台任务完成后，将 `BackgroundSubagentCompleted` 命令发送回父 session control channel。父 session 通过该命令接收最终结果并完成父侧处理。

### Job 控制工具

- `agent__jobs`：读取父 transcript 投影的 job board，再用 Pool 的 active jobs 覆盖同一 `run_id` 的实时状态；结果按 `pool_ordinal` 和 `run_id` 排序；
- `agent__status`：要求非空 `run_id`，优先返回 Pool completed result，再返回 active job，最后从 child transcript 投影查询；未知 run id 返回错误；
- `agent__wait`：只能等待 active job。它先 claim foreground；同一运行已 terminal 或已被其他等待者 claim 时返回错误。等待者通过 `wait_for_result()` 阻塞到 completed result，成功后返回 terminal 结果并保留 foreground claim；
- `agent__cancel`：要求非空 `run_id`。active run 存在取消发送端时发送取消请求并返回 `cancellation_requested: true`；已结束或无法再次发送取消时返回当前 job；未知 run id 返回错误。

Pool 还提供 `cancel_active()`、`cancel_active_run_ids()` 和 `wait_until_idle()`。批量取消会标记 active slots 的 cancellation requested，并向已激活及尚在启动阶段的运行发送取消信号；`wait_until_idle()` 最多等待 3 秒。

## 事件与 transcript

### 运行级 transcript 事件

子代理运行由三类事件描述：

- `SubagentStarted`：`run_id`、父 session/run、child session、专家名称、启动摘要和 `pool_ordinal`；
- `SubagentLifecycle`：`run_id`、父 session/run、专家名称、状态和可选 detail。创建时写入 `running`，完成或取消时写入 terminal 状态；
- `SubagentResult`：`run_id`、父 session/run、child session、专家名称、状态和 summary。

完成时，父 transcript 记录结构化 `SubagentResult`。结构化结果还会被封装为 `Evidence`，其来源包含 run、child session、parent tool、parent turn 和 parent session 信息。child transcript 至少记录对应的 lifecycle completion。

父 transcript 位于父 session 文件；child transcript 位于 session 目录下的 `children/<child_session_id>.jsonl`。`project_job_board()` 读取父记录和 child 记录，先用父侧 `SubagentStarted` 建立 running job，再用父侧 `SubagentResult` 或 child 侧 lifecycle 更新 terminal 状态和摘要。

### 运行时事件发送器

`SubagentEventSender` 保存状态通知、错误通知和 child prompt 执行器，并可附带父 tool call id：

- `emit_status()` 调用状态 callback；
- `emit_error()` 调用错误 callback；
- `run_child_prompt()` 把 child agent、prompt、transcript、child session id、permission origin 和 parent tool call id 传给会话 runner。

会话 runner 创建的 sender 将状态消息映射为 `SessionTransportEvent::Notice`，错误消息映射为 `SessionTransportEvent::Error`。child prompt 由 `AgentRunner` 运行；有 permission origin 时使用 permission passthrough runner，否则使用普通 child streaming runner。

没有 sender 的 headless child 会把 user message、AgentEvent 和错误写入 child transcript，permission request 记录为 `Denied in headless child execution`。有 sender 时，child 的 AgentEvent 和权限交互由会话 runner 处理。

## 结果结构

`SubagentRunSummary` 包含：

- `run_id`；
- `child_session_id`；
- `agent_name`；
- `status`；
- `failure_kind`：运行时失败为 `hard`；写入范围或锁覆盖问题可为 `logical`；
- `summary`；
- `structured_result`。

结构化结果包含 `status`、`summary`、`findings`、`files_read`、`files_changed`、`commands_run`、`validation`、`blockers`、`next_steps`、`run_id`、`child_session_id` 及必要时的原始输出摘要。child 返回文本会尝试解析为结构化 JSON；无法解析时仍以完成或运行时失败状态生成结构化结果。

前台委派的 `ToolResult.data` 包含 run/session identity、`status`、`failure_kind`、摘要、`structured_result` 和 `active: false`。后台启动结果包含 run/session identity、`status: "running"`、启动摘要、`active: true` 和 `background: true`。

控制工具使用各自的数据结构：`agent__jobs` 返回 `jobs`；`agent__status` 返回单个 job；`agent__wait` 返回 terminal result；`agent__cancel` 返回 `run_id`、`cancellation_requested`，无法发送取消时还包含当前 job。`structured_result` 内包含 `findings`、`files_read`、`files_changed`、`commands_run`、`validation`、`blockers` 和 `next_steps`。
