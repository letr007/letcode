# Subagents

## 专家与运行入口

Subagent 委派通过 Agent 的 delegation tools 进入 `SubagentPool`。当前六个可委派专家是：

| 专家 | 工具 | 用途 | 写入 |
| --- | --- | --- | --- |
| `explorer` | `agent__explore` | 聚焦只读仓库探索 | 否 |
| `fixer` | `agent__fixer` | 限定范围的实现或修复 | 是 |
| `oracle` | `agent__oracle` | 根因、风险和验证判断 | 否 |
| `designer` | `agent__designer` | 设计、方案和接口梳理 | 否 |
| `librarian` | `agent__librarian` | 资料、证据和上下文整理 | 否 |
| `general` | `agent__general` | 边界明确的只读通用辅助 | 否 |

`reviewer` 是 permission review 专家，不属于六个可委派 subagent tool 的目录。child agent 不暴露任何 `agent__*` delegation tool，因此不能递归委派。

Pool 控制入口为：

- `agent__jobs`：列出 parent transcript projection 与 active pool 合并后的 jobs；
- `agent__status`：查询单个 run 的 active/completed/transcript 状态；
- `agent__wait`：claim 并等待 active run 的 terminal result；
- `agent__cancel`：请求取消 active run，或返回当前 terminal job。

## 输入与治理

归一化输入至少包含非空 `task` 或 `objective`；`objective` 优先。还可以指定 `success_criteria`、`allowed_paths`、`forbidden_paths`、`owned_paths`、`model`、`target_child_session_id` 和 `background`。

`model` 与 `target_child_session_id` 互斥。`fixer` 必须提供非空 `owned_paths`，因为 Pool 需要为可写运行建立路径锁。模板的 timeout、tool budget、scope、permission mode、write/delegate capability 和结果 shape 由 expert template/runtime 应用，不由模型在 prompt 中自行扩大。

child prompt 包含任务、成功标准、路径范围、route/takeover 信息、执行边界以及固定的“不递归委派、保持给定范围、简洁报告结果”约定。读工具受 `allowed_paths`/`owned_paths` 限制，写工具只能落在 `owned_paths`；`forbidden_paths` 优先于其它范围。

Pool reservation 使用 canonicalized path roots：读-读可并行；读-写和写-写在路径重叠或祖先/后代关系下冲突。完成时 Pool 对比 child transcript 的实际变更路径和 owned-path lock；锁外变更会使运行结果成为 logical failure。

## Child route authority

每个 child run 都在创建前解析并准备自己的 model route。`ExpertRouteFactory` 根据 expert policy、parent route、requested model 和 runtime catalog 生成 `PreparedPrimaryRoute`，再创建带 resolved runtime route 的 child Agent。

route authority 的规则是：

- 普通 child 若提供 model override，必须通过 expert allowed-model policy 和 parent route preparation；
- 未提供 override 时使用 expert default route，缺省再使用 parent primary route；
- route 一旦准备并安装，child turn 直接使用其 `ResolvedModelRoute` 和 binding；执行期间不会按模型字符串重新猜测 provider、protocol 或 endpoint；
- child 的 compaction one-shot 和其它纯文本 helper 使用同一 child resolved route，不另选 provider route；
- 子代理工具、child runner 和 one-shot helper 的权限/事件边界可以不同，但 route identity 必须来自该 child 的 resolved authority。

## 新建运行

`SubagentPool::start_named_governed` 创建 reservation，`run_named_governed` 提供同步等待包装。流程为：

1. 查找 expert template；
2. 归一化 input，计算 path access、timeout、tool budget 和 route；
3. 申请 reservation，分配稳定 `run_id` 与单调 `pool_ordinal`；
4. 普通运行创建 child transcript，写入 `SessionStarted`、running lifecycle 和 `SubagentStarted`；
5. 创建 child Agent，安装 route、path scope 和 child transcript context scope；
6. parent transcript 写入对应的 start event；
7. 激活 slot 和 cancellation channel，运行 child prompt。

active slot 同时覆盖 reservation/starting 和 running 状态，因此尚未完成 child 初始化的运行也能被 cancel。

## Exact retained route takeover

`target_child_session_id` 表示接管一个已经存在的 child session。takeover 只接受：

- child 属于当前 parent；
- child transcript 存在且能读取；
- recorded agent name 与本次 expert template 完全一致；
- child 状态为 terminal（completed、failed、budget_exhausted、cancelled 或 timed_out；兼容读取 errored）；
- child transcript 中存在可解析的 provider/model route；
- 没有另一个 active takeover 使用同一 child session。

takeover 不接受 model override。child transcript 中记录的 provider/model route 是 exact retained authority；Pool 通过 `parent.prepare_primary_route` 恢复它。若 route 仍在当前 selectable catalog 中，则使用当前 factory；若 route 已被移除，则使用与该 route 同一 epoch 保存的 `RetainedRoutePreparation`（旧 factory + resolved route）。该 exact 分支只允许已有 takeover/resume 使用，不会把 removed route 放回 selectable catalog。

成功 takeover 会在原 child transcript 上 append 新的 running lifecycle 和 started event，恢复其 turn sequence/context branch，保留原 `pool_ordinal`，并生成新的 `run_id`。child session ID 和历史 transcript 不变。exact retained route 在 takeover 中可绕过当前 expert allowed-model 列表；非 takeover 的新委派仍受当前 policy。正在运行的 child、缺失或无效 route、缺少 exact preparation、agent/parent 关系不匹配都会直接失败。

## 生命周期与状态

Pool 状态依次为 reservation、active execution、terminal reconciliation 和 completed lookup。terminal 状态包括 `completed`、`failed`、`budget_exhausted`、`cancelled` 和 `timed_out`。tool budget 超限映射为 `budget_exhausted`；executor error 映射为 `failed`；timeout/cancel 分别映射为对应状态。

terminal reconciliation 会检查实际写入范围、写 child completion 和 parent result，并将 active slot 移到 completed map。`ActiveRunGuard` 在未正常完成而被 drop 时生成 cancelled summary；正常完成由 `guard.complete()` 保存 completed result 并唤醒 waiter。

## Foreground、background 与 jobs

foreground delegate 等待 `complete_started_run`，返回带 run/session identity、expert、status、failure kind、summary、structured result 和 `active: false` 的 `ToolResult`。

background delegate 需要可用的 parent background control channel。它启动独立 task 后立即返回 `status: running`、`active: true` 和 `background: true`；完成时向 parent session control channel 发送 `BackgroundSubagentCompleted`。

`agent__wait` 只等待 active job，并且同一 run 只允许一个 foreground claim。`agent__status` 优先查询 completed、再查询 active、最后查询 child transcript projection。`agent__jobs` 以 pool ordinal/run ID 排序并覆盖实时 active status。`agent__cancel` 只对 active run 发 cancellation request；未知 run ID 直接报错。

## Transcript、evidence 与结果

运行级 transcript 事件为：

- `SubagentStarted`：run/parent/child identity、expert、summary 和 pool ordinal；
- `SubagentLifecycle`：run identity、expert、status 和 detail；
- `SubagentResult`：run/parent/child identity、expert、status 和 summary。

完成的 structured result 还可作为 parent-side `Evidence`，来源包含 run、child session、parent tool、parent turn 和 parent session。parent transcript 保存 child relationship；child transcript 保存其自身 lifecycle 和 assistant/tool 内容。job board 可以从 parent start/result 与 child lifecycle 重建运行状态。

结果结构包含 `run_id`、`child_session_id`、`agent_name`、`status`、`failure_kind`、`summary`、`structured_result`。structured result 约定包含 `findings`、`files_read`、`files_changed`、`commands_run`、`validation`、`blockers` 和 `next_steps`。

## 源码索引

- `src/agent/catalog.rs` — 六个 expert template、subagent catalog 和 capability contract。
- `src/tool/delegation.rs` — input normalization、path scope 和 delegation schema。
- `src/subagent/route_factory.rs` — child route preparation、allowed models 和 takeover route validation。
- `src/subagent/pool.rs` — reservation、path locks、takeover、lifecycle、wait/cancel 和 reconciliation。
- `src/subagent/result.rs` — summary、structured result 和 terminal mapping。
- `src/session/subagent_delegate.rs` — session-owned route display、background control 和 parent integration。
- `src/transcript/recorder.rs` — subagent start/lifecycle/result persistence。
