# Agent

## 运行时边界

`Agent` 是持有会话运行状态的执行对象。它保存当前 `ModelRoute` 与 resolved runtime route、runtime catalog、`RuntimeSnapshot`、协议历史投影、工具注册表、权限会话、turn 状态、上下文压缩配置、重试配置、技能和子代理工厂。

provider 的可执行身份不是由一个裸 model 字符串决定的。`resolved_model_route` 是当前 turn 和 helper 请求使用的 route authority，包含 provider/model、protocol ID、endpoint、auth、headers/query、capabilities、generation settings、cache/retry 配置、transport 和 immutable `ProtocolBinding`。普通 turn 在没有 installed resolved route 时直接失败。

### Route preparation 与 authority

`PrimaryRouteFactory::prepare_route` 将配置中的 `ModelRoute` 校验并解析为 `PreparedPrimaryRoute`。安装时，`PreparedPrimaryRouteInstall::apply` 或 `Agent::apply_prepared_route` 一次性更新逻辑 route、protocol、model metadata、retry 配置和 resolved route。需要同时更新 route 与 resolved runtime identity 的路径使用 `set_model_route_authority`。

`active_protocol` 仍可用于展示和兼容判断；发送请求时以 installed `ResolvedModelRoute` 和其 binding 为准。切换模型或重新安装 route 会使下一次 request projection 按新 route 构造。

## Turn 入口

`Agent::run` 是非流式便捷入口。交互式调用使用 `run_stream_async`，内容型调用使用 `run_stream_content_with_interactions_async`。后者安装本次调用的 question handler，并传递：

- `on_delta`：可见 assistant text delta；
- `on_event`：reasoning、tool、usage、retry、compaction 和 turn 生命周期事件；
- `approve`：permission request 的决策；
- question handler：模型调用 question tool 时的响应路径。

入口首先检查 installed resolved route，再把实际执行交给 `run_resolved_turn_async`（`src/agent/protocol_stream.rs`）。因此 normal turn、provider protocol 选择和 route authority 在同一条 resolved runtime 路径上收敛。

## `run_resolved_turn_async`

`run_resolved_turn_async` 完成一次普通 Agent turn 的协调准备：

1. 克隆当前 `ResolvedModelRoute` 并取得对应 protocol；
2. 生成包含 selected skills 的 turn prelude；
3. 记录当前 active history 长度，设置 current-turn protected boundary，并追加用户内容；
4. 发出 `TurnStarted`；
5. 创建 `ResolvedTurnDriver`，连接 Agent 状态、delta/event/approval callbacks、turn counters、cache usage、response metadata 和可选 fake decorator；
6. 使用 `TurnOrchestrator::new(ModelRuntime::default(), TurnLimits)` 执行 turn；
7. 返回 final text，空文本时返回 `No response content`。

协议差异位于 model runtime 的 binding preparation、transport 和 adapter decoder；Agent driver 负责 turn state、历史、工具、事件和 session side effects。

## TurnOrchestrator、ModelRuntime 与 TurnDriver

`TurnOrchestrator` 是统一的 iteration/retry/continuation 控制器。每次 iteration：

1. 检查 iteration limit；
2. 由 `TurnDriver::prepare_iteration` 产生 `ModelRequestInput`；
3. 调用 resolved route 的 `ProtocolBinding::prepare_request`；
4. 允许 driver 装饰 prepared request；
5. 通过 `ModelRuntime` 发送并解码一次 attempt；
6. 对无 observable side effect 的失败执行 physical retry；
7. 对已有 side effect 的可恢复失败调用 `recover_iteration` 并开始新的 iteration；
8. 持久化 assistant result；若 terminal 为 tool use，则执行 tools 后继续；否则按 continuation decision finalize。

`TurnLimits` 控制最大 iteration 和 tool calls。`TurnOrchestrator` 不理解 provider wire fields，只消费 `ModelEvent`、`ModelAttemptResult` 和 `ModelFailure`。

`ModelRuntime` 使用 `ResolvedProviderTransport` 发送 `PreparedHttpRequest`，由 `route.binding.new_decoder()` 将响应 chunks 解码为统一 `ModelEvent`。它负责 HTTP status 检查、decoder push/finish、事件累积、terminal validation、side-effect snapshot 和 retry boundary。`execute_text_oneshot` 是无 nested Agent turn 的窄文本执行入口。

`TurnDriver` 是 Agent 与统一 runtime 之间的 side-effect contract。除 `prepare_iteration` 外，它负责 request decoration、attempt lifecycle、首次发送前提交、事件观察、assistant 持久化、工具执行、错误恢复、continuation decision 和 finalize。`ResolvedTurnDriver` 是 normal Agent turn 的实现。

## Request projection 与 adapter decoder

一次 iteration 的 request preparation 由 `prepare_protocol_stream_request` 完成，步骤是：

```text
RuntimeSnapshot + prelude + tools + metadata
  -> PromptPlanner / PromptPlan
  -> model_request_from_prompt_plan
  -> ModelRequestInput
  -> ResolvedModelRoute.binding.prepare_request
  -> PreparedHttpRequest
  -> binding.inspect_prepared_request
  -> request telemetry / logical observation
```

Responses、Completions 和 Anthropic adapter 各自提供 `ProtocolBinding` 和 `ModelStreamDecoder`。decoder 将 provider-specific SSE/JSON chunks 转换为 `Reasoning*`、`TextDelta`、`Tool*`、`Usage`、`Cache`、`ResponseMetadata` 和 `Terminal` 事件；terminal status、tool completion 和 replay state 的校验由 adapter/runtime contract 完成。

request decorator（例如 fake client）只能修改已准备的 request metadata/body contract，不替换 binding decoder，也不接管 terminal validation。

## 工具与历史副作用

模型生成的 tool calls 进入 `execute_tool_calls_and_record`。Agent 先解析 alias、scope、directive 和 permission，再执行 handler；必要时并行执行已声明为 parallel 的调用，按模型顺序 reconcile，并将 `ToolOutput` 写回 runtime snapshot。工具输出中的图片可通过外部事件保留，但文本历史使用适合 history 的结果表示。

`append_history_item` 将 protocol history item 转换为带 provenance 的 derived runtime frame。`RuntimeSnapshot` 是当前协议上下文的权威存储；request builder 只从其 provider-visible projection 构造下一次请求。

## One-shot helper

窄用途 helper 不创建嵌套 Agent turn。compaction 和其它纯文本 helper 使用：

- `preflight_resolved_oneshot_text_request`：构造并校验 resolved route 可接受的 request；
- `stream_resolved_oneshot_text_async`：通过 `ModelRuntime::execute_text_oneshot` 流式执行；
- `execute_resolved_text_oneshot`：忽略 delta 的非流式包装。

这些 helper 关闭 tools、reasoning、parallel tool calls 和 Fast Mode，要求 text completion，并直接使用调用方传入的 `ResolvedModelRoute`。它们与 normal turn 共享 adapter、transport、decoder 和 retry semantics，但不进入 Agent 的 iteration/tool loop。

## 完成、恢复与错误

成功时，driver 持久化 assistant turn，执行需要的工具或 continuation，发出 `TurnFinalized`，并由 `finish_current_turn` 清理 turn-local state。session runner 将 AgentEvent 转换为 transcript 和 session transport events。

请求创建失败或无副作用的 stream failure 可以按 resolved route 的 retry config 重试。已经产生 text、reasoning 或 tool side effect 后的失败只能进入受控 recovery iteration，不能把部分响应伪装成完整成功。非成功 HTTP 响应会在固定字节上限内读取 provider response body，保留 JSON、文本或 HTML 诊断并脱敏请求凭据，再随 session error 向用户展示；读取上限只约束无界响应，不按未知 schema 丢弃字段。provider failed/error/incomplete、decoder validation failure、permission denial、tool reconcile failure 和 interrupt 都沿错误路径返回。

## 源码索引

- `src/agent.rs` — Agent state、route authority、入口和工具/历史 side effects。
- `src/agent/protocol_stream.rs` — `run_resolved_turn_async`、`ResolvedTurnDriver`、request preparation、normal/one-shot helpers。
- `src/model_runtime/runtime.rs` — `ModelRuntime`、`TurnOrchestrator`、`TurnDriver` 和 retry/recovery contract。
- `src/model_runtime/projection.rs` — prompt plan 到 semantic model request 的投影。
- `src/model_runtime/adapters.rs` — protocol binding、prepared request 和 adapter decoder。
- `src/model_runtime/mod.rs` — resolved route、semantic request/event types 和 adapter interfaces。
