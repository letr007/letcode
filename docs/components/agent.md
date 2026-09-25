# Agent

## 运行时边界

`Agent` 是持有会话运行状态的核心执行对象。它保存当前逻辑路由 `ModelRoute`、已解析运行时路由、运行时目录、`RuntimeSnapshot`、协议历史投影、工具注册表、权限会话、回合状态、上下文压缩配置、重试策略、技能集合与子代理工厂。

服务商的可执行身份由完整的运行时路由决定，不能仅凭模型名称字符串标识。`resolved_model_route` 是当前回合与辅助请求的权威路由。它包含模型标识、协议类型、端点地址、认证信息、请求头与查询参数、模型能力、生成设置、缓存与重试策略、传输层，以及不可变的 `ProtocolBinding`。如果普通回合未安装已解析路由，执行会直接报错。

## 路由准备与权威路由

`PrimaryRouteFactory::prepare_route` 负责校验配置中的 `ModelRoute`，并将其解析为 `PreparedPrimaryRoute`。安装路由时，系统通过 `PreparedPrimaryRouteInstall::apply` 或 `Agent::apply_prepared_route`，一次性更新逻辑路由、协议类型、模型元数据、重试策略和已解析路由。若需要同时更新路由与运行时标识，使用 `set_model_route_authority`。

`active_protocol` 用于状态展示与兼容性判断；发起请求时以已安装的 `ResolvedModelRoute` 及其绑定为准。切换模型或重新安装路由后，下一次请求投影会按新路由构造。

## 回合入口

`Agent::run` 是非流式调用的便捷入口。交互式调用使用 `run_stream_async`，内容型调用使用 `run_stream_content_with_interactions_async`。后者会安装本次调用的问题处理器（Question Handler），并传递以下回调：

- `on_delta`：输出可见的助手文本增量；
- `on_event`：派发思考过程、工具调用、Token 用量、重试、上下文压缩及生命周期事件；
- `approve`：接收权限审批决策；
- `question handler`：响应模型提出的交互式问题。

入口先检查当前是否已安装有效路由，再将实际执行委托给 `src/agent/protocol_stream.rs` 中的 `run_resolved_turn_async`。因此普通回合、协议选择与路由权威均在同一条运行时路径上收敛。

## 普通回合执行

`run_resolved_turn_async` 负责普通 Agent 回合的执行准备工作：

1. 克隆当前已解析路由，并获取对应协议；
2. 生成包含所选技能的回合前言（Prelude）；
3. 记录当前活跃历史长度，设置当前回合保护边界，并追加用户输入；
4. 发出 `TurnStarted` 事件；
5. 创建 `ResolvedTurnDriver`，连接 Agent 状态、文本与事件回调、审批通道、回合计数器、缓存用量、响应元数据与仿真装饰器；
6. 使用 `TurnOrchestrator::new(ModelRuntime::default(), TurnLimits)` 执行该回合；
7. 返回最终文本；若输出为空，返回 `No response content`。

不同协议的处理细节收敛在模型运行时的绑定准备、传输层和适配器解码器中；Agent 驱动层则专注于管理回合状态、历史记录、工具调用、事件派发和会话副作用。

## 编排器与驱动层

`TurnOrchestrator` 负责统一控制迭代、重试与流程继续。在每次迭代中：

1. 检查迭代次数上限；
2. 由 `TurnDriver::prepare_iteration` 生成 `ModelRequestInput`；
3. 调用已解析路由的 `ProtocolBinding::prepare_request`；
4. 允许驱动层装饰准备好的请求；
5. 通过 `ModelRuntime` 发送请求并解码单次尝试结果；
6. 若失败且未产生可观察副作用，触发物理重试；
7. 若产生副作用且错误可恢复，调用 `recover_iteration` 开启新一次迭代；
8. 持久化模型回复。若终止状态为工具调用，执行工具后继续回合；否则根据继续决策结束流程。

`TurnLimits` 控制最大迭代次数与工具调用次数。`TurnOrchestrator` 不解析具体服务商的网络传输字段，只消费统一的 `ModelEvent`、`ModelAttemptResult` 和 `ModelFailure`。

`ModelRuntime` 通过 `ResolvedProviderTransport` 发送 `PreparedHttpRequest`。响应数据块由 `route.binding.new_decoder()` 统一解码为 `ModelEvent`。它负责校验 HTTP 状态码、推进解码器状态、收集事件、验证终止状态、记录副作用快照并管理重试边界。`execute_text_oneshot` 是用于纯文本生成的窄接口，不触发嵌套回合。

`TurnDriver` 定义了 Agent 与统一运行时之间的副作用交互契约。除 `prepare_iteration` 外，它还负责请求装饰、尝试生命周期、初次发送前提交、事件观察、助手回复持久化、工具执行、错误恢复、继续决策和完成收尾。`ResolvedTurnDriver` 是普通 Agent 回合的具体实现。

## 请求投影与适配器解码

一次迭代的请求准备由 `prepare_protocol_stream_request` 完成，具体执行过程如下：

```text
RuntimeSnapshot + 前言 + 工具定义 + 元数据
  -> PromptPlanner / PromptPlan
  -> model_request_from_prompt_plan
  -> ModelRequestInput
  -> ResolvedModelRoute.binding.prepare_request
  -> PreparedHttpRequest
  -> binding.inspect_prepared_request
  -> 请求遥测与逻辑观测
```

Responses、Completions 和 Anthropic 适配器各自提供对应的 `ProtocolBinding` 与 `ModelStreamDecoder`。解码器将特定服务商的 SSE 或 JSON 数据块，转换为统一的思考过程、文本增量、工具调用、Token 用量、缓存状态、响应元数据和终止事件。终止状态校验与重放兼容性判断由适配器和运行时契约共同保证。

请求装饰器（例如客户端特征伪装）只能调整请求元数据与请求体结构，不能替换解码器，也不能接管终止状态校验。

## 工具调用与历史副作用

模型生成的工具调用由 `execute_tool_calls_and_record` 处理。Agent 先解析工具别名、作用域与权限策略，再执行具体处理逻辑。系统并发执行声明了并行能力的工具调用，并按模型返回的顺序调和结果，最后将 `ToolOutput` 写入运行时快照。工具输出中的图片可通过外部事件传递，文本历史则保存规范化的结果文本。

`append_history_item` 将协议历史项转换为带有来源标记的运行时帧。`RuntimeSnapshot` 是当前上下文的权威数据源；请求构建器仅根据面向服务商的投影数据构造下一次请求。

## 单次辅助方法

单次辅助方法用于纯文本处理，不创建嵌套的 Agent 回合。上下文压缩和其他纯文本任务使用以下方法：

- `preflight_resolved_oneshot_text_request`：构建并校验路由支持的请求；
- `stream_resolved_oneshot_text_async`：通过 `ModelRuntime::execute_text_oneshot` 发起流式执行；
- `execute_resolved_text_oneshot`：非流式封装，忽略中间增量文本。

这些辅助方法关闭工具调用、思考过程、并行工具和快速模式，强制要求纯文本输出，并直接使用调用方提供的已解析路由。它们与普通回合共享适配器、传输层、解码器与重试语义，但不进入 Agent 的迭代与工具循环。

## 完成、恢复与错误处理

执行成功后，驱动层持久化助手回复，执行后续工具或触发继续逻辑，发出 `TurnFinalized` 事件，并由 `finish_current_turn` 清理回合局部状态。Session Runner 随后将 `AgentEvent` 转换为持久化记录与传输流事件。

未产生副作用的流异常或请求构建失败，可根据路由配置自动重试。一旦产生文本、思考过程或工具调用等副作用，失败只能进入受控的恢复迭代，不能把部分输出视为成功。请求失败时，系统最多读取固定上限字节的响应体。诊断信息（JSON、文本或 HTML）在脱敏凭据后随错误返回给用户。读取上限只用于截断过长响应，不会主动过滤未知字段。服务商错误、解码器校验失败、权限拒绝、工具调用冲突和用户中断均统一沿错误路径返回。

## 源码索引

- `src/agent.rs`：管理 Agent 状态与路由，处理工具调用与历史副作用。
- `src/agent/protocol_stream.rs`：实现普通回合调度、驱动层逻辑、请求准备与单次执行辅助函数。
- `src/model_runtime/runtime.rs`：提供统一编排器、物理重试与恢复契约。
- `src/model_runtime/projection.rs`：将 Prompt 计划转换为语义化请求结构。
- `src/model_runtime/adapters.rs`：实现各协议绑定、请求构造与流式解码器。
- `src/model_runtime/mod.rs`：定义已解析路由、模型请求结构、事件契约与适配器接口。
