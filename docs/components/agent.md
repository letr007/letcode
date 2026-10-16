# Agent

## 运行时职责与状态

`Agent` 保存会话运行状态，驱动单次回合循环。它管理当前逻辑路由、已解析运行时路由、历史快照 `RuntimeSnapshot`、前言、工具注册表、权限会话，以及迭代次数与工具调用上限。

上下文组装说明见 [request-building.md](request-building.md)，协议与网络层实现见 [model-runtime.md](model-runtime.md)。

## 路由解析与安装

发起模型调用需要完整的运行时配置，仅凭模型名称无法确定网络行为。

`PrimaryRouteFactory::prepare_route` 读取配置中的 `ModelRoute`，验证端点、凭据和协议设置，生成包含协议选项、模型目录、重试配置与已解析运行时路由的 `PreparedPrimaryRoute`。

安装完整路由使用 `PreparedPrimaryRouteInstall::apply` 或 `Agent::apply_prepared_route`，一次性更新逻辑路由、协议选项、模型目录、重试配置与 `ResolvedModelRoute`。`Agent::set_model_route_authority` 仅用于直接绑定逻辑路由与已解析路由，并同步模型名称。

普通回合未安装已解析路由时，执行会直接报错返回。切换模型或更新路由后，Agent 会让当前请求投影失效，下次请求按新配置重新构建。

## 回合执行流程

Agent 提供三种执行入口。

- `run` 执行非流式请求，等待全部生成完成后返回文本。
- `run_stream_async` 执行流式请求，通过回调通知文本增量与生命周期事件。
- `run_stream_content_with_interactions_async` 支持交互提问，允许模型向用户提出结构化问题。

入口方法在确认路由就绪后，调用 `src/agent/protocol_stream.rs` 中的 `run_resolved_turn_async` 启动执行。

回合按以下步骤执行。

1. 结合用户选中的技能构建回合前言，记录当前历史长度作为受保护起点，追加用户输入，并触发 `TurnStarted` 事件。
2. 创建 `ResolvedTurnDriver`，挂接状态数据、文本增量回调、事件回调与权限通道。
3. 初始化 `TurnOrchestrator`，传入运行实例与迭代限制。
4. 检查迭代次数与工具调用次数，超过配置上限时返回错误。
5. 驱动层构造请求，模型运行时发送请求并解码响应。失败时按下文的重试与恢复条件处理。
6. 模型返回工具调用时，执行工具并写入历史，随后进入下一轮迭代。
7. 模型返回文本并持久化后，驱动层决定继续迭代或结束回合。结束时发出 `TurnFinalized`，清理回合状态并返回文本。输出为空时返回 `No response content`。

## 工具执行与记录

模型在回复中给出工具调用时，驱动层调用 `execute_tool_calls_and_record` 处理整组工具。

### 权限与作用域检查

Agent 根据工具作用域检查工具是否可用，并对照会话权限规则核验调用。如果权限配置为拒绝，或者自动审批拒绝，工具直接生成拒绝记录；若已持有会话授权，无需用户确认即可放行；需要人工审批时再调用外部交互通道。

权限拒绝与普通工具执行错误通常打包为错误结果写回历史，供模型在后续迭代中读取并调整，不会直接中断整个回合。

### 并发与记录顺序

模型元数据开启并行工具调用时，连续出现的同类工具可以分组并发执行。

- 普通并行工具要求工具自身声明支持并行、工具作用域允许调用，且在预检中通过无交互授权放行。所有并行任务完成后，系统按模型生成的原始调用顺序记录结果。
- 子代理工具由子代理池并发执行，任务完成后按实际完成顺序逐项发布事件并写入历史。详细规则见 [subagents.md](subagents.md)。
- 不满足并发条件的工具转为串行执行。

### 批次等待与历史写入

前台当前批次的所有工具执行完毕后，Agent 才会启动下一次模型迭代。如果部分工具尚未返回，系统不会提前向模型发送下一轮 Prompt。

工具产出的文本数据与图片同时写入历史记录的 `ToolOutput`。详细工具系统设计见 [tools.md](tools.md)。

## 单次辅助调用

`stream_resolved_oneshot_text_async` 与 `execute_resolved_text_oneshot` 用于上下文压缩或摘要生成等单次文本请求。

这些辅助方法关闭工具调用、思考过程、并行工具和快速模式，不创建驱动层实例，也不修改会话历史。请求直接使用传入的已解析路由，复用底层的协议绑定与网络传输。

## 错误处理与恢复边界

调用异常根据是否产生副作用与错误类别分别处理。

- 未产生可见文本、思考片段或工具调用的请求失败，在满足可重试错误类别且未超过尝试次数时，按配置退避重试。
- 产生部分输出后中断的请求失败，在满足可恢复错误类别且未超过恢复上限时，系统保留已有片段并开启恢复迭代。不满足恢复条件时终止回合。

服务商返回 HTTP 错误时，系统读取固定截断上限以内的响应体，脱敏凭据后返回错误信息。用户取消或主动中断沿错误路径返回。

## 源码索引

- `src/agent.rs` 定义 `Agent` 结构体，管理会话状态、路由安装与工具调用调度。
- `src/agent/protocol_stream.rs` 实现回合调度 `run_resolved_turn_async`、驱动层 `ResolvedTurnDriver` 与单次辅助方法。
- `src/agent/tool_execution.rs` 实现普通并行工具批量执行与子代理并发调度。
- `src/model_runtime/runtime.rs` 实现编排器 `TurnOrchestrator` 与物理重试逻辑。
- `src/model_runtime/mod.rs` 定义已解析路由 `ResolvedModelRoute` 与底层模型事件。
