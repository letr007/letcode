# 模型运行时

模型运行时（Model Runtime）负责与大语言模型服务商进行底层网络通信与协议适配。它将上层规划好的语义请求转换为网络数据流，接收并解码服务端响应，归一化模型事件，并处理网络重试与错误恢复。

## 架构职责

模型运行时在整个系统中的分工如下：

```mermaid
flowchart LR
    Plan[Prompt 计划] --> Projection[语义输入 ModelRequestInput]
    Projection --> Route[权威路由 ResolvedModelRoute]
    Route --> Binding[协议绑定 ProtocolBinding]
    Binding --> Prepare[PreparedHttpRequest]
    Prepare --> Transport[网络传输 Transport / WebSocket]
    Transport --> Decoder[流式解码器 ModelStreamDecoder]
    Decoder --> Events[归一化事件 ModelEvent]
    Events --> Orchestrator[回合编排器 TurnOrchestrator]
```

- **权威路由 `ResolvedModelRoute`**：打包了已通过认证的服务商端点、网络凭据、请求头、协议配置、模型能力开关、生成参数、缓存设置与不可变的 `ProtocolBinding`；
- **协议绑定 `ProtocolBinding`**：各协议专属的数据格式转换器，负责构造网络请求体与创建匹配的流式解码器；
- **传输层 `ResolvedProviderTransport`**：管理底层 HTTP 连接池、超时控制与可选的 WebSocket 全双工通道；
- **编排器 `TurnOrchestrator`**：消费归一化事件，驱动工具执行与重试恢复循环。

## 协议适配器

系统内置针对不同服务商协议的原生适配器：

### 1. OpenAI Responses 协议

面向标准 Responses 接口规范设计：

- 支持标准 Responses 语义，端点路径默认为 `/v1/responses`；
- 支持 DeepSeek 协议变体与特性；
- 支持针对高吞吐场景的 Astra 策略与异步工具调用扩展；
- 支持在端点允许时启用 WebSocket 全双工流式传输，降低网络往返延迟。

### 2. OpenAI Chat Completions 协议

面向通用 Completions 接口规范设计：

- 兼容各主流大模型平台的 `/v1/chat/completions` 端点；
- 正确映射 `system`、`developer`、`user`、`assistant` 和 `tool` 角色；
- 处理并行的模型工具调用流，并在流结束前校验 JSON 参数完整性。

### 3. Anthropic Messages 协议

面向 Anthropic 原生 `/v1/messages` 接口规范设计：

- 将系统全局指令自动提升至顶层 `system` 字段，不混入普通对话历史；
- 原生支持自适应思考模式（Adaptive Thinking）与专属 Beta 扩展请求头；
- 原生支持显式断点缓存控制（`cache_control`），降低重复 Prompt 的调用开销。

## 响应流式解码

不同服务商返回的原始流格式（如 Server-Sent Events、JSON 块、WebSocket 数据帧）存在显著差异。

各协议的 `ModelStreamDecoder` 负责消除协议差异，将网络数据块统一解码为标准的 `ModelEvent`：

- `ReasoningDelta` 与 `ReasoningFinished`：模型的逐步思考过程与耗时；
- `TextDelta`：可见助手的增量文本回复；
- `ToolCallStarted`、`ToolCallArgumentsDelta` 与 `ToolCallFinished`：工具调用标识、增量参数流与调用完成标记；
- `UsageUpdate`：实时上报输入、输出与上下文缓存的 Token 消耗；
- `CacheUpdate`：上报服务商侧命中缓存与写入缓存的详细统计；
- `Terminal`：终结事件，记录正常停止、工具触发、异常截断或内容审查错误。

公共回合驱动层只消费这些标准事件，无需了解具体服务商的底层数据结构。

## 不透明重放状态

在连续多轮对话或会话恢复场景下，部分服务商要求客户端回传前次回合的会话锚点或私有状态标识。

系统将此类私有数据抽象为带有命名空间与版本号的 `OpaqueReplayState`：

- 仅在相同协议绑定的路由之间允许复用重放状态；
- 若会话切换到其他协议的服务商，系统主动剥离不兼容的重放标记，防止协议串线导致请求失败。

## 物理重试与恢复策略

网络请求遇到异常时，系统执行细粒度的错误分类与恢复：

- **物理重试**：在尚未向用户展示文本、未派发思考片段且未执行任何工具时发生网络超时、连接重置或 5xx 状态码，系统直接发起物理重试。重试间隔支持退避倍数与抖动补偿；
- **受控恢复**：一旦模型已经输出了部分文本或触发了工具调用，网络意外中断将被判定为带副作用的失败。系统不会重发已产生副作用的数据，而是开启新的恢复迭代，防止重复执行或状态损坏。

## 源码索引

- `src/model_runtime/mod.rs`：定义模型运行时基础特征、已解析路由结构、事件枚举与协议绑定契约。
- `src/model_runtime/adapters.rs`：实现 Responses、Completions 和 Anthropic 适配器与流解码器。
- `src/model_runtime/runtime.rs`：提供统一运行时执行器、单次纯文本调用与物理重试机制。
- `src/model_runtime/websocket.rs`：提供 Responses 协议的 WebSocket 双向流式通信实现。
- `src/model_runtime/strategy.rs`：实现 Astra 等特定服务商优化策略。
- `src/model_runtime/projection.rs`：将 Prompt 计划映射为语义请求模型。
