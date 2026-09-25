# 请求构建

请求构建负责将运行时上下文整理为独立于服务商的 Prompt 计划与模型请求输入。它专注于内容规划；网络协议格式化、请求体构造、参数校验与响应流解码由已解析路由上的 `ProtocolBinding` 负责。

## 请求处理流程

```mermaid
flowchart LR
    A[RuntimeSnapshot + 前言 + 工具定义 + 模型元数据]
      --> B[PromptPlanner]
    B --> C[PromptPlan]
    C --> D[canonicalize_prompt_plan]
    D --> E[model_request_from_prompt_plan]
    E --> F[ModelRequestInput]
    F --> G[ResolvedModelRoute.binding]
    G --> H[ProtocolBinding]
    H --> I[PreparedHttpRequest]
    I --> J[网络发送]
    H --> K[PreparedRequestInspection]
    K --> L[逻辑请求观测]
    L --> M[请求遥测与请求比对]
```

`src/request_builder.rs` 中的 `build_request_with_policy` 先创建规划器输入 `PromptPlannerInput`，调用 `PromptPlanner::plan` 生成计划，再通过标准化排序得到 `BuildResult`，其中包含 `PromptPlan` 与预算报告 `BudgetReport`。随后回合驱动层将计划转换为 `ModelRequestInput`，由已解析路由的协议绑定生成最终的 HTTP 请求。

各阶段的职责分工如下：

- `request_builder`：计算运行时可见性投影、筛选历史与证据、分配 Token 预算、为分段确定稳定顺序，并提供逻辑观测所需的语义元数据；
- `model_runtime::projection`：将 `PromptPlan` 转换为语义请求对象 `ModelRequestInput`，保留分段来源、工具定义、生成配置与缓存意图；
- `ProtocolBinding`：根据路由配置的协议类型、配置特征和模型能力，将语义请求编码为网络请求 `PreparedHttpRequest`，并创建匹配的流解码器；
- `ModelRuntime`：负责网络通信、收集解码器事件、处理请求终止状态、执行物理重试以及单次文本请求；
- Agent：将协议绑定提供的检查信息转换为逻辑请求观测，记录请求遥测指标，过程中不暴露原始 Prompt 文本。

`PreparedHttpRequest` 包含 HTTP 请求方法、目标 URL、协议请求头、请求体数据以及分段溯源索引。`PreparedRequestInspection` 包含请求形态特征、分段语义标识和缓存分析结果。请求构建层只消费结构化的检查数据，不反向解析协议请求体。

## 运行时可见性与 Prompt 规划

`RuntimeSnapshot` 是可见上下文的数据源。投影函数只提取处于活动状态、未被压缩排除且未被废弃的运行时帧，并转换为协议帧 `ProtocolFrame`。上下文视图与上下文树用于界面展示、状态导航和工具定位，不直接拼接入模型请求。

系统通过 `protected_start_index_for_snapshot` 计算受保护上下文的起始位置。若不存在匹配的保护帧，保护边界默认对齐到列表末尾。

`PromptPlan` 是不依赖具体服务商的分段列表。每个分段记录角色、贡献者、来源标识、稳定性等级、保留策略、保护状态、Token 估算、纯文本内容和强类型数据。强类型数据支持普通文本、结构化用户内容、助手工具调用与工具输出，采用统一的内存数据结构表达，不依赖具体服务商协议。

规划器接收模型元数据、模型标识、回合前言、快照数据、工具定义、冻结证据和保护策略，按以下顺序处理：

1. 从快照中提取面向服务商的协议帧；
2. 计算受保护边界、历史保留区间和证据预算；
3. 确保工具调用与其执行结果始终成对保留；
4. 依序组装前言、运行时素材、受保护尾部、证据分段与历史分段；
5. 生成稳定的分段标识并输出 Token 统计。

标准化排序将系统内核、运行环境、审计证据、持久上下文、历史记录与当前回合排成确定顺序，并刷新缓存元数据。连续的稳定分段前缀构成可缓存前缀。`PromptPlan::stable_prefix_hash` 仅在计划层度量前缀的结构稳定性。

## 预算与历史保留

系统根据模型元数据、输出预留、有效输入上限和工具定义，计算请求的输入预算。若模型不支持工具调用，工具定义的 Token 不计入预算。当受保护内容自身超出输入预算时，系统直接报错，不会静默裁剪受保护内容。

历史保留以完整交互单元为最小边界。助手工具调用与对应的工具输出视为一个不可拆分的批次。若保护边界落在批次内部，边界会自动扩展到该批次的起点。证据材料享有独立预算，本次请求选中的消息和标识会固定在当前逻辑回合中。

`BuildResult.budget` 记录估算值与排序后的计划用量，包含总用量、稳定部分用量、可变部分用量、可缓存前缀用量，以及边界之后的稳定用量。系统依据这些指标判定请求内容是否超出模型的输入上限。

## 语义输入与协议绑定

`src/model_runtime/projection.rs` 中的 `model_request_from_prompt_plan` 将计划分段映射为语义输入对象：

- 系统与开发者指令映射为控制分段；
- 用户、助手与工具输出映射为强类型消息 `ModelMessage`；
- 助手的思考过程、工具调用与结果保持原始结构；
- 工具规格转换为服务商中立的 `ToolDefinition`；
- 模型元数据与路由配置生成生成设置 `GenerationSettings`；
- 稳定前缀特征与模型缓存开关生成缓存意图 `CacheIntent`。

`ModelRequestInput` 同时记录消息与分段的溯源关系，便于协议绑定将生成的网络请求单元映射回原始计划。绑定属于不可变的路由组件，它不保存单次调用的输入状态，也不将服务商私有字段暴露给请求构建层。

每个 `ProtocolBinding` 负责提供：

- `prepare_request`：将语义输入编码为 `PreparedHttpRequest`；
- `inspect_prepared_request`：提取请求检查信息 `PreparedRequestInspection`；
- `new_decoder`：创建匹配的流式解码器 `ModelStreamDecoder`；
- `is_replay_compatible`：判断当前绑定与历史重放状态的兼容性。

系统内置了 Responses、Completions 和 Anthropic 绑定。各适配器独立负责特定协议的请求体封装、请求头填充、缓存标记注入、思考参数传递和流式解码，公共运行时保持协议中立。

## 缓存检查与请求遥测

缓存元数据在 `PromptPlan` 中标记稳定前缀，在 `ModelRequestInput` 中表达为缓存意图，最终由各协议绑定决定具体的网络协议字段。绑定的检查逻辑会报告是否发送了缓存提示、保留时长、本地前缀指纹与路由键。

普通回合在准备好网络请求后调用检查逻辑。`observe_prepared_model_request` 提取请求单元与分段信息，生成进程内的逻辑请求观测 `LogicalRequestObservation`。它记录请求形态摘要、语义类别、Token 估算和字节大小，不保留原始 Prompt 文本，也不会持久化到会话日志中。

Agent 将逻辑观测、请求标识、回合轮次、模型名称、协议类型、工具数量和预算整合为 `LlmRequestTelemetry`。物理重试复用同一个逻辑请求锚点；只有当发生新一轮迭代、重新规划请求或切换路由时，系统才会创建新的逻辑请求上下文。

## 历史分段与证据处理

`HistoryPublished` 保存已整理的历史产物，原有的完整对话记录依然完整保留。`HistoryApplied` 确定进入当前上下文的历史段落与表述档位。新历史块作为动态会话材料放入协议消息，不提升为系统指令；Historian 自身的静态任务约束直接写入原生高权威字段。事实材料不再重复拼入普通证据提示中，具体细节参见[历史整理与证据恢复](context-history.md)。

## 单次执行请求

上下文压缩等单次纯文本任务通过 `build_oneshot_text_request` 生成仅包含受保护用户输入的最小计划，同时禁用思考过程、工具调用、工具并发和快速模式。构建后的对象同样按照统一流程转换并发起调用：

```text
PromptPlan -> ModelRequestInput -> ProtocolBinding -> ModelRuntime
```

`preflight_resolved_oneshot_text_request` 负责校验计划构建与请求准备。流式调用由 `ModelRuntime::execute_text_oneshot` 执行，要求响应必须是纯文本，不得包含工具调用。单次调用直接沿用调用方指定的已解析路由，复用其协议绑定、传输层与解码器。

## 源码索引

- `src/request_builder.rs`：提供请求构建入口、预算管理、结果组装与逻辑请求观测。
- `src/request_builder/prompt_plan.rs`：定义 Prompt 计划结构、分段类型、标准排序与缓存元数据。
- `src/request_builder/history_budget.rs`：计算受保护上下文边界、历史保留区间与证据预算。
- `src/request_builder/runtime_projection.rs`：提取面向服务商的协议运行时帧。
- `src/model_runtime/projection.rs`：将 Prompt 计划映射为语义请求模型。
- `src/model_runtime/mod.rs`：定义协议适配器、绑定契约、请求结构与检查模型。
- `src/model_runtime/adapters.rs`：实现各协议适配器、网络请求编码与流式解码。
- `src/model_runtime/runtime.rs`：提供统一运行时、物理重试、事件汇总与单次执行逻辑。
- `src/agent/protocol_stream.rs`：连接普通回合的请求准备、状态检查、遥测与驱动逻辑。
