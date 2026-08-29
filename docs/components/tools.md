# Tools

Tools 将模型工具调用分派到本地 handler、子代理和 MCP 服务。每个 `ToolHandler` 提供名称、描述、参数 JSON Schema、strict 标志、权限类别、并发属性和异步执行入口。调用结果统一为 `ToolResult`，包含成功标志、工具名、可选数据、图片和错误信息。

## 注册、目录与契约

内置工具包括：

- 基础与交互：`util__echo`、`question`；
- 工作流与辅助：`workflow__todos`、`workflow__auto_continue`、`memory__recall`、`config__validate`；
- 子代理：`agent__explore`、`agent__fixer`；
- 文件与代码：`fs__list`、`fs__read`、`fs__write`、`fs__append`、`fs__mkdir`、`edit__apply_patch`、`code__ast_search`、`code__ast_replace_preview`；
- 命令与仓库：`shell__exec`、`search__rg`、`git__status`、`git__diff`、`git__log`；
- 网络：`web__fetch`。

`ToolRegistry` 按名称保存 handler。`register` 注册 handler，`try_register` 还会拒绝受保护名称、重复名称和不满足并行 override 的 handler，`remove` 按名称移除工具。`context__checkpoint` 和 `context__return` 不能由动态工具注册。

`spec()` 将 handler 的名称、描述、参数和 strict 标志转换为模型目录中的 `ToolSpec`。strict 工具在 debug 构建中检查 schema 的 `properties` 是否都列在 `required` 中；handler 仍执行运行时参数校验。工具目录按 scope 和可执行性筛选；调用时继续检查 scope 和权限。

权限类别为 `Read`、`Preview`、`Write`、`Command` 和 `Unknown`。handler 可以显式覆盖类别；否则按工具名分类。并发属性为 `Parallel` 或 `Exclusive`，handler 默认是 exclusive，只有明确声明支持重叠调用时才是 parallel。registry 可以把工具收紧为 exclusive，但不能把未声明 parallel 的 handler 配置为 parallel。并发批处理和结果协调由 Agent 执行层处理，单次 registry 调用不会自动并行执行所有工具。

工具 scope 包括 `FullAccess` 和 `ReadOnlyExplorer`。scope 同时影响目录生成和调用执行：调用时先检查 scope，再查找 handler；不允许的调用返回 scope 错误，未找到的 handler 返回 unknown 错误。

## 调用与结果

模型调用进入 `execute_tool_call`。执行层解析工具别名并建立 trace span；JSON 参数无法解析时记录为 `Rejected` 并发送完成事件。有效参数依次经过可执行性、scope、执行 directive、写入目标预解析、外部 workspace 访问和权限决策，然后调用 registry 中的 handler。

权限允许后，Agent 为 handler 创建 `ToolExecutionContext`，其中包含外部 workspace 授权、问题回调和预先绑定的写入或 patch 目标。registry 的 streaming 调用再次检查 scope，查找 handler，调用 `execute_streaming`，并将 handler 错误转换为 `ToolResult::err`。工具输出可通过 `Stdout` 和 `Stderr` 增量发送。

`ToolResult::ok` 返回 `ok: true` 和数据；handler 或 registry 错误返回 `ok: false` 与 `ToolError`。Agent 的执行记录区分 `Executed`、`Rejected` 和 `TimedOut`，并保留 invalid JSON、directive blocked、scope denied、delegation scope denied、permission policy denied 和 user denied 等拒绝原因。调用过程会发送 started、output delta、cancelled 和 finished 事件，并产生工具执行摘要。

## 权限与工作区授权

`PermissionSessionState::approval_snapshot` 根据工具、参数、权限类别、执行 directive、权限 mode、外部 workspace 访问和内部工具标记计算 `Allow`、`Ask` 或 `Deny`。需要询问时，执行层通过 callback 或自动审查取得 `Deny`、`AllowOnce` 或 `AllowAlways`；只有在权限 mode 支持 session grant、资源存在且 permission generation 未变化时，`AllowAlways` 才会写入 grant。

`fs__write` 和 `fs__append` 会预先绑定可写目标，`edit__apply_patch` 会预先解析 patch 目标。授权后，执行阶段重新检查路径、父目录和文件身份；目标发生变化时拒绝继续。

外部 workspace 路径会形成 `ExternalWorkspaceAccess`，在 default/auto 等 mode 下可能将原本允许的决定提升为 `Ask`。letcode 自身 fold artifact 目录的读取有单独的 trusted artifact 判断，仅适用于对应 artifact 路径。

## 超时、取消与进程清理

普通工具调用可按工具名应用 `non_shell_tool_timeout_secs`。执行层用 `tokio::select!` 同时处理工具结果、增量输出和 timeout；超时后发送 `ToolCallCancelled`，排空已产生的输出，返回超时错误，并将记录标为 `TimedOut`。`ToolHandler` 没有统一的 cancellation token；future 停止后的资源清理取决于具体 handler 和底层资源。

`shell__exec` 默认超时为 300 秒，也可通过 `timeout_secs` 指定，但不能超过 `MAX_COMMAND_TIMEOUT_SECS`。命令在 workspace 根目录启动，stdout 和 stderr 分别捕获；超时会终止并等待子进程，结果包含 `command timed out after Ns`。streaming 路径实时发送 stdout/stderr，超时后终止子进程并收集剩余输出，结果中的 `success` 为 false。

本地 MCP 会话为 JSON-RPC 写入、刷新和读取设置 server timeout；关闭时先关闭 stdin，等待进程退出，必要时优雅终止、强制终止或交给 reaper。远程 MCP 使用带 timeout 的 HTTP client，通过 HTTP/JSON 或 SSE 解析。远端服务已经开始的操作是否停止，不由客户端超时单方面保证。

`workflow__auto_continue` 只校验 `enabled` 为 boolean 并返回参数；自动继续、显式中断、shutdown、permission 请求和 question 请求由上层流程处理。

## 专用工具后端

- `fs__*` 处理工作区文件和目录读写，并应用行数、字节数、图像大小和路径限制；
- `search__rg` 处理文本搜索；
- `git__status`、`git__diff` 和 `git__log` 在 workspace 根目录执行固定 git 子命令；
- `edit__apply_patch` 处理精确文本替换，并在授权和 worker 阶段绑定 patch 目标；
- `code__ast_search` 和 `code__ast_replace_preview` 使用 AST-aware 后端；preview 不写文件，实际写入通过显式 patch 工具完成；
- `web__fetch` 接受公开 URL，设置请求和总超时，限制重定向、内容类型和响应大小；较大的响应会折叠为预览及本地 artifact；
- `workflow__todos` 校验最多 100 项、字段长度、唯一 ID 和状态值；
- `question` 通过 `ToolExecutionContext.question_handler` 连接交互运行时，没有交互 runtime 时返回错误；
- `memory__recall` 调用 memory 域查询；`config__validate` 调用配置解析和校验实现，并声明为 read/parallel。

`shell__exec` 的描述要求在专用工具适用时优先使用专用工具，并为循环、监听器和长任务设置退出或超时边界。

## MCP

MCP server 配置包含 enabled 状态、`timeout_ms` 和 transport。发现阶段对启用 server 调用 `tools/list`：本地 transport 使用 stdio 子进程，remote transport 使用 HTTP；多个 server 的发现可并发执行，结果按配置顺序返回。禁用 server 不会被连接；单个 server 失败记录为 Offline，不影响其他 server 的发现结果。

发现结果包含 MCP 工具名称、描述和 `inputSchema`。包装为 `McpTool` 时，工具名规范化为 `<server>__<tool>`，非 ASCII 字母、数字或下划线的字符压缩为下划线，空名称组件会被拒绝。描述带有 `[MCP <server>]` 前缀，缺失描述时生成默认描述。

`McpTool` 将发现到的 schema 作为 parameters，执行时按 transport 建立会话，发送 `initialize`、`notifications/initialized` 和 `tools/call`。本地调用通过 JSON-RPC stdio 与 MCP server 通信；远程调用通过 HTTP 请求并维护 MCP session header。

MCP handler 在 letcode 中的权限类别为 `Read`，并发属性为 `Exclusive`，strict 为 false。MCP server 实际执行的操作、鉴权要求和资源生命周期由 server 与 transport 决定。

MCP 和本地工具共享 Agent 的目录、scope、directive、权限决策、审批、事件和结果记录链路。MCP 工具没有当前文件目标的 prepared resource，但仍会经过可执行性、scope、directive 和 permission decision 检查。

## 错误与事件

调用失败通常表现为 `ToolResult { ok: false, error: ... }`；registry 将 handler 返回的错误转换为 recoverable tool error。Agent 的执行记录区分 `Executed`、`Rejected` 和 `TimedOut`，并保留拒绝原因。调用过程发送工具输出增量、取消和完成事件，供 Transcript、TUI 和 audit 使用。
