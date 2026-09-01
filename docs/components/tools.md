# Tools

Tools 将模型工具调用分派到本地 handler、subagent pool 和 MCP 服务。每个 `ToolHandler` 提供名称、描述、参数 schema、strict 标志、权限类别、并发属性和异步执行入口；调用统一返回 `ToolResult`。

## 工具目录

当前内置目录包括：

- 基础与交互：`util__echo`、`question`；
- workflow 与辅助：`workflow__todos`、`workflow__auto_continue`、`memory__recall`、`config__validate`；
- subagent experts：`agent__explore`、`agent__fixer`、`agent__oracle`、`agent__designer`、`agent__librarian`、`agent__general`；
- subagent jobs：`agent__jobs`、`agent__status`、`agent__wait`、`agent__cancel`；
- 文件与代码：`fs__list`、`fs__read`、`fs__write`、`fs__append`、`fs__mkdir`、`edit__apply_patch`、`code__ast_search`、`code__ast_replace_preview`；
- 命令与仓库：`shell__exec`、`search__rg`、`git__status`、`git__diff`、`git__log`；
- 网络：`web__fetch`。

六个 expert delegation tool 的用途是：explorer 只读探索、fixer 限定范围修复、oracle 根因/风险判断、designer 设计梳理、librarian 资料和证据整理、general 只读通用辅助。reviewer 是独立的 permission review 专家，不是 delegation tool；job control tools 也不创建新的 expert。

`ToolRegistry` 按名称注册 handler。`register`、`try_register` 和 `remove` 维护目录；受保护的 context checkpoint/return 名称不能由动态工具注册。`spec()` 将 handler 转换为模型可见 `ToolSpec`，scope、可执行性和当前 runtime 能力会进一步筛选目录。

## Scope、权限与并发

scope 包括 `FullAccess` 和 `ReadOnlyExplorer`。scope 同时影响目录和执行：调用时先检查 scope，再查找 handler；未授权返回 scope error，未知名称返回 unknown error。

权限类别为 `Read`、`Preview`、`Write`、`Command` 和 `Unknown`。默认 handler 为 exclusive；只有显式声明支持重叠调用的 handler 才能成为 parallel。Agent 负责并行 batch 的 permission preflight、执行和结果 reconcile，registry 单次调用不会自动并行所有工具。

permission decision 综合工具、参数、permission class、execution directive、permission mode、外部 workspace access 和 internal-tool 标记，产生 Allow/Ask/Deny。`AllowAlways` 只有在 session grant 条件和 generation 仍有效时才写入 permission session。

subagent tools 额外受 normalized task、path scope、owned-path lock、expert policy、background capability 和 takeover route gate 约束。`agent__wait`、`agent__status`、`agent__jobs`、`agent__cancel` 只操作已存在的 Pool run，不会隐式创建 child。

## 本地文件与 patch

`fs__*` 处理 workspace 文件和目录，执行路径 canonicalization、大小/行数/图像限制和 scope checks。`fs__write`/`fs__append` 在执行前绑定 writable target；`edit__apply_patch` 预解析 patch targets。授权后，执行阶段重新检查路径、父目录和 file identity，目标变化则拒绝。

`code__ast_search` 和 `code__ast_replace_preview` 使用 AST-aware backend；preview 只返回 diff，不写文件，实际修改必须经过显式 patch 工具。`search__rg` 负责文本搜索。`config__validate` 只解析并校验指定 letcode 配置，不应用配置变更。

## Shell、timeout 与取消

`shell__exec` 在 workspace root 启动进程，分别捕获 stdout/stderr，默认 timeout 为 300 秒，调用方可缩短但不能超过系统上限。超时会终止并等待子进程，结果标记失败并保留 timeout 信息；streaming path 同样清理子进程并收集剩余输出。

普通工具按工具名使用 non-shell timeout。执行层通过 `tokio::select!` 处理 future、增量输出和 timeout；timeout 会发出 `ToolCallCancelled` 并返回超时结果。handler 没有统一 cancellation token，future 停止后的底层资源清理由具体 handler 决定。

`workflow__auto_continue` 只验证 enabled 参数；自动继续、显式 interrupt、shutdown、permission 和 question 的状态推进由 Agent/SessionEngine 处理。MCP client timeout 不保证远端 server 已经开始的操作被停止。

## MCP

MCP discovery 对 enabled server 调用 `tools/list`：stdio server 使用子进程，remote server 使用 HTTP。多个 server 可并发 discovery，并按配置顺序汇总；单个 server offline 不阻塞其它 server。

MCP tool 名称规范化为 `<server>__<tool>`，schema 作为 parameters，调用前建立/复用 transport session，发送 initialize、initialized notification 和 tools/call。MCP handler 在 letcode 中默认属于 Read/Exclusive，仍经过 Agent 的 scope、directive、permission、timeout、event 和 result 链路。

## 事件与结果

一次工具调用会产生 started、output delta、cancelled 和 finished 事件，并记录 `Executed`、`Rejected` 或 `TimedOut`。拒绝原因包括 invalid JSON、directive blocked、scope denied、delegation scope denied、permission policy denied 和 user denied。

`ToolResult::ok` 返回 `ok: true` 与可选 data；handler/registry failure 返回 `ok: false` 与 `ToolError`。工具 execution summary 会保留 tool identity、effects、status、拒绝原因和必要的 primary path/command，供 Agent、Session、Transcript、TUI 和 audit 使用。

subagent delegation 的前台结果带 `active: false`，后台启动结果带 `status: running`、`active: true` 和 `background: true`；jobs/status/wait/cancel 使用各自的 run/job data structures，不把 job 状态伪装成普通 tool output。

## 源码索引

- `src/tool/registry.rs` — handler registration、scope、spec 和 streaming call。
- `src/tool/delegation.rs` — expert delegation schema、normalization 和 path scope。
- `src/tool/fs.rs`、`src/tool/apply_patch.rs` — filesystem/patch handlers。
- `src/tool/command.rs` — shell execution、streaming、timeout 和 process cleanup。
- `src/tool/workflow.rs`、`src/tool/memory.rs`、`src/tool/config_validate.rs` — workflow/auxiliary handlers。
- `src/tool/code_analysis.rs`、`src/tool/search.rs` — AST/text analysis tools。
- `src/tool/git.rs`、`src/tool/web_fetch.rs`、`src/tool/question.rs` — repository/network/interaction tools。
- `src/subagent/pool.rs`、`src/session/subagent_delegate.rs` — expert jobs and control tool behavior。
