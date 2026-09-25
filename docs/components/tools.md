# Tools

Tools 模块将模型工具调用分发到本地处理器、子代理任务池或 MCP 外部服务。每个 `ToolHandler` 提供工具名称、功能描述、参数结构模式、严格模式标记、权限分类、并发能力定义与异步执行入口，调用结果统一封装为 `ToolResult`。

## 内置工具目录

系统内置以下工具：

- 基础与交互：`util__echo`、`question`；
- 工作流与辅助：`workflow__todos`、`workflow__auto_continue`、`memory__recall`、`config__validate`；
- 子代理委派：`agent__explore`、`agent__fixer`、`agent__oracle`、`agent__designer`、`agent__librarian`、`agent__general`；
- 任务池控制：`agent__jobs`、`agent__status`、`agent__wait`、`agent__cancel`；
- 文件与代码：`fs__list`、`fs__read`、`fs__write`、`fs__append`、`fs__mkdir`、`edit__apply_patch`、`code__ast_search`、`code__ast_replace_preview`；
- 命令与版本库：`shell__exec`、`search__rg`、`git__status`、`git__diff`、`git__log`；
- 网络通信：`web__fetch`。

六个专家委派工具各有专长：`explorer` 用于只读代码探索，`fixer` 负责限定范围的代码修复，`oracle` 提供根因分析与风险审查，`designer` 负责方案设计与交互梳理，`librarian` 整理资料与检索代码，`general` 处理明确的通用辅助任务。`reviewer` 与 `historian` 属于内部系统专家，不提供委派工具；任务控制工具用于管理已有任务，不创建新专家。`context__search` 与 `context__expand` 提供受控的历史记录与证据检索，不恢复旧的运行时状态。

`ToolRegistry` 按名称统一管理工具处理器，提供 `register`、`try_register` 和 `remove` 操作。受保护的系统检查点与控制指令禁止被动态覆盖。`spec()` 方法将处理器描述转换为面向模型的结构化定义，系统根据当前调用方权限和运行环境动态筛选可见工具。

## 作用域、权限与并发控制

调用作用域分为完全访问（`FullAccess`）与只读探索（`ReadOnlyExplorer`）。执行时先验证调用方作用域，再查找匹配的处理函数；未授权调用返回权限错误，未注册名称返回未知工具错误。

工具权限划分为读取（`Read`）、预览（`Preview`）、写入（`Write`）、系统命令（`Command`）与未知（`Unknown`）五类。处理器默认排他执行（exclusive），仅显式声明支持重叠调用的处理器允许并行执行（parallel）。Agent 负责处理并行批次的权限预检、并发调度与结果调和，注册表单次调用不会盲目并发所有工具。

权限决策综合考虑工具类型、调用参数、权限等级、会话模式、工作区边界与内部工具标记，产出允许（Allow）、询问（Ask）或拒绝（Deny）。会话级授权只有在上下文约束与当前授权版本均有效时才记录。

子代理工具额外受到规范化任务、路径作用域、独占写锁、专家策略和后台支持能力的约束。任务控制工具严格操作已存在的任务池实例，不隐式启动子任务。

## 文件操作与代码补丁

文件工具 `fs__*` 统一操作工作区内的文件与目录，强制执行路径规范化、文件大小限制、行数截断和作用域检查。`fs__write` 与 `fs__append` 在执行前先锁定目标路径；`edit__apply_patch` 预先解析补丁受影响的文件。获得用户授权后，执行引擎在真正修改前重新核验目标路径与文件特征，若文件被外部篡改则立即终止。

`code__ast_search` 和 `code__ast_replace_preview` 依赖语法树分析后端。预览工具仅返回代码差异（Diff），不落盘修改文件；实际变更必须通过显式调用补丁工具完成。`search__rg` 负责高性能文本检索。`config__validate` 仅解析和检验指定的配置文件，不自动应用变更。

## 命令执行与超时控制

`shell__exec` 在工作区根目录下启动子进程，独立捕获标准输出与标准错误。命令默认超时为 300 秒，调用方可指定更短时长，但不能突破上限。发生超时时，系统主动终止子进程并回收资源，结果标记为失败并附带超时说明；流式输出通道同样在回收子进程后汇总剩余内容。

非 Shell 工具采用各自独立的超时配置。执行层通过异步等待统一调度操作、增量输出与超时监听；超时时触发 `ToolCallCancelled` 并返回对应错误。具体处理器的底层资源回收由各自的实现逻辑保证。

`workflow__auto_continue` 负责维护自动继续标记。自动继续推进、用户显式中断、进程关闭、权限弹窗和交互式问题均由会话引擎统一调度。MCP 客户端超时仅中断本地等待，无法保证远端服务端已经启动的任务立即停止。

## MCP 外部工具扩展

系统支持通过标准 MCP 协议接入外部能力。开启服务时，系统对可用服务端调用 `tools/list` 发现工具：本地服务通过子进程标准输入输出通信，远程服务基于 HTTP 协议交互。多个服务端支持并发发现并按配置顺序合并；单个服务离线不影响其他服务注册。

MCP 工具统一使用 `<server>__<tool>` 格式命名，参数模式作为输入模式注入模型。工具调用前建立或复用会话通道，依序完成初始化与方法调用。MCP 处理器在系统中默认视为只读与排他工具，依然由 Agent 统一管理，执行作用域校验、权限判定、超时控制、事件派发和结果处理。

## 执行事件与调用结果

工具调用完整派发启动、输出增量、取消和结束事件，并如实记录成功（`Executed`）、被拒绝（`Rejected`）或超时（`TimedOut`）。常见拒绝原因包含 JSON 解析错误、作用域越界、委派路径冲突、权限策略拒绝、自动审查退回补充说明以及用户主动取消。

调用成功时通过 `ToolResult::ok` 返回成功标识与输出内容；处理失败时返回 `ToolError` 错误详情。工具执行摘要持久化记录工具名称、产生副作用、状态码、拒绝原因和关键参数，供前端展示与后续审计。

子代理委派在前台等待完成时返回非活动标记（`active: false`），后台启动时返回运行中状态与后台标记；任务看板与控制工具直接使用结构化任务对象，不把后台状态伪装成普通工具输出。

## 源码索引

- `src/tool/registry.rs`：实现工具注册表、作用域管理、规格定义与流式调用。
- `src/tool/delegation.rs`：定义专家委派模式、参数规范化与路径检查。
- `src/tool/fs.rs`、`src/tool/apply_patch.rs`：提供文件系统操作与代码补丁处理器。
- `src/tool/command.rs`：负责系统命令执行、输出流式捕获、超时监控与进程清理。
- `src/tool/workflow.rs`、`src/tool/memory.rs`、`src/tool/config_validate.rs`：实现待办事项、项目记忆与配置校验工具。
- `src/tool/code_analysis.rs`、`src/tool/search.rs`：提供代码语法树分析与文本搜索工具。
- `src/tool/git.rs`、`src/tool/web_fetch.rs`、`src/tool/question.rs`：实现版本控制、网页拉取与交互提问工具。
- `src/subagent/pool.rs`、`src/session/subagent_delegate.rs`：管理子代理任务池与控制工具行为。
