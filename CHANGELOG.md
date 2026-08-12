# Changelog

本文件记录项目的重要变更。

格式参考 [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)，
版本号遵循 [Semantic Versioning](https://semver.org/)。

## [Unreleased]

## [0.3.0] - 2026-08-12

### Added

- Markdown 在 TUI 中原生渲染 LaTeX 数学公式与 Mermaid 图表；支持流程图、时序图、状态图、类图、ER 图、甘特图、思维导图和时间线，并在无法渲染时保留源码回退
- 新增 `/thoughts <compact|titles|full>`，可切换思考过程的紧凑、仅标题和完整显示模式
- 专家配置新增 `allowed_models`，委派时可在允许的 `provider/model` 路由中单次选模，并在接管子会话时恢复原路由

### Changed

- Mermaid 终端布局与语法覆盖扩展：支持更多流程图方向、节点与连线样式、子图和标签，改进正交路由、箭头、时序语法及各类图表的窄屏布局
- 会话运行时与 TUI 交互收敛：活动回合中的设置变更按命令语义排队或拒绝，剪贴板粘贴可区分文本与图片，工具卡片、选择与窄终端渲染更稳定
- 思考过程和 auto-review 决策整合进时间线；审批结果支持展开理由，移除重复通知
- Footer 优先展示 token 状态，并按空间显示 Git 分支、上下文分支和帮助提示；压缩期间使用扫描动画
- 配置热重载保留当前会话的模型、专家路由和设置；已从配置目录移除的活动路由可继续用于当前会话，直到显式切换或新建会话

### Fixed

- 恢复会话时修复未正常结束的孤立回合，并清理未完成的工具调用与子代理运行状态
- auto 模式仅将需要审批的工具调用交给 reviewer，避免重复或无意义审批
- 父会话与子会话的时间线、展开状态和运行时投影相互隔离，避免切换视图后状态串扰
- 配置热重载不再覆盖会话级 Fast Mode、推理等级等运行时设置

## [0.2.0] - 2026-08-06

### Added

- 子代理目录授权：`allowed_paths` 只读、`owned_paths` 读写、`forbidden_paths` 优先；越界硬拒绝，scope 内结构化工具预授权，避免 auto 模式下逐文件重复审批
- 子代理 PermissionSession 隔离：只继承权限模式，不继承 AllowAlways grants；父子/兄弟互不影响；Shell 仍按完整命令 AllowAlways
- auto 模式下普通子代理继承 reviewer 服务；reviewer 子代理不继承，防止递归审批；并发 auto-review 串行化，避免 reviewer busy
- TUI 主题：内置部分主题，并从 `themes/*.toml` 加载可编辑主题，选择时热重载
- 内置 `customize-letcode` skill，以及 `config__validate` / `letcode config validate`
- 退出横幅打印 `letcode resume <id>`；新增同名 CLI，可直接恢复指定会话
- 已选 skill 在时间线中持续可见

### Fixed

- Windows 默认配置路径：无 `HOME` 时回退 `USERPROFILE`（修复 exe 启动即报 `HOME is not set`）
- 恢复会话时投影权限模式，避免 resume 后回落到配置默认（常为 yolo）
- 活动回合中非导航命令错误入队导致 UI 忙等卡死
- CJK IME 光标锚定与 composer 光标脉冲；shell 输出中的 VT 转义不再污染 TUI
- 用户回复卡片、running footer 扫描色与主题 token 对齐

### Changed

- `/resume` 使用 sessions sidecar 索引异步加载，避免全量扫描 transcript 阻塞 TUI
- Markdown 链接改为 Ctrl/Cmd-click 打开；精简多余 loading/toast
- 测试套件收敛，保留 fail-closed、恢复与数据安全相关契约

## [0.1.0] - 2026-08-05

首个公开版本。

### Added

- Rust 终端 Agent：Ratatui TUI 与行命令式 CLI/REPL，共用同一会话引擎
- 多 Provider 配置（API Key / Base URL / 协议），模型展示名、工具调用、并行工具请求、推理等级与文本详细度
- 专家 Agent 独立模型路由；`@` 委托探索、修复、设计、检索、综合等专家，并支持子会话浏览与回到父会话
- 权限模式 `safe` / `default` / `auto` / `yolo`：读写与命令按模式自动放行、询问或全放行；`default` / `auto` 支持会话内「始终允许」；`auto` 由 sticky reviewer 专家完成审批，并在子视图以请求/决策卡片呈现
- 内置工具：shell、文件系统读写、搜索、web fetch、git、代码 AST、工作流 todo / 自动续跑、记忆召回、skill 与 MCP 工具发现与调用
- 工具并行策略可配置；shell 输出、diff、todo、权限与子代理结果以结构化卡片展示
- 会话以追加写入的 JSONL transcript 持久化，支持恢复、历史树浏览，以及 TUI 内 undo / redo 与上下文压缩
- 运行时配置热重载；可选 Langfuse / OpenTelemetry 追踪
- TUI 主题、工具输出展开、滚动条与 `/` 本地命令补全

[Unreleased]: https://github.com/letr007/letcode/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/letr007/letcode/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/letr007/letcode/releases/tag/v0.2.0
[0.1.0]: https://github.com/letr007/letcode/releases/tag/v0.1.0
