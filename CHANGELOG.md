# Changelog

本文件记录项目的重要变更。

格式参考 [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)，
版本号遵循 [Semantic Versioning](https://semver.org/)。

## [Unreleased]

## [0.6.0] - 2026-08-21

### Added

- TUI 启动后在后台检查稳定版更新；发现新版本时通过现有 info toast 提示运行 `letcode update`，网络失败不影响启动。
- TUI 新增可选会话面板，可通过 `/panel` 或 `Ctrl-X B` 切换，展示会话、模型、权限、上下文和完整工作流 TODO 状态；宽屏使用分栏，窄屏使用覆盖布局。
- 长文本粘贴在 composer 中折叠为行数 token，提交、排队和历史恢复仍保留完整原文。
- `Ctrl-X` 前缀新增模型、权限、推理、思考、专家、上下文、MCP、skill 和帮助等常用本地命令快捷键。

### Changed

- `Ctrl-X` 前缀等待时间延长到约 3 秒，同时保留原有面板切换和子会话方向键导航。
- 流式 TUI 更新按帧限制处理量和耗时，降低高频事件下的输入与渲染阻塞。
- 自更新命令改用结构化终端流程，精确匹配平台资产，并在替换二进制前校验 GitHub Release 提供的 SHA-256 digest。
- 工作流 TODO 与自动续跑状态并入 `RuntimeSnapshot` 单一运行时权威，由 branch/leaf 对应的 transcript 事件投影恢复；移除从已压缩工具历史反推 TODO 的双状态同步路径。

### Fixed

- 子代理运行状态使用当前 TUI 语言渲染，并在 `/lang` 切换后刷新 transcript 缓存，不再固定显示系统语言。
- 上下文再次压缩时完整保留已有执行检查点，不再固定截断为前 8,000 字符，避免跨多次压缩丢失后部的下一步与关键上下文。
- 修复工作流工具调用被压缩退休后，后续压缩可能清空结构化 TODO，并错误重置当前回合验证计数的问题。

## [0.5.2] - 2026-08-20

### Fixed

- 修复一次 `edit__apply_patch` 修改多个文件时，TUI 将所有变更连续渲染在同一张 diff 卡片中的问题；现在每个文件独立显示，同一文件的多项修改保持原始顺序。

## [0.5.1] - 2026-08-20

### Added

- 新增 `--version`、`update check` 和交互式 `update` 命令，可从 GitHub Releases 检查并安装新版本；发布产物同步生成 SHA-256 校验清单。

## [0.5.0] - 2026-08-20

### Added

- 新增内置工程工作流 skills：`git`、`simplify`、`verification-planning` 和 `worktrees`，覆盖仓库操作、行为保持的代码简化、验证路径设计与隔离工作树管理。
- 新增 DeepSeek V4 请求策略，兼容 reasoning 内容、工具调用回放和流式恢复状态。
- 新增可选的 Anchored Bootstrap 实验：对配置白名单内的模型先以最小工具和提示启动，再按持久化信号恢复完整工具目录；可通过 `/anchored` 在会话内切换，并在 composer 显示启用状态。
- TUI 支持 `en` / `zh-CN` 国际化，可通过 `/language`（`/lang`）切换；未显式选择时跟随系统 locale，显式语言保存到 `tui-preferences.json`。
- 大体积 `shell__exec` 输出会写入可信临时 artifact，仅内联有界预览和 `local_path`；`fs__read` / `search__rg` 可按需读取完整内容。
- 大体积 `search__rg` 结果会折叠为包含真实文件数、匹配数、预览和 `local_path` 的摘要，避免向模型上下文注入完整结果集。
- `/agents` 专家模型二级菜单支持多选：使用 `Space` 选中或取消模型，`Enter` 保存到 `agents.<expert>.allowed_models`，`Esc` 放弃修改并返回专家列表。

### Changed

- Agent 会话内存状态统一以 `RuntimeSnapshot` 为单一事实来源，移除 history、protocol frames 等镜像状态及请求前重复投影，恢复、压缩和协议流均读取同一快照。
- 会话、transcript、subagent、tool、配置持久化和 TUI 渲染代码拆分为职责更清晰的模块，收紧模块边界并保留原有 API 兼容入口。
- Assistant 流式输出改为平滑的 typewriter 投影，并同步渲染几何信息，减少文本跳动并确保流式期间仍可滚动到 transcript 底部。
- TUI 输入与绘制职责解耦后恢复同步终端绘制路径，避免渲染任务并发修改终端状态；Git 分支轮询、模型目录和运行时辅助逻辑改为独立组件。
- 工程任务默认提示优先使用原生 LaTeX 与 Mermaid 渲染，并合并统一的工程工作流约束。
- 上下文压缩改用有界工具输出前缀、复用 transcript 分析结果和有界协议指纹，避免超大上下文下的全量序列化停顿。
- Mermaid 代码块仅在匹配的 fenced code 围栏闭合后渲染，避免流式输出期间反复解析和图表高度抖动；成功结果按源码和宽度使用有界缓存。
- Mermaid flowchart 扩展支持嵌套 `subgraph`、子图内方向声明、循环关系和带连字符的节点 ID；无法使用二维布局的循环图自动使用线性布局。
- Footer 缓存命中率提高到小数点后两位。
- `question` 工具等待用户回答时，终端标题使用 `?` 状态标记并保留当前会话标题。

### Fixed

- 修复 DeepSeek V4 流式响应中 reasoning 内容、工具回放和恢复状态丢失，避免恢复后重复执行或协议状态不一致。
- 修复 Assistant 流式渲染不平滑、卡片宽度变化和 transcript 底部不可达的问题。
- 修复补丁、文件写入 diff、搜索摘要和工具 trace 中 ANSI/控制字符破坏 TUI 布局的问题；磁盘原始内容保持不变。
- Windows 不再启用不受支持的 kitty keyboard enhancement；TUI 仅处理 key press 事件，避免按键重复输入。
- TUI 的权限、帮助、picker、composer 等界面文案完整接入国际化，语言切换后保持一致。
- 专家可选模型变更会同步当前运行时和已打开的选择器；配置热重载后不再出现勾选状态过期。
- 修改 reviewer 可选模型后会清理 sticky reviewer 会话，确保后续审批使用最新模型策略。
- Mermaid 围栏闭合检测绑定 Markdown parser 的源码范围，避免引用、嵌套列表和缩进代码造成错误的提前渲染。

### Security

- letcode 自身生成的折叠 artifact 目录仅获得受限的只读信任；其他外部路径仍遵循原有权限审批规则。

## [0.4.0] - 2026-08-13

### Added

- `shell__exec` 支持按调用设置 1–3600 秒超时，未指定时继续使用 300 秒默认值

### Fixed

- 父会话视图展示后台子代理的重试原因、等待时间与尝试次数
- Footer 的 token 使用量和压缩状态跟随当前父/子会话视图，切回父视图时保留已知用量

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

[Unreleased]: https://github.com/letr007/letcode/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/letr007/letcode/compare/v0.5.2...v0.6.0
[0.5.2]: https://github.com/letr007/letcode/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/letr007/letcode/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/letr007/letcode/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/letr007/letcode/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/letr007/letcode/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/letr007/letcode/releases/tag/v0.2.0
[0.1.0]: https://github.com/letr007/letcode/releases/tag/v0.1.0
