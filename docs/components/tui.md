# 终端用户界面

letcode 提供基于 Ratatui 的终端用户界面（TUI），支持流式对话、工具调用、权限审批和配置编辑。

在项目目录启动 TUI。

```sh
letcode
```

## 界面布局

- 时间线显示对话记录、模型回复、思考内容、工具调用卡片和子代理状态。助手文本随输出追加。
- 输入面板支持多行编辑、输入历史、命令补全和专家补全。面板显示当前模型、服务商、推理强度和权限模式。
- 底部状态栏显示执行状态。Token 用量、缓存命中率、输出速率和分支信息按可用宽度显示。
- 侧边栏显示会话信息、上下文用量、MCP 服务状态和任务清单。先按 `Ctrl+X`，再按 `b` 切换显示。

## 队列消息

回合运行期间提交的消息进入队列，界面标记为 `QUEUED`。
在主输入区按 `Ctrl+Backspace`，移除最新一条尚未派发的消息。
当前执行和输入草稿保持不变。

移除的消息保留在输入历史中，可按 `↑` 找回。
该快捷键需要终端能区分 `Ctrl+Backspace` 与普通退格。

## 富文本与图表排版

- Markdown 渲染支持标题、强调文本、列表、引用、表格和代码块。代码块支持语法高亮。
- 思考内容默认完整展开（`full`）。`/thoughts` 可切换为紧凑显示（`compact`）、仅标题（`titles`）或滚动窗口（`scroll`）。
- LaTeX 排版支持行内公式 `$...$` 和独立公式 `$$...$$`，使用 Unicode 字符显示上下标、根号、分数和矩阵。
- Mermaid 渲染解析图表代码块，在终端网格中绘制节点、连线和标签。

## 帮助阅读器

输入 `/help` 或 `/?` 打开内置帮助手册。
手册包含命令速查、快捷键、会话操作、专家委派和权限说明。
命令速查从现有命令定义生成。

宽屏终端同时显示目录和正文，窄屏终端按 `Tab` 切换。
按 `↑` / `↓` 或 `k` / `j` 选择章节或滚动正文。
按 `PgUp` / `PgDn` 翻页，按 `Home` / `End` 跳到首尾。
按 `Esc` 或 `q` 关闭帮助，输入草稿保持不变。

阅读帮助时，后台任务继续运行。
新的提问或审批请求到达后，帮助面板关闭，界面显示待处理请求。

## 斜杠命令

在输入框输入 `/` 打开命令补全列表。
常用命令见下表，完整用法见 `/help` 的“命令速查”。

| 命令 | 别名 | 用途 |
| --- | --- | --- |
| `/help` | `/?` | 打开帮助手册 |
| `/model` | — | 查看或切换当前模型 |
| `/permission` | `/perm` | 查看或切换权限模式 |
| `/agents` | — | 配置专家模型 |
| `/reasoning` | `/think` | 查看或切换推理强度 |
| `/thoughts` | — | 调整思考内容的显示方式 |
| `/tools` | `/tool-output` | 调整工具输出的显示方式 |
| `/compact` | — | 整理会话历史并压缩上下文 |
| `/language` | `/lang` | 查看或切换界面语言 |
| `/theme` | — | 查看或切换配色主题 |
| `/config` | — | 查看、编辑和保存配置 |

## 界面语言与主题

TUI 内置英文和简体中文。
输入 `/language en` 或 `/language zh-CN` 切换语言。
语言设置保存后，下次启动继续使用。

内置主题包括 `dark`、`plain`、`glass`、`wireframe` 和 `rainbow`。
`/theme` 打开主题选择器。
外部主题使用 `themes/*.toml` 配置前景色、背景色和强调色。

## 源码索引

- `src/tui/terminal.rs` 管理终端初始化、原始输入模式和终端恢复。
- `src/tui/runtime.rs` 处理事件、输入动作、弹窗和视图切换。
- `src/tui/timeline.rs` 管理时间线条目和事件投影。
- `src/tui/components/transcript.rs` 负责时间线排版、卡片渲染和虚拟滚动。
- `src/tui/components/composer.rs` 渲染输入面板和模型、权限信息。
- `src/tui/components/footer.rs` 渲染执行状态、Token 用量和速率信息。
- `src/tui/components/sidebar.rs` 渲染上下文用量、MCP 状态和任务清单。
- `src/tui/help.rs` 管理手册章节和阅读状态。
- `src/tui/components/help_reader.rs` 渲染帮助面板、目录和正文。
- `src/tui/input.rs` 映射键盘、鼠标和粘贴事件。
- `src/command.rs` 定义命令元数据和参数解析。
- `src/tui/slash.rs` 匹配命令与专家补全条目。
- `src/tui/markdown.rs`、`src/tui/math.rs` 和 `src/tui/mermaid/` 渲染 Markdown、公式和图表。
- `src/tui/i18n.rs` 提供界面翻译。
- `src/tui/theme.rs` 和 `src/tui/theme_file.rs` 管理内置与外部主题。
