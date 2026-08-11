---
name: customize-letcode
description: >
  仅在用户编辑或创建 letcode 自身配置时使用：
  letcode.toml、~/.config/letcode/ 下文件、项目 .letcode/、
  letcode 会话用的 AGENTS.md、letcode skills、MCP 服务、
  provider/model 路由、权限模式或 TUI 主题。
  不要用于用户的应用代码，也不要用于并非在配置 letcode 本身的项目。
---

<!--
  内置 skill 正文，在磁盘发现之前注册。
-->

# 定制 letcode

letcode 会校验 `letcode.toml`：字段写错，或缺少必需的
provider/model 时会拒绝启动。下面这些是合法配置面。

## 如何生效

运行时会监视 `letcode.toml`。热重载成功时，只应用**支持热更新的字段**
（providers/models/routes、retry、compaction、tool timeout、tool parallelism
覆盖）。MCP、权限模式、Fast Mode，以及 `max_iterations` / `max_tool_calls`
仍需重启。

坏配置不会弄崩正在跑的会话（重载会保留内存里的旧配置），但磁盘上的坏文件会挡住下一次冷启动。

每次改完 `letcode.toml` 后，调用 `config__validate`（`path` 为 `null` = 默认
`~/.config/letcode/letcode.toml`）。若 `valid` 为 false，按返回的 `error` 修，
再验，直到通过。改动尽量小。

会话外：`letcode config validate [path]`（无效时 exit 1）。

## 文件放哪

| 范围 | 路径 |
| --- | --- |
| 全局配置 | `~/.config/letcode/letcode.toml`（必需；缺失是硬错误） |
| 全局 skills | `~/.config/letcode/skills/<name>/SKILL.md` |
| 全局 AGENTS.md | `~/.config/letcode/AGENTS.md`（用户级指令，若存在） |
| 全局主题 | `~/.config/letcode/themes/<id>.toml` |
| 会话 / 日志 | 在配置目录下，由 `global.sessions_dir` / `global.log_file` 决定（默认 `sessions`、`logs/combined.log`） |
| 项目 skills | `.letcode/skills/<name>/SKILL.md`（从 git 根到 cwd） |
| 项目 AGENTS.md | 从仓库根到 cwd 的 `AGENTS.md`（后面的追加并优先） |

新建 skill 放在上面的 letcode skill 目录里。同名 skill：后发现的根覆盖先发现的。

## letcode.toml

未知顶层键会被拒绝。至少需要一个带至少一个 model 的 `[providers.<name>]`。
`active_provider` 默认取第一个 provider 键。

```toml
active_provider = "openai"
fast_mode = false

[global]
# max_iterations = 64
# max_tool_calls = 128
# tool_timeout_secs = 60
sessions_dir = "sessions"
log_file = "logs/combined.log"

[global.compaction]
# preserve_recent_tokens = 12000

[global.retry]
# enabled = true
# max_attempts = 50
# max_recovery_attempts = 3
# initial_delay_secs = 1
# backoff_multiplier = 2.0
# jitter_secs = 1

[permissions]
mode = "default" # safe | default | auto | yolo（solo 是 yolo 的别名）

# 可选 expert 路由。省略 provider 则跟随 active_provider。
# [agents.explorer]
# provider = "openai"
# model = "gpt-5.5"
# allowed_models = ["openai/gpt-5.5"] # 单次委派可选路由；不改变默认模型
# 同样适用于：fixer, oracle, designer, librarian, general, reviewer

[tools.parallelism]
# "fs__read" = "parallel"   # 只能收窄已经声明 Parallel 的工具
# "web__fetch" = "exclusive"

[mcp.example_local]
type = "local"
command = ["/path/to/mcp-server", "--stdio"]
# environment = { FOO = "bar" }   # 别名：env
# enabled = true
# timeout = 5000                  # 毫秒

[mcp.example_remote]
type = "remote"
url = "https://example.com/mcp"
# headers = { Authorization = "Bearer ..." }
# enabled = true
# timeout = 10000
# oauth = false                   # true 会被拒绝（暂不支持 OAuth）

[providers.openai]
api_key = "YOUR_API_KEY"
base_url = "https://api.openai.com/v1"
protocol = "responses" # responses | completions（provider 名不是 openai 时必填）
default_model = "gpt-5.5"
# [providers.openai.retry]        # 可选：按 provider 覆盖 retry

[providers.openai.models."gpt-5.5"]
display_name = "GPT-5.5"          # 别名：name
# protocol = "completions"        # 可选：覆盖 provider 的 protocol
# context_window = 400000
# effective_input_limit_tokens = 256000
# max_output_tokens = 128000
supports_tools = true             # 省略默认为 true
parallel_tool_calls = false
supports_reasoning = true         # 省略默认为 true
reasoning_effort = "medium"       # none|minimal|low|medium|high|xhigh|max|自定义字符串
reasoning_efforts = ["none", "low", "medium", "high", "max"]
reasoning_summary = "auto"        # auto|concise|detailed
text_verbosity = "medium"         # low|medium|high
# temperature = 0.2
# top_p = 1.0

# [providers.openai.models."gpt-5.5".prompt_cache]
# enabled = true
# retention = "in_memory"         # in_memory | 24h
# namespace = "my-cache"
```

形状说明：

- 相对路径的 `sessions_dir` / `log_file` 相对**配置文件所在目录**解析。
- Provider 环境变量覆盖 TOML：`<PROVIDER>_API_KEY`、`<PROVIDER>_BASE_URL`
  （provider 名大写；非字母数字变 `_`）。例：`openai` →
  `OPENAI_API_KEY` / `OPENAI_BASE_URL`。
- TOML 里 `protocol` 用 kebab-case：`responses`、`completions`。
- `[tools.parallelism]` 只能**收窄**已声明 `Parallel` 的工具（例如强制
  `exclusive`）。把 exclusive 工具提成 `parallel` 会被拒绝。
- 本地 MCP 的 `command` 必须是字符串数组，不能是单个字符串。`type` 必填。
  remote 不能设 `command`/`environment`；local 不能设 `url`/`headers`/`oauth`。
- 内置 expert agent 键固定为：`explorer`、`fixer`、`oracle`、`designer`、
  `librarian`、`general`、`reviewer`。没有自由形式的 agent map。
- `agents.<expert>.allowed_models` 仅接受 `provider/model`，用于 `agent__*`
  单次委派选择；省略 `model` 时仍使用 expert 默认路由，单次选择不会写回配置。
- `permissions.mode = "auto"` 的 Ask 矩阵与 `default` 相同，但由粘性的
  `reviewer` expert 来回答审批。

## Skills

Skill 目录里正好有一个 `SKILL.md`（普通文件，不能是符号链接），带 YAML
frontmatter：

```markdown
---
name: my-skill
description: 一句话说明这个 skill 做什么，以及何时触发。把具体关键词/文件名往前放；需要时用 "Use ONLY when..." 收窄。
---

# My Skill

说明、示例、参考。
```

- `name`：必填，小写 kebab-case，1–64 字符，必须与文件夹名一致。
- `description`：必填；这是给模型看的发现信号。
- `SKILL.md` 旁可放可选资源文件，用 `skill__resource_list` /
  `skill__resource_read` 读取。
- 新建文件放 `~/.config/letcode/skills/` 或 `.letcode/skills/`，别发明配置键——
  letcode 没有 `skills.paths` / `skills.urls`。
- 磁盘上同名 `customize-letcode` 会覆盖这个内置 skill。

## AGENTS.md

letcode 把指令 markdown 载入 agent prompt：

1. 全局文件：`<config_dir>/AGENTS.md`（通常是 `~/.config/letcode/AGENTS.md`）。
2. 工作区链：从 git 根（没有 `.git` 则从 cwd）到当前目录，每个 `AGENTS.md`
   接在全局文件后面追加。越深的文件越靠后，冲突时优先。缺文件就跳过；重载是幂等的。

## 主题

自定义 TUI 主题：`~/.config/letcode/themes/<id>.toml`。

- 文件名 stem 即主题 `id`：小写 ASCII 字母/数字/`-`/`_`；`tokyo-night` /
  `tokyo_night` 会规范成 `tokyonight`。
- 保留 id（文件会被忽略，不能覆盖）：`dark`（别名 `default`）、`rainbow`。
- 若不存在会种子写入的捆绑主题：`ocean`、`forest`、`rose`、`tokyonight`
  （可改；不会覆盖已有文件）。
- TUI 用 `/theme` 选择；热重载会重新发现主题目录。

可接受字段（均为可选；颜色未写则回退 dark 调色板）。颜色值必须是
`#RRGGBB` 或 `#RGB` 字符串：

```toml
# 元数据（字符串）
label = "Sunset"                 # 选择器显示名；省略则用 id
description = "Warm accent"      # 选择器说明；省略则用路径提示

# 表面 / 文字
root_bg = "#121212"
surface_bg = "#18181a"
element_bg = "#1e1e20"
elevated_bg = "#262628"
border = "#323236"
text = "#dcdcdc"
muted_text = "#828282"
dim_text = "#505050"

# 语义色
accent = "#50b4dc"
assistant = "#64d282"
user = "#50b4dc"
success = "#64c864"
warning = "#b4b464"
error = "#dc5050"
approval = "#dcb43c"
notice = "#646464"

# Diff
diff_add_bg = "#162d20"
diff_delete_bg = "#36202a"
diff_hunk_bg = "#1f283c"
```

只写需要覆盖的键即可，例如：

```toml
label = "Sunset"
accent = "#f60"
user = "#ff8800"
```

## 提议修改时

- 保留用户没要求改的 providers、models、MCP servers、agent routes。
- 只用本 skill 里的键。未知顶层键，以及 `$schema`、`plugin`、自由形式
  `agent` map、`skills.urls`、`permission.bash` 模式映射这类形状会被拒绝。
- 长说明放进 letcode skills 目录下的 skill 文件；`letcode.toml` 没有 skill
  正文字段。
- 写完 `letcode.toml` 后跑 `config__validate`，修到 `valid` 为 true。仅 MCP /
  权限 / Fast Mode / max iteration·tool caps 需要重启。
