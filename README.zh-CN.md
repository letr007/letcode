<h1 align="center">
  LetCode
</h1>

<p align="center">
  letcode 是一个由 Rust 编写的终端 Agent。
</p>

<p align="center">
  <a href="https://github.com/letr007/letcode/actions/workflows/test.yml"><img src="https://img.shields.io/github/actions/workflow/status/letr007/letcode/test.yml?branch=main&style=flat-square" alt="Test"></a>
  <a href="CHANGELOG.md"><img src="https://img.shields.io/badge/changelog-0.3.0-informational?style=flat-square" alt="Changelog"></a>
  <a href="LICENSE-MIT"><img src="https://img.shields.io/badge/license-MIT%20%7C%20Apache--2.0-blue?style=flat-square" alt="MIT License | Apache-2.0 License"></a>
</p>

<p align="center">
  中文 | <a href="README.md">English</a>
</p>

![letcode TUI](docs/letcode.png)

提供基于 Ratatui 的仿 `opencode` 风格 TUI，也保留了 REPL CLI 模式。

[技术文档](docs/index.md)

## 构建和运行

```sh
cargo build
cargo test
cargo fmt --check
```

运行默认 TUI：

```sh
cargo run
```

运行行命令式 CLI：

```sh
cargo run -- --cli
```

CLI 模式也可以通过 `cli` 或 `repl` 选择。TUI 可以通过 `--tui` 或 `tui` 显式选择。

查看已安装版本并检查 GitHub Release 中的新版本：

```sh
letcode --version
letcode update check
```

确认后更新通过 Release 安装的二进制：

```sh
letcode update
```

TUI 支持 English（`en`）和简体中文（`zh-CN`）。运行时可使用 `/language` 或别名 `/lang` 切换语言。

## 外部依赖

从源码构建需要 Rust 工具链。部分内置工具还会调用以下外部程序，请确保它们位于 `PATH` 中：

| 程序 | 使用位置 | 是否必需 |
| --- | --- | --- |
| [`git`](https://git-scm.com/) | `git__status`、`git__diff`、`git__log`，以及 TUI 分支状态 | 推荐安装；缺少时仅相关 Git 能力不可用 |
| [`rg`](https://github.com/BurntSushi/ripgrep) | `search__rg` 文本搜索 | 推荐安装；缺少时该工具不可用 |
| [`ast-grep`](https://ast-grep.github.io/) | `code__ast_search`、`code__ast_replace_preview` | 可选；缺少时仅 AST 工具不可用 |

此外，`shell__exec` 和本地 MCP 依赖实际调用的系统命令；`web__fetch` 与远程 MCP 需要可用的网络连接。

## 配置

`letcode` 从以下路径加载配置：

```text
~/.config/letcode/letcode.toml
```

配置示例：

```toml
# 可选；省略时使用配置中最先出现的 provider。
active_provider = "openai"
# 可选；默认 false。
fast_mode = false

# 可选；以下均有默认值。
[global]
# max_iterations = 64
# max_tool_calls = 128
# tool_timeout_secs = 60
sessions_dir = "sessions"
log_file = "logs/combined.log"

# 可选；省略时根据当前模型输入预算保留最近上下文。
[global.compaction]
# preserve_recent_tokens = 12000

# 可选；以下为默认值。
[global.retry]
enabled = true
max_attempts = 50
max_recovery_attempts = 3
initial_delay_secs = 1
backoff_multiplier = 2.0
jitter_secs = 1

# 可选；默认 default。可选值：safe | default | auto | yolo。
[permissions]
mode = "default" # solo 是 yolo 的兼容别名

# 可选；为内置专家指定默认路由或单次委派可选路由。
# [agents.explorer]
# provider = "openai"
# model = "gpt-5.5"
# allowed_models = ["openai/gpt-5.5"]
# 同样适用于 fixer、oracle、designer、librarian、general、reviewer。

# 可选；只能收窄工具自身声明的并行能力。
[tools.parallelism]
# "fs__read" = "parallel"
# "web__fetch" = "exclusive"

# 可选；本地 MCP 服务。
# [mcp.example_local]
# type = "local"
# command = ["/path/to/mcp-server", "--stdio"]
# environment = { FOO = "bar" }
# enabled = true
# timeout = 5000

# 可选；远程 MCP 服务，OAuth 暂不支持。
# [mcp.example_remote]
# type = "remote"
# url = "https://example.com/mcp"
# headers = { Authorization = "Bearer ..." }
# enabled = true
# timeout = 10000

# 必需：至少配置一个 provider，且其中至少配置一个 model。
[providers.openai]
# 可选；也可使用 OPENAI_API_KEY 环境变量。
api_key = "YOUR_API_KEY"
# OpenAI provider 可省略，默认 https://api.openai.com/v1。
base_url = "https://api.openai.com/v1"
# OpenAI provider 可省略，默认 responses；其他 provider 必须配置。
protocol = "responses" # responses | completions
# 可选；省略时使用该 provider 下最先出现的 model。
default_model = "gpt-5.5"

# 必需：每个 provider 至少配置一个 model；model 内字段均可省略。
[providers.openai.models."gpt-5.5"]
display_name = "GPT-5.5"
# protocol = "completions" # 可覆盖 provider 协议
# context_window = 400000
# effective_input_limit_tokens = 256000
# max_output_tokens = 128000
supports_tools = true # 默认 true
parallel_tool_calls = true # 默认 true
supports_reasoning = true # 默认 true
reasoning_effort = "medium"
# 可选；限制可选推理等级及 TUI 循环顺序。
reasoning_efforts = ["none", "low", "medium", "high", "max"]
reasoning_summary = "auto" # auto | concise | detailed
text_verbosity = "medium" # low | medium | high
# temperature = 0.2
# top_p = 1.0

# 可选；模型级 prompt cache。
# [providers.openai.models."gpt-5.5".prompt_cache]
# enabled = true
# retention = "in_memory" # in_memory | 24h
# namespace = "openai"
```

Provider API key 和 base URL 也可以来自按 provider 名称生成的环境变量，例如 `OPENAI_API_KEY` / `OPENAI_BASE_URL`；若 provider 名为 `compat`，对应为 `COMPAT_API_KEY` / `COMPAT_BASE_URL`。

相对路径形式的 `sessions_dir` 和 `log_file` 会按配置文件所在目录解析。

可选的 Langfuse/OpenTelemetry tracing 默认关闭。可用 `LETCODE_LANGFUSE_ENABLED=true`，并配置 `LANGFUSE_PUBLIC_KEY`、`LANGFUSE_SECRET_KEY` 与可选的 `LANGFUSE_HOST`（或写在本地 `.env`）启用。缺少凭据时 tracing 保持关闭，不影响 Agent 运行。

## 更新日志

发布说明见 [CHANGELOG.md](CHANGELOG.md)。

## 开源协议

本项目采用 MIT License 或 Apache License 2.0 双协议授权。
使用、修改或再分发本项目时，你可以任选其中一种协议。

- MIT License：见 [LICENSE-MIT](LICENSE-MIT)
- Apache License 2.0：见 [LICENSE-APACHE](LICENSE-APACHE)
