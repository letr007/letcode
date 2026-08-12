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

## 配置

`letcode` 从以下路径加载配置：

```text
~/.config/letcode/letcode.toml
```

最小示例：

```toml
active_provider = "openai"

[global]
# 可选的运行时限制：
# max_iterations = 64
# max_tool_calls = 128
sessions_dir = "sessions"
log_file = "logs/combined.log"

[permissions]
mode = "default" # safe/default/auto/yolo；读取时兼容旧的 solo
# auto = Ask 集合与 default 相同，但由 sticky reviewer 专家回答审批

# 可选：permission mode = "auto" 时的 reviewer 模型路由
# [agents.reviewer]
# provider = "openai"
# model = "gpt-5.5"
# 可选：单次委派时允许选择的 provider-qualified 路由
# allowed_models = ["openai/gpt-5.5"]

# 可选的本地执行策略。经过审查的读取工具可以声明支持并行；
# 其他工具保持单例执行，除非其处理器明确选择并行。
[tools.parallelism]
# "fs__read" = "parallel"
# "web__fetch" = "exclusive"

[providers.openai]
api_key = "YOUR_API_KEY"
base_url = "https://api.openai.com/v1"
protocol = "responses" # responses/completions
default_model = "gpt-5.5"

[providers.openai.models."gpt-5.5"]
display_name = "GPT-5.5"
# context_window = 400000
# effective_input_limit_tokens = 256000 # 可选：当前 provider/model 路径输入预算
supports_tools = true
parallel_tool_calls = false # 允许模型在一次响应中请求多个工具
supports_reasoning = true
reasoning_effort = "medium" # 该模型的默认值
# 可选：限制可选思考等级，并控制 TUI 循环切换顺序。
# 支持：none、minimal、low、medium、high、xhigh、max
reasoning_efforts = ["none", "low", "medium", "high", "max"]
reasoning_summary = "auto"
text_verbosity = "medium"
```

Provider API key 和 base URL 也可以来自按 provider 名称生成的环境变量，例如 `OPENAI_API_KEY` / `OPENAI_BASE_URL`；若 provider 名为 `compat`，对应为 `COMPAT_API_KEY` / `COMPAT_BASE_URL`。

相对路径形式的 `sessions_dir` 和 `log_file` 会按配置文件所在目录解析。

可选的 Langfuse/OpenTelemetry tracing 默认关闭。可用 `LETCODE_LANGFUSE_ENABLED=true`，并配置 `LANGFUSE_PUBLIC_KEY`、`LANGFUSE_SECRET_KEY` 与可选的 `LANGFUSE_HOST`（或写在本地 `.env`）启用。缺少凭据时 tracing 保持关闭，不影响 Agent 运行。

## 会话

会话 transcript 以 append-only JSONL 保存在 `sessions_dir` 下，之后可以恢复。在 TUI 中可用 `/tree` 浏览历史，用 `/undo` / `/redo` 在已完成用户回合间移动，完整本地命令见 `/help`。行命令式 CLI 支持只读的 `/tree`；`/undo` 与 `/redo` 需要 TUI。

## 更新日志

发布说明见 [CHANGELOG.md](CHANGELOG.md)。

## 开源协议

本项目采用 MIT License 或 Apache License 2.0 双协议授权。
使用、修改或再分发本项目时，你可以任选其中一种协议。

- MIT License：见 [LICENSE-MIT](LICENSE-MIT)
- Apache License 2.0：见 [LICENSE-APACHE](LICENSE-APACHE)
