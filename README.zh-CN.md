# letcode

中文 | [English](README.md)

`letcode` 是一个由 Rust 编写的终端 Agent。默认提供基于 Ratatui 的仿`opencode`风格 TUI，也保留了REPL  CLI 模式。

## 构建和测试

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
mode = "default"

[providers.openai]
api_key = "YOUR_API_KEY"
base_url = "https://api.openai.com/v1"
protocol = "responses" # responses/completions
default_model = "gpt-5.5"

[providers.openai.models."gpt-5.5"]
display_name = "GPT-5.5"
supports_tools = true
supports_reasoning = true
reasoning_effort = "medium"
reasoning_summary = "auto"
text_verbosity = "medium"
```

Provider API key 和 base URL 也可以来自环境变量。变量名会根据 provider 名称生成：

```sh
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

如果 provider 名为 `compat`，对应变量为 `COMPAT_API_KEY` 和 `COMPAT_BASE_URL`。

相对路径形式的 `sessions_dir` 和 `log_file` 会按配置文件所在目录解析。

### 可选 Langfuse tracing

Langfuse/OpenTelemetry tracing 默认关闭，不会改变 Agent 行为。启用后，`letcode` 只导出 LLM turn、流式模型调用、tool call、状态、token 计数和延迟等安全运行元数据；不会导出原始 prompt、原始工具参数、原始工具输出、API key 或 `.env` 内容。

可以通过环境变量启用，也可以把相同变量放在本地 `.env` 文件中：

```sh
LETCODE_LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

如果缺少 Langfuse 凭据或 tracing 初始化失败，`letcode` 会继续运行，并自动禁用 Langfuse tracing。

## 项目结构

```text
src/main.rs          程序入口、配置加载、TUI/CLI 选择
src/config.rs        TOML 配置解析和校验
src/agent.rs         模型循环、工具执行、turn 生命周期
src/tool.rs          内置工具注册表和工具结果模型
src/permission.rs    权限模式、scope 和请求分类
src/transcript.rs    JSONL transcript 持久化和恢复辅助函数
src/subagent.rs      subagent相关
src/mcp.rs           MCP 工具发现
src/tui/             Ratatui/Crossterm UI、runtime、state、events、rendering
```

## 开源协议

本项目采用 MIT License 或 Apache License 2.0 双协议授权。
使用、修改或再分发本项目时，你可以任选其中一种协议。

- MIT License：见 [LICENSE-MIT](LICENSE-MIT)
- Apache License 2.0：见 [LICENSE-APACHE](LICENSE-APACHE)
