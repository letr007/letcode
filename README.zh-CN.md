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
# 可选：限制该模型可选的思考等级，并控制 TUI 中循环切换的顺序。
# 支持：none、minimal、low、medium、high、xhigh、max
reasoning_efforts = ["none", "low", "medium", "high", "max"]
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

`parallel_tool_calls` 是 OpenAI 的请求级开关。设为 true 后，模型一次响应可以返回多个工具调用。本地执行仍服从每个工具声明的策略：支持并行的读取工具可以重叠执行，写入、命令、工作流控制、question 和 MCP 工具保持单例。默认值为 false。`[tools.parallelism]` 可以把支持并行的工具收紧为 `exclusive`，或显式保留为 `parallel`；不安全工具不能提升为并行。

`reasoning_effort` 设置模型启动时的默认思考等级。`reasoning_efforts` 可选地限制该模型在 `/reasoning`、`/think`、Ctrl+T 和 TUI 选择器中能切换的等级；数组顺序也是循环切换顺序。省略时保持兼容，默认可选 `none`、`minimal`、`low`、`medium`、`high`、`xhigh`。对于支持该值的兼容 provider，`max` 会通过原始请求序列化发送。设置了 `reasoning_efforts` 时，`reasoning_effort` 必须包含在该列表中。

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

## 会话上下文和恢复

会话 transcript 以 append-only JSONL 记录保存在 `sessions_dir` 下。恢复会从这些记录重建对话历史、context branch、context tree 和 prompt context view。archive 或 remove-from-view 等 context view 操作只追加元数据，不会清除原始 transcript 事件。

context tree 是严格树结构，用于记录会话/任务上下文节点的 active 和 archived 状态，服务于 prompt 组装和 TUI 展示；它不表示文件系统回滚。hard constraint、current user requirement、未解决错误（包括 invariant violation）、权限决策、文件写入事实、验证/测试结果和 commit hash 在恢复后仍属于 protected context。

context view 是派生的 prompt 投影。它会把稳定 hard context 放在动态细节之前，支持 pinned block、summary 和 opened detail，并把 archived 或 removed 的软上下文从 prompt 可见区域隐藏。大型 shell 输出默认折叠为可打开的元数据；folded output 不是语义摘要。

在 TUI 中可用 `/context` 浏览 context node、block、summary 和 folded output。旧的 `context__checkpoint` / `context__return` 记录仍兼容新的 context tree 元数据。

## 项目结构

```text
src/main.rs          程序入口、配置加载、TUI/CLI 选择
src/config.rs        TOML 配置解析和校验
src/agent.rs         模型循环、工具执行、turn 生命周期
src/context_tree.rs  会话 context tree 回放和 invariant
src/context_view.rs  派生 prompt context block、summary 和 folded output
src/tool.rs          内置工具注册表和工具结果模型
src/permission.rs    权限模式、scope 和请求分类
src/transcript.rs    JSONL transcript 持久化和恢复辅助函数
src/request_builder.rs prompt 组装和 context view 注入
src/subagent.rs      subagent相关
src/mcp.rs           MCP 工具发现
src/tui/             Ratatui/Crossterm UI、runtime、state、events、rendering
```

## 开源协议

本项目采用 MIT License 或 Apache License 2.0 双协议授权。
使用、修改或再分发本项目时，你可以任选其中一种协议。

- MIT License：见 [LICENSE-MIT](LICENSE-MIT)
- Apache License 2.0：见 [LICENSE-APACHE](LICENSE-APACHE)
