# 配置参考手册

letcode 从用户主目录下的 TOML 文件读取系统配置：

```text
~/.config/letcode/letcode.toml
```

配置文件支持使用环境变量，也可以使用命令进行脱机格式验证：

```sh
letcode config validate
```

## 顶层配置

| 配置项 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `active_provider` | 字符串 | 配置文件中首个服务商 | 指定当前会话默认使用的模型服务商 |
| `fast_mode` | 布尔值 | `false` | 是否开启快速模式，关闭非必要的重试与思考等待 |

## 全局配置 `[global]`

控制运行环境、存储路径与执行边界：

```toml
[global]
sessions_dir = "sessions"
log_file = "logs/combined.log"
tool_timeout_secs = 60
# max_iterations = 64
# max_tool_calls = 128
```

- `sessions_dir`：会话日志存储目录。相对路径按配置文件所在目录解析；
- `log_file`：运行日志输出文件。相对路径按配置文件所在目录解析；
- `tool_timeout_secs`：普通工具的默认超时秒数（系统命令 `shell__exec` 内部默认为 300 秒）；
- `max_iterations`：单个交互回合的最大迭代次数。省略时不设上限；
- `max_tool_calls`：单个交互回合的最大工具调用总数。省略时不设上限。

### 上下文保留 `[global.compaction]`

```toml
[global.compaction]
preserve_recent_tokens = 12000
```

- `preserve_recent_tokens`：执行上下文压缩时，在对话历史尾部强制保留的最近 Token 预算。省略时根据当前模型的有效输入上限动态计算。

### 物理重试策略 `[global.retry]`

```toml
[global.retry]
enabled = true
max_attempts = 50
max_recovery_attempts = 3
initial_delay_secs = 1
backoff_multiplier = 2.0
jitter_secs = 1
```

- `enabled`：是否开启物理网络请求重试；
- `max_attempts`：未产生可观察副作用时的最大网络重试次数；
- `max_recovery_attempts`：产生部分输出后，受控恢复迭代的最大重试次数；
- `initial_delay_secs`：首次重试前的初始等待时间（秒）；
- `backoff_multiplier`：指数退避乘数；
- `jitter_secs`：退避时间的随机抖动范围（秒）。

### 会话自动归档 `[global.session_archive]`

```toml
[global.session_archive]
enabled = true
older_than_days = 7
```

- `enabled`：是否在后台压缩长期未活动的旧会话；
- `older_than_days`：判定会话闲置的天数阈值。归档会话仍可正常列出，在执行恢复（Resume）时自动解压并加载。

## 权限控制 `[permissions]`

设置系统执行外部操作时的默认授权级别：

```toml
[permissions]
mode = "default" # safe | default | auto | yolo
```

- `mode`：运行模式，可选 `safe`（全提问）、`default`（读放行写提问）、`auto`（自动审查判定）或 `yolo`（全放行，`solo` 为兼容别名）。

## 工具并发控制 `[tools.parallelism]`

限制特定工具的并发执行策略：

```toml
[tools.parallelism]
"fs__read" = "parallel"
"web__fetch" = "exclusive"
```

- 可选值包括 `parallel`（允许在同批次中重叠并发执行）和 `exclusive`（必须排他单线程执行）。该配置只能收窄工具自身声明的能力，不能强制并发未支持的工具。

## MCP 外部服务配置 `[mcp.<name>]`

接入基于标准 Model Context Protocol（MCP）协议的外部工具服务：

### 本地子进程服务

```toml
[mcp.local_server]
type = "local"
command = ["/path/to/server", "--stdio"]
environment = { ENV_VAR = "value" }
enabled = true
timeout = 5000
```

- `command`：可执行程序路径及参数列表；
- `environment`：传递给子进程的键值对环境变量；
- `timeout`：服务响应与方法调用的超时毫秒数，默认 5000 毫秒。

### 远程网络服务

```toml
[mcp.remote_server]
type = "remote"
url = "https://example.com/mcp"
headers = { Authorization = "Bearer YOUR_TOKEN" }
enabled = true
timeout = 10000
```

- `url`：支持 Server-Sent Events 或 POST 的远程服务端点；
- `headers`：发起 HTTP 请求时附带的自定义标头。

## 服务商配置 `[providers.<name>]`

声明大语言模型服务商与认证参数：

```toml
[providers.openai]
protocol = "responses" # responses | completions | anthropic
flavor = "standard"    # standard | deepseek
default_model = "gpt-5.5"

[providers.openai.auth]
type = "bearer" # bearer | header | query | none
credential_env = "OPENAI_API_KEY"

[providers.openai.endpoints]
base_url = "https://api.openai.com/v1"

[providers.openai.endpoints.responses]
path = "responses"
```

- `protocol`：服务商协议类型，支持 `responses`、`completions` 与 `anthropic`；
- `flavor`：协议特征模式，支持 `standard` 或针对 DeepSeek 调整的 `deepseek`；
- `default_model`：该服务商下默认采用的模型标识；
- `auth.type`：认证方式，支持 Bearer Token、自定义请求头、URL 查询参数或无凭据；
- `auth.credential_env`：存储 API 密钥的环境变量名；也可直接用 `credential = "..."` 显式声明；
- `endpoints.base_url`：服务商 API 基础网关地址；
- `endpoints.*.path`：特定协议端点的相对路由路径。

## 模型配置 `[providers.<name>.models."<model>"]`

在指定服务商下定义模型的生成参数与调用能力：

```toml
[providers.openai.models."gpt-5.5"]
display = "GPT-5.5"
context_window = 400000
effective_input_limit_tokens = 256000

[providers.openai.models."gpt-5.5".transport]
websocket = false

[providers.openai.models."gpt-5.5".capabilities]
tools = true
parallel_tool_calls = true
reasoning = true
input_images = false

[providers.openai.models."gpt-5.5".generation]
temperature = 0.2
top_p = 1.0
max_output_tokens = 128000
reasoning_effort = "medium"
reasoning_efforts = ["none", "low", "medium", "high", "max"]

[providers.openai.models."gpt-5.5".cache]
enabled = false
retention = "in_memory" # in_memory | 24h
namespace = "openai"

[providers.openai.models."gpt-5.5".protocol_settings]
# 针对 Anthropic 模型的扩展参数
# anthropic_thinking = { mode = "adaptive" }
# anthropic_betas = ["context-1m-2025-08-07"]
```

- `display`：在 TUI 界面中展示的人类友好名称；
- `context_window`：总上下文窗口尺寸；
- `effective_input_limit_tokens`：有效输入上限 Token 预算；
- `transport.websocket`：对于 Responses 协议，是否启用全双工 WebSocket 流式传输；
- `capabilities`：模型能力开关，包括工具调用、并行工具调用、推理过程与图片支持；
- `generation`：生成参数，包括采样温度、Top-P、最大输出 Token 与思考深度档位；
- `cache`：服务商原生 Prompt 缓存开关、生命周期策略与命名空间；
- `protocol_settings`：特定协议专属选项，如 Anthropic 的自适应思考与 Beta 请求头。

## 专家路由配置 `[agents.<expert>]`

为特定内置专家指定独立的服务商与模型：

```toml
[agents.explorer]
provider = "openai"
model = "gpt-5.5"
allowed_models = ["openai/gpt-5.5"]

[agents.historian]
provider = "openai"
model = "gpt-5.5"
```

支持配置以下 8 个内置专家：

1. `explorer`：代码库只读检索与探索专家；
2. `fixer`：代码修改与故障修复专家；
3. `oracle`：根因分析与风险审查专家；
4. `designer`：系统方案与交互设计专家；
5. `librarian`：技术资料与上下文归档专家；
6. `general`：通用辅助专家；
7. `reviewer`：权限安全审查内部专家；
8. `historian`：会话历史整理与上下文压缩内部专家。

`allowed_models` 限制该专家在单次委派时允许动态覆盖的模型白名单。

## 仿真伪装配置 `[fake]`

配置请求元数据伪装参数，模拟兼容环境特征：

```toml
[fake.identity]
# installation_id = "..."
# agent_name = "Hypatia"

[fake.clock]
timezone = "Asia/Shanghai"
# date = "2026-09-12"

[fake.client]
version = "0.153.4"
originator = "Codex Desktop"

[fake.environment]
sandbox = "none"
sandbox_mode = "danger-full-access"
shell = "zsh"
```

未显式指定的字段由系统在运行时自动检测当前主机的真实环境值补充。详情参见[客户端特征仿真](fake.md)。

## 源码索引

- `src/config.rs`：实现所有配置项结构体、默认值逻辑、路径解析与 TOML 校验。
- `src/config/`：提供配置加载子模块与环境变量读取逻辑。
- `src/model_runtime/mod.rs`：根据配置组装运行时服务商与已解析路由。
- `src/session/engine/config_reload.rs`：监控配置文件修改并执行热重载。
