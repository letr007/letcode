# 配置参考手册

letcode 从用户主目录下的 TOML 文件读取系统配置：

```text
~/.config/letcode/letcode.toml
```

配置文件支持使用环境变量，也可以使用命令脱机验证格式：

```sh
letcode config validate
```

## 编辑器补全与校验

仓库根目录的 `letcode.schema.json` 使用 JSON Schema（draft-07）描述配置结构。在 `letcode.toml` 首行加入注释指令，编辑器即可识别并提供补全与校验：

```toml
#:schema https://raw.githubusercontent.com/letr007/letcode/main/letcode.schema.json
```

Taplo、VS Code 的 Even Better TOML 与 Zed 等编辑器均支持该指令，也可指向本地文件路径。letcode 解析配置时会自动忽略注释。首次启动生成的默认配置已自带该声明。

Schema 检查字段名称、类型、枚举值与必填项，并拒绝未声明的未知字段。

## 交互式配置编辑器（/config）

会话运行期间，在终端输入 `/config` 打开内置的交互式配置编辑器。

### 浏览与搜索

- 编辑器按表逐层显示字段。按 `Enter` 或 `→` 进入子表，按 `Esc` 返回上一级。
- 在字段浏览列表中输入文字，可搜索整个配置中的字段路径。按 `Backspace` 删减搜索词。
- 选中字段后显示当前值、操作提示和本地化说明。凭据字段（如 `credential`）在列表中脱敏显示。

### 字段编辑与列表操作

- 文本和数字字段按 `Enter` 进入编辑，修改后按 `Enter` 确认，按 `Esc` 取消。数字格式错误时保留输入供继续修改。
- 布尔字段按 `Enter` 切换值。枚举和推理强度字段按 `Enter` 展开选项，用 `↑` / `↓` 选择，再按 `Enter` 确认。
- 数组字段（如 `allowed_models`、`command`）按 `Enter` 展开，用 `a` 追加元素，用 `d` 或 `Delete` 删除选中元素。按 `←` 或 `Esc` 收起列表。
- `providers`、`models` 和 `mcp` 等表提供新增条目选项。输入名称后生成新条目的初始字段。

### 草稿保存与配置重载

- 字段修改保留在内存草稿中。关闭有未保存改动的编辑器时，可选择保存、丢弃或继续编辑。
- 未编辑字段时，按 `Ctrl+S` 保存。保存前校验整个配置，写回时保留 TOML 注释。
- 配置读取和保存共用文件锁，读取会等待正在进行的写入完成。
- Windows 文件替换失败时保留恢复文件，并在错误中列出临时文件和备份目录路径。替换成功后，备份清理失败只记录警告。
- 配置监听器检测到保存后，在引擎空闲时重载支持热更新的设置。当前路由已从配置删除时，继续保留该会话路由并发出提示。

### 启动异常修复

当 `letcode.toml` 存在语法错误或非法字段时，TUI 模式在启动阶段进入原地修复流程，输出错误原因并提示按 `Enter` 调用环境变量 `$EDITOR` 打开文件，修复保存后重新载入；输入 `q` 则安全退出。非交互模式（如 `--cli`、`-p` 批处理与 `acp`）保持快速失败并输出错误。

## 顶层配置

| 配置项 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `active_provider` | 字符串 | `[providers]` 下首个服务商 | 当前会话默认使用的模型服务商名称 |
| `fast_mode` | 布尔值 | `false` | 快速模式初始开关。开启时为受支持的 GPT 模型请求优先服务，运行时偏好保存在 `preferences.json` 中 |

## 全局配置 `[global]`

控制运行环境、文件路径与执行限制：

```toml
[global]
sessions_dir = "sessions"
log_file = "logs/combined.log"
tool_timeout_secs = 60
# max_iterations = 64
# max_tool_calls = 128
```

- `sessions_dir`：会话记录存储目录，相对路径按配置文件所在目录解析。
- `log_file`：运行日志输出文件，相对路径按配置文件所在目录解析。
- `tool_timeout_secs`：常规工具默认超时秒数。`shell__exec` 内部保持独立默认值 300 秒。
- `max_iterations`：单个交互回合内允许的最大循环轮数。省略表示不设限制。
- `max_tool_calls`：单个交互回合内允许调用的工具总次数。省略表示不设限制。

### 上下文保留 `[global.compaction]`

```toml
[global.compaction]
preserve_recent_tokens = 12000
```

- `preserve_recent_tokens`：执行上下文压缩时，在历史末尾强制保留的 Token 预算。省略时根据当前模型的有效输入上限动态计算。

### 物理重试策略 `[global.retry]`

```toml
[global.retry]
enabled = true
max_attempts = 50
max_recovery_attempts = 3
initial_delay_secs = 1
exponential_backoff = true
backoff_multiplier = 2.0
jitter_secs = 1
```

- `enabled`：是否开启物理网络请求重试。
- `max_attempts`：未产生可观察副作用前允许的最大网络重试次数，计入首次请求。
- `max_recovery_attempts`：模型产生部分输出后，受控恢复迭代的最大重试次数。
- `initial_delay_secs`：首次重试前的初始等待秒数；在 `exponential_backoff = false` 时作为固定等待间隔。
- `exponential_backoff`：是否使用指数退避算法增加等待时间。
- `backoff_multiplier`：指数退避乘数。
- `jitter_secs`：退避时间的随机抖动范围秒数。

### 会话自动归档 `[global.session_archive]`

```toml
[global.session_archive]
enabled = true
older_than_days = 7
```

- `enabled`：是否在后台归档长期闲置的历史会话。
- `older_than_days`：判定会话闲置的天数阈值。归档会话仍可在列表中检索，在执行恢复时自动解压载入。

## 权限控制 `[permissions]`

设置系统执行外部操作时的默认授权级别：

```toml
[permissions]
mode = "default" # safe | default | auto | yolo
```

- `mode`：运行模式，支持四种级别：
  - `safe`：最严格模式。每次调用任何工具前均向用户弹窗请求确认。
  - `default`：默认模式。只读操作直接放行，写入与外部变更前请求确认。
  - `auto`：智能审查模式。由内置的安全审查专家（Reviewer）自动裁决操作安全性。
  - `yolo`：全放行模式。不弹出任何权限询问，直接执行所有工具调用。历史别名 `solo` 会被自动映射到 `yolo`。

## 工具并发控制 `[tools.parallelism]`

限制特定工具的并发执行策略：

```toml
[tools.parallelism]
"fs__read" = "parallel"
"web__fetch" = "exclusive"
```

- 可选值包括 `parallel`（允许在同批次中重叠并发执行）与 `exclusive`（必须排他单线程执行）。该配置只能收窄工具自身声明的能力，不能强制并发未支持的工具。

## MCP 外部服务配置 `[mcp.<name>]`

接入基于 Model Context Protocol（MCP）协议的外部工具服务：

### 本地子进程服务

```toml
[mcp.local_server]
type = "local"
command = ["/path/to/server", "--stdio"]
environment = { ENV_VAR = "value" }
enabled = true
timeout = 5000
```

- `type`：固定为 `"local"`。必填。
- `command`：可执行程序路径及参数列表。必填。
- `environment`：传递给子进程的环境变量键值对。别名为 `env`。
- `enabled`：是否启用该服务，默认为 `true`。也可在会话内使用 `/mcp` 快捷切换。
- `timeout`：服务响应与方法调用的超时毫秒数，默认为 5000 毫秒。

### 远程网络服务

```toml
[mcp.remote_server]
type = "remote"
url = "https://example.com/mcp"
headers = { Authorization = "Bearer YOUR_TOKEN" }
enabled = true
timeout = 10000
```

- `type`：固定为 `"remote"`。必填。
- `url`：远程服务 HTTP/SSE 端点地址。必填。
- `headers`：请求时附带的自定义标头。
- `timeout`：网络请求超时毫秒数。
- `oauth`：远程 MCP 暂不支持 OAuth 授权，必须保持为 `false`。

## 服务商配置 `[providers.<name>]`

声明模型服务商基础信息与认证参数：

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
```

- `protocol`：传输协议类型，支持 `responses`、`completions` 与 `anthropic`。
- `flavor`：接口特征模式，支持 `standard` 或适配 DeepSeek 的 `deepseek`。
- `default_model`：未显式指定模型时的默认模型标识。必须是该服务商 `[models]` 下已定义的键。
- `reviewer`：可选设置。设为 `"jev"` 时将该服务商标记为审批审查后端。Jev 审查要求 `auth.type = "bearer"`。
- `headers`：附加的固定请求头映射表。
- `query`：附加的固定 URL 查询参数映射表。

### 连接与传输 `[providers.<name>.transport]`

```toml
[providers.openai.transport]
connect_timeout_secs = 10
no_proxy_loopback = true
```

- `connect_timeout_secs`：TCP 与 TLS 握手建连超时秒数。
- `no_proxy_loopback`：回环地址（如 `127.0.0.1`）是否强制绕过代理。

### 服务商级重试 `[providers.<name>.retry]`

可选配置。继承 `[global.retry]` 的未声明字段，用于单独调整特定服务商的重试频率与上限。

### 认证配置 `[providers.<name>.auth]`

```toml
[providers.openai.auth]
type = "bearer"
credential_env = "OPENAI_API_KEY"
```

- `type`：认证方案，支持 `bearer`、`header`、`query` 与 `none`。
- `credential_env`：存储 API 密钥的环境变量名。默认值为 `<PROVIDER>_API_KEY`。
- `credential`：明文 API 密钥。优先读取 `credential_env` 指定的环境变量；环境变量为空时才使用该明文。
- `name`：当 `type` 为 `header` 或 `query` 时，携带凭据的标头名或查询参数键名。

### 端点地址与重写 `[providers.<name>.endpoints]`

```toml
[providers.openai.endpoints]
base_url = "https://api.openai.com/v1"

[providers.openai.endpoints.responses]
path = "responses"
query = { api_version = "2026-01-01" }
```

- `base_url`：服务商 API 基础网关地址。
- 子表重写：支持配置 `responses`、`completions`、`anthropic` 与 `jev`（别名 `reviewer`）子表。每个子表支持 `path`（覆盖默认协议路径，可为绝对路径）与 `query`（追加查询参数）。

## 模型配置 `[providers.<name>.models."<model>"]`

在指定服务商下定义模型的推理能力与生成参数：

```toml
[providers.openai.models."gpt-5.5"]
display = "GPT-5.5"
context_window = 400000
effective_input_limit_tokens = 256000

[providers.openai.models."gpt-5.5".capabilities]
tools = true
parallel_tool_calls = true
reasoning = true

[providers.openai.models."gpt-5.5".generation]
temperature = 0.2
max_output_tokens = 128000
reasoning_effort = "medium"
reasoning_efforts = ["none", "low", "medium", "high", "max"]
```

### 基础属性

- `display`：在 TUI 界面与选择菜单中显示的友好名称。
- `model_override`：实际发送给上游网关的真实模型标识；仅在与配置键名不一致时填写。
- `strategy`：模型请求封装策略，省略时根据模型名自动推断。
- `context_window`：总上下文窗口 Token 容量。
- `effective_input_limit_tokens`：有效输入上限 Token 预算。超过该限制后系统启动上下文整理。
- `protocol`：单独覆盖该模型所用的传输协议。
- `flavor`：单独覆盖该模型所用的接口特征模式。
- `transport.websocket`：对于 Responses 协议，是否启用全双工 WebSocket 流式传输。

### 能力声明 `[capabilities]`

声明模型具备的基础能力。所有标志默认均为 `false`：

- `tools`：是否支持工具调用。
- `parallel_tool_calls`：是否支持单次响应中并行调用多个工具。
- `reasoning`：是否支持输出思考推导过程。
- `input_images`：是否支持输入多模态图片。
- `tool_result_images`：是否支持工具返回图片并在多轮对话中传回模型。
- `priority_service`：是否声明优先级调度服务。

### 生成参数与默认值 `[generation]`

设置常规回合的生成默认参数。**注意：每个参数生效的前提是已在 `capabilities` 中开启对应标志**。

- `temperature`：采样温度。
- `top_p`：核采样阈值。
- `max_output_tokens`：最大单次输出 Token 上限。
- `stop_sequences`：停止词列表。
- `reasoning_effort`：默认推理强度，可选 `none`、`minimal`、`low`、`medium`、`high`、`xhigh`、`max` 或服务商自定义名称。要求 `capabilities.generation.reasoning = true`。
- `reasoning_efforts`：提供给当前会话选择的推理强度档位列表。
- `reasoning_summary`：推理摘要模式，可选 `auto`、`concise` 或 `detailed`。
- `text_verbosity`：文本详细程度，可选 `low`、`medium` 或 `high`。
- `parallel_tool_calls`：是否开启并行工具调用。要求 `capabilities.parallel_tool_calls = true`。
- `async_tools`：异步派发的工具名列表。需要 Astra 策略、Responses 协议且 `capabilities.tools = true`。

### 结构化输出 `[capabilities.generation.structured_output]`

声明模型对结构化输出的支持级别：

- `json_schema`：在解码器层硬性约束 JSON 语法有效性。
- `json_object`：仅在系统层面表达输出 JSON 意图。

### 提示词缓存 `[cache]`

```toml
[providers.openai.models."gpt-5.5".cache]
enabled = true
retention = "in_memory" # in_memory | 24h
namespace = "openai"
```

- `enabled`：是否启用服务商原生 Prompt 缓存。
- `retention`：缓存保留策略，`24h` 仅部分服务商的 Responses 协议支持。
- `namespace`：缓存命名空间划分。

### 协议专属选项 `[protocol_settings]`

针对特定协议的专属参数：

```toml
[providers.anthropic.models."claude-3-7-sonnet".protocol_settings]
anthropic_thinking = { mode = "adaptive", budget_tokens = 16000 }
anthropic_betas = ["context-1m-2025-08-07"]
```

- `anthropic_thinking`：Anthropic 思考参数结构，包含 `mode`（`disabled`、`adaptive`、`budget`）与 `budget_tokens`。
- `anthropic_betas`：发送给 Anthropic 接口的 Beta 特性请求头数组。

## 专家路由配置 `[agents.<expert>]`

为特定内置专家指定独立的服务商与模型。配置专家时必须同时填写 `provider` 与 `model`：

```toml
[agents.explorer]
provider = "openai"
model = "gpt-5.5"
allowed_models = ["openai/gpt-5.5"]
reasoning_effort = "medium"
```

支持配置以下 8 个内置专家：

1. `explorer`：代码库只读检索与探索专家。
2. `fixer`：代码实现与故障修复专家。
3. `oracle`：根因分析与风险审查专家。
4. `designer`：系统方案与交互设计专家。
5. `librarian`：技术资料与上下文归档专家。
6. `general`：通用任务辅助专家。
7. `reviewer`：权限安全审查内部专家。
8. `historian`：会话历史整理与上下文压缩内部专家。

- `allowed_models`：该专家允许使用的路由列表，格式为 `provider/model`。列表第一项作为默认路由；后续项限定单次 `agent__*` 委派时可通过参数覆盖的目标模型范围。未配置专家路由时，系统自动回退至全局默认模型。
- `reasoning_effort`：该专家每次运行使用的思考强度。所选模型必须支持该档位。

## 仿真伪装配置 `[fake]`

在开启 `/fake` 时伪装请求元数据特征，模拟特定开发工具环境：

```toml
[fake.identity]
# installation_id = "..."
# agent_name = "Hypatia"

[fake.clock]
timezone = "Asia/Shanghai"

[fake.codex]
version = "0.153.4"
originator = "Codex Desktop"
sandbox = "none"
sandbox_mode = "danger-full-access"
shell = "zsh"

[fake.claude]
version = "2.1.285"
package_version = "0.74.0"
timeout = "600"
```

- `[fake.identity]`：客户端安装 ID 与代理标识。
- `[fake.clock]`：请求模拟使用的 IANA 时区名称。
- `[fake.codex]`：Codex 桌面端环境剖面参数。`[fake.codex.extra]` 支持最多 16 个额外的请求元数据键。
- `[fake.claude]`：Claude Code 命令行工具剖面参数。

未声明的字段由系统在运行时自动检测当前主机的真实环境补充。详细规范参见[客户端特征仿真](fake.md)。

## 最小可用示例

创建 `~/.config/letcode/letcode.toml` 并写入以下内容即可启动：

```toml
#:schema https://raw.githubusercontent.com/letr007/letcode/main/letcode.schema.json

active_provider = "openai"

[permissions]
mode = "default"

[providers.openai]
protocol = "responses"
default_model = "gpt-5.5"

[providers.openai.auth]
type = "bearer"
credential_env = "OPENAI_API_KEY"

[providers.openai.endpoints]
base_url = "https://api.openai.com/v1"

[providers.openai.models."gpt-5.5"]
display = "GPT-5.5"

[providers.openai.models."gpt-5.5".capabilities]
tools = true
reasoning = true

[providers.openai.models."gpt-5.5".generation]
reasoning_effort = "medium"
```

## 源码索引

- `src/config.rs`：配置结构定义、默认值与加载校验逻辑。
- `src/config/persistence.rs`：配置文件保格式读写、Schema 字段提取与原子落盘。
- `src/config/schema.rs`：内置 JSON Schema 解析与字段元数据查询。
- `src/tui/runtime.rs`：`/config` TUI 配置编辑器交互处理与状态管理。
- `src/tui/recover.rs`：启动阶段错误配置文件交互式修复入口。
