# CLI 命令行与批处理指南

letcode 提供多种命令行运行方式。除默认的图形化终端界面（TUI）外，系统支持轻量级交互命令行（REPL）、单次批处理调用、会话恢复、配置脱机校验、在线二进制更新以及外部协议服务。

## 运行模式汇总

```sh
letcode [子命令 / 参数]
```

| 命令形态 | 运行模式 | 交互形式 | 适用场景 |
| --- | --- | --- | --- |
| `letcode` / `letcode --tui` | 默认 TUI 模式 | 全屏终端界面 | 交互式日常开发与复杂编程任务 |
| `letcode --cli` / `letcode repl` | 行命令 REPL 模式 | 逐行提示符输入 | 简易终端、SSH 会话或轻量资源环境 |
| `letcode -p "..."` | 单次批处理模式 | 非交互单回合运行 | 脚本化调用、自动化任务与管道处理 |
| `letcode resume <id>` | 会话恢复模式 | 进入 TUI 恢复指定会话 | 恢复中断的工作或切换历史上下文 |
| `letcode acp` | ACP 服务端模式 | stdio 协议通信 | 供 Zed、VS Code 或外部控制中心接入 |
| `letcode config validate` | 配置校验命令 | 命令行输出校验结果 | 检查 `letcode.toml` 语法与模型配置 |
| `letcode update` | 自动更新命令 | 交互式更新提示 | 检查并升级 GitHub Release 二进制文件 |

## 命令行选项

### 1. 启动交互式终端

默认情况下，直接执行 `letcode` 进入全屏 TUI 模式。若需显式指定：

```sh
letcode --tui
```

若需在纯文本终端或简单终端下工作，可启动行交互式 REPL：

```sh
letcode --cli
# 或使用等价别名
letcode repl
```

在 REPL 模式下，用户可直接输入提示词与模型对话。需要退出时输入 `/exit` 或按 `Ctrl+D`。

### 2. 单次批处理执行

使用 `-p` 或 `--prompt` 可以直接发起单次非交互调用。执行完毕后系统自动退出，适合与 Shell 管道或自动化脚本集成：

```sh
letcode -p "审查当前 git diff 并给出修改建议"
```

配合 `--json` 参数可以将输出转换为结构化 JSON 流：

```sh
letcode -p "分析当前代码复杂度" --json
```

在 JSON 输出模式下，标准输出逐行打印序列化的事件对象，包含思考片段、工具执行过程与最终生成的响应文本。

### 3. 会话恢复（Resume）

系统支持根据会话 ID 快速恢复既有上下文：

```sh
letcode resume session-20260925-102400
```

命令会加载该会话对应的 JSONL 日志，重建对话分支与运行时快照，并直接打开 TUI 界面供用户继续操作。

### 4. 配置文件脱机校验

在调整 `letcode.toml` 后，无需启动完整会话即可校验配置合法性：

```sh
# 校验默认路径 ~/.config/letcode/letcode.toml
letcode config validate

# 校验指定路径的配置文件
letcode config validate ./my-custom-config.toml
```

校验器会检查 TOML 语法、必填字段、服务商协议、模型配置引用以及专家路由规则。若存在错误，命令输出清晰的排查提示并返回非零状态码。

### 5. 版本检查与自动升级

letcode 内置基于 GitHub Release 的安全升级机制：

```sh
# 仅检查是否有新版本可用
letcode update check

# 检查、下载并自动替换当前二进制文件
letcode update
```

升级流程会自动检测当前操作系统的平台架构（如 `x86_64` 或 `aarch64`），下载对应的打包产物，校验 SHA-256 哈希值，对当前运行的文件执行备份，并完成原地原子替换。

## 环境变量参考

letcode 在启动时读取以下环境变量：

| 环境变量 | 作用说明 |
| --- | --- |
| `HOME` | 定位用户主目录，用于解析默认配置文件与全局技能路径 |
| `OPENAI_API_KEY` | OpenAI 服务商默认读取的凭据变量（可在配置中自定义） |
| `ANTHROPIC_API_KEY` | Anthropic 服务商默认读取的凭据变量 |
| `LETCODE_LANGFUSE_ENABLED` | 设置为 `true` 开启 Langfuse / OpenTelemetry 调用遥测 |
| `LANGFUSE_PUBLIC_KEY` | Langfuse 遥测公钥 |
| `LANGFUSE_SECRET_KEY` | Langfuse 遥测私钥 |
| `LANGFUSE_HOST` | 自定义 Langfuse 服务端地址，默认连接官方云服务 |

## 源码索引

- `src/main.rs`：负责解析命令行入参、匹配 `EntryMode` 并分发到具体前端。
- `src/cli.rs`：实现行命令 REPL 模式与单次批处理调用的执行逻辑。
- `src/updater.rs`：实现 GitHub Release 检查、平台包匹配、哈希校验与原子更新。
- `src/config.rs`：实现配置校验器 `run_config_validate`。
- `src/acp/mod.rs`：处理 `acp` 子命令启动。
