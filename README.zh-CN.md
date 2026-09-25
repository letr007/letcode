<h1 align="center">
  LetCode
</h1>

<p align="center">
  letcode 是 Rust 编写的终端 Agent。
</p>

<p align="center">
  <a href="https://github.com/letr007/letcode/actions/workflows/test.yml"><img src="https://img.shields.io/github/actions/workflow/status/letr007/letcode/test.yml?branch=main&style=flat-square" alt="Test"></a>
  <a href="CHANGELOG.md"><img src="https://img.shields.io/badge/changelog-0.17.0-informational?style=flat-square" alt="Changelog"></a>
  <a href="LICENSE-MIT"><img src="https://img.shields.io/badge/license-MIT%20%7C%20Apache--2.0-blue?style=flat-square" alt="MIT License | Apache-2.0 License"></a>
</p>

<p align="center">
  中文 | <a href="README.md">English</a>
</p>

![letcode TUI](docs/letcode.png)

letcode 是面向生产力打造的终端编程 Agent。它基于 Ratatui 构建了交互式终端界面（TUI），提供行命令式 REPL、自动化单次批处理调用，并内置标准 Agent Client Protocol（ACP）协议服务端。

## 核心特性

- **多专家协同机制**：内置 `explorer`（只读探索）、`fixer`（代码修复）、`oracle`（根因分析与风险审查）、`designer`（方案设计）、`librarian`（资料检索）与 `general`（通用辅助）六类专家，基于任务池管理并发调度与独占文件路径锁；
- **终端原生富文本渲染**：在终端网格中原生渲染流式 Markdown、LaTeX 数学公式与 Mermaid 矢量图表（流程图、时序图、状态图），无需外部浏览器辅助；
- **原生多协议支持**：完整支持 OpenAI Responses、Chat Completions 与 Anthropic Messages 协议，支持 Responses 协议下的全双工 WebSocket 流式传输；
- **分层安全权限模型**：提供 `safe`（完全人工确认）、`default`（读放行写确认）、`auto`（智能审查自动判定）与 `yolo`（全自动）四种运行模式，支持会话级授权表与参数特征比对；
- **智能历史整理与长期记忆**：通过内部 `historian` 专家执行三档增量上下文压缩，配合后台 SQLite 数据库实现工作区隔离的长期项目记忆库；
- **开放外部生态**：支持本地 stdio 与远程 HTTP 形式的 Model Context Protocol（MCP）工具服务，内置 ACP 服务端支持与 Zed 等现代编辑器直接集成。

## 快速上手

### 1. 安装

**方式 A：下载预编译二进制（推荐）**

从 [GitHub Releases](https://github.com/letr007/letcode/releases) 下载适合当前操作系统的打包文件，解压后将 `letcode` 二进制放入系统的 `PATH` 路径中。

**方式 B：通过源码编译安装**

确保本地已安装 Rust 工具链（建议 1.85+），在源码目录执行：

```sh
cargo install --path .
```

### 2. 配置

letcode 启动时读取 `~/.config/letcode/letcode.toml` 配置文件。创建该文件并填入最小配置：

```toml
active_provider = "openai"

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
```

随后在终端导出对应的 API 密钥：

```sh
export OPENAI_API_KEY="your-api-key"
```

完整的高级设置（多服务商配置、超时策略、重试控制、MCP 服务与环境仿真）参见 [配置参考手册](docs/components/configuration.md)。

### 3. 运行

在任意工作区目录下直接启动默认的终端界面：

```sh
letcode
```

TUI 界面支持 English（`en`）与简体中文（`zh-CN`）。进入界面后可输入 `/language zh-CN` 切换显示语言。

## 常用命令

| 使用场景 | 命令示例 | 说明 |
| --- | --- | --- |
| **交互式 TUI** | `letcode` 或 `letcode --tui` | 启动全功能终端用户界面（默认模式） |
| **行交互 REPL** | `letcode --cli` 或 `letcode repl` | 启动轻量级行交互命令行，按行提交提示词 |
| **脚本批处理** | `letcode -p "审查当前 git diff" --json` | 单次执行提示词，输出结构化 JSON 数据流并自动退出 |
| **恢复既有会话** | `letcode resume <session-id>` | 读取指定历史日志，恢复分支上下文并进入 TUI |
| **ACP 协议服务** | `letcode acp` | 启动标准输入输出协议服务，供 Zed 等外部编辑器接入 |
| **脱机配置校验** | `letcode config validate` | 校验 `letcode.toml` 格式与模型参数合法性 |
| **版本检查与升级** | `letcode update check`<br/>`letcode update` | 检查 GitHub Release 新版本并执行安全原子升级 |

详尽的命令行选项与参数说明参见 [CLI 命令行与批处理指南](docs/components/cli.md)。

## 技术文档

项目提供完备的组件设计与架构技术文档。参见 [技术文档总览](docs/index.md)。

## 外部依赖

从源码构建需要 Rust 工具链。部分内置工具会调用以下外部程序，请确保它们位于 `PATH` 中：

| 程序 | 使用位置 | 是否必需 |
| --- | --- | --- |
| [`git`](https://git-scm.com/) | `git__status`、`git__diff`、`git__log` 与分支状态显示 | 推荐安装。缺少时仅 Git 工具与分支状态受限 |
| [`rg`](https://github.com/BurntSushi/ripgrep) | `search__rg` 快速文本搜索 | 推荐安装。缺少时文本搜索工具不可用 |
| [`ast-grep`](https://ast-grep.github.io/) | `code__ast_search` 与 `code__ast_replace_preview` 语法树分析 | 可选。缺少时仅语法树分析工具不可用 |

`shell__exec` 与本地 MCP 依赖实际调用的系统命令；`web__fetch` 与远程 MCP 需要可用的网络连接。

## 开源协议

本项目采用 MIT License 与 Apache License 2.0 双协议授权。使用、修改或再分发本项目时，你可以任选其中一种协议：

- MIT License：见 [LICENSE-MIT](LICENSE-MIT)
- Apache License 2.0：见 [LICENSE-APACHE](LICENSE-APACHE)
