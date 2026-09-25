<h1 align="center">
  LetCode
</h1>

<p align="center">
  letcode is a terminal Agent written in Rust.
</p>

<p align="center">
  <a href="https://github.com/letr007/letcode/actions/workflows/test.yml"><img src="https://img.shields.io/github/actions/workflow/status/letr007/letcode/test.yml?branch=main&style=flat-square" alt="Test"></a>
  <a href="CHANGELOG.md"><img src="https://img.shields.io/badge/changelog-0.17.0-informational?style=flat-square" alt="Changelog"></a>
  <a href="LICENSE-MIT"><img src="https://img.shields.io/badge/license-MIT%20%7C%20Apache--2.0-blue?style=flat-square" alt="MIT License | Apache-2.0 License"></a>
</p>

<p align="center">
  <a href="README.zh-CN.md">中文</a> | English
</p>

![letcode TUI](docs/letcode.png)

letcode is a terminal coding Agent built for developer productivity. Powered by Ratatui, it offers an interactive terminal user interface (TUI), a line-based REPL CLI, automated single-shot batch processing, and a built-in standard Agent Client Protocol (ACP) server.

## Key Features

- **Multi-Expert Collaboration**: Six built-in subagents (`explorer`, `fixer`, `oracle`, `designer`, `librarian`, `general`) coordinated by a concurrent pool with path-based exclusive locking.
- **Terminal-Native Rich Rendering**: Streams Markdown, renders LaTeX math formulas, and draws Mermaid vector diagrams (flowcharts, sequence diagrams, state diagrams) natively in the terminal grid.
- **Native Multi-Protocol Support**: Full native support for OpenAI Responses, Chat Completions, and Anthropic Messages protocols, including full-duplex WebSocket streaming for Responses.
- **Layered Security & Permission Model**: Four operational modes (`safe`, `default`, `auto`, `yolo`) with session-scoped grant tables and parameter fingerprinting.
- **Intelligent Compaction & Long-Term Memory**: Three-tier incremental history compaction via the internal `historian` expert, paired with a workspace-scoped SQLite store for long-term project memory.
- **Extensible Ecosystem**: Supports local stdio and remote HTTP Model Context Protocol (MCP) tool servers, alongside a native ACP server for seamless integration with modern editors like Zed.

## Quick Start

### 1. Installation

**Method A: Pre-built binaries (Recommended)**

Download the pre-compiled archive for your OS and architecture from [GitHub Releases](https://github.com/letr007/letcode/releases), extract it, and place the `letcode` binary into your `PATH`.

**Method B: Build from source**

Ensure the Rust toolchain (1.85+ recommended) is installed, then run in the repository root:

```sh
cargo install --path .
```

### 2. Configuration

letcode loads its configuration from `~/.config/letcode/letcode.toml`. Create this file with minimal settings:

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

Export your API key in your shell:

```sh
export OPENAI_API_KEY="your-api-key"
```

For advanced settings (multi-provider routing, timeouts, retries, MCP services, and environment disguises), see the [Configuration Reference Manual](docs/components/configuration.md).

### 3. Run

Launch the default terminal interface in any project directory:

```sh
letcode
```

The TUI supports English (`en`) and Simplified Chinese (`zh-CN`). Switch languages with `/language zh-CN` or `/language en`.

## Common Commands

| Use Case | Command Example | Description |
| --- | --- | --- |
| **Interactive TUI** | `letcode` or `letcode --tui` | Launch the full-featured terminal interface (default) |
| **Line REPL** | `letcode --cli` or `letcode repl` | Launch the lightweight line-based CLI |
| **Batch Processing** | `letcode -p "review git diff" --json` | Single-shot prompt execution with streaming JSON output |
| **Resume Session** | `letcode resume <session-id>` | Restore conversation context and open the TUI |
| **ACP Server** | `letcode acp` | Start the stdio protocol server for Zed and other editors |
| **Validate Config** | `letcode config validate` | Validate `letcode.toml` syntax and model parameters |
| **Check & Update** | `letcode update check`<br/>`letcode update` | Check GitHub releases and perform an in-place atomic update |

For detailed command-line options, see the [CLI & Batch Processing Guide](docs/components/cli.md).

## Technical Documentation

Comprehensive architectural and component documentation is available in the [Technical Documentation Index](docs/index.md).

## External Dependencies

Building from source requires the Rust toolchain. Some built-in tools also invoke external programs when available on `PATH`:

| Program | Used by | Requirement |
| --- | --- | --- |
| [`git`](https://git-scm.com/) | `git__status`, `git__diff`, `git__log`, and TUI branch indicator | Recommended; only Git tools and branch indicators are affected when missing |
| [`rg`](https://github.com/BurntSushi/ripgrep) | `search__rg` fast text search | Recommended; search tools are unavailable when missing |
| [`ast-grep`](https://ast-grep.github.io/) | `code__ast_search` and `code__ast_replace_preview` AST analysis | Optional; only AST tools are unavailable when missing |

`shell__exec` and local MCP servers depend on the underlying system commands invoked; `web__fetch` and remote MCP servers require network access.

## License

Dual-licensed under MIT and Apache-2.0. You may use, modify, or distribute this project under either license:

- MIT License: see [LICENSE-MIT](LICENSE-MIT)
- Apache License 2.0: see [LICENSE-APACHE](LICENSE-APACHE)
