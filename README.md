<h1 align="center">
  LetCode
</h1>

<p align="center">
  letcode is a terminal Agent written in Rust.
</p>

<p align="center">
  <a href="https://github.com/letr007/letcode/actions/workflows/test.yml"><img src="https://img.shields.io/github/actions/workflow/status/letr007/letcode/test.yml?branch=main&style=flat-square" alt="Test"></a>
  <a href="CHANGELOG.md"><img src="https://img.shields.io/badge/changelog-0.11.0-informational?style=flat-square" alt="Changelog"></a>
  <a href="LICENSE-MIT"><img src="https://img.shields.io/badge/license-MIT%20%7C%20Apache--2.0-blue?style=flat-square" alt="MIT License | Apache-2.0 License"></a>
</p>

<p align="center">
  <a href="README.zh-CN.md">中文</a> | English
</p>

![letcode TUI](docs/letcode.png)

It provides an `opencode`-style TUI based on Ratatui, and also keeps a REPL CLI mode.

[Technical documentation](docs/index.md)

## Build and run

```sh
cargo build
cargo test
cargo fmt --check
```

Run the default TUI:

```sh
cargo run
```

Run the line-based CLI:

```sh
cargo run -- --cli
```

CLI mode can also be selected with `cli` or `repl`. TUI can be selected explicitly with `--tui` or `tui`.

Show the installed version and check for a newer GitHub release:

```sh
letcode --version
letcode update check
```

Update a release-installed binary after an interactive confirmation:

```sh
letcode update
```

The TUI supports English (`en`) and Simplified Chinese (`zh-CN`). Use `/language` or its `/lang` alias to switch languages at runtime.

## External dependencies

Building from source requires the Rust toolchain. Some built-in tools also invoke the following external programs, which must be available on `PATH`:

| Program | Used by | Requirement |
| --- | --- | --- |
| [`git`](https://git-scm.com/) | `git__status`, `git__diff`, `git__log`, and the TUI branch indicator | Recommended; only Git-related capabilities are unavailable when missing |
| [`rg`](https://github.com/BurntSushi/ripgrep) | `search__rg` text search | Recommended; the search tool is unavailable when missing |
| [`ast-grep`](https://ast-grep.github.io/) | `code__ast_search` and `code__ast_replace_preview` | Optional; only AST tools are unavailable when missing |

In addition, `shell__exec` and local MCP servers depend on the system commands they invoke, while `web__fetch` and remote MCP require network access.

## Configuration

`letcode` loads configuration from:

```text
~/.config/letcode/letcode.toml
```

Configuration example:

```toml
# Optional; defaults to the first provider in the file.
active_provider = "openai"
# Optional; defaults to false.
fast_mode = false

# Optional; all values below have defaults.
[global]
# max_iterations = 64
# max_tool_calls = 128
# tool_timeout_secs = 60
sessions_dir = "sessions"
log_file = "logs/combined.log"

# Optional; by default, recent context is preserved according to the active model's input budget.
[global.compaction]
# preserve_recent_tokens = 12000

# Optional; values below are the defaults.
[global.retry]
enabled = true
max_attempts = 50
max_recovery_attempts = 3
initial_delay_secs = 1
backoff_multiplier = 2.0
jitter_secs = 1

# Optional; defaults to default. Values: safe | default | auto | yolo.
[permissions]
mode = "default" # solo remains accepted as a yolo alias

# Optional; choose a default route and per-invocation allowed routes for an expert.
# [agents.explorer]
# provider = "openai"
# model = "gpt-5.5"
# allowed_models = ["openai/gpt-5.5"]
# The same shape applies to fixer, oracle, designer, librarian, general, and reviewer.

# Optional; this can only narrow parallelism declared by a tool itself.
[tools.parallelism]
# "fs__read" = "parallel"
# "web__fetch" = "exclusive"

# Optional local MCP server.
# [mcp.example_local]
# type = "local"
# command = ["/path/to/mcp-server", "--stdio"]
# environment = { FOO = "bar" }
# enabled = true
# timeout = 5000

# Optional remote MCP server; OAuth is not currently supported.
# [mcp.example_remote]
# type = "remote"
# url = "https://example.com/mcp"
# headers = { Authorization = "Bearer ..." }
# enabled = true
# timeout = 10000

# Required: configure at least one provider with at least one model.
[providers.openai]
protocol = "responses" # responses | completions | anthropic
flavor = "standard" # standard | deepseek
# Required; must reference a model configured for this provider.
default_model = "gpt-5.5"

[providers.openai.auth]
type = "bearer" # bearer | header | query | none
credential_env = "OPENAI_API_KEY"
# credential = "YOUR_API_KEY" # use this instead of credential_env when appropriate

[providers.openai.endpoints]
base_url = "https://api.openai.com/v1"
[providers.openai.endpoints.responses]
path = "responses"

# Optional provider connection settings.
# [providers.openai.transport]
# connect_timeout_secs = 10
# no_proxy_loopback = true

# Required: each provider needs at least one model; every field inside the model is optional.
[providers.openai.models."gpt-5.5"]
display = "GPT-5.5"
# protocol = "completions" # overrides the provider protocol
# flavor = "standard" # overrides the provider flavor; deepseek selects the explicit DeepSeek profile
# context_window = 400000
# effective_input_limit_tokens = 256000

# Optional model transport; defaults to HTTP/SSE. Set websocket = true for normal Agent
# turns using Responses on a WebSocket-capable endpoint. Title generation and compaction
# one-shot requests remain on HTTP/SSE.
[providers.openai.models."gpt-5.5".transport]
websocket = false

# Capability flags default to false when omitted.
[providers.openai.models."gpt-5.5".capabilities]
tools = true
parallel_tool_calls = true
reasoning = true
input_images = false
tool_result_images = false
prompt_cache = false
priority_service = false
[providers.openai.models."gpt-5.5".capabilities.generation]
temperature = true
top_p = true
max_output_tokens = true
reasoning = true
reasoning_summary = true
text_verbosity = true
parallel_tool_calls = true

[providers.openai.models."gpt-5.5".generation]
temperature = 0.2
top_p = 1.0
max_output_tokens = 128000
reasoning_effort = "medium"
# Optional; restricts selectable reasoning levels and the TUI cycle order.
reasoning_efforts = ["none", "low", "medium", "high", "max"]
reasoning_summary = "auto" # auto | concise | detailed
text_verbosity = "medium" # low | medium | high
parallel_tool_calls = true

[providers.openai.models."gpt-5.5".cache]
enabled = false
# retention = "in_memory" # in_memory | 24h; set when cache is enabled
# namespace = "openai"

# Adapter-specific settings are validated by the selected protocol binding.
# Anthropic routes may set, for example:
# anthropic_thinking = { mode = "adaptive" }
# anthropic_betas = ["context-1m-2025-08-07"]
[providers.openai.models."gpt-5.5".protocol_settings]
```

Provider credentials can use `credential_env` or the default environment variable named from the provider, for example `OPENAI_API_KEY`; endpoint URLs and protocol-specific paths are configured under `endpoints`.

Relative `sessions_dir` and `log_file` paths are resolved relative to the config file directory.

Optional Langfuse/OpenTelemetry tracing is off by default. Enable it with `LETCODE_LANGFUSE_ENABLED=true`, and set `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and optional `LANGFUSE_HOST` (or the same variables in a local `.env`). Missing credentials leave tracing disabled without stopping the agent.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for release notes.

## License

This project is dual-licensed under the MIT License OR the Apache License 2.0.
You may choose either license when using, modifying, or redistributing this project.

- MIT License: see [LICENSE-MIT](LICENSE-MIT)
- Apache License 2.0: see [LICENSE-APACHE](LICENSE-APACHE)
