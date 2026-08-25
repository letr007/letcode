<h1 align="center">
  LetCode
</h1>

<p align="center">
  letcode is a terminal Agent written in Rust.
</p>

<p align="center">
  <a href="https://github.com/letr007/letcode/actions/workflows/test.yml"><img src="https://img.shields.io/github/actions/workflow/status/letr007/letcode/test.yml?branch=main&style=flat-square" alt="Test"></a>
  <a href="CHANGELOG.md"><img src="https://img.shields.io/badge/changelog-0.3.0-informational?style=flat-square" alt="Changelog"></a>
  <a href="LICENSE-MIT"><img src="https://img.shields.io/badge/license-MIT%20%7C%20Apache--2.0-blue?style=flat-square" alt="MIT License | Apache-2.0 License"></a>
</p>

<p align="center">
  <a href="README.zh-CN.md">中文</a> | English
</p>

![letcode TUI](docs/letcode.png)

It provides an `opencode`-style TUI based on Ratatui, and also keeps a REPL CLI mode.

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
# Optional; OPENAI_API_KEY may be used instead.
api_key = "YOUR_API_KEY"
# Optional for the OpenAI provider; defaults to https://api.openai.com/v1.
base_url = "https://api.openai.com/v1"
# Optional for the OpenAI provider, where it defaults to responses; required for other providers.
protocol = "responses" # responses | completions
# Optional; defaults to the first model configured for this provider.
default_model = "gpt-5.5"

# Required: each provider needs at least one model; every field inside the model is optional.
[providers.openai.models."gpt-5.5"]
display_name = "GPT-5.5"
# protocol = "completions" # overrides the provider protocol
# context_window = 400000
# effective_input_limit_tokens = 256000
# max_output_tokens = 128000
supports_tools = true # default: true
parallel_tool_calls = true # default: true
supports_reasoning = true # default: true
reasoning_effort = "medium"
# Optional; restricts selectable reasoning levels and the TUI cycle order.
reasoning_efforts = ["none", "low", "medium", "high", "max"]
reasoning_summary = "auto" # auto | concise | detailed
text_verbosity = "medium" # low | medium | high
# temperature = 0.2
# top_p = 1.0

# Optional model-level prompt cache.
# [providers.openai.models."gpt-5.5".prompt_cache]
# enabled = true
# retention = "in_memory" # in_memory | 24h
# namespace = "openai"
```

Provider API keys and base URLs can also come from environment variables named from the provider, for example `OPENAI_API_KEY` / `OPENAI_BASE_URL`; for a provider named `compat`, use `COMPAT_API_KEY` / `COMPAT_BASE_URL`.

Relative `sessions_dir` and `log_file` paths are resolved relative to the config file directory.

Optional Langfuse/OpenTelemetry tracing is off by default. Enable it with `LETCODE_LANGFUSE_ENABLED=true`, and set `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and optional `LANGFUSE_HOST` (or the same variables in a local `.env`). Missing credentials leave tracing disabled without stopping the agent.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for release notes.

## License

This project is dual-licensed under the MIT License OR the Apache License 2.0.
You may choose either license when using, modifying, or redistributing this project.

- MIT License: see [LICENSE-MIT](LICENSE-MIT)
- Apache License 2.0: see [LICENSE-APACHE](LICENSE-APACHE)
