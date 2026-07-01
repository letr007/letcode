# letcode

[中文](README.zh-CN.md) | English

`letcode` is a terminal Agent written in Rust. It provides an `opencode`-style TUI based on Ratatui by default, while keeping a REPL CLI mode available.

## Build and test

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

## Configuration

`letcode` loads configuration from:

```text
~/.config/letcode/letcode.toml
```

Minimal example:

```toml
active_provider = "openai"

[global]
# Optional runtime limits:
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
# context_window = 400000
# effective_input_limit_tokens = 256000 # optional provider/model route input budget
supports_tools = true
supports_reasoning = true
reasoning_effort = "medium"
reasoning_summary = "auto"
text_verbosity = "medium"
```

Provider API keys and base URLs can also come from environment variables. The variable names are generated from the provider name:

```sh
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

If the provider is named `compat`, the corresponding variables are `COMPAT_API_KEY` and `COMPAT_BASE_URL`.

Relative `sessions_dir` and `log_file` paths are resolved relative to the config file directory.

### Optional Langfuse tracing

Langfuse/OpenTelemetry tracing is off by default and does not change agent behavior. When enabled, `letcode` exports only safe operational metadata for LLM turns, streamed model calls, tool calls, status, token counts, and latency. It does not export raw prompts, raw tool arguments, raw tool outputs, API keys, or `.env` contents.

Enable it with environment variables, or place the same variables in a local `.env` file:

```sh
LETCODE_LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

If Langfuse credentials are missing or tracing cannot initialize, `letcode` continues running with Langfuse tracing disabled.

## Session context and restore

Session transcripts are stored as append-only JSONL records under `sessions_dir`. Restore rebuilds conversation history, context branches, the context tree, and the prompt context view from those records. Context view operations such as archive or remove-from-view append metadata only; they do not purge raw transcript events.

The context tree is a strict tree of session/task context nodes. It records active and archived nodes for prompt assembly and TUI display, but it does not imply filesystem rollback. Hard constraints, current user requirements, unresolved errors including invariant violations, permission decisions, file write facts, validation/test results, and commit hashes remain protected context when they are part of the restored session.

The context view is a derived prompt projection. It keeps stable hard context ahead of dynamic details, supports pinned blocks, summaries, and opened details, and hides archived or removed soft blocks from prompt-visible sections. Large shell outputs are folded into openable metadata by default; folded output is not a semantic summary.

In the TUI, use `/context` to browse context nodes, blocks, summaries, and folded outputs. Legacy `context__checkpoint` / `context__return` records remain compatible with the newer context tree metadata.

## Project layout

```text
src/main.rs          entry point, config loading, TUI/CLI selection
src/config.rs        TOML config parsing and validation
src/agent.rs         model loop, tool execution, turn lifecycle
src/context_tree.rs  session context tree replay and invariants
src/context_view.rs  derived prompt context blocks, summaries, and folding
src/tool.rs          built-in tool registry and tool result model
src/permission.rs    permission modes, scopes, and request classification
src/transcript.rs    JSONL transcript persistence and restore helpers
src/request_builder.rs prompt assembly and context-view insertion
src/subagent.rs      subagent-related code
src/mcp.rs           MCP tool discovery
src/tui/             Ratatui/Crossterm UI, runtime, state, events, rendering
```

## License

This project is dual-licensed under the MIT License OR the Apache License 2.0.
You may choose either license when using, modifying, or redistributing this project.

- MIT License: see [LICENSE-MIT](LICENSE-MIT)
- Apache License 2.0: see [LICENSE-APACHE](LICENSE-APACHE)
