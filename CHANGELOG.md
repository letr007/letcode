# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-08-05

First public release. letcode has been dogfooded on its own development for some time; this tag freezes that baseline.

### Added

- Ratatui TUI (opencode-inspired) and line-based CLI/REPL frontends over a shared session engine
- Multi-provider model routing, reasoning effort controls, and expert agent model overrides
- Permission modes: `safe`, `default`, `auto`, `yolo` (legacy `solo` still accepted when reading config)
- Sticky reviewer expert for `auto` mode, with compact request/decision cards in the reviewer child view
- Tool surface for shell, filesystem, search, web fetch, git, workflows, skills, and subagents
- Parallel tool calls where handlers opt in; session-local AllowAlways grants on the default/auto ask matrix
- Append-only JSONL session transcripts with resume, history tree, and undo/redo in the TUI
- Context compaction, hot-reload for supported runtime config, and optional Langfuse/OpenTelemetry tracing
- Selectable TUI themes and structured cards for tools, todos, permissions, and subagent results

### Changed

- Default/Auto no longer hard-deny risky shell commands (`curl`, `rm -rf`, …); they Ask (human or reviewer) instead
- Agent prompts and operator-facing guidance localized to Chinese

### Fixed

- Session resume append safety, child/parent view transitions, and compaction/context-branch validation
- Retry, streaming, and interruption edge cases around subagents and queued prompts

[0.1.0]: https://github.com/letr007/letcode/releases/tag/v0.1.0
