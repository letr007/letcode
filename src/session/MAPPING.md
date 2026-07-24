# Session Event / Command Contract Mapping

`session::SessionEvent` is the frontend-neutral outbound event contract. It owns
all event payloads and has no dependency on `crate::tui`.

`session::SessionCommand` is the frontend-neutral inbound command contract.

## Compatibility mapping

| Legacy TUI symbol | Contract | Direction |
| --- | --- | --- |
| `tui::events::AppEvent` | `session::SessionEvent` | Type alias (`pub use ... as AppEvent`) |
| `tui::events::<Payload>` | `session::<Payload>` | Re-export |
| `tui::runner::RunnerEvent` | `session::RunnerEvent` | Re-export shim; runtime compatibility bridge |
| `tui::runtime::RuntimeCommand` | `session::SessionCommand` | Type alias |

## Runner event mapping

`session::RunnerEvent` remains the richer runner-to-frontend transport during
migration. `session::SessionEvent` is the frontend-neutral projection where a
one-to-one event exists.

| `RunnerEvent` variant | `SessionEvent` mapping | Notes |
| --- | --- | --- |
| `UserMessage`, `ReasoningDelta`, `ReasoningDone`, `AssistantDelta`, `AssistantDone` | Same-named variant | Direct payload mapping. |
| `TokenUsage`, `ToolPending`, `ToolCancelled`, `ToolStarted`, `ToolFinished`, `ToolOutputDelta`, `TodoSnapshot`, `AutoContinueChanged` | Same-named variant | Direct payload mapping. |
| `PermissionRequested`, `PermissionResolved`, `ProcessIssue`, `Notice`, compaction, runtime-context, and context-projection events | Same-named variant | Direct payload mapping. |
| `Interrupted`, `Error`, `Done` | Same-named variant | Terminal/control mapping. |
| `Quit` / `Tick` | Session-only (not on runner) | Frontend/local control. |

### Runner-only variants (Phase C documentation)

These stay on the runner channel for now. They are **not** incomplete SessionEvent
bugs; multi-frontend product surfaces may later promote a subset.

| Runner-only variant | Category | Phase C status |
| --- | --- | --- |
| `SessionTokenUsage` | session lifecycle metrics | Runner extra; promote when CLI/GUI need parity |
| `SessionStarted`, `SessionResumed` | session lifecycle | Runner extra |
| `QueuedPromptAccepted` | TUI queue handshake | TUI-private transport for now |
| `QuestionRequested`, `ChildQuestionRequested` | interactive handles | Runner transport (oneshot handles) |
| `ChildPermissionRequested`, `ChildAppEvent`, `ChildSessionViewed` | subagent/child view | Runner + TUI child surface |
| `ContextBranchChanged`, `ContextBranchesLoaded` | branch UI | Runner extra until SessionEvent grows branch ops |
| `McpToolsDiscovered`, `McpServerUpdating`, `McpServerUpdated`, `McpServerToolsUpdated`, `McpDiscoveryUnavailable`, `McpDiagnostic` | MCP catalog UI | Runner extra |
| `ToolBatchFinished` | turn batching signal | Runner extra |

The aliases in `src/tui/events.rs` preserve existing imports in the TUI. New
backend code must use `crate::session::{SessionEvent, ...}` directly.

## Command path (Phase C)

```text
CommandIntent
  ├─ SessionCommand::from_command_intent → Some(SessionCommand)
  │     CLI:  → ReplCommand (execution or Unsupported)
  │     TUI:  → RuntimeCommand alias → SessionCommandHandler adapter
  │              → RunnerControl { Command(RunnerCommand) | Interrupt }
  └─ None (presentation-only) → frontend-local handling
```

| Component | Location | Notes |
| --- | --- | --- |
| `SessionCommand` | `session::command` | FE → BE |
| `SessionCommand::from_command_intent` | `session::command` | Shared backend vs local classification |
| `SessionCommandHandler` | `session::ports` | Trait; TUI implements adapter |
| TUI adapter | `tui::runtime::session_command_adapter` | Maps to private `RunnerCommand` |
| `SessionEvent` | `session::event` | BE → FE projection |
| `AgentRunner` / `RunnerEvent` | `session::runner` | Turn bridge; TUI re-exports shim |
| `SessionEventSink` | `session::ports` | UI-agnostic emit port |

## Ownership after Phase B/C

| Component | Location | Notes |
| --- | --- | --- |
| TUI runtime | `tui::runtime` | View orchestration; command send path uses handler |
| Line CLI | `main.rs` | Uses `from_command_intent` for backend-owned ops |
| Private `RunnerCommand` | `tui::runtime` | Still TUI-private (child anchors, test inspect) |

## Remaining migration (Phase D+)

1. Optionally promote selected runner-only events into `SessionEvent` when a second frontend needs them.
2. Grow a session-owned coordinator that absorbs more of `RunnerCommand` execution.
3. Keep `src/session` free of `crate::tui` / ratatui forever.
