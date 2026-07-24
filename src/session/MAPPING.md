# Session Event Contract Mapping (Phase A)

`session::SessionEvent` is the frontend-neutral outbound event contract. It owns
all event payloads and has no dependency on `crate::tui`.

## Compatibility mapping

| Legacy TUI symbol | Phase A contract | Direction |
| --- | --- | --- |
| `tui::events::AppEvent` | `session::SessionEvent` | Type alias (`pub use ... as AppEvent`) |
| `tui::events::<Payload>` | `session::<Payload>` | Re-export |
| `tui::runner::RunnerEvent` | `session::RunnerEvent` | Re-export shim; runtime compatibility bridge |

## Runner event mapping

`session::RunnerEvent` remains the richer runner-to-frontend transport during
migration. `session::SessionEvent` is the frontend-neutral projection where a
one-to-one event exists.

| `RunnerEvent` variant | `SessionEvent` mapping | Notes |
| --- | --- | --- |
| `UserMessage`, `ReasoningDelta`, `ReasoningDone`, `AssistantDelta`, `AssistantDone` | Same-named variant | Direct payload mapping. |
| `TokenUsage`, `ToolPending`, `ToolCancelled`, `ToolStarted`, `ToolFinished`, `ToolOutputDelta`, `TodoSnapshot`, `AutoContinueChanged` | Same-named variant | Direct payload mapping. |
| `PermissionResolved`, `ProcessIssue`, `Notice`, compaction, runtime-context, and context-projection events | Same-named variant | Direct payload mapping. |
| `Interrupted`, `Error`, `Done`, `Quit` | Same-named variant | Terminal/control mapping. |
| `SessionTokenUsage`, queued prompts, permission/question handles, child events, MCP catalog events, session/branch lifecycle events | No current `SessionEvent` variant | Runner transport remains backend-owned until the session port expands. |

The aliases in `src/tui/events.rs` preserve existing imports in the TUI. New
backend code must use `crate::session::{SessionEvent, ...}` directly.

## Ownership after Phase B

| Component | Location | Notes |
| --- | --- | --- |
| `SessionCommand` | `session::command` | FE → BE |
| `SessionEvent` | `session::event` | BE → FE projection |
| `AgentRunner` / `RunnerEvent` | `session::runner` | Turn bridge; TUI re-exports shim |
| `SessionEventSink` | `session::ports` | UI-agnostic emit port |
| TUI runtime | `tui::runtime` | View orchestration; must not re-own agent policy long-term |

## Remaining migration (Phase C+)

1. Expand `SessionEvent` for runner-only variants still listed above, or document permanent runner-channel extras for multi-frontend use.
2. Route `TuiRuntime` command handling through a session command handler.
3. Map line CLI `CommandIntent` → `SessionCommand` for shared ops.
4. Keep `src/session` free of `crate::tui` / ratatui forever.
