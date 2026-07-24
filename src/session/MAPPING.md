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
| `SessionTokenUsage` | session lifecycle metrics | **Projected** to `SessionEvent::SessionTokenUsage` |
| `SessionStarted`, `SessionResumed` | session lifecycle | **Projected** (without raw transcript records) |
| `ContextBranchChanged`, `ContextBranchesLoaded` | branch UI | **Projected** to SessionEvent |
| `ToolBatchFinished` | turn batching signal | **Projected** to SessionEvent |
| `QueuedPromptAccepted` | TUI queue handshake | TUI-private transport for now |
| `QuestionRequested`, `ChildQuestionRequested` | interactive handles | Runner transport (oneshot handles) |
| `ChildPermissionRequested`, `ChildAppEvent`, `ChildSessionViewed` | subagent/child view | Runner + TUI child surface |
| `McpToolsDiscovered`, `McpServerUpdating`, `McpServerUpdated`, `McpServerToolsUpdated`, `McpDiscoveryUnavailable`, `McpDiagnostic` | MCP catalog UI | Runner extra |

The aliases in `src/tui/events.rs` preserve existing imports in the TUI. New
backend code must use `crate::session::{SessionEvent, ...}` directly.

## Command path (Phase C)

```text
CommandIntent
  ├─ SessionCommand::from_command_intent → Some(SessionCommand)
  │     CLI:  → ReplCommand (execution or Unsupported)
  │     TUI:  → handle_backend_session_command (local side effects / RuntimeCommand)
  │              → SessionCommandHandler adapter
  │              → RunnerControl { Command(RunnerCommand) | Interrupt }
  └─ None (presentation-only) → frontend-local handling
```

Phase D: both CLI (`parse_repl_command`) and TUI (`handle_parsed_command`) use
`SessionCommand::from_command_intent` as the shared backend vs local classifier.

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

## Phase F session branch queries

`session::branch_query` owns context-branch load + listing format helpers:

| Helper | Consumers |
| --- | --- |
| `load_context_branches` | TUI runner arms (`ShowBranchTree` / `ListBranches`), line CLI |
| `format_branch_listing` | TUI notice path + branch dialog re-export |
| `format_branch_listing_multiline` | Line CLI `/tree` and `/branches` |

## Phase G session settings

`session::settings` owns idle agent setting mutations with transcript provenance:

| Helper | Consumers |
| --- | --- |
| `apply_permission_mode` | TUI `SetPermissionMode`, CLI `/permission` |
| `apply_model` | TUI `SetModel`, CLI `/model` (after catalog check) |
| `apply_reasoning_effort` | TUI `SetReasoningEffort`, CLI `/reasoning` |

## Phase H SessionCoordinator (idle dispatch)

`session::SessionCoordinator::dispatch_idle_command` is the session-owned entry
for idle commands that emit `RunnerEvent`s and/or mutate agent state without
starting a turn:

- `ShowBranchTree`, `ListBranches`
- `SetPermissionMode`, `SetModel`, `SetReasoningEffort`

TUI runner arms delegate these to the coordinator. Turn-bearing commands
(`SubmitPrompt`, `Compact`, `Delegate`, resume/new, child nav, MCP, interrupt)
still return `IdleDispatch::NotIdle` and remain TUI-loop hosted.

## Phase I session lifecycle bootstrap

`session::lifecycle` owns new-session transcript create/start/swap helpers:

| Helper | Consumers |
| --- | --- |
| `bootstrap_new_transcript` | CLI `start_new_session`, TUI `NewSession` arm |
| `replace_live_transcript` / `cleanup_empty_session_file` | CLI new-session swap |

## Phase K session resume resolve/open

| Helper | Consumers |
| --- | --- |
| `resolve_session_prefix` | CLI `/resume`, TUI `ResumeSession` |
| `load_session_records` | CLI/TUI resume load |
| `open_resume_transcript` | CLI/TUI open existing |
| `cleanup_replaced_empty_session` | CLI resume swap cleanup |

## Phase L restore projection

`session::restore::project_runtime_restore_snapshot_with_children` is shared by
CLI resume and TUI restore paths. Agent restore + timeline mapping stay frontend.

Coordinator still marks `ResumeSession` as `FrontendHosted` (orchestration).

Context-scope prepare/apply remain frontend-owned for now.

CLI residual: `@delegate`, `/child`, `/parent`, MCP toggle, interrupt still Unsupported.

## Phase J coordinator ownership table

`SessionCoordinator::ownership` classifies every `SessionCommand` as
`IdleCoordinator` or `FrontendHosted`. Adding a command variant must update this
table (compile-driven match) and the idle dispatch path.

## Remaining migration (post-pipeline)

1. Optionally promote selected runner-only lifecycle/branch events into `SessionEvent` when a second frontend needs them (requires public payloads for branch listings / restore messages).
2. Expand `SessionCoordinator` to absorb turn-bearing commands (`SubmitPrompt`, `Compact`, resume/new, child nav) and eventually the async control loop.
3. Keep `src/session` free of `crate::tui` / ratatui forever.


## Phase E projection

`RunnerEvent::session_event()` (alias of the expanded `app_event` mapping)
projects pure lifecycle/branch events into `SessionEvent` for multi-frontend use.
TUI restore still consumes richer `RunnerEvent` payloads (transcript records).
