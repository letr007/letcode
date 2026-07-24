//! Session backend boundary for all frontends (TUI, line CLI, future GUI).
//!
//! ## Split intent
//!
//! - **Backend (`session`)**: turn execution, tools, permissions, subagents,
//!   transcript/session lifecycle. Frontends submit commands and observe
//!   results/events; they do not own agent internals.
//! - **Frontend (`tui`, CLI, GUI)**: presentation, input, layout, and local
//!   view-model state. Frontends may full-redraw; they should not re-implement
//!   session policy.
//!
//! ## Current surface
//!
//! The session boundary exposes frontend-neutral inbound commands
//! ([`SessionCommand`]) and outbound events ([`SessionEvent`]). The TUI retains
//! compatibility aliases while its runner and runtime migrate incrementally.
//!
//! ```text
//!   TUI / CLI / GUI
//!        │ SessionCommand
//!        ▼
//!   session (engine boundary)
//!        │ agent / tools / transcript / subagent pool
//!        ▼
//!   model providers + local workspace
//! ```

pub mod branch_query;
pub mod command;
pub mod context_scope;
pub mod coordinator;
pub mod event;
pub mod lifecycle;
pub mod ports;
pub mod restore;
pub mod runner;
pub mod settings;

pub use branch_query::{
    format_branch_listing, format_branch_listing_multiline, load_context_branches,
};
pub use command::SessionCommand;
pub use coordinator::{CommandOwnership, IdleDispatch, SessionCoordinator};
pub use lifecycle::{
    PreparedNewSession, ResolveSessionError, apply_prepared_new_session_to_agent,
    bootstrap_new_transcript, cleanup_empty_session_file, cleanup_replaced_empty_session,
    install_new_session_for_agent, load_session_records, open_resume_transcript,
    prepare_new_session_package, replace_live_transcript, resolve_session_prefix,
    start_new_transcript_session,
};
pub use context_scope::{
    PreparedContextScope, apply_prepared_context_scope, prepare_context_scope,
    sync_agent_context_scope_from_recorder,
};
pub use restore::{
    PreparedResume, apply_prepared_resume_to_agent, default_resume_cursor, prepare_resume_package,
    project_runtime_restore_snapshot_with_children,
};
pub use settings::{apply_model, apply_permission_mode, apply_reasoning_effort};
pub use ports::{SessionCommandHandler, SessionEventSink, SessionPorts};
pub use runner::{
    AgentRunner, PermissionResponse, RunnerEvent, RunnerEventSender, RunnerPermissionRequest,
    RunnerQuestionRequest, subagent_event_sender,
};
pub use event::{
    AssistantDeltaEvent, AutoContinueChangedEvent, ContextDetailOpenedEvent,
    ContextSummaryUpdatedEvent, ContextTreeUpdatedEvent, ContextViewUpdatedEvent, ErrorEvent,
    FoldedOutputsUpdatedEvent, NoticeEvent, NoticeKind, PermissionDecision, PermissionRequestEvent,
    PermissionResolutionEvent, ProcessIssueEvent, ReasoningDeltaEvent, ReasoningDoneEvent,
    RuntimeContextDisposition, RuntimeContextUpdatedEvent, SessionEvent, TodoSnapshotEvent,
    TokenUsageEvent, ToolCancelledEvent, ToolFinishedEvent, ToolOutcome, ToolOutputDeltaEvent,
    ToolPendingEvent, ToolStartedEvent, UserMessageEvent,
};
