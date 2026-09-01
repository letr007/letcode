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
//! ([`SessionCommand`]) and outbound events ([`SessionEvent`]). Its engine egress
//! additionally carries [`SessionTransportEvent`] for session-local interaction
//! handles and raw restore projections that do not belong in `SessionEvent`.
//!
//! Turn execution is shared: every frontend goes through
//! [`SessionEngine`] → [`AgentRunner`]. Frontends only submit [`SessionCommand`]
//! and present [`SessionTransportEvent`]s.
//!
//! ```text
//!   TUI ──┐
//!         ├──SessionCommand──► SessionEngine ──► AgentRunner ─► agent/tools/transcript
//!   CLI ──┘
//! ```

pub mod auto_review;
pub mod child_view;
pub mod command;
pub mod context_scope;
pub mod coordinator;
pub mod engine;
pub mod event;
pub mod interrupt;
pub mod lifecycle;
pub mod ports;
pub mod restore;
pub mod runner;
pub mod settings;

#[cfg(test)]
pub(crate) use child_view::project_parent_session_view;
pub use child_view::{current_session_records, list_child_sessions_for_view};
pub(crate) use command::ActiveTurnCommandDisposition;
pub use command::SessionCommand;
pub use context_scope::sync_agent_context_scope_from_recorder;
pub(crate) use coordinator::IdleDispatch;
pub use coordinator::SessionCoordinator;
pub use engine::{
    SessionEngine, SessionEngineConfig, SessionEngineIngress, SessionEngineProjection,
};
pub use event::{
    AssistantDeltaEvent, AutoContinueChangedEvent, ContextDetailOpenedEvent,
    ContextSummaryUpdatedEvent, ContextTreeUpdatedEvent, ContextViewUpdatedEvent, ErrorEvent,
    NoticeEvent, NoticeKind, PermissionDecision, PermissionRequestEvent, PermissionResolutionEvent,
    ProcessIssueEvent, ReasoningDeltaEvent, ReasoningDoneEvent, RetryLifecycleEvent,
    RuntimeContextDisposition, RuntimeContextUpdatedEvent, SessionEvent, TodoSnapshotEvent,
    TokenUsageEvent, ToolCancelledEvent, ToolFinishedEvent, ToolOutcome, ToolOutputDeltaEvent,
    ToolPendingEvent, ToolStartedEvent, UserMessageEvent,
};

pub use lifecycle::{prepare_new_session_package, resolve_session_prefix};
pub use ports::SessionCommandHandler;
#[cfg(test)]
pub(crate) use restore::install_prepared_routed_resume_for_agent;
pub use restore::{prepare_resume_package, project_runtime_restore_snapshot_with_children};
#[cfg(test)]
pub(crate) use runner::PermissionResponse;
pub(crate) use runner::{
    AgentRunner, RunnerPermissionRequest, RunnerQuestionRequest, SessionTransportEvent,
    subagent_event_sender,
};
