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
pub mod coordinator;
pub mod event;
pub mod ports;
pub mod runner;
pub mod settings;

pub use branch_query::{
    format_branch_listing, format_branch_listing_multiline, load_context_branches,
};
pub use command::SessionCommand;
pub use coordinator::{IdleDispatch, SessionCoordinator};
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
