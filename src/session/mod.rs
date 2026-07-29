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

pub mod child_view;
pub mod command;
pub mod context_scope;
pub mod coordinator;
pub mod event;
pub mod lifecycle;
pub mod ports;
pub mod restore;
pub mod runner;
pub mod settings;

pub use child_view::{
    ChildViewProjection, ParentViewProjection, current_session_records,
    list_child_sessions_for_view, project_child_session_view, project_parent_session_view,
    select_child_navigation_index, sessions_dir_from_transcript,
};
pub use command::SessionCommand;
pub use context_scope::{
    PreparedContextScope, apply_prepared_context_scope, prepare_context_scope,
    sync_agent_context_scope_from_recorder,
};
pub use coordinator::{CommandOwnership, IdleDispatch, SessionCoordinator};
pub use event::{
    AssistantDeltaEvent, AutoContinueChangedEvent, ContextDetailOpenedEvent,
    ContextSummaryUpdatedEvent, ContextTreeUpdatedEvent, ContextViewUpdatedEvent, ErrorEvent,
    NoticeEvent, NoticeKind, PermissionDecision, PermissionRequestEvent, PermissionResolutionEvent,
    ProcessIssueEvent, ReasoningDeltaEvent, ReasoningDoneEvent, RetryLifecycleEvent,
    RuntimeContextDisposition, RuntimeContextUpdatedEvent, SessionEvent, TodoSnapshotEvent,
    TokenUsageEvent, ToolCancelledEvent, ToolFinishedEvent, ToolOutcome, ToolOutputDeltaEvent,
    ToolPendingEvent, ToolStartedEvent, UserMessageEvent,
};
pub use lifecycle::{
    PreparedNewSession, ResolveSessionError, bootstrap_new_transcript, cleanup_empty_session_file,
    cleanup_replaced_empty_session, install_new_session_for_agent,
    install_prepared_new_session_for_agent, load_session_records, open_resume_transcript,
    prepare_new_session_package, replace_live_transcript, resolve_session_prefix,
    session_started_event, start_new_transcript_session,
};
pub use ports::{SessionCommandHandler, SessionEventSink, SessionPorts};
pub use restore::{
    PreparedResume, default_resume_cursor, install_prepared_resume_for_agent,
    prepare_resume_package, project_runtime_restore_snapshot_with_children,
    restored_messages_from_protocol_frames, restored_session_token_usage, session_resumed_event,
};
pub use runner::{
    AgentRunner, PermissionResponse, RunnerEvent, RunnerEventSender, RunnerPermissionRequest,
    RunnerQuestionRequest, subagent_event_sender,
};
pub use settings::{apply_model, apply_permission_mode, apply_reasoning_effort};
