//! TUI compatibility aliases for the session event contract.
//!
//! New backend code must import from `crate::session`; these aliases preserve
//! existing TUI event names while migration proceeds.

pub use crate::session::{
    AssistantDeltaEvent, AutoContinueChangedEvent, ContextDetailOpenedEvent,
    ContextSummaryUpdatedEvent, ContextTreeUpdatedEvent, ContextViewUpdatedEvent, ErrorEvent,
    FoldedOutputsUpdatedEvent, NoticeEvent, NoticeKind, PermissionDecision, PermissionRequestEvent,
    PermissionResolutionEvent, ProcessIssueEvent, ReasoningDeltaEvent, ReasoningDoneEvent,
    RuntimeContextDisposition, RuntimeContextUpdatedEvent, SessionEvent as AppEvent,
    TodoSnapshotEvent, TokenUsageEvent, ToolCancelledEvent, ToolFinishedEvent, ToolOutcome,
    ToolOutputDeltaEvent, ToolPendingEvent, ToolStartedEvent, UserMessageEvent,
};
