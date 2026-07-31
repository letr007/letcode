//! TUI compatibility aliases for the session event contract.
//!
//! New backend code must import from `crate::session`; these aliases preserve
//! existing TUI event names while migration proceeds.

pub use crate::session::{
    AssistantDeltaEvent, AutoContinueChangedEvent, ErrorEvent, NoticeKind, PermissionDecision,
    PermissionRequestEvent, PermissionResolutionEvent, ReasoningDeltaEvent, ReasoningDoneEvent,
    RuntimeContextDisposition, RuntimeContextUpdatedEvent, SessionEvent, TodoSnapshotEvent,
    TokenUsageEvent, ToolFinishedEvent, ToolOutcome, ToolOutputDeltaEvent, ToolPendingEvent,
    ToolStartedEvent, UserMessageEvent,
};

// 仅测试模块引用的历史别名。
#[cfg(test)]
pub use crate::session::{
    ContextTreeUpdatedEvent, ContextViewUpdatedEvent, NoticeEvent, ProcessIssueEvent,
    RetryLifecycleEvent, ToolCancelledEvent,
};
