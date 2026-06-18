use crate::agent::{AutoContinueState, TodoItem};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AppEvent {
    Tick,
    UserMessage(UserMessageEvent),
    ReasoningDelta(ReasoningDeltaEvent),
    ReasoningDone(ReasoningDoneEvent),
    AssistantDelta(AssistantDeltaEvent),
    AssistantDone { message_id: Option<String> },
    TokenUsage(TokenUsageEvent),
    ToolPending(ToolPendingEvent),
    ToolStarted(ToolStartedEvent),
    ToolFinished(ToolFinishedEvent),
    TodoSnapshot(TodoSnapshotEvent),
    AutoContinueChanged(AutoContinueChangedEvent),
    PermissionRequested(PermissionRequestEvent),
    PermissionResolved(PermissionResolutionEvent),
    Interrupted,
    Error(ErrorEvent),
    Done,
    Quit,
}

impl AppEvent {
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Done | Self::Quit | Self::Interrupted)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolPendingEvent {
    pub call_id: String,
    pub name: String,
}

impl ToolPendingEvent {
    pub fn new(call_id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            call_id: call_id.into(),
            name: name.into(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenUsageEvent {
    pub used_tokens: u64,
    pub context_window_tokens: u64,
}

impl TokenUsageEvent {
    pub fn new(used_tokens: u64, context_window_tokens: u64) -> Self {
        Self {
            used_tokens,
            context_window_tokens,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UserMessageEvent {
    pub message_id: Option<String>,
    pub content: String,
}

impl UserMessageEvent {
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            message_id: None,
            content: content.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReasoningDeltaEvent {
    pub item_id: String,
    pub delta: String,
}

impl ReasoningDeltaEvent {
    pub fn new(item_id: impl Into<String>, delta: impl Into<String>) -> Self {
        Self {
            item_id: item_id.into(),
            delta: delta.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReasoningDoneEvent {
    pub item_id: String,
    pub text: String,
}

impl ReasoningDoneEvent {
    pub fn new(item_id: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            item_id: item_id.into(),
            text: text.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssistantDeltaEvent {
    pub message_id: Option<String>,
    pub delta: String,
}

impl AssistantDeltaEvent {
    pub fn new(delta: impl Into<String>) -> Self {
        Self {
            message_id: None,
            delta: delta.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolStartedEvent {
    pub call_id: String,
    pub name: String,
    pub summary: String,
    pub arguments: Option<String>,
}

impl ToolStartedEvent {
    pub fn new(
        call_id: impl Into<String>,
        name: impl Into<String>,
        summary: impl Into<String>,
    ) -> Self {
        Self {
            call_id: call_id.into(),
            name: name.into(),
            summary: summary.into(),
            arguments: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolFinishedEvent {
    pub call_id: String,
    pub name: String,
    pub summary: String,
    pub outcome: ToolOutcome,
    pub output: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TodoSnapshotEvent {
    pub items: Vec<TodoItem>,
}

impl TodoSnapshotEvent {
    pub fn new(items: Vec<TodoItem>) -> Self {
        Self { items }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AutoContinueChangedEvent {
    pub state: AutoContinueState,
}

impl AutoContinueChangedEvent {
    pub fn new(state: AutoContinueState) -> Self {
        Self { state }
    }
}

impl ToolFinishedEvent {
    pub fn new(
        call_id: impl Into<String>,
        name: impl Into<String>,
        summary: impl Into<String>,
        outcome: ToolOutcome,
    ) -> Self {
        Self {
            call_id: call_id.into(),
            name: name.into(),
            summary: summary.into(),
            outcome,
            output: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolOutcome {
    Success,
    Failure,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PermissionRequestEvent {
    pub call_id: String,
    pub tool_name: String,
    pub summary: String,
    pub arguments: Option<String>,
    pub rationale: Option<String>,
    pub origin_label: Option<String>,
}

impl PermissionRequestEvent {
    pub fn new(
        call_id: impl Into<String>,
        tool_name: impl Into<String>,
        summary: impl Into<String>,
    ) -> Self {
        Self {
            call_id: call_id.into(),
            tool_name: tool_name.into(),
            summary: summary.into(),
            arguments: None,
            rationale: None,
            origin_label: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PermissionResolutionEvent {
    pub call_id: String,
    pub decision: PermissionDecision,
    pub reason: Option<String>,
}

impl PermissionResolutionEvent {
    pub fn approved(call_id: impl Into<String>) -> Self {
        Self {
            call_id: call_id.into(),
            decision: PermissionDecision::Approved,
            reason: None,
        }
    }

    pub fn denied(call_id: impl Into<String>, reason: Option<String>) -> Self {
        Self {
            call_id: call_id.into(),
            decision: PermissionDecision::Denied,
            reason,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PermissionDecision {
    Approved,
    Denied,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ErrorEvent {
    pub message: String,
    pub details: Option<String>,
}

impl ErrorEvent {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            details: None,
        }
    }
}
