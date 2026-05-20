#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AppEvent {
    Tick,
    UserMessage(UserMessageEvent),
    AssistantDelta(AssistantDeltaEvent),
    AssistantDone { message_id: Option<String> },
    ToolStarted(ToolStartedEvent),
    ToolFinished(ToolFinishedEvent),
    PermissionRequested(PermissionRequestEvent),
    PermissionResolved(PermissionResolutionEvent),
    Error(ErrorEvent),
    Done,
    Quit,
}

impl AppEvent {
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Done | Self::Quit)
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
