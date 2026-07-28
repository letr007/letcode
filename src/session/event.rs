//! Frontend-neutral outbound events produced by a session.
//!
//! This contract is independent of frontend modules. Compatibility aliases
//! remain at the frontend boundary during the migration.

use crate::agent::{AutoContinueState, CacheUsageReport, ConversationMessage, TodoItem};
use crate::context_tree::ContextTreeState;
use crate::context_view::{ContextViewProjection, SummaryArtifact};
use crate::runtime_context::RuntimeActiveContext;
use crate::user_content::{UserImageAttachment, UserMessageContent, UserMessageSubmission};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionEvent {
    Tick,
    UserMessage(UserMessageEvent),
    ReasoningDelta(ReasoningDeltaEvent),
    ReasoningDone(ReasoningDoneEvent),
    AssistantDelta(AssistantDeltaEvent),
    AssistantDone {
        message_id: Option<String>,
    },
    TokenUsage(TokenUsageEvent),
    ToolPending(ToolPendingEvent),
    ToolCancelled(ToolCancelledEvent),
    ToolStarted(ToolStartedEvent),
    ToolFinished(ToolFinishedEvent),
    ToolOutputDelta(ToolOutputDeltaEvent),
    TodoSnapshot(TodoSnapshotEvent),
    AutoContinueChanged(AutoContinueChangedEvent),
    PermissionRequested(PermissionRequestEvent),
    PermissionResolved(PermissionResolutionEvent),
    ProcessIssue(ProcessIssueEvent),
    Notice(NoticeEvent),
    CompactionStarted,
    CompactionPreviewDelta {
        delta: String,
    },
    /// Durable commit. `summary` is the authoritative text when known (journal).
    CompactionCommitted {
        summary: Option<String>,
    },
    CompactionNoProgress {
        blockers: Vec<String>,
    },
    CompactionFailed,
    RuntimeContextUpdated(RuntimeContextUpdatedEvent),
    ContextTreeUpdated(ContextTreeUpdatedEvent),
    ContextViewUpdated(ContextViewUpdatedEvent),
    ContextDetailOpened(ContextDetailOpenedEvent),
    ContextSummaryUpdated(ContextSummaryUpdatedEvent),
    /// Session identity became active (new or switched).
    ///
    /// Full transcript records remain on [`crate::session::RunnerEvent`] for the
    /// TUI restore path until SessionEngine owns journal loading end-to-end.
    SessionStarted {
        session_id: String,
        runtime_context: RuntimeActiveContext,
    },
    /// Session restored from transcript with history and scope projection.
    SessionResumed {
        session_id: String,
        branch_id: String,
        messages: Vec<ConversationMessage>,
        evidence_count: usize,
        model_id: Option<String>,
        token_usage: Option<TokenUsageEvent>,
        runtime_context: RuntimeActiveContext,
    },
    /// Aggregate session token usage (distinct from per-turn [`TokenUsage`]).
    SessionTokenUsage(TokenUsageEvent),
    /// Active context branch changed.
    ContextBranchChanged {
        branch_id: String,
    },
    /// Context branch listing for tree/branch UIs.
    /// Tool batch boundary for a turn (pure signal, no payload).
    ToolBatchFinished,
    Interrupted,
    Error(ErrorEvent),
    Done,
    Quit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeContextDisposition {
    Advance,
    ReplaceScope,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeContextUpdatedEvent {
    pub context: RuntimeActiveContext,
    pub disposition: RuntimeContextDisposition,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextTreeUpdatedEvent {
    pub tree: ContextTreeState,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextViewUpdatedEvent {
    pub projection: ContextViewProjection,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextDetailOpenedEvent {
    pub open_detail_block_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextSummaryUpdatedEvent {
    pub summaries: Vec<SummaryArtifact>,
}

impl SessionEvent {
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCancelledEvent {
    pub call_id: String,
    pub name: String,
}

impl ToolCancelledEvent {
    pub fn new(call_id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            call_id: call_id.into(),
            name: name.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenUsageEvent {
    pub used_tokens: u64,
    pub context_window_tokens: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cached_tokens: u64,
    pub cache_report: Option<CacheUsageReport>,
}

impl TokenUsageEvent {
    pub fn new(used_tokens: u64, context_window_tokens: u64) -> Self {
        Self::with_breakdown(used_tokens, context_window_tokens, used_tokens, 0, 0)
    }

    pub fn with_breakdown(
        used_tokens: u64,
        context_window_tokens: u64,
        input_tokens: u64,
        output_tokens: u64,
        cached_tokens: u64,
    ) -> Self {
        Self {
            used_tokens,
            context_window_tokens,
            input_tokens,
            output_tokens,
            cached_tokens,
            cache_report: None,
        }
    }

    pub fn with_cache_report(mut self, cache_report: Option<CacheUsageReport>) -> Self {
        self.cache_report = cache_report;
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UserMessageEvent {
    pub message_id: Option<String>,
    pub submission_id: String,
    pub content: UserMessageContent,
    pub queued: bool,
}

impl UserMessageEvent {
    pub fn new(content: impl Into<String>) -> Self {
        Self::from_submission(UserMessageSubmission::new(
            "live-user-message",
            UserMessageContent::new(content, Vec::new()),
        ))
    }

    pub fn queued(content: impl Into<String>) -> Self {
        Self::queued_submission(UserMessageSubmission::new(
            "queued-user-message",
            UserMessageContent::new(content, Vec::new()),
        ))
    }

    pub fn from_submission(submission: UserMessageSubmission) -> Self {
        Self {
            message_id: None,
            submission_id: submission.id,
            content: submission.content,
            queued: false,
        }
    }

    pub fn queued_submission(submission: UserMessageSubmission) -> Self {
        Self {
            message_id: None,
            submission_id: submission.id,
            content: submission.content,
            queued: true,
        }
    }

    pub fn text(&self) -> &str {
        &self.content.text
    }

    pub fn attachments(&self) -> &[UserImageAttachment] {
        &self.content.attachments
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

    pub fn with_message_id(message_id: impl Into<String>, delta: impl Into<String>) -> Self {
        Self {
            message_id: Some(message_id.into()),
            delta: delta.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NoticeEvent {
    pub message: String,
    pub kind: NoticeKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoticeKind {
    Info,
    Success,
    RecoverableError,
}

impl NoticeEvent {
    pub fn new(message: impl Into<String>, kind: NoticeKind) -> Self {
        Self {
            message: message.into(),
            kind,
        }
    }

    pub fn info(message: impl Into<String>) -> Self {
        Self::new(message, NoticeKind::Info)
    }

    pub fn success(message: impl Into<String>) -> Self {
        Self::new(message, NoticeKind::Success)
    }

    pub fn recoverable_error(message: impl Into<String>) -> Self {
        Self::new(message, NoticeKind::RecoverableError)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcessIssueEvent {
    pub message: String,
    pub detail: Option<String>,
    pub action: Option<String>,
}

impl ProcessIssueEvent {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            detail: None,
            action: None,
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
pub struct ToolOutputDeltaEvent {
    pub call_id: String,
    pub stream: crate::tool::ToolOutputStream,
    pub chunk: String,
}

impl ToolOutputDeltaEvent {
    pub fn new(
        call_id: impl Into<String>,
        stream: crate::tool::ToolOutputStream,
        chunk: impl Into<String>,
    ) -> Self {
        Self {
            call_id: call_id.into(),
            stream,
            chunk: chunk.into(),
        }
    }
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
    pub can_allow_always: bool,
    pub grant_summary: Option<String>,
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
            can_allow_always: false,
            grant_summary: None,
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
