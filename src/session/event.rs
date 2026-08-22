//! Frontend-neutral outbound events produced by a session.
//!
//! This contract is independent of frontend modules. Compatibility aliases
//! remain at the frontend boundary during the migration.

use crate::agent::{
    AutoContinueState, CacheUsageReport, ConversationMessage, PromptCompositionEntry, TodoItem,
};
use crate::context_tree::ContextTreeState;
use crate::context_view::{ContextViewProjection, SummaryArtifact};
use crate::runtime_context::RuntimeActiveContext;
use crate::user_content::{UserMessageContent, UserMessageSubmission};
use std::time::Instant;

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
    /// Full transcript records remain on [`crate::session::SessionTransportEvent`] for the
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
    RetryScheduled(RetryLifecycleEvent),
    RetryStarted(RetryLifecycleEvent),
    Interrupted,
    Error(ErrorEvent),
    Done,
    Quit,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetryLifecycleEvent {
    pub attempt: usize,
    pub max_attempts: usize,
    pub delay_secs: u64,
    pub error: String,
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
    pub prompt_composition: Vec<PromptCompositionEntry>,
}

impl TokenUsageEvent {
    #[cfg(test)]
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
            prompt_composition: Vec::new(),
        }
    }

    pub fn with_cache_report(mut self, cache_report: Option<CacheUsageReport>) -> Self {
        self.cache_report = cache_report;
        self
    }

    pub fn with_prompt_composition(
        mut self,
        prompt_composition: Vec<PromptCompositionEntry>,
    ) -> Self {
        self.prompt_composition = prompt_composition;
        self
    }

    pub fn merge_prompt_composition_from(&mut self, previous: &Self) {
        if self.prompt_composition.is_empty() {
            self.prompt_composition = previous.prompt_composition.clone();
        }
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
    #[cfg(test)]
    pub fn new(content: impl Into<String>) -> Self {
        Self::from_submission(UserMessageSubmission::new(
            "live-user-message",
            UserMessageContent::new(content, Vec::new()),
        ))
    }

    #[cfg(test)]
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
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReasoningDeltaEvent {
    pub item_id: String,
    pub delta: String,
    pub observed_at: Instant,
}

impl ReasoningDeltaEvent {
    pub fn new(item_id: impl Into<String>, delta: impl Into<String>) -> Self {
        Self {
            item_id: item_id.into(),
            delta: delta.into(),
            observed_at: Instant::now(),
        }
    }

    #[cfg(test)]
    pub fn at(item_id: impl Into<String>, delta: impl Into<String>, observed_at: Instant) -> Self {
        Self {
            item_id: item_id.into(),
            delta: delta.into(),
            observed_at,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReasoningDoneEvent {
    pub item_id: String,
    pub text: String,
    pub observed_at: Instant,
}

impl ReasoningDoneEvent {
    pub fn new(item_id: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            item_id: item_id.into(),
            text: text.into(),
            observed_at: Instant::now(),
        }
    }

    #[cfg(test)]
    pub fn at(item_id: impl Into<String>, text: impl Into<String>, observed_at: Instant) -> Self {
        Self {
            item_id: item_id.into(),
            text: text.into(),
            observed_at,
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
    #[allow(dead_code)]
    // Retained for frontend event compatibility; current production notices use Info.
    Success,
    #[allow(dead_code)]
    // Retained for frontend event compatibility; current production notices use Info.
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
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcessIssueEvent {
    pub message: String,
    pub detail: Option<String>,
    pub action: Option<String>,
}

impl ProcessIssueEvent {
    #[cfg(test)]
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
    #[cfg(test)]
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
    #[cfg(test)]
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
    /// Present when resolution was not preceded by a PermissionRequested UI prompt
    /// (e.g. Auto-mode reviewer). Used to build a timeline card.
    pub tool_name: Option<String>,
    pub summary: Option<String>,
    /// Fixed identity label for auto-review cards (`"reviewer"`).
    pub origin_label: Option<String>,
    pub approval: Option<String>,
    pub risk: Option<String>,
    pub reviewer_child_session_id: Option<String>,
}

impl PermissionResolutionEvent {
    #[cfg(test)]
    pub fn approved(call_id: impl Into<String>) -> Self {
        Self {
            call_id: call_id.into(),
            decision: PermissionDecision::Approved,
            reason: None,
            tool_name: None,
            summary: None,
            origin_label: None,
            approval: None,
            risk: None,
            reviewer_child_session_id: None,
        }
    }

    pub fn denied(call_id: impl Into<String>, reason: Option<String>) -> Self {
        Self {
            call_id: call_id.into(),
            decision: PermissionDecision::Denied,
            reason,
            tool_name: None,
            summary: None,
            origin_label: None,
            approval: None,
            risk: None,
            reviewer_child_session_id: None,
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
