use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Result};
use serde_json::Value;
use tokio::sync::{mpsc, oneshot};

use crate::agent::{ConversationMessage, LlmRetryLifecycle};
use crate::permission::PermissionRequest;
use crate::request_builder::ModelReasoningEffort;
use crate::runtime_context::RuntimeActiveContext;
use crate::user_content::UserMessageSubmission;
use crate::tool::{QuestionRequest, QuestionResponse, ToolResult};
use crate::tool_format::format_tool_call;
use crate::transcript::{read_records, TranscriptRecord, TranscriptRecorder};

use crate::session::{
    AssistantDeltaEvent, AutoContinueChangedEvent, ContextDetailOpenedEvent,
    ContextSummaryUpdatedEvent, ContextTreeUpdatedEvent, ContextViewUpdatedEvent, ErrorEvent,
    NoticeEvent, PermissionRequestEvent, PermissionResolutionEvent, ProcessIssueEvent,
    ReasoningDeltaEvent, ReasoningDoneEvent, RetryLifecycleEvent, RuntimeContextDisposition,
    RuntimeContextUpdatedEvent, SessionEvent, TodoSnapshotEvent, TokenUsageEvent,
    ToolCancelledEvent, ToolFinishedEvent, ToolOutcome, ToolOutputDeltaEvent, ToolPendingEvent,
    ToolStartedEvent, UserMessageEvent,
};

use super::formatting::{output_json, output_summary};

pub(crate) type SessionTransportEventSender = mpsc::UnboundedSender<SessionTransportEvent>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SessionTransportEventMode {
    Emit,
    SilentDenyPermissions,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PermissionResponse {
    AllowOnce,
    AllowAlways,
    Deny,
}

impl PermissionResponse {
    pub fn allowed(self) -> bool {
        !matches!(self, Self::Deny)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct RunnerPermissionRequest {
    sender: Arc<Mutex<Option<oneshot::Sender<PermissionResponse>>>>,
}

#[derive(Debug, Clone)]
pub(crate) struct RunnerQuestionRequest {
    sender: Arc<Mutex<Option<oneshot::Sender<std::result::Result<QuestionResponse, String>>>>>,
}

impl RunnerQuestionRequest {
    pub fn new(sender: oneshot::Sender<std::result::Result<QuestionResponse, String>>) -> Self {
        Self {
            sender: Arc::new(Mutex::new(Some(sender))),
        }
    }

    fn respond(&self, response: std::result::Result<QuestionResponse, String>) -> Result<()> {
        let sender = self
            .sender
            .lock()
            .map_err(|_| anyhow!("question request lock poisoned"))?
            .take()
            .ok_or_else(|| anyhow!("question request already resolved"))?;

        sender
            .send(response)
            .map_err(|_| anyhow!("question response receiver dropped"))
    }

    pub fn answer(&self, response: QuestionResponse) -> Result<()> {
        self.respond(Ok(response))
    }

    pub fn cancel(&self, reason: impl Into<String>) -> Result<()> {
        self.respond(Err(reason.into()))
    }
}

impl RunnerPermissionRequest {
    pub fn new(sender: oneshot::Sender<PermissionResponse>) -> Self {
        Self {
            sender: Arc::new(Mutex::new(Some(sender))),
        }
    }

    pub fn respond(&self, response: PermissionResponse) -> Result<()> {
        let sender = self
            .sender
            .lock()
            .map_err(|_| anyhow!("permission request lock poisoned"))?
            .take()
            .ok_or_else(|| anyhow!("permission request already resolved"))?;

        sender
            .send(response)
            .map_err(|_| anyhow!("permission response receiver dropped"))
    }

    pub fn approve(&self) -> Result<()> {
        self.respond(PermissionResponse::AllowOnce)
    }

    pub fn allow_always(&self) -> Result<()> {
        self.respond(PermissionResponse::AllowAlways)
    }

    pub fn deny(&self) -> Result<()> {
        self.respond(PermissionResponse::Deny)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ModelCatalogReasoning {
    pub effort: Option<String>,
    pub efforts: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ModelCatalogEntry {
    pub id: String,
    pub label: String,
    pub provider: String,
    pub context_window_tokens: Option<u64>,
    pub reasoning: ModelCatalogReasoning,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ModelCatalogUpdatedEvent {
    pub models: Vec<ModelCatalogEntry>,
}

#[derive(Debug, Clone)]
pub(crate) enum SessionTransportEvent {
    UserMessage(UserMessageEvent),
    ReasoningDelta(ReasoningDeltaEvent),
    ReasoningDone(ReasoningDoneEvent),
    AssistantDelta(AssistantDeltaEvent),
    AssistantDone {
        message_id: Option<String>,
    },
    TokenUsage(TokenUsageEvent),
    /// A complete local estimate of the committed session, rather than
    /// provider telemetry for the current turn.
    SessionTokenUsage(TokenUsageEvent),
    ToolPending(ToolPendingEvent),
    ToolCancelled(ToolCancelledEvent),
    ToolStarted(ToolStartedEvent),
    ToolFinished(ToolFinishedEvent),
    ToolOutputDelta(ToolOutputDeltaEvent),
    ToolBatchFinished,
    RetryScheduled(RetryLifecycleEvent),
    RetryStarted(RetryLifecycleEvent),
    QueuedPromptAccepted {
        prompt: UserMessageSubmission,
    },
    TodoSnapshot(TodoSnapshotEvent),
    AutoContinueChanged(AutoContinueChangedEvent),
    FastModeChanged {
        enabled: bool,
    },
    AnchoredChanged {
        active: bool,
    },
    ModelChanged {
        model_id: String,
    },
    ExpertModelChanged {
        agent_name: String,
        model_id: String,
    },
    PermissionModeChanged {
        mode: String,
    },
    ReasoningEffortChanged {
        effort: ModelReasoningEffort,
    },
    ModelCatalogUpdated(ModelCatalogUpdatedEvent),
    SettingChangeFailed {
        command: crate::session::SessionCommand,
    },
    PermissionRequested {
        event: PermissionRequestEvent,
        handle: RunnerPermissionRequest,
    },
    QuestionRequested {
        request: QuestionRequest,
        handle: RunnerQuestionRequest,
    },
    ChildPermissionRequested {
        child_session_id: String,
        agent_name: Option<String>,
        parent_tool_call_id: Option<String>,
        event: PermissionRequestEvent,
        handle: RunnerPermissionRequest,
    },
    ChildQuestionRequested {
        child_session_id: String,
        request: QuestionRequest,
        handle: RunnerQuestionRequest,
    },
    ChildSessionEvent {
        child_session_id: String,
        agent_name: Option<String>,
        parent_tool_call_id: Option<String>,
        event: SessionEvent,
    },
    PermissionResolved(PermissionResolutionEvent),
    ProcessIssue(ProcessIssueEvent),
    Notice(NoticeEvent),
    CompactionStarted,
    CompactionPreviewDelta {
        delta: String,
    },
    CompactionCommitted {
        summary: Option<String>,
    },
    CompactionNoProgress {
        blockers: Vec<String>,
    },
    CompactionFailed,
    RuntimeContextUpdated(RuntimeContextUpdatedEvent),
    #[allow(dead_code)]
    // Transport variants retained for context projection consumers.
    ContextTreeUpdated(ContextTreeUpdatedEvent),
    #[allow(dead_code)]
    // Transport variants retained for context projection consumers.
    ContextViewUpdated(ContextViewUpdatedEvent),
    #[allow(dead_code)]
    // Transport variants retained for context projection consumers.
    ContextDetailOpened(ContextDetailOpenedEvent),
    #[allow(dead_code)]
    // Transport variants retained for context projection consumers.
    ContextSummaryUpdated(ContextSummaryUpdatedEvent),
    McpToolsDiscovered(Vec<crate::mcp::McpServerCatalogEntry>),
    McpServerUpdated(crate::mcp::McpServerCatalogEntry),
    McpServerUpdating {
        name: String,
        updating: bool,
    },
    McpServerToolsUpdated {
        name: String,
        tools: Vec<crate::mcp::McpToolCatalogEntry>,
    },
    #[allow(dead_code)]
    // Retained so frontends can distinguish discovery failure from diagnostics.
    McpDiscoveryUnavailable(String),
    McpDiagnostic(String),
    SessionTitleUpdated {
        session_id: String,
        title: String,
    },
    Interrupted,
    SessionResumed {
        session_id: String,
        branch_id: String,
        messages: Vec<ConversationMessage>,
        records: Vec<TranscriptRecord>,
        evidence_count: usize,
        model_id: Option<String>,
        token_usage: Option<TokenUsageEvent>,
        runtime_context: RuntimeActiveContext,
        expert_models: indexmap::IndexMap<String, String>,
    },
    #[allow(dead_code)]
    // Branch changes are consumed by the TUI transport projection.
    ContextBranchChanged {
        branch_id: String,
    },
    SessionHistoryLoaded {
        entries: Vec<crate::transcript::transcript_projection::SessionHistoryEntry>,
    },
    ChildSessionViewed {
        parent_session_id: String,
        child_session_id: String,
        agent_name: String,
        index: usize,
        total: usize,
        pool_ordinal: u32,
        records: Vec<TranscriptRecord>,
        runtime_context: RuntimeActiveContext,
    },
    /// Parent transcript view navigation, symmetrical to [`Self::ChildSessionViewed`].
    /// Unlike [`Self::SessionResumed`] this is not a session restore: frontends must
    /// preserve in-flight runtime state (queued prompts, active turn, permissions).
    ParentSessionViewed {
        session_id: String,
        branch_id: String,
        records: Vec<TranscriptRecord>,
        model_id: Option<String>,
        token_usage: Option<TokenUsageEvent>,
        runtime_context: RuntimeActiveContext,
    },
    SessionStarted {
        session_id: String,
        records: Vec<TranscriptRecord>,
        runtime_context: RuntimeActiveContext,
        expert_models: indexmap::IndexMap<String, String>,
    },
    Error(ErrorEvent),
    Done,
}

impl SessionTransportEvent {
    pub fn session_event(&self) -> Option<SessionEvent> {
        match self {
            Self::UserMessage(event) => Some(SessionEvent::UserMessage(event.clone())),
            Self::ReasoningDelta(event) => Some(SessionEvent::ReasoningDelta(event.clone())),
            Self::ReasoningDone(event) => Some(SessionEvent::ReasoningDone(event.clone())),
            Self::AssistantDelta(event) => Some(SessionEvent::AssistantDelta(event.clone())),
            Self::AssistantDone { message_id } => Some(SessionEvent::AssistantDone {
                message_id: message_id.clone(),
            }),
            Self::TokenUsage(event) => Some(SessionEvent::TokenUsage(event.clone())),
            Self::SessionTokenUsage(event) => Some(SessionEvent::SessionTokenUsage(event.clone())),
            Self::ToolPending(event) => Some(SessionEvent::ToolPending(event.clone())),
            Self::ToolCancelled(event) => Some(SessionEvent::ToolCancelled(event.clone())),
            Self::ToolStarted(event) => Some(SessionEvent::ToolStarted(event.clone())),
            Self::ToolFinished(event) => Some(SessionEvent::ToolFinished(event.clone())),
            Self::ToolOutputDelta(event) => Some(SessionEvent::ToolOutputDelta(event.clone())),
            Self::ToolBatchFinished => Some(SessionEvent::ToolBatchFinished),
            Self::RetryScheduled(event) => Some(SessionEvent::RetryScheduled(event.clone())),
            Self::RetryStarted(event) => Some(SessionEvent::RetryStarted(event.clone())),
            Self::FastModeChanged { .. }
            | Self::AnchoredChanged { .. }
            | Self::ModelChanged { .. }
            | Self::ExpertModelChanged { .. }
            | Self::PermissionModeChanged { .. }
            | Self::ReasoningEffortChanged { .. }
            | Self::ModelCatalogUpdated(_)
            | Self::SettingChangeFailed { .. }
            | Self::QueuedPromptAccepted { .. } => None,
            Self::TodoSnapshot(event) => Some(SessionEvent::TodoSnapshot(event.clone())),
            Self::AutoContinueChanged(event) => {
                Some(SessionEvent::AutoContinueChanged(event.clone()))
            }
            Self::PermissionRequested { event, .. } => {
                Some(SessionEvent::PermissionRequested(event.clone()))
            }
            Self::QuestionRequested { .. }
            | Self::ChildPermissionRequested { .. }
            | Self::ChildQuestionRequested { .. }
            | Self::ChildSessionEvent { .. } => None,
            Self::PermissionResolved(event) => {
                Some(SessionEvent::PermissionResolved(event.clone()))
            }
            Self::ProcessIssue(event) => Some(SessionEvent::ProcessIssue(event.clone())),
            Self::Notice(event) => Some(SessionEvent::Notice(event.clone())),
            Self::RuntimeContextUpdated(event) => {
                Some(SessionEvent::RuntimeContextUpdated(event.clone()))
            }
            Self::ContextTreeUpdated(event) => {
                Some(SessionEvent::ContextTreeUpdated(event.clone()))
            }
            Self::ContextViewUpdated(event) => {
                Some(SessionEvent::ContextViewUpdated(event.clone()))
            }
            Self::ContextDetailOpened(event) => {
                Some(SessionEvent::ContextDetailOpened(event.clone()))
            }
            Self::ContextSummaryUpdated(event) => {
                Some(SessionEvent::ContextSummaryUpdated(event.clone()))
            }
            Self::CompactionStarted => Some(SessionEvent::CompactionStarted),
            Self::CompactionPreviewDelta { delta } => Some(SessionEvent::CompactionPreviewDelta {
                delta: delta.clone(),
            }),
            Self::CompactionCommitted { summary } => Some(SessionEvent::CompactionCommitted {
                summary: summary.clone(),
            }),
            Self::CompactionNoProgress { blockers } => Some(SessionEvent::CompactionNoProgress {
                blockers: blockers.clone(),
            }),
            Self::CompactionFailed => Some(SessionEvent::CompactionFailed),
            Self::McpToolsDiscovered(_)
            | Self::McpServerUpdated(_)
            | Self::McpServerUpdating { .. }
            | Self::McpServerToolsUpdated { .. }
            | Self::McpDiscoveryUnavailable(_)
            | Self::McpDiagnostic(_)
            | Self::SessionTitleUpdated { .. } => None,
            Self::Interrupted => Some(SessionEvent::Interrupted),
            Self::SessionStarted {
                session_id,
                runtime_context,
                ..
            } => Some(SessionEvent::SessionStarted {
                session_id: session_id.clone(),
                runtime_context: runtime_context.clone(),
            }),
            Self::SessionResumed {
                session_id,
                branch_id,
                messages,
                evidence_count,
                model_id,
                token_usage,
                runtime_context,
                ..
            } => Some(SessionEvent::SessionResumed {
                session_id: session_id.clone(),
                branch_id: branch_id.clone(),
                messages: messages.clone(),
                evidence_count: *evidence_count,
                model_id: model_id.clone(),
                token_usage: token_usage.clone(),
                runtime_context: runtime_context.clone(),
            }),
            Self::ContextBranchChanged { branch_id } => Some(SessionEvent::ContextBranchChanged {
                branch_id: branch_id.clone(),
            }),
            Self::SessionHistoryLoaded { .. } => None,
            Self::ChildSessionViewed { .. } | Self::ParentSessionViewed { .. } => None,
            Self::Error(event) => Some(SessionEvent::Error(event.clone())),
            Self::Done => Some(SessionEvent::Done),
        }
    }
}

pub(super) fn wrap_child_session_transport_event(
    child_session_id: String,
    agent_name: Option<String>,
    parent_tool_call_id: Option<String>,
    event: SessionTransportEvent,
) -> SessionTransportEvent {
    match event {
        SessionTransportEvent::UserMessage(event) => SessionTransportEvent::ChildSessionEvent {
            child_session_id,
            agent_name: agent_name.clone(),
            parent_tool_call_id: parent_tool_call_id.clone(),
            event: SessionEvent::UserMessage(event),
        },
        SessionTransportEvent::ReasoningDelta(event) => SessionTransportEvent::ChildSessionEvent {
            child_session_id,
            agent_name: agent_name.clone(),
            parent_tool_call_id: parent_tool_call_id.clone(),
            event: SessionEvent::ReasoningDelta(event),
        },
        SessionTransportEvent::ReasoningDone(event) => SessionTransportEvent::ChildSessionEvent {
            child_session_id,
            agent_name: agent_name.clone(),
            parent_tool_call_id: parent_tool_call_id.clone(),
            event: SessionEvent::ReasoningDone(event),
        },
        SessionTransportEvent::AssistantDelta(event) => SessionTransportEvent::ChildSessionEvent {
            child_session_id,
            agent_name: agent_name.clone(),
            parent_tool_call_id: parent_tool_call_id.clone(),
            event: SessionEvent::AssistantDelta(event),
        },
        SessionTransportEvent::AssistantDone { message_id } => {
            SessionTransportEvent::ChildSessionEvent {
                child_session_id,
                agent_name: agent_name.clone(),
                parent_tool_call_id: parent_tool_call_id.clone(),
                event: SessionEvent::AssistantDone { message_id },
            }
        }
        SessionTransportEvent::TokenUsage(event) => SessionTransportEvent::ChildSessionEvent {
            child_session_id,
            agent_name: agent_name.clone(),
            parent_tool_call_id: parent_tool_call_id.clone(),
            event: SessionEvent::TokenUsage(event),
        },
        SessionTransportEvent::ToolPending(event) => SessionTransportEvent::ChildSessionEvent {
            child_session_id,
            agent_name: agent_name.clone(),
            parent_tool_call_id: parent_tool_call_id.clone(),
            event: SessionEvent::ToolPending(event),
        },
        SessionTransportEvent::ToolCancelled(event) => SessionTransportEvent::ChildSessionEvent {
            child_session_id,
            agent_name: agent_name.clone(),
            parent_tool_call_id: parent_tool_call_id.clone(),
            event: SessionEvent::ToolCancelled(event),
        },
        SessionTransportEvent::ToolStarted(event) => SessionTransportEvent::ChildSessionEvent {
            child_session_id,
            agent_name: agent_name.clone(),
            parent_tool_call_id: parent_tool_call_id.clone(),
            event: SessionEvent::ToolStarted(event),
        },
        SessionTransportEvent::ToolFinished(event) => SessionTransportEvent::ChildSessionEvent {
            child_session_id,
            agent_name: agent_name.clone(),
            parent_tool_call_id: parent_tool_call_id.clone(),
            event: SessionEvent::ToolFinished(event),
        },
        SessionTransportEvent::ToolOutputDelta(event) => SessionTransportEvent::ChildSessionEvent {
            child_session_id,
            agent_name: agent_name.clone(),
            parent_tool_call_id: parent_tool_call_id.clone(),
            event: SessionEvent::ToolOutputDelta(event),
        },
        SessionTransportEvent::TodoSnapshot(event) => SessionTransportEvent::ChildSessionEvent {
            child_session_id,
            agent_name: agent_name.clone(),
            parent_tool_call_id: parent_tool_call_id.clone(),
            event: SessionEvent::TodoSnapshot(event),
        },
        SessionTransportEvent::AutoContinueChanged(event) => {
            SessionTransportEvent::ChildSessionEvent {
                child_session_id,
                agent_name: agent_name.clone(),
                parent_tool_call_id: parent_tool_call_id.clone(),
                event: SessionEvent::AutoContinueChanged(event),
            }
        }
        SessionTransportEvent::FastModeChanged { .. }
        | SessionTransportEvent::ModelChanged { .. }
        | SessionTransportEvent::ExpertModelChanged { .. }
        | SessionTransportEvent::PermissionModeChanged { .. }
        | SessionTransportEvent::ReasoningEffortChanged { .. }
        | SessionTransportEvent::SettingChangeFailed { .. } => event,
        SessionTransportEvent::PermissionResolved(event) => {
            SessionTransportEvent::ChildSessionEvent {
                child_session_id,
                agent_name: agent_name.clone(),
                parent_tool_call_id: parent_tool_call_id.clone(),
                event: SessionEvent::PermissionResolved(event),
            }
        }
        SessionTransportEvent::ProcessIssue(event) => SessionTransportEvent::ChildSessionEvent {
            child_session_id,
            agent_name: agent_name.clone(),
            parent_tool_call_id: parent_tool_call_id.clone(),
            event: SessionEvent::ProcessIssue(event),
        },
        SessionTransportEvent::Notice(event) => SessionTransportEvent::ChildSessionEvent {
            child_session_id,
            agent_name: agent_name.clone(),
            parent_tool_call_id: parent_tool_call_id.clone(),
            event: SessionEvent::Notice(event),
        },
        SessionTransportEvent::CompactionStarted => SessionTransportEvent::ChildSessionEvent {
            child_session_id,
            agent_name: agent_name.clone(),
            parent_tool_call_id: parent_tool_call_id.clone(),
            event: SessionEvent::CompactionStarted,
        },
        SessionTransportEvent::CompactionPreviewDelta { delta } => {
            SessionTransportEvent::ChildSessionEvent {
                child_session_id,
                agent_name: agent_name.clone(),
                parent_tool_call_id: parent_tool_call_id.clone(),
                event: SessionEvent::CompactionPreviewDelta { delta },
            }
        }
        SessionTransportEvent::CompactionCommitted { summary } => {
            SessionTransportEvent::ChildSessionEvent {
                child_session_id,
                agent_name: agent_name.clone(),
                parent_tool_call_id: parent_tool_call_id.clone(),
                event: SessionEvent::CompactionCommitted { summary },
            }
        }
        SessionTransportEvent::CompactionNoProgress { blockers } => {
            SessionTransportEvent::ChildSessionEvent {
                child_session_id,
                agent_name: agent_name.clone(),
                parent_tool_call_id: parent_tool_call_id.clone(),
                event: SessionEvent::CompactionNoProgress { blockers },
            }
        }
        SessionTransportEvent::CompactionFailed => SessionTransportEvent::ChildSessionEvent {
            child_session_id,
            agent_name: agent_name.clone(),
            parent_tool_call_id: parent_tool_call_id.clone(),
            event: SessionEvent::CompactionFailed,
        },
        SessionTransportEvent::RuntimeContextUpdated(event) => {
            SessionTransportEvent::ChildSessionEvent {
                child_session_id,
                agent_name: agent_name.clone(),
                parent_tool_call_id: parent_tool_call_id.clone(),
                event: SessionEvent::RuntimeContextUpdated(event),
            }
        }
        SessionTransportEvent::ContextTreeUpdated(event) => {
            SessionTransportEvent::ChildSessionEvent {
                child_session_id,
                agent_name: agent_name.clone(),
                parent_tool_call_id: parent_tool_call_id.clone(),
                event: SessionEvent::ContextTreeUpdated(event),
            }
        }
        SessionTransportEvent::ContextViewUpdated(event) => {
            SessionTransportEvent::ChildSessionEvent {
                child_session_id,
                agent_name: agent_name.clone(),
                parent_tool_call_id: parent_tool_call_id.clone(),
                event: SessionEvent::ContextViewUpdated(event),
            }
        }
        SessionTransportEvent::ContextDetailOpened(event) => {
            SessionTransportEvent::ChildSessionEvent {
                child_session_id,
                agent_name: agent_name.clone(),
                parent_tool_call_id: parent_tool_call_id.clone(),
                event: SessionEvent::ContextDetailOpened(event),
            }
        }
        SessionTransportEvent::ContextSummaryUpdated(event) => {
            SessionTransportEvent::ChildSessionEvent {
                child_session_id,
                agent_name: agent_name.clone(),
                parent_tool_call_id: parent_tool_call_id.clone(),
                event: SessionEvent::ContextSummaryUpdated(event),
            }
        }
        SessionTransportEvent::RetryScheduled(event) => SessionTransportEvent::ChildSessionEvent {
            child_session_id,
            agent_name: agent_name.clone(),
            parent_tool_call_id: parent_tool_call_id.clone(),
            event: SessionEvent::RetryScheduled(event),
        },
        SessionTransportEvent::RetryStarted(event) => SessionTransportEvent::ChildSessionEvent {
            child_session_id,
            agent_name: agent_name.clone(),
            parent_tool_call_id: parent_tool_call_id.clone(),
            event: SessionEvent::RetryStarted(event),
        },
        SessionTransportEvent::SessionTitleUpdated { .. } => event,
        SessionTransportEvent::Interrupted => SessionTransportEvent::ChildSessionEvent {
            child_session_id,
            agent_name: agent_name.clone(),
            parent_tool_call_id: parent_tool_call_id.clone(),
            event: SessionEvent::Interrupted,
        },
        SessionTransportEvent::Error(event) => SessionTransportEvent::ChildSessionEvent {
            child_session_id,
            agent_name: agent_name.clone(),
            parent_tool_call_id: parent_tool_call_id.clone(),
            event: SessionEvent::Error(event),
        },
        SessionTransportEvent::Done => SessionTransportEvent::ChildSessionEvent {
            child_session_id,
            agent_name: agent_name.clone(),
            parent_tool_call_id: parent_tool_call_id.clone(),
            event: SessionEvent::Done,
        },
        event => event,
    }
}

pub(super) fn send_scoped_event(
    sender: &Option<SessionTransportEventSender>,
    child_session_id: Option<&str>,
    agent_name: Option<&str>,
    parent_tool_call_id: Option<&str>,
    event: SessionTransportEvent,
) -> Result<()> {
    let event = match child_session_id {
        Some(child_session_id) => wrap_child_session_transport_event(
            child_session_id.to_string(),
            agent_name.map(ToOwned::to_owned),
            parent_tool_call_id.map(ToOwned::to_owned),
            event,
        ),
        None => event,
    };
    send_optional_event(sender, event)
}

pub(super) fn emit_context_projection_updates(
    sender: &Option<SessionTransportEventSender>,
    transcript: &Option<Arc<Mutex<TranscriptRecorder>>>,
    child_session_id: Option<&str>,
    agent_name: Option<&str>,
    parent_tool_call_id: Option<&str>,
) -> Result<()> {
    emit_context_projection_update(
        sender,
        transcript,
        child_session_id,
        agent_name,
        parent_tool_call_id,
        RuntimeContextDisposition::Advance,
    )
}

pub(super) fn emit_context_projection_update(
    sender: &Option<SessionTransportEventSender>,
    transcript: &Option<Arc<Mutex<TranscriptRecorder>>>,
    child_session_id: Option<&str>,
    agent_name: Option<&str>,
    parent_tool_call_id: Option<&str>,
    disposition: RuntimeContextDisposition,
) -> Result<()> {
    let Some(transcript) = transcript else {
        return Ok(());
    };
    let recorder = transcript
        .lock()
        .map_err(|_| anyhow!("transcript recorder poisoned"))?;
    let path = recorder.path().to_path_buf();
    let session_id = recorder.session_id().to_string();
    let branch_id = recorder.current_context_branch_id().map(str::to_string);
    drop(recorder);
    let records = read_records(&path)?;
    let snapshot = crate::transcript::transcript_projection::project_runtime_restore_snapshot(
        session_id,
        records,
        crate::transcript::transcript_projection::SessionContextCursor {
            branch_id,
            leaf_sequence: None,
        },
        &[],
    )?
    .snapshot;
    let context = RuntimeActiveContext::try_from(&snapshot)?;
    send_scoped_event(
        sender,
        child_session_id,
        agent_name,
        parent_tool_call_id,
        SessionTransportEvent::RuntimeContextUpdated(RuntimeContextUpdatedEvent {
            context,
            disposition,
        }),
    )?;
    Ok(())
}

pub(super) fn send_optional_event(
    sender: &Option<SessionTransportEventSender>,
    event: SessionTransportEvent,
) -> Result<()> {
    if let Some(sender) = sender {
        sender
            .send(event)
            .map_err(|_| anyhow!("runner event channel closed"))?;
    }
    Ok(())
}

pub(super) fn record_transcript<F>(
    transcript: &Option<Arc<Mutex<TranscriptRecorder>>>,
    f: F,
) -> Result<()>
where
    F: FnOnce(&mut TranscriptRecorder) -> Result<()>,
{
    let Some(transcript) = transcript else {
        return Ok(());
    };

    let mut recorder = transcript
        .lock()
        .map_err(|_| anyhow!("transcript recorder poisoned"))?;
    f(&mut recorder)
}

pub(super) fn retry_lifecycle_event(retry: LlmRetryLifecycle) -> RetryLifecycleEvent {
    RetryLifecycleEvent {
        attempt: retry.attempt,
        max_attempts: retry.max_attempts,
        delay_secs: retry.delay_secs,
        error: retry.error,
    }
}

pub(super) fn tool_started_event(call_id: String, name: String, args: Value) -> ToolStartedEvent {
    let summary = format_tool_call(&name, &args);
    ToolStartedEvent {
        call_id,
        name,
        summary,
        arguments: Some(args.to_string()),
    }
}

pub(super) fn tool_finished_event(
    call_id: String,
    name: String,
    ok: bool,
    output: ToolResult,
) -> ToolFinishedEvent {
    let summary = output_summary(&output).unwrap_or_else(|| name.clone());
    ToolFinishedEvent {
        call_id,
        name,
        summary,
        outcome: if ok {
            ToolOutcome::Success
        } else {
            ToolOutcome::Failure
        },
        output: Some(output_json(&output).to_string()),
    }
}

pub(super) fn permission_request_event(
    request: &PermissionRequest,
    permission_origin: Option<&str>,
) -> PermissionRequestEvent {
    let mut event = PermissionRequestEvent::new(
        request
            .call_id
            .clone()
            .unwrap_or_else(|| request.tool.clone()),
        request.tool.clone(),
        request.summary.clone(),
    );
    event.arguments = Some(request.args.to_string());
    event.rationale = Some(format!("{} permission requires approval", request.class));
    event.origin_label = permission_origin.map(ToOwned::to_owned);
    event.can_allow_always = request.can_allow_always;
    event.grant_summary = request.grant_summary.clone();
    event
}

pub(super) fn permission_resolution_event(
    request: &PermissionRequest,
    response: PermissionResponse,
) -> PermissionResolutionEvent {
    let call_id = request
        .call_id
        .clone()
        .unwrap_or_else(|| request.tool.clone());
    match response {
        PermissionResponse::AllowOnce => PermissionResolutionEvent {
            call_id,
            decision: crate::session::PermissionDecision::Approved,
            reason: Some("Allow once".into()),
            tool_name: None,
            summary: None,
            origin_label: None,
            approval: None,
            risk: None,
            reviewer_child_session_id: None,
        },
        PermissionResponse::AllowAlways => PermissionResolutionEvent {
            call_id,
            decision: crate::session::PermissionDecision::Approved,
            reason: Some("Allowed for this session".into()),
            tool_name: None,
            summary: None,
            origin_label: None,
            approval: None,
            risk: None,
            reviewer_child_session_id: None,
        },
        PermissionResponse::Deny => {
            PermissionResolutionEvent::denied(call_id, Some("Denied".into()))
        }
    }
}
