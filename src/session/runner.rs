use std::path::PathBuf;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use anyhow::{Result, anyhow, bail};
use async_openai::config::Config;
use serde_json::{Value, json};
use tokio::sync::{mpsc, oneshot};
use tracing::warn;

use crate::agent::{
    Agent, AgentEvent, ConversationMessage, LlmRetryLifecycle, SubagentDelegate,
    SubagentInvocation, is_subagent_tool_name, subagent_tool_name_for_agent_name,
};
use crate::agent_event_journal::{ContextProjection, JournalEffect, persist_agent_event};
use crate::permission::{PermissionApproval, PermissionRequest};
use crate::runtime_context::RuntimeActiveContext;
use crate::subagent::{SubagentFailureKind, SubagentPool, SubagentStatus};
use crate::subagent_events::SubagentEventSender;
use crate::tool::{QuestionRequest, QuestionResponse, ToolResult};
use crate::tool_format::format_tool_call;
use crate::tool_names;
use crate::transcript::{
    TranscriptRecord, TranscriptRecorder, read_records, transcript_has_session_title,
    transcript_has_user_message,
};
use crate::user_content::UserMessageContent;
use crate::user_content::UserMessageSubmission;

use crate::session::{
    AssistantDeltaEvent, AutoContinueChangedEvent, ContextDetailOpenedEvent,
    ContextSummaryUpdatedEvent, ContextTreeUpdatedEvent, ContextViewUpdatedEvent, ErrorEvent,
    NoticeEvent, PermissionRequestEvent, PermissionResolutionEvent, ProcessIssueEvent,
    ReasoningDeltaEvent, ReasoningDoneEvent, RetryLifecycleEvent, RuntimeContextDisposition,
    RuntimeContextUpdatedEvent, SessionEvent, TodoSnapshotEvent, TokenUsageEvent,
    ToolCancelledEvent, ToolFinishedEvent, ToolOutcome, ToolOutputDeltaEvent, ToolPendingEvent,
    ToolStartedEvent, UserMessageEvent,
};

pub(crate) type SessionTransportEventSender = mpsc::UnboundedSender<SessionTransportEvent>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SessionTransportEventMode {
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
    ModelChanged {
        model_id: String,
    },
    ExpertModelChanged {
        agent_name: String,
        model_id: String,
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
            | Self::ModelChanged { .. }
            | Self::ExpertModelChanged { .. }
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

pub(crate) struct AgentRunner<C: Config> {
    event_tx: Option<SessionTransportEventSender>,
    permission_event_tx: Option<SessionTransportEventSender>,
    session_title_event_tx: Option<SessionTransportEventSender>,
    transcript: Option<Arc<Mutex<TranscriptRecorder>>>,
    event_mode: SessionTransportEventMode,
    child_session_id: Option<String>,
    permission_origin: Option<String>,
    parent_tool_call_id: Option<String>,
    subagent_delegate: Option<Arc<dyn SubagentDelegate<C>>>,
    _config: std::marker::PhantomData<C>,
}

struct RunnerSubagentDelegate {
    runtime: SubagentPool,
    sessions_dir: PathBuf,
    transcript: Arc<Mutex<TranscriptRecorder>>,
    event_tx: Option<SessionTransportEventSender>,
    expert_model_routes: indexmap::IndexMap<String, crate::config::ModelRoute>,
    route_api_key_configured: indexmap::IndexMap<String, bool>,
    provider_api_key_hints: indexmap::IndexMap<String, String>,
    api_key_hint: String,
}

impl RunnerSubagentDelegate {
    fn route_display_name(
        &self,
        parent: &Agent<async_openai::config::OpenAIConfig>,
        agent_name: &str,
        target_child_session_id: Option<&str>,
    ) -> Result<String> {
        let Some(target_child_session_id) = target_child_session_id else {
            return Ok(self.expert_model_routes.get(agent_name).map_or_else(
                || parent.route_display_name(),
                crate::config::ModelRoute::display_name,
            ));
        };
        let parent_records = self
            .transcript
            .lock()
            .map_err(|_| anyhow!("transcript recorder poisoned"))
            .and_then(|recorder| {
                crate::transcript::read_records(recorder.path()).map_err(Into::into)
            })?;
        let child = SubagentPool::child_sessions(&self.sessions_dir, &parent_records)
            .into_iter()
            .find(|child| child.child_session_id == target_child_session_id)
            .ok_or_else(|| {
                anyhow!(
                    "takeover failed: child_session_id `{target_child_session_id}` is not a known child of this parent"
                )
            })?;
        if child.agent_name != agent_name {
            bail!(
                "takeover failed: child `{target_child_session_id}` is agent `{}`, expected `{agent_name}`",
                child.agent_name
            );
        }
        let child_records = crate::transcript::read_records_allow_partial_tail(
            crate::transcript::child_sessions_dir(&self.sessions_dir)
                .join(format!("{target_child_session_id}.jsonl")),
        )?;
        crate::transcript::restore_latest_model(&child_records).ok_or_else(|| {
            anyhow!(
                "takeover failed: child `{target_child_session_id}` has no recorded model route"
            )
        })
    }

    fn missing_api_key_result(
        &self,
        tool_name: &str,
        agent_name: &str,
        route_display_name: String,
    ) -> ToolResult {
        let provider = route_display_name
            .split_once('/')
            .map(|(provider, _)| provider)
            .unwrap_or("selected");
        let hint = self
            .provider_api_key_hints
            .get(provider)
            .cloned()
            .unwrap_or_else(|| self.api_key_hint.clone());
        let summary = format!("API key is not set for the selected provider. {hint}");
        let data = json!({
            "agent_name": agent_name,
            "route": route_display_name,
            "status": SubagentStatus::Failed.as_str(),
            "failure_kind": SubagentFailureKind::Hard.as_str(),
            "summary": summary,
            "full_summary": summary,
            "active": false,
            "unreconciled": false,
            "reconciled": false,
            "reusable": false,
        });
        ToolResult::err_with_data(tool_name, summary, data)
    }
}

impl SubagentDelegate<async_openai::config::OpenAIConfig> for RunnerSubagentDelegate {
    fn run_named<'a>(
        &'a self,
        parent: &'a Agent<async_openai::config::OpenAIConfig>,
        agent_name: &'a str,
        invocation: SubagentInvocation,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<ToolResult>> + Send + 'a>> {
        Box::pin(async move {
            let tool_name = subagent_tool_name_for_agent_name(agent_name)
                .expect("runner dispatched unknown subagent agent name");
            let route_display_name = self.route_display_name(
                parent,
                agent_name,
                invocation.input.target_child_session_id.as_deref(),
            )?;
            if !self
                .route_api_key_configured
                .get(&route_display_name)
                .copied()
                .unwrap_or(false)
            {
                return Ok(self.missing_api_key_result(tool_name, agent_name, route_display_name));
            }
            let target_child_session_id = invocation.input.target_child_session_id.clone();
            let parent_session_id = match self.transcript.lock() {
                Ok(recorder) => recorder.session_id().to_string(),
                Err(_) => {
                    let summary = "transcript recorder poisoned".to_string();
                    let data = json!({
                        "agent_name": agent_name,
                        "child_session_id": target_child_session_id,
                        "status": SubagentStatus::Failed.as_str(),
                        "failure_kind": SubagentFailureKind::Hard.as_str(),
                        "summary": summary,
                        "full_summary": summary,
                        "active": false,
                        "unreconciled": false,
                        "reconciled": false,
                        "reusable": false,
                    });
                    return Ok(ToolResult::err_with_data(tool_name, summary, data));
                }
            };
            let parent_turn_id = format!(
                "turn-{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis()
            );
            let summary = self
                .runtime
                .run_named_governed(
                    parent,
                    agent_name,
                    invocation,
                    self.sessions_dir.clone(),
                    parent_session_id,
                    parent_turn_id,
                    Some(self.transcript.clone()),
                    self.event_tx.clone().map(subagent_event_sender),
                )
                .await;

            let summary = match summary {
                Ok(summary) => summary,
                Err(error) => {
                    let summary = error.to_string();
                    let data = json!({
                        "agent_name": agent_name,
                        "child_session_id": target_child_session_id,
                        "status": SubagentStatus::Failed.as_str(),
                        "failure_kind": SubagentFailureKind::Hard.as_str(),
                        "summary": compact_subagent_summary(&summary),
                        "full_summary": summary,
                        "active": false,
                        "unreconciled": false,
                        "reconciled": false,
                        "reusable": false,
                    });
                    return Ok(ToolResult::err_with_data(tool_name, summary, data));
                }
            };

            let status = summary.status;
            let failure_kind = summary.failure_kind;
            let summary_text = summary.summary.clone();
            let compact_summary = compact_subagent_summary(&summary.summary);

            let data = json!({
                "run_id": summary.run_id,
                "child_session_id": summary.child_session_id,
                "agent_name": summary.agent_name,
                "status": status.as_str(),
                "failure_kind": failure_kind.map(|kind| kind.as_str()),
                "summary": compact_summary,
                "full_summary": summary.summary,
                "structured_result": summary.structured_result,
                "active": false,
                "unreconciled": status == SubagentStatus::Completed,
                "reconciled": false,
                "reusable": false,
            });

            if status == SubagentStatus::Completed {
                Ok(ToolResult::ok(tool_name, data))
            } else {
                Ok(ToolResult::err_with_data(tool_name, summary_text, data))
            }
        })
    }
}

impl AgentRunner<async_openai::config::OpenAIConfig> {
    pub fn with_subagent_runtime(
        self,
        runtime: SubagentPool,
        sessions_dir: PathBuf,
        expert_model_routes: indexmap::IndexMap<String, crate::config::ModelRoute>,
        route_api_key_configured: indexmap::IndexMap<String, bool>,
        provider_api_key_hints: indexmap::IndexMap<String, String>,
        api_key_hint: String,
    ) -> Self {
        let mut self_ = self;
        if let Some(transcript) = self_.transcript.clone() {
            self_.subagent_delegate = Some(Arc::new(RunnerSubagentDelegate {
                runtime,
                sessions_dir,
                transcript,
                event_tx: self_.event_tx.clone(),
                expert_model_routes,
                route_api_key_configured,
                provider_api_key_hints,
                api_key_hint,
            }));
        }
        self_
    }
}

impl<C: Config> AgentRunner<C> {
    #[cfg(test)]
    pub fn new(event_tx: SessionTransportEventSender) -> Self {
        Self {
            event_tx: Some(event_tx),
            permission_event_tx: None,
            session_title_event_tx: None,
            transcript: None,
            event_mode: SessionTransportEventMode::Emit,
            child_session_id: None,
            permission_origin: None,
            parent_tool_call_id: None,
            subagent_delegate: None,
            _config: std::marker::PhantomData,
        }
    }

    pub fn with_transcript(
        event_tx: SessionTransportEventSender,
        transcript: Arc<Mutex<TranscriptRecorder>>,
    ) -> Self {
        Self {
            event_tx: Some(event_tx),
            permission_event_tx: None,
            session_title_event_tx: None,
            transcript: Some(transcript),
            event_mode: SessionTransportEventMode::Emit,
            child_session_id: None,
            permission_origin: None,
            parent_tool_call_id: None,
            subagent_delegate: None,
            _config: std::marker::PhantomData,
        }
    }

    pub fn with_session_title_event_sender(
        mut self,
        event_tx: SessionTransportEventSender,
    ) -> Self {
        self.session_title_event_tx = Some(event_tx);
        self
    }

    #[cfg(test)]
    pub fn silent_with_transcript(transcript: Arc<Mutex<TranscriptRecorder>>) -> Self {
        Self {
            event_tx: None,
            permission_event_tx: None,
            session_title_event_tx: None,
            transcript: Some(transcript),
            event_mode: SessionTransportEventMode::SilentDenyPermissions,
            child_session_id: None,
            permission_origin: None,
            parent_tool_call_id: None,
            subagent_delegate: None,
            _config: std::marker::PhantomData,
        }
    }

    pub fn child_streaming_with_transcript(
        transcript: Arc<Mutex<TranscriptRecorder>>,
        event_tx: SessionTransportEventSender,
        child_session_id: impl Into<String>,
    ) -> Self {
        Self {
            event_tx: Some(event_tx),
            permission_event_tx: None,
            session_title_event_tx: None,
            transcript: Some(transcript),
            event_mode: SessionTransportEventMode::SilentDenyPermissions,
            child_session_id: Some(child_session_id.into()),
            permission_origin: None,
            parent_tool_call_id: None,
            subagent_delegate: None,
            _config: std::marker::PhantomData,
        }
    }

    pub fn child_streaming_with_permission_passthrough(
        transcript: Arc<Mutex<TranscriptRecorder>>,
        event_tx: SessionTransportEventSender,
        child_session_id: impl Into<String>,
        permission_origin: impl Into<String>,
        parent_tool_call_id: Option<String>,
    ) -> Self {
        Self {
            event_tx: Some(event_tx.clone()),
            permission_event_tx: Some(event_tx),
            session_title_event_tx: None,
            transcript: Some(transcript),
            event_mode: SessionTransportEventMode::Emit,
            child_session_id: Some(child_session_id.into()),
            permission_origin: Some(permission_origin.into()),
            parent_tool_call_id,
            subagent_delegate: None,
            _config: std::marker::PhantomData,
        }
    }

    pub async fn run_prompt(
        &self,
        agent: &mut Agent<C>,
        prompt: UserMessageSubmission,
    ) -> Result<String>
    where
        C: Clone + Send + Sync + 'static,
    {
        self.run_prompt_with_options(agent, prompt, true).await
    }

    #[cfg(test)]
    pub async fn run_internal_prompt(
        &self,
        agent: &mut Agent<C>,
        prompt: impl Into<String>,
    ) -> Result<String>
    where
        C: Clone + Send + Sync + 'static,
    {
        self.run_prompt_with_options(
            agent,
            UserMessageSubmission::new(
                "internal-continuation",
                UserMessageContent::new(prompt, Vec::new()),
            ),
            false,
        )
        .await
    }

    async fn run_prompt_with_options(
        &self,
        agent: &mut Agent<C>,
        prompt: UserMessageSubmission,
        record_user_prompt: bool,
    ) -> Result<String>
    where
        C: Clone + Send + Sync + 'static,
    {
        let prompt_content = prompt.content.clone();
        let prompt_text = prompt_content.text.clone();
        if let Some(transcript) = self.transcript.clone() {
            agent.clear_logical_checkpoint_candidate_provider();
            agent.set_runtime_snapshot_provider(Arc::new(move || {
                let transcript = transcript
                    .lock()
                    .map_err(|_| anyhow!("transcript recorder poisoned"))?;
                let records = read_records(transcript.path())?;
                Ok(
                    crate::transcript::transcript_projection::project_runtime_restore_snapshot(
                        transcript.session_id().to_string(),
                        records,
                        crate::transcript::transcript_projection::SessionContextCursor {
                            branch_id: transcript.current_context_branch_id().map(str::to_string),
                            leaf_sequence: None,
                        },
                        &[],
                    )?
                    .snapshot,
                )
            }));
        } else {
            agent.clear_runtime_snapshot_provider();
            agent.clear_logical_checkpoint_candidate_provider();
        }
        if let Some(delegate) = self.subagent_delegate.clone() {
            agent.set_subagent_delegate(delegate);
        }
        if record_user_prompt {
            let user_event = UserMessageEvent::from_submission(prompt.clone());
            self.emit(SessionTransportEvent::UserMessage(user_event))?;
        }
        let pending_title = match self.pending_session_title(agent, record_user_prompt) {
            Ok(pending_title) => pending_title,
            Err(error) => {
                self.finish_with_error(error)?;
                unreachable!("finish_with_error always returns an error");
            }
        };
        if record_user_prompt {
            self.record(|recorder| recorder.record_user_message_content(prompt_content.clone()))
                .or_else(|error| self.finish_with_error(error))?;
            emit_context_projection_updates(
                &self.event_tx,
                &self.transcript,
                self.child_session_id.as_deref(),
                self.permission_origin.as_deref(),
                self.parent_tool_call_id.as_deref(),
            )
            .or_else(|error| self.finish_with_error(error))?;
        }
        if let Some((session_id, mut title_agent)) = pending_title {
            let transcript = self.transcript.clone();
            let event_tx = self
                .session_title_event_tx
                .clone()
                .or_else(|| self.event_tx.clone());
            let prompt = prompt_text.clone();
            tokio::spawn(async move {
                match title_agent.generate_session_title(&prompt).await {
                    Ok(title) => {
                        let Some(transcript) = transcript else {
                            return;
                        };
                        let mut recorder = match transcript.lock() {
                            Ok(recorder) => recorder,
                            Err(_) => {
                                warn!(
                                    session_id,
                                    "failed to record session title: transcript recorder poisoned"
                                );
                                return;
                            }
                        };
                        if recorder.session_id() != session_id {
                            return;
                        }
                        if let Err(error) = recorder.record_session_title(title.clone()) {
                            warn!(error = %error, session_id, "failed to persist generated session title");
                        } else if let Err(error) = send_optional_event(
                            &event_tx,
                            SessionTransportEvent::SessionTitleUpdated {
                                session_id: session_id.clone(),
                                title,
                            },
                        ) {
                            warn!(error = %error, session_id, "failed to emit generated session title update");
                        }
                    }
                    Err(error) => {
                        warn!(error = %error, session_id, "failed to generate session title")
                    }
                }
            });
        }

        let sender = self.event_tx.clone();
        let child_session_id = self.child_session_id.clone();
        let agent_name = self.permission_origin.clone();
        let parent_tool_call_id = self.parent_tool_call_id.clone();
        let response = agent
            .run_stream_content_with_interactions_async(
                prompt_content.clone(),
                move |delta| {
                    let sender = sender.clone();
                    let child_session_id = child_session_id.clone();
                    let agent_name = agent_name.clone();
                    let parent_tool_call_id = parent_tool_call_id.clone();
                    let delta = delta.to_string();
                    async move {
                        send_scoped_event(
                            &sender,
                            child_session_id.as_deref(),
                            agent_name.as_deref(),
                            parent_tool_call_id.as_deref(),
                            SessionTransportEvent::AssistantDelta(AssistantDeltaEvent::new(delta)),
                        )
                    }
                },
                {
                    let sender = self.event_tx.clone();
                    let transcript = self.transcript.clone();
                    let child_session_id = self.child_session_id.clone();
                    let agent_name = self.permission_origin.clone();
                    let parent_tool_call_id = self.parent_tool_call_id.clone();
                    move |event| {
                        let sender = sender.clone();
                        let transcript = transcript.clone();
                        let child_session_id = child_session_id.clone();
                        let agent_name = agent_name.clone();
                        let parent_tool_call_id = parent_tool_call_id.clone();
                        async move {
                            let journal_effect = match transcript.as_ref() {
                                None => JournalEffect {
                                    persisted: false,
                                    context_projection: ContextProjection::None,
                                    compaction_terminal: false,
                                },
                                Some(transcript) => match transcript
                                    .lock()
                                    .map_err(|_| anyhow!("transcript recorder poisoned"))
                                    .and_then(|mut recorder| persist_agent_event(&mut recorder, &event))
                                {
                                Ok(effect) => effect,
                                Err(error)
                                    if matches!(
                                        event,
                                        AgentEvent::TurnStarted(_)
                                            | AgentEvent::ToolExecutionSummary(_)
                                            | AgentEvent::TurnFinalized(_)
                                    ) => {
                                        warn!(error = %error, "failed to record agent audit event; continuing runner");
                                        JournalEffect {
                                            persisted: false,
                                            context_projection: ContextProjection::None,
                                            compaction_terminal: false,
                                        }
                                    }
                                Err(error) => return Err(error),
                                },
                            };
                            match event {
                                AgentEvent::ContextCompactionStarted { .. } => send_scoped_event(
                                    &sender,
                                    child_session_id.as_deref(),
                                    agent_name.as_deref(),
                                    parent_tool_call_id.as_deref(),
                                    SessionTransportEvent::CompactionStarted,
                                )?,
                                AgentEvent::ContextCompactionNoProgress(no_progress) => {
                                    send_scoped_event(
                                        &sender,
                                        child_session_id.as_deref(),
                                        agent_name.as_deref(),
                                        parent_tool_call_id.as_deref(),
                                        SessionTransportEvent::CompactionNoProgress {
                                            blockers: no_progress.blockers.into_iter()
                                                .map(|blocker| blocker.label().to_string())
                                                .collect(),
                                        },
                                    )?;
                                }
                                AgentEvent::ContextCompactionFailed { .. } => send_scoped_event(
                                    &sender,
                                    child_session_id.as_deref(),
                                    agent_name.as_deref(),
                                    parent_tool_call_id.as_deref(),
                                    SessionTransportEvent::CompactionFailed,
                                )?,
                                AgentEvent::ContextCompactionDelta { delta } => send_scoped_event(
                                    &sender,
                                    child_session_id.as_deref(),
                                    agent_name.as_deref(),
                                    parent_tool_call_id.as_deref(),
                                    SessionTransportEvent::CompactionPreviewDelta { delta },
                                )?,
                                AgentEvent::TokenUsageUpdated {
                                    used_tokens,
                                    context_window_tokens,
                                    input_tokens,
                                    output_tokens,
                                    cached_tokens,
                                    cache_report,
                                } => {
                                    send_scoped_event(
                                        &sender,
                                        child_session_id.as_deref(),
                                        agent_name.as_deref(),
                                        parent_tool_call_id.as_deref(),
                                        SessionTransportEvent::TokenUsage(TokenUsageEvent::with_breakdown(
                                            used_tokens,
                                            context_window_tokens,
                                            input_tokens,
                                            output_tokens,
                                            cached_tokens,
                                        ).with_cache_report(cache_report)),
                                    )?;
                                }
                                AgentEvent::LlmRequestTelemetry(_) => {}
                                AgentEvent::FastModeChanged { enabled } => {
                                    send_scoped_event(
                                        &sender,
                                        child_session_id.as_deref(),
                                        agent_name.as_deref(),
                                        parent_tool_call_id.as_deref(),
                                        SessionTransportEvent::FastModeChanged { enabled },
                                    )?;
                                    send_scoped_event(
                                        &sender,
                                        child_session_id.as_deref(),
                                        agent_name.as_deref(),
                                        parent_tool_call_id.as_deref(),
                                        SessionTransportEvent::Notice(NoticeEvent::info(
                                            "Fast mode auto-disabled: current model is unavailable",
                                        )),
                                    )?;
                                }
                                AgentEvent::LlmRetryScheduled(retry) => {
                                    send_scoped_event(
                                        &sender,
                                        child_session_id.as_deref(),
                                        agent_name.as_deref(),
                                        parent_tool_call_id.as_deref(),
                                        SessionTransportEvent::RetryScheduled(retry_lifecycle_event(retry)),
                                    )?;
                                }
                                AgentEvent::LlmRetryStarted(retry) => {
                                    send_scoped_event(
                                        &sender,
                                        child_session_id.as_deref(),
                                        agent_name.as_deref(),
                                        parent_tool_call_id.as_deref(),
                                        SessionTransportEvent::RetryStarted(retry_lifecycle_event(retry)),
                                    )?;
                                }
                                AgentEvent::TurnStarted(event) => {
                                    let _ = event;
                                }
                                AgentEvent::EvidenceRecorded(_) => {
                                    emit_context_projection_updates(
                                        &sender,
                                        &transcript,
                                        child_session_id.as_deref(),
                                        agent_name.as_deref(),
                                        parent_tool_call_id.as_deref(),
                                    )?;
                                }
                                AgentEvent::ReasoningDelta { item_id, delta } => {
                                    send_scoped_event(
                                        &sender,
                                        child_session_id.as_deref(),
                                        agent_name.as_deref(),
                                        parent_tool_call_id.as_deref(),
                                        SessionTransportEvent::ReasoningDelta(ReasoningDeltaEvent::new(
                                            item_id, delta,
                                        )),
                                    )?;
                                }
                                AgentEvent::ReasoningDone { item_id, text } => {
                                    emit_context_projection_updates(
                                        &sender,
                                        &transcript,
                                        child_session_id.as_deref(),
                                        agent_name.as_deref(),
                                        parent_tool_call_id.as_deref(),
                                    )?;
                                    send_scoped_event(
                                        &sender,
                                        child_session_id.as_deref(),
                                        agent_name.as_deref(),
                                        parent_tool_call_id.as_deref(),
                                        SessionTransportEvent::ReasoningDone(ReasoningDoneEvent::new(
                                            item_id, text,
                                        )),
                                    )?;
                                }
                                AgentEvent::ModelStreamIssue {
                                    message,
                                    detail,
                                    action,
                                } => {
                                    let issue = ProcessIssueEvent {
                                        message,
                                        detail,
                                        action: Some(action),
                                    };
                                    send_scoped_event(
                                        &sender,
                                        child_session_id.as_deref(),
                                        agent_name.as_deref(),
                                        parent_tool_call_id.as_deref(),
                                        SessionTransportEvent::ProcessIssue(issue),
                                    )?;
                                }
                                AgentEvent::AssistantMessage { .. }
                                | AgentEvent::AssistantToolCallBatch { .. }
                                | AgentEvent::InternalContinuation { .. } => {}
                                AgentEvent::ToolCallPending { call_id, name } => {
                                    send_scoped_event(
                                        &sender,
                                        child_session_id.as_deref(),
                                        agent_name.as_deref(),
                                        parent_tool_call_id.as_deref(),
                                        SessionTransportEvent::ToolPending(ToolPendingEvent::new(
                                            call_id, name,
                                        )),
                                    )?;
                                }
                                AgentEvent::ToolCallStarted {
                                    call_id,
                                    name,
                                    args,
                                } => {
                                    let started = tool_started_event(call_id, name, args);
                                    send_scoped_event(
                                        &sender,
                                        child_session_id.as_deref(),
                                        agent_name.as_deref(),
                                        parent_tool_call_id.as_deref(),
                                        SessionTransportEvent::ToolStarted(started),
                                    )?;
                                }
                                AgentEvent::ToolOutputDelta {
                                    call_id,
                                    stream,
                                    chunk,
                                } => {
                                    send_scoped_event(
                                        &sender,
                                        child_session_id.as_deref(),
                                        agent_name.as_deref(),
                                        parent_tool_call_id.as_deref(),
                                        SessionTransportEvent::ToolOutputDelta(ToolOutputDeltaEvent::new(
                                            call_id, stream, chunk,
                                        )),
                                    )?;
                                }
                                AgentEvent::ToolCallFinished {
                                    call_id,
                                    name,
                                    ok,
                                    output,
                                } => {
                                    let finished =
                                        tool_finished_event(call_id, name, ok, output.clone());
                                    let disposition = if matches!(
                                        journal_effect.context_projection,
                                        ContextProjection::ReplaceScope
                                    ) {
                                        RuntimeContextDisposition::ReplaceScope
                                    } else {
                                        RuntimeContextDisposition::Advance
                                    };
                                    send_scoped_event(
                                        &sender,
                                        child_session_id.as_deref(),
                                        agent_name.as_deref(),
                                        parent_tool_call_id.as_deref(),
                                        SessionTransportEvent::ToolFinished(finished),
                                    )?;
                                    // The durable terminal event is authoritative even if its
                                    // subsequent projection cannot be rebuilt.
                                    emit_context_projection_update(
                                        &sender,
                                        &transcript,
                                        child_session_id.as_deref(),
                                        agent_name.as_deref(),
                                        parent_tool_call_id.as_deref(),
                                        disposition,
                                    )?;
                                }
                                AgentEvent::ToolCallBatchFinished => {
                                    if child_session_id.is_none() {
                                        send_scoped_event(
                                            &sender,
                                            child_session_id.as_deref(),
                                            agent_name.as_deref(),
                                            parent_tool_call_id.as_deref(),
                                            SessionTransportEvent::ToolBatchFinished,
                                        )?;
                                    }
                                }
                                AgentEvent::TodoSnapshotUpdated { items } => {
                                    send_scoped_event(
                                        &sender,
                                        child_session_id.as_deref(),
                                        agent_name.as_deref(),
                                        parent_tool_call_id.as_deref(),
                                        SessionTransportEvent::TodoSnapshot(TodoSnapshotEvent::new(items)),
                                    )?;
                                }
                                AgentEvent::AutoContinueChanged { state } => {
                                    send_scoped_event(
                                        &sender,
                                        child_session_id.as_deref(),
                                        agent_name.as_deref(),
                                        parent_tool_call_id.as_deref(),
                                        SessionTransportEvent::AutoContinueChanged(
                                            AutoContinueChangedEvent::new(state),
                                        ),
                                    )?;
                                }
                                AgentEvent::AutoContinuationScheduled {
                                    ..
                                } => {}
                                AgentEvent::ValidationAdvisory(_) => {}
                                AgentEvent::ToolCallCancelled { call_id, name } => {
                                    send_scoped_event(
                                        &sender,
                                        child_session_id.as_deref(),
                                        agent_name.as_deref(),
                                        parent_tool_call_id.as_deref(),
                                        SessionTransportEvent::ToolCancelled(ToolCancelledEvent::new(
                                            call_id, name,
                                        )),
                                    )?;
                                }
                                AgentEvent::ToolExecutionSummary(_) => {}
                                AgentEvent::ContextCompacted(event) => {
                                    // Recorder success is the compaction acknowledgement;
                                    // presentation delivery cannot roll it back.
                                    let _ = emit_context_projection_updates(
                                        &sender,
                                        &transcript,
                                        child_session_id.as_deref(),
                                        agent_name.as_deref(),
                                        parent_tool_call_id.as_deref(),
                                    );
                                    let _ = send_scoped_event(
                                        &sender,
                                        child_session_id.as_deref(),
                                        agent_name.as_deref(),
                                        parent_tool_call_id.as_deref(),
                                        SessionTransportEvent::CompactionCommitted {
                                            summary: Some(event.summary.clone()),
                                        },
                                    );
                                }
                                AgentEvent::TurnFinalized(_) => {}
                            }

                            Ok(())
                        }
                    }
                },
                {
                    let sender = self.event_tx.clone();
                    let permission_sender = self.permission_event_tx.clone();
                    let transcript = self.transcript.clone();
                    let permission_origin = self.permission_origin.clone();
                    let child_session_id = self.child_session_id.clone();
                    let parent_tool_call_id = self.parent_tool_call_id.clone();
                    let event_mode = self.event_mode;
                    move |request| {
                        let sender = sender.clone();
                        let permission_sender = permission_sender.clone();
                        let transcript = transcript.clone();
                        let permission_origin = permission_origin.clone();
                        let child_session_id = child_session_id.clone();
                        let parent_tool_call_id = parent_tool_call_id.clone();
                        let event_mode = event_mode;
                        async move {
                            // Permission decisions are not AgentEvent stream entries.
                            let request_event =
                                permission_request_event(&request, permission_origin.as_deref());
                            if matches!(event_mode, SessionTransportEventMode::SilentDenyPermissions) {
                                let resolution =
                                    permission_resolution_event(&request, PermissionResponse::Deny);
                                record_transcript(&transcript, |recorder| {
                                    recorder.record_permission_decision_details(
                                        request.call_id.clone(),
                                        request.tool.clone(),
                                        request.args.clone(),
                                        false,
                                        resolution.reason.clone(),
                                    )
                                })?;
                                return Ok(PermissionApproval::Deny);
                            }

                            let (response_tx, response_rx) = oneshot::channel();
                            let handle = RunnerPermissionRequest::new(response_tx);
                            let permission_target = permission_sender.clone().or(sender.clone());
                            send_optional_event(
                                &permission_target,
                                match child_session_id.clone() {
                                    Some(child_session_id) => {
                                        SessionTransportEvent::ChildPermissionRequested {
                                            child_session_id,
                                            agent_name: permission_origin.clone(),
                                            parent_tool_call_id: parent_tool_call_id.clone(),
                                            event: request_event.clone(),
                                            handle,
                                        }
                                    }
                                    None => SessionTransportEvent::PermissionRequested {
                                        event: request_event.clone(),
                                        handle,
                                    },
                                },
                            )?;

                            let response = response_rx
                                .await
                                .map_err(|_| anyhow!("permission response sender dropped"))?;
                            let resolution = permission_resolution_event(&request, response);
                            record_transcript(&transcript, |recorder| {
                                recorder.record_permission_decision_details(
                                    request.call_id.clone(),
                                    request.tool.clone(),
                                    request.args.clone(),
                                    response.allowed(),
                                    resolution.reason.clone(),
                                )
                            })?;
                            emit_context_projection_updates(
                                &permission_target,
                                &transcript,
                                child_session_id.as_deref(),
                                permission_origin.as_deref(),
                                parent_tool_call_id.as_deref(),
                            )?;
                            let permission_target = permission_sender.clone().or(sender.clone());
                            send_optional_event(
                                &permission_target,
                                match child_session_id.clone() {
                                    Some(child_session_id) => SessionTransportEvent::ChildSessionEvent {
                                        child_session_id,
                                        agent_name: permission_origin.clone(),
                                        parent_tool_call_id: parent_tool_call_id.clone(),
                                        event: SessionEvent::PermissionResolved(resolution),
                                    },
                                    None => SessionTransportEvent::PermissionResolved(resolution),
                                },
                            )?;

                            Ok(match response {
                                PermissionResponse::AllowOnce => PermissionApproval::AllowOnce,
                                PermissionResponse::AllowAlways if request.can_allow_always => PermissionApproval::AllowAlways,
                                PermissionResponse::AllowAlways | PermissionResponse::Deny => PermissionApproval::Deny,
                            })
                        }
                    }
                },
                {
                    let sender = self.event_tx.clone();
                    let permission_sender = self.permission_event_tx.clone();
                    let child_session_id = self.child_session_id.clone();
                    let event_mode = self.event_mode;
                    move |request| {
                        let sender = sender.clone();
                        let permission_sender = permission_sender.clone();
                        let child_session_id = child_session_id.clone();
                        let event_mode = event_mode;
                        async move {
                            if matches!(event_mode, SessionTransportEventMode::SilentDenyPermissions) {
                                return Err(anyhow!(
                                    "question tool is unavailable while this runtime is auto-denying interactive requests"
                                ));
                            }

                            let (response_tx, response_rx) = oneshot::channel();
                            let handle = RunnerQuestionRequest::new(response_tx);
                            let target = permission_sender.clone().or(sender.clone());
                            send_optional_event(
                                &target,
                                match child_session_id.clone() {
                                    Some(child_session_id) => SessionTransportEvent::ChildQuestionRequested {
                                        child_session_id,
                                        request: request.clone(),
                                        handle,
                                    },
                                    None => SessionTransportEvent::QuestionRequested {
                                        request: request.clone(),
                                        handle,
                                    },
                                },
                            )?;

                            match response_rx
                                .await
                                .map_err(|_| anyhow!("question response sender dropped"))?
                            {
                                Ok(response) => Ok(response),
                                Err(message) => Err(anyhow!(message)),
                            }
                        }
                    }
                },
            )
            .await;

        match response {
            Ok(message) => {
                emit_context_projection_updates(
                    &self.event_tx,
                    &self.transcript,
                    self.child_session_id.as_deref(),
                    self.permission_origin.as_deref(),
                    self.parent_tool_call_id.as_deref(),
                )
                .or_else(|error| self.finish_with_error(error))?;
                self.emit(SessionTransportEvent::AssistantDone { message_id: None })?;
                self.emit(SessionTransportEvent::Done)?;
                Ok(message)
            }
            Err(error) => {
                let error_message = format!("{error:#}");
                let event = ErrorEvent::new(error_message.clone());
                if let Err(record_error) =
                    self.record(|recorder| recorder.record_error(error_message.clone()))
                {
                    let composite_message = format!(
                        "{} (additionally failed to record transcript error: {})",
                        error_message, record_error
                    );
                    self.finish_with_error(anyhow!(composite_message.clone()))?;
                    return Err(anyhow!(composite_message));
                }
                if let Err(projection_error) = emit_context_projection_updates(
                    &self.event_tx,
                    &self.transcript,
                    self.child_session_id.as_deref(),
                    self.permission_origin.as_deref(),
                    self.parent_tool_call_id.as_deref(),
                ) {
                    let composite = anyhow!(
                        "{} (additionally failed context projection: {})",
                        error_message,
                        projection_error
                    );
                    self.finish_with_error(composite)?;
                }
                self.emit(SessionTransportEvent::Error(event))?;
                self.emit(SessionTransportEvent::Done)?;
                Err(error)
            }
        }
    }

    #[cfg(test)]
    fn emit_session_title_updated(&self, session_id: String, title: String) -> Result<()> {
        send_optional_event(
            &self
                .session_title_event_tx
                .as_ref()
                .or(self.event_tx.as_ref())
                .cloned(),
            SessionTransportEvent::SessionTitleUpdated { session_id, title },
        )
    }

    #[cfg(test)]
    pub fn record_model_changed(&self, previous_model: &str, new_model: &str) -> Result<()> {
        self.record(|recorder| {
            recorder.record_model_changed(previous_model.to_string(), new_model.to_string())
        })
    }

    #[cfg(test)]
    pub fn record_permission_mode_changed(
        &self,
        previous_mode: &str,
        new_mode: &str,
    ) -> Result<()> {
        self.record(|recorder| {
            recorder.record_permission_mode_changed(previous_mode.to_string(), new_mode.to_string())
        })
    }

    fn emit(&self, event: SessionTransportEvent) -> Result<()> {
        let event = if let Some(child_session_id) = &self.child_session_id {
            wrap_child_session_transport_event(
                child_session_id.clone(),
                self.permission_origin.clone(),
                self.parent_tool_call_id.clone(),
                event,
            )
        } else {
            event
        };
        send_optional_event(&self.event_tx, event)
    }

    fn record<F>(&self, f: F) -> Result<()>
    where
        F: FnOnce(&mut TranscriptRecorder) -> Result<()>,
    {
        record_transcript(&self.transcript, f)
    }

    fn finish_with_error(&self, error: anyhow::Error) -> Result<()> {
        let event = ErrorEvent::new(format!("{error:#}"));
        self.emit(SessionTransportEvent::Error(event))?;
        self.emit(SessionTransportEvent::Done)?;
        Err(error)
    }

    fn pending_session_title(
        &self,
        agent: &Agent<C>,
        record_user_prompt: bool,
    ) -> Result<Option<(String, Agent<C>)>>
    where
        C: Clone,
    {
        if !record_user_prompt || self.child_session_id.is_some() {
            return Ok(None);
        }
        let Some(transcript) = &self.transcript else {
            return Ok(None);
        };

        let (session_id, path) = {
            let recorder = transcript
                .lock()
                .map_err(|_| anyhow!("transcript recorder poisoned"))?;
            (
                recorder.session_id().to_string(),
                recorder.path().to_path_buf(),
            )
        };
        let records = read_records(&path)?;
        if transcript_has_user_message(&records) || transcript_has_session_title(&records) {
            return Ok(None);
        }

        Ok(Some((session_id, agent.session_title_agent())))
    }
}

pub(crate) fn subagent_event_sender(
    event_tx: SessionTransportEventSender,
) -> SubagentEventSender<async_openai::config::OpenAIConfig> {
    let status_tx = event_tx.clone();
    let error_tx = event_tx.clone();
    SubagentEventSender::new(
        Arc::new(move |message| {
            status_tx
                .send(SessionTransportEvent::Notice(NoticeEvent::info(message)))
                .map_err(|_| anyhow!("runner event channel closed"))
        }),
        Arc::new(move |message| {
            error_tx
                .send(SessionTransportEvent::Error(ErrorEvent::new(message)))
                .map_err(|_| anyhow!("runner event channel closed"))
        }),
        Arc::new(
            move |agent,
                  prompt,
                  transcript,
                  child_session_id,
                  permission_origin,
                  parent_tool_call_id| {
                let runner: AgentRunner<async_openai::config::OpenAIConfig> =
                    if let Some(permission_origin) = permission_origin {
                        AgentRunner::child_streaming_with_permission_passthrough(
                            transcript,
                            event_tx.clone(),
                            child_session_id,
                            permission_origin,
                            parent_tool_call_id,
                        )
                    } else {
                        AgentRunner::child_streaming_with_transcript(
                            transcript,
                            event_tx.clone(),
                            child_session_id,
                        )
                    };
                Box::pin(async move {
                    let mut agent = agent;
                    runner
                        .run_prompt(
                            &mut agent,
                            UserMessageSubmission::new(
                                "child-stream-prompt",
                                UserMessageContent::new(prompt, Vec::new()),
                            ),
                        )
                        .await
                })
            },
        ),
    )
}

fn wrap_child_session_transport_event(
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
        | SessionTransportEvent::ExpertModelChanged { .. } => event,
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

fn send_scoped_event(
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

fn emit_context_projection_updates(
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

fn emit_context_projection_update(
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

fn send_optional_event(
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

fn record_transcript<F>(transcript: &Option<Arc<Mutex<TranscriptRecorder>>>, f: F) -> Result<()>
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

fn retry_lifecycle_event(retry: LlmRetryLifecycle) -> RetryLifecycleEvent {
    RetryLifecycleEvent {
        attempt: retry.attempt,
        max_attempts: retry.max_attempts,
        delay_secs: retry.delay_secs,
        error: retry.error,
    }
}

fn tool_started_event(call_id: String, name: String, args: Value) -> ToolStartedEvent {
    let summary = format_tool_call(&name, &args);
    ToolStartedEvent {
        call_id,
        name,
        summary,
        arguments: Some(args.to_string()),
    }
}

fn tool_finished_event(
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

fn permission_request_event(
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

fn permission_resolution_event(
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
        },
        PermissionResponse::AllowAlways => PermissionResolutionEvent {
            call_id,
            decision: crate::session::PermissionDecision::Approved,
            reason: Some("Allowed for this session".into()),
            tool_name: None,
            summary: None,
            origin_label: None,
        },
        PermissionResponse::Deny => {
            PermissionResolutionEvent::denied(call_id, Some("Denied".into()))
        }
    }
}

fn output_summary(output: &ToolResult) -> Option<String> {
    if let Some(error) = &output.error {
        return Some(error.message.clone());
    }

    let data = output.data.as_ref()?;
    Some(match output.tool.as_str() {
        tool_names::TOOL_UTIL_ECHO => summarize_echo(data),
        tool_names::TOOL_FS_LIST => summarize_array_count(data, "entries", "entries"),
        tool_names::TOOL_FS_READ => summarize_read_file(data),
        tool_names::TOOL_FS_WRITE => summarize_bytes(data, "bytes_written", "wrote"),
        tool_names::TOOL_FS_APPEND => summarize_bytes(data, "bytes_appended", "appended"),
        tool_names::TOOL_FS_MKDIR => summarize_path_action(data, "created"),
        tool_names::TOOL_SEARCH_RG => summarize_array_count(data, "matches", "matches"),
        tool_names::TOOL_WEB_FETCH => summarize_web_fetch(data),
        tool_names::TOOL_SHELL_EXEC
        | tool_names::TOOL_GIT_STATUS
        | tool_names::TOOL_GIT_DIFF
        | tool_names::TOOL_GIT_LOG => summarize_command(data),
        tool_names::TOOL_EDIT_APPLY_PATCH => summarize_apply_patch(data),
        tool_names::TOOL_CODE_AST_SEARCH => summarize_array_count(data, "matches", "matches"),
        tool_names::TOOL_CODE_AST_REPLACE_PREVIEW => {
            summarize_array_count(data, "replacements", "replacements")
        }
        tool_names::TOOL_WORKFLOW_TODOS => summarize_todos(data),
        tool_names::TOOL_WORKFLOW_AUTO_CONTINUE => summarize_auto_continue(data),
        name if is_subagent_tool_name(name) => summarize_subagent_tool(data),
        _ => summarize_generic(data),
    })
}

fn summarize_subagent_tool(data: &Value) -> String {
    let agent_name = data
        .get("agent_name")
        .and_then(Value::as_str)
        .unwrap_or("subagent");
    let status = data
        .get("status")
        .and_then(Value::as_str)
        .unwrap_or("completed");
    let child = data
        .get("child_session_id")
        .and_then(Value::as_str)
        .map(|id| id.get(..12).unwrap_or(id))
        .unwrap_or("child");
    let flags = summarize_subagent_flags(data);
    if flags.is_empty() {
        format!("{agent_name} {status} · {child}")
    } else {
        format!("{agent_name} {status} · {} · {child}", flags.join("/"))
    }
}

fn summarize_subagent_flags(data: &Value) -> Vec<&'static str> {
    let mut flags = Vec::new();
    if data.get("active").and_then(Value::as_bool) == Some(true) {
        flags.push("active");
    }
    if data.get("unreconciled").and_then(Value::as_bool) == Some(true) {
        flags.push("unreconciled");
    }
    if data.get("reconciled").and_then(Value::as_bool) == Some(true) {
        flags.push("reconciled");
    }
    if data.get("reusable").and_then(Value::as_bool) == Some(true) {
        flags.push("reusable");
    }
    if data
        .get("structured_result")
        .and_then(|value| value.get("malformed"))
        .and_then(Value::as_bool)
        == Some(true)
    {
        flags.push("malformed");
    }
    flags
}

fn compact_subagent_summary(summary: &str) -> String {
    let single_line = summary.split_whitespace().collect::<Vec<_>>().join(" ");
    if single_line.chars().count() <= 160 {
        return single_line;
    }
    let mut truncated = single_line.chars().take(160).collect::<String>();
    truncated.push('…');
    truncated
}

fn summarize_todos(data: &Value) -> String {
    let count = data
        .get("items")
        .and_then(Value::as_array)
        .map(Vec::len)
        .unwrap_or(0);
    format!("updated {count} todos")
}

fn summarize_auto_continue(data: &Value) -> String {
    let enabled = data
        .get("enabled")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if enabled {
        "enabled auto-continue".into()
    } else {
        "disabled auto-continue".into()
    }
}

fn summarize_echo(data: &Value) -> String {
    let chars = data
        .get("result")
        .and_then(Value::as_str)
        .map(str::chars)
        .map(Iterator::count)
        .unwrap_or(0);
    format!("returned {chars} chars")
}

fn summarize_array_count(data: &Value, key: &str, label: &str) -> String {
    let count = data
        .get(key)
        .and_then(Value::as_array)
        .map(Vec::len)
        .unwrap_or(0);
    let truncated = data
        .get("truncated")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if truncated {
        format!("{count} {label} shown · truncated")
    } else {
        format!("{count} {label}")
    }
}

fn summarize_web_fetch(data: &Value) -> String {
    let status = data.get("status").and_then(Value::as_u64).unwrap_or(0);
    let bytes = data
        .get("content_bytes")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let redirects = data
        .get("redirects")
        .and_then(Value::as_array)
        .map(Vec::len)
        .unwrap_or(0);
    let truncated = data
        .get("truncated")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let redirect_suffix = if redirects > 0 {
        format!(" · {redirects} redirects")
    } else {
        String::new()
    };
    let truncation_suffix = if truncated { " · truncated" } else { "" };
    format!("HTTP {status} · {bytes} bytes{redirect_suffix}{truncation_suffix}")
}

fn summarize_read_file(data: &Value) -> String {
    let path = data.get("path").and_then(Value::as_str).unwrap_or("file");
    let lines = data.get("lines_read").and_then(Value::as_u64).unwrap_or(0);
    let start = data.get("start_line").and_then(Value::as_u64);
    let end = data.get("end_line").and_then(Value::as_u64);
    let suffix = if data
        .get("truncated")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        " · has more"
    } else {
        ""
    };

    match (start, end) {
        (Some(start), Some(end)) => format!("read {path}:{start}-{end} ({lines} lines){suffix}"),
        _ => format!("read {path} ({lines} lines){suffix}"),
    }
}

fn summarize_bytes(data: &Value, key: &str, verb: &str) -> String {
    let path = data.get("path").and_then(Value::as_str).unwrap_or("file");
    let bytes = data.get(key).and_then(Value::as_u64).unwrap_or(0);
    format!("{verb} {bytes} bytes to {path}")
}

fn summarize_path_action(data: &Value, action: &str) -> String {
    let path = data.get("path").and_then(Value::as_str).unwrap_or("path");
    format!("{action} {path}")
}

fn summarize_command(data: &Value) -> String {
    if let Some(error) = data.get("error").and_then(Value::as_str) {
        return error.to_string();
    }

    let status = data
        .get("status")
        .and_then(Value::as_i64)
        .map(|status| format!("exit {status}"))
        .unwrap_or_else(|| "completed".to_string());
    let stdout = output_line_count(data, "stdout", "stdout_truncated");
    let stderr = output_line_count(data, "stderr", "stderr_truncated");
    let mut parts = vec![status];
    if let Some(stdout) = stdout {
        parts.push(format!("stdout {stdout}"));
    }
    if let Some(stderr) = stderr {
        parts.push(format!("stderr {stderr}"));
    }
    parts.join(" · ")
}

fn output_line_count(data: &Value, key: &str, truncated_key: &str) -> Option<String> {
    let text = data.get(key).and_then(Value::as_str)?;
    if text.trim().is_empty() {
        return None;
    }
    let count = text.lines().count().max(1);
    let suffix = if data
        .get(truncated_key)
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        "+"
    } else {
        ""
    };
    Some(format!("{count}{suffix} lines"))
}

fn summarize_apply_patch(data: &Value) -> String {
    let files = data
        .get("files_changed")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let edits = data
        .get("edits_applied")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    format!("patched {files} files · {edits} edits")
}

fn summarize_generic(data: &Value) -> String {
    match data {
        Value::Array(items) => format!("{} items", items.len()),
        Value::Object(fields) => format!("{} fields", fields.len()),
        Value::String(text) => format!("returned {} chars", text.chars().count()),
        Value::Null => "completed".into(),
        _ => "completed".into(),
    }
}

fn output_json(output: &ToolResult) -> Value {
    serde_json::to_value(output)
        .unwrap_or_else(|_| Value::String("<unserializable tool output>".into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{AutoContinueState, CacheUsageReport, TodoItem, TodoStatus};
    use crate::session::{PermissionDecision, SessionEvent};
    use crate::transcript::TranscriptRecorder;
    use async_openai::{Client, config::OpenAIConfig};
    use serde_json::json;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[tokio::test]
    async fn permission_request_handle_delivers_approval() {
        let (tx, rx) = oneshot::channel();
        let handle = RunnerPermissionRequest::new(tx);

        handle.approve().expect("approve succeeds");

        assert_eq!(
            rx.await.expect("receiver gets response"),
            PermissionResponse::AllowOnce
        );
    }

    #[test]
    fn permission_resolution_event_maps_denial() {
        let request = PermissionRequest {
            call_id: Some("call-7".into()),
            tool: "shell__exec".into(),
            args: Value::Null,
            class: crate::permission::ToolPermissionClass::Command,
            summary: "shell__exec cargo test".into(),
            preview: None,
            can_allow_always: false,
            grant_summary: None,
        };

        let resolution = permission_resolution_event(&request, PermissionResponse::Deny);

        assert_eq!(resolution.call_id, "call-7");
        assert_eq!(resolution.decision, PermissionDecision::Denied);
        assert!(resolution.reason.is_some());
    }

    #[test]
    fn token_usage_session_event_preserves_cache_report() {
        let report = CacheUsageReport {
            configured: true,
            hint_serialized: true,
            retention_sent: None,
            stable_prefix_segments: 1,
            stable_prompt_tokens: 100,
            volatile_prompt_tokens: 20,
            cacheable_prefix_tokens: 80,
            stable_after_boundary_tokens: 20,
            local_prefix_fingerprint: Some("prefix".into()),
            routing_key: Some("route".into()),
            actual_cached_tokens: Some(80),
        };
        let event = SessionTransportEvent::TokenUsage(
            TokenUsageEvent::with_breakdown(120, 1_000, 100, 20, 80)
                .with_cache_report(Some(report.clone())),
        );

        assert_eq!(
            event.session_event(),
            Some(SessionEvent::TokenUsage(
                TokenUsageEvent::with_breakdown(120, 1_000, 100, 20, 80)
                    .with_cache_report(Some(report)),
            ))
        );
    }

    #[test]
    fn mcp_diagnostic_does_not_map_to_an_session_event() {
        assert_eq!(
            SessionTransportEvent::McpDiagnostic("MCP server 'docs' is offline".into())
                .session_event(),
            None
        );
    }

    #[test]
    fn compaction_preview_maps_to_a_dedicated_session_event_and_preserves_child_scope() {
        let event = SessionTransportEvent::CompactionPreviewDelta {
            delta: "summary chunk".into(),
        };
        assert_eq!(
            event.session_event(),
            Some(SessionEvent::CompactionPreviewDelta {
                delta: "summary chunk".into(),
            })
        );
        assert!(matches!(
            wrap_child_session_transport_event("child-1".into(), None, None, event),
            SessionTransportEvent::ChildSessionEvent { child_session_id, agent_name: _, event: SessionEvent::CompactionPreviewDelta { delta }, .. }
                if child_session_id == "child-1" && delta == "summary chunk"
        ));
    }

    #[test]
    fn permission_request_event_carries_subagent_origin() {
        let request = PermissionRequest {
            call_id: Some("call-8".into()),
            tool: "shell__exec".into(),
            args: json!({"command": "cargo test"}),
            class: crate::permission::ToolPermissionClass::Command,
            summary: "shell__exec cargo test".into(),
            preview: None,
            can_allow_always: false,
            grant_summary: None,
        };

        let event = permission_request_event(&request, Some("fixer"));

        assert_eq!(event.call_id, "call-8");
        assert_eq!(event.tool_name, "shell__exec");
        assert_eq!(event.origin_label.as_deref(), Some("fixer"));
    }

    #[test]
    fn tool_output_summary_avoids_dumping_json_payloads() {
        let output = ToolResult::ok(
            "util__echo",
            serde_json::json!({ "result": "已调用工具。" }),
        );

        assert_eq!(output_summary(&output).as_deref(), Some("returned 6 chars"));

        let read = ToolResult::ok(
            "fs__read",
            serde_json::json!({
                "path": "src/main.rs",
                "start_line": 10,
                "end_line": 20,
                "lines_read": 11,
                "truncated": true
            }),
        );

        assert_eq!(
            output_summary(&read).as_deref(),
            Some("read src/main.rs:10-20 (11 lines) · has more")
        );
    }

    #[test]
    fn command_summary_reports_counts_not_output_text() {
        let output = ToolResult::ok(
            "shell__exec",
            serde_json::json!({
                "command": "cargo test",
                "status": 0,
                "success": true,
                "stdout": "line one\nline two\n",
                "stdout_truncated": false,
                "stderr": "warning\n",
                "stderr_truncated": true
            }),
        );

        assert_eq!(
            output_summary(&output).as_deref(),
            Some("exit 0 · stdout 2 lines · stderr 1+ lines")
        );
    }

    #[test]
    fn workflow_control_tools_have_compact_summaries() {
        let todos = ToolResult::ok(
            "workflow__todos",
            serde_json::json!({
                "items": [
                    {"id": "t1", "content": "one", "status": "pending"},
                    {"id": "t2", "content": "two", "status": "completed"}
                ]
            }),
        );
        assert_eq!(output_summary(&todos).as_deref(), Some("updated 2 todos"));

        let auto_continue = ToolResult::ok(
            "workflow__auto_continue",
            serde_json::json!({"enabled": true}),
        );
        assert_eq!(
            output_summary(&auto_continue).as_deref(),
            Some("enabled auto-continue")
        );
    }

    #[test]
    fn agent_explore_summary_is_compacted_for_tool_output() {
        let long = "word ".repeat(80);
        let output = ToolResult::ok(
            "agent__explore",
            serde_json::json!({
                "status": "completed",
                "child_session_id": "child-session-1234567890",
                "agent_name": "explorer",
                "summary": compact_subagent_summary(&long),
                "full_summary": long,
            }),
        );

        assert_eq!(
            output_summary(&output).as_deref(),
            Some("explorer completed · child-sessio"),
        );
    }

    #[test]
    fn compact_subagent_summary_collapses_newlines_and_truncates() {
        let summary = format!("first line\n\n{}", "detail ".repeat(40));

        let compact = compact_subagent_summary(&summary);

        assert!(!compact.contains('\n'));
        assert!(compact.starts_with("first line detail"), "{compact}");
        assert!(compact.chars().count() <= 161, "{compact}");
    }

    #[test]
    fn tool_driven_expert_delegation_requires_its_route_credential() {
        let delegate = RunnerSubagentDelegate {
            runtime: SubagentPool::new(),
            sessions_dir: std::env::temp_dir(),
            transcript: temp_transcript(),
            event_tx: None,
            expert_model_routes: indexmap::IndexMap::from([(
                "explorer".into(),
                crate::config::ModelRoute::new("expert", "shared"),
            )]),
            route_api_key_configured: indexmap::IndexMap::from([
                ("primary/shared".into(), true),
                ("expert/shared".into(), false),
            ]),
            provider_api_key_hints: indexmap::IndexMap::from([(
                "expert".into(),
                "Set EXPERT_API_KEY.".into(),
            )]),
            api_key_hint: "Set <PROVIDER>_API_KEY.".into(),
        };
        let mut parent = Agent::new(Client::with_config(OpenAIConfig::new()), "shared", 1, 1);
        parent.set_primary_route(crate::config::ModelRoute::new("primary", "shared"));
        let invocation = SubagentInvocation {
            input: crate::tool::NormalizedSubagentInput {
                objective: "inspect route credentials".into(),
                success_criteria: Vec::new(),
                allowed_paths: Vec::new(),
                forbidden_paths: Vec::new(),
                owned_paths: Vec::new(),
                timeout_secs: None,
                max_tool_calls: None,
                target_child_session_id: None,
            },
            prompt: "inspect route credentials".into(),
            parent_tool_call_id: Some("call-1".into()),
        };

        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("create runtime");
        let output = runtime.block_on(delegate.run_named(&parent, "explorer", invocation));

        let output = output.expect("credential denial is a tool result");
        assert!(!output.ok);
        assert!(matches!(
            output.error.as_ref(),
            Some(crate::tool::ToolError { message, .. })
                if message == "API key is not set for the selected provider. Set EXPERT_API_KEY."
        ));
        let data = output
            .data
            .expect("credential denial includes route metadata");
        assert_eq!(data.get("route"), Some(&json!("expert/shared")));
        assert_eq!(data.get("agent_name"), Some(&json!("explorer")));
    }

    #[test]
    fn agent_fixer_summary_uses_agent_name() {
        let output = ToolResult::ok(
            "agent__fixer",
            serde_json::json!({
                "agent_name": "fixer",
                "status": "completed",
                "child_session_id": "child-session-1234567890",
                "summary": "applied change"
            }),
        );

        assert_eq!(
            output_summary(&output).as_deref(),
            Some("fixer completed · child-sessio")
        );
    }

    #[test]
    fn subagent_summary_includes_compact_governance_and_reconciliation_flags() {
        let output = ToolResult::ok(
            "agent__fixer",
            serde_json::json!({
                "agent_name": "fixer",
                "status": "budget_exhausted",
                "child_session_id": "child-session-1234567890",
                "summary": "tool budget hit",
                "unreconciled": true,
                "structured_result": {
                    "status": "budget_exhausted",
                    "summary": "tool budget hit",
                    "malformed": true
                }
            }),
        );

        assert_eq!(
            output_summary(&output).as_deref(),
            Some("fixer budget_exhausted · unreconciled/malformed · child-sessio")
        );
    }

    #[test]
    fn readonly_expert_subagent_summary_uses_generic_status_path() {
        let output = ToolResult::ok(
            "agent__oracle",
            serde_json::json!({
                "agent_name": "oracle",
                "status": "completed",
                "child_session_id": "child-session-1234567890",
                "summary": "root cause analyzed"
            }),
        );

        assert_eq!(
            output_summary(&output).as_deref(),
            Some("oracle completed · child-sessio")
        );
    }

    #[test]
    fn scoped_child_tool_event_preserves_agent_name() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        send_scoped_event(
            &Some(tx),
            Some("explorer-child"),
            Some("explorer"),
            Some("parent-call"),
            SessionTransportEvent::ToolStarted(ToolStartedEvent::new(
                "child-call",
                "search__rg",
                "search source",
            )),
        )
        .expect("send scoped event");

        assert!(matches!(
            rx.try_recv().expect("child event"),
            SessionTransportEvent::ChildSessionEvent {
                child_session_id,
                agent_name,
                parent_tool_call_id,
                event: SessionEvent::ToolStarted(ToolStartedEvent { call_id, .. }),
            } if child_session_id == "explorer-child"
                && agent_name.as_deref() == Some("explorer")
                && parent_tool_call_id.as_deref() == Some("parent-call")
                && call_id == "child-call"
        ));
    }

    #[test]
    fn child_session_title_updates_remain_parent_scoped() {
        let wrapped = wrap_child_session_transport_event(
            "child-session".into(),
            Some("explorer".into()),
            None,
            SessionTransportEvent::SessionTitleUpdated {
                session_id: "parent-session".into(),
                title: "Parent title".into(),
            },
        );

        assert!(matches!(
            wrapped,
            SessionTransportEvent::SessionTitleUpdated { session_id, title }
                if session_id == "parent-session" && title == "Parent title"
        ));
    }

    #[test]
    fn child_streaming_runner_wraps_session_events_with_child_session_id() {
        let (tx, _rx) = mpsc::unbounded_channel();
        let _runner = AgentRunner::<OpenAIConfig>::child_streaming_with_transcript(
            temp_transcript(),
            tx,
            "child-session",
        );

        let wrapped = wrap_child_session_transport_event(
            "child-session".into(),
            Some("explorer".into()),
            None,
            SessionTransportEvent::AssistantDelta(AssistantDeltaEvent::new("hi")),
        );

        assert!(matches!(
            wrapped,
            SessionTransportEvent::ChildSessionEvent { child_session_id, agent_name, event: SessionEvent::AssistantDelta(delta), .. }
                if child_session_id == "child-session"
                    && agent_name.as_deref() == Some("explorer")
                    && delta.delta == "hi"
        ));

        let wrapped_notice = wrap_child_session_transport_event(
            "child-session".into(),
            Some("explorer".into()),
            None,
            SessionTransportEvent::Notice(NoticeEvent::info("child notice")),
        );

        assert!(matches!(
            wrapped_notice,
            SessionTransportEvent::ChildSessionEvent { child_session_id, agent_name, event: SessionEvent::Notice(notice), .. }
                if child_session_id == "child-session"
                    && agent_name.as_deref() == Some("explorer")
                    && notice.message == "child notice"
        ));
    }

    #[tokio::test]
    async fn transcript_failure_emits_error_and_done() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let runner = AgentRunner::with_transcript(tx, poisoned_transcript());
        let config = OpenAIConfig::new()
            .with_api_base("https://api.openai.com/v1")
            .with_api_key("test-key");
        let client = Client::with_config(config);
        let mut agent = Agent::new(client, "gpt-5.5", 1, 1);
        let prompt = UserMessageSubmission::new(
            "runner-test",
            UserMessageContent::new(
                "hello",
                vec![crate::user_content::UserImageAttachment {
                    id: "img-1".into(),
                    label: "screen.png".into(),
                    mime: "image/png".into(),
                    data_url: "data:image/png;base64,AAAA".into(),
                }],
            ),
        );

        let error = runner
            .run_prompt(&mut agent, prompt)
            .await
            .expect_err("transcript failure should error");

        assert!(error.to_string().contains("transcript recorder poisoned"));

        assert!(matches!(
            rx.recv().await,
            Some(SessionTransportEvent::UserMessage(UserMessageEvent { content, .. }))
                if content.text == "hello"
                    && content.attachments.len() == 1
                    && content.attachments[0].label == "screen.png"
        ));
        assert!(matches!(
            rx.recv().await,
            Some(SessionTransportEvent::Error(ErrorEvent { message, .. })) if message.contains("transcript recorder poisoned")
        ));
        assert!(matches!(rx.recv().await, Some(SessionTransportEvent::Done)));
    }

    #[test]
    fn session_title_update_uses_dedicated_session_event_sender() {
        let (turn_tx, mut turn_rx) = mpsc::unbounded_channel();
        let (title_tx, mut title_rx) = mpsc::unbounded_channel();
        let runner =
            AgentRunner::<OpenAIConfig>::new(turn_tx).with_session_title_event_sender(title_tx);

        runner
            .emit_session_title_updated("session-1".into(), "Session title".into())
            .expect("title update should emit");

        assert!(matches!(
            title_rx.try_recv(),
            Ok(SessionTransportEvent::SessionTitleUpdated { session_id, title })
                if session_id == "session-1" && title == "Session title"
        ));
        assert!(turn_rx.try_recv().is_err());
    }

    #[test]
    fn runner_can_record_model_and_permission_provenance_events() {
        let base_dir = std::env::temp_dir().join(format!(
            "letcode-runner-provenance-test-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("time ok")
                .as_nanos()
        ));
        let recorder = TranscriptRecorder::create(&base_dir).expect("create transcript");
        let session_id = recorder.session_id().to_string();
        let transcript = Arc::new(Mutex::new(recorder));
        let (tx, _rx) = mpsc::unbounded_channel();
        let runner = AgentRunner::<OpenAIConfig>::with_transcript(tx, transcript);

        runner
            .record_model_changed("gpt-5.5", "gpt-5.5-mini")
            .expect("record model");
        runner
            .record_permission_mode_changed("default", "safe")
            .expect("record permission");

        let records = crate::transcript::read_records(base_dir.join(format!("{session_id}.jsonl")))
            .expect("read records");
        assert_eq!(records.len(), 2);
        let first = serde_json::to_value(&records[0]).expect("serialize");
        assert_eq!(first.get("kind"), Some(&json!("model_changed")));
        let second = serde_json::to_value(&records[1]).expect("serialize");
        assert_eq!(second.get("kind"), Some(&json!("permission_mode_changed")));
    }

    fn temp_transcript() -> Arc<Mutex<TranscriptRecorder>> {
        let base_dir = std::env::temp_dir().join(format!(
            "letcode-runner-child-streaming-test-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("time ok")
                .as_nanos()
        ));
        std::fs::create_dir_all(&base_dir).expect("temp dir created");
        Arc::new(Mutex::new(
            TranscriptRecorder::create(&base_dir).expect("transcript created"),
        ))
    }

    #[test]
    fn todo_session_transport_events_map_to_session_events() {
        let pending_event = SessionTransportEvent::ToolPending(ToolPendingEvent::new(
            "call-pending",
            "edit__apply_patch",
        ));
        assert!(matches!(
            pending_event.session_event(),
            Some(SessionEvent::ToolPending(_))
        ));

        let todo_event =
            SessionTransportEvent::TodoSnapshot(TodoSnapshotEvent::new(vec![TodoItem {
                id: "t1".into(),
                content: "inspect".into(),
                status: TodoStatus::Pending,
            }]));
        assert!(matches!(
            todo_event.session_event(),
            Some(SessionEvent::TodoSnapshot(_))
        ));

        let auto_event = SessionTransportEvent::AutoContinueChanged(AutoContinueChangedEvent::new(
            AutoContinueState { enabled: true },
        ));
        assert!(matches!(
            auto_event.session_event(),
            Some(SessionEvent::AutoContinueChanged(_))
        ));
    }

    #[tokio::test]
    async fn request_preparation_auto_disable_projects_fast_mode_state_and_notice() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("test port should bind");
        let address = listener.local_addr().expect("test listener has an address");
        drop(listener);
        let fast_mode_dir = std::env::temp_dir().join(format!(
            "letcode-runner-fast-mode-test-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("time ok")
                .as_nanos()
        ));
        std::fs::create_dir_all(&fast_mode_dir).expect("create Fast Mode config directory");
        let fast_mode_path = fast_mode_dir.join("letcode.toml");
        std::fs::write(
            &fast_mode_path,
            r#"active_provider = "primary"

[providers.primary]
base_url = "https://primary.invalid/v1"
api_key = "primary-key"
protocol = "responses"
[providers.primary.models."gpt-5.5"]
"#,
        )
        .expect("write Fast Mode config");
        let fast_mode = crate::fast_mode::FastMode::load(fast_mode_path, true);

        let (tx, mut rx) = mpsc::unbounded_channel();
        let runner = AgentRunner::new(tx);
        let client = Client::with_config(
            OpenAIConfig::new()
                .with_api_base(format!("http://{address}"))
                .with_api_key("test-key"),
        );
        let mut agent = Agent::new(client, "claude-4", 1, 1);
        agent.set_fast_mode(fast_mode);

        let _ = runner
            .run_prompt(
                &mut agent,
                UserMessageSubmission::new(
                    "runner-fast-mode-test",
                    UserMessageContent::new("hello", Vec::new()),
                ),
            )
            .await;

        let events = std::iter::from_fn(|| rx.try_recv().ok()).collect::<Vec<_>>();
        assert!(events.iter().any(|event| matches!(
            event,
            SessionTransportEvent::FastModeChanged { enabled: false }
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            SessionTransportEvent::Notice(notice)
                if notice.message == "Fast mode auto-disabled: current model is unavailable"
        )));
        assert!(!agent.fast_mode_enabled(), "auto-disable must persist");
    }

    #[tokio::test]
    async fn internal_prompt_does_not_emit_user_message_event() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let runner = AgentRunner::new(tx);
        let config = OpenAIConfig::new()
            .with_api_base("https://api.openai.com/v1")
            .with_api_key("test-key");
        let client = Client::with_config(config);
        let mut agent = Agent::new(client, "gpt-5.5", 1, 1);

        let _ = runner
            .run_internal_prompt(&mut agent, "continue internally")
            .await;

        while let Ok(event) = rx.try_recv() {
            assert!(!matches!(event, SessionTransportEvent::UserMessage(_)));
        }
    }

    fn poisoned_transcript() -> Arc<Mutex<TranscriptRecorder>> {
        let base_dir = std::env::temp_dir().join(format!(
            "letcode-group8-runner-test-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("time ok")
                .as_nanos()
        ));
        let recorder = TranscriptRecorder::create(&base_dir).expect("create transcript recorder");
        let transcript = Arc::new(Mutex::new(recorder));
        let cloned = Arc::clone(&transcript);

        let join = std::thread::spawn(move || {
            let _guard = cloned.lock().expect("lock transcript");
            panic!("poison transcript mutex for test");
        });
        let _ = join.join();

        transcript
    }

    #[test]
    fn session_event_projects_pure_lifecycle_signals() {
        use crate::session::SessionEvent;

        let token = TokenUsageEvent::new(10, 100);
        assert!(matches!(
            SessionTransportEvent::SessionTokenUsage(token).session_event(),
            Some(SessionEvent::SessionTokenUsage(_))
        ));
        assert!(matches!(
            SessionTransportEvent::ToolBatchFinished.session_event(),
            Some(SessionEvent::ToolBatchFinished)
        ));
        assert!(matches!(
            SessionTransportEvent::ContextBranchChanged {
                branch_id: "main".into()
            }
            .session_event(),
            Some(SessionEvent::ContextBranchChanged { branch_id }) if branch_id == "main"
        ));
        assert!(
            SessionTransportEvent::QueuedPromptAccepted {
                prompt: crate::user_content::UserMessageSubmission::from("queued")
            }
            .session_event()
            .is_none()
        );
    }
}
