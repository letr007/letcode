use std::path::PathBuf;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use anyhow::{Result, anyhow};
use async_openai::config::Config;
use serde_json::{Value, json};
use tokio::sync::{mpsc, oneshot};
use tracing::warn;

use crate::agent::{
    Agent, AgentEvent, ConversationMessage, SubagentDelegate, SubagentInvocation,
    is_subagent_tool_name, subagent_tool_name_for_agent_name,
};
use crate::agent_event_journal::{ContextProjection, JournalEffect, persist_agent_event};
use crate::permission::{PermissionApproval, PermissionRequest};
use crate::runtime_context::RuntimeActiveContext;
use crate::subagent::{SubagentRuntime, SubagentStatus};
use crate::subagent_events::SubagentEventSender;
use crate::tool::{QuestionRequest, QuestionResponse, ToolResult};
use crate::tool_format::format_tool_call;
use crate::tool_names;
use crate::transcript::transcript_projection::ContextBranchInfo;
use crate::transcript::{
    TranscriptRecord, TranscriptRecorder, read_records, transcript_has_session_title,
    transcript_has_user_message,
};
use crate::user_content::UserMessageContent;
use crate::user_content::UserMessageSubmission;

use super::events::{
    AppEvent, AssistantDeltaEvent, AutoContinueChangedEvent, ContextDetailOpenedEvent,
    ContextSummaryUpdatedEvent, ContextTreeUpdatedEvent, ContextViewUpdatedEvent, ErrorEvent,
    FoldedOutputsUpdatedEvent, NoticeEvent, PermissionRequestEvent, PermissionResolutionEvent,
    ProcessIssueEvent, ReasoningDeltaEvent, ReasoningDoneEvent, RuntimeContextDisposition,
    RuntimeContextUpdatedEvent, TodoSnapshotEvent, TokenUsageEvent, ToolCancelledEvent,
    ToolFinishedEvent, ToolOutcome, ToolOutputDeltaEvent, ToolPendingEvent, ToolStartedEvent,
    UserMessageEvent,
};
use super::timeline::{COMPACTION_MESSAGE_ID, COMPACTION_SEPARATOR_LABEL, compaction_separator};

pub type RunnerEventSender = mpsc::UnboundedSender<RunnerEvent>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RunnerEventMode {
    Emit,
    SilentDenyPermissions,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PermissionResponse {
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
pub struct RunnerPermissionRequest {
    sender: Arc<Mutex<Option<oneshot::Sender<PermissionResponse>>>>,
}

#[derive(Debug, Clone)]
pub struct RunnerQuestionRequest {
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
pub enum RunnerEvent {
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
    ToolBatchFinished,
    QueuedPromptAccepted {
        prompt: UserMessageSubmission,
    },
    TodoSnapshot(TodoSnapshotEvent),
    AutoContinueChanged(AutoContinueChangedEvent),
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
        event: PermissionRequestEvent,
        handle: RunnerPermissionRequest,
    },
    ChildQuestionRequested {
        child_session_id: String,
        request: QuestionRequest,
        handle: RunnerQuestionRequest,
    },
    ChildAppEvent {
        child_session_id: String,
        event: AppEvent,
    },
    PermissionResolved(PermissionResolutionEvent),
    ProcessIssue(ProcessIssueEvent),
    Notice(NoticeEvent),
    RuntimeContextUpdated(RuntimeContextUpdatedEvent),
    ContextTreeUpdated(ContextTreeUpdatedEvent),
    ContextViewUpdated(ContextViewUpdatedEvent),
    ContextDetailOpened(ContextDetailOpenedEvent),
    FoldedOutputsUpdated(FoldedOutputsUpdatedEvent),
    ContextSummaryUpdated(ContextSummaryUpdatedEvent),
    Status(String),
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
    ContextBranchChanged {
        branch_id: String,
    },
    ChildSessionViewed {
        parent_session_id: String,
        child_session_id: String,
        agent_name: String,
        index: usize,
        total: usize,
        records: Vec<TranscriptRecord>,
        runtime_context: RuntimeActiveContext,
    },
    ContextBranchesLoaded {
        branches: Vec<ContextBranchInfo>,
    },
    SessionStarted {
        session_id: String,
        records: Vec<TranscriptRecord>,
        runtime_context: RuntimeActiveContext,
    },
    Error(ErrorEvent),
    Done,
}

impl RunnerEvent {
    pub fn app_event(&self) -> Option<AppEvent> {
        match self {
            Self::UserMessage(event) => Some(AppEvent::UserMessage(event.clone())),
            Self::ReasoningDelta(event) => Some(AppEvent::ReasoningDelta(event.clone())),
            Self::ReasoningDone(event) => Some(AppEvent::ReasoningDone(event.clone())),
            Self::AssistantDelta(event) => Some(AppEvent::AssistantDelta(event.clone())),
            Self::AssistantDone { message_id } => Some(AppEvent::AssistantDone {
                message_id: message_id.clone(),
            }),
            Self::TokenUsage(event) => Some(AppEvent::TokenUsage(event.clone())),
            Self::ToolPending(event) => Some(AppEvent::ToolPending(event.clone())),
            Self::ToolCancelled(event) => Some(AppEvent::ToolCancelled(event.clone())),
            Self::ToolStarted(event) => Some(AppEvent::ToolStarted(event.clone())),
            Self::ToolFinished(event) => Some(AppEvent::ToolFinished(event.clone())),
            Self::ToolOutputDelta(event) => Some(AppEvent::ToolOutputDelta(event.clone())),
            Self::ToolBatchFinished => None,
            Self::QueuedPromptAccepted { .. } => None,
            Self::TodoSnapshot(event) => Some(AppEvent::TodoSnapshot(event.clone())),
            Self::AutoContinueChanged(event) => Some(AppEvent::AutoContinueChanged(event.clone())),
            Self::PermissionRequested { event, .. } => {
                Some(AppEvent::PermissionRequested(event.clone()))
            }
            Self::QuestionRequested { .. }
            | Self::ChildPermissionRequested { .. }
            | Self::ChildQuestionRequested { .. }
            | Self::ChildAppEvent { .. } => None,
            Self::PermissionResolved(event) => Some(AppEvent::PermissionResolved(event.clone())),
            Self::ProcessIssue(event) => Some(AppEvent::ProcessIssue(event.clone())),
            Self::Notice(event) => Some(AppEvent::Notice(event.clone())),
            Self::RuntimeContextUpdated(event) => {
                Some(AppEvent::RuntimeContextUpdated(event.clone()))
            }
            Self::ContextTreeUpdated(event) => Some(AppEvent::ContextTreeUpdated(event.clone())),
            Self::ContextViewUpdated(event) => Some(AppEvent::ContextViewUpdated(event.clone())),
            Self::ContextDetailOpened(event) => Some(AppEvent::ContextDetailOpened(event.clone())),
            Self::FoldedOutputsUpdated(event) => {
                Some(AppEvent::FoldedOutputsUpdated(event.clone()))
            }
            Self::ContextSummaryUpdated(event) => {
                Some(AppEvent::ContextSummaryUpdated(event.clone()))
            }
            Self::Status(_) => None,
            Self::Interrupted => Some(AppEvent::Interrupted),
            Self::SessionResumed { .. }
            | Self::ContextBranchChanged { .. }
            | Self::ChildSessionViewed { .. }
            | Self::ContextBranchesLoaded { .. }
            | Self::SessionStarted { .. } => None,
            Self::Error(event) => Some(AppEvent::Error(event.clone())),
            Self::Done => Some(AppEvent::Done),
        }
    }
}

pub struct AgentRunner<C: Config> {
    event_tx: Option<RunnerEventSender>,
    permission_event_tx: Option<RunnerEventSender>,
    transcript: Option<Arc<Mutex<TranscriptRecorder>>>,
    event_mode: RunnerEventMode,
    child_session_id: Option<String>,
    permission_origin: Option<String>,
    subagent_delegate: Option<Arc<dyn SubagentDelegate<C>>>,
    _config: std::marker::PhantomData<C>,
}

struct RunnerSubagentDelegate<C: Config> {
    runtime: SubagentRuntime,
    sessions_dir: PathBuf,
    transcript: Arc<Mutex<TranscriptRecorder>>,
    event_tx: Option<RunnerEventSender>,
    _config: std::marker::PhantomData<C>,
}

#[derive(Debug, Default)]
struct AutoCompactionUiState {
    started: bool,
    streamed: bool,
    finished: bool,
}

impl<C> SubagentDelegate<C> for RunnerSubagentDelegate<C>
where
    C: Config + Clone + Send + Sync + 'static,
{
    fn run_named<'a>(
        &'a self,
        parent: &'a Agent<C>,
        agent_name: &'a str,
        invocation: SubagentInvocation,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<ToolResult>> + Send + 'a>> {
        Box::pin(async move {
            let parent_session_id = self
                .transcript
                .lock()
                .map_err(|_| anyhow!("transcript recorder poisoned"))?
                .session_id()
                .to_string();
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
                    self.event_tx.clone().map(subagent_event_sender::<C>),
                )
                .await?;

            let status = summary.status;
            let summary_text = summary.summary.clone();
            let compact_summary = compact_subagent_summary(&summary.summary);

            let data = json!({
                "run_id": summary.run_id,
                "child_session_id": summary.child_session_id,
                "agent_name": summary.agent_name,
                "status": status.as_str(),
                "summary": compact_summary,
                "full_summary": summary.summary,
                "structured_result": summary.structured_result,
                "active": false,
                "unreconciled": status == SubagentStatus::Completed,
                "reconciled": false,
                "reusable": false,
            });

            let tool_name = subagent_tool_name_for_agent_name(agent_name)
                .expect("runner dispatched unknown subagent agent name");

            if status == SubagentStatus::Completed {
                Ok(ToolResult::ok(tool_name, data))
            } else {
                Ok(ToolResult::err_with_data(tool_name, summary_text, data))
            }
        })
    }
}

impl<C: Config> AgentRunner<C> {
    pub fn subagent_event_sender(event_tx: RunnerEventSender) -> SubagentEventSender<C>
    where
        C: Clone + Send + Sync + 'static,
    {
        subagent_event_sender(event_tx)
    }

    pub fn new(event_tx: RunnerEventSender) -> Self {
        Self {
            event_tx: Some(event_tx),
            permission_event_tx: None,
            transcript: None,
            event_mode: RunnerEventMode::Emit,
            child_session_id: None,
            permission_origin: None,
            subagent_delegate: None,
            _config: std::marker::PhantomData,
        }
    }

    pub fn with_transcript(
        event_tx: RunnerEventSender,
        transcript: Arc<Mutex<TranscriptRecorder>>,
    ) -> Self {
        Self {
            event_tx: Some(event_tx),
            permission_event_tx: None,
            transcript: Some(transcript),
            event_mode: RunnerEventMode::Emit,
            child_session_id: None,
            permission_origin: None,
            subagent_delegate: None,
            _config: std::marker::PhantomData,
        }
    }

    pub fn with_subagent_runtime(self, runtime: SubagentRuntime, sessions_dir: PathBuf) -> Self
    where
        C: Clone + Send + Sync + 'static,
    {
        let mut self_ = self;
        if let Some(transcript) = self_.transcript.clone() {
            self_.subagent_delegate = Some(Arc::new(RunnerSubagentDelegate::<C> {
                runtime,
                sessions_dir,
                transcript,
                event_tx: self_.event_tx.clone(),
                _config: std::marker::PhantomData,
            }));
        }
        self_
    }

    pub fn silent_with_transcript(transcript: Arc<Mutex<TranscriptRecorder>>) -> Self {
        Self {
            event_tx: None,
            permission_event_tx: None,
            transcript: Some(transcript),
            event_mode: RunnerEventMode::SilentDenyPermissions,
            child_session_id: None,
            permission_origin: None,
            subagent_delegate: None,
            _config: std::marker::PhantomData,
        }
    }

    pub fn child_streaming_with_transcript(
        transcript: Arc<Mutex<TranscriptRecorder>>,
        event_tx: RunnerEventSender,
        child_session_id: impl Into<String>,
    ) -> Self {
        Self {
            event_tx: Some(event_tx),
            permission_event_tx: None,
            transcript: Some(transcript),
            event_mode: RunnerEventMode::SilentDenyPermissions,
            child_session_id: Some(child_session_id.into()),
            permission_origin: None,
            subagent_delegate: None,
            _config: std::marker::PhantomData,
        }
    }

    pub fn child_streaming_with_permission_passthrough(
        transcript: Arc<Mutex<TranscriptRecorder>>,
        event_tx: RunnerEventSender,
        child_session_id: impl Into<String>,
        permission_origin: impl Into<String>,
    ) -> Self {
        Self {
            event_tx: Some(event_tx.clone()),
            permission_event_tx: Some(event_tx),
            transcript: Some(transcript),
            event_mode: RunnerEventMode::Emit,
            child_session_id: Some(child_session_id.into()),
            permission_origin: Some(permission_origin.into()),
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
        }
        if let Some(delegate) = self.subagent_delegate.clone() {
            agent.set_subagent_delegate(delegate);
        }
        if record_user_prompt {
            let user_event = UserMessageEvent::from_submission(prompt.clone());
            self.emit(RunnerEvent::UserMessage(user_event))?;
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
            )
            .or_else(|error| self.finish_with_error(error))?;
        }
        if let Some((session_id, mut title_agent)) = pending_title {
            let transcript = self.transcript.clone();
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
                        if let Err(error) = recorder.record_session_title(title) {
                            warn!(error = %error, session_id, "failed to persist generated session title");
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
        let auto_compaction = Arc::new(Mutex::new(AutoCompactionUiState::default()));
        let response = agent
            .run_stream_content_with_interactions_async(
                prompt_content.clone(),
                move |delta| {
                    let sender = sender.clone();
                    let child_session_id = child_session_id.clone();
                    let delta = delta.to_string();
                    async move {
                        send_scoped_event(
                            &sender,
                            child_session_id.as_deref(),
                            RunnerEvent::AssistantDelta(AssistantDeltaEvent::new(delta)),
                        )
                    }
                },
                {
                    let sender = self.event_tx.clone();
                    let transcript = self.transcript.clone();
                    let child_session_id = self.child_session_id.clone();
                    let auto_compaction = Arc::clone(&auto_compaction);
                    move |event| {
                        let sender = sender.clone();
                        let transcript = transcript.clone();
                        let child_session_id = child_session_id.clone();
                        let auto_compaction = Arc::clone(&auto_compaction);
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
                                AgentEvent::ContextCompactionStarted => {
                                    emit_auto_compaction_started(
                                        &sender,
                                        child_session_id.as_deref(),
                                        &auto_compaction,
                                    )?;
                                }
                                AgentEvent::ContextCompactionDelta { delta } => {
                                    emit_auto_compaction_delta(
                                        &sender,
                                        child_session_id.as_deref(),
                                        &auto_compaction,
                                        delta,
                                    )?;
                                }
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
                                        RunnerEvent::TokenUsage(TokenUsageEvent::with_breakdown(
                                            used_tokens,
                                            context_window_tokens,
                                            input_tokens,
                                            output_tokens,
                                            cached_tokens,
                                        ).with_cache_report(cache_report)),
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
                                    )?;
                                }
                                AgentEvent::ReasoningDelta { item_id, delta } => {
                                    send_scoped_event(
                                        &sender,
                                        child_session_id.as_deref(),
                                        RunnerEvent::ReasoningDelta(ReasoningDeltaEvent::new(
                                            item_id, delta,
                                        )),
                                    )?;
                                }
                                AgentEvent::ReasoningDone { item_id, text } => {
                                    emit_context_projection_updates(
                                        &sender,
                                        &transcript,
                                        child_session_id.as_deref(),
                                    )?;
                                    send_scoped_event(
                                        &sender,
                                        child_session_id.as_deref(),
                                        RunnerEvent::ReasoningDone(ReasoningDoneEvent::new(
                                            item_id, text,
                                        )),
                                    )?;
                                }
                                AgentEvent::ModelStreamIssue {
                                    message,
                                    detail,
                                    action,
                                } => {
                                    if is_compaction_terminal_message(&message) {
                                        emit_auto_compaction_terminal_issue(
                                            &sender,
                                            child_session_id.as_deref(),
                                            &auto_compaction,
                                            &message,
                                        )?;
                                    }
                                    let issue = ProcessIssueEvent {
                                        message,
                                        detail,
                                        action: Some(action),
                                    };
                                    send_scoped_event(
                                        &sender,
                                        child_session_id.as_deref(),
                                        RunnerEvent::ProcessIssue(issue),
                                    )?;
                                }
                                AgentEvent::AssistantMessage { .. }
                                | AgentEvent::AssistantToolCallBatch { .. }
                                | AgentEvent::InternalContinuation { .. } => {}
                                AgentEvent::ToolCallPending { call_id, name } => {
                                    send_scoped_event(
                                        &sender,
                                        child_session_id.as_deref(),
                                        RunnerEvent::ToolPending(ToolPendingEvent::new(
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
                                        RunnerEvent::ToolStarted(started),
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
                                        RunnerEvent::ToolOutputDelta(ToolOutputDeltaEvent::new(
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
                                        RunnerEvent::ToolFinished(finished),
                                    )?;
                                    // The durable terminal event is authoritative even if its
                                    // subsequent projection cannot be rebuilt.
                                    emit_context_projection_update(
                                        &sender,
                                        &transcript,
                                        child_session_id.as_deref(),
                                        disposition,
                                    )?;
                                }
                                AgentEvent::ToolCallBatchFinished => {
                                    if child_session_id.is_none() {
                                        send_scoped_event(
                                            &sender,
                                            child_session_id.as_deref(),
                                            RunnerEvent::ToolBatchFinished,
                                        )?;
                                    }
                                }
                                AgentEvent::TodoSnapshotUpdated { items } => {
                                    send_scoped_event(
                                        &sender,
                                        child_session_id.as_deref(),
                                        RunnerEvent::TodoSnapshot(TodoSnapshotEvent::new(items)),
                                    )?;
                                }
                                AgentEvent::AutoContinueChanged { state } => {
                                    send_scoped_event(
                                        &sender,
                                        child_session_id.as_deref(),
                                        RunnerEvent::AutoContinueChanged(
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
                                        RunnerEvent::ToolCancelled(ToolCancelledEvent::new(
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
                                    );
                                    let _ = emit_auto_compaction_finished(
                                        &sender,
                                        child_session_id.as_deref(),
                                        &auto_compaction,
                                        &event.summary,
                                        event.retained_history_items,
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
                    let event_mode = self.event_mode;
                    move |request| {
                        let sender = sender.clone();
                        let permission_sender = permission_sender.clone();
                        let transcript = transcript.clone();
                        let permission_origin = permission_origin.clone();
                        let child_session_id = child_session_id.clone();
                        let event_mode = event_mode;
                        async move {
                            // Permission decisions are not AgentEvent stream entries.
                            let request_event =
                                permission_request_event(&request, permission_origin.as_deref());
                            if matches!(event_mode, RunnerEventMode::SilentDenyPermissions) {
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
                                        RunnerEvent::ChildPermissionRequested {
                                            child_session_id,
                                            event: request_event.clone(),
                                            handle,
                                        }
                                    }
                                    None => RunnerEvent::PermissionRequested {
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
                            )?;
                            let permission_target = permission_sender.clone().or(sender.clone());
                            send_optional_event(
                                &permission_target,
                                match child_session_id.clone() {
                                    Some(child_session_id) => RunnerEvent::ChildAppEvent {
                                        child_session_id,
                                        event: AppEvent::PermissionResolved(resolution),
                                    },
                                    None => RunnerEvent::PermissionResolved(resolution),
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
                            if matches!(event_mode, RunnerEventMode::SilentDenyPermissions) {
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
                                    Some(child_session_id) => RunnerEvent::ChildQuestionRequested {
                                        child_session_id,
                                        request: request.clone(),
                                        handle,
                                    },
                                    None => RunnerEvent::QuestionRequested {
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
                )
                .or_else(|error| self.finish_with_error(error))?;
                self.emit(RunnerEvent::AssistantDone { message_id: None })?;
                self.emit(RunnerEvent::Done)?;
                Ok(message)
            }
            Err(error) => {
                let error_message = format!("{error:#}");
                finish_auto_compaction_with_error(
                    &self.event_tx,
                    self.child_session_id.as_deref(),
                    &auto_compaction,
                )?;
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
                ) {
                    let composite = anyhow!(
                        "{} (additionally failed context projection: {})",
                        error_message,
                        projection_error
                    );
                    self.finish_with_error(composite)?;
                }
                self.emit(RunnerEvent::Error(event))?;
                self.emit(RunnerEvent::Done)?;
                Err(error)
            }
        }
    }

    pub fn record_model_changed(&self, previous_model: &str, new_model: &str) -> Result<()> {
        self.record(|recorder| {
            recorder.record_model_changed(previous_model.to_string(), new_model.to_string())
        })
    }

    pub fn record_permission_mode_changed(
        &self,
        previous_mode: &str,
        new_mode: &str,
    ) -> Result<()> {
        self.record(|recorder| {
            recorder.record_permission_mode_changed(previous_mode.to_string(), new_mode.to_string())
        })
    }

    fn emit(&self, event: RunnerEvent) -> Result<()> {
        let event = if let Some(child_session_id) = &self.child_session_id {
            wrap_child_runner_event(child_session_id.clone(), event)
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
        self.emit(RunnerEvent::Error(event))?;
        self.emit(RunnerEvent::Done)?;
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

pub fn subagent_event_sender<C>(event_tx: RunnerEventSender) -> SubagentEventSender<C>
where
    C: Config + Clone + Send + Sync + 'static,
{
    let status_tx = event_tx.clone();
    let error_tx = event_tx.clone();
    SubagentEventSender::new(
        Arc::new(move |message| {
            status_tx
                .send(RunnerEvent::Status(message))
                .map_err(|_| anyhow!("runner event channel closed"))
        }),
        Arc::new(move |message| {
            error_tx
                .send(RunnerEvent::Error(ErrorEvent::new(message)))
                .map_err(|_| anyhow!("runner event channel closed"))
        }),
        Arc::new(
            move |agent, prompt, transcript, child_session_id, permission_origin| {
                let runner: AgentRunner<C> = if let Some(permission_origin) = permission_origin {
                    AgentRunner::child_streaming_with_permission_passthrough(
                        transcript,
                        event_tx.clone(),
                        child_session_id,
                        permission_origin,
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

fn wrap_child_runner_event(child_session_id: String, event: RunnerEvent) -> RunnerEvent {
    match event {
        RunnerEvent::UserMessage(event) => RunnerEvent::ChildAppEvent {
            child_session_id,
            event: AppEvent::UserMessage(event),
        },
        RunnerEvent::ReasoningDelta(event) => RunnerEvent::ChildAppEvent {
            child_session_id,
            event: AppEvent::ReasoningDelta(event),
        },
        RunnerEvent::ReasoningDone(event) => RunnerEvent::ChildAppEvent {
            child_session_id,
            event: AppEvent::ReasoningDone(event),
        },
        RunnerEvent::AssistantDelta(event) => RunnerEvent::ChildAppEvent {
            child_session_id,
            event: AppEvent::AssistantDelta(event),
        },
        RunnerEvent::AssistantDone { message_id } => RunnerEvent::ChildAppEvent {
            child_session_id,
            event: AppEvent::AssistantDone { message_id },
        },
        RunnerEvent::TokenUsage(event) => RunnerEvent::ChildAppEvent {
            child_session_id,
            event: AppEvent::TokenUsage(event),
        },
        RunnerEvent::ToolPending(event) => RunnerEvent::ChildAppEvent {
            child_session_id,
            event: AppEvent::ToolPending(event),
        },
        RunnerEvent::ToolCancelled(event) => RunnerEvent::ChildAppEvent {
            child_session_id,
            event: AppEvent::ToolCancelled(event),
        },
        RunnerEvent::ToolStarted(event) => RunnerEvent::ChildAppEvent {
            child_session_id,
            event: AppEvent::ToolStarted(event),
        },
        RunnerEvent::ToolFinished(event) => RunnerEvent::ChildAppEvent {
            child_session_id,
            event: AppEvent::ToolFinished(event),
        },
        RunnerEvent::ToolOutputDelta(event) => RunnerEvent::ChildAppEvent {
            child_session_id,
            event: AppEvent::ToolOutputDelta(event),
        },
        RunnerEvent::TodoSnapshot(event) => RunnerEvent::ChildAppEvent {
            child_session_id,
            event: AppEvent::TodoSnapshot(event),
        },
        RunnerEvent::AutoContinueChanged(event) => RunnerEvent::ChildAppEvent {
            child_session_id,
            event: AppEvent::AutoContinueChanged(event),
        },
        RunnerEvent::PermissionResolved(event) => RunnerEvent::ChildAppEvent {
            child_session_id,
            event: AppEvent::PermissionResolved(event),
        },
        RunnerEvent::ProcessIssue(event) => RunnerEvent::ChildAppEvent {
            child_session_id,
            event: AppEvent::ProcessIssue(event),
        },
        RunnerEvent::Notice(event) => RunnerEvent::ChildAppEvent {
            child_session_id,
            event: AppEvent::Notice(event),
        },
        RunnerEvent::RuntimeContextUpdated(event) => RunnerEvent::ChildAppEvent {
            child_session_id,
            event: AppEvent::RuntimeContextUpdated(event),
        },
        RunnerEvent::ContextTreeUpdated(event) => RunnerEvent::ChildAppEvent {
            child_session_id,
            event: AppEvent::ContextTreeUpdated(event),
        },
        RunnerEvent::ContextViewUpdated(event) => RunnerEvent::ChildAppEvent {
            child_session_id,
            event: AppEvent::ContextViewUpdated(event),
        },
        RunnerEvent::ContextDetailOpened(event) => RunnerEvent::ChildAppEvent {
            child_session_id,
            event: AppEvent::ContextDetailOpened(event),
        },
        RunnerEvent::FoldedOutputsUpdated(event) => RunnerEvent::ChildAppEvent {
            child_session_id,
            event: AppEvent::FoldedOutputsUpdated(event),
        },
        RunnerEvent::ContextSummaryUpdated(event) => RunnerEvent::ChildAppEvent {
            child_session_id,
            event: AppEvent::ContextSummaryUpdated(event),
        },
        RunnerEvent::Interrupted => RunnerEvent::ChildAppEvent {
            child_session_id,
            event: AppEvent::Interrupted,
        },
        RunnerEvent::Error(event) => RunnerEvent::ChildAppEvent {
            child_session_id,
            event: AppEvent::Error(event),
        },
        RunnerEvent::Done => RunnerEvent::ChildAppEvent {
            child_session_id,
            event: AppEvent::Done,
        },
        event => event,
    }
}

fn send_scoped_event(
    sender: &Option<RunnerEventSender>,
    child_session_id: Option<&str>,
    event: RunnerEvent,
) -> Result<()> {
    let event = match child_session_id {
        Some(child_session_id) => wrap_child_runner_event(child_session_id.to_string(), event),
        None => event,
    };
    send_optional_event(sender, event)
}

fn emit_context_projection_updates(
    sender: &Option<RunnerEventSender>,
    transcript: &Option<Arc<Mutex<TranscriptRecorder>>>,
    child_session_id: Option<&str>,
) -> Result<()> {
    emit_context_projection_update(
        sender,
        transcript,
        child_session_id,
        RuntimeContextDisposition::Advance,
    )
}

fn emit_context_projection_update(
    sender: &Option<RunnerEventSender>,
    transcript: &Option<Arc<Mutex<TranscriptRecorder>>>,
    child_session_id: Option<&str>,
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
        RunnerEvent::RuntimeContextUpdated(RuntimeContextUpdatedEvent {
            context,
            disposition,
        }),
    )?;
    Ok(())
}

fn send_optional_event(sender: &Option<RunnerEventSender>, event: RunnerEvent) -> Result<()> {
    if let Some(sender) = sender {
        sender
            .send(event)
            .map_err(|_| anyhow!("runner event channel closed"))?;
    }
    Ok(())
}

fn emit_auto_compaction_started(
    sender: &Option<RunnerEventSender>,
    child_session_id: Option<&str>,
    state: &Arc<Mutex<AutoCompactionUiState>>,
) -> Result<()> {
    let mut state = state
        .lock()
        .map_err(|_| anyhow!("auto compaction UI state poisoned"))?;
    if state.started && !state.finished {
        return Ok(());
    }
    state.started = true;
    state.streamed = false;
    state.finished = false;
    drop(state);

    send_scoped_event(
        sender,
        child_session_id,
        RunnerEvent::Status("Compacting context".into()),
    )?;
    send_scoped_event(
        sender,
        child_session_id,
        RunnerEvent::Notice(NoticeEvent::new(compaction_separator(
            COMPACTION_SEPARATOR_LABEL,
        ))),
    )
}

fn emit_auto_compaction_delta(
    sender: &Option<RunnerEventSender>,
    child_session_id: Option<&str>,
    state: &Arc<Mutex<AutoCompactionUiState>>,
    delta: String,
) -> Result<()> {
    emit_auto_compaction_started(sender, child_session_id, state)?;
    let mut state = state
        .lock()
        .map_err(|_| anyhow!("auto compaction UI state poisoned"))?;
    state.streamed = true;
    drop(state);

    send_scoped_event(
        sender,
        child_session_id,
        RunnerEvent::AssistantDelta(AssistantDeltaEvent::with_message_id(
            COMPACTION_MESSAGE_ID,
            delta,
        )),
    )
}

fn emit_auto_compaction_finished(
    sender: &Option<RunnerEventSender>,
    child_session_id: Option<&str>,
    state: &Arc<Mutex<AutoCompactionUiState>>,
    summary: &str,
    retained_history_items: usize,
) -> Result<()> {
    emit_auto_compaction_started(sender, child_session_id, state)?;

    let should_emit_summary = {
        let mut state = state
            .lock()
            .map_err(|_| anyhow!("auto compaction UI state poisoned"))?;
        if state.finished {
            return Ok(());
        }
        state.finished = true;
        !state.streamed
    };

    if should_emit_summary {
        send_scoped_event(
            sender,
            child_session_id,
            RunnerEvent::AssistantDelta(AssistantDeltaEvent::with_message_id(
                COMPACTION_MESSAGE_ID,
                summary.to_string(),
            )),
        )?;
    }

    send_scoped_event(
        sender,
        child_session_id,
        RunnerEvent::AssistantDone {
            message_id: Some(COMPACTION_MESSAGE_ID.into()),
        },
    )?;
    send_scoped_event(
        sender,
        child_session_id,
        RunnerEvent::Notice(NoticeEvent::new(compaction_separator(
            COMPACTION_SEPARATOR_LABEL,
        ))),
    )?;
    send_scoped_event(
        sender,
        child_session_id,
        RunnerEvent::Status(format!(
            "Context compacted ({retained_history_items} history items retained)"
        )),
    )
}

fn emit_auto_compaction_terminal_issue(
    sender: &Option<RunnerEventSender>,
    child_session_id: Option<&str>,
    state: &Arc<Mutex<AutoCompactionUiState>>,
    status: &str,
) -> Result<()> {
    let should_emit_terminal = {
        let mut state = state
            .lock()
            .map_err(|_| anyhow!("auto compaction UI state poisoned"))?;
        if !state.started || state.finished {
            return Ok(());
        }
        state.finished = true;
        true
    };

    if !should_emit_terminal {
        return Ok(());
    }

    send_scoped_event(
        sender,
        child_session_id,
        RunnerEvent::AssistantDone {
            message_id: Some(COMPACTION_MESSAGE_ID.into()),
        },
    )?;
    send_scoped_event(
        sender,
        child_session_id,
        RunnerEvent::Notice(NoticeEvent::new(compaction_separator(
            COMPACTION_SEPARATOR_LABEL,
        ))),
    )?;
    send_scoped_event(
        sender,
        child_session_id,
        RunnerEvent::Status(status.to_string()),
    )
}

fn is_compaction_terminal_message(message: &str) -> bool {
    matches!(
        message,
        "Context compaction failed" | "Context compaction cancelled"
    )
}

fn finish_auto_compaction_with_error(
    sender: &Option<RunnerEventSender>,
    child_session_id: Option<&str>,
    state: &Arc<Mutex<AutoCompactionUiState>>,
) -> Result<()> {
    let should_emit_done = {
        let mut state = state
            .lock()
            .map_err(|_| anyhow!("auto compaction UI state poisoned"))?;
        if !state.started || state.finished {
            false
        } else {
            state.finished = true;
            true
        }
    };

    if should_emit_done {
        send_scoped_event(
            sender,
            child_session_id,
            RunnerEvent::AssistantDone {
                message_id: Some(COMPACTION_MESSAGE_ID.into()),
            },
        )?;
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
            decision: super::events::PermissionDecision::Approved,
            reason: Some("Allow once".into()),
        },
        PermissionResponse::AllowAlways => PermissionResolutionEvent {
            call_id,
            decision: super::events::PermissionDecision::Approved,
            reason: Some("Allowed for this session".into()),
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
    let max = data
        .get("max_continuations")
        .and_then(Value::as_u64)
        .map(|value| format!(" · max {value}"))
        .unwrap_or_default();
    if enabled {
        format!("enabled auto-continue{max}")
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
    use crate::transcript::TranscriptRecorder;
    use crate::tui::events::{AppEvent, PermissionDecision};
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
    fn token_usage_app_event_preserves_cache_report() {
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
        let event = RunnerEvent::TokenUsage(
            TokenUsageEvent::with_breakdown(120, 1_000, 100, 20, 80)
                .with_cache_report(Some(report.clone())),
        );

        assert_eq!(
            event.app_event(),
            Some(AppEvent::TokenUsage(
                TokenUsageEvent::with_breakdown(120, 1_000, 100, 20, 80)
                    .with_cache_report(Some(report)),
            ))
        );
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
            serde_json::json!({"enabled": true, "max_continuations": 2}),
        );
        assert_eq!(
            output_summary(&auto_continue).as_deref(),
            Some("enabled auto-continue · max 2")
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
    fn auto_compaction_finish_backfills_summary_when_no_delta_streamed() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let sender = Some(tx);
        let state = Arc::new(Mutex::new(AutoCompactionUiState {
            started: true,
            streamed: false,
            finished: false,
        }));

        emit_auto_compaction_finished(&sender, None, &state, "summary fallback", 3)
            .expect("emit succeeds");

        assert!(
            matches!(rx.try_recv(), Ok(RunnerEvent::AssistantDelta(event))
            if event.message_id.as_deref() == Some(COMPACTION_MESSAGE_ID)
                && event.delta == "summary fallback")
        );
        assert!(
            matches!(rx.try_recv(), Ok(RunnerEvent::AssistantDone { message_id })
            if message_id.as_deref() == Some(COMPACTION_MESSAGE_ID))
        );
        assert!(matches!(rx.try_recv(), Ok(RunnerEvent::Notice(notice))
            if notice.message == compaction_separator(COMPACTION_SEPARATOR_LABEL)));
        assert!(matches!(rx.try_recv(), Ok(RunnerEvent::Status(status))
            if status == "Context compacted (3 history items retained)"));
    }

    #[test]
    fn auto_compaction_ui_state_resets_for_multiple_cycles() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let sender = Some(tx);
        let state = Arc::new(Mutex::new(AutoCompactionUiState::default()));

        emit_auto_compaction_started(&sender, None, &state).expect("first start emits");
        emit_auto_compaction_delta(&sender, None, &state, "first".into())
            .expect("first delta emits");
        emit_auto_compaction_finished(&sender, None, &state, "first fallback", 2)
            .expect("first finish emits");
        while rx.try_recv().is_ok() {}

        emit_auto_compaction_started(&sender, None, &state).expect("second start emits");
        emit_auto_compaction_delta(&sender, None, &state, "second".into())
            .expect("second delta emits");
        emit_auto_compaction_finished(&sender, None, &state, "second fallback", 3)
            .expect("second finish emits");

        assert!(matches!(rx.try_recv(), Ok(RunnerEvent::Status(status))
            if status == "Compacting context"));
        assert!(matches!(rx.try_recv(), Ok(RunnerEvent::Notice(notice))
            if notice.message == compaction_separator(COMPACTION_SEPARATOR_LABEL)));
        assert!(
            matches!(rx.try_recv(), Ok(RunnerEvent::AssistantDelta(event))
            if event.message_id.as_deref() == Some(COMPACTION_MESSAGE_ID)
                && event.delta == "second")
        );
        assert!(
            matches!(rx.try_recv(), Ok(RunnerEvent::AssistantDone { message_id })
            if message_id.as_deref() == Some(COMPACTION_MESSAGE_ID))
        );
        assert!(matches!(rx.try_recv(), Ok(RunnerEvent::Notice(notice))
            if notice.message == compaction_separator(COMPACTION_SEPARATOR_LABEL)));
        assert!(matches!(rx.try_recv(), Ok(RunnerEvent::Status(status))
            if status == "Context compacted (3 history items retained)"));
    }

    #[test]
    fn auto_compaction_terminal_issue_closes_started_cycle() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let sender = Some(tx);
        let state = Arc::new(Mutex::new(AutoCompactionUiState {
            started: true,
            streamed: true,
            finished: false,
        }));

        emit_auto_compaction_terminal_issue(&sender, None, &state, "Context compaction failed")
            .expect("terminal issue emits");

        assert!(
            matches!(rx.try_recv(), Ok(RunnerEvent::AssistantDone { message_id })
            if message_id.as_deref() == Some(COMPACTION_MESSAGE_ID))
        );
        assert!(matches!(rx.try_recv(), Ok(RunnerEvent::Notice(notice))
            if notice.message == compaction_separator(COMPACTION_SEPARATOR_LABEL)));
        assert!(matches!(rx.try_recv(), Ok(RunnerEvent::Status(status))
            if status == "Context compaction failed"));
    }

    #[test]
    fn child_streaming_runner_wraps_app_events_with_child_session_id() {
        let (tx, _rx) = mpsc::unbounded_channel();
        let _runner = AgentRunner::<OpenAIConfig>::child_streaming_with_transcript(
            temp_transcript(),
            tx,
            "child-session",
        );

        let wrapped = wrap_child_runner_event(
            "child-session".into(),
            RunnerEvent::AssistantDelta(AssistantDeltaEvent::new("hi")),
        );

        assert!(matches!(
            wrapped,
            RunnerEvent::ChildAppEvent { child_session_id, event: AppEvent::AssistantDelta(delta) }
                if child_session_id == "child-session" && delta.delta == "hi"
        ));

        let wrapped_notice = wrap_child_runner_event(
            "child-session".into(),
            RunnerEvent::Notice(NoticeEvent::new("child notice")),
        );

        assert!(matches!(
            wrapped_notice,
            RunnerEvent::ChildAppEvent { child_session_id, event: AppEvent::Notice(notice) }
                if child_session_id == "child-session" && notice.message == "child notice"
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
            Some(RunnerEvent::UserMessage(UserMessageEvent { content, .. }))
                if content.text == "hello"
                    && content.attachments.len() == 1
                    && content.attachments[0].label == "screen.png"
        ));
        assert!(matches!(
            rx.recv().await,
            Some(RunnerEvent::Error(ErrorEvent { message, .. })) if message.contains("transcript recorder poisoned")
        ));
        assert!(matches!(rx.recv().await, Some(RunnerEvent::Done)));
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
    fn todo_runner_events_map_to_app_events() {
        let pending_event =
            RunnerEvent::ToolPending(ToolPendingEvent::new("call-pending", "edit__apply_patch"));
        assert!(matches!(
            pending_event.app_event(),
            Some(AppEvent::ToolPending(_))
        ));

        let todo_event = RunnerEvent::TodoSnapshot(TodoSnapshotEvent::new(vec![TodoItem {
            id: "t1".into(),
            content: "inspect".into(),
            status: TodoStatus::Pending,
        }]));
        assert!(matches!(
            todo_event.app_event(),
            Some(AppEvent::TodoSnapshot(_))
        ));

        let auto_event =
            RunnerEvent::AutoContinueChanged(AutoContinueChangedEvent::new(AutoContinueState {
                enabled: true,
                max_continuations: 2,
            }));
        assert!(matches!(
            auto_event.app_event(),
            Some(AppEvent::AutoContinueChanged(_))
        ));
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
            assert!(!matches!(event, RunnerEvent::UserMessage(_)));
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
}
