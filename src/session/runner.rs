use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use anyhow::{Result, anyhow};
use async_openai::config::Config;
use tokio::sync::oneshot;
use tracing::warn;

use crate::agent::{Agent, AgentEvent, SubagentDelegate};
use crate::agent_event_journal::{ContextProjection, JournalEffect, persist_agent_event};
use crate::permission::PermissionApproval;
use crate::subagent::SubagentPool;
use crate::subagent_events::SubagentEventSender;
use crate::transcript::{
    TranscriptRecorder, read_records, transcript_has_session_title, transcript_has_user_message,
};
use crate::user_content::{UserMessageContent, UserMessageSubmission};

#[path = "events.rs"]
mod events;
#[path = "formatting.rs"]
mod formatting;
#[path = "subagent_delegate.rs"]
mod subagent_delegate;

use events::SessionTransportEventMode;
pub(crate) use events::{
    ModelCatalogEntry, ModelCatalogReasoning, ModelCatalogUpdatedEvent, PermissionResponse,
    RunnerPermissionRequest, RunnerQuestionRequest, SessionTransportEvent,
    SessionTransportEventSender,
};

use events::{
    emit_context_projection_update, emit_context_projection_updates, permission_request_event,
    permission_resolution_event, record_transcript, retry_lifecycle_event, send_optional_event,
    send_scoped_event, tool_finished_event, tool_started_event, wrap_child_session_transport_event,
};

use subagent_delegate::RunnerSubagentDelegate;

use crate::session::{
    AssistantDeltaEvent, AutoContinueChangedEvent, ErrorEvent, NoticeEvent, ProcessIssueEvent,
    ReasoningDeltaEvent, ReasoningDoneEvent, RuntimeContextDisposition, SessionEvent,
    TodoSnapshotEvent, TokenUsageEvent, ToolCancelledEvent, ToolFinishedEvent, ToolOutcome,
    ToolOutputDeltaEvent, ToolPendingEvent, ToolStartedEvent, UserMessageEvent,
};
use crate::tool_format::format_tool_call;
use formatting::{output_json, output_summary};

#[derive(Debug, Clone, PartialEq, Eq)]
struct ToolBatchReconciliation {
    active: bool,
    disposition: RuntimeContextDisposition,
}

impl Default for ToolBatchReconciliation {
    fn default() -> Self {
        Self {
            active: false,
            disposition: RuntimeContextDisposition::Advance,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ToolBatchReconciliationAction {
    None,
    ProjectEvidence,
    FinishBatch(RuntimeContextDisposition),
}

impl ToolBatchReconciliation {
    fn observe_event(&mut self, event: &AgentEvent) {
        if matches!(
            event,
            AgentEvent::ToolCallPending { .. }
                | AgentEvent::ToolCallStarted { .. }
                | AgentEvent::ToolCallFinished { .. }
                | AgentEvent::ToolCallCancelled { .. }
        ) {
            self.active = true;
        }
    }

    fn should_project_evidence(&self) -> bool {
        !self.active
    }

    fn record_journal_effect(&mut self, effect: JournalEffect) {
        if effect.context_projection == ContextProjection::ReplaceScope {
            self.disposition = RuntimeContextDisposition::ReplaceScope;
        }
    }

    fn finish(&mut self) -> RuntimeContextDisposition {
        let disposition = self.disposition;
        *self = Self::default();
        disposition
    }

    fn reconcile(
        &mut self,
        event: &AgentEvent,
        journal_effect: JournalEffect,
    ) -> ToolBatchReconciliationAction {
        let project_evidence =
            matches!(event, AgentEvent::EvidenceRecorded(_)) && self.should_project_evidence();
        if !matches!(event, AgentEvent::ToolCallBatchFinished) {
            self.observe_event(event);
        }
        self.record_journal_effect(journal_effect);

        if matches!(event, AgentEvent::ToolCallBatchFinished) {
            ToolBatchReconciliationAction::FinishBatch(self.finish())
        } else if project_evidence {
            ToolBatchReconciliationAction::ProjectEvidence
        } else {
            ToolBatchReconciliationAction::None
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

impl AgentRunner<async_openai::config::OpenAIConfig> {
    pub fn with_subagent_runtime(
        self,
        runtime: SubagentPool,
        sessions_dir: PathBuf,
        expert_model_routes: indexmap::IndexMap<String, crate::config::ModelRoute>,
        route_api_key_configured: indexmap::IndexMap<String, bool>,
        provider_api_key_hints: indexmap::IndexMap<String, String>,
        api_key_hint: String,
        background_control_tx: Option<
            tokio::sync::mpsc::UnboundedSender<super::engine::SessionEngineControl>,
        >,
        background_event_tx: Option<SessionTransportEventSender>,
    ) -> Self {
        self.with_subagent_runtime_inner(
            runtime,
            sessions_dir,
            expert_model_routes,
            route_api_key_configured,
            provider_api_key_hints,
            api_key_hint,
            background_control_tx,
            background_event_tx,
            None,
        )
    }

    fn with_subagent_runtime_inner(
        self,
        runtime: SubagentPool,
        sessions_dir: PathBuf,
        expert_model_routes: indexmap::IndexMap<String, crate::config::ModelRoute>,
        route_api_key_configured: indexmap::IndexMap<String, bool>,
        provider_api_key_hints: indexmap::IndexMap<String, String>,
        api_key_hint: String,
        background_control_tx: Option<
            tokio::sync::mpsc::UnboundedSender<super::engine::SessionEngineControl>,
        >,
        background_event_tx: Option<SessionTransportEventSender>,
        #[cfg(test)] background_child_started_tx: Option<
            tokio::sync::mpsc::UnboundedSender<String>,
        >,
        #[cfg(not(test))] _background_child_started_tx: Option<()>,
    ) -> Self {
        let mut self_ = self;
        if let Some(transcript) = self_.transcript.clone() {
            self_.subagent_delegate = Some(Arc::new(RunnerSubagentDelegate {
                runtime,
                sessions_dir,
                transcript,
                event_tx: self_.event_tx.clone(),
                background_event_tx,
                #[cfg(test)]
                background_child_started_tx,
                route_api_key_configured,
                retained_session_routes: expert_model_routes
                    .into_values()
                    .map(|route| route.display_name())
                    .collect(),
                provider_api_key_hints,
                api_key_hint,
                background_control_tx,
            }));
        }
        self_
    }

    #[cfg(test)]
    pub(crate) fn with_subagent_runtime_test_hooks(
        self,
        runtime: SubagentPool,
        sessions_dir: PathBuf,
        expert_model_routes: indexmap::IndexMap<String, crate::config::ModelRoute>,
        route_api_key_configured: indexmap::IndexMap<String, bool>,
        provider_api_key_hints: indexmap::IndexMap<String, String>,
        api_key_hint: String,
        background_control_tx: Option<
            tokio::sync::mpsc::UnboundedSender<super::engine::SessionEngineControl>,
        >,
        background_event_tx: Option<SessionTransportEventSender>,
        background_child_started_tx: tokio::sync::mpsc::UnboundedSender<String>,
    ) -> Self {
        self.with_subagent_runtime_inner(
            runtime,
            sessions_dir,
            expert_model_routes,
            route_api_key_configured,
            provider_api_key_hints,
            api_key_hint,
            background_control_tx,
            background_event_tx,
            Some(background_child_started_tx),
        )
    }

    #[cfg(test)]
    pub(crate) fn install_subagent_delegate_for_test(
        &self,
        agent: &mut Agent<async_openai::config::OpenAIConfig>,
    ) {
        if let Some(delegate) = self.subagent_delegate.clone() {
            agent.set_subagent_delegate(delegate);
        }
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
        let queue = Arc::new(Mutex::new(crate::agent::TurnContinuationQueue::default()));
        self.run_prompt_with_continuations(agent, prompt, queue)
            .await
    }

    pub(crate) async fn run_prompt_with_continuations(
        &self,
        agent: &mut Agent<C>,
        prompt: UserMessageSubmission,
        turn_continuation_queue: Arc<Mutex<crate::agent::TurnContinuationQueue>>,
    ) -> Result<String>
    where
        C: Clone + Send + Sync + 'static,
    {
        let continuation_queue = Arc::clone(&turn_continuation_queue);
        let mut continuation_guard = agent.turn_continuation_provider_guard(Arc::new(move || {
            let mut queue = continuation_queue
                .lock()
                .map_err(|_| anyhow!("turn continuation queue poisoned"))?;
            Ok(queue.drain_ready())
        }));
        self.run_prompt_with_options(continuation_guard.agent(), prompt, true)
            .await
    }

    #[cfg(test)]
    pub async fn continue_session(&self, agent: &mut Agent<C>) -> Result<String>
    where
        C: Clone + Send + Sync + 'static,
    {
        let queue = Arc::new(Mutex::new(crate::agent::TurnContinuationQueue::default()));
        self.run_existing_history_with_continuations(agent, queue)
            .await
    }

    #[cfg(test)]
    pub(crate) async fn run_existing_history(&self, agent: &mut Agent<C>) -> Result<String>
    where
        C: Clone + Send + Sync + 'static,
    {
        let queue = Arc::new(Mutex::new(crate::agent::TurnContinuationQueue::default()));
        self.run_existing_history_with_continuations(agent, queue)
            .await
    }

    pub(crate) async fn run_existing_history_with_continuations(
        &self,
        agent: &mut Agent<C>,
        turn_continuation_queue: Arc<Mutex<crate::agent::TurnContinuationQueue>>,
    ) -> Result<String>
    where
        C: Clone + Send + Sync + 'static,
    {
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
        }
        if let Some(delegate) = self.subagent_delegate.clone() {
            agent.set_subagent_delegate(delegate);
        }
        let continuation_queue = Arc::clone(&turn_continuation_queue);
        let mut continuation_guard = agent.turn_continuation_provider_guard(Arc::new(move || {
            let mut queue = continuation_queue
                .lock()
                .map_err(|_| anyhow!("turn continuation queue poisoned"))?;
            Ok(queue.drain_ready())
        }));
        let agent = continuation_guard.agent();
        let sender = self.event_tx.clone();
        let response = agent
            .run_stream_content_with_interactions_async(
                UserMessageContent::default(),
                move |delta| {
                    let sender = sender.clone();
                    let delta = delta.to_string();
                    async move {
                        send_optional_event(
                            &sender,
                            SessionTransportEvent::AssistantDelta(AssistantDeltaEvent::new(delta)),
                        )
                    }
                },
                {
                    let sender = self.event_tx.clone();
                    let transcript = self.transcript.clone();
                    let reconciliation = Arc::new(tokio::sync::Mutex::new(
                        ToolBatchReconciliation::default(),
                    ));
                    move |event| {
                        let sender = sender.clone();
                        let transcript = transcript.clone();
                        let reconciliation = Arc::clone(&reconciliation);
                        async move {
                            let journal_effect = if let Some(transcript) = transcript.as_ref() {
                                let mut recorder = transcript
                                    .lock()
                                    .map_err(|_| anyhow!("transcript recorder poisoned"))?;
                                persist_agent_event(&mut recorder, &event)?
                            } else {
                                JournalEffect {
                                    persisted: false,
                                    context_projection: ContextProjection::None,
                                    compaction_terminal: false,
                                }
                            };
                            let reconciliation_action = {
                                let mut state = reconciliation.lock().await;
                                state.reconcile(&event, journal_effect)
                            };
                            match event {
                                AgentEvent::ToolCallStarted { call_id, name, args } => {
                                    send_optional_event(
                                        &sender,
                                        SessionTransportEvent::ToolStarted(ToolStartedEvent {
                                            call_id,
                                            name: name.clone(),
                                            summary: format_tool_call(&name, &args),
                                            arguments: Some(args.to_string()),
                                        }),
                                    )?;
                                }
                                AgentEvent::ToolCallFinished {
                                    call_id,
                                    name,
                                    ok,
                                    output,
                                } => {
                                    send_optional_event(
                                        &sender,
                                        SessionTransportEvent::ToolFinished(ToolFinishedEvent {
                                            call_id,
                                            name,
                                            summary: output_summary(&output)
                                                .unwrap_or_else(|| "tool completed".into()),
                                            outcome: if ok {
                                                ToolOutcome::Success
                                            } else {
                                                ToolOutcome::Failure
                                            },
                                            output: Some(output_json(&output).to_string()),
                                        }),
                                    )?;
                                }
                                AgentEvent::ToolCallBatchFinished => {
                                    let ToolBatchReconciliationAction::FinishBatch(disposition) =
                                        reconciliation_action
                                    else {
                                        unreachable!("tool batch boundary must finish reconciliation");
                                    };
                                    emit_context_projection_update(
                                        &sender,
                                        &transcript,
                                        None,
                                        None,
                                        None,
                                        disposition,
                                    )?;
                                    send_optional_event(&sender, SessionTransportEvent::ToolBatchFinished)?;
                                }
                                AgentEvent::ReasoningDelta { item_id, delta } => {
                                    send_optional_event(
                                        &sender,
                                        SessionTransportEvent::ReasoningDelta(
                                            ReasoningDeltaEvent::new(item_id, delta),
                                        ),
                                    )?;
                                }
                                AgentEvent::ReasoningDone { item_id, text } => {
                                    send_optional_event(
                                        &sender,
                                        SessionTransportEvent::ReasoningDone(ReasoningDoneEvent::new(
                                            item_id, text,
                                        )),
                                    )?;
                                }
                                AgentEvent::TodoSnapshotUpdated { items } => {
                                    send_optional_event(
                                        &sender,
                                        SessionTransportEvent::TodoSnapshot(TodoSnapshotEvent::new(items)),
                                    )?;
                                }
                                AgentEvent::AutoContinueChanged { state } => {
                                    send_optional_event(
                                        &sender,
                                        SessionTransportEvent::AutoContinueChanged(
                                            AutoContinueChangedEvent::new(state),
                                        ),
                                    )?;
                                }
                                AgentEvent::TurnContinuationBoundary => {
                                    tokio::task::yield_now().await;
                                }
                                _ => {}
                            }
                            Ok(())
                        }
                    }
                },
                |_| async { Ok(PermissionApproval::Deny) },
                |request| async move {
                    Err(anyhow!(
                        "question tool is unavailable during background continuation; received {} question(s)",
                        request.questions.len()
                    ))
                },
            )
            .await;
        match response {
            Ok(message) => {
                self.emit(SessionTransportEvent::AssistantDone { message_id: None })?;
                self.emit(SessionTransportEvent::Done)?;
                Ok(message)
            }
            Err(error) => {
                self.emit(SessionTransportEvent::Error(ErrorEvent::new(format!(
                    "{error:#}"
                ))))?;
                self.emit(SessionTransportEvent::Done)?;
                Err(error)
            }
        }
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
        let tool_batch_state =
            Arc::new(tokio::sync::Mutex::new(ToolBatchReconciliation::default()));
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
                        let tool_batch_state = Arc::clone(&tool_batch_state);
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
                            let reconciliation_action = {
                                let mut state = tool_batch_state.lock().await;
                                state.reconcile(&event, journal_effect)
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
                                AgentEvent::LlmRequestTelemetry(telemetry) => {
                                    if telemetry.phase == crate::agent::LlmRequestTelemetryPhase::Prepared {
                                        send_scoped_event(
                                            &sender,
                                            child_session_id.as_deref(),
                                            agent_name.as_deref(),
                                            parent_tool_call_id.as_deref(),
                                            SessionTransportEvent::PreparedTokenUsage(
                                                TokenUsageEvent::with_breakdown(
                                                    telemetry.estimated_request_tokens,
                                                    telemetry.context_window_tokens,
                                                    telemetry.estimated_request_tokens,
                                                    0,
                                                    0,
                                                )
                                                .with_prompt_composition(telemetry.prompt_composition.clone()),
                                            ),
                                        )?;
                                    }
                                }
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
                                AgentEvent::EvidenceRecorded(_)
                                    if reconciliation_action
                                        == ToolBatchReconciliationAction::ProjectEvidence => {
                                    emit_context_projection_updates(
                                        &sender,
                                        &transcript,
                                        child_session_id.as_deref(),
                                        agent_name.as_deref(),
                                        parent_tool_call_id.as_deref(),
                                    )?;
                                }
                                AgentEvent::EvidenceRecorded(_) => {}
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
                                    send_scoped_event(
                                        &sender,
                                        child_session_id.as_deref(),
                                        agent_name.as_deref(),
                                        parent_tool_call_id.as_deref(),
                                        SessionTransportEvent::ToolFinished(finished),
                                    )?;
                                    // The durable terminal event is authoritative; the completed
                                    // batch owns the single projection flush.
                                }
                                AgentEvent::ToolCallBatchFinished => {
                                    let ToolBatchReconciliationAction::FinishBatch(disposition) =
                                        reconciliation_action
                                    else {
                                        unreachable!("tool batch boundary must finish reconciliation");
                                    };
                                    emit_context_projection_update(
                                        &sender,
                                        &transcript,
                                        child_session_id.as_deref(),
                                        agent_name.as_deref(),
                                        parent_tool_call_id.as_deref(),
                                        disposition,
                                    )?;
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
                                AgentEvent::TurnContinuationBoundary => {
                                    tokio::task::yield_now().await;
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
            let _ = status_tx.send(SessionTransportEvent::Notice(NoticeEvent::info(message)));
            Ok(())
        }),
        Arc::new(move |message| {
            let _ = error_tx.send(SessionTransportEvent::Error(ErrorEvent::new(message)));
            Ok(())
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

#[cfg(test)]
mod tests {
    use super::formatting::summarize_search;
    use super::*;
    use crate::agent::{
        AutoContinueState, CacheUsageReport, SubagentInvocation, TodoItem, TodoStatus,
    };
    use crate::permission::PermissionRequest;
    use crate::session::{PermissionDecision, SessionEvent};
    use crate::transcript::TranscriptRecorder;
    use async_openai::{Client, config::OpenAIConfig};
    use serde_json::json;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn summarize_search_uses_aggregate_and_marks_folded() {
        assert_eq!(
            summarize_search(&json!({
                "matches": [],
                "total_matches": 843,
                "files": 12,
                "folded": true,
            })),
            "843 matches · 12 files · folded"
        );
        // Legacy payloads without aggregates fall back to the inline array.
        assert_eq!(
            summarize_search(&json!({
                "matches": [{"path": "a"}, {"path": "b"}],
            })),
            "2 matches"
        );
    }

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

    fn credential_delegate(
        parent: &mut Agent<OpenAIConfig>,
        default_route: crate::config::ModelRoute,
        allowed_models: Vec<crate::config::ModelRoute>,
    ) -> RunnerSubagentDelegate {
        let provider = crate::config::ProviderConfig {
            base_url: "http://127.0.0.1:9/v1".into(),
            auth_mode: crate::config::ProviderAuthMode::ApiKey,
            api_key: "expert-key".into(),
            protocol: crate::config::ApiProtocol::Completions,
            default_model: "shared".into(),
            retry: None,
            models: indexmap::IndexMap::from([(
                "shared".into(),
                crate::config::ModelConfig {
                    display_name: None,
                    protocol: crate::config::ApiProtocol::Completions,
                    anthropic_thinking: Default::default(),
                    anthropic_betas: Vec::new(),
                    cache_control: false,
                    context_window: None,
                    effective_input_limit_tokens: None,
                    max_output_tokens: None,
                    supports_tools: false,
                    supports_reasoning: false,
                    reasoning_effort: None,
                    reasoning_efforts: Vec::new(),
                    reasoning_summary: None,
                    text_verbosity: None,
                    temperature: None,
                    top_p: None,
                    prompt_cache: crate::config::PromptCacheConfig::default(),
                    parallel_tool_calls: false,
                },
            )]),
        };
        let factory = crate::subagent::ExpertRouteFactory::new_with_policies(
            [("explorer".into(), Some(default_route), allowed_models)],
            &indexmap::IndexMap::from([("expert".into(), provider)]),
            &crate::config::RetryConfig::default(),
        )
        .expect("credential test factory");
        parent.set_subagent_child_factory(Arc::new(factory));
        RunnerSubagentDelegate {
            runtime: SubagentPool::new(),
            sessions_dir: std::env::temp_dir(),
            transcript: temp_transcript(),
            event_tx: None,
            route_api_key_configured: indexmap::IndexMap::from([
                ("primary/shared".into(), true),
                ("expert/shared".into(), false),
            ]),
            retained_session_routes: std::collections::HashSet::new(),
            provider_api_key_hints: indexmap::IndexMap::from([(
                "expert".into(),
                "Set EXPERT_API_KEY.".into(),
            )]),
            api_key_hint: "Set <PROVIDER>_API_KEY.".into(),
            background_control_tx: None,
            background_event_tx: None,
            background_child_started_tx: None,
        }
    }

    async fn drive_tool_batch_events(
        events: Vec<(AgentEvent, JournalEffect)>,
    ) -> Vec<&'static str> {
        let state = Arc::new(tokio::sync::Mutex::new(ToolBatchReconciliation::default()));
        let mut trace = Vec::new();
        for (event, journal_effect) in events {
            let event_name = match &event {
                AgentEvent::ToolCallFinished { .. } => "tool_finished",
                AgentEvent::EvidenceRecorded(_) => "evidence",
                AgentEvent::ToolCallBatchFinished => "batch_finished",
                AgentEvent::ToolCallCancelled { .. } => "tool_cancelled",
                _ => "other",
            };
            if journal_effect.persisted {
                trace.push(match event_name {
                    "tool_finished" => "persist:tool_finished",
                    "evidence" => "persist:evidence",
                    "batch_finished" => "persist:batch_finished",
                    "tool_cancelled" => "persist:tool_cancelled",
                    _ => "persist:other",
                });
            }
            tokio::task::yield_now().await;
            let action = {
                let mut state = state.lock().await;
                state.reconcile(&event, journal_effect)
            };
            match action {
                ToolBatchReconciliationAction::ProjectEvidence => trace.push("project"),
                ToolBatchReconciliationAction::FinishBatch(_) => {
                    trace.push("boundary:batch_finished");
                    trace.push("project");
                    trace.push("ui:batch_finished");
                }
                ToolBatchReconciliationAction::None => match event_name {
                    "tool_finished" => trace.push("ui:tool_finished"),
                    "tool_cancelled" => trace.push("ui:tool_cancelled"),
                    _ => {}
                },
            }
        }
        trace
    }

    fn test_evidence() -> crate::evidence::EvidenceRecord {
        crate::evidence::EvidenceRecord {
            id: "evidence-1".into(),
            sequence: 1,
            timestamp_ms: 0,
            evidence_kind: crate::evidence::EvidenceKind::CommandResult,
            title: "command result".into(),
            summary: "command completed".into(),
            detail: None,
            source: crate::evidence::EvidenceSource::Command {
                command: "cargo check".into(),
                status: Some(0),
            },
            tags: Vec::new(),
        }
    }

    #[tokio::test]
    async fn tool_batch_event_orchestration_persists_before_ui_and_projects_once() {
        let output = crate::tool::ToolResult::ok("shell__exec", json!({}));
        let trace = drive_tool_batch_events(vec![
            (
                AgentEvent::ToolCallFinished {
                    call_id: "call-1".into(),
                    name: "shell__exec".into(),
                    ok: true,
                    output,
                },
                JournalEffect {
                    persisted: true,
                    context_projection: ContextProjection::Advance,
                    compaction_terminal: false,
                },
            ),
            (
                AgentEvent::EvidenceRecorded(test_evidence()),
                JournalEffect {
                    persisted: true,
                    context_projection: ContextProjection::Advance,
                    compaction_terminal: false,
                },
            ),
            (
                AgentEvent::ToolCallBatchFinished,
                JournalEffect {
                    persisted: false,
                    context_projection: ContextProjection::None,
                    compaction_terminal: false,
                },
            ),
        ])
        .await;

        assert_eq!(
            trace,
            vec![
                "persist:tool_finished",
                "ui:tool_finished",
                "persist:evidence",
                "boundary:batch_finished",
                "project",
                "ui:batch_finished",
            ]
        );
        assert_eq!(trace.iter().filter(|entry| **entry == "project").count(), 1);
    }

    #[test]
    fn tool_batch_reconciliation_defers_projection_until_batch_boundary() {
        let mut reconciliation = ToolBatchReconciliation::default();
        reconciliation.observe_event(&AgentEvent::ToolCallFinished {
            call_id: "call-1".into(),
            name: "shell__exec".into(),
            ok: true,
            output: crate::tool::ToolResult::ok("shell__exec", json!({})),
        });

        assert!(!reconciliation.should_project_evidence());

        let disposition = reconciliation.finish();
        assert_eq!(disposition, RuntimeContextDisposition::Advance);
        assert!(reconciliation.should_project_evidence());
    }

    #[test]
    fn tool_batch_reconciliation_keeps_non_tool_evidence_on_its_own_boundary() {
        let mut reconciliation = ToolBatchReconciliation::default();
        assert!(reconciliation.should_project_evidence());

        reconciliation.record_journal_effect(JournalEffect {
            persisted: true,
            context_projection: ContextProjection::ReplaceScope,
            compaction_terminal: false,
        });
        assert_eq!(
            reconciliation.finish(),
            RuntimeContextDisposition::ReplaceScope
        );
        assert!(reconciliation.should_project_evidence());
    }

    #[test]
    fn permission_request_event_carries_subagent_origin() {
        let request = PermissionRequest {
            call_id: Some("call-8".into()),
            tool: "shell__exec".into(),
            args: json!({"command": "cargo test"}),
            class: crate::permission::ToolPermissionClass::Command,
            directive: crate::permission::ExecutionDirective::None,
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
    fn tool_driven_invalid_override_is_reported_before_credential_lookup() {
        let mut parent = Agent::new(Client::with_config(OpenAIConfig::new()), "shared", 1, 1);
        parent.set_primary_route(crate::config::ModelRoute::new("primary", "shared"));
        let selected = crate::config::ModelRoute::new("expert", "shared");
        let delegate = credential_delegate(
            &mut parent,
            crate::config::ModelRoute::new("expert", "shared"),
            Vec::new(),
        );
        let invocation = SubagentInvocation {
            input: crate::tool::NormalizedSubagentInput {
                objective: "inspect route credentials".into(),
                success_criteria: Vec::new(),
                allowed_paths: Vec::new(),
                forbidden_paths: Vec::new(),
                owned_paths: Vec::new(),
                timeout_secs: None,
                max_tool_calls: None,
                model: Some(selected.display_name()),
                target_child_session_id: None,
                background: false,
            },
            model: Some(selected),
            prompt: "inspect route credentials".into(),
            parent_tool_call_id: Some("call-invalid".into()),
        };

        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("create runtime");
        let output = runtime
            .block_on(delegate.run_named(&parent, "explorer", invocation))
            .expect("invalid route is a tool result");
        let message = &output.error.as_ref().expect("route error").message;
        assert!(message.contains("requested model route 'expert/shared' is not allowed"));
        assert!(!message.contains("API key is not set"));
    }

    #[test]
    fn tool_driven_expert_delegation_requires_its_route_credential() {
        let mut parent = Agent::new(Client::with_config(OpenAIConfig::new()), "shared", 1, 1);
        parent.set_primary_route(crate::config::ModelRoute::new("primary", "shared"));
        let delegate = credential_delegate(
            &mut parent,
            crate::config::ModelRoute::new("expert", "shared"),
            Vec::new(),
        );
        let invocation = SubagentInvocation {
            input: crate::tool::NormalizedSubagentInput {
                objective: "inspect route credentials".into(),
                success_criteria: Vec::new(),
                allowed_paths: Vec::new(),
                forbidden_paths: Vec::new(),
                owned_paths: Vec::new(),
                timeout_secs: None,
                max_tool_calls: None,
                model: None,
                target_child_session_id: None,
                background: false,
            },
            model: None,
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
    fn retained_current_expert_route_keeps_its_session_credential() {
        let mut parent = Agent::new(Client::with_config(OpenAIConfig::new()), "shared", 1, 1);
        parent.set_primary_route(crate::config::ModelRoute::new("primary", "shared"));
        let retained = crate::config::ModelRoute::new("expert", "shared");
        let mut delegate = credential_delegate(&mut parent, retained.clone(), Vec::new());
        delegate
            .route_api_key_configured
            .shift_remove(&retained.display_name());
        delegate
            .retained_session_routes
            .insert(retained.display_name());

        let has_credential = delegate
            .route_api_key_configured
            .get(&retained.display_name())
            .copied()
            .unwrap_or_else(|| {
                delegate
                    .retained_session_routes
                    .contains(&retained.display_name())
            });

        assert!(has_credential);
    }

    #[test]
    fn tool_driven_override_credential_check_uses_the_requested_route() {
        let mut parent = Agent::new(Client::with_config(OpenAIConfig::new()), "shared", 1, 1);
        parent.set_primary_route(crate::config::ModelRoute::new("primary", "shared"));
        let selected = crate::config::ModelRoute::new("expert", "shared");
        let delegate = credential_delegate(
            &mut parent,
            crate::config::ModelRoute::new("expert", "shared"),
            vec![selected.clone()],
        );
        let invocation = SubagentInvocation {
            input: crate::tool::NormalizedSubagentInput {
                objective: "inspect route credentials".into(),
                success_criteria: Vec::new(),
                allowed_paths: Vec::new(),
                forbidden_paths: Vec::new(),
                owned_paths: Vec::new(),
                timeout_secs: None,
                max_tool_calls: None,
                model: Some(selected.display_name()),
                target_child_session_id: None,
                background: false,
            },
            model: Some(selected),
            prompt: "inspect route credentials".into(),
            parent_tool_call_id: Some("call-2".into()),
        };

        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("create runtime");
        let output = runtime
            .block_on(delegate.run_named(&parent, "explorer", invocation))
            .expect("credential denial is a tool result");
        assert!(!output.ok);
        assert_eq!(
            output.data.as_ref().and_then(|data| data.get("route")),
            Some(&json!("expert/shared"))
        );
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
}
