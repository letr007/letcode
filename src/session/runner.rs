use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use anyhow::{Result, anyhow};
use tokio::sync::oneshot;
use tracing::warn;

use crate::agent::{Agent, AgentEvent, SubagentDelegate};
use crate::agent_event_journal::{ContextProjection, JournalEffect, persist_agent_event};
use crate::permission::PermissionApproval;
use crate::session::child_view::ChildLiveText;
use crate::subagent::SubagentPool;
use crate::subagent_events::SubagentEventSender;
use crate::transcript::{
    TranscriptRecorder, any_record_where, read_records, record_is_session_title,
    record_is_user_message,
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
    AssistantDeltaEvent, AutoContinueChangedEvent, ErrorEvent, NoticeEvent, NoticeId,
    ProcessIssueEvent, ReasoningDeltaEvent, ReasoningDoneEvent, RuntimeContextDisposition,
    SessionEvent, TodoSnapshotEvent, TokenUsageEvent, ToolCancelledEvent, ToolFinishedEvent,
    ToolOutcome, ToolOutputDeltaEvent, ToolPendingEvent, ToolStartedEvent, UserMessageEvent,
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

pub(crate) struct AgentRunner {
    event_tx: Option<SessionTransportEventSender>,
    permission_event_tx: Option<SessionTransportEventSender>,
    session_title_event_tx: Option<SessionTransportEventSender>,
    transcript: Option<Arc<Mutex<TranscriptRecorder>>>,
    event_mode: SessionTransportEventMode,
    child_session_id: Option<String>,
    permission_origin: Option<String>,
    parent_tool_call_id: Option<String>,
    subagent_delegate: Option<Arc<dyn SubagentDelegate>>,
    child_live_text: Option<ChildLiveText>,
}

impl AgentRunner {
    pub fn with_subagent_runtime(
        self,
        runtime: SubagentPool,
        sessions_dir: PathBuf,
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
                child_completions: None,
                runtime,
                sessions_dir,
                transcript,
                event_tx: self_.event_tx.clone(),
                background_event_tx,
                child_live_text: self_.child_live_text.clone(),
                #[cfg(test)]
                background_child_started_tx,
                route_api_key_configured,
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
            route_api_key_configured,
            provider_api_key_hints,
            api_key_hint,
            background_control_tx,
            background_event_tx,
            Some(background_child_started_tx),
        )
    }

    pub(crate) fn prepare_subagent_runtime(&self, agent: &mut Agent) -> Result<()> {
        if let Some(delegate) = self.subagent_delegate.clone() {
            agent.set_subagent_delegate(delegate);
        }
        agent.prepare_subagent_runtime()
    }

    #[cfg(test)]
    pub(crate) fn install_subagent_delegate_for_test(&self, agent: &mut Agent) {
        if let Some(delegate) = self.subagent_delegate.clone() {
            agent.set_subagent_delegate(delegate);
        }
    }
}

impl AgentRunner {
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
            child_live_text: None,
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
            child_live_text: None,
        }
    }

    pub fn with_child_live_text(mut self, child_live_text: ChildLiveText) -> Self {
        self.child_live_text = Some(child_live_text);
        self
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
            child_live_text: None,
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
            child_live_text: None,
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
            child_live_text: None,
        }
    }

    pub async fn run_prompt(
        &self,
        agent: &mut Agent,
        prompt: UserMessageSubmission,
    ) -> Result<String>
where {
        let queue = Arc::new(Mutex::new(crate::agent::TurnContinuationQueue::default()));
        self.run_prompt_with_continuations(agent, prompt, queue)
            .await
    }

    pub(crate) async fn run_prompt_with_continuations(
        &self,
        agent: &mut Agent,
        prompt: UserMessageSubmission,
        turn_continuation_queue: Arc<Mutex<crate::agent::TurnContinuationQueue>>,
    ) -> Result<String>
where {
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
    pub async fn continue_session(&self, agent: &mut Agent) -> Result<String>
where {
        let queue = Arc::new(Mutex::new(crate::agent::TurnContinuationQueue::default()));
        self.run_existing_history_with_continuations(agent, queue)
            .await
    }

    #[cfg(test)]
    pub(crate) async fn run_existing_history(&self, agent: &mut Agent) -> Result<String>
where {
        let queue = Arc::new(Mutex::new(crate::agent::TurnContinuationQueue::default()));
        self.run_existing_history_with_continuations(agent, queue)
            .await
    }

    pub(crate) async fn run_existing_history_with_continuations(
        &self,
        agent: &mut Agent,
        turn_continuation_queue: Arc<Mutex<crate::agent::TurnContinuationQueue>>,
    ) -> Result<String>
where {
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
        agent.prepare_subagent_runtime()?;
        self.install_member_message_recorder(agent);
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
                                AgentEvent::LlmRetryScheduled(retry) => {
                                    send_optional_event(
                                        &sender,
                                        SessionTransportEvent::RetryScheduled(retry_lifecycle_event(retry)),
                                    )?;
                                }
                                AgentEvent::LlmRetryStarted(retry) => {
                                    send_optional_event(
                                        &sender,
                                        SessionTransportEvent::RetryStarted(retry_lifecycle_event(retry)),
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
                self.emit(SessionTransportEvent::Error(error_event(&error)))?;
                self.emit(SessionTransportEvent::Done)?;
                Err(error)
            }
        }
    }

    async fn run_prompt_with_options(
        &self,
        agent: &mut Agent,
        prompt: UserMessageSubmission,
        record_user_prompt: bool,
    ) -> Result<String>
where {
        let prompt_content = prompt.content.clone();
        let prompt_text = prompt_content.text.clone();
        if self.child_session_id.is_none()
            && let Some(transcript) = &self.transcript
        {
            let recorder = transcript
                .lock()
                .map_err(|_| anyhow!("transcript recorder poisoned"))?;
            if let Err(error) = crate::project_memory::enroll(&recorder) {
                warn!(error = %error, "could not register project memory source");
                self.emit(SessionTransportEvent::Notice(
                    NoticeEvent::info(
                        "Project memory recording is unavailable; see the application log",
                    )
                    .with_message_id(NoticeId::ProjectMemoryUnavailable),
                ))?;
            }
        }
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
        agent.prepare_subagent_runtime()?;
        self.install_member_message_recorder(agent);
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
        let child_live_text = self.child_live_text.clone();
        let delta_live_text = child_live_text.clone();
        let required_turn_close =
            agent.subagents_policy().mode == crate::config::SubagentMode::Swarm;
        let response = agent
            .run_stream_content_with_interactions_async(
                prompt_content.clone(),
                move |delta| {
                    let sender = sender.clone();
                    let child_session_id = child_session_id.clone();
                    let agent_name = agent_name.clone();
                    let parent_tool_call_id = parent_tool_call_id.clone();
                    let live_text = delta_live_text.clone();
                    let delta = delta.to_string();
                    async move {
                        if let (Some(live_text), Some(child_session_id)) =
                            (live_text.as_ref(), child_session_id.as_deref())
                        {
                            live_text.append(child_session_id, &delta);
                        }
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
                    let child_live_text = child_live_text.clone();
                    move |event| {
                        let sender = sender.clone();
                        let transcript = transcript.clone();
                        let child_session_id = child_session_id.clone();
                        let agent_name = agent_name.clone();
                        let parent_tool_call_id = parent_tool_call_id.clone();
                        let child_live_text = child_live_text.clone();
                        let tool_batch_state = Arc::clone(&tool_batch_state);
                        async move {
                            if let (Some(live_text), Some(child_session_id)) =
                                (child_live_text.as_ref(), child_session_id.as_deref())
                            {
                                clear_live_text_before_assistant_persist(
                                    live_text,
                                    child_session_id,
                                    &event,
                                );
                            }
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
                                        ) && !(required_turn_close && matches!(event, AgentEvent::TurnFinalized(_))) => {
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
                                    if child_session_id.is_none() {
                                        send_optional_event(
                                            &sender,
                                            SessionTransportEvent::FastModeChanged { enabled },
                                        )?;
                                    }
                                    send_scoped_event(
                                        &sender,
                                        child_session_id.as_deref(),
                                        agent_name.as_deref(),
                                        parent_tool_call_id.as_deref(),
                                        SessionTransportEvent::Notice(
                                            NoticeEvent::info(
                                                "Fast mode auto-disabled: current model is unavailable",
                                            )
                                            .with_message_id(NoticeId::FastModeAutoDisabled),
                                        ),
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
                                AgentEvent::HistoryPublished { .. } => {}
                                AgentEvent::HistoryApplied { blocking, .. } => {
                                    let _ = emit_context_projection_updates(&sender, &transcript, child_session_id.as_deref(), agent_name.as_deref(), parent_tool_call_id.as_deref());
                                    if blocking {
                                        let _ = send_scoped_event(&sender, child_session_id.as_deref(), agent_name.as_deref(), parent_tool_call_id.as_deref(), SessionTransportEvent::CompactionCommitted { summary: None });
                                    }
                                }
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
                let event = error_event(&error);
                let error_message = recorded_error_message(&event);
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

    fn install_member_message_recorder(&self, agent: &mut Agent) {
        let transcript = self.transcript.clone();
        let sender = self.event_tx.clone();
        let child = self.child_session_id.clone();
        let origin = self.permission_origin.clone();
        let parent_call = self.parent_tool_call_id.clone();
        agent.set_member_message_recorder(Arc::new(move |message| {
            let submission = UserMessageSubmission::new(
                crate::subagent::generate_message_id(),
                UserMessageContent::from(message),
            );
            let event =
                SessionTransportEvent::UserMessage(UserMessageEvent::from_submission(submission));
            let event = match &child {
                Some(child) => wrap_child_session_transport_event(
                    child.clone(),
                    origin.clone(),
                    parent_call.clone(),
                    event,
                ),
                None => event,
            };
            send_optional_event(&sender, event)?;
            record_transcript(&transcript, |recorder| {
                recorder.record_user_message(message)
            })?;
            emit_context_projection_updates(
                &sender,
                &transcript,
                child.as_deref(),
                origin.as_deref(),
                parent_call.as_deref(),
            )
        }));
    }

    fn emit(&self, event: SessionTransportEvent) -> Result<()> {
        let event = if let Some(child_session_id) = &self.child_session_id {
            if let Some(live_text) = &self.child_live_text {
                clear_live_text_at_child_terminal(live_text, child_session_id, &event);
            }
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
        self.emit(SessionTransportEvent::Error(error_event(&error)))?;
        self.emit(SessionTransportEvent::Done)?;
        Err(error)
    }

    fn pending_session_title(
        &self,
        agent: &Agent,
        record_user_prompt: bool,
    ) -> Result<Option<(String, Agent)>>
where {
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
        let already_asked = any_record_where(&path, |record| {
            record_is_user_message(record) || record_is_session_title(record)
        })?;
        if already_asked {
            return Ok(None);
        }

        Ok(Some((session_id, agent.session_title_agent())))
    }
}

fn error_event(error: &anyhow::Error) -> ErrorEvent {
    let event = ErrorEvent::new(format!("{error:#}"));
    let Some(detail) = error
        .downcast_ref::<crate::model_runtime::ModelFailure>()
        .map(|failure| failure.detail())
        .filter(|detail| !detail.is_empty())
    else {
        return event;
    };
    event.with_details(detail)
}

/// Live text holds only what the journal does not yet: drop the accumulated tail
/// before the assistant message becomes durable so a projection cannot show the
/// same text twice.
pub(crate) fn clear_live_text_before_assistant_persist(
    live_text: &ChildLiveText,
    child_session_id: &str,
    event: &AgentEvent,
) {
    if matches!(
        event,
        AgentEvent::AssistantMessage { .. } | AgentEvent::AssistantToolCallBatch { .. }
    ) {
        live_text.clear(child_session_id);
    }
}

/// A child's streaming tail is only meaningful while its turn is live; drop it
/// when the producer emits a terminal event so no partial text outlives the turn.
pub(crate) fn clear_live_text_at_child_terminal(
    live_text: &ChildLiveText,
    child_session_id: &str,
    event: &SessionTransportEvent,
) {
    if matches!(
        event,
        SessionTransportEvent::AssistantDone { .. }
            | SessionTransportEvent::Error(_)
            | SessionTransportEvent::Done
    ) {
        live_text.clear(child_session_id);
    }
}

fn recorded_error_message(event: &ErrorEvent) -> String {
    match event.details.as_deref() {
        Some(details) if !details.is_empty() => {
            format!("{}\n\nProvider response:\n{}", event.message, details)
        }
        _ => event.message.clone(),
    }
}

/// Source for [`subagent_event_sender`]. A bare sender carries no live text; a
/// sender paired with [`ChildLiveText`] lets the child runner own its buffer.
pub(crate) struct SubagentEventSenderSource {
    event_tx: SessionTransportEventSender,
    child_live_text: Option<ChildLiveText>,
}

impl From<SessionTransportEventSender> for SubagentEventSenderSource {
    fn from(event_tx: SessionTransportEventSender) -> Self {
        Self {
            event_tx,
            child_live_text: None,
        }
    }
}

impl From<(SessionTransportEventSender, Option<ChildLiveText>)> for SubagentEventSenderSource {
    fn from(
        (event_tx, child_live_text): (SessionTransportEventSender, Option<ChildLiveText>),
    ) -> Self {
        Self {
            event_tx,
            child_live_text,
        }
    }
}

pub(crate) fn subagent_event_sender(
    source: impl Into<SubagentEventSenderSource>,
) -> SubagentEventSender {
    let source = source.into();
    build_subagent_event_sender(source.event_tx, source.child_live_text)
}

fn build_subagent_event_sender(
    event_tx: SessionTransportEventSender,
    child_live_text: Option<ChildLiveText>,
) -> SubagentEventSender {
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
                let runner: AgentRunner = if let Some(permission_origin) = permission_origin {
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
                let runner = match child_live_text.as_ref() {
                    Some(live_text) => runner.with_child_live_text(live_text.clone()),
                    None => runner,
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
#[path = "../../tests/internal/session/runner/tests.rs"]
mod tests;
