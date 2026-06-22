use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use anyhow::{Result, anyhow};
use async_openai::config::Config;

use crate::agent::{Agent, AgentEvent};
use crate::permission::PermissionRequest;
use crate::transcript::TranscriptRecorder;

type SubagentPromptFuture = Pin<Box<dyn Future<Output = Result<String>> + Send>>;
type EmitFn = Arc<dyn Fn(String) -> Result<()> + Send + Sync>;
type PromptFn<C> = Arc<
    dyn Fn(
            Agent<C>,
            String,
            Arc<Mutex<TranscriptRecorder>>,
            String,
            Option<String>,
        ) -> SubagentPromptFuture
        + Send
        + Sync,
>;

#[derive(Clone)]
pub struct SubagentEventSender<C: Config> {
    emit_status: EmitFn,
    emit_error: EmitFn,
    run_prompt: PromptFn<C>,
}

impl<C: Config> SubagentEventSender<C> {
    pub fn new(emit_status: EmitFn, emit_error: EmitFn, run_prompt: PromptFn<C>) -> Self {
        Self {
            emit_status,
            emit_error,
            run_prompt,
        }
    }

    pub fn emit_status(&self, message: String) -> Result<()> {
        (self.emit_status)(message)
    }

    pub fn emit_error(&self, message: String) -> Result<()> {
        (self.emit_error)(message)
    }

    pub async fn run_child_prompt(
        &self,
        agent: Agent<C>,
        prompt: String,
        transcript: Arc<Mutex<TranscriptRecorder>>,
        child_session_id: String,
        permission_origin: Option<String>,
    ) -> Result<String> {
        (self.run_prompt)(
            agent,
            prompt,
            transcript,
            child_session_id,
            permission_origin,
        )
        .await
    }
}

pub async fn run_child_prompt<C>(
    mut agent: Agent<C>,
    prompt: String,
    transcript: Arc<Mutex<TranscriptRecorder>>,
    sender: Option<SubagentEventSender<C>>,
    child_session_id: String,
    permission_origin: Option<String>,
) -> Result<String>
where
    C: Config + Clone + Send + Sync + 'static,
{
    if let Some(sender) = sender {
        return sender
            .run_child_prompt(
                agent,
                prompt,
                transcript,
                child_session_id,
                permission_origin,
            )
            .await;
    }

    record_transcript(&transcript, |recorder| {
        recorder.record_user_message(prompt.clone())
    })?;

    let response = agent
        .run_stream_async(
            &prompt,
            |_| async { Ok(()) },
            {
                let transcript = Arc::clone(&transcript);
                move |event| {
                    let transcript = Arc::clone(&transcript);
                    async move {
                        match event {
                            AgentEvent::TokenUsageUpdated { .. } => {}
                            AgentEvent::TurnStarted(event) => {
                                record_transcript(&transcript, |recorder| {
                                    recorder.record_turn_started(event)
                                })?;
                            }
                            AgentEvent::EvidenceRecorded(evidence) => {
                                record_transcript(&transcript, |recorder| {
                                    recorder.record_evidence_record(evidence.clone())
                                })?;
                            }
                            AgentEvent::ReasoningDelta { .. } => {}
                            AgentEvent::ReasoningDone { text, .. } => {
                                record_transcript(&transcript, |recorder| {
                                    recorder.record_reasoning_message(text.clone())
                                })?;
                            }
                            AgentEvent::ToolCallPending { .. } => {}
                            AgentEvent::ToolCallStarted {
                                call_id,
                                name,
                                args,
                            } => {
                                record_transcript(&transcript, |recorder| {
                                    recorder.record_tool_call_started(call_id, name, args)
                                })?;
                            }
                            AgentEvent::ToolCallFinished {
                                call_id,
                                name,
                                ok,
                                output,
                            } => {
                                record_transcript(&transcript, |recorder| {
                                    recorder.record_tool_call_finished(call_id, name, ok, output)
                                })?;
                            }
                            AgentEvent::ToolCallBatchFinished => {}
                            AgentEvent::TodoSnapshotUpdated { items } => {
                                record_transcript(&transcript, |recorder| {
                                    recorder.record_todo_snapshot(items)
                                })?;
                            }
                            AgentEvent::AutoContinueChanged { state } => {
                                record_transcript(&transcript, |recorder| {
                                    recorder.record_auto_continue_changed(state)
                                })?;
                            }
                            AgentEvent::AutoContinuationScheduled {
                                continuation_count,
                                remaining_unfinished,
                            } => {
                                record_transcript(&transcript, |recorder| {
                                    recorder.record_auto_continuation_scheduled(
                                        continuation_count,
                                        remaining_unfinished,
                                    )
                                })?;
                            }
                            AgentEvent::ValidationAdvisory(advisory) => {
                                record_transcript(&transcript, |recorder| {
                                    recorder.record_validation_advisory(advisory)
                                })?;
                            }
                            AgentEvent::ToolExecutionSummary(event) => {
                                record_transcript(&transcript, |recorder| {
                                    recorder.record_tool_execution_summary(event)
                                })?;
                            }
                            AgentEvent::ContextCompacted(event) => {
                                record_transcript(&transcript, |recorder| {
                                    recorder.record_context_compaction(event)
                                })?;
                            }
                            AgentEvent::TurnFinalized(event) => {
                                record_transcript(&transcript, |recorder| {
                                    recorder.record_turn_finalized(event)
                                })?;
                            }
                        }
                        Ok(())
                    }
                }
            },
            {
                let transcript = Arc::clone(&transcript);
                move |request: PermissionRequest| {
                    let transcript = Arc::clone(&transcript);
                    async move {
                        record_transcript(&transcript, |recorder| {
                            recorder.record_permission_decision_details(
                                request.call_id,
                                request.tool,
                                request.args,
                                false,
                                Some("Denied by user from TUI permission prompt".into()),
                            )
                        })?;
                        Ok(false)
                    }
                }
            },
        )
        .await;

    match response {
        Ok(message) => {
            record_transcript(&transcript, |recorder| {
                recorder.record_assistant_message(message.clone())
            })?;
            Ok(message)
        }
        Err(error) => {
            record_transcript(&transcript, |recorder| {
                recorder.record_error(error.to_string())
            })?;
            Err(error)
        }
    }
}

pub fn emit_status<C: Config>(sender: &Option<SubagentEventSender<C>>, message: String) {
    if let Some(sender) = sender {
        let _ = sender.emit_status(message);
    }
}

pub fn emit_error<C: Config>(sender: &Option<SubagentEventSender<C>>, message: String) {
    if let Some(sender) = sender {
        let _ = sender.emit_error(message);
    }
}

fn record_transcript<F>(transcript: &Arc<Mutex<TranscriptRecorder>>, f: F) -> Result<()>
where
    F: FnOnce(&mut TranscriptRecorder) -> Result<()>,
{
    let mut recorder = transcript
        .lock()
        .map_err(|_| anyhow!("transcript recorder poisoned"))?;
    f(&mut recorder)
}
