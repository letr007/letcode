use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use anyhow::{Result, anyhow};
use async_openai::config::Config;

use crate::agent::{Agent, AgentEvent};
use crate::agent_event_journal::persist_agent_event;
use crate::permission::{PermissionApproval, PermissionRequest};
use crate::transcript::{TranscriptRecorder, read_records_allow_partial_tail};

type SubagentPromptFuture = Pin<Box<dyn Future<Output = Result<String>> + Send>>;
type EmitFn = Arc<dyn Fn(String) -> Result<()> + Send + Sync>;
const HEADLESS_CHILD_PERMISSION_DENIED_REASON: &str = "Denied in headless child execution";
type PromptFn<C> = Arc<
    dyn Fn(
            Agent<C>,
            String,
            Arc<Mutex<TranscriptRecorder>>,
            String,
            Option<String>,
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
    parent_tool_call_id: Option<String>,
}

impl<C: Config> SubagentEventSender<C> {
    pub fn new(emit_status: EmitFn, emit_error: EmitFn, run_prompt: PromptFn<C>) -> Self {
        Self {
            emit_status,
            emit_error,
            run_prompt,
            parent_tool_call_id: None,
        }
    }

    pub fn with_parent_tool_call_id(mut self, parent_tool_call_id: Option<String>) -> Self {
        self.parent_tool_call_id = parent_tool_call_id;
        self
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
            self.parent_tool_call_id.clone(),
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

    {
        let runtime_transcript = Arc::clone(&transcript);
        agent.set_runtime_snapshot_provider(Arc::new(move || {
            let recorder = runtime_transcript
                .lock()
                .map_err(|_| anyhow!("transcript recorder poisoned"))?;
            let records = read_records_allow_partial_tail(recorder.path())?;
            Ok(
                crate::transcript::transcript_projection::project_runtime_restore_snapshot(
                    recorder.session_id().to_string(),
                    records,
                    crate::transcript::transcript_projection::SessionContextCursor {
                        branch_id: recorder.current_context_branch_id().map(str::to_string),
                        leaf_sequence: None,
                    },
                    &[],
                )?
                .snapshot,
            )
        }));
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
                        // Permission decisions are independent of AgentEvent journaling.
                        record_transcript(&transcript, |recorder| {
                            persist_agent_event(recorder, &event).map(|_| ())
                        })?;
                        match event {
                            AgentEvent::ContextCompactionStarted { .. }
                            | AgentEvent::ContextCompactionNoProgress(_)
                            | AgentEvent::ContextCompactionFailed { .. }
                            | AgentEvent::ContextCompactionDelta { .. }
                            | AgentEvent::TokenUsageUpdated { .. }
                            | AgentEvent::LlmRequestTelemetry(_)
                            | AgentEvent::TurnStarted(_)
                            | AgentEvent::EvidenceRecorded(_) => {}
                            AgentEvent::ModelStreamIssue { .. }
                            | AgentEvent::AssistantMessage { .. }
                            | AgentEvent::AssistantToolCallBatch { .. }
                            | AgentEvent::InternalContinuation { .. }
                            | AgentEvent::ReasoningDelta { .. }
                            | AgentEvent::ReasoningDone { .. }
                            | AgentEvent::ToolCallPending { .. }
                            | AgentEvent::ToolCallStarted { .. }
                            | AgentEvent::ToolCallCancelled { .. }
                            | AgentEvent::ToolCallFinished { .. }
                            | AgentEvent::ToolOutputDelta { .. }
                            | AgentEvent::ToolCallBatchFinished
                            | AgentEvent::TodoSnapshotUpdated { .. }
                            | AgentEvent::AutoContinueChanged { .. }
                            | AgentEvent::AutoContinuationScheduled { .. }
                            | AgentEvent::ValidationAdvisory(_)
                            | AgentEvent::ToolExecutionSummary(_)
                            | AgentEvent::ContextCompacted(_)
                            | AgentEvent::TurnFinalized(_) => {}
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
                                Some(HEADLESS_CHILD_PERMISSION_DENIED_REASON.into()),
                            )
                        })?;
                        Ok(PermissionApproval::Deny)
                    }
                }
            },
        )
        .await;

    match response {
        Ok(message) => Ok(message),
        Err(error) => {
            let error_message = format!("{error:#}");
            record_transcript(&transcript, |recorder| {
                recorder.record_error(error_message.clone())
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

#[cfg(test)]
mod tests {
    use super::HEADLESS_CHILD_PERMISSION_DENIED_REASON;

    #[test]
    fn headless_child_permission_denial_reason_does_not_claim_tui_approval() {
        assert_eq!(
            HEADLESS_CHILD_PERMISSION_DENIED_REASON,
            "Denied in headless child execution"
        );
    }
}
