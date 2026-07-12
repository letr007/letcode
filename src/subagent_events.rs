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
                            AgentEvent::ContextCompactionStarted
                            | AgentEvent::ContextCompactionDelta { .. }
                            | AgentEvent::TokenUsageUpdated { .. }
                            | AgentEvent::TurnStarted(_)
                            | AgentEvent::EvidenceRecorded(_) => {}
                            AgentEvent::ModelStreamIssue {
                                message,
                                detail,
                                action,
                            } => {
                                if let Some(error) = compaction_terminal_issue_transcript_error(
                                    &message,
                                    detail.as_deref(),
                                    &action,
                                ) {
                                    record_transcript(&transcript, |recorder| {
                                        recorder.record_error(error)
                                    })?;
                                }
                            }
                            AgentEvent::AssistantMessage { .. }
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

fn compaction_terminal_issue_transcript_error(
    message: &str,
    detail: Option<&str>,
    action: &str,
) -> Option<String> {
    if !matches!(
        message,
        "Context compaction failed" | "Context compaction cancelled"
    ) {
        return None;
    }

    let mut error = message.to_string();
    if let Some(detail) = detail.filter(|detail| !detail.trim().is_empty()) {
        error.push_str(": ");
        error.push_str(detail);
    }
    if !action.trim().is_empty() {
        error.push_str(" (");
        error.push_str(action);
        error.push(')');
    }
    Some(error)
}

#[cfg(test)]
mod tests {
    use super::{
        HEADLESS_CHILD_PERMISSION_DENIED_REASON, compaction_terminal_issue_transcript_error,
    };

    #[test]
    fn headless_child_permission_denial_reason_does_not_claim_tui_approval() {
        assert_eq!(
            HEADLESS_CHILD_PERMISSION_DENIED_REASON,
            "Denied in headless child execution"
        );
    }

    #[test]
    fn compaction_terminal_issue_records_child_transcript_error_text() {
        assert_eq!(
            compaction_terminal_issue_transcript_error(
                "Context compaction failed",
                Some("summary model returned empty output"),
                "Continuing without compaction",
            ),
            Some(
                "Context compaction failed: summary model returned empty output (Continuing without compaction)"
                    .to_string()
            )
        );
    }

    #[test]
    fn non_compaction_stream_issue_is_not_recorded_as_child_transcript_error() {
        assert_eq!(
            compaction_terminal_issue_transcript_error(
                "Model stream interrupted",
                Some("network read reset"),
                "Continuing",
            ),
            None
        );
    }
}
