//! Rebuild a `Timeline` model from persisted transcript records.
//!
//! The read side of the transcript pipeline: `TranscriptRecord`s written to disk
//! become a renderable `Timeline`. Backend-agnostic and independent of the
//! terminal rendering layers (`transcript_render` / `transcript_ratatui` /
//! `components::transcript`).

use super::events::{
    AutoContinueChangedEvent, ErrorEvent, TodoSnapshotEvent, ToolFinishedEvent, ToolOutcome,
    ToolStartedEvent, UserMessageEvent,
};
use super::timeline::{
    DelegationView, MessageRole, PermissionPromptStatus, Timeline, restored_tool_summary,
};
use crate::agent::AutoContinueState;
use crate::tool_format::format_tool_call;
use crate::transcript::{TranscriptEvent, TranscriptRecord};
use crate::user_content::UserMessageSubmission;

pub(crate) fn timeline_from_transcript_records(records: &[TranscriptRecord]) -> Timeline {
    let mut projection = TranscriptTimelineProjection::default();
    projection.timeline.begin_bulk_restore();
    for record in records {
        projection.apply_record(record);
    }
    projection.timeline.finish_bulk_restore();
    projection.timeline
}

#[derive(Debug, Default)]
struct TranscriptTimelineProjection {
    timeline: Timeline,
    current_auto_continue: AutoContinueState,
    /// The child's delegation prompt is recorded as a user message; the delegation block replaces it.
    pending_delegation_prompt: bool,
}

impl TranscriptTimelineProjection {
    fn apply_record(&mut self, record: &TranscriptRecord) {
        match &record.event {
            TranscriptEvent::UserMessage { content } => {
                self.current_auto_continue = AutoContinueState::default();
                if std::mem::take(&mut self.pending_delegation_prompt) {
                    return;
                }
                self.timeline
                    .push_user_message(UserMessageEvent::from_submission(
                        UserMessageSubmission::new(
                            format!("restored-user-message-{}", record.sequence),
                            content.clone(),
                        ),
                    ))
            }
            TranscriptEvent::AssistantTurn(turn) => {
                if let Some(content) = turn.text.clone() {
                    self.timeline
                        .push_restored_message(MessageRole::Assistant, content);
                }
            }
            TranscriptEvent::AssistantMessage { content } => self
                .timeline
                .push_restored_message(MessageRole::Assistant, content.clone()),
            TranscriptEvent::ContextCompaction(event) => {
                // Durable transcript: restore the full compaction block with summary.
                self.timeline
                    .push_restored_compaction(event.summary.clone());
            }
            TranscriptEvent::ReasoningMessage {
                content,
                duration_ms,
            } => {
                self.timeline.push_restored_reasoning(
                    format!("restored-reasoning-{}", record.sequence),
                    content.clone(),
                    *duration_ms,
                );
            }
            TranscriptEvent::ContextExperimentReturned {
                branch_id,
                outcome,
                summary,
                next_action,
                had_writes,
                ..
            } => self.timeline.push_restored_message(
                MessageRole::Assistant,
                crate::transcript::format_context_experiment_return(
                    branch_id,
                    outcome,
                    summary,
                    next_action.as_deref(),
                    *had_writes,
                ),
            ),
            TranscriptEvent::ToolCallStarted {
                call_id,
                name,
                args,
            } => {
                let started = ToolStartedEvent {
                    call_id: call_id.clone(),
                    name: name.clone(),
                    summary: format_tool_call(name, args),
                    arguments: Some(args.to_string()),
                };
                if name == crate::tool_names::TOOL_AGENT_WAIT
                    && args
                        .get("run_id")
                        .and_then(serde_json::Value::as_str)
                        .is_some_and(|run_id| self.timeline.begin_subagent_wait(call_id, run_id))
                {
                    return;
                }
                self.timeline.push_tool_started(started);
            }
            TranscriptEvent::ToolCallFinished {
                call_id,
                name,
                ok,
                output,
            } => {
                self.timeline.push_tool_finished(ToolFinishedEvent {
                    call_id: call_id.clone(),
                    name: name.clone(),
                    summary: restored_tool_summary(name, *ok),
                    outcome: if *ok {
                        ToolOutcome::Success
                    } else {
                        ToolOutcome::Failure
                    },
                    output: serde_json::to_value(output)
                        .ok()
                        .map(|value| value.to_string()),
                });
            }
            TranscriptEvent::ToolCallCancelled { call_id, name } => {
                if name == crate::tool_names::TOOL_AGENT_WAIT
                    && self.timeline.cancel_foreground_subagent_wait(call_id)
                {
                } else {
                    self.timeline.cancel_tool(call_id, name);
                }
            }
            TranscriptEvent::TodoSnapshot { items } => {
                self.timeline
                    .push_todo_snapshot(TodoSnapshotEvent::new(items.clone()));
                self.timeline
                    .apply_auto_continue_changed(AutoContinueChangedEvent::new(
                        self.current_auto_continue.clone(),
                    ));
            }
            TranscriptEvent::AutoContinueChanged { state } => {
                self.current_auto_continue = state.clone();
                self.timeline
                    .apply_auto_continue_changed(AutoContinueChangedEvent::new(state.clone()));
            }
            TranscriptEvent::PermissionDecision {
                call_id,
                tool,
                args,
                allowed,
                reason,
                reviewer,
                approval,
                risk,
                reviewer_child_session_id,
            } => {
                let restored_reason = reason.clone();
                let auto_review = matches!(reviewer.as_deref(), Some("auto"));
                let origin_label = auto_review.then(|| "reviewer".to_string());
                let reviewer_child_session_id = reviewer_child_session_id.clone();
                let call_id = call_id.clone().unwrap_or_else(|| {
                    if auto_review {
                        format!("restored-auto-review-{}", record.sequence)
                    } else {
                        tool.clone()
                    }
                });
                self.timeline.push_restored_permission_decision(
                    call_id,
                    tool.clone(),
                    format_tool_call(tool, args),
                    Some(args.to_string()),
                    if *allowed {
                        PermissionPromptStatus::Approved
                    } else {
                        PermissionPromptStatus::Denied
                    },
                    restored_reason,
                    origin_label,
                    approval.clone(),
                    risk.clone(),
                    reviewer_child_session_id,
                );
            }
            TranscriptEvent::Error { message } => {
                self.current_auto_continue = AutoContinueState::default();
                self.timeline.push_error(ErrorEvent::new(message.clone()));
            }
            TranscriptEvent::TurnInterrupted { .. } => {
                self.current_auto_continue = AutoContinueState::default();
                self.timeline.cancel_foreground_subagent_waits();
                self.timeline.cancel_active_tools();
            }
            TranscriptEvent::TurnFinalized(event) => {
                if event.outcome == "interrupted" {
                    self.current_auto_continue = AutoContinueState::default();
                    self.timeline.cancel_foreground_subagent_waits();
                    self.timeline.cancel_active_tools();
                }
            }
            TranscriptEvent::SubagentStarted {
                run_id,
                child_session_id,
                agent_name,
                summary,
                ..
            } => {
                self.timeline.register_subagent_started(
                    run_id,
                    child_session_id,
                    agent_name,
                    summary,
                );
            }
            TranscriptEvent::SubagentDelegation { input, .. } => {
                self.pending_delegation_prompt = true;
                self.timeline.push_restored_delegation(DelegationView {
                    objective: input.objective.clone(),
                    success_criteria: input.success_criteria.clone(),
                    owned_paths: input.owned_paths.clone(),
                    allowed_paths: input.allowed_paths.clone(),
                    forbidden_paths: input.forbidden_paths.clone(),
                    model: input.model.clone(),
                    target_child_session_id: input.target_child_session_id.clone(),
                });
            }
            TranscriptEvent::SubagentResult { .. }
            | TranscriptEvent::SubagentActivated { .. }
            | TranscriptEvent::SubagentLifecycle { .. }
            | TranscriptEvent::LlmRequestTelemetry { .. }
            | TranscriptEvent::SessionStarted { .. }
            | TranscriptEvent::SessionWorkspace { .. }
            | TranscriptEvent::SessionSubagentsPolicy { .. }
            | TranscriptEvent::SessionTitle { .. }
            | TranscriptEvent::ContextBranchCreated { .. }
            | TranscriptEvent::ContextBranchSummary { .. }
            | TranscriptEvent::ContextCheckout { .. }
            | TranscriptEvent::HistoryNavigation { .. }
            | TranscriptEvent::ContextExperimentStarted { .. }
            | TranscriptEvent::ContextNodeCreated { .. }
            | TranscriptEvent::ContextNodeLifecycle { .. }
            | TranscriptEvent::ContextViewOperationMetadata { .. }
            | TranscriptEvent::HistoryPublished(_)
            | TranscriptEvent::HistoryApplied(_)
            | TranscriptEvent::ContextSummaryArtifactMetadata { .. }
            | TranscriptEvent::FoldedOutputMetadata { .. }
            | TranscriptEvent::TurnStarted(_)
            | TranscriptEvent::ModelChanged { .. }
            | TranscriptEvent::ReasoningEffortChanged { .. }
            | TranscriptEvent::PermissionModeChanged { .. }
            | TranscriptEvent::FakeClientChanged { .. }
            | TranscriptEvent::AutoContinuationScheduled { .. }
            | TranscriptEvent::AssistantToolCallBatch { .. }
            | TranscriptEvent::InternalContinuation { .. }
            | TranscriptEvent::ValidationAdvisory(_)
            | TranscriptEvent::ToolExecutionSummary(_)
            | TranscriptEvent::LogicalCheckpoint(_)
            | TranscriptEvent::Evidence { .. }
            | TranscriptEvent::HistorianExchange(_)
            | TranscriptEvent::Unknown => {}
        }
    }
}

#[cfg(test)]
#[path = "../../tests/internal/tui/transcript_read_model/tests.rs"]
mod tests;
