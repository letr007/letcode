//! Canonical durable projection of agent stream events.
//!
//! UI, CLI, and subagent runners own presentation and non-agent transcript
//! entries (permissions, user input, session commands). Agent events enter the
//! transcript only through this module.

use anyhow::Result;

use crate::agent::AgentEvent;
use crate::transcript::TranscriptRecorder;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)] // ReplaceScope is consumed by session runner projection updates.
pub enum ContextProjection {
    None,
    Advance,
    ReplaceScope,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JournalEffect {
    pub persisted: bool,
    pub context_projection: ContextProjection,
    pub compaction_terminal: bool,
}

impl JournalEffect {
    const IGNORED: Self = Self {
        persisted: false,
        context_projection: ContextProjection::None,
        compaction_terminal: false,
    };

    const fn persisted(context_projection: ContextProjection) -> Self {
        Self {
            persisted: true,
            context_projection,
            compaction_terminal: false,
        }
    }
}

/// Persists the durable portion of one `AgentEvent` and reports its post-commit
/// effect. Errors are intentionally returned to the caller; audit-policy
/// decisions belong to the individual runner.
pub fn persist_agent_event(
    recorder: &mut TranscriptRecorder,
    event: &AgentEvent,
) -> Result<JournalEffect> {
    let effect = match event {
        AgentEvent::TurnStarted(event) => {
            recorder.clear_reasoning_observations();
            recorder.record_turn_started(event.clone())?;
            JournalEffect::persisted(ContextProjection::None)
        }
        AgentEvent::LlmRequestTelemetry(telemetry) => {
            recorder.record_llm_request_telemetry(telemetry.clone())?;
            JournalEffect::persisted(ContextProjection::None)
        }
        AgentEvent::EvidenceRecorded(evidence) => {
            recorder.record_evidence_record(evidence.clone())?;
            JournalEffect::persisted(ContextProjection::Advance)
        }
        AgentEvent::ReasoningDone { item_id, text } => {
            recorder.record_reasoning_message(item_id, text.clone())?;
            JournalEffect::persisted(ContextProjection::Advance)
        }
        AgentEvent::AssistantMessage { content } => {
            recorder.record_assistant_message(content.clone())?;
            JournalEffect::persisted(ContextProjection::None)
        }
        AgentEvent::AssistantToolCallBatch {
            text,
            reasoning_content,
            reasoning_wire,
            calls,
        } => {
            recorder.record_assistant_tool_call_batch(
                text.clone(),
                reasoning_content.clone(),
                reasoning_wire.clone(),
                calls.clone(),
            )?;
            JournalEffect::persisted(ContextProjection::None)
        }
        AgentEvent::InternalContinuation { text, source } => {
            recorder.record_internal_continuation(text.clone(), *source)?;
            JournalEffect::persisted(ContextProjection::None)
        }
        AgentEvent::ToolCallStarted {
            call_id,
            name,
            args,
        } => {
            recorder.record_tool_call_started(call_id.clone(), name.clone(), args.clone())?;
            JournalEffect::persisted(ContextProjection::None)
        }
        AgentEvent::ToolCallCancelled { call_id, name } => {
            recorder.record_tool_call_cancelled(call_id.clone(), name.clone())?;
            JournalEffect::persisted(ContextProjection::None)
        }
        AgentEvent::ToolCallFinished {
            call_id,
            name,
            ok,
            output,
        } => {
            recorder.record_tool_call_finished(
                call_id.clone(),
                name.clone(),
                *ok,
                output.clone(),
            )?;
            JournalEffect::persisted(ContextProjection::Advance)
        }
        AgentEvent::TodoSnapshotUpdated { items } => {
            recorder.record_todo_snapshot(items.clone())?;
            JournalEffect::persisted(ContextProjection::None)
        }
        AgentEvent::AutoContinueChanged { state } => {
            recorder.record_auto_continue_changed(state.clone())?;
            JournalEffect::persisted(ContextProjection::None)
        }
        AgentEvent::TurnContinuationBoundary => JournalEffect::IGNORED,
        AgentEvent::AutoContinuationScheduled {
            continuation_count,
            remaining_unfinished,
        } => {
            recorder
                .record_auto_continuation_scheduled(*continuation_count, *remaining_unfinished)?;
            JournalEffect::persisted(ContextProjection::None)
        }
        AgentEvent::ValidationAdvisory(advisory) => {
            recorder.record_validation_advisory(advisory.clone())?;
            JournalEffect::persisted(ContextProjection::None)
        }
        AgentEvent::ToolExecutionSummary(event) => {
            recorder.record_tool_execution_summary(event.clone())?;
            JournalEffect::persisted(ContextProjection::None)
        }
        AgentEvent::HistoryPublished {
            publication,
            revision,
        } => {
            recorder.record_history_event(
                crate::transcript::TranscriptEvent::HistoryPublished(publication.clone()),
                *revision,
            )?;
            JournalEffect::persisted(ContextProjection::Advance)
        }
        AgentEvent::HistoryApplied {
            application,
            revision,
            blocking,
        } => {
            recorder.record_history_event(
                crate::transcript::TranscriptEvent::HistoryApplied(application.clone()),
                *revision,
            )?;
            JournalEffect {
                persisted: true,
                context_projection: ContextProjection::Advance,
                compaction_terminal: *blocking,
            }
        }
        AgentEvent::ContextCompacted(event) => {
            recorder.record_context_compaction(event.clone())?;
            JournalEffect {
                persisted: true,
                context_projection: ContextProjection::Advance,
                compaction_terminal: true,
            }
        }
        AgentEvent::TurnFinalized(event) => {
            recorder.clear_reasoning_observations();
            recorder.record_turn_finalized(event.clone())?;
            JournalEffect::persisted(ContextProjection::None)
        }
        AgentEvent::ReasoningDelta { item_id, .. } => {
            recorder.observe_reasoning_delta(item_id);
            JournalEffect::IGNORED
        }
        AgentEvent::ContextCompactionStarted { .. }
        | AgentEvent::ContextCompactionNoProgress(_)
        | AgentEvent::ContextCompactionFailed { .. }
        | AgentEvent::ContextCompactionDelta { .. }
        | AgentEvent::TokenUsageUpdated { .. }
        | AgentEvent::FastModeChanged { .. }
        | AgentEvent::LlmRetryScheduled(_)
        | AgentEvent::LlmRetryStarted(_)
        | AgentEvent::ModelStreamIssue { .. }
        | AgentEvent::ToolCallPending { .. }
        | AgentEvent::ToolOutputDelta { .. }
        | AgentEvent::ToolCallBatchFinished => JournalEffect::IGNORED,
    };
    Ok(effect)
}

#[cfg(test)]
#[path = "../tests/internal/agent_event_journal/integration_tests.rs"]
mod integration_tests;
