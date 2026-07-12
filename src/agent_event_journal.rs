//! Canonical durable projection of agent stream events.
//!
//! UI, CLI, and subagent runners own presentation and non-agent transcript
//! entries (permissions, user input, session commands). Agent events enter the
//! transcript only through this module.

use anyhow::Result;

use crate::agent::AgentEvent;
use crate::tool_names;
use crate::transcript::TranscriptRecorder;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
            recorder.record_turn_started(event.clone())?;
            JournalEffect::persisted(ContextProjection::None)
        }
        AgentEvent::EvidenceRecorded(evidence) => {
            recorder.record_evidence_record(evidence.clone())?;
            JournalEffect::persisted(ContextProjection::Advance)
        }
        AgentEvent::ReasoningDone { text, .. } => {
            recorder.record_reasoning_message(text.clone())?;
            JournalEffect::persisted(ContextProjection::Advance)
        }
        AgentEvent::AssistantMessage { content } => {
            recorder.record_assistant_message(content.clone())?;
            JournalEffect::persisted(ContextProjection::None)
        }
        AgentEvent::AssistantToolCallBatch { text, calls } => {
            recorder.record_assistant_tool_call_batch(text.clone(), calls.clone())?;
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
            recorder.record_tool_call_finished_and_apply_context_control(
                call_id.clone(),
                name.clone(),
                *ok,
                output.clone(),
            )?;
            recorder.record_context_tool_pending_metadata(name, *ok, output)?;
            let projection = if *ok
                && matches!(
                    name.as_str(),
                    tool_names::TOOL_CONTEXT_CHECKPOINT | tool_names::TOOL_CONTEXT_RETURN
                ) {
                ContextProjection::ReplaceScope
            } else {
                ContextProjection::Advance
            };
            JournalEffect::persisted(projection)
        }
        AgentEvent::TodoSnapshotUpdated { items } => {
            recorder.record_todo_snapshot(items.clone())?;
            JournalEffect::persisted(ContextProjection::None)
        }
        AgentEvent::AutoContinueChanged { state } => {
            recorder.record_auto_continue_changed(state.clone())?;
            JournalEffect::persisted(ContextProjection::None)
        }
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
        AgentEvent::ContextCompacted(event) => {
            recorder.record_context_compaction(event.clone())?;
            JournalEffect {
                persisted: true,
                context_projection: ContextProjection::Advance,
                compaction_terminal: true,
            }
        }
        AgentEvent::TurnFinalized(event) => {
            recorder.record_turn_finalized(event.clone())?;
            JournalEffect::persisted(ContextProjection::None)
        }
        AgentEvent::ContextCompactionStarted
        | AgentEvent::ContextCompactionDelta { .. }
        | AgentEvent::TokenUsageUpdated { .. }
        | AgentEvent::ReasoningDelta { .. }
        | AgentEvent::ModelStreamIssue { .. }
        | AgentEvent::ToolCallPending { .. }
        | AgentEvent::ToolOutputDelta { .. }
        | AgentEvent::ToolCallBatchFinished => JournalEffect::IGNORED,
    };
    Ok(effect)
}

#[cfg(test)]
mod tests {
    use super::{ContextProjection, persist_agent_event};
    use crate::agent::{AgentEvent, ContextCompactionEvent};
    use crate::evidence::EvidenceDraft;
    use crate::tool::ToolResult;
    use crate::transcript::{TranscriptEvent, TranscriptRecorder, read_records};
    use serde_json::json;

    fn recorder(name: &str) -> TranscriptRecorder {
        let directory = std::env::temp_dir().join(format!(
            "letcode-agent-event-journal-{name}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        TranscriptRecorder::create(directory).expect("create recorder")
    }

    #[test]
    fn table_driven_durable_and_ephemeral_events_have_expected_effects() {
        let mut recorder = recorder("table");
        let evidence = recorder
            .record_evidence(EvidenceDraft::from_tool_result(
                "call-source",
                "fs__read",
                json!({"path": "src/main.rs"}),
                &ToolResult::ok("fs__read", json!({"content": "ok"})),
            ))
            .expect("seed evidence");
        let cases = [
            (
                AgentEvent::EvidenceRecorded(evidence),
                true,
                ContextProjection::Advance,
            ),
            (
                AgentEvent::AssistantMessage {
                    content: "answer".into(),
                },
                true,
                ContextProjection::None,
            ),
            (
                AgentEvent::ReasoningDelta {
                    item_id: "r1".into(),
                    delta: "partial".into(),
                },
                false,
                ContextProjection::None,
            ),
        ];

        for (event, persisted, projection) in cases {
            let effect = persist_agent_event(&mut recorder, &event).expect("persist event");
            assert_eq!(effect.persisted, persisted);
            assert_eq!(effect.context_projection, projection);
        }

        let records = read_records(recorder.path()).expect("read records");
        assert!(matches!(records[1].event, TranscriptEvent::Evidence { .. }));
        assert!(matches!(
            records[2].event,
            TranscriptEvent::AssistantMessage { .. }
        ));
    }

    #[test]
    fn tool_finish_effects_distinguish_normal_checkpoint_and_return() {
        let mut recorder = recorder("tool-finish");
        let normal = AgentEvent::ToolCallFinished {
            call_id: "read-1".into(),
            name: "fs__read".into(),
            ok: true,
            output: ToolResult::ok("fs__read", json!({"content": "ok"})),
        };
        assert_eq!(
            persist_agent_event(&mut recorder, &normal)
                .expect("normal finish")
                .context_projection,
            ContextProjection::Advance
        );

        let checkpoint = AgentEvent::ToolCallFinished {
            call_id: "checkpoint-1".into(),
            name: crate::tool_names::TOOL_CONTEXT_CHECKPOINT.into(),
            ok: true,
            output: ToolResult::ok(
                crate::tool_names::TOOL_CONTEXT_CHECKPOINT,
                json!({
                    "label": "Explore parser", "reason": "isolate change",
                    "context_only": true, "filesystem_rolled_back": false
                }),
            ),
        };
        assert_eq!(
            persist_agent_event(&mut recorder, &checkpoint)
                .expect("checkpoint finish")
                .context_projection,
            ContextProjection::ReplaceScope
        );

        let returned = AgentEvent::ToolCallFinished {
            call_id: "return-1".into(),
            name: crate::tool_names::TOOL_CONTEXT_RETURN.into(),
            ok: true,
            output: ToolResult::ok(
                crate::tool_names::TOOL_CONTEXT_RETURN,
                json!({
                    "outcome": "useful", "summary": "found it",
                    "context_restored": true, "filesystem_rolled_back": false
                }),
            ),
        };
        assert_eq!(
            persist_agent_event(&mut recorder, &returned)
                .expect("return finish")
                .context_projection,
            ContextProjection::ReplaceScope
        );
    }

    #[test]
    fn compaction_terminal_is_durable() {
        let mut recorder = recorder("compaction");
        let effect = persist_agent_event(
            &mut recorder,
            &AgentEvent::ContextCompacted(ContextCompactionEvent {
                outcome: "failed".into(),
                summary: String::new(),
                tail_start_index: 0,
                original_history_items: 0,
                retained_history_items: 0,
                retired_source_spans: Vec::new(),
                frame_identity_bindings: Vec::new(),
                detail: Some("empty summary".into()),
            }),
        )
        .expect("persist compaction");
        assert!(effect.persisted && effect.compaction_terminal);
    }
}
