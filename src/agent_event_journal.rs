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
mod tests {
    use super::{ContextProjection, JournalEffect, persist_agent_event};
    use crate::agent::{
        AgentEvent, CompactionBlocker, CompactionNoProgress, CompactionTrigger,
        ContextCompactionEvent, LlmRequestTelemetry, LlmRequestTelemetryPhase,
        ProviderUsageCompleteness, TokenUsageEstimate,
    };
    use crate::config::ApiProtocol;
    use crate::evidence::EvidenceDraft;
    use crate::tool::ToolResult;
    use crate::transcript::{
        TranscriptEvent, TranscriptRecorder, has_session_content, read_records,
        restore_runtime_snapshot, restore_session_evidence, restore_session_history,
    };
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

    fn telemetry(
        phase: LlmRequestTelemetryPhase,
        attempt: usize,
        usage: Option<TokenUsageEstimate>,
    ) -> AgentEvent {
        AgentEvent::LlmRequestTelemetry(LlmRequestTelemetry {
            logical_request_id: "turn-7-iteration-2".into(),
            turn_id: 7,
            iteration: 2,
            attempt,
            phase,
            model: "test-model".into(),
            protocol: ApiProtocol::Responses,
            context_window_tokens: 1_000,
            input_budget_tokens: 800,
            estimated_request_tokens: 600,
            estimated_prelude_tokens: 100,
            estimated_protected_tokens: 50,
            protected_safe_ceiling_tokens: 600,
            protected_reserve_tokens: 200,
            estimated_unaddressable_protected_tokens: 10,
            estimated_retained_history_tokens: 200,
            estimated_tools_tokens: 100,
            estimated_evidence_tokens: 50,
            estimated_required_fallback_tokens: 20,
            original_history_items: 4,
            retained_history_items: 3,
            dropped_history_items: 1,
            selected_evidence_items: 1,
            dropped_evidence_items: 0,
            selected_evidence_ids: vec!["evidence-1".into()],
            evidence_fingerprint: "fte-v1-test".into(),
            truncated: true,
            prompt_segment_count: 4,
            prompt_contributor_count: 3,
            prompt_composition: Vec::new(),
            prompt_stable_prefix_hash: Some("opaque-prefix-hash".into()),
            cache_first_volatile_index: Some(2),
            cache_configured: true,
            cache_hint_serialized: true,
            cache_retention_sent: None,
            cache_stable_prefix_segments: 2,
            cache_stable_prompt_tokens: 400,
            cache_volatile_prompt_tokens: 200,
            cacheable_prefix_tokens: 350,
            cache_stable_after_boundary_tokens: 50,
            local_prefix_fingerprint: Some("opaque-fingerprint".into()),
            routing_key: Some("opaque-routing-key".into()),
            tool_call_count_before: 1,
            tool_definitions_count: 2,
            adjacent_lcp_units: Some(2),
            adjacent_lcp_bytes: Some(128),
            adjacent_lcp_estimated_tokens: Some(32),
            current_unit_count: 3,
            first_breaker: None,
            cohort_comparable: true,
            cohort_changed: false,
            usage,
            usage_completeness: ProviderUsageCompleteness::Complete,
            cache_write_tokens: None,
            provider_response_id: Some("opaque-response-id".into()),
            error_class: None,
        })
    }

    #[test]
    fn terminal_turn_events_clear_unfinished_reasoning_observations() {
        fn assert_boundary_clears(name: &str, boundary: impl FnOnce(&mut TranscriptRecorder)) {
            let mut recorder = recorder(name);
            persist_agent_event(
                &mut recorder,
                &AgentEvent::ReasoningDelta {
                    item_id: "reused-reasoning".into(),
                    delta: "Draft".into(),
                },
            )
            .expect("observe unfinished reasoning");
            boundary(&mut recorder);
            persist_agent_event(
                &mut recorder,
                &AgentEvent::ReasoningDone {
                    item_id: "reused-reasoning".into(),
                    text: "Recovered".into(),
                },
            )
            .expect("persist later reasoning without a new delta");

            let records = read_records(recorder.path()).expect("read records");
            assert!(matches!(
                records.last(),
                Some(crate::transcript::TranscriptRecord {
                    event: TranscriptEvent::ReasoningMessage {
                        content,
                        duration_ms: None,
                    },
                    ..
                }) if content == "Recovered"
            ));
        }

        assert_boundary_clears("reasoning-turn-start-clear", |recorder| {
            persist_agent_event(
                recorder,
                &AgentEvent::TurnStarted(crate::agent::TurnStartedEvent {
                    turn_id: 2,
                    intent: "continue".into(),
                    directive: "continue".into(),
                    validation_reminder: String::new(),
                }),
            )
            .expect("start next turn");
        });
        assert_boundary_clears("reasoning-turn-final-clear", |recorder| {
            persist_agent_event(
                recorder,
                &AgentEvent::TurnFinalized(crate::agent::TurnFinalizedEvent {
                    turn_id: 1,
                    outcome: "completed".into(),
                    tool_call_count: 0,
                    continuation_count: 0,
                    write_effects: 0,
                    validation_effects: 0,
                    failed_validation_effects: 0,
                    validation_advisory_emitted: false,
                }),
            )
            .expect("finalize turn");
        });
        assert_boundary_clears("reasoning-turn-interrupt-clear", |recorder| {
            recorder
                .record_turn_interrupted(Some(1))
                .expect("interrupt turn");
        });
    }

    #[test]
    fn reasoning_duration_persists_from_first_delta_and_missing_start_stays_unknown() {
        let mut recorder = recorder("reasoning-duration");
        let delta = AgentEvent::ReasoningDelta {
            item_id: "reasoning-1".into(),
            delta: "Draft".into(),
        };
        let done = AgentEvent::ReasoningDone {
            item_id: "reasoning-1".into(),
            text: "Final".into(),
        };

        assert_eq!(
            persist_agent_event(&mut recorder, &delta).expect("observe reasoning delta"),
            JournalEffect::IGNORED
        );
        std::thread::sleep(std::time::Duration::from_millis(2));
        persist_agent_event(&mut recorder, &done).expect("persist reasoning");
        persist_agent_event(
            &mut recorder,
            &AgentEvent::ReasoningDone {
                item_id: "reasoning-without-delta".into(),
                text: "Recovered".into(),
            },
        )
        .expect("persist reasoning without delta");

        let records = read_records(recorder.path()).expect("read records");
        assert!(matches!(
            records.as_slice(),
            [
                crate::transcript::TranscriptRecord {
                    event: TranscriptEvent::ReasoningMessage {
                        content: first,
                        duration_ms: Some(duration_ms),
                    },
                    ..
                },
                crate::transcript::TranscriptRecord {
                    event: TranscriptEvent::ReasoningMessage {
                        content: second,
                        duration_ms: None,
                    },
                    ..
                }
            ] if first == "Final" && *duration_ms >= 1 && second == "Recovered"
        ));
    }

    #[test]
    fn assistant_tool_call_reasoning_content_persists_and_restores() {
        let mut recorder = recorder("assistant-tool-reasoning");
        let event = AgentEvent::AssistantToolCallBatch {
            text: None,
            reasoning_content: Some("inspect the requested file".into()),
            reasoning_wire: Some(
                r#"[{"type":"thinking","thinking":"inspect","signature":"signed"}]"#.into(),
            ),
            calls: vec![crate::request_builder::HistoryToolCall {
                call_id: "call-1".into(),
                name: "fs__read".into(),
                arguments_json: r#"{"path":"src/main.rs"}"#.into(),
            }],
        };

        let effect = persist_agent_event(&mut recorder, &event).expect("persist tool call batch");
        assert!(effect.persisted);
        persist_agent_event(
            &mut recorder,
            &AgentEvent::ToolCallFinished {
                call_id: "call-1".into(),
                name: "fs__read".into(),
                ok: true,
                output: ToolResult::ok("fs__read", json!({"path":"src/main.rs"})),
            },
        )
        .expect("persist tool output");
        let records = read_records(recorder.path()).expect("read records");
        assert!(matches!(
            &records[0].event,
            TranscriptEvent::AssistantTurn(turn)
                if turn.reasoning_content.as_deref() == Some("inspect the requested file")
                    && turn
                        .replay
                        .as_ref()
                        .and_then(crate::model_runtime::OpaqueReplayState::payload_json)
                        .is_some_and(|wire| wire.contains("\"signature\":\"signed\""))
        ));
        let old: TranscriptEvent =
            serde_json::from_str(r#"{"kind":"assistant_tool_call_batch","text":null,"calls":[]}"#)
                .expect("old tool-call batch deserializes");
        assert!(matches!(
            old,
            TranscriptEvent::AssistantToolCallBatch {
                reasoning_content: None,
                reasoning_wire: None,
                ..
            }
        ));
        assert!(matches!(
            restore_session_history(&records).expect("restore history").as_slice(),
            [
                crate::request_builder::HistoryItem::AssistantTurn {
                    reasoning_content: Some(reasoning_content),
                    replay: Some(replay),
                    ..
                },
                crate::request_builder::HistoryItem::ToolOutput { .. }
            ] if reasoning_content == "inspect the requested file"
                && replay
                    .payload_json()
                    .is_some_and(|wire| wire.contains("\"signature\":\"signed\""))
        ));
    }

    #[test]
    fn telemetry_terminal_failure_is_durable_and_opaque_ids_never_leak() {
        let mut recorder = recorder("telemetry-terminal");
        let mut event = telemetry(LlmRequestTelemetryPhase::Failed, 1, None);
        let AgentEvent::LlmRequestTelemetry(telemetry) = &mut event else {
            unreachable!()
        };
        telemetry.error_class = Some(crate::agent::LlmRequestErrorClass::RequestCreation);
        telemetry.provider_response_id =
            Some("SECRET-PROMPT-OR-TOOL-OR-EVIDENCE-CONTENT\nhttps://example.invalid/token".into());
        persist_agent_event(&mut recorder, &event).expect("persist failure");
        let records = read_records(recorder.path()).expect("read failure");
        let json = serde_json::to_string(&records[0].event).expect("serialize failure");
        assert!(json.contains("\"phase\":\"failed\""));
        assert!(json.contains("\"error_class\":\"request_creation\""));
        assert!(json.contains("opaque-"));
        assert!(!json.contains("SECRET-PROMPT-OR-TOOL-OR-EVIDENCE-CONTENT"));
        assert!(!json.contains("https://example.invalid"));
    }

    #[test]
    fn compaction_terminal_is_durable() {
        let mut recorder = recorder("compaction");
        recorder
            .record_user_message("retained")
            .expect("record retained entry");
        let effect = persist_agent_event(
            &mut recorder,
            &AgentEvent::ContextCompacted(ContextCompactionEvent::succeeded_at(
                "durable summary",
                Some("raw:1".into()),
            )),
        )
        .expect("persist compaction");
        assert!(effect.persisted && effect.compaction_terminal);
    }

    #[test]
    fn ephemeral_compaction_lifecycle_events_are_never_journaled() {
        let mut recorder = recorder("compaction-lifecycle");
        let events = [
            AgentEvent::ContextCompactionStarted {
                trigger: CompactionTrigger::Manual,
            },
            AgentEvent::ContextCompactionNoProgress(CompactionNoProgress {
                trigger: CompactionTrigger::Manual,
                blockers: vec![CompactionBlocker::NoHistoricalItems],
            }),
            AgentEvent::ContextCompactionFailed {
                trigger: CompactionTrigger::Manual,
            },
            AgentEvent::ContextCompactionDelta {
                delta: "preview only".into(),
            },
        ];
        for event in events {
            assert!(
                !persist_agent_event(&mut recorder, &event)
                    .expect("ignore lifecycle event")
                    .persisted
            );
        }
        assert!(
            read_records(recorder.path())
                .expect("read records")
                .is_empty()
        );
    }

    #[test]
    fn compaction_deltas_are_not_persisted_alongside_the_durable_summary() {
        let mut recorder = recorder("compaction-preview");
        recorder
            .record_user_message("retained")
            .expect("record retained entry");
        let delta = AgentEvent::ContextCompactionDelta {
            delta: "transient preview".into(),
        };
        assert!(
            !persist_agent_event(&mut recorder, &delta)
                .expect("ignore preview")
                .persisted
        );
        persist_agent_event(
            &mut recorder,
            &AgentEvent::ContextCompacted(ContextCompactionEvent::succeeded_at(
                "durable summary",
                Some("raw:1".into()),
            )),
        )
        .expect("persist durable summary");

        let records = read_records(recorder.path()).expect("read records");
        assert_eq!(records.len(), 2);
        assert!(matches!(
            records[1].event,
            TranscriptEvent::ContextCompaction { .. }
        ));
        assert!(
            !serde_json::to_string(&records)
                .expect("serialize records")
                .contains("transient preview")
        );
    }
}
