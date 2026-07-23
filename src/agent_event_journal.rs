//! Canonical durable projection of agent stream events.
//!
//! UI, CLI, and subagent runners own presentation and non-agent transcript
//! entries (permissions, user input, session commands). Agent events enter the
//! transcript only through this module.

use anyhow::{Result, bail};

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
        AgentEvent::LlmRequestTelemetry(telemetry) => {
            recorder.record_llm_request_telemetry(telemetry.clone())?;
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
        AgentEvent::LogicalCheckpoint {
            expected_journal_frontier,
            expected_branch_id,
            event,
        } => {
            recorder.record_logical_checkpoint_at_frontier(
                *expected_journal_frontier,
                expected_branch_id,
                event.clone(),
            )?;
            JournalEffect::persisted(ContextProjection::Advance)
        }
        AgentEvent::TurnFinalized(event) => {
            recorder.record_turn_finalized(event.clone())?;
            JournalEffect::persisted(ContextProjection::None)
        }
        AgentEvent::ContextCompactionStarted { .. }
        | AgentEvent::ContextCompactionNoProgress(_)
        | AgentEvent::ContextCompactionFailed { .. }
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
            estimated_foldable_protected_tokens: 40,
            estimated_provider_folded_protected_tokens: 10,
            estimated_unaddressable_protected_tokens: 10,
            provider_folded_output_count: 1,
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
    fn llm_request_telemetry_persists_prepared_completed_and_retries_as_audit_only_records() {
        let mut recorder = recorder("llm-telemetry");
        let provider_usage = TokenUsageEstimate {
            used_tokens: 120,
            context_window_tokens: 1_000,
            input_tokens: 100,
            output_tokens: 20,
            cached_tokens: 80,
        };
        for event in [
            telemetry(LlmRequestTelemetryPhase::Prepared, 1, None),
            telemetry(LlmRequestTelemetryPhase::Prepared, 2, None),
            telemetry(LlmRequestTelemetryPhase::Completed, 2, Some(provider_usage)),
        ] {
            let effect = persist_agent_event(&mut recorder, &event).expect("persist telemetry");
            assert!(effect.persisted);
            assert_eq!(effect.context_projection, ContextProjection::None);
        }

        let records = read_records(recorder.path()).expect("read telemetry");
        assert!(!has_session_content(&records));
        assert!(
            restore_session_history(&records)
                .expect("restore history")
                .is_empty()
        );
        assert!(
            restore_session_evidence(&records)
                .expect("restore evidence")
                .is_empty()
        );
        assert!(
            restore_runtime_snapshot(&records)
                .expect("restore runtime snapshot")
                .frames
                .is_empty()
        );
        let telemetry = records
            .iter()
            .map(|record| &record.event)
            .collect::<Vec<_>>();
        let TranscriptEvent::LlmRequestTelemetry {
            logical_request_id,
            attempt,
            phase,
            provider_cached_tokens,
            ..
        } = telemetry[0]
        else {
            panic!("prepared telemetry")
        };
        assert_eq!(logical_request_id, "turn-7-iteration-2");
        assert_eq!(
            (*attempt, phase.as_str(), *provider_cached_tokens),
            (1, "prepared", None)
        );
        let TranscriptEvent::LlmRequestTelemetry {
            logical_request_id,
            attempt,
            phase,
            provider_cached_tokens,
            ..
        } = telemetry[1]
        else {
            panic!("retry telemetry")
        };
        assert_eq!(
            (logical_request_id.as_str(), *attempt, phase.as_str()),
            ("turn-7-iteration-2", 2, "prepared")
        );
        assert_eq!(*provider_cached_tokens, None);
        let TranscriptEvent::LlmRequestTelemetry {
            logical_request_id,
            attempt,
            phase,
            provider_cached_tokens,
            ..
        } = telemetry[2]
        else {
            panic!("completed telemetry")
        };
        assert_eq!(
            (
                logical_request_id.as_str(),
                *attempt,
                phase.as_str(),
                *provider_cached_tokens
            ),
            ("turn-7-iteration-2", 2, "completed", Some(80))
        );

        let json = serde_json::to_string(&records[0].event).expect("serialize telemetry");
        assert!(!json.contains("SECRET-PROMPT-OR-TOOL-OR-EVIDENCE-CONTENT"));
        for forbidden_field in [
            "prompt", "request", "tool", "evidence", "headers", "endpoint", "error",
        ] {
            assert!(!json.contains(&format!("\"{forbidden_field}\"")));
        }
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
                derived_coverage: None,
                detail: Some("empty summary".into()),
            }),
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
            &AgentEvent::ContextCompacted(ContextCompactionEvent {
                outcome: "succeeded".into(),
                summary: "durable summary".into(),
                tail_start_index: 0,
                original_history_items: 0,
                retained_history_items: 0,
                retired_source_spans: Vec::new(),
                frame_identity_bindings: Vec::new(),
                derived_coverage: None,
                detail: None,
            }),
        )
        .expect("persist durable summary");

        let records = read_records(recorder.path()).expect("read records");
        assert_eq!(records.len(), 1);
        assert!(matches!(
            records[0].event,
            TranscriptEvent::ContextCompaction { .. }
        ));
        assert!(
            !serde_json::to_string(&records)
                .expect("serialize records")
                .contains("transient preview")
        );
    }
}
