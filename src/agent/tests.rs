use super::*;
use crate::agent_event_journal::persist_agent_event;
use crate::context_tree::{ContextNodeStatus, ContextTreeState};
use crate::context_view::{ContextBlockId, ContextViewProjection};
use crate::protocol_frames::ProtocolFrameItem;
use crate::request_builder::{TestRequestBuilderInput, build_test_request};
use crate::runtime_context::{
    FrameVisibility, PromptContributorKind, PromptContributorPlaceholder, RuntimeChildSession,
    RuntimeFrame, RuntimeFrameIdSeed, RuntimeFrameKind, RuntimeFrameProvenance, RuntimeSource,
    SourceSpan,
};
use crate::transcript::transcript_projection::{
    SessionContextCursor, project_context_tree, project_context_view,
    project_runtime_restore_snapshot,
};
use crate::transcript::{
    ActiveContextExperiment, ROOT_CONTEXT_BRANCH_ID, TranscriptEvent, TranscriptRecord,
    TranscriptRecorder, read_records, restore_runtime_snapshot, restore_session_history,
};
use async_openai::config::OpenAIConfig;
use async_trait::async_trait;
use serde_json::json;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Barrier, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::task::JoinHandle;
use tokio::time::{Duration, sleep};

fn agents_test_dir() -> PathBuf {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time before unix epoch")
        .as_nanos();
    let path = std::env::temp_dir().join(format!("letcode-agents-test-{timestamp}"));
    fs::create_dir_all(&path).expect("test directory should be created");
    path
}

fn observed_units(digests: &[&str]) -> crate::request_builder::LogicalRequestObservation {
    crate::request_builder::LogicalRequestObservation {
        cohort: crate::request_builder::LogicalRequestCohort {
            request_shape_digest: "same-request-shape".into(),
        },
        units: digests
            .iter()
            .map(|digest| crate::request_builder::LogicalRequestUnit {
                category: crate::request_builder::LogicalRequestUnitCategory::User,
                estimated_tokens: 1,
                byte_count: 4,
                digest: (*digest).into(),
            })
            .collect(),
    }
}

#[test]
fn adjacent_lcp_distinguishes_equal_append_middle_and_removed_suffix() {
    let mut tracker = LogicalRequestObservationTracker::default();
    tracker.commit(observed_units(&["a", "b"]));

    let exact = tracker.preview(observed_units(&["a", "b"]));
    assert_eq!(exact.lcp_units, 2);
    assert_eq!(exact.first_breaker, None);

    let append = tracker.preview(observed_units(&["a", "b", "c"]));
    assert_eq!(append.lcp_units, 2);
    assert_eq!(append.first_breaker, None);

    let middle = tracker.preview(observed_units(&["a", "changed", "c"]));
    assert_eq!(middle.lcp_units, 1);
    assert_eq!(
        middle.first_breaker,
        Some(crate::request_builder::LogicalRequestBreaker::CurrentUnit(
            crate::request_builder::LogicalRequestUnitCategory::User
        ))
    );

    let shrink = tracker.preview(observed_units(&["a"]));
    assert_eq!(shrink.lcp_units, 1);
    assert_eq!(
        shrink.first_breaker,
        Some(crate::request_builder::LogicalRequestBreaker::RemovedSuffix)
    );
}

#[test]
fn observation_preview_does_not_advance_the_baseline() {
    let mut tracker = LogicalRequestObservationTracker::default();
    let preview = tracker.preview(observed_units(&["a"]));
    assert!(!preview.cohort_comparable);
    // An unsent preview must leave the next preview without a baseline.
    assert!(!tracker.preview(observed_units(&["a"])).cohort_comparable);
    tracker.commit(observed_units(&["a"]));
    assert!(tracker.preview(observed_units(&["a"])).cohort_comparable);
}

#[test]
fn pressure_compaction_consumes_one_attempt_per_frontier_and_honors_suppression() {
    let first = PressureCompactionFrontier {
        frame_count: 3,
        protocol_prefix_digest: "first".into(),
    };
    let second = PressureCompactionFrontier {
        frame_count: 4,
        protocol_prefix_digest: "second".into(),
    };
    let mut state = PressureCompactionState::default();

    state
        .mark_attempted(first.clone())
        .expect("first frontier is available");
    assert!(
        state.mark_attempted(first).is_err(),
        "no-progress retries are spent"
    );
    state
        .mark_attempted(second)
        .expect("a changed protocol identity permits one new attempt");

    state.suppress();
    assert!(
        state
            .mark_attempted(PressureCompactionFrontier {
                frame_count: 5,
                protocol_prefix_digest: "summary-agent".into(),
            })
            .is_err()
    );
}

#[test]
fn summary_pressure_suppression_survives_turn_initialization() {
    let mut agent = test_agent();
    agent.pressure_compaction_suppressed = true;
    agent.prepare_turn_prelude("summary request");

    let error = agent
        .turn
        .pressure_compaction
        .mark_attempted(PressureCompactionFrontier {
            frame_count: 1,
            protocol_prefix_digest: "summary-pressure".into(),
        })
        .expect_err("a summary request must not enter pressure compaction");
    assert!(error.to_string().contains("suppressed"));

    let mut normal_agent = test_agent();
    normal_agent.prepare_turn_prelude("normal request");
    normal_agent
        .turn
        .pressure_compaction
        .mark_attempted(PressureCompactionFrontier {
            frame_count: 1,
            protocol_prefix_digest: "normal-pressure".into(),
        })
        .expect("normal turns retain a fresh pressure frontier");
}

fn assert_request_telemetry_is_terminal_once(events: &[LlmRequestTelemetry]) {
    let mut terminals = HashMap::new();
    for event in events {
        match event.phase {
            LlmRequestTelemetryPhase::Prepared => {
                assert!(
                    terminals
                        .insert((event.logical_request_id.as_str(), event.attempt), None,)
                        .is_none(),
                    "duplicate prepared event for physical request"
                );
            }
            LlmRequestTelemetryPhase::Completed
            | LlmRequestTelemetryPhase::Failed
            | LlmRequestTelemetryPhase::Interrupted => {
                let key = (event.logical_request_id.as_str(), event.attempt);
                let terminal = terminals
                    .get_mut(&key)
                    .expect("terminal event without prepared event");
                assert!(
                    terminal.replace(event.phase).is_none(),
                    "duplicate terminal event"
                );
            }
        }
    }
    assert!(
        terminals.values().all(Option::is_some),
        "prepared event without terminal event"
    );
}

fn test_skill_registry() -> Arc<SkillRegistry> {
    Arc::new(
        SkillRegistry::from_entries(vec![crate::skills::SkillEntry {
            name: "rust-audit".into(),
            description: "Inspect Rust code".into(),
            body: "# Private body".into(),
            content: "---\nname: rust-audit\ndescription: Inspect Rust code\n---\n# Private body\n"
                .into(),
            location: ".letcode/skills".into(),
            path: PathBuf::from("/workspace/.letcode/skills/rust-audit/SKILL.md"),
            base_dir: PathBuf::from("/workspace/.letcode/skills/rust-audit"),
        }])
        .expect("skill registry"),
    )
}

fn test_agent() -> Agent<OpenAIConfig> {
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base("https://api.openai.com/v1")
            .with_api_key("test"),
    );
    Agent::new(client, "m1", 4, 4)
}

fn provider_usage(used_tokens: u64) -> TokenUsageEstimate {
    TokenUsageEstimate {
        used_tokens,
        context_window_tokens: 10_000,
        input_tokens: used_tokens,
        output_tokens: 0,
        cached_tokens: 0,
    }
}

fn active_epoch_agent(history: Vec<HistoryItem>) -> Agent<OpenAIConfig> {
    let mut agent = test_agent();
    agent.history = history;
    agent.runtime_snapshot = runtime_snapshot_for_history("active-epoch", &agent.history);
    agent.runtime_snapshot.current_turn_id = Some(1);
    agent.protocol_frames = agent.runtime_snapshot.active_protocol_frames();
    agent.turn.turn_id = 1;
    agent
}

fn replace_active_epoch_history(agent: &mut Agent<OpenAIConfig>, history: Vec<HistoryItem>) {
    agent.history = history;
    agent.runtime_snapshot = runtime_snapshot_for_history("active-epoch", &agent.history);
    agent.runtime_snapshot.current_turn_id = Some(agent.turn.turn_id);
    agent.protocol_frames = agent.runtime_snapshot.active_protocol_frames();
}

fn active_epoch_tools() -> Vec<crate::request_builder::ToolSpec> {
    vec![crate::request_builder::ToolSpec {
        name: "lookup".into(),
        description: "Lookup a value".into(),
        parameters: json!({"type": "object", "properties": {"key": {"type": "string"}}}),
        strict: true,
    }]
}

fn active_epoch_history_with_complete_tool_group() -> Vec<HistoryItem> {
    vec![
        HistoryItem::user("seed"),
        HistoryItem::AssistantToolCalls {
            text: Some("calling tools".into()),
            reasoning_content: None,
            calls: vec![
                crate::protocol_frames::ProtocolToolCall {
                    call_id: "call-1".into(),
                    name: "lookup".into(),
                    arguments_json: r#"{"key":"one"}"#.into(),
                },
                crate::protocol_frames::ProtocolToolCall {
                    call_id: "call-2".into(),
                    name: "lookup".into(),
                    arguments_json: r#"{"key":"two"}"#.into(),
                },
            ],
        },
        HistoryItem::ToolOutput {
            call_id: "call-1".into(),
            output_json: r#"{"value":1}"#.into(),
        },
        HistoryItem::ToolOutput {
            call_id: "call-2".into(),
            output_json: r#"{"value":2}"#.into(),
        },
    ]
}

#[test]
fn provider_usage_projection_adds_only_trailing_frame_estimates() {
    let mut agent = test_agent();
    agent
        .append_history_item(HistoryItem::assistant("baseline"))
        .expect("baseline frame appends");
    agent.install_provider_usage_anchor_for_test(provider_usage(100));
    agent
        .append_history_item(HistoryItem::user("trailing context"))
        .expect("trailing frame appends");

    let expected_delta = serde_json::to_string(agent.history_for_test().last().unwrap())
        .expect("trailing item serializes")
        .len()
        .div_ceil(4) as u64;
    assert_eq!(
        agent.projected_token_usage(),
        Some(TokenUsageEstimate {
            used_tokens: 100 + expected_delta,
            input_tokens: 100 + expected_delta,
            ..provider_usage(100)
        })
    );
}

#[test]
fn provider_usage_projection_matches_provider_baseline_at_anchor() {
    let mut agent = test_agent();
    agent
        .append_history_item(HistoryItem::assistant("baseline"))
        .expect("baseline frame appends");
    let usage = TokenUsageEstimate {
        used_tokens: 120,
        context_window_tokens: 10_000,
        input_tokens: 100,
        output_tokens: 20,
        cached_tokens: 40,
    };
    agent.install_provider_usage_anchor_for_test(usage);

    assert_eq!(agent.projected_token_usage(), Some(usage));
}

#[test]
fn provider_usage_projection_fails_open_for_invalid_frontier() {
    let mut agent = test_agent();
    agent
        .append_history_item(HistoryItem::assistant("baseline"))
        .expect("baseline frame appends");
    agent.install_provider_usage_anchor_for_test(provider_usage(100));
    agent.protocol_frames[0].item = ProtocolFrameItem::assistant("mutated baseline");

    assert_eq!(agent.projected_token_usage(), None);
}

#[test]
fn provider_usage_projection_fails_open_after_replacement() {
    let mut agent = test_agent();
    agent
        .append_history_item(HistoryItem::assistant("baseline"))
        .expect("baseline frame appends");
    agent.install_provider_usage_anchor_for_test(provider_usage(100));
    agent
        .replace_history(vec![HistoryItem::assistant("replacement")])
        .expect("replacement succeeds");

    assert_eq!(agent.projected_token_usage(), None);
    assert_eq!(agent.provider_usage_anchor_for_test(), None);
}

#[test]
fn active_epoch_appends_complete_groups_for_both_protocols() {
    for protocol in [ApiProtocol::Responses, ApiProtocol::Completions] {
        let tools = active_epoch_tools();
        let mut agent = active_epoch_agent(vec![HistoryItem::user("seed")]);
        let cold = agent
            .preview_active_epoch(protocol, &[], &tools)
            .expect("cold preview");
        assert!(matches!(cold.transition, ActiveEpochTransition::Cold));
        let cold_units = cold.epoch.observation.units.clone();
        let cold_cacheable_prefix = cold.build.budget.plan_cacheable_prefix_tokens;
        agent.commit_active_epoch(cold);

        replace_active_epoch_history(&mut agent, active_epoch_history_with_complete_tool_group());
        let append = agent
            .preview_active_epoch(protocol, &[], &tools)
            .expect("complete tool group appends");
        assert!(matches!(
            append.transition,
            ActiveEpochTransition::Append { added: 3 }
        ));
        assert_eq!(
            &append.epoch.observation.units[..cold_units.len()],
            cold_units.as_slice(),
            "{protocol:?} preserves the exact provider-unit prefix"
        );
        let prefix_len = agent
            .active_epoch
            .as_ref()
            .expect("committed epoch")
            .committed_plan
            .segments
            .len();
        assert!(
            append.epoch.committed_plan.segments[..prefix_len]
                .iter()
                .all(|segment| segment.stability
                    == crate::request_builder::prompt_plan::PromptSegmentStability::Stable)
        );
        assert!(
            append.epoch.committed_plan.segments[prefix_len..]
                .iter()
                .all(|segment| segment.stability
                    == crate::request_builder::prompt_plan::PromptSegmentStability::Volatile)
        );
        assert!(
            append.build.budget.plan_cacheable_prefix_tokens > cold_cacheable_prefix,
            "{protocol:?} has a cacheable prefix after append"
        );
        agent.commit_active_epoch(append);

        let repeat = agent
            .preview_active_epoch(protocol, &[], &tools)
            .expect("exact repeat appends nothing");
        assert!(matches!(
            repeat.transition,
            ActiveEpochTransition::Append { added: 0 }
        ));
    }
}

#[test]
fn active_epoch_rejects_non_append_changes_without_advancing() {
    let tools = active_epoch_tools();
    for mutation in ["mutated", "truncated"] {
        let mut agent = active_epoch_agent(vec![HistoryItem::user("seed")]);
        let preview = agent
            .preview_active_epoch(ApiProtocol::Responses, &[], &tools)
            .expect("cold preview");
        agent.commit_active_epoch(preview);
        let before = agent.active_epoch.clone();
        if mutation == "mutated" {
            agent.protocol_frames[0].item = ProtocolFrameItem::UserMessage {
                content: crate::user_content::UserMessageContent::new("changed", Vec::new()),
            };
        } else {
            agent.protocol_frames.clear();
        }
        assert!(
            agent
                .preview_active_epoch(ApiProtocol::Responses, &[], &tools)
                .is_err()
        );
        assert_eq!(agent.active_epoch, before);
    }

    for item in [
        HistoryItem::AssistantToolCalls {
            text: None,
            reasoning_content: None,
            calls: vec![crate::protocol_frames::ProtocolToolCall {
                call_id: "call-1".into(),
                name: "lookup".into(),
                arguments_json: "{}".into(),
            }],
        },
        HistoryItem::ToolOutput {
            call_id: "orphan".into(),
            output_json: "{}".into(),
        },
    ] {
        let mut agent = active_epoch_agent(vec![HistoryItem::user("seed")]);
        let preview = agent
            .preview_active_epoch(ApiProtocol::Responses, &[], &tools)
            .expect("cold preview");
        agent.commit_active_epoch(preview);
        let before = agent.active_epoch.clone();
        let mut history = agent.history.clone();
        history.push(item);
        replace_active_epoch_history(&mut agent, history);
        assert!(
            agent
                .preview_active_epoch(ApiProtocol::Responses, &[], &tools)
                .is_err()
        );
        assert_eq!(agent.active_epoch, before);
    }
}

#[test]
fn active_epoch_cold_rebuilds_on_provider_identity_changes() {
    let tools = active_epoch_tools();
    let mut agent = active_epoch_agent(vec![HistoryItem::user("seed")]);
    let mut prelude = vec![PromptMessage::developer("kernel")];
    let initial = agent
        .preview_active_epoch(ApiProtocol::Responses, &prelude, &tools)
        .expect("initial preview");
    agent.commit_active_epoch(initial);

    let protocol = agent
        .preview_active_epoch(ApiProtocol::Completions, &prelude, &tools)
        .expect("protocol change rebuilds cold");
    assert!(matches!(protocol.transition, ActiveEpochTransition::Cold));
    agent.commit_active_epoch(protocol);

    let mut changed_tools = tools.clone();
    changed_tools[0].parameters = json!({
        "type": "object",
        "properties": {"key": {"type": "string"}, "limit": {"type": "integer"}}
    });
    let shape = agent
        .preview_active_epoch(ApiProtocol::Completions, &prelude, &changed_tools)
        .expect("tool schema change rebuilds cold");
    assert!(matches!(shape.transition, ActiveEpochTransition::Cold));
    agent.commit_active_epoch(shape);

    prelude.push(PromptMessage::developer("changed kernel"));
    let kernel = agent
        .preview_active_epoch(ApiProtocol::Completions, &prelude, &changed_tools)
        .expect("kernel change rebuilds cold");
    assert!(matches!(kernel.transition, ActiveEpochTransition::Cold));
    agent.commit_active_epoch(kernel);

    agent.turn.frozen_evidence = Some(FrozenTurnEvidence {
        message: Some("changed envelope evidence".into()),
        selected_ids: vec!["active-epoch".into()],
    });
    let envelope = agent
        .preview_active_epoch(ApiProtocol::Completions, &prelude, &changed_tools)
        .expect("evidence change rebuilds cold");
    assert!(matches!(envelope.transition, ActiveEpochTransition::Cold));
}

#[test]
fn active_epoch_resets_at_lifecycle_boundaries() {
    let tools = active_epoch_tools();
    let mut agent = active_epoch_agent(vec![HistoryItem::user("seed")]);
    let preview = agent
        .preview_active_epoch(ApiProtocol::Responses, &[], &tools)
        .expect("cold preview");
    agent.commit_active_epoch(preview);
    agent.prepare_turn_prelude("next turn");
    assert!(agent.active_epoch.is_none());
    assert!(matches!(
        agent
            .preview_active_epoch(ApiProtocol::Responses, &[], &tools)
            .expect("new turn preview")
            .transition,
        ActiveEpochTransition::Cold
    ));

    let frames = agent.protocol_frames.clone();
    let snapshot = agent.runtime_snapshot.clone();
    agent.commit_active_epoch(
        agent
            .preview_active_epoch(ApiProtocol::Responses, &[], &tools)
            .expect("preview before restore"),
    );
    agent
        .restore_runtime_snapshot(frames, snapshot)
        .expect("restore succeeds");
    assert!(agent.active_epoch.is_none());
    assert!(matches!(
        agent
            .preview_active_epoch(ApiProtocol::Responses, &[], &tools)
            .expect("restored preview")
            .transition,
        ActiveEpochTransition::Cold
    ));
}

fn test_evidence(id: &str, sequence: u64) -> EvidenceRecord {
    EvidenceRecord {
        id: id.into(),
        sequence,
        timestamp_ms: 0,
        evidence_kind: crate::evidence::EvidenceKind::Decision,
        title: format!("evidence {id}"),
        summary: format!("summary {id}"),
        detail: None,
        source: EvidenceSource::Transcript { sequence },
        tags: Vec::new(),
    }
}

fn runtime_frames_for_history(history: &[HistoryItem]) -> Vec<RuntimeFrame> {
    crate::protocol_frames::history_items_to_frames(history)
        .iter()
        .enumerate()
        .map(|(ordinal, frame)| runtime_frame_from_protocol_frame(frame, ordinal as u32))
        .collect()
}

fn runtime_snapshot_for_history(
    branch_id: impl Into<String>,
    history: &[HistoryItem],
) -> RuntimeSnapshot {
    let mut snapshot = RuntimeSnapshot::new(branch_id);
    snapshot.frames = runtime_frames_for_history(history);
    snapshot
}

#[test]
fn runtime_compaction_no_longer_emits_overlap_retained_state_error() {
    // The fail-fast "compaction retirement spans overlap retained runtime state"
    // gate was removed; shared spans are accepted. Lock that decision.
    let src = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/agent.rs"));
    assert!(
        !src.contains("compaction retirement spans overlap retained runtime state"),
        "overlap retained runtime state fail-fast must stay deleted"
    );
}

#[test]
fn runtime_snapshot_provider_refresh_preserves_valid_provider_usage_anchor() {
    let mut agent = test_agent();
    let history = vec![HistoryItem::user("current")];
    agent
        .replace_history(history.clone())
        .expect("valid history");
    agent.runtime_snapshot = runtime_snapshot_for_history("main", &history);
    agent.install_provider_usage_anchor_for_test(provider_usage(100));
    let projected = agent.runtime_snapshot.clone();
    agent.set_runtime_snapshot_provider(Arc::new(move || Ok(projected.clone())));

    agent
        .refresh_runtime_snapshot_from_provider()
        .expect("refresh succeeds");

    assert_eq!(agent.projected_token_usage(), Some(provider_usage(100)));
}

#[test]
fn runtime_snapshot_provider_refresh_clears_invalid_provider_usage_anchor() {
    let mut agent = test_agent();
    agent
        .append_history_item(HistoryItem::assistant("baseline"))
        .expect("baseline frame appends");
    agent.install_provider_usage_anchor_for_test(provider_usage(100));
    agent.history[0] = HistoryItem::assistant("mutated baseline");
    let projected = agent.runtime_snapshot.clone();
    agent.set_runtime_snapshot_provider(Arc::new(move || Ok(projected.clone())));

    agent
        .refresh_runtime_snapshot_from_provider()
        .expect("refresh succeeds");

    assert_eq!(agent.provider_usage_anchor_for_test(), None);
}

#[test]
fn runtime_snapshot_provider_refresh_accepts_empty_context_projection() {
    let mut agent = test_agent();
    let history = vec![HistoryItem::user("current")];
    agent
        .replace_history(history.clone())
        .expect("valid history");
    agent.runtime_snapshot = runtime_snapshot_for_history("main", &history);
    let records = vec![transcript_record(
        1,
        TranscriptEvent::AssistantMessage {
            content: "stale context".into(),
        },
    )];
    agent
        .runtime_snapshot
        .set_context_view(project_context_view(&records).expect("context view"));
    agent
        .runtime_snapshot
        .set_context_tree(project_context_tree(&records).expect("context tree"));
    let projected = runtime_snapshot_for_history("main", &history);
    agent.set_runtime_snapshot_provider(Arc::new(move || Ok(projected.clone())));

    agent
        .refresh_runtime_snapshot_from_provider()
        .expect("refresh succeeds");

    assert_eq!(
        agent.runtime_snapshot.context_view,
        ContextViewProjection::default()
    );
    assert_eq!(
        agent.runtime_snapshot.context_tree,
        ContextTreeState::with_default_root()
    );
}

#[test]
fn evidence_has_one_runtime_snapshot_authority_and_failed_candidates_are_atomic() {
    let mut agent = test_agent();
    agent
        .add_evidence(test_evidence("ev-live", 1))
        .expect("add live evidence");
    assert_eq!(agent.evidence(), agent.runtime_snapshot.evidence.as_slice());

    let before = agent.runtime_snapshot.evidence.clone();
    assert!(agent.add_evidence(test_evidence("ev-live", 2)).is_err());
    assert_eq!(agent.runtime_snapshot.evidence, before);

    assert!(
        agent
            .restore_evidence(vec![
                test_evidence("ev-duplicate", 2),
                test_evidence("ev-duplicate", 3)
            ])
            .is_err()
    );
    assert_eq!(agent.runtime_snapshot.evidence, before);

    assert!(
        agent
            .restore_session_history(
                Vec::new(),
                vec![
                    test_evidence("ev-duplicate", 2),
                    test_evidence("ev-duplicate", 3),
                ],
                0,
            )
            .is_err()
    );
    assert_eq!(agent.runtime_snapshot.evidence, before);

    let mut invalid_restore = RuntimeSnapshot::new(ROOT_CONTEXT_BRANCH_ID);
    invalid_restore.set_evidence(vec![
        test_evidence("ev-duplicate", 2),
        test_evidence("ev-duplicate", 3),
    ]);
    assert!(
        agent
            .restore_runtime_snapshot(Vec::new(), invalid_restore)
            .is_err()
    );
    assert_eq!(agent.runtime_snapshot.evidence, before);

    let mut replacement = RuntimeSnapshot::new(ROOT_CONTEXT_BRANCH_ID);
    replacement.set_evidence(vec![test_evidence("ev-provider", 4)]);
    agent.set_runtime_snapshot_provider(Arc::new(move || Ok(replacement.clone())));
    agent
        .replace_runtime_snapshot_from_provider()
        .expect("replace provider snapshot");
    assert_eq!(agent.evidence(), agent.runtime_snapshot.evidence.as_slice());
    assert_eq!(agent.evidence()[0].id, "ev-provider");

    let before = agent.runtime_snapshot.evidence.clone();
    let mut invalid = RuntimeSnapshot::new(ROOT_CONTEXT_BRANCH_ID);
    invalid.set_evidence(vec![
        test_evidence("ev-duplicate", 5),
        test_evidence("ev-duplicate", 6),
    ]);
    agent.set_runtime_snapshot_provider(Arc::new(move || Ok(invalid.clone())));
    assert!(agent.replace_runtime_snapshot_from_provider().is_err());
    assert_eq!(agent.runtime_snapshot.evidence, before);
}

fn transcript_record(sequence: u64, event: TranscriptEvent) -> TranscriptRecord {
    TranscriptRecord {
        session_id: "s".into(),
        sequence,
        timestamp_ms: 0,
        context_branch_id: None,
        event,
    }
}

#[tokio::test]
async fn non_context_tool_execution_does_not_require_snapshot_provider() {
    let mut agent = test_agent();
    let call = HistoryToolCall {
        call_id: "call-echo".into(),
        name: tool_names::TOOL_UTIL_ECHO.into(),
        arguments_json: json!({"text":"hello"}).to_string(),
    };

    let record = tool_execution::execute_tool_call(
        &mut agent,
        &call,
        &mut |_| async { Ok(()) },
        &mut |_| async { Ok(PermissionApproval::Deny) },
    )
    .await
    .expect("non-context tool executes without snapshots");

    assert!(record.output.ok, "{:?}", record.output);
}

#[test]
fn agent_iteration_limit_allows_tool_budget_plus_final_round() {
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base("https://api.openai.com/v1")
            .with_api_key("test"),
    );
    let agent = Agent::new(client, "m1", 64, 128);

    assert_eq!(agent.max_tool_calls_limit(), Some(128));
    assert_eq!(agent.max_iterations_limit(), Some(64));
}

#[test]
fn agent_iteration_limit_preserves_larger_configured_limit() {
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base("https://api.openai.com/v1")
            .with_api_key("test"),
    );
    let agent = Agent::new(client, "m1", 200, 128);

    assert_eq!(agent.max_iterations_limit(), Some(200));
}

#[test]
fn agent_limits_are_unbounded_by_default_when_omitted() {
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base("https://api.openai.com/v1")
            .with_api_key("test"),
    );
    let agent = Agent::new(client, "m1", None, None);

    assert_eq!(agent.max_iterations_limit(), None);
    assert_eq!(agent.max_tool_calls_limit(), None);
}

fn complete_http_request_len(request: &[u8]) -> Option<usize> {
    let header_end = request
        .windows(4)
        .position(|window| window == b"\r\n\r\n")?;
    let headers =
        std::str::from_utf8(&request[..header_end]).expect("test client sends UTF-8 HTTP headers");
    let content_length = headers
        .lines()
        .find_map(|header| {
            header
                .split_once(':')
                .filter(|(name, _)| name.eq_ignore_ascii_case("content-length"))
                .map(|(_, value)| {
                    value
                        .trim()
                        .parse::<usize>()
                        .expect("test client sends a numeric content length")
                })
        })
        .unwrap_or(0);
    Some(header_end + 4 + content_length)
}

async fn read_complete_http_request(socket: &mut tokio::net::TcpStream) -> Vec<u8> {
    let mut request = Vec::new();
    loop {
        if complete_http_request_len(&request).is_some_and(|length| request.len() >= length) {
            return request;
        }
        let read = socket
            .read_buf(&mut request)
            .await
            .expect("server reads request");
        assert_ne!(read, 0, "test client closed before completing its request");
    }
}

async fn spawn_chat_completion_server(
    responses: Vec<&'static str>,
) -> (String, Arc<AtomicUsize>, JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("test server should bind");
    let addr = listener.local_addr().expect("test server has local addr");
    let count = Arc::new(AtomicUsize::new(0));
    let server_count = count.clone();
    let handle = tokio::spawn(async move {
        for response in responses {
            let (mut socket, _) = listener.accept().await.expect("server accepts request");
            // A single read can stop anywhere in the headers or request body.  Serving
            // the next scripted response at that point lets a connection close race the
            // client's upload, which made response sequencing intermittent under a busy
            // serial suite.  Consume the complete request before advancing the script.
            let _ = read_complete_http_request(&mut socket).await;
            server_count.fetch_add(1, Ordering::SeqCst);
            socket
                .write_all(response.as_bytes())
                .await
                .expect("server writes response");
            socket.shutdown().await.expect("server closes response");
        }
    });
    (format!("http://{addr}"), count, handle)
}

fn sse_response(body: String) -> &'static str {
    Box::leak(
            format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            )
            .into_boxed_str(),
        )
}

fn responses_tool_batch_sse(calls: Vec<serde_json::Value>) -> &'static str {
    let response = json!({
        "type": "response.completed", "sequence_number": 1,
        "response": {
            "id": "r-tools", "object": "response", "created_at": 1,
            "status": "completed", "background": false, "error": null,
            "incomplete_details": null, "instructions": null, "max_output_tokens": null,
            "model": "m1", "output": calls, "parallel_tool_calls": true,
            "previous_response_id": null, "reasoning": {}, "store": true,
            "temperature": 1, "text": {"format": {"type": "text"}},
            "tool_choice": "auto", "tools": [], "top_p": 1,
            "truncation": "disabled",
            "usage": {"input_tokens": 1, "input_tokens_details": {"cached_tokens": 0}, "output_tokens": 1, "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 2},
            "user": null, "metadata": {}
        }
    });
    let response = serde_json::to_string(&response).expect("response serializes");
    sse_response(format!("data: {response}\n\ndata: [DONE]\n\n"))
}

fn chat_tool_batch_sse(name: &str, call_id: &str, arguments: String) -> &'static str {
    let reasoning_content = "inspect ";
    let reasoning = "then call";
    let thinking = " the tool";
    let response = json!({
        "choices": [
            {
                "index": 0,
                "delta": {"reasoning_content": reasoning_content},
                "finish_reason": null
            },
            {
                "index": 0,
                "delta": {"reasoning": reasoning},
                "finish_reason": null
            },
            {
                "index": 0,
                "delta": {
                    "thinking": thinking,
                    "tool_calls": [{
                        "index": 0,
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": arguments}
                    }]
                },
                "finish_reason": "tool_calls"
            }
        ]
    });
    let response = serde_json::to_string(&response).expect("response serializes");
    sse_response(format!("data: {response}\n\ndata: [DONE]\n\n"))
}

fn responses_terminal_sse(
    event_type: &str,
    status: &str,
    error: Option<serde_json::Value>,
    incomplete_details: Option<serde_json::Value>,
) -> &'static str {
    let response = json!({
        "type": event_type, "sequence_number": 1,
        "response": {
            "id": "r-terminal", "object": "response", "created_at": 1,
            "status": status, "background": false, "error": error,
            "incomplete_details": incomplete_details, "instructions": null,
            "max_output_tokens": null, "model": "m1", "output": [],
            "parallel_tool_calls": true, "previous_response_id": null, "reasoning": {},
            "store": true, "temperature": 1, "text": {"format": {"type": "text"}},
            "tool_choice": "auto", "tools": [], "top_p": 1, "truncation": "disabled",
            "usage": null, "user": null, "metadata": {}
        }
    });
    let response = serde_json::to_string(&response).expect("response serializes");
    sse_response(format!("data: {response}\n\ndata: [DONE]\n\n"))
}

fn response_error_sse(code: Option<&str>, message: &str) -> &'static str {
    let event = json!({
        "type": "error",
        "sequence_number": 1,
        "code": code,
        "message": message,
    });
    let event = serde_json::to_string(&event).expect("response error serializes");
    sse_response(format!("data: {event}\n\ndata: [DONE]\n\n"))
}

fn responses_final_sse(text: &str) -> &'static str {
    let response = json!({
        "type": "response.completed", "sequence_number": 1,
        "response": {
            "id": "r-final", "object": "response", "created_at": 1,
            "status": "completed", "background": false, "error": null,
            "incomplete_details": null, "instructions": null, "max_output_tokens": null,
            "model": "m1", "output": [{"type": "message", "id": "m1", "status": "completed", "role": "assistant", "content": [{"type": "output_text", "text": text, "annotations": []}]}],
            "parallel_tool_calls": true, "previous_response_id": null, "reasoning": {},
            "store": true, "temperature": 1, "text": {"format": {"type": "text"}},
            "tool_choice": "auto", "tools": [], "top_p": 1, "truncation": "disabled",
            "usage": {"input_tokens": 1, "input_tokens_details": {"cached_tokens": 0}, "output_tokens": 1, "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 2},
            "user": null, "metadata": {}
        }
    });
    let response = serde_json::to_string(&response).expect("response serializes");
    sse_response(format!("data: {response}\n\ndata: [DONE]\n\n"))
}

fn chat_final_sse(text: &str) -> &'static str {
    let text = serde_json::to_string(text).expect("chat content serializes");
    sse_response(format!(
        "data: {{\"choices\":[{{\"index\":0,\"delta\":{{\"content\":{text}}},\"finish_reason\":\"stop\"}}]}}\n\ndata: [DONE]\n\n"
    ))
}

fn chat_final_sse_with_usage(text: &str) -> &'static str {
    let text = serde_json::to_string(text).expect("chat content serializes");
    sse_response(format!(
        "data: {{\"choices\":[{{\"index\":0,\"delta\":{{\"content\":{text}}},\"finish_reason\":\"stop\"}}],\"usage\":{{\"prompt_tokens\":10,\"completion_tokens\":2,\"total_tokens\":12,\"prompt_tokens_details\":{{\"cached_tokens\":3}}}}}}\n\ndata: [DONE]\n\n"
    ))
}

#[tokio::test]
async fn responses_installs_provider_anchor_after_assistant_frame() {
    let (base_url, _, server) =
        spawn_chat_completion_server(vec![responses_final_sse("final reply")]).await;
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base(base_url)
            .with_api_key("test"),
    );
    let mut agent = Agent::new(client, "m1", 1, 0);

    let result = agent
        .run_stream_async(
            "hello",
            |_| std::future::ready(Ok(())),
            |_| std::future::ready(Ok(())),
            |_| std::future::ready(Ok(PermissionApproval::Deny)),
        )
        .await
        .expect("responses stream completes");

    assert_eq!(result, "final reply");
    let expected = TokenUsageEstimate {
        used_tokens: 2,
        context_window_tokens: 8_192,
        input_tokens: 1,
        output_tokens: 1,
        cached_tokens: 0,
    };
    assert_eq!(agent.provider_usage_anchor_for_test(), Some(expected));
    assert_eq!(agent.projected_token_usage(), Some(expected));
    server.await.expect("server task should finish");
}

#[tokio::test]
async fn chat_tool_calls_round_trip_reasoning_content_in_follow_up_request() {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("test server should bind");
    let base_url = format!(
        "http://{}",
        listener.local_addr().expect("server has address")
    );
    let server = tokio::spawn(async move {
        let (mut first_socket, _) = listener
            .accept()
            .await
            .expect("server accepts first request");
        let _ = read_complete_http_request(&mut first_socket).await;
        first_socket
            .write_all(
                chat_tool_batch_sse("workflow__todos", "call-1", r#"{"items":[]}"#.into())
                    .as_bytes(),
            )
            .await
            .expect("server writes first response");
        first_socket
            .shutdown()
            .await
            .expect("server closes first response");

        let (mut second_socket, _) = listener
            .accept()
            .await
            .expect("server accepts second request");
        let second_request = read_complete_http_request(&mut second_socket).await;
        let body_start = second_request
            .windows(4)
            .position(|window| window == b"\r\n\r\n")
            .expect("request has headers")
            + 4;
        let body: serde_json::Value =
            serde_json::from_slice(&second_request[body_start..]).expect("request body is JSON");
        let assistant = body["messages"]
            .as_array()
            .expect("request has messages")
            .iter()
            .find(|message| message["role"] == "assistant")
            .expect("follow-up request has assistant tool-call message");
        assert_eq!(assistant["reasoning_content"], "inspect then call the tool");
        second_socket
            .write_all(chat_final_sse("done").as_bytes())
            .await
            .expect("server writes second response");
        second_socket
            .shutdown()
            .await
            .expect("server closes second response");
    });
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base(base_url)
            .with_api_key("test"),
    );
    let mut agent = Agent::new(client, "m1", 2, 1);

    let result = agent
        .run_oai_comp_stream_async(
            "hello",
            |_| std::future::ready(Ok(())),
            |_| std::future::ready(Ok(())),
            |_| std::future::ready(Ok(PermissionApproval::Deny)),
        )
        .await
        .expect("chat stream completes");

    assert_eq!(result, "done");
    server.await.expect("server task should finish");
}

#[tokio::test]
async fn chat_tool_calls_preserve_native_reasoning_content() {
    let (base_url, _, server) = spawn_chat_completion_server(vec![
        chat_tool_batch_sse("workflow__todos", "call-1", r#"{"items":[]}"#.into()),
        chat_final_sse("done"),
    ])
    .await;
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base(base_url)
            .with_api_key("test"),
    );
    let mut agent = Agent::new(client, "m1", 2, 1);

    agent
        .run_oai_comp_stream_async(
            "hello",
            |_| std::future::ready(Ok(())),
            |_| std::future::ready(Ok(())),
            |_| std::future::ready(Ok(PermissionApproval::Deny)),
        )
        .await
        .expect("chat tool call stream completes");

    assert!(matches!(
        agent.history_for_test().get(1),
        Some(HistoryItem::AssistantToolCalls {
            reasoning_content: Some(reasoning_content),
            ..
        }) if reasoning_content == "inspect then call the tool"
    ));
    server.await.expect("server task should finish");
}

#[tokio::test]
async fn chat_installs_provider_anchor_after_assistant_frame() {
    let (base_url, _, server) =
        spawn_chat_completion_server(vec![chat_final_sse_with_usage("final reply")]).await;
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base(base_url)
            .with_api_key("test"),
    );
    let mut agent = Agent::new(client, "m1", 1, 0);

    let result = agent
        .run_oai_comp_stream_async(
            "hello",
            |_| std::future::ready(Ok(())),
            |_| std::future::ready(Ok(())),
            |_| std::future::ready(Ok(PermissionApproval::Deny)),
        )
        .await
        .expect("chat stream completes");

    let expected = TokenUsageEstimate {
        used_tokens: 12,
        context_window_tokens: 8_192,
        input_tokens: 10,
        output_tokens: 2,
        cached_tokens: 3,
    };
    assert_eq!(result, "final reply");
    assert_eq!(agent.provider_usage_anchor_for_test(), Some(expected));
    assert_eq!(agent.projected_token_usage(), Some(expected));
    server.await.expect("server task should finish");
}

fn phase2_pressure_agent(base_url: String, protocol: ApiProtocol) -> Agent<OpenAIConfig> {
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base(base_url)
            .with_api_key("test"),
    );
    let mut agent = Agent::new(client, "m1", 4, 4);
    agent.set_default_protocol(protocol);
    agent.set_model_catalog(HashMap::from([(
        "m1".into(),
        ModelRequestMetadata {
            context_window: Some(10_000),
            effective_input_limit_tokens: Some(8_000),
            max_output_tokens: Some(128),
            supports_tools: false,
            supports_reasoning: false,
            ..Default::default()
        },
    )]));
    let history = vec![
        HistoryItem::user("historical request ".repeat(1_750)),
        HistoryItem::assistant("historical reply ".repeat(1_750)),
    ];
    agent
        .replace_history(history.clone())
        .expect("complete history");
    agent.runtime_snapshot = runtime_snapshot_for_history("main", &history);
    agent
        .adopt_snapshot_as_history_seed()
        .expect("seeded protocol frames match the runtime snapshot");
    agent
}

async fn assert_phase2_pressure_compacts_normal_stream(protocol: ApiProtocol) {
    let (summary_response, final_response) = match protocol {
        ApiProtocol::Responses => (
            responses_final_sse(&valid_checkpoint("pressure summary")),
            responses_final_sse("final reply"),
        ),
        ApiProtocol::Completions => (
            chat_final_sse(&valid_checkpoint("pressure summary")),
            chat_final_sse("final reply"),
        ),
    };
    let (base_url, requests, server) =
        spawn_chat_completion_server(vec![summary_response, final_response]).await;
    let mut agent = phase2_pressure_agent(base_url, protocol);
    agent.install_provider_usage_anchor_for_test(provider_usage(8_000));
    let mut compacted = 0;
    let mut events = Vec::new();
    let result = match protocol {
        ApiProtocol::Responses => {
            agent
                .run_stream_async(
                    "current user",
                    |_| std::future::ready(Ok(())),
                    |event| {
                        events.push(event.clone());
                        compacted += usize::from(matches!(event, AgentEvent::ContextCompacted(_)));
                        std::future::ready(Ok(()))
                    },
                    |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
                )
                .await
        }
        ApiProtocol::Completions => {
            agent
                .run_oai_comp_stream_async(
                    "current user",
                    |_| std::future::ready(Ok(())),
                    |event| {
                        events.push(event.clone());
                        compacted += usize::from(matches!(event, AgentEvent::ContextCompacted(_)));
                        std::future::ready(Ok(()))
                    },
                    |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
                )
                .await
        }
    }
    .expect("pressure compaction successor should complete");

    assert_eq!(result, "final reply");
    assert_eq!(compacted, 1);
    let started_at = events
        .iter()
        .position(|event| {
            matches!(
                event,
                AgentEvent::ContextCompactionStarted {
                    trigger: CompactionTrigger::RequestPressure
                }
            )
        })
        .expect("pressure compaction starts");
    let compacted_at = events
        .iter()
        .position(|event| matches!(event, AgentEvent::ContextCompacted(_)))
        .expect("pressure compaction completes");
    assert!(started_at < compacted_at);
    if protocol == ApiProtocol::Completions {
        assert!(
            events[started_at..compacted_at]
                .iter()
                .any(|event| matches!(event, AgentEvent::ContextCompactionDelta { .. }))
        );
    }
    assert_eq!(
        requests.load(Ordering::SeqCst),
        2,
        "one summary and one final request"
    );
    server.await.expect("server task should finish");
}

#[tokio::test]
async fn soft_pressure_skips_when_provider_usage_is_unknown() {
    let mut agent = phase2_pressure_agent("http://127.0.0.1:1".into(), ApiProtocol::Responses);
    let protected_start = agent.history.len();
    let prelude = agent.prepare_turn_prelude("current user");
    agent.turn.current_turn_start_index = Some(protected_start);
    agent
        .append_history_item(HistoryItem::user("current user"))
        .expect("stream path appends the current message");
    let tools = agent.tool_definitions();
    let mut protected = protected_start;
    let mut events = Vec::new();

    protocol_stream::prepare_canonical_protocol_stream_request_for_test(
        &mut agent,
        ApiProtocol::Responses,
        &prelude,
        &mut protected,
        &tools,
        &mut |event| {
            events.push(event);
            std::future::ready(Ok(()))
        },
    )
    .await
    .expect("unknown provider usage leaves soft pressure inactive");

    assert!(!events.iter().any(|event| matches!(
        event,
        AgentEvent::ContextCompactionStarted {
            trigger: CompactionTrigger::RequestPressure
        }
    )));
}

#[tokio::test]
async fn pressure_compaction_accepts_soft_unsafe_successor() {
    // Soft-unsafe successors (above high watermark but under hard limit) must
    // still commit. Hard-failing them was turning near-limit sessions into error loops.
    let oversized_summary = valid_checkpoint(&"pressure summary ".repeat(1_000));
    let (base_url, _requests, _server) =
        spawn_chat_completion_server(vec![responses_final_sse(&oversized_summary)]).await;
    let mut agent = phase2_pressure_agent(base_url, ApiProtocol::Responses);
    agent.set_model_catalog(HashMap::from([(
        "m1".into(),
        ModelRequestMetadata {
            context_window: Some(10_000),
            effective_input_limit_tokens: Some(6_000),
            max_output_tokens: Some(128),
            supports_tools: false,
            supports_reasoning: false,
            ..Default::default()
        },
    )]));
    let protected_start = agent.history.len();
    let prelude = agent.prepare_turn_prelude("current user");
    agent.turn.current_turn_start_index = Some(protected_start);
    agent
        .append_history_item(HistoryItem::user("current user"))
        .expect("stream path appends the current message");
    agent.install_provider_usage_anchor_for_test(provider_usage(6_000));
    let tools = agent.tool_definitions();
    let mut protected = protected_start;
    let mut events = Vec::new();
    let mut on_event = |event: AgentEvent| {
        events.push(event);
        std::future::ready(Ok(()))
    };

    let result = protocol_stream::prepare_canonical_protocol_stream_request_for_test(
        &mut agent,
        ApiProtocol::Responses,
        &prelude,
        &mut protected,
        &tools,
        &mut on_event,
    )
    .await;

    // Soft-unsafe is no longer a hard error.
    result.expect("soft-unsafe pressure successors must prepare a request");
    assert!(
        events
            .iter()
            .any(|event| matches!(event, AgentEvent::ContextCompacted(_)))
    );
}

#[tokio::test]
async fn phase2_pressure_compacts_normal_responses_stream() {
    assert_phase2_pressure_compacts_normal_stream(ApiProtocol::Responses).await;
}

#[tokio::test]
async fn phase2_pressure_compacts_normal_completions_stream() {
    assert_phase2_pressure_compacts_normal_stream(ApiProtocol::Completions).await;
}

#[tokio::test]
async fn phase2_pressure_compaction_is_not_repeated_for_physical_retry() {
    let (base_url, requests, server) = spawn_chat_completion_server(vec![
        responses_final_sse(&valid_checkpoint("pressure summary")),
        "HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
        responses_final_sse("final reply"),
    ])
    .await;
    let mut agent = phase2_pressure_agent(base_url, ApiProtocol::Responses);
    agent.install_provider_usage_anchor_for_test(provider_usage(8_000));
    agent.set_retry_config(test_retry_config());
    let mut compacted = 0;
    let result = agent
        .run_stream_async(
            "current user",
            |_| std::future::ready(Ok(())),
            |event| {
                compacted += usize::from(matches!(event, AgentEvent::ContextCompacted(_)));
                std::future::ready(Ok(()))
            },
            |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await
        .expect("retry after a pre-stream HTTP failure should complete");

    assert_eq!(result, "final reply");
    assert_eq!(compacted, 1);
    assert_eq!(
        requests.load(Ordering::SeqCst),
        3,
        "summary, failed attempt, retry"
    );
    server.await.expect("server task should finish");
}

#[tokio::test]
async fn phase2_pressure_callback_failure_is_atomic_and_consumes_its_frontier() {
    let (base_url, requests, server) = spawn_chat_completion_server(vec![
        responses_final_sse(&valid_checkpoint("pressure summary")),
        responses_final_sse(&valid_checkpoint("changed-frontier summary")),
    ])
    .await;
    let mut agent = phase2_pressure_agent(base_url, ApiProtocol::Responses);
    agent.install_provider_usage_anchor_for_test(provider_usage(8_000));
    let protected_start = agent.history.len();
    let prelude = agent.prepare_turn_prelude("current user");
    agent.turn.current_turn_start_index = Some(protected_start);
    agent
        .append_history_item(HistoryItem::user("current user"))
        .expect("stream path appends the current message");
    let history = agent.history.clone();
    let frames = agent.protocol_frames.clone();
    let snapshot = agent.runtime_snapshot.clone();
    let active_epoch = agent.active_epoch.clone();
    let start = agent.turn.current_turn_start_index;
    let tools = agent.tool_definitions();
    let mut protected = protected_start;
    let mut events = Vec::new();
    let mut failed_callback = |event: AgentEvent| {
        events.push(event.clone());
        if matches!(event, AgentEvent::ContextCompacted(_)) {
            return std::future::ready(Err(anyhow!("durable compaction callback failed")));
        }
        std::future::ready(Ok(()))
    };

    let result = protocol_stream::prepare_canonical_protocol_stream_request_for_test(
        &mut agent,
        ApiProtocol::Responses,
        &prelude,
        &mut protected,
        &tools,
        &mut failed_callback,
    )
    .await;
    let Err(error) = result else {
        panic!("the durable callback rejects the prepared summary");
    };
    assert!(
        error
            .to_string()
            .contains("durable compaction callback failed")
    );
    assert_eq!(agent.history, history);
    assert_eq!(agent.protocol_frames, frames);
    assert_eq!(agent.runtime_snapshot, snapshot);
    assert_eq!(agent.active_epoch, active_epoch);
    assert_eq!(agent.turn.current_turn_start_index, start);
    assert!(matches!(
        events.first(),
        Some(AgentEvent::ContextCompactionStarted {
            trigger: CompactionTrigger::RequestPressure
        })
    ));
    assert!(matches!(
        events.last(),
        Some(AgentEvent::ContextCompactionFailed {
            trigger: CompactionTrigger::RequestPressure
        })
    ));

    let mut same_frontier_callback = |_| std::future::ready(Ok(()));
    let same_frontier = protocol_stream::prepare_canonical_protocol_stream_request_for_test(
        &mut agent,
        ApiProtocol::Responses,
        &prelude,
        &mut protected,
        &tools,
        &mut same_frontier_callback,
    )
    .await;
    let Err(same_frontier) = same_frontier else {
        panic!("the same pressure frontier is single-use after callback failure");
    };
    assert!(same_frontier.to_string().contains("already attempted"));

    let changed = HistoryItem::user("current user with changed frame identity");
    agent.history[protected_start] = changed.clone();
    let changed_item = protocol_frame_item_from_history_item(&changed);
    agent.protocol_frames[protected_start].item = changed_item.clone();
    agent.runtime_snapshot.frames[protected_start].protocol = Some(changed_item);
    let mut changed_frontier_callback = |_| std::future::ready(Ok(()));
    protocol_stream::prepare_canonical_protocol_stream_request_for_test(
        &mut agent,
        ApiProtocol::Responses,
        &prelude,
        &mut protected,
        &tools,
        &mut changed_frontier_callback,
    )
    .await
    .expect("a changed frame identity may make a fresh pressure attempt");
    assert_eq!(requests.load(Ordering::SeqCst), 2);
    server.await.expect("summary server completes");
}

#[tokio::test]
async fn phase2_pressure_rejects_incomplete_tool_group_before_summary_callback() {
    let mut agent = phase2_pressure_agent("http://127.0.0.1:1".into(), ApiProtocol::Responses);
    let protected_start = agent.history.len();
    let prelude = agent.prepare_turn_prelude("current user");
    agent.turn.current_turn_start_index = Some(protected_start);
    agent
        .append_history_item(HistoryItem::user("current user"))
        .expect("current message appends");
    agent
        .append_history_item(HistoryItem::AssistantToolCalls {
            text: None,
            reasoning_content: None,
            calls: vec![HistoryToolCall {
                call_id: "pending".into(),
                name: "fs__read".into(),
                arguments_json: "{}".into(),
            }],
        })
        .expect("incomplete current tool group is representable");
    let history = agent.history.clone();
    let mut events = Vec::new();
    let mut protected = protected_start;
    let tools = agent.tool_definitions();
    let result = protocol_stream::prepare_canonical_protocol_stream_request_for_test(
        &mut agent,
        ApiProtocol::Responses,
        &prelude,
        &mut protected,
        &tools,
        &mut |event| {
            events.push(event);
            std::future::ready(Ok(()))
        },
    )
    .await;
    let Err(error) = result else {
        panic!("incomplete groups cannot be summarized under pressure");
    };
    assert!(
        error.to_string().contains("dangling assistant tool calls"),
        "unexpected pressure rejection: {error:#}"
    );
    assert_eq!(agent.history, history, "group remains intact");
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, AgentEvent::ContextCompacted(_)))
            .count(),
        0
    );
}

#[tokio::test]
async fn phase2_recognized_protected_request_overflow_attempts_compaction() {
    let (base_url, requests, server) = spawn_chat_completion_server(vec![responses_final_sse(
        &valid_checkpoint("overflow summary"),
    )])
    .await;
    let mut agent = phase2_pressure_agent(base_url, ApiProtocol::Responses);
    let protected_start = agent.history.len();
    let prelude = agent.prepare_turn_prelude("oversized current user");
    agent.turn.current_turn_start_index = Some(protected_start);
    agent
        .append_history_item(HistoryItem::user("protected ".repeat(20_000)))
        .expect("oversized current message appends");
    let mut compacted = 0;
    let mut protected = protected_start;
    let tools = agent.tool_definitions();
    let result = protocol_stream::prepare_canonical_protocol_stream_request_for_test(
        &mut agent,
        ApiProtocol::Responses,
        &prelude,
        &mut protected,
        &tools,
        &mut |event| {
            compacted += usize::from(matches!(event, AgentEvent::ContextCompacted(_)));
            std::future::ready(Ok(()))
        },
    )
    .await;
    let Err(error) = result else {
        panic!("the successor retains the deliberately overflowing current message");
    };
    assert!(
        error
            .to_string()
            .contains("protected current context exceeds input budget")
    );
    assert_eq!(
        compacted, 1,
        "durably committed compaction remains installed even when the protected current message still exceeds budget"
    );
    assert!(matches!(
        agent.history.first(),
        Some(HistoryItem::ContextSummary { .. })
    ));
    assert_eq!(agent.history.len(), 2);
    assert_eq!(
        requests.load(Ordering::SeqCst),
        1,
        "the rolled-back pressure summary is the only request"
    );
    server.await.expect("summary server completes");
}

#[test]
fn helper_agents_allow_one_iteration_plus_semantic_recovery_attempts() {
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base("https://api.openai.com/v1")
            .with_api_key("test"),
    );
    let mut agent = Agent::new(client, "m1", 4, 4);
    let mut retry = test_retry_config();
    retry.max_recovery_attempts = 2;
    agent.set_retry_config(retry);

    assert_eq!(agent.helper_max_iterations(), Some(3));
    assert_eq!(agent.session_title_agent().max_iterations_limit(), Some(3));
}

fn test_retry_config() -> RetryConfig {
    RetryConfig {
        enabled: true,
        max_attempts: 3,
        max_elapsed_ms: 100,
        max_recovery_attempts: 3,
        initial_delay_ms: 1,
        max_delay_ms: 5,
        backoff_multiplier: 2.0,
        jitter_ms: 0,
    }
}

fn test_tool_call(name: &str, arguments_json: &str) -> HistoryToolCall {
    HistoryToolCall {
        call_id: format!("call-{name}"),
        name: name.into(),
        arguments_json: arguments_json.into(),
    }
}

fn test_execution_record(tool_name: &str, output: ToolResult) -> ToolExecutionRecord {
    ToolExecutionRecord {
        call_id: format!("call-{tool_name}"),
        tool_name: tool_name.into(),
        arguments: Some(json!({})),
        permission_class: crate::permission::ToolPermissionClass::Read,
        directive: ExecutionDirective::None,
        status: ToolExecutionStatus::Executed,
        rejection: None,
        output,
        effects: ToolEffects {
            kind: ToolEffectKind::Read,
            primary_path: None,
            edited_paths: vec![],
            command: None,
        },
    }
}

struct ParallelReadTool {
    name: &'static str,
    barrier: Arc<Barrier>,
}

#[async_trait]
impl ToolHandler for ParallelReadTool {
    fn name(&self) -> &str {
        self.name
    }

    fn description(&self) -> &str {
        "parallel read test tool"
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": false
        })
    }

    fn permission_class(&self) -> crate::permission::ToolPermissionClass {
        crate::permission::ToolPermissionClass::Read
    }

    fn parallelism(&self) -> ToolParallelism {
        ToolParallelism::Parallel
    }

    async fn execute(&self, _args: Value) -> Result<Value> {
        let barrier = Arc::clone(&self.barrier);
        tokio::task::spawn_blocking(move || barrier.wait()).await?;
        Ok(json!({"name": self.name}))
    }
}

struct ParallelCountingReadTool {
    name: &'static str,
    active: Arc<AtomicUsize>,
    max_active: Arc<AtomicUsize>,
}

#[async_trait]
impl ToolHandler for ParallelCountingReadTool {
    fn name(&self) -> &str {
        self.name
    }

    fn description(&self) -> &str {
        "parallel counting read test tool"
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": false
        })
    }

    fn permission_class(&self) -> crate::permission::ToolPermissionClass {
        crate::permission::ToolPermissionClass::Read
    }

    fn parallelism(&self) -> ToolParallelism {
        ToolParallelism::Parallel
    }

    async fn execute(&self, _args: Value) -> Result<Value> {
        let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
        self.max_active.fetch_max(active, Ordering::SeqCst);
        sleep(Duration::from_millis(20)).await;
        self.active.fetch_sub(1, Ordering::SeqCst);
        Ok(json!({"name": self.name}))
    }
}

struct ExclusiveReadTool {
    name: &'static str,
    active: Arc<AtomicUsize>,
    max_active: Arc<AtomicUsize>,
}

#[async_trait]
impl ToolHandler for ExclusiveReadTool {
    fn name(&self) -> &str {
        self.name
    }

    fn description(&self) -> &str {
        "exclusive read test tool"
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": false
        })
    }

    fn permission_class(&self) -> crate::permission::ToolPermissionClass {
        crate::permission::ToolPermissionClass::Read
    }

    fn parallelism(&self) -> ToolParallelism {
        ToolParallelism::Exclusive
    }

    async fn execute(&self, _args: Value) -> Result<Value> {
        let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
        self.max_active.fetch_max(active, Ordering::SeqCst);
        sleep(Duration::from_millis(20)).await;
        self.active.fetch_sub(1, Ordering::SeqCst);
        Ok(json!({"name": self.name}))
    }
}

#[tokio::test]
async fn contiguous_parallel_read_tools_overlap_and_record_in_model_order() {
    let mut agent = test_agent();
    agent.set_model_catalog(HashMap::from([(
        "m1".to_string(),
        ModelRequestMetadata {
            parallel_tool_calls: true,
            supports_tools: true,
            ..Default::default()
        },
    )]));
    let barrier = Arc::new(Barrier::new(2));
    agent.register_tool(ParallelReadTool {
        name: "test__parallel_one",
        barrier: Arc::clone(&barrier),
    });
    agent.register_tool(ParallelReadTool {
        name: "test__parallel_two",
        barrier,
    });
    let calls = vec![
        test_tool_call("test__parallel_one", "{}"),
        test_tool_call("test__parallel_two", "{}"),
    ];
    agent
        .append_assistant_tool_calls("", &calls)
        .expect("append tool calls");

    tokio::time::timeout(
        Duration::from_secs(1),
        agent.execute_tool_calls_and_record(&calls, &mut |_| async { Ok(()) }, &mut |_| async {
            Ok(PermissionApproval::AllowOnce)
        }),
    )
    .await
    .expect("parallel reads should overlap")
    .expect("batch executes");

    let outputs = agent
        .history
        .iter()
        .filter_map(|item| match item {
            HistoryItem::ToolOutput { call_id, .. } => Some(call_id.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        outputs,
        vec!["call-test__parallel_one", "call-test__parallel_two"]
    );
}

#[tokio::test]
async fn parallel_start_failure_cancels_announced_calls_without_execution() {
    let mut agent = test_agent();
    agent.set_model_catalog(HashMap::from([(
        "m1".to_string(),
        ModelRequestMetadata {
            parallel_tool_calls: true,
            supports_tools: true,
            ..Default::default()
        },
    )]));
    let active = Arc::new(AtomicUsize::new(0));
    let max_active = Arc::new(AtomicUsize::new(0));
    for name in ["test__parallel_one", "test__parallel_two"] {
        agent.register_tool(ParallelCountingReadTool {
            name,
            active: Arc::clone(&active),
            max_active: Arc::clone(&max_active),
        });
    }
    let calls = vec![
        test_tool_call("test__parallel_one", "{}"),
        test_tool_call("test__parallel_two", "{}"),
    ];
    agent
        .append_assistant_tool_calls("", &calls)
        .expect("append tool calls");
    let events = Arc::new(Mutex::new(Vec::new()));
    let start_count = Arc::new(AtomicUsize::new(0));

    let result = agent
        .execute_tool_calls_and_record(
            &calls,
            &mut |event| {
                events.lock().expect("events lock").push(event.clone());
                let should_fail = matches!(event, AgentEvent::ToolCallStarted { .. })
                    && start_count.fetch_add(1, Ordering::SeqCst) == 1;
                std::future::ready(if should_fail {
                    Err(anyhow!("start callback failed"))
                } else {
                    Ok(())
                })
            },
            &mut |_| async { Ok(PermissionApproval::AllowOnce) },
        )
        .await;

    assert!(result.is_err());
    assert_eq!(max_active.load(Ordering::SeqCst), 0);
    let cancelled = events
        .lock()
        .expect("events lock")
        .iter()
        .filter_map(|event| match event {
            AgentEvent::ToolCallCancelled { call_id, .. } => Some(call_id.clone()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        cancelled,
        vec![
            "call-test__parallel_one".to_string(),
            "call-test__parallel_two".to_string()
        ]
    );
}

#[tokio::test]
async fn parallel_permission_change_during_starts_cancels_batch_without_execution() {
    let mut agent = test_agent();
    agent.set_model_catalog(HashMap::from([(
        "m1".to_string(),
        ModelRequestMetadata {
            parallel_tool_calls: true,
            supports_tools: true,
            ..Default::default()
        },
    )]));
    let active = Arc::new(AtomicUsize::new(0));
    let max_active = Arc::new(AtomicUsize::new(0));
    for name in ["test__parallel_one", "test__parallel_two"] {
        agent.register_tool(ParallelCountingReadTool {
            name,
            active: Arc::clone(&active),
            max_active: Arc::clone(&max_active),
        });
    }
    let calls = vec![
        test_tool_call("test__parallel_one", "{}"),
        test_tool_call("test__parallel_two", "{}"),
    ];
    agent
        .append_assistant_tool_calls("", &calls)
        .expect("append tool calls");
    let permission_session = Arc::clone(&agent.permission_session);
    let events = Arc::new(Mutex::new(Vec::new()));
    let changed = Arc::new(AtomicUsize::new(0));

    let result = agent
        .execute_tool_calls_and_record(
            &calls,
            &mut |event| {
                events.lock().expect("events lock").push(event.clone());
                if matches!(event, AgentEvent::ToolCallStarted { .. })
                    && changed.fetch_add(1, Ordering::SeqCst) == 0
                {
                    permission_session
                        .lock()
                        .expect("permission lock")
                        .clear_grants();
                }
                std::future::ready(Ok(()))
            },
            &mut |_| async { Ok(PermissionApproval::AllowOnce) },
        )
        .await;

    assert!(
        result
            .expect_err("permission generation change must reject the batch")
            .to_string()
            .contains("parallel permission preflight changed")
    );
    assert_eq!(max_active.load(Ordering::SeqCst), 0);
    assert_eq!(
        events
            .lock()
            .expect("events lock")
            .iter()
            .filter(|event| matches!(event, AgentEvent::ToolCallCancelled { .. }))
            .count(),
        2
    );
}

#[tokio::test]
async fn disabled_model_parallel_tool_calls_keep_parallel_tools_sequential() {
    let mut agent = test_agent();
    agent.set_model_catalog(HashMap::from([(
        "m1".to_string(),
        ModelRequestMetadata {
            parallel_tool_calls: false,
            supports_tools: true,
            ..Default::default()
        },
    )]));
    let active = Arc::new(AtomicUsize::new(0));
    let max_active = Arc::new(AtomicUsize::new(0));
    for name in ["test__parallel_one", "test__parallel_two"] {
        agent.register_tool(ParallelCountingReadTool {
            name,
            active: Arc::clone(&active),
            max_active: Arc::clone(&max_active),
        });
    }
    let calls = vec![
        test_tool_call("test__parallel_one", "{}"),
        test_tool_call("test__parallel_two", "{}"),
    ];
    agent
        .append_assistant_tool_calls("", &calls)
        .expect("append tool calls");

    agent
        .execute_tool_calls_and_record(&calls, &mut |_| async { Ok(()) }, &mut |_| async {
            Ok(PermissionApproval::AllowOnce)
        })
        .await
        .expect("parallel-capable reads execute sequentially");

    assert_eq!(max_active.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn exclusive_read_tools_remain_ordering_barriers() {
    let mut agent = test_agent();
    let active = Arc::new(AtomicUsize::new(0));
    let max_active = Arc::new(AtomicUsize::new(0));
    for name in ["test__exclusive_one", "test__exclusive_two"] {
        agent.register_tool(ExclusiveReadTool {
            name,
            active: Arc::clone(&active),
            max_active: Arc::clone(&max_active),
        });
    }
    let calls = vec![
        test_tool_call("test__exclusive_one", "{}"),
        test_tool_call("test__exclusive_two", "{}"),
    ];
    agent
        .append_assistant_tool_calls("", &calls)
        .expect("append tool calls");

    agent
        .execute_tool_calls_and_record(&calls, &mut |_| async { Ok(()) }, &mut |_| async {
            Ok(PermissionApproval::AllowOnce)
        })
        .await
        .expect("exclusive reads execute");

    assert_eq!(max_active.load(Ordering::SeqCst), 1);
}

struct SleepTool;

#[async_trait]
impl ToolHandler for SleepTool {
    fn name(&self) -> &str {
        "test__sleep"
    }

    fn description(&self) -> &str {
        "sleep test tool"
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": false
        })
    }

    async fn execute(&self, _args: Value) -> Result<Value> {
        sleep(Duration::from_millis(1_100)).await;
        Ok(json!({"done": true}))
    }
}

struct ReplayGuardTool(Arc<AtomicUsize>);

#[async_trait]
impl ToolHandler for ReplayGuardTool {
    fn name(&self) -> &str {
        "test__replay_guard"
    }

    fn description(&self) -> &str {
        "counts executions"
    }

    fn parameters(&self) -> Value {
        json!({"type": "object", "properties": {}, "additionalProperties": false})
    }

    async fn execute(&self, _args: Value) -> Result<Value> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(json!({"executed": true}))
    }
}

#[test]
fn completed_tool_output_projection_and_restore_never_reexecutes_handler() {
    let records = vec![
        TranscriptRecord {
            session_id: "s".into(),
            sequence: 1,
            timestamp_ms: 0,
            context_branch_id: None,
            event: TranscriptEvent::UserMessage {
                content: crate::user_content::UserMessageContent::from("resume"),
            },
        },
        TranscriptRecord {
            session_id: "s".into(),
            sequence: 2,
            timestamp_ms: 0,
            context_branch_id: None,
            event: TranscriptEvent::AssistantToolCallBatch {
                text: None,
                reasoning_content: None,
                calls: vec![HistoryToolCall {
                    call_id: "finished".into(),
                    name: "test__replay_guard".into(),
                    arguments_json: "{}".into(),
                }],
            },
        },
        TranscriptRecord {
            session_id: "s".into(),
            sequence: 3,
            timestamp_ms: 0,
            context_branch_id: None,
            event: TranscriptEvent::ToolCallFinished {
                call_id: "finished".into(),
                name: "test__replay_guard".into(),
                ok: true,
                output: ToolResult::ok("test__replay_guard", json!({"persisted": true})),
            },
        },
    ];
    let projected = crate::transcript::transcript_projection::project_runtime_restore_snapshot(
        "s".into(),
        records,
        crate::transcript::transcript_projection::SessionContextCursor {
            branch_id: None,
            leaf_sequence: None,
        },
        &[],
    )
    .expect("project completed output");
    let executions = Arc::new(AtomicUsize::new(0));
    let mut agent = test_agent();
    agent.register_tool(ReplayGuardTool(executions.clone()));
    agent
        .restore_runtime_snapshot(projected.protocol_frames, projected.snapshot)
        .expect("restore persisted output");
    assert_eq!(executions.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn agent_lifecycle_finalizes_without_live_experiment_semantics() {
    let mut agent = test_agent();
    agent.set_context_scope_state(Arc::new(std::sync::Mutex::new(ContextScopeState {
        active_experiment: Some(ActiveContextExperiment {
            branch_id: "branch-1".into(),
            parent_branch_id: ROOT_CONTEXT_BRANCH_ID.into(),
            base_sequence: 4,
            writes_observed: false,
        }),
    })));

    let mut events = Vec::new();
    let continued = agent
        .continue_or_finalize_no_tool_reply(
            &mut |event| {
                events.push(event);
                std::future::ready(Ok(()))
            },
            0,
            &mut 0,
        )
        .await
        .expect("turn should finalize normally");
    assert!(!continued);
    assert!(
        events
            .iter()
            .any(|event| matches!(event, AgentEvent::TurnFinalized(_)))
    );
}

#[tokio::test]
async fn non_shell_tool_timeout_emits_cancelled_and_timed_out_terminal_events() {
    let mut agent = test_agent();
    agent.set_tool_timeout_secs(Some(1));
    agent
        .try_register_tool(SleepTool)
        .expect("register sleep tool");

    let call = test_tool_call("test__sleep", "{}");
    let mut events = Vec::new();
    let record = agent
        .execute_tool_call(
            &call,
            &mut |event| {
                events.push(event);
                std::future::ready(Ok(()))
            },
            &mut |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await
        .expect("tool call should return timeout record");

    assert_eq!(record.status, ToolExecutionStatus::TimedOut);
    assert!(!record.output.ok);
    assert_eq!(
        record
            .output
            .data
            .as_ref()
            .and_then(|data| data.get("status"))
            .and_then(Value::as_str),
        Some("timed_out")
    );
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::ToolCallCancelled { call_id, name }
            if call_id == "call-test__sleep" && name == "test__sleep"
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::ToolCallFinished { ok, output, .. }
            if !ok
                && output
                    .data
                    .as_ref()
                    .and_then(|data| data.get("status"))
                    .and_then(Value::as_str)
                    == Some("timed_out")
    )));
}

#[test]
fn evidence_ids_remain_unique_after_restoring_older_evidence_snapshot() {
    let mut agent = test_agent();
    let first = agent
        .remember_tool_evidence(&test_execution_record(
            "fs__read",
            ToolResult::ok("fs__read", json!({"content": "one"})),
        ))
        .expect("first evidence");
    let older_snapshot = agent.runtime_snapshot.evidence.clone();

    let second = agent
        .remember_tool_evidence(&test_execution_record(
            "fs__read",
            ToolResult::ok("fs__read", json!({"content": "two"})),
        ))
        .expect("second evidence");
    assert_ne!(first.id, second.id);

    agent
        .restore_session_history(agent.history.clone(), older_snapshot, agent.next_turn_id)
        .expect("restore older evidence snapshot");

    let third = agent
        .remember_tool_evidence(&test_execution_record(
            "fs__read",
            ToolResult::ok("fs__read", json!({"content": "three"})),
        ))
        .expect("third evidence after restore");

    assert_ne!(first.id, third.id);
    assert_ne!(second.id, third.id);
}

fn large_tool_output_json(field: &str) -> String {
    json!({field: "line ".repeat((COMPACTION_TOOL_OUTPUT_CHAR_CAP + 500) / 5)}).to_string()
}

struct StaticSubagentDelegate {
    result: ToolResult,
}

struct CapturingSubagentDelegate {
    result: ToolResult,
    explorer_tasks: Arc<std::sync::Mutex<Vec<String>>>,
    fixer_tasks: Arc<std::sync::Mutex<Vec<String>>>,
}

impl SubagentDelegate<OpenAIConfig> for StaticSubagentDelegate {
    fn run_named<'a>(
        &'a self,
        _parent: &'a Agent<OpenAIConfig>,
        _agent_name: &'a str,
        _invocation: SubagentInvocation,
    ) -> Pin<Box<dyn Future<Output = Result<ToolResult>> + Send + 'a>> {
        let result = self.result.clone();
        Box::pin(async move { Ok(result) })
    }
}

impl SubagentDelegate<OpenAIConfig> for CapturingSubagentDelegate {
    fn run_named<'a>(
        &'a self,
        _parent: &'a Agent<OpenAIConfig>,
        agent_name: &'a str,
        invocation: SubagentInvocation,
    ) -> Pin<Box<dyn Future<Output = Result<ToolResult>> + Send + 'a>> {
        match agent_name {
            "fixer" => self
                .fixer_tasks
                .lock()
                .expect("fixer capture lock")
                .push(invocation.prompt),
            _ => self
                .explorer_tasks
                .lock()
                .expect("explorer capture lock")
                .push(invocation.prompt),
        }
        let result = self.result.clone();
        Box::pin(async move { Ok(result) })
    }
}

struct OverlapSubagentDelegate {
    barrier: Arc<Barrier>,
    started: Arc<Mutex<Vec<String>>>,
}

struct PollCountingSubagentDelegate {
    polls: Arc<AtomicUsize>,
}

impl SubagentDelegate<OpenAIConfig> for PollCountingSubagentDelegate {
    fn run_named<'a>(
        &'a self,
        _parent: &'a Agent<OpenAIConfig>,
        agent_name: &'a str,
        _invocation: SubagentInvocation,
    ) -> Pin<Box<dyn Future<Output = Result<ToolResult>> + Send + 'a>> {
        let polls = Arc::clone(&self.polls);
        let agent_name = agent_name.to_string();
        Box::pin(async move {
            polls.fetch_add(1, Ordering::SeqCst);
            Ok(ToolResult::ok(
                &format!("agent__{agent_name}"),
                json!({"ok": true}),
            ))
        })
    }
}

impl SubagentDelegate<OpenAIConfig> for OverlapSubagentDelegate {
    fn run_named<'a>(
        &'a self,
        _parent: &'a Agent<OpenAIConfig>,
        agent_name: &'a str,
        _invocation: SubagentInvocation,
    ) -> Pin<Box<dyn Future<Output = Result<ToolResult>> + Send + 'a>> {
        let barrier = Arc::clone(&self.barrier);
        let started = Arc::clone(&self.started);
        let agent_name = agent_name.to_string();
        Box::pin(async move {
            started
                .lock()
                .expect("started lock")
                .push(agent_name.clone());
            tokio::task::spawn_blocking(move || barrier.wait())
                .await
                .expect("barrier task joins");
            Ok(ToolResult::ok(
                &format!("agent__{agent_name}"),
                json!({"ok": true}),
            ))
        })
    }
}

fn static_delegate(result: ToolResult) -> Arc<dyn SubagentDelegate<OpenAIConfig>> {
    Arc::new(StaticSubagentDelegate { result })
}

fn capturing_delegate(
    result: ToolResult,
) -> (
    Arc<dyn SubagentDelegate<OpenAIConfig>>,
    Arc<std::sync::Mutex<Vec<String>>>,
    Arc<std::sync::Mutex<Vec<String>>>,
) {
    let explorer_tasks = Arc::new(std::sync::Mutex::new(Vec::new()));
    let fixer_tasks = Arc::new(std::sync::Mutex::new(Vec::new()));
    (
        Arc::new(CapturingSubagentDelegate {
            result,
            explorer_tasks: Arc::clone(&explorer_tasks),
            fixer_tasks: Arc::clone(&fixer_tasks),
        }),
        explorer_tasks,
        fixer_tasks,
    )
}

#[test]
fn tool_effects_classify_read_write_validation_command_diagnostic_and_workflow_control() {
    let read = ToolEffects::derive(
        "fs__read",
        Some(&json!({"path": "src/lib.rs"})),
        &ToolResult::ok(
            "fs__read",
            json!({"path": "src/lib.rs", "content": "fn main() {}"}),
        ),
    );
    assert_eq!(read.kind, ToolEffectKind::Read);
    assert_eq!(read.primary_path.as_deref(), Some("src/lib.rs"));
    assert!(read.edited_paths.is_empty());
    assert_eq!(read.command, None);

    let write = ToolEffects::derive(
        "edit__apply_patch",
        None,
        &ToolResult::ok(
            "edit__apply_patch",
            json!({"edits": [{"path": "src/lib.rs"}, {"path": "src/agent.rs"}]}),
        ),
    );
    assert_eq!(write.kind, ToolEffectKind::Write);
    assert_eq!(write.edited_paths, vec!["src/lib.rs", "src/agent.rs"]);

    let validation = ToolEffects::derive(
        "shell__exec",
        Some(&json!({"command": "cargo test transcript"})),
        &ToolResult::ok(
            "shell__exec",
            json!({"command": "cargo test transcript", "status": 0}),
        ),
    );
    assert_eq!(validation.kind, ToolEffectKind::Validation);
    assert_eq!(validation.command.as_deref(), Some("cargo test transcript"));

    let failed_validation = ToolEffects::derive(
        "shell__exec",
        Some(&json!({"command": "cargo test transcript"})),
        &ToolResult::ok(
            "shell__exec",
            json!({"command": "cargo test transcript", "status": 101, "success": false}),
        ),
    );
    assert_eq!(failed_validation.kind, ToolEffectKind::Diagnostic);
    assert_eq!(
        failed_validation.command.as_deref(),
        Some("cargo test transcript")
    );

    let contradictory_failed_validation = ToolEffects::derive(
        "shell__exec",
        Some(&json!({"command": "cargo test transcript"})),
        &ToolResult::ok(
            "shell__exec",
            json!({"command": "cargo test transcript", "status": 101, "success": true}),
        ),
    );
    assert_eq!(
        contradictory_failed_validation.kind,
        ToolEffectKind::Diagnostic
    );

    let checkout = ToolEffects::derive(
        "shell__exec",
        Some(&json!({"command": "git checkout main"})),
        &ToolResult::ok(
            "shell__exec",
            json!({"command": "git checkout main", "status": 0, "success": true}),
        ),
    );
    assert_eq!(checkout.kind, ToolEffectKind::Command);

    let command = ToolEffects::derive(
        "shell__exec",
        Some(&json!({"command": "ls src"})),
        &ToolResult::ok("shell__exec", json!({"command": "ls src", "status": 0})),
    );
    assert_eq!(command.kind, ToolEffectKind::Command);
    assert_eq!(command.command.as_deref(), Some("ls src"));

    let diagnostic = ToolEffects::derive(
        "shell__exec",
        Some(&json!({"command": "cargo test agent::tests::tool"})),
        &ToolResult::err("shell__exec", "command failed"),
    );
    assert_eq!(diagnostic.kind, ToolEffectKind::Diagnostic);
    assert_eq!(
        diagnostic.command.as_deref(),
        Some("cargo test agent::tests::tool")
    );

    let workflow = ToolEffects::derive(
        "workflow__todos",
        Some(&json!({"items": [{"id": "t1", "content": "x", "status": "pending"}]})),
        &ToolResult::ok("workflow__todos", json!({"ok": true})),
    );
    assert_eq!(workflow.kind, ToolEffectKind::WorkflowControl);
}

#[tokio::test]
async fn contiguous_different_role_subagents_overlap_and_reconcile_in_model_order() {
    let mut agent = test_agent();
    let started = Arc::new(Mutex::new(Vec::new()));
    agent.set_subagent_delegate(Arc::new(OverlapSubagentDelegate {
        barrier: Arc::new(Barrier::new(2)),
        started: Arc::clone(&started),
    }));
    let calls = vec![
        test_tool_call("agent__explore", r#"{"task":"inspect"}"#),
        test_tool_call("agent__fixer", r#"{"task":"change"}"#),
    ];
    agent
        .append_assistant_tool_calls("", &calls)
        .expect("append tool calls");

    tokio::time::timeout(
        Duration::from_secs(1),
        agent.execute_tool_calls_and_record(&calls, &mut |_| async { Ok(()) }, &mut |_| async {
            Ok(PermissionApproval::AllowOnce)
        }),
    )
    .await
    .expect("different roles should overlap")
    .expect("batch executes");

    assert_eq!(
        *started.lock().expect("started lock"),
        vec!["explorer", "fixer"]
    );
    let outputs = agent
        .history
        .iter()
        .filter_map(|item| match item {
            HistoryItem::ToolOutput { call_id, .. } => Some(call_id.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(outputs, vec!["call-agent__explore", "call-agent__fixer"]);
}

#[tokio::test]
async fn subagent_batch_reconciles_each_call_before_finalizing_the_next() {
    let mut agent = test_agent();
    agent.set_subagent_delegate(Arc::new(OverlapSubagentDelegate {
        barrier: Arc::new(Barrier::new(2)),
        started: Arc::new(Mutex::new(Vec::new())),
    }));
    let calls = vec![
        test_tool_call("agent__explore", r#"{"task":"inspect"}"#),
        test_tool_call("agent__fixer", r#"{"task":"change"}"#),
    ];
    agent
        .append_assistant_tool_calls("", &calls)
        .expect("append tool calls");
    let events = Arc::new(Mutex::new(Vec::new()));
    let recorded_events = Arc::clone(&events);

    agent
        .execute_tool_calls_and_record(
            &calls,
            &mut move |event| {
                let label = match event {
                    AgentEvent::ToolCallFinished { call_id, .. } => format!("finished:{call_id}"),
                    AgentEvent::ToolExecutionSummary(summary) => {
                        format!("summary:{}", summary.call_id)
                    }
                    AgentEvent::EvidenceRecorded(_) => "evidence".to_string(),
                    _ => return std::future::ready(Ok(())),
                };
                recorded_events.lock().expect("events lock").push(label);
                std::future::ready(Ok(()))
            },
            &mut |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await
        .expect("batch executes");

    assert_eq!(
        *events.lock().expect("events lock"),
        vec![
            "finished:call-agent__explore",
            "summary:call-agent__explore",
            "evidence",
            "finished:call-agent__fixer",
            "summary:call-agent__fixer",
            "evidence",
        ]
    );
}

#[tokio::test]
async fn subagent_batch_cancelled_record_callback_failure_is_not_treated_as_cancellation() {
    let mut agent = test_agent();
    agent.set_subagent_delegate(static_delegate(ToolResult::err_with_data(
        "agent__explore",
        "explorer cancelled",
        json!({
            "run_id": "run-1",
            "child_session_id": "child-session",
            "agent_name": "explorer",
            "status": "cancelled",
            "summary": "explorer cancelled",
        }),
    )));
    let call = test_tool_call("agent__explore", r#"{"task":"inspect"}"#);
    agent
        .append_assistant_tool_calls("", std::slice::from_ref(&call))
        .expect("append tool call");

    let error = agent
        .execute_tool_calls_and_record(
            std::slice::from_ref(&call),
            &mut |event| {
                if matches!(event, AgentEvent::EvidenceRecorded(_)) {
                    std::future::ready(Err(anyhow::anyhow!("evidence callback failed")))
                } else {
                    std::future::ready(Ok(()))
                }
            },
            &mut |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await
        .expect_err("callback failure must win over cancellation");

    assert_eq!(error.to_string(), "evidence callback failed");
}

#[tokio::test]
async fn subagent_batch_finished_callback_failure_does_not_record_effects() {
    let mut agent = test_agent();
    agent.set_subagent_delegate(static_delegate(ToolResult::ok(
        "agent__fixer",
        json!({
            "run_id": "run-1",
            "child_session_id": "child-session",
            "agent_name": "fixer",
            "status": "completed",
            "summary": "implemented change",
            "structured_result": {
                "status": "completed",
                "summary": "implemented change",
                "malformed": false,
                "findings": [],
                "files_read": [],
                "files_changed": ["src/agent/tool_execution.rs"],
                "commands_run": [],
                "validation": [],
                "blockers": [],
                "next_steps": [],
                "run_id": "run-1",
                "child_session_id": "child-session"
            }
        }),
    )));
    let call = test_tool_call("agent__fixer", r#"{"task":"apply change"}"#);
    agent
        .append_assistant_tool_calls("", std::slice::from_ref(&call))
        .expect("append tool call");

    let error = agent
        .execute_tool_calls_and_record(
            std::slice::from_ref(&call),
            &mut |event| {
                if matches!(event, AgentEvent::ToolCallFinished { .. }) {
                    std::future::ready(Err(anyhow::anyhow!("finished callback failed")))
                } else {
                    std::future::ready(Ok(()))
                }
            },
            &mut |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await
        .expect_err("finished callback failure aborts reconciliation");

    assert_eq!(error.to_string(), "finished callback failed");
    assert_eq!(agent.turn.counters.child_write_effects, 0);
    assert!(
        agent
            .history
            .iter()
            .all(|item| !matches!(item, HistoryItem::ToolOutput { .. }))
    );
}

#[tokio::test]
async fn subagent_batch_started_callback_failure_does_not_launch_or_record_history() {
    let mut agent = test_agent();
    let polls = Arc::new(AtomicUsize::new(0));
    agent.set_subagent_delegate(Arc::new(PollCountingSubagentDelegate {
        polls: Arc::clone(&polls),
    }));
    let calls = vec![
        test_tool_call("agent__explore", r#"{"task":"inspect"}"#),
        test_tool_call("agent__fixer", r#"{"task":"change"}"#),
    ];
    agent
        .append_assistant_tool_calls("", &calls)
        .expect("append tool calls");

    let events = Arc::new(Mutex::new(Vec::new()));
    let recorded_events = Arc::clone(&events);
    let result = agent
        .execute_tool_calls_and_record(
            &calls,
            &mut move |event| {
                let label = match event {
                    AgentEvent::ToolCallStarted { call_id, .. } => format!("started:{call_id}"),
                    AgentEvent::ToolCallFinished { call_id, .. } => format!("finished:{call_id}"),
                    AgentEvent::EvidenceRecorded(_) => "evidence".to_string(),
                    _ => return std::future::ready(Ok(())),
                };
                let fail = label == "started:call-agent__fixer";
                recorded_events.lock().expect("events lock").push(label);
                if fail {
                    std::future::ready(Err(anyhow::anyhow!("started callback failed")))
                } else {
                    std::future::ready(Ok(()))
                }
            },
            &mut |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await;

    assert!(result.is_err());
    assert_eq!(polls.load(Ordering::SeqCst), 0);
    assert_eq!(
        *events.lock().expect("events lock"),
        vec!["started:call-agent__explore", "started:call-agent__fixer"]
    );
    assert!(
        agent
            .history
            .iter()
            .all(|item| !matches!(item, HistoryItem::ToolOutput { .. }))
    );
}

#[tokio::test]
async fn ordinary_tool_is_a_barrier_between_subagent_batches() {
    let mut agent = test_agent();
    let started = Arc::new(Mutex::new(Vec::new()));
    agent.set_subagent_delegate(Arc::new(OverlapSubagentDelegate {
        barrier: Arc::new(Barrier::new(1)),
        started: Arc::clone(&started),
    }));
    let calls = vec![
        test_tool_call("agent__explore", r#"{"task":"inspect"}"#),
        test_tool_call("util__echo", r#"{"text":"barrier"}"#),
        test_tool_call("agent__fixer", r#"{"task":"change"}"#),
    ];
    agent
        .append_assistant_tool_calls("", &calls)
        .expect("append tool calls");
    agent
        .execute_tool_calls_and_record(&calls, &mut |_| async { Ok(()) }, &mut |_| async {
            Ok(PermissionApproval::AllowOnce)
        })
        .await
        .expect("calls execute");

    let outputs = agent
        .history
        .iter()
        .filter_map(|item| match item {
            HistoryItem::ToolOutput { call_id, .. } => Some(call_id.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        outputs,
        vec![
            "call-agent__explore",
            "call-util__echo",
            "call-agent__fixer"
        ]
    );
}

#[test]
fn agent_tool_definitions_hide_subagent_tools_until_delegate_is_installed() {
    let mut agent = test_agent();
    let specs = agent.tool_definitions();
    assert!(
        specs
            .iter()
            .any(|spec| spec.name == tool_names::TOOL_AGENT_RECONCILE)
    );
    for name in [
        "agent__explore",
        "agent__fixer",
        "agent__oracle",
        "agent__designer",
        "agent__librarian",
        "agent__general",
    ] {
        assert!(
            !specs.iter().any(|spec| spec.name == name),
            "{name} should be hidden"
        );
    }

    agent.set_subagent_delegate(static_delegate(ToolResult::ok(
        "agent__explore",
        json!({
            "run_id": "run-1",
            "child_session_id": "child-session",
            "agent_name": "explorer",
            "status": "completed",
            "summary": "done",
        }),
    )));

    let specs = agent.tool_definitions();
    for name in [
        "agent__explore",
        "agent__fixer",
        "agent__oracle",
        "agent__designer",
        "agent__librarian",
        "agent__general",
    ] {
        assert!(
            specs.iter().any(|spec| spec.name == name),
            "{name} should be exposed"
        );
    }
}

#[test]
fn agent_templates_expose_capability_contracts() {
    let explorer = AgentTemplate::explorer().capability_contract();
    assert_eq!(explorer.name, "explorer");
    assert_eq!(explorer.tool_scope, ToolScope::ReadOnlyExplorer);
    assert_eq!(explorer.permission_mode, PermissionMode::Default);
    assert!(!explorer.can_write);
    assert!(!explorer.can_delegate);
    assert_eq!(explorer.default_max_tool_calls, None);
    assert!(explorer.input_expectations.contains("task 或 objective"));
    assert!(explorer.expected_result_shape.contains("run_id"));

    let fixer = AgentTemplate::fixer().capability_contract();
    assert_eq!(fixer.name, "fixer");
    assert_eq!(fixer.tool_scope, ToolScope::FullAccess);
    assert!(fixer.can_write);
    assert!(!fixer.can_delegate);
    assert_eq!(fixer.default_max_tool_calls, None);

    let readonly_names = ["oracle", "designer", "librarian", "general"];
    for name in readonly_names {
        let template = AgentTemplate::from_name(name).expect("known template");
        let contract = template.capability_contract();
        assert_eq!(contract.name, name);
        assert_eq!(contract.tool_scope, ToolScope::ReadOnlyExplorer);
        assert!(!contract.can_write);
        assert!(!contract.can_delegate);
    }
}

#[test]
fn child_agent_applies_runtime_max_tool_call_override() {
    let agent = test_agent();
    let child =
        AgentFactory::create_child_with_max_tool_calls(&agent, &AgentTemplate::fixer(), Some(1));

    assert_eq!(child.max_tool_calls_limit(), Some(1));
    let error = child
        .ensure_tool_call_budget(0, 2)
        .expect_err("tool-call budget should be enforced");
    assert!(error.to_string().contains("too many tool calls"));
}

#[test]
fn child_agent_retains_explicit_tool_call_override_with_unbounded_parent() {
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base("https://api.openai.com/v1")
            .with_api_key("test"),
    );
    let agent = Agent::new(client, "m1", None, None);
    let child =
        AgentFactory::create_child_with_max_tool_calls(&agent, &AgentTemplate::fixer(), Some(2));

    assert_eq!(child.max_tool_calls_limit(), Some(2));
    child
        .ensure_tool_call_budget(0, 2)
        .expect("budget edge should pass");
    let error = child
        .ensure_tool_call_budget(0, 3)
        .expect_err("explicit child budget should still be enforced");
    assert!(error.to_string().contains("max 2"));
}

#[test]
fn child_agent_inherits_parent_tool_call_limit_without_template_budget() {
    let agent = test_agent();
    let child = AgentFactory::create_child(&agent, &AgentTemplate::fixer());

    assert_eq!(child.max_tool_calls_limit(), agent.max_tool_calls_limit());
}

#[tokio::test]
async fn subagent_tool_execution_normalizes_bounded_input_before_delegation() {
    let mut agent = test_agent();
    let (delegate, explorer_tasks, fixer_tasks) = capturing_delegate(ToolResult::ok(
        "agent__fixer",
        json!({
            "run_id": "run-1",
            "child_session_id": "child-session",
            "agent_name": "fixer",
            "status": "completed",
            "summary": "done"
        }),
    ));
    agent.set_subagent_delegate(delegate);

    let output = agent
        .execute_subagent_tool(
            "agent__fixer",
            &json!({
                "objective": "Implement contract",
                "success_criteria": ["tests pass"],
                "allowed_paths": ["src/agent.rs"],
                "owned_paths": ["src/tool.rs"],
                "timeout_secs": 30,
                "max_tool_calls": 5
            }),
        )
        .await;

    assert!(output.ok);
    assert!(explorer_tasks.lock().expect("explorer tasks").is_empty());
    let fixer_tasks = fixer_tasks.lock().expect("fixer tasks");
    assert_eq!(fixer_tasks.len(), 1);
    let prompt = &fixer_tasks[0];
    assert!(prompt.contains("Objective: Implement contract"));
    assert!(prompt.contains("Success criteria:"));
    assert!(prompt.contains("Allowed paths: src/agent.rs"));
    assert!(prompt.contains("Owned paths: src/tool.rs"));
    assert!(prompt.contains("Execution bounds: timeout_secs=30, max_tool_calls=5"));
    assert!(prompt.contains("do not recursively delegate"));
}

#[tokio::test]
async fn subagent_tool_execution_supports_legacy_task_only_input() {
    let mut agent = test_agent();
    let (delegate, explorer_tasks, _fixer_tasks) = capturing_delegate(ToolResult::ok(
        "agent__explore",
        json!({
            "run_id": "run-1",
            "child_session_id": "child-session",
            "agent_name": "explorer",
            "status": "completed",
            "summary": "done"
        }),
    ));
    agent.set_subagent_delegate(delegate);

    let output = agent
        .execute_subagent_tool(
            "agent__explore",
            &json!({"task": "inspect src/subagent.rs"}),
        )
        .await;

    assert!(output.ok);
    let explorer_tasks = explorer_tasks.lock().expect("explorer tasks");
    assert_eq!(explorer_tasks.len(), 1);
    assert!(explorer_tasks[0].contains("Objective: inspect src/subagent.rs"));
    assert!(explorer_tasks[0].contains("Mode: read-only exploration only."));
}

#[tokio::test]
async fn readonly_expert_subagent_tool_execution_routes_through_generic_delegate() {
    let mut agent = test_agent();
    let (delegate, explorer_tasks, fixer_tasks) = capturing_delegate(ToolResult::ok(
        "agent__oracle",
        json!({
            "run_id": "run-1",
            "child_session_id": "child-session",
            "agent_name": "oracle",
            "status": "completed",
            "summary": "done"
        }),
    ));
    agent.set_subagent_delegate(delegate);

    let output = agent
        .execute_subagent_tool("agent__oracle", &json!({"task": "analyze failure mode"}))
        .await;

    assert!(output.ok);
    assert!(fixer_tasks.lock().expect("fixer tasks").is_empty());
    let readonly_tasks = explorer_tasks.lock().expect("readonly tasks");
    assert_eq!(readonly_tasks.len(), 1);
    assert!(readonly_tasks[0].contains("Objective: analyze failure mode"));
}

#[tokio::test]
async fn subagent_tool_execution_returns_validation_error_for_missing_objective() {
    let agent = test_agent();

    let output = agent
        .execute_subagent_tool("agent__explore", &json!({}))
        .await;

    assert!(!output.ok);
    assert!(
        output
            .error
            .as_ref()
            .expect("validation error")
            .message
            .contains("requires a non-empty 'task' or 'objective'")
    );
}

#[test]
fn remembered_subagent_evidence_carries_parent_turn_provenance() {
    let mut agent = test_agent();
    agent.turn.turn_id = 42;
    let record = ToolExecutionRecord {
        call_id: "call-agent__explore".into(),
        tool_name: "agent__explore".into(),
        arguments: Some(json!({"task": "inspect"})),
        permission_class: crate::permission::ToolPermissionClass::Preview,
        directive: crate::permission::ExecutionDirective::None,
        status: ToolExecutionStatus::Executed,
        rejection: None,
        output: ToolResult::ok(
            "agent__explore",
            json!({
                "run_id": "run-1",
                "child_session_id": "child-1",
                "status": "completed",
                "summary": "done"
            }),
        ),
        effects: ToolEffects {
            kind: ToolEffectKind::Read,
            primary_path: None,
            edited_paths: vec![],
            command: None,
        },
    };

    let evidence = agent
        .remember_tool_evidence(&record)
        .expect("subagent evidence should be recorded");

    match evidence.source {
        EvidenceSource::Subagent {
            run_id,
            child_session_id,
            parent_tool,
            parent_turn_id,
            ..
        } => {
            assert_eq!(run_id, "run-1");
            assert_eq!(child_session_id, "child-1");
            assert_eq!(parent_tool, "agent__explore");
            assert_eq!(parent_turn_id.as_deref(), Some("turn-42"));
        }
        other => panic!("unexpected evidence source: {other:?}"),
    }
}

#[tokio::test]
async fn tool_output_emits_projected_provider_usage() {
    let mut agent = test_agent();
    let call = test_tool_call("fs__read", "{}");
    agent
        .append_assistant_tool_calls("", std::slice::from_ref(&call))
        .expect("assistant tool calls append");
    agent.install_provider_usage_anchor_for_test(provider_usage(100));
    let mut events = Vec::new();

    agent
        .execute_tool_call_and_record(
            &call,
            &mut |event| {
                events.push(event);
                std::future::ready(Ok(()))
            },
            &mut |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await
        .expect("tool call completes");

    let expected_delta = serde_json::to_string(
        agent
            .history_for_test()
            .last()
            .expect("tool output appended"),
    )
    .expect("tool output serializes")
    .len()
    .div_ceil(4) as u64;
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::TokenUsageUpdated {
            used_tokens,
            input_tokens,
            output_tokens,
            cache_report,
            ..
        } if *used_tokens == 100 + expected_delta
            && *input_tokens == 100 + expected_delta
            && *output_tokens == 0
            && cache_report.is_none()
    )));
}

#[tokio::test]
async fn cancelled_agent_explore_records_tool_output_before_interrupting_turn() {
    let mut agent = test_agent();
    agent.set_subagent_delegate(static_delegate(ToolResult::err_with_data(
        "agent__explore",
        "explorer cancelled",
        json!({
            "run_id": "run-1",
            "child_session_id": "child-session",
            "agent_name": "explorer",
            "status": "cancelled",
            "summary": "explorer cancelled",
        }),
    )));
    let call = test_tool_call("agent__explore", r#"{"task":"inspect"}"#);
    agent
        .append_assistant_tool_calls("", std::slice::from_ref(&call))
        .expect("assistant tool calls should append");
    let mut events = Vec::new();

    let error = agent
        .execute_tool_call_and_record(
            &call,
            &mut |event| {
                events.push(event);
                std::future::ready(Ok(()))
            },
            &mut |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await
        .expect_err("cancelled explorer interrupts the turn after recording output");

    assert!(error.to_string().contains("agent__explore cancelled"));
    assert!(matches!(
        agent.history.last(),
        Some(HistoryItem::ToolOutput {
            call_id,
            output_json,
        }) if call_id == "call-agent__explore"
            && output_json.contains("cancelled")
            && output_json.contains("child-session")
    ));
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::ToolCallFinished {
            name,
            ok: false,
            output,
            ..
        } if name == "agent__explore"
            && output
                .data
                .as_ref()
                .and_then(|data| data.get("status"))
                .and_then(Value::as_str)
                == Some("cancelled")
    )));
}

#[tokio::test]
async fn cancelled_agent_fixer_records_tool_output_before_interrupting_turn() {
    let mut agent = test_agent();
    agent.set_subagent_delegate(static_delegate(ToolResult::err_with_data(
        "agent__fixer",
        "fixer cancelled",
        json!({
            "run_id": "run-1",
            "child_session_id": "child-session",
            "agent_name": "fixer",
            "status": "cancelled",
            "summary": "fixer cancelled",
        }),
    )));
    let call = test_tool_call("agent__fixer", r#"{"task":"apply requested fix"}"#);
    agent
        .append_assistant_tool_calls("", std::slice::from_ref(&call))
        .expect("assistant tool calls should append");
    let mut events = Vec::new();

    let error = agent
        .execute_tool_call_and_record(
            &call,
            &mut |event| {
                events.push(event);
                std::future::ready(Ok(()))
            },
            &mut |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await
        .expect_err("cancelled fixer interrupts the turn after recording output");

    assert!(error.to_string().contains("agent__fixer cancelled"));
    assert!(matches!(
        agent.history.last(),
        Some(HistoryItem::ToolOutput {
            call_id,
            output_json,
        }) if call_id == "call-agent__fixer"
            && output_json.contains("cancelled")
            && output_json.contains("child-session")
    ));
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::ToolCallFinished {
            name,
            ok: false,
            output,
            ..
        } if name == "agent__fixer"
            && output
                .data
                .as_ref()
                .and_then(|data| data.get("status"))
                .and_then(Value::as_str)
                == Some("cancelled")
    )));
}

#[tokio::test]
async fn delegated_structured_subagent_results_surface_in_next_turn_prelude() {
    let mut agent = test_agent();
    agent.prepare_turn_prelude("Delegate implementation work");
    agent.set_subagent_delegate(static_delegate(ToolResult::ok(
        "agent__fixer",
        json!({
            "run_id": "run-structured-1",
            "child_session_id": "child-structured-1",
            "agent_name": "fixer",
            "status": "completed",
            "summary": "implemented bounded fix",
            "structured_result": {
                "status": "completed",
                "summary": "implemented bounded fix",
                "malformed": false,
                "findings": [],
                "files_read": ["src/agent.rs"],
                "files_changed": ["src/agent.rs"],
                "commands_run": ["cargo test subagent --quiet"],
                "validation": ["cargo test subagent --quiet passed"],
                "blockers": [],
                "next_steps": ["reconcile in parent turn"],
                "run_id": "run-structured-1",
                "child_session_id": "child-structured-1"
            }
        }),
    )));

    let call = test_tool_call("agent__fixer", r#"{"task":"implement bounded fix"}"#);
    agent
        .append_assistant_tool_calls("", std::slice::from_ref(&call))
        .expect("assistant tool calls should append");
    let mut events = Vec::new();
    agent
        .execute_tool_call_and_record(
            &call,
            &mut |event| {
                events.push(event);
                std::future::ready(Ok(()))
            },
            &mut |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await
        .expect("subagent tool execution should succeed");

    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::EvidenceRecorded(record)
            if record.tags.iter().any(|tag| tag == "subagent_result")
                && record.tags.iter().any(|tag| tag == "unreconciled")
    )));

    let jobs = agent.pending_subagent_jobs();
    assert_eq!(jobs.len(), 1);
    assert_eq!(jobs[0].agent_name, "fixer");
    assert_eq!(jobs[0].run_id, "run-structured-1");
    assert_eq!(jobs[0].child_session_id, "child-structured-1");
    assert_eq!(jobs[0].summary, "implemented bounded fix");

    let prelude = agent.prepare_turn_prelude("Reconcile child work");
    assert!(prelude.iter().any(|message| {
        message.text.contains("Pending child subagent results")
            && message.text.contains("run-structured-1")
            && message.text.contains("implemented bounded fix")
            && message.text.contains("child-structured-1")
            && message.text.contains("child_session_id=child-structured-1")
            && message.text.contains("child transcript navigation")
    }));
}

#[test]
fn model_switch_uses_new_metadata_for_next_request_build() {
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base("https://api.openai.com/v1")
            .with_api_key("test"),
    );
    let mut agent = Agent::new(client, "m1", 1, 1);

    let mut catalog = HashMap::new();
    catalog.insert(
        "m1".to_string(),
        ModelRequestMetadata {
            context_window: Some(2048),
            max_output_tokens: Some(256),
            supports_tools: true,
            supports_reasoning: false,
            ..Default::default()
        },
    );
    catalog.insert(
        "m2".to_string(),
        ModelRequestMetadata {
            context_window: Some(128_000),
            max_output_tokens: Some(256),
            supports_tools: true,
            supports_reasoning: false,
            ..Default::default()
        },
    );
    agent.set_model_catalog(catalog);

    // Simulate first user message.
    agent
        .append_history_item(HistoryItem::user("hello"))
        .expect("history append succeeds");
    let history = agent.history_items();
    let b1 = build_test_request(TestRequestBuilderInput {
        protocol: ApiProtocol::Responses,
        model_id: agent.model(),
        model: agent.active_model_metadata(),
        prelude: &agent.prelude,
        history: &history,
        protected_start_index: history.len().saturating_sub(1),
        tools: &[],
        evidence: &[],
        history_adapter: None,
        context_view: None,
    })
    .expect("request builds");
    assert_eq!(b1.budget.context_window_tokens, 2048.max(1024));

    // Switch model and build again.
    agent.set_model("m2");
    let history = agent.history_items();
    let b2 = build_test_request(TestRequestBuilderInput {
        protocol: ApiProtocol::Responses,
        model_id: agent.model(),
        model: agent.active_model_metadata(),
        prelude: &agent.prelude,
        history: &history,
        protected_start_index: history.len().saturating_sub(1),
        tools: &[],
        evidence: &[],
        history_adapter: None,
        context_view: None,
    })
    .expect("request builds");
    assert!(b2.budget.context_window_tokens > b1.budget.context_window_tokens);
}

#[test]
fn inline_reasoning_extractor_splits_think_tags_from_visible_text() {
    let mut extractor = InlineReasoningExtractor::new("r-1");

    let mut parts = extractor.push("hello <thi");
    parts.extend(extractor.push("nk>plan</think> answer"));
    parts.extend(extractor.finish());

    assert_eq!(
        parts,
        vec![
            StreamTextPart::Visible("hello ".into()),
            StreamTextPart::ReasoningDelta {
                item_id: "r-1".into(),
                delta: "plan".into(),
            },
            StreamTextPart::ReasoningDone {
                item_id: "r-1".into(),
                text: "plan".into(),
            },
            StreamTextPart::Visible(" answer".into()),
        ]
    );
}

#[test]
fn compatible_chat_delta_reads_native_reasoning_fields() {
    for (field, expected) in [
        ("reasoning_content", "plan"),
        ("reasoning", "think"),
        ("thinking", "ponder"),
    ] {
        let raw = serde_json::json!({
            "content": null,
            field: expected,
        });
        let delta: CompatibleChatCompletionStreamResponseDelta =
            serde_json::from_value(raw).expect("delta deserializes");

        assert_eq!(delta.reasoning_delta().as_deref(), Some(expected));
    }
}

#[test]
fn protocol_frames_remain_authoritative_for_history_cache() {
    let mut agent = test_agent();
    agent
        .append_history_item(HistoryItem::user("hello"))
        .expect("user append succeeds");
    agent
        .append_history_item(HistoryItem::AssistantToolCalls {
            text: Some("working".into()),
            reasoning_content: None,
            calls: vec![test_tool_call("fs__read", r#"{"path":"src/main.rs"}"#)],
        })
        .expect("tool call append succeeds");
    agent
        .append_history_item(HistoryItem::ToolOutput {
            call_id: "call-fs__read".into(),
            output_json: r#"{"ok":true}"#.into(),
        })
        .expect("tool output append succeeds");

    assert_eq!(
        crate::protocol_frames::history_items_from_frames(agent.protocol_frames_for_test()),
        agent.history_for_test()
    );
    assert_eq!(
        agent.runtime_snapshot.frames.len(),
        agent.protocol_frames_for_test().len()
    );
}

#[test]
fn append_history_item_is_atomic_when_protocol_validation_fails() {
    let mut agent = test_agent();
    agent
        .append_history_item(HistoryItem::user("hello"))
        .expect("user append succeeds");

    let history_before = agent.history.clone();
    let frames_before = agent.protocol_frames.clone();
    let snapshot_before = agent.runtime_snapshot.clone();

    let error = agent
        .append_history_item(HistoryItem::ToolOutput {
            call_id: "call-orphan".into(),
            output_json: "{}".into(),
        })
        .expect_err("orphan tool output must fail");

    assert!(error.to_string().contains("orphan tool output"));
    assert_eq!(agent.history, history_before);
    assert_eq!(agent.protocol_frames, frames_before);
    assert_eq!(agent.runtime_snapshot, snapshot_before);
}

#[test]
fn compatible_chat_delta_reads_object_and_array_reasoning() {
    let raw = serde_json::json!({
        "reasoning_content": [
            {"text": "step "},
            {"content": "one"}
        ]
    });
    let delta: CompatibleChatCompletionStreamResponseDelta =
        serde_json::from_value(raw).expect("delta deserializes");

    assert_eq!(delta.reasoning_delta().as_deref(), Some("step one"));
}

#[test]
fn compatible_chat_stream_accepts_terminal_chunk_without_delta() {
    let raw = serde_json::json!({
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 1780856440_u64,
        "model": "gpt-5.5",
        "choices": [{
            "index": 0,
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 3060,
            "completion_tokens": 25,
            "total_tokens": 3085
        }
    });

    let response: CompatibleChatCompletionStreamResponse =
        serde_json::from_value(raw).expect("terminal chunk deserializes");

    assert_eq!(response.choices.len(), 1);
    assert!(response.choices[0].delta.is_none());
    assert_eq!(response.choices[0].finish_reason, Some(FinishReason::Stop));
    assert_eq!(
        response.usage.as_ref().map(|usage| usage.prompt_tokens),
        Some(3060)
    );
    assert_eq!(
        response.usage.as_ref().map(|usage| usage.completion_tokens),
        Some(25)
    );
}

#[test]
fn sse_parser_drains_data_events_and_done_marker() {
    let mut buffer = String::new();
    append_sse_chunk(&mut buffer, b"data: {\"choices\":[]}\n\ndata: [DONE]\n\n");

    assert_eq!(
        drain_sse_data_events(&mut buffer),
        vec![Some(r#"{"choices":[]}"#.into()), None]
    );
    assert!(buffer.is_empty());
}

#[test]
fn ignores_non_terminal_lifecycle_events_missing_model() {
    for event_type in ["response.created", "response.in_progress"] {
        let raw = serde_json::json!({
            "type": event_type,
            "sequence_number": 1,
            "response": {
                "id": "resp_test",
                "object": "response",
                "created_at": 1780765723_u64,
                "status": "in_progress",
                "background": false,
                "error": null,
                "output": []
            }
        });
        assert!(
            is_ignorable_response_lifecycle_event(&raw),
            "{event_type} should be ignored"
        );
    }
}

#[test]
fn does_not_ignore_other_stream_deserialize_errors() {
    let raw = serde_json::json!({
        "type": "response.completed",
        "sequence_number": 1,
        "response": {
            "id": "resp_test",
            "object": "response",
            "created_at": 1780765723_u64,
            "status": "completed",
            "background": false,
            "error": null,
            "output": []
        }
    });
    assert!(!is_ignorable_response_lifecycle_event(&raw));
}

#[test]
fn projects_provider_reasoning_efforts_without_mutating_completed_events() {
    for effort in ["max", "provider-ultra"] {
        let raw = serde_json::json!({
            "type": "response.completed", "sequence_number": 1,
            "response": {
                "id": "resp_test", "object": "response", "created_at": 1780765723_u64,
                "status": "completed", "background": false, "error": null,
                "incomplete_details": null, "instructions": null, "max_output_tokens": null,
                "model": "m1", "output": [], "parallel_tool_calls": true,
                "previous_response_id": null, "reasoning": {"effort": effort}, "store": true,
                "temperature": 1, "text": {"format": {"type": "text"}}, "tool_choice": "auto",
                "tools": [], "top_p": 1, "truncation": "disabled",
                "usage": {"input_tokens": 5, "input_tokens_details": {"cached_tokens": 0}, "output_tokens": 3, "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 8},
                "user": null, "metadata": {}
            }
        });
        let event = project_response_stream_event(&raw)
            .expect("completed event should project")
            .expect("completed event must not be ignored");
        let ResponseStreamEvent::ResponseCompleted(event) = event else {
            panic!("expected completion")
        };
        assert_eq!(event.response.id, "resp_test");
        assert!(event.response.output.is_empty());
        assert_eq!(event.response.usage.expect("usage").total_tokens, 8);
        assert_eq!(raw["response"]["reasoning"]["effort"], effort);
    }
}

#[test]
fn rejects_malformed_completed_events_after_projection() {
    let raw = serde_json::json!({"type": "response.completed", "response": {"reasoning": {"effort": "max"}}});
    assert!(project_response_stream_event(&raw).is_err());
}

#[test]
fn compact_indexed_chat_tool_calls_does_not_synthesize_missing_indices() {
    let mut indexed = BTreeMap::new();
    let mut call = ChatCompletionMessageToolCall::default();
    call.id = "call-1".into();
    call.function.name = "fs__write".into();
    call.function.arguments = r#"{"path":"a.txt","content":"ok"}"#.into();
    indexed.insert(1, call);

    let compacted = compact_indexed_chat_tool_calls(indexed);

    assert_eq!(compacted.len(), 1);
    assert_eq!(compacted[0].id, "call-1");
    assert_eq!(compacted[0].function.name, "fs__write");
    validate_chat_tool_calls(&compacted).expect("valid sparse-index tool call");
}

#[test]
fn chat_tool_call_chunk_empty_name_does_not_overwrite_real_name() {
    let mut indexed = BTreeMap::new();
    for raw in [
        serde_json::json!({
            "index": 0,
            "id": "call-1",
            "type": "function",
            "function": {"name": "fs__write", "arguments": ""}
        }),
        serde_json::json!({
            "index": 0,
            "function": {"name": "", "arguments": "{\"path\":"}
        }),
        serde_json::json!({
            "index": 0,
            "function": {"name": "", "arguments": "\"a.txt\",\"content\":\"ok\"}"}
        }),
    ] {
        let chunk: ChatCompletionMessageToolCallChunk =
            serde_json::from_value(raw).expect("chunk deserializes");
        merge_chat_tool_call_chunk(&mut indexed, chunk);
    }

    let compacted = compact_indexed_chat_tool_calls(indexed);

    assert_eq!(compacted.len(), 1);
    assert_eq!(compacted[0].id, "call-1");
    assert_eq!(compacted[0].function.name, "fs__write");
    assert_eq!(
        compacted[0].function.arguments,
        r#"{"path":"a.txt","content":"ok"}"#
    );
    validate_chat_tool_calls(&compacted).expect("valid streamed tool call");
}

#[test]
fn classifies_lightweight_and_engineering_turns() {
    assert_eq!(
        classify_turn_intent("Explain how Rust ownership works."),
        TurnIntent::Lightweight
    );
    assert_eq!(
        classify_turn_intent("Explain what this function does."),
        TurnIntent::Lightweight
    );
    assert_eq!(
        classify_turn_intent(
            "Fix the failing tests in src/agent.rs and update the implementation."
        ),
        TurnIntent::Engineering
    );
}

#[test]
fn prepare_turn_prelude_assigns_incrementing_turn_ids() {
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base("https://api.openai.com/v1")
            .with_api_key("test"),
    );
    let mut agent = Agent::new(client, "m1", 1, 1);

    agent.prepare_turn_prelude("first turn");
    assert_eq!(agent.current_turn_id(), 1);

    agent.prepare_turn_prelude("second turn");
    assert_eq!(agent.current_turn_id(), 2);
}

#[test]
fn restore_session_context_seeds_next_turn_id() {
    let mut agent = test_agent();

    agent
        .restore_session_context(Vec::new(), Vec::new(), 7)
        .expect("restore session context");
    agent.prepare_turn_prelude("resumed turn");

    assert_eq!(agent.current_turn_id(), 8);
}

#[test]
fn candidate_session_usage_failure_leaves_live_agent_unchanged() {
    let mut agent = test_agent();
    agent
        .restore_session_history(vec![HistoryItem::user("old session")], Vec::new(), 7)
        .expect("restore old session");
    agent.prepare_turn_prelude("active turn");
    let mut invalid_metadata = ModelRequestMetadata::default();
    invalid_metadata.effective_input_limit_tokens = Some(0);
    agent.set_model_catalog(HashMap::from([(
        String::from("invalid-model"),
        invalid_metadata,
    )]));
    let target_snapshot = runtime_snapshot_for_history(
        ROOT_CONTEXT_BRANCH_ID,
        &[HistoryItem::user("target session")],
    );
    let model = agent.model.clone();
    let history = agent.history.clone();
    let protocol_frames = agent.protocol_frames.clone();
    let runtime_snapshot = agent.runtime_snapshot.clone();
    let turn_id = agent.current_turn_id();
    let next_turn_id = agent.next_turn_id;

    let error = agent
        .candidate_session_token_usage("invalid-model", &target_snapshot)
        .expect_err("invalid target metadata must fail");

    assert!(
        error
            .to_string()
            .contains("effective_input_limit_tokens must be greater than 0")
    );
    assert_eq!(agent.model, model);
    assert_eq!(agent.history, history);
    assert_eq!(agent.protocol_frames, protocol_frames);
    assert_eq!(agent.runtime_snapshot, runtime_snapshot);
    assert_eq!(agent.current_turn_id(), turn_id);
    assert_eq!(agent.next_turn_id, next_turn_id);
}

#[test]
fn restore_runtime_snapshot_keeps_projected_runtime_state_authoritative() {
    let mut agent = test_agent();
    let history = vec![
        HistoryItem::user("resume question"),
        HistoryItem::assistant("resume answer"),
    ];
    let frames = crate::protocol_frames::history_items_to_frames(&history);
    let mut snapshot = RuntimeSnapshot::new("feature")
        .with_session_id("session-1")
        .with_latest_model("m1")
        .with_leaf_sequence(12)
        .with_current_turn_id(7);
    snapshot.frames = runtime_frames_for_history(&history);
    snapshot.set_evidence(vec![EvidenceRecord {
        id: "evidence-1".into(),
        sequence: 1,
        timestamp_ms: 1,
        evidence_kind: crate::evidence::EvidenceKind::Decision,
        title: "Restored evidence".into(),
        summary: "restored evidence".into(),
        detail: None,
        source: EvidenceSource::Transcript { sequence: 1 },
        tags: Vec::new(),
    }]);

    agent
        .restore_runtime_snapshot(frames.clone(), snapshot.clone())
        .expect("restore runtime snapshot");

    assert_eq!(
        agent.protocol_frames_for_test(),
        snapshot.active_protocol_frames().as_slice()
    );
    assert_eq!(
        agent.history_for_test(),
        crate::protocol_frames::history_items_from_frames(&frames).as_slice()
    );
    assert_eq!(agent.runtime_snapshot_for_test(), &snapshot);
    assert_eq!(agent.evidence(), snapshot.evidence.as_slice());
    agent.prepare_turn_prelude("continued turn");
    assert_eq!(agent.current_turn_id(), 8);
}

#[test]
fn new_session_reset_discards_restored_runtime_metadata() {
    let mut agent = test_agent();
    let history = vec![HistoryItem::user("old prompt")];
    let mut snapshot = RuntimeSnapshot::new("feature")
        .with_session_id("old-session")
        .with_latest_model("old-model")
        .with_leaf_sequence(12)
        .with_current_turn_id(7);
    snapshot.frames = runtime_frames_for_history(&history);
    agent
        .restore_runtime_snapshot(
            crate::protocol_frames::history_items_to_frames(&history),
            snapshot,
        )
        .expect("restore runtime snapshot");

    agent.reset_for_new_session();

    assert!(agent.history_for_test().is_empty());
    assert!(agent.protocol_frames_for_test().is_empty());
    assert!(agent.evidence().is_empty());
    assert_eq!(
        agent.runtime_snapshot_for_test(),
        &RuntimeSnapshot::new(ROOT_CONTEXT_BRANCH_ID).with_latest_model("m1")
    );
    agent.prepare_turn_prelude("fresh turn");
    assert_eq!(agent.current_turn_id(), 1);
}

#[tokio::test]
async fn manual_compaction_noops_when_history_is_empty() {
    let mut agent = test_agent();
    let mut events = Vec::new();

    let outcome = agent
        .compact_session_async(|event| {
            events.push(event);
            std::future::ready(Ok(()))
        })
        .await
        .expect("manual compaction should not fail");

    assert_eq!(
        outcome,
        ManualCompactionOutcome::NoProgress(CompactionNoProgress {
            trigger: CompactionTrigger::Manual,
            blockers: vec![CompactionBlocker::NoHistoricalItems],
        })
    );
    assert!(matches!(
        events.as_slice(),
        [
            AgentEvent::ContextCompactionStarted {
                trigger: CompactionTrigger::Manual
            },
            AgentEvent::ContextCompactionNoProgress(CompactionNoProgress {
                trigger: CompactionTrigger::Manual,
                blockers,
            }),
        ] if blockers == &vec![CompactionBlocker::NoHistoricalItems]
    ));
}

#[tokio::test]
async fn manual_compaction_compacts_short_completed_history() {
    let checkpoint = valid_checkpoint("compact summary");
    let (base_url, requests, server) =
        spawn_chat_completion_server(vec![sse_response(format!(
            "data: {{\"choices\":[{{\"index\":0,\"delta\":{{\"content\":{}}},\"finish_reason\":\"stop\"}}]}}\n\ndata: [DONE]\n\n",
            serde_json::to_string(&checkpoint).expect("checkpoint JSON")
        ))])
        .await;
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base(base_url)
            .with_api_key("test"),
    );
    let mut agent = Agent::new(client, "m1", 4, 4);
    agent.set_default_protocol(ApiProtocol::Completions);
    agent
        .replace_history(vec![
            HistoryItem::user("short prompt"),
            HistoryItem::assistant("reply"),
        ])
        .expect("history replace succeeds");
    agent.runtime_snapshot = runtime_snapshot_for_history("main", &agent.history);

    let mut events = Vec::new();
    let outcome = agent
        .compact_session_async(|event| {
            events.push(event);
            async { Ok(()) }
        })
        .await
        .expect("manual compaction should not fail");

    assert_eq!(
        outcome,
        ManualCompactionOutcome::Compacted { retained_items: 1 }
    );
    assert!(matches!(
        agent.history.as_slice(),
        [HistoryItem::ContextSummary { text }] if text == "compact summary"
    ));
    assert!(matches!(
        events.as_slice(),
        [
            AgentEvent::ContextCompactionStarted {
                trigger: CompactionTrigger::Manual
            },
            AgentEvent::ContextCompactionDelta { .. },
            AgentEvent::ContextCompacted(_),
        ]
    ));
    assert_eq!(requests.load(Ordering::SeqCst), 1);
    server.await.expect("summary server completes");
}

fn valid_checkpoint(summary: &str) -> String {
    summary.to_string()
}

#[tokio::test]
async fn manual_compaction_retires_completed_active_turn_prefix_and_rebases_to_incomplete_suffix() {
    let checkpoint = valid_checkpoint("continue validating the active request");
    let (base_url, _requests, server) =
        spawn_chat_completion_server(vec![chat_final_sse(&checkpoint)]).await;
    let mut agent = Agent::new(
        Client::with_config(
            OpenAIConfig::new()
                .with_api_base(base_url)
                .with_api_key("test"),
        ),
        "m1",
        8,
        8,
    );
    agent.set_default_protocol(ApiProtocol::Completions);
    agent
        .replace_history(vec![
            HistoryItem::user("older turn"),
            HistoryItem::assistant("older answer"),
            HistoryItem::user("active request"),
            HistoryItem::AssistantToolCalls {
                text: None,
                reasoning_content: None,
                calls: vec![HistoryToolCall {
                    call_id: "call-pending".into(),
                    name: "fs__read".into(),
                    arguments_json: r#"{"path":"src/main.rs"}"#.into(),
                }],
            },
        ])
        .expect("active incomplete history");
    agent.runtime_snapshot = runtime_snapshot_for_history("main", &agent.history);
    agent.turn.turn_id = 9;
    agent.turn.current_turn_start_index = Some(2);
    agent.runtime_snapshot.current_turn_id = Some(9);
    agent.runtime_snapshot.current_segment_id = Some(0);
    let pending_id = agent.runtime_snapshot.frames[3].id;
    agent
        .runtime_snapshot
        .set_turn_protected_frame_ids(vec![pending_id]);

    let mut events = Vec::new();
    agent
        .compact_session_async(|event| {
            events.push(event);
            std::future::ready(Ok(()))
        })
        .await
        .expect("compacts older turns only");

    assert!(matches!(
        events.first(),
        Some(AgentEvent::ContextCompactionStarted {
            trigger: CompactionTrigger::Manual
        })
    ));
    assert!(matches!(
        events.last(),
        Some(AgentEvent::ContextCompacted(_))
    ));
    assert!(
        events
            .iter()
            .any(|event| matches!(event, AgentEvent::ContextCompactionDelta { .. }))
    );
    assert_eq!(agent.current_turn_id(), 9);
    assert_eq!(agent.runtime_snapshot.current_turn_id, Some(9));
    assert_eq!(agent.runtime_snapshot.current_segment_id, Some(0));
    // summary + pending tool call; the active user is part of the summary.
    assert_eq!(agent.history_for_test().len(), 2);
    assert!(matches!(
        agent.history_for_test().first(),
        Some(HistoryItem::ContextSummary { .. })
    ));
    assert_eq!(agent.turn.current_turn_start_index, None);
    assert!(matches!(
        agent.history_for_test().get(1),
        Some(HistoryItem::AssistantToolCalls { .. })
    ));
    server.await.expect("server task should finish");
}
#[tokio::test]
async fn manual_compaction_retires_an_entire_completed_active_turn_and_keeps_it_live() {
    let checkpoint = valid_checkpoint("continue validating the active request");
    let (base_url, _requests, server) =
        spawn_chat_completion_server(vec![chat_final_sse(&checkpoint)]).await;
    let mut agent = Agent::new(
        Client::with_config(
            OpenAIConfig::new()
                .with_api_base(base_url)
                .with_api_key("test"),
        ),
        "m1",
        8,
        8,
    );
    agent.set_default_protocol(ApiProtocol::Completions);
    agent
        .replace_history(vec![
            HistoryItem::user("older request"),
            HistoryItem::assistant("older work"),
            HistoryItem::user("active request"),
            HistoryItem::assistant("active work"),
        ])
        .expect("history");
    agent.runtime_snapshot = runtime_snapshot_for_history("main", &agent.history);
    agent.turn.turn_id = 10;
    agent.turn.current_turn_start_index = Some(2);
    agent.runtime_snapshot.current_turn_id = Some(10);
    agent.runtime_snapshot.current_segment_id = Some(3);

    let outcome = agent
        .compact_session_async(|_| std::future::ready(Ok(())))
        .await
        .expect("compacts older turns");

    assert!(matches!(outcome, ManualCompactionOutcome::Compacted { .. }));
    assert_eq!(agent.current_turn_id(), 10);
    assert_eq!(agent.runtime_snapshot.current_turn_id, Some(10));
    assert_eq!(agent.runtime_snapshot.current_segment_id, Some(3));
    assert_eq!(agent.turn.current_turn_start_index, None);
    let history = agent.history_for_test();
    assert_eq!(history.len(), 1);
    assert!(matches!(history[0], HistoryItem::ContextSummary { .. }));
    server.await.expect("server task should finish");
}
#[tokio::test]
async fn active_turn_compaction_callback_failure_is_atomic_after_identity_rebase() {
    let checkpoint = valid_checkpoint("continue validating the active request");
    let (base_url, _requests, server) =
        spawn_chat_completion_server(vec![chat_final_sse(&checkpoint)]).await;
    let mut agent = Agent::new(
        Client::with_config(
            OpenAIConfig::new()
                .with_api_base(base_url)
                .with_api_key("test"),
        ),
        "m1",
        8,
        8,
    );
    agent.set_default_protocol(ApiProtocol::Completions);
    let history = vec![
        HistoryItem::user("older turn"),
        HistoryItem::assistant("older answer"),
        HistoryItem::user("active request"),
        HistoryItem::AssistantToolCalls {
            text: None,
            reasoning_content: None,
            calls: vec![HistoryToolCall {
                call_id: "call-pending".into(),
                name: "fs__read".into(),
                arguments_json: r#"{"path":"src/main.rs"}"#.into(),
            }],
        },
    ];
    agent
        .replace_history(history.clone())
        .expect("active incomplete history");
    agent.runtime_snapshot = runtime_snapshot_for_history("main", &agent.history);
    agent.turn.turn_id = 4;
    agent.turn.current_turn_start_index = Some(2);
    agent.runtime_snapshot.current_turn_id = Some(4);
    agent.runtime_snapshot.current_segment_id = Some(1);
    let before_history = agent.history_for_test().to_vec();
    let before_turn_start = agent.turn.current_turn_start_index;

    let error = agent
        .compact_session_async(|event| {
            if matches!(event, AgentEvent::ContextCompacted(_)) {
                return std::future::ready(Err(anyhow::anyhow!("journal rejected compact")));
            }
            std::future::ready(Ok(()))
        })
        .await
        .expect_err("callback failure should abort install");

    assert!(error.to_string().contains("journal rejected compact"));
    assert_eq!(agent.history_for_test(), before_history.as_slice());
    assert_eq!(agent.turn.current_turn_start_index, before_turn_start);
    assert_eq!(agent.current_turn_id(), 4);
    assert_eq!(agent.runtime_snapshot.current_turn_id, Some(4));
    assert_eq!(agent.runtime_snapshot.current_segment_id, Some(1));
    server.await.expect("server task should finish");
}
#[tokio::test]
async fn manual_compaction_co_retires_ordinary_context_and_keeps_retaining_context() {
    // History-first compact cuts older turns only; co-retire of context materials
    // is no longer part of the compact selection model.
    let checkpoint = valid_checkpoint("continue validating the active request");
    let (base_url, _requests, server) =
        spawn_chat_completion_server(vec![chat_final_sse(&checkpoint)]).await;
    let mut agent = Agent::new(
        Client::with_config(
            OpenAIConfig::new()
                .with_api_base(base_url)
                .with_api_key("test"),
        ),
        "m1",
        8,
        8,
    );
    agent.set_default_protocol(ApiProtocol::Completions);
    agent
        .replace_history(vec![
            HistoryItem::user("older"),
            HistoryItem::assistant("older answer"),
            HistoryItem::user("current"),
            HistoryItem::assistant("current answer"),
        ])
        .expect("history");
    agent.runtime_snapshot = runtime_snapshot_for_history("main", &agent.history);
    agent.turn.current_turn_start_index = Some(2);

    let outcome = agent
        .compact_session_async(|_| std::future::ready(Ok(())))
        .await
        .expect("compacts older turns");
    assert!(matches!(outcome, ManualCompactionOutcome::Compacted { .. }));
    let history = agent.history_for_test();
    assert_eq!(history.len(), 1);
    assert!(matches!(history[0], HistoryItem::ContextSummary { .. }));
    assert_eq!(agent.turn.current_turn_start_index, None);
    server.await.expect("server task should finish");
}

#[tokio::test]
async fn failed_manual_compaction_returns_its_error_without_a_stream_issue() {
    let (base_url, _requests, server) =
        spawn_chat_completion_server(vec![sse_response("data: [DONE]\n\n".into())]).await;
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base(base_url)
            .with_api_key("test"),
    );
    let mut agent = Agent::new(client, "m1", 4, 4);
    agent.set_default_protocol(ApiProtocol::Completions);
    agent
        .replace_history(vec![
            HistoryItem::user("short prompt"),
            HistoryItem::assistant("reply"),
        ])
        .expect("history replace succeeds");
    agent.runtime_snapshot = runtime_snapshot_for_history("main", &agent.history);
    let mut events = Vec::new();

    let error = agent
        .compact_session_async(|event| {
            events.push(event);
            std::future::ready(Ok(()))
        })
        .await
        .expect_err("empty compaction summary fails");

    assert!(
        !error.to_string().is_empty(),
        "the original compaction error remains authoritative"
    );
    assert!(matches!(
        events.as_slice(),
        [
            AgentEvent::ContextCompactionStarted {
                trigger: CompactionTrigger::Manual
            },
            AgentEvent::ContextCompactionFailed {
                trigger: CompactionTrigger::Manual
            },
        ]
    ));
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, AgentEvent::ContextCompacted(_))),
        "failed compaction must not produce a durable success event"
    );
    server.await.expect("summary server completes");
}

#[test]
fn compaction_default_token_tail_uses_active_model_budget() {
    assert_eq!(CompactionConfig::default().preserve_recent_tokens, None);
}

#[test]
fn default_preserve_recent_budget_uses_reserve_aware_pi_style_20k_tail() {
    assert_eq!(default_preserve_recent_budget(1_000), 1_000);
    assert_eq!(default_preserve_recent_budget(20_000), 3_616);
    assert_eq!(default_preserve_recent_budget(100_000), 20_000);
}

#[test]
fn compaction_history_char_budget_scales_with_model_window() {
    let small = compaction_history_char_budget(ModelRequestMetadata {
        context_window: Some(1_024),
        max_output_tokens: Some(128),
        ..ModelRequestMetadata::default()
    });
    let large = compaction_history_char_budget(ModelRequestMetadata {
        context_window: Some(128_000),
        max_output_tokens: Some(4_096),
        ..ModelRequestMetadata::default()
    });

    assert!(small <= 1_000);
    assert!(large > small);
    assert!(large <= COMPACTION_HISTORY_MAX_CHAR_BUDGET);
}

#[test]
fn compaction_history_char_budget_uses_effective_input_limit() {
    let uncapped = compaction_history_char_budget(ModelRequestMetadata {
        context_window: Some(128_000),
        max_output_tokens: Some(4_096),
        ..ModelRequestMetadata::default()
    });
    let capped = compaction_history_char_budget(ModelRequestMetadata {
        context_window: Some(128_000),
        effective_input_limit_tokens: Some(4_000),
        max_output_tokens: Some(4_096),
        ..ModelRequestMetadata::default()
    });

    assert!(capped < uncapped);
    assert!(capped <= 4_000);
}

#[test]
fn render_compaction_prompt_distinguishes_initial_and_incremental_modes() {
    let items = vec![HistoryItem::user("修复 src/agent.rs")];

    let initial = render_compaction_prompt(None, &items, 16_000);
    assert!(initial.contains("生成新的执行检查点"));
    assert!(!initial.contains("更新已有执行检查点"));

    let incremental = render_compaction_prompt(Some("已有执行检查点"), &items, 16_000);
    assert!(incremental.contains("更新已有执行检查点"));
    assert!(incremental.contains("删除过时或被推翻的信息"));
}

#[test]
fn render_compaction_tool_output_caps_large_payloads() {
    let rendered = compaction::describe_history_item(&HistoryItem::ToolOutput {
        call_id: "call-big".into(),
        output_json: large_tool_output_json("stdout"),
    });

    assert!(rendered.contains(COMPACTION_TOOL_OUTPUT_TRUNCATION_MARKER));
    assert!(rendered.chars().count() < 2_200);
}

#[test]
fn render_compaction_tool_output_strips_media_like_fields() {
    let base64 = "A".repeat(3_000);
    let rendered = compaction::describe_history_item(&HistoryItem::ToolOutput {
        call_id: "call-media".into(),
        output_json: json!({
            "image_base64": base64,
            "preview_url": "blob:https://example.invalid/123",
            "stdout": "kept text"
        })
        .to_string(),
    });

    assert!(rendered.contains("stripped media/blob-like field"));
    assert!(rendered.contains("kept text"));
    assert!(!rendered.contains("blob:https://example.invalid/123"));
    assert!(!rendered.contains(&"A".repeat(128)));
}

#[test]
fn render_compaction_prompt_applies_total_history_cap() {
    let items = (0..20)
        .map(|index| HistoryItem::ToolOutput {
            call_id: format!("call-{index}"),
            output_json: large_tool_output_json("stdout"),
        })
        .collect::<Vec<_>>();

    let rendered = render_bounded_compaction_history(&items, 4_000);

    assert!(rendered.contains(COMPACTION_HISTORY_TRUNCATION_MARKER));
    assert!(rendered.chars().count() <= 4_000);
    assert!(rendered.contains("call-19"));
    assert!(!rendered.contains("call-0"));
}

#[test]
fn setting_reasoning_effort_does_not_make_fast_mode_sticky_in_the_catalog() {
    let fast_mode_dir = agents_test_dir();
    let fast_mode_path = fast_mode_dir.join("letcode.toml");
    std::fs::write(
        &fast_mode_path,
        r#"active_provider = "primary"

[providers.primary]
base_url = "https://primary.invalid/v1"
api_key = "primary-key"
protocol = "responses"
[providers.primary.models."gpt-5.5"]
"#,
    )
    .expect("write Fast Mode config");
    let fast_mode = crate::fast_mode::FastMode::load(fast_mode_path, true);

    let mut agent = test_agent();
    agent.set_model("gpt-5.5");
    agent.set_model_catalog(HashMap::from([(
        "gpt-5.5".into(),
        ModelRequestMetadata {
            supports_reasoning: true,
            ..Default::default()
        },
    )]));
    agent.set_fast_mode(Arc::clone(&fast_mode));

    agent
        .set_reasoning_effort(ModelReasoningEffort::Low)
        .expect("set reasoning effort");
    assert!(agent.active_model_metadata().fast_mode);
    assert!(
        !agent.model_catalog["gpt-5.5"].fast_mode,
        "runtime Fast Mode must not be persisted in the model catalog"
    );

    assert!(matches!(
        fast_mode.toggle("gpt-5.5").expect("disable fast mode"),
        crate::fast_mode::FastModeToggle::Disabled
    ));
    assert!(!agent.active_model_metadata().fast_mode);
}

#[tokio::test]
async fn request_preparation_auto_disables_fast_mode_and_emits_projection() {
    let fast_mode_dir = agents_test_dir();
    let fast_mode_path = fast_mode_dir.join("letcode.toml");
    std::fs::write(
        &fast_mode_path,
        r#"active_provider = "primary"

[providers.primary]
base_url = "https://primary.invalid/v1"
api_key = "primary-key"
protocol = "responses"
[providers.primary.models."gpt-5.5"]
"#,
    )
    .expect("write Fast Mode config");
    let fast_mode = crate::fast_mode::FastMode::load(fast_mode_path, true);

    let mut agent = test_agent();
    agent.set_fast_mode(Arc::clone(&fast_mode));
    let mut events = Vec::new();
    let mut on_event = |event| {
        events.push(event);
        std::future::ready(Ok(()))
    };

    let prepared = compaction::prepare_request_build(
        &mut agent,
        ApiProtocol::Responses,
        &[],
        0,
        &[],
        &mut on_event,
    )
    .await
    .expect("request preparation succeeds");

    assert!(matches!(
        events.as_slice(),
        [AgentEvent::FastModeChanged { enabled: false }]
    ));
    assert!(!agent.fast_mode_enabled());
    assert!(matches!(
        prepared.build.request,
        BuiltRequest::Responses(ref request)
            if serde_json::to_value(request)
                .expect("serialize typed request")
                .get("service_tier")
                .is_none()
    ));
    assert!(!fast_mode.enabled(), "auto-disable must persist");
}

#[tokio::test]
async fn ordinary_request_build_uses_installed_runtime_snapshot_only() {
    let mut agent = test_agent();
    agent.history = vec![HistoryItem::user("EXTERNAL-TRANSCRIPT-CONTENT")];
    agent.runtime_snapshot = runtime_snapshot_for_history(
        ROOT_CONTEXT_BRANCH_ID,
        &[HistoryItem::user("INSTALLED-RUNTIME-SNAPSHOT-CONTENT")],
    );
    let mut on_event = |_| std::future::ready(Ok(()));

    let prepared = compaction::prepare_request_build(
        &mut agent,
        ApiProtocol::Responses,
        &[],
        0,
        &[],
        &mut on_event,
    )
    .await
    .expect("ordinary request builds from the installed runtime snapshot");

    let request = match prepared.build.request {
        BuiltRequest::Responses(request) => serde_json::to_value(request),
        BuiltRequest::ResponsesCompatible(request) => Ok(request),
        BuiltRequest::Completions(_) | BuiltRequest::CompletionsCompatible(_) => {
            panic!("expected responses request")
        }
    };
    let request = request.expect("request serializes");
    let json = serde_json::to_string(&request).expect("request serializes");
    assert!(json.contains("INSTALLED-RUNTIME-SNAPSHOT-CONTENT"));
    assert!(!json.contains("EXTERNAL-TRANSCRIPT-CONTENT"));
}

#[tokio::test]
async fn chat_stream_creation_failure_includes_request_budget_diagnostic() {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("test port should bind");
    let addr = listener.local_addr().expect("test listener has local addr");
    drop(listener);
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base(format!("http://{addr}"))
            .with_api_key("test"),
    );
    let mut agent = Agent::new(client, "m1", 4, 4);
    agent.set_model_catalog(HashMap::from([(
        "m1".into(),
        ModelRequestMetadata {
            context_window: Some(32_000),
            effective_input_limit_tokens: Some(4_096),
            max_output_tokens: Some(2_000),
            supports_tools: false,
            supports_reasoning: false,
            ..Default::default()
        },
    )]));
    let mut retry_config = test_retry_config();
    retry_config.enabled = false;
    agent.set_retry_config(retry_config);

    let error = agent
        .run_oai_comp_stream_async(
            "hello",
            |_| std::future::ready(Ok(())),
            |_| std::future::ready(Ok(())),
            |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await
        .expect_err("stream creation should fail");
    let message = format!("{error:#}");

    assert!(
        message.contains("failed to create streamed chat completion"),
        "{message}"
    );
    assert!(message.contains("model=m1"), "{message}");
    assert!(message.contains("estimated_request_tokens="), "{message}");
    assert!(message.contains("input_budget_tokens=4096"), "{message}");
    assert!(
        message.contains("effective_input_limit_tokens=4096"),
        "{message}"
    );
    assert!(message.contains("protected_tokens="), "{message}");
}

#[tokio::test]
async fn compatible_chat_stream_sends_one_physical_request() {
    let (base_url, request_count, server) = spawn_chat_completion_server(vec![
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: 14\r\nConnection: close\r\n\r\ndata: [DONE]\n\n",
        ])
        .await;
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base(base_url)
            .with_api_key("test"),
    );
    let request = json!({"model": "m1", "stream": true, "messages": []});
    let response = send_compatible_chat_completion_stream(&client, &request)
        .await
        .expect("request should succeed");

    assert_eq!(response.status(), reqwest::StatusCode::OK);
    assert_eq!(request_count.load(Ordering::SeqCst), 1);
    server.await.expect("server task should finish");
}

#[tokio::test]
async fn responses_completion_over_tool_call_budget_emits_completed_telemetry() {
    let body = r#"data: {"type":"response.completed","sequence_number":1,"response":{"id":"resp-over-budget","object":"response","created_at":1780856440,"status":"completed","background":false,"error":null,"incomplete_details":null,"instructions":null,"max_output_tokens":null,"model":"m1","output":[{"type":"function_call","id":"fc-1","call_id":"call-1","name":"first","arguments":"{}","status":"completed"},{"type":"function_call","id":"fc-2","call_id":"call-2","name":"second","arguments":"{}","status":"completed"}],"parallel_tool_calls":true,"previous_response_id":null,"reasoning":{},"store":true,"temperature":1,"text":{"format":{"type":"text"}},"tool_choice":"auto","tools":[],"top_p":1,"truncation":"disabled","usage":{"input_tokens":5,"input_tokens_details":{"cached_tokens":0},"output_tokens":3,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":8},"user":null,"metadata":{}}}

data: [DONE]

"#;
    let response = Box::leak(
            format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            )
            .into_boxed_str(),
        );
    let (base_url, request_count, server) = spawn_chat_completion_server(vec![response]).await;
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base(base_url)
            .with_api_key("test"),
    );
    let mut agent = Agent::new(client, "m1", 4, 1);
    agent.set_model_catalog(HashMap::from([(
        "m1".into(),
        ModelRequestMetadata {
            context_window: Some(32_000),
            max_output_tokens: Some(2_000),
            supports_tools: true,
            supports_reasoning: false,
            ..Default::default()
        },
    )]));
    let mut audit_telemetry = Vec::new();

    let error = agent
        .run_stream_async(
            "hello",
            |_| std::future::ready(Ok(())),
            |event| {
                if let AgentEvent::LlmRequestTelemetry(telemetry) = event {
                    audit_telemetry.push(telemetry);
                }
                std::future::ready(Ok(()))
            },
            |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await
        .expect_err("over-budget tool calls should fail locally");

    assert!(error.to_string().contains("too many tool calls"));
    assert_eq!(
        audit_telemetry
            .iter()
            .map(|telemetry| telemetry.phase)
            .collect::<Vec<_>>(),
        vec![
            LlmRequestTelemetryPhase::Prepared,
            LlmRequestTelemetryPhase::Completed,
        ]
    );
    let completed = audit_telemetry
        .iter()
        .find(|telemetry| telemetry.phase == LlmRequestTelemetryPhase::Completed)
        .expect("provider completion telemetry");
    assert_eq!(
        completed.provider_response_id.as_deref(),
        Some("resp-over-budget")
    );
    assert_eq!(
        completed.usage.as_ref().map(|usage| usage.used_tokens),
        Some(8)
    );
    assert_request_telemetry_is_terminal_once(&audit_telemetry);
    assert_eq!(request_count.load(Ordering::SeqCst), 1);
    server.await.expect("server task should finish");
}

#[tokio::test]
async fn responses_stream_recovers_from_malformed_event_after_visible_delta() {
    let first_body = r#"data: {"type":"response.output_text.delta","sequence_number":1,"item_id":"msg-1","output_index":0,"content_index":0,"delta":"partial "}

data: {"type":"response.completed","response":{"reasoning":{"effort":"max"}}}

"#;
    let second_body = r#"data: {"type":"response.completed","sequence_number":1,"response":{"id":"resp-recovered","object":"response","created_at":1780856440,"status":"completed","background":false,"error":null,"incomplete_details":null,"instructions":null,"max_output_tokens":null,"model":"m1","output":[{"type":"message","id":"msg-2","status":"completed","role":"assistant","content":[{"type":"output_text","text":"continued","annotations":[]}]}],"parallel_tool_calls":true,"previous_response_id":null,"reasoning":{},"store":true,"temperature":1,"text":{"format":{"type":"text"}},"tool_choice":"auto","tools":[],"top_p":1,"truncation":"disabled","usage":{"input_tokens":5,"input_tokens_details":{"cached_tokens":0},"output_tokens":3,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":8},"user":null,"metadata":{}}}

data: [DONE]

"#;
    let first_response = Box::leak(
            format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{first_body}",
                first_body.len()
            )
            .into_boxed_str(),
        );
    let second_response = Box::leak(
            format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{second_body}",
                second_body.len()
            )
            .into_boxed_str(),
        );
    let (base_url, request_count, server) =
        spawn_chat_completion_server(vec![first_response, second_response]).await;
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base(base_url)
            .with_api_key("test"),
    );
    let mut agent = Agent::new(client, "m1", 4, 4);
    agent.set_model_catalog(HashMap::from([(
        "m1".into(),
        ModelRequestMetadata {
            context_window: Some(32_000),
            max_output_tokens: Some(2_000),
            supports_tools: false,
            supports_reasoning: false,
            ..Default::default()
        },
    )]));
    agent.set_retry_config(test_retry_config());
    let mut deltas = Vec::new();
    let mut stream_issues = Vec::new();
    let mut audit_telemetry = Vec::new();

    let result = agent
        .run_stream_async(
            "hello",
            |delta| {
                deltas.push(delta.to_string());
                std::future::ready(Ok(()))
            },
            |event| {
                match event {
                    AgentEvent::ModelStreamIssue { message, .. } => stream_issues.push(message),
                    AgentEvent::LlmRequestTelemetry(telemetry) => audit_telemetry.push(telemetry),
                    _ => {}
                }
                std::future::ready(Ok(()))
            },
            |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await
        .expect("malformed event after output should continue with a fresh iteration");

    assert_eq!(result, "partial continued");
    assert_eq!(deltas, vec!["partial "]);
    assert_eq!(stream_issues, vec!["Model stream interrupted"]);
    assert_eq!(
        audit_telemetry
            .iter()
            .map(|telemetry| (telemetry.phase, telemetry.error_class))
            .collect::<Vec<_>>(),
        vec![
            (LlmRequestTelemetryPhase::Prepared, None),
            (
                LlmRequestTelemetryPhase::Interrupted,
                Some(LlmRequestErrorClass::ProtocolValidation),
            ),
            (LlmRequestTelemetryPhase::Prepared, None),
            (LlmRequestTelemetryPhase::Completed, None),
        ]
    );
    assert_request_telemetry_is_terminal_once(&audit_telemetry);
    assert_eq!(request_count.load(Ordering::SeqCst), 2);
    server.await.expect("server task should finish");
}

#[tokio::test]
async fn responses_stream_retries_transient_response_failed_before_side_effects() {
    let (base_url, request_count, server) = spawn_chat_completion_server(vec![
        responses_terminal_sse(
            "response.failed",
            "failed",
            Some(json!({"code": "server_error", "message": "temporary upstream failure"})),
            None,
        ),
        responses_final_sse("recovered"),
    ])
    .await;
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base(base_url)
            .with_api_key("test"),
    );
    let mut agent = Agent::new(client, "m1", 4, 4);
    agent.set_retry_config(test_retry_config());
    let mut lifecycle = Vec::new();

    let result = agent
        .run_stream_async(
            "hello",
            |_| std::future::ready(Ok(())),
            |event| {
                if matches!(
                    event,
                    AgentEvent::LlmRetryScheduled(_) | AgentEvent::LlmRetryStarted(_)
                ) {
                    lifecycle.push(event);
                }
                std::future::ready(Ok(()))
            },
            |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await
        .expect("transient response.failed should retry");

    assert_eq!(result, "recovered");
    assert!(matches!(lifecycle.as_slice(), [
        AgentEvent::LlmRetryScheduled(retry),
        AgentEvent::LlmRetryStarted(started)
    ] if retry.attempt == 2 && retry.delay_ms == 1 && retry == started));
    assert_eq!(request_count.load(Ordering::SeqCst), 2);
    server.await.expect("server task should finish");
}

#[tokio::test]
async fn responses_stream_fails_fast_for_non_retryable_response_failed() {
    let (base_url, request_count, server) =
        spawn_chat_completion_server(vec![responses_terminal_sse(
            "response.failed",
            "failed",
            Some(json!({"code": "invalid_request", "message": "invalid input"})),
            None,
        )])
        .await;
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base(base_url)
            .with_api_key("test"),
    );
    let mut agent = Agent::new(client, "m1", 4, 4);
    agent.set_retry_config(test_retry_config());

    let error = agent
        .run_stream_async(
            "hello",
            |_| std::future::ready(Ok(())),
            |_| std::future::ready(Ok(())),
            |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await
        .expect_err("deterministic response.failed must fail fast");

    assert!(error.to_string().contains("code=invalid_request"));
    assert_eq!(request_count.load(Ordering::SeqCst), 1);
    server.await.expect("server task should finish");
}

#[tokio::test]
async fn responses_stream_fails_fast_when_deterministic_code_has_transient_message() {
    let (base_url, request_count, server) =
        spawn_chat_completion_server(vec![responses_terminal_sse(
            "response.failed",
            "failed",
            Some(json!({
                "code": "invalid_request",
                "message": "temporary upstream connection failure"
            })),
            None,
        )])
        .await;
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base(base_url)
            .with_api_key("test"),
    );
    let mut agent = Agent::new(client, "m1", 4, 4);
    agent.set_retry_config(test_retry_config());

    let error = agent
        .run_stream_async(
            "hello",
            |_| std::future::ready(Ok(())),
            |_| std::future::ready(Ok(())),
            |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await
        .expect_err("structured deterministic provider codes must take precedence");

    assert!(error.to_string().contains("code=invalid_request"));
    assert_eq!(request_count.load(Ordering::SeqCst), 1);
    server.await.expect("server task should finish");
}

#[tokio::test]
async fn responses_stream_retries_transient_response_error_before_side_effects() {
    let (base_url, request_count, server) = spawn_chat_completion_server(vec![
        response_error_sse(Some("server_error"), "temporary upstream failure"),
        responses_final_sse("recovered"),
    ])
    .await;
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base(base_url)
            .with_api_key("test"),
    );
    let mut agent = Agent::new(client, "m1", 4, 4);
    agent.set_retry_config(test_retry_config());

    let result = agent
        .run_stream_async(
            "hello",
            |_| std::future::ready(Ok(())),
            |_| std::future::ready(Ok(())),
            |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await
        .expect("transient response.error should retry");

    assert_eq!(result, "recovered");
    assert_eq!(request_count.load(Ordering::SeqCst), 2);
    server.await.expect("server task should finish");
}

#[tokio::test]
async fn responses_stream_recovers_from_response_error_after_visible_output() {
    let interrupted = sse_response(
        "data: {\"type\":\"response.output_text.delta\",\"sequence_number\":1,\"item_id\":\"msg-1\",\"output_index\":0,\"content_index\":0,\"delta\":\"partial \"}\n\ndata: {\"type\":\"error\",\"sequence_number\":2,\"code\":\"server_error\",\"message\":\"temporary upstream failure\"}\n\ndata: [DONE]\n\n"
            .into(),
    );
    let (base_url, request_count, server) =
        spawn_chat_completion_server(vec![interrupted, responses_final_sse("continued")]).await;
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base(base_url)
            .with_api_key("test"),
    );
    let mut agent = Agent::new(client, "m1", 4, 4);
    agent.set_retry_config(test_retry_config());
    let mut deltas = Vec::new();
    let mut issues = Vec::new();

    let result = agent
        .run_stream_async(
            "hello",
            |delta| {
                deltas.push(delta.to_owned());
                std::future::ready(Ok(()))
            },
            |event| {
                if let AgentEvent::ModelStreamIssue { message, .. } = event {
                    issues.push(message);
                }
                std::future::ready(Ok(()))
            },
            |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await
        .expect("response.error after output should recover semantically");

    assert_eq!(result, "partial continued");
    assert_eq!(deltas, vec!["partial "]);
    assert_eq!(issues, vec!["Model stream interrupted"]);
    assert_eq!(request_count.load(Ordering::SeqCst), 2);
    server.await.expect("server task should finish");
}

#[tokio::test]
async fn responses_stream_does_not_semantically_recover_hard_failure_after_visible_output() {
    let failed = sse_response(
        "data: {\"type\":\"response.output_text.delta\",\"sequence_number\":1,\"item_id\":\"msg-1\",\"output_index\":0,\"content_index\":0,\"delta\":\"partial \"}\n\ndata: {\"type\":\"response.failed\",\"sequence_number\":2,\"response\":{\"id\":\"r-hard\",\"object\":\"response\",\"created_at\":1,\"status\":\"failed\",\"background\":false,\"error\":{\"code\":\"invalid_request\",\"message\":\"temporary upstream connection failure\"},\"incomplete_details\":null,\"instructions\":null,\"max_output_tokens\":null,\"model\":\"m1\",\"output\":[],\"parallel_tool_calls\":true,\"previous_response_id\":null,\"reasoning\":{},\"store\":true,\"temperature\":1,\"text\":{\"format\":{\"type\":\"text\"}},\"tool_choice\":\"auto\",\"tools\":[],\"top_p\":1,\"truncation\":\"disabled\",\"usage\":null,\"user\":null,\"metadata\":{}}}\n\ndata: [DONE]\n\n"
            .into(),
    );
    let (base_url, request_count, server) = spawn_chat_completion_server(vec![failed]).await;
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base(base_url)
            .with_api_key("test"),
    );
    let mut agent = Agent::new(client, "m1", 4, 4);
    agent.set_retry_config(test_retry_config());
    let mut issues = Vec::new();

    let error = agent
        .run_stream_async(
            "hello",
            |_| std::future::ready(Ok(())),
            |event| {
                if let AgentEvent::ModelStreamIssue { message, .. } = event {
                    issues.push(message);
                }
                std::future::ready(Ok(()))
            },
            |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await
        .expect_err("hard terminal failures after output must not start semantic continuation");

    assert!(error.to_string().contains("code=invalid_request"));
    assert!(issues.is_empty());
    assert_eq!(request_count.load(Ordering::SeqCst), 1);
    server.await.expect("server task should finish");
}

#[tokio::test]
async fn responses_incomplete_retries_when_its_reason_is_transient_independently_of_error() {
    let (base_url, request_count, server) = spawn_chat_completion_server(vec![
        responses_terminal_sse(
            "response.incomplete",
            "incomplete",
            Some(json!({"code": "invalid_request", "message": "invalid input"})),
            Some(json!({"reason": "temporarily_unavailable"})),
        ),
        responses_final_sse("recovered"),
    ])
    .await;
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base(base_url)
            .with_api_key("test"),
    );
    let mut agent = Agent::new(client, "m1", 4, 4);
    agent.set_retry_config(test_retry_config());

    let result = agent
        .run_stream_async(
            "hello",
            |_| std::future::ready(Ok(())),
            |_| std::future::ready(Ok(())),
            |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await
        .expect("a transient incomplete reason should independently permit retry");

    assert_eq!(result, "recovered");
    assert_eq!(request_count.load(Ordering::SeqCst), 2);
    server.await.expect("server task should finish");
}

#[tokio::test]
async fn responses_completed_seals_the_stream_against_trailing_malformed_data() {
    let completed = responses_final_sse("sealed");
    let completed = completed.replacen("data: [DONE]", "data: {malformed}\n\ndata: [DONE]", 1);
    let response = Box::leak(completed.into_boxed_str());
    let (base_url, request_count, server) = spawn_chat_completion_server(vec![response]).await;
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base(base_url)
            .with_api_key("test"),
    );
    let mut agent = Agent::new(client, "m1", 4, 4);
    agent.set_retry_config(test_retry_config());

    let result = agent
        .run_stream_async(
            "hello",
            |_| std::future::ready(Ok(())),
            |_| std::future::ready(Ok(())),
            |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await
        .expect("response.completed must seal the stream");

    assert_eq!(result, "sealed");
    assert_eq!(request_count.load(Ordering::SeqCst), 1);
    server.await.expect("server task should finish");
}

#[tokio::test]
async fn responses_stream_exhausts_semantic_recovery_budget_across_iterations() {
    let interrupted = sse_response(
        "data: {\"type\":\"response.output_text.delta\",\"sequence_number\":1,\"item_id\":\"msg-1\",\"output_index\":0,\"content_index\":0,\"delta\":\"partial\"}\n\ndata: {malformed}\n\n"
            .into(),
    );
    let (base_url, request_count, server) =
        spawn_chat_completion_server(vec![interrupted, interrupted]).await;
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base(base_url)
            .with_api_key("test"),
    );
    let mut agent = Agent::new(client, "m1", 4, 4);
    let mut retry = test_retry_config();
    retry.max_recovery_attempts = 1;
    agent.set_retry_config(retry);

    let error = agent
        .run_stream_async(
            "hello",
            |_| std::future::ready(Ok(())),
            |_| std::future::ready(Ok(())),
            |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await
        .expect_err("second semantic recovery must exhaust the turn budget");

    assert!(
        error
            .to_string()
            .contains("stream recovery budget exhausted after 1 attempts")
    );
    assert_eq!(request_count.load(Ordering::SeqCst), 2);
    server.await.expect("server task should finish");
}

#[tokio::test]
async fn compatible_chat_stream_propagates_http_retry_after_delay() {
    let (base_url, request_count, server) = spawn_chat_completion_server(vec![
        "HTTP/1.1 429 Too Many Requests\r\nRetry-After: 0\r\nContent-Length: 9\r\nConnection: close\r\n\r\ntransient",
        chat_final_sse("ok"),
    ])
    .await;
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base(base_url)
            .with_api_key("test"),
    );
    let mut agent = Agent::new(client, "m1", 4, 4);
    let mut retry = test_retry_config();
    retry.initial_delay_ms = 1;
    agent.set_retry_config(retry);
    let mut scheduled = Vec::new();

    let result = agent
        .run_oai_comp_stream_async(
            "hello",
            |_| std::future::ready(Ok(())),
            |event| {
                if let AgentEvent::LlmRetryScheduled(retry) = event {
                    scheduled.push(retry);
                }
                std::future::ready(Ok(()))
            },
            |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await
        .expect("429 should retry");

    assert_eq!(result, "ok");
    assert_eq!(scheduled.len(), 1);
    assert_eq!(scheduled[0].delay_ms, 1);
    assert_eq!(request_count.load(Ordering::SeqCst), 2);
    server.await.expect("server task should finish");
}

#[tokio::test]
async fn compatible_chat_stream_retries_read_error_before_visible_output() {
    let body = r#"data: {"choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}

data: [DONE]

"#;
    let first_response = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: 100\r\nConnection: close\r\n\r\n";
    let second_response = Box::leak(
            format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            )
            .into_boxed_str(),
        );
    let (base_url, request_count, server) =
        spawn_chat_completion_server(vec![first_response, second_response]).await;
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base(base_url)
            .with_api_key("test"),
    );
    let mut agent = Agent::new(client, "m1", 4, 4);
    agent.set_model_catalog(HashMap::from([(
        "m1".into(),
        ModelRequestMetadata {
            context_window: Some(32_000),
            max_output_tokens: Some(2_000),
            supports_tools: true,
            supports_reasoning: false,
            ..Default::default()
        },
    )]));
    agent.set_retry_config(test_retry_config());
    let mut deltas = Vec::new();
    let mut audit_telemetry = Vec::new();

    let result = agent
        .run_oai_comp_stream_async(
            "hello",
            |delta| {
                deltas.push(delta.to_string());
                std::future::ready(Ok(()))
            },
            |event| {
                if let AgentEvent::LlmRequestTelemetry(telemetry) = event {
                    audit_telemetry.push(telemetry);
                }
                std::future::ready(Ok(()))
            },
            |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await
        .expect("pre-output stream read failure should retry");

    assert_eq!(result, "ok");
    assert_eq!(deltas, vec!["ok"]);
    assert_eq!(
        audit_telemetry
            .iter()
            .map(|telemetry| (telemetry.phase, telemetry.attempt))
            .collect::<Vec<_>>(),
        vec![
            (LlmRequestTelemetryPhase::Prepared, 1),
            (LlmRequestTelemetryPhase::Failed, 1),
            (LlmRequestTelemetryPhase::Prepared, 2),
            (LlmRequestTelemetryPhase::Completed, 2),
        ]
    );
    assert_request_telemetry_is_terminal_once(&audit_telemetry);
    assert_eq!(request_count.load(Ordering::SeqCst), 2);
    server.await.expect("server task should finish");
}

#[tokio::test]
async fn compatible_chat_stream_continues_after_streamed_usage_event() {
    let first_body = r#"data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":0,"total_tokens":5}}

"#;
    let second_body = r#"data: {"choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}

data: [DONE]

"#;
    let first_response = Box::leak(
            format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: 1000\r\nConnection: close\r\n\r\n{first_body}"
            )
            .into_boxed_str(),
        );
    let second_response = Box::leak(
            format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{second_body}",
                second_body.len()
            )
            .into_boxed_str(),
        );
    let (base_url, request_count, server) =
        spawn_chat_completion_server(vec![first_response, second_response]).await;
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base(base_url)
            .with_api_key("test"),
    );
    let mut agent = Agent::new(client, "m1", 4, 4);
    agent.set_model_catalog(HashMap::from([(
        "m1".into(),
        ModelRequestMetadata {
            context_window: Some(32_000),
            max_output_tokens: Some(2_000),
            supports_tools: true,
            supports_reasoning: false,
            ..Default::default()
        },
    )]));
    agent.set_retry_config(test_retry_config());
    let mut deltas = Vec::new();
    let mut usage_events = Vec::new();
    let mut stream_issues = Vec::new();

    let result = agent
        .run_oai_comp_stream_async(
            "hello",
            |delta| {
                deltas.push(delta.to_string());
                std::future::ready(Ok(()))
            },
            |event| {
                match event {
                    AgentEvent::TokenUsageUpdated { used_tokens, .. } => {
                        usage_events.push(used_tokens);
                    }
                    AgentEvent::ModelStreamIssue { message, .. } => {
                        stream_issues.push(message);
                    }
                    _ => {}
                }
                std::future::ready(Ok(()))
            },
            |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await
        .expect("stream read failure after usage event should continue with a fresh iteration");

    assert_eq!(result, "ok");
    assert_eq!(deltas, vec!["ok"]);
    assert_eq!(stream_issues, vec!["Model stream interrupted"]);
    assert!(
        usage_events.contains(&5),
        "missing streamed usage event: {usage_events:?}"
    );
    assert_eq!(request_count.load(Ordering::SeqCst), 2);
    server.await.expect("server task should finish");
}

#[tokio::test]
async fn compatible_chat_stream_retries_incomplete_json_event_before_visible_output() {
    let first_body = "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"partial\"}\n\n";
    let second_body = r#"data: {"choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}

data: [DONE]

"#;
    let first_response = Box::leak(
            format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n{first_body}"
            )
            .into_boxed_str(),
        );
    let second_response = Box::leak(
            format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{second_body}",
                second_body.len()
            )
            .into_boxed_str(),
        );
    let (base_url, request_count, server) =
        spawn_chat_completion_server(vec![first_response, second_response]).await;
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base(base_url)
            .with_api_key("test"),
    );
    let mut agent = Agent::new(client, "m1", 4, 4);
    agent.set_model_catalog(HashMap::from([(
        "m1".into(),
        ModelRequestMetadata {
            context_window: Some(32_000),
            max_output_tokens: Some(2_000),
            supports_tools: true,
            supports_reasoning: false,
            ..Default::default()
        },
    )]));
    agent.set_retry_config(test_retry_config());
    let mut deltas = Vec::new();

    let result = agent
        .run_oai_comp_stream_async(
            "hello",
            |delta| {
                deltas.push(delta.to_string());
                std::future::ready(Ok(()))
            },
            |_| std::future::ready(Ok(())),
            |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await
        .expect("incomplete pre-output json event should retry");

    assert_eq!(result, "ok");
    assert_eq!(deltas, vec!["ok"]);
    assert_eq!(request_count.load(Ordering::SeqCst), 2);
    server.await.expect("server task should finish");
}

#[tokio::test]
async fn compatible_chat_stream_shares_retry_budget_across_creation_and_read() {
    let body = r#"data: {"choices":[{"index":0,"delta":{"content":"too-late"},"finish_reason":"stop"}]}

data: [DONE]

"#;
    let third_response = Box::leak(
            format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            )
            .into_boxed_str(),
        );
    let (base_url, request_count, server) = spawn_chat_completion_server(vec![
            "HTTP/1.1 500 Internal Server Error\r\nContent-Length: 9\r\nConnection: close\r\n\r\ntransient",
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: 100\r\nConnection: close\r\n\r\n",
            third_response,
        ])
        .await;
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base(base_url)
            .with_api_key("test"),
    );
    let mut retry_config = test_retry_config();
    retry_config.max_attempts = 2;
    let mut agent = Agent::new(client, "m1", 4, 4);
    agent.set_model_catalog(HashMap::from([(
        "m1".into(),
        ModelRequestMetadata {
            context_window: Some(32_000),
            max_output_tokens: Some(2_000),
            supports_tools: true,
            supports_reasoning: false,
            ..Default::default()
        },
    )]));
    agent.set_retry_config(retry_config);
    let mut deltas = Vec::new();

    let error = agent
        .run_oai_comp_stream_async(
            "hello",
            |delta| {
                deltas.push(delta.to_string());
                std::future::ready(Ok(()))
            },
            |_| std::future::ready(Ok(())),
            |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await
        .expect_err("read retry should not exceed shared max_attempts budget");

    assert!(
        !error.to_string().trim().is_empty(),
        "unexpected empty error message: {error:?}"
    );
    assert!(deltas.is_empty());
    assert_eq!(request_count.load(Ordering::SeqCst), 2);
    server.abort();
}

#[tokio::test]
async fn compatible_chat_stream_recovers_read_error_after_visible_text() {
    let body = r#"data: {"choices":[{"index":0,"delta":{"content":"partial"},"finish_reason":null}]}

"#;
    let first_response = Box::leak(
            format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: 1000\r\nConnection: close\r\n\r\n{body}"
            )
            .into_boxed_str(),
        );
    let second_body = r#"data: {"choices":[{"index":0,"delta":{"content":" continuation"},"finish_reason":"stop"}]}

data: [DONE]

"#;
    let second_response = Box::leak(
            format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{second_body}",
                second_body.len()
            )
            .into_boxed_str(),
        );
    let (base_url, request_count, server) =
        spawn_chat_completion_server(vec![first_response, second_response]).await;
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base(base_url)
            .with_api_key("test"),
    );
    let mut agent = Agent::new(client, "m1", 4, 4);
    agent.set_model_catalog(HashMap::from([(
        "m1".into(),
        ModelRequestMetadata {
            context_window: Some(32_000),
            max_output_tokens: Some(2_000),
            supports_tools: true,
            supports_reasoning: false,
            ..Default::default()
        },
    )]));
    agent.set_retry_config(test_retry_config());
    let mut deltas = Vec::new();
    let mut stream_issues = Vec::new();
    let mut finalized_outcomes = Vec::new();

    let result = agent
        .run_oai_comp_stream_async(
            "hello",
            |delta| {
                deltas.push(delta.to_string());
                std::future::ready(Ok(()))
            },
            |event| {
                match event {
                    AgentEvent::TurnFinalized(event) => finalized_outcomes.push(event.outcome),
                    AgentEvent::ModelStreamIssue { message, .. } => stream_issues.push(message),
                    _ => {}
                }
                std::future::ready(Ok(()))
            },
            |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await
        .expect("post-output stream read failure should continue with a fresh iteration");

    let expected = "partial continuation".to_string();
    assert_eq!(result, expected);
    assert_eq!(deltas, vec!["partial", " continuation"]);
    assert_eq!(stream_issues, vec!["Model stream interrupted"]);
    assert_eq!(finalized_outcomes, vec!["completed"]);
    assert_eq!(request_count.load(Ordering::SeqCst), 2);
    assert!(
        agent
            .history
            .iter()
            .any(|item| matches!(item, HistoryItem::AssistantText { text } if text == "partial"))
    );
    server.await.expect("server task should finish");
}

#[tokio::test]
async fn compatible_chat_stream_recovers_missing_finish_reason_after_visible_text() {
    let body = r#"data: {"choices":[{"index":0,"delta":{"content":"partial"},"finish_reason":null}]}

data: [DONE]

"#;
    let first_response = Box::leak(
            format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            )
            .into_boxed_str(),
        );
    let second_body = r#"data: {"choices":[{"index":0,"delta":{"content":" continuation"},"finish_reason":"stop"}]}

data: [DONE]

"#;
    let second_response = Box::leak(
            format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{second_body}",
                second_body.len()
            )
            .into_boxed_str(),
        );
    let (base_url, request_count, server) =
        spawn_chat_completion_server(vec![first_response, second_response]).await;
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base(base_url)
            .with_api_key("test"),
    );
    let mut agent = Agent::new(client, "m1", 4, 4);
    agent.set_model_catalog(HashMap::from([(
        "m1".into(),
        ModelRequestMetadata {
            context_window: Some(32_000),
            max_output_tokens: Some(2_000),
            supports_tools: true,
            supports_reasoning: false,
            ..Default::default()
        },
    )]));
    agent.set_retry_config(test_retry_config());

    let result = agent
        .run_oai_comp_stream_async(
            "hello",
            |_| std::future::ready(Ok(())),
            |_| std::future::ready(Ok(())),
            |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await
        .expect("missing finish_reason after visible text should continue next iteration");

    let expected = "partial continuation".to_string();
    assert_eq!(result, expected);
    assert!(matches!(
        agent.history.first(),
        Some(HistoryItem::UserMessage { .. })
    ));
    assert!(
        agent
            .history
            .iter()
            .any(|item| matches!(item, HistoryItem::AssistantText { text } if text == "partial"))
    );
    assert_eq!(request_count.load(Ordering::SeqCst), 2);
    server.await.expect("server task should finish");
}

#[tokio::test]
async fn compatible_chat_stream_cancels_pending_tool_call_on_invalid_finish_reason() {
    let first_body = r#"data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call-interrupted","type":"function","function":{"name":"shell__exec","arguments":""}}]},"finish_reason":"stop"}]}

data: [DONE]

"#;
    let second_body = r#"data: {"choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}

data: [DONE]

"#;
    let first_response = Box::leak(
            format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{first_body}",
                first_body.len()
            )
            .into_boxed_str(),
        );
    let second_response = Box::leak(
            format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{second_body}",
                second_body.len()
            )
            .into_boxed_str(),
        );
    let (base_url, request_count, server) =
        spawn_chat_completion_server(vec![first_response, second_response]).await;
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base(base_url)
            .with_api_key("test"),
    );
    let mut agent = Agent::new(client, "m1", 4, 4);
    agent.set_model_catalog(HashMap::from([(
        "m1".into(),
        ModelRequestMetadata {
            context_window: Some(32_000),
            max_output_tokens: Some(2_000),
            supports_tools: true,
            supports_reasoning: false,
            ..Default::default()
        },
    )]));
    agent.set_retry_config(test_retry_config());
    let mut cancelled_calls = Vec::new();
    let mut started_calls = Vec::new();
    let mut finished_calls = Vec::new();
    let mut stream_issues = Vec::new();

    let result = agent
        .run_oai_comp_stream_async(
            "hello",
            |_| std::future::ready(Ok(())),
            |event| {
                match event {
                    AgentEvent::ToolCallCancelled { call_id, name } => {
                        cancelled_calls.push((call_id, name));
                    }
                    AgentEvent::ToolCallStarted { call_id, .. } => started_calls.push(call_id),
                    AgentEvent::ToolCallFinished { call_id, .. } => finished_calls.push(call_id),
                    AgentEvent::ModelStreamIssue { message, .. } => stream_issues.push(message),
                    _ => {}
                }
                std::future::ready(Ok(()))
            },
            |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await
        .expect("invalid finish reason with pending tool should cancel and continue");

    assert_eq!(result, "ok");
    assert_eq!(
        cancelled_calls,
        vec![("call-interrupted".to_string(), "shell__exec".to_string())]
    );
    assert!(started_calls.is_empty());
    assert!(finished_calls.is_empty());
    assert_eq!(stream_issues, vec!["Model stream interrupted"]);
    assert!(!agent.history.iter().any(|item| matches!(
        item,
        HistoryItem::AssistantToolCalls { calls, .. }
            if calls.iter().any(|call| call.call_id == "call-interrupted")
    )));
    assert_eq!(request_count.load(Ordering::SeqCst), 2);
    server.await.expect("server task should finish");
}

#[tokio::test]
async fn compatible_chat_stream_cancels_pending_tool_call_before_terminal_finish_error() {
    let body = r#"data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call-filtered","type":"function","function":{"name":"shell__exec","arguments":""}}]},"finish_reason":"content_filter"}]}

data: [DONE]

"#;
    let response = Box::leak(
            format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            )
            .into_boxed_str(),
        );
    let (base_url, request_count, server) = spawn_chat_completion_server(vec![response]).await;
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base(base_url)
            .with_api_key("test"),
    );
    let mut agent = Agent::new(client, "m1", 4, 4);
    agent.set_model_catalog(HashMap::from([(
        "m1".into(),
        ModelRequestMetadata {
            context_window: Some(32_000),
            max_output_tokens: Some(2_000),
            supports_tools: true,
            supports_reasoning: false,
            ..Default::default()
        },
    )]));
    agent.set_retry_config(test_retry_config());
    let mut cancelled_calls = Vec::new();
    let mut started_calls = Vec::new();

    let error = agent
        .run_oai_comp_stream_async(
            "hello",
            |_| std::future::ready(Ok(())),
            |event| {
                match event {
                    AgentEvent::ToolCallCancelled { call_id, name } => {
                        cancelled_calls.push((call_id, name));
                    }
                    AgentEvent::ToolCallStarted { call_id, .. } => started_calls.push(call_id),
                    _ => {}
                }
                std::future::ready(Ok(()))
            },
            |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await
        .expect_err("content_filter should remain terminal after cancelling pending tool");

    assert!(error.to_string().contains("finish_reason=content_filter"));
    assert_eq!(
        cancelled_calls,
        vec![("call-filtered".to_string(), "shell__exec".to_string())]
    );
    assert!(started_calls.is_empty());
    assert_eq!(request_count.load(Ordering::SeqCst), 1);
    server.await.expect("server task should finish");
}

#[tokio::test]
async fn compatible_chat_stream_does_not_recover_terminal_finish_reason_errors() {
    for (finish_reason, expected_error) in [
        ("length", "finish_reason=length"),
        ("content_filter", "finish_reason=content_filter"),
    ] {
        let body = format!(
            r#"data: {{"choices":[{{"index":0,"delta":{{"content":"partial"}},"finish_reason":"{finish_reason}"}}]}}

data: [DONE]

"#
        );
        let response = Box::leak(
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                )
                .into_boxed_str(),
            );
        let (base_url, request_count, server) = spawn_chat_completion_server(vec![response]).await;
        let client = Client::with_config(
            OpenAIConfig::new()
                .with_api_base(base_url)
                .with_api_key("test"),
        );
        let mut agent = Agent::new(client, "m1", 4, 4);
        agent.set_model_catalog(HashMap::from([(
            "m1".into(),
            ModelRequestMetadata {
                context_window: Some(32_000),
                max_output_tokens: Some(2_000),
                supports_tools: true,
                supports_reasoning: false,
                ..Default::default()
            },
        )]));
        agent.set_retry_config(test_retry_config());
        let mut deltas = Vec::new();

        let error = agent
            .run_oai_comp_stream_async(
                "hello",
                |delta| {
                    deltas.push(delta.to_string());
                    std::future::ready(Ok(()))
                },
                |_| std::future::ready(Ok(())),
                |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
            )
            .await
            .expect_err("terminal finish_reason errors should fail explicitly");

        assert!(
            error.to_string().contains(expected_error),
            "unexpected error for {finish_reason}: {error:?}"
        );
        assert_eq!(deltas, vec!["partial"]);
        assert_eq!(request_count.load(Ordering::SeqCst), 1);
        assert!(
            !agent
                .history
                .iter()
                .any(|item| matches!(item, HistoryItem::AssistantText { .. }))
        );
        server.await.expect("server task should finish");
    }
}

#[tokio::test]
async fn compatible_chat_stream_does_not_retry_bad_request() {
    let (base_url, request_count, server) = spawn_chat_completion_server(vec![
        "HTTP/1.1 400 Bad Request\r\nContent-Length: 11\r\nConnection: close\r\n\r\nbad request",
    ])
    .await;
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base(base_url)
            .with_api_key("test"),
    );
    let request = json!({"model": "m1", "stream": true, "messages": []});
    let error = send_compatible_chat_completion_stream(&client, &request)
        .await
        .expect_err("400 should fail fast");

    assert!(
        error
            .to_string()
            .contains("chat completions request failed with status 400 Bad Request")
    );
    assert_eq!(request_count.load(Ordering::SeqCst), 1);
    server.await.expect("server task should finish");
}

#[test]
fn auto_continue_defaults_to_disabled() {
    let agent = test_agent();

    assert_eq!(agent.auto_continue(), &AutoContinueState::default());
    assert!(agent.todos().is_empty());
}

#[tokio::test]
async fn workflow_auto_continue_tool_enables_llm_controlled_state() {
    let mut agent = test_agent();
    let call = HistoryToolCall {
        call_id: "call-auto".into(),
        name: "workflow__auto_continue".into(),
        arguments_json: r#"{"enabled":true}"#.into(),
    };

    let record = agent
        .execute_tool_call(&call, &mut |_| std::future::ready(Ok(())), &mut |_| {
            std::future::ready(Ok(PermissionApproval::AllowOnce))
        })
        .await
        .expect("control tool should succeed");

    assert!(record.output.ok);
    assert!(agent.auto_continue().enabled);
}

#[tokio::test]
async fn auto_continue_runs_past_agent_limits_until_llm_disables_it() {
    let (base_url, request_count, server) = spawn_chat_completion_server(vec![
        responses_tool_batch_sse(vec![json!({
            "type": "function_call", "id": "fc-enable", "call_id": "call-enable",
            "name": "workflow__auto_continue", "arguments": "{\"enabled\":true}",
            "status": "completed"
        })]),
        responses_final_sse("first"),
        responses_tool_batch_sse(vec![json!({
            "type": "function_call", "id": "fc-disable", "call_id": "call-disable",
            "name": "workflow__auto_continue", "arguments": "{\"enabled\":false}",
            "status": "completed"
        })]),
        responses_final_sse("final"),
    ])
    .await;
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base(base_url)
            .with_api_key("test"),
    );
    let mut agent = Agent::new(client, "m1", 1, 1);

    let result = agent
        .run_stream_async(
            "continue until disabled",
            |_| std::future::ready(Ok(())),
            |_| std::future::ready(Ok(())),
            |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await
        .expect("auto-continue should outlive ordinary agent limits");

    assert_eq!(result, "firstfinal");
    assert_eq!(request_count.load(Ordering::SeqCst), 4);
    assert!(!agent.auto_continue().enabled);
    server.await.expect("server task should finish");
}

#[tokio::test]
async fn execute_tool_call_records_success_status_effects_and_started_finished_events() {
    let mut agent = test_agent();
    let call = test_tool_call(
        "workflow__todos",
        r#"{"items":[{"id":"t1","content":"first","status":"pending"}]}"#,
    );
    let mut events = Vec::new();

    let record = agent
        .execute_tool_call(
            &call,
            &mut |event| {
                events.push(event);
                std::future::ready(Ok(()))
            },
            &mut |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await
        .expect("tool call should succeed");

    assert_eq!(record.status, ToolExecutionStatus::Executed);
    assert_eq!(record.rejection, None);
    assert!(record.output.ok);
    assert_eq!(record.effects.kind, ToolEffectKind::WorkflowControl);
    assert!(matches!(
        events.as_slice(),
        [
            AgentEvent::ToolCallStarted { .. },
            AgentEvent::TodoSnapshotUpdated { .. },
            AgentEvent::ToolCallFinished { ok: true, .. },
            AgentEvent::ToolExecutionSummary(ToolExecutionSummaryEvent {
                status,
                effect_kind,
                ..
            })
        ] if status == "executed" && effect_kind == "workflow_control"
    ));
}

#[tokio::test]
async fn workflow_todos_tool_updates_todo_state() {
    let mut agent = test_agent();
    let call = HistoryToolCall {
            call_id: "call-todos".into(),
            name: "workflow__todos".into(),
            arguments_json: r#"{"items":[{"id":"t1","content":"first","status":"pending"},{"id":"t2","content":"done","status":"completed"}]}"#.into(),
        };

    agent
        .execute_tool_call(&call, &mut |_| std::future::ready(Ok(())), &mut |_| {
            std::future::ready(Ok(PermissionApproval::AllowOnce))
        })
        .await
        .expect("todo control tool should succeed");

    assert_eq!(agent.todos().len(), 2);
    assert_eq!(agent.todos()[0].status, TodoStatus::Pending);
    assert_eq!(agent.todos()[1].status, TodoStatus::Completed);
}

#[tokio::test]
async fn workflow_todos_event_failure_does_not_mutate_state() {
    let mut agent = test_agent();
    let previous = vec![TodoItem {
        id: "old".into(),
        content: "old task".into(),
        status: TodoStatus::InProgress,
    }];
    agent.turn.workflow.todos = previous.clone();
    let args = json!({
        "items": [{"id":"new","content":"new task","status":"pending"}]
    });

    let result = agent
        .apply_control_tool_state("workflow__todos", &args, &mut |_| {
            std::future::ready(Err(anyhow!("event sink failed")))
        })
        .await;

    assert!(result.is_err());
    assert_eq!(agent.todos(), previous.as_slice());
}

#[tokio::test]
async fn workflow_auto_continue_event_failure_does_not_mutate_state() {
    let mut agent = test_agent();
    let previous = AutoContinueState { enabled: false };
    agent.turn.workflow.auto_continue = previous.clone();
    let args = json!({"enabled": true});

    let result = agent
        .apply_control_tool_state("workflow__auto_continue", &args, &mut |_| {
            std::future::ready(Err(anyhow!("event sink failed")))
        })
        .await;

    assert!(result.is_err());
    assert_eq!(agent.auto_continue(), &previous);
}

fn writable_call(
    call_id: &str,
    tool: &str,
    path: &std::path::Path,
    content: &str,
) -> HistoryToolCall {
    HistoryToolCall {
        call_id: call_id.into(),
        name: tool.into(),
        arguments_json: json!({"path": path, "content": content}).to_string(),
    }
}

fn writable_event_phases(events: &[AgentEvent]) -> Vec<bool> {
    events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::ToolCallStarted { .. } => Some(true),
            AgentEvent::ToolCallFinished { ok, .. } => Some(*ok),
            _ => None,
        })
        .collect()
}

fn apply_patch_call(call_id: &str, edits: Vec<Value>) -> HistoryToolCall {
    HistoryToolCall {
        call_id: call_id.into(),
        name: "edit__apply_patch".into(),
        arguments_json: json!({"edits": edits}).to_string(),
    }
}

fn apply_patch_edit(path: &std::path::Path, find: &str, replace: &str) -> Value {
    json!({
        "path": path,
        "find": find,
        "replace": replace,
        "replace_all": false,
    })
}

#[cfg(unix)]
struct UnixWritableFixture {
    external: PathBuf,
    workspace: PathBuf,
}

#[cfg(unix)]
impl UnixWritableFixture {
    fn new(name: &str) -> Self {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time")
            .as_nanos();
        let external = std::env::temp_dir().join(format!("letcode-agent-{name}-{unique}"));
        let workspace = PathBuf::from("target").join(format!("letcode-agent-{name}-{unique}"));
        std::fs::create_dir_all(&external).expect("create external fixture");
        std::fs::create_dir_all(&workspace).expect("create workspace fixture");
        Self {
            external,
            workspace,
        }
    }
}

#[cfg(unix)]
impl Drop for UnixWritableFixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.workspace);
        let _ = std::fs::remove_dir_all(&self.external);
    }
}

#[cfg(unix)]
#[tokio::test]
async fn apply_patch_permission_modes_authorize_external_and_mixed_batches() {
    let fixture = UnixWritableFixture::new("apply-patch-mode-matrix");
    for (mode, batch) in [
        (PermissionMode::Default, "external"),
        (PermissionMode::Default, "mixed"),
        (PermissionMode::Safe, "external"),
        (PermissionMode::Safe, "mixed"),
        (PermissionMode::Yolo, "external"),
        (PermissionMode::Yolo, "mixed"),
    ] {
        let external = fixture
            .external
            .join(format!("{mode:?}-{batch}-external.txt"));
        let internal = fixture
            .workspace
            .join(format!("{mode:?}-{batch}-internal.txt"));
        std::fs::write(&external, "old external").expect("seed external target");
        std::fs::write(&internal, "old internal").expect("seed internal target");
        let mut edits = vec![apply_patch_edit(&external, "old", "new")];
        if batch == "mixed" {
            edits.push(apply_patch_edit(&internal, "old", "new"));
        }
        let call = apply_patch_call(&format!("apply-patch-{mode:?}-{batch}"), edits);
        let expected_preview = format!(
            "Outside-workspace access requested:\n- {}",
            external
                .canonicalize()
                .expect("canonical external target")
                .display()
        );
        let expected_targets = if batch == "mixed" { 2 } else { 1 };
        let mut agent = test_agent();
        agent.set_permission_mode(mode);
        let mut approvals = 0;
        let mut events = Vec::new();
        let record = agent
            .execute_tool_call(
                &call,
                &mut |event| {
                    events.push(event);
                    std::future::ready(Ok(()))
                },
                &mut |request| {
                    approvals += 1;
                    assert_eq!(request.preview.as_deref(), Some(expected_preview.as_str()));
                    assert_eq!(request.can_allow_always, mode == PermissionMode::Default);
                    assert_eq!(
                        request.grant_summary.as_deref(),
                        (mode == PermissionMode::Default)
                            .then(|| {
                                format!("edit__apply_patch: {expected_targets} target path(s)")
                            })
                            .as_deref()
                    );
                    std::future::ready(Ok(PermissionApproval::AllowOnce))
                },
            )
            .await
            .expect("approved patch executes");

        assert_eq!(approvals, usize::from(mode != PermissionMode::Yolo));
        assert_eq!(writable_event_phases(&events), vec![true, true]);
        assert_eq!(record.status, ToolExecutionStatus::Executed);
        assert_eq!(record.rejection, None);
        assert!(
            record.output.ok,
            "{mode:?} {batch}: {:?}",
            record.output.error
        );
        assert_eq!(std::fs::read_to_string(&external).unwrap(), "new external");
        if batch == "mixed" {
            assert_eq!(std::fs::read_to_string(&internal).unwrap(), "new internal");
        }
    }
}

#[cfg(unix)]
#[tokio::test]
async fn apply_patch_allow_always_grants_exact_targets_and_prepares_each_call() {
    let fixture = UnixWritableFixture::new("apply-patch-allow-always");
    let first = fixture.external.join("first.txt");
    let second = fixture.external.join("second.txt");
    let third = fixture.external.join("third.txt");
    for path in [&first, &second, &third] {
        std::fs::write(path, "old").expect("seed target");
    }
    let mut agent = test_agent();
    let mut approvals = 0;
    let mut previews = Vec::new();
    let mut grant_summaries = Vec::new();

    let first_record = agent
        .execute_tool_call(
            &apply_patch_call(
                "apply-patch-grant-first",
                vec![
                    apply_patch_edit(&first, "old", "first"),
                    apply_patch_edit(&second, "old", "first"),
                ],
            ),
            &mut |_| std::future::ready(Ok(())),
            &mut |request| {
                approvals += 1;
                previews.push(request.preview.expect("external preview"));
                grant_summaries.push(request.grant_summary.expect("target-set grant"));
                std::future::ready(Ok(PermissionApproval::AllowAlways))
            },
        )
        .await
        .expect("first target set executes");
    assert!(first_record.output.ok, "{:?}", first_record.output.error);

    let replacement = fixture.external.join("first-replacement.txt");
    std::fs::write(&replacement, "replacement").expect("seed replacement inode");
    std::fs::rename(&replacement, &first).expect("replace first target inode");
    let repeated = agent
        .execute_tool_call(
            &apply_patch_call(
                "apply-patch-grant-repeated",
                vec![
                    apply_patch_edit(&first, "replacement", "second"),
                    apply_patch_edit(&second, "first", "second"),
                ],
            ),
            &mut |_| std::future::ready(Ok(())),
            &mut |_| {
                approvals += 1;
                std::future::ready(Ok(PermissionApproval::Deny))
            },
        )
        .await
        .expect("matching target set reuses grant and prepares afresh");
    assert!(repeated.output.ok, "{:?}", repeated.output.error);
    assert_eq!(approvals, 1, "identical canonical target set reuses grant");
    assert_eq!(std::fs::read_to_string(&first).unwrap(), "second");
    assert_eq!(std::fs::read_to_string(&second).unwrap(), "second");

    let distinct = agent
        .execute_tool_call(
            &apply_patch_call(
                "apply-patch-grant-distinct",
                vec![
                    apply_patch_edit(&first, "second", "third"),
                    apply_patch_edit(&third, "old", "third"),
                ],
            ),
            &mut |_| std::future::ready(Ok(())),
            &mut |request| {
                approvals += 1;
                previews.push(request.preview.expect("external preview"));
                grant_summaries.push(request.grant_summary.expect("target-set grant"));
                std::future::ready(Ok(PermissionApproval::AllowOnce))
            },
        )
        .await
        .expect("distinct target set requests approval");
    assert!(distinct.output.ok, "{:?}", distinct.output.error);
    assert_eq!(approvals, 2);
    assert_eq!(
        grant_summaries,
        vec!["edit__apply_patch: 2 target path(s)"; 2]
    );
    assert_ne!(
        previews[0], previews[1],
        "distinct canonical target set previews differ"
    );
    assert_eq!(std::fs::read_to_string(&third).unwrap(), "third");
}

#[cfg(unix)]
#[tokio::test]
async fn apply_patch_approval_and_started_rebinding_fail_closed() {
    use std::os::unix::fs::symlink;

    for timing in ["approval", "started"] {
        for scenario in [
            "parent",
            "leaf",
            "inside-outside",
            "outside-inside",
            "missing",
        ] {
            let fixture = UnixWritableFixture::new(&format!("apply-patch-{timing}-{scenario}"));
            let raw = match scenario {
                "inside-outside" => fixture.workspace.join("target.txt"),
                "parent" => fixture.external.join("parent").join("target.txt"),
                _ => fixture.external.join("target.txt"),
            };
            std::fs::create_dir_all(raw.parent().expect("target parent"))
                .expect("create target parent");
            std::fs::write(&raw, "authorized").expect("seed authorized target");
            let inside = fixture.workspace.join("inside.txt");
            let outside = fixture.external.join("outside.txt");
            std::fs::write(&inside, "inside").expect("seed inside alternate");
            std::fs::write(&outside, "outside").expect("seed outside alternate");
            let parent_stash = fixture.external.join("parent-stash");
            let missing = fixture.external.join("missing").join("target.txt");
            let call = apply_patch_call(
                &format!("apply-patch-{timing}-{scenario}"),
                vec![apply_patch_edit(&raw, "authorized", "must not write")],
            );
            let mutate = || match scenario {
                "parent" => {
                    let parent = raw.parent().unwrap();
                    std::fs::rename(parent, &parent_stash).expect("replace authorized parent");
                    std::fs::create_dir(parent).expect("create replacement parent");
                    std::fs::write(&raw, "replacement").expect("seed replacement target");
                }
                "leaf" => {
                    let replacement = fixture.external.join("replacement.txt");
                    std::fs::write(&replacement, "replacement").expect("seed replacement inode");
                    std::fs::rename(&replacement, &raw).expect("replace authorized leaf");
                }
                "inside-outside" => {
                    std::fs::remove_file(&raw).expect("remove inside target");
                    symlink(&outside, &raw).expect("rebind inside path outside");
                }
                "outside-inside" => {
                    std::fs::remove_file(&raw).expect("remove outside target");
                    symlink(&inside, &raw).expect("rebind outside path inside");
                }
                "missing" => {
                    std::fs::remove_file(&raw).expect("remove authorized target");
                    symlink(&missing, &raw).expect("rebind to missing target");
                }
                _ => unreachable!(),
            };
            let mut agent = test_agent();
            let mut events = Vec::new();
            let mut approvals = 0;
            let record = agent
                .execute_tool_call(
                    &call,
                    &mut |event| {
                        if timing == "started"
                            && matches!(event, AgentEvent::ToolCallStarted { .. })
                        {
                            mutate();
                        }
                        events.push(event);
                        std::future::ready(Ok(()))
                    },
                    &mut |_| {
                        approvals += 1;
                        if timing == "approval" {
                            mutate();
                        }
                        std::future::ready(Ok(PermissionApproval::AllowOnce))
                    },
                )
                .await
                .expect("security failure is recorded");

            assert_eq!(approvals, 1, "{timing} {scenario}");
            assert_eq!(
                writable_event_phases(&events),
                vec![true, false],
                "{timing} {scenario}"
            );
            assert_eq!(
                record.status,
                ToolExecutionStatus::Executed,
                "{timing} {scenario}"
            );
            assert_eq!(record.rejection, None, "{timing} {scenario}");
            assert_eq!(
                record
                    .output
                    .error
                    .as_ref()
                    .map(|error| error.message.as_str()),
                Some("apply patch target changed after authorization"),
                "{timing} {scenario}"
            );
            assert_eq!(std::fs::read_to_string(&inside).unwrap(), "inside");
            assert_eq!(std::fs::read_to_string(&outside).unwrap(), "outside");
            if scenario == "parent" {
                assert_eq!(
                    std::fs::read_to_string(parent_stash.join("target.txt")).unwrap(),
                    "authorized"
                );
                assert_eq!(std::fs::read_to_string(&raw).unwrap(), "replacement");
            } else if scenario == "leaf" {
                assert_eq!(std::fs::read_to_string(&raw).unwrap(), "replacement");
            }
        }
    }
}

#[cfg(unix)]
#[tokio::test]
async fn apply_patch_target_cap_fails_before_approval_or_started() {
    let fixture = UnixWritableFixture::new("apply-patch-target-cap");
    let edits = (0..65)
        .map(|index| {
            let path = fixture.external.join(format!("{index}.txt"));
            std::fs::write(&path, "old").expect("seed target");
            apply_patch_edit(&path, "old", "new")
        })
        .collect();
    let call = apply_patch_call("apply-patch-target-cap", edits);
    let mut agent = test_agent();
    let mut approvals = 0;
    let mut events = Vec::new();
    let record = agent
        .execute_tool_call(
            &call,
            &mut |event| {
                events.push(event);
                std::future::ready(Ok(()))
            },
            &mut |_| {
                approvals += 1;
                std::future::ready(Ok(PermissionApproval::AllowOnce))
            },
        )
        .await
        .expect("prepare failure is recorded");
    assert_eq!(approvals, 0);
    assert_eq!(writable_event_phases(&events), vec![false]);
    assert_eq!(record.status, ToolExecutionStatus::Rejected);
    assert_eq!(record.rejection, None);
    assert_eq!(
        record
            .output
            .error
            .as_ref()
            .map(|error| error.message.as_str()),
        Some("apply patch accepts at most 64 unique target files")
    );
    for index in 0..65 {
        assert_eq!(
            std::fs::read_to_string(fixture.external.join(format!("{index}.txt"))).unwrap(),
            "old"
        );
    }
}

#[cfg(not(unix))]
#[tokio::test]
async fn apply_patch_unsupported_platform_fails_before_approval_or_started() {
    let call = apply_patch_call(
        "apply-patch-unsupported-platform",
        vec![apply_patch_edit(
            std::path::Path::new("Cargo.toml"),
            "[package]",
            "[package]",
        )],
    );
    let mut agent = test_agent();
    let mut approvals = 0;
    let mut events = Vec::new();
    let record = agent
        .execute_tool_call(
            &call,
            &mut |event| {
                events.push(event);
                std::future::ready(Ok(()))
            },
            &mut |_| {
                approvals += 1;
                std::future::ready(Ok(PermissionApproval::AllowOnce))
            },
        )
        .await
        .expect("unsupported prepare failure is recorded");
    assert_eq!(approvals, 0);
    assert_eq!(writable_event_phases(&events), vec![false]);
    assert_eq!(record.status, ToolExecutionStatus::Rejected);
    assert_eq!(record.rejection, None);
    assert_eq!(
        record
            .output
            .error
            .as_ref()
            .map(|error| error.message.as_str()),
        Some("secure apply patch authorization is unsupported on this platform")
    );
}

#[cfg(unix)]
#[tokio::test]
async fn default_allow_always_writable_grants_use_raw_path_identity() {
    use std::os::unix::ffi::OsStringExt;
    use std::os::unix::fs::symlink;

    let fixture = UnixWritableFixture::new("raw-grant-identity");
    let target_a = fixture
        .external
        .join(std::ffi::OsString::from_vec(b"same-lossy-\x80".to_vec()));
    let target_b = fixture
        .external
        .join(std::ffi::OsString::from_vec(b"same-lossy-\x81".to_vec()));
    let workspace_a = fixture.workspace.join("a");
    let workspace_b = fixture.workspace.join("b");
    // Keep the user-supplied paths UTF-8 while resolving the leaf through a
    // symlink. APFS permits these dangling symlinks even though it rejects a
    // non-UTF-8 leaf at open time. Linux filesystems that permit those leaves
    // exercise the successful write path below without changing this fixture.
    symlink(&target_a, &workspace_a).expect("link target A");
    symlink(&target_b, &workspace_b).expect("link target B");

    let call_a = writable_call("raw-a", "fs__write", &workspace_a, "A");
    let call_a_again = writable_call("raw-a-again", "fs__write", &workspace_a, "A again");
    let call_b = writable_call("raw-b", "fs__write", &workspace_b, "B");
    let mut previews = Vec::new();
    let mut grant_summaries = Vec::new();
    let mut agent_a = test_agent();

    let first = agent_a
        .execute_tool_call(
            &call_a,
            &mut |_| std::future::ready(Ok(())),
            &mut |request| {
                previews.push(request.preview.expect("external preview"));
                grant_summaries.push(request.grant_summary.expect("grant summary"));
                assert!(request.can_allow_always);
                std::future::ready(Ok(PermissionApproval::AllowAlways))
            },
        )
        .await
        .expect("A is approved");
    assert_eq!(first.status, ToolExecutionStatus::Executed);

    let mut repeated_approval_requested = false;
    let repeated = agent_a
        .execute_tool_call(
            &call_a_again,
            &mut |_| std::future::ready(Ok(())),
            &mut |_| {
                repeated_approval_requested = true;
                std::future::ready(Ok(PermissionApproval::Deny))
            },
        )
        .await
        .expect("A grant executes");
    assert_eq!(repeated.status, ToolExecutionStatus::Executed);
    assert!(
        !repeated_approval_requested,
        "matching raw-byte grant must not request approval"
    );

    let denied_b = agent_a
        .execute_tool_call(
            &call_b,
            &mut |_| std::future::ready(Ok(())),
            &mut |request| {
                previews.push(request.preview.expect("external preview"));
                grant_summaries.push(request.grant_summary.expect("grant summary"));
                std::future::ready(Ok(PermissionApproval::Deny))
            },
        )
        .await
        .expect("B denial is recorded");
    assert_eq!(denied_b.status, ToolExecutionStatus::Rejected);
    assert_eq!(
        denied_b.rejection,
        Some(ToolExecutionRejection::PermissionDeniedByUser)
    );

    assert_eq!(previews.len(), 2);
    assert_eq!(grant_summaries.len(), 2);
    assert_ne!(
        previews[0], previews[1],
        "raw-byte preview markers distinguish A and B"
    );
    assert_ne!(
        grant_summaries[0], grant_summaries[1],
        "raw-byte grant markers distinguish A and B"
    );
    if cfg!(target_os = "linux") {
        assert!(
            first.output.ok,
            "Linux raw-byte leaf should be writable: {:?}",
            first.output.error
        );
        assert!(
            repeated.output.ok,
            "Linux raw-byte leaf should be writable: {:?}",
            repeated.output.error
        );
        assert_eq!(std::fs::read(&target_a).expect("A was written"), b"A again");
        assert!(
            !target_b.exists(),
            "denied B must not create its raw-byte leaf"
        );
    }
}

#[cfg(unix)]
#[tokio::test]
async fn writable_inside_to_outside_rebind_is_a_post_started_security_failure() {
    use std::os::unix::fs::symlink;

    let fixture = UnixWritableFixture::new("inside-outside-rebind");
    let raw = fixture.workspace.join("inside.txt");
    let outside = fixture.external.join("outside.txt");
    std::fs::write(&raw, "inside").expect("create inside target");
    let call = writable_call("inside-outside", "fs__write", &raw, "must not write");
    let mut agent = test_agent();
    let mut events = Vec::new();
    let record = agent
        .execute_tool_call(
            &call,
            &mut |event| {
                if matches!(event, AgentEvent::ToolCallStarted { .. }) {
                    std::fs::remove_file(&raw).expect("remove inside target");
                    symlink(&outside, &raw).expect("rebind to external leaf");
                }
                events.push(event);
                std::future::ready(Ok(()))
            },
            &mut |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await
        .expect("security failure is recorded");

    assert_eq!(writable_event_phases(&events), vec![true, false]);
    assert_eq!(record.status, ToolExecutionStatus::Executed);
    assert_eq!(record.rejection, None);
    assert_eq!(
        record
            .output
            .error
            .as_ref()
            .map(|error| error.message.as_str()),
        Some("writable destination changed after authorization")
    );
    assert!(
        !raw.exists(),
        "rebinding must not recreate the original target"
    );
    assert!(!outside.exists(), "rebinding must not write outside");
}

#[cfg(unix)]
#[tokio::test]
async fn writable_unresolvable_rebind_is_a_post_started_security_failure() {
    use std::os::unix::fs::symlink;

    let fixture = UnixWritableFixture::new("unresolvable-rebind");
    let raw = fixture.workspace.join("authorized.txt");
    let missing = fixture.external.join("missing-parent").join("outside.txt");
    std::fs::write(&raw, "original").expect("create authorized target");
    let call = writable_call("unresolvable-rebind", "fs__write", &raw, "must not write");
    let mut agent = test_agent();
    let mut events = Vec::new();
    let record = agent
        .execute_tool_call(
            &call,
            &mut |event| {
                if matches!(event, AgentEvent::ToolCallStarted { .. }) {
                    std::fs::remove_file(&raw).expect("remove authorized target");
                    symlink(&missing, &raw).expect("rebind to missing parent");
                }
                events.push(event);
                std::future::ready(Ok(()))
            },
            &mut |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await
        .expect("security failure is recorded");

    assert_eq!(writable_event_phases(&events), vec![true, false]);
    assert_eq!(record.status, ToolExecutionStatus::Executed);
    assert_eq!(record.rejection, None);
    assert_eq!(
        record
            .output
            .error
            .as_ref()
            .map(|error| error.message.as_str()),
        Some("writable destination changed after authorization")
    );
    assert!(
        !raw.exists(),
        "unresolvable rebind must not recreate original"
    );
    assert!(
        !missing.exists(),
        "unresolvable rebind must not write elsewhere"
    );
}

#[cfg(unix)]
#[tokio::test]
async fn writable_tools_cover_allow_once_and_yolo_external_execution_events() {
    let fixture = UnixWritableFixture::new("mode-events");
    for (tool, mode) in [
        ("fs__write", PermissionMode::Default),
        ("fs__write", PermissionMode::Safe),
        ("fs__append", PermissionMode::Default),
        ("fs__append", PermissionMode::Safe),
    ] {
        let path = fixture.external.join(format!("allow-once-{tool}-{mode:?}"));
        if tool == "fs__append" {
            std::fs::write(&path, "seed").expect("seed append target");
        }
        let call = writable_call(
            &format!("allow-once-{tool}-{mode:?}"),
            tool,
            &path,
            "approved",
        );
        let mut agent = test_agent();
        agent.set_permission_mode(mode);
        let mut approvals = 0;
        let mut events = Vec::new();
        let record = agent
            .execute_tool_call(
                &call,
                &mut |event| {
                    events.push(event);
                    std::future::ready(Ok(()))
                },
                &mut |request| {
                    approvals += 1;
                    assert_eq!(request.can_allow_always, mode == PermissionMode::Default);
                    std::future::ready(Ok(PermissionApproval::AllowOnce))
                },
            )
            .await
            .expect("allow-once execution");
        assert_eq!(approvals, 1, "{tool} {mode:?}");
        assert_eq!(
            writable_event_phases(&events),
            vec![true, true],
            "{tool} {mode:?}"
        );
        assert_eq!(record.status, ToolExecutionStatus::Executed);
        assert_eq!(record.rejection, None);
        assert_eq!(
            std::fs::read_to_string(&path).expect("written target"),
            if tool == "fs__append" {
                "seedapproved"
            } else {
                "approved"
            },
            "{tool} {mode:?} effect"
        );
    }

    for tool in ["fs__write", "fs__append"] {
        let path = fixture.external.join(format!("yolo-{tool}"));
        if tool == "fs__append" {
            std::fs::write(&path, "seed").expect("seed append target");
        }
        let call = writable_call(&format!("yolo-{tool}"), tool, &path, "yolo");
        let mut agent = test_agent();
        agent.set_permission_mode(PermissionMode::Yolo);
        let mut approvals = 0;
        let mut events = Vec::new();
        let record = agent
            .execute_tool_call(
                &call,
                &mut |event| {
                    events.push(event);
                    std::future::ready(Ok(()))
                },
                &mut |_| {
                    approvals += 1;
                    std::future::ready(Ok(PermissionApproval::Deny))
                },
            )
            .await
            .expect("yolo execution");
        assert_eq!(approvals, 0, "{tool}");
        assert_eq!(writable_event_phases(&events), vec![true, true], "{tool}");
        assert!(record.output.ok, "{tool}: {:?}", record.output.error);
        assert_eq!(
            std::fs::read_to_string(&path).expect("yolo target"),
            if tool == "fs__append" {
                "seedyolo"
            } else {
                "yolo"
            },
            "{tool} yolo effect"
        );
    }
}

#[cfg(unix)]
#[tokio::test]
async fn writable_tools_cover_user_and_directive_denials_without_starting() {
    let fixture = UnixWritableFixture::new("denial-events");
    for tool in ["fs__write", "fs__append"] {
        for (denial, modes) in [
            ("user", &[PermissionMode::Default, PermissionMode::Safe][..]),
            ("directive", &[PermissionMode::Default][..]),
        ] {
            for mode in modes {
                let path = fixture.external.join(format!("{denial}-{tool}"));
                let call = writable_call(&format!("{denial}-{tool}"), tool, &path, "denied");
                let mut agent = test_agent();
                agent.set_permission_mode(*mode);
                if denial == "directive" {
                    agent.turn = TurnRuntimeState::new(
                        1,
                        WorkflowTurnState::from_user_input("Read-only: inspect and report only."),
                    );
                }
                let mut approvals = 0;
                let mut events = Vec::new();
                let record = agent
                    .execute_tool_call(
                        &call,
                        &mut |event| {
                            events.push(event);
                            std::future::ready(Ok(()))
                        },
                        &mut |_| {
                            approvals += 1;
                            std::future::ready(Ok(PermissionApproval::Deny))
                        },
                    )
                    .await
                    .expect("denial is recorded");
                assert_eq!(
                    writable_event_phases(&events),
                    vec![false],
                    "{tool} {denial} {mode:?}"
                );
                assert!(
                    !events
                        .iter()
                        .any(|event| matches!(event, AgentEvent::ToolCallStarted { .. })),
                    "{tool} {denial} {mode:?} must not start"
                );
                assert!(
                    events.iter().any(|event| matches!(
                        event,
                        AgentEvent::ToolCallFinished { ok: false, .. }
                    )),
                    "{tool} {denial} {mode:?} must finish unsuccessfully"
                );
                assert_eq!(record.status, ToolExecutionStatus::Rejected);
                assert_eq!(
                    record.rejection,
                    Some(if denial == "user" {
                        ToolExecutionRejection::PermissionDeniedByUser
                    } else {
                        ToolExecutionRejection::DirectiveBlocked
                    }),
                    "{tool} {denial} {mode:?} rejection"
                );
                assert_eq!(
                    approvals,
                    usize::from(denial == "user"),
                    "{tool} {denial} {mode:?}"
                );
                assert!(!path.exists(), "{tool} {denial} {mode:?} must not write");
            }
        }
    }
}

#[cfg(unix)]
#[tokio::test]
async fn writable_tools_cover_post_started_rebind_failures() {
    use std::os::unix::fs::symlink;

    let fixture = UnixWritableFixture::new("post-started-events");
    for tool in ["fs__write", "fs__append"] {
        let original = fixture.external.join(format!("original-{tool}"));
        let rebound = fixture.external.join(format!("rebound-{tool}"));
        let raw = fixture.workspace.join(format!("raw-{tool}"));
        std::fs::write(&original, "original").expect("create original target");
        std::fs::write(&rebound, "rebound").expect("create rebound target");
        symlink(&original, &raw).expect("create initial leaf link");
        let call = writable_call(
            &format!("post-started-{tool}"),
            tool,
            &raw,
            "must not write",
        );
        let mut agent = test_agent();
        let mut events = Vec::new();
        let record = agent
            .execute_tool_call(
                &call,
                &mut |event| {
                    if matches!(event, AgentEvent::ToolCallStarted { .. }) {
                        std::fs::remove_file(&raw).expect("remove initial leaf link");
                        symlink(&rebound, &raw).expect("rebind leaf link");
                    }
                    events.push(event);
                    std::future::ready(Ok(()))
                },
                &mut |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
            )
            .await
            .expect("security failure is recorded");
        assert_eq!(writable_event_phases(&events), vec![true, false], "{tool}");
        assert_eq!(record.status, ToolExecutionStatus::Executed, "{tool}");
        assert_eq!(record.rejection, None, "{tool}");
        assert_eq!(
            record
                .output
                .error
                .as_ref()
                .map(|error| error.message.as_str()),
            Some("writable destination changed after authorization"),
            "{tool}"
        );
        assert_eq!(
            std::fs::read_to_string(&original).unwrap(),
            "original",
            "{tool}"
        );
        assert_eq!(
            std::fs::read_to_string(&rebound).unwrap(),
            "rebound",
            "{tool}"
        );
    }
}

#[tokio::test]
async fn yolo_external_workspace_read_executes_without_approval() {
    let mut agent = test_agent();
    agent.set_permission_mode(PermissionMode::Yolo);
    let outside_path = std::env::temp_dir().join(format!(
        "letcode-outside-agent-read-{}.txt",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time")
            .as_nanos()
    ));
    std::fs::write(&outside_path, "outside\n").expect("write outside fixture");
    let outside = outside_path.to_string_lossy().to_string();
    let call = HistoryToolCall {
        call_id: "call-outside-read".into(),
        name: "fs__read".into(),
        arguments_json: json!({"path": outside, "offset": 1, "limit": 10}).to_string(),
    };
    let mut permission_requests = Vec::new();

    let record = agent
        .execute_tool_call(&call, &mut |_| std::future::ready(Ok(())), &mut |request| {
            permission_requests.push(request);
            std::future::ready(Ok(PermissionApproval::AllowOnce))
        })
        .await
        .expect("outside read should execute in yolo mode");

    assert!(record.output.ok, "{:?}", record.output.error);
    assert!(permission_requests.is_empty());
    assert!(
        record
            .output
            .data
            .as_ref()
            .and_then(|data| data.get("content"))
            .and_then(Value::as_str)
            .is_some_and(|content| content.contains("outside"))
    );

    let _ = std::fs::remove_file(outside_path);
}

#[tokio::test]
async fn default_external_workspace_read_allow_always_grants_only_matching_resource() {
    let mut agent = test_agent();
    let fixture_root = std::env::temp_dir().join(format!(
        "letcode-external-read-grant-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time")
            .as_nanos()
    ));
    let first_directory = fixture_root.join("first");
    let second_directory = fixture_root.join("second");
    std::fs::create_dir_all(&first_directory).expect("create first fixture directory");
    std::fs::create_dir_all(&second_directory).expect("create second fixture directory");
    let first_path = first_directory.join("read.txt");
    let second_path = second_directory.join("read.txt");
    std::fs::write(&first_path, "first\n").expect("write first fixture");
    std::fs::write(&second_path, "second\n").expect("write second fixture");

    let first_call = HistoryToolCall {
        call_id: "call-external-read-first".into(),
        name: "fs__read".into(),
        arguments_json: json!({"path": first_path, "offset": 1, "limit": 10}).to_string(),
    };
    let repeated_call = HistoryToolCall {
        call_id: "call-external-read-repeated".into(),
        name: "fs__read".into(),
        arguments_json: first_call.arguments_json.clone(),
    };
    let other_directory_call = HistoryToolCall {
        call_id: "call-external-read-other".into(),
        name: "fs__read".into(),
        arguments_json: json!({"path": second_path, "offset": 1, "limit": 10}).to_string(),
    };
    let mut approval_requests = 0;

    let first = agent
        .execute_tool_call(
            &first_call,
            &mut |_| std::future::ready(Ok(())),
            &mut |request| {
                approval_requests += 1;
                assert!(request.can_allow_always);
                std::future::ready(Ok(PermissionApproval::AllowAlways))
            },
        )
        .await
        .expect("first external read should execute after approval");
    assert!(first.output.ok);
    assert_eq!(approval_requests, 1);

    let repeated = agent
        .execute_tool_call(
            &repeated_call,
            &mut |_| std::future::ready(Ok(())),
            &mut |_| {
                approval_requests += 1;
                std::future::ready(Ok(PermissionApproval::Deny))
            },
        )
        .await
        .expect("matching grant should execute without approval");
    assert!(repeated.output.ok);
    assert_eq!(approval_requests, 1, "matching grant must bypass approval");

    let other_directory = agent
        .execute_tool_call(
            &other_directory_call,
            &mut |_| std::future::ready(Ok(())),
            &mut |_| {
                approval_requests += 1;
                std::future::ready(Ok(PermissionApproval::Deny))
            },
        )
        .await
        .expect("other external directory should request approval");
    assert_eq!(approval_requests, 2);
    assert_eq!(other_directory.status, ToolExecutionStatus::Rejected);
    assert_eq!(
        other_directory.rejection,
        Some(ToolExecutionRejection::PermissionDeniedByUser)
    );

    let _ = std::fs::remove_dir_all(fixture_root);
}

#[tokio::test]
async fn matching_grant_does_not_override_base_policy_denial() {
    let mut agent = test_agent();
    let command = "rm -rf letcode-test-target";
    agent
        .permission_session
        .lock()
        .expect("permission session")
        .grant(crate::permission::PermissionResource::Exact {
            tool: "shell__exec".into(),
            value: command.into(),
        });
    let call = HistoryToolCall {
        call_id: "call-granted-policy-denial".into(),
        name: "shell__exec".into(),
        arguments_json: json!({"command": command}).to_string(),
    };
    let mut approval_requested = false;

    let record = agent
        .execute_tool_call(&call, &mut |_| std::future::ready(Ok(())), &mut |_| {
            approval_requested = true;
            std::future::ready(Ok(PermissionApproval::AllowOnce))
        })
        .await
        .expect("policy denial should produce a rejection record");

    assert!(!approval_requested);
    assert_eq!(record.status, ToolExecutionStatus::Rejected);
    assert_eq!(
        record.rejection,
        Some(ToolExecutionRejection::PermissionDeniedByPolicy)
    );
}

#[tokio::test]
async fn yolo_external_workspace_write_executes_without_approval() {
    let mut agent = test_agent();
    agent.set_permission_mode(PermissionMode::Yolo);
    let outside_path = std::env::temp_dir().join(format!(
        "letcode-outside-agent-denied-write-{}.txt",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time")
            .as_nanos()
    ));
    let outside = outside_path.to_string_lossy().to_string();
    let call = HistoryToolCall {
        call_id: "call-outside-write-denied".into(),
        name: "fs__write".into(),
        arguments_json: json!({"path": outside, "content": "denied"}).to_string(),
    };
    let mut permission_requests = Vec::new();
    let mut events = Vec::new();

    let record = agent
        .execute_tool_call(
            &call,
            &mut |event| {
                events.push(event);
                std::future::ready(Ok(()))
            },
            &mut |request| {
                permission_requests.push(request);
                std::future::ready(Ok(PermissionApproval::Deny))
            },
        )
        .await
        .expect("outside write should execute in yolo mode");

    assert!(permission_requests.is_empty());
    assert_eq!(record.status, ToolExecutionStatus::Executed);
    assert!(record.output.ok);
    assert!(outside_path.exists());
    assert!(
        events
            .iter()
            .any(|event| matches!(event, AgentEvent::ToolCallFinished { ok: true, .. }))
    );
    let _ = std::fs::remove_file(outside_path);
}

#[tokio::test]
async fn default_and_safe_allow_once_authorize_external_writes() {
    for mode in [PermissionMode::Default, PermissionMode::Safe] {
        let mut agent = test_agent();
        agent.set_permission_mode(mode);
        let outside_path = std::env::temp_dir().join(format!(
            "letcode-external-write-allow-once-{mode:?}-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system time")
                .as_nanos()
        ));
        let call = HistoryToolCall {
            call_id: format!("call-external-write-allow-once-{mode:?}"),
            name: "fs__write".into(),
            arguments_json: json!({"path": outside_path, "content": "approved"}).to_string(),
        };
        let mut approvals = 0;
        let mut events = Vec::new();
        let record = agent
            .execute_tool_call(
                &call,
                &mut |event| {
                    events.push(event);
                    std::future::ready(Ok(()))
                },
                &mut |_| {
                    approvals += 1;
                    std::future::ready(Ok(PermissionApproval::AllowOnce))
                },
            )
            .await
            .expect("approved write should execute");
        assert_eq!(approvals, 1);
        assert_eq!(record.status, ToolExecutionStatus::Executed);
        assert!(record.output.ok, "{:?}", record.output.error);
        assert!(
            events
                .iter()
                .any(|event| matches!(event, AgentEvent::ToolCallStarted { .. }))
        );
        assert!(
            events
                .iter()
                .any(|event| matches!(event, AgentEvent::ToolCallFinished { ok: true, .. }))
        );
        let _ = std::fs::remove_file(outside_path);
    }
}

#[cfg(unix)]
#[tokio::test]
async fn default_allow_always_external_write_grant_is_cleared_at_top_level_session_boundary() {
    let fixture = UnixWritableFixture::new("allow-always-session-boundary");
    let path = fixture.external.join("written.txt");
    let first_call = writable_call("allow-always-first", "fs__write", &path, "first");
    let repeated_call = writable_call("allow-always-repeated", "fs__write", &path, "second");
    let after_reset_call = writable_call("allow-always-after-reset", "fs__write", &path, "denied");
    let mut agent = test_agent();
    let mut approvals = 0;

    let first = agent
        .execute_tool_call(
            &first_call,
            &mut |_| std::future::ready(Ok(())),
            &mut |request| {
                approvals += 1;
                assert!(request.can_allow_always);
                std::future::ready(Ok(PermissionApproval::AllowAlways))
            },
        )
        .await
        .expect("first AllowAlways write executes");
    assert!(first.output.ok, "{:?}", first.output.error);
    assert_eq!(approvals, 1);
    assert_eq!(
        std::fs::read_to_string(&path).expect("first write"),
        "first"
    );

    let repeated = agent
        .execute_tool_call(
            &repeated_call,
            &mut |_| std::future::ready(Ok(())),
            &mut |_| {
                approvals += 1;
                std::future::ready(Ok(PermissionApproval::Deny))
            },
        )
        .await
        .expect("matching session grant executes");
    assert!(repeated.output.ok, "{:?}", repeated.output.error);
    assert_eq!(approvals, 1, "matching destination must reuse the grant");
    assert_eq!(
        std::fs::read_to_string(&path).expect("repeated write"),
        "second"
    );

    // This is the production top-level session transition, not a replacement
    // Agent instance: it must discard the grant held by this Agent.
    agent.reset_for_new_session();
    let after_reset = agent
        .execute_tool_call(
            &after_reset_call,
            &mut |_| std::future::ready(Ok(())),
            &mut |request| {
                approvals += 1;
                assert!(request.can_allow_always);
                std::future::ready(Ok(PermissionApproval::Deny))
            },
        )
        .await
        .expect("post-reset denial is recorded");
    assert_eq!(approvals, 2, "reset session must request approval again");
    assert_eq!(after_reset.status, ToolExecutionStatus::Rejected);
    assert_eq!(
        after_reset.rejection,
        Some(ToolExecutionRejection::PermissionDeniedByUser)
    );
    assert_eq!(
        std::fs::read_to_string(&path).expect("denial has no write"),
        "second"
    );
}

#[cfg(unix)]
#[tokio::test]
async fn approved_external_write_rebinding_is_rejected_after_started() {
    use std::os::unix::fs::symlink;

    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time")
        .as_nanos();
    let first = std::env::temp_dir().join(format!("letcode-rebind-first-{unique}"));
    let second = std::env::temp_dir().join(format!("letcode-rebind-second-{unique}"));
    let link = PathBuf::from("target").join(format!("letcode-rebind-link-{unique}"));
    std::fs::create_dir_all(&first).expect("create first fixture");
    std::fs::create_dir_all(&second).expect("create second fixture");
    std::fs::create_dir_all("target").expect("create target fixture");
    symlink(&first, &link).expect("create initial link");
    let raw = link.join("target.txt");

    let mut agent = test_agent();
    let call = HistoryToolCall {
        call_id: "call-rebound-external-write".into(),
        name: "fs__write".into(),
        arguments_json: json!({"path": raw, "content": "must not write"}).to_string(),
    };
    let mut events = Vec::new();
    let record = agent
        .execute_tool_call(
            &call,
            &mut |event| {
                events.push(event);
                std::future::ready(Ok(()))
            },
            &mut |_| {
                std::fs::remove_file(&link).expect("remove initial link");
                symlink(&second, &link).expect("rebind link");
                std::future::ready(Ok(PermissionApproval::AllowOnce))
            },
        )
        .await
        .expect("rebound write should produce a tool record");

    assert_eq!(record.status, ToolExecutionStatus::Executed);
    assert_eq!(record.rejection, None);
    assert!(!record.output.ok);
    assert!(record.output.error.as_ref().is_some_and(|error| {
        error.message == "writable destination changed after authorization"
    }));
    assert!(
        events
            .iter()
            .any(|event| matches!(event, AgentEvent::ToolCallStarted { .. }))
    );
    assert!(
        events
            .iter()
            .any(|event| matches!(event, AgentEvent::ToolCallFinished { ok: false, .. }))
    );
    assert!(!first.join("target.txt").exists());
    assert!(!second.join("target.txt").exists());
    let _ = std::fs::remove_file(link);
    let _ = std::fs::remove_dir_all(first);
    let _ = std::fs::remove_dir_all(second);
}

#[cfg(unix)]
#[tokio::test]
async fn started_callback_rebinding_is_rejected_without_writing_either_destination() {
    use std::os::unix::fs::symlink;

    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time")
        .as_nanos();
    let first = std::env::temp_dir().join(format!("letcode-started-rebind-first-{unique}"));
    let second = std::env::temp_dir().join(format!("letcode-started-rebind-second-{unique}"));
    let link = PathBuf::from("target").join(format!("letcode-started-rebind-link-{unique}"));
    std::fs::create_dir_all(&first).expect("create first fixture");
    std::fs::create_dir_all(&second).expect("create second fixture");
    std::fs::create_dir_all("target").expect("create target fixture");
    symlink(&first, &link).expect("create initial link");
    let raw = link.join("target.txt");

    let mut agent = test_agent();
    let call = HistoryToolCall {
        call_id: "call-started-rebound-external-write".into(),
        name: "fs__write".into(),
        arguments_json: json!({"path": raw, "content": "must not write"}).to_string(),
    };
    let mut events = Vec::new();
    let record = agent
        .execute_tool_call(
            &call,
            &mut |event| {
                if matches!(event, AgentEvent::ToolCallStarted { .. }) {
                    std::fs::remove_file(&link).expect("remove initial link");
                    symlink(&second, &link).expect("rebind link");
                }
                events.push(event);
                std::future::ready(Ok(()))
            },
            &mut |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await
        .expect("rebound write should produce a tool record");

    assert_eq!(record.status, ToolExecutionStatus::Executed);
    assert_eq!(record.rejection, None);
    assert!(!record.output.ok);
    assert_eq!(
        record
            .output
            .error
            .as_ref()
            .map(|error| error.message.as_str()),
        Some("writable destination changed after authorization")
    );
    assert!(
        events
            .iter()
            .any(|event| matches!(event, AgentEvent::ToolCallStarted { .. }))
    );
    assert!(
        events
            .iter()
            .any(|event| matches!(event, AgentEvent::ToolCallFinished { ok: false, .. }))
    );
    assert!(!first.join("target.txt").exists());
    assert!(!second.join("target.txt").exists());
    let _ = std::fs::remove_file(link);
    let _ = std::fs::remove_dir_all(first);
    let _ = std::fs::remove_dir_all(second);
}

#[cfg(unix)]
#[tokio::test]
async fn leaf_symlink_callback_rebinding_is_post_started_security_failure() {
    use std::os::unix::fs::symlink;

    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let first = std::env::temp_dir().join(format!("letcode-leaf-callback-a-{unique}"));
    let second = std::env::temp_dir().join(format!("letcode-leaf-callback-b-{unique}"));
    let link = PathBuf::from("target").join(format!("letcode-leaf-callback-{unique}"));
    std::fs::create_dir_all("target").unwrap();
    std::fs::write(&first, "A").unwrap();
    std::fs::write(&second, "B").unwrap();
    symlink(&first, &link).unwrap();
    let call = HistoryToolCall {
        call_id: "call-leaf-callback-rebind".into(),
        name: "fs__write".into(),
        arguments_json: json!({"path": link, "content": "must not write"}).to_string(),
    };
    let mut agent = test_agent();
    let mut events = Vec::new();
    let record = agent
        .execute_tool_call(
            &call,
            &mut |event| {
                events.push(event);
                std::future::ready(Ok(()))
            },
            &mut |_| {
                std::fs::remove_file(&link).unwrap();
                symlink(&second, &link).unwrap();
                std::future::ready(Ok(PermissionApproval::AllowOnce))
            },
        )
        .await
        .expect("security failure is a completed execution record");

    assert_eq!(record.status, ToolExecutionStatus::Executed);
    assert_eq!(record.rejection, None);
    assert!(!record.output.ok);
    assert_eq!(
        record
            .output
            .error
            .as_ref()
            .map(|error| error.message.as_str()),
        Some("writable destination changed after authorization")
    );
    assert!(
        events
            .iter()
            .any(|event| matches!(event, AgentEvent::ToolCallStarted { .. }))
    );
    assert!(
        events
            .iter()
            .any(|event| matches!(event, AgentEvent::ToolCallFinished { ok: false, .. }))
    );
    assert_eq!(std::fs::read_to_string(&first).unwrap(), "A");
    assert_eq!(std::fs::read_to_string(&second).unwrap(), "B");
    let _ = std::fs::remove_file(link);
    let _ = std::fs::remove_file(first);
    let _ = std::fs::remove_file(second);
}

#[tokio::test]
async fn writable_leaf_preparation_failure_skips_approval_and_started_event() {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let call = test_tool_call(
        "fs__write",
        &json!({"path": format!("target/missing-parent-{unique}/leaf"), "content": "x"})
            .to_string(),
    );
    let mut agent = test_agent();
    let mut approvals = 0;
    let mut events = Vec::new();
    let record = agent
        .execute_tool_call(
            &call,
            &mut |event| {
                events.push(event);
                std::future::ready(Ok(()))
            },
            &mut |_| {
                approvals += 1;
                std::future::ready(Ok(PermissionApproval::AllowOnce))
            },
        )
        .await
        .expect("preparation failure becomes a rejection record");

    assert_eq!(approvals, 0);
    assert_eq!(record.status, ToolExecutionStatus::Rejected);
    assert_eq!(record.rejection, None);
    assert!(
        record
            .output
            .error
            .as_ref()
            .unwrap()
            .message
            .contains("parent directory does not exist")
    );
    assert!(
        events
            .iter()
            .any(|event| matches!(event, AgentEvent::ToolCallFinished { ok: false, .. }))
    );
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, AgentEvent::ToolCallStarted { .. }))
    );
}

#[tokio::test]
async fn yolo_mode_executes_commands_that_default_mode_denies_by_policy() {
    let mut agent = test_agent();
    agent.set_permission_mode(PermissionMode::Yolo);
    let call = HistoryToolCall {
        call_id: "call-yolo-deny-risk".into(),
        name: "shell__exec".into(),
        arguments_json: json!({"command": "curl --version"}).to_string(),
    };
    let mut approval_requested = false;

    let record = agent
        .execute_tool_call(&call, &mut |_| std::future::ready(Ok(())), &mut |_| {
            approval_requested = true;
            std::future::ready(Ok(PermissionApproval::Deny))
        })
        .await
        .expect("yolo mode should execute command without asking");

    assert!(!approval_requested);
    assert_eq!(record.status, ToolExecutionStatus::Executed);
    assert_ne!(
        record.rejection,
        Some(ToolExecutionRejection::PermissionDeniedByPolicy)
    );
}

#[tokio::test]
async fn auto_continue_continues_without_todo_progress() {
    let mut agent = test_agent();
    agent.prepare_turn_prelude("implement a feature");
    let turn_id = agent.current_turn_id();
    agent.turn.workflow.auto_continue = AutoContinueState { enabled: true };
    let mut continuation_count = 0;
    let mut events = Vec::new();

    let should_continue = agent
        .continue_after_no_tool_reply(
            &mut |event| {
                events.push(event);
                std::future::ready(Ok(()))
            },
            &mut continuation_count,
        )
        .await
        .expect("continuation decision succeeds");

    assert!(should_continue);
    assert_eq!(continuation_count, 1);
    assert_eq!(agent.current_turn_id(), turn_id);
    assert!(matches!(
        agent.history.last(),
        Some(HistoryItem::InternalContinuation { .. })
    ));
    assert!(matches!(
        events.as_slice(),
        [
            AgentEvent::InternalContinuation {
                source: crate::transcript::InternalContinuationSource::AutoContinue,
                ..
            },
            AgentEvent::AutoContinuationScheduled {
                continuation_count: 1,
                remaining_unfinished: 0,
            }
        ]
    ));
}

#[tokio::test]
async fn auto_continue_stops_only_when_llm_disables_it() {
    let mut agent = test_agent();
    agent.turn.workflow.auto_continue.enabled = true;
    agent.turn.workflow.todos = vec![TodoItem {
        id: "blocked".into(),
        content: "LLM records this as blocked but remains in control".into(),
        status: TodoStatus::Blocked,
    }];
    let mut continuation_count = 0;

    assert!(
        agent
            .continue_after_no_tool_reply(
                &mut |_| std::future::ready(Ok(())),
                &mut continuation_count
            )
            .await
            .expect("blocked todo must not stop auto-continue")
    );

    agent.turn.workflow.auto_continue.enabled = false;
    assert!(
        !agent
            .continue_after_no_tool_reply(
                &mut |_| std::future::ready(Ok(())),
                &mut continuation_count
            )
            .await
            .expect("LLM disable must stop auto-continue")
    );
}

#[test]
fn engineering_turn_prelude_adds_workflow_context_and_validation_reminder() {
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base("https://api.openai.com/v1")
            .with_api_key("test"),
    );
    let mut agent = Agent::new(client, "m1", 1, 1);

    let turn_prelude =
        agent.prepare_turn_prelude("Implement the fix in src/agent.rs and run cargo test.");

    assert_eq!(agent.current_turn().intent, TurnIntent::Engineering);
    assert_eq!(agent.current_turn().directive, ExecutionDirective::None);
    assert_eq!(turn_prelude.len(), agent.prelude.len() + 2);
    let runtime_message = &turn_prelude[turn_prelude.len() - 2];
    assert_eq!(
        runtime_message.role,
        crate::request_builder::PromptRole::Developer
    );
    assert!(runtime_message.text.contains("Runtime context"));
    assert!(runtime_message.text.contains("Current date:"));
    assert!(runtime_message.text.contains("Timezone:"));
    assert!(!runtime_message.text.contains("Current time:"));
    let workflow_message = &turn_prelude[turn_prelude.len() - 1];
    assert_eq!(
        workflow_message.role,
        crate::request_builder::PromptRole::Developer
    );
    assert!(workflow_message.text.contains("engineering workflow task"));
    assert!(workflow_message.text.contains("Delegate bounded work"));
    assert!(workflow_message.text.contains("context hygiene"));
    assert!(workflow_message.text.contains("targeted validation"));
}

#[test]
fn workspace_agents_files_are_loaded_in_order_and_injected_into_each_turn() {
    let workspace_root = agents_test_dir();
    let current_dir = workspace_root.join("project").join("nested");
    fs::create_dir_all(&current_dir).expect("nested workspace should be created");
    let workspace_root = workspace_root
        .canonicalize()
        .expect("workspace root resolves");
    let current_dir = current_dir
        .canonicalize()
        .expect("current directory resolves");
    fs::write(workspace_root.join("AGENTS.md"), "root instructions").expect("root file");
    fs::write(
        workspace_root.join("project/AGENTS.md"),
        "project instructions",
    )
    .expect("project file");
    fs::write(current_dir.join("AGENTS.md"), "nested instructions").expect("nested file");

    let mut agent = test_agent();
    agent
        .load_workspace_instructions(&workspace_root, &current_dir)
        .expect("workspace instructions load");

    let prelude = agent.prepare_turn_prelude("Summarize the change.");
    let instructions = prelude
        .iter()
        .filter(|message| message.text.starts_with("Instructions from "))
        .collect::<Vec<_>>();
    assert_eq!(instructions.len(), 3);
    assert!(
        instructions
            .iter()
            .all(|message| message.role == crate::request_builder::PromptRole::Developer)
    );
    assert!(instructions[0].text.contains(&format!(
        "Instructions from {}",
        workspace_root.join("AGENTS.md").display()
    )));
    assert!(instructions[0].text.ends_with("root instructions"));
    assert!(instructions[1].text.ends_with("project instructions"));
    assert!(instructions[2].text.ends_with("nested instructions"));

    fs::remove_dir_all(workspace_root).expect("test directory should be removed");
}

#[test]
fn config_agents_file_loads_before_workspace_instructions_without_duplicates() {
    let root = agents_test_dir();
    let config_dir = root.join("config");
    let workspace_root = root.join("workspace");
    let current_dir = workspace_root.join("nested");
    fs::create_dir_all(&config_dir).expect("config directory");
    fs::create_dir_all(workspace_root.join(".git")).expect("repo marker");
    fs::create_dir_all(&current_dir).expect("nested workspace");
    fs::write(config_dir.join("AGENTS.md"), "global instructions").expect("global file");
    fs::write(workspace_root.join("AGENTS.md"), "workspace instructions").expect("workspace file");

    let mut agent = test_agent();
    agent
        .load_instruction_files_from(&config_dir, &current_dir)
        .expect("instruction files load");
    agent
        .load_instruction_files_from(&config_dir, &current_dir)
        .expect("reloading is idempotent");

    let instructions = agent
        .prelude
        .iter()
        .filter(|message| message.text.starts_with("Instructions from "))
        .collect::<Vec<_>>();
    assert_eq!(instructions.len(), 2);
    assert!(instructions[0].text.ends_with("global instructions"));
    assert!(instructions[1].text.ends_with("workspace instructions"));

    fs::remove_dir_all(root).expect("test directory should be removed");
}

#[test]
fn missing_workspace_agents_files_leave_the_prelude_unchanged() {
    let workspace_root = agents_test_dir();
    let current_dir = workspace_root.join("nested");
    fs::create_dir_all(&current_dir).expect("nested workspace should be created");

    let mut agent = test_agent();
    let initial_prelude = agent.prelude.clone();
    agent
        .load_workspace_instructions(&workspace_root, &current_dir)
        .expect("missing instructions should not fail");

    assert_eq!(agent.prelude, initial_prelude);
    fs::remove_dir_all(workspace_root).expect("test directory should be removed");
}

#[test]
fn lightweight_turn_prelude_adds_only_runtime_context() {
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base("https://api.openai.com/v1")
            .with_api_key("test"),
    );
    let mut agent = Agent::new(client, "m1", 1, 1);

    let turn_prelude = agent.prepare_turn_prelude("Summarize what this tool does.");

    assert_eq!(agent.current_turn().intent, TurnIntent::Lightweight);
    assert_eq!(turn_prelude.len(), agent.prelude.len() + 1);
    assert_eq!(
        &turn_prelude[..agent.prelude.len()],
        agent.prelude.as_slice()
    );
    let runtime_message = turn_prelude.last().expect("runtime context present");
    assert_eq!(
        runtime_message.role,
        crate::request_builder::PromptRole::Developer
    );
    assert!(runtime_message.text.contains("Runtime context"));
    assert!(runtime_message.text.contains("Current date:"));
    assert!(runtime_message.text.contains("Timezone:"));
    assert!(!runtime_message.text.contains("Current time:"));
}

#[test]
fn selected_skill_injects_exact_turn_scoped_material() {
    let mut agent = test_agent();
    agent
        .register_skill_registry(test_skill_registry())
        .expect("register skills");

    let prelude = agent
        .try_prepare_turn_prelude_with_skills("Please inspect this module.", &["rust-audit".into()])
        .expect("selected skill resolves");
    let material = prelude
        .iter()
        .find(|message| message.origin == PromptMessageOrigin::SkillMaterial)
        .expect("skill material injected");
    assert_eq!(
        material.text,
        test_skill_registry().get("rust-audit").unwrap().content
    );

    let error = agent
        .try_prepare_turn_prelude_with_skills("Use the selected skill.", &["missing".into()])
        .expect_err("unknown selected skill fails");
    assert!(
        error
            .to_string()
            .contains("unknown selected skill: missing")
    );
}

#[test]
fn internal_agent_without_skill_registry_ignores_marker_like_source_text() {
    let mut agent = test_agent();

    let prelude = agent
        .try_prepare_turn_prelude(
            r#"source excerpt: if trimmed.starts_with("@skill(") { return prompt; } @skill(rust-audit)"#,
        )
        .expect("internal prompt must not parse manual skill markers");

    assert!(
        prelude
            .iter()
            .all(|message| message.origin != PromptMessageOrigin::SkillMaterial)
    );
}

#[test]
fn marker_like_user_text_does_not_select_a_skill() {
    let mut agent = test_agent();
    agent
        .register_skill_registry(test_skill_registry())
        .expect("register skills");

    let prelude = agent
        .try_prepare_turn_prelude("Please inspect literal @skill(rust-audit) text.")
        .expect("marker-like text remains ordinary text");

    assert!(
        prelude
            .iter()
            .all(|message| message.origin != PromptMessageOrigin::SkillMaterial)
    );
}

#[test]
fn selected_cjk_skill_injects_selected_material() {
    let content =
        "---\nname: humanizer-zh\ndescription: Humanize Chinese text\n---\n# 完整技能内容\n";
    let registry = Arc::new(
        SkillRegistry::from_entries(vec![crate::skills::SkillEntry {
            name: "humanizer-zh".into(),
            description: "Humanize Chinese text".into(),
            body: "# 完整技能内容".into(),
            content: content.into(),
            location: ".letcode/skills".into(),
            path: PathBuf::from("/workspace/.letcode/skills/humanizer-zh/SKILL.md"),
            base_dir: PathBuf::from("/workspace/.letcode/skills/humanizer-zh"),
        }])
        .expect("skill registry"),
    );
    let mut agent = test_agent();
    agent
        .register_skill_registry(registry)
        .expect("register skills");

    let prelude = agent
        .try_prepare_turn_prelude_with_skills("这个skill是干什么的", &["humanizer-zh".into()])
        .expect("selected skill resolves");
    let materials = prelude
        .iter()
        .filter(|message| message.origin == PromptMessageOrigin::SkillMaterial)
        .collect::<Vec<_>>();

    assert_eq!(materials.len(), 1);
    assert_eq!(
        materials[0].role,
        crate::request_builder::PromptRole::Developer
    );
    assert_eq!(materials[0].text, content);
}

#[test]
fn normalize_session_title_trims_and_strips_wrapping_quotes() {
    assert_eq!(
        normalize_session_title("  \"Fix startup crash in CI\"  ").expect("normalize title"),
        "Fix startup crash in CI"
    );
    assert_eq!(
        normalize_session_title("`Debug flaky transcript tests`\nextra").expect("normalize title"),
        "Debug flaky transcript tests"
    );
}

#[test]
fn session_title_prelude_requires_a_non_conversational_label() {
    assert!(SESSION_TITLE_PRELUDE.contains("准确概括其主题、意图或任务"));
    assert!(SESSION_TITLE_PRELUDE.contains("待命名的内容"));
    assert!(SESSION_TITLE_PRELUDE.contains("描述性标题，而不是对用户消息的直接回应"));
    assert!(!SESSION_TITLE_PRELUDE.contains("问候"));
    assert!(SESSION_TITLE_PRELUDE.contains("只返回标题文本"));
    assert!(SESSION_TITLE_PRELUDE.contains("不要使用引号、项目符号、Markdown、前缀或解释"));
    assert!(SESSION_TITLE_PRELUDE.contains("不超过 80 个字符"));

    let title_agent = test_agent().session_title_agent();
    assert_eq!(title_agent.prelude.len(), 1);
    assert_eq!(
        title_agent.prelude[0].role,
        crate::request_builder::PromptRole::Developer
    );
    assert_eq!(title_agent.prelude[0].text, SESSION_TITLE_PRELUDE);
}

#[test]
fn session_title_agent_has_no_tools_or_history() {
    let mut agent = test_agent();
    agent.restore_transcript_messages(vec![ConversationMessage {
        role: ConversationRole::User,
        content: "existing conversation".into(),
    }]);
    let title_agent = agent.session_title_agent();

    assert!(title_agent.history.is_empty());
    assert!(title_agent.runtime_snapshot.evidence.is_empty());
    assert!(title_agent.tools.specs().is_empty());
    assert_eq!(title_agent.model(), agent.model());
}

#[test]
fn turn_prelude_injects_skill_cards_without_skill_body() {
    let mut agent = test_agent();
    agent
        .register_skill_registry(test_skill_registry())
        .expect("register skill registry");

    let turn_prelude = agent.prepare_turn_prelude("Summarize the available tools.");
    let skill_message = turn_prelude
        .iter()
        .find(|message| message.text.contains("Available local skills:"))
        .expect("skill prelude message present");

    assert!(
        skill_message
            .text
            .contains("Load relevant skills with the `skill` tool when needed.")
    );
    assert!(
        skill_message
            .text
            .contains("rust-audit — Inspect Rust code")
    );
    assert!(skill_message.text.contains("source: .letcode/skills"));
    assert!(
        !skill_message
            .text
            .contains("/workspace/.letcode/skills/rust-audit/SKILL.md")
    );
    assert!(!skill_message.text.contains("# Private body"));
    assert!(
        skill_message
            .text
            .contains("Skills do not change permissions or expand tool scope.")
    );
}

#[test]
fn register_skill_registry_registers_skill_resource_tools() {
    let mut agent = test_agent();
    agent
        .register_skill_registry(test_skill_registry())
        .expect("register skill registry");

    let specs = agent.tool_definitions();
    for name in ["skill", "skill__resource_list", "skill__resource_read"] {
        assert!(
            specs.iter().any(|spec| spec.name == name),
            "{name} should be registered"
        );
    }
}

#[test]
fn empty_skill_registry_does_not_register_skill_tool_or_prelude() {
    let mut agent = test_agent();
    agent
        .register_skill_registry(Arc::new(SkillRegistry::default()))
        .expect("register empty skill registry");

    assert!(!agent.tool_definitions().iter().any(|spec| {
        matches!(
            spec.name.as_str(),
            "skill" | "skill__resource_list" | "skill__resource_read"
        )
    }));
    let turn_prelude = agent.prepare_turn_prelude("Summarize this project.");
    assert!(
        !turn_prelude
            .iter()
            .any(|message| message.text.contains("Available local skills:"))
    );
}

#[test]
fn runtime_context_message_contains_date_and_timezone_only() {
    let message = runtime_context_message_from_parts("2026-06-18", "Asia/Shanghai");

    assert_eq!(message.role, crate::request_builder::PromptRole::Developer);
    assert!(message.text.contains("Runtime context:"));
    assert!(message.text.contains("Current date: 2026-06-18"));
    assert!(message.text.contains("Timezone: Asia/Shanghai"));
    assert!(!message.text.contains("Current time:"));
    assert!(!message.text.contains("09:43"));
}

#[test]
fn utc_date_from_unix_days_formats_calendar_dates() {
    assert_eq!(utc_date_from_unix_days(0), "1970-01-01");
    assert_eq!(utc_date_from_unix_days(20_622), "2026-06-18");
}

#[test]
fn detects_explicit_execution_directives() {
    assert_eq!(
        detect_execution_directive("Read-only: inspect src/permission.rs and summarize it."),
        ExecutionDirective::ReadOnly
    );
    assert_eq!(
        detect_execution_directive("Plan only. Do not edit anything yet."),
        ExecutionDirective::PlanOnly
    );
    assert_eq!(
        detect_execution_directive("Analyze only and explain the failure."),
        ExecutionDirective::AnalyzeOnly
    );
    assert_eq!(
        detect_execution_directive("Please investigate, but do not edit files."),
        ExecutionDirective::DoNotEdit
    );
}

#[tokio::test]
async fn execute_tool_call_blocks_write_tools_under_read_only_directive() {
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base("https://api.openai.com/v1")
            .with_api_key("test"),
    );
    let mut agent = Agent::new(client, "m1", 1, 1);
    agent.turn = TurnRuntimeState::new(
        1,
        WorkflowTurnState::from_user_input("Read-only: inspect and report only."),
    );

    let call = HistoryToolCall {
        call_id: "call-1".into(),
        name: "fs__write".into(),
        arguments_json: r#"{"path":"a.txt","content":"x"}"#.into(),
    };
    let mut events = Vec::new();

    let record = agent
        .execute_tool_call(
            &call,
            &mut |event| {
                events.push(event);
                std::future::ready(Ok(()))
            },
            &mut |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await
        .expect("tool call should complete with visible error");

    assert!(!record.output.ok);
    assert!(
        record
            .output
            .error
            .as_ref()
            .expect("error payload")
            .message
            .contains("read_only directive")
    );
    assert!(matches!(
        events.as_slice(),
        [
            AgentEvent::ToolCallFinished { .. },
            AgentEvent::ToolExecutionSummary(ToolExecutionSummaryEvent {
                status,
                rejection: Some(rejection),
                effect_kind,
                ..
            })
        ] if status == "rejected"
                && rejection == "directive_blocked"
                && effect_kind == "diagnostic"
    ));
    assert_eq!(record.status, ToolExecutionStatus::Rejected);
    assert_eq!(
        record.rejection,
        Some(ToolExecutionRejection::DirectiveBlocked)
    );
    assert_eq!(record.effects.kind, ToolEffectKind::Diagnostic);
}

#[tokio::test]
async fn execute_tool_call_blocks_non_read_only_commands_under_read_only_directive() {
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base("https://api.openai.com/v1")
            .with_api_key("test"),
    );
    let mut agent = Agent::new(client, "m1", 1, 1);
    agent.turn = TurnRuntimeState::new(
        1,
        WorkflowTurnState::from_user_input("Read only. Analyze and report."),
    );

    let call = HistoryToolCall {
        call_id: "call-2".into(),
        name: "shell__exec".into(),
        arguments_json: r#"{"command":"cargo test permission::tests"}"#.into(),
    };

    let record = agent
        .execute_tool_call(&call, &mut |_| std::future::ready(Ok(())), &mut |_| {
            std::future::ready(Ok(PermissionApproval::AllowOnce))
        })
        .await
        .expect("tool call should complete with visible error");

    assert!(!record.output.ok);
    assert!(
        record
            .output
            .error
            .as_ref()
            .expect("error payload")
            .message
            .contains("not read-only compatible")
    );
}

#[tokio::test]
async fn execute_tool_call_emits_finished_event_for_policy_denial() {
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base("https://api.openai.com/v1")
            .with_api_key("test"),
    );
    let mut agent = Agent::new(client, "m1", 1, 1);
    let call = HistoryToolCall {
        call_id: "call-denied".into(),
        name: "shell__exec".into(),
        arguments_json: r#"{"command":"rm -rf target"}"#.into(),
    };
    let mut events = Vec::new();

    let record = agent
        .execute_tool_call(
            &call,
            &mut |event| {
                events.push(event);
                std::future::ready(Ok(()))
            },
            &mut |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await
        .expect("policy denial should be reported as tool output");

    assert!(!record.output.ok);
    assert!(matches!(
        events.as_slice(),
        [
            AgentEvent::ToolCallFinished { ok: false, .. },
            AgentEvent::ToolExecutionSummary(ToolExecutionSummaryEvent {
                status,
                rejection: Some(rejection),
                effect_kind,
                ..
            })
        ] if status == "rejected"
                && rejection == "permission_denied_by_policy"
                && effect_kind == "diagnostic"
    ));
    assert_eq!(record.status, ToolExecutionStatus::Rejected);
    assert_eq!(
        record.rejection,
        Some(ToolExecutionRejection::PermissionDeniedByPolicy)
    );
    assert_eq!(record.effects.kind, ToolEffectKind::Diagnostic);
}

#[tokio::test]
async fn execute_tool_call_invalid_json_emits_finished_event_and_records_rejection() {
    let mut agent = test_agent();
    let call = test_tool_call("fs__write", r#"{"path":"a.txt","content": }"#);
    let mut events = Vec::new();

    let record = agent
        .execute_tool_call(
            &call,
            &mut |event| {
                events.push(event);
                std::future::ready(Ok(()))
            },
            &mut |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await
        .expect("invalid json should still produce a record");

    assert_eq!(record.status, ToolExecutionStatus::Rejected);
    assert_eq!(
        record.rejection,
        Some(ToolExecutionRejection::InvalidJsonArguments)
    );
    assert!(!record.output.ok);
    assert_eq!(record.arguments, None);
    assert_eq!(record.effects.kind, ToolEffectKind::Diagnostic);
    assert!(matches!(
        events.as_slice(),
        [
            AgentEvent::ToolCallFinished { ok: false, .. },
            AgentEvent::ToolExecutionSummary(ToolExecutionSummaryEvent {
                status,
                rejection: Some(rejection),
                effect_kind,
                ..
            })
        ] if status == "rejected"
            && rejection == "invalid_json_arguments"
            && effect_kind == "diagnostic"
    ));
}

#[tokio::test]
async fn audit_event_failures_do_not_fail_tool_execution() {
    let mut agent = test_agent();
    let call = test_tool_call("fs__write", r#"{"path":"a.txt","content": }"#);
    let mut event_count = 0;

    let record = agent
        .execute_tool_call(
            &call,
            &mut |event| {
                assert!(matches!(
                    event,
                    AgentEvent::ToolCallFinished { .. } | AgentEvent::ToolExecutionSummary(_)
                ));
                event_count += 1;
                if matches!(event, AgentEvent::ToolExecutionSummary(_)) {
                    std::future::ready(Err(anyhow!("audit sink failed")))
                } else {
                    std::future::ready(Ok(()))
                }
            },
            &mut |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await
        .expect("audit failure should not fail tool execution");

    assert_eq!(event_count, 2);
    assert_eq!(record.status, ToolExecutionStatus::Rejected);
    assert_eq!(
        record.rejection,
        Some(ToolExecutionRejection::InvalidJsonArguments)
    );
}

#[test]
fn pending_validation_advisory_only_emits_for_write_without_validation() {
    let mut agent = test_agent();
    assert!(agent.pending_validation_advisory().is_none());

    agent.turn.counters.write_effects = 1;
    let advisory = agent
        .pending_validation_advisory()
        .expect("write without validation should emit advisory");
    assert_eq!(advisory.write_effects, 1);
    assert_eq!(advisory.validation_effects, 0);
    assert_eq!(advisory.failed_validation_effects, 0);
    assert!(advisory.message.contains("without running validation"));

    agent.turn.counters.failed_validation_effects = 1;
    let advisory = agent
        .pending_validation_advisory()
        .expect("failed validation should emit advisory");
    assert_eq!(advisory.write_effects, 1);
    assert_eq!(advisory.validation_effects, 0);
    assert_eq!(advisory.failed_validation_effects, 1);
    assert!(advisory.message.contains("validation ran but failed"));

    agent.turn.counters.validation_effects = 1;
    let advisory = agent
        .pending_validation_advisory()
        .expect("failed validation should continue to emit advisory");
    assert_eq!(advisory.validation_effects, 1);
    assert_eq!(advisory.failed_validation_effects, 1);
    assert!(advisory.message.contains("validation ran but failed"));
}

#[test]
fn pending_validation_advisory_includes_child_write_and_validation_failures() {
    let mut agent = test_agent();
    agent.turn.counters.child_write_effects = 2;
    let advisory = agent
        .pending_validation_advisory()
        .expect("child writes without validation should emit advisory");
    assert_eq!(advisory.write_effects, 2);
    assert_eq!(advisory.validation_effects, 0);
    assert!(advisory.message.contains("delegated child work"));

    agent.turn.counters.child_validation_effects = 1;
    agent.turn.counters.child_failed_validation_effects = 1;
    let advisory = agent
        .pending_validation_advisory()
        .expect("child validation failures should emit advisory");
    assert_eq!(advisory.failed_validation_effects, 1);
    assert!(advisory.message.contains("validation failed"));
}

#[test]
fn prepare_turn_prelude_includes_unreconciled_subagent_results() {
    let mut agent = test_agent();
    agent
        .add_evidence(
            EvidenceDraft {
                id: Some("ev-1".into()),
                evidence_kind: crate::evidence::EvidenceKind::Decision,
                title: "subagent result".into(),
                summary: "child completed".into(),
                detail: Some(
                    serde_json::to_string(&crate::subagent::StructuredSubagentResult {
                        status: "completed".into(),
                        summary: "child completed".into(),
                        malformed: false,
                        findings: vec![],
                        files_read: vec![],
                        files_changed: vec![],
                        commands_run: vec![],
                        validation: vec![],
                        blockers: vec![],
                        next_steps: vec![],
                        run_id: "run-1".into(),
                        child_session_id: "child-1".into(),
                        raw_excerpt: None,
                    })
                    .expect("serialize structured result"),
                ),
                source: EvidenceSource::Subagent {
                    run_id: "run-1".into(),
                    child_session_id: "child-1".into(),
                    source_session_id: "child-1".into(),
                    parent_tool: "agent__explore".into(),
                    parent_turn_id: Some("turn-1".into()),
                    parent_session_id: None,
                },
                tags: vec![
                    "explorer".into(),
                    "subagent_result".into(),
                    "unreconciled".into(),
                ],
            }
            .into_record("ev-1".into(), 1, 0)
            .expect("build evidence"),
        )
        .expect("add evidence");

    let prelude = agent.prepare_turn_prelude("Implement next step");
    assert!(prelude.iter().any(|message| {
        message.text.contains("Pending child subagent results")
            && message.text.contains("agent__reconcile")
            && message.text.contains("run-1")
            && message.text.contains("child completed")
            && message.text.contains("child_session_id=child-1")
            && message.text.contains("child transcript navigation")
    }));
}

#[test]
fn pending_subagent_jobs_clear_after_live_reconciliation_evidence() {
    let mut agent = test_agent();
    agent
        .add_evidence(
            EvidenceDraft {
                id: Some("ev-result".into()),
                evidence_kind: crate::evidence::EvidenceKind::Decision,
                title: "subagent result".into(),
                summary: "child completed".into(),
                detail: Some(
                    serde_json::to_string(&crate::subagent::StructuredSubagentResult {
                        status: "completed".into(),
                        summary: "child completed".into(),
                        malformed: false,
                        findings: vec![],
                        files_read: vec![],
                        files_changed: vec![],
                        commands_run: vec![],
                        validation: vec![],
                        blockers: vec![],
                        next_steps: vec![],
                        run_id: "run-1".into(),
                        child_session_id: "child-1".into(),
                        raw_excerpt: None,
                    })
                    .expect("serialize structured result"),
                ),
                source: EvidenceSource::Subagent {
                    run_id: "run-1".into(),
                    child_session_id: "child-1".into(),
                    source_session_id: "child-1".into(),
                    parent_tool: "agent__explore".into(),
                    parent_turn_id: Some("turn-1".into()),
                    parent_session_id: None,
                },
                tags: vec![
                    "explorer".into(),
                    "subagent_result".into(),
                    "unreconciled".into(),
                ],
            }
            .into_record("ev-result".into(), 1, 0)
            .expect("build evidence"),
        )
        .expect("add result evidence");
    assert_eq!(agent.pending_subagent_jobs().len(), 1);

    let record = ToolExecutionRecord::new(
        &test_tool_call(
            tool_names::TOOL_AGENT_RECONCILE,
            r#"{"run_id":"run-1","child_session_id":"child-1","agent_name":"explorer","decision":"accepted","summary":"accepted child result"}"#,
        ),
        Some(json!({
            "run_id": "run-1",
            "child_session_id": "child-1",
            "agent_name": "explorer",
            "decision": "accepted",
            "summary": "accepted child result"
        })),
        crate::permission::ToolPermissionClass::Preview,
        ExecutionDirective::None,
        ToolExecutionStatus::Executed,
        None,
        ToolResult::ok(
            tool_names::TOOL_AGENT_RECONCILE,
            json!({
                "run_id": "run-1",
                "child_session_id": "child-1",
                "agent_name": "explorer",
                "decision": "accepted",
                "summary": "accepted child result",
                "reconciled": true,
                "pending_recording": true
            }),
        ),
    );
    agent.record_tool_effects(&record);
    let evidence = agent
        .remember_tool_evidence(&record)
        .expect("record live reconciliation evidence");
    assert!(
        evidence
            .tags
            .iter()
            .any(|tag| tag == "subagent_reconciliation")
    );
    assert!(evidence.tags.iter().any(|tag| tag == "reconciled"));
    assert_eq!(agent.pending_subagent_jobs().len(), 0);
}

#[test]
fn default_prelude_and_engineering_guidance_frame_non_trivial_work_as_orchestration() {
    assert!(DEFAULT_AGENT_PRELUDE.contains("workflow manager first"));
    assert!(
        DEFAULT_AGENT_PRELUDE
            .contains("Direct execution is for trivial, single-file, clearly bounded work")
    );
    assert!(ENGINEERING_WORKFLOW_PRELUDE.contains("specialist lane is needed"));
    assert!(ENGINEERING_WORKFLOW_PRELUDE.contains("explorer for broad or unknown code search"));
    assert!(ENGINEERING_WORKFLOW_PRELUDE.contains("prefer completed or reconciled sessions"));
    assert!(ENGINEERING_WORKFLOW_PRELUDE.contains("Never reuse cancelled or errored sessions"));
    assert!(ENGINEERING_WORKFLOW_PRELUDE.contains("one active run per specialist role"));
    assert!(ENGINEERING_WORKFLOW_PRELUDE.contains("Delegates do not queue on a busy role"));
    let mut agent = test_agent();
    let prelude = agent.prepare_turn_prelude("Implement a non-trivial feature with validation");
    assert!(
        prelude
            .iter()
            .any(|message| message.text.contains("workflow manager first"))
    );
    assert!(
        prelude
            .iter()
            .any(|message| message.text.contains("specialist lane is needed"))
    );
    assert!(prelude.iter().any(|message| {
        message
            .text
            .contains("explorer for broad or unknown code search")
    }));
}

#[tokio::test]
async fn finalization_does_not_auto_reconcile_unreconciled_subagent_jobs() {
    let mut agent = test_agent();
    agent.prepare_turn_prelude("Follow up on child work");
    agent
        .add_evidence(
            EvidenceDraft {
                id: Some("ev-1".into()),
                evidence_kind: crate::evidence::EvidenceKind::Decision,
                title: "subagent result".into(),
                summary: "child completed".into(),
                detail: Some(
                    serde_json::to_string(&crate::subagent::StructuredSubagentResult {
                        status: "completed".into(),
                        summary: "child completed".into(),
                        malformed: false,
                        findings: vec![],
                        files_read: vec![],
                        files_changed: vec![],
                        commands_run: vec![],
                        validation: vec![],
                        blockers: vec![],
                        next_steps: vec![],
                        run_id: "run-1".into(),
                        child_session_id: "child-1".into(),
                        raw_excerpt: None,
                    })
                    .expect("serialize structured result"),
                ),
                source: EvidenceSource::Subagent {
                    run_id: "run-1".into(),
                    child_session_id: "child-1".into(),
                    source_session_id: "child-1".into(),
                    parent_tool: "agent__explore".into(),
                    parent_turn_id: Some("turn-1".into()),
                    parent_session_id: None,
                },
                tags: vec![
                    "explorer".into(),
                    "subagent_result".into(),
                    "unreconciled".into(),
                ],
            }
            .into_record("ev-1".into(), 1, 0)
            .expect("build evidence"),
        )
        .expect("add evidence");

    let mut events = Vec::new();
    let continued = agent
        .continue_or_finalize_no_tool_reply(
            &mut |event| {
                events.push(event);
                std::future::ready(Ok(()))
            },
            0,
            &mut 0,
        )
        .await
        .expect("finalization succeeds");

    assert!(!continued);
    assert!(!events.iter().any(|event| matches!(
        event,
        AgentEvent::EvidenceRecorded(record)
            if record.tags.iter().any(|tag| tag == "subagent_reconciliation")
    )));
    assert_eq!(agent.pending_subagent_jobs().len(), 1);
}

#[test]
fn child_validation_classification_ignores_not_run_and_counts_object_failures() {
    let (ran, failed) = classify_child_validation_entries(&[
        "cargo test not_run".into(),
        "cargo fmt passed".into(),
        "cargo test failed".into(),
    ]);

    assert_eq!(ran, 2);
    assert_eq!(failed, 1);
}

#[test]
fn auto_continue_bypasses_agent_iteration_and_tool_call_limits() {
    let mut agent = test_agent();
    agent.turn.auto_continue_active = true;

    assert!(agent.ensure_tool_call_budget(4, 1).is_ok());
}

#[test]
fn auto_continue_runtime_flag_resets_for_the_next_turn() {
    let mut agent = test_agent();
    agent.turn.auto_continue_active = true;

    agent.prepare_turn_prelude("next task");

    assert!(!agent.turn.auto_continue_active);
}

#[test]
fn turn_lifecycle_events_capture_expected_snapshot_fields() {
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base("https://api.openai.com/v1")
            .with_api_key("test"),
    );
    let mut agent = Agent::new(client, "m1", 1, 1);

    agent.prepare_turn_prelude("Implement fix in src/agent.rs and run cargo test transcript");
    agent.turn.counters.write_effects = 2;
    agent.turn.counters.validation_effects = 1;
    agent.turn.counters.failed_validation_effects = 0;

    let started = agent.turn_started_event();
    assert_eq!(started.turn_id, 1);
    assert_eq!(started.intent, "engineering");
    assert_eq!(started.directive, "none");
    assert_eq!(started.validation_reminder, "targeted");

    let finalized = agent.turn_finalized_event("completed", 3, 1, true);
    assert_eq!(finalized.turn_id, 1);
    assert_eq!(finalized.outcome, "completed");
    assert_eq!(finalized.tool_call_count, 3);
    assert_eq!(finalized.continuation_count, 1);
    assert_eq!(finalized.write_effects, 2);
    assert_eq!(finalized.validation_effects, 1);
    assert!(finalized.validation_advisory_emitted);
}

#[test]
fn tool_execution_summary_event_omits_full_output_and_captures_audit_fields() {
    let mut agent = test_agent();
    agent.prepare_turn_prelude("Implement fix");
    let record = ToolExecutionRecord::new(
        &test_tool_call("shell__exec", r#"{"command":"cargo test transcript"}"#),
        Some(json!({"command": "cargo test transcript", "path": "src/agent.rs"})),
        crate::permission::ToolPermissionClass::Command,
        ExecutionDirective::None,
        ToolExecutionStatus::Executed,
        None,
        ToolResult::ok(
            "shell__exec",
            json!({"command": "cargo test transcript", "status": 0, "stdout": "lots"}),
        ),
    );

    let summary = agent.tool_execution_summary_event(&record);
    assert_eq!(summary.turn_id, 1);
    assert_eq!(summary.call_id, "call-shell__exec");
    assert_eq!(summary.name, "shell__exec");
    assert_eq!(summary.status, "executed");
    assert_eq!(summary.effect_kind, "validation");
    assert_eq!(summary.primary_path.as_deref(), Some("src/agent.rs"));
    assert_eq!(summary.command.as_deref(), Some("cargo test transcript"));
    assert_eq!(summary.rejection, None);
}
