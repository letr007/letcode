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

#[test]
fn fake_modes_select_only_their_target_protocol() {
    let mut agent = test_agent();

    agent
        .set_fake_client(Some(crate::fake::FakeClient::Codex))
        .expect("responses protocol supports codex fake");
    assert!(
        agent
            .fake_turn_context(crate::fake::FakeClient::Codex)
            .is_some()
    );
    assert!(
        agent
            .fake_turn_context(crate::fake::FakeClient::Anthropic)
            .is_none()
    );

    agent.set_default_protocol(ApiProtocol::Anthropic);
    agent
        .set_fake_client(Some(crate::fake::FakeClient::Anthropic))
        .expect("anthropic protocol supports anthropic fake");
    assert!(
        agent
            .fake_turn_context(crate::fake::FakeClient::Codex)
            .is_none()
    );
    assert!(
        agent
            .fake_turn_context(crate::fake::FakeClient::Anthropic)
            .is_some()
    );

    agent
        .set_fake_client(Some(crate::fake::FakeClient::Auto))
        .expect("anthropic protocol supports auto fake");
    assert!(
        agent
            .fake_turn_context(crate::fake::FakeClient::Codex)
            .is_some()
    );
    assert!(
        agent
            .fake_turn_context(crate::fake::FakeClient::Anthropic)
            .is_some()
    );

    agent.set_default_protocol(ApiProtocol::Completions);
    let previous = agent.fake_client();
    let error = agent
        .set_fake_client(Some(crate::fake::FakeClient::Codex))
        .expect_err("completions protocol rejects fake modes");
    assert!(error.to_string().contains("not supported"));
    assert_eq!(agent.fake_client(), previous);
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
    agent.set_history_for_test(history.clone());
    agent.runtime_snapshot = runtime_snapshot_for_history("active-epoch", &history);
    agent.runtime_snapshot.current_turn_id = Some(1);
    agent.turn.turn_id = 1;
    agent
}

fn append_active_epoch_history(agent: &mut Agent<OpenAIConfig>, history: Vec<HistoryItem>) {
    let existing = agent.history_for_test();
    assert!(history.starts_with(&existing));
    for item in history.into_iter().skip(existing.len()) {
        agent
            .append_history_item(item)
            .expect("history suffix is protocol-compatible");
    }
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
            reasoning_wire: None,
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
            images: Vec::new(),
        },
        HistoryItem::ToolOutput {
            call_id: "call-2".into(),
            output_json: r#"{"value":2}"#.into(),
            images: Vec::new(),
        },
    ]
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

        append_active_epoch_history(&mut agent, active_epoch_history_with_complete_tool_group());
        let append = agent
            .resolved_epoch_preview_for_test(protocol, &[], &tools)
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

        assert!(matches!(
            agent
                .prepare_active_epoch(protocol, &[], &tools)
                .expect("repeat preparation"),
            ActiveEpochPreparation::ColdRequired(ColdRequiredReason::UnsupportedAppendShape)
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
        for frame in agent.runtime_snapshot.frames.iter_mut() {
            if frame.visibility == FrameVisibility::Active && frame.protocol.is_some() {
                if mutation == "mutated" {
                    frame.protocol = Some(ProtocolFrameItem::UserMessage {
                        content: crate::user_content::UserMessageContent::new(
                            "changed",
                            Vec::new(),
                        ),
                    });
                } else {
                    frame.protocol = None;
                }
                break;
            }
        }
        assert!(matches!(
            agent
                .prepare_active_epoch(ApiProtocol::Responses, &[], &tools)
                .expect("non-append mutation requires cold planning"),
            ActiveEpochPreparation::ColdRequired(_)
        ));
        assert_eq!(agent.active_epoch, before);
    }

    let mut agent = active_epoch_agent(vec![HistoryItem::user("seed")]);
    let preview = agent
        .preview_active_epoch(ApiProtocol::Responses, &[], &tools)
        .expect("cold preview");
    agent.commit_active_epoch(preview);
    let before = agent.active_epoch.clone();
    assert!(
        agent
            .append_history_item(HistoryItem::ToolOutput {
                call_id: "orphan".into(),
                output_json: "{}".into(),
                images: Vec::new(),
            })
            .is_err()
    );
    assert_eq!(agent.active_epoch, before);

    let mut agent = active_epoch_agent(vec![HistoryItem::user("seed")]);
    let preview = agent
        .preview_active_epoch(ApiProtocol::Responses, &[], &tools)
        .expect("cold preview");
    agent.commit_active_epoch(preview);
    agent
        .append_history_item(HistoryItem::AssistantToolCalls {
            text: None,
            reasoning_content: None,
            reasoning_wire: None,
            calls: vec![crate::protocol_frames::ProtocolToolCall {
                call_id: "call-1".into(),
                name: "lookup".into(),
                arguments_json: "{}".into(),
            }],
        })
        .expect("assistant tool call opens a valid group");
    assert!(matches!(
        agent
            .prepare_active_epoch(ApiProtocol::Responses, &[], &tools)
            .expect("incomplete suffix requires cold planning"),
        ActiveEpochPreparation::ColdRequired(ColdRequiredReason::SuffixValidationFailed)
    ));
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
fn active_epoch_freezes_effective_evidence_and_allows_first_post_tool_warm_append() {
    let tools = active_epoch_tools();
    let mut agent = active_epoch_agent(vec![HistoryItem::user("seed")]);
    agent
        .add_evidence(test_evidence("ev-before-request", 1))
        .expect("seed evidence");
    crate::request_builder::prompt_plan::reset_plan_call_count();
    let preview = agent
        .preview_active_epoch(ApiProtocol::Responses, &[], &tools)
        .expect("initial cold preview");
    let selected_ids = preview.build.selected_evidence_ids.clone();
    agent.commit_active_epoch(preview);
    assert_eq!(crate::request_builder::prompt_plan::plan_call_count(), 1);
    assert_eq!(
        agent
            .turn
            .frozen_evidence
            .as_ref()
            .expect("effective evidence frozen")
            .selected_ids,
        selected_ids
    );

    agent
        .add_evidence(test_evidence("ev-after-request", 2))
        .expect("tool evidence after freeze");
    append_active_epoch_history(&mut agent, active_epoch_history_with_complete_tool_group());
    let warm = agent
        .resolved_epoch_preview_for_test(ApiProtocol::Responses, &[], &tools)
        .expect("first post-tool request remains warm");
    assert!(matches!(
        warm.transition,
        ActiveEpochTransition::Append { added: 3 }
    ));
    assert_eq!(crate::request_builder::prompt_plan::plan_call_count(), 1);
}

#[test]
fn skill_tool_output_invalidates_warm_projection_and_cold_plan_includes_detached_material() {
    let tools = active_epoch_tools();
    let mut agent = active_epoch_agent(vec![HistoryItem::user("seed")]);
    let cold = agent
        .preview_active_epoch(ApiProtocol::Responses, &[], &tools)
        .expect("initial cold preview");
    agent.commit_active_epoch(cold);

    let skill_content = "detached skill material must reach the provider";
    agent
        .append_history_item(HistoryItem::AssistantToolCalls {
            text: None,
            reasoning_content: None,
            reasoning_wire: None,
            calls: vec![HistoryToolCall {
                call_id: "skill-call-1".into(),
                name: "skill".into(),
                arguments_json: r#"{\"name\":\"rust-audit\"}"#.into(),
            }],
        })
        .expect("skill call frame appends");
    agent
        .append_history_item(HistoryItem::ToolOutput {
            call_id: "skill-call-1".into(),
            output_json: serde_json::to_string(&crate::tool::ToolResult::ok(
                "skill",
                json!({
                    "name": "rust-audit",
                    "content": skill_content,
                }),
            ))
            .expect("skill output serializes"),
            images: Vec::new(),
        })
        .expect("skill output frame appends");

    agent
        .reconcile_loaded_skill_material()
        .expect("persisted skill output reconciles");

    assert!(matches!(
        agent
            .prepare_active_epoch(ApiProtocol::Responses, &[], &tools)
            .expect("skill material change is classified"),
        ActiveEpochPreparation::ColdRequired(ColdRequiredReason::NoActiveEpoch)
    ));
    let rebuilt = agent
        .resolved_epoch_preview_for_test(ApiProtocol::Responses, &[], &tools)
        .expect("cold rebuild includes detached skill material");
    assert!(
        rebuilt
            .build
            .prompt_plan
            .segments
            .iter()
            .any(|segment| segment.text.contains(skill_content))
    );
}

#[test]
fn active_epoch_falls_back_cold_when_suffix_requires_history_reselection() {
    let tools = active_epoch_tools();
    let mut agent = active_epoch_agent(vec![HistoryItem::user("seed")]);
    agent.set_model_catalog(HashMap::from([(
        agent.model().to_string(),
        ModelRequestMetadata {
            context_window: Some(1_500),
            max_output_tokens: Some(128),
            supports_tools: true,
            ..ModelRequestMetadata::default()
        },
    )]));
    let preview = agent
        .preview_active_epoch(ApiProtocol::Responses, &[], &tools)
        .expect("initial cold preview");
    agent.commit_active_epoch(preview);
    agent
        .append_history_item(HistoryItem::assistant("x".repeat(12_000)))
        .expect("append oversized protected suffix");
    assert!(matches!(
        agent
            .prepare_active_epoch(ApiProtocol::Responses, &[], &tools)
            .expect("oversized warm suffix requires cold planning"),
        ActiveEpochPreparation::ColdRequired(ColdRequiredReason::BudgetRequiresColdPlan)
    ));
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

    let frames = agent.protocol_frames_for_test();
    let snapshot = agent.runtime_snapshot.clone();
    agent.commit_active_epoch(
        agent
            .preview_active_epoch(ApiProtocol::Responses, &[], &tools)
            .expect("preview before restore"),
    );
    agent
        .restore_runtime_snapshot(snapshot)
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
    assert!(agent.restore_runtime_snapshot(invalid_restore).is_err());
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

fn responses_reasoning_tool_batch_sse(
    reasoning_text: &str,
    call: serde_json::Value,
) -> &'static str {
    let reasoning_delta = json!({
        "type": "response.reasoning_text.delta",
        "sequence_number": 1,
        "item_id": "reasoning-1",
        "output_index": 0,
        "content_index": 0,
        "delta": reasoning_text,
    });
    let reasoning_done = json!({
        "type": "response.reasoning_text.done",
        "sequence_number": 2,
        "item_id": "reasoning-1",
        "output_index": 0,
        "content_index": 0,
        "text": reasoning_text,
    });
    let response = json!({
        "type": "response.completed", "sequence_number": 3,
        "response": {
            "id": "r-reasoning-tools", "object": "response", "created_at": 1,
            "status": "completed", "background": false, "error": null,
            "incomplete_details": null, "instructions": null, "max_output_tokens": null,
            "model": "m1", "output": [call], "parallel_tool_calls": true,
            "previous_response_id": null, "reasoning": {}, "store": true,
            "temperature": 1, "text": {"format": {"type": "text"}},
            "tool_choice": "auto", "tools": [], "top_p": 1,
            "truncation": "disabled",
            "usage": {"input_tokens": 1, "input_tokens_details": {"cached_tokens": 0}, "output_tokens": 1, "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 2},
            "user": null, "metadata": {}
        }
    });
    let reasoning_delta =
        serde_json::to_string(&reasoning_delta).expect("reasoning delta serializes");
    let reasoning_done = serde_json::to_string(&reasoning_done).expect("reasoning done serializes");
    let response = serde_json::to_string(&response).expect("response serializes");
    sse_response(format!(
        "data: {reasoning_delta}\n\ndata: {reasoning_done}\n\ndata: {response}\n\ndata: [DONE]\n\n"
    ))
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

fn chat_tool_call_fragment_sse() -> &'static str {
    let first = json!({
        "choices": [{
            "index": 0,
            "delta": {
                "content": "same delta",
                "tool_calls": [{
                    "index": 0,
                    "id": "",
                    "type": "function",
                    "function": {"name": "", "arguments": "{"}
                }]
            },
            "finish_reason": null
        }]
    });
    let second = json!({
        "choices": [{
            "index": 0,
            "delta": {
                "tool_calls": [{
                    "index": 0,
                    "id": "call-8",
                    "type": "function",
                    "function": {"name": "workflow__todos", "arguments": "\\\"items\\\":[]}"}
                }]
            },
            "finish_reason": "tool_calls"
        }]
    });
    let first = serde_json::to_string(&first).expect("first response serializes");
    let second = serde_json::to_string(&second).expect("second response serializes");
    sse_response(format!(
        "data: {first}\n\ndata: {second}\n\ndata: [DONE]\n\n"
    ))
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

fn chat_multiple_usage_sse() -> &'static str {
    sse_response(concat!(
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"done\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":2,\"total_tokens\":12,\"prompt_tokens_details\":{\"cached_tokens\":3}}}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":4,\"total_tokens\":14,\"prompt_tokens_details\":{\"cached_tokens\":3}}}\n\n",
        "data: [DONE]\n\n"
    ).to_string())
}

#[tokio::test]
async fn chat_stream_emits_one_final_usage_event_for_multiple_usage_snapshots() {
    let (base_url, _, server) = spawn_chat_completion_server(vec![chat_multiple_usage_sse()]).await;
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base(base_url)
            .with_api_key("test"),
    );
    let mut agent = Agent::new(client, "m1", 1, 0);
    let mut events = Vec::new();

    agent
        .run_oai_comp_stream_async(
            "hello",
            |_| std::future::ready(Ok(())),
            |event| {
                events.push(event);
                std::future::ready(Ok(()))
            },
            |_| std::future::ready(Ok(PermissionApproval::Deny)),
        )
        .await
        .expect("chat stream completes");

    let usage_events = events
        .iter()
        .filter(|event| matches!(event, AgentEvent::TokenUsageUpdated { .. }))
        .collect::<Vec<_>>();
    assert_eq!(usage_events.len(), 1);
    assert!(matches!(
        usage_events[0],
        AgentEvent::TokenUsageUpdated {
            used_tokens: 14,
            input_tokens: 10,
            output_tokens: 4,
            ..
        }
    ));
    server.await.expect("server task should finish");
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
async fn responses_tool_calls_preserve_reasoning_in_live_event_and_follow_up_request() {
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
                responses_reasoning_tool_batch_sse(
                    "inspect the requested file",
                    json!({
                        "type": "function_call", "id": "fc-reasoning", "call_id": "call-reasoning",
                        "name": "workflow__todos", "arguments": r#"{"items":[]}"#,
                        "status": "completed"
                    }),
                )
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
        let input = body["input"]
            .as_array()
            .expect("responses request has input");
        let reasoning_index = input
            .iter()
            .position(|item| item["type"] == "reasoning")
            .expect("follow-up request preserves reasoning input");
        assert_eq!(
            input[reasoning_index]["content"][0]["text"],
            "inspect the requested file"
        );
        let tool_call_index = input
            .iter()
            .position(|item| item["type"] == "function_call")
            .expect("follow-up request preserves function call");
        assert!(reasoning_index < tool_call_index);
        second_socket
            .write_all(responses_final_sse("done").as_bytes())
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
    let mut agent = Agent::new(client, "deepseek-v4-flash", 2, 1);
    let reasoning_batches = Arc::new(Mutex::new(Vec::new()));
    let observed_batches = reasoning_batches.clone();

    let result = agent
        .run_stream_async(
            "hello",
            |_| std::future::ready(Ok(())),
            move |event| {
                if let AgentEvent::AssistantToolCallBatch {
                    reasoning_content, ..
                } = event
                {
                    observed_batches
                        .lock()
                        .expect("reasoning batch lock")
                        .push(reasoning_content);
                }
                std::future::ready(Ok(()))
            },
            |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await
        .expect("responses stream completes");

    assert_eq!(result, "done");
    assert_eq!(
        Arc::try_unwrap(reasoning_batches)
            .expect("all event callbacks have completed")
            .into_inner()
            .expect("reasoning batch lock is not poisoned"),
        vec![Some("inspect the requested file".into())]
    );
    server.await.expect("server task should finish");
}

#[tokio::test]
async fn responses_completed_finalizes_with_unfinished_todos_when_auto_continue_is_disabled() {
    let todo_response = responses_tool_batch_sse(vec![json!({
        "type": "function_call", "id": "fc-todo-pending", "call_id": "call-todo-pending",
        "name": "workflow__todos",
        "arguments": r#"{"items":[{"id":"pending","content":"remain pending","status":"pending"}]}"#,
        "status": "completed"
    })]);
    let (base_url, request_count, server) =
        spawn_chat_completion_server(vec![todo_response, responses_final_sse("done")]).await;
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base(base_url)
            .with_api_key("test"),
    );
    let mut agent = Agent::new(client, "m1", 2, 1);
    let mut internal_continuations = 0;
    let mut scheduled_continuations = 0;
    let mut finalized_outcomes = Vec::new();

    let result = agent
        .run_stream_async(
            "hello",
            |_| std::future::ready(Ok(())),
            |event| {
                match event {
                    AgentEvent::InternalContinuation { .. } => internal_continuations += 1,
                    AgentEvent::AutoContinuationScheduled { .. } => scheduled_continuations += 1,
                    AgentEvent::TurnFinalized(event) => finalized_outcomes.push(event.outcome),
                    _ => {}
                }
                std::future::ready(Ok(()))
            },
            |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await
        .expect("completed response should finalize normally");

    assert_eq!(result, "done");
    assert_eq!(request_count.load(Ordering::SeqCst), 2);
    assert_eq!(internal_continuations, 0);
    assert_eq!(scheduled_continuations, 0);
    assert_eq!(finalized_outcomes, vec!["completed"]);
    assert_eq!(agent.todos()[0].status, TodoStatus::Pending);
    server.await.expect("server task should finish");
}

#[tokio::test]
async fn chat_tool_call_pending_waits_for_complete_descriptor_and_follows_text_delta() {
    #[derive(Debug, PartialEq, Eq)]
    enum Observation {
        Text(String),
        ToolCallPending { call_id: String, name: String },
    }

    let (base_url, _, server) =
        spawn_chat_completion_server(vec![chat_tool_call_fragment_sse(), chat_final_sse("done")])
            .await;
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base(base_url)
            .with_api_key("test"),
    );
    let mut agent = Agent::new(client, "m1", 2, 1);
    let observations = Arc::new(Mutex::new(Vec::new()));
    let delta_observations = observations.clone();
    let event_observations = observations.clone();

    let result = agent
        .run_oai_comp_stream_async(
            "hello",
            move |delta| {
                delta_observations
                    .lock()
                    .expect("observation lock")
                    .push(Observation::Text(delta.to_string()));
                std::future::ready(Ok(()))
            },
            move |event| {
                if let AgentEvent::ToolCallPending { call_id, name } = event {
                    event_observations
                        .lock()
                        .expect("observation lock")
                        .push(Observation::ToolCallPending { call_id, name });
                }
                std::future::ready(Ok(()))
            },
            |_| std::future::ready(Ok(PermissionApproval::Deny)),
        )
        .await
        .expect("chat stream completes");

    assert_eq!(result, "same deltadone");
    let observations = Arc::try_unwrap(observations)
        .expect("all observation callbacks have completed")
        .into_inner()
        .expect("observation lock is not poisoned");
    assert_eq!(
        observations,
        vec![
            Observation::Text("same delta".into()),
            Observation::ToolCallPending {
                call_id: "call-8".into(),
                name: "workflow__todos".into(),
            },
            Observation::Text("done".into()),
        ]
    );
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
        HistoryItem::user("historical request ".repeat(700)),
        HistoryItem::assistant("historical reply ".repeat(700)),
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
        ApiProtocol::Anthropic => unreachable!("pressure test does not use Anthropic"),
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
        ApiProtocol::Anthropic => unreachable!("pressure test does not use Anthropic"),
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
async fn phase2_pressure_compacts_normal_responses_stream() {
    assert_phase2_pressure_compacts_normal_stream(ApiProtocol::Responses).await;
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
    let protected_start = agent.history_for_test().len();
    let prelude = agent.prepare_turn_prelude("current user");
    agent.turn.current_turn_start_index = Some(protected_start);
    agent
        .append_history_item(HistoryItem::user("current user"))
        .expect("stream path appends the current message");
    let history = agent.history_for_test();
    let frames = agent.protocol_frames_for_test();
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
    assert_eq!(agent.history_for_test(), history);
    assert_eq!(agent.protocol_frames_for_test(), frames);
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
    let changed_item = protocol_frame_item_from_history_item(&changed);
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
    let protected_start = agent.history_for_test().len();
    let prelude = agent.prepare_turn_prelude("current user");
    agent.turn.current_turn_start_index = Some(protected_start);
    agent
        .append_history_item(HistoryItem::user("current user"))
        .expect("current message appends");
    agent
        .append_history_item(HistoryItem::AssistantToolCalls {
            text: None,
            reasoning_content: None,
            reasoning_wire: None,
            calls: vec![HistoryToolCall {
                call_id: "pending".into(),
                name: "fs__read".into(),
                arguments_json: "{}".into(),
            }],
        })
        .expect("incomplete current tool group is representable");
    let history = agent.history_for_test();
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
    assert_eq!(agent.history_for_test(), history, "group remains intact");
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
    let protected_start = agent.history_for_test().len();
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
        agent.history_for_test().first(),
        Some(HistoryItem::ContextSummary { .. })
    ));
    let compacted_history = agent.history_for_test();
    assert!(
        compacted_history.len() <= protected_start + 1,
        "pressure compaction must retire at least one pre-protected history item"
    );
    assert!(matches!(
        compacted_history.last(),
        Some(HistoryItem::UserMessage { content }) if content.text.starts_with("protected ")
    ));
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
        max_recovery_attempts: 3,
        initial_delay_secs: 1,
        exponential_backoff: true,
        backoff_multiplier: 2.0,
        jitter_secs: 0,
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

    let history = agent.history_for_test();
    let outputs = history
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
async fn missing_model_metadata_still_parallelizes_parallel_tools() {
    let mut agent = test_agent();
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
        .expect("parallel-capable reads execute concurrently");

    assert_eq!(max_active.load(Ordering::SeqCst), 2);
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
                reasoning_wire: None,
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
        .restore_runtime_snapshot(projected.snapshot)
        .expect("restore persisted output");
    assert_eq!(executions.load(Ordering::SeqCst), 0);
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
        .restore_session_history(agent.history_for_test(), older_snapshot, agent.next_turn_id)
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

#[test]
fn protocol_prefix_digest_is_stable_for_identical_frames_and_sensitive_to_change() {
    let item = |text: &str| crate::protocol_frames::ProtocolItem::context_summary(text.to_string());
    let frames = |texts: &[&str]| {
        texts
            .iter()
            .map(|&t| crate::protocol_frames::ProtocolFrame::derived(item(t)))
            .collect::<Vec<_>>()
    };
    let d1 = protocol_prefix_digest(&frames(&["alpha", "beta"]));
    let d2 = protocol_prefix_digest(&frames(&["alpha", "beta"]));
    let d3 = protocol_prefix_digest(&frames(&["alpha", "gamma"]));
    assert_eq!(d1, d2, "identical history must produce a stable digest");
    assert_ne!(d1, d3, "content change must change the digest");
}

#[test]
fn protocol_prefix_digest_is_length_sensitive_for_large_tool_output() {
    // 工具输出远超 128 字符的有界前缀，长度不同必须仍能区分（不能退化为前缀-only）。
    let mk = |n: usize| {
        crate::protocol_frames::ProtocolFrame::derived(
            crate::protocol_frames::ProtocolItem::ToolOutput {
                call_id: "c1".into(),
                output_json: "x".repeat(n),
                images: Vec::new(),
            },
        )
    };
    assert_ne!(
        protocol_prefix_digest(&[mk(200)]),
        protocol_prefix_digest(&[mk(300)])
    );
}

struct StaticSubagentDelegate {
    result: ToolResult,
}

struct RecordingControlDelegate {
    calls: Arc<Mutex<Vec<String>>>,
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

impl SubagentDelegate<OpenAIConfig> for RecordingControlDelegate {
    fn run_named<'a>(
        &'a self,
        _parent: &'a Agent<OpenAIConfig>,
        agent_name: &'a str,
        _invocation: SubagentInvocation,
    ) -> Pin<Box<dyn Future<Output = Result<ToolResult>> + Send + 'a>> {
        let calls = Arc::clone(&self.calls);
        let agent_name = agent_name.to_string();
        Box::pin(async move {
            calls.lock().expect("calls lock").push(agent_name);
            Ok(ToolResult::ok("agent__explore", json!({"ok": true})))
        })
    }

    fn control<'a>(
        &'a self,
        tool_name: &'a str,
        _args: &'a Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolResult>> + Send + 'a>> {
        let calls = Arc::clone(&self.calls);
        let tool_name = tool_name.to_string();
        Box::pin(async move {
            calls.lock().expect("calls lock").push(tool_name.clone());
            Ok(ToolResult::ok(tool_name, json!({"ok": true})))
        })
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

struct CountingSubagentDelegate {
    active: Arc<AtomicUsize>,
    max_active: Arc<AtomicUsize>,
}

impl SubagentDelegate<OpenAIConfig> for CountingSubagentDelegate {
    fn run_named<'a>(
        &'a self,
        _parent: &'a Agent<OpenAIConfig>,
        agent_name: &'a str,
        _invocation: SubagentInvocation,
    ) -> Pin<Box<dyn Future<Output = Result<ToolResult>> + Send + 'a>> {
        let active = Arc::clone(&self.active);
        let max_active = Arc::clone(&self.max_active);
        let agent_name = agent_name.to_string();
        Box::pin(async move {
            let current = active.fetch_add(1, Ordering::SeqCst) + 1;
            max_active.fetch_max(current, Ordering::SeqCst);
            sleep(Duration::from_millis(20)).await;
            active.fetch_sub(1, Ordering::SeqCst);
            Ok(ToolResult::ok(
                format!("agent__{agent_name}"),
                json!({"ok": true}),
            ))
        })
    }
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
                format!("agent__{agent_name}"),
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
                format!("agent__{agent_name}"),
                json!({"ok": true}),
            ))
        })
    }
}

fn static_delegate(result: ToolResult) -> Arc<dyn SubagentDelegate<OpenAIConfig>> {
    Arc::new(StaticSubagentDelegate { result })
}

#[test]
fn parent_with_delegate_advertises_subagent_control_tools() {
    let mut agent = test_agent();
    agent.set_subagent_delegate(static_delegate(ToolResult::ok(
        "agent__explore",
        json!({"status": "completed"}),
    )));

    let tools = agent
        .tool_definitions_for_test()
        .into_iter()
        .map(|spec| spec.name)
        .collect::<Vec<_>>();
    for name in [
        "agent__jobs",
        "agent__status",
        "agent__wait",
        "agent__cancel",
    ] {
        assert!(tools.iter().any(|tool| tool == name), "missing {name}");
    }
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

#[tokio::test]
async fn disabled_model_parallel_tool_calls_keep_subagents_sequential() {
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
    agent.set_subagent_delegate(Arc::new(CountingSubagentDelegate {
        active: Arc::clone(&active),
        max_active: Arc::clone(&max_active),
    }));
    let calls = vec![
        test_tool_call("agent__explore", r#"{"task":"inspect"}"#),
        test_tool_call(
            "agent__fixer",
            r#"{"task":"change","owned_paths":["src/agent.rs"]}"#,
        ),
    ];
    agent
        .append_assistant_tool_calls("", &calls)
        .expect("append tool calls");

    agent
        .execute_tool_calls_and_record(&calls, &mut |_| async { Ok(()) }, &mut |_| async {
            Ok(PermissionApproval::AllowOnce)
        })
        .await
        .expect("subagents execute sequentially");

    assert_eq!(max_active.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn contiguous_different_role_subagents_overlap_and_return_in_model_order() {
    let mut agent = test_agent();
    let started = Arc::new(Mutex::new(Vec::new()));
    agent.set_subagent_delegate(Arc::new(OverlapSubagentDelegate {
        barrier: Arc::new(Barrier::new(2)),
        started: Arc::clone(&started),
    }));
    let calls = vec![
        test_tool_call("agent__explore", r#"{"task":"inspect"}"#),
        test_tool_call(
            "agent__fixer",
            r#"{"task":"change","owned_paths":["src/agent.rs"]}"#,
        ),
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
    let history = agent.history_for_test();
    let outputs = history
        .iter()
        .filter_map(|item| match item {
            HistoryItem::ToolOutput { call_id, .. } => Some(call_id.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(outputs, vec!["call-agent__explore", "call-agent__fixer"]);
}

#[tokio::test]
async fn subagent_control_tool_is_a_barrier_between_subagent_batches() {
    let mut agent = test_agent();
    let calls_seen = Arc::new(Mutex::new(Vec::new()));
    agent.set_subagent_delegate(Arc::new(RecordingControlDelegate {
        calls: Arc::clone(&calls_seen),
    }));
    let calls = vec![
        HistoryToolCall {
            call_id: "call-explore-first".into(),
            name: "agent__explore".into(),
            arguments_json: r#"{"task":"first"}"#.into(),
        },
        test_tool_call("agent__jobs", r#"{}"#),
        HistoryToolCall {
            call_id: "call-explore-second".into(),
            name: "agent__explore".into(),
            arguments_json: r#"{"task":"second"}"#.into(),
        },
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

    assert_eq!(
        *calls_seen.lock().expect("calls lock"),
        vec!["explorer", "agent__jobs", "explorer"]
    );
    let history = agent.history_for_test();
    let outputs = history
        .iter()
        .filter_map(|item| match item {
            HistoryItem::ToolOutput { call_id, .. } => Some(call_id.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        outputs,
        vec![
            "call-explore-first",
            "call-agent__jobs",
            "call-explore-second"
        ]
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
        test_tool_call(
            "agent__fixer",
            r#"{"task":"change","owned_paths":["src/agent.rs"]}"#,
        ),
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

    let history = agent.history_for_test();
    let outputs = history
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
        agent.history_for_test().last(),
        Some(HistoryItem::ToolOutput {
            call_id,
            output_json,
            ..
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
async fn delegated_structured_subagent_results_are_recorded_as_evidence() {
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
                "next_steps": ["continue parent task"],
                "run_id": "run-structured-1",
                "child_session_id": "child-structured-1"
            }
        }),
    )));

    let call = test_tool_call(
        "agent__fixer",
        r#"{"task":"implement bounded fix","owned_paths":["src/agent.rs"]}"#,
    );
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
    )));
}

#[tokio::test]
async fn wait_and_background_delivery_apply_subagent_effects_once_per_turn() {
    let mut agent = test_agent();
    agent.prepare_turn_prelude("Reconcile one background result");
    let result = json!({
        "run_id": "run-dedupe-1",
        "child_session_id": "child-dedupe-1",
        "agent_name": "fixer",
        "status": "completed",
        "summary": "implemented and validated",
        "structured_result": {
            "status": "completed",
            "summary": "implemented and validated",
            "malformed": false,
            "findings": [],
            "files_read": [],
            "files_changed": ["src/agent.rs"],
            "commands_run": ["cargo test"],
            "validation": ["cargo test passed"],
            "blockers": [],
            "next_steps": [],
            "run_id": "run-dedupe-1",
            "child_session_id": "child-dedupe-1"
        }
    });
    let wait_record = ToolExecutionRecord {
        call_id: "wait-dedupe-1".into(),
        tool_name: tool_names::TOOL_AGENT_WAIT.into(),
        arguments: Some(json!({"run_id": "run-dedupe-1"})),
        permission_class: crate::permission::ToolPermissionClass::Preview,
        directive: ExecutionDirective::None,
        status: ToolExecutionStatus::Executed,
        rejection: None,
        output: ToolResult::ok("agent__wait", result.clone()),
        effects: ToolEffects {
            kind: ToolEffectKind::Read,
            primary_path: None,
            edited_paths: Vec::new(),
            command: None,
        },
    };
    agent.record_tool_effects(&wait_record);
    assert_eq!(agent.child_effect_counts_for_test(), (1, 1, 0));
    let _ = agent
        .remember_tool_evidence(&wait_record)
        .expect("remember wait evidence");

    let summary = crate::subagent::SubagentRunSummary {
        run_id: "run-dedupe-1".into(),
        child_session_id: "child-dedupe-1".into(),
        agent_name: "fixer".into(),
        status: crate::subagent::SubagentStatus::Completed,
        failure_kind: None,
        summary: "implemented and validated".into(),
        structured_result: serde_json::from_value(result["structured_result"].clone())
            .expect("structured result"),
    };
    agent
        .install_background_subagent_result(&summary)
        .expect("background result installs");
    assert_eq!(agent.child_effect_counts_for_test(), (1, 1, 0));
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
            context_window: Some(4096),
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
    assert_eq!(b1.budget.context_window_tokens, 4096);

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
fn reasoning_effort_selection_is_scoped_to_each_model() {
    let mut agent = test_agent();
    agent.set_model_catalog(HashMap::from([
        (
            "m1".into(),
            ModelRequestMetadata {
                supports_reasoning: true,
                reasoning_effort: Some(ModelReasoningEffort::Medium),
                reasoning_efforts: vec![ModelReasoningEffort::Medium, ModelReasoningEffort::High],
                ..Default::default()
            },
        ),
        (
            "m2".into(),
            ModelRequestMetadata {
                supports_reasoning: true,
                reasoning_effort: Some(ModelReasoningEffort::Low),
                reasoning_efforts: vec![ModelReasoningEffort::Low, ModelReasoningEffort::High],
                ..Default::default()
            },
        ),
    ]));

    agent
        .set_reasoning_effort(ModelReasoningEffort::High)
        .expect("set first model effort");
    agent.set_model("m2");
    assert_eq!(agent.reasoning_effort(), Some(ModelReasoningEffort::Low));

    agent.set_model("m1");
    assert_eq!(agent.reasoning_effort(), Some(ModelReasoningEffort::High));
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
            reasoning_wire: None,
            calls: vec![test_tool_call("fs__read", r#"{"path":"src/main.rs"}"#)],
        })
        .expect("tool call append succeeds");
    agent
        .append_history_item(HistoryItem::ToolOutput {
            call_id: "call-fs__read".into(),
            output_json: r#"{"ok":true}"#.into(),
            images: Vec::new(),
        })
        .expect("tool output append succeeds");

    assert_eq!(
        crate::protocol_frames::history_items_from_frames(&agent.protocol_frames_for_test()),
        agent.history_for_test()
    );
    assert_eq!(
        agent.runtime_snapshot.frames.len(),
        agent.protocol_frames_for_test().len()
    );
}

/// Phase 0 baseline: pin the live in-memory consistency of the three protocol
/// representations (history / protocol_frames / snapshot.active_protocol_frames)
/// before the event-sourcing refactor. Item-wise, not just length.
fn assert_three_way_protocol_consistency(agent: &Agent<OpenAIConfig>) {
    let frames = agent.protocol_frames_for_test();
    let history = agent.history_for_test();
    let active = agent.runtime_snapshot.active_protocol_frames();
    assert_eq!(
        crate::protocol_frames::history_items_from_frames(&frames),
        history,
        "history must equal protocol_frames payload"
    );
    assert_eq!(
        frames, active,
        "protocol_frames must equal snapshot.active_protocol_frames"
    );
    assert_eq!(
        crate::protocol_frames::history_items_from_frames(&active),
        history,
        "history must equal snapshot.active_protocol_frames payload"
    );
}

#[test]
fn session_state_consistency_live_append_three_way() {
    let mut agent = test_agent();
    agent
        .append_history_item(HistoryItem::user("hello"))
        .expect("user append");
    agent
        .append_history_item(HistoryItem::AssistantToolCalls {
            text: Some("working".into()),
            reasoning_content: None,
            reasoning_wire: None,
            calls: vec![test_tool_call("fs__read", r#"{"path":"src/main.rs"}"#)],
        })
        .expect("tool call append");
    agent
        .append_history_item(HistoryItem::ToolOutput {
            call_id: "call-fs__read".into(),
            output_json: r#"{"ok":true}"#.into(),
            images: Vec::new(),
        })
        .expect("tool output append");
    agent
        .append_history_item(HistoryItem::assistant("done"))
        .expect("assistant append");
    assert_three_way_protocol_consistency(&agent);
}

#[test]
fn session_state_consistency_restore_three_way() {
    let mut agent = test_agent();
    let history = vec![HistoryItem::user("seed"), HistoryItem::assistant("answer")];
    agent
        .restore_session_history(history.clone(), Vec::new(), 3)
        .expect("restore session history");
    assert_three_way_protocol_consistency(&agent);
}

#[test]
fn restore_rejects_snapshot_with_protocol_invalid_internal_stream() {
    // RuntimeSnapshot is the single source of truth: restore must validate the
    // snapshot's own active protocol stream, not some external frame list. A
    // snapshot carrying an orphan tool output must be rejected outright.
    let mut agent = test_agent();
    let mut snapshot = RuntimeSnapshot::new("main");
    snapshot.frames = runtime_frames_for_history(&[HistoryItem::ToolOutput {
        call_id: "orphan".into(),
        output_json: "{}".into(),
        images: Vec::new(),
    }]);
    assert!(
        agent.restore_runtime_snapshot(snapshot).is_err(),
        "restore must reject a snapshot whose own active stream is invalid"
    );
}

#[test]
fn resume_clears_proc_local_identity_and_observation_state() {
    let tools = active_epoch_tools();
    let seed = || {
        let mut agent = active_epoch_agent(vec![HistoryItem::user("seed")]);
        let preview = agent
            .preview_active_epoch(ApiProtocol::Responses, &[], &tools)
            .expect("cold preview");
        agent.commit_active_epoch(preview.clone());
        agent.install_provider_usage_anchor_for_test(provider_usage(8_000));
        agent.commit_final_logical_request(&preview.build);
        agent
    };

    let mut agent = seed();
    assert!(agent.active_epoch.is_some());
    assert!(agent.provider_usage_anchor_for_test().is_some());
    assert!(agent.has_set_logical_request_observation_for_test());

    agent
        .restore_session_history(vec![HistoryItem::user("seeded")], Vec::new(), 4)
        .expect("restore session history");
    assert!(agent.active_epoch.is_none(), "resume wipes active epoch");
    assert!(
        agent.provider_usage_anchor_for_test().is_none(),
        "resume wipes provider usage anchor"
    );
    assert!(
        !agent.has_set_logical_request_observation_for_test(),
        "resume wipes logical request observation"
    );

    let mut agent = seed();
    agent.reset_for_new_session();
    assert!(agent.active_epoch.is_none(), "reset wipes active epoch");
    assert!(
        agent.provider_usage_anchor_for_test().is_none(),
        "reset wipes provider usage anchor"
    );
    assert!(
        !agent.has_set_logical_request_observation_for_test(),
        "reset wipes logical request observation"
    );
}

#[tokio::test]
async fn session_state_consistency_after_manual_compaction_three_way() {
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
    agent.runtime_snapshot = runtime_snapshot_for_history("main", &agent.history_for_test());
    agent.turn.turn_id = 10;
    agent.turn.current_turn_start_index = Some(2);
    agent.runtime_snapshot.current_turn_id = Some(10);
    agent.runtime_snapshot.current_segment_id = Some(3);

    let outcome = agent
        .compact_session_async(|_| std::future::ready(Ok(())))
        .await
        .expect("compacts older turns");
    assert!(matches!(outcome, ManualCompactionOutcome::Compacted { .. }));
    assert_three_way_protocol_consistency(&agent);
    server.await.expect("server task should finish");
}

#[test]
fn session_state_consistency_journal_resume_three_way() {
    // Oracle Phase-1 priority: real transcript-journal resume projection path
    // (TranscriptRecorder -> read_records -> project_runtime_restore_snapshot ->
    // install into Agent) must also yield three-way consistency.
    let dir = agents_test_dir();
    let mut rec = TranscriptRecorder::create(&dir).expect("create recorder");
    rec.record_session_started("m1").expect("session started");
    let session_id = rec.session_id().to_string();
    let path = rec.path().to_path_buf();
    rec.record_user_message("hello").expect("user message");
    rec.record_turn_started(TurnStartedEvent {
        turn_id: 1,
        intent: "engineering".into(),
        directive: "none".into(),
        validation_reminder: "none".into(),
    })
    .expect("turn started");
    rec.record_assistant_tool_call_batch(
        Some("working".into()),
        None,
        None,
        vec![HistoryToolCall {
            call_id: "call-1".into(),
            name: "fs__read".into(),
            arguments_json: r#"{"path":"src/main.rs"}"#.into(),
        }],
    )
    .expect("tool call batch");
    rec.record_tool_call_finished(
        "call-1",
        "fs__read",
        true,
        ToolResult::ok("fs__read", json!({"ok": true})),
    )
    .expect("tool call finished");
    rec.record_assistant_message("done")
        .expect("assistant message");
    rec.record_turn_finalized(TurnFinalizedEvent {
        turn_id: 1,
        outcome: "completed".into(),
        tool_call_count: 1,
        continuation_count: 0,
        write_effects: 0,
        validation_effects: 0,
        failed_validation_effects: 0,
        validation_advisory_emitted: false,
    })
    .expect("turn finalized");
    drop(rec);

    let records = read_records(&path).expect("read records");
    let projected = project_runtime_restore_snapshot(
        session_id,
        records,
        SessionContextCursor {
            branch_id: Some(ROOT_CONTEXT_BRANCH_ID.into()),
            leaf_sequence: None,
        },
        &[],
    )
    .expect("project runtime restore snapshot");

    let mut agent = test_agent();
    agent
        .restore_runtime_snapshot(projected.snapshot)
        .expect("restore runtime snapshot");
    assert_three_way_protocol_consistency(&agent);

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn append_history_item_is_atomic_when_protocol_validation_fails() {
    let mut agent = test_agent();
    agent
        .append_history_item(HistoryItem::user("hello"))
        .expect("user append succeeds");

    let history_before = agent.history_for_test();
    let frames_before = agent.protocol_frames_for_test();
    let snapshot_before = agent.runtime_snapshot.clone();

    let error = agent
        .append_history_item(HistoryItem::ToolOutput {
            call_id: "call-orphan".into(),
            output_json: "{}".into(),
            images: Vec::new(),
        })
        .expect_err("orphan tool output must fail");

    assert!(error.to_string().contains("orphan tool output"));
    assert_eq!(agent.history_for_test(), history_before);
    assert_eq!(agent.protocol_frames_for_test(), frames_before);
    assert_eq!(agent.runtime_snapshot, snapshot_before);
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
    let invalid_metadata = ModelRequestMetadata {
        effective_input_limit_tokens: Some(0),
        ..Default::default()
    };
    agent.set_model_catalog(HashMap::from([(
        String::from("invalid-model"),
        invalid_metadata,
    )]));
    let target_snapshot = runtime_snapshot_for_history(
        ROOT_CONTEXT_BRANCH_ID,
        &[HistoryItem::user("target session")],
    );
    let model = agent.model.clone();
    let history = agent.history_for_test();
    let protocol_frames = agent.protocol_frames_for_test();
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
    assert_eq!(agent.history_for_test(), history);
    assert_eq!(agent.protocol_frames_for_test(), protocol_frames);
    assert_eq!(agent.runtime_snapshot, runtime_snapshot);
    assert_eq!(agent.current_turn_id(), turn_id);
    assert_eq!(agent.next_turn_id, next_turn_id);
}

#[test]
fn uncatalogued_candidate_session_usage_keeps_tools_enabled() {
    let agent = test_agent();
    let snapshot = runtime_snapshot_for_history(
        ROOT_CONTEXT_BRANCH_ID,
        &[HistoryItem::user("target session")],
    );

    let (usage, composition) = agent
        .candidate_session_usage_with_composition("uncatalogued-model", &snapshot)
        .expect("uncatalogued model uses backward-compatible defaults");

    assert!(usage.used_tokens > 0);
    assert!(
        composition
            .iter()
            .any(|entry| entry.category == "tools" && entry.estimated_tokens > 0)
    );
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
        .restore_runtime_snapshot(snapshot.clone())
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
        .restore_runtime_snapshot(snapshot)
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
                reasoning_wire: None,
                calls: vec![HistoryToolCall {
                    call_id: "call-pending".into(),
                    name: "fs__read".into(),
                    arguments_json: r#"{"path":"src/main.rs"}"#.into(),
                }],
            },
        ])
        .expect("active incomplete history");
    agent.runtime_snapshot = runtime_snapshot_for_history("main", &agent.history_for_test());
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
    agent.runtime_snapshot = runtime_snapshot_for_history("main", &agent.history_for_test());
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
            reasoning_wire: None,
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
    agent.runtime_snapshot = runtime_snapshot_for_history("main", &agent.history_for_test());
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
    agent.runtime_snapshot = runtime_snapshot_for_history("main", &agent.history_for_test());
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
    agent.runtime_snapshot = runtime_snapshot_for_history("main", &agent.history_for_test());
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
fn default_preserve_recent_budget_uses_reserve_aware_pi_style_20k_tail() {
    assert_eq!(default_preserve_recent_budget(1_000), 1_000);
    assert_eq!(default_preserve_recent_budget(20_000), 3_616);
    assert_eq!(default_preserve_recent_budget(100_000), 20_000);
}

#[test]
fn render_compaction_tool_output_caps_large_payloads() {
    let rendered = compaction::describe_history_item(&HistoryItem::ToolOutput {
        call_id: "call-big".into(),
        output_json: large_tool_output_json("stdout"),
        images: Vec::new(),
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
        images: Vec::new(),
    });

    assert!(rendered.contains("stripped media/blob-like field"));
    assert!(rendered.contains("kept text"));
    assert!(!rendered.contains("blob:https://example.invalid/123"));
    assert!(!rendered.contains(&"A".repeat(128)));
}

#[test]
fn render_compaction_prompt_preserves_complete_previous_summary() {
    let previous_summary = format!(
        "{}\nTAIL-CURRENT-STATE-MUST-SURVIVE",
        "old checkpoint detail ".repeat(600)
    );

    let prompt = compaction::render_compaction_prompt(
        Some(&previous_summary),
        &[HistoryItem::assistant("new history")],
    );

    assert!(previous_summary.chars().count() > 8_000);
    assert!(prompt.contains(&previous_summary));
    assert!(prompt.contains("TAIL-CURRENT-STATE-MUST-SURVIVE"));
    assert!(!prompt.contains("先前摘要已为压缩而缩减"));
}

#[test]
fn render_compaction_prompt_includes_every_retired_history_item() {
    let items = (0..20)
        .map(|index| HistoryItem::ToolOutput {
            call_id: format!("call-{index}"),
            output_json: large_tool_output_json("stdout"),
            images: Vec::new(),
        })
        .collect::<Vec<_>>();

    let rendered = compaction::render_compaction_history(&items);

    assert!(rendered.contains("call-0"));
    assert!(rendered.contains("call-19"));
    assert_eq!(rendered.lines().count(), items.len());
}

#[tokio::test]
async fn ordinary_request_build_uses_installed_runtime_snapshot_only() {
    let mut agent = test_agent();
    agent.set_history_for_test(vec![HistoryItem::user("EXTERNAL-TRANSCRIPT-CONTENT")]);
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
        BuiltRequest::Anthropic(_) => panic!("expected responses request"),
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
async fn responses_stream_preserves_recovery_state_when_iteration_budget_is_exhausted() {
    let interrupted = sse_response(
        "data: {\"type\":\"response.output_text.delta\",\"sequence_number\":1,\"item_id\":\"msg-1\",\"output_index\":0,\"content_index\":0,\"delta\":\"partial\"}\n\ndata: {malformed}\n\n"
            .into(),
    );
    let (base_url, request_count, server) = spawn_chat_completion_server(vec![interrupted]).await;
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base(base_url)
            .with_api_key("test"),
    );
    let mut agent = Agent::new(client, "m1", 1, 4);
    let mut issues = 0;
    let mut recovery_continuations = 0;
    let mut assistant_messages = Vec::new();
    let mut deltas = Vec::new();

    let error = agent
        .run_stream_async(
            "hello",
            |delta| {
                deltas.push(delta.to_string());
                std::future::ready(Ok(()))
            },
            |event| {
                match event {
                    AgentEvent::AssistantMessage { content } => assistant_messages.push(content),
                    AgentEvent::ModelStreamIssue { .. } => issues += 1,
                    AgentEvent::InternalContinuation { .. } => recovery_continuations += 1,
                    _ => {}
                }
                std::future::ready(Ok(()))
            },
            |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await
        .expect_err("the next model request should be rejected by the existing iteration gate");

    assert!(
        error
            .to_string()
            .contains("stopped: too many agent iterations (max 1)")
    );
    assert_eq!(deltas, vec!["partial"]);
    assert_eq!(assistant_messages, vec!["partial"]);
    assert_eq!(issues, 1);
    assert_eq!(recovery_continuations, 1);
    assert!(
        agent
            .history_for_test()
            .iter()
            .any(|item| matches!(item, HistoryItem::AssistantText { text } if text == "partial"))
    );
    assert!(
        agent
            .history_for_test()
            .iter()
            .any(|item| matches!(item, HistoryItem::InternalContinuation { .. }))
    );
    assert_eq!(request_count.load(Ordering::SeqCst), 1);
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
async fn compatible_chat_stream_preserves_recovery_state_when_iteration_budget_is_exhausted() {
    let interrupted = sse_response(
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"partial\"},\"finish_reason\":null}]}\n\ndata: {malformed}\n\n"
            .into(),
    );
    let (base_url, request_count, server) = spawn_chat_completion_server(vec![interrupted]).await;
    let client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base(base_url)
            .with_api_key("test"),
    );
    let mut agent = Agent::new(client, "m1", 1, 4);
    let mut issues = 0;
    let mut recovery_continuations = 0;
    let mut assistant_messages = Vec::new();
    let mut deltas = Vec::new();

    let error = agent
        .run_oai_comp_stream_async(
            "hello",
            |delta| {
                deltas.push(delta.to_string());
                std::future::ready(Ok(()))
            },
            |event| {
                match event {
                    AgentEvent::AssistantMessage { content } => assistant_messages.push(content),
                    AgentEvent::ModelStreamIssue { .. } => issues += 1,
                    AgentEvent::InternalContinuation { .. } => recovery_continuations += 1,
                    _ => {}
                }
                std::future::ready(Ok(()))
            },
            |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await
        .expect_err("the next model request should be rejected by the existing iteration gate");

    assert!(
        error
            .to_string()
            .contains("stopped: too many agent iterations (max 1)")
    );
    assert_eq!(deltas, vec!["partial"]);
    assert_eq!(assistant_messages, vec!["partial"]);
    assert_eq!(issues, 1);
    assert_eq!(recovery_continuations, 1);
    assert!(
        agent
            .history_for_test()
            .iter()
            .any(|item| matches!(item, HistoryItem::AssistantText { text } if text == "partial"))
    );
    assert!(
        agent
            .history_for_test()
            .iter()
            .any(|item| matches!(item, HistoryItem::InternalContinuation { .. }))
    );
    assert_eq!(request_count.load(Ordering::SeqCst), 1);
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
            .history_for_test()
            .iter()
            .any(|item| matches!(item, HistoryItem::AssistantText { text } if text == "partial"))
    );
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
    assert!(!agent.history_for_test().iter().any(|item| matches!(
        item,
        HistoryItem::AssistantToolCalls { calls, .. }
            if calls.iter().any(|call| call.call_id == "call-interrupted")
    )));
    assert_eq!(request_count.load(Ordering::SeqCst), 2);
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
                .history_for_test()
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
async fn workflow_todos_tool_updates_todo_state() {
    let mut agent = test_agent();
    agent.set_permission_mode(PermissionMode::Auto);
    let call = HistoryToolCall {
            call_id: "call-todos".into(),
            name: "workflow__todos".into(),
            arguments_json: r#"{"items":[{"id":"t1","content":"first","status":"pending"},{"id":"t2","content":"done","status":"completed"}]}"#.into(),
        };

    let mut approvals = 0usize;
    agent
        .execute_tool_call(&call, &mut |_| std::future::ready(Ok(())), &mut |_| {
            approvals += 1;
            std::future::ready(Ok(PermissionApproval::Deny))
        })
        .await
        .expect("todo control tool should succeed without reviewer or human approval");

    assert_eq!(approvals, 0);
    assert_eq!(agent.todos().len(), 2);
    assert_eq!(agent.todos()[0].status, TodoStatus::Pending);
    assert_eq!(agent.todos()[1].status, TodoStatus::Completed);
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
async fn matching_grant_allows_high_risk_command_without_reasking() {
    let mut agent = test_agent();
    let command = "curl --version";
    agent
        .permission_session
        .lock()
        .expect("permission session")
        .grant(crate::permission::PermissionResource::Exact {
            tool: "shell__exec".into(),
            value: command.into(),
        });
    let call = HistoryToolCall {
        call_id: "call-granted-high-risk".into(),
        name: "shell__exec".into(),
        arguments_json: json!({"command": command}).to_string(),
    };
    let mut approval_requested = false;

    let record = agent
        .execute_tool_call(&call, &mut |_| std::future::ready(Ok(())), &mut |_| {
            approval_requested = true;
            std::future::ready(Ok(PermissionApproval::Deny))
        })
        .await
        .expect("session grant should admit previously blacklisted commands");

    assert!(!approval_requested);
    assert_eq!(record.status, ToolExecutionStatus::Executed);
    assert_ne!(
        record.rejection,
        Some(ToolExecutionRejection::PermissionDeniedByPolicy)
    );
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

struct MockAutoReviewService {
    approval: Mutex<PermissionApproval>,
    calls: AtomicUsize,
    child_session_id: String,
}

struct CapturingAutoReviewService {
    approval: PermissionApproval,
    calls: AtomicUsize,
    last_request: Mutex<Option<PermissionRequest>>,
}

impl AutoReviewService<OpenAIConfig> for CapturingAutoReviewService {
    fn review<'a>(
        &'a self,
        _parent: &'a Agent<OpenAIConfig>,
        request: PermissionRequest,
        _user_goal: Option<String>,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = anyhow::Result<AutoReviewResolution>> + Send + 'a>,
    > {
        Box::pin(async move {
            self.calls.fetch_add(1, Ordering::SeqCst);
            *self.last_request.lock().expect("request lock") = Some(request);
            Ok(AutoReviewResolution {
                approval: self.approval,
                reason: "captured".into(),
                risk: Some("low".into()),
                approval_label: "once",
                reviewer_child_session_id: "reviewer-child-capturing".into(),
            })
        })
    }

    fn clear_sticky(&self) {}
}

impl AutoReviewService<OpenAIConfig> for MockAutoReviewService {
    fn review<'a>(
        &'a self,
        _parent: &'a Agent<OpenAIConfig>,
        _request: PermissionRequest,
        _user_goal: Option<String>,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = anyhow::Result<AutoReviewResolution>> + Send + 'a>,
    > {
        Box::pin(async move {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let approval = *self.approval.lock().expect("approval lock");
            let approval_label = match approval {
                PermissionApproval::AllowOnce => "once",
                PermissionApproval::AllowAlways => "always",
                PermissionApproval::Deny => "deny",
            };
            Ok(AutoReviewResolution {
                approval,
                reason: format!("mock-{approval_label}"),
                risk: Some("low".into()),
                approval_label,
                reviewer_child_session_id: self.child_session_id.clone(),
            })
        })
    }

    fn clear_sticky(&self) {}
}

#[tokio::test]
async fn auto_mode_child_inherits_reviewer_service() {
    let mut parent = test_agent();
    parent.set_permission_mode(PermissionMode::Auto);
    let service = Arc::new(MockAutoReviewService {
        approval: Mutex::new(PermissionApproval::AllowOnce),
        calls: AtomicUsize::new(0),
        child_session_id: "reviewer-child-inherited".into(),
    });
    parent.set_auto_review_service(Some(service.clone()));
    let mut child = AgentFactory::create_child(&parent, &AgentTemplate::fixer());

    let path = std::env::temp_dir().join(format!(
        "letcode-auto-review-child-{}.txt",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time")
            .as_nanos()
    ));
    let call = HistoryToolCall {
        call_id: "call-auto-child".into(),
        name: "fs__write".into(),
        arguments_json: json!({"path": path, "content": "ok"}).to_string(),
    };
    let mut human_approvals = 0usize;
    let record = child
        .execute_tool_call(&call, &mut |_| std::future::ready(Ok(())), &mut |_| {
            human_approvals += 1;
            std::future::ready(Ok(PermissionApproval::Deny))
        })
        .await
        .expect("child auto approval executes");

    assert_eq!(human_approvals, 0);
    assert_eq!(service.calls.load(Ordering::SeqCst), 1);
    assert_eq!(record.status, ToolExecutionStatus::Executed);
    assert!(record.output.ok, "{:?}", record.output.error);
    let _ = std::fs::remove_file(path);
}

#[test]
fn reviewer_child_does_not_inherit_auto_review_service() {
    let mut parent = test_agent();
    parent.set_permission_mode(PermissionMode::Auto);
    parent.set_auto_review_service(Some(Arc::new(MockAutoReviewService {
        approval: Mutex::new(PermissionApproval::AllowOnce),
        calls: AtomicUsize::new(0),
        child_session_id: "reviewer-child-recursion-guard".into(),
    })));

    let reviewer = AgentFactory::create_child(&parent, &AgentTemplate::reviewer());

    assert!(reviewer.auto_review_service.is_none());
    assert_eq!(reviewer.permission_mode(), PermissionMode::Yolo);
}

#[test]
fn child_permission_session_inherits_mode_without_grants() {
    let resource = crate::permission::PermissionResource::Exact {
        tool: "shell__exec".into(),
        value: "curl --version".into(),
    };
    let mut parent = test_agent();
    parent.set_permission_mode(PermissionMode::Auto);
    parent
        .permission_session
        .lock()
        .expect("permission session")
        .grant(resource.clone());

    let child = AgentFactory::create_child(&parent, &AgentTemplate::fixer());
    assert_eq!(child.permission_mode(), PermissionMode::Auto);
    assert!(
        !child
            .permission_session
            .lock()
            .expect("child permission session")
            .allows_grant(&resource)
    );
    assert!(child.subagent_path_scope.is_none());
}

#[test]
fn explicit_child_permission_mode_overrides_parent_mode() {
    let mut parent = test_agent();
    parent.set_permission_mode(PermissionMode::Auto);
    let mut template = AgentTemplate::fixer();
    template.permission_mode = PermissionMode::Safe;

    let child = AgentFactory::create_child(&parent, &template);

    assert_eq!(child.permission_mode(), PermissionMode::Safe);
}

#[cfg(unix)]
#[tokio::test]
async fn child_shell_allow_always_stays_within_child_session() {
    let command = "curl --version";
    let call = HistoryToolCall {
        call_id: "call-child-shell".into(),
        name: "shell__exec".into(),
        arguments_json: json!({"command": command}).to_string(),
    };
    let resource = crate::permission::PermissionResource::Exact {
        tool: "shell__exec".into(),
        value: command.into(),
    };

    let mut parent = test_agent();
    parent.set_permission_mode(PermissionMode::Default);
    parent
        .permission_session
        .lock()
        .expect("permission session")
        .grant(resource.clone());

    let mut child = AgentFactory::create_child(&parent, &AgentTemplate::fixer());
    let mut sibling = AgentFactory::create_child(&parent, &AgentTemplate::fixer());
    let mut child_approvals = 0usize;
    let first = child
        .execute_tool_call(&call, &mut |_| std::future::ready(Ok(())), &mut |_| {
            child_approvals += 1;
            std::future::ready(Ok(PermissionApproval::AllowAlways))
        })
        .await
        .expect("child first shell");
    assert_eq!(child_approvals, 1);
    assert_eq!(first.status, ToolExecutionStatus::Executed);

    let mut second_approvals = 0usize;
    let second = child
        .execute_tool_call(&call, &mut |_| std::future::ready(Ok(())), &mut |_| {
            second_approvals += 1;
            std::future::ready(Ok(PermissionApproval::Deny))
        })
        .await
        .expect("child grant reuse");
    assert_eq!(second_approvals, 0);
    assert_eq!(second.status, ToolExecutionStatus::Executed);

    let mut sibling_approvals = 0usize;
    let sibling_record = sibling
        .execute_tool_call(&call, &mut |_| std::future::ready(Ok(())), &mut |_| {
            sibling_approvals += 1;
            std::future::ready(Ok(PermissionApproval::Deny))
        })
        .await
        .expect("sibling isolated");
    assert_eq!(sibling_approvals, 1);
    assert_eq!(sibling_record.status, ToolExecutionStatus::Rejected);
}

#[cfg(unix)]
#[tokio::test]
async fn delegation_scope_authorizes_owned_writes_and_denies_outside() {
    let fixture = UnixWritableFixture::new("delegation-scope-write");
    let owned = fixture.external.join("owned");
    let outside = fixture.external.join("outside");
    fs::create_dir_all(&owned).expect("owned dir");
    fs::create_dir_all(&outside).expect("outside dir");
    let owned_file = owned.join("ok.txt");
    let outside_file = outside.join("no.txt");

    let scope = crate::tool::SubagentPathScope::from_input(&NormalizedSubagentInput {
        objective: "scoped write".into(),
        success_criteria: Vec::new(),
        allowed_paths: Vec::new(),
        forbidden_paths: Vec::new(),
        owned_paths: vec![owned.to_string_lossy().into()],
        timeout_secs: None,
        max_tool_calls: None,
        model: None,
        target_child_session_id: None,
        background: false,
    })
    .expect("scope")
    .expect("non-empty scope");

    let mut child = AgentFactory::create_child(&test_agent(), &AgentTemplate::fixer());
    child.set_permission_mode(PermissionMode::Default);
    child.set_subagent_path_scope(Some(Arc::new(scope)));

    let mut approvals = 0usize;
    let allowed = child
        .execute_tool_call(
            &HistoryToolCall {
                call_id: "scoped-write-ok".into(),
                name: "fs__write".into(),
                arguments_json: json!({
                    "path": owned_file,
                    "content": "ok",
                })
                .to_string(),
            },
            &mut |_| std::future::ready(Ok(())),
            &mut |_| {
                approvals += 1;
                std::future::ready(Ok(PermissionApproval::Deny))
            },
        )
        .await
        .expect("owned write");
    assert_eq!(approvals, 0, "owned_paths must pre-authorize writes");
    assert_eq!(allowed.status, ToolExecutionStatus::Executed);
    assert_eq!(fs::read_to_string(&owned_file).expect("read"), "ok");

    let denied = child
        .execute_tool_call(
            &HistoryToolCall {
                call_id: "scoped-write-denied".into(),
                name: "fs__write".into(),
                arguments_json: json!({
                    "path": outside_file,
                    "content": "no",
                })
                .to_string(),
            },
            &mut |_| std::future::ready(Ok(())),
            &mut |_| {
                approvals += 1;
                std::future::ready(Ok(PermissionApproval::AllowOnce))
            },
        )
        .await
        .expect("outside write");
    assert_eq!(approvals, 0, "scope denial must not reach approver");
    assert_eq!(denied.status, ToolExecutionStatus::Rejected);
    assert_eq!(
        denied.rejection,
        Some(ToolExecutionRejection::DelegationScopeDenied)
    );
    assert!(!outside_file.exists());
}

#[cfg(unix)]
#[tokio::test]
async fn delegation_scope_allows_reads_in_allowed_paths_and_forbids_forbidden() {
    let fixture = UnixWritableFixture::new("delegation-scope-read");
    let allowed = fixture.external.join("allowed");
    let forbidden = fixture.external.join("forbidden");
    fs::create_dir_all(&allowed).expect("allowed");
    fs::create_dir_all(&forbidden).expect("forbidden");
    let allowed_file = allowed.join("a.txt");
    let forbidden_file = forbidden.join("b.txt");
    fs::write(&allowed_file, "hello").expect("seed allowed");
    fs::write(&forbidden_file, "secret").expect("seed forbidden");

    let scope = crate::tool::SubagentPathScope::from_input(&NormalizedSubagentInput {
        objective: "scoped read".into(),
        success_criteria: Vec::new(),
        allowed_paths: vec![allowed.to_string_lossy().into()],
        forbidden_paths: vec![forbidden.to_string_lossy().into()],
        owned_paths: Vec::new(),
        timeout_secs: None,
        max_tool_calls: None,
        model: None,
        target_child_session_id: None,
        background: false,
    })
    .expect("scope")
    .expect("non-empty scope");

    let mut reader = AgentFactory::create_child(&test_agent(), &AgentTemplate::explorer());
    reader.set_subagent_path_scope(Some(Arc::new(scope.clone())));

    let ok = reader
        .execute_tool_call(
            &HistoryToolCall {
                call_id: "scoped-read-ok".into(),
                name: "fs__read".into(),
                arguments_json: json!({"path": allowed_file}).to_string(),
            },
            &mut |_| std::future::ready(Ok(())),
            &mut |_| std::future::ready(Ok(PermissionApproval::Deny)),
        )
        .await
        .expect("allowed read");
    assert_eq!(ok.status, ToolExecutionStatus::Executed);

    let blocked = reader
        .execute_tool_call(
            &HistoryToolCall {
                call_id: "scoped-read-forbidden".into(),
                name: "fs__read".into(),
                arguments_json: json!({"path": forbidden_file}).to_string(),
            },
            &mut |_| std::future::ready(Ok(())),
            &mut |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await
        .expect("forbidden read");
    assert_eq!(blocked.status, ToolExecutionStatus::Rejected);
    assert_eq!(
        blocked.rejection,
        Some(ToolExecutionRejection::DelegationScopeDenied)
    );

    let mut writer = AgentFactory::create_child(&test_agent(), &AgentTemplate::fixer());
    writer.set_subagent_path_scope(Some(Arc::new(scope)));
    let write_in_allowed = writer
        .execute_tool_call(
            &HistoryToolCall {
                call_id: "scoped-write-allowed-only".into(),
                name: "fs__write".into(),
                arguments_json: json!({
                    "path": allowed.join("new.txt"),
                    "content": "nope",
                })
                .to_string(),
            },
            &mut |_| std::future::ready(Ok(())),
            &mut |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await
        .expect("allowed-only write");
    assert_eq!(write_in_allowed.status, ToolExecutionStatus::Rejected);
    assert_eq!(
        write_in_allowed.rejection,
        Some(ToolExecutionRejection::DelegationScopeDenied)
    );
}

#[cfg(unix)]
#[tokio::test]
async fn delegation_scope_apply_patch_requires_all_targets_owned() {
    let fixture = UnixWritableFixture::new("delegation-scope-patch");
    let owned = fixture.external.join("owned");
    let other = fixture.external.join("other");
    fs::create_dir_all(&owned).expect("owned");
    fs::create_dir_all(&other).expect("other");
    let owned_file = owned.join("a.txt");
    let other_file = other.join("b.txt");
    fs::write(&owned_file, "old-a").expect("seed a");
    fs::write(&other_file, "old-b").expect("seed b");

    let scope = crate::tool::SubagentPathScope::from_input(&NormalizedSubagentInput {
        objective: "scoped patch".into(),
        success_criteria: Vec::new(),
        allowed_paths: Vec::new(),
        forbidden_paths: Vec::new(),
        owned_paths: vec![owned.to_string_lossy().into()],
        timeout_secs: None,
        max_tool_calls: None,
        model: None,
        target_child_session_id: None,
        background: false,
    })
    .expect("scope")
    .expect("non-empty scope");

    let mut child = AgentFactory::create_child(&test_agent(), &AgentTemplate::fixer());
    child.set_permission_mode(PermissionMode::Default);
    child.set_subagent_path_scope(Some(Arc::new(scope)));

    let ok = child
        .execute_tool_call(
            &HistoryToolCall {
                call_id: "scoped-patch-ok".into(),
                name: "edit__apply_patch".into(),
                arguments_json: json!({
                    "edits": [{
                        "path": owned_file,
                        "find": "old-a",
                        "replace": "new-a",
                        "replace_all": false
                    }]
                })
                .to_string(),
            },
            &mut |_| std::future::ready(Ok(())),
            &mut |_| std::future::ready(Ok(PermissionApproval::Deny)),
        )
        .await
        .expect("owned patch");
    assert_eq!(ok.status, ToolExecutionStatus::Executed);
    assert_eq!(fs::read_to_string(&owned_file).expect("read a"), "new-a");

    let denied = child
        .execute_tool_call(
            &HistoryToolCall {
                call_id: "scoped-patch-mixed".into(),
                name: "edit__apply_patch".into(),
                arguments_json: json!({
                    "edits": [
                        {
                            "path": owned_file,
                            "find": "new-a",
                            "replace": "mutated",
                            "replace_all": false
                        },
                        {
                            "path": other_file,
                            "find": "old-b",
                            "replace": "mutated-b",
                            "replace_all": false
                        }
                    ]
                })
                .to_string(),
            },
            &mut |_| std::future::ready(Ok(())),
            &mut |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await
        .expect("mixed patch");
    assert_eq!(denied.status, ToolExecutionStatus::Rejected);
    assert_eq!(
        denied.rejection,
        Some(ToolExecutionRejection::DelegationScopeDenied)
    );
    assert_eq!(fs::read_to_string(&owned_file).expect("read a"), "new-a");
    assert_eq!(fs::read_to_string(&other_file).expect("read b"), "old-b");
}

#[tokio::test]
async fn auto_mode_uses_reviewer_service_and_skips_human_approve() {
    let mut agent = test_agent();
    agent.set_permission_mode(PermissionMode::Auto);
    let service = Arc::new(MockAutoReviewService {
        approval: Mutex::new(PermissionApproval::AllowOnce),
        calls: AtomicUsize::new(0),
        child_session_id: "reviewer-child-1".into(),
    });
    agent.set_auto_review_service(Some(service.clone()));

    let path = std::env::temp_dir().join(format!(
        "letcode-auto-review-once-{}.txt",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time")
            .as_nanos()
    ));
    let call = HistoryToolCall {
        call_id: "call-auto-once".into(),
        name: "fs__write".into(),
        arguments_json: json!({"path": path, "content": "ok"}).to_string(),
    };
    let mut human_approvals = 0usize;
    let record = agent
        .execute_tool_call(&call, &mut |_| std::future::ready(Ok(())), &mut |_| {
            human_approvals += 1;
            std::future::ready(Ok(PermissionApproval::Deny))
        })
        .await
        .expect("auto allow_once executes");

    assert_eq!(human_approvals, 0);
    assert_eq!(service.calls.load(Ordering::SeqCst), 1);
    assert_eq!(record.status, ToolExecutionStatus::Executed);
    assert!(record.output.ok, "{:?}", record.output.error);
    let _ = std::fs::remove_file(path);
}

#[tokio::test]
async fn auto_mode_reviewer_decides_calls_that_conflict_with_execution_directive() {
    let mut agent = test_agent();
    agent.set_permission_mode(PermissionMode::Auto);
    agent.turn = TurnRuntimeState::new(
        1,
        WorkflowTurnState::from_user_input("Read only. Analyze and report."),
    );
    let service = Arc::new(CapturingAutoReviewService {
        approval: PermissionApproval::AllowOnce,
        calls: AtomicUsize::new(0),
        last_request: Mutex::new(None),
    });
    agent.set_auto_review_service(Some(service.clone()));

    let path = std::env::temp_dir().join(format!(
        "letcode-auto-review-directive-{}.txt",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time")
            .as_nanos()
    ));
    let call = HistoryToolCall {
        call_id: "call-auto-directive".into(),
        name: "fs__write".into(),
        arguments_json: json!({"path": path, "content": "approved"}).to_string(),
    };
    let mut human_approvals = 0usize;
    let record = agent
        .execute_tool_call(&call, &mut |_| std::future::ready(Ok(())), &mut |_| {
            human_approvals += 1;
            std::future::ready(Ok(PermissionApproval::Deny))
        })
        .await
        .expect("reviewer approval overrides static directive denial");

    assert_eq!(human_approvals, 0);
    assert_eq!(service.calls.load(Ordering::SeqCst), 1);
    let request = service
        .last_request
        .lock()
        .expect("request lock")
        .clone()
        .expect("review request");
    assert_eq!(request.directive, ExecutionDirective::ReadOnly);
    assert!(!request.can_allow_always);
    assert!(request.grant_summary.is_none());
    assert_eq!(record.status, ToolExecutionStatus::Executed);
    assert!(record.output.ok, "{:?}", record.output.error);
    assert_eq!(fs::read_to_string(&path).expect("read output"), "approved");
    let _ = fs::remove_file(path);
}

#[cfg(unix)]
#[tokio::test]
async fn auto_mode_keeps_explicit_subagent_scope_as_hard_boundary() {
    let fixture = UnixWritableFixture::new("auto-subagent-scope");
    let owned = fixture.external.join("owned");
    let outside = fixture.external.join("outside");
    fs::create_dir_all(&owned).expect("owned dir");
    fs::create_dir_all(&outside).expect("outside dir");
    let owned_file = owned.join("ok.txt");
    let outside_file = outside.join("no.txt");
    let scope = crate::tool::SubagentPathScope::from_input(&NormalizedSubagentInput {
        objective: "scoped auto write".into(),
        success_criteria: Vec::new(),
        allowed_paths: Vec::new(),
        forbidden_paths: Vec::new(),
        owned_paths: vec![owned.to_string_lossy().into()],
        timeout_secs: None,
        max_tool_calls: None,
        model: None,
        target_child_session_id: None,
        background: false,
    })
    .expect("scope")
    .expect("non-empty scope");

    let mut child = AgentFactory::create_child(&test_agent(), &AgentTemplate::fixer());
    child.set_permission_mode(PermissionMode::Auto);
    child.set_subagent_path_scope(Some(Arc::new(scope)));
    let service = Arc::new(MockAutoReviewService {
        approval: Mutex::new(PermissionApproval::Deny),
        calls: AtomicUsize::new(0),
        child_session_id: "reviewer-child-scope".into(),
    });
    child.set_auto_review_service(Some(service.clone()));

    let allowed = child
        .execute_tool_call(
            &HistoryToolCall {
                call_id: "auto-scope-owned".into(),
                name: "fs__write".into(),
                arguments_json: json!({"path": owned_file, "content": "ok"}).to_string(),
            },
            &mut |_| std::future::ready(Ok(())),
            &mut |_| std::future::ready(Ok(PermissionApproval::Deny)),
        )
        .await
        .expect("owned write");
    assert_eq!(allowed.status, ToolExecutionStatus::Executed);
    assert_eq!(service.calls.load(Ordering::SeqCst), 0);

    let denied = child
        .execute_tool_call(
            &HistoryToolCall {
                call_id: "auto-scope-outside".into(),
                name: "fs__write".into(),
                arguments_json: json!({"path": outside_file, "content": "no"}).to_string(),
            },
            &mut |_| std::future::ready(Ok(())),
            &mut |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await
        .expect("outside write");
    assert_eq!(denied.status, ToolExecutionStatus::Rejected);
    assert_eq!(
        denied.rejection,
        Some(ToolExecutionRejection::DelegationScopeDenied)
    );
    assert_eq!(service.calls.load(Ordering::SeqCst), 0);
    assert!(!outside_file.exists());
}

#[tokio::test]
async fn auto_mode_subagent_batch_uses_reviewer_and_skips_human_approve() {
    let mut agent = test_agent();
    agent.set_permission_mode(PermissionMode::Auto);
    let service = Arc::new(MockAutoReviewService {
        approval: Mutex::new(PermissionApproval::AllowOnce),
        calls: AtomicUsize::new(0),
        child_session_id: "reviewer-child-subagent".into(),
    });
    agent.set_auto_review_service(Some(service.clone()));
    agent.set_subagent_delegate(static_delegate(ToolResult::ok(
        "agent__fixer",
        json!({
            "run_id": "run-auto-fixer",
            "child_session_id": "child-auto-fixer",
            "status": "completed",
            "summary": "done"
        }),
    )));

    let call = test_tool_call(
        "agent__fixer",
        r#"{"task":"apply focused fix","owned_paths":["src/agent.rs"]}"#,
    );
    agent
        .append_assistant_tool_calls("", std::slice::from_ref(&call))
        .expect("append subagent call");
    let mut human_approvals = 0usize;
    agent
        .execute_tool_calls_and_record(
            std::slice::from_ref(&call),
            &mut |_| std::future::ready(Ok(())),
            &mut |_| {
                human_approvals += 1;
                std::future::ready(Ok(PermissionApproval::Deny))
            },
        )
        .await
        .expect("auto-reviewed subagent executes");

    assert_eq!(human_approvals, 0);
    assert_eq!(service.calls.load(Ordering::SeqCst), 1);
    assert!(agent.history_for_test().iter().any(|item| matches!(
        item,
        HistoryItem::ToolOutput { call_id, .. } if call_id == &call.call_id
    )));
}

#[tokio::test]
async fn auto_mode_deny_includes_reviewer_rationale() {
    let mut agent = test_agent();
    agent.set_permission_mode(PermissionMode::Auto);
    let service = Arc::new(MockAutoReviewService {
        approval: Mutex::new(PermissionApproval::Deny),
        calls: AtomicUsize::new(0),
        child_session_id: "reviewer-child-deny".into(),
    });
    agent.set_auto_review_service(Some(service));

    let call = HistoryToolCall {
        call_id: "call-auto-deny".into(),
        name: "fs__write".into(),
        arguments_json: json!({"path": "deny.txt", "content": "no"}).to_string(),
    };
    let record = agent
        .execute_tool_call(&call, &mut |_| std::future::ready(Ok(())), &mut |_| {
            std::future::ready(Ok(PermissionApproval::AllowOnce))
        })
        .await
        .expect("auto deny is a tool record");

    assert_eq!(record.status, ToolExecutionStatus::Rejected);
    assert_eq!(
        record.rejection,
        Some(ToolExecutionRejection::PermissionDeniedByUser)
    );
    assert!(record.output.error.as_ref().is_some_and(|error| {
        error
            .message
            .contains("auto-review denied permission: mock-deny")
    }));
}

#[cfg(unix)]
#[tokio::test]
async fn auto_mode_does_not_reuse_allow_always_between_reviews() {
    let fixture = UnixWritableFixture::new("auto-allow-always");
    let path = fixture.external.join("written.txt");
    let first_call = writable_call("auto-always-first", "fs__write", &path, "first");
    let second_call = writable_call("auto-always-second", "fs__write", &path, "second");
    let mut agent = test_agent();
    agent.set_permission_mode(PermissionMode::Auto);
    let service = Arc::new(MockAutoReviewService {
        approval: Mutex::new(PermissionApproval::AllowAlways),
        calls: AtomicUsize::new(0),
        child_session_id: "reviewer-child-sticky".into(),
    });
    agent.set_auto_review_service(Some(service.clone()));

    let first = agent
        .execute_tool_call(
            &first_call,
            &mut |_| std::future::ready(Ok(())),
            &mut |_| std::future::ready(Ok(PermissionApproval::Deny)),
        )
        .await
        .expect("first auto allow_always");
    assert!(first.output.ok, "{:?}", first.output.error);
    assert_eq!(service.calls.load(Ordering::SeqCst), 1);

    let second = agent
        .execute_tool_call(
            &second_call,
            &mut |_| std::future::ready(Ok(())),
            &mut |_| std::future::ready(Ok(PermissionApproval::Deny)),
        )
        .await
        .expect("second auto review");
    assert!(second.output.ok, "{:?}", second.output.error);
    assert_eq!(
        service.calls.load(Ordering::SeqCst),
        2,
        "Auto mode must review each call instead of reusing a session grant"
    );
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
async fn yolo_mode_executes_commands_that_default_mode_asks() {
    let mut agent = test_agent();
    agent.set_permission_mode(PermissionMode::Yolo);
    let call = HistoryToolCall {
        call_id: "call-yolo-ask-risk".into(),
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

#[test]
fn legacy_session_history_restore_clears_runtime_workflow_state() {
    let mut agent = test_agent();
    agent.runtime_snapshot.workflow.todos = vec![TodoItem {
        id: "stale".into(),
        content: "old session".into(),
        status: TodoStatus::Pending,
    }];
    agent.runtime_snapshot.workflow.auto_continue.enabled = true;

    agent
        .restore_session_history(vec![HistoryItem::user("restored session")], Vec::new(), 4)
        .expect("legacy history restore");

    assert!(agent.runtime_snapshot.workflow.is_empty());
}

#[test]
fn new_turn_preserves_todos_and_resets_auto_continue() {
    let mut agent = test_agent();
    agent.runtime_snapshot.workflow.todos = vec![TodoItem {
        id: "stale".into(),
        content: "old turn".into(),
        status: TodoStatus::Pending,
    }];
    agent.runtime_snapshot.workflow.auto_continue.enabled = true;

    agent.prepare_turn_prelude("start a new turn");

    assert_eq!(agent.runtime_snapshot.workflow.todos.len(), 1);
    assert_eq!(agent.runtime_snapshot.workflow.todos[0].id, "stale");
    assert!(!agent.runtime_snapshot.workflow.auto_continue.enabled);
}

#[tokio::test]
async fn auto_continue_stops_only_when_llm_disables_it() {
    let mut agent = test_agent();
    agent.runtime_snapshot.workflow.auto_continue.enabled = true;
    agent.runtime_snapshot.workflow.todos = vec![TodoItem {
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

    agent.runtime_snapshot.workflow.auto_continue.enabled = false;
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
        .filter(|message| message.text.starts_with("来自 "))
        .collect::<Vec<_>>();
    assert_eq!(instructions.len(), 3);
    assert!(
        instructions
            .iter()
            .all(|message| message.role == crate::request_builder::PromptRole::System)
    );
    assert!(instructions[0].text.contains(&format!(
        "来自 {} 的指令：",
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
        .filter(|message| message.text.starts_with("来自 "))
        .collect::<Vec<_>>();
    assert_eq!(instructions.len(), 2);
    assert!(
        instructions
            .iter()
            .all(|message| message.role == crate::request_builder::PromptRole::System)
    );
    assert!(instructions[0].text.ends_with("global instructions"));
    assert!(instructions[1].text.ends_with("workspace instructions"));

    fs::remove_dir_all(root).expect("test directory should be removed");
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
async fn execute_tool_call_emits_finished_event_for_user_denial_of_high_risk_command() {
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
    let mut approval_requested = false;

    let record = agent
        .execute_tool_call(
            &call,
            &mut |event| {
                events.push(event);
                std::future::ready(Ok(()))
            },
            &mut |_| {
                approval_requested = true;
                std::future::ready(Ok(PermissionApproval::Deny))
            },
        )
        .await
        .expect("user denial should be reported as tool output");

    assert!(
        approval_requested,
        "high-risk commands must Ask, not hard-deny"
    );
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
                && rejection == "permission_denied_by_user"
                && effect_kind == "diagnostic"
    ));
    assert_eq!(record.status, ToolExecutionStatus::Rejected);
    assert_eq!(
        record.rejection,
        Some(ToolExecutionRejection::PermissionDeniedByUser)
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
