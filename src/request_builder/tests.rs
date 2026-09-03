use super::*;

#[test]
fn apply_patch_single_tool_spec_budget_accounts_for_batch_prevalidation_contract() {
    let apply_patch = crate::tool::ToolRegistry::default_tools()
        .specs()
        .into_iter()
        .find(|spec| spec.name == "edit__apply_patch")
        .expect("default registry includes ApplyPatch");
    let accepted = estimate_tools_tokens(std::slice::from_ref(&apply_patch));

    // The truthful batch-prevalidation/non-transactional contract added
    // 164 estimated tokens; preserve the accepted single-tool ceiling.
    assert!(
        accepted <= 569,
        "ApplyPatch contract budget must remain within its 569-token acceptance ceiling; got {accepted}"
    );
}

#[test]
fn evidence_budget_tokens_preserves_saturating_15_percent_clamp() {
    assert_eq!(evidence_budget_tokens(3_413), 512);
    assert_eq!(evidence_budget_tokens(3_414), 512);
    assert_eq!(evidence_budget_tokens(10_000), 1_500);
    assert_eq!(evidence_budget_tokens(20_000), 3_000);
    assert_eq!(evidence_budget_tokens(u64::MAX), 3_000);
}

use crate::agent::{ToolExecutionSummaryEvent, ValidationAdvisory};
use crate::context_tree::ContextNodeStatus;
use crate::context_view::{ContextBlockId, ContextViewStatus};
use crate::evidence::{EvidenceKind, EvidenceRecord, EvidenceSource};
use crate::runtime_context::RuntimeChildSession;
use crate::tool::ToolResult;
use crate::transcript::transcript_projection::{
    project_context_tree, project_context_view as project_restored_context_view,
    project_session_restore_snapshot, restore_session_history_projection,
};
use crate::transcript::{TranscriptEvent, TranscriptRecord};
use serde_json::json;

fn metadata(context_window: u64) -> ModelRequestMetadata {
    ModelRequestMetadata {
        context_window: Some(context_window),
        max_output_tokens: Some(256),
        supports_tools: true,
        supports_reasoning: false,
        ..Default::default()
    }
}

fn metadata_with_effective_input_limit(
    context_window: u64,
    effective_input_limit_tokens: u64,
) -> ModelRequestMetadata {
    ModelRequestMetadata {
        effective_input_limit_tokens: Some(effective_input_limit_tokens),
        ..metadata(context_window)
    }
}

fn deepseek_metadata() -> ModelRequestMetadata {
    ModelRequestMetadata {
        supports_reasoning: true,
        reasoning_effort: Some(ModelReasoningEffort::High),
        max_output_tokens: Some(32_768),
        ..metadata(384_000)
    }
}

fn evidence(id: &str, summary: &str, path: &str, sequence: u64) -> EvidenceRecord {
    EvidenceRecord {
        id: id.to_string(),
        sequence,
        timestamp_ms: 0,
        evidence_kind: EvidenceKind::FileExcerpt,
        title: format!("read {path}"),
        summary: summary.to_string(),
        detail: None,
        source: EvidenceSource::File {
            path: path.to_string(),
            start_line: Some(1),
            end_line: Some(3),
        },
        tags: vec![path.to_string()],
    }
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
fn anthropic_logical_units_preserve_one_unit_per_prompt_segment() {
    let result = build_test_request(TestRequestBuilderInput {
        model_id: "claude-test",
        model: metadata(8_192),
        prelude: &[
            PromptMessage::system("system"),
            PromptMessage::developer("developer"),
        ],
        history: &[
            HistoryItem::user("question"),
            HistoryItem::assistant("answer"),
        ],
        protected_start_index: 2,
        tools: &[],
        evidence: &[],
    })
    .expect("anthropic request builds");

    let segment_count = result.prompt_plan.segments.len();
    assert!(segment_count > 0);
    assert_eq!(observe_logical_request(&result).units.len(), segment_count);
    for prefix in 0..=segment_count {
        let _ = provider_unit_prefix_digest(&result, prefix);
    }
}

#[test]
fn anthropic_messages_applies_adaptive_thinking_effort() {
    let history = vec![HistoryItem::user("current")];
    let mut model = metadata(8_192);
    model.supports_reasoning = true;
    model.reasoning_effort = Some(ModelReasoningEffort::Medium);
    model.anthropic_thinking = AnthropicThinkingConfig {
        mode: AnthropicThinkingMode::Adaptive,
        budget_tokens: None,
    };
    let result = build_test_request(TestRequestBuilderInput {
        model_id: "claude-test",
        model,
        prelude: &[],
        history: &history,
        protected_start_index: 0,
        tools: &[],
        evidence: &[],
    })
    .expect("anthropic adaptive request builds");
    assert!(
        result
            .prompt_plan
            .segments
            .iter()
            .any(|segment| segment.text == "current")
    );
}

#[test]
fn deepseek_completions_preserves_skill_material_across_developer_messages() {
    let history = vec![HistoryItem::user("current")];
    let result = build_test_request(TestRequestBuilderInput {
        model_id: "deepseek-v4-flash",
        model: deepseek_metadata(),
        prelude: &[
            PromptMessage::developer("## persona stable instructions"),
            PromptMessage::developer_with_origin(
                "可用本地 skills：- ctf-web — 用于 web 攻防",
                PromptMessageOrigin::SkillCatalog,
            ),
            PromptMessage::developer_with_origin(
                "---\nname: ctf-web\ndescription: web\n---\n# SKILL BODY MAGIC",
                PromptMessageOrigin::SkillMaterial,
            ),
        ],
        history: &history,
        protected_start_index: 0,
        tools: &[],
        evidence: &[],
    })
    .expect("deepseek chat request builds");
    let contains = |needle: &str| {
        result
            .prompt_plan
            .segments
            .iter()
            .any(|segment| segment.text.contains(needle))
    };
    assert!(contains("SKILL BODY MAGIC"));
    assert!(contains("persona stable instructions"));
    assert!(contains("ctf-web"));
}

#[test]
fn deepseek_chat_compat_preserves_empty_reasoning_content_for_tool_turns() {
    let history = vec![
        HistoryItem::AssistantTurn {
            text: None,
            reasoning_content: None,
            replay: None,
            calls: vec![HistoryToolCall {
                call_id: "call-1".into(),
                name: "read_file".into(),
                arguments_json: "{}".into(),
            }],
        },
        HistoryItem::ToolOutput {
            call_id: "call-1".into(),
            output_json: "ok".into(),
            images: Vec::new(),
        },
        HistoryItem::user("continue"),
    ];
    let result = build_test_request(TestRequestBuilderInput {
        model_id: "deepseek-v4-flash",
        model: deepseek_metadata(),
        prelude: &[],
        history: &history,
        protected_start_index: 0,
        tools: &[],
        evidence: &[],
    })
    .expect("deepseek tool replay builds");
    assert!(
        result
            .prompt_plan
            .segments
            .iter()
            .any(|segment| segment.text == "continue")
    );
}

#[test]
fn orphan_tool_outputs_fail_fast_when_building_chat_request() {
    let history = vec![
        HistoryItem::context_summary("旧工具调用已总结"),
        HistoryItem::ToolOutput {
            call_id: "call-orphan".into(),
            output_json: r#"{"ok":true}"#.into(),
            images: Vec::new(),
        },
        HistoryItem::user("continue"),
    ];

    let error = build_test_request(TestRequestBuilderInput {
        model_id: "chat-test",
        model: metadata(8192),
        prelude: &[],
        history: &history,
        protected_start_index: 2,
        tools: &[],
        evidence: &[],
    })
    .expect_err("orphan tool output must fail");
    assert!(error.to_string().contains("orphan tool output"));
}

#[test]
fn truncates_oldest_history_but_keeps_protected_items() {
    let long = "x".repeat(10_000);
    let history = vec![
        HistoryItem::user("old"),
        HistoryItem::assistant(long),
        HistoryItem::user("current"),
    ];
    let result = build_test_request(TestRequestBuilderInput {
        model_id: "gpt-test",
        model: metadata(1200),
        prelude: &[],
        history: &history,
        protected_start_index: 2,
        tools: &[],
        evidence: &[],
    })
    .expect("request builds");

    assert!(result.budget.truncated);
    assert!(
        result
            .prompt_plan
            .segments
            .iter()
            .any(|segment| segment.text == "current")
    );
}

#[test]
fn effective_input_limit_counts_tool_schema_tokens() {
    let history = vec![HistoryItem::user("current")];
    let tools = vec![ToolSpec {
        name: "read_file".to_string(),
        description: "read a file".to_string(),
        parameters: json!({
            "type": "object",
            "properties": { "path": { "type": "string" } },
            "required": ["path"],
            "additionalProperties": false
        }),
        strict: true,
    }];

    let capped = build_test_request(TestRequestBuilderInput {
        model_id: "gpt-test",
        model: metadata_with_effective_input_limit(32_000, 2_000),
        prelude: &[],
        history: &history,
        protected_start_index: 0,
        tools: &tools,
        evidence: &[],
    })
    .expect("effective-input-limited request with tools builds");

    assert!(capped.budget.estimated_tools_tokens > 0);
    assert_eq!(
        capped.budget.input_budget_tokens + capped.budget.estimated_tools_tokens,
        2_000
    );
    assert!(capped.budget.estimated_request_tokens <= 2_000);
}

#[test]
fn evidence_is_dropped_when_current_turn_leaves_no_context_room() {
    let model = metadata(1024);
    let input_budget = model
        .context_window_tokens()
        .saturating_sub(model.output_reserve_tokens())
        .saturating_sub(SAFETY_OVERHEAD_TOKENS)
        .max(1);
    let exact_fit = (0..10_000)
        .map(|len| "x".repeat(len))
        .find(|text| estimate_history_item_tokens(&HistoryItem::user(text.clone())) == input_budget)
        .expect("should find exact fit for input budget");
    let history = vec![HistoryItem::user(exact_fit)];
    let evidence = vec![evidence(
        "ev-1",
        "src/evidence.rs defines compact evidence records",
        "src/evidence.rs",
        1,
    )];

    let result = build_test_request(TestRequestBuilderInput {
        model_id: "gpt-test",
        model,
        prelude: &[],
        history: &history,
        protected_start_index: 0,
        tools: &[],
        evidence: &evidence,
    })
    .expect("request builds");

    assert!(result.selected_evidence_ids.is_empty());
    assert_eq!(result.budget.dropped_evidence_items, 1);
}

#[test]
fn oversized_optional_evidence_is_dropped_instead_of_failing_protected_context() {
    let model = metadata(1024);
    let input_budget = model
        .context_window_tokens()
        .saturating_sub(model.output_reserve_tokens())
        .saturating_sub(SAFETY_OVERHEAD_TOKENS)
        .max(1);
    let near_fit = (0..10_000)
        .map(|len| "x".repeat(len))
        .find(|text| {
            estimate_history_item_tokens(&HistoryItem::user(text.clone()))
                == input_budget.saturating_sub(1)
        })
        .expect("should find near fit for input budget");
    let history = vec![HistoryItem::user(near_fit)];
    let evidence = vec![evidence(
        "ev-1",
        "x ".repeat(200).as_str(),
        "src/evidence.rs",
        1,
    )];

    let result = build_test_request(TestRequestBuilderInput {
        model_id: "gpt-test",
        model,
        prelude: &[],
        history: &history,
        protected_start_index: 0,
        tools: &[],
        evidence: &evidence,
    })
    .expect("optional evidence should be dropped instead of failing protected context");

    assert!(result.selected_evidence_ids.is_empty());
    assert_eq!(result.budget.dropped_evidence_items, 1);
}

#[test]
fn returns_error_when_protected_current_turn_exceeds_input_budget() {
    let history = vec![
        HistoryItem::user("old context"),
        HistoryItem::assistant("old reply"),
        HistoryItem::user("x".repeat(20_000)),
    ];

    let err = build_test_request(TestRequestBuilderInput {
        model_id: "gpt-test",
        model: metadata(1024),
        prelude: &[],
        history: &history,
        protected_start_index: 2,
        tools: &[],
        evidence: &[],
    })
    .expect_err("protected current turn should fail fast");

    let message = err.to_string();
    assert!(message.contains("protected"));
    assert!(message.contains("current"));
    assert!(message.contains("context"));
    assert!(message.contains("budget"));
}

#[test]
fn returns_error_when_protected_current_turn_exceeds_effective_input_limit() {
    let history = vec![HistoryItem::user("x".repeat(20_000))];

    let err = build_test_request(TestRequestBuilderInput {
        model_id: "gpt-test",
        model: metadata_with_effective_input_limit(32_000, 300),
        prelude: &[],
        history: &history,
        protected_start_index: 0,
        tools: &[],
        evidence: &[],
    })
    .expect_err("effective-input-limited protected current turn should fail fast");

    let message = err.to_string();
    assert!(message.contains("protected/current context tokens"));
    assert!(message.contains("exceed budget (300)"));
}

#[test]
fn rejects_zero_effective_input_limit_metadata() {
    let err = build_test_request(TestRequestBuilderInput {
        model_id: "gpt-test",
        model: ModelRequestMetadata {
            effective_input_limit_tokens: Some(0),
            ..metadata(32_000)
        },
        prelude: &[],
        history: &[HistoryItem::user("current")],
        protected_start_index: 0,
        tools: &[],
        evidence: &[],
    })
    .expect_err("zero effective input limit should fail fast");

    assert!(
        err.to_string()
            .contains("model.effective_input_limit_tokens must be greater than 0")
    );
}

#[test]
fn protected_current_oversize_still_fails() {
    let history = vec![
        HistoryItem::user("old"),
        HistoryItem::user("x".repeat(20_000)),
    ];
    let err = build_test_request(TestRequestBuilderInput {
        model_id: "gpt-test",
        model: metadata(1024),
        prelude: &[],
        history: &history,
        protected_start_index: 1,
        tools: &[],
        evidence: &[],
    })
    .expect_err("protected current turn should still fail");
    assert!(
        err.to_string()
            .contains("protected current context exceeds input budget")
    );
}

#[test]
fn restored_context_view_remains_separate_from_provider_prompt() {
    let large_stdout = "stdout-body-".repeat(1_000);
    let large_stderr = "stderr-body-".repeat(1_000);
    let records = vec![
        transcript_record(
            1,
            TranscriptEvent::SessionStarted {
                model: "gpt-test".into(),
            },
        ),
        transcript_record(
            2,
            TranscriptEvent::UserMessage {
                content: UserMessageContent::from(
                    "MUST keep raw transcript events append-only; do not purge requirements",
                ),
            },
        ),
        transcript_record(
            3,
            TranscriptEvent::ContextNodeCreated {
                node_id: "child".into(),
                parent_node_id: Some("root".into()),
                label: Some("Restored child".into()),
                purpose: Some("Replay projected context tree".into()),
                block_ref: None,
                source_ref: None,
            },
        ),
        transcript_record(
            4,
            TranscriptEvent::ContextNodeLifecycle {
                node_id: "root".into(),
                status: ContextNodeStatus::Inactive,
            },
        ),
        transcript_record(
            5,
            TranscriptEvent::ContextNodeLifecycle {
                node_id: "child".into(),
                status: ContextNodeStatus::Active,
            },
        ),
        transcript_record(
            6,
            TranscriptEvent::ToolExecutionSummary(ToolExecutionSummaryEvent {
                turn_id: 1,
                call_id: "call-write".into(),
                name: "fs__write".into(),
                status: "executed".into(),
                rejection: None,
                effect_kind: "write".into(),
                primary_path: Some("src/lib.rs".into()),
                command: None,
            }),
        ),
        transcript_record(
            7,
            TranscriptEvent::PermissionDecision {
                call_id: Some("call-shell".into()),
                tool: "shell__exec".into(),
                args: json!({"command": "cargo test --quiet"}),
                allowed: false,
                reason: Some("Denied from restored permission prompt".into()),
                reviewer: None,
                approval: None,
                risk: None,
                reviewer_child_session_id: None,
            },
        ),
        transcript_record(
            8,
            TranscriptEvent::Error {
                message: "invariant violation: raw event missing".into(),
            },
        ),
        transcript_record(
            9,
            TranscriptEvent::ValidationAdvisory(ValidationAdvisory {
                write_effects: 1,
                validation_effects: 1,
                failed_validation_effects: 1,
                message: "cargo test failed".into(),
            }),
        ),
        transcript_record(
            10,
            TranscriptEvent::AssistantMessage {
                content: "commit a270dda is current base".into(),
            },
        ),
        transcript_record(
            11,
            TranscriptEvent::AssistantMessage {
                content: "soft archived note".into(),
            },
        ),
        transcript_record(
            12,
            TranscriptEvent::ContextViewOperationMetadata {
                operation: "archive".into(),
                node_id: None,
                block_id: Some("block-seq-11-note".into()),
                detail: None,
            },
        ),
        transcript_record(
            13,
            TranscriptEvent::AssistantMessage {
                content: "soft removed note".into(),
            },
        ),
        transcript_record(
            14,
            TranscriptEvent::ContextViewOperationMetadata {
                operation: "remove_from_view".into(),
                node_id: None,
                block_id: Some("block-seq-13-note".into()),
                detail: None,
            },
        ),
        transcript_record(
            15,
            TranscriptEvent::ToolCallStarted {
                call_id: "call-shell".into(),
                name: "shell__exec".into(),
                args: json!({"command": "cargo test --quiet"}),
            },
        ),
        transcript_record(
            16,
            TranscriptEvent::ToolCallFinished {
                call_id: "call-shell".into(),
                name: "shell__exec".into(),
                ok: false,
                output: ToolResult::err_with_data(
                    "shell__exec",
                    "command failed",
                    json!({
                        "status": 101,
                        "stdout": large_stdout,
                        "stdout_truncated": false,
                        "stderr": large_stderr,
                        "stderr_truncated": false,
                    }),
                ),
            },
        ),
        transcript_record(
            17,
            TranscriptEvent::ContextSummaryArtifactMetadata {
                node_id: "child".into(),
                artifact_id: "summary-1".into(),
                artifact_kind: "summary".into(),
                version: Some(1),
                summary: Some("child summary artifact".into()),
                source_node_id: Some("child".into()),
                source_block_id: Some("block-seq-10-note".into()),
                source_start_sequence: Some(10),
                source_end_sequence: Some(10),
            },
        ),
    ];
    let original_len = records.len();

    let snapshot = project_session_restore_snapshot("s".into(), records.clone()).expect("snapshot");
    let tree = project_context_tree(&records).expect("context tree");
    assert_eq!(tree.active_node_id().map(|id| id.as_str()), Some("child"));

    let projection = project_restored_context_view(&records).expect("context view");
    assert_eq!(
        projection
            .view_state
            .status(&ContextBlockId::new("block-seq-11-note").expect("archived block id")),
        Some(ContextViewStatus::Archived)
    );
    assert_eq!(
        projection
            .view_state
            .status(&ContextBlockId::new("block-seq-13-note").expect("removed block id")),
        Some(ContextViewStatus::RemovedFromView)
    );

    assert!(!snapshot.history.is_empty());
    let current_history = vec![HistoryItem::user("continue from restored context")];

    // ContextView remains available for TUI/tool addressing, while provider
    // requests are built solely from the supplied history frames.
    let result = build_test_request(TestRequestBuilderInput {
        model_id: "gpt-test",
        model: metadata(32_768),
        prelude: &[],
        history: &current_history,
        protected_start_index: 0,
        tools: &[],
        evidence: &snapshot.evidence,
    })
    .expect("request builds independently of restored projection");
    let rendered = result
        .prompt_plan
        .segments
        .iter()
        .map(|segment| segment.text.as_str())
        .collect::<Vec<_>>()
        .join("\n");

    assert!(rendered.contains("continue from restored context"));
    assert!(!rendered.contains("[Context: Hard Context]"));
    assert!(
        !rendered
            .contains("MUST keep raw transcript events append-only; do not purge requirements")
    );
    assert!(!rendered.contains("Permission denied"));
    assert!(!rendered.contains("soft archived note"));
    assert!(!rendered.contains("soft removed note"));
    assert!(!rendered.contains(&large_stdout));
    assert!(!rendered.contains(&large_stderr));
    assert_eq!(records.len(), original_len);
}

#[test]
fn planner_is_pure_deterministic_and_preserves_protected_tool_groups() {
    let frame = |kind, ordinal, item| {
        RuntimeFrame::new(
            kind,
            FrameVisibility::Active,
            RuntimeFrameProvenance::new(RuntimeSource::Transcript),
            RuntimeFrameIdSeed {
                frame_kind: kind,
                source: RuntimeSource::Transcript,
                ordinal,
                stable_key: "planner-purity",
                source_span: None,
            },
        )
        .with_protocol(item)
    };
    let tool_call = frame(
        RuntimeFrameKind::ToolCall,
        0,
        ProtocolFrameItem::AssistantTurn {
            text: None,
            reasoning_content: None,
            replay: None,
            calls: vec![HistoryToolCall {
                call_id: "call-1".into(),
                name: "read".into(),
                arguments_json: "{}".into(),
            }],
        },
    );
    let tool_output = frame(
        RuntimeFrameKind::ToolOutput,
        1,
        ProtocolFrameItem::ToolOutput {
            call_id: "call-1".into(),
            output_json: r#"{"ok":true}"#.into(),
            images: Vec::new(),
        },
    );
    let user = frame(
        RuntimeFrameKind::User,
        2,
        ProtocolFrameItem::UserMessage {
            content: UserMessageContent::from("current request"),
        },
    );
    let mut snapshot = RuntimeSnapshot::new("planner-purity");
    snapshot.compaction.protected_frame_ids.push(tool_output.id);
    snapshot.push_frame(tool_call);
    snapshot.push_frame(tool_output);
    snapshot.push_frame(user);
    let mut retired = frame(
        RuntimeFrameKind::Assistant,
        3,
        ProtocolFrameItem::AssistantTurn {
            text: Some("RETIRED-PLANNER-FRAME".into()),
            reasoning_content: None,
            replay: None,
            calls: Vec::new(),
        },
    );
    retired.visibility = FrameVisibility::Retired;
    snapshot.push_frame(retired);
    let before = snapshot.clone();
    let input = PromptPlannerInput {
        model: metadata(8192),
        model_id: "gpt-test",
        prelude: &[PromptMessage::system("system")],
        snapshot: &snapshot,
        tools: &[],
        frozen_evidence: None,
        protected_context_policy: ProtectedContextPolicy::from_configured_reserve(None, 0),
    };

    let first = PromptPlanner::plan(input.clone()).expect("planner succeeds");
    let second = PromptPlanner::plan(input).expect("planner repeats");

    assert_eq!(snapshot, before);
    assert_eq!(first.prompt_plan.segments, second.prompt_plan.segments);
    assert_eq!(first.budget, second.budget);
    assert_eq!(first.selected_evidence_ids, second.selected_evidence_ids);
    assert_eq!(
        first.prompt_plan.stable_prefix_hash(),
        second.prompt_plan.stable_prefix_hash()
    );
    assert!(first.prompt_plan.segments.iter().any(|segment| {
        matches!(
            segment.content,
            PromptSegmentContent::AssistantToolCalls { ref calls, .. }
                if calls.iter().any(|call| call.call_id == "call-1")
        )
    }));
    assert!(first.prompt_plan.segments.iter().any(|segment| {
        matches!(
            segment.content,
            PromptSegmentContent::ToolOutput { ref call_id, .. } if call_id == "call-1"
        )
    }));
    assert!(
        first
            .prompt_plan
            .segments
            .iter()
            .all(|segment| segment.text != "RETIRED-PLANNER-FRAME")
    );
}
