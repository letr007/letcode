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
fn history_aliases_share_protocol_item_type_and_json_shape() {
    let legacy = HistoryItem::AssistantToolCalls {
        text: Some("working".into()),
        calls: vec![HistoryToolCall {
            call_id: "call-1".into(),
            name: "fs__read".into(),
            arguments_json: r#"{"path":"src/main.rs"}"#.into(),
        }],
    };
    let canonical: crate::protocol_frames::ProtocolItem = legacy.clone();
    let legacy_again: HistoryItem = canonical.clone();

    assert_eq!(legacy_again, legacy);
    assert_eq!(
        serde_json::to_value(&legacy).expect("legacy item serializes"),
        serde_json::to_value(&canonical).expect("canonical item serializes")
    );
    assert_eq!(
        serde_json::to_string(&canonical).expect("canonical item serializes"),
        r#"{"kind":"assistant_tool_calls","text":"working","calls":[{"call_id":"call-1","name":"fs__read","arguments_json":"{\"path\":\"src/main.rs\"}"}]}"#
    );
}

#[test]
fn model_metadata_validation_preserves_error_precedence_and_messages() {
    let invalid = ModelRequestMetadata {
        effective_input_limit_tokens: Some(0),
        max_output_tokens: Some(u32::MAX as u64 + 1),
        temperature: Some(2.1),
        top_p: Some(1.1),
        ..Default::default()
    };

    assert_eq!(
        validate_model_metadata(invalid.clone())
            .expect_err("zero effective input limit is rejected first")
            .to_string(),
        "model.effective_input_limit_tokens must be greater than 0"
    );

    let without_zero_effective_limit = ModelRequestMetadata {
        effective_input_limit_tokens: Some(1),
        ..invalid.clone()
    };
    assert_eq!(
        validate_model_metadata(without_zero_effective_limit)
            .expect_err("output token overflow is rejected second")
            .to_string(),
        format!("model.max_output_tokens must be at most {}", u32::MAX)
    );

    let without_output_overflow = ModelRequestMetadata {
        effective_input_limit_tokens: Some(1),
        max_output_tokens: Some(u32::MAX as u64),
        ..invalid.clone()
    };
    assert_eq!(
        validate_model_metadata(without_output_overflow)
            .expect_err("temperature is rejected before top_p")
            .to_string(),
        "model.temperature must be between 0 and 2"
    );

    let without_invalid_temperature = ModelRequestMetadata {
        effective_input_limit_tokens: Some(1),
        max_output_tokens: Some(u32::MAX as u64),
        temperature: Some(2.0),
        ..invalid
    };
    assert_eq!(
        validate_model_metadata(without_invalid_temperature)
            .expect_err("top_p is rejected last")
            .to_string(),
        "model.top_p must be between 0 and 1"
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

#[test]
fn current_user_query_prefers_protected_suffix_then_falls_back_to_latest_global_user() {
    let history = vec![
        HistoryItem::user("global"),
        HistoryItem::assistant("intermediate"),
        HistoryItem::user("protected"),
        HistoryItem::assistant("latest"),
    ];

    assert_eq!(current_user_query(&history, 2), "protected");
    assert_eq!(current_user_query(&history, 4), "protected");
    assert_eq!(
        current_user_query(&[HistoryItem::assistant("no user")], 0),
        ""
    );
}

#[test]
fn sha256_matches_nist_vectors_and_canonical_json_is_key_order_independent() {
    assert_eq!(
        sha256_hex(b""),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        sha256_hex(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    assert_eq!(
        canonical_bytes(&serde_json::json!({"b": [2, 1], "a": {"z": true, "x": null}})),
        canonical_bytes(&serde_json::json!({"a": {"x": null, "z": true}, "b": [2, 1]}))
    );
}

#[test]
fn canonical_cache_input_uses_exact_serialized_protocol_shape() {
    let plan = build_prompt_plan(PromptPlanBuildInput {
        protocol: ApiProtocol::Responses,
        model_id: "gpt-test",
        prelude: &[PromptMessage::system("stable")],
        snapshot: &RuntimeSnapshot::new("canonical-cache-test"),
        selected_frames: &[],
        protected_suffix_len: 0,
        evidence_message: None,
        selected_evidence_ids: &[],
    });
    let tools = [ToolSpec {
        name: "read".into(),
        description: "Read a file".into(),
        parameters: json!({"type": "object", "properties": {"path": {"type": "string"}}}),
        strict: true,
    }];
    let model = ModelRequestMetadata {
        supports_tools: true,
        prompt_cache: PromptCacheConfig {
            enabled: true,
            retention: None,
            namespace: Some("test".into()),
        },
        ..Default::default()
    };

    let responses_request = build_responses_request("gpt-test", model.clone(), &plan, &tools);
    assert_eq!(responses_request.parallel_tool_calls, Some(false));
    let responses = serde_json::to_value(responses_request).expect("responses request serializes");
    let responses_canonical = canonical_cache_input(
        "test",
        ApiProtocol::Responses,
        "gpt-test",
        &plan.segments,
        &tools,
        true,
    );
    assert_eq!(responses_canonical["items"], responses["input"]);
    assert_eq!(responses_canonical["tools"], responses["tools"]);
    assert_eq!(responses_canonical["shape_version"], 2);
    assert_eq!(
        responses_canonical["input_shape"]["parallel_tool_calls"],
        false
    );
    assert_eq!(responses["parallel_tool_calls"], false);

    let chat_plan = PromptPlan {
        protocol: ApiProtocol::Completions,
        ..plan.clone()
    };
    let chat_request = build_completions_request("gpt-test", model, &chat_plan, &tools);
    assert_eq!(chat_request.parallel_tool_calls, Some(false));
    let chat = serde_json::to_value(chat_request).expect("chat request serializes");
    let chat_canonical = canonical_cache_input(
        "test",
        ApiProtocol::Completions,
        "gpt-test",
        &chat_plan.segments,
        &tools,
        true,
    );
    assert_eq!(chat_canonical["items"], chat["messages"]);
    assert_eq!(chat_canonical["tools"], chat["tools"]);
    assert_eq!(chat_canonical["input_shape"]["parallel_tool_calls"], false);
    assert_eq!(chat["parallel_tool_calls"], false);
}
use crate::agent::{ToolExecutionSummaryEvent, ValidationAdvisory};
use crate::context_tree::ContextNodeStatus;
use crate::context_view::{
    ContextBlock, ContextBlockId, ContextBlockKind, ContextBlockSource, ContextViewStatus,
    DEFAULT_FOLDED_OUTPUT_THRESHOLD_BYTES, FoldedOutputMetadata, project_context_view,
};
use crate::evidence::{EvidenceKind, EvidenceRecord, EvidenceSource};
use crate::protocol_frames::history_items_from_frames;
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

fn adapter_summary_texts(adapter: &HistoryAdapterProjection) -> Vec<String> {
    history_items_from_frames(&adapter.history_prefix)
        .into_iter()
        .filter_map(|item| match item {
            HistoryItem::ContextSummary { text } => Some(text),
            _ => None,
        })
        .collect()
}

fn sample_context_view(open_detail: bool) -> crate::context_view::ContextViewProjection {
    let mut records = vec![
        transcript_record(
            1,
            TranscriptEvent::UserMessage {
                content: UserMessageContent::from("Do not drop hard constraints"),
            },
        ),
        transcript_record(
            2,
            TranscriptEvent::AssistantMessage {
                content: "Pinned context note".into(),
            },
        ),
        transcript_record(
            3,
            TranscriptEvent::ContextViewOperationMetadata {
                operation: "pin".into(),
                node_id: None,
                block_id: Some("block-seq-2-note".into()),
                detail: None,
            },
        ),
        transcript_record(
            4,
            TranscriptEvent::ToolCallStarted {
                call_id: "call-1".into(),
                name: "shell__exec".into(),
                args: json!({"command": "cargo test"}),
            },
        ),
        transcript_record(
            5,
            TranscriptEvent::ToolCallFinished {
                call_id: "call-1".into(),
                name: "shell__exec".into(),
                ok: true,
                output: crate::tool::ToolResult::ok(
                    "shell__exec",
                    json!({
                        "status": 0,
                        "stdout": "x".repeat(5000),
                        "stdout_truncated": false,
                        "stderr": "",
                        "stderr_truncated": false
                    }),
                ),
            },
        ),
        transcript_record(
            6,
            TranscriptEvent::ContextSummaryArtifactMetadata {
                node_id: "node-a".into(),
                artifact_id: "sum-1".into(),
                artifact_kind: "summary".into(),
                version: Some(1),
                summary: Some("Summary text".into()),
                source_node_id: Some("node-a".into()),
                source_block_id: Some("block-seq-2-note".into()),
                source_start_sequence: Some(2),
                source_end_sequence: Some(2),
            },
        ),
    ];
    if open_detail {
        records.push(transcript_record(
            7,
            TranscriptEvent::ContextViewOperationMetadata {
                operation: "open_detail".into(),
                node_id: None,
                block_id: Some("block-seq-2-note".into()),
                detail: None,
            },
        ));
    }
    project_context_view(&records).expect("context view projection")
}

fn request_json(result: BuildResult) -> String {
    match result.request {
        BuiltRequest::Responses(request) => serde_json::to_string(&request).expect("serialize"),
        BuiltRequest::ResponsesCompatible(request)
        | BuiltRequest::CompletionsCompatible(request) => {
            serde_json::to_string(&request).expect("serialize")
        }
        BuiltRequest::Completions(request) => serde_json::to_string(&request).expect("serialize"),
    }
}

fn cache_config(retention: Option<PromptCacheRetention>) -> PromptCacheConfig {
    PromptCacheConfig {
        enabled: true,
        retention,
        namespace: Some("cache-test".into()),
    }
}

fn cache_test_result(
    protocol: ApiProtocol,
    prompt_cache: PromptCacheConfig,
    tools: &[ToolSpec],
) -> BuildResult {
    let mut model = metadata(8192);
    model.prompt_cache = prompt_cache;
    build_test_request(TestRequestBuilderInput {
        protocol,
        model_id: "cache-model",
        model,
        prelude: &[PromptMessage::system("stable instructions")],
        history: &[
            HistoryItem::assistant("prior answer"),
            HistoryItem::user("current question"),
        ],
        protected_start_index: 1,
        tools,
        evidence: &[],
        history_adapter: None,
        context_view: None,
    })
    .expect("cache test request builds")
}

#[test]
fn configured_reasoning_efforts_constrain_selectable_levels() {
    let metadata = ModelRequestMetadata {
        supports_reasoning: true,
        reasoning_efforts: vec![
            ModelReasoningEffort::None,
            ModelReasoningEffort::Low,
            ModelReasoningEffort::Max,
        ],
        ..Default::default()
    };

    assert!(metadata.allows_reasoning_effort(&ModelReasoningEffort::Low));
    assert!(metadata.allows_reasoning_effort(&ModelReasoningEffort::Max));
    assert!(!metadata.allows_reasoning_effort(&ModelReasoningEffort::High));
}

#[test]
fn implicit_reasoning_efforts_include_the_active_compatible_effort() {
    let metadata = ModelRequestMetadata {
        supports_reasoning: true,
        reasoning_effort: Some(ModelReasoningEffort::Custom("provider-ultra".into())),
        ..Default::default()
    };

    assert_eq!(
        metadata.selectable_reasoning_efforts(),
        [
            DEFAULT_REASONING_EFFORTS.as_slice(),
            &[ModelReasoningEffort::Custom("provider-ultra".into())],
        ]
        .concat()
    );
    assert!(
        metadata.allows_reasoning_effort(&ModelReasoningEffort::Custom("provider-ultra".into()))
    );
    assert!(metadata.reasoning_efforts.is_empty());
}

fn request_value(result: &BuildResult) -> Value {
    match &result.request {
        BuiltRequest::Responses(request) => serde_json::to_value(request),
        BuiltRequest::ResponsesCompatible(request)
        | BuiltRequest::CompletionsCompatible(request) => Ok(request.clone()),
        BuiltRequest::Completions(request) => serde_json::to_value(request),
    }
    .expect("request serializes")
}

#[test]
fn logical_request_categories_follow_mutated_prompt_segment_sources() {
    let rebuild_with_mutation = |mut build: BuildResult, segment_index: usize, text: &str| {
        let segment = &mut build.prompt_plan.segments[segment_index];
        segment.text = text.into();
        segment.content = PromptSegmentContent::Text { text: text.into() };
        build_request_from_selected_prompt(SelectedPromptRequestInput {
            protocol: build.prompt_plan.protocol,
            model_id: "gpt-test",
            model: metadata(8192),
            tools: &[],
            prompt_plan: build.prompt_plan,
            budget: build.budget,
            selected_evidence_ids: build.selected_evidence_ids,
            selected_evidence_message: build.selected_evidence_message,
        })
        .expect("mutated prompt rebuilds")
    };

    let stable = build_test_request(TestRequestBuilderInput {
        protocol: ApiProtocol::Responses,
        model_id: "gpt-test",
        model: metadata(8192),
        prelude: &[PromptMessage::developer("stable developer prelude")],
        history: &[HistoryItem::user("question")],
        protected_start_index: 0,
        tools: &[],
        evidence: &[],
        history_adapter: None,
        context_view: None,
    })
    .expect("stable request builds");
    let stable_changed = rebuild_with_mutation(stable, 0, "mutated developer prelude");
    assert_eq!(
        observe_logical_request(&stable_changed).units[0].category,
        LogicalRequestUnitCategory::StableKernel
    );

    let mut runtime_model = metadata(8192);
    let runtime = build_test_request(TestRequestBuilderInput {
        protocol: ApiProtocol::Responses,
        model_id: "gpt-test",
        model: runtime_model,
        prelude: &[PromptMessage::developer_with_origin(
            "Runtime context: original",
            PromptMessageOrigin::RuntimeClock,
        )],
        history: &[HistoryItem::user("question")],
        protected_start_index: 0,
        tools: &[],
        evidence: &[],
        history_adapter: None,
        context_view: None,
    })
    .expect("runtime request builds");
    let runtime_index = runtime
        .prompt_plan
        .segments
        .iter()
        .position(|segment| {
            segment.source.contributor_kind == PromptContributorKind::RuntimeContext
        })
        .expect("runtime context segment");
    let runtime_changed = rebuild_with_mutation(runtime, runtime_index, "Runtime context: mutated");
    assert_eq!(
        observe_logical_request(&runtime_changed).units[runtime_index].category,
        LogicalRequestUnitCategory::RuntimeContext
    );

    let snapshot = RuntimeSnapshot::new("evidence-category-test");
    let frozen = FrozenEvidence {
        message: Some("original evidence".into()),
        selected_ids: vec!["evidence-1".into()],
    };
    let evidence = build_request_with_frozen(
        RequestBuilderInput {
            protocol: ApiProtocol::Responses,
            model_id: "gpt-test",
            model: metadata(8192),
            prelude: &[],
            snapshot: &snapshot,
            tools: &[],
        },
        Some(&frozen),
    )
    .expect("evidence request builds");
    let evidence_index = evidence
        .prompt_plan
        .segments
        .iter()
        .position(|segment| segment.source.contributor_kind == PromptContributorKind::Evidence)
        .expect("evidence segment");
    let evidence_changed = rebuild_with_mutation(evidence, evidence_index, "mutated evidence");
    assert_eq!(
        observe_logical_request(&evidence_changed).units[evidence_index].category,
        LogicalRequestUnitCategory::Evidence
    );
}

fn without_cache_fields(mut request: Value) -> Value {
    let fields = request
        .as_object_mut()
        .expect("serialized request is an object");
    fields.remove("prompt_cache_key");
    fields.remove("prompt_cache_retention");
    request
}

#[test]
fn prompt_cache_serialization_and_omission_follow_protocol_and_prefix() {
    for (protocol, retention, expected_retention) in [
        (
            ApiProtocol::Responses,
            Some(PromptCacheRetention::InMemory),
            Some("in_memory"),
        ),
        (
            ApiProtocol::Responses,
            Some(PromptCacheRetention::TwentyFourHours),
            Some("24h"),
        ),
        (ApiProtocol::Responses, None, None),
        (
            ApiProtocol::Completions,
            Some(PromptCacheRetention::InMemory),
            None,
        ),
    ] {
        let result = cache_test_result(protocol, cache_config(retention), &[]);
        let request = request_value(&result);
        let key = request["prompt_cache_key"]
            .as_str()
            .expect("enabled stable cache serializes a key");
        assert!(key.starts_with("lc-pc-v2-"));
        assert_eq!(key.len(), 41);
        assert!(key.bytes().all(|byte| byte.is_ascii()));
        assert!(
            key[9..]
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        );
        match expected_retention {
            Some(retention) => assert_eq!(request["prompt_cache_retention"], retention),
            None => assert!(request.get("prompt_cache_retention").is_none()),
        }
        assert_eq!(result.cache.hint_serialized, true);
        assert_eq!(
            result.cache.retention_sent,
            retention.filter(|_| protocol == ApiProtocol::Responses)
        );
    }

    let disabled = cache_test_result(ApiProtocol::Responses, PromptCacheConfig::default(), &[]);
    let disabled_request = request_value(&disabled);
    assert!(disabled_request.get("prompt_cache_key").is_none());
    assert!(disabled_request.get("prompt_cache_retention").is_none());
    assert!(disabled.cache.local_prefix_segments > 0);
    assert!(!disabled.cache.configured);
    assert!(!disabled.cache.hint_serialized);
    assert_eq!(disabled.cache.retention_sent, None);
    assert_eq!(disabled.cache.local_prefix_fingerprint, None);
    assert_eq!(disabled.cache.routing_key, None);

    let mut model = metadata(8192);
    model.prompt_cache = cache_config(Some(PromptCacheRetention::InMemory));
    let no_prefix = build_test_request(TestRequestBuilderInput {
        protocol: ApiProtocol::Responses,
        model_id: "cache-model",
        model,
        prelude: &[],
        history: &[HistoryItem::user("current question")],
        protected_start_index: 0,
        tools: &[],
        evidence: &[],
        history_adapter: None,
        context_view: None,
    })
    .expect("zero-prefix request builds");
    let no_prefix_request = request_value(&no_prefix);
    assert!(no_prefix_request.get("prompt_cache_key").is_none());
    assert!(no_prefix_request.get("prompt_cache_retention").is_none());
    assert_eq!(no_prefix.cache.local_prefix_segments, 0);
    assert!(no_prefix.cache.configured);
    assert!(!no_prefix.cache.hint_serialized);
    assert_eq!(no_prefix.cache.local_prefix_fingerprint, None);
    assert_eq!(no_prefix.cache.routing_key, None);
}

#[test]
fn prompt_cache_is_a_provider_noop_and_budget_reports_match_final_plan() {
    let tools = [
        ToolSpec {
            name: "read".into(),
            description: "Read a file".into(),
            parameters: json!({"type": "object", "properties": {"path": {"type": "string"}}}),
            strict: true,
        },
        ToolSpec {
            name: "write".into(),
            description: "Write a file".into(),
            parameters: json!({"type": "object", "properties": {"path": {"type": "string"}}}),
            strict: true,
        },
    ];
    for protocol in [ApiProtocol::Responses, ApiProtocol::Completions] {
        let disabled = cache_test_result(protocol, PromptCacheConfig::default(), &tools);
        let enabled = cache_test_result(protocol, cache_config(None), &tools);

        assert_eq!(
            without_cache_fields(request_value(&disabled)),
            without_cache_fields(request_value(&enabled)),
            "cache controls must not change provider content"
        );
        assert_eq!(disabled.prompt_plan, enabled.prompt_plan);
        assert_eq!(
            disabled.selected_evidence_ids,
            enabled.selected_evidence_ids
        );
        assert_eq!(disabled.budget, enabled.budget);

        let report = enabled.prompt_plan.token_report();
        assert_eq!(
            enabled.budget.plan_total_prompt_tokens,
            report.total_prompt_tokens
        );
        assert_eq!(
            enabled.budget.plan_stable_prompt_tokens,
            report.stable_prompt_tokens
        );
        assert_eq!(
            enabled.budget.plan_volatile_prompt_tokens,
            report.volatile_prompt_tokens
        );
        assert_eq!(
            enabled.budget.plan_cacheable_prefix_tokens,
            report.cacheable_prefix_tokens
        );
        assert_eq!(
            enabled.budget.plan_stable_after_boundary_tokens,
            report.stable_after_boundary_tokens
        );
        assert_eq!(
            enabled.budget.estimated_request_tokens,
            report.total_prompt_tokens + enabled.budget.estimated_tools_tokens
        );
        assert_eq!(
            enabled.budget.estimated_tools_tokens,
            estimate_tools_tokens(&tools)
        );
        assert_eq!(
            enabled.budget.input_budget_tokens,
            effective_input_budget_tokens(enabled_model_metadata(), &tools)
        );
        assert_eq!(
            enabled.cache.local_prefix_segments,
            enabled.prompt_plan.cacheable_prefix_len()
        );
        let namespace = "cache-test";
        let canonical = canonical_cache_input(
            namespace,
            protocol,
            "cache-model",
            &enabled.prompt_plan.segments[..enabled.cache.local_prefix_segments],
            &tools,
            true,
        );
        let request = request_value(&enabled);
        let rendered = match protocol {
            ApiProtocol::Responses => &request["input"],
            ApiProtocol::Completions => &request["messages"],
        };
        assert_eq!(
            canonical["items"],
            Value::Array(
                rendered.as_array().expect("request items")[..enabled.cache.local_prefix_segments]
                    .to_vec()
            )
        );
    }
}

fn enabled_model_metadata() -> ModelRequestMetadata {
    metadata(8192)
}

#[test]
fn prompt_cache_fingerprints_and_routing_keys_follow_identity_boundaries() {
    let tools = [
        ToolSpec {
            name: "read".into(),
            description: "Read".into(),
            parameters: json!({"type": "object"}),
            strict: true,
        },
        ToolSpec {
            name: "write".into(),
            description: "Write".into(),
            parameters: json!({"type": "object"}),
            strict: true,
        },
    ];
    let base = cache_test_result(ApiProtocol::Responses, cache_config(None), &tools);
    let base_report = base.cache.clone();
    let report = |namespace: &str,
                  protocol: ApiProtocol,
                  model: &str,
                  plan: &PromptPlan,
                  tool_specs: &[ToolSpec],
                  supports_tools: bool| {
        prompt_cache_report(
            protocol,
            model,
            &PromptCacheConfig {
                enabled: true,
                retention: None,
                namespace: Some(namespace.into()),
            },
            plan,
            tool_specs,
            supports_tools,
        )
    };
    let base_again = report(
        "cache-test",
        ApiProtocol::Responses,
        "cache-model",
        &base.prompt_plan,
        &tools,
        true,
    );
    assert_eq!(base_report, base_again);
    assert!(
        base_report
            .local_prefix_fingerprint
            .as_deref()
            .is_some_and(|fingerprint| fingerprint.starts_with("ppf-v2-"))
    );

    let mut changed_stable = base.prompt_plan.clone();
    changed_stable.segments[0].text = "changed stable instructions".into();
    changed_stable.segments[0].content = PromptSegmentContent::Text {
        text: "changed stable instructions".into(),
    };
    let mut changed_role = base.prompt_plan.clone();
    changed_role.segments[0].role = PromptSegmentRole::Developer;
    let mut changed_schema = tools.clone();
    changed_schema[0].parameters = json!({"type": "object", "additionalProperties": false});
    let mut changed_image = base.prompt_plan.clone();
    changed_image.segments[0].role = PromptSegmentRole::User;
    changed_image.segments[0].content = PromptSegmentContent::UserContent {
        content: UserMessageContent::new(
            "image",
            vec![UserImageAttachment {
                id: "image-1".into(),
                label: "image.png".into(),
                mime: "image/png".into(),
                data_url: "data:image/png;base64,AAAA".into(),
            }],
        ),
    };
    let mut changed_suffix = base.prompt_plan.clone();
    let suffix = changed_suffix.segments.last_mut().expect("volatile suffix");
    suffix.text = "changed volatile suffix".into();
    suffix.content = PromptSegmentContent::Text {
        text: "changed volatile suffix".into(),
    };
    let mut stable_after_boundary = base.prompt_plan.clone();
    stable_after_boundary
        .segments
        .push(base.prompt_plan.segments[0].clone());
    let after_boundary = stable_after_boundary
        .segments
        .last_mut()
        .expect("appended segment");
    after_boundary.text = "changed stable after boundary".into();
    after_boundary.content = PromptSegmentContent::Text {
        text: "changed stable after boundary".into(),
    };

    for changed in [
        report(
            "cache-test",
            ApiProtocol::Responses,
            "cache-model",
            &changed_stable,
            &tools,
            true,
        ),
        report(
            "cache-test",
            ApiProtocol::Responses,
            "cache-model",
            &changed_role,
            &tools,
            true,
        ),
        report(
            "cache-test",
            ApiProtocol::Completions,
            "cache-model",
            &base.prompt_plan,
            &tools,
            true,
        ),
        report(
            "other",
            ApiProtocol::Responses,
            "cache-model",
            &base.prompt_plan,
            &tools,
            true,
        ),
        report(
            "cache-test",
            ApiProtocol::Responses,
            "other-model",
            &base.prompt_plan,
            &tools,
            true,
        ),
        report(
            "cache-test",
            ApiProtocol::Responses,
            "cache-model",
            &base.prompt_plan,
            &changed_schema,
            true,
        ),
        report(
            "cache-test",
            ApiProtocol::Responses,
            "cache-model",
            &base.prompt_plan,
            &tools[..1],
            true,
        ),
        report(
            "cache-test",
            ApiProtocol::Responses,
            "cache-model",
            &base.prompt_plan,
            &tools.iter().cloned().rev().collect::<Vec<_>>(),
            true,
        ),
        report(
            "cache-test",
            ApiProtocol::Responses,
            "cache-model",
            &base.prompt_plan,
            &tools,
            false,
        ),
        report(
            "cache-test",
            ApiProtocol::Responses,
            "cache-model",
            &changed_image,
            &tools,
            true,
        ),
    ] {
        assert_ne!(
            changed.local_prefix_fingerprint,
            base_report.local_prefix_fingerprint
        );
    }

    for unchanged in [
        report(
            "cache-test",
            ApiProtocol::Responses,
            "cache-model",
            &changed_suffix,
            &tools,
            true,
        ),
        report(
            "cache-test",
            ApiProtocol::Responses,
            "cache-model",
            &stable_after_boundary,
            &tools,
            true,
        ),
    ] {
        assert_eq!(
            unchanged.local_prefix_fingerprint,
            base_report.local_prefix_fingerprint
        );
        assert_eq!(unchanged.routing_key, base_report.routing_key);
    }

    assert_eq!(
        report(
            "cache-test",
            ApiProtocol::Responses,
            "cache-model",
            &changed_stable,
            &tools,
            true
        )
        .routing_key,
        base_report.routing_key
    );
    assert_ne!(
        report(
            "other",
            ApiProtocol::Responses,
            "cache-model",
            &base.prompt_plan,
            &tools,
            true
        )
        .routing_key,
        base_report.routing_key
    );
    assert_ne!(
        report(
            "cache-test",
            ApiProtocol::Completions,
            "cache-model",
            &base.prompt_plan,
            &tools,
            true
        )
        .routing_key,
        base_report.routing_key
    );
    assert_ne!(
        report(
            "cache-test",
            ApiProtocol::Responses,
            "other-model",
            &base.prompt_plan,
            &tools,
            true
        )
        .routing_key,
        base_report.routing_key
    );
    assert_ne!(
        report(
            "cache-test",
            ApiProtocol::Responses,
            "cache-model",
            &base.prompt_plan,
            &changed_schema,
            true
        )
        .routing_key,
        base_report.routing_key
    );
    assert_ne!(
        report(
            "cache-test",
            ApiProtocol::Responses,
            "cache-model",
            &base.prompt_plan,
            &tools[..1],
            true
        )
        .routing_key,
        base_report.routing_key
    );
    assert_ne!(
        report(
            "cache-test",
            ApiProtocol::Responses,
            "cache-model",
            &base.prompt_plan,
            &tools.iter().cloned().rev().collect::<Vec<_>>(),
            true
        )
        .routing_key,
        base_report.routing_key
    );
    assert_ne!(
        report(
            "cache-test",
            ApiProtocol::Responses,
            "cache-model",
            &base.prompt_plan,
            &tools,
            false
        )
        .routing_key,
        base_report.routing_key
    );
}

#[test]
fn selected_prompt_rebuild_preserves_responses_request_and_metadata() {
    let prelude = vec![PromptMessage::system("system")];
    let history = vec![
        HistoryItem::assistant("older assistant"),
        HistoryItem::user("latest user"),
    ];
    let evidence = vec![evidence("ev-1", "summary", "src/main.rs", 1)];
    let original = build_test_request(TestRequestBuilderInput {
        protocol: ApiProtocol::Responses,
        model_id: "gpt-test",
        model: metadata(8192),
        prelude: &prelude,
        history: &history,
        protected_start_index: 1,
        tools: &[],
        evidence: &evidence,
        history_adapter: None,
        context_view: None,
    })
    .expect("request builds");

    let rebuilt = build_request_from_selected_prompt(SelectedPromptRequestInput {
        protocol: ApiProtocol::Responses,
        model_id: "gpt-test",
        model: metadata(8192),
        tools: &[],
        prompt_plan: original.prompt_plan.clone(),
        budget: original.budget,
        selected_evidence_ids: original.selected_evidence_ids.clone(),
        selected_evidence_message: original.selected_evidence_message.clone(),
    })
    .expect("selected prompt rebuilds");

    assert_eq!(
        request_json(original.clone()),
        request_json(rebuilt.clone())
    );
    assert_eq!(rebuilt.budget, original.budget);
    assert_eq!(rebuilt.prompt_plan, original.prompt_plan);
    assert_eq!(
        rebuilt.selected_evidence_ids,
        original.selected_evidence_ids
    );
}

#[test]
fn builds_responses_request_from_unified_history() {
    let history = vec![HistoryItem::user("hello"), HistoryItem::assistant("hi")];
    let result = build_test_request(TestRequestBuilderInput {
        protocol: ApiProtocol::Responses,
        model_id: "gpt-test",
        model: metadata(8192),
        prelude: &[],
        history: &history,
        protected_start_index: 0,
        tools: &[],
        evidence: &[],
        history_adapter: None,
        context_view: None,
    })
    .expect("request builds");

    let BuiltRequest::Responses(_) = &result.request else {
        panic!("expected responses request");
    };
    let request = request_value(&result);
    assert_eq!(request["stream"], true);
    assert!(request.to_string().contains("hello"));
    assert!(request.to_string().contains("hi"));
}

#[test]
fn responses_request_includes_model_generation_parameters() {
    let result = build_test_request(TestRequestBuilderInput {
        protocol: ApiProtocol::Responses,
        model_id: "gpt-test",
        model: ModelRequestMetadata {
            context_window: Some(8192),
            effective_input_limit_tokens: None,
            max_output_tokens: Some(2048),
            supports_tools: true,
            supports_reasoning: true,
            reasoning_effort: Some(ModelReasoningEffort::High),
            reasoning_summary: Some(ModelReasoningSummary::Auto),
            text_verbosity: Some(ModelTextVerbosity::Low),
            temperature: Some(0.2),
            top_p: Some(0.8),
            ..ModelRequestMetadata::default()
        },
        prelude: &[],
        history: &[HistoryItem::user("hello")],
        protected_start_index: 0,
        tools: &[],
        evidence: &[],
        history_adapter: None,
        context_view: None,
    })
    .expect("request builds");

    let BuiltRequest::Responses(_) = &result.request else {
        panic!("expected responses request");
    };
    let json = request_value(&result);

    assert_eq!(json["max_output_tokens"], 2048);
    assert_eq!(json["stream"], true);
    assert_eq!(json["reasoning"]["effort"], "high");
    assert_eq!(json["reasoning"]["summary"], "auto");
    assert_eq!(json["text"]["verbosity"], "low");
    assert_json_f64_close(&json["temperature"], 0.2);
    assert_json_f64_close(&json["top_p"], 0.8);
}

#[test]
fn builds_completions_request_from_unified_history() {
    let history = vec![HistoryItem::user("hello")];
    let result = build_test_request(TestRequestBuilderInput {
        protocol: ApiProtocol::Completions,
        model_id: "chat-test",
        model: metadata(8192),
        prelude: &[],
        history: &history,
        protected_start_index: 0,
        tools: &[],
        evidence: &[],
        history_adapter: None,
        context_view: None,
    })
    .expect("request builds");

    let BuiltRequest::Completions(request) = result.request else {
        panic!("expected completions request");
    };
    assert_eq!(request.model, "chat-test");
    assert_eq!(request.messages.len(), 1);
    assert_eq!(request.stream, Some(true));
    assert_eq!(
        request
            .stream_options
            .as_ref()
            .and_then(|options| options.include_usage),
        Some(true)
    );
}

#[test]
fn completions_request_serializes_multimodal_user_message_parts() {
    let history = vec![HistoryItem::user_content(UserMessageContent::new(
        "describe this image",
        vec![UserImageAttachment {
            id: "img-1".into(),
            label: "screen.png".into(),
            mime: "image/png".into(),
            data_url: "data:image/png;base64,AAAA".into(),
        }],
    ))];
    let result = build_test_request(TestRequestBuilderInput {
        protocol: ApiProtocol::Completions,
        model_id: "chat-test",
        model: metadata(8192),
        prelude: &[],
        history: &history,
        protected_start_index: 0,
        tools: &[],
        evidence: &[],
        history_adapter: None,
        context_view: None,
    })
    .expect("request builds");

    let BuiltRequest::Completions(request) = result.request else {
        panic!("expected completions request");
    };
    let json = serde_json::to_value(&request).expect("request serializes");
    let content = &json["messages"][0]["content"];
    assert_eq!(content[0]["type"], "text");
    assert_eq!(content[0]["text"], "describe this image");
    assert_eq!(content[1]["type"], "image_url");
    assert_eq!(content[1]["image_url"]["url"], "data:image/png;base64,AAAA");
}

#[test]
fn responses_request_serializes_multimodal_user_message_parts() {
    let history = vec![HistoryItem::user_content(UserMessageContent::new(
        "describe this image",
        vec![UserImageAttachment {
            id: "img-1".into(),
            label: "screen.png".into(),
            mime: "image/png".into(),
            data_url: "data:image/png;base64,AAAA".into(),
        }],
    ))];
    let result = build_test_request(TestRequestBuilderInput {
        protocol: ApiProtocol::Responses,
        model_id: "resp-test",
        model: metadata(8192),
        prelude: &[],
        history: &history,
        protected_start_index: 0,
        tools: &[],
        evidence: &[],
        history_adapter: None,
        context_view: None,
    })
    .expect("request builds");

    let BuiltRequest::Responses(request) = result.request else {
        panic!("expected responses request");
    };
    let json = serde_json::to_value(&request).expect("request serializes");
    let content = &json["input"][0]["content"];
    assert_eq!(json["input"][0]["type"], "message");
    assert_eq!(json["input"][0]["role"], "user");
    assert_eq!(content[0]["type"], "input_text");
    assert_eq!(content[0]["text"], "describe this image");
    assert_eq!(content[1]["type"], "input_image");
    assert_eq!(content[1]["image_url"], "data:image/png;base64,AAAA");
}

#[test]
fn responses_request_serializes_max_reasoning_effort_through_compatible_payload() {
    let result = build_test_request(TestRequestBuilderInput {
        protocol: ApiProtocol::Responses,
        model_id: "gpt-5.6-terra",
        model: ModelRequestMetadata {
            supports_reasoning: true,
            reasoning_effort: Some(ModelReasoningEffort::Max),
            reasoning_summary: Some(ModelReasoningSummary::Auto),
            ..metadata(8192)
        },
        prelude: &[],
        history: &[HistoryItem::user("hello")],
        protected_start_index: 0,
        tools: &[],
        evidence: &[],
        history_adapter: None,
        context_view: None,
    })
    .expect("request builds");

    let BuiltRequest::ResponsesCompatible(_) = &result.request else {
        panic!("expected compatible responses request");
    };
    let request = request_value(&result);
    assert_eq!(request["reasoning"]["effort"], "max");
    assert_eq!(request["reasoning"]["summary"], "auto");
    assert_eq!(request["stream"], true);
}

#[test]
fn responses_request_serializes_max_reasoning_effort_without_summary() {
    let result = build_test_request(TestRequestBuilderInput {
        protocol: ApiProtocol::Responses,
        model_id: "gpt-5.6-terra",
        model: ModelRequestMetadata {
            supports_reasoning: true,
            reasoning_effort: Some(ModelReasoningEffort::Max),
            ..metadata(8192)
        },
        prelude: &[],
        history: &[HistoryItem::user("hello")],
        protected_start_index: 0,
        tools: &[],
        evidence: &[],
        history_adapter: None,
        context_view: None,
    })
    .expect("request builds");

    let BuiltRequest::ResponsesCompatible(request) = result.request else {
        panic!("expected compatible responses request");
    };
    assert_eq!(request["reasoning"]["effort"], "max");
    assert!(request["reasoning"].get("summary").is_none());
}

#[test]
fn completions_request_serializes_max_reasoning_effort_through_compatible_payload() {
    let result = build_test_request(TestRequestBuilderInput {
        protocol: ApiProtocol::Completions,
        model_id: "gpt-5.6-terra",
        model: ModelRequestMetadata {
            supports_reasoning: true,
            reasoning_effort: Some(ModelReasoningEffort::Max),
            ..metadata(8192)
        },
        prelude: &[],
        history: &[HistoryItem::user("hello")],
        protected_start_index: 0,
        tools: &[],
        evidence: &[],
        history_adapter: None,
        context_view: None,
    })
    .expect("request builds");

    let BuiltRequest::CompletionsCompatible(request) = result.request else {
        panic!("expected compatible chat completions request");
    };
    assert_eq!(request["reasoning_effort"], "max");
}

#[test]
fn custom_reasoning_effort_serializes_through_compatible_payloads_without_panicking() {
    for (protocol, expected_field) in [
        (ApiProtocol::Responses, "reasoning.effort"),
        (ApiProtocol::Completions, "reasoning_effort"),
    ] {
        let result = build_test_request(TestRequestBuilderInput {
            protocol,
            model_id: "provider-model",
            model: ModelRequestMetadata {
                supports_reasoning: true,
                reasoning_effort: Some(ModelReasoningEffort::Custom("provider-ultra".into())),
                ..metadata(8192)
            },
            prelude: &[],
            history: &[HistoryItem::user("hello")],
            protected_start_index: 0,
            tools: &[],
            evidence: &[],
            history_adapter: None,
            context_view: None,
        })
        .expect("custom compatible request builds without panic");

        let request = request_value(&result);
        match protocol {
            ApiProtocol::Responses => {
                assert!(matches!(
                    result.request,
                    BuiltRequest::ResponsesCompatible(_)
                ));
                assert_eq!(expected_field, "reasoning.effort");
                assert_eq!(request["reasoning"]["effort"], "provider-ultra");
            }
            ApiProtocol::Completions => {
                assert!(matches!(
                    result.request,
                    BuiltRequest::CompletionsCompatible(_)
                ));
                assert_eq!(expected_field, "reasoning_effort");
                assert_eq!(request["reasoning_effort"], "provider-ultra");
            }
        }
    }
}

#[test]
fn completions_request_includes_model_generation_parameters() {
    let result = build_test_request(TestRequestBuilderInput {
        protocol: ApiProtocol::Completions,
        model_id: "chat-test",
        model: ModelRequestMetadata {
            context_window: Some(8192),
            effective_input_limit_tokens: None,
            max_output_tokens: Some(2048),
            supports_tools: true,
            supports_reasoning: true,
            reasoning_effort: Some(ModelReasoningEffort::Minimal),
            reasoning_summary: Some(ModelReasoningSummary::Detailed),
            text_verbosity: Some(ModelTextVerbosity::High),
            temperature: Some(0.3),
            top_p: Some(0.7),
            ..ModelRequestMetadata::default()
        },
        prelude: &[],
        history: &[HistoryItem::user("hello")],
        protected_start_index: 0,
        tools: &[],
        evidence: &[],
        history_adapter: None,
        context_view: None,
    })
    .expect("request builds");

    let BuiltRequest::Completions(request) = result.request else {
        panic!("expected completions request");
    };
    let json = serde_json::to_value(&request).expect("request serializes");

    assert_eq!(json["max_completion_tokens"], 2048);
    assert_eq!(json["reasoning_effort"], "minimal");
    assert_eq!(json["verbosity"], "high");
    assert_json_f64_close(&json["temperature"], 0.3);
    assert_json_f64_close(&json["top_p"], 0.7);
    assert!(json.get("reasoning_summary").is_none());
}

fn assert_json_f64_close(value: &serde_json::Value, expected: f64) {
    let actual = value.as_f64().expect("value should be a number");
    assert!(
        (actual - expected).abs() < 0.000_001,
        "{actual} != {expected}"
    );
}

#[test]
fn responses_prelude_is_stable_prefix_before_history() {
    let prelude = vec![
        PromptMessage::system("stable system"),
        PromptMessage::developer("stable developer"),
    ];
    let history = vec![HistoryItem::user("current user")];

    let result = build_test_request(TestRequestBuilderInput {
        protocol: ApiProtocol::Responses,
        model_id: "gpt-test",
        model: metadata(8192),
        prelude: &prelude,
        history: &history,
        protected_start_index: 0,
        tools: &[],
        evidence: &[],
        history_adapter: None,
        context_view: None,
    })
    .expect("request builds");

    let BuiltRequest::Responses(request) = result.request else {
        panic!("expected responses request");
    };
    let json = serde_json::to_value(&request).expect("request serializes");
    let input = json["input"].as_array().expect("input should be array");
    assert_eq!(input[0]["role"], "system");
    assert_eq!(input[1]["role"], "developer");
    assert_eq!(input[2]["role"], "user");
    assert!(result.budget.estimated_prelude_tokens > 0);
}

#[test]
fn context_summary_is_encoded_as_developer_message_for_both_protocols() {
    let history = vec![HistoryItem::context_summary("目标\n- 修复 compaction")];

    let responses = build_test_request(TestRequestBuilderInput {
        protocol: ApiProtocol::Responses,
        model_id: "gpt-test",
        model: metadata(8192),
        prelude: &[],
        history: &history,
        protected_start_index: 0,
        tools: &[],
        evidence: &[],
        history_adapter: None,
        context_view: None,
    })
    .expect("responses request builds");
    let BuiltRequest::Responses(response_request) = responses.request else {
        panic!("expected responses request");
    };
    let response_json = serde_json::to_string(&response_request).expect("serialize response");
    assert!(response_json.contains("developer"));
    assert!(response_json.contains("以下是当前会话的结构化摘要"));

    let completions = build_test_request(TestRequestBuilderInput {
        protocol: ApiProtocol::Completions,
        model_id: "gpt-test",
        model: metadata(8192),
        prelude: &[],
        history: &history,
        protected_start_index: 0,
        tools: &[],
        evidence: &[],
        history_adapter: None,
        context_view: None,
    })
    .expect("completions request builds");
    let BuiltRequest::Completions(chat_request) = completions.request else {
        panic!("expected completions request");
    };
    assert!(matches!(
        chat_request.messages[0],
        ChatCompletionRequestMessage::Developer(_)
    ));
    let chat_json = serde_json::to_string(&chat_request.messages[0]).expect("serialize chat");
    assert!(chat_json.contains("以下是当前会话的结构化摘要"));
}

#[test]
fn orphan_tool_outputs_fail_fast_when_building_chat_request() {
    let history = vec![
        HistoryItem::context_summary("旧工具调用已总结"),
        HistoryItem::ToolOutput {
            call_id: "call-orphan".into(),
            output_json: r#"{"ok":true}"#.into(),
        },
        HistoryItem::user("continue"),
    ];

    let error = build_test_request(TestRequestBuilderInput {
        protocol: ApiProtocol::Completions,
        model_id: "chat-test",
        model: metadata(8192),
        prelude: &[],
        history: &history,
        protected_start_index: 2,
        tools: &[],
        evidence: &[],
        history_adapter: None,
        context_view: None,
    })
    .expect_err("orphan tool output must fail");
    assert!(error.to_string().contains("orphan tool output"));
}

#[test]
fn complete_tool_call_output_pairs_are_kept_when_building_chat_request() {
    let history = vec![
        HistoryItem::AssistantToolCalls {
            text: None,
            calls: vec![HistoryToolCall {
                call_id: "call-read".into(),
                name: "fs__read".into(),
                arguments_json: r#"{"path":"src/main.rs"}"#.into(),
            }],
        },
        HistoryItem::ToolOutput {
            call_id: "call-read".into(),
            output_json: r#"{"ok":true}"#.into(),
        },
        HistoryItem::user("continue"),
    ];

    let result = build_test_request(TestRequestBuilderInput {
        protocol: ApiProtocol::Completions,
        model_id: "chat-test",
        model: metadata(8192),
        prelude: &[],
        history: &history,
        protected_start_index: 2,
        tools: &[],
        evidence: &[],
        history_adapter: None,
        context_view: None,
    })
    .expect("request builds");

    let BuiltRequest::Completions(request) = result.request else {
        panic!("expected chat completions request");
    };
    assert!(
        request
            .messages
            .iter()
            .any(|message| matches!(message, ChatCompletionRequestMessage::Assistant(_)))
    );
    assert!(
        request
            .messages
            .iter()
            .any(|message| matches!(message, ChatCompletionRequestMessage::Tool(_)))
    );
}

#[test]
fn completions_prelude_is_stable_prefix_before_history() {
    let prelude = vec![
        PromptMessage::system("stable system"),
        PromptMessage::developer("stable developer"),
    ];
    let history = vec![HistoryItem::user("current user")];

    let result = build_test_request(TestRequestBuilderInput {
        protocol: ApiProtocol::Completions,
        model_id: "chat-test",
        model: metadata(8192),
        prelude: &prelude,
        history: &history,
        protected_start_index: 0,
        tools: &[],
        evidence: &[],
        history_adapter: None,
        context_view: None,
    })
    .expect("request builds");

    let BuiltRequest::Completions(request) = result.request else {
        panic!("expected completions request");
    };
    assert_eq!(request.messages.len(), 3);
    assert!(matches!(
        request.messages[0],
        ChatCompletionRequestMessage::System(_)
    ));
    assert!(matches!(
        request.messages[1],
        ChatCompletionRequestMessage::Developer(_)
    ));
    assert!(matches!(
        request.messages[2],
        ChatCompletionRequestMessage::User(_)
    ));
    assert!(result.budget.estimated_prelude_tokens > 0);
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
        protocol: ApiProtocol::Responses,
        model_id: "gpt-test",
        model: metadata(1200),
        prelude: &[],
        history: &history,
        protected_start_index: 2,
        tools: &[],
        evidence: &[],
        history_adapter: None,
        context_view: None,
    })
    .expect("request builds");

    assert!(result.budget.truncated);
    let BuiltRequest::Responses(request) = result.request else {
        panic!("expected responses request");
    };
    let json = serde_json::to_string(&request).expect("request serializes");
    assert!(json.contains("current"));
}

#[test]
fn truncation_retains_or_drops_complete_tool_call_batches_atomically_for_both_providers() {
    let history = vec![
        HistoryItem::user("old ordinary turn"),
        HistoryItem::assistant("x".repeat(10_000)),
        HistoryItem::AssistantToolCalls {
            text: None,
            calls: vec![
                HistoryToolCall {
                    call_id: "batch-a".into(),
                    name: "fs__read".into(),
                    arguments_json: r#"{"path":"a"}"#.into(),
                },
                HistoryToolCall {
                    call_id: "batch-b".into(),
                    name: "fs__read".into(),
                    arguments_json: r#"{"path":"b"}"#.into(),
                },
            ],
        },
        HistoryItem::ToolOutput {
            call_id: "batch-a".into(),
            output_json: r#"{"body":"output-a"}"#.into(),
        },
        HistoryItem::ToolOutput {
            call_id: "batch-b".into(),
            output_json: r#"{"body":"output-b"}"#.into(),
        },
        HistoryItem::user("current turn"),
    ];

    for protocol in [ApiProtocol::Responses, ApiProtocol::Completions] {
        let fit = build_test_request(TestRequestBuilderInput {
            protocol,
            model_id: "gpt-test",
            model: metadata_with_effective_input_limit(32_000, 300),
            prelude: &[],
            history: &history,
            protected_start_index: 5,
            tools: &[],
            evidence: &[],
            history_adapter: None,
            context_view: None,
        })
        .expect("batch fits after old history is dropped");
        assert_eq!(fit.budget.retained_history_items, 4);
        let fit_json: serde_json::Value =
            serde_json::from_str(&request_json(fit)).expect("request JSON");
        match protocol {
            ApiProtocol::Responses => {
                let items = fit_json["input"].as_array().expect("responses input");
                assert_eq!(items.len(), 5);
                assert_eq!(items[0]["type"], "function_call");
                assert_eq!(items[0]["call_id"], "batch-a");
                assert_eq!(items[1]["type"], "function_call");
                assert_eq!(items[1]["call_id"], "batch-b");
                assert_eq!(items[2]["type"], "function_call_output");
                assert_eq!(items[2]["call_id"], "batch-a");
                assert_eq!(items[3]["type"], "function_call_output");
                assert_eq!(items[3]["call_id"], "batch-b");
                assert_eq!(items[4]["role"], "user");
                assert_eq!(items[4]["content"][0]["text"], "current turn");
            }
            ApiProtocol::Completions => {
                let messages = fit_json["messages"].as_array().expect("chat messages");
                assert_eq!(messages.len(), 4);
                assert_eq!(messages[0]["role"], "assistant");
                assert_eq!(messages[0]["tool_calls"][0]["id"], "batch-a");
                assert_eq!(messages[0]["tool_calls"][1]["id"], "batch-b");
                assert_eq!(messages[1]["role"], "tool");
                assert_eq!(messages[1]["tool_call_id"], "batch-a");
                assert_eq!(messages[2]["role"], "tool");
                assert_eq!(messages[2]["tool_call_id"], "batch-b");
                assert_eq!(messages[3]["role"], "user");
                assert_eq!(messages[3]["content"], "current turn");
            }
        }
        assert!(
            !request_json(
                build_test_request(TestRequestBuilderInput {
                    protocol,
                    model_id: "gpt-test",
                    model: metadata_with_effective_input_limit(32_000, 150),
                    prelude: &[],
                    history: &history,
                    protected_start_index: 5,
                    tools: &[],
                    evidence: &[],
                    history_adapter: None,
                    context_view: None,
                })
                .expect("current turn fits")
            )
            .contains("batch-a")
        );
    }
}

#[test]
fn tool_schema_size_counts_toward_budget() {
    let long = "x".repeat(6000);
    let history = vec![
        HistoryItem::user("old"),
        HistoryItem::assistant(long),
        HistoryItem::user("current"),
    ];
    let tools = vec![ToolSpec {
        name: "big_tool".to_string(),
        description: "big".to_string(),
        parameters: json!({
            "type": "object",
            "properties": { "payload": { "type": "string", "description": "x".repeat(8000) } },
            "required": ["payload"],
            "additionalProperties": false
        }),
        strict: true,
    }];

    let without_tools = build_test_request(TestRequestBuilderInput {
        protocol: ApiProtocol::Responses,
        model_id: "gpt-test",
        model: metadata(4096),
        prelude: &[],
        history: &history,
        protected_start_index: 2,
        tools: &[],
        evidence: &[],
        history_adapter: None,
        context_view: None,
    })
    .expect("request builds");
    let with_tools = build_test_request(TestRequestBuilderInput {
        protocol: ApiProtocol::Responses,
        model_id: "gpt-test",
        model: metadata(4096),
        prelude: &[],
        history: &history,
        protected_start_index: 2,
        tools: &tools,
        evidence: &[],
        history_adapter: None,
        context_view: None,
    })
    .expect("request builds");

    assert!(with_tools.budget.estimated_tools_tokens > 0);
    assert!(
        with_tools.budget.retained_history_items <= without_tools.budget.retained_history_items
    );
}

#[test]
fn effective_input_limit_bounds_retained_history_budget() {
    let old_context = "x".repeat(6_000);
    let history = vec![
        HistoryItem::user("old question"),
        HistoryItem::assistant(old_context),
        HistoryItem::user("current question"),
    ];

    let uncapped = build_test_request(TestRequestBuilderInput {
        protocol: ApiProtocol::Responses,
        model_id: "gpt-test",
        model: metadata(32_000),
        prelude: &[],
        history: &history,
        protected_start_index: 2,
        tools: &[],
        evidence: &[],
        history_adapter: None,
        context_view: None,
    })
    .expect("uncapped request builds");
    let capped = build_test_request(TestRequestBuilderInput {
        protocol: ApiProtocol::Responses,
        model_id: "gpt-test",
        model: metadata_with_effective_input_limit(32_000, 900),
        prelude: &[],
        history: &history,
        protected_start_index: 2,
        tools: &[],
        evidence: &[],
        history_adapter: None,
        context_view: None,
    })
    .expect("effective-input-limited request builds");

    assert_eq!(capped.budget.input_budget_tokens, 900);
    assert!(capped.budget.truncated);
    assert!(uncapped.budget.retained_history_items > capped.budget.retained_history_items);
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
        protocol: ApiProtocol::Responses,
        model_id: "gpt-test",
        model: metadata_with_effective_input_limit(32_000, 2_000),
        prelude: &[],
        history: &history,
        protected_start_index: 0,
        tools: &tools,
        evidence: &[],
        history_adapter: None,
        context_view: None,
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
fn selected_evidence_is_injected_before_current_user_for_both_protocols() {
    let history = vec![
        HistoryItem::user("old question"),
        HistoryItem::assistant("old answer"),
        HistoryItem::user("What did src/evidence.rs say?"),
    ];
    let evidence = vec![evidence(
        "ev-1",
        "src/evidence.rs defines compact evidence records",
        "src/evidence.rs",
        1,
    )];

    let responses = build_test_request(TestRequestBuilderInput {
        protocol: ApiProtocol::Responses,
        model_id: "gpt-test",
        model: metadata(8192),
        prelude: &[],
        history: &history,
        protected_start_index: 2,
        tools: &[],
        evidence: &evidence,
        history_adapter: None,
        context_view: None,
    })
    .expect("request builds");
    let BuiltRequest::Responses(request) = responses.request else {
        panic!("expected responses request");
    };
    let json = serde_json::to_value(&request).expect("request serializes");
    let input = json["input"].as_array().expect("input array");
    let evidence_index = input
        .iter()
        .position(|item| item["role"] == "developer" && item.to_string().contains("ev-1"))
        .expect("evidence developer item");
    let current_user_index = input
        .iter()
        .rposition(|item| item["role"] == "user")
        .expect("current user item");
    assert!(evidence_index < current_user_index);
    assert_eq!(responses.selected_evidence_ids, vec!["ev-1"]);
    assert_eq!(responses.budget.selected_evidence_items, 1);

    let completions = build_test_request(TestRequestBuilderInput {
        protocol: ApiProtocol::Completions,
        model_id: "chat-test",
        model: metadata(8192),
        prelude: &[],
        history: &history,
        protected_start_index: 2,
        tools: &[],
        evidence: &evidence,
        history_adapter: None,
        context_view: None,
    })
    .expect("request builds");
    let BuiltRequest::Completions(request) = completions.request else {
        panic!("expected completions request");
    };
    let messages = serde_json::to_value(&request.messages).expect("messages serialize");
    let messages = messages.as_array().expect("messages array");
    let evidence_index = messages
        .iter()
        .position(|message| message["role"] == "developer" && message.to_string().contains("ev-1"))
        .expect("evidence developer message");
    let current_user_index = messages
        .iter()
        .rposition(|message| message["role"] == "user")
        .expect("current user message");
    assert!(evidence_index < current_user_index);
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
        protocol: ApiProtocol::Responses,
        model_id: "gpt-test",
        model,
        prelude: &[],
        history: &history,
        protected_start_index: 0,
        tools: &[],
        evidence: &evidence,
        history_adapter: None,
        context_view: None,
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
        protocol: ApiProtocol::Responses,
        model_id: "gpt-test",
        model,
        prelude: &[],
        history: &history,
        protected_start_index: 0,
        tools: &[],
        evidence: &evidence,
        history_adapter: None,
        context_view: None,
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
        protocol: ApiProtocol::Responses,
        model_id: "gpt-test",
        model: metadata(1024),
        prelude: &[],
        history: &history,
        protected_start_index: 2,
        tools: &[],
        evidence: &[],
        history_adapter: None,
        context_view: None,
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
        protocol: ApiProtocol::Responses,
        model_id: "gpt-test",
        model: metadata_with_effective_input_limit(32_000, 300),
        prelude: &[],
        history: &history,
        protected_start_index: 0,
        tools: &[],
        evidence: &[],
        history_adapter: None,
        context_view: None,
    })
    .expect_err("effective-input-limited protected current turn should fail fast");

    let message = err.to_string();
    assert!(message.contains("protected/current context tokens"));
    assert!(message.contains("exceed budget (300)"));
}

#[test]
fn rejects_zero_effective_input_limit_metadata() {
    let err = build_test_request(TestRequestBuilderInput {
        protocol: ApiProtocol::Responses,
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
        history_adapter: None,
        context_view: None,
    })
    .expect_err("zero effective input limit should fail fast");

    assert!(
        err.to_string()
            .contains("model.effective_input_limit_tokens must be greater than 0")
    );
}

#[test]
fn none_context_view_preserves_request_shape() {
    let history = vec![HistoryItem::user("hello"), HistoryItem::assistant("hi")];
    let baseline = build_test_request(TestRequestBuilderInput {
        protocol: ApiProtocol::Responses,
        model_id: "gpt-test",
        model: metadata(8192),
        prelude: &[],
        history: &history,
        protected_start_index: 0,
        tools: &[],
        evidence: &[],
        history_adapter: None,
        context_view: None,
    })
    .expect("request builds");
    let repeat = build_test_request(TestRequestBuilderInput {
        protocol: ApiProtocol::Responses,
        model_id: "gpt-test",
        model: metadata(8192),
        prelude: &[],
        history: &history,
        protected_start_index: 0,
        tools: &[],
        evidence: &[],
        history_adapter: None,
        context_view: None,
    })
    .expect("request rebuilds");

    let baseline_json = request_json(baseline);
    let repeat_json = request_json(repeat);
    assert_eq!(baseline_json, repeat_json);
    assert!(!baseline_json.contains("[Context:"));
}

#[test]
fn context_view_prompt_sections_are_deterministic() {
    let history = vec![
        HistoryItem::assistant("previous"),
        HistoryItem::user("current user"),
    ];
    let context_view = sample_context_view(false);
    let first = request_json(
        build_test_request(TestRequestBuilderInput {
            protocol: ApiProtocol::Responses,
            model_id: "gpt-test",
            model: metadata(8192),
            prelude: &[],
            history: &history,
            protected_start_index: 1,
            tools: &[],
            evidence: &[],
            history_adapter: None,
            context_view: Some(&context_view),
        })
        .expect("request builds"),
    );
    let second = request_json(
        build_test_request(TestRequestBuilderInput {
            protocol: ApiProtocol::Responses,
            model_id: "gpt-test",
            model: metadata(8192),
            prelude: &[],
            history: &history,
            protected_start_index: 1,
            tools: &[],
            evidence: &[],
            history_adapter: None,
            context_view: Some(&context_view),
        })
        .expect("request rebuilds"),
    );
    assert_eq!(first, second);
}

#[test]
fn runtime_session_state_stops_stable_prefix_and_changes_both_provider_requests() {
    fn snapshot(runtime_material: &str) -> RuntimeSnapshot {
        let frame = |kind, source, ordinal, stable_key, item| {
            RuntimeFrame::new(
                kind,
                FrameVisibility::Active,
                RuntimeFrameProvenance::new(source),
                RuntimeFrameIdSeed {
                    frame_kind: kind,
                    source,
                    ordinal,
                    stable_key,
                    source_span: None,
                },
            )
            .with_protocol(item)
        };

        let mut snapshot = RuntimeSnapshot::new("runtime-provenance-cache-test");
        snapshot.push_frame(frame(
            RuntimeFrameKind::Summary,
            RuntimeSource::SessionState,
            1,
            "child-session-runtime-material",
            ProtocolFrameItem::ContextSummary {
                text: format!("[Context: Child Sessions]\n- {runtime_material}"),
            },
        ));
        snapshot.push_frame(frame(
            RuntimeFrameKind::Assistant,
            RuntimeSource::Transcript,
            2,
            "older-ordinary-transcript",
            ProtocolFrameItem::AssistantText {
                text: "older ordinary transcript".into(),
            },
        ));
        snapshot
    }

    let prelude = [PromptMessage::system("stable static prelude")];
    for protocol in [ApiProtocol::Responses, ApiProtocol::Completions] {
        let first = build_request(RequestBuilderInput {
            protocol,
            model_id: "gpt-test",
            model: metadata(8192),
            prelude: &prelude,
            snapshot: &snapshot("child status: running"),
            tools: &[],
        })
        .expect("request builds");
        let first_prefix = first
            .prompt_plan
            .stable_prefix_hash()
            .expect("stable static prelude has fingerprint")
            .to_string();
        assert_eq!(first.prompt_plan.stable_prefix_end, Some(0));
        assert_eq!(
            first.prompt_plan.segments[1].stability,
            prompt_plan::PromptSegmentStability::Volatile
        );
        assert_eq!(
            first.prompt_plan.segments[1].source.provenance.source,
            RuntimeSource::SessionState
        );
        assert_eq!(
            first.prompt_plan.segments[2].stability,
            prompt_plan::PromptSegmentStability::Volatile
        );
        assert!(!first.prompt_plan.segments[2].cache.cache_eligible);
        let runtime_tokens = first.prompt_plan.segments[1]
            .tokens
            .estimated_input_tokens
            .expect("runtime material has token estimate");
        assert!(first.prompt_plan.token_report().volatile_prompt_tokens >= runtime_tokens);
        assert_eq!(
            first
                .prompt_plan
                .token_report()
                .stable_after_boundary_tokens,
            0
        );
        let first_json = request_json(first);

        let changed = build_request(RequestBuilderInput {
            protocol,
            model_id: "gpt-test",
            model: metadata(8192),
            prelude: &prelude,
            snapshot: &snapshot("child status: complete"),
            tools: &[],
        })
        .expect("changed request builds");
        assert_eq!(
            changed.prompt_plan.stable_prefix_hash(),
            Some(first_prefix.as_str())
        );
        let changed_json = request_json(changed);
        assert_ne!(first_json, changed_json);
        assert!(changed_json.contains("child status: complete"));
    }
}

#[test]
fn context_view_adapter_hard_and_pinned_preludes_are_volatile() {
    let context_view = sample_context_view(false);
    let adapter = context_view_history_adapter(&context_view, &[], 0);
    assert_eq!(adapter.prelude.len(), 2);
    assert!(
        adapter.prelude[0]
            .text
            .starts_with("[Context: Hard Context]")
    );
    assert!(
        adapter.prelude[1]
            .text
            .starts_with("[Context: Pinned Context]")
    );
    assert!(
        adapter
            .prelude
            .iter()
            .all(|message| { message.origin == PromptMessageOrigin::RuntimeContextView })
    );

    let plan_for = |prelude: &[PromptMessage]| {
        build_prompt_plan(PromptPlanBuildInput {
            protocol: ApiProtocol::Responses,
            model_id: "gpt-test",
            prelude,
            snapshot: &RuntimeSnapshot::new("context-origin-test"),
            selected_frames: &[],
            protected_suffix_len: 0,
            evidence_message: None,
            selected_evidence_ids: &[],
        })
    };
    let context_first = plan_for(&adapter.prelude);
    assert_eq!(context_first.cacheable_prefix_len(), 0);
    assert_eq!(context_first.stable_prefix_end, None);

    let mut prelude = vec![PromptMessage::system("stable system")];
    prelude.extend(adapter.prelude.clone());
    let plan = plan_for(&prelude);
    assert_eq!(plan.cacheable_prefix_len(), 1);
    assert_eq!(
        plan.segments[1].stability,
        prompt_plan::PromptSegmentStability::Volatile
    );
    assert_eq!(
        plan.segments[2].stability,
        prompt_plan::PromptSegmentStability::Volatile
    );
    let report = plan.token_report();
    let stable_tokens = plan.segments[0].tokens.estimated_input_tokens.unwrap();
    let volatile_tokens = plan.segments[1..]
        .iter()
        .map(|segment| segment.tokens.estimated_input_tokens.unwrap())
        .sum::<u64>();
    assert_eq!(report.total_prompt_tokens, stable_tokens + volatile_tokens);
    assert_eq!(report.stable_prompt_tokens, stable_tokens);
    assert_eq!(report.volatile_prompt_tokens, volatile_tokens);
    assert_eq!(report.cacheable_prefix_tokens, stable_tokens);
    assert_eq!(report.stable_after_boundary_tokens, 0);
    assert_eq!(report.first_volatile_index, Some(1));

    let mut changed_prelude = prelude;
    changed_prelude[1].text.push_str(" changed");
    let changed = plan_for(&changed_prelude);
    assert_eq!(plan.stable_prefix_hash(), changed.stable_prefix_hash());
}

#[test]
fn explicit_history_adapter_matches_context_view_compatibility_path() {
    let history = vec![
        HistoryItem::assistant("previous"),
        HistoryItem::user("current user"),
    ];
    let context_view = sample_context_view(true);
    let adapter = context_view_history_adapter(&context_view, &history, 1);

    let compatibility = request_json(
        build_test_request(TestRequestBuilderInput {
            protocol: ApiProtocol::Responses,
            model_id: "gpt-test",
            model: metadata(8192),
            prelude: &[],
            history: &history,
            protected_start_index: 1,
            tools: &[],
            evidence: &[],
            history_adapter: None,
            context_view: Some(&context_view),
        })
        .expect("compatibility request builds"),
    );
    let explicit = request_json(
        build_test_request(TestRequestBuilderInput {
            protocol: ApiProtocol::Responses,
            model_id: "gpt-test",
            model: metadata(8192),
            prelude: &[],
            history: &history,
            protected_start_index: 1,
            tools: &[],
            evidence: &[],
            history_adapter: Some(&adapter),
            context_view: None,
        })
        .expect("adapter request builds"),
    );

    assert_eq!(explicit, compatibility);
}

#[test]
fn context_view_sections_appear_in_required_order() {
    let history = vec![
        HistoryItem::assistant("previous"),
        HistoryItem::user("current user"),
    ];
    let context_view = sample_context_view(true);
    let sections = assemble_context_view_sections(&context_view, &history, 1);
    let mut combined = sections
        .prelude
        .iter()
        .map(|message| message.text.as_str())
        .collect::<Vec<_>>();
    let summary_texts = adapter_summary_texts(&sections);
    combined.extend(summary_texts.iter().map(String::as_str));
    combined.push("current user");
    let combined = combined.join("\n");
    let mut cursor = 0usize;
    for needle in [
        "[Context: Hard Context]",
        "[Context: Pinned Context]",
        "[Context: Active Tail]",
        "[Context: Index]",
        "[Context: Summaries]",
        "[Context: Folded Outputs]",
        "[Context: Opened Details]",
        "current user",
    ] {
        let next = combined[cursor..].find(needle).expect("section present") + cursor;
        cursor = next + needle.len();
    }
}

#[test]
fn opened_detail_only_changes_suffix_after_stable_context_prefix() {
    let history = vec![
        HistoryItem::assistant("previous"),
        HistoryItem::user("current user"),
    ];
    let closed_json = request_json(
        build_test_request(TestRequestBuilderInput {
            protocol: ApiProtocol::Responses,
            model_id: "gpt-test",
            model: metadata(8192),
            prelude: &[],
            history: &history,
            protected_start_index: 1,
            tools: &[],
            evidence: &[],
            history_adapter: None,
            context_view: Some(&sample_context_view(false)),
        })
        .expect("closed request builds"),
    );
    let open_json = request_json(
        build_test_request(TestRequestBuilderInput {
            protocol: ApiProtocol::Responses,
            model_id: "gpt-test",
            model: metadata(8192),
            prelude: &[],
            history: &history,
            protected_start_index: 1,
            tools: &[],
            evidence: &[],
            history_adapter: None,
            context_view: Some(&sample_context_view(true)),
        })
        .expect("open request builds"),
    );
    let marker = "[Context: Opened Details]";
    let folded_marker = "[Context: Folded Outputs]";
    let stable_end = open_json.find(marker).expect("opened marker present");
    let closed_end = closed_json
        .find(folded_marker)
        .expect("folded marker present")
        + folded_marker.len();
    assert_eq!(&closed_json[..closed_end], &open_json[..closed_end]);
    assert!(open_json[stable_end..].contains(marker));
}

#[test]
fn protected_current_oversize_still_fails_with_context_view_present() {
    let history = vec![
        HistoryItem::user("old"),
        HistoryItem::user("x".repeat(20_000)),
    ];
    let context_view = sample_context_view(true);
    let err = build_test_request(TestRequestBuilderInput {
        protocol: ApiProtocol::Responses,
        model_id: "gpt-test",
        model: metadata(1024),
        prelude: &[],
        history: &history,
        protected_start_index: 1,
        tools: &[],
        evidence: &[],
        history_adapter: None,
        context_view: Some(&context_view),
    })
    .expect_err("protected current turn should still fail");
    assert!(
        err.to_string()
            .contains("protected current context exceeds input budget")
    );
}

#[test]
fn hard_context_includes_full_protected_detail_without_truncation() {
    let long_detail = format!("HARD-CONTEXT-START {} HARD-CONTEXT-END", "x".repeat(600));
    let context_view = project_context_view(&[transcript_record(
        1,
        TranscriptEvent::UserMessage {
            content: UserMessageContent::from(long_detail.clone()),
        },
    )])
    .expect("context view projection");

    let result = build_test_request(TestRequestBuilderInput {
        protocol: ApiProtocol::Responses,
        model_id: "gpt-test",
        model: metadata(8192),
        prelude: &[],
        history: &[HistoryItem::user("current user")],
        protected_start_index: 0,
        tools: &[],
        evidence: &[],
        history_adapter: None,
        context_view: Some(&context_view),
    })
    .expect("request builds");

    let json = request_json(result);
    assert!(json.contains("[Context: Hard Context]"));
    assert!(json.contains(&long_detail));
}

#[test]
fn archived_and_removed_blocks_are_suppressed_from_context_sections() {
    let context_view = project_context_view(&[
        transcript_record(
            1,
            TranscriptEvent::AssistantMessage {
                content: "visible note".into(),
            },
        ),
        transcript_record(
            2,
            TranscriptEvent::AssistantMessage {
                content: "archived note detail".into(),
            },
        ),
        transcript_record(
            3,
            TranscriptEvent::ContextViewOperationMetadata {
                operation: "archive".into(),
                node_id: None,
                block_id: Some("block-seq-2-note".into()),
                detail: None,
            },
        ),
        transcript_record(
            4,
            TranscriptEvent::AssistantMessage {
                content: "removed note detail".into(),
            },
        ),
        transcript_record(
            5,
            TranscriptEvent::ContextViewOperationMetadata {
                operation: "remove_from_view".into(),
                node_id: None,
                block_id: Some("block-seq-4-note".into()),
                detail: None,
            },
        ),
    ])
    .expect("context view projection");

    let sections =
        assemble_context_view_sections(&context_view, &[HistoryItem::user("current")], 0);
    let combined = sections
        .prelude
        .iter()
        .map(|message| message.text.as_str())
        .chain(adapter_summary_texts(&sections).iter().map(String::as_str))
        .collect::<Vec<_>>()
        .join("\n");

    assert!(combined.contains("visible note"));
    assert!(!combined.contains("archived note detail"));
    assert!(!combined.contains("removed note detail"));
}

#[test]
fn resolved_unresolved_errors_are_suppressed_from_context_sections() {
    let context_view = project_context_view(&[
        transcript_record(
            1,
            TranscriptEvent::Error {
                message: "context view projection unavailable".into(),
            },
        ),
        transcript_record(
            2,
            TranscriptEvent::ContextViewOperationMetadata {
                operation: "resolve".into(),
                node_id: None,
                block_id: Some("block-seq-1-error".into()),
                detail: None,
            },
        ),
    ])
    .expect("resolved context view projection");

    let sections =
        assemble_context_view_sections(&context_view, &[HistoryItem::user("current")], 0);
    let combined = sections
        .prelude
        .iter()
        .map(|message| message.text.as_str())
        .chain(adapter_summary_texts(&sections).iter().map(String::as_str))
        .collect::<Vec<_>>()
        .join("\n");

    assert!(!combined.contains("context view projection unavailable"));
    assert!(!combined.contains("unresolved_error"));
}

#[test]
fn reasoning_debug_notes_are_hidden_from_context_index_unless_opened() {
    let context_view = project_context_view(&[
        transcript_record(
            1,
            TranscriptEvent::ReasoningMessage {
                content: "scratch reasoning trace".into(),
            },
        ),
        transcript_record(
            2,
            TranscriptEvent::AssistantMessage {
                content: "durable assistant note".into(),
            },
        ),
    ])
    .expect("context view projection");

    let sections =
        assemble_context_view_sections(&context_view, &[HistoryItem::user("current")], 0);
    let combined = adapter_summary_texts(&sections).join("\n");

    assert!(combined.contains("durable assistant note"));
    assert!(!combined.contains("scratch reasoning trace"));

    let opened_context_view = project_context_view(&[
        transcript_record(
            1,
            TranscriptEvent::ReasoningMessage {
                content: "scratch reasoning trace".into(),
            },
        ),
        transcript_record(
            2,
            TranscriptEvent::ContextViewOperationMetadata {
                operation: "open_detail".into(),
                node_id: None,
                block_id: Some("block-seq-1-reasoning-note".into()),
                detail: None,
            },
        ),
    ])
    .expect("opened context view projection");
    let opened_sections =
        assemble_context_view_sections(&opened_context_view, &[HistoryItem::user("current")], 0);
    let opened_combined = adapter_summary_texts(&opened_sections).join("\n");

    assert!(opened_combined.contains("[Context: Opened Details]"));
    assert!(opened_combined.contains("scratch reasoning trace"));
}

#[test]
fn folded_placeholders_respect_archive_and_remove_visibility() {
    let context_view = project_context_view(&[
        transcript_record(
            1,
            TranscriptEvent::ToolCallStarted {
                call_id: "call-1".into(),
                name: "shell__exec".into(),
                args: json!({"command": "cargo test"}),
            },
        ),
        transcript_record(
            2,
            TranscriptEvent::ToolCallFinished {
                call_id: "call-1".into(),
                name: "shell__exec".into(),
                ok: true,
                output: crate::tool::ToolResult::ok(
                    "shell__exec",
                    json!({
                        "status": 0,
                        "stdout": "a".repeat(5000),
                        "stdout_truncated": false,
                        "stderr": "b".repeat(5000),
                        "stderr_truncated": false
                    }),
                ),
            },
        ),
        transcript_record(
            3,
            TranscriptEvent::ContextViewOperationMetadata {
                operation: "archive".into(),
                node_id: None,
                block_id: Some("block-seq-2-folded-output-folded-output-seq-2-stdout".into()),
                detail: None,
            },
        ),
        transcript_record(
            4,
            TranscriptEvent::ContextViewOperationMetadata {
                operation: "remove_from_view".into(),
                node_id: None,
                block_id: Some("block-seq-2-folded-output-folded-output-seq-2-stderr".into()),
                detail: None,
            },
        ),
    ])
    .expect("context view projection");

    let sections =
        assemble_context_view_sections(&context_view, &[HistoryItem::user("current")], 0);
    let combined = adapter_summary_texts(&sections).join("\n");

    assert!(!combined.contains("folded-output-seq-2-stdout"));
    assert!(!combined.contains("folded-output-seq-2-stderr"));
}

#[test]
fn compacted_projection_excludes_old_raw_context_and_folded_placeholders() {
    let records = vec![
        transcript_record(
            1,
            TranscriptEvent::AssistantMessage {
                content: "old raw note should disappear".into(),
            },
        ),
        transcript_record(
            2,
            TranscriptEvent::ToolCallStarted {
                call_id: "call-1".into(),
                name: "shell__exec".into(),
                args: json!({"command": "cargo test"}),
            },
        ),
        transcript_record(
            3,
            TranscriptEvent::ToolCallFinished {
                call_id: "call-1".into(),
                name: "shell__exec".into(),
                ok: true,
                output: crate::tool::ToolResult::ok(
                    "shell__exec",
                    json!({
                        "status": 0,
                        "stdout": "x".repeat(5000),
                        "stdout_truncated": false,
                        "stderr": "",
                        "stderr_truncated": false
                    }),
                ),
            },
        ),
        transcript_record(
            4,
            TranscriptEvent::ContextCompaction(crate::agent::ContextCompactionEvent {
                outcome: "succeeded".into(),
                summary: "compacted summary survives".into(),
                tail_start_index: 3,
                original_history_items: 3,
                retained_history_items: 1,
                retired_source_spans: vec![crate::agent::ContextCompactionSourceSpan {
                    start_sequence: 1,
                    end_sequence: 3,
                }],
                frame_identity_bindings: Vec::new(),
                detail: None,
            }),
        ),
        transcript_record(
            5,
            TranscriptEvent::UserMessage {
                content: UserMessageContent::from("current tail stays"),
            },
        ),
    ];
    let context_view = project_context_view(&records).expect("context view projection");
    let history = restore_session_history_projection(&records);

    let json = request_json(
        build_test_request(TestRequestBuilderInput {
            protocol: ApiProtocol::Responses,
            model_id: "gpt-test",
            model: metadata(8192),
            prelude: &[],
            history: &history,
            protected_start_index: history.len().saturating_sub(1),
            tools: &[],
            evidence: &[],
            history_adapter: None,
            context_view: Some(&context_view),
        })
        .expect("request builds"),
    );

    assert!(json.contains("compacted summary survives"));
    assert!(json.contains("current tail stays"));
    assert!(!json.contains("old raw note should disappear"));
    assert!(!json.contains("folded-output-seq-3-stdout"));
}

#[test]
fn restored_context_view_prompt_preserves_protected_context_and_hides_soft_deleted_blocks() {
    let large_stdout =
        "stdout-body-".repeat((DEFAULT_FOLDED_OUTPUT_THRESHOLD_BYTES / "stdout-body-".len()) + 32);
    let large_stderr =
        "stderr-body-".repeat((DEFAULT_FOLDED_OUTPUT_THRESHOLD_BYTES / "stderr-body-".len()) + 32);
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

    let result = build_test_request(TestRequestBuilderInput {
        protocol: ApiProtocol::Responses,
        model_id: "gpt-test",
        model: metadata(32_768),
        prelude: &[],
        history: &current_history,
        protected_start_index: 0,
        tools: &[],
        evidence: &snapshot.evidence,
        history_adapter: None,
        context_view: Some(&projection),
    })
    .expect("request builds from restored projection");
    let json = request_json(result);

    assert!(json.contains("[Context: Hard Context]"));
    assert!(
        json.contains("MUST keep raw transcript events append-only; do not purge requirements")
    );
    assert!(json.contains("Permission denied"));
    assert!(json.contains("src/lib.rs"));
    assert!(json.contains("cargo test failed"));
    assert!(json.contains("a270dda"));
    assert!(json.contains("invariant violation"));
    assert!(json.contains("[Context: Folded Outputs]"));
    assert!(json.contains("folded-output-seq-16-stdout"));
    assert!(json.contains("folded-output-seq-16-stderr"));
    assert!(json.contains("tool=shell__exec"));
    assert!(json.contains("stream=stdout"));
    assert!(json.contains("stream=stderr"));
    assert!(json.contains("command=cargo test --quiet"));
    assert!(!json.contains("soft archived note"));
    assert!(!json.contains("soft removed note"));
    assert!(!json.contains(&large_stdout));
    assert!(!json.contains(&large_stderr));
    assert_eq!(records.len(), original_len);
}

#[test]
fn legacy_session_restore_builds_prompt_without_context_metadata() {
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
                content: UserMessageContent::from("legacy user one"),
            },
        ),
        transcript_record(
            3,
            TranscriptEvent::AssistantMessage {
                content: "legacy assistant".into(),
            },
        ),
        transcript_record(
            4,
            TranscriptEvent::UserMessage {
                content: UserMessageContent::from("legacy user two"),
            },
        ),
    ];

    let snapshot = project_session_restore_snapshot("s".into(), records.clone()).expect("snapshot");
    let tree = project_context_tree(&records).expect("legacy tree defaults to root");
    assert_eq!(tree.root_node_id().as_str(), "root");
    assert_eq!(tree.active_node_id().map(|id| id.as_str()), Some("root"));
    let projection = project_restored_context_view(&records).expect("legacy context view");

    let result = build_test_request(TestRequestBuilderInput {
        protocol: ApiProtocol::Responses,
        model_id: "gpt-test",
        model: metadata(8192),
        prelude: &[],
        history: &snapshot.history,
        protected_start_index: snapshot.history.len().saturating_sub(1),
        tools: &[],
        evidence: &snapshot.evidence,
        history_adapter: None,
        context_view: Some(&projection),
    })
    .expect("legacy request builds");
    let json = request_json(result);

    assert!(json.contains("legacy user one"));
    assert!(json.contains("legacy assistant"));
    assert!(json.contains("legacy user two"));
    assert!(!json.contains("context_node_created"));
    assert!(!json.contains("context_branch_created"));
}

#[test]
fn both_providers_render_mixed_skill_source_and_fallback_once_in_legal_order() {
    fn frame(
        kind: RuntimeFrameKind,
        visibility: FrameVisibility,
        ordinal: u32,
        item: ProtocolFrameItem,
    ) -> RuntimeFrame {
        RuntimeFrame::new(
            kind,
            visibility,
            RuntimeFrameProvenance::new(RuntimeSource::Transcript),
            RuntimeFrameIdSeed {
                frame_kind: kind,
                source: RuntimeSource::Transcript,
                ordinal,
                stable_key: "mixed-skill-protocol",
                source_span: None,
            },
        )
        .with_protocol(item)
    }

    let mut snapshot = RuntimeSnapshot::new("main");
    snapshot.push_frame(frame(
        RuntimeFrameKind::Assistant,
        FrameVisibility::Active,
        0,
        ProtocolFrameItem::AssistantText {
            text: "OLD-UNRELATED-HISTORY".repeat(2_000),
        },
    ));
    snapshot.push_frame(frame(
        RuntimeFrameKind::ToolCall,
        FrameVisibility::Retired,
        1,
        ProtocolFrameItem::AssistantToolCalls {
            text: None,
            calls: vec![HistoryToolCall {
                call_id: "retired-skill".into(),
                name: "skill".into(),
                arguments_json: "{}".into(),
            }],
        },
    ));
    snapshot.push_frame(frame(
            RuntimeFrameKind::ToolOutput,
            FrameVisibility::Retired,
            2,
            ProtocolFrameItem::ToolOutput {
                call_id: "retired-skill".into(),
                output_json: r#"{"ok":true,"tool":"skill","data":{"name":"retired","content":"RETIRED-SKILL-BODY"}}"#.into(),
            },
        ));
    snapshot.push_frame(frame(
        RuntimeFrameKind::ToolCall,
        FrameVisibility::Active,
        3,
        ProtocolFrameItem::AssistantToolCalls {
            text: None,
            calls: vec![HistoryToolCall {
                call_id: "active-skill".into(),
                name: "skill".into(),
                arguments_json: "{}".into(),
            }],
        },
    ));
    snapshot.push_frame(frame(
            RuntimeFrameKind::ToolOutput,
            FrameVisibility::Active,
            4,
            ProtocolFrameItem::ToolOutput {
                call_id: "active-skill".into(),
                output_json: r#"{"ok":true,"tool":"skill","data":{"name":"active","content":"ACTIVE-SKILL-BODY"}}"#.into(),
            },
        ));
    snapshot.push_frame(frame(
        RuntimeFrameKind::User,
        FrameVisibility::Active,
        5,
        ProtocolFrameItem::UserMessage {
            content: UserMessageContent::from("continue"),
        },
    ));
    crate::skills::reconcile_loaded_skill_material(&mut snapshot)
        .expect("reconciles persisted skill material");
    let retired_source_id = snapshot
        .prompt_contributors
        .iter()
        .find(|contributor| contributor.contributor_id == "skill-material:retired-skill")
        .expect("retired skill contributor")
        .source_frame_ids[0];
    let source = snapshot
        .frames
        .iter_mut()
        .find(|frame| frame.id == retired_source_id)
        .expect("retired skill source");
    let ProtocolFrameItem::ToolOutput { output_json, .. } = source.protocol.as_mut().unwrap()
    else {
        panic!("retired skill source is a tool output");
    };
    *output_json = r#"{"_compaction":{"pruned":true,"reason":"tool output pruned by compaction.prune","original_chars":9999,"tool":"skill"}}"#.into();
    let source_json: serde_json::Value =
        serde_json::from_str(output_json).expect("structural compaction marker");
    assert_eq!(source_json["_compaction"]["pruned"], true);
    crate::skills::reconcile_loaded_skill_material(&mut snapshot)
        .expect("preserves detached skill material after pruning");

    for protocol in [ApiProtocol::Responses, ApiProtocol::Completions] {
        let result = build_request(RequestBuilderInput {
            protocol,
            model_id: "gpt-test",
            model: metadata_with_effective_input_limit(32_000, 350),
            prelude: &[],
            snapshot: &snapshot,
            tools: &[],
        })
        .expect("request builds");
        let json = request_json(result);

        assert!(json.contains("RETIRED-SKILL-BODY"));
        assert!(json.contains("ACTIVE-SKILL-BODY"));
        assert_eq!(json.matches("RETIRED-SKILL-BODY").count(), 1);
        assert_eq!(json.matches("ACTIVE-SKILL-BODY").count(), 1);
        assert!(!json.contains("tool output pruned by compaction.prune"));
        assert!(!json.contains("OLD-UNRELATED-HISTORY"));
        match protocol {
            ApiProtocol::Responses => {
                let input = serde_json::from_str::<serde_json::Value>(&json)
                    .expect("responses request JSON")["input"]
                    .as_array()
                    .expect("responses input array")
                    .clone();
                assert_eq!(input[0]["role"], "developer");
                assert_eq!(input[1]["type"], "function_call");
                assert_eq!(input[2]["type"], "function_call_output");
                assert_eq!(input[3]["role"], "user");
            }
            ApiProtocol::Completions => {
                let messages = serde_json::from_str::<serde_json::Value>(&json)
                    .expect("chat request JSON")["messages"]
                    .as_array()
                    .expect("chat messages array")
                    .clone();
                assert_eq!(messages[0]["role"], "developer");
                assert_eq!(messages[1]["role"], "assistant");
                assert_eq!(messages[2]["role"], "tool");
                assert_eq!(messages[3]["role"], "user");
            }
        }
    }
}

#[test]
fn group_16_both_provider_requests_share_canonical_surviving_context() {
    let snapshot = crate::context_tools::group_16_runtime_snapshot();
    crate::protocol_frames::validate_history_items_complete(&snapshot.active_history_items(), None)
        .expect("canonical protocol frames remain complete");

    for protocol in [ApiProtocol::Responses, ApiProtocol::Completions] {
        let json = request_json(
            build_request(RequestBuilderInput {
                protocol,
                model_id: "gpt-test",
                model: metadata(8192),
                prelude: &[],
                snapshot: &snapshot,
                tools: &[],
            })
            .expect("canonical request builds"),
        );

        for surviving in [
            "CANONICAL ACTIVE TITLE",
            "CANONICAL ACTIVE CONTENT CURRENT-TAIL-SENTINEL",
            "PINNED ACTIVE TITLE",
            "ACTIVE-FOLDED-SENTINEL",
            "SURVIVING-PROTOCOL-SENTINEL",
        ] {
            assert!(json.contains(surviving), "{protocol:?}: {json}");
        }
        for retired in [
            "RETIRED-RAW-SENTINEL",
            "RETIRED-FOLDED-SENTINEL",
            "COMPACTED FOLDED TITLE",
        ] {
            assert!(!json.contains(retired), "{protocol:?}: {json}");
        }
        let request: serde_json::Value = serde_json::from_str(&json).expect("request JSON");
        match protocol {
            ApiProtocol::Responses => {
                assert!(request["input"].as_array().is_some_and(|items| {
                    items.iter().any(|item| {
                        item["type"] == "function_call_output" && item["call_id"] == "current-call"
                    })
                }));
            }
            ApiProtocol::Completions => {
                assert!(request["messages"].as_array().is_some_and(|items| {
                    items.iter().any(|item| {
                        item["role"] == "tool" && item["tool_call_id"] == "current-call"
                    })
                }));
            }
        }
    }
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
        ProtocolFrameItem::AssistantToolCalls {
            text: None,
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
        ProtocolFrameItem::AssistantText {
            text: "RETIRED-PLANNER-FRAME".into(),
        },
    );
    retired.visibility = FrameVisibility::Retired;
    snapshot.push_frame(retired);
    let before = snapshot.clone();
    let input = PromptPlannerInput {
        protocol: ApiProtocol::Responses,
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

fn protected_foldable_snapshot(raw_len: usize, with_reference: bool) -> RuntimeSnapshot {
    let frame = |kind, ordinal, item| {
        let span = crate::runtime_context::SourceSpan::new(10, 10).expect("valid span");
        RuntimeFrame::new(
            kind,
            FrameVisibility::Active,
            RuntimeFrameProvenance::new(RuntimeSource::Transcript).with_span(span),
            RuntimeFrameIdSeed {
                frame_kind: kind,
                source: RuntimeSource::Transcript,
                ordinal,
                stable_key: "protected-foldable-output",
                source_span: Some(span),
            },
        )
        .with_protocol(item)
    };
    let tool_call = frame(
        RuntimeFrameKind::ToolCall,
        0,
        ProtocolFrameItem::AssistantToolCalls {
            text: None,
            calls: vec![HistoryToolCall {
                call_id: "protected-call".into(),
                name: "shell__exec".into(),
                arguments_json: "{}".into(),
            }],
        },
    );
    let tool_output = frame(
            RuntimeFrameKind::ToolOutput,
            1,
            ProtocolFrameItem::ToolOutput {
                call_id: "protected-call".into(),
                output_json: serde_json::to_string(&ToolResult::ok(
                    "shell__exec",
                    json!({"status": 0, "stdout": format!("PROTECTED-RAW-SENTINEL-{}", "x".repeat(raw_len))}),
                ))
                .expect("ToolResult serializes"),
            },
        );
    let user = frame(
        RuntimeFrameKind::User,
        2,
        ProtocolFrameItem::UserMessage {
            content: UserMessageContent::from("continue after protected output"),
        },
    );
    let raw_output = match &tool_output.protocol {
        Some(ProtocolFrameItem::ToolOutput { output_json, .. }) => output_json.clone(),
        _ => unreachable!("tool output frame"),
    };
    let mut snapshot = RuntimeSnapshot::new("protected-foldable-output");
    snapshot.set_protected_frame_ids(vec![tool_call.id, tool_output.id, user.id]);
    snapshot.push_frame(tool_call);
    snapshot.push_frame(tool_output);
    snapshot.push_frame(user);
    if with_reference {
        let output_id = "folded-output-seq-10-tool-result";
        snapshot.context_view.folded_outputs.insert(
            output_id.into(),
            FoldedOutputMetadata {
                output_id: output_id.into(),
                node_id: None,
                output_kind: "tool_result".into(),
                call_id: Some("protected-call".into()),
                tool_name: Some("shell__exec".into()),
                stream: Some("tool_result".into()),
                byte_count: raw_output.len(),
                content: raw_output,
                line_count: 1,
                truncated: false,
                shell_command: None,
                source_start_sequence: Some(10),
                source_end_sequence: Some(10),
                available_sequence: Some(10),
                tool_ok: Some(true),
                exit_status: None,
                provider_metadata: None,
                provider_fold_eligible: true,
            },
        );
        let block_id = ContextBlockId::new("protected-output-block").expect("block id");
        snapshot.context_view.blocks.insert(
            block_id.clone(),
            ContextBlock {
                block_id,
                node_id: None,
                kind: ContextBlockKind::ToolOutput,
                title: "protected output".into(),
                detail: String::new(),
                source: ContextBlockSource::FoldedOutput {
                    output_id: output_id.into(),
                },
                source_start_sequence: Some(10),
                available_sequence: Some(10),
                protected_reasons: Vec::new(),
                folded_output_id: Some(output_id.into()),
            },
        );
    }
    snapshot
}

fn archived_opened_protected_foldable_snapshot(raw_len: usize) -> RuntimeSnapshot {
    let mut snapshot = protected_foldable_snapshot(raw_len, true);
    let block_id = ContextBlockId::new("protected-output-block").expect("block id");
    snapshot.context_view.view_state = crate::context_view::ContextViewState::replay(
        &snapshot.context_view.blocks,
        &[
            crate::context_view::ContextViewOperation::Archive {
                block_id: block_id.clone(),
            },
            crate::context_view::ContextViewOperation::OpenDetail { block_id },
        ],
    )
    .expect("archived block can be opened");
    snapshot
}

fn append_addressable_protected_tool_groups(
    snapshot: &mut RuntimeSnapshot,
    first_group: usize,
    count: usize,
    raw_len: usize,
) {
    for group in first_group..first_group + count {
        let sequence = (group as u64 + 1) * 10;
        let span =
            crate::runtime_context::SourceSpan::new(sequence, sequence).expect("valid source span");
        let call_id = format!("protected-call-{group:03}");
        let output_id = format!("folded-output-seq-{sequence}-tool-result");
        let block_id =
            ContextBlockId::new(format!("protected-block-{group:03}")).expect("block id");
        let stable_key = format!("protected-addressable-group-{group:03}");
        let call = RuntimeFrame::new(
            RuntimeFrameKind::ToolCall,
            FrameVisibility::Active,
            RuntimeFrameProvenance::new(RuntimeSource::Transcript).with_span(span),
            RuntimeFrameIdSeed {
                frame_kind: RuntimeFrameKind::ToolCall,
                source: RuntimeSource::Transcript,
                ordinal: (group * 2) as u32,
                stable_key: &stable_key,
                source_span: Some(span),
            },
        )
        .with_protocol(ProtocolFrameItem::AssistantToolCalls {
            text: None,
            calls: vec![HistoryToolCall {
                call_id: call_id.clone(),
                name: "shell__exec".into(),
                arguments_json: "{}".into(),
            }],
        });
        snapshot.compaction.protected_frame_ids.push(call.id);
        snapshot.push_frame(call);
        let output_json = serde_json::to_string(&ToolResult::ok(
                "shell__exec",
                json!({"status": 0, "stdout": format!("PROTECTED-GROUP-{group:03}-{}", "x".repeat(raw_len))}),
            ))
            .expect("ToolResult serializes");
        let output = RuntimeFrame::new(
            RuntimeFrameKind::ToolOutput,
            FrameVisibility::Active,
            RuntimeFrameProvenance::new(RuntimeSource::Transcript).with_span(span),
            RuntimeFrameIdSeed {
                frame_kind: RuntimeFrameKind::ToolOutput,
                source: RuntimeSource::Transcript,
                ordinal: (group * 2 + 1) as u32,
                stable_key: &stable_key,
                source_span: Some(span),
            },
        )
        .with_protocol(ProtocolFrameItem::ToolOutput {
            call_id: call_id.clone(),
            output_json: output_json.clone(),
        });
        snapshot.compaction.protected_frame_ids.push(output.id);
        snapshot.push_frame(output);
        snapshot.context_view.folded_outputs.insert(
            output_id.clone(),
            FoldedOutputMetadata {
                output_id: output_id.clone(),
                node_id: None,
                output_kind: "tool_result".into(),
                call_id: Some(call_id),
                tool_name: Some("shell__exec".into()),
                stream: Some("tool_result".into()),
                byte_count: output_json.len(),
                content: output_json,
                line_count: 1,
                truncated: false,
                shell_command: None,
                source_start_sequence: Some(sequence),
                source_end_sequence: Some(sequence),
                available_sequence: Some(sequence),
                tool_ok: Some(true),
                exit_status: None,
                provider_metadata: None,
                provider_fold_eligible: true,
            },
        );
        snapshot.context_view.blocks.insert(
            block_id.clone(),
            ContextBlock {
                block_id,
                node_id: None,
                kind: ContextBlockKind::ToolOutput,
                title: format!("protected output {group}"),
                detail: String::new(),
                source: ContextBlockSource::FoldedOutput { output_id },
                source_start_sequence: Some(sequence),
                available_sequence: Some(sequence),
                protected_reasons: Vec::new(),
                folded_output_id: Some(format!("folded-output-seq-{sequence}-tool-result")),
            },
        );
    }
    let user = RuntimeFrame::new(
        RuntimeFrameKind::User,
        FrameVisibility::Active,
        RuntimeFrameProvenance::new(RuntimeSource::Transcript),
        RuntimeFrameIdSeed {
            frame_kind: RuntimeFrameKind::User,
            source: RuntimeSource::Transcript,
            ordinal: (first_group + count) as u32 * 2,
            stable_key: "protected-addressable-current-request",
            source_span: None,
        },
    )
    .with_protocol(ProtocolFrameItem::UserMessage {
        content: UserMessageContent::from("continue after protected outputs"),
    });
    snapshot.compaction.protected_frame_ids.push(user.id);
    snapshot.push_frame(user);
    snapshot.set_protected_frame_ids(snapshot.compaction.protected_frame_ids.clone());
}

fn addressable_protected_tool_groups_snapshot(count: usize, raw_len: usize) -> RuntimeSnapshot {
    let mut snapshot = RuntimeSnapshot::new("protected-addressable-groups");
    append_addressable_protected_tool_groups(&mut snapshot, 0, count, raw_len);
    snapshot
}

fn folded_responses_call_ids(request: &Value) -> Vec<String> {
    request["input"]
        .as_array()
        .expect("responses input")
        .iter()
        .filter(|item| item["type"] == "function_call_output")
        .filter(|item| {
            item["output"]
                .as_str()
                .is_some_and(|output| output.contains("folded-output-seq-"))
        })
        .map(|item| {
            item["call_id"]
                .as_str()
                .expect("folded output call id")
                .to_owned()
        })
        .collect()
}

fn assert_responses_tool_pairs_are_complete_and_ordered(request: &Value, expected: &[String]) {
    let input = request["input"].as_array().expect("responses input");
    let calls = input
        .iter()
        .filter(|item| item["type"] == "function_call")
        .map(|item| {
            item["call_id"]
                .as_str()
                .expect("function call id")
                .to_owned()
        })
        .collect::<Vec<_>>();
    let outputs = input
        .iter()
        .filter(|item| item["type"] == "function_call_output")
        .map(|item| {
            item["call_id"]
                .as_str()
                .expect("function output id")
                .to_owned()
        })
        .collect::<Vec<_>>();
    assert_eq!(calls, expected, "function calls retain source order");
    assert_eq!(outputs, expected, "function outputs retain source order");
}

#[test]
fn provider_folding_is_append_monotonic_for_addressable_protected_outputs() {
    let initial = addressable_protected_tool_groups_snapshot(3, 5_000);
    let policy = ProtectedContextPolicy::from_configured_reserve(Some(1_500), 3_000);
    let first = build_request_with_policy(
        RequestBuilderInput {
            protocol: ApiProtocol::Responses,
            model_id: "gpt-test",
            model: metadata_with_effective_input_limit(8_192, 3_000),
            prelude: &[],
            snapshot: &initial,
            tools: &[],
        },
        None,
        Some(policy),
    )
    .expect("initial protected outputs fit after folding");
    let first_request: Value = serde_json::from_str(&request_json(first)).expect("request JSON");
    let initial_folded = folded_responses_call_ids(&first_request);
    let initial_call_ids = (0..3)
        .map(|group| format!("protected-call-{group:03}"))
        .collect::<Vec<_>>();
    assert_eq!(initial_folded, initial_call_ids);
    assert_responses_tool_pairs_are_complete_and_ordered(&first_request, &initial_call_ids);

    let mut appended = initial.clone();
    append_addressable_protected_tool_groups(&mut appended, 3, 1, 12_000);
    let second = build_request_with_policy(
        RequestBuilderInput {
            protocol: ApiProtocol::Responses,
            model_id: "gpt-test",
            model: metadata_with_effective_input_limit(8_192, 3_000),
            prelude: &[],
            snapshot: &appended,
            tools: &[],
        },
        None,
        Some(policy),
    )
    .expect("appended protected output fits after folding");
    let second_request: Value = serde_json::from_str(&request_json(second)).expect("request JSON");
    let second_call_ids = (0..4)
        .map(|group| format!("protected-call-{group:03}"))
        .collect::<Vec<_>>();
    let second_folded = folded_responses_call_ids(&second_request);
    assert!(
        initial_folded
            .iter()
            .all(|call_id| second_folded.contains(call_id)),
        "every call folded before the append remains folded"
    );
    assert_eq!(
        second_folded, second_call_ids,
        "folding remains a source-order prefix"
    );
    assert_responses_tool_pairs_are_complete_and_ordered(&second_request, &second_call_ids);
}

#[test]
fn provider_folding_caps_over_one_hundred_protected_groups_in_source_order() {
    let snapshot = addressable_protected_tool_groups_snapshot(101, 5_000);
    let model = metadata_with_effective_input_limit(65_536, 50_000);
    let result = build_request_with_policy(
        RequestBuilderInput {
            protocol: ApiProtocol::Responses,
            model_id: "gpt-test",
            model,
            prelude: &[],
            snapshot: &snapshot,
            tools: &[],
        },
        None,
        Some(ProtectedContextPolicy::from_configured_reserve(
            Some(4_000),
            50_000,
        )),
    )
    .expect("canonical ToolResults retain every complete protected group");
    let request: Value = serde_json::from_str(&request_json(result)).expect("request JSON");
    let expected = (0..101)
        .map(|group| format!("protected-call-{group:03}"))
        .collect::<Vec<_>>();
    assert_responses_tool_pairs_are_complete_and_ordered(&request, &expected);
}

#[test]
fn protected_output_folding_builds_valid_provider_pairs_without_mutating_snapshot() {
    let snapshot = protected_foldable_snapshot(5_000, true);
    let before = snapshot.clone();
    for protocol in [ApiProtocol::Responses, ApiProtocol::Completions] {
        let model = metadata_with_effective_input_limit(8_192, 1_500);
        let result = build_request(RequestBuilderInput {
            protocol,
            model_id: "gpt-test",
            model,
            prelude: &[],
            snapshot: &snapshot,
            tools: &[],
        })
        .expect("folded protected output fits request budget");
        assert!(result.budget.estimated_request_tokens <= result.budget.input_budget_tokens);
        assert_eq!(result.budget.provider_folded_output_count, 1);
        let request: Value = serde_json::from_str(&request_json(result)).expect("request JSON");
        assert!(!request.to_string().contains("PROTECTED-RAW-SENTINEL"));
        match protocol {
            ApiProtocol::Responses => {
                let input = request["input"].as_array().expect("responses input");
                let call = input
                    .iter()
                    .find(|item| item["type"] == "function_call")
                    .expect("assistant function call");
                let output = input
                    .iter()
                    .find(|item| item["type"] == "function_call_output")
                    .expect("function output");
                assert_eq!(call["call_id"], "protected-call");
                assert_eq!(output["call_id"], "protected-call");
                let placeholder: Value =
                    serde_json::from_str(output["output"].as_str().expect("function output JSON"))
                        .expect("folded placeholder JSON");
                assert_eq!(
                    placeholder["folded_outputs"][0]["ref_id"],
                    "folded-output-seq-10-tool-result"
                );
            }
            ApiProtocol::Completions => {
                let messages = request["messages"].as_array().expect("chat messages");
                let call = messages
                    .iter()
                    .find(|item| item["role"] == "assistant" && item["tool_calls"].is_array())
                    .expect("assistant tool call");
                let output = messages
                    .iter()
                    .find(|item| item["role"] == "tool")
                    .expect("tool output");
                assert_eq!(call["tool_calls"][0]["id"], "protected-call");
                assert_eq!(output["tool_call_id"], "protected-call");
                let placeholder: Value =
                    serde_json::from_str(output["content"].as_str().expect("tool output JSON"))
                        .expect("folded placeholder JSON");
                assert_eq!(
                    placeholder["folded_outputs"][0]["ref_id"],
                    "folded-output-seq-10-tool-result"
                );
            }
        }
    }
    assert_eq!(snapshot, before);
}

#[test]
fn protected_context_over_inline_limit_fails_without_canonical_aggregate() {
    let snapshot = protected_foldable_snapshot(3_500, false);
    let before = snapshot.clone();
    for protocol in [ApiProtocol::Responses, ApiProtocol::Completions] {
        let model = metadata_with_effective_input_limit(8_192, 500);
        let error = build_request(RequestBuilderInput {
            protocol,
            model_id: "gpt-test",
            model,
            prelude: &[],
            snapshot: &snapshot,
            tools: &[],
        })
        .expect_err("unaddressable protected context must fail fast");
        assert!(
            error
                .to_string()
                .starts_with("protected current context exceeds input budget:")
        );
    }
    assert_eq!(snapshot, before);
}

#[test]
fn protected_policy_is_reactive_only_at_zero_and_caps_dynamic_reserve() {
    assert_eq!(
        ProtectedContextPolicy::from_configured_reserve(None, 10_000).reserve_tokens,
        2_000
    );
    assert_eq!(
        ProtectedContextPolicy::from_configured_reserve(None, 1_000_000).reserve_tokens,
        65_536
    );
    assert_eq!(
        ProtectedContextPolicy::from_configured_reserve(Some(0), 10_000).reserve_tokens,
        0
    );
    assert_eq!(
        ProtectedContextPolicy::from_configured_reserve(Some(1_001), 1_000).reserve_tokens,
        1_000
    );
}

#[test]
fn below_watermark_projection_is_byte_identical_to_reactive_only_policy() {
    let snapshot = protected_foldable_snapshot(300, true);
    let mut model = metadata_with_effective_input_limit(8_192, 2_000);
    let input = RequestBuilderInput {
        protocol: ApiProtocol::Responses,
        model_id: "gpt-test",
        model,
        prelude: &[],
        snapshot: &snapshot,
        tools: &[],
    };
    let reactive = build_request_with_policy(
        input.clone(),
        None,
        Some(ProtectedContextPolicy::from_configured_reserve(
            Some(0),
            2_000,
        )),
    )
    .expect("reactive request builds");
    let proactive = build_request_with_policy(
        input,
        None,
        Some(ProtectedContextPolicy::from_configured_reserve(None, 2_000)),
    )
    .expect("below-watermark request builds");

    assert_eq!(proactive.budget.provider_folded_output_count, 0);
    assert_eq!(
        proactive.budget.estimated_provider_folded_protected_tokens,
        0
    );
    assert_eq!(request_json(reactive), request_json(proactive));
}

#[test]
fn unaddressable_soft_band_payload_stays_raw_with_pressure_telemetry() {
    let snapshot = protected_foldable_snapshot(2_200, false);
    let result = build_request_with_policy(
        RequestBuilderInput {
            protocol: ApiProtocol::Responses,
            model_id: "gpt-test",
            model: metadata_with_effective_input_limit(8_192, 1_000),
            prelude: &[],
            snapshot: &snapshot,
            tools: &[],
        },
        None,
        Some(ProtectedContextPolicy::from_configured_reserve(
            Some(600),
            1_000,
        )),
    )
    .expect("soft-band unaddressable output remains raw");

    assert_eq!(result.budget.provider_folded_output_count, 0);
    assert_eq!(result.budget.estimated_foldable_protected_tokens, 0);
    assert_eq!(
        result.budget.estimated_unaddressable_protected_tokens,
        result.budget.estimated_protected_tokens
    );
    assert!(request_json(result).contains("PROTECTED-RAW-SENTINEL"));
}

#[test]
fn archived_opened_first_exposure_is_pressure_invariant() {
    let snapshot = archived_opened_protected_foldable_snapshot(5_000);
    let before = snapshot.clone();
    let model = metadata_with_effective_input_limit(8_192, 1_000);
    let result = build_request_with_policy(
        RequestBuilderInput {
            protocol: ApiProtocol::Responses,
            model_id: "gpt-test",
            model,
            prelude: &[],
            snapshot: &snapshot,
            tools: &[],
        },
        None,
        Some(ProtectedContextPolicy::from_configured_reserve(
            Some(950),
            1_000,
        )),
    )
    .expect("canonical aggregate folds on first exposure");
    assert_eq!(result.budget.provider_folded_output_count, 1);
    assert!(request_json(result).contains("folded-output-seq-10-tool-result"));
    assert_eq!(snapshot, before);
}

#[test]
fn protected_output_folding_leaves_unaddressable_overflow_raw_and_fails() {
    let snapshot = protected_foldable_snapshot(3_500, false);
    let before = snapshot.clone();
    let error = build_request(RequestBuilderInput {
        protocol: ApiProtocol::Responses,
        model_id: "gpt-test",
        model: metadata_with_effective_input_limit(8_192, 500),
        prelude: &[],
        snapshot: &snapshot,
        tools: &[],
    })
    .expect_err("unaddressable protected output cannot be folded");
    assert!(
        error
            .to_string()
            .starts_with("protected current context exceeds input budget:")
    );
    assert_eq!(snapshot, before);
}

#[test]
fn canonical_frozen_evidence_keeps_raw_output_when_admission_fits() {
    let snapshot = protected_foldable_snapshot(1_300, true);
    let before = snapshot.clone();
    let frozen = FrozenEvidence {
        message: Some(format!("FROZEN-EVIDENCE-SENTINEL {}", "e".repeat(50))),
        selected_ids: vec!["frozen-1".into()],
    };
    let mut model = metadata_with_effective_input_limit(8_192, 800);
    let mut raw_snapshot = protected_foldable_snapshot(1_300, true);
    for metadata in raw_snapshot.context_view.folded_outputs.values_mut() {
        metadata.provider_fold_eligible = false;
    }
    let raw = PromptPlanner::plan(PromptPlannerInput {
        protocol: ApiProtocol::Responses,
        model: model.clone(),
        model_id: "gpt-test",
        prelude: &[],
        snapshot: &raw_snapshot,
        tools: &[],
        frozen_evidence: Some(&frozen),
        protected_context_policy: ProtectedContextPolicy::from_configured_reserve(Some(0), 800),
    })
    .expect("canonical projection retains the raw output when it fits");
    assert!(raw.prompt_plan.segments.iter().any(|segment| {
        matches!(
            &segment.content,
            PromptSegmentContent::ToolOutput { output_json, .. }
                if output_json.contains("PROTECTED-RAW-SENTINEL")
        )
    }));
    let planned = PromptPlanner::plan(PromptPlannerInput {
        protocol: ApiProtocol::Responses,
        model,
        model_id: "gpt-test",
        prelude: &[],
        snapshot: &snapshot,
        tools: &[],
        frozen_evidence: Some(&frozen),
        protected_context_policy: ProtectedContextPolicy::from_configured_reserve(Some(0), 800),
    })
    .expect("canonical budget admission succeeds");

    assert_eq!(planned.selected_evidence_ids, frozen.selected_ids);
    assert_eq!(planned.selected_evidence_message, frozen.message);
    assert_eq!(
        planned
            .prompt_plan
            .segments
            .iter()
            .filter(|segment| segment.text.contains("FROZEN-EVIDENCE-SENTINEL"))
            .count(),
        1,
        "frozen evidence appears exactly once"
    );
    assert!(planned.prompt_plan.segments.iter().any(|segment| {
        matches!(
            &segment.content,
            PromptSegmentContent::ToolOutput { output_json, .. }
                if output_json.contains("PROTECTED-RAW-SENTINEL")
        )
    }));
    assert_eq!(snapshot, before);

    let built = build_request_from_selected_prompt(SelectedPromptRequestInput {
        protocol: ApiProtocol::Responses,
        model_id: "gpt-test",
        model: metadata_with_effective_input_limit(8_192, 800),
        tools: &[],
        prompt_plan: planned.prompt_plan,
        budget: planned.budget,
        selected_evidence_ids: planned.selected_evidence_ids,
        selected_evidence_message: planned.selected_evidence_message,
    })
    .expect("folded canonical prompt fits the strict input budget");
    assert!(built.budget.estimated_request_tokens <= built.budget.input_budget_tokens);
}

#[test]
fn build_request_matches_direct_planner_then_serializer_for_both_protocols() {
    let mut snapshot = RuntimeSnapshot::new("planner-serializer-equivalence");
    snapshot.push_frame(
        RuntimeFrame::new(
            RuntimeFrameKind::User,
            FrameVisibility::Active,
            RuntimeFrameProvenance::new(RuntimeSource::Transcript),
            RuntimeFrameIdSeed {
                frame_kind: RuntimeFrameKind::User,
                source: RuntimeSource::Transcript,
                ordinal: 0,
                stable_key: "planner-serializer-equivalence",
                source_span: None,
            },
        )
        .with_protocol(ProtocolFrameItem::UserMessage {
            content: UserMessageContent::from("hello"),
        }),
    );
    let prelude = [PromptMessage::system("system")];
    for protocol in [ApiProtocol::Responses, ApiProtocol::Completions] {
        let model = metadata(8192);
        let planned = PromptPlanner::plan(PromptPlannerInput {
            protocol,
            model: model.clone(),
            model_id: "gpt-test",
            prelude: &prelude,
            snapshot: &snapshot,
            tools: &[],
            frozen_evidence: None,
            protected_context_policy: ProtectedContextPolicy::from_configured_reserve(None, 0),
        })
        .expect("planner succeeds");
        let direct = build_request_from_selected_prompt(SelectedPromptRequestInput {
            protocol,
            model_id: "gpt-test",
            model: model.clone(),
            tools: &[],
            prompt_plan: planned.prompt_plan,
            budget: planned.budget,
            selected_evidence_ids: planned.selected_evidence_ids,
            selected_evidence_message: planned.selected_evidence_message,
        })
        .expect("serializer succeeds");
        let built = build_request(RequestBuilderInput {
            protocol,
            model_id: "gpt-test",
            model,
            prelude: &prelude,
            snapshot: &snapshot,
            tools: &[],
        })
        .expect("builder succeeds");
        assert_eq!(request_json(direct), request_json(built));
    }
}

#[test]
fn frozen_empty_evidence_bypasses_new_snapshot_evidence() {
    let mut snapshot = RuntimeSnapshot::new("frozen-evidence");
    snapshot.push_frame(
        RuntimeFrame::new(
            RuntimeFrameKind::User,
            FrameVisibility::Active,
            RuntimeFrameProvenance::new(RuntimeSource::Transcript),
            RuntimeFrameIdSeed {
                frame_kind: RuntimeFrameKind::User,
                source: RuntimeSource::Transcript,
                ordinal: 0,
                stable_key: "frozen-evidence-user",
                source_span: None,
            },
        )
        .with_protocol(ProtocolFrameItem::UserMessage {
            content: UserMessageContent::from("current request"),
        }),
    );
    snapshot.set_evidence(vec![evidence("new-evidence", "new", "src/new.rs", 1)]);
    let frozen = FrozenEvidence {
        message: None,
        selected_ids: vec![],
    };

    let planned = PromptPlanner::plan(PromptPlannerInput {
        protocol: ApiProtocol::Responses,
        model: metadata(8192),
        model_id: "gpt-test",
        prelude: &[],
        snapshot: &snapshot,
        tools: &[],
        frozen_evidence: Some(&frozen),
        protected_context_policy: ProtectedContextPolicy::from_configured_reserve(None, 0),
    })
    .expect("frozen evidence fits");

    assert!(planned.selected_evidence_ids.is_empty());
    assert_eq!(planned.selected_evidence_message, None);
    assert_eq!(planned.budget.selected_evidence_items, 0);
    assert_eq!(planned.budget.dropped_evidence_items, 1);
    assert!(
        planned
            .prompt_plan
            .segments
            .iter()
            .all(|segment| { segment.source.source_label.as_deref() != Some("evidence_message") })
    );
}

#[test]
fn frozen_evidence_survives_old_history_retention_and_counts_only_current_ids() {
    let mut snapshot = RuntimeSnapshot::new("frozen-evidence-history");
    for (ordinal, item) in [
        ProtocolFrameItem::UserMessage {
            content: UserMessageContent::from("old request"),
        },
        ProtocolFrameItem::AssistantText {
            text: "old context ".repeat(10_000),
        },
        ProtocolFrameItem::UserMessage {
            content: UserMessageContent::from("current request"),
        },
    ]
    .into_iter()
    .enumerate()
    {
        snapshot.push_frame(
            RuntimeFrame::new(
                match &item {
                    ProtocolFrameItem::UserMessage { .. } => RuntimeFrameKind::User,
                    ProtocolFrameItem::AssistantText { .. } => RuntimeFrameKind::Assistant,
                    _ => unreachable!(),
                },
                FrameVisibility::Active,
                RuntimeFrameProvenance::new(RuntimeSource::Transcript),
                RuntimeFrameIdSeed {
                    frame_kind: match &item {
                        ProtocolFrameItem::UserMessage { .. } => RuntimeFrameKind::User,
                        ProtocolFrameItem::AssistantText { .. } => RuntimeFrameKind::Assistant,
                        _ => unreachable!(),
                    },
                    source: RuntimeSource::Transcript,
                    ordinal: ordinal as u32,
                    stable_key: "frozen-evidence-history",
                    source_span: None,
                },
            )
            .with_protocol(item),
        );
    }
    snapshot
        .compaction
        .protected_frame_ids
        .push(snapshot.frames.last().expect("current user frame").id);
    snapshot.set_evidence(vec![
        evidence("ev-1", "first", "src/one.rs", 1),
        evidence("ev-2", "second", "src/two.rs", 2),
    ]);
    let frozen = FrozenEvidence {
        message: Some("frozen evidence message".into()),
        selected_ids: vec!["ev-2".into(), "missing".into(), "ev-1".into()],
    };

    let planned = PromptPlanner::plan(PromptPlannerInput {
        protocol: ApiProtocol::Responses,
        model: metadata(1024),
        model_id: "gpt-test",
        prelude: &[],
        snapshot: &snapshot,
        tools: &[],
        frozen_evidence: Some(&frozen),
        protected_context_policy: ProtectedContextPolicy::from_configured_reserve(None, 0),
    })
    .expect("frozen evidence fits while old history is dropped");

    assert_eq!(planned.selected_evidence_message, frozen.message);
    assert_eq!(planned.selected_evidence_ids, frozen.selected_ids);
    assert!(planned.budget.dropped_history_items > 0);
    assert_eq!(planned.budget.selected_evidence_items, 2);
    assert_eq!(planned.budget.dropped_evidence_items, 0);
}

#[test]
fn impossible_frozen_evidence_returns_budget_error_without_reselection() {
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
    let mut snapshot = RuntimeSnapshot::new("impossible-frozen-evidence");
    snapshot.push_frame(
        RuntimeFrame::new(
            RuntimeFrameKind::User,
            FrameVisibility::Active,
            RuntimeFrameProvenance::new(RuntimeSource::Transcript),
            RuntimeFrameIdSeed {
                frame_kind: RuntimeFrameKind::User,
                source: RuntimeSource::Transcript,
                ordinal: 0,
                stable_key: "impossible-frozen-evidence",
                source_span: None,
            },
        )
        .with_protocol(ProtocolFrameItem::UserMessage {
            content: UserMessageContent::from(exact_fit),
        }),
    );
    snapshot
        .compaction
        .protected_frame_ids
        .push(snapshot.frames.last().expect("current user frame").id);
    let before = snapshot.clone();
    let frozen = FrozenEvidence {
        message: Some("must remain frozen".into()),
        selected_ids: vec!["ev-1".into()],
    };

    let error = PromptPlanner::plan(PromptPlannerInput {
        protocol: ApiProtocol::Responses,
        model,
        model_id: "gpt-test",
        prelude: &[],
        snapshot: &snapshot,
        tools: &[],
        frozen_evidence: Some(&frozen),
        protected_context_policy: ProtectedContextPolicy::from_configured_reserve(None, 0),
    })
    .expect_err("frozen evidence cannot be dropped to make an exact-fit turn fit");

    assert!(
        error
            .to_string()
            .contains("protected current context exceeds input budget")
    );
    assert_eq!(snapshot, before);
}

#[test]
fn child_session_metadata_does_not_change_provider_prompt_or_subagent_evidence_context() {
    let mut snapshot = RuntimeSnapshot::new("child-session-metadata");
    snapshot.push_frame(
        RuntimeFrame::new(
            RuntimeFrameKind::User,
            FrameVisibility::Active,
            RuntimeFrameProvenance::new(RuntimeSource::Transcript),
            RuntimeFrameIdSeed {
                frame_kind: RuntimeFrameKind::User,
                source: RuntimeSource::Transcript,
                ordinal: 0,
                stable_key: "child-session-metadata-user",
                source_span: None,
            },
        )
        .with_protocol(ProtocolFrameItem::UserMessage {
            content: UserMessageContent::from("continue delegated work with src/subagent.rs"),
        }),
    );
    let mut subagent_evidence = evidence(
        "subagent-evidence",
        "SUBAGENT-EVIDENCE-SENTINEL",
        "src/subagent.rs",
        1,
    );
    subagent_evidence.source = EvidenceSource::Subagent {
        run_id: "run-1".into(),
        child_session_id: "child-1".into(),
        source_session_id: "child-session-1".into(),
        parent_tool: "agent__implementer".into(),
        parent_turn_id: Some("turn-1".into()),
        parent_session_id: Some("parent-session".into()),
    };
    snapshot.set_evidence(vec![subagent_evidence]);
    let mut with_child_session = snapshot.clone();
    with_child_session.push_child_session(RuntimeChildSession {
        parent_run_id: "run-1".into(),
        child_session_id: "child-1".into(),
        agent_name: "implementer".into(),
        status: "completed".into(),
        summary: "CHILD-SESSION-METADATA-SENTINEL".into(),
        timestamp_ms: 1,
    });
    let prelude = [PromptMessage::developer_with_origin(
        "UNRECONCILED-SUBAGENT-CONTEXT-SENTINEL",
        PromptMessageOrigin::UnreconciledSubagentContext,
    )];

    for protocol in [ApiProtocol::Responses, ApiProtocol::Completions] {
        let without_child_session = build_request(RequestBuilderInput {
            protocol,
            model_id: "gpt-test",
            model: metadata(8192),
            prelude: &prelude,
            snapshot: &snapshot,
            tools: &[],
        })
        .expect("request without child session builds");
        let with_child_session_request = build_request(RequestBuilderInput {
            protocol,
            model_id: "gpt-test",
            model: metadata(8192),
            prelude: &prelude,
            snapshot: &with_child_session,
            tools: &[],
        })
        .expect("request with child session builds");

        assert_eq!(
            request_value(&without_child_session),
            request_value(&with_child_session_request),
            "{protocol:?} provider request must ignore child session metadata"
        );
        assert_eq!(
            without_child_session.prompt_plan, with_child_session_request.prompt_plan,
            "{protocol:?} prompt plan must ignore child session metadata"
        );
        assert_eq!(
            without_child_session.selected_evidence_ids,
            with_child_session_request.selected_evidence_ids
        );
        let request = request_json(with_child_session_request);
        assert!(request.contains("SUBAGENT-EVIDENCE-SENTINEL"));
        assert!(!request.contains("UNRECONCILED-SUBAGENT-CONTEXT-SENTINEL"));
        assert!(!request.contains("CHILD-SESSION-METADATA-SENTINEL"));
    }
}

fn phase1b_exact_tool_result_json(bytes: usize, tool: &str) -> String {
    let empty = ToolResult::ok(tool, json!({"payload": ""}));
    let fixed = serde_json::to_string(&empty)
        .expect("ToolResult serializes")
        .len();
    assert!(bytes >= fixed, "fixture must fit ToolResult framing");
    let result = ToolResult::ok(tool, json!({"payload": "x".repeat(bytes - fixed)}));
    let serialized = serde_json::to_string(&result).expect("ToolResult serializes");
    assert_eq!(
        serialized.len(),
        bytes,
        "fixture has exact serialized length"
    );
    serialized
}

fn canonical_tool_result_snapshot(bytes: usize) -> RuntimeSnapshot {
    let tool = "shell__exec";
    let output_json = phase1b_exact_tool_result_json(bytes, tool);
    let span = crate::runtime_context::SourceSpan::new(42, 42).expect("singleton span");
    let frame = |kind, ordinal, item| {
        RuntimeFrame::new(
            kind,
            FrameVisibility::Active,
            RuntimeFrameProvenance::new(RuntimeSource::Transcript).with_span(span),
            RuntimeFrameIdSeed {
                frame_kind: kind,
                source: RuntimeSource::Transcript,
                ordinal,
                stable_key: "canonical-pressure",
                source_span: Some(span),
            },
        )
        .with_protocol(item)
    };
    let call = frame(
        RuntimeFrameKind::ToolCall,
        0,
        ProtocolFrameItem::AssistantToolCalls {
            text: None,
            calls: vec![HistoryToolCall {
                call_id: "phase1b-call".into(),
                name: tool.into(),
                arguments_json: "{}".into(),
            }],
        },
    );
    let output = frame(
        RuntimeFrameKind::ToolOutput,
        1,
        ProtocolFrameItem::ToolOutput {
            call_id: "phase1b-call".into(),
            output_json: output_json.clone(),
        },
    );
    let user = frame(
        RuntimeFrameKind::User,
        2,
        ProtocolFrameItem::UserMessage {
            content: UserMessageContent::from("continue"),
        },
    );
    let mut snapshot = RuntimeSnapshot::new("canonical-pressure");
    snapshot.set_protected_frame_ids(vec![call.id, output.id, user.id]);
    snapshot.push_frame(call);
    snapshot.push_frame(output);
    snapshot.push_frame(user);
    if bytes > crate::context_view::INLINE_TOOL_RESULT_MAX_BYTES {
        let output_id = "folded-output-seq-42-tool-result";
        snapshot.context_view.folded_outputs.insert(
            output_id.into(),
            FoldedOutputMetadata {
                output_id: output_id.into(),
                node_id: None,
                output_kind: "tool_result".into(),
                call_id: Some("phase1b-call".into()),
                tool_name: Some(tool.into()),
                stream: Some("tool_result".into()),
                content: output_json.clone(),
                byte_count: output_json.len(),
                line_count: 1,
                truncated: false,
                shell_command: None,
                source_start_sequence: Some(42),
                source_end_sequence: Some(42),
                available_sequence: Some(42),
                tool_ok: Some(true),
                exit_status: None,
                provider_metadata: None,
                provider_fold_eligible: true,
            },
        );
        let block_id = ContextBlockId::new("phase1b-aggregate-block").expect("block id");
        snapshot.context_view.blocks.insert(
            block_id.clone(),
            ContextBlock {
                block_id,
                node_id: None,
                kind: ContextBlockKind::ToolOutput,
                title: "aggregate".into(),
                detail: String::new(),
                source: ContextBlockSource::FoldedOutput {
                    output_id: output_id.into(),
                },
                source_start_sequence: Some(42),
                available_sequence: Some(42),
                protected_reasons: Vec::new(),
                folded_output_id: Some(output_id.into()),
            },
        );
    }
    snapshot
}

#[test]
fn canonical_first_exposure_representation_is_pressure_invariant_and_impossible_budget_fails_fast()
{
    for (label, bytes, expected_raw) in [
        (
            "under_limit_raw",
            crate::context_view::INLINE_TOOL_RESULT_MAX_BYTES,
            true,
        ),
        (
            "oversized_placeholder",
            crate::context_view::INLINE_TOOL_RESULT_MAX_BYTES + 1,
            false,
        ),
    ] {
        let snapshot = canonical_tool_result_snapshot(bytes);
        let before = snapshot.clone();
        for protocol in [ApiProtocol::Responses, ApiProtocol::Completions] {
            let mut model = metadata_with_effective_input_limit(16_384, 1_500);
            let mut rendered = None;
            for (pressure, reserve) in [("none", 0), ("soft", 600), ("hard", 1_300)] {
                let result = build_request_with_policy(
                    RequestBuilderInput {
                        protocol,
                        model_id: "gpt-test",
                        model: model.clone(),
                        prelude: &[],
                        snapshot: &snapshot,
                        tools: &[],
                    },
                    None,
                    Some(ProtectedContextPolicy::from_configured_reserve(
                        Some(reserve),
                        1_500,
                    )),
                )
                .unwrap_or_else(|error| panic!("{label} {protocol:?} {pressure}: {error}"));
                let request = request_value(&result);
                let (call_ids, output_call_ids, outputs) = match protocol {
                    ApiProtocol::Responses => (
                        request["input"]
                            .as_array()
                            .expect("responses input")
                            .iter()
                            .filter(|item| item["type"] == "function_call")
                            .map(|item| item["call_id"].as_str().unwrap().to_owned())
                            .collect::<Vec<_>>(),
                        request["input"]
                            .as_array()
                            .expect("responses input")
                            .iter()
                            .filter(|item| item["type"] == "function_call_output")
                            .map(|item| item["call_id"].as_str().unwrap().to_owned())
                            .collect::<Vec<_>>(),
                        request["input"]
                            .as_array()
                            .expect("responses input")
                            .iter()
                            .filter(|item| item["type"] == "function_call_output")
                            .map(|item| item["output"].as_str().unwrap().to_owned())
                            .collect::<Vec<_>>(),
                    ),
                    ApiProtocol::Completions => (
                        request["messages"]
                            .as_array()
                            .expect("chat messages")
                            .iter()
                            .filter(|item| item["role"] == "assistant")
                            .flat_map(|item| item["tool_calls"].as_array().into_iter().flatten())
                            .map(|item| item["id"].as_str().unwrap().to_owned())
                            .collect::<Vec<_>>(),
                        request["messages"]
                            .as_array()
                            .expect("chat messages")
                            .iter()
                            .filter(|item| item["role"] == "tool")
                            .map(|item| item["tool_call_id"].as_str().unwrap().to_owned())
                            .collect::<Vec<_>>(),
                        request["messages"]
                            .as_array()
                            .expect("chat messages")
                            .iter()
                            .filter(|item| item["role"] == "tool")
                            .map(|item| item["content"].as_str().unwrap().to_owned())
                            .collect::<Vec<_>>(),
                    ),
                };
                assert_eq!(call_ids, ["phase1b-call"]);
                assert_eq!(output_call_ids, call_ids, "tool pair/order is preserved");
                assert_eq!(outputs.len(), 1);
                assert_eq!(
                    outputs[0] == phase1b_exact_tool_result_json(bytes, "shell__exec"),
                    expected_raw
                );
                if !expected_raw {
                    assert!(outputs[0].contains("folded-output-seq-42-tool-result"));
                }
                let bytes = serde_json::to_vec(&request).expect("provider request serializes");
                if let Some(previous) = &rendered {
                    assert_eq!(
                        previous, &bytes,
                        "{label} {protocol:?} changes under {pressure}"
                    );
                } else {
                    rendered = Some(bytes);
                }
            }
        }
        assert_eq!(
            snapshot, before,
            "{label} pressure must not mutate authority"
        );
    }

    let snapshot =
        canonical_tool_result_snapshot(crate::context_view::INLINE_TOOL_RESULT_MAX_BYTES + 1);
    let before = snapshot.clone();
    let mut model = metadata_with_effective_input_limit(16_384, 1);
    let error = build_request_with_policy(
        RequestBuilderInput {
            protocol: ApiProtocol::Responses,
            model,
            model_id: "gpt-test",
            prelude: &[],
            snapshot: &snapshot,
            tools: &[],
        },
        None,
        Some(ProtectedContextPolicy::from_configured_reserve(Some(1), 1)),
    )
    .expect_err("impossible canonical hard budget must fail without a pressure projection");
    assert!(
        error
            .to_string()
            .contains("protected current context exceeds input budget")
    );
    assert_eq!(snapshot, before);
}

#[test]
fn phase2_transcript_artifacts_fold_for_both_provider_protocols_with_native_metadata() {
    let large = format!(
        "phase2-needle-{}",
        "x".repeat(DEFAULT_FOLDED_OUTPUT_THRESHOLD_BYTES)
    );
    let mixed_mcp_text = format!(
        "MIXED-MCP-DERIVED-TEXT-{}",
        "m".repeat(DEFAULT_FOLDED_OUTPUT_THRESHOLD_BYTES)
    );
    let records = vec![
        transcript_record(
            1,
            TranscriptEvent::UserMessage {
                content: UserMessageContent::from("inspect artifacts"),
            },
        ),
        transcript_record(
            2,
            TranscriptEvent::TurnStarted(crate::agent::TurnStartedEvent {
                turn_id: 1,
                intent: "engineering".into(),
                directive: "none".into(),
                validation_reminder: "none".into(),
            }),
        ),
        transcript_record(
            3,
            TranscriptEvent::ToolCallStarted {
                call_id: "duplicate-call".into(),
                name: crate::tool_names::TOOL_FS_READ.into(),
                args: json!({"path":"src/lib.rs"}),
            },
        ),
        transcript_record(
            4,
            TranscriptEvent::ToolCallFinished {
                call_id: "duplicate-call".into(),
                name: crate::tool_names::TOOL_FS_READ.into(),
                ok: true,
                output: crate::tool::ToolResult::ok(
                    crate::tool_names::TOOL_FS_READ,
                    json!({
                        "path":"src/lib.rs", "content":large, "offset":10, "start_line":10, "end_line":20, "next_offset":21, "has_more":true, "total_bytes":99999, "truncated":true
                    }),
                ),
            },
        ),
        transcript_record(
            5,
            TranscriptEvent::ToolCallStarted {
                call_id: "duplicate-call".into(),
                name: crate::tool_names::TOOL_SEARCH_RG.into(),
                args: json!({"pattern":"needle"}),
            },
        ),
        transcript_record(
            6,
            TranscriptEvent::ToolCallFinished {
                call_id: "duplicate-call".into(),
                name: crate::tool_names::TOOL_SEARCH_RG.into(),
                ok: true,
                output: crate::tool::ToolResult::ok(
                    crate::tool_names::TOOL_SEARCH_RG,
                    json!({
                        "pattern":"needle", "path":"src", "matches":[{"text":large}], "truncated":true, "status":0, "success":true
                    }),
                ),
            },
        ),
        transcript_record(
            7,
            TranscriptEvent::ToolCallStarted {
                call_id: "mcp-call".into(),
                name: "mcp__call".into(),
                args: json!({}),
            },
        ),
        transcript_record(
            8,
            TranscriptEvent::ToolCallFinished {
                call_id: "mcp-call".into(),
                name: "mcp__call".into(),
                ok: true,
                output: crate::tool::ToolResult::ok(
                    "mcp__call",
                    json!({
                        "server":"github", "tool":"search", "content":[{"type":"text","text":large}]
                    }),
                ),
            },
        ),
        transcript_record(
            9,
            TranscriptEvent::ToolCallStarted {
                call_id: "mixed-mcp-call".into(),
                name: "mcp__call".into(),
                args: json!({}),
            },
        ),
        transcript_record(
            10,
            TranscriptEvent::ToolCallFinished {
                call_id: "mixed-mcp-call".into(),
                name: "mcp__call".into(),
                ok: true,
                output: crate::tool::ToolResult::ok(
                    "mcp__call",
                    json!({
                        "server":"github", "tool":"search", "content":[
                            {"type":"text","text":mixed_mcp_text},
                            {"type":"image","data":"MIXED-MCP-IMAGE-RAW"}
                        ]
                    }),
                ),
            },
        ),
    ];
    let restored: Vec<TranscriptRecord> =
        serde_json::from_str(&serde_json::to_string(&records).expect("serialize transcript"))
            .expect("read transcript");
    let snapshot = crate::transcript::transcript_projection::project_runtime_restore_snapshot(
        "s".into(),
        restored,
        crate::transcript::transcript_projection::SessionContextCursor {
            branch_id: None,
            leaf_sequence: None,
        },
        &[],
    )
    .expect("project transcript")
    .snapshot;
    let expected = [
        "duplicate-call",
        "duplicate-call",
        "mcp-call",
        "mixed-mcp-call",
    ];
    let mixed = snapshot
        .context_view
        .folded_outputs
        .get("folded-output-seq-10-text")
        .expect("mixed MCP text remains addressable");
    assert!(mixed.content.starts_with("MIXED-MCP-DERIVED-TEXT-"));
    assert!(!mixed.provider_fold_eligible);
    assert_eq!(
        snapshot
            .context_view
            .folded_outputs
            .get("folded-output-seq-4-content")
            .expect("first duplicate-call artifact")
            .source_start_sequence,
        Some(4)
    );
    assert_eq!(
        snapshot
            .context_view
            .folded_outputs
            .get("folded-output-seq-6-matches")
            .expect("second duplicate-call artifact")
            .source_start_sequence,
        Some(6)
    );

    for protocol in [ApiProtocol::Responses, ApiProtocol::Completions] {
        let result = build_request_with_policy(
            RequestBuilderInput {
                protocol,
                model_id: "gpt-test",
                model: metadata_with_effective_input_limit(10_192, 8_192),
                prelude: &[],
                snapshot: &snapshot,
                tools: &[],
            },
            None,
            Some(ProtectedContextPolicy::from_configured_reserve(
                Some(5_000),
                8_192,
            )),
        )
        .expect("provider request");
        assert_eq!(result.budget.provider_folded_output_count, expected.len());
        let request: Value = serde_json::from_str(&request_json(result)).expect("request JSON");
        let assistant_calls = match protocol {
            ApiProtocol::Responses => request["input"]
                .as_array()
                .expect("responses input")
                .iter()
                .filter(|item| item["type"] == "function_call")
                .map(|item| item["call_id"].as_str().unwrap())
                .collect::<Vec<_>>(),
            ApiProtocol::Completions => request["messages"]
                .as_array()
                .expect("completion messages")
                .iter()
                .filter(|item| item["role"] == "assistant")
                .flat_map(|item| {
                    item["tool_calls"]
                        .as_array()
                        .into_iter()
                        .flatten()
                        .map(|call| call["id"].as_str().unwrap())
                })
                .collect::<Vec<_>>(),
        };
        let outputs = match protocol {
            ApiProtocol::Responses => request["input"]
                .as_array()
                .expect("responses input")
                .iter()
                .filter(|item| item["type"] == "function_call_output")
                .map(|item| {
                    (
                        item["call_id"].as_str().unwrap(),
                        item["output"].as_str().unwrap(),
                    )
                })
                .collect::<Vec<_>>(),
            ApiProtocol::Completions => request["messages"]
                .as_array()
                .expect("completion messages")
                .iter()
                .filter(|item| item["role"] == "tool")
                .map(|item| {
                    (
                        item["tool_call_id"].as_str().unwrap(),
                        item["content"].as_str().unwrap(),
                    )
                })
                .collect::<Vec<_>>(),
        };
        assert_eq!(
            outputs.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            expected
        );
        assert_eq!(assistant_calls, expected);
        assert_eq!(
            assistant_calls,
            outputs.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            "{protocol:?} assistant calls and outputs must pair in source order"
        );
        let mixed_output = match protocol {
            ApiProtocol::Responses => request["input"]
                .as_array()
                .expect("responses input")
                .iter()
                .find(|item| {
                    item["type"] == "function_call_output" && item["call_id"] == "mixed-mcp-call"
                })
                .and_then(|item| item["output"].as_str()),
            ApiProtocol::Completions => request["messages"]
                .as_array()
                .expect("completion messages")
                .iter()
                .find(|item| item["role"] == "tool" && item["tool_call_id"] == "mixed-mcp-call")
                .and_then(|item| item["content"].as_str()),
        }
        .expect("mixed MCP canonical output");
        assert!(mixed_output.contains("folded-output-seq-10-tool-result"));
        assert!(!mixed_output.contains("MIXED-MCP-IMAGE-RAW"));
        assert!(!mixed_output.contains("MIXED-MCP-DERIVED-TEXT"));
        let placeholders = outputs
            .iter()
            .map(|(_, output)| serde_json::from_str::<Value>(output).expect("placeholder"))
            .collect::<Vec<_>>();
        assert_eq!(placeholders.len(), 4);
        assert!(placeholders.iter().all(|placeholder| {
            placeholder["folded_outputs"][0]["provider_metadata"].is_null()
        }));
    }
}

#[test]
fn phase5b5_default_view_fallback_characterizes_visibility_and_provenance() {
    use crate::runtime_context::{FoldedOutputReference, SourceSpan};
    let make = |kind, visibility, ordinal, text, span: Option<SourceSpan>| {
        RuntimeFrame::new(
            kind,
            visibility,
            RuntimeFrameProvenance::new(RuntimeSource::ContextView)
                .with_source_id(format!("f-{ordinal}"))
                .with_span(span.unwrap_or_else(|| {
                    SourceSpan::new(100 + ordinal as u64, 100 + ordinal as u64).unwrap()
                })),
            RuntimeFrameIdSeed {
                frame_kind: kind,
                source: RuntimeSource::ContextView,
                ordinal,
                stable_key: "phase5b5-fallback",
                source_span: span,
            },
        )
        .with_summary(text)
    };
    let mut snapshot = RuntimeSnapshot::new("phase5b5-fallback");
    let context = make(
        RuntimeFrameKind::ContextBlock,
        FrameVisibility::Active,
        0,
        "VISIBLE-CONTEXT",
        None,
    );
    let summary = make(
        RuntimeFrameKind::Summary,
        FrameVisibility::Active,
        1,
        "VISIBLE-SUMMARY",
        None,
    );
    let folded = make(
        RuntimeFrameKind::ContextBlock,
        FrameVisibility::Folded,
        2,
        "FOLDED",
        None,
    );
    let retired = make(
        RuntimeFrameKind::Summary,
        FrameVisibility::Retired,
        3,
        "RETIRED",
        None,
    );
    let compacted = make(
        RuntimeFrameKind::ContextBlock,
        FrameVisibility::Active,
        4,
        "COMPACTED",
        None,
    );
    let span = SourceSpan::new(50, 50).unwrap();
    let retired_span = make(
        RuntimeFrameKind::Summary,
        FrameVisibility::Active,
        5,
        "RETIRED-SPAN",
        Some(span),
    );
    snapshot.compaction.compacted_frame_ids.push(compacted.id);
    snapshot.compaction.retired_source_spans.push(span);
    for frame in [
        context.clone(),
        summary.clone(),
        folded,
        retired,
        compacted,
        retired_span,
    ] {
        snapshot.push_frame(frame);
    }
    for (output_id, call_id, tool_name, source_span) in [
        ("visible-folded", Some("call"), Some("tool"), None),
        ("folded-output-seq-99-tool-result", None, None, None),
        ("retired-folded", None, None, Some(span)),
    ] {
        snapshot.push_folded_output(FoldedOutputReference {
            output_id: output_id.into(),
            node_id: None,
            call_id: call_id.map(str::to_owned),
            tool_name: tool_name.map(str::to_owned),
            source_span,
        });
    }
    let projection = runtime_context_history_adapter(&snapshot, &[], 0);
    let repeated = runtime_context_history_adapter(&snapshot, &[], 0);
    assert_eq!(projection.prelude, repeated.prelude);
    assert_eq!(projection.history_prefix, repeated.history_prefix);
    assert_eq!(projection.history_prefix.len(), 3);
    assert_eq!(
        projection
            .history_prefix
            .iter()
            .map(|f| &f.item)
            .collect::<Vec<_>>(),
        vec![
            &ProtocolFrameItem::ContextSummary {
                text: "[Context: Runtime Material]\nVISIBLE-CONTEXT".into()
            },
            &ProtocolFrameItem::ContextSummary {
                text: "[Context: Runtime Material]\nVISIBLE-SUMMARY".into()
            },
            &ProtocolFrameItem::ContextSummary {
                text:
                    "[Context: Folded Outputs]\n- output_id=visible-folded tool=tool call_id=call"
                        .into()
            },
        ]
    );
    assert_eq!(
        projection.history_prefix[0].runtime_frame_id,
        Some(context.id)
    );
    assert_eq!(
        projection.history_prefix[1].runtime_frame_id,
        Some(summary.id)
    );
    assert_eq!(projection.history_prefix[2].runtime_frame_id, None);
    assert!(
        projection
            .history_prefix
            .iter()
            .all(|f| f.history_index == usize::MAX)
    );
    assert_eq!(
        projection.history_prefix[0].source_provenance,
        Some(context.provenance)
    );
    assert_eq!(
        projection.history_prefix[1].source_provenance,
        Some(summary.provenance)
    );
    assert_eq!(
        projection.history_prefix[2].source_provenance,
        Some(
            RuntimeFrameProvenance::new(RuntimeSource::FoldedOutput)
                .with_source_id("visible-folded")
        )
    );
}

#[test]
fn phase5b5_generic_contributors_characterize_filtering_and_sections() {
    use crate::runtime_context::{PromptContributorPlaceholder, SourceSpan};
    let make = |ordinal, visibility, text: Option<&str>, span: Option<SourceSpan>| {
        let frame = RuntimeFrame::new(
            RuntimeFrameKind::PromptContributor,
            visibility,
            RuntimeFrameProvenance::new(RuntimeSource::PromptContributor).with_span(
                span.unwrap_or_else(|| {
                    SourceSpan::new(200 + ordinal as u64, 200 + ordinal as u64).unwrap()
                }),
            ),
            RuntimeFrameIdSeed {
                frame_kind: RuntimeFrameKind::PromptContributor,
                source: RuntimeSource::PromptContributor,
                ordinal,
                stable_key: "phase5b5-contributor",
                source_span: span,
            },
        );
        text.map_or(frame.clone(), |text| frame.with_summary(text))
    };
    let mut snapshot = RuntimeSnapshot::new("phase5b5-contributor");
    let one = make(0, FrameVisibility::Active, Some("ONE"), None);
    let two = make(1, FrameVisibility::Active, Some("TWO"), None);
    let unlabeled = make(2, FrameVisibility::Active, Some("UNLABELED"), None);
    let retired = make(3, FrameVisibility::Retired, Some("RETIRED"), None);
    let folded = make(4, FrameVisibility::Folded, Some("FOLDED"), None);
    let retired_span = SourceSpan::new(300, 300).unwrap();
    let spanned = make(
        5,
        FrameVisibility::Active,
        Some("SPANNED"),
        Some(retired_span),
    );
    let no_summary = make(6, FrameVisibility::Active, None, None);
    for frame in [
        one.clone(),
        two.clone(),
        unlabeled.clone(),
        retired.clone(),
        folded.clone(),
        spanned.clone(),
        no_summary.clone(),
    ] {
        snapshot.push_frame(frame);
    }
    snapshot.compaction.retired_source_spans.push(retired_span);
    let add =
        |snapshot: &mut RuntimeSnapshot, id: &str, kind, label: Option<&str>, ids, provenance| {
            snapshot.push_prompt_contributor(PromptContributorPlaceholder {
                contributor_id: id.into(),
                kind,
                label: label.map(str::to_owned),
                provenance,
                frame_ids: ids,
                source_frame_ids: vec![],
            })
        };
    for id in [
        "context-view-active",
        "evidence",
        "summary-artifacts",
        "folded-outputs",
        "child-sessions",
    ] {
        add(
            &mut snapshot,
            id,
            PromptContributorKind::Other,
            Some("RESERVED"),
            vec![one.id],
            RuntimeFrameProvenance::new(RuntimeSource::PromptContributor),
        );
    }
    add(
        &mut snapshot,
        "skill",
        PromptContributorKind::SkillMaterial,
        None,
        vec![one.id],
        RuntimeFrameProvenance::new(RuntimeSource::PromptContributor),
    );
    let contributor_span = SourceSpan::new(400, 400).unwrap();
    snapshot
        .compaction
        .retired_source_spans
        .push(contributor_span);
    add(
        &mut snapshot,
        "retired-contributor",
        PromptContributorKind::Other,
        None,
        vec![one.id],
        RuntimeFrameProvenance::new(RuntimeSource::PromptContributor).with_span(contributor_span),
    );
    for (id, frame) in [
        ("retired", retired.id),
        ("folded", folded.id),
        ("spanned", spanned.id),
        ("none", no_summary.id),
    ] {
        add(
            &mut snapshot,
            id,
            PromptContributorKind::Other,
            None,
            vec![frame],
            RuntimeFrameProvenance::new(RuntimeSource::PromptContributor),
        );
    }
    let labeled_provenance =
        RuntimeFrameProvenance::new(RuntimeSource::PromptContributor).with_source_id("labeled");
    let unlabeled_provenance =
        RuntimeFrameProvenance::new(RuntimeSource::PromptContributor).with_source_id("unlabeled");
    add(
        &mut snapshot,
        "labeled",
        PromptContributorKind::Other,
        Some("Labeled"),
        vec![one.id, two.id],
        labeled_provenance.clone(),
    );
    add(
        &mut snapshot,
        "unlabeled",
        PromptContributorKind::Other,
        None,
        vec![unlabeled.id],
        unlabeled_provenance.clone(),
    );
    snapshot.recompute_protected_frame_ids();
    snapshot.validate_references().unwrap();
    let sections = runtime_context_history_adapter(&snapshot, &[], 0).history_prefix;
    assert_eq!(
        sections.iter().map(|f| &f.item).collect::<Vec<_>>(),
        vec![
            &ProtocolFrameItem::ContextSummary {
                text: "[Context: Labeled]\nONE\nTWO".into()
            },
            &ProtocolFrameItem::ContextSummary {
                text: "[Context: unlabeled]\nUNLABELED".into()
            }
        ]
    );
    assert!(
        sections
            .iter()
            .all(|f| f.runtime_frame_id.is_none() && f.history_index == usize::MAX)
    );
    assert_eq!(sections[0].source_provenance, Some(labeled_provenance));
    assert_eq!(sections[1].source_provenance, Some(unlabeled_provenance));
}

#[test]
fn phase5b5_provider_visible_protocol_frames_characterize_dense_projection() {
    use crate::runtime_context::SourceSpan;
    let make = |kind, ordinal, item, span: Option<SourceSpan>| {
        RuntimeFrame::new(
            kind,
            FrameVisibility::Active,
            RuntimeFrameProvenance::new(RuntimeSource::Transcript)
                .with_source_id(format!("p-{ordinal}"))
                .with_span(span.unwrap_or_else(|| {
                    SourceSpan::new(500 + ordinal as u64, 500 + ordinal as u64).unwrap()
                })),
            RuntimeFrameIdSeed {
                frame_kind: kind,
                source: RuntimeSource::Transcript,
                ordinal,
                stable_key: "phase5b5-protocol",
                source_span: span,
            },
        )
        .with_protocol(item)
    };
    let mut snapshot = RuntimeSnapshot::new("phase5b5-protocol");
    let retired = RuntimeFrame::new(
        RuntimeFrameKind::Assistant,
        FrameVisibility::Retired,
        RuntimeFrameProvenance::new(RuntimeSource::Transcript),
        RuntimeFrameIdSeed {
            frame_kind: RuntimeFrameKind::Assistant,
            source: RuntimeSource::Transcript,
            ordinal: 0,
            stable_key: "retired",
            source_span: None,
        },
    )
    .with_protocol(ProtocolFrameItem::assistant("RETIRED"));
    let call = make(
        RuntimeFrameKind::ToolCall,
        1,
        ProtocolFrameItem::AssistantToolCalls {
            text: None,
            calls: vec![HistoryToolCall {
                call_id: "call".into(),
                name: "tool".into(),
                arguments_json: "{}".into(),
            }],
        },
        None,
    );
    let nonprotocol = RuntimeFrame::new(
        RuntimeFrameKind::ContextBlock,
        FrameVisibility::Active,
        RuntimeFrameProvenance::new(RuntimeSource::ContextView),
        RuntimeFrameIdSeed {
            frame_kind: RuntimeFrameKind::ContextBlock,
            source: RuntimeSource::ContextView,
            ordinal: 2,
            stable_key: "nonprotocol",
            source_span: None,
        },
    )
    .with_summary("NONPROTOCOL");
    let output = make(
        RuntimeFrameKind::ToolOutput,
        3,
        ProtocolFrameItem::ToolOutput {
            call_id: "call".into(),
            output_json: "output".into(),
        },
        None,
    );
    let user = make(
        RuntimeFrameKind::User,
        4,
        ProtocolFrameItem::user("USER"),
        None,
    );
    let compacted = make(
        RuntimeFrameKind::Assistant,
        5,
        ProtocolFrameItem::assistant("COMPACTED"),
        None,
    );
    let span = SourceSpan::new(600, 600).unwrap();
    let retired_span = make(
        RuntimeFrameKind::Assistant,
        6,
        ProtocolFrameItem::assistant("RETIRED-SPAN"),
        Some(span),
    );
    snapshot.compaction.compacted_frame_ids.push(compacted.id);
    snapshot.compaction.retired_source_spans.push(span);
    for frame in [
        retired,
        call.clone(),
        nonprotocol,
        output.clone(),
        user.clone(),
        compacted,
        retired_span,
    ] {
        snapshot.push_frame(frame);
    }
    let frames = provider_visible_protocol_frames(&snapshot);
    assert_eq!(
        frames.iter().map(|f| f.history_index).collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
    assert_eq!(
        frames
            .iter()
            .map(|f| (
                f.runtime_frame_id,
                f.source_provenance.clone(),
                f.item.clone()
            ))
            .collect::<Vec<_>>(),
        vec![
            (Some(call.id), Some(call.provenance), call.protocol.unwrap()),
            (
                Some(output.id),
                Some(output.provenance),
                output.protocol.unwrap()
            ),
            (Some(user.id), Some(user.provenance), user.protocol.unwrap())
        ]
    );
    validate_history_items_complete(&history_items_from_frames(&frames), None).unwrap();
}

#[test]
fn phase5b5_protected_boundary_characterizes_dense_group_expansion_without_mutation() {
    let make = |kind, ordinal, item| {
        RuntimeFrame::new(
            kind,
            FrameVisibility::Active,
            RuntimeFrameProvenance::new(RuntimeSource::Transcript),
            RuntimeFrameIdSeed {
                frame_kind: kind,
                source: RuntimeSource::Transcript,
                ordinal,
                stable_key: "phase5b5-protected",
                source_span: None,
            },
        )
        .with_protocol(item)
    };
    let mut snapshot = RuntimeSnapshot::new("phase5b5-protected");
    let filtered = make(
        RuntimeFrameKind::User,
        0,
        ProtocolFrameItem::user("FILTERED"),
    );
    let unprotected = make(
        RuntimeFrameKind::Assistant,
        1,
        ProtocolFrameItem::assistant("UNPROTECTED"),
    );
    let nonprotocol = RuntimeFrame::new(
        RuntimeFrameKind::ContextBlock,
        FrameVisibility::Active,
        RuntimeFrameProvenance::new(RuntimeSource::ContextView),
        RuntimeFrameIdSeed {
            frame_kind: RuntimeFrameKind::ContextBlock,
            source: RuntimeSource::ContextView,
            ordinal: 2,
            stable_key: "phase5b5-protected",
            source_span: None,
        },
    )
    .with_summary("NONPROTOCOL");
    let call = make(
        RuntimeFrameKind::ToolCall,
        3,
        ProtocolFrameItem::AssistantToolCalls {
            text: None,
            calls: vec![HistoryToolCall {
                call_id: "call".into(),
                name: "tool".into(),
                arguments_json: "{}".into(),
            }],
        },
    );
    let output = make(
        RuntimeFrameKind::ToolOutput,
        4,
        ProtocolFrameItem::ToolOutput {
            call_id: "call".into(),
            output_json: "output".into(),
        },
    );
    let current = make(
        RuntimeFrameKind::User,
        5,
        ProtocolFrameItem::user("CURRENT"),
    );
    snapshot.compaction.compacted_frame_ids.push(filtered.id);
    snapshot.set_protected_frame_ids(vec![output.id, current.id]);
    for frame in [
        filtered,
        unprotected,
        nonprotocol,
        call.clone(),
        output.clone(),
        current.clone(),
    ] {
        snapshot.push_frame(frame);
    }
    let before = snapshot.clone();
    let frames = provider_visible_protocol_frames(&snapshot);
    let start = protected_start_index_for_snapshot(&snapshot, &frames);
    assert_eq!(start, 2);
    let history = history_items_from_frames(&frames);
    let transcript = validate_history_items_complete(&history, Some(start)).unwrap();
    assert_eq!(transcript.tool_call_groups.len(), 1);
    assert_eq!(transcript.tool_call_groups[0].assistant_index, 1);
    assert_eq!(transcript.tool_call_groups[0].tool_output_indexes, vec![2]);
    assert!(transcript.tool_call_groups[0].protection.current_turn);
    assert_eq!(expand_protected_start_to_group(&history, start).unwrap(), 1);
    assert!(
        !snapshot.compaction.protected_frame_ids.contains(&call.id)
            && snapshot.compaction.protected_frame_ids.contains(&output.id)
            && snapshot
                .compaction
                .protected_frame_ids
                .contains(&current.id)
    );
    assert_eq!(snapshot, before);
}
