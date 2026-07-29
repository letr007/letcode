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
    project_context_view,
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
fn fast_mode_serializes_typed_priority_tier() {
    for protocol in [ApiProtocol::Responses, ApiProtocol::Completions] {
        let mut model = metadata(8192);
        model.fast_mode = true;
        let result = build_test_request(TestRequestBuilderInput {
            protocol,
            model_id: "gpt-test",
            model,
            prelude: &[],
            history: &[HistoryItem::user("current question")],
            protected_start_index: 0,
            tools: &[],
            evidence: &[],
            history_adapter: None,
            context_view: None,
        })
        .expect("fast mode request builds");

        assert!(matches!(
            (&protocol, &result.request),
            (ApiProtocol::Responses, BuiltRequest::Responses(_))
                | (ApiProtocol::Completions, BuiltRequest::Completions(_))
        ));
        assert_eq!(request_value(&result)["service_tier"], "priority");
    }
}

#[test]
fn normal_requests_omit_fast_mode_service_tier() {
    for protocol in [ApiProtocol::Responses, ApiProtocol::Completions] {
        let result = build_test_request(TestRequestBuilderInput {
            protocol,
            model_id: "gpt-test",
            model: metadata(8192),
            prelude: &[],
            history: &[HistoryItem::user("current question")],
            protected_start_index: 0,
            tools: &[],
            evidence: &[],
            history_adapter: None,
            context_view: None,
        })
        .expect("normal request builds");

        assert!(request_value(&result).get("service_tier").is_none());
    }
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
fn provider_serialization_preserves_interleaved_user_content_part_order() {
    let image = |id: &str| UserImageAttachment {
        id: id.into(),
        label: format!("{id}.png"),
        mime: "image/png".into(),
        data_url: format!("data:image/png;base64,{id}"),
    };
    let content = UserMessageContent::from_parts(vec![
        crate::user_content::UserMessagePart::Text {
            text: "before".into(),
        },
        crate::user_content::UserMessagePart::Image {
            attachment: image("first"),
        },
        crate::user_content::UserMessagePart::Text {
            text: "between".into(),
        },
        crate::user_content::UserMessagePart::Image {
            attachment: image("second"),
        },
        crate::user_content::UserMessagePart::Text {
            text: "after".into(),
        },
    ])
    .with_selected_skills(vec!["rust-audit".into()]);

    for protocol in [ApiProtocol::Completions, ApiProtocol::Responses] {
        let history = vec![HistoryItem::user_content(content.clone())];
        let result = build_test_request(TestRequestBuilderInput {
            protocol,
            model_id: "ordered-parts-test",
            model: metadata(20_000),
            prelude: &[],
            history: &history,
            protected_start_index: 0,
            tools: &[],
            evidence: &[],
            history_adapter: None,
            context_view: None,
        })
        .expect("request builds");
        let json = match result.request {
            BuiltRequest::Completions(request) => serde_json::to_value(request),
            BuiltRequest::Responses(request) => serde_json::to_value(request),
            _ => unreachable!("selected protocol must build its request type"),
        }
        .expect("request serializes");
        let parts = match protocol {
            ApiProtocol::Completions => &json["messages"][0]["content"],
            ApiProtocol::Responses => &json["input"][0]["content"],
        };
        assert!(!json.to_string().contains("rust-audit"));
        let types = parts
            .as_array()
            .expect("multimodal content array")
            .iter()
            .map(|part| part["type"].as_str().expect("part type"))
            .collect::<Vec<_>>();
        let expected = match protocol {
            ApiProtocol::Completions => ["text", "image_url", "text", "image_url", "text"],
            ApiProtocol::Responses => [
                "input_text",
                "input_image",
                "input_text",
                "input_image",
                "input_text",
            ],
        };
        assert_eq!(types, expected);
    }
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

fn png_data_url(width: u32, height: u32, trailing_bytes: usize) -> String {
    use base64::{Engine as _, engine::general_purpose::STANDARD};

    let mut png = vec![137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13];
    png.extend_from_slice(b"IHDR");
    png.extend_from_slice(&width.to_be_bytes());
    png.extend_from_slice(&height.to_be_bytes());
    png.resize(png.len() + trailing_bytes, 0);
    format!("data:image/png;base64,{}", STANDARD.encode(png))
}

#[test]
fn png_images_use_dimension_budget_and_preserve_payloads_in_both_protocols() {
    let image = |data_url: String| {
        HistoryItem::user_content(UserMessageContent::new(
            "describe this image",
            vec![UserImageAttachment {
                id: "img-1".into(),
                label: "screen.png".into(),
                mime: "image/png".into(),
                data_url,
            }],
        ))
    };
    let compact_png = png_data_url(3840, 2160, 1);
    let compressed_png = png_data_url(3840, 2160, 2_100_000);
    let smaller = image(png_data_url(1920, 1080, 1));
    let compact = image(compact_png);
    let compressed = image(compressed_png.clone());

    let HistoryItem::UserMessage { content } = &compact else {
        panic!("expected user image history item");
    };
    assert_eq!(content.attachments[0].visual_token_charge(), 8_160);
    assert_eq!(
        UserImageAttachment {
            id: "opaque".into(),
            label: "opaque.jpg".into(),
            mime: "image/jpeg".into(),
            data_url: "data:image/jpeg;base64,AAAA".into(),
        }
        .visual_token_charge(),
        4_096,
        "uninspectable images receive the documented bounded visual charge"
    );
    assert_eq!(
        estimate_history_item_tokens(&compressed),
        estimate_history_item_tokens(&compact),
        "PNG transport/compression bytes must not affect the image budget"
    );
    assert!(
        estimate_history_item_tokens(&compressed) > estimate_history_item_tokens(&smaller),
        "larger pixel dimensions must consume more visual budget"
    );

    for protocol in [ApiProtocol::Responses, ApiProtocol::Completions] {
        let history = [compressed.clone()];
        let result = build_test_request(TestRequestBuilderInput {
            protocol,
            model_id: "image-test",
            model: metadata(10_000),
            prelude: &[],
            history: &history,
            protected_start_index: 0,
            tools: &[],
            evidence: &[],
            history_adapter: None,
            context_view: None,
        })
        .expect("dimension-budgeted image should fit a sufficient input budget");
        assert!(result.budget.estimated_protected_tokens >= 8_160);

        let json = match result.request {
            BuiltRequest::Responses(request) => {
                serde_json::to_value(request).expect("responses request serializes")
            }
            BuiltRequest::Completions(request) => {
                serde_json::to_value(request).expect("completions request serializes")
            }
            _ => panic!("expected request matching selected protocol"),
        };
        let actual_url = match protocol {
            ApiProtocol::Responses => json["input"][0]["content"][1]["image_url"].as_str(),
            ApiProtocol::Completions => {
                json["messages"][0]["content"][1]["image_url"]["url"].as_str()
            }
        };
        assert_eq!(actual_url, Some(compressed_png.as_str()));
    }

    let history = [compressed];
    let error = build_test_request(TestRequestBuilderInput {
        protocol: ApiProtocol::Responses,
        model_id: "image-test",
        model: metadata(4096),
        prelude: &[],
        history: &history,
        protected_start_index: 0,
        tools: &[],
        evidence: &[],
        history_adapter: None,
        context_view: None,
    })
    .expect_err("one protected 4K image must exceed a 4096-token input budget");
    assert!(
        error
            .to_string()
            .contains("protected current context exceeds input budget")
    );
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

    // History-only: neither the compatibility ContextView path nor an explicit
    // history adapter may inject synthetic prompt material through the test
    // fixture. Both must reduce to the same history-backed request.
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
    let history_only = request_json(
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
            context_view: None,
        })
        .expect("history-only request builds"),
    );

    assert_eq!(explicit, compatibility);
    assert_eq!(compatibility, history_only);
    assert!(!compatibility.contains("[Context:"));
}

#[test]
fn opened_detail_only_changes_suffix_after_stable_context_prefix() {
    let history = vec![
        HistoryItem::assistant("previous"),
        HistoryItem::user("current user"),
    ];
    // History-only: open vs closed ContextView details must not alter the
    // provider request body. Both builds collapse to the same history frames.
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
    assert_eq!(closed_json, open_json);
    assert!(!open_json.contains("[Context: Opened Details]"));
    assert!(open_json.contains("current user"));
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

    // History-only: hard-context detail lives on the ContextView surface for
    // tools/TUI, but must not be injected into the provider prompt unless the
    // same content is also present as ordinary history.
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
    assert!(!json.contains("[Context: Hard Context]"));
    assert!(!json.contains(&long_detail));
    assert!(json.contains("current user"));

    let with_history = build_test_request(TestRequestBuilderInput {
        protocol: ApiProtocol::Responses,
        model_id: "gpt-test",
        model: metadata(8192),
        prelude: &[],
        history: &[
            HistoryItem::user(long_detail.clone()),
            HistoryItem::user("current user"),
        ],
        protected_start_index: 1,
        tools: &[],
        evidence: &[],
        history_adapter: None,
        context_view: None,
    })
    .expect("history request builds");
    let history_json = request_json(with_history);
    assert!(history_json.contains(&long_detail));
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
fn restored_context_view_prompt_preserves_protected_context_and_hides_soft_deleted_blocks() {
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

    // History-only prompt path: restored ContextView projection is retained for
    // TUI/tool addressing assertions above, but the provider request is built
    // solely from the supplied history frames.
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

    assert!(json.contains("continue from restored context"));
    assert!(!json.contains("[Context: Hard Context]"));
    assert!(
        !json.contains("MUST keep raw transcript events append-only; do not purge requirements")
    );
    assert!(!json.contains("Permission denied"));
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
    let snapshot = crate::runtime_context::group_16_runtime_snapshot();
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

        // History-only prompt path: only protocol-backed history and the
        // current-turn tail are provider-visible. Context-view titles / pins /
        // Runtime context inventory does not inject synthetic prompt material.
        for surviving in [
            "CURRENT-TAIL-SENTINEL",
            "ACTIVE-FOLDED-SENTINEL",
            "SURVIVING-PROTOCOL-SENTINEL",
        ] {
            assert!(json.contains(surviving), "{protocol:?}: {json}");
        }
        for non_history_context in [
            "CANONICAL ACTIVE TITLE",
            "CANONICAL ACTIVE CONTENT",
            "PINNED ACTIVE TITLE",
        ] {
            assert!(
                !json.contains(non_history_context),
                "{protocol:?}: context-view material leaked into history-only prompt: {json}"
            );
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
    snapshot
}

#[test]
fn canonical_first_exposure_representation_is_pressure_invariant_and_impossible_budget_fails_fast()
{
    for (label, bytes) in [
        ("under_limit_raw", 4 * 1024),
        ("oversized_raw", 4 * 1024 + 1),
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
                    outputs[0],
                    phase1b_exact_tool_result_json(bytes, "shell__exec"),
                    "{label} {protocol:?} preserves the complete raw ToolResult"
                );
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

    let snapshot = canonical_tool_result_snapshot(4 * 1024 + 1);
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
