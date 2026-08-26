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
        BuiltRequest::Anthropic(request) => serde_json::to_string(&request).expect("serialize"),
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

fn request_value(result: &BuildResult) -> Value {
    match &result.request {
        BuiltRequest::Responses(request) => serde_json::to_value(request),
        BuiltRequest::ResponsesCompatible(request)
        | BuiltRequest::CompletionsCompatible(request) => Ok(request.clone()),
        BuiltRequest::Completions(request) => serde_json::to_value(request),
        BuiltRequest::Anthropic(request) => Ok(request.clone()),
    }
    .expect("request serializes")
}

fn without_cache_fields(mut request: Value) -> Value {
    let fields = request
        .as_object_mut()
        .expect("serialized request is an object");
    fields.remove("prompt_cache_key");
    fields.remove("prompt_cache_retention");
    request
}

fn enabled_model_metadata() -> ModelRequestMetadata {
    metadata(8192)
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

fn assert_json_f64_close(value: &serde_json::Value, expected: f64) {
    let actual = value.as_f64().expect("value should be a number");
    assert!(
        (actual - expected).abs() < 0.000_001,
        "{actual} != {expected}"
    );
}

#[test]
fn deepseek_chat_compat_uses_legacy_roles_and_tokens_and_thinking() {
    let history = vec![HistoryItem::user("current")];
    let result = build_test_request(TestRequestBuilderInput {
        protocol: ApiProtocol::Completions,
        model_id: "deepseek-v4-flash",
        model: deepseek_metadata(),
        prelude: &[PromptMessage::developer("developer instructions")],
        history: &history,
        protected_start_index: 0,
        tools: &[],
        evidence: &[],
        history_adapter: None,
        context_view: None,
    })
    .expect("deepseek chat request builds");
    let request = request_value(&result);
    assert_eq!(request["messages"][0]["role"], "system");
    assert_eq!(request["max_tokens"], 32_768);
    assert!(request.get("max_completion_tokens").is_none());
    assert_eq!(request["thinking"]["type"], "enabled");
    assert_eq!(request["reasoning_effort"], "high");
}

#[test]
fn anthropic_messages_maps_tool_turn_and_replays_signed_thinking() {
    let reasoning_wire = serde_json::to_string(&vec![json!({
        "type": "thinking",
        "thinking": "inspect the file",
        "signature": "signed-state",
    })])
    .expect("reasoning wire serializes");
    let history = vec![
        HistoryItem::AssistantToolCalls {
            text: Some("I will inspect it".into()),
            reasoning_content: Some("inspect the file".into()),
            reasoning_wire: Some(reasoning_wire),
            calls: vec![HistoryToolCall {
                call_id: "call-1".into(),
                name: "read_file".into(),
                arguments_json: "{\"path\":\"src/main.rs\"}".into(),
            }],
        },
        HistoryItem::ToolOutput {
            call_id: "call-1".into(),
            output_json: "ok".into(),
            images: Vec::new(),
        },
        HistoryItem::user("continue"),
    ];
    let mut model = metadata(8_192);
    model.cache_control = true;
    model.anthropic_thinking = AnthropicThinkingConfig {
        mode: AnthropicThinkingMode::Budget,
        budget_tokens: Some(1_024),
    };
    let result = build_test_request(TestRequestBuilderInput {
        protocol: ApiProtocol::Anthropic,
        model_id: "claude-test",
        model,
        prelude: &[PromptMessage::system("stable instructions")],
        history: &history,
        protected_start_index: 0,
        tools: &[],
        evidence: &[],
        history_adapter: None,
        context_view: None,
    })
    .expect("anthropic request builds");
    let request = request_value(&result);

    assert_eq!(request["model"], "claude-test");
    assert_eq!(request["max_tokens"], 256);
    assert_eq!(request["stream"], true);
    assert_eq!(request["system"][0]["text"], "stable instructions");
    assert_eq!(request["thinking"]["type"], "enabled");
    assert_eq!(request["thinking"]["budget_tokens"], 1_024);

    let messages = request["messages"].as_array().expect("messages array");
    assert_eq!(messages.len(), 3);
    let assistant_content = messages[0]["content"].as_array().expect("assistant blocks");
    assert_eq!(assistant_content[0]["type"], "thinking");
    assert_eq!(assistant_content[0]["thinking"], "inspect the file");
    assert_eq!(assistant_content[0]["signature"], "signed-state");
    assert_eq!(assistant_content[1]["type"], "text");
    assert_eq!(assistant_content[2]["type"], "tool_use");
    assert_eq!(assistant_content[2]["id"], "call-1");
    assert_eq!(messages[1]["role"], "user");
    assert_eq!(messages[1]["content"][0]["type"], "tool_result");
    assert_eq!(messages[1]["content"][0]["tool_use_id"], "call-1");
}

#[test]
fn anthropic_logical_units_preserve_one_unit_per_prompt_segment() {
    let result = build_test_request(TestRequestBuilderInput {
        protocol: ApiProtocol::Anthropic,
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
        history_adapter: None,
        context_view: None,
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
        protocol: ApiProtocol::Anthropic,
        model_id: "claude-test",
        model,
        prelude: &[],
        history: &history,
        protected_start_index: 0,
        tools: &[],
        evidence: &[],
        history_adapter: None,
        context_view: None,
    })
    .expect("anthropic adaptive request builds");
    let request = request_value(&result);
    assert_eq!(request["thinking"], json!({ "type": "adaptive" }));
    assert_eq!(request["output_config"], json!({ "effort": "medium" }));
}

#[test]
fn deepseek_completions_preserves_skill_material_across_developer_messages() {
    let history = vec![HistoryItem::user("current")];
    let result = build_test_request(TestRequestBuilderInput {
        protocol: ApiProtocol::Completions,
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
        history_adapter: None,
        context_view: None,
    })
    .expect("deepseek chat request builds");
    let request = request_value(&result);
    let messages = request["messages"].as_array().expect("messages array");
    let system_texts: Vec<String> = messages
        .iter()
        .filter(|message| message["role"] == "system")
        .filter_map(|message| message["content"].as_str().map(String::from))
        .collect::<Vec<_>>();
    assert_eq!(
        system_texts.len(),
        3,
        "all developer preludes become system: {messages:?}"
    );
    assert!(
        system_texts
            .iter()
            .any(|text| text.contains("SKILL BODY MAGIC")),
        "skill material must be preserved in the request: {system_texts:?}"
    );
    assert!(
        system_texts
            .iter()
            .any(|text| text.contains("persona stable instructions"))
    );
    assert!(system_texts.iter().any(|text| text.contains("ctf-web")));
}

#[test]
fn deepseek_chat_compat_preserves_empty_reasoning_content_for_tool_turns() {
    let history = vec![
        HistoryItem::AssistantToolCalls {
            text: None,
            reasoning_content: None,
            reasoning_wire: None,
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
        protocol: ApiProtocol::Completions,
        model_id: "deepseek-v4-flash",
        model: deepseek_metadata(),
        prelude: &[],
        history: &history,
        protected_start_index: 0,
        tools: &[],
        evidence: &[],
        history_adapter: None,
        context_view: None,
    })
    .expect("deepseek tool replay builds");
    let request = request_value(&result);
    let assistant = request["messages"]
        .as_array()
        .expect("messages")
        .iter()
        .find(|message| message["role"] == "assistant")
        .expect("assistant tool message");
    assert_eq!(assistant["reasoning_content"], "");
}

#[test]
fn deepseek_responses_replays_reasoning_before_tool_call() {
    let history = vec![
        HistoryItem::AssistantToolCalls {
            text: Some("I will inspect it".into()),
            reasoning_content: Some("inspect the file".into()),
            reasoning_wire: None,
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
        protocol: ApiProtocol::Responses,
        model_id: "deepseek-v4-flash",
        model: deepseek_metadata(),
        prelude: &[],
        history: &history,
        protected_start_index: 0,
        tools: &[],
        evidence: &[],
        history_adapter: None,
        context_view: None,
    })
    .expect("deepseek responses replay builds");
    let request = request_value(&result);
    let input = request["input"].as_array().expect("responses input");
    assert!(input.iter().any(|item| {
        item["type"] == "reasoning"
            && item["content"][0]["type"] == "reasoning_text"
            && item["content"][0]["text"] == "inspect the file"
    }));
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
fn responses_serializes_tool_output_images_as_input_image_content() {
    let image = crate::user_content::UserImageAttachment::from_bytes(
        "pixel.png",
        "image/png",
        b"image-bytes",
    );
    let history = vec![
        HistoryItem::user("inspect"),
        HistoryItem::AssistantToolCalls {
            text: None,
            reasoning_content: None,
            reasoning_wire: None,
            calls: vec![HistoryToolCall {
                call_id: "call-image".into(),
                name: "fs__read".into(),
                arguments_json: r#"{"path":"pixel.png"}"#.into(),
            }],
        },
        HistoryItem::ToolOutput {
            call_id: "call-image".into(),
            output_json: r#"{"ok":true,"tool":"fs__read","data":{"kind":"image"}}"#.into(),
            images: vec![image.clone()],
        },
    ];

    let result = build_test_request(TestRequestBuilderInput {
        protocol: ApiProtocol::Responses,
        model_id: "gpt-test",
        model: metadata(16_384),
        prelude: &[],
        history: &history,
        protected_start_index: 0,
        tools: &[],
        evidence: &[],
        history_adapter: None,
        context_view: None,
    })
    .expect("responses request builds");
    let request = request_value(&result);
    let output = request["input"]
        .as_array()
        .expect("responses input")
        .iter()
        .find(|item| item["type"] == "function_call_output")
        .expect("function call output");
    let content = output["output"].as_array().expect("multimodal output");

    assert_eq!(content[0]["type"], "input_text");
    assert_eq!(content[1]["type"], "input_image");
    assert_eq!(content[1]["image_url"], image.data_url);
}

#[test]
fn completions_rejects_tool_output_images_without_text_fallback() {
    let history = vec![
        HistoryItem::user("inspect"),
        HistoryItem::AssistantToolCalls {
            text: None,
            reasoning_content: None,
            reasoning_wire: None,
            calls: vec![HistoryToolCall {
                call_id: "call-image".into(),
                name: "fs__read".into(),
                arguments_json: r#"{"path":"pixel.png"}"#.into(),
            }],
        },
        HistoryItem::ToolOutput {
            call_id: "call-image".into(),
            output_json: r#"{"ok":true}"#.into(),
            images: vec![crate::user_content::UserImageAttachment::from_bytes(
                "pixel.png",
                "image/png",
                b"image-bytes",
            )],
        },
    ];

    let error = build_test_request(TestRequestBuilderInput {
        protocol: ApiProtocol::Completions,
        model_id: "chat-test",
        model: metadata(16_384),
        prelude: &[],
        history: &history,
        protected_start_index: 0,
        tools: &[],
        evidence: &[],
        history_adapter: None,
        context_view: None,
    })
    .expect_err("chat completions must reject image tool outputs");

    assert!(
        error
            .to_string()
            .contains("does not support image content in tool outputs"),
        "{error:#}"
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
            reasoning_content: None,
            reasoning_wire: None,
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
            reasoning_content: None,
            reasoning_wire: None,
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
            images: Vec::new(),
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
                        provider: None,
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
                    ApiProtocol::Anthropic => unreachable!("pressure test does not use Anthropic"),
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
            provider: None,
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
