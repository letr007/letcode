use super::*;
use crate::model_runtime::{
    CacheIntent, CacheRetention, ControlSegment, GenerationSettings, MessageRole, ReasoningIntent,
    ReasoningSummary, RouteCapabilities, StablePrefixMetadata, ToolDefinition, Verbosity,
};

fn identity(flavor: &str) -> BindingIdentity {
    BindingIdentity::new(
        ProtocolId::new("responses").unwrap(),
        super::super::ProfileIdentity::new(flavor).unwrap(),
        super::super::RouteIdentity::new("route").unwrap(),
    )
}

fn binding(
    flavor: &str,
    capabilities: RouteCapabilities,
    generation: GenerationSupport,
) -> Arc<dyn ProtocolBinding> {
    ResponsesAdapter::new()
        .bind(ProtocolBindInput::new(
            identity(flavor),
            "https://example.invalid/v1/responses",
            BindingFlavor::new(flavor).unwrap(),
            super::super::ProfileSettings::default(),
            super::super::ProtocolSettings::default(),
            capabilities,
            generation,
        ))
        .unwrap()
}

fn anthropic_binding(settings: &str) -> Result<Arc<dyn ProtocolBinding>, ModelFailure> {
    let (capabilities, generation) = all_support();
    anthropic_binding_with(settings, capabilities, generation)
}

fn anthropic_binding_with(
    settings: &str,
    capabilities: RouteCapabilities,
    generation: GenerationSupport,
) -> Result<Arc<dyn ProtocolBinding>, ModelFailure> {
    let settings = toml::from_str::<toml::Value>(settings).unwrap();
    AnthropicAdapter::new().bind(ProtocolBindInput::new(
        BindingIdentity::new(
            ProtocolId::new("anthropic").unwrap(),
            super::super::ProfileIdentity::new("anthropic-profile").unwrap(),
            super::super::RouteIdentity::new("route").unwrap(),
        ),
        "https://example.invalid/v1/messages",
        BindingFlavor::new("standard").unwrap(),
        super::super::ProfileSettings::default(),
        super::super::ProtocolSettings::new(settings),
        capabilities,
        generation,
    ))
}

fn completions_binding(flavor: &str) -> Arc<dyn ProtocolBinding> {
    let (capabilities, generation) = all_support();
    completions_binding_with(flavor, capabilities, generation)
}

fn completions_binding_with(
    flavor: &str,
    capabilities: RouteCapabilities,
    generation: GenerationSupport,
) -> Arc<dyn ProtocolBinding> {
    CompletionsAdapter::new()
        .bind(ProtocolBindInput::new(
            BindingIdentity::new(
                ProtocolId::new("completions").unwrap(),
                super::super::ProfileIdentity::new(flavor).unwrap(),
                super::super::RouteIdentity::new("route").unwrap(),
            ),
            "https://example.invalid/v1/chat/completions",
            BindingFlavor::new(flavor).unwrap(),
            super::super::ProfileSettings::default(),
            super::super::ProtocolSettings::default(),
            capabilities,
            generation,
        ))
        .unwrap()
}

fn all_support() -> (RouteCapabilities, GenerationSupport) {
    (
        RouteCapabilities {
            tools: true,
            parallel_tool_calls: true,
            reasoning: true,
            input_images: true,
            tool_result_images: true,
            priority_service: true,
        },
        GenerationSupport {
            temperature: true,
            top_p: true,
            max_output_tokens: true,
            stop_sequences: true,
            reasoning: true,
            reasoning_summary: true,
            text_verbosity: true,
            parallel_tool_calls: true,
            priority_service: true,
            structured_output: None,
        },
    )
}

fn sse(event: &str, data: &str) -> Vec<u8> {
    format!("event: {event}\ndata: {data}\n\n").into_bytes()
}

#[test]
fn responses_websocket_frame_uses_response_create_and_previous_response_id() {
    let (capabilities, generation) = all_support();
    let binding = binding("standard", capabilities, generation);
    let input = ModelRequestInput::new("gpt-responses", Vec::new());
    let request = binding.prepare_request(&input).unwrap();
    let normal: Value = serde_json::from_slice(&request.body).unwrap();
    assert!(normal.get("previous_response_id").is_none());
    let first_frame: Value =
        serde_json::from_slice(&binding.websocket_frame(&request, None, None).unwrap()).unwrap();
    assert_eq!(first_frame["type"], "response.create");
    assert_eq!(first_frame["model"], "gpt-responses");
    assert!(first_frame.get("previous_response_id").is_none());
    let frame: Value = serde_json::from_slice(
        &binding
            .websocket_frame(&request, Some("resp-previous"), None)
            .unwrap(),
    )
    .unwrap();
    assert_eq!(frame["type"], "response.create");
    assert_eq!(frame["model"], "gpt-responses");
    assert_eq!(frame["previous_response_id"], "resp-previous");
    assert!(frame.get("stream_id").is_none());
    assert!(frame.get("generate").is_none());
}

#[test]
fn responses_websocket_frame_crops_only_input_and_preserves_request_controls() {
    let (capabilities, generation) = all_support();
    let binding = binding("standard", capabilities, generation);
    let mut input = ModelRequestInput::new(
        "gpt-responses",
        vec![
            ModelMessage::text(MessageRole::User, "first"),
            ModelMessage::text(MessageRole::User, "tool output"),
        ],
    );
    input.segments = vec![super::super::ControlSegment::system("system instructions")];
    input.segment_origins = vec!["system".into()];
    input.message_origins = vec!["user-1".into(), "tool-1".into()];
    input.tools = vec![super::super::ToolDefinition::new(
        "lookup",
        "look something up",
        serde_json::json!({"type":"object"}),
    )];
    let request = binding.prepare_request(&input).unwrap();
    let normal: Value = serde_json::from_slice(&request.body).unwrap();
    assert!(normal.get("previous_response_id").is_none());
    let frame: Value = serde_json::from_slice(
        &binding
            .websocket_frame(&request, Some("resp-1"), Some(2))
            .unwrap(),
    )
    .unwrap();
    assert_eq!(frame["type"], "response.create");
    assert_eq!(frame["instructions"], "system instructions");
    assert_eq!(frame["tools"][0]["name"], "lookup");
    assert_eq!(frame["previous_response_id"], "resp-1");
    assert_eq!(frame["input"].as_array().unwrap().len(), 1);
    assert_eq!(frame["input"][0]["content"][0]["text"], "tool output");
}

#[test]
fn responses_websocket_frame_has_no_implicit_openai_beta_header() {
    let (capabilities, generation) = all_support();
    let binding = binding("standard", capabilities, generation);
    let request = binding
        .prepare_request(&ModelRequestInput::new("gpt-responses", Vec::new()))
        .unwrap();
    assert!(!request.protocol_headers.contains_key("openai-beta"));
}

#[test]
fn responses_websocket_json_frames_reuse_responses_event_state_machine() {
    let (capabilities, generation) = all_support();
    let binding = binding("standard", capabilities, generation);
    let mut decoder = binding.new_websocket_decoder();
    let delta = decoder
        .push(br#"{"type":"response.output_text.delta","delta":"hello"}"#)
        .unwrap();
    assert_eq!(
        delta,
        vec![ModelEvent::TextDelta {
            text: "hello".into()
        }]
    );
    let terminal = decoder
        .push(br#"{"type":"response.completed","response":{"id":"resp-1","status":"completed"}}"#)
        .unwrap();
    assert!(terminal.iter().any(|event| matches!(
        event,
        ModelEvent::ResponseMetadata { response_id } if response_id == "resp-1"
    )));
    assert!(terminal.iter().any(|event| matches!(
        event,
        ModelEvent::Terminal {
            status: TerminalStatus::Completed
        }
    )));
    assert!(decoder.finish().is_ok());
}

#[test]
fn responses_websocket_error_envelope_accepts_numeric_status_and_retries_previous_response_not_found()
 {
    let (capabilities, generation) = all_support();
    let binding = binding("standard", capabilities, generation);
    let mut decoder = binding.new_websocket_decoder();
    let events = decoder
            .push(
                br#"{"type":"error","status":404,"error":{"code":"previous_response_not_found","message":"response was not found","param":"previous_response_id"}}"#,
            )
            .unwrap();
    let ModelEvent::Failure(failure) = &events[0] else {
        panic!("expected typed failure event");
    };
    assert_eq!(failure.status, Some(404));
    assert_eq!(failure.code.as_deref(), Some("previous_response_not_found"));
    assert_eq!(failure.retry_hint, RetryHint::Retryable);
    assert!(failure.detail().contains("response was not found"));
}

#[test]
fn responses_websocket_error_envelope_accepts_status_code_and_error_type() {
    let (capabilities, generation) = all_support();
    let binding = binding("standard", capabilities, generation);
    let mut decoder = binding.new_websocket_decoder();
    let events = decoder
            .push(
                br#"{"type":"error","status_code":429,"error":{"type":"rate_limit_error","message":"try later"}}"#,
            )
            .unwrap();
    let ModelEvent::Failure(failure) = &events[0] else {
        panic!("expected typed failure event");
    };
    assert_eq!(failure.status, Some(429));
    assert_eq!(failure.code.as_deref(), Some("rate_limit_error"));
    assert_eq!(failure.kind, FailureKind::RateLimited);
    assert_eq!(failure.retry_hint, RetryHint::Retryable);
    assert!(failure.detail().contains("try later"));
}

#[test]
fn responses_websocket_error_envelope_without_status_preserves_provider_detail() {
    let (capabilities, generation) = all_support();
    let binding = binding("standard", capabilities, generation);
    let mut decoder = binding.new_websocket_decoder();
    let events = decoder
            .push(
                br#"{"type":"error","error":{"code":"request_too_large","message":"serialized request exceeded upstream limit"}}"#,
            )
            .unwrap();
    let ModelEvent::Failure(failure) = &events[0] else {
        panic!("expected typed failure event");
    };
    assert_eq!(failure.status, None);
    assert_eq!(failure.code.as_deref(), Some("request_too_large"));
    assert_eq!(failure.retry_hint, RetryHint::Never);
    assert!(
        failure
            .detail()
            .contains("serialized request exceeded upstream limit")
    );
}

#[test]
fn responses_websocket_close_before_terminal_is_typed() {
    let (capabilities, generation) = all_support();
    let binding = binding("standard", capabilities, generation);
    let mut decoder = binding.new_websocket_decoder();
    decoder
        .push(br#"{"type":"response.output_text.delta","delta":"partial"}"#)
        .unwrap();
    let failure = decoder.finish().unwrap_err();
    assert_eq!(failure.phase, FailurePhase::Finish);
    assert_eq!(
        failure.code.as_deref(),
        Some("websocket_closed_before_terminal")
    );
}

#[test]
fn registry_uses_real_responses_and_validation_only_placeholders() {
    let registry = crate::model_runtime::ProtocolRegistry::builtins();
    let responses = registry.lookup_str("responses").unwrap();
    let (capabilities, generation) = all_support();
    let binding = responses
        .bind(ProtocolBindInput::new(
            identity("standard"),
            "https://example.invalid/responses",
            BindingFlavor::new("standard").unwrap(),
            super::super::ProfileSettings::default(),
            super::super::ProtocolSettings::default(),
            capabilities,
            generation,
        ))
        .unwrap();
    assert_eq!(binding.replay_scope(), ReplayScope::Profile);
    let completions_binding = registry
        .lookup_str("completions")
        .unwrap()
        .bind(ProtocolBindInput::new(
            BindingIdentity::new(
                ProtocolId::new("completions").unwrap(),
                super::super::ProfileIdentity::new("standard").unwrap(),
                super::super::RouteIdentity::new("route").unwrap(),
            ),
            "https://example.invalid/completions",
            BindingFlavor::new("standard").unwrap(),
            super::super::ProfileSettings::default(),
            super::super::ProtocolSettings::default(),
            RouteCapabilities::default(),
            GenerationSupport::default(),
        ))
        .unwrap();
    assert!(
        completions_binding
            .prepare_request(&ModelRequestInput::new("m", vec![]))
            .is_ok()
    );
    assert_eq!(completions_binding.replay_scope(), ReplayScope::None);
    let anthropic_binding = registry
        .lookup_str("anthropic")
        .unwrap()
        .bind(ProtocolBindInput::new(
            BindingIdentity::new(
                ProtocolId::new("anthropic").unwrap(),
                super::super::ProfileIdentity::new("standard").unwrap(),
                super::super::RouteIdentity::new("route").unwrap(),
            ),
            "https://example.invalid/messages",
            BindingFlavor::new("standard").unwrap(),
            super::super::ProfileSettings::default(),
            super::super::ProtocolSettings::default(),
            RouteCapabilities::default(),
            GenerationSupport::default(),
        ))
        .unwrap();
    assert!(
        anthropic_binding
            .prepare_request(&ModelRequestInput::new("m", vec![]))
            .is_ok()
    );
}

#[test]
fn bind_rejects_non_responses_flavors() {
    let (capabilities, generation) = all_support();
    let result = ResponsesAdapter::new().bind(ProtocolBindInput::new(
        identity("vendor"),
        "https://example.invalid/responses",
        BindingFlavor::new("vendor").unwrap(),
        super::super::ProfileSettings::default(),
        super::super::ProtocolSettings::default(),
        capabilities,
        generation,
    ));
    let error = match result {
        Ok(_) => panic!("unsupported flavor must be rejected"),
        Err(error) => error,
    };
    assert_eq!(error.code.as_deref(), Some("unsupported_responses_flavor"));
}

#[test]
fn anthropic_replay_filter_preserves_cache_boundary_and_origins() {
    let binding = anthropic_binding("").unwrap();
    let reasoning = ModelMessage {
        role: MessageRole::Assistant,
        content: vec![ContentPart::Reasoning {
            item_id: "rs_foreign".into(),
            text: "foreign reasoning".into(),
            replay: Some(OpaqueReplayState::new(
                "responses.reasoning",
                1,
                ReplayProducer::new(ReplayScope::Profile, &identity("standard")),
                serde_json::json!({"type":"reasoning", "id":"rs_foreign", "encrypted_content":"opaque"}),
            )),
        }],
    };
    let mut input = ModelRequestInput::new(
        "claude",
        vec![
            ModelMessage::text(MessageRole::User, "question"),
            reasoning.clone(),
            ModelMessage::text(MessageRole::Assistant, "answer"),
            reasoning,
            ModelMessage::text(MessageRole::User, "current"),
        ],
    );
    input.segments = vec![ControlSegment::system("rules")];
    input.segment_origins = vec!["system".into()];
    input.message_origins = [
        "question",
        "reasoning-1",
        "answer",
        "reasoning-2",
        "current",
    ]
    .map(str::to_owned)
    .to_vec();
    input.cache.enabled = true;
    for (segment_count, cached_message) in [
        (1, None),
        (2, Some(0)),
        (3, Some(0)),
        (4, Some(1)),
        (5, Some(1)),
    ] {
        input.cache.stable_prefix = Some(StablePrefixMetadata { segment_count });
        let prepared = binding.prepare_request(&input).unwrap();
        let body: Value = serde_json::from_slice(&prepared.body).unwrap();
        assert_eq!(body["messages"].as_array().unwrap().len(), 3);
        for (index, message) in body["messages"].as_array().unwrap().iter().enumerate() {
            assert_eq!(
                message["content"][0].get("cache_control").is_some(),
                cached_message == Some(index)
            );
        }
        assert_eq!(
            prepared.prompt_unit_origins,
            vec![
                vec!["system".to_string()],
                vec!["question".to_string()],
                vec!["answer".to_string()],
                vec!["current".to_string()],
            ]
        );
        let inspection = binding.inspect_prepared_request(&prepared, None).unwrap();
        assert_eq!(inspection.prompt_units.len(), 4);
        assert_eq!(
            inspection.prompt_units[3].semantic_segment_ids,
            vec!["current"]
        );
        assert!(inspection.cache.hint_serialized);
    }
}

#[test]
fn deepseek_completions_preserves_reasoning_with_content_and_omits_turnless_assistants() {
    let binding = completions_binding("deepseek");
    let mut input = ModelRequestInput::new(
        "deepseek",
        vec![
            ModelMessage::text(MessageRole::User, "question"),
            ModelMessage {
                role: MessageRole::Assistant,
                content: vec![
                    ContentPart::Reasoning {
                        item_id: "reasoning".into(),
                        text: "plain reasoning".into(),
                        replay: None,
                    },
                    ContentPart::Text("answer".into()),
                ],
            },
            ModelMessage {
                role: MessageRole::Assistant,
                content: vec![ContentPart::Reasoning {
                    item_id: "opaque".into(),
                    text: String::new(),
                    replay: OpaqueReplayState::from_anthropic_thinking_blocks_json(
                        r#"[{"type":"redacted_thinking","data":"opaque"}]"#,
                    ),
                }],
            },
            ModelMessage {
                role: MessageRole::Assistant,
                content: vec![ContentPart::Reasoning {
                    item_id: "orphan".into(),
                    text: "orphan reasoning".into(),
                    replay: None,
                }],
            },
            ModelMessage::text(MessageRole::User, "continue"),
        ],
    );
    input.message_origins = ["question", "reasoning", "opaque", "orphan", "continue"]
        .map(str::to_owned)
        .to_vec();
    let prepared = binding.prepare_request(&input).unwrap();
    let body: Value = serde_json::from_slice(&prepared.body).unwrap();
    assert_eq!(body["messages"].as_array().unwrap().len(), 3);
    assert_eq!(body["messages"][1]["content"], "answer");
    assert_eq!(body["messages"][1]["reasoning_content"], "plain reasoning");
    let wire = body.to_string();
    assert!(!wire.contains("opaque"));
    assert!(!wire.contains("orphan reasoning"));
    let inspection = binding.inspect_prepared_request(&prepared, None).unwrap();
    assert_eq!(
        inspection.prompt_units[2].semantic_segment_ids,
        vec!["continue"]
    );
}

#[test]
fn anthropic_adapter_request_headers_replay_cache_and_block_types() {
    let binding = anthropic_binding(
        r#"anthropic_betas = ["beta-a", "beta-b"]
anthropic_thinking = { mode = "adaptive" }"#,
    )
    .unwrap();
    let replay = OpaqueReplayState::new(
        "anthropic.thinking_blocks",
        1,
        ReplayProducer {
            scope: ReplayScope::Protocol,
            protocol_id: ProtocolId::new("anthropic").unwrap(),
            profile_identity: None,
            route_identity: None,
        },
        serde_json::json!([
            {"type":"thinking","thinking":"inspect","signature":"signed"},
            {"type":"redacted_thinking","data":"opaque"}
        ]),
    );
    let mut request = ModelRequestInput::new(
        "claude-model",
        vec![
            ModelMessage {
                role: MessageRole::Assistant,
                content: vec![
                    ContentPart::Reasoning {
                        item_id: "r1".into(),
                        text: "inspect".into(),
                        replay: Some(replay),
                    },
                    ContentPart::Text("answer".into()),
                    ContentPart::ToolCall {
                        id: "call-1".into(),
                        name: "search".into(),
                        arguments: serde_json::json!({"q":"rust"}),
                    },
                ],
            },
            ModelMessage::user_image("image/png", vec![1, 2, 3]),
        ],
    );
    request.segments = vec![
        ControlSegment::system("system"),
        ControlSegment {
            kind: ControlSegmentKind::Developer,
            text: "developer".into(),
        },
    ];
    request.tools = vec![ToolDefinition::new(
        "search",
        "Search",
        serde_json::json!({"type":"object"}),
    )];
    request.generation.max_output_tokens = Some(2048);
    request.cache = CacheIntent {
        enabled: true,
        namespace: None,
        retention: None,
        stable_prefix: Some(StablePrefixMetadata { segment_count: 3 }),
    };
    let prepared = binding.prepare_request(&request).unwrap();
    assert_eq!(prepared.protocol_headers["anthropic-version"], "2023-06-01");
    assert_eq!(prepared.protocol_headers["anthropic-beta"], "beta-a,beta-b");
    let body: Value = serde_json::from_slice(&prepared.body).unwrap();
    assert_eq!(body["system"][0]["type"], "text");
    assert!(body["system"][0].get("cache_control").is_none());
    assert_eq!(body["system"][1]["cache_control"]["type"], "ephemeral");
    assert_eq!(body["messages"][0]["content"][0]["type"], "thinking");
    assert_eq!(
        body["messages"][0]["content"][1]["type"],
        "redacted_thinking"
    );
    assert_eq!(body["messages"][0]["content"][2]["type"], "text");
    assert_eq!(body["messages"][0]["content"][3]["type"], "tool_use");
    assert_eq!(body["messages"][1]["content"][0]["type"], "image");
    assert_eq!(body["tools"][0]["name"], "search");
    assert_eq!(body["tools"][0]["strict"], true);
    assert_eq!(body["tools"][0]["cache_control"]["type"], "ephemeral");
    assert_eq!(body["thinking"]["type"], "adaptive");
    assert!(body.get("prompt_cache_key").is_none());
}

#[test]
fn anthropic_adapter_settings_and_capability_guards() {
    assert!(anthropic_binding(r#"anthropic_betas = ["dup", "dup"]"#).is_err());
    assert!(anthropic_binding(r#"anthropic_betas = [" spaced "]"#).is_err());
    assert!(
        anthropic_binding(r#"anthropic_thinking = { mode = "budget", budget_tokens = 1000 }"#)
            .is_err()
    );
    let binding_without_thinking = anthropic_binding("").unwrap();
    let mut reasoning = ModelRequestInput::new("claude", Vec::new());
    reasoning.generation.reasoning.enabled = true;
    assert!(
        binding_without_thinking
            .prepare_request(&reasoning)
            .is_err()
    );
    let disabled = anthropic_binding(r#"anthropic_thinking = { mode = "disabled" }"#).unwrap();
    assert!(disabled.prepare_request(&reasoning).is_err());

    let binding =
        anthropic_binding(r#"anthropic_thinking = { mode = "budget", budget_tokens = 1024 }"#)
            .unwrap();
    let mut request = ModelRequestInput::new("claude", Vec::new());
    request.generation.max_output_tokens = Some(1024);
    assert!(binding.prepare_request(&request).is_err());
    request.generation.max_output_tokens = Some(2048);
    request.generation.temperature = Some(0.2);
    request.generation.top_p = Some(0.8);
    request.generation.stop_sequences = vec!["END".into()];
    let body: Value =
        serde_json::from_slice(&binding.prepare_request(&request).unwrap().body).unwrap();
    assert_eq!(body["thinking"]["type"], "enabled");
    assert_eq!(body["thinking"]["budget_tokens"], 1024);
    assert!(body.get("output_config").is_none());
    assert_eq!(body["temperature"], 0.2);
    assert_eq!(body["top_p"], 0.8);
    assert_eq!(body["stop_sequences"][0], "END");
}

#[test]
fn anthropic_adapter_chunked_signed_tool_usage_ping_and_terminal() {
    let binding = anthropic_binding("").unwrap();
    let mut decoder = binding.new_decoder();
    let events = [
        sse(
            "message_start",
            r#"{"type":"message_start","message":{"usage":{"input_tokens":10,"cache_read_input_tokens":3}}}"#,
        ),
        sse("ping", r#"{"type":"ping"}"#),
        sse(
            "content_block_start",
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"in","signature":"sig"}}"#,
        ),
        sse(
            "content_block_delta",
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"spect"}}"#,
        ),
        sse(
            "content_block_delta",
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"ned"}}"#,
        ),
        sse(
            "content_block_stop",
            r#"{"type":"content_block_stop","index":0}"#,
        ),
        sse(
            "content_block_start",
            r#"{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"call-1","name":"search","input":{}}}"#,
        ),
        sse(
            "content_block_delta",
            r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"q\":"}}"#,
        ),
        sse(
            "content_block_delta",
            r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"\"rust\"}"}}"#,
        ),
        sse(
            "content_block_stop",
            r#"{"type":"content_block_stop","index":1}"#,
        ),
        sse(
            "message_delta",
            r#"{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":4,"cache_creation_input_tokens":2}}"#,
        ),
        sse("message_stop", r#"{"type":"message_stop"}"#),
    ];
    let mut output = Vec::new();
    for event in events {
        let midpoint = event.len() / 2;
        output.extend(decoder.push(&event[..midpoint]).unwrap());
        output.extend(decoder.push(&event[midpoint..]).unwrap());
    }
    output.extend(decoder.finish().unwrap());
    assert!(output.iter().any(|event| matches!(
        event,
        ModelEvent::ReasoningDone { text, replay: Some(replay), .. }
            if text == "inspect"
                && replay.payload[0]["signature"] == "signed"
    )));
    assert!(output.iter().any(|event| matches!(
        event,
        ModelEvent::ToolDone { id, arguments, .. }
            if id == "call-1" && arguments["q"] == "rust"
    )));
    assert!(output.iter().any(|event| matches!(
        event,
        ModelEvent::Usage {
            input_tokens: 10,
            output_tokens: 4,
            total_tokens: 14,
            ..
        }
    )));
    assert!(output.iter().any(|event| matches!(
        event,
        ModelEvent::Cache {
            read_tokens: 3,
            write_tokens: 2,
            ..
        }
    )));
    assert!(matches!(
        output.last(),
        Some(ModelEvent::Terminal {
            status: TerminalStatus::ToolUse
        })
    ));
}

#[test]
fn anthropic_replay_rejects_thinking_without_text() {
    let replay = OpaqueReplayState::new(
        "anthropic.thinking_blocks",
        1,
        ReplayProducer {
            scope: ReplayScope::Protocol,
            protocol_id: ProtocolId::new("anthropic").unwrap(),
            profile_identity: None,
            route_identity: None,
        },
        serde_json::json!([{"type":"thinking","thinking":"","signature":"encrypted"}]),
    );
    let input = ModelRequestInput::new(
        "claude-model",
        vec![ModelMessage {
            role: MessageRole::Assistant,
            content: vec![
                ContentPart::Reasoning {
                    item_id: "r1".into(),
                    text: String::new(),
                    replay: Some(replay),
                },
                ContentPart::Text("answer".into()),
            ],
        }],
    );
    assert!(
        anthropic_binding("")
            .unwrap()
            .prepare_request(&input)
            .is_err()
    );
}

#[test]
fn anthropic_decoder_skips_thinking_replay_without_text() {
    let thinking_stream = |signature: &str| {
        vec![
            sse("message_start", r#"{"type":"message_start","message":{}}"#),
            sse(
                "content_block_start",
                &format!(
                    r#"{{"type":"content_block_start","index":0,"content_block":{{"type":"thinking","thinking":"","signature":"{signature}"}}}}"#
                ),
            ),
            sse(
                "content_block_stop",
                r#"{"type":"content_block_stop","index":0}"#,
            ),
            sse(
                "message_delta",
                r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"}}"#,
            ),
            sse("message_stop", r#"{"type":"message_stop"}"#),
        ]
    };
    let replay_for = |signature: &str| {
        let mut decoder = AnthropicDecoder::new();
        let mut output = Vec::new();
        for event in thinking_stream(signature) {
            output.extend(decoder.push(&event).unwrap());
        }
        output.extend(decoder.finish().unwrap());
        output
            .into_iter()
            .find_map(|event| match event {
                ModelEvent::ReasoningDone { replay, .. } => Some(replay),
                _ => None,
            })
            .flatten()
    };
    assert!(replay_for("encrypted").is_none());
    assert!(replay_for("").is_none());
}

#[test]
fn anthropic_adapter_terminal_error_and_order_guards() {
    let binding = anthropic_binding("").unwrap();
    let mut missing = binding.new_decoder();
    missing
        .push(&sse(
            "message_start",
            r#"{"type":"message_start","message":{}}"#,
        ))
        .unwrap();
    assert!(missing.finish().is_err());

    let mut early_delta = binding.new_decoder();
    assert!(
        early_delta
            .push(&sse(
                "message_delta",
                r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"}}"#,
            ))
            .is_err()
    );

    let adaptive = anthropic_binding(r#"anthropic_thinking = { mode = "adaptive" }"#).unwrap();
    let mut adaptive_request = ModelRequestInput::new("claude", Vec::new());
    adaptive_request.generation.reasoning = ReasoningIntent {
        enabled: true,
        effort: Some(super::super::ReasoningEffort::High),
    };
    let body: Value = serde_json::from_slice(
        &adaptive
            .prepare_request(&adaptive_request)
            .expect("adaptive reasoning request")
            .body,
    )
    .unwrap();
    assert_eq!(body["output_config"]["effort"], "high");

    let mut usage_before_stop = binding.new_decoder();
    usage_before_stop
        .push(&sse(
            "message_start",
            r#"{"type":"message_start","message":{}}"#,
        ))
        .unwrap();
    usage_before_stop
            .push(&sse(
                "content_block_start",
                r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":"x"}}"#,
            ))
            .unwrap();
    assert!(
        usage_before_stop
            .push(&sse(
                "message_delta",
                r#"{"type":"message_delta","usage":{"output_tokens":1}}"#,
            ))
            .is_err()
    );

    let mut usage_closes_blocks = binding.new_decoder();
    usage_closes_blocks
        .push(&sse(
            "message_start",
            r#"{"type":"message_start","message":{}}"#,
        ))
        .unwrap();
    usage_closes_blocks
        .push(&sse(
            "message_delta",
            r#"{"type":"message_delta","usage":{"output_tokens":1}}"#,
        ))
        .unwrap();
    assert!(usage_closes_blocks
            .push(&sse(
                "content_block_start",
                r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":"late"}}"#,
            ))
            .is_err());

    let mut premature_stop = binding.new_decoder();
    premature_stop
        .push(&sse(
            "message_start",
            r#"{"type":"message_start","message":{}}"#,
        ))
        .unwrap();
    premature_stop
            .push(&sse(
                "content_block_start",
                r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":"x"}}"#,
            ))
            .unwrap();
    assert!(
        premature_stop
            .push(&sse(
                "message_delta",
                r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"}}"#,
            ))
            .is_err()
    );

    let mut post_terminal = binding.new_decoder();
    post_terminal
        .push(&sse(
            "message_start",
            r#"{"type":"message_start","message":{}}"#,
        ))
        .unwrap();
    post_terminal
        .push(&sse(
            "message_delta",
            r#"{"type":"message_delta","delta":{"stop_reason":"refusal"}}"#,
        ))
        .unwrap();
    post_terminal
        .push(&sse("message_stop", r#"{"type":"message_stop"}"#))
        .unwrap();
    assert!(
        post_terminal
            .push(&sse("ping", r#"{"type":"ping"}"#))
            .is_err()
    );

    let mut overloaded = binding.new_decoder();
    let events = overloaded
        .push(&sse(
            "error",
            r#"{"type":"error","error":{"type":"overloaded_error","message":"busy"}}"#,
        ))
        .unwrap();
    assert!(matches!(
        events.as_slice(),
        [ModelEvent::Failure(ModelFailure {
            kind: FailureKind::Http,
            retry_hint: RetryHint::Retryable,
            ..
        })]
    ));
}

#[test]
fn completions_adapter_standard_and_deepseek_request_goldens() {
    let standard = completions_binding("standard");
    let mut request = ModelRequestInput::new(
        "chat-model",
        vec![
            ModelMessage::text(MessageRole::Assistant, "assistant-history"),
            ModelMessage::user_image("image/png", vec![1, 2, 3]),
        ],
    );
    request.segments = vec![
        ControlSegment::system("system"),
        ControlSegment {
            kind: ControlSegmentKind::Developer,
            text: "developer".into(),
        },
    ];
    request.generation = GenerationSettings {
        max_output_tokens: Some(42),
        stop_sequences: vec!["END".into()],
        reasoning: ReasoningIntent {
            enabled: true,
            effort: Some(super::super::ReasoningEffort::High),
        },
        verbosity: Some(Verbosity::Low),
        parallel_tool_calls: Some(true),
        priority_service: Some(true),
        ..GenerationSettings::default()
    };
    request.cache = CacheIntent {
        enabled: true,
        namespace: Some("chat".into()),
        stable_prefix: Some(StablePrefixMetadata { segment_count: 2 }),
        retention: None,
    };
    let body: Value =
        serde_json::from_slice(&standard.prepare_request(&request).unwrap().body).unwrap();
    assert_eq!(body["messages"][0]["role"], "system");
    assert_eq!(body["messages"][1]["role"], "developer");
    assert_eq!(body["messages"][1]["content"], "developer");
    assert_eq!(body["messages"][2]["content"], "assistant-history");
    assert_eq!(body["messages"][3]["content"][0]["type"], "image_url");
    assert_eq!(body["max_completion_tokens"], 42);
    assert_eq!(body["stop"][0], "END");
    assert_eq!(body["stream_options"]["include_usage"], true);
    assert!(
        body["prompt_cache_key"]
            .as_str()
            .is_some_and(|key| key.starts_with("chat:standard:route:"))
    );
    assert_eq!(body["service_tier"], "priority");

    let deepseek = completions_binding("deepseek");
    let mut request = ModelRequestInput::new("deepseek", Vec::new());
    request.segments = vec![ControlSegment {
        kind: ControlSegmentKind::Developer,
        text: "developer".into(),
    }];
    request.messages = vec![ModelMessage {
        role: MessageRole::Assistant,
        content: vec![ContentPart::ToolCall {
            id: "call-1".into(),
            name: "search".into(),
            arguments: serde_json::json!({"q":"rust"}),
        }],
    }];
    request.generation.reasoning = ReasoningIntent {
        enabled: true,
        effort: Some(super::super::ReasoningEffort::None),
    };
    request.generation.max_output_tokens = Some(24);
    let body: Value =
        serde_json::from_slice(&deepseek.prepare_request(&request).unwrap().body).unwrap();
    assert_eq!(body["messages"][0]["role"], "system");
    assert_eq!(body["messages"][1]["reasoning_content"], "");
    assert_eq!(body["max_tokens"], 24);
    assert_eq!(body["thinking"]["type"], "disabled");
    assert!(body.get("max_completion_tokens").is_none());
    assert!(body.get("verbosity").is_none());
    assert!(body.get("prompt_cache_key").is_none());
    assert!(body.get("service_tier").is_none());

    let mut standard_none = ModelRequestInput::new("chat-model", Vec::new());
    standard_none.generation.reasoning = ReasoningIntent {
        enabled: true,
        effort: Some(super::super::ReasoningEffort::None),
    };
    let body: Value = serde_json::from_slice(
        &standard
            .prepare_request(&standard_none)
            .expect("standard none reasoning request")
            .body,
    )
    .unwrap();
    assert_eq!(body["reasoning_effort"], "none");
}

#[test]
fn completions_adapter_cache_key_ignores_volatile_messages_but_tracks_shape() {
    let binding = completions_binding("standard");
    let request = |message: &str, model: &str| {
        let mut request =
            ModelRequestInput::new(model, vec![ModelMessage::text(MessageRole::User, message)]);
        request.cache = CacheIntent {
            enabled: true,
            namespace: Some("chat".into()),
            retention: None,
            stable_prefix: Some(StablePrefixMetadata { segment_count: 1 }),
        };
        request
    };
    let key = |request: &ModelRequestInput| {
        let body: Value = serde_json::from_slice(
            &binding
                .prepare_request(request)
                .expect("cache request")
                .body,
        )
        .unwrap();
        body["prompt_cache_key"].as_str().unwrap().to_string()
    };
    let first = key(&request("first volatile message", "chat-model"));
    let second = key(&request("second volatile message", "chat-model"));
    let other_model = key(&request("first volatile message", "other-model"));
    assert_eq!(first, second);
    assert_ne!(first, other_model);
}

#[test]
fn completions_adapter_fragmented_tools_reasoning_usage_and_done() {
    let binding = completions_binding("standard");
    let mut decoder = binding.new_decoder();
    let chunks = [
        sse(
            "message",
            r#"{"choices":[{"index":0,"delta":{"reasoning_content":{"text":"plan"}},"finish_reason":null}]}"#,
        ),
        sse(
            "message",
            r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"q\":"}}]},"finish_reason":null}]}"#,
        ),
        sse(
            "message",
            r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call-1","function":{"name":"search","arguments":"\"rust\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":10,"completion_tokens":4,"total_tokens":14,"prompt_tokens_details":{"cached_tokens":3,"cache_write_tokens":2},"completion_tokens_details":{"reasoning_tokens":1}}}"#,
        ),
        b"data: [DONE]\n\n".to_vec(),
    ];
    let mut events = Vec::new();
    for chunk in chunks {
        let midpoint = chunk.len() / 2;
        events.extend(decoder.push(&chunk[..midpoint]).unwrap());
        events.extend(decoder.push(&chunk[midpoint..]).unwrap());
    }
    events.extend(decoder.finish().unwrap());
    assert!(matches!(events.last(), Some(ModelEvent::Terminal { .. })));
    let started = events
        .iter()
        .position(|event| matches!(event, ModelEvent::ToolStarted { id, .. } if id == "call-1"))
        .unwrap();
    let argument = events
        .iter()
        .position(
            |event| matches!(event, ModelEvent::ToolArgumentsDelta { id, .. } if id == "call-1"),
        )
        .unwrap();
    assert!(started < argument);
    let reconstructed = events
        .iter()
        .filter_map(|event| match event {
            ModelEvent::ToolArgumentsDelta { id, delta } if id == "call-1" => Some(delta.as_str()),
            _ => None,
        })
        .collect::<String>();
    assert_eq!(reconstructed, r#"{"q":"rust"}"#);
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, ModelEvent::ReasoningDone { .. }))
            .count(),
        1
    );
    assert!(events.iter().any(|event| matches!(
        event,
        ModelEvent::Usage {
            reasoning_tokens: Some(1),
            cached_input_tokens: Some(3),
            ..
        }
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        ModelEvent::Cache {
            read_tokens: 3,
            write_tokens: 2,
            ..
        }
    )));
    assert!(matches!(
        events.last(),
        Some(ModelEvent::Terminal {
            status: TerminalStatus::ToolUse
        })
    ));
}

#[test]
fn completions_adapter_rejects_done_and_invalid_terminal_contracts() {
    let binding = completions_binding("standard");
    let mut no_reason = binding.new_decoder();
    assert!(no_reason.push(b"data: [DONE]\n\n").is_err());

    let mut missing_done = binding.new_decoder();
    missing_done
            .push(
                b"data: {\"id\":\"chatcmpl-1\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            )
            .unwrap();
    assert!(missing_done.finish().is_err());

    for reason in ["length", "content_filter"] {
        let mut decoder = binding.new_decoder();
        let body = format!(
            "data: {{\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":\"{reason}\"}}]}}\n\ndata: [DONE]\n\n"
        );
        assert!(decoder.push(body.as_bytes()).is_err());
    }

    let mut post_done = binding.new_decoder();
    post_done
            .push(
                b"data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n",
            )
            .unwrap();
    assert!(post_done
            .push(
                b"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"late\"},\"finish_reason\":null}]}\n\n",
            )
            .is_err());

    let mut invalid_choice = binding.new_decoder();
    assert!(
        invalid_choice
            .push(&sse(
                "message",
                r#"{"choices":[{"index":1,"delta":{"content":"x"},"finish_reason":"stop"}]}"#,
            ))
            .is_err()
    );
}

fn structured_schema() -> super::super::StructuredOutputSchema {
    super::super::StructuredOutputSchema {
        name: "demo".into(),
        strict: true,
        schema: serde_json::json!({
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
            "additionalProperties": false
        }),
    }
}

fn structured_request() -> ModelRequestInput {
    let mut request = ModelRequestInput::new(
        "structured",
        vec![ModelMessage::text(MessageRole::User, "hello")],
    );
    request.generation.structured_output = Some(super::super::StructuredOutput::JsonSchema(
        structured_schema(),
    ));
    request
}

#[test]
fn structured_output_maps_to_each_protocol_contract() {
    let (capabilities, mut generation) = all_support();
    generation.structured_output = Some(super::super::StructuredOutputSupport::JsonSchema);
    let request = structured_request();

    let responses = binding("standard", capabilities.clone(), generation.clone());
    let body: Value =
        serde_json::from_slice(&responses.prepare_request(&request).unwrap().body).unwrap();
    assert_eq!(body["text"]["format"]["type"], "json_schema");
    assert_eq!(body["text"]["format"]["name"], "demo");
    assert_eq!(body["text"]["format"]["strict"], true);

    let completions =
        completions_binding_with("standard", capabilities.clone(), generation.clone());
    let body: Value =
        serde_json::from_slice(&completions.prepare_request(&request).unwrap().body).unwrap();
    assert_eq!(body["response_format"]["type"], "json_schema");
    assert_eq!(body["response_format"]["json_schema"]["name"], "demo");
    assert_eq!(body["response_format"]["json_schema"]["strict"], true);

    let anthropic = anthropic_binding_with("", capabilities.clone(), generation.clone()).unwrap();
    let body: Value =
        serde_json::from_slice(&anthropic.prepare_request(&request).unwrap().body).unwrap();
    assert_eq!(body["output_config"]["format"]["type"], "json_schema");
    assert!(body["output_config"]["format"].get("name").is_none());
    assert!(body["output_config"]["format"].get("strict").is_none());
}

#[test]
fn json_object_maps_to_text_and_response_format_and_anthropic_rejects_it() {
    let (capabilities, mut generation) = all_support();
    generation.structured_output = Some(super::super::StructuredOutputSupport::JsonObject);
    let mut request = ModelRequestInput::new(
        "structured",
        vec![ModelMessage::text(MessageRole::User, "hello")],
    );
    request.generation.structured_output = Some(super::super::StructuredOutput::JsonObject);

    let responses = binding("standard", capabilities.clone(), generation.clone());
    let body: Value =
        serde_json::from_slice(&responses.prepare_request(&request).unwrap().body).unwrap();
    assert_eq!(body["text"]["format"]["type"], "json_object");

    let completions =
        completions_binding_with("standard", capabilities.clone(), generation.clone());
    let body: Value =
        serde_json::from_slice(&completions.prepare_request(&request).unwrap().body).unwrap();
    assert_eq!(body["response_format"]["type"], "json_object");

    let anthropic = anthropic_binding_with("", capabilities, generation).unwrap();
    let error = anthropic.prepare_request(&request).unwrap_err();
    assert_eq!(error.code.as_deref(), Some("unsupported_request_field"));
}

#[test]
fn structured_output_requires_a_declared_capability() {
    let (capabilities, generation) = all_support();
    let request = structured_request();
    let prepared = [
        binding("standard", capabilities.clone(), generation.clone()).prepare_request(&request),
        completions_binding_with("standard", capabilities.clone(), generation.clone())
            .prepare_request(&request),
        anthropic_binding_with("", capabilities, generation)
            .unwrap()
            .prepare_request(&request),
    ];
    for result in prepared {
        let error = result.unwrap_err();
        assert_eq!(error.code.as_deref(), Some("unsupported_request_field"));
    }
}

#[test]
fn responses_websocket_frame_preserves_structured_output() {
    let (capabilities, mut generation) = all_support();
    generation.structured_output = Some(super::super::StructuredOutputSupport::JsonSchema);
    let binding = binding("standard", capabilities, generation);
    let request = binding.prepare_request(&structured_request()).unwrap();
    let frame: Value =
        serde_json::from_slice(&binding.websocket_frame(&request, None, None).unwrap()).unwrap();
    assert_eq!(frame["text"]["format"]["type"], "json_schema");
    assert_eq!(frame["text"]["format"]["name"], "demo");
}

#[test]
fn standard_request_golden_maps_controls_messages_generation_and_cache() {
    let (capabilities, generation) = all_support();
    let binding = binding("standard", capabilities, generation);
    let mut request = ModelRequestInput::new(
        "gpt-responses",
        vec![ModelMessage::text(MessageRole::User, "hello")],
    );
    request.segments = vec![
        ControlSegment::system("system one"),
        ControlSegment::system("system two"),
        ControlSegment {
            kind: ControlSegmentKind::Developer,
            text: "developer".into(),
        },
    ];
    request.generation = GenerationSettings {
        temperature: Some(0.2),
        top_p: Some(0.8),
        max_output_tokens: Some(42),
        stop_sequences: Vec::new(),
        reasoning: ReasoningIntent {
            enabled: true,
            effort: Some(super::super::ReasoningEffort::High),
        },
        summary: Some(ReasoningSummary::Concise),
        verbosity: Some(Verbosity::Low),
        parallel_tool_calls: Some(true),
        priority_service: Some(true),
        structured_output: None,
    };
    request.cache = CacheIntent {
        enabled: true,
        namespace: Some("golden".into()),
        retention: Some(CacheRetention::TwentyFourHours),
        stable_prefix: Some(StablePrefixMetadata { segment_count: 2 }),
    };
    let body: Value =
        serde_json::from_slice(&binding.prepare_request(&request).unwrap().body).unwrap();
    assert_eq!(body["instructions"], "system one\n\nsystem two");
    assert_eq!(body["input"][0]["role"], "developer");
    assert_eq!(body["service_tier"], "priority");
    assert_eq!(body["prompt_cache_key"], "golden:standard:route");
    assert_eq!(body["prompt_cache_retention"], "24h");
    assert_eq!(body["reasoning"]["effort"], "high");
}

#[test]
fn responses_request_uses_role_appropriate_text_blocks() {
    let (capabilities, generation) = all_support();
    let binding = binding("standard", capabilities, generation);
    let request = ModelRequestInput::new(
        "gpt-responses",
        vec![
            ModelMessage::text(MessageRole::User, "question"),
            ModelMessage::text(MessageRole::Assistant, "answer"),
        ],
    );

    let body: Value =
        serde_json::from_slice(&binding.prepare_request(&request).unwrap().body).unwrap();

    assert_eq!(body["input"][0]["content"][0]["type"], "input_text");
    assert_eq!(body["input"][1]["content"][0]["type"], "output_text");
}

#[test]
fn responses_tool_outputs_use_string_for_text_and_blocks_for_images() {
    let (capabilities, generation) = all_support();
    let binding = binding("standard", capabilities, generation);
    let request = ModelRequestInput {
        control: super::super::RequestControl::new("model"),
        segments: vec![],
        segment_origins: vec![],
        messages: vec![
            ModelMessage {
                role: MessageRole::Tool,
                content: vec![ContentPart::ToolResult {
                    id: "call-text".into(),
                    content: vec![ContentPart::Text(r#"{"ok":true}"#.into())],
                }],
            },
            ModelMessage {
                role: MessageRole::Tool,
                content: vec![ContentPart::ToolResult {
                    id: "call-image".into(),
                    content: vec![
                        ContentPart::Text(r#"{"ok":true}"#.into()),
                        ContentPart::Image {
                            media_type: "image/png".into(),
                            data: vec![1, 2, 3],
                        },
                    ],
                }],
            },
        ],
        message_origins: vec!["text-output".into(), "image-output".into()],
        tools: vec![],
        generation: GenerationSettings::default(),
        cache: CacheIntent::default(),
    };

    let body: Value =
        serde_json::from_slice(&binding.prepare_request(&request).unwrap().body).unwrap();
    assert_eq!(body["input"][0]["type"], "function_call_output");
    assert_eq!(body["input"][0]["output"], r#"{"ok":true}"#);
    assert_eq!(body["input"][1]["type"], "function_call_output");
    assert!(body["input"][1]["output"].is_array());
    assert_eq!(body["input"][1]["output"][0]["type"], "input_text");
    assert_eq!(body["input"][1]["output"][1]["type"], "input_image");
}

#[test]
fn completions_translates_tool_result_images_to_user_image_message() {
    let binding = completions_binding("deepseek");
    let request = ModelRequestInput {
        control: super::super::RequestControl::new("deepseek-flash"),
        segments: vec![],
        segment_origins: vec![],
        messages: vec![ModelMessage {
            role: MessageRole::Tool,
            content: vec![ContentPart::ToolResult {
                id: "call-image".into(),
                content: vec![
                    ContentPart::Text("image metadata".into()),
                    ContentPart::Image {
                        media_type: "image/png".into(),
                        data: vec![1, 2, 3],
                    },
                ],
            }],
        }],
        message_origins: vec!["tool-output".into()],
        tools: vec![],
        generation: GenerationSettings::default(),
        cache: CacheIntent::default(),
    };
    let body: Value =
        serde_json::from_slice(&binding.prepare_request(&request).unwrap().body).unwrap();
    assert_eq!(body["messages"][0]["role"], "tool");
    assert_eq!(body["messages"][0]["content"], "image metadata");
    assert_eq!(body["messages"][1]["role"], "user");
    assert_eq!(body["messages"][1]["content"][0]["type"], "image_url");
    assert_eq!(
        body["messages"][1]["content"][0]["image_url"]["url"],
        "data:image/png;base64,AQID"
    );
}

#[test]
fn completions_keeps_all_tool_messages_before_translated_images() {
    let binding = completions_binding("deepseek");
    let request = ModelRequestInput {
        control: super::super::RequestControl::new("deepseek-flash"),
        segments: vec![],
        segment_origins: vec![],
        messages: vec![
            ModelMessage {
                role: MessageRole::Assistant,
                content: vec![
                    ContentPart::ToolCall {
                        id: "call-1".into(),
                        name: "read".into(),
                        arguments: serde_json::json!({}),
                    },
                    ContentPart::ToolCall {
                        id: "call-2".into(),
                        name: "read".into(),
                        arguments: serde_json::json!({}),
                    },
                    ContentPart::ToolCall {
                        id: "call-3".into(),
                        name: "read".into(),
                        arguments: serde_json::json!({}),
                    },
                ],
            },
            ModelMessage {
                role: MessageRole::Tool,
                content: vec![ContentPart::ToolResult {
                    id: "call-1".into(),
                    content: vec![
                        ContentPart::Text("first".into()),
                        ContentPart::Image {
                            media_type: "image/png".into(),
                            data: vec![1],
                        },
                    ],
                }],
            },
            ModelMessage {
                role: MessageRole::Tool,
                content: vec![ContentPart::ToolResult {
                    id: "call-2".into(),
                    content: vec![
                        ContentPart::Text("second".into()),
                        ContentPart::Image {
                            media_type: "image/png".into(),
                            data: vec![2],
                        },
                    ],
                }],
            },
            ModelMessage {
                role: MessageRole::Tool,
                content: vec![ContentPart::ToolResult {
                    id: "call-3".into(),
                    content: vec![ContentPart::Text("third".into())],
                }],
            },
        ],
        message_origins: vec![
            "assistant".into(),
            "tool-1".into(),
            "tool-2".into(),
            "tool-3".into(),
        ],
        tools: vec![],
        generation: GenerationSettings::default(),
        cache: CacheIntent::default(),
    };
    let prepared = binding.prepare_request(&request).unwrap();
    let body: Value = serde_json::from_slice(&prepared.body).unwrap();
    binding.inspect_prepared_request(&prepared, None).unwrap();
    assert_eq!(body["messages"][0]["role"], "assistant");
    assert_eq!(body["messages"][1]["role"], "tool");
    assert_eq!(body["messages"][1]["content"], "first");
    assert_eq!(body["messages"][2]["role"], "tool");
    assert_eq!(body["messages"][2]["content"], "second");
    assert_eq!(body["messages"][3]["role"], "tool");
    assert_eq!(body["messages"][3]["content"], "third");
    assert_eq!(body["messages"][4]["role"], "user");
    assert_eq!(body["messages"][4]["content"].as_array().unwrap().len(), 2);
    assert_eq!(
        prepared.prompt_unit_origins,
        vec![
            vec!["assistant".to_string()],
            vec!["tool-1".to_string()],
            vec!["tool-2".to_string()],
            vec!["tool-3".to_string()],
            vec!["tool-1".to_string(), "tool-2".to_string()],
        ]
    );
}

#[test]
fn deepseek_replay_is_before_tool_call_and_images_and_tools_are_typed() {
    let (capabilities, generation) = all_support();
    let binding = binding("deepseek", capabilities, generation);
    let replay = OpaqueReplayState::new(
        "responses.reasoning.deepseek",
        1,
        ReplayProducer::new(ReplayScope::Profile, &identity("deepseek")),
        serde_json::json!({"type":"reasoning","id":"r1","encrypted_content":"opaque"}),
    );
    let request = ModelRequestInput {
        control: super::super::RequestControl::new("deepseek-reasoner"),
        segments: vec![],
        segment_origins: vec![],
        messages: vec![
            ModelMessage {
                role: MessageRole::Assistant,
                content: vec![
                    ContentPart::Reasoning {
                        item_id: "r1".into(),
                        text: "thought".into(),
                        replay: Some(replay),
                    },
                    ContentPart::ToolCall {
                        id: "call-1".into(),
                        name: "search".into(),
                        arguments: serde_json::json!({"q":"rust"}),
                    },
                ],
            },
            ModelMessage::user_image("image/png", vec![1, 2, 3]),
        ],
        message_origins: vec!["assistant".into(), "image".into()],
        tools: vec![ToolDefinition::new(
            "search",
            "Search",
            serde_json::json!({"type":"object"}),
        )],
        generation: GenerationSettings::default(),
        cache: CacheIntent::default(),
    };
    let body: Value =
        serde_json::from_slice(&binding.prepare_request(&request).unwrap().body).unwrap();
    assert_eq!(body["input"][0]["type"], "reasoning");
    assert_eq!(body["input"][1]["type"], "function_call");
    assert_eq!(body["input"][2]["content"][0]["type"], "input_image");
    assert_eq!(body["tools"][0]["type"], "function");
}

#[test]
fn decoder_handles_chunked_reasoning_tools_usage_cache_and_terminal() {
    let (capabilities, generation) = all_support();
    let binding = binding("standard", capabilities, generation);
    let mut decoder = binding.new_decoder();
    let chunks = [
        sse(
            "response.output_item.added",
            r#"{"type":"response.output_item.added","item":{"type":"reasoning","id":"r1"}}"#,
        ),
        sse(
            "response.reasoning_summary_text.delta",
            r#"{"type":"response.reasoning_summary_text.delta","item_id":"r1","summary_index":0,"delta":"think"}"#,
        ),
        sse(
            "response.reasoning_summary_text.done",
            r#"{"type":"response.reasoning_summary_text.done","item_id":"r1","summary_index":0,"text":"think"}"#,
        ),
        sse(
            "response.reasoning_summary_text.delta",
            r#"{"type":"response.reasoning_summary_text.delta","item_id":"r1","summary_index":1,"delta":"plan"}"#,
        ),
        sse(
            "response.reasoning_summary_text.done",
            r#"{"type":"response.reasoning_summary_text.done","item_id":"r1","summary_index":1,"text":"plan"}"#,
        ),
        sse(
            "response.output_item.done",
            r#"{"type":"response.output_item.done","item":{"type":"reasoning","id":"r1","encrypted_content":"enc","summary":[{"type":"summary_text","text":"think"},{"type":"summary_text","text":"plan"}],"content":[]}}"#,
        ),
        sse(
            "response.output_item.added",
            r#"{"type":"response.output_item.added","item":{"type":"function_call","id":"item-c1","call_id":"c1","name":"search"}}"#,
        ),
        sse(
            "response.function_call_arguments.delta",
            r#"{"type":"response.function_call_arguments.delta","item_id":"item-c1","delta":"{\"q\":"}"#,
        ),
        sse(
            "response.function_call_arguments.done",
            r#"{"type":"response.function_call_arguments.done","item_id":"item-c1","arguments":"{\"q\":\"rust\"}"}"#,
        ),
        sse(
            "response.output_item.done",
            r#"{"type":"response.output_item.done","item":{"type":"function_call","id":"item-c1","call_id":"c1","name":"search","arguments":"{\"q\":\"rust\"}"}}"#,
        ),
        sse(
            "response.completed",
            r#"{"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":10,"output_tokens":4,"input_tokens_details":{"cached_tokens":6}}}}"#,
        ),
    ];
    let mut events = Vec::new();
    for chunk in chunks {
        let midpoint = chunk.len() / 2;
        events.extend(decoder.push(&chunk[..midpoint]).unwrap());
        events.extend(decoder.push(&chunk[midpoint..]).unwrap());
    }
    events.extend(decoder.finish().unwrap());
    assert!(
        events.iter().any(
            |event| matches!(event, ModelEvent::ReasoningStarted { item_id } if item_id == "r1")
        )
    );
    assert!(events.iter().any(|event| matches!(
        event,
        ModelEvent::ReasoningDone {
            item_id,
            text,
            replay: Some(replay),
        } if item_id == "r1"
            && text == "think\n\nplan"
            && replay.payload["encrypted_content"] == "enc"
            && replay.payload["summary"].as_array().is_some()
    )));
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, ModelEvent::ReasoningDone { .. }))
            .count(),
        1
    );
    assert!(events.iter().any(|event| matches!(
        event,
        ModelEvent::ReasoningSummaryDelta {
            item_id,
            summary_index: 1,
            text,
        } if item_id == "r1" && text == "plan"
    )));
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, ModelEvent::ReasoningSummaryDone { .. }))
            .count(),
        2
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, ModelEvent::ToolDone { id, .. } if id == "c1"))
            .count(),
        1
    );
    assert!(events.iter().any(|event| matches!(
        event,
        ModelEvent::Cache {
            hit: true,
            read_tokens: 6,
            ..
        }
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        ModelEvent::Terminal {
            status: TerminalStatus::Completed
        }
    )));
}

#[test]
fn decoder_rejects_incomplete_or_contradictory_output_items() {
    let (capabilities, generation) = all_support();
    let binding = binding("standard", capabilities, generation);

    let mut incomplete_reasoning = binding.new_decoder();
    incomplete_reasoning
        .push(&sse(
            "response.output_item.added",
            r#"{"type":"response.output_item.added","item":{"type":"reasoning","id":"r1"}}"#,
        ))
        .unwrap();
    let error = incomplete_reasoning
        .push(&sse(
            "response.completed",
            r#"{"type":"response.completed","response":{"status":"completed"}}"#,
        ))
        .unwrap_err();
    assert!(error.detail().contains("incomplete output items"));

    let mut contradictory_tool = binding.new_decoder();
    contradictory_tool
            .push(&sse(
                "response.output_item.added",
                r#"{"type":"response.output_item.added","item":{"type":"function_call","id":"item-c1","call_id":"c1","name":"search"}}"#,
            ))
            .unwrap();
    contradictory_tool
            .push(&sse(
                "response.function_call_arguments.done",
                r#"{"type":"response.function_call_arguments.done","item_id":"item-c1","name":"search","arguments":"{\"q\":\"rust\"}"}"#,
            ))
            .unwrap();
    let error = contradictory_tool
            .push(&sse(
                "response.output_item.done",
                r#"{"type":"response.output_item.done","item":{"type":"function_call","id":"item-c1","call_id":"c1","name":"search","arguments":"{\"q\":\"go\"}"}}"#,
            ))
            .unwrap_err();
    assert!(error.detail().contains("arguments changed"));
}

#[test]
fn decoder_reports_malformed_failed_error_incomplete_and_missing_terminal() {
    let (capabilities, generation) = all_support();
    let binding = binding("standard", capabilities, generation);
    let mut malformed = binding.new_decoder();
    assert_eq!(
        malformed
            .push(b"data: {bad\n\n")
            .unwrap_err()
            .code
            .as_deref(),
        Some("malformed_json")
    );

    let mut failed = binding.new_decoder();
    let events = failed.push(&sse("response.failed", r#"{"type":"response.failed","response":{"error":{"code":"server_error","message":"nope"}}}"#)).unwrap();
    assert!(matches!(events.as_slice(), [ModelEvent::Failure(_)]));
    assert!(failed.finish().is_ok());

    let mut overloaded = binding.new_decoder();
    let events = overloaded.push(&sse("error", r#"{"type":"error","error":{"code":"server_is_overloaded","message":"Our servers are currently overloaded. Please try again later."}}"#)).unwrap();
    let ModelEvent::Failure(failure) = &events[0] else {
        panic!("expected overload failure");
    };
    assert_eq!(failure.code.as_deref(), Some("server_is_overloaded"));
    assert_eq!(failure.retry_hint, RetryHint::Retryable);
    assert!(overloaded.finish().is_ok());

    let mut incomplete = binding.new_decoder();
    let events = incomplete
        .push(&sse(
            "response.incomplete",
            r#"{"type":"response.incomplete","response":{"status":"incomplete"}}"#,
        ))
        .unwrap();
    assert!(matches!(
        events.as_slice(),
        [ModelEvent::Terminal {
            status: TerminalStatus::Incomplete
        }]
    ));

    let mut missing = binding.new_decoder();
    missing
        .push(&sse(
            "response.output_text.delta",
            r#"{"type":"response.output_text.delta","delta":"x"}"#,
        ))
        .unwrap();
    assert_eq!(
        missing.finish().unwrap_err().code.as_deref(),
        Some("missing_terminal_event")
    );
}
