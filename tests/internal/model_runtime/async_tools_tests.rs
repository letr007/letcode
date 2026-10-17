use super::{
    BindingFlavor, BindingIdentity, ContentPart, FailureKind, GenerationSupport, MessageRole,
    ModelEvent, ModelRequestInput, ProfileIdentity, ProtocolBindInput, ProtocolBinding, ProtocolId,
    ProtocolRegistry, RetryHint, RouteCapabilities, RouteIdentity, RuntimeConfig, TerminalStatus,
    ToolDefinition,
};
use serde_json::json;

fn astra_config(async_tools: &str) -> String {
    format!(
        r#"active_provider = "vendor"
[providers.vendor]
protocol = "responses"
default_model = "gpt-6-astra"
[providers.vendor.auth]
type = "none"
[providers.vendor.endpoints]
base_url = "https://example.invalid/v1"
[providers.vendor.models.gpt-6-astra]
strategy = "astra"
[providers.vendor.models.gpt-6-astra.capabilities]
tools = true
[providers.vendor.models.gpt-6-astra.generation]
async_tools = {async_tools}
"#
    )
}

fn responses_binding() -> std::sync::Arc<dyn ProtocolBinding> {
    let adapter = ProtocolRegistry::builtins()
        .lookup_str("responses")
        .expect("Responses adapter");
    adapter
        .bind(ProtocolBindInput::new(
            BindingIdentity::new(
                ProtocolId::new("responses").unwrap(),
                ProfileIdentity::new("vendor").unwrap(),
                RouteIdentity::new("vendor/model").unwrap(),
            ),
            "https://example.invalid/v1/responses",
            BindingFlavor::new("standard").unwrap(),
            Default::default(),
            Default::default(),
            RouteCapabilities {
                tools: true,
                ..Default::default()
            },
            GenerationSupport::default(),
        ))
        .unwrap()
}

#[test]
fn async_tool_config_requires_astra_responses_tools_and_unique_names() {
    let route = RuntimeConfig::from_toml(&astra_config("[\"lookup\", \"notify\"]"))
        .unwrap()
        .resolve(&ProtocolRegistry::builtins())
        .unwrap();
    assert_eq!(
        route
            .route("vendor", "gpt-6-astra")
            .unwrap()
            .async_tools
            .iter()
            .cloned()
            .collect::<Vec<_>>(),
        vec!["lookup", "notify"]
    );

    for tools in ["[\"lookup\", \"lookup\"]", "[\"\"]"] {
        let error = RuntimeConfig::from_toml(&astra_config(tools)).unwrap_err();
        assert!(error.to_string().contains("async_tools"));
    }

    let unsupported =
        astra_config("[\"lookup\"]").replace("strategy = \"astra\"", "strategy = \"default\"");
    let error = RuntimeConfig::from_toml(&unsupported).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("only supported for the Astra strategy")
    );
}

#[test]
fn async_tools_change_route_fingerprint() {
    let first = RuntimeConfig::from_toml(&astra_config("[\"lookup\"]"))
        .unwrap()
        .resolve(&ProtocolRegistry::builtins())
        .unwrap();
    let second = RuntimeConfig::from_toml(&astra_config("[\"notify\"]"))
        .unwrap()
        .resolve(&ProtocolRegistry::builtins())
        .unwrap();
    assert_ne!(first.fingerprint(), second.fingerprint());
}

#[test]
fn responses_wire_marks_only_declared_tools_and_keeps_replay_canonical() {
    let binding = responses_binding();
    let mut input = ModelRequestInput::new("model", Vec::new());
    let mut async_tool = ToolDefinition::new("lookup", "lookup", json!({"type":"object"}));
    async_tool.async_call = true;
    input.tools = vec![
        async_tool,
        ToolDefinition::new("plain", "plain", json!({"type":"object"})),
    ];
    input.messages.push(crate::model_runtime::ModelMessage {
        role: MessageRole::Assistant,
        content: vec![ContentPart::ToolCall {
            id: "call-1".into(),
            name: "lookup".into(),
            arguments: json!({}),
        }],
    });
    let body: serde_json::Value =
        serde_json::from_slice(&binding.prepare_request(&input).unwrap().body).unwrap();
    assert_eq!(body["tools"][0]["async"], true);
    assert!(body["tools"][1].get("async").is_none());
    assert!(body["input"][0].get("async").is_none());
}

#[test]
fn responses_decoder_accepts_async_function_call_field_without_changing_events() {
    let binding = responses_binding();
    let mut decoder = binding.new_decoder();
    let chunk = concat!(
        "event: response.output_item.added\n",
        "data: {\"type\":\"response.output_item.added\",\"item\":{\"type\":\"function_call\",\"id\":\"item-1\",\"call_id\":\"call-1\",\"name\":\"lookup\",\"async\":true}}\n\n",
        "event: response.function_call_arguments.done\n",
        "data: {\"type\":\"response.function_call_arguments.done\",\"item_id\":\"item-1\",\"arguments\":\"{}\"}\n\n",
        "event: response.output_item.done\n",
        "data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"function_call\",\"id\":\"item-1\",\"call_id\":\"call-1\",\"name\":\"lookup\",\"async\":true,\"arguments\":\"{}\"}}\n\n",
        "event: response.completed\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\"}}\n\n"
    );
    let events = decoder.push(chunk.as_bytes()).unwrap();
    assert!(
        events
            .iter()
            .any(|event| matches!(event, ModelEvent::ToolStarted { id, .. } if id == "call-1"))
    );
    assert!(
        events
            .iter()
            .any(|event| matches!(event, ModelEvent::ToolDone { id, .. } if id == "call-1"))
    );
    assert!(
        events
            .iter()
            .any(|event| matches!(event, ModelEvent::Terminal { .. }))
    );
}

#[test]
fn responses_403_misalignment_error_is_invalid_and_never_retryable() {
    let binding = responses_binding();
    let failure = binding
        .decode_http_error(
            403,
            br#"{"error":{"code":"misalignment_policy_violation","message":"blocked"}}"#,
            RetryHint::Retryable,
        )
        .expect("Responses error envelope");
    assert_eq!(failure.kind, FailureKind::InvalidRequest);
    assert_eq!(
        failure.code.as_deref(),
        Some("misalignment_policy_violation")
    );
    assert_eq!(failure.status, Some(403));
    assert_eq!(failure.retry_hint, RetryHint::Never);
}

#[test]
fn responses_http_error_uses_transport_status_for_existing_retry_classification() {
    let binding = responses_binding();
    let failure = binding
        .decode_http_error(
            503,
            br#"{"error":{"status":400,"code":"provider_error","message":"unavailable"}}"#,
            RetryHint::Retryable,
        )
        .expect("Responses error envelope");
    assert_eq!(failure.kind, FailureKind::Http);
    assert_eq!(failure.status, Some(503));
    assert_eq!(failure.retry_hint, RetryHint::Retryable);

    let failure = binding
        .decode_http_error(
            400,
            br#"{"error":{"code":"server_error","message":"bad request"}}"#,
            RetryHint::Retryable,
        )
        .expect("Responses error envelope");
    assert_eq!(failure.kind, FailureKind::Http);
    assert_eq!(failure.status, Some(400));
    assert_eq!(failure.retry_hint, RetryHint::Never);
}
