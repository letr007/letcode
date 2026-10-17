use super::*;

fn binding(protocol: &str, profile: &str, route: &str) -> BindingIdentity {
    BindingIdentity::new(
        ProtocolId::new(protocol).expect("protocol"),
        ProfileIdentity::new(profile).expect("profile"),
        RouteIdentity::new(route).expect("route"),
    )
}

fn bind_input(identity: BindingIdentity) -> ProtocolBindInput {
    ProtocolBindInput::new(
        identity,
        "https://example.invalid/v1/responses",
        BindingFlavor::new("chat").expect("flavor"),
        ProfileSettings::default(),
        ProtocolSettings::default(),
        RouteCapabilities::default(),
        GenerationSupport::default(),
    )
}

#[test]
fn default_route_capabilities_are_all_off() {
    let capabilities = RouteCapabilities::default();
    assert!(!capabilities.tools);
    assert!(!capabilities.parallel_tool_calls);
    assert!(!capabilities.reasoning);
    assert!(!capabilities.input_images);
    assert!(!capabilities.tool_result_images);
    assert!(!capabilities.priority_service);
}

#[test]
fn protocol_id_is_extensible_and_validated() {
    let custom = ProtocolId::new("vendor.protocol-v2").expect("custom id");
    assert_eq!(custom.as_str(), "vendor.protocol-v2");
    assert!(ProtocolId::new("").is_err());
    assert!(ProtocolId::new(" responses").is_err());
}

#[test]
fn builtins_register_responses_and_validation_only_adapters() {
    let registry = ProtocolRegistry::builtins();
    assert_eq!(registry.len(), 3);
    let responses = registry.lookup_str("responses").expect("built-in adapter");
    let response_binding = responses
        .bind(ProtocolBindInput::new(
            binding("responses", "profile", "route"),
            "https://example.invalid/v1/responses",
            BindingFlavor::new("standard").expect("standard flavor"),
            ProfileSettings::default(),
            ProtocolSettings::default(),
            RouteCapabilities::default(),
            GenerationSupport::default(),
        ))
        .expect("Responses adapter binds supported flavor");
    assert_eq!(response_binding.protocol_id().as_str(), "responses");
    let model = ModelRequestInput::new("model", Vec::new());
    assert!(response_binding.prepare_request(&model).is_ok());

    let completions = registry
        .lookup_str("completions")
        .expect("built-in adapter");
    let completion_binding = completions
        .bind(ProtocolBindInput::new(
            binding("completions", "profile", "route"),
            "https://example.invalid/v1/chat/completions",
            BindingFlavor::new("standard").expect("standard flavor"),
            ProfileSettings::default(),
            ProtocolSettings::default(),
            RouteCapabilities::default(),
            GenerationSupport::default(),
        ))
        .expect("Completions adapter binds supported flavor");
    assert_eq!(completion_binding.protocol_id().as_str(), "completions");
    assert!(completion_binding.prepare_request(&model).is_ok());

    let anthropic = registry.lookup_str("anthropic").expect("built-in adapter");
    let anthropic_binding = anthropic
        .bind(ProtocolBindInput::new(
            BindingIdentity::new(
                ProtocolId::new("anthropic").unwrap(),
                ProfileIdentity::new("profile").unwrap(),
                RouteIdentity::new("route").unwrap(),
            ),
            "https://example.invalid/v1/messages",
            BindingFlavor::new("standard").unwrap(),
            ProfileSettings::default(),
            ProtocolSettings::default(),
            RouteCapabilities::default(),
            GenerationSupport::default(),
        ))
        .expect("Anthropic adapter binds supported flavor");
    assert_eq!(anthropic_binding.protocol_id().as_str(), "anthropic");
    assert!(anthropic_binding.prepare_request(&model).is_ok());
}

#[test]
fn registry_rejects_duplicate_protocol_ids_and_returns_adapter() {
    let id = ProtocolId::new("duplicate").expect("id");
    let adapters = [
        Arc::new(adapters::PlaceholderAdapter::new(id.clone())) as Arc<dyn ProtocolAdapter>,
        Arc::new(adapters::PlaceholderAdapter::new(id)) as Arc<dyn ProtocolAdapter>,
    ];
    assert!(matches!(
        ProtocolRegistry::new(adapters),
        Err(RegistryError::DuplicateProtocol(_))
    ));

    let registry = ProtocolRegistry::builtins();
    assert_eq!(
        registry
            .lookup_str("responses")
            .expect("registered adapter")
            .protocol_id()
            .as_str(),
        "responses"
    );
}

#[test]
fn replay_scope_uses_producer_identity_and_target_binding() {
    let target = binding("responses", "profile", "route");
    let same_profile_other_route = binding("responses", "profile", "other-route");
    let other_profile = binding("responses", "other-profile", "route");
    let protocol_producer = ReplayProducer::new(ReplayScope::Protocol, &target);
    let profile_producer = ReplayProducer::new(ReplayScope::Profile, &target);
    let route_producer = ReplayProducer::new(ReplayScope::Route, &target);

    assert!(ReplayScope::Protocol.is_compatible_with(&protocol_producer, &other_profile));
    assert!(ReplayScope::Profile.is_compatible_with(&profile_producer, &same_profile_other_route));
    assert!(!ReplayScope::Profile.is_compatible_with(&profile_producer, &other_profile));
    assert!(ReplayScope::Route.is_compatible_with(&route_producer, &target));
    assert!(!ReplayScope::Route.is_compatible_with(&route_producer, &same_profile_other_route));
    assert!(!ReplayScope::None.is_compatible_with(&route_producer, &target));
}

#[test]
fn model_failure_display_omits_bounded_redacted_detail() {
    let failure = ModelFailure::new(FailurePhase::Transport, FailureKind::Http)
        .with_status(500)
        .with_code("server_error")
        .with_detail("Authorization: bearer-secret response body");
    assert!(!failure.detail().contains("bearer-secret"));
    assert!(!failure.to_string().contains("bearer-secret"));
    assert!(failure.to_string().contains("status 500"));
}

#[test]
fn model_failure_detail_preserves_unknown_provider_json_and_redacts_credentials() {
    let failure = ModelFailure::new(FailurePhase::Transport, FailureKind::Http)
            .with_detail_redacted(
                r#"{"error":{"unfamiliar":{"explanation":"unsupported input","request_fragment":{"model":"demo","prompt":"keep this diagnostic"}}},"apiKey":"response-secret","nested":[{"accessToken":"echoed-token"}],"clientSecret":"client-secret","proxyAuthorization":"proxy-secret","credential":{"value":"nested-secret"}}"#,
                &["response-secret", "echoed-token"],
            );
    assert!(failure.detail().contains("unsupported input"));
    assert!(failure.detail().contains("keep this diagnostic"));
    assert!(failure.detail().contains("\"unfamiliar\""));
    assert!(failure.detail().contains("\"apiKey\":\"[redacted]\""));
    assert!(failure.detail().contains("\"accessToken\":\"[redacted]\""));
    assert!(failure.detail().contains("\"clientSecret\":\"[redacted]\""));
    assert!(
        failure
            .detail()
            .contains("\"proxyAuthorization\":\"[redacted]\"")
    );
    assert!(failure.detail().contains("\"credential\":\"[redacted]\""));
    assert!(!failure.detail().contains("response-secret"));
    assert!(!failure.detail().contains("echoed-token"));
    assert!(!failure.detail().contains("client-secret"));
    assert!(!failure.detail().contains("proxy-secret"));
    assert!(!failure.detail().contains("nested-secret"));
}

#[test]
fn model_failure_detail_redacts_whitespace_credentials_without_hiding_diagnostics() {
    let failure = ModelFailure::new(FailurePhase::Transport, FailureKind::Http).with_detail(
            "Authorization: Bearer colon-secret colon request rejected\nAuthorization Bearer abc123 request rejected\napi-key: colon-key colon endpoint unavailable\napi key key-value endpoint unavailable\naccess token token-value scope denied\ntoken count 123\nsecret explanation remains useful",
        );
    assert!(!failure.detail().contains("colon-secret"));
    assert!(!failure.detail().contains("abc123"));
    assert!(!failure.detail().contains("colon-key"));
    assert!(!failure.detail().contains("key-value"));
    assert!(!failure.detail().contains("token-value"));
    assert!(failure.detail().contains("colon request rejected"));
    assert!(failure.detail().contains("request rejected"));
    assert!(failure.detail().contains("colon endpoint unavailable"));
    assert!(failure.detail().contains("endpoint unavailable"));
    assert!(failure.detail().contains("scope denied"));
    assert!(failure.detail().contains("token count 123"));
    assert!(
        failure
            .detail()
            .contains("secret explanation remains useful")
    );
}

#[test]
fn model_failure_detail_preserves_large_provider_diagnostics() {
    let diagnostic = "x".repeat(300 * 1024);
    let failure = ModelFailure::new(FailurePhase::Transport, FailureKind::Http)
        .with_detail(diagnostic.clone());
    assert_eq!(failure.detail(), diagnostic);
}

#[test]
fn model_failure_detail_preserves_multiline_text_and_html() {
    let failure = ModelFailure::new(FailurePhase::Transport, FailureKind::Http).with_detail(
        "<html>\n<body>invalid field: input[2]</body>\n</html>\napi-key=provider-secret",
    );
    assert!(
        failure
            .detail()
            .contains("<body>invalid field: input[2]</body>")
    );
    assert!(failure.detail().contains('\n'));
    assert!(!failure.detail().contains("provider-secret"));
}

fn runtime_config(provider: &str, auth: &str, provider_extra: &str, model_extra: &str) -> String {
    let model_extra = model_extra
        .replace(
            "[capabilities]",
            &format!("[providers.{provider}.models.model.capabilities]"),
        )
        .replace(
            "[generation]",
            &format!("[providers.{provider}.models.model.generation]"),
        )
        .replace(
            "[protocol_settings]",
            &format!("[providers.{provider}.models.model.protocol_settings]"),
        );
    format!(
        r#"active_provider = "{provider}"
[providers.{provider}]
protocol = "responses"
default_model = "model"
{provider_extra}
[providers.{provider}.auth]
{auth}
[providers.{provider}.endpoints]
base_url = "https://example.invalid/v1"
[providers.{provider}.models.model]
{model_extra}
"#
    )
}

#[test]
fn chat_completions_preserves_declared_tool_result_images() {
    let catalog = RuntimeConfig::from_toml(
        r#"
active_provider = "cpa"
[providers.cpa]
protocol = "completions"
default_model = "deepseek-flash"
[providers.cpa.auth]
type = "none"
[providers.cpa.endpoints]
base_url = "https://example.invalid/v1"
[providers.cpa.models."deepseek-flash"]
[providers.cpa.models."deepseek-flash".capabilities]
input_images = true
tool_result_images = true
"#,
    )
    .unwrap()
    .resolve(&ProtocolRegistry::builtins())
    .unwrap();
    let route = catalog.route("cpa", "deepseek-flash").unwrap();
    assert!(route.capabilities.input_images);
    assert!(route.capabilities.tool_result_images);
}

#[test]
fn structured_output_capability_is_resolved_from_model_configuration() {
    let resolve = |setting: &str| {
        RuntimeConfig::from_toml(&runtime_config("vendor", "type = \"none\"", "", setting))
            .unwrap()
            .resolve(&ProtocolRegistry::builtins())
            .unwrap()
    };
    let undeclared = resolve("");
    let schema = resolve(
        "[providers.vendor.models.model.capabilities.generation]\nstructured_output = \"json_schema\"",
    );
    let object = resolve(
        "[providers.vendor.models.model.capabilities.generation]\nstructured_output = \"json_object\"",
    );
    assert_eq!(
        undeclared
            .route("vendor", "model")
            .unwrap()
            .generation
            .structured_output,
        None
    );
    assert_eq!(
        schema
            .route("vendor", "model")
            .unwrap()
            .generation
            .structured_output,
        Some(StructuredOutputSupport::JsonSchema)
    );
    assert_eq!(
        object
            .route("vendor", "model")
            .unwrap()
            .generation
            .structured_output,
        Some(StructuredOutputSupport::JsonObject)
    );
    assert!(StructuredOutputSupport::JsonSchema.supports(StructuredOutputSupport::JsonObject));
    assert!(!StructuredOutputSupport::JsonObject.supports(StructuredOutputSupport::JsonSchema));
}

#[test]
fn websocket_transport_defaults_off_and_is_resolved_into_model_fingerprint() {
    let resolve = |setting: &str| {
        RuntimeConfig::from_toml(&runtime_config("vendor", "type = \"none\"", "", setting))
            .unwrap()
            .resolve(&ProtocolRegistry::builtins())
            .unwrap()
    };
    let default = resolve("");
    let disabled = resolve("[providers.vendor.models.model.transport]\nwebsocket = false");
    let enabled = resolve("[providers.vendor.models.model.transport]\nwebsocket = true");
    assert!(!default.route("vendor", "model").unwrap().websocket);
    assert!(!default.fingerprint().providers["vendor"].models["model"].websocket);
    assert!(enabled.route("vendor", "model").unwrap().websocket);
    assert!(enabled.fingerprint().providers["vendor"].models["model"].websocket);
    assert_eq!(default.fingerprint(), disabled.fingerprint());
    assert_ne!(default.fingerprint(), enabled.fingerprint());
}

#[test]
fn websocket_transport_is_model_scoped_with_a_shared_provider_client() {
    let config = runtime_config(
        "aggregator",
        "type = \"none\"",
        "[providers.aggregator.transport]\nconnect_timeout_secs = 17\nno_proxy_loopback = false",
        r#"[providers.aggregator.models.model.transport]
websocket = true
[providers.aggregator.models.other]
[providers.aggregator.models.disabled.transport]
websocket = false
[providers.aggregator.models.chat]
protocol = "completions"
"#,
    );
    let resolved = RuntimeConfig::from_toml(&config)
        .unwrap()
        .resolve(&ProtocolRegistry::builtins())
        .unwrap();
    let enabled = resolved.route("aggregator", "model").unwrap();
    assert!(enabled.websocket);
    for model in ["other", "disabled", "chat"] {
        let route = resolved.route("aggregator", model).unwrap();
        assert!(!route.websocket, "{model}");
        assert!(Arc::ptr_eq(&enabled.transport, &route.transport));
        assert_eq!(route.transport.connect_timeout().as_secs(), 17);
        assert!(!route.transport.no_proxy_loopback());
    }
}

#[test]
fn bearer_auth_is_applied_by_the_real_request_builder() {
    let transport = ProviderTransport::new(&RuntimeTransportConfig::default()).unwrap();
    let auth = RuntimeAuthConfig {
        scheme: AuthScheme::Bearer,
        name: None,
        credential: Some("bearer-secret".into()),
        credential_env: None,
    };
    let headers = BTreeMap::from([("x-provider-version".into(), "2026-09-01".into())]);
    let query = BTreeMap::from([("region".into(), "global".into())]);
    let request = transport
        .request(
            reqwest::Method::POST,
            "https://example.invalid/v1/responses",
            "vendor",
            &auth,
            &headers,
            &query,
        )
        .unwrap()
        .build()
        .unwrap();
    assert_eq!(request.headers()["authorization"], "Bearer bearer-secret");
    assert_eq!(request.headers()["x-provider-version"], "2026-09-01");
    assert_eq!(
        request
            .url()
            .query_pairs()
            .find(|(key, _)| key == "region")
            .unwrap()
            .1,
        "global"
    );
}

#[test]
fn named_header_auth_is_applied_by_the_real_request_builder() {
    let transport = ProviderTransport::new(&RuntimeTransportConfig::default()).unwrap();
    let auth = RuntimeAuthConfig {
        scheme: AuthScheme::Header,
        name: Some("x-vendor-key".into()),
        credential: Some("header-secret".into()),
        credential_env: None,
    };
    let headers = BTreeMap::from([("x-provider-version".into(), "2026-09-01".into())]);
    let query = BTreeMap::from([("region".into(), "global".into())]);
    let request = transport
        .request(
            reqwest::Method::POST,
            "https://example.invalid/v1/responses",
            "vendor",
            &auth,
            &headers,
            &query,
        )
        .unwrap()
        .build()
        .unwrap();
    assert_eq!(request.headers()["x-vendor-key"], "header-secret");
    assert_eq!(request.headers()["x-provider-version"], "2026-09-01");
    assert!(request.headers().get("authorization").is_none());
    assert_eq!(
        request
            .url()
            .query_pairs()
            .find(|(key, _)| key == "region")
            .unwrap()
            .1,
        "global"
    );
}

#[test]
fn named_query_auth_is_applied_by_the_real_request_builder() {
    let transport = ProviderTransport::new(&RuntimeTransportConfig::default()).unwrap();
    let auth = RuntimeAuthConfig {
        scheme: AuthScheme::Query,
        name: Some("api-token".into()),
        credential: Some("query-secret".into()),
        credential_env: None,
    };
    let headers = BTreeMap::from([("x-provider-version".into(), "2026-09-01".into())]);
    let query = BTreeMap::from([("region".into(), "global".into())]);
    let request = transport
        .request(
            reqwest::Method::GET,
            "https://example.invalid/v1/responses",
            "vendor",
            &auth,
            &headers,
            &query,
        )
        .unwrap()
        .build()
        .unwrap();
    assert_eq!(request.headers()["x-provider-version"], "2026-09-01");
    assert_eq!(
        request
            .url()
            .query_pairs()
            .find(|(key, _)| key == "api-token")
            .unwrap()
            .1,
        "query-secret"
    );
    assert_eq!(
        request
            .url()
            .query_pairs()
            .find(|(key, _)| key == "region")
            .unwrap()
            .1,
        "global"
    );
}

#[test]
fn jev_reviewer_configuration_requires_bearer_auth() {
    let rejected = runtime_config(
        "typesafe",
        "type = \"header\"\nname = \"x-api-key\"\ncredential = \"secret\"",
        "reviewer = \"jev\"",
        "",
    );
    let error = RuntimeConfig::from_toml(&rejected).unwrap_err();
    assert!(
        error.to_string().contains("providers.typesafe.auth"),
        "{error}"
    );
    assert!(error.to_string().contains("bearer"), "{error}");

    let accepted = runtime_config(
        "typesafe",
        "type = \"bearer\"\ncredential = \"secret\"",
        "reviewer = \"jev\"",
        "",
    );
    assert!(RuntimeConfig::from_toml(&accepted).is_ok());
}

#[test]
fn provider_and_endpoint_query_conflicts_are_rejected() {
    let value = runtime_config(
            "vendor",
            "type = \"bearer\"\ncredential = \"secret\"",
            "query = { version = \"provider\" }",
            "",
        )
        .replace(
            "base_url = \"https://example.invalid/v1\"",
            "base_url = \"https://example.invalid/v1\"\n[providers.vendor.endpoints.responses]\nquery = { version = \"endpoint\" }",
        );
    let error = RuntimeConfig::from_toml(&value).unwrap_err();
    assert!(error.to_string().contains("conflicts with provider query"));
}

#[test]
fn exact_generation_defaults_change_the_runtime_fingerprint() {
    let base = runtime_config(
        "vendor",
        "type = \"bearer\"\ncredential = \"secret\"",
        "",
        "[capabilities]\ngeneration = { max_output_tokens = true }\n[generation]\nmax_output_tokens = 128",
    );
    let changed = base.replace("max_output_tokens = 128", "max_output_tokens = 256");
    let first = RuntimeConfig::from_toml(&base)
        .unwrap()
        .resolve(&ProtocolRegistry::builtins())
        .unwrap();
    let second = RuntimeConfig::from_toml(&changed)
        .unwrap()
        .resolve(&ProtocolRegistry::builtins())
        .unwrap();
    assert_ne!(first.fingerprint(), second.fingerprint());
}

#[test]
fn retry_validation_rejects_zero_and_out_of_range_attempts() {
    for retry in [
        "max_attempts = 0\nmax_recovery_attempts = 1\ninitial_delay_secs = 1\nbackoff_multiplier = 2.0",
        "max_attempts = 9001\nmax_recovery_attempts = 1\ninitial_delay_secs = 1\nbackoff_multiplier = 2.0",
    ] {
        let value = runtime_config(
            "vendor",
            "type = \"bearer\"\ncredential = \"secret\"",
            "",
            "",
        )
        .replace(
            "default_model = \"model\"",
            &format!("default_model = \"model\"\n[providers.vendor.retry]\n{retry}"),
        );
        assert!(RuntimeConfig::from_toml(&value).is_err(), "{retry}");
    }
}

#[test]
fn malformed_and_unknown_protocol_settings_are_rejected() {
    let malformed = runtime_config(
        "vendor",
        "type = \"bearer\"\ncredential = \"secret\"",
        "",
        "protocol_settings = \"not-a-table\"",
    );
    assert!(RuntimeConfig::from_toml(&malformed).is_err());

    let unknown = runtime_config(
        "vendor",
        "type = \"bearer\"\ncredential = \"secret\"",
        "",
        "[protocol_settings]\nunknown = true",
    );
    let error = RuntimeConfig::from_toml(&unknown).unwrap_err();
    assert!(error.to_string().contains("unknown protocol setting"));
}

#[test]
fn credential_debug_output_is_redacted_but_request_value_is_preserved() {
    let auth = RuntimeAuthConfig {
        scheme: AuthScheme::Bearer,
        name: None,
        credential: Some("credential-secret".into()),
        credential_env: None,
    };
    assert!(!format!("{auth:?}").contains("credential-secret"));
    assert_eq!(auth.credential_value("vendor"), "credential-secret");
}

#[test]
fn credential_snapshot_is_immutable_after_environment_mutation() {
    let key = "LETcode_GROUP03A_SNAPSHOT";
    unsafe { std::env::set_var(key, "first-secret") };
    let value = runtime_config(
        "vendor",
        &format!("type = \"bearer\"\ncredential_env = \"{key}\""),
        "",
        "",
    );
    let config = RuntimeConfig::from_toml(&value).unwrap();
    let first = config.resolve(&ProtocolRegistry::builtins()).unwrap();
    unsafe { std::env::set_var(key, "second-secret") };
    let second = config.resolve(&ProtocolRegistry::builtins()).unwrap();
    assert_eq!(
        first
            .route("vendor", "model")
            .unwrap()
            .auth
            .credential
            .as_deref(),
        Some("first-secret")
    );
    assert_ne!(first.fingerprint(), second.fingerprint());
    unsafe { std::env::remove_var(key) };
}

#[test]
fn default_provider_environment_credential_is_snapshotted() {
    let key = "GROUP03A_DEFAULT_ENV_ONLY";
    unsafe { std::env::set_var(key, "default-first") };
    let value = runtime_config(
        "default-env-vendor",
        &format!("type = \"bearer\"\ncredential_env = \"{key}\""),
        "",
        "",
    );
    let config = RuntimeConfig::from_toml(&value).unwrap();
    let first = config.resolve(&ProtocolRegistry::builtins()).unwrap();
    unsafe { std::env::set_var(key, "default-second") };
    assert_eq!(
        first
            .route("default-env-vendor", "model")
            .unwrap()
            .auth
            .credential
            .as_deref(),
        Some("default-first")
    );
    unsafe { std::env::remove_var(key) };
}

#[tokio::test]
async fn transport_response_preserves_status_headers_and_chunks() {
    let mut headers = BTreeMap::new();
    headers.insert("retry-after".into(), "7".into());
    let mut response =
        TransportResponse::from_chunks(429, headers, vec![b"first".to_vec(), b"second".to_vec()]);
    assert_eq!(response.status, 429);
    assert_eq!(response.retry_hint(), RetryHint::RetryAfterSeconds(7));
    assert_eq!(response.next_chunk().await.unwrap().unwrap(), b"first");
    assert_eq!(response.next_chunk().await.unwrap().unwrap(), b"second");
    assert!(response.next_chunk().await.is_none());
}

mod integration_tests {
    use super::*;

    #[tokio::test]
    async fn provider_requests_identify_letcode_unless_a_header_overrides_it() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let url = format!("http://{address}/v1/responses");
        let server = tokio::spawn(async move {
            let mut seen = Vec::new();
            for _ in 0..2 {
                let (mut stream, _) = listener.accept().await.unwrap();
                let mut buffer = vec![0u8; 4096];
                let read = stream.read(&mut buffer).await.unwrap();
                seen.push(String::from_utf8_lossy(&buffer[..read]).to_lowercase());
                stream
                    .write_all(b"HTTP/1.1 200 OK\r\ncontent-length: 0\r\nconnection: close\r\n\r\n")
                    .await
                    .unwrap();
            }
            seen
        });

        let transport =
            ProviderTransport::new_for_endpoint(&RuntimeTransportConfig::default(), Some(&url))
                .unwrap();
        let auth = RuntimeAuthConfig {
            scheme: AuthScheme::None,
            name: None,
            credential: None,
            credential_env: None,
        };
        let sent = [
            BTreeMap::new(),
            BTreeMap::from([(
                "user-agent".to_string(),
                "codex_cli_rs/9.9.9 (Plan9; vax) vt100/1".to_string(),
            )]),
        ];
        for headers in &sent {
            let request = transport
                .request(
                    reqwest::Method::POST,
                    &url,
                    "vendor",
                    &auth,
                    headers,
                    &BTreeMap::new(),
                )
                .unwrap()
                .build()
                .unwrap();
            let _ = transport.client().execute(request).await;
        }

        let seen = server.await.unwrap();
        assert!(
            seen[0].contains(&format!("user-agent: {}", user_agent().to_lowercase())),
            "{}",
            seen[0]
        );
        assert!(
            seen[1].contains("user-agent: codex_cli_rs/9.9.9 (plan9; vax) vt100/1"),
            "{}",
            seen[1]
        );
    }

    #[tokio::test]
    async fn transport_errors_redact_request_metadata() {
        let transport = ProviderTransport::new_for_endpoint(
            &RuntimeTransportConfig {
                connect_timeout_secs: 1,
                ..RuntimeTransportConfig::default()
            },
            Some("http://127.0.0.1:1"),
        )
        .unwrap();
        let auth = RuntimeAuthConfig {
            scheme: AuthScheme::Query,
            name: Some("api-token".into()),
            credential: Some("secret with / reserved?&=+".into()),
            credential_env: None,
        };
        let request = transport
            .request(
                reqwest::Method::GET,
                "http://127.0.0.1:1/unreachable",
                "vendor",
                &auth,
                &BTreeMap::from([(String::from("x-configured"), String::from("header secret"))]),
                &BTreeMap::from([(String::from("trace"), String::from("query value"))]),
            )
            .unwrap();
        let failure = transport.send(request, &auth).await.unwrap_err();
        assert_eq!(failure.status, None);
        assert_eq!(failure.retry_hint, RetryHint::Retryable);
        for representation in [
            "secret with / reserved?&=+",
            "secret%20with%20%2F%20reserved%3F%26%3D%2B",
            "secret+with+%2F+reserved%3F%26%3D%2B",
            "secret%20with%20%2f%20reserved%3f%26%3d%2b",
            "secret+with+%2f+reserved%3f%26%3d%2b",
        ] {
            assert!(!failure.detail().contains(representation));
            assert!(!format!("{failure:?}").contains(representation));
            assert!(!failure.to_string().contains(representation));
        }
    }

    #[tokio::test]
    async fn local_chunked_transport_response_preserves_boundary_metadata() {
        use std::io::Write;
        use std::net::TcpListener;
        use std::thread;

        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0_u8; 2048];
            let _ = std::io::Read::read(&mut stream, &mut request);
            for part in [
                b"HTTP/1.1 429 Too Many Requests\r\nRetry-After: 7\r\nX-Test: boundary\r\nTransfer-Encoding: chunked\r\n\r\n".as_slice(),
                b"5\r\nfirst\r\n".as_slice(),
                b"6\r\nsecond\r\n".as_slice(),
                b"0\r\n\r\n".as_slice(),
            ] {
                stream.write_all(part).unwrap();
                stream.flush().unwrap();
                thread::yield_now();
            }
        });
        let transport = ProviderTransport::new_for_endpoint(
            &RuntimeTransportConfig::default(),
            Some(&format!("http://{address}")),
        )
        .unwrap();
        let auth = RuntimeAuthConfig {
            scheme: AuthScheme::Query,
            name: Some("key".into()),
            credential: Some("secret".into()),
            credential_env: None,
        };
        let request = transport
            .request(
                reqwest::Method::GET,
                &format!("http://{address}/model"),
                "local",
                &auth,
                &BTreeMap::new(),
                &BTreeMap::from([(String::from("trace"), String::from("one"))]),
            )
            .unwrap();
        let mut response = transport.send(request, &auth).await.unwrap();
        assert_eq!(response.status, 429);
        assert_eq!(
            response.headers.get("x-test"),
            Some(&String::from("boundary"))
        );
        assert_eq!(response.retry_hint(), RetryHint::RetryAfterSeconds(7));
        let mut body = Vec::new();
        while let Some(chunk) = response.next_chunk().await {
            body.extend_from_slice(&chunk.unwrap());
        }
        assert_eq!(body, b"firstsecond");
        assert!(!body.windows(3).any(|window| window == b"5\r\n"));
        assert!(!body.windows(3).any(|window| window == b"6\r\n"));
        server.join().unwrap();
    }

    #[tokio::test]
    async fn truncated_response_chunk_preserves_status_and_retry_hint() {
        use std::io::Write;
        use std::net::TcpListener;
        use std::thread;

        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0_u8; 2048];
            let _ = std::io::Read::read(&mut stream, &mut request);
            stream
                .write_all(
                    b"HTTP/1.1 429 Too Many Requests\r\nRetry-After: 7\r\nContent-Length: 10\r\nConnection: close\r\n\r\nfirst",
                )
                .unwrap();
            stream.flush().unwrap();
        });
        let transport = ProviderTransport::new_for_endpoint(
            &RuntimeTransportConfig::default(),
            Some(&format!("http://{address}")),
        )
        .unwrap();
        let auth = RuntimeAuthConfig {
            scheme: AuthScheme::None,
            name: None,
            credential: None,
            credential_env: None,
        };
        let request = transport
            .request(
                reqwest::Method::GET,
                &format!("http://{address}/model"),
                "local",
                &auth,
                &BTreeMap::new(),
                &BTreeMap::new(),
            )
            .unwrap();
        let mut response = transport.send(request, &auth).await.unwrap();
        assert_eq!(response.status, 429);
        let mut failure = None;
        while let Some(chunk) = response.next_chunk().await {
            if let Err(error) = chunk {
                failure = Some(error);
                break;
            }
        }
        let failure = failure.expect("truncated body must fail while reading chunks");
        assert_eq!(failure.status, Some(429));
        assert_eq!(failure.retry_hint, RetryHint::RetryAfterSeconds(7));
        server.join().unwrap();
    }
}
