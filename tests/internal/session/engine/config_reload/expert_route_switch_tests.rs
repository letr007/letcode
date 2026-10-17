use super::*;

fn provider(models: &[&str]) -> ProviderConfig {
    ProviderConfig {
        base_url: "http://127.0.0.1:9/v1".into(),
        jev_endpoint: None,
        auth_mode: crate::config::ProviderAuthMode::ApiKey,
        api_key: "key".into(),
        protocol: crate::config::ApiProtocol::Completions,
        default_model: models.first().copied().unwrap_or_default().into(),
        retry: None,
        reviewer: None,
        models: models
            .iter()
            .map(|model| {
                (
                    (*model).to_string(),
                    crate::config::ModelConfig {
                        display_name: None,
                        protocol: crate::config::ApiProtocol::Completions,
                        anthropic_thinking: Default::default(),
                        anthropic_betas: Vec::new(),
                        context_window: None,
                        effective_input_limit_tokens: None,
                        max_output_tokens: None,
                        supports_tools: false,
                        supports_input_images: false,
                        supports_tool_result_images: false,
                        supports_reasoning: false,
                        reasoning_effort: None,
                        reasoning_efforts: Vec::new(),
                        reasoning_summary: None,
                        text_verbosity: None,
                        temperature: None,
                        top_p: None,
                        prompt_cache: crate::config::PromptCacheConfig::default(),
                        parallel_tool_calls: false,
                    },
                )
            })
            .collect(),
    }
}

#[test]
fn default_route_only_reload_is_not_treated_as_noop() {
    assert!(reload_has_runtime_delta(
        true, true, true, true, true, false
    ));
    assert!(!reload_has_runtime_delta(
        true, true, true, true, true, true
    ));
}

#[test]
fn route_fingerprint_detects_websocket_transport_changes() {
    let config = |websocket: bool| {
        let websocket = if websocket {
            "[providers.vendor.models.model.transport]\nwebsocket = true\n"
        } else {
            ""
        };
        crate::model_runtime::RuntimeConfig::from_toml(&format!(
            r#"active_provider = "vendor"
[providers.vendor]
protocol = "responses"
default_model = "model"
[providers.vendor.auth]
type = "none"
[providers.vendor.endpoints]
base_url = "https://example.invalid/v1"
[providers.vendor.models.model]
{websocket}[providers.vendor.models.other]
"#
        ))
        .unwrap()
        .resolve(&crate::model_runtime::ProtocolRegistry::builtins())
        .unwrap()
        .fingerprint()
        .clone()
    };
    let disabled = config(false);
    let enabled = config(true);
    assert!(!route_runtime_fingerprint_eq(
        &disabled,
        &enabled,
        &ModelRoute::new("vendor", "model"),
    ));
    assert!(route_runtime_fingerprint_eq(
        &disabled,
        &enabled,
        &ModelRoute::new("vendor", "other"),
    ));
}

#[test]
fn unrelated_expert_reload_does_not_change_reviewer_policy() {
    let previous_routes = indexmap::IndexMap::from([(
        "explorer".into(),
        ModelRoute::new("primary", "old-explorer"),
    )]);
    let current_routes = indexmap::IndexMap::from([(
        "explorer".into(),
        ModelRoute::new("primary", "new-explorer"),
    )]);
    let allowed_models = indexmap::IndexMap::new();

    assert!(!reviewer_policy_changed(
        &previous_routes,
        &current_routes,
        &allowed_models,
        &allowed_models,
    ));
}
