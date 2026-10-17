use super::should_use_responses_websocket;
use crate::model_runtime::{ProtocolRegistry, RuntimeConfig};

#[test]
fn websocket_selection_uses_model_setting_and_responses_protocol() {
    for (protocol, enabled, model, model_override, expected) in [
        ("responses", true, "gpt-5.5", None, true),
        ("responses", false, "gpt-5.5", None, false),
        ("completions", true, "gpt-5.5", None, false),
        ("anthropic", true, "gpt-5.5", None, false),
        ("responses", true, "glm-5.3-flash", None, true),
        ("responses", false, "glm-5.3-flash", None, false),
        ("responses", true, "deepseek-v4-flash", None, true),
        ("responses", true, "alias", Some("gpt-5.5"), true),
        ("responses", false, "alias", Some("gpt-5.5"), false),
        ("responses", true, "gpt-5.5", Some("glm-5.3-flash"), true),
    ] {
        let override_setting = model_override
            .map(|value| format!("model_override = {value:?}"))
            .unwrap_or_default();
        let config = RuntimeConfig::from_toml(&format!(
            r#"active_provider = "aggregator"
[providers.aggregator]
protocol = "{protocol}"
default_model = "{model}"
[providers.aggregator.auth]
type = "none"
[providers.aggregator.endpoints]
base_url = "https://example.invalid/v1"
[providers.aggregator.models."{model}"]
{override_setting}
[providers.aggregator.models."{model}".transport]
websocket = {enabled}
"#
        ))
        .unwrap()
        .resolve(&ProtocolRegistry::builtins())
        .unwrap();
        let route = config.route("aggregator", model).unwrap();
        assert_eq!(
            should_use_responses_websocket(route),
            expected,
            "protocol={protocol}, websocket={enabled}, model={model}, wire_model={}",
            route.model_override,
        );
    }
}
