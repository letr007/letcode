use super::persistence::{
    acquire_config_lock, atomic_write_config, config_lock_path, open_config_lock_file, replace_file,
};
use super::*;
use crate::model_runtime::{GenerationSupport, RouteCapabilities};
use std::fs;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

fn config(provider: &str, model: &str, extra: &str) -> String {
    format!(
        r#"active_provider = "{provider}"

[providers.{provider}]
protocol = "responses"
default_model = "{model}"
flavor = "standard"

[providers.{provider}.auth]
type = "bearer"
credential = "secret-value"

[providers.{provider}.endpoints]
base_url = "https://example.invalid/v1"

[providers.{provider}.models."{model}"]
{extra}
"#
    )
    .replace(
        "[capabilities]",
        &format!("[providers.{provider}.models.\"{model}\".capabilities]"),
    )
    .replace(
        "[generation]",
        &format!("[providers.{provider}.models.\"{model}\".generation]"),
    )
    .replace(
        "[cache]",
        &format!("[providers.{provider}.models.\"{model}\".cache]"),
    )
    .replace(
        "[protocol_settings]",
        &format!("[providers.{provider}.models.\"{model}\".protocol_settings]"),
    )
}

#[test]
fn subagents_defaults_to_expert_and_preserves_expert_routing() {
    let expected = SubagentsConfig {
        mode: SubagentMode::Expert,
        max_depth: 3,
        max_concurrent: 16,
        max_pending_messages: 16,
        max_message_bytes: 16384,
        run_timeout_secs: 600,
    };
    for section in ["", "[subagents]", "[subagents]\nmode = \"expert\""] {
        let text = format!("{}\n{section}\n", config("openai", "model", ""));
        let loaded = AppConfig::load_from_str_at_path(Path::new("letcode.toml"), &text)
            .expect("expert configuration loads");
        assert_eq!(loaded.subagents, expected);
        assert_eq!(
            loaded.expert_route_for("explorer"),
            Some(ModelRoute::new("openai", "model"))
        );
        assert_eq!(
            loaded.global.tool_timeout_secs,
            Some(DEFAULT_TOOL_TIMEOUT_SECS)
        );
    }
}

#[test]
fn subagents_parses_swarm_and_numeric_overrides() {
    for (fields, expected) in [
        (
            "",
            SubagentsConfig {
                mode: SubagentMode::Swarm,
                ..SubagentsConfig::default()
            },
        ),
        (
            "max_depth = 1\nmax_concurrent = 2\nmax_pending_messages = 3\nmax_message_bytes = 4\nrun_timeout_secs = 5",
            SubagentsConfig {
                mode: SubagentMode::Swarm,
                max_depth: 1,
                max_concurrent: 2,
                max_pending_messages: 3,
                max_message_bytes: 4,
                run_timeout_secs: 5,
            },
        ),
    ] {
        let text = format!(
            "{}\n[subagents]\nmode = \"swarm\"\n{fields}\n",
            config("openai", "model", "")
        );
        let loaded = AppConfig::load_from_str_at_path(Path::new("letcode.toml"), &text)
            .expect("swarm configuration loads");
        assert_eq!(loaded.subagents, expected);
        let snapshot = serde_json::to_value(&loaded.subagents).expect("serialize configuration");
        assert_eq!(snapshot["mode"], "swarm");
        assert_eq!(snapshot["max_concurrent"], expected.max_concurrent);
        assert!(snapshot.get("max_descendants").is_none());
        assert_eq!(
            serde_json::from_value::<SubagentsConfig>(snapshot.clone()).unwrap(),
            expected
        );
        let mut legacy = snapshot;
        let limit = legacy
            .as_object_mut()
            .unwrap()
            .remove("max_concurrent")
            .unwrap();
        legacy["max_descendants"] = limit;
        assert_eq!(
            serde_json::from_value::<SubagentsConfig>(legacy).unwrap(),
            expected
        );
        let legacy_text = text.replace("max_concurrent", "max_descendants");
        let loaded =
            AppConfig::load_from_str_at_path(Path::new("letcode.toml"), &legacy_text).unwrap();
        assert_eq!(loaded.subagents, expected);
    }
}

#[test]
fn subagents_rejects_invalid_fields_and_limits() {
    for (fields, message) in [
        ("mode = \"invalid-mode\"", "invalid-mode"),
        ("unknown_limit = 1", "unknown field `unknown_limit`"),
        ("max_depth = 0", "subagents.max_depth"),
        ("max_concurrent = 0", "subagents.max_concurrent"),
        ("max_descendants = 0", "subagents.max_concurrent"),
        (
            "max_concurrent = 1\nmax_descendants = 2",
            "duplicate field `max_concurrent`",
        ),
        ("max_pending_messages = 0", "subagents.max_pending_messages"),
        ("max_message_bytes = 0", "subagents.max_message_bytes"),
        ("run_timeout_secs = 0", "subagents.run_timeout_secs"),
        ("max_depth = -1", "max_depth"),
        ("max_message_bytes = 1.5", "max_message_bytes"),
    ] {
        let text = format!("{}\n[subagents]\n{fields}\n", config("openai", "model", ""));
        let error = AppConfig::load_from_str_at_path(Path::new("letcode.toml"), &text)
            .expect_err("invalid subagent configuration is rejected");
        assert!(
            format!("{error:#}").contains(message),
            "{fields}: {error:#}"
        );
    }
}

#[test]
fn unspecified_protocol_defaults_to_completions_for_every_strategy() {
    for model in ["model", "deepseek-chat", "gpt-6-astra"] {
        let text = config("openai", model, "").replace("protocol = \"responses\"\n", "");
        let loaded = AppConfig::load_from_path(write_temp_config(text))
            .expect("config without protocol should load");
        assert_eq!(
            loaded
                .runtime_catalog
                .route("openai", model)
                .unwrap()
                .protocol_id
                .as_str(),
            "completions"
        );
    }
}

#[test]
fn permissions_jev_section_is_rejected_as_unknown() {
    let text = format!(
        "{}\n[permissions]\nmode = \"auto\"\n\n[permissions.jev]\nbase_url = \"https://api.typesafe.ai\"\nmodel = \"jev-latest\"\n",
        config("primary", "model-a", "")
    );
    let error = AppConfig::load_from_str_at_path(Path::new("letcode.toml"), &text)
        .expect_err("removed section must be rejected");
    assert!(
        format!("{error:#}").contains("unknown field `jev`"),
        "{error:#}"
    );
}

#[test]
fn reviewer_provider_marker_selects_the_jev_backend() {
    let text = format!(
        "{}\n[providers.typesafe]\nprotocol = \"responses\"\ndefault_model = \"jev-latest\"\nreviewer = \"jev\"\n\n[providers.typesafe.auth]\ntype = \"bearer\"\ncredential = \"typesafe-key\"\n\n[providers.typesafe.endpoints]\nbase_url = \"https://api.typesafe.ai\"\n\n[providers.typesafe.models.\"jev-latest\"]\n\n[agents.reviewer]\nprovider = \"typesafe\"\nmodel = \"jev-latest\"\n",
        config("primary", "model-a", "")
    );
    let loaded = AppConfig::load_from_str_at_path(Path::new("letcode.toml"), &text)
        .expect("config should load");
    let route = loaded
        .agents
        .reviewer
        .route
        .clone()
        .expect("reviewer route");
    let jev = jev_review_for_route(&loaded.providers, &route).expect("jev backend");

    assert_eq!(jev.provider, "typesafe");
    assert_eq!(jev.base_url, "https://api.typesafe.ai");
    assert_eq!(jev.endpoint, "https://api.typesafe.ai/v1/systemone");
    assert_eq!(jev.model, "jev-latest");
    assert_eq!(jev.credential, "typesafe-key");
    assert_eq!(jev.timeout_secs, DEFAULT_JEV_REVIEW_TIMEOUT_SECS);
}

#[test]
fn reviewer_provider_supports_custom_endpoint_and_alias() {
    let text = format!(
        r#"{}
[providers.newapi]
protocol = "responses"
default_model = "jev-latest"
reviewer = "jev"

[providers.newapi.auth]
type = "bearer"
credential = "newapi-key"

[providers.newapi.endpoints]
base_url = "https://fuck2.letr7.com/v1"

[providers.newapi.endpoints.jev]
path = "decisions"

[providers.newapi.models."jev-latest"]

[agents.reviewer]
provider = "newapi"
model = "jev-latest"
"#,
        config("primary", "model-a", "")
    );
    let loaded = AppConfig::load_from_str_at_path(Path::new("letcode.toml"), &text)
        .expect("config should load");
    let route = loaded
        .agents
        .reviewer
        .route
        .clone()
        .expect("reviewer route");
    let jev = jev_review_for_route(&loaded.providers, &route).expect("jev backend");

    assert_eq!(jev.provider, "newapi");
    assert_eq!(jev.base_url, "https://fuck2.letr7.com/v1");
    assert_eq!(jev.endpoint, "https://fuck2.letr7.com/v1/decisions");
    assert_eq!(jev.model, "jev-latest");
    assert_eq!(jev.credential, "newapi-key");

    // Alias `reviewer` and path with `/v1/decisions`
    let text_alias = format!(
        r#"{}
[providers.newapi]
protocol = "responses"
default_model = "jev-latest"
reviewer = "jev"

[providers.newapi.auth]
type = "bearer"
credential = "newapi-key"

[providers.newapi.endpoints]
base_url = "https://fuck2.letr7.com/v1"

[providers.newapi.endpoints.reviewer]
path = "/v1/decisions"

[providers.newapi.models."jev-latest"]

[agents.reviewer]
provider = "newapi"
model = "jev-latest"
"#,
        config("primary", "model-a", "")
    );
    let loaded_alias = AppConfig::load_from_str_at_path(Path::new("letcode.toml"), &text_alias)
        .expect("config should load");
    let route_alias = loaded_alias
        .agents
        .reviewer
        .route
        .clone()
        .expect("reviewer route");
    let jev_alias =
        jev_review_for_route(&loaded_alias.providers, &route_alias).expect("jev backend");
    assert_eq!(jev_alias.endpoint, "https://fuck2.letr7.com/v1/decisions");
}

#[test]
fn unmarked_reviewer_provider_keeps_the_chat_backend() {
    let text = format!(
        "{}\n[agents.reviewer]\nprovider = \"primary\"\nmodel = \"model-a\"\n",
        config("primary", "model-a", "")
    );
    let loaded = AppConfig::load_from_str_at_path(Path::new("letcode.toml"), &text)
        .expect("config should load");
    let route = loaded
        .agents
        .reviewer
        .route
        .clone()
        .expect("reviewer route");

    assert!(jev_review_for_route(&loaded.providers, &route).is_none());
}

#[test]
fn loads_new_schema_and_resolves_non_empty_catalog() {
    let path = write_temp_config(config(
        "standard",
        "standard-model",
        r#"display = "Standard"
model_override = "wire-standard"
[capabilities]
tools = true
generation = { max_output_tokens = true }
[generation]
max_output_tokens = 256
[cache]
enabled = true
namespace = "standard-cache"
"#,
    ));
    let loaded = AppConfig::load_from_path(&path).expect("new schema should load");
    assert!(!loaded.runtime_catalog.providers.is_empty());
    assert_eq!(loaded.runtime_catalog.active_provider, "standard");
    let standard = loaded
        .runtime_catalog
        .route("standard", "standard-model")
        .expect("standard route");
    assert_eq!(standard.model_override, "wire-standard");
    assert_eq!(standard.flavor, ProviderFlavor::Standard);
    assert!(standard.capabilities.tools);
    assert!(standard.generation.max_output_tokens);
    assert!(loaded.providers["standard"].models["standard-model"].supports_tools);
}

#[test]
fn astra_strategy_is_inferred_and_can_be_overridden_explicitly() {
    let inferred = AppConfig::load_from_path(write_temp_config(config(
        "openai",
        "gpt-6-astra",
        r#"[capabilities]
reasoning = true
generation = { reasoning = true }
"#,
    )))
    .expect("astra model should infer its strategy");
    let inferred_route = inferred
        .runtime_catalog
        .route("openai", "gpt-6-astra")
        .unwrap();
    assert_eq!(
        inferred_route.strategy,
        crate::model_runtime::strategy::ModelStrategyId::Astra
    );
    assert_eq!(inferred_route.protocol_id.as_str(), "responses");
    assert_eq!(inferred_route.flavor, ProviderFlavor::Standard);
    let metadata = inferred.providers["openai"].models["gpt-6-astra"].request_metadata();
    assert_eq!(metadata.reasoning_effort, Some(ModelReasoningEffort::Low));
    assert_eq!(
        metadata.selectable_reasoning_efforts(),
        vec![
            ModelReasoningEffort::Low,
            ModelReasoningEffort::Medium,
            ModelReasoningEffort::High,
            ModelReasoningEffort::Xhigh,
            ModelReasoningEffort::Max,
        ]
    );

    let overridden = AppConfig::load_from_path(write_temp_config(config(
        "openai",
        "gpt-6-astra",
        "strategy = \"default\"\nprotocol = \"completions\"\n",
    )))
    .expect("explicit default strategy should override inference");
    let overridden_route = overridden
        .runtime_catalog
        .route("openai", "gpt-6-astra")
        .unwrap();
    assert_eq!(
        overridden_route.strategy,
        crate::model_runtime::strategy::ModelStrategyId::Default
    );
    assert_eq!(overridden_route.protocol_id.as_str(), "completions");

    let explicit_efforts = AppConfig::load_from_path(write_temp_config(config(
        "openai",
        "gpt-6-astra",
        r#"[capabilities]
reasoning = true
generation = { reasoning = true }
[generation]
reasoning_efforts = ["high"]
"#,
    )))
    .expect("explicit Astra reasoning efforts should remain authoritative");
    let metadata = explicit_efforts.providers["openai"].models["gpt-6-astra"].request_metadata();
    assert_eq!(metadata.reasoning_effort, Some(ModelReasoningEffort::High));
    assert_eq!(metadata.reasoning_efforts, vec![ModelReasoningEffort::High]);
}

#[test]
fn astra_strategy_uses_the_protocol_default_and_rejects_other_flavors() {
    let inferred = r#"active_provider = "openai"
[providers.openai]
default_model = "gpt-6-astra"
[providers.openai.auth]
type = "bearer"
credential = "secret-value"
[providers.openai.endpoints]
base_url = "https://example.invalid/v1"
[providers.openai.models.gpt-6-astra]
"#;
    let loaded = AppConfig::load_from_path(write_temp_config(inferred))
        .expect("strategy defaults should supply protocol and flavor");
    let route = loaded
        .runtime_catalog
        .route("openai", "gpt-6-astra")
        .unwrap();
    assert_eq!(route.protocol_id.as_str(), "completions");
    assert_eq!(route.flavor, ProviderFlavor::Standard);

    let explicit_responses = config(
        "openai",
        "alias",
        "strategy = \"astra\"\nprotocol = \"responses\"\n",
    );
    let loaded = AppConfig::load_from_path(write_temp_config(explicit_responses))
        .expect("an explicit astra protocol should load");
    assert_eq!(
        loaded
            .runtime_catalog
            .route("openai", "alias")
            .unwrap()
            .protocol_id
            .as_str(),
        "responses"
    );

    let incompatible_flavor = config(
        "openai",
        "alias",
        "strategy = \"astra\"\nflavor = \"deepseek\"\n",
    );
    let error = AppConfig::load_from_path(write_temp_config(incompatible_flavor))
        .expect_err("incompatible astra flavor must fail");
    assert!(format!("{error:#}").contains("requires flavor 'standard'"));

    let missing_generation_support = config(
        "openai",
        "gpt-6-astra",
        "[capabilities]\nreasoning = true\n",
    );
    let error = AppConfig::load_from_path(write_temp_config(missing_generation_support))
        .expect_err("strategy-provided effort requires generation support");
    assert!(
        format!("{error:#}")
            .contains("requires capabilities.generation.reasoning when reasoning is enabled")
    );
}

#[test]
fn strategy_changes_runtime_fingerprint_and_unknown_values_fail() {
    let default = AppConfig::load_from_path(write_temp_config(config(
        "openai",
        "alias",
        "strategy = \"default\"\n",
    )))
    .unwrap();
    let astra = AppConfig::load_from_path(write_temp_config(config(
        "openai",
        "alias",
        "strategy = \"astra\"\n",
    )))
    .unwrap();
    assert_ne!(
        default.runtime_catalog.fingerprint(),
        astra.runtime_catalog.fingerprint()
    );

    let unknown = config("openai", "alias", "strategy = \"unknown\"\n");
    let error = AppConfig::load_from_path(write_temp_config(unknown))
        .expect_err("unknown strategy must fail");
    assert!(format!("{error:#}").contains("unknown variant"));
}

#[test]
fn model_websocket_settings_are_loaded_into_runtime_routes() {
    let text = config(
        "aggregator",
        "enabled",
        r#"[providers.aggregator.models.enabled.transport]
websocket = true
[providers.aggregator.models.other]
[providers.aggregator.models.disabled.transport]
websocket = false
"#,
    );
    let loaded = AppConfig::load_from_path(write_temp_config(text)).unwrap();
    for (model, expected) in [("enabled", true), ("other", false), ("disabled", false)] {
        assert_eq!(
            loaded
                .runtime_catalog
                .route("aggregator", model)
                .unwrap()
                .websocket,
            expected,
        );
    }
}

#[test]
fn provider_and_model_flavor_protocol_and_endpoint_overrides_are_deterministic() {
    let path = write_temp_config(
        r#"active_provider = "deepseek"

[providers.deepseek]
protocol = "responses"
default_model = "chat"
flavor = "deepseek"

[providers.deepseek.auth]
type = "header"
name = "x-vendor-key"
credential = "deep-secret"

[providers.deepseek.endpoints]
base_url = "https://deep.example/v1"
[providers.deepseek.endpoints.responses]
path = "/custom/responses"
query = { version = "2" }

[providers.deepseek.models.chat]

[providers.deepseek.models.override]
protocol = "completions"
flavor = "standard"
model_override = "wire-model"
"#,
    );
    let loaded = AppConfig::load_from_path(&path).expect("config should load");
    let chat = loaded.runtime_catalog.route("deepseek", "chat").unwrap();
    assert_eq!(chat.flavor, ProviderFlavor::Deepseek);
    assert_eq!(chat.endpoint, "https://deep.example/v1/custom/responses");
    assert_eq!(chat.query["version"], "2");
    assert_eq!(chat.auth.name.as_deref(), Some("x-vendor-key"));
    let override_route = loaded
        .runtime_catalog
        .route("deepseek", "override")
        .unwrap();
    assert_eq!(override_route.flavor, ProviderFlavor::Standard);
    assert_eq!(override_route.protocol_id.as_str(), "completions");
    assert_eq!(override_route.model_override, "wire-model");
}

#[test]
fn rejects_representative_legacy_provider_and_model_fields() {
    for legacy in [
        "api_key = \"legacy\"",
        "base_url = \"https://legacy.invalid\"",
        "auth_mode = \"bearer\"",
    ] {
        let error = AppConfig::load_from_path(write_temp_config(format!(
            "{}\n{}\n",
            config("openai", "model", ""),
            legacy
        )))
        .expect_err("legacy provider field must be rejected");
        assert!(
            error
                .chain()
                .any(|cause| cause.to_string().contains("unknown field"))
        );
    }
    for legacy in [
        "name = \"legacy\"",
        "supports_tools = true",
        "supports_reasoning = true",
        "prompt_cache = true",
        "cache_control = true",
        "anthropic_betas = [\"legacy\"]",
    ] {
        let error = AppConfig::load_from_path(write_temp_config(format!(
            "{}\n{}\n",
            config("openai", "model", ""),
            legacy
        )))
        .expect_err("legacy model field must be rejected");
        assert!(
            error
                .chain()
                .any(|cause| cause.to_string().contains("unknown field"))
        );
    }
}

#[test]
fn capabilities_default_with_tools_enabled_and_typed_cache_settings_validate() {
    let loaded = AppConfig::load_from_path(write_temp_config(config("openai", "model", "")))
        .expect("minimal new config should load");
    let route = loaded.runtime_catalog.route("openai", "model").unwrap();
    assert_eq!(
        route.capabilities,
        RouteCapabilities {
            tools: true,
            ..RouteCapabilities::default()
        }
    );
    assert_eq!(route.generation, GenerationSupport::default());
    assert!(!route.cache.enabled);

    let partial = AppConfig::load_from_path(write_temp_config(config(
        "openai",
        "model",
        "[capabilities]\ninput_images = true\n",
    )))
    .expect("partial capabilities table should load");
    let partial_route = partial.runtime_catalog.route("openai", "model").unwrap();
    assert!(partial_route.capabilities.tools);
    assert!(partial_route.capabilities.input_images);

    let invalid = config(
        "openai",
        "model",
        "[cache]\nenabled = false\nretention = \"24h\"\n",
    );
    let error = AppConfig::load_from_path(write_temp_config(invalid));
    assert!(error.is_err(), "disabled cache retention must fail");
}

#[test]
fn protocol_settings_are_typed_and_fingerprints_redact_and_detect_credentials() {
    let first = write_temp_config(config(
        "openai",
        "model",
        "protocol = \"anthropic\"\n[protocol_settings]\nanthropic_betas = [\"beta-one\"]\n",
    ));
    let first_config = AppConfig::load_from_path(&first).expect("first config");
    let debug = format!("{:?}", first_config.runtime_catalog);
    assert!(!debug.contains("secret-value"));
    assert_eq!(
        first_config
            .runtime_catalog
            .route("openai", "model")
            .unwrap()
            .protocol_settings
            .value["anthropic_betas"][0],
        toml::Value::String("beta-one".into())
    );

    let second = write_temp_config(config(
        "openai",
        "model",
        "protocol = \"anthropic\"\n[protocol_settings]\nanthropic_betas = [\"beta-one\"]\n",
    ));
    fs::write(
        &second,
        fs::read_to_string(&second)
            .unwrap()
            .replace("secret-value", "changed-secret"),
    )
    .unwrap();
    let second_config = AppConfig::load_from_path(&second).expect("second config");
    assert_ne!(
        first_config.runtime_catalog.fingerprint(),
        second_config.runtime_catalog.fingerprint()
    );
}

#[test]
fn projected_provider_uses_resolved_credential_snapshot() {
    let _guard = lock_env();
    let env_name = "LETCODE_CONFIG_SNAPSHOT_TEST_KEY";
    unsafe { env::set_var(env_name, "first-secret") };
    let path = write_temp_config(config("openai", "model", "").replace(
        "credential = \"secret-value\"",
        &format!("credential_env = {env_name:?}"),
    ));
    let loaded = AppConfig::load_from_path(&path).expect("environment credential resolves");
    unsafe { env::set_var(env_name, "second-secret") };

    assert_eq!(loaded.providers["openai"].api_key, "first-secret");
    assert_eq!(
        loaded.runtime_catalog.providers["openai"]
            .auth
            .credential
            .as_deref(),
        Some("first-secret")
    );
    unsafe { env::remove_var(env_name) };
}

#[test]
fn missing_config_example_is_valid() {
    let path = env::temp_dir().join(format!(
        "letcode-missing-config-{}.toml",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let message = missing_config_message(&path);
    let example = message
        .split_once("Create it with at least:\n\n")
        .map(|(_, example)| example)
        .expect("missing config message contains example");
    let example = example.replace("YOUR_API_KEY", "test-key");
    let example_path = write_temp_config(example);

    AppConfig::load_from_path(example_path).expect("missing-config example should validate");
}

#[test]
fn persists_new_schema_without_overwriting_external_changes() {
    let path = write_temp_config(format!(
        "{}\n[agents.explorer]\nmodel = \"model\"\n",
        config("openai", "model", "")
    ));
    let before = fs::read_to_string(&path).unwrap();
    let metadata = fs::metadata(&path).unwrap();
    fs::write(&path, "external edit\n").unwrap();
    let error = atomic_write_config(&path, &before, &metadata, b"our update\n")
        .expect_err("external edit must win");
    assert!(error.to_string().contains("changed while updating"));
    assert_eq!(fs::read_to_string(path).unwrap(), "external edit\n");
}

#[test]
fn replace_file_overwrites_existing_destination() {
    let destination = write_temp_config("old\n");
    let source = destination.with_extension("replacement");
    fs::write(&source, "new\n").unwrap();
    replace_file(&source, &destination).unwrap();
    assert_eq!(fs::read_to_string(destination).unwrap(), "new\n");
    assert!(!source.exists());
}

mod integration_tests {
    use super::*;

    #[cfg(unix)]
    #[test]
    fn sibling_lock_remains_held_after_config_target_replacement() {
        use std::os::fd::AsRawFd;
        let path = write_temp_config("original\n");
        let target = fs::canonicalize(&path).unwrap();
        let lock_path = config_lock_path(&target).unwrap();
        let lock = acquire_config_lock(&target).unwrap();
        let replacement = target.with_file_name("config-lock-replacement");
        fs::write(&replacement, "replacement\n").unwrap();
        fs::rename(&replacement, &target).unwrap();
        let second = open_config_lock_file(&lock_path).unwrap();
        let result = unsafe { libc::flock(second.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        assert_eq!(result, -1);
        drop(second);
        drop(lock);
        acquire_config_lock(&target).unwrap();
    }

    fn assert_config_read_waits_for_writer(path: PathBuf, target: PathBuf) {
        use std::sync::mpsc;
        use std::time::Duration;

        let lock = acquire_config_lock(&target).unwrap();
        let original = target.with_extension("original");
        fs::rename(&target, &original).unwrap();
        let (ready_tx, ready_rx) = mpsc::channel();
        let (tx, rx) = mpsc::channel();
        let reader = std::thread::spawn(move || {
            ready_tx.send(()).unwrap();
            let result = AppConfig::load_from_path(&path);
            let _ = tx.send(result);
        });
        ready_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        let before_unlock = rx.recv_timeout(Duration::from_millis(100));
        fs::rename(&original, &target).unwrap();
        drop(lock);
        assert!(matches!(
            before_unlock,
            Err(mpsc::RecvTimeoutError::Timeout)
        ));
        rx.recv_timeout(Duration::from_secs(5)).unwrap().unwrap();
        reader.join().unwrap();
    }

    #[test]
    fn config_read_waits_for_replacement_before_checking_the_target() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("letcode.toml");
        fs::write(&path, config("openai", "model", "")).unwrap();
        let target = fs::canonicalize(&path).unwrap();
        assert_config_read_waits_for_writer(path, target);
    }

    #[cfg(unix)]
    #[test]
    fn config_symlink_read_waits_for_the_target_writer() {
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir().unwrap();
        let target_dir = directory.path().join("target");
        let alias_dir = directory.path().join("alias");
        fs::create_dir(&target_dir).unwrap();
        fs::create_dir(&alias_dir).unwrap();
        let path = alias_dir.join("letcode.toml");
        let target = target_dir.join("config.toml");
        fs::write(&target, config("openai", "model", "")).unwrap();
        symlink("../target/config.toml", alias_dir.join("next.toml")).unwrap();
        symlink("next.toml", &path).unwrap();
        let target = fs::canonicalize(&target).unwrap();
        assert_config_read_waits_for_writer(path.clone(), target);
        assert!(fs::symlink_metadata(path).unwrap().file_type().is_symlink());
    }

    #[test]
    fn config_save_waits_for_an_existing_writer_before_opening_the_target() {
        use std::sync::mpsc;
        use std::time::Duration;

        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("letcode.toml");
        let text = config("openai", "model", "");
        fs::write(&path, &text).unwrap();
        let mut document = text.parse::<toml_edit::DocumentMut>().unwrap();
        document["global"]["max_iterations"] = toml_edit::value(42);
        let target = fs::canonicalize(&path).unwrap();
        let lock = acquire_config_lock(&target).unwrap();
        let original = target.with_extension("original");
        fs::rename(&target, &original).unwrap();
        let (ready_tx, ready_rx) = mpsc::channel();
        let (tx, rx) = mpsc::channel();
        let writer_path = path.clone();
        let writer = std::thread::spawn(move || {
            ready_tx.send(()).unwrap();
            let _ = tx.send(save_config_document(&writer_path, &document));
        });
        ready_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        let before_unlock = rx.recv_timeout(Duration::from_millis(100));
        fs::rename(&original, &target).unwrap();
        drop(lock);
        assert!(matches!(
            before_unlock,
            Err(mpsc::RecvTimeoutError::Timeout)
        ));
        rx.recv_timeout(Duration::from_secs(5)).unwrap().unwrap();
        writer.join().unwrap();
        assert_eq!(
            AppConfig::load_from_path(path)
                .unwrap()
                .global
                .max_iterations,
            Some(42)
        );
    }
}

#[test]
fn config_read_preserves_missing_and_other_io_errors() {
    let directory = tempfile::tempdir().unwrap();
    for missing in [
        directory.path().join("missing.toml"),
        directory.path().join("missing-dir/letcode.toml"),
    ] {
        let error = AppConfig::load_from_path(missing).unwrap_err();
        assert!(error.to_string().starts_with("config file not found:"));
        assert_eq!(
            error.downcast_ref::<std::io::Error>().unwrap().kind(),
            std::io::ErrorKind::NotFound
        );
    }
    let invalid = directory.path().join("directory.toml");
    fs::create_dir(&invalid).unwrap();
    let error = AppConfig::load_from_path(invalid).unwrap_err();
    assert!(error.to_string().starts_with("failed to read config file"));
    assert!(error.downcast_ref::<std::io::Error>().is_some());
}

#[cfg(unix)]
#[test]
fn config_symlink_target_and_lock_are_stable_when_the_target_is_missing() {
    use std::os::unix::fs::symlink;

    let directory = tempfile::tempdir().unwrap();
    let alias_dir = directory.path().join("alias");
    let target_dir = directory.path().join("target");
    fs::create_dir(&alias_dir).unwrap();
    fs::create_dir(&target_dir).unwrap();
    let alias = alias_dir.join("config.toml");
    let target = target_dir.join("letcode.toml");
    symlink("next.toml", &alias).unwrap();
    symlink("../target/letcode.toml", alias_dir.join("next.toml")).unwrap();
    let missing = resolve_config_target(&alias).unwrap();
    assert_eq!(missing, resolve_config_target(&target).unwrap());
    fs::write(&target, "original").unwrap();
    let present = resolve_config_target(&alias).unwrap();
    assert_eq!(missing, present);
    assert_eq!(present, fs::canonicalize(target).unwrap());
    assert_eq!(
        config_lock_path(&missing).unwrap(),
        config_lock_path(&present).unwrap()
    );
}

#[cfg(unix)]
#[test]
fn config_symlink_cycles_report_resolution_errors() {
    use std::os::unix::fs::symlink;

    let directory = tempfile::tempdir().unwrap();
    let first = directory.path().join("first.toml");
    let second = directory.path().join("second.toml");
    symlink("second.toml", &first).unwrap();
    symlink("first.toml", &second).unwrap();
    let error = AppConfig::load_from_path(first).unwrap_err();
    assert!(error.to_string().contains("config symlink cycle"));
}

#[test]
fn fake_section_is_absent_by_default() {
    let loaded = AppConfig::load_from_path(write_temp_config(config("openai", "model", "")))
        .expect("config should load");
    assert_eq!(loaded.fake, FakeConfig::default());
}

#[test]
fn fake_section_loads_declared_values() {
    let loaded = AppConfig::load_from_path(write_temp_config(config(
        "openai",
        "model",
        r#"[fake.identity]
installation_id = "declared-installation"
agent_name = "declared-agent"
account_uuid = "declared-account"
[fake.clock]
timezone = "Asia/Shanghai"
date = "2026-01-02"
[fake.codex]
version = "1.2.3"
originator = "codex_cli_rs"
beta_features = ["one", "two"]
sandbox_mode = "read-only"
auto_review_enabled = true
[fake.codex.extra]
custom_key = "custom value"
[fake.claude]
version = "2.1.69"
build = "6bf"
package_version = "0.74.0"
runtime_version = "v22.16.0"
timeout = "600"
os = "MacOS"
arch = "x64"
betas = ["claude-code-20250219"]
"#,
    )))
    .expect("config should load");

    assert_eq!(
        loaded.fake.identity.installation_id.as_deref(),
        Some("declared-installation")
    );
    assert_eq!(
        loaded.fake.identity.agent_name.as_deref(),
        Some("declared-agent")
    );
    assert_eq!(
        loaded.fake.identity.account_uuid.as_deref(),
        Some("declared-account")
    );
    assert_eq!(loaded.fake.clock.timezone.as_deref(), Some("Asia/Shanghai"));
    assert_eq!(loaded.fake.clock.date.as_deref(), Some("2026-01-02"));
    assert_eq!(loaded.fake.codex.version.as_deref(), Some("1.2.3"));
    assert_eq!(
        loaded.fake.codex.beta_features.as_deref(),
        Some(["one".to_string(), "two".to_string()].as_slice())
    );
    assert_eq!(loaded.fake.codex.sandbox_mode.as_deref(), Some("read-only"));
    assert_eq!(loaded.fake.codex.auto_review_enabled, Some(true));
    assert_eq!(
        loaded
            .fake
            .codex
            .extra
            .get("custom_key")
            .map(String::as_str),
        Some("custom value")
    );
    assert_eq!(loaded.fake.claude.version.as_deref(), Some("2.1.69"));
    assert_eq!(loaded.fake.claude.build.as_deref(), Some("6bf"));
    assert_eq!(
        loaded.fake.claude.package_version.as_deref(),
        Some("0.74.0")
    );
    assert_eq!(
        loaded.fake.claude.runtime_version.as_deref(),
        Some("v22.16.0")
    );
    assert_eq!(loaded.fake.claude.timeout.as_deref(), Some("600"));
    assert_eq!(loaded.fake.claude.os.as_deref(), Some("MacOS"));
    assert_eq!(loaded.fake.claude.arch.as_deref(), Some("x64"));
    assert_eq!(
        loaded.fake.claude.betas.as_deref(),
        Some(["claude-code-20250219".to_string()].as_slice())
    );
}

#[test]
fn fake_section_rejects_unknown_keys() {
    let error = AppConfig::load_from_path(write_temp_config(config(
        "openai",
        "model",
        "[fake]\nunknown_key = 1\n",
    )))
    .expect_err("config should be rejected");
    assert!(format!("{error:#}").contains("failed to parse config file"));
}

#[test]
fn fake_rejects_blank_declared_values() {
    let error = AppConfig::load_from_path(write_temp_config(config(
        "openai",
        "model",
        "[fake.clock]\ntimezone = \"   \"\n",
    )))
    .expect_err("config should be rejected");
    assert!(format!("{error:#}").contains("fake.clock.timezone cannot be empty"));
}

#[test]
fn fake_extra_rejects_reserved_keys() {
    let error = AppConfig::load_from_path(write_temp_config(config(
        "openai",
        "model",
        "[fake.codex.extra]\nsession_id = \"other\"\n",
    )))
    .expect_err("config should be rejected");
    assert!(format!("{error:#}").contains("reserved key 'session_id'"));

    let error = AppConfig::load_from_path(write_temp_config(config(
        "openai",
        "model",
        "[fake.codex.extra]\nworkspaces = \"text\"\n",
    )))
    .expect_err("config should be rejected");
    assert!(format!("{error:#}").contains("reserved key 'workspaces'"));
}

#[test]
fn fake_extra_enforces_upstream_extension_limits() {
    let too_many = (0..17)
        .map(|index| format!("key{index} = \"value\"\n"))
        .collect::<String>();
    let error = AppConfig::load_from_path(write_temp_config(config(
        "openai",
        "model",
        &format!("[fake.codex.extra]\n{too_many}"),
    )))
    .expect_err("config should be rejected");
    assert!(format!("{error:#}").contains("at most 16 entries"));

    let error = AppConfig::load_from_path(write_temp_config(config(
        "openai",
        "model",
        "[fake.codex.extra]\n\"1bad\" = \"value\"\n",
    )))
    .expect_err("config should be rejected");
    assert!(format!("{error:#}").contains("must be short ASCII identifiers"));

    let long = "v".repeat(129);
    let error = AppConfig::load_from_path(write_temp_config(config(
        "openai",
        "model",
        &format!("[fake.codex.extra]\nkey = \"{long}\"\n"),
    )))
    .expect_err("config should be rejected");
    assert!(format!("{error:#}").contains("at most 128 bytes"));
}

#[test]
fn rejects_retry_attempts_above_maximum() {
    let error = AppConfig::load_from_path(write_temp_config(config(
        "openai",
        "model",
        "[global.retry]\nmax_attempts = 9001\n",
    )))
    .expect_err("config should be rejected");
    assert!(format!("{error:#}").contains("global.retry.max_attempts must be at most 9000"));
}

#[test]
fn session_archive_settings_default_and_override() {
    let default =
        AppConfig::load_from_path(write_temp_config(config("openai", "model", ""))).unwrap();
    assert_eq!(
        default.global.session_archive,
        SessionArchiveConfig::default()
    );
    assert!(default.global.session_archive.enabled);
    assert_eq!(default.global.session_archive.older_than_days, 7);

    let overridden = AppConfig::load_from_path(write_temp_config(config(
        "openai",
        "model",
        "[global.session_archive]\nenabled = false\nolder_than_days = 30\n",
    )))
    .unwrap();
    assert!(!overridden.global.session_archive.enabled);
    assert_eq!(overridden.global.session_archive.older_than_days, 30);
}

#[test]
fn rejects_session_archive_window_of_zero_days() {
    let error = AppConfig::load_from_path(write_temp_config(config(
        "openai",
        "model",
        "[global.session_archive]\nolder_than_days = 0\n",
    )))
    .expect_err("config should be rejected");
    assert!(
        format!("{error:#}")
            .contains("global.session_archive.older_than_days must be greater than 0")
    );
}

#[test]
fn provider_retry_can_override_backoff_mode_interval_and_attempts() {
    let loaded = AppConfig::load_from_path(write_temp_config(
        r#"active_provider = "openai"
[global.retry]
enabled = true
max_attempts = 9
initial_delay_secs = 1
exponential_backoff = true
backoff_multiplier = 2.0
jitter_secs = 0
[providers.openai]
protocol = "responses"
default_model = "model"
flavor = "standard"
[providers.openai.retry]
max_attempts = 4
initial_delay_secs = 7
exponential_backoff = false
[providers.openai.auth]
type = "bearer"
credential = "config-key"
[providers.openai.endpoints]
base_url = "https://example.invalid/v1"
[providers.openai.models.model]
"#,
    ))
    .expect("provider retry config loads");
    let retry = loaded.providers["openai"].retry.as_ref().unwrap();
    assert!(retry.enabled);
    assert_eq!(retry.max_attempts, 4);
    assert_eq!(retry.initial_delay_secs, 7);
    assert!(!retry.exponential_backoff);
    assert_eq!(retry.backoff_multiplier, 2.0);
    assert_eq!(retry.jitter_secs, 0);
}

#[test]
fn rejects_invalid_model_specific_reasoning_efforts() {
    for extra in [
        "[capabilities]\nreasoning = false\n[generation]\nreasoning_effort = \"low\"",
        "[capabilities]\nreasoning = true\n[generation]\nreasoning_effort = \"high\"\nreasoning_efforts = [\"low\", \"medium\"]",
        "[capabilities]\nreasoning = true\n[generation]\nreasoning_efforts = [\"low\", \"low\"]",
        "[capabilities]\nreasoning = true\n[generation]\nreasoning_effort = \"provider ultra\"",
    ] {
        assert!(
            AppConfig::load_from_path(write_temp_config(config("openai", "model", extra))).is_err(),
            "invalid reasoning config should be rejected: {extra}"
        );
    }
}

#[test]
fn model_parallel_tool_calls_default_to_disabled_and_can_be_enabled() {
    let default_config =
        AppConfig::load_from_path(write_temp_config(config("openai", "model", "")))
            .expect("default config should load");
    assert!(!default_config.providers["openai"].models["model"].parallel_tool_calls);

    let enabled = config(
        "openai",
        "model",
        "[capabilities]\nparallel_tool_calls = true\ngeneration = { parallel_tool_calls = true }\n[generation]\nparallel_tool_calls = true",
    );
    let enabled_config =
        AppConfig::load_from_path(write_temp_config(enabled)).expect("enabled config should load");
    assert!(enabled_config.providers["openai"].models["model"].parallel_tool_calls);
}

#[test]
fn anthropic_betas_are_explicit_and_protocol_scoped() {
    let anthropic = config(
        "anyrouter",
        "claude-opus",
        "protocol = \"anthropic\"\n[protocol_settings]\nanthropic_betas = [\"context-1m-2025-08-07\"]",
    );
    let loaded = AppConfig::load_from_path(write_temp_config(anthropic))
        .expect("anthropic beta config loads");
    assert_eq!(
        loaded.providers["anyrouter"].models["claude-opus"].anthropic_betas,
        ["context-1m-2025-08-07"]
    );

    let invalid = config(
        "compat",
        "model",
        "[protocol_settings]\nanthropic_betas = [\"context-1m-2025-08-07\"]",
    );
    let error = AppConfig::load_from_path(write_temp_config(invalid))
        .expect_err("non-anthropic beta config should be rejected");
    assert!(format!("{error:#}").contains("protocol"));
}

#[test]
fn anthropic_model_override_requires_provider_auth_mode() {
    let missing = config("mixed", "claude", "protocol = \"anthropic\"");
    let loaded = AppConfig::load_from_path(write_temp_config(missing))
        .expect("named auth is supplied by the new schema");
    assert_eq!(
        loaded.providers["mixed"].auth_mode,
        ProviderAuthMode::Bearer
    );

    let configured = r#"active_provider = "mixed"
[providers.mixed]
protocol = "anthropic"
default_model = "claude"
[providers.mixed.auth]
type = "header"
name = "x-vendor-key"
credential = "config-key"
[providers.mixed.endpoints]
base_url = "https://example.invalid/v1"
[providers.mixed.models.claude]
"#;
    let loaded = AppConfig::load_from_path(write_temp_config(configured))
        .expect("configured override should load");
    assert_eq!(
        loaded.providers["mixed"].auth_mode,
        ProviderAuthMode::ApiKey
    );
}

#[test]
fn rejects_zero_model_effective_input_limit_tokens() {
    let error = AppConfig::load_from_path(write_temp_config(config(
        "openai",
        "model",
        "effective_input_limit_tokens = 0",
    )))
    .expect_err("config should be rejected");
    assert!(format!("{error:#}").contains("effective_input_limit_tokens"));
}

#[test]
fn parses_provider_qualified_expert_routes_and_preserves_duplicate_model_ids() {
    let loaded = AppConfig::load_from_path(write_temp_config(
        r#"active_provider = "primary"
[agents.explorer]
provider = "primary"
model = "shared"
[agents.fixer]
provider = "expert"
model = "shared"
[providers.primary]
protocol = "responses"
default_model = "shared"
flavor = "standard"
[providers.primary.auth]
type = "bearer"
credential = "primary-key"
[providers.primary.endpoints]
base_url = "https://primary.invalid/v1"
[providers.primary.models.shared]
[providers.expert]
protocol = "completions"
default_model = "shared"
flavor = "standard"
[providers.expert.auth]
type = "bearer"
credential = "expert-key"
[providers.expert.endpoints]
base_url = "https://expert.invalid/v1"
[providers.expert.models.shared]
"#,
    ))
    .expect("config should load");
    assert_eq!(
        loaded.model_route_for("explorer"),
        Some(&ModelRoute::new("primary", "shared"))
    );
    assert_eq!(
        loaded.model_route_for("fixer"),
        Some(&ModelRoute::new("expert", "shared"))
    );
    assert_eq!(loaded.active_route(), ModelRoute::new("primary", "shared"));
    assert_eq!(
        loaded
            .resolve_route(loaded.model_route_for("fixer").unwrap())
            .unwrap()
            .protocol,
        ApiProtocol::Completions
    );
}

#[test]
fn parses_and_validates_expert_reasoning_effort() {
    let loaded = AppConfig::load_from_path(write_temp_config(
        r#"active_provider = "primary"
[agents.oracle]
provider = "primary"
model = "shared"
reasoning_effort = "high"
[providers.primary]
default_model = "shared"
[providers.primary.auth]
type = "bearer"
credential = "primary-key"
[providers.primary.endpoints]
base_url = "https://primary.invalid/v1"
[providers.primary.models.shared]
"#,
    ))
    .expect("expert reasoning config loads");
    assert_eq!(
        loaded.agents.reasoning_effort_for("oracle"),
        Some(ModelReasoningEffort::High)
    );
    assert_eq!(loaded.agents.reasoning_effort_for("explorer"), None);
    assert_eq!(
        loaded.expert_reasoning_efforts().get("oracle"),
        Some(&ModelReasoningEffort::High)
    );

    // Provider-specific levels stay available to the model metadata check.
    let custom = AppConfig::load_from_path(write_temp_config(
        r#"active_provider = "primary"
[agents.oracle]
provider = "primary"
model = "shared"
reasoning_effort = "provider ultra"
[providers.primary]
default_model = "shared"
[providers.primary.auth]
type = "bearer"
credential = "primary-key"
[providers.primary.endpoints]
base_url = "https://primary.invalid/v1"
[providers.primary.models.shared]
"#,
    ))
    .expect("a provider-specific level loads");
    assert_eq!(
        custom.agents.reasoning_effort_for("oracle"),
        Some(ModelReasoningEffort::Custom("provider ultra".into()))
    );

    let error = AppConfig::load_from_path(write_temp_config(
        r#"active_provider = "primary"
[agents.oracle]
provider = "primary"
model = "shared"
reasoning_effort = ""
[providers.primary]
default_model = "shared"
[providers.primary.auth]
type = "bearer"
credential = "primary-key"
[providers.primary.endpoints]
base_url = "https://primary.invalid/v1"
[providers.primary.models.shared]
"#,
    ))
    .expect_err("an empty reasoning level is rejected");
    assert!(
        error.to_string().contains("agents.oracle.reasoning_effort"),
        "{error}"
    );
}

#[test]
fn parses_and_validates_expert_allowed_models() {
    let loaded = AppConfig::load_from_path(write_temp_config(
        r#"active_provider = "primary"
[agents.explorer]
provider = "primary"
model = "shared"
allowed_models = ["expert/special"]
[providers.primary]
default_model = "shared"
[providers.primary.auth]
type = "bearer"
credential = "primary-key"
[providers.primary.endpoints]
base_url = "https://primary.invalid/v1"
[providers.primary.models.shared]
[providers.expert]
default_model = "special"
[providers.expert.auth]
type = "bearer"
credential = "expert-key"
[providers.expert.endpoints]
base_url = "https://expert.invalid/v1"
[providers.expert.models.special]
"#,
    ))
    .expect("allowed model config loads");
    assert_eq!(
        loaded.agents.allowed_models_for("explorer"),
        Some([ModelRoute::new("expert", "special")].as_slice())
    );
    assert_eq!(
        loaded.model_route_for("explorer"),
        Some(&ModelRoute::new("primary", "shared"))
    );

    for allowed in [
        "special",
        "missing/special",
        "expert/missing",
        "expert/special, expert/special",
    ] {
        let mut text = loaded_config_for_allowlist(allowed);
        if allowed == "expert/special, expert/special" {
            text = text.replace(
                "allowed_models = [\"expert/special\"]",
                "allowed_models = [\"expert/special\", \"expert/special\"]",
            );
        }
        assert!(
            AppConfig::load_from_path(write_temp_config(text)).is_err(),
            "{allowed}"
        );
    }
}

#[test]
fn rejects_incomplete_or_unknown_provider_qualified_expert_routes() {
    let incomplete = r#"[agents.explorer]
provider = "expert"
[providers.openai]
[providers.openai.auth]
type = "bearer"
credential = "key"
[providers.openai.endpoints]
base_url = "https://example.invalid"
[providers.openai.models.model]
"#;
    assert!(AppConfig::load_from_path(write_temp_config(incomplete)).is_err());

    let unknown_provider = config(
        "openai",
        "model",
        "[agents.explorer]\nprovider = \"missing\"\nmodel = \"model\"",
    );
    assert!(AppConfig::load_from_path(write_temp_config(unknown_provider)).is_err());

    let unknown_model = config(
        "openai",
        "model",
        "[agents.explorer]\nprovider = \"openai\"\nmodel = \"missing\"",
    );
    assert!(AppConfig::load_from_path(write_temp_config(unknown_model)).is_err());
}

#[test]
fn rejects_unknown_subagent_name_and_model_override() {
    let unknown_agent = config(
        "openai",
        "model",
        "[agents.nosuch]\nprovider = \"openai\"\nmodel = \"model\"",
    );
    assert!(AppConfig::load_from_path(write_temp_config(unknown_agent)).is_err());
    let unknown_model = config(
        "openai",
        "model",
        "[agents.explorer]\nprovider = \"openai\"\nmodel = \"missing-model\"",
    );
    assert!(AppConfig::load_from_path(write_temp_config(unknown_model)).is_err());
}

#[test]
fn rejects_reasoning_parameters_when_reasoning_is_disabled() {
    let error = AppConfig::load_from_path(write_temp_config(config(
        "openai",
        "model",
        "[capabilities]\nreasoning = false\n[generation]\nreasoning_effort = \"medium\"",
    )))
    .expect_err("load should fail");
    assert!(format!("{error:#}").contains("reasoning capabilities"));
}

#[test]
fn anthropic_reasoning_defaults_require_adaptive_thinking() {
    let route = |extra: &str| {
        config(
            "vendor",
            "model",
            &format!("protocol = \"anthropic\"\n{extra}"),
        )
    };

    let error = AppConfig::load_from_path(write_temp_config(route(
            "[capabilities]\nreasoning = true\ngeneration = { reasoning = true }\n[generation]\nreasoning_effort = \"high\"",
        )))
        .expect_err("reasoning without thinking settings should fail");
    assert!(format!("{error:#}").contains("anthropic_thinking"));

    let loaded = AppConfig::load_from_path(write_temp_config(route(
            "[capabilities]\nreasoning = true\ngeneration = { reasoning = true }\n[generation]\nreasoning_effort = \"high\"\n[protocol_settings]\nanthropic_thinking = { mode = \"adaptive\" }",
        )))
        .expect("adaptive thinking should load");
    assert!(
        !loaded.providers["vendor"].models["model"]
            .request_metadata()
            .selectable_reasoning_efforts()
            .is_empty()
    );
}

#[test]
fn reasoning_effort_is_not_offered_when_the_route_cannot_express_it() {
    let efforts = |extra: &str| {
        let loaded = AppConfig::load_from_path(write_temp_config(config("vendor", "model", extra)))
            .expect("route should load");
        loaded.providers["vendor"].models["model"]
            .request_metadata()
            .selectable_reasoning_efforts()
    };

    // Anthropic reasons only through thinking settings; without them no effort can be sent.
    assert!(efforts("protocol = \"anthropic\"\n[capabilities]\nreasoning = true").is_empty());
    // Other protocols need the generation support flag, not just the model capability.
    assert!(efforts("[capabilities]\nreasoning = true").is_empty());
    assert!(
        !efforts("[capabilities]\nreasoning = true\ngeneration = { reasoning = true }").is_empty()
    );
}

#[test]
fn rejects_out_of_range_sampling_parameters() {
    let error = AppConfig::load_from_path(write_temp_config(config(
        "openai",
        "model",
        "[capabilities]\ngeneration = { temperature = true }\n[generation]\ntemperature = 3.0",
    )))
    .expect_err("load should fail");
    assert!(format!("{error:#}").contains("temperature"));
}

#[test]
fn persists_expert_allowed_models_without_rewriting_unrelated_config() {
    let path = write_temp_config(
        "# keep this comment\nactive_provider = \"primary\"\n\n[providers.primary]\nprotocol = \"responses\"\ndefault_model = \"old\"\nflavor = \"standard\"\n[providers.primary.auth]\ntype = \"bearer\"\ncredential = \"primary-key\"\n[providers.primary.endpoints]\nbase_url = \"https://primary.invalid/v1\"\n[providers.primary.models.old]\n\n[providers.expert]\nprotocol = \"responses\"\ndefault_model = \"shared\"\nflavor = \"standard\"\n[providers.expert.auth]\ntype = \"bearer\"\ncredential = \"expert-key\"\n[providers.expert.endpoints]\nbase_url = \"https://expert.invalid/v1\"\n[providers.expert.models.shared]\n\n[agents.explorer]\nprovider = \"primary\"\nmodel = \"old\"\n# preserve this trailing comment\n",
    );
    persist_expert_allowed_models(
        &path,
        "explorer",
        &[
            ModelRoute::new("primary", "old"),
            ModelRoute::new("expert", "shared"),
        ],
    )
    .expect("persist expert allowed models");
    let written = fs::read_to_string(&path).unwrap();
    assert!(written.contains("# keep this comment"));
    assert!(written.contains("# preserve this trailing comment"));
    assert!(written.contains("primary/old"));
    assert!(written.contains("expert/shared"));
}

#[test]
fn persists_scalar_without_rewriting_unrelated_config() {
    let path = write_temp_config(format!(
        "# keep this comment\n{}\n",
        config("openai", "model", "")
    ));
    let mut document = fs::read_to_string(&path)
        .unwrap()
        .parse::<toml_edit::DocumentMut>()
        .unwrap();
    set_config_scalar(
        &mut document,
        &["global", "max_iterations"],
        ConfigScalar::Integer(42),
    )
    .expect("set scalar");
    save_config_document(&path, &document).expect("save config");
    let written = fs::read_to_string(&path).unwrap();
    assert!(written.contains("# keep this comment"));
    assert!(written.contains("max_iterations = 42"));
}

#[cfg(unix)]
#[test]
fn persists_through_config_symlink_without_replacing_it() {
    use std::os::unix::fs::symlink;
    let target = write_temp_config(format!(
        "{}\n\n[mcp.alpha]\ntype = \"local\"\ncommand = [\"alpha\"]",
        config("openai", "model", "")
    ));
    let link = target.with_file_name("letcode-link.toml");
    symlink(&target, &link).expect("create config symlink");
    persist_mcp_server_enabled(&link, "alpha", false).expect("persist through symlink");
    assert!(
        fs::symlink_metadata(&link)
            .unwrap()
            .file_type()
            .is_symlink()
    );
    assert!(
        fs::read_to_string(&target)
            .unwrap()
            .contains("enabled = false")
    );
}

#[test]
fn rejects_invalid_mcp_local_config() {
    let error = AppConfig::load_from_path(write_temp_config(format!(
        "{}\n\n[mcp.empty]\ntype = \"local\"\ncommand = []",
        config("openai", "model", "")
    )))
    .expect_err("load should fail");
    assert!(format!("{error:#}").contains("mcp.empty.command cannot be empty"));
}

#[test]
fn rejects_remote_mcp_oauth_until_supported() {
    let error = AppConfig::load_from_path(write_temp_config(format!(
        "{}\n\n[mcp.docs]\ntype = \"remote\"\nurl = \"https://example.invalid/mcp\"\noauth = true",
        config("openai", "model", "")
    )))
    .expect_err("load should fail");
    assert!(format!("{error:#}").contains("remote MCP OAuth is not supported"));
}

#[test]
fn rejects_unknown_fields_and_zero_limits_and_empty_identifiers() {
    let root_unknown = format!("unexpected = true\n{}", config("openai", "model", ""));
    assert!(AppConfig::load_from_path(write_temp_config(root_unknown)).is_err());
    let zero = format!(
        "[global]\nmax_iterations = 0\n\n{}",
        config("openai", "model", "")
    );
    assert!(AppConfig::load_from_path(write_temp_config(zero)).is_err());
    let empty = config("openai", "model", "default_model = \"   \"");
    assert!(AppConfig::load_from_path(write_temp_config(empty)).is_err());
}

#[test]
fn errors_when_active_provider_is_missing() {
    let error =
        AppConfig::load_from_path(write_temp_config(config("openai", "model", "").replace(
            "active_provider = \"openai\"",
            "active_provider = \"missing\"",
        )))
        .expect_err("load should fail");
    assert!(format!("{error:#}").contains("active_provider"));
}

#[test]
fn prompt_cache_rejects_retention_namespace_and_unknown_fields() {
    let disabled = config(
        "openai",
        "model",
        "[cache]\nenabled = false\nretention = \"in_memory\"",
    );
    assert!(AppConfig::load_from_path(write_temp_config(disabled)).is_err());
    let completions = config(
        "openai",
        "model",
        "protocol = \"completions\"\n[cache]\nenabled = true\nretention = \"24h\"",
    );
    assert!(AppConfig::load_from_path(write_temp_config(completions)).is_err());
    for namespace in ["   ", &"a".repeat(65), "valid\u{0007}name"] {
        let value = config(
            "openai",
            "model",
            &format!("[cache]\nenabled = true\nnamespace = {namespace:?}"),
        );
        assert!(AppConfig::load_from_path(write_temp_config(value)).is_err());
    }
    let unknown = config(
        "openai",
        "model",
        "[cache]\nenabled = true\nlayout = \"v1\"",
    );
    assert!(AppConfig::load_from_path(write_temp_config(unknown)).is_err());
}

#[test]
fn initialize_config_writes_a_minimal_valid_config() {
    let path = write_missing_config_path();
    initialize_config(
        &path,
        "https://example.invalid/v1",
        "secret-key",
        "example-model",
    )
    .expect("initial config should be written");

    let written = fs::read_to_string(&path).expect("initial config should be readable");
    assert!(written.starts_with(CONFIG_SCHEMA_COMMENT));

    let loaded = AppConfig::load_from_path(&path).expect("initial config should load");
    assert_eq!(loaded.active_provider, "default");
    assert_eq!(loaded.active_route().model, "example-model");
    assert_eq!(loaded.providers["default"].api_key, "secret-key");
    assert!(loaded.providers["default"].models["example-model"].supports_tools);
}

fn write_missing_config_path() -> PathBuf {
    static NEXT_ID: AtomicU64 = AtomicU64::new(0);
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    let base = env::temp_dir().join(format!("letcode-initialize-config-test-{timestamp}-{id}"));
    fs::create_dir_all(&base).unwrap();
    base.join("nested").join("letcode.toml")
}

fn loaded_config_for_allowlist(allowed_models: &str) -> String {
    format!(
        "active_provider = \"primary\"\n[agents.explorer]\nmodel = \"shared\"\nallowed_models = [{allowed_models:?}]\n{}{}",
        config("primary", "shared", ""),
        config("expert", "special", "")
    )
}

fn write_temp_config(contents: impl AsRef<str>) -> PathBuf {
    static NEXT_ID: AtomicU64 = AtomicU64::new(0);
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    let base = env::temp_dir().join(format!("letcode-config-test-{timestamp}-{id}"));
    fs::create_dir_all(&base).unwrap();
    let path = base.join("letcode.toml");
    fs::write(&path, contents.as_ref()).unwrap();
    path
}

fn env_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

#[allow(dead_code)]
fn lock_env() -> std::sync::MutexGuard<'static, ()> {
    env_lock()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}
