//! Session-owned idle settings mutations shared by TUI and line CLI.
//!
//! These helpers apply agent configuration changes and record transcript
//! provenance when the value actually changes. Frontends retain presentation
//! concerns (confirm dialogs, model catalog validation, status printing).

use std::sync::{Arc, Mutex};

use anyhow::{Result, anyhow};
use async_openai::config::Config;

use crate::agent::Agent;
use crate::config::ModelRoute;
use crate::permission::PermissionMode;
use crate::request_builder::ModelReasoningEffort;
use crate::transcript::TranscriptRecorder;

/// The result of a model apply that can disable Fast Mode before transcript
/// provenance is recorded.
#[derive(Debug)]
pub struct ModelApplyError {
    error: anyhow::Error,
    fast_mode_auto_disabled: bool,
}

impl ModelApplyError {
    pub(crate) fn new(error: anyhow::Error, fast_mode_auto_disabled: bool) -> Self {
        Self {
            error,
            fast_mode_auto_disabled,
        }
    }

    pub fn fast_mode_auto_disabled(&self) -> bool {
        self.fast_mode_auto_disabled
    }
}

impl std::fmt::Display for ModelApplyError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.error.fmt(formatter)
    }
}

impl std::error::Error for ModelApplyError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.error.as_ref())
    }
}

/// Apply a permission mode and record provenance when it changes.
pub fn apply_permission_mode<C: Config>(
    agent: &mut Agent<C>,
    transcript: &Arc<Mutex<TranscriptRecorder>>,
    mode: PermissionMode,
) -> Result<()> {
    let previous = agent.permission_mode();
    agent.set_permission_mode(mode);
    if previous != mode {
        transcript
            .lock()
            .map_err(|_| anyhow!("transcript recorder poisoned"))?
            .record_permission_mode_changed(previous.to_string(), mode.to_string())?;
    }
    Ok(())
}

pub(crate) fn apply_model_route_with(
    agent: &mut Agent<async_openai::config::OpenAIConfig>,
    transcript: &Arc<Mutex<TranscriptRecorder>>,
    route: ModelRoute,
    prepared_route: crate::agent::PreparedPrimaryRoute<async_openai::config::OpenAIConfig>,
) -> std::result::Result<bool, ModelApplyError> {
    let previous_route = agent.route_display_name();
    let new_route = route.display_name();
    let route_changed = previous_route != new_route;
    let fast_mode_auto_disabled = agent
        .auto_disable_fast_mode_for_model(&route.model)
        .map_err(|error| ModelApplyError::new(error, false))?;

    if route_changed {
        transcript
            .lock()
            .map_err(|_| {
                ModelApplyError::new(
                    anyhow!("transcript recorder poisoned"),
                    fast_mode_auto_disabled,
                )
            })?
            .record_model_changed(previous_route, new_route)
            .map_err(|error| ModelApplyError::new(error, fast_mode_auto_disabled))?;
    }

    agent.apply_prepared_route(prepared_route);
    Ok(fast_mode_auto_disabled)
}

/// Persist a primary-route selection before installing it. If the live route
/// cannot be applied afterwards, restore the previous persisted selection.
#[cfg(test)]
pub(crate) fn persist_and_apply_model_route_with(
    agent: &mut Agent<async_openai::config::OpenAIConfig>,
    transcript: &Arc<Mutex<TranscriptRecorder>>,
    route: ModelRoute,
    prepared_route: crate::agent::PreparedPrimaryRoute<async_openai::config::OpenAIConfig>,
    mut persist: impl FnMut(&ModelRoute) -> Result<()>,
) -> std::result::Result<bool, ModelApplyError> {
    let previous_route = agent.primary_route().cloned().ok_or_else(|| {
        ModelApplyError::new(anyhow!("current primary route is unavailable"), false)
    })?;
    let route_changed = previous_route != route;

    persist(&route).map_err(|error| {
        ModelApplyError::new(
            anyhow!(
                "failed to persist model route '{}': {error}",
                route.display_name()
            ),
            false,
        )
    })?;

    match apply_model_route_with(agent, transcript, route, prepared_route) {
        Ok(fast_mode_auto_disabled) => Ok(fast_mode_auto_disabled),
        Err(error) if !route_changed => Err(error),
        Err(error) => {
            if let Err(rollback_error) = persist(&previous_route) {
                return Err(ModelApplyError::new(
                    anyhow!(
                        "failed to apply model route after persisting it: {error}; failed to restore persisted route '{}': {rollback_error}",
                        previous_route.display_name()
                    ),
                    error.fast_mode_auto_disabled(),
                ));
            }
            Err(error)
        }
    }
}

/// Apply reasoning effort. No transcript provenance event today.
pub fn apply_reasoning_effort<C: Config>(
    agent: &mut Agent<C>,
    effort: ModelReasoningEffort,
) -> Result<()> {
    agent.set_reasoning_effort(effort)
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_openai::{Client, config::OpenAIConfig};
    use serde_json::json;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_transcript() -> (std::path::PathBuf, Arc<Mutex<TranscriptRecorder>>, String) {
        let base_dir = std::env::temp_dir().join(format!(
            "letcode-session-settings-test-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("time ok")
                .as_nanos()
        ));
        let recorder = TranscriptRecorder::create(&base_dir).expect("create transcript");
        let session_id = recorder.session_id().to_string();
        (base_dir, Arc::new(Mutex::new(recorder)), session_id)
    }

    fn test_agent() -> Agent<OpenAIConfig> {
        let config = OpenAIConfig::new()
            .with_api_base("http://127.0.0.1:9/v1")
            .with_api_key("test-key");
        Agent::new(Client::with_config(config), "gpt-5.5", 1, 1)
    }

    fn configured_agent(fast_mode_path: &std::path::Path) -> Agent<OpenAIConfig> {
        std::fs::write(
            fast_mode_path,
            r#"active_provider = "primary"

[providers.primary]
base_url = "https://primary.invalid/v1"
api_key = "primary-key"
protocol = "responses"
[providers.primary.models."gpt-5.5"]
"#,
        )
        .expect("write Fast Mode config");
        let mut agent = test_agent();
        agent.set_primary_route(ModelRoute::new("primary", "gpt-5.5"));
        let fast_mode = crate::fast_mode::FastMode::load(fast_mode_path, true);
        agent.set_fast_mode(fast_mode);
        agent
    }

    fn prepared_route(route: ModelRoute) -> crate::agent::PreparedPrimaryRoute<OpenAIConfig> {
        crate::agent::PreparedPrimaryRoute::new(
            Client::with_config(
                OpenAIConfig::new()
                    .with_api_base("http://127.0.0.1:9/v1")
                    .with_api_key("expert-key"),
            ),
            route,
            crate::config::ApiProtocol::Responses,
            std::collections::HashMap::new(),
            std::collections::HashMap::new(),
            crate::config::RetryConfig::default(),
        )
    }

    #[test]
    fn reasoning_effort_change_is_session_local() {
        let base_dir = std::env::temp_dir().join(format!(
            "letcode-session-reasoning-config-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("time ok")
                .as_nanos()
        ));
        std::fs::create_dir_all(&base_dir).expect("create config directory");
        let config_path = base_dir.join("letcode.toml");
        std::fs::write(&config_path, "session reasoning sentinel\n").expect("write config");
        let before = std::fs::read_to_string(&config_path).expect("read config before change");
        let mut agent = test_agent();
        agent.set_model_catalog(std::collections::HashMap::from([(
            "gpt-5.5".into(),
            crate::request_builder::ModelRequestMetadata {
                supports_reasoning: true,
                reasoning_effort: Some(ModelReasoningEffort::Medium),
                reasoning_efforts: vec![ModelReasoningEffort::Medium, ModelReasoningEffort::High],
                ..Default::default()
            },
        )]));

        apply_reasoning_effort(&mut agent, ModelReasoningEffort::High)
            .expect("reasoning effort changes in memory");

        assert_eq!(agent.reasoning_effort(), Some(ModelReasoningEffort::High));
        assert_eq!(
            agent.model_catalog()["gpt-5.5"].reasoning_effort,
            Some(ModelReasoningEffort::Medium),
            "session selection must not mutate configured model metadata"
        );
        assert_eq!(
            std::fs::read_to_string(&config_path).expect("read config after change"),
            before
        );
    }

    #[test]
    fn model_route_apply_failure_restores_the_persisted_route() {
        let (_base_dir, transcript, _session_id) = temp_transcript();
        let cloned = Arc::clone(&transcript);
        let join = std::thread::spawn(move || {
            let _guard = cloned.lock().expect("lock transcript");
            panic!("poison transcript mutex for test");
        });
        let _ = join.join();
        let mut agent = test_agent();
        let previous_route = ModelRoute::new("primary", "gpt-5.5");
        let selected_route = ModelRoute::new("expert", "claude-4");
        agent.set_primary_route(previous_route.clone());
        let mut persisted_routes = Vec::new();

        let error = persist_and_apply_model_route_with(
            &mut agent,
            &transcript,
            selected_route.clone(),
            prepared_route(selected_route.clone()),
            |route| {
                persisted_routes.push(route.clone());
                Ok(())
            },
        )
        .expect_err("poisoned transcript must prevent route application");

        assert!(!error.fast_mode_auto_disabled());
        assert_eq!(persisted_routes, [selected_route, previous_route]);
        assert_eq!(agent.model(), "gpt-5.5");
        assert_eq!(
            agent.primary_route(),
            Some(&ModelRoute::new("primary", "gpt-5.5"))
        );
    }

    #[test]
    fn model_route_persistence_failure_leaves_live_agent_unchanged() {
        let (_base_dir, transcript, _session_id) = temp_transcript();
        let cloned = Arc::clone(&transcript);
        let join = std::thread::spawn(move || {
            let _guard = cloned.lock().expect("lock transcript");
            panic!("poison transcript mutex for test");
        });
        let _ = join.join();
        let mut agent = test_agent();
        agent.set_primary_route(ModelRoute::new("primary", "gpt-5.5"));

        let prepared_route = crate::agent::PreparedPrimaryRoute::new(
            Client::with_config(
                OpenAIConfig::new()
                    .with_api_base("http://127.0.0.1:9/v1")
                    .with_api_key("expert-key"),
            ),
            ModelRoute::new("expert", "claude-4"),
            crate::config::ApiProtocol::Responses,
            std::collections::HashMap::new(),
            std::collections::HashMap::new(),
            crate::config::RetryConfig::default(),
        );
        let error = apply_model_route_with(
            &mut agent,
            &transcript,
            ModelRoute::new("expert", "claude-4"),
            prepared_route,
        )
        .expect_err("poisoned transcript must fail before applying the route");

        assert!(!error.fast_mode_auto_disabled());
        assert_eq!(agent.model(), "gpt-5.5");
        assert_eq!(
            agent.primary_route(),
            Some(&ModelRoute::new("primary", "gpt-5.5"))
        );
    }

    #[test]
    fn model_route_fast_mode_auto_disable_is_memory_only() {
        let (base_dir, transcript, session_id) = temp_transcript();
        let fast_mode_path = base_dir.join("letcode.toml");
        std::fs::write(
            &fast_mode_path,
            r#"active_provider = "primary"

[providers.primary]
base_url = "https://primary.invalid/v1"
api_key = "primary-key"
protocol = "responses"
[providers.primary.models."gpt-5.5"]
"#,
        )
        .expect("write Fast Mode config");
        let before = std::fs::read_to_string(&fast_mode_path).expect("read config before apply");
        let mut agent = test_agent();
        agent.set_primary_route(ModelRoute::new("primary", "gpt-5.5"));
        agent.set_fast_mode(crate::fast_mode::FastMode::load(&fast_mode_path, true));
        let cloned = Arc::clone(&transcript);
        let join = std::thread::spawn(move || {
            let _guard = cloned.lock().expect("lock transcript");
            panic!("poison transcript mutex for test");
        });
        let _ = join.join();

        let error = apply_model_route_with(
            &mut agent,
            &transcript,
            ModelRoute::new("expert", "claude-4"),
            prepared_route(ModelRoute::new("expert", "claude-4")),
        )
        .expect_err("poisoned transcript must prevent route application");

        assert!(error.fast_mode_auto_disabled());
        assert!(!agent.fast_mode_enabled());
        assert_eq!(
            std::fs::read_to_string(&fast_mode_path).expect("read config after apply"),
            before
        );
        assert_eq!(agent.model(), "gpt-5.5");
        assert_eq!(
            agent.primary_route(),
            Some(&ModelRoute::new("primary", "gpt-5.5"))
        );
        assert!(
            crate::transcript::read_records(base_dir.join(format!("{session_id}.jsonl")))
                .expect("read records")
                .is_empty()
        );
    }

    #[test]
    fn model_route_transcript_failure_follows_fast_mode_reconciliation_without_route_apply() {
        let (base_dir, transcript, session_id) = temp_transcript();
        let fast_mode_path = base_dir.join("letcode.toml");
        let mut agent = configured_agent(&fast_mode_path);
        let cloned = Arc::clone(&transcript);
        let join = std::thread::spawn(move || {
            let _guard = cloned.lock().expect("lock transcript");
            panic!("poison transcript mutex for test");
        });
        let _ = join.join();

        let error = apply_model_route_with(
            &mut agent,
            &transcript,
            ModelRoute::new("expert", "claude-4"),
            prepared_route(ModelRoute::new("expert", "claude-4")),
        )
        .expect_err("transcript failure must prevent route application");

        assert!(error.fast_mode_auto_disabled());
        assert!(!agent.fast_mode_enabled());
        assert_eq!(agent.model(), "gpt-5.5");
        assert_eq!(
            agent.primary_route(),
            Some(&ModelRoute::new("primary", "gpt-5.5"))
        );
        assert!(
            crate::transcript::read_records(base_dir.join(format!("{session_id}.jsonl")))
                .expect("read records")
                .is_empty()
        );
    }
}
