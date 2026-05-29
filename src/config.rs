use anyhow::{Context, Result, anyhow, bail};
use indexmap::IndexMap;
use serde::Deserialize;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use crate::permission::PermissionMode;

const DEFAULT_CONFIG_HOME_RELATIVE_PATH: &str = ".config/letcode/letcode.toml";
const DEFAULT_OPENAI_BASE_URL: &str = "https://api.openai.com/v1";
const DEFAULT_MAX_ITERATIONS: usize = 64;
const DEFAULT_MAX_TOOL_CALLS: usize = 128;
const DEFAULT_SESSIONS_DIR: &str = "sessions";
const DEFAULT_LOG_FILE: &str = "logs/combined.log";

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct AppConfig {
    pub config_path: PathBuf,
    pub config_dir: PathBuf,
    pub active_provider: String,
    pub global: GlobalConfig,
    pub permissions: PermissionsConfig,
    pub providers: IndexMap<String, ProviderConfig>,
}

impl AppConfig {
    pub fn load() -> Result<Self> {
        Self::load_from_path(default_config_path()?)
    }

    pub fn load_from_path(path: impl AsRef<Path>) -> Result<Self> {
        let config_path = path.as_ref().to_path_buf();
        if !config_path.exists() {
            bail!(missing_config_message(&config_path));
        }
        let config_text = fs::read_to_string(&config_path)
            .with_context(|| format!("failed to read config file {}", config_path.display()))?;
        let raw: RawAppConfig = toml::from_str(&config_text)
            .with_context(|| format!("failed to parse config file {}", config_path.display()))?;
        let config_dir = config_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));

        if raw.providers.is_empty() {
            bail!("config must define at least one provider under [providers]");
        }

        let active_provider = required_non_empty(
            "active_provider",
            raw.active_provider.unwrap_or_else(|| {
                raw.providers
                    .keys()
                    .next()
                    .expect("providers should contain at least one entry")
                    .clone()
            }),
        )?;

        let providers = raw
            .providers
            .into_iter()
            .map(|(name, provider)| build_provider_config(&name, provider))
            .collect::<Result<IndexMap<_, _>>>()?;

        if !providers.contains_key(&active_provider) {
            bail!(
                "active_provider '{}' does not exist under [providers]",
                active_provider
            );
        }

        let raw_global = raw.global.unwrap_or_default();
        let global = GlobalConfig {
            max_iterations: positive_usize(
                "global.max_iterations",
                raw_global.max_iterations.unwrap_or(DEFAULT_MAX_ITERATIONS),
            )?,
            max_tool_calls: positive_usize(
                "global.max_tool_calls",
                raw_global.max_tool_calls.unwrap_or(DEFAULT_MAX_TOOL_CALLS),
            )?,
            sessions_dir: resolve_relative_path(
                &config_dir,
                &required_non_empty(
                    "global.sessions_dir",
                    raw_global
                        .sessions_dir
                        .unwrap_or_else(|| DEFAULT_SESSIONS_DIR.to_string()),
                )?,
            ),
            log_file: resolve_relative_path(
                &config_dir,
                &required_non_empty(
                    "global.log_file",
                    raw_global
                        .log_file
                        .unwrap_or_else(|| DEFAULT_LOG_FILE.to_string()),
                )?,
            ),
        };

        let permissions = PermissionsConfig {
            mode: raw.permissions.unwrap_or_default().mode.unwrap_or_default(),
        };

        Ok(Self {
            config_path,
            config_dir,
            active_provider,
            global,
            permissions,
            providers,
        })
    }

    pub fn active_provider(&self) -> (&str, &ProviderConfig) {
        let provider = self
            .providers
            .get(&self.active_provider)
            .expect("active provider should be validated at load time");
        (&self.active_provider, provider)
    }

    pub fn active_provider_api_key_env_var(&self) -> String {
        provider_env_var(&self.active_provider, "API_KEY")
    }

    pub fn active_provider_model_label(&self, model_id: &str) -> String {
        let (_, provider) = self.active_provider();
        provider.model_label(model_id)
    }
}

#[derive(Debug, Clone)]
pub struct GlobalConfig {
    pub max_iterations: usize,
    pub max_tool_calls: usize,
    pub sessions_dir: PathBuf,
    pub log_file: PathBuf,
}

#[derive(Debug, Clone)]
pub struct PermissionsConfig {
    pub mode: PermissionMode,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ProviderConfig {
    pub base_url: String,
    pub api_key: String,
    pub default_model: String,
    pub models: IndexMap<String, ModelConfig>,
}

impl ProviderConfig {
    pub fn model_label(&self, model_id: &str) -> String {
        self.models
            .get(model_id)
            .and_then(|model| model.display_name.as_deref())
            .unwrap_or(model_id)
            .to_string()
    }

    pub fn has_model(&self, model_id: &str) -> bool {
        self.models.contains_key(model_id)
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub display_name: Option<String>,
    pub context_window: Option<u64>,
    pub max_output_tokens: Option<u64>,
    pub supports_tools: bool,
    pub supports_reasoning: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawAppConfig {
    #[serde(default)]
    active_provider: Option<String>,
    #[serde(default)]
    global: Option<RawGlobalConfig>,
    #[serde(default)]
    permissions: Option<RawPermissionsConfig>,
    #[serde(default)]
    providers: IndexMap<String, RawProviderConfig>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawGlobalConfig {
    max_iterations: Option<usize>,
    max_tool_calls: Option<usize>,
    sessions_dir: Option<String>,
    log_file: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawPermissionsConfig {
    mode: Option<PermissionMode>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawProviderConfig {
    base_url: Option<String>,
    api_key: Option<String>,
    default_model: Option<String>,
    #[serde(default)]
    models: IndexMap<String, RawModelConfig>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawModelConfig {
    #[serde(default, alias = "name")]
    display_name: Option<String>,
    context_window: Option<u64>,
    max_output_tokens: Option<u64>,
    #[serde(default)]
    supports_tools: bool,
    #[serde(default)]
    supports_reasoning: bool,
}

fn build_provider_config(name: &str, raw: RawProviderConfig) -> Result<(String, ProviderConfig)> {
    let name = validate_identifier("providers key", name)?;

    if raw.models.is_empty() {
        bail!(
            "provider '{}' must define at least one model under [providers.{}.models]",
            name,
            name
        );
    }

    let base_url = env_override(name, "BASE_URL")
        .or(raw.base_url.map(|value| value.trim().to_string()))
        .or_else(|| {
            if name.eq_ignore_ascii_case("openai") {
                Some(DEFAULT_OPENAI_BASE_URL.to_string())
            } else {
                None
            }
        })
        .ok_or_else(|| anyhow!("provider '{}' is missing base_url", name))?;
    let base_url = required_non_empty(&format!("providers.{name}.base_url"), base_url)?;

    let api_key = env_override(name, "API_KEY")
        .or(raw.api_key.map(|value| value.trim().to_string()))
        .unwrap_or_default();

    let default_model = required_non_empty(
        &format!("providers.{name}.default_model"),
        raw.default_model.unwrap_or_else(|| {
            raw.models
                .keys()
                .next()
                .expect("models should contain at least one entry")
                .clone()
        }),
    )?;

    let models = raw
        .models
        .into_iter()
        .map(|(model_id, model)| normalize_model_config(name, &model_id, model))
        .collect::<Result<IndexMap<_, _>>>()?;

    if !models.contains_key(&default_model) {
        bail!(
            "provider '{}' default_model '{}' is not defined under [providers.{}.models]",
            name,
            default_model,
            name
        );
    }

    Ok((
        name.to_string(),
        ProviderConfig {
            base_url,
            api_key,
            default_model,
            models,
        },
    ))
}

fn normalize_model_config(
    provider_name: &str,
    model_id: &str,
    raw: RawModelConfig,
) -> Result<(String, ModelConfig)> {
    let model_id = validate_identifier(&format!("providers.{provider_name}.models key"), model_id)?
        .to_string();
    let display_name = raw
        .display_name
        .map(|value| {
            required_non_empty(
                &format!("providers.{provider_name}.models.{model_id}.display_name"),
                value,
            )
        })
        .transpose()?;

    Ok((
        model_id,
        ModelConfig {
            display_name,
            context_window: raw.context_window,
            max_output_tokens: raw.max_output_tokens,
            supports_tools: raw.supports_tools,
            supports_reasoning: raw.supports_reasoning,
        },
    ))
}

fn env_override(provider_name: &str, suffix: &str) -> Option<String> {
    let key = provider_env_var(provider_name, suffix);
    env::var(&key)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn provider_env_var(provider_name: &str, suffix: &str) -> String {
    let normalized = provider_name
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_uppercase()
            } else {
                '_'
            }
        })
        .collect::<String>();
    format!("{}_{}", normalized, suffix)
}

fn resolve_relative_path(base_dir: &Path, path: &str) -> PathBuf {
    let path = PathBuf::from(path);
    if path.is_absolute() {
        path
    } else {
        base_dir.join(path)
    }
}

fn validate_identifier<'a>(label: &str, value: &'a str) -> Result<&'a str> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        bail!("{} cannot be empty", label);
    }
    if trimmed != value {
        bail!("{} must not have leading or trailing whitespace", label);
    }
    Ok(value)
}

fn required_non_empty(label: &str, value: String) -> Result<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        bail!("{} cannot be empty", label);
    }
    Ok(trimmed.to_string())
}

fn positive_usize(label: &str, value: usize) -> Result<usize> {
    if value == 0 {
        bail!("{} must be greater than 0", label);
    }
    Ok(value)
}

fn missing_config_message(path: &Path) -> String {
    format!(
        "config file not found: {}\n\nCreate it with at least:\n\nactive_provider = \"openai\"\n\n[global]\nmax_iterations = 64\nmax_tool_calls = 128\nsessions_dir = \"sessions\"\nlog_file = \"logs/combined.log\"\n\n[permissions]\nmode = \"default\"\n\n[providers.openai]\napi_key = \"YOUR_API_KEY\"\nbase_url = \"https://api.openai.com/v1\"\ndefault_model = \"gpt-5.5\"\n\n[providers.openai.models.\"gpt-5.5\"]\ndisplay_name = \"GPT-5.5\"\n",
        path.display()
    )
}

fn default_config_path() -> Result<PathBuf> {
    let home = env::var_os("HOME").ok_or_else(|| anyhow!("HOME is not set"))?;
    Ok(PathBuf::from(home).join(DEFAULT_CONFIG_HOME_RELATIVE_PATH))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::{Mutex, OnceLock};
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn resolves_relative_paths_from_config_dir() {
        let _guard = lock_env();
        let path = write_temp_config(
            r#"
            [global]
            sessions_dir = "sessions"
            log_file = "logs/combined.log"

            [providers.openai]
            api_key = "config-key"

            [providers.openai.models."gpt-5.5"]
            name = "GPT-5.5"
            "#,
        );

        let config = AppConfig::load_from_path(&path).expect("config should load");
        let config_dir = path.parent().expect("config file should have parent");

        assert_eq!(config.global.sessions_dir, config_dir.join("sessions"));
        assert_eq!(config.global.log_file, config_dir.join("logs/combined.log"));
    }

    #[test]
    fn defaults_to_first_provider_and_first_model() {
        let _guard = lock_env();
        let path = write_temp_config(
            r#"
            [providers.openai]
            api_key = "config-key"

            [providers.openai.models."gpt-5.5"]
            name = "GPT-5.5"
            "#,
        );

        let config = AppConfig::load_from_path(&path).expect("config should load");
        let (provider_name, provider) = config.active_provider();

        assert_eq!(provider_name, "openai");
        assert_eq!(provider.default_model, "gpt-5.5");
    }

    #[test]
    fn rejects_unknown_fields() {
        let _guard = lock_env();
        let path = write_temp_config(
            r#"
            active_provider = "openai"
            unexpected = true

            [providers.openai]
            api_key = "config-key"

            [providers.openai.models."gpt-5.5"]
            name = "GPT-5.5"
            "#,
        );

        let error = AppConfig::load_from_path(&path).expect_err("load should fail");
        assert!(
            error
                .chain()
                .any(|cause| cause.to_string().contains("unknown field"))
        );
    }

    #[test]
    fn rejects_zero_limits_and_empty_strings() {
        let _guard = lock_env();
        let path = write_temp_config(
            r#"
            active_provider = "openai"

            [global]
            max_iterations = 0

            [providers.openai]
            api_key = "config-key"

            [providers.openai.models."gpt-5.5"]
            name = "GPT-5.5"
            "#,
        );

        let error = AppConfig::load_from_path(&path).expect_err("load should fail");
        assert!(
            error
                .to_string()
                .contains("global.max_iterations must be greater than 0")
        );

        let path = write_temp_config(
            r#"
            active_provider = "openai"

            [providers.openai]
            api_key = "config-key"
            default_model = "   "

            [providers.openai.models."gpt-5.5"]
            name = "GPT-5.5"
            "#,
        );

        let error = AppConfig::load_from_path(&path).expect_err("load should fail");
        assert!(
            error
                .to_string()
                .contains("providers.openai.default_model cannot be empty")
        );
    }

    #[test]
    fn missing_config_error_includes_example() {
        let _guard = lock_env();
        let path = env::temp_dir().join("letcode-missing-config-test.toml");
        let _ = fs::remove_file(&path);

        let error = AppConfig::load_from_path(&path).expect_err("load should fail");
        let message = error.to_string();
        assert!(message.contains("config file not found"));
        assert!(message.contains("[providers.openai]"));
    }

    #[test]
    fn provider_env_overrides_config_values() {
        let _guard = lock_env();
        let path = write_temp_config(
            r#"
            active_provider = "openai"

            [providers.openai]
            base_url = "https://example.invalid/v1"
            api_key = "config-key"

            [providers.openai.models."gpt-5.5"]
            name = "GPT-5.5"
            "#,
        );

        unsafe {
            env::set_var("OPENAI_BASE_URL", "https://env.example/v1");
            env::set_var("OPENAI_API_KEY", "env-key");
        }

        let config = AppConfig::load_from_path(&path).expect("config should load");
        let (_, provider) = config.active_provider();

        assert_eq!(provider.base_url, "https://env.example/v1");
        assert_eq!(provider.api_key, "env-key");

        unsafe {
            env::remove_var("OPENAI_BASE_URL");
            env::remove_var("OPENAI_API_KEY");
        }
    }

    #[test]
    fn errors_when_active_provider_is_missing() {
        let _guard = lock_env();
        let path = write_temp_config(
            r#"
            active_provider = "missing"

            [providers.openai]
            api_key = "config-key"

            [providers.openai.models."gpt-5.5"]
            name = "GPT-5.5"
            "#,
        );

        let error = AppConfig::load_from_path(&path).expect_err("load should fail");
        assert!(error.to_string().contains("active_provider 'missing'"));
    }

    fn write_temp_config(contents: &str) -> PathBuf {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time before unix epoch")
            .as_nanos();
        let base = env::temp_dir().join(format!("letcode-config-test-{timestamp}"));
        fs::create_dir_all(&base).expect("temp config dir should be created");
        let path = base.join("letcode.toml");
        fs::write(&path, contents).expect("temp config should be written");
        path
    }

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    fn lock_env() -> std::sync::MutexGuard<'static, ()> {
        env_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}
