use crate::permission::PermissionMode;
use crate::request_builder::{
    ModelReasoningEffort, ModelReasoningSummary, ModelRequestMetadata, ModelTextVerbosity,
};
use anyhow::{Context, Result, anyhow, bail};
use async_openai::Client;
use async_openai::config::OpenAIConfig;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
#[derive(Default)]
pub enum ApiProtocol {
    #[default]
    Responses,
    Completions,
    Anthropic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ProviderAuthMode {
    ApiKey,
    Bearer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PromptCacheRetention {
    InMemory,
    #[serde(rename = "24h")]
    TwentyFourHours,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PromptCacheConfig {
    pub enabled: bool,
    pub retention: Option<PromptCacheRetention>,
    pub namespace: Option<String>,
}

const DEFAULT_CONFIG_HOME_RELATIVE_PATH: &str = ".config/letcode/letcode.toml";
const DEFAULT_OPENAI_BASE_URL: &str = "https://api.openai.com/v1";
const DEFAULT_MCP_TIMEOUT_MS: u64 = 5_000;
const DEFAULT_SESSIONS_DIR: &str = "sessions";
const DEFAULT_LOG_FILE: &str = "logs/combined.log";
const MAX_RETRY_ATTEMPTS: usize = 50;
const MAX_RECOVERY_ATTEMPTS: usize = 10;
mod persistence;

// Kept re-exported for API compatibility; persist_mcp_server_enabled is the
// only one referenced by production code today.
use persistence::acquire_config_read_lock;
pub(crate) use persistence::replace_file;
#[allow(unused_imports)]
pub use persistence::{
    persist_expert_allowed_models, persist_expert_model_route, persist_mcp_server_enabled,
    persist_primary_model_route,
};

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct AppConfig {
    pub config_path: PathBuf,
    pub config_dir: PathBuf,
    pub fast_mode_enabled: bool,
    pub active_provider: String,
    pub global: GlobalConfig,
    pub agents: AgentsConfig,
    pub permissions: PermissionsConfig,
    pub tools: ToolsConfig,
    pub experiments: ExperimentsConfig,
    pub mcp: IndexMap<String, McpServerConfig>,
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
        // Lock the canonical target rather than the directory entry. Writers
        // replace the target atomically, so this keeps reads compatible with
        // the existing writer lock across an atomic rename (and symlinks).
        let config_target = fs::canonicalize(&config_path)
            .with_context(|| format!("failed to resolve config file {}", config_path.display()))?;
        let _lock = acquire_config_read_lock(&config_target)?;
        let config_text = fs::read_to_string(&config_target)
            .with_context(|| format!("failed to read config file {}", config_path.display()))?;
        Self::load_from_str_at_path(&config_path, &config_text)
    }

    fn load_from_str_at_path(config_path: &Path, config_text: &str) -> Result<Self> {
        let config_path = config_path.to_path_buf();
        let raw: RawAppConfig = toml::from_str(config_text)
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

        let raw_global = raw.global.unwrap_or_default();
        let compaction = build_compaction_config(raw_global.compaction.unwrap_or_default())?;
        let global = GlobalConfig {
            max_iterations: optional_positive_usize(
                "global.max_iterations",
                raw_global.max_iterations,
            )?,
            max_tool_calls: optional_positive_usize(
                "global.max_tool_calls",
                raw_global.max_tool_calls,
            )?,
            tool_timeout_secs: optional_positive_u64(
                "global.tool_timeout_secs",
                raw_global.tool_timeout_secs,
            )?
            .or(Some(60)),
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
            compaction,
            retry: build_retry_config(raw_global.retry.unwrap_or_default(), "global.retry")?,
        };

        let providers = raw
            .providers
            .into_iter()
            .map(|(name, provider)| build_provider_config(&name, provider, &global.retry))
            .collect::<Result<IndexMap<_, _>>>()?;

        if !providers.contains_key(&active_provider) {
            bail!(
                "active_provider '{}' does not exist under [providers]",
                active_provider
            );
        }

        let permissions = PermissionsConfig {
            mode: raw.permissions.unwrap_or_default().mode.unwrap_or_default(),
        };
        let tools = build_tools_config(raw.tools.unwrap_or_default())?;
        let experiments = build_experiments_config(raw.experiments.unwrap_or_default())?;
        let agents =
            build_agents_config(raw.agents.unwrap_or_default(), &active_provider, &providers)?;
        let mcp = raw
            .mcp
            .into_iter()
            .map(|(name, server)| build_mcp_server_config(&name, server))
            .collect::<Result<IndexMap<_, _>>>()?;

        Ok(Self {
            config_path,
            config_dir,
            fast_mode_enabled: raw.fast_mode.unwrap_or(false),
            active_provider,
            global,
            agents,
            permissions,
            tools,
            experiments,
            mcp,
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

    pub fn active_route(&self) -> ModelRoute {
        let (_, provider) = self.active_provider();
        ModelRoute::new(self.active_provider.clone(), provider.default_model.clone())
    }

    #[allow(dead_code)]
    pub fn provider_for_route(&self, route: &ModelRoute) -> Result<&ProviderConfig> {
        self.providers.get(&route.provider).ok_or_else(|| {
            anyhow!(
                "provider '{}' is not defined under [providers]",
                route.provider
            )
        })
    }

    #[allow(dead_code)]
    pub fn resolve_route(&self, route: &ModelRoute) -> Result<&ProviderConfig> {
        let provider = self.provider_for_route(route)?;
        if !provider.has_model(&route.model) {
            bail!(
                "model '{}' is not defined under [providers.{}.models]",
                route.model,
                route.provider
            );
        }
        Ok(provider)
    }

    pub fn model_route_for(&self, agent_name: &str) -> Option<&ModelRoute> {
        self.agents.route_for(agent_name)
    }

    #[allow(dead_code)]
    pub fn expert_route_for(&self, agent_name: &str) -> Option<ModelRoute> {
        if !crate::delegation::supported_agent_names().any(|name| name == agent_name) {
            return None;
        }
        self.model_route_for(agent_name)
            .map(|route| {
                if self.agents.follows_active_provider(agent_name) {
                    ModelRoute::new(self.active_provider.clone(), route.model.clone())
                } else {
                    route.clone()
                }
            })
            .or_else(|| Some(self.active_route()))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelRoute {
    pub provider: String,
    pub model: String,
}

impl ModelRoute {
    pub fn new(provider: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            provider: provider.into(),
            model: model.into(),
        }
    }

    pub fn display_name(&self) -> String {
        format!("{}/{}", self.provider, self.model)
    }

    pub fn parse(value: &str) -> Result<Self> {
        let value = value.trim();
        let Some((provider, model)) = value.split_once('/') else {
            bail!("model route '{value}' must use provider/model form");
        };
        let provider = required_non_empty("model route provider", provider.to_string())?;
        let model = required_non_empty("model route model", model.to_string())?;
        if model.contains('/') {
            bail!("model route '{value}' must use provider/model form");
        }
        Ok(Self::new(provider, model))
    }

    pub fn build_client(&self, provider: &ProviderConfig) -> Client<OpenAIConfig> {
        let effective_protocol = provider
            .models
            .get(&self.model)
            .map(|model| model.protocol)
            .unwrap_or(provider.protocol);
        let mut config = OpenAIConfig::new()
            .with_api_base(provider.base_url.clone())
            .with_api_key(provider.api_key.clone());
        if effective_protocol == ApiProtocol::Anthropic {
            config = match provider.auth_mode {
                ProviderAuthMode::ApiKey => config
                    .with_header("x-api-key", provider.api_key.as_str())
                    .expect("provider api key is a valid header value"),
                ProviderAuthMode::Bearer => config,
            };
        }
        Client::with_config(config)
    }
}

#[derive(Debug, Clone, Default)]
pub struct AgentsConfig {
    pub explorer: AgentConfig,
    pub fixer: AgentConfig,
    pub oracle: AgentConfig,
    pub designer: AgentConfig,
    pub librarian: AgentConfig,
    pub general: AgentConfig,
    pub reviewer: AgentConfig,
}

impl AgentsConfig {
    pub fn route_for(&self, agent_name: &str) -> Option<&ModelRoute> {
        self.config_for(agent_name)
            .and_then(|config| config.route.as_ref())
    }

    pub fn allowed_models_for(&self, agent_name: &str) -> Option<&[ModelRoute]> {
        self.config_for(agent_name)
            .map(|config| config.allowed_models.as_slice())
    }

    #[allow(dead_code)]
    pub fn follows_active_provider(&self, agent_name: &str) -> bool {
        self.config_for(agent_name)
            .is_some_and(|config| config.follows_active_provider)
    }

    fn config_for(&self, agent_name: &str) -> Option<&AgentConfig> {
        match agent_name {
            "explorer" => Some(&self.explorer),
            "fixer" => Some(&self.fixer),
            "oracle" => Some(&self.oracle),
            "designer" => Some(&self.designer),
            "librarian" => Some(&self.librarian),
            "general" => Some(&self.general),
            "reviewer" => Some(&self.reviewer),
            _ => None,
        }
    }

    #[allow(dead_code)]
    pub fn model_for(&self, agent_name: &str) -> Option<&str> {
        self.route_for(agent_name).map(|route| route.model.as_str())
    }
}

#[derive(Debug, Clone, Default)]
pub struct AgentConfig {
    pub route: Option<ModelRoute>,
    pub allowed_models: Vec<ModelRoute>,
    #[allow(dead_code)]
    pub follows_active_provider: bool,
}

#[derive(Debug, Clone)]
pub struct GlobalConfig {
    pub max_iterations: Option<usize>,
    pub max_tool_calls: Option<usize>,
    pub tool_timeout_secs: Option<u64>,
    pub sessions_dir: PathBuf,
    pub log_file: PathBuf,
    pub compaction: CompactionConfig,
    pub retry: RetryConfig,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CompactionConfig {
    /// Tokens to retain at the end of history. When absent, use the selected
    /// model's active input budget.
    pub preserve_recent_tokens: Option<u64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RetryConfig {
    pub enabled: bool,
    pub max_attempts: usize,
    pub max_recovery_attempts: usize,
    pub initial_delay_secs: u64,
    pub backoff_multiplier: f32,
    pub jitter_secs: u64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_attempts: 50,
            max_recovery_attempts: 3,
            initial_delay_secs: 1,
            backoff_multiplier: 2.0,
            jitter_secs: 1,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PermissionsConfig {
    pub mode: PermissionMode,
}

#[derive(Debug, Clone, Default)]
pub struct ToolsConfig {
    pub parallelism: IndexMap<String, crate::tool::ToolParallelism>,
}

/// Anchored-bootstrap experiment settings. Defaults keep the experiment
/// disabled; nothing in the agent pipeline consults it unless enabled.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ExperimentsConfig {
    pub anchored_bootstrap: AnchoredBootstrapConfig,
}

/// Promotion signal semantics for the anchored bootstrap experiment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PromoteOn {
    /// First durable `tool/call` OR first `assistant/message`, whichever first.
    #[default]
    Either,
    /// Only a durable tool call promotes the session.
    ToolCall,
    /// Only a durable assistant message promotes the session.
    AssistantMessage,
}

/// Two-phase "anchored bootstrap" experiment: the first model request sees a
/// Minimal-aligned tool pair (alias names) and no injected context; after the
/// first durable promotion signal the session restores the full catalog.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct AnchoredBootstrapConfig {
    pub enabled: bool,
    /// Model IDs this experiment applies to (empty when disabled).
    pub models: Vec<String>,
    pub promote_on: PromoteOn,
    /// Core work set exposed after a compaction, before re-promotion.
    pub compaction_tools: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct McpServerConfig {
    pub enabled: bool,
    pub timeout_ms: u64,
    pub transport: McpTransportConfig,
}

#[derive(Debug, Clone)]
pub enum McpTransportConfig {
    Local(McpLocalServerConfig),
    Remote(McpRemoteServerConfig),
}

#[derive(Debug, Clone)]
pub struct McpLocalServerConfig {
    pub command: Vec<String>,
    pub environment: IndexMap<String, String>,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct McpRemoteServerConfig {
    pub url: String,
    pub headers: IndexMap<String, String>,
    pub oauth: bool,
}

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
pub struct ProviderConfig {
    pub base_url: String,
    pub api_key: String,
    pub auth_mode: ProviderAuthMode,
    pub protocol: ApiProtocol,
    pub default_model: String,
    pub retry: Option<RetryConfig>,
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
#[derive(Debug, Clone, PartialEq)]
pub struct ModelConfig {
    pub display_name: Option<String>,
    pub protocol: ApiProtocol,
    pub anthropic_thinking: crate::request_builder::AnthropicThinkingConfig,
    pub cache_control: bool,
    pub context_window: Option<u64>,
    pub effective_input_limit_tokens: Option<u64>,
    pub max_output_tokens: Option<u64>,
    pub supports_tools: bool,
    pub supports_reasoning: bool,
    pub reasoning_effort: Option<ModelReasoningEffort>,
    pub reasoning_efforts: Vec<ModelReasoningEffort>,
    pub reasoning_summary: Option<ModelReasoningSummary>,
    pub text_verbosity: Option<ModelTextVerbosity>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub prompt_cache: PromptCacheConfig,
    pub parallel_tool_calls: bool,
}

impl ModelConfig {
    pub fn request_metadata(&self) -> ModelRequestMetadata {
        ModelRequestMetadata {
            context_window: self.context_window,
            effective_input_limit_tokens: self.effective_input_limit_tokens,
            max_output_tokens: self.max_output_tokens,
            supports_tools: self.supports_tools,
            supports_reasoning: self.supports_reasoning,
            reasoning_effort: self.reasoning_effort.clone(),
            reasoning_efforts: self.reasoning_efforts.clone(),
            reasoning_summary: self.reasoning_summary,
            text_verbosity: self.text_verbosity,
            temperature: self.temperature,
            top_p: self.top_p,
            prompt_cache: self.prompt_cache.clone(),
            parallel_tool_calls: self.parallel_tool_calls,
            fast_mode: false,
            anthropic_thinking: self.anthropic_thinking,
            cache_control: self.cache_control,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawAppConfig {
    #[serde(default)]
    fast_mode: Option<bool>,
    #[serde(default)]
    active_provider: Option<String>,
    #[serde(default)]
    global: Option<RawGlobalConfig>,
    #[serde(default)]
    agents: Option<RawAgentsConfig>,
    #[serde(default)]
    permissions: Option<RawPermissionsConfig>,
    #[serde(default)]
    tools: Option<RawToolsConfig>,
    #[serde(default)]
    experiments: Option<RawExperimentsConfig>,
    #[serde(default)]
    mcp: IndexMap<String, RawMcpServerConfig>,
    #[serde(default)]
    providers: IndexMap<String, RawProviderConfig>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawGlobalConfig {
    max_iterations: Option<usize>,
    max_tool_calls: Option<usize>,
    tool_timeout_secs: Option<u64>,
    sessions_dir: Option<String>,
    log_file: Option<String>,
    compaction: Option<RawCompactionConfig>,
    retry: Option<RawRetryConfig>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawCompactionConfig {
    preserve_recent_tokens: Option<u64>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawRetryConfig {
    enabled: Option<bool>,
    max_attempts: Option<usize>,
    max_recovery_attempts: Option<usize>,
    initial_delay_secs: Option<u64>,
    backoff_multiplier: Option<f32>,
    jitter_secs: Option<u64>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawAgentsConfig {
    explorer: Option<RawAgentConfig>,
    fixer: Option<RawAgentConfig>,
    oracle: Option<RawAgentConfig>,
    designer: Option<RawAgentConfig>,
    librarian: Option<RawAgentConfig>,
    general: Option<RawAgentConfig>,
    reviewer: Option<RawAgentConfig>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawAgentConfig {
    provider: Option<String>,
    model: Option<String>,
    #[serde(default)]
    allowed_models: Vec<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawPermissionsConfig {
    mode: Option<PermissionMode>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawToolsConfig {
    #[serde(default)]
    parallelism: IndexMap<String, crate::tool::ToolParallelism>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawExperimentsConfig {
    anchored_bootstrap: Option<RawAnchoredBootstrapConfig>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawAnchoredBootstrapConfig {
    enabled: Option<bool>,
    models: Option<Vec<String>>,
    promote_on: Option<String>,
    compaction_tools: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawMcpServerConfig {
    #[serde(rename = "type")]
    kind: RawMcpServerKind,
    enabled: Option<bool>,
    timeout: Option<u64>,
    command: Option<Vec<String>>,
    #[serde(default, alias = "env")]
    environment: IndexMap<String, String>,
    url: Option<String>,
    #[serde(default)]
    headers: IndexMap<String, String>,
    oauth: Option<bool>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
enum RawMcpServerKind {
    Local,
    Remote,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawProviderConfig {
    base_url: Option<String>,
    api_key: Option<String>,
    auth_mode: Option<ProviderAuthMode>,
    protocol: Option<ApiProtocol>,
    default_model: Option<String>,
    retry: Option<RawRetryConfig>,
    #[serde(default)]
    models: IndexMap<String, RawModelConfig>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawModelConfig {
    #[serde(default, alias = "name")]
    display_name: Option<String>,
    protocol: Option<ApiProtocol>,
    anthropic_thinking: Option<crate::request_builder::AnthropicThinkingConfig>,
    #[serde(default)]
    cache_control: bool,
    context_window: Option<u64>,
    effective_input_limit_tokens: Option<u64>,
    max_output_tokens: Option<u64>,
    // Transitional defaults: if omitted, assume true to avoid surprising
    // tool/reasoning disablement for existing configs.
    supports_tools: Option<bool>,
    supports_reasoning: Option<bool>,
    reasoning_effort: Option<ModelReasoningEffort>,
    #[serde(default, alias = "reasoning_levels")]
    reasoning_efforts: Vec<ModelReasoningEffort>,
    reasoning_summary: Option<ModelReasoningSummary>,
    text_verbosity: Option<ModelTextVerbosity>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    parallel_tool_calls: Option<bool>,
    #[serde(default)]
    prompt_cache: RawPromptCacheConfig,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawPromptCacheConfig {
    enabled: Option<bool>,
    retention: Option<PromptCacheRetention>,
    namespace: Option<String>,
}

fn build_provider_config(
    name: &str,
    raw: RawProviderConfig,
    global_retry: &RetryConfig,
) -> Result<(String, ProviderConfig)> {
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

    let protocol = match raw.protocol {
        Some(protocol) => protocol,
        None if name.eq_ignore_ascii_case("openai") => ApiProtocol::Responses,
        None => bail!(
            "provider '{}' must set protocol to 'responses', 'completions', or 'anthropic'",
            name
        ),
    };
    let has_anthropic_model_override = raw
        .models
        .values()
        .any(|model| model.protocol == Some(ApiProtocol::Anthropic));
    let auth_mode = match raw.auth_mode {
        Some(mode) => mode,
        None if protocol == ApiProtocol::Anthropic => bail!(
            "provider '{}' with protocol = 'anthropic' requires auth_mode = 'api-key' or 'bearer'",
            name
        ),
        None if has_anthropic_model_override => bail!(
            "provider '{}' with a model protocol = 'anthropic' requires auth_mode = 'api-key' or 'bearer'",
            name
        ),
        None => ProviderAuthMode::Bearer,
    };

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
        .map(|(model_id, model)| normalize_model_config(name, &model_id, protocol, model))
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
            auth_mode,
            protocol,
            default_model,
            retry: raw
                .retry
                .map(|retry| {
                    build_retry_config_overlay(
                        retry,
                        global_retry,
                        &format!("providers.{name}.retry"),
                    )
                })
                .transpose()?,
            models,
        },
    ))
}

fn normalize_model_config(
    provider_name: &str,
    model_id: &str,
    provider_protocol: ApiProtocol,
    mut raw: RawModelConfig,
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

    let supports_reasoning = raw.supports_reasoning.unwrap_or(true);
    if let Some(max_output_tokens) = raw.max_output_tokens
        && max_output_tokens > u32::MAX as u64
    {
        bail!(
            "providers.{provider_name}.models.{model_id}.max_output_tokens must be at most {}",
            u32::MAX
        );
    }
    let effective_input_limit_tokens = optional_positive_u64(
        &format!("providers.{provider_name}.models.{model_id}.effective_input_limit_tokens"),
        raw.effective_input_limit_tokens,
    )?;
    if !supports_reasoning && raw.reasoning_effort.is_some() {
        bail!(
            "providers.{provider_name}.models.{model_id}.reasoning_effort requires supports_reasoning = true"
        );
    }
    if !supports_reasoning && !raw.reasoning_efforts.is_empty() {
        bail!(
            "providers.{provider_name}.models.{model_id}.reasoning_efforts requires supports_reasoning = true"
        );
    }
    if !supports_reasoning && raw.reasoning_summary.is_some() {
        bail!(
            "providers.{provider_name}.models.{model_id}.reasoning_summary requires supports_reasoning = true"
        );
    }
    let reasoning_efforts = normalize_reasoning_efforts(
        &format!("providers.{provider_name}.models.{model_id}.reasoning_efforts"),
        std::mem::take(&mut raw.reasoning_efforts),
        raw.reasoning_effort.clone(),
    )?;

    if let Some(temperature) = raw.temperature {
        validate_f32_range(
            &format!("providers.{provider_name}.models.{model_id}.temperature"),
            temperature,
            0.0,
            2.0,
        )?;
    }
    if let Some(top_p) = raw.top_p {
        validate_f32_range(
            &format!("providers.{provider_name}.models.{model_id}.top_p"),
            top_p,
            0.0,
            1.0,
        )?;
    }

    let protocol = raw.protocol.unwrap_or(provider_protocol);
    let anthropic_thinking = raw.anthropic_thinking.unwrap_or_default();
    if anthropic_thinking.mode != crate::request_builder::AnthropicThinkingMode::Disabled {
        if protocol != ApiProtocol::Anthropic {
            bail!(
                "providers.{provider_name}.models.{model_id}.anthropic_thinking is only supported for anthropic protocol"
            );
        }
        if !supports_reasoning {
            bail!(
                "providers.{provider_name}.models.{model_id}.anthropic_thinking requires supports_reasoning = true"
            );
        }
        if anthropic_thinking.mode == crate::request_builder::AnthropicThinkingMode::Budget {
            let budget_path = format!(
                "providers.{provider_name}.models.{model_id}.anthropic_thinking.budget_tokens"
            );
            let budget = optional_positive_u64(&budget_path, anthropic_thinking.budget_tokens)?;
            if budget.is_none() {
                bail!("{budget_path} is required when thinking mode is 'budget'");
            }
            if let Some(max_output_tokens) = raw.max_output_tokens
                && budget >= Some(max_output_tokens)
            {
                bail!("{budget_path} must be less than max_output_tokens");
            }
        }
    }
    let cache_path = format!("providers.{provider_name}.models.{model_id}.prompt_cache");
    let cache_enabled = raw.prompt_cache.enabled.unwrap_or(false);
    if !cache_enabled && raw.prompt_cache.retention.is_some() {
        bail!("{cache_path}.retention requires enabled = true");
    }
    if raw.prompt_cache.retention.is_some() && protocol == ApiProtocol::Completions {
        bail!("{cache_path}.retention is only supported for responses protocol");
    }
    let namespace = raw.prompt_cache.namespace.map(|namespace| {
        if namespace.trim().is_empty() || namespace.len() > 64 || namespace.chars().any(char::is_control) {
            bail!("{cache_path}.namespace must be non-empty, at most 64 bytes, and contain no control characters");
        }
        Ok(namespace)
    }).transpose()?;
    let prompt_cache = PromptCacheConfig {
        enabled: cache_enabled,
        retention: raw.prompt_cache.retention,
        namespace: cache_enabled
            .then(|| namespace.unwrap_or_else(|| provider_name.to_ascii_lowercase())),
    };

    Ok((
        model_id,
        ModelConfig {
            display_name,
            protocol,
            anthropic_thinking,
            cache_control: raw.cache_control,
            context_window: raw.context_window,
            effective_input_limit_tokens,
            max_output_tokens: raw.max_output_tokens,
            supports_tools: raw.supports_tools.unwrap_or(true),
            supports_reasoning,
            reasoning_effort: raw.reasoning_effort,
            reasoning_efforts,
            reasoning_summary: raw.reasoning_summary,
            text_verbosity: raw.text_verbosity,
            temperature: raw.temperature,
            top_p: raw.top_p,
            prompt_cache,
            parallel_tool_calls: raw.parallel_tool_calls.unwrap_or(true),
        },
    ))
}

fn normalize_reasoning_efforts(
    path: &str,
    configured: Vec<ModelReasoningEffort>,
    default_effort: Option<ModelReasoningEffort>,
) -> Result<Vec<ModelReasoningEffort>> {
    let mut efforts = Vec::with_capacity(configured.len());
    for effort in configured {
        validate_reasoning_effort(path, &effort)?;
        if efforts.contains(&effort) {
            bail!("{path} contains duplicate effort '{effort:?}'");
        }
        efforts.push(effort);
    }

    if let Some(default_effort) = &default_effort {
        validate_reasoning_effort(path, default_effort)?;
    }

    if let Some(default_effort) = default_effort
        && !efforts.is_empty()
        && !efforts.contains(&default_effort)
    {
        bail!("{path} must include the configured reasoning_effort");
    }

    Ok(efforts)
}

fn validate_reasoning_effort(path: &str, effort: &ModelReasoningEffort) -> Result<()> {
    let ModelReasoningEffort::Custom(value) = effort else {
        return Ok(());
    };
    if value.is_empty()
        || value.len() > 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        bail!("{path} custom efforts must be 1-64 ASCII letters, digits, '-', '_', or '.'");
    }
    Ok(())
}

fn build_tools_config(raw: RawToolsConfig) -> Result<ToolsConfig> {
    let parallelism = raw
        .parallelism
        .into_iter()
        .map(|(name, mode)| {
            let name = validate_identifier("tools.parallelism key", &name)?.to_string();
            Ok((name, mode))
        })
        .collect::<Result<IndexMap<_, _>>>()?;
    Ok(ToolsConfig { parallelism })
}

fn build_experiments_config(raw: RawExperimentsConfig) -> Result<ExperimentsConfig> {
    let Some(raw_anchored) = raw.anchored_bootstrap else {
        return Ok(ExperimentsConfig::default());
    };
    if !raw_anchored.enabled.unwrap_or(false) {
        return Ok(ExperimentsConfig::default());
    }

    let models = raw_anchored
        .models
        .ok_or_else(|| anyhow!("experiments.anchored_bootstrap.models is required when enabled"))?;
    if models.is_empty() {
        bail!("experiments.anchored_bootstrap.models cannot be empty when enabled");
    }
    for (index, model) in models.iter().enumerate() {
        required_non_empty(
            &format!("experiments.anchored_bootstrap.models[{index}]"),
            model.clone(),
        )?;
    }

    let promote_on = match raw_anchored.promote_on.as_deref() {
        None | Some("either") => PromoteOn::Either,
        Some("tool-call") => PromoteOn::ToolCall,
        Some("assistant-message") => PromoteOn::AssistantMessage,
        Some(other) => bail!(
            "experiments.anchored_bootstrap.promote_on must be one of \"either\", \"tool-call\", \"assistant-message\"; got {other:?}"
        ),
    };

    let compaction_tools = raw_anchored.compaction_tools.ok_or_else(|| {
        anyhow!("experiments.anchored_bootstrap.compaction_tools is required when enabled")
    })?;
    if compaction_tools.is_empty() {
        bail!("experiments.anchored_bootstrap.compaction_tools cannot be empty when enabled");
    }
    for (index, tool) in compaction_tools.iter().enumerate() {
        required_non_empty(
            &format!("experiments.anchored_bootstrap.compaction_tools[{index}]"),
            tool.clone(),
        )?;
    }

    Ok(ExperimentsConfig {
        anchored_bootstrap: AnchoredBootstrapConfig {
            enabled: true,
            models,
            promote_on,
            compaction_tools,
        },
    })
}

fn build_mcp_server_config(
    name: &str,
    raw: RawMcpServerConfig,
) -> Result<(String, McpServerConfig)> {
    let name = validate_identifier("mcp server key", name)?.to_string();
    let enabled = raw.enabled.unwrap_or(true);
    let timeout_ms = positive_u64(
        &format!("mcp.{name}.timeout"),
        raw.timeout.unwrap_or(DEFAULT_MCP_TIMEOUT_MS),
    )?;

    let transport = match raw.kind {
        RawMcpServerKind::Local => {
            if raw.url.is_some() || !raw.headers.is_empty() || raw.oauth.is_some() {
                bail!("mcp.{name} local server must not set remote-only fields");
            }
            let command = raw
                .command
                .ok_or_else(|| anyhow!("mcp.{name}.command is required for local servers"))?;
            if command.is_empty() {
                bail!("mcp.{name}.command cannot be empty");
            }
            let command = command
                .into_iter()
                .enumerate()
                .map(|(index, value)| {
                    required_non_empty(&format!("mcp.{name}.command[{index}]"), value)
                })
                .collect::<Result<Vec<_>>>()?;
            McpTransportConfig::Local(McpLocalServerConfig {
                command,
                environment: raw.environment,
            })
        }
        RawMcpServerKind::Remote => {
            if raw.command.is_some() || !raw.environment.is_empty() {
                bail!("mcp.{name} remote server must not set local-only fields");
            }
            let url = required_non_empty(
                &format!("mcp.{name}.url"),
                raw.url
                    .ok_or_else(|| anyhow!("mcp.{name}.url is required for remote servers"))?,
            )?;
            if raw.oauth.unwrap_or(false) {
                bail!(
                    "mcp.{name}.oauth is enabled, but remote MCP OAuth is not supported yet; set oauth = false and provide headers"
                );
            }
            McpTransportConfig::Remote(McpRemoteServerConfig {
                url,
                headers: raw.headers,
                oauth: false,
            })
        }
    };

    Ok((
        name,
        McpServerConfig {
            enabled,
            timeout_ms,
            transport,
        },
    ))
}

fn build_agents_config(
    raw: RawAgentsConfig,
    active_provider: &str,
    providers: &IndexMap<String, ProviderConfig>,
) -> Result<AgentsConfig> {
    Ok(AgentsConfig {
        explorer: build_agent_config(raw.explorer, "explorer", active_provider, providers)?,
        fixer: build_agent_config(raw.fixer, "fixer", active_provider, providers)?,
        oracle: build_agent_config(raw.oracle, "oracle", active_provider, providers)?,
        designer: build_agent_config(raw.designer, "designer", active_provider, providers)?,
        librarian: build_agent_config(raw.librarian, "librarian", active_provider, providers)?,
        general: build_agent_config(raw.general, "general", active_provider, providers)?,
        reviewer: build_agent_config(raw.reviewer, "reviewer", active_provider, providers)?,
    })
}

fn build_agent_config(
    raw: Option<RawAgentConfig>,
    agent_name: &str,
    active_provider: &str,
    providers: &IndexMap<String, ProviderConfig>,
) -> Result<AgentConfig> {
    let Some(raw) = raw else {
        return Ok(AgentConfig::default());
    };

    let allowed_models = raw
        .allowed_models
        .into_iter()
        .enumerate()
        .map(|(index, value)| {
            ModelRoute::parse(&required_non_empty(
                &format!("agents.{agent_name}.allowed_models[{index}]"),
                value,
            )?)
            .with_context(|| {
                format!(
                    "agents.{agent_name}.allowed_models[{index}] must be a provider/model route"
                )
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let mut seen_allowed = std::collections::HashSet::new();
    for route in &allowed_models {
        let provider = providers.get(&route.provider).ok_or_else(|| {
            anyhow!(
                "agents.{agent_name}.allowed_models references unknown provider '{}'",
                route.provider
            )
        })?;
        if !provider.has_model(&route.model) {
            bail!(
                "agents.{agent_name}.allowed_models model '{}' is not defined under [providers.{}.models]",
                route.model,
                route.provider
            );
        }
        if !seen_allowed.insert(route.clone()) {
            bail!(
                "agents.{agent_name}.allowed_models contains duplicate route '{}'",
                route.display_name()
            );
        }
    }

    let provider_name = raw
        .provider
        .map(|value| required_non_empty(&format!("agents.{agent_name}.provider"), value))
        .transpose()?;
    let model = raw
        .model
        .map(|value| required_non_empty(&format!("agents.{agent_name}.model"), value))
        .transpose()?;

    let follows_active_provider = provider_name.is_none() && model.is_some();
    let route = match (provider_name, model) {
        (None, None) => None,
        (Some(provider), Some(model)) => Some(ModelRoute::new(provider, model)),
        (None, Some(model)) => Some(ModelRoute::new(active_provider, model)),
        (Some(_), None) => bail!("agents.{agent_name}.provider requires agents.{agent_name}.model"),
    };

    if let Some(route) = &route {
        let provider = providers.get(&route.provider).ok_or_else(|| {
            anyhow!(
                "agents.{agent_name}.provider '{}' is not defined under [providers]",
                route.provider
            )
        })?;
        if !provider.has_model(&route.model) {
            bail!(
                "agents.{agent_name}.model '{}' is not defined under [providers.{}.models]",
                route.model,
                route.provider
            );
        }
    }

    Ok(AgentConfig {
        route,
        allowed_models,
        follows_active_provider,
    })
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

pub fn provider_api_key_env_var(provider_name: &str) -> String {
    provider_env_var(provider_name, "API_KEY")
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

fn optional_positive_usize(label: &str, value: Option<usize>) -> Result<Option<usize>> {
    value.map(|value| positive_usize(label, value)).transpose()
}

fn optional_positive_u64(label: &str, value: Option<u64>) -> Result<Option<u64>> {
    value.map(|value| positive_u64(label, value)).transpose()
}

fn positive_u64(label: &str, value: u64) -> Result<u64> {
    if value == 0 {
        bail!("{} must be greater than 0", label);
    }
    Ok(value)
}

fn validate_f32_range(label: &str, value: f32, min: f32, max: f32) -> Result<()> {
    if !value.is_finite() || value < min || value > max {
        bail!("{label} must be between {min} and {max}");
    }
    Ok(())
}

fn build_compaction_config(raw: RawCompactionConfig) -> Result<CompactionConfig> {
    Ok(CompactionConfig {
        preserve_recent_tokens: raw.preserve_recent_tokens,
    })
}

fn build_retry_config(raw: RawRetryConfig, path: &str) -> Result<RetryConfig> {
    build_retry_config_overlay(raw, &RetryConfig::default(), path)
}

fn build_retry_config_overlay(
    raw: RawRetryConfig,
    base: &RetryConfig,
    path: &str,
) -> Result<RetryConfig> {
    let max_attempts = positive_usize(
        &format!("{path}.max_attempts"),
        raw.max_attempts.unwrap_or(base.max_attempts),
    )?;
    if max_attempts > MAX_RETRY_ATTEMPTS {
        bail!("{path}.max_attempts must be at most {MAX_RETRY_ATTEMPTS}");
    }
    let max_recovery_attempts = positive_usize(
        &format!("{path}.max_recovery_attempts"),
        raw.max_recovery_attempts
            .unwrap_or(base.max_recovery_attempts),
    )?;
    if max_recovery_attempts > MAX_RECOVERY_ATTEMPTS {
        bail!("{path}.max_recovery_attempts must be at most {MAX_RECOVERY_ATTEMPTS}");
    }
    let initial_delay_secs = positive_u64(
        &format!("{path}.initial_delay_secs"),
        raw.initial_delay_secs.unwrap_or(base.initial_delay_secs),
    )?;
    let jitter_secs = raw.jitter_secs.unwrap_or(base.jitter_secs);
    let backoff_multiplier = raw.backoff_multiplier.unwrap_or(base.backoff_multiplier);
    validate_f32_range(
        &format!("{path}.backoff_multiplier"),
        backoff_multiplier,
        1.0,
        10.0,
    )?;

    Ok(RetryConfig {
        enabled: raw.enabled.unwrap_or(base.enabled),
        max_attempts,
        max_recovery_attempts,
        initial_delay_secs,
        backoff_multiplier,
        jitter_secs,
    })
}

/// Load-check a config file with the same parser startup and hot-reload use.
/// Never panics; invalid configs return `valid: false` with the error text.
pub fn validate_config_file(path: impl AsRef<Path>) -> ConfigValidationReport {
    let path = path.as_ref();
    let path_display = path.display().to_string();
    match AppConfig::load_from_path(path) {
        Ok(config) => {
            let active_route = config.active_route();
            ConfigValidationReport {
                valid: true,
                path: path_display,
                error: None,
                active_provider: Some(config.active_provider.clone()),
                active_route: Some(active_route.display_name()),
                providers: config.providers.keys().cloned().collect(),
                mcp_servers: config.mcp.keys().cloned().collect(),
                permission_mode: Some(config.permissions.mode.as_str().to_string()),
                fast_mode: Some(config.fast_mode_enabled),
            }
        }
        Err(error) => ConfigValidationReport {
            valid: false,
            path: path_display,
            error: Some(format!("{error:#}")),
            active_provider: None,
            active_route: None,
            providers: Vec::new(),
            mcp_servers: Vec::new(),
            permission_mode: None,
            fast_mode: None,
        },
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ConfigValidationReport {
    pub valid: bool,
    pub path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub active_provider: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub active_route: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub providers: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub mcp_servers: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub permission_mode: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fast_mode: Option<bool>,
}

fn missing_config_message(path: &Path) -> String {
    format!(
        "config file not found: {}\n\nCreate it with at least:\n\nactive_provider = \"openai\"\n\n[global]\n# Optional runtime limits:\n# max_iterations = 64\n# max_tool_calls = 128\n# tool_timeout_secs = 60\nsessions_dir = \"sessions\"\nlog_file = \"logs/combined.log\"\n\n[global.compaction]\n# preserve_recent_tokens defaults to the active model input budget\n# preserve_recent_tokens = 4096\n\n[global.retry]\nenabled = true\nmax_attempts = 50\nmax_recovery_attempts = 3\ninitial_delay_secs = 1\nbackoff_multiplier = 2.0\njitter_secs = 1\n\n[permissions]\nmode = \"default\"\n\n# Optional local tool execution policy:\n# [tools.parallelism]\n# \"fs__read\" = \"parallel\"\n# \"web__fetch\" = \"exclusive\"\n\n[providers.openai]\napi_key = \"YOUR_API_KEY\"\nbase_url = \"https://api.openai.com/v1\"\nprotocol = \"responses\"\ndefault_model = \"gpt-5.5\"\n\n# Optional provider-specific retry override:\n# [providers.openai.retry]\n# enabled = true\n# max_attempts = 50\n# max_recovery_attempts = 3\n# initial_delay_secs = 1\n# backoff_multiplier = 2.0\n# jitter_secs = 1\n\n[providers.openai.models.\"gpt-5.5\"]\ndisplay_name = \"GPT-5.5\"\nsupports_tools = true\nparallel_tool_calls = true\nsupports_reasoning = true\nreasoning_effort = \"medium\"\n# Optional per-model selectable levels and TUI cycle order:\n# reasoning_efforts = [\"none\", \"low\", \"medium\", \"high\", \"max\"]\nreasoning_summary = \"auto\"\ntext_verbosity = \"medium\"\n\n# OpenAI-compatible Chat Completions provider:\n# [providers.compat]\n# api_key = \"YOUR_API_KEY\"\n# base_url = \"https://example.com/v1\"\n# protocol = \"completions\"\n# default_model = \"your-model\"\n",
        path.display()
    )
}

pub fn default_config_path() -> Result<PathBuf> {
    // Unix shells set HOME; Windows typically only has USERPROFILE.
    let home = env::var_os("HOME")
        .or_else(|| env::var_os("USERPROFILE"))
        .ok_or_else(|| anyhow!("neither HOME nor USERPROFILE is set"))?;
    Ok(PathBuf::from(home).join(DEFAULT_CONFIG_HOME_RELATIVE_PATH))
}

#[cfg(test)]
mod tests {
    use super::persistence::{
        acquire_config_lock, atomic_write_config, config_lock_path, open_config_lock_file,
        replace_file,
    };
    use super::*;
    use std::fs;
    use std::sync::{Mutex, OnceLock};
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn rejects_retry_attempts_above_maximum() {
        let _guard = lock_env();
        let path = write_temp_config(
            r#"
            [global.retry]
            max_attempts = 51

            [providers.openai]
            api_key = "config-key"

            [providers.openai.models."gpt-5.5"]
            name = "GPT-5.5"
            "#,
        );

        let error = AppConfig::load_from_path(&path).expect_err("config should be rejected");
        assert!(
            error
                .to_string()
                .contains("global.retry.max_attempts must be at most 50")
        );
    }

    #[test]
    fn rejects_invalid_model_specific_reasoning_efforts() {
        let _guard = lock_env();
        for config in [
            r#"
            [providers.openai]
            api_key = "config-key"

            [providers.openai.models.model]
            supports_reasoning = false
            reasoning_efforts = ["low"]
            "#,
            r#"
            [providers.openai]
            api_key = "config-key"

            [providers.openai.models.model]
            reasoning_effort = "high"
            reasoning_efforts = ["low", "medium"]
            "#,
            r#"
            [providers.openai]
            api_key = "config-key"

            [providers.openai.models.model]
            reasoning_efforts = ["low", "low"]
            "#,
            r#"
            [providers.openai]
            api_key = "config-key"
            [providers.openai.models.model]
            supports_reasoning = true
            reasoning_effort = "provider ultra"
            "#,
        ] {
            let path = write_temp_config(config);
            assert!(AppConfig::load_from_path(&path).is_err());
        }
    }

    #[test]
    fn model_parallel_tool_calls_default_to_enabled_and_can_be_disabled() {
        let _guard = lock_env();
        let default_path = write_temp_config(
            r#"
            [providers.openai]
            api_key = "config-key"

            [providers.openai.models."gpt-default"]
            "#,
        );
        let disabled_path = write_temp_config(
            r#"
            [providers.openai]
            api_key = "config-key"

            [providers.openai.models."gpt-disabled"]
            parallel_tool_calls = false
            "#,
        );

        let default_config =
            AppConfig::load_from_path(&default_path).expect("default config should load");
        let disabled_config =
            AppConfig::load_from_path(&disabled_path).expect("disabled config should load");

        assert!(default_config.providers["openai"].models["gpt-default"].parallel_tool_calls);
        assert!(!disabled_config.providers["openai"].models["gpt-disabled"].parallel_tool_calls);
    }

    #[test]
    fn anthropic_model_override_requires_provider_auth_mode() {
        let _guard = lock_env();
        let missing_path = write_temp_config(
            r#"
            [providers.mixed]
            api_key = "config-key"
            base_url = "https://anthropic.invalid/v1"
            protocol = "completions"

            [providers.mixed.models."claude"]
            protocol = "anthropic"
            "#,
        );
        let error =
            AppConfig::load_from_path(&missing_path).expect_err("config should be rejected");
        assert!(error.to_string().contains(
            "provider 'mixed' with a model protocol = 'anthropic' requires auth_mode = 'api-key' or 'bearer'"
        ));

        let configured_path = write_temp_config(
            r#"
            [providers.mixed]
            api_key = "config-key"
            base_url = "https://anthropic.invalid/v1"
            protocol = "completions"
            auth_mode = "api-key"

            [providers.mixed.models."claude"]
            protocol = "anthropic"
            "#,
        );
        let config =
            AppConfig::load_from_path(&configured_path).expect("configured override should load");
        assert_eq!(
            config.providers["mixed"].auth_mode,
            ProviderAuthMode::ApiKey
        );
    }

    #[test]
    fn rejects_zero_model_effective_input_limit_tokens() {
        let _guard = lock_env();
        let path = write_temp_config(
            r#"
            [providers.openai]
            api_key = "config-key"

            [providers.openai.models."gpt-5.5"]
            name = "GPT-5.5"
            effective_input_limit_tokens = 0
            "#,
        );

        let error = AppConfig::load_from_path(&path).expect_err("config should be rejected");
        assert!(error.to_string().contains(
            "providers.openai.models.gpt-5.5.effective_input_limit_tokens must be greater than 0"
        ));
    }

    #[test]
    fn parses_provider_qualified_expert_routes_and_preserves_duplicate_model_ids() {
        let _guard = lock_env();
        let path = write_temp_config(
            r#"
            active_provider = "primary"

            [agents.explorer]
            model = "shared"

            [agents.fixer]
            provider = "expert"
            model = "shared"

            [agents.general]
            provider = "expert"
            model = "shared"

            [providers.primary]
            base_url = "https://primary.invalid/v1"
            api_key = "primary-key"
            protocol = "responses"
            default_model = "shared"

            [providers.primary.models.shared]
            name = "Primary Shared"

            [providers.expert]
            base_url = "https://expert.invalid/v1"
            api_key = "expert-key"
            protocol = "completions"
            default_model = "shared"

            [providers.expert.models.shared]
            name = "Expert Shared"
            supports_reasoning = false
            "#,
        );

        let config = AppConfig::load_from_path(&path).expect("config should load");
        assert_eq!(
            config.model_route_for("explorer"),
            Some(&ModelRoute::new("primary", "shared"))
        );
        assert_eq!(
            config.model_route_for("fixer"),
            Some(&ModelRoute::new("expert", "shared"))
        );
        assert_eq!(
            config.model_route_for("general"),
            Some(&ModelRoute::new("expert", "shared"))
        );
        assert_eq!(config.agents.model_for("explorer"), Some("shared"));
        assert_eq!(config.agents.model_for("fixer"), Some("shared"));
        assert_eq!(config.active_route(), ModelRoute::new("primary", "shared"));
        assert!(
            config
                .agents
                .allowed_models_for("explorer")
                .expect("known expert")
                .is_empty()
        );
        assert_eq!(
            config
                .resolve_route(config.model_route_for("fixer").expect("fixer route"))
                .expect("route resolves")
                .protocol,
            ApiProtocol::Completions
        );
        assert_eq!(
            config
                .resolve_route(config.model_route_for("general").expect("general route"))
                .expect("route resolves")
                .protocol,
            ApiProtocol::Completions
        );
    }

    #[test]
    fn parses_and_validates_expert_allowed_models() {
        let _guard = lock_env();
        let path = write_temp_config(
            r#"
            active_provider = "primary"

            [agents.explorer]
            model = "shared"
            allowed_models = ["expert/special"]

            [providers.primary]
            base_url = "https://primary.invalid/v1"
            api_key = "primary-key"
            protocol = "responses"
            default_model = "shared"

            [providers.primary.models.shared]

            [providers.expert]
            base_url = "https://expert.invalid/v1"
            api_key = "expert-key"
            protocol = "responses"
            default_model = "special"

            [providers.expert.models.special]
            "#,
        );
        let before = fs::read_to_string(&path).expect("read config before load");
        let config = AppConfig::load_from_path(&path).expect("allowed model config loads");
        assert_eq!(
            fs::read_to_string(&path).expect("read config after load"),
            before,
            "loading and selecting allowed models must not persist configuration changes"
        );
        assert_eq!(
            config.agents.allowed_models_for("explorer"),
            Some([ModelRoute::new("expert", "special")].as_slice())
        );
        assert_eq!(
            config.model_route_for("explorer"),
            Some(&ModelRoute::new("primary", "shared")),
            "allowlist must not change the legacy default route"
        );

        for (allowed_models, expected) in [
            ("[\"special\"]", "must use provider/model form"),
            (
                "[\"missing/special\"]",
                "references unknown provider 'missing'",
            ),
            (
                "[\"expert/missing\"]",
                "model 'missing' is not defined under [providers.expert.models]",
            ),
            (
                "[\"expert/special\", \"expert/special\"]",
                "contains duplicate route 'expert/special'",
            ),
        ] {
            let path = write_temp_config(&format!(
                r#"
                [agents.explorer]
                allowed_models = {allowed_models}

                [providers.primary]
                base_url = "https://primary.invalid/v1"
                api_key = "primary-key"
                protocol = "responses"
                default_model = "shared"

                [providers.primary.models.shared]

                [providers.expert]
                base_url = "https://expert.invalid/v1"
                api_key = "expert-key"
                protocol = "responses"
                default_model = "special"

                [providers.expert.models.special]
                "#
            ));
            let error = AppConfig::load_from_path(&path).expect_err("invalid allowlist fails");
            assert!(format!("{error:#}").contains(expected), "{error:#}");
        }
    }

    #[test]
    fn rejects_incomplete_or_unknown_provider_qualified_expert_routes() {
        let _guard = lock_env();
        let path = write_temp_config(
            r#"
            [agents.explorer]
            provider = "expert"

            [providers.openai]
            api_key = "config-key"

            [providers.openai.models.model]
            "#,
        );
        let error = AppConfig::load_from_path(&path).expect_err("incomplete route must fail");
        assert!(
            error
                .to_string()
                .contains("agents.explorer.provider requires agents.explorer.model")
        );

        let path = write_temp_config(
            r#"
            [agents.explorer]
            provider = "missing"
            model = "model"

            [providers.openai]
            api_key = "config-key"

            [providers.openai.models.model]
            "#,
        );
        let error = AppConfig::load_from_path(&path).expect_err("unknown provider must fail");
        assert!(
            error
                .to_string()
                .contains("agents.explorer.provider 'missing' is not defined")
        );

        let path = write_temp_config(
            r#"
            [agents.explorer]
            provider = "expert"
            model = "missing"

            [providers.openai]
            api_key = "config-key"

            [providers.openai.models.model]

            [providers.expert]
            base_url = "https://expert.invalid/v1"
            api_key = "expert-key"
            protocol = "responses"

            [providers.expert.models.model]
            "#,
        );
        let error = AppConfig::load_from_path(&path).expect_err("unknown model must fail");
        assert!(error.to_string().contains(
            "agents.explorer.model 'missing' is not defined under [providers.expert.models]"
        ));
    }

    #[test]
    fn rejects_unknown_subagent_name() {
        let _guard = lock_env();
        let path = write_temp_config(
            r#"
            [agents.nosuch]
            model = "gpt-5.5"

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
        assert!(error.chain().any(|cause| {
            let message = cause.to_string();
            message.contains("explorer")
                && message.contains("fixer")
                && message.contains("oracle")
                && message.contains("designer")
                && message.contains("librarian")
                && message.contains("general")
                && message.contains("reviewer")
        }));
    }

    #[test]
    fn rejects_unknown_subagent_model_override() {
        let _guard = lock_env();
        let path = write_temp_config(
            r#"
            [agents.explorer]
            model = "missing-model"

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
                .contains("agents.explorer.model 'missing-model' is not defined")
        );
    }

    #[test]
    fn rejects_reasoning_parameters_when_reasoning_is_disabled() {
        let _guard = lock_env();
        let path = write_temp_config(
            r#"
            [providers.openai]
            api_key = "config-key"

            [providers.openai.models."gpt-5.5"]
            name = "GPT-5.5"
            supports_reasoning = false
            reasoning_effort = "medium"
            "#,
        );

        let error = AppConfig::load_from_path(&path).expect_err("load should fail");
        assert!(error.to_string().contains("reasoning_effort requires"));
    }

    #[test]
    fn rejects_out_of_range_sampling_parameters() {
        let _guard = lock_env();
        let path = write_temp_config(
            r#"
            [providers.openai]
            api_key = "config-key"

            [providers.openai.models."gpt-5.5"]
            name = "GPT-5.5"
            temperature = 3.0
            "#,
        );

        let error = AppConfig::load_from_path(&path).expect_err("load should fail");
        assert!(error.to_string().contains("temperature must be between"));
    }

    #[test]
    fn persists_primary_and_expert_model_routes_without_rewriting_unrelated_config() {
        let path = write_temp_config(
            r#"# keep this comment
active_provider = "primary"

[providers.primary]
base_url = "https://primary.invalid/v1"
api_key = "primary-key"
protocol = "responses"
default_model = "old"
[providers.primary.models.old]
[providers.primary.models.shared]

[providers.expert]
base_url = "https://expert.invalid/v1"
api_key = "expert-key"
protocol = "responses"
[providers.expert.models.shared]

[agents.explorer]
model = "shared"

# preserve this trailing comment
"#,
        );

        persist_primary_model_route(&path, &ModelRoute::new("expert", "shared"))
            .expect("persist primary route");
        persist_expert_model_route(&path, "explorer", &ModelRoute::new("primary", "shared"))
            .expect("persist expert route");
        persist_expert_allowed_models(
            &path,
            "explorer",
            &[
                ModelRoute::new("primary", "old"),
                ModelRoute::new("expert", "shared"),
            ],
        )
        .expect("persist expert allowed models");

        let written = fs::read_to_string(&path).expect("read updated config");
        assert!(written.contains("# keep this comment"));
        assert!(written.contains("# preserve this trailing comment"));
        let config = AppConfig::load_from_path(&path).expect("reload updated config");
        assert_eq!(config.active_provider().0, "expert");
        assert_eq!(config.active_provider().1.default_model, "shared");
        assert_eq!(
            config.model_route_for("explorer"),
            Some(&ModelRoute::new("primary", "shared"))
        );
        assert_eq!(
            config.agents.allowed_models_for("explorer"),
            Some(
                [
                    ModelRoute::new("primary", "old"),
                    ModelRoute::new("expert", "shared"),
                ]
                .as_slice()
            )
        );
    }

    #[test]
    fn refuses_to_overwrite_config_changed_since_read() {
        let path = write_temp_config("original config\n");
        let original_contents = fs::read_to_string(&path).expect("read original config");
        let original_metadata = fs::metadata(&path).expect("stat original config");
        fs::write(&path, "external edit\n").expect("write external edit");

        let error = atomic_write_config(
            &path,
            &original_contents,
            &original_metadata,
            b"our update\n",
        )
        .expect_err("changed config must not be overwritten");

        assert!(error.to_string().contains("changed while updating"));
        assert_eq!(
            fs::read_to_string(&path).expect("read externally changed config"),
            "external edit\n"
        );
    }

    #[test]
    fn replace_file_overwrites_existing_destination() {
        let destination = write_temp_config("old\n");
        let source = destination.with_extension("replacement");
        fs::write(&source, "new\n").expect("write replacement");

        replace_file(&source, &destination).expect("replace destination");

        assert_eq!(fs::read_to_string(&destination).unwrap(), "new\n");
        assert!(!source.exists());
    }

    #[cfg(unix)]
    #[test]
    fn sibling_lock_remains_held_after_config_target_replacement() {
        use std::os::fd::AsRawFd;

        let path = write_temp_config("original\n");
        let target = fs::canonicalize(&path).expect("canonical config target");
        let lock_path = config_lock_path(&target).expect("lock path");
        let lock = acquire_config_lock(&target).expect("acquire first lock");

        let replacement = target.with_file_name("config-lock-replacement");
        fs::write(&replacement, "replacement\n").expect("write replacement");
        fs::rename(&replacement, &target).expect("replace config target");

        let second = open_config_lock_file(&lock_path).expect("open second lock handle");
        let result = unsafe { libc::flock(second.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        assert_eq!(result, -1, "stable lock must survive target replacement");
        drop(second);
        drop(lock);
        acquire_config_lock(&target).expect("lock is released with first writer");
    }

    #[cfg(unix)]
    #[test]
    fn persists_through_config_symlink_without_replacing_it() {
        use std::os::unix::fs::symlink;

        let target = write_temp_config(
            "[mcp.alpha]\ntype = \"local\"\ncommand = [\"alpha\"]\n\n[providers.openai]\napi_key = \"config-key\"\n[providers.openai.models.model]\n",
        );
        let link = target.with_file_name("letcode-link.toml");
        symlink(&target, &link).expect("create config symlink");

        persist_mcp_server_enabled(&link, "alpha", false).expect("persist through symlink");

        assert!(
            fs::symlink_metadata(&link)
                .expect("stat symlink")
                .file_type()
                .is_symlink()
        );
        assert!(
            fs::read_to_string(&target)
                .expect("read target")
                .contains("enabled = false")
        );
        assert!(
            !AppConfig::load_from_path(&link)
                .expect("reload through symlink")
                .mcp["alpha"]
                .enabled
        );
    }

    #[test]
    fn rejects_invalid_mcp_local_config() {
        let _guard = lock_env();
        let path = write_temp_config(
            r#"
            [mcp.empty]
            type = "local"
            command = []

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
                .contains("mcp.empty.command cannot be empty")
        );
    }

    #[test]
    fn rejects_remote_mcp_oauth_until_supported() {
        let _guard = lock_env();
        let path = write_temp_config(
            r#"
            [mcp.docs]
            type = "remote"
            url = "https://example.invalid/mcp"
            oauth = true

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
                .contains("remote MCP OAuth is not supported")
        );
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

    #[test]
    fn prompt_cache_rejects_retention_when_disabled() {
        let _guard = lock_env();
        let error = load_openai_prompt_cache_error(
            r#"
            [providers.openai.models."gpt-test".prompt_cache]
            enabled = false
            retention = "in_memory"
            "#,
        );

        assert!(
            error
                .to_string()
                .contains("prompt_cache.retention requires enabled = true")
        );
    }

    #[test]
    fn prompt_cache_rejects_retention_for_completions_models() {
        let _guard = lock_env();
        let path = write_temp_config(
            r#"
            [providers.compat]
            base_url = "https://example.invalid/v1"
            api_key = "config-key"
            protocol = "completions"

            [providers.compat.models."compat-model"]

            [providers.compat.models."compat-model".prompt_cache]
            enabled = true
            retention = "24h"
            "#,
        );

        let error = AppConfig::load_from_path(&path).expect_err("config should be rejected");
        assert!(
            error
                .to_string()
                .contains("prompt_cache.retention is only supported for responses protocol")
        );
    }

    #[test]
    fn prompt_cache_rejects_invalid_namespaces() {
        let _guard = lock_env();
        for namespace in ["   ", &"a".repeat(65), "valid\\u0007name"] {
            let error = load_openai_prompt_cache_error(&format!(
                r#"
                [providers.openai.models."gpt-test".prompt_cache]
                enabled = true
                namespace = "{namespace}"
                "#,
            ));
            assert!(
                error.to_string().contains(
                    "prompt_cache.namespace must be non-empty, at most 64 bytes, and contain no control characters"
                ),
                "unexpected error for namespace {namespace:?}: {error:#}"
            );
        }
    }

    #[test]
    fn prompt_cache_rejects_layout_and_other_unknown_fields() {
        let _guard = lock_env();
        for (field, expected_unknown_field) in [
            ("layout = \"v1\"", "layout"),
            ("layout = \"v2\"", "layout"),
            ("unknown = true", "unknown"),
        ] {
            let error = load_openai_prompt_cache_error(&format!(
                r#"
                [providers.openai.models."gpt-test".prompt_cache]
                enabled = true
                {field}
                "#,
            ));
            assert!(
                error.chain().any(|cause| cause
                    .to_string()
                    .contains(&format!("unknown field `{expected_unknown_field}`"))),
                "unexpected error for {field}: {error:#}"
            );
        }
    }

    fn load_openai_prompt_cache_model(prompt_cache: &str) -> ModelConfig {
        let path = write_temp_config(&format!(
            r#"
            [providers.openai]
            api_key = "config-key"

            [providers.openai.models."gpt-test"]
            {prompt_cache}
            "#,
        ));
        let config = AppConfig::load_from_path(&path).expect("config should load");
        config.providers["openai"].models["gpt-test"].clone()
    }

    fn load_openai_prompt_cache_error(prompt_cache: &str) -> anyhow::Error {
        let path = write_temp_config(&format!(
            r#"
            [providers.openai]
            api_key = "config-key"

            [providers.openai.models."gpt-test"]
            {prompt_cache}
            "#,
        ));
        AppConfig::load_from_path(&path).expect_err("config should be rejected")
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
