use crate::model_runtime::{
    AuthScheme, CacheRetention, ProtocolRegistry, ProtocolSettings, ProviderFlavor,
    ResolvedModelRoute, ResolvedProvider, RuntimeAuthConfig, RuntimeConfig,
    RuntimeEndpointOverride, RuntimeEndpoints, RuntimeModelConfig, RuntimeProviderConfig,
    RuntimeRetryConfig, RuntimeTransportConfig,
};
use crate::permission::PermissionMode;
use crate::request_builder::{
    ModelReasoningEffort, ModelReasoningSummary, ModelRequestMetadata, ModelTextVerbosity,
};
use anyhow::{Context, Result, anyhow, bail};
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

impl ApiProtocol {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Responses => "responses",
            Self::Completions => "completions",
            Self::Anthropic => "anthropic",
        }
    }
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
const DEFAULT_MCP_TIMEOUT_MS: u64 = 5_000;
const DEFAULT_SESSIONS_DIR: &str = "sessions";
const DEFAULT_LOG_FILE: &str = "logs/combined.log";
const MAX_RETRY_ATTEMPTS: usize = 9_000;
const MAX_RECOVERY_ATTEMPTS: usize = 10;
mod persistence;

use persistence::acquire_config_read_lock;
pub(crate) use persistence::replace_file;
#[allow(unused_imports)]
pub use persistence::{persist_expert_allowed_models, persist_mcp_server_enabled};

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
    pub runtime_catalog: crate::model_runtime::ResolvedRuntimeCatalog,
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
        let runtime_config = build_runtime_config(raw.clone())?;
        let resolved_catalog = runtime_config
            .resolve(&ProtocolRegistry::builtins())
            .map_err(|error| anyhow!(error))?;
        if resolved_catalog.providers.is_empty() {
            bail!("resolved runtime catalog must not be empty");
        }

        if raw.providers.is_empty() {
            bail!("config must define at least one provider under [providers]");
        }

        let active_provider = required_non_empty(
            "active_provider",
            raw.active_provider.clone().unwrap_or_else(|| {
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

        let providers = resolved_catalog
            .providers
            .iter()
            .map(|(name, provider)| Ok((name.clone(), project_provider_config(provider)?)))
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
            runtime_catalog: resolved_catalog,
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
    pub exponential_backoff: bool,
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
            exponential_backoff: true,
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
#[derive(Clone, PartialEq)]
pub struct ProviderConfig {
    pub base_url: String,
    pub api_key: String,
    pub auth_mode: ProviderAuthMode,
    pub protocol: ApiProtocol,
    pub default_model: String,
    pub retry: Option<RetryConfig>,
    pub models: IndexMap<String, ModelConfig>,
}

impl std::fmt::Debug for ProviderConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProviderConfig")
            .field("base_url", &self.base_url)
            .field("api_key", &"<redacted>")
            .field("auth_mode", &self.auth_mode)
            .field("protocol", &self.protocol)
            .field("default_model", &self.default_model)
            .field("retry", &self.retry)
            .field("models", &self.models)
            .finish()
    }
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
    pub anthropic_betas: Vec<String>,
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
            anthropic_betas: self.anthropic_betas.clone(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
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

#[derive(Debug, Clone, Default, Deserialize)]
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

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawCompactionConfig {
    preserve_recent_tokens: Option<u64>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawRetryConfig {
    enabled: Option<bool>,
    max_attempts: Option<usize>,
    max_recovery_attempts: Option<usize>,
    initial_delay_secs: Option<u64>,
    exponential_backoff: Option<bool>,
    backoff_multiplier: Option<f32>,
    jitter_secs: Option<u64>,
}

#[derive(Debug, Clone, Default, Deserialize)]
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

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawAgentConfig {
    provider: Option<String>,
    model: Option<String>,
    #[serde(default)]
    allowed_models: Vec<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawPermissionsConfig {
    mode: Option<PermissionMode>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawToolsConfig {
    #[serde(default)]
    parallelism: IndexMap<String, crate::tool::ToolParallelism>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawExperimentsConfig {
    anchored_bootstrap: Option<RawAnchoredBootstrapConfig>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawAnchoredBootstrapConfig {
    enabled: Option<bool>,
    models: Option<Vec<String>>,
    promote_on: Option<String>,
    compaction_tools: Option<Vec<String>>,
}

#[derive(Debug, Clone, Deserialize)]
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

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "lowercase")]
enum RawMcpServerKind {
    Local,
    Remote,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
#[derive(Clone)]
struct RawProviderConfig {
    #[serde(default)]
    protocol: Option<String>,
    #[serde(default)]
    default_model: Option<String>,
    #[serde(default)]
    retry: Option<RawRetryConfig>,
    #[serde(default)]
    flavor: Option<ProviderFlavor>,
    auth: Option<RuntimeAuthConfig>,
    endpoints: Option<RawRuntimeEndpoints>,
    #[serde(default)]
    transport: RuntimeTransportConfig,
    #[serde(default)]
    headers: std::collections::BTreeMap<String, String>,
    #[serde(default)]
    query: std::collections::BTreeMap<String, String>,
    #[serde(default)]
    models: IndexMap<String, RuntimeModelConfig>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
#[derive(Clone)]
struct RawRuntimeEndpoints {
    base_url: String,
    #[serde(default)]
    responses: RuntimeEndpointOverride,
    #[serde(default)]
    completions: RuntimeEndpointOverride,
    #[serde(default)]
    anthropic: RuntimeEndpointOverride,
}

fn runtime_retry(
    raw: Option<RawRetryConfig>,
    base: &RetryConfig,
    path: &str,
) -> Result<RuntimeRetryConfig> {
    let resolved = build_retry_config_overlay(raw.unwrap_or_default(), base, path)?;
    Ok(RuntimeRetryConfig {
        enabled: resolved.enabled,
        max_attempts: resolved.max_attempts,
        max_recovery_attempts: resolved.max_recovery_attempts,
        initial_delay_secs: resolved.initial_delay_secs,
        exponential_backoff: resolved.exponential_backoff,
        backoff_multiplier: resolved.backoff_multiplier,
        jitter_secs: resolved.jitter_secs,
    })
}

fn build_runtime_config(raw: RawAppConfig) -> Result<RuntimeConfig> {
    if raw.providers.is_empty() {
        bail!("config must define at least one provider under [providers]");
    }
    let active_provider = required_non_empty(
        "active_provider",
        raw.active_provider.clone().unwrap_or_else(|| {
            raw.providers
                .keys()
                .next()
                .expect("providers should contain at least one entry")
                .clone()
        }),
    )?;
    let global_retry = build_retry_config(
        raw.global
            .as_ref()
            .and_then(|global| global.retry.clone())
            .unwrap_or_default(),
        "global.retry",
    )?;
    let providers = raw
        .providers
        .into_iter()
        .map(|(name, provider)| {
            let auth = provider
                .auth
                .ok_or_else(|| anyhow!("providers.{name}.auth is required"))?;
            let endpoints = provider
                .endpoints
                .ok_or_else(|| anyhow!("providers.{name}.endpoints is required"))?;
            let models = provider.models.into_iter().collect();
            let endpoint_protocol = provider.protocol.clone();
            let endpoints = RuntimeEndpoints {
                base_url: endpoints.base_url,
                responses: endpoints.responses,
                completions: endpoints.completions,
                anthropic: endpoints.anthropic,
            };
            Ok((
                name.clone(),
                RuntimeProviderConfig {
                    protocol: provider.protocol.or(endpoint_protocol),
                    default_model: provider.default_model,
                    retry: Some(runtime_retry(
                        provider.retry,
                        &global_retry,
                        &format!("providers.{name}.retry"),
                    )?),
                    flavor: provider.flavor.unwrap_or_default(),
                    auth,
                    endpoints,
                    transport: provider.transport,
                    headers: provider.headers,
                    query: provider.query,
                    models,
                },
            ))
        })
        .collect::<Result<std::collections::BTreeMap<_, _>>>()?;
    let runtime_config = RuntimeConfig {
        active_provider,
        providers,
    };
    runtime_config
        .clone()
        .validate()
        .map_err(|error| anyhow!(error))?;
    Ok(runtime_config)
}

fn parse_reasoning_effort(value: &str) -> Result<ModelReasoningEffort> {
    Ok(match value {
        "none" => ModelReasoningEffort::None,
        "minimal" => ModelReasoningEffort::Minimal,
        "low" => ModelReasoningEffort::Low,
        "medium" => ModelReasoningEffort::Medium,
        "high" => ModelReasoningEffort::High,
        "xhigh" => ModelReasoningEffort::Xhigh,
        "max" => ModelReasoningEffort::Max,
        value => ModelReasoningEffort::Custom(value.to_owned()),
    })
}

fn parse_reasoning_summary(value: &str) -> Result<ModelReasoningSummary> {
    match value {
        "auto" => Ok(ModelReasoningSummary::Auto),
        "concise" => Ok(ModelReasoningSummary::Concise),
        "detailed" => Ok(ModelReasoningSummary::Detailed),
        other => bail!("unknown reasoning summary '{other}'"),
    }
}

fn parse_text_verbosity(value: &str) -> Result<ModelTextVerbosity> {
    match value {
        "low" => Ok(ModelTextVerbosity::Low),
        "medium" => Ok(ModelTextVerbosity::Medium),
        "high" => Ok(ModelTextVerbosity::High),
        other => bail!("unknown text verbosity '{other}'"),
    }
}

fn runtime_protocol(value: Option<&str>) -> Result<ApiProtocol> {
    match value.unwrap_or("responses") {
        "responses" => Ok(ApiProtocol::Responses),
        "completions" => Ok(ApiProtocol::Completions),
        "anthropic" => Ok(ApiProtocol::Anthropic),
        other => bail!("unknown protocol '{other}'"),
    }
}

fn protocol_setting<T: for<'de> Deserialize<'de>>(
    settings: &ProtocolSettings,
    key: &str,
) -> Result<Option<T>> {
    let Some(value) = settings.value.get(key).cloned() else {
        return Ok(None);
    };
    value
        .try_into()
        .map(Some)
        .map_err(|error| anyhow!("invalid protocol setting {key}: {error}"))
}

fn project_resolved_model_config(model: &ResolvedModelRoute) -> Result<ModelConfig> {
    let protocol = runtime_protocol(Some(model.protocol_id.as_str()))?;
    let anthropic_thinking: crate::request_builder::AnthropicThinkingConfig =
        protocol_setting(&model.protocol_settings, "anthropic_thinking")?.unwrap_or_default();
    let anthropic_betas =
        protocol_setting::<Vec<String>>(&model.protocol_settings, "anthropic_betas")?
            .unwrap_or_default();
    if protocol != ApiProtocol::Anthropic
        && (!anthropic_betas.is_empty()
            || anthropic_thinking.mode != crate::request_builder::AnthropicThinkingMode::Disabled)
    {
        bail!(
            "providers.{}.models.{}.protocol_settings require anthropic protocol",
            model.provider,
            model.model
        );
    }
    let reasoning_effort = model
        .generation_defaults
        .reasoning_effort
        .as_deref()
        .map(parse_reasoning_effort)
        .transpose()?;
    let reasoning_efforts = model
        .generation_defaults
        .reasoning_efforts
        .iter()
        .map(|value| parse_reasoning_effort(value))
        .collect::<Result<Vec<_>>>()?;
    let reasoning_summary = model
        .generation_defaults
        .reasoning_summary
        .as_deref()
        .map(parse_reasoning_summary)
        .transpose()?;
    let text_verbosity = model
        .generation_defaults
        .text_verbosity
        .as_deref()
        .map(parse_text_verbosity)
        .transpose()?;
    if let Some(max_output_tokens) = model.generation_defaults.max_output_tokens
        && max_output_tokens > u32::MAX as u64
    {
        bail!(
            "providers.{}.models.{}.generation.max_output_tokens must be at most {}",
            model.provider,
            model.model,
            u32::MAX
        );
    }
    let retention = model.cache.retention.map(|value| match value {
        CacheRetention::InMemory => PromptCacheRetention::InMemory,
        CacheRetention::TwentyFourHours => PromptCacheRetention::TwentyFourHours,
    });
    let prompt_cache = PromptCacheConfig {
        enabled: model.cache.enabled,
        retention,
        namespace: model.cache.namespace.clone(),
    };
    Ok(ModelConfig {
        display_name: model.display.clone(),
        protocol,
        anthropic_thinking,
        anthropic_betas,
        context_window: model.context_window,
        effective_input_limit_tokens: model.effective_input_limit_tokens,
        max_output_tokens: model.generation_defaults.max_output_tokens,
        supports_tools: model.capabilities.tools,
        supports_reasoning: model.capabilities.reasoning,
        reasoning_effort,
        reasoning_efforts,
        reasoning_summary,
        text_verbosity,
        temperature: model.generation_defaults.temperature,
        top_p: model.generation_defaults.top_p,
        prompt_cache,
        parallel_tool_calls: model.capabilities.parallel_tool_calls
            && model
                .generation_defaults
                .parallel_tool_calls
                .unwrap_or(false),
    })
}

fn project_provider_config(provider: &ResolvedProvider) -> Result<ProviderConfig> {
    let name = &provider.name;
    let default_model = provider.default_model.clone();
    let protocol = provider
        .models
        .get(&default_model)
        .map(|route| route.protocol_id.as_str())
        .map(|value| runtime_protocol(Some(value)))
        .transpose()?
        .unwrap_or(ApiProtocol::Responses);
    if !provider.models.contains_key(&default_model) {
        bail!("provider '{name}' default_model '{default_model}' is not defined");
    }
    let models = provider
        .models
        .iter()
        .map(|(model_name, model)| Ok((model_name.clone(), project_resolved_model_config(model)?)))
        .collect::<Result<IndexMap<_, _>>>()?;
    let api_key = provider.auth.credential.clone().unwrap_or_default();
    let auth_mode = match provider.auth.scheme {
        AuthScheme::Header => ProviderAuthMode::ApiKey,
        _ => ProviderAuthMode::Bearer,
    };
    let retry = provider.retry.as_ref().map(|retry| RetryConfig {
        enabled: retry.enabled,
        max_attempts: retry.max_attempts,
        max_recovery_attempts: retry.max_recovery_attempts,
        initial_delay_secs: retry.initial_delay_secs,
        exponential_backoff: retry.exponential_backoff,
        backoff_multiplier: retry.backoff_multiplier,
        jitter_secs: retry.jitter_secs,
    });
    Ok(ProviderConfig {
        base_url: provider.endpoint.clone(),
        api_key,
        auth_mode,
        protocol,
        default_model,
        retry,
        models,
    })
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
    let exponential_backoff = raw.exponential_backoff.unwrap_or(base.exponential_backoff);
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
        exponential_backoff,
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
        "config file not found: {}\n\nCreate it with at least:\n\nactive_provider = \"openai\"\n\n[providers.openai]\nprotocol = \"responses\"\ndefault_model = \"gpt-5.5\"\nflavor = \"standard\"\n\n[providers.openai.auth]\ntype = \"bearer\"\ncredential = \"YOUR_API_KEY\"\n\n[providers.openai.endpoints]\nbase_url = \"https://api.openai.com/v1\"\n\n[providers.openai.models.\"gpt-5.5\"]\ndisplay = \"GPT-5.5\"\n\n[providers.openai.models.\"gpt-5.5\".capabilities]\ntools = true\nreasoning = true\n\n[providers.openai.models.\"gpt-5.5\".capabilities.generation]\nmax_output_tokens = true\n\n[providers.openai.models.\"gpt-5.5\".generation]\nmax_output_tokens = 4096\n",
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
            let error = AppConfig::load_from_path(&write_temp_config(&format!(
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
            let error = AppConfig::load_from_path(&write_temp_config(&format!(
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
    fn capabilities_default_off_and_typed_cache_settings_validate() {
        let loaded = AppConfig::load_from_path(&write_temp_config(config("openai", "model", "")))
            .expect("minimal new config should load");
        let route = loaded.runtime_catalog.route("openai", "model").unwrap();
        assert_eq!(route.capabilities, RouteCapabilities::default());
        assert_eq!(route.generation, GenerationSupport::default());
        assert!(!route.cache.enabled);

        let invalid = config(
            "openai",
            "model",
            "[cache]\nenabled = false\nretention = \"24h\"\n",
        );
        let error = AppConfig::load_from_path(&write_temp_config(invalid));
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
        let path = write_temp_config(&format!(
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

    #[test]
    fn rejects_retry_attempts_above_maximum() {
        let error = AppConfig::load_from_path(&write_temp_config(config(
            "openai",
            "model",
            "[global.retry]\nmax_attempts = 9001\n",
        )))
        .expect_err("config should be rejected");
        assert!(format!("{error:#}").contains("global.retry.max_attempts must be at most 9000"));
    }

    #[test]
    fn provider_retry_can_override_backoff_mode_interval_and_attempts() {
        let loaded = AppConfig::load_from_path(&write_temp_config(
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
                AppConfig::load_from_path(&write_temp_config(config("openai", "model", extra)))
                    .is_err(),
                "invalid reasoning config should be rejected: {extra}"
            );
        }
    }

    #[test]
    fn model_parallel_tool_calls_default_to_disabled_and_can_be_enabled() {
        let default_config =
            AppConfig::load_from_path(&write_temp_config(config("openai", "model", "")))
                .expect("default config should load");
        assert!(!default_config.providers["openai"].models["model"].parallel_tool_calls);

        let enabled = config(
            "openai",
            "model",
            "[capabilities]\nparallel_tool_calls = true\ngeneration = { parallel_tool_calls = true }\n[generation]\nparallel_tool_calls = true",
        );
        let enabled_config = AppConfig::load_from_path(&write_temp_config(enabled))
            .expect("enabled config should load");
        assert!(enabled_config.providers["openai"].models["model"].parallel_tool_calls);
    }

    #[test]
    fn anthropic_betas_are_explicit_and_protocol_scoped() {
        let anthropic = config(
            "anyrouter",
            "claude-opus",
            "protocol = \"anthropic\"\n[protocol_settings]\nanthropic_betas = [\"context-1m-2025-08-07\"]",
        );
        let loaded = AppConfig::load_from_path(&write_temp_config(anthropic))
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
        let error = AppConfig::load_from_path(&write_temp_config(invalid))
            .expect_err("non-anthropic beta config should be rejected");
        assert!(format!("{error:#}").contains("protocol"));
    }

    #[test]
    fn anthropic_model_override_requires_provider_auth_mode() {
        let missing = config("mixed", "claude", "protocol = \"anthropic\"");
        let loaded = AppConfig::load_from_path(&write_temp_config(missing))
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
        let loaded = AppConfig::load_from_path(&write_temp_config(configured))
            .expect("configured override should load");
        assert_eq!(
            loaded.providers["mixed"].auth_mode,
            ProviderAuthMode::ApiKey
        );
    }

    #[test]
    fn rejects_zero_model_effective_input_limit_tokens() {
        let error = AppConfig::load_from_path(&write_temp_config(config(
            "openai",
            "model",
            "effective_input_limit_tokens = 0",
        )))
        .expect_err("config should be rejected");
        assert!(format!("{error:#}").contains("effective_input_limit_tokens"));
    }

    #[test]
    fn parses_provider_qualified_expert_routes_and_preserves_duplicate_model_ids() {
        let loaded = AppConfig::load_from_path(&write_temp_config(
            r#"active_provider = "primary"
[agents.explorer]
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
    fn parses_and_validates_expert_allowed_models() {
        let loaded = AppConfig::load_from_path(&write_temp_config(
            r#"active_provider = "primary"
[agents.explorer]
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
                AppConfig::load_from_path(&write_temp_config(text)).is_err(),
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
        assert!(AppConfig::load_from_path(&write_temp_config(incomplete)).is_err());

        let unknown_provider = config(
            "openai",
            "model",
            "[agents.explorer]\nprovider = \"missing\"\nmodel = \"model\"",
        );
        assert!(AppConfig::load_from_path(&write_temp_config(unknown_provider)).is_err());

        let unknown_model = config(
            "openai",
            "model",
            "[agents.explorer]\nprovider = \"openai\"\nmodel = \"missing\"",
        );
        assert!(AppConfig::load_from_path(&write_temp_config(unknown_model)).is_err());
    }

    #[test]
    fn rejects_unknown_subagent_name_and_model_override() {
        let unknown_agent = config("openai", "model", "[agents.nosuch]\nmodel = \"model\"");
        assert!(AppConfig::load_from_path(&write_temp_config(unknown_agent)).is_err());
        let unknown_model = config(
            "openai",
            "model",
            "[agents.explorer]\nmodel = \"missing-model\"",
        );
        assert!(AppConfig::load_from_path(&write_temp_config(unknown_model)).is_err());
    }

    #[test]
    fn rejects_reasoning_parameters_when_reasoning_is_disabled() {
        let error = AppConfig::load_from_path(&write_temp_config(config(
            "openai",
            "model",
            "[capabilities]\nreasoning = false\n[generation]\nreasoning_effort = \"medium\"",
        )))
        .expect_err("load should fail");
        assert!(format!("{error:#}").contains("reasoning capabilities"));
    }

    #[test]
    fn rejects_out_of_range_sampling_parameters() {
        let error = AppConfig::load_from_path(&write_temp_config(config(
            "openai",
            "model",
            "[capabilities]\ngeneration = { temperature = true }\n[generation]\ntemperature = 3.0",
        )))
        .expect_err("load should fail");
        assert!(format!("{error:#}").contains("temperature"));
    }

    #[test]
    fn persists_expert_allowed_models_without_rewriting_unrelated_config() {
        let path = write_temp_config(&format!(
            "# keep this comment\nactive_provider = \"primary\"\n\n[providers.primary]\nprotocol = \"responses\"\ndefault_model = \"old\"\nflavor = \"standard\"\n[providers.primary.auth]\ntype = \"bearer\"\ncredential = \"primary-key\"\n[providers.primary.endpoints]\nbase_url = \"https://primary.invalid/v1\"\n[providers.primary.models.old]\n\n[providers.expert]\nprotocol = \"responses\"\ndefault_model = \"shared\"\nflavor = \"standard\"\n[providers.expert.auth]\ntype = \"bearer\"\ncredential = \"expert-key\"\n[providers.expert.endpoints]\nbase_url = \"https://expert.invalid/v1\"\n[providers.expert.models.shared]\n\n[agents.explorer]\nmodel = \"old\"\n# preserve this trailing comment\n"
        ));
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

    #[cfg(unix)]
    #[test]
    fn persists_through_config_symlink_without_replacing_it() {
        use std::os::unix::fs::symlink;
        let target = write_temp_config(&format!(
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
        let error = AppConfig::load_from_path(&write_temp_config(&format!(
            "{}\n\n[mcp.empty]\ntype = \"local\"\ncommand = []",
            config("openai", "model", "")
        )))
        .expect_err("load should fail");
        assert!(format!("{error:#}").contains("mcp.empty.command cannot be empty"));
    }

    #[test]
    fn rejects_remote_mcp_oauth_until_supported() {
        let error = AppConfig::load_from_path(&write_temp_config(&format!(
            "{}\n\n[mcp.docs]\ntype = \"remote\"\nurl = \"https://example.invalid/mcp\"\noauth = true",
            config("openai", "model", "")
        )))
        .expect_err("load should fail");
        assert!(format!("{error:#}").contains("remote MCP OAuth is not supported"));
    }

    #[test]
    fn rejects_unknown_fields_and_zero_limits_and_empty_identifiers() {
        let root_unknown = format!("unexpected = true\n{}", config("openai", "model", ""));
        assert!(AppConfig::load_from_path(&write_temp_config(root_unknown)).is_err());
        let zero = format!(
            "[global]\nmax_iterations = 0\n\n{}",
            config("openai", "model", "")
        );
        assert!(AppConfig::load_from_path(&write_temp_config(zero)).is_err());
        let empty = config("openai", "model", "default_model = \"   \"");
        assert!(AppConfig::load_from_path(&write_temp_config(empty)).is_err());
    }

    #[test]
    fn errors_when_active_provider_is_missing() {
        let error =
            AppConfig::load_from_path(&write_temp_config(config("openai", "model", "").replace(
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
        assert!(AppConfig::load_from_path(&write_temp_config(disabled)).is_err());
        let completions = config(
            "openai",
            "model",
            "protocol = \"completions\"\n[cache]\nenabled = true\nretention = \"24h\"",
        );
        assert!(AppConfig::load_from_path(&write_temp_config(completions)).is_err());
        for namespace in ["   ", &"a".repeat(65), "valid\u{0007}name"] {
            let value = config(
                "openai",
                "model",
                &format!("[cache]\nenabled = true\nnamespace = {namespace:?}"),
            );
            assert!(AppConfig::load_from_path(&write_temp_config(value)).is_err());
        }
        let unknown = config(
            "openai",
            "model",
            "[cache]\nenabled = true\nlayout = \"v1\"",
        );
        assert!(AppConfig::load_from_path(&write_temp_config(unknown)).is_err());
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
}
