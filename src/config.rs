use crate::model_runtime::{
    AuthScheme, CacheRetention, ProtocolRegistry, ProtocolSettings, ProviderFlavor,
    ProviderReviewerBackend, ResolvedModelRoute, ResolvedProvider, RuntimeAuthConfig,
    RuntimeConfig, RuntimeEndpointOverride, RuntimeEndpoints, RuntimeModelConfig,
    RuntimeProviderConfig, RuntimeRetryConfig, RuntimeTransportConfig,
};
use crate::permission::PermissionMode;
use crate::request_builder::{
    ModelReasoningEffort, ModelReasoningSummary, ModelRequestMetadata, ModelTextVerbosity,
};
use crate::session::archive::SessionArchiveConfig;
use anyhow::{Context, Result, anyhow, bail};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::io::Write;
#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;
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
    #[cfg(test)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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
pub(crate) const DEFAULT_TOOL_TIMEOUT_SECS: u64 = 60;
const DEFAULT_SESSIONS_DIR: &str = "sessions";
const DEFAULT_LOG_FILE: &str = "logs/combined.log";
/// Editor directive (Taplo, Even Better TOML, Zed) naming the published
/// configuration schema. letcode itself ignores the comment.
const CONFIG_SCHEMA_COMMENT: &str =
    "#:schema https://raw.githubusercontent.com/letr007/letcode/main/letcode.schema.json";
/// Mirrors the Codex turn-metadata extension limits so declared extra keys
/// cannot exceed what a real client would send.
const MAX_FAKE_EXTRA_ENTRIES: usize = 16;
const MAX_FAKE_EXTRA_KEY_BYTES: usize = 64;
const MAX_FAKE_EXTRA_VALUE_BYTES: usize = 128;
/// Keys the Codex turn metadata owns. A declared `[fake.codex.extra]` entry may
/// not shadow them, mirroring the reserved-key rule the upstream metadata shape
/// applies.
const RESERVED_FAKE_EXTRA_KEYS: &[&str] = &[
    "installation_id",
    "x-codex-installation-id",
    "session_id",
    "thread_id",
    "agent_name",
    "turn_id",
    "window_id",
    "x-codex-window-id",
    "x-codex-turn-metadata",
    "x-codex-parent-thread-id",
    "x-openai-subagent",
    "request_kind",
    "root_turn_id",
    "parent_turn_id",
    "parent_thread_id",
    "forked_from_thread_id",
    "subagent_kind",
    "thread_source",
    "sandbox",
    "sandbox_mode",
    "auto_review_enabled",
    "node_repl_auto_review_required",
    "node_repl_disabled",
    "workspaces",
    "turn_started_at_unix_ms",
    "tool_namespaces_info",
    "code_mode_tool_names",
    "compaction",
];
const MAX_RETRY_ATTEMPTS: usize = 9_000;
const MAX_RECOVERY_ATTEMPTS: usize = 10;
mod persistence;
mod schema;
mod schema_gen;

use persistence::acquire_config_read_lock;
#[allow(unused_imports)]
pub use persistence::{
    ConfigEntry, ConfigEntryKind, ConfigScalar, config_array_in, config_entries_in,
    config_value_in, leaf_config_entry, persist_expert_allowed_models, persist_mcp_server_enabled,
    remove_config_table, save_config_document, set_config_item, set_config_scalar,
};
pub(crate) use persistence::{acquire_config_lock, replace_file, resolve_config_target};
#[cfg(test)]
pub use schema::schema_keys;
pub use schema::{
    default_value, entry_tables, field_enum, field_schema, schema_properties, table_accepts_entries,
};
pub(crate) use schema_gen::{SCHEMA_FILE, generate_schema};

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct AppConfig {
    pub config_path: PathBuf,
    pub config_dir: PathBuf,
    pub fast_mode_enabled: bool,
    pub active_provider: String,
    pub global: GlobalConfig,
    pub agents: AgentsConfig,
    pub subagents: SubagentsConfig,
    pub permissions: PermissionsConfig,
    pub tools: ToolsConfig,
    pub mcp: IndexMap<String, McpServerConfig>,
    pub providers: IndexMap<String, ProviderConfig>,
    pub runtime_catalog: crate::model_runtime::ResolvedRuntimeCatalog,
    pub fake: FakeConfig,
}

pub(crate) fn initialize_config(
    path: &Path,
    base_url: &str,
    api_key: &str,
    model: &str,
) -> Result<()> {
    if base_url.trim().is_empty() {
        bail!("Base URL cannot be empty");
    }
    if api_key.trim().is_empty() {
        bail!("API key cannot be empty");
    }
    if model.trim().is_empty() {
        bail!("Model cannot be empty");
    }
    if fs::symlink_metadata(path).is_ok() {
        bail!("configuration file already exists: {}", path.display());
    }

    let model = toml_string(model);
    let config_text = format!(
        "{CONFIG_SCHEMA_COMMENT}\n\nactive_provider = \"default\"\n\n[providers.default]\nprotocol = \"completions\"\ndefault_model = {model}\n\n[providers.default.auth]\ntype = \"bearer\"\ncredential = {}\n\n[providers.default.endpoints]\nbase_url = {}\n\n[providers.default.models.{model}.capabilities]\ntools = true\n",
        toml_string(api_key),
        toml_string(base_url),
    );
    AppConfig::load_from_str_at_path(path, &config_text)
        .context("generated setup configuration is invalid")?;

    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)
        .with_context(|| format!("failed to create config directory {}", parent.display()))?;
    let config_target = resolve_config_target(path)?;
    let _lock = acquire_config_lock(&config_target)?;
    if fs::symlink_metadata(path).is_ok() {
        bail!(
            "configuration file was created while setting up: {}",
            path.display()
        );
    }

    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    let temp_path = parent.join(format!(
        ".{}.{}.{}.setup.tmp",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("letcode.toml"),
        std::process::id(),
        stamp
    ));
    let mut options = fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    options.mode(0o600);
    let mut file = options
        .open(&temp_path)
        .with_context(|| format!("failed to create temporary config {}", temp_path.display()))?;
    let write_result = (|| -> Result<()> {
        file.write_all(config_text.as_bytes())
            .with_context(|| format!("failed to write temporary config {}", temp_path.display()))?;
        file.sync_all()
            .with_context(|| format!("failed to sync temporary config {}", temp_path.display()))?;
        Ok(())
    })();
    drop(file);
    if let Err(error) = write_result {
        let _ = fs::remove_file(&temp_path);
        return Err(error);
    }
    replace_file(&temp_path, path)
        .with_context(|| format!("failed to install config {}", path.display()))
}

fn toml_string(value: &str) -> String {
    toml::Value::String(value.to_owned()).to_string()
}

impl AppConfig {
    pub fn load() -> Result<Self> {
        Self::load_from_path(default_config_path()?)
    }

    pub fn load_from_path(path: impl AsRef<Path>) -> Result<Self> {
        let config_path = path.as_ref().to_path_buf();
        let config_target = resolve_config_target(&config_path).map_err(|error| {
            if error
                .downcast_ref::<std::io::Error>()
                .is_some_and(|cause| cause.kind() == std::io::ErrorKind::NotFound)
            {
                error.context(missing_config_message(&config_path))
            } else {
                error
            }
        })?;
        let _lock = acquire_config_read_lock(&config_target)?;
        let config_text = fs::read_to_string(&config_target).map_err(|error| {
            let message = if error.kind() == std::io::ErrorKind::NotFound {
                missing_config_message(&config_path)
            } else {
                format!("failed to read config file {}", config_path.display())
            };
            anyhow::Error::new(error).context(message)
        })?;
        Self::load_from_str_at_path(&config_path, &config_text)
    }

    pub(crate) fn load_from_str_at_path(config_path: &Path, config_text: &str) -> Result<Self> {
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
            .or(Some(DEFAULT_TOOL_TIMEOUT_SECS)),
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
            submit_while_running: raw_global.submit_while_running.unwrap_or_default(),
            compaction,
            retry: build_retry_config(raw_global.retry.unwrap_or_default(), "global.retry")?,
            session_archive: build_session_archive_config(
                raw_global.session_archive.unwrap_or_default(),
            )?,
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

        let permissions = build_permissions_config(raw.permissions.unwrap_or_default())?;
        let tools = build_tools_config(raw.tools.unwrap_or_default())?;
        let fake = build_fake_config(raw.fake.unwrap_or_default())?;
        let agents = build_agents_config(raw.agents.unwrap_or_default(), &providers)?;
        let subagents = build_subagents_config(raw.subagents.unwrap_or_default())?;
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
            subagents,
            permissions,
            tools,
            mcp,
            providers,
            runtime_catalog: resolved_catalog,
            fake,
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

    pub fn expert_route_for(&self, agent_name: &str) -> Option<ModelRoute> {
        if !crate::delegation::supported_agent_names().any(|name| name == agent_name) {
            return None;
        }
        self.agents
            .allowed_models_for(agent_name)
            .and_then(|models| models.first().cloned())
            .or_else(|| self.model_route_for(agent_name).cloned())
            .or_else(|| Some(self.active_route()))
    }

    pub fn expert_reasoning_efforts(&self) -> IndexMap<String, ModelReasoningEffort> {
        crate::delegation::supported_agent_names()
            .filter_map(|agent_name| {
                self.agents
                    .reasoning_effort_for(agent_name)
                    .map(|effort| (agent_name.to_string(), effort))
            })
            .collect()
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubagentMode {
    #[default]
    Expert,
    Swarm,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SubagentsConfig {
    pub mode: SubagentMode,
    pub max_depth: usize,
    #[serde(alias = "max_descendants")]
    pub max_concurrent: usize,
    pub max_pending_messages: usize,
    pub max_message_bytes: usize,
    pub run_timeout_secs: u64,
}

impl SubagentsConfig {
    pub(crate) fn validate(&self) -> Result<()> {
        positive_usize("subagents.max_depth", self.max_depth)?;
        positive_usize("subagents.max_concurrent", self.max_concurrent)?;
        positive_usize("subagents.max_pending_messages", self.max_pending_messages)?;
        positive_usize("subagents.max_message_bytes", self.max_message_bytes)?;
        positive_u64("subagents.run_timeout_secs", self.run_timeout_secs)?;
        if std::time::Instant::now()
            .checked_add(std::time::Duration::from_secs(self.run_timeout_secs))
            .is_none()
        {
            bail!("subagents.run_timeout_secs exceeds the supported deadline range");
        }
        Ok(())
    }
}

impl Default for SubagentsConfig {
    fn default() -> Self {
        Self {
            mode: SubagentMode::default(),
            max_depth: 3,
            max_concurrent: 16,
            max_pending_messages: 16,
            max_message_bytes: 16384,
            run_timeout_secs: 600,
        }
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
    pub historian: AgentConfig,
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

    pub fn reasoning_effort_for(&self, agent_name: &str) -> Option<ModelReasoningEffort> {
        self.config_for(agent_name)
            .and_then(|config| config.reasoning_effort.clone())
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
            "historian" => Some(&self.historian),
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
    pub reasoning_effort: Option<ModelReasoningEffort>,
}

#[derive(Debug, Clone)]
pub struct GlobalConfig {
    pub max_iterations: Option<usize>,
    pub max_tool_calls: Option<usize>,
    pub tool_timeout_secs: Option<u64>,
    pub sessions_dir: PathBuf,
    pub log_file: PathBuf,
    pub submit_while_running: SubmitWhileRunning,
    pub compaction: CompactionConfig,
    pub retry: RetryConfig,
    pub session_archive: SessionArchiveConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SubmitWhileRunning {
    /// Close the active turn at the next tool batch boundary and run the prompt next.
    #[default]
    Insert,
    /// Keep the prompt queued until the active turn finishes.
    Queue,
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

/// Typesafe Jev reviewer: `POST {base_url}/v1/systemone` with a `state` object
/// and one `choice` question, answered with probabilities instead of prose.
///
/// Resolved from the provider the reviewer route names, so the recorded route
/// and the backend that answered it are the same provider/model pair.
#[derive(Clone)]
pub struct JevReviewConfig {
    pub provider: String,
    pub base_url: String,
    pub endpoint: String,
    pub model: String,
    pub credential: String,
    pub timeout_secs: u64,
}

impl std::fmt::Debug for JevReviewConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JevReviewConfig")
            .field("provider", &self.provider)
            .field("base_url", &self.base_url)
            .field("endpoint", &self.endpoint)
            .field("model", &self.model)
            .field("credential", &"<redacted>")
            .field("timeout_secs", &self.timeout_secs)
            .finish()
    }
}

/// Declarative values used when the session-level `/fake` switch is on.
///
/// Every field is optional. Absent values resolve in [`crate::fake`] rather
/// than here: outward-facing attributes (OS, architecture, terminal, working
/// directory) come from the real host, identity values are generated, and
/// attributes letcode has no equivalent for fall back to the imitated client's
/// typical values.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FakeConfig {
    pub identity: FakeIdentityConfig,
    pub clock: FakeClockConfig,
    pub codex: FakeCodexConfig,
    pub claude: FakeClaudeConfig,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FakeIdentityConfig {
    pub installation_id: Option<String>,
    pub agent_name: Option<String>,
    /// Account id the Claude profile reports; derived when absent.
    pub account_uuid: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FakeClockConfig {
    /// IANA zone name (for example `Asia/Shanghai`). When absent the real
    /// host zone is used.
    pub timezone: Option<String>,
    /// Explicit `YYYY-MM-DD` override. When absent the date is derived from
    /// the real clock in the declared zone.
    pub date: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FakeCodexConfig {
    pub version: Option<String>,
    pub originator: Option<String>,
    pub os: Option<String>,
    pub arch: Option<String>,
    pub terminal: Option<String>,
    pub beta_features: Option<Vec<String>>,
    pub sandbox: Option<String>,
    pub sandbox_mode: Option<String>,
    pub auto_review_enabled: Option<bool>,
    pub node_repl_auto_review_required: Option<bool>,
    pub node_repl_disabled: Option<bool>,
    pub cwd: Option<String>,
    pub workspace: Option<String>,
    pub shell: Option<String>,
    pub git_commit_hash: Option<String>,
    pub git_remote_url: Option<String>,
    pub git_has_changes: Option<bool>,
    pub extra: IndexMap<String, String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FakeClaudeConfig {
    pub version: Option<String>,
    pub build: Option<String>,
    pub package_version: Option<String>,
    pub runtime_version: Option<String>,
    pub timeout: Option<String>,
    pub os: Option<String>,
    pub arch: Option<String>,
    pub betas: Option<Vec<String>>,
}

#[derive(Debug, Clone, Default)]
pub struct ToolsConfig {
    pub parallelism: IndexMap<String, crate::tool::ToolParallelism>,
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
    pub jev_endpoint: Option<String>,
    pub api_key: String,
    pub auth_mode: ProviderAuthMode,
    pub protocol: ApiProtocol,
    pub default_model: String,
    pub retry: Option<RetryConfig>,
    pub reviewer: Option<ProviderReviewerBackend>,
    pub models: IndexMap<String, ModelConfig>,
}

impl std::fmt::Debug for ProviderConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProviderConfig")
            .field("base_url", &self.base_url)
            .field("jev_endpoint", &self.jev_endpoint)
            .field("api_key", &"<redacted>")
            .field("auth_mode", &self.auth_mode)
            .field("protocol", &self.protocol)
            .field("default_model", &self.default_model)
            .field("retry", &self.retry)
            .field("reviewer", &self.reviewer)
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
    pub supports_input_images: bool,
    pub supports_tool_result_images: bool,
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
            supports_input_images: self.supports_input_images,
            supports_tool_result_images: self.supports_tool_result_images,
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
    subagents: Option<RawSubagentsConfig>,
    #[serde(default)]
    permissions: Option<RawPermissionsConfig>,
    #[serde(default)]
    tools: Option<RawToolsConfig>,
    #[serde(default)]
    mcp: IndexMap<String, RawMcpServerConfig>,
    #[serde(default)]
    providers: IndexMap<String, RawProviderConfig>,
    #[serde(default)]
    fake: Option<RawFakeConfig>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawFakeConfig {
    identity: Option<RawFakeIdentityConfig>,
    clock: Option<RawFakeClockConfig>,
    codex: Option<RawFakeCodexConfig>,
    claude: Option<RawFakeClaudeConfig>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawFakeIdentityConfig {
    installation_id: Option<String>,
    agent_name: Option<String>,
    account_uuid: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawFakeClockConfig {
    timezone: Option<String>,
    date: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawFakeCodexConfig {
    version: Option<String>,
    originator: Option<String>,
    os: Option<String>,
    arch: Option<String>,
    terminal: Option<String>,
    beta_features: Option<Vec<String>>,
    sandbox: Option<String>,
    sandbox_mode: Option<String>,
    auto_review_enabled: Option<bool>,
    node_repl_auto_review_required: Option<bool>,
    node_repl_disabled: Option<bool>,
    cwd: Option<String>,
    workspace: Option<String>,
    shell: Option<String>,
    git_commit_hash: Option<String>,
    git_remote_url: Option<String>,
    git_has_changes: Option<bool>,
    #[serde(default)]
    extra: IndexMap<String, String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawFakeClaudeConfig {
    version: Option<String>,
    build: Option<String>,
    package_version: Option<String>,
    runtime_version: Option<String>,
    timeout: Option<String>,
    os: Option<String>,
    arch: Option<String>,
    betas: Option<Vec<String>>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawGlobalConfig {
    max_iterations: Option<usize>,
    max_tool_calls: Option<usize>,
    tool_timeout_secs: Option<u64>,
    sessions_dir: Option<String>,
    log_file: Option<String>,
    submit_while_running: Option<SubmitWhileRunning>,
    compaction: Option<RawCompactionConfig>,
    retry: Option<RawRetryConfig>,
    session_archive: Option<RawSessionArchiveConfig>,
}

/// Idle session transcripts older than this are archived in the background.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawSessionArchiveConfig {
    enabled: Option<bool>,
    older_than_days: Option<u64>,
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
struct RawSubagentsConfig {
    mode: Option<SubagentMode>,
    max_depth: Option<usize>,
    #[serde(alias = "max_descendants")]
    max_concurrent: Option<usize>,
    max_pending_messages: Option<usize>,
    max_message_bytes: Option<usize>,
    run_timeout_secs: Option<u64>,
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
    historian: Option<RawAgentConfig>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawAgentConfig {
    provider: Option<String>,
    model: Option<String>,
    #[serde(default)]
    allowed_models: Vec<String>,
    reasoning_effort: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawPermissionsConfig {
    mode: Option<PermissionMode>,
}

/// Bounds one whole Jev review call, which sits on the permission path of the
/// turn that waits for it.
const DEFAULT_JEV_REVIEW_TIMEOUT_SECS: u64 = 10;

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawToolsConfig {
    #[serde(default)]
    parallelism: IndexMap<String, crate::tool::ToolParallelism>,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
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
    #[serde(default)]
    reviewer: Option<ProviderReviewerBackend>,
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
    #[serde(default, alias = "reviewer")]
    jev: RuntimeEndpointOverride,
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
                jev: endpoints.jev,
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
                    flavor: provider.flavor,
                    reviewer: provider.reviewer,
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

/// Reasoning levels the runtime names; providers may also accept their own.
pub const REASONING_EFFORTS: [&str; 7] =
    ["none", "minimal", "low", "medium", "high", "xhigh", "max"];

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
    // Reasoning effort is only offered when the route can express it: Anthropic needs
    // adaptive thinking, other protocols need the generation support flag, or selecting
    // an effort would fail every later turn in prepare.
    let reasoning_settable = match protocol {
        ApiProtocol::Anthropic => {
            anthropic_thinking.mode == crate::request_builder::AnthropicThinkingMode::Adaptive
        }
        _ => model.generation.reasoning,
    };
    let mut config = ModelConfig {
        display_name: model.display.clone(),
        protocol,
        anthropic_thinking,
        anthropic_betas,
        context_window: model.context_window,
        effective_input_limit_tokens: model.effective_input_limit_tokens,
        max_output_tokens: model.generation_defaults.max_output_tokens,
        supports_tools: model.capabilities.tools,
        supports_input_images: model.capabilities.input_images,
        supports_tool_result_images: model.capabilities.tool_result_images,
        supports_reasoning: model.capabilities.reasoning && reasoning_settable,
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
    };
    let mut metadata = config.request_metadata();
    model.strategy.normalize_request_metadata(&mut metadata);
    config.reasoning_effort = metadata.reasoning_effort;
    config.reasoning_efforts = metadata.reasoning_efforts;
    Ok(config)
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
        jev_endpoint: provider.jev_endpoint.clone(),
        api_key,
        auth_mode,
        protocol,
        default_model,
        retry,
        reviewer: provider.reviewer,
        models,
    })
}

fn build_permissions_config(raw: RawPermissionsConfig) -> Result<PermissionsConfig> {
    Ok(PermissionsConfig {
        mode: raw.mode.unwrap_or_default(),
    })
}

/// Typesafe Jev settings when `route` names a provider that declares
/// `reviewer = "jev"`. The provider supplies the endpoint and credential it is
/// reached with; the route supplies the model id.
pub(crate) fn jev_review_for_route(
    providers: &IndexMap<String, ProviderConfig>,
    route: &ModelRoute,
) -> Option<JevReviewConfig> {
    let provider = providers.get(&route.provider)?;
    if provider.reviewer != Some(ProviderReviewerBackend::Jev) {
        return None;
    }
    let endpoint = provider
        .jev_endpoint
        .clone()
        .unwrap_or_else(|| format!("{}/v1/systemone", provider.base_url.trim_end_matches('/')));
    Some(JevReviewConfig {
        provider: route.provider.clone(),
        base_url: provider.base_url.clone(),
        endpoint,
        model: route.model.clone(),
        credential: provider.api_key.clone(),
        timeout_secs: DEFAULT_JEV_REVIEW_TIMEOUT_SECS,
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

fn build_fake_config(raw: RawFakeConfig) -> Result<FakeConfig> {
    let identity = raw.identity.unwrap_or_default();
    let clock = raw.clock.unwrap_or_default();
    let codex = raw.codex.unwrap_or_default();
    let claude = raw.claude.unwrap_or_default();

    Ok(FakeConfig {
        identity: FakeIdentityConfig {
            installation_id: optional_non_empty(
                "fake.identity.installation_id",
                identity.installation_id,
            )?,
            agent_name: optional_non_empty("fake.identity.agent_name", identity.agent_name)?,
            account_uuid: optional_non_empty("fake.identity.account_uuid", identity.account_uuid)?,
        },
        clock: FakeClockConfig {
            timezone: optional_non_empty("fake.clock.timezone", clock.timezone)?,
            date: optional_non_empty("fake.clock.date", clock.date)?,
        },
        codex: FakeCodexConfig {
            version: optional_non_empty("fake.codex.version", codex.version)?,
            originator: optional_non_empty("fake.codex.originator", codex.originator)?,
            os: optional_non_empty("fake.codex.os", codex.os)?,
            arch: optional_non_empty("fake.codex.arch", codex.arch)?,
            terminal: optional_non_empty("fake.codex.terminal", codex.terminal)?,
            beta_features: codex.beta_features,
            sandbox: optional_non_empty("fake.codex.sandbox", codex.sandbox)?,
            sandbox_mode: optional_non_empty("fake.codex.sandbox_mode", codex.sandbox_mode)?,
            auto_review_enabled: codex.auto_review_enabled,
            node_repl_auto_review_required: codex.node_repl_auto_review_required,
            node_repl_disabled: codex.node_repl_disabled,
            cwd: optional_non_empty("fake.codex.cwd", codex.cwd)?,
            workspace: optional_non_empty("fake.codex.workspace", codex.workspace)?,
            shell: optional_non_empty("fake.codex.shell", codex.shell)?,
            git_commit_hash: optional_non_empty(
                "fake.codex.git_commit_hash",
                codex.git_commit_hash,
            )?,
            git_remote_url: optional_non_empty("fake.codex.git_remote_url", codex.git_remote_url)?,
            git_has_changes: codex.git_has_changes,
            extra: build_fake_extra(codex.extra)?,
        },
        claude: FakeClaudeConfig {
            version: optional_non_empty("fake.claude.version", claude.version)?,
            build: optional_non_empty("fake.claude.build", claude.build)?,
            package_version: optional_non_empty(
                "fake.claude.package_version",
                claude.package_version,
            )?,
            runtime_version: optional_non_empty(
                "fake.claude.runtime_version",
                claude.runtime_version,
            )?,
            timeout: optional_non_empty("fake.claude.timeout", claude.timeout)?,
            os: optional_non_empty("fake.claude.os", claude.os)?,
            arch: optional_non_empty("fake.claude.arch", claude.arch)?,
            betas: claude.betas,
        },
    })
}

fn build_fake_extra(extra: IndexMap<String, String>) -> Result<IndexMap<String, String>> {
    if extra.len() > MAX_FAKE_EXTRA_ENTRIES {
        bail!("fake.codex.extra may contain at most {MAX_FAKE_EXTRA_ENTRIES} entries");
    }
    for (key, value) in &extra {
        if key.len() > MAX_FAKE_EXTRA_KEY_BYTES || !valid_fake_extra_key(key) {
            bail!("fake.codex.extra keys must be short ASCII identifiers: '{key}'");
        }
        if value.len() > MAX_FAKE_EXTRA_VALUE_BYTES {
            bail!(
                "fake.codex.extra values may contain at most {MAX_FAKE_EXTRA_VALUE_BYTES} bytes: '{key}'"
            );
        }
        if RESERVED_FAKE_EXTRA_KEYS.contains(&key.as_str()) {
            bail!("fake.codex.extra may not declare the reserved key '{key}'");
        }
    }
    Ok(extra)
}

fn valid_fake_extra_key(key: &str) -> bool {
    let mut bytes = key.bytes();
    bytes.next().is_some_and(|byte| byte.is_ascii_alphabetic())
        && bytes.all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'.' | b'-'))
}

/// A declared but blank string is treated as absent, so an empty value is never
/// emitted on the wire by accident.
fn optional_non_empty(label: &str, value: Option<String>) -> Result<Option<String>> {
    value
        .map(|value| required_non_empty(label, value))
        .transpose()
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

fn build_subagents_config(raw: RawSubagentsConfig) -> Result<SubagentsConfig> {
    let defaults = SubagentsConfig::default();
    let config = SubagentsConfig {
        mode: raw.mode.unwrap_or(defaults.mode),
        max_depth: raw.max_depth.unwrap_or(defaults.max_depth),
        max_concurrent: raw.max_concurrent.unwrap_or(defaults.max_concurrent),
        max_pending_messages: raw
            .max_pending_messages
            .unwrap_or(defaults.max_pending_messages),
        max_message_bytes: raw.max_message_bytes.unwrap_or(defaults.max_message_bytes),
        run_timeout_secs: raw.run_timeout_secs.unwrap_or(defaults.run_timeout_secs),
    };
    config.validate()?;
    Ok(config)
}

fn build_agents_config(
    raw: RawAgentsConfig,
    providers: &IndexMap<String, ProviderConfig>,
) -> Result<AgentsConfig> {
    Ok(AgentsConfig {
        explorer: build_agent_config(raw.explorer, "explorer", providers)?,
        fixer: build_agent_config(raw.fixer, "fixer", providers)?,
        oracle: build_agent_config(raw.oracle, "oracle", providers)?,
        designer: build_agent_config(raw.designer, "designer", providers)?,
        librarian: build_agent_config(raw.librarian, "librarian", providers)?,
        general: build_agent_config(raw.general, "general", providers)?,
        reviewer: build_agent_config(raw.reviewer, "reviewer", providers)?,
        historian: build_agent_config(raw.historian, "historian", providers)?,
    })
}

fn build_agent_config(
    raw: Option<RawAgentConfig>,
    agent_name: &str,
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

    let route = match (provider_name, model) {
        (None, None) => None,
        (Some(provider), Some(model)) => Some(ModelRoute::new(provider, model)),
        (None, Some(_)) => bail!("agents.{agent_name}.model requires agents.{agent_name}.provider"),
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

    let reasoning_effort = raw
        .reasoning_effort
        .map(|value| {
            let value =
                required_non_empty(&format!("agents.{agent_name}.reasoning_effort"), value)?;
            parse_reasoning_effort(&value).with_context(|| {
                format!("agents.{agent_name}.reasoning_effort is not a known level")
            })
        })
        .transpose()?;

    Ok(AgentConfig {
        route,
        allowed_models,
        reasoning_effort,
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

fn build_session_archive_config(raw: RawSessionArchiveConfig) -> Result<SessionArchiveConfig> {
    let defaults = SessionArchiveConfig::default();
    Ok(SessionArchiveConfig {
        enabled: raw.enabled.unwrap_or(defaults.enabled),
        older_than_days: raw
            .older_than_days
            .map(|days| positive_u64("global.session_archive.older_than_days", days))
            .transpose()?
            .unwrap_or(defaults.older_than_days),
    })
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
        "config file not found: {}\n\nCreate it with at least:\n\n{CONFIG_SCHEMA_COMMENT}\n\nactive_provider = \"openai\"\n\n[providers.openai]\nprotocol = \"responses\"\ndefault_model = \"gpt-5.5\"\nflavor = \"standard\"\n\n[providers.openai.auth]\ntype = \"bearer\"\ncredential = \"YOUR_API_KEY\"\n\n[providers.openai.endpoints]\nbase_url = \"https://api.openai.com/v1\"\n\n[providers.openai.models.\"gpt-5.5\"]\ndisplay = \"GPT-5.5\"\n\n[providers.openai.models.\"gpt-5.5\".capabilities]\ntools = true\nreasoning = true\n\n[providers.openai.models.\"gpt-5.5\".capabilities.generation]\nmax_output_tokens = true\n\n[providers.openai.models.\"gpt-5.5\".generation]\nmax_output_tokens = 4096\n",
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
#[path = "../tests/internal/config/tests.rs"]
mod tests;

#[cfg(test)]
#[path = "../tests/internal/config/historian_config_tests.rs"]
mod historian_config_tests;
