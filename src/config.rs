use anyhow::{Context, Result, anyhow, bail};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use toml_edit::{DocumentMut, Item, value};

#[cfg(unix)]
use std::os::fd::AsRawFd;
#[cfg(unix)]
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};

use crate::permission::PermissionMode;
use crate::request_builder::{
    ModelReasoningEffort, ModelReasoningSummary, ModelRequestMetadata, ModelTextVerbosity,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ApiProtocol {
    Responses,
    Completions,
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

impl Default for ApiProtocol {
    fn default() -> Self {
        Self::Responses
    }
}

const DEFAULT_CONFIG_HOME_RELATIVE_PATH: &str = ".config/letcode/letcode.toml";
const DEFAULT_OPENAI_BASE_URL: &str = "https://api.openai.com/v1";
const DEFAULT_MCP_TIMEOUT_MS: u64 = 5_000;
const DEFAULT_SESSIONS_DIR: &str = "sessions";
const DEFAULT_LOG_FILE: &str = "logs/combined.log";
const MAX_RETRY_ATTEMPTS: usize = 10;
const MAX_RETRY_DELAY_MS: u64 = 60_000;
const MAX_RETRY_ELAPSED_MS: u64 = 300_000;
const MAX_RECOVERY_ATTEMPTS: usize = 10;
static MCP_CONFIG_WRITE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Persist one configured MCP server's enabled state without rewriting unrelated
/// configuration content.
pub fn persist_mcp_server_enabled(
    config_path: &Path,
    server_name: &str,
    enabled: bool,
) -> Result<McpServerConfig> {
    // Resolve before editing so a config symlink remains a symlink while its
    // existing target is atomically replaced.
    let config_target = fs::canonicalize(config_path)
        .with_context(|| format!("failed to resolve config file {}", config_path.display()))?;
    // The lock lives beside the canonical target, so it survives the target's
    // atomic replacement and serializes all cooperating letcode writers.
    let _lock = acquire_config_lock(&config_target)?;
    let mut config_file = fs::File::open(&config_target)
        .with_context(|| format!("failed to open config file {}", config_target.display()))?;
    let original_metadata = config_file
        .metadata()
        .with_context(|| format!("failed to stat config file {}", config_target.display()))?;
    let mut config_text = String::new();
    config_file
        .read_to_string(&mut config_text)
        .with_context(|| format!("failed to read config file {}", config_target.display()))?;
    let mut document = config_text
        .parse::<DocumentMut>()
        .with_context(|| format!("failed to parse config file {}", config_target.display()))?;
    let mcp = document
        .get_mut("mcp")
        .and_then(Item::as_table_like_mut)
        .ok_or_else(|| anyhow!("config does not define an [mcp] table"))?;
    let server = mcp
        .get_mut(server_name)
        .and_then(Item::as_table_like_mut)
        .ok_or_else(|| anyhow!("MCP server '{server_name}' is not a configured table"))?;
    server.insert("enabled", value(enabled));

    let updated_config = document.to_string();
    let raw_document = toml::from_str::<toml::Value>(&updated_config)
        .context("failed to parse updated MCP server configuration")?;
    let raw_server = raw_document
        .get("mcp")
        .and_then(toml::Value::as_table)
        .and_then(|mcp| mcp.get(server_name))
        .cloned()
        .ok_or_else(|| anyhow!("updated MCP server '{server_name}' is missing"))?
        .try_into::<RawMcpServerConfig>()
        .with_context(|| format!("failed to parse MCP server '{server_name}'"))?;
    let (_, persisted_server) = build_mcp_server_config(server_name, raw_server)?;

    atomic_write_config(
        &config_target,
        &config_text,
        &original_metadata,
        updated_config.as_bytes(),
    )?;
    Ok(persisted_server)
}

fn atomic_write_config(
    config_path: &Path,
    original_contents: &str,
    original_metadata: &fs::Metadata,
    contents: &[u8],
) -> Result<()> {
    let parent = config_path.parent().unwrap_or_else(|| Path::new("."));
    let file_name = config_path
        .file_name()
        .ok_or_else(|| anyhow!("config path has no file name: {}", config_path.display()))?;
    let temp_path = parent.join(format!(
        ".{}.{}.{}.tmp",
        file_name.to_string_lossy(),
        std::process::id(),
        MCP_CONFIG_WRITE_SEQUENCE.fetch_add(1, Ordering::Relaxed)
    ));
    let write_result = (|| -> Result<()> {
        let mut temp = create_config_temp_file(&temp_path, original_metadata)?;
        temp.write_all(contents).with_context(|| {
            format!(
                "failed to write temporary config file {}",
                temp_path.display()
            )
        })?;
        temp.sync_all().with_context(|| {
            format!(
                "failed to sync temporary config file {}",
                temp_path.display()
            )
        })?;
        revalidate_config_source(config_path, original_contents, original_metadata)?;
        fs::rename(&temp_path, config_path).with_context(|| {
            format!(
                "failed to atomically replace config file {} with {}",
                config_path.display(),
                temp_path.display()
            )
        })?;
        Ok(())
    })();
    if write_result.is_err() {
        let _ = fs::remove_file(&temp_path);
    }
    write_result
}

fn create_config_temp_file(temp_path: &Path, source_metadata: &fs::Metadata) -> Result<fs::File> {
    let mut options = fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        // `mode` is applied by the creation syscall, before the path can be
        // observed. The umask may make it more restrictive, never broader.
        options.mode(source_metadata.mode() & 0o777);
    }
    let temp = options.open(temp_path).with_context(|| {
        format!(
            "failed to create temporary config file {}",
            temp_path.display()
        )
    })?;
    // Restore the exact original permissions when a restrictive umask changed
    // them. This occurs only after safe initial creation.
    temp.set_permissions(source_metadata.permissions())
        .with_context(|| {
            format!(
                "failed to preserve config permissions for {}",
                temp_path.display()
            )
        })?;
    Ok(temp)
}

fn revalidate_config_source(
    config_path: &Path,
    original_contents: &str,
    original_metadata: &fs::Metadata,
) -> Result<()> {
    let current_metadata = fs::metadata(config_path)
        .with_context(|| format!("failed to restat config file {}", config_path.display()))?;
    let current_contents = fs::read_to_string(config_path)
        .with_context(|| format!("failed to reread config file {}", config_path.display()))?;
    if current_contents != original_contents
        || !config_metadata_matches(original_metadata, &current_metadata)
    {
        bail!(
            "config file {} changed while updating MCP server state; refusing to overwrite it",
            config_path.display()
        );
    }
    Ok(())
}

fn config_metadata_matches(expected: &fs::Metadata, current: &fs::Metadata) -> bool {
    if expected.len() != current.len() || expected.modified().ok() != current.modified().ok() {
        return false;
    }
    #[cfg(unix)]
    {
        expected.dev() == current.dev() && expected.ino() == current.ino()
    }
    #[cfg(not(unix))]
    {
        true
    }
}

struct ConfigLock {
    _file: fs::File,
}

fn acquire_config_lock(config_target: &Path) -> Result<ConfigLock> {
    let lock_path = config_lock_path(config_target)?;
    let file = open_config_lock_file(&lock_path)?;
    lock_file(&file)?;
    Ok(ConfigLock { _file: file })
}

fn config_lock_path(config_target: &Path) -> Result<PathBuf> {
    let parent = config_target.parent().unwrap_or_else(|| Path::new("."));
    let file_name = config_target
        .file_name()
        .ok_or_else(|| anyhow!("config path has no file name: {}", config_target.display()))?;
    Ok(parent.join(format!(".{}.lock", file_name.to_string_lossy())))
}

fn open_config_lock_file(lock_path: &Path) -> Result<fs::File> {
    #[cfg(unix)]
    {
        let mut create = fs::OpenOptions::new();
        create
            .read(true)
            .write(true)
            .create_new(true)
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW);
        match create.open(lock_path) {
            Ok(file) => return Ok(file),
            Err(error) if error.kind() != std::io::ErrorKind::AlreadyExists => {
                return Err(error).with_context(|| {
                    format!("failed to create config lock file {}", lock_path.display())
                });
            }
            Err(_) => {}
        }

        let mut open = fs::OpenOptions::new();
        open.read(true).write(true).custom_flags(libc::O_NOFOLLOW);
        let file = open
            .open(lock_path)
            .with_context(|| format!("failed to open config lock file {}", lock_path.display()))?;
        if !file
            .metadata()
            .with_context(|| format!("failed to stat config lock file {}", lock_path.display()))?
            .is_file()
        {
            bail!(
                "config lock path is not a regular file: {}",
                lock_path.display()
            );
        }
        return Ok(file);
    }

    #[cfg(not(unix))]
    {
        fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(lock_path)
            .with_context(|| format!("failed to open config lock file {}", lock_path.display()))
    }
}

#[cfg(unix)]
fn lock_file(file: &fs::File) -> Result<()> {
    let result = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) };
    if result != 0 {
        return Err(std::io::Error::last_os_error()).context("failed to lock config lock file");
    }
    Ok(())
}

#[cfg(not(unix))]
fn lock_file(_file: &fs::File) -> Result<()> {
    // std does not expose a cross-platform advisory file lock. Source
    // revalidation detects changes before replacement but cannot serialize a
    // writer that races between revalidation and rename.
    Ok(())
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct AppConfig {
    pub config_path: PathBuf,
    pub config_dir: PathBuf,
    pub active_provider: String,
    pub global: GlobalConfig,
    pub agents: AgentsConfig,
    pub permissions: PermissionsConfig,
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
            active_provider,
            global,
            agents,
            permissions,
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

    pub fn active_provider_api_key_env_var(&self) -> String {
        provider_env_var(&self.active_provider, "API_KEY")
    }

    pub fn active_provider_model_label(&self, model_id: &str) -> String {
        let (_, provider) = self.active_provider();
        provider.model_label(model_id)
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
}

impl AgentsConfig {
    pub fn model_for(&self, agent_name: &str) -> Option<&str> {
        match agent_name {
            "explorer" => self.explorer.model.as_deref(),
            "fixer" => self.fixer.model.as_deref(),
            "oracle" => self.oracle.model.as_deref(),
            "designer" => self.designer.model.as_deref(),
            "librarian" => self.librarian.model.as_deref(),
            "general" => self.general.model.as_deref(),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct AgentConfig {
    pub model: Option<String>,
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompactionConfig {
    /// Tokens to retain at the end of history. When absent, use the selected
    /// model's active input budget.
    pub preserve_recent_tokens: Option<u64>,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            preserve_recent_tokens: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RetryConfig {
    pub enabled: bool,
    pub max_attempts: usize,
    pub max_elapsed_ms: u64,
    pub max_recovery_attempts: usize,
    pub initial_delay_ms: u64,
    pub max_delay_ms: u64,
    pub backoff_multiplier: f32,
    pub jitter_ms: u64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_attempts: 3,
            max_elapsed_ms: 30_000,
            max_recovery_attempts: 3,
            initial_delay_ms: 250,
            max_delay_ms: 2_000,
            backoff_multiplier: 2.0,
            jitter_ms: 100,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PermissionsConfig {
    pub mode: PermissionMode,
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
#[derive(Debug, Clone)]
pub struct ProviderConfig {
    pub base_url: String,
    pub api_key: String,
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
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub display_name: Option<String>,
    pub protocol: ApiProtocol,
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
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawAppConfig {
    #[serde(default)]
    active_provider: Option<String>,
    #[serde(default)]
    global: Option<RawGlobalConfig>,
    #[serde(default)]
    agents: Option<RawAgentsConfig>,
    #[serde(default)]
    permissions: Option<RawPermissionsConfig>,
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
    max_elapsed_ms: Option<u64>,
    max_recovery_attempts: Option<usize>,
    initial_delay_ms: Option<u64>,
    max_delay_ms: Option<u64>,
    backoff_multiplier: Option<f32>,
    jitter_ms: Option<u64>,
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
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawAgentConfig {
    model: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawPermissionsConfig {
    mode: Option<PermissionMode>,
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
            "provider '{}' must set protocol to 'responses' or 'completions'",
            name
        ),
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
    if let Some(max_output_tokens) = raw.max_output_tokens {
        if max_output_tokens > u32::MAX as u64 {
            bail!(
                "providers.{provider_name}.models.{model_id}.max_output_tokens must be at most {}",
                u32::MAX
            );
        }
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
    let provider = providers
        .get(active_provider)
        .expect("active provider should be validated at load time");

    Ok(AgentsConfig {
        explorer: build_agent_config(raw.explorer, "explorer", active_provider, provider)?,
        fixer: build_agent_config(raw.fixer, "fixer", active_provider, provider)?,
        oracle: build_agent_config(raw.oracle, "oracle", active_provider, provider)?,
        designer: build_agent_config(raw.designer, "designer", active_provider, provider)?,
        librarian: build_agent_config(raw.librarian, "librarian", active_provider, provider)?,
        general: build_agent_config(raw.general, "general", active_provider, provider)?,
    })
}

fn build_agent_config(
    raw: Option<RawAgentConfig>,
    agent_name: &str,
    active_provider: &str,
    provider: &ProviderConfig,
) -> Result<AgentConfig> {
    let Some(raw) = raw else {
        return Ok(AgentConfig::default());
    };

    let model = raw
        .model
        .map(|value| required_non_empty(&format!("agents.{agent_name}.model"), value))
        .transpose()?;

    if let Some(model_id) = &model {
        if !provider.has_model(model_id) {
            bail!(
                "agents.{agent_name}.model '{}' is not defined under [providers.{}.models]",
                model_id,
                active_provider
            );
        }
    }

    Ok(AgentConfig { model })
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
    let max_elapsed_ms = positive_u64(
        &format!("{path}.max_elapsed_ms"),
        raw.max_elapsed_ms.unwrap_or(base.max_elapsed_ms),
    )?;
    if max_elapsed_ms > MAX_RETRY_ELAPSED_MS {
        bail!("{path}.max_elapsed_ms must be at most {MAX_RETRY_ELAPSED_MS}");
    }
    let max_recovery_attempts = positive_usize(
        &format!("{path}.max_recovery_attempts"),
        raw.max_recovery_attempts
            .unwrap_or(base.max_recovery_attempts),
    )?;
    if max_recovery_attempts > MAX_RECOVERY_ATTEMPTS {
        bail!("{path}.max_recovery_attempts must be at most {MAX_RECOVERY_ATTEMPTS}");
    }
    let initial_delay_ms = positive_u64(
        &format!("{path}.initial_delay_ms"),
        raw.initial_delay_ms.unwrap_or(base.initial_delay_ms),
    )?;
    let max_delay_ms = positive_u64(
        &format!("{path}.max_delay_ms"),
        raw.max_delay_ms.unwrap_or(base.max_delay_ms),
    )?;
    if max_delay_ms > MAX_RETRY_DELAY_MS {
        bail!("{path}.max_delay_ms must be at most {MAX_RETRY_DELAY_MS}");
    }
    if max_delay_ms < initial_delay_ms {
        bail!("{path}.max_delay_ms must be greater than or equal to initial_delay_ms");
    }
    let jitter_ms = raw.jitter_ms.unwrap_or(base.jitter_ms);
    if jitter_ms > max_delay_ms {
        bail!("{path}.jitter_ms must be less than or equal to max_delay_ms");
    }
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
        max_elapsed_ms,
        max_recovery_attempts,
        initial_delay_ms,
        max_delay_ms,
        backoff_multiplier,
        jitter_ms,
    })
}

fn missing_config_message(path: &Path) -> String {
    format!(
        "config file not found: {}\n\nCreate it with at least:\n\nactive_provider = \"openai\"\n\n[global]\n# Optional runtime limits:\n# max_iterations = 64\n# max_tool_calls = 128\n# tool_timeout_secs = 60\nsessions_dir = \"sessions\"\nlog_file = \"logs/combined.log\"\n\n[global.compaction]\n# preserve_recent_tokens defaults to the active model input budget\n# preserve_recent_tokens = 4096\n\n[global.retry]\nenabled = true\nmax_attempts = 3\nmax_elapsed_ms = 30000\nmax_recovery_attempts = 3\ninitial_delay_ms = 250\nmax_delay_ms = 2000\nbackoff_multiplier = 2.0\njitter_ms = 100\n\n[permissions]\nmode = \"default\"\n\n[providers.openai]\napi_key = \"YOUR_API_KEY\"\nbase_url = \"https://api.openai.com/v1\"\nprotocol = \"responses\"\ndefault_model = \"gpt-5.5\"\n\n# Optional provider-specific retry override:\n# [providers.openai.retry]\n# enabled = true\n# max_attempts = 3\n# max_elapsed_ms = 30000\n# max_recovery_attempts = 3\n# initial_delay_ms = 250\n# max_delay_ms = 2000\n# backoff_multiplier = 2.0\n# jitter_ms = 100\n\n[providers.openai.models.\"gpt-5.5\"]\ndisplay_name = \"GPT-5.5\"\nsupports_tools = true\nsupports_reasoning = true\nreasoning_effort = \"medium\"\n# Optional per-model selectable levels and TUI cycle order:\n# reasoning_efforts = [\"none\", \"low\", \"medium\", \"high\", \"max\"]\nreasoning_summary = \"auto\"\ntext_verbosity = \"medium\"\n\n# OpenAI-compatible Chat Completions provider:\n# [providers.compat]\n# api_key = \"YOUR_API_KEY\"\n# base_url = \"https://example.com/v1\"\n# protocol = \"completions\"\n# default_model = \"your-model\"\n",
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
        assert_eq!(config.global.tool_timeout_secs, Some(60));
        assert_eq!(config.global.compaction, CompactionConfig::default());
        assert_eq!(config.global.retry, RetryConfig::default());
    }

    #[test]
    fn parses_global_tool_timeout_config() {
        let _guard = lock_env();
        let path = write_temp_config(
            r#"
            [global]
            tool_timeout_secs = 9

            [providers.openai]
            api_key = "config-key"

            [providers.openai.models."gpt-5.5"]
            name = "GPT-5.5"
            "#,
        );

        let config = AppConfig::load_from_path(&path).expect("config should load");
        assert_eq!(config.global.tool_timeout_secs, Some(9));
    }

    #[test]
    fn parses_global_compaction_config() {
        let _guard = lock_env();
        let path = write_temp_config(
            r#"
            [global.compaction]
            preserve_recent_tokens = 2048

            [providers.openai]
            api_key = "config-key"

            [providers.openai.models."gpt-5.5"]
            name = "GPT-5.5"
            "#,
        );

        let config = AppConfig::load_from_path(&path).expect("config should load");
        assert_eq!(
            config.global.compaction,
            CompactionConfig {
                preserve_recent_tokens: Some(2048),
            }
        );
    }

    #[test]
    fn missing_config_message_mentions_compaction_section() {
        let message = missing_config_message(Path::new("/tmp/missing.toml"));
        assert!(message.contains("[global.compaction]"));
        assert!(message.contains("active model input budget"));
        assert!(message.contains("[global.retry]"));
        assert!(message.contains("max_attempts = 3"));
        assert!(message.contains("# Optional runtime limits:"));
        assert!(message.contains("tool_timeout_secs = 60"));
    }

    #[test]
    fn omitting_global_runtime_limits_leaves_them_unbounded() {
        let _guard = lock_env();
        let path = write_temp_config(
            r#"
            active_provider = "openai"

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
        assert_eq!(config.global.max_iterations, None);
        assert_eq!(config.global.max_tool_calls, None);
    }

    #[test]
    fn parses_global_retry_config() {
        let _guard = lock_env();
        let path = write_temp_config(
            r#"
            [global.retry]
            enabled = false
            max_attempts = 4
            initial_delay_ms = 100
            max_delay_ms = 800
            backoff_multiplier = 1.5
            jitter_ms = 0

            [providers.openai]
            api_key = "config-key"

            [providers.openai.models."gpt-5.5"]
            name = "GPT-5.5"
            "#,
        );

        let config = AppConfig::load_from_path(&path).expect("config should load");
        assert_eq!(
            config.global.retry,
            RetryConfig {
                enabled: false,
                max_attempts: 4,
                max_elapsed_ms: 30_000,
                max_recovery_attempts: 3,
                initial_delay_ms: 100,
                max_delay_ms: 800,
                backoff_multiplier: 1.5,
                jitter_ms: 0,
            }
        );
    }

    #[test]
    fn parses_provider_retry_override() {
        let _guard = lock_env();
        let path = write_temp_config(
            r#"
            [global.retry]
            enabled = false
            max_attempts = 2
            initial_delay_ms = 100
            max_delay_ms = 10000
            backoff_multiplier = 3.0
            jitter_ms = 20

            [providers.openai]
            api_key = "config-key"

            [providers.openai.retry]
            max_attempts = 5

            [providers.openai.models."gpt-5.5"]
            name = "GPT-5.5"
            "#,
        );

        let config = AppConfig::load_from_path(&path).expect("config should load");
        assert_eq!(config.global.retry.max_attempts, 2);
        let (_, provider) = config.active_provider();
        assert_eq!(
            provider.retry,
            Some(RetryConfig {
                enabled: false,
                max_attempts: 5,
                max_elapsed_ms: 30_000,
                max_recovery_attempts: 3,
                initial_delay_ms: 100,
                max_delay_ms: 10000,
                backoff_multiplier: 3.0,
                jitter_ms: 20,
            })
        );
    }

    #[test]
    fn rejects_invalid_global_retry_config() {
        let _guard = lock_env();
        let path = write_temp_config(
            r#"
            [global.retry]
            initial_delay_ms = 1000
            max_delay_ms = 500

            [providers.openai]
            api_key = "config-key"

            [providers.openai.models."gpt-5.5"]
            name = "GPT-5.5"
            "#,
        );

        let error = AppConfig::load_from_path(&path).expect_err("config should be rejected");
        assert!(error.to_string().contains(
            "global.retry.max_delay_ms must be greater than or equal to initial_delay_ms"
        ));
    }

    #[test]
    fn rejects_invalid_provider_retry_config_with_provider_path() {
        let _guard = lock_env();
        let path = write_temp_config(
            r#"
            [providers.openai]
            api_key = "config-key"

            [providers.openai.retry]
            initial_delay_ms = 1000
            max_delay_ms = 500

            [providers.openai.models."gpt-5.5"]
            name = "GPT-5.5"
            "#,
        );

        let error = AppConfig::load_from_path(&path).expect_err("config should be rejected");
        assert!(error.to_string().contains(
            "providers.openai.retry.max_delay_ms must be greater than or equal to initial_delay_ms"
        ));
    }

    #[test]
    fn rejects_retry_jitter_larger_than_max_delay() {
        let _guard = lock_env();
        let path = write_temp_config(
            r#"
            [global.retry]
            max_delay_ms = 500
            jitter_ms = 501

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
                .contains("global.retry.jitter_ms must be less than or equal to max_delay_ms")
        );
    }

    #[test]
    fn rejects_retry_limits_above_safety_bounds() {
        let _guard = lock_env();
        let path = write_temp_config(
            r#"
            [global.retry]
            max_attempts = 11

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
                .contains("global.retry.max_attempts must be at most 10")
        );

        let path = write_temp_config(
            r#"
            [global.retry]
            max_delay_ms = 60001

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
                .contains("global.retry.max_delay_ms must be at most 60000")
        );
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
        assert_eq!(provider.protocol, ApiProtocol::Responses);
        assert_eq!(provider.models["gpt-5.5"].protocol, ApiProtocol::Responses);
    }

    #[test]
    fn parses_model_generation_parameters() {
        let _guard = lock_env();
        let path = write_temp_config(
            r#"
            [providers.openai]
            api_key = "config-key"

            [providers.openai.models."gpt-5.5"]
            name = "GPT-5.5"
            context_window = 400000
            effective_input_limit_tokens = 256000
            max_output_tokens = 8192
            supports_reasoning = true
            reasoning_effort = "high"
            reasoning_summary = "auto"
            text_verbosity = "low"
            temperature = 0.2
            top_p = 0.8
            "#,
        );

        let config = AppConfig::load_from_path(&path).expect("config should load");
        let (_, provider) = config.active_provider();
        let model = &provider.models["gpt-5.5"];

        assert_eq!(model.context_window, Some(400000));
        assert_eq!(model.effective_input_limit_tokens, Some(256000));
        assert_eq!(model.max_output_tokens, Some(8192));
        assert_eq!(model.reasoning_effort, Some(ModelReasoningEffort::High));
        assert!(model.reasoning_efforts.is_empty());
        assert_eq!(model.reasoning_summary, Some(ModelReasoningSummary::Auto));
        assert_eq!(model.text_verbosity, Some(ModelTextVerbosity::Low));
        assert_eq!(model.temperature, Some(0.2));
        assert_eq!(model.top_p, Some(0.8));
    }

    #[test]
    fn parses_model_specific_reasoning_efforts_including_max() {
        let _guard = lock_env();
        let path = write_temp_config(
            r#"
            [providers.openai]
            api_key = "config-key"

            [providers.openai.models."gpt-5.6-terra"]
            supports_reasoning = true
            reasoning_effort = "medium"
            reasoning_efforts = ["none", "low", "medium", "high", "max"]
            "#,
        );

        let config = AppConfig::load_from_path(&path).expect("config should load");
        let (_, provider) = config.active_provider();
        let model = &provider.models["gpt-5.6-terra"];

        assert_eq!(model.reasoning_effort, Some(ModelReasoningEffort::Medium));
        assert_eq!(
            model.reasoning_efforts,
            vec![
                ModelReasoningEffort::None,
                ModelReasoningEffort::Low,
                ModelReasoningEffort::Medium,
                ModelReasoningEffort::High,
                ModelReasoningEffort::Max,
            ]
        );
        assert_eq!(
            model.request_metadata().selectable_reasoning_efforts(),
            model.reasoning_efforts
        );
    }

    #[test]
    fn preserves_custom_reasoning_efforts() {
        let _guard = lock_env();
        let path = write_temp_config(
            r#"
            [providers.openai]
            api_key = "config-key"
            [providers.openai.models.model]
            supports_reasoning = true
            reasoning_effort = "provider-ultra"
            reasoning_efforts = ["low", "provider-ultra"]
        "#,
        );
        let config = AppConfig::load_from_path(&path).expect("config should load");
        let (_, provider) = config.active_provider();
        let effort = provider.models["model"].reasoning_effort.clone();
        assert_eq!(
            effort,
            Some(ModelReasoningEffort::Custom("provider-ultra".into()))
        );
        #[derive(Serialize)]
        struct SerializedEffort {
            effort: Option<ModelReasoningEffort>,
        }
        assert_eq!(
            toml::to_string(&SerializedEffort { effort }).unwrap(),
            "effort = \"provider-ultra\"\n"
        );
    }

    #[test]
    fn omitted_reasoning_efforts_include_custom_default_in_selectable_values() {
        let _guard = lock_env();
        let path = write_temp_config(
            r#"
            [providers.openai]
            api_key = "config-key"
            [providers.openai.models.model]
            supports_reasoning = true
            reasoning_effort = "provider-ultra"
        "#,
        );
        let config = AppConfig::load_from_path(&path).expect("config should load");
        let (_, provider) = config.active_provider();
        let model = &provider.models["model"];

        assert!(model.reasoning_efforts.is_empty());
        assert_eq!(
            model
                .request_metadata()
                .selectable_reasoning_efforts()
                .last(),
            Some(&ModelReasoningEffort::Custom("provider-ultra".into()))
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
    fn parses_subagent_model_overrides() {
        let _guard = lock_env();
        let path = write_temp_config(
            r#"
            [agents.explorer]
            model = "gpt-5.5-mini"

            [agents.fixer]
            model = "gpt-5.5-coder"

            [agents.oracle]
            model = "gpt-5.5-oracle"

            [agents.designer]
            model = "gpt-5.5-designer"

            [agents.librarian]
            model = "gpt-5.5-librarian"

            [agents.general]
            model = "gpt-5.5-general"

            [providers.openai]
            api_key = "config-key"

            [providers.openai.models."gpt-5.5"]
            name = "GPT-5.5"

            [providers.openai.models."gpt-5.5-mini"]
            name = "GPT-5.5 Mini"

            [providers.openai.models."gpt-5.5-coder"]
            name = "GPT-5.5 Coder"

            [providers.openai.models."gpt-5.5-oracle"]
            name = "GPT-5.5 Oracle"

            [providers.openai.models."gpt-5.5-designer"]
            name = "GPT-5.5 Designer"

            [providers.openai.models."gpt-5.5-librarian"]
            name = "GPT-5.5 Librarian"

            [providers.openai.models."gpt-5.5-general"]
            name = "GPT-5.5 General"
            "#,
        );

        let config = AppConfig::load_from_path(&path).expect("config should load");

        assert_eq!(config.agents.model_for("explorer"), Some("gpt-5.5-mini"));
        assert_eq!(config.agents.model_for("fixer"), Some("gpt-5.5-coder"));
        assert_eq!(config.agents.model_for("oracle"), Some("gpt-5.5-oracle"));
        assert_eq!(
            config.agents.model_for("designer"),
            Some("gpt-5.5-designer")
        );
        assert_eq!(
            config.agents.model_for("librarian"),
            Some("gpt-5.5-librarian")
        );
        assert_eq!(config.agents.model_for("general"), Some("gpt-5.5-general"));
    }

    #[test]
    fn rejects_unknown_subagent_name() {
        let _guard = lock_env();
        let path = write_temp_config(
            r#"
            [agents.reviewer]
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
    fn parses_completions_provider_protocol_default() {
        let _guard = lock_env();
        let path = write_temp_config(
            r#"
            active_provider = "compat"

            [providers.compat]
            base_url = "https://example.invalid/v1"
            api_key = "config-key"
            protocol = "completions"

            [providers.compat.models."compat-model"]
            name = "Compat Model"
            supports_reasoning = false
            "#,
        );

        let config = AppConfig::load_from_path(&path).expect("config should load");
        let (provider_name, provider) = config.active_provider();

        assert_eq!(provider_name, "compat");
        assert_eq!(provider.protocol, ApiProtocol::Completions);
        assert_eq!(
            provider.models["compat-model"].protocol,
            ApiProtocol::Completions
        );
    }

    #[test]
    fn parses_opencode_style_local_mcp_servers() {
        let _guard = lock_env();
        let path = write_temp_config(
            r#"
            [mcp.context7]
            type = "local"
            command = ["npx", "-y", "@upstash/context7-mcp"]
            environment = { CONTEXT7_API_KEY = "secret" }
            enabled = true
            timeout = 7000

            [providers.openai]
            api_key = "config-key"

            [providers.openai.models."gpt-5.5"]
            name = "GPT-5.5"
            "#,
        );

        let config = AppConfig::load_from_path(&path).expect("config should load");
        let server = config.mcp.get("context7").expect("mcp server should exist");

        assert!(server.enabled);
        assert_eq!(server.timeout_ms, 7000);
        let McpTransportConfig::Local(local) = &server.transport else {
            panic!("expected local MCP server");
        };
        assert_eq!(local.command, ["npx", "-y", "@upstash/context7-mcp"]);
        assert_eq!(local.environment["CONTEXT7_API_KEY"], "secret");
    }

    #[test]
    fn persists_only_one_mcp_enabled_field_and_reloads_it() {
        let _guard = lock_env();
        let path = write_temp_config(
            r#"# preserve this comment
[mcp.alpha]
type = "local"
command = ["alpha"] # preserve this too

[mcp.beta]
type = "local"
command = ["beta"]
enabled = false

[providers.openai]
api_key = "config-key"
[providers.openai.models.model]
"#,
        );

        persist_mcp_server_enabled(&path, "alpha", false).expect("persist MCP setting");
        let written = fs::read_to_string(&path).expect("read updated config");
        assert!(written.contains("# preserve this comment"));
        assert!(written.contains("command = [\"alpha\"] # preserve this too"));
        assert!(written.contains("enabled = false\n\n[mcp.beta]"));
        assert!(
            written.contains("[mcp.beta]\ntype = \"local\"\ncommand = [\"beta\"]\nenabled = false")
        );

        let reloaded = AppConfig::load_from_path(&path).expect("reload updated config");
        assert!(!reloaded.mcp["alpha"].enabled);
        assert!(!reloaded.mcp["beta"].enabled);
    }

    #[test]
    fn persists_enabled_state_in_top_level_inline_mcp_table() {
        let path = write_temp_config(
            "mcp = { alpha = { type = \"local\", command = [\"alpha\"] }, beta = { type = \"local\", command = [\"beta\"] } } # preserve this comment\n",
        );

        persist_mcp_server_enabled(&path, "alpha", false).expect("persist MCP setting");

        let written = fs::read_to_string(&path).expect("read updated config");
        assert!(written.starts_with("mcp = {"));
        assert!(written.contains("alpha = {"));
        assert!(written.contains("enabled = false"), "{written}");
        assert!(written.contains("beta = { type = \"local\", command = [\"beta\"] }"));
        assert!(written.contains("# preserve this comment"));
        assert!(!written.contains("[mcp]"));
    }

    #[test]
    fn persists_enabled_state_in_inline_server_under_mcp_table() {
        let path = write_temp_config(
            "[mcp]\nalpha = { type = \"local\", command = [\"alpha\"] }\nbeta = { type = \"local\", command = [\"beta\"] } # preserve this comment\n",
        );

        persist_mcp_server_enabled(&path, "alpha", false).expect("persist MCP setting");

        let written = fs::read_to_string(&path).expect("read updated config");
        assert!(written.contains("[mcp]"));
        assert!(written.contains("alpha = {"));
        assert!(written.contains("enabled = false"), "{written}");
        assert!(
            written.contains(
                "beta = { type = \"local\", command = [\"beta\"] } # preserve this comment"
            )
        );
    }

    #[test]
    fn persisting_mcp_enabled_state_returns_current_server_config() {
        let path = write_temp_config(
            "[mcp.alpha]\ntype = \"local\"\ncommand = [\"alpha\", \"--serve\"]\ntimeout = 7500\nenabled = true\n",
        );

        let server =
            persist_mcp_server_enabled(&path, "alpha", false).expect("persist MCP setting");

        assert!(!server.enabled);
        assert_eq!(server.timeout_ms, 7500);
        let McpTransportConfig::Local(local) = server.transport else {
            panic!("expected local MCP server");
        };
        assert_eq!(local.command, ["alpha", "--serve"]);
    }

    #[test]
    fn persisting_unknown_mcp_server_preserves_original_file() {
        let path = write_temp_config("[mcp.alpha]\ntype = \"local\"\ncommand = [\"alpha\"]\n");
        let original = fs::read_to_string(&path).expect("read config");
        let error = persist_mcp_server_enabled(&path, "missing", false)
            .expect_err("unknown server should fail");
        assert!(error.to_string().contains("missing"));
        assert_eq!(fs::read_to_string(&path).expect("read config"), original);
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

    #[cfg(unix)]
    #[test]
    fn temp_config_file_is_created_with_source_permissions() {
        use std::os::unix::fs::PermissionsExt;

        let path = write_temp_config("secret\n");
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600))
            .expect("restrict source config permissions");
        let temp_path = path.with_file_name("config-temp-permissions-test");
        let metadata = fs::metadata(&path).expect("stat source config");

        let temp = create_config_temp_file(&temp_path, &metadata).expect("create temp config");
        assert_eq!(
            temp.metadata().expect("stat temp config").mode() & 0o777,
            0o600
        );
        drop(temp);
        fs::remove_file(temp_path).expect("remove temp config");
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
    fn parses_opencode_style_remote_mcp_servers() {
        let _guard = lock_env();
        let path = write_temp_config(
            r#"
            [mcp.docs]
            type = "remote"
            url = "https://example.invalid/mcp"
            headers = { Authorization = "Bearer token" }
            oauth = false

            [providers.openai]
            api_key = "config-key"

            [providers.openai.models."gpt-5.5"]
            name = "GPT-5.5"
            "#,
        );

        let config = AppConfig::load_from_path(&path).expect("config should load");
        let server = config.mcp.get("docs").expect("mcp server should exist");
        let McpTransportConfig::Remote(remote) = &server.transport else {
            panic!("expected remote MCP server");
        };
        assert_eq!(remote.url, "https://example.invalid/mcp");
        assert_eq!(remote.headers["Authorization"], "Bearer token");
        assert!(!remote.oauth);
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
    fn model_protocol_overrides_provider_protocol() {
        let _guard = lock_env();
        let path = write_temp_config(
            r#"
            active_provider = "mixed"

            [providers.mixed]
            base_url = "https://example.invalid/v1"
            api_key = "config-key"
            protocol = "responses"

            [providers.mixed.models."responses-model"]
            name = "Responses Model"

            [providers.mixed.models."chat-model"]
            name = "Chat Model"
            protocol = "completions"
            supports_reasoning = false
            "#,
        );

        let config = AppConfig::load_from_path(&path).expect("config should load");
        let (_, provider) = config.active_provider();

        assert_eq!(provider.protocol, ApiProtocol::Responses);
        assert_eq!(
            provider.models["responses-model"].protocol,
            ApiProtocol::Responses
        );
        assert_eq!(
            provider.models["chat-model"].protocol,
            ApiProtocol::Completions
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
    fn missing_config_error_includes_example() {
        let _guard = lock_env();
        let path = env::temp_dir().join("letcode-missing-config-test.toml");
        let _ = fs::remove_file(&path);

        let error = AppConfig::load_from_path(&path).expect_err("load should fail");
        let message = error.to_string();
        assert!(message.contains("config file not found"));
        assert!(message.contains("[providers.openai]"));
        assert!(message.contains("# max_iterations = 64"));
        assert!(message.contains("# max_tool_calls = 128"));
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

    #[test]
    fn prompt_cache_omitted_defaults_to_disabled_without_retention_or_namespace() {
        let _guard = lock_env();
        let model = load_openai_prompt_cache_model("");

        assert_eq!(model.prompt_cache, PromptCacheConfig::default());
    }

    #[test]
    fn prompt_cache_accepts_enabled_responses_in_memory_with_explicit_namespace() {
        let _guard = lock_env();
        let model = load_openai_prompt_cache_model(
            r#"
            [providers.openai.models."gpt-test".prompt_cache]
            enabled = true
            retention = "in_memory"
            namespace = "team-alpha"
            "#,
        );

        assert_eq!(
            model.prompt_cache,
            PromptCacheConfig {
                enabled: true,
                retention: Some(PromptCacheRetention::InMemory),
                namespace: Some("team-alpha".to_string()),
            }
        );
        assert_eq!(model.request_metadata().prompt_cache, model.prompt_cache);
    }

    #[test]
    fn prompt_cache_accepts_enabled_responses_24h_retention() {
        let _guard = lock_env();
        let model = load_openai_prompt_cache_model(
            r#"
            [providers.openai.models."gpt-test".prompt_cache]
            enabled = true
            retention = "24h"
            "#,
        );

        assert_eq!(
            model.prompt_cache.retention,
            Some(PromptCacheRetention::TwentyFourHours)
        );
    }

    #[test]
    fn prompt_cache_enabled_without_namespace_defaults_to_normalized_provider_name() {
        let _guard = lock_env();
        let path = write_temp_config(
            r#"
            active_provider = "OpenAI-Primary"

            [providers."OpenAI-Primary"]
            base_url = "https://example.invalid/v1"
            api_key = "config-key"
            protocol = "responses"

            [providers."OpenAI-Primary".models."gpt-test"]

            [providers."OpenAI-Primary".models."gpt-test".prompt_cache]
            enabled = true
            "#,
        );

        let config = AppConfig::load_from_path(&path).expect("config should load");
        let model = &config.providers["OpenAI-Primary"].models["gpt-test"];
        assert_eq!(
            model.prompt_cache.namespace,
            Some("openai-primary".to_string())
        );
        assert_eq!(model.request_metadata().prompt_cache, model.prompt_cache);
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
    fn prompt_cache_omitted_or_disabled_leaves_compatible_provider_unaffected() {
        let _guard = lock_env();
        let path = write_temp_config(
            r#"
            [providers.compat]
            base_url = "https://example.invalid/v1"
            api_key = "config-key"
            protocol = "completions"

            [providers.compat.models.omitted]

            [providers.compat.models.disabled.prompt_cache]
            enabled = false
            "#,
        );

        let config = AppConfig::load_from_path(&path).expect("config should load");
        let models = &config.providers["compat"].models;
        assert_eq!(models["omitted"].prompt_cache, PromptCacheConfig::default());
        assert_eq!(
            models["disabled"].prompt_cache,
            PromptCacheConfig::default()
        );
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
