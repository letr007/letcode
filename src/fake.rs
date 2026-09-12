//! Outbound request disguises for compatible coding-agent clients.
//!
//! A fake covers two surfaces: the transport metadata around letcode's existing
//! prompt and tools, and the environment block letcode reports about itself. It
//! deliberately does not replace the agent persona or the tool catalog.
//!
//! Declared values live in `[fake]` in `letcode.toml`. Absent values are derived
//! rather than hardcoded wherever the real host can answer: OS, architecture,
//! terminal, working directory, shell and repository state come from the machine
//! running letcode, because a real client reports its real host too. Attributes
//! letcode has no equivalent for fall back to Codex-typical values.

use crate::config::FakeConfig;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::path::Path;
use std::process::Command;

/// Disguise mode selected by the user.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FakeClient {
    /// Select the profile that matches the active provider protocol.
    Auto,
    /// Apply the Codex Responses wire profile.
    Codex,
    /// Apply the Anthropic Messages transport profile.
    Anthropic,
}

impl FakeClient {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Codex => "codex",
            Self::Anthropic => "anthropic",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "auto" => Some(Self::Auto),
            "codex" => Some(Self::Codex),
            "anthropic" => Some(Self::Anthropic),
            _ => None,
        }
    }

    pub const fn supports_protocol(self, protocol: crate::config::ApiProtocol) -> bool {
        match self {
            Self::Auto => matches!(
                protocol,
                crate::config::ApiProtocol::Responses | crate::config::ApiProtocol::Anthropic
            ),
            Self::Codex => matches!(protocol, crate::config::ApiProtocol::Responses),
            Self::Anthropic => matches!(protocol, crate::config::ApiProtocol::Anthropic),
        }
    }

    pub(crate) fn supports_protocol_id(self, protocol: &crate::model_runtime::ProtocolId) -> bool {
        match self {
            Self::Auto => matches!(protocol.as_str(), "responses" | "anthropic"),
            Self::Codex => protocol.as_str() == "responses",
            Self::Anthropic => protocol.as_str() == "anthropic",
        }
    }
}

/// Stable synthetic identity used for one fake-enabled agent session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodexIdentity {
    pub installation_id: String,
    pub session_id: String,
}

impl CodexIdentity {
    pub fn new(installation_id: impl Into<String>) -> Self {
        Self {
            installation_id: installation_id.into(),
            session_id: synthetic_uuid(),
        }
    }

    /// Resolves the per-turn context. Identity stays frozen for the session;
    /// declared and host-derived values are re-resolved on every turn so a
    /// configuration reload takes effect without rebuilding the session.
    pub(crate) fn turn_context(
        &self,
        config: &FakeConfig,
        cwd: Option<&Path>,
    ) -> CodexRequestContext {
        CodexRequestContext::resolve(self, config, cwd)
    }
}

/// Per-turn values injected into a Codex-shaped request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodexRequestContext {
    // Session identity, frozen when the fake was enabled.
    pub installation_id: String,
    pub session_id: String,
    pub thread_id: String,
    // Per-turn identity.
    pub turn_id: String,
    pub root_turn_id: String,
    pub started_at_unix_ms: u128,
    // Client profile.
    pub version: String,
    pub originator: String,
    pub os: String,
    pub arch: String,
    pub terminal: String,
    pub beta_features: Vec<String>,
    // Clock.
    pub timezone: String,
    /// Absent when the declared zone cannot be read from the real clock, so a
    /// date belonging to another zone is never reported as this one.
    pub current_date: Option<String>,
    // Agent identity reported in turn metadata.
    pub agent_name: Option<String>,
    // Environment.
    pub sandbox: String,
    pub sandbox_mode: String,
    pub auto_review_enabled: bool,
    pub node_repl_auto_review_required: bool,
    pub node_repl_disabled: bool,
    pub cwd: String,
    pub workspace: String,
    pub shell: String,
    pub git_commit_hash: Option<String>,
    pub git_remote_url: Option<String>,
    pub git_has_changes: Option<bool>,
    pub extra: indexmap::IndexMap<String, String>,
}

/// Marks the runtime-context message letcode emits while the Codex fake is on.
/// The decorator keys off it to select exactly that input item.
pub const ENVIRONMENT_CONTEXT_MARKER: &str = "<environment_context>";

/// Codex-typical values for attributes letcode has no equivalent for.
mod defaults {
    pub(super) const VERSION: &str = "0.149.1";
    pub(super) const ORIGINATOR: &str = "codex_exec";
    pub(super) const TERMINAL: &str = "ghostty/1.3.1";
    pub(super) const BETA_FEATURES: &[&str] = &["remote_compaction_v2"];
    pub(super) const SANDBOX: &str = "none";
    pub(super) const SANDBOX_MODE: &str = "danger-full-access";
    /// Mirrors the upstream fallback when the host zone cannot be named.
    pub(super) const TIMEZONE_FALLBACK: &str = "Etc/UTC";
    pub(super) const SHELL_FALLBACK: &str = "zsh";
}

impl CodexRequestContext {
    fn resolve(identity: &CodexIdentity, config: &FakeConfig, cwd: Option<&Path>) -> Self {
        let cwd = cwd
            .map(Path::to_path_buf)
            .or_else(|| std::env::current_dir().ok())
            .unwrap_or_else(|| std::path::PathBuf::from("/"));
        let cwd_text = cwd.to_string_lossy().to_string();
        let workspace = config
            .environment
            .workspace
            .clone()
            .unwrap_or_else(|| cwd_text.clone());
        let timezone = config.clock.timezone.clone().unwrap_or_else(host_timezone);
        let current_date = match config.clock.date.clone() {
            Some(date) => Some(date),
            // Derived, never hardcoded: the real clock read in the declared zone.
            None => date_in_timezone(&timezone),
        };
        // Only the git fields the configuration leaves open are read from the
        // host, so a fully declared `[fake.environment]` spawns no subprocess.
        let git = if config.environment.git_commit_hash.is_none()
            || config.environment.git_remote_url.is_none()
            || config.environment.git_has_changes.is_none()
        {
            probe_git(&cwd)
        } else {
            GitFacts::default()
        };

        Self {
            installation_id: identity.installation_id.clone(),
            session_id: identity.session_id.clone(),
            thread_id: identity.session_id.clone(),
            turn_id: synthetic_uuid(),
            root_turn_id: synthetic_uuid(),
            started_at_unix_ms: unix_timestamp_ms(),
            version: config
                .client
                .version
                .clone()
                .unwrap_or_else(|| defaults::VERSION.to_string()),
            originator: config
                .client
                .originator
                .clone()
                .unwrap_or_else(|| defaults::ORIGINATOR.to_string()),
            os: config.client.os.clone().unwrap_or_else(host_os),
            arch: config.client.arch.clone().unwrap_or_else(host_arch),
            terminal: config
                .client
                .terminal
                .clone()
                .or_else(host_terminal)
                .unwrap_or_else(|| defaults::TERMINAL.to_string()),
            beta_features: config.client.beta_features.clone().unwrap_or_else(|| {
                defaults::BETA_FEATURES
                    .iter()
                    .map(|value| (*value).to_string())
                    .collect()
            }),
            timezone,
            current_date,
            agent_name: config.identity.agent_name.clone(),
            sandbox: config
                .environment
                .sandbox
                .clone()
                .unwrap_or_else(|| defaults::SANDBOX.to_string()),
            sandbox_mode: config
                .environment
                .sandbox_mode
                .clone()
                .unwrap_or_else(|| defaults::SANDBOX_MODE.to_string()),
            auto_review_enabled: config.environment.auto_review_enabled.unwrap_or(false),
            node_repl_auto_review_required: config
                .environment
                .node_repl_auto_review_required
                .unwrap_or(false),
            node_repl_disabled: config.environment.node_repl_disabled.unwrap_or(false),
            cwd: config.environment.cwd.clone().unwrap_or(cwd_text),
            workspace,
            shell: config
                .environment
                .shell
                .clone()
                .or_else(host_shell)
                .unwrap_or_else(|| defaults::SHELL_FALLBACK.to_string()),
            git_commit_hash: config
                .environment
                .git_commit_hash
                .clone()
                .or(git.commit_hash),
            git_remote_url: config.environment.git_remote_url.clone().or(git.remote_url),
            git_has_changes: config.environment.git_has_changes.or(git.has_changes),
            extra: config.extra.clone(),
        }
    }

    pub fn window_id(&self) -> String {
        format!("{}:0", self.session_id)
    }

    /// HTTP headers added on top of the provider client's authentication and
    /// JSON/SSE headers.
    pub fn headers(&self) -> Vec<(String, String)> {
        vec![
            ("accept".into(), "text/event-stream".into()),
            ("originator".into(), self.originator.clone()),
            (
                "user-agent".into(),
                format!(
                    "{}/{} ({}; {}) {}",
                    self.originator, self.version, self.os, self.arch, self.terminal
                ),
            ),
            ("session-id".into(), self.session_id.clone()),
            ("thread-id".into(), self.thread_id.clone()),
            ("x-client-request-id".into(), self.session_id.clone()),
            ("x-codex-window-id".into(), self.window_id()),
            (
                "x-openai-internal-codex-responses-lite".into(),
                "true".into(),
            ),
            ("x-codex-beta-features".into(), self.beta_features.join(",")),
            (
                "x-codex-turn-metadata".into(),
                self.turn_metadata_json().to_string(),
            ),
        ]
    }

    pub fn turn_metadata_json(&self) -> Value {
        let mut metadata = Map::new();
        metadata.insert(
            "installation_id".into(),
            Value::String(self.installation_id.clone()),
        );
        metadata.insert("session_id".into(), Value::String(self.session_id.clone()));
        metadata.insert("thread_id".into(), Value::String(self.thread_id.clone()));
        metadata.insert("turn_id".into(), Value::String(self.turn_id.clone()));
        metadata.insert("window_id".into(), Value::String(self.window_id()));
        metadata.insert("request_kind".into(), Value::String("turn".into()));
        // The imitated client only names an agent that reserved a nickname, so a
        // plain top-level turn carries no `agent_name` at all.
        if let Some(agent_name) = &self.agent_name {
            metadata.insert("agent_name".into(), Value::String(agent_name.clone()));
        }
        metadata.insert(
            "root_turn_id".into(),
            Value::String(self.root_turn_id.clone()),
        );
        metadata.insert("thread_source".into(), Value::String("user".into()));
        metadata.insert("sandbox".into(), Value::String(self.sandbox.clone()));
        metadata.insert(
            "sandbox_mode".into(),
            Value::String(self.sandbox_mode.clone()),
        );
        metadata.insert(
            "auto_review_enabled".into(),
            Value::Bool(self.auto_review_enabled),
        );
        metadata.insert(
            "node_repl_auto_review_required".into(),
            Value::Bool(self.node_repl_auto_review_required),
        );
        metadata.insert(
            "node_repl_disabled".into(),
            Value::Bool(self.node_repl_disabled),
        );

        // Workspace entries mirror the upstream optional shape: absent values are
        // omitted rather than fabricated.
        let mut workspace = Map::new();
        if let Some(url) = &self.git_remote_url {
            workspace.insert(
                "associated_remote_urls".into(),
                serde_json::json!({ "origin": url }),
            );
        }
        if let Some(hash) = &self.git_commit_hash {
            workspace.insert("latest_git_commit_hash".into(), Value::String(hash.clone()));
        }
        if let Some(has_changes) = self.git_has_changes {
            workspace.insert("has_changes".into(), Value::Bool(has_changes));
        }
        metadata.insert(
            "workspaces".into(),
            serde_json::json!({ self.workspace.clone(): Value::Object(workspace) }),
        );

        for (key, value) in &self.extra {
            metadata.insert(key.clone(), Value::String(value.clone()));
        }
        metadata.insert(
            "turn_started_at_unix_ms".into(),
            Value::Number(serde_json::Number::from(self.started_at_unix_ms as u64)),
        );
        Value::Object(metadata)
    }

    /// The environment block letcode reports about itself. Rendered as the
    /// single-message `<environment_context>` shape a real Codex client sends.
    pub fn environment_context_text(&self) -> String {
        let mut body = String::from(ENVIRONMENT_CONTEXT_MARKER);
        body.push('\n');
        push_element(&mut body, "cwd", &self.cwd);
        push_element(&mut body, "shell", &self.shell);
        if let Some(current_date) = &self.current_date {
            push_element(&mut body, "current_date", current_date);
        }
        push_element(&mut body, "timezone", &self.timezone);
        body.push_str("</environment_context>");
        body
    }

    /// HTTP headers applied to Anthropic Messages requests when the fake is
    /// active. The Messages body keeps its native shape; only transport
    /// metadata is disguised.
    pub fn anthropic_headers(&self) -> Vec<(String, String)> {
        self.headers()
            .into_iter()
            .filter(|(name, _)| name != "accept")
            .collect()
    }

    pub fn client_metadata(&self) -> Value {
        serde_json::json!({
            "thread_id": self.thread_id,
            "x-codex-turn-metadata": self.turn_metadata_json().to_string(),
            "session_id": self.session_id,
            "x-codex-installation-id": self.installation_id,
            "turn_id": self.turn_id,
            "x-codex-window-id": self.window_id(),
            "root_turn_id": self.root_turn_id
        })
    }
}

/// Rewrite a serialized OpenAI Responses request into the observed Codex
/// wire-shape. Prompt content and tools are intentionally preserved, except for
/// the runtime-context message, which is re-roled to the `user` message a Codex
/// client would send. The message count never changes: the origin accounting in
/// `prepared_request_origin_mismatch` stays valid.
pub fn apply_codex_response_shape(request: &mut Value, context: &CodexRequestContext) {
    let Some(object) = request.as_object_mut() else {
        return;
    };

    let preserved = object
        .iter()
        .filter(|(key, _)| {
            matches!(
                key.as_str(),
                "model" | "instructions" | "input" | "tools" | "max_output_tokens" | "reasoning"
            )
        })
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect::<Map<String, Value>>();

    *object = preserved;
    object.insert("tool_choice".into(), Value::String("auto".into()));
    object.insert("parallel_tool_calls".into(), Value::Bool(false));

    let mut reasoning = object
        .get("reasoning")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    if !reasoning.contains_key("effort") {
        reasoning.insert("effort".into(), Value::String("low".into()));
    }
    reasoning.insert("context".into(), Value::String("all_turns".into()));
    object.insert("reasoning".into(), Value::Object(reasoning));

    object.insert("store".into(), Value::Bool(false));
    object.insert("stream".into(), Value::Bool(true));
    object.insert(
        "include".into(),
        Value::Array(vec![Value::String("reasoning.encrypted_content".into())]),
    );
    object.insert(
        "prompt_cache_key".into(),
        Value::String(context.session_id.clone()),
    );
    object.insert("text".into(), serde_json::json!({ "verbosity": "low" }));
    object.insert("client_metadata".into(), context.client_metadata());

    apply_environment_context_role(request, context);
}

/// Flips the runtime-context input item to the `user` role Codex uses.
///
/// The item is selected by the marker letcode itself emits for this purpose, so
/// no unrelated developer message can be matched. The message count never
/// changes, which keeps the prepared-request origin accounting valid.
fn apply_environment_context_role(request: &mut Value, _context: &CodexRequestContext) {
    let Some(items) = request.get_mut("input").and_then(Value::as_array_mut) else {
        return;
    };
    for item in items {
        if item.get("role").and_then(Value::as_str) != Some("developer") {
            continue;
        }
        let carries_marker = item
            .get("content")
            .and_then(Value::as_array)
            .is_some_and(|content| {
                content.iter().any(|block| {
                    block
                        .get("text")
                        .and_then(Value::as_str)
                        .is_some_and(|text| text.starts_with(ENVIRONMENT_CONTEXT_MARKER))
                })
            });
        if carries_marker && let Some(object) = item.as_object_mut() {
            object.insert("role".into(), Value::String("user".into()));
        }
    }
}

fn push_element(rendered: &mut String, name: &str, value: &str) {
    rendered.push_str("  <");
    rendered.push_str(name);
    rendered.push('>');
    push_xml_escaped_text(rendered, value);
    rendered.push_str("</");
    rendered.push_str(name);
    rendered.push_str(">\n");
}

fn push_xml_escaped_text(rendered: &mut String, value: &str) {
    for character in value.chars() {
        match character {
            '&' => rendered.push_str("&amp;"),
            '<' => rendered.push_str("&lt;"),
            '>' => rendered.push_str("&gt;"),
            '"' => rendered.push_str("&quot;"),
            '\'' => rendered.push_str("&apos;"),
            _ => rendered.push(character),
        }
    }
}

/// Beta value the WebSocket transport uses to negotiate its protocol version.
/// The HTTP transport does not send it.
pub const CODEX_RESPONSES_WEBSOCKET_BETA: &str = "responses_websockets=2026-02-06";

#[derive(Default)]
struct GitFacts {
    commit_hash: Option<String>,
    remote_url: Option<String>,
    has_changes: Option<bool>,
}

/// Reads the real repository state. A missing or unavailable repository yields
/// absent fields, which the metadata shape omits rather than invents.
fn probe_git(cwd: &Path) -> GitFacts {
    let status = command_output(
        "git",
        &[
            "-C",
            &cwd.to_string_lossy(),
            "status",
            "--porcelain=v2",
            "--branch",
        ],
    );
    let (commit_hash, has_changes) = status.as_deref().map_or((None, None), |status| {
        let mut commit = None;
        let mut dirty = false;
        for line in status.lines() {
            if let Some(value) = line.strip_prefix("# branch.oid ") {
                let value = value.trim();
                if !value.is_empty() && value != "(initial)" {
                    commit = Some(value.to_string());
                }
            } else if !line.starts_with('#') && !line.trim().is_empty() {
                dirty = true;
            }
        }
        (commit, Some(dirty))
    });
    let remote_url = command_output(
        "git",
        &["-C", &cwd.to_string_lossy(), "remote", "get-url", "origin"],
    )
    .filter(|value| !value.trim().is_empty());
    GitFacts {
        commit_hash,
        remote_url,
        has_changes,
    }
}

/// Names the host time zone the way a real client reports it. Mirrors the
/// upstream fallback when the zone cannot be named from the filesystem.
fn host_timezone() -> String {
    if let Ok(value) = std::env::var("TZ") {
        let value = value.trim();
        if !value.is_empty() {
            return value.to_string();
        }
    }
    if let Ok(link) = std::fs::read_link("/etc/localtime")
        && let Some(index) = link
            .to_string_lossy()
            .rfind("zoneinfo/")
            .map(|index| index + "zoneinfo/".len())
        && let Some(name) = link.to_string_lossy().get(index..)
        && !name.trim().is_empty()
    {
        return name.trim().to_string();
    }
    defaults::TIMEZONE_FALLBACK.to_string()
}

/// Derives the local date from the real clock in the declared zone, so the
/// reported date stays consistent with the reported zone instead of being a
/// frozen literal.
/// The failure is logged and the date omitted rather than filled in from
/// another zone: a date that does not belong to the declared zone is worse than
/// an absent one.
fn date_in_timezone(timezone: &str) -> Option<String> {
    let date = command_output_with_timezone(timezone, "+%Y-%m-%d");
    if date.is_none() {
        tracing::warn!(
            timezone,
            "could not read the date in the reported time zone; omitting current_date"
        );
    }
    date
}

/// Names the host terminal the way a real client reports it.
fn host_terminal() -> Option<String> {
    let program = std::env::var("TERM_PROGRAM").ok()?;
    let program = program.trim();
    if program.is_empty() {
        return None;
    }
    // The raw `TERM_PROGRAM` value is the token, verbatim: terminals that
    // report themselves in mixed case (`Apple_Terminal`) keep it.
    match std::env::var("TERM_PROGRAM_VERSION") {
        Ok(version) if !version.trim().is_empty() => Some(format!("{program}/{}", version.trim())),
        _ => Some(program.to_string()),
    }
}

/// Names the host login shell the way a real client reports it.
fn host_shell() -> Option<String> {
    let shell = std::env::var("SHELL").ok()?;
    let name = Path::new(shell.trim()).file_name()?;
    let name = name.to_string_lossy();
    (!name.trim().is_empty()).then(|| name.trim().to_string())
}

fn command_output_with_timezone(timezone: &str, format: &str) -> Option<String> {
    Command::new("date")
        .env("TZ", timezone)
        .arg(format)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|text| text.trim().to_string())
        .filter(|text| !text.is_empty())
}

fn command_output(command: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(command).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8(output.stdout).ok()?;
    let trimmed = text.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_owned())
}

fn host_os() -> String {
    let name = match std::env::consts::OS {
        "macos" => "Mac OS",
        "windows" => "Windows",
        "linux" => "Linux",
        other => other,
    };
    match os_version() {
        Some(version) => format!("{name} {version}"),
        None => name.to_string(),
    }
}

#[cfg(target_os = "macos")]
fn os_version() -> Option<String> {
    command_output("sw_vers", &["-productVersion"])
}

#[cfg(target_os = "linux")]
fn os_version() -> Option<String> {
    let release = std::fs::read_to_string("/proc/sys/kernel/osrelease").ok()?;
    let release = release.trim();
    (!release.is_empty()).then(|| release.to_string())
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn os_version() -> Option<String> {
    None
}

fn host_arch() -> String {
    std::env::consts::ARCH.to_string()
}

fn unix_timestamp_ms() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

/// A 64-bit word from the OS-seeded hasher `RandomState` keeps. The keys differ
/// per call and `salt` separates the callers, so the bits are spread over the
/// whole range instead of collapsing into the visible pattern a counter or a
/// process id produces.
fn random_word(salt: u64) -> u64 {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_CALL: AtomicU64 = AtomicU64::new(0);
    let mut hasher = RandomState::new().build_hasher();
    hasher.write_u64(salt);
    hasher.write_u64(NEXT_CALL.fetch_add(1, Ordering::Relaxed));
    hasher.finish()
}

fn format_uuid(hi: u64, lo: u64) -> String {
    format!(
        "{:08x}-{:04x}-{:04x}-{:04x}-{:012x}",
        (hi >> 32) as u32,
        (hi >> 16) as u16,
        hi as u16,
        (lo >> 48) as u16,
        lo & 0xffff_ffff_ffff
    )
}

/// Session, thread and turn ids are version 7 in the client this profile
/// imitates: 48 bits of the current millisecond, then random bits. Keeping the
/// timestamp prefix matches that shape without pinning the high bytes.
fn synthetic_uuid() -> String {
    let millis = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_millis() as u64)
        .unwrap_or(0);
    let hi = ((millis & 0x0000_ffff_ffff_ffff) << 16) | 0x7000 | (random_word(0) & 0x0fff);
    let lo = 0x8000_0000_0000_0000 | (random_word(1) & 0x3fff_ffff_ffff_ffff);
    format_uuid(hi, lo)
}

/// Installation ids are version 4 in the client this profile imitates, so they
/// carry no timestamp.
pub(crate) fn synthetic_installation_id() -> String {
    let hi = (random_word(2) & 0xffff_ffff_ffff_0fff) | 0x4000;
    let lo = 0x8000_0000_0000_0000 | (random_word(3) & 0x3fff_ffff_ffff_ffff);
    format_uuid(hi, lo)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn context() -> CodexRequestContext {
        CodexIdentity::new("installation").turn_context(&FakeConfig::default(), None)
    }

    #[test]
    fn fake_client_parse_supports_all_modes() {
        assert_eq!(FakeClient::parse("auto"), Some(FakeClient::Auto));
        assert_eq!(FakeClient::parse("codex"), Some(FakeClient::Codex));
        assert_eq!(FakeClient::parse("anthropic"), Some(FakeClient::Anthropic));
        assert_eq!(FakeClient::parse("other"), None);
    }

    #[test]
    fn fake_modes_are_protocol_scoped() {
        use crate::config::ApiProtocol;

        assert!(FakeClient::Auto.supports_protocol(ApiProtocol::Responses));
        assert!(FakeClient::Auto.supports_protocol(ApiProtocol::Anthropic));
        assert!(!FakeClient::Auto.supports_protocol(ApiProtocol::Completions));
        assert!(FakeClient::Codex.supports_protocol(ApiProtocol::Responses));
        assert!(!FakeClient::Codex.supports_protocol(ApiProtocol::Anthropic));
        assert!(FakeClient::Anthropic.supports_protocol(ApiProtocol::Anthropic));
        assert!(!FakeClient::Anthropic.supports_protocol(ApiProtocol::Responses));

        let responses = crate::model_runtime::ProtocolId::new("responses").unwrap();
        let anthropic = crate::model_runtime::ProtocolId::new("anthropic").unwrap();
        let completions = crate::model_runtime::ProtocolId::new("completions").unwrap();
        assert!(FakeClient::Auto.supports_protocol_id(&responses));
        assert!(FakeClient::Auto.supports_protocol_id(&anthropic));
        assert!(!FakeClient::Auto.supports_protocol_id(&completions));
    }

    #[test]
    fn synthetic_ids_match_the_shape_the_imitated_client_sends() {
        let session = synthetic_uuid();
        assert_eq!(session.as_bytes()[14], b'7', "session ids are version 7");
        let variant = session.as_bytes()[19] as char;
        assert!(
            ('8'..='b').contains(&variant),
            "variant bits are 10, got {variant}"
        );
        // The prefix is the current millisecond, so it must not read as zeros.
        assert!(!session.starts_with("0000"), "{session}");
        assert_ne!(session, synthetic_uuid());

        let installation = synthetic_installation_id();
        assert_eq!(
            installation.as_bytes()[14],
            b'4',
            "installation ids are version 4"
        );
        let variant = installation.as_bytes()[19] as char;
        assert!(('8'..='b').contains(&variant), "got {variant}");
        // Random, not a fixed value: two calls must differ.
        assert_ne!(installation, synthetic_installation_id());
    }

    #[test]
    fn top_level_turns_carry_no_agent_name_unless_declared() {
        let identity = CodexIdentity::new(synthetic_installation_id());
        let derived = identity.turn_context(&crate::config::FakeConfig::default(), None);
        assert!(
            !derived
                .turn_metadata_json()
                .to_string()
                .contains("agent_name")
        );

        let mut declared = crate::config::FakeConfig::default();
        declared.identity.agent_name = Some("Hypatia".into());
        let named = identity.turn_context(&declared, None);
        assert_eq!(named.turn_metadata_json()["agent_name"], "Hypatia");
    }

    #[test]
    fn codex_identity_uses_stable_session_ids_within_a_context() {
        let identity = CodexIdentity::new("installation");
        let context = identity.turn_context(&FakeConfig::default(), None);

        assert_eq!(context.session_id, context.thread_id);
        assert_eq!(context.window_id(), format!("{}:0", context.session_id));
        assert_eq!(context.installation_id, "installation");
    }

    #[test]
    fn declared_values_win_over_derived_defaults() {
        let mut config = FakeConfig::default();
        config.client.version = Some("9.9.9".into());
        config.client.originator = Some("codex_cli_rs".into());
        config.client.os = Some("Plan9".into());
        config.client.arch = Some("vax".into());
        config.client.terminal = Some("vt100/1".into());
        config.client.beta_features = Some(vec!["one".into(), "two".into()]);
        config.clock.timezone = Some("Asia/Shanghai".into());
        config.clock.date = Some("1999-01-01".into());
        config.identity.agent_name = Some("unit-test".into());
        config.environment.sandbox = Some("seatbelt".into());
        config.environment.sandbox_mode = Some("read-only".into());
        config.environment.auto_review_enabled = Some(true);
        config.environment.node_repl_disabled = Some(true);
        config.environment.cwd = Some("/declared/cwd".into());
        config.environment.workspace = Some("/declared/ws".into());
        config.environment.shell = Some("fish".into());
        config.environment.git_commit_hash = Some("deadbeef".into());
        config.environment.git_remote_url = Some("https://example.test/repo.git".into());
        config.environment.git_has_changes = Some(true);
        config.extra.insert("custom_key".into(), "custom".into());

        let context = CodexIdentity::new("installation").turn_context(&config, None);
        let headers = context.headers();
        let user_agent = headers
            .iter()
            .find(|(name, _)| name == "user-agent")
            .map(|(_, value)| value.as_str())
            .expect("user agent header");
        assert_eq!(user_agent, "codex_cli_rs/9.9.9 (Plan9; vax) vt100/1");
        assert_eq!(context.current_date.as_deref(), Some("1999-01-01"));
        assert_eq!(context.timezone, "Asia/Shanghai");

        let metadata = context.turn_metadata_json();
        assert_eq!(metadata["agent_name"], "unit-test");
        assert_eq!(metadata["sandbox"], "seatbelt");
        assert_eq!(metadata["sandbox_mode"], "read-only");
        assert_eq!(metadata["auto_review_enabled"], true);
        assert_eq!(metadata["node_repl_disabled"], true);
        assert_eq!(metadata["custom_key"], "custom");
        let workspace = &metadata["workspaces"]["/declared/ws"];
        assert_eq!(workspace["latest_git_commit_hash"], "deadbeef");
        assert_eq!(workspace["has_changes"], true);
        assert_eq!(
            workspace["associated_remote_urls"]["origin"],
            "https://example.test/repo.git"
        );

        let block = context.environment_context_text();
        assert!(block.starts_with("<environment_context>\n"));
        assert!(block.contains("  <cwd>/declared/cwd</cwd>\n"));
        assert!(block.contains("  <shell>fish</shell>\n"));
        assert!(block.contains("  <current_date>1999-01-01</current_date>\n"));
        assert!(block.contains("  <timezone>Asia/Shanghai</timezone>\n"));
        assert!(block.ends_with("</environment_context>"));
    }

    #[test]
    fn environment_context_escapes_xml_text() {
        let mut config = FakeConfig::default();
        config.environment.cwd = Some("/a<b>&\"c\"".into());
        let context = CodexIdentity::new("installation").turn_context(&config, None);
        let block = context.environment_context_text();

        assert!(block.contains("/a&lt;b&gt;&amp;&quot;c&quot;"));
        assert!(!block.contains("/a<b>"));
    }

    #[test]
    fn response_shape_preserves_prompt_and_tools() {
        let mut request = serde_json::json!({
            "model": "gpt-5.6-sol",
            "instructions": "core and workspace instructions",
            "input": [{"type": "message"}],
            "tools": [{"name": "fs__read"}],
            "reasoning": {"effort": "minimal", "summary": "concise"},
            "temperature": 0.2,
            "service_tier": "priority"
        });
        let context = context();
        apply_codex_response_shape(&mut request, &context);

        assert_eq!(request["model"], "gpt-5.6-sol");
        assert_eq!(request["instructions"], "core and workspace instructions");
        assert_eq!(request["parallel_tool_calls"], false);
        assert_eq!(request["tool_choice"], "auto");
        assert_eq!(request["store"], false);
        assert_eq!(request["reasoning"]["effort"], "minimal");
        assert_eq!(request["reasoning"]["summary"], "concise");
        assert_eq!(request["reasoning"]["context"], "all_turns");
        assert_eq!(request["prompt_cache_key"], context.session_id);
        assert!(request.get("temperature").is_none());
        assert!(request.get("service_tier").is_none());
        assert!(
            request["client_metadata"]["x-codex-installation-id"]
                .as_str()
                .is_some_and(|value| value == "installation")
        );
    }

    #[test]
    fn runtime_context_item_is_reroled_without_changing_the_message_count() {
        let runtime_text = context().environment_context_text();
        let mut request = serde_json::json!({
            "model": "gpt",
            "instructions": "system",
            "input": [
                {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "skill prelude"}]},
                {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": runtime_text}]},
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}
            ],
            "tools": []
        });
        let before = request["input"].as_array().unwrap().len();
        apply_codex_response_shape(&mut request, &context());

        let items = request["input"].as_array().unwrap();
        assert_eq!(items.len(), before, "message count must not change");
        assert_eq!(
            items[0]["role"], "developer",
            "other developer messages stay"
        );
        assert_eq!(
            items[1]["role"], "user",
            "runtime context becomes a user block"
        );
        assert_eq!(items[2]["role"], "user");
    }

    #[test]
    fn unrelated_developer_messages_keep_their_role() {
        let mut request = serde_json::json!({
            "model": "gpt",
            "input": [
                {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "运行时上下文：\n- 当前日期：2026-09-11"}]}
            ]
        });
        apply_codex_response_shape(&mut request, &context());

        assert_eq!(request["input"][0]["role"], "developer");
    }

    #[test]
    fn headers_expose_no_real_local_paths_unless_declared() {
        let mut config = FakeConfig::default();
        config.environment.workspace = Some("/workspace".into());
        config.environment.cwd = Some("/workspace".into());
        let context = CodexIdentity::new("installation").turn_context(&config, None);
        let headers = context.headers();
        let metadata = headers
            .iter()
            .find(|(name, _)| name == "x-codex-turn-metadata")
            .map(|(_, value)| value)
            .expect("turn metadata header");

        assert!(metadata.contains("/workspace"));
        assert!(!metadata.contains(r"/Users/"));
    }

    #[test]
    fn anthropic_headers_keep_identity_and_drop_duplicate_accept() {
        let context = context();
        let headers = context.anthropic_headers();

        assert!(
            headers.iter().all(|(name, _)| name != "accept"),
            "anthropic transport already sends its own Accept header"
        );
        assert!(
            headers
                .iter()
                .any(|(name, value)| name == "originator" && !value.is_empty())
        );
        assert!(
            headers
                .iter()
                .any(|(name, _)| name == "x-codex-turn-metadata")
        );
    }
}
