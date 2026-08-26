//! Outbound request disguises for compatible coding-agent clients.
//!
//! A fake is a wire-shape profile only: it changes transport metadata around
//! letcode's existing prompt and tools. It deliberately does not replace the
//! agent persona or tool catalog.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

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

    pub(crate) fn turn_context(&self) -> CodexRequestContext {
        CodexRequestContext {
            installation_id: self.installation_id.clone(),
            session_id: self.session_id.clone(),
            thread_id: self.session_id.clone(),
            turn_id: synthetic_uuid(),
            root_turn_id: synthetic_uuid(),
            started_at_unix_ms: unix_timestamp_ms(),
        }
    }
}

/// Per-turn synthetic values injected into a Codex-shaped request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodexRequestContext {
    pub installation_id: String,
    pub session_id: String,
    pub thread_id: String,
    pub turn_id: String,
    pub root_turn_id: String,
    pub started_at_unix_ms: u128,
}

impl CodexRequestContext {
    pub fn window_id(&self) -> String {
        format!("{}:0", self.session_id)
    }

    /// HTTP headers added on top of the provider client's authentication and
    /// JSON/SSE headers.
    pub fn headers(&self) -> Vec<(String, String)> {
        vec![
            ("accept".into(), "text/event-stream".into()),
            ("originator".into(), "codex_exec".into()),
            (
                "user-agent".into(),
                "codex_exec/0.149.1 (Mac OS 26.4.1; arm64) ghostty/1.3.1 (codex_exec; 0.149.1)"
                    .into(),
            ),
            ("session-id".into(), self.session_id.clone()),
            ("thread-id".into(), self.thread_id.clone()),
            ("x-client-request-id".into(), self.session_id.clone()),
            ("x-codex-window-id".into(), self.window_id()),
            (
                "x-openai-internal-codex-responses-lite".into(),
                "true".into(),
            ),
            (
                "x-codex-beta-features".into(),
                "remote_compaction_v2".into(),
            ),
            (
                "x-codex-turn-metadata".into(),
                self.turn_metadata_json().to_string(),
            ),
        ]
    }

    pub fn turn_metadata_json(&self) -> Value {
        serde_json::json!({
            "installation_id": self.installation_id,
            "session_id": self.session_id,
            "thread_id": self.thread_id,
            "agent_name": "/root",
            "turn_id": self.turn_id,
            "window_id": self.window_id(),
            "request_kind": "turn",
            "root_turn_id": self.root_turn_id,
            "thread_source": "user",
            "sandbox": "none",
            "sandbox_mode": "danger-full-access",
            "auto_review_enabled": false,
            "node_repl_auto_review_required": false,
            "node_repl_disabled": false,
            "workspaces": {
                "/workspace": {
                    "associated_remote_urls": {
                        "origin": "https://example.invalid/origin.git"
                    },
                    "latest_git_commit_hash": "0000000000000000000000000000000000000000",
                    "has_changes": false
                }
            },
            "turn_started_at_unix_ms": self.started_at_unix_ms as u64
        })
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
/// wire-shape. Prompt content and tools are intentionally preserved.
pub fn apply_codex_response_shape(request: &mut Value, context: &CodexRequestContext) {
    let Some(object) = request.as_object_mut() else {
        return;
    };

    let preserved = object
        .iter()
        .filter(|(key, _)| {
            matches!(
                key.as_str(),
                "model" | "instructions" | "input" | "tools" | "max_output_tokens"
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
}

fn unix_timestamp_ms() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn synthetic_uuid() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_ID: AtomicU64 = AtomicU64::new(0);
    let counter = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let process = u128::from(std::process::id());
    let mixed = nanos ^ (process << 64) ^ (u128::from(counter).wrapping_mul(0x9e37_79b9_7f4a_7c15));

    let bytes = mixed.to_be_bytes();
    format!(
        "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-4{:01x}{:02x}-8{:01x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        bytes[0],
        bytes[1],
        bytes[2],
        bytes[3],
        bytes[4],
        bytes[5],
        bytes[6] & 0x0f,
        bytes[7],
        bytes[8] & 0x0f,
        bytes[9],
        bytes[10],
        bytes[11],
        bytes[12],
        bytes[13],
        bytes[14],
        bytes[15]
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fake_client_parse_supports_all_modes() {
        assert_eq!(FakeClient::parse("auto"), Some(FakeClient::Auto));
        assert_eq!(FakeClient::parse("codex"), Some(FakeClient::Codex));
        assert_eq!(FakeClient::parse("anthropic"), Some(FakeClient::Anthropic));
        assert_eq!(FakeClient::parse("other"), None);
    }

    #[test]
    fn codex_identity_uses_stable_session_ids_within_a_context() {
        let identity = CodexIdentity::new("installation");
        let context = identity.turn_context();

        assert_eq!(context.session_id, context.thread_id);
        assert_eq!(context.window_id(), format!("{}:0", context.session_id));
        assert_eq!(context.installation_id, "installation");
    }

    #[test]
    fn response_shape_preserves_prompt_and_tools() {
        let mut request = serde_json::json!({
            "model": "gpt-5.6-sol",
            "instructions": "core and workspace instructions",
            "input": [{"type": "message"}],
            "tools": [{"name": "fs__read"}],
            "temperature": 0.2,
            "service_tier": "priority"
        });
        let identity = CodexIdentity::new("installation");
        let context = identity.turn_context();
        apply_codex_response_shape(&mut request, &context);

        assert_eq!(request["model"], "gpt-5.6-sol");
        assert_eq!(request["instructions"], "core and workspace instructions");
        assert_eq!(request["parallel_tool_calls"], false);
        assert_eq!(request["tool_choice"], "auto");
        assert_eq!(request["store"], false);
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
    fn headers_expose_no_real_local_paths() {
        let identity = CodexIdentity::new("installation");
        let context = identity.turn_context();
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
        let identity = CodexIdentity::new("installation");
        let context = identity.turn_context();
        let headers = context.anthropic_headers();

        assert!(
            headers.iter().all(|(name, _)| name != "accept"),
            "anthropic transport already sends its own Accept header"
        );
        assert!(
            headers
                .iter()
                .any(|(name, value)| name == "originator" && value == "codex_exec")
        );
        assert!(
            headers
                .iter()
                .any(|(name, _)| name == "x-codex-turn-metadata")
        );
    }
}
