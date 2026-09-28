use super::{FailureKind, FailurePhase, ModelFailure, PreparedHttpRequest, ProtocolId};
use crate::fake::{FakeClient, FakeRequestContext};
use serde_json::Value;
use std::collections::BTreeSet;

/// Per-request fake decoration selected independently from provider flavor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FakeRequestDecorator {
    client: FakeClient,
    context: FakeRequestContext,
    /// The WebSocket transport negotiates its protocol version with a beta
    /// header the HTTP transport does not send.
    responses_websocket: bool,
}

impl FakeRequestDecorator {
    pub fn new(
        client: FakeClient,
        protocol_id: &ProtocolId,
        context: FakeRequestContext,
        responses_websocket: bool,
    ) -> Result<Self, ModelFailure> {
        if !client.supports_protocol_id(protocol_id)
            || !context.profile().supports_protocol_id(protocol_id)
        {
            return Err(
                ModelFailure::new(FailurePhase::Prepare, FailureKind::UnsupportedProtocol)
                    .with_code("fake_protocol_mismatch"),
            );
        }
        Ok(Self {
            client,
            context,
            responses_websocket,
        })
    }

    /// Decorate adapter-prepared wire data without wrapping or replacing the
    /// adapter decoder. Terminal validation therefore remains adapter-owned.
    pub fn decorate(
        &self,
        protocol_id: &ProtocolId,
        mut request: PreparedHttpRequest,
    ) -> Result<PreparedHttpRequest, ModelFailure> {
        if !self.client.supports_protocol_id(protocol_id) {
            return Err(
                ModelFailure::new(FailurePhase::Prepare, FailureKind::UnsupportedProtocol)
                    .with_code("fake_protocol_mismatch"),
            );
        }
        match (protocol_id.as_str(), &self.context) {
            ("responses", FakeRequestContext::Codex(context)) => {
                request.body = rewrite_body(&request.body, |body| {
                    crate::fake::apply_codex_response_shape(body, context);
                })?;
                let mut headers = context.headers();
                if self.responses_websocket {
                    headers.push((
                        "openai-beta".into(),
                        crate::fake::CODEX_RESPONSES_WEBSOCKET_BETA.to_string(),
                    ));
                }
                self.merge_headers(&mut request, headers)?;
            }
            ("anthropic", FakeRequestContext::Claude(context)) => {
                request.body = rewrite_body(&request.body, |body| {
                    crate::fake::apply_claude_body_shape(body, context);
                })?;
                self.replace_headers(&mut request, context.headers())?;
            }
            _ => {
                return Err(ModelFailure::new(
                    FailurePhase::Prepare,
                    FailureKind::UnsupportedProtocol,
                )
                .with_code("fake_protocol_mismatch"));
            }
        }
        Ok(request)
    }

    /// Installs headers the profile owns, replacing the adapter's value for the same name.
    fn replace_headers(
        &self,
        request: &mut PreparedHttpRequest,
        headers: Vec<(String, String)>,
    ) -> Result<(), ModelFailure> {
        for (name, value) in headers {
            let normalized = name.to_ascii_lowercase();
            let name =
                reqwest::header::HeaderName::from_bytes(normalized.as_bytes()).map_err(|_| {
                    ModelFailure::new(FailurePhase::Prepare, FailureKind::InvalidRequest)
                        .with_code("invalid_fake_header_name")
                })?;
            reqwest::header::HeaderValue::from_str(&value).map_err(|_| {
                ModelFailure::new(FailurePhase::Prepare, FailureKind::InvalidRequest)
                    .with_code("invalid_fake_header_value")
            })?;
            request
                .protocol_headers
                .insert(name.as_str().to_owned(), value);
        }
        Ok(())
    }

    fn merge_headers(
        &self,
        request: &mut PreparedHttpRequest,
        headers: Vec<(String, String)>,
    ) -> Result<(), ModelFailure> {
        let mut seen = request
            .protocol_headers
            .keys()
            .map(|name| name.to_ascii_lowercase())
            .collect::<BTreeSet<_>>();
        for (name, value) in headers {
            let normalized = name.to_ascii_lowercase();
            if !seen.insert(normalized.clone()) {
                if request.protocol_headers.get(&normalized) == Some(&value) {
                    continue;
                }
                return Err(
                    ModelFailure::new(FailurePhase::Prepare, FailureKind::InvalidRequest)
                        .with_code("duplicate_fake_header"),
                );
            }
            let name =
                reqwest::header::HeaderName::from_bytes(normalized.as_bytes()).map_err(|_| {
                    ModelFailure::new(FailurePhase::Prepare, FailureKind::InvalidRequest)
                        .with_code("invalid_fake_header_name")
                })?;
            reqwest::header::HeaderValue::from_str(&value).map_err(|_| {
                ModelFailure::new(FailurePhase::Prepare, FailureKind::InvalidRequest)
                    .with_code("invalid_fake_header_value")
            })?;
            request
                .protocol_headers
                .insert(name.as_str().to_owned(), value);
        }
        Ok(())
    }
}

fn rewrite_body(body: &[u8], apply: impl FnOnce(&mut Value)) -> Result<Vec<u8>, ModelFailure> {
    let mut value = serde_json::from_slice::<Value>(body).map_err(|error| {
        ModelFailure::new(FailurePhase::Prepare, FailureKind::InvalidRequest)
            .with_code("fake_request_body")
            .with_detail(error.to_string())
    })?;
    apply(&mut value);
    serde_json::to_vec(&value).map_err(|error| {
        ModelFailure::new(FailurePhase::Prepare, FailureKind::Internal)
            .with_code("fake_request_serialization")
            .with_detail(error.to_string())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fake::{FakeIdentity, FakeRequestContext};
    use crate::model_runtime::{HttpMethod, ModelStreamDecoder, TerminalStatus};
    use std::collections::BTreeMap;

    fn request(protocol: &str, body: Value) -> PreparedHttpRequest {
        let mut protocol_headers = BTreeMap::new();
        protocol_headers.insert("accept".into(), "text/event-stream".into());
        protocol_headers.insert("content-type".into(), "application/json".into());
        if protocol == "anthropic" {
            protocol_headers.insert("anthropic-version".into(), "2023-06-01".into());
        }
        PreparedHttpRequest {
            method: HttpMethod::Post,
            url: format!("https://example.invalid/{protocol}"),
            protocol_headers,
            body: serde_json::to_vec(&body).unwrap(),
            prompt_unit_origins: Vec::new(),
        }
    }

    fn codex_context() -> FakeRequestContext {
        FakeRequestContext::Codex(
            FakeIdentity::new("fake-installation")
                .turn_context(&crate::config::FakeConfig::default(), None),
        )
    }

    fn claude_context() -> FakeRequestContext {
        FakeRequestContext::Claude(
            FakeIdentity::new("fake-installation")
                .claude_turn_context(&crate::config::FakeConfig::default()),
        )
    }

    #[test]
    fn codex_decorator_carries_declared_metadata_without_credentials() {
        let protocol = ProtocolId::new("responses").unwrap();
        let decorator =
            FakeRequestDecorator::new(FakeClient::Codex, &protocol, codex_context(), false)
                .unwrap();
        let decorated = decorator
            .decorate(
                &protocol,
                request(
                    "responses",
                    serde_json::json!({
                        "model": "gpt",
                        "instructions": "system",
                        "input": [{"type":"message"}],
                        "tools": [],
                        "temperature": 0.2
                    }),
                ),
            )
            .unwrap();
        let body: Value = serde_json::from_slice(&decorated.body).unwrap();
        assert_eq!(body["model"], "gpt");
        assert_eq!(body["instructions"], "system");
        assert!(body.get("temperature").is_none());
        assert_eq!(body["stream"], true);
        assert_eq!(
            decorated.protocol_headers.get("accept").map(String::as_str),
            Some("text/event-stream")
        );
        assert_eq!(
            decorated
                .protocol_headers
                .keys()
                .filter(|name| name.as_str() == "accept")
                .count(),
            1
        );
        let wire = format!(
            "{}{}",
            String::from_utf8(decorated.body).unwrap(),
            decorated
                .protocol_headers
                .values()
                .cloned()
                .collect::<String>()
        );
        assert!(!wire.contains("authorization"));
        assert!(!wire.contains("api-key"));
        // The turn metadata is a compatibility projection of the declared
        // identity, not a second source of truth.
        assert!(
            decorated
                .protocol_headers
                .values()
                .any(|value| value.contains("fake-installation"))
        );
    }

    #[test]
    fn anthropic_decorator_carries_the_claude_code_profile() {
        let protocol = ProtocolId::new("anthropic").unwrap();
        let decorator =
            FakeRequestDecorator::new(FakeClient::Anthropic, &protocol, claude_context(), false)
                .unwrap();
        let original = serde_json::json!({
            "model":"claude",
            "messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}],
            "stream":true
        });
        let decorated = decorator
            .decorate(&protocol, request("anthropic", original.clone()))
            .unwrap();
        let body: Value = serde_json::from_slice(&decorated.body).unwrap();
        assert_eq!(body["model"], original["model"]);
        assert_eq!(body["messages"], original["messages"]);
        assert_eq!(body["stream"], original["stream"]);
        let user_id = body["metadata"]["user_id"].as_str().expect("user id");
        assert!(user_id.contains("fake-installation"), "{user_id}");
        assert!(user_id.contains("session_id"), "{user_id}");

        assert_eq!(
            decorated.protocol_headers["anthropic-version"],
            "2023-06-01"
        );
        assert_eq!(decorated.protocol_headers["accept"], "application/json");
        assert_eq!(decorated.protocol_headers["x-app"], "cli");
        let user_agent = &decorated.protocol_headers["user-agent"];
        assert!(user_agent.starts_with("claude-cli/"), "{user_agent}");
        assert!(user_agent.ends_with("(external, cli)"), "{user_agent}");
        assert!(decorated.protocol_headers.contains_key("anthropic-beta"));
        assert!(
            decorated
                .protocol_headers
                .contains_key("x-claude-code-session-id")
        );
        // Codex-only transport identity must not leak into this profile.
        assert!(!decorated.protocol_headers.contains_key("originator"));
        assert!(
            !decorated
                .protocol_headers
                .contains_key("x-codex-turn-metadata")
        );
    }

    #[test]
    fn claude_requests_share_the_session_but_not_the_request_id() {
        let protocol = ProtocolId::new("anthropic").unwrap();
        let decorator =
            FakeRequestDecorator::new(FakeClient::Anthropic, &protocol, claude_context(), false)
                .unwrap();
        let body = serde_json::json!({
            "model": "claude",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true
        });
        let first = decorator
            .decorate(&protocol, request("anthropic", body.clone()))
            .unwrap();
        let second = decorator
            .decorate(&protocol, request("anthropic", body))
            .unwrap();

        assert_eq!(
            first.protocol_headers["x-claude-code-session-id"],
            second.protocol_headers["x-claude-code-session-id"]
        );
        assert_ne!(
            first.protocol_headers["x-client-request-id"],
            second.protocol_headers["x-client-request-id"]
        );
        let session = |decorated: &PreparedHttpRequest| {
            let body: Value = serde_json::from_slice(&decorated.body).unwrap();
            body["metadata"]["user_id"]
                .as_str()
                .expect("user id")
                .to_string()
        };
        assert_eq!(session(&first), session(&second));
        assert!(session(&first).contains(&first.protocol_headers["x-claude-code-session-id"]));
    }

    #[test]
    fn claude_profile_replaces_the_adapter_transport_values() {
        let protocol = ProtocolId::new("anthropic").unwrap();
        let decorator =
            FakeRequestDecorator::new(FakeClient::Anthropic, &protocol, claude_context(), false)
                .unwrap();
        let mut prepared = request("anthropic", serde_json::json!({"model": "claude"}));
        prepared
            .protocol_headers
            .insert("anthropic-beta".into(), "provider-beta".into());
        prepared
            .protocol_headers
            .insert("accept".into(), "text/event-stream".into());

        let decorated = decorator.decorate(&protocol, prepared).unwrap();
        let beta = &decorated.protocol_headers["anthropic-beta"];
        assert!(!beta.contains("provider-beta"), "{beta}");
        assert!(beta.starts_with("claude-code-"), "{beta}");
        assert_eq!(decorated.protocol_headers["accept"], "application/json");
    }

    #[test]
    fn decorator_rejects_incompatible_protocols() {
        let completions = ProtocolId::new("completions").unwrap();
        assert!(
            FakeRequestDecorator::new(FakeClient::Auto, &completions, codex_context(), false)
                .is_err()
        );
        let responses = ProtocolId::new("responses").unwrap();
        assert!(
            FakeRequestDecorator::new(FakeClient::Codex, &responses, claude_context(), false)
                .is_err()
        );
        let anthropic = ProtocolId::new("anthropic").unwrap();
        assert!(
            FakeRequestDecorator::new(FakeClient::Anthropic, &anthropic, codex_context(), false)
                .is_err()
        );
    }

    #[test]
    fn only_websocket_responses_requests_carry_the_beta_header() {
        let protocol = ProtocolId::new("responses").unwrap();
        let body = serde_json::json!({
            "model": "gpt",
            "instructions": "system",
            "input": [{"type":"message"}],
            "tools": []
        });

        let http = FakeRequestDecorator::new(FakeClient::Codex, &protocol, codex_context(), false)
            .unwrap()
            .decorate(&protocol, request("responses", body.clone()))
            .unwrap();
        assert!(!http.protocol_headers.contains_key("openai-beta"));

        let websocket =
            FakeRequestDecorator::new(FakeClient::Codex, &protocol, codex_context(), true)
                .unwrap()
                .decorate(&protocol, request("responses", body))
                .unwrap();
        assert_eq!(
            websocket.protocol_headers["openai-beta"],
            crate::fake::CODEX_RESPONSES_WEBSOCKET_BETA
        );
    }

    struct TerminalDecoder;

    impl ModelStreamDecoder for TerminalDecoder {
        fn push(
            &mut self,
            _chunk: &[u8],
        ) -> Result<Vec<crate::model_runtime::ModelEvent>, ModelFailure> {
            Ok(Vec::new())
        }

        fn finish(&mut self) -> Result<Vec<crate::model_runtime::ModelEvent>, ModelFailure> {
            Ok(vec![crate::model_runtime::ModelEvent::Terminal {
                status: TerminalStatus::Length,
            }])
        }
    }

    #[test]
    fn request_decorator_has_no_decoder_or_terminal_override() {
        let mut decoder = TerminalDecoder;
        let events = decoder.finish().unwrap();
        assert!(matches!(
            events.as_slice(),
            [crate::model_runtime::ModelEvent::Terminal {
                status: TerminalStatus::Length,
                ..
            }]
        ));
    }
}
