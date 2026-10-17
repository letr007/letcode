use super::{FailureKind, FailurePhase, ModelFailure, PreparedHttpRequest, ProtocolId};
use crate::fake::{FakeProfile, FakeRequestContext};
use serde_json::Value;
use std::collections::BTreeSet;

/// Per-request fake decoration selected independently from provider flavor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FakeRequestDecorator {
    profile: FakeProfile,
    context: FakeRequestContext,
    /// The WebSocket transport negotiates its protocol version with a beta
    /// header the HTTP transport does not send.
    responses_websocket: bool,
}

impl FakeRequestDecorator {
    pub fn new(
        profile: FakeProfile,
        protocol_id: &ProtocolId,
        context: FakeRequestContext,
        responses_websocket: bool,
    ) -> Result<Self, ModelFailure> {
        if !profile.supports_protocol_id(protocol_id) || context.profile() != profile {
            return Err(
                ModelFailure::new(FailurePhase::Prepare, FailureKind::UnsupportedProtocol)
                    .with_code("fake_protocol_mismatch"),
            );
        }
        Ok(Self {
            profile,
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
        if !self.profile.supports_protocol_id(protocol_id) {
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
                if !request.url.contains('?') {
                    request.url.push_str(crate::fake::CLAUDE_MESSAGES_QUERY);
                }
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
#[path = "../../tests/internal/model_runtime/decorator/tests.rs"]
mod tests;
