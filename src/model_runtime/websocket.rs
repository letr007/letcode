use super::{FailureKind, FailurePhase, ModelFailure, RetryHint};
use futures_util::{SinkExt, StreamExt};
use std::time::Duration;
use tokio_tungstenite::tungstenite::handshake::client::generate_key;
use tokio_tungstenite::tungstenite::{self, Message};
use tokio_tungstenite::{WebSocketStream, tungstenite::handshake::derive_accept_key};

type Socket = WebSocketStream<reqwest::Upgraded>;

/// A turn-local, single-lane WebSocket session. Callers own the session and
/// must await each send before reading or sending the next frame.
pub(crate) struct TurnLocalWsSession {
    socket: Socket,
}

impl TurnLocalWsSession {
    pub(crate) async fn connect(
        client: reqwest::Client,
        request: reqwest::Request,
        timeout: Duration,
        secrets: &[&str],
    ) -> Result<Self, ModelFailure> {
        let url = request.url().clone();
        let key = generate_key();
        let builder = client
            .get(url)
            .headers(request.headers().clone())
            .header("Upgrade", "websocket")
            .header("Connection", "Upgrade")
            .header("Sec-WebSocket-Key", &key)
            .header("Sec-WebSocket-Version", "13");
        let handshake = builder.build().map_err(|error| {
            ModelFailure::new(FailurePhase::Prepare, FailureKind::InvalidRequest)
                .with_code("websocket_handshake_request")
                .with_detail_redacted(error.to_string(), secrets)
        })?;

        let response = tokio::time::timeout(timeout, client.execute(handshake))
            .await
            .map_err(|_| {
                ModelFailure::new(FailurePhase::Transport, FailureKind::Timeout)
                    .with_code("websocket_connect_timeout")
                    .with_retry_hint(RetryHint::Retryable)
            })?
            .map_err(|error| handshake_failure(error, secrets))?;

        let status = response.status().as_u16();
        if status != 101 {
            let kind = match status {
                401 | 403 => FailureKind::Authentication,
                429 => FailureKind::RateLimited,
                _ => FailureKind::Http,
            };
            return Err(ModelFailure::new(FailurePhase::Transport, kind)
                .with_status(status)
                .with_code("websocket_handshake_status")
                .with_retry_hint(if status == 429 || status >= 500 {
                    RetryHint::Retryable
                } else {
                    RetryHint::Never
                }));
        }
        let expected_accept = derive_accept_key(key.as_bytes());
        let valid_accept = response
            .headers()
            .get("sec-websocket-accept")
            .and_then(|value| value.to_str().ok())
            .is_some_and(|value| value == expected_accept);
        if !valid_accept {
            return Err(
                ModelFailure::new(FailurePhase::Transport, FailureKind::MalformedResponse)
                    .with_status(status)
                    .with_code("websocket_invalid_accept"),
            );
        }

        let upgraded = response.upgrade().await.map_err(|error| {
            ModelFailure::new(FailurePhase::Transport, FailureKind::Http)
                .with_code("websocket_upgrade_failed")
                .with_retry_hint(RetryHint::Retryable)
                .with_detail_redacted(error.to_string(), secrets)
        })?;
        let socket =
            WebSocketStream::from_raw_socket(upgraded, tungstenite::protocol::Role::Client, None)
                .await;
        Ok(Self { socket })
    }

    pub(crate) async fn send_text(
        &mut self,
        frame: Vec<u8>,
        secrets: &[&str],
    ) -> Result<(), ModelFailure> {
        let text = String::from_utf8(frame).map_err(|_| {
            ModelFailure::new(FailurePhase::Prepare, FailureKind::InvalidRequest)
                .with_code("websocket_frame_utf8")
        })?;
        self.socket
            .send(Message::Text(text.into()))
            .await
            .map_err(|error| {
                ModelFailure::new(FailurePhase::Transport, FailureKind::Http)
                    .with_code("websocket_send_failed")
                    .with_retry_hint(RetryHint::Retryable)
                    .with_detail_redacted(error.to_string(), secrets)
            })
    }

    pub(crate) async fn next_text(
        &mut self,
        secrets: &[&str],
    ) -> Result<Option<Vec<u8>>, ModelFailure> {
        loop {
            match self.socket.next().await {
                Some(Ok(Message::Text(text))) => return Ok(Some(text.as_bytes().to_vec())),
                Some(Ok(Message::Ping(payload))) => {
                    self.socket
                        .send(Message::Pong(payload))
                        .await
                        .map_err(|error| {
                            ModelFailure::new(FailurePhase::Transport, FailureKind::Http)
                                .with_code("websocket_pong_failed")
                                .with_retry_hint(RetryHint::Retryable)
                                .with_detail_redacted(error.to_string(), secrets)
                        })?;
                }
                Some(Ok(Message::Pong(_))) => {}
                Some(Ok(Message::Close(_))) | None => return Ok(None),
                Some(Ok(Message::Binary(_))) => {
                    return Err(ModelFailure::new(
                        FailurePhase::Decode,
                        FailureKind::MalformedResponse,
                    )
                    .with_code("websocket_binary_frame"));
                }
                Some(Ok(Message::Frame(_))) => {}
                Some(Err(error)) => return Err(websocket_io_failure(error, secrets)),
            }
        }
    }
}

fn handshake_failure(error: reqwest::Error, secrets: &[&str]) -> ModelFailure {
    let kind = if error.is_timeout() {
        FailureKind::Timeout
    } else {
        FailureKind::Http
    };
    let mut failure = ModelFailure::new(FailurePhase::Transport, kind)
        .with_code("websocket_handshake_failed")
        .with_retry_hint(if error.is_timeout() {
            RetryHint::Retryable
        } else {
            RetryHint::Never
        })
        .with_detail_redacted(error.to_string(), secrets);
    if let Some(status) = error.status() {
        failure = failure.with_status(status.as_u16());
    }
    failure
}

fn websocket_io_failure(
    error: tokio_tungstenite::tungstenite::Error,
    secrets: &[&str],
) -> ModelFailure {
    let kind = match &error {
        tungstenite::Error::Io(error) if error.kind() == std::io::ErrorKind::TimedOut => {
            FailureKind::Timeout
        }
        _ => FailureKind::Http,
    };
    ModelFailure::new(FailurePhase::Transport, kind)
        .with_code("websocket_receive_failed")
        .with_retry_hint(RetryHint::Retryable)
        .with_detail_redacted(error.to_string(), secrets)
}

#[cfg(test)]
mod tests {
    use crate::model_runtime::{
        AuthScheme, ProviderTransport, RuntimeAuthConfig, RuntimeTransportConfig,
    };
    use futures_util::{SinkExt, StreamExt};
    use std::collections::BTreeMap;
    use tokio::net::TcpListener;
    use tokio_tungstenite::accept_hdr_async;
    use tokio_tungstenite::tungstenite::Message;

    #[tokio::test]
    #[allow(clippy::result_large_err)]
    async fn reqwest_upgrade_accepts_101_and_preserves_request_metadata_and_control_frames() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let callback = |request: &tokio_tungstenite::tungstenite::http::Request<()>,
                            response| {
                assert_eq!(request.uri().path(), "/responses");
                assert_eq!(request.uri().query(), Some("trace=one"));
                assert_eq!(request.headers()["authorization"], "Bearer bearer-secret");
                assert_eq!(request.headers()["x-provider-version"], "2026-09-01");
                assert_ne!(
                    request
                        .headers()
                        .get("accept")
                        .and_then(|value| value.to_str().ok()),
                    Some("text/event-stream")
                );
                assert_ne!(
                    request
                        .headers()
                        .get("content-type")
                        .and_then(|value| value.to_str().ok()),
                    Some("application/json")
                );
                Ok(response)
            };
            let mut socket = accept_hdr_async(stream, callback).await.unwrap();
            for round in 1..=2 {
                match socket.next().await.unwrap().unwrap() {
                    Message::Text(text) => {
                        let value: serde_json::Value = serde_json::from_str(&text).unwrap();
                        assert_eq!(value["type"], "response.create");
                    }
                    message => panic!("unexpected request message: {message:?}"),
                }
                socket
                    .send(Message::Ping(vec![round].into()))
                    .await
                    .unwrap();
                assert!(matches!(
                    socket.next().await.unwrap().unwrap(),
                    Message::Pong(payload) if payload.as_ref() == [round]
                ));
                socket
                    .send(Message::Text(
                        format!(
                            r#"{{"type":"response.output_text.delta","delta":"hello-{round}"}}"#
                        )
                        .into(),
                    ))
                    .await
                    .unwrap();
                socket
                    .send(Message::Text(
                        format!(
                            r#"{{"type":"response.completed","response":{{"id":"resp-{round}","status":"completed"}}}}"#
                        )
                        .into(),
                    ))
                    .await
                    .unwrap();
            }
        });

        let transport = ProviderTransport::new_for_endpoint(
            &RuntimeTransportConfig {
                websocket: true,
                ..RuntimeTransportConfig::default()
            },
            Some(&format!("http://{address}/responses")),
        )
        .unwrap();
        let auth = RuntimeAuthConfig {
            scheme: AuthScheme::Bearer,
            name: None,
            credential: Some("bearer-secret".into()),
            credential_env: None,
        };
        let request = transport
            .request(
                reqwest::Method::POST,
                &format!("http://{address}/responses"),
                "local",
                &auth,
                &BTreeMap::from([("x-provider-version".into(), "2026-09-01".into())]),
                &BTreeMap::from([("trace".into(), "one".into())]),
            )
            .unwrap();
        let mut session = transport
            .open_websocket(
                &crate::model_runtime::ProtocolId::new("responses").unwrap(),
                request,
                &auth,
            )
            .await
            .unwrap();
        for round in 1..=2 {
            session
                .send_text(
                    format!(r#"{{"type":"response.create","round":{round}}}"#).into_bytes(),
                    &["bearer-secret"],
                )
                .await
                .unwrap();
            let first = session
                .next_text(&["bearer-secret"])
                .await
                .unwrap()
                .unwrap();
            assert!(
                std::str::from_utf8(&first)
                    .unwrap()
                    .contains(&format!("hello-{round}"))
            );
            let second = session
                .next_text(&["bearer-secret"])
                .await
                .unwrap()
                .unwrap();
            assert!(
                std::str::from_utf8(&second)
                    .unwrap()
                    .contains(&format!("resp-{round}"))
            );
        }
        server.await.unwrap();
    }
}
