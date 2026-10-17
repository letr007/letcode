use super::{FailureKind, FailurePhase, ModelFailure, RetryHint};
use futures_util::{SinkExt, StreamExt};
use std::error::Error as StdError;
use std::io::ErrorKind;
use std::time::Duration;
use tokio::sync::{mpsc, oneshot};
use tokio_tungstenite::tungstenite::handshake::client::generate_key;
use tokio_tungstenite::tungstenite::{self, Message};
use tokio_tungstenite::{WebSocketStream, tungstenite::handshake::derive_accept_key};

type Socket = WebSocketStream<reqwest::Upgraded>;

pub(crate) struct TurnLocalWsSession {
    outgoing: mpsc::Sender<WsSend>,
    incoming: mpsc::Receiver<Result<Vec<u8>, ModelFailure>>,
    task: tokio::task::JoinHandle<()>,
}

struct WsSend {
    text: String,
    completed: oneshot::Sender<Result<(), ModelFailure>>,
}

impl Drop for TurnLocalWsSession {
    fn drop(&mut self) {
        self.task.abort();
    }
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
        let (outgoing, requests) = mpsc::channel(1);
        let (events, incoming) = mpsc::channel(1);
        let secrets = secrets.iter().map(|secret| (*secret).to_string()).collect();
        let task = tokio::spawn(drive_socket(socket, requests, events, secrets));
        Ok(Self {
            outgoing,
            incoming,
            task,
        })
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
        let text_bytes = text.len();
        let closed = || {
            ModelFailure::new(FailurePhase::Transport, FailureKind::Http)
                .with_code("websocket_send_failed")
                .with_retry_hint(RetryHint::Retryable)
                .with_detail_redacted(
                    format!("WebSocket connection closed before send; request_bytes={text_bytes}"),
                    secrets,
                )
        };
        let (completed, result) = oneshot::channel();
        self.outgoing
            .send(WsSend { text, completed })
            .await
            .map_err(|_| closed())?;
        result.await.map_err(|_| closed())?
    }

    pub(crate) fn is_closed(&self) -> bool {
        self.outgoing.is_closed() || self.task.is_finished()
    }

    pub(crate) async fn next_text(&mut self, secrets: &[&str]) -> Result<Vec<u8>, ModelFailure> {
        self.incoming
            .recv()
            .await
            .unwrap_or_else(|| Err(websocket_close_failure(None, None, secrets)))
    }
}

async fn drive_socket(
    mut socket: Socket,
    mut requests: mpsc::Receiver<WsSend>,
    events: mpsc::Sender<Result<Vec<u8>, ModelFailure>>,
    secrets: Vec<String>,
) {
    let secrets = secrets.iter().map(String::as_str).collect::<Vec<_>>();
    let mut last_sent_text_bytes = None;
    let failure = loop {
        tokio::select! {
            biased;
            message = socket.next() => match message {
                Some(Ok(Message::Text(text))) => {
                    if events.send(Ok(text.as_bytes().to_vec())).await.is_err() {
                        return;
                    }
                }
                // Tungstenite flushes queued Pong replies on the next read or write.
                Some(Ok(Message::Ping(_) | Message::Pong(_))) => {}
                Some(Ok(Message::Close(frame))) => {
                    requests.close();
                    let _ = socket.flush().await;
                    break websocket_close_failure(frame, last_sent_text_bytes, &secrets);
                }
                None => break websocket_close_failure(None, last_sent_text_bytes, &secrets),
                Some(Ok(Message::Binary(_))) => {
                    break ModelFailure::new(FailurePhase::Decode, FailureKind::MalformedResponse)
                        .with_code("websocket_binary_frame");
                }
                Some(Ok(Message::Frame(_))) => {}
                Some(Err(error)) => {
                    break websocket_io_failure(error, last_sent_text_bytes, &secrets);
                }
            },
            request = requests.recv() => {
                let Some(WsSend { text, completed }) = request else {
                    return;
                };
                let text_bytes = text.len();
                let result = socket.send(Message::Text(text.into())).await.map_err(|error| {
                    ModelFailure::new(FailurePhase::Transport, FailureKind::Http)
                        .with_code("websocket_send_failed")
                        .with_retry_hint(RetryHint::Retryable)
                        .with_detail_redacted(format!("{error}; request_bytes={text_bytes}"), &secrets)
                });
                let failure = result.as_ref().err().cloned();
                let _ = completed.send(result);
                if let Some(failure) = failure {
                    break failure;
                }
                last_sent_text_bytes = Some(text_bytes);
            }
        }
    };
    drop(requests);
    drop(socket);
    let _ = events.send(Err(failure)).await;
}

const MAX_HANDSHAKE_ERROR_SOURCES: usize = 8;

fn handshake_failure(error: reqwest::Error, secrets: &[&str]) -> ModelFailure {
    let timed_out = error.is_timeout();
    let status = error.status();
    let retryable = status.is_none() && (timed_out || error_chain_has_retryable_io(&error));
    let kind = if timed_out {
        FailureKind::Timeout
    } else {
        FailureKind::Http
    };
    let mut failure = ModelFailure::new(FailurePhase::Transport, kind)
        .with_code("websocket_handshake_failed")
        .with_retry_hint(if retryable {
            RetryHint::Retryable
        } else {
            RetryHint::Never
        })
        .with_detail_redacted(format_error_chain(&error), secrets);
    if let Some(status) = status {
        failure = failure.with_status(status.as_u16());
    }
    failure
}

fn error_chain_has_retryable_io(error: &(dyn StdError + 'static)) -> bool {
    let mut source = Some(error);
    for _ in 0..=MAX_HANDSHAKE_ERROR_SOURCES {
        let Some(error) = source else {
            break;
        };
        if error
            .downcast_ref::<std::io::Error>()
            .is_some_and(|error| is_retryable_io_kind(error.kind()))
        {
            return true;
        }
        source = error.source();
    }
    false
}

fn is_retryable_io_kind(kind: ErrorKind) -> bool {
    matches!(
        kind,
        ErrorKind::ConnectionReset
            | ErrorKind::ConnectionAborted
            | ErrorKind::BrokenPipe
            | ErrorKind::UnexpectedEof
            | ErrorKind::ConnectionRefused
            | ErrorKind::TimedOut
    )
}

fn format_error_chain(error: &(dyn StdError + 'static)) -> String {
    let mut detail = error.to_string();
    let mut source = error.source();
    for depth in 1..=MAX_HANDSHAKE_ERROR_SOURCES {
        let Some(error) = source else {
            break;
        };
        detail.push_str(&format!("; source[{depth}]={error}"));
        source = error.source();
    }
    detail
}

fn websocket_close_failure(
    frame: Option<tungstenite::protocol::CloseFrame>,
    request_bytes: Option<usize>,
    secrets: &[&str],
) -> ModelFailure {
    let message_too_big = frame
        .as_ref()
        .is_some_and(|frame| frame.code == tungstenite::protocol::frame::coding::CloseCode::Size);
    let mut detail = match frame {
        Some(frame) => format!(
            "peer closed WebSocket before terminal response: code={} reason={}",
            frame.code, frame.reason
        ),
        None => "WebSocket ended before terminal response without a close frame".to_owned(),
    };
    if let Some(request_bytes) = request_bytes {
        detail.push_str(&format!("; request_bytes={request_bytes}"));
    }
    let (phase, kind) = if message_too_big {
        (FailurePhase::Transport, FailureKind::Http)
    } else {
        (FailurePhase::Finish, FailureKind::MalformedResponse)
    };
    ModelFailure::new(phase, kind)
        .with_code(if message_too_big {
            "websocket_message_too_big"
        } else {
            "websocket_closed_before_terminal"
        })
        .with_retry_hint(if message_too_big {
            RetryHint::RetryAfterSeconds(0)
        } else {
            RetryHint::Retryable
        })
        .with_detail_redacted(detail, secrets)
}

fn websocket_io_failure(
    error: tokio_tungstenite::tungstenite::Error,
    request_bytes: Option<usize>,
    secrets: &[&str],
) -> ModelFailure {
    let kind = match &error {
        tungstenite::Error::Io(error) if error.kind() == std::io::ErrorKind::TimedOut => {
            FailureKind::Timeout
        }
        _ => FailureKind::Http,
    };
    let detail = match request_bytes {
        Some(request_bytes) => format!("{error}; request_bytes={request_bytes}"),
        None => error.to_string(),
    };
    ModelFailure::new(FailurePhase::Transport, kind)
        .with_code("websocket_receive_failed")
        .with_retry_hint(RetryHint::Retryable)
        .with_detail_redacted(detail, secrets)
}

#[cfg(test)]
mod tests {
    use super::{
        Duration, TurnLocalWsSession, format_error_chain, handshake_failure, is_retryable_io_kind,
    };
    use crate::model_runtime::{
        AuthScheme, ProviderTransport, RetryHint, RuntimeAuthConfig, RuntimeTransportConfig,
    };
    use futures_util::{SinkExt, StreamExt};
    use std::collections::BTreeMap;
    use std::error::Error as StdError;
    use std::io::ErrorKind;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;
    use tokio_tungstenite::accept_hdr_async;
    use tokio_tungstenite::tungstenite::Message;
    use tokio_tungstenite::tungstenite::protocol::CloseFrame;
    use tokio_tungstenite::tungstenite::protocol::frame::coding::CloseCode;

    #[test]
    fn retryable_io_classification_is_narrow_and_typed() {
        for kind in [
            ErrorKind::ConnectionReset,
            ErrorKind::ConnectionAborted,
            ErrorKind::BrokenPipe,
            ErrorKind::UnexpectedEof,
            ErrorKind::ConnectionRefused,
            ErrorKind::TimedOut,
        ] {
            assert!(is_retryable_io_kind(kind));
        }
        assert!(!is_retryable_io_kind(ErrorKind::InvalidInput));
        assert!(!is_retryable_io_kind(ErrorKind::NotFound));
    }

    #[test]
    fn source_chain_format_is_depth_bounded() {
        #[derive(Debug)]
        struct ChainedError {
            message: String,
            source: Option<Box<Self>>,
        }

        impl std::fmt::Display for ChainedError {
            fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str(&self.message)
            }
        }

        impl StdError for ChainedError {
            fn source(&self) -> Option<&(dyn StdError + 'static)> {
                self.source
                    .as_deref()
                    .map(|error| error as &(dyn StdError + 'static))
            }
        }

        let mut error = ChainedError {
            message: "source-9".to_owned(),
            source: None,
        };
        for index in (0..9).rev() {
            error = ChainedError {
                message: format!("source-{index}"),
                source: Some(Box::new(error)),
            };
        }
        let detail = format_error_chain(&error as &(dyn StdError + 'static));
        assert!(detail.contains("source[8]=source-8"));
        assert!(!detail.contains("source[9]="));
    }

    #[tokio::test]
    #[allow(clippy::result_large_err)]
    async fn reqwest_builder_error_is_not_retryable() {
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let error = client.get("http://[invalid").build().unwrap_err();
        let failure = handshake_failure(error, &[]);
        assert_eq!(failure.retry_hint, RetryHint::Never);
    }

    mod integration_tests {
        use super::*;

        #[cfg(unix)]
        #[tokio::test]
        #[allow(clippy::result_large_err)]
        async fn closed_handshake_preserves_redacted_source_diagnostic_and_is_retryable() {
            use std::os::fd::AsRawFd;

            let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
            let address = listener.local_addr().unwrap();
            let server = tokio::spawn(async move {
                let (stream, _) = tokio::time::timeout(Duration::from_secs(5), listener.accept())
                    .await
                    .unwrap()
                    .unwrap();
                let linger = libc::linger {
                    l_onoff: 1,
                    l_linger: 0,
                };
                let result = unsafe {
                    libc::setsockopt(
                        stream.as_raw_fd(),
                        libc::SOL_SOCKET,
                        libc::SO_LINGER,
                        (&linger as *const libc::linger).cast(),
                        std::mem::size_of::<libc::linger>() as libc::socklen_t,
                    )
                };
                assert_eq!(result, 0);
                drop(stream);
            });

            let client = reqwest::Client::builder().no_proxy().build().unwrap();
            let response = client
                .get(format!("http://{address}/?token=top-secret"))
                .timeout(Duration::from_secs(5))
                .send()
                .await
                .unwrap_err();
            let failure = handshake_failure(response, &["top-secret"]);
            assert_eq!(failure.retry_hint, RetryHint::Retryable);
            assert!(failure.detail().contains("source["));
            assert!(failure.detail().contains("connection"));
            assert!(!failure.detail().contains("top-secret"));
            server.await.unwrap();
        }

        #[tokio::test]
        #[allow(clippy::result_large_err)]
        async fn client_error_status_is_not_retryable() {
            let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
            let address = listener.local_addr().unwrap();
            let server = tokio::spawn(async move {
                let (mut stream, _) =
                    tokio::time::timeout(Duration::from_secs(5), listener.accept())
                        .await
                        .unwrap()
                        .unwrap();
                // Consume the handshake before closing; unread request bytes can
                // cause a TCP reset that hides the HTTP response on Windows.
                tokio::time::timeout(Duration::from_secs(5), async {
                    let mut request = Vec::new();
                    while !request.ends_with(b"\r\n\r\n") {
                        assert!(
                            request.len() < 16_384,
                            "handshake headers exceed test limit"
                        );
                        request.push(stream.read_u8().await.unwrap());
                    }
                })
                .await
                .expect("handshake request timed out");
                stream
                    .write_all(
                        b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                    )
                    .await
                    .unwrap();
            });

            let client = reqwest::Client::builder().no_proxy().build().unwrap();
            let request = client.get(format!("http://{address}/")).build().unwrap();
            let failure =
                match TurnLocalWsSession::connect(client, request, Duration::from_secs(10), &[])
                    .await
                {
                    Ok(_) => panic!("a 403 handshake must fail"),
                    Err(failure) => failure,
                };
            assert_eq!(failure.status, Some(403));
            assert_eq!(failure.retry_hint, RetryHint::Never);
            server.await.unwrap();
        }

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
                socket
                    .send(Message::Close(Some(CloseFrame {
                        code: CloseCode::Size,
                        reason: "Message Too Big".into(),
                    })))
                    .await
                    .unwrap();
            });

            let transport = ProviderTransport::new_for_endpoint(
                &RuntimeTransportConfig::default(),
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
                let first = session.next_text(&["bearer-secret"]).await.unwrap();
                assert!(
                    std::str::from_utf8(&first)
                        .unwrap()
                        .contains(&format!("hello-{round}"))
                );
                let second = session.next_text(&["bearer-secret"]).await.unwrap();
                assert!(
                    std::str::from_utf8(&second)
                        .unwrap()
                        .contains(&format!("resp-{round}"))
                );
            }
            let failure = session.next_text(&["bearer-secret"]).await.unwrap_err();
            assert_eq!(failure.code.as_deref(), Some("websocket_message_too_big"));
            assert_eq!(failure.retry_hint, RetryHint::RetryAfterSeconds(0));
            assert!(failure.detail().contains("code=1009"));
            assert!(failure.detail().contains("reason=Message Too Big"));
            assert!(failure.detail().contains("request_bytes=36"));
            server.await.unwrap();
        }
    }
}
