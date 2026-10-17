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
#[path = "../../tests/internal/model_runtime/websocket/tests.rs"]
mod tests;
