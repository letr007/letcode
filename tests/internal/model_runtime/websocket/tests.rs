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
            let (mut stream, _) = tokio::time::timeout(Duration::from_secs(5), listener.accept())
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
        let failure = match TurnLocalWsSession::connect(
            client,
            request,
            Duration::from_secs(10),
            &[],
        )
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
