use std::error::Error;

use reqwest::StatusCode;

pub(crate) fn is_retryable_reqwest_error(error: &reqwest::Error) -> bool {
    if let Some(status) = error.status() {
        return is_retryable_http_status(status);
    }
    if error.is_builder() || error.is_redirect() {
        return false;
    }
    if error.is_decode() {
        return error_chain_has_transient_message(error);
    }
    error.is_timeout() || error.is_connect() || error.is_body() || error.is_request()
}

pub(crate) fn is_retryable_http_status(status: StatusCode) -> bool {
    status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error()
}

fn is_transient_error_message(message: &str) -> bool {
    let message = message.to_ascii_lowercase();
    [
        "timeout",
        "timed out",
        "connection",
        "connect",
        "reset",
        "closed",
        "eof",
        "broken pipe",
        "incomplete message",
        "transport",
        "http2",
        "h2",
        "temporary",
        "temporarily",
    ]
    .iter()
    .any(|needle| message.contains(needle))
}

pub(crate) fn is_retryable_provider_error_message(message: &str) -> bool {
    is_transient_error_message(message) || content_has_gateway_or_upstream_signal(message)
}

pub(crate) fn is_transient_stream_decode_error_message(message: &str) -> bool {
    is_retryable_provider_error_message(message) || {
        let message = message.to_ascii_lowercase();
        [
            "unexpected eof",
            "end of file",
            "eof while parsing",
            "unterminated",
            "incomplete",
            "error decoding response body",
            "error decoding",
            "unexpected end of file",
        ]
        .iter()
        .any(|needle| message.contains(needle))
    }
}

fn content_has_gateway_or_upstream_signal(message: &str) -> bool {
    let message = message.to_ascii_lowercase();
    [
        "bad gateway",
        "gateway timeout",
        "service unavailable",
        "upstream",
        "502",
        "503",
        "504",
    ]
    .iter()
    .any(|needle| message.contains(needle))
}

fn error_chain_has_transient_message(error: &(dyn Error + 'static)) -> bool {
    if is_transient_stream_decode_error_message(&error.to_string()) {
        return true;
    }
    let mut source = error.source();
    while let Some(error) = source {
        if is_transient_stream_decode_error_message(&error.to_string()) {
            return true;
        }
        source = error.source();
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_classifier_only_accepts_transient_statuses() {
        assert!(is_retryable_http_status(StatusCode::TOO_MANY_REQUESTS));
        assert!(is_retryable_http_status(StatusCode::INTERNAL_SERVER_ERROR));
        assert!(is_retryable_http_status(StatusCode::BAD_GATEWAY));
        assert!(!is_retryable_http_status(StatusCode::BAD_REQUEST));
        assert!(!is_retryable_http_status(StatusCode::UNAUTHORIZED));
        assert!(!is_retryable_http_status(StatusCode::FORBIDDEN));
        assert!(!is_retryable_http_status(StatusCode::NOT_FOUND));
    }

    #[test]
    fn transient_message_classifier_accepts_transport_read_failures_only() {
        assert!(is_transient_stream_decode_error_message(
            "error reading a body from connection: end of file before message length reached"
        ));
        assert!(is_transient_stream_decode_error_message(
            "connection reset by peer"
        ));
        assert!(is_transient_stream_decode_error_message(
            "502 Bad Gateway while decoding stream event"
        ));
        assert!(!is_transient_stream_decode_error_message(
            "expected value at line 1 column 1"
        ));
        assert!(!is_transient_stream_decode_error_message(
            "invalid gzip header"
        ));
    }
}
