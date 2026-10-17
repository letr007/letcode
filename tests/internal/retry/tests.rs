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
