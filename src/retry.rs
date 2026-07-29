use std::error::Error;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use async_openai::error::{ApiError, OpenAIError, StreamError};
use reqwest::StatusCode;
use reqwest::header::{HeaderMap, RETRY_AFTER};
use serde_json::error::Category as JsonErrorCategory;

use crate::config::RetryConfig;

pub(crate) fn should_retry_openai_stream_creation(
    config: &RetryConfig,
    attempt: usize,
    error: &OpenAIError,
) -> bool {
    can_retry_attempt(config, attempt)
        && match error {
            OpenAIError::Reqwest(error) => is_retryable_reqwest_error(error),
            OpenAIError::JSONDeserialize(source, content) => {
                is_retryable_json_deserialize_error(source, content)
            }
            OpenAIError::ApiError(error) => is_retryable_openai_api_error(error),
            _ => false,
        }
}

pub(crate) fn should_retry_openai_stream_read(
    config: &RetryConfig,
    attempt: usize,
    error: &OpenAIError,
) -> bool {
    can_retry_attempt(config, attempt)
        && match error {
            OpenAIError::Reqwest(error) => is_retryable_reqwest_error(error),
            OpenAIError::StreamError(error) => is_retryable_stream_error(error),
            OpenAIError::JSONDeserialize(source, content) => {
                is_retryable_json_deserialize_error(source, content)
            }
            OpenAIError::ApiError(error) => is_retryable_openai_api_error(error),
            _ => false,
        }
}

pub(crate) fn should_retry_reqwest_error(
    config: &RetryConfig,
    attempt: usize,
    error: &reqwest::Error,
) -> bool {
    can_retry_attempt(config, attempt) && is_retryable_reqwest_error(error)
}

pub(crate) fn should_retry_http_status(
    config: &RetryConfig,
    attempt: usize,
    status: StatusCode,
) -> bool {
    can_retry_attempt(config, attempt) && is_retryable_http_status(status)
}

pub(crate) fn can_retry_attempt(config: &RetryConfig, attempt: usize) -> bool {
    config.enabled && attempt < config.max_attempts
}

pub(crate) fn retry_delay_within_elapsed_budget(
    max_elapsed_ms: u64,
    retry_started_at: std::time::Instant,
    delay: Duration,
) -> Option<Duration> {
    retry_started_at
        .elapsed()
        .checked_add(delay)
        .filter(|elapsed| *elapsed <= Duration::from_millis(max_elapsed_ms))
        .map(|_| delay)
}

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

pub(crate) fn is_retryable_json_deserialize_error(
    error: &serde_json::Error,
    content: &str,
) -> bool {
    match error.classify() {
        JsonErrorCategory::Data => false,
        JsonErrorCategory::Io | JsonErrorCategory::Eof => true,
        JsonErrorCategory::Syntax => {
            is_transient_stream_decode_error_message(&error.to_string())
                || is_transient_stream_decode_error_message(content)
                || content_has_gateway_or_upstream_signal(content)
        }
    }
}

fn is_retryable_openai_api_error(error: &ApiError) -> bool {
    is_retryable_provider_error_fields(
        error.r#type.as_deref(),
        error.code.as_deref(),
        Some(error.message.as_str()),
    )
}

pub(crate) fn is_retryable_provider_error_fields(
    kind: Option<&str>,
    code: Option<&str>,
    message: Option<&str>,
) -> bool {
    let structured_fields = [kind, code].into_iter().flatten().collect::<Vec<_>>();
    if structured_fields
        .iter()
        .any(|value| is_deterministic_provider_error_field(value))
    {
        return false;
    }
    if structured_fields
        .iter()
        .any(|value| is_transient_provider_error_field(value))
    {
        return true;
    }
    message.is_some_and(is_retryable_provider_error_message)
}

fn is_deterministic_provider_error_field(value: &str) -> bool {
    let value = value.to_ascii_lowercase();
    [
        "invalid_request",
        "authentication",
        "permission",
        "unauthorized",
        "forbidden",
        "not_found",
        "content_filter",
        "context_length",
        "billing",
        "insufficient_quota",
        "unsupported",
        "model_not_found",
    ]
    .iter()
    .any(|marker| value.contains(marker))
}

fn is_transient_provider_error_field(value: &str) -> bool {
    let value = value.to_ascii_lowercase();
    [
        "rate_limit",
        "server_error",
        "temporarily_unavailable",
        "service_unavailable",
        "overloaded",
        "timeout",
    ]
    .iter()
    .any(|marker| value.contains(marker))
}

fn is_retryable_stream_error(error: &StreamError) -> bool {
    match error {
        StreamError::EventStream(message) => is_transient_stream_decode_error_message(message),
        StreamError::UnknownEvent(_) => false,
    }
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

pub(crate) fn retry_delay(config: &RetryConfig, attempt: usize) -> Duration {
    let base_ms = retry_backoff_delay_ms(config, attempt);
    let delay_ms = base_ms
        .saturating_add(retry_jitter_ms(config.jitter_ms))
        .min(config.max_delay_ms);
    Duration::from_millis(delay_ms)
}

pub(crate) fn retry_delay_from_headers(
    config: &RetryConfig,
    attempt: usize,
    headers: &HeaderMap,
) -> Duration {
    let Some(retry_after_ms) = retry_after_delay_ms(headers) else {
        return retry_delay(config, attempt);
    };
    Duration::from_millis(retry_after_ms.min(config.max_delay_ms))
}

pub(crate) fn retry_after_delay_ms(headers: &HeaderMap) -> Option<u64> {
    let value = headers.get(RETRY_AFTER)?.to_str().ok()?.trim();
    if let Ok(seconds) = value.parse::<u64>() {
        return (seconds > 0).then(|| seconds.saturating_mul(1_000));
    }

    let retry_at = httpdate::parse_http_date(value).ok()?;
    retry_at
        .duration_since(SystemTime::now())
        .ok()?
        .as_millis()
        .try_into()
        .ok()
}

pub(crate) fn retry_backoff_delay_ms(config: &RetryConfig, attempt: usize) -> u64 {
    let exponent = attempt.saturating_sub(1) as i32;
    let delay =
        (config.initial_delay_ms as f64) * (config.backoff_multiplier as f64).powi(exponent);
    let delay = delay.min(config.max_delay_ms as f64);
    delay.round() as u64
}

fn retry_jitter_ms(max_jitter_ms: u64) -> u64 {
    if max_jitter_ms == 0 {
        return 0;
    }
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| u64::from(duration.subsec_nanos()) % (max_jitter_ms + 1))
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_retry_config() -> RetryConfig {
        RetryConfig {
            enabled: true,
            max_attempts: 3,
            max_elapsed_ms: 10_000,
            max_recovery_attempts: 3,
            initial_delay_ms: 100,
            max_delay_ms: 250,
            backoff_multiplier: 2.0,
            jitter_ms: 0,
        }
    }

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
    fn policy_respects_enabled_and_attempt_limit() {
        let mut config = test_retry_config();

        assert!(can_retry_attempt(&config, 1));
        assert!(can_retry_attempt(&config, 2));
        assert!(!can_retry_attempt(&config, 3));

        config.enabled = false;
        assert!(!can_retry_attempt(&config, 1));
    }

    #[test]
    fn backoff_delay_is_capped() {
        let config = test_retry_config();

        assert_eq!(retry_backoff_delay_ms(&config, 1), 100);
        assert_eq!(retry_backoff_delay_ms(&config, 2), 200);
        assert_eq!(retry_backoff_delay_ms(&config, 3), 250);
        assert_eq!(retry_delay(&config, 3), Duration::from_millis(250));

        let jittered = RetryConfig {
            jitter_ms: 100,
            ..config
        };
        assert_eq!(retry_delay(&jittered, 3), Duration::from_millis(250));
    }

    #[test]
    fn retry_after_header_overrides_backoff_but_is_capped() {
        let config = RetryConfig {
            enabled: true,
            max_attempts: 3,
            max_elapsed_ms: 10_000,
            max_recovery_attempts: 3,
            initial_delay_ms: 250,
            max_delay_ms: 1_500,
            backoff_multiplier: 2.0,
            jitter_ms: 0,
        };
        let mut headers = HeaderMap::new();
        headers.insert(RETRY_AFTER, "1".parse().unwrap());

        assert_eq!(retry_after_delay_ms(&headers), Some(1_000));
        assert_eq!(
            retry_delay_from_headers(&config, 1, &headers),
            Duration::from_millis(1_000)
        );

        headers.insert(RETRY_AFTER, "30".parse().unwrap());
        assert_eq!(
            retry_delay_from_headers(&config, 1, &headers),
            Duration::from_millis(1_500)
        );
    }

    #[test]
    fn retry_after_malformed_zero_and_past_values_fall_back_to_local_backoff() {
        let config = RetryConfig {
            enabled: true,
            max_attempts: 3,
            max_elapsed_ms: 10_000,
            max_recovery_attempts: 3,
            initial_delay_ms: 250,
            max_delay_ms: 2_000,
            backoff_multiplier: 2.0,
            jitter_ms: 0,
        };
        let mut headers = HeaderMap::new();

        headers.insert(RETRY_AFTER, "not-a-date".parse().unwrap());
        assert_eq!(
            retry_delay_from_headers(&config, 1, &headers),
            Duration::from_millis(250)
        );

        headers.insert(RETRY_AFTER, "0".parse().unwrap());
        assert_eq!(
            retry_delay_from_headers(&config, 1, &headers),
            Duration::from_millis(250)
        );

        headers.insert(
            RETRY_AFTER,
            httpdate::fmt_http_date(SystemTime::now() - Duration::from_secs(60))
                .parse()
                .unwrap(),
        );
        assert_eq!(
            retry_delay_from_headers(&config, 1, &headers),
            Duration::from_millis(250)
        );

        headers.insert(RETRY_AFTER, "30".parse().unwrap());
        assert_eq!(
            retry_delay_from_headers(&config, 1, &headers),
            Duration::from_millis(2_000)
        );
    }

    #[test]
    fn elapsed_budget_rejects_a_retry_that_would_exceed_it() {
        let started_at = std::time::Instant::now();
        assert!(
            retry_delay_within_elapsed_budget(10_000, started_at, Duration::from_millis(1))
                .is_some()
        );
        assert!(
            retry_delay_within_elapsed_budget(0, started_at, Duration::from_millis(1)).is_none()
        );
    }

    #[test]
    fn retry_after_header_accepts_http_date() {
        let config = RetryConfig {
            enabled: true,
            max_attempts: 3,
            max_elapsed_ms: 10_000,
            max_recovery_attempts: 3,
            initial_delay_ms: 250,
            max_delay_ms: 2_000,
            backoff_multiplier: 2.0,
            jitter_ms: 0,
        };
        let retry_at = SystemTime::now() + Duration::from_secs(1);
        let mut headers = HeaderMap::new();
        headers.insert(
            RETRY_AFTER,
            httpdate::fmt_http_date(retry_at).parse().unwrap(),
        );

        let delay = retry_delay_from_headers(&config, 1, &headers);
        assert!(delay <= Duration::from_millis(2_000));
        assert!(delay > Duration::from_millis(0));
    }

    #[test]
    fn deterministic_provider_code_takes_precedence_over_transient_message_text() {
        assert!(!is_retryable_provider_error_fields(
            None,
            Some("invalid_request"),
            Some("temporary upstream connection failure")
        ));
        assert!(is_retryable_provider_error_fields(
            None,
            Some("server_error"),
            Some("invalid connection state")
        ));
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

    #[test]
    fn json_deserialize_classifier_accepts_truncated_and_gateway_like_content_only() {
        let eof = serde_json::from_str::<serde_json::Value>("{\"choices\":[")
            .expect_err("truncated json should fail");
        assert!(is_retryable_json_deserialize_error(&eof, "{\"choices\":["));

        let gateway = serde_json::from_str::<serde_json::Value>("<html>502 Bad Gateway</html>")
            .expect_err("gateway html should fail");
        assert!(is_retryable_json_deserialize_error(
            &gateway,
            "<html>502 Bad Gateway</html>"
        ));

        let missing_field = serde_json::from_str::<TestMissingField>(r#"{"present":1}"#)
            .expect_err("schema mismatch should fail");
        assert!(!is_retryable_json_deserialize_error(
            &missing_field,
            r#"{"present":1}"#
        ));

        let invalid_enum = serde_json::from_str::<TestEnumHolder>(r#"{"kind":"unknown"}"#)
            .expect_err("invalid enum should fail");
        assert!(!is_retryable_json_deserialize_error(
            &invalid_enum,
            r#"{"kind":"unknown"}"#
        ));
    }

    #[derive(Debug, serde::Deserialize)]
    struct TestMissingField {
        required: u32,
    }

    #[derive(Debug, serde::Deserialize)]
    struct TestEnumHolder {
        kind: TestEnum,
    }

    #[derive(Debug, serde::Deserialize)]
    #[serde(rename_all = "snake_case")]
    enum TestEnum {
        Expected,
    }
}
