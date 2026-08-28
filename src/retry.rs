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
    let jitter_secs = if config.exponential_backoff {
        retry_jitter_secs(config.jitter_secs)
    } else {
        0
    };
    Duration::from_secs(retry_backoff_delay_secs(config, attempt).saturating_add(jitter_secs))
}

pub(crate) fn retry_delay_from_headers(
    config: &RetryConfig,
    attempt: usize,
    headers: &HeaderMap,
) -> Duration {
    retry_after_delay(headers).unwrap_or_else(|| retry_delay(config, attempt))
}

pub(crate) fn retry_after_delay(headers: &HeaderMap) -> Option<Duration> {
    let value = headers.get(RETRY_AFTER)?.to_str().ok()?.trim();
    if let Ok(seconds) = value.parse::<u64>() {
        return (seconds > 0).then(|| Duration::from_secs(seconds));
    }

    let retry_at = httpdate::parse_http_date(value).ok()?;
    retry_at.duration_since(SystemTime::now()).ok()
}

pub(crate) fn retry_backoff_delay_secs(config: &RetryConfig, attempt: usize) -> u64 {
    if !config.exponential_backoff {
        return config.initial_delay_secs;
    }
    let exponent = i32::try_from(attempt.saturating_sub(1)).unwrap_or(i32::MAX);
    let delay =
        (config.initial_delay_secs as f64) * (config.backoff_multiplier as f64).powi(exponent);
    if !delay.is_finite() || delay >= u64::MAX as f64 {
        return u64::MAX;
    }
    delay.round() as u64
}

fn retry_jitter_secs(max_jitter_secs: u64) -> u64 {
    if max_jitter_secs == 0 {
        return 0;
    }
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| u64::from(duration.subsec_nanos()) % max_jitter_secs.saturating_add(1))
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_retry_config() -> RetryConfig {
        RetryConfig {
            enabled: true,
            max_attempts: 3,
            max_recovery_attempts: 3,
            initial_delay_secs: 1,
            exponential_backoff: true,
            backoff_multiplier: 2.0,
            jitter_secs: 0,
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
    fn retry_delay_can_use_fixed_interval_without_exponential_backoff() {
        let mut config = test_retry_config();
        config.initial_delay_secs = 3;
        config.exponential_backoff = false;
        config.backoff_multiplier = 9.0;
        config.jitter_secs = 10;

        assert_eq!(retry_backoff_delay_secs(&config, 1), 3);
        assert_eq!(retry_backoff_delay_secs(&config, 5), 3);
        assert_eq!(retry_delay(&config, 5), Duration::from_secs(3));

        config.exponential_backoff = true;
        config.backoff_multiplier = 2.0;
        assert_eq!(retry_backoff_delay_secs(&config, 1), 3);
        assert_eq!(retry_backoff_delay_secs(&config, 3), 12);
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
