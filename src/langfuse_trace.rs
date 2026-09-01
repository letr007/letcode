use anyhow::Result;
use serde_json::json;
#[cfg(test)]
use serde_json::{Map, Value};
use tracing::{Span, field};

#[cfg(test)]
use crate::agent::CacheUsageReport;
use crate::agent::{ToolEffectKind, ToolExecutionRecord, ToolExecutionStatus};
pub const TARGET: &str = "letcode::langfuse";

/// The cache-specific Langfuse metadata for one request iteration.  This
/// deliberately contains only scalar telemetry, never prompt material.
#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CacheMetadataProjection {
    pub configured: bool,
    pub hint_serialized: bool,
    pub retention_sent: Option<String>,
    pub stable_prefix_segments: u64,
    pub first_volatile_index: Option<u64>,
    pub stable_prompt_tokens: u64,
    pub volatile_prompt_tokens: u64,
    pub cacheable_prefix_tokens: u64,
    pub stable_after_boundary_tokens: u64,
    pub local_prefix_fingerprint: Option<String>,
    pub routing_key: Option<String>,
    pub actual_cached_tokens: Option<u64>,
}

#[cfg(test)]
pub(crate) fn cache_metadata_projection(
    cache: &CacheUsageReport,
    first_volatile_index: Option<usize>,
) -> CacheMetadataProjection {
    CacheMetadataProjection {
        configured: cache.configured,
        hint_serialized: cache.hint_serialized,
        retention_sent: cache
            .retention_sent
            .map(|retention| format!("{retention:?}")),
        stable_prefix_segments: as_u64(cache.stable_prefix_segments),
        first_volatile_index: first_volatile_index.map(as_u64),
        stable_prompt_tokens: cache.stable_prompt_tokens,
        volatile_prompt_tokens: cache.volatile_prompt_tokens,
        cacheable_prefix_tokens: cache.cacheable_prefix_tokens,
        stable_after_boundary_tokens: cache.stable_after_boundary_tokens,
        local_prefix_fingerprint: cache
            .local_prefix_fingerprint
            .as_deref()
            .map(bounded_metadata_string),
        routing_key: cache.routing_key.as_deref().map(bounded_metadata_string),
        actual_cached_tokens: cache.actual_cached_tokens,
    }
}

#[cfg(test)]
fn bounded_metadata_string(value: &str) -> String {
    value.chars().take(128).collect()
}

#[cfg(test)]
impl CacheMetadataProjection {
    fn values(&self) -> Map<String, Value> {
        let mut values = Map::new();
        values.insert("configured".into(), Value::Bool(self.configured));
        values.insert("hint_serialized".into(), Value::Bool(self.hint_serialized));
        values.insert(
            "stable_prefix_segments".into(),
            self.stable_prefix_segments.into(),
        );
        values.insert(
            "stable_prompt_tokens".into(),
            self.stable_prompt_tokens.into(),
        );
        values.insert(
            "volatile_prompt_tokens".into(),
            self.volatile_prompt_tokens.into(),
        );
        values.insert(
            "cacheable_prefix_tokens".into(),
            self.cacheable_prefix_tokens.into(),
        );
        values.insert(
            "stable_after_boundary_tokens".into(),
            self.stable_after_boundary_tokens.into(),
        );
        for (name, value) in [
            ("retention_sent", self.retention_sent.as_ref()),
            (
                "local_prefix_fingerprint",
                self.local_prefix_fingerprint.as_ref(),
            ),
            ("routing_key", self.routing_key.as_ref()),
        ] {
            if let Some(value) = value {
                values.insert(name.into(), Value::String(value.clone()));
            }
        }
        for (name, value) in [
            ("first_volatile_index", self.first_volatile_index),
            ("actual_cached_tokens", self.actual_cached_tokens),
        ] {
            if let Some(value) = value {
                values.insert(name.into(), value.into());
            }
        }
        values
    }
}

pub fn tool_span(turn_id: u64, tool_name: &str, call_id: &str, args_json_len: usize) -> Span {
    tracing::info_span!(
        target: TARGET,
        "tool.call",
        "otel.name" = %format!("tool {tool_name}"),
        "otel.kind" = "internal",
        "langfuse.observation.type" = "tool",
        "langfuse.observation.input" = %safe_tool_input_json(tool_name, args_json_len),
        "langfuse.observation.output" = field::Empty,
        "langfuse.observation.metadata.tool_name" = %tool_name,
        "langfuse.observation.metadata.call_id" = %call_id,
        "langfuse.observation.metadata.args_json_bytes" = as_u64(args_json_len),
        "langfuse.observation.metadata.permission_class" = field::Empty,
        "langfuse.observation.metadata.directive" = field::Empty,
        "langfuse.observation.metadata.execution_status" = field::Empty,
        "langfuse.observation.metadata.execution_rejection" = field::Empty,
        "langfuse.observation.metadata.effect_kind" = field::Empty,
        "langfuse.observation.metadata.output_ok" = field::Empty,
        "langfuse.observation.metadata.output_has_data" = field::Empty,
        "langfuse.observation.metadata.error_recoverable" = field::Empty,
        "langfuse.observation.metadata.error_message_chars" = field::Empty,
        "letcode.turn.id" = turn_id,
        "tool.name" = %tool_name,
        "tool.call_id" = %call_id,
        "tool.args_json.bytes" = as_u64(args_json_len),
        "tool.permission_class" = field::Empty,
        "tool.directive" = field::Empty,
        "tool.execution.status" = field::Empty,
        "tool.execution.rejection" = field::Empty,
        "tool.effect.kind" = field::Empty,
        "tool.output.ok" = field::Empty,
        "tool.output.has_data" = field::Empty,
        "tool.error.recoverable" = field::Empty,
        "tool.error.message.chars" = field::Empty,
        "otel.status_code" = field::Empty,
        "otel.status_description" = field::Empty,
        "error.type" = field::Empty,
    )
}

pub fn finish_tool_span(span: &Span, result: &Result<ToolExecutionRecord>) {
    match result {
        Ok(record) => {
            let effect_kind = tool_effect_kind_label(record.effects.kind);
            let error_recoverable = record.output.error.as_ref().map(|error| error.recoverable);
            let error_message_chars = record
                .output
                .error
                .as_ref()
                .map(|error| error.message.chars().count())
                .unwrap_or(0);

            span.record("tool.permission_class", record.permission_class.as_str());
            span.record("tool.directive", record.directive.as_str());
            span.record("tool.execution.status", record.status.as_str());
            span.record("tool.effect.kind", effect_kind);
            span.record(
                "langfuse.observation.metadata.permission_class",
                record.permission_class.as_str(),
            );
            span.record(
                "langfuse.observation.metadata.directive",
                record.directive.as_str(),
            );
            span.record(
                "langfuse.observation.metadata.execution_status",
                record.status.as_str(),
            );
            span.record("langfuse.observation.metadata.effect_kind", effect_kind);
            if let Some(rejection) = record.rejection {
                span.record("tool.execution.rejection", rejection.as_str());
                span.record(
                    "langfuse.observation.metadata.execution_rejection",
                    rejection.as_str(),
                );
            }
            span.record("tool.output.ok", record.output.ok);
            span.record("tool.output.has_data", record.output.data.is_some());
            span.record("langfuse.observation.metadata.output_ok", record.output.ok);
            span.record(
                "langfuse.observation.metadata.output_has_data",
                record.output.data.is_some(),
            );
            if let Some(error) = &record.output.error {
                span.record("tool.error.recoverable", error.recoverable);
                span.record(
                    "langfuse.observation.metadata.error_recoverable",
                    error.recoverable,
                );
            }
            span.record("tool.error.message.chars", as_u64(error_message_chars));
            span.record(
                "langfuse.observation.metadata.error_message_chars",
                as_u64(error_message_chars),
            );

            let safe_output = safe_tool_output_json(
                record.status.as_str(),
                record.rejection.map(|rejection| rejection.as_str()),
                effect_kind,
                record.output.ok,
                record.output.data.is_some(),
                error_recoverable,
                error_message_chars,
            );
            span.record("langfuse.observation.output", safe_output.as_str());

            if record.status == ToolExecutionStatus::Executed && !record.output.ok {
                record_error(span, "tool_output_error");
            } else {
                record_ok(span);
            }
        }
        Err(_) => {
            let safe_output = json!({ "status": "error" }).to_string();
            span.record("langfuse.observation.output", safe_output.as_str());
            record_error(span, "tool_execution_error");
        }
    }
}

fn record_ok(span: &Span) {
    span.record("otel.status_code", "Ok");
}

fn record_error(span: &Span, error_type: &'static str) {
    span.record("otel.status_code", "Error");
    span.record("otel.status_description", error_type);
    span.record("error.type", error_type);
}

fn safe_tool_input_json(tool_name: &str, args_json_len: usize) -> String {
    json!({
        "tool_name": tool_name,
        "args_json_bytes": as_u64(args_json_len),
    })
    .to_string()
}

fn safe_tool_output_json(
    status: &'static str,
    rejection: Option<&'static str>,
    effect_kind: &'static str,
    output_ok: bool,
    output_has_data: bool,
    error_recoverable: Option<bool>,
    error_message_chars: usize,
) -> String {
    json!({
        "status": status,
        "rejection": rejection,
        "effect_kind": effect_kind,
        "output_ok": output_ok,
        "output_has_data": output_has_data,
        "error_recoverable": error_recoverable,
        "error_message_chars": as_u64(error_message_chars),
    })
    .to_string()
}

fn tool_effect_kind_label(kind: ToolEffectKind) -> &'static str {
    match kind {
        ToolEffectKind::Read => "read",
        ToolEffectKind::Write => "write",
        ToolEffectKind::Command => "command",
        ToolEffectKind::Validation => "validation",
        ToolEffectKind::WorkflowControl => "workflow_control",
        ToolEffectKind::Diagnostic => "diagnostic",
        ToolEffectKind::Unknown => "unknown",
    }
}

fn as_u64(value: usize) -> u64 {
    value.try_into().unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cache_report() -> CacheUsageReport {
        CacheUsageReport {
            configured: true,
            hint_serialized: true,
            retention_sent: None,
            stable_prefix_segments: 2,
            stable_prompt_tokens: 120,
            volatile_prompt_tokens: 30,
            cacheable_prefix_tokens: 100,
            stable_after_boundary_tokens: 20,
            local_prefix_fingerprint: Some("fp-123".into()),
            routing_key: Some("route-123".into()),
            actual_cached_tokens: None,
        }
    }

    #[test]
    fn cache_metadata_projection_preserves_absent_and_zero_boundaries() {
        let no_prefix = CacheUsageReport {
            stable_prefix_segments: 0,
            local_prefix_fingerprint: None,
            routing_key: None,
            ..cache_report()
        };
        let absent = cache_metadata_projection(&no_prefix, None);
        assert_eq!(absent.first_volatile_index, None);
        assert_eq!(absent.local_prefix_fingerprint, None);
        assert_eq!(absent.routing_key, None);

        let volatile_first = cache_metadata_projection(&no_prefix, Some(0));
        assert_eq!(volatile_first.first_volatile_index, Some(0));
        assert_eq!(volatile_first.local_prefix_fingerprint, None);
        assert_eq!(volatile_first.routing_key, None);

        let stable_prefix = cache_metadata_projection(&cache_report(), Some(2));
        assert_eq!(stable_prefix.stable_prefix_segments, 2);
        assert_eq!(stable_prefix.first_volatile_index, Some(2));
        assert_eq!(
            stable_prefix.local_prefix_fingerprint.as_deref(),
            Some("fp-123")
        );
        assert_eq!(stable_prefix.routing_key.as_deref(), Some("route-123"));
    }

    #[test]
    fn cache_metadata_projection_keeps_plan_and_actual_usage_separate_and_safe() {
        let prepared = cache_metadata_projection(&cache_report(), Some(2));
        let actual =
            cache_metadata_projection(&cache_report().with_actual_cached_tokens(77), Some(2));
        assert_eq!(prepared.actual_cached_tokens, None);
        assert_eq!(actual.actual_cached_tokens, Some(77));
        assert_eq!(prepared.stable_prompt_tokens, actual.stable_prompt_tokens);
        assert_eq!(
            prepared.cacheable_prefix_tokens,
            actual.cacheable_prefix_tokens
        );

        let values = actual.values();
        assert!(
            values
                .values()
                .all(|value| matches!(value, Value::Bool(_) | Value::Number(_) | Value::String(_)))
        );
        let serialized = Value::Object(values).to_string();
        for forbidden in [
            "user prompt",
            "evidence",
            "tool arguments",
            "/private/source",
            "canonical bytes",
        ] {
            assert!(!serialized.contains(forbidden));
        }
        assert!(
            actual
                .local_prefix_fingerprint
                .as_ref()
                .is_some_and(|value| value.len() <= 128)
        );
        assert!(
            actual
                .routing_key
                .as_ref()
                .is_some_and(|value| value.len() <= 128)
        );
    }
}
