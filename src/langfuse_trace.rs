use anyhow::Result;
use serde_json::json;
#[cfg(test)]
use serde_json::{Map, Value};
use tracing::{Span, field};

use crate::agent::{CacheUsageReport, ToolEffectKind, ToolExecutionRecord, ToolExecutionStatus};
use crate::request_builder::BudgetReport;
use crate::request_builder::prompt_plan::PromptPlan;

pub const TARGET: &str = "letcode::langfuse";

/// The cache-specific Langfuse metadata for one request iteration.  This
/// deliberately contains only scalar telemetry, never prompt material.
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

fn bounded_metadata_string(value: &str) -> String {
    value.chars().take(128).collect()
}

impl CacheMetadataProjection {
    #[cfg(test)]
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

pub fn llm_turn_span(
    turn_id: u64,
    protocol: &'static str,
    model: &str,
    max_iterations: Option<usize>,
    max_tool_calls: Option<usize>,
    user_input_chars: usize,
    history_len: usize,
) -> Span {
    tracing::info_span!(
        target: TARGET,
        "llm.turn",
        "otel.name" = %format!("letcode turn {turn_id}"),
        "otel.kind" = "internal",
        "langfuse.observation.type" = "span",
        "langfuse.observation.input" = %safe_turn_input_json(
            user_input_chars,
            history_len,
            max_iterations,
            max_tool_calls,
        ),
        "langfuse.observation.output" = field::Empty,
        "langfuse.trace.name" = %format!("letcode turn {turn_id}"),
        "langfuse.trace.metadata.turn_id" = turn_id,
        "langfuse.trace.metadata.protocol" = protocol,
        "langfuse.trace.metadata.model" = %model,
        "langfuse.trace.metadata.user_input_chars" = as_u64(user_input_chars),
        "langfuse.trace.metadata.history_items" = as_u64(history_len),
        "langfuse.trace.metadata.max_iterations" = max_iterations.map(as_u64).unwrap_or(0),
        "langfuse.trace.metadata.max_iterations_unbounded" = max_iterations.is_none(),
        "langfuse.trace.metadata.max_tool_calls" = max_tool_calls.map(as_u64).unwrap_or(0),
        "langfuse.trace.metadata.max_tool_calls_unbounded" = max_tool_calls.is_none(),
        "langfuse.trace.metadata.tool_call_count" = field::Empty,
        "langfuse.trace.metadata.continuation_count" = field::Empty,
        "langfuse.trace.metadata.output_chars" = field::Empty,
        "gen_ai.operation.name" = "chat",
        "gen_ai.request.model" = %model,
        "letcode.turn.id" = turn_id,
        "letcode.protocol" = protocol,
        "letcode.user_input.chars" = as_u64(user_input_chars),
        "letcode.history.items" = as_u64(history_len),
        "letcode.max_iterations" = max_iterations.map(as_u64).unwrap_or(0),
        "letcode.max_iterations.unbounded" = max_iterations.is_none(),
        "letcode.max_tool_calls" = max_tool_calls.map(as_u64).unwrap_or(0),
        "letcode.max_tool_calls.unbounded" = max_tool_calls.is_none(),
        "letcode.tool_call.count" = field::Empty,
        "letcode.continuation.count" = field::Empty,
        "letcode.output.chars" = field::Empty,
        "otel.status_code" = field::Empty,
        "otel.status_description" = field::Empty,
        "error.type" = field::Empty,
    )
}

pub fn finish_llm_turn_span(
    span: &Span,
    result: &Result<String>,
    tool_call_count: usize,
    continuation_count: usize,
    history_len: usize,
) {
    span.record("letcode.tool_call.count", as_u64(tool_call_count));
    span.record("letcode.continuation.count", as_u64(continuation_count));
    span.record("letcode.history.items", as_u64(history_len));
    span.record(
        "langfuse.trace.metadata.tool_call_count",
        as_u64(tool_call_count),
    );
    span.record(
        "langfuse.trace.metadata.continuation_count",
        as_u64(continuation_count),
    );
    span.record("langfuse.trace.metadata.history_items", as_u64(history_len));
    match result {
        Ok(output) => {
            let output_chars = output.chars().count();
            span.record("letcode.output.chars", as_u64(output_chars));
            span.record("langfuse.trace.metadata.output_chars", as_u64(output_chars));
            let safe_output = safe_turn_output_json(
                output_chars,
                tool_call_count,
                continuation_count,
                history_len,
                "ok",
            );
            span.record("langfuse.observation.output", safe_output.as_str());
            record_ok(span);
        }
        Err(_) => {
            let safe_output =
                safe_turn_output_json(0, tool_call_count, continuation_count, history_len, "error");
            span.record("langfuse.observation.output", safe_output.as_str());
            record_error(span, "llm_turn_error");
        }
    }
}

pub fn llm_iteration_span(
    turn_id: u64,
    protocol: &'static str,
    model: &str,
    iteration: usize,
    history_len: usize,
    tool_call_count: usize,
    tool_definitions_count: usize,
) -> Span {
    tracing::info_span!(
        target: TARGET,
        "llm.stream",
        "otel.name" = %format!("{protocol} stream"),
        "otel.kind" = "client",
        "langfuse.observation.type" = "generation",
        "langfuse.observation.input" = %safe_iteration_input_json(
            protocol,
            iteration,
            history_len,
            tool_call_count,
            tool_definitions_count,
        ),
        "langfuse.observation.output" = field::Empty,
        "langfuse.observation.model.name" = %model,
        "langfuse.observation.usage_details" = field::Empty,
        "langfuse.observation.metadata.protocol" = protocol,
        "langfuse.observation.metadata.turn_id" = turn_id,
        "langfuse.observation.metadata.iteration" = as_u64(iteration),
        "langfuse.observation.metadata.history_items" = as_u64(history_len),
        "langfuse.observation.metadata.tool_call_count_before" = as_u64(tool_call_count),
        "langfuse.observation.metadata.tool_definitions_count" = as_u64(tool_definitions_count),
        "langfuse.observation.metadata.context_window_tokens" = field::Empty,
        "langfuse.observation.metadata.input_budget_tokens" = field::Empty,
        "langfuse.observation.metadata.request_estimated_tokens" = field::Empty,
        "langfuse.observation.metadata.prelude_estimated_tokens" = field::Empty,
        "langfuse.observation.metadata.protected_estimated_tokens" = field::Empty,
        "langfuse.observation.metadata.retained_history_estimated_tokens" = field::Empty,
        "langfuse.observation.metadata.tools_estimated_tokens" = field::Empty,
        "langfuse.observation.metadata.evidence_estimated_tokens" = field::Empty,
        "langfuse.observation.metadata.original_history_items" = field::Empty,
        "langfuse.observation.metadata.retained_history_items" = field::Empty,
        "langfuse.observation.metadata.dropped_history_items" = field::Empty,
        "langfuse.observation.metadata.selected_evidence_items" = field::Empty,
        "langfuse.observation.metadata.dropped_evidence_items" = field::Empty,
        "langfuse.observation.metadata.request_truncated" = field::Empty,
        "langfuse.observation.metadata.prompt_segment_count" = field::Empty,
        "langfuse.observation.metadata.prompt_contributor_count" = field::Empty,
        "langfuse.observation.metadata.prompt_stable_prefix_hash" = field::Empty,
        "langfuse.observation.metadata.cache_configured" = field::Empty,
        "langfuse.observation.metadata.cache_hint_serialized" = field::Empty,
        "langfuse.observation.metadata.cache_retention_sent" = field::Empty,
        "langfuse.observation.metadata.cache_stable_prefix_segments" = field::Empty,
        "langfuse.observation.metadata.cache_has_stable_prefix" = field::Empty,
        "langfuse.observation.metadata.cache_first_volatile_index" = field::Empty,
        "langfuse.observation.metadata.cache_stable_prompt_tokens" = field::Empty,
        "langfuse.observation.metadata.cache_volatile_prompt_tokens" = field::Empty,
        "langfuse.observation.metadata.cache_cacheable_prefix_tokens" = field::Empty,
        "langfuse.observation.metadata.cache_stable_after_boundary_tokens" = field::Empty,
        "langfuse.observation.metadata.cache_local_prefix_fingerprint" = field::Empty,
        "langfuse.observation.metadata.cache_routing_key" = field::Empty,
        "langfuse.observation.metadata.cache_actual_cached_tokens" = field::Empty,
        "langfuse.observation.metadata.output_chars" = field::Empty,
        "langfuse.observation.metadata.tool_call_count" = field::Empty,
        "langfuse.observation.metadata.response_items" = field::Empty,
        "gen_ai.operation.name" = "chat",
        "gen_ai.request.model" = %model,
        "letcode.turn.id" = turn_id,
        "letcode.protocol" = protocol,
        "letcode.iteration" = as_u64(iteration),
        "letcode.history.items" = as_u64(history_len),
        "letcode.tool_call.count_before" = as_u64(tool_call_count),
        "letcode.tool_definitions.count" = as_u64(tool_definitions_count),
        "letcode.request.estimated_tokens" = field::Empty,
        "letcode.context_window.tokens" = field::Empty,
        "letcode.input_budget.tokens" = field::Empty,
        "letcode.prelude.estimated_tokens" = field::Empty,
        "letcode.protected.estimated_tokens" = field::Empty,
        "letcode.retained_history.estimated_tokens" = field::Empty,
        "letcode.tools.estimated_tokens" = field::Empty,
        "letcode.evidence.estimated_tokens" = field::Empty,
        "letcode.history.original_items" = field::Empty,
        "letcode.history.retained_items" = field::Empty,
        "letcode.history.dropped_items" = field::Empty,
        "letcode.evidence.selected_items" = field::Empty,
        "letcode.evidence.dropped_items" = field::Empty,
        "letcode.request.truncated" = field::Empty,
        "letcode.prompt.segments" = field::Empty,
        "letcode.prompt.contributors" = field::Empty,
        "gen_ai.usage.input_tokens" = field::Empty,
        "gen_ai.usage.output_tokens" = field::Empty,
        "gen_ai.usage.cached_tokens" = field::Empty,
        "gen_ai.usage.total_tokens" = field::Empty,
        "gen_ai.response.finish_reasons" = field::Empty,
        "letcode.output.chars" = field::Empty,
        "letcode.tool_call.count" = field::Empty,
        "letcode.response.items" = field::Empty,
        "otel.status_code" = field::Empty,
        "otel.status_description" = field::Empty,
        "error.type" = field::Empty,
    )
}

pub fn record_llm_request_budget(span: &Span, budget: &BudgetReport) {
    span.record(
        "letcode.request.estimated_tokens",
        budget.estimated_request_tokens,
    );
    span.record(
        "letcode.context_window.tokens",
        budget.context_window_tokens,
    );
    span.record("letcode.input_budget.tokens", budget.input_budget_tokens);
    span.record(
        "letcode.prelude.estimated_tokens",
        budget.estimated_prelude_tokens,
    );
    span.record(
        "letcode.protected.estimated_tokens",
        budget.estimated_protected_tokens,
    );
    span.record(
        "letcode.retained_history.estimated_tokens",
        budget.estimated_retained_history_tokens,
    );
    span.record(
        "letcode.tools.estimated_tokens",
        budget.estimated_tools_tokens,
    );
    span.record(
        "letcode.evidence.estimated_tokens",
        budget.estimated_evidence_tokens,
    );
    span.record(
        "letcode.history.original_items",
        as_u64(budget.original_history_items),
    );
    span.record(
        "letcode.history.retained_items",
        as_u64(budget.retained_history_items),
    );
    span.record(
        "letcode.history.dropped_items",
        as_u64(budget.dropped_history_items),
    );
    span.record(
        "letcode.evidence.selected_items",
        as_u64(budget.selected_evidence_items),
    );
    span.record(
        "letcode.evidence.dropped_items",
        as_u64(budget.dropped_evidence_items),
    );
    span.record("letcode.request.truncated", budget.truncated);
    span.record("gen_ai.usage.input_tokens", budget.estimated_request_tokens);

    span.record(
        "langfuse.observation.metadata.context_window_tokens",
        budget.context_window_tokens,
    );
    span.record(
        "langfuse.observation.metadata.input_budget_tokens",
        budget.input_budget_tokens,
    );
    span.record(
        "langfuse.observation.metadata.request_estimated_tokens",
        budget.estimated_request_tokens,
    );
    span.record(
        "langfuse.observation.metadata.prelude_estimated_tokens",
        budget.estimated_prelude_tokens,
    );
    span.record(
        "langfuse.observation.metadata.protected_estimated_tokens",
        budget.estimated_protected_tokens,
    );
    span.record(
        "langfuse.observation.metadata.retained_history_estimated_tokens",
        budget.estimated_retained_history_tokens,
    );
    span.record(
        "langfuse.observation.metadata.tools_estimated_tokens",
        budget.estimated_tools_tokens,
    );
    span.record(
        "langfuse.observation.metadata.evidence_estimated_tokens",
        budget.estimated_evidence_tokens,
    );
    span.record(
        "langfuse.observation.metadata.original_history_items",
        as_u64(budget.original_history_items),
    );
    span.record(
        "langfuse.observation.metadata.retained_history_items",
        as_u64(budget.retained_history_items),
    );
    span.record(
        "langfuse.observation.metadata.dropped_history_items",
        as_u64(budget.dropped_history_items),
    );
    span.record(
        "langfuse.observation.metadata.selected_evidence_items",
        as_u64(budget.selected_evidence_items),
    );
    span.record(
        "langfuse.observation.metadata.dropped_evidence_items",
        as_u64(budget.dropped_evidence_items),
    );
    span.record(
        "langfuse.observation.metadata.request_truncated",
        budget.truncated,
    );
}

pub fn record_llm_prompt_plan(span: &Span, prompt_plan: &PromptPlan) {
    span.record(
        "letcode.prompt.segments",
        as_u64(prompt_plan.segments.len()),
    );
    span.record(
        "letcode.prompt.contributors",
        as_u64(prompt_plan.contributors.len()),
    );
    span.record(
        "langfuse.observation.metadata.prompt_segment_count",
        as_u64(prompt_plan.segments.len()),
    );
    span.record(
        "langfuse.observation.metadata.prompt_contributor_count",
        as_u64(prompt_plan.contributors.len()),
    );
    if let Some(prefix_hash) = prompt_plan.stable_prefix_hash() {
        span.record(
            "langfuse.observation.metadata.prompt_stable_prefix_hash",
            prefix_hash,
        );
    }
}

pub fn record_llm_cache_metadata(span: &Span, cache: &CacheUsageReport, prompt_plan: &PromptPlan) {
    let metadata =
        cache_metadata_projection(cache, prompt_plan.token_report().first_volatile_index);
    span.record(
        "langfuse.observation.metadata.cache_configured",
        metadata.configured,
    );
    span.record(
        "langfuse.observation.metadata.cache_hint_serialized",
        metadata.hint_serialized,
    );
    if let Some(retention) = metadata.retention_sent.as_deref() {
        span.record(
            "langfuse.observation.metadata.cache_retention_sent",
            retention,
        );
    }
    span.record(
        "langfuse.observation.metadata.cache_stable_prefix_segments",
        metadata.stable_prefix_segments,
    );
    span.record(
        "langfuse.observation.metadata.cache_has_stable_prefix",
        metadata.stable_prefix_segments > 0,
    );
    if let Some(first_volatile_index) = metadata.first_volatile_index {
        span.record(
            "langfuse.observation.metadata.cache_first_volatile_index",
            first_volatile_index,
        );
    }
    span.record(
        "langfuse.observation.metadata.cache_stable_prompt_tokens",
        metadata.stable_prompt_tokens,
    );
    span.record(
        "langfuse.observation.metadata.cache_volatile_prompt_tokens",
        metadata.volatile_prompt_tokens,
    );
    span.record(
        "langfuse.observation.metadata.cache_cacheable_prefix_tokens",
        metadata.cacheable_prefix_tokens,
    );
    span.record(
        "langfuse.observation.metadata.cache_stable_after_boundary_tokens",
        metadata.stable_after_boundary_tokens,
    );
    if let Some(fingerprint) = metadata.local_prefix_fingerprint.as_deref() {
        span.record(
            "langfuse.observation.metadata.cache_local_prefix_fingerprint",
            fingerprint,
        );
    }
    if let Some(routing_key) = metadata.routing_key.as_deref() {
        span.record(
            "langfuse.observation.metadata.cache_routing_key",
            routing_key,
        );
    }
    if let Some(actual_cached_tokens) = metadata.actual_cached_tokens {
        span.record(
            "langfuse.observation.metadata.cache_actual_cached_tokens",
            actual_cached_tokens,
        );
    }
}

pub fn record_llm_usage(
    span: &Span,
    input_tokens: u64,
    output_tokens: u64,
    cached_tokens: u64,
    total_tokens: u64,
    cache_report: &CacheUsageReport,
) {
    span.record("gen_ai.usage.input_tokens", input_tokens);
    span.record("gen_ai.usage.output_tokens", output_tokens);
    span.record("gen_ai.usage.cached_tokens", cached_tokens);
    span.record("gen_ai.usage.total_tokens", total_tokens);
    let usage_details =
        safe_usage_details_json(input_tokens, output_tokens, cached_tokens, total_tokens);
    span.record("langfuse.observation.usage_details", usage_details.as_str());
    let cache_report = cache_report.with_actual_cached_tokens(cached_tokens);
    record_llm_cache_metadata_actual_usage(span, &cache_report);
}

fn record_llm_cache_metadata_actual_usage(span: &Span, cache: &CacheUsageReport) {
    if let Some(actual_cached_tokens) = cache.actual_cached_tokens {
        span.record(
            "langfuse.observation.metadata.cache_actual_cached_tokens",
            actual_cached_tokens,
        );
    }
}

pub fn finish_llm_iteration_span(
    span: &Span,
    output_chars: usize,
    tool_call_count: usize,
    response_items: usize,
    finish_reasons: Option<&str>,
) {
    span.record("letcode.output.chars", as_u64(output_chars));
    span.record("letcode.tool_call.count", as_u64(tool_call_count));
    span.record("letcode.response.items", as_u64(response_items));
    span.record(
        "langfuse.observation.metadata.output_chars",
        as_u64(output_chars),
    );
    span.record(
        "langfuse.observation.metadata.tool_call_count",
        as_u64(tool_call_count),
    );
    span.record(
        "langfuse.observation.metadata.response_items",
        as_u64(response_items),
    );
    if let Some(finish_reasons) = finish_reasons {
        span.record("gen_ai.response.finish_reasons", finish_reasons);
    }
    let safe_output = safe_iteration_output_json(
        output_chars,
        tool_call_count,
        response_items,
        finish_reasons,
    );
    span.record("langfuse.observation.output", safe_output.as_str());
    record_ok(span);
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

fn safe_turn_input_json(
    user_input_chars: usize,
    history_len: usize,
    max_iterations: Option<usize>,
    max_tool_calls: Option<usize>,
) -> String {
    json!({
        "user_input_chars": as_u64(user_input_chars),
        "history_items": as_u64(history_len),
        "max_iterations": max_iterations.map(as_u64),
        "max_tool_calls": max_tool_calls.map(as_u64),
    })
    .to_string()
}

fn safe_turn_output_json(
    output_chars: usize,
    tool_call_count: usize,
    continuation_count: usize,
    history_len: usize,
    status: &'static str,
) -> String {
    json!({
        "status": status,
        "output_chars": as_u64(output_chars),
        "tool_call_count": as_u64(tool_call_count),
        "continuation_count": as_u64(continuation_count),
        "history_items": as_u64(history_len),
    })
    .to_string()
}

fn safe_iteration_input_json(
    protocol: &'static str,
    iteration: usize,
    history_len: usize,
    tool_call_count: usize,
    tool_definitions_count: usize,
) -> String {
    json!({
        "protocol": protocol,
        "iteration": as_u64(iteration),
        "history_items": as_u64(history_len),
        "tool_call_count_before": as_u64(tool_call_count),
        "tool_definitions_count": as_u64(tool_definitions_count),
    })
    .to_string()
}

fn safe_usage_details_json(
    input_tokens: u64,
    output_tokens: u64,
    cached_tokens: u64,
    total_tokens: u64,
) -> String {
    json!({
        "input": input_tokens,
        "output": output_tokens,
        "cache_read_input_tokens": cached_tokens,
        "total": total_tokens,
    })
    .to_string()
}

fn safe_iteration_output_json(
    output_chars: usize,
    tool_call_count: usize,
    response_items: usize,
    finish_reasons: Option<&str>,
) -> String {
    json!({
        "output_chars": as_u64(output_chars),
        "tool_call_count": as_u64(tool_call_count),
        "response_items": as_u64(response_items),
        "finish_reasons": finish_reasons,
    })
    .to_string()
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
