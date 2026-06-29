use anyhow::Result;
use tracing::{Span, field};

use crate::agent::{ToolExecutionRecord, ToolExecutionStatus};

pub const TARGET: &str = "letcode::langfuse";

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
        "langfuse.trace.name" = %format!("letcode turn {turn_id}"),
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
    match result {
        Ok(output) => {
            span.record("letcode.output.chars", as_u64(output.chars().count()));
            record_ok(span);
        }
        Err(_) => record_error(span, "llm_turn_error"),
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
        "letcode.request.truncated" = field::Empty,
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

pub fn record_llm_request_budget(
    span: &Span,
    estimated_request_tokens: u64,
    context_window_tokens: u64,
    truncated: bool,
) {
    span.record("letcode.request.estimated_tokens", estimated_request_tokens);
    span.record("letcode.context_window.tokens", context_window_tokens);
    span.record("letcode.request.truncated", truncated);
    span.record("gen_ai.usage.input_tokens", estimated_request_tokens);
}

pub fn record_llm_usage(
    span: &Span,
    input_tokens: u64,
    output_tokens: u64,
    cached_tokens: u64,
    total_tokens: u64,
) {
    span.record("gen_ai.usage.input_tokens", input_tokens);
    span.record("gen_ai.usage.output_tokens", output_tokens);
    span.record("gen_ai.usage.cached_tokens", cached_tokens);
    span.record("gen_ai.usage.total_tokens", total_tokens);
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
    if let Some(finish_reasons) = finish_reasons {
        span.record("gen_ai.response.finish_reasons", finish_reasons);
    }
    record_ok(span);
}

pub fn tool_span(turn_id: u64, tool_name: &str, call_id: &str, args_json_len: usize) -> Span {
    tracing::info_span!(
        target: TARGET,
        "tool.call",
        "otel.name" = %format!("tool {tool_name}"),
        "otel.kind" = "internal",
        "letcode.turn.id" = turn_id,
        "tool.name" = %tool_name,
        "tool.call_id" = %call_id,
        "tool.args_json.bytes" = as_u64(args_json_len),
        "tool.permission_class" = field::Empty,
        "tool.directive" = field::Empty,
        "tool.execution.status" = field::Empty,
        "tool.execution.rejection" = field::Empty,
        "tool.output.ok" = field::Empty,
        "tool.output.has_data" = field::Empty,
        "tool.error.recoverable" = field::Empty,
        "otel.status_code" = field::Empty,
        "otel.status_description" = field::Empty,
        "error.type" = field::Empty,
    )
}

pub fn finish_tool_span(span: &Span, result: &Result<ToolExecutionRecord>) {
    match result {
        Ok(record) => {
            span.record("tool.permission_class", record.permission_class.as_str());
            span.record("tool.directive", record.directive.as_str());
            span.record("tool.execution.status", record.status.as_str());
            if let Some(rejection) = record.rejection {
                span.record("tool.execution.rejection", rejection.as_str());
            }
            span.record("tool.output.ok", record.output.ok);
            span.record("tool.output.has_data", record.output.data.is_some());
            if let Some(error) = &record.output.error {
                span.record("tool.error.recoverable", error.recoverable);
            }

            if record.status == ToolExecutionStatus::Executed && !record.output.ok {
                record_error(span, "tool_output_error");
            } else {
                record_ok(span);
            }
        }
        Err(_) => record_error(span, "tool_execution_error"),
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

fn as_u64(value: usize) -> u64 {
    value.try_into().unwrap_or(u64::MAX)
}
