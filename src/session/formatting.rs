use serde_json::Value;

use crate::agent::is_subagent_tool_name;
use crate::tool::ToolResult;
use crate::tool_names;

pub(super) fn output_summary(output: &ToolResult) -> Option<String> {
    if let Some(error) = &output.error {
        return Some(error.message.clone());
    }

    let data = output.data.as_ref()?;
    Some(match output.tool.as_str() {
        tool_names::TOOL_UTIL_ECHO => summarize_echo(data),
        tool_names::TOOL_FS_LIST => summarize_array_count(data, "entries", "entries"),
        tool_names::TOOL_FS_READ => summarize_read_file(data),
        tool_names::TOOL_FS_WRITE => summarize_bytes(data, "bytes_written", "wrote"),
        tool_names::TOOL_FS_APPEND => summarize_bytes(data, "bytes_appended", "appended"),
        tool_names::TOOL_FS_MKDIR => summarize_path_action(data, "created"),
        tool_names::TOOL_SEARCH_RG => summarize_search(data),
        tool_names::TOOL_WEB_FETCH => summarize_web_fetch(data),
        tool_names::TOOL_SHELL_EXEC
        | tool_names::TOOL_GIT_STATUS
        | tool_names::TOOL_GIT_DIFF
        | tool_names::TOOL_GIT_LOG => summarize_command(data),
        tool_names::TOOL_EDIT_APPLY_PATCH => summarize_apply_patch(data),
        tool_names::TOOL_CODE_AST_SEARCH => summarize_array_count(data, "matches", "matches"),
        tool_names::TOOL_CODE_AST_REPLACE_PREVIEW => {
            summarize_array_count(data, "replacements", "replacements")
        }
        tool_names::TOOL_WORKFLOW_TODOS => summarize_todos(data),
        tool_names::TOOL_WORKFLOW_AUTO_CONTINUE => summarize_auto_continue(data),
        name if is_subagent_tool_name(name) => summarize_subagent_tool(data),
        _ => summarize_generic(data),
    })
}

fn summarize_subagent_tool(data: &Value) -> String {
    let agent_name = data
        .get("agent_name")
        .and_then(Value::as_str)
        .unwrap_or("subagent");
    let status = data
        .get("status")
        .and_then(Value::as_str)
        .unwrap_or("completed");
    let child = data
        .get("child_session_id")
        .and_then(Value::as_str)
        .map(|id| id.get(..12).unwrap_or(id))
        .unwrap_or("child");
    let flags = summarize_subagent_flags(data);
    if flags.is_empty() {
        format!("{agent_name} {status} · {child}")
    } else {
        format!("{agent_name} {status} · {} · {child}", flags.join("/"))
    }
}

fn summarize_subagent_flags(data: &Value) -> Vec<&'static str> {
    let mut flags = Vec::new();
    if data.get("active").and_then(Value::as_bool) == Some(true) {
        flags.push("active");
    }
    if data.get("unreconciled").and_then(Value::as_bool) == Some(true) {
        flags.push("unreconciled");
    }
    if data.get("reconciled").and_then(Value::as_bool) == Some(true) {
        flags.push("reconciled");
    }
    if data.get("reusable").and_then(Value::as_bool) == Some(true) {
        flags.push("reusable");
    }
    if data
        .get("structured_result")
        .and_then(|value| value.get("malformed"))
        .and_then(Value::as_bool)
        == Some(true)
    {
        flags.push("malformed");
    }
    flags
}

pub(super) fn compact_subagent_summary(summary: &str) -> String {
    let single_line = summary.split_whitespace().collect::<Vec<_>>().join(" ");
    if single_line.chars().count() <= 160 {
        return single_line;
    }
    let mut truncated = single_line.chars().take(160).collect::<String>();
    truncated.push('…');
    truncated
}

fn summarize_todos(data: &Value) -> String {
    let count = data
        .get("items")
        .and_then(Value::as_array)
        .map(Vec::len)
        .unwrap_or(0);
    format!("updated {count} todos")
}

fn summarize_auto_continue(data: &Value) -> String {
    let enabled = data
        .get("enabled")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if enabled {
        "enabled auto-continue".into()
    } else {
        "disabled auto-continue".into()
    }
}

fn summarize_echo(data: &Value) -> String {
    let chars = data
        .get("result")
        .and_then(Value::as_str)
        .map(str::chars)
        .map(Iterator::count)
        .unwrap_or(0);
    format!("returned {chars} chars")
}

pub(super) fn summarize_search(data: &Value) -> String {
    // Prefer the explicit aggregate; fall back to the inline array length for
    // older/legacy payloads that predate `total_matches`.
    let total = data
        .get("total_matches")
        .and_then(Value::as_u64)
        .or_else(|| {
            data.get("matches")
                .and_then(Value::as_array)
                .map(|matches| matches.len() as u64)
        })
        .unwrap_or(0);
    let files = data.get("files").and_then(Value::as_u64).unwrap_or(0);
    let mut parts = vec![format!("{total} matches")];
    if files > 0 {
        parts.push(format!("{files} files"));
    }
    let text = parts.join(" · ");
    if data.get("folded").and_then(Value::as_bool).unwrap_or(false) {
        format!("{text} · folded")
    } else {
        text
    }
}

fn summarize_array_count(data: &Value, key: &str, label: &str) -> String {
    let count = data
        .get(key)
        .and_then(Value::as_array)
        .map(Vec::len)
        .unwrap_or(0);
    let truncated = data
        .get("truncated")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if truncated {
        format!("{count} {label} shown · truncated")
    } else {
        format!("{count} {label}")
    }
}

fn summarize_web_fetch(data: &Value) -> String {
    let status = data.get("status").and_then(Value::as_u64).unwrap_or(0);
    let bytes = data
        .get("content_bytes")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let redirects = data
        .get("redirects")
        .and_then(Value::as_array)
        .map(Vec::len)
        .unwrap_or(0);
    let truncated = data
        .get("truncated")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let redirect_suffix = if redirects > 0 {
        format!(" · {redirects} redirects")
    } else {
        String::new()
    };
    let truncation_suffix = if truncated { " · truncated" } else { "" };
    format!("HTTP {status} · {bytes} bytes{redirect_suffix}{truncation_suffix}")
}

fn summarize_read_file(data: &Value) -> String {
    let path = data.get("path").and_then(Value::as_str).unwrap_or("file");
    if data.get("kind").and_then(Value::as_str) == Some("image") {
        let bytes = data.get("bytes").and_then(Value::as_u64).unwrap_or(0);
        let mime = data.get("mime").and_then(Value::as_str).unwrap_or("image");
        return format!("read image {path} ({mime}, {bytes} bytes)");
    }
    let lines = data.get("lines_read").and_then(Value::as_u64).unwrap_or(0);
    let start = data.get("start_line").and_then(Value::as_u64);
    let end = data.get("end_line").and_then(Value::as_u64);
    let suffix = if data
        .get("truncated")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        " · has more"
    } else {
        ""
    };

    match (start, end) {
        (Some(start), Some(end)) => format!("read {path}:{start}-{end} ({lines} lines){suffix}"),
        _ => format!("read {path} ({lines} lines){suffix}"),
    }
}

fn summarize_bytes(data: &Value, key: &str, verb: &str) -> String {
    let path = data.get("path").and_then(Value::as_str).unwrap_or("file");
    let bytes = data.get(key).and_then(Value::as_u64).unwrap_or(0);
    format!("{verb} {bytes} bytes to {path}")
}

fn summarize_path_action(data: &Value, action: &str) -> String {
    let path = data.get("path").and_then(Value::as_str).unwrap_or("path");
    format!("{action} {path}")
}

fn summarize_command(data: &Value) -> String {
    if let Some(error) = data.get("error").and_then(Value::as_str) {
        return error.to_string();
    }

    let status = data
        .get("status")
        .and_then(Value::as_i64)
        .map(|status| format!("exit {status}"))
        .unwrap_or_else(|| "completed".to_string());
    let stdout = output_line_count(data, "stdout", "stdout_truncated");
    let stderr = output_line_count(data, "stderr", "stderr_truncated");
    let mut parts = vec![status];
    if let Some(stdout) = stdout {
        parts.push(format!("stdout {stdout}{}", folded_suffix(data, "stdout")));
    }
    if let Some(stderr) = stderr {
        parts.push(format!("stderr {stderr}{}", folded_suffix(data, "stderr")));
    }
    parts.join(" · ")
}

fn folded_suffix(data: &Value, label: &str) -> &'static str {
    let folded = data
        .get(format!("{label}_folded"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if folded {
        " · folded"
    } else {
        ""
    }
}

fn output_line_count(data: &Value, key: &str, truncated_key: &str) -> Option<String> {
    let text = data.get(key).and_then(Value::as_str)?;
    if text.trim().is_empty() {
        return None;
    }
    let count = text.lines().count().max(1);
    let suffix = if data
        .get(truncated_key)
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        "+"
    } else {
        ""
    };
    Some(format!("{count}{suffix} lines"))
}

fn summarize_apply_patch(data: &Value) -> String {
    let files = data
        .get("files_changed")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let edits = data
        .get("edits_applied")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    format!("patched {files} files · {edits} edits")
}

fn summarize_generic(data: &Value) -> String {
    match data {
        Value::Array(items) => format!("{} items", items.len()),
        Value::Object(fields) => format!("{} fields", fields.len()),
        Value::String(text) => format!("returned {} chars", text.chars().count()),
        Value::Null => "completed".into(),
        _ => "completed".into(),
    }
}

pub(super) fn output_json(output: &ToolResult) -> Value {
    serde_json::to_value(output)
        .unwrap_or_else(|_| Value::String("<unserializable tool output>".into()))
}
