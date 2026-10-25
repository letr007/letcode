use std::collections::HashSet;

use anyhow::Result;
use async_trait::async_trait;
use serde_json::{Map, Value, json};

use super::args::{optional_bool, optional_string, optional_usize, required_string};
use super::fold_artifact::{SEARCH_ARTIFACT_DIR, write_artifact};
use super::paths::{display_workspace_relative, existing_workspace_path};
use super::{
    COMMAND_TIMEOUT_SECS, ToolExecutionContext, ToolHandler, ToolParallelism, ToolRegistry,
};

const SEARCH_FOLD_THRESHOLD_BYTES: usize = 32 * 1024;
const SEARCH_PREVIEW_MATCHES: usize = 5;
const SEARCH_TEXT_PREVIEW_CHARS: usize = 120;
const SEARCH_MAX_CONTEXT_LINES: usize = 20;

pub(super) fn register(registry: &mut ToolRegistry) {
    registry.register(RgTool);
}

struct RgTool;

#[async_trait]
impl ToolHandler for RgTool {
    fn name(&self) -> &'static str {
        "search__rg"
    }

    fn description(&self) -> &'static str {
        "Search text in the current workspace using ripgrep. Returns matching lines with optional context, matching file paths, or per-file match counts."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Ripgrep regular expression"
                },
                "path": {
                    "type": ["string", "null"],
                    "description": "Directory or file path relative to the workspace. Null defaults to ."
                },
                "include": {
                    "type": ["string", "null"],
                    "description": "Glob include pattern, e.g. *.rs. Null applies no glob filter"
                },
                "case_sensitive": {
                    "type": ["boolean", "null"],
                    "description": "Whether the search is case-sensitive. Null defaults to false"
                },
                "context": {
                    "type": ["integer", "null"],
                    "description": "Context lines before and after each match, capped at 20. Null defaults to 0"
                },
                "mode": {
                    "type": ["string", "null"],
                    "description": "matches returns matching lines, files returns every matching file path, count returns per-file counts. Files and counts stay complete and fold to a local file when large. Null defaults to matches"
                },
                "max_results": {
                    "type": ["integer", "null"],
                    "description": "Maximum matching lines in matches mode. Null defaults to 100, capped at 1000"
                }
            },
            "required": [
                "pattern",
                "path",
                "include",
                "case_sensitive",
                "context",
                "mode",
                "max_results"
            ],
            "additionalProperties": false
        })
    }

    fn parallelism(&self) -> ToolParallelism {
        ToolParallelism::Parallel
    }

    async fn execute(&self, args: Value) -> Result<Value> {
        rg(args, ToolExecutionContext::default()).await
    }

    async fn execute_with_context(
        &self,
        args: Value,
        context: ToolExecutionContext,
    ) -> Result<Value> {
        rg(args, context).await
    }
}

async fn rg(args: Value, context: ToolExecutionContext) -> Result<Value> {
    let pattern = required_string(&args, "pattern")?;
    let raw_path = optional_string(&args, "path").unwrap_or(".");
    let path = existing_workspace_path(raw_path, &context)?;
    let include = optional_string(&args, "include");
    let case_sensitive = optional_bool(&args, "case_sensitive").unwrap_or(false);
    let context_lines = optional_usize(&args, "context")
        .unwrap_or(0)
        .min(SEARCH_MAX_CONTEXT_LINES);
    let max_results = optional_usize(&args, "max_results")
        .unwrap_or(100)
        .clamp(1, 1000);
    let mode = optional_string(&args, "mode").unwrap_or("matches");
    if !matches!(mode, "matches" | "files" | "count") {
        anyhow::bail!("unsupported mode {mode:?}; expected matches, files or count");
    }

    let mut command_args = Vec::new();
    match mode {
        "files" => {
            command_args.push("--files-with-matches".to_string());
            command_args.push("--null".to_string());
        }
        "count" => {
            command_args.push("--count".to_string());
            command_args.push("--null".to_string());
        }
        _ => {
            // `--json` emits one structured NDJSON object per event, so matches
            // are read from `data.line_number` / `data.submatches[].start`
            // below. Text output (`path:line:column:text`) would split on ':'
            // and break for paths that contain a colon, e.g. `C:\...` drive
            // prefixes on Windows.
            command_args.push("--json".to_string());
            if context_lines > 0 {
                command_args.push("--context".to_string());
                command_args.push(context_lines.to_string());
            }
        }
    }

    if !case_sensitive {
        command_args.push("--ignore-case".to_string());
    }

    if let Some(include) = include {
        command_args.push("--glob".to_string());
        command_args.push(include.to_string());
    }

    let mut target_path = display_workspace_relative(&path)?;
    if target_path.is_empty() {
        target_path = ".".to_string();
    }

    command_args.push(pattern.to_string());
    command_args.push(target_path.clone());

    let output =
        super::command::run_workspace_command("rg", &command_args, COMMAND_TIMEOUT_SECS).await?;
    let stdout = output
        .get("stdout")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let mut truncated = output
        .get("stdout_truncated")
        .and_then(Value::as_bool)
        .unwrap_or(false);

    let payload = match mode {
        "files" => {
            let paths = parse_file_paths(stdout);
            present_file_paths(paths, truncated).await?
        }
        "count" => {
            let (counts, total) = parse_counts(stdout);
            present_counts(counts, total, truncated).await?
        }
        _ => {
            let (matches, hit_limit) = parse_match_events(stdout, max_results);
            truncated |= hit_limit;
            present_search_matches(matches, truncated).await?
        }
    };

    let mut data = json!({
        "pattern": pattern,
        "path": target_path,
        "status": output.get("status").cloned().unwrap_or(Value::Null),
        "success": output.get("success").cloned().unwrap_or(Value::Bool(false)),
        "stderr": output.get("stderr").cloned().unwrap_or(Value::String(String::new())),
    });
    for (key, value) in payload {
        data[key] = value;
    }
    Ok(data)
}

fn parse_match_events(stdout: &str, max_results: usize) -> (Vec<Value>, bool) {
    let mut items = Vec::new();
    let mut match_count = 0usize;
    let mut truncated = false;

    for line in stdout.lines() {
        let Ok(event) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        let is_context = match event.get("type").and_then(Value::as_str) {
            Some("match") => false,
            Some("context") => true,
            _ => continue,
        };
        let data = event.get("data");
        let Some(path) = data
            .and_then(|data| data.get("path"))
            .and_then(|path| path.get("text"))
            .and_then(Value::as_str)
        else {
            continue;
        };
        let Some(line_number) = data
            .and_then(|data| data.get("line_number"))
            .and_then(Value::as_u64)
        else {
            continue;
        };
        let text = data
            .and_then(|data| data.get("lines"))
            .and_then(|lines| lines.get("text"))
            .and_then(Value::as_str)
            .unwrap_or_default()
            .trim_end_matches('\n')
            .to_string();

        if is_context {
            items.push(json!({
                "path": path,
                "line": line_number,
                "text": text,
                "is_context": true,
            }));
            continue;
        }

        if match_count >= max_results {
            truncated = true;
            break;
        }
        match_count += 1;

        // rg reports byte offsets (0-based); columns are exposed 1-based.
        let column = data
            .and_then(|data| data.get("submatches"))
            .and_then(Value::as_array)
            .and_then(|submatches| submatches.first())
            .and_then(|submatch| submatch.get("start"))
            .and_then(Value::as_u64)
            .unwrap_or(0)
            .saturating_add(1);

        items.push(json!({
            "path": path,
            "line": line_number,
            "column": column,
            "text": text,
        }));
    }

    (items, truncated)
}

fn parse_file_paths(stdout: &str) -> Vec<String> {
    stdout
        .split('\0')
        .filter(|entry| !entry.is_empty())
        .map(str::to_string)
        .collect()
}

fn parse_counts(stdout: &str) -> (Vec<Value>, u64) {
    let mut counts = Vec::new();
    let mut total = 0u64;

    let mut segments = stdout.split('\0');
    let mut path = match segments.next() {
        Some(path) if !path.is_empty() => path,
        _ => return (counts, total),
    };

    for segment in segments {
        // rg terminates the count with a newline and starts the next path after it.
        let (count_text, next_path) = match segment.split_once('\n') {
            Some((count, next)) => (count, Some(next)),
            None => (segment, None),
        };
        let count = count_text.trim().parse::<u64>().unwrap_or(0);
        counts.push(json!({"path": path, "count": count}));
        total += count;
        match next_path {
            Some(next) if !next.is_empty() => path = next,
            _ => break,
        }
    }

    (counts, total)
}

fn match_line_text(m: &Value) -> String {
    let path = m.get("path").and_then(Value::as_str).unwrap_or("?");
    let line = m.get("line").and_then(Value::as_u64).unwrap_or(0);
    let text = m.get("text").and_then(Value::as_str).unwrap_or("");
    match m.get("column").and_then(Value::as_u64) {
        Some(column) => format!("{path}:{line}:{column}: {text}"),
        None => format!("{path}:{line}: {text}"),
    }
}

fn count_line_text(count: &Value) -> String {
    let path = count.get("path").and_then(Value::as_str).unwrap_or("?");
    let count = count.get("count").and_then(Value::as_u64).unwrap_or(0);
    format!("{path}:{count}")
}

fn truncate_match_text(mut m: Value) -> Value {
    if let Some(text) = m.get("text").and_then(Value::as_str) {
        let preview: String = text.chars().take(SEARCH_TEXT_PREVIEW_CHARS).collect();
        if preview.chars().count() < text.chars().count() {
            m["text"] = json!(preview);
            m["text_truncated"] = json!(true);
        }
    }
    m
}

struct Folded {
    items: Vec<Value>,
    truncated: bool,
    local_path: Option<String>,
}

/// Decide how the result rows are surfaced. Small result sets are returned
/// inline in full; large ones are folded to a local artifact with a short
/// preview plus `local_path`, so the model can decide whether and where to read
/// the full list on demand.
async fn fold_items(
    items: Vec<Value>,
    outer_truncated: bool,
    artifact_line: impl Fn(&Value) -> String,
) -> Result<Folded> {
    if serde_json::to_string(&items)?.len() <= SEARCH_FOLD_THRESHOLD_BYTES {
        return Ok(Folded {
            items,
            truncated: outer_truncated,
            local_path: None,
        });
    }

    let body = items
        .iter()
        .map(artifact_line)
        .collect::<Vec<_>>()
        .join("\n");
    let local_path = write_artifact(SEARCH_ARTIFACT_DIR, body.as_bytes(), "txt").await?;
    let preview = items
        .iter()
        .take(SEARCH_PREVIEW_MATCHES)
        .cloned()
        .map(truncate_match_text)
        .collect();

    Ok(Folded {
        items: preview,
        truncated: true,
        local_path: Some(local_path),
    })
}

async fn present_search_matches(
    matches: Vec<Value>,
    outer_truncated: bool,
) -> Result<Map<String, Value>> {
    let total_matches = matches
        .iter()
        .filter(|m| m.get("is_context").is_none())
        .count();
    let files: HashSet<&str> = matches
        .iter()
        .filter(|m| m.get("is_context").is_none())
        .filter_map(|m| m.get("path").and_then(Value::as_str))
        .collect();

    let mut map = Map::new();
    map.insert("total_matches".into(), json!(total_matches));
    map.insert("files".into(), json!(files.len()));

    if matches.is_empty() {
        map.insert("matches".into(), json!([]));
        map.insert("truncated".into(), json!(outer_truncated));
        return Ok(map);
    }

    let folded = fold_items(matches, outer_truncated, match_line_text).await?;
    map.insert("matches".into(), json!(folded.items));
    if let Some(local_path) = folded.local_path {
        map.insert("folded".into(), json!(true));
        map.insert("local_path".into(), json!(local_path));
    }
    map.insert("truncated".into(), json!(folded.truncated));
    Ok(map)
}

async fn present_file_paths(
    paths: Vec<String>,
    outer_truncated: bool,
) -> Result<Map<String, Value>> {
    let total_files = paths.len();
    let items = paths
        .into_iter()
        .map(|path| json!(path))
        .collect::<Vec<_>>();
    let folded = fold_items(items, outer_truncated, |item| {
        item.as_str().unwrap_or_default().to_string()
    })
    .await?;

    let mut map = Map::new();
    map.insert("files".into(), json!(total_files));
    map.insert("file_paths".into(), json!(folded.items));
    if let Some(local_path) = folded.local_path {
        map.insert("folded".into(), json!(true));
        map.insert("local_path".into(), json!(local_path));
    }
    map.insert("truncated".into(), json!(folded.truncated));
    Ok(map)
}

async fn present_counts(
    counts: Vec<Value>,
    total_matches: u64,
    outer_truncated: bool,
) -> Result<Map<String, Value>> {
    let total_files = counts.len();
    let folded = fold_items(counts, outer_truncated, count_line_text).await?;

    let mut map = Map::new();
    map.insert("files".into(), json!(total_files));
    map.insert("total_matches".into(), json!(total_matches));
    map.insert("counts".into(), json!(folded.items));
    if let Some(local_path) = folded.local_path {
        map.insert("folded".into(), json!(true));
        map.insert("local_path".into(), json!(local_path));
    }
    map.insert("truncated".into(), json!(folded.truncated));
    Ok(map)
}

#[cfg(test)]
#[path = "../../tests/internal/tool/search/tests.rs"]
mod tests;
