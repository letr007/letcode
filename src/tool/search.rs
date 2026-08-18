use std::collections::HashSet;

use anyhow::Result;
use async_trait::async_trait;
use serde_json::{Value, json};

use super::fold_artifact::{SEARCH_ARTIFACT_DIR, write_artifact};
use super::{
    COMMAND_TIMEOUT_SECS, ToolExecutionContext, ToolHandler, ToolParallelism, ToolRegistry,
    display_workspace_relative, existing_workspace_path, optional_bool, optional_string,
    optional_usize, required_string,
};

const SEARCH_FOLD_THRESHOLD_BYTES: usize = 32 * 1024;
const SEARCH_PREVIEW_MATCHES: usize = 5;
const SEARCH_TEXT_PREVIEW_CHARS: usize = 120;

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
        "Search text in the current workspace using ripgrep. Returns structured matches."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Ripgrep search pattern"
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file path relative to the workspace. Defaults to ."
                },
                "include": {
                    "type": "string",
                    "description": "Optional glob include pattern, e.g. *.rs"
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether the search is case-sensitive. Defaults to false"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum matches to return. Defaults to 100, capped at 1000"
                }
            },
            "required": ["pattern", "path", "include", "case_sensitive", "max_results"],
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
    let max_results = optional_usize(&args, "max_results")
        .unwrap_or(100)
        .clamp(1, 1000);

    // `--json` emits one structured NDJSON object per event, so matches are
    // read from `data.line_number` / `data.submatches[].start` below. Text
    // output (`path:line:column:text`) would split on ':' and break for paths
    // that contain a colon, e.g. `C:\...` drive prefixes on Windows.
    let mut command_args = vec!["--json".to_string()];

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
    command_args.push(target_path);

    let output =
        super::command::run_workspace_command("rg", &command_args, COMMAND_TIMEOUT_SECS).await?;
    let stdout = output
        .get("stdout")
        .and_then(Value::as_str)
        .unwrap_or_default();

    let mut matches = Vec::new();
    let mut truncated = output
        .get("stdout_truncated")
        .and_then(Value::as_bool)
        .unwrap_or(false);

    for line in stdout.lines() {
        if matches.len() >= max_results {
            truncated = true;
            break;
        }

        let Ok(event) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        if event.get("type").and_then(Value::as_str) != Some("match") {
            continue;
        }
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
        let Some(first_submatch) = data
            .and_then(|data| data.get("submatches"))
            .and_then(Value::as_array)
            .and_then(|submatches| submatches.first())
        else {
            continue;
        };
        // rg reports byte offsets (0-based); expose a 1-based column like the
        // previous `--column` output did.
        let column = first_submatch
            .get("start")
            .and_then(Value::as_u64)
            .unwrap_or(0)
            .saturating_add(1);
        let text = data
            .and_then(|data| data.get("lines"))
            .and_then(|lines| lines.get("text"))
            .and_then(Value::as_str)
            .unwrap_or_default()
            .trim_end_matches('\n')
            .to_string();

        matches.push(json!({
            "path": path,
            "line": line_number,
            "column": column,
            "text": text,
        }));
    }

    let mut data = json!({
        "pattern": pattern,
        "path": display_workspace_relative(&path)?,
        "status": output.get("status").cloned().unwrap_or(Value::Null),
        "success": output.get("success").cloned().unwrap_or(Value::Bool(false)),
        "stderr": output.get("stderr").cloned().unwrap_or(Value::String(String::new())),
    });
    let payload = present_search_matches(matches, truncated).await?;
    for (key, value) in payload {
        data[key] = value;
    }
    Ok(data)
}

fn match_line_text(m: &Value) -> String {
    let path = m.get("path").and_then(Value::as_str).unwrap_or("?");
    let line = m.get("line").and_then(Value::as_u64).unwrap_or(0);
    let column = m.get("column").and_then(Value::as_u64).unwrap_or(0);
    let text = m.get("text").and_then(Value::as_str).unwrap_or("");
    format!("{path}:{line}:{column}: {text}")
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

/// Decide how the matched results are surfaced. Small result sets are returned
/// inline in full; large ones are folded to a local artifact with a short
/// preview plus `local_path`, so the model can decide whether and where to read
/// the full list on demand.
async fn present_search_matches(
    matches: Vec<Value>,
    outer_truncated: bool,
) -> Result<serde_json::Map<String, Value>> {
    let total_matches = matches.len();
    let files: HashSet<&str> = matches
        .iter()
        .filter_map(|m| m.get("path").and_then(Value::as_str))
        .collect();
    let mut map = serde_json::Map::new();
    map.insert("total_matches".into(), json!(total_matches));
    map.insert("files".into(), json!(files.len()));

    if matches.is_empty() {
        map.insert("matches".into(), json!([]));
        map.insert("truncated".into(), json!(outer_truncated));
        return Ok(map);
    }

    let inline_size = serde_json::to_string(&matches)?.len();
    if inline_size > SEARCH_FOLD_THRESHOLD_BYTES {
        let body = matches
            .iter()
            .map(match_line_text)
            .collect::<Vec<_>>()
            .join("\n");
        let local_path = write_artifact(SEARCH_ARTIFACT_DIR, body.as_bytes(), "txt").await?;
        let preview = matches
            .iter()
            .take(SEARCH_PREVIEW_MATCHES)
            .cloned()
            .map(truncate_match_text)
            .collect::<Vec<_>>();
        map.insert("matches".into(), json!(preview));
        map.insert("folded".into(), json!(true));
        map.insert("local_path".into(), json!(local_path));
        map.insert("truncated".into(), json!(true));
    } else {
        map.insert("matches".into(), json!(matches));
        map.insert("truncated".into(), json!(outer_truncated));
    }
    Ok(map)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn match_item(path: &str, line: u64, text: &str) -> Value {
        json!({"path": path, "line": line, "column": 1, "text": text})
    }

    #[tokio::test]
    async fn large_result_set_folds_to_artifact_with_preview() {
        let matches = (0..200)
            .map(|i| match_item("src/a.rs", i as u64 + 1, &"x".repeat(300)))
            .collect::<Vec<_>>();
        let map = present_search_matches(matches, false).await.unwrap();
        assert_eq!(map.get("folded").and_then(Value::as_bool), Some(true));
        assert_eq!(map.get("truncated").and_then(Value::as_bool), Some(true));
        assert_eq!(map.get("total_matches").and_then(Value::as_u64), Some(200));
        assert_eq!(map.get("files").and_then(Value::as_u64), Some(1));
        let preview = map.get("matches").and_then(Value::as_array).unwrap();
        assert_eq!(preview.len(), super::SEARCH_PREVIEW_MATCHES);
        assert!(
            preview
                .iter()
                .all(|m| m.get("text_truncated") == Some(&json!(true))),
            "long lines in preview are truncated"
        );
        let path = map.get("local_path").and_then(Value::as_str).unwrap();
        let on_disk = std::fs::read(path).unwrap();
        assert_eq!(on_disk.split(|&b| b == b'\n').count(), 200);
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn small_result_set_stays_inline() {
        let matches = vec![
            match_item("src/a.rs", 1, "hello"),
            match_item("src/b.rs", 2, "world"),
        ];
        let map = present_search_matches(matches, false).await.unwrap();
        assert_eq!(map.get("folded"), None);
        assert_eq!(map.get("local_path"), None);
        assert_eq!(map.get("truncated").and_then(Value::as_bool), Some(false));
        assert_eq!(map.get("files").and_then(Value::as_u64), Some(2));
        let matches = map.get("matches").and_then(Value::as_array).unwrap();
        assert_eq!(matches.len(), 2);
        assert!(matches.iter().all(|m| m.get("text_truncated").is_none()));
    }
}
