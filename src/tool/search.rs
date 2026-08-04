use anyhow::Result;
use async_trait::async_trait;
use serde_json::{Value, json};

use super::{
    COMMAND_TIMEOUT_SECS, ToolExecutionContext, ToolHandler, ToolParallelism, ToolRegistry,
    display_workspace_relative, existing_workspace_path, optional_bool, optional_string,
    optional_usize, required_string,
};

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
        let Some(line_number) = data.and_then(|data| data.get("line_number")).and_then(Value::as_u64)
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

    Ok(json!({
        "pattern": pattern,
        "path": display_workspace_relative(&path)?,
        "matches": matches,
        "truncated": truncated,
        "status": output.get("status").cloned().unwrap_or(Value::Null),
        "success": output.get("success").cloned().unwrap_or(Value::Bool(false)),
        "stderr": output.get("stderr").cloned().unwrap_or(Value::String(String::new())),
    }))
}
