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

    let mut command_args = vec![
        "--line-number".to_string(),
        "--column".to_string(),
        "--no-heading".to_string(),
        "--color".to_string(),
        "never".to_string(),
    ];

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

        let mut parts = line.splitn(4, ':');
        let Some(path) = parts.next() else { continue };
        let Some(line_number) = parts.next().and_then(|value| value.parse::<u64>().ok()) else {
            continue;
        };
        let Some(column) = parts.next().and_then(|value| value.parse::<u64>().ok()) else {
            continue;
        };
        let Some(text) = parts.next() else { continue };

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
