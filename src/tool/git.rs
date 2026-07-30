use anyhow::Result;
use async_trait::async_trait;
use serde_json::{Value, json};

use super::{
    COMMAND_TIMEOUT_SECS, ToolHandler, ToolParallelism, ToolRegistry, optional_bool,
    optional_string, optional_usize, safe_relative_path_arg,
};

pub(super) fn register(registry: &mut ToolRegistry) {
    registry.register(GitStatusTool);
    registry.register(GitDiffTool);
    registry.register(GitLogTool);
}

struct GitStatusTool;

#[async_trait]
impl ToolHandler for GitStatusTool {
    fn name(&self) -> &'static str {
        "git__status"
    }

    fn description(&self) -> &'static str {
        "Show git working tree status for the current workspace."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": false
        })
    }

    fn parallelism(&self) -> ToolParallelism {
        ToolParallelism::Parallel
    }

    async fn execute(&self, _args: Value) -> Result<Value> {
        git_status().await
    }
}

struct GitDiffTool;

#[async_trait]
impl ToolHandler for GitDiffTool {
    fn name(&self) -> &'static str {
        "git__diff"
    }

    fn description(&self) -> &'static str {
        "Show git diff for the current workspace. Supports unstaged or staged diff and optional path."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "staged": {
                    "type": "boolean",
                    "description": "Show staged diff instead of unstaged diff. Defaults to false"
                },
                "path": {
                    "type": ["string", "null"],
                    "description": "Optional workspace-relative path to limit the diff"
                }
            },
            "required": ["staged", "path"],
            "additionalProperties": false
        })
    }

    fn parallelism(&self) -> ToolParallelism {
        ToolParallelism::Parallel
    }

    async fn execute(&self, args: Value) -> Result<Value> {
        git_diff(args).await
    }
}

struct GitLogTool;

#[async_trait]
impl ToolHandler for GitLogTool {
    fn name(&self) -> &'static str {
        "git__log"
    }

    fn description(&self) -> &'static str {
        "Show recent git commits for the current workspace."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "max_count": {
                    "type": "integer",
                    "description": "Maximum number of commits. Defaults to 10, capped at 50"
                }
            },
            "required": ["max_count"],
            "additionalProperties": false
        })
    }

    fn parallelism(&self) -> ToolParallelism {
        ToolParallelism::Parallel
    }

    async fn execute(&self, args: Value) -> Result<Value> {
        git_log(args).await
    }
}

async fn git_status() -> Result<Value> {
    let args = vec![
        "status".to_string(),
        "--short".to_string(),
        "--branch".to_string(),
    ];
    super::command::run_workspace_command("git", &args, COMMAND_TIMEOUT_SECS).await
}

async fn git_diff(args: Value) -> Result<Value> {
    let staged = optional_bool(&args, "staged").unwrap_or(false);
    let mut command_args = vec!["diff".to_string()];

    if staged {
        command_args.push("--cached".to_string());
    }

    if let Some(path) = optional_string(&args, "path").filter(|path| !path.trim().is_empty()) {
        command_args.push("--".to_string());
        command_args.push(safe_relative_path_arg(path)?);
    }

    super::command::run_workspace_command("git", &command_args, COMMAND_TIMEOUT_SECS).await
}

async fn git_log(args: Value) -> Result<Value> {
    let max_count = optional_usize(&args, "max_count")
        .unwrap_or(10)
        .clamp(1, 50);
    let command_args = vec![
        "log".to_string(),
        format!("--max-count={max_count}"),
        "--oneline".to_string(),
        "--decorate".to_string(),
    ];

    super::command::run_workspace_command("git", &command_args, COMMAND_TIMEOUT_SECS).await
}
