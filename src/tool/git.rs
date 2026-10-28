use anyhow::{Result, bail};
use async_trait::async_trait;
use serde_json::{Value, json};

use super::args::{optional_bool, optional_string, optional_usize, required_string};
use super::paths::safe_relative_path_arg;
use super::{COMMAND_TIMEOUT_SECS, ToolHandler, ToolParallelism, ToolRegistry};

pub(super) fn register(registry: &mut ToolRegistry) {
    registry.register(GitTool);
}

struct GitTool;

#[async_trait]
impl ToolHandler for GitTool {
    fn name(&self) -> &'static str {
        "git"
    }

    fn description(&self) -> &'static str {
        "Run a read-only git query in the current workspace: status, diff, log, show, blame, or branch."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["status", "diff", "log", "show", "blame", "branch"],
                    "description": "Query to run. status shows working tree state; diff shows changes; log lists commits; show renders one commit; blame annotates a file; branch lists local branches."
                },
                "path": {
                    "type": ["string", "null"],
                    "description": "Workspace-relative file or directory to scope the query. Required for blame."
                },
                "staged": {
                    "type": ["boolean", "null"],
                    "description": "For diff: show staged changes instead of unstaged. Defaults to false."
                },
                "max_count": {
                    "type": ["integer", "null"],
                    "description": "For log: maximum commits to show. Defaults to 10, capped at 50."
                },
                "ref": {
                    "type": ["string", "null"],
                    "description": "Commit, branch, or tag to inspect for diff, log, show, and blame. show defaults to HEAD."
                }
            },
            "required": ["action", "path", "staged", "max_count", "ref"],
            "additionalProperties": false
        })
    }

    fn parallelism(&self) -> ToolParallelism {
        ToolParallelism::Parallel
    }

    async fn execute(&self, args: Value) -> Result<Value> {
        let action = required_string(&args, "action")?;
        let path = optional_string(&args, "path").filter(|path| !path.trim().is_empty());
        let staged = optional_bool(&args, "staged").unwrap_or(false);
        let max_count = optional_usize(&args, "max_count")
            .unwrap_or(10)
            .clamp(1, 50);
        let reference = optional_string(&args, "ref").filter(|value| !value.trim().is_empty());

        let mut git_args: Vec<String> = Vec::new();
        match action {
            "status" => {
                git_args.extend(["status".into(), "--short".into(), "--branch".into()]);
            }
            "diff" => {
                git_args.push("diff".into());
                if staged {
                    git_args.push("--cached".into());
                }
                push_ref(&mut git_args, reference)?;
                push_path(&mut git_args, path)?;
            }
            "log" => {
                git_args.push("log".into());
                git_args.push(format!("--max-count={max_count}"));
                git_args.push("--oneline".into());
                git_args.push("--decorate".into());
                push_ref(&mut git_args, reference)?;
                push_path(&mut git_args, path)?;
            }
            "show" => {
                git_args.push("show".into());
                git_args.push("--no-color".into());
                git_args.push(
                    reference
                        .map(safe_ref_arg)
                        .transpose()?
                        .unwrap_or_else(|| "HEAD".to_string()),
                );
                push_path(&mut git_args, path)?;
            }
            "blame" => {
                let Some(path) = path else {
                    bail!("blame requires a path");
                };
                git_args.push("blame".into());
                git_args.push("--no-color".into());
                push_ref(&mut git_args, reference)?;
                push_path(&mut git_args, Some(path))?;
            }
            "branch" => {
                git_args.extend(["branch".into(), "--list".into(), "--no-color".into()]);
            }
            other => bail!("unsupported git action: {other}"),
        }

        super::command::run_workspace_command("git", &git_args, COMMAND_TIMEOUT_SECS).await
    }
}

fn push_ref(args: &mut Vec<String>, reference: Option<&str>) -> Result<()> {
    if let Some(reference) = reference {
        args.push(safe_ref_arg(reference)?);
    }
    Ok(())
}

fn push_path(args: &mut Vec<String>, path: Option<&str>) -> Result<()> {
    if let Some(path) = path {
        args.push("--".to_string());
        args.push(safe_relative_path_arg(path)?);
    }
    Ok(())
}

fn safe_ref_arg(reference: &str) -> Result<String> {
    let reference = reference.trim();
    if reference.is_empty() {
        bail!("ref cannot be empty");
    }
    if reference.starts_with('-') {
        bail!("ref must not start with '-': {reference}");
    }
    if reference
        .chars()
        .any(|ch| ch.is_whitespace() || ch.is_control())
    {
        bail!("ref must not contain whitespace or control characters: {reference}");
    }
    Ok(reference.to_string())
}

#[cfg(test)]
#[path = "../../tests/internal/tool/git/tests.rs"]
mod tests;
