use anyhow::{Result, bail};
use async_trait::async_trait;
use serde_json::{Value, json};
use std::collections::BTreeSet;

use super::{ToolHandler, ToolRegistry};

const MAX_WORKFLOW_TODOS: usize = 100;
const MAX_WORKFLOW_TODO_FIELD_CHARS: usize = 1_000;
const MAX_WORKFLOW_AUTO_CONTINUATIONS: u64 = 16;

struct WorkflowTodosTool;

struct WorkflowAutoContinueTool;

pub(super) fn register(registry: &mut ToolRegistry) {
    registry.register(WorkflowTodosTool);
    registry.register(WorkflowAutoContinueTool);
}

#[async_trait]
impl ToolHandler for WorkflowTodosTool {
    fn name(&self) -> &'static str {
        "workflow__todos"
    }

    fn description(&self) -> &'static str {
        "Update the agent's current todo list for this turn."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "maxItems": MAX_WORKFLOW_TODOS,
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "string",
                                "maxLength": MAX_WORKFLOW_TODO_FIELD_CHARS,
                                "description": "Stable todo item id"
                            },
                            "content": {
                                "type": "string",
                                "maxLength": MAX_WORKFLOW_TODO_FIELD_CHARS,
                                "description": "Short todo description"
                            },
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "blocked", "completed", "cancelled"],
                                "description": "Todo status"
                            }
                        },
                        "required": ["id", "content", "status"],
                        "additionalProperties": false
                    },
                    "description": "Current turn todo list snapshot"
                }
            },
            "required": ["items"],
            "additionalProperties": false
        })
    }

    async fn execute(&self, args: Value) -> Result<Value> {
        validate_workflow_todos(&args)?;
        Ok(args)
    }
}

#[async_trait]
impl ToolHandler for WorkflowAutoContinueTool {
    fn name(&self) -> &'static str {
        "workflow__auto_continue"
    }

    fn description(&self) -> &'static str {
        "Enable or disable bounded internal auto-continuation for unfinished todos."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "enabled": {
                    "type": "boolean",
                    "description": "Whether bounded auto-continuation is enabled"
                },
                "max_continuations": {
                    "type": ["integer", "null"],
                    "minimum": 0,
                    "maximum": MAX_WORKFLOW_AUTO_CONTINUATIONS,
                    "description": "Optional per-turn continuation limit. Use null to keep the default."
                }
            },
            "required": ["enabled", "max_continuations"],
            "additionalProperties": false
        })
    }

    async fn execute(&self, args: Value) -> Result<Value> {
        validate_workflow_auto_continue(&args)?;
        Ok(args)
    }
}

fn validate_workflow_todos(args: &Value) -> Result<()> {
    let Some(items) = args.get("items").and_then(Value::as_array) else {
        bail!("workflow__todos requires an items array");
    };

    if items.len() > MAX_WORKFLOW_TODOS {
        bail!(
            "workflow__todos accepts at most {MAX_WORKFLOW_TODOS} items, got {}",
            items.len()
        );
    }

    let mut seen_ids = BTreeSet::new();
    let mut in_progress_count = 0;

    for (index, item) in items.iter().enumerate() {
        let mut id_value = None;

        for field in ["id", "content"] {
            let Some(value) = item.get(field).and_then(Value::as_str) else {
                bail!("workflow__todos item {index} requires string field '{field}'");
            };
            let length = value.chars().count();
            if length > MAX_WORKFLOW_TODO_FIELD_CHARS {
                bail!(
                    "workflow__todos item {index} field '{field}' exceeds {MAX_WORKFLOW_TODO_FIELD_CHARS} characters"
                );
            }

            if value.trim().is_empty() {
                bail!(
                    "workflow__todos item {index} field '{field}' must not be empty or whitespace"
                );
            }

            if field == "id" {
                id_value = Some(value);
            }
        }

        let id = id_value.expect("id must be captured after validation");
        if !seen_ids.insert(id) {
            bail!("workflow__todos item {index} has duplicate id '{id}'");
        }

        if item.get("status").and_then(Value::as_str) == Some("in_progress") {
            in_progress_count += 1;
            if in_progress_count > 1 {
                bail!("workflow__todos allows at most one item with status 'in_progress'");
            }
        }

        match item.get("status").and_then(Value::as_str) {
            Some("pending" | "in_progress" | "blocked" | "completed" | "cancelled") => {}
            Some(status) => {
                bail!("workflow__todos item {index} has invalid status '{status}'");
            }
            None => bail!("workflow__todos item {index} requires string field 'status'"),
        }
    }

    Ok(())
}

fn validate_workflow_auto_continue(args: &Value) -> Result<()> {
    if args.get("enabled").and_then(Value::as_bool).is_none() {
        bail!("workflow__auto_continue requires boolean field 'enabled'");
    }

    let Some(max_continuations) = args.get("max_continuations") else {
        bail!("workflow__auto_continue requires field 'max_continuations' as integer or null");
    };

    if max_continuations.is_null() {
        return Ok(());
    }

    let Some(max_continuations) = max_continuations.as_u64() else {
        bail!("workflow__auto_continue field 'max_continuations' must be integer or null");
    };
    if max_continuations > MAX_WORKFLOW_AUTO_CONTINUATIONS {
        bail!(
            "workflow__auto_continue max_continuations must be <= {MAX_WORKFLOW_AUTO_CONTINUATIONS}, got {max_continuations}"
        );
    }
    Ok(())
}
