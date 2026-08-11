use anyhow::{Result, bail};
use async_trait::async_trait;
use serde_json::{Value, json};
use std::collections::BTreeSet;

use super::{ToolHandler, ToolRegistry};

const MAX_WORKFLOW_TODOS: usize = 100;
const MAX_WORKFLOW_TODO_FIELD_CHARS: usize = 1_000;
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
        "Replace the agent's current todo list with a full snapshot. One snapshot may update multiple items, and multiple items may be in_progress."
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
                    "description": "Full current-turn todo list snapshot; one snapshot may update multiple items, including multiple items with status in_progress"
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
        "Enable or disable autonomous continuation. Use workflow__todos to track work when useful, but todo statuses do not control continuation. Once enabled, normal model completion continues until the agent disables this extension; explicit interruption, shutdown, and interactive permission or question requests remain stopping boundaries."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "enabled": {
                    "type": "boolean",
                    "description": "Whether bounded auto-continuation is enabled"
                },
            },
            "required": ["enabled"],
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

#[cfg(test)]
mod tests {
    use super::validate_workflow_todos;
    use serde_json::json;

    #[test]
    fn accepts_multiple_simultaneous_status_changes_in_one_snapshot() {
        let snapshot = json!({
            "items": [
                {"id": "investigate", "content": "Investigate issue", "status": "completed"},
                {"id": "implement", "content": "Implement fix", "status": "in_progress"},
                {"id": "validate", "content": "Validate fix", "status": "pending"}
            ]
        });

        assert!(validate_workflow_todos(&snapshot).is_ok());
    }

    #[test]
    fn accepts_multiple_in_progress_items() {
        let snapshot = json!({
            "items": [
                {"id": "implement", "content": "Implement fix", "status": "in_progress"},
                {"id": "review", "content": "Review fix", "status": "in_progress"}
            ]
        });

        assert!(validate_workflow_todos(&snapshot).is_ok());
    }
}

fn validate_workflow_auto_continue(args: &Value) -> Result<()> {
    if args.get("enabled").and_then(Value::as_bool).is_none() {
        bail!("workflow__auto_continue requires boolean field 'enabled'");
    }

    Ok(())
}
