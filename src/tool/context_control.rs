use anyhow::{Result, bail};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};

use super::{ToolHandler, ToolRegistry, optional_trimmed_string};
use crate::tool_names;

const MAX_CONTEXT_CHECKPOINT_REASON_CHARS: usize = 2_000;
const MAX_CONTEXT_CHECKPOINT_LABEL_CHARS: usize = 120;
const MAX_CONTEXT_RETURN_SUMMARY_CHARS: usize = 2_000;
const MAX_CONTEXT_RETURN_NEXT_ACTION_CHARS: usize = 1_000;

pub(super) fn register(registry: &mut ToolRegistry) {
    registry.register(ContextCheckpointTool);
    registry.register(ContextReturnTool);
}

struct ContextCheckpointTool;

struct ContextReturnTool;

#[async_trait]
impl ToolHandler for ContextCheckpointTool {
    fn name(&self) -> &'static str {
        tool_names::TOOL_CONTEXT_CHECKPOINT
    }

    fn description(&self) -> &'static str {
        "Create a context-only checkpoint before risky exploration or alternative approaches so later work continues on a new branch. This does not revert, isolate, or roll back files in the workspace."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "label": {
                    "type": ["string", "null"],
                    "maxLength": MAX_CONTEXT_CHECKPOINT_LABEL_CHARS,
                    "description": "Optional short branch label, such as 'try parser fix'"
                },
                "reason": {
                    "type": "string",
                    "maxLength": MAX_CONTEXT_CHECKPOINT_REASON_CHARS,
                    "description": "Why a new context branch is needed"
                }
            },
            "required": ["label", "reason"],
            "additionalProperties": false
        })
    }

    async fn execute(&self, args: Value) -> Result<Value> {
        let payload = validate_context_checkpoint(&args)?;
        Ok(json!({
            "label": payload.label,
            "reason": payload.reason,
            "context_only": true,
            "filesystem_rolled_back": false,
            "message": "Created a context checkpoint request. After this tool call is recorded, the agent will continue on a new context branch. This only affects agent context; files were not reverted."
        }))
    }
}

#[async_trait]
impl ToolHandler for ContextReturnTool {
    fn name(&self) -> &'static str {
        tool_names::TOOL_CONTEXT_RETURN
    }

    fn description(&self) -> &'static str {
        "Return from the current context experiment to the parent context and carry back a concise conclusion. This restores agent context only and does not revert files in the workspace."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "outcome": {
                    "type": "string",
                    "enum": ["useful", "dead_end", "blocked"],
                    "description": "How the current context experiment ended"
                },
                "summary": {
                    "type": "string",
                    "maxLength": MAX_CONTEXT_RETURN_SUMMARY_CHARS,
                    "description": "Concise conclusion to carry back into the parent context"
                },
                "next_action": {
                    "type": ["string", "null"],
                    "maxLength": MAX_CONTEXT_RETURN_NEXT_ACTION_CHARS,
                    "description": "Optional recommended next action after returning to the parent context"
                }
            },
            "required": ["outcome", "summary", "next_action"],
            "additionalProperties": false
        })
    }

    async fn execute(&self, args: Value) -> Result<Value> {
        let payload = validate_context_return(&args)?;
        Ok(json!({
            "outcome": payload.outcome,
            "summary": payload.summary,
            "next_action": payload.next_action,
            "context_restored": true,
            "filesystem_rolled_back": false,
            "message": "Returned from the current context experiment to the parent context. Files were not reverted."
        }))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
struct ContextCheckpointPayload {
    label: Option<String>,
    reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
struct ContextReturnPayload {
    outcome: String,
    summary: String,
    next_action: Option<String>,
}

fn validate_context_checkpoint(args: &Value) -> Result<ContextCheckpointPayload> {
    let label = optional_trimmed_string(args, "label")?;
    if let Some(label) = &label
        && label.chars().count() > MAX_CONTEXT_CHECKPOINT_LABEL_CHARS
    {
        bail!(
            "context__checkpoint field 'label' exceeds {MAX_CONTEXT_CHECKPOINT_LABEL_CHARS} characters"
        );
    }

    let Some(reason) = args.get("reason") else {
        bail!("context__checkpoint requires string field 'reason'");
    };
    let Some(reason) = reason.as_str() else {
        bail!("context__checkpoint requires string field 'reason'");
    };
    let reason = reason.trim();
    if reason.is_empty() {
        bail!("context__checkpoint field 'reason' must not be empty or whitespace");
    }
    if reason.chars().count() > MAX_CONTEXT_CHECKPOINT_REASON_CHARS {
        bail!(
            "context__checkpoint field 'reason' exceeds {MAX_CONTEXT_CHECKPOINT_REASON_CHARS} characters"
        );
    }

    Ok(ContextCheckpointPayload {
        label,
        reason: reason.to_string(),
    })
}

fn validate_context_return(args: &Value) -> Result<ContextReturnPayload> {
    let Some(outcome) = args.get("outcome").and_then(Value::as_str) else {
        bail!("context__return requires string field 'outcome'");
    };
    if !matches!(outcome, "useful" | "dead_end" | "blocked") {
        bail!("context__return field 'outcome' must be one of: useful, dead_end, blocked");
    }

    let Some(summary) = args.get("summary").and_then(Value::as_str) else {
        bail!("context__return requires string field 'summary'");
    };
    let summary = summary.trim();
    if summary.is_empty() {
        bail!("context__return field 'summary' must not be empty or whitespace");
    }
    if summary.chars().count() > MAX_CONTEXT_RETURN_SUMMARY_CHARS {
        bail!(
            "context__return field 'summary' exceeds {MAX_CONTEXT_RETURN_SUMMARY_CHARS} characters"
        );
    }

    let next_action = optional_trimmed_string(args, "next_action")?;
    if let Some(next_action) = &next_action
        && next_action.chars().count() > MAX_CONTEXT_RETURN_NEXT_ACTION_CHARS
    {
        bail!(
            "context__return field 'next_action' exceeds {MAX_CONTEXT_RETURN_NEXT_ACTION_CHARS} characters"
        );
    }

    Ok(ContextReturnPayload {
        outcome: outcome.to_string(),
        summary: summary.to_string(),
        next_action,
    })
}
