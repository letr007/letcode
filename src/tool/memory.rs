use anyhow::Result;
use async_trait::async_trait;
use serde_json::{Value, json};

use super::{ToolHandler, ToolRegistry};
use crate::memory as memory_domain;
use crate::permission::ToolPermissionClass;
use crate::tool_names;

struct MemoryRecallTool;

pub(super) fn register(registry: &mut ToolRegistry) {
    registry.register(MemoryRecallTool);
}

#[async_trait]
impl ToolHandler for MemoryRecallTool {
    fn name(&self) -> &'static str {
        tool_names::TOOL_MEMORY_RECALL
    }

    fn description(&self) -> &'static str {
        "Recall useful experiment results, decisions, validations, or diagnostics from recent top-level sessions before repeating investigation or retrying a failed approach."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "query": {"type": ["string", "null"]},
                "paths": {"type": ["array", "null"], "items": {"type": "string"}},
                "kinds": {
                    "type": ["array", "null"],
                    "items": {"type": "string", "enum": ["experiment_result", "decision", "validation", "diagnostic"]}
                },
                "statuses": {
                    "type": ["array", "null"],
                    "items": {"type": "string", "enum": ["active", "useful", "dead_end", "blocked"]}
                },
                "limit": {"type": ["integer", "null"], "minimum": 1, "maximum": 20}
            },
            "required": ["query", "paths", "kinds", "statuses", "limit"],
            "additionalProperties": false
        })
    }

    fn permission_class(&self) -> ToolPermissionClass {
        ToolPermissionClass::Read
    }

    async fn execute(&self, args: Value) -> Result<Value> {
        let query = memory_domain::validate_memory_recall_query(&args)?;
        let memories = memory_domain::recall_recent_memories(&query)?;
        Ok(json!({"memories": memories}))
    }
}
