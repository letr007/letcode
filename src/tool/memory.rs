use anyhow::Result;
use async_trait::async_trait;
use serde_json::{Value, json};

use super::{ToolHandler, ToolParallelism, ToolRegistry};
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
        "Search system-maintained project memory for the current workspace by keywords, code symbols or paths. Read-only; no session scan or automatic history import. Returns source session/branch/raw IDs for context__expand. Memories can be incomplete or outdated; verify against current code. Use short keywords (including Chinese), not a full question."
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

    fn parallelism(&self) -> ToolParallelism {
        ToolParallelism::Parallel
    }

    async fn execute(&self, args: Value) -> Result<Value> {
        let query = memory_domain::validate_memory_recall_query(&args)?;
        let store = crate::project_memory::configured_store()?
            .ok_or_else(|| anyhow::anyhow!("project memory is not configured"))?;
        tokio::task::spawn_blocking(move || {
            let memories = store.query(&query)?;
            let status = store.status()?;
            Ok(json!({"memories": memories, "status": status}))
        })
        .await?
    }
}
