use super::{ToolExecutionContext, ToolHandler, ToolParallelism, ToolRegistry};
use crate::context_history::{HistoryCursor, read_sources, source_entries};
use crate::permission::ToolPermissionClass;
use anyhow::{Result, bail, ensure};
use async_trait::async_trait;
use serde_json::{Value, json};

pub(super) fn register(registry: &mut ToolRegistry) {
    registry.register(HistoryTool { expand: false });
    registry.register(HistoryTool { expand: true });
}
struct HistoryTool {
    expand: bool,
}
#[async_trait]
impl ToolHandler for HistoryTool {
    fn name(&self) -> &'static str {
        if self.expand {
            "context__expand"
        } else {
            "context__search"
        }
    }
    fn description(&self) -> &'static str {
        if self.expand {
            "Read an archived history compartment's original messages, or a recorded evidence excerpt, by ID. Read-only: does not restore runtime state or execute historical tools. Results are character-paginated; evidence excerpts are not guaranteed complete original output."
        } else {
            "Search the current session branch's raw conversation, archived compartments and evidence by text. Returns source IDs and bounded previews; use context__expand for details. Other sessions require an explicit session_id and retain their historical provenance."
        }
    }
    fn parameters(&self) -> Value {
        let mut properties = json!({
            "session_id": {"type":["string","null"],"description":"Null uses the current session."},
            "branch_id": {"type":["string","null"],"description":"Null uses the current branch, or root for an explicitly selected other session."},
            "offset": {"type":["integer","null"],"minimum":0},
            "limit": {"type":["integer","null"],"minimum":1}
        });
        properties[if self.expand { "id" } else { "query" }] = json!({"type":"string"});
        json!({"type":"object","properties":properties,"required":[if self.expand {"id"} else {"query"},"session_id","branch_id","offset","limit"],"additionalProperties":false})
    }
    fn permission_class(&self) -> ToolPermissionClass {
        ToolPermissionClass::Read
    }
    fn parallelism(&self) -> ToolParallelism {
        ToolParallelism::Parallel
    }
    async fn execute(&self, _args: Value) -> Result<Value> {
        bail!("history tools require a session context")
    }
    async fn execute_with_context(
        &self,
        args: Value,
        context: ToolExecutionContext,
    ) -> Result<Value> {
        let current = context
            .history_cursor
            .ok_or_else(|| anyhow::anyhow!("history tools require a persisted session"))?;
        let session_id =
            optional_string(&args, "session_id")?.unwrap_or_else(|| current.session_id.clone());
        let same_session = session_id == current.session_id;
        let branch_id = optional_string(&args, "branch_id")?.unwrap_or_else(|| {
            if same_session {
                current.branch_id.clone()
            } else {
                crate::transcript::ROOT_CONTEXT_BRANCH_ID.into()
            }
        });
        let leaf_sequence = if same_session && branch_id == current.branch_id {
            current.leaf_sequence
        } else {
            None
        };
        let cursor = HistoryCursor {
            session_id,
            branch_id,
            leaf_sequence,
        };
        let records = read_sources(&cursor)?;
        let entries = source_entries(&records)?;
        let offset = number(&args, "offset", 0, usize::MAX)?;
        if self.expand {
            let id = args["id"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("id must be a string"))?;
            let limit = number(&args, "limit", 8000, 32000)?;
            ensure!(limit > 0, "limit must be positive");
            let entry = entries.iter().find(|e| e.id == id).ok_or_else(|| {
                anyhow::anyhow!("history source '{id}' is unavailable in this branch")
            })?;
            let text = if matches!(entry.kind, "compartment" | "fact") {
                let mut parts = Vec::new();
                for source in &entry.source_ids {
                    let raw = entries.iter().find(|e| &e.id == source).ok_or_else(|| {
                        anyhow::anyhow!("original history source '{source}' is unavailable")
                    })?;
                    parts.push(format!("[{}]\n{}", raw.id, raw.text));
                }
                parts.join("\n\n")
            } else {
                entry.text.clone()
            };
            let count = text.chars().count();
            let content: String = text.chars().skip(offset).take(limit).collect();
            Ok(
                json!({"session_id":cursor.session_id,"branch_id":cursor.branch_id,"id":id,"kind":entry.kind,"source_ids":entry.source_ids,"content":content,"next_offset":(offset.saturating_add(limit)<count).then_some(offset.saturating_add(limit)),"total_chars":count}),
            )
        } else {
            let query = args["query"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("query must be a string"))?
                .trim()
                .to_lowercase();
            ensure!(!query.is_empty(), "query must not be empty");
            let limit = number(&args, "limit", 10, 50)?;
            ensure!(limit > 0, "limit must be positive");
            let matching: Vec<_> = entries
                .iter()
                .filter(|e| e.text.to_lowercase().contains(&query) || e.id == query)
                .collect();
            let results: Vec<_> = matching.iter().skip(offset).take(limit).map(|e| json!({"id":e.id,"kind":e.kind,"preview":e.text.chars().take(500).collect::<String>(),"source_ids":e.source_ids})).collect();
            Ok(
                json!({"session_id":cursor.session_id,"branch_id":cursor.branch_id,"results":results,"next_offset":(offset.saturating_add(limit)<matching.len()).then_some(offset.saturating_add(limit)),"total":matching.len()}),
            )
        }
    }
}
fn optional_string(args: &Value, key: &str) -> Result<Option<String>> {
    match args.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) if !s.trim().is_empty() => Ok(Some(s.trim().to_string())),
        _ => bail!("{key} must be a nonempty string or null"),
    }
}
fn number(args: &Value, key: &str, default: usize, max: usize) -> Result<usize> {
    match args.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(v) => {
            let n = v
                .as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .ok_or_else(|| anyhow::anyhow!("{key} must be a nonnegative integer"))?;
            ensure!(n <= max, "{key} exceeds {max}");
            Ok(n)
        }
    }
}
