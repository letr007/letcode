use anyhow::{Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::{LazyLock, Mutex};

const DEFAULT_MEMORY_RECALL_LIMIT: usize = 5;
const MAX_MEMORY_RECALL_LIMIT: usize = 20;

// The session root remains the source for explicit transcript history tools and
// folded artifacts. Project-memory recall uses project_memory::MemoryStore.
static MEMORY_SESSIONS_DIR: LazyLock<Mutex<Option<PathBuf>>> = LazyLock::new(|| Mutex::new(None));

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryKind {
    ExperimentResult,
    Decision,
    Validation,
    Diagnostic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryStatus {
    Active,
    Useful,
    DeadEnd,
    Blocked,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryRecallQuery {
    pub query: Option<String>,
    pub paths: Vec<String>,
    pub kinds: Vec<MemoryKind>,
    pub statuses: Vec<MemoryStatus>,
    pub limit: usize,
}

pub fn set_memory_sessions_dir(path: PathBuf) {
    if let Ok(mut guard) = MEMORY_SESSIONS_DIR.lock() {
        *guard = Some(path);
    }
}

pub(crate) fn configured_memory_sessions_dir() -> Result<PathBuf> {
    MEMORY_SESSIONS_DIR
        .lock()
        .map_err(|_| anyhow!("memory sessions dir lock poisoned"))?
        .clone()
        .ok_or_else(|| anyhow!("memory sessions directory is not configured"))
}

pub fn validate_memory_recall_query(args: &serde_json::Value) -> Result<MemoryRecallQuery> {
    let query = optional_trimmed_string(args, "query")?;
    let paths = optional_trimmed_string_list(args, "paths")?;
    let kinds = optional_enum_list(args, "kinds", parse_memory_kind)?;
    let statuses = optional_enum_list(args, "statuses", parse_memory_status)?;
    let limit = match args.get("limit") {
        None | Some(serde_json::Value::Null) => DEFAULT_MEMORY_RECALL_LIMIT,
        Some(value) => {
            let Some(limit) = value.as_u64() else {
                bail!("memory__recall field 'limit' must be an integer or null");
            };
            if limit == 0 || limit as usize > MAX_MEMORY_RECALL_LIMIT {
                bail!(
                    "memory__recall field 'limit' must be between 1 and {MAX_MEMORY_RECALL_LIMIT}"
                );
            }
            limit as usize
        }
    };
    Ok(MemoryRecallQuery {
        query,
        paths,
        kinds,
        statuses,
        limit,
    })
}

fn parse_memory_kind(value: &str) -> Result<MemoryKind> {
    match value {
        "experiment_result" => Ok(MemoryKind::ExperimentResult),
        "decision" => Ok(MemoryKind::Decision),
        "validation" => Ok(MemoryKind::Validation),
        "diagnostic" => Ok(MemoryKind::Diagnostic),
        _ => bail!("unknown memory kind '{value}'"),
    }
}

fn parse_memory_status(value: &str) -> Result<MemoryStatus> {
    match value {
        "active" => Ok(MemoryStatus::Active),
        "useful" => Ok(MemoryStatus::Useful),
        "dead_end" => Ok(MemoryStatus::DeadEnd),
        "blocked" => Ok(MemoryStatus::Blocked),
        _ => bail!("unknown memory status '{value}'"),
    }
}

fn optional_trimmed_string(args: &serde_json::Value, field: &str) -> Result<Option<String>> {
    match args.get(field) {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(serde_json::Value::String(value)) => {
            let trimmed = value.trim();
            if trimmed.is_empty() {
                return Ok(None);
            }
            Ok(Some(trimmed.to_string()))
        }
        Some(_) => bail!("memory__recall field '{field}' must be a string or null"),
    }
}

fn optional_trimmed_string_list(args: &serde_json::Value, field: &str) -> Result<Vec<String>> {
    match args.get(field) {
        None | Some(serde_json::Value::Null) => Ok(Vec::new()),
        Some(serde_json::Value::Array(items)) => items
            .iter()
            .enumerate()
            .map(|(index, item)| {
                let Some(value) = item.as_str() else {
                    bail!("memory__recall field '{field}' item {index} must be a string");
                };
                let trimmed = value.trim();
                if trimmed.is_empty() {
                    bail!(
                        "memory__recall field '{field}' item {index} must not be empty or whitespace"
                    );
                }
                Ok(trimmed.to_string())
            })
            .collect(),
        Some(_) => bail!("memory__recall field '{field}' must be an array of strings or null"),
    }
}

fn optional_enum_list<T, F>(args: &serde_json::Value, field: &str, parse: F) -> Result<Vec<T>>
where
    F: Fn(&str) -> Result<T>,
{
    optional_trimmed_string_list(args, field)?
        .into_iter()
        .map(|value| parse(&value))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn validates_memory_recall_query() {
        let query = validate_memory_recall_query(&json!({
            "query": " parser ",
            "paths": ["src/parser.rs"],
            "kinds": ["decision", "validation"],
            "statuses": ["active", "useful"],
            "limit": 3
        }))
        .unwrap();
        assert_eq!(query.query.as_deref(), Some("parser"));
        assert_eq!(query.paths, vec!["src/parser.rs"]);
        assert_eq!(query.kinds.len(), 2);
        assert_eq!(query.statuses.len(), 2);
        assert_eq!(query.limit, 3);
        assert!(validate_memory_recall_query(&json!({"limit": 99})).is_err());
    }

    #[test]
    fn normalizes_empty_query_to_none_without_relaxing_validation() {
        let null_query = validate_memory_recall_query(&json!({"query": null})).unwrap();
        let empty_query = validate_memory_recall_query(&json!({"query": ""})).unwrap();
        let whitespace_query = validate_memory_recall_query(&json!({"query": "  \t\n"})).unwrap();

        assert_eq!(empty_query, null_query);
        assert_eq!(whitespace_query, null_query);
        assert!(validate_memory_recall_query(&json!({"query": 1})).is_err());
        assert!(validate_memory_recall_query(&json!({"paths": [""]})).is_err());
        assert!(validate_memory_recall_query(&json!({"paths": [" "]})).is_err());
    }
}
