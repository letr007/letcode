use anyhow::{Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::{LazyLock, Mutex};

use crate::evidence::{EvidenceKind, EvidenceRecord, EvidenceSource, restore_evidence_records};
use crate::transcript::{TranscriptEvent, TranscriptRecord, list_sessions, read_records};

const DEFAULT_MEMORY_RECALL_LIMIT: usize = 5;
const MAX_MEMORY_RECALL_LIMIT: usize = 20;
const MAX_MEMORY_RECALL_SESSIONS: usize = 20;

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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryObject {
    pub id: String,
    pub session_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub branch_id: Option<String>,
    pub sequence: u64,
    pub timestamp_ms: u128,
    pub kind: MemoryKind,
    pub status: MemoryStatus,
    pub title: String,
    pub summary: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub paths: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
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

pub fn project_memory_objects(
    session_id: &str,
    records: &[TranscriptRecord],
) -> Result<Vec<MemoryObject>> {
    let mut memories = Vec::new();

    for record in records {
        if let TranscriptEvent::ContextExperimentReturned {
            branch_id,
            base_sequence,
            outcome,
            summary,
            next_action,
            had_writes,
            ..
        } = &record.event
        {
            let paths = experiment_paths(records, branch_id, *base_sequence, record.sequence);
            let detail = match (next_action.as_deref(), *had_writes) {
                (Some(next_action), true) => Some(format!(
                    "Next action: {next_action}. Context restored, files were NOT reverted."
                )),
                (Some(next_action), false) => Some(format!("Next action: {next_action}.")),
                (None, true) => Some("Context restored, files were NOT reverted.".into()),
                (None, false) => None,
            };
            memories.push(MemoryObject {
                id: format!("{session_id}:experiment:{branch_id}:{}", record.sequence),
                session_id: session_id.to_string(),
                branch_id: Some(branch_id.clone()),
                sequence: record.sequence,
                timestamp_ms: record.timestamp_ms,
                kind: MemoryKind::ExperimentResult,
                status: memory_status_for_outcome(outcome)?,
                title: format!("{} · {branch_id}", experiment_title(outcome)),
                summary: summary.clone(),
                detail,
                paths,
                tags: experiment_tags(branch_id, outcome, *had_writes),
            });
        }
    }

    for evidence in restore_evidence_records(records)? {
        if let Some(memory) = memory_from_evidence(session_id, &evidence) {
            memories.push(memory);
        }
    }

    Ok(memories)
}

pub fn recall_memory_objects(
    memories: &[MemoryObject],
    query: &MemoryRecallQuery,
) -> Vec<MemoryObject> {
    let needle = query.query.as_ref().map(|value| value.to_ascii_lowercase());
    let mut filtered = memories
        .iter()
        .filter(|memory| query.kinds.is_empty() || query.kinds.contains(&memory.kind))
        .filter(|memory| query.statuses.is_empty() || query.statuses.contains(&memory.status))
        .filter(|memory| query.paths.is_empty() || memory_path_matches(memory, &query.paths))
        .filter(|memory| {
            needle
                .as_ref()
                .is_none_or(|needle| memory_matches_query(memory, needle))
        })
        .cloned()
        .collect::<Vec<_>>();

    filtered.sort_by(|left, right| {
        memory_relevance_score(right, needle.as_deref(), &query.paths)
            .cmp(&memory_relevance_score(
                left,
                needle.as_deref(),
                &query.paths,
            ))
            .then_with(|| memory_status_rank(left.status).cmp(&memory_status_rank(right.status)))
            .then_with(|| right.timestamp_ms.cmp(&left.timestamp_ms))
            .then_with(|| right.sequence.cmp(&left.sequence))
            .then_with(|| left.id.cmp(&right.id))
    });
    filtered.truncate(query.limit);
    filtered
}

pub fn recall_recent_memories(query: &MemoryRecallQuery) -> Result<Vec<MemoryObject>> {
    let sessions_dir = configured_memory_sessions_dir()?;
    let sessions = list_sessions(&sessions_dir)?;
    let mut all = Vec::new();
    for session in sessions.into_iter().take(MAX_MEMORY_RECALL_SESSIONS) {
        let path = sessions_dir.join(format!("{}.jsonl", session.session_id));
        let records = read_records(&path)?;
        all.extend(project_memory_objects(&session.session_id, &records)?);
    }
    Ok(recall_memory_objects(&all, query))
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

fn configured_memory_sessions_dir() -> Result<PathBuf> {
    MEMORY_SESSIONS_DIR
        .lock()
        .map_err(|_| anyhow!("memory sessions dir lock poisoned"))?
        .clone()
        .ok_or_else(|| anyhow!("memory sessions directory is not configured"))
}

fn memory_from_evidence(session_id: &str, evidence: &EvidenceRecord) -> Option<MemoryObject> {
    let kind = match evidence.evidence_kind {
        EvidenceKind::Decision => MemoryKind::Decision,
        EvidenceKind::Validation => MemoryKind::Validation,
        EvidenceKind::Diagnostic => MemoryKind::Diagnostic,
        _ => return None,
    };
    Some(MemoryObject {
        id: format!("{session_id}:evidence:{}", evidence.id),
        session_id: session_id.to_string(),
        branch_id: None,
        sequence: evidence.sequence,
        timestamp_ms: evidence.timestamp_ms,
        kind,
        status: memory_status_from_evidence(evidence, kind),
        title: evidence.title.clone(),
        summary: evidence.summary.clone(),
        detail: evidence.detail.clone(),
        paths: evidence_path(&evidence.source).into_iter().collect(),
        tags: evidence.tags.clone(),
    })
}

fn memory_status_from_evidence(evidence: &EvidenceRecord, kind: MemoryKind) -> MemoryStatus {
    let lower_title = evidence.title.to_ascii_lowercase();
    let lower_summary = evidence.summary.to_ascii_lowercase();
    match kind {
        MemoryKind::Validation => {
            if ["fail", "failed", "error", "blocked"]
                .iter()
                .any(|needle| lower_title.contains(needle) || lower_summary.contains(needle))
            {
                MemoryStatus::Blocked
            } else {
                MemoryStatus::Useful
            }
        }
        MemoryKind::Decision => MemoryStatus::Useful,
        MemoryKind::Diagnostic => MemoryStatus::Active,
        MemoryKind::ExperimentResult => MemoryStatus::Active,
    }
}

fn evidence_path(source: &EvidenceSource) -> Option<String> {
    match source {
        EvidenceSource::File { path, .. } => Some(path.clone()),
        EvidenceSource::Command { .. }
        | EvidenceSource::Subagent { .. }
        | EvidenceSource::ToolCall { .. }
        | EvidenceSource::Transcript { .. } => None,
    }
}

fn memory_status_for_outcome(outcome: &str) -> Result<MemoryStatus> {
    match outcome {
        "useful" => Ok(MemoryStatus::Useful),
        "dead_end" => Ok(MemoryStatus::DeadEnd),
        "blocked" => Ok(MemoryStatus::Blocked),
        other => Err(anyhow!("unknown context experiment outcome '{other}'")),
    }
}

fn experiment_title(outcome: &str) -> &'static str {
    match outcome {
        "useful" => "Useful experiment",
        "dead_end" => "Dead end experiment",
        "blocked" => "Blocked experiment",
        _ => "Experiment result",
    }
}

fn experiment_tags(branch_id: &str, outcome: &str, had_writes: bool) -> Vec<String> {
    let mut tags = vec![
        branch_id.to_string(),
        outcome.to_string(),
        "context_experiment".into(),
    ];
    if had_writes {
        tags.push("had_writes".into());
    }
    tags
}

fn experiment_paths(
    records: &[TranscriptRecord],
    branch_id: &str,
    base_sequence: u64,
    leaf_sequence: u64,
) -> Vec<String> {
    let mut paths = Vec::<String>::new();
    for record in records {
        if record.sequence < base_sequence || record.sequence > leaf_sequence {
            continue;
        }
        if record.context_branch_id.as_deref() != Some(branch_id) {
            continue;
        }
        match &record.event {
            TranscriptEvent::Evidence { source, tags, .. } => {
                if let EvidenceSource::File { path, .. } = source {
                    push_unique_path(&mut paths, path);
                }
                for tag in tags {
                    if looks_like_path(tag) {
                        push_unique_path(&mut paths, tag);
                    }
                }
            }
            TranscriptEvent::ToolExecutionSummary(event) => {
                if let Some(path) = &event.primary_path {
                    push_unique_path(&mut paths, path);
                }
            }
            _ => {}
        }
    }
    paths
}

fn push_unique_path(paths: &mut Vec<String>, path: &str) {
    if !paths.iter().any(|existing| existing == path) {
        paths.push(path.to_string());
    }
}

fn looks_like_path(value: &str) -> bool {
    value.contains('/')
        || value.ends_with(".rs")
        || value.ends_with(".toml")
        || value.ends_with(".json")
        || value.ends_with(".md")
}

fn memory_status_rank(status: MemoryStatus) -> u8 {
    match status {
        MemoryStatus::Useful => 0,
        MemoryStatus::Active => 1,
        MemoryStatus::Blocked => 2,
        MemoryStatus::DeadEnd => 3,
    }
}

fn memory_matches_query(memory: &MemoryObject, needle: &str) -> bool {
    memory.title.to_ascii_lowercase().contains(needle)
        || memory.summary.to_ascii_lowercase().contains(needle)
        || memory
            .detail
            .as_ref()
            .is_some_and(|detail| detail.to_ascii_lowercase().contains(needle))
        || memory
            .tags
            .iter()
            .any(|tag| tag.to_ascii_lowercase().contains(needle))
        || memory
            .paths
            .iter()
            .any(|path| path.to_ascii_lowercase().contains(needle))
        || memory
            .branch_id
            .as_ref()
            .is_some_and(|branch| branch.to_ascii_lowercase().contains(needle))
}

fn memory_path_matches(memory: &MemoryObject, paths: &[String]) -> bool {
    memory.paths.iter().any(|path| {
        paths
            .iter()
            .any(|candidate| path == candidate || path.starts_with(&format!("{candidate}/")))
    })
}

fn memory_relevance_score(memory: &MemoryObject, needle: Option<&str>, paths: &[String]) -> usize {
    let mut score = 0usize;
    if let Some(needle) = needle {
        if memory.title.to_ascii_lowercase().contains(needle) {
            score += 10;
        }
        if memory.summary.to_ascii_lowercase().contains(needle) {
            score += 6;
        }
        if memory
            .detail
            .as_ref()
            .is_some_and(|detail| detail.to_ascii_lowercase().contains(needle))
        {
            score += 3;
        }
        if memory
            .tags
            .iter()
            .any(|tag| tag.to_ascii_lowercase().contains(needle))
        {
            score += 5;
        }
        if memory
            .paths
            .iter()
            .any(|path| path.to_ascii_lowercase().contains(needle))
        {
            score += 8;
        }
        if memory
            .branch_id
            .as_ref()
            .is_some_and(|branch| branch.to_ascii_lowercase().contains(needle))
        {
            score += 4;
        }
    }
    for candidate in paths {
        if memory.paths.iter().any(|path| path == candidate) {
            score += 10;
        } else if memory
            .paths
            .iter()
            .any(|path| path.starts_with(&format!("{candidate}/")))
        {
            score += 6;
        }
    }
    score
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
                bail!(
                    "memory__recall field '{field}' must not be empty or whitespace when provided"
                );
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
                    bail!("memory__recall field '{field}' item {index} must not be empty or whitespace");
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
    use crate::evidence::{EvidenceKind, EvidenceSource};
    use crate::transcript::{TranscriptEvent, TranscriptRecord, TranscriptRecorder};
    use serde_json::json;

    fn record(sequence: u64, event: TranscriptEvent) -> TranscriptRecord {
        TranscriptRecord {
            session_id: "s".into(),
            sequence,
            timestamp_ms: sequence as u128,
            context_branch_id: None,
            event,
        }
    }

    #[test]
    fn recall_filters_by_query_kind_status_path_and_limit() {
        let memories = vec![
            MemoryObject {
                id: "1".into(),
                session_id: "s".into(),
                branch_id: Some("branch-a".into()),
                sequence: 1,
                timestamp_ms: 10,
                kind: MemoryKind::ExperimentResult,
                status: MemoryStatus::Useful,
                title: "Experiment result · parser".into(),
                summary: "Parser root cause".into(),
                detail: None,
                paths: Vec::new(),
                tags: vec!["parser".into()],
            },
            MemoryObject {
                id: "2".into(),
                session_id: "s".into(),
                branch_id: None,
                sequence: 2,
                timestamp_ms: 20,
                kind: MemoryKind::Validation,
                status: MemoryStatus::Active,
                title: "Validation".into(),
                summary: "Ran parser tests".into(),
                detail: None,
                paths: vec!["src/parser.rs".into()],
                tags: vec!["tests".into()],
            },
            MemoryObject {
                id: "3".into(),
                session_id: "s".into(),
                branch_id: None,
                sequence: 3,
                timestamp_ms: 30,
                kind: MemoryKind::Diagnostic,
                status: MemoryStatus::Blocked,
                title: "Diagnostic".into(),
                summary: "Parser blocked on fixture".into(),
                detail: None,
                paths: vec!["src/parser.rs".into()],
                tags: vec!["fixture".into()],
            },
        ];

        let recalled = recall_memory_objects(
            &memories,
            &MemoryRecallQuery {
                query: Some("parser".into()),
                paths: vec!["src".into()],
                kinds: vec![MemoryKind::Validation, MemoryKind::Diagnostic],
                statuses: vec![MemoryStatus::Active, MemoryStatus::Blocked],
                limit: 1,
            },
        );

        assert_eq!(recalled.len(), 1);
        assert_eq!(recalled[0].id, "2");
    }

    #[test]
    fn validates_memory_recall_query() {
        let query = validate_memory_recall_query(&json!({
            "query": " parser ",
            "paths": ["src/parser.rs"],
            "kinds": ["decision", "validation"],
            "statuses": ["active", "useful"],
            "limit": 3
        }))
        .expect("valid query");

        assert_eq!(query.query.as_deref(), Some("parser"));
        assert_eq!(query.paths, vec!["src/parser.rs"]);
        assert_eq!(query.kinds.len(), 2);
        assert_eq!(query.statuses.len(), 2);
        assert_eq!(query.limit, 3);

        assert!(validate_memory_recall_query(&json!({"limit": 99})).is_err());
    }
}
