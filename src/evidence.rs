use anyhow::{Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashSet;

use crate::agent::{ToolEffectKind, ToolExecutionRecord};
use crate::tool::ToolResult;
use crate::transcript::{TranscriptEvent, TranscriptRecord};

const MAX_EVIDENCE_DETAIL_BYTES: usize = 8 * 1024;
const MAX_EVIDENCE_SUMMARY_CHARS: usize = 500;
const MAX_EVIDENCE_TAGS: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceKind {
    FileExcerpt,
    SearchResult,
    CommandResult,
    Change,
    Diagnostic,
    Decision,
    Validation,
}

impl EvidenceKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::FileExcerpt => "file_excerpt",
            Self::SearchResult => "search_result",
            Self::CommandResult => "command_result",
            Self::Change => "change",
            Self::Diagnostic => "diagnostic",
            Self::Decision => "decision",
            Self::Validation => "validation",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EvidenceSource {
    ToolCall {
        call_id: String,
        tool: String,
    },
    File {
        path: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        start_line: Option<u64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        end_line: Option<u64>,
    },
    Command {
        command: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        status: Option<i32>,
    },
    Subagent {
        run_id: String,
        child_session_id: String,
        source_session_id: String,
        parent_tool: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        parent_turn_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        parent_session_id: Option<String>,
    },
    Transcript {
        sequence: u64,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceDraft {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub evidence_kind: EvidenceKind,
    pub title: String,
    pub summary: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    pub source: EvidenceSource,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceRecord {
    pub id: String,
    pub sequence: u64,
    pub timestamp_ms: u128,
    pub evidence_kind: EvidenceKind,
    pub title: String,
    pub summary: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    pub source: EvidenceSource,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
}

impl EvidenceDraft {
    #[allow(dead_code)]
    pub fn from_tool_result(
        call_id: impl Into<String>,
        tool_name: impl Into<String>,
        args: Value,
        output: &ToolResult,
    ) -> Self {
        let call_id = call_id.into();
        let tool_name = tool_name.into();
        let source = evidence_source_for_tool(&call_id, &tool_name, &args, output);
        let kind = evidence_kind_for_tool(&tool_name, output);
        let title = evidence_title(&tool_name, output, &args);
        let summary = truncate_chars(
            &evidence_summary(&tool_name, output, &args),
            MAX_EVIDENCE_SUMMARY_CHARS,
        );
        let detail = evidence_detail(output)
            .map(|detail| truncate_bytes(&detail, MAX_EVIDENCE_DETAIL_BYTES));
        let tags = evidence_tags(&tool_name, &args, output);

        Self {
            id: None,
            evidence_kind: kind,
            title,
            summary,
            detail,
            source,
            tags,
        }
    }

    pub fn from_tool_execution_record(record: &ToolExecutionRecord) -> Self {
        let args = record.arguments.clone().unwrap_or(Value::Null);
        let source = evidence_source_for_record(record, &args);
        let kind = evidence_kind_for_record(record);
        let title = evidence_title(&record.tool_name, &record.output, &args);
        let summary = truncate_chars(
            &evidence_summary(&record.tool_name, &record.output, &args),
            MAX_EVIDENCE_SUMMARY_CHARS,
        );
        let detail = evidence_detail(&record.output)
            .map(|detail| truncate_bytes(&detail, MAX_EVIDENCE_DETAIL_BYTES));
        let tags = evidence_tags_for_record(record, &args);

        Self {
            id: None,
            evidence_kind: kind,
            title,
            summary,
            detail,
            source,
            tags,
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.title.trim().is_empty() {
            bail!("evidence title cannot be empty");
        }
        if self.summary.trim().is_empty() {
            bail!("evidence summary cannot be empty");
        }
        if let Some(detail) = &self.detail
            && detail.len() > MAX_EVIDENCE_DETAIL_BYTES
        {
            bail!("evidence detail exceeds {MAX_EVIDENCE_DETAIL_BYTES} bytes");
        }
        if self.tags.len() > MAX_EVIDENCE_TAGS {
            bail!("evidence has too many tags");
        }
        validate_source(&self.source)
    }

    pub fn into_record(
        self,
        id: String,
        sequence: u64,
        timestamp_ms: u128,
    ) -> Result<EvidenceRecord> {
        self.validate()?;
        Ok(EvidenceRecord {
            id,
            sequence,
            timestamp_ms,
            evidence_kind: self.evidence_kind,
            title: self.title,
            summary: self.summary,
            detail: self.detail,
            source: self.source,
            tags: self.tags,
        })
    }
}

impl EvidenceRecord {
    pub fn compact_line(&self) -> String {
        let mut line = format!(
            "[{}] {} {} — {}",
            self.id,
            self.evidence_kind.as_str(),
            source_label(&self.source),
            self.summary
        );
        if matches!(self.source, EvidenceSource::Subagent { .. }) {
            return line;
        }
        if let Some(detail) = &self.detail {
            let compact_detail = detail.split_whitespace().collect::<Vec<_>>().join(" ");
            if !compact_detail.is_empty() {
                line.push_str(" :: ");
                line.push_str(&truncate_chars(&compact_detail, 700));
            }
        }
        line
    }
}

pub fn restore_evidence_records(records: &[TranscriptRecord]) -> Result<Vec<EvidenceRecord>> {
    let mut evidence = Vec::new();
    let mut seen = HashSet::new();

    for record in records {
        let TranscriptEvent::Evidence {
            id,
            evidence_kind,
            title,
            summary,
            detail,
            source,
            tags,
        } = &record.event
        else {
            continue;
        };

        if !seen.insert(id.clone()) {
            bail!("duplicate evidence id: {id}");
        }

        let draft = EvidenceDraft {
            id: Some(id.clone()),
            evidence_kind: *evidence_kind,
            title: title.clone(),
            summary: summary.clone(),
            detail: detail.clone(),
            source: source.clone(),
            tags: tags.clone(),
        };
        evidence.push(draft.into_record(id.clone(), record.sequence, record.timestamp_ms)?);
    }

    Ok(evidence)
}

pub fn evidence_context_message(
    evidence: &[EvidenceRecord],
    current_query: &str,
    budget_tokens: u64,
) -> (Option<String>, Vec<String>, usize) {
    let selected = select_relevant_evidence(evidence, current_query, budget_tokens);
    if selected.is_empty() {
        return (None, Vec::new(), evidence.len());
    }

    let ids = selected
        .iter()
        .map(|item| item.id.clone())
        .collect::<Vec<_>>();
    let mut text = String::from(
        "Relevant evidence:\nUse these compact references to prior observations. Cite evidence IDs when relying on them. If needed evidence is missing or stale, inspect the repository instead of assuming.\n",
    );
    for item in &selected {
        text.push_str("- ");
        text.push_str(&item.compact_line());
        text.push('\n');
    }
    let dropped = evidence.len().saturating_sub(selected.len());
    (Some(text), ids, dropped)
}

fn select_relevant_evidence(
    evidence: &[EvidenceRecord],
    current_query: &str,
    budget_tokens: u64,
) -> Vec<EvidenceRecord> {
    let budget_chars = budget_tokens.saturating_mul(4) as usize;
    let query_tokens = query_tokens(current_query);
    let latest_change_paths = latest_change_paths(evidence);
    let mut scored = evidence
        .iter()
        .filter(|record| !is_stale_file_evidence(record, &latest_change_paths))
        .map(|record| (evidence_score(record, &query_tokens), record))
        .filter(|(score, _)| *score > 0)
        .collect::<Vec<_>>();

    scored.sort_by(|(left_score, left), (right_score, right)| {
        right_score
            .cmp(left_score)
            .then_with(|| right.sequence.cmp(&left.sequence))
    });

    let mut selected = Vec::new();
    let mut used_chars = 0usize;
    let mut seen_sources = HashSet::new();
    for (_, record) in scored {
        let source = source_label(&record.source);
        if !seen_sources.insert(source.clone())
            && !matches!(
                record.evidence_kind,
                EvidenceKind::Diagnostic | EvidenceKind::Validation
            )
        {
            continue;
        }
        let cost = record.compact_line().len().saturating_add(4);
        if used_chars.saturating_add(cost) > budget_chars && !selected.is_empty() {
            continue;
        }
        used_chars = used_chars.saturating_add(cost);
        selected.push(record.clone());
    }

    selected
}

fn evidence_score(record: &EvidenceRecord, query_tokens: &HashSet<String>) -> i32 {
    let mut score = 0;
    if query_tokens.is_empty() {
        if matches!(
            record.evidence_kind,
            EvidenceKind::Diagnostic | EvidenceKind::Validation | EvidenceKind::Change
        ) {
            return 10 + record.sequence.min(20) as i32;
        }
        return 0;
    }
    let haystack = format!(
        "{} {} {} {}",
        record.title,
        record.summary,
        record.tags.join(" "),
        source_label(&record.source)
    )
    .to_lowercase();

    for token in query_tokens {
        if source_label(&record.source).to_lowercase().contains(token) {
            score += 100;
        } else if record
            .tags
            .iter()
            .any(|tag| tag.to_lowercase().contains(token))
        {
            score += 60;
        } else if haystack.contains(token) {
            score += 40;
        }
    }

    if score > 0 {
        score += record.sequence.min(50) as i32;
    }

    if matches!(
        record.evidence_kind,
        EvidenceKind::Diagnostic | EvidenceKind::Validation
    ) {
        score += 20;
    }

    score
}

fn query_tokens(query: &str) -> HashSet<String> {
    const STOP: &[&str] = &[
        "the", "and", "for", "that", "this", "with", "from", "what", "怎么", "当前",
    ];
    query
        .split(|ch: char| !(ch.is_alphanumeric() || matches!(ch, '/' | '_' | '.' | '-')))
        .map(str::trim)
        .filter(|token| token.chars().count() >= 3)
        .map(|token| token.to_lowercase())
        .filter(|token| !STOP.contains(&token.as_str()))
        .collect()
}

fn latest_change_paths(evidence: &[EvidenceRecord]) -> Vec<(String, u64)> {
    evidence
        .iter()
        .filter(|record| matches!(record.evidence_kind, EvidenceKind::Change))
        .filter_map(|record| match &record.source {
            EvidenceSource::File { path, .. } => Some((path.clone(), record.sequence)),
            _ => record
                .tags
                .iter()
                .find(|tag| tag.contains('/'))
                .cloned()
                .map(|path| (path, record.sequence)),
        })
        .collect()
}

fn is_stale_file_evidence(record: &EvidenceRecord, changes: &[(String, u64)]) -> bool {
    let EvidenceSource::File { path, .. } = &record.source else {
        return false;
    };
    changes
        .iter()
        .any(|(changed_path, sequence)| changed_path == path && *sequence > record.sequence)
}

#[allow(dead_code)]
fn evidence_kind_for_tool(tool_name: &str, output: &ToolResult) -> EvidenceKind {
    if !output.ok {
        return EvidenceKind::Diagnostic;
    }
    match tool_name {
        "fs__read" => EvidenceKind::FileExcerpt,
        "search__rg" | "code__ast_search" => EvidenceKind::SearchResult,
        "edit__apply_patch" | "fs__write" | "fs__append" | "fs__mkdir" => EvidenceKind::Change,
        "shell__exec"
            if command_text(output)
                .as_deref()
                .is_some_and(is_validation_command) =>
        {
            EvidenceKind::Validation
        }
        "shell__exec" => EvidenceKind::CommandResult,
        _ => EvidenceKind::Decision,
    }
}

fn evidence_kind_for_record(record: &ToolExecutionRecord) -> EvidenceKind {
    match record.effects.kind {
        ToolEffectKind::Read => match record.tool_name.as_str() {
            "fs__read" => EvidenceKind::FileExcerpt,
            "search__rg" | "code__ast_search" => EvidenceKind::SearchResult,
            "shell__exec" => EvidenceKind::CommandResult,
            _ => EvidenceKind::Decision,
        },
        ToolEffectKind::Write => EvidenceKind::Change,
        ToolEffectKind::Command => EvidenceKind::CommandResult,
        ToolEffectKind::Validation => EvidenceKind::Validation,
        ToolEffectKind::WorkflowControl | ToolEffectKind::Unknown => EvidenceKind::Decision,
        ToolEffectKind::Diagnostic => EvidenceKind::Diagnostic,
    }
}

#[allow(dead_code)]
fn evidence_source_for_tool(
    call_id: &str,
    tool_name: &str,
    args: &Value,
    output: &ToolResult,
) -> EvidenceSource {
    if let Some(path) = value_path(args).or_else(|| data_string(output, "path")) {
        return EvidenceSource::File {
            path,
            start_line: data_u64(output, "start_line").or_else(|| data_u64(output, "offset")),
            end_line: data_u64(output, "end_line"),
        };
    }
    if let Some(command) = value_string(args, "command").or_else(|| command_text(output)) {
        return EvidenceSource::Command {
            command,
            status: data_i64(output, "status").map(|status| status as i32),
        };
    }
    EvidenceSource::ToolCall {
        call_id: call_id.to_string(),
        tool: tool_name.to_string(),
    }
}

fn evidence_source_for_record(record: &ToolExecutionRecord, args: &Value) -> EvidenceSource {
    if matches!(record.tool_name.as_str(), "agent__explore" | "agent__fixer")
        && let Some(run_id) = data_string(&record.output, "run_id")
        && let Some(child_session_id) = data_string(&record.output, "child_session_id")
    {
        return EvidenceSource::Subagent {
            source_session_id: child_session_id.clone(),
            run_id,
            child_session_id,
            parent_tool: record.tool_name.clone(),
            parent_turn_id: None,
            parent_session_id: None,
        };
    }
    if let Some(path) = record
        .effects
        .primary_path
        .clone()
        .or_else(|| value_path(args))
        .or_else(|| data_string(&record.output, "path"))
    {
        return EvidenceSource::File {
            path,
            start_line: data_u64(&record.output, "start_line")
                .or_else(|| data_u64(&record.output, "offset")),
            end_line: data_u64(&record.output, "end_line"),
        };
    }
    if let Some(command) = record
        .effects
        .command
        .clone()
        .or_else(|| value_string(args, "command"))
        .or_else(|| command_text(&record.output))
    {
        return EvidenceSource::Command {
            command,
            status: data_i64(&record.output, "status").map(|status| status as i32),
        };
    }
    EvidenceSource::ToolCall {
        call_id: record.call_id.clone(),
        tool: record.tool_name.clone(),
    }
}

fn evidence_title(tool_name: &str, output: &ToolResult, args: &Value) -> String {
    if !output.ok {
        return format!("{tool_name} failed");
    }
    if tool_result_data_failed(output) {
        return format!("{tool_name} failed");
    }
    match value_path(args).or_else(|| data_string(output, "path")) {
        Some(path) => format!("{tool_name} {path}"),
        None => match value_string(args, "command").or_else(|| command_text(output)) {
            Some(command) => format!("{tool_name} {command}"),
            None => format!("{tool_name} completed"),
        },
    }
}

fn evidence_summary(tool_name: &str, output: &ToolResult, args: &Value) -> String {
    if let Some(error) = &output.error {
        return format!("{tool_name} failed: {}", error.message);
    }
    if tool_result_data_failed(output) {
        if let Some(command) = value_string(args, "command").or_else(|| command_text(output)) {
            return format!("command failed: {command}");
        }
        return format!("{tool_name} failed");
    }
    if let Some(summary) = data_string(output, "summary") {
        return summary;
    }
    if let Some(path) = value_path(args).or_else(|| data_string(output, "path")) {
        return format!("{tool_name} observed {path}");
    }
    if let Some(command) = value_string(args, "command").or_else(|| command_text(output)) {
        return format!("command completed: {command}");
    }
    format!("{tool_name} completed successfully")
}

fn evidence_detail(output: &ToolResult) -> Option<String> {
    if let Some(error) = &output.error {
        return Some(error.message.clone());
    }
    let data = output.data.as_ref()?;
    if let Some(content) = data.get("content").and_then(Value::as_str) {
        return Some(content.to_string());
    }
    if let Some(stdout) = data
        .get("stdout")
        .and_then(Value::as_str)
        .filter(|stdout| !stdout.trim().is_empty())
    {
        return Some(stdout.to_string());
    }
    if let Some(stderr) = data
        .get("stderr")
        .and_then(Value::as_str)
        .filter(|stderr| !stderr.trim().is_empty())
    {
        return Some(stderr.to_string());
    }
    serde_json::to_string(data).ok()
}

#[allow(dead_code)]
fn evidence_tags(tool_name: &str, args: &Value, output: &ToolResult) -> Vec<String> {
    let mut tags = vec![tool_name.to_string(), output.tool.clone()];
    if let Some(path) = value_path(args).or_else(|| data_string(output, "path")) {
        tags.push(path);
    }
    if let Some(pattern) = value_string(args, "pattern") {
        tags.push(pattern);
    }
    tags.extend(edited_paths(output));
    tags.sort();
    tags.dedup();
    tags.truncate(MAX_EVIDENCE_TAGS);
    tags
}

fn evidence_tags_for_record(record: &ToolExecutionRecord, args: &Value) -> Vec<String> {
    let mut tags = vec![record.tool_name.clone(), record.output.tool.clone()];
    if let Some(run_id) = data_string(&record.output, "run_id") {
        tags.push(run_id);
    }
    if let Some(child_session_id) = data_string(&record.output, "child_session_id") {
        tags.push(child_session_id);
    }
    if let Some(path) = record
        .effects
        .primary_path
        .clone()
        .or_else(|| value_path(args))
        .or_else(|| data_string(&record.output, "path"))
    {
        tags.push(path);
    }
    if let Some(pattern) = value_string(args, "pattern") {
        tags.push(pattern);
    }
    tags.extend(record.effects.edited_paths.clone());
    tags.sort();
    tags.dedup();
    tags.truncate(MAX_EVIDENCE_TAGS);
    tags
}

fn validate_source(source: &EvidenceSource) -> Result<()> {
    match source {
        EvidenceSource::ToolCall { call_id, tool } => {
            if call_id.trim().is_empty() || tool.trim().is_empty() {
                bail!("tool-call evidence source requires call_id and tool");
            }
        }
        EvidenceSource::File {
            path,
            start_line,
            end_line,
        } => {
            if path.trim().is_empty() {
                bail!("file evidence source requires path");
            }
            if let (Some(start), Some(end)) = (start_line, end_line)
                && end < start
            {
                bail!("file evidence source has invalid line range");
            }
        }
        EvidenceSource::Command { command, .. } => {
            if command.trim().is_empty() {
                bail!("command evidence source requires command");
            }
        }
        EvidenceSource::Subagent {
            run_id,
            child_session_id,
            source_session_id,
            parent_tool,
            ..
        } => {
            if run_id.trim().is_empty()
                || child_session_id.trim().is_empty()
                || source_session_id.trim().is_empty()
                || parent_tool.trim().is_empty()
            {
                bail!("subagent evidence source requires run/session/tool provenance");
            }
        }
        EvidenceSource::Transcript { sequence } if *sequence == 0 => {
            bail!("transcript evidence source requires positive sequence");
        }
        EvidenceSource::Transcript { .. } => {}
    }
    Ok(())
}

fn source_label(source: &EvidenceSource) -> String {
    match source {
        EvidenceSource::ToolCall { tool, call_id } => format!("{tool}:{call_id}"),
        EvidenceSource::File {
            path,
            start_line,
            end_line,
        } => match (start_line, end_line) {
            (Some(start), Some(end)) => format!("{path}:{start}-{end}"),
            (Some(start), None) => format!("{path}:{start}"),
            _ => path.clone(),
        },
        EvidenceSource::Command { command, status } => match status {
            Some(status) => format!("{command} (status {status})"),
            None => command.clone(),
        },
        EvidenceSource::Subagent {
            parent_tool,
            run_id,
            child_session_id,
            parent_turn_id,
            ..
        } => match parent_turn_id {
            Some(parent_turn_id) => {
                format!(
                    "{parent_tool}:{parent_turn_id}:{run_id}:/child {}",
                    truncate_chars(child_session_id, 16)
                )
            }
            None => format!(
                "{parent_tool}:{run_id}:/child {}",
                truncate_chars(child_session_id, 16)
            ),
        },
        EvidenceSource::Transcript { sequence } => format!("transcript:{sequence}"),
    }
}

#[allow(dead_code)]
fn edited_paths(output: &ToolResult) -> Vec<String> {
    output
        .data
        .as_ref()
        .and_then(|data| data.get("edits"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|edit| edit.get("path").and_then(Value::as_str))
        .map(str::to_string)
        .collect()
}

fn value_path(args: &Value) -> Option<String> {
    value_string(args, "path")
        .or_else(|| value_string(args, "file_path"))
        .or_else(|| value_string(args, "filePath"))
}

fn value_string(value: &Value, key: &str) -> Option<String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

fn data_string(output: &ToolResult, key: &str) -> Option<String> {
    output
        .data
        .as_ref()?
        .get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

fn data_u64(output: &ToolResult, key: &str) -> Option<u64> {
    output.data.as_ref()?.get(key).and_then(Value::as_u64)
}

fn data_i64(output: &ToolResult, key: &str) -> Option<i64> {
    output.data.as_ref()?.get(key).and_then(Value::as_i64)
}

fn data_bool(output: &ToolResult, key: &str) -> Option<bool> {
    output.data.as_ref()?.get(key).and_then(Value::as_bool)
}

fn tool_result_data_failed(output: &ToolResult) -> bool {
    if !output.ok {
        return true;
    }

    if data_bool(output, "success") == Some(false) {
        return true;
    }

    data_i64(output, "status").is_some_and(|status| status != 0)
        || output
            .data
            .as_ref()
            .is_some_and(|data| data.get("error").is_some())
}

fn command_text(output: &ToolResult) -> Option<String> {
    data_string(output, "command")
}

#[allow(dead_code)]
fn is_validation_command(command: &str) -> bool {
    ["test", "check", "clippy", "fmt"]
        .iter()
        .any(|needle| command.contains(needle))
}

fn truncate_chars(value: &str, max_chars: usize) -> String {
    let mut out = value.chars().take(max_chars).collect::<String>();
    if value.chars().count() > max_chars {
        out.push('…');
    }
    out
}

fn truncate_bytes(value: &str, max_bytes: usize) -> String {
    if value.len() <= max_bytes {
        return value.to_string();
    }
    let ellipsis = "…";
    let limit = max_bytes.saturating_sub(ellipsis.len());
    let end = value
        .char_indices()
        .map(|(index, _)| index)
        .take_while(|index| *index <= limit)
        .last()
        .unwrap_or(0);
    format!("{}{}", &value[..end], ellipsis)
}

pub fn estimate_evidence_tokens(text: &str) -> u64 {
    ((text.len() as u64 + 2) / 3).saturating_add(16)
}

pub fn evidence_id_for_sequence(sequence: u64) -> String {
    format!("ev-{sequence:06}")
}

pub fn require_unique_evidence_id(existing: &[EvidenceRecord], id: &str) -> Result<()> {
    if existing.iter().any(|record| record.id == id) {
        return Err(anyhow!("duplicate evidence id: {id}"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{
        ToolEffectKind, ToolEffects, ToolExecutionRecord, ToolExecutionRejection,
        ToolExecutionStatus,
    };
    use crate::permission::{ExecutionDirective, ToolPermissionClass};
    use serde_json::json;

    fn file_record(id: &str, path: &str, sequence: u64) -> EvidenceRecord {
        EvidenceRecord {
            id: id.into(),
            sequence,
            timestamp_ms: 0,
            evidence_kind: EvidenceKind::FileExcerpt,
            title: format!("read {path}"),
            summary: format!("read {path}"),
            detail: Some("old contents".into()),
            source: EvidenceSource::File {
                path: path.into(),
                start_line: Some(1),
                end_line: Some(2),
            },
            tags: vec![path.into()],
        }
    }

    fn tool_execution_record(
        tool_name: &str,
        arguments: Option<Value>,
        output: ToolResult,
        effects: ToolEffects,
    ) -> ToolExecutionRecord {
        ToolExecutionRecord {
            call_id: format!("call-{tool_name}"),
            tool_name: tool_name.into(),
            arguments,
            permission_class: ToolPermissionClass::Unknown,
            directive: ExecutionDirective::None,
            status: if output.ok {
                ToolExecutionStatus::Executed
            } else {
                ToolExecutionStatus::Rejected
            },
            rejection: (!output.ok).then_some(ToolExecutionRejection::PermissionDeniedByPolicy),
            output,
            effects,
        }
    }

    #[test]
    fn long_tool_output_is_truncated_to_valid_evidence_detail_size() {
        let output = ToolResult::ok(
            "fs__read",
            json!({
                "path": "src/lib.rs",
                "content": "x".repeat(MAX_EVIDENCE_DETAIL_BYTES + 100),
            }),
        );

        let draft = EvidenceDraft::from_tool_result(
            "call-1",
            "fs__read",
            json!({"path": "src/lib.rs"}),
            &output,
        );
        let record = draft
            .into_record("ev-1".into(), 1, 0)
            .expect("truncated evidence is valid");

        assert!(record.detail.expect("detail").len() <= MAX_EVIDENCE_DETAIL_BYTES);
    }

    #[test]
    fn empty_path_tool_result_uses_tool_call_evidence_source() {
        let output = ToolResult::ok(
            "git__diff",
            json!({
                "command": "git diff",
                "stdout": "diff --git a/src/lib.rs b/src/lib.rs\n",
            }),
        );

        let draft = EvidenceDraft::from_tool_result(
            "call-diff",
            "git__diff",
            json!({"staged": false, "path": ""}),
            &output,
        );
        let record = draft
            .into_record("ev-diff".into(), 1, 0)
            .expect("empty path is ignored for evidence source");

        assert_eq!(
            record.source,
            EvidenceSource::Command {
                command: "git diff".into(),
                status: None,
            }
        );
    }

    #[test]
    fn apply_patch_change_tags_make_older_file_evidence_stale() {
        let old_file = file_record("ev-old", "src/lib.rs", 1);
        let change = EvidenceRecord {
            id: "ev-change".into(),
            sequence: 2,
            timestamp_ms: 0,
            evidence_kind: EvidenceKind::Change,
            title: "apply patch".into(),
            summary: "changed src/lib.rs".into(),
            detail: None,
            source: EvidenceSource::ToolCall {
                call_id: "call-patch".into(),
                tool: "edit__apply_patch".into(),
            },
            tags: vec!["src/lib.rs".into()],
        };

        let (message, ids, _) = evidence_context_message(&[old_file, change], "src/lib.rs", 1024);
        let message = message.expect("evidence selected");
        assert!(ids.contains(&"ev-change".to_string()), "{message}");
        assert!(!ids.contains(&"ev-old".to_string()), "{message}");
    }

    #[test]
    fn empty_query_only_selects_high_signal_recent_evidence() {
        let file = file_record("ev-file", "src/lib.rs", 1);
        let diagnostic = EvidenceRecord {
            id: "ev-diag".into(),
            sequence: 2,
            timestamp_ms: 0,
            evidence_kind: EvidenceKind::Diagnostic,
            title: "cargo test failed".into(),
            summary: "test failed".into(),
            detail: None,
            source: EvidenceSource::Command {
                command: "cargo test".into(),
                status: Some(101),
            },
            tags: vec![],
        };

        let (_, ids, _) = evidence_context_message(&[file, diagnostic], "嗯", 1024);
        assert_eq!(ids, vec!["ev-diag".to_string()]);
    }

    #[test]
    fn from_tool_execution_record_maps_write_effect_to_change_file_source() {
        let record = tool_execution_record(
            "fs__write",
            Some(json!({"path": "src/lib.rs", "content": "updated"})),
            ToolResult::ok("fs__write", json!({"path": "src/lib.rs"})),
            ToolEffects {
                kind: ToolEffectKind::Write,
                primary_path: Some("src/lib.rs".into()),
                edited_paths: vec!["src/lib.rs".into()],
                command: None,
            },
        );

        let draft = EvidenceDraft::from_tool_execution_record(&record);

        assert_eq!(draft.evidence_kind, EvidenceKind::Change);
        assert_eq!(
            draft.source,
            EvidenceSource::File {
                path: "src/lib.rs".into(),
                start_line: None,
                end_line: None,
            }
        );
        assert!(draft.tags.contains(&"src/lib.rs".to_string()));
    }

    #[test]
    fn from_tool_execution_record_maps_validation_effect_to_validation_command_source() {
        let record = tool_execution_record(
            "shell__exec",
            Some(json!({"command": "cargo check"})),
            ToolResult::ok(
                "shell__exec",
                json!({"command": "cargo check", "status": 0, "stdout": "ok"}),
            ),
            ToolEffects {
                kind: ToolEffectKind::Validation,
                primary_path: None,
                edited_paths: vec![],
                command: Some("cargo check".into()),
            },
        );

        let draft = EvidenceDraft::from_tool_execution_record(&record);

        assert_eq!(draft.evidence_kind, EvidenceKind::Validation);
        assert_eq!(
            draft.source,
            EvidenceSource::Command {
                command: "cargo check".into(),
                status: Some(0),
            }
        );
    }

    #[test]
    fn from_tool_execution_record_maps_failed_effect_to_diagnostic_and_preserves_command() {
        let record = tool_execution_record(
            "shell__exec",
            Some(json!({"command": "cargo test evidence"})),
            ToolResult::err("shell__exec", "tests failed"),
            ToolEffects {
                kind: ToolEffectKind::Diagnostic,
                primary_path: None,
                edited_paths: vec![],
                command: Some("cargo test evidence".into()),
            },
        );

        let draft = EvidenceDraft::from_tool_execution_record(&record);

        assert_eq!(draft.evidence_kind, EvidenceKind::Diagnostic);
        assert_eq!(
            draft.source,
            EvidenceSource::Command {
                command: "cargo test evidence".into(),
                status: None,
            }
        );
        assert_eq!(draft.detail.as_deref(), Some("tests failed"));
    }

    #[test]
    fn from_tool_execution_record_maps_failed_validation_output_to_diagnostic() {
        let record = tool_execution_record(
            "shell__exec",
            Some(json!({"command": "cargo test evidence"})),
            ToolResult::ok(
                "shell__exec",
                json!({
                    "command": "cargo test evidence",
                    "status": 101,
                    "success": false,
                    "stdout": "",
                    "stderr": "test failed"
                }),
            ),
            ToolEffects {
                kind: ToolEffectKind::Diagnostic,
                primary_path: None,
                edited_paths: vec![],
                command: Some("cargo test evidence".into()),
            },
        );

        let draft = EvidenceDraft::from_tool_execution_record(&record);

        assert_eq!(draft.evidence_kind, EvidenceKind::Diagnostic);
        assert_eq!(draft.title, "shell__exec failed");
        assert_eq!(draft.summary, "command failed: cargo test evidence");
        assert_eq!(draft.detail.as_deref(), Some("test failed"));
        assert_eq!(
            draft.source,
            EvidenceSource::Command {
                command: "cargo test evidence".into(),
                status: Some(101),
            }
        );
    }

    #[test]
    fn from_tool_execution_record_uses_subagent_provenance_source() {
        let record = tool_execution_record(
            "agent__explore",
            Some(json!({"task": "inspect"})),
            ToolResult::ok(
                "agent__explore",
                json!({
                    "run_id": "run-1",
                    "child_session_id": "child-1",
                    "status": "completed",
                    "summary": "done"
                }),
            ),
            ToolEffects {
                kind: ToolEffectKind::Read,
                primary_path: None,
                edited_paths: vec![],
                command: None,
            },
        );

        let draft = EvidenceDraft::from_tool_execution_record(&record);

        assert_eq!(
            draft.source,
            EvidenceSource::Subagent {
                run_id: "run-1".into(),
                child_session_id: "child-1".into(),
                source_session_id: "child-1".into(),
                parent_tool: "agent__explore".into(),
                parent_turn_id: None,
                parent_session_id: None,
            }
        );
        assert!(draft.tags.contains(&"run-1".to_string()));
        assert!(draft.tags.contains(&"child-1".to_string()));
    }

    #[test]
    fn subagent_compact_line_keeps_parent_context_compact() {
        let record = EvidenceRecord {
            id: "ev-subagent".into(),
            sequence: 7,
            timestamp_ms: 0,
            evidence_kind: EvidenceKind::Decision,
            title: "subagent fixer result".into(),
            summary: "implemented bounded fix".into(),
            detail: Some(
                "{\"raw\":\"very large structured payload\",\"full_summary\":\"hidden\"}".into(),
            ),
            source: EvidenceSource::Subagent {
                run_id: "run-7".into(),
                child_session_id: "child-session-1234567890".into(),
                source_session_id: "child-session-1234567890".into(),
                parent_tool: "agent__fixer".into(),
                parent_turn_id: Some("turn-4".into()),
                parent_session_id: Some("parent-session".into()),
            },
            tags: vec!["subagent_result".into()],
        };

        let compact = record.compact_line();
        assert!(compact.contains("/child child-session-1"), "{compact}");
        assert!(
            !compact.contains("very large structured payload"),
            "{compact}"
        );
        assert!(!compact.contains("full_summary"), "{compact}");
    }
}
