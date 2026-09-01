use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(test)]
use crate::agent::{
    AutoContinueState, ContextCompactionEvent, ConversationMessage, ConversationRole,
    LlmRequestTelemetry, LlmRequestTelemetryPhase, TodoItem, ToolExecutionSummaryEvent,
    TurnFinalizedEvent, TurnStartedEvent, ValidationAdvisory,
};
#[cfg(test)]
use crate::context_tree::{
    ContextBlockRef, ContextNodeId, ContextNodeStatus, ContextSourceRef, ContextTreeOp,
};
#[cfg(test)]
use crate::evidence::{
    EvidenceDraft, EvidenceKind, EvidenceRecord, EvidenceSource, restore_evidence_records,
};
#[cfg(test)]
use crate::request_builder::{HistoryItem, HistoryToolCall};
#[cfg(test)]
use crate::tool::ToolResult;
#[cfg(test)]
use crate::user_content::UserMessageContent;
#[cfg(test)]
use serde_json::Value;
#[cfg(test)]
use std::fs::OpenOptions;
#[cfg(test)]
use std::io::{self, Write};

mod model;
mod session_index;

pub(crate) use model::TranscriptFileFingerprint;
pub use model::{
    HistoryNavigationOperation, TranscriptAssistantTurn, TranscriptEvent, TranscriptRecord,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum InternalContinuationSource {
    #[default]
    Legacy,
    AutoContinue,
    StreamRecovery,
    LogicalCheckpoint,
    SubagentCompletion,
}

/// Durable, provider-facing facts retained when a logical segment is closed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LogicalCheckpointRetainedKindV1 {
    UserRequirement,
    UnresolvedError,
    FileWriteFact,
    TestResult,
    Permission,
    Commit,
    WorkflowState,
}

impl LogicalCheckpointRetainedKindV1 {
    pub(crate) fn rank(self) -> u8 {
        match self {
            Self::UserRequirement => 0,
            Self::UnresolvedError => 1,
            Self::FileWriteFact => 2,
            Self::TestResult => 3,
            Self::Permission => 4,
            Self::Commit => 5,
            Self::WorkflowState => 6,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum LogicalCheckpointAuditSourceV1 {
    TranscriptSpan {
        start_sequence: u64,
        end_sequence: u64,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LogicalCheckpointRetainedItemV1 {
    pub kind: LogicalCheckpointRetainedKindV1,
    pub title: String,
    pub detail: String,
    pub audit_source: LogicalCheckpointAuditSourceV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LogicalCheckpointSourceSpanV1 {
    pub start_sequence: u64,
    pub end_sequence: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LogicalCheckpointEventV1 {
    pub schema_version: u32,
    pub checkpoint_id: String,
    pub turn_id: u64,
    pub previous_segment_id: u64,
    pub segment_id: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_checkpoint_id: Option<String>,
    pub boundary_sequence: u64,
    pub context_scope_revision: u64,
    pub covered_source_spans: Vec<LogicalCheckpointSourceSpanV1>,
    pub retained_items: Vec<LogicalCheckpointRetainedItemV1>,
}

#[path = "transcript/transcript_projection.rs"]
pub(crate) mod transcript_projection;

mod journal;
mod recorder;
mod restore;

#[cfg(test)]
pub use journal::read_records_with_fingerprint;
#[cfg(test)]
pub(crate) use journal::{
    JOURNAL_SCHEMA_VERSION, JournalRecordEnvelope, JournalScope, JournalSink,
    JournalTransactionCommit, LEGACY_JOURNAL_SCHEMA_VERSION,
    content_tail_is_uncommitted_transaction, journal_scope_for, parse_records_content,
    scan_transcript_content, serialize_journal_record, transcript_file_fingerprint,
    transcript_records_match, validate_journal_entries,
};
pub(crate) use journal::{
    ParsedJournalLine, parse_journal_line, repair_partial_tail, transaction_fields,
};
pub use journal::{
    read_records, read_records_allow_partial_tail, read_resumable_records_with_fingerprint,
};
pub use recorder::TranscriptRecorder;
#[cfg(test)]
pub(crate) use recorder::{ActiveContextExperiment, RecorderHealth};
pub(crate) use recorder::{
    ContextScopeState, ROOT_CONTEXT_BRANCH_ID, format_context_experiment_return,
    render_checkpoint_continuation_v1, render_checkpoint_v1, sync_recorder_branch,
};
pub(crate) use restore::{
    append_history_item_from_transcript_record, history_item_to_conversation_message,
};
#[cfg(test)]
pub(crate) use restore::{
    restore_compacted_conversation_messages, restore_conversation_messages, restore_job_board,
    restore_latest_expert_models_for_cursor, restore_max_turn_id, restore_runtime_snapshot,
    restore_session_evidence, restore_session_history, restore_session_protocol_frames,
};
pub use restore::{
    restore_latest_auto_continue_state, restore_latest_expert_models, restore_latest_model,
    restore_latest_permission_mode, restore_latest_reasoning_effort, restore_latest_todo_snapshot,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionSummary {
    pub session_id: String,
    pub record_count: usize,
    pub first_timestamp_ms: Option<u128>,
    pub last_timestamp_ms: Option<u128>,
    pub model: Option<String>,
    pub title: Option<String>,
    pub last_user_summary: Option<String>,
    pub last_assistant_summary: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChildSessionSummary {
    pub parent_session_id: String,
    pub parent_run_id: String,
    pub child_session_id: String,
    pub agent_name: String,
    pub status: String,
    pub summary: String,
    pub timestamp_ms: u128,
    /// Stable pool slot assigned at first create (1-based). Never renumbered.
    pub pool_ordinal: u32,
}

pub fn sort_child_session_summaries(children: &mut [ChildSessionSummary]) {
    children.sort_by(|left, right| {
        // Prefer stable pool ordinal; fall back to time/id for legacy rows (ordinal 0).
        match (left.pool_ordinal == 0, right.pool_ordinal == 0) {
            (false, false) => left
                .pool_ordinal
                .cmp(&right.pool_ordinal)
                .then_with(|| left.timestamp_ms.cmp(&right.timestamp_ms))
                .then_with(|| left.child_session_id.cmp(&right.child_session_id)),
            (false, true) => std::cmp::Ordering::Less,
            (true, false) => std::cmp::Ordering::Greater,
            (true, true) => left
                .timestamp_ms
                .cmp(&right.timestamp_ms)
                .then_with(|| left.child_session_id.cmp(&right.child_session_id)),
        }
    });
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JobBoardEntry {
    pub active: bool,
    pub run_id: String,
    pub child_session_id: String,
    pub agent_name: String,
    pub status: String,
    pub summary: String,
}

pub(crate) fn project_subagent_jobs(
    base_dir: impl AsRef<Path>,
    parent_records: &[TranscriptRecord],
) -> Result<Vec<crate::subagent::SubagentJob>> {
    let ordinals = parent_records
        .iter()
        .filter_map(|record| match &record.event {
            TranscriptEvent::SubagentStarted {
                run_id,
                pool_ordinal,
                ..
            } => Some((run_id.clone(), *pool_ordinal)),
            _ => None,
        })
        .collect::<std::collections::HashMap<_, _>>();
    Ok(
        transcript_projection::project_job_board(&child_sessions_dir(base_dir), parent_records)?
            .into_iter()
            .map(|job| crate::subagent::SubagentJob {
                active: job.active,
                pool_ordinal: ordinals.get(&job.run_id).copied().unwrap_or(0),
                run_id: job.run_id,
                child_session_id: job.child_session_id,
                agent_name: job.agent_name,
                status: job.status,
                summary: job.summary,
            })
            .collect(),
    )
}

pub fn list_sessions(base_dir: impl AsRef<Path>) -> Result<Vec<SessionSummary>> {
    let base_dir = base_dir.as_ref();

    if !base_dir.exists() {
        return Ok(Vec::new());
    }

    session_index::list_sessions_with_index(base_dir, summarize_session_file)
}

#[derive(Default)]
struct SessionSummaryAcc {
    record_count: usize,
    first_timestamp_ms: Option<u128>,
    last_timestamp_ms: Option<u128>,
    model: Option<String>,
    title: Option<String>,
    last_user_summary: Option<String>,
    last_assistant_summary: Option<String>,
    has_content: bool,
}

fn fold_session_summary(acc: &mut SessionSummaryAcc, record: &TranscriptRecord) {
    acc.record_count = acc.record_count.saturating_add(1);
    if acc.first_timestamp_ms.is_none() {
        acc.first_timestamp_ms = Some(record.timestamp_ms);
    }
    acc.last_timestamp_ms = Some(record.timestamp_ms);
    match &record.event {
        TranscriptEvent::SessionStarted { model } => acc.model = Some(model.clone()),
        TranscriptEvent::ModelChanged { new_model, .. } => acc.model = Some(new_model.clone()),
        TranscriptEvent::SessionTitle { title } => acc.title = Some(title.clone()),
        TranscriptEvent::UserMessage { content } => {
            acc.has_content = true;
            acc.last_user_summary = Some(summarize_text(&content.display_text()));
        }
        TranscriptEvent::AssistantTurn(turn) => {
            acc.has_content = true;
            if let Some(text) = turn.text.as_deref() {
                acc.last_assistant_summary = Some(summarize_text(text));
            }
        }
        TranscriptEvent::AssistantMessage { content } => {
            acc.has_content = true;
            acc.last_assistant_summary = Some(summarize_text(content));
        }
        event if event.is_session_content() => acc.has_content = true,
        _ => {}
    }
}

/// Lightweight listing scan: stream lines, accept committed transactions without
/// digest/revision hardening, and never retain the full record list.
fn summarize_session_file(path: &Path, session_id: String) -> Result<Option<SessionSummary>> {
    let file = File::open(path)
        .with_context(|| format!("failed to read transcript {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut acc = SessionSummaryAcc::default();
    let mut pending: Option<Vec<TranscriptRecord>> = None;

    for line in reader.lines() {
        let line = line.with_context(|| format!("failed to read transcript {}", path.display()))?;
        if line.trim().is_empty() {
            continue;
        }

        match parse_journal_line(&line) {
            Ok(ParsedJournalLine::Record(entry)) => {
                let transactional = transaction_fields(&entry.envelope)?.is_some();
                if transactional {
                    pending.get_or_insert_with(Vec::new).push(entry.record);
                } else {
                    // A non-transactional record closes any incomplete transaction tail.
                    pending = None;
                    fold_session_summary(&mut acc, &entry.record);
                }
            }
            Ok(ParsedJournalLine::Commit(_)) => {
                if let Some(entries) = pending.take() {
                    for record in entries {
                        fold_session_summary(&mut acc, &record);
                    }
                }
            }
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("failed to parse transcript {}", path.display()));
            }
        }
    }

    if !acc.has_content {
        return Ok(None);
    }

    Ok(Some(SessionSummary {
        session_id,
        record_count: acc.record_count,
        first_timestamp_ms: acc.first_timestamp_ms,
        last_timestamp_ms: acc.last_timestamp_ms,
        model: acc.model,
        title: acc.title,
        last_user_summary: acc.last_user_summary,
        last_assistant_summary: acc.last_assistant_summary,
    }))
}

pub fn list_child_sessions_for_parent(
    base_dir: impl AsRef<Path>,
    parent_records: &[TranscriptRecord],
) -> Vec<ChildSessionSummary> {
    transcript_projection::project_child_session_summaries(
        &child_sessions_dir(base_dir),
        parent_records,
    )
}

#[cfg(test)]
pub fn read_child_session_records(
    base_dir: impl AsRef<Path>,
    child_session_id: &str,
) -> Result<Vec<TranscriptRecord>> {
    read_records(session_path(
        &child_sessions_dir(base_dir),
        child_session_id,
    ))
}

pub(crate) fn read_child_session_records_allow_partial_tail(
    base_dir: impl AsRef<Path>,
    child_session_id: &str,
) -> Result<Vec<TranscriptRecord>> {
    read_records_allow_partial_tail(session_path(
        &child_sessions_dir(base_dir),
        child_session_id,
    ))
}

pub fn has_session_content(records: &[TranscriptRecord]) -> bool {
    records
        .iter()
        .any(|record| record.event.is_session_content())
}

pub fn transcript_has_user_message(records: &[TranscriptRecord]) -> bool {
    records
        .iter()
        .any(|record| matches!(record.event, TranscriptEvent::UserMessage { .. }))
}

pub fn transcript_has_session_title(records: &[TranscriptRecord]) -> bool {
    records
        .iter()
        .any(|record| matches!(record.event, TranscriptEvent::SessionTitle { .. }))
}

impl TranscriptEvent {
    pub(crate) fn is_context_branch_metadata(&self) -> bool {
        matches!(
            self,
            Self::ContextBranchCreated { .. }
                | Self::ContextBranchSummary { .. }
                | Self::ContextCheckout { .. }
                | Self::HistoryNavigation { .. }
                | Self::ContextExperimentStarted { .. }
                | Self::ContextNodeCreated { .. }
                | Self::ContextNodeLifecycle { .. }
                | Self::ContextViewOperationMetadata { .. }
                | Self::ContextSummaryArtifactMetadata { .. }
                | Self::FoldedOutputMetadata { .. }
        )
    }

    /// Events represented as selectable transcript history entries. Metadata
    /// remains journal-visible but is deliberately absent from the user tree.
    pub(crate) fn is_session_history_entry(&self) -> bool {
        matches!(
            self,
            Self::UserMessage { .. }
                | Self::AssistantTurn(_)
                | Self::AssistantMessage { .. }
                | Self::ReasoningMessage { .. }
                | Self::AssistantToolCallBatch { .. }
                | Self::ToolCallStarted { .. }
                | Self::ToolCallFinished { .. }
                | Self::ToolCallCancelled { .. }
                | Self::InternalContinuation { .. }
                | Self::ContextCompaction(_)
                | Self::LogicalCheckpoint(_)
                | Self::Error { .. }
        )
    }

    pub(crate) fn is_session_content(&self) -> bool {
        matches!(
            self,
            Self::UserMessage { .. }
                | Self::AssistantTurn(_)
                | Self::AssistantMessage { .. }
                | Self::ReasoningMessage { .. }
                | Self::ToolCallStarted { .. }
                | Self::ToolCallFinished { .. }
                | Self::ToolCallCancelled { .. }
                | Self::PermissionDecision { .. }
                | Self::TodoSnapshot { .. }
                | Self::AutoContinueChanged { .. }
                | Self::Error { .. }
                | Self::Evidence { .. }
                | Self::ContextCompaction(..)
                | Self::ContextExperimentReturned { .. }
        )
    }
}

pub fn remove_empty_session_file(path: impl AsRef<Path>) -> Result<bool> {
    let path = path.as_ref();
    let records = read_records(path)?;
    if has_session_content(&records) {
        return Ok(false);
    }

    let session_id = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .map(str::to_string);
    fs::remove_file(path).with_context(|| {
        format!(
            "failed to remove empty session transcript '{}'",
            path.display()
        )
    })?;
    if let (Some(base_dir), Some(session_id)) = (path.parent(), session_id) {
        session_index::remove_session(base_dir, &session_id);
    }
    Ok(true)
}

pub fn resolve_session_id(
    sessions: &[SessionSummary],
    query: &str,
) -> std::result::Result<String, Vec<String>> {
    let matches = sessions
        .iter()
        .filter(|session| session.session_id.starts_with(query))
        .map(|session| session.session_id.clone())
        .collect::<Vec<_>>();

    if matches.len() == 1 {
        Ok(matches[0].clone())
    } else {
        Err(matches)
    }
}

pub fn child_sessions_dir(base_dir: impl AsRef<Path>) -> PathBuf {
    base_dir.as_ref().join("children")
}

fn session_path(base_dir: &Path, session_id: &str) -> PathBuf {
    base_dir.join(format!("{session_id}.jsonl"))
}

fn generate_session_id() -> String {
    static NEXT_SESSION_ID: AtomicU64 = AtomicU64::new(0);
    let suffix = NEXT_SESSION_ID.fetch_add(1, Ordering::SeqCst);
    format!("{}-{}-{suffix}", unix_timestamp_ms(), process::id())
}

fn unix_timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

pub(crate) fn summarize_text(content: &str) -> String {
    let single_line = content.split_whitespace().collect::<Vec<_>>().join(" ");
    truncate_text(&single_line, 80)
}

/// Return a bounded safe provider identifier, hashing hostile values so no
/// provider payload, URL, or control-delimited secret enters the journal.
fn sanitize_opaque_identifier(value: &str) -> String {
    const MAX_LEN: usize = 128;
    if !value.is_empty()
        && value.len() <= MAX_LEN
        && !value.contains("://")
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.' | b':'))
    {
        return value.into();
    }
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    value.hash(&mut hasher);
    format!("opaque-{:016x}", hasher.finish())
}

fn truncate_text(content: &str, max_chars: usize) -> String {
    let mut truncated = content.chars().take(max_chars).collect::<String>();
    if content.chars().count() > max_chars {
        truncated.push('…');
    }
    truncated
}

#[cfg(test)]
mod session_title_tests {
    use super::*;
}

#[cfg(test)]
mod tests;
