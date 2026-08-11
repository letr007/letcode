use anyhow::{Context, Result, anyhow, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::agent::subagent_evidence_parent_tool;
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
use crate::evidence::restore_evidence_records;
use crate::evidence::{
    EvidenceDraft, EvidenceKind, EvidenceRecord, EvidenceSource, evidence_id_for_sequence,
};
use crate::request_builder::{HistoryItem, HistoryToolCall};
use crate::subagent::StructuredSubagentResult;
use crate::tool::ToolResult;
use crate::user_content::UserMessageContent;

mod model;
mod session_index;

pub use model::{HistoryNavigationOperation, TranscriptEvent, TranscriptRecord};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TranscriptFileFingerprint {
    content_len: usize,
    content_digest: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InternalContinuationSource {
    Legacy,
    AutoContinue,
    StreamRecovery,
    LogicalCheckpoint,
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

#[derive(Serialize)]
struct CheckpointHeaderRenderer<'a> {
    schema_version: u32,
    checkpoint_id: &'a str,
    turn_id: u64,
    previous_segment_id: u64,
    segment_id: u64,
}

#[derive(Serialize)]
struct CheckpointItemRenderer<'a> {
    kind: LogicalCheckpointRetainedKindV1,
    title: &'a str,
    detail: &'a str,
    audit_source: &'a LogicalCheckpointAuditSourceV1,
}

pub(crate) fn render_checkpoint_v1(event: &LogicalCheckpointEventV1) -> Result<String> {
    let mut lines = vec![
        "[logical-checkpoint-v1]".to_string(),
        serde_json::to_string(&CheckpointHeaderRenderer {
            schema_version: event.schema_version,
            checkpoint_id: &event.checkpoint_id,
            turn_id: event.turn_id,
            previous_segment_id: event.previous_segment_id,
            segment_id: event.segment_id,
        })?,
        "[retained-items]".to_string(),
    ];
    for item in &event.retained_items {
        lines.push(serde_json::to_string(&CheckpointItemRenderer {
            kind: item.kind,
            title: &item.title,
            detail: &item.detail,
            audit_source: &item.audit_source,
        })?);
    }
    Ok(lines.join("\n"))
}

pub(crate) fn render_checkpoint_continuation_v1(event: &LogicalCheckpointEventV1) -> String {
    format!(
        "Resume the same user turn from logical checkpoint {}. Treat the retained checkpoint context above as authoritative; retired sources are audit-only and are not directly openable.",
        event.checkpoint_id
    )
}

impl Default for InternalContinuationSource {
    fn default() -> Self {
        Self::Legacy
    }
}

#[path = "transcript/transcript_projection.rs"]
pub(crate) mod transcript_projection;

const JOURNAL_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum JournalScope {
    Global,
    Branch,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct JournalRecordV1 {
    schema_version: u32,
    event_id: String,
    scope: JournalScope,
    base_revision: u64,
    resulting_revision: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    transaction_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    transaction_index: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    transaction_count: Option<usize>,
    #[serde(flatten)]
    record: TranscriptRecord,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct JournalTransactionCommitV1 {
    schema_version: u32,
    journal_entry: String,
    transaction_id: String,
    transaction_count: usize,
    base_revision: u64,
    resulting_revision: u64,
    payload_length: usize,
    payload_digest: String,
}

const JOURNAL_TRANSACTION_COMMIT: &str = "transaction_commit";

trait JournalSink: Send {
    fn write_all(&mut self, bytes: &[u8]) -> io::Result<()>;
    fn flush(&mut self) -> io::Result<()>;
    fn sync_data(&mut self) -> io::Result<()>;
}

struct FileJournalSink(File);

impl JournalSink for FileJournalSink {
    fn write_all(&mut self, bytes: &[u8]) -> io::Result<()> {
        self.0.write_all(bytes)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.0.flush()
    }

    fn sync_data(&mut self) -> io::Result<()> {
        self.0.sync_data()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RecorderHealth {
    Healthy,
    Poisoned,
}

pub(crate) const ROOT_CONTEXT_BRANCH_ID: &str = "main";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActiveContextExperiment {
    pub branch_id: String,
    pub parent_branch_id: String,
    pub base_sequence: u64,
    pub writes_observed: bool,
}

#[derive(Debug, Clone, Default)]
pub struct ContextScopeState {
    pub active_experiment: Option<ActiveContextExperiment>,
}

pub struct TranscriptRecorder {
    session_id: String,
    #[allow(dead_code)]
    path: PathBuf,
    sink: Box<dyn JournalSink>,
    sequence: u64,
    health: RecorderHealth,
    current_context_branch_id: Option<String>,
    context_scope_state: Arc<Mutex<ContextScopeState>>,
    reasoning_started_at: std::collections::HashMap<String, std::time::Instant>,
}

impl TranscriptRecorder {
    pub fn create(base_dir: impl AsRef<Path>) -> Result<Self> {
        fs::create_dir_all(base_dir.as_ref())?;

        let session_id = generate_session_id();
        let file_path = session_path(base_dir.as_ref(), &session_id);
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&file_path)?;

        Ok(Self {
            session_id,
            path: file_path,
            sink: Box::new(FileJournalSink(file)),
            sequence: 0,
            health: RecorderHealth::Healthy,
            current_context_branch_id: None,
            context_scope_state: Arc::new(Mutex::new(ContextScopeState::default())),
            reasoning_started_at: std::collections::HashMap::new(),
        })
    }

    /// Open an existing session transcript for append (takeover / resume writes).
    pub fn open(base_dir: impl AsRef<Path>, session_id: impl Into<String>) -> Result<Self> {
        Self::open_existing(base_dir, &session_id.into())
    }

    pub fn open_existing(base_dir: impl AsRef<Path>, session_id: &str) -> Result<Self> {
        let base_dir = base_dir.as_ref();
        fs::create_dir_all(base_dir)?;
        let file_path = session_path(base_dir, session_id);
        let (records, fingerprint) = read_records_with_fingerprint(&file_path)?;
        Self::open_existing_with_records_at_fingerprint(
            base_dir,
            session_id,
            &records,
            &fingerprint,
        )
    }

    /// Open an existing session transcript for append using records already
    /// loaded from that transcript. The transcript must still match the loaded
    /// records, otherwise resume fails instead of appending from a stale
    /// sequence frontier.
    #[cfg(test)]
    pub fn open_existing_with_records(
        base_dir: impl AsRef<Path>,
        session_id: &str,
        records: &[TranscriptRecord],
    ) -> Result<Self> {
        let base_dir = base_dir.as_ref();
        fs::create_dir_all(base_dir)?;
        let file_path = session_path(base_dir, session_id);
        let content = fs::read_to_string(&file_path)
            .with_context(|| format!("failed to read transcript {}", file_path.display()))?;
        Self::open_existing_with_records_and_content(base_dir, session_id, records, &content)
    }

    pub(crate) fn open_existing_with_records_at_fingerprint(
        base_dir: impl AsRef<Path>,
        session_id: &str,
        records: &[TranscriptRecord],
        fingerprint: &TranscriptFileFingerprint,
    ) -> Result<Self> {
        let base_dir = base_dir.as_ref();
        fs::create_dir_all(base_dir)?;
        let file_path = session_path(base_dir, session_id);
        let content = fs::read_to_string(&file_path)
            .with_context(|| format!("failed to read transcript {}", file_path.display()))?;
        ensure!(
            transcript_file_fingerprint(&content) == *fingerprint,
            "transcript changed after records were loaded; retry resume"
        );
        ensure!(
            !content_tail_is_uncommitted_transaction(&file_path, &content)?,
            "transcript has an uncommitted transaction tail and cannot safely accept new records"
        );
        Self::open_existing_with_validated_records(base_dir, session_id, records)
    }

    #[cfg(test)]
    fn open_existing_with_records_and_content(
        base_dir: &Path,
        session_id: &str,
        records: &[TranscriptRecord],
        content: &str,
    ) -> Result<Self> {
        let file_path = session_path(base_dir, session_id);
        ensure!(
            !content_tail_is_uncommitted_transaction(&file_path, content)?,
            "transcript has an uncommitted transaction tail and cannot safely accept new records"
        );
        ensure!(
            records.iter().all(|record| record.session_id == session_id),
            "transcript contains records for a different session"
        );
        let current_records = parse_records_content(&file_path, content, false)?;
        ensure!(
            transcript_records_match(&current_records, records)?,
            "transcript changed after records were loaded; retry resume"
        );
        Self::open_existing_with_validated_records(base_dir, session_id, records)
    }

    fn open_existing_with_validated_records(
        base_dir: &Path,
        session_id: &str,
        records: &[TranscriptRecord],
    ) -> Result<Self> {
        ensure!(
            records.iter().all(|record| record.session_id == session_id),
            "transcript contains records for a different session"
        );
        let file_path = session_path(base_dir, session_id);
        let sequence = records
            .iter()
            .map(|record| record.sequence)
            .max()
            .unwrap_or(0);
        let file = OpenOptions::new().append(true).open(&file_path)?;

        let context_scope_state = Arc::new(Mutex::new(reconstruct_context_scope_state(records)?));

        Ok(Self {
            session_id: session_id.to_string(),
            path: file_path,
            sink: Box::new(FileJournalSink(file)),
            sequence,
            health: RecorderHealth::Healthy,
            current_context_branch_id: None,
            context_scope_state,
            reasoning_started_at: std::collections::HashMap::new(),
        })
    }

    #[allow(dead_code)]
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    #[allow(dead_code)]
    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn record_session_started(&mut self, model: impl Into<String>) -> Result<()> {
        self.append(TranscriptEvent::SessionStarted {
            model: model.into(),
        })
    }

    pub fn record_expert_model_changed(
        &mut self,
        agent_name: impl Into<String>,
        model: impl Into<String>,
    ) -> Result<()> {
        self.append_metadata(TranscriptEvent::ExpertModelChanged {
            agent_name: agent_name.into(),
            model: model.into(),
        })
    }

    pub fn record_model_changed(
        &mut self,
        previous_model: impl Into<String>,
        new_model: impl Into<String>,
    ) -> Result<()> {
        self.append(TranscriptEvent::ModelChanged {
            previous_model: previous_model.into(),
            new_model: new_model.into(),
        })
    }

    pub fn set_current_context_branch_id(&mut self, branch_id: Option<String>) {
        self.current_context_branch_id = branch_id;
    }

    pub fn current_context_branch_id(&self) -> Option<&str> {
        self.current_context_branch_id.as_deref()
    }

    /// Adopts a projected legacy branch as the ordinary append scope when a
    /// session is resumed. This is deliberately in-memory only: legacy
    /// experiment state is compatibility metadata, not a live experiment.
    pub fn adopt_legacy_linear_branch(&mut self, selected_branch_id: &str) -> Result<()> {
        let active_experiment = self.active_context_experiment();
        if let Some(experiment) = active_experiment {
            ensure!(
                experiment.branch_id == selected_branch_id,
                "cannot resume selected branch '{}' while legacy experiment branch '{}' is unreturned",
                selected_branch_id,
                experiment.branch_id
            );
        }

        self.current_context_branch_id =
            (selected_branch_id != ROOT_CONTEXT_BRANCH_ID).then(|| selected_branch_id.to_string());
        // An unreturned legacy experiment must not restore a live Agent lock:
        // normal subsequent turns own their usual finalization lifecycle.
        self.set_active_context_experiment(None);
        Ok(())
    }

    pub fn context_scope_state(&self) -> Arc<Mutex<ContextScopeState>> {
        Arc::clone(&self.context_scope_state)
    }

    pub fn active_context_experiment(&self) -> Option<ActiveContextExperiment> {
        self.context_scope_state
            .lock()
            .ok()
            .and_then(|state| state.active_experiment.clone())
    }

    #[cfg(test)]
    pub fn record_context_branch_created(
        &mut self,
        branch_id: impl Into<String>,
        parent_branch_id: impl Into<String>,
        base_sequence: u64,
        label: Option<String>,
    ) -> Result<()> {
        self.append_metadata(TranscriptEvent::ContextBranchCreated {
            branch_id: branch_id.into(),
            parent_branch_id: parent_branch_id.into(),
            base_sequence,
            label,
        })
    }

    #[cfg(test)]
    pub fn record_context_branch_summary(
        &mut self,
        branch_id: impl Into<String>,
        leaf_sequence: u64,
        summary: impl Into<String>,
    ) -> Result<()> {
        self.append_metadata(TranscriptEvent::ContextBranchSummary {
            branch_id: branch_id.into(),
            leaf_sequence,
            summary: summary.into(),
        })
    }

    #[cfg(test)]
    pub fn record_context_checkout(
        &mut self,
        branch_id: impl Into<String>,
        leaf_sequence: u64,
    ) -> Result<()> {
        self.append_metadata(TranscriptEvent::ContextCheckout {
            branch_id: branch_id.into(),
            leaf_sequence,
        })
    }

    /// Atomically records the internal checkout that makes a history selection
    /// the active append path and its durable navigation state.
    pub fn record_history_navigation_transaction(
        &mut self,
        branch_id: String,
        parent_branch_id: String,
        target_sequence: u64,
        operation: HistoryNavigationOperation,
        redo_stack: Vec<u64>,
    ) -> Result<()> {
        let events = Self::history_navigation_events(
            branch_id,
            parent_branch_id,
            target_sequence,
            operation,
            redo_stack,
        );
        self.append_transaction(events)
    }

    pub(crate) fn preflight_history_navigation_transaction(
        &self,
        branch_id: String,
        parent_branch_id: String,
        target_sequence: u64,
        operation: HistoryNavigationOperation,
        redo_stack: Vec<u64>,
    ) -> Result<()> {
        let events = Self::history_navigation_events(
            branch_id,
            parent_branch_id,
            target_sequence,
            operation,
            redo_stack,
        );
        self.prepare_transaction_buffer(&events).map(|_| ())
    }

    fn history_navigation_events(
        branch_id: String,
        parent_branch_id: String,
        target_sequence: u64,
        operation: HistoryNavigationOperation,
        redo_stack: Vec<u64>,
    ) -> Vec<(TranscriptEvent, Option<String>)> {
        vec![
            (
                TranscriptEvent::ContextBranchCreated {
                    branch_id: branch_id.clone(),
                    parent_branch_id,
                    base_sequence: target_sequence,
                    label: None,
                },
                None,
            ),
            (
                TranscriptEvent::ContextCheckout {
                    branch_id,
                    leaf_sequence: target_sequence,
                },
                None,
            ),
            (
                TranscriptEvent::HistoryNavigation {
                    operation,
                    target_sequence,
                    redo_stack,
                    redo_target_sequence: None,
                },
                None,
            ),
        ]
    }

    #[cfg(test)]
    pub fn record_context_experiment_returned(
        &mut self,
        branch_id: impl Into<String>,
        parent_branch_id: impl Into<String>,
        base_sequence: u64,
        outcome: impl Into<String>,
        summary: impl Into<String>,
        next_action: Option<String>,
        had_writes: bool,
    ) -> Result<()> {
        self.append(TranscriptEvent::ContextExperimentReturned {
            branch_id: branch_id.into(),
            parent_branch_id: parent_branch_id.into(),
            base_sequence,
            outcome: outcome.into(),
            summary: summary.into(),
            next_action,
            had_writes,
        })
    }

    #[cfg(test)]
    pub fn record_context_node_created(
        &mut self,
        node_id: impl Into<String>,
        parent_node_id: impl Into<String>,
        label: Option<String>,
        purpose: Option<String>,
        block_ref: Option<ContextBlockRef>,
        source_ref: Option<ContextSourceRef>,
    ) -> Result<()> {
        self.append_metadata(TranscriptEvent::ContextNodeCreated {
            node_id: node_id.into(),
            parent_node_id: Some(parent_node_id.into()),
            label,
            purpose,
            block_ref,
            source_ref,
        })
    }

    #[cfg(test)]
    pub fn record_context_node_activated(&mut self, node_id: impl Into<String>) -> Result<()> {
        self.activate_context_node(node_id)
    }

    #[cfg(test)]
    pub fn create_context_node(
        &mut self,
        node_id: impl Into<String>,
        parent_node_id: impl Into<String>,
        label: Option<String>,
        purpose: Option<String>,
        block_ref: Option<ContextBlockRef>,
        source_ref: Option<ContextSourceRef>,
    ) -> Result<()> {
        let node_id = node_id.into();
        let parent_node_id = parent_node_id.into();
        self.validate_context_tree_op(ContextTreeOp::CreateNode {
            node_id: ContextNodeId::new(node_id.clone())?,
            parent_node_id: Some(ContextNodeId::new(parent_node_id.clone())?),
            label: label.clone(),
            purpose: purpose.clone(),
            block_ref: block_ref.clone(),
            source_ref: source_ref.clone(),
        })?;
        self.record_context_node_created(
            node_id,
            parent_node_id,
            label,
            purpose,
            block_ref,
            source_ref,
        )
    }

    #[cfg(test)]
    pub fn activate_context_node(&mut self, node_id: impl Into<String>) -> Result<()> {
        self.record_context_node_status_change(node_id.into(), ContextNodeStatus::Active)
    }

    #[cfg(test)]
    pub fn suspend_context_node(&mut self, node_id: impl Into<String>) -> Result<()> {
        self.record_context_node_status_change(node_id.into(), ContextNodeStatus::Inactive)
    }

    #[cfg(test)]
    pub fn terminalize_context_node(&mut self, node_id: impl Into<String>) -> Result<()> {
        self.record_context_node_status_change(node_id.into(), ContextNodeStatus::Archived)
    }

    #[cfg(test)]
    pub fn record_context_node_lifecycle(
        &mut self,
        node_id: impl Into<String>,
        status: ContextNodeStatus,
    ) -> Result<()> {
        self.append_metadata(TranscriptEvent::ContextNodeLifecycle {
            node_id: node_id.into(),
            status,
        })
    }

    #[cfg(test)]
    pub fn record_context_view_operation_metadata(
        &mut self,
        operation: impl Into<String>,
        block_id: Option<String>,
        node_id: Option<String>,
        detail: Option<String>,
    ) -> Result<()> {
        self.append_metadata(TranscriptEvent::ContextViewOperationMetadata {
            operation: operation.into(),
            node_id,
            block_id,
            detail,
        })
    }

    #[cfg(test)]
    pub fn record_context_summary_artifact_metadata(
        &mut self,
        node_id: impl Into<String>,
        artifact_id: impl Into<String>,
        artifact_kind: impl Into<String>,
        version: Option<u32>,
        summary: Option<String>,
        source_node_id: Option<String>,
        source_block_id: Option<String>,
        source_start_sequence: Option<u64>,
        source_end_sequence: Option<u64>,
    ) -> Result<()> {
        self.append_metadata(TranscriptEvent::ContextSummaryArtifactMetadata {
            node_id: node_id.into(),
            artifact_id: artifact_id.into(),
            artifact_kind: artifact_kind.into(),
            version,
            summary,
            source_node_id,
            source_block_id,
            source_start_sequence,
            source_end_sequence,
        })
    }

    #[cfg(test)]
    pub fn record_context_tool_pending_metadata(
        &mut self,
        _tool_name: impl AsRef<str>,
        _ok: bool,
        _output: &ToolResult,
    ) -> Result<()> {
        // Removed with context tools. Kept as a no-op compatibility entrypoint.
        Ok(())
    }

    pub fn record_session_title(&mut self, title: impl Into<String>) -> Result<()> {
        self.append_metadata(TranscriptEvent::SessionTitle {
            title: title.into(),
        })
    }

    pub fn record_subagent_started(
        &mut self,
        run_id: impl Into<String>,
        parent_session_id: impl Into<String>,
        parent_run_id: impl Into<String>,
        child_session_id: impl Into<String>,
        agent_name: impl Into<String>,
        summary: impl Into<String>,
        pool_ordinal: u32,
    ) -> Result<()> {
        self.append(TranscriptEvent::SubagentStarted {
            run_id: run_id.into(),
            parent_session_id: parent_session_id.into(),
            parent_run_id: parent_run_id.into(),
            child_session_id: child_session_id.into(),
            agent_name: agent_name.into(),
            summary: summary.into(),
            pool_ordinal,
        })
    }

    pub fn record_subagent_lifecycle(
        &mut self,
        run_id: impl Into<String>,
        parent_session_id: impl Into<String>,
        parent_run_id: impl Into<String>,
        agent_name: impl Into<String>,
        status: impl Into<String>,
        detail: Option<String>,
    ) -> Result<()> {
        self.append(TranscriptEvent::SubagentLifecycle {
            run_id: run_id.into(),
            parent_session_id: parent_session_id.into(),
            parent_run_id: parent_run_id.into(),
            agent_name: agent_name.into(),
            status: status.into(),
            detail,
        })
    }

    #[cfg(test)]
    pub fn record_subagent_result(
        &mut self,
        run_id: impl Into<String>,
        parent_session_id: impl Into<String>,
        parent_run_id: impl Into<String>,
        child_session_id: impl Into<String>,
        agent_name: impl Into<String>,
        status: impl Into<String>,
        summary: impl Into<String>,
    ) -> Result<()> {
        self.record_subagent_result_structured(
            run_id,
            parent_session_id,
            parent_run_id,
            child_session_id,
            agent_name,
            status,
            summary,
            None,
        )
    }

    pub fn record_subagent_result_structured(
        &mut self,
        run_id: impl Into<String>,
        parent_session_id: impl Into<String>,
        parent_run_id: impl Into<String>,
        child_session_id: impl Into<String>,
        agent_name: impl Into<String>,
        status: impl Into<String>,
        summary: impl Into<String>,
        structured_result: Option<StructuredSubagentResult>,
    ) -> Result<()> {
        let run_id = run_id.into();
        let parent_session_id = parent_session_id.into();
        let parent_run_id = parent_run_id.into();
        let child_session_id = child_session_id.into();
        let agent_name = agent_name.into();
        self.append(TranscriptEvent::SubagentResult {
            run_id: run_id.clone(),
            parent_session_id: parent_session_id.clone(),
            parent_run_id: parent_run_id.clone(),
            child_session_id: child_session_id.clone(),
            agent_name: agent_name.clone(),
            status: status.into(),
            summary: summary.into(),
        })?;
        if let Some(result) = structured_result {
            let parent_tool =
                subagent_evidence_parent_tool(agent_name.as_str()).ok_or_else(|| {
                    anyhow!("subagent result recorded with unknown agent name: {agent_name}")
                })?;
            // System experts (e.g. auto-reviewer) are not user-delegated jobs — mark
            // reconciled immediately so they never pollute agent__reconcile context.
            let system_expert = parent_tool.starts_with("system__");
            let detail = serde_json::to_string(&result).ok();
            let evidence = EvidenceDraft {
                id: None,
                evidence_kind: EvidenceKind::Decision,
                title: format!("subagent {agent_name} result"),
                summary: result.summary.clone(),
                detail,
                source: EvidenceSource::Subagent {
                    run_id,
                    child_session_id: child_session_id.clone(),
                    source_session_id: child_session_id,
                    parent_tool,
                    parent_turn_id: Some(parent_run_id),
                    parent_session_id: Some(parent_session_id),
                },
                tags: if system_expert {
                    vec![
                        agent_name,
                        "subagent_result".into(),
                        "system_expert".into(),
                        "reconciled".into(),
                    ]
                } else {
                    vec![agent_name, "subagent_result".into(), "unreconciled".into()]
                },
            };
            self.record_evidence(evidence)?;
        }
        Ok(())
    }

    #[cfg(test)]
    pub fn record_subagent_reconciliation(
        &mut self,
        run_id: impl Into<String>,
        child_session_id: impl Into<String>,
        agent_name: impl Into<String>,
        parent_turn_id: impl Into<String>,
        summary: impl Into<String>,
    ) -> Result<EvidenceRecord> {
        let run_id = run_id.into();
        let child_session_id = child_session_id.into();
        let agent_name = agent_name.into();
        self.record_evidence(EvidenceDraft {
            id: None,
            evidence_kind: EvidenceKind::Decision,
            title: format!("reconciled subagent {agent_name} result"),
            summary: summary.into(),
            detail: None,
            source: EvidenceSource::Subagent {
                run_id,
                child_session_id: child_session_id.clone(),
                source_session_id: child_session_id,
                parent_tool: subagent_evidence_parent_tool(agent_name.as_str()).ok_or_else(
                    || {
                        anyhow!(
                            "subagent reconciliation recorded with unknown agent name: {agent_name}"
                        )
                    },
                )?,
                parent_turn_id: Some(parent_turn_id.into()),
                parent_session_id: Some(self.session_id.clone()),
            },
            tags: vec![
                agent_name,
                "subagent_reconciliation".into(),
                "reconciled".into(),
            ],
        })
    }

    pub fn record_user_message(&mut self, content: impl Into<String>) -> Result<()> {
        self.record_user_message_content(UserMessageContent::new(content, Vec::new()))
    }

    pub fn record_user_message_content(&mut self, content: UserMessageContent) -> Result<()> {
        self.append(TranscriptEvent::UserMessage { content })
    }

    pub fn record_assistant_message(&mut self, content: impl Into<String>) -> Result<()> {
        self.append(TranscriptEvent::AssistantMessage {
            content: content.into(),
        })
    }

    pub fn record_turn_started(&mut self, event: TurnStartedEvent) -> Result<()> {
        self.append(TranscriptEvent::TurnStarted(event))
    }

    pub fn record_llm_request_telemetry(&mut self, telemetry: LlmRequestTelemetry) -> Result<()> {
        let usage = telemetry.usage;
        let phase = match telemetry.phase {
            LlmRequestTelemetryPhase::Prepared => "prepared",
            LlmRequestTelemetryPhase::Completed => "completed",
            LlmRequestTelemetryPhase::Failed => "failed",
            LlmRequestTelemetryPhase::Interrupted => "interrupted",
        };
        self.append_durable(TranscriptEvent::LlmRequestTelemetry {
            version: 6,
            logical_request_id: truncate_text(&telemetry.logical_request_id, 128),
            turn_id: telemetry.turn_id,
            iteration: telemetry.iteration,
            attempt: telemetry.attempt,
            phase: phase.into(),
            error_class: telemetry.error_class.map(|value| value.as_str().into()),
            model: truncate_text(&telemetry.model, 256),
            protocol: match telemetry.protocol {
                crate::config::ApiProtocol::Responses => "responses",
                crate::config::ApiProtocol::Completions => "chat_completions",
            }
            .into(),
            context_window_tokens: telemetry.context_window_tokens,
            input_budget_tokens: telemetry.input_budget_tokens,
            estimated_request_tokens: telemetry.estimated_request_tokens,
            estimated_prelude_tokens: telemetry.estimated_prelude_tokens,
            estimated_protected_tokens: telemetry.estimated_protected_tokens,
            protected_safe_ceiling_tokens: telemetry.protected_safe_ceiling_tokens,
            protected_reserve_tokens: telemetry.protected_reserve_tokens,
            estimated_unaddressable_protected_tokens: telemetry
                .estimated_unaddressable_protected_tokens,
            estimated_retained_history_tokens: telemetry.estimated_retained_history_tokens,
            estimated_tools_tokens: telemetry.estimated_tools_tokens,
            estimated_evidence_tokens: telemetry.estimated_evidence_tokens,
            estimated_required_fallback_tokens: telemetry.estimated_required_fallback_tokens,
            dropped_history_items: telemetry.dropped_history_items,
            selected_evidence_items: telemetry.selected_evidence_items,
            dropped_evidence_items: telemetry.dropped_evidence_items,
            selected_evidence_ids: telemetry
                .selected_evidence_ids
                .iter()
                .map(|id| sanitize_opaque_identifier(id))
                .collect(),
            evidence_fingerprint: truncate_text(&telemetry.evidence_fingerprint, 128),
            truncated: telemetry.truncated,
            prompt_segment_count: telemetry.prompt_segment_count,
            prompt_contributor_count: telemetry.prompt_contributor_count,
            prompt_stable_prefix_hash: telemetry
                .prompt_stable_prefix_hash
                .map(|value| truncate_text(&value, 256)),
            cache_first_volatile_index: telemetry.cache_first_volatile_index,
            plan_total_prompt_tokens: telemetry.cache_stable_prompt_tokens
                + telemetry.cache_volatile_prompt_tokens,
            plan_stable_prompt_tokens: telemetry.cache_stable_prompt_tokens,
            plan_volatile_prompt_tokens: telemetry.cache_volatile_prompt_tokens,
            plan_cacheable_prefix_tokens: telemetry.cacheable_prefix_tokens,
            plan_stable_after_boundary_tokens: telemetry.cache_stable_after_boundary_tokens,
            cache_configured: telemetry.cache_configured,
            cache_hint_serialized: telemetry.cache_hint_serialized,
            cache_retention_sent: telemetry.cache_retention_sent.map(|value| {
                match value {
                    crate::config::PromptCacheRetention::InMemory => "in_memory",
                    crate::config::PromptCacheRetention::TwentyFourHours => "24h",
                }
                .into()
            }),
            cache_stable_prefix_segments: telemetry.cache_stable_prefix_segments,
            cache_stable_prompt_tokens: telemetry.cache_stable_prompt_tokens,
            cache_volatile_prompt_tokens: telemetry.cache_volatile_prompt_tokens,
            cacheable_prefix_tokens: telemetry.cacheable_prefix_tokens,
            cache_stable_after_boundary_tokens: telemetry.cache_stable_after_boundary_tokens,
            tool_call_count_before: telemetry.tool_call_count_before,
            tool_definitions_count: telemetry.tool_definitions_count,
            local_prefix_fingerprint: telemetry
                .local_prefix_fingerprint
                .map(|value| truncate_text(&value, 256)),
            routing_key: telemetry
                .routing_key
                .map(|value| truncate_text(&value, 256)),
            // `cached_tokens == 0` is retained for UI compatibility when a
            // provider omits cache details, but it is not a provider fact.
            provider_cached_tokens: (telemetry.usage_completeness
                == crate::agent::ProviderUsageCompleteness::Complete)
                .then(|| usage.map(|value| value.cached_tokens))
                .flatten(),
            provider_input_tokens: usage.map(|value| value.input_tokens),
            provider_output_tokens: usage.map(|value| value.output_tokens),
            provider_total_tokens: usage.map(|value| value.used_tokens),
            provider_response_id: telemetry
                .provider_response_id
                .as_deref()
                .map(sanitize_opaque_identifier),
            adjacent_lcp_units: telemetry.adjacent_lcp_units,
            adjacent_lcp_bytes: telemetry.adjacent_lcp_bytes,
            adjacent_lcp_estimated_tokens: telemetry.adjacent_lcp_estimated_tokens,
            current_unit_count: telemetry.current_unit_count,
            first_breaker: telemetry.first_breaker.map(|value| value.as_str().into()),
            cohort_comparable: telemetry.cohort_comparable,
            cohort_changed: telemetry.cohort_changed,
            usage_completeness: telemetry.usage_completeness.as_str().into(),
            cache_write_tokens: telemetry.cache_write_tokens,
            original_history_items: telemetry.original_history_items,
            retained_history_items: telemetry.retained_history_items,
        })
    }

    pub fn observe_reasoning_delta(&mut self, item_id: &str) {
        self.reasoning_started_at
            .entry(item_id.to_string())
            .or_insert_with(std::time::Instant::now);
    }

    pub fn clear_reasoning_observations(&mut self) {
        self.reasoning_started_at.clear();
    }

    pub fn record_reasoning_message(
        &mut self,
        item_id: &str,
        content: impl Into<String>,
    ) -> Result<()> {
        let duration_ms = self
            .reasoning_started_at
            .remove(item_id)
            .map(|started_at| u64::try_from(started_at.elapsed().as_millis()).unwrap_or(u64::MAX));
        self.append(TranscriptEvent::ReasoningMessage {
            content: content.into(),
            duration_ms,
        })
    }

    pub fn record_tool_call_started(
        &mut self,
        call_id: impl Into<String>,
        name: impl Into<String>,
        args: Value,
    ) -> Result<()> {
        self.append(TranscriptEvent::ToolCallStarted {
            call_id: call_id.into(),
            name: name.into(),
            args,
        })
    }

    pub fn record_assistant_tool_call_batch(
        &mut self,
        text: Option<String>,
        reasoning_content: Option<String>,
        calls: Vec<HistoryToolCall>,
    ) -> Result<()> {
        self.append(TranscriptEvent::AssistantToolCallBatch {
            text,
            reasoning_content,
            calls,
        })
    }

    pub fn record_tool_call_finished(
        &mut self,
        call_id: impl Into<String>,
        name: impl Into<String>,
        ok: bool,
        output: ToolResult,
    ) -> Result<()> {
        self.append(TranscriptEvent::ToolCallFinished {
            call_id: call_id.into(),
            name: name.into(),
            ok,
            output,
        })
    }

    #[cfg(test)]
    pub fn record_tool_call_finished_and_apply_context_control(
        &mut self,
        call_id: impl Into<String>,
        name: impl Into<String>,
        ok: bool,
        output: ToolResult,
    ) -> Result<()> {
        // Context-control tools are removed; this remains a compatibility alias
        // for the ordinary tool-finished journal path.
        self.record_tool_call_finished(call_id, name, ok, output)
    }

    pub fn record_tool_call_cancelled(
        &mut self,
        call_id: impl Into<String>,
        name: impl Into<String>,
    ) -> Result<()> {
        self.append(TranscriptEvent::ToolCallCancelled {
            call_id: call_id.into(),
            name: name.into(),
        })
    }

    #[cfg(test)]
    pub fn record_permission_decision(
        &mut self,
        tool: impl Into<String>,
        args: Value,
        allowed: bool,
    ) -> Result<()> {
        self.record_permission_decision_details(None, tool, args, allowed, None)
    }

    pub fn record_permission_decision_details(
        &mut self,
        call_id: Option<String>,
        tool: impl Into<String>,
        args: Value,
        allowed: bool,
        reason: Option<String>,
    ) -> Result<()> {
        self.record_permission_decision_full(
            call_id, tool, args, allowed, reason, None, None, None, None,
        )
    }

    pub fn record_permission_decision_full(
        &mut self,
        call_id: Option<String>,
        tool: impl Into<String>,
        args: Value,
        allowed: bool,
        reason: Option<String>,
        reviewer: Option<String>,
        approval: Option<String>,
        risk: Option<String>,
        reviewer_child_session_id: Option<String>,
    ) -> Result<()> {
        self.append(TranscriptEvent::PermissionDecision {
            call_id,
            tool: tool.into(),
            args,
            allowed,
            reason,
            reviewer,
            approval,
            risk,
            reviewer_child_session_id,
        })
    }

    pub fn record_permission_mode_changed(
        &mut self,
        previous_mode: impl Into<String>,
        new_mode: impl Into<String>,
    ) -> Result<()> {
        self.append(TranscriptEvent::PermissionModeChanged {
            previous_mode: previous_mode.into(),
            new_mode: new_mode.into(),
        })
    }

    pub fn record_todo_snapshot(&mut self, items: Vec<TodoItem>) -> Result<()> {
        self.append(TranscriptEvent::TodoSnapshot { items })
    }

    pub fn record_auto_continue_changed(&mut self, state: AutoContinueState) -> Result<()> {
        self.append(TranscriptEvent::AutoContinueChanged { state })
    }

    pub fn record_auto_continuation_scheduled(
        &mut self,
        continuation_count: usize,
        remaining_unfinished: usize,
    ) -> Result<()> {
        self.append(TranscriptEvent::AutoContinuationScheduled {
            continuation_count,
            remaining_unfinished,
        })
    }

    pub fn record_internal_continuation(
        &mut self,
        text: impl Into<String>,
        source: InternalContinuationSource,
    ) -> Result<()> {
        self.append(TranscriptEvent::InternalContinuation {
            text: text.into(),
            source,
        })
    }

    pub fn record_validation_advisory(&mut self, advisory: ValidationAdvisory) -> Result<()> {
        self.append(TranscriptEvent::ValidationAdvisory(advisory))
    }

    pub fn record_tool_execution_summary(
        &mut self,
        event: ToolExecutionSummaryEvent,
    ) -> Result<()> {
        self.append(TranscriptEvent::ToolExecutionSummary(event))
    }

    pub fn record_context_compaction(&mut self, event: ContextCompactionEvent) -> Result<()> {
        ensure!(
            event.tail_start_index.is_none(),
            "new context compaction records must omit legacy tail_start_index"
        );
        ensure!(
            event.checkpoint.is_none(),
            "new context compaction records must omit legacy checkpoint"
        );
        {
            let records = read_records(self.path())?;
            ensure!(
                records
                    .iter()
                    .map(|record| record.sequence)
                    .max()
                    .unwrap_or(0)
                    == self.sequence,
                "context compaction recorder frontier does not match committed transcript"
            );
            if self.current_context_branch_id.is_none() {
                ensure!(
                    transcript_projection::effective_branch_id_at_frontier(&records)?
                        == ROOT_CONTEXT_BRANCH_ID,
                    "context compaction recorder scope is ambiguous: an explicit branch cursor is required when the latest checkout is non-root"
                );
            }
            // A recorder can be positioned on a branch whose sibling has a
            // divergent history. Compaction anchors describe only that visible
            // branch/leaf, never the complete append-only journal.
            let scope = transcript_projection::context_compaction_validation_scope(
                &records,
                self.sequence,
                transcript_projection::SessionContextCursor {
                    branch_id: self.current_context_branch_id.clone(),
                    leaf_sequence: None,
                },
            )?;
            transcript_projection::validate_context_compaction_event_in_scope(&scope, &event)?;
            // This acknowledgement is the commit point for compaction.  It must
            // survive a crash before the agent adopts the resulting projection.
            self.append_durable_on_branch(
                TranscriptEvent::ContextCompaction(event),
                scope.actual_append_branch_id().clone(),
            )
        }
    }

    pub fn record_turn_finalized(&mut self, event: TurnFinalizedEvent) -> Result<()> {
        self.append(TranscriptEvent::TurnFinalized(event))
    }

    pub fn record_turn_interrupted(&mut self, turn_id: Option<u64>) -> Result<()> {
        self.clear_reasoning_observations();
        self.append(TranscriptEvent::TurnInterrupted { turn_id })
    }

    pub fn record_error(&mut self, message: impl Into<String>) -> Result<()> {
        self.append(TranscriptEvent::Error {
            message: message.into(),
        })
    }

    pub fn record_evidence(&mut self, draft: EvidenceDraft) -> Result<EvidenceRecord> {
        draft.validate()?;
        let sequence = self.sequence.saturating_add(1);
        let timestamp_ms = unix_timestamp_ms();
        let id = draft
            .id
            .clone()
            .unwrap_or_else(|| evidence_id_for_sequence(sequence));
        let event = TranscriptEvent::Evidence {
            id: id.clone(),
            evidence_kind: draft.evidence_kind,
            title: draft.title.clone(),
            summary: draft.summary.clone(),
            detail: draft.detail.clone(),
            source: draft.source.clone(),
            tags: draft.tags.clone(),
        };

        let record = draft.into_record(id, sequence, timestamp_ms)?;
        self.append_with_timestamp(event, timestamp_ms)?;
        Ok(record)
    }

    pub fn record_evidence_record(&mut self, evidence: EvidenceRecord) -> Result<()> {
        let draft = EvidenceDraft {
            id: Some(evidence.id.clone()),
            evidence_kind: evidence.evidence_kind,
            title: evidence.title.clone(),
            summary: evidence.summary.clone(),
            detail: evidence.detail.clone(),
            source: evidence.source.clone(),
            tags: evidence.tags.clone(),
        };
        draft.validate()?;
        self.append(TranscriptEvent::Evidence {
            id: evidence.id,
            evidence_kind: evidence.evidence_kind,
            title: evidence.title,
            summary: evidence.summary,
            detail: evidence.detail,
            source: evidence.source,
            tags: evidence.tags,
        })
    }

    pub fn append(&mut self, event: TranscriptEvent) -> Result<()> {
        self.append_with_timestamp(event, unix_timestamp_ms())
    }

    fn append_durable(&mut self, event: TranscriptEvent) -> Result<()> {
        self.append_with_timestamp_durable(event, unix_timestamp_ms())
    }

    fn append_durable_on_branch(
        &mut self,
        event: TranscriptEvent,
        context_branch_id: Option<String>,
    ) -> Result<()> {
        self.append_record(event, unix_timestamp_ms(), context_branch_id, true)
    }

    pub fn append_metadata(&mut self, event: TranscriptEvent) -> Result<()> {
        self.append_with_timestamp_and_branch(event, unix_timestamp_ms(), None)
    }

    fn set_active_context_experiment(&self, experiment: Option<ActiveContextExperiment>) {
        if let Ok(mut state) = self.context_scope_state.lock() {
            state.active_experiment = experiment;
        }
    }

    #[cfg(test)]
    fn record_context_node_status_change(
        &mut self,
        node_id: String,
        status: ContextNodeStatus,
    ) -> Result<()> {
        self.validate_context_tree_op(ContextTreeOp::SetNodeStatus {
            node_id: ContextNodeId::new(node_id.clone())?,
            status: status.clone(),
        })?;
        self.record_context_node_lifecycle(node_id, status)
    }

    #[cfg(test)]
    fn current_context_tree_state(&self) -> Result<crate::context_tree::ContextTreeState> {
        let records = read_records(self.path())?;
        transcript_projection::project_context_tree_for_active_branch(
            &records,
            self.current_context_branch_id(),
        )
    }

    #[cfg(test)]
    fn current_active_context_node_id(&self) -> Result<String> {
        self.current_context_tree_state()?
            .active_node_id()
            .map(|node_id| node_id.as_str().to_string())
            .ok_or_else(|| anyhow!("context tree has no active node"))
    }

    #[cfg(test)]
    fn validate_context_tree_op(&self, op: ContextTreeOp) -> Result<()> {
        let mut state = self.current_context_tree_state()?;
        state.apply(&op)
    }

    fn append_with_timestamp(&mut self, event: TranscriptEvent, timestamp_ms: u128) -> Result<()> {
        let context_branch_id = if event.is_context_branch_metadata() {
            None
        } else {
            self.current_context_branch_id.clone()
        };
        self.append_with_timestamp_and_branch(event, timestamp_ms, context_branch_id)
    }

    fn append_with_timestamp_durable(
        &mut self,
        event: TranscriptEvent,
        timestamp_ms: u128,
    ) -> Result<()> {
        let context_branch_id = if event.is_context_branch_metadata() {
            None
        } else {
            self.current_context_branch_id.clone()
        };
        self.append_record(event, timestamp_ms, context_branch_id, true)
    }

    fn append_with_timestamp_and_branch(
        &mut self,
        event: TranscriptEvent,
        timestamp_ms: u128,
        context_branch_id: Option<String>,
    ) -> Result<()> {
        self.append_record(event, timestamp_ms, context_branch_id, false)
    }

    fn append_record(
        &mut self,
        event: TranscriptEvent,
        timestamp_ms: u128,
        context_branch_id: Option<String>,
        durable: bool,
    ) -> Result<()> {
        ensure!(
            self.health == RecorderHealth::Healthy,
            "transcript recorder is poisoned after a previous I/O failure"
        );
        let sequence = self
            .sequence
            .checked_add(1)
            .ok_or_else(|| anyhow!("transcript sequence overflow"))?;
        let record = TranscriptRecord {
            session_id: self.session_id.clone(),
            sequence,
            timestamp_ms,
            context_branch_id,
            event,
        };

        let envelope = JournalRecordV1 {
            schema_version: JOURNAL_SCHEMA_VERSION,
            event_id: format!("{}:{sequence}", self.session_id),
            scope: journal_scope_for(&record),
            base_revision: sequence - 1,
            resulting_revision: sequence,
            transaction_id: None,
            transaction_index: None,
            transaction_count: None,
            record,
        };
        let mut line = serialize_journal_record(&envelope)?;
        line.push(b'\n');
        let durable = durable || requires_durable_commit(&envelope.record.event);
        if let Err(error) = self.sink.write_all(&line) {
            self.health = RecorderHealth::Poisoned;
            return Err(error.into());
        }
        if let Err(error) = self.sink.flush() {
            self.health = RecorderHealth::Poisoned;
            return Err(error.into());
        }
        if durable && let Err(error) = self.sink.sync_data() {
            self.health = RecorderHealth::Poisoned;
            return Err(error.into());
        }
        self.sequence = sequence;
        session_index::upsert_from_record(&self.path, &envelope.record);
        Ok(())
    }

    fn prepare_transaction_buffer(
        &self,
        events: &[(TranscriptEvent, Option<String>)],
    ) -> Result<(Vec<u8>, Vec<TranscriptRecord>, u64)> {
        ensure!(
            self.health == RecorderHealth::Healthy,
            "transcript recorder is poisoned after a previous I/O failure"
        );
        ensure!(
            !events.is_empty(),
            "transcript transaction must not be empty"
        );
        let count = events.len();
        let base_revision = self.sequence;
        let resulting_revision = base_revision
            .checked_add(
                u64::try_from(count).map_err(|_| anyhow!("transcript transaction is too large"))?,
            )
            .ok_or_else(|| anyhow!("transcript sequence overflow"))?;
        let transaction_id = format!(
            "{}:{}:{}",
            self.session_id,
            base_revision + 1,
            unix_timestamp_ms()
        );
        let timestamp_ms = unix_timestamp_ms();
        let mut payload = Vec::new();
        let mut index_records = Vec::with_capacity(count);
        for (index, (event, context_branch_id)) in events.iter().cloned().enumerate() {
            let sequence = base_revision + u64::try_from(index).unwrap() + 1;
            let record = TranscriptRecord {
                session_id: self.session_id.clone(),
                sequence,
                timestamp_ms,
                context_branch_id,
                event,
            };
            index_records.push(record.clone());
            let envelope = JournalRecordV1 {
                schema_version: JOURNAL_SCHEMA_VERSION,
                event_id: format!("{}:{sequence}", self.session_id),
                scope: journal_scope_for(&record),
                base_revision: sequence - 1,
                resulting_revision: sequence,
                transaction_id: Some(transaction_id.clone()),
                transaction_index: Some(index),
                transaction_count: Some(count),
                record,
            };
            payload.extend(serialize_journal_record(&envelope)?);
            payload.push(b'\n');
        }
        let commit = JournalTransactionCommitV1 {
            schema_version: JOURNAL_SCHEMA_VERSION,
            journal_entry: JOURNAL_TRANSACTION_COMMIT.into(),
            transaction_id,
            transaction_count: count,
            base_revision,
            resulting_revision,
            payload_length: payload.len(),
            payload_digest: journal_payload_digest(&payload),
        };
        let mut buffer = payload;
        serde_json::to_writer(&mut buffer, &commit)?;
        buffer.push(b'\n');
        Ok((buffer, index_records, resulting_revision))
    }

    pub fn append_transaction(
        &mut self,
        events: Vec<(TranscriptEvent, Option<String>)>,
    ) -> Result<()> {
        let (buffer, index_records, resulting_revision) =
            self.prepare_transaction_buffer(&events)?;
        if let Err(error) = self.sink.write_all(&buffer) {
            self.health = RecorderHealth::Poisoned;
            return Err(error.into());
        }
        if let Err(error) = self.sink.flush() {
            self.health = RecorderHealth::Poisoned;
            return Err(error.into());
        }
        if let Err(error) = self.sink.sync_data() {
            self.health = RecorderHealth::Poisoned;
            return Err(error.into());
        }
        self.sequence = resulting_revision;
        session_index::upsert_from_records(&self.path, &index_records);
        Ok(())
    }
}

/// Logical checkpoint payloads own their frozen `schema_version` field. Keep
/// the existing journal field for every other event, but use a distinct outer
/// name for this one flattened payload so JSON never contains duplicate keys.
fn serialize_journal_record(envelope: &JournalRecordV1) -> Result<Vec<u8>> {
    if !matches!(envelope.record.event, TranscriptEvent::LogicalCheckpoint(_)) {
        return Ok(serde_json::to_vec(envelope)?);
    }
    let mut value = serde_json::to_value(&envelope.record)?;
    let object = value
        .as_object_mut()
        .ok_or_else(|| anyhow!("serialized transcript record is not an object"))?;
    object.insert(
        "journal_schema_version".into(),
        Value::from(envelope.schema_version),
    );
    object.insert("event_id".into(), Value::from(envelope.event_id.clone()));
    object.insert("scope".into(), serde_json::to_value(envelope.scope)?);
    object.insert("base_revision".into(), Value::from(envelope.base_revision));
    object.insert(
        "resulting_revision".into(),
        Value::from(envelope.resulting_revision),
    );
    if let Some(value) = &envelope.transaction_id {
        object.insert("transaction_id".into(), Value::from(value.clone()));
    }
    if let Some(value) = envelope.transaction_index {
        object.insert("transaction_index".into(), Value::from(value));
    }
    if let Some(value) = envelope.transaction_count {
        object.insert("transaction_count".into(), Value::from(value));
    }
    Ok(serde_json::to_vec(&value)?)
}

fn journal_payload_digest(bytes: &[u8]) -> String {
    // A deterministic corruption guard, not a cryptographic integrity mechanism.
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

#[allow(dead_code)]
pub fn read_records(path: impl AsRef<Path>) -> Result<Vec<TranscriptRecord>> {
    read_records_inner(path, false)
}

pub(crate) fn read_records_with_fingerprint(
    path: impl AsRef<Path>,
) -> Result<(Vec<TranscriptRecord>, TranscriptFileFingerprint)> {
    let path = path.as_ref();
    let content = fs::read_to_string(path)
        .with_context(|| format!("failed to read transcript {}", path.display()))?;
    let records = parse_records_content(path, &content, false)?;
    Ok((records, transcript_file_fingerprint(&content)))
}

fn transcript_file_fingerprint(content: &str) -> TranscriptFileFingerprint {
    TranscriptFileFingerprint {
        content_len: content.len(),
        content_digest: journal_payload_digest(content.as_bytes()),
    }
}

fn content_tail_is_uncommitted_transaction(path: &Path, content: &str) -> Result<bool> {
    Ok(scan_transcript_content(path, content)?.has_uncommitted_transaction_tail)
}

pub(crate) fn read_records_allow_partial_tail(
    path: impl AsRef<Path>,
) -> Result<Vec<TranscriptRecord>> {
    read_records_inner(path, true)
}

fn read_records_inner(
    path: impl AsRef<Path>,
    allow_partial_tail: bool,
) -> Result<Vec<TranscriptRecord>> {
    let path = path.as_ref();
    let content = fs::read_to_string(path)
        .with_context(|| format!("failed to read transcript {}", path.display()))?;
    parse_records_content(path, &content, allow_partial_tail)
}

fn parse_records_content(
    path: &Path,
    content: &str,
    allow_partial_tail: bool,
) -> Result<Vec<TranscriptRecord>> {
    let has_complete_tail = content.ends_with('\n');
    let mut last_non_empty_line = None;
    for (index, line) in content.lines().enumerate() {
        if !line.trim().is_empty() {
            last_non_empty_line = Some(index);
        }
    }

    let mut records = Vec::new();
    let mut pending_transaction: Option<PendingTransaction> = None;
    for (index, line) in content.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }

        match parse_journal_line(line) {
            Ok(ParsedJournalLine::Record(entry)) => {
                let transaction = transaction_fields(&entry.v1)?;
                match transaction {
                    Some((transaction_id, transaction_index, transaction_count)) => {
                        ensure!(
                            transaction_count > 0,
                            "transcript transaction count must be positive"
                        );
                        let pending =
                            pending_transaction.get_or_insert_with(|| PendingTransaction {
                                transaction_id: transaction_id.clone(),
                                transaction_count,
                                base_revision: entry.v1.as_ref().unwrap().base_revision,
                                payload: Vec::new(),
                                entries: Vec::new(),
                            });
                        ensure!(
                            pending.transaction_id == transaction_id,
                            "transcript transaction is interrupted by a different transaction"
                        );
                        ensure!(
                            pending.transaction_count == transaction_count,
                            "transcript transaction count changes mid-transaction"
                        );
                        ensure!(
                            transaction_index == pending.entries.len(),
                            "transcript transaction records are not contiguous"
                        );
                        ensure!(
                            transaction_index < transaction_count,
                            "transcript transaction index exceeds its count"
                        );
                        pending
                            .payload
                            .extend(serialize_journal_record(entry.v1.as_ref().unwrap())?);
                        pending.payload.push(b'\n');
                        pending.entries.push(entry);
                    }
                    None => {
                        ensure!(
                            pending_transaction.is_none(),
                            "transcript transaction is missing its commit marker before another record"
                        );
                        records.push(entry);
                    }
                }
            }
            Ok(ParsedJournalLine::Commit(commit)) => {
                let pending = pending_transaction
                    .take()
                    .ok_or_else(|| anyhow!("transcript transaction commit has no records"))?;
                ensure!(
                    commit.schema_version == JOURNAL_SCHEMA_VERSION,
                    "unsupported transcript journal schema version {}",
                    commit.schema_version
                );
                ensure!(
                    commit.journal_entry == JOURNAL_TRANSACTION_COMMIT,
                    "unknown transcript journal entry '{}'",
                    commit.journal_entry
                );
                ensure!(
                    commit.transaction_id == pending.transaction_id,
                    "transcript transaction commit id does not match records"
                );
                ensure!(
                    commit.transaction_count == pending.transaction_count
                        && pending.entries.len() == pending.transaction_count,
                    "transcript transaction commit count does not match records"
                );
                ensure!(
                    commit.base_revision == pending.base_revision,
                    "transcript transaction commit base revision does not match records"
                );
                let last_payload_revision = pending
                    .entries
                    .last()
                    .and_then(|entry| entry.v1.as_ref())
                    .ok_or_else(|| anyhow!("transcript transaction commit has no payload records"))?
                    .resulting_revision;
                ensure!(
                    commit.resulting_revision == last_payload_revision,
                    "transcript transaction commit resulting revision does not match payload records"
                );
                ensure!(
                    commit.resulting_revision
                        == commit.base_revision + u64::try_from(commit.transaction_count).unwrap(),
                    "transcript transaction commit revision does not match count"
                );
                ensure!(
                    commit.payload_length == pending.payload.len()
                        && commit.payload_digest == journal_payload_digest(&pending.payload),
                    "transcript transaction commit payload does not match records"
                );
                records.extend(pending.entries);
            }
            Err(error)
                if allow_partial_tail
                    && !has_complete_tail
                    && Some(index) == last_non_empty_line =>
            {
                tracing::debug!(
                    transcript = %path.display(),
                    line = index + 1,
                    error = %error,
                    "ignored incomplete transcript tail while reading live transcript"
                );
                break;
            }
            Err(error) => {
                return Err(error).with_context(|| {
                    format!(
                        "failed to parse line {} from transcript {}",
                        index + 1,
                        path.display()
                    )
                });
            }
        }
    }

    // A complete but uncommitted transaction can only be the physical tail.
    // It is deliberately invisible to projections and recovery.
    validate_journal_entries(&records)?;
    Ok(records.into_iter().map(|entry| entry.record).collect())
}

#[cfg(test)]
fn transcript_records_match(
    current: &[TranscriptRecord],
    expected: &[TranscriptRecord],
) -> Result<bool> {
    if current.len() != expected.len() {
        return Ok(false);
    }
    for (current, expected) in current.iter().zip(expected) {
        if serde_json::to_vec(current)? != serde_json::to_vec(expected)? {
            return Ok(false);
        }
    }
    Ok(true)
}

struct TranscriptContentState {
    has_uncommitted_transaction_tail: bool,
}

fn scan_transcript_content(path: &Path, content: &str) -> Result<TranscriptContentState> {
    let mut pending_transaction: Option<PendingTransaction> = None;

    for (index, line) in content.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        match parse_journal_line(line).with_context(|| {
            format!(
                "failed to parse line {} from transcript {}",
                index + 1,
                path.display()
            )
        })? {
            ParsedJournalLine::Record(entry) => match transaction_fields(&entry.v1)? {
                Some((transaction_id, transaction_index, transaction_count)) => {
                    ensure!(
                        transaction_count > 0,
                        "transcript transaction count must be positive"
                    );
                    let pending = pending_transaction.get_or_insert_with(|| PendingTransaction {
                        transaction_id: transaction_id.clone(),
                        transaction_count,
                        base_revision: entry.v1.as_ref().unwrap().base_revision,
                        payload: Vec::new(),
                        entries: Vec::new(),
                    });
                    ensure!(
                        pending.transaction_id == transaction_id,
                        "transcript transaction is interrupted by a different transaction"
                    );
                    ensure!(
                        pending.transaction_count == transaction_count,
                        "transcript transaction count changes mid-transaction"
                    );
                    ensure!(
                        transaction_index == pending.entries.len(),
                        "transcript transaction records are not contiguous"
                    );
                    ensure!(
                        transaction_index < transaction_count,
                        "transcript transaction index exceeds its count"
                    );
                    pending
                        .payload
                        .extend(serialize_journal_record(entry.v1.as_ref().unwrap())?);
                    pending.payload.push(b'\n');
                    pending.entries.push(entry);
                }
                None => ensure!(
                    pending_transaction.is_none(),
                    "transcript transaction is missing its commit marker before another record"
                ),
            },
            ParsedJournalLine::Commit(commit) => {
                let pending = pending_transaction
                    .take()
                    .ok_or_else(|| anyhow!("transcript transaction commit has no records"))?;
                ensure!(
                    commit.schema_version == JOURNAL_SCHEMA_VERSION,
                    "unsupported transcript journal schema version {}",
                    commit.schema_version
                );
                ensure!(
                    commit.journal_entry == JOURNAL_TRANSACTION_COMMIT,
                    "unknown transcript journal entry '{}'",
                    commit.journal_entry
                );
                ensure!(
                    commit.transaction_id == pending.transaction_id,
                    "transcript transaction commit id does not match records"
                );
                ensure!(
                    commit.transaction_count == pending.transaction_count
                        && pending.entries.len() == pending.transaction_count,
                    "transcript transaction commit count does not match records"
                );
                ensure!(
                    commit.base_revision == pending.base_revision,
                    "transcript transaction commit base revision does not match records"
                );
                let last_payload_revision = pending
                    .entries
                    .last()
                    .and_then(|entry| entry.v1.as_ref())
                    .ok_or_else(|| anyhow!("transcript transaction commit has no payload records"))?
                    .resulting_revision;
                ensure!(
                    commit.resulting_revision == last_payload_revision,
                    "transcript transaction commit resulting revision does not match payload records"
                );
                ensure!(
                    commit.resulting_revision
                        == commit.base_revision + u64::try_from(commit.transaction_count).unwrap(),
                    "transcript transaction commit revision does not match count"
                );
                ensure!(
                    commit.payload_length == pending.payload.len()
                        && commit.payload_digest == journal_payload_digest(&pending.payload),
                    "transcript transaction commit payload does not match records"
                );
            }
        }
    }

    Ok(TranscriptContentState {
        has_uncommitted_transaction_tail: pending_transaction.is_some(),
    })
}

#[derive(Debug)]
struct JournalEntry {
    record: TranscriptRecord,
    v1: Option<JournalRecordV1>,
}

struct PendingTransaction {
    transaction_id: String,
    transaction_count: usize,
    base_revision: u64,
    payload: Vec<u8>,
    entries: Vec<JournalEntry>,
}

enum ParsedJournalLine {
    Record(JournalEntry),
    Commit(JournalTransactionCommitV1),
}

fn parse_journal_line(line: &str) -> Result<ParsedJournalLine> {
    if has_top_level_json_field(line, "journal_entry") {
        return Ok(ParsedJournalLine::Commit(serde_json::from_str(line)?));
    }
    if has_top_level_json_field(line, "journal_schema_version")
        || has_top_level_json_field(line, "schema_version")
    {
        let v1 = parse_journal_v1(line)?;
        ensure!(
            v1.schema_version == JOURNAL_SCHEMA_VERSION,
            "unsupported transcript journal schema version {}",
            v1.schema_version
        );
        Ok(ParsedJournalLine::Record(JournalEntry {
            record: v1.record.clone(),
            v1: Some(v1),
        }))
    } else {
        Ok(ParsedJournalLine::Record(JournalEntry {
            record: serde_json::from_str(line)?,
            v1: None,
        }))
    }
}

fn has_top_level_json_field(line: &str, field: &str) -> bool {
    let bytes = line.as_bytes();
    let mut depth = 0usize;
    let mut index = 0usize;
    while index < bytes.len() {
        match bytes[index] {
            b'{' | b'[' => depth += 1,
            b'}' | b']' => depth = depth.saturating_sub(1),
            b'"' => {
                let start = index + 1;
                index += 1;
                while index < bytes.len() {
                    if bytes[index] == b'\\' {
                        index += 2;
                    } else if bytes[index] == b'"' {
                        break;
                    } else {
                        index += 1;
                    }
                }
                if index >= bytes.len() {
                    return false;
                }
                let mut next = index + 1;
                while next < bytes.len() && bytes[next].is_ascii_whitespace() {
                    next += 1;
                }
                if depth == 1
                    && bytes[start..index] == *field.as_bytes()
                    && bytes.get(next) == Some(&b':')
                {
                    return true;
                }
            }
            _ => {}
        }
        index += 1;
    }
    false
}

fn parse_journal_v1(line: &str) -> Result<JournalRecordV1> {
    #[derive(Deserialize)]
    struct JournalMetadata {
        #[serde(rename = "journal_schema_version")]
        schema_version: u32,
        event_id: String,
        scope: JournalScope,
        base_revision: u64,
        resulting_revision: u64,
        #[serde(default)]
        transaction_id: Option<String>,
        #[serde(default)]
        transaction_index: Option<usize>,
        #[serde(default)]
        transaction_count: Option<usize>,
        timestamp_ms: u128,
    }

    let metadata: JournalMetadata = if has_top_level_json_field(line, "journal_schema_version") {
        serde_json::from_str(line)?
    } else {
        #[derive(Deserialize)]
        struct LegacyJournalMetadata {
            schema_version: u32,
            event_id: String,
            scope: JournalScope,
            base_revision: u64,
            resulting_revision: u64,
            #[serde(default)]
            transaction_id: Option<String>,
            #[serde(default)]
            transaction_index: Option<usize>,
            #[serde(default)]
            transaction_count: Option<usize>,
            timestamp_ms: u128,
        }
        let legacy: LegacyJournalMetadata = serde_json::from_str(line)?;
        JournalMetadata {
            schema_version: legacy.schema_version,
            event_id: legacy.event_id,
            scope: legacy.scope,
            base_revision: legacy.base_revision,
            resulting_revision: legacy.resulting_revision,
            transaction_id: legacy.transaction_id,
            transaction_index: legacy.transaction_index,
            transaction_count: legacy.transaction_count,
            timestamp_ms: legacy.timestamp_ms,
        }
    };
    let record: TranscriptRecord = serde_json::from_str(line)?;
    ensure!(
        metadata.timestamp_ms == record.timestamp_ms,
        "transcript v1 timestamp metadata is inconsistent"
    );
    Ok(JournalRecordV1 {
        schema_version: metadata.schema_version,
        event_id: metadata.event_id,
        scope: metadata.scope,
        base_revision: metadata.base_revision,
        resulting_revision: metadata.resulting_revision,
        transaction_id: metadata.transaction_id,
        transaction_index: metadata.transaction_index,
        transaction_count: metadata.transaction_count,
        record,
    })
}

fn transaction_fields(v1: &Option<JournalRecordV1>) -> Result<Option<(String, usize, usize)>> {
    let Some(v1) = v1 else { return Ok(None) };
    match (
        &v1.transaction_id,
        v1.transaction_index,
        v1.transaction_count,
    ) {
        (None, None, None) => Ok(None),
        (Some(id), Some(index), Some(count)) => Ok(Some((id.clone(), index, count))),
        _ => Err(anyhow!(
            "transcript transaction fields must be present together"
        )),
    }
}

fn validate_journal_entries(entries: &[JournalEntry]) -> Result<()> {
    let mut session_id = None;
    let mut previous_sequence = None;
    let mut previous_revision = None;
    let mut saw_v1 = false;
    let mut event_ids = std::collections::BTreeSet::new();

    for entry in entries {
        if let Some(expected) = &session_id {
            ensure!(
                entry.record.session_id == *expected,
                "transcript contains records from multiple sessions"
            );
        } else {
            session_id = Some(entry.record.session_id.clone());
        }
        if let Some(previous) = previous_sequence {
            ensure!(
                entry.record.sequence > previous,
                "transcript sequence must be strictly increasing"
            );
        }
        previous_sequence = Some(entry.record.sequence);

        match &entry.v1 {
            Some(v1) => {
                saw_v1 = true;
                ensure!(
                    v1.event_id == format!("{}:{}", entry.record.session_id, entry.record.sequence),
                    "transcript v1 event_id does not match record identity"
                );
                ensure!(
                    event_ids.insert(v1.event_id.as_str()),
                    "transcript v1 event_id must be unique"
                );
                ensure!(
                    v1.scope == journal_scope_for(&entry.record),
                    "transcript v1 scope does not match context_branch_id"
                );
                ensure!(
                    v1.resulting_revision == v1.base_revision + 1,
                    "transcript v1 revisions must be consecutive"
                );
                ensure!(
                    v1.resulting_revision == entry.record.sequence,
                    "transcript v1 resulting_revision must equal sequence"
                );
                let expected_base = previous_revision.unwrap_or(0);
                ensure!(
                    v1.base_revision == expected_base,
                    "transcript v1 base_revision is not continuous"
                );
                previous_revision = Some(v1.resulting_revision);
            }
            None => {
                ensure!(!saw_v1, "legacy transcript record cannot follow v1 records");
                previous_revision = Some(entry.record.sequence);
            }
        }
    }
    Ok(())
}

fn journal_scope_for(record: &TranscriptRecord) -> JournalScope {
    if record.context_branch_id.is_some() {
        JournalScope::Branch
    } else {
        JournalScope::Global
    }
}

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

#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JobBoardEntry {
    pub active: bool,
    pub unreconciled: bool,
    pub reconciled: bool,
    pub reusable_eligible: bool,
    pub run_id: String,
    pub child_session_id: String,
    pub agent_name: String,
    pub status: String,
    pub summary: String,
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
                let transactional = transaction_fields(&entry.v1)?.is_some();
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

#[cfg(test)]
pub fn restore_job_board(
    base_dir: impl AsRef<Path>,
    parent_records: &[TranscriptRecord],
) -> Result<Vec<JobBoardEntry>> {
    transcript_projection::project_job_board(&child_sessions_dir(base_dir), parent_records)
}

#[cfg(test)]
pub fn restore_session_history(records: &[TranscriptRecord]) -> Result<Vec<HistoryItem>> {
    transcript_projection::validate_context_projection_events(records)?;
    Ok(transcript_projection::restore_session_history_projection(
        records,
    ))
}

#[cfg(test)]
pub(crate) fn restore_session_protocol_frames(
    records: &[TranscriptRecord],
) -> Result<Vec<crate::protocol_frames::ProtocolFrame>> {
    transcript_projection::restore_session_protocol_frames_projection(records)
}

#[cfg(test)]
pub(crate) fn restore_runtime_snapshot(
    records: &[TranscriptRecord],
) -> Result<crate::runtime_context::RuntimeSnapshot> {
    let session_id = records
        .first()
        .map(|record| record.session_id.clone())
        .unwrap_or_else(|| "restored-session".into());
    Ok(transcript_projection::project_runtime_restore_snapshot(
        session_id,
        records.to_vec(),
        transcript_projection::SessionContextCursor {
            branch_id: None,
            leaf_sequence: None,
        },
        &[],
    )?
    .snapshot)
}

#[cfg(test)]
pub fn restore_compacted_conversation_messages(
    records: &[TranscriptRecord],
) -> Result<Vec<ConversationMessage>> {
    Ok(restore_session_history(records)?
        .into_iter()
        .filter_map(history_item_to_conversation_message)
        .collect())
}

#[cfg(test)]
pub fn restore_conversation_messages(
    records: &[TranscriptRecord],
) -> Result<Vec<ConversationMessage>> {
    restore_compacted_conversation_messages(records)
}

pub fn restore_latest_model(records: &[TranscriptRecord]) -> Option<String> {
    transcript_projection::restore_latest_model_projection(records)
}

pub fn restore_latest_expert_models(
    records: &[TranscriptRecord],
) -> indexmap::IndexMap<String, String> {
    let mut models = indexmap::IndexMap::new();
    for record in records {
        if let TranscriptEvent::ExpertModelChanged { agent_name, model } = &record.event {
            models.insert(agent_name.clone(), model.clone());
        }
    }
    models
}

#[cfg(test)]
pub(crate) fn restore_latest_expert_models_for_cursor(
    session_id: &str,
    records: &[TranscriptRecord],
    cursor: transcript_projection::SessionContextCursor,
) -> anyhow::Result<indexmap::IndexMap<String, String>> {
    let snapshot = transcript_projection::build_session_context_snapshot(
        session_id.to_string(),
        records.to_vec(),
        cursor,
    )?;
    Ok(restore_latest_expert_models(&snapshot.records))
}

pub fn restore_latest_permission_mode(records: &[TranscriptRecord]) -> Option<String> {
    transcript_projection::restore_latest_permission_mode_projection(records)
}

#[cfg(test)]
pub fn restore_session_evidence(records: &[TranscriptRecord]) -> Result<Vec<EvidenceRecord>> {
    restore_evidence_records(records)
}

pub fn restore_latest_todo_snapshot(records: &[TranscriptRecord]) -> Option<Vec<TodoItem>> {
    let mut latest = None;
    for record in records {
        match &record.event {
            TranscriptEvent::UserMessage { .. }
            | TranscriptEvent::TurnStarted(_)
            | TranscriptEvent::TurnInterrupted { .. }
            | TranscriptEvent::Error { .. } => latest = None,
            TranscriptEvent::TodoSnapshot { items } => latest = Some(items.clone()),
            _ => {}
        }
    }
    latest
}

pub fn restore_latest_auto_continue_state(
    records: &[TranscriptRecord],
) -> Option<AutoContinueState> {
    let mut latest = None;
    for record in records {
        match &record.event {
            TranscriptEvent::UserMessage { .. }
            | TranscriptEvent::TurnStarted(_)
            | TranscriptEvent::TurnInterrupted { .. }
            | TranscriptEvent::Error { .. } => latest = None,
            TranscriptEvent::AutoContinueChanged { state } => latest = Some(state.clone()),
            _ => {}
        }
    }
    latest
}

#[cfg(test)]
pub fn restore_max_turn_id(records: &[TranscriptRecord]) -> u64 {
    transcript_projection::restore_max_turn_id_projection(records)
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

fn requires_durable_commit(event: &TranscriptEvent) -> bool {
    matches!(
        event,
        TranscriptEvent::SessionStarted { .. }
            | TranscriptEvent::ModelChanged { .. }
            | TranscriptEvent::ExpertModelChanged { .. }
            | TranscriptEvent::SubagentLifecycle { .. }
            | TranscriptEvent::SubagentResult { .. }
            | TranscriptEvent::ContextBranchCreated { .. }
            | TranscriptEvent::ContextBranchSummary { .. }
            | TranscriptEvent::ContextCheckout { .. }
            | TranscriptEvent::HistoryNavigation { .. }
            | TranscriptEvent::ContextExperimentStarted { .. }
            | TranscriptEvent::ContextNodeCreated { .. }
            | TranscriptEvent::ContextNodeLifecycle { .. }
            | TranscriptEvent::ContextViewOperationMetadata { .. }
            | TranscriptEvent::ContextSummaryArtifactMetadata { .. }
            | TranscriptEvent::FoldedOutputMetadata { .. }
            | TranscriptEvent::ContextExperimentReturned { .. }
            | TranscriptEvent::UserMessage { .. }
            | TranscriptEvent::AssistantMessage { .. }
            | TranscriptEvent::ReasoningMessage { .. }
            | TranscriptEvent::AssistantToolCallBatch { .. }
            | TranscriptEvent::ToolCallStarted { .. }
            | TranscriptEvent::ToolCallFinished { .. }
            | TranscriptEvent::ToolCallCancelled { .. }
            | TranscriptEvent::PermissionDecision { .. }
            | TranscriptEvent::PermissionModeChanged { .. }
            | TranscriptEvent::TodoSnapshot { .. }
            | TranscriptEvent::AutoContinueChanged { .. }
            | TranscriptEvent::AutoContinuationScheduled { .. }
            | TranscriptEvent::InternalContinuation { .. }
            | TranscriptEvent::TurnInterrupted { .. }
            | TranscriptEvent::Evidence { .. }
    ) || matches!(event, TranscriptEvent::ContextCompaction(_))
}

pub(crate) fn sync_recorder_branch(recorder: &mut TranscriptRecorder, branch_id: &str) {
    if branch_id == ROOT_CONTEXT_BRANCH_ID {
        recorder.set_current_context_branch_id(None);
    } else {
        recorder.set_current_context_branch_id(Some(branch_id.to_string()));
    }
}

#[cfg(test)]
fn required_context_return_string(data: &Value, field: &str) -> Result<String> {
    let value = data
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("context__return output field '{field}' must be a string"))?;
    let value = value.trim();
    ensure!(
        !value.is_empty(),
        "context__return output field '{field}' must not be empty"
    );
    Ok(value.to_string())
}

#[cfg(test)]
fn optional_context_return_string(data: &Value, field: &str) -> Result<Option<String>> {
    match data.get(field) {
        Some(Value::String(value)) => {
            let value = value.trim();
            ensure!(
                !value.is_empty(),
                "context__return output field '{field}' must not be empty when provided"
            );
            Ok(Some(value.to_string()))
        }
        Some(Value::Null) | None => Ok(None),
        Some(_) => Err(anyhow!(
            "context__return output field '{field}' must be string or null"
        )),
    }
}

fn reconstruct_context_scope_state(records: &[TranscriptRecord]) -> Result<ContextScopeState> {
    let mut state = ContextScopeState::default();
    for record in records {
        match &record.event {
            TranscriptEvent::ContextExperimentStarted {
                branch_id,
                parent_branch_id,
                base_sequence,
            } => {
                ensure!(
                    state.active_experiment.is_none(),
                    "nested context experiments are not supported in transcript replay"
                );
                state.active_experiment = Some(ActiveContextExperiment {
                    branch_id: branch_id.clone(),
                    parent_branch_id: parent_branch_id.clone(),
                    base_sequence: *base_sequence,
                    writes_observed: false,
                });
            }
            TranscriptEvent::ContextExperimentReturned { branch_id, .. } => {
                ensure!(
                    state
                        .active_experiment
                        .as_ref()
                        .is_some_and(|experiment| experiment.branch_id == *branch_id),
                    "context experiment return for unknown branch '{branch_id}'"
                );
                state.active_experiment = None;
            }
            TranscriptEvent::ToolExecutionSummary(event) => {
                if event.effect_kind == "write"
                    && let Some(experiment) = state.active_experiment.as_mut()
                    && record
                        .context_branch_id
                        .as_deref()
                        .unwrap_or(ROOT_CONTEXT_BRANCH_ID)
                        == experiment.branch_id
                {
                    experiment.writes_observed = true;
                }
            }
            _ => {}
        }
    }
    Ok(state)
}

pub(crate) fn format_context_experiment_return(
    branch_id: &str,
    outcome: &str,
    summary: &str,
    next_action: Option<&str>,
    had_writes: bool,
) -> String {
    let mut text = format!("Returned from context experiment {branch_id} ({outcome}): {summary}");
    if let Some(next_action) = next_action {
        text.push_str(&format!(" Next action: {next_action}."));
    }
    if had_writes {
        text.push_str(" Context restored, files were NOT reverted.");
    }
    text
}

fn append_history_item_from_transcript_record(record: &TranscriptRecord) -> Option<HistoryItem> {
    match &record.event {
        TranscriptEvent::UserMessage { content } => {
            Some(HistoryItem::user_content(content.clone()))
        }
        TranscriptEvent::AssistantMessage { content } => {
            Some(HistoryItem::assistant(content.clone()))
        }
        TranscriptEvent::InternalContinuation { text, .. } => {
            Some(HistoryItem::internal_continuation(text.clone()))
        }
        TranscriptEvent::AssistantToolCallBatch {
            text,
            reasoning_content,
            calls,
        } => Some(HistoryItem::AssistantToolCalls {
            text: text.clone(),
            reasoning_content: reasoning_content.clone(),
            calls: calls.clone(),
        }),
        TranscriptEvent::ToolCallStarted {
            call_id,
            name,
            args,
        } => Some(HistoryItem::AssistantToolCalls {
            text: None,
            reasoning_content: None,
            calls: vec![HistoryToolCall {
                call_id: call_id.clone(),
                name: name.clone(),
                arguments_json: args.to_string(),
            }],
        }),
        TranscriptEvent::ToolCallFinished {
            call_id, output, ..
        } => Some(HistoryItem::ToolOutput {
            call_id: call_id.clone(),
            output_json: serde_json::to_string(&output.for_text_history())
                .unwrap_or_else(|_| "null".to_string()),
            images: output.images.clone(),
        }),
        TranscriptEvent::ToolCallCancelled { .. } => None,
        TranscriptEvent::ContextExperimentReturned {
            branch_id,
            outcome,
            summary,
            next_action,
            had_writes,
            ..
        } => Some(HistoryItem::context_summary(
            format_context_experiment_return(
                branch_id,
                outcome,
                summary,
                next_action.as_deref(),
                *had_writes,
            ),
        )),
        _ => None,
    }
}

#[cfg(test)]
fn required_metadata_string(metadata: &Value, field: &str) -> Result<String> {
    metadata
        .get(field)
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .ok_or_else(|| anyhow!("invalid pending metadata: missing string field '{field}'"))
}

#[cfg(test)]
fn optional_metadata_string(metadata: &Value, field: &str) -> Option<String> {
    metadata
        .get(field)
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
}

#[cfg(test)]
fn optional_metadata_u32(metadata: &Value, field: &str) -> Result<Option<u32>> {
    metadata
        .get(field)
        .map(|value| match value {
            Value::Null => Ok(None),
            Value::Number(number) => number
                .as_u64()
                .ok_or_else(|| anyhow!("invalid pending metadata: field '{field}' must be u32"))
                .and_then(|value| {
                    u32::try_from(value).map_err(|_| {
                        anyhow!("invalid pending metadata: field '{field}' exceeds u32")
                    })
                })
                .map(Some),
            _ => Err(anyhow!(
                "invalid pending metadata: field '{field}' must be u32 or null"
            )),
        })
        .transpose()
        .map(|value| value.flatten())
}

#[cfg(test)]
fn optional_metadata_u64(metadata: &Value, field: &str) -> Result<Option<u64>> {
    metadata
        .get(field)
        .map(|value| match value {
            Value::Null => Ok(None),
            Value::Number(number) => number
                .as_u64()
                .ok_or_else(|| anyhow!("invalid pending metadata: field '{field}' must be u64"))
                .map(Some),
            _ => Err(anyhow!(
                "invalid pending metadata: field '{field}' must be u64 or null"
            )),
        })
        .transpose()
        .map(|value| value.flatten())
}

fn history_item_to_conversation_message(item: HistoryItem) -> Option<ConversationMessage> {
    match item {
        HistoryItem::ContextSummary { text } => Some(ConversationMessage {
            role: ConversationRole::Summary,
            content: text,
        }),
        HistoryItem::UserMessage { content } => Some(ConversationMessage {
            role: ConversationRole::User,
            content: content.display_text(),
        }),
        HistoryItem::InternalContinuation { text } => Some(ConversationMessage {
            role: ConversationRole::User,
            content: text,
        }),
        HistoryItem::AssistantText { text } => Some(ConversationMessage {
            role: ConversationRole::Assistant,
            content: text,
        }),
        HistoryItem::AssistantToolCalls { .. } | HistoryItem::ToolOutput { .. } => None,
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
