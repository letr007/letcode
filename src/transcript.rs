use anyhow::{Context, Result, anyhow, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::agent::subagent_tool_name_for_agent_name;
use crate::agent::{
    AutoContinueState, ContextCompactionEvent, ConversationMessage, ConversationRole,
    LlmRequestTelemetry, LlmRequestTelemetryPhase, TodoItem, ToolExecutionSummaryEvent,
    TurnFinalizedEvent, TurnStartedEvent, ValidationAdvisory,
};
use crate::context_tree::{
    ContextBlockRef, ContextNodeId, ContextNodeStatus, ContextSourceRef, ContextTreeOp,
};
use crate::evidence::{
    EvidenceDraft, EvidenceKind, EvidenceRecord, EvidenceSource, evidence_id_for_sequence,
    restore_evidence_records,
};
use crate::protocol_frames::ProtocolFrame;
use crate::request_builder::{HistoryItem, HistoryToolCall};
use crate::runtime_context::RuntimeSnapshot;
use crate::subagent::StructuredSubagentResult;
use crate::tool::ToolResult;
use crate::tool_names;
use crate::user_content::UserMessageContent;

mod model;

pub use model::{TranscriptEvent, TranscriptRecord};

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
    FoldedOutputAudit {
        output_id: String,
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

/// The durable workflow state reconstructed for a prepared checkpoint.
/// This is deliberately typed so consumers compare state rather than rendered
/// retained-item transport data.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct CheckpointWorkflowProjection {
    pub todos: Vec<TodoItem>,
    pub auto_continue: AutoContinueState,
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

/// A transcript-backed checkpoint candidate. Preparing is pure with respect to
/// the journal; only `record_logical_checkpoint_at_frontier` acknowledges it.
#[derive(Debug, Clone)]
pub(crate) struct PreparedLogicalCheckpoint {
    pub expected_journal_frontier: u64,
    pub expected_branch_id: String,
    pub event: LogicalCheckpointEventV1,
    pub projected_snapshot: RuntimeSnapshot,
    pub projected_protocol_frames: Vec<ProtocolFrame>,
    pub projected_workflow: Option<CheckpointWorkflowProjection>,
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
        })
    }

    pub fn open_existing(base_dir: impl AsRef<Path>, session_id: &str) -> Result<Self> {
        fs::create_dir_all(base_dir.as_ref())?;

        let file_path = session_path(base_dir.as_ref(), session_id);
        let records = read_records(&file_path)?;
        ensure!(
            !has_uncommitted_transaction_tail(&file_path)?,
            "transcript has an uncommitted transaction tail and cannot safely accept new records"
        );
        ensure!(
            records.iter().all(|record| record.session_id == session_id),
            "transcript contains records for a different session"
        );
        let sequence = records
            .iter()
            .map(|record| record.sequence)
            .max()
            .unwrap_or(0);
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&file_path)?;

        let context_scope_state = Arc::new(Mutex::new(reconstruct_context_scope_state(&records)?));

        Ok(Self {
            session_id: session_id.to_string(),
            path: file_path,
            sink: Box::new(FileJournalSink(file)),
            sequence,
            health: RecorderHealth::Healthy,
            current_context_branch_id: None,
            context_scope_state,
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

    pub fn record_context_node_activated(&mut self, node_id: impl Into<String>) -> Result<()> {
        self.activate_context_node(node_id)
    }

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

    pub fn activate_context_node(&mut self, node_id: impl Into<String>) -> Result<()> {
        self.record_context_node_status_change(node_id.into(), ContextNodeStatus::Active)
    }

    pub fn suspend_context_node(&mut self, node_id: impl Into<String>) -> Result<()> {
        self.record_context_node_status_change(node_id.into(), ContextNodeStatus::Inactive)
    }

    pub fn terminalize_context_node(&mut self, node_id: impl Into<String>) -> Result<()> {
        self.record_context_node_status_change(node_id.into(), ContextNodeStatus::Archived)
    }

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

    pub fn record_context_tool_pending_metadata(
        &mut self,
        tool_name: &str,
        ok: bool,
        output: &ToolResult,
    ) -> Result<()> {
        if !ok || !context_tool_allows_pending_metadata(tool_name) {
            return Ok(());
        }
        let Some(data) = output.data.as_ref() else {
            return Ok(());
        };
        if data.get("pending_recording").and_then(Value::as_bool) != Some(true) {
            return Ok(());
        }

        if let Some(operation_metadata) = data.get("operation_metadata") {
            self.record_context_view_operation_metadata(
                required_metadata_string(operation_metadata, "operation")?,
                optional_metadata_string(operation_metadata, "block_id"),
                optional_metadata_string(operation_metadata, "node_id"),
                optional_metadata_string(operation_metadata, "detail"),
            )?;
        }

        if let Some(summary_metadata) = data.get("summary_metadata") {
            self.record_context_summary_artifact_metadata(
                required_metadata_string(summary_metadata, "node_id")?,
                required_metadata_string(summary_metadata, "artifact_id")?,
                required_metadata_string(summary_metadata, "artifact_kind")?,
                optional_metadata_u32(summary_metadata, "version")?,
                optional_metadata_string(summary_metadata, "summary"),
                optional_metadata_string(summary_metadata, "source_node_id"),
                optional_metadata_string(summary_metadata, "source_block_id"),
                optional_metadata_u64(summary_metadata, "source_start_sequence")?,
                optional_metadata_u64(summary_metadata, "source_end_sequence")?,
            )?;
        }

        Ok(())
    }

    pub fn record_folded_output_metadata(
        &mut self,
        node_id: Option<String>,
        output_id: impl Into<String>,
        output_kind: impl Into<String>,
        call_id: Option<String>,
        tool_name: Option<String>,
        stream: Option<String>,
        content: Option<String>,
        byte_count: Option<usize>,
        line_count: Option<usize>,
        truncated: Option<bool>,
        shell_command: Option<String>,
        source_start_sequence: Option<u64>,
        source_end_sequence: Option<u64>,
        tool_ok: Option<bool>,
        exit_status: Option<i32>,
        provider_metadata: Option<Value>,
        provider_fold_eligible: Option<bool>,
    ) -> Result<()> {
        self.append_metadata(TranscriptEvent::FoldedOutputMetadata {
            node_id,
            output_id: output_id.into(),
            output_kind: output_kind.into(),
            call_id,
            tool_name,
            stream,
            content,
            byte_count,
            line_count,
            truncated,
            shell_command,
            source_start_sequence,
            source_end_sequence,
            tool_ok,
            exit_status,
            provider_metadata,
            provider_fold_eligible,
        })
    }

    pub fn record_session_title(&mut self, title: impl Into<String>) -> Result<()> {
        self.append(TranscriptEvent::SessionTitle {
            title: title.into(),
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
                    parent_tool: subagent_tool_name_for_agent_name(agent_name.as_str())
                        .expect("subagent result recorded with unknown agent name")
                        .to_string(),
                    parent_turn_id: Some(parent_run_id),
                    parent_session_id: Some(parent_session_id),
                },
                tags: vec![agent_name, "subagent_result".into(), "unreconciled".into()],
            };
            self.record_evidence(evidence)?;
        }
        Ok(())
    }

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
                parent_tool: subagent_tool_name_for_agent_name(agent_name.as_str())
                    .expect("subagent reconciliation recorded with unknown agent name")
                    .to_string(),
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
            estimated_foldable_protected_tokens: telemetry.estimated_foldable_protected_tokens,
            estimated_provider_folded_protected_tokens: telemetry
                .estimated_provider_folded_protected_tokens,
            estimated_unaddressable_protected_tokens: telemetry
                .estimated_unaddressable_protected_tokens,
            provider_folded_output_count: telemetry.provider_folded_output_count,
            estimated_retained_history_tokens: telemetry.estimated_retained_history_tokens,
            estimated_tools_tokens: telemetry.estimated_tools_tokens,
            estimated_evidence_tokens: telemetry.estimated_evidence_tokens,
            estimated_required_fallback_tokens: telemetry.estimated_required_fallback_tokens,
            original_history_items: telemetry.original_history_items,
            retained_history_items: telemetry.retained_history_items,
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
        })
    }

    pub fn record_reasoning_message(&mut self, content: impl Into<String>) -> Result<()> {
        self.append(TranscriptEvent::ReasoningMessage {
            content: content.into(),
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
        calls: Vec<HistoryToolCall>,
    ) -> Result<()> {
        self.append(TranscriptEvent::AssistantToolCallBatch { text, calls })
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

    pub fn record_tool_call_finished_and_apply_context_control(
        &mut self,
        call_id: impl Into<String>,
        name: impl Into<String>,
        ok: bool,
        output: ToolResult,
    ) -> Result<()> {
        let call_id = call_id.into();
        let name = name.into();
        let output = if name == tool_names::TOOL_CONTEXT_RETURN && ok {
            self.enrich_context_return_output(output)?
        } else {
            output
        };
        if name == tool_names::TOOL_CONTEXT_CHECKPOINT && ok {
            let (events, experiment, branch_id) = self.context_checkpoint_transaction(
                TranscriptEvent::ToolCallFinished {
                    call_id,
                    name,
                    ok,
                    output: output.clone(),
                },
                &output,
            )?;
            self.append_transaction(events)?;
            self.current_context_branch_id = branch_id;
            self.set_active_context_experiment(Some(experiment));
        } else if name == tool_names::TOOL_CONTEXT_RETURN && ok {
            let (events, parent_branch_id) = self.context_return_transaction(
                TranscriptEvent::ToolCallFinished {
                    call_id,
                    name,
                    ok,
                    output: output.clone(),
                },
                &output,
            )?;
            self.append_transaction(events)?;
            self.current_context_branch_id = parent_branch_id;
            self.set_active_context_experiment(None);
        } else {
            self.record_tool_call_finished(call_id, name, ok, output)?;
        }
        Ok(())
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
        self.append(TranscriptEvent::PermissionDecision {
            call_id,
            tool: tool.into(),
            args,
            allowed,
            reason,
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
        if event.outcome == "succeeded" {
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
            // divergent history.  Compaction indices describe only that visible
            // branch/leaf, never the complete append-only journal.
            let scope = transcript_projection::context_compaction_validation_scope(
                &records,
                self.sequence,
                transcript_projection::SessionContextCursor {
                    branch_id: self.current_context_branch_id.clone(),
                    leaf_sequence: None,
                },
            )?;
            let event = if event.retired_source_spans.is_empty() {
                ContextCompactionEvent {
                    retired_source_spans: transcript_projection::derive_retired_source_spans(
                        scope.selected_history_records(),
                        event.tail_start_index,
                    ),
                    ..event
                }
            } else {
                event
            };
            transcript_projection::validate_context_compaction_event_in_scope(&scope, &event)?;
            // Validate every successful candidate against the exact record and
            // cursor that the durable append will use. Empty bindings are not a
            // compatibility escape for malformed modern compactions.
            transcript_projection::validate_context_compaction_candidate_replay(
                &self.session_id,
                &scope,
                &event,
            )?;
            // This acknowledgement is the commit point for compaction.  It must
            // survive a crash before the agent is allowed to swap its candidate.
            self.append_durable_on_branch(
                TranscriptEvent::ContextCompaction(event),
                scope.actual_append_branch_id().clone(),
            )
        } else {
            // Failed outcomes are append-only audit records and never affect the
            // runtime/history projection.
            let detail = event
                .detail
                .filter(|detail| !detail.trim().is_empty())
                .unwrap_or_else(|| "no additional detail".to_string());
            self.append(TranscriptEvent::Error {
                message: format!("context compaction {}: {}", event.outcome, detail),
            })
        }
    }

    /// Records exactly one logical-checkpoint event in an acknowledged journal
    /// transaction. This does not alter any live agent state.
    pub fn record_logical_checkpoint(&mut self, event: LogicalCheckpointEventV1) -> Result<()> {
        let expected_journal_frontier = self.sequence;
        let records = read_records(self.path())?;
        let branch_id =
            logical_checkpoint_branch_id(&records, self.current_context_branch_id.as_deref())?;
        self.record_logical_checkpoint_at_frontier(expected_journal_frontier, &branch_id, event)
    }

    pub(crate) fn prepare_logical_checkpoint(&self) -> Result<PreparedLogicalCheckpoint> {
        let records = read_records(self.path())?;
        ensure!(
            records
                .iter()
                .all(|record| record.session_id == self.session_id),
            "transcript contains records for a different session"
        );
        ensure!(
            records
                .iter()
                .map(|record| record.sequence)
                .max()
                .unwrap_or(0)
                == self.sequence,
            "logical checkpoint recorder frontier does not match committed transcript"
        );
        let branch_id =
            logical_checkpoint_branch_id(&records, self.current_context_branch_id.as_deref())?;
        let event = transcript_projection::prepare_logical_checkpoint_candidate(
            &self.session_id,
            &records,
            branch_id.clone(),
            self.sequence,
        )?;
        let checkpoint_sequence = self
            .sequence
            .checked_add(1)
            .ok_or_else(|| anyhow!("logical checkpoint journal frontier overflow"))?;
        let mut candidate_records = records;
        candidate_records.push(TranscriptRecord {
            session_id: self.session_id.clone(),
            sequence: checkpoint_sequence,
            timestamp_ms: 0,
            context_branch_id: Some(branch_id.clone()),
            event: TranscriptEvent::LogicalCheckpoint(event.clone()),
        });
        let projected = transcript_projection::project_runtime_restore_snapshot(
            self.session_id.clone(),
            candidate_records,
            transcript_projection::SessionContextCursor {
                branch_id: Some(branch_id.clone()),
                leaf_sequence: Some(checkpoint_sequence),
            },
            &[],
        )?;
        Ok(PreparedLogicalCheckpoint {
            expected_journal_frontier: self.sequence,
            expected_branch_id: branch_id,
            projected_workflow: event
                .retained_items
                .iter()
                .find(|item| item.kind == LogicalCheckpointRetainedKindV1::WorkflowState)
                .map(|item| serde_json::from_str(&item.detail))
                .transpose()
                .context("logical checkpoint workflow item has invalid typed detail")?,
            projected_snapshot: projected.snapshot,
            projected_protocol_frames: projected.protocol_frames,
            event,
        })
    }

    /// Reject stale candidates before validation or any durable write.
    pub fn record_logical_checkpoint_at_frontier(
        &mut self,
        expected_journal_frontier: u64,
        expected_branch_id: &str,
        event: LogicalCheckpointEventV1,
    ) -> Result<()> {
        ensure!(
            self.sequence == expected_journal_frontier,
            "logical checkpoint candidate is stale: expected journal frontier {}, found {}",
            expected_journal_frontier,
            self.sequence
        );
        let records = read_records(self.path())?;
        // The recorder cursor is authoritative while it is set. Only restored
        // recorders without a cursor consult the journal's latest checkout.
        let branch_id =
            logical_checkpoint_branch_id(&records, self.current_context_branch_id.as_deref())?;
        ensure!(
            branch_id == expected_branch_id,
            "logical checkpoint candidate is stale: expected branch '{}', found '{}'",
            expected_branch_id,
            branch_id
        );
        let sequence = self
            .sequence
            .checked_add(1)
            .ok_or_else(|| anyhow!("transcript sequence overflow"))?;
        transcript_projection::validate_logical_checkpoint_candidate(
            &self.session_id,
            &records,
            Some(branch_id.clone()),
            self.sequence,
            sequence,
            &event,
        )?;
        // Checkpoints are always explicitly branch-scoped.  Root checkpoints
        // must not use the legacy global (`None`) scope.
        self.append_transaction(vec![(
            TranscriptEvent::LogicalCheckpoint(event),
            Some(branch_id),
        )])
    }

    pub fn record_turn_finalized(&mut self, event: TurnFinalizedEvent) -> Result<()> {
        self.append(TranscriptEvent::TurnFinalized(event))
    }

    pub fn record_turn_interrupted(&mut self, turn_id: Option<u64>) -> Result<()> {
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

    fn context_checkpoint_transaction(
        &self,
        tool_finished: TranscriptEvent,
        output: &ToolResult,
    ) -> Result<(
        Vec<(TranscriptEvent, Option<String>)>,
        ActiveContextExperiment,
        Option<String>,
    )> {
        ensure!(
            self.active_context_experiment().is_none(),
            "context__checkpoint cannot start a nested experiment while another experiment is active"
        );
        let data = output
            .data
            .as_ref()
            .ok_or_else(|| anyhow!("context__checkpoint requires output data"))?;
        let label = match data.get("label") {
            Some(Value::String(label)) => Some(label.trim().to_string()),
            Some(Value::Null) | None => None,
            Some(_) => {
                return Err(anyhow!(
                    "context__checkpoint output field 'label' must be string or null"
                ));
            }
        }
        .filter(|label| !label.is_empty());
        let purpose = match data.get("reason") {
            Some(Value::String(reason)) => Some(reason.trim().to_string()),
            Some(Value::Null) | None => None,
            Some(_) => {
                return Err(anyhow!(
                    "context__checkpoint output field 'reason' must be string or null"
                ));
            }
        }
        .filter(|reason| !reason.is_empty());

        let mut snapshot = self.active_context_snapshot()?;
        // The first record in this transaction is the successful tool result;
        // the branch forks from that durable parent-side fact.
        snapshot.leaf_sequence = snapshot
            .leaf_sequence
            .checked_add(1)
            .ok_or_else(|| anyhow!("transcript sequence overflow"))?;
        let parent_node_id = self.current_active_context_node_id()?;
        let records = read_records(self.path())?;
        let branches = transcript_projection::list_context_branches(
            &records,
            self.current_context_branch_id(),
        )?;
        let branch_id = next_context_branch_id(&branches, label.as_deref());
        let branch_node_id = context_node_id_for_branch(&branch_id);
        let mut tree = self.current_context_tree_state()?;
        tree.apply(&ContextTreeOp::CreateNode {
            node_id: ContextNodeId::new(branch_node_id.clone())?,
            parent_node_id: Some(ContextNodeId::new(parent_node_id.clone())?),
            label: label.clone(),
            purpose: purpose.clone(),
            block_ref: None,
            source_ref: Some(ContextSourceRef {
                source_kind: "context_branch".into(),
                source_id: Some(branch_id.clone()),
            }),
        })?;
        tree.apply(&ContextTreeOp::SetNodeStatus {
            node_id: ContextNodeId::new(parent_node_id.clone())?,
            status: ContextNodeStatus::Inactive,
        })?;
        tree.apply(&ContextTreeOp::SetNodeStatus {
            node_id: ContextNodeId::new(branch_node_id.clone())?,
            status: ContextNodeStatus::Active,
        })?;

        let events = vec![
            (tool_finished, self.current_context_branch_id.clone()),
            (
                TranscriptEvent::ContextBranchCreated {
                    branch_id: branch_id.clone(),
                    parent_branch_id: snapshot.branch_id.clone(),
                    base_sequence: snapshot.leaf_sequence,
                    label: label.clone(),
                },
                None,
            ),
            (
                TranscriptEvent::ContextCheckout {
                    branch_id: branch_id.clone(),
                    leaf_sequence: snapshot.leaf_sequence,
                },
                None,
            ),
            (
                TranscriptEvent::ContextExperimentStarted {
                    branch_id: branch_id.clone(),
                    parent_branch_id: snapshot.branch_id.clone(),
                    base_sequence: snapshot.leaf_sequence,
                },
                None,
            ),
            (
                TranscriptEvent::ContextNodeCreated {
                    node_id: branch_node_id.clone(),
                    parent_node_id: Some(parent_node_id.clone()),
                    label,
                    purpose,
                    block_ref: None,
                    source_ref: Some(ContextSourceRef {
                        source_kind: "context_branch".into(),
                        source_id: Some(branch_id.clone()),
                    }),
                },
                None,
            ),
            (
                TranscriptEvent::ContextNodeLifecycle {
                    node_id: parent_node_id,
                    status: ContextNodeStatus::Inactive,
                },
                None,
            ),
            (
                TranscriptEvent::ContextNodeLifecycle {
                    node_id: branch_node_id,
                    status: ContextNodeStatus::Active,
                },
                None,
            ),
        ];
        Ok((
            events,
            ActiveContextExperiment {
                branch_id: branch_id.clone(),
                parent_branch_id: snapshot.branch_id,
                base_sequence: snapshot.leaf_sequence,
                writes_observed: false,
            },
            Some(branch_id),
        ))
    }

    fn enrich_context_return_output(&self, mut output: ToolResult) -> Result<ToolResult> {
        let Some(experiment) = self.active_context_experiment() else {
            return Err(anyhow!(
                "context__return requires an active context experiment"
            ));
        };
        if !experiment.writes_observed {
            return Ok(output);
        }
        let Some(data) = output.data.as_mut() else {
            return Ok(output);
        };
        data["warning"] = Value::String("Context restored, files were NOT reverted".to_string());
        if let Some(message) = data.get("message").and_then(Value::as_str) {
            data["message"] = Value::String(format!(
                "{message} Context restored, files were NOT reverted."
            ));
        }
        Ok(output)
    }

    fn context_return_transaction(
        &self,
        tool_finished: TranscriptEvent,
        output: &ToolResult,
    ) -> Result<(Vec<(TranscriptEvent, Option<String>)>, Option<String>)> {
        let experiment = self
            .active_context_experiment()
            .ok_or_else(|| anyhow!("context__return requires an active context experiment"))?;
        ensure!(
            self.current_context_branch_id() == Some(experiment.branch_id.as_str()),
            "context__return must finish on the active experiment branch"
        );
        let data = output
            .data
            .as_ref()
            .ok_or_else(|| anyhow!("context__return requires output data"))?;
        let outcome = required_context_return_string(data, "outcome")?;
        let summary = required_context_return_string(data, "summary")?;
        let next_action = optional_context_return_string(data, "next_action")?;
        let records = read_records(self.path())?;
        let branches = transcript_projection::list_context_branches(
            &records,
            self.current_context_branch_id(),
        )?;
        let parent_tip = branches
            .iter()
            .find(|branch| branch.branch_id == experiment.parent_branch_id)
            .map(|branch| branch.tip_sequence)
            .ok_or_else(|| {
                anyhow!(
                    "parent context branch '{}' is missing during context__return",
                    experiment.parent_branch_id
                )
            })?;
        let active_node = self.current_active_context_node_id()?;
        let expected_node_id = context_node_id_for_branch(&experiment.branch_id);
        ensure!(
            active_node == expected_node_id,
            "active context node '{}' does not match experiment branch '{}'",
            active_node,
            experiment.branch_id
        );
        let parent_node_id = self
            .current_context_tree_state()?
            .node(&ContextNodeId::new(active_node.clone())?)
            .and_then(|node| node.parent_node_id.as_ref())
            .map(|node_id| node_id.as_str().to_string())
            .ok_or_else(|| {
                anyhow!(
                    "active experiment context node '{}' is missing a parent",
                    active_node
                )
            })?;

        let mut tree = self.current_context_tree_state()?;
        tree.apply(&ContextTreeOp::SetNodeStatus {
            node_id: ContextNodeId::new(active_node.clone())?,
            status: ContextNodeStatus::Archived,
        })?;
        tree.apply(&ContextTreeOp::SetNodeStatus {
            node_id: ContextNodeId::new(parent_node_id.clone())?,
            status: ContextNodeStatus::Active,
        })?;
        let events = vec![
            (tool_finished, self.current_context_branch_id.clone()),
            (
                TranscriptEvent::ContextCheckout {
                    branch_id: experiment.parent_branch_id.clone(),
                    leaf_sequence: parent_tip,
                },
                None,
            ),
            (
                TranscriptEvent::ContextExperimentReturned {
                    branch_id: experiment.branch_id.clone(),
                    parent_branch_id: experiment.parent_branch_id.clone(),
                    base_sequence: experiment.base_sequence,
                    outcome,
                    summary,
                    next_action,
                    had_writes: experiment.writes_observed,
                },
                None,
            ),
            (
                TranscriptEvent::ContextNodeLifecycle {
                    node_id: active_node,
                    status: ContextNodeStatus::Archived,
                },
                None,
            ),
            (
                TranscriptEvent::ContextNodeLifecycle {
                    node_id: parent_node_id,
                    status: ContextNodeStatus::Active,
                },
                None,
            ),
        ];
        let parent = if experiment.parent_branch_id == ROOT_CONTEXT_BRANCH_ID {
            None
        } else {
            Some(experiment.parent_branch_id)
        };
        Ok((events, parent))
    }

    fn active_context_snapshot(&self) -> Result<transcript_projection::SessionRestoreSnapshot> {
        transcript_projection::build_session_context_snapshot(
            self.session_id().to_string(),
            read_records(self.path())?,
            transcript_projection::SessionContextCursor {
                branch_id: self.current_context_branch_id().map(str::to_string),
                leaf_sequence: None,
            },
        )
    }

    fn set_active_context_experiment(&self, experiment: Option<ActiveContextExperiment>) {
        if let Ok(mut state) = self.context_scope_state.lock() {
            state.active_experiment = experiment;
        }
    }

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

    fn current_context_tree_state(&self) -> Result<crate::context_tree::ContextTreeState> {
        transcript_projection::project_context_tree(&read_records(self.path())?)
    }

    fn current_active_context_node_id(&self) -> Result<String> {
        self.current_context_tree_state()?
            .active_node_id()
            .map(|node_id| node_id.as_str().to_string())
            .ok_or_else(|| anyhow!("context tree has no active node"))
    }

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
        Ok(())
    }

    pub fn append_transaction(
        &mut self,
        events: Vec<(TranscriptEvent, Option<String>)>,
    ) -> Result<()> {
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
        for (index, (event, context_branch_id)) in events.into_iter().enumerate() {
            let sequence = base_revision + u64::try_from(index).unwrap() + 1;
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

fn context_node_id_for_branch(branch_id: &str) -> String {
    format!("branch/{branch_id}")
}

#[allow(dead_code)]
pub fn read_records(path: impl AsRef<Path>) -> Result<Vec<TranscriptRecord>> {
    read_records_inner(path, false)
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

fn has_uncommitted_transaction_tail(path: &Path) -> Result<bool> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("failed to read transcript {}", path.display()))?;
    let mut pending = false;
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
                Some(_) => {
                    ensure!(
                        !pending
                            || entry
                                .v1
                                .as_ref()
                                .and_then(|record| record.transaction_index)
                                != Some(0),
                        "transcript transaction is interrupted by a different transaction"
                    );
                    pending = true;
                }
                None => ensure!(
                    !pending,
                    "transcript transaction is missing its commit marker before another record"
                ),
            },
            ParsedJournalLine::Commit(_) => pending = false,
        }
    }
    Ok(pending)
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
}

pub fn sort_child_session_summaries(children: &mut [ChildSessionSummary]) {
    children.sort_by(|left, right| {
        left.timestamp_ms
            .cmp(&right.timestamp_ms)
            .then_with(|| left.child_session_id.cmp(&right.child_session_id))
    });
}

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

    let mut sessions = Vec::new();

    for entry in fs::read_dir(base_dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.extension().and_then(|ext| ext.to_str()) != Some("jsonl") {
            continue;
        }

        let session_id = match path.file_stem().and_then(|stem| stem.to_str()) {
            Some(session_id) => session_id.to_string(),
            None => continue,
        };

        let records = read_records(&path)?;
        if !has_session_content(&records) {
            continue;
        }

        let first_timestamp_ms = records.first().map(|record| record.timestamp_ms);
        let last_timestamp_ms = records.last().map(|record| record.timestamp_ms);
        let model = restore_latest_model(&records);
        let title = records.iter().rev().find_map(|record| match &record.event {
            TranscriptEvent::SessionTitle { title } => Some(title.clone()),
            _ => None,
        });
        let last_user_summary = records.iter().rev().find_map(|record| match &record.event {
            TranscriptEvent::UserMessage { content } => {
                Some(summarize_text(&content.display_text()))
            }
            _ => None,
        });
        let last_assistant_summary = records.iter().rev().find_map(|record| match &record.event {
            TranscriptEvent::AssistantMessage { content } => Some(summarize_text(content)),
            _ => None,
        });

        sessions.push(SessionSummary {
            session_id,
            record_count: records.len(),
            first_timestamp_ms,
            last_timestamp_ms,
            model,
            title,
            last_user_summary,
            last_assistant_summary,
        });
    }

    sessions.sort_by_key(|session| session.last_timestamp_ms.unwrap_or(0));
    sessions.reverse();

    Ok(sessions)
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

pub fn restore_job_board(
    base_dir: impl AsRef<Path>,
    parent_records: &[TranscriptRecord],
) -> Result<Vec<JobBoardEntry>> {
    transcript_projection::project_job_board(&child_sessions_dir(base_dir), parent_records)
}

pub fn restore_session_history(records: &[TranscriptRecord]) -> Result<Vec<HistoryItem>> {
    transcript_projection::validate_context_projection_events(records)?;
    Ok(transcript_projection::restore_session_history_projection(
        records,
    ))
}

pub(crate) fn restore_session_protocol_frames(
    records: &[TranscriptRecord],
) -> Result<Vec<crate::protocol_frames::ProtocolFrame>> {
    transcript_projection::restore_session_protocol_frames_projection(records)
}

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

pub fn restore_compacted_conversation_messages(
    records: &[TranscriptRecord],
) -> Result<Vec<ConversationMessage>> {
    Ok(restore_session_history(records)?
        .into_iter()
        .filter_map(history_item_to_conversation_message)
        .collect())
}

pub fn restore_conversation_messages(
    records: &[TranscriptRecord],
) -> Result<Vec<ConversationMessage>> {
    restore_compacted_conversation_messages(records)
}

pub fn restore_latest_model(records: &[TranscriptRecord]) -> Option<String> {
    transcript_projection::restore_latest_model_projection(records)
}

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

pub fn restore_max_turn_id(records: &[TranscriptRecord]) -> u64 {
    transcript_projection::restore_max_turn_id_projection(records)
}

fn logical_checkpoint_branch_id(
    records: &[TranscriptRecord],
    current_context_branch_id: Option<&str>,
) -> Result<String> {
    match current_context_branch_id {
        Some(branch_id) => Ok(branch_id.to_string()),
        None => transcript_projection::effective_branch_id_at_frontier(records),
    }
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
                | Self::ContextExperimentStarted { .. }
                | Self::ContextNodeCreated { .. }
                | Self::ContextNodeLifecycle { .. }
                | Self::ContextViewOperationMetadata { .. }
                | Self::ContextSummaryArtifactMetadata { .. }
                | Self::FoldedOutputMetadata { .. }
        )
    }

    fn is_session_content(&self) -> bool {
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
            | TranscriptEvent::SubagentLifecycle { .. }
            | TranscriptEvent::SubagentResult { .. }
            | TranscriptEvent::ContextBranchCreated { .. }
            | TranscriptEvent::ContextBranchSummary { .. }
            | TranscriptEvent::ContextCheckout { .. }
            | TranscriptEvent::ContextExperimentStarted { .. }
            | TranscriptEvent::ContextNodeCreated { .. }
            | TranscriptEvent::ContextNodeLifecycle { .. }
            | TranscriptEvent::ContextViewOperationMetadata { .. }
            | TranscriptEvent::ContextSummaryArtifactMetadata { .. }
            | TranscriptEvent::FoldedOutputMetadata { .. }
            | TranscriptEvent::ContextExperimentReturned { .. }
            | TranscriptEvent::UserMessage { .. }
            | TranscriptEvent::AssistantMessage { .. }
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
    ) || matches!(event, TranscriptEvent::ContextCompaction(event) if event.outcome == "succeeded")
}

pub(crate) fn sync_recorder_branch(recorder: &mut TranscriptRecorder, branch_id: &str) {
    if branch_id == ROOT_CONTEXT_BRANCH_ID {
        recorder.set_current_context_branch_id(None);
    } else {
        recorder.set_current_context_branch_id(Some(branch_id.to_string()));
    }
}

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

fn slugify_branch_label(label: &str) -> String {
    let mut slug = String::new();
    let mut last_was_dash = false;
    for ch in label.chars().flat_map(|ch| ch.to_lowercase()) {
        if ch.is_ascii_alphanumeric() {
            slug.push(ch);
            last_was_dash = false;
        } else if !last_was_dash {
            slug.push('-');
            last_was_dash = true;
        }
    }
    slug.trim_matches('-').to_string()
}

fn next_context_branch_id(
    branches: &[transcript_projection::ContextBranchInfo],
    label: Option<&str>,
) -> String {
    let existing = branches
        .iter()
        .map(|branch| branch.branch_id.as_str())
        .collect::<std::collections::BTreeSet<_>>();
    let base = label
        .map(slugify_branch_label)
        .filter(|slug| !slug.is_empty())
        .unwrap_or_else(|| "branch".into());
    if !existing.contains(base.as_str()) {
        return base;
    }
    let mut suffix = 2u64;
    loop {
        let candidate = format!("{base}-{suffix}");
        if !existing.contains(candidate.as_str()) {
            return candidate;
        }
        suffix += 1;
    }
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
        TranscriptEvent::AssistantToolCallBatch { text, calls } => {
            Some(HistoryItem::AssistantToolCalls {
                text: text.clone(),
                calls: calls.clone(),
            })
        }
        TranscriptEvent::ToolCallStarted {
            call_id,
            name,
            args,
        } => Some(HistoryItem::AssistantToolCalls {
            text: None,
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
            output_json: serde_json::to_string(output).unwrap_or_else(|_| "null".to_string()),
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

fn required_metadata_string(metadata: &Value, field: &str) -> Result<String> {
    metadata
        .get(field)
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .ok_or_else(|| anyhow!("invalid pending metadata: missing string field '{field}'"))
}

fn optional_metadata_string(metadata: &Value, field: &str) -> Option<String> {
    metadata
        .get(field)
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
}

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

fn context_tool_allows_pending_metadata(tool_name: &str) -> bool {
    matches!(
        tool_name,
        tool_names::TOOL_CONTEXT_PIN
            | tool_names::TOOL_CONTEXT_ARCHIVE
            | tool_names::TOOL_CONTEXT_REMOVE
            | tool_names::TOOL_CONTEXT_RESOLVE
            | tool_names::TOOL_CONTEXT_SUMMARIZE
            | tool_names::TOOL_CONTEXT_OPEN
    )
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

    fs::remove_file(path).with_context(|| {
        format!(
            "failed to remove empty session transcript '{}'",
            path.display()
        )
    })?;
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

fn summarize_text(content: &str) -> String {
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
mod tests;
