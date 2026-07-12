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
    AutoContinueState, ContextCompactionEvent, ConversationMessage, ConversationRole, TodoItem,
    ToolExecutionSummaryEvent, TurnFinalizedEvent, TurnStartedEvent, ValidationAdvisory,
};
use crate::context_tree::{
    ContextBlockRef, ContextNodeId, ContextNodeStatus, ContextSourceRef, ContextTreeOp,
};
use crate::evidence::{
    EvidenceDraft, EvidenceKind, EvidenceRecord, EvidenceSource, evidence_id_for_sequence,
    restore_evidence_records,
};
use crate::request_builder::{HistoryItem, HistoryToolCall};
use crate::subagent::StructuredSubagentResult;
use crate::tool::ToolResult;
use crate::tool_names;
use crate::user_content::UserMessageContent;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InternalContinuationSource {
    Legacy,
    AutoContinue,
    StreamRecovery,
}

impl Default for InternalContinuationSource {
    fn default() -> Self {
        Self::Legacy
    }
}

#[path = "transcript_projection.rs"]
pub(crate) mod transcript_projection;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptRecord {
    pub session_id: String,
    pub sequence: u64,
    pub timestamp_ms: u128,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_branch_id: Option<String>,
    #[serde(flatten)]
    pub event: TranscriptEvent,
}

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

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TranscriptEvent {
    SessionStarted {
        model: String,
    },
    SessionTitle {
        title: String,
    },
    SubagentLifecycle {
        run_id: String,
        parent_session_id: String,
        parent_run_id: String,
        agent_name: String,
        status: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        detail: Option<String>,
    },
    SubagentResult {
        run_id: String,
        parent_session_id: String,
        parent_run_id: String,
        child_session_id: String,
        agent_name: String,
        status: String,
        summary: String,
    },
    ModelChanged {
        previous_model: String,
        new_model: String,
    },
    ContextBranchCreated {
        branch_id: String,
        parent_branch_id: String,
        base_sequence: u64,
        #[serde(skip_serializing_if = "Option::is_none")]
        label: Option<String>,
    },
    ContextBranchSummary {
        branch_id: String,
        leaf_sequence: u64,
        summary: String,
    },
    ContextCheckout {
        branch_id: String,
        leaf_sequence: u64,
    },
    ContextExperimentStarted {
        branch_id: String,
        parent_branch_id: String,
        base_sequence: u64,
    },
    ContextNodeCreated {
        node_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        parent_node_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        label: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        purpose: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        block_ref: Option<ContextBlockRef>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        source_ref: Option<ContextSourceRef>,
    },
    ContextNodeLifecycle {
        node_id: String,
        status: ContextNodeStatus,
    },
    ContextViewOperationMetadata {
        operation: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        node_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        block_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        detail: Option<String>,
    },
    ContextSummaryArtifactMetadata {
        node_id: String,
        artifact_id: String,
        artifact_kind: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        version: Option<u32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        summary: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        source_node_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        source_block_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        source_start_sequence: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        source_end_sequence: Option<u64>,
    },
    FoldedOutputMetadata {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        node_id: Option<String>,
        output_id: String,
        output_kind: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        call_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tool_name: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        stream: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        content: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        byte_count: Option<usize>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        line_count: Option<usize>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        truncated: Option<bool>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        shell_command: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        source_start_sequence: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        source_end_sequence: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tool_ok: Option<bool>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        exit_status: Option<i32>,
    },
    UserMessage {
        content: UserMessageContent,
    },
    TurnStarted(TurnStartedEvent),
    AssistantMessage {
        content: String,
    },
    ReasoningMessage {
        content: String,
    },
    AssistantToolCallBatch {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        text: Option<String>,
        calls: Vec<HistoryToolCall>,
    },
    ToolCallStarted {
        call_id: String,
        name: String,
        args: Value,
    },
    ToolCallFinished {
        call_id: String,
        name: String,
        ok: bool,
        output: ToolResult,
    },
    ToolCallCancelled {
        call_id: String,
        name: String,
    },
    PermissionDecision {
        #[serde(skip_serializing_if = "Option::is_none")]
        call_id: Option<String>,
        tool: String,
        args: Value,
        allowed: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
    },
    PermissionModeChanged {
        previous_mode: String,
        new_mode: String,
    },
    TodoSnapshot {
        items: Vec<TodoItem>,
    },
    AutoContinueChanged {
        state: AutoContinueState,
    },
    AutoContinuationScheduled {
        continuation_count: usize,
        remaining_unfinished: usize,
    },
    InternalContinuation {
        text: String,
        #[serde(default)]
        source: InternalContinuationSource,
    },
    ValidationAdvisory(ValidationAdvisory),
    ToolExecutionSummary(ToolExecutionSummaryEvent),
    ContextCompaction(ContextCompactionEvent),
    TurnFinalized(TurnFinalizedEvent),
    TurnInterrupted {
        #[serde(skip_serializing_if = "Option::is_none")]
        turn_id: Option<u64>,
    },
    Evidence {
        id: String,
        evidence_kind: EvidenceKind,
        title: String,
        summary: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        detail: Option<String>,
        source: EvidenceSource,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        tags: Vec<String>,
    },
    ContextExperimentReturned {
        branch_id: String,
        parent_branch_id: String,
        base_sequence: u64,
        outcome: String,
        summary: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        next_action: Option<String>,
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        had_writes: bool,
    },
    Error {
        message: String,
    },
    #[serde(other)]
    Unknown,
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
            // A recorder can be positioned on a branch whose sibling has a
            // divergent history.  Compaction indices describe only that visible
            // branch/leaf, never the complete append-only journal.
            let visible = transcript_projection::build_session_context_snapshot(
                self.session_id.clone(),
                records.clone(),
                transcript_projection::SessionContextCursor {
                    branch_id: self.current_context_branch_id.clone(),
                    leaf_sequence: Some(self.sequence),
                },
            )?;
            let event = if event.retired_source_spans.is_empty() {
                ContextCompactionEvent {
                    retired_source_spans: transcript_projection::derive_retired_source_spans(
                        &visible.records,
                        event.tail_start_index,
                    ),
                    ..event
                }
            } else {
                event
            };
            transcript_projection::validate_context_compaction_event(&visible.records, &event)?;
            // Validate modern durable bindings against the same projected
            // candidate that replay will see. This keeps malformed bindings
            // from becoming a durable acknowledgement in the first place.
            if !event.frame_identity_bindings.is_empty() {
                let mut candidate_records = visible.records.clone();
                candidate_records.push(TranscriptRecord {
                    session_id: self.session_id.clone(),
                    sequence: self.sequence.saturating_add(1),
                    timestamp_ms: 0,
                    context_branch_id: self.current_context_branch_id.clone(),
                    event: TranscriptEvent::ContextCompaction(event.clone()),
                });
                transcript_projection::project_runtime_restore_snapshot(
                    self.session_id.clone(),
                    candidate_records,
                    transcript_projection::SessionContextCursor {
                        branch_id: self.current_context_branch_id.clone(),
                        leaf_sequence: Some(self.sequence.saturating_add(1)),
                    },
                    &[],
                )?;
            }
            // This acknowledgement is the commit point for compaction.  It must
            // survive a crash before the agent is allowed to swap its candidate.
            self.append_durable(TranscriptEvent::ContextCompaction(event))
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
        let mut line = serde_json::to_vec(&envelope)?;
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
            serde_json::to_writer(&mut payload, &envelope)?;
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
                        serde_json::to_writer(&mut pending.payload, entry.v1.as_ref().unwrap())?;
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
    if has_top_level_json_field(line, "schema_version") {
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

    let metadata: JournalMetadata = serde_json::from_str(line)?;
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

pub fn restore_session_history(records: &[TranscriptRecord]) -> Vec<HistoryItem> {
    transcript_projection::restore_session_history_projection(records)
}

pub(crate) fn restore_session_protocol_frames(
    records: &[TranscriptRecord],
) -> Vec<crate::protocol_frames::ProtocolFrame> {
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
) -> Vec<ConversationMessage> {
    restore_session_history(records)
        .into_iter()
        .filter_map(history_item_to_conversation_message)
        .collect()
}

pub fn restore_conversation_messages(records: &[TranscriptRecord]) -> Vec<ConversationMessage> {
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

fn truncate_text(content: &str, max_chars: usize) -> String {
    let mut truncated = content.chars().take(max_chars).collect::<String>();
    if content.chars().count() > max_chars {
        truncated.push('…');
    }
    truncated
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ApiProtocol;
    use crate::protocol_frames::{analyze_history_items, history_items_from_frames};
    use crate::request_builder::{ModelRequestMetadata, RequestBuilderInput, build_request};
    use crate::subagent::StructuredSubagentResult;
    use serde_json::json;
    use std::sync::{Arc, Mutex};

    #[derive(Clone, Copy)]
    enum FailPoint {
        Write,
        Flush,
        Sync,
    }

    struct FailingSink {
        fail: FailPoint,
        calls: Arc<Mutex<Vec<&'static str>>>,
    }

    impl JournalSink for FailingSink {
        fn write_all(&mut self, _: &[u8]) -> io::Result<()> {
            self.calls.lock().unwrap().push("write");
            if matches!(self.fail, FailPoint::Write) {
                Err(io::Error::other("injected write failure"))
            } else {
                Ok(())
            }
        }

        fn flush(&mut self) -> io::Result<()> {
            self.calls.lock().unwrap().push("flush");
            if matches!(self.fail, FailPoint::Flush) {
                Err(io::Error::other("injected flush failure"))
            } else {
                Ok(())
            }
        }

        fn sync_data(&mut self) -> io::Result<()> {
            self.calls.lock().unwrap().push("sync");
            if matches!(self.fail, FailPoint::Sync) {
                Err(io::Error::other("injected sync failure"))
            } else {
                Ok(())
            }
        }
    }

    fn recorder_with_sink(sink: impl JournalSink + 'static) -> TranscriptRecorder {
        TranscriptRecorder {
            session_id: "test-session".into(),
            path: PathBuf::from("unused.jsonl"),
            sink: Box::new(sink),
            sequence: 0,
            health: RecorderHealth::Healthy,
            current_context_branch_id: None,
            context_scope_state: Arc::new(Mutex::new(ContextScopeState::default())),
        }
    }

    fn journal_test_dir(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("letcode-journal-{name}-{}", unix_timestamp_ms()))
    }

    fn legacy_record(sequence: u64) -> TranscriptRecord {
        TranscriptRecord {
            session_id: "session".into(),
            sequence,
            timestamp_ms: sequence as u128,
            context_branch_id: None,
            event: TranscriptEvent::SessionTitle {
                title: format!("title-{sequence}"),
            },
        }
    }

    fn v1_record(sequence: u64) -> JournalRecordV1 {
        let record = legacy_record(sequence);
        JournalRecordV1 {
            schema_version: JOURNAL_SCHEMA_VERSION,
            event_id: format!("{}:{sequence}", record.session_id),
            scope: journal_scope_for(&record),
            base_revision: sequence - 1,
            resulting_revision: sequence,
            transaction_id: None,
            transaction_index: None,
            transaction_count: None,
            record,
        }
    }

    #[test]
    fn journal_v1_round_trips_and_writes_envelope() {
        let base_dir = journal_test_dir("v1-roundtrip");
        let mut recorder = TranscriptRecorder::create(&base_dir).unwrap();
        recorder.record_user_message("hello").unwrap();
        let path = base_dir.join(format!("{}.jsonl", recorder.session_id()));
        let raw = fs::read_to_string(&path).unwrap();
        assert!(raw.contains("\"schema_version\":1"));
        assert!(raw.contains("\"scope\":\"global\""));
        assert!(raw.contains("\"base_revision\":0"));
        assert!(raw.contains("\"resulting_revision\":1"));
        let records = read_records(path).unwrap();
        assert_eq!(records.len(), 1);
        assert!(matches!(
            records[0].event,
            TranscriptEvent::UserMessage { .. }
        ));
    }

    #[test]
    fn journal_reader_accepts_legacy_and_legacy_to_v1_records() {
        let base_dir = journal_test_dir("legacy");
        fs::create_dir_all(&base_dir).unwrap();
        let legacy_path = base_dir.join("legacy.jsonl");
        fs::write(
            &legacy_path,
            format!("{}\n", serde_json::to_string(&legacy_record(1)).unwrap()),
        )
        .unwrap();
        assert_eq!(read_records(&legacy_path).unwrap()[0].sequence, 1);

        let mixed_path = base_dir.join("mixed.jsonl");
        fs::write(
            &mixed_path,
            format!(
                "{}\n{}\n",
                serde_json::to_string(&legacy_record(1)).unwrap(),
                serde_json::to_string(&v1_record(2)).unwrap()
            ),
        )
        .unwrap();
        let records = read_records(&mixed_path).unwrap();
        assert_eq!(
            records
                .iter()
                .map(|record| record.sequence)
                .collect::<Vec<_>>(),
            [1, 2]
        );
    }

    #[test]
    fn journal_reader_rejects_invalid_contracts() {
        let base_dir = journal_test_dir("invalid");
        fs::create_dir_all(&base_dir).unwrap();
        let cases = [
            ("duplicate-event", {
                let first = v1_record(1);
                let mut second = v1_record(2);
                second.event_id = first.event_id.clone();
                vec![first, second]
            }),
            ("revision", {
                let first = v1_record(1);
                let mut second = v1_record(2);
                second.base_revision = 0;
                vec![first, second]
            }),
            ("sequence", {
                let first = v1_record(1);
                let second = v1_record(1);
                vec![first, second]
            }),
            ("session", {
                let first = v1_record(1);
                let mut second = v1_record(2);
                second.record.session_id = "other".into();
                vec![first, second]
            }),
            ("scope", {
                let first = v1_record(1);
                let mut second = v1_record(2);
                second.scope = JournalScope::Branch;
                vec![first, second]
            }),
        ];
        for (name, records) in cases {
            let path = base_dir.join(format!("{name}.jsonl"));
            fs::write(
                &path,
                records
                    .iter()
                    .map(|record| serde_json::to_string(record).unwrap())
                    .collect::<Vec<_>>()
                    .join("\n")
                    + "\n",
            )
            .unwrap();
            assert!(read_records(path).is_err(), "{name} must be rejected");
        }
    }

    #[test]
    fn journal_reader_rejects_v1_with_nonzero_initial_base_revision() {
        let base_dir = journal_test_dir("v1-initial-base");
        fs::create_dir_all(&base_dir).unwrap();
        let path = base_dir.join("invalid.jsonl");
        let record = v1_record(2);
        fs::write(
            &path,
            format!("{}\n", serde_json::to_string(&record).unwrap()),
        )
        .unwrap();

        assert!(read_records(path).is_err());
    }

    #[test]
    fn journal_reader_rejects_v1_with_forged_event_id() {
        let base_dir = journal_test_dir("v1-event-id");
        fs::create_dir_all(&base_dir).unwrap();
        let path = base_dir.join("invalid.jsonl");
        let mut record = v1_record(1);
        record.event_id = "forged:1".into();
        fs::write(
            &path,
            format!("{}\n", serde_json::to_string(&record).unwrap()),
        )
        .unwrap();

        assert!(read_records(path).is_err());
    }

    #[test]
    fn journal_io_failures_poison_recorder_without_advancing_sequence() {
        for fail in [FailPoint::Write, FailPoint::Flush, FailPoint::Sync] {
            let calls = Arc::new(Mutex::new(Vec::new()));
            let mut recorder = recorder_with_sink(FailingSink {
                fail,
                calls: Arc::clone(&calls),
            });
            assert!(recorder.record_user_message("first").is_err());
            assert_eq!(recorder.health, RecorderHealth::Poisoned);
            assert_eq!(recorder.sequence, 0);
            assert!(recorder.record_user_message("second").is_err());
            assert_eq!(recorder.sequence, 0);
            let call_count = calls.lock().unwrap().len();
            assert_eq!(
                call_count,
                match fail {
                    FailPoint::Write => 1,
                    FailPoint::Flush => 2,
                    FailPoint::Sync => 3,
                }
            );
        }
    }

    #[test]
    fn transaction_round_trip_commits_all_records_and_uncommitted_tail_is_ignored() {
        let base_dir = journal_test_dir("transaction-tail");
        let mut recorder = TranscriptRecorder::create(&base_dir).unwrap();
        recorder
            .append_transaction(vec![
                (
                    TranscriptEvent::SessionTitle {
                        title: "first".into(),
                    },
                    None,
                ),
                (
                    TranscriptEvent::AssistantMessage {
                        content: "second".into(),
                    },
                    Some("branch-a".into()),
                ),
            ])
            .unwrap();
        let path = base_dir.join(format!("{}.jsonl", recorder.session_id()));
        let records = read_records(&path).unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].sequence, 1);
        assert_eq!(records[1].sequence, 2);

        let lines = fs::read_to_string(&path)
            .unwrap()
            .lines()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let mut corrupt_commit: Value = serde_json::from_str(lines.last().unwrap()).unwrap();
        corrupt_commit["payload_digest"] = Value::String("wrong".into());
        let mut corrupt_lines = lines.clone();
        *corrupt_lines.last_mut().unwrap() = serde_json::to_string(&corrupt_commit).unwrap();
        fs::write(&path, corrupt_lines.join("\n") + "\n").unwrap();
        assert!(read_records(&path).is_err());

        let mut lines = lines;
        lines.pop(); // Remove only the private transaction commit marker.
        fs::write(&path, lines.join("\n") + "\n").unwrap();
        assert!(read_records(&path).unwrap().is_empty());
        fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(
                format!("{}\n", serde_json::to_string(&legacy_record(3)).unwrap()).as_bytes(),
            )
            .unwrap();
        assert!(read_records(&path).is_err());
        fs::write(&path, lines.join("\n") + "\n").unwrap();
        assert!(TranscriptRecorder::open_existing(&base_dir, recorder.session_id()).is_err());
    }

    #[test]
    fn journal_reader_rejects_transaction_commit_with_mismatched_resulting_revision() {
        let base_dir = journal_test_dir("transaction-resulting-revision");
        let mut recorder = TranscriptRecorder::create(&base_dir).unwrap();
        recorder
            .append_transaction(vec![
                (
                    TranscriptEvent::SessionTitle {
                        title: "first".into(),
                    },
                    None,
                ),
                (
                    TranscriptEvent::AssistantMessage {
                        content: "second".into(),
                    },
                    None,
                ),
            ])
            .unwrap();
        let path = base_dir.join(format!("{}.jsonl", recorder.session_id()));
        let mut lines = fs::read_to_string(&path)
            .unwrap()
            .lines()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let mut commit: Value = serde_json::from_str(lines.last().unwrap()).unwrap();
        commit["resulting_revision"] = Value::from(1);
        *lines.last_mut().unwrap() = serde_json::to_string(&commit).unwrap();
        fs::write(&path, lines.join("\n") + "\n").unwrap();

        assert!(read_records(path).is_err());
    }

    #[test]
    fn transaction_io_failure_poison_does_not_advance_or_switch_scope() {
        for fail in [FailPoint::Write, FailPoint::Flush, FailPoint::Sync] {
            let calls = Arc::new(Mutex::new(Vec::new()));
            let mut recorder = recorder_with_sink(FailingSink {
                fail,
                calls: Arc::clone(&calls),
            });
            recorder.set_current_context_branch_id(Some("parent".into()));
            assert!(
                recorder
                    .append_transaction(vec![(
                        TranscriptEvent::AssistantMessage {
                            content: "atomic".into(),
                        },
                        Some("child".into()),
                    )])
                    .is_err()
            );
            assert_eq!(recorder.sequence, 0);
            assert_eq!(recorder.current_context_branch_id(), Some("parent"));
            assert_eq!(recorder.health, RecorderHealth::Poisoned);
            assert_eq!(
                *calls.lock().unwrap(),
                match fail {
                    FailPoint::Write => vec!["write"],
                    FailPoint::Flush => vec!["write", "flush"],
                    FailPoint::Sync => vec!["write", "flush", "sync"],
                }
            );
        }
    }

    #[test]
    fn records_model_and_permission_mode_changes_with_expected_shape() {
        let base_dir = std::env::temp_dir().join(format!(
            "letcode-transcript-provenance-test-{}",
            unix_timestamp_ms()
        ));
        let mut recorder = TranscriptRecorder::create(&base_dir).expect("create recorder");

        recorder
            .record_model_changed("gpt-5.5", "gpt-5.5-mini")
            .expect("record model change");
        recorder
            .record_permission_mode_changed("default", "safe")
            .expect("record permission change");

        let records = read_records(base_dir.join(format!("{}.jsonl", recorder.session_id())))
            .expect("read records");

        assert_eq!(records.len(), 2);

        let first = serde_json::to_value(&records[0]).expect("serialize");
        assert_eq!(first.get("kind"), Some(&json!("model_changed")));
        assert_eq!(first.get("previous_model"), Some(&json!("gpt-5.5")));
        assert_eq!(first.get("new_model"), Some(&json!("gpt-5.5-mini")));

        let second = serde_json::to_value(&records[1]).expect("serialize");
        assert_eq!(second.get("kind"), Some(&json!("permission_mode_changed")));
        assert_eq!(second.get("previous_mode"), Some(&json!("default")));
        assert_eq!(second.get("new_mode"), Some(&json!("safe")));
    }

    #[test]
    fn restore_latest_model_replays_session_start_and_model_changes() {
        let records = vec![
            TranscriptRecord {
                session_id: "s".into(),
                sequence: 1,
                timestamp_ms: 0,
                context_branch_id: None,
                event: TranscriptEvent::SessionStarted { model: "m1".into() },
            },
            TranscriptRecord {
                session_id: "s".into(),
                sequence: 2,
                timestamp_ms: 1,
                context_branch_id: None,
                event: TranscriptEvent::ModelChanged {
                    previous_model: "m1".into(),
                    new_model: "m2".into(),
                },
            },
            TranscriptRecord {
                session_id: "s".into(),
                sequence: 3,
                timestamp_ms: 2,
                context_branch_id: None,
                event: TranscriptEvent::ModelChanged {
                    previous_model: "m2".into(),
                    new_model: "m3".into(),
                },
            },
        ];

        assert_eq!(restore_latest_model(&records).as_deref(), Some("m3"));
    }

    #[test]
    fn restore_conversation_messages_ignores_provenance_events() {
        let records = vec![
            TranscriptRecord {
                session_id: "s".into(),
                sequence: 1,
                timestamp_ms: 0,
                context_branch_id: None,
                event: TranscriptEvent::UserMessage {
                    content: "hi".into(),
                },
            },
            TranscriptRecord {
                session_id: "s".into(),
                sequence: 2,
                timestamp_ms: 1,
                context_branch_id: None,
                event: TranscriptEvent::TurnStarted(TurnStartedEvent {
                    turn_id: 1,
                    intent: "engineering".into(),
                    directive: "none".into(),
                    validation_reminder: "focused".into(),
                }),
            },
            TranscriptRecord {
                session_id: "s".into(),
                sequence: 3,
                timestamp_ms: 2,
                context_branch_id: None,
                event: TranscriptEvent::SubagentLifecycle {
                    run_id: "sub-1".into(),
                    parent_session_id: "s".into(),
                    parent_run_id: "turn-1".into(),
                    agent_name: "explorer".into(),
                    status: "running".into(),
                    detail: None,
                },
            },
            TranscriptRecord {
                session_id: "s".into(),
                sequence: 4,
                timestamp_ms: 3,
                context_branch_id: None,
                event: TranscriptEvent::ModelChanged {
                    previous_model: "a".into(),
                    new_model: "b".into(),
                },
            },
            TranscriptRecord {
                session_id: "s".into(),
                sequence: 5,
                timestamp_ms: 4,
                context_branch_id: None,
                event: TranscriptEvent::AssistantMessage {
                    content: "hello".into(),
                },
            },
            TranscriptRecord {
                session_id: "s".into(),
                sequence: 6,
                timestamp_ms: 5,
                context_branch_id: None,
                event: TranscriptEvent::PermissionModeChanged {
                    previous_mode: "default".into(),
                    new_mode: "safe".into(),
                },
            },
            TranscriptRecord {
                session_id: "s".into(),
                sequence: 7,
                timestamp_ms: 6,
                context_branch_id: None,
                event: TranscriptEvent::AutoContinuationScheduled {
                    continuation_count: 1,
                    remaining_unfinished: 2,
                },
            },
            TranscriptRecord {
                session_id: "s".into(),
                sequence: 8,
                timestamp_ms: 7,
                context_branch_id: None,
                event: TranscriptEvent::ValidationAdvisory(ValidationAdvisory {
                    write_effects: 1,
                    validation_effects: 0,
                    failed_validation_effects: 0,
                    message: "validation reminder".into(),
                }),
            },
            TranscriptRecord {
                session_id: "s".into(),
                sequence: 9,
                timestamp_ms: 8,
                context_branch_id: None,
                event: TranscriptEvent::ToolExecutionSummary(ToolExecutionSummaryEvent {
                    turn_id: 1,
                    call_id: "call-1".into(),
                    name: "fs__write".into(),
                    status: "executed".into(),
                    rejection: None,
                    effect_kind: "write".into(),
                    primary_path: Some("src/main.rs".into()),
                    command: None,
                }),
            },
            TranscriptRecord {
                session_id: "s".into(),
                sequence: 10,
                timestamp_ms: 9,
                context_branch_id: None,
                event: TranscriptEvent::TurnFinalized(TurnFinalizedEvent {
                    turn_id: 1,
                    outcome: "completed".into(),
                    tool_call_count: 1,
                    continuation_count: 0,
                    write_effects: 1,
                    validation_effects: 0,
                    failed_validation_effects: 0,
                    validation_advisory_emitted: true,
                }),
            },
        ];

        let restored = restore_conversation_messages(&records);
        assert_eq!(restored.len(), 2);
        assert!(matches!(restored[0].role, ConversationRole::User));
        assert_eq!(restored[0].content, "hi");
        assert!(matches!(restored[1].role, ConversationRole::Assistant));
        assert_eq!(restored[1].content, "hello");
    }

    #[test]
    fn restore_session_history_uses_latest_compaction_view() {
        let records = vec![
            TranscriptRecord {
                session_id: "s".into(),
                sequence: 1,
                timestamp_ms: 0,
                context_branch_id: None,
                event: TranscriptEvent::UserMessage {
                    content: "old user".into(),
                },
            },
            TranscriptRecord {
                session_id: "s".into(),
                sequence: 2,
                timestamp_ms: 1,
                context_branch_id: None,
                event: TranscriptEvent::UserMessage {
                    content: "tail user".into(),
                },
            },
            TranscriptRecord {
                session_id: "s".into(),
                sequence: 3,
                timestamp_ms: 1,
                context_branch_id: None,
                event: TranscriptEvent::AssistantMessage {
                    content: "tail assistant".into(),
                },
            },
            TranscriptRecord {
                session_id: "s".into(),
                sequence: 4,
                timestamp_ms: 2,
                context_branch_id: None,
                event: TranscriptEvent::ContextCompaction(ContextCompactionEvent {
                    outcome: "succeeded".into(),
                    summary: "目标\n- 保留摘要".into(),
                    tail_start_index: 1,
                    original_history_items: 3,
                    retained_history_items: 3,
                    retired_source_spans: Vec::new(),
                    frame_identity_bindings: Vec::new(),
                    detail: None,
                }),
            },
            TranscriptRecord {
                session_id: "s".into(),
                sequence: 5,
                timestamp_ms: 3,
                context_branch_id: None,
                event: TranscriptEvent::AssistantMessage {
                    content: "new assistant".into(),
                },
            },
        ];

        let history = restore_session_history(&records);
        assert!(matches!(history[0], HistoryItem::ContextSummary { .. }));
        assert!(matches!(history[1], HistoryItem::UserMessage { .. }));
        assert!(matches!(history[2], HistoryItem::AssistantText { .. }));
        assert!(matches!(history[3], HistoryItem::AssistantText { .. }));

        let messages = restore_compacted_conversation_messages(&records);
        assert!(matches!(messages[0].role, ConversationRole::Summary));
        assert_eq!(messages[1].content, "tail user");
        assert_eq!(messages[2].content, "tail assistant");
        assert_eq!(messages[3].content, "new assistant");

        let compaction = serde_json::to_value(&records[3]).expect("serialize compaction");
        assert_eq!(compaction["original_history_items"], json!(3));
        assert_eq!(compaction["retained_history_items"], json!(3));
        assert!(
            compaction
                .get("original_history_items")
                .unwrap()
                .is_number()
        );
        assert!(
            compaction
                .get("retained_history_items")
                .unwrap()
                .is_number()
        );
        let compaction_text =
            serde_json::to_string(&records[3]).expect("serialize compaction text");
        assert!(!compaction_text.contains("old user"));
        assert!(!compaction_text.contains("tail assistant"));
    }

    #[test]
    fn restore_session_history_preserves_tool_calls_permission_decisions_and_cancelled_tools() {
        let records = vec![
            TranscriptRecord {
                session_id: "s".into(),
                sequence: 1,
                timestamp_ms: 0,
                context_branch_id: None,
                event: TranscriptEvent::ToolCallStarted {
                    call_id: "call-1".into(),
                    name: "shell__exec".into(),
                    args: json!({"command": "cargo test"}),
                },
            },
            TranscriptRecord {
                session_id: "s".into(),
                sequence: 2,
                timestamp_ms: 1,
                context_branch_id: None,
                event: TranscriptEvent::PermissionDecision {
                    call_id: Some("call-1".into()),
                    tool: "shell__exec".into(),
                    args: json!({"command": "cargo test"}),
                    allowed: false,
                    reason: Some("Denied by user from TUI permission prompt".into()),
                },
            },
            TranscriptRecord {
                session_id: "s".into(),
                sequence: 3,
                timestamp_ms: 2,
                context_branch_id: None,
                event: TranscriptEvent::ToolCallCancelled {
                    call_id: "call-1".into(),
                    name: "shell__exec".into(),
                },
            },
        ];

        let history = restore_session_history(&records);
        assert!(matches!(
            history.first(),
            Some(HistoryItem::AssistantToolCalls { calls, .. })
                if calls.len() == 1 && calls[0].call_id == "call-1"
        ));
        assert!(matches!(
            history.get(1),
            Some(HistoryItem::ToolOutput {
                call_id,
                output_json,
            }) if call_id == "call-1"
                && output_json == r#"{"status":"cancelled","summary":"user cancelled"}"#
        ));
    }

    #[test]
    fn records_validation_advisory_with_expected_shape() {
        let base_dir = std::env::temp_dir().join(format!(
            "letcode-transcript-validation-advisory-test-{}",
            unix_timestamp_ms()
        ));
        let mut recorder = TranscriptRecorder::create(&base_dir).expect("create recorder");

        recorder
            .record_validation_advisory(ValidationAdvisory {
                write_effects: 2,
                validation_effects: 0,
                failed_validation_effects: 1,
                message: "validation reminder".into(),
            })
            .expect("record validation advisory");

        let records = read_records(base_dir.join(format!("{}.jsonl", recorder.session_id())))
            .expect("read records");

        assert_eq!(records.len(), 1);
        let record = serde_json::to_value(&records[0]).expect("serialize");
        assert_eq!(record.get("kind"), Some(&json!("validation_advisory")));
        assert_eq!(record.get("write_effects"), Some(&json!(2)));
        assert_eq!(record.get("validation_effects"), Some(&json!(0)));
        assert_eq!(record.get("failed_validation_effects"), Some(&json!(1)));
        assert_eq!(record.get("message"), Some(&json!("validation reminder")));
    }

    #[test]
    fn records_turn_lifecycle_and_tool_summary_with_expected_shape() {
        let base_dir = std::env::temp_dir().join(format!(
            "letcode-transcript-turn-audit-test-{}",
            unix_timestamp_ms()
        ));
        let mut recorder = TranscriptRecorder::create(&base_dir).expect("create recorder");

        recorder
            .record_turn_started(TurnStartedEvent {
                turn_id: 7,
                intent: "engineering".into(),
                directive: "none".into(),
                validation_reminder: "targeted".into(),
            })
            .expect("record turn started");
        recorder
            .record_tool_execution_summary(ToolExecutionSummaryEvent {
                turn_id: 7,
                call_id: "call-1".into(),
                name: "shell__exec".into(),
                status: "executed".into(),
                rejection: None,
                effect_kind: "validation".into(),
                primary_path: Some("src/agent.rs".into()),
                command: Some("cargo test transcript".into()),
            })
            .expect("record tool summary");
        recorder
            .record_turn_finalized(TurnFinalizedEvent {
                turn_id: 7,
                outcome: "completed".into(),
                tool_call_count: 3,
                continuation_count: 1,
                write_effects: 1,
                validation_effects: 1,
                failed_validation_effects: 0,
                validation_advisory_emitted: false,
            })
            .expect("record turn finalized");

        let records = read_records(base_dir.join(format!("{}.jsonl", recorder.session_id())))
            .expect("read records");
        assert_eq!(records.len(), 3);

        let started = serde_json::to_value(&records[0]).expect("serialize");
        assert_eq!(started.get("kind"), Some(&json!("turn_started")));
        assert_eq!(started.get("turn_id"), Some(&json!(7)));
        assert_eq!(started.get("intent"), Some(&json!("engineering")));

        let summary = serde_json::to_value(&records[1]).expect("serialize");
        assert_eq!(summary.get("kind"), Some(&json!("tool_execution_summary")));
        assert_eq!(summary.get("call_id"), Some(&json!("call-1")));
        assert!(summary.get("output").is_none());

        let finalized = serde_json::to_value(&records[2]).expect("serialize");
        assert_eq!(finalized.get("kind"), Some(&json!("turn_finalized")));
        assert_eq!(finalized.get("turn_id"), Some(&json!(7)));
        assert_eq!(finalized.get("outcome"), Some(&json!("completed")));
        assert_eq!(
            finalized.get("validation_advisory_emitted"),
            Some(&json!(false))
        );
    }

    #[test]
    fn failed_compaction_is_recorded_as_error_without_rewriting_history() {
        let base_dir = std::env::temp_dir().join(format!(
            "letcode-transcript-compaction-failure-test-{}",
            unix_timestamp_ms()
        ));
        let mut recorder = TranscriptRecorder::create(&base_dir).expect("create recorder");

        recorder
            .record_context_compaction(ContextCompactionEvent {
                outcome: "failed".into(),
                summary: String::new(),
                tail_start_index: 0,
                original_history_items: 3,
                retained_history_items: 3,
                retired_source_spans: Vec::new(),
                frame_identity_bindings: Vec::new(),
                detail: Some("summary model returned empty output".into()),
            })
            .expect("record failed compaction");

        let records = read_records(base_dir.join(format!("{}.jsonl", recorder.session_id())))
            .expect("read records");
        assert_eq!(records.len(), 1);
        let value = serde_json::to_value(&records[0]).expect("serialize");
        assert_eq!(value.get("kind"), Some(&json!("error")));
        assert_eq!(
            value.get("message"),
            Some(&json!(
                "context compaction failed: summary model returned empty output"
            ))
        );
    }

    #[test]
    fn compaction_event_deserializes_without_retired_source_spans() {
        let event: ContextCompactionEvent = serde_json::from_value(json!({
            "outcome": "succeeded",
            "summary": "summary",
            "tail_start_index": 1,
            "original_history_items": 3,
            "retained_history_items": 2
        }))
        .expect("legacy compaction event deserializes");

        assert!(event.retired_source_spans.is_empty());
    }

    #[test]
    fn record_context_compaction_populates_retired_source_spans_when_missing() {
        let base_dir = std::env::temp_dir().join(format!(
            "letcode-transcript-compaction-span-test-{}",
            unix_timestamp_ms()
        ));
        let mut recorder = TranscriptRecorder::create(&base_dir).expect("create recorder");
        recorder
            .append(TranscriptEvent::UserMessage {
                content: UserMessageContent::from("old user"),
            })
            .expect("record user");
        recorder
            .append(TranscriptEvent::AssistantMessage {
                content: "tail note".into(),
            })
            .expect("record assistant");

        recorder
            .record_context_compaction(ContextCompactionEvent {
                outcome: "succeeded".into(),
                summary: "summary".into(),
                tail_start_index: 1,
                original_history_items: 2,
                retained_history_items: 2,
                retired_source_spans: Vec::new(),
                frame_identity_bindings: Vec::new(),
                detail: None,
            })
            .expect("record compaction");

        let records = read_records(base_dir.join(format!("{}.jsonl", recorder.session_id())))
            .expect("read records");
        let event = records
            .iter()
            .find_map(|record| match &record.event {
                TranscriptEvent::ContextCompaction(event) => Some(event),
                _ => None,
            })
            .expect("compaction event present");
        assert_eq!(event.retired_source_spans.len(), 1);
        assert_eq!(event.retired_source_spans[0].start_sequence, 1);
        assert_eq!(event.retired_source_spans[0].end_sequence, 1);
    }

    #[test]
    fn write_summary_still_restores_legacy_write_observed_state() {
        let records = vec![
            TranscriptRecord {
                session_id: "s".into(),
                sequence: 1,
                timestamp_ms: 0,
                context_branch_id: None,
                event: TranscriptEvent::ContextExperimentStarted {
                    branch_id: "branch-1".into(),
                    parent_branch_id: ROOT_CONTEXT_BRANCH_ID.into(),
                    base_sequence: 4,
                },
            },
            TranscriptRecord {
                session_id: "s".into(),
                sequence: 2,
                timestamp_ms: 1,
                context_branch_id: Some("branch-1".into()),
                event: TranscriptEvent::ToolExecutionSummary(ToolExecutionSummaryEvent {
                    turn_id: 1,
                    call_id: "call-write".into(),
                    name: "fs__write".into(),
                    status: "executed".into(),
                    rejection: None,
                    effect_kind: "write".into(),
                    primary_path: Some("src/lib.rs".into()),
                    command: None,
                }),
            },
        ];

        let state = reconstruct_context_scope_state(&records).expect("reconstruct state");
        assert!(
            state
                .active_experiment
                .as_ref()
                .is_some_and(|experiment| experiment.writes_observed)
        );
    }

    #[test]
    fn records_tool_cancellation_and_turn_interruption() {
        let base_dir = std::env::temp_dir().join(format!(
            "letcode-transcript-interrupt-test-{}",
            unix_timestamp_ms()
        ));
        let mut recorder = TranscriptRecorder::create(&base_dir).expect("create recorder");

        recorder
            .record_tool_call_cancelled("call-1", "shell__exec")
            .expect("record tool cancellation");
        recorder
            .record_turn_interrupted(Some(7))
            .expect("record turn interruption");

        let records = read_records(base_dir.join(format!("{}.jsonl", recorder.session_id())))
            .expect("read records");
        assert_eq!(records.len(), 2);

        let cancelled = serde_json::to_value(&records[0]).expect("serialize cancelled");
        assert_eq!(cancelled.get("kind"), Some(&json!("tool_call_cancelled")));
        assert_eq!(cancelled.get("call_id"), Some(&json!("call-1")));

        let interrupted = serde_json::to_value(&records[1]).expect("serialize interrupted");
        assert_eq!(interrupted.get("kind"), Some(&json!("turn_interrupted")));
        assert_eq!(interrupted.get("turn_id"), Some(&json!(7)));
    }

    #[test]
    fn restore_max_turn_id_includes_turn_interrupted_events() {
        let records = vec![TranscriptRecord {
            session_id: "s".into(),
            sequence: 1,
            timestamp_ms: 0,
            context_branch_id: None,
            event: TranscriptEvent::TurnInterrupted { turn_id: Some(9) },
        }];

        assert_eq!(restore_max_turn_id(&records), 9);
    }

    #[test]
    fn restore_session_history_closes_dangling_user_turn_on_interrupt() {
        let records = vec![
            TranscriptRecord {
                session_id: "s".into(),
                sequence: 1,
                timestamp_ms: 0,
                context_branch_id: None,
                event: TranscriptEvent::UserMessage {
                    content: "unfinished".into(),
                },
            },
            TranscriptRecord {
                session_id: "s".into(),
                sequence: 2,
                timestamp_ms: 1,
                context_branch_id: None,
                event: TranscriptEvent::TurnInterrupted { turn_id: Some(1) },
            },
        ];

        let history = restore_session_history(&records);
        assert!(matches!(
            history.as_slice(),
            [HistoryItem::UserMessage { content }, HistoryItem::AssistantText { text: assistant_text }]
                if content.text == "unfinished" && assistant_text.is_empty()
        ));

        let messages = restore_conversation_messages(&records);
        assert_eq!(messages.len(), 2);
        assert!(matches!(messages[0].role, ConversationRole::User));
        assert!(matches!(messages[1].role, ConversationRole::Assistant));
        assert!(messages[1].content.is_empty());
    }

    #[test]
    fn restore_session_history_closes_interrupted_turn_after_tool_output() {
        let records = vec![
            TranscriptRecord {
                session_id: "s".into(),
                sequence: 1,
                timestamp_ms: 0,
                context_branch_id: None,
                event: TranscriptEvent::UserMessage {
                    content: "run it".into(),
                },
            },
            TranscriptRecord {
                session_id: "s".into(),
                sequence: 2,
                timestamp_ms: 1,
                context_branch_id: None,
                event: TranscriptEvent::ToolCallStarted {
                    call_id: "call-1".into(),
                    name: "shell__exec".into(),
                    args: json!({"command": "sleep 10"}),
                },
            },
            TranscriptRecord {
                session_id: "s".into(),
                sequence: 3,
                timestamp_ms: 2,
                context_branch_id: None,
                event: TranscriptEvent::ToolCallFinished {
                    call_id: "call-1".into(),
                    name: "shell__exec".into(),
                    ok: true,
                    output: ToolResult::ok("shell__exec", json!({"stdout": "started"})),
                },
            },
            TranscriptRecord {
                session_id: "s".into(),
                sequence: 4,
                timestamp_ms: 3,
                context_branch_id: None,
                event: TranscriptEvent::TurnInterrupted { turn_id: Some(1) },
            },
        ];

        let messages = restore_conversation_messages(&records);
        assert_eq!(messages.len(), 2);
        assert!(matches!(messages[0].role, ConversationRole::User));
        assert!(matches!(messages[1].role, ConversationRole::Assistant));
        assert_eq!(messages[0].content, "run it");
        assert!(messages[1].content.is_empty());
    }

    #[test]
    fn evidence_records_round_trip_and_restore_from_transcript() {
        let base_dir = std::env::temp_dir().join(format!(
            "letcode-transcript-evidence-test-{}",
            unix_timestamp_ms()
        ));
        let mut recorder = TranscriptRecorder::create(&base_dir).expect("create recorder");
        let draft = EvidenceDraft {
            id: Some("ev-test".into()),
            evidence_kind: EvidenceKind::FileExcerpt,
            title: "read config".into(),
            summary: "config has active provider".into(),
            detail: Some("active_provider = openai".into()),
            source: EvidenceSource::File {
                path: "letcode.toml".into(),
                start_line: Some(1),
                end_line: Some(1),
            },
            tags: vec!["letcode.toml".into()],
        };

        let record = recorder.record_evidence(draft).expect("record evidence");
        assert_eq!(record.id, "ev-test");

        let records = read_records(base_dir.join(format!("{}.jsonl", recorder.session_id())))
            .expect("read records");
        let evidence = restore_session_evidence(&records).expect("restore evidence");
        assert_eq!(evidence.len(), 1);
        assert_eq!(evidence[0].id, "ev-test");
        assert_eq!(evidence[0].summary, "config has active provider");
        assert!(restore_conversation_messages(&records).is_empty());
    }

    #[test]
    fn child_transcript_records_parent_attribution_without_affecting_parent_restore() {
        let base_dir = std::env::temp_dir().join(format!(
            "letcode-transcript-child-test-{}",
            unix_timestamp_ms()
        ));

        let mut parent = TranscriptRecorder::create(&base_dir).expect("create parent recorder");
        parent
            .record_user_message("parent question")
            .expect("record parent user");
        parent
            .record_assistant_message("parent answer")
            .expect("record parent assistant");

        let child_dir = child_sessions_dir(&base_dir);
        let mut child = TranscriptRecorder::create(&child_dir).expect("create child recorder");
        child
            .record_session_started("gpt-test")
            .expect("record child start");
        child
            .record_subagent_lifecycle(
                "sub-1",
                parent.session_id(),
                "turn-1",
                "explorer",
                "running",
                Some("inspect src".into()),
            )
            .expect("record lifecycle");
        child
            .record_assistant_message("child summary")
            .expect("record child message");

        let parent_records = read_records(base_dir.join(format!("{}.jsonl", parent.session_id())))
            .expect("read parent records");
        let child_records = read_records(child_dir.join(format!("{}.jsonl", child.session_id())))
            .expect("read child records");

        let restored = restore_conversation_messages(&parent_records);
        assert_eq!(restored.len(), 2);
        assert_eq!(restored[0].content, "parent question");
        assert_eq!(restored[1].content, "parent answer");
        assert!(matches!(
            child_records[1].event,
            TranscriptEvent::SubagentLifecycle { .. }
        ));

        match &child_records[1].event {
            TranscriptEvent::SubagentLifecycle {
                parent_session_id,
                parent_run_id,
                agent_name,
                status,
                ..
            } => {
                assert_eq!(parent_session_id, parent.session_id());
                assert_eq!(parent_run_id, "turn-1");
                assert_eq!(agent_name, "explorer");
                assert_eq!(status, "running");
            }
            other => panic!("unexpected event: {other:?}"),
        }
    }

    #[test]
    fn child_session_helpers_only_list_existing_children_and_restore_child_records() {
        let base_dir = std::env::temp_dir().join(format!(
            "letcode-transcript-child-helper-test-{}",
            unix_timestamp_ms()
        ));

        let mut parent = TranscriptRecorder::create(&base_dir).expect("create parent recorder");
        let child_dir = child_sessions_dir(&base_dir);
        fs::create_dir_all(&child_dir).expect("create child dir");
        let parent_session_id = parent.session_id().to_string();

        parent
            .record_subagent_result(
                "run-1",
                &parent_session_id,
                "turn-1",
                "placeholder-existing",
                "explorer",
                "completed",
                "inspected src/tool.rs",
            )
            .expect("record first child result");
        parent
            .record_subagent_result(
                "run-2",
                &parent_session_id,
                "turn-2",
                "missing-child",
                "explorer",
                "completed",
                "should be ignored",
            )
            .expect("record second child result");

        let mut child = TranscriptRecorder::create(&child_dir).expect("create child recorder");
        let child_session_id = child.session_id().to_string();
        child
            .record_user_message("inspect state")
            .expect("record child user message");
        child
            .record_assistant_message("done")
            .expect("record child assistant message");

        let mut parent_records =
            read_records(base_dir.join(format!("{}.jsonl", parent.session_id())))
                .expect("read parent records");
        match &mut parent_records[0].event {
            TranscriptEvent::SubagentResult {
                child_session_id: recorded_id,
                ..
            } => *recorded_id = child_session_id.clone(),
            other => panic!("unexpected event: {other:?}"),
        }

        let children = list_child_sessions_for_parent(&base_dir, &parent_records);
        assert_eq!(children.len(), 1);
        assert_eq!(children[0].child_session_id, child_session_id);
        assert_eq!(children[0].agent_name, "explorer");
        assert_eq!(children[0].status, "completed");
        assert_eq!(children[0].summary, "inspected src/tool.rs");

        let child_records = read_child_session_records(&base_dir, &children[0].child_session_id)
            .expect("read child session records");
        assert_eq!(child_records.len(), 2);
        assert!(matches!(
            child_records[0].event,
            TranscriptEvent::UserMessage { .. }
        ));
        assert!(matches!(
            child_records[1].event,
            TranscriptEvent::AssistantMessage { .. }
        ));
    }

    #[test]
    fn child_session_listing_uses_parent_results_not_lifecycle_records() {
        let base_dir = std::env::temp_dir().join(format!(
            "letcode-transcript-child-listing-test-{}",
            unix_timestamp_ms()
        ));

        let mut parent = TranscriptRecorder::create(&base_dir).expect("create parent recorder");
        let child_dir = child_sessions_dir(&base_dir);
        let child = TranscriptRecorder::create(&child_dir).expect("create child recorder");
        let parent_session_id = parent.session_id().to_string();
        let child_session_id = child.session_id().to_string();

        parent
            .record_subagent_lifecycle(
                "run-1",
                &parent_session_id,
                "turn-1",
                "explorer",
                "running",
                Some("inspect src".into()),
            )
            .expect("record lifecycle");

        let records = read_records(base_dir.join(format!("{}.jsonl", parent.session_id())))
            .expect("read parent records");
        assert!(list_child_sessions_for_parent(&base_dir, &records).is_empty());

        parent
            .record_subagent_result(
                "run-1",
                &parent_session_id,
                "turn-1",
                &child_session_id,
                "explorer",
                "completed",
                "inspection done",
            )
            .expect("record result");

        let records = read_records(base_dir.join(format!("{}.jsonl", parent.session_id())))
            .expect("read updated parent records");
        let children = list_child_sessions_for_parent(&base_dir, &records);
        assert_eq!(children.len(), 1);
        assert_eq!(children[0].child_session_id, child_session_id);
        assert_eq!(children[0].status, "completed");
    }

    #[test]
    fn duplicate_child_results_are_listed_once_with_latest_summary() {
        let base_dir = std::env::temp_dir().join(format!(
            "letcode-transcript-child-dedupe-test-{}",
            unix_timestamp_ms()
        ));

        let mut parent = TranscriptRecorder::create(&base_dir).expect("create parent recorder");
        let child_dir = child_sessions_dir(&base_dir);
        let child = TranscriptRecorder::create(&child_dir).expect("create child recorder");
        let parent_session_id = parent.session_id().to_string();
        let child_session_id = child.session_id().to_string();

        parent
            .record_subagent_result(
                "run-1",
                &parent_session_id,
                "turn-1",
                &child_session_id,
                "explorer",
                "running",
                "first summary",
            )
            .expect("record first result");
        parent
            .record_subagent_result(
                "run-1",
                &parent_session_id,
                "turn-1",
                &child_session_id,
                "explorer",
                "completed",
                "latest summary",
            )
            .expect("record second result");

        let records = read_records(base_dir.join(format!("{}.jsonl", parent.session_id())))
            .expect("read parent records");
        let children = list_child_sessions_for_parent(&base_dir, &records);

        assert_eq!(children.len(), 1);
        assert_eq!(children[0].child_session_id, child_session_id);
        assert_eq!(children[0].status, "completed");
        assert_eq!(children[0].summary, "latest summary");
    }

    #[test]
    fn subagent_result_round_trips_structured_payload() {
        let base_dir = std::env::temp_dir().join(format!(
            "letcode-transcript-structured-subagent-test-{}",
            unix_timestamp_ms()
        ));
        let mut recorder = TranscriptRecorder::create(&base_dir).expect("create recorder");

        recorder
            .record_subagent_result_structured(
                "run-1",
                "parent-session",
                "turn-1",
                "child-session",
                "explorer",
                "completed",
                "inspection done",
                Some(StructuredSubagentResult {
                    status: "completed".into(),
                    summary: "inspection done".into(),
                    malformed: false,
                    findings: vec!["found contract".into()],
                    files_read: vec!["src/subagent.rs".into()],
                    files_changed: vec![],
                    commands_run: vec!["cargo test subagent::tests".into()],
                    validation: vec!["passed".into()],
                    blockers: vec![],
                    next_steps: vec!["report".into()],
                    run_id: "run-1".into(),
                    child_session_id: "child-session".into(),
                    raw_excerpt: None,
                }),
            )
            .expect("record structured result");

        let records = read_records(base_dir.join(format!("{}.jsonl", recorder.session_id())))
            .expect("read records");
        match &records[1].event {
            TranscriptEvent::Evidence {
                source:
                    EvidenceSource::Subagent {
                        run_id,
                        child_session_id,
                        parent_tool,
                        parent_turn_id,
                        parent_session_id,
                        ..
                    },
                summary,
                detail,
                ..
            } => {
                assert_eq!(run_id, "run-1");
                assert_eq!(child_session_id, "child-session");
                assert_eq!(parent_tool, tool_names::TOOL_AGENT_EXPLORE);
                assert_eq!(parent_turn_id.as_deref(), Some("turn-1"));
                assert_eq!(parent_session_id.as_deref(), Some("parent-session"));
                assert_eq!(summary, "inspection done");
                let detail = detail.as_deref().expect("structured detail");
                assert!(detail.contains("found contract"));
                assert!(detail.contains("src/subagent.rs"));
            }
            other => panic!("unexpected event: {other:?}"),
        }
    }

    #[test]
    fn restore_job_board_derives_unreconciled_reconciled_and_reusable_states() {
        let base_dir = std::env::temp_dir().join(format!(
            "letcode-transcript-job-board-test-{}",
            unix_timestamp_ms()
        ));
        let mut recorder = TranscriptRecorder::create(&base_dir).expect("create recorder");

        recorder
            .record_subagent_result_structured(
                "run-1",
                "parent-session",
                "turn-1",
                "child-1",
                "explorer",
                "completed",
                "done",
                Some(StructuredSubagentResult {
                    status: "completed".into(),
                    summary: "done".into(),
                    malformed: false,
                    findings: vec![],
                    files_read: vec!["src/lib.rs".into()],
                    files_changed: vec![],
                    commands_run: vec![],
                    validation: vec![],
                    blockers: vec![],
                    next_steps: vec![],
                    run_id: "run-1".into(),
                    child_session_id: "child-1".into(),
                    raw_excerpt: None,
                }),
            )
            .expect("record result");
        let records = read_records(base_dir.join(format!("{}.jsonl", recorder.session_id())))
            .expect("read records");
        let job_board = restore_job_board(&base_dir, &records).expect("derive unreconciled board");
        assert_eq!(job_board.len(), 1);
        assert!(job_board[0].unreconciled);
        assert!(!job_board[0].reconciled);
        assert!(!job_board[0].reusable_eligible);

        let mut recorder = TranscriptRecorder::open_existing(&base_dir, recorder.session_id())
            .expect("reopen recorder");
        recorder
            .record_subagent_reconciliation(
                "run-1",
                "child-1",
                "explorer",
                "turn-2",
                "reconciled child run run-1",
            )
            .expect("record reconciliation");
        let records = read_records(base_dir.join(format!("{}.jsonl", recorder.session_id())))
            .expect("read updated records");
        let job_board = restore_job_board(&base_dir, &records).expect("derive reconciled board");
        assert_eq!(job_board.len(), 1);
        assert!(!job_board[0].unreconciled);
        assert!(job_board[0].reconciled);
        assert!(job_board[0].reusable_eligible);
    }

    #[test]
    fn context_view_remove_is_append_only_metadata_not_raw_purge() {
        let base_dir = std::env::temp_dir().join(format!(
            "letcode-transcript-context-view-append-only-test-{}",
            unix_timestamp_ms()
        ));
        let mut recorder = TranscriptRecorder::create(&base_dir).expect("create recorder");

        recorder
            .record_assistant_message("soft note that may be hidden from derived view")
            .expect("record assistant note");
        recorder
            .record_context_view_operation_metadata(
                "remove_from_view",
                Some("block-seq-1-note".into()),
                None,
                Some("hide from prompt-derived context view only".into()),
            )
            .expect("record remove-from-view metadata");

        let transcript_path = base_dir.join(format!("{}.jsonl", recorder.session_id()));
        let records = read_records(&transcript_path).expect("read records");

        assert_eq!(records.len(), 2);
        assert!(matches!(
            &records[0].event,
            TranscriptEvent::AssistantMessage { content }
                if content == "soft note that may be hidden from derived view"
        ));
        assert!(matches!(
            &records[1].event,
            TranscriptEvent::ContextViewOperationMetadata {
                operation,
                block_id,
                node_id,
                ..
            } if operation == "remove_from_view"
                && block_id.as_deref() == Some("block-seq-1-note")
                && node_id.is_none()
        ));
        assert_eq!(
            records
                .iter()
                .map(|record| record.sequence)
                .collect::<Vec<_>>(),
            vec![1, 2]
        );

        let mut reopened = TranscriptRecorder::open_existing(&base_dir, recorder.session_id())
            .expect("reopen recorder");
        reopened
            .record_user_message("new message after reopen")
            .expect("append after reopen");

        let reopened_records = read_records(&transcript_path).expect("read reopened records");
        assert_eq!(reopened_records.len(), 3);
        assert!(matches!(
            &reopened_records[0].event,
            TranscriptEvent::AssistantMessage { content }
                if content == "soft note that may be hidden from derived view"
        ));
        assert!(matches!(
            &reopened_records[1].event,
            TranscriptEvent::ContextViewOperationMetadata { operation, .. }
                if operation == "remove_from_view"
        ));
        assert!(matches!(
            &reopened_records[2].event,
            TranscriptEvent::UserMessage { content }
                if content.display_text() == "new message after reopen"
        ));
        let sequences = reopened_records
            .iter()
            .map(|record| record.sequence)
            .collect::<Vec<_>>();
        assert_eq!(sequences, vec![1, 2, 3]);
        assert_eq!(
            reopened_records.last().map(|record| record.sequence),
            Some(
                records
                    .iter()
                    .map(|record| record.sequence)
                    .max()
                    .unwrap_or(0)
                    + 1
            )
        );
    }

    #[test]
    fn context_resolve_pending_metadata_is_recorded() {
        let base_dir = std::env::temp_dir().join(format!(
            "letcode-transcript-context-resolve-metadata-test-{}",
            unix_timestamp_ms()
        ));
        let mut recorder = TranscriptRecorder::create(&base_dir).expect("create recorder");
        recorder
            .record_error("context view projection unavailable")
            .expect("record error");

        let output = ToolResult::ok(
            tool_names::TOOL_CONTEXT_RESOLVE,
            json!({
                "ok": true,
                "operation_metadata": {"operation": "resolve", "block_id": "block-seq-1-error"},
                "pending_recording": true
            }),
        );
        recorder
            .record_context_tool_pending_metadata(tool_names::TOOL_CONTEXT_RESOLVE, true, &output)
            .expect("record resolve metadata");

        let transcript_path = base_dir.join(format!("{}.jsonl", recorder.session_id()));
        let records = read_records(&transcript_path).expect("read records");
        assert_eq!(records.len(), 2);
        assert!(matches!(
            &records[1].event,
            TranscriptEvent::ContextViewOperationMetadata { operation, block_id, .. }
                if operation == "resolve" && block_id.as_deref() == Some("block-seq-1-error")
        ));

        let projection = transcript_projection::project_context_view(&records)
            .expect("project context view with resolve metadata");
        assert_eq!(
            projection.view_state.status(
                &crate::context_view::ContextBlockId::new("block-seq-1-error").expect("id")
            ),
            Some(crate::context_view::ContextViewStatus::Resolved)
        );
    }

    #[test]
    fn context_tool_pending_metadata_is_gated_by_tool_name_and_success() {
        let base_dir = std::env::temp_dir().join(format!(
            "letcode-transcript-context-tool-metadata-gating-test-{}",
            unix_timestamp_ms()
        ));
        let mut recorder = TranscriptRecorder::create(&base_dir).expect("create recorder");
        let pending_output = ToolResult::ok(
            tool_names::TOOL_CONTEXT_PIN,
            json!({
                "pending_recording": true,
                "operation_metadata": {"operation": "pin", "block_id": "block-seq-1-note"}
            }),
        );

        recorder
            .record_context_tool_pending_metadata(tool_names::TOOL_FS_READ, true, &pending_output)
            .expect("ignore non-context metadata");
        recorder
            .record_context_tool_pending_metadata(
                tool_names::TOOL_CONTEXT_PIN,
                false,
                &pending_output,
            )
            .expect("ignore failed context metadata");

        let transcript_path = base_dir.join(format!("{}.jsonl", recorder.session_id()));
        let records = read_records(&transcript_path).expect("read records");
        assert!(records.is_empty());
    }

    #[test]
    fn successful_context_open_block_records_open_detail_metadata() {
        let base_dir = std::env::temp_dir().join(format!(
            "letcode-transcript-context-open-metadata-test-{}",
            unix_timestamp_ms()
        ));
        let mut recorder = TranscriptRecorder::create(&base_dir).expect("create recorder");
        recorder
            .record_assistant_message("visible note")
            .expect("record visible note");

        let output = ToolResult::ok(
            tool_names::TOOL_CONTEXT_OPEN,
            json!({
                "ok": true,
                "ref_type": "block",
                "ref_id": "block-seq-1-note",
                "operation_metadata": {"operation": "open_detail", "block_id": "block-seq-1-note"},
                "pending_recording": true
            }),
        );
        recorder
            .record_context_tool_pending_metadata(tool_names::TOOL_CONTEXT_OPEN, true, &output)
            .expect("record open detail metadata");

        let transcript_path = base_dir.join(format!("{}.jsonl", recorder.session_id()));
        let records = read_records(&transcript_path).expect("read records");
        assert_eq!(records.len(), 2);
        assert!(matches!(
            &records[1].event,
            TranscriptEvent::ContextViewOperationMetadata { operation, block_id, .. }
                if operation == "open_detail" && block_id.as_deref() == Some("block-seq-1-note")
        ));

        let projection = transcript_projection::project_context_view(&records)
            .expect("project context view with open detail metadata");
        assert_eq!(
            projection
                .view_state
                .open_detail_block_id()
                .map(|block_id| block_id.as_str()),
            Some("block-seq-1-note")
        );
    }

    #[test]
    fn restore_job_board_derives_active_state_from_child_transcript() {
        let base_dir = std::env::temp_dir().join(format!(
            "letcode-transcript-active-job-board-test-{}",
            unix_timestamp_ms()
        ));
        let child_dir = child_sessions_dir(&base_dir);
        let mut child = TranscriptRecorder::create(&child_dir).expect("create child recorder");
        let child_session_id = child.session_id().to_string();
        child
            .record_subagent_lifecycle(
                "run-active",
                "parent-session",
                "turn-1",
                "fixer",
                "running",
                Some("apply patch".into()),
            )
            .expect("record running lifecycle");

        let mut child_file = OpenOptions::new()
            .append(true)
            .open(child.path())
            .expect("open child transcript for partial append");
        child_file
            .write_all(
                br#"{"session_id":"child","sequence":2,"timestamp_ms":1,"kind":"tool_call_finished""#,
            )
            .expect("append partial live record");

        let job_board = restore_job_board(&base_dir, &[]).expect("derive active board");
        assert_eq!(job_board.len(), 1);
        assert!(job_board[0].active);
        assert_eq!(job_board[0].child_session_id, child_session_id);
        assert_eq!(job_board[0].status, "running");
    }

    #[test]
    fn read_records_accepts_legacy_subagent_result_without_structured_payload() {
        let base_dir = std::env::temp_dir().join(format!(
            "letcode-transcript-legacy-subagent-test-{}",
            unix_timestamp_ms()
        ));
        fs::create_dir_all(&base_dir).expect("create temp dir");
        let path = base_dir.join("legacy.jsonl");
        fs::write(
            &path,
            r#"{"session_id":"s","sequence":1,"timestamp_ms":0,"kind":"subagent_result","run_id":"run-1","parent_session_id":"parent","parent_run_id":"turn-1","child_session_id":"child","agent_name":"explorer","status":"completed","summary":"done"}
"#,
        )
        .expect("write transcript");

        let records = read_records(&path).expect("read legacy transcript");
        match &records[0].event {
            TranscriptEvent::SubagentResult { .. } => {}
            other => panic!("unexpected event: {other:?}"),
        }
    }

    #[test]
    fn child_session_summaries_sort_by_timestamp_then_session_id() {
        let mut children = vec![
            ChildSessionSummary {
                parent_session_id: "parent".into(),
                parent_run_id: "turn".into(),
                child_session_id: "child-c".into(),
                agent_name: "explorer".into(),
                status: "completed".into(),
                summary: "third".into(),
                timestamp_ms: 2,
            },
            ChildSessionSummary {
                parent_session_id: "parent".into(),
                parent_run_id: "turn".into(),
                child_session_id: "child-b".into(),
                agent_name: "explorer".into(),
                status: "completed".into(),
                summary: "second".into(),
                timestamp_ms: 1,
            },
            ChildSessionSummary {
                parent_session_id: "parent".into(),
                parent_run_id: "turn".into(),
                child_session_id: "child-a".into(),
                agent_name: "explorer".into(),
                status: "completed".into(),
                summary: "first".into(),
                timestamp_ms: 1,
            },
        ];

        sort_child_session_summaries(&mut children);

        let ordered_ids = children
            .iter()
            .map(|child| child.child_session_id.as_str())
            .collect::<Vec<_>>();
        assert_eq!(ordered_ids, ["child-a", "child-b", "child-c"]);
    }

    #[test]
    fn rapidly_created_recorders_get_unique_session_ids_and_paths() {
        let base_dir = std::env::temp_dir().join(format!(
            "letcode-transcript-unique-id-test-{}",
            unix_timestamp_ms()
        ));

        let first = TranscriptRecorder::create(&base_dir).expect("create first recorder");
        let second = TranscriptRecorder::create(&base_dir).expect("create second recorder");

        assert_ne!(first.session_id(), second.session_id());
        assert_ne!(first.path(), second.path());
        assert!(first.path().exists());
        assert!(second.path().exists());
    }

    #[test]
    fn duplicate_evidence_ids_fail_restore() {
        let records = vec![
            TranscriptRecord {
                session_id: "s".into(),
                sequence: 1,
                timestamp_ms: 0,
                context_branch_id: None,
                event: TranscriptEvent::Evidence {
                    id: "ev-1".into(),
                    evidence_kind: EvidenceKind::Decision,
                    title: "one".into(),
                    summary: "one".into(),
                    detail: None,
                    source: EvidenceSource::Transcript { sequence: 1 },
                    tags: vec![],
                },
            },
            TranscriptRecord {
                session_id: "s".into(),
                sequence: 2,
                timestamp_ms: 1,
                context_branch_id: None,
                event: TranscriptEvent::Evidence {
                    id: "ev-1".into(),
                    evidence_kind: EvidenceKind::Decision,
                    title: "two".into(),
                    summary: "two".into(),
                    detail: None,
                    source: EvidenceSource::Transcript { sequence: 2 },
                    tags: vec![],
                },
            },
        ];

        assert!(restore_session_evidence(&records).is_err());
    }

    #[test]
    fn todo_and_auto_continue_events_round_trip_and_restore_latest_state() {
        let base_dir = std::env::temp_dir().join(format!(
            "letcode-transcript-todo-test-{}",
            unix_timestamp_ms()
        ));
        let mut recorder = TranscriptRecorder::create(&base_dir).expect("create recorder");

        recorder
            .record_todo_snapshot(vec![TodoItem {
                id: "t1".into(),
                content: "inspect".into(),
                status: crate::agent::TodoStatus::Pending,
            }])
            .expect("record first todo snapshot");
        recorder
            .record_auto_continue_changed(AutoContinueState {
                enabled: true,
                max_continuations: 2,
            })
            .expect("record auto-continue");
        recorder
            .record_auto_continuation_scheduled(1, 1)
            .expect("record auto-continuation scheduled");
        recorder
            .record_todo_snapshot(vec![
                TodoItem {
                    id: "t1".into(),
                    content: "inspect".into(),
                    status: crate::agent::TodoStatus::Completed,
                },
                TodoItem {
                    id: "t2".into(),
                    content: "validate".into(),
                    status: crate::agent::TodoStatus::InProgress,
                },
            ])
            .expect("record second todo snapshot");

        let records = read_records(base_dir.join(format!("{}.jsonl", recorder.session_id())))
            .expect("read records");

        let latest_todos = restore_latest_todo_snapshot(&records).expect("latest todos");
        assert_eq!(latest_todos.len(), 2);
        assert_eq!(latest_todos[0].status, crate::agent::TodoStatus::Completed);
        assert_eq!(latest_todos[1].status, crate::agent::TodoStatus::InProgress);

        let auto_continue =
            restore_latest_auto_continue_state(&records).expect("latest auto-continue");
        assert!(auto_continue.enabled);
        assert_eq!(auto_continue.max_continuations, 2);
        assert!(restore_conversation_messages(&records).is_empty());
        assert!(matches!(
            records[2].event,
            TranscriptEvent::AutoContinuationScheduled {
                continuation_count: 1,
                remaining_unfinished: 1,
            }
        ));
    }

    #[test]
    fn restore_latest_workflow_state_resets_on_new_turn_and_error() {
        let stale_todo = TodoItem {
            id: "stale".into(),
            content: "stale task".into(),
            status: crate::agent::TodoStatus::InProgress,
        };
        let records = vec![
            TranscriptRecord {
                session_id: "s".into(),
                sequence: 1,
                timestamp_ms: 0,
                context_branch_id: None,
                event: TranscriptEvent::TodoSnapshot {
                    items: vec![stale_todo.clone()],
                },
            },
            TranscriptRecord {
                session_id: "s".into(),
                sequence: 2,
                timestamp_ms: 1,
                context_branch_id: None,
                event: TranscriptEvent::AutoContinueChanged {
                    state: AutoContinueState {
                        enabled: true,
                        max_continuations: 2,
                    },
                },
            },
            TranscriptRecord {
                session_id: "s".into(),
                sequence: 3,
                timestamp_ms: 2,
                context_branch_id: None,
                event: TranscriptEvent::Error {
                    message: "tool event failed".into(),
                },
            },
        ];

        assert!(restore_latest_todo_snapshot(&records).is_none());
        assert!(restore_latest_auto_continue_state(&records).is_none());

        let mut records = records;
        records.push(TranscriptRecord {
            session_id: "s".into(),
            sequence: 4,
            timestamp_ms: 3,
            context_branch_id: None,
            event: TranscriptEvent::TodoSnapshot {
                items: vec![stale_todo],
            },
        });
        records.push(TranscriptRecord {
            session_id: "s".into(),
            sequence: 5,
            timestamp_ms: 4,
            context_branch_id: None,
            event: TranscriptEvent::UserMessage {
                content: crate::user_content::UserMessageContent::new("next", Vec::new()),
            },
        });

        assert!(restore_latest_todo_snapshot(&records).is_none());
        assert!(restore_latest_auto_continue_state(&records).is_none());
    }

    #[test]
    fn session_started_only_is_not_session_content() {
        let mut records = vec![TranscriptRecord {
            session_id: "s".into(),
            sequence: 1,
            timestamp_ms: 0,
            context_branch_id: None,
            event: TranscriptEvent::SessionStarted {
                model: "gpt-test".into(),
            },
        }];
        assert!(!has_session_content(&records));

        records.push(TranscriptRecord {
            session_id: "s".into(),
            sequence: 2,
            timestamp_ms: 1,
            context_branch_id: None,
            event: TranscriptEvent::UserMessage {
                content: "hello".into(),
            },
        });
        assert!(has_session_content(&records));
    }

    #[test]
    fn session_title_does_not_make_session_non_empty() {
        let records = vec![TranscriptRecord {
            session_id: "s".into(),
            sequence: 1,
            timestamp_ms: 0,
            context_branch_id: None,
            event: TranscriptEvent::SessionTitle {
                title: "hello".into(),
            },
        }];

        assert!(!has_session_content(&records));
    }

    #[test]
    fn unknown_transcript_events_are_read_and_ignored_for_restore() {
        let base_dir = std::env::temp_dir().join(format!(
            "letcode-transcript-unknown-event-test-{}",
            unix_timestamp_ms()
        ));
        fs::create_dir_all(&base_dir).expect("create temp dir");
        let path = base_dir.join("unknown.jsonl");
        fs::write(
            &path,
            r#"{"session_id":"s","sequence":1,"timestamp_ms":0,"kind":"future_audit_event","extra":"ignored"}
{"session_id":"s","sequence":2,"timestamp_ms":1,"kind":"user_message","content":"hi"}
"#,
        )
        .expect("write transcript");

        let records = read_records(&path).expect("read unknown transcript event");
        assert_eq!(records.len(), 2);
        assert!(matches!(records[0].event, TranscriptEvent::Unknown));

        let restored = restore_conversation_messages(&records);
        assert_eq!(restored.len(), 1);
        assert_eq!(restored[0].content, "hi");
    }

    #[test]
    fn known_transcript_events_with_missing_required_fields_still_fail() {
        let base_dir = std::env::temp_dir().join(format!(
            "letcode-transcript-known-event-fail-test-{}",
            unix_timestamp_ms()
        ));
        fs::create_dir_all(&base_dir).expect("create temp dir");
        let path = base_dir.join("malformed-known.jsonl");
        fs::write(
            &path,
            r#"{"session_id":"s","sequence":1,"timestamp_ms":0,"kind":"user_message"}
"#,
        )
        .expect("write transcript");

        let error = read_records(&path).expect_err("known malformed event should fail");
        assert!(error.to_string().contains("failed to parse line 1"));
    }

    #[test]
    fn strict_read_records_fails_on_partial_tail_but_live_read_ignores_it() {
        let base_dir = std::env::temp_dir().join(format!(
            "letcode-transcript-partial-tail-test-{}",
            unix_timestamp_ms()
        ));
        fs::create_dir_all(&base_dir).expect("create temp dir");
        let path = base_dir.join("partial.jsonl");
        fs::write(
            &path,
            concat!(
                r#"{"session_id":"s","sequence":1,"timestamp_ms":0,"kind":"user_message","content":"hi"}"#,
                "\n",
                r#"{"session_id":"s","sequence":2,"timestamp_ms":1,"kind":"tool_call_finished""#
            ),
        )
        .expect("write partial transcript");

        let strict_error = read_records(&path).expect_err("strict read should reject partial tail");
        assert!(strict_error.to_string().contains("failed to parse line 2"));

        let records =
            read_records_allow_partial_tail(&path).expect("live read ignores partial tail");
        assert_eq!(records.len(), 1);
        assert!(matches!(
            records[0].event,
            TranscriptEvent::UserMessage { .. }
        ));
    }

    #[test]
    fn live_read_records_keeps_complete_tail_strict() {
        let base_dir = std::env::temp_dir().join(format!(
            "letcode-transcript-complete-malformed-tail-test-{}",
            unix_timestamp_ms()
        ));
        fs::create_dir_all(&base_dir).expect("create temp dir");
        let path = base_dir.join("malformed-tail.jsonl");
        fs::write(
            &path,
            concat!(
                r#"{"session_id":"s","sequence":1,"timestamp_ms":0,"kind":"user_message","content":"hi"}"#,
                "\n",
                r#"{"session_id":"s","sequence":2,"timestamp_ms":1,"kind":"tool_call_finished""#,
                "\n"
            ),
        )
        .expect("write malformed complete transcript");

        let error = read_records_allow_partial_tail(&path)
            .expect_err("complete malformed tail should still fail");
        assert!(error.to_string().contains("failed to parse line 2"));
    }

    #[test]
    fn live_partial_tail_keeps_incomplete_batch_protected_until_final_output_arrives() {
        let base_dir = std::env::temp_dir().join(format!(
            "letcode-transcript-live-batch-tail-test-{}",
            unix_timestamp_ms()
        ));
        fs::create_dir_all(&base_dir).expect("create temp dir");
        let path = base_dir.join("live.jsonl");
        let calls = vec![
            HistoryToolCall {
                call_id: "call-1".into(),
                name: "fs__read".into(),
                arguments_json: r#"{"path":"one"}"#.into(),
            },
            HistoryToolCall {
                call_id: "call-2".into(),
                name: "fs__read".into(),
                arguments_json: r#"{"path":"two"}"#.into(),
            },
        ];
        let prefix = vec![
            TranscriptRecord {
                session_id: "live".into(),
                sequence: 1,
                timestamp_ms: 0,
                context_branch_id: None,
                event: TranscriptEvent::TurnStarted(TurnStartedEvent {
                    turn_id: 1,
                    intent: "inspect".into(),
                    directive: "read both files".into(),
                    validation_reminder: String::new(),
                }),
            },
            TranscriptRecord {
                session_id: "live".into(),
                sequence: 2,
                timestamp_ms: 1,
                context_branch_id: None,
                event: TranscriptEvent::UserMessage {
                    content: UserMessageContent::from("inspect both"),
                },
            },
            TranscriptRecord {
                session_id: "live".into(),
                sequence: 3,
                timestamp_ms: 2,
                context_branch_id: None,
                event: TranscriptEvent::AssistantToolCallBatch { text: None, calls },
            },
            TranscriptRecord {
                session_id: "live".into(),
                sequence: 4,
                timestamp_ms: 3,
                context_branch_id: None,
                event: TranscriptEvent::ToolCallFinished {
                    call_id: "call-1".into(),
                    name: "fs__read".into(),
                    ok: true,
                    output: ToolResult::ok("fs__read", json!({"contents":"one"})),
                },
            },
        ];
        let final_record = TranscriptRecord {
            session_id: "live".into(),
            sequence: 5,
            timestamp_ms: 4,
            context_branch_id: None,
            event: TranscriptEvent::ToolCallFinished {
                call_id: "call-2".into(),
                name: "fs__read".into(),
                ok: true,
                output: ToolResult::ok("fs__read", json!({"contents":"two"})),
            },
        };
        let final_line = serde_json::to_string(&final_record).expect("serialize final output");
        let partial_len = final_line.len() - 1;
        let mut content = prefix
            .iter()
            .map(|record| serde_json::to_string(record).expect("serialize prefix"))
            .collect::<Vec<_>>()
            .join("\n");
        content.push('\n');
        content.push_str(&final_line[..partial_len]);
        fs::write(&path, content).expect("write live partial transcript");

        let live_records =
            read_records_allow_partial_tail(&path).expect("read complete live prefix");
        assert_eq!(live_records.len(), 4);
        let live = transcript_projection::project_runtime_restore_snapshot(
            "live".into(),
            live_records.clone(),
            transcript_projection::SessionContextCursor {
                branch_id: None,
                leaf_sequence: None,
            },
            &[],
        )
        .expect("project incomplete live runtime");
        let live_history = history_items_from_frames(&live.protocol_frames);
        assert!(
            analyze_history_items(&live_history, None)
                .expect("analyze live group")
                .has_incomplete_tool_call_groups()
        );
        assert!(live.snapshot.compaction.protected_frame_ids.len() >= 3);
        let model = ModelRequestMetadata {
            supports_tools: true,
            ..Default::default()
        };
        for protocol in [ApiProtocol::Responses, ApiProtocol::Completions] {
            assert!(
                build_request(RequestBuilderInput {
                    protocol,
                    model_id: "gpt-test",
                    model: model.clone(),
                    prelude: &[],
                    snapshot: &live.snapshot,
                    tools: &[]
                })
                .is_err(),
                "{protocol:?} must reject the incomplete batch"
            );
        }
        assert_eq!(
            serde_json::to_value(&live_records).expect("serialize live records"),
            serde_json::to_value(&prefix).expect("serialize source prefix")
        );

        let mut file = OpenOptions::new()
            .append(true)
            .open(&path)
            .expect("open live transcript");
        file.write_all(final_line[partial_len..].as_bytes())
            .expect("complete final record");
        file.write_all(b"\n").expect("terminate final record");
        file.flush().expect("flush final record");
        let complete_records =
            read_records_allow_partial_tail(&path).expect("read completed live transcript");
        assert_eq!(complete_records.len(), 5);
        let complete = transcript_projection::project_runtime_restore_snapshot(
            "live".into(),
            complete_records,
            transcript_projection::SessionContextCursor {
                branch_id: None,
                leaf_sequence: None,
            },
            &[],
        )
        .expect("project complete live runtime");
        let complete_history = history_items_from_frames(&complete.protocol_frames);
        assert!(
            !analyze_history_items(&complete_history, None)
                .expect("analyze completed group")
                .has_incomplete_tool_call_groups()
        );
        for protocol in [ApiProtocol::Responses, ApiProtocol::Completions] {
            build_request(RequestBuilderInput {
                protocol,
                model_id: "gpt-test",
                model: model.clone(),
                prelude: &[],
                snapshot: &complete.snapshot,
                tools: &[],
            })
            .expect("complete batch builds for both protocols");
        }
    }

    #[test]
    fn audit_and_unknown_events_are_not_session_content() {
        let records = vec![
            TranscriptRecord {
                session_id: "s".into(),
                sequence: 1,
                timestamp_ms: 0,
                context_branch_id: None,
                event: TranscriptEvent::SessionStarted {
                    model: "gpt-test".into(),
                },
            },
            TranscriptRecord {
                session_id: "s".into(),
                sequence: 2,
                timestamp_ms: 1,
                context_branch_id: None,
                event: TranscriptEvent::TurnStarted(TurnStartedEvent {
                    turn_id: 1,
                    intent: "engineering".into(),
                    directive: "none".into(),
                    validation_reminder: "targeted".into(),
                }),
            },
            TranscriptRecord {
                session_id: "s".into(),
                sequence: 3,
                timestamp_ms: 2,
                context_branch_id: None,
                event: TranscriptEvent::ToolExecutionSummary(ToolExecutionSummaryEvent {
                    turn_id: 1,
                    call_id: "call-1".into(),
                    name: "fs__read".into(),
                    status: "executed".into(),
                    rejection: None,
                    effect_kind: "read".into(),
                    primary_path: Some("src/main.rs".into()),
                    command: None,
                }),
            },
            TranscriptRecord {
                session_id: "s".into(),
                sequence: 4,
                timestamp_ms: 3,
                context_branch_id: None,
                event: TranscriptEvent::TurnFinalized(TurnFinalizedEvent {
                    turn_id: 1,
                    outcome: "completed".into(),
                    tool_call_count: 1,
                    continuation_count: 0,
                    write_effects: 0,
                    validation_effects: 0,
                    failed_validation_effects: 0,
                    validation_advisory_emitted: false,
                }),
            },
            TranscriptRecord {
                session_id: "s".into(),
                sequence: 5,
                timestamp_ms: 4,
                context_branch_id: None,
                event: TranscriptEvent::Unknown,
            },
        ];

        assert!(!has_session_content(&records));
    }

    #[test]
    fn restore_max_turn_id_uses_all_turn_audit_events() {
        let records = vec![
            TranscriptRecord {
                session_id: "s".into(),
                sequence: 1,
                timestamp_ms: 0,
                context_branch_id: None,
                event: TranscriptEvent::TurnStarted(TurnStartedEvent {
                    turn_id: 3,
                    intent: "engineering".into(),
                    directive: "none".into(),
                    validation_reminder: "targeted".into(),
                }),
            },
            TranscriptRecord {
                session_id: "s".into(),
                sequence: 2,
                timestamp_ms: 1,
                context_branch_id: None,
                event: TranscriptEvent::ToolExecutionSummary(ToolExecutionSummaryEvent {
                    turn_id: 5,
                    call_id: "call-1".into(),
                    name: "shell__exec".into(),
                    status: "executed".into(),
                    rejection: None,
                    effect_kind: "validation".into(),
                    primary_path: None,
                    command: Some("cargo test".into()),
                }),
            },
            TranscriptRecord {
                session_id: "s".into(),
                sequence: 3,
                timestamp_ms: 2,
                context_branch_id: None,
                event: TranscriptEvent::TurnFinalized(TurnFinalizedEvent {
                    turn_id: 4,
                    outcome: "completed".into(),
                    tool_call_count: 1,
                    continuation_count: 0,
                    write_effects: 0,
                    validation_effects: 1,
                    failed_validation_effects: 0,
                    validation_advisory_emitted: false,
                }),
            },
        ];

        assert_eq!(restore_max_turn_id(&records), 5);
    }

    #[test]
    fn context_checkpoint_finishes_on_old_branch_then_switches_subsequent_records() {
        let base_dir = std::env::temp_dir().join(format!(
            "letcode-transcript-context-checkpoint-test-{}",
            unix_timestamp_ms()
        ));
        let mut recorder = TranscriptRecorder::create(&base_dir).expect("create recorder");
        recorder
            .record_session_started("gpt-test")
            .expect("session started");
        recorder
            .record_user_message("root prompt")
            .expect("root prompt");
        recorder
            .record_tool_call_started(
                "call-1",
                tool_names::TOOL_CONTEXT_CHECKPOINT,
                json!({"label": "Try parser fix", "reason": "Need risky exploration"}),
            )
            .expect("tool started");

        recorder
            .record_tool_call_finished_and_apply_context_control(
                "call-1",
                tool_names::TOOL_CONTEXT_CHECKPOINT,
                true,
                ToolResult::ok(
                    tool_names::TOOL_CONTEXT_CHECKPOINT,
                    json!({
                        "label": "Try parser fix",
                        "reason": "Need risky exploration",
                        "context_only": true,
                        "filesystem_rolled_back": false,
                        "message": "Created a context checkpoint request."
                    }),
                ),
            )
            .expect("tool finished with checkpoint");
        recorder
            .record_assistant_message("branch-only response")
            .expect("assistant on new branch");

        let records = read_records(base_dir.join(format!("{}.jsonl", recorder.session_id())))
            .expect("read records");
        assert!(matches!(
            &records[2],
            TranscriptRecord {
                context_branch_id: None,
                event: TranscriptEvent::ToolCallStarted { .. },
                ..
            }
        ));
        assert!(matches!(
            &records[3],
            TranscriptRecord {
                context_branch_id: None,
                event: TranscriptEvent::ToolCallFinished { .. },
                ..
            }
        ));
        assert!(matches!(
            &records[4].event,
            TranscriptEvent::ContextBranchCreated {
                branch_id,
                parent_branch_id,
                base_sequence,
                label,
            } if branch_id == "try-parser-fix"
                && parent_branch_id == ROOT_CONTEXT_BRANCH_ID
                && *base_sequence == 4
                && label.as_deref() == Some("Try parser fix")
        ));
        assert!(matches!(
            &records[5].event,
            TranscriptEvent::ContextCheckout { branch_id, leaf_sequence }
                if branch_id == "try-parser-fix" && *leaf_sequence == 4
        ));
        assert!(matches!(
            &records[6].event,
            TranscriptEvent::ContextExperimentStarted { branch_id, parent_branch_id, base_sequence }
                if branch_id == "try-parser-fix"
                    && parent_branch_id == ROOT_CONTEXT_BRANCH_ID
                    && *base_sequence == 4
        ));
        assert!(matches!(
            &records[7].event,
            TranscriptEvent::ContextNodeCreated {
                node_id,
                parent_node_id,
                label,
                purpose,
                source_ref,
                ..
            } if node_id == "branch/try-parser-fix"
                && parent_node_id.as_deref() == Some("root")
                && label.as_deref() == Some("Try parser fix")
                && purpose.as_deref() == Some("Need risky exploration")
                && source_ref.as_ref().is_some_and(|source| source.source_kind == "context_branch"
                    && source.source_id.as_deref() == Some("try-parser-fix"))
        ));
        assert!(matches!(
            &records[8].event,
            TranscriptEvent::ContextNodeLifecycle { node_id, status }
                if node_id == "root" && *status == ContextNodeStatus::Inactive
        ));
        assert!(matches!(
            &records[9].event,
            TranscriptEvent::ContextNodeLifecycle { node_id, status }
                if node_id == "branch/try-parser-fix" && *status == ContextNodeStatus::Active
        ));
        assert_eq!(
            records[10].context_branch_id.as_deref(),
            Some("try-parser-fix")
        );
        assert!(matches!(
            &records[10].event,
            TranscriptEvent::AssistantMessage { content } if content == "branch-only response"
        ));
        assert_eq!(recorder.current_context_branch_id(), Some("try-parser-fix"));
        assert!(matches!(
            recorder.active_context_experiment(),
            Some(ActiveContextExperiment { branch_id, parent_branch_id, base_sequence, writes_observed })
                if branch_id == "try-parser-fix"
                    && parent_branch_id == ROOT_CONTEXT_BRANCH_ID
                    && base_sequence == 4
                    && !writes_observed
        ));

        let tree =
            transcript_projection::project_context_tree(&records).expect("project context tree");
        assert_eq!(
            tree.active_node_id().map(|id| id.as_str()),
            Some("branch/try-parser-fix")
        );
    }

    #[test]
    fn non_checkpoint_tool_finished_does_not_switch_branch() {
        let base_dir = std::env::temp_dir().join(format!(
            "letcode-transcript-non-checkpoint-tool-test-{}",
            unix_timestamp_ms()
        ));
        let mut recorder = TranscriptRecorder::create(&base_dir).expect("create recorder");
        recorder
            .record_session_started("gpt-test")
            .expect("session started");
        recorder
            .record_tool_call_started("call-1", "fs__read", json!({"path": "src/main.rs"}))
            .expect("tool started");

        recorder
            .record_tool_call_finished_and_apply_context_control(
                "call-1",
                "fs__read",
                true,
                ToolResult::ok("fs__read", json!({"content": "ok"})),
            )
            .expect("tool finished");
        recorder
            .record_assistant_message("still on main")
            .expect("assistant message");

        let records = read_records(base_dir.join(format!("{}.jsonl", recorder.session_id())))
            .expect("read records");
        assert_eq!(records.len(), 4);
        assert!(matches!(
            records[2].event,
            TranscriptEvent::ToolCallFinished { .. }
        ));
        assert!(matches!(
            records[3].event,
            TranscriptEvent::AssistantMessage { .. }
        ));
        assert_eq!(records[3].context_branch_id, None);
        assert_eq!(recorder.current_context_branch_id(), None);
    }

    #[test]
    fn context_return_switches_back_to_parent_and_carries_summary_forward() {
        let base_dir = std::env::temp_dir().join(format!(
            "letcode-transcript-context-return-test-{}",
            unix_timestamp_ms()
        ));
        let mut recorder = TranscriptRecorder::create(&base_dir).expect("create recorder");
        recorder
            .record_session_started("gpt-test")
            .expect("session started");
        recorder
            .record_user_message("root prompt")
            .expect("root prompt");
        recorder
            .record_tool_call_started(
                "call-1",
                tool_names::TOOL_CONTEXT_CHECKPOINT,
                json!({"label": "Try parser fix", "reason": "Need risky exploration"}),
            )
            .expect("checkpoint started");
        recorder
            .record_tool_call_finished_and_apply_context_control(
                "call-1",
                tool_names::TOOL_CONTEXT_CHECKPOINT,
                true,
                ToolResult::ok(
                    tool_names::TOOL_CONTEXT_CHECKPOINT,
                    json!({
                        "label": "Try parser fix",
                        "reason": "Need risky exploration",
                        "context_only": true,
                        "filesystem_rolled_back": false,
                        "message": "Created a context checkpoint request."
                    }),
                ),
            )
            .expect("checkpoint finished");
        recorder
            .record_tool_execution_summary(ToolExecutionSummaryEvent {
                turn_id: 1,
                call_id: "call-write".into(),
                name: "fs__write".into(),
                status: "completed".into(),
                rejection: None,
                effect_kind: "write".into(),
                primary_path: Some("src/lib.rs".into()),
                command: None,
            })
            .expect("write summary");
        {
            let scope_state = recorder.context_scope_state();
            let mut state = scope_state.lock().expect("scope state lock");
            state
                .active_experiment
                .as_mut()
                .expect("active experiment")
                .writes_observed = true;
        }
        recorder
            .record_tool_call_started(
                "call-2",
                tool_names::TOOL_CONTEXT_RETURN,
                json!({"outcome": "useful", "summary": "Parser path found the root cause", "next_action": "apply the fix on main"}),
            )
            .expect("return started");
        recorder
            .record_tool_call_finished_and_apply_context_control(
                "call-2",
                tool_names::TOOL_CONTEXT_RETURN,
                true,
                ToolResult::ok(
                    tool_names::TOOL_CONTEXT_RETURN,
                    json!({
                        "outcome": "useful",
                        "summary": "Parser path found the root cause",
                        "next_action": "apply the fix on main",
                        "context_restored": true,
                        "filesystem_rolled_back": false,
                        "message": "Returned from the current context experiment to the parent context. Files were not reverted."
                    }),
                ),
            )
            .expect("return finished");

        let records = read_records(base_dir.join(format!("{}.jsonl", recorder.session_id())))
            .expect("read records");
        assert!(matches!(
            &records[11],
            TranscriptRecord {
                context_branch_id: Some(branch_id),
                event: TranscriptEvent::ToolCallStarted { name, .. },
                ..
            } if branch_id == "try-parser-fix" && name == tool_names::TOOL_CONTEXT_RETURN
        ));
        assert!(matches!(
            &records[12],
            TranscriptRecord {
                context_branch_id: Some(branch_id),
                event: TranscriptEvent::ToolCallFinished { output, .. },
                ..
            } if branch_id == "try-parser-fix"
                && output.data.as_ref().and_then(|data| data.get("warning")).and_then(serde_json::Value::as_str)
                    == Some("Context restored, files were NOT reverted")
        ));
        assert!(matches!(
            &records[13].event,
            TranscriptEvent::ContextCheckout { branch_id, leaf_sequence }
                if branch_id == ROOT_CONTEXT_BRANCH_ID && *leaf_sequence == 4
        ));
        assert!(matches!(
            &records[14],
            TranscriptRecord {
                context_branch_id: None,
                event: TranscriptEvent::ContextExperimentReturned {
                    branch_id,
                    parent_branch_id,
                    base_sequence,
                    outcome,
                    summary,
                    next_action,
                    had_writes,
                },
                ..
            } if branch_id == "try-parser-fix"
                && parent_branch_id == ROOT_CONTEXT_BRANCH_ID
                && *base_sequence == 4
                && outcome == "useful"
                && summary == "Parser path found the root cause"
                && next_action.as_deref() == Some("apply the fix on main")
                && *had_writes
        ));
        assert!(matches!(
            &records[15].event,
            TranscriptEvent::ContextNodeLifecycle { node_id, status }
                if node_id == "branch/try-parser-fix" && *status == ContextNodeStatus::Archived
        ));
        assert!(matches!(
            &records[16].event,
            TranscriptEvent::ContextNodeLifecycle { node_id, status }
                if node_id == "root" && *status == ContextNodeStatus::Active
        ));
        assert_eq!(recorder.current_context_branch_id(), None);
        assert!(recorder.active_context_experiment().is_none());

        let tree =
            transcript_projection::project_context_tree(&records).expect("project context tree");
        assert_eq!(tree.active_node_id().map(|id| id.as_str()), Some("root"));
        assert_eq!(
            tree.node(&ContextNodeId::new("branch/try-parser-fix").expect("node id"))
                .map(|node| &node.status),
            Some(&ContextNodeStatus::Archived)
        );

        let history = restore_session_history(&records);
        assert!(matches!(
            history.last(),
            Some(HistoryItem::ContextSummary { text })
                if text.contains("Parser path found the root cause")
                    && text.contains("files were NOT reverted")
        ));
    }

    #[test]
    fn list_sessions_skips_session_started_only_transcripts() {
        let base_dir = std::env::temp_dir().join(format!(
            "letcode-transcript-list-empty-test-{}",
            unix_timestamp_ms()
        ));

        let mut empty = TranscriptRecorder::create(&base_dir).expect("create empty recorder");
        empty
            .record_session_started("gpt-test")
            .expect("record empty session start");

        let mut content = TranscriptRecorder::create(&base_dir).expect("create content recorder");
        content
            .record_session_started("gpt-test")
            .expect("record content session start");
        content
            .record_user_message("keep me")
            .expect("record user message");

        let sessions = list_sessions(&base_dir).expect("list sessions");

        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].session_id, content.session_id());
    }

    #[test]
    fn list_sessions_prefers_latest_recorded_title() {
        let base_dir = std::env::temp_dir().join(format!(
            "letcode-transcript-list-title-test-{}",
            unix_timestamp_ms()
        ));

        let mut recorder = TranscriptRecorder::create(&base_dir).expect("create recorder");
        recorder
            .record_session_started("gpt-test")
            .expect("record session start");
        recorder
            .record_user_message("please help debug startup")
            .expect("record user message");
        recorder
            .record_session_title("Debug startup")
            .expect("record first title");
        recorder
            .record_session_title("Debug startup failure")
            .expect("record latest title");

        let sessions = list_sessions(&base_dir).expect("list sessions");

        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].title.as_deref(), Some("Debug startup failure"));
        assert_eq!(
            sessions[0].last_user_summary.as_deref(),
            Some("please help debug startup")
        );
    }

    #[test]
    fn list_sessions_reports_latest_model_after_model_changes() {
        let base_dir = std::env::temp_dir().join(format!(
            "letcode-transcript-list-model-test-{}",
            unix_timestamp_ms()
        ));

        let mut recorder = TranscriptRecorder::create(&base_dir).expect("create recorder");
        recorder
            .record_session_started("gpt-test")
            .expect("record session start");
        recorder
            .record_model_changed("gpt-test", "gpt-test-mini")
            .expect("record model change");
        recorder
            .record_user_message("keep me")
            .expect("record user message");

        let sessions = list_sessions(&base_dir).expect("list sessions");

        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].model.as_deref(), Some("gpt-test-mini"));
    }

    #[test]
    fn remove_empty_session_file_only_deletes_empty_transcripts() {
        let base_dir = std::env::temp_dir().join(format!(
            "letcode-transcript-remove-empty-test-{}",
            unix_timestamp_ms()
        ));

        let mut empty = TranscriptRecorder::create(&base_dir).expect("create empty recorder");
        empty
            .record_session_started("gpt-test")
            .expect("record empty session start");
        let empty_path = empty.path().to_path_buf();

        assert!(remove_empty_session_file(&empty_path).expect("remove empty session"));
        assert!(!empty_path.exists());

        let mut content = TranscriptRecorder::create(&base_dir).expect("create content recorder");
        content
            .record_session_started("gpt-test")
            .expect("record content session start");
        content
            .record_user_message("keep me")
            .expect("record user message");
        let content_path = content.path().to_path_buf();

        assert!(!remove_empty_session_file(&content_path).expect("keep content session"));
        assert!(content_path.exists());
    }
}
