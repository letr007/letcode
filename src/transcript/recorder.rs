use anyhow::{Context, Result, anyhow, ensure};
use serde::Serialize;
use serde_json::Value;
use std::fs::{self, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use super::*;
use crate::agent::{
    AutoContinueState, ContextCompactionEvent, LlmRequestTelemetry, LlmRequestTelemetryPhase,
    TodoItem, ToolExecutionSummaryEvent, TurnFinalizedEvent, TurnStartedEvent, ValidationAdvisory,
    subagent_evidence_parent_tool,
};
#[cfg(test)]
use crate::context_tree::{
    ContextBlockRef, ContextNodeId, ContextNodeStatus, ContextSourceRef, ContextTreeOp,
};
use crate::evidence::{
    EvidenceDraft, EvidenceKind, EvidenceRecord, EvidenceSource, evidence_id_for_sequence,
};
use crate::request_builder::HistoryToolCall;
use crate::subagent::StructuredSubagentResult;
use crate::tool::ToolResult;
use crate::user_content::UserMessageContent;

use super::journal::{
    FileJournalSink, JOURNAL_SCHEMA_VERSION, JOURNAL_TRANSACTION_COMMIT, JournalRecordV1,
    JournalSink, JournalTransactionCommitV1, journal_payload_digest, journal_scope_for,
    serialize_journal_record,
};

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

pub fn render_checkpoint_v1(event: &LogicalCheckpointEventV1) -> Result<String> {
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

pub fn render_checkpoint_continuation_v1(event: &LogicalCheckpointEventV1) -> String {
    format!(
        "Resume the same user turn from logical checkpoint {}. Treat the retained checkpoint context above as authoritative; retired sources are audit-only and are not directly openable.",
        event.checkpoint_id
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecorderHealth {
    Healthy,
    Poisoned,
}

pub const ROOT_CONTEXT_BRANCH_ID: &str = "main";

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
    pub(crate) session_id: String,
    #[allow(dead_code)]
    pub(crate) path: PathBuf,
    pub(crate) sink: Box<dyn JournalSink>,
    pub(crate) sequence: u64,
    pub(crate) health: RecorderHealth,
    pub(crate) current_context_branch_id: Option<String>,
    pub(crate) context_scope_state: Arc<Mutex<ContextScopeState>>,
    pub(crate) reasoning_started_at: std::collections::HashMap<String, std::time::Instant>,
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
            journal::transcript_file_fingerprint(&content) == *fingerprint,
            "transcript changed after records were loaded; retry resume"
        );
        ensure!(
            !journal::content_tail_is_uncommitted_transaction(&file_path, &content)?,
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
                tags: vec![agent_name, "subagent_result".into()],
            };
            self.record_evidence(evidence)?;
        }
        Ok(())
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

pub(crate) fn requires_durable_commit(event: &TranscriptEvent) -> bool {
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

pub fn sync_recorder_branch(recorder: &mut TranscriptRecorder, branch_id: &str) {
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

pub fn format_context_experiment_return(
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
