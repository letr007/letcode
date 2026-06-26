use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::agent::subagent_tool_name_for_agent_name;
use crate::agent::{
    AutoContinueState, ContextCompactionEvent, ConversationMessage, ConversationRole, TodoItem,
    ToolExecutionSummaryEvent, TurnFinalizedEvent, TurnStartedEvent, ValidationAdvisory,
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

pub(crate) const ROOT_CONTEXT_BRANCH_ID: &str = "main";

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
    file: File,
    sequence: u64,
    current_context_branch_id: Option<String>,
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
            file,
            sequence: 0,
            current_context_branch_id: None,
        })
    }

    pub fn open_existing(base_dir: impl AsRef<Path>, session_id: &str) -> Result<Self> {
        fs::create_dir_all(base_dir.as_ref())?;

        let file_path = session_path(base_dir.as_ref(), session_id);
        let records = read_records(&file_path)?;
        let sequence = records
            .iter()
            .map(|record| record.sequence)
            .max()
            .unwrap_or(0);
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&file_path)?;

        Ok(Self {
            session_id: session_id.to_string(),
            path: file_path,
            file,
            sequence,
            current_context_branch_id: None,
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

    pub fn record_tool_call_finished_and_apply_context_checkpoint(
        &mut self,
        call_id: impl Into<String>,
        name: impl Into<String>,
        ok: bool,
        output: ToolResult,
    ) -> Result<()> {
        let call_id = call_id.into();
        let name = name.into();
        self.record_tool_call_finished(call_id, name.clone(), ok, output.clone())?;
        if name == tool_names::TOOL_CONTEXT_CHECKPOINT && ok {
            self.apply_context_checkpoint_from_output(&output)?;
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
        self.append(TranscriptEvent::ContextCompaction(event))
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

    pub fn append_metadata(&mut self, event: TranscriptEvent) -> Result<()> {
        self.append_with_timestamp_and_branch(event, unix_timestamp_ms(), None)
    }

    fn apply_context_checkpoint_from_output(&mut self, output: &ToolResult) -> Result<()> {
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

        let snapshot = self.active_context_snapshot()?;
        let records = read_records(self.path())?;
        let branches = transcript_projection::list_context_branches(
            &records,
            self.current_context_branch_id(),
        )?;
        let branch_id = next_context_branch_id(&branches, label.as_deref());
        self.record_context_branch_created(
            branch_id.clone(),
            snapshot.branch_id.clone(),
            snapshot.leaf_sequence,
            label,
        )?;
        self.record_context_checkout(branch_id.clone(), snapshot.leaf_sequence)?;
        sync_recorder_branch(self, &branch_id);
        Ok(())
    }

    fn active_context_snapshot(&self) -> Result<transcript_projection::SessionRestoreSnapshot> {
        transcript_projection::build_session_context_snapshot(
            self.session_id().to_string(),
            read_records(self.path())?,
            None,
            transcript_projection::SessionContextCursor {
                branch_id: self.current_context_branch_id().map(str::to_string),
                leaf_sequence: None,
            },
        )
    }

    fn append_with_timestamp(&mut self, event: TranscriptEvent, timestamp_ms: u128) -> Result<()> {
        let context_branch_id = if event.is_context_branch_metadata() {
            None
        } else {
            self.current_context_branch_id.clone()
        };
        self.append_with_timestamp_and_branch(event, timestamp_ms, context_branch_id)
    }

    fn append_with_timestamp_and_branch(
        &mut self,
        event: TranscriptEvent,
        timestamp_ms: u128,
        context_branch_id: Option<String>,
    ) -> Result<()> {
        self.sequence += 1;

        let record = TranscriptRecord {
            session_id: self.session_id.clone(),
            sequence: self.sequence,
            timestamp_ms,
            context_branch_id,
            event,
        };

        serde_json::to_writer(&mut self.file, &record)?;
        self.file.write_all(b"\n")?;
        self.file.flush()?;

        Ok(())
    }
}

#[allow(dead_code)]
pub fn read_records(path: impl AsRef<Path>) -> Result<Vec<TranscriptRecord>> {
    let path = path.as_ref();
    let file = File::open(path)
        .with_context(|| format!("failed to open transcript {}", path.display()))?;
    let reader = BufReader::new(file);

    let mut records = Vec::new();
    for (index, line) in reader.lines().enumerate() {
        let line = line.with_context(|| {
            format!(
                "failed to read line {} from transcript {}",
                index + 1,
                path.display()
            )
        })?;

        if line.trim().is_empty() {
            continue;
        }

        records.push(serde_json::from_str(&line).with_context(|| {
            format!(
                "failed to parse line {} from transcript {}",
                index + 1,
                path.display()
            )
        })?);
    }

    Ok(records)
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

pub fn restore_job_board(
    base_dir: impl AsRef<Path>,
    parent_records: &[TranscriptRecord],
) -> Result<Vec<JobBoardEntry>> {
    transcript_projection::project_job_board(&child_sessions_dir(base_dir), parent_records)
}

pub fn restore_session_history(records: &[TranscriptRecord]) -> Vec<HistoryItem> {
    transcript_projection::restore_session_history_projection(records)
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
    records.iter().rev().find_map(|record| match &record.event {
        TranscriptEvent::TodoSnapshot { items } => Some(items.clone()),
        _ => None,
    })
}

pub fn restore_latest_auto_continue_state(
    records: &[TranscriptRecord],
) -> Option<AutoContinueState> {
    records.iter().rev().find_map(|record| match &record.event {
        TranscriptEvent::AutoContinueChanged { state } => Some(state.clone()),
        _ => None,
    })
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
        )
    }
}

pub(crate) fn sync_recorder_branch(recorder: &mut TranscriptRecorder, branch_id: &str) {
    if branch_id == ROOT_CONTEXT_BRANCH_ID {
        recorder.set_current_context_branch_id(None);
    } else {
        recorder.set_current_context_branch_id(Some(branch_id.to_string()));
    }
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

fn append_history_item_from_transcript_record(
    history: &mut Vec<HistoryItem>,
    record: &TranscriptRecord,
) {
    let item = match &record.event {
        TranscriptEvent::UserMessage { content } => {
            Some(HistoryItem::user_content(content.clone()))
        }
        TranscriptEvent::AssistantMessage { content } => {
            Some(HistoryItem::assistant(content.clone()))
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
        _ => None,
    };
    if let Some(item) = item {
        history.push(item);
    }
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
    use crate::subagent::StructuredSubagentResult;
    use serde_json::json;

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
                    summary: "目标\n- 保留摘要".into(),
                    tail_start_index: 1,
                    original_history_items: 3,
                    retained_history_items: 3,
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
        assert_eq!(history.len(), 1);
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
            .record_tool_call_finished_and_apply_context_checkpoint(
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
        assert_eq!(
            records[6].context_branch_id.as_deref(),
            Some("try-parser-fix")
        );
        assert!(matches!(
            &records[6].event,
            TranscriptEvent::AssistantMessage { content } if content == "branch-only response"
        ));
        assert_eq!(recorder.current_context_branch_id(), Some("try-parser-fix"));
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
            .record_tool_call_finished_and_apply_context_checkpoint(
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
