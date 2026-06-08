use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::agent::{
    AutoContinueState, ConversationMessage, ConversationRole, TodoItem, ToolExecutionSummaryEvent,
    TurnFinalizedEvent, TurnStartedEvent, ValidationAdvisory,
};
use crate::evidence::{
    EvidenceDraft, EvidenceKind, EvidenceRecord, EvidenceSource, evidence_id_for_sequence,
    restore_evidence_records,
};
use crate::tool::ToolResult;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptRecord {
    pub session_id: String,
    pub sequence: u64,
    pub timestamp_ms: u128,
    #[serde(flatten)]
    pub event: TranscriptEvent,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TranscriptEvent {
    SessionStarted {
        model: String,
    },
    ModelChanged {
        previous_model: String,
        new_model: String,
    },
    UserMessage {
        content: String,
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
    TurnFinalized(TurnFinalizedEvent),
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

    pub fn record_user_message(&mut self, content: impl Into<String>) -> Result<()> {
        self.append(TranscriptEvent::UserMessage {
            content: content.into(),
        })
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

    pub fn record_turn_finalized(&mut self, event: TurnFinalizedEvent) -> Result<()> {
        self.append(TranscriptEvent::TurnFinalized(event))
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

    fn append_with_timestamp(&mut self, event: TranscriptEvent, timestamp_ms: u128) -> Result<()> {
        self.sequence += 1;

        let record = TranscriptRecord {
            session_id: self.session_id.clone(),
            sequence: self.sequence,
            timestamp_ms,
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
    pub last_user_summary: Option<String>,
    pub last_assistant_summary: Option<String>,
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
        let model = records.iter().find_map(|record| match &record.event {
            TranscriptEvent::SessionStarted { model } => Some(model.clone()),
            _ => None,
        });
        let last_user_summary = records.iter().rev().find_map(|record| match &record.event {
            TranscriptEvent::UserMessage { content } => Some(summarize_text(content)),
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
            last_user_summary,
            last_assistant_summary,
        });
    }

    sessions.sort_by_key(|session| session.last_timestamp_ms.unwrap_or(0));
    sessions.reverse();

    Ok(sessions)
}

pub fn restore_conversation_messages(records: &[TranscriptRecord]) -> Vec<ConversationMessage> {
    records
        .iter()
        .filter_map(|record| match &record.event {
            TranscriptEvent::UserMessage { content } => Some(ConversationMessage {
                role: ConversationRole::User,
                content: content.clone(),
            }),
            TranscriptEvent::AssistantMessage { content } => Some(ConversationMessage {
                role: ConversationRole::Assistant,
                content: content.clone(),
            }),
            _ => None,
        })
        .collect()
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
    records
        .iter()
        .filter_map(|record| match &record.event {
            TranscriptEvent::TurnStarted(event) => Some(event.turn_id),
            TranscriptEvent::ToolExecutionSummary(event) => Some(event.turn_id),
            TranscriptEvent::TurnFinalized(event) => Some(event.turn_id),
            _ => None,
        })
        .max()
        .unwrap_or(0)
}

pub fn has_session_content(records: &[TranscriptRecord]) -> bool {
    records
        .iter()
        .any(|record| record.event.is_session_content())
}

impl TranscriptEvent {
    fn is_session_content(&self) -> bool {
        matches!(
            self,
            Self::UserMessage { .. }
                | Self::AssistantMessage { .. }
                | Self::ReasoningMessage { .. }
                | Self::ToolCallStarted { .. }
                | Self::ToolCallFinished { .. }
                | Self::PermissionDecision { .. }
                | Self::TodoSnapshot { .. }
                | Self::AutoContinueChanged { .. }
                | Self::Error { .. }
                | Self::Evidence { .. }
        )
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

fn session_path(base_dir: &Path, session_id: &str) -> PathBuf {
    base_dir.join(format!("{session_id}.jsonl"))
}

fn generate_session_id() -> String {
    format!("{}-{}", unix_timestamp_ms(), process::id())
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
    fn restore_conversation_messages_ignores_provenance_events() {
        let records = vec![
            TranscriptRecord {
                session_id: "s".into(),
                sequence: 1,
                timestamp_ms: 0,
                event: TranscriptEvent::UserMessage {
                    content: "hi".into(),
                },
            },
            TranscriptRecord {
                session_id: "s".into(),
                sequence: 2,
                timestamp_ms: 1,
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
                event: TranscriptEvent::ModelChanged {
                    previous_model: "a".into(),
                    new_model: "b".into(),
                },
            },
            TranscriptRecord {
                session_id: "s".into(),
                sequence: 4,
                timestamp_ms: 3,
                event: TranscriptEvent::AssistantMessage {
                    content: "hello".into(),
                },
            },
            TranscriptRecord {
                session_id: "s".into(),
                sequence: 5,
                timestamp_ms: 4,
                event: TranscriptEvent::PermissionModeChanged {
                    previous_mode: "default".into(),
                    new_mode: "safe".into(),
                },
            },
            TranscriptRecord {
                session_id: "s".into(),
                sequence: 6,
                timestamp_ms: 5,
                event: TranscriptEvent::AutoContinuationScheduled {
                    continuation_count: 1,
                    remaining_unfinished: 2,
                },
            },
            TranscriptRecord {
                session_id: "s".into(),
                sequence: 7,
                timestamp_ms: 6,
                event: TranscriptEvent::ValidationAdvisory(ValidationAdvisory {
                    write_effects: 1,
                    validation_effects: 0,
                    failed_validation_effects: 0,
                    message: "validation reminder".into(),
                }),
            },
            TranscriptRecord {
                session_id: "s".into(),
                sequence: 8,
                timestamp_ms: 7,
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
                sequence: 9,
                timestamp_ms: 8,
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
    fn duplicate_evidence_ids_fail_restore() {
        let records = vec![
            TranscriptRecord {
                session_id: "s".into(),
                sequence: 1,
                timestamp_ms: 0,
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
            event: TranscriptEvent::SessionStarted {
                model: "gpt-test".into(),
            },
        }];
        assert!(!has_session_content(&records));

        records.push(TranscriptRecord {
            session_id: "s".into(),
            sequence: 2,
            timestamp_ms: 1,
            event: TranscriptEvent::UserMessage {
                content: "hello".into(),
            },
        });
        assert!(has_session_content(&records));
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
                event: TranscriptEvent::SessionStarted {
                    model: "gpt-test".into(),
                },
            },
            TranscriptRecord {
                session_id: "s".into(),
                sequence: 2,
                timestamp_ms: 1,
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
