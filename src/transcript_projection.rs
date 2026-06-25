use crate::agent::AutoContinueState;
use crate::evidence::EvidenceRecord;
use crate::request_builder::HistoryItem;
use crate::tool_format::format_tool_call;
use crate::transcript::{ChildSessionSummary, JobBoardEntry, TranscriptEvent, TranscriptRecord};
use crate::tui::events::{
    AutoContinueChangedEvent, ErrorEvent, TodoSnapshotEvent, TokenUsageEvent, ToolFinishedEvent,
    ToolOutcome, ToolStartedEvent, UserMessageEvent,
};
use crate::tui::timeline::{
    COMPACTION_SEPARATOR_LABEL, MessageRole, PermissionPromptStatus, Timeline,
    restored_tool_summary,
};
use crate::user_content::UserMessageSubmission;
use crate::{agent::ConversationMessage, subagent::StructuredSubagentResult};
use std::collections::BTreeMap;
use std::path::Path;

/// Transcript restore intentionally differs from live projection in a few places:
/// - Permission decisions restore as terminal approved/denied permission items, not as pending prompts.
/// - Context compaction restores as separator + assistant summary + separator.
/// - Subagent lifecycle/result records are ignored in restored timelines.
/// - Turn audit and unknown transcript events are ignored during timeline restore.
pub(crate) fn timeline_from_transcript_records(records: &[TranscriptRecord]) -> Timeline {
    let mut projection = TranscriptTimelineProjection::default();
    for record in records {
        projection.apply_record(record);
    }
    projection.timeline
}

#[derive(Debug, Clone)]
pub(crate) struct SessionRestoreSnapshot {
    pub session_id: String,
    pub records: Vec<TranscriptRecord>,
    pub messages: Vec<ConversationMessage>,
    pub history: Vec<HistoryItem>,
    pub evidence: Vec<EvidenceRecord>,
    pub latest_model: Option<String>,
    pub max_turn_id: u64,
    pub token_usage: Option<TokenUsageEvent>,
}

impl SessionRestoreSnapshot {
    pub(crate) fn evidence_count(&self) -> usize {
        self.evidence.len()
    }
}

pub(crate) fn project_session_restore_snapshot(
    session_id: String,
    records: Vec<TranscriptRecord>,
    token_usage: Option<TokenUsageEvent>,
) -> anyhow::Result<SessionRestoreSnapshot> {
    let history = restore_session_history_projection(&records);
    let messages = history
        .clone()
        .into_iter()
        .filter_map(super::history_item_to_conversation_message)
        .collect();
    let evidence = crate::evidence::restore_evidence_records(&records)?;
    let latest_model = restore_latest_model_projection(&records);
    let max_turn_id = restore_max_turn_id_projection(&records);

    Ok(SessionRestoreSnapshot {
        session_id,
        records,
        messages,
        history,
        evidence,
        latest_model,
        max_turn_id,
        token_usage,
    })
}

pub(crate) fn restore_session_history_projection(records: &[TranscriptRecord]) -> Vec<HistoryItem> {
    let mut history = Vec::new();
    for record in records {
        match &record.event {
            TranscriptEvent::ContextCompaction(event) => {
                let tail_start = event.tail_start_index.min(history.len());
                let mut compacted =
                    Vec::with_capacity(1 + history.len().saturating_sub(tail_start));
                compacted.push(HistoryItem::context_summary(event.summary.clone()));
                compacted.extend(history.drain(tail_start..));
                history = compacted;
            }
            TranscriptEvent::TurnInterrupted { .. } => {
                close_interrupted_turn(&mut history);
            }
            TranscriptEvent::TurnFinalized(event) if event.outcome == "interrupted" => {
                close_interrupted_turn(&mut history);
            }
            _ => super::append_history_item_from_transcript_record(&mut history, record),
        }
    }
    history
}

fn close_interrupted_turn(history: &mut Vec<HistoryItem>) {
    let Some(last_conversation_item) = history.iter().rfind(|item| {
        matches!(
            item,
            HistoryItem::UserMessage { .. }
                | HistoryItem::InternalContinuation { .. }
                | HistoryItem::AssistantText { .. }
                | HistoryItem::ContextSummary { .. }
        )
    }) else {
        return;
    };

    if matches!(
        last_conversation_item,
        HistoryItem::UserMessage { .. } | HistoryItem::InternalContinuation { .. }
    ) {
        history.push(HistoryItem::assistant(String::new()));
    }
}

pub(crate) fn restore_latest_model_projection(records: &[TranscriptRecord]) -> Option<String> {
    let mut model = None;
    for record in records {
        match &record.event {
            TranscriptEvent::SessionStarted { model: started } => model = Some(started.clone()),
            TranscriptEvent::ModelChanged { new_model, .. } => model = Some(new_model.clone()),
            _ => {}
        }
    }
    model
}

pub(crate) fn restore_max_turn_id_projection(records: &[TranscriptRecord]) -> u64 {
    records
        .iter()
        .filter_map(|record| match &record.event {
            TranscriptEvent::TurnStarted(event) => Some(event.turn_id),
            TranscriptEvent::ToolExecutionSummary(event) => Some(event.turn_id),
            TranscriptEvent::TurnFinalized(event) => Some(event.turn_id),
            TranscriptEvent::TurnInterrupted { turn_id } => *turn_id,
            _ => None,
        })
        .max()
        .unwrap_or(0)
}

pub(crate) fn project_child_session_summaries(
    child_dir: &Path,
    parent_records: &[TranscriptRecord],
) -> Vec<ChildSessionSummary> {
    let mut children = BTreeMap::new();

    for record in parent_records {
        if let TranscriptEvent::SubagentResult {
            parent_session_id,
            parent_run_id,
            child_session_id,
            agent_name,
            status,
            summary,
            ..
        } = &record.event
            && child_dir.join(format!("{child_session_id}.jsonl")).exists()
        {
            children.insert(
                child_session_id.clone(),
                ChildSessionSummary {
                    parent_session_id: parent_session_id.clone(),
                    parent_run_id: parent_run_id.clone(),
                    child_session_id: child_session_id.clone(),
                    agent_name: agent_name.clone(),
                    status: status.clone(),
                    summary: summary.clone(),
                    timestamp_ms: record.timestamp_ms,
                },
            );
        }
    }

    let mut children = children.into_values().collect::<Vec<_>>();
    children.sort_by(|left, right| {
        left.timestamp_ms
            .cmp(&right.timestamp_ms)
            .then_with(|| left.child_session_id.cmp(&right.child_session_id))
    });
    children
}

pub(crate) fn project_job_board(
    child_dir: &Path,
    parent_records: &[TranscriptRecord],
) -> anyhow::Result<Vec<JobBoardEntry>> {
    let mut jobs = BTreeMap::<String, JobBoardAccumulator>::new();

    for record in parent_records {
        match &record.event {
            TranscriptEvent::SubagentResult {
                run_id,
                child_session_id,
                agent_name,
                status,
                summary,
                ..
            } => {
                let entry = jobs.entry(run_id.clone()).or_default();
                entry.run_id = run_id.clone();
                entry.child_session_id = child_session_id.clone();
                entry.agent_name = agent_name.clone();
                entry.status = status.clone();
                entry.summary = summary.clone();
                entry.terminal = true;
                entry.active = false;
            }
            TranscriptEvent::Evidence {
                source:
                    crate::evidence::EvidenceSource::Subagent {
                        run_id,
                        child_session_id,
                        parent_tool,
                        ..
                    },
                summary,
                detail,
                tags,
                ..
            } => {
                let entry = jobs.entry(run_id.clone()).or_default();
                entry.run_id = run_id.clone();
                if entry.child_session_id.is_empty() {
                    entry.child_session_id = child_session_id.clone();
                }
                if entry.agent_name.is_empty() {
                    entry.agent_name = parent_tool.trim_start_matches("agent__").to_string();
                }
                if tags.iter().any(|tag| tag == "subagent_result") {
                    entry.summary = summary.clone();
                    if let Some(detail) = detail
                        && let Ok(structured) =
                            serde_json::from_str::<StructuredSubagentResult>(detail)
                    {
                        entry.malformed = structured.malformed;
                        entry.structured_status = Some(structured.status.clone());
                        if entry.status.is_empty() {
                            entry.status = structured.status;
                        }
                    }
                }
                if tags
                    .iter()
                    .any(|tag| tag == "subagent_reconciliation" || tag == "reconciled")
                {
                    entry.reconciled = true;
                }
            }
            _ => {}
        }
    }

    if child_dir.exists() {
        for entry in std::fs::read_dir(child_dir)? {
            let path = entry?.path();
            if path.extension().and_then(|ext| ext.to_str()) != Some("jsonl") {
                continue;
            }
            let child_session_id = match path.file_stem().and_then(|stem| stem.to_str()) {
                Some(value) => value.to_string(),
                None => continue,
            };
            let child_records = crate::transcript::read_records(&path)?;
            let latest = child_records
                .iter()
                .rev()
                .find_map(|record| match &record.event {
                    TranscriptEvent::SubagentLifecycle {
                        run_id,
                        agent_name,
                        status,
                        detail,
                        ..
                    } => Some((
                        run_id.clone(),
                        agent_name.clone(),
                        status.clone(),
                        detail.clone(),
                    )),
                    _ => None,
                });
            let Some((run_id, agent_name, status, detail)) = latest else {
                continue;
            };
            if status != "running" {
                continue;
            }
            let job = jobs.entry(run_id.clone()).or_default();
            if job.terminal {
                continue;
            }
            job.run_id = run_id;
            job.child_session_id = child_session_id;
            job.agent_name = agent_name;
            job.status = status;
            job.summary = detail.unwrap_or_else(|| "subagent running".into());
            job.active = true;
        }
    }

    let mut entries = jobs
        .into_values()
        .filter(|entry| !entry.run_id.is_empty())
        .map(|entry| {
            let reconciled = entry.terminal && entry.reconciled;
            let unreconciled = entry.terminal && !entry.reconciled;
            let reusable_eligible = reconciled
                && entry.status == "completed"
                && entry.structured_status.as_deref() == Some("completed")
                && !entry.malformed;
            JobBoardEntry {
                active: entry.active,
                unreconciled,
                reconciled,
                reusable_eligible,
                run_id: entry.run_id,
                child_session_id: entry.child_session_id,
                agent_name: entry.agent_name,
                status: entry.status,
                summary: entry.summary,
            }
        })
        .collect::<Vec<_>>();
    entries.sort_by(|a, b| a.run_id.cmp(&b.run_id));
    Ok(entries)
}

#[derive(Debug, Clone, Default)]
struct JobBoardAccumulator {
    run_id: String,
    child_session_id: String,
    agent_name: String,
    status: String,
    summary: String,
    active: bool,
    terminal: bool,
    reconciled: bool,
    malformed: bool,
    structured_status: Option<String>,
}

#[derive(Debug, Default)]
struct TranscriptTimelineProjection {
    timeline: Timeline,
    current_auto_continue: AutoContinueState,
}

impl TranscriptTimelineProjection {
    fn apply_record(&mut self, record: &TranscriptRecord) {
        match &record.event {
            TranscriptEvent::UserMessage { content } => {
                self.timeline
                    .push_user_message(UserMessageEvent::from_submission(
                        UserMessageSubmission::new(
                            format!("restored-user-message-{}", record.sequence),
                            content.clone(),
                        ),
                    ))
            }
            TranscriptEvent::AssistantMessage { content } => self
                .timeline
                .push_restored_message(MessageRole::Assistant, content.clone()),
            TranscriptEvent::ContextCompaction(event) => {
                self.timeline
                    .push_compaction_separator(COMPACTION_SEPARATOR_LABEL);
                self.timeline
                    .push_restored_message(MessageRole::Assistant, event.summary.clone());
                self.timeline
                    .push_compaction_separator(COMPACTION_SEPARATOR_LABEL);
            }
            TranscriptEvent::ReasoningMessage { content } => {
                self.timeline.push_restored_reasoning(
                    format!("restored-reasoning-{}", record.sequence),
                    content.clone(),
                );
            }
            TranscriptEvent::ToolCallStarted {
                call_id,
                name,
                args,
            } => {
                self.timeline.push_tool_started(ToolStartedEvent {
                    call_id: call_id.clone(),
                    name: name.clone(),
                    summary: format_tool_call(name, args),
                    arguments: Some(args.to_string()),
                });
            }
            TranscriptEvent::ToolCallFinished {
                call_id,
                name,
                ok,
                output,
            } => {
                self.timeline.push_tool_finished(ToolFinishedEvent {
                    call_id: call_id.clone(),
                    name: name.clone(),
                    summary: restored_tool_summary(name, *ok),
                    outcome: if *ok {
                        ToolOutcome::Success
                    } else {
                        ToolOutcome::Failure
                    },
                    output: serde_json::to_value(output)
                        .ok()
                        .map(|value| value.to_string()),
                });
            }
            TranscriptEvent::ToolCallCancelled { call_id, name } => {
                self.timeline.cancel_tool(call_id, name);
            }
            TranscriptEvent::TodoSnapshot { items } => {
                self.timeline
                    .push_todo_snapshot(TodoSnapshotEvent::new(items.clone()));
                self.timeline
                    .apply_auto_continue_changed(AutoContinueChangedEvent::new(
                        self.current_auto_continue.clone(),
                    ));
            }
            TranscriptEvent::AutoContinueChanged { state } => {
                self.current_auto_continue = state.clone();
                self.timeline
                    .apply_auto_continue_changed(AutoContinueChangedEvent::new(state.clone()));
            }
            TranscriptEvent::PermissionDecision {
                call_id,
                tool,
                args,
                allowed,
                reason,
            } => {
                self.timeline.push_restored_permission_decision(
                    call_id.clone().unwrap_or_else(|| tool.clone()),
                    tool.clone(),
                    format_tool_call(tool, args),
                    Some(args.to_string()),
                    if *allowed {
                        PermissionPromptStatus::Approved
                    } else {
                        PermissionPromptStatus::Denied
                    },
                    reason.clone(),
                );
            }
            TranscriptEvent::Error { message } => {
                self.timeline.push_error(ErrorEvent::new(message.clone()));
            }
            TranscriptEvent::TurnInterrupted { .. } => {
                self.timeline.cancel_active_tools();
                self.timeline.push_notice("Interrupted by user");
            }
            TranscriptEvent::TurnFinalized(event) => {
                if event.outcome == "interrupted" {
                    self.timeline.cancel_active_tools();
                    self.timeline.push_notice("Interrupted by user");
                }
            }
            TranscriptEvent::SubagentResult { .. }
            | TranscriptEvent::SubagentLifecycle { .. }
            | TranscriptEvent::SessionStarted { .. }
            | TranscriptEvent::SessionTitle { .. }
            | TranscriptEvent::TurnStarted(_)
            | TranscriptEvent::ModelChanged { .. }
            | TranscriptEvent::PermissionModeChanged { .. }
            | TranscriptEvent::AutoContinuationScheduled { .. }
            | TranscriptEvent::ValidationAdvisory(_)
            | TranscriptEvent::ToolExecutionSummary(_)
            | TranscriptEvent::Evidence { .. }
            | TranscriptEvent::Unknown => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::ContextCompactionEvent;
    use crate::tui::timeline::{TimelineItem, ToolExecutionStatus};
    use crate::user_content::{UserImageAttachment, UserMessageContent};
    use serde_json::json;

    fn record(event: TranscriptEvent) -> TranscriptRecord {
        TranscriptRecord {
            session_id: "s".into(),
            sequence: 1,
            timestamp_ms: 0,
            event,
        }
    }

    #[test]
    fn restored_permissions_are_terminal_not_pending_prompts() {
        let timeline =
            timeline_from_transcript_records(&[record(TranscriptEvent::PermissionDecision {
                call_id: Some("call-1".into()),
                tool: "shell__exec".into(),
                args: json!({"command": "cargo test"}),
                allowed: false,
                reason: Some("Denied by user from TUI permission prompt".into()),
            })]);

        assert!(matches!(
            timeline.items().first(),
            Some(TimelineItem::Permission(permission))
                if permission.status == PermissionPromptStatus::Denied
                    && permission.resolution_reason.as_deref() == Some("Denied by user from TUI permission prompt")
        ));
    }

    #[test]
    fn restored_compaction_uses_separator_summary_separator_shape() {
        let timeline = timeline_from_transcript_records(&[record(
            TranscriptEvent::ContextCompaction(ContextCompactionEvent {
                summary: "Earlier context summary".into(),
                tail_start_index: 5,
                original_history_items: 11,
                retained_history_items: 3,
            }),
        )]);

        assert_eq!(timeline.items().len(), 3);
        assert!(matches!(timeline.items()[0], TimelineItem::Notice(_)));
        assert!(matches!(timeline.items()[1], TimelineItem::Assistant(_)));
        assert!(matches!(timeline.items()[2], TimelineItem::Notice(_)));
    }

    #[test]
    fn restored_subagent_records_are_ignored_in_timeline_projection() {
        let timeline = timeline_from_transcript_records(&[
            record(TranscriptEvent::SubagentLifecycle {
                run_id: "run-1".into(),
                parent_session_id: "parent".into(),
                parent_run_id: "turn-1".into(),
                agent_name: "explorer".into(),
                status: "running".into(),
                detail: None,
            }),
            TranscriptRecord {
                session_id: "s".into(),
                sequence: 2,
                timestamp_ms: 1,
                event: TranscriptEvent::SubagentResult {
                    run_id: "run-1".into(),
                    parent_session_id: "parent".into(),
                    parent_run_id: "turn-1".into(),
                    child_session_id: "child".into(),
                    agent_name: "explorer".into(),
                    status: "completed".into(),
                    summary: "done".into(),
                },
            },
        ]);

        assert!(timeline.items().is_empty());
    }

    #[test]
    fn restored_tool_events_keep_terminal_outcomes_without_live_pending_path() {
        let timeline = timeline_from_transcript_records(&[
            record(TranscriptEvent::ToolCallStarted {
                call_id: "call-1".into(),
                name: "shell__exec".into(),
                args: json!({"command": "sleep 10"}),
            }),
            TranscriptRecord {
                session_id: "s".into(),
                sequence: 2,
                timestamp_ms: 1,
                event: TranscriptEvent::ToolCallCancelled {
                    call_id: "call-1".into(),
                    name: "shell__exec".into(),
                },
            },
        ]);

        assert!(matches!(
            timeline.items().first(),
            Some(TimelineItem::Tool(tool)) if tool.status == ToolExecutionStatus::Cancelled
        ));
        assert!(timeline.active_tool().is_none());
    }

    #[test]
    fn restored_user_messages_keep_image_attachments() {
        let timeline = timeline_from_transcript_records(&[record(TranscriptEvent::UserMessage {
            content: UserMessageContent::new(
                "inspect this",
                vec![UserImageAttachment {
                    id: "img-1".into(),
                    label: "screen.png".into(),
                    mime: "image/png".into(),
                    data_url: "data:image/png;base64,AAAA".into(),
                }],
            ),
        })]);

        assert!(matches!(
            timeline.items().first(),
            Some(TimelineItem::User(message))
                if message.text == "inspect this"
                    && message.attachments.len() == 1
                    && message.attachments[0].label == "screen.png"
        ));

        let history = restore_session_history_projection(&[record(TranscriptEvent::UserMessage {
            content: UserMessageContent::new(
                "inspect this",
                vec![UserImageAttachment {
                    id: "img-1".into(),
                    label: "screen.png".into(),
                    mime: "image/png".into(),
                    data_url: "data:image/png;base64,AAAA".into(),
                }],
            ),
        })]);
        assert!(matches!(
            history.first(),
            Some(HistoryItem::UserMessage { content })
                if content.attachments.len() == 1 && content.attachments[0].label == "screen.png"
        ));
    }
}
