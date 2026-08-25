//! Rebuild a `Timeline` model from persisted transcript records.
//!
//! The read side of the transcript pipeline: `TranscriptRecord`s written to disk
//! become a renderable `Timeline`. Backend-agnostic and independent of the
//! terminal rendering layers (`transcript_render` / `transcript_ratatui` /
//! `components::transcript`).

use super::events::{
    AutoContinueChangedEvent, ErrorEvent, TodoSnapshotEvent, ToolFinishedEvent, ToolOutcome,
    ToolStartedEvent, UserMessageEvent,
};
use super::timeline::{MessageRole, PermissionPromptStatus, Timeline, restored_tool_summary};
use crate::agent::AutoContinueState;
use crate::tool_format::format_tool_call;
use crate::transcript::{TranscriptEvent, TranscriptRecord};
use crate::user_content::UserMessageSubmission;

pub(crate) fn timeline_from_transcript_records(records: &[TranscriptRecord]) -> Timeline {
    let mut projection = TranscriptTimelineProjection::default();
    for record in records {
        projection.apply_record(record);
    }
    projection.timeline
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
                // Durable transcript: restore the full compaction block with summary.
                self.timeline
                    .push_restored_compaction(event.summary.clone());
            }
            TranscriptEvent::ReasoningMessage {
                content,
                duration_ms,
            } => {
                self.timeline.push_restored_reasoning(
                    format!("restored-reasoning-{}", record.sequence),
                    content.clone(),
                    *duration_ms,
                );
            }
            TranscriptEvent::ContextExperimentReturned {
                branch_id,
                outcome,
                summary,
                next_action,
                had_writes,
                ..
            } => self.timeline.push_restored_message(
                MessageRole::Assistant,
                crate::transcript::format_context_experiment_return(
                    branch_id,
                    outcome,
                    summary,
                    next_action.as_deref(),
                    *had_writes,
                ),
            ),
            TranscriptEvent::ToolCallStarted {
                call_id,
                name,
                args,
            } => {
                let started = ToolStartedEvent {
                    call_id: call_id.clone(),
                    name: name.clone(),
                    summary: format_tool_call(name, args),
                    arguments: Some(args.to_string()),
                };
                if name == crate::tool_names::TOOL_AGENT_WAIT
                    && args
                        .get("run_id")
                        .and_then(serde_json::Value::as_str)
                        .is_some_and(|run_id| self.timeline.begin_subagent_wait(call_id, run_id))
                {
                    return;
                }
                self.timeline.push_tool_started(started);
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
                if name == crate::tool_names::TOOL_AGENT_WAIT
                    && self.timeline.cancel_foreground_subagent_wait(call_id)
                {
                } else {
                    self.timeline.cancel_tool(call_id, name);
                }
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
                reviewer,
                approval,
                risk,
                reviewer_child_session_id,
            } => {
                let restored_reason = reason.clone();
                let auto_review = matches!(reviewer.as_deref(), Some("auto"));
                let origin_label = auto_review.then(|| "reviewer".to_string());
                let reviewer_child_session_id = reviewer_child_session_id.clone();
                let call_id = call_id.clone().unwrap_or_else(|| {
                    if auto_review {
                        format!("restored-auto-review-{}", record.sequence)
                    } else {
                        tool.clone()
                    }
                });
                self.timeline.push_restored_permission_decision(
                    call_id,
                    tool.clone(),
                    format_tool_call(tool, args),
                    Some(args.to_string()),
                    if *allowed {
                        PermissionPromptStatus::Approved
                    } else {
                        PermissionPromptStatus::Denied
                    },
                    restored_reason,
                    origin_label,
                    approval.clone(),
                    risk.clone(),
                    reviewer_child_session_id,
                );
            }
            TranscriptEvent::Error { message } => {
                self.timeline.push_error(ErrorEvent::new(message.clone()));
            }
            TranscriptEvent::TurnInterrupted { .. } => {
                self.timeline.cancel_foreground_subagent_waits();
                self.timeline.cancel_active_tools();
            }
            TranscriptEvent::TurnFinalized(event) => {
                if event.outcome == "interrupted" {
                    self.timeline.cancel_foreground_subagent_waits();
                    self.timeline.cancel_active_tools();
                }
            }
            TranscriptEvent::SubagentStarted {
                run_id,
                child_session_id,
                agent_name,
                summary,
                ..
            } => {
                self.timeline.register_subagent_started(
                    run_id,
                    child_session_id,
                    agent_name,
                    summary,
                );
            }
            TranscriptEvent::SubagentResult { .. }
            | TranscriptEvent::SubagentLifecycle { .. }
            | TranscriptEvent::LlmRequestTelemetry { .. }
            | TranscriptEvent::SessionStarted { .. }
            | TranscriptEvent::SessionTitle { .. }
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
            | TranscriptEvent::TurnStarted(_)
            | TranscriptEvent::ModelChanged { .. }
            | TranscriptEvent::ReasoningEffortChanged { .. }
            | TranscriptEvent::ExpertModelChanged { .. }
            | TranscriptEvent::PermissionModeChanged { .. }
            | TranscriptEvent::AutoContinuationScheduled { .. }
            | TranscriptEvent::AssistantToolCallBatch { .. }
            | TranscriptEvent::InternalContinuation { .. }
            | TranscriptEvent::ValidationAdvisory(_)
            | TranscriptEvent::ToolExecutionSummary(_)
            | TranscriptEvent::LogicalCheckpoint(_)
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

    fn record(sequence: u64, event: TranscriptEvent) -> TranscriptRecord {
        TranscriptRecord {
            session_id: "s".into(),
            sequence,
            timestamp_ms: 0,
            context_branch_id: None,
            event,
        }
    }

    #[test]
    fn restored_permissions_are_terminal_not_pending_prompts() {
        let timeline = timeline_from_transcript_records(&[record(
            1,
            TranscriptEvent::PermissionDecision {
                call_id: Some("call-1".into()),
                tool: "shell__exec".into(),
                args: json!({"command": "cargo test"}),
                allowed: false,
                reason: Some("Denied by user from TUI permission prompt".into()),
                reviewer: None,
                approval: None,
                risk: None,
                reviewer_child_session_id: None,
            },
        )]);

        assert!(matches!(
            timeline.items().first(),
            Some(TimelineItem::Permission(permission))
                if permission.status == PermissionPromptStatus::Denied
                    && permission.resolution_reason.as_deref() == Some("Denied by user from TUI permission prompt")
        ));
    }

    #[test]
    fn restored_auto_review_permissions_keep_reviewer_identity() {
        let timeline = timeline_from_transcript_records(&[
            record(
                1,
                TranscriptEvent::PermissionDecision {
                    call_id: Some("call-auto".into()),
                    tool: "fs__write".into(),
                    args: json!({"path": "a.txt", "content": "x"}),
                    allowed: true,
                    reason: Some("safe edit".into()),
                    reviewer: Some("auto".into()),
                    approval: Some("once".into()),
                    risk: Some("low".into()),
                    reviewer_child_session_id: Some("child-reviewer".into()),
                },
            ),
            record(
                2,
                TranscriptEvent::PermissionDecision {
                    call_id: Some("call-auto-2".into()),
                    tool: "shell__exec".into(),
                    args: json!({"command": "git status"}),
                    allowed: false,
                    reason: Some("unsafe".into()),
                    reviewer: Some("auto".into()),
                    approval: Some("deny".into()),
                    risk: Some("high".into()),
                    reviewer_child_session_id: Some("child-reviewer".into()),
                },
            ),
        ]);

        assert!(matches!(
            timeline.items(),
            [TimelineItem::AutoReview(first), TimelineItem::AutoReview(second)]
                if first.call_id == "call-auto"
                    && first.approval == "once"
                    && first.risk.as_deref() == Some("low")
                    && first.rationale == "safe edit"
                    && first.allowed
                    && second.call_id == "call-auto-2"
                    && second.approval == "deny"
                    && !second.allowed
        ));
    }

    #[test]
    fn restored_legacy_auto_review_permissions_share_stable_reviewer_group() {
        let timeline = timeline_from_transcript_records(&[
            record(
                1,
                TranscriptEvent::PermissionDecision {
                    call_id: Some("legacy-1".into()),
                    tool: "fs__read".into(),
                    args: json!({"path": "a.txt"}),
                    allowed: true,
                    reason: Some("safe".into()),
                    reviewer: Some("auto".into()),
                    approval: Some("once".into()),
                    risk: Some("low".into()),
                    reviewer_child_session_id: None,
                },
            ),
            record(
                2,
                TranscriptEvent::PermissionDecision {
                    call_id: Some("current-2".into()),
                    tool: "shell__exec".into(),
                    args: json!({"command": "git status"}),
                    allowed: false,
                    reason: Some("blocked".into()),
                    reviewer: Some("auto".into()),
                    approval: Some("deny".into()),
                    risk: Some("high".into()),
                    reviewer_child_session_id: Some("child-reviewer".into()),
                },
            ),
        ]);

        assert!(matches!(
            timeline.items(),
            [TimelineItem::AutoReview(first), TimelineItem::AutoReview(second)]
                if first.call_id == "legacy-1"
                    && first.allowed
                    && second.call_id == "current-2"
                    && !second.allowed
        ));
    }

    #[test]
    fn restored_legacy_auto_reviews_without_call_ids_keep_event_order() {
        let timeline = timeline_from_transcript_records(&[
            record(
                1,
                TranscriptEvent::PermissionDecision {
                    call_id: None,
                    tool: "shell__exec".into(),
                    args: json!({"command": "git status"}),
                    allowed: true,
                    reason: Some("safe".into()),
                    reviewer: Some("auto".into()),
                    approval: Some("once".into()),
                    risk: Some("low".into()),
                    reviewer_child_session_id: None,
                },
            ),
            record(
                2,
                TranscriptEvent::PermissionDecision {
                    call_id: None,
                    tool: "shell__exec".into(),
                    args: json!({"command": "git diff"}),
                    allowed: false,
                    reason: Some("blocked".into()),
                    reviewer: Some("auto".into()),
                    approval: Some("deny".into()),
                    risk: Some("high".into()),
                    reviewer_child_session_id: None,
                },
            ),
        ]);

        assert!(matches!(
            timeline.items(),
            [TimelineItem::AutoReview(first), TimelineItem::AutoReview(second)]
                if first.call_id == "restored-auto-review-1"
                    && first.allowed
                    && second.call_id == "restored-auto-review-2"
                    && !second.allowed
        ));
    }

    #[test]
    fn restored_reasoning_keeps_optional_duration_without_inventing_legacy_time() {
        let timeline = timeline_from_transcript_records(&[
            record(
                1,
                TranscriptEvent::ReasoningMessage {
                    content: "legacy reasoning".into(),
                    duration_ms: None,
                },
            ),
            record(
                2,
                TranscriptEvent::ReasoningMessage {
                    content: "timed reasoning".into(),
                    duration_ms: Some(1_250),
                },
            ),
        ]);

        assert!(matches!(
            timeline.items(),
            [TimelineItem::Reasoning(legacy), TimelineItem::Reasoning(timed)]
                if legacy.text == "legacy reasoning"
                    && legacy.started_at.is_none()
                    && legacy.duration_ms.is_none()
                    && timed.text == "timed reasoning"
                    && timed.started_at.is_none()
                    && timed.duration_ms == Some(1_250)
        ));
    }

    #[test]
    fn restored_compaction_keeps_summary_in_durable_block() {
        let timeline = timeline_from_transcript_records(&[record(
            1,
            TranscriptEvent::ContextCompaction(ContextCompactionEvent::succeeded(
                "Earlier context summary",
                5,
            )),
        )]);

        assert!(matches!(
            timeline.items(),
            [TimelineItem::Compaction(view)]
                if !view.streaming && view.summary == "Earlier context summary"
        ));
    }

    #[test]
    fn restored_legacy_reconcile_tool_is_preserved_as_an_ordinary_tool_record() {
        let timeline = timeline_from_transcript_records(&[
            record(
                1,
                TranscriptEvent::ToolCallStarted {
                    call_id: "legacy-reconcile".into(),
                    name: "agent__reconcile".into(),
                    args: json!({
                        "run_id": "run-1",
                        "child_session_id": "child-1",
                        "decision": "accepted"
                    }),
                },
            ),
            record(
                2,
                TranscriptEvent::ToolCallFinished {
                    call_id: "legacy-reconcile".into(),
                    name: "agent__reconcile".into(),
                    ok: true,
                    output: crate::tool::ToolResult::ok(
                        "agent__reconcile",
                        json!({"reconciled": true}),
                    ),
                },
            ),
        ]);

        assert!(matches!(
            timeline.items(),
            [TimelineItem::Tool(tool)]
                if tool.name == "agent__reconcile"
                    && tool.status == ToolExecutionStatus::Succeeded
        ));
    }

    #[test]
    fn restored_subagent_records_are_ignored_in_timeline_projection() {
        let timeline = timeline_from_transcript_records(&[record(
            1,
            TranscriptEvent::SubagentLifecycle {
                run_id: "run-1".into(),
                parent_session_id: "parent".into(),
                parent_run_id: "turn-1".into(),
                agent_name: "explorer".into(),
                status: "running".into(),
                detail: None,
            },
        )]);
        assert!(timeline.items().is_empty());
    }

    #[test]
    fn restored_wait_keeps_background_history_and_cloned_wait_card() {
        let structured = crate::subagent::StructuredSubagentResult {
            status: "completed".into(),
            summary: "wait restored".into(),
            malformed: false,
            findings: Vec::new(),
            files_read: Vec::new(),
            files_changed: Vec::new(),
            commands_run: Vec::new(),
            validation: Vec::new(),
            blockers: Vec::new(),
            next_steps: Vec::new(),
            run_id: "run-bg".into(),
            child_session_id: "child-bg".into(),
            raw_excerpt: None,
        };
        let timeline = timeline_from_transcript_records(&[
            record(
                1,
                TranscriptEvent::ToolCallStarted {
                    call_id: "background-call".into(),
                    name: "agent__explore".into(),
                    args: json!({"task": "inspect wait flow", "background": true}),
                },
            ),
            record(
                2,
                TranscriptEvent::SubagentStarted {
                    run_id: "run-bg".into(),
                    parent_session_id: "s".into(),
                    parent_run_id: "turn-1".into(),
                    child_session_id: "child-bg".into(),
                    agent_name: "explorer".into(),
                    summary: "inspect wait flow".into(),
                    pool_ordinal: 1,
                },
            ),
            record(
                3,
                TranscriptEvent::ToolCallFinished {
                    call_id: "background-call".into(),
                    name: "agent__explore".into(),
                    ok: true,
                    output: crate::tool::ToolResult::ok(
                        "agent__explore",
                        json!({
                            "run_id": "run-bg",
                            "child_session_id": "child-bg",
                            "agent_name": "explorer",
                            "status": "running",
                            "summary": "inspect wait flow",
                            "active": true,
                            "background": true
                        }),
                    ),
                },
            ),
            record(
                4,
                TranscriptEvent::ToolCallStarted {
                    call_id: "wait-call".into(),
                    name: crate::tool_names::TOOL_AGENT_WAIT.into(),
                    args: json!({"run_id": "run-bg"}),
                },
            ),
            record(
                5,
                TranscriptEvent::ToolCallFinished {
                    call_id: "wait-call".into(),
                    name: crate::tool_names::TOOL_AGENT_WAIT.into(),
                    ok: true,
                    output: crate::tool::ToolResult::ok(
                        crate::tool_names::TOOL_AGENT_WAIT,
                        json!({
                            "run_id": "run-bg",
                            "child_session_id": "child-bg",
                            "agent_name": "explorer",
                            "status": "completed",
                            "failure_kind": null,
                            "summary": "wait restored",
                            "structured_result": structured,
                            "active": false
                        }),
                    ),
                },
            ),
        ]);

        let tools = timeline
            .items()
            .iter()
            .filter_map(|item| match item {
                TimelineItem::Tool(tool) => Some(tool),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            tools
                .iter()
                .map(|tool| (
                    tool.call_id.as_str(),
                    tool.name.as_str(),
                    tool.status,
                    tool.summary.as_str(),
                ))
                .collect::<Vec<_>>(),
            vec![
                (
                    "background-call",
                    "agent__explore",
                    ToolExecutionStatus::Succeeded,
                    "agent__explore completed",
                ),
                (
                    "wait-call",
                    "agent__explore",
                    ToolExecutionStatus::Succeeded,
                    "wait restored",
                ),
            ]
        );
    }

    #[test]
    fn restored_interrupted_wait_is_terminal() {
        let timeline = timeline_from_transcript_records(&[
            record(
                1,
                TranscriptEvent::ToolCallStarted {
                    call_id: "background-call".into(),
                    name: "agent__explore".into(),
                    args: json!({"task": "inspect wait flow", "background": true}),
                },
            ),
            record(
                2,
                TranscriptEvent::SubagentStarted {
                    run_id: "run-bg".into(),
                    parent_session_id: "s".into(),
                    parent_run_id: "turn-1".into(),
                    child_session_id: "child-bg".into(),
                    agent_name: "explorer".into(),
                    summary: "inspect wait flow".into(),
                    pool_ordinal: 1,
                },
            ),
            record(
                3,
                TranscriptEvent::ToolCallFinished {
                    call_id: "background-call".into(),
                    name: "agent__explore".into(),
                    ok: true,
                    output: crate::tool::ToolResult::ok(
                        "agent__explore",
                        json!({
                            "run_id": "run-bg",
                            "child_session_id": "child-bg",
                            "agent_name": "explorer",
                            "status": "running",
                            "summary": "inspect wait flow",
                            "active": true,
                            "background": true
                        }),
                    ),
                },
            ),
            record(
                4,
                TranscriptEvent::ToolCallStarted {
                    call_id: "wait-call".into(),
                    name: crate::tool_names::TOOL_AGENT_WAIT.into(),
                    args: json!({"run_id": "run-bg"}),
                },
            ),
            record(5, TranscriptEvent::TurnInterrupted { turn_id: Some(1) }),
        ]);

        assert!(matches!(
            timeline.items(),
            [TimelineItem::Tool(background), TimelineItem::Tool(waiting)]
                if background.call_id == "background-call"
                    && background.status == ToolExecutionStatus::Succeeded
                    && waiting.call_id == "wait-call"
                    && waiting.status == ToolExecutionStatus::Cancelled
                    && waiting.summary == "subagent wait cancelled"
        ));
    }

    #[test]
    fn restored_tool_events_keep_terminal_outcomes_without_live_pending_path() {
        let timeline = timeline_from_transcript_records(&[
            record(
                1,
                TranscriptEvent::ToolCallStarted {
                    call_id: "call-1".into(),
                    name: "shell__exec".into(),
                    args: json!({"command": "sleep 10"}),
                },
            ),
            record(
                2,
                TranscriptEvent::ToolCallCancelled {
                    call_id: "call-1".into(),
                    name: "shell__exec".into(),
                },
            ),
        ]);
        assert!(matches!(timeline.items().first(),
            Some(TimelineItem::Tool(tool)) if tool.status == ToolExecutionStatus::Cancelled));
        assert!(timeline.active_tool().is_none());
    }

    #[test]
    fn restored_user_messages_keep_image_attachments() {
        let timeline = timeline_from_transcript_records(&[record(
            1,
            TranscriptEvent::UserMessage {
                content: UserMessageContent::new(
                    "inspect this",
                    vec![UserImageAttachment {
                        id: "img-1".into(),
                        label: "screen.png".into(),
                        mime: "image/png".into(),
                        data_url: "data:image/png;base64,AAAA".into(),
                    }],
                ),
            },
        )]);
        assert!(matches!(timeline.items().first(),
            Some(TimelineItem::User(message)) if message.text == "inspect this"
                && message.attachments.first().is_some_and(|attachment| attachment.label == "screen.png")));
    }
}
