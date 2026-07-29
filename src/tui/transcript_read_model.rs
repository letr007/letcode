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
            TranscriptEvent::ReasoningMessage { content } => {
                self.timeline.push_restored_reasoning(
                    format!("restored-reasoning-{}", record.sequence),
                    content.clone(),
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
            }
            TranscriptEvent::TurnFinalized(event) => {
                if event.outcome == "interrupted" {
                    self.timeline.cancel_active_tools();
                }
            }
            TranscriptEvent::SubagentStarted { .. }
            | TranscriptEvent::SubagentResult { .. }
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
    fn restored_successful_reconcile_updates_matching_subagent_card() {
        let timeline = timeline_from_transcript_records(&[
            record(
                1,
                TranscriptEvent::ToolCallFinished {
                    call_id: "subagent-1".into(),
                    name: "agent__fixer".into(),
                    ok: true,
                    output: crate::tool::ToolResult::ok(
                        "agent__fixer",
                        json!({
                            "run_id": "run-1",
                            "child_session_id": "child-1",
                            "unreconciled": true,
                            "reconciled": false,
                        }),
                    ),
                },
            ),
            record(
                2,
                TranscriptEvent::ToolCallFinished {
                    call_id: "reconcile-1".into(),
                    name: "agent__reconcile".into(),
                    ok: true,
                    output: crate::tool::ToolResult::ok(
                        "agent__reconcile",
                        json!({
                            "run_id": "run-1",
                            "child_session_id": "child-1",
                            "reconciled": true,
                        }),
                    ),
                },
            ),
        ]);

        let TimelineItem::Tool(tool) = &timeline.items()[0] else {
            panic!("expected subagent tool item");
        };
        let output = serde_json::from_str::<serde_json::Value>(
            tool.output.as_deref().expect("subagent output"),
        )
        .expect("valid subagent output");
        assert_eq!(output["data"]["unreconciled"], json!(false));
        assert_eq!(output["data"]["reconciled"], json!(true));
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
    fn logical_checkpoint_is_a_timeline_noop() {
        let checkpoint = crate::transcript::LogicalCheckpointEventV1 {
            schema_version: 1,
            checkpoint_id: "checkpoint-1".into(),
            turn_id: 1,
            previous_segment_id: 0,
            segment_id: 1,
            previous_checkpoint_id: None,
            boundary_sequence: 1,
            context_scope_revision: 0,
            covered_source_spans: vec![crate::transcript::LogicalCheckpointSourceSpanV1 {
                start_sequence: 1,
                end_sequence: 1,
            }],
            retained_items: Vec::new(),
        };
        let timeline = timeline_from_transcript_records(&[
            record(
                1,
                TranscriptEvent::AssistantMessage {
                    content: "before".into(),
                },
            ),
            record(2, TranscriptEvent::LogicalCheckpoint(checkpoint)),
            record(
                3,
                TranscriptEvent::AssistantMessage {
                    content: "after".into(),
                },
            ),
        ]);

        assert_eq!(timeline.items().len(), 2);
        assert!(matches!(
            timeline.items(),
            [TimelineItem::Assistant(_), TimelineItem::Assistant(_)]
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
