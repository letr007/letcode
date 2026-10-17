use super::*;
use crate::agent::ContextCompactionEvent;
use crate::transcript::TranscriptAssistantTurn;
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
fn restored_v2_assistant_turn_projects_text_without_tool_metadata_noise() {
    let timeline = timeline_from_transcript_records(&[record(
        1,
        TranscriptEvent::AssistantTurn(TranscriptAssistantTurn {
            text: Some("final answer".into()),
            reasoning_content: Some("private reasoning".into()),
            replay: None,
            calls: Vec::new(),
        }),
    )]);

    assert!(matches!(
        timeline.items(),
        [TimelineItem::Assistant(message)] if message.text == "final answer"
    ));
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
fn restored_delegation_renders_one_block_instead_of_the_prompt_message() {
    let input = crate::tool::NormalizedSubagentInput {
        objective: "implement bounded fix".into(),
        success_criteria: vec!["tests pass".into()],
        allowed_paths: vec!["src".into()],
        forbidden_paths: vec!["docs".into()],
        owned_paths: vec!["src/tui".into()],
        timeout_secs: None,
        max_tool_calls: None,
        model: Some("NewAPI/deepseek-flash".into()),
        target_child_session_id: Some("child-42".into()),
        background: false,
    };
    let prompt = input.render_for_delegate(crate::tool_names::TOOL_AGENT_FIXER);
    let timeline = timeline_from_transcript_records(&[
        record(
            1,
            TranscriptEvent::SubagentDelegation {
                run_id: "run-1".into(),
                input,
            },
        ),
        record(
            2,
            TranscriptEvent::UserMessage {
                content: prompt.into(),
            },
        ),
    ]);

    assert!(matches!(
        timeline.items(),
        [TimelineItem::Delegation(delegation)]
            if delegation.objective == "implement bounded fix"
                && delegation.success_criteria == ["tests pass"]
                && delegation.owned_paths == ["src/tui"]
                && delegation.allowed_paths == ["src"]
                && delegation.forbidden_paths == ["docs"]
                && delegation.model.as_deref() == Some("NewAPI/deepseek-flash")
                && delegation.target_child_session_id.as_deref() == Some("child-42")
    ));
}

#[test]
fn restored_message_run_wait_remains_a_subagent_card() {
    let timeline = timeline_from_transcript_records(&[
        record(
            1,
            TranscriptEvent::ToolCallStarted {
                call_id: "wait-message".into(),
                name: crate::tool_names::TOOL_AGENT_WAIT.into(),
                args: json!({"run_id": "message-run"}),
            },
        ),
        record(
            2,
            TranscriptEvent::ToolCallFinished {
                call_id: "wait-message".into(),
                name: crate::tool_names::TOOL_AGENT_WAIT.into(),
                ok: true,
                output: crate::tool::ToolResult::ok(
                    crate::tool_names::TOOL_AGENT_WAIT,
                    json!({
                        "run_id": "message-run",
                        "child_session_id": "child-message",
                        "agent_name": "explorer",
                        "status": "completed",
                        "summary": "message processed",
                        "active": false,
                    }),
                ),
            },
        ),
    ]);
    let tool = timeline
        .items()
        .iter()
        .find_map(|item| match item {
            TimelineItem::Tool(tool) => Some(tool),
            _ => None,
        })
        .unwrap();
    assert_eq!(tool.status, ToolExecutionStatus::Succeeded);
    let document = crate::tui::components::tool_card::render_tool_card_document(
        tool,
        crate::tui::theme::Theme::dark(),
        120,
        0,
        false,
        &crate::tui::i18n::Translator::new(crate::tui::i18n::Language::En),
    );
    let rendered = crate::tui::transcript_ratatui::document_to_ratatui(&document)
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        rendered.contains("completed explorer message processed"),
        "{rendered}"
    );
    assert!(rendered.contains("/child child-message"), "{rendered}");
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
