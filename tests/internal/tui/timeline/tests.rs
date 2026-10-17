use super::*;
use crate::agent::{AutoContinueState, TodoItem, TodoStatus};
use crate::tool::{ToolOutputStream, ToolResult};
use crate::transcript::{TranscriptEvent, TranscriptRecord};
use serde_json::json;

#[test]
fn reasoning_elapsed_ticks_at_subsecond_then_second_boundaries() {
    let start = std::time::Instant::now();
    let mut timeline = Timeline::new();
    timeline.push_reasoning_delta(ReasoningDeltaEvent::at("reasoning-1", "Inspecting", start));
    let initial_revision = timeline.item_revisions()[0];

    timeline.tick_reasoning_elapsed(start + std::time::Duration::from_millis(99));
    assert_eq!(timeline.item_revisions()[0], initial_revision);

    timeline.tick_reasoning_elapsed(start + std::time::Duration::from_millis(100));
    let subsecond_revision = timeline.item_revisions()[0];
    assert_ne!(subsecond_revision, initial_revision);

    timeline.tick_reasoning_elapsed(start + std::time::Duration::from_millis(999));
    let last_subsecond_revision = timeline.item_revisions()[0];
    assert_ne!(last_subsecond_revision, subsecond_revision);

    timeline.tick_reasoning_elapsed(start + std::time::Duration::from_millis(1_500));
    let first_second_revision = timeline.item_revisions()[0];
    assert_ne!(first_second_revision, last_subsecond_revision);

    timeline.tick_reasoning_elapsed(start + std::time::Duration::from_millis(1_999));
    assert_eq!(timeline.item_revisions()[0], first_second_revision);

    timeline.tick_reasoning_elapsed(start + std::time::Duration::from_millis(2_000));
    assert_ne!(timeline.item_revisions()[0], first_second_revision);
}

#[test]
fn quick_reasoning_completion_keeps_title_transition_until_animation_finishes() {
    let start = std::time::Instant::now();
    let mut timeline = Timeline::new();
    timeline.push_reasoning_delta(ReasoningDeltaEvent::at("reasoning-1", "First", start));
    timeline.finalize_reasoning(crate::tui::events::ReasoningDoneEvent::at(
        "reasoning-1",
        "First",
        start + std::time::Duration::from_millis(20),
    ));
    timeline.push_reasoning_delta(ReasoningDeltaEvent::at(
        "reasoning-2",
        "Second",
        start + std::time::Duration::from_millis(30),
    ));
    timeline.finalize_reasoning(crate::tui::events::ReasoningDoneEvent::at(
        "reasoning-2",
        "Second",
        start + std::time::Duration::from_millis(50),
    ));

    assert!(matches!(
        timeline.items().last(),
        Some(TimelineItem::Reasoning(reasoning))
            if reasoning.transition_from.as_deref() == Some("First")
                && reasoning.transition_started_at.is_none()
    ));
    timeline.begin_reasoning_transitions(start + std::time::Duration::from_millis(30));
    timeline.finish_reasoning_transitions(start + std::time::Duration::from_millis(149));
    assert!(matches!(
        timeline.items().last(),
        Some(TimelineItem::Reasoning(reasoning))
            if reasoning.transition_from.as_deref() == Some("First")
    ));
    timeline.finish_reasoning_transitions(start + std::time::Duration::from_millis(150));
    assert!(matches!(
        timeline.items().last(),
        Some(TimelineItem::Reasoning(reasoning))
            if reasoning.transition_from.is_none()
                && reasoning.transition_started_at.is_none()
    ));
}

#[test]
fn reasoning_done_without_delta_refreshes_preceding_compact_item() {
    let start = std::time::Instant::now();
    let mut timeline = Timeline::new();
    timeline.push_reasoning_delta(ReasoningDeltaEvent::at("reasoning-1", "First", start));
    timeline.finalize_reasoning(crate::tui::events::ReasoningDoneEvent::at(
        "reasoning-1",
        "First",
        start + std::time::Duration::from_millis(100),
    ));
    let preceding_revision = timeline.item_revisions()[0];

    timeline.finalize_reasoning(crate::tui::events::ReasoningDoneEvent::at(
        "reasoning-without-delta",
        "Second",
        start + std::time::Duration::from_millis(200),
    ));

    assert_ne!(timeline.item_revisions()[0], preceding_revision);
    assert!(matches!(
        timeline.items(),
        [TimelineItem::Reasoning(first), TimelineItem::Reasoning(second)]
            if first.text == "First" && second.text == "Second"
    ));
}

#[test]
fn reused_reasoning_item_id_finalizes_the_latest_streaming_item() {
    let start = std::time::Instant::now();
    let mut timeline = Timeline::new();
    timeline.push_reasoning_delta(ReasoningDeltaEvent::at("reused", "First draft", start));
    timeline.finalize_reasoning(crate::tui::events::ReasoningDoneEvent::at(
        "reused",
        "First final",
        start + std::time::Duration::from_millis(100),
    ));
    timeline.push_reasoning_delta(ReasoningDeltaEvent::at(
        "reused",
        "Second draft",
        start + std::time::Duration::from_millis(200),
    ));
    timeline.finalize_reasoning(crate::tui::events::ReasoningDoneEvent::at(
        "reused",
        "Second final",
        start + std::time::Duration::from_millis(500),
    ));

    assert!(matches!(
        timeline.items(),
        [TimelineItem::Reasoning(first), TimelineItem::Reasoning(second)]
            if first.text == "First final"
                && first.duration_ms == Some(100)
                && !first.streaming
                && second.text == "Second final"
                && second.duration_ms == Some(300)
                && !second.streaming
    ));
}

#[test]
fn reasoning_done_records_duration_and_missing_start_stays_unknown() {
    let start = std::time::Instant::now();
    let mut timeline = Timeline::new();
    timeline.push_reasoning_delta(ReasoningDeltaEvent::at("reasoning-1", "Draft", start));
    timeline.finalize_reasoning(crate::tui::events::ReasoningDoneEvent::at(
        "reasoning-1",
        "Final",
        start + std::time::Duration::from_millis(725),
    ));
    timeline.finalize_reasoning(crate::tui::events::ReasoningDoneEvent::at(
        "reasoning-without-delta",
        "Recovered",
        start + std::time::Duration::from_secs(1),
    ));

    assert!(matches!(
        timeline.items(),
        [TimelineItem::Reasoning(first), TimelineItem::Reasoning(second)]
            if !first.streaming
                && first.started_at.is_none()
                && first.duration_ms == Some(725)
                && first.text == "Final"
                && !second.streaming
                && second.started_at.is_none()
                && second.duration_ms.is_none()
    ));
}

#[test]
fn compaction_preview_accumulates_and_commits_as_durable_block() {
    let mut timeline = Timeline::new();
    timeline.start_compaction();
    timeline.append_compaction_preview("first ");
    timeline.append_compaction_preview("second");

    assert!(matches!(
        timeline.items(),
        [TimelineItem::Compaction(view)] if view.streaming && view.summary == "first second"
    ));

    timeline.finish_compaction(true);
    assert!(matches!(
        timeline.items(),
        [TimelineItem::Compaction(view)] if !view.streaming && view.summary == "first second"
    ));
}

#[test]
fn compaction_without_summary_keeps_existing_transcript_unchanged() {
    let mut timeline = Timeline::new();
    timeline.push_restored_compaction("An existing summary");
    for _ in 0..2 {
        timeline.start_compaction();
        assert_eq!(timeline.items().len(), 1);
        timeline.finish_compaction(true);
        assert!(matches!(timeline.items(), [TimelineItem::Compaction(view)]
            if !view.streaming && view.summary == "An existing summary"));
    }
}

#[test]
fn compaction_preview_is_discarded_on_non_success() {
    let mut timeline = Timeline::new();
    timeline.start_compaction();
    timeline.append_compaction_preview("transient");
    timeline.finish_compaction(false);

    assert!(timeline.items().is_empty());
}

#[test]
fn restored_conversation_hides_runtime_context_sections_but_keeps_normal_summaries() {
    let timeline = Timeline::from_conversation(vec![
        ConversationMessage {
            role: ConversationRole::Summary,
            content: "[Context: Hard Context]\n- hidden runtime context".into(),
        },
        ConversationMessage {
            role: ConversationRole::Summary,
            content: "Compacted conversation summary".into(),
        },
        ConversationMessage {
            role: ConversationRole::User,
            content: "continue".into(),
        },
    ]);

    assert_eq!(timeline.items().len(), 2);
    assert!(matches!(
        &timeline.items()[0],
        TimelineItem::Compaction(view)
            if !view.streaming && view.summary == "Compacted conversation summary"
    ));
    assert!(matches!(
        &timeline.items()[1],
        TimelineItem::User(message) if message.text == "continue"
    ));
}

#[test]
fn late_duplicate_pending_does_not_regress_running_item() {
    let mut timeline = Timeline::new();

    timeline.push_tool_pending(ToolPendingEvent::new("tool-1", "workflow__todos"));
    timeline.push_tool_started(ToolStartedEvent {
        call_id: "tool-1".into(),
        name: "workflow__todos".into(),
        summary: "update todo list".into(),
        arguments: Some(r#"{"items":[]}"#.into()),
    });
    timeline.push_tool_pending(ToolPendingEvent::new("tool-1", "workflow__todos"));

    let items = timeline.items();
    assert_eq!(items.len(), 1);
    match &items[0] {
        TimelineItem::Tool(tool) => {
            assert_eq!(tool.status, ToolExecutionStatus::Running);
            assert_eq!(tool.summary, "update todo list");
            assert_eq!(tool.arguments.as_deref(), Some(r#"{"items":[]}"#));
        }
        other => panic!("expected tool item, got {other:?}"),
    }
}

#[test]
fn permission_resolution_updates_existing_prompt() {
    let mut timeline = Timeline::new();

    timeline.push_permission_request(PermissionRequestEvent {
        call_id: "call-2".into(),
        tool_name: "git".into(),
        summary: "commit changes".into(),
        arguments: Some("git commit -m test".into()),
        rationale: Some("Needed to save work".into()),
        origin_label: None,
        can_allow_always: false,
        grant_summary: None,
    });
    timeline.resolve_permission(PermissionResolutionEvent::denied(
        "call-2",
        Some("not approved".into()),
    ));

    let items = timeline.items();
    assert_eq!(items.len(), 1);
    match &items[0] {
        TimelineItem::Permission(permission) => {
            assert_eq!(permission.status, PermissionPromptStatus::Denied);
            assert_eq!(
                permission.resolution_reason.as_deref(),
                Some("not approved")
            );
        }
        other => panic!("expected permission item, got {other:?}"),
    }
}

#[test]
fn reviewer_permission_resolutions_keep_event_order() {
    let mut timeline = Timeline::new();
    timeline.resolve_permission(PermissionResolutionEvent {
        call_id: "call-1".into(),
        decision: PermissionDecision::Approved,
        reason: Some("safe".into()),
        tool_name: Some("fs__write".into()),
        summary: Some("write file".into()),
        origin_label: Some("reviewer".into()),
        approval: Some("once".into()),
        risk: Some("low".into()),
        reviewer_child_session_id: Some("reviewer-a".into()),
    });
    timeline.push_tool_started(ToolStartedEvent::new(
        "tool-between",
        "fs__read",
        "read file",
    ));
    timeline.resolve_permission(PermissionResolutionEvent {
        call_id: "call-2".into(),
        decision: PermissionDecision::Denied,
        reason: Some("blocked".into()),
        tool_name: Some("shell__exec".into()),
        summary: Some("run command".into()),
        origin_label: Some("reviewer".into()),
        approval: Some("deny".into()),
        risk: Some("high".into()),
        reviewer_child_session_id: Some("reviewer-a".into()),
    });

    assert!(matches!(
        timeline.items(),
        [TimelineItem::AutoReview(first), TimelineItem::Tool(_), TimelineItem::AutoReview(second)]
            if first.call_id == "call-1"
                && first.allowed
                && second.call_id == "call-2"
                && !second.allowed
    ));
}

#[test]
fn reviewer_permission_resolution_updates_matching_call_in_place() {
    let mut timeline = Timeline::new();
    let resolution = |decision, reason: &str| PermissionResolutionEvent {
        call_id: "call-1".into(),
        decision,
        reason: Some(reason.into()),
        tool_name: Some("shell__exec".into()),
        summary: Some("run command".into()),
        origin_label: Some("reviewer".into()),
        approval: Some(if decision == PermissionDecision::Approved {
            "once".into()
        } else {
            "deny".into()
        }),
        risk: Some("high".into()),
        reviewer_child_session_id: None,
    };
    timeline.resolve_permission(resolution(PermissionDecision::Approved, "safe"));
    timeline.resolve_permission(resolution(PermissionDecision::Denied, "updated"));

    assert!(matches!(
        timeline.items(),
        [TimelineItem::AutoReview(decision)]
            if decision.call_id == "call-1"
                && !decision.allowed
                && decision.rationale == "updated"
    ));
}

#[test]
fn transcript_restore_uses_one_committed_compaction_separator() {
    let records = vec![TranscriptRecord {
        session_id: "session".into(),
        sequence: 1,
        timestamp_ms: 0,
        context_branch_id: None,
        event: TranscriptEvent::ContextCompaction(crate::agent::ContextCompactionEvent::succeeded(
            "目标\n- 继续任务",
            2,
        )),
    }];

    let timeline = Timeline::from_transcript_records(&records);
    let items = timeline.items();

    assert_eq!(items.len(), 1);
    assert!(matches!(
        &items[0],
        TimelineItem::Compaction(view)
            if !view.streaming && view.summary == "目标\n- 继续任务"
    ));
}

#[test]
fn transcript_restore_renders_permission_decision_item() {
    let records = vec![TranscriptRecord {
        session_id: "s".into(),
        sequence: 1,
        timestamp_ms: 0,
        context_branch_id: None,
        event: TranscriptEvent::PermissionDecision {
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
    }];

    let timeline = Timeline::from_transcript_records(&records);
    assert!(matches!(
        timeline.items().first(),
        Some(TimelineItem::Permission(permission))
            if permission.call_id == "call-1"
                && permission.tool_name == "shell__exec"
                && permission.status == PermissionPromptStatus::Denied
                && permission.resolution_reason.as_deref() == Some("Denied by user from TUI permission prompt")
    ));
}

#[test]
fn transcript_restore_renders_cancelled_tool_call_as_terminal_tool() {
    let records = vec![TranscriptRecord {
        session_id: "s".into(),
        sequence: 1,
        timestamp_ms: 0,
        context_branch_id: None,
        event: TranscriptEvent::ToolCallCancelled {
            call_id: "call-1".into(),
            name: "shell__exec".into(),
        },
    }];

    let timeline = Timeline::from_transcript_records(&records);
    assert!(matches!(
        timeline.items().first(),
        Some(TimelineItem::Tool(tool))
            if tool.call_id == "call-1"
                && tool.name == "shell__exec"
                && tool.status == ToolExecutionStatus::Cancelled
    ));
    assert!(timeline.active_tool().is_none());
}

#[test]
fn interrupt_cancels_only_active_tools() {
    let mut timeline = Timeline::new();
    timeline.push_tool_pending(ToolPendingEvent::new("call-pending", "shell__exec"));
    timeline.push_tool_started(ToolStartedEvent::new(
        "call-running",
        "fs__write",
        "write file",
    ));
    timeline.push_tool_finished(ToolFinishedEvent::new(
        "call-done",
        "fs__read",
        "read file",
        ToolOutcome::Success,
    ));

    assert_eq!(timeline.cancel_active_tools(), 2);

    let tools = timeline
        .items()
        .iter()
        .filter_map(|item| match item {
            TimelineItem::Tool(tool) => Some(tool),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(tools[0].status, ToolExecutionStatus::Cancelled);
    assert_eq!(tools[1].status, ToolExecutionStatus::Cancelled);
    assert_eq!(tools[2].status, ToolExecutionStatus::Succeeded);
}

#[test]
fn transcript_restore_closes_interrupted_tool_turns() {
    let records = vec![
        TranscriptRecord {
            session_id: "s".into(),
            sequence: 1,
            timestamp_ms: 0,
            context_branch_id: None,
            event: TranscriptEvent::ToolCallStarted {
                call_id: "call-1".into(),
                name: "shell__exec".into(),
                args: json!({"command": "sleep 10"}),
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

    let timeline = Timeline::from_transcript_records(&records);
    assert!(matches!(
        timeline.items().first(),
        Some(TimelineItem::Tool(tool)) if tool.status == ToolExecutionStatus::Cancelled
    ));
    assert_eq!(timeline.items().len(), 1);
    assert!(timeline.active_tool().is_none());
}

#[test]
fn cancelled_tool_ignores_late_finished_event() {
    let mut timeline = Timeline::new();
    timeline.push_tool_started(ToolStartedEvent::new(
        "call-1",
        "shell__exec",
        "run command",
    ));
    timeline.cancel_active_tools();

    timeline.push_tool_finished(ToolFinishedEvent::new(
        "call-1",
        "shell__exec",
        "shell__exec completed",
        ToolOutcome::Success,
    ));

    assert!(matches!(
        timeline.items().first(),
        Some(TimelineItem::Tool(tool)) if tool.status == ToolExecutionStatus::Cancelled
    ));
}

#[test]
fn child_live_summaries_remain_bound_to_their_parent_tool_cards() {
    let mut timeline = Timeline::new();
    timeline.push_tool_started(ToolStartedEvent::new(
        "explorer-call",
        "agent__explore",
        "inspect src/tui",
    ));
    timeline.push_tool_started(ToolStartedEvent::new(
        "fixer-call",
        "agent__fixer",
        "fix src/tui",
    ));
    // Events may arrive in a different order from the parent tool cards.
    assert!(timeline.update_active_subagent_tool_live_summary(
        "explorer-child",
        Some("explorer"),
        Some("explorer-call"),
        "running",
        "explorer working",
    ));
    assert!(timeline.update_active_subagent_tool_live_summary(
        "fixer-child",
        Some("fixer"),
        Some("fixer-call"),
        "running",
        "fixer working",
    ));
    assert!(timeline.update_active_subagent_tool_live_summary(
        "explorer-child",
        Some("explorer"),
        Some("explorer-call"),
        "completed",
        "explorer done",
    ));

    let tools = timeline
        .items()
        .iter()
        .filter_map(|item| match item {
            TimelineItem::Tool(tool) => Some(tool),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(tools[0].summary, "explorer done");
    let explorer_data = tools[0]
        .output
        .as_deref()
        .and_then(|output| serde_json::from_str::<serde_json::Value>(output).ok())
        .expect("explorer live data");
    assert_eq!(
        explorer_data
            .get("background")
            .and_then(serde_json::Value::as_bool),
        Some(false)
    );
    assert_eq!(tools[1].summary, "fixer working");
    let fixer_data = tools[1]
        .output
        .as_deref()
        .and_then(|output| serde_json::from_str::<serde_json::Value>(output).ok())
        .expect("fixer live data");
    assert_eq!(
        fixer_data
            .get("background")
            .and_then(serde_json::Value::as_bool),
        Some(false)
    );
}

#[test]
fn background_child_live_summary_keeps_background_flag() {
    let mut timeline = Timeline::new();
    timeline.push_tool_started(ToolStartedEvent {
        call_id: "background-call".into(),
        name: "agent__explore".into(),
        summary: "inspect in background".into(),
        arguments: Some(json!({"task": "inspect", "background": true}).to_string()),
    });

    assert!(timeline.update_active_subagent_tool_live_summary(
        "background-child",
        Some("explorer"),
        Some("background-call"),
        "running",
        "background explorer working",
    ));

    let Some(TimelineItem::Tool(tool)) = timeline.items().first() else {
        panic!("expected background subagent tool");
    };
    let data = tool
        .output
        .as_deref()
        .and_then(|output| serde_json::from_str::<serde_json::Value>(output).ok())
        .expect("background live data");
    assert_eq!(
        data.get("background").and_then(serde_json::Value::as_bool),
        Some(true)
    );
}

#[test]
fn waiting_run_clones_a_live_card_below_the_background_receipt() {
    let mut timeline = Timeline::new();
    timeline.push_tool_started(ToolStartedEvent {
        call_id: "background-call".into(),
        name: "agent__explore".into(),
        summary: "explore in background".into(),
        arguments: Some(json!({"task": "inspect wait flow", "background": true}).to_string()),
    });
    timeline.push_tool_finished(ToolFinishedEvent {
        call_id: "background-call".into(),
        name: "agent__explore".into(),
        summary: "explorer running".into(),
        outcome: ToolOutcome::Success,
        output: Some(
            json!({
                "ok": true,
                "tool": "agent__explore",
                "data": {
                    "run_id": "run-bg",
                    "child_session_id": "child-bg",
                    "agent_name": "explorer",
                    "status": "running",
                    "summary": "inspect wait flow",
                    "active": true,
                    "background": true
                }
            })
            .to_string(),
        ),
    });
    assert!(timeline.begin_subagent_wait("wait-call", "run-bg"));
    assert!(matches!(
        timeline.items().last(),
        Some(TimelineItem::Tool(waiting))
            if waiting.call_id == "wait-call"
                && waiting.status == ToolExecutionStatus::Running
    ));
    assert!(timeline.update_active_subagent_tool_live_summary(
        "child-bg",
        Some("explorer"),
        Some("background-call"),
        "running",
        "search__rg — searching wait flow",
    ));

    let tools = timeline
        .items()
        .iter()
        .filter_map(|item| match item {
            TimelineItem::Tool(tool) => Some(tool),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(tools.len(), 2);
    assert_eq!(tools[0].call_id, "background-call");
    assert_eq!(tools[0].status, ToolExecutionStatus::Succeeded);
    assert_eq!(tools[0].summary, "explorer running");
    let background_data = tools[0]
        .output
        .as_deref()
        .and_then(|output| serde_json::from_str::<serde_json::Value>(output).ok())
        .expect("background card data");
    assert_eq!(
        background_data
            .get("data")
            .unwrap_or(&background_data)
            .get("background")
            .and_then(serde_json::Value::as_bool),
        Some(true)
    );
    assert_eq!(tools[1].call_id, "wait-call");
    assert_eq!(tools[1].name, "agent__explore");
    assert_eq!(tools[1].status, ToolExecutionStatus::Running);
    assert_eq!(tools[1].summary, "search__rg — searching wait flow");
    let data = tools[1]
        .output
        .as_deref()
        .and_then(|output| serde_json::from_str::<serde_json::Value>(output).ok())
        .expect("waiting card data");
    assert_eq!(
        data.get("waiting").and_then(serde_json::Value::as_bool),
        Some(true)
    );
    assert_eq!(
        data.get("background").and_then(serde_json::Value::as_bool),
        Some(false)
    );
}

#[test]
fn waiting_run_completes_the_cloned_wait_card() {
    let mut timeline = Timeline::new();
    timeline.push_tool_started(ToolStartedEvent {
        call_id: "background-call".into(),
        name: "agent__explore".into(),
        summary: "explore in background".into(),
        arguments: Some(json!({"task": "inspect wait flow", "background": true}).to_string()),
    });
    timeline.push_tool_finished(ToolFinishedEvent {
        call_id: "background-call".into(),
        name: "agent__explore".into(),
        summary: "explorer running".into(),
        outcome: ToolOutcome::Success,
        output: Some(
            json!({
                "ok": true,
                "tool": "agent__explore",
                "data": {
                    "run_id": "run-bg",
                    "child_session_id": "child-bg",
                    "agent_name": "explorer",
                    "status": "running",
                    "summary": "inspect wait flow",
                    "active": true,
                    "background": true
                }
            })
            .to_string(),
        ),
    });
    assert!(timeline.begin_subagent_wait("wait-call", "run-bg"));
    let result = crate::subagent::SubagentRunSummary {
        run_id: "run-bg".into(),
        child_session_id: "child-bg".into(),
        agent_name: "explorer".into(),
        status: crate::subagent::SubagentStatus::Completed,
        failure_kind: None,
        summary: "wait flow verified".into(),
        structured_result: crate::subagent::StructuredSubagentResult {
            status: "completed".into(),
            summary: "wait flow verified".into(),
            malformed: false,
            findings: Vec::new(),
            files_read: vec!["src/tui/timeline.rs".into()],
            files_changed: Vec::new(),
            commands_run: Vec::new(),
            validation: Vec::new(),
            blockers: Vec::new(),
            next_steps: Vec::new(),
            run_id: "run-bg".into(),
            child_session_id: "child-bg".into(),
            raw_excerpt: None,
        },
    };
    let output = ToolResult::ok(
        crate::tool_names::TOOL_AGENT_WAIT,
        json!({
            "run_id": result.run_id,
            "child_session_id": result.child_session_id,
            "agent_name": result.agent_name,
            "status": result.status.as_str(),
            "failure_kind": serde_json::Value::Null,
            "summary": result.summary,
            "structured_result": result.structured_result,
            "active": false
        }),
    );

    assert!(
        timeline.push_tool_finished(ToolFinishedEvent {
            call_id: "wait-call".into(),
            name: crate::tool_names::TOOL_AGENT_WAIT.into(),
            summary: "explorer completed".into(),
            outcome: ToolOutcome::Success,
            output: Some(
                serde_json::to_value(output)
                    .expect("serialize output")
                    .to_string()
            ),
        })
    );

    let tools = timeline
        .items()
        .iter()
        .filter_map(|item| match item {
            TimelineItem::Tool(tool) => Some(tool),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(tools.len(), 2);
    assert_eq!(tools[0].call_id, "background-call");
    assert_eq!(tools[0].status, ToolExecutionStatus::Succeeded);
    assert_eq!(tools[0].summary, "explorer running");
    assert!(
        tools[0]
            .output
            .as_deref()
            .is_some_and(|output| output.contains("\"background\":true"))
    );
    assert_eq!(tools[1].call_id, "wait-call");
    assert_eq!(tools[1].status, ToolExecutionStatus::Succeeded);
    assert_eq!(tools[1].summary, "wait flow verified");
    assert!(
        tools[1]
            .output
            .as_deref()
            .is_some_and(|output| output.contains("\"waiting\":false"))
    );
}

#[test]
fn interrupted_wait_stays_terminal_after_late_child_activity() {
    let mut timeline = Timeline::new();
    timeline.push_tool_started(ToolStartedEvent {
        call_id: "background-call".into(),
        name: "agent__explore".into(),
        summary: "explore in background".into(),
        arguments: Some(json!({"task": "inspect wait flow", "background": true}).to_string()),
    });
    timeline.push_tool_finished(ToolFinishedEvent {
        call_id: "background-call".into(),
        name: "agent__explore".into(),
        summary: "explorer running".into(),
        outcome: ToolOutcome::Success,
        output: Some(
            json!({
                "ok": true,
                "tool": "agent__explore",
                "data": {
                    "run_id": "run-bg",
                    "child_session_id": "child-bg",
                    "agent_name": "explorer",
                    "status": "running",
                    "summary": "inspect wait flow",
                    "active": true,
                    "background": true
                }
            })
            .to_string(),
        ),
    });
    assert!(timeline.begin_subagent_wait("wait-call", "run-bg"));
    assert_eq!(timeline.cancel_foreground_subagent_waits(), 1);
    assert!(!timeline.update_active_subagent_tool_live_summary(
        "child-bg",
        Some("explorer"),
        Some("background-call"),
        "running",
        "late child action",
    ));
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
fn unstructured_wait_failure_clears_foreground_routing() {
    let mut timeline = Timeline::new();
    timeline.push_tool_started(ToolStartedEvent {
        call_id: "background-call".into(),
        name: "agent__explore".into(),
        summary: "explore in background".into(),
        arguments: Some(json!({"task": "inspect wait flow", "background": true}).to_string()),
    });
    timeline.push_tool_finished(ToolFinishedEvent {
        call_id: "background-call".into(),
        name: "agent__explore".into(),
        summary: "explorer running".into(),
        outcome: ToolOutcome::Success,
        output: Some(
            json!({
                "ok": true,
                "tool": "agent__explore",
                "data": {
                    "run_id": "run-bg",
                    "child_session_id": "child-bg",
                    "agent_name": "explorer",
                    "status": "running",
                    "summary": "inspect wait flow",
                    "active": true,
                    "background": true
                }
            })
            .to_string(),
        ),
    });
    assert!(timeline.begin_subagent_wait("wait-call", "run-bg"));
    assert!(timeline.push_tool_finished(ToolFinishedEvent {
        call_id: "wait-call".into(),
        name: crate::tool_names::TOOL_AGENT_WAIT.into(),
        summary: "wait failed".into(),
        outcome: ToolOutcome::Failure,
        output: Some(json!({"ok": false, "error": "wait failed"}).to_string()),
    }));
    assert!(!timeline.update_active_subagent_tool_live_summary(
        "child-bg",
        Some("explorer"),
        Some("background-call"),
        "running",
        "late child activity",
    ));
    assert!(matches!(
        timeline.items(),
        [TimelineItem::Tool(background), TimelineItem::Tool(waiting)]
            if background.summary == "explorer running"
                && waiting.call_id == "wait-call"
                && waiting.status == ToolExecutionStatus::Failed
    ));
}

#[test]
fn late_background_completion_does_not_overwrite_a_completed_wait_clone() {
    let mut timeline = Timeline::new();
    timeline.push_tool_started(ToolStartedEvent {
        call_id: "background-call".into(),
        name: "agent__explore".into(),
        summary: "explore in background".into(),
        arguments: Some(json!({"task": "inspect wait flow", "background": true}).to_string()),
    });
    timeline.push_tool_finished(ToolFinishedEvent {
        call_id: "background-call".into(),
        name: "agent__explore".into(),
        summary: "explorer running".into(),
        outcome: ToolOutcome::Success,
        output: Some(
            json!({
                "ok": true,
                "tool": "agent__explore",
                "data": {
                    "run_id": "run-bg",
                    "child_session_id": "child-bg",
                    "agent_name": "explorer",
                    "status": "running",
                    "summary": "inspect wait flow",
                    "active": true,
                    "background": true
                }
            })
            .to_string(),
        ),
    });
    assert!(timeline.begin_subagent_wait("wait-call", "run-bg"));
    let result = crate::subagent::SubagentRunSummary {
        run_id: "run-bg".into(),
        child_session_id: "child-bg".into(),
        agent_name: "explorer".into(),
        status: crate::subagent::SubagentStatus::Completed,
        failure_kind: None,
        summary: "wait completed".into(),
        structured_result: crate::subagent::StructuredSubagentResult {
            status: "completed".into(),
            summary: "wait completed".into(),
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
        },
    };

    assert!(timeline.finish_subagent_wait("wait-call", &result));
    assert!(!timeline.finish_background_subagent_tool("background-call", &result));
    assert!(matches!(
        timeline.items(),
        [TimelineItem::Tool(background), TimelineItem::Tool(waiting)]
            if background.summary == "explorer running"
                && waiting.summary == "wait completed"
    ));
}

#[test]
fn finished_subagent_tool_output_keeps_the_host_failure_kind() {
    let mut timeline = Timeline::new();
    timeline.push_tool_started(ToolStartedEvent {
        call_id: "background-call".into(),
        name: "agent__fixer".into(),
        summary: "apply fix in background".into(),
        arguments: Some(json!({"task": "apply fix", "background": true}).to_string()),
    });
    timeline.push_tool_finished(ToolFinishedEvent {
        call_id: "background-call".into(),
        name: "agent__fixer".into(),
        summary: "fixer running".into(),
        outcome: ToolOutcome::Success,
        output: Some(
            json!({
                "ok": true,
                "tool": "agent__fixer",
                "data": {
                    "run_id": "run-fk",
                    "child_session_id": "child-fk",
                    "agent_name": "fixer",
                    "status": "running",
                    "summary": "apply fix",
                    "active": true,
                    "background": true
                }
            })
            .to_string(),
        ),
    });
    assert!(timeline.begin_subagent_wait("wait-call", "run-fk"));
    let result = crate::subagent::SubagentRunSummary {
        run_id: "run-fk".into(),
        child_session_id: "child-fk".into(),
        agent_name: "fixer".into(),
        status: crate::subagent::SubagentStatus::Failed,
        failure_kind: Some(crate::subagent::SubagentFailureKind::Logical),
        summary: "out-of-scope changes detected".into(),
        structured_result: crate::subagent::StructuredSubagentResult {
            status: "failed".into(),
            summary: "out-of-scope changes detected".into(),
            malformed: false,
            findings: Vec::new(),
            files_read: Vec::new(),
            files_changed: Vec::new(),
            commands_run: Vec::new(),
            validation: Vec::new(),
            blockers: Vec::new(),
            next_steps: Vec::new(),
            run_id: "run-fk".into(),
            child_session_id: "child-fk".into(),
            raw_excerpt: None,
        },
    };

    assert!(timeline.finish_subagent_wait("wait-call", &result));
    assert!(matches!(
        timeline.items(),
        [TimelineItem::Tool(background), TimelineItem::Tool(waiting)]
            if background.summary == "fixer running"
                && waiting.output.as_deref().is_some_and(|output| output
                    .contains("\"failure_kind\":\"logical\""))
    ));
}

#[test]
fn a_second_wait_does_not_replace_the_active_wait_clone() {
    let mut timeline = Timeline::new();
    timeline.push_tool_started(ToolStartedEvent {
        call_id: "background-call".into(),
        name: "agent__explore".into(),
        summary: "explore in background".into(),
        arguments: Some(json!({"task": "inspect wait flow", "background": true}).to_string()),
    });
    timeline.push_tool_finished(ToolFinishedEvent {
        call_id: "background-call".into(),
        name: "agent__explore".into(),
        summary: "explorer running".into(),
        outcome: ToolOutcome::Success,
        output: Some(
            json!({
                "ok": true,
                "tool": "agent__explore",
                "data": {
                    "run_id": "run-bg",
                    "child_session_id": "child-bg",
                    "agent_name": "explorer",
                    "status": "running",
                    "summary": "inspect wait flow",
                    "active": true,
                    "background": true
                }
            })
            .to_string(),
        ),
    });

    assert!(timeline.begin_subagent_wait("wait-call", "run-bg"));
    assert!(!timeline.begin_subagent_wait("wait-call-2", "run-bg"));
    assert_eq!(
        timeline
            .items()
            .iter()
            .filter(|item| matches!(item, TimelineItem::Tool(_)))
            .count(),
        2
    );
}

#[test]
fn waiting_run_uses_its_terminal_result_when_completion_events_race() {
    let mut timeline = Timeline::new();
    timeline.push_tool_started(ToolStartedEvent {
        call_id: "background-call".into(),
        name: "agent__explore".into(),
        summary: "explore in background".into(),
        arguments: Some(json!({"task": "inspect wait flow", "background": true}).to_string()),
    });
    timeline.push_tool_finished(ToolFinishedEvent {
        call_id: "background-call".into(),
        name: "agent__explore".into(),
        summary: "explorer running".into(),
        outcome: ToolOutcome::Success,
        output: Some(
            json!({
                "ok": true,
                "tool": "agent__explore",
                "data": {
                    "run_id": "run-bg",
                    "child_session_id": "child-bg",
                    "agent_name": "explorer",
                    "status": "running",
                    "summary": "inspect wait flow",
                    "active": true,
                    "background": true
                }
            })
            .to_string(),
        ),
    });
    assert!(timeline.begin_subagent_wait("wait-call", "run-bg"));
    let result = crate::subagent::SubagentRunSummary {
        run_id: "run-bg".into(),
        child_session_id: "child-bg".into(),
        agent_name: "explorer".into(),
        status: crate::subagent::SubagentStatus::Completed,
        failure_kind: None,
        summary: "late completion".into(),
        structured_result: crate::subagent::StructuredSubagentResult {
            status: "completed".into(),
            summary: "late completion".into(),
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
        },
    };
    assert!(!timeline.finish_background_subagent_tool("background-call", &result));
    assert!(timeline.finish_subagent_wait("wait-call", &result));
    assert!(matches!(
        timeline.items(),
        [TimelineItem::Tool(background), TimelineItem::Tool(waiting)]
            if background.call_id == "background-call"
                && background.status == ToolExecutionStatus::Succeeded
                && background.summary == "explorer running"
                && waiting.call_id == "wait-call"
                && waiting.status == ToolExecutionStatus::Succeeded
                && waiting.summary == "late completion"
    ));
}

#[test]
fn background_completion_replaces_running_receipt_on_parent_tool() {
    let mut timeline = Timeline::new();
    timeline.push_tool_started(ToolStartedEvent::new(
        "background-call",
        "agent__explore",
        "inspect in background",
    ));
    timeline.push_tool_finished(ToolFinishedEvent::new(
        "background-call",
        "agent__explore",
        "explorer running",
        ToolOutcome::Success,
    ));
    let result = crate::subagent::SubagentRunSummary {
        run_id: "run-bg".into(),
        child_session_id: "child-bg".into(),
        agent_name: "explorer".into(),
        status: crate::subagent::SubagentStatus::Completed,
        failure_kind: None,
        summary: "background finding".into(),
        structured_result: crate::subagent::StructuredSubagentResult {
            status: "completed".into(),
            summary: "background finding".into(),
            malformed: false,
            findings: vec!["found it".into()],
            files_read: Vec::new(),
            files_changed: Vec::new(),
            commands_run: Vec::new(),
            validation: Vec::new(),
            blockers: Vec::new(),
            next_steps: Vec::new(),
            run_id: "run-bg".into(),
            child_session_id: "child-bg".into(),
            raw_excerpt: None,
        },
    };

    assert!(timeline.finish_background_subagent_tool("background-call", &result));
    assert!(matches!(
        timeline.items().first(),
        Some(TimelineItem::Tool(tool))
            if tool.status == ToolExecutionStatus::Succeeded
                && tool.summary == "background finding"
                && tool.output.as_deref().is_some_and(|output| output.contains("child-bg"))
    ));
}

#[test]
fn duplicate_role_child_events_use_exact_parent_call_id() {
    let mut timeline = Timeline::new();
    timeline.push_tool_started(ToolStartedEvent::new(
        "older-explorer-call",
        "agent__explore",
        "inspect older path",
    ));
    timeline.push_tool_started(ToolStartedEvent::new(
        "newer-explorer-call",
        "agent__explore",
        "inspect newer path",
    ));

    assert!(timeline.update_active_subagent_tool_live_summary(
        "older-child",
        Some("explorer"),
        Some("older-explorer-call"),
        "running",
        "older explorer working",
    ));
    assert!(timeline.update_active_subagent_tool_live_summary(
        "newer-child",
        Some("explorer"),
        Some("newer-explorer-call"),
        "running",
        "newer explorer working",
    ));

    let tools = timeline
        .items()
        .iter()
        .filter_map(|item| match item {
            TimelineItem::Tool(tool) => Some(tool),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(tools[0].summary, "older explorer working");
    assert_eq!(tools[1].summary, "newer explorer working");
    assert!(
        tools[0]
            .output
            .as_deref()
            .is_some_and(|output| output.contains("older-child"))
    );
    assert!(
        tools[1]
            .output
            .as_deref()
            .is_some_and(|output| output.contains("newer-child"))
    );
}

#[test]
fn late_child_event_does_not_rebind_after_parent_tool_finishes() {
    let mut timeline = Timeline::new();
    timeline.push_tool_started(ToolStartedEvent::new(
        "old-parent-call",
        "agent__explore",
        "inspect old path",
    ));
    timeline.push_tool_finished(ToolFinishedEvent::new(
        "old-parent-call",
        "agent__explore",
        "old explorer completed",
        ToolOutcome::Success,
    ));
    timeline.push_tool_started(ToolStartedEvent::new(
        "new-parent-call",
        "agent__explore",
        "inspect new path",
    ));

    assert!(!timeline.update_active_subagent_tool_live_summary(
        "old-child",
        Some("explorer"),
        Some("old-parent-call"),
        "running",
        "late old event",
    ));

    let tools = timeline
        .items()
        .iter()
        .filter_map(|item| match item {
            TimelineItem::Tool(tool) => Some(tool),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(tools[0].summary, "old explorer completed");
    assert_eq!(tools[0].status, ToolExecutionStatus::Succeeded);
    assert_eq!(tools[1].summary, "inspect new path");
    assert!(tools[1].output.is_none());
}

fn assistant_message() -> MessageView {
    MessageView {
        id: None,
        submission_id: None,
        role: MessageRole::Assistant,
        text: String::new(),
        attachments: Vec::new(),
        selected_skills: Vec::new(),
        streaming: true,
        queued: false,
        reveal_tail: VecDeque::new(),
    }
}

#[test]
fn reveal_tail_fades_for_the_window_then_expires() {
    let start = std::time::Instant::now();
    let mut message = assistant_message();
    message.record_reveal(start, 3);

    assert!(message.is_fading(start));
    assert!(message.is_fading(start + std::time::Duration::from_millis(199)));
    assert!(!message.is_fading(start + std::time::Duration::from_millis(200)));
}

#[test]
fn reveal_tail_drops_batches_older_than_the_fade_window() {
    let start = std::time::Instant::now();
    let mut message = assistant_message();
    message.record_reveal(start, 3);
    message.record_reveal(start + std::time::Duration::from_millis(300), 2);

    assert_eq!(message.reveal_tail.len(), 1);
    assert_eq!(message.reveal_tail.back().map(|(_, count)| *count), Some(2));
}

#[test]
fn live_delta_records_a_reveal_batch() {
    let mut timeline = Timeline::new();
    timeline.push_assistant_delta(AssistantDeltaEvent::new("hi"));
    let TimelineItem::Assistant(message) = &timeline.items()[0] else {
        panic!("expected an assistant item");
    };
    assert_eq!(message.reveal_tail.len(), 1);
    assert_eq!(message.reveal_tail[0].1, 2);
}

#[test]
fn restored_text_never_fades() {
    let mut timeline = Timeline::new();
    timeline.push_restored_assistant_text(AssistantDeltaEvent::new("hello"));
    let TimelineItem::Assistant(message) = &timeline.items()[0] else {
        panic!("expected an assistant item");
    };
    assert!(message.reveal_tail.is_empty());
    assert!(!message.is_fading(std::time::Instant::now()));
}
