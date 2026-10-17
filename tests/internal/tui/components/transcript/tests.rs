use crate::tui::components::tool_card;

use super::{
    animated_compact_reasoning_title_at, cached_item_lines, cached_transcript_row_count,
    cached_transcript_row_count_with_reflow, render_timeline_item_document, render_transcript,
    transcript_lines, transcript_row_count, try_render_reviewer_view_item,
    visible_cached_transcript_lines, visible_document_lines, visible_transcript_lines,
};
use crate::{
    agent::{AutoContinueState, TodoItem, TodoStatus},
    tool::ToolResult,
    transcript::{TranscriptEvent, TranscriptRecord},
    tui::{
        AssistantDeltaEvent, ErrorEvent, PermissionRequestEvent, ReasoningDeltaEvent,
        ReasoningDoneEvent, SessionEvent, ToolFinishedEvent, ToolOutcome, ToolStartedEvent,
        UserMessageEvent,
        events::{AutoContinueChangedEvent, TodoSnapshotEvent, ToolCancelledEvent},
        measure::display_width,
        state::{ContextDetailTarget, TuiState},
        surface,
        theme::{Theme, ThemeName},
        timeline::{
            CompactionView, ErrorView, MessageRole, MessageView, PermissionPromptStatus,
            PermissionView, ReasoningView, Timeline, TimelineItem, TodoView, ToolExecutionStatus,
            ToolView,
        },
    },
    user_content::{UserImageAttachment, UserMessageContent, UserMessageSubmission},
};
use ratatui::{Terminal, backend::TestBackend, layout::Rect};
use serde_json::json;

#[test]
fn live_reasoning_visible_window_tracks_elapsed_layout() {
    use crate::command::ThoughtsDisplayMode::{Compact, Full, Titles};

    for (display, width) in [(Titles, 80), (Full, 80), (Compact, 18)] {
        for child_view in [false, true] {
            let mut state = TuiState::default();
            let theme = Theme::dark();
            state.set_thoughts_display(display);
            state
                .timeline
                .push_restored_message(MessageRole::Assistant, "parent".into());
            cached_transcript_row_count(&mut state, theme, width);
            let event = SessionEvent::ReasoningDelta(ReasoningDeltaEvent::at(
                "reasoning",
                "B".repeat(64),
                std::time::Instant::now() - std::time::Duration::from_secs(2),
            ));
            if child_view {
                state.replace_child_timeline_from_records(
                    &[],
                    "parent",
                    "child",
                    "explorer",
                    0,
                    1,
                    1,
                );
                state.apply_child_session_event("child", event);
            } else {
                state.apply_event(event);
            }
            cached_transcript_row_count(&mut state, theme, width);
            let index = state.active_timeline().items().len() - 1;

            // Seed a prior frame's millisecond layout using an explicit elapsed
            // sample, without a timing-sensitive sleep around the seconds boundary.
            let previous = render_timeline_item_document(
                &state.active_timeline().items()[index],
                theme,
                width,
                state.status_spinner_frame,
                false,
                false,
                None,
                display,
                false,
                state.tools_display,
                false,
                tool_card::ToolGroupStats::default(),
                Some(960),
                std::time::Instant::now(),
                theme.canvas(),
                &state.translator(),
                None,
            );
            let previous_rows = previous.lines.len();
            state.transcript_render_cache.entries[index].document = previous;
            state.transcript_render_cache.invalidate_row_metadata();
            cached_transcript_row_count(&mut state, theme, width);
            assert_eq!(
                state.transcript_render_cache.row_counts[index],
                previous_rows
            );
            let revision = state.active_timeline().mutation_revision();

            // A presentation frame can change without a timeline mutation.
            state.status_spinner_frame += 1;
            let visible = visible_cached_transcript_lines(&mut state, theme, width, 20, 0);
            assert_eq!(state.active_timeline().mutation_revision(), revision);
            assert_eq!(
                state.transcript_render_cache.row_counts[index],
                previous_rows - 1
            );
            assert_eq!(
                state.transcript_render_cache.row_counts[index],
                state.transcript_render_cache.entries[index]
                    .document
                    .lines
                    .len(),
            );
            assert_eq!(visible, transcript_lines(&state, theme, width));
            assert_eq!(visible_document_lines(&state, 20, 0).len(), visible.len());
            assert_eq!(
                cached_transcript_row_count(&mut state, theme, width),
                visible.len()
            );
        }
    }
}

fn test_attachment(id: &str, label: &str) -> UserImageAttachment {
    UserImageAttachment {
        id: id.into(),
        label: label.into(),
        mime: "image/png".into(),
        data_url: "data:image/png;base64,AAAA".into(),
    }
}

#[allow(dead_code)]
fn record(sequence: u64, event: TranscriptEvent) -> TranscriptRecord {
    TranscriptRecord {
        session_id: "s".into(),
        sequence,
        timestamp_ms: sequence as u128,
        context_branch_id: None,
        event,
    }
}

#[allow(dead_code)]
fn context_records() -> Vec<TranscriptRecord> {
    vec![
        record(
            1,
            TranscriptEvent::ContextNodeLifecycle {
                node_id: "root".into(),
                status: crate::context_tree::ContextNodeStatus::Inactive,
            },
        ),
        record(
            2,
            TranscriptEvent::ContextNodeCreated {
                node_id: "node-a".into(),
                parent_node_id: Some("root".into()),
                label: Some("Investigate".into()),
                purpose: Some("Trace context flow".into()),
                block_ref: None,
                source_ref: Some(crate::context_tree::ContextSourceRef {
                    source_kind: "summary".into(),
                    source_id: Some("sum-1".into()),
                }),
            },
        ),
        record(
            3,
            TranscriptEvent::ContextNodeLifecycle {
                node_id: "node-a".into(),
                status: crate::context_tree::ContextNodeStatus::Active,
            },
        ),
        record(
            4,
            TranscriptEvent::AssistantMessage {
                content: "Captured note for summary source".into(),
            },
        ),
        record(
            5,
            TranscriptEvent::ContextViewOperationMetadata {
                operation: "pin".into(),
                node_id: None,
                block_id: Some("block-seq-4-note".into()),
                detail: None,
            },
        ),
        record(
            6,
            TranscriptEvent::AssistantMessage {
                content: "Archived block note".into(),
            },
        ),
        record(
            7,
            TranscriptEvent::ContextViewOperationMetadata {
                operation: "archive".into(),
                node_id: None,
                block_id: Some("block-seq-6-note".into()),
                detail: None,
            },
        ),
        record(
            8,
            TranscriptEvent::ToolCallStarted {
                call_id: "call-1".into(),
                name: "shell__exec".into(),
                args: json!({"command": "cargo test --bin letcode"}),
            },
        ),
        record(
            9,
            TranscriptEvent::ToolCallFinished {
                call_id: "call-1".into(),
                name: "shell__exec".into(),
                ok: true,
                output: ToolResult::ok(
                    "shell__exec",
                    json!({
                        "stdout": "x".repeat(5_000),
                        "stdout_truncated": false,
                        "stderr": "",
                        "stderr_truncated": false
                    }),
                ),
            },
        ),
        record(
            10,
            TranscriptEvent::ContextSummaryArtifactMetadata {
                node_id: "node-a".into(),
                artifact_id: "sum-1".into(),
                artifact_kind: "session_summary".into(),
                version: Some(1),
                summary: Some("Short retained summary".into()),
                source_node_id: Some("node-a".into()),
                source_block_id: Some("block-seq-4-note".into()),
                source_start_sequence: Some(4),
                source_end_sequence: Some(4),
            },
        ),
    ]
}

#[test]
fn tool_permission_and_error_cards_wrap_to_target_width() {
    let mut state = TuiState::default();
    state.apply_event(SessionEvent::UserMessage(UserMessageEvent::new("seed")));

    let mut tool_started = ToolStartedEvent::new("call-tool", "shell__exec", "run");
    tool_started.arguments = Some("--really-long-arg ".repeat(20));
    state.apply_event(SessionEvent::ToolStarted(tool_started));
    let mut tool_finished =
        ToolFinishedEvent::new("call-tool", "shell__exec", "run", ToolOutcome::Failure);
    tool_finished.output = Some("output=".to_string() + &"x".repeat(200));
    state.apply_event(SessionEvent::ToolFinished(tool_finished));

    let mut request = PermissionRequestEvent::new("call-perm", "shell__exec", "needs approval");
    request.arguments = Some("arg ".repeat(60));
    request.rationale = Some("because ".repeat(80));
    state.apply_event(SessionEvent::PermissionRequested(request));

    let mut err = ErrorEvent::new("boom");
    err.details = Some("detail ".repeat(90));
    state.apply_event(SessionEvent::Error(err));

    let theme = Theme::dark();
    let width = 44usize;
    let lines = transcript_lines(&state, theme, width);

    // Ensure no generated line exceeds target width in display cells.
    for line in &lines {
        let w = crate::tui::measure::display_width(&line.to_string());
        assert!(
            w <= width,
            "line width {w} > {width}: {:?}",
            line.to_string()
        );
    }
}

#[test]
fn auto_review_renders_lock_status_and_expanded_rationale() {
    let item = TimelineItem::AutoReview(crate::tui::timeline::AutoReviewDecisionView {
        call_id: "call-deny".into(),
        tool_name: "shell__exec".into(),
        approval: "deny".into(),
        risk: Some("high".into()),
        rationale: "unsafe command".into(),
        allowed: false,
    });
    let theme = Theme::dark();

    let collapsed = render_timeline_item_document(
        &item,
        theme,
        80,
        0,
        false,
        false,
        None,
        crate::command::ThoughtsDisplayMode::Full,
        false,
        crate::command::ToolsDisplayMode::Detailed,
        false,
        tool_card::ToolGroupStats::default(),
        None,
        std::time::Instant::now(),
        theme.canvas(),
        &crate::tui::i18n::Translator::new(crate::tui::i18n::Language::En),
        None,
    );
    let expanded = render_timeline_item_document(
        &item,
        theme,
        80,
        0,
        true,
        false,
        None,
        crate::command::ThoughtsDisplayMode::Full,
        false,
        crate::command::ToolsDisplayMode::Detailed,
        false,
        tool_card::ToolGroupStats::default(),
        None,
        std::time::Instant::now(),
        theme.canvas(),
        &crate::tui::i18n::Translator::new(crate::tui::i18n::Language::En),
        None,
    );
    let collapsed_text = collapsed
        .lines
        .iter()
        .flat_map(|line| line.spans.iter())
        .map(|span| span.text.as_str())
        .collect::<String>();
    let expanded_text = expanded
        .lines
        .iter()
        .flat_map(|line| line.spans.iter())
        .map(|span| span.text.as_str())
        .collect::<String>();

    assert!(collapsed_text.contains("⛨ Auto-review denied shell__exec · deny · expand"));
    assert!(!collapsed_text.contains("unsafe command"));
    assert!(expanded_text.contains("⛨ Auto-review denied shell__exec · deny · collapse"));
    assert!(expanded_text.contains("risk high · unsafe command · call call-deny"));
    for document in [&collapsed, &expanded] {
        assert!(document.lines.iter().all(|line| {
            let text = line
                .spans
                .iter()
                .map(|span| span.text.as_str())
                .collect::<String>();
            text.starts_with("  ") && !text.starts_with("   ")
        }));
    }
    assert!(expanded.validate());

    let narrow = render_timeline_item_document(
        &item,
        theme,
        24,
        0,
        true,
        false,
        None,
        crate::command::ThoughtsDisplayMode::Full,
        false,
        crate::command::ToolsDisplayMode::Detailed,
        false,
        tool_card::ToolGroupStats::default(),
        None,
        std::time::Instant::now(),
        theme.canvas(),
        &crate::tui::i18n::Translator::new(crate::tui::i18n::Language::En),
        None,
    );
    assert!(narrow.validate());
    assert!(
        narrow.lines.iter().all(|line| {
            crate::tui::measure::display_width(
                &line
                    .spans
                    .iter()
                    .map(|span| span.text.as_str())
                    .collect::<String>(),
            ) == 24
        }),
        "{narrow:?}"
    );
}

#[test]
fn transcript_cards_use_right_pad_without_narrowing_ordinary_assistant() {
    let theme = Theme::dark();
    let width = 32usize;
    let long_text = "x".repeat(width);
    let cards = [
        TimelineItem::Tool(ToolView {
            call_id: "call-tool".into(),
            name: "shell__exec".into(),
            summary: "run command".into(),
            arguments: Some("arg".into()),
            output: Some("output".into()),
            status: ToolExecutionStatus::Failed,
        }),
        TimelineItem::Todo(TodoView {
            items: vec![TodoItem {
                id: "todo-1".into(),
                content: long_text.clone(),
                status: TodoStatus::Pending,
            }],
            auto_continue: AutoContinueState::default(),
        }),
        TimelineItem::Permission(PermissionView {
            call_id: "call-perm".into(),
            tool_name: "shell__exec".into(),
            summary: long_text.clone(),
            arguments: None,
            rationale: Some("because".into()),
            origin_label: None,
            can_allow_always: false,
            grant_summary: None,
            status: PermissionPromptStatus::Approved,
            resolution_reason: Some("approved".into()),
        }),
        TimelineItem::Error(ErrorView {
            message: long_text.clone(),
            details: Some("details".into()),
        }),
        TimelineItem::Compaction(CompactionView {
            summary: long_text.clone(),
            streaming: false,
        }),
        TimelineItem::Assistant(MessageView {
            id: None,
            submission_id: None,
            role: MessageRole::Assistant,
            text: json!({
                "status": "completed",
                "summary": long_text.clone(),
                "findings": ["finding"],
            })
            .to_string(),
            attachments: Vec::new(),
            selected_skills: Vec::new(),
            streaming: false,
            queued: false,
            reveal_tail: std::collections::VecDeque::new(),
        }),
    ];

    for item in cards {
        let lines = ratatui::text::Text::from(crate::tui::transcript_ratatui::document_to_ratatui(
            &render_timeline_item_document(
                &item,
                theme,
                width,
                0,
                false,
                false,
                None,
                crate::command::ThoughtsDisplayMode::Full,
                false,
                crate::command::ToolsDisplayMode::Detailed,
                false,
                tool_card::ToolGroupStats::default(),
                None,
                std::time::Instant::now(),
                theme.canvas(),
                &crate::tui::i18n::Translator::new(crate::tui::i18n::Language::En),
                None,
            ),
        ));
        assert!(
            lines
                .lines
                .iter()
                .all(|line| { crate::tui::measure::display_width(&line.to_string()) == width }),
            "card did not fill width {width}: {lines:?}"
        );
    }

    let ordinary = TimelineItem::Assistant(MessageView {
        id: None,
        submission_id: None,
        role: MessageRole::Assistant,
        text: long_text,
        attachments: Vec::new(),
        selected_skills: Vec::new(),
        streaming: false,
        queued: false,
        reveal_tail: std::collections::VecDeque::new(),
    });
    let ordinary_lines =
        crate::tui::transcript_ratatui::document_to_ratatui(&render_timeline_item_document(
            &ordinary,
            theme,
            width,
            0,
            false,
            false,
            None,
            crate::command::ThoughtsDisplayMode::Full,
            false,
            crate::command::ToolsDisplayMode::Detailed,
            false,
            tool_card::ToolGroupStats::default(),
            None,
            std::time::Instant::now(),
            theme.canvas(),
            &crate::tui::i18n::Translator::new(crate::tui::i18n::Language::En),
            None,
        ));
    assert!(
        ordinary_lines
            .iter()
            .any(|line| { crate::tui::measure::display_width(&line.to_string()) == width })
    );

    let review_request = TimelineItem::User(MessageView {
        id: None,
        submission_id: None,
        role: MessageRole::User,
        text: "Approve or deny this tool permission request.\nTool: shell__exec\nClass: tool\nSummary: run command\ncan_allow_always: false".into(),
        attachments: Vec::new(),
        selected_skills: Vec::new(),
        streaming: false,
        queued: false,
        reveal_tail: std::collections::VecDeque::new(),
    });
    let review_decision = TimelineItem::Assistant(MessageView {
        id: None,
        submission_id: None,
        role: MessageRole::Assistant,
        text: json!({"decision": "deny", "risk": "high", "rationale": "unsafe"}).to_string(),
        attachments: Vec::new(),
        selected_skills: Vec::new(),
        streaming: false,
        queued: false,
        reveal_tail: std::collections::VecDeque::new(),
    });
    for item in [&review_request, &review_decision] {
        let document =
            try_render_reviewer_view_item(item, theme, width).expect("reviewer card should parse");
        assert!(
            crate::tui::transcript_ratatui::document_to_ratatui(&document)
                .iter()
                .all(|line| crate::tui::measure::display_width(&line.to_string()) == width)
        );
    }
}

#[test]
fn error_card_stays_within_narrow_widths() {
    let mut state = TuiState::default();
    let mut error = ErrorEvent::new("stream stopped");
    error.details = Some("retry after backoff".into());
    state.apply_event(SessionEvent::Error(error));

    for width in 1..=4 {
        for line in transcript_lines(&state, Theme::dark(), width) {
            assert!(
                crate::tui::measure::display_width(&line.to_string()) <= width,
                "error card overflowed width {width}: {:?}",
                line.to_string()
            );
        }
    }
}

#[test]
fn committed_compaction_renders_durable_block_with_summary() {
    let mut state = TuiState::default();
    state.timeline.push_restored_compaction(
        "## Goal\n\nEarlier context was summarized here.\n\n- keep path `src/agent.rs`",
    );

    let lines = transcript_lines(&state, Theme::dark(), 80)
        .into_iter()
        .map(|line| line.to_string())
        .collect::<Vec<_>>();
    let nonempty_lines = lines
        .iter()
        .filter(|line| !line.is_empty())
        .map(String::as_str)
        .collect::<Vec<_>>();

    let rule = "─".repeat(80);
    assert_eq!(
        nonempty_lines.iter().filter(|line| **line == rule).count(),
        2,
        "expected top and bottom drawn rules: {lines:?}"
    );
    assert!(
        nonempty_lines
            .iter()
            .any(|line| line.contains("Earlier context was summarized")),
        "{lines:?}"
    );
    // Markdown body should render (not raw ## markers as the only content).
    assert!(
        nonempty_lines.iter().any(|line| line.contains("Goal")),
        "expected markdown heading content: {lines:?}"
    );
}

#[test]
fn width_reflow_clears_render_line_selection_anchors() {
    let mut state = TuiState::default();
    state.apply_event(SessionEvent::AssistantDelta(AssistantDeltaEvent::new(
        "a line that wraps when the transcript becomes narrower",
    )));
    cached_transcript_row_count(&mut state, Theme::dark(), 80);
    state.text_selection = Some(crate::tui::state::TextSelection {
        start: crate::tui::state::SelectionAnchor {
            item_index: 0,
            rendered_line_offset: 0,
            char_offset: 0,
        },
        end: crate::tui::state::SelectionAnchor {
            item_index: 0,
            rendered_line_offset: 0,
            char_offset: 4,
        },
    });
    state.selection_in_progress = true;

    cached_transcript_row_count(&mut state, Theme::dark(), 40);

    assert!(state.text_selection.is_none());
    assert!(!state.selection_in_progress);
}

#[test]
fn visible_window_clips_transcript_rows_before_rendering() {
    let lines = (0..20)
        .map(|index| ratatui::text::Line::from(format!("row-{index}")))
        .collect::<Vec<_>>();

    let visible = visible_transcript_lines(&lines, 5, 12)
        .into_iter()
        .map(|line| line.to_string())
        .collect::<Vec<_>>();

    assert_eq!(
        visible,
        vec!["row-12", "row-13", "row-14", "row-15", "row-16"]
    );

    let bottom = visible_transcript_lines(&lines, 5, 18)
        .into_iter()
        .map(|line| line.to_string())
        .collect::<Vec<_>>();
    assert_eq!(bottom, vec!["row-18", "row-19"]);
}

#[test]
fn visible_window_supports_large_document_offsets() {
    let start = 100_000usize;
    let lines = (0..start + 10)
        .map(|index| ratatui::text::Line::from(format!("row-{index}")))
        .collect::<Vec<_>>();

    let visible = visible_transcript_lines(&lines, 4, start)
        .into_iter()
        .map(|line| line.to_string())
        .collect::<Vec<_>>();

    assert_eq!(
        visible,
        vec![
            format!("row-{start}"),
            format!("row-{}", start + 1),
            format!("row-{}", start + 2),
            format!("row-{}", start + 3),
        ]
    );
}

#[test]
fn compact_reasoning_cache_refreshes_when_adjacent_item_is_added() {
    let start = std::time::Instant::now() - std::time::Duration::from_secs(2);
    let mut state = TuiState::default();
    state.set_thoughts_display(crate::command::ThoughtsDisplayMode::Compact);
    state.apply_event(SessionEvent::ReasoningDelta(ReasoningDeltaEvent::at(
        "reasoning-1",
        "First title\nFirst body",
        start,
    )));
    state.apply_event(SessionEvent::ReasoningDone(ReasoningDoneEvent::at(
        "reasoning-1",
        "First title\nFirst body",
        start + std::time::Duration::from_millis(400),
    )));

    let theme = Theme::dark();
    let width = 80;
    cached_transcript_row_count(&mut state, theme, width);
    let before = visible_cached_transcript_lines(&mut state, theme, width, 8, 0)
        .into_iter()
        .map(|line| line.to_string())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(before.contains("First title"), "{before}");

    state.apply_event(SessionEvent::ReasoningDelta(ReasoningDeltaEvent::at(
        "reasoning-2",
        "Second title\nSecond body",
        start + std::time::Duration::from_millis(500),
    )));
    cached_transcript_row_count(&mut state, theme, width);
    let after = visible_cached_transcript_lines(&mut state, theme, width, 8, 0)
        .into_iter()
        .map(|line| line.to_string())
        .collect::<Vec<_>>()
        .join("\n");

    assert!(!after.contains("First title"), "{after}");
    assert!(after.contains("Second title"), "{after}");
}

#[test]
fn compact_reasoning_accumulates_adjacent_thinking_durations() {
    let start = std::time::Instant::now() - std::time::Duration::from_secs(3);
    let mut state = TuiState::default();
    state.set_thoughts_display(crate::command::ThoughtsDisplayMode::Compact);
    state.apply_event(SessionEvent::ReasoningDelta(ReasoningDeltaEvent::at(
        "reasoning-1",
        "First title",
        start,
    )));
    state.apply_event(SessionEvent::ReasoningDone(ReasoningDoneEvent::at(
        "reasoning-1",
        "First title",
        start + std::time::Duration::from_millis(600),
    )));
    state.apply_event(SessionEvent::ReasoningDelta(ReasoningDeltaEvent::at(
        "reasoning-2",
        "Second title",
        start + std::time::Duration::from_millis(700),
    )));
    state.apply_event(SessionEvent::ReasoningDone(ReasoningDoneEvent::at(
        "reasoning-2",
        "Second title",
        start + std::time::Duration::from_millis(1_300),
    )));

    let rendered = transcript_lines(&state, Theme::dark(), 80)
        .into_iter()
        .map(|line| line.to_string())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(rendered.contains("Thought · 1s"), "{rendered}");
    assert!(!rendered.contains("First title"), "{rendered}");
    assert!(rendered.contains("Second title"), "{rendered}");
}

#[test]
fn compact_reasoning_title_replaces_one_grapheme_per_frame() {
    let start = std::time::Instant::now();
    let reasoning = ReasoningView {
        item_id: "reasoning-2".into(),
        text: "新标题".into(),
        streaming: true,
        started_at: Some(start),
        duration_ms: None,
        transition_from: Some("旧文案".into()),
        transition_started_at: Some(start),
    };

    assert_eq!(
        animated_compact_reasoning_title_at(&reasoning, "新标题", 40, start),
        "旧文案"
    );
    assert_eq!(
        animated_compact_reasoning_title_at(
            &reasoning,
            "新标题",
            40,
            start + std::time::Duration::from_millis(20),
        ),
        "新文案"
    );
    assert_eq!(
        animated_compact_reasoning_title_at(
            &reasoning,
            "新标题",
            40,
            start + std::time::Duration::from_millis(40),
        ),
        "新标案"
    );
    assert_eq!(
        animated_compact_reasoning_title_at(
            &reasoning,
            "新标题",
            40,
            start + std::time::Duration::from_millis(60),
        ),
        "新标题"
    );
}

#[test]
fn compact_tools_render_fixed_header_above_latest_call() {
    let mut state = TuiState::default();
    state.set_language(Some(crate::tui::i18n::Language::En));
    state.set_tools_display(crate::command::ToolsDisplayMode::Compact);
    let mut first = ToolStartedEvent::new("call-1", "fs__read", "read first");
    first.arguments = Some(json!({"path": "first.rs"}).to_string());
    state.apply_event(SessionEvent::ToolStarted(first));
    state.apply_event(SessionEvent::ToolFinished(ToolFinishedEvent {
        call_id: "call-1".into(),
        name: "fs__read".into(),
        summary: "read first".into(),
        outcome: ToolOutcome::Failure,
        output: None,
    }));
    let mut second = ToolStartedEvent::new("call-2", "fs__read", "read second");
    second.arguments = Some(json!({"path": "second.rs"}).to_string());
    state.apply_event(SessionEvent::ToolStarted(second));

    let compact = transcript_lines(&state, Theme::dark(), 80)
        .into_iter()
        .map(|line| line.to_string())
        .collect::<Vec<_>>();
    let header_index = compact
        .iter()
        .position(|line| line.contains("Tools · 2"))
        .expect("compact tools header");
    assert!(compact[header_index].contains("1 failed"), "{compact:?}");
    assert!(
        compact
            .get(header_index + 1)
            .is_some_and(|line| line.contains("second.rs")),
        "{compact:?}"
    );
    assert!(!compact.iter().any(|line| line.contains("first.rs")));

    state.set_language(Some(crate::tui::i18n::Language::ZhCn));
    let localized = transcript_lines(&state, Theme::dark(), 80)
        .into_iter()
        .map(|line| line.to_string())
        .collect::<Vec<_>>();
    assert!(localized.iter().any(|line| line.contains("Tools · 2")));
    assert!(!localized.iter().any(|line| line.contains("工具 · 2")));

    state.set_tools_display(crate::command::ToolsDisplayMode::Detailed);
    let detailed = transcript_lines(&state, Theme::dark(), 80)
        .into_iter()
        .map(|line| line.to_string())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(detailed.contains("first.rs"), "{detailed}");
    assert!(detailed.contains("second.rs"), "{detailed}");
}

#[test]
fn compact_tool_cache_refreshes_when_an_earlier_call_finishes() {
    let mut state = TuiState::default();
    state.set_language(Some(crate::tui::i18n::Language::En));
    state.set_tools_display(crate::command::ToolsDisplayMode::Compact);
    state.apply_event(SessionEvent::ToolStarted(ToolStartedEvent::new(
        "call-1",
        "fs__read",
        "read first",
    )));
    state.apply_event(SessionEvent::ToolStarted(ToolStartedEvent::new(
        "call-2",
        "fs__read",
        "read second",
    )));
    state.apply_event(SessionEvent::ToolFinished(ToolFinishedEvent {
        call_id: "call-2".into(),
        name: "fs__read".into(),
        summary: "read second".into(),
        outcome: ToolOutcome::Success,
        output: None,
    }));

    cached_transcript_row_count_with_reflow(&mut state, Theme::dark(), 80);
    let before = cached_item_lines(&mut state, 1, Theme::dark(), 80)
        .into_iter()
        .map(|line| line.to_string())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(!before.contains("failed"), "{before}");

    state.apply_event(SessionEvent::ToolFinished(ToolFinishedEvent {
        call_id: "call-1".into(),
        name: "fs__read".into(),
        summary: "read first".into(),
        outcome: ToolOutcome::Failure,
        output: None,
    }));
    cached_transcript_row_count_with_reflow(&mut state, Theme::dark(), 80);
    let after = cached_item_lines(&mut state, 1, Theme::dark(), 80)
        .into_iter()
        .map(|line| line.to_string())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(after.contains("1 failed"), "{after}");
}

#[test]
fn compact_tools_render_background_subagents_as_detailed_cards() {
    let mut state = TuiState::default();
    state.set_language(Some(crate::tui::i18n::Language::En));
    state.set_tools_display(crate::command::ToolsDisplayMode::Compact);
    state.apply_event(SessionEvent::ToolStarted(ToolStartedEvent::new(
        "call-1",
        "agent__explore",
        "inspect repository",
    )));
    state.apply_event(SessionEvent::ToolFinished(ToolFinishedEvent {
        call_id: "call-1".into(),
        name: "agent__explore".into(),
        summary: "background run started".into(),
        outcome: ToolOutcome::Success,
        output: Some(
            json!({
                "data": {
                    "run_id": "run-1",
                    "child_session_id": "child-1",
                    "agent_name": "explorer",
                    "status": "running",
                    "background": true,
                    "active": true
                }
            })
            .to_string(),
        ),
    }));

    cached_transcript_row_count_with_reflow(&mut state, Theme::dark(), 80);
    let active = cached_item_lines(&mut state, 0, Theme::dark(), 80)
        .into_iter()
        .map(|line| line.to_string())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        active.contains("running [background/active] explorer"),
        "{active}"
    );
    assert!(active.contains("/child child-1"), "{active}");
    assert!(!active.contains("Tools ·"), "{active}");

    let compact_active = transcript_lines(&state, Theme::dark(), 80)
        .into_iter()
        .map(|line| line.to_string())
        .collect::<Vec<_>>()
        .join("\n");
    state.set_tools_display(crate::command::ToolsDisplayMode::Detailed);
    let detailed_active = transcript_lines(&state, Theme::dark(), 80)
        .into_iter()
        .map(|line| line.to_string())
        .collect::<Vec<_>>()
        .join("\n");
    assert_eq!(compact_active, detailed_active);
    state.set_tools_display(crate::command::ToolsDisplayMode::Compact);

    state.apply_event(SessionEvent::ToolFinished(ToolFinishedEvent {
        call_id: "call-1".into(),
        name: "agent__explore".into(),
        summary: "background run completed".into(),
        outcome: ToolOutcome::Success,
        output: Some(
            json!({
                "data": {
                    "run_id": "run-1",
                    "child_session_id": "child-1",
                    "agent_name": "explorer",
                    "status": "completed",
                    "background": true,
                    "active": false
                }
            })
            .to_string(),
        ),
    }));
    cached_transcript_row_count_with_reflow(&mut state, Theme::dark(), 80);
    let completed = cached_item_lines(&mut state, 0, Theme::dark(), 80)
        .into_iter()
        .map(|line| line.to_string())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        completed.contains("completed [background] explorer"),
        "{completed}"
    );
    assert!(!completed.contains("Tools ·"), "{completed}");
}

#[test]
fn compact_tools_keep_subagent_cards_between_ordinary_tool_groups() {
    let mut state = TuiState::default();
    state.set_language(Some(crate::tui::i18n::Language::En));
    state.set_tools_display(crate::command::ToolsDisplayMode::Compact);

    for (call_id, path) in [("call-1", "first.rs"), ("call-3", "second.rs")] {
        let mut started = ToolStartedEvent::new(call_id, "fs__read", "read file");
        started.arguments = Some(json!({"path": path}).to_string());
        state.apply_event(SessionEvent::ToolStarted(started));
        state.apply_event(SessionEvent::ToolFinished(ToolFinishedEvent {
            call_id: call_id.into(),
            name: "fs__read".into(),
            summary: "read file".into(),
            outcome: ToolOutcome::Success,
            output: None,
        }));
        if call_id == "call-1" {
            let mut subagent =
                ToolStartedEvent::new("call-2", "agent__explore", "inspect repository");
            subagent.arguments = Some(json!({"task": "inspect repository"}).to_string());
            state.apply_event(SessionEvent::ToolStarted(subagent));
            state.apply_event(SessionEvent::ToolFinished(ToolFinishedEvent {
                call_id: "call-2".into(),
                name: "agent__explore".into(),
                summary: "inspection completed".into(),
                outcome: ToolOutcome::Success,
                output: Some(
                    json!({
                        "data": {
                            "run_id": "run-1",
                            "child_session_id": "child-1",
                            "agent_name": "explorer",
                            "status": "completed",
                            "summary": "inspection completed",
                            "active": false
                        }
                    })
                    .to_string(),
                ),
            }));
        }
    }

    let compact = transcript_lines(&state, Theme::dark(), 80)
        .into_iter()
        .map(|line| line.to_string())
        .collect::<Vec<_>>()
        .join("\n");
    assert_eq!(compact.matches("Tools · 1").count(), 2, "{compact}");
    assert!(!compact.contains("Tools · 3"), "{compact}");
    assert!(compact.contains("first.rs"), "{compact}");
    assert!(
        compact.contains("completed explorer inspection completed"),
        "{compact}"
    );
    assert!(compact.contains("/child child-1"), "{compact}");
    assert!(compact.contains("second.rs"), "{compact}");
}

#[test]
fn compact_tools_render_subagent_control_calls_like_detailed_mode() {
    let mut state = TuiState::default();
    state.set_language(Some(crate::tui::i18n::Language::En));
    let mut started =
        ToolStartedEvent::new("call-status", "agent__status", "inspect subagent status");
    started.arguments = Some(json!({"run_id": "run-1"}).to_string());
    state.apply_event(SessionEvent::ToolStarted(started));
    state.apply_event(SessionEvent::ToolFinished(ToolFinishedEvent {
        call_id: "call-status".into(),
        name: "agent__status".into(),
        summary: "subagent completed".into(),
        outcome: ToolOutcome::Success,
        output: Some(json!({"data": {"run_id": "run-1", "active": false}}).to_string()),
    }));

    state.set_tools_display(crate::command::ToolsDisplayMode::Compact);
    let compact = transcript_lines(&state, Theme::dark(), 80)
        .into_iter()
        .map(|line| line.to_string())
        .collect::<Vec<_>>()
        .join("\n");
    state.set_tools_display(crate::command::ToolsDisplayMode::Detailed);
    let detailed = transcript_lines(&state, Theme::dark(), 80)
        .into_iter()
        .map(|line| line.to_string())
        .collect::<Vec<_>>()
        .join("\n");

    assert_eq!(compact, detailed);
    assert!(compact.contains("status completed"), "{compact}");
    assert!(!compact.contains("Tools ·"), "{compact}");
}

#[test]
fn compact_tools_render_cancelled_subagent_card_despite_stale_active_output() {
    let mut state = TuiState::default();
    state.set_language(Some(crate::tui::i18n::Language::En));
    state.set_tools_display(crate::command::ToolsDisplayMode::Compact);
    state.apply_event(SessionEvent::ToolStarted(ToolStartedEvent::new(
        "call-1",
        "agent__explore",
        "inspect repository",
    )));
    assert!(state.timeline.update_active_subagent_tool_live_summary(
        "child-1",
        Some("explorer"),
        Some("call-1"),
        "running",
        "background run started",
    ));
    state.apply_event(SessionEvent::ToolCancelled(ToolCancelledEvent::new(
        "call-1",
        "agent__explore",
    )));

    cached_transcript_row_count_with_reflow(&mut state, Theme::dark(), 80);
    let cancelled = cached_item_lines(&mut state, 0, Theme::dark(), 80)
        .into_iter()
        .map(|line| line.to_string())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(cancelled.contains("explorer"), "{cancelled}");
    assert!(!cancelled.contains("Tools ·"), "{cancelled}");

    let compact = transcript_lines(&state, Theme::dark(), 80)
        .into_iter()
        .map(|line| line.to_string())
        .collect::<Vec<_>>()
        .join("\n");
    state.set_tools_display(crate::command::ToolsDisplayMode::Detailed);
    let detailed = transcript_lines(&state, Theme::dark(), 80)
        .into_iter()
        .map(|line| line.to_string())
        .collect::<Vec<_>>()
        .join("\n");
    assert_eq!(compact, detailed);
}

#[test]
fn streaming_reasoning_status_wraps_at_narrow_width() {
    let mut state = TuiState::default();
    state.set_thoughts_display(crate::command::ThoughtsDisplayMode::Compact);
    state.apply_event(SessionEvent::ReasoningDelta(ReasoningDeltaEvent::new(
        "reasoning-1",
        "检查缓存状态",
    )));

    for width in [8, 12, 18] {
        let lines = transcript_lines(&state, Theme::dark(), width);
        assert!(
            lines
                .iter()
                .all(|line| display_width(&line.to_string()) <= width),
            "width {width}: {lines:?}"
        );
    }
}

#[test]
fn reasoning_display_modes_and_narrow_width_render_cleanly() {
    let start = std::time::Instant::now();
    let mut state = TuiState::default();
    state.apply_event(SessionEvent::ReasoningDelta(ReasoningDeltaEvent::at(
        "reasoning-1",
        "Inspecting cache\nRead the cached document and compare adjacent items.",
        start,
    )));
    state.apply_event(SessionEvent::ReasoningDone(ReasoningDoneEvent::at(
        "reasoning-1",
        "Inspecting cache\nRead the cached document and compare adjacent items.",
        start + std::time::Duration::from_millis(1_250),
    )));

    for mode in [
        crate::command::ThoughtsDisplayMode::Compact,
        crate::command::ThoughtsDisplayMode::Titles,
        crate::command::ThoughtsDisplayMode::Scroll,
        crate::command::ThoughtsDisplayMode::Full,
    ] {
        state.set_thoughts_display(mode);
        let lines = transcript_lines(&state, Theme::dark(), 18);
        assert!(
            lines
                .iter()
                .all(|line| display_width(&line.to_string()) <= 18)
        );
        let text = lines
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains("Inspecting cache"), "{mode:?}: {text}");
        assert_eq!(
            text.contains("Read the cached"),
            mode == crate::command::ThoughtsDisplayMode::Full,
            "{mode:?}: {text}"
        );
    }
}

#[test]
fn scroll_reasoning_shows_only_the_latest_three_lines() {
    let start = std::time::Instant::now();
    let text = "Title\nline one\nline two\nline three\nline four\nline five";
    let mut state = TuiState::default();
    state.set_thoughts_display(crate::command::ThoughtsDisplayMode::Scroll);
    state.apply_event(SessionEvent::ReasoningDelta(ReasoningDeltaEvent::at(
        "reasoning-1",
        text,
        start,
    )));
    state.apply_event(SessionEvent::ReasoningDone(ReasoningDoneEvent::at(
        "reasoning-1",
        text,
        start,
    )));

    let rendered = transcript_lines(&state, Theme::dark(), 80)
        .into_iter()
        .map(|line| line.to_string())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(rendered.contains("Title"));
    assert!(!rendered.contains("line one"));
    assert!(!rendered.contains("line two"));
    assert!(rendered.contains("line three"));
    assert!(rendered.contains("line four"));
    assert!(rendered.contains("line five"));
}

#[test]
fn consecutive_reasoning_titles_are_separated_outside_compact_mode() {
    let start = std::time::Instant::now() - std::time::Duration::from_secs(1);
    let mut state = TuiState::default();
    for (index, title) in ["First thought", "Second thought", "Third thought"]
        .into_iter()
        .enumerate()
    {
        let observed_at = start + std::time::Duration::from_millis(index as u64 * 100);
        let item_id = format!("reasoning-{index}");
        state.apply_event(SessionEvent::ReasoningDelta(ReasoningDeltaEvent::at(
            &item_id,
            title,
            observed_at,
        )));
        state.apply_event(SessionEvent::ReasoningDone(ReasoningDoneEvent::at(
            item_id,
            title,
            observed_at + std::time::Duration::from_millis(50),
        )));
    }

    for mode in [
        crate::command::ThoughtsDisplayMode::Titles,
        crate::command::ThoughtsDisplayMode::Full,
    ] {
        state.set_thoughts_display(mode);
        let lines = transcript_lines(&state, Theme::dark(), 80)
            .into_iter()
            .map(|line| line.to_string())
            .collect::<Vec<_>>();
        let title_rows = lines
            .iter()
            .enumerate()
            .filter_map(|(index, line)| line.contains("Thought:").then_some(index))
            .collect::<Vec<_>>();

        assert_eq!(title_rows.len(), 3, "{mode:?}: {lines:?}");
        assert!(
            title_rows
                .windows(2)
                .all(|rows| lines[rows[0] + 1..rows[1]].iter().any(String::is_empty)),
            "{mode:?}: {lines:?}"
        );

        let total_rows = cached_transcript_row_count(&mut state, Theme::dark(), 80);
        let cached = visible_cached_transcript_lines(
            &mut state,
            Theme::dark(),
            80,
            u16::try_from(total_rows).expect("test transcript fits in u16 rows"),
            0,
        )
        .into_iter()
        .map(|line| line.to_string())
        .collect::<Vec<_>>();
        assert_eq!(cached, lines, "{mode:?}");
    }

    state.set_thoughts_display(crate::command::ThoughtsDisplayMode::Compact);
    let compact = transcript_lines(&state, Theme::dark(), 80)
        .into_iter()
        .map(|line| line.to_string())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(!compact.contains("First thought"), "{compact}");
    assert!(!compact.contains("Second thought"), "{compact}");
    assert!(compact.contains("Third thought"), "{compact}");
}

#[test]
fn cached_and_document_visible_windows_share_geometry() {
    let mut state = TuiState::default();
    state.apply_event(SessionEvent::UserMessage(UserMessageEvent::new(
        "first item with enough text to wrap across rows",
    )));
    state.apply_event(SessionEvent::AssistantDelta(AssistantDeltaEvent::new(
        "second item with enough text to wrap across rows",
    )));
    state.apply_event(SessionEvent::AssistantDone { message_id: None });
    state.apply_event(SessionEvent::ToolStarted(ToolStartedEvent::new(
        "call-tail",
        "fs__list",
        "tail item",
    )));

    let theme = Theme::dark();
    let width = 18;
    let total_rows = cached_transcript_row_count(&mut state, theme, width);
    let item_rows = state.transcript_render_cache.row_counts.clone();
    assert!(item_rows.iter().any(|count| *count > 1));
    assert!(total_rows > surface::TRANSCRIPT_TOP_SPACER + 2);

    for (scroll, rows) in [
        (0, 2),
        (surface::TRANSCRIPT_TOP_SPACER, 3),
        (surface::TRANSCRIPT_TOP_SPACER + item_rows[0] + 1, 3),
        (total_rows.saturating_sub(3), 3),
        (total_rows + 100, 3),
    ] {
        let cached = visible_cached_transcript_lines(&mut state, theme, width, rows, scroll)
            .into_iter()
            .map(|line| line.to_string())
            .collect::<Vec<_>>();
        let documents = visible_document_lines(&state, rows, scroll)
            .into_iter()
            .map(|line| {
                line.map(|line| crate::tui::transcript_ratatui::line_to_ratatui(line).to_string())
                    .unwrap_or_default()
            })
            .collect::<Vec<_>>();
        assert_eq!(cached, documents, "scroll={scroll}, rows={rows}");
        assert!(cached.len() <= rows as usize);
    }
}

#[test]
fn cached_visible_window_reuses_geometry_after_width_reflow() {
    let mut state = TuiState::default();
    state.apply_event(SessionEvent::AssistantDelta(AssistantDeltaEvent::new(
        "a deliberately long transcript item that reflows at a narrower width",
    )));
    state.apply_event(SessionEvent::AssistantDone { message_id: None });
    state.apply_event(SessionEvent::UserMessage(UserMessageEvent::new(
        "a second item follows the reflowed item",
    )));

    let theme = Theme::dark();
    cached_transcript_row_count(&mut state, theme, 80);
    let (total_rows, _) = cached_transcript_row_count_with_reflow(&mut state, theme, 20);
    for scroll in [0, 2, total_rows.saturating_sub(4)] {
        let cached = visible_cached_transcript_lines(&mut state, theme, 20, 4, scroll)
            .into_iter()
            .map(|line| line.to_string())
            .collect::<Vec<_>>();
        let documents = visible_document_lines(&state, 4, scroll)
            .into_iter()
            .map(|line| {
                line.map(|line| crate::tui::transcript_ratatui::line_to_ratatui(line).to_string())
                    .unwrap_or_default()
            })
            .collect::<Vec<_>>();
        assert_eq!(cached, documents, "scroll={scroll}");
    }
}

#[test]
fn cached_visible_transcript_matches_full_transcript_window() {
    let mut state = TuiState::default();
    state.apply_event(SessionEvent::UserMessage(UserMessageEvent::new("seed")));
    for index in 0..30 {
        state
            .timeline
            .push_assistant_delta(AssistantDeltaEvent::new(format!("history line {index}")));
    }
    state.apply_event(SessionEvent::AssistantDelta(AssistantDeltaEvent::new(
        "# Heading\n```rust\nlet value = 42;\n```\n- done",
    )));
    state.apply_event(SessionEvent::AssistantDone { message_id: None });

    let theme = Theme::dark();
    let width = 72;
    let full = transcript_lines(&state, theme, width);
    let total_rows = cached_transcript_row_count(&mut state, theme, width);
    let visible = visible_cached_transcript_lines(&mut state, theme, width, 9, 12)
        .into_iter()
        .map(|line| line.to_string())
        .collect::<Vec<_>>();
    let expected = visible_transcript_lines(&full, 9, 12)
        .into_iter()
        .map(|line| line.to_string())
        .collect::<Vec<_>>();

    assert_eq!(total_rows, full.len());
    assert_eq!(visible, expected);
}

#[test]
fn cached_row_count_uses_stable_timeline_fast_path() {
    let mut state = TuiState::default();
    state.apply_event(SessionEvent::UserMessage(UserMessageEvent::new("first")));
    state.apply_event(SessionEvent::UserMessage(UserMessageEvent::new("second")));

    let theme = Theme::dark();
    let width = 80;
    let expected_rows = cached_transcript_row_count(&mut state, theme, width);
    let rebuilds = state.transcript_render_cache.row_count_rebuilds;

    assert_eq!(
        cached_transcript_row_count(&mut state, theme, width),
        expected_rows
    );
    assert_eq!(state.transcript_render_cache.row_count_rebuilds, rebuilds);
}

#[test]
fn cached_row_count_refreshes_after_timeline_mutation() {
    let mut state = TuiState::default();
    state.apply_event(SessionEvent::AssistantDelta(AssistantDeltaEvent::new(
        "first",
    )));

    let theme = Theme::dark();
    let width = 80;
    let before_rows = cached_transcript_row_count(&mut state, theme, width);
    let before_rebuilds = state.transcript_render_cache.row_count_rebuilds;

    state.apply_event(SessionEvent::AssistantDelta(AssistantDeltaEvent::new(
        " second",
    )));

    let after_rows = cached_transcript_row_count(&mut state, theme, width);
    assert_eq!(
        state.transcript_render_cache.row_count_rebuilds,
        before_rebuilds + 1
    );
    assert!(after_rows >= before_rows);
    assert_eq!(after_rows, transcript_lines(&state, theme, width).len());
}

/// Each frame re-parses only the block that can still change.
#[test]
fn streaming_markdown_cost_stays_within_one_full_render_per_frame() {
    let mut state = TuiState::default();
    let theme = Theme::dark();
    let width = 80;
    let chunk = "a streamed line with some words in it\n\n";
    let frames = 200;

    for _ in 0..frames {
        state.apply_event(SessionEvent::AssistantDelta(AssistantDeltaEvent::new(
            chunk,
        )));
        cached_transcript_row_count(&mut state, theme, width);
    }

    let answer_chars = state
        .active_timeline()
        .items()
        .iter()
        .find_map(|item| match item {
            TimelineItem::Assistant(message) => Some(message.text.len()),
            _ => None,
        })
        .expect("the streamed answer is a timeline item");
    let rendered_chars = state.transcript_render_cache.assistant_markdown_chars;

    assert_eq!(answer_chars, chunk.len() * frames);
    // One full pass over the answer, plus one blank line carried between frames.
    assert!(
        rendered_chars >= answer_chars && rendered_chars <= answer_chars * 2,
        "rendered {rendered_chars} characters for a {answer_chars} character answer"
    );
}

#[test]
fn transcript_cache_invalidates_streaming_assistant_item() {
    let mut state = TuiState::default();
    state.apply_event(SessionEvent::AssistantDelta(AssistantDeltaEvent::new(
        "first",
    )));

    let theme = Theme::dark();
    let width = 80;
    let before_rows = cached_transcript_row_count(&mut state, theme, width);
    let before_revision = state.transcript_render_cache.entries[0].revision;

    state.apply_event(SessionEvent::AssistantDelta(AssistantDeltaEvent::new(
        " second",
    )));

    let after_rows = cached_transcript_row_count(&mut state, theme, width);
    let after_revision = state.transcript_render_cache.entries[0].revision;
    let visible = visible_cached_transcript_lines(&mut state, theme, width, 8, 0)
        .into_iter()
        .map(|line| line.to_string())
        .collect::<Vec<_>>()
        .join("\n");

    assert_eq!(before_rows, after_rows);
    assert_ne!(before_revision, after_revision);
    assert!(visible.contains("first second"), "{visible}");
}

#[test]
fn transcript_cache_is_namespaced_by_timeline_replacement() {
    let mut state = TuiState::default();
    state
        .timeline
        .push_assistant_delta(AssistantDeltaEvent::new("old timeline"));
    let theme = Theme::dark();
    let width = 80;

    cached_transcript_row_count(&mut state, theme, width);
    assert!(
        visible_cached_transcript_lines(&mut state, theme, width, 8, 0)
            .into_iter()
            .any(|line| line.to_string().contains("old timeline"))
    );

    state.timeline = Timeline::new();
    state
        .timeline
        .push_assistant_delta(AssistantDeltaEvent::new("new timeline"));

    let visible = visible_cached_transcript_lines(&mut state, theme, width, 8, 0)
        .into_iter()
        .map(|line| line.to_string())
        .collect::<Vec<_>>()
        .join("\n");

    assert!(visible.contains("new timeline"), "{visible}");
    assert!(!visible.contains("old timeline"), "{visible}");
}

#[test]
fn manual_history_view_stays_anchored_when_streaming_rows_append() {
    let theme = Theme::dark();
    let width = 80;
    let viewport_rows = 5;
    let mut state = TuiState::default();

    for index in 0..24 {
        state
            .timeline
            .push_assistant_delta(AssistantDeltaEvent::new(format!("history line {index}")));
    }

    let before_lines = transcript_lines(&state, theme, width);
    state.sync_transcript_viewport_rows_with_reflow(before_lines.len(), false);
    let target_top = 6usize;
    let before_max_scroll = crate::tui::measure::max_scroll(before_lines.len(), viewport_rows);
    state.transcript_scroll = before_max_scroll.saturating_sub(target_top);
    state.auto_scroll = false;

    let before_top = crate::tui::measure::resolved_scroll_offset(
        before_lines.len(),
        viewport_rows,
        state.transcript_scroll,
        state.auto_scroll,
    );
    let before_visible = visible_transcript_lines(&before_lines, viewport_rows, before_top);
    let before_first = before_visible
        .first()
        .map(|line| line.to_string())
        .expect("visible row before append");

    state.apply_event(SessionEvent::AssistantDelta(AssistantDeltaEvent::new(
        "streaming row one\nstreaming row two\nstreaming row three",
    )));

    let after_lines = transcript_lines(&state, theme, width);
    state.sync_transcript_viewport_rows_with_reflow(after_lines.len(), false);
    let after_top = crate::tui::measure::resolved_scroll_offset(
        after_lines.len(),
        viewport_rows,
        state.transcript_scroll,
        state.auto_scroll,
    );
    let after_visible = visible_transcript_lines(&after_lines, viewport_rows, after_top);
    let after_first = after_visible
        .first()
        .map(|line| line.to_string())
        .expect("visible row after append");

    assert_eq!(after_top, before_top);
    assert_eq!(after_first, before_first);
    assert!(!state.auto_scroll);
}

#[test]
fn manual_history_offset_is_stable_across_width_reflow() {
    let theme = Theme::dark();
    let mut state = TuiState::default();
    let content = std::iter::repeat_n("x".repeat(90), 200)
        .collect::<Vec<_>>()
        .join("\n");
    state.apply_event(SessionEvent::AssistantDelta(AssistantDeltaEvent::new(
        content,
    )));
    state.apply_event(SessionEvent::AssistantDone { message_id: None });

    let (wide_rows, wide_reflowed) =
        cached_transcript_row_count_with_reflow(&mut state, theme, 120);
    assert!(!wide_reflowed);
    state.last_transcript_area.height = 10;
    state.sync_transcript_viewport_rows_with_reflow(wide_rows, wide_reflowed);
    state.scroll_transcript_up(25);

    let (narrow_rows, narrow_reflowed) =
        cached_transcript_row_count_with_reflow(&mut state, theme, 60);
    assert!(narrow_reflowed);
    assert!(narrow_rows > wide_rows);
    state.sync_transcript_viewport_rows_with_reflow(narrow_rows, narrow_reflowed);

    assert_eq!(state.transcript_scroll, 25);
    assert!(!state.auto_scroll);
}

#[test]
fn pending_permission_is_hidden_from_transcript_while_panel_is_active() {
    let mut state = TuiState::default();
    let mut request = PermissionRequestEvent::new("call-perm", "shell__exec", "cargo test all");
    request.arguments = Some("cargo test all".into());
    request.rationale = Some("tests need confirmation".into());
    state.apply_event(SessionEvent::PermissionRequested(request));

    let lines = transcript_lines(&state, Theme::dark(), 60)
        .into_iter()
        .map(|line| line.to_string())
        .collect::<Vec<_>>();

    assert!(
        !lines
            .iter()
            .any(|line| line.contains("Permission required")),
        "{lines:?}"
    );
    assert!(
        !lines.iter().any(|line| line.contains("cargo test all")),
        "{lines:?}"
    );
}

#[test]
fn running_subagent_status_follows_language_switch_and_refreshes_cache() {
    let mut state = TuiState::default();
    state.set_language(Some(crate::tui::i18n::Language::En));
    state.apply_event(SessionEvent::ToolStarted(ToolStartedEvent {
        call_id: "run-localized".into(),
        name: "agent__explore".into(),
        summary: "explorer running".into(),
        arguments: Some(serde_json::json!({"task":"inspect localization"}).to_string()),
    }));

    cached_transcript_row_count(&mut state, Theme::dark(), 100);
    let english = state.transcript_render_cache.entries()[0]
        .document
        .lines
        .iter()
        .flat_map(|line| line.spans.iter())
        .map(|span| span.text.as_str())
        .collect::<String>();
    assert!(english.contains("running"), "{english}");
    assert!(!english.contains("运行中"), "{english}");

    state.set_language(Some(crate::tui::i18n::Language::ZhCn));
    assert!(state.transcript_render_cache.is_empty());
    cached_transcript_row_count(&mut state, Theme::dark(), 100);
    let chinese = state.transcript_render_cache.entries()[0]
        .document
        .lines
        .iter()
        .flat_map(|line| line.spans.iter())
        .map(|span| span.text.as_str())
        .collect::<String>();
    assert!(chinese.contains("运行中"), "{chinese}");
}

#[test]
fn subagent_parent_transcript_stays_compact_and_keeps_child_details_out() {
    let mut state = TuiState::default();
    state.set_language(Some(crate::tui::i18n::Language::En));
    state.apply_event(SessionEvent::ToolStarted(ToolStartedEvent {
        call_id: "run-1".into(),
        name: "agent__explore".into(),
        summary: "explorer running · child-sessio".into(),
        arguments: Some(serde_json::json!({"task":"inspect runner flow"}).to_string()),
    }));
    state.apply_event(SessionEvent::ToolFinished(ToolFinishedEvent {
        call_id: "run-1".into(),
        name: "agent__explore".into(),
        summary: "explorer completed · child-sessio".into(),
        outcome: ToolOutcome::Success,
        output: Some(
            serde_json::json!({
                "ok": true,
                "tool": "agent__explore",
                "data": {
                    "agent_name": "Explorer",
                    "status": "completed",
                    "summary": "inspected runner flow and isolated parent timeline noise",
                    "full_summary": "inspected runner flow and isolated parent timeline noise\nlong child body line should stay in child view",
                    "child_session_id": "child-session-1234567890"
                }
            })
            .to_string(),
        ),
    }));

    let lines = transcript_lines(&state, Theme::dark(), 120)
        .into_iter()
        .map(|line| line.to_string())
        .collect::<Vec<_>>();

    let non_empty = lines
        .into_iter()
        .filter(|line| !line.trim().is_empty())
        .collect::<Vec<_>>();
    let agent_lines = non_empty
        .iter()
        .filter(|line| line.contains("Explorer") || line.contains("/child"))
        .collect::<Vec<_>>();

    assert_eq!(agent_lines.len(), 1, "{non_empty:?}");
    assert!(
        agent_lines[0].contains(
            "completed Explorer inspected runner flow and isolated parent timeline noise · /child"
        ),
        "{}",
        agent_lines[0]
    );
    assert!(
        !non_empty
            .iter()
            .any(|line| line.contains("long child body line should stay in child view")),
        "{non_empty:?}"
    );
}

#[test]
fn structured_subagent_tags_render_as_compact_card_at_narrow_width() {
    let mut state = TuiState::default();
    state.apply_event(SessionEvent::AssistantDelta(AssistantDeltaEvent::new(
        "<subagent_report><status>completed</status><summary>Updated transcript rendering</summary><finding>Structured results render as a card</finding><file_read>src/tui/components/transcript.rs</file_read><file_changed>src/tui/components/transcript.rs</file_changed><command_run>cargo test transcript</command_run><validation>focused tests passed</validation><next_step>review the card</next_step></subagent_report>",
    )));
    state.apply_event(SessionEvent::AssistantDone { message_id: None });

    let lines = transcript_lines(&state, Theme::dark(), 24)
        .into_iter()
        .map(|line| line.to_string())
        .collect::<Vec<_>>();
    let rendered = lines.join("\n");

    assert!(rendered.contains("Subagent"), "{rendered}");
    assert!(rendered.contains("completed"), "{rendered}");
    assert!(rendered.contains("Updated transcript"), "{rendered}");
    assert!(rendered.contains("read"), "{rendered}");
    assert!(rendered.contains("Findings"), "{rendered}");
    assert!(!rendered.contains("<status>"), "{rendered}");
    assert!(!rendered.contains("<subagent_report>"), "{rendered}");
    assert!(
        lines.iter().all(|line| line.chars().count() <= 24),
        "{lines:?}"
    );
}

#[test]
fn structured_subagent_json_accepts_object_summary_and_findings() {
    let mut state = TuiState::default();
    state.apply_event(SessionEvent::AssistantDelta(AssistantDeltaEvent::new(
        json!({
            "status": "completed",
            "summary": {"conclusion": "Reviewed the child result"},
            "findings": [{"evidence": "The child returned structured evidence"}],
            "files_read": [],
        })
        .to_string(),
    )));
    state.apply_event(SessionEvent::AssistantDone { message_id: None });

    let lines = transcript_lines(&state, Theme::dark(), 32)
        .into_iter()
        .map(|line| line.to_string())
        .collect::<Vec<_>>();
    let rendered = lines.join("\n");

    assert!(rendered.contains("Subagent"), "{rendered}");
    assert!(rendered.contains("completed"), "{rendered}");
    assert!(rendered.contains("Reviewed the child"), "{rendered}");
    assert!(rendered.contains("Findings"), "{rendered}");
    assert!(!rendered.contains("\"conclusion\""), "{rendered}");
    assert!(!rendered.contains('{'), "{rendered}");
    assert!(
        lines.iter().all(|line| line.chars().count() <= 32),
        "{lines:?}"
    );
}

#[test]
fn non_structured_assistant_json_uses_markdown_fallback() {
    let mut state = TuiState::default();
    state.apply_event(SessionEvent::AssistantDelta(AssistantDeltaEvent::new(
        r#"{"message":"ordinary JSON"}"#,
    )));
    state.apply_event(SessionEvent::AssistantDone { message_id: None });

    let rendered = transcript_lines(&state, Theme::dark(), 80)
        .into_iter()
        .map(|line| line.to_string())
        .collect::<Vec<_>>()
        .join("\n");

    assert!(rendered.contains("message"), "{rendered}");
    assert!(rendered.contains("ordinary JSON"), "{rendered}");
    assert!(!rendered.contains("Subagent"), "{rendered}");
}

#[test]
fn streaming_compaction_renders_rules_only_when_summary_text_arrives() {
    let mut state = TuiState::default();
    state.apply_event(SessionEvent::CompactionStarted);

    let rule = "─".repeat(80);
    let started = transcript_lines(&state, Theme::dark(), 80)
        .into_iter()
        .map(|line| line.to_string())
        .collect::<Vec<_>>();
    assert_eq!(started.iter().filter(|line| *line == &rule).count(), 0);
    assert!(!started.iter().any(|line| line.contains('…')));

    state.apply_event(SessionEvent::CompactionPreviewDelta {
        delta: "A transient summary preview".into(),
    });
    let streaming = transcript_lines(&state, Theme::dark(), 80)
        .into_iter()
        .map(|line| line.to_string())
        .collect::<Vec<_>>();
    let rule_index = streaming
        .iter()
        .position(|line| line == &rule)
        .expect("opening rule");
    let preview_index = streaming
        .iter()
        .position(|line| line.contains("A transient summary preview"))
        .expect("preview body");
    assert_eq!(streaming.iter().filter(|line| *line == &rule).count(), 1);
    assert!(rule_index < preview_index);
    assert!(!streaming.iter().any(|line| line.contains('…')));

    state.apply_event(SessionEvent::CompactionCommitted {
        summary: Some("A transient summary preview".into()),
    });
    let committed = transcript_lines(&state, Theme::dark(), 80)
        .into_iter()
        .map(|line| line.to_string())
        .collect::<Vec<_>>();
    assert_eq!(committed.iter().filter(|line| *line == &rule).count(), 2);
    let last_rule = committed
        .iter()
        .rposition(|line| line == &rule)
        .expect("closing rule");
    let body = committed
        .iter()
        .position(|line| line.contains("A transient summary preview"))
        .expect("committed body");
    assert!(body < last_rule);

    let narrow = transcript_lines(&state, Theme::dark(), 10)
        .into_iter()
        .map(|line| line.to_string())
        .collect::<Vec<_>>();
    assert!(narrow.iter().all(|line| line.chars().count() <= 10));
}

#[test]
fn assistant_reasoning_and_tool_trace_share_left_indent() {
    let mut state = TuiState::default();
    state.apply_event(SessionEvent::ReasoningDone(ReasoningDoneEvent::new(
        "r-1",
        "Answering\nThinking body",
    )));
    state.apply_event(SessionEvent::AssistantDelta(AssistantDeltaEvent::new(
        "# Title",
    )));
    state.apply_event(SessionEvent::AssistantDone { message_id: None });
    state.apply_event(SessionEvent::ToolStarted(ToolStartedEvent::new(
        "call-list",
        "fs__list",
        "List src",
    )));

    let lines = transcript_lines(&state, Theme::dark(), 80)
        .into_iter()
        .map(|line| line.to_string())
        .collect::<Vec<_>>();
    let reasoning = lines
        .iter()
        .find(|line| line.contains("Thought: Answering"))
        .expect("reasoning line renders");
    let markdown = lines
        .iter()
        .find(|line| line.contains("▌ Title"))
        .expect("assistant markdown line renders");
    let tool = lines
        .iter()
        .find(|line| line.contains("List src"))
        .expect("tool trace line renders");

    for line in [reasoning, markdown, tool] {
        assert!(line.starts_with("  "), "{line:?}");
        assert!(!line.starts_with("   "), "{line:?}");
    }
}

#[test]
fn user_card_bar_is_separated_from_card_background() {
    let mut state = TuiState::default();
    state.apply_event(SessionEvent::UserMessage(UserMessageEvent::new("hello")));

    let backend = TestBackend::new(40, 8);
    let mut terminal = Terminal::new(backend).expect("terminal");
    terminal
        .draw(|frame| {
            let area = Rect::new(0, 0, 40, 8);
            render_transcript(frame, &mut state, area, Theme::dark());
        })
        .expect("draw");

    let buffer = terminal.backend().buffer();
    let bar_cell = &buffer[(0, 2)];
    let card_cell = &buffer[(1, 2)];

    assert_eq!(bar_cell.bg, Theme::dark().root_bg);
    assert_eq!(card_cell.bg, Theme::dark().element_bg);
}

#[test]
fn wireframe_user_card_has_a_continuous_box() {
    let mut state = TuiState::default();
    state.apply_event(SessionEvent::UserMessage(UserMessageEvent::new("hello")));

    let lines = transcript_lines(&state, Theme::wireframe(), 40)
        .into_iter()
        .map(|line| line.to_string())
        .collect::<Vec<_>>();

    let top = lines
        .iter()
        .position(|line| line.starts_with('┌') && line.ends_with('┐'))
        .expect("top border");
    let content = lines
        .iter()
        .position(|line| line.contains("hello"))
        .expect("message content");
    let bottom = lines
        .iter()
        .position(|line| line.starts_with('└') && line.ends_with('┘'))
        .expect("bottom border");

    assert_eq!(content - top, 2, "{lines:?}");
    assert_eq!(bottom - content, 2, "{lines:?}");
    assert!(lines[content].starts_with('│') && lines[content].ends_with('│'));
}

#[test]
fn queued_user_message_renders_badge() {
    let mut state = TuiState::default();
    state.apply_event(SessionEvent::UserMessage(UserMessageEvent::queued(
        "follow up",
    )));

    let lines = transcript_lines(&state, Theme::dark(), 40)
        .into_iter()
        .map(|line| line.to_string())
        .collect::<Vec<_>>();

    assert!(
        lines.iter().any(|line| line.contains("QUEUED")),
        "{lines:?}"
    );
}

#[test]
fn user_message_renders_inline_image_placeholder_and_original_attachment_row() {
    let mut state = TuiState::default();
    state.apply_event(SessionEvent::UserMessage(
        UserMessageEvent::from_submission(UserMessageSubmission::new(
            "user-with-inline-image",
            UserMessageContent::from_parts(vec![
                crate::user_content::UserMessagePart::Text {
                    text: "[Image 1]".into(),
                },
                crate::user_content::UserMessagePart::Image {
                    attachment: test_attachment("img-1", "clipboard"),
                },
                crate::user_content::UserMessagePart::Text {
                    text: " 测试消息".into(),
                },
            ]),
        )),
    ));

    let lines = transcript_lines(&state, Theme::dark(), 60)
        .into_iter()
        .map(|line| line.to_string())
        .collect::<Vec<_>>();

    let body_index = lines
        .iter()
        .position(|line| line.contains("[Image 1] 测试消息"))
        .expect("inline image placeholder in message body");
    let attachment_index = lines
        .iter()
        .position(|line| line.contains("ATTACHMENTS"))
        .expect("attachment badge line");

    assert!(body_index < attachment_index, "{lines:?}");
    assert!(
        lines
            .iter()
            .any(|line| line.contains("image 1") && line.contains("clipboard")),
        "{lines:?}"
    );
}

#[test]
fn user_message_renders_selected_skills_beneath_body() {
    let mut state = TuiState::default();
    state.apply_event(SessionEvent::UserMessage(
        UserMessageEvent::from_submission(UserMessageSubmission::new(
            "user-with-skill",
            UserMessageContent::from("use this skill")
                .with_selected_skills(vec!["rust-audit".into()]),
        )),
    ));

    let lines = transcript_lines(&state, Theme::dark(), 60)
        .into_iter()
        .map(|line| line.to_string())
        .collect::<Vec<_>>();

    let body_index = lines
        .iter()
        .position(|line| line.contains("use this skill"))
        .expect("message body line");
    let skills_index = lines
        .iter()
        .position(|line| line.contains("SKILLS"))
        .expect("skills badge line");

    assert!(skills_index > body_index, "{lines:?}");
    assert!(
        lines
            .iter()
            .any(|line| line.contains("skill 1") && line.contains("rust-audit")),
        "{lines:?}"
    );
}

#[test]
fn user_message_renders_attachment_placeholders_beneath_body() {
    let mut state = TuiState::default();
    state.apply_event(SessionEvent::UserMessage(
        UserMessageEvent::from_submission(UserMessageSubmission::new(
            "user-with-images",
            UserMessageContent::new(
                "describe this",
                vec![
                    test_attachment("img-1", "clipboard"),
                    test_attachment("img-2", "diagram.png"),
                ],
            ),
        )),
    ));

    let lines = transcript_lines(&state, Theme::dark(), 60)
        .into_iter()
        .map(|line| line.to_string())
        .collect::<Vec<_>>();

    let body_index = lines
        .iter()
        .position(|line| line.contains("describe this"))
        .expect("message body line");
    let attachment_index = lines
        .iter()
        .position(|line| line.contains("ATTACHMENTS"))
        .expect("attachment badge line");
    let queued_index = lines.iter().position(|line| line.contains("QUEUED"));

    assert!(attachment_index > body_index, "{lines:?}");
    assert!(queued_index.is_none(), "{lines:?}");
    assert!(
        lines
            .iter()
            .any(|line| line.contains("image 1") && line.contains("clipboard")),
        "{lines:?}"
    );
    assert!(
        lines
            .iter()
            .any(|line| line.contains("image 2") && line.contains("diagram.png")),
        "{lines:?}"
    );
}

#[test]
fn wireframe_todo_card_has_a_continuous_box() {
    let mut state = TuiState::default();
    state.apply_event(SessionEvent::TodoSnapshot(TodoSnapshotEvent::new(vec![
        TodoItem {
            id: "t1".into(),
            content: "Draw the card boundary".into(),
            status: TodoStatus::InProgress,
        },
    ])));

    let theme = Theme::for_name(ThemeName::Wireframe, 0, None);
    let lines = transcript_lines(&state, theme, 60)
        .into_iter()
        .map(|line| line.to_string())
        .collect::<Vec<_>>();

    assert!(
        lines
            .iter()
            .any(|line| line.starts_with('┌') && line.ends_with('┐')),
        "{lines:?}"
    );
    assert!(
        lines
            .iter()
            .any(|line| line.starts_with('└') && line.ends_with('┘')),
        "{lines:?}"
    );
    assert!(
        lines
            .iter()
            .any(|line| line.contains("# Todos") && line.starts_with('│') && line.ends_with('│')),
        "{lines:?}"
    );
}

#[test]
fn tool_card_then_todo_card_keeps_timeline_separator() {
    let mut state = TuiState::default();
    state.apply_event(SessionEvent::ToolStarted(ToolStartedEvent {
        call_id: "call-shell".into(),
        name: "shell__exec".into(),
        summary: "run".into(),
        arguments: Some(json!({"command": "cargo test"}).to_string()),
    }));
    state.apply_event(SessionEvent::ToolFinished(ToolFinishedEvent {
        call_id: "call-shell".into(),
        name: "shell__exec".into(),
        summary: "exit 0 · stdout 3 lines".into(),
        outcome: ToolOutcome::Success,
        output: Some(
            json!({
                "ok": true,
                "tool": "shell__exec",
                "data": {
                    "status": 0,
                    "success": true,
                    "stdout": "line1\nline2\nline3\n",
                    "stdout_truncated": false,
                    "stderr": "",
                    "stderr_truncated": false
                }
            })
            .to_string(),
        ),
    }));
    state.apply_event(SessionEvent::TodoSnapshot(TodoSnapshotEvent::new(vec![
        TodoItem {
            id: "t1".into(),
            content: "Do something".into(),
            status: TodoStatus::InProgress,
        },
    ])));

    let lines = transcript_lines(&state, Theme::dark(), 80)
        .into_iter()
        .map(|line| line.to_string())
        .collect::<Vec<_>>();

    let shell_end = lines
        .iter()
        .position(|line| line.contains("line3"))
        .expect("shell output line");
    let todo_title = lines
        .iter()
        .position(|line| line.contains("# Todos"))
        .expect("todo card title");
    let between = &lines[shell_end + 1..todo_title];
    assert!(
        between.iter().any(|line| line.is_empty()),
        "expected a blank timeline separator between tool output and todo card: {lines:?}"
    );
}

#[test]
fn streaming_assistant_tail_is_faded() {
    let theme = Theme::dark();
    let now = std::time::Instant::now();
    let item = TimelineItem::Assistant(MessageView {
        id: None,
        submission_id: None,
        role: MessageRole::Assistant,
        text: "hello world".into(),
        attachments: Vec::new(),
        selected_skills: Vec::new(),
        streaming: true,
        queued: false,
        reveal_tail: std::collections::VecDeque::from([(now, 5u32)]),
    });

    let document = render_timeline_item_document(
        &item,
        theme,
        40,
        0,
        false,
        false,
        None,
        crate::command::ThoughtsDisplayMode::Full,
        false,
        crate::command::ToolsDisplayMode::Detailed,
        false,
        tool_card::ToolGroupStats::default(),
        None,
        now,
        theme.canvas(),
        &crate::tui::i18n::Translator::new(crate::tui::i18n::Language::En),
        None,
    );

    assert!(document.validate());
    let text: String = document.lines[0]
        .spans
        .iter()
        .map(|span| span.text.as_str())
        .collect();
    assert_eq!(text, "  hello world");
    assert_eq!(document.lines[0].spans[1].style.fg, Some(theme.text));
    let trailing = document.lines[0].spans.last().expect("trailing span");
    assert_eq!(trailing.text, "d");
    assert_ne!(trailing.style.fg, Some(theme.text));
}
