use super::*;
use crate::agent::{AutoContinueState, TodoItem, TodoStatus};
use crate::tool::{QuestionOption, QuestionRequest, QuestionSpec};
use crate::transcript::{TranscriptEvent, TranscriptRecord};
use crate::tui::components::layout::SIDEBAR_WIDTH;
use crate::tui::events::{
    AutoContinueChangedEvent, ContextTreeUpdatedEvent, ContextViewUpdatedEvent, ErrorEvent,
    NoticeEvent, NoticeKind, PermissionResolutionEvent, ProcessIssueEvent, RetryLifecycleEvent,
    SessionEvent, TodoSnapshotEvent, ToolCancelledEvent, ToolPendingEvent,
};

#[test]
fn explicit_and_system_following_language_resolution_are_distinct() {
    let mut state = TuiState::default();
    assert_eq!(
        state.effective_language(),
        crate::tui::i18n::system_language()
    );
    state.set_language(Some(crate::tui::i18n::Language::ZhCn));
    assert_eq!(state.effective_language(), crate::tui::i18n::Language::ZhCn);
    state.set_language(None);
    assert_eq!(
        state.effective_language(),
        crate::tui::i18n::system_language()
    );
}

fn dialog(items: &[(&str, &str, Option<&str>)]) -> DialogState {
    DialogState::new(
        DialogKind::ModelPicker,
        "Pick",
        None,
        items
            .iter()
            .map(|(id, label, detail)| DialogItem::new(*id, *label, detail.map(str::to_string)))
            .collect(),
    )
}

#[test]
fn dialog_visibility_and_selection_preserve_item_indices() {
    let cases = [
        ("", vec![0, 1, 2, 3], 0, 1, 3),
        ("match", vec![1, 3], 0, 3, 3),
        ("missing", Vec::new(), 2, 0, 0),
    ];

    for (query, expected_visible, initial, next, previous) in cases {
        let mut state = dialog(&[
            ("zero", "Zero", None),
            ("match-one", "One", None),
            ("other", "Other", None),
            ("match-two", "Two", None),
        ]);
        state.query = query.into();
        state.selected = initial;

        let visible = state
            .visible_items()
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        assert_eq!(visible, expected_visible, "query={query}");

        state.select_next();
        assert_eq!(state.selected, next, "next query={query}");
        state.selected = initial;
        state.select_previous();
        assert_eq!(state.selected, previous, "previous query={query}");
    }
}

#[test]
fn dialog_query_clamps_to_first_match_and_selected_item_uses_original_index() {
    let mut state = dialog(&[
        ("zero", "Zero", None),
        ("target", "Target", None),
        ("other", "Other", None),
    ]);
    state.selected = 2;
    for ch in "tar".chars() {
        state.insert_query_char(ch);
    }

    assert_eq!(state.selected, 1);
    assert_eq!(
        state.selected_item().map(|item| item.id.as_str()),
        Some("target")
    );
    assert_eq!(
        state
            .visible_items()
            .map(|(index, _)| index)
            .collect::<Vec<_>>(),
        vec![1]
    );
}

#[test]
fn dialog_wraps_filtered_selection_in_both_directions() {
    let mut state = dialog(&[
        ("first", "First", None),
        ("skip", "Skip", None),
        ("last", "Last", None),
    ]);
    state.query = "st".into();
    state.selected = 2;
    state.select_next();
    assert_eq!(state.selected, 0);
    state.select_previous();
    assert_eq!(state.selected, 2);
}

fn custom_edit_item(initial: &str) -> PendingQuestionItem {
    PendingQuestionItem {
        question: "q".into(),
        header: "h".into(),
        options: Vec::new(),
        multiple: false,
        selected_labels: Vec::new(),
        custom_text: String::new(),
        custom_cursor: 0,
        custom_edit_text: initial.into(),
        custom_edit_cursor: initial.len(),
    }
}

#[test]
fn custom_edit_insert_backspace_delete_keep_unicode_intact() {
    let mut item = custom_edit_item("猫😺a");
    item.move_custom_cursor_left();
    item.move_custom_cursor_left();
    item.insert_custom_edit("喵");
    assert_eq!(item.custom_edit_text, "猫喵😺a");

    item.backspace_custom_edit();
    assert_eq!(item.custom_edit_text, "猫😺a");

    item.move_custom_cursor_home();
    item.delete_custom_edit();
    assert_eq!(item.custom_edit_text, "😺a");

    item.move_custom_cursor_end();
    item.insert_custom_edit("！");
    assert_eq!(item.custom_edit_text, "😺a！");
}

#[test]
fn custom_edit_cursor_never_lands_inside_a_multibyte_char() {
    let mut item = custom_edit_item("😺😺");
    item.move_custom_cursor_left();
    // Cursor points at a char boundary; left again hits the first char.
    assert_eq!(item.custom_edit_cursor, "😺".len());
    item.move_custom_cursor_left();
    assert_eq!(item.custom_edit_cursor, 0);
    // Left at the start is a no-op.
    item.move_custom_cursor_left();
    assert_eq!(item.custom_edit_cursor, 0);

    item.move_custom_cursor_end();
    item.move_custom_cursor_right();
    assert_eq!(item.custom_edit_cursor, item.custom_edit_text.len());
}

#[test]
fn custom_edit_backspace_and_delete_at_boundaries_are_noops() {
    let mut item = custom_edit_item("ab");
    item.move_custom_cursor_home();
    item.backspace_custom_edit();
    assert_eq!(item.custom_edit_text, "ab");
    item.move_custom_cursor_end();
    item.delete_custom_edit();
    assert_eq!(item.custom_edit_text, "ab");
}

fn question_state(questions: Vec<QuestionSpec>) -> PendingQuestionState {
    PendingQuestionState::new(QuestionRequest { questions }, None)
}

fn runtime_context(
    session_id: &str,
    branch_id: &str,
    revision: u64,
    leaf: u64,
) -> RuntimeActiveContext {
    let snapshot = crate::transcript::transcript_projection::project_runtime_restore_snapshot(
        session_id.into(),
        Vec::new(),
        crate::transcript::transcript_projection::SessionContextCursor {
            branch_id: Some(crate::transcript::ROOT_CONTEXT_BRANCH_ID.into()),
            leaf_sequence: None,
        },
        &[],
    )
    .unwrap()
    .snapshot;
    let mut context = RuntimeActiveContext::try_from(&snapshot).unwrap();
    context.active_context.branch_id = branch_id.into();
    context.context_scope_revision = revision;
    context.leaf_sequence = leaf;
    context
}

fn runtime_update(
    session_id: &str,
    branch_id: &str,
    revision: u64,
    leaf: u64,
    disposition: crate::tui::events::RuntimeContextDisposition,
) -> crate::tui::events::RuntimeContextUpdatedEvent {
    crate::tui::events::RuntimeContextUpdatedEvent {
        context: runtime_context(session_id, branch_id, revision, leaf),
        disposition,
    }
}

fn option(label: &str) -> QuestionOption {
    QuestionOption {
        label: label.into(),
        description: format!("{label} option"),
    }
}

#[test]
fn single_question_does_not_show_confirm_tab_and_submits_immediately() {
    let mut state = question_state(vec![QuestionSpec {
        question: "Choose one".into(),
        header: "Mode".into(),
        options: vec![option("Fast"), option("Safe")],
        multiple: false,
    }]);

    assert!(!state.show_confirm_tab());
    assert_eq!(state.total_tabs(), 1);
    assert_eq!(state.pick_option(1), QuestionAdvance::Submit);
    assert_eq!(state.questions[0].answers(), vec!["Safe".to_string()]);
}

#[test]
fn multi_question_custom_answer_advances_to_the_next_question() {
    let mut state = question_state(vec![
        QuestionSpec {
            question: "Choose several".into(),
            header: "Features".into(),
            options: vec![option("Alpha")],
            multiple: true,
        },
        QuestionSpec {
            question: "Choose a mode".into(),
            header: "Mode".into(),
            options: vec![option("Fast")],
            multiple: false,
        },
    ]);
    state.active_row.focus(1);
    state.begin_custom_edit();
    state.questions[0].custom_edit_text = "Gamma".into();

    assert_eq!(state.commit_custom_answer(), QuestionAdvance::Advanced);
    assert_eq!(state.active_tab, 1);
    assert_eq!(state.questions[0].answers(), vec!["Gamma".to_string()]);
}

#[test]
fn auto_review_toggle_key_uses_call_id() {
    let mut state = TuiState::default();
    assert!(
        !state
            .tool_output_overrides
            .contains_key("auto-review:call-review")
    );

    state.toggle_tool_output("auto-review:call-review");
    assert_eq!(
        state.tool_output_overrides.get("auto-review:call-review"),
        Some(&true)
    );

    state.toggle_tool_output("auto-review:call-review");
    assert_eq!(
        state.tool_output_overrides.get("auto-review:call-review"),
        Some(&false)
    );
}

#[test]
fn confirm_scroll_clamps_and_resets_on_tab_or_answer_changes() {
    let mut state = question_state(vec![
        QuestionSpec {
            question: "First".into(),
            header: "One".into(),
            options: vec![option("A")],
            multiple: true,
        },
        QuestionSpec {
            question: "Second".into(),
            header: "Two".into(),
            options: vec![option("B")],
            multiple: true,
        },
    ]);
    state.focus_tab(2);
    state.set_confirm_scroll_max(3);
    state.move_next_row();
    state.move_next_row();
    state.move_next_row();
    state.move_next_row();
    assert_eq!(state.confirm_scroll, 3);
    state.move_prev_row();
    assert_eq!(state.confirm_scroll, 2);
    state.focus_tab(0);
    assert_eq!(state.confirm_scroll, 0);
    state.focus_tab(2);
    state.set_confirm_scroll_max(3);
    state.move_next_row();
    state.pick_option(0);
    assert_eq!(state.confirm_scroll, 0);
}

#[test]
fn unanswered_confirm_focuses_first_missing_question() {
    let mut state = question_state(vec![
        QuestionSpec {
            question: "First".into(),
            header: "One".into(),
            options: vec![option("A")],
            multiple: false,
        },
        QuestionSpec {
            question: "Second".into(),
            header: "Two".into(),
            options: vec![option("B")],
            multiple: false,
        },
    ]);
    state.pick_option(0);
    state.focus_tab(2);

    assert_eq!(state.first_unanswered_tab(), Some(1));
    assert!(state.is_confirm_tab());
}

#[test]
fn sidebar_visibility_respects_terminal_width_and_user_preference() {
    let mut state = TuiState::default();
    let width = SIDEBAR_WIDTH * 3;
    assert!(state.sidebar_visible(width));
    assert!(!state.sidebar_visible(width - 1));

    state.last_terminal_width = width;
    state.toggle_sidebar();
    assert!(!state.sidebar_visible(width));
    state.toggle_sidebar();
    assert!(state.sidebar_visible(width));
}

#[test]
fn sidebar_is_dropped_below_the_minimum_terminal_width_even_when_forced_open() {
    let mut state = TuiState::default();
    let narrow = SIDEBAR_WIDTH * 3 - 1;
    state.last_terminal_width = narrow;
    state.toggle_sidebar();

    assert!(!state.sidebar_visible(narrow));
    assert!(state.sidebar_visible(SIDEBAR_WIDTH * 3));
}

#[test]
fn composer_draft_keeps_boundary_skill_tokens_after_outer_text_trim() {
    let mut state = TuiState::default();
    state.add_composer_skill("rust-audit".into());
    state.input_buffer.push_str("  inspect  ");
    state.input_cursor = state.input_buffer.len();

    let mut content = state.composer_content();
    content.trim_outer_text();
    assert_eq!(content.text, "inspect");
    assert_eq!(content.selected_skills, vec!["rust-audit"]);
}

#[test]
fn compaction_started_preserves_token_usage_and_marks_active() {
    let mut state = TuiState::default();
    state.apply_live_token_usage(ModelTokenUsage {
        used_tokens: 1_000,
        context_window_tokens: 10_000,
        input_tokens: 900,
        output_tokens: 100,
        cached_tokens: 400,
        cache_report: None,
        prompt_composition: Vec::new(),
    });
    state.sidebar_model_token_usage = Some(ModelTokenUsage {
        used_tokens: 800,
        context_window_tokens: 10_000,
        input_tokens: 700,
        output_tokens: 100,
        cached_tokens: 200,
        cache_report: None,
        prompt_composition: Vec::new(),
    });

    state.status_spinner_frame = 77;
    state.apply_event(SessionEvent::CompactionStarted);

    assert!(state.compaction_active);
    assert_eq!(state.compaction_animation_start_frame, 77);
    assert_eq!(
        state
            .active_model_token_usage()
            .map(|usage| usage.used_tokens),
        Some(1_000)
    );
    assert_eq!(
        state
            .active_sidebar_model_token_usage()
            .map(|usage| usage.used_tokens),
        Some(800)
    );

    state.apply_event(SessionEvent::CompactionCommitted {
        summary: Some("compacted".into()),
    });
    assert!(!state.compaction_active);
}

#[test]
fn compaction_commit_preserves_unrelated_notice() {
    let mut state = TuiState::default();
    state.apply_event(SessionEvent::CompactionStarted);
    state.apply_event(SessionEvent::Notice(NoticeEvent::info(
        "Fast mode auto-disabled: current model is unavailable",
    )));

    state.apply_event(SessionEvent::CompactionCommitted {
        summary: Some("compacted".into()),
    });

    assert_eq!(
        state.toast().map(|toast| toast.message.as_str()),
        Some("Fast mode auto-disabled: current model is unavailable")
    );
}

#[test]
fn compaction_failure_and_interrupt_clear_active_flag() {
    let mut state = TuiState::default();
    state.apply_event(SessionEvent::CompactionStarted);
    assert!(state.compaction_active);

    state.apply_event(SessionEvent::CompactionFailed);
    assert!(!state.compaction_active);

    state.apply_event(SessionEvent::CompactionStarted);
    state.apply_event(SessionEvent::Interrupted);
    assert!(!state.compaction_active);
}

#[test]
fn compaction_cancelled_clears_streaming_block_without_error_toast() {
    let mut state = TuiState::default();
    state.apply_event(SessionEvent::CompactionStarted);
    state.apply_event(SessionEvent::CompactionPreviewDelta {
        delta: "working summary".into(),
    });
    assert!(state.compaction_active);

    state.apply_event(SessionEvent::CompactionCancelled);

    assert!(!state.compaction_active);
    assert!(
        !state
            .timeline
            .items()
            .iter()
            .any(|item| matches!(item, crate::tui::timeline::TimelineItem::Compaction(_))),
        "cancelled compaction must not leave a compaction block"
    );
    assert!(state.toast().is_none());
}

#[test]
fn permission_resolved_clears_active_tool_and_pending_permission() {
    let mut state = TuiState::default();
    let request = PermissionRequestEvent::new("call-1", "shell__exec", "run ls");

    state.set_pending_permission_projection(Some(PermissionView::from_request(request.clone())));
    state.apply_event(SessionEvent::PermissionRequested(request));
    assert_eq!(state.phase, AppPhase::WaitingForPermission);
    assert_eq!(state.active_tool_call_id.as_deref(), Some("call-1"));
    assert!(state.pending_permission.is_some());

    state.apply_event(SessionEvent::PermissionResolved(
        PermissionResolutionEvent::approved("call-1"),
    ));

    assert_eq!(state.phase, AppPhase::Running);
    assert_eq!(state.active_tool_call_id, None);
    assert!(state.pending_permission.is_none());

    let permission = state
        .timeline
        .items()
        .iter()
        .find_map(|item| match item {
            crate::tui::timeline::TimelineItem::Permission(permission) => Some(permission),
            _ => None,
        })
        .expect("permission item exists");
    assert_eq!(
        permission.status,
        crate::tui::timeline::PermissionPromptStatus::Approved
    );
}

#[test]
fn permission_choice_wraps_and_restarts_with_each_request() {
    let mut state = TuiState::default();
    let mut request = PermissionRequestEvent::new("call-1", "shell__exec", "run tests");
    request.can_allow_always = true;
    state.set_pending_permission_projection(Some(PermissionView::from_request(request)));

    assert_eq!(
        state.highlighted_permission_choice(),
        Some(PermissionChoice::AllowOnce)
    );
    state.move_permission_choice_next();
    assert_eq!(
        state.highlighted_permission_choice(),
        Some(PermissionChoice::AllowAlways)
    );
    state.move_permission_choice_prev();
    assert_eq!(
        state.highlighted_permission_choice(),
        Some(PermissionChoice::AllowOnce)
    );
    state.move_permission_choice_prev();
    assert_eq!(
        state.highlighted_permission_choice(),
        Some(PermissionChoice::Reject)
    );

    state.set_pending_permission_projection(Some(PermissionView::from_request(
        PermissionRequestEvent::new("call-2", "shell__exec", "run ls"),
    )));
    assert_eq!(
        state.highlighted_permission_choice(),
        Some(PermissionChoice::AllowOnce)
    );
    state.move_permission_choice_next();
    assert_eq!(
        state.highlighted_permission_choice(),
        Some(PermissionChoice::Reject)
    );
}

#[test]
fn interrupted_event_cancels_active_tool_and_clears_permission() {
    let mut state = TuiState::default();
    state.apply_event(SessionEvent::ToolPending(ToolPendingEvent::new(
        "call-1",
        "shell__exec",
    )));
    state.set_pending_permission_projection(Some(PermissionView::from_request(
        PermissionRequestEvent::new("call-1", "shell__exec", "run ls"),
    )));
    state.apply_event(SessionEvent::PermissionRequested(
        PermissionRequestEvent::new("call-1", "shell__exec", "run ls"),
    ));

    state.apply_event(SessionEvent::Interrupted);

    assert_eq!(state.phase, AppPhase::Completed);
    assert_eq!(state.active_tool_call_id, None);
    assert!(state.pending_permission.is_none());
    assert!(matches!(
        state.timeline.items().iter().find_map(|item| match item {
            crate::tui::timeline::TimelineItem::Tool(tool) => Some(tool),
            _ => None,
        }),
        Some(tool) if tool.status == crate::tui::timeline::ToolExecutionStatus::Cancelled
    ));
}

#[test]
fn unseen_late_tool_events_do_not_revive_parent_state_after_interrupt() {
    let mut state = TuiState::default();
    state.apply_event(SessionEvent::ToolPending(ToolPendingEvent::new(
        "call-1",
        "shell__exec",
    )));
    state.apply_event(SessionEvent::Interrupted);

    state.apply_event(SessionEvent::ToolFinished(
        crate::tui::events::ToolFinishedEvent::new(
            "late-call",
            "fs__write",
            "fs__write completed",
            ToolOutcome::Success,
        ),
    ));

    assert_eq!(state.phase, AppPhase::Completed);
    assert_eq!(state.active_tool_call_id, None);

    assert_eq!(
        state
            .timeline
            .items()
            .iter()
            .filter(|item| matches!(item, crate::tui::timeline::TimelineItem::Tool(_)))
            .count(),
        1
    );
}

#[test]
fn unseen_late_tool_cancelled_does_not_override_interrupt_state() {
    let mut state = TuiState::default();
    state.apply_event(SessionEvent::ToolPending(ToolPendingEvent::new(
        "call-1",
        "shell__exec",
    )));
    state.apply_event(SessionEvent::Interrupted);

    state.apply_event(SessionEvent::ToolCancelled(ToolCancelledEvent::new(
        "late-call",
        "fs__write",
    )));

    assert_eq!(state.phase, AppPhase::Completed);
    assert_eq!(state.active_tool_call_id, None);

    assert_eq!(
        state
            .timeline
            .items()
            .iter()
            .filter(|item| matches!(item, crate::tui::timeline::TimelineItem::Tool(_)))
            .count(),
        1
    );
}

#[test]
fn unseen_late_tool_events_do_not_revive_child_state_after_interrupt() {
    let mut state = TuiState::default();
    state.replace_child_timeline_from_records(
        &[],
        "parent-session",
        "child-session",
        "explorer",
        0,
        1,
        1,
    );
    state.apply_child_session_event(
        "child-session",
        SessionEvent::ToolPending(ToolPendingEvent::new("call-1", "shell__exec")),
    );
    state.apply_child_session_event("child-session", SessionEvent::Interrupted);

    state.apply_child_session_event(
        "child-session",
        SessionEvent::ToolFinished(crate::tui::events::ToolFinishedEvent::new(
            "late-call",
            "fs__write",
            "fs__write completed",
            ToolOutcome::Success,
        )),
    );

    // Child lifecycle state is local; the parent remains untouched.
    assert_eq!(state.phase, AppPhase::Idle);
    assert_eq!(state.active_tool_call_id, None);
    assert_eq!(state.child_view_phase(), Some(AppPhase::Completed));

    assert!(matches!(
        state.active_timeline().items().iter().find_map(|item| match item {
            crate::tui::timeline::TimelineItem::Tool(tool) => Some(tool),
            _ => None,
        }),
        Some(tool) if tool.status == crate::tui::timeline::ToolExecutionStatus::Cancelled
    ));
    assert_eq!(
        state
            .active_timeline()
            .items()
            .iter()
            .filter(|item| matches!(item, crate::tui::timeline::TimelineItem::Tool(_)))
            .count(),
        1
    );
}

#[test]
fn transcript_scroll_uses_bottom_relative_offset() {
    let mut state = TuiState::default();

    assert_eq!(state.transcript_scroll_offset(), 0);
    assert!(state.auto_scroll);

    state.scroll_transcript_up(3);
    assert_eq!(state.transcript_scroll_offset(), 3);
    assert!(!state.auto_scroll);

    state.scroll_transcript_down(2);
    assert_eq!(state.transcript_scroll_offset(), 1);
    assert!(!state.auto_scroll);

    state.scroll_transcript_down(10);
    assert_eq!(state.transcript_scroll_offset(), 0);
    assert!(state.auto_scroll);
}

#[test]
fn transcript_append_preserves_manual_scroll_offset() {
    let mut state = TuiState::default();
    state.scroll_transcript_up(4);

    state.apply_event(SessionEvent::UserMessage(UserMessageEvent::new("hello")));

    assert_eq!(state.transcript_scroll_offset(), 4);
    assert!(!state.auto_scroll);
    assert_eq!(state.timeline.items().len(), 1);
}

#[test]
fn scroll_up_while_transcript_fits_viewport_keeps_auto_scroll() {
    let mut state = TuiState::default();
    state.last_transcript_area.height = 10;
    state.sync_transcript_viewport_rows_with_reflow(3, false);

    state.scroll_transcript_up(1);
    state.sync_transcript_viewport_rows_with_reflow(12, false);

    assert_eq!(state.transcript_scroll_offset(), 0);
    assert!(state.auto_scroll);
}

#[test]
fn manual_history_offset_tracks_large_transcript_growth() {
    let mut state = TuiState::default();
    state.last_transcript_area.height = 30;
    state.sync_transcript_viewport_rows_with_reflow(60_000, false);
    state.scroll_transcript_up(59_970);
    assert_eq!(state.transcript_scroll_offset(), 59_970);
    assert!(!state.auto_scroll);

    state.sync_transcript_viewport_rows_with_reflow(90_000, false);
    assert_eq!(state.transcript_scroll_offset(), 89_970);
    state.scroll_transcript_up(1);
    assert_eq!(state.transcript_scroll_offset(), 89_970);
}

#[test]
fn sync_transcript_viewport_rows_clamps_invalid_offset_and_restores_auto_scroll() {
    let mut state = TuiState::default();
    state.last_transcript_area.height = 5;
    state.sync_transcript_viewport_rows_with_reflow(10, false);
    state.scroll_transcript_up(4);
    assert_eq!(state.transcript_scroll_offset(), 4);
    assert!(!state.auto_scroll);

    state.last_transcript_area.height = 10;
    state.sync_transcript_viewport_rows_with_reflow(10, false);

    assert_eq!(state.transcript_scroll_offset(), 0);
    assert!(state.auto_scroll);
}

#[test]
fn transcript_bottom_detection_handles_fitting_and_scrolled_content() {
    let state = TuiState::default();

    assert!(state.transcript_is_at_bottom(3, 10));
    assert!(state.transcript_is_at_bottom(20, 5));

    let mut scrolled = TuiState::default();
    scrolled.scroll_transcript_up(5);

    assert!(!scrolled.transcript_is_at_bottom(20, 5));
    assert!(scrolled.transcript_is_at_bottom(3, 10));
}

#[test]
fn slash_panel_is_hidden_in_child_view() {
    let mut state = TuiState::default();

    state.set_input("/p");
    assert!(state.slash_panel_is_open());

    state.replace_child_timeline_from_records(
        &[],
        "parent-session",
        "child-session",
        "explorer",
        0,
        1,
        1,
    );

    assert!(!state.slash_panel_is_open());
    assert!(state.input_buffer.is_empty());
}

#[test]
fn child_view_metadata_prefers_latest_model_change() {
    let records = vec![
        TranscriptRecord {
            session_id: "child-session".into(),
            sequence: 1,
            timestamp_ms: 0,
            context_branch_id: None,
            event: TranscriptEvent::SessionStarted {
                model: "gpt-5.5".into(),
            },
        },
        TranscriptRecord {
            session_id: "child-session".into(),
            sequence: 2,
            timestamp_ms: 1,
            context_branch_id: None,
            event: TranscriptEvent::ModelChanged {
                previous_model: "gpt-5.5".into(),
                new_model: "gpt-5.5-mini".into(),
            },
        },
    ];
    let mut state = TuiState::default();
    state.replace_child_timeline_from_records(
        &records,
        "parent-session",
        "child-session",
        "fixer",
        1,
        2,
        1,
    );

    let metadata = state.child_view_metadata().expect("child metadata");
    assert_eq!(metadata.model.as_deref(), Some("gpt-5.5-mini"));
    assert_eq!(metadata.record_count, 2);
}

#[test]
fn child_view_replacement_preserves_running_phase_and_pending_permission() {
    let mut state = TuiState::default();
    let request = PermissionRequestEvent::new("call-1", "shell__exec", "run ls");
    state.set_pending_permission_projection(Some(PermissionView::from_request(request.clone())));
    state.apply_event(SessionEvent::PermissionRequested(request));
    state.open_dialog(DialogState::new(
        DialogKind::ModelPicker,
        "Model",
        None,
        vec![DialogItem::new("m1", "Model 1", None)],
    ));
    state.set_input("/p");
    state.scroll_transcript_up(3);

    state.replace_child_timeline_from_records(
        &[],
        "parent-session",
        "child-session",
        "explorer",
        0,
        1,
        1,
    );

    assert_eq!(state.phase, AppPhase::WaitingForPermission);
    assert_eq!(state.active_tool_call_id.as_deref(), Some("call-1"));
    assert!(state.pending_permission.is_some());
    assert!(state.dialog().is_none());
    assert!(state.input_buffer.is_empty());
    assert!(!state.slash_panel_is_open());
    assert_eq!(state.transcript_scroll_offset(), 0);
    assert!(state.auto_scroll);
}

#[test]
fn child_view_refresh_preserves_runtime_state_and_updates_record_count() {
    let mut state = TuiState {
        phase: AppPhase::Running,
        active_tool_call_id: Some("call-2".into()),
        pending_permission: Some(PermissionView::from_request(PermissionRequestEvent::new(
            "call-2",
            "shell__exec",
            "run cargo test",
        ))),
        ..Default::default()
    };
    state.replace_child_timeline_from_records(
        &[],
        "parent-session",
        "child-session",
        "explorer",
        0,
        1,
        1,
    );
    state.open_dialog(DialogState::new(
        DialogKind::ModelPicker,
        "Model",
        None,
        vec![DialogItem::new("m1", "Model 1", None)],
    ));

    state.refresh_child_timeline_from_records(&[
        TranscriptRecord {
            session_id: "child-session".into(),
            sequence: 1,
            timestamp_ms: 0,
            context_branch_id: None,
            event: TranscriptEvent::SessionStarted {
                model: "gpt-test".into(),
            },
        },
        TranscriptRecord {
            session_id: "child-session".into(),
            sequence: 2,
            timestamp_ms: 1,
            context_branch_id: None,
            event: TranscriptEvent::AssistantMessage {
                content: "updated child output".into(),
            },
        },
    ]);

    let metadata = state.child_view_metadata().expect("child metadata");
    assert_eq!(metadata.record_count, 2);
    assert_eq!(metadata.model.as_deref(), Some("gpt-test"));
    assert_eq!(state.phase, AppPhase::WaitingForPermission);
    assert_eq!(state.active_tool_call_id.as_deref(), Some("call-2"));
    assert!(state.pending_permission.is_some());
    assert!(state.dialog().is_some());
}

#[test]
fn restore_parent_view_preserves_pending_permission() {
    let mut state = TuiState::default();
    state.set_pending_permission_projection(Some(PermissionView::from_request(
        PermissionRequestEvent::new("call-1", "shell__exec", "run ls"),
    )));
    state.apply_event(SessionEvent::PermissionRequested(
        PermissionRequestEvent::new("call-1", "shell__exec", "run ls"),
    ));
    state.replace_child_timeline_from_records(
        &[],
        "parent-session",
        "child-session",
        "explorer",
        0,
        1,
        1,
    );

    state.restore_parent_timeline_view();

    assert_eq!(state.phase, AppPhase::WaitingForPermission);
    assert!(state.pending_permission.is_some());
    assert_eq!(state.active_tool_call_id.as_deref(), Some("call-1"));
}

#[test]
fn child_permission_events_project_into_active_parent_subagent_card() {
    let mut state = TuiState::default();
    state.apply_event(SessionEvent::ToolStarted(
        crate::tui::events::ToolStartedEvent::new(
            "parent-call",
            "agent__fixer",
            "apply requested fix",
        ),
    ));

    let mut request =
        PermissionRequestEvent::new("perm-1", "shell__exec", "run cargo test --bin letcode");
    request.rationale = Some("validation".into());

    state.apply_child_session_event_with_agent(
        "child-session",
        Some("fixer"),
        Some("parent-call"),
        SessionEvent::PermissionRequested(request),
    );

    let tool = state
        .timeline
        .items()
        .iter()
        .find_map(|item| match item {
            crate::tui::timeline::TimelineItem::Tool(tool) => Some(tool),
            _ => None,
        })
        .expect("parent subagent tool exists");
    let output = tool.output.as_deref().expect("live summary payload exists");
    assert!(output.contains("approval needed"), "{output}");
    assert!(output.contains("shell__exec"), "{output}");
    assert!(output.contains("run cargo test --bin letcode"), "{output}");
}

#[test]
fn child_feedback_respects_view_ownership_and_labels_errors() {
    let mut state = TuiState::default();
    state.replace_child_timeline_from_records(
        &[],
        "parent-session",
        "child-session",
        "explorer",
        0,
        1,
        1,
    );

    for (kind, message, expected_kind) in [
        (NoticeKind::Info, "child info", ToastKind::Info),
        (NoticeKind::Success, "child success", ToastKind::Success),
    ] {
        state.apply_child_session_event_with_agent(
            "child-session",
            Some("explorer"),
            None,
            SessionEvent::Notice(NoticeEvent::new(message, kind)),
        );
        assert_eq!(
            state
                .toast()
                .map(|toast| (toast.message.as_str(), toast.kind)),
            Some((message, expected_kind))
        );
    }

    for kind in [NoticeKind::Info, NoticeKind::Success] {
        state.show_toast("keep me", ToastKind::Success);
        state.apply_child_session_event_with_agent(
            "background-child",
            Some("fixer"),
            None,
            SessionEvent::Notice(NoticeEvent::new("background notice", kind)),
        );
        assert_eq!(
            state
                .toast()
                .map(|toast| (toast.message.as_str(), toast.kind)),
            Some(("keep me", ToastKind::Success))
        );
    }

    state.apply_child_session_event_with_agent(
        "background-child",
        None,
        None,
        SessionEvent::Notice(NoticeEvent::new(
            "child recoverable error",
            NoticeKind::RecoverableError,
        )),
    );
    assert_eq!(
        state
            .toast()
            .map(|toast| (toast.message.as_str(), toast.kind)),
        Some((
            "background-child: child recoverable error",
            ToastKind::Error
        ))
    );

    state.apply_child_session_event_with_agent(
        "background-child",
        Some("explorer"),
        None,
        SessionEvent::RetryScheduled(RetryLifecycleEvent {
            attempt: 2,
            max_attempts: 3,
            delay_secs: 4,
            error: "network request failed".into(),
        }),
    );
    assert_eq!(
        state
            .toast()
            .map(|toast| (toast.message.as_str(), toast.kind)),
        Some((
            "explorer · background-child: network request failed · Retrying in 4s · attempt 2 of 3",
            ToastKind::Error
        ))
    );
    assert_eq!(
        state
            .child_timeline_cache
            .get("background-child")
            .and_then(|child| child.retry.as_ref())
            .map(|retry| (retry.attempt, retry.max_attempts, retry.delay_secs)),
        Some((2, 3, 4))
    );
    let now = std::time::Instant::now();
    state
        .child_timeline_cache
        .get_mut("background-child")
        .and_then(|child| child.retry.as_mut())
        .expect("child retry")
        .started_at = now - std::time::Duration::from_secs(1);
    state.refresh_live_presentations(now);
    assert_eq!(
        state.toast().map(|toast| toast.message.as_str()),
        Some(
            "explorer · background-child: network request failed · Retrying in 3s · attempt 2 of 3"
        )
    );
    state.apply_child_session_event_with_agent(
        "background-child",
        Some("explorer"),
        None,
        SessionEvent::RetryStarted(RetryLifecycleEvent {
            attempt: 2,
            max_attempts: 3,
            delay_secs: 4,
            error: "network request failed".into(),
        }),
    );
    assert!(state.toast().is_none());

    state.apply_child_session_event_with_agent(
        "background-child-session-with-long-id",
        Some("fixer"),
        None,
        SessionEvent::ProcessIssue(ProcessIssueEvent::new("child process issue")),
    );
    let toast = state.toast().expect("background child issue toast");
    assert_eq!(toast.kind, ToastKind::Error);
    assert!(
        toast.message.starts_with("fixer · background-chil…:"),
        "{}",
        toast.message
    );
    assert!(
        toast.message.ends_with("child process issue"),
        "{}",
        toast.message
    );
    assert_eq!(toast.ticks_remaining(), ToastState::DEFAULT_TICKS);
}

#[test]
fn parent_retry_countdown_stays_live_until_retry_starts() {
    let mut state = TuiState::default();
    let retry = RetryLifecycleEvent {
        attempt: 2,
        max_attempts: 4,
        delay_secs: 2,
        error: "provider unavailable".into(),
    };
    let now = std::time::Instant::now();
    state.retry = Some(RetryNoticeState::from_lifecycle_at(retry.clone(), now));
    state.retry_toast_scope = Some(RetryToastScope::Parent);
    state.refresh_live_presentations(now);
    assert_eq!(
        state.toast().map(|toast| toast.message.as_str()),
        Some("Retrying in 2s · attempt 2 of 4")
    );
    state.refresh_live_presentations(now + std::time::Duration::from_secs(1));
    assert_eq!(
        state.toast().map(|toast| toast.message.as_str()),
        Some("Retrying in 1s · attempt 2 of 4")
    );
    state.apply_event(SessionEvent::RetryStarted(retry.clone()));
    assert!(state.toast().is_none());

    state.retry = Some(RetryNoticeState::from_lifecycle_at(retry.clone(), now));
    state.retry_toast_scope = Some(RetryToastScope::Parent);
    state.refresh_live_presentations(now);
    state.apply_event(SessionEvent::Interrupted);
    assert!(state.toast().is_none());

    state.retry = Some(RetryNoticeState::from_lifecycle_at(retry.clone(), now));
    state.retry_toast_scope = Some(RetryToastScope::Parent);
    state.refresh_live_presentations(now);
    state.apply_event(SessionEvent::Done);
    assert!(state.toast().is_none());

    state.retry = Some(RetryNoticeState::from_lifecycle_at(retry, now));
    state.retry_toast_scope = Some(RetryToastScope::Parent);
    state.refresh_live_presentations(now);
    state.apply_event(SessionEvent::Error(crate::tui::events::ErrorEvent::new(
        "failed",
    )));
    assert!(state.toast().is_none());
}

#[test]
fn child_retry_toast_clears_on_terminal_event() {
    let mut state = TuiState::default();
    let retry = RetryLifecycleEvent {
        attempt: 2,
        max_attempts: 4,
        delay_secs: 2,
        error: "provider unavailable".into(),
    };
    state.apply_child_session_event_with_agent(
        "child-1",
        Some("explorer"),
        None,
        SessionEvent::RetryScheduled(retry.clone()),
    );
    assert!(state.toast().is_some());
    state.apply_child_session_event_with_agent(
        "child-1",
        Some("explorer"),
        None,
        SessionEvent::Done,
    );
    assert!(state.toast().is_none());

    state.apply_child_session_event_with_agent(
        "child-2",
        Some("explorer"),
        None,
        SessionEvent::RetryScheduled(retry),
    );
    assert!(state.toast().is_some());
    state.apply_child_session_event_with_agent(
        "child-2",
        Some("explorer"),
        None,
        SessionEvent::Error(crate::tui::events::ErrorEvent::new("failed")),
    );
    assert!(state.toast().is_none());
}

#[test]
fn parent_notice_still_uses_plain_message() {
    let mut state = TuiState::default();
    state.apply_event(SessionEvent::Notice(NoticeEvent::info("parent info")));
    assert_eq!(
        state
            .toast()
            .map(|toast| (toast.message.as_str(), toast.kind)),
        Some(("parent info", ToastKind::Info))
    );
}

#[test]
fn terminal_events_seal_active_reasoning() {
    for terminal in [
        SessionEvent::Interrupted,
        SessionEvent::Done,
        SessionEvent::Error(crate::tui::events::ErrorEvent::new("failed")),
    ] {
        let start = std::time::Instant::now() - std::time::Duration::from_millis(250);
        let mut state = TuiState::default();
        state.apply_event(SessionEvent::ReasoningDelta(
            crate::tui::events::ReasoningDeltaEvent::at("reasoning-1", "working", start),
        ));
        state.apply_event(terminal);

        assert!(matches!(
            state.timeline.items().first(),
            Some(crate::tui::timeline::TimelineItem::Reasoning(reasoning))
                if !reasoning.streaming
                    && reasoning.started_at.is_none()
                    && reasoning.duration_ms.is_some()
        ));
        let revision = state.timeline.item_revisions()[0];
        state.apply_event(SessionEvent::Tick);
        assert_eq!(state.timeline.item_revisions()[0], revision);
    }
}

#[test]
fn compaction_lifecycle_only_commits_after_acknowledged_terminal_event() {
    let mut state = TuiState::default();
    state.apply_event(SessionEvent::CompactionStarted);
    state.apply_event(SessionEvent::CompactionPreviewDelta {
        delta: "working summary".into(),
    });
    assert!(matches!(
        state.timeline.items(),
        [crate::tui::timeline::TimelineItem::Compaction(view)]
            if view.streaming && view.summary == "working summary"
    ));

    state.apply_event(SessionEvent::CompactionCommitted {
        summary: Some("working summary".into()),
    });
    assert!(matches!(
        state.timeline.items(),
        [crate::tui::timeline::TimelineItem::Compaction(view)]
            if !view.streaming && view.summary == "working summary"
    ));
}

#[test]
fn compaction_non_success_and_terminal_events_clear_pending() {
    let terminals = [
        SessionEvent::CompactionNoProgress {
            blockers: vec!["no_historical_items".into()],
        },
        SessionEvent::CompactionFailed,
        SessionEvent::Interrupted,
        SessionEvent::Done,
        SessionEvent::Error(crate::tui::events::ErrorEvent::new("failed")),
    ];
    for terminal in terminals {
        let mut state = TuiState::default();
        state.apply_event(SessionEvent::CompactionStarted);
        state.apply_event(terminal);
        assert!(
            !state.timeline.items().iter().any(|item| matches!(
                item,
                crate::tui::timeline::TimelineItem::Compaction(view) if view.streaming
            )),
            "terminal event must clear streaming compaction"
        );
        assert!(
            !state.timeline.items().iter().any(|item| matches!(
                item,
                crate::tui::timeline::TimelineItem::Compaction(view) if !view.streaming
            )),
            "non-success must not create a committed compaction"
        );
    }

    let mut state = TuiState::default();
    state.apply_event(SessionEvent::CompactionStarted);
    state.apply_event(SessionEvent::CompactionNoProgress {
        blockers: vec!["no_safe_boundary".into()],
    });
    assert_eq!(
        state.toast().map(|toast| toast.message.as_str()),
        Some("Context limit reached; earlier context cannot be compacted safely yet.")
    );
}
