use super::*;

#[test]
fn programmer_day_intro_steps_through_each_frame() {
    let mut state = TuiState::default();
    let frame = WELCOME_INTRO_FADE_IN + WELCOME_INTRO_HOLD + WELCOME_INTRO_FADE_OUT;
    let at = |state: &mut TuiState, elapsed: std::time::Duration| {
        state.welcome_intro_started_at = Some(std::time::Instant::now() - elapsed);
        welcome_mark(state, true)
    };

    let fading_in = at(&mut state, WELCOME_INTRO_FADE_IN / 2);
    assert_eq!(fading_in.left.join("\n"), WELCOME_ART_1024_LEFT.join("\n"));
    assert!(fading_in.brightness > 0.0 && fading_in.brightness < 1.0);

    let held = at(&mut state, WELCOME_INTRO_FADE_IN + WELCOME_INTRO_HOLD / 2);
    assert_eq!(held.right.join("\n"), WELCOME_ART_1024_RIGHT.join("\n"));
    assert_eq!(held.brightness, 1.0);

    let fading_out = at(&mut state, frame - WELCOME_INTRO_FADE_OUT / 2);
    assert_eq!(fading_out.left.join("\n"), WELCOME_ART_1024_LEFT.join("\n"));
    assert!(fading_out.brightness > 0.0 && fading_out.brightness < 1.0);

    let errors = at(
        &mut state,
        frame + WELCOME_INTRO_FADE_IN + WELCOME_INTRO_HOLD / 2,
    );
    assert_eq!(errors.left.join("\n"), WELCOME_ART_ERR_LEFT.join("\n"));
    assert_eq!(errors.right.join("\n"), WELCOME_ART_ERR_RIGHT.join("\n"));
    assert_eq!(errors.brightness, 1.0);

    let warnings = at(
        &mut state,
        frame * 2 + WELCOME_INTRO_FADE_IN + WELCOME_INTRO_HOLD / 2,
    );
    assert_eq!(warnings.right.join("\n"), WELCOME_ART_WRN_RIGHT.join("\n"));
    assert_eq!(warnings.brightness, 1.0);

    let revealing = at(&mut state, frame * 3 + WELCOME_INTRO_REVEAL / 2);
    assert_eq!(revealing.left.join("\n"), WELCOME_ART_LEFT.join("\n"));
    assert!(revealing.brightness > 0.0 && revealing.brightness < 1.0);

    let settled = at(&mut state, frame * 3 + WELCOME_INTRO_REVEAL);
    assert_eq!(settled.left.join("\n"), WELCOME_ART_LEFT.join("\n"));
    assert_eq!(settled.brightness, 1.0);

    let mut plain = TuiState::default();
    let everyday = welcome_mark(&mut plain, false);
    assert_eq!(everyday.left.join("\n"), WELCOME_ART_LEFT.join("\n"));
    assert_eq!(plain.welcome_intro_started_at, None);
}

#[test]
fn wordmark_shadow_ignores_the_background() {
    let expected = wordmark_shadow_color(Theme::dark());

    assert_eq!(wordmark_shadow_color(Theme::plain_for(None)), expected);
    assert_eq!(wordmark_shadow_color(Theme::glass()), expected);
}

use super::*;
use crate::context_tree::{ContextNodeId, ContextTreeOp, ContextTreeState};
use crate::context_view::{
    ContextBlock, ContextBlockId, ContextBlockKind, ContextBlockSource, ContextViewProjection,
};
use crate::tui::events::ToolPendingEvent;
use crate::tui::surface;
use crate::tui::{
    AssistantDeltaEvent, ErrorEvent, PermissionRequestEvent, SessionEvent, ToolFinishedEvent,
    ToolOutcome, ToolStartedEvent, UserMessageEvent,
};
use ratatui::{
    Terminal,
    backend::TestBackend,
    layout::{Position, Rect},
};
use std::collections::BTreeMap;

fn sample_context_state() -> crate::tui::state::ContextPaneState {
    let tree = ContextTreeState::replay(&[ContextTreeOp::CreateNode {
        node_id: ContextNodeId::new("node-1").expect("node id"),
        parent_node_id: Some(ContextNodeId::root()),
        label: Some("Active task".into()),
        purpose: Some("Track current work".into()),
        block_ref: None,
        source_ref: None,
    }])
    .expect("tree");
    let mut blocks = BTreeMap::new();
    let block_id = ContextBlockId::new("block-1").expect("block id");
    blocks.insert(
        block_id.clone(),
        ContextBlock {
            block_id,
            node_id: Some("node-1".into()),
            kind: ContextBlockKind::Note,
            title: "Current plan".into(),
            detail: "Outline next steps".into(),
            source: ContextBlockSource::TranscriptSpan {
                start_sequence: 1,
                end_sequence: 2,
            },
            source_start_sequence: Some(1),
            available_sequence: Some(2),
            protected_reasons: Vec::new(),
        },
    );

    crate::tui::state::ContextPaneState {
        tree,
        view: ContextViewProjection {
            blocks,
            ..ContextViewProjection::default()
        },
        runtime_context: None,
        open_detail: None,
    }
}

fn draw_to_string(state: &mut TuiState, width: u16, height: u16) -> String {
    if state.language.is_none() {
        state.set_language(Some(crate::tui::i18n::Language::En));
    }
    let backend = TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend).expect("test terminal is created");

    terminal
        .draw(|frame| render(frame, state))
        .expect("render succeeds");

    terminal
        .backend()
        .buffer()
        .content()
        .iter()
        .map(|cell| cell.symbol())
        .collect()
}

fn draw_rows(state: &mut TuiState, width: u16, height: u16) -> Vec<String> {
    let backend = TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend).expect("test terminal is created");

    terminal
        .draw(|frame| render(frame, state))
        .expect("render succeeds");

    let buffer = terminal.backend().buffer();
    (0..height)
        .map(|y| {
            (0..width)
                .filter_map(|x| buffer.cell((x, y)).map(|cell| cell.symbol()))
                .collect::<String>()
        })
        .collect()
}

#[test]
fn sidebar_columns_do_not_render_workspace_content() {
    let mut state = TuiState::default();
    state.reasoning_effort_label = Some("leak-probe-reasoning".into());
    state.toggle_sidebar();

    let (width, height) = (layout::SIDEBAR_MIN_TERMINAL_WIDTH, 30u16);
    let rows = draw_rows(&mut state, width, height);
    let sidebar = layout::split_sidebar_layout(Rect::new(0, 0, width, height), true)
        .sidebar
        .expect("forced sidebar is rendered");
    assert_eq!(state.last_sidebar_bounds, sidebar);
    assert_eq!(sidebar.right(), width);

    for (index, row) in rows.iter().enumerate() {
        let panel: String = row.chars().skip(sidebar.x as usize).collect();
        assert!(
            !panel.contains("leak-probe-reasoning"),
            "row {index} leaked workspace content into the panel: {row}"
        );
    }
}

#[test]
fn sidebar_is_not_rendered_below_the_minimum_terminal_width() {
    let mut forced = TuiState::default();
    forced.toggle_sidebar();
    let mut hidden = TuiState::default();
    hidden.sidebar_hidden = true;

    let narrow = layout::SIDEBAR_WIDTH * 3 - 1;
    assert_eq!(
        draw_rows(&mut forced, narrow, 30),
        draw_rows(&mut hidden, narrow, 30)
    );
    assert_ne!(
        draw_rows(&mut forced, narrow + 1, 30),
        draw_rows(&mut hidden, narrow + 1, 30)
    );
}

#[test]
fn sidebar_columns_do_not_expose_transcript_hits() {
    let mut state = TuiState::default();
    state.toggle_sidebar();
    state.apply_event(SessionEvent::ToolPending(ToolPendingEvent::new(
        "call-1",
        "shell__exec",
    )));

    let (width, height) = (layout::SIDEBAR_MIN_TERMINAL_WIDTH, 30u16);
    draw_rows(&mut state, width, height);
    let sidebar = layout::split_sidebar_layout(Rect::new(0, 0, width, height), true)
        .sidebar
        .expect("forced sidebar is rendered");
    let transcript = state.last_transcript_area;
    assert_eq!(state.last_sidebar_bounds, sidebar);
    assert!(
        transcript.width > 0 && sidebar.x >= transcript.right() && sidebar.right() == width,
        "expected a sidebar next to a live transcript: {transcript:?} {sidebar:?}"
    );

    let mut visible_hits = 0usize;
    for row in transcript.top()..transcript.bottom() {
        for col in transcript.left()..sidebar.x {
            visible_hits += usize::from(state.transcript_click_target(col, row).is_some());
        }
        for col in sidebar.left()..sidebar.right() {
            assert!(
                state.transcript_click_target(col, row).is_none(),
                "row {row} col {col} kept a transcript click target under the panel"
            );
            assert!(
                state.map_mouse_to_anchor(col, row).is_none(),
                "row {row} col {col} kept a selection anchor under the panel"
            );
        }
    }
    assert!(
        visible_hits > 0,
        "the probe needs a live transcript hit next to the panel"
    );
}

#[test]
fn child_read_only_status_centers_in_the_cap_excluded_surface() {
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

    let rows = draw_rows(&mut state, 100, 24);
    let workspace = layout::workspace_area(Rect::new(0, 0, 100, 24));
    let metrics = layout::workspace_metrics(workspace, "", &[], false, false, true, 0, false);
    let [_transcript, _gap, _slash, composer, footer] =
        layout::split_workspace_layout(workspace, metrics);
    let surface_center = composer.y + composer.height.saturating_sub(1) / 2;
    let cap_row = composer.bottom().saturating_sub(1);

    assert_eq!(composer.height, 5);
    assert_eq!(surface_center.saturating_sub(composer.y), 2);
    assert_eq!(cap_row.saturating_sub(surface_center), 2);
    assert!(
        rows[composer.y as usize].contains(surface::PROMPT_TOP_LEFT_GLYPH)
            || rows[composer.y as usize].contains(surface::PROMPT_TOP_CAP_GLYPH),
        "{rows:?}"
    );
    assert!(
        rows[surface_center as usize].contains("explorer"),
        "{rows:?}"
    );
    assert!(
        rows[..surface_center as usize]
            .iter()
            .rev()
            .take(1)
            .all(|row| !row.contains("explorer")),
        "{rows:?}"
    );
    assert!(
        !rows[(surface_center + 1) as usize].contains("explorer"),
        "{rows:?}"
    );
    assert!(
        rows[cap_row as usize].contains(surface::PROMPT_BOTTOM_LEFT_GLYPH)
            || rows[cap_row as usize].contains(surface::PROMPT_BOTTOM_CAP_GLYPH),
        "{rows:?}"
    );
    assert_eq!(footer.y, composer.bottom());
}

#[test]
fn notice_geometry_adapts_width_and_height_to_content_and_area() {
    let area = Rect::new(0, 0, 100, 30);
    let short = notice_geometry(area, "ok");
    let long = notice_geometry(
        area,
        "a long notice that wraps over several rows "
            .repeat(4)
            .as_str(),
    );

    assert_eq!(short.width, 14);
    assert_eq!(short.height, 3);
    assert!(long.width > short.width);
    assert!(long.width <= 60);
    assert!(long.height > short.height);
    assert!(long.height <= 12);
}

#[test]
fn long_toast_renders_wrapped_rows_and_visible_truncation() {
    let mut state = TuiState::default();
    state.mark_session_active();
    state.show_toast(
        "A very long toast message that must wrap and be visibly truncated when the transcript is short.",
        ToastKind::Info,
    );

    let rendered = draw_to_string(&mut state, 80, 24);

    assert!(rendered.contains("A very long toast"), "{rendered}");
    assert!(rendered.contains('…'), "{rendered}");
}

#[test]
fn zh_cn_narrow_rendering_stays_within_terminal_bounds() {
    let mut state = TuiState::default();
    state.set_language(Some(crate::tui::i18n::Language::ZhCn));
    state.mark_session_active();
    state.show_toast("中文提示：请继续操作", ToastKind::Info);
    let rows = draw_rows(&mut state, 40, 12);
    assert_eq!(rows.len(), 12);
    assert!(rows.iter().all(|row| !row.is_empty()));
}

#[test]
fn notice_geometry_handles_narrow_terminals_and_wide_characters() {
    let narrow = notice_geometry(Rect::new(0, 0, 16, 10), "hello world");
    let wide = notice_geometry(Rect::new(0, 0, 40, 20), "你好世界");

    assert_eq!(narrow.width, 14);
    assert!(narrow.width <= 16);
    assert!(wide.width >= 16);
    assert!(wide.height >= 3);
}

#[test]
fn notice_geometry_caps_wrapped_content_and_marks_truncation() {
    let area = Rect::new(0, 0, 80, 10);
    let geometry = notice_geometry(area, &"long message ".repeat(100));
    let lines = notice_message_lines(
        &"long message ".repeat(100),
        geometry.width.saturating_sub(4),
        geometry.height.saturating_sub(2),
        Style::default(),
    );

    assert_eq!(geometry.width, 47);
    assert_eq!(geometry.height, 3);
    assert_eq!(lines.len(), 1);
    assert!(
        lines
            .last()
            .is_some_and(|line| line.to_string().contains('…'))
    );
}

#[test]
fn empty_welcome_view_renders_without_panic() {
    let mut state = TuiState::new("gpt-5.5", "gpt-5.5", "default");

    let rendered = draw_to_string(&mut state, 80, 20);
    assert!(rendered.contains("/resume"), "{rendered}");

    let tiny = draw_to_string(&mut state, 10, 2);
    assert!(!tiny.is_empty());
}

#[test]
fn pending_permission_prompt_displays_hint_and_tool_summary() {
    let mut state = TuiState::default();
    let mut request = PermissionRequestEvent::new("call-1", "shell__exec", "cargo test all");
    request.arguments = Some("cargo test".into());
    request.rationale = Some("tests need confirmation".into());
    state.apply_event(SessionEvent::PermissionRequested(request));

    let rendered = draw_to_string(&mut state, 96, 24);

    assert!(
        rendered.contains("Approve tool") || rendered.contains("Run command"),
        "{rendered}"
    );
    assert!(
        rendered.contains("Allow once") || rendered.contains("allow once"),
        "{rendered}"
    );
    assert!(
        rendered.contains("Reject") || rendered.contains("reject"),
        "{rendered}"
    );
    assert!(rendered.contains("cargo test all"), "{rendered}");
    assert!(!rendered.contains("message letcode"), "{rendered}");
    assert!(!rendered.contains("args"), "{rendered}");
}

#[test]
fn pending_question_renders_as_bottom_panel_not_centered_modal() {
    let mut state = TuiState::default();
    state.mark_session_active();
    state.pending_question = Some(crate::tui::state::PendingQuestionState::new(
        crate::tool::QuestionRequest {
            questions: vec![crate::tool::QuestionSpec {
                question: "Choose a mode".into(),
                header: "Mode".into(),
                options: vec![
                    crate::tool::QuestionOption {
                        label: "Fast".into(),
                        description: "Fast path".into(),
                    },
                    crate::tool::QuestionOption {
                        label: "Safe".into(),
                        description: "Safe path".into(),
                    },
                ],
                multiple: false,
            }],
        },
        None,
    ));

    let rendered = draw_to_string(&mut state, 100, 24);

    assert!(rendered.contains("Choose a mode"), "{rendered}");
    assert!(rendered.contains("1. Fast"), "{rendered}");
    assert!(
        rendered.contains(surface::PROMPT_BOTTOM_LEFT_GLYPH),
        "{rendered}"
    );
    assert!(
        rendered.contains(surface::PROMPT_BOTTOM_CAP_GLYPH),
        "{rendered}"
    );
    assert!(!rendered.contains("message letcode"), "{rendered}");
}

#[test]
fn confirm_body_keeps_answers_on_independent_indented_lines() {
    let mut state = TuiState::default();
    state.mark_session_active();
    let mut question = crate::tui::state::PendingQuestionState::new(
        crate::tool::QuestionRequest {
            questions: vec![
                crate::tool::QuestionSpec {
                    question: "Pick features".into(),
                    header: "Features".into(),
                    options: vec![
                        crate::tool::QuestionOption {
                            label: "Alpha".into(),
                            description: "A".into(),
                        },
                        crate::tool::QuestionOption {
                            label: "Beta".into(),
                            description: "B".into(),
                        },
                    ],
                    multiple: true,
                },
                crate::tool::QuestionSpec {
                    question: "Pick mode".into(),
                    header: "Mode".into(),
                    options: vec![],
                    multiple: false,
                },
            ],
        },
        None,
    );
    question.questions[0].selected_labels = vec!["Alpha".into(), "Beta".into()];
    question.focus_tab(2);
    state.pending_question = Some(question);

    let rendered = draw_to_string(&mut state, 100, 24);

    assert!(rendered.contains("1. Features"), "{rendered}");
    assert!(rendered.contains("  - Alpha"), "{rendered}");
    assert!(rendered.contains("  - Beta"), "{rendered}");
    assert!(rendered.contains("2. Mode"), "{rendered}");
    assert!(rendered.contains("(not answered)"), "{rendered}");
    assert!(!rendered.contains("Alpha, Beta"), "{rendered}");
}

#[test]
fn narrow_confirm_tabs_wrap_into_multiple_rows() {
    let question = crate::tui::state::PendingQuestionState::new(
        crate::tool::QuestionRequest {
            questions: (0..3)
                .map(|index| crate::tool::QuestionSpec {
                    question: format!("Question {index}"),
                    header: format!("Long tab {index}"),
                    options: vec![],
                    multiple: true,
                })
                .collect(),
        },
        None,
    );
    assert!(question_tab_row_count(&question, 12) > 1);
    assert_eq!(question_tab_row_count(&question, 200), 1);
}

#[test]
fn oversized_tab_label_is_truncated_without_exceeding_row_width() {
    let question = crate::tui::state::PendingQuestionState::new(
        crate::tool::QuestionRequest {
            questions: vec![crate::tool::QuestionSpec {
                question: "Question".into(),
                header: "A header much longer than the panel".into(),
                options: vec![],
                multiple: true,
            }],
        },
        None,
    );
    let width = 7;
    let lines = question_tab_lines(&question, width, Theme::default());

    assert_eq!(question_tab_row_count(&question, width), lines.len());
    assert!(!lines.is_empty());
    assert!(
        lines
            .iter()
            .all(|line| display_width(&line.to_string()) <= width)
    );
}

#[test]
fn very_short_confirm_panel_skips_zero_height_body_safely() {
    let mut state = TuiState::default();
    state.mark_session_active();
    let mut question = crate::tui::state::PendingQuestionState::new(
        crate::tool::QuestionRequest {
            questions: vec![crate::tool::QuestionSpec {
                question: "Question".into(),
                header: "Header".into(),
                options: vec![],
                multiple: true,
            }],
        },
        None,
    );
    question.focus_tab(1);
    question.confirm_scroll = usize::MAX;
    state.pending_question = Some(question);

    let rendered = draw_to_string(&mut state, 12, 5);
    assert!(!rendered.is_empty());
    let question = state
        .pending_question
        .as_ref()
        .expect("question remains pending");
    assert!(question.confirm_scroll <= question.confirm_scroll_max());
}

#[test]
fn short_question_panel_uses_content_height_instead_of_the_full_workspace() {
    let question = crate::tui::state::PendingQuestionState::new(
        crate::tool::QuestionRequest {
            questions: vec![crate::tool::QuestionSpec {
                question: "Choose a mode".into(),
                header: "Mode".into(),
                options: vec![
                    crate::tool::QuestionOption {
                        label: "Fast".into(),
                        description: "Finish quickly".into(),
                    },
                    crate::tool::QuestionOption {
                        label: "Careful".into(),
                        description: "Review each step".into(),
                    },
                ],
                multiple: false,
            }],
        },
        None,
    );
    let workspace = Rect::new(2, 0, 96, 23);
    let content_rows =
        question_full_row_count(&question, question_content_width(workspace.width, false));
    let height = question_composer_height(&question, workspace, false);

    assert_eq!(height, content_rows as u16 + 4);
    assert_eq!(
        question_composer_height(&question, workspace, true),
        height + 1
    );
    assert!(height < workspace.height.saturating_sub(2));
}

#[test]
fn detailed_question_panel_grows_then_uses_compact_viewport_at_workspace_limit() {
    let mut question = crate::tui::state::PendingQuestionState::new(
        crate::tool::QuestionRequest {
            questions: vec![crate::tool::QuestionSpec {
                question: "A long question that wraps on a narrow terminal and needs room for each available choice".into(),
                header: "Mode".into(),
                options: (0..6)
                    .map(|index| crate::tool::QuestionOption {
                        label: format!("Option {index} with a long label"),
                        description: "A detailed description that also wraps in a narrow viewport".into(),
                    })
                    .collect(),
                multiple: true,
            }],
        },
        Some("Child question".into()),
    );
    question.active_row.focus(6);
    question.begin_custom_edit();
    question.questions[0].custom_edit_text = "custom answer kept visible".into();
    question.questions[0].custom_edit_cursor = question.questions[0].custom_edit_text.len();
    let workspace = Rect::new(2, 0, 44, 13);

    assert_eq!(question_composer_height(&question, workspace, false), 11);
    assert!(question_full_row_count(&question, question_content_width(workspace.width, false)) > 7);

    let mut state = TuiState::default();
    state.mark_session_active();
    state.pending_question = Some(question);
    let rendered = draw_to_string(&mut state, 48, 14);
    assert!(rendered.contains("kept visible"), "{rendered}");
    assert!(rendered.contains('▏'), "{rendered}");
    assert!(rendered.contains("save answer"), "{rendered}");
}

#[test]
fn single_select_question_keeps_custom_editor_and_submit_action_visible() {
    let mut state = TuiState::default();
    state.mark_session_active();
    let mut question = crate::tui::state::PendingQuestionState::new(
        crate::tool::QuestionRequest {
            questions: vec![crate::tool::QuestionSpec {
                question: "Choose a mode".into(),
                header: "Mode".into(),
                options: vec![
                    crate::tool::QuestionOption {
                        label: "Fast".into(),
                        description: "Finish quickly".into(),
                    },
                    crate::tool::QuestionOption {
                        label: "Careful".into(),
                        description: "Review each step".into(),
                    },
                ],
                multiple: false,
            }],
        },
        None,
    );
    question.active_row.focus(2);
    question.begin_custom_edit();
    question.questions[0].custom_edit_text = "Tailored plan".into();
    question.questions[0].custom_edit_cursor = "Tailored plan".len();
    state.pending_question = Some(question);

    let rendered = draw_to_string(&mut state, 100, 24);

    assert!(rendered.contains("Finish quickly"), "{rendered}");
    assert!(rendered.contains("Review each step"), "{rendered}");
    assert!(rendered.contains("Type your own answer"), "{rendered}");
    assert!(rendered.contains("Tailored plan"), "{rendered}");
    assert!(rendered.contains("submit answer"), "{rendered}");
    assert!(rendered.contains("cancel edit"), "{rendered}");
    assert!(!rendered.contains("enter save"), "{rendered}");
}

#[test]
fn small_question_viewport_keeps_active_custom_row_within_bounds() {
    let mut state = TuiState::default();
    state.mark_session_active();
    let mut question = crate::tui::state::PendingQuestionState::new(
        crate::tool::QuestionRequest {
            questions: vec![crate::tool::QuestionSpec {
                question: "Choose a mode".into(),
                header: "Mode".into(),
                options: vec![
                    crate::tool::QuestionOption {
                        label: "Fast".into(),
                        description: "Finish quickly".into(),
                    },
                    crate::tool::QuestionOption {
                        label: "Careful".into(),
                        description: "Review each step".into(),
                    },
                ],
                multiple: false,
            }],
        },
        Some("Child question".into()),
    );
    question.active_row.focus(2);
    question.begin_custom_edit();
    question.questions[0].custom_edit_text = "Own answer".into();
    question.questions[0].custom_edit_cursor = "Own answer".len();
    state.pending_question = Some(question);

    let rendered = draw_to_string(&mut state, 80, 10);

    assert!(rendered.contains("Type your own answer"), "{rendered}");
    assert!(rendered.contains("Own answer"), "{rendered}");
    assert!(rendered.contains("submit answer"), "{rendered}");
}

#[test]
fn height_seven_terminal_keeps_question_cursor_above_footer() {
    let mut state = TuiState::default();
    state.mark_session_active();
    let mut question = crate::tui::state::PendingQuestionState::new(
        crate::tool::QuestionRequest {
            questions: vec![crate::tool::QuestionSpec {
                question: "Choose a mode".into(),
                header: "Mode".into(),
                options: vec![crate::tool::QuestionOption {
                    label: "Fast".into(),
                    description: "Finish quickly".into(),
                }],
                multiple: false,
            }],
        },
        None,
    );
    question.active_row.focus(1);
    question.begin_custom_edit();
    question.questions[0].custom_edit_text = "answer".into();
    question.questions[0].custom_edit_cursor = "answer".len();
    state.pending_question = Some(question);

    // This goes through render -> workspace_area (height 6) -> split_workspace_layout.
    let rendered = draw_to_string(&mut state, 80, 7);

    assert!(rendered.contains("answer"), "{rendered}");
    assert!(rendered.contains('▏'), "{rendered}");
}

#[test]
fn compact_question_keeps_child_origin_and_active_tab_context() {
    let mut state = TuiState::default();
    state.mark_session_active();
    let request = crate::tool::QuestionRequest {
        questions: (1..=3)
            .map(|index| crate::tool::QuestionSpec {
                question: format!("Question {index}"),
                header: if index == 2 {
                    "Header".into()
                } else {
                    format!("Question {index}")
                },
                options: vec![crate::tool::QuestionOption {
                    label: "Option with a deliberately long label".into(),
                    description: "Description that forces compact question rendering".into(),
                }],
                multiple: false,
            })
            .collect(),
    };
    let mut question = crate::tui::state::PendingQuestionState::new(request, Some("Child".into()));
    question.active_tab = 1;
    question.active_row.focus(1);
    question.begin_custom_edit();
    question.questions[1].custom_edit_text = "custom".into();
    question.questions[1].custom_edit_cursor = "custom".len();
    state.pending_question = Some(question);

    let rendered = draw_to_string(&mut state, 80, 10);

    assert!(rendered.contains("Child · 2/3 Header"), "{rendered}");
    assert!(rendered.contains("custom"), "{rendered}");
    assert!(rendered.contains('▏'), "{rendered}");
}

#[test]
fn multi_question_tabs_descriptions_and_custom_editor_fit_a_normal_terminal() {
    let mut state = TuiState::default();
    state.mark_session_active();
    let mut question = crate::tui::state::PendingQuestionState::new(
        crate::tool::QuestionRequest {
            questions: vec![
                crate::tool::QuestionSpec {
                    question: "Choose the delivery plan".into(),
                    header: "Plan".into(),
                    options: vec![
                        crate::tool::QuestionOption {
                            label: "Fast".into(),
                            description: "Ship the smallest safe change".into(),
                        },
                        crate::tool::QuestionOption {
                            label: "Careful".into(),
                            description: "Review every edge case".into(),
                        },
                    ],
                    multiple: false,
                },
                crate::tool::QuestionSpec {
                    question: "Choose the rollout".into(),
                    header: "Rollout".into(),
                    options: vec![],
                    multiple: false,
                },
            ],
        },
        None,
    );
    question.active_row.focus(2);
    question.begin_custom_edit();
    question.questions[0].custom_edit_text = "A tailored staged rollout".into();
    question.questions[0].custom_edit_cursor = question.questions[0].custom_edit_text.len();
    state.pending_question = Some(question);

    let rendered = draw_to_string(&mut state, 100, 24);

    for expected in [
        "Plan",
        "Rollout",
        "Confirm",
        "Ship the smallest safe change",
        "Review every edge case",
        "A tailored staged rollout",
        "next question",
    ] {
        assert!(
            rendered.contains(expected),
            "missing {expected}: {rendered}"
        );
    }
}

#[test]
fn narrow_wrapped_question_keeps_custom_editor_and_footer_visible() {
    let mut state = TuiState::default();
    state.mark_session_active();
    let mut question = crate::tui::state::PendingQuestionState::new(
        crate::tool::QuestionRequest {
            questions: vec![crate::tool::QuestionSpec {
                question: "A deliberately long question that cannot fit on one narrow terminal row"
                    .into(),
                header: "Long".into(),
                options: vec![crate::tool::QuestionOption {
                    label: "A deliberately long option label".into(),
                    description: "A deliberately long option description that wraps several times"
                        .into(),
                }],
                multiple: false,
            }],
        },
        Some("Child question from a nested session".into()),
    );
    question.active_row.focus(1);
    question.begin_custom_edit();
    question.questions[0].custom_edit_text = "narrow custom answer".into();
    question.questions[0].custom_edit_cursor = question.questions[0].custom_edit_text.len();
    state.pending_question = Some(question);

    let rendered = draw_to_string(&mut state, 48, 10);

    assert!(rendered.contains("custom answer"), "{rendered}");
    assert!(rendered.contains('▏'), "{rendered}");
    assert!(rendered.contains("submit answer"), "{rendered}");
}

#[test]
fn footer_enter_detail_tracks_the_actual_question_transition() {
    let request = |questions: usize, multiple: bool| crate::tool::QuestionRequest {
        questions: (0..questions)
            .map(|index| crate::tool::QuestionSpec {
                question: format!("Question {index}"),
                header: format!("Q{index}"),
                options: vec![crate::tool::QuestionOption {
                    label: "Option".into(),
                    description: "Description".into(),
                }],
                multiple,
            })
            .collect(),
    };
    let mut cases = Vec::new();

    let mut unanswered_confirm =
        crate::tui::state::PendingQuestionState::new(request(2, false), None);
    unanswered_confirm.focus_tab(2);
    cases.push((unanswered_confirm, "go to unanswered"));

    let mut answered_confirm =
        crate::tui::state::PendingQuestionState::new(request(2, false), None);
    answered_confirm
        .questions
        .iter_mut()
        .for_each(|item| item.selected_labels.push("Option".into()));
    answered_confirm.focus_tab(2);
    cases.push((answered_confirm, "submit"));

    let mut empty_custom = crate::tui::state::PendingQuestionState::new(request(1, false), None);
    empty_custom.active_row.focus(1);
    empty_custom.begin_custom_edit();
    cases.push((empty_custom, "close edit"));

    let mut single_custom = crate::tui::state::PendingQuestionState::new(request(1, false), None);
    single_custom.active_row.focus(1);
    single_custom.begin_custom_edit();
    single_custom.questions[0].custom_edit_text = "answer".into();
    cases.push((single_custom, "submit answer"));

    let mut multi_question_custom =
        crate::tui::state::PendingQuestionState::new(request(2, false), None);
    multi_question_custom.active_row.focus(1);
    multi_question_custom.begin_custom_edit();
    multi_question_custom.questions[0].custom_edit_text = "answer".into();
    cases.push((multi_question_custom, "next question"));

    let mut final_multi_question_custom =
        crate::tui::state::PendingQuestionState::new(request(2, false), None);
    final_multi_question_custom.active_tab = 1;
    final_multi_question_custom.active_row.focus(1);
    final_multi_question_custom.begin_custom_edit();
    final_multi_question_custom.questions[1].custom_edit_text = "answer".into();
    cases.push((final_multi_question_custom, "review answers"));

    let mut active_custom = crate::tui::state::PendingQuestionState::new(request(1, false), None);
    active_custom.active_row.focus(1);
    cases.push((active_custom, "type answer"));

    cases.push((
        crate::tui::state::PendingQuestionState::new(request(1, true), None),
        "toggle",
    ));
    cases.push((
        crate::tui::state::PendingQuestionState::new(request(1, false), None),
        "choose & submit",
    ));

    let mut final_multi_question_option =
        crate::tui::state::PendingQuestionState::new(request(2, false), None);
    final_multi_question_option.active_tab = 1;
    cases.push((final_multi_question_option, "choose & review"));

    for (question, expected) in cases {
        assert_eq!(question_enter_detail(&question), expected);
    }
}

#[test]
fn custom_answer_footer_matches_the_commit_transition() {
    let request = |questions: usize, multiple: bool| crate::tool::QuestionRequest {
        questions: (0..questions)
            .map(|index| crate::tool::QuestionSpec {
                question: format!("Question {index}"),
                header: format!("Q{index}"),
                options: vec![crate::tool::QuestionOption {
                    label: "Option".into(),
                    description: "Description".into(),
                }],
                multiple,
            })
            .collect(),
    };
    let mut single_multi = crate::tui::state::PendingQuestionState::new(request(1, true), None);
    single_multi.active_row.focus(1);
    single_multi.begin_custom_edit();
    single_multi.questions[0].custom_edit_text = "answer".into();

    let mut next_question = crate::tui::state::PendingQuestionState::new(request(2, false), None);
    next_question.active_row.focus(1);
    next_question.begin_custom_edit();
    next_question.questions[0].custom_edit_text = "answer".into();

    let mut review = crate::tui::state::PendingQuestionState::new(request(2, false), None);
    review.active_tab = 1;
    review.active_row.focus(1);
    review.begin_custom_edit();
    review.questions[1].custom_edit_text = "answer".into();

    for (question, action, detail) in [
        (
            single_multi,
            crate::tui::state::QuestionAdvance::None,
            "save answer",
        ),
        (
            next_question,
            crate::tui::state::QuestionAdvance::Advanced,
            "next question",
        ),
        (
            review,
            crate::tui::state::QuestionAdvance::Advanced,
            "review answers",
        ),
    ] {
        assert_eq!(question_enter_detail(&question), detail);
        let mut committed = question.clone();
        assert_eq!(committed.commit_custom_answer(), action);
    }
}

#[test]
fn group_16_context_picker_renders_only_canonical_surviving_detail() {
    let snapshot = crate::runtime_context::group_16_runtime_snapshot();
    let context = crate::runtime_context::RuntimeActiveContext::try_from(&snapshot)
        .expect("canonical runtime context");
    let mut state = TuiState::new("gpt-5.5", "gpt-5.5", "default");
    state.apply_event(SessionEvent::RuntimeContextUpdated(
        crate::tui::events::RuntimeContextUpdatedEvent {
            context,
            disposition: crate::tui::events::RuntimeContextDisposition::Advance,
        },
    ));
    let items = crate::tui::state::context_dialog_items(state.active_context());
    let mut dialog = crate::tui::state::DialogState::new(
        crate::tui::state::DialogKind::ContextPicker,
        "Context",
        None,
        items,
    );
    dialog.selected = dialog
        .items
        .iter()
        .position(|item| item.id == "block:active-block")
        .expect("canonical active block item");
    state.open_dialog(dialog);
    state.sync_context_picker_preview();

    let rendered = draw_to_string(&mut state, 110, 30);

    assert!(rendered.contains("CANONICAL ACTIVE TITLE"), "{rendered}");
    assert!(rendered.contains("CANONICAL ACTIVE CONTENT"), "{rendered}");
    assert!(rendered.contains("CURRENT-TAIL-SENTINEL"), "{rendered}");
    assert!(!rendered.contains("RETIRED-RAW-SENTINEL"), "{rendered}");
    assert!(!rendered.contains("RETIRED-FOLDED-SENTINEL"), "{rendered}");
}

#[test]
fn config_editor_renders_grouped_fields_and_footer() {
    let mut state = TuiState::default();
    let items = vec![
        crate::tui::state::DialogItem::new(
            "active_provider",
            "Active provider",
            Some("openai".into()),
        )
        .with_section("Provider"),
        crate::tui::state::DialogItem::new(
            "permissions.mode",
            "Permission mode",
            Some("default".into()),
        )
        .with_section("Permission")
        .with_right_detail("▸"),
    ];
    state.open_dialog(crate::tui::state::DialogState::new(
        crate::tui::state::DialogKind::ConfigEditor,
        "Configuration",
        None,
        items,
    ));

    let rendered = draw_to_string(&mut state, 100, 24);

    assert!(rendered.contains("Provider"), "{rendered}");
    assert!(rendered.contains("Permission"), "{rendered}");
    assert!(rendered.contains("Enter"), "{rendered}");
}

fn chinese_config_state() -> TuiState {
    use crate::tui::state::{ConfigFieldRef, DialogItem, DialogKind, DialogState};
    let mut state = TuiState::default();
    state.set_language(Some(crate::tui::i18n::Language::ZhCn));
    let mut dialog = DialogState::new(
        DialogKind::ConfigEditor,
        state.t("config.title"),
        Some("配置文件说明".repeat(40)),
        vec![
            DialogItem::new("prompt", "ActualConfigField", Some("value".into()))
                .with_section("设置"),
        ],
    );
    dialog.config_fields = vec![ConfigFieldRef::Field(vec!["prompt".into()])];
    dialog.config_dirty = true;
    state.open_dialog(dialog);
    state
}

fn config_screen_rows(state: &mut TuiState, width: u16, height: u16) -> Vec<String> {
    let mut terminal = Terminal::new(TestBackend::new(width, height)).expect("terminal");
    terminal.draw(|frame| render(frame, state)).expect("draw");
    let buffer = terminal.backend().buffer();
    (0..height)
        .map(|y| {
            let mut row = String::new();
            let mut x = 0;
            while x < width {
                let symbol = buffer[(x, y)].symbol();
                row.push_str(symbol);
                x += display_width(symbol).max(1) as u16;
            }
            row
        })
        .collect()
}

#[test]
fn config_render_small_windows_preserve_fields_and_actions() {
    for height in [8, 14, 24] {
        let mut state = chinese_config_state();
        let rows = config_screen_rows(&mut state, 80, height);
        for text in [
            "ActualConfigField",
            "Ctrl-S 保存",
            "Esc 关闭",
            &state.t("config.unsaved"),
        ] {
            assert!(
                rows.iter().any(|row| row.contains(text)),
                "{height}: {rows:?}"
            );
        }
    }
}

#[test]
fn config_render_close_confirmation_keeps_the_selection_visible() {
    for height in [7, 24] {
        for selected in 0..3 {
            let mut state = chinese_config_state();
            state.dialog_mut().expect("dialog").config_close_selected = Some(selected);
            let options = [
                state.t("config.save_and_close"),
                state.t("config.discard_and_close"),
                state.t("config.continue_editing"),
            ];
            let rows = config_screen_rows(&mut state, 80, height);
            assert!(
                rows.iter()
                    .any(|row| row.contains(&format!("● {}", options[selected]))),
                "{height}: {rows:?}"
            );
            assert!(
                rows.iter()
                    .any(|row| row.contains("Enter 选择") && row.contains("Esc 取消")),
                "{height}: {rows:?}"
            );
            if height >= 14 {
                for option in options {
                    assert!(
                        rows.iter().any(|row| row.contains(&option)),
                        "{height}: {rows:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn config_render_error_preserves_the_diagnostic_and_fields() {
    for height in [14, 24] {
        let mut state = chinese_config_state();
        let error = "cannot save config at /very/long/project/path/letcode.toml because permissions.mode is invalid: expected safe/default/auto/yolo; ERROR-END";
        state.dialog_mut().expect("config dialog").config_error = Some(error.into());
        let rows = config_screen_rows(&mut state, 80, height);
        for detail in [
            "permissions.mode",
            "invalid",
            "ERROR-END",
            "ActualConfigField",
        ] {
            assert!(
                rows.iter().any(|row| row.contains(detail)),
                "{height}: {rows:?}"
            );
        }
    }
}

#[test]
fn permission_dialog_uses_picker_style() {
    let mut state = TuiState::new("gpt-5.5", "GPT-5.5", "default");
    let mut dialog = crate::tui::state::DialogState::new(
        crate::tui::state::DialogKind::PermissionPicker,
        "Permission mode",
        Some("Select how much freedom the agent has when using tools".into()),
        vec![
            crate::tui::state::DialogItem::new("safe", "Safe", Some("Ask before all tools".into())),
            crate::tui::state::DialogItem::new(
                "default",
                "Default",
                Some("Allow read/preview, ask for risky tools".into()),
            ),
            crate::tui::state::DialogItem::new(
                "yolo",
                "YOLO",
                Some("Allow write and command tools without asking".into()),
            ),
        ],
    );
    dialog.selected = 1;
    state.open_dialog(dialog);

    let rendered = draw_to_string(&mut state, 100, 24);

    assert!(rendered.contains("Permission mode"), "{rendered}");
    assert!(rendered.contains("Select how much freedom"), "{rendered}");
    assert!(!rendered.contains("Search"), "{rendered}");
    assert!(rendered.contains("Default"), "{rendered}");
    assert!(rendered.contains("Allow read/preview"), "{rendered}");
    assert!(rendered.contains("esc"), "{rendered}");
}
