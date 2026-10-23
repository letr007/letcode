use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use ratatui::{
    Frame,
    layout::{Alignment, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Clear, Paragraph, Wrap},
};

use super::{
    assistant_fade,
    components::{choice, composer, dialog, footer, layout, sidebar, slash_panel, transcript},
    measure::{display_width, wrap_text_to_width, wrapped_row_count},
    state::{ToastKind, TuiState},
    surface,
    theme::Theme,
};

const WELCOME_ART_LEFT: &[&str] = &[
    "▄          ▄  ",
    "█    █▀▀█ ▀█▀▀",
    "█    █▀▀▀  █  ",
    "▀▀▀▀ ▀▀▀▀  ▀▀▀",
];
const WELCOME_ART_RIGHT: &[&str] = &[
    "             ▄     ",
    "█▀▀▀ █▀▀█ █▀▀█ █▀▀█",
    "█    █  █ █  █ █▀▀▀",
    "▀▀▀▀ ▀▀▀▀ ▀▀▀▀ ▀▀▀▀",
];
const WELCOME_ART_1024_LEFT: &[&str] = &["  ▄  ▄▄▄▄", "  █  █  █", "  █  █  █", "  █  █▄▄█"];
const WELCOME_ART_1024_RIGHT: &[&str] = &["▄▄▄▄ ▄  ▄", "   █ █  █", "█▀▀▀ ▀▀▀█", "█▄▄▄    █"];
const WELCOME_ART_ERR_LEFT: &[&str] = &["▄▄▄▄", "█  █", "█  █", "█▄▄█"];
const WELCOME_ART_ERR_RIGHT: &[&str] = &[
    "   ▄▄▄▄ ▄▄▄▄ ▄▄▄▄",
    "   █    █  █ █  █",
    "   █▀▀  █▀▀█ █▀▀█",
    "   █▄▄▄ █  █ █  █",
];
const WELCOME_ART_WRN_LEFT: &[&str] = &["▄▄▄▄", "█  █", "█  █", "█▄▄█"];
const WELCOME_ART_WRN_RIGHT: &[&str] = &[
    "   ▄  ▄ ▄▄▄▄ ▄  ▄",
    "   █  █ █  █ █▄ █",
    "   █  █ █▀▀█ █ ▀█",
    "   ▀▄▄▀ █  █ █  █",
];
const WELCOME_ART_ROWS: u16 = 4;
const WELCOME_INTRO_FRAMES: [(&[&str], &[&str]); 3] = [
    (WELCOME_ART_1024_LEFT, WELCOME_ART_1024_RIGHT),
    (WELCOME_ART_ERR_LEFT, WELCOME_ART_ERR_RIGHT),
    (WELCOME_ART_WRN_LEFT, WELCOME_ART_WRN_RIGHT),
];
const WELCOME_INTRO_FADE_IN: Duration = Duration::from_millis(400);
const WELCOME_INTRO_HOLD: Duration = Duration::from_millis(600);
const WELCOME_INTRO_FADE_OUT: Duration = Duration::from_millis(400);
const WELCOME_INTRO_REVEAL: Duration = Duration::from_millis(300);
const EXIT_EPILOGUE_TITLE_CHARS: usize = 50;

/// Plain-text exit banner for the restored main screen (logo + resume hint).
pub(crate) fn format_exit_epilogue(session_id: &str, session_title: Option<&str>) -> String {
    const INDENT: &str = "  ";
    let mut lines = WELCOME_ART_LEFT
        .iter()
        .zip(WELCOME_ART_RIGHT.iter())
        .map(|(left, right)| format!("{INDENT}{left} {right}"))
        .collect::<Vec<_>>();
    lines.push(String::new());

    let title = session_title
        .map(str::trim)
        .filter(|title| !title.is_empty())
        .unwrap_or(session_id);
    let title = truncate_chars(title, EXIT_EPILOGUE_TITLE_CHARS);
    lines.push(format!("{INDENT}{:<10}{title}", "Session"));
    lines.push(format!(
        "{INDENT}{:<10}letcode resume {session_id}",
        "Continue"
    ));
    lines.push(String::new());
    lines.join("\n")
}

fn truncate_chars(value: &str, max_chars: usize) -> String {
    let mut chars = value.chars();
    let kept = chars.by_ref().take(max_chars).collect::<String>();
    if chars.next().is_some() {
        format!("{kept}…")
    } else {
        kept
    }
}
/// Render the TUI from the current state using ratatui widgets only.
///
/// Rendering may refresh viewport bookkeeping, but it never invokes tools, resolves permissions,
/// persists transcripts, or mutates runtime/business state.
pub fn render(frame: &mut Frame<'_>, state: &mut TuiState) {
    state.frame_hyperlink_cells.clear();
    state.last_sidebar_area = Rect::default();
    state.last_welcome_logo = Rect::default();
    state.last_sidebar_bounds = Rect::default();
    state.last_sidebar_context_header = Rect::default();
    state.last_sidebar_mcp_header = Rect::default();
    state.last_sidebar_todos_header = Rect::default();
    let theme = state.theme();
    let area = frame.area();
    state.last_terminal_width = area.width;

    if area.is_empty() {
        return;
    }

    // Root background.
    frame.render_widget(Block::new().style(theme.app_style()), area);

    let sidebar_allowed = state.pending_permission.is_none()
        && state.pending_question.is_none()
        && !state.dialog_is_open();
    let sidebar_layout =
        layout::split_sidebar_layout(area, sidebar_allowed && state.sidebar_visible(area.width));
    let workspace = layout::workspace_area(sidebar_layout.main);
    if workspace.height == 0 {
        // If bottom padding collapses the workspace, still render a 1-row footer.
        footer::render_footer(frame, state, area, theme);
        return;
    }
    if workspace.height == 1 {
        footer::render_footer(frame, state, workspace, theme);
        return;
    }

    if state.show_dashboard() {
        render_dashboard(frame, state, workspace, theme);
        render_transcript_toast(frame, state, workspace, theme);
        if let Some(sidebar_area) = sidebar_layout.sidebar {
            sidebar::render_sidebar(frame, state, sidebar_area, theme);
        }
        dialog::render_dialog(frame, state, area, theme);
        render_pending_question(frame, state, area, theme);
        return;
    }

    let mut metrics = layout::workspace_metrics(
        workspace,
        &state.input_buffer,
        &state.composer_tokens,
        state.pending_permission.is_some(),
        state.pending_question.is_some(),
        state.is_read_only_child_view(),
        layout::slash_panel_height(state),
        theme.card_frame,
    );
    if let Some(question) = state.pending_question.as_ref() {
        metrics.composer_height = question_composer_height(question, workspace, theme.card_frame);
        metrics.transcript_viewport_height = workspace
            .height
            .saturating_sub(metrics.composer_height)
            .saturating_sub(if workspace.height >= 7 {
                surface::CONTENT_GAP
            } else {
                0
            })
            .saturating_sub(1);
    }
    let [
        transcript_area,
        _gap_area,
        slash_panel_area,
        composer_area,
        footer_area,
    ] = layout::split_workspace_layout(workspace, metrics);

    if state.active_timeline().items().is_empty() {
        render_welcome(frame, state, transcript_area, theme);
    } else {
        transcript::render_transcript(frame, state, transcript_area, state.transcript_theme());
    }
    render_transcript_toast(frame, state, transcript_area, theme);

    slash_panel::render_slash_panel(frame, state, slash_panel_area, theme);
    if state.pending_question.is_some() {
        render_pending_question(frame, state, composer_area, theme);
    } else {
        composer::render_composer(frame, state, composer_area, theme);
    }
    footer::render_footer(frame, state, footer_area, theme);
    if let Some(sidebar_area) = sidebar_layout.sidebar {
        sidebar::render_sidebar(frame, state, sidebar_area, theme);
    }
    dialog::render_dialog(frame, state, area, theme);
}

fn render_pending_question(frame: &mut Frame<'_>, state: &mut TuiState, area: Rect, theme: Theme) {
    let Some(question) = state.pending_question.as_ref().cloned() else {
        return;
    };

    if area.is_empty() {
        return;
    }

    let Some(shell) = composer::render_connected_prompt_shell(
        frame,
        area,
        theme,
        surface::SurfaceEmphasis::Notice,
        1,
    ) else {
        return;
    };

    let panel_style = surface::surface_style(theme, surface::SurfaceKind::Element);
    let inner = shell.content_area;
    let is_confirm = question.is_confirm_tab();
    let tab_rows = if question.show_confirm_tab() {
        question_tab_row_count(&question, inner.width.max(1) as usize)
    } else {
        0
    };

    let mut lines = Vec::new();
    if question.show_confirm_tab() && !is_confirm {
        lines.extend(question_tab_lines(
            &question,
            inner.width.max(1) as usize,
            theme,
        ));
        lines.push(Line::default());
    }

    if question.is_confirm_tab() {
        let translator = state.translator();
        let body_lines = confirm_body_lines(&question, theme, &translator);
        let tab_height = tab_rows.min(inner.height as usize) as u16;
        let body_height = inner.height.saturating_sub(tab_height).saturating_sub(1);
        let body_rows = confirm_body_row_count(&question, inner.width.max(1) as usize, &translator);
        let scroll_max = body_rows.saturating_sub(body_height as usize);
        if let Some(question) = state.pending_question.as_mut() {
            question.set_confirm_scroll_max(scroll_max);
        }
        let scroll = state
            .pending_question
            .as_ref()
            .map(|question| question.confirm_scroll)
            .unwrap_or_default();
        if inner.width > 0 && tab_height > 0 {
            let tabs_area = Rect::new(inner.x, inner.y, inner.width, tab_height);
            frame.render_widget(
                Paragraph::new(question_tab_lines(
                    &question,
                    inner.width.max(1) as usize,
                    theme,
                )),
                tabs_area,
            );
        }
        if inner.width > 0 && body_height > 0 {
            let body_area = Rect::new(
                inner.x,
                inner.y.saturating_add(tab_height).saturating_add(1),
                inner.width,
                body_height,
            );
            frame.render_widget(
                Paragraph::new(body_lines)
                    .style(panel_style)
                    .wrap(Wrap { trim: false })
                    .scroll((u16::try_from(scroll).unwrap_or(u16::MAX), 0)),
                body_area,
            );
        }
    } else if let Some(current) = question.current_question() {
        let content_width = inner.width.max(1) as usize;
        let compact_options =
            question_full_row_count(&question, content_width) > shell.content_area.height as usize;
        if compact_options {
            lines = compact_question_lines(
                &question,
                content_width,
                shell.content_area.height as usize,
                theme,
            );
        } else {
            if let Some(origin) = &question.origin_label {
                lines.push(Line::from(Span::styled(
                    origin.clone(),
                    Style::default().fg(theme.notice),
                )));
            }
            lines.push(Line::from(Span::styled(
                if current.multiple {
                    format!("{} (select all that apply)", current.question)
                } else {
                    current.question.clone()
                },
                Style::default().fg(theme.text).add_modifier(Modifier::BOLD),
            )));
            lines.push(Line::default());

            for index in 0..current.options.len() {
                let option = &current.options[index];
                let active = question.active_row.index() == index;
                let selected = current.option_selected(&option.label);
                let check = if current.multiple {
                    if selected { "[✓] " } else { "[ ] " }
                } else {
                    ""
                };
                let trailing = if !current.multiple && selected {
                    " ✓"
                } else {
                    ""
                };
                let marker = choice::choice_marker(active);
                let option_style = if active {
                    Style::default().fg(theme.notice).bg(theme.element_bg)
                } else if selected {
                    Style::default().fg(theme.success)
                } else {
                    Style::default().fg(theme.text)
                };
                let meta_style = if active {
                    Style::default().fg(theme.muted_text).bg(theme.element_bg)
                } else {
                    Style::default().fg(theme.muted_text)
                };
                lines.push(Line::from(vec![Span::styled(
                    format!("{marker} {}. {check}{}{trailing}", index + 1, option.label),
                    option_style,
                )]));
                lines.push(Line::from(Span::styled(
                    format!("   {}", option.description),
                    meta_style,
                )));
            }

            let custom_active = question.active_custom_row();
            let custom_marker = choice::choice_marker(custom_active);
            let custom_selected = current.custom_selected();
            let custom_prefix = if current.multiple {
                if custom_selected { "[✓] " } else { "[ ] " }
            } else {
                ""
            };
            let custom_trailing = if !current.multiple && custom_selected {
                " ✓"
            } else {
                ""
            };
            let custom_label = format!("{custom_prefix}Type your own answer{custom_trailing}");
            let custom_base_style = if custom_active {
                Style::default().fg(theme.notice).bg(theme.element_bg)
            } else if custom_selected {
                Style::default().fg(theme.success)
            } else {
                Style::default().fg(theme.text)
            };
            lines.push(Line::from(vec![
                Span::styled(
                    format!("{custom_marker} {}. ", current.options.len() + 1),
                    custom_base_style,
                ),
                Span::styled(custom_label, custom_base_style),
            ]));
            if question.editing_custom {
                lines.push(Line::from(Span::styled(
                    format_custom_edit_line(
                        current.custom_edit_text.as_str(),
                        current.custom_edit_cursor,
                        inner.width as usize,
                    ),
                    Style::default().fg(theme.text).bg(theme.element_bg),
                )));
            } else if !current.custom_text.trim().is_empty() {
                lines.push(Line::from(Span::styled(
                    format!("   {}", current.custom_text.trim()),
                    Style::default().fg(theme.muted_text),
                )));
            }
        }
    }

    let footer_detail = translate_question_detail(state, question_enter_detail(&question));
    if !is_confirm {
        frame.render_widget(
            Paragraph::new(lines)
                .style(panel_style)
                .wrap(Wrap { trim: false }),
            inner,
        );
    }

    if let Some(footer_area) = shell.footer_area {
        let mut footer = if question.is_confirm_tab() {
            vec![
                Span::styled(
                    "↑↓",
                    Style::default()
                        .fg(theme.accent)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!(" {}  ", state.t("ui.scroll")),
                    Style::default().fg(theme.muted_text),
                ),
            ]
        } else if question.editing_custom {
            vec![
                Span::styled(
                    "typing",
                    Style::default()
                        .fg(theme.accent)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled("  ", Style::default().fg(theme.muted_text)),
            ]
        } else {
            vec![
                Span::styled(
                    "↑↓",
                    Style::default()
                        .fg(theme.accent)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!(" {}  ", state.t("ui.select")),
                    Style::default().fg(theme.muted_text),
                ),
            ]
        };
        footer.extend([
            Span::styled(
                "enter",
                Style::default()
                    .fg(if question.is_confirm_tab() && !question.all_answered() {
                        theme.error
                    } else {
                        theme.notice
                    })
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!(" {footer_detail}  "),
                Style::default().fg(theme.muted_text),
            ),
            Span::styled(
                "esc",
                Style::default()
                    .fg(theme.accent)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                if question.editing_custom {
                    format!(" {}", state.t("ui.cancel_edit"))
                } else {
                    format!(" {}", state.t("ui.dismiss"))
                },
                Style::default().fg(theme.muted_text),
            ),
        ]);
        frame.render_widget(
            Paragraph::new(Line::from(footer)).style(panel_style),
            footer_area,
        );
    }
}

fn question_tab_token(label: &str, width: usize) -> (String, bool, usize) {
    let width = width.max(1);
    if width <= 2 {
        let token = crate::tui::components::tool_card::truncate_display_width(label, width);
        let token_width = display_width(&token);
        return (token, false, token_width);
    }

    let label_width = width.saturating_sub(3).max(1);
    let label = crate::tui::components::tool_card::truncate_display_width(label, label_width);
    let token = format!(" {label} ");
    let token_width = display_width(&token);
    let has_separator = width >= 4;
    (
        token,
        has_separator,
        token_width + usize::from(has_separator),
    )
}

fn question_tab_lines(
    question: &crate::tui::state::PendingQuestionState,
    width: usize,
    theme: Theme,
) -> Vec<Line<'static>> {
    let width = width.max(1);
    let mut rows = Vec::new();
    let mut current = Vec::new();
    let mut current_width = 0usize;
    for index in 0..question.total_tabs() {
        let label = question.active_tab_label(index).unwrap_or_default();
        let (token, has_separator, token_width) = question_tab_token(label, width);
        if !current.is_empty() && current_width.saturating_add(token_width) > width {
            rows.push(Line::from(std::mem::take(&mut current)));
            current_width = 0;
        }
        let answered = question
            .questions
            .get(index)
            .is_some_and(|item| item.is_answered());
        let style = if index == question.active_tab {
            choice::highlighted_choice_style(theme, theme.accent)
        } else if answered {
            Style::default().fg(theme.text).bg(theme.element_bg)
        } else {
            Style::default().fg(theme.muted_text).bg(theme.element_bg)
        };
        current.push(Span::styled(token, style));
        if has_separator {
            current.push(Span::styled(" ", Style::default().bg(theme.element_bg)));
        }
        current_width = current_width.saturating_add(token_width);
    }
    if !current.is_empty() {
        rows.push(Line::from(current));
    }
    rows
}

fn question_tab_row_count(
    question: &crate::tui::state::PendingQuestionState,
    width: usize,
) -> usize {
    let width = width.max(1);
    let mut rows = 1usize;
    let mut row_width = 0usize;
    for index in 0..question.total_tabs() {
        let label = question.active_tab_label(index).unwrap_or_default();
        let (_, _, token_width) = question_tab_token(label, width);
        if row_width > 0 && row_width.saturating_add(token_width) > width {
            rows += 1;
            row_width = 0;
        }
        row_width = row_width.saturating_add(token_width);
    }
    rows
}

fn confirm_body_lines(
    question: &crate::tui::state::PendingQuestionState,
    theme: Theme,
    translator: &crate::tui::i18n::Translator,
) -> Vec<Line<'static>> {
    let mut lines = vec![
        Line::from(Span::styled(
            translator.t("question.confirm"),
            Style::default().fg(theme.text).add_modifier(Modifier::BOLD),
        )),
        Line::default(),
    ];
    for (index, item) in question.questions.iter().enumerate() {
        lines.push(Line::from(Span::styled(
            format!("{}. {}", index + 1, item.header),
            Style::default().fg(theme.text).add_modifier(Modifier::BOLD),
        )));
        let answers = item.answers();
        if answers.is_empty() {
            lines.push(Line::from(Span::styled(
                translator.t("ui.not_answered"),
                Style::default()
                    .fg(theme.error)
                    .add_modifier(Modifier::BOLD),
            )));
        } else {
            for answer in answers {
                lines.push(Line::from(Span::styled(
                    format!("  - {answer}"),
                    Style::default().fg(theme.muted_text),
                )));
            }
        }
    }
    lines
}

fn confirm_body_row_count(
    question: &crate::tui::state::PendingQuestionState,
    width: usize,
    translator: &crate::tui::i18n::Translator,
) -> usize {
    let width = width.max(1);
    let mut rows = wrapped_row_count(&translator.t("question.confirm"), width) + 1;
    for (index, item) in question.questions.iter().enumerate() {
        rows += wrapped_row_count(&format!("{}. {}", index + 1, item.header), width);
        let answers = item.answers();
        if answers.is_empty() {
            rows += wrapped_row_count(&translator.t("ui.not_answered"), width);
        } else {
            rows += answers
                .iter()
                .map(|answer| wrapped_row_count(&format!("  - {answer}"), width))
                .sum::<usize>();
        }
    }
    rows
}

fn question_content_width(area_width: u16, card_frame: bool) -> usize {
    area_width
        .saturating_sub(surface::ACCENT_BAR_WIDTH)
        .saturating_sub(surface::PROMPT_INNER_PAD_X)
        .saturating_sub(surface::CARD_PAD_RIGHT)
        .saturating_sub(u16::from(card_frame))
        .max(1) as usize
}

fn question_composer_height(
    question: &crate::tui::state::PendingQuestionState,
    workspace: Rect,
    card_frame: bool,
) -> u16 {
    layout::question_composer_height_for_content(
        workspace.height,
        question_full_row_count(
            question,
            question_content_width(workspace.width, card_frame),
        ),
        card_frame,
    )
}

fn question_enter_detail(question: &crate::tui::state::PendingQuestionState) -> &'static str {
    if question.is_confirm_tab() {
        return if question.all_answered() {
            "submit"
        } else {
            "go to unanswered"
        };
    }
    if question.editing_custom {
        let custom_is_empty = question
            .current_question()
            .is_none_or(|item| item.custom_edit_text.trim().is_empty());
        if custom_is_empty {
            return "close edit";
        }
        return if question.single_select_fast_path() {
            "submit answer"
        } else if question
            .current_question()
            .is_some_and(|item| item.multiple && question.questions.len() == 1)
        {
            "save answer"
        } else if question.active_tab + 1 < question.questions.len() {
            "next question"
        } else {
            "review answers"
        };
    }
    if question.active_custom_row() {
        return "type answer";
    }
    if question
        .current_question()
        .is_some_and(|item| item.multiple)
    {
        "toggle"
    } else if question.single_select_fast_path() {
        "choose & submit"
    } else if question.active_tab + 1 < question.questions.len() {
        "choose & next"
    } else {
        "choose & review"
    }
}

fn translate_question_detail(state: &TuiState, detail: &str) -> String {
    let key = match detail {
        "submit" => "ui.submit",
        "submit answer" => "ui.submit_answer",
        "save answer" => "ui.save_answer",
        "go to unanswered" => "ui.go_to_unanswered",
        "close edit" => "ui.close_edit",
        "next question" => "ui.next_question",
        "review answers" => "ui.review_answers",
        "type answer" => "ui.type_answer",
        "toggle" => "ui.toggle",
        "choose & submit" => "ui.choose_submit",
        "choose & next" => "ui.choose_next",
        "choose & review" => "ui.choose_review",
        _ => return detail.to_string(),
    };
    state.t(key)
}

fn question_full_row_count(
    question: &crate::tui::state::PendingQuestionState,
    width: usize,
) -> usize {
    let width = width.max(1);
    let rows = |text: &str| wrapped_row_count(text, width);
    if question.is_confirm_tab() {
        return question_tab_row_count(question, width)
            + 1
            + confirm_body_row_count(
                question,
                width,
                &crate::tui::state::TuiState::default().translator(),
            );
    }
    let Some(current) = question.current_question() else {
        return 0;
    };
    let mut total = 0;
    if question.show_confirm_tab() {
        total += question_tab_row_count(question, width) + 1;
    }
    if let Some(origin) = &question.origin_label {
        total += rows(origin);
    }
    let title = if current.multiple {
        format!("{} (select all that apply)", current.question)
    } else {
        current.question.clone()
    };
    total += rows(&title) + 1;
    for (index, option) in current.options.iter().enumerate() {
        let active = question.active_row.index() == index;
        let selected = current.option_selected(&option.label);
        let check = if current.multiple {
            if selected { "[✓] " } else { "[ ] " }
        } else {
            ""
        };
        let trailing = if !current.multiple && selected {
            " ✓"
        } else {
            ""
        };
        let marker = choice::choice_marker(active);
        total += rows(&format!(
            "{marker} {}. {check}{}{trailing}",
            index + 1,
            option.label
        ));
        total += rows(&format!("   {}", option.description));
    }
    let custom_active = question.active_custom_row();
    let custom_prefix = if current.multiple {
        if current.custom_selected() {
            "[✓] "
        } else {
            "[ ] "
        }
    } else {
        ""
    };
    let custom_trailing = if !current.multiple && current.custom_selected() {
        " ✓"
    } else {
        ""
    };
    total += rows(&format!(
        "{} {}. {custom_prefix}Type your own answer{custom_trailing}",
        choice::choice_marker(custom_active),
        current.options.len() + 1,
    ));
    if question.editing_custom {
        total += rows(&format_custom_edit_line(
            &current.custom_edit_text,
            current.custom_edit_cursor,
            width,
        ));
    } else if !current.custom_text.trim().is_empty() {
        total += rows(&format!("   {}", current.custom_text.trim()));
    }
    total
}

fn compact_question_lines(
    question: &crate::tui::state::PendingQuestionState,
    width: usize,
    height: usize,
    theme: Theme,
) -> Vec<Line<'static>> {
    let Some(current) = question.current_question() else {
        return Vec::new();
    };
    let width = width.max(1);
    let mut lines = Vec::new();
    let mut remaining = height.max(1);
    let mut push = |text: String, style: Style| {
        if remaining == 0 {
            return;
        }
        // Compact rows deliberately stay single-line: this makes the space guarantee stable.
        lines.push(Line::from(Span::styled(
            crate::tui::components::tool_card::truncate_display_width(&text, width),
            style,
        )));
        remaining -= 1;
    };

    // These rows are ordered by interaction priority, not source order. The footer is in its
    // own shell area; keep the current input/custom affordance above everything else.
    let custom_active = question.active_custom_row();
    let custom_prefix = if current.multiple && current.custom_selected() {
        "[✓] "
    } else if current.multiple {
        "[ ] "
    } else {
        ""
    };
    let custom_label = format!(
        "{} {}. {custom_prefix}Type your own answer",
        choice::choice_marker(custom_active),
        current.options.len() + 1
    );
    let custom_style = if custom_active {
        Style::default().fg(theme.notice).bg(theme.element_bg)
    } else {
        Style::default().fg(theme.text)
    };
    if question.editing_custom {
        // The label and editor deliberately share one compact row. Even a one-row viewport can
        // therefore retain both the editable target and its cursor.
        let label_width = width.saturating_div(2).max(1);
        let label =
            crate::tui::components::tool_card::truncate_display_width(&custom_label, label_width);
        let editor_width = width
            .saturating_sub(display_width(&label))
            .saturating_sub(3)
            .max(1);
        push(
            format!(
                "{label} · {}",
                format_custom_edit_line(
                    &current.custom_edit_text,
                    current.custom_edit_cursor,
                    editor_width.saturating_add(3),
                )
                .trim_start()
            ),
            custom_style,
        );
    } else {
        push(custom_label, custom_style);
    }
    if question.questions.len() > 1 || question.origin_label.is_some() {
        let context = match &question.origin_label {
            Some(origin) => format!(
                "{origin} · {}/{} {}",
                question.active_tab + 1,
                question.questions.len(),
                current.header
            ),
            None => format!(
                "{}/{} {}",
                question.active_tab + 1,
                question.questions.len(),
                current.header
            ),
        };
        push(context, Style::default().fg(theme.notice));
    }
    if !question.editing_custom && !current.custom_text.trim().is_empty() {
        push(
            format!("   {}", current.custom_text.trim()),
            Style::default().fg(theme.muted_text),
        );
    }
    if question.active_row.index() < current.options.len() {
        let active_row = question.active_row.index();
        let option = &current.options[active_row];
        push(
            format!(
                "{} {}. {}",
                choice::choice_marker(true),
                active_row + 1,
                option.label
            ),
            Style::default().fg(theme.notice).bg(theme.element_bg),
        );
    }
    let title = if current.multiple {
        format!("{} (select all that apply)", current.question)
    } else {
        current.question.clone()
    };
    push(
        title,
        Style::default().fg(theme.text).add_modifier(Modifier::BOLD),
    );
    lines
}

fn format_custom_edit_line(text: &str, cursor: usize, width: usize) -> String {
    let mut cursor = cursor.min(text.len());
    while cursor > 0 && !text.is_char_boundary(cursor) {
        cursor -= 1;
    }
    let available = width.saturating_sub(3).max(1);
    if available == 1 {
        return "   ▏".to_string();
    }
    // Reserve room for both omission indicators up front. They are only rendered when needed,
    // but this keeps the marker inside the one-row viewport in every combination.
    let content = available.saturating_sub(3); // cursor glyph plus two possible ellipses
    let before = &text[..cursor];
    let after = &text[cursor..];
    let right_budget = content / 2;
    let (right, right_hidden) = take_display_prefix(after, right_budget);
    let left_budget = content.saturating_sub(display_width(&right));
    let (left, left_hidden) = take_display_suffix(before, left_budget);
    let mut rendered = String::from("   ");
    if left_hidden {
        rendered.push('…');
    }
    rendered.push_str(&left);
    rendered.push('▏');
    rendered.push_str(&right);
    if right_hidden {
        rendered.push('…');
    }
    rendered
}

fn take_display_prefix(text: &str, width: usize) -> (String, bool) {
    let mut out = String::new();
    for ch in text.chars() {
        if display_width(&out).saturating_add(display_width(&ch.to_string())) > width {
            return (out, true);
        }
        out.push(ch);
    }
    (out, false)
}

fn take_display_suffix(text: &str, width: usize) -> (String, bool) {
    let mut out = String::new();
    for ch in text.chars().rev() {
        if display_width(&out).saturating_add(display_width(&ch.to_string())) > width {
            return (out.chars().rev().collect(), true);
        }
        out.push(ch);
    }
    (out.chars().rev().collect(), false)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct NoticeGeometry {
    width: u16,
    height: u16,
}

fn notice_geometry(area: Rect, message: &str) -> NoticeGeometry {
    const MIN_WIDTH: u16 = 14;
    const HORIZONTAL_CHROME: u16 = 4;
    const VERTICAL_CHROME: u16 = 2;

    let available_width = area.width.saturating_sub(1);
    let min_width = MIN_WIDTH.min(available_width);
    let width_cap = available_width
        .saturating_mul(3)
        .saturating_div(5)
        .max(min_width)
        .min(available_width);
    let content_width = message
        .split('\n')
        .map(display_width)
        .max()
        .unwrap_or_default();
    let desired_width = u16::try_from(content_width.saturating_add(8)).unwrap_or(u16::MAX);
    let width = desired_width.clamp(min_width, width_cap);
    let message_width = width.saturating_sub(HORIZONTAL_CHROME).max(1) as usize;
    let measured_rows = wrapped_row_count(message, message_width);
    let available_height = area.height.saturating_sub(1);
    let min_height = 3.min(available_height);
    let height_cap = available_height
        .saturating_mul(2)
        .saturating_div(5)
        .max(min_height)
        .min(available_height);
    let desired_height =
        u16::try_from(measured_rows.saturating_add(VERTICAL_CHROME as usize)).unwrap_or(u16::MAX);

    NoticeGeometry {
        width,
        height: desired_height.clamp(min_height, height_cap),
    }
}

fn notice_message_lines(
    message: &str,
    width: u16,
    max_rows: u16,
    style: Style,
) -> Vec<Line<'static>> {
    let width = width.max(1) as usize;
    let max_rows = max_rows.max(1) as usize;
    let mut rows = wrap_text_to_width(message, width);
    let truncated = rows.len() > max_rows;
    rows.truncate(max_rows);
    if truncated && let Some(last) = rows.last_mut() {
        let mut marked = last.clone();
        marked.push('…');
        *last = crate::tui::components::tool_card::truncate_display_width(&marked, width);
    }

    rows.into_iter()
        .map(|row| Line::from(Span::styled(row, style)))
        .collect()
}

fn render_transcript_toast(frame: &mut Frame<'_>, state: &TuiState, area: Rect, theme: Theme) {
    let Some(toast) = state.toast() else {
        return;
    };

    let area =
        if !state.active_timeline().items().is_empty() && state.last_transcript_area.height > 0 {
            state.last_transcript_area
        } else {
            area
        };

    if area.width < 12 || area.height < 4 {
        return;
    }

    let geometry = notice_geometry(area, &toast.message);
    let toast_width = geometry.width;
    let toast_height = geometry.height;
    let toast_x = area.right().saturating_sub(toast_width).saturating_sub(1);
    let toast_y = area.y.saturating_add(1);
    let toast_area = Rect::new(toast_x, toast_y, toast_width, toast_height.min(area.height));

    // 目前 toast 的视觉语义先收敛成两档：
    // - 普通 / 成功：主题 accent 蓝
    // - 错误：error 红
    // 这样既满足当前产品预期，也给后续扩展更多 kind 留接口。
    let accent_color = match toast.kind {
        ToastKind::Info | ToastKind::Success => theme.accent,
        ToastKind::Error => theme.error,
    };
    let bar_style = Style::default()
        .fg(accent_color)
        .bg(theme.root_bg)
        .add_modifier(Modifier::BOLD);
    let body_style = Style::default().fg(theme.text).bg(theme.elevated_bg);

    let left_bar_area = Rect::new(toast_area.x, toast_area.y, 1, toast_area.height);
    let right_bar_area = Rect::new(
        toast_area.right().saturating_sub(1),
        toast_area.y,
        1,
        toast_area.height,
    );
    let body_area = Rect::new(
        toast_area.x.saturating_add(1),
        toast_area.y,
        toast_area.width.saturating_sub(2),
        toast_area.height,
    );
    let message_area = Rect::new(
        body_area.x.saturating_add(1),
        body_area.y.saturating_add(1),
        body_area.width.saturating_sub(2),
        toast_area.height.saturating_sub(2),
    );
    let bar_lines = vec![
        Line::from(Span::styled(surface::ACCENT_BAR_GLYPH, bar_style));
        toast_area.height as usize
    ];
    let paragraph = Paragraph::new(notice_message_lines(
        &toast.message,
        message_area.width,
        message_area.height,
        body_style,
    ))
    .alignment(Alignment::Center)
    .style(body_style)
    .wrap(Wrap { trim: true });

    frame.render_widget(Clear, toast_area);
    frame.render_widget(Block::new().style(body_style), body_area);
    frame.render_widget(
        Paragraph::new(bar_lines.clone()).style(bar_style),
        left_bar_area,
    );
    frame.render_widget(Paragraph::new(bar_lines).style(bar_style), right_bar_area);
    frame.render_widget(paragraph, message_area);
}

fn render_dashboard(frame: &mut Frame<'_>, state: &mut TuiState, area: Rect, theme: Theme) {
    let footer_area = Rect::new(
        area.x,
        area.y + area.height.saturating_sub(1),
        area.width,
        1,
    );
    let content_area = Rect::new(area.x, area.y, area.width, area.height.saturating_sub(1));

    if content_area.height == 0 {
        footer::render_footer(frame, state, footer_area, theme);
        return;
    }

    let prompt_width = content_area
        .width
        .clamp(1, surface::WELCOME_PROMPT_MAX_WIDTH);
    let prompt_height = layout::composer_height(
        content_area.height,
        &state.input_buffer,
        &state.composer_tokens,
        prompt_width as usize,
    )
    .saturating_add(u16::from(theme.card_frame))
    .clamp(1, content_area.height);
    let slash_height =
        layout::slash_panel_height(state).min(content_area.height.saturating_sub(prompt_height));
    let logo_height: u16 = if content_area.width >= 52 && content_area.height >= 4 {
        4
    } else {
        1
    };
    let logo_gap: u16 = if content_area.height >= 12 { 2 } else { 1 };
    let prompt_gap: u16 = if slash_height > 0 { 1 } else { 0 };
    let hint_height: u16 = if content_area.height >= 10 { 1 } else { 0 };
    let hint_gap: u16 = if hint_height > 0 { 1 } else { 0 };
    let stack_height = logo_height
        .saturating_add(logo_gap)
        .saturating_add(prompt_height)
        .saturating_add(prompt_gap)
        .saturating_add(slash_height)
        .saturating_add(hint_gap)
        .saturating_add(hint_height)
        .min(content_area.height);
    let stack_y = content_area.y
        + content_area
            .height
            .saturating_sub(stack_height)
            .saturating_div(2);

    let logo_area = Rect::new(area.x, stack_y, area.width, logo_height);
    render_welcome(frame, state, logo_area, theme);

    let prompt_y = stack_y.saturating_add(logo_height).saturating_add(logo_gap);
    let prompt_x = content_area.x
        + content_area
            .width
            .saturating_sub(prompt_width)
            .saturating_div(2);
    let prompt_area = Rect::new(prompt_x, prompt_y, prompt_width, prompt_height);
    composer::render_composer(frame, state, prompt_area, theme);

    if slash_height > 0 {
        let slash_area = Rect::new(
            prompt_x,
            prompt_y
                .saturating_add(prompt_height)
                .saturating_add(prompt_gap),
            prompt_width,
            slash_height,
        );
        slash_panel::render_slash_panel(frame, state, slash_area, theme);
    }

    if hint_height > 0 {
        let hint_y = prompt_y
            .saturating_add(prompt_height)
            .saturating_add(prompt_gap)
            .saturating_add(slash_height)
            .saturating_add(hint_gap);
        render_dashboard_hint(
            frame,
            state,
            Rect::new(prompt_x, hint_y, prompt_width, 1),
            theme,
        );
    }

    footer::render_footer(frame, state, footer_area, theme);
}

fn render_dashboard_hint(frame: &mut Frame<'_>, state: &TuiState, area: Rect, theme: Theme) {
    if area.is_empty() || state.slash_panel_is_open() {
        return;
    }

    let line = Line::from(vec![
        Span::styled("/resume", dashboard_hint_key_style(theme)),
        Span::styled(
            format!(" {}   ", state.t("ui.sessions")),
            dashboard_hint_style(theme),
        ),
        Span::styled("/help", dashboard_hint_key_style(theme)),
        Span::styled(
            format!(" {}", state.t("ui.commands")),
            dashboard_hint_style(theme),
        ),
    ]);
    frame.render_widget(
        Paragraph::new(line)
            .style(theme.app_style())
            .alignment(Alignment::Right),
        area,
    );
}

struct WelcomeMark {
    left: &'static [&'static str],
    right: &'static [&'static str],
    brightness: f32,
}

impl WelcomeMark {
    fn with_brightness(self, brightness: f32) -> Self {
        Self {
            brightness: brightness.clamp(0.0, 1.0),
            ..self
        }
    }
}

fn welcome_mark(state: &mut TuiState, celebrate: bool) -> WelcomeMark {
    let wordmark = WelcomeMark {
        left: WELCOME_ART_LEFT,
        right: WELCOME_ART_RIGHT,
        brightness: 1.0,
    };
    if !celebrate {
        return wordmark;
    }
    let started = *state
        .welcome_intro_started_at
        .get_or_insert_with(Instant::now);
    let elapsed = started.elapsed();

    let frame = WELCOME_INTRO_FADE_IN + WELCOME_INTRO_HOLD + WELCOME_INTRO_FADE_OUT;
    let intro = frame * WELCOME_INTRO_FRAMES.len() as u32;
    if elapsed >= intro {
        let reveal = elapsed - intro;
        if reveal < WELCOME_INTRO_REVEAL {
            return wordmark.with_brightness(ease(reveal, WELCOME_INTRO_REVEAL));
        }
        return wordmark;
    }
    let index = (elapsed.as_millis() / frame.as_millis()) as usize;
    let (left, right) = WELCOME_INTRO_FRAMES[index];
    let mark = WelcomeMark {
        left,
        right,
        brightness: 1.0,
    };
    let elapsed = elapsed - frame * index as u32;
    if elapsed < WELCOME_INTRO_FADE_IN {
        return mark.with_brightness(ease(elapsed, WELCOME_INTRO_FADE_IN));
    }
    let elapsed = elapsed - WELCOME_INTRO_FADE_IN;
    if elapsed < WELCOME_INTRO_HOLD {
        return mark;
    }
    let elapsed = elapsed - WELCOME_INTRO_HOLD;
    mark.with_brightness(1.0 - ease(elapsed, WELCOME_INTRO_FADE_OUT))
}

fn ease(elapsed: Duration, window: Duration) -> f32 {
    let progress = (elapsed.as_secs_f32() / window.as_secs_f32()).clamp(0.0, 1.0);
    progress * progress * (3.0 - 2.0 * progress)
}

fn is_programmer_day() -> bool {
    if crate::env_flag_enabled("LETCODE_DEBUG_PROGRAMMER_DAY") {
        return true;
    }
    let Some(local) = local_time() else {
        return false;
    };
    local.tm_mon == 9 && local.tm_mday == 24
}

fn local_time() -> Option<libc::tm> {
    let since_epoch = SystemTime::now().duration_since(UNIX_EPOCH).ok()?;
    let seconds = libc::time_t::try_from(since_epoch.as_secs()).ok()?;
    local_tm(seconds)
}

#[cfg(unix)]
fn local_tm(seconds: libc::time_t) -> Option<libc::tm> {
    let mut local: libc::tm = unsafe { std::mem::zeroed() };
    // SAFETY: localtime_r only writes into the buffer it is given.
    if unsafe { libc::localtime_r(&seconds, &mut local) }.is_null() {
        return None;
    }
    Some(local)
}

#[cfg(windows)]
fn local_tm(seconds: libc::time_t) -> Option<libc::tm> {
    let mut local: libc::tm = unsafe { std::mem::zeroed() };
    // SAFETY: localtime_s writes into the tm it is given and reports failure by return value.
    if unsafe { libc::localtime_s(&mut local, &seconds) } != 0 {
        return None;
    }
    Some(local)
}

#[cfg(not(any(unix, windows)))]
fn local_tm(_seconds: libc::time_t) -> Option<libc::tm> {
    None
}

fn render_welcome(frame: &mut Frame<'_>, state: &mut TuiState, area: Rect, theme: Theme) {
    if area.is_empty() {
        return;
    }

    if area.width >= 52 && area.height >= 4 {
        let celebrate = is_programmer_day();
        let mark = welcome_mark(state, celebrate);
        let backdrop = assistant_fade::fade_backdrop(theme, state.terminal_background());
        let (title_style, shadow_style) = match backdrop {
            Color::Rgb(red, green, blue) => (
                assistant_fade::faded_style(
                    wordmark_style(theme),
                    (red, green, blue),
                    mark.brightness,
                ),
                assistant_fade::faded_style(
                    wordmark_shadow_style(theme),
                    (red, green, blue),
                    mark.brightness,
                ),
            ),
            _ => (wordmark_style(theme), wordmark_shadow_style(theme)),
        };
        let lines: Vec<Line<'static>> = mark
            .left
            .iter()
            .zip(mark.right.iter())
            .map(|(left, right)| {
                Line::from(vec![
                    Span::styled(format!("{left} "), shadow_style),
                    Span::styled((*right).to_string(), title_style),
                ])
            })
            .collect();
        let lines_height = lines.len() as u16;
        let block_height = WELCOME_ART_ROWS.max(lines_height);
        let title_y = area.y
            + area.height.saturating_sub(block_height).saturating_div(2)
            + block_height.saturating_sub(lines_height);
        if celebrate {
            let logo_width = display_width(mark.left[0]) + 1 + display_width(mark.right[0]);
            let logo_x = area.x + area.width.saturating_sub(logo_width as u16) / 2;
            state.last_welcome_logo = Rect::new(logo_x, title_y, logo_width as u16, lines_height);
        }
        frame.render_widget(
            Paragraph::new(lines)
                .style(theme.app_style())
                .alignment(Alignment::Center),
            Rect::new(area.x, title_y, area.width, lines_height),
        );
        return;
    }

    let title = if area.width >= 14 { "LETCODE" } else { "LC" };
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(title, wordmark_style(theme))))
            .style(theme.app_style())
            .alignment(Alignment::Center),
        area,
    );
}

fn wordmark_style(theme: Theme) -> Style {
    Style::default()
        .fg(theme.text)
        .bg(theme.root_bg)
        .add_modifier(Modifier::BOLD)
}

fn wordmark_shadow_style(theme: Theme) -> Style {
    Style::default()
        .fg(wordmark_shadow_color(theme))
        .bg(theme.root_bg)
}

/// Approximate a dim foreground without emitting SGR DIM, whose treatment of
/// block glyphs differs between terminal emulators.
///
/// Blends toward the scheme's tone so the shadow recedes on either background.
fn wordmark_shadow_color(theme: Theme) -> Color {
    const FOREGROUND_WEIGHT: u16 = 38;
    const DARK_ANCHOR: (u8, u8, u8) = (18, 18, 18);
    const LIGHT_ANCHOR: (u8, u8, u8) = (255, 255, 255);
    let Color::Rgb(red, green, blue) = theme.notice else {
        return theme.dim_text;
    };
    let anchor = if theme.light {
        LIGHT_ANCHOR
    } else {
        DARK_ANCHOR
    };
    let blend = |foreground: u8, background: u8| {
        ((u16::from(foreground) * FOREGROUND_WEIGHT
            + u16::from(background) * (100 - FOREGROUND_WEIGHT))
            / 100) as u8
    };
    Color::Rgb(
        blend(red, anchor.0),
        blend(green, anchor.1),
        blend(blue, anchor.2),
    )
}

fn dashboard_hint_style(theme: Theme) -> Style {
    Style::default().fg(theme.muted_text).bg(theme.root_bg)
}

fn dashboard_hint_key_style(theme: Theme) -> Style {
    Style::default()
        .fg(theme.text)
        .bg(theme.root_bg)
        .add_modifier(Modifier::BOLD)
}

#[cfg(test)]
#[path = "../../tests/internal/tui/render/tests.rs"]
mod tests;
