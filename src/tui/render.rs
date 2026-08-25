use ratatui::{
    Frame,
    layout::{Alignment, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Clear, Paragraph, Wrap},
};

use super::{
    components::{composer, dialog, footer, layout, sidebar, slash_panel, transcript},
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
    );
    if let Some(question) = state.pending_question.as_ref() {
        metrics.composer_height = question_composer_height(question, workspace);
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
        render_welcome(frame, transcript_area, theme);
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
                let active = question.active_row == index;
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
                let marker = if active { "›" } else { " " };
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
            let custom_marker = if custom_active { "›" } else { " " };
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
            Style::default()
                .fg(theme.root_bg)
                .bg(theme.accent)
                .add_modifier(Modifier::BOLD)
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

fn question_content_width(area_width: u16) -> usize {
    area_width
        .saturating_sub(surface::ACCENT_BAR_WIDTH)
        .saturating_sub(surface::PROMPT_INNER_PAD_X)
        .saturating_sub(surface::CARD_PAD_RIGHT)
        .max(1) as usize
}

fn question_composer_height(
    question: &crate::tui::state::PendingQuestionState,
    workspace: Rect,
) -> u16 {
    layout::question_composer_height_for_content(
        workspace.height,
        question_full_row_count(question, question_content_width(workspace.width)),
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
        let active = question.active_row == index;
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
        let marker = if active { "›" } else { " " };
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
        if custom_active { "›" } else { " " },
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
        if custom_active { "›" } else { " " },
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
    if question.active_row < current.options.len() {
        let option = &current.options[question.active_row];
        push(
            format!("› {}. {}", question.active_row + 1, option.label),
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
    render_welcome(frame, logo_area, theme);

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

fn render_welcome(frame: &mut Frame<'_>, area: Rect, theme: Theme) {
    if area.is_empty() {
        return;
    }

    if area.width >= 52 && area.height >= 4 {
        let lines: Vec<Line<'static>> = WELCOME_ART_LEFT
            .iter()
            .zip(WELCOME_ART_RIGHT.iter())
            .map(|(left, right)| {
                Line::from(vec![
                    Span::styled(format!("{left} "), wordmark_shadow_style(theme)),
                    Span::styled((*right).to_string(), wordmark_style(theme)),
                ])
            })
            .collect();
        let lines_height = lines.len() as u16;
        let title_y = area.y + area.height.saturating_sub(lines_height).saturating_div(2);
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
fn wordmark_shadow_color(theme: Theme) -> Color {
    const FOREGROUND_WEIGHT: u16 = 38;
    match (theme.notice, theme.root_bg) {
        (Color::Rgb(red, green, blue), Color::Rgb(bg_red, bg_green, bg_blue)) => {
            let blend = |foreground: u8, background: u8| {
                ((foreground as u16 * FOREGROUND_WEIGHT
                    + background as u16 * (100 - FOREGROUND_WEIGHT))
                    / 100) as u8
            };
            Color::Rgb(
                blend(red, bg_red),
                blend(green, bg_green),
                blend(blue, bg_blue),
            )
        }
        _ => theme.dim_text,
    }
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
mod tests {
    use super::*;
    use crate::context_tree::{ContextNodeId, ContextTreeOp, ContextTreeState};
    use crate::context_view::{
        ContextBlock, ContextBlockId, ContextBlockKind, ContextBlockSource, ContextViewProjection,
    };
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
        let metrics = layout::workspace_metrics(workspace, "", &[], false, false, true, 0);
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
    fn empty_welcome_view_renders_wordmark_without_panic() {
        let mut state = TuiState::new("gpt-5.5", "gpt-5.5", "default");

        let rendered = draw_to_string(&mut state, 80, 20);
        assert!(
            rendered.contains("█    █▀▀█ ▀█▀▀") || rendered.contains("LETCODE"),
            "{rendered}"
        );
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
            question_full_row_count(&question, question_content_width(workspace.width));
        let height = question_composer_height(&question, workspace);

        assert_eq!(height, content_rows as u16 + 4);
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
        question.active_row = 6;
        question.begin_custom_edit();
        question.questions[0].custom_edit_text = "custom answer kept visible".into();
        question.questions[0].custom_edit_cursor = question.questions[0].custom_edit_text.len();
        let workspace = Rect::new(2, 0, 44, 13);

        assert_eq!(question_composer_height(&question, workspace), 11);
        assert!(question_full_row_count(&question, question_content_width(workspace.width)) > 7);

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
        question.active_row = 2;
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
        question.active_row = 2;
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
        question.active_row = 1;
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
        let mut question =
            crate::tui::state::PendingQuestionState::new(request, Some("Child".into()));
        question.active_tab = 1;
        question.active_row = 1;
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
        question.active_row = 2;
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
                    question:
                        "A deliberately long question that cannot fit on one narrow terminal row"
                            .into(),
                    header: "Long".into(),
                    options: vec![crate::tool::QuestionOption {
                        label: "A deliberately long option label".into(),
                        description:
                            "A deliberately long option description that wraps several times".into(),
                    }],
                    multiple: false,
                }],
            },
            Some("Child question from a nested session".into()),
        );
        question.active_row = 1;
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

        let mut empty_custom =
            crate::tui::state::PendingQuestionState::new(request(1, false), None);
        empty_custom.active_row = 1;
        empty_custom.begin_custom_edit();
        cases.push((empty_custom, "close edit"));

        let mut single_custom =
            crate::tui::state::PendingQuestionState::new(request(1, false), None);
        single_custom.active_row = 1;
        single_custom.begin_custom_edit();
        single_custom.questions[0].custom_edit_text = "answer".into();
        cases.push((single_custom, "submit answer"));

        let mut multi_question_custom =
            crate::tui::state::PendingQuestionState::new(request(2, false), None);
        multi_question_custom.active_row = 1;
        multi_question_custom.begin_custom_edit();
        multi_question_custom.questions[0].custom_edit_text = "answer".into();
        cases.push((multi_question_custom, "next question"));

        let mut final_multi_question_custom =
            crate::tui::state::PendingQuestionState::new(request(2, false), None);
        final_multi_question_custom.active_tab = 1;
        final_multi_question_custom.active_row = 1;
        final_multi_question_custom.begin_custom_edit();
        final_multi_question_custom.questions[1].custom_edit_text = "answer".into();
        cases.push((final_multi_question_custom, "review answers"));

        let mut active_custom =
            crate::tui::state::PendingQuestionState::new(request(1, false), None);
        active_custom.active_row = 1;
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
        single_multi.active_row = 1;
        single_multi.begin_custom_edit();
        single_multi.questions[0].custom_edit_text = "answer".into();

        let mut next_question =
            crate::tui::state::PendingQuestionState::new(request(2, false), None);
        next_question.active_row = 1;
        next_question.begin_custom_edit();
        next_question.questions[0].custom_edit_text = "answer".into();

        let mut review = crate::tui::state::PendingQuestionState::new(request(2, false), None);
        review.active_tab = 1;
        review.active_row = 1;
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
    fn permission_dialog_uses_picker_style() {
        let mut state = TuiState::new("gpt-5.5", "GPT-5.5", "default");
        let mut dialog = crate::tui::state::DialogState::new(
            crate::tui::state::DialogKind::PermissionPicker,
            "Permission mode",
            Some("Select how much freedom the agent has when using tools".into()),
            vec![
                crate::tui::state::DialogItem::new(
                    "safe",
                    "Safe",
                    Some("Ask before all tools".into()),
                ),
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
}
