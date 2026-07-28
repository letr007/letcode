use ratatui::{
    Frame,
    layout::{Alignment, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Clear, Paragraph, Wrap},
};

use super::{
    components::{composer, dialog, footer, layout, slash_panel, transcript},
    measure::{display_width, wrapped_row_count},
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
/// Render the TUI from the current state using ratatui widgets only.
///
/// Rendering may refresh viewport bookkeeping, but it never invokes tools, resolves permissions,
/// persists transcripts, or mutates runtime/business state.
pub fn render(frame: &mut Frame<'_>, state: &mut TuiState) {
    let theme = Theme::dark();
    let area = frame.area();

    if area.is_empty() {
        return;
    }

    // Root background.
    frame.render_widget(Block::new().style(theme.app_style()), area);

    let workspace = layout::workspace_area(area);
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
        dialog::render_dialog(frame, state, area, theme);
        render_pending_question(frame, state, area, theme);
        return;
    }

    let mut metrics = layout::workspace_metrics(
        workspace,
        &state.input_buffer,
        &state.composer_attachments,
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
        transcript::render_transcript(frame, state, transcript_area, theme);
    }
    render_transcript_toast(frame, state, transcript_area, theme);

    slash_panel::render_slash_panel(frame, state, slash_panel_area, theme);
    if state.pending_question.is_some() {
        render_pending_question(frame, state, composer_area, theme);
    } else {
        composer::render_composer(frame, state, composer_area, theme);
    }
    footer::render_footer(frame, state, footer_area, theme);
    dialog::render_dialog(frame, state, area, theme);
}

fn render_pending_question(frame: &mut Frame<'_>, state: &TuiState, area: Rect, theme: Theme) {
    let Some(question) = state.pending_question.as_ref() else {
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

    let mut lines = Vec::new();
    if question.show_confirm_tab() {
        let tabs: Vec<Span<'static>> = (0..question.total_tabs())
            .flat_map(|index| {
                let label = question
                    .active_tab_label(index)
                    .unwrap_or_default()
                    .to_string();
                let answered = question
                    .questions
                    .get(index)
                    .is_some_and(|item| item.is_answered());
                let style = if index == question.active_tab {
                    Style::default()
                        .fg(theme.text)
                        .bg(theme.accent)
                        .add_modifier(Modifier::BOLD)
                } else if answered {
                    Style::default().fg(theme.text).bg(theme.element_bg)
                } else {
                    Style::default().fg(theme.muted_text).bg(theme.element_bg)
                };
                [
                    Span::styled(format!(" {} ", label), style),
                    Span::styled(" ", Style::default().bg(theme.element_bg)),
                ]
            })
            .collect();
        lines.push(Line::from(tabs));
        lines.push(Line::default());
    }

    if question.is_confirm_tab() {
        lines.push(Line::from(Span::styled(
            "Confirm",
            Style::default().fg(theme.text).add_modifier(Modifier::BOLD),
        )));
        lines.push(Line::default());
        for (index, item) in question.questions.iter().enumerate() {
            let answers = item.answers();
            let unanswered = answers.is_empty();
            let answer_text = if unanswered {
                "(not answered)".to_string()
            } else {
                answers.join(", ")
            };
            lines.push(Line::from(vec![
                Span::styled(
                    format!("{}. {} ", index + 1, item.header),
                    Style::default().fg(theme.text).add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    answer_text,
                    if unanswered {
                        Style::default()
                            .fg(theme.error)
                            .add_modifier(Modifier::BOLD)
                    } else {
                        Style::default().fg(theme.muted_text)
                    },
                ),
            ]));
        }
    } else if let Some(current) = question.current_question() {
        let content_width = inner.width.max(1) as usize;
        let compact_options =
            question_full_row_count(question, content_width) > shell.content_area.height as usize;
        if compact_options {
            lines = compact_question_lines(
                question,
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

    let footer_detail = question_enter_detail(question);
    frame.render_widget(
        Paragraph::new(lines)
            .style(panel_style)
            .wrap(Wrap { trim: false }),
        inner,
    );

    if let Some(footer_area) = shell.footer_area {
        let mut footer = if question.editing_custom {
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
                Span::styled(" select  ", Style::default().fg(theme.muted_text)),
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
                    " cancel edit"
                } else {
                    " dismiss"
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

fn question_full_row_count(
    question: &crate::tui::state::PendingQuestionState,
    width: usize,
) -> usize {
    let width = width.max(1);
    let rows = |text: &str| wrapped_row_count(text, width);
    if question.is_confirm_tab() {
        let tabs = (0..question.total_tabs())
            .filter_map(|index| question.active_tab_label(index))
            .map(|label| format!(" {label}  "))
            .collect::<String>();
        return rows(&tabs)
            + 2
            + question
                .questions
                .iter()
                .enumerate()
                .map(|(index, item)| {
                    let answer = if item.answers().is_empty() {
                        "(not answered)".to_string()
                    } else {
                        item.answers().join(", ")
                    };
                    rows(&format!("{}. {} {}", index + 1, item.header, answer))
                })
                .sum::<usize>();
    }
    let Some(current) = question.current_question() else {
        return 0;
    };
    let mut total = 0;
    if question.show_confirm_tab() {
        let tabs = (0..question.total_tabs())
            .filter_map(|index| question.active_tab_label(index))
            .map(|label| format!(" {label}  "))
            .collect::<String>();
        total += rows(&tabs) + 1;
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

    if area.width < 12 || area.height < 3 {
        return;
    }

    let max_width = area.width.saturating_sub(2).clamp(12, 44);
    let content_width = display_width(&toast.message) as u16;
    let toast_width = content_width.saturating_add(8).clamp(14, max_width);
    let toast_height = 3;
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
        1,
    );
    let bar_lines = vec![
        Line::from(Span::styled(surface::ACCENT_BAR_GLYPH, bar_style));
        toast_area.height as usize
    ];
    let paragraph = Paragraph::new(Line::from(Span::styled(toast.message.clone(), body_style)))
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

fn render_dashboard(frame: &mut Frame<'_>, state: &TuiState, area: Rect, theme: Theme) {
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
        .min(surface::WELCOME_PROMPT_MAX_WIDTH)
        .max(1);
    let prompt_height = layout::composer_height(
        content_area.height,
        &state.input_buffer,
        &state.composer_attachments,
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
        Span::styled(" sessions   ", dashboard_hint_style(theme)),
        Span::styled("/help", dashboard_hint_key_style(theme)),
        Span::styled(" commands", dashboard_hint_style(theme)),
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
        .fg(theme.notice)
        .bg(theme.root_bg)
        .add_modifier(Modifier::DIM)
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
        AppEvent, AssistantDeltaEvent, ErrorEvent, PermissionRequestEvent, ToolFinishedEvent,
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

    #[test]
    fn empty_welcome_view_renders_wordmark_without_panic() {
        let mut state = TuiState::new("gpt-5.5", "gpt-5.5", "default");

        let rendered = draw_to_string(&mut state, 80, 20);
        assert!(
            rendered.contains("█    █▀▀█ ▀█▀▀") || rendered.contains("LETCODE"),
            "{rendered}"
        );
        assert!(rendered.contains("/resume sessions"), "{rendered}");

        let tiny = draw_to_string(&mut state, 10, 2);
        assert!(!tiny.is_empty());
    }

    #[test]
    fn active_empty_session_uses_normal_workspace_layout() {
        let mut state = TuiState::new("gpt-5.5", "gpt-5.5", "default");
        state.mark_session_active();

        let rendered = draw_to_string(&mut state, 100, 24);

        assert!(rendered.contains("message letcode"), "{rendered}");
        assert!(!rendered.contains("/resume sessions"), "{rendered}");
    }

    #[test]
    fn user_and_assistant_timeline_content_appears() {
        let mut state = TuiState::default();
        state.apply_event(AppEvent::UserMessage(UserMessageEvent::new("hello tui")));
        state.apply_event(AppEvent::AssistantDelta(AssistantDeltaEvent::new(
            "hi there",
        )));
        state.apply_event(AppEvent::AssistantDone { message_id: None });

        let rendered = draw_to_string(&mut state, 90, 20);

        assert!(rendered.contains("hello tui"), "{rendered}");
        assert!(rendered.contains("hi there"), "{rendered}");
        assert!(rendered.contains(surface::ACCENT_BAR_GLYPH), "{rendered}");
        assert!(!rendered.contains("streaming"), "{rendered}");
    }

    #[test]
    fn toast_renders_inside_transcript_surface() {
        let mut state = TuiState::default();
        state.apply_event(AppEvent::UserMessage(UserMessageEvent::new("hello tui")));
        state.show_toast("Copied to clipboard", crate::tui::state::ToastKind::Success);

        let rendered = draw_to_string(&mut state, 90, 20);

        assert!(rendered.contains("Copied to clipboard"), "{rendered}");
        assert!(rendered.contains("hello tui"), "{rendered}");
    }

    #[test]
    fn toast_renders_on_empty_dashboard() {
        let mut state = TuiState::default();
        state.show_toast(
            "Langfuse missing: host",
            crate::tui::state::ToastKind::Error,
        );

        let rendered = draw_to_string(&mut state, 90, 20);

        assert!(rendered.contains("Langfuse missing: host"), "{rendered}");
    }

    #[test]
    fn pending_permission_prompt_displays_hint_and_tool_summary() {
        let mut state = TuiState::default();
        let mut request = PermissionRequestEvent::new("call-1", "shell__exec", "cargo test all");
        request.arguments = Some("cargo test".into());
        request.rationale = Some("tests need confirmation".into());
        state.apply_event(AppEvent::PermissionRequested(request));

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
    fn single_select_option_action_says_choose_and_submit() {
        let mut state = TuiState::default();
        state.pending_question = Some(crate::tui::state::PendingQuestionState::new(
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
        ));

        let rendered = draw_to_string(&mut state, 100, 24);

        assert!(rendered.contains("choose & submit"), "{rendered}");
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
    fn custom_editor_viewport_keeps_ascii_and_cjk_cursors_visible() {
        let ascii = format_custom_edit_line(
            "prefix prefix cursor-tail",
            "prefix prefix cursor-tail".len(),
            14,
        );
        assert!(ascii.contains('▏'));
        assert!(ascii.contains("tail"), "{ascii}");
        assert!(display_width(&ascii) <= 14, "{ascii}");

        let cjk_text = "前缀前缀中间光标后缀后缀";
        let cursor = cjk_text.find("光标").expect("cursor text");
        let cjk = format_custom_edit_line(cjk_text, cursor, 16);
        assert!(cjk.contains('▏'));
        assert!(cjk.contains('间'), "{cjk}");
        assert!(cjk.contains('光'), "{cjk}");
        assert!(display_width(&cjk) <= 16, "{cjk}");
    }

    #[test]
    fn footer_uses_compact_help_hint_without_duplicate_metadata() {
        let mut state = TuiState::new("gpt-5.5-mini", "gpt-5.5-mini", "safe");
        state.set_token_usage(crate::tui::state::ModelTokenUsage {
            used_tokens: 50_000,
            context_window_tokens: 100_000,
            input_tokens: 40_000,
            output_tokens: 10_000,
            cached_tokens: 20_000,
            cache_report: None,
        });
        let rendered = draw_to_string(&mut state, 100, 16);

        assert!(!rendered.contains("model gpt-5.5-mini"), "{rendered}");
        assert!(
            rendered.contains("██████████ ↑40.0k ↓10.0k 50% ~50% · /help commands"),
            "{rendered}"
        );
        assert!(!rendered.contains("exit to quit"), "{rendered}");
    }

    #[test]
    fn slash_panel_renders_above_composer_in_full_view() {
        let mut state = TuiState::default();
        state.set_input("/per");

        let rendered = draw_to_string(&mut state, 100, 20);

        assert!(
            rendered.contains("Show or switch permission mode"),
            "{rendered}"
        );
        assert!(rendered.contains("/per"), "{rendered}");
        assert!(!rendered.contains("prompt ·"), "{rendered}");
    }

    #[test]
    fn expert_panel_renders_above_composer_in_full_view() {
        let mut state = TuiState::default();
        state.set_input("@or");

        let rendered = draw_to_string(&mut state, 100, 20);

        assert!(rendered.contains("@oracle"), "{rendered}");
        assert!(rendered.contains("review or audit task"), "{rendered}");
        assert!(rendered.contains("@or"), "{rendered}");
        assert!(!rendered.contains("prompt ·"), "{rendered}");
    }

    #[test]
    fn child_view_scroll_redraw_keeps_read_only_status_bar() {
        let mut state = TuiState::new("gpt-5.5", "GPT-5.5", "default");
        state.replace_child_timeline_from_records(
            &[crate::transcript::TranscriptRecord {
                session_id: "child-session".into(),
                sequence: 1,
                timestamp_ms: 0,
                context_branch_id: None,
                event: crate::transcript::TranscriptEvent::SessionStarted {
                    model: "gpt-5.5-mini".into(),
                },
            }],
            "parent-session",
            "child-session-1234567890",
            "explorer",
            0,
            1,
            1,
        );

        let before = draw_to_string(&mut state, 100, 18);
        state.scroll_transcript_down(1);
        let after = draw_to_string(&mut state, 100, 18);

        assert!(before.contains("explorer"), "{before}");
        assert!(after.contains("explorer"), "{after}");
        assert!(after.contains("gpt-5.5-mini"), "{after}");
        assert!(after.contains("Parent"), "{after}");
        assert!(!after.contains("Read-only child view"), "{after}");
        assert!(!after.contains("child-session-1234567890"), "{after}");
        assert!(!after.contains("records"), "{after}");
        assert!(!after.contains("parent-session"), "{after}");
        assert!(!after.contains("message letcode"), "{after}");
    }

    #[test]
    fn dialog_overlay_renders_title_and_items() {
        let mut state = TuiState::new("gpt-5.5", "GPT-5.5", "default");
        state.open_dialog(crate::tui::state::DialogState::new(
            crate::tui::state::DialogKind::ModelPicker,
            "Switch model",
            Some("Select a model".into()),
            vec![
                crate::tui::state::DialogItem::new(
                    "gpt-5.5",
                    "GPT-5.5",
                    Some("gpt-5.5 · current".into()),
                ),
                crate::tui::state::DialogItem::new(
                    "gpt-5.5-mini",
                    "GPT-5.5 Mini",
                    Some("gpt-5.5-mini".into()),
                ),
            ],
        ));

        let rendered = draw_to_string(&mut state, 100, 24);

        assert!(rendered.contains("Switch model"), "{rendered}");
        assert!(rendered.contains("GPT-5.5 Mini"), "{rendered}");
        assert!(rendered.contains("Search"), "{rendered}");
        assert!(rendered.contains("Recent"), "{rendered}");
    }

    #[test]
    fn dialog_overlay_does_not_leave_dashboard_composer_cursor_visible() {
        let backend = TestBackend::new(100, 24);
        let mut terminal = Terminal::new(backend).expect("test terminal is created");
        let mut state = TuiState::new("gpt-5.5", "GPT-5.5", "default");
        state.open_dialog(crate::tui::state::DialogState::new(
            crate::tui::state::DialogKind::ModelPicker,
            "Select model",
            None,
            vec![crate::tui::state::DialogItem::new(
                "gpt-5.5",
                "GPT-5.5",
                Some("gpt-5.5".into()),
            )],
        ));

        terminal
            .draw(|frame| render(frame, &mut state))
            .expect("render succeeds");

        assert_eq!(
            terminal.get_cursor_position().expect("cursor position"),
            Position::ORIGIN
        );
    }

    #[test]
    fn context_picker_renders_integrated_preview() {
        let mut state = TuiState::new("gpt-5.5", "GPT-5.5", "default");
        state.set_parent_context_for_test(sample_context_state());
        state.open_dialog(crate::tui::state::DialogState::new(
            crate::tui::state::DialogKind::ContextPicker,
            "Context",
            None,
            vec![
                crate::tui::state::DialogItem::new(
                    "block:block-1",
                    "Current plan",
                    Some("Note".into()),
                )
                .with_section("Blocks"),
            ],
        ));
        state.sync_context_picker_preview();

        let rendered = draw_to_string(&mut state, 100, 24);

        assert!(rendered.contains("Current plan"), "{rendered}");
        assert!(rendered.contains("Outline next steps"), "{rendered}");
        assert!(!rendered.contains("Detail ·"), "{rendered}");
    }

    #[test]
    fn group_16_context_picker_renders_only_canonical_surviving_detail() {
        let snapshot = crate::runtime_context::group_16_runtime_snapshot();
        let context = crate::runtime_context::RuntimeActiveContext::try_from(&snapshot)
            .expect("canonical runtime context");
        let mut state = TuiState::new("gpt-5.5", "gpt-5.5", "default");
        state.apply_event(AppEvent::RuntimeContextUpdated(
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
    fn context_picker_preview_wraps_full_detail_content() {
        let mut state = TuiState::new("gpt-5.5", "GPT-5.5", "default");
        let mut context = sample_context_state();
        if let Some(block) = context.view.blocks.values_mut().next() {
            block.detail = "This detail line is intentionally long enough to wrap across the inspector preview without being cut off early by a fixed character cap.".into();
        }
        state.set_parent_context_for_test(context);
        state.open_dialog(crate::tui::state::DialogState::new(
            crate::tui::state::DialogKind::ContextPicker,
            "Context",
            None,
            vec![
                crate::tui::state::DialogItem::new(
                    "block:block-1",
                    "Current plan",
                    Some("Note".into()),
                )
                .with_section("Blocks"),
            ],
        ));
        state.sync_context_picker_preview();

        let rendered = draw_to_string(&mut state, 80, 24);

        assert!(rendered.contains("intentionally long enough"), "{rendered}");
        assert!(rendered.contains("fixed character cap"), "{rendered}");
    }

    #[test]
    fn context_picker_clamps_detail_scroll_to_content() {
        let mut state = TuiState::new("gpt-5.5", "GPT-5.5", "default");
        let mut context = sample_context_state();
        if let Some(block) = context.view.blocks.values_mut().next() {
            block.detail = (0..32)
                .map(|index| format!("detail row {index}"))
                .collect::<Vec<_>>()
                .join(" ");
        }
        state.set_parent_context_for_test(context);
        state.open_dialog(crate::tui::state::DialogState::new(
            crate::tui::state::DialogKind::ContextPicker,
            "Context",
            None,
            vec![
                crate::tui::state::DialogItem::new(
                    "block:block-1",
                    "Current plan",
                    Some("Note".into()),
                )
                .with_section("Blocks"),
            ],
        ));
        state.sync_context_picker_preview();
        if let Some(dialog) = state.dialog.as_mut() {
            dialog.detail_focused = true;
            dialog.detail_scroll = u16::MAX;
        }

        let rendered = draw_to_string(&mut state, 80, 24);
        let dialog = state.dialog().expect("dialog open");

        assert!(dialog.detail_scroll <= dialog.detail_scroll_max);
        assert!(dialog.detail_scroll_max < u16::MAX);
        assert!(rendered.contains("detail row"), "{rendered}");
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
                    "solo",
                    "Solo",
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

    #[test]
    fn model_dialog_scrolls_to_selected_item() {
        let mut state = TuiState::new("model-00", "Model 00", "default");
        let items = (0..20)
            .map(|index| {
                crate::tui::state::DialogItem::new(
                    format!("model-{index:02}"),
                    format!("Model {index:02}"),
                    Some(format!("provider-{index:02}")),
                )
            })
            .collect::<Vec<_>>();
        let mut dialog = crate::tui::state::DialogState::new(
            crate::tui::state::DialogKind::ModelPicker,
            "Select model",
            None,
            items,
        );
        dialog.selected = 14;
        state.open_dialog(dialog);

        let rendered = draw_to_string(&mut state, 100, 20);

        assert!(rendered.contains("Model 14"), "{rendered}");
        assert!(!rendered.contains("Model 01"), "{rendered}");
    }

    #[test]
    fn session_dialog_scrolls_to_selected_item() {
        let mut state = TuiState::default();
        let items = (0..20)
            .map(|index| {
                crate::tui::state::DialogItem::new(
                    format!("session-{index:02}"),
                    format!("Session {index:02}"),
                    Some(format!("detail-{index:02}")),
                )
                .with_section("Today")
            })
            .collect::<Vec<_>>();
        let mut dialog = crate::tui::state::DialogState::new(
            crate::tui::state::DialogKind::SessionPicker,
            "Sessions",
            None,
            items,
        );
        dialog.selected = 14;
        state.open_dialog(dialog);

        let rendered = draw_to_string(&mut state, 100, 20);

        assert!(rendered.contains("Session 14"), "{rendered}");
        assert!(!rendered.contains("Session 01"), "{rendered}");
    }

    #[test]
    fn tool_cards_and_errors_use_structured_timeline_fields() {
        let mut state = TuiState::default();
        let mut started = ToolStartedEvent::new("tool-7", "shell__exec", "run cargo check");
        started.arguments = Some("cargo check".into());
        state.apply_event(AppEvent::ToolStarted(started));
        let mut finished = ToolFinishedEvent::new(
            "tool-7",
            "shell__exec",
            "run cargo check",
            ToolOutcome::Failure,
        );
        finished.output = Some("compiler said no".into());
        state.apply_event(AppEvent::ToolFinished(finished));
        let mut error = ErrorEvent::new("render problem");
        error.details = Some("missing widget area".into());
        state.apply_event(AppEvent::Error(error));

        let rendered = draw_to_string(&mut state, 100, 24);

        assert!(rendered.contains("→"), "{rendered}");
        assert!(rendered.contains("Run"), "{rendered}");
        assert!(rendered.contains("cargo check"), "{rendered}");
        assert!(!rendered.contains("compiler said no"), "{rendered}");
        assert!(!rendered.contains("Error:"), "{rendered}");
        assert!(rendered.contains("render problem"), "{rendered}");
    }
}
