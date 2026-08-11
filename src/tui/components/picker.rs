use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Margin, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Clear, Paragraph, Wrap},
};

use crate::tui::{
    state::{DialogItem, DialogKind, DialogState, TuiState},
    theme::Theme,
};

const PICKER_MIN_WIDTH: u16 = 64;
const PICKER_MAX_WIDTH: u16 = 96;
const PICKER_MIN_HEIGHT: u16 = 18;
const PICKER_MAX_HEIGHT: u16 = 28;

pub fn render_picker(
    frame: &mut Frame<'_>,
    state: &mut TuiState,
    area: Rect,
    theme: Theme,
    dialog: &DialogState,
) {
    let picker_area = centered_picker_area(area);
    frame.render_widget(Clear, picker_area);
    frame.render_widget(Block::default().style(theme.elevated_style()), picker_area);

    let inner = picker_area.inner(Margin::new(3, 2));
    if inner.is_empty() {
        return;
    }

    render_header(
        frame,
        Rect::new(inner.x, inner.y, inner.width, 1),
        theme,
        &dialog.title,
    );

    let footer_y = inner.bottom().saturating_sub(1);
    let body_y = if dialog.kind == DialogKind::McpToolsPicker {
        let description_y = inner.y.saturating_add(2);
        if let Some(description) = mcp_tools_description(dialog)
            && description_y < footer_y
        {
            render_description(
                frame,
                Rect::new(inner.x, description_y, inner.width, 1),
                theme,
                description,
            );
        }

        let search_y = description_y.saturating_add(2);
        if search_y < footer_y {
            render_search(
                frame,
                Rect::new(inner.x, search_y, inner.width, 1),
                theme,
                dialog,
            );
        }

        search_y.saturating_add(2)
    } else if picker_has_search(dialog) {
        let search_y = inner.y.saturating_add(3);
        if search_y < inner.bottom() {
            render_search(
                frame,
                Rect::new(inner.x, search_y, inner.width, 1),
                theme,
                dialog,
            );
        }

        search_y.saturating_add(2)
    } else if let Some(description) = dialog.description.as_deref() {
        let description_y = inner.y.saturating_add(2);
        if description_y < footer_y {
            render_description(
                frame,
                Rect::new(inner.x, description_y, inner.width, 1),
                theme,
                description,
            );
        }

        description_y.saturating_add(2)
    } else {
        inner.y.saturating_add(3)
    };

    let body_height = footer_y.saturating_sub(body_y).saturating_sub(1);
    let body_area = Rect::new(inner.x, body_y, inner.width, body_height);
    if body_height > 0 {
        if dialog.kind == DialogKind::ContextPicker {
            render_context_picker_body(frame, body_area, theme, state, dialog);
        } else {
            render_picker_body(frame, body_area, theme, state, dialog);
        }
    }

    if footer_y > inner.y {
        let footer_area = Rect::new(inner.x, footer_y, inner.width, 1);
        if dialog.kind == DialogKind::ContextPicker {
            render_context_picker_footer(frame, footer_area, theme, dialog);
        } else if matches!(
            dialog.kind,
            DialogKind::McpPicker | DialogKind::McpToolsPicker
        ) {
            render_mcp_picker_footer(frame, footer_area, theme, dialog.kind.clone());
        } else {
            frame.render_widget(Block::default().style(theme.elevated_style()), footer_area);
        }
    }
}

fn mcp_tools_description(dialog: &DialogState) -> Option<&str> {
    let description = dialog.description.as_deref()?;
    if description.starts_with("Offline") {
        Some("Offline")
    } else if description.starts_with("Online") || description.starts_with("Disabled") {
        Some(description)
    } else {
        None
    }
}

fn picker_has_search(dialog: &DialogState) -> bool {
    matches!(
        dialog.kind,
        DialogKind::ModelPicker
            | DialogKind::AgentPicker
            | DialogKind::ExpertModelPicker(_)
            | DialogKind::SessionPicker
            | DialogKind::HistoryTree
            | DialogKind::ContextPicker
            | DialogKind::McpPicker
            | DialogKind::McpToolsPicker
            | DialogKind::SkillPicker
    )
}

fn render_description(frame: &mut Frame<'_>, area: Rect, theme: Theme, description: &str) {
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            description.to_string(),
            muted_style(theme),
        )))
        .style(theme.elevated_style()),
        area,
    );
}

fn render_header(frame: &mut Frame<'_>, area: Rect, theme: Theme, title: &str) {
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            title.to_string(),
            Style::default()
                .fg(theme.text)
                .bg(theme.elevated_bg)
                .add_modifier(Modifier::BOLD),
        )))
        .style(theme.elevated_style()),
        area,
    );

    let esc_width = 3.min(area.width);
    let esc_area = Rect::new(
        area.right().saturating_sub(esc_width),
        area.y,
        esc_width,
        area.height,
    );
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled("esc", muted_style(theme))))
            .style(theme.elevated_style()),
        esc_area,
    );
}

fn render_search(frame: &mut Frame<'_>, area: Rect, theme: Theme, dialog: &DialogState) {
    let text = if dialog.query.is_empty() {
        Span::styled("Search", muted_style(theme))
    } else {
        Span::styled(dialog.query.clone(), item_style(theme))
    };

    frame.render_widget(
        Paragraph::new(Line::from(text)).style(theme.elevated_style()),
        area,
    );
}

fn render_picker_body(
    frame: &mut Frame<'_>,
    area: Rect,
    theme: Theme,
    state: &TuiState,
    dialog: &DialogState,
) {
    let mut y = area.y;
    let rows = area.bottom().saturating_sub(y) as usize;
    if rows == 0 {
        return;
    }

    let mut rendered_any = false;
    for entry in visible_picker_entries(dialog, rows) {
        if y >= area.bottom() {
            break;
        }
        match entry {
            PickerEntry::Heading(section) => {
                render_section_heading(
                    frame,
                    Rect::new(area.x, y, area.width, 1),
                    theme,
                    section,
                    theme.accent,
                );
            }
            PickerEntry::Item(index, item) => {
                rendered_any = true;
                let row = Rect::new(area.x, y, area.width, 1);
                let selected = index == dialog.selected;
                match dialog.kind {
                    DialogKind::ModelPicker => render_model_row(
                        frame,
                        row,
                        theme,
                        item,
                        selected,
                        item.id == state.model_id,
                    ),
                    DialogKind::AgentPicker => {
                        render_session_row(frame, row, theme, item, selected, None)
                    }
                    DialogKind::ExpertModelPicker(_) => {
                        render_model_row(frame, row, theme, item, selected, false)
                    }
                    DialogKind::SessionPicker
                    | DialogKind::HistoryTree
                    | DialogKind::ContextPicker
                    | DialogKind::SkillPicker
                    | DialogKind::McpToolsPicker => {
                        render_session_row(frame, row, theme, item, selected, None)
                    }
                    DialogKind::McpPicker => render_session_row(
                        frame,
                        row,
                        theme,
                        item,
                        selected,
                        Some(mcp_status_color(item, theme)),
                    ),
                    DialogKind::PermissionPicker => render_permission_row(
                        frame,
                        row,
                        theme,
                        item,
                        selected,
                        item.id == state.permission_mode_label,
                    ),
                    DialogKind::ThemePicker => render_permission_row(
                        frame,
                        row,
                        theme,
                        item,
                        selected,
                        item.id == state.theme_id,
                    ),
                    DialogKind::ReasoningPicker => render_reasoning_row(
                        frame,
                        row,
                        theme,
                        item,
                        selected,
                        reasoning_item_is_current(
                            state.reasoning_effort_label.as_deref(),
                            &item.id,
                        ),
                    ),
                    DialogKind::ThoughtsPicker => render_reasoning_row(
                        frame,
                        row,
                        theme,
                        item,
                        selected,
                        item.id == state.thoughts_display.as_str(),
                    ),
                    DialogKind::ContextDetail => {
                        render_session_row(frame, row, theme, item, selected, None)
                    }
                }
            }
        }
        y = y.saturating_add(1);
    }

    if !rendered_any && y < area.bottom() {
        let empty_label = match dialog.kind {
            DialogKind::SessionPicker => "No sessions found",
            DialogKind::HistoryTree => "No history entries found",
            DialogKind::ContextPicker => "No context items found",
            DialogKind::McpPicker => dialog
                .description
                .as_deref()
                .unwrap_or("No MCP tools discovered"),
            DialogKind::McpToolsPicker => "No tools discovered for this server",
            DialogKind::SkillPicker => "No local skills found",
            DialogKind::PermissionPicker => "No permission modes found",
            DialogKind::ThemePicker => "No themes found",
            DialogKind::ReasoningPicker => "No reasoning efforts found",
            DialogKind::ThoughtsPicker => "No thinking display modes found",
            DialogKind::AgentPicker => "No experts found",
            DialogKind::ExpertModelPicker(_) | DialogKind::ModelPicker => "No models found",
            _ => "No items found",
        };
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(empty_label, muted_style(theme))))
                .style(theme.elevated_style()),
            Rect::new(area.x, y, area.width, 1),
        );
    }
}

fn render_mcp_picker_footer(frame: &mut Frame<'_>, area: Rect, theme: Theme, kind: DialogKind) {
    let spans = match kind {
        DialogKind::McpPicker => vec![
            Span::styled("Space", accent_style(theme)),
            Span::styled(" toggle", muted_style(theme)),
            Span::styled("  ·  ", muted_style(theme)),
            Span::styled("Enter", accent_style(theme)),
            Span::styled(" tools", muted_style(theme)),
            Span::styled("  ·  ", muted_style(theme)),
            Span::styled("Esc", accent_style(theme)),
            Span::styled(" close", muted_style(theme)),
        ],
        DialogKind::McpToolsPicker => vec![
            Span::styled("Esc", accent_style(theme)),
            Span::styled(" back", muted_style(theme)),
        ],
        _ => unreachable!("MCP picker footer only renders MCP picker kinds"),
    };
    frame.render_widget(
        Paragraph::new(Line::from(spans)).style(theme.elevated_style()),
        area,
    );
}

fn render_context_picker_body(
    frame: &mut Frame<'_>,
    area: Rect,
    theme: Theme,
    state: &mut TuiState,
    dialog: &DialogState,
) {
    if area.is_empty() {
        return;
    }

    let [list_area, gap_area, preview_area] = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(46),
            Constraint::Length(2),
            Constraint::Percentage(54),
        ])
        .split(area)
        .as_ref()
        .try_into()
        .unwrap_or([
            area,
            Rect::new(area.x, area.y, 0, 0),
            Rect::new(area.x, area.y, 0, 0),
        ]);

    render_picker_body(frame, list_area, theme, state, dialog);

    if gap_area.width > 0 {
        frame.render_widget(
            Paragraph::new(vec![
                Line::from(Span::styled("│", muted_style(theme)));
                gap_area.height as usize
            ])
            .style(theme.elevated_style()),
            Rect::new(
                gap_area
                    .x
                    .saturating_add(gap_area.width.saturating_sub(1) / 2),
                gap_area.y,
                1.min(gap_area.width),
                gap_area.height,
            ),
        );
    }

    render_context_preview(frame, preview_area, theme, state, dialog);
}

fn render_context_preview(
    frame: &mut Frame<'_>,
    area: Rect,
    theme: Theme,
    state: &mut TuiState,
    dialog: &DialogState,
) {
    if area.is_empty() {
        return;
    }

    state.update_context_picker_detail_viewport(area.width, area.height);
    let detail_scroll = state
        .dialog()
        .filter(|dialog| dialog.kind == DialogKind::ContextPicker)
        .map(|dialog| dialog.detail_scroll.min(dialog.detail_scroll_max))
        .unwrap_or_else(|| dialog.detail_scroll.min(dialog.detail_scroll_max));

    let mut lines = Vec::new();
    if let Some(detail) = state.active_context_open_detail() {
        lines.push(Line::from(Span::styled(
            detail.title,
            Style::default()
                .fg(theme.text)
                .bg(theme.elevated_bg)
                .add_modifier(Modifier::BOLD),
        )));
        if !detail.badges.is_empty() {
            lines.push(Line::from(Span::styled(
                detail.badges.join(" · "),
                muted_style(theme),
            )));
        }
        if !detail.lines.is_empty() {
            lines.push(Line::default());
            lines.extend(
                detail
                    .lines
                    .into_iter()
                    .map(|line| Line::from(Span::styled(line, item_style(theme)))),
            );
        }
    } else {
        lines.push(Line::from(Span::styled(
            "No detail available",
            muted_style(theme),
        )));
    }

    frame.render_widget(
        Paragraph::new(lines)
            .style(theme.elevated_style())
            .wrap(Wrap { trim: false })
            .scroll((detail_scroll, 0)),
        area,
    );
}

fn render_context_picker_footer(
    frame: &mut Frame<'_>,
    area: Rect,
    theme: Theme,
    dialog: &DialogState,
) {
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled("↑/↓", accent_style(theme)),
            Span::styled(
                if dialog.detail_focused {
                    " scroll"
                } else {
                    " browse"
                },
                muted_style(theme),
            ),
            Span::styled("  •  ", muted_style(theme)),
            if dialog.detail_focused {
                Span::styled("Esc", accent_style(theme))
            } else {
                Span::styled("Enter", accent_style(theme))
            },
            Span::styled(
                if dialog.detail_focused {
                    " back"
                } else {
                    " detail"
                },
                muted_style(theme),
            ),
            if dialog.detail_focused {
                Span::styled("", muted_style(theme))
            } else {
                Span::styled("  •  ", muted_style(theme))
            },
            if dialog.detail_focused {
                Span::styled("", muted_style(theme))
            } else {
                Span::styled("Esc", accent_style(theme))
            },
            Span::styled(
                if dialog.detail_focused { "" } else { " close" },
                muted_style(theme),
            ),
        ]))
        .style(theme.elevated_style()),
        area,
    );
}

enum PickerEntry<'a> {
    Heading(&'a str),
    Item(usize, &'a DialogItem),
}

fn visible_picker_entries<'a>(dialog: &'a DialogState, rows: usize) -> Vec<PickerEntry<'a>> {
    if rows == 0 {
        return Vec::new();
    }

    let entries = picker_entries(dialog);
    if entries.len() <= rows {
        return entries;
    }

    let selected_position = entries
        .iter()
        .position(|entry| matches!(entry, PickerEntry::Item(index, _) if *index == dialog.selected))
        .unwrap_or(0);
    let start = selected_position.saturating_sub(rows.saturating_sub(1));

    entries.into_iter().skip(start).take(rows).collect()
}

fn picker_entries<'a>(dialog: &'a DialogState) -> Vec<PickerEntry<'a>> {
    let mut entries = Vec::new();
    let mut previous_section: Option<&str> = None;

    for (index, item) in dialog.visible_items() {
        if matches!(
            dialog.kind,
            DialogKind::ModelPicker
                | DialogKind::AgentPicker
                | DialogKind::ExpertModelPicker(_)
                | DialogKind::SessionPicker
                | DialogKind::HistoryTree
                | DialogKind::ContextPicker
        ) {
            let section = item.section.as_deref().unwrap_or_else(|| {
                if dialog.kind == DialogKind::HistoryTree {
                    "Branches"
                } else if dialog.kind == DialogKind::ContextPicker {
                    "Context"
                } else if dialog.kind == DialogKind::AgentPicker {
                    "Experts"
                } else if matches!(
                    dialog.kind,
                    DialogKind::ModelPicker | DialogKind::ExpertModelPicker(_)
                ) {
                    "Models"
                } else {
                    "Sessions"
                }
            });
            if previous_section != Some(section) {
                entries.push(PickerEntry::Heading(section));
                previous_section = Some(section);
            }
        }

        entries.push(PickerEntry::Item(index, item));
    }

    entries
}

fn render_section_heading(
    frame: &mut Frame<'_>,
    area: Rect,
    theme: Theme,
    heading: &str,
    color: ratatui::style::Color,
) {
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            heading.to_string(),
            Style::default()
                .fg(color)
                .bg(theme.elevated_bg)
                .add_modifier(Modifier::BOLD),
        )))
        .style(theme.elevated_style()),
        area,
    );
}

fn render_model_row(
    frame: &mut Frame<'_>,
    area: Rect,
    theme: Theme,
    item: &DialogItem,
    selected: bool,
    current: bool,
) {
    let row_style = if selected {
        selected_item_style(theme)
    } else {
        item_style(theme)
    };
    frame.render_widget(Block::default().style(row_style), area);

    let content = area.inner(Margin::new(1, 0));
    if content.is_empty() {
        return;
    }

    let marker = if current { "● " } else { "  " };
    let mut spans = vec![Span::styled(marker, row_style)];
    spans.push(Span::styled(item.label.clone(), row_style));

    if let Some(detail) = &item.detail {
        spans.push(Span::styled(" ", row_style));
        spans.push(Span::styled(
            detail.clone(),
            if selected {
                selected_muted_style(theme)
            } else {
                muted_style(theme)
            },
        ));
    }

    frame.render_widget(Paragraph::new(Line::from(spans)).style(row_style), content);
}

fn render_permission_row(
    frame: &mut Frame<'_>,
    area: Rect,
    theme: Theme,
    item: &DialogItem,
    selected: bool,
    current: bool,
) {
    render_model_row(frame, area, theme, item, selected, current);
}

fn render_reasoning_row(
    frame: &mut Frame<'_>,
    area: Rect,
    theme: Theme,
    item: &DialogItem,
    selected: bool,
    current: bool,
) {
    render_model_row(frame, area, theme, item, selected, current);
}

fn reasoning_item_is_current(current: Option<&str>, item_id: &str) -> bool {
    match current {
        Some("off") => item_id == "none",
        Some(value) => value == item_id,
        None => item_id == "none",
    }
}

fn render_session_row(
    frame: &mut Frame<'_>,
    area: Rect,
    theme: Theme,
    item: &DialogItem,
    selected: bool,
    status_color: Option<ratatui::style::Color>,
) {
    let row_style = if selected {
        selected_item_style(theme)
    } else {
        item_style(theme)
    };
    frame.render_widget(Block::default().style(row_style), area);

    let content = area.inner(Margin::new(1, 0));
    if content.is_empty() {
        return;
    }

    let right_width = item
        .right_detail
        .as_ref()
        .map(|detail| detail.chars().count() as u16)
        .unwrap_or(0)
        .min(content.width);
    let left_width = content.width.saturating_sub(right_width.saturating_add(2));
    let left_area = Rect::new(content.x, content.y, left_width, content.height);
    let right_area = Rect::new(
        content.right().saturating_sub(right_width),
        content.y,
        right_width,
        content.height,
    );

    let marker = if selected { "● " } else { "  " };
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(marker, row_style),
            Span::styled(item.label.clone(), row_style),
        ]))
        .style(row_style),
        left_area,
    );

    if let Some(right_detail) = &item.right_detail {
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(
                right_detail.clone(),
                if let Some(color) = status_color {
                    Style::default().fg(color).bg(if selected {
                        theme.element_bg
                    } else {
                        theme.elevated_bg
                    })
                } else if selected {
                    selected_item_style(theme)
                } else {
                    muted_style(theme)
                },
            )))
            .style(row_style),
            right_area,
        );
    }
}

fn mcp_status_color(item: &DialogItem, theme: Theme) -> ratatui::style::Color {
    match item.right_detail.as_deref() {
        Some(status) if status.contains("Online") => theme.success,
        Some(status) if status.contains("Offline") => theme.error,
        _ => theme.muted_text,
    }
}

fn centered_picker_area(area: Rect) -> Rect {
    let target_width = area.width.saturating_mul(3) / 4;
    let width = target_width
        .clamp(PICKER_MIN_WIDTH, PICKER_MAX_WIDTH)
        .min(area.width.saturating_sub(2))
        .max(1);
    let target_height = area.height.saturating_mul(4) / 5;
    let height = target_height
        .clamp(PICKER_MIN_HEIGHT, PICKER_MAX_HEIGHT)
        .min(area.height.saturating_sub(2))
        .max(1);

    Rect::new(
        area.x + area.width.saturating_sub(width) / 2,
        area.y + area.height.saturating_sub(height) / 2,
        width,
        height,
    )
}

fn item_style(theme: Theme) -> Style {
    Style::default().fg(theme.text).bg(theme.elevated_bg)
}

fn selected_item_style(theme: Theme) -> Style {
    Style::default()
        .fg(theme.text)
        .bg(theme.element_bg)
        .add_modifier(Modifier::BOLD)
}

fn muted_style(theme: Theme) -> Style {
    Style::default().fg(theme.muted_text).bg(theme.elevated_bg)
}

fn accent_style(theme: Theme) -> Style {
    Style::default()
        .fg(theme.accent)
        .bg(theme.elevated_bg)
        .add_modifier(Modifier::BOLD)
}

fn selected_muted_style(theme: Theme) -> Style {
    Style::default().fg(theme.muted_text).bg(theme.element_bg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::{Terminal, backend::TestBackend};

    #[test]
    fn mcp_tools_picker_hides_raw_offline_diagnostics() {
        let theme = Theme::dark();
        let area = Rect::new(0, 0, 100, 30);
        let diagnostic = "Offline · connection refused at https://private.example";
        let dialog = DialogState::new(
            DialogKind::McpToolsPicker,
            "Tools · local",
            Some(diagnostic.into()),
            Vec::new(),
        );
        let mut state = TuiState::default();
        let backend = TestBackend::new(area.width, area.height);
        let mut terminal = Terminal::new(backend).expect("terminal");

        terminal
            .draw(|frame| render_picker(frame, &mut state, area, theme, &dialog))
            .expect("draw");

        let rendered = terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|cell| cell.symbol())
            .collect::<String>();
        assert!(rendered.contains("Offline"));
        assert!(!rendered.contains("connection refused"));
        assert!(!rendered.contains("private.example"));
        assert!(rendered.contains("No tools discovered for this server"));
        assert!(rendered.contains("Esc back"));
    }
}
