use ratatui::{
    Frame,
    layout::{Margin, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Paragraph, Wrap},
};

use crate::tui::{
    state::{DialogItem, DialogKind, DialogState, TuiState},
    theme::Theme,
};

const DIALOG_MIN_WIDTH: u16 = 36;
const DIALOG_MAX_WIDTH: u16 = 72;
const MODEL_DIALOG_MIN_WIDTH: u16 = 64;
const MODEL_DIALOG_MAX_WIDTH: u16 = 96;
const MODEL_DIALOG_MIN_HEIGHT: u16 = 18;
const MODEL_DIALOG_MAX_HEIGHT: u16 = 28;

pub fn render_dialog(frame: &mut Frame<'_>, state: &TuiState, area: Rect, theme: Theme) {
    let Some(dialog) = state.dialog() else {
        return;
    };

    if matches!(
        dialog.kind,
        DialogKind::ModelPicker | DialogKind::SessionPicker
    ) {
        render_large_picker(frame, state, area, theme, dialog);
        return;
    }

    let dialog_area = centered_dialog_area(area, dialog);
    frame.render_widget(Clear, dialog_area);
    frame.render_widget(
        Block::default()
            .borders(Borders::ALL)
            .title(dialog.title.clone())
            .style(Style::default().bg(theme.elevated_bg).fg(theme.text))
            .border_style(border_style(theme)),
        dialog_area,
    );

    let inner = dialog_area.inner(Margin::new(1, 1));
    if inner.is_empty() {
        return;
    }

    let mut lines = Vec::new();
    if let Some(description) = &dialog.description {
        lines.push(Line::from(Span::styled(
            description.clone(),
            muted_style(theme),
        )));
        lines.push(Line::default());
    }

    for (index, item) in dialog.items.iter().enumerate() {
        lines.push(render_item_line(item, index == dialog.selected, theme));
    }

    if !dialog.items.is_empty() {
        lines.push(Line::default());
    }

    lines.push(Line::from(vec![
        Span::styled("↑/↓", accent_style(theme)),
        Span::styled(" navigate", muted_style(theme)),
        Span::styled("  •  ", muted_style(theme)),
        Span::styled("Enter", accent_style(theme)),
        Span::styled(" confirm", muted_style(theme)),
        Span::styled("  •  ", muted_style(theme)),
        Span::styled("Esc", accent_style(theme)),
        Span::styled(" cancel", muted_style(theme)),
    ]));

    frame.render_widget(
        Paragraph::new(lines)
            .style(Style::default().bg(theme.elevated_bg).fg(theme.text))
            .wrap(Wrap { trim: false }),
        inner,
    );
}

fn render_large_picker(
    frame: &mut Frame<'_>,
    state: &TuiState,
    area: Rect,
    theme: Theme,
    dialog: &DialogState,
) {
    let dialog_area = centered_model_dialog_area(area);
    frame.render_widget(Clear, dialog_area);
    frame.render_widget(Block::default().style(theme.elevated_style()), dialog_area);

    let inner = dialog_area.inner(Margin::new(3, 2));
    if inner.is_empty() {
        return;
    }

    let header = Rect::new(inner.x, inner.y, inner.width, 1);
    render_header(frame, header, theme, &dialog.title);

    let search_y = inner.y.saturating_add(3);
    if search_y < inner.bottom() {
        render_search(
            frame,
            Rect::new(inner.x, search_y, inner.width, 1),
            theme,
            dialog,
        );
    }

    let footer_y = inner.bottom().saturating_sub(1);
    let body_y = search_y.saturating_add(2);
    let body_height = footer_y.saturating_sub(body_y).saturating_sub(1);
    if body_height > 0 {
        render_large_picker_body(
            frame,
            Rect::new(inner.x, body_y, inner.width, body_height),
            theme,
            state,
            dialog,
        );
    }

    if footer_y > inner.y {
        frame.render_widget(
            Block::default().style(theme.elevated_style()),
            Rect::new(inner.x, footer_y, inner.width, 1),
        );
    }
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

fn render_large_picker_body(
    frame: &mut Frame<'_>,
    area: Rect,
    theme: Theme,
    state: &TuiState,
    dialog: &DialogState,
) {
    let mut y = area.y;
    if dialog.kind == DialogKind::ModelPicker {
        render_section_heading(
            frame,
            Rect::new(area.x, y, area.width, 1),
            theme,
            "Recent",
            theme.accent,
        );
        y = y.saturating_add(1);
    }

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
                    DialogKind::ModelPicker => {
                        render_model_row(
                            frame,
                            row,
                            theme,
                            item,
                            selected,
                            item.id == state.model_id,
                        );
                    }
                    DialogKind::SessionPicker => {
                        render_session_row(frame, row, theme, item, selected);
                    }
                    DialogKind::PermissionPicker => {}
                }
            }
        }
        y = y.saturating_add(1);
    }

    if !rendered_any && y < area.bottom() {
        let empty_label = match dialog.kind {
            DialogKind::SessionPicker => "No sessions found",
            _ => "No models found",
        };
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(empty_label, muted_style(theme))))
                .style(theme.elevated_style()),
            Rect::new(area.x, y, area.width, 1),
        );
    }
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
        if dialog.kind == DialogKind::SessionPicker {
            let section = item.section.as_deref().unwrap_or("Sessions");
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

fn render_session_row(
    frame: &mut Frame<'_>,
    area: Rect,
    theme: Theme,
    item: &DialogItem,
    selected: bool,
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
                if selected {
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

fn centered_dialog_area(area: Rect, dialog: &DialogState) -> Rect {
    let desired_width = dialog_width(dialog).clamp(DIALOG_MIN_WIDTH, DIALOG_MAX_WIDTH);
    let width = desired_width.min(area.width.saturating_sub(2)).max(1);
    let description_rows = if dialog.description.is_some() { 2 } else { 0 };
    let content_rows = dialog.items.len() as u16;
    let footer_rows = 2;
    let height = (description_rows + content_rows + footer_rows + 2)
        .min(area.height.saturating_sub(2))
        .max(1);

    Rect::new(
        area.x + area.width.saturating_sub(width) / 2,
        area.y + area.height.saturating_sub(height) / 2,
        width,
        height,
    )
}

fn centered_model_dialog_area(area: Rect) -> Rect {
    let target_width = area.width.saturating_mul(3) / 4;
    let width = target_width
        .clamp(MODEL_DIALOG_MIN_WIDTH, MODEL_DIALOG_MAX_WIDTH)
        .min(area.width.saturating_sub(2))
        .max(1);
    let target_height = area.height.saturating_mul(4) / 5;
    let height = target_height
        .clamp(MODEL_DIALOG_MIN_HEIGHT, MODEL_DIALOG_MAX_HEIGHT)
        .min(area.height.saturating_sub(2))
        .max(1);

    Rect::new(
        area.x + area.width.saturating_sub(width) / 2,
        area.y + area.height.saturating_sub(height) / 2,
        width,
        height,
    )
}

fn dialog_width(dialog: &DialogState) -> u16 {
    let title_width = dialog.title.chars().count();
    let description_width = dialog
        .description
        .as_ref()
        .map(|text| text.chars().count())
        .unwrap_or(0);
    let item_width = dialog.items.iter().map(item_width).max().unwrap_or(0) + 6;
    title_width.max(description_width).max(item_width) as u16
}

fn item_width(item: &DialogItem) -> usize {
    item.label.chars().count()
        + item
            .detail
            .as_ref()
            .map(|detail| detail.chars().count() + 3)
            .unwrap_or(0)
}

fn render_item_line(item: &DialogItem, selected: bool, theme: Theme) -> Line<'static> {
    let marker = if selected { "› " } else { "  " };
    let mut spans = vec![Span::styled(
        marker,
        if selected {
            accent_style(theme)
        } else {
            muted_style(theme)
        },
    )];

    spans.push(Span::styled(
        item.label.clone(),
        if selected {
            selected_item_style(theme)
        } else {
            item_style(theme)
        },
    ));

    if let Some(detail) = &item.detail {
        spans.push(Span::styled(" · ", muted_style(theme)));
        spans.push(Span::styled(detail.clone(), muted_style(theme)));
    }

    Line::from(spans)
}

fn border_style(theme: Theme) -> Style {
    Style::default()
        .fg(theme.accent)
        .bg(theme.elevated_bg)
        .add_modifier(Modifier::BOLD)
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

fn selected_muted_style(theme: Theme) -> Style {
    Style::default().fg(theme.muted_text).bg(theme.element_bg)
}

fn accent_style(theme: Theme) -> Style {
    Style::default()
        .fg(theme.accent)
        .bg(theme.elevated_bg)
        .add_modifier(Modifier::BOLD)
}
