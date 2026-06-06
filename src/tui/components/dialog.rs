use ratatui::{
    Frame,
    layout::{Margin, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Paragraph, Wrap},
};

use crate::tui::{
    components::picker,
    state::{DialogItem, DialogKind, DialogState, TuiState},
    theme::Theme,
};

const DIALOG_MIN_WIDTH: u16 = 36;
const DIALOG_MAX_WIDTH: u16 = 72;

pub fn render_dialog(frame: &mut Frame<'_>, state: &TuiState, area: Rect, theme: Theme) {
    let Some(dialog) = state.dialog() else {
        return;
    };

    if matches!(
        dialog.kind,
        DialogKind::ModelPicker | DialogKind::SessionPicker
    ) {
        picker::render_picker(frame, state, area, theme, dialog);
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

fn accent_style(theme: Theme) -> Style {
    Style::default()
        .fg(theme.accent)
        .bg(theme.elevated_bg)
        .add_modifier(Modifier::BOLD)
}
