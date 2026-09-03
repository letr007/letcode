use ratatui::{
    Frame,
    layout::{Margin, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Clear, Paragraph, Wrap},
};

use crate::tui::{
    components::picker,
    measure::{display_width, wrapped_row_count},
    state::{DialogItem, DialogKind, DialogState, TuiState},
    theme::Theme,
};

const DIALOG_MIN_WIDTH: u16 = 36;
const DIALOG_MAX_WIDTH: u16 = 72;

pub fn render_dialog(frame: &mut Frame<'_>, state: &mut TuiState, area: Rect, theme: Theme) {
    let Some(dialog) = state.dialog().cloned() else {
        return;
    };

    if matches!(
        dialog.kind,
        DialogKind::ModelPicker
            | DialogKind::AgentPicker
            | DialogKind::ExpertModelPicker(_)
            | DialogKind::SessionPicker
            | DialogKind::HistoryTree
            | DialogKind::ContextPicker
            | DialogKind::PermissionPicker
            | DialogKind::ReasoningPicker
            | DialogKind::ThoughtsPicker
            | DialogKind::ToolsPicker
            | DialogKind::ThemePicker
            | DialogKind::FakePicker
            | DialogKind::McpPicker
            | DialogKind::McpToolsPicker
            | DialogKind::SkillPicker
            | DialogKind::LanguagePicker
    ) {
        picker::render_picker(frame, state, area, theme, &dialog);
        return;
    }

    let dialog_area = centered_dialog_area(area, &dialog);
    frame.render_widget(Clear, dialog_area);
    frame.render_widget(Block::default().style(theme.elevated_style()), dialog_area);

    let inner = dialog_area.inner(Margin::new(3, 2));
    if inner.is_empty() {
        return;
    }

    let mut lines = vec![
        Line::from(Span::styled(
            dialog.title.clone(),
            dialog_title_style(theme),
        )),
        Line::default(),
    ];
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

    let footer = if matches!(dialog.kind, DialogKind::ContextDetail) {
        Line::from(vec![
            Span::styled("Esc", accent_style(theme)),
            Span::styled(format!(" {}", state.t("ui.close")), muted_style(theme)),
        ])
    } else {
        Line::from(vec![
            Span::styled("↑/↓", accent_style(theme)),
            Span::styled(format!(" {}", state.t("ui.navigate")), muted_style(theme)),
            Span::styled("  •  ", muted_style(theme)),
            Span::styled("Enter", accent_style(theme)),
            Span::styled(format!(" {}", state.t("ui.confirm")), muted_style(theme)),
            Span::styled("  •  ", muted_style(theme)),
            Span::styled("Esc", accent_style(theme)),
            Span::styled(format!(" {}", state.t("ui.cancel")), muted_style(theme)),
        ])
    };
    lines.push(footer);

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
    let content_rows = dialog
        .items
        .iter()
        .map(|item| {
            let text = item
                .detail
                .as_ref()
                .map(|detail| format!("{} · {detail}", item.label))
                .unwrap_or_else(|| item.label.clone());
            text.lines()
                .map(|line| wrapped_row_count(line, 68).max(1))
                .sum::<usize>() as u16
        })
        .sum::<u16>();
    let footer_rows = 2;
    let height = (description_rows + content_rows + footer_rows + 6)
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
    let title_width = display_width(&dialog.title);
    let description_width = dialog
        .description
        .as_ref()
        .map(|text| display_width(text))
        .unwrap_or(0);
    let item_width = dialog.items.iter().map(item_width).max().unwrap_or(0) + 6;
    title_width.max(description_width).max(item_width) as u16
}

fn item_width(item: &DialogItem) -> usize {
    display_width(&item.label)
        + item
            .detail
            .as_ref()
            .map(|detail| display_width(detail) + 3)
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

fn dialog_title_style(theme: Theme) -> Style {
    Style::default()
        .fg(theme.text)
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

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::{Terminal, backend::TestBackend};
}
