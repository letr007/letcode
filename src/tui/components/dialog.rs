use ratatui::{
    Frame,
    layout::{Margin, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Clear, Paragraph, Wrap},
};

use crate::tui::{
    components::picker,
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
            | DialogKind::ThemePicker
            | DialogKind::McpPicker
            | DialogKind::McpToolsPicker
            | DialogKind::SkillPicker
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
            Span::styled(" close", muted_style(theme)),
        ])
    } else {
        Line::from(vec![
            Span::styled("↑/↓", accent_style(theme)),
            Span::styled(" navigate", muted_style(theme)),
            Span::styled("  •  ", muted_style(theme)),
            Span::styled("Enter", accent_style(theme)),
            Span::styled(" confirm", muted_style(theme)),
            Span::styled("  •  ", muted_style(theme)),
            Span::styled("Esc", accent_style(theme)),
            Span::styled(" cancel", muted_style(theme)),
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
                .map(|line| (line.chars().count().saturating_add(67) / 68).max(1))
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

    #[test]
    fn agent_pickers_render_with_shared_picker_layout() {
        let theme = Theme::dark();
        let area = Rect::new(0, 0, 100, 30);
        let cases = [
            (
                DialogKind::AgentPicker,
                vec![
                    DialogItem::new("explorer", "explorer", None)
                        .with_section("Experts")
                        .with_right_detail("CPA/gpt-5.6-luna"),
                ],
                "CPA/gpt-5.6-luna",
            ),
            (
                DialogKind::ExpertModelPicker("explorer".into()),
                vec![DialogItem::new("CPA/gpt-5.6-luna", "GPT-5.6 Luna", None).with_section("CPA")],
                "CPA",
            ),
        ];

        for (kind, items, expected_label) in cases {
            let mut state = TuiState::default();
            state.open_dialog(DialogState::new(kind, "Select expert model", None, items));
            let backend = TestBackend::new(area.width, area.height);
            let mut terminal = Terminal::new(backend).expect("terminal");

            terminal
                .draw(|frame| render_dialog(frame, &mut state, area, theme))
                .expect("draw");

            let rendered = terminal
                .backend()
                .buffer()
                .content()
                .iter()
                .map(|cell| cell.symbol())
                .collect::<String>();
            assert!(rendered.contains("Search"));
            assert!(rendered.contains(expected_label));
        }
    }

    #[test]
    fn generic_dialogs_render_as_borderless_elevated_panels() {
        let theme = Theme::dark();
        let area = Rect::new(0, 0, 100, 30);

        for (kind, title) in [(DialogKind::ContextDetail, "Detail · current context")] {
            let mut state = TuiState::default();
            state.open_dialog(DialogState::new(
                kind,
                title,
                None,
                vec![DialogItem::new(
                    "detail",
                    "Description\nUseful detail",
                    None,
                )],
            ));
            let dialog_area = centered_dialog_area(area, state.dialog().expect("dialog"));
            let backend = TestBackend::new(area.width, area.height);
            let mut terminal = Terminal::new(backend).expect("terminal");

            terminal
                .draw(|frame| render_dialog(frame, &mut state, area, theme))
                .expect("draw");

            let buffer = terminal.backend().buffer();
            let top_left = buffer
                .cell((dialog_area.x, dialog_area.y))
                .expect("panel cell");
            let rendered = buffer
                .content()
                .iter()
                .map(|cell| cell.symbol())
                .collect::<String>();

            assert_eq!(top_left.symbol(), " ");
            assert_eq!(top_left.bg, theme.elevated_bg);
            assert!(!rendered.contains('┌'));
            assert!(!rendered.contains('┐'));
            assert!(!rendered.contains('└'));
            assert!(!rendered.contains('┘'));
            assert!(rendered.contains(title));
            assert!(rendered.contains("Esc close"));
        }
    }
}
