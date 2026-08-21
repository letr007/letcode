use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Paragraph, Wrap},
};

use crate::agent::TodoStatus;

use super::super::{measure::display_width, state::TuiState, surface, theme::Theme};

pub fn render_sidebar(frame: &mut Frame<'_>, state: &TuiState, area: Rect, theme: Theme) {
    if area.is_empty() {
        return;
    }

    let style = surface::surface_style(theme, surface::SurfaceKind::Element);
    frame.render_widget(Block::new().style(style), area);
    render_panel_guide(frame, area, theme);

    let inner = Rect::new(
        area.x.saturating_add(3),
        area.y.saturating_add(1),
        area.width.saturating_sub(5),
        area.height.saturating_sub(2),
    );
    if inner.is_empty() {
        return;
    }

    let label_style = Style::default().fg(theme.muted_text).bg(theme.element_bg);
    let value_style = Style::default().fg(theme.text).bg(theme.element_bg);
    let mut lines = Vec::new();

    lines.push(Line::from(Span::styled(
        state.t("sidebar.title"),
        Style::default()
            .fg(theme.accent)
            .bg(theme.element_bg)
            .add_modifier(Modifier::BOLD),
    )));
    lines.push(Line::from(Span::styled(
        truncate_middle(
            state.session_id.as_deref().unwrap_or("—"),
            inner.width as usize,
        ),
        label_style,
    )));
    lines.push(Line::default());

    compact_field(
        &mut lines,
        state.t("sidebar.model"),
        &state.model_label,
        inner.width as usize,
        theme.accent,
        theme,
    );
    compact_field(
        &mut lines,
        state.t("sidebar.provider"),
        &state.provider_label,
        inner.width as usize,
        theme.assistant,
        theme,
    );
    compact_field(
        &mut lines,
        state.t("sidebar.permission"),
        &state.permission_mode_label,
        inner.width as usize,
        permission_color(&state.permission_mode_label, theme),
        theme,
    );
    if let Some(branch) = state.git_branch.as_deref() {
        compact_field(
            &mut lines,
            state.t("sidebar.git"),
            branch,
            inner.width as usize,
            theme.user,
            theme,
        );
    }

    let context_usage = state
        .active_model_token_usage()
        .filter(|usage| usage.context_window_tokens > 0)
        .map(|usage| {
            let percent = ((usage.used_tokens.min(usage.context_window_tokens) as f64
                / usage.context_window_tokens as f64)
                * 100.0)
                .round() as u64;
            (
                format!(
                    "{} · {} / {} ({percent}%)",
                    state.current_context_branch,
                    compact_count(usage.used_tokens),
                    compact_count(usage.context_window_tokens)
                ),
                context_usage_color(percent, theme),
            )
        });
    let (context_value, context_color) =
        context_usage.unwrap_or_else(|| (state.current_context_branch.clone(), theme.notice));
    compact_field(
        &mut lines,
        state.t("sidebar.context"),
        &context_value,
        inner.width as usize,
        context_color,
        theme,
    );

    if let Some(todo) = state.latest_todo.as_ref() {
        let items = todo.items.iter().collect::<Vec<_>>();
        if !items.is_empty() {
            lines.push(Line::default());
            lines.push(Line::from(vec![
                Span::styled(
                    state.t("sidebar.todos"),
                    Style::default()
                        .fg(theme.approval)
                        .bg(theme.element_bg)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(format!("  {}", items.len()), label_style),
            ]));
            for item in items {
                let (marker, marker_color) = match item.status {
                    TodoStatus::Pending => ("○", theme.muted_text),
                    TodoStatus::InProgress => ("●", theme.approval),
                    TodoStatus::Blocked => ("!", theme.error),
                    TodoStatus::Completed => ("✓", theme.success),
                    TodoStatus::Cancelled => ("×", theme.error),
                };
                lines.push(Line::from(vec![
                    Span::styled(
                        format!("{marker} "),
                        Style::default()
                            .fg(marker_color)
                            .bg(theme.element_bg)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        truncate_to_width(&item.content, inner.width.saturating_sub(2) as usize),
                        value_style,
                    ),
                ]));
            }
        }
    }

    let footer_height = u16::from(inner.height > 1);
    let areas = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(0), Constraint::Length(footer_height)])
        .split(inner);
    frame.render_widget(
        Paragraph::new(lines)
            .style(style)
            .wrap(Wrap { trim: false }),
        areas[0],
    );
    if footer_height > 0 {
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(
                state.t("sidebar.toggle_hint"),
                Style::default().fg(theme.dim_text).bg(theme.element_bg),
            )))
            .style(style),
            areas[1],
        );
    }
}

fn compact_field(
    lines: &mut Vec<Line<'static>>,
    label: String,
    value: &str,
    width: usize,
    value_color: Color,
    theme: Theme,
) {
    const LABEL_COLUMN_WIDTH: usize = 10;
    let label = padded_label(&label, LABEL_COLUMN_WIDTH);
    let value_width = width.saturating_sub(LABEL_COLUMN_WIDTH);
    lines.push(Line::from(vec![
        Span::styled(
            label,
            Style::default().fg(theme.muted_text).bg(theme.element_bg),
        ),
        Span::styled(
            truncate_to_width(value, value_width),
            Style::default()
                .fg(value_color)
                .bg(theme.element_bg)
                .add_modifier(Modifier::BOLD),
        ),
    ]));
}

fn render_panel_guide(frame: &mut Frame<'_>, area: Rect, theme: Theme) {
    if area.height == 0 || area.width == 0 {
        return;
    }
    for y in area.y..area.bottom() {
        if let Some(cell) = frame.buffer_mut().cell_mut((area.x, y)) {
            cell.set_symbol("▎");
            cell.set_style(Style::default().fg(theme.accent).bg(theme.element_bg));
        }
    }
}

fn permission_color(permission: &str, theme: Theme) -> Color {
    match permission.to_ascii_lowercase().as_str() {
        "safe" => theme.success,
        "auto" => theme.accent,
        "yolo" | "solo" => theme.warning,
        _ => theme.approval,
    }
}

fn context_usage_color(percent: u64, theme: Theme) -> Color {
    match percent {
        0..=59 => theme.success,
        60..=79 => theme.approval,
        80..=94 => theme.warning,
        _ => theme.error,
    }
}

fn padded_label(label: &str, width: usize) -> String {
    format!(
        "{label}{}",
        " ".repeat(width.saturating_sub(display_width(label)))
    )
}

fn compact_count(value: u64) -> String {
    if value >= 1_000_000 {
        format!("{:.1}m", value as f64 / 1_000_000.0)
    } else if value >= 1_000 {
        format!("{:.1}k", value as f64 / 1_000.0)
    } else {
        value.to_string()
    }
}

fn truncate_middle(text: &str, max_width: usize) -> String {
    if display_width(text) <= max_width || max_width < 5 {
        return truncate_to_width(text, max_width);
    }
    let keep = max_width.saturating_sub(1) / 2;
    let start = truncate_to_width_without_ellipsis(text, keep);
    let end = text
        .chars()
        .rev()
        .take(keep)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<String>();
    format!("{start}…{end}")
}

fn truncate_to_width(text: &str, max_width: usize) -> String {
    if display_width(text) <= max_width {
        return text.to_string();
    }
    if max_width == 0 {
        return String::new();
    }
    let result = truncate_to_width_without_ellipsis(text, max_width.saturating_sub(1));
    format!("{result}…")
}

fn truncate_to_width_without_ellipsis(text: &str, max_width: usize) -> String {
    let mut result = String::new();
    let mut width = 0usize;
    for ch in text.chars() {
        let next = display_width(&ch.to_string());
        if width.saturating_add(next) > max_width {
            break;
        }
        result.push(ch);
        width += next;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::{Terminal, backend::TestBackend};

    #[test]
    fn sidebar_renders_compact_session_and_model_details() {
        let mut state = TuiState::default();
        state.session_id = Some("session-1234567890".into());
        state.model_label = "Test Model".into();
        let backend = TestBackend::new(42, 18);
        let mut terminal = Terminal::new(backend).expect("terminal");
        terminal
            .draw(|frame| render_sidebar(frame, &state, frame.area(), Theme::dark()))
            .expect("draw");
        let rendered = terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|cell| cell.symbol())
            .collect::<String>();
        assert!(rendered.contains("session-1234567890"), "{rendered}");
        assert!(rendered.contains("Test Model"), "{rendered}");
        assert!(
            rendered.contains("Git       ") || !rendered.contains("Git"),
            "{rendered}"
        );
        assert_eq!(
            rendered.matches("session-1234567890").count(),
            1,
            "{rendered}"
        );
    }

    #[test]
    fn sidebar_labels_align_by_terminal_display_width() {
        assert_eq!(display_width(&padded_label("模型", 10)), 10);
        assert_eq!(display_width(&padded_label("Git", 10)), 10);
        assert_eq!(display_width(&padded_label("服务商", 10)), 10);
    }
}
