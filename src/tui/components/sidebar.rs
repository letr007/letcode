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

    if let Some(usage) = state
        .active_model_token_usage()
        .filter(|usage| usage.context_window_tokens > 0)
    {
        let percent = context_used_percent(usage.used_tokens, usage.context_window_tokens);
        let usage_summary = format!(
            "{} / {} ({percent}%)",
            compact_count(usage.used_tokens),
            compact_count(usage.context_window_tokens)
        );
        wrapped_context_field(
            &mut lines,
            state.t("sidebar.context"),
            &state.current_context_branch,
            &usage_summary,
            inner.width as usize,
            context_usage_color(percent, theme),
            theme,
        );
    } else {
        compact_field(
            &mut lines,
            state.t("sidebar.context"),
            &state.current_context_branch,
            inner.width as usize,
            theme.notice,
            theme,
        );
    }
    let todo_rows = todo_row_count(state, inner.width.saturating_sub(2) as usize);
    let mcp_rows = mcp_row_count(state, inner.height as usize, lines.len(), todo_rows);
    let reserved_rows = mcp_rows.saturating_add(todo_rows);
    if let Some(usage) = state
        .active_model_token_usage()
        .filter(|usage| usage.context_window_tokens > 0)
    {
        render_context_usage_details(
            &mut lines,
            usage,
            inner.width as usize,
            inner.height as usize,
            reserved_rows,
            state,
            theme,
        );
    }

    render_mcp_status(&mut lines, state, inner.width as usize, mcp_rows, theme);

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
                let content_width = inner.width.saturating_sub(2) as usize;
                for (index, row) in wrap_to_width(&item.content, content_width)
                    .into_iter()
                    .enumerate()
                {
                    lines.push(Line::from(vec![
                        Span::styled(
                            if index == 0 {
                                format!("{marker} ")
                            } else {
                                "  ".into()
                            },
                            Style::default()
                                .fg(marker_color)
                                .bg(theme.element_bg)
                                .add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(row, value_style),
                    ]));
                }
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

fn render_context_usage_details(
    lines: &mut Vec<Line<'static>>,
    usage: &crate::tui::state::ModelTokenUsage,
    width: usize,
    available_height: usize,
    reserved_rows: usize,
    state: &TuiState,
    theme: Theme,
) {
    if width < 8 || available_height.saturating_sub(lines.len().saturating_add(reserved_rows)) < 4 {
        return;
    }

    let estimated_composition_tokens =
        usage.prompt_composition.iter().fold(0u64, |total, entry| {
            total.saturating_add(entry.estimated_tokens)
        });
    let output_context_tokens = usage.used_tokens.saturating_sub(usage.input_tokens);
    let display_used_tokens = usage.used_tokens;
    let remaining = usage
        .context_window_tokens
        .saturating_sub(display_used_tokens.min(usage.context_window_tokens));
    let percent = context_used_percent(display_used_tokens, usage.context_window_tokens);

    lines.push(Line::default());
    lines.push(Line::from(vec![
        Span::styled(
            state.t("sidebar.context_usage"),
            Style::default()
                .fg(context_usage_color(percent, theme))
                .bg(theme.element_bg)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!(
                "  {} / {} · {percent}%",
                compact_count(display_used_tokens),
                compact_count(usage.context_window_tokens)
            ),
            Style::default().fg(theme.muted_text).bg(theme.element_bg),
        ),
    ]));
    lines.push(context_bar_line(
        width,
        &usage.prompt_composition,
        estimated_composition_tokens,
        usage.input_tokens,
        output_context_tokens,
        usage.context_window_tokens,
        theme,
    ));

    let detail_capacity = available_height
        .saturating_sub(lines.len().saturating_add(reserved_rows).saturating_add(1));
    if detail_capacity == 0 {
        return;
    }
    let mut details = usage
        .prompt_composition
        .iter()
        .enumerate()
        .map(|(index, entry)| {
            context_detail_line(
                format_composition_label(entry, state),
                scaled_composition_tokens(
                    entry.estimated_tokens,
                    estimated_composition_tokens,
                    usage.input_tokens,
                ),
                context_composition_color(entry, index, theme),
                theme,
            )
        })
        .collect::<Vec<_>>();
    if output_context_tokens > 0 {
        details.push(context_detail_line(
            state.t("sidebar.context_output"),
            output_context_tokens,
            theme.assistant,
            theme,
        ));
    }
    details.push(context_detail_line(
        state.t("sidebar.context_remaining"),
        remaining,
        theme.muted_text,
        theme,
    ));
    lines.extend(details.into_iter().take(detail_capacity));
}

fn format_composition_label(
    entry: &crate::agent::PromptCompositionEntry,
    state: &TuiState,
) -> String {
    state.t(match composition_category(entry.category.as_str()) {
        "system" => "sidebar.context_system",
        "tools" => "sidebar.context_tools",
        "skills" => "sidebar.context_skills",
        "context" => "sidebar.context_material",
        "messages" => "sidebar.context_messages",
        _ => "sidebar.context_other",
    })
}

fn context_detail_line(label: String, tokens: u64, color: Color, theme: Theme) -> Line<'static> {
    Line::from(vec![
        Span::styled("● ", Style::default().fg(color).bg(theme.element_bg)),
        Span::styled(
            padded_label(&label, 16),
            Style::default().fg(theme.muted_text).bg(theme.element_bg),
        ),
        Span::styled(
            compact_count(tokens),
            Style::default()
                .fg(color)
                .bg(theme.element_bg)
                .add_modifier(Modifier::BOLD),
        ),
    ])
}

fn context_bar_line(
    width: usize,
    composition: &[crate::agent::PromptCompositionEntry],
    estimated_total: u64,
    actual_input_tokens: u64,
    output_context_tokens: u64,
    context_window_tokens: u64,
    theme: Theme,
) -> Line<'static> {
    let bar_width = width.max(1);
    let mut remaining_cells = bar_width;
    let mut spans = Vec::new();
    for (index, entry) in composition.iter().enumerate() {
        let tokens =
            scaled_composition_tokens(entry.estimated_tokens, estimated_total, actual_input_tokens);
        let cells =
            proportional_cells(tokens, context_window_tokens, bar_width).min(remaining_cells);
        if cells == 0 {
            continue;
        }
        spans.push(Span::styled(
            "█".repeat(cells),
            Style::default()
                .fg(context_composition_color(entry, index, theme))
                .bg(theme.element_bg),
        ));
        remaining_cells = remaining_cells.saturating_sub(cells);
        if remaining_cells == 0 {
            break;
        }
    }
    if remaining_cells > 0 && output_context_tokens > 0 {
        let cells = proportional_cells(output_context_tokens, context_window_tokens, bar_width)
            .min(remaining_cells);
        if cells > 0 {
            spans.push(Span::styled(
                "█".repeat(cells),
                Style::default().fg(theme.assistant).bg(theme.element_bg),
            ));
            remaining_cells = remaining_cells.saturating_sub(cells);
        }
    }
    if remaining_cells > 0 {
        spans.push(Span::styled(
            " ".repeat(remaining_cells),
            Style::default().fg(theme.border).bg(theme.elevated_bg),
        ));
    }
    Line::from(spans)
}

fn context_composition_color(
    entry: &crate::agent::PromptCompositionEntry,
    index: usize,
    theme: Theme,
) -> Color {
    match composition_category(entry.category.as_str()) {
        "system" => theme.notice,
        "tools" => theme.warning,
        "skills" => theme.success,
        "context" => theme.approval,
        "messages" => theme.user,
        _ => [theme.accent, theme.success, theme.error][index % 3],
    }
}

fn composition_category(value: &str) -> &'static str {
    if value == "tools" || value == "tool_definitions" {
        "tools"
    } else if value == "skills" || value.starts_with("SkillMaterial:") {
        "skills"
    } else if value == "system"
        || value.starts_with("SystemPrelude:")
        || value.starts_with("DeveloperPrelude:")
    {
        "system"
    } else if value == "context"
        || value.starts_with("RuntimeContext:")
        || value.starts_with("ContextMaterial:")
        || value.starts_with("ContextIndex:")
        || value.starts_with("Evidence:")
    {
        "context"
    } else if value == "messages"
        || value.starts_with("TranscriptFrame:")
        || value.starts_with("CurrentTurn:")
    {
        "messages"
    } else {
        "other"
    }
}

fn scaled_composition_tokens(estimated: u64, estimated_total: u64, actual_total: u64) -> u64 {
    if estimated_total == 0 || actual_total == 0 {
        return estimated;
    }
    ((estimated as u128 * actual_total as u128) / estimated_total as u128) as u64
}

fn proportional_cells(tokens: u64, total: u64, width: usize) -> usize {
    if tokens == 0 || total == 0 || width == 0 {
        return 0;
    }
    (((tokens.min(total) as f64 / total as f64) * width as f64).round() as usize).clamp(1, width)
}

fn context_used_percent(used: u64, total: u64) -> u64 {
    if total == 0 {
        return 0;
    }
    (((used.min(total) as f64 / total as f64) * 100.0).round() as u64).min(100)
}

const LABEL_COLUMN_WIDTH: usize = 10;

fn compact_field(
    lines: &mut Vec<Line<'static>>,
    label: String,
    value: &str,
    width: usize,
    value_color: Color,
    theme: Theme,
) {
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

fn wrapped_context_field(
    lines: &mut Vec<Line<'static>>,
    label: String,
    branch: &str,
    usage: &str,
    width: usize,
    value_color: Color,
    theme: Theme,
) {
    let value_width = width.saturating_sub(LABEL_COLUMN_WIDTH);
    let inline = format!("{branch} · {usage}");
    if display_width(&inline) <= value_width {
        compact_field(lines, label, &inline, width, value_color, theme);
        return;
    }

    compact_field(lines, label, branch, width, value_color, theme);
    lines.push(Line::from(vec![
        Span::raw(" ".repeat(LABEL_COLUMN_WIDTH)),
        Span::styled(
            truncate_to_width(usage, value_width),
            Style::default()
                .fg(value_color)
                .bg(theme.element_bg)
                .add_modifier(Modifier::BOLD),
        ),
    ]));
}

fn mcp_row_count(
    state: &TuiState,
    available_height: usize,
    current_rows: usize,
    todo_rows: usize,
) -> usize {
    let desired = match state.mcp_discovery {
        crate::tui::state::McpDiscoveryState::Ready => {
            usize::from(!state.mcp_servers.is_empty()) + state.mcp_servers.len()
        }
        crate::tui::state::McpDiscoveryState::Loading
        | crate::tui::state::McpDiscoveryState::Unavailable => 1,
    };
    let capacity =
        available_height.saturating_sub(current_rows.saturating_add(todo_rows).saturating_add(1));
    if desired > 1 && capacity == 1 {
        0
    } else {
        desired.min(capacity)
    }
}

fn todo_row_count(state: &TuiState, content_width: usize) -> usize {
    state.latest_todo.as_ref().map_or(0, |todo| {
        let items = todo.items.iter().collect::<Vec<_>>();
        if items.is_empty() {
            0
        } else {
            2 + items
                .iter()
                .map(|item| wrap_to_width(&item.content, content_width).len())
                .sum::<usize>()
        }
    })
}

fn render_mcp_status(
    lines: &mut Vec<Line<'static>>,
    state: &TuiState,
    width: usize,
    row_limit: usize,
    theme: Theme,
) {
    use crate::mcp::McpServerStatus;
    use crate::tui::state::McpDiscoveryState;

    if row_limit == 0 {
        return;
    }

    match state.mcp_discovery {
        McpDiscoveryState::Loading => {
            compact_field(
                lines,
                state.t("sidebar.mcp"),
                &state.t("sidebar.mcp_loading"),
                width,
                theme.notice,
                theme,
            );
        }
        McpDiscoveryState::Unavailable => {
            compact_field(
                lines,
                state.t("sidebar.mcp"),
                &state.t("sidebar.mcp_unavailable"),
                width,
                theme.error,
                theme,
            );
        }
        McpDiscoveryState::Ready if state.mcp_servers.is_empty() => {}
        McpDiscoveryState::Ready => {
            lines.push(Line::from(Span::styled(
                state.t("sidebar.mcp"),
                Style::default()
                    .fg(theme.accent)
                    .bg(theme.element_bg)
                    .add_modifier(Modifier::BOLD),
            )));
            for server in state.mcp_servers.iter().take(row_limit.saturating_sub(1)) {
                let (marker, status, color) = if state.mcp_updating.contains(&server.name) {
                    ("◌", state.t("status.updating"), theme.warning)
                } else {
                    match server.status {
                        McpServerStatus::Disabled => {
                            ("○", state.t("status.disabled"), theme.muted_text)
                        }
                        McpServerStatus::Online { tool_count } => (
                            "●",
                            state.t_fmt(
                                "sidebar.mcp_server_online",
                                &[("count", &tool_count.to_string())],
                            ),
                            theme.success,
                        ),
                        McpServerStatus::Offline { .. } => {
                            ("●", state.t("status.offline"), theme.error)
                        }
                    }
                };
                lines.push(mcp_server_line(
                    &server.name,
                    marker,
                    &status,
                    width,
                    color,
                    theme,
                ));
            }
        }
    }
}

fn mcp_server_line(
    name: &str,
    marker: &str,
    status: &str,
    width: usize,
    color: Color,
    theme: Theme,
) -> Line<'static> {
    let marker_width = display_width(marker).saturating_add(1);
    if width <= marker_width {
        return Line::from(Span::styled(
            truncate_to_width(marker, width),
            Style::default().fg(color).bg(theme.element_bg),
        ));
    }
    let content_width = width.saturating_sub(marker_width);
    let status = truncate_to_width(status, content_width);
    let status_width = display_width(&status);
    let gap_width = usize::from(status_width > 0 && content_width > status_width);
    let name_width = content_width.saturating_sub(status_width.saturating_add(gap_width));
    Line::from(vec![
        Span::styled(
            format!("{marker} "),
            Style::default().fg(color).bg(theme.element_bg),
        ),
        Span::styled(
            padded_label(&truncate_to_width(name, name_width), name_width),
            Style::default().fg(theme.text).bg(theme.element_bg),
        ),
        Span::styled(" ".repeat(gap_width), Style::default().bg(theme.element_bg)),
        Span::styled(
            status,
            Style::default()
                .fg(color)
                .bg(theme.element_bg)
                .add_modifier(Modifier::BOLD),
        ),
    ])
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

fn wrap_to_width(text: &str, max_width: usize) -> Vec<String> {
    if max_width == 0 {
        return vec![String::new()];
    }

    let mut rows = Vec::new();
    for logical_line in text.lines() {
        let mut remaining = logical_line;
        if remaining.is_empty() {
            rows.push(String::new());
            continue;
        }
        while !remaining.is_empty() {
            let row = truncate_to_width_without_ellipsis(remaining, max_width);
            if row.is_empty() {
                break;
            }
            let consumed = row.len();
            rows.push(row);
            remaining = &remaining[consumed..];
        }
    }
    if rows.is_empty() {
        rows.push(String::new());
    }
    rows
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
    fn sidebar_renders_detailed_context_usage_breakdown() {
        let mut state = TuiState::default();
        state.set_language(Some(crate::tui::i18n::Language::En));
        state.model_token_usage = Some(crate::tui::state::ModelTokenUsage {
            used_tokens: 70_000,
            context_window_tokens: 128_000,
            input_tokens: 64_000,
            output_tokens: 6_000,
            cached_tokens: 24_000,
            prompt_composition: vec![
                crate::agent::PromptCompositionEntry {
                    category: "system".into(),
                    estimated_tokens: 18_000,
                    segments: 2,
                },
                crate::agent::PromptCompositionEntry {
                    category: "skills".into(),
                    estimated_tokens: 12_000,
                    segments: 2,
                },
                crate::agent::PromptCompositionEntry {
                    category: "tools".into(),
                    estimated_tokens: 20_000,
                    segments: 1,
                },
                crate::agent::PromptCompositionEntry {
                    category: "messages".into(),
                    estimated_tokens: 20_000,
                    segments: 3,
                },
            ],
            cache_report: Some(crate::agent::CacheUsageReport {
                configured: true,
                hint_serialized: true,
                retention_sent: None,
                stable_prefix_segments: 2,
                stable_prompt_tokens: 42_000,
                volatile_prompt_tokens: 22_000,
                cacheable_prefix_tokens: 40_000,
                stable_after_boundary_tokens: 2_000,
                local_prefix_fingerprint: None,
                routing_key: None,
                actual_cached_tokens: Some(24_000),
            }),
        });
        let backend = TestBackend::new(42, 28);
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

        for expected in [
            "Context usage",
            "70.0k / 128.0k · 55%",
            "System prompt",
            "16.5k",
            "Skills",
            "11.0k",
            "Tools",
            "18.3k",
            "Messages",
            "18.3k",
            "Output",
            "6.0k",
            "Remaining",
            "58.0k",
        ] {
            assert!(
                rendered.contains(expected),
                "missing {expected:?}: {rendered}"
            );
        }
    }

    #[test]
    fn context_breakdown_includes_current_output_tokens() {
        let mut state = TuiState::default();
        state.set_language(Some(crate::tui::i18n::Language::En));
        state.model_token_usage = Some(crate::tui::state::ModelTokenUsage {
            used_tokens: 12_000,
            context_window_tokens: 20_000,
            input_tokens: 10_000,
            output_tokens: 2_000,
            cached_tokens: 0,
            cache_report: None,
            prompt_composition: vec![crate::agent::PromptCompositionEntry {
                category: "messages".into(),
                estimated_tokens: 10_000,
                segments: 1,
            }],
        });
        let backend = TestBackend::new(42, 22);
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

        assert!(rendered.contains("Messages        10.0k"), "{rendered}");
        assert!(rendered.contains("Output          2.0k"), "{rendered}");
        let bar = terminal
            .backend()
            .buffer()
            .content()
            .chunks(42)
            .map(|row| row.iter().map(|cell| cell.symbol()).collect::<String>())
            .find(|row| row.matches('█').count() > 0)
            .expect("context bar");
        assert_eq!(bar.matches('█').count(), 23, "{bar}");
        assert!(rendered.contains("Remaining       8.0k"), "{rendered}");
    }

    #[test]
    fn sidebar_wraps_long_context_summary_and_todo_content() {
        let mut state = TuiState::default();
        state.set_language(Some(crate::tui::i18n::Language::En));
        state.current_context_branch = "history-13".into();
        state.mcp_discovery = crate::tui::state::McpDiscoveryState::Ready;
        state.model_token_usage = Some(crate::tui::state::ModelTokenUsage {
            used_tokens: 84_100,
            context_window_tokens: 1_000_000,
            input_tokens: 84_100,
            output_tokens: 0,
            cached_tokens: 0,
            cache_report: None,
            prompt_composition: Vec::new(),
        });
        state.latest_todo = Some(crate::tui::timeline::TodoView {
            items: vec![crate::agent::TodoItem {
                id: "wrap".into(),
                content: "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789".into(),
                status: crate::agent::TodoStatus::InProgress,
            }],
            auto_continue: crate::agent::AutoContinueState::default(),
        });
        let backend = TestBackend::new(32, 24);
        let mut terminal = Terminal::new(backend).expect("terminal");
        terminal
            .draw(|frame| render_sidebar(frame, &state, frame.area(), Theme::dark()))
            .expect("draw");
        let rows = (0..terminal.backend().buffer().area.height)
            .map(|y| {
                terminal
                    .backend()
                    .buffer()
                    .content()
                    .iter()
                    .skip(y as usize * 32)
                    .take(32)
                    .map(|cell| cell.symbol())
                    .collect::<String>()
            })
            .collect::<Vec<_>>();

        let context_row = rows
            .iter()
            .position(|row| row.contains("history-13"))
            .expect("context branch row");
        assert!(rows[context_row + 1].contains("84.1k / 1.0m (8%)"));
        let todo_start = rows
            .iter()
            .position(|row| row.contains("Todos  1"))
            .expect("todo heading");
        let todo_rows = rows
            .iter()
            .skip(todo_start + 1)
            .take(2)
            .cloned()
            .collect::<Vec<_>>();
        assert_eq!(todo_rows.len(), 2, "{rows:?}");
        assert_eq!(
            todo_rows
                .iter()
                .flat_map(|row| row.chars().filter(|ch| ch.is_ascii_alphanumeric()))
                .collect::<String>(),
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        );
        assert!(!todo_rows.iter().any(|row| row.contains('…')));
    }

    #[test]
    fn context_details_leave_room_for_mcp_and_todos() {
        let mut state = TuiState::default();
        state.set_language(Some(crate::tui::i18n::Language::En));
        state.set_mcp_servers(vec![crate::mcp::McpServerCatalogEntry {
            name: "docs".into(),
            enabled: true,
            status: crate::mcp::McpServerStatus::Online { tool_count: 2 },
        }]);
        state.model_token_usage = Some(crate::tui::state::ModelTokenUsage {
            used_tokens: 10_000,
            context_window_tokens: 20_000,
            input_tokens: 9_000,
            output_tokens: 1_000,
            cached_tokens: 0,
            cache_report: None,
            prompt_composition: (0..8)
                .map(|index| crate::agent::PromptCompositionEntry {
                    category: format!("other-{index}"),
                    estimated_tokens: 1_000,
                    segments: 1,
                })
                .collect(),
        });
        state.latest_todo = Some(crate::tui::timeline::TodoView {
            items: vec![crate::agent::TodoItem {
                id: "keep".into(),
                content: "important todo".into(),
                status: crate::agent::TodoStatus::Pending,
            }],
            auto_continue: crate::agent::AutoContinueState::default(),
        });
        let backend = TestBackend::new(42, 20);
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

        assert!(rendered.contains("MCP"), "{rendered}");
        assert!(rendered.contains("docs"), "{rendered}");
        assert!(rendered.contains("2 tools"), "{rendered}");
        assert!(rendered.contains("Todos  1"), "{rendered}");
        assert!(rendered.contains("important todo"), "{rendered}");
    }

    #[test]
    fn mcp_list_respects_width_and_height_limits() {
        let mut state = TuiState::default();
        state.set_language(Some(crate::tui::i18n::Language::En));
        state.set_mcp_servers(
            (0..6)
                .map(|index| crate::mcp::McpServerCatalogEntry {
                    name: format!("very-long-mcp-server-name-{index}"),
                    enabled: true,
                    status: crate::mcp::McpServerStatus::Online { tool_count: 12 },
                })
                .collect(),
        );
        let backend = TestBackend::new(24, 10);
        let mut terminal = Terminal::new(backend).expect("terminal");
        terminal
            .draw(|frame| render_sidebar(frame, &state, frame.area(), Theme::dark()))
            .expect("draw");
        let rows = (0..terminal.backend().buffer().area.height)
            .map(|y| {
                terminal
                    .backend()
                    .buffer()
                    .content()
                    .iter()
                    .skip(y as usize * 24)
                    .take(24)
                    .map(|cell| cell.symbol())
                    .collect::<String>()
            })
            .collect::<Vec<_>>();

        assert!(
            rows.iter().filter(|row| row.contains("12 tools")).count() <= 2,
            "{rows:?}"
        );
        assert!(rows.iter().all(|row| display_width(row) <= 24), "{rows:?}");
        assert!(rows.iter().any(|row| row.contains("Ctrl-X B")), "{rows:?}");
    }

    #[test]
    fn sidebar_renders_mcp_server_list() {
        let mut state = TuiState::default();
        state.set_language(Some(crate::tui::i18n::Language::En));
        state.set_mcp_servers(vec![
            crate::mcp::McpServerCatalogEntry {
                name: "docs".into(),
                enabled: true,
                status: crate::mcp::McpServerStatus::Online { tool_count: 3 },
            },
            crate::mcp::McpServerCatalogEntry {
                name: "broken".into(),
                enabled: true,
                status: crate::mcp::McpServerStatus::Offline {
                    message: "unreachable".into(),
                },
            },
        ]);
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

        for expected in ["MCP", "docs", "3 tools", "broken", "Offline"] {
            assert!(
                rendered.contains(expected),
                "missing {expected:?}: {rendered}"
            );
        }
    }

    #[test]
    fn context_bar_fills_used_share_and_leaves_capacity_visible() {
        let composition = vec![
            crate::agent::PromptCompositionEntry {
                category: "system".into(),
                estimated_tokens: 4_000,
                segments: 1,
            },
            crate::agent::PromptCompositionEntry {
                category: "messages".into(),
                estimated_tokens: 6_000,
                segments: 1,
            },
        ];
        let line = context_bar_line(20, &composition, 10_000, 10_000, 0, 20_000, Theme::dark());
        assert_eq!(line.width(), 20);
        assert_eq!(line.spans[0].content.chars().count(), 4);
        assert_eq!(line.spans[1].content.chars().count(), 6);
        assert_eq!(line.spans[2].content.chars().count(), 10);
        assert_eq!(line.spans[2].style.bg, Some(Theme::dark().elevated_bg));
    }

    #[test]
    fn sidebar_labels_align_by_terminal_display_width() {
        assert_eq!(display_width(&padded_label("模型", 10)), 10);
        assert_eq!(display_width(&padded_label("Git", 10)), 10);
        assert_eq!(display_width(&padded_label("服务商", 10)), 10);
    }
}
