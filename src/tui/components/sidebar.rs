use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Paragraph, Wrap},
};

use crate::agent::TodoStatus;

use super::super::{measure::display_width, state::TuiState, surface, theme::Theme};

pub fn render_sidebar(frame: &mut Frame<'_>, state: &mut TuiState, area: Rect, theme: Theme) {
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

    state.last_sidebar_area = inner;
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

    let context_header_line = lines.len();
    if let Some(usage) = state
        .active_model_token_usage()
        .filter(|usage| usage.context_window_tokens > 0)
    {
        let percent = context_used_percent(usage.input_tokens, usage.context_window_tokens);
        let usage_summary = format!(
            "{} / {} ({percent}%)",
            compact_count(usage.input_tokens),
            compact_count(usage.context_window_tokens)
        );
        collapsible_context_field(
            &mut lines,
            state.sidebar_context_expanded,
            state.t("sidebar.context"),
            &state.current_context_branch,
            &usage_summary,
            inner.width as usize,
            context_usage_color(percent, theme),
            theme,
        );
    } else {
        collapsible_field(
            &mut lines,
            state.sidebar_context_expanded,
            state.t("sidebar.context"),
            &state.current_context_branch,
            inner.width as usize,
            theme.notice,
            theme,
        );
    }
    let mcp_rows = mcp_row_count(state);
    if state.sidebar_context_expanded
        && let Some(usage) = state
            .active_model_token_usage()
            .filter(|usage| usage.context_window_tokens > 0)
    {
        render_context_usage_details(&mut lines, usage, inner.width as usize, state, theme);
    }

    let context_rendered = state
        .active_model_token_usage()
        .is_some_and(|usage| usage.context_window_tokens > 0);
    let mcp_rendered = mcp_rows > 0
        && (state.mcp_discovery != crate::tui::state::McpDiscoveryState::Ready
            || !state.mcp_servers.is_empty());
    if context_rendered && mcp_rendered {
        lines.push(Line::default());
    }
    let mcp_header_line = mcp_rendered.then_some(lines.len());
    render_mcp_status(&mut lines, state, inner.width as usize, mcp_rows, theme);

    let mut todos_header_line = None;
    if let Some(todo) = state.latest_todo.as_ref() {
        let items = todo.items.iter().collect::<Vec<_>>();
        if !items.is_empty() {
            lines.push(Line::default());
            todos_header_line = Some(lines.len());
            lines.push(Line::from(vec![
                section_arrow(state.sidebar_todos_expanded, theme),
                Span::styled(
                    state.t("sidebar.todos"),
                    Style::default()
                        .fg(theme.approval)
                        .bg(theme.element_bg)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(format!("  {}", items.len()), label_style),
            ]));
            if state.sidebar_todos_expanded {
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
    }

    let footer_height = u16::from(inner.height > 1);
    let areas = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(0), Constraint::Length(footer_height)])
        .split(inner);
    let context_header_row =
        rendered_row_before(&lines, context_header_line, areas[0].width, style);
    let mcp_header_row =
        mcp_header_line.map(|line| rendered_row_before(&lines, line, areas[0].width, style));
    let todos_header_row =
        todos_header_line.map(|line| rendered_row_before(&lines, line, areas[0].width, style));
    let paragraph = Paragraph::new(lines)
        .style(style)
        .wrap(Wrap { trim: false });
    let total_rows = paragraph.line_count(areas[0].width);
    state.sync_sidebar_scroll(total_rows, areas[0].height);
    let scroll = state.sidebar_scroll;
    state.last_sidebar_context_header = sidebar_header_rect(areas[0], context_header_row, scroll);
    state.last_sidebar_mcp_header = mcp_header_row
        .map(|row| sidebar_header_rect(areas[0], row, scroll))
        .unwrap_or_default();
    state.last_sidebar_todos_header = todos_header_row
        .map(|row| sidebar_header_rect(areas[0], row, scroll))
        .unwrap_or_default();
    frame.render_widget(paragraph.scroll((scroll, 0)), areas[0]);
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

fn section_arrow(expanded: bool, theme: Theme) -> Span<'static> {
    Span::styled(
        if expanded { "▾ " } else { "▸ " },
        Style::default()
            .fg(theme.dim_text)
            .bg(theme.element_bg)
            .add_modifier(Modifier::BOLD),
    )
}

fn rendered_row_before(lines: &[Line<'static>], line: usize, width: u16, style: Style) -> usize {
    Paragraph::new(lines[..line].to_vec())
        .style(style)
        .wrap(Wrap { trim: false })
        .line_count(width)
}

fn sidebar_header_rect(area: Rect, rendered_line: usize, scroll: u16) -> Rect {
    let visible_line = rendered_line.saturating_sub(scroll as usize);
    if rendered_line < scroll as usize || visible_line >= area.height as usize {
        Rect::default()
    } else {
        Rect::new(
            area.x,
            area.y.saturating_add(visible_line as u16),
            area.width,
            1,
        )
    }
}

fn render_context_usage_details(
    lines: &mut Vec<Line<'static>>,
    usage: &crate::tui::state::ModelTokenUsage,
    width: usize,
    state: &TuiState,
    theme: Theme,
) {
    if width < 8 {
        return;
    }

    let estimated_composition_tokens =
        usage.prompt_composition.iter().fold(0u64, |total, entry| {
            total.saturating_add(entry.estimated_tokens)
        });
    let display_used_tokens = usage.input_tokens;
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
        usage.context_window_tokens,
        theme,
    ));

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
    details.push(context_detail_line(
        state.t("sidebar.context_remaining"),
        remaining,
        theme.muted_text,
        theme,
    ));
    lines.extend(details);
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
    context_window_tokens: u64,
    theme: Theme,
) -> Line<'static> {
    let bar_width = width.max(1);
    let target_cells = proportional_cells(actual_input_tokens, context_window_tokens, bar_width);
    let segments = composition
        .iter()
        .enumerate()
        .map(|(index, entry)| {
            (
                scaled_composition_tokens(
                    entry.estimated_tokens,
                    estimated_total,
                    actual_input_tokens,
                ),
                context_composition_color(entry, index, theme),
            )
        })
        .collect::<Vec<_>>();

    context_bar_spans(bar_width, target_cells, &segments, theme)
}

fn context_bar_spans(
    width: usize,
    target_cells: usize,
    segments: &[(u64, Color)],
    theme: Theme,
) -> Line<'static> {
    const UNITS_PER_CELL: usize = 8;

    let target_units = target_cells.saturating_mul(UNITS_PER_CELL);
    let weights = segments
        .iter()
        .map(|(tokens, _)| *tokens)
        .collect::<Vec<_>>();
    let mut units = allocate_bar_units(&weights, target_units);
    let fallback_color = segments
        .iter()
        .zip(&units)
        .find_map(|((_, color), units)| (*units > 0).then_some(*color))
        .unwrap_or(theme.accent);
    units = spread_bar_boundaries(units, target_units, UNITS_PER_CELL);
    let mut subcells = Vec::with_capacity(width.saturating_mul(UNITS_PER_CELL));
    for ((_, color), units) in segments.iter().zip(units) {
        subcells.extend(std::iter::repeat_n(Some(*color), units));
    }
    subcells.resize(target_units, Some(fallback_color));
    subcells.truncate(target_units);
    subcells.resize(width.saturating_mul(UNITS_PER_CELL), None);

    Line::from(
        subcells
            .chunks(UNITS_PER_CELL)
            .map(|cell| context_bar_cell(cell, theme))
            .collect::<Vec<_>>(),
    )
}

fn context_bar_cell(cell: &[Option<Color>], theme: Theme) -> Span<'static> {
    debug_assert_eq!(cell.len(), 8);
    if cell.iter().all(|color| *color == cell[0]) {
        return match cell[0] {
            Some(color) => Span::styled("█", Style::default().fg(color).bg(theme.element_bg)),
            None => Span::styled(
                "█",
                Style::default().fg(theme.elevated_bg).bg(theme.element_bg),
            ),
        };
    }

    let runs =
        cell.iter()
            .copied()
            .fold(Vec::<(Option<Color>, usize)>::new(), |mut runs, color| {
                if let Some((last_color, count)) = runs.last_mut()
                    && *last_color == color
                {
                    *count += 1;
                } else {
                    runs.push((color, 1));
                }
                runs
            });
    if runs.len() > 2 {
        let (color, _) = runs
            .into_iter()
            .enumerate()
            .max_by_key(|(index, (_, count))| (*count, *index))
            .map(|(_, run)| run)
            .unwrap_or((None, cell.len()));
        return match color {
            Some(color) => Span::styled("█", Style::default().fg(color).bg(theme.element_bg)),
            None => Span::styled(
                "█",
                Style::default().fg(theme.elevated_bg).bg(theme.element_bg),
            ),
        };
    }

    let split = cell
        .windows(2)
        .position(|pair| pair[0] != pair[1])
        .map_or(cell.len(), |index| index + 1);
    let foreground = cell[..split]
        .iter()
        .flatten()
        .copied()
        .next()
        .unwrap_or(theme.elevated_bg);
    let background = cell[split..]
        .iter()
        .flatten()
        .copied()
        .next()
        .unwrap_or(theme.elevated_bg);
    Span::styled(
        partial_context_block(split),
        Style::default().fg(foreground).bg(background),
    )
}

fn partial_context_block(units: usize) -> &'static str {
    match units.clamp(1, 7) {
        1 => "▏",
        2 => "▎",
        3 => "▍",
        4 => "▌",
        5 => "▋",
        6 => "▊",
        _ => "▉",
    }
}

fn allocate_bar_units(weights: &[u64], target_units: usize) -> Vec<usize> {
    let mut units = vec![0; weights.len()];
    if target_units == 0 || weights.is_empty() {
        return units;
    }
    let positive = weights
        .iter()
        .enumerate()
        .filter_map(|(index, weight)| (*weight > 0).then_some(index))
        .collect::<Vec<_>>();
    if positive.is_empty() {
        return units;
    }

    let remaining = if positive.len() <= target_units {
        for index in &positive {
            units[*index] = 1;
        }
        target_units - positive.len()
    } else {
        target_units
    };
    if remaining == 0 {
        return units;
    }

    let total = positive.iter().fold(0u128, |sum, index| {
        sum.saturating_add(weights[*index] as u128)
    });
    let mut remainders = Vec::with_capacity(positive.len());
    let mut allocated = 0usize;
    for index in positive {
        let scaled = (weights[index] as u128).saturating_mul(remaining as u128);
        let base = usize::try_from(scaled / total).unwrap_or(remaining);
        units[index] = units[index].saturating_add(base);
        allocated = allocated.saturating_add(base);
        remainders.push((scaled % total, index));
    }
    remainders.sort_by(|left, right| right.0.cmp(&left.0).then_with(|| left.1.cmp(&right.1)));
    for (_, index) in remainders
        .into_iter()
        .take(remaining.saturating_sub(allocated))
    {
        units[index] = units[index].saturating_add(1);
    }
    units
}

fn spread_bar_boundaries(
    units: Vec<usize>,
    total_units: usize,
    units_per_cell: usize,
) -> Vec<usize> {
    if units.len() < 2 || units_per_cell == 0 || units.iter().any(|units| *units == 0) {
        return units;
    }

    let mut original_boundaries = Vec::with_capacity(units.len().saturating_sub(1));
    let mut cumulative = 0usize;
    for units in units.iter().take(units.len() - 1) {
        cumulative = cumulative.saturating_add(*units);
        original_boundaries.push(cumulative);
    }

    let mut adjusted_boundaries = Vec::with_capacity(original_boundaries.len());
    let mut mixed_cells = std::collections::HashSet::new();
    let mut previous = 0usize;
    for (index, desired) in original_boundaries.into_iter().enumerate() {
        let remaining_segments = units.len().saturating_sub(index + 1);
        let lower = previous.saturating_add(1);
        let upper = total_units.saturating_sub(remaining_segments);
        let desired = desired.clamp(lower, upper);
        let boundary =
            nearest_visible_boundary(desired, lower, upper, units_per_cell, &mixed_cells);
        if boundary % units_per_cell != 0 {
            mixed_cells.insert(boundary / units_per_cell);
        }
        adjusted_boundaries.push(boundary);
        previous = boundary;
    }

    let mut adjusted = Vec::with_capacity(units.len());
    let mut start = 0usize;
    for boundary in adjusted_boundaries {
        adjusted.push(boundary.saturating_sub(start));
        start = boundary;
    }
    adjusted.push(total_units.saturating_sub(start));
    adjusted
}

fn nearest_visible_boundary(
    desired: usize,
    lower: usize,
    upper: usize,
    units_per_cell: usize,
    mixed_cells: &std::collections::HashSet<usize>,
) -> usize {
    let valid = |boundary: usize| {
        boundary % units_per_cell == 0 || !mixed_cells.contains(&(boundary / units_per_cell))
    };
    if valid(desired) {
        return desired;
    }

    let desired_cell = desired / units_per_cell;
    let next_cell_start = desired_cell
        .saturating_add(1)
        .saturating_mul(units_per_cell);
    let next_cell_boundary = next_cell_start.saturating_add(1);
    if next_cell_boundary >= lower && next_cell_boundary <= upper && valid(next_cell_boundary) {
        return next_cell_boundary;
    }

    for distance in 1..=upper.saturating_sub(lower) {
        let left = desired.saturating_sub(distance);
        if left >= lower && valid(left) {
            return left;
        }
        let right = desired.saturating_add(distance);
        if right <= upper && valid(right) {
            return right;
        }
    }
    desired
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

fn compact_field_spans(
    label: &str,
    value: &str,
    width: usize,
    value_color: Color,
    theme: Theme,
) -> Vec<Span<'static>> {
    let label = padded_label(label, LABEL_COLUMN_WIDTH);
    let value_width = width.saturating_sub(LABEL_COLUMN_WIDTH);
    vec![
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
    ]
}

fn compact_field_width(branch: &str, usage: &str) -> usize {
    LABEL_COLUMN_WIDTH + display_width(branch) + 2 + display_width(usage)
}

fn collapsible_field(
    lines: &mut Vec<Line<'static>>,
    expanded: bool,
    label: String,
    value: &str,
    width: usize,
    value_color: Color,
    theme: Theme,
) {
    let content_width = width.saturating_sub(2);
    let mut line = vec![section_arrow(expanded, theme)];
    line.extend(compact_field_spans(
        &label,
        value,
        content_width,
        value_color,
        theme,
    ));
    lines.push(Line::from(line));
}

fn collapsible_context_field(
    lines: &mut Vec<Line<'static>>,
    expanded: bool,
    label: String,
    branch: &str,
    usage: &str,
    width: usize,
    value_color: Color,
    theme: Theme,
) {
    let content_width = width.saturating_sub(2);
    if compact_field_width(branch, usage) <= content_width {
        let value = format!("{branch}  {usage}");
        collapsible_field(lines, expanded, label, &value, width, value_color, theme);
        return;
    }

    let mut line = vec![section_arrow(expanded, theme)];
    line.extend(compact_field_spans(
        &label,
        branch,
        content_width,
        value_color,
        theme,
    ));
    lines.push(Line::from(line));
    lines.push(Line::from(vec![
        Span::raw(" ".repeat(LABEL_COLUMN_WIDTH)),
        Span::styled(
            truncate_to_width(usage, content_width.saturating_sub(LABEL_COLUMN_WIDTH)),
            Style::default()
                .fg(value_color)
                .bg(theme.element_bg)
                .add_modifier(Modifier::BOLD),
        ),
    ]));
}

fn mcp_row_count(state: &TuiState) -> usize {
    match state.mcp_discovery {
        crate::tui::state::McpDiscoveryState::Ready => {
            usize::from(!state.mcp_servers.is_empty()) + state.mcp_servers.len()
        }
        crate::tui::state::McpDiscoveryState::Loading
        | crate::tui::state::McpDiscoveryState::Unavailable => 1,
    }
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
            collapsible_field(
                lines,
                state.sidebar_mcp_expanded,
                state.t("sidebar.mcp"),
                &state.t("sidebar.mcp_loading"),
                width,
                theme.notice,
                theme,
            );
        }
        McpDiscoveryState::Unavailable => {
            collapsible_field(
                lines,
                state.sidebar_mcp_expanded,
                state.t("sidebar.mcp"),
                &state.t("sidebar.mcp_unavailable"),
                width,
                theme.error,
                theme,
            );
        }
        McpDiscoveryState::Ready if state.mcp_servers.is_empty() => {}
        McpDiscoveryState::Ready => {
            lines.push(Line::from(vec![
                section_arrow(state.sidebar_mcp_expanded, theme),
                Span::styled(
                    state.t("sidebar.mcp"),
                    Style::default()
                        .fg(theme.accent)
                        .bg(theme.element_bg)
                        .add_modifier(Modifier::BOLD),
                ),
            ]));
            if state.sidebar_mcp_expanded {
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
            .draw(|frame| render_sidebar(frame, &mut state, frame.area(), Theme::dark()))
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
            .draw(|frame| render_sidebar(frame, &mut state, frame.area(), Theme::dark()))
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
            "64.0k / 128.0k · 50%",
            "System prompt",
            "16.5k",
            "Skills",
            "11.0k",
            "Tools",
            "18.3k",
            "Messages",
            "18.3k",
            "Remaining",
            "64.0k",
        ] {
            assert!(
                rendered.contains(expected),
                "missing {expected:?}: {rendered}"
            );
        }
        assert!(!rendered.contains("Output"), "{rendered}");
    }

    #[test]
    fn context_breakdown_excludes_current_output_tokens() {
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
            .draw(|frame| render_sidebar(frame, &mut state, frame.area(), Theme::dark()))
            .expect("draw");
        let rendered = terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|cell| cell.symbol())
            .collect::<String>();

        assert!(rendered.contains("Messages        10.0k"), "{rendered}");
        assert!(!rendered.contains("Output"), "{rendered}");
        assert!(rendered.contains("10.0k / 20.0k · 50%"), "{rendered}");
        let bar_cells = terminal
            .backend()
            .buffer()
            .content()
            .chunks(42)
            .filter(|row| row.iter().any(|cell| "█▉▊▋▌▍▎▏".contains(cell.symbol())))
            .max_by_key(|row| {
                row.iter()
                    .filter(|cell| "█▉▊▋▌▍▎▏".contains(cell.symbol()))
                    .count()
            })
            .expect("context bar");
        assert_eq!(
            bar_cells
                .iter()
                .filter(|cell| cell.symbol() == "█" && cell.fg != Theme::dark().elevated_bg)
                .count(),
            19
        );
        assert_eq!(
            bar_cells
                .iter()
                .filter(|cell| "▉▊▋▌▍▎▏".contains(cell.symbol()))
                .count(),
            1
        );
        assert!(rendered.contains("Remaining       10.0k"), "{rendered}");
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
            .draw(|frame| render_sidebar(frame, &mut state, frame.area(), Theme::dark()))
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
        assert!(
            rows.iter()
                .skip(context_row + 1)
                .take(2)
                .any(|row| row.contains("84.1k / 1.0m")),
            "{rows:?}"
        );
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
            .draw(|frame| render_sidebar(frame, &mut state, frame.area(), Theme::dark()))
            .expect("draw");
        let rendered = terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|cell| cell.symbol())
            .collect::<String>();

        assert!(state.sidebar_max_scroll > 0);
        state.scroll_sidebar_to_bottom();
        terminal
            .draw(|frame| render_sidebar(frame, &mut state, frame.area(), Theme::dark()))
            .expect("draw scrolled sidebar");
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
        let rows = (0..terminal.backend().buffer().area.height)
            .map(|y| {
                terminal
                    .backend()
                    .buffer()
                    .content()
                    .iter()
                    .skip(y as usize * 42)
                    .take(42)
                    .map(|cell| cell.symbol())
                    .collect::<String>()
            })
            .collect::<Vec<_>>();
        let mcp_row = rows
            .iter()
            .position(|row| row.contains("MCP"))
            .expect("MCP heading");
        assert!(
            rows[mcp_row.saturating_sub(1)]
                .trim_matches(|ch: char| ch == '▎' || ch.is_whitespace())
                .is_empty(),
            "{rows:?}"
        );
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
            .draw(|frame| render_sidebar(frame, &mut state, frame.area(), Theme::dark()))
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
        assert!(
            rows.iter().all(|row| !row.contains("Scroll panel")),
            "{rows:?}"
        );
    }

    #[test]
    fn sidebar_collapses_context_mcp_and_todo_sections() {
        let mut state = TuiState::default();
        state.set_language(Some(crate::tui::i18n::Language::En));
        state.sidebar_context_expanded = false;
        state.sidebar_mcp_expanded = false;
        state.sidebar_todos_expanded = false;
        state.model_token_usage = Some(crate::tui::state::ModelTokenUsage {
            used_tokens: 10_000,
            context_window_tokens: 20_000,
            input_tokens: 10_000,
            output_tokens: 0,
            cached_tokens: 0,
            cache_report: None,
            prompt_composition: vec![crate::agent::PromptCompositionEntry {
                category: "messages".into(),
                estimated_tokens: 10_000,
                segments: 1,
            }],
        });
        state.set_mcp_servers(vec![crate::mcp::McpServerCatalogEntry {
            name: "docs".into(),
            enabled: true,
            status: crate::mcp::McpServerStatus::Online { tool_count: 2 },
        }]);
        state.latest_todo = Some(crate::tui::timeline::TodoView {
            items: vec![crate::agent::TodoItem {
                id: "todo".into(),
                content: "hidden todo".into(),
                status: crate::agent::TodoStatus::Pending,
            }],
            auto_continue: crate::agent::AutoContinueState::default(),
        });
        let backend = TestBackend::new(42, 24);
        let mut terminal = Terminal::new(backend).expect("terminal");
        terminal
            .draw(|frame| render_sidebar(frame, &mut state, frame.area(), Theme::dark()))
            .expect("draw");
        let rendered = terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|cell| cell.symbol())
            .collect::<String>();

        assert!(rendered.contains("▸ Context"), "{rendered}");
        assert!(rendered.contains("▸ MCP"), "{rendered}");
        assert!(rendered.contains("▸ Todos"), "{rendered}");
        assert!(!rendered.contains("Context usage"), "{rendered}");
        assert!(!rendered.contains("docs"), "{rendered}");
        assert!(!rendered.contains("hidden todo"), "{rendered}");
    }

    #[test]
    fn sidebar_scroll_reveals_overflow_content() {
        let mut state = TuiState::default();
        state.sidebar_scroll = 3;
        state.set_mcp_servers(
            (0..8)
                .map(|index| crate::mcp::McpServerCatalogEntry {
                    name: format!("server-{index}"),
                    enabled: true,
                    status: crate::mcp::McpServerStatus::Online {
                        tool_count: index + 1,
                    },
                })
                .collect(),
        );
        state.latest_todo = Some(crate::tui::timeline::TodoView {
            items: (0..5)
                .map(|index| crate::agent::TodoItem {
                    id: format!("todo-{index}"),
                    content: format!("long scrolling todo item {index}"),
                    status: crate::agent::TodoStatus::Pending,
                })
                .collect(),
            auto_continue: crate::agent::AutoContinueState::default(),
        });
        let backend = TestBackend::new(42, 16);
        let mut terminal = Terminal::new(backend).expect("terminal");
        terminal
            .draw(|frame| render_sidebar(frame, &mut state, frame.area(), Theme::dark()))
            .expect("draw");

        assert!(state.sidebar_max_scroll > 0);
        assert_eq!(state.sidebar_scroll, 3.min(state.sidebar_max_scroll));
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
            .draw(|frame| render_sidebar(frame, &mut state, frame.area(), Theme::dark()))
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
        let line = context_bar_line(20, &composition, 10_000, 10_000, 20_000, Theme::dark());
        assert_eq!(line.width(), 20);
        assert_eq!(line.spans.len(), 20);
        assert_eq!(
            line.spans
                .iter()
                .filter(|span| span.style.fg == Some(Theme::dark().elevated_bg))
                .count(),
            10
        );
    }

    #[test]
    fn context_bar_keeps_small_positive_categories_visible() {
        let units = spread_bar_boundaries(
            allocate_bar_units(&[7_600, 7_300, 4_100, 668_800, 4_600], 43 * 8),
            43 * 8,
            8,
        );

        assert_eq!(units.iter().sum::<usize>(), 43 * 8);
        assert!(units.iter().all(|units| *units >= 1), "{units:?}");
        assert!(units[3] > units[0]);
        let boundaries = units
            .iter()
            .scan(0usize, |total, units| {
                *total += *units;
                Some(*total)
            })
            .take(units.len() - 1)
            .filter(|boundary| *boundary % 8 != 0)
            .map(|boundary| boundary / 8)
            .collect::<Vec<_>>();
        assert_eq!(
            boundaries
                .iter()
                .copied()
                .collect::<std::collections::HashSet<_>>()
                .len(),
            boundaries.len(),
            "{units:?}"
        );
    }

    #[test]
    fn context_bar_small_categories_use_two_color_cells() {
        let theme = Theme::dark();
        let segments = [
            (7_600, theme.notice),
            (7_300, theme.warning),
            (4_100, theme.success),
            (668_800, theme.user),
            (4_600, theme.error),
        ];
        let line = context_bar_spans(43, 43, &segments, theme);

        assert_eq!(line.width(), 43);
        for pair in segments.windows(2) {
            assert!(
                line.spans.iter().any(|span| {
                    span.style.fg == Some(pair[0].1)
                        && span.style.bg == Some(pair[1].1)
                        && "▉▊▋▌▍▎▏".contains(span.content.as_ref())
                }),
                "missing boundary for {pair:?}"
            );
        }
    }

    #[test]
    fn empty_context_bar_uses_visible_track_cells() {
        let theme = Theme::dark();
        let line = context_bar_spans(4, 0, &[], theme);

        assert_eq!(line.width(), 4);
        assert!(line.spans.iter().all(|span| span.content.as_ref() == "█"));
        assert!(
            line.spans
                .iter()
                .all(|span| span.style.fg == Some(theme.elevated_bg))
        );
    }

    #[test]
    fn context_bar_falls_back_to_used_color_without_composition() {
        let theme = Theme::dark();
        let line = context_bar_spans(4, 2, &[], theme);

        assert_eq!(line.width(), 4);
        assert_eq!(line.spans[0].style.fg, Some(theme.accent));
        assert_eq!(line.spans[1].style.fg, Some(theme.accent));
        assert_eq!(line.spans[2].content.as_ref(), "█");
        assert_eq!(line.spans[2].style.fg, Some(theme.elevated_bg));
        assert_eq!(line.spans[3].content.as_ref(), "█");
        assert_eq!(line.spans[3].style.fg, Some(theme.elevated_bg));
    }

    #[test]
    fn context_bar_cell_uses_dominant_run_when_three_colors_share_a_cell() {
        let theme = Theme::dark();
        let cell = [
            Some(theme.notice),
            Some(theme.warning),
            Some(theme.success),
            Some(theme.success),
            Some(theme.success),
            Some(theme.user),
            Some(theme.user),
            Some(theme.user),
        ];
        let span = context_bar_cell(&cell, theme);

        assert_eq!(span.content.as_ref(), "█");
        assert_eq!(span.style.fg, Some(theme.user));
        assert_eq!(span.style.bg, Some(theme.element_bg));
    }

    #[test]
    fn context_bar_handles_more_categories_than_subcells() {
        let theme = Theme::dark();
        let colors = [
            theme.user,
            theme.accent,
            theme.approval,
            theme.assistant,
            theme.notice,
            theme.warning,
            theme.success,
            theme.error,
        ];
        let segments = (0..12)
            .map(|index| (1, colors[index % colors.len()]))
            .collect::<Vec<_>>();
        let line = context_bar_spans(1, 1, &segments, theme);

        assert_eq!(line.width(), 1);
        assert_eq!(line.spans.len(), 1);
    }

    #[test]
    fn tool_iteration_adds_prior_output_only_after_it_becomes_input() {
        let before = context_bar_line(
            40,
            &[crate::agent::PromptCompositionEntry {
                category: "messages".into(),
                estimated_tokens: 100,
                segments: 1,
            }],
            100,
            100,
            200,
            Theme::dark(),
        );
        let after = context_bar_line(
            40,
            &[crate::agent::PromptCompositionEntry {
                category: "messages".into(),
                estimated_tokens: 120,
                segments: 1,
            }],
            120,
            120,
            200,
            Theme::dark(),
        );
        let filled_cells = |line: &Line<'_>| {
            line.spans
                .iter()
                .filter(|span| span.style.fg != Some(Theme::dark().elevated_bg))
                .count()
        };

        assert_eq!(filled_cells(&before), 20);
        assert_eq!(filled_cells(&after), 24);
    }

    #[test]
    fn sidebar_labels_align_by_terminal_display_width() {
        assert_eq!(display_width(&padded_label("模型", 10)), 10);
        assert_eq!(display_width(&padded_label("Git", 10)), 10);
        assert_eq!(display_width(&padded_label("服务商", 10)), 10);
    }
}
