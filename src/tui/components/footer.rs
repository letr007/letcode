use ratatui::{
    Frame,
    layout::{Alignment, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Paragraph},
};

use crate::tui::{
    state::{AppPhase, TuiState},
    surface,
    theme::Theme,
};

pub fn footer_scanner_cells(frame: usize) -> Vec<(char, Color)> {
    scanner_cells(frame)
}

pub fn render_footer(frame: &mut Frame<'_>, state: &TuiState, area: Rect, theme: Theme) {
    if area.is_empty() {
        return;
    }

    // Root background.
    frame.render_widget(Block::new().style(theme.app_style()), area);

    let mut left_spans = footer_status_spans(state, theme);

    if !matches!(state.phase, AppPhase::WaitingForPermission)
        && let Some(active_tool_call_id) = &state.active_tool_call_id
    {
        left_spans.push(Span::styled(" · active ", footer_dim_style(theme)));
        left_spans.push(Span::styled(
            active_tool_call_id.clone(),
            footer_value_style(theme),
        ));
    }

    // Keep this compact: right side acts like a stable status hint bar.
    let right_line = Line::from(footer_hint_spans(state, theme));

    let right_width = right_line.width() as u16;
    let left_line = Line::from(left_spans);
    let left_width = left_line.width() as u16;

    // Render the right side first so it doesn't overlap the left.
    frame.render_widget(
        Paragraph::new(right_line)
            .style(theme.app_style())
            .alignment(Alignment::Right),
        area,
    );

    // Leave 1 col padding between sides when possible.
    let left_available_width = area
        .width
        .saturating_sub(1)
        .saturating_sub(right_width.saturating_add(1));

    if left_width > 0 && left_available_width > 0 {
        frame.render_widget(
            Paragraph::new(left_line).style(theme.app_style()),
            Rect::new(
                area.x.saturating_add(1),
                area.y,
                left_width.min(left_available_width),
                1,
            ),
        );
    }
}

fn footer_hint_spans(state: &TuiState, theme: Theme) -> Vec<Span<'static>> {
    let mut spans = Vec::new();

    if let Some(usage) = &state.model_token_usage {
        spans.extend(token_budget_spans(usage, theme));
    }

    if !matches!(state.phase, AppPhase::WaitingForPermission) && !state.slash_panel_is_open() {
        if !spans.is_empty() {
            spans.push(Span::styled(" · ", footer_dim_style(theme)));
        }
        spans.push(Span::styled("/help", footer_dim_style(theme)));
        spans.push(Span::styled(" commands", footer_muted_style(theme)));
    }

    spans
}

fn token_budget_spans(
    usage: &crate::tui::state::ModelTokenUsage,
    theme: Theme,
) -> Vec<Span<'static>> {
    const BAR_WIDTH: usize = 10;

    let cached_input_tokens = usage.cached_tokens.min(usage.input_tokens);
    let uncached_input_tokens = usage.input_tokens.saturating_sub(cached_input_tokens);
    let accounted_tokens = usage.input_tokens.saturating_add(usage.output_tokens);
    let used_tokens = usage.used_tokens.max(accounted_tokens);
    let unknown_used_tokens = used_tokens.saturating_sub(accounted_tokens);
    let cache_hit_percent = token_budget_cache_hit_percent(usage.input_tokens, cached_input_tokens);
    let used_percent = token_budget_used_percent(usage.context_window_tokens, used_tokens);

    let [cache_units, input_units, output_units] = token_budget_segment_units(
        usage.context_window_tokens,
        BAR_WIDTH,
        [
            cached_input_tokens,
            uncached_input_tokens.saturating_add(unknown_used_tokens),
            usage.output_tokens,
        ],
    );

    let mut spans = Vec::with_capacity(8);
    spans.extend(token_budget_bar_spans(
        BAR_WIDTH,
        [
            (cache_units, TokenBudgetSegment::Cache),
            (input_units, TokenBudgetSegment::Input),
            (output_units, TokenBudgetSegment::Output),
        ],
        theme,
    ));

    spans.push(Span::styled(" ", footer_dim_style(theme)));
    spans.push(Span::styled(
        format!("↑{}", format_compact_count(usage.input_tokens)),
        token_budget_input_text_style(theme),
    ));
    spans.push(Span::styled(" ", footer_dim_style(theme)));
    spans.push(Span::styled(
        format!("↓{}", format_compact_count(usage.output_tokens)),
        token_budget_output_text_style(theme),
    ));
    if let Some(cache_hit_percent) = cache_hit_percent {
        spans.push(Span::styled(" ", footer_dim_style(theme)));
        spans.push(Span::styled(
            format!("{cache_hit_percent}%"),
            token_budget_cache_text_style(theme),
        ));
    }
    spans.push(Span::styled(
        format!(" {used_percent}%"),
        footer_muted_style(theme),
    ));
    spans
}

fn token_budget_cache_hit_percent(input_tokens: u64, cached_input_tokens: u64) -> Option<u64> {
    if input_tokens == 0 || cached_input_tokens == 0 {
        return None;
    }

    Some(
        (((cached_input_tokens.min(input_tokens) as f64 / input_tokens as f64) * 100.0).round()
            as u64)
            .min(100),
    )
}

fn token_budget_used_percent(context_window_tokens: u64, used_tokens: u64) -> u64 {
    if context_window_tokens == 0 {
        return 0;
    }

    (((used_tokens.min(context_window_tokens) as f64 / context_window_tokens as f64) * 100.0)
        .round() as u64)
        .min(100)
}

fn token_budget_segment_units(
    context_window_tokens: u64,
    width: usize,
    tokens: [u64; 3],
) -> [usize; 3] {
    let total_units = width.saturating_mul(8);
    if context_window_tokens == 0 || total_units == 0 {
        return [0, 0, 0];
    }

    let mut units = [0; 3];
    let total_token_count = tokens
        .into_iter()
        .fold(0u64, |total, token_count| total.saturating_add(token_count))
        .max(1);
    let used_tokens = total_token_count.min(context_window_tokens);
    let segment_denominator = context_window_tokens.max(total_token_count) as f64;
    let target_units = (((used_tokens as f64 / context_window_tokens as f64) * total_units as f64)
        .round() as usize)
        .min(total_units);
    let target_units = if used_tokens > 0 && target_units == 0 {
        1
    } else {
        target_units
    };

    let mut remainders = [(0usize, 0.0f64); 3];
    for (index, token_count) in tokens.into_iter().enumerate() {
        let exact_units = (token_count as f64 / segment_denominator) * total_units as f64;
        let floor_units = exact_units.floor() as usize;
        units[index] = floor_units.min(target_units);
        remainders[index] = (index, exact_units.fract());
    }

    let mut allocated_units = units.iter().sum::<usize>();
    remainders.sort_by(
        |(left_index, left_remainder), (right_index, right_remainder)| {
            right_remainder
                .total_cmp(left_remainder)
                .then_with(|| segment_priority(*right_index).cmp(&segment_priority(*left_index)))
        },
    );
    for (index, _) in remainders {
        if allocated_units >= target_units {
            break;
        }
        if tokens[index] == 0 {
            continue;
        }
        units[index] = units[index].saturating_add(1);
        allocated_units = allocated_units.saturating_add(1);
    }

    if units.iter().sum::<usize>() == 0
        && let Some((index, _)) = tokens
            .iter()
            .enumerate()
            .filter(|(_, token_count)| **token_count > 0)
            .max_by_key(|(_, token_count)| **token_count)
    {
        units[index] = 1;
    }

    units
}

fn segment_priority(index: usize) -> usize {
    match index {
        2 => 3,
        1 => 2,
        _ => 1,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TokenBudgetSegment {
    Cache,
    Input,
    Output,
    Empty,
}

fn token_budget_bar_spans(
    width: usize,
    segments: [(usize, TokenBudgetSegment); 3],
    theme: Theme,
) -> Vec<Span<'static>> {
    let total_units = width.saturating_mul(8);
    let mut cells = Vec::with_capacity(total_units);
    for (units, segment) in segments {
        cells.extend(std::iter::repeat_n(segment, units));
    }
    cells.truncate(total_units);
    cells.resize(total_units, TokenBudgetSegment::Empty);

    let mut spans = Vec::with_capacity(width);
    for cell in cells.chunks(8) {
        let (glyph, foreground, background) = token_budget_cell(cell);
        spans.push(Span::styled(
            glyph.to_string(),
            token_budget_bar_cell_style(foreground, background, theme),
        ));
    }

    spans
}

fn token_budget_cell(
    cell: &[TokenBudgetSegment],
) -> (char, TokenBudgetSegment, TokenBudgetSegment) {
    debug_assert_eq!(cell.len(), 8);

    if cell.iter().all(|segment| *segment == cell[0]) {
        return ('█', cell[0], TokenBudgetSegment::Empty);
    }

    if cell.contains(&TokenBudgetSegment::Empty) {
        let used_units = cell
            .iter()
            .position(|segment| *segment == TokenBudgetSegment::Empty)
            .unwrap_or(cell.len());
        if used_units == 0 {
            return ('█', TokenBudgetSegment::Empty, TokenBudgetSegment::Empty);
        }

        let foreground = dominant_used_segment(&cell[..used_units]);
        return (
            partial_block(used_units),
            foreground,
            TokenBudgetSegment::Empty,
        );
    }

    if cell.contains(&TokenBudgetSegment::Output) {
        let input_units = cell
            .iter()
            .take_while(|segment| **segment != TokenBudgetSegment::Output)
            .count();
        return (
            partial_block(input_units),
            TokenBudgetSegment::Input,
            TokenBudgetSegment::Output,
        );
    }

    let cache_units = cell
        .iter()
        .take_while(|segment| **segment == TokenBudgetSegment::Cache)
        .count();
    (
        partial_block(cache_units),
        TokenBudgetSegment::Cache,
        TokenBudgetSegment::Input,
    )
}

fn dominant_used_segment(cell: &[TokenBudgetSegment]) -> TokenBudgetSegment {
    [
        TokenBudgetSegment::Output,
        TokenBudgetSegment::Input,
        TokenBudgetSegment::Cache,
    ]
    .into_iter()
    .max_by_key(|segment| {
        let count = cell
            .iter()
            .filter(|candidate| **candidate == *segment)
            .count();
        let priority = match segment {
            TokenBudgetSegment::Output => 3,
            TokenBudgetSegment::Input => 2,
            TokenBudgetSegment::Cache => 1,
            TokenBudgetSegment::Empty => 0,
        };
        (count, priority)
    })
    .unwrap_or(TokenBudgetSegment::Input)
}

fn partial_block(units: usize) -> char {
    match units.clamp(1, 7) {
        1 => '▏',
        2 => '▎',
        3 => '▍',
        4 => '▌',
        5 => '▋',
        6 => '▊',
        _ => '▉',
    }
}

fn token_budget_bar_cell_style(
    foreground: TokenBudgetSegment,
    background: TokenBudgetSegment,
    theme: Theme,
) -> Style {
    Style::default()
        .fg(token_budget_segment_color(foreground, theme))
        .bg(token_budget_segment_color(background, theme))
}

fn token_budget_segment_color(segment: TokenBudgetSegment, theme: Theme) -> Color {
    match segment {
        TokenBudgetSegment::Cache => theme.approval,
        TokenBudgetSegment::Input => theme.accent,
        TokenBudgetSegment::Output => theme.assistant,
        TokenBudgetSegment::Empty => theme.element_bg,
    }
}

fn format_compact_count(value: u64) -> String {
    const K: f64 = 1_000.0;
    const M: f64 = 1_000_000.0;

    if value >= 1_000_000 {
        format!("{:.1}m", value as f64 / M)
    } else if value >= 1_000 {
        format!("{:.1}k", value as f64 / K)
    } else {
        value.to_string()
    }
}

fn scanner_cells(frame: usize) -> Vec<(char, Color)> {
    const WIDTH: usize = 8;
    const HOLD_END: usize = 9;
    const HOLD_START: usize = 30;
    const TRAIL: usize = 6;

    let cycle = WIDTH + HOLD_END + (WIDTH - 1) + HOLD_START;
    let position = frame % cycle;
    let forward_end = WIDTH;
    let hold_end = forward_end + HOLD_END;
    let reverse_end = hold_end + WIDTH - 1;

    let (head, forward, hold_progress) = if position < forward_end {
        (position, true, 0usize)
    } else if position < hold_end {
        (WIDTH - 1, true, position - forward_end)
    } else if position < reverse_end {
        (WIDTH - 2 - (position - hold_end), false, 0usize)
    } else {
        (0, false, position - reverse_end)
    };

    (0..WIDTH)
        .map(|index| {
            let distance = if forward {
                if index <= head { head - index } else { TRAIL }
            } else if index >= head {
                index - head
            } else {
                TRAIL
            };
            let distance = distance.saturating_add(hold_progress);
            let active = distance < TRAIL;
            let glyph = if active { '■' } else { '⬝' };
            let color = if active {
                match distance {
                    0 => Color::Rgb(80, 180, 220),
                    1 => Color::Rgb(85, 188, 230),
                    2 => Color::Rgb(58, 123, 149),
                    3 => Color::Rgb(44, 86, 103),
                    4 => Color::Rgb(35, 63, 73),
                    _ => Color::Rgb(29, 47, 54),
                }
            } else {
                Color::Rgb(30, 50, 58)
            };
            (glyph, color)
        })
        .collect()
}

fn footer_value_style(theme: Theme) -> Style {
    Style::default().fg(theme.text).bg(theme.root_bg)
}

fn footer_muted_style(theme: Theme) -> Style {
    Style::default().fg(theme.muted_text).bg(theme.root_bg)
}

fn footer_dim_style(theme: Theme) -> Style {
    Style::default()
        .fg(theme.dim_text)
        .bg(theme.root_bg)
        .add_modifier(Modifier::DIM)
}

fn token_budget_input_text_style(theme: Theme) -> Style {
    Style::default().fg(theme.accent).bg(theme.root_bg)
}

fn token_budget_cache_text_style(theme: Theme) -> Style {
    Style::default().fg(theme.approval).bg(theme.root_bg)
}

fn token_budget_output_text_style(theme: Theme) -> Style {
    Style::default().fg(theme.assistant).bg(theme.root_bg)
}

fn phase_style(phase: AppPhase, theme: Theme) -> Style {
    match phase {
        AppPhase::Idle | AppPhase::Editing | AppPhase::Completed => {
            Style::default().fg(theme.user).bg(theme.root_bg)
        }
        AppPhase::Running => Style::default().fg(theme.assistant).bg(theme.root_bg),
        AppPhase::WaitingForPermission => Style::default().fg(theme.approval).bg(theme.root_bg),
        AppPhase::Error => Style::default().fg(theme.error).bg(theme.root_bg),
        AppPhase::Quitting => footer_muted_style(theme),
    }
}

fn phase_indicator_spans(state: &TuiState, theme: Theme) -> Vec<Span<'static>> {
    match state.phase {
        AppPhase::Running => scanner_frame_spans(state.status_spinner_frame, theme),
        AppPhase::Idle | AppPhase::Editing | AppPhase::Completed => {
            vec![Span::styled("◆", phase_style(state.phase, theme))]
        }
        AppPhase::WaitingForPermission => vec![Span::styled("▲", phase_style(state.phase, theme))],
        AppPhase::Error => vec![Span::styled("✕", phase_style(state.phase, theme))],
        AppPhase::Quitting => vec![Span::styled("◇", phase_style(state.phase, theme))],
    }
}

fn footer_status_spans(state: &TuiState, theme: Theme) -> Vec<Span<'static>> {
    if matches!(state.phase, AppPhase::Running) {
        return running_status_spans(state, theme);
    }

    if matches!(state.phase, AppPhase::WaitingForPermission) {
        return Vec::new();
    }

    if should_silence_footer_status(state) {
        return phase_indicator_spans(state, theme);
    }

    let mut spans = phase_indicator_spans(state, theme);
    if !spans.is_empty() {
        spans.push(Span::styled(" ", footer_value_style(theme)));
    }
    spans.push(Span::styled(
        state.footer_status.summary.clone(),
        phase_style(state.phase, theme),
    ));

    if let Some(detail) = &state.footer_status.detail {
        spans.push(Span::styled(" · ", footer_dim_style(theme)));
        spans.push(Span::styled(detail.clone(), footer_muted_style(theme)));
    }

    spans
}

#[cfg(test)]
mod tests {
    use super::{
        TokenBudgetSegment, token_budget_cache_hit_percent, token_budget_cell,
        token_budget_segment_units,
    };

    #[test]
    fn token_budget_units_keep_cache_input_and_output_segments() {
        assert_eq!(
            token_budget_segment_units(100, 10, [20, 30, 10]),
            [16, 24, 8]
        );
    }

    #[test]
    fn token_budget_units_keep_low_usage_visible() {
        assert_eq!(
            token_budget_segment_units(400_000, 10, [0, 15_900, 1_400]),
            [0, 3, 0]
        );
    }

    #[test]
    fn token_budget_units_make_tiny_used_total_visible_once() {
        assert_eq!(
            token_budget_segment_units(1_000_000, 10, [0, 10, 0]),
            [0, 1, 0]
        );
    }

    #[test]
    fn token_budget_cell_uses_adjacent_color_as_partial_background() {
        assert_eq!(
            token_budget_cell(&[
                TokenBudgetSegment::Cache,
                TokenBudgetSegment::Cache,
                TokenBudgetSegment::Input,
                TokenBudgetSegment::Input,
                TokenBudgetSegment::Input,
                TokenBudgetSegment::Input,
                TokenBudgetSegment::Input,
                TokenBudgetSegment::Input,
            ]),
            ('▎', TokenBudgetSegment::Cache, TokenBudgetSegment::Input)
        );
    }

    #[test]
    fn token_budget_cell_collapses_three_color_boundary_to_used_vs_unused() {
        assert_eq!(
            token_budget_cell(&[
                TokenBudgetSegment::Input,
                TokenBudgetSegment::Input,
                TokenBudgetSegment::Input,
                TokenBudgetSegment::Output,
                TokenBudgetSegment::Empty,
                TokenBudgetSegment::Empty,
                TokenBudgetSegment::Empty,
                TokenBudgetSegment::Empty,
            ]),
            ('▌', TokenBudgetSegment::Input, TokenBudgetSegment::Empty)
        );
    }

    #[test]
    fn token_budget_cache_hit_percent_is_hidden_without_cached_tokens() {
        assert_eq!(token_budget_cache_hit_percent(40_000, 0), None);
        assert_eq!(token_budget_cache_hit_percent(0, 20_000), None);
    }

    #[test]
    fn token_budget_cache_hit_percent_uses_cached_over_input_tokens() {
        assert_eq!(token_budget_cache_hit_percent(40_000, 20_000), Some(50));
        assert_eq!(token_budget_cache_hit_percent(40_000, 80_000), Some(100));
    }
}

fn running_status_spans(state: &TuiState, theme: Theme) -> Vec<Span<'static>> {
    let mut spans = scanner_frame_spans(state.status_spinner_frame, theme);
    spans.push(Span::styled(" ", activity_text_style(theme)));
    spans.push(Span::styled(
        state.footer_status.summary.clone(),
        activity_text_style(theme),
    ));

    if let Some(detail) = &state.footer_status.detail {
        spans.push(Span::styled(" · ", footer_dim_style(theme)));
        spans.push(Span::styled(detail.clone(), footer_muted_style(theme)));
    }

    spans
}

fn activity_text_style(theme: Theme) -> Style {
    Style::default().fg(theme.user).bg(theme.root_bg)
}

fn should_silence_footer_status(state: &TuiState) -> bool {
    matches!(
        state.phase,
        AppPhase::Idle | AppPhase::Editing | AppPhase::Completed
    ) && state.footer_status.summary == "Ready"
}

fn scanner_frame_spans(frame: usize, theme: Theme) -> Vec<Span<'static>> {
    footer_scanner_cells(frame)
        .into_iter()
        .map(|(glyph, color)| {
            Span::styled(
                glyph.to_string(),
                Style::default()
                    .fg(color)
                    .bg(surface::surface_bg(theme, surface::SurfaceKind::Root)),
            )
        })
        .collect()
}
