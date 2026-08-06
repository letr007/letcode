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

pub fn footer_scanner_cells(frame: usize, theme: Theme) -> Vec<(char, Color)> {
    scanner_cells(frame, theme)
}

pub fn render_footer(frame: &mut Frame<'_>, state: &TuiState, area: Rect, theme: Theme) {
    if area.is_empty() {
        return;
    }

    // Root background.
    frame.render_widget(Block::new().style(theme.app_style()), area);

    let left_spans = footer_status_spans(state, theme);

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

    if !state.current_context_branch.is_empty() {
        spans.push(Span::styled(
            state.current_context_branch.clone(),
            footer_value_style(theme),
        ));
    }

    if state.compaction_active {
        // 压缩中：指示条转为开火车式往返扫描，隐藏过期的 token 数字。
        if !spans.is_empty() {
            spans.push(Span::styled(" · ", footer_dim_style(theme)));
        }
        let animation_frame = state
            .status_spinner_frame
            .wrapping_sub(state.compaction_animation_start_frame);
        spans.extend(compaction_indicator_spans(animation_frame, theme));
    } else if let Some(usage) = &state.model_token_usage {
        if !spans.is_empty() {
            spans.push(Span::styled(" · ", footer_dim_style(theme)));
        }
        spans.extend(token_budget_spans(usage, theme));
    }

    if !state.slash_panel_is_open() {
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
    let usage_prefix = if usage.cache_report.is_none() && used_tokens > 0 {
        "~"
    } else {
        ""
    };
    spans.push(Span::styled(
        format!(" {usage_prefix}{used_percent}%"),
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

/// 压缩中指示条动画：开火车式往返扫描。
///
/// 只使用现有分块字形（█▉▊▋▌▍▎▏）与背景空单元，不引入新字符。
/// 四阶段循环：从左到右逐渐填满 → 从左到右逐渐清空 →
/// 从右到左逐渐填满 → 从右到左逐渐清空，类似火车往返开动。
fn compaction_indicator_spans(frame: usize, theme: Theme) -> Vec<Span<'static>> {
    const BAR_WIDTH: usize = 10;
    const UNITS_PER_CELL: usize = 8;
    const UNITS_PER_FRAME: usize = 2;
    const TOTAL_UNITS: usize = BAR_WIDTH * UNITS_PER_CELL;
    const PHASE_FRAMES: usize = TOTAL_UNITS / UNITS_PER_FRAME;

    // 阶段：0 = fill L→R，1 = empty L→R，2 = fill R→L，3 = empty R→L。
    let phase = (frame / PHASE_FRAMES) % 4;
    let progress = frame % PHASE_FRAMES;
    let swept_units = (progress + 1) * UNITS_PER_FRAME;
    let (active_start, active_end) = match phase {
        0 => (0, swept_units),
        1 => (swept_units, TOTAL_UNITS),
        2 => (TOTAL_UNITS - swept_units, TOTAL_UNITS),
        _ => (0, TOTAL_UNITS - swept_units),
    };
    let dimmed = phase.is_multiple_of(2);

    (0..BAR_WIDTH)
        .map(|index| {
            let (level, reverse) = compaction_cell(index, UNITS_PER_CELL, active_start, active_end);
            Span::styled(
                block_glyph(level).to_string(),
                compaction_cell_style(theme, level, reverse, dimmed),
            )
        })
        .collect()
}

/// 将整条 bar 的连续活动区间投影到单格。
///
/// 正向部分格使用左对齐分块字形；活动区间落在格子右侧时，通过反色绘制成
/// 右对齐亮块。边界每帧固定移动 2/8 格，左右方向保持完全对称。
fn compaction_cell(
    index: usize,
    units_per_cell: usize,
    active_start: usize,
    active_end: usize,
) -> (usize, bool) {
    let cell_start = index * units_per_cell;
    let cell_end = cell_start + units_per_cell;
    let overlap_start = active_start.max(cell_start);
    let overlap_end = active_end.min(cell_end);
    let active_units = overlap_end.saturating_sub(overlap_start);

    if active_units == 0 {
        (0, false)
    } else if active_units == units_per_cell {
        (units_per_cell, false)
    } else if overlap_start == cell_start {
        (active_units, false)
    } else {
        // 反色字形覆盖左侧非活动区，背景色露出右侧活动区。
        (units_per_cell - active_units, true)
    }
}

/// 0 = 空单元（背景），1..=8 对应 ▏▎▍▌▋▊▉█。
fn block_glyph(level: usize) -> char {
    match level {
        0 => ' ',
        1 => '▏',
        2 => '▎',
        3 => '▍',
        4 => '▌',
        5 => '▋',
        6 => '▊',
        7 => '▉',
        _ => '█',
    }
}

/// 压缩动画统一使用 accent 单色，与正常多段彩条形成区分。
/// 填充阶段使用明确计算出的暗色，避免终端 `DIM` 对反色背景造成闪烁。
fn compaction_cell_style(theme: Theme, level: usize, reverse: bool, dimmed: bool) -> Style {
    let accent = if dimmed {
        dimmed_compaction_color(theme)
    } else {
        theme.accent
    };
    if reverse {
        Style::default().fg(theme.root_bg).bg(accent)
    } else if level == 0 {
        Style::default().fg(theme.element_bg).bg(theme.root_bg)
    } else {
        Style::default().fg(accent).bg(theme.root_bg)
    }
}

fn dimmed_compaction_color(theme: Theme) -> Color {
    match (theme.accent, theme.root_bg) {
        (Color::Rgb(red, green, blue), Color::Rgb(bg_red, bg_green, bg_blue)) => Color::Rgb(
            ((red as u16 * 2 + bg_red as u16) / 3) as u8,
            ((green as u16 * 2 + bg_green as u16) / 3) as u8,
            ((blue as u16 * 2 + bg_blue as u16) / 3) as u8,
        ),
        _ => theme.dim_text,
    }
}

fn scanner_cells(frame: usize, theme: Theme) -> Vec<(char, Color)> {
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

    let head_color = phase_style(AppPhase::Idle, theme)
        .fg
        .unwrap_or(theme.user);
    let background = theme.root_bg;
    let gradient = [
        blend_toward_background(head_color, background, 0.00),
        blend_toward_background(head_color, background, 0.20),
        blend_toward_background(head_color, background, 0.40),
        blend_toward_background(head_color, background, 0.55),
        blend_toward_background(head_color, background, 0.70),
        blend_toward_background(head_color, background, 0.82),
    ];
    let trail_color = blend_toward_background(head_color, background, 0.88);

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
                gradient[distance.min(gradient.len() - 1)]
            } else {
                trail_color
            };
            (glyph, color)
        })
        .collect()
}

fn blend_toward_background(color: Color, background: Color, amount: f64) -> Color {
    match (color, background) {
        (Color::Rgb(red, green, blue), Color::Rgb(bg_red, bg_green, bg_blue)) => {
            let mix = |value: u8, bg_value: u8| {
                (value as f64 * (1.0 - amount) + bg_value as f64 * amount).round() as u8
            };
            Color::Rgb(mix(red, bg_red), mix(green, bg_green), mix(blue, bg_blue))
        }
        _ => color,
    }
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
        AppPhase::Error => footer_muted_style(theme),
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
        AppPhase::Error => vec![Span::styled("◆", phase_style(state.phase, theme))],
        AppPhase::Quitting => vec![Span::styled("◇", phase_style(state.phase, theme))],
    }
}

fn footer_status_spans(state: &TuiState, theme: Theme) -> Vec<Span<'static>> {
    phase_indicator_spans(state, theme)
}

#[cfg(test)]
mod tests {
    use super::{
        TokenBudgetSegment, compaction_indicator_spans, footer_hint_spans, footer_status_spans,
        render_footer, token_budget_cache_hit_percent, token_budget_cell,
        token_budget_segment_units,
    };
    use crate::{
        session::RetryLifecycleEvent,
        tui::{AppPhase, TuiState},
    };
    use ratatui::{Terminal, backend::TestBackend, layout::Rect};

    #[test]
    fn footer_omits_scheduled_retry_details() {
        let mut state = TuiState::default();
        state.phase = AppPhase::Running;
        state.retry = Some(crate::tui::RetryNoticeState::from_lifecycle(
            RetryLifecycleEvent {
                attempt: 2,
                max_attempts: 3,
                delay_secs: 1,
                error: "temporary upstream failure".into(),
            },
        ));

        let status_with_retry = footer_status_spans(&state, crate::tui::Theme::dark())
            .into_iter()
            .map(|span| span.content.into_owned())
            .collect::<String>();
        state.retry = None;
        let status_without_retry = footer_status_spans(&state, crate::tui::Theme::dark())
            .into_iter()
            .map(|span| span.content.into_owned())
            .collect::<String>();

        assert_eq!(status_with_retry, status_without_retry);
    }

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

    #[test]
    fn projected_token_usage_marks_percent_as_estimated() {
        let usage = crate::tui::state::ModelTokenUsage {
            used_tokens: 50_000,
            context_window_tokens: 100_000,
            input_tokens: 50_000,
            output_tokens: 0,
            cached_tokens: 0,
            cache_report: None,
        };

        let rendered = super::token_budget_spans(&usage, crate::tui::Theme::dark())
            .iter()
            .map(|span| span.content.as_ref())
            .collect::<String>();

        assert!(rendered.contains("~50%"), "{rendered}");
    }

    #[test]
    fn footer_omits_dynamic_status_content_but_keeps_intrinsic_chrome() {
        let mut state = TuiState::default();
        state.phase = AppPhase::WaitingForPermission;
        state.active_tool_call_id = Some("shell__exec-42".into());
        state.set_current_context_branch("parser-fix");

        let hint = footer_hint_spans(&state, crate::tui::Theme::dark());
        let status = footer_status_spans(&state, crate::tui::Theme::dark());
        let rendered_hint = hint
            .iter()
            .map(|span| span.content.as_ref())
            .collect::<String>();

        let rendered_status = status
            .iter()
            .map(|span| span.content.as_ref())
            .collect::<String>();

        assert!(rendered_hint.contains("/help"), "{rendered_hint}");
        assert!(rendered_hint.contains("parser-fix"), "{rendered_hint}");
        assert!(!rendered_status.contains("raw operation failure: connection refused"));
        assert!(!rendered_status.contains("secret diagnostic detail"));
        assert!(!rendered_status.contains("shell__exec-42"));
        assert!(!rendered_status.contains('✕'));

        let area = Rect::new(0, 0, 120, 1);
        let backend = TestBackend::new(area.width, area.height);
        let mut terminal = Terminal::new(backend).expect("terminal");
        terminal
            .draw(|frame| render_footer(frame, &state, area, crate::tui::Theme::dark()))
            .expect("draw");
        let rendered = terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|cell| cell.symbol())
            .collect::<String>();

        assert!(rendered.contains("parser-fix"));
        assert!(rendered.contains("/help"));
        assert!(!rendered.contains("raw operation failure: connection refused"));
        assert!(!rendered.contains("secret diagnostic detail"));
        assert!(!rendered.contains("shell__exec-42"));
    }

    #[test]
    fn footer_hint_shows_current_context_branch_name() {
        let mut state = TuiState::default();
        state.set_current_context_branch("parser-fix");

        let hint = footer_hint_spans(&state, crate::tui::Theme::dark());
        let rendered = hint
            .iter()
            .map(|span| span.content.as_ref())
            .collect::<String>();

        assert!(rendered.contains("parser-fix"), "{rendered}");
    }

    #[test]
    fn compaction_indicator_train_cycles_fill_and_empty_both_directions() {
        let theme = crate::tui::Theme::dark();
        let glyphs = |frame: usize| {
            compaction_indicator_spans(frame, theme)
                .into_iter()
                .map(|span| span.content.into_owned())
                .collect::<String>()
        };

        // 阶段 0：从左到右填满，每帧固定推进 2/8 格。
        assert_eq!(glyphs(0), "▎         ");
        assert_eq!(glyphs(1), "▌         ");
        assert_eq!(glyphs(2), "▊         ");
        assert_eq!(glyphs(3), "█         ");
        assert_eq!(glyphs(4), "█▎        ");
        assert_eq!(glyphs(39), "██████████");
        // 阶段 1：从左到右清空，当前格保留右侧部分（反色渲染）。
        assert_eq!(glyphs(40), "▎█████████");
        assert_eq!(glyphs(41), "▌█████████");
        assert_eq!(glyphs(42), "▊█████████");
        assert_eq!(glyphs(43), " █████████");
        assert_eq!(glyphs(79), "          ");
        // 阶段 2：从右到左填满，当前格从右侧长出（反色渲染）。
        assert_eq!(glyphs(80), "         ▊");
        assert_eq!(glyphs(81), "         ▌");
        assert_eq!(glyphs(82), "         ▎");
        assert_eq!(glyphs(83), "         █");
        assert_eq!(glyphs(119), "██████████");
        // 阶段 3：从右到左清空，当前格从右侧消失。
        assert_eq!(glyphs(120), "█████████▊");
        assert_eq!(glyphs(121), "█████████▌");
        assert_eq!(glyphs(122), "█████████▎");
        assert_eq!(glyphs(123), "█████████ ");
        assert_eq!(glyphs(159), "          ");
        // 周期循环回到阶段 0。
        assert_eq!(glyphs(160), "▎         ");
    }

    #[test]
    fn compaction_indicator_uses_stable_colors_for_partial_edges() {
        let theme = crate::tui::Theme::dark();
        let dimmed = super::dimmed_compaction_color(theme);

        // 阶段 0：填充使用固定压暗色，不使用终端 DIM 修饰符。
        let fill_spans = compaction_indicator_spans(0, theme);
        assert_eq!(fill_spans[0].style.fg, Some(dimmed));
        assert!(fill_spans[0].style.add_modifier.is_empty());
        // 阶段 1：左→右清空的部分格以正常 accent 背景呈现右对齐亮块。
        let empty_spans = compaction_indicator_spans(40, theme);
        assert_eq!(empty_spans[0].style.bg, Some(theme.accent));
        assert!(empty_spans[0].style.add_modifier.is_empty());
        // 阶段 2：右→左填充的部分格使用压暗 accent 背景。
        let reverse_fill_spans = compaction_indicator_spans(80, theme);
        assert_eq!(reverse_fill_spans[9].style.bg, Some(dimmed));
        assert!(reverse_fill_spans[9].style.add_modifier.is_empty());
    }

    #[test]
    fn footer_hint_swaps_token_bar_for_animation_while_compacting() {
        let mut state = TuiState::default();
        state.model_token_usage = Some(crate::tui::state::ModelTokenUsage {
            used_tokens: 50,
            context_window_tokens: 100,
            input_tokens: 50,
            output_tokens: 0,
            cached_tokens: 0,
            cache_report: None,
        });

        state.compaction_active = true;
        state.status_spinner_frame = 3;
        let hint = footer_hint_spans(&state, crate::tui::Theme::dark())
            .into_iter()
            .map(|span| span.content.into_owned())
            .collect::<String>();
        assert!(hint.contains('█'), "{hint}");
        assert!(!hint.contains('↑'), "{hint}");
        assert!(!hint.contains('%'), "{hint}");

        // 压缩结束：恢复真实数字指示条。
        state.compaction_active = false;
        let hint = footer_hint_spans(&state, crate::tui::Theme::dark())
            .into_iter()
            .map(|span| span.content.into_owned())
            .collect::<String>();
        assert!(hint.contains("↑50"), "{hint}");
        assert!(hint.contains('%'), "{hint}");
    }
}

fn scanner_frame_spans(frame: usize, theme: Theme) -> Vec<Span<'static>> {
    footer_scanner_cells(frame, theme)
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
