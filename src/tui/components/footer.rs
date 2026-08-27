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

    let left_line = Line::from(footer_status_spans(state, theme));
    let left_width = left_line.width().min(area.width as usize) as u16;
    let left_padding = u16::from(area.width > left_width);
    let gap_width =
        u16::from(left_width > 0 && area.width > left_padding.saturating_add(left_width));
    let right_max_width = area
        .width
        .saturating_sub(left_padding)
        .saturating_sub(left_width)
        .saturating_sub(gap_width) as usize;
    let right_line = Line::from(footer_hint_spans(state, theme, right_max_width));
    let right_width = right_line.width() as u16;

    if right_width > 0 {
        frame.render_widget(
            Paragraph::new(right_line)
                .style(theme.app_style())
                .alignment(Alignment::Right),
            Rect::new(
                area.right().saturating_sub(right_width),
                area.y,
                right_width,
                1,
            ),
        );
    }

    if left_width > 0 {
        frame.render_widget(
            Paragraph::new(left_line).style(theme.app_style()),
            Rect::new(area.x.saturating_add(left_padding), area.y, left_width, 1),
        );
    }
}

fn footer_hint_spans(state: &TuiState, theme: Theme, max_width: usize) -> Vec<Span<'static>> {
    if max_width == 0 {
        return Vec::new();
    }

    let status = if let Some(start_frame) = state.active_compaction_animation_start_frame() {
        // 压缩中：指示条转为开火车式往返扫描，隐藏过期的 token 数字。
        let animation_frame = state.status_spinner_frame.wrapping_sub(start_frame);
        compaction_indicator_spans(animation_frame, theme)
    } else {
        let mut spans = state
            .active_model_token_usage()
            .map(|usage| token_budget_spans(usage, theme))
            .unwrap_or_default();
        if let Some(rate) = state.active_output_token_rate() {
            if !spans.is_empty() {
                spans.push(Span::styled(" · ", footer_dim_style(theme)));
            }
            spans.push(Span::styled(
                format!("{rate}t/s"),
                output_token_rate_style(rate, theme),
            ));
        }
        spans
    };
    let status_width = spans_width(&status);
    if status_width > max_width {
        return truncate_spans_display_width(status, max_width);
    }

    let metadata_max_width =
        max_width.saturating_sub(status_width.saturating_add(if status_width > 0 {
            3
        } else {
            Default::default()
        }));
    let metadata = footer_metadata_spans(state, theme, metadata_max_width);

    let mut spans = metadata;
    if !status.is_empty() {
        if !spans.is_empty() {
            spans.push(Span::styled(" · ", footer_dim_style(theme)));
        }
        spans.extend(status);
    }

    if !state.slash_panel_is_open() {
        append_help_hint(&mut spans, theme, max_width);
    }
    spans
}

fn footer_metadata_spans(state: &TuiState, theme: Theme, max_width: usize) -> Vec<Span<'static>> {
    if max_width == 0 {
        return Vec::new();
    }

    let git = state
        .git_branch
        .as_deref()
        .map(|branch| labeled_footer_value_spans(" ", branch, theme, usize::MAX));
    let context = (!state.current_context_branch.is_empty()).then(|| {
        labeled_footer_value_spans("󰙅 ", &state.current_context_branch, theme, usize::MAX)
    });

    match (git, context) {
        (Some(git), Some(context)) => {
            let git_width = spans_width(&git);
            let context_width = spans_width(&context);
            if git_width.saturating_add(3).saturating_add(context_width) <= max_width {
                let mut spans = git;
                spans.push(Span::styled(" · ", footer_dim_style(theme)));
                spans.extend(context);
                return spans;
            }
            if context_width >= max_width {
                return truncate_spans_display_width(context, max_width);
            }

            let git_max_width = max_width.saturating_sub(context_width).saturating_sub(3);
            let git = truncate_spans_display_width(git, git_max_width);
            if git.is_empty() {
                return context;
            }
            let mut spans = git;
            spans.push(Span::styled(" · ", footer_dim_style(theme)));
            spans.extend(context);
            spans
        }
        (Some(git), None) => truncate_spans_display_width(git, max_width),
        (None, Some(context)) => truncate_spans_display_width(context, max_width),
        (None, None) => Vec::new(),
    }
}

fn labeled_footer_value_spans(
    label: &'static str,
    value: &str,
    theme: Theme,
    max_width: usize,
) -> Vec<Span<'static>> {
    truncate_spans_display_width(
        vec![
            Span::styled(label, footer_dim_style(theme)),
            Span::styled(value.to_string(), footer_value_style(theme)),
        ],
        max_width,
    )
}

fn append_help_hint(spans: &mut Vec<Span<'static>>, theme: Theme, max_width: usize) {
    let current_width = spans_width(spans);
    let separator_width = usize::from(!spans.is_empty()) * 3;
    let remaining = max_width
        .saturating_sub(current_width)
        .saturating_sub(separator_width);
    let full_help = vec![
        Span::styled("/help", footer_dim_style(theme)),
        Span::styled(" commands", footer_muted_style(theme)),
    ];
    let compact_help = vec![Span::styled("/help", footer_dim_style(theme))];
    let help = if spans_width(&full_help) <= remaining {
        full_help
    } else if spans_width(&compact_help) <= remaining {
        compact_help
    } else {
        return;
    };

    if !spans.is_empty() {
        spans.push(Span::styled(" · ", footer_dim_style(theme)));
    }
    spans.extend(help);
}

fn spans_width(spans: &[Span<'static>]) -> usize {
    Line::from(spans.to_vec()).width()
}

fn truncate_spans_display_width(spans: Vec<Span<'static>>, max_width: usize) -> Vec<Span<'static>> {
    let mut remaining = max_width;
    let mut truncated = Vec::new();

    for span in spans {
        if remaining == 0 {
            break;
        }
        let width = crate::tui::measure::display_width(span.content.as_ref());
        if width <= remaining {
            remaining = remaining.saturating_sub(width);
            truncated.push(span);
            continue;
        }

        let content = crate::tui::components::tool_card::truncate_display_width(
            span.content.as_ref(),
            remaining,
        );
        if !content.is_empty() {
            truncated.push(Span::styled(content, span.style));
        }
        break;
    }
    truncated
}

fn token_budget_spans(
    usage: &crate::tui::state::ModelTokenUsage,
    theme: Theme,
) -> Vec<Span<'static>> {
    const BAR_WIDTH: usize = 10;

    let cached_input_tokens = usage.cached_tokens.min(usage.input_tokens);
    let uncached_input_tokens = usage.input_tokens.saturating_sub(cached_input_tokens);
    let used_tokens = usage.used_tokens;
    let current_output_tokens = used_tokens.saturating_sub(usage.input_tokens);
    let cache_hit_percent = token_budget_cache_hit_percent(usage.input_tokens, cached_input_tokens);
    let used_percent = token_budget_used_percent(usage.context_window_tokens, used_tokens);

    let [cache_units, input_units, output_units] = token_budget_segment_units(
        usage.context_window_tokens,
        BAR_WIDTH,
        [
            cached_input_tokens,
            uncached_input_tokens,
            current_output_tokens,
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
            format!("{cache_hit_percent:.2}%"),
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

fn output_token_rate_style(rate: u64, theme: Theme) -> Style {
    let color = match rate {
        0..=19 => theme.error,
        20..=39 => theme.approval,
        40..=79 => theme.warning,
        _ => theme.success,
    };
    Style::default().fg(color).bg(theme.root_bg)
}

fn token_budget_cache_hit_percent(input_tokens: u64, cached_input_tokens: u64) -> Option<f64> {
    if input_tokens == 0 || cached_input_tokens == 0 {
        return None;
    }

    Some((cached_input_tokens.min(input_tokens) as f64 / input_tokens as f64) * 100.0)
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

    let head_color = phase_style(AppPhase::Idle, theme).fg.unwrap_or(theme.user);
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
    let phase = state.active_phase();
    match phase {
        AppPhase::Running => scanner_frame_spans(state.status_spinner_frame, theme),
        AppPhase::Idle | AppPhase::Editing | AppPhase::Completed => {
            vec![Span::styled("◆", phase_style(phase, theme))]
        }
        AppPhase::WaitingForPermission => vec![Span::styled("▲", phase_style(phase, theme))],
        AppPhase::Error => vec![Span::styled("◆", phase_style(phase, theme))],
        AppPhase::Quitting => vec![Span::styled("◇", phase_style(phase, theme))],
    }
}

fn footer_status_spans(state: &TuiState, theme: Theme) -> Vec<Span<'static>> {
    phase_indicator_spans(state, theme)
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

#[cfg(test)]
mod tests {
    use super::{
        TokenBudgetSegment, compaction_indicator_spans, footer_hint_spans, footer_status_spans,
        output_token_rate_style, render_footer, token_budget_cache_hit_percent, token_budget_cell,
        token_budget_segment_units, token_budget_spans,
    };
    use crate::{
        session::RetryLifecycleEvent,
        tui::{AppPhase, TuiState},
    };
    use ratatui::{Terminal, backend::TestBackend, layout::Rect};

    #[test]
    fn footer_places_git_branch_before_context_branch() {
        let mut state = TuiState::default();
        state.set_git_branch(Some("feature/branch".into()));
        state.set_current_context_branch("context-2");

        let rendered = footer_hint_spans(&state, crate::tui::Theme::dark(), 80)
            .iter()
            .map(|span| span.content.as_ref())
            .collect::<String>();

        assert!(
            rendered.starts_with(" feature/branch · 󰙅 context-2"),
            "{rendered}"
        );
    }

    #[test]
    fn footer_preserves_context_when_long_git_branch_is_truncated() {
        let mut state = TuiState::default();
        state.set_git_branch(Some("feature/超长分支名称-for-footer-budget".into()));
        state.set_current_context_branch("root");

        let spans = footer_hint_spans(&state, crate::tui::Theme::dark(), 18);
        let line = ratatui::text::Line::from(spans);
        let rendered = line.to_string();

        assert!(line.width() <= 18, "{rendered}");
        assert!(rendered.contains("󰙅 root"), "{rendered}");
        assert!(rendered.contains('…'), "{rendered}");
    }

    #[test]
    fn footer_prioritizes_token_status_over_metadata_and_help() {
        let mut state = TuiState::default();
        state.set_git_branch(Some("feature/long-branch".into()));
        state.set_current_context_branch("root");
        state.model_token_usage = Some(crate::tui::state::ModelTokenUsage {
            input_tokens: 600,
            output_tokens: 100,
            cached_tokens: 200,
            used_tokens: 700,
            context_window_tokens: 1_000,
            cache_report: None,
            prompt_composition: Vec::new(),
        });

        let status = token_budget_spans(
            state.model_token_usage.as_ref().expect("token usage"),
            crate::tui::Theme::dark(),
        );
        let status_width = ratatui::text::Line::from(status.clone()).width();
        let rendered = footer_hint_spans(&state, crate::tui::Theme::dark(), status_width)
            .iter()
            .map(|span| span.content.as_ref())
            .collect::<String>();
        let expected = status
            .iter()
            .map(|span| span.content.as_ref())
            .collect::<String>();

        assert_eq!(rendered, expected);
        assert!(!rendered.contains(" "), "{rendered}");
        assert!(!rendered.contains("󰙅 "), "{rendered}");
        assert!(!rendered.contains("/help"), "{rendered}");
    }

    #[test]
    fn footer_appends_output_token_rate_after_context_status() {
        let mut state = TuiState::default();
        state.model_token_usage = Some(crate::tui::state::ModelTokenUsage {
            input_tokens: 600,
            output_tokens: 100,
            cached_tokens: 0,
            used_tokens: 700,
            context_window_tokens: 1_000,
            cache_report: None,
            prompt_composition: Vec::new(),
        });
        state.set_output_token_rate(Some(60));

        let rendered = footer_hint_spans(&state, crate::tui::Theme::dark(), 80)
            .iter()
            .map(|span| span.content.as_ref())
            .collect::<String>();

        assert!(rendered.contains("~70% · 60t/s"), "{rendered}");
    }

    #[test]
    fn footer_shows_output_token_rate_before_provider_usage_arrives() {
        let mut state = TuiState::default();
        state.set_output_token_rate(Some(60));

        let rendered = footer_hint_spans(&state, crate::tui::Theme::dark(), 80)
            .iter()
            .map(|span| span.content.as_ref())
            .collect::<String>();

        assert!(rendered.contains(" · 60t/s"), "{rendered}");
    }

    #[test]
    fn output_token_rate_color_scales_from_slow_to_fast() {
        let theme = crate::tui::Theme::dark();

        assert_eq!(output_token_rate_style(19, theme).fg, Some(theme.error));
        assert_eq!(output_token_rate_style(20, theme).fg, Some(theme.approval));
        assert_eq!(output_token_rate_style(39, theme).fg, Some(theme.approval));
        assert_eq!(output_token_rate_style(40, theme).fg, Some(theme.warning));
        assert_eq!(output_token_rate_style(79, theme).fg, Some(theme.warning));
        assert_eq!(output_token_rate_style(80, theme).fg, Some(theme.success));
    }

    #[test]
    fn footer_uses_token_usage_from_active_child_view() {
        let mut state = TuiState::default();
        state.set_token_usage(crate::tui::state::ModelTokenUsage {
            used_tokens: 100,
            context_window_tokens: 1_000,
            input_tokens: 100,
            output_tokens: 0,
            cached_tokens: 0,
            cache_report: None,
            prompt_composition: Vec::new(),
        });
        state.replace_child_timeline_from_records(
            &[],
            "parent-session",
            "child-session",
            "explorer",
            0,
            1,
            1,
        );
        state.apply_child_session_event(
            "child-session",
            crate::tui::SessionEvent::TokenUsage(crate::session::TokenUsageEvent::with_breakdown(
                700, 1_000, 600, 100, 0,
            )),
        );

        let rendered = footer_hint_spans(&state, crate::tui::Theme::dark(), 80)
            .iter()
            .map(|span| span.content.as_ref())
            .collect::<String>();

        assert!(rendered.contains("↑600"), "{rendered}");
        assert!(rendered.contains("↓100"), "{rendered}");
        assert!(rendered.contains("~70%"), "{rendered}");
        assert!(!rendered.contains("~10%"), "{rendered}");
    }

    #[test]
    fn footer_truncates_long_metadata_without_exceeding_budget() {
        let mut state = TuiState::default();
        state.set_git_branch(Some(
            "feature/a-very-long-branch-name-that-must-be-clipped".into(),
        ));
        state.set_current_context_branch("context-with-a-long-name");

        let spans = footer_hint_spans(&state, crate::tui::Theme::dark(), 24);
        let line = ratatui::text::Line::from(spans);
        let rendered = line.to_string();

        assert!(line.width() <= 24, "{rendered}");
        assert!(rendered.contains('…'), "{rendered}");
    }

    #[test]
    fn footer_phase_indicator_tracks_the_visible_child_session() {
        let mut state = TuiState::default();
        state.phase = AppPhase::Completed;
        state.replace_child_timeline_from_records(
            &[],
            "parent-session",
            "child-session",
            "explorer",
            0,
            1,
            1,
        );
        state.set_child_view_phase_for_test(AppPhase::WaitingForPermission);

        let rendered = footer_status_spans(&state, crate::tui::Theme::dark())
            .iter()
            .map(|span| span.content.as_ref())
            .collect::<String>();

        assert_eq!(state.active_phase(), AppPhase::WaitingForPermission);
        assert_eq!(rendered, "▲");
    }

    #[test]
    fn footer_phase_indicator_returns_to_parent_session_state() {
        let mut state = TuiState::default();
        state.phase = AppPhase::Running;
        state.replace_child_timeline_from_records(
            &[],
            "parent-session",
            "child-session",
            "explorer",
            0,
            1,
            1,
        );
        state.set_child_view_phase_for_test(AppPhase::Completed);
        assert_eq!(state.active_phase(), AppPhase::Completed);

        state.transcript_view = crate::tui::state::TranscriptViewState::Parent;

        assert_eq!(state.active_phase(), AppPhase::Running);
        assert_eq!(
            footer_status_spans(&state, crate::tui::Theme::dark()).len(),
            8
        );
    }

    #[test]
    fn narrow_footer_keeps_phase_indicator_visible() {
        let mut state = TuiState::default();
        state.set_git_branch(Some("feature/extremely-long-branch-name".into()));
        state.set_current_context_branch("context-long");
        let backend = TestBackend::new(12, 1);
        let mut terminal = Terminal::new(backend).expect("terminal");

        terminal
            .draw(|frame| {
                render_footer(
                    frame,
                    &state,
                    Rect::new(0, 0, 12, 1),
                    crate::tui::Theme::dark(),
                )
            })
            .expect("render footer");

        let row = (0..12)
            .map(|x| {
                terminal
                    .backend()
                    .buffer()
                    .cell((x, 0))
                    .expect("footer cell")
                    .symbol()
            })
            .collect::<String>();
        assert!(row.contains('◆'), "{row}");
    }

    #[test]
    fn footer_hides_git_branch_outside_a_repository() {
        let mut state = TuiState::default();
        state.set_git_branch(None);
        state.set_current_context_branch("root");

        let rendered = footer_hint_spans(&state, crate::tui::Theme::dark(), 80)
            .iter()
            .map(|span| span.content.as_ref())
            .collect::<String>();

        assert!(rendered.starts_with("󰙅 root"), "{rendered}");
        assert!(!rendered.contains(" "), "{rendered}");
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
    fn token_budget_uses_snapshot_usage_for_percent_and_output_bar() {
        let usage = crate::tui::state::ModelTokenUsage {
            used_tokens: 228_000,
            context_window_tokens: 1_000_000,
            input_tokens: 200_000,
            output_tokens: 1_300_000,
            cached_tokens: 0,
            cache_report: None,
            prompt_composition: Vec::new(),
        };

        let rendered = token_budget_spans(&usage, crate::tui::Theme::dark())
            .iter()
            .map(|span| span.content.as_ref())
            .collect::<String>();

        assert!(rendered.contains("↓1.3m"), "{rendered}");
        assert!(rendered.contains("~23%"), "{rendered}");
        assert!(!rendered.contains("100%"), "{rendered}");
        assert_eq!(
            token_budget_segment_units(1_000_000, 10, [0, 200_000, 28_000]),
            [0, 16, 2]
        );
    }

    #[test]
    fn token_budget_cache_hit_percent_is_hidden_without_cached_tokens() {
        assert_eq!(token_budget_cache_hit_percent(40_000, 0), None);
        assert_eq!(token_budget_cache_hit_percent(0, 20_000), None);
    }

    #[test]
    fn token_budget_cache_hit_percent_uses_cached_over_input_tokens() {
        assert_eq!(token_budget_cache_hit_percent(40_000, 20_000), Some(50.0));
        assert_eq!(token_budget_cache_hit_percent(40_000, 80_000), Some(100.0));
        assert_eq!(
            token_budget_cache_hit_percent(3, 1),
            Some(33.33333333333333)
        );
    }

    #[test]
    fn token_budget_cache_hit_percent_renders_two_decimal_places() {
        let usage = crate::tui::state::ModelTokenUsage {
            used_tokens: 3,
            context_window_tokens: 10,
            input_tokens: 3,
            output_tokens: 0,
            cached_tokens: 1,
            cache_report: None,
            prompt_composition: Vec::new(),
        };

        let rendered = token_budget_spans(&usage, crate::tui::Theme::dark())
            .iter()
            .map(|span| span.content.as_ref())
            .collect::<String>();

        assert!(rendered.contains("33.33%"), "{rendered}");
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
            prompt_composition: Vec::new(),
        };

        let rendered = super::token_budget_spans(&usage, crate::tui::Theme::dark())
            .iter()
            .map(|span| span.content.as_ref())
            .collect::<String>();

        assert!(rendered.contains("~50%"), "{rendered}");
    }
}
