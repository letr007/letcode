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

    if let Some(usage) = state.model_token_usage {
        spans.push(Span::styled(
            format_token_usage(usage),
            footer_dim_style(theme),
        ));
    }

    if !matches!(state.phase, AppPhase::WaitingForPermission) && !state.slash_panel_is_open() {
        if !spans.is_empty() {
            spans.push(Span::styled(" · ", footer_dim_style(theme)));
        }
        spans.push(Span::styled(
            "/help commands · exit to quit",
            footer_dim_style(theme),
        ));
    }

    spans
}

fn format_token_usage(usage: crate::tui::state::ModelTokenUsage) -> String {
    let percent = if usage.context_window_tokens == 0 {
        0
    } else {
        usage.used_tokens.saturating_mul(100) / usage.context_window_tokens
    };

    format!(
        "{}/{} ({}%)",
        format_token_window(usage.used_tokens),
        format_token_window(usage.context_window_tokens),
        percent
    )
}

fn format_token_window(tokens: u64) -> String {
    if tokens >= 1_000_000 {
        format!("{:.1}M", tokens as f64 / 1_000_000.0)
    } else if tokens >= 1_000 {
        format!("{:.1}K", tokens as f64 / 1_000.0)
    } else {
        tokens.to_string()
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
