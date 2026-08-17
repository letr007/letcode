use ratatui::{
    Frame,
    layout::Rect,
    style::{Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Clear, Paragraph},
};

use crate::tui::{
    measure::display_width,
    slash::{MAX_VISIBLE_SLASH_COMMANDS, matching_completion_commands},
    state::TuiState,
    surface,
    theme::Theme,
};

use super::composer::one_line_snippet;

pub fn slash_panel_row_count(state: &TuiState) -> u16 {
    if !state.slash_panel_is_open() {
        return 0;
    }

    matching_completion_commands(&state.input_buffer)
        .len()
        .clamp(1, MAX_VISIBLE_SLASH_COMMANDS) as u16
}

pub fn slash_panel_reserved_height(state: &TuiState) -> u16 {
    slash_panel_row_count(state)
}

pub fn render_slash_panel(frame: &mut Frame<'_>, state: &TuiState, area: Rect, theme: Theme) {
    if area.is_empty() || !state.slash_panel_is_open() {
        return;
    }

    let matches = matching_completion_commands(&state.input_buffer);
    let row_count = matches.len().clamp(1, MAX_VISIBLE_SLASH_COMMANDS) as u16;
    let panel_area = Rect::new(
        area.x + surface::ACCENT_BAR_WIDTH,
        area.y,
        area.width.saturating_sub(surface::ACCENT_BAR_WIDTH),
        row_count,
    );

    let surface_style = surface::surface_style(theme, surface::SurfaceKind::Elevated);
    render_accent_bar(
        frame,
        Rect::new(area.x, area.y, area.width, row_count),
        theme,
    );
    frame.render_widget(Block::new().style(surface_style), panel_area);

    if matches.is_empty() {
        let line = Line::from(vec![
            Span::styled("  ", row_padding_style(theme, false)),
            Span::styled(state.t("parse.no_match"), description_style(theme, false)),
        ]);
        frame.render_widget(Paragraph::new(line).style(surface_style), panel_area);
        return;
    }

    let selected = state
        .slash_panel_selected
        .min(matches.len().saturating_sub(1));
    let visible_rows = row_count as usize;
    let content_width = panel_area.width as usize;
    let viewport_start = slash_panel_viewport_start(matches.len(), selected, visible_rows);
    let visible_matches = matches
        .iter()
        .skip(viewport_start)
        .take(visible_rows)
        .copied()
        .collect::<Vec<_>>();
    let command_width = matches
        .iter()
        .skip(viewport_start)
        .take(visible_rows)
        .map(|entry| display_width(entry.command))
        .max()
        .unwrap_or(0)
        .min(content_width.saturating_sub(4));

    for (row_index, entry) in visible_matches.into_iter().enumerate() {
        let command_index = viewport_start + row_index;
        let is_selected = command_index == selected;
        let row_area = Rect::new(
            panel_area.x,
            panel_area.y + row_index as u16,
            panel_area.width,
            1,
        );
        let row_style = if is_selected {
            highlighted_row_style(theme)
        } else {
            surface_style
        };
        frame.render_widget(Block::new().style(row_style), row_area);

        let scroll_indicator =
            scroll_indicator(row_index, visible_rows, viewport_start, matches.len());
        let spacer_width = command_width.saturating_sub(display_width(entry.command)) + 2;
        let description_width = panel_area
            .width
            .saturating_sub(4)
            .saturating_sub(display_width(entry.command) as u16)
            .saturating_sub(spacer_width as u16)
            .max(1) as usize;
        let description = one_line_snippet(&state.t(entry.description_key), description_width);

        let line = Line::from(vec![
            Span::styled(
                selection_marker(is_selected),
                marker_style(theme, is_selected),
            ),
            Span::styled(entry.command.to_string(), command_style(theme, is_selected)),
            Span::styled(" ".repeat(spacer_width), row_style),
            Span::styled(description, description_style(theme, is_selected)),
            Span::styled(scroll_indicator, indicator_style(theme, is_selected)),
        ]);
        frame.render_widget(Paragraph::new(line).style(row_style), row_area);
    }
}

fn render_accent_bar(frame: &mut Frame<'_>, area: Rect, theme: Theme) {
    if area.is_empty() {
        return;
    }

    let bar_area = Rect::new(
        area.x,
        area.y,
        surface::ACCENT_BAR_WIDTH.min(area.width),
        area.height,
    );
    let style = surface::accent_style(
        theme,
        surface::SurfaceEmphasis::User,
        surface::SurfaceKind::Root,
    );
    frame.render_widget(Clear, bar_area);
    let lines =
        vec![Line::from(Span::styled(surface::ACCENT_BAR_GLYPH, style)); area.height as usize];
    frame.render_widget(Paragraph::new(Text::from(lines)).style(style), bar_area);
}

fn slash_panel_viewport_start(total: usize, selected: usize, visible_rows: usize) -> usize {
    if visible_rows == 0 || total <= visible_rows {
        return 0;
    }

    selected
        .saturating_add(1)
        .saturating_sub(visible_rows)
        .min(total.saturating_sub(visible_rows))
}

fn selection_marker(selected: bool) -> &'static str {
    if selected { "› " } else { "  " }
}

fn scroll_indicator(
    row: usize,
    visible_rows: usize,
    viewport_start: usize,
    total: usize,
) -> &'static str {
    if row == 0 && viewport_start > 0 {
        " ↑"
    } else if row + 1 == visible_rows && viewport_start + visible_rows < total {
        " ↓"
    } else {
        "  "
    }
}

fn highlighted_row_style(theme: Theme) -> Style {
    Style::default().bg(theme.element_bg).fg(theme.text)
}

fn row_padding_style(theme: Theme, selected: bool) -> Style {
    let background = if selected {
        theme.element_bg
    } else {
        theme.elevated_bg
    };
    Style::default().fg(theme.text).bg(background)
}

fn marker_style(theme: Theme, selected: bool) -> Style {
    Style::default()
        .fg(if selected {
            theme.accent
        } else {
            theme.dim_text
        })
        .bg(if selected {
            theme.element_bg
        } else {
            theme.elevated_bg
        })
        .add_modifier(if selected {
            Modifier::BOLD
        } else {
            Modifier::empty()
        })
}

fn command_style(theme: Theme, selected: bool) -> Style {
    let background = if selected {
        theme.element_bg
    } else {
        theme.elevated_bg
    };
    Style::default()
        .fg(if selected { theme.text } else { theme.user })
        .bg(background)
        .add_modifier(Modifier::BOLD)
}

fn description_style(theme: Theme, selected: bool) -> Style {
    Style::default()
        .fg(if selected {
            theme.text
        } else {
            theme.muted_text
        })
        .bg(if selected {
            theme.element_bg
        } else {
            theme.elevated_bg
        })
}

fn indicator_style(theme: Theme, selected: bool) -> Style {
    let background = if selected {
        theme.element_bg
    } else {
        theme.elevated_bg
    };
    Style::default().fg(theme.dim_text).bg(background)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::{Terminal, backend::TestBackend};

    fn draw_panel(state: &TuiState, width: u16, height: u16) -> String {
        let backend = TestBackend::new(width, height);
        let mut terminal = Terminal::new(backend).expect("terminal");

        terminal
            .draw(|frame| {
                render_slash_panel(frame, state, Rect::new(0, 0, width, height), Theme::dark())
            })
            .expect("draw");

        terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|cell| cell.symbol())
            .collect()
    }

    #[test]
    fn completion_panel_is_hidden_in_child_read_only_view() {
        let mut state = TuiState::default();
        state.replace_child_timeline_from_records(
            &[],
            "parent-session",
            "child-session",
            "explorer",
            0,
            1,
            1,
        );
        state.set_input("@fi");

        let rendered = draw_panel(&state, 72, 6);
        assert!(!rendered.contains("@fixer"), "{rendered}");
    }
}
