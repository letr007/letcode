use ratatui::{
    Frame,
    layout::Rect,
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Paragraph},
};

use crate::tui::{
    measure::display_width,
    slash::{MAX_VISIBLE_SLASH_COMMANDS, matching_slash_commands},
    state::TuiState,
    surface,
    theme::Theme,
};

use super::composer::one_line_snippet;

pub fn slash_panel_row_count(state: &TuiState) -> u16 {
    if !state.slash_panel_is_open() {
        return 0;
    }

    matching_slash_commands(&state.input_buffer)
        .len()
        .clamp(1, MAX_VISIBLE_SLASH_COMMANDS) as u16
}

pub fn slash_panel_reserved_height(state: &TuiState) -> u16 {
    let rows = slash_panel_row_count(state);
    rows
}

pub fn render_slash_panel(frame: &mut Frame<'_>, state: &TuiState, area: Rect, theme: Theme) {
    if area.is_empty() || !state.slash_panel_is_open() {
        return;
    }

    let matches = matching_slash_commands(&state.input_buffer);
    let row_count = matches.len().clamp(1, MAX_VISIBLE_SLASH_COMMANDS) as u16;
    let panel_area = Rect::new(area.x, area.y, area.width, row_count);

    let surface_style = surface::surface_style(theme, surface::SurfaceKind::Elevated);
    frame.render_widget(Block::new().style(surface_style), panel_area);

    if matches.is_empty() {
        let line = Line::from(vec![
            Span::styled(" ", row_padding_style(theme, false)),
            Span::styled("No matching commands", description_style(theme, false)),
        ]);
        frame.render_widget(Paragraph::new(line).style(surface_style), panel_area);
        return;
    }

    let selected = state
        .slash_panel_selected
        .min(matches.len().saturating_sub(1));
    let command_width = matches
        .iter()
        .take(MAX_VISIBLE_SLASH_COMMANDS)
        .map(|entry| display_width(entry.command))
        .max()
        .unwrap_or(0)
        .min(area.width.saturating_sub(3) as usize);

    for (index, entry) in matches
        .into_iter()
        .take(MAX_VISIBLE_SLASH_COMMANDS)
        .enumerate()
    {
        let is_selected = index == selected;
        let row_area = Rect::new(area.x, area.y + index as u16, area.width, 1);
        let row_style = if is_selected {
            highlighted_row_style(theme)
        } else {
            surface_style
        };
        frame.render_widget(Block::new().style(row_style), row_area);

        let spacer_width = command_width.saturating_sub(display_width(entry.command)) + 3;
        let description_width = area
            .width
            .saturating_sub(1)
            .saturating_sub(display_width(entry.command) as u16)
            .saturating_sub(spacer_width as u16)
            .max(1) as usize;
        let description = one_line_snippet(entry.description, description_width);

        let line = Line::from(vec![
            Span::styled(" ", row_padding_style(theme, is_selected)),
            Span::styled(entry.command.to_string(), command_style(theme, is_selected)),
            Span::styled(" ".repeat(spacer_width), row_style),
            Span::styled(description, description_style(theme, is_selected)),
        ]);
        frame.render_widget(Paragraph::new(line).style(row_style), row_area);
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
    fn slash_panel_renders_matching_commands() {
        let mut state = TuiState::default();
        state.set_input("/per");

        let rendered = draw_panel(&state, 72, 6);
        assert!(rendered.contains("/permission"), "{rendered}");
        assert!(
            rendered.contains("Show current permission mode"),
            "{rendered}"
        );
    }

    #[test]
    fn slash_panel_shows_empty_state_for_unknown_command() {
        let mut state = TuiState::default();
        state.set_input("/wat");

        let rendered = draw_panel(&state, 40, 3);
        assert!(rendered.contains("No matching commands"), "{rendered}");
    }
}
