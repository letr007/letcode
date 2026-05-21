use ratatui::{
    Frame,
    layout::Rect,
    style::{Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Clear, Paragraph, Wrap},
};

use crate::tui::{
    measure::{CursorVisualPosition, end_cursor_visual_position, wrapped_row_count},
    surface,
    theme::Theme,
};

use super::super::state::TuiState;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ComposerCursor {
    pub row: usize,
    pub column: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ComposerMetrics {
    pub row_count: usize,
    pub cursor: ComposerCursor,
}

impl From<CursorVisualPosition> for ComposerCursor {
    fn from(value: CursorVisualPosition) -> Self {
        Self {
            row: value.row,
            column: value.column,
        }
    }
}

pub fn composer_row_count(input: &str, width: usize) -> usize {
    wrapped_row_count(input, width)
}

pub fn composer_cursor_position(input: &str, width: usize) -> ComposerCursor {
    end_cursor_visual_position(input, width).into()
}

pub fn composer_metrics(input: &str, width: usize) -> ComposerMetrics {
    let cursor = composer_cursor_position(input, width);
    let row_count = composer_row_count(input, width).max(cursor.row.saturating_add(1));

    ComposerMetrics { row_count, cursor }
}

pub fn render_composer(frame: &mut Frame<'_>, state: &TuiState, area: Rect, theme: Theme) {
    if area.is_empty() {
        return;
    }

    if area.height < 3 || area.width < 16 {
        render_composer_tiny(frame, state, area, theme);
        return;
    }

    render_composer_panel(frame, state, area, theme);
}

fn render_composer_tiny(frame: &mut Frame<'_>, state: &TuiState, area: Rect, theme: Theme) {
    let content = if state.input_buffer.is_empty() {
        "message…".to_string()
    } else {
        state.input_buffer.clone()
    };

    let bar_style = surface::accent_style(
        theme,
        surface::SurfaceEmphasis::User,
        surface::SurfaceKind::Root,
    );
    let element_style = surface::surface_style(theme, surface::SurfaceKind::Element);

    let line = Line::from(vec![
        Span::styled(surface::ACCENT_BAR_GLYPH, bar_style),
        Span::styled(" ", element_style),
        Span::styled(content, element_style),
    ]);
    frame.render_widget(Paragraph::new(line).style(element_style), area);

    if state.pending_permission.is_none() {
        // Only set a cursor if we have a usable content cell.
        // Layout is: [bar][space][content...]. Cursor starts in the content region.
        if area.width < 3 {
            return;
        }

        let available = area.width.saturating_sub(2) as usize;
        let cursor = composer_cursor_position(&state.input_buffer, available);
        let desired_x = area
            .x
            .saturating_add(2)
            .saturating_add(cursor.column.min(available.saturating_sub(1)) as u16);
        let max_x = area.x.saturating_add(area.width.saturating_sub(1));
        let x = desired_x.min(max_x);
        frame.set_cursor_position((x, area.y));
    }
}

fn render_composer_panel(frame: &mut Frame<'_>, state: &TuiState, area: Rect, theme: Theme) {
    // Accent bar at the left edge.
    let bar_style = surface::accent_style(
        theme,
        surface::SurfaceEmphasis::User,
        surface::SurfaceKind::Root,
    );
    render_accent_bar(frame, area, bar_style);

    // Element background excluding the bottom cap row.
    let element_style = surface::surface_style(theme, surface::SurfaceKind::Element);
    let surface_area = Rect::new(
        area.x + surface::ACCENT_BAR_WIDTH,
        area.y,
        area.width.saturating_sub(surface::ACCENT_BAR_WIDTH),
        area.height.saturating_sub(1),
    );
    frame.render_widget(Block::new().style(element_style), surface_area);

    // Textarea area inside the element surface.
    let textarea_area = Rect::new(
        area.x + surface::ACCENT_BAR_WIDTH + surface::PROMPT_INNER_PAD_X,
        area.y + surface::PROMPT_INNER_PAD_TOP,
        area.width
            .saturating_sub(surface::ACCENT_BAR_WIDTH)
            .saturating_sub(surface::PROMPT_INNER_PAD_X)
            .saturating_sub(surface::CARD_PAD_RIGHT)
            .max(1),
        area.height
            .saturating_sub(1)
            .saturating_sub(surface::PROMPT_INNER_PAD_TOP)
            .saturating_sub(surface::PROMPT_INNER_PAD_BOTTOM)
            .max(1),
    );

    let content = if state.input_buffer.is_empty() {
        Line::from(Span::styled(
            "message letcode…",
            surface::muted_style(theme, surface::SurfaceKind::Element)
                .add_modifier(Modifier::ITALIC),
        ))
    } else {
        Line::from(Span::styled(state.input_buffer.clone(), element_style))
    };

    frame.render_widget(
        Paragraph::new(content)
            .style(element_style)
            .wrap(Wrap { trim: false }),
        textarea_area,
    );

    if state.pending_permission.is_none() {
        place_composer_cursor(frame, state, textarea_area);
    }

    render_prompt_metadata(frame, state, area, theme);
    render_prompt_cap(frame, area, theme);
}

fn place_composer_cursor(frame: &mut Frame<'_>, state: &TuiState, textarea_area: Rect) {
    if textarea_area.is_empty() {
        return;
    }

    let width = textarea_area.width as usize;
    let metrics = composer_metrics(&state.input_buffer, width);

    // Clamp to the visible textarea so we never set an out-of-bounds cursor.
    let max_row = textarea_area.height.saturating_sub(1) as usize;
    let row = metrics.cursor.row.min(max_row);
    let max_col = textarea_area.width.saturating_sub(1) as usize;
    let col = metrics.cursor.column.min(max_col);

    frame.set_cursor_position((textarea_area.x + col as u16, textarea_area.y + row as u16));
}

fn render_prompt_metadata(frame: &mut Frame<'_>, state: &TuiState, area: Rect, theme: Theme) {
    if area.height < 5 {
        return;
    }

    let metadata_y = area.y + area.height.saturating_sub(2);
    if metadata_y <= area.y {
        return;
    }

    let element_style = surface::surface_style(theme, surface::SurfaceKind::Element);
    let accent = surface::accent_style(
        theme,
        surface::SurfaceEmphasis::User,
        surface::SurfaceKind::Element,
    )
    .add_modifier(Modifier::BOLD);
    let dim = surface::muted_style(theme, surface::SurfaceKind::Element);

    let mode = if state.pending_permission.is_some() {
        "approval pending"
    } else {
        "prompt"
    };

    let metadata = Line::from(vec![
        Span::styled(mode, accent),
        Span::styled(" · ", dim),
        Span::styled(state.model_label.clone(), element_style),
        Span::styled(" · permission ", dim),
        Span::styled(state.permission_mode_label.clone(), element_style),
    ]);

    let metadata_area = Rect::new(
        area.x + surface::ACCENT_BAR_WIDTH + surface::PROMPT_INNER_PAD_X,
        metadata_y,
        area.width
            .saturating_sub(surface::ACCENT_BAR_WIDTH)
            .saturating_sub(surface::PROMPT_INNER_PAD_X)
            .saturating_sub(surface::CARD_PAD_RIGHT)
            .max(1),
        1,
    );

    frame.render_widget(Paragraph::new(metadata).style(element_style), metadata_area);
}

fn render_prompt_cap(frame: &mut Frame<'_>, area: Rect, theme: Theme) {
    if area.height == 0 || area.width == 0 {
        return;
    }

    let cap_y = area.y + area.height.saturating_sub(1);
    let cap_area = Rect::new(area.x, cap_y, area.width, 1);

    frame.render_widget(
        Block::new().style(surface::surface_style(theme, surface::SurfaceKind::Root)),
        cap_area,
    );

    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            surface::PROMPT_BOTTOM_LEFT_GLYPH,
            surface::accent_style(
                theme,
                surface::SurfaceEmphasis::User,
                surface::SurfaceKind::Root,
            ),
        ))),
        Rect::new(area.x, cap_y, 1.min(area.width), 1),
    );

    let cap_width = area.width.saturating_sub(1);
    if cap_width > 0 {
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(
                surface::PROMPT_BOTTOM_CAP_GLYPH.repeat(cap_width as usize),
                Style::default()
                    .fg(surface::surface_bg(theme, surface::SurfaceKind::Element))
                    .bg(surface::surface_bg(theme, surface::SurfaceKind::Root)),
            ))),
            Rect::new(area.x + 1, cap_y, cap_width, 1),
        );
    }
}

fn render_accent_bar(frame: &mut Frame<'_>, area: Rect, style: Style) {
    if area.is_empty() {
        return;
    }

    let bar_area = Rect::new(
        area.x,
        area.y,
        surface::ACCENT_BAR_WIDTH.min(area.width),
        area.height,
    );
    frame.render_widget(Clear, bar_area);
    let lines =
        vec![Line::from(Span::styled(surface::ACCENT_BAR_GLYPH, style)); area.height as usize];
    frame.render_widget(Paragraph::new(Text::from(lines)).style(style), bar_area);
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::{Terminal, backend::TestBackend};

    #[test]
    fn composer_cursor_never_returns_column_equal_to_width() {
        let width = 4;
        let cursor = composer_cursor_position("abcd", width);
        assert!(cursor.column < width, "{cursor:?}");

        let cursor = composer_cursor_position("你", 2);
        assert!(cursor.column < 2, "{cursor:?}");
    }

    #[test]
    fn composer_metrics_account_for_wrapping_and_trailing_newline() {
        // width 2: "你你a" wraps as ["你", "你", "a"] => 3 rows.
        let metrics = composer_metrics("你你a", 2);
        assert_eq!(metrics.row_count, 3);
        assert_eq!(metrics.cursor, ComposerCursor { row: 2, column: 1 });

        // Trailing newline puts the cursor on the next empty row.
        let metrics = composer_metrics("hi\n", 10);
        assert_eq!(metrics.cursor, ComposerCursor { row: 1, column: 0 });
    }

    #[test]
    fn composer_metrics_row_count_always_includes_cursor_row() {
        // Exact-fill should advance cursor row; row_count must still include it.
        let metrics = composer_metrics("abcd", 4);
        assert_eq!(metrics.cursor, ComposerCursor { row: 1, column: 0 });
        assert_eq!(metrics.row_count, 2);
    }

    #[test]
    fn long_single_line_cjk_wraps_into_multiple_rows() {
        // Each CJK char is width 2.
        // width 6 => 3 chars per row.
        let input = "你好世界你好"; // 6 chars => 2 wrapped rows.
        let metrics = composer_metrics(input, 6);
        // Exact-fill on the final row advances the cursor to a new empty visual row.
        assert_eq!(metrics.row_count, 3);
        assert_eq!(metrics.cursor, ComposerCursor { row: 2, column: 0 });

        // Mixed widths.
        let mixed = "ab你cd你ef"; // widths: 1+1+2+1+1+2+1+1 = 10
        let metrics = composer_metrics(mixed, 4);
        assert!(metrics.row_count >= 3, "{metrics:?}");
    }

    #[test]
    fn composer_height_uses_wrapped_rows_instead_of_logical_lines() {
        // Single line that wraps into multiple visual rows at narrow widths.
        let input = "你你你你你你"; // 6 chars => width 12
        let height_narrow = crate::tui::components::layout::composer_height(30, input, 12);
        let height_wide = crate::tui::components::layout::composer_height(30, input, 80);
        assert!(
            height_narrow >= height_wide,
            "{height_narrow} vs {height_wide}"
        );
    }

    #[test]
    fn render_composer_tiny_does_not_panic_on_minuscule_areas() {
        let backend = TestBackend::new(2, 1);
        let mut terminal = Terminal::new(backend).expect("terminal");
        let mut state = TuiState::default();
        state.input_buffer = "你".into();

        terminal
            .draw(|frame| {
                let area = Rect::new(0, 0, 2, 1);
                render_composer(frame, &state, area, Theme::dark());
            })
            .expect("draw");
    }
}
