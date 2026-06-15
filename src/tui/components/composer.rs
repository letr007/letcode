use ratatui::{
    Frame,
    layout::Rect,
    style::{Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Clear, Paragraph, Wrap},
};
use serde_json::Value;

use crate::tui::{
    measure::{
        CursorVisualPosition, display_width, end_cursor_visual_position, wrap_text_to_width,
        wrapped_row_count,
    },
    surface,
    theme::Theme,
    timeline::PermissionView,
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

    if let Some(permission) = &state.pending_permission {
        if area.height < 3 || area.width < 16 {
            render_pending_approval_tiny(frame, permission, area, theme);
        } else {
            render_pending_approval_panel(frame, permission, area, theme);
        }
        return;
    }

    if state.is_read_only_child_view() {
        if area.height < 3 || area.width < 16 {
            render_child_read_only_tiny(frame, state, area, theme);
        } else {
            render_child_read_only_panel(frame, state, area, theme);
        }
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

    let prompt_emphasis = if state.child_navigation_prefix {
        surface::SurfaceEmphasis::Notice
    } else {
        surface::SurfaceEmphasis::User
    };
    let bar_style = surface::accent_style(theme, prompt_emphasis, surface::SurfaceKind::Root);
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
    let prompt_emphasis = if state.child_navigation_prefix {
        surface::SurfaceEmphasis::Notice
    } else {
        surface::SurfaceEmphasis::User
    };
    let bar_style = surface::accent_style(theme, prompt_emphasis, surface::SurfaceKind::Root);
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

    if !state.slash_panel_is_open() {
        render_prompt_metadata(frame, state, area, theme);
    }
    render_prompt_cap(frame, area, theme, prompt_emphasis);
}

fn render_pending_approval_tiny(
    frame: &mut Frame<'_>,
    permission: &PermissionView,
    area: Rect,
    theme: Theme,
) {
    let bar_style = surface::accent_style(
        theme,
        surface::SurfaceEmphasis::Approval,
        surface::SurfaceKind::Root,
    )
    .add_modifier(Modifier::BOLD);
    let element_style = surface::surface_style(theme, surface::SurfaceKind::Element);

    let line = Line::from(vec![
        Span::styled(surface::ACCENT_BAR_GLYPH, bar_style),
        Span::styled(" ", element_style),
        Span::styled(
            compact_permission_summary(permission, area.width as usize),
            element_style,
        ),
    ]);
    frame.render_widget(Paragraph::new(line).style(element_style), area);
}

fn render_pending_approval_panel(
    frame: &mut Frame<'_>,
    permission: &PermissionView,
    area: Rect,
    theme: Theme,
) {
    let bar_style = surface::accent_style(
        theme,
        surface::SurfaceEmphasis::Approval,
        surface::SurfaceKind::Root,
    )
    .add_modifier(Modifier::BOLD);
    render_accent_bar(frame, area, bar_style);

    let pending_style = surface::surface_style(theme, surface::SurfaceKind::Element);
    let surface_area = Rect::new(
        area.x + surface::ACCENT_BAR_WIDTH,
        area.y,
        area.width.saturating_sub(surface::ACCENT_BAR_WIDTH),
        area.height.saturating_sub(1),
    );
    frame.render_widget(Clear, surface_area);
    frame.render_widget(Block::new().style(pending_style), surface_area);

    let content_area = Rect::new(
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

    let lines = vec![
        approval_primary_line(permission, theme, content_area.width as usize),
        Line::from(vec![
            Span::styled("[a] allow once", theme.approval_style()),
            Span::styled(" · ", muted_pending(theme)),
            Span::styled("[d] reject", muted_pending(theme)),
            Span::styled(" · ", muted_pending(theme)),
            Span::styled("Esc deny", muted_pending(theme)),
        ]),
    ];

    frame.render_widget(
        Paragraph::new(Text::from(lines))
            .style(pending_style)
            .wrap(Wrap { trim: false }),
        content_area,
    );

    render_prompt_cap(frame, area, theme, surface::SurfaceEmphasis::Approval);
}

fn render_child_read_only_tiny(frame: &mut Frame<'_>, state: &TuiState, area: Rect, theme: Theme) {
    let element_style = surface::surface_style(theme, surface::SurfaceKind::Element);
    let bar_style = surface::accent_style(
        theme,
        surface::SurfaceEmphasis::Notice,
        surface::SurfaceKind::Root,
    );
    let summary = child_read_only_primary_text(state, area.width.saturating_sub(2) as usize);

    let line = Line::from(vec![
        Span::styled(surface::ACCENT_BAR_GLYPH, bar_style),
        Span::styled(" ", element_style),
        Span::styled(summary, element_style),
    ]);
    frame.render_widget(Paragraph::new(line).style(element_style), area);
}

fn render_child_read_only_panel(frame: &mut Frame<'_>, state: &TuiState, area: Rect, theme: Theme) {
    let emphasis = surface::SurfaceEmphasis::Notice;
    let bar_style = surface::accent_style(theme, emphasis, surface::SurfaceKind::Root);
    render_accent_bar(frame, area, bar_style);

    let element_style = surface::surface_style(theme, surface::SurfaceKind::Element);
    let surface_area = Rect::new(
        area.x + surface::ACCENT_BAR_WIDTH,
        area.y,
        area.width.saturating_sub(surface::ACCENT_BAR_WIDTH),
        area.height.saturating_sub(1),
    );
    frame.render_widget(Block::new().style(element_style), surface_area);

    let content_area = Rect::new(
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
    let [left_area, right_area] = split_read_only_content(content_area);

    frame.render_widget(
        Paragraph::new(Text::from(child_read_only_lines(state, theme, left_area.width as usize)))
            .style(element_style)
            .wrap(Wrap { trim: false }),
        left_area,
    );
    frame.render_widget(
        Paragraph::new(Text::from(child_read_only_hints(theme)))
            .style(element_style)
            .alignment(ratatui::layout::Alignment::Right)
            .wrap(Wrap { trim: false }),
        right_area,
    );

    render_prompt_cap(frame, area, theme, emphasis);
}

fn split_read_only_content(area: Rect) -> [Rect; 2] {
    if area.width < 48 || area.height < 2 {
        return [area, Rect::new(area.x, area.y, 0, 0)];
    }

    let right_width = area.width.min(28);
    let left_width = area.width.saturating_sub(right_width);
    [
        Rect::new(area.x, area.y, left_width, area.height),
        Rect::new(area.x + left_width, area.y, right_width, area.height),
    ]
}

fn child_read_only_lines(state: &TuiState, theme: Theme, width: usize) -> Vec<Line<'static>> {
    let Some(child) = state.child_view_metadata() else {
        return vec![Line::from(Span::raw("Child transcript"))];
    };

    let mut top = vec![
        Span::styled(child.agent_name.clone(), muted_pending(theme)),
        Span::styled(" · ", muted_pending(theme)),
        Span::styled(
            format!("{}/{}", child.index + 1, child.total),
            inline_pending(theme),
        ),
        Span::styled(" · ", muted_pending(theme)),
        Span::styled(
            one_line_snippet(&child.child_session_id, width.saturating_sub(12).max(1)),
            inline_pending(theme),
        ),
    ];

    let mut details = Vec::new();
    if let Some(model) = child.model {
        details.push(format!("model {model}"));
    }
    details.push(format!("{} records", child.record_count));
    details.push(format!(
        "parent {}",
        one_line_snippet(&child.parent_session_id, 16)
    ));

    let bottom = Line::from(Span::styled(
        one_line_snippet(&details.join(" · "), width.max(1)),
        muted_pending(theme),
    ));

    if top.is_empty() {
        vec![bottom]
    } else {
        vec![Line::from(std::mem::take(&mut top)), bottom]
    }
}

fn child_read_only_hints(theme: Theme) -> Vec<Line<'static>> {
    vec![
        Line::from(Span::styled("Read-only child view", muted_pending(theme))),
        Line::from(vec![
            Span::styled("↑", inline_pending(theme).add_modifier(Modifier::BOLD)),
            Span::styled(" Parent", muted_pending(theme)),
            Span::styled("   ", muted_pending(theme)),
            Span::styled("←", inline_pending(theme).add_modifier(Modifier::BOLD)),
            Span::styled(" Prev", muted_pending(theme)),
            Span::styled("   ", muted_pending(theme)),
            Span::styled("→", inline_pending(theme).add_modifier(Modifier::BOLD)),
            Span::styled(" Next", muted_pending(theme)),
        ]),
    ]
}

fn child_read_only_primary_text(state: &TuiState, width: usize) -> String {
    let Some(child) = state.child_view_metadata() else {
        return "Child transcript · ↑ Parent".into();
    };

    one_line_snippet(
        &format!(
            "{} {}/{} · {} · ↑ Parent ← Prev → Next",
            child.agent_name,
            child.index + 1,
            child.total,
            child.child_session_id
        ),
        width.max(1),
    )
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

    let mut spans = vec![
        Span::styled(mode, accent),
        Span::styled(" · ", dim),
        Span::styled(state.model_label.clone(), element_style),
        Span::styled(" · ", dim),
        Span::styled(state.provider_label.clone(), element_style),
    ];
    if let Some(reasoning) = &state.reasoning_effort_label {
        spans.push(Span::styled(" · ", dim));
        spans.push(Span::styled(reasoning.clone(), element_style));
    }
    spans.extend([
        Span::styled(" · ", dim),
        Span::styled(state.permission_mode_label.clone(), element_style),
    ]);

    let metadata = Line::from(spans);

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

fn render_prompt_cap(
    frame: &mut Frame<'_>,
    area: Rect,
    theme: Theme,
    emphasis: surface::SurfaceEmphasis,
) {
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
            surface::accent_style(theme, emphasis, surface::SurfaceKind::Root),
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

fn compact_permission_summary(permission: &PermissionView, width: usize) -> String {
    one_line_snippet(
        &approval_primary_text(permission),
        width.saturating_sub(2).max(1),
    )
}

fn approval_primary_line(permission: &PermissionView, theme: Theme, width: usize) -> Line<'static> {
    let label = approval_action_label(permission);
    let label_width = display_width(label);
    let subject = one_line_snippet(
        &approval_subject(permission),
        width.saturating_sub(label_width + 3).max(1),
    );

    Line::from(vec![
        Span::styled(label.to_string(), theme.approval_style()),
        Span::styled(" · ", muted_pending(theme)),
        Span::styled(subject, inline_pending(theme)),
    ])
}

fn approval_primary_text(permission: &PermissionView) -> String {
    format!(
        "{} · {}",
        approval_action_label(permission),
        approval_subject(permission)
    )
}

fn approval_action_label(permission: &PermissionView) -> &'static str {
    match permission.tool_name.as_str() {
        "shell__exec" => "Run command",
        "fs__read" => "Read file",
        "fs__write" => "Write file",
        "fs__append" => "Append file",
        "fs__mkdir" => "Create directory",
        "search__rg" => "Search text",
        "code__ast_search" => "Search code",
        "edit__apply_patch" => "Apply patch",
        _ => "Approve tool",
    }
}

fn approval_subject(permission: &PermissionView) -> String {
    match permission.tool_name.as_str() {
        "shell__exec" => extract_json_argument(permission, &["command"]),
        "fs__read" | "fs__write" | "fs__append" | "fs__mkdir" => {
            extract_json_argument(permission, &["path", "filePath"])
        }
        "search__rg" => extract_json_argument(permission, &["pattern"]),
        "code__ast_search" => extract_json_argument(permission, &["pattern", "query"]),
        _ => None,
    }
    .unwrap_or_else(|| permission.summary.clone())
}

fn extract_json_argument(permission: &PermissionView, keys: &[&str]) -> Option<String> {
    let raw = permission.arguments.as_deref()?;
    let value = serde_json::from_str::<Value>(raw).ok()?;

    keys.iter().find_map(|key| {
        value
            .get(*key)
            .and_then(Value::as_str)
            .map(ToString::to_string)
    })
}

pub(crate) fn one_line_snippet(value: &str, width: usize) -> String {
    let compact = value.split_whitespace().collect::<Vec<_>>().join(" ");
    if compact.is_empty() {
        return String::new();
    }

    let wrapped = wrap_text_to_width(&compact, width.max(1));
    let mut first = wrapped.first().cloned().unwrap_or_default();
    if wrapped.len() > 1 || display_width(&compact) > width.max(1) {
        if !first.ends_with('…') {
            first.push('…');
        }
    }
    first
}

fn inline_pending(theme: Theme) -> Style {
    Style::default().fg(theme.text).bg(theme.element_bg)
}

fn muted_pending(theme: Theme) -> Style {
    Style::default().fg(theme.muted_text).bg(theme.element_bg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tui::{AppEvent, PermissionRequestEvent};
    use ratatui::style::Color;
    use ratatui::{Terminal, backend::TestBackend};

    fn draw_to_string(state: &TuiState, width: u16, height: u16) -> String {
        let backend = TestBackend::new(width, height);
        let mut terminal = Terminal::new(backend).expect("terminal");

        terminal
            .draw(|frame| {
                let area = Rect::new(0, 0, width, height);
                render_composer(frame, state, area, Theme::dark());
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

    fn leading_bar_color(state: &TuiState, width: u16, height: u16) -> Option<Color> {
        let backend = TestBackend::new(width, height);
        let mut terminal = Terminal::new(backend).expect("terminal");

        terminal
            .draw(|frame| {
                let area = Rect::new(0, 0, width, height);
                render_composer(frame, state, area, Theme::dark());
            })
            .expect("draw");

        terminal.backend().buffer().cell((0, 0)).map(|cell| cell.fg)
    }

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

    #[test]
    fn pending_permission_takes_over_composer_surface() {
        let mut state = TuiState::default();
        let mut request =
            PermissionRequestEvent::new("call-1", "shell__exec", "cargo test --workspace");
        request.arguments = Some("cargo test --workspace".into());
        state.apply_event(AppEvent::PermissionRequested(request));

        let rendered = draw_to_string(&state, 80, 8);

        assert!(
            rendered.contains("Approve tool") || rendered.contains("Run command"),
            "{rendered}"
        );
        assert!(rendered.contains("allow once"), "{rendered}");
        assert!(rendered.contains("reject"), "{rendered}");
        assert!(!rendered.contains("message letcode"), "{rendered}");
        assert!(!rendered.contains("args"), "{rendered}");
    }

    #[test]
    fn slash_panel_content_is_not_rendered_inside_composer_surface() {
        let mut state = TuiState::default();
        state.set_input("/per");

        let rendered = draw_to_string(&state, 100, 12);
        assert!(
            !rendered.contains("Show current permission mode"),
            "{rendered}"
        );
        assert!(rendered.contains("/per"), "{rendered}");
        assert!(!rendered.contains("prompt ·"), "{rendered}");
    }

    #[test]
    fn composer_metadata_includes_model_provider_and_permission() {
        let mut state = TuiState::new("gpt-5.5", "GPT-5.5", "default");
        state.set_provider_label("CLI Proxy API");
        state.set_reasoning_effort_label(Some("medium".into()));

        let rendered = draw_to_string(&state, 100, 8);

        assert!(rendered.contains("prompt"), "{rendered}");
        assert!(rendered.contains("GPT-5.5"), "{rendered}");
        assert!(rendered.contains("CLI Proxy API"), "{rendered}");
        assert!(rendered.contains("medium"), "{rendered}");
        assert!(rendered.contains("default"), "{rendered}");
        assert!(!rendered.contains("permission default"), "{rendered}");
    }

    #[test]
    fn child_navigation_prefix_mutes_composer_leading_bar() {
        let mut state = TuiState::default();

        let normal = leading_bar_color(&state, 80, 8).expect("normal color");
        state.child_navigation_prefix = true;
        let prefixed = leading_bar_color(&state, 80, 8).expect("prefixed color");

        assert_eq!(normal, Theme::dark().user);
        assert_eq!(prefixed, Theme::dark().notice);
    }

    #[test]
    fn child_view_renders_read_only_status_bar_instead_of_input_composer() {
        let mut state = TuiState::new("gpt-5.5", "GPT-5.5", "default");
        state.set_provider_label("CLI Proxy API");
        state.set_input("hidden input should not render");
        state.replace_child_timeline_from_records(
            &[crate::transcript::TranscriptRecord {
                session_id: "child-session".into(),
                sequence: 1,
                timestamp_ms: 0,
                event: crate::transcript::TranscriptEvent::SessionStarted {
                    model: "gpt-5.5-mini".into(),
                },
            }],
            "parent-session",
            "child-session-1234567890",
            "fixer",
            1,
            3,
        );

        let rendered = draw_to_string(&state, 100, 8);

        assert!(rendered.contains("Read-only child view"), "{rendered}");
        assert!(rendered.contains("fixer"), "{rendered}");
        assert!(rendered.contains("2/3"), "{rendered}");
        assert!(rendered.contains("gpt-5.5-mini"), "{rendered}");
        assert!(rendered.contains("Parent"), "{rendered}");
        assert!(rendered.contains("Prev"), "{rendered}");
        assert!(rendered.contains("Next"), "{rendered}");
        assert!(!rendered.contains("message letcode"), "{rendered}");
        assert!(!rendered.contains("hidden input should not render"), "{rendered}");
    }

    #[test]
    fn slash_panel_is_hidden_while_permission_prompt_is_pending() {
        let mut state = TuiState::default();
        state.set_input("/per");
        let request =
            PermissionRequestEvent::new("call-1", "shell__exec", "cargo test --workspace");
        state.apply_event(AppEvent::PermissionRequested(request));

        let rendered = draw_to_string(&state, 100, 12);
        assert!(
            !rendered.contains("Show current permission mode"),
            "{rendered}"
        );
    }
}
