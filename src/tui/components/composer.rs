use ratatui::{
    Frame,
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Clear, Paragraph, Wrap},
};
use serde_json::Value;

use super::super::state::{PermissionChoice, TuiState};
use super::choice;
use crate::tui::{
    measure::{display_width, wrap_text_to_width},
    surface,
    theme::Theme,
    timeline::PermissionView,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ConnectedPromptShell {
    pub content_area: Rect,
    pub footer_area: Option<Rect>,
}

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ComposerCursorPulse {
    bg: Color,
    fg: Color,
}

const CURSOR_FRAME_INTERVAL_MS: usize = 33;
const CURSOR_CYCLE_DURATION_MS: usize = 1_000;

#[cfg(test)]
impl From<crate::tui::measure::CursorVisualPosition> for ComposerCursor {
    fn from(value: crate::tui::measure::CursorVisualPosition) -> Self {
        Self {
            row: value.row,
            column: value.column,
        }
    }
}

#[cfg(test)]
pub(crate) fn composer_row_count(input: &str, width: usize) -> usize {
    crate::tui::measure::wrapped_row_count(input, width)
}

#[cfg(test)]
pub(crate) fn composer_cursor_position(
    input: &str,
    width: usize,
    cursor_byte_index: usize,
) -> ComposerCursor {
    crate::tui::measure::cursor_visual_position(input, width, cursor_byte_index).into()
}

#[cfg(test)]
pub(crate) fn composer_metrics(
    input: &str,
    width: usize,
    cursor_byte_index: usize,
) -> ComposerMetrics {
    let cursor = composer_cursor_position(input, width, cursor_byte_index);
    let row_count = composer_row_count(input, width).max(cursor.row.saturating_add(1));

    ComposerMetrics { row_count, cursor }
}

pub(crate) fn composer_textarea_width(area_width: u16) -> usize {
    area_width
        .saturating_sub(surface::ACCENT_BAR_WIDTH)
        .saturating_sub(surface::PROMPT_INNER_PAD_X)
        .saturating_sub(surface::CARD_PAD_RIGHT) as usize
}

fn composer_metrics_with_attachments(state: &TuiState, width: usize) -> ComposerMetrics {
    state.assert_composer_token_invariant();
    let width = width.max(1);
    let mut row = 0usize;
    let mut col = 0usize;
    let mut cursor = None;
    let mut tokens = state.composer_tokens.iter();
    let mut image_index = 0usize;
    let mut byte_index = 0usize;
    let mut ended_by_exact_fill = false;

    for ch in state.input_buffer.chars() {
        if byte_index == state.input_cursor && cursor.is_none() {
            cursor = Some(ComposerCursor { row, column: col });
        }

        if ch == crate::tui::state::COMPOSER_ATTACHMENT_MARKER {
            let token = tokens.next().expect("composer marker has matching token");
            let token_width = display_width(&token.display_text(image_index));
            if col > 0 && col + token_width > width {
                row = row.saturating_add(1);
                col = 0;
            }
            col += token_width;
            if matches!(token, crate::tui::state::ComposerToken::Image(_)) {
                image_index += 1;
            }
        } else if ch == '\n' {
            if !ended_by_exact_fill {
                row = row.saturating_add(1);
            }
            col = 0;
        } else {
            let ch_width = display_width(&ch.to_string());
            if ch_width > 0 && col > 0 && col + ch_width > width {
                row = row.saturating_add(1);
                col = 0;
            }
            col += ch_width;
        }

        byte_index += ch.len_utf8();
        ended_by_exact_fill = ch != '\n' && col >= width;
        if col >= width {
            row = row.saturating_add(1);
            col = 0;
        }
    }

    if byte_index == state.input_cursor && cursor.is_none() {
        cursor = Some(ComposerCursor { row, column: col });
    }
    let cursor = cursor.unwrap_or(ComposerCursor { row, column: col });
    ComposerMetrics {
        row_count: row.saturating_add(1).max(cursor.row.saturating_add(1)),
        cursor,
    }
}

fn composer_scroll_row(metrics: ComposerMetrics, visible_rows: u16) -> usize {
    let visible_rows = visible_rows.max(1) as usize;
    metrics
        .cursor
        .row
        .saturating_sub(visible_rows.saturating_sub(1))
}

pub fn render_composer(frame: &mut Frame<'_>, state: &mut TuiState, area: Rect, theme: Theme) {
    if area.is_empty() {
        return;
    }

    if let Some(permission) = &state.pending_permission {
        let translator = state.translator();
        let highlighted = state.pending_permission_choice.index();
        if area.height < 3 || area.width < 16 {
            render_pending_approval_tiny(frame, permission, area, theme, &translator);
        } else {
            render_pending_approval_panel(frame, permission, highlighted, area, theme, &translator);
        }
        return;
    }

    if state.is_read_only_child_view() && state.input_buffer.is_empty() {
        if area.height < 4 || area.width < 16 {
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

fn render_composer_tiny(frame: &mut Frame<'_>, state: &mut TuiState, area: Rect, theme: Theme) {
    let prompt_emphasis = if state.child_navigation_prefix {
        surface::SurfaceEmphasis::Notice
    } else {
        surface::SurfaceEmphasis::User
    };
    let bar_style = surface::accent_style(theme, prompt_emphasis, surface::SurfaceKind::Root);
    let element_style = surface::surface_style(theme, surface::SurfaceKind::Element);

    let inline = composer_inline_lines(state, area.width.saturating_sub(2) as usize, theme)
        .into_iter()
        .next()
        .unwrap_or_else(|| {
            Line::from(Span::styled(
                state.t("ui.message_placeholder"),
                element_style,
            ))
        });
    let mut spans = vec![
        Span::styled(surface::ACCENT_BAR_GLYPH, bar_style),
        Span::styled(" ", element_style),
    ];
    spans.extend(inline.spans);
    let line = Line::from(spans);
    frame.render_widget(Paragraph::new(line).style(element_style), area);

    if state.pending_permission.is_none() && !state.dialog_is_open() {
        render_tiny_composer_cursor(frame, state, area, theme);
    }
}

fn render_composer_panel(frame: &mut Frame<'_>, state: &mut TuiState, area: Rect, theme: Theme) {
    let framed = theme.card_frame && area.width >= 3 && area.height >= 3;
    let top_pad = surface::PROMPT_INNER_PAD_TOP;

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
        area.y + u16::from(framed),
        area.width
            .saturating_sub(surface::ACCENT_BAR_WIDTH)
            .saturating_sub(u16::from(framed)),
        area.height
            .saturating_sub(1)
            .saturating_sub(u16::from(framed)),
    );
    frame.render_widget(Block::new().style(element_style), surface_area);

    // Textarea area inside the element surface.
    let textarea_area = Rect::new(
        area.x + surface::ACCENT_BAR_WIDTH + surface::PROMPT_INNER_PAD_X,
        area.y + top_pad + u16::from(framed),
        u16::try_from(composer_textarea_width(area.width))
            .unwrap_or(u16::MAX)
            .saturating_sub(u16::from(framed))
            .max(1),
        area.height
            .saturating_sub(1)
            .saturating_sub(u16::from(framed))
            .saturating_sub(top_pad)
            .saturating_sub(surface::PROMPT_INNER_PAD_BOTTOM)
            .max(1),
    );

    let metrics = composer_metrics_with_attachments(state, textarea_area.width as usize);
    let scroll_row = composer_scroll_row(metrics, textarea_area.height);
    let content = Text::from(composer_inline_lines(
        state,
        textarea_area.width as usize,
        theme,
    ));

    frame.render_widget(
        Paragraph::new(content)
            .style(element_style)
            .scroll((u16::try_from(scroll_row).unwrap_or(u16::MAX), 0))
            .wrap(Wrap { trim: false }),
        textarea_area,
    );

    if state.pending_permission.is_none() && !state.dialog_is_open() {
        render_panel_composer_cursor(frame, state, metrics, scroll_row, textarea_area, theme);
    }

    if !state.slash_panel_is_open() {
        render_prompt_metadata(frame, state, area, theme);
    }
    render_prompt_cap(frame, area, theme, prompt_emphasis);
    render_three_sided_frame(frame, area, theme);
}

pub(crate) fn render_connected_prompt_shell(
    frame: &mut Frame<'_>,
    area: Rect,
    theme: Theme,
    emphasis: surface::SurfaceEmphasis,
    footer_height: u16,
) -> Option<ConnectedPromptShell> {
    render_connected_prompt_shell_with_bar(frame, area, theme, emphasis, footer_height, true)
}

pub(crate) fn render_connected_prompt_shell_without_bar(
    frame: &mut Frame<'_>,
    area: Rect,
    theme: Theme,
    emphasis: surface::SurfaceEmphasis,
    footer_height: u16,
) -> Option<ConnectedPromptShell> {
    render_connected_prompt_shell_with_bar(frame, area, theme, emphasis, footer_height, false)
}

fn render_connected_prompt_shell_with_bar(
    frame: &mut Frame<'_>,
    area: Rect,
    theme: Theme,
    emphasis: surface::SurfaceEmphasis,
    footer_height: u16,
    show_accent_bar: bool,
) -> Option<ConnectedPromptShell> {
    if area.is_empty() {
        return None;
    }

    let framed = theme.card_frame && area.width >= 3 && area.height >= 3;
    let top_pad = surface::PROMPT_INNER_PAD_TOP;
    let bar_width = if show_accent_bar {
        surface::ACCENT_BAR_WIDTH
    } else {
        0
    };
    if show_accent_bar {
        let bar_style = surface::accent_style(theme, emphasis, surface::SurfaceKind::Root)
            .add_modifier(Modifier::BOLD);
        render_accent_bar(frame, area, bar_style);
    }

    let panel_style = surface::surface_style(theme, surface::SurfaceKind::Element);
    let surface_area = Rect::new(
        area.x + bar_width,
        area.y + u16::from(framed),
        area.width
            .saturating_sub(bar_width)
            .saturating_sub(u16::from(framed)),
        area.height
            .saturating_sub(1)
            .saturating_sub(u16::from(framed)),
    );
    frame.render_widget(Block::new().style(panel_style), surface_area);

    render_prompt_cap_with_bar(frame, area, theme, emphasis, show_accent_bar);
    render_three_sided_frame(frame, area, theme);

    let inner_x = area.x + bar_width + surface::PROMPT_INNER_PAD_X;
    let inner_y = area.y + top_pad + u16::from(framed);
    let inner_width = area
        .width
        .saturating_sub(bar_width)
        .saturating_sub(surface::PROMPT_INNER_PAD_X)
        .saturating_sub(surface::CARD_PAD_RIGHT)
        .saturating_sub(u16::from(framed))
        .max(1);
    let inner_height = area
        .height
        .saturating_sub(1)
        .saturating_sub(u16::from(framed))
        .saturating_sub(top_pad)
        .saturating_sub(surface::PROMPT_INNER_PAD_BOTTOM)
        .max(1);

    // A two-row connected prompt only has one usable inner row after the cap and top padding.
    // Keep that row for the question content (and, critically, an active custom-input cursor)
    // instead of placing content and the footer on top of one another.
    let footer_height = if inner_height >= 2 {
        footer_height.min(inner_height - 1)
    } else {
        0
    };
    let content_height = inner_height.saturating_sub(footer_height);
    let content_area = Rect::new(inner_x, inner_y, inner_width, content_height);
    let footer_area = if footer_height > 0 {
        Some(Rect::new(
            inner_x,
            inner_y + inner_height.saturating_sub(footer_height),
            inner_width,
            footer_height,
        ))
    } else {
        None
    };

    Some(ConnectedPromptShell {
        content_area,
        footer_area,
    })
}

fn render_pending_approval_tiny(
    frame: &mut Frame<'_>,
    permission: &PermissionView,
    area: Rect,
    theme: Theme,
    translator: &crate::tui::i18n::Translator,
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
            compact_permission_summary(permission, area.width as usize, translator),
            element_style,
        ),
    ]);
    frame.render_widget(Paragraph::new(line).style(element_style), area);
}

fn render_pending_approval_panel(
    frame: &mut Frame<'_>,
    permission: &PermissionView,
    highlighted: usize,
    area: Rect,
    theme: Theme,
    translator: &crate::tui::i18n::Translator,
) {
    let Some(shell) =
        render_connected_prompt_shell(frame, area, theme, surface::SurfaceEmphasis::Approval, 1)
    else {
        return;
    };

    let pending_style = surface::surface_style(theme, surface::SurfaceKind::Element);
    let heading = approval_heading_line(
        permission,
        theme,
        shell.content_area.width as usize,
        translator,
    );
    let summary = approval_primary_line(
        permission,
        theme,
        shell.content_area.width as usize,
        translator,
    );
    let mut lines = vec![heading, summary];
    lines.extend(approval_detail_lines(
        permission,
        theme,
        shell.content_area.width as usize,
        translator,
    ));

    frame.render_widget(
        Paragraph::new(Text::from(lines))
            .style(pending_style)
            .wrap(Wrap { trim: false }),
        shell.content_area,
    );

    if let Some(footer_area) = shell.footer_area {
        render_pending_approval_footer(
            frame,
            footer_area,
            permission,
            highlighted,
            theme,
            translator,
        );
    }
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
    let symmetric_caps = area.height >= 5;
    if symmetric_caps {
        render_child_prompt_top_cap(frame, area, theme, emphasis);
    } else {
        render_accent_bar(frame, area, bar_style);
    }

    let element_style = surface::surface_style(theme, surface::SurfaceKind::Element);
    let surface_y = area.y + u16::from(symmetric_caps);
    let surface_area = Rect::new(
        area.x + surface::ACCENT_BAR_WIDTH,
        surface_y,
        area.width.saturating_sub(surface::ACCENT_BAR_WIDTH),
        area.height
            .saturating_sub(1)
            .saturating_sub(u16::from(symmetric_caps)),
    );
    frame.render_widget(Block::new().style(element_style), surface_area);
    if symmetric_caps {
        render_accent_bar(
            frame,
            Rect::new(area.x, surface_y, area.width, surface_area.height),
            bar_style,
        );
    }

    let content_area = Rect::new(
        area.x + surface::ACCENT_BAR_WIDTH + surface::PROMPT_INNER_PAD_X,
        surface_area.y,
        area.width
            .saturating_sub(surface::ACCENT_BAR_WIDTH)
            .saturating_sub(surface::PROMPT_INNER_PAD_X)
            .saturating_sub(surface::CARD_PAD_RIGHT)
            .max(1),
        surface_area.height,
    );
    let lines = child_read_only_lines(state, theme, content_area.width as usize);
    let line_y = content_area.y + content_area.height / 2;
    let (left_area, right_area) = if content_area.width >= 48 {
        let right_width = content_area.width.min(28);
        let left_width = content_area.width.saturating_sub(right_width);
        (
            Rect::new(content_area.x, line_y, left_width, 1),
            Rect::new(content_area.x + left_width, line_y, right_width, 1),
        )
    } else {
        (
            Rect::new(content_area.x, line_y, content_area.width, 1),
            Rect::new(content_area.x, line_y, 0, 0),
        )
    };

    if let Some(top) = lines.first() {
        frame.render_widget(Paragraph::new(top.clone()).style(element_style), left_area);
    }
    if right_area.width > 0
        && let Some(bottom) = lines.get(1)
    {
        frame.render_widget(
            Paragraph::new(bottom.clone())
                .style(element_style)
                .alignment(ratatui::layout::Alignment::Right),
            right_area,
        );
    }

    render_prompt_cap(frame, area, theme, emphasis);
}

fn render_child_prompt_top_cap(
    frame: &mut Frame<'_>,
    area: Rect,
    theme: Theme,
    emphasis: surface::SurfaceEmphasis,
) {
    let cap_area = Rect::new(area.x, area.y, area.width, 1);
    frame.render_widget(
        Block::new().style(surface::surface_style(theme, surface::SurfaceKind::Root)),
        cap_area,
    );
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            surface::PROMPT_TOP_LEFT_GLYPH,
            surface::accent_style(theme, emphasis, surface::SurfaceKind::Root),
        ))),
        Rect::new(area.x, area.y, 1.min(area.width), 1),
    );

    let cap_width = area.width.saturating_sub(1);
    if cap_width > 0 && theme.paints_panels() {
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(
                surface::PROMPT_TOP_CAP_GLYPH.repeat(cap_width as usize),
                Style::default()
                    .fg(surface::surface_bg(theme, surface::SurfaceKind::Element))
                    .bg(surface::surface_bg(theme, surface::SurfaceKind::Root)),
            ))),
            Rect::new(area.x + 1, area.y, cap_width, 1),
        );
    }
}

fn child_read_only_lines(state: &TuiState, theme: Theme, _width: usize) -> Vec<Line<'static>> {
    let Some(child) = state.child_view_metadata() else {
        return vec![Line::from(Span::raw("Child transcript"))];
    };

    let mut top = vec![
        Span::styled(child.agent_name.clone(), muted_pending(theme)),
        Span::styled(" · ", muted_pending(theme)),
        Span::styled(
            format!(
                "#{} · {}/{}",
                child.pool_ordinal,
                child.index + 1,
                child.total
            ),
            inline_pending(theme),
        ),
    ];
    if let Some(model) = child.model {
        top.push(Span::styled(" · ", muted_pending(theme)));
        top.push(Span::styled(model, inline_pending(theme)));
    }

    let bottom = Line::from(vec![
        Span::styled("↑", inline_pending(theme).add_modifier(Modifier::BOLD)),
        Span::styled(" Parent", muted_pending(theme)),
        Span::styled("   ", muted_pending(theme)),
        Span::styled("←", inline_pending(theme).add_modifier(Modifier::BOLD)),
        Span::styled(" Prev", muted_pending(theme)),
        Span::styled("   ", muted_pending(theme)),
        Span::styled("→", inline_pending(theme).add_modifier(Modifier::BOLD)),
        Span::styled(" Next", muted_pending(theme)),
    ]);

    if top.is_empty() {
        vec![bottom]
    } else {
        vec![Line::from(std::mem::take(&mut top)), bottom]
    }
}

fn child_read_only_primary_text(state: &TuiState, width: usize) -> String {
    let Some(child) = state.child_view_metadata() else {
        return "Child transcript".into();
    };

    one_line_snippet(
        &format!(
            "{} {}/{} · ↑ Parent ← Prev → Next",
            child.agent_name,
            child.index + 1,
            child.total,
        ),
        width.max(1),
    )
}

fn composer_inline_lines(state: &TuiState, width: usize, theme: Theme) -> Vec<Line<'static>> {
    state.assert_composer_token_invariant();
    let width = width.max(1);
    let element_style = surface::surface_style(theme, surface::SurfaceKind::Element);
    let placeholder_style =
        surface::muted_style(theme, surface::SurfaceKind::Element).add_modifier(Modifier::ITALIC);
    if state.input_buffer.is_empty() && state.composer_tokens.is_empty() {
        return vec![Line::from(Span::styled(
            state.t("ui.message_placeholder"),
            placeholder_style,
        ))];
    }

    let mut lines: Vec<Vec<Span<'static>>> = vec![Vec::new()];
    let mut line_width = 0usize;
    let mut text = String::new();
    let mut tokens = state.composer_tokens.iter();
    let mut image_index = 0usize;
    let flush_text = |lines: &mut Vec<Vec<Span<'static>>>, text: &mut String| {
        if !text.is_empty() {
            lines
                .last_mut()
                .expect("composer line")
                .push(Span::styled(std::mem::take(text), element_style));
        }
    };

    for ch in state.input_buffer.chars() {
        if ch == crate::tui::state::COMPOSER_ATTACHMENT_MARKER {
            flush_text(&mut lines, &mut text);
            let token = tokens.next().expect("composer marker has matching token");
            let token_text = token.display_text(image_index);
            let token_width = display_width(&token_text);
            if line_width > 0 && line_width + token_width > width {
                lines.push(Vec::new());
                line_width = 0;
            }
            lines.last_mut().expect("composer line").push(Span::styled(
                token_text,
                attachment_chip_style(theme, surface::SurfaceKind::Element),
            ));
            line_width += token_width;
            if matches!(token, crate::tui::state::ComposerToken::Image(_)) {
                image_index += 1;
            }
        } else if ch == '\n' {
            flush_text(&mut lines, &mut text);
            lines.push(Vec::new());
            line_width = 0;
        } else {
            let ch_width = display_width(&ch.to_string());
            if ch_width > 0 && line_width > 0 && line_width + ch_width > width {
                flush_text(&mut lines, &mut text);
                lines.push(Vec::new());
                line_width = 0;
            }
            text.push(ch);
            line_width += ch_width;
        }
    }
    flush_text(&mut lines, &mut text);
    if line_width >= width {
        lines.push(Vec::new());
    }

    lines.into_iter().map(Line::from).collect()
}

fn attachment_chip_style(theme: Theme, kind: surface::SurfaceKind) -> Style {
    Style::default()
        .fg(theme.on_accent)
        .bg(mix_color(surface::surface_bg(theme, kind), theme.user, 70))
        .add_modifier(Modifier::BOLD)
}

fn panel_composer_cursor_area(
    metrics: ComposerMetrics,
    scroll_row: usize,
    textarea_area: Rect,
) -> Option<Rect> {
    if textarea_area.is_empty() {
        return None;
    }

    // Clamp to the visible textarea so we never set an out-of-bounds cursor.
    let max_row = textarea_area.height.saturating_sub(1) as usize;
    let row = metrics.cursor.row.saturating_sub(scroll_row).min(max_row);
    let max_col = textarea_area.width.saturating_sub(1) as usize;
    let col = metrics.cursor.column.min(max_col);

    Some(Rect::new(
        textarea_area.x + col as u16,
        textarea_area.y + row as u16,
        1,
        1,
    ))
}

fn tiny_composer_cursor_area(state: &TuiState, area: Rect) -> Option<Rect> {
    if area.width < 3 || area.height == 0 {
        return None;
    }

    // Layout is: [bar][space][content...]. Cursor starts in the content region.
    let available = area.width.saturating_sub(2) as usize;
    let cursor = composer_metrics_with_attachments(state, available).cursor;
    let desired_x = area
        .x
        .saturating_add(2)
        .saturating_add(cursor.column.min(available.saturating_sub(1)) as u16);
    let max_x = area.x.saturating_add(area.width.saturating_sub(1));
    Some(Rect::new(desired_x.min(max_x), area.y, 1, 1))
}

fn render_tiny_composer_cursor(
    frame: &mut Frame<'_>,
    state: &mut TuiState,
    area: Rect,
    theme: Theme,
) {
    if let Some(cursor_area) = tiny_composer_cursor_area(state, area) {
        render_composer_cursor_block(
            frame,
            state,
            cursor_area,
            theme,
            composer_cursor_style(state, theme),
        );
    }
}

fn render_panel_composer_cursor(
    frame: &mut Frame<'_>,
    state: &mut TuiState,
    metrics: ComposerMetrics,
    scroll_row: usize,
    textarea_area: Rect,
    theme: Theme,
) {
    if let Some(cursor_area) = panel_composer_cursor_area(metrics, scroll_row, textarea_area) {
        render_composer_cursor_block(
            frame,
            state,
            cursor_area,
            theme,
            composer_cursor_style(state, theme),
        );
    }
}

fn render_composer_cursor_block(
    frame: &mut Frame<'_>,
    _state: &mut TuiState,
    cursor_area: Rect,
    theme: Theme,
    pulse: ComposerCursorPulse,
) {
    let _ = theme;
    if let Some(cell) = frame.buffer_mut().cell_mut((cursor_area.x, cursor_area.y)) {
        cell.set_style(Style::default().bg(pulse.bg).fg(pulse.fg));
    }
    // Soft caret remains buffer-only so terminal cursor movement cannot disturb IME popups.
}

/// Soft pulse driven by the shared tick clock. Spinner only advances on Tick now,
/// so Running can breathe at the same rate as Idle without event-storm flicker.
fn composer_cursor_style(state: &TuiState, theme: Theme) -> ComposerCursorPulse {
    composer_cursor_pulse(theme, state.status_spinner_frame)
}

fn composer_cursor_pulse(theme: Theme, animation_frame: usize) -> ComposerCursorPulse {
    const LIT: f32 = 0.5;

    let intensity = cursor_pulse_intensity(animation_frame);
    let cursor_bg = composer_cursor_target_color(theme);
    if !theme.paints_panels() {
        // `Color::Reset` cannot be interpolated, so a caret without a panel can only blink.
        return if intensity >= LIT {
            ComposerCursorPulse {
                bg: cursor_bg,
                fg: mix_color_f32(theme.text, theme.canvas(), 0.82),
            }
        } else {
            ComposerCursorPulse {
                bg: Color::Reset,
                fg: theme.text,
            }
        };
    }

    ComposerCursorPulse {
        bg: mix_color_f32(theme.element_bg, cursor_bg, intensity),
        fg: mix_color_f32(theme.text, theme.element_bg, intensity * 0.82),
    }
}

fn composer_cursor_target_color(theme: Theme) -> Color {
    let target = if theme.light {
        Color::Rgb(0, 0, 0)
    } else {
        Color::Rgb(255, 255, 255)
    };
    mix_color(theme.text, target, 24)
}

fn cursor_pulse_intensity(animation_frame: usize) -> f32 {
    let cycle_frames = (CURSOR_CYCLE_DURATION_MS / CURSOR_FRAME_INTERVAL_MS).max(1);
    let phase = (animation_frame % cycle_frames) as f32 / cycle_frames as f32;

    match phase {
        phase if phase < 0.18 => 0.0,
        phase if phase < 0.62 => ease_in_out_sine((phase - 0.18) / 0.44),
        phase if phase < 0.76 => 1.0,
        _ => 1.0 - ease_in_out_sine((phase - 0.76) / 0.24),
    }
}

fn ease_in_out_sine(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn mix_color(from: Color, to: Color, mix_percent: u8) -> Color {
    mix_color_f32(from, to, f32::from(mix_percent.min(100)) / 100.0)
}

fn mix_color_f32(from: Color, to: Color, mix: f32) -> Color {
    match (from, to) {
        (Color::Rgb(fr, fg, fb), Color::Rgb(tr, tg, tb)) => Color::Rgb(
            mix_channel_f32(fr, tr, mix),
            mix_channel_f32(fg, tg, mix),
            mix_channel_f32(fb, tb, mix),
        ),
        _ => to,
    }
}

fn mix_channel_f32(from: u8, to: u8, mix: f32) -> u8 {
    let mix = mix.clamp(0.0, 1.0);
    let from = f32::from(from);
    let to = f32::from(to);
    (from + ((to - from) * mix)).round() as u8
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
    let pending_style = Style::default()
        .fg(theme.dim_text)
        .bg(theme.element_bg)
        .add_modifier(Modifier::DIM);
    let accent = surface::accent_style(
        theme,
        surface::SurfaceEmphasis::User,
        surface::SurfaceKind::Element,
    )
    .add_modifier(Modifier::BOLD);
    let dim = surface::muted_style(theme, surface::SurfaceKind::Element);

    let mode = if state.pending_permission.is_some() {
        state.t("permission.pending")
    } else {
        state.t("ui.prompt")
    };

    let (model_label, model_style) = state
        .pending_composer_settings
        .model
        .as_ref()
        .map(|(_, label)| (label.clone(), pending_style))
        .unwrap_or_else(|| (state.model_label.clone(), element_style));
    let reasoning = state
        .pending_composer_settings
        .reasoning_effort
        .as_ref()
        .map(|label| (label.clone(), pending_style))
        .or_else(|| {
            state
                .reasoning_effort_label
                .as_ref()
                .map(|label| (label.clone(), element_style))
        });
    let (permission_mode, permission_style) = state
        .pending_composer_settings
        .permission_mode
        .as_ref()
        .map(|label| (label.clone(), pending_style))
        .unwrap_or_else(|| (state.permission_mode_label.clone(), element_style));

    let mut spans = vec![
        Span::styled(mode, accent),
        Span::styled(" · ", dim),
        Span::styled(model_label, model_style),
        Span::styled(" · ", dim),
        Span::styled(state.provider_label.clone(), element_style),
    ];
    if let Some((reasoning, reasoning_style)) = reasoning {
        spans.push(Span::styled(" · ", dim));
        spans.push(Span::styled(reasoning, reasoning_style));
    }
    spans.extend([
        Span::styled(" · ", dim),
        Span::styled(permission_mode, permission_style),
    ]);
    if state.fast_mode_enabled {
        spans.push(Span::styled(" · ", dim));
        spans.push(Span::styled("fast", accent));
    }
    if let Some(client) = state.fake_client {
        let fake_style = Style::default()
            .fg(theme.fake)
            .bg(theme.element_bg)
            .add_modifier(Modifier::BOLD);
        spans.push(Span::styled(" · ", dim));
        spans.push(Span::styled(client.as_str(), fake_style));
    }

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

pub(crate) fn render_prompt_cap(
    frame: &mut Frame<'_>,
    area: Rect,
    theme: Theme,
    emphasis: surface::SurfaceEmphasis,
) {
    render_prompt_cap_with_bar(frame, area, theme, emphasis, true);
}

fn render_prompt_cap_with_bar(
    frame: &mut Frame<'_>,
    area: Rect,
    theme: Theme,
    emphasis: surface::SurfaceEmphasis,
    show_accent_bar: bool,
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

    let cap_width = if show_accent_bar {
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(
                surface::PROMPT_BOTTOM_LEFT_GLYPH,
                surface::accent_style(theme, emphasis, surface::SurfaceKind::Root),
            ))),
            Rect::new(area.x, cap_y, 1.min(area.width), 1),
        );
        area.width.saturating_sub(1)
    } else {
        area.width
    };
    if cap_width > 0 && theme.paints_panels() {
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(
                surface::PROMPT_BOTTOM_CAP_GLYPH.repeat(cap_width as usize),
                Style::default()
                    .fg(surface::surface_bg(theme, surface::SurfaceKind::Element))
                    .bg(surface::surface_bg(theme, surface::SurfaceKind::Root)),
            ))),
            Rect::new(area.x + u16::from(show_accent_bar), cap_y, cap_width, 1),
        );
    }
}

pub(crate) fn render_accent_bar(frame: &mut Frame<'_>, area: Rect, style: Style) {
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

fn render_three_sided_frame(frame: &mut Frame<'_>, area: Rect, theme: Theme) {
    if !theme.card_frame || area.width < 3 || area.height < 3 {
        return;
    }

    let style = Style::default().fg(theme.border).bg(theme.root_bg);
    let horizontal = "─".repeat(area.width.saturating_sub(2) as usize);
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(format!("┌{horizontal}┐"), style))),
        Rect::new(area.x, area.y, area.width, 1),
    );

    let side_lines = (0..area.height.saturating_sub(2))
        .map(|_| Line::from(Span::styled("│", style)))
        .collect::<Vec<_>>();
    frame.render_widget(
        Paragraph::new(Text::from(side_lines.clone())),
        Rect::new(area.x, area.y + 1, 1, area.height.saturating_sub(2)),
    );
    frame.render_widget(
        Paragraph::new(Text::from(side_lines)),
        Rect::new(
            area.right().saturating_sub(1),
            area.y + 1,
            1,
            area.height.saturating_sub(2),
        ),
    );

    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(format!("└{horizontal}┘"), style))),
        Rect::new(area.x, area.bottom().saturating_sub(1), area.width, 1),
    );
}

fn compact_permission_summary(
    permission: &PermissionView,
    width: usize,
    translator: &crate::tui::i18n::Translator,
) -> String {
    one_line_snippet(
        &approval_primary_text(permission, translator),
        width.saturating_sub(2).max(1),
    )
}

fn approval_primary_line(
    permission: &PermissionView,
    theme: Theme,
    width: usize,
    translator: &crate::tui::i18n::Translator,
) -> Line<'static> {
    let label = approval_action_label(permission, translator);
    let label_width = display_width(&label);
    let subject = one_line_snippet(
        &approval_subject(permission),
        width.saturating_sub(label_width + 3).max(1),
    );

    Line::from(vec![
        Span::styled(label, theme.approval_style()),
        Span::styled(" · ", muted_pending(theme)),
        Span::styled(subject, inline_pending(theme)),
    ])
}

fn approval_heading_line(
    permission: &PermissionView,
    theme: Theme,
    width: usize,
    translator: &crate::tui::i18n::Translator,
) -> Line<'static> {
    let title = approval_action_label(permission, translator);
    let origin = permission
        .origin_label
        .as_deref()
        .unwrap_or("needs approval");
    let detail = one_line_snippet(
        origin,
        width.saturating_sub(display_width(&title) + 4).max(1),
    );

    Line::from(vec![
        Span::styled("⚠ ", theme.approval_style().add_modifier(Modifier::BOLD)),
        Span::styled(title, theme.approval_style().add_modifier(Modifier::BOLD)),
        Span::styled("  ", muted_pending(theme)),
        Span::styled(detail, muted_pending(theme)),
    ])
}

fn approval_detail_lines(
    permission: &PermissionView,
    theme: Theme,
    width: usize,
    translator: &crate::tui::i18n::Translator,
) -> Vec<Line<'static>> {
    let mut lines = Vec::new();

    if let Some(arguments) = permission.arguments.as_deref() {
        lines.push(section_heading(&translator.t("permission.patterns"), theme));
        lines.push(section_value(one_line_snippet(arguments, width), theme));
    }

    if let Some(rationale) = permission.rationale.as_deref() {
        lines.push(section_heading(&translator.t("permission.values"), theme));
        lines.push(section_value(one_line_snippet(rationale, width), theme));
    }

    if permission.can_allow_always
        && let Some(scope) = permission.grant_summary.as_deref()
    {
        lines.push(section_heading(
            &translator.t("permission.session_scope"),
            theme,
        ));
        lines.push(section_value(one_line_snippet(scope, width), theme));
    }

    if lines.is_empty() {
        lines.push(section_heading(&translator.t("permission.summary"), theme));
        lines.push(section_value(
            one_line_snippet(&permission.summary, width),
            theme,
        ));
    }

    lines
}

fn section_heading(label: &str, theme: Theme) -> Line<'static> {
    Line::from(Span::styled(
        label.to_string(),
        muted_pending(theme).add_modifier(Modifier::BOLD),
    ))
}

fn section_value(value: String, theme: Theme) -> Line<'static> {
    Line::from(Span::styled(value, inline_pending(theme)))
}

fn render_pending_approval_footer(
    frame: &mut Frame<'_>,
    area: Rect,
    permission: &PermissionView,
    highlighted: usize,
    theme: Theme,
    translator: &crate::tui::i18n::Translator,
) {
    if area.is_empty() {
        return;
    }
    let can_allow_always = permission.can_allow_always;

    let (left_area, right_area) = if area.width >= 56 {
        let right_width = area.width.min(34);
        let left_width = area.width.saturating_sub(right_width).max(24);
        (
            Rect::new(area.x, area.y, left_width, area.height),
            Rect::new(
                area.x + left_width,
                area.y,
                area.width.saturating_sub(left_width),
                area.height,
            ),
        )
    } else {
        (area, Rect::new(area.x, area.y, 0, 0))
    };

    let choices = PermissionChoice::options(can_allow_always);
    let active = highlighted.min(choices.len().saturating_sub(1));
    let mut left_spans = Vec::new();
    for (index, choice) in choices.iter().enumerate() {
        if index > 0 {
            left_spans.push(Span::styled(" ", inline_pending(theme)));
        }
        let label = translator.t(approval_choice_label(*choice));
        let chip_style = if index == active {
            choice::highlighted_choice_style(theme, theme.approval)
        } else {
            muted_pending(theme)
        };
        left_spans.push(Span::styled(
            format!(" {} {label} ", choice::choice_marker(index == active)),
            chip_style,
        ));
    }
    // The options never truncate; the navigation hint only joins them when the
    // remaining width can hold it without spilling into the shortcut hints.
    let nav_spans = [
        Span::styled("  ", inline_pending(theme)),
        Span::styled("←→", muted_pending(theme).add_modifier(Modifier::BOLD)),
        Span::styled(
            format!(" {}  ", translator.t("ui.select")),
            muted_pending(theme),
        ),
        Span::styled("enter", muted_pending(theme).add_modifier(Modifier::BOLD)),
    ];
    if spans_width(&left_spans) + spans_width(&nav_spans) <= left_area.width as usize {
        left_spans.extend(nav_spans);
    }
    let left = Line::from(left_spans);
    frame.render_widget(Paragraph::new(left).style(inline_pending(theme)), left_area);

    if right_area.width > 0 {
        let mut hint_spans = vec![
            Span::styled("y/o", muted_pending(theme).add_modifier(Modifier::BOLD)),
            Span::styled(
                format!(" {}  ", translator.t("ui.once")),
                muted_pending(theme),
            ),
        ];
        if can_allow_always {
            hint_spans.push(Span::styled(
                "a",
                muted_pending(theme).add_modifier(Modifier::BOLD),
            ));
            hint_spans.push(Span::styled(
                format!(" {}  ", translator.t("ui.always")),
                muted_pending(theme),
            ));
        }
        hint_spans.extend([
            Span::styled("n/d", muted_pending(theme).add_modifier(Modifier::BOLD)),
            Span::styled(
                format!(" {}  ", translator.t("ui.reject")),
                muted_pending(theme),
            ),
            Span::styled("esc", muted_pending(theme).add_modifier(Modifier::BOLD)),
            Span::styled(
                format!(" {}", translator.t("ui.interrupt")),
                muted_pending(theme),
            ),
        ]);
        let hints = Line::from(hint_spans);
        frame.render_widget(
            Paragraph::new(hints)
                .style(inline_pending(theme))
                .alignment(ratatui::layout::Alignment::Right),
            right_area,
        );
    }
}

fn spans_width(spans: &[Span<'_>]) -> usize {
    spans
        .iter()
        .map(|span| display_width(span.content.as_ref()))
        .sum()
}

fn approval_choice_label(choice: PermissionChoice) -> &'static str {
    match choice {
        PermissionChoice::AllowOnce => "permission.allow_once",
        PermissionChoice::AllowAlways => "permission.allow_always",
        PermissionChoice::Reject => "permission.reject",
    }
}

fn approval_primary_text(
    permission: &PermissionView,
    translator: &crate::tui::i18n::Translator,
) -> String {
    format!(
        "{} · {}",
        approval_action_label(permission, translator),
        approval_subject(permission)
    )
}

fn approval_action_label(
    permission: &PermissionView,
    translator: &crate::tui::i18n::Translator,
) -> String {
    let key = match permission.tool_name.as_str() {
        "shell__exec" => "permission.run_command",
        "fs__read" => "permission.read_file",
        "fs__write" => "permission.write_file",
        "fs__append" => "permission.append_file",
        "fs__mkdir" => "permission.create_directory",
        "search__rg" => "permission.search_text",
        "web__fetch" => "permission.fetch_url",
        "code__ast_search" => "permission.search_code",
        "edit__apply_patch" => "permission.apply_patch",
        _ => "permission.approve_tool",
    };
    translator.t(key)
}

fn approval_subject(permission: &PermissionView) -> String {
    let subject = match permission.tool_name.as_str() {
        "shell__exec" => extract_json_argument(permission, &["command"]),
        "fs__read" | "fs__write" | "fs__append" | "fs__mkdir" => {
            extract_json_argument(permission, &["path", "filePath"])
        }
        "search__rg" => extract_json_argument(permission, &["pattern"]),
        "web__fetch" => extract_json_argument(permission, &["url"]),
        "code__ast_search" => extract_json_argument(permission, &["pattern", "query"]),
        _ => None,
    }
    .unwrap_or_else(|| permission.summary.clone());

    if let Some(origin) = permission.origin_label.as_deref() {
        format!("{origin} · {subject}")
    } else {
        subject
    }
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
    if (wrapped.len() > 1 || display_width(&compact) > width.max(1)) && !first.ends_with('…') {
        first.push('…');
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
#[path = "../../../tests/internal/tui/components/composer/tests.rs"]
mod tests;
