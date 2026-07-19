use ratatui::{
    Frame,
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Clear, Paragraph, Wrap},
};
use serde_json::Value;

use super::super::state::TuiState;
use crate::tui::{
    measure::{
        CursorVisualPosition, cursor_visual_position, display_width, wrap_text_to_width,
        wrapped_row_count,
    },
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

pub fn composer_cursor_position(
    input: &str,
    width: usize,
    cursor_byte_index: usize,
) -> ComposerCursor {
    cursor_visual_position(input, width, cursor_byte_index).into()
}

pub fn composer_metrics(input: &str, width: usize, cursor_byte_index: usize) -> ComposerMetrics {
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
    state.assert_composer_attachment_invariant();
    let width = width.max(1);
    let mut row = 0usize;
    let mut col = 0usize;
    let mut cursor = None;
    let mut attachment_index = 0usize;
    let mut byte_index = 0usize;
    let mut ended_by_exact_fill = false;

    for ch in state.input_buffer.chars() {
        if byte_index == state.input_cursor && cursor.is_none() {
            cursor = Some(ComposerCursor { row, column: col });
        }

        if ch == crate::tui::state::COMPOSER_ATTACHMENT_MARKER {
            let token_width = display_width(&composer_attachment_token(attachment_index));
            if col > 0 && col + token_width > width {
                row = row.saturating_add(1);
                col = 0;
            }
            col += token_width;
            attachment_index += 1;
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

fn render_composer_tiny(frame: &mut Frame<'_>, state: &TuiState, area: Rect, theme: Theme) {
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
        .unwrap_or_else(|| Line::from(Span::styled("message…", element_style)));
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
        u16::try_from(composer_textarea_width(area.width))
            .unwrap_or(u16::MAX)
            .max(1),
        area.height
            .saturating_sub(1)
            .saturating_sub(surface::PROMPT_INNER_PAD_TOP)
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
}

pub(crate) fn render_connected_prompt_shell(
    frame: &mut Frame<'_>,
    area: Rect,
    theme: Theme,
    emphasis: surface::SurfaceEmphasis,
    footer_height: u16,
) -> Option<ConnectedPromptShell> {
    if area.is_empty() {
        return None;
    }

    let bar_style = surface::accent_style(theme, emphasis, surface::SurfaceKind::Root)
        .add_modifier(Modifier::BOLD);
    render_accent_bar(frame, area, bar_style);

    let panel_style = surface::surface_style(theme, surface::SurfaceKind::Element);
    let surface_area = Rect::new(
        area.x + surface::ACCENT_BAR_WIDTH,
        area.y,
        area.width.saturating_sub(surface::ACCENT_BAR_WIDTH),
        area.height.saturating_sub(1),
    );
    frame.render_widget(Block::new().style(panel_style), surface_area);

    render_prompt_cap(frame, area, theme, emphasis);

    let inner_x = area.x + surface::ACCENT_BAR_WIDTH + surface::PROMPT_INNER_PAD_X;
    let inner_y = area.y + surface::PROMPT_INNER_PAD_TOP;
    let inner_width = area
        .width
        .saturating_sub(surface::ACCENT_BAR_WIDTH)
        .saturating_sub(surface::PROMPT_INNER_PAD_X)
        .saturating_sub(surface::CARD_PAD_RIGHT)
        .max(1);
    let inner_height = area
        .height
        .saturating_sub(1)
        .saturating_sub(surface::PROMPT_INNER_PAD_TOP)
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
    let Some(shell) =
        render_connected_prompt_shell(frame, area, theme, surface::SurfaceEmphasis::Approval, 1)
    else {
        return;
    };

    let pending_style = surface::surface_style(theme, surface::SurfaceKind::Element);
    let heading = approval_heading_line(permission, theme, shell.content_area.width as usize);
    let summary = approval_primary_line(permission, theme, shell.content_area.width as usize);
    let mut lines = vec![heading, summary];
    lines.extend(approval_detail_lines(
        permission,
        theme,
        shell.content_area.width as usize,
    ));

    frame.render_widget(
        Paragraph::new(Text::from(lines))
            .style(pending_style)
            .wrap(Wrap { trim: false }),
        shell.content_area,
    );

    if let Some(footer_area) = shell.footer_area {
        render_pending_approval_footer(frame, footer_area, permission.can_allow_always, theme);
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
        1,
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

fn child_read_only_lines(state: &TuiState, theme: Theme, _width: usize) -> Vec<Line<'static>> {
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
    state.assert_composer_attachment_invariant();
    let width = width.max(1);
    let element_style = surface::surface_style(theme, surface::SurfaceKind::Element);
    let placeholder_style =
        surface::muted_style(theme, surface::SurfaceKind::Element).add_modifier(Modifier::ITALIC);
    if state.input_buffer.is_empty() && state.composer_attachments.is_empty() {
        return vec![Line::from(Span::styled(
            "message letcode…",
            placeholder_style,
        ))];
    }

    let mut lines: Vec<Vec<Span<'static>>> = vec![Vec::new()];
    let mut line_width = 0usize;
    let mut text = String::new();
    let mut attachment_index = 0usize;
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
            let token = composer_attachment_token(attachment_index);
            let token_width = display_width(&token);
            if line_width > 0 && line_width + token_width > width {
                lines.push(Vec::new());
                line_width = 0;
            }
            lines.last_mut().expect("composer line").push(Span::styled(
                token,
                attachment_chip_style(theme, surface::SurfaceKind::Element),
            ));
            line_width += token_width;
            attachment_index += 1;
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

fn composer_attachment_token(index: usize) -> String {
    format!("[Image {}]", index + 1)
}

fn attachment_chip_style(theme: Theme, kind: surface::SurfaceKind) -> Style {
    Style::default()
        .fg(theme.root_bg)
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

fn render_tiny_composer_cursor(frame: &mut Frame<'_>, state: &TuiState, area: Rect, theme: Theme) {
    if let Some(cursor_area) = tiny_composer_cursor_area(state, area) {
        render_composer_cursor_block(frame, cursor_area, theme, state.status_spinner_frame);
    }
}

fn render_panel_composer_cursor(
    frame: &mut Frame<'_>,
    state: &TuiState,
    metrics: ComposerMetrics,
    scroll_row: usize,
    textarea_area: Rect,
    theme: Theme,
) {
    if let Some(cursor_area) = panel_composer_cursor_area(metrics, scroll_row, textarea_area) {
        render_composer_cursor_block(frame, cursor_area, theme, state.status_spinner_frame);
    }
}

fn render_composer_cursor_block(
    frame: &mut Frame<'_>,
    cursor_area: Rect,
    theme: Theme,
    animation_frame: usize,
) {
    let pulse = composer_cursor_pulse(theme, animation_frame);
    if let Some(cell) = frame.buffer_mut().cell_mut((cursor_area.x, cursor_area.y)) {
        cell.set_style(Style::default().bg(pulse.bg).fg(pulse.fg));
    }
}

fn composer_cursor_pulse(theme: Theme, animation_frame: usize) -> ComposerCursorPulse {
    let intensity = cursor_pulse_intensity(animation_frame);
    let cursor_bg = composer_cursor_target_color(theme);
    ComposerCursorPulse {
        bg: mix_color_f32(theme.element_bg, cursor_bg, intensity),
        fg: mix_color_f32(theme.text, theme.element_bg, intensity * 0.82),
    }
}

fn composer_cursor_target_color(theme: Theme) -> Color {
    mix_color(theme.text, Color::Rgb(255, 255, 255), 24)
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

fn mix_channel(from: u8, to: u8, mix_percent: u8) -> u8 {
    mix_channel_f32(from, to, f32::from(mix_percent.min(100)) / 100.0)
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

pub(crate) fn render_prompt_cap(
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

fn approval_heading_line(permission: &PermissionView, theme: Theme, width: usize) -> Line<'static> {
    let title = approval_action_label(permission);
    let origin = permission
        .origin_label
        .as_deref()
        .unwrap_or("needs approval");
    let detail = one_line_snippet(
        origin,
        width.saturating_sub(display_width(title) + 4).max(1),
    );

    Line::from(vec![
        Span::styled("⚠ ", theme.approval_style().add_modifier(Modifier::BOLD)),
        Span::styled(
            title.to_string(),
            theme.approval_style().add_modifier(Modifier::BOLD),
        ),
        Span::styled("  ", muted_pending(theme)),
        Span::styled(detail, muted_pending(theme)),
    ])
}

fn approval_detail_lines(
    permission: &PermissionView,
    theme: Theme,
    width: usize,
) -> Vec<Line<'static>> {
    let mut lines = Vec::new();

    if let Some(arguments) = permission.arguments.as_deref() {
        lines.push(section_heading("Patterns", theme));
        lines.push(section_value(one_line_snippet(arguments, width), theme));
    }

    if let Some(rationale) = permission.rationale.as_deref() {
        lines.push(section_heading("Values", theme));
        lines.push(section_value(one_line_snippet(rationale, width), theme));
    }

    if permission.can_allow_always
        && let Some(scope) = permission.grant_summary.as_deref()
    {
        lines.push(section_heading("Session scope", theme));
        lines.push(section_value(one_line_snippet(scope, width), theme));
    }

    if lines.is_empty() {
        lines.push(section_heading("Summary", theme));
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
    can_allow_always: bool,
    theme: Theme,
) {
    if area.is_empty() {
        return;
    }

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

    let selected = Style::default()
        .fg(theme.root_bg)
        .bg(theme.approval)
        .add_modifier(Modifier::BOLD);
    let chip = |label: &str, active: bool| {
        if active {
            Span::styled(format!(" {label} "), selected)
        } else {
            Span::styled(format!(" {label} "), muted_pending(theme))
        }
    };

    let mut left_spans = vec![chip("Allow once", true)];
    if can_allow_always {
        left_spans.push(Span::styled(" ", inline_pending(theme)));
        left_spans.push(chip("Allow always", false));
    }
    left_spans.push(Span::styled(" ", inline_pending(theme)));
    left_spans.push(chip("Reject", false));
    let left = Line::from(left_spans);
    frame.render_widget(Paragraph::new(left).style(inline_pending(theme)), left_area);

    if right_area.width > 0 {
        let mut hint_spans = vec![
            Span::styled("y/o", muted_pending(theme).add_modifier(Modifier::BOLD)),
            Span::styled(" once  ", muted_pending(theme)),
        ];
        if can_allow_always {
            hint_spans.push(Span::styled(
                "a",
                muted_pending(theme).add_modifier(Modifier::BOLD),
            ));
            hint_spans.push(Span::styled(" always  ", muted_pending(theme)));
        }
        hint_spans.extend([
            Span::styled("n/d", muted_pending(theme).add_modifier(Modifier::BOLD)),
            Span::styled(" reject  ", muted_pending(theme)),
            Span::styled("esc", muted_pending(theme).add_modifier(Modifier::BOLD)),
            Span::styled(" interrupt", muted_pending(theme)),
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
    let subject = match permission.tool_name.as_str() {
        "shell__exec" => extract_json_argument(permission, &["command"]),
        "fs__read" | "fs__write" | "fs__append" | "fs__mkdir" => {
            extract_json_argument(permission, &["path", "filePath"])
        }
        "search__rg" => extract_json_argument(permission, &["pattern"]),
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
    use crate::user_content::UserImageAttachment;
    use ratatui::style::Color;
    use ratatui::{Terminal, backend::TestBackend};

    fn test_attachment(id: &str, label: &str) -> UserImageAttachment {
        UserImageAttachment {
            id: id.into(),
            label: label.into(),
            mime: "image/png".into(),
            data_url: "data:image/png;base64,AAAA".into(),
        }
    }

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

    fn draw_rows(state: &TuiState, width: u16, height: u16) -> Vec<String> {
        let backend = TestBackend::new(width, height);
        let mut terminal = Terminal::new(backend).expect("terminal");

        terminal
            .draw(|frame| {
                let area = Rect::new(0, 0, width, height);
                render_composer(frame, state, area, Theme::dark());
            })
            .expect("draw");

        let buffer = terminal.backend().buffer();
        (0..height)
            .map(|y| {
                (0..width)
                    .filter_map(|x| buffer.cell((x, y)).map(|cell| cell.symbol()))
                    .collect::<String>()
            })
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

    fn composer_cell_style(
        state: &TuiState,
        width: u16,
        height: u16,
        x: u16,
        y: u16,
    ) -> Option<(String, Color, Color)> {
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
            .cell((x, y))
            .map(|cell| (cell.symbol().to_string(), cell.fg, cell.bg))
    }

    #[test]
    fn composer_cursor_never_returns_column_equal_to_width() {
        let width = 4;
        let cursor = composer_cursor_position("abcd", width, 4);
        assert!(cursor.column < width, "{cursor:?}");

        let cursor = composer_cursor_position("你", 2, "你".len());
        assert!(cursor.column < 2, "{cursor:?}");
    }

    #[test]
    fn composer_metrics_account_for_wrapping_and_trailing_newline() {
        // width 2: "你你a" wraps as ["你", "你", "a"] => 3 rows.
        let metrics = composer_metrics("你你a", 2, "你你a".len());
        assert_eq!(metrics.row_count, 3);
        assert_eq!(metrics.cursor, ComposerCursor { row: 2, column: 1 });

        // Trailing newline puts the cursor on the next empty row.
        let metrics = composer_metrics("hi\n", 10, 3);
        assert_eq!(metrics.cursor, ComposerCursor { row: 1, column: 0 });
    }

    #[test]
    fn composer_metrics_row_count_always_includes_cursor_row() {
        // Exact-fill should advance cursor row; row_count must still include it.
        let metrics = composer_metrics("abcd", 4, 4);
        assert_eq!(metrics.cursor, ComposerCursor { row: 1, column: 0 });
        assert_eq!(metrics.row_count, 2);
    }

    #[test]
    fn long_single_line_cjk_wraps_into_multiple_rows() {
        // Each CJK char is width 2.
        // width 6 => 3 chars per row.
        let input = "你好世界你好"; // 6 chars => 2 wrapped rows.
        let metrics = composer_metrics(input, 6, input.len());
        // Exact-fill on the final row advances the cursor to a new empty visual row.
        assert_eq!(metrics.row_count, 3);
        assert_eq!(metrics.cursor, ComposerCursor { row: 2, column: 0 });

        // Mixed widths.
        let mixed = "ab你cd你ef"; // widths: 1+1+2+1+1+2+1+1 = 10
        let metrics = composer_metrics(mixed, 4, mixed.len());
        assert!(metrics.row_count >= 3, "{metrics:?}");
    }

    #[test]
    fn composer_cursor_tracks_arbitrary_input_cursor() {
        let input = "ab\ncd";
        let metrics = composer_metrics(input, 10, 2);
        assert_eq!(metrics.cursor, ComposerCursor { row: 0, column: 2 });

        let metrics = composer_metrics(input, 10, 3);
        assert_eq!(metrics.cursor, ComposerCursor { row: 1, column: 0 });
    }

    #[test]
    fn composer_renders_input_newlines_as_visual_rows() {
        let mut state = TuiState::default();
        state.set_input("first\nsecond");

        let rows = draw_rows(&state, 80, 8);

        assert!(rows[1].contains("first"), "{rows:?}");
        assert!(rows[2].contains("second"), "{rows:?}");
        assert!(!rows[1].contains("firstsecond"), "{rows:?}");
    }

    #[test]
    fn composer_inline_lines_wrap_wide_char_before_overflow() {
        let mut state = TuiState::default();
        state.set_input("abc你d");

        let lines = composer_inline_lines(&state, 4, Theme::dark())
            .into_iter()
            .map(|line| {
                line.spans
                    .into_iter()
                    .map(|span| span.content.into_owned())
                    .collect::<String>()
            })
            .collect::<Vec<_>>();

        assert_eq!(lines, vec!["abc", "你d"]);
    }

    #[test]
    fn composer_scrolls_to_keep_cursor_visible() {
        let metrics = ComposerMetrics {
            row_count: 8,
            cursor: ComposerCursor { row: 7, column: 0 },
        };

        assert_eq!(composer_scroll_row(metrics, 3), 5);
        assert_eq!(composer_scroll_row(metrics, 10), 0);
    }

    #[test]
    fn composer_height_uses_wrapped_rows_instead_of_logical_lines() {
        // Single line that wraps into multiple visual rows at narrow widths.
        let input = "你你你你你你"; // 6 chars => width 12
        let height_narrow = crate::tui::components::layout::composer_height(30, input, &[], 12);
        let height_wide = crate::tui::components::layout::composer_height(30, input, &[], 80);
        assert!(
            height_narrow >= height_wide,
            "{height_narrow} vs {height_wide}"
        );
    }

    #[test]
    fn composer_renders_attachments_at_their_inline_positions() {
        let mut state = TuiState::default();
        state.set_input("before after");
        state.input_cursor = "before ".len();
        state.add_composer_attachment(test_attachment("img-1", "clipboard"));
        state.input_cursor = state.input_buffer.len();
        state.add_composer_attachment(test_attachment("img-2", "diagram.png"));

        let rows = draw_rows(&state, 80, 10);

        assert!(
            rows.iter()
                .any(|row| row.contains("before [Image 1]after[Image 2]")),
            "{rows:?}"
        );
    }

    #[test]
    fn composer_inline_tokens_wrap_at_their_logical_positions() {
        let mut state = TuiState::default();
        state.set_input("abx");
        state.input_cursor = 2;
        state.add_composer_attachment(test_attachment("img-1", "clipboard"));

        let lines = composer_inline_lines(&state, 10, Theme::dark())
            .into_iter()
            .map(|line| {
                line.spans
                    .into_iter()
                    .map(|span| span.content.into_owned())
                    .collect::<String>()
            })
            .collect::<Vec<_>>();

        assert_eq!(lines, vec!["ab", "[Image 1]x", ""]);
        assert_eq!(
            composer_metrics_with_attachments(&state, 10),
            ComposerMetrics {
                row_count: 3,
                cursor: ComposerCursor { row: 1, column: 9 },
            }
        );
    }

    #[test]
    fn exact_width_text_allocates_a_cursor_row() {
        let mut state = TuiState::default();
        state.set_input("abcd");

        assert_eq!(
            composer_inline_lines(&state, 4, Theme::dark()).len(),
            2,
            "exact-width text reserves a cursor row"
        );
        assert_eq!(
            composer_metrics_with_attachments(&state, 4),
            ComposerMetrics {
                row_count: 2,
                cursor: ComposerCursor { row: 1, column: 0 },
            }
        );
    }

    #[test]
    fn narrow_attachment_tokens_reserve_the_same_rows_as_the_renderer() {
        let attachment = test_attachment("img-1", "clipboard");
        let mut state = TuiState::default();
        state.set_input("ab");
        state.input_cursor = 2;
        state.add_composer_attachment(attachment.clone());

        assert_eq!(composer_inline_lines(&state, 5, Theme::dark()).len(), 3);
        assert_eq!(composer_metrics_with_attachments(&state, 5).row_count, 3);
    }

    #[test]
    fn composer_height_reserves_rows_for_attachments() {
        let base = crate::tui::components::layout::composer_height(30, "hello", &[], 80);
        let with_attachment = crate::tui::components::layout::composer_height(
            30,
            "hello\u{fffc}",
            &[test_attachment("img-1", "clipboard")],
            80,
        );

        assert!(with_attachment >= base, "{with_attachment} vs {base}");
    }

    #[test]
    fn composer_metrics_wraps_new_attachment_after_existing_input() {
        let mut state = TuiState::default();
        state.set_input("hello");
        state.add_composer_attachment(test_attachment("img-1", "clipboard"));
        state.input_cursor = "hello".len();

        let metrics = composer_metrics_with_attachments(&state, 14);

        assert_eq!(metrics.cursor, ComposerCursor { row: 0, column: 5 });
        assert_eq!(metrics.row_count, 2);
    }

    #[test]
    fn composer_metrics_keep_cursor_at_the_inline_attachment_boundary() {
        let mut state = TuiState::default();
        state.set_input("测试");
        state.add_composer_attachment(test_attachment("img-1", "clipboard"));
        state.input_cursor = "测试".len();

        let metrics = composer_metrics_with_attachments(&state, 80);
        assert_eq!(metrics.cursor.row, 0);
        assert_eq!(metrics.cursor.column, display_width("测试"));
    }

    #[test]
    fn composer_metrics_wrap_attachment_tokens_atomically() {
        let mut state = TuiState::default();
        state.set_input("ab");
        state.add_composer_attachment(test_attachment("img-1", "clipboard"));
        state.add_composer_attachment(test_attachment("img-2", "clipboard"));
        state.input_cursor = "ab".len();

        let metrics = composer_metrics_with_attachments(&state, 14);
        assert!(metrics.row_count >= 2, "{metrics:?}");
        assert_eq!(metrics.cursor, ComposerCursor { row: 0, column: 2 });
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
        assert!(
            rendered.contains("Allow once") || rendered.contains("allow once"),
            "{rendered}"
        );
        assert!(
            rendered.contains("Reject") || rendered.contains("reject"),
            "{rendered}"
        );
        assert!(!rendered.contains("message letcode"), "{rendered}");
        assert!(!rendered.contains("args"), "{rendered}");
    }

    #[test]
    fn pending_permission_footer_uses_connected_action_row() {
        let mut state = TuiState::default();
        let mut request =
            PermissionRequestEvent::new("call-1", "shell__exec", "cargo test --workspace");
        request.arguments = Some(r#"{"command":"cargo test --workspace"}"#.into());
        request.rationale = Some("project tests require approval".into());
        state.apply_event(AppEvent::PermissionRequested(request));

        let rendered = draw_to_string(&state, 100, 10);

        assert!(rendered.contains("Patterns"), "{rendered}");
        assert!(rendered.contains("Values"), "{rendered}");
        assert!(rendered.contains("Allow once"), "{rendered}");
        assert!(rendered.contains("Reject"), "{rendered}");
        assert!(!rendered.contains("Allow always"), "{rendered}");
        assert!(rendered.contains("y/o"), "{rendered}");
        assert!(rendered.contains("n/d"), "{rendered}");
        assert!(rendered.contains("esc"), "{rendered}");
        assert!(!rendered.contains("fullscreen"), "{rendered}");
    }

    #[test]
    fn pending_default_permission_shows_session_allowance() {
        let mut state = TuiState::default();
        let mut request = PermissionRequestEvent::new("call-1", "fs__write", "write src/lib.rs");
        request.can_allow_always = true;
        state.apply_event(AppEvent::PermissionRequested(request));

        let rendered = draw_to_string(&state, 100, 10);

        assert!(rendered.contains("Allow once"), "{rendered}");
        assert!(rendered.contains("Allow always"), "{rendered}");
        assert!(rendered.contains("y/o"), "{rendered}");
        assert!(rendered.contains("a"), "{rendered}");
        assert!(rendered.contains("n/d"), "{rendered}");
    }

    #[test]
    fn pending_permission_shows_subagent_origin_when_present() {
        let mut state = TuiState::default();
        let mut request =
            PermissionRequestEvent::new("call-1", "shell__exec", "cargo test --workspace");
        request.arguments = Some(r#"{"command":"cargo test --workspace"}"#.into());
        request.origin_label = Some("fixer".into());
        state.apply_event(AppEvent::PermissionRequested(request));

        let rendered = draw_to_string(&state, 80, 8);

        assert!(rendered.contains("fixer"), "{rendered}");
        assert!(rendered.contains("cargo test --workspace"), "{rendered}");
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
    fn composer_cursor_pulse_uses_slow_breathing_cycle() {
        let cycle_frames = CURSOR_CYCLE_DURATION_MS / CURSOR_FRAME_INTERVAL_MS;

        assert_eq!(cursor_pulse_intensity(0), 0.0);
        assert_eq!(cursor_pulse_intensity(cycle_frames / 8), 0.0);
        assert!(cursor_pulse_intensity(cycle_frames * 3 / 5) > 0.9);
        assert!(cursor_pulse_intensity(cycle_frames * 3 / 4) > 0.9);
        assert!(cursor_pulse_intensity(cycle_frames - 1) < 0.1);
    }

    #[test]
    fn composer_cursor_pulse_targets_white_instead_of_user_accent() {
        let theme = Theme::dark();
        let target = composer_cursor_target_color(theme);

        assert!(color_distance(target, theme.text) < color_distance(target, theme.user));
        assert!(
            color_distance(target, Color::Rgb(255, 255, 255)) < color_distance(target, theme.user)
        );
    }

    #[test]
    fn composer_draws_custom_cursor_block_without_using_terminal_cursor() {
        let mut state = TuiState::default();
        state.set_input("hello");
        state.input_cursor = 2;
        state.status_spinner_frame = (CURSOR_CYCLE_DURATION_MS / CURSOR_FRAME_INTERVAL_MS) * 3 / 5;

        let (symbol, fg, bg) = composer_cell_style(&state, 80, 8, 5, 1).expect("cursor cell");

        assert_eq!(symbol, "l");
        assert_ne!(bg, Theme::dark().element_bg);
        assert_ne!(fg, Theme::dark().element_bg);
    }

    fn color_distance(a: Color, b: Color) -> u32 {
        match (a, b) {
            (Color::Rgb(ar, ag, ab), Color::Rgb(br, bg, bb)) => {
                u32::from(ar.abs_diff(br)) + u32::from(ag.abs_diff(bg)) + u32::from(ab.abs_diff(bb))
            }
            _ => u32::MAX,
        }
    }

    #[test]
    fn child_view_renders_read_only_status_bar_instead_of_input_composer() {
        let mut state = TuiState::new("gpt-5.5", "GPT-5.5", "default");
        state.set_provider_label("CLI Proxy API");
        state.replace_child_timeline_from_records(
            &[crate::transcript::TranscriptRecord {
                session_id: "child-session".into(),
                sequence: 1,
                timestamp_ms: 0,
                context_branch_id: None,
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

        assert!(rendered.contains("fixer"), "{rendered}");
        assert!(rendered.contains("2/3"), "{rendered}");
        assert!(rendered.contains("gpt-5.5-mini"), "{rendered}");
        assert!(rendered.contains("Parent"), "{rendered}");
        assert!(rendered.contains("Prev"), "{rendered}");
        assert!(rendered.contains("Next"), "{rendered}");
        assert!(!rendered.contains("Read-only child view"), "{rendered}");
        assert!(!rendered.contains("child-session-1234567890"), "{rendered}");
        assert!(!rendered.contains("model gpt-5.5-mini"), "{rendered}");
        assert!(!rendered.contains("records"), "{rendered}");
        assert!(!rendered.contains("parent-session"), "{rendered}");
        assert!(!rendered.contains("message letcode"), "{rendered}");
        assert!(
            !rendered.contains("hidden input should not render"),
            "{rendered}"
        );
    }

    #[test]
    fn child_read_only_panel_uses_symmetric_capped_composer_rows() {
        let mut state = TuiState::new("gpt-5.5", "GPT-5.5", "default");
        state.replace_child_timeline_from_records(
            &[crate::transcript::TranscriptRecord {
                session_id: "child-session".into(),
                sequence: 1,
                timestamp_ms: 0,
                context_branch_id: None,
                event: crate::transcript::TranscriptEvent::SessionStarted {
                    model: "gpt-5.5-mini".into(),
                },
            }],
            "parent-session",
            "child-session-1234567890",
            "explorer",
            0,
            1,
        );

        let rows = draw_rows(&state, 100, 4);

        assert!(!rows[0].contains("explorer"), "{rows:?}");
        assert!(rows[1].contains("explorer"), "{rows:?}");
        assert!(rows[1].contains("gpt-5.5-mini"), "{rows:?}");
        assert!(rows[1].contains("Parent"), "{rows:?}");
        assert!(!rows[2].contains("explorer"), "{rows:?}");
        assert!(
            rows[3].contains(surface::PROMPT_BOTTOM_LEFT_GLYPH)
                || rows[3].contains(surface::PROMPT_BOTTOM_CAP_GLYPH),
            "{rows:?}"
        );
    }

    #[test]
    fn child_view_with_command_input_renders_composer_instead_of_read_only_panel() {
        let mut state = TuiState::new("gpt-5.5", "GPT-5.5", "default");
        state.replace_child_timeline_from_records(
            &[],
            "parent-session",
            "child-session",
            "explorer",
            0,
            1,
        );
        state.set_input("/tool-output".to_string());

        let rendered = draw_to_string(&state, 100, 8);

        assert!(rendered.contains("/tool-output"), "{rendered}");
        assert!(!rendered.contains("Read-only child view"), "{rendered}");
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
