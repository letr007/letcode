use ratatui::{
    Frame,
    layout::{Constraint, Layout, Rect},
    style::Style,
    text::{Line, Span, Text},
    widgets::{Block, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState},
};

use crate::tui::{
    markdown::{MarkdownRenderOptions, render_markdown},
    measure::{display_width, wrap_text_to_width},
    surface,
    theme::Theme,
    timeline::{
        ErrorView, MessageView, NoticeView, PermissionPromptStatus, PermissionView, ReasoningView,
        TimelineItem, ToolView,
    },
};

use super::super::state::TuiState;
use super::{todo_card, tool_card};

#[derive(Debug, Clone, Default)]
pub struct TranscriptRenderCache {
    width: Option<usize>,
    theme: Option<Theme>,
    timeline_cache_id: Option<u64>,
    entries: Vec<TranscriptRenderCacheEntry>,
    row_starts: Vec<usize>,
    row_counts: Vec<usize>,
}

impl TranscriptRenderCache {
    pub fn clear(&mut self) {
        self.width = None;
        self.theme = None;
        self.timeline_cache_id = None;
        self.entries.clear();
        self.row_starts.clear();
        self.row_counts.clear();
    }

    fn prepare(&mut self, width: usize, theme: Theme, timeline_cache_id: u64) {
        if self.width != Some(width)
            || self.theme != Some(theme)
            || self.timeline_cache_id != Some(timeline_cache_id)
        {
            self.width = Some(width);
            self.theme = Some(theme);
            self.timeline_cache_id = Some(timeline_cache_id);
            self.entries.clear();
            self.row_starts.clear();
            self.row_counts.clear();
        }
    }
}

impl PartialEq for TranscriptRenderCache {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl Eq for TranscriptRenderCache {}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TranscriptRenderCacheEntry {
    revision: Option<u64>,
    lines: Vec<Line<'static>>,
}

pub fn render_transcript(frame: &mut Frame<'_>, state: &mut TuiState, area: Rect, theme: Theme) {
    if area.is_empty() {
        return;
    }

    if state.timeline.items().is_empty() {
        // Welcome rendering is handled at a higher level.
        frame.render_widget(Block::new().style(theme.app_style()), area);
        return;
    }

    let has_scrollbar = area.width >= 24;
    let (content_area, scrollbar_area) = if has_scrollbar {
        let chunks = Layout::horizontal([Constraint::Min(1), Constraint::Length(1)]).split(area);
        (chunks[0], Some(chunks[1]))
    } else {
        (area, None)
    };

    let width = content_area.width.max(1) as usize;
    let total_rows = cached_transcript_row_count(state, theme, width);
    state.sync_transcript_viewport_rows(total_rows);
    let visible_rows = content_area.height;
    let scroll = crate::tui::measure::resolved_scroll_offset(
        total_rows,
        visible_rows,
        state.transcript_scroll,
        state.auto_scroll,
    );

    let visible_lines = visible_cached_transcript_lines(state, theme, width, visible_rows, scroll);
    let paragraph = Paragraph::new(Text::from(visible_lines)).style(theme.app_style());

    frame.render_widget(paragraph, content_area);

    if let Some(scrollbar_area) = scrollbar_area
        && total_rows > visible_rows as usize
        && visible_rows > 0
    {
        let mut scrollbar_state = ScrollbarState::new(total_rows)
            .position(scroll as usize)
            .viewport_content_length(visible_rows as usize);
        let scrollbar = Scrollbar::new(ScrollbarOrientation::VerticalRight)
            .begin_symbol(None)
            .end_symbol(None)
            .thumb_style(Style::default().fg(theme.dim_text).bg(theme.root_bg))
            .track_style(Style::default().fg(theme.element_bg).bg(theme.root_bg));
        frame.render_stateful_widget(scrollbar, scrollbar_area, &mut scrollbar_state);
    }
}

fn visible_transcript_lines(
    lines: &[Line<'static>],
    visible_rows: u16,
    top_scroll: u16,
) -> Vec<Line<'static>> {
    let visible_rows = visible_rows as usize;
    if visible_rows == 0 {
        return Vec::new();
    }

    let start = (top_scroll as usize).min(lines.len());
    let end = start.saturating_add(visible_rows).min(lines.len());
    lines[start..end].to_vec()
}

#[cfg(test)]
pub fn transcript_row_count(state: &TuiState, theme: Theme, width: usize) -> usize {
    transcript_lines(state, theme, width).len()
}

pub fn transcript_lines(state: &TuiState, theme: Theme, width: usize) -> Vec<Line<'static>> {
    let mut lines = Vec::new();

    if !state.timeline.items().is_empty() {
        lines.extend((0..surface::TRANSCRIPT_TOP_SPACER).map(|_| Line::from("")));
    }

    for (index, item) in state.timeline.items().iter().enumerate() {
        if index > 0 {
            lines.push(Line::from(""));
        }

        lines.extend(render_timeline_item_lines(item, theme, width));
    }

    lines
}

fn cached_transcript_row_count(state: &mut TuiState, theme: Theme, width: usize) -> usize {
    let item_count = state.timeline.items().len();
    if item_count == 0 {
        return 0;
    }

    state
        .transcript_render_cache
        .prepare(width, theme, state.timeline.cache_id());
    state
        .transcript_render_cache
        .entries
        .resize_with(item_count, || TranscriptRenderCacheEntry {
            revision: None,
            lines: Vec::new(),
        });

    let mut rows = surface::TRANSCRIPT_TOP_SPACER;
    state.transcript_render_cache.row_starts.clear();
    state.transcript_render_cache.row_counts.clear();

    for index in 0..item_count {
        rows = rows.saturating_add(if index > 0 { 1 } else { 0 });
        state.transcript_render_cache.row_starts.push(rows);
        let line_count = cached_item_lines(state, index, theme, width).len();
        state.transcript_render_cache.row_counts.push(line_count);
        rows = rows.saturating_add(line_count);
    }

    state.transcript_render_cache.entries.truncate(item_count);
    state
        .transcript_render_cache
        .row_starts
        .truncate(item_count);
    state
        .transcript_render_cache
        .row_counts
        .truncate(item_count);
    rows
}

fn visible_cached_transcript_lines(
    state: &mut TuiState,
    theme: Theme,
    width: usize,
    visible_rows: u16,
    top_scroll: u16,
) -> Vec<Line<'static>> {
    let visible_rows = visible_rows as usize;
    if visible_rows == 0 || state.timeline.items().is_empty() {
        return Vec::new();
    }

    state
        .transcript_render_cache
        .prepare(width, theme, state.timeline.cache_id());
    if !transcript_row_metadata_is_current(state) {
        cached_transcript_row_count(state, theme, width);
    }

    let start = top_scroll as usize;
    let end = start.saturating_add(visible_rows);
    let mut visible = Vec::with_capacity(visible_rows);

    let top_spacer_end = surface::TRANSCRIPT_TOP_SPACER.min(end);
    for row in start..top_spacer_end {
        if row < surface::TRANSCRIPT_TOP_SPACER {
            visible.push(Line::from(""));
        }
    }

    let item_count = state.timeline.items().len();
    let first_item = state
        .transcript_render_cache
        .row_starts
        .partition_point(|row_start| *row_start < start)
        .saturating_sub(1);

    for index in first_item..item_count {
        let item_start = state.transcript_render_cache.row_starts[index];
        let item_count = state.transcript_render_cache.row_counts[index];
        let separator_start = item_start.saturating_sub(if index > 0 { 1 } else { 0 });
        let item_end = item_start.saturating_add(item_count);

        if separator_start >= end || visible.len() >= visible_rows {
            break;
        }

        if index > 0 && separator_start >= start && separator_start < end {
            visible.push(Line::from(""));
        }

        if item_end <= start {
            continue;
        }

        let line_start = start.saturating_sub(item_start).min(item_count);
        let line_end = end.saturating_sub(item_start).min(item_count);
        let lines = cached_item_lines(state, index, theme, width);
        for line in &lines[line_start..line_end] {
            visible.push(line.clone());
            if visible.len() >= visible_rows {
                return visible;
            }
        }
    }

    visible
}

fn transcript_row_metadata_is_current(state: &TuiState) -> bool {
    let item_count = state.timeline.items().len();
    let cache = &state.transcript_render_cache;
    cache.row_starts.len() == item_count
        && cache.row_counts.len() == item_count
        && cache.entries.len() >= item_count
        && state
            .timeline
            .item_revisions()
            .iter()
            .enumerate()
            .all(|(index, revision)| cache.entries[index].revision == Some(*revision))
}

fn cached_item_lines(
    state: &mut TuiState,
    index: usize,
    theme: Theme,
    width: usize,
) -> &[Line<'static>] {
    let item = &state.timeline.items()[index];
    let revision = state.timeline.item_revisions().get(index).copied();
    let cache = &mut state.transcript_render_cache;

    if cache.entries.len() <= index {
        cache
            .entries
            .resize_with(index + 1, || TranscriptRenderCacheEntry {
                revision: None,
                lines: Vec::new(),
            });
    }

    let entry = &mut cache.entries[index];
    if entry.revision != revision {
        entry.revision = revision;
        entry.lines = render_timeline_item_lines(item, theme, width);
    }

    &entry.lines
}

fn render_timeline_item_lines(
    item: &TimelineItem,
    theme: Theme,
    width: usize,
) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    match item {
        TimelineItem::User(message) => push_user_message_lines(&mut lines, message, theme, width),
        TimelineItem::Reasoning(reasoning) => {
            push_reasoning_lines(&mut lines, reasoning, theme, width)
        }
        TimelineItem::Assistant(message) => push_assistant_message_lines(
            &mut lines,
            message_text(message),
            message.streaming,
            theme,
            width,
        ),
        TimelineItem::Tool(tool) => push_tool_lines(&mut lines, tool, theme, width),
        TimelineItem::Todo(todo) => {
            lines.extend(todo_card::render_todo_card_lines(todo, theme, width))
        }
        TimelineItem::Permission(permission) => {
            push_permission_lines(&mut lines, permission, theme, width)
        }
        TimelineItem::Error(error) => push_error_lines(&mut lines, error, theme, width),
        TimelineItem::Notice(notice) => push_notice_lines(&mut lines, notice, theme, width),
    }
    lines
}

fn push_reasoning_lines(
    lines: &mut Vec<Line<'static>>,
    reasoning: &ReasoningView,
    theme: Theme,
    width: usize,
) {
    let content_width = width.saturating_sub(2).max(1);
    let (title, body) = reasoning_title_and_body(&reasoning.text);
    let title = title.unwrap_or_else(|| {
        if reasoning.streaming {
            "Thinking".to_string()
        } else {
            "Thought".to_string()
        }
    });
    let title_suffix = if reasoning.streaming { " …" } else { "" };

    lines.push(Line::from(vec![
        Span::styled("  Thought: ", reasoning_label_style(theme)),
        Span::styled(title, reasoning_label_style(theme)),
        Span::styled(title_suffix, reasoning_label_style(theme)),
    ]));

    let mut pushed = false;
    for raw in body.lines() {
        let raw = clean_reasoning_line(raw);
        let wrapped = if raw.is_empty() {
            vec![String::new()]
        } else {
            wrap_text_to_width(&raw, content_width)
        };

        for content in wrapped {
            pushed = true;
            lines.push(Line::from(vec![
                Span::styled("  ", theme.app_style()),
                Span::styled(content, reasoning_text_style(theme)),
            ]));
        }
    }

    if !pushed && reasoning.streaming {
        lines.push(Line::from(Span::styled("  …", root_dim_style(theme))));
    }
}

fn reasoning_title_and_body(text: &str) -> (Option<String>, String) {
    let mut title = None;
    let mut body_lines = Vec::new();
    let mut consumed_title = false;

    for raw in text.lines() {
        if !consumed_title {
            let cleaned = clean_reasoning_line(raw);
            if cleaned.is_empty() {
                continue;
            }
            title = Some(
                wrap_text_to_width(&cleaned, 80)
                    .into_iter()
                    .next()
                    .unwrap_or(cleaned),
            );
            consumed_title = true;
            continue;
        }
        body_lines.push(raw);
    }

    while body_lines
        .first()
        .is_some_and(|line| line.trim().is_empty())
    {
        body_lines.remove(0);
    }
    (title, body_lines.join("\n"))
}

fn clean_reasoning_line(raw: &str) -> String {
    let mut text = raw.trim().to_string();
    while let Some(rest) = text.strip_prefix('#') {
        text = rest.trim_start().to_string();
    }
    for marker in ["**", "__", "`"] {
        text = text.replace(marker, "");
    }
    text
}

fn message_text(message: &MessageView) -> &str {
    if message.text.is_empty() && message.streaming {
        "…"
    } else {
        &message.text
    }
}

fn push_user_message_lines(
    lines: &mut Vec<Line<'static>>,
    message: &MessageView,
    theme: Theme,
    width: usize,
) {
    let text = message_text(message);
    let content_width = width.saturating_sub(5).max(1);

    push_user_card_line(lines, "", width, theme);

    let mut pushed = false;
    for raw in text.lines() {
        let wrapped = if raw.is_empty() {
            vec![String::new()]
        } else {
            wrap_text_to_width(raw, content_width)
        };

        for content in wrapped {
            pushed = true;
            push_user_card_line(lines, &content, width, theme);
        }
    }

    if !pushed {
        push_user_card_line(lines, "", width, theme);
    }

    push_user_card_line(lines, "", width, theme);
}

fn push_user_card_line(lines: &mut Vec<Line<'static>>, content: &str, width: usize, theme: Theme) {
    let panel_style = user_prompt_panel_style(theme);
    let bar_style = surface::accent_style(
        theme,
        surface::SurfaceEmphasis::User,
        surface::SurfaceKind::Root,
    );
    let pad_style = user_prompt_padding_style(theme);

    let mut line = Line::from(vec![
        Span::styled(surface::ACCENT_BAR_GLYPH, bar_style),
        Span::styled("  ", pad_style),
        Span::styled(content.to_string(), panel_style),
    ]);

    let used = display_width(&line.to_string());
    if width > used {
        line.spans
            .push(Span::styled(" ".repeat(width - used), pad_style));
    } else {
        line.spans.push(Span::styled("  ", pad_style));
    }

    lines.push(line);
}

fn push_assistant_message_lines(
    lines: &mut Vec<Line<'static>>,
    text: &str,
    _streaming: bool,
    theme: Theme,
    width: usize,
) {
    let content_width = width.saturating_sub(3).max(1);
    if text.is_empty() {
        lines.push(Line::from(Span::styled("   …", root_muted_style(theme))));
        return;
    }

    for rendered in render_markdown(text, theme, MarkdownRenderOptions::new(content_width)) {
        let mut spans = vec![Span::styled("   ", theme.app_style())];
        spans.extend(rendered.spans);
        lines.push(Line::from(spans));
    }
}

fn push_tool_lines(lines: &mut Vec<Line<'static>>, tool: &ToolView, theme: Theme, width: usize) {
    lines.extend(tool_card::render_tool_card_lines(tool, theme, width));
}

fn push_permission_lines(
    lines: &mut Vec<Line<'static>>,
    permission: &PermissionView,
    theme: Theme,
    width: usize,
) {
    if permission.status == PermissionPromptStatus::Pending {
        return;
    }

    lines.extend(tool_card::render_permission_card_lines(
        permission, theme, width,
    ));
}

fn push_error_lines(lines: &mut Vec<Line<'static>>, error: &ErrorView, theme: Theme, width: usize) {
    let accent = theme.error;
    let bg = theme.elevated_bg;
    let value_style = elevated_error_style(theme);

    push_card_blank_line(lines, accent, bg, theme, width);

    push_wrapped_card_line(
        lines,
        &format!("error {}", error.message),
        accent,
        value_style,
        theme,
        width,
    );

    push_card_optional_field(
        lines,
        "details",
        error.details.as_deref(),
        accent,
        bg,
        theme,
        width,
    );

    push_card_blank_line(lines, accent, bg, theme, width);
}

fn push_notice_lines(
    lines: &mut Vec<Line<'static>>,
    notice: &NoticeView,
    theme: Theme,
    width: usize,
) {
    let content_width = width.saturating_sub(2).max(1);

    for raw in notice.message.lines() {
        let wrapped = if raw.is_empty() {
            vec![String::new()]
        } else {
            wrap_text_to_width(raw, content_width)
        };

        for content in wrapped {
            lines.push(Line::from(vec![
                Span::styled("  ", theme.app_style()),
                Span::styled(content, root_dim_style(theme)),
            ]));
        }
    }
}

fn push_card_optional_field(
    lines: &mut Vec<Line<'static>>,
    label: &str,
    value: Option<&str>,
    accent: ratatui::style::Color,
    bg: ratatui::style::Color,
    theme: Theme,
    width: usize,
) {
    if let Some(value) = value.filter(|value| !value.is_empty()) {
        push_card_multiline_key_value(lines, label, value, accent, bg, theme, width);
    }
}

fn push_wrapped_card_line(
    lines: &mut Vec<Line<'static>>,
    content: &str,
    accent: ratatui::style::Color,
    value_style: ratatui::style::Style,
    theme: Theme,
    width: usize,
) {
    let prefix_width = display_width(&format!("{} ", surface::ACCENT_BAR_GLYPH));
    let content_width = width.saturating_sub(prefix_width).max(1);
    for wrapped in wrap_text_to_width(content, content_width) {
        let mut line = Line::from(vec![
            Span::styled(
                surface::ACCENT_BAR_GLYPH,
                card_bar_style(accent, theme.root_bg),
            ),
            Span::styled(" ", value_style),
            Span::styled(wrapped, value_style),
        ]);
        pad_card_line_to_width(&mut line, width, value_style);
        lines.push(line);
    }
}

fn push_card_blank_line(
    lines: &mut Vec<Line<'static>>,
    accent: ratatui::style::Color,
    bg: ratatui::style::Color,
    theme: Theme,
    width: usize,
) {
    if width == 0 {
        return;
    }

    let fill_style = ratatui::style::Style::default().bg(bg);
    let mut line = Line::from(vec![Span::styled(
        surface::ACCENT_BAR_GLYPH,
        card_bar_style(accent, theme.root_bg),
    )]);
    pad_card_line_to_width(&mut line, width, fill_style);
    lines.push(line);
}

fn push_card_multiline_key_value(
    lines: &mut Vec<Line<'static>>,
    label: &str,
    value: &str,
    accent: ratatui::style::Color,
    bg: ratatui::style::Color,
    theme: Theme,
    width: usize,
) {
    let (label_style, value_style) = if bg == theme.elevated_bg {
        (elevated_muted(theme), inline_elevated(theme))
    } else {
        (element_muted_style(theme), theme.element_style())
    };
    // Prefix is: accent bar + one card padding cell + "{label:<7}". Wrap value rows to the remaining width so we don't
    // overrun the viewport and get re-wrapped by ratatui Paragraph::wrap.
    let prefix = format!("{} {:<7}", surface::ACCENT_BAR_GLYPH, "");
    let prefix_width = display_width(&prefix);
    let content_width = width.saturating_sub(prefix_width).max(1);

    let mut rows = Vec::new();
    for raw in value.lines() {
        if raw.is_empty() {
            rows.push(String::new());
        } else {
            rows.extend(wrap_text_to_width(raw, content_width));
        }
    }
    if rows.is_empty() {
        rows.push(String::new());
    }

    const MAX_FIELD_ROWS: usize = 8;
    for (index, row) in rows.into_iter().enumerate() {
        if index >= MAX_FIELD_ROWS {
            let mut line = Line::from(vec![
                Span::styled(
                    surface::ACCENT_BAR_GLYPH,
                    card_bar_style(accent, theme.root_bg),
                ),
                Span::styled("…      ", label_style),
                Span::styled("truncated", label_style),
            ]);
            pad_card_line_to_width(&mut line, width, label_style);
            lines.push(line);
            break;
        }

        let field_label = if index == 0 { label } else { "" };
        let mut line = Line::from(vec![
            Span::styled(
                surface::ACCENT_BAR_GLYPH,
                card_bar_style(accent, theme.root_bg),
            ),
            Span::styled(format!(" {field_label:<7}"), label_style),
            Span::styled(row, value_style),
        ]);
        pad_card_line_to_width(&mut line, width, value_style);
        lines.push(line);
    }
}

fn pad_card_line_to_width(
    line: &mut Line<'static>,
    width: usize,
    fill_style: ratatui::style::Style,
) {
    let used = display_width(&line.to_string());
    if width > used {
        line.spans
            .push(Span::styled(" ".repeat(width - used), fill_style));
    }
}

fn card_bar_style(
    accent: ratatui::style::Color,
    bg: ratatui::style::Color,
) -> ratatui::style::Style {
    ratatui::style::Style::default().fg(accent).bg(bg)
}

fn user_prompt_panel_style(theme: Theme) -> ratatui::style::Style {
    ratatui::style::Style::default()
        .fg(theme.text)
        .bg(theme.element_bg)
}

fn user_prompt_padding_style(theme: Theme) -> ratatui::style::Style {
    ratatui::style::Style::default()
        .fg(theme.muted_text)
        .bg(theme.element_bg)
}

fn element_muted_style(theme: Theme) -> ratatui::style::Style {
    ratatui::style::Style::default()
        .fg(theme.muted_text)
        .bg(theme.element_bg)
}

fn inline_elevated(theme: Theme) -> ratatui::style::Style {
    ratatui::style::Style::default()
        .fg(theme.text)
        .bg(theme.elevated_bg)
}

fn elevated_muted(theme: Theme) -> ratatui::style::Style {
    ratatui::style::Style::default()
        .fg(theme.muted_text)
        .bg(theme.elevated_bg)
}

fn elevated_error_style(theme: Theme) -> ratatui::style::Style {
    ratatui::style::Style::default()
        .fg(theme.error)
        .bg(theme.elevated_bg)
}

fn root_muted_style(theme: Theme) -> ratatui::style::Style {
    ratatui::style::Style::default()
        .fg(theme.muted_text)
        .bg(theme.root_bg)
}

fn root_dim_style(theme: Theme) -> ratatui::style::Style {
    ratatui::style::Style::default()
        .fg(theme.dim_text)
        .bg(theme.root_bg)
}

fn reasoning_label_style(theme: Theme) -> ratatui::style::Style {
    ratatui::style::Style::default()
        .fg(theme.accent)
        .bg(theme.root_bg)
}

fn reasoning_text_style(theme: Theme) -> ratatui::style::Style {
    ratatui::style::Style::default()
        .fg(theme.muted_text)
        .bg(theme.root_bg)
}

#[cfg(test)]
mod tests {
    use super::{
        cached_transcript_row_count, render_transcript, transcript_lines, transcript_row_count,
        visible_cached_transcript_lines, visible_transcript_lines,
    };
    use crate::{
        agent::{AutoContinueState, TodoItem, TodoStatus},
        tui::{
            AppEvent, AssistantDeltaEvent, ErrorEvent, PermissionRequestEvent, ReasoningDeltaEvent,
            ReasoningDoneEvent, ToolFinishedEvent, ToolOutcome, ToolStartedEvent, UserMessageEvent,
            events::{AutoContinueChangedEvent, TodoSnapshotEvent},
            state::TuiState,
            theme::Theme,
            timeline::Timeline,
        },
    };
    use ratatui::{Terminal, backend::TestBackend, layout::Rect};

    #[test]
    fn transcript_rows_wrap_using_display_width() {
        let mut state = TuiState::default();
        state.apply_event(crate::tui::events::AppEvent::UserMessage(
            UserMessageEvent::new("a你b"),
        ));

        let theme = Theme::dark();
        let lines = transcript_lines(&state, theme, 6);

        assert_eq!(transcript_row_count(&state, theme, 6), lines.len());
        assert_eq!(lines.len(), 6);
        assert!(lines.iter().any(|line| line.to_string().contains('你')));
    }

    #[test]
    fn key_value_fields_wrap_to_target_width_for_tool_permission_and_error() {
        let mut state = TuiState::default();
        state.apply_event(AppEvent::UserMessage(UserMessageEvent::new("seed")));

        let mut tool_started = ToolStartedEvent::new("call-tool", "shell__exec", "run");
        tool_started.arguments = Some("--really-long-arg ".repeat(20));
        state.apply_event(AppEvent::ToolStarted(tool_started));
        let mut tool_finished =
            ToolFinishedEvent::new("call-tool", "shell__exec", "run", ToolOutcome::Failure);
        tool_finished.output = Some("output=".to_string() + &"x".repeat(200));
        state.apply_event(AppEvent::ToolFinished(tool_finished));

        let mut request = PermissionRequestEvent::new("call-perm", "shell__exec", "needs approval");
        request.arguments = Some("arg ".repeat(60));
        request.rationale = Some("because ".repeat(80));
        state.apply_event(AppEvent::PermissionRequested(request));

        let mut err = ErrorEvent::new("boom");
        err.details = Some("detail ".repeat(90));
        state.apply_event(AppEvent::Error(err));

        let theme = Theme::dark();
        let width = 44usize;
        let lines = transcript_lines(&state, theme, width);

        // Ensure no generated line exceeds target width in display cells.
        for line in &lines {
            let w = crate::tui::measure::display_width(&line.to_string());
            assert!(
                w <= width,
                "line width {w} > {width}: {:?}",
                line.to_string()
            );
        }

        // Ensure key-value fields are capped to MAX_FIELD_ROWS (8) + truncated indicator.
        let truncated_rows = lines
            .iter()
            .filter(|l| l.to_string().contains("truncated"))
            .count();
        assert!(
            truncated_rows >= 1,
            "expected at least one truncated indicator row"
        );
    }

    #[test]
    fn error_card_uses_composer_style_red_guide() {
        let mut state = TuiState::default();
        state.apply_event(AppEvent::Error(ErrorEvent::new("stream stopped")));

        let theme = Theme::dark();
        let lines = transcript_lines(&state, theme, 64);
        let error_line = lines
            .iter()
            .find(|line| line.to_string().contains("error stream stopped"))
            .expect("error line renders");
        let guide = error_line.spans.first().expect("error line has guide");

        assert_eq!(
            guide.content.as_ref(),
            crate::tui::surface::ACCENT_BAR_GLYPH
        );
        assert_eq!(guide.style.fg, Some(theme.error));
        assert_eq!(guide.style.bg, Some(theme.root_bg));

        let card_pad = error_line.spans.get(1).expect("error line has card pad");
        assert_eq!(card_pad.content.as_ref(), " ");
        assert_eq!(card_pad.style.bg, Some(theme.elevated_bg));

        let error_index = lines
            .iter()
            .position(|line| line.to_string().contains("error stream stopped"))
            .expect("error line index");
        assert!(error_index > 0, "error card has top padding row");
        let top_pad = &lines[error_index - 1];
        let bottom_pad = &lines[error_index + 1];
        for pad in [top_pad, bottom_pad] {
            assert_eq!(pad.spans[0].style.fg, Some(theme.error));
            assert_eq!(pad.spans[0].style.bg, Some(theme.root_bg));
            assert!(pad.spans[1].content.as_ref().starts_with(' '));
            assert_eq!(pad.spans[1].style.bg, Some(theme.elevated_bg));
        }
    }

    #[test]
    fn todo_timeline_items_render_full_card_sections() {
        let mut state = TuiState::default();
        state.apply_event(AppEvent::AutoContinueChanged(
            AutoContinueChangedEvent::new(AutoContinueState {
                enabled: true,
                max_continuations: 2,
            }),
        ));
        state.apply_event(AppEvent::TodoSnapshot(TodoSnapshotEvent::new(vec![
            TodoItem {
                id: "t1".into(),
                content: "Inspect timeline integration".into(),
                status: TodoStatus::InProgress,
            },
            TodoItem {
                id: "t2".into(),
                content: "Keep wrapping stable at narrow widths".into(),
                status: TodoStatus::Pending,
            },
            TodoItem {
                id: "t3".into(),
                content: "Snapshot final layout".into(),
                status: TodoStatus::Completed,
            },
        ])));

        let lines = transcript_lines(&state, Theme::dark(), 56)
            .into_iter()
            .map(|line| line.to_string())
            .collect::<Vec<_>>();
        let joined = lines.join("\n");

        assert!(joined.contains("# Todos"));
        assert!(joined.contains("[•] Inspect timeline integration"));
        assert!(joined.contains("[ ] Keep wrapping stable at narrow widths"));
        assert!(joined.contains("[✓] Snapshot final layout"));
        assert!(!joined.contains("auto on"));
        assert!(!joined.contains("current"));
        assert!(!joined.contains("items · auto-continue"));

        for rendered in lines {
            let measured = crate::tui::measure::display_width(&rendered);
            assert!(measured <= 56, "line width {measured} > 56: {rendered:?}");
        }
    }

    #[test]
    fn visible_window_clips_transcript_rows_before_rendering() {
        let lines = (0..20)
            .map(|index| ratatui::text::Line::from(format!("row-{index}")))
            .collect::<Vec<_>>();

        let visible = visible_transcript_lines(&lines, 5, 12)
            .into_iter()
            .map(|line| line.to_string())
            .collect::<Vec<_>>();

        assert_eq!(
            visible,
            vec!["row-12", "row-13", "row-14", "row-15", "row-16"]
        );

        let bottom = visible_transcript_lines(&lines, 5, 18)
            .into_iter()
            .map(|line| line.to_string())
            .collect::<Vec<_>>();
        assert_eq!(bottom, vec!["row-18", "row-19"]);
    }

    #[test]
    fn cached_visible_transcript_matches_full_transcript_window() {
        let mut state = TuiState::default();
        state.apply_event(AppEvent::UserMessage(UserMessageEvent::new("seed")));
        for index in 0..30 {
            state.timeline.push_notice(format!("history line {index}"));
        }
        state.apply_event(AppEvent::AssistantDelta(AssistantDeltaEvent::new(
            "# Heading\n```rust\nlet value = 42;\n```\n- done",
        )));
        state.apply_event(AppEvent::AssistantDone { message_id: None });

        let theme = Theme::dark();
        let width = 72;
        let full = transcript_lines(&state, theme, width);
        let total_rows = cached_transcript_row_count(&mut state, theme, width);
        let visible = visible_cached_transcript_lines(&mut state, theme, width, 9, 12)
            .into_iter()
            .map(|line| line.to_string())
            .collect::<Vec<_>>();
        let expected = visible_transcript_lines(&full, 9, 12)
            .into_iter()
            .map(|line| line.to_string())
            .collect::<Vec<_>>();

        assert_eq!(total_rows, full.len());
        assert_eq!(visible, expected);
    }

    #[test]
    fn transcript_cache_invalidates_streaming_assistant_item() {
        let mut state = TuiState::default();
        state.apply_event(AppEvent::AssistantDelta(AssistantDeltaEvent::new("first")));

        let theme = Theme::dark();
        let width = 80;
        let before_rows = cached_transcript_row_count(&mut state, theme, width);
        let before_revision = state.transcript_render_cache.entries[0].revision;

        state.apply_event(AppEvent::AssistantDelta(AssistantDeltaEvent::new(
            " second",
        )));

        let after_rows = cached_transcript_row_count(&mut state, theme, width);
        let after_revision = state.transcript_render_cache.entries[0].revision;
        let visible = visible_cached_transcript_lines(&mut state, theme, width, 8, 0)
            .into_iter()
            .map(|line| line.to_string())
            .collect::<Vec<_>>()
            .join("\n");

        assert_eq!(before_rows, after_rows);
        assert_ne!(before_revision, after_revision);
        assert!(visible.contains("first second"), "{visible}");
    }

    #[test]
    fn transcript_cache_is_namespaced_by_timeline_replacement() {
        let mut state = TuiState::default();
        state.timeline.push_notice("old timeline");
        let theme = Theme::dark();
        let width = 80;

        cached_transcript_row_count(&mut state, theme, width);
        assert!(
            visible_cached_transcript_lines(&mut state, theme, width, 8, 0)
                .into_iter()
                .any(|line| line.to_string().contains("old timeline"))
        );

        state.timeline = Timeline::new();
        state.timeline.push_notice("new timeline");

        let visible = visible_cached_transcript_lines(&mut state, theme, width, 8, 0)
            .into_iter()
            .map(|line| line.to_string())
            .collect::<Vec<_>>()
            .join("\n");

        assert!(visible.contains("new timeline"), "{visible}");
        assert!(!visible.contains("old timeline"), "{visible}");
    }

    #[test]
    fn manual_history_view_stays_anchored_when_streaming_rows_append() {
        let theme = Theme::dark();
        let width = 80;
        let viewport_rows = 5;
        let mut state = TuiState::default();

        for index in 0..24 {
            state.timeline.push_notice(format!("history line {index}"));
        }

        let before_lines = transcript_lines(&state, theme, width);
        state.sync_transcript_viewport_rows(before_lines.len());
        let target_top = 6usize;
        let before_max_scroll = crate::tui::measure::max_scroll(before_lines.len(), viewport_rows);
        state.transcript_scroll = before_max_scroll.saturating_sub(target_top as u16);
        state.auto_scroll = false;

        let before_top = crate::tui::measure::resolved_scroll_offset(
            before_lines.len(),
            viewport_rows,
            state.transcript_scroll,
            state.auto_scroll,
        );
        let before_visible = visible_transcript_lines(&before_lines, viewport_rows, before_top);
        let before_first = before_visible
            .first()
            .map(|line| line.to_string())
            .expect("visible row before append");

        state.apply_event(AppEvent::AssistantDelta(AssistantDeltaEvent::new(
            "streaming row one\nstreaming row two\nstreaming row three",
        )));

        let after_lines = transcript_lines(&state, theme, width);
        state.sync_transcript_viewport_rows(after_lines.len());
        let after_top = crate::tui::measure::resolved_scroll_offset(
            after_lines.len(),
            viewport_rows,
            state.transcript_scroll,
            state.auto_scroll,
        );
        let after_visible = visible_transcript_lines(&after_lines, viewport_rows, after_top);
        let after_first = after_visible
            .first()
            .map(|line| line.to_string())
            .expect("visible row after append");

        assert_eq!(after_top, before_top);
        assert_eq!(after_first, before_first);
        assert!(!state.auto_scroll);
    }

    #[test]
    fn pending_permission_is_hidden_from_transcript_while_panel_is_active() {
        let mut state = TuiState::default();
        let mut request = PermissionRequestEvent::new("call-perm", "shell__exec", "cargo test all");
        request.arguments = Some("cargo test all".into());
        request.rationale = Some("tests need confirmation".into());
        state.apply_event(AppEvent::PermissionRequested(request));

        let lines = transcript_lines(&state, Theme::dark(), 60)
            .into_iter()
            .map(|line| line.to_string())
            .collect::<Vec<_>>();

        assert!(
            !lines
                .iter()
                .any(|line| line.contains("Permission required")),
            "{lines:?}"
        );
        assert!(
            !lines.iter().any(|line| line.contains("cargo test all")),
            "{lines:?}"
        );
    }

    #[test]
    fn reasoning_content_renders_inline_in_transcript() {
        let mut state = TuiState::default();
        state.apply_event(AppEvent::ReasoningDelta(ReasoningDeltaEvent::new(
            "r-1",
            "Inspecting workflow",
        )));
        state.apply_event(AppEvent::ReasoningDone(ReasoningDoneEvent::new(
            "r-1",
            "Inspecting workflow",
        )));

        let lines = transcript_lines(&state, Theme::dark(), 60)
            .into_iter()
            .map(|line| line.to_string())
            .collect::<Vec<_>>();

        assert!(
            lines.iter().any(|line| line.contains("Thought")),
            "{lines:?}"
        );
        assert!(
            lines
                .iter()
                .any(|line| line.contains("Inspecting workflow")),
            "{lines:?}"
        );
    }

    #[test]
    fn reasoning_title_strips_markdown_and_body_is_indented() {
        let mut state = TuiState::default();
        state.apply_event(AppEvent::ReasoningDone(ReasoningDoneEvent::new(
            "r-1",
            "**Evaluating code status**\n\nI need to check `git diff` output.",
        )));

        let lines = transcript_lines(&state, Theme::dark(), 80)
            .into_iter()
            .map(|line| line.to_string())
            .collect::<Vec<_>>();

        assert!(
            lines
                .iter()
                .any(|line| line.contains("Thought: Evaluating code status")),
            "{lines:?}"
        );
        assert!(!lines.iter().any(|line| line.contains("**")), "{lines:?}");
        assert!(
            lines
                .iter()
                .any(|line| line.contains("I need to check git diff output.")),
            "{lines:?}"
        );
    }

    #[test]
    fn assistant_markdown_is_rendered_as_formatted_plain_tui_lines() {
        let mut state = TuiState::default();
        state.apply_event(AppEvent::AssistantDelta(
            crate::tui::events::AssistantDeltaEvent::new(
                "# Title\n- **item** with `code`\n```\nlet x = 1;\n```",
            ),
        ));
        state.apply_event(AppEvent::AssistantDone { message_id: None });

        let lines = transcript_lines(&state, Theme::dark(), 80)
            .into_iter()
            .map(|line| line.to_string())
            .collect::<Vec<_>>()
            .join("\n");

        assert!(lines.contains("▌ Title"), "{lines}");
        assert!(lines.contains("• item with code"), "{lines}");
        assert!(lines.contains("let x = 1;"), "{lines}");
        assert!(!lines.contains("**item**"), "{lines}");
        assert!(!lines.contains("```"), "{lines}");
    }

    #[test]
    fn user_card_bar_is_separated_from_card_background() {
        let mut state = TuiState::default();
        state.apply_event(AppEvent::UserMessage(UserMessageEvent::new("hello")));

        let backend = TestBackend::new(40, 8);
        let mut terminal = Terminal::new(backend).expect("terminal");
        terminal
            .draw(|frame| {
                let area = Rect::new(0, 0, 40, 8);
                render_transcript(frame, &mut state, area, Theme::dark());
            })
            .expect("draw");

        let buffer = terminal.backend().buffer();
        let bar_cell = &buffer[(0, 2)];
        let card_cell = &buffer[(1, 2)];

        assert_eq!(bar_cell.bg, Theme::dark().root_bg);
        assert_eq!(card_cell.bg, Theme::dark().element_bg);
    }
}
