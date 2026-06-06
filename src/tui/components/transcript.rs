use ratatui::{
    Frame,
    layout::{Constraint, Layout, Rect},
    style::Style,
    text::{Line, Span, Text},
    widgets::{Block, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState},
};

use crate::tui::{
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
    let lines = transcript_lines(state, theme, width);
    let total_rows = lines.len();
    state.sync_transcript_viewport_rows(total_rows);
    let visible_rows = content_area.height;
    let scroll = crate::tui::measure::resolved_scroll_offset(
        total_rows,
        visible_rows,
        state.transcript_scroll,
        state.auto_scroll,
    );

    let visible_lines = visible_transcript_lines(&lines, visible_rows, scroll);
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

        match item {
            TimelineItem::User(message) => {
                push_user_message_lines(&mut lines, message, theme, width)
            }
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
    }

    lines
}

fn push_reasoning_lines(
    lines: &mut Vec<Line<'static>>,
    reasoning: &ReasoningView,
    theme: Theme,
    width: usize,
) {
    let content_width = width.saturating_sub(5).max(1);
    lines.push(Line::from(vec![
        Span::styled("  thinking", reasoning_label_style(theme)),
        Span::styled(
            if reasoning.streaming { " …" } else { "" },
            reasoning_label_style(theme),
        ),
    ]));

    let mut pushed = false;
    for raw in reasoning.text.lines() {
        let wrapped = if raw.is_empty() {
            vec![String::new()]
        } else {
            wrap_text_to_width(raw, content_width)
        };

        for content in wrapped {
            pushed = true;
            lines.push(Line::from(vec![
                Span::styled("     ", theme.app_style()),
                Span::styled(content, reasoning_text_style(theme)),
            ]));
        }
    }

    if !pushed {
        lines.push(Line::from(Span::styled("     …", root_dim_style(theme))));
    }
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
    let mut pushed = false;
    let content_width = width.saturating_sub(3).max(1);
    let mut markdown = MarkdownRenderState::default();

    for raw in text.lines() {
        let rendered = markdown.render_line(raw, theme);
        let plain = rendered
            .iter()
            .map(|span| span.content.as_ref())
            .collect::<String>();

        if plain.is_empty() {
            pushed = true;
            lines.push(Line::from(vec![
                Span::styled("   ", theme.app_style()),
                Span::styled(String::new(), markdown.current_style(theme)),
            ]));
            continue;
        }

        if display_width(&plain) <= content_width {
            pushed = true;
            let mut spans = vec![Span::styled("   ", theme.app_style())];
            spans.extend(rendered);
            lines.push(Line::from(spans));
            continue;
        }

        for content in wrap_text_to_width(&plain, content_width) {
            pushed = true;
            lines.push(Line::from(vec![
                Span::styled("   ", theme.app_style()),
                Span::styled(content, markdown.current_style(theme)),
            ]));
        }
    }

    if !pushed {
        lines.push(Line::from(Span::styled("   …", root_muted_style(theme))));
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct MarkdownRenderState {
    in_code_block: bool,
}

impl MarkdownRenderState {
    fn render_line(&mut self, raw: &str, theme: Theme) -> Vec<Span<'static>> {
        let trimmed = raw.trim_start();
        if trimmed.starts_with("```") {
            self.in_code_block = !self.in_code_block;
            return Vec::new();
        }

        if self.in_code_block {
            return vec![Span::styled(raw.to_string(), markdown_code_style(theme))];
        }

        if let Some((level, heading)) = parse_heading(trimmed) {
            let prefix = if level <= 2 { "▌ " } else { "• " };
            return vec![Span::styled(
                format!("{prefix}{heading}"),
                markdown_heading_style(theme),
            )];
        }

        let line = normalize_list_marker(raw);
        parse_inline_markdown(&line, theme)
    }

    fn current_style(self, theme: Theme) -> ratatui::style::Style {
        if self.in_code_block {
            markdown_code_style(theme)
        } else {
            theme.app_style()
        }
    }
}

fn parse_heading(line: &str) -> Option<(usize, String)> {
    let hashes = line.chars().take_while(|ch| *ch == '#').count();
    if !(1..=6).contains(&hashes) {
        return None;
    }
    let rest = line.get(hashes..)?.strip_prefix(' ')?;
    Some((hashes, rest.trim().to_string()))
}

fn normalize_list_marker(raw: &str) -> String {
    let trimmed = raw.trim_start();
    let indent = raw.len().saturating_sub(trimmed.len());
    if let Some(rest) = trimmed
        .strip_prefix("- ")
        .or_else(|| trimmed.strip_prefix("* "))
        .or_else(|| trimmed.strip_prefix("+ "))
    {
        return format!("{}• {rest}", " ".repeat(indent));
    }

    if let Some((marker, rest)) = trimmed.split_once(' ')
        && marker.ends_with('.')
        && marker[..marker.len().saturating_sub(1)]
            .chars()
            .all(|ch| ch.is_ascii_digit())
        && !rest.is_empty()
    {
        return format!("{}{} {rest}", " ".repeat(indent), marker);
    }

    raw.to_string()
}

fn parse_inline_markdown(raw: &str, theme: Theme) -> Vec<Span<'static>> {
    let mut spans = Vec::new();
    let mut rest = raw;
    let mut bold = false;
    let mut code = false;

    while !rest.is_empty() {
        let next_bold = rest.find("**");
        let next_code = rest.find('`');
        let next = match (next_bold, next_code) {
            (Some(b), Some(c)) if b <= c => Some((b, "**")),
            (Some(_), Some(c)) => Some((c, "`")),
            (Some(b), None) => Some((b, "**")),
            (None, Some(c)) => Some((c, "`")),
            (None, None) => None,
        };

        let Some((index, marker)) = next else {
            spans.push(Span::styled(
                rest.to_string(),
                inline_style(theme, bold, code),
            ));
            break;
        };

        if index > 0 {
            spans.push(Span::styled(
                rest[..index].to_string(),
                inline_style(theme, bold, code),
            ));
        }

        if marker == "**" {
            bold = !bold;
        } else {
            code = !code;
        }
        rest = &rest[index + marker.len()..];
    }

    spans
}

fn inline_style(theme: Theme, bold: bool, code: bool) -> ratatui::style::Style {
    let mut style = if code {
        markdown_code_style(theme)
    } else {
        theme.app_style()
    };
    if bold {
        style = style.add_modifier(ratatui::style::Modifier::BOLD);
    }
    style
}

fn markdown_heading_style(theme: Theme) -> ratatui::style::Style {
    theme
        .app_style()
        .fg(theme.text)
        .add_modifier(ratatui::style::Modifier::BOLD)
}

fn markdown_code_style(theme: Theme) -> ratatui::style::Style {
    ratatui::style::Style::default()
        .fg(theme.text)
        .bg(theme.element_bg)
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

    push_wrapped_card_line(
        lines,
        &format!("error {}", error.message),
        accent,
        theme.elevated_bg,
        elevated_error_style(theme),
        width,
    );

    push_card_optional_field(
        lines,
        "details",
        error.details.as_deref(),
        accent,
        theme.elevated_bg,
        theme,
        width,
    );
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
    bg: ratatui::style::Color,
    value_style: ratatui::style::Style,
    width: usize,
) {
    let content_width = width.saturating_sub(4).max(1);
    for wrapped in wrap_text_to_width(content, content_width) {
        let mut line = Line::from(vec![
            Span::styled(surface::ACCENT_BAR_GLYPH, card_bar_style(accent, bg)),
            Span::styled("  ", value_style),
            Span::styled(wrapped, value_style),
        ]);
        pad_card_line_to_width(&mut line, width, value_style);
        lines.push(line);
    }
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
    // Prefix is: accent bar + "  {label:<7}". Wrap value rows to the remaining width so we don't
    // overrun the viewport and get re-wrapped by ratatui Paragraph::wrap.
    let prefix = format!("{}  {:<7}", surface::ACCENT_BAR_GLYPH, "");
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
                Span::styled(surface::ACCENT_BAR_GLYPH, card_bar_style(accent, bg)),
                Span::styled("  …      ", label_style),
                Span::styled("truncated", label_style),
            ]);
            pad_card_line_to_width(&mut line, width, label_style);
            lines.push(line);
            break;
        }

        let field_label = if index == 0 { label } else { "" };
        let mut line = Line::from(vec![
            Span::styled(surface::ACCENT_BAR_GLYPH, card_bar_style(accent, bg)),
            Span::styled(format!("  {field_label:<7}"), label_style),
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
        .fg(theme.notice)
        .bg(theme.root_bg)
        .add_modifier(ratatui::style::Modifier::BOLD)
}

fn reasoning_text_style(theme: Theme) -> ratatui::style::Style {
    ratatui::style::Style::default()
        .fg(theme.muted_text)
        .bg(theme.root_bg)
}

#[cfg(test)]
mod tests {
    use super::{
        render_transcript, transcript_lines, transcript_row_count, visible_transcript_lines,
    };
    use crate::{
        agent::{AutoContinueState, TodoItem, TodoStatus},
        tui::{
            AppEvent, AssistantDeltaEvent, ErrorEvent, PermissionRequestEvent, ReasoningDeltaEvent,
            ReasoningDoneEvent, ToolFinishedEvent, ToolOutcome, ToolStartedEvent, UserMessageEvent,
            events::{AutoContinueChangedEvent, TodoSnapshotEvent},
            state::TuiState,
            theme::Theme,
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
            lines.iter().any(|line| line.contains("thinking")),
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
