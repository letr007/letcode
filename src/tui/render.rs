use ratatui::{
    Frame,
    layout::{Alignment, Constraint, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Clear, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState, Wrap},
};

use super::{
    state::{AppPhase, TuiState},
    theme::Theme,
    timeline::{
        ErrorView, MessageRole, MessageView, NoticeView, PermissionPromptStatus, PermissionView,
        TimelineItem, ToolExecutionStatus, ToolView,
    },
};

const MAX_FIELD_LINES: usize = 8;
const PERMISSION_HINT: &str = "[a] approve  [d] deny";
const OUTER_PAD_X: u16 = 2;
const CONTENT_GAP: u16 = 1;
const ACCENT_BAR_GLYPH: &str = "┃";
const PROMPT_BOTTOM_LEFT_GLYPH: &str = "╹";
const PROMPT_BOTTOM_CAP_GLYPH: &str = "▀";
const WELCOME_ART_LEFT: &[&str] = &[
    "▄          ▄  ",
    "█    █▀▀█ ▀█▀▀",
    "█    █▀▀▀  █  ",
    "▀▀▀▀ ▀▀▀▀  ▀▀▀",
];
const WELCOME_ART_RIGHT: &[&str] = &[
    "             ▄     ",
    "█▀▀▀ █▀▀█ █▀▀█ █▀▀█",
    "█    █  █ █  █ █▀▀▀",
    "▀▀▀▀ ▀▀▀▀ ▀▀▀▀ ▀▀▀▀",
];

/// Render the TUI from the current state using ratatui widgets only.
///
/// This function is intentionally pure from the application's point of view: it reads the
/// immutable [`TuiState`] plus theme tokens and never invokes tools, resolves permissions,
/// persists transcripts, or mutates runtime/business state.
pub fn render(frame: &mut Frame<'_>, state: &TuiState) {
    let theme = Theme::dark();
    let area = frame.area();

    if area.is_empty() {
        return;
    }

    frame.render_widget(Block::new().style(theme.app_style()), area);

    let workspace = workspace_area(area);

    if workspace.height == 1 {
        render_footer(frame, state, workspace, theme);
        return;
    }

    let composer_height = composer_height(workspace.height, &state.input_buffer);
    let permission_height = if state.pending_permission.is_some() {
        permission_height(workspace.height)
    } else {
        0
    };
    let gap_height = if workspace.height >= 7 {
        CONTENT_GAP
    } else {
        0
    };

    let mut constraints = vec![Constraint::Min(0)];
    if state.pending_permission.is_some() {
        constraints.push(Constraint::Length(permission_height));
    }
    constraints.push(Constraint::Length(gap_height));
    constraints.push(Constraint::Length(composer_height));
    constraints.push(Constraint::Length(1));

    let chunks = Layout::vertical(constraints).split(workspace);
    let mut chunk_index = 1;

    render_transcript(frame, state, chunks[0], theme);

    if let Some(permission) = &state.pending_permission {
        render_pending_permission(frame, permission, chunks[chunk_index], theme);
        chunk_index += 1;
    }

    chunk_index += 1;
    render_composer(frame, state, chunks[chunk_index], theme);
    chunk_index += 1;
    render_footer(frame, state, chunks[chunk_index], theme);
}

fn workspace_area(area: Rect) -> Rect {
    if area.width <= OUTER_PAD_X * 2 + 4 {
        return area;
    }

    Rect::new(
        area.x + OUTER_PAD_X,
        area.y,
        area.width.saturating_sub(OUTER_PAD_X * 2),
        area.height,
    )
}

fn composer_height(total_height: u16, input: &str) -> u16 {
    match total_height {
        0..=2 => 0,
        3..=5 => 1,
        6..=8 => 3,
        _ => {
            let rows = input.lines().count().max(1) as u16;
            (rows + 4).clamp(5, 7).min(total_height.saturating_sub(2))
        }
    }
}

fn permission_height(total_height: u16) -> u16 {
    match total_height {
        0..=2 => 0,
        3..=6 => 1,
        7..=10 => 3,
        _ => 5,
    }
}

fn render_transcript(frame: &mut Frame<'_>, state: &TuiState, area: Rect, theme: Theme) {
    if area.is_empty() {
        return;
    }

    if state.timeline.items().is_empty() {
        render_welcome(frame, area, theme);
        return;
    }

    let has_scrollbar = area.width >= 24;
    let (content_area, scrollbar_area) = if has_scrollbar {
        let chunks = Layout::horizontal([Constraint::Min(1), Constraint::Length(1)]).split(area);
        (chunks[0], Some(chunks[1]))
    } else {
        (area, None)
    };
    let lines = transcript_lines(state, theme, content_area.width as usize);
    let visible_rows = content_area.height;
    let max_scroll =
        u16::try_from(lines.len().saturating_sub(visible_rows as usize)).unwrap_or(u16::MAX);
    let scroll = if state.auto_scroll {
        max_scroll
    } else {
        state.transcript_scroll.min(max_scroll)
    };

    let paragraph = Paragraph::new(Text::from(lines.clone()))
        .style(theme.app_style())
        .wrap(Wrap { trim: false })
        .scroll((scroll, 0));

    frame.render_widget(paragraph, content_area);

    if let Some(scrollbar_area) = scrollbar_area
        && lines.len() > visible_rows as usize
        && visible_rows > 0
    {
        let mut scrollbar_state = ScrollbarState::new(lines.len())
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

fn render_welcome(frame: &mut Frame<'_>, area: Rect, theme: Theme) {
    if area.is_empty() {
        return;
    }

    if area.width >= 52 && area.height >= 4 {
        let lines: Vec<Line<'static>> = WELCOME_ART_LEFT
            .iter()
            .zip(WELCOME_ART_RIGHT.iter())
            .map(|(left, right)| {
                Line::from(vec![
                    Span::styled(format!("{left} "), wordmark_shadow_style(theme)),
                    Span::styled((*right).to_string(), wordmark_style(theme)),
                ])
            })
            .collect();
        let lines_height = lines.len() as u16;
        let title_y = area.y + area.height.saturating_sub(lines_height).saturating_div(2);
        frame.render_widget(
            Paragraph::new(lines)
                .style(theme.app_style())
                .alignment(Alignment::Center),
            Rect::new(area.x, title_y, area.width, lines_height),
        );
        return;
    }

    let title = if area.width >= 14 { "LETCODE" } else { "LC" };
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(title, wordmark_style(theme))))
            .style(theme.app_style())
            .alignment(Alignment::Center),
        area,
    );
}

fn render_pending_permission(
    frame: &mut Frame<'_>,
    permission: &PermissionView,
    area: Rect,
    theme: Theme,
) {
    if area.is_empty() {
        return;
    }

    if area.height < 3 || area.width < 24 {
        let line = Line::from(vec![
            Span::styled("PERMISSION ", theme.approval_style()),
            Span::styled(permission.tool_name.clone(), inline_elevated(theme)),
            Span::styled(": ", inline_elevated(theme)),
            Span::styled(permission.summary.clone(), inline_elevated(theme)),
            Span::styled("  ", inline_elevated(theme)),
            Span::styled(PERMISSION_HINT, theme.approval_style()),
        ]);
        frame.render_widget(
            Paragraph::new(line)
                .style(theme.elevated_style())
                .wrap(Wrap { trim: true }),
            area,
        );
        return;
    }

    frame.render_widget(Clear, area);
    render_surface_fill(frame, area, theme.elevated_style());
    render_accent_bar(frame, area, theme.approval_style());

    let mut lines = vec![
        Line::from(vec![
            Span::styled("Permission required", elevated_title_style(theme)),
            Span::styled("  ", inline_elevated(theme)),
            Span::styled(permission.call_id.clone(), elevated_muted(theme)),
        ]),
        Line::from(vec![
            Span::styled("tool ", elevated_muted(theme)),
            Span::styled(permission.tool_name.clone(), theme.approval_style()),
            Span::styled(" — ", inline_elevated(theme)),
            Span::styled(permission.summary.clone(), inline_elevated(theme)),
        ]),
        Line::from(Span::styled(PERMISSION_HINT, theme.approval_style())),
    ];

    push_optional_field_elevated(&mut lines, "args", permission.arguments.as_deref(), theme);
    push_optional_field_elevated(&mut lines, "why", permission.rationale.as_deref(), theme);

    let content_area = inset_left(area, 3);
    let paragraph = Paragraph::new(Text::from(lines))
        .style(theme.elevated_style())
        .wrap(Wrap { trim: false });

    frame.render_widget(paragraph, content_area);
}

fn render_composer(frame: &mut Frame<'_>, state: &TuiState, area: Rect, theme: Theme) {
    if area.is_empty() {
        return;
    }

    if area.height < 3 || area.width < 16 {
        let content = if state.input_buffer.is_empty() {
            "message…".to_string()
        } else {
            state.input_buffer.clone()
        };
        let line = Line::from(vec![
            Span::styled(
                ACCENT_BAR_GLYPH,
                Style::default().fg(theme.user).bg(theme.root_bg),
            ),
            Span::styled(" ", theme.element_style()),
            Span::styled(content, theme.element_style()),
        ]);
        frame.render_widget(Paragraph::new(line).style(theme.element_style()), area);
        if state.pending_permission.is_none() {
            let cursor_x = area.x.saturating_add(2).saturating_add(
                state
                    .input_buffer
                    .chars()
                    .count()
                    .min(area.width.saturating_sub(3) as usize) as u16,
            );
            frame.set_cursor_position((cursor_x, area.y));
        }
        return;
    }

    render_accent_bar(
        frame,
        area,
        Style::default().fg(theme.user).bg(theme.root_bg),
    );

    let surface_area = Rect::new(
        area.x + 1,
        area.y,
        area.width.saturating_sub(1),
        area.height.saturating_sub(1),
    );
    render_surface_fill(frame, surface_area, theme.element_style());

    let content = if state.input_buffer.is_empty() {
        Line::from(Span::styled(
            "message letcode…",
            element_muted_style(theme).add_modifier(Modifier::ITALIC),
        ))
    } else {
        Line::from(Span::styled(
            state.input_buffer.clone(),
            theme.element_style(),
        ))
    };

    let textarea_area = Rect::new(
        area.x + 3,
        area.y + 1,
        area.width.saturating_sub(5),
        area.height.saturating_sub(3).max(1),
    );
    let paragraph = Paragraph::new(content)
        .style(theme.element_style())
        .wrap(Wrap { trim: false });
    frame.render_widget(paragraph, textarea_area);

    if state.pending_permission.is_none() {
        let cursor_col = state
            .input_buffer
            .chars()
            .count()
            .min(textarea_area.width.saturating_sub(1) as usize) as u16;
        frame.set_cursor_position((textarea_area.x + cursor_col, textarea_area.y));
    }

    let metadata_y = area.y + area.height.saturating_sub(2);
    if area.height >= 5 && metadata_y > area.y && metadata_y < area.y + area.height {
        let mode = if state.pending_permission.is_some() {
            "approval pending"
        } else {
            "prompt"
        };
        let metadata = Line::from(vec![
            Span::styled(mode, element_accent_style(theme)),
            Span::styled(" · ", element_dim_style(theme)),
            Span::styled(state.model_label.clone(), theme.element_style()),
            Span::styled(" · permission ", element_muted_style(theme)),
            Span::styled(state.permission_mode_label.clone(), theme.element_style()),
        ]);
        frame.render_widget(
            Paragraph::new(metadata).style(theme.element_style()),
            Rect::new(area.x + 3, metadata_y, area.width.saturating_sub(5), 1),
        );
    }

    render_composer_cap(frame, area, theme);
}

fn render_footer(frame: &mut Frame<'_>, state: &TuiState, area: Rect, theme: Theme) {
    if area.is_empty() {
        return;
    }

    frame.render_widget(Block::new().style(theme.app_style()), area);

    let mut left_spans = phase_indicator_spans(state, theme);
    left_spans.push(Span::styled(" ", footer_value_style(theme)));
    left_spans.push(Span::styled(
        state.footer_status.summary.clone(),
        phase_style(state.phase, theme),
    ));

    if let Some(detail) = &state.footer_status.detail {
        left_spans.push(Span::styled(" · ", footer_dim_style(theme)));
        left_spans.push(Span::styled(detail.clone(), footer_muted_style(theme)));
    }

    if let Some(active_tool_call_id) = &state.active_tool_call_id {
        left_spans.push(Span::styled(" · active ", footer_dim_style(theme)));
        left_spans.push(Span::styled(
            active_tool_call_id.clone(),
            footer_value_style(theme),
        ));
    }

    let right_line = Line::from(vec![
        Span::styled("model ", footer_dim_style(theme)),
        Span::styled(state.model_label.clone(), footer_value_style(theme)),
        Span::styled(" · permission ", footer_dim_style(theme)),
        Span::styled(
            state.permission_mode_label.clone(),
            footer_value_style(theme),
        ),
        Span::styled(" · /help commands · exit to quit", footer_dim_style(theme)),
    ]);

    let right_width = right_line.width() as u16;
    let left_width = area.width.saturating_sub(right_width.saturating_add(2));

    if left_width > 0 {
        frame.render_widget(
            Paragraph::new(Line::from(left_spans)).style(theme.app_style()),
            Rect::new(area.x, area.y, left_width, 1),
        );
    }

    frame.render_widget(
        Paragraph::new(right_line)
            .style(theme.app_style())
            .alignment(Alignment::Right),
        area,
    );
}

fn transcript_lines(state: &TuiState, theme: Theme, width: usize) -> Vec<Line<'static>> {
    let mut lines = Vec::new();

    for (index, item) in state.timeline.items().iter().enumerate() {
        if index > 0 {
            lines.push(Line::from(""));
        }

        match item {
            TimelineItem::User(message) => {
                push_user_message_lines(&mut lines, message, theme, width)
            }
            TimelineItem::Assistant(message) => push_assistant_message_lines(
                &mut lines,
                message_text(message),
                message.streaming,
                theme,
            ),
            TimelineItem::Tool(tool) => push_tool_lines(&mut lines, tool, theme),
            TimelineItem::Permission(permission) => {
                push_permission_lines(&mut lines, permission, theme)
            }
            TimelineItem::Error(error) => push_error_lines(&mut lines, error, theme),
            TimelineItem::Notice(notice) => push_notice_lines(&mut lines, notice, theme),
        }
    }

    lines
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
        if raw.is_empty() {
            pushed = true;
            push_user_card_line(lines, "", width, theme);
            continue;
        }

        for content in wrap_line_for_width(raw, content_width) {
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
    let bar_style = card_bar_style(theme.user, theme.element_bg);
    let pad_style = user_prompt_padding_style(theme);

    let mut line = Line::from(vec![
        Span::styled(ACCENT_BAR_GLYPH, bar_style),
        Span::styled("  ", pad_style),
        Span::styled(content.to_string(), panel_style),
    ]);

    let used = line.width();
    if width > used {
        line.spans
            .push(Span::styled(" ".repeat(width - used), pad_style));
    } else {
        line.spans.push(Span::styled("  ", pad_style));
    }

    lines.push(line);
}

fn wrap_line_for_width(text: &str, width: usize) -> Vec<String> {
    if width == 0 {
        return vec![text.to_string()];
    }

    let mut chunks = Vec::new();
    let mut current = String::new();

    for ch in text.chars() {
        let mut candidate = current.clone();
        candidate.push(ch);
        if !current.is_empty() && Line::from(candidate.as_str()).width() > width {
            chunks.push(current);
            current = ch.to_string();
        } else {
            current = candidate;
        }
    }

    if !current.is_empty() {
        chunks.push(current);
    }

    if chunks.is_empty() {
        chunks.push(String::new());
    }

    chunks
}

fn push_assistant_message_lines(
    lines: &mut Vec<Line<'static>>,
    text: &str,
    _streaming: bool,
    theme: Theme,
) {
    let mut pushed = false;
    for content in text.lines() {
        pushed = true;
        lines.push(Line::from(vec![
            Span::styled("   ", theme.app_style()),
            Span::styled(content.to_string(), theme.app_style()),
        ]));
    }

    if !pushed {
        lines.push(Line::from(Span::styled("   …", root_muted_style(theme))));
    }
}

fn push_tool_lines(lines: &mut Vec<Line<'static>>, tool: &ToolView, theme: Theme) {
    let accent = tool_status_color(tool.status, theme);
    let title_style = element_title_style(theme);
    let muted = element_muted_style(theme);
    let body = theme.element_style();

    lines.push(Line::from(vec![
        Span::styled(ACCENT_BAR_GLYPH, card_bar_style(accent, theme.element_bg)),
        Span::styled("  # ", muted),
        Span::styled(tool.name.clone(), title_style),
    ]));
    lines.push(Line::from(vec![
        Span::styled(ACCENT_BAR_GLYPH, card_bar_style(accent, theme.element_bg)),
        Span::styled("  tool ", element_dim_style(theme)),
        Span::styled(
            tool_status_label(tool.status),
            tool_status_style(tool.status, theme),
        ),
        Span::styled(" · ", element_dim_style(theme)),
        Span::styled(tool.summary.clone(), body),
    ]));

    push_card_key_value(
        lines,
        "call",
        &tool.call_id,
        accent,
        theme.element_bg,
        theme,
    );
    push_card_optional_field(
        lines,
        "args",
        tool.arguments.as_deref(),
        accent,
        theme.element_bg,
        theme,
    );
    push_card_optional_field(
        lines,
        "out",
        tool.output.as_deref(),
        accent,
        theme.element_bg,
        theme,
    );
}

fn push_permission_lines(
    lines: &mut Vec<Line<'static>>,
    permission: &PermissionView,
    theme: Theme,
) {
    let accent = permission_status_color(permission.status, theme);

    lines.push(Line::from(vec![
        Span::styled(ACCENT_BAR_GLYPH, card_bar_style(accent, theme.elevated_bg)),
        Span::styled("  permission ", elevated_muted(theme)),
        Span::styled(
            permission.status.label(),
            permission_status_style(permission.status, theme),
        ),
        Span::styled(" · ", elevated_muted(theme)),
        Span::styled(permission.tool_name.clone(), elevated_title_style(theme)),
    ]));
    lines.push(Line::from(vec![
        Span::styled(ACCENT_BAR_GLYPH, card_bar_style(accent, theme.elevated_bg)),
        Span::styled("  ", inline_elevated(theme)),
        Span::styled(permission.summary.clone(), inline_elevated(theme)),
    ]));

    push_card_key_value(
        lines,
        "call",
        &permission.call_id,
        accent,
        theme.elevated_bg,
        theme,
    );
    push_card_optional_field(
        lines,
        "args",
        permission.arguments.as_deref(),
        accent,
        theme.elevated_bg,
        theme,
    );
    push_card_optional_field(
        lines,
        "why",
        permission.rationale.as_deref(),
        accent,
        theme.elevated_bg,
        theme,
    );
    push_card_optional_field(
        lines,
        "resolution",
        permission.resolution_reason.as_deref(),
        accent,
        theme.elevated_bg,
        theme,
    );
}

fn push_error_lines(lines: &mut Vec<Line<'static>>, error: &ErrorView, theme: Theme) {
    let accent = theme.error;

    lines.push(Line::from(vec![
        Span::styled(ACCENT_BAR_GLYPH, card_bar_style(accent, theme.elevated_bg)),
        Span::styled("  error ", elevated_muted(theme)),
        Span::styled(error.message.clone(), elevated_error_style(theme)),
    ]));

    push_card_optional_field(
        lines,
        "details",
        error.details.as_deref(),
        accent,
        theme.elevated_bg,
        theme,
    );
}

fn push_notice_lines(lines: &mut Vec<Line<'static>>, notice: &NoticeView, theme: Theme) {
    lines.push(Line::from(vec![
        Span::styled("  ", theme.app_style()),
        Span::styled(notice.message.clone(), root_dim_style(theme)),
    ]));
}

fn push_optional_field(
    lines: &mut Vec<Line<'static>>,
    label: &str,
    value: Option<&str>,
    theme: Theme,
) {
    if let Some(value) = value.filter(|value| !value.is_empty()) {
        push_multiline_key_value(
            lines,
            label,
            value,
            theme.muted_style(),
            theme.surface_style(),
        );
    }
}

fn push_optional_field_elevated(
    lines: &mut Vec<Line<'static>>,
    label: &str,
    value: Option<&str>,
    theme: Theme,
) {
    if let Some(value) = value.filter(|value| !value.is_empty()) {
        push_multiline_key_value(
            lines,
            label,
            value,
            elevated_muted(theme),
            inline_elevated(theme),
        );
    }
}

fn push_card_optional_field(
    lines: &mut Vec<Line<'static>>,
    label: &str,
    value: Option<&str>,
    accent: Color,
    bg: Color,
    theme: Theme,
) {
    if let Some(value) = value.filter(|value| !value.is_empty()) {
        push_card_multiline_key_value(lines, label, value, accent, bg, theme);
    }
}

fn push_card_key_value(
    lines: &mut Vec<Line<'static>>,
    label: &str,
    value: &str,
    accent: Color,
    bg: Color,
    theme: Theme,
) {
    push_card_multiline_key_value(lines, label, value, accent, bg, theme);
}

fn push_card_multiline_key_value(
    lines: &mut Vec<Line<'static>>,
    label: &str,
    value: &str,
    accent: Color,
    bg: Color,
    theme: Theme,
) {
    let (label_style, value_style) = if bg == theme.elevated_bg {
        (elevated_muted(theme), inline_elevated(theme))
    } else {
        (element_muted_style(theme), theme.element_style())
    };
    let mut iter = value.lines().peekable();
    let mut count = 0usize;

    while let Some(part) = iter.next() {
        if count >= MAX_FIELD_LINES {
            lines.push(Line::from(vec![
                Span::styled(ACCENT_BAR_GLYPH, card_bar_style(accent, bg)),
                Span::styled("  …      ", label_style),
                Span::styled("truncated", label_style),
            ]));
            break;
        }

        let field_label = if count == 0 { label } else { "" };
        lines.push(Line::from(vec![
            Span::styled(ACCENT_BAR_GLYPH, card_bar_style(accent, bg)),
            Span::styled(format!("  {field_label:<7}"), label_style),
            Span::styled(part.to_string(), value_style),
        ]));
        count += 1;

        if iter.peek().is_none() {
            break;
        }
    }
}

fn push_key_value(
    lines: &mut Vec<Line<'static>>,
    label: &str,
    value: &str,
    label_style: Style,
    value_style: Style,
) {
    lines.push(Line::from(vec![
        Span::styled(format!("  {label:<10}"), label_style),
        Span::styled(value.to_string(), value_style),
    ]));
}

fn push_multiline_key_value(
    lines: &mut Vec<Line<'static>>,
    label: &str,
    value: &str,
    label_style: Style,
    value_style: Style,
) {
    let mut iter = value.lines().peekable();
    let mut count = 0usize;

    while let Some(part) = iter.next() {
        if count >= MAX_FIELD_LINES {
            lines.push(Line::from(vec![
                Span::styled("  …         ", label_style),
                Span::styled("output truncated for display", label_style),
            ]));
            break;
        }

        let field_label = if count == 0 { label } else { "" };
        lines.push(Line::from(vec![
            Span::styled(format!("  {field_label:<10}"), label_style),
            Span::styled(part.to_string(), value_style),
        ]));
        count += 1;

        if iter.peek().is_none() {
            break;
        }
    }

    if count == 0 {
        push_key_value(lines, label, "", label_style, value_style);
    }
}

fn render_surface_fill(frame: &mut Frame<'_>, area: Rect, style: Style) {
    if !area.is_empty() {
        frame.render_widget(Block::new().style(style), area);
    }
}

fn render_accent_bar(frame: &mut Frame<'_>, area: Rect, style: Style) {
    if area.is_empty() {
        return;
    }

    let bar_area = Rect::new(area.x, area.y, 1.min(area.width), area.height);
    let lines = vec![Line::from(Span::styled(ACCENT_BAR_GLYPH, style)); area.height as usize];
    frame.render_widget(Paragraph::new(lines).style(style), bar_area);
}

fn render_composer_cap(frame: &mut Frame<'_>, area: Rect, theme: Theme) {
    if area.height == 0 || area.width == 0 {
        return;
    }

    let cap_y = area.y + area.height.saturating_sub(1);
    let cap_area = Rect::new(area.x, cap_y, area.width, 1);
    frame.render_widget(Block::new().style(theme.app_style()), cap_area);

    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            PROMPT_BOTTOM_LEFT_GLYPH,
            Style::default().fg(theme.user).bg(theme.root_bg),
        ))),
        Rect::new(area.x, cap_y, 1.min(area.width), 1),
    );

    let cap_width = area.width.saturating_sub(1);
    if cap_width > 0 {
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(
                PROMPT_BOTTOM_CAP_GLYPH.repeat(cap_width as usize),
                Style::default().fg(theme.element_bg).bg(theme.root_bg),
            ))),
            Rect::new(area.x + 1, cap_y, cap_width, 1),
        );
    }
}

fn inset_left(area: Rect, amount: u16) -> Rect {
    let inset = amount.min(area.width);
    Rect::new(
        area.x + inset,
        area.y,
        area.width.saturating_sub(inset),
        area.height,
    )
}

fn wordmark_style(theme: Theme) -> Style {
    Style::default()
        .fg(theme.text)
        .bg(theme.root_bg)
        .add_modifier(Modifier::BOLD)
}

fn wordmark_shadow_style(theme: Theme) -> Style {
    Style::default()
        .fg(theme.notice)
        .bg(theme.root_bg)
        .add_modifier(Modifier::DIM)
}

fn root_muted_style(theme: Theme) -> Style {
    Style::default().fg(theme.muted_text).bg(theme.root_bg)
}

fn root_dim_style(theme: Theme) -> Style {
    Style::default().fg(theme.dim_text).bg(theme.root_bg)
}

fn user_prompt_panel_style(theme: Theme) -> Style {
    Style::default().fg(theme.text).bg(theme.element_bg)
}

fn user_prompt_padding_style(theme: Theme) -> Style {
    Style::default().fg(theme.muted_text).bg(theme.element_bg)
}

fn element_title_style(theme: Theme) -> Style {
    Style::default()
        .fg(theme.text)
        .bg(theme.element_bg)
        .add_modifier(Modifier::BOLD)
}

fn element_accent_style(theme: Theme) -> Style {
    Style::default()
        .fg(theme.user)
        .bg(theme.element_bg)
        .add_modifier(Modifier::BOLD)
}

fn element_muted_style(theme: Theme) -> Style {
    Style::default().fg(theme.muted_text).bg(theme.element_bg)
}

fn element_dim_style(theme: Theme) -> Style {
    Style::default().fg(theme.dim_text).bg(theme.element_bg)
}

fn card_bar_style(accent: Color, bg: Color) -> Style {
    Style::default().fg(accent).bg(bg)
}

fn inline_elevated(theme: Theme) -> Style {
    Style::default().fg(theme.text).bg(theme.elevated_bg)
}

fn elevated_title_style(theme: Theme) -> Style {
    Style::default()
        .fg(theme.text)
        .bg(theme.elevated_bg)
        .add_modifier(Modifier::BOLD)
}

fn elevated_muted(theme: Theme) -> Style {
    Style::default().fg(theme.muted_text).bg(theme.elevated_bg)
}

fn elevated_error_style(theme: Theme) -> Style {
    Style::default().fg(theme.error).bg(theme.elevated_bg)
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

fn scanner_frame_spans(frame: usize, theme: Theme) -> Vec<Span<'static>> {
    scanner_cells(frame, theme)
        .into_iter()
        .map(|(glyph, color)| {
            Span::styled(
                glyph.to_string(),
                Style::default().fg(color).bg(theme.root_bg),
            )
        })
        .collect()
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
                scanner_trail_color(distance, theme)
            } else {
                Color::Rgb(30, 50, 58)
            };
            (glyph, color)
        })
        .collect()
}

fn scanner_trail_color(distance: usize, theme: Theme) -> Color {
    match distance {
        0 => theme.user,
        1 => Color::Rgb(85, 188, 230),
        2 => Color::Rgb(58, 123, 149),
        3 => Color::Rgb(44, 86, 103),
        4 => Color::Rgb(35, 63, 73),
        _ => Color::Rgb(29, 47, 54),
    }
}

fn tool_status_style(status: ToolExecutionStatus, theme: Theme) -> Style {
    Style::default()
        .fg(tool_status_color(status, theme))
        .bg(theme.element_bg)
        .add_modifier(Modifier::BOLD)
}

fn tool_status_color(status: ToolExecutionStatus, theme: Theme) -> Color {
    match status {
        ToolExecutionStatus::Running => theme.warning,
        ToolExecutionStatus::Succeeded => theme.success,
        ToolExecutionStatus::Failed => theme.error,
    }
}

fn tool_status_label(status: ToolExecutionStatus) -> &'static str {
    match status {
        ToolExecutionStatus::Running => "running",
        ToolExecutionStatus::Succeeded => "succeeded",
        ToolExecutionStatus::Failed => "failed",
    }
}

fn permission_status_style(status: PermissionPromptStatus, theme: Theme) -> Style {
    Style::default()
        .fg(permission_status_color(status, theme))
        .bg(theme.elevated_bg)
        .add_modifier(Modifier::BOLD)
}

fn permission_status_color(status: PermissionPromptStatus, theme: Theme) -> Color {
    match status {
        PermissionPromptStatus::Pending => theme.approval,
        PermissionPromptStatus::Approved => theme.success,
        PermissionPromptStatus::Denied => theme.error,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tui::{
        AppEvent, AssistantDeltaEvent, ErrorEvent, PermissionRequestEvent, ToolFinishedEvent,
        ToolOutcome, ToolStartedEvent, UserMessageEvent,
    };
    use ratatui::{Terminal, backend::TestBackend};

    fn draw_to_string(state: &TuiState, width: u16, height: u16) -> String {
        let backend = TestBackend::new(width, height);
        let mut terminal = Terminal::new(backend).expect("test terminal is created");

        terminal
            .draw(|frame| render(frame, state))
            .expect("render succeeds");

        terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|cell| cell.symbol())
            .collect()
    }

    #[test]
    fn empty_welcome_view_renders_wordmark_without_panic() {
        let state = TuiState::new("gpt-5.5", "default");

        let rendered = draw_to_string(&state, 80, 20);
        assert!(
            rendered.contains("█    █▀▀█ ▀█▀▀") || rendered.contains("LETCODE"),
            "{rendered}"
        );

        let tiny = draw_to_string(&state, 10, 2);
        assert!(!tiny.is_empty());
    }

    #[test]
    fn user_and_assistant_timeline_content_appears() {
        let mut state = TuiState::default();
        state.apply_event(AppEvent::UserMessage(UserMessageEvent::new("hello tui")));
        state.apply_event(AppEvent::AssistantDelta(AssistantDeltaEvent::new(
            "hi there",
        )));
        state.apply_event(AppEvent::AssistantDone { message_id: None });

        let rendered = draw_to_string(&state, 90, 20);

        assert!(rendered.contains("hello tui"), "{rendered}");
        assert!(rendered.contains("hi there"), "{rendered}");
        assert!(rendered.contains(ACCENT_BAR_GLYPH), "{rendered}");
        assert!(!rendered.contains("streaming"), "{rendered}");
    }

    #[test]
    fn pending_permission_prompt_displays_hint_and_tool_summary() {
        let mut state = TuiState::default();
        let mut request = PermissionRequestEvent::new("call-1", "bash", "cargo test all");
        request.arguments = Some("cargo test".into());
        request.rationale = Some("tests need confirmation".into());
        state.apply_event(AppEvent::PermissionRequested(request));

        let rendered = draw_to_string(&state, 96, 24);

        assert!(rendered.contains("Permission required"), "{rendered}");
        assert!(rendered.contains("approve"), "{rendered}");
        assert!(rendered.contains("deny"), "{rendered}");
        assert!(rendered.contains("bash"), "{rendered}");
        assert!(rendered.contains("cargo test all"), "{rendered}");
    }

    #[test]
    fn footer_contains_model_and_permission_mode_labels() {
        let mut state = TuiState::new("gpt-5.5-mini", "safe");
        state.set_footer("Ready", Some("detail text".into()));

        let rendered = draw_to_string(&state, 100, 16);

        assert!(rendered.contains("model"), "{rendered}");
        assert!(rendered.contains("gpt-5.5-mini"), "{rendered}");
        assert!(rendered.contains("permission"), "{rendered}");
        assert!(rendered.contains("safe"), "{rendered}");
    }

    #[test]
    fn tool_cards_and_errors_use_structured_timeline_fields() {
        let mut state = TuiState::default();
        let mut started = ToolStartedEvent::new("tool-7", "bash", "run cargo check");
        started.arguments = Some("cargo check".into());
        state.apply_event(AppEvent::ToolStarted(started));
        let mut finished =
            ToolFinishedEvent::new("tool-7", "bash", "run cargo check", ToolOutcome::Failure);
        finished.output = Some("compiler said no".into());
        state.apply_event(AppEvent::ToolFinished(finished));
        let mut error = ErrorEvent::new("render problem");
        error.details = Some("missing widget area".into());
        state.apply_event(AppEvent::Error(error));

        let rendered = draw_to_string(&state, 100, 24);

        assert!(rendered.contains("tool"), "{rendered}");
        assert!(rendered.contains("bash"), "{rendered}");
        assert!(rendered.contains("cargo check"), "{rendered}");
        assert!(rendered.contains("compiler said no"), "{rendered}");
        assert!(rendered.contains("error"), "{rendered}");
        assert!(rendered.contains("render problem"), "{rendered}");
    }
}
