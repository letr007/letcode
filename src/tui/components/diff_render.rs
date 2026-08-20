//! Diff-style tool card rendering (write/append/edit/patch).

use ratatui::style::{Modifier, Style};

use super::semantic_spans::*;
use crate::tui::{
    measure::display_width,
    theme::Theme,
    timeline::ToolView,
    transcript_render::{Break, SemanticLine, SemanticSpan},
};

pub(super) const DIFF_CARD_HEADER_ARROW: &str = "←";

pub(super) fn render_write_diff_lines(
    tool: &ToolView,
    theme: Theme,
    width: usize,
) -> Vec<SemanticLine<Style>> {
    let Some(args) = tool_arguments(tool) else {
        return Vec::new();
    };
    let path = args
        .get("path")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("file");
    let content = args
        .get("content")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");

    let mut diff = String::new();
    if content.is_empty() {
        diff.push_str("+\n");
    } else {
        for line in content.lines() {
            diff.push('+');
            diff.push_str(line);
            diff.push('\n');
        }
    }

    render_diff_block(
        diff_card_header_title("Write", &[path.to_string()]),
        &diff,
        None,
        theme,
        width,
    )
}

pub(super) fn render_append_diff_lines(
    tool: &ToolView,
    theme: Theme,
    width: usize,
) -> Vec<SemanticLine<Style>> {
    let Some(args) = tool_arguments(tool) else {
        return Vec::new();
    };
    let path = args
        .get("path")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("file");
    let content = args
        .get("content")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");

    let mut diff = String::new();
    if content.is_empty() {
        diff.push_str("+\n");
    } else {
        for line in content.lines() {
            diff.push('+');
            diff.push_str(line);
            diff.push('\n');
        }
    }

    render_diff_block(
        diff_card_header_title("Append", &[path.to_string()]),
        &diff,
        None,
        theme,
        width,
    )
}

pub(super) fn render_edit_diff_lines(
    tool: &ToolView,
    theme: Theme,
    width: usize,
) -> Vec<SemanticLine<Style>> {
    let Some(args) = tool_arguments(tool) else {
        return Vec::new();
    };
    let Some(edits) = args.get("edits").and_then(serde_json::Value::as_array) else {
        return Vec::new();
    };

    let mut file_diffs = Vec::<(String, String)>::new();
    for edit in edits {
        let path = edit
            .get("path")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("file");
        let find = edit
            .get("find")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");
        let replace = edit
            .get("replace")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");

        if find.is_empty() && replace.is_empty() {
            continue;
        }

        let diff = match file_diffs.iter_mut().find(|(file, _)| file == path) {
            Some((_, diff)) => diff,
            None => {
                file_diffs.push((path.to_string(), String::new()));
                &mut file_diffs.last_mut().expect("just pushed file diff").1
            }
        };
        for line in find.lines() {
            diff.push('-');
            diff.push_str(&terminal_safe_text(line));
            diff.push('\n');
        }
        for line in replace.lines() {
            diff.push('+');
            diff.push_str(&terminal_safe_text(line));
            diff.push('\n');
        }
    }

    file_diffs
        .into_iter()
        .flat_map(|(path, diff)| {
            render_diff_block(
                diff_card_header_title("Patch", &[path]),
                &diff,
                None,
                theme,
                width,
            )
        })
        .collect()
}

pub(super) fn render_diff_block(
    title: String,
    diff: &str,
    truncated: Option<&serde_json::Value>,
    theme: Theme,
    width: usize,
) -> Vec<SemanticLine<Style>> {
    if width == 0 {
        return Vec::new();
    }

    // Title carries user-controlled paths/labels; keep it display-safe too.
    let title = terminal_safe_text(&title);
    let mut lines = Vec::new();
    let header = if truncated
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
    {
        format!("{title} · truncated")
    } else {
        title
    };
    lines.push(render_diff_card_spacer_line(theme, width));
    lines.push(render_diff_card_header_line(&header, theme, width));
    lines.push(render_diff_card_spacer_line(theme, width));

    let mut body = diff
        .lines()
        .filter(|line| !is_diff_file_header_line(line))
        .take(max_body_lines().saturating_add(1))
        .collect::<Vec<_>>();
    if body.len() > max_body_lines() {
        body.pop();
        body.push("… output clipped in TUI");
    }

    if diff_uses_side_by_side_layout(width)
        && body.iter().any(|line| {
            (line.starts_with('+') || line.starts_with('-')) && !is_diff_file_header_line(line)
        })
    {
        lines.extend(render_side_by_side_diff_body(&body, theme, width));
    } else {
        let mut state = DiffLineNumbers::default();
        for line in body {
            let (old_no, new_no) = state.next(line);
            lines.push(render_diff_card_body_line(
                old_no,
                new_no,
                line,
                diff_line_style(line, theme),
                theme,
                width,
            ));
        }
    }
    lines.push(render_diff_card_spacer_line(theme, width));
    lines
}

pub(super) const DIFF_SIDE_BY_SIDE_SEPARATOR: &str = " │ ";

pub(super) const DIFF_SIDE_BY_SIDE_MIN_PANEL_WIDTH: usize = 20;

pub(super) fn diff_uses_side_by_side_layout(width: usize) -> bool {
    let content_width = shell_card_content_width(width);
    content_width
        >= DIFF_SIDE_BY_SIDE_MIN_PANEL_WIDTH * 2 + display_width(DIFF_SIDE_BY_SIDE_SEPARATOR)
}

pub(super) fn render_side_by_side_diff_body(
    body: &[&str],
    theme: Theme,
    width: usize,
) -> Vec<SemanticLine<Style>> {
    let content_width = shell_card_content_width(width);
    let panel_width = content_width.saturating_sub(display_width(DIFF_SIDE_BY_SIDE_SEPARATOR)) / 2;
    let mut lines = Vec::new();
    let mut state = DiffLineNumbers::default();
    let mut index = 0;

    while index < body.len() {
        let line = body[index];
        if line.starts_with('-') && !is_diff_file_header_line(line) {
            let mut removed = Vec::new();
            while index < body.len()
                && body[index].starts_with('-')
                && !is_diff_file_header_line(body[index])
            {
                let (old_no, _) = state.next(body[index]);
                removed.push((old_no, body[index]));
                index += 1;
            }

            let mut added = Vec::new();
            while index < body.len()
                && body[index].starts_with('+')
                && !is_diff_file_header_line(body[index])
            {
                let (_, new_no) = state.next(body[index]);
                added.push((new_no, body[index]));
                index += 1;
            }

            for row in 0..removed.len().max(added.len()) {
                lines.push(render_side_by_side_diff_line(
                    removed.get(row).copied(),
                    added.get(row).copied(),
                    panel_width,
                    theme,
                    width,
                ));
            }
            continue;
        }

        if line.starts_with('+') && !is_diff_file_header_line(line) {
            let mut added = Vec::new();
            while index < body.len()
                && body[index].starts_with('+')
                && !is_diff_file_header_line(body[index])
            {
                let (_, new_no) = state.next(body[index]);
                added.push((new_no, body[index]));
                index += 1;
            }
            for added in added {
                lines.push(render_side_by_side_diff_line(
                    None,
                    Some(added),
                    panel_width,
                    theme,
                    width,
                ));
            }
            continue;
        }

        let (old_no, new_no) = state.next(line);
        if line.starts_with(' ') {
            lines.push(render_side_by_side_diff_line(
                Some((old_no, line)),
                Some((new_no, line)),
                panel_width,
                theme,
                width,
            ));
        } else {
            lines.push(render_diff_card_body_line(
                old_no,
                new_no,
                line,
                diff_line_style(line, theme),
                theme,
                width,
            ));
        }
        index += 1;
    }

    lines
}

pub(super) fn render_side_by_side_diff_line(
    removed: Option<(Option<usize>, &str)>,
    added: Option<(Option<usize>, &str)>,
    panel_width: usize,
    theme: Theme,
    width: usize,
) -> SemanticLine<Style> {
    let mut segments = render_diff_side(removed, panel_width, theme);
    segments.push(SemanticSpan::decoration(
        DIFF_SIDE_BY_SIDE_SEPARATOR,
        diff_meta_style(theme),
    ));
    segments.extend(render_diff_side(added, panel_width, theme));
    render_card_line_with_guide(
        &segments,
        diff_header_fill_style(theme),
        theme.card_guide(),
        theme,
        width,
        Break::HardBreak,
    )
}

pub(super) fn render_diff_side(
    entry: Option<(Option<usize>, &str)>,
    width: usize,
    theme: Theme,
) -> Vec<SemanticSpan<Style>> {
    let Some((number, content)) = entry else {
        return vec![SemanticSpan::decoration(
            " ".repeat(width),
            diff_header_fill_style(theme),
        )];
    };

    let gutter_style = Style::default().fg(theme.muted_text).bg(theme.card_bg());
    let gutter_pad_style = Style::default().bg(theme.card_bg());
    let content_style = diff_line_style(content, theme);
    let pad_style = Style::default().bg(content_style.bg.unwrap_or(theme.card_bg()));
    let (marker, body, marker_style) = diff_marker_and_body(content, theme);
    // Strip control/ANSI from the diff body before it reaches display cells; the
    // on-disk content itself stays untouched — filtering is presentation-only.
    let body = terminal_safe_text(&body);
    let clipped_notice = content == "… output clipped in TUI";
    let marker_span = if clipped_notice {
        SemanticSpan::decoration(marker, marker_style)
    } else {
        SemanticSpan::source(marker, marker_style)
    };
    let body_span = if clipped_notice {
        SemanticSpan::decoration(body, content_style)
    } else {
        SemanticSpan::source(body, content_style)
    };
    let mut spans = clip_semantic_spans(
        vec![
            SemanticSpan::decoration(diff_line_number(number), gutter_style),
            SemanticSpan::decoration(" ", gutter_pad_style),
            marker_span,
            SemanticSpan::decoration(" ", pad_style),
            body_span,
        ],
        width,
    );
    let used = spans
        .iter()
        .map(|span| display_width(&span.text))
        .sum::<usize>();
    if used < width {
        spans.push(SemanticSpan::decoration(
            " ".repeat(width - used),
            pad_style,
        ));
    }
    spans
}

pub(super) fn diff_line_style(line: &str, theme: Theme) -> Style {
    if line.starts_with("diff --git") || line.starts_with("index ") {
        diff_meta_style(theme).add_modifier(Modifier::BOLD)
    } else if line.starts_with("+++") || line.starts_with("---") {
        diff_meta_style(theme)
    } else if line.starts_with('+') {
        Style::default().fg(theme.text).bg(theme.diff_add_bg)
    } else if line.starts_with('-') {
        Style::default().fg(theme.text).bg(theme.diff_delete_bg)
    } else if line.starts_with("@@") {
        Style::default().fg(theme.user).bg(theme.diff_hunk_bg)
    } else {
        Style::default().fg(theme.text).bg(theme.card_bg())
    }
}

pub(super) fn is_diff_file_header_line(line: &str) -> bool {
    line.starts_with("---") || line.starts_with("+++")
}

pub(super) fn diff_meta_style(theme: Theme) -> Style {
    Style::default().fg(theme.muted_text).bg(theme.card_bg())
}

pub(super) fn render_diff_card_header_line(
    title: &str,
    theme: Theme,
    width: usize,
) -> SemanticLine<Style> {
    let text = format!(" {DIFF_CARD_HEADER_ARROW} {title}");
    render_card_line(
        &[(text, diff_header_style(theme))],
        diff_header_fill_style(theme),
        theme,
        width,
    )
}

pub(super) fn render_diff_card_spacer_line(theme: Theme, width: usize) -> SemanticLine<Style> {
    render_card_line(&[], diff_header_fill_style(theme), theme, width)
}

pub(super) fn render_diff_card_body_line(
    old_no: Option<usize>,
    new_no: Option<usize>,
    content: &str,
    content_style: Style,
    theme: Theme,
    width: usize,
) -> SemanticLine<Style> {
    let gutter_style = Style::default().fg(theme.muted_text).bg(theme.card_bg());
    let number = diff_line_number(new_no.or(old_no));
    let bg = content_style.bg.unwrap_or(theme.card_bg());
    let pad_style = Style::default().bg(bg);
    let gutter_pad_style = Style::default().bg(theme.card_bg());
    let (marker, body, marker_style) = diff_marker_and_body(content, theme);
    // Strip control/ANSI from the diff body before it reaches display cells; the
    // on-disk content itself stays untouched — filtering is presentation-only.
    let body = terminal_safe_text(&body);
    let clipped_notice = content == "… output clipped in TUI";
    let marker_span = if clipped_notice {
        SemanticSpan::decoration(marker, marker_style)
    } else {
        SemanticSpan::source(marker, marker_style)
    };
    let body_span = if clipped_notice {
        SemanticSpan::decoration(body, content_style)
    } else {
        SemanticSpan::source(body, content_style)
    };
    let segments = vec![
        SemanticSpan::decoration("", gutter_pad_style),
        SemanticSpan::decoration(number, gutter_style),
        SemanticSpan::decoration(" ", gutter_pad_style),
        marker_span,
        SemanticSpan::decoration(" ", pad_style),
        body_span,
    ];
    render_card_line_with_guide(
        &segments,
        content_style,
        theme.card_guide(),
        theme,
        width,
        Break::HardBreak,
    )
}

pub(super) fn diff_marker_and_body(content: &str, theme: Theme) -> (String, String, Style) {
    match content.chars().next() {
        Some('+') if !content.starts_with("+++") => (
            "+".to_string(),
            content.chars().skip(1).collect(),
            Style::default().fg(theme.success).bg(theme.card_bg()),
        ),
        Some('-') if !content.starts_with("---") => (
            "-".to_string(),
            content.chars().skip(1).collect(),
            Style::default().fg(theme.error).bg(theme.card_bg()),
        ),
        _ => (
            " ".to_string(),
            content.to_string(),
            Style::default().fg(theme.muted_text).bg(theme.card_bg()),
        ),
    }
}

pub(super) fn diff_header_style(theme: Theme) -> Style {
    Style::default().fg(theme.muted_text).bg(theme.card_bg())
}

pub(super) fn diff_header_fill_style(theme: Theme) -> Style {
    Style::default().bg(theme.card_bg())
}

pub(super) fn diff_line_number(number: Option<usize>) -> String {
    match number {
        Some(value) => format!("{:>3}", value),
        None => "   ".to_string(),
    }
}

#[derive(Default)]
pub(super) struct DiffLineNumbers {
    old_next: Option<usize>,
    new_next: Option<usize>,
}

impl DiffLineNumbers {
    fn next(&mut self, line: &str) -> (Option<usize>, Option<usize>) {
        if let Some((old_start, new_start)) = parse_hunk_header(line) {
            self.old_next = Some(old_start);
            self.new_next = Some(new_start);
            return (None, None);
        }

        if line.starts_with("diff --git")
            || line.starts_with("index ")
            || line.starts_with("---")
            || line.starts_with("+++")
        {
            self.old_next = None;
            self.new_next = None;
            return (None, None);
        }

        match line.chars().next() {
            Some('+') => {
                let next = self.new_next.get_or_insert(1);
                let current = *next;
                *next += 1;
                (None, Some(current))
            }
            Some('-') => {
                let next = self.old_next.get_or_insert(1);
                let current = *next;
                *next += 1;
                (Some(current), None)
            }
            Some(' ') => {
                let old = {
                    let next = self.old_next.get_or_insert(1);
                    let current = *next;
                    *next += 1;
                    current
                };
                let new = {
                    let next = self.new_next.get_or_insert(1);
                    let current = *next;
                    *next += 1;
                    current
                };
                (Some(old), Some(new))
            }
            _ => (None, None),
        }
    }
}

pub(super) fn parse_hunk_header(line: &str) -> Option<(usize, usize)> {
    if !line.starts_with("@@") {
        return None;
    }

    let mut parts = line.split_whitespace();
    let _ = parts.next()?;
    let old = parts.next()?;
    let new = parts.next()?;
    Some((parse_hunk_range_start(old)?, parse_hunk_range_start(new)?))
}

pub(super) fn parse_hunk_range_start(part: &str) -> Option<usize> {
    let trimmed = part.strip_prefix(['-', '+'])?;
    trimmed
        .split(',')
        .next()
        .and_then(|value| value.parse::<usize>().ok())
}

pub(super) fn diff_card_header_title(label: &str, paths: &[String]) -> String {
    match paths.first() {
        Some(first) if paths.len() > 1 => format!("{label} {} +{}", first, paths.len() - 1),
        Some(first) => format!("{label} {first}"),
        None => label.to_string(),
    }
}
