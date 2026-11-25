use ratatui::style::Style;
use similar::{ChangeTag, TextDiff};
use unicode_segmentation::UnicodeSegmentation;

use super::semantic_spans::*;
use crate::tui::{
    measure::display_width,
    theme::Theme,
    timeline::ToolView,
    transcript_render::{Break, CopyMode, Document, Line, SourceRange, Span},
};

const DIFF_CARD_HEADER_ARROW: &str = "←";
const DIFF_EXPAND_HINT: &str = "… click to expand for details";
pub(super) const DIFF_SIDE_BY_SIDE_SEPARATOR: &str = " │ ";
const DIFF_SIDE_BY_SIDE_MIN_PANEL_WIDTH: usize = 20;
const DIFF_TAB_SIZE: usize = 4;

struct DiffLine {
    kind: ChangeTag,
    text: String,
    old_number: Option<usize>,
    new_number: Option<usize>,
}

pub(super) fn render_write_diff_document(
    tool: &ToolView,
    theme: Theme,
    width: usize,
    expanded_output: bool,
) -> Document<Style> {
    render_content_document(tool, "Write", theme, width, expanded_output)
}

pub(super) fn render_append_diff_document(
    tool: &ToolView,
    theme: Theme,
    width: usize,
    expanded_output: bool,
) -> Document<Style> {
    render_content_document(tool, "Append", theme, width, expanded_output)
}

fn render_content_document(
    tool: &ToolView,
    label: &str,
    theme: Theme,
    width: usize,
    expanded_output: bool,
) -> Document<Style> {
    let Some(args) = tool_arguments(tool) else {
        return Document::default();
    };
    let path = args
        .get("path")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("file");
    let content = args
        .get("content")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");
    let lines = if content.is_empty() {
        vec![""]
    } else {
        content.lines().collect()
    };
    let body = lines
        .into_iter()
        .enumerate()
        .map(|(index, text)| DiffLine {
            kind: ChangeTag::Insert,
            text: text.to_string(),
            old_number: None,
            new_number: Some(index + 1),
        })
        .collect();
    render_diff_document(
        &format!("{label} {path}"),
        &[body],
        false,
        theme,
        width,
        expanded_output,
    )
}

pub(super) fn render_edit_diff_document(
    tool: &ToolView,
    theme: Theme,
    width: usize,
    expanded_output: bool,
) -> Document<Style> {
    let Some(args) = tool_arguments(tool) else {
        return Document::default();
    };
    let Some(edits) = args.get("edits").and_then(serde_json::Value::as_array) else {
        return Document::default();
    };
    let mut files = Vec::<(String, Vec<Vec<DiffLine>>)>::new();
    for edit in edits {
        let path = edit
            .get("path")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("file");
        let old = edit
            .get("find")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");
        let new = edit
            .get("replace")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");
        if old.is_empty() && new.is_empty() {
            continue;
        }
        let diff = TextDiff::from_lines(old, new);
        let lines = diff
            .iter_all_changes()
            .map(|change| DiffLine {
                kind: change.tag(),
                text: change.value().lines().next().unwrap_or("").to_string(),
                old_number: change.old_index().map(|index| index + 1),
                new_number: change.new_index().map(|index| index + 1),
            })
            .collect();
        match files.iter_mut().find(|(file, _)| file == path) {
            Some((_, groups)) => groups.push(lines),
            None => files.push((path.to_string(), vec![lines])),
        }
    }
    let mut document = Document::default();
    for (path, groups) in files {
        document.append(render_diff_document(
            &format!("Patch {path}"),
            &groups,
            true,
            theme,
            width,
            expanded_output,
        ));
    }
    document
}

fn render_diff_document(
    title: &str,
    groups: &[Vec<DiffLine>],
    compare: bool,
    theme: Theme,
    width: usize,
    expanded_output: bool,
) -> Document<Style> {
    let mut document = Document::default();
    if width == 0 {
        return document;
    }
    let fill = Style::default().bg(theme.card_bg());
    let meta = Style::default().fg(theme.muted_text).bg(theme.card_bg());
    document.push_semantic_line(render_card_line(&[], fill, theme, width));
    document.push_semantic_line(render_card_line(
        &[(
            format!(" {DIFF_CARD_HEADER_ARROW} {}", terminal_safe_text(title)),
            meta,
        )],
        fill,
        theme,
        width,
    ));
    document.push_semantic_line(render_card_line(&[], fill, theme, width));
    let content_width = width.saturating_sub(display_width(TOOL_GUIDE_GLYPH) + 2);
    let side_by_side = compare
        && content_width
            >= DIFF_SIDE_BY_SIDE_MIN_PANEL_WIDTH * 2 + display_width(DIFF_SIDE_BY_SIDE_SEPARATOR);
    let total = groups.iter().map(Vec::len).sum::<usize>();
    let mut remaining = if expanded_output {
        total
    } else {
        max_body_lines()
    };
    let number_width = groups
        .iter()
        .flatten()
        .flat_map(|line| [line.old_number, line.new_number])
        .flatten()
        .max()
        .unwrap_or(1)
        .to_string()
        .len()
        .max(3);
    for group in groups {
        let body = &group[..group.len().min(remaining)];
        remaining = remaining.saturating_sub(body.len());
        if side_by_side {
            render_comparison(&mut document, body, number_width, theme, width);
        } else {
            for line in body {
                let rows = render_diff_side(
                    &mut document,
                    Some(line),
                    false,
                    content_width,
                    number_width,
                    theme,
                );
                push_diff_rows(
                    &mut document,
                    rows,
                    if line.text.is_empty() {
                        Break::SoftWrap
                    } else {
                        Break::HardBreak
                    },
                    theme,
                    width,
                );
            }
        }
    }
    if !expanded_output && total > max_body_lines() {
        document.push_semantic_line(render_card_line(
            &[(DIFF_EXPAND_HINT.to_string(), meta)],
            fill,
            theme,
            width,
        ));
    }
    document.push_semantic_line(render_card_line(&[], fill, theme, width));
    document.finish();
    document
}

fn render_comparison(
    document: &mut Document<Style>,
    body: &[DiffLine],
    number_width: usize,
    theme: Theme,
    width: usize,
) {
    let mut index = 0;
    while index < body.len() {
        if body[index].kind == ChangeTag::Equal {
            render_diff_pair(
                document,
                Some(&body[index]),
                Some(&body[index]),
                number_width,
                theme,
                width,
            );
            index += 1;
            continue;
        }
        let start = index;
        while index < body.len() && body[index].kind != ChangeTag::Equal {
            index += 1;
        }
        let removed = body[start..index]
            .iter()
            .filter(|line| line.kind == ChangeTag::Delete)
            .collect::<Vec<_>>();
        let added = body[start..index]
            .iter()
            .filter(|line| line.kind == ChangeTag::Insert)
            .collect::<Vec<_>>();
        for row in 0..removed.len().max(added.len()) {
            render_diff_pair(
                document,
                removed.get(row).copied(),
                added.get(row).copied(),
                number_width,
                theme,
                width,
            );
        }
    }
}

fn render_diff_pair(
    document: &mut Document<Style>,
    old: Option<&DiffLine>,
    new: Option<&DiffLine>,
    number_width: usize,
    theme: Theme,
    width: usize,
) {
    let panel_width = shell_card_content_width(width)
        .saturating_sub(display_width(DIFF_SIDE_BY_SIDE_SEPARATOR))
        / 2;
    let left = render_diff_side(document, old, true, panel_width, number_width, theme);
    let right = render_diff_side(document, new, false, panel_width, number_width, theme);
    let fill = Style::default().bg(theme.card_bg());
    let blank = || vec![Span::decoration(" ".repeat(panel_width), fill)];
    let rows = (0..left.len().max(right.len()))
        .map(|row| {
            let mut spans = left.get(row).cloned().unwrap_or_else(blank);
            spans.push(Span::decoration(
                DIFF_SIDE_BY_SIDE_SEPARATOR,
                fill.fg(theme.muted_text),
            ));
            spans.extend(right.get(row).cloned().unwrap_or_else(blank));
            spans
        })
        .collect();
    let boundary = if old.is_none_or(|line| line.text.is_empty())
        && new.is_none_or(|line| line.text.is_empty())
    {
        Break::SoftWrap
    } else {
        Break::HardBreak
    };
    push_diff_rows(document, rows, boundary, theme, width);
}

fn render_diff_side(
    document: &mut Document<Style>,
    entry: Option<&DiffLine>,
    old: bool,
    width: usize,
    number_width: usize,
    theme: Theme,
) -> Vec<Vec<Span<Style>>> {
    let fill = Style::default().bg(theme.card_bg());
    let Some(line) = entry else {
        return vec![vec![Span::decoration(" ".repeat(width), fill)]];
    };
    let (marker, marker_color, bg) = match line.kind {
        ChangeTag::Insert => ("+", theme.success, theme.diff_add_bg),
        ChangeTag::Delete => ("-", theme.error, theme.diff_delete_bg),
        ChangeTag::Equal => (" ", theme.muted_text, theme.card_bg()),
    };
    let style = Style::default().fg(theme.text).bg(bg);
    let number = if old {
        line.old_number
    } else {
        line.new_number.or(line.old_number)
    };
    let number = number
        .map(|number| format!("{number:>number_width$}"))
        .unwrap_or_else(|| " ".repeat(number_width));
    let prefix_width = number_width + 3;
    let chunks = wrap_code_line(
        document,
        &line.text,
        style,
        width.saturating_sub(prefix_width),
    );
    chunks
        .into_iter()
        .enumerate()
        .map(|(index, chunk)| {
            let mut spans = vec![
                Span::decoration(
                    if index == 0 {
                        number.clone()
                    } else {
                        " ".repeat(number_width)
                    },
                    fill.fg(theme.muted_text),
                ),
                Span::decoration(" ", fill),
                Span::decoration(if index == 0 { marker } else { " " }, fill.fg(marker_color)),
                Span::decoration(" ", style),
            ];
            spans.extend(chunk);
            let used = spans
                .iter()
                .map(|span| display_width(&span.text))
                .sum::<usize>();
            if used < width {
                spans.push(Span::decoration(" ".repeat(width - used), style));
            }
            spans
        })
        .collect()
}

fn code_text(text: &str) -> String {
    let mut result = String::new();
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\u{1b}' {
            match chars.next() {
                Some('[') => {
                    consume_csi_sequence(&mut chars);
                }
                Some(']') => {
                    consume_osc_sequence(&mut chars);
                }
                _ => {}
            }
        } else if ch == '\t' || !ch.is_control() {
            result.push(ch);
        }
    }
    result
}

fn wrap_code_line(
    document: &mut Document<Style>,
    text: &str,
    style: Style,
    width: usize,
) -> Vec<Vec<Span<Style>>> {
    if text.is_empty() && width > 0 {
        let block = document.add_source("\n");
        return vec![vec![Span::source_atomic(
            " ",
            style,
            SourceRange::new(block, 0, 1),
        )]];
    }
    let text = code_text(text);
    let block = document.add_source(text.clone());
    let mut rows = Vec::new();
    let mut row = Vec::new();
    let mut row_width = 0;
    let mut column = 0;
    let mut offset = 0;
    if width == 0 {
        return vec![row];
    }
    for grapheme in text.graphemes(true) {
        let end = offset + grapheme.chars().count();
        let range = SourceRange::new(block, offset, end);
        if grapheme == "\t" {
            let tab_width = DIFF_TAB_SIZE - column % DIFF_TAB_SIZE;
            let mut spaces = tab_width;
            while spaces > 0 {
                if row_width == width {
                    rows.push(std::mem::take(&mut row));
                    row_width = 0;
                }
                let count = spaces.min(width - row_width);
                row.push(Span::source_atomic(" ".repeat(count), style, range));
                row_width += count;
                spaces -= count;
            }
            column += tab_width;
        } else {
            let cells = display_width(grapheme);
            if cells <= width {
                if row_width + cells > width {
                    rows.push(std::mem::take(&mut row));
                    row_width = 0;
                }
                if let Some(last) = row.last_mut()
                    && last.copy_mode == CopyMode::Exact
                    && last.source.is_some_and(|source| source.end == offset)
                {
                    last.text.push_str(grapheme);
                    last.source.as_mut().unwrap().end = end;
                } else {
                    row.push(Span::source(grapheme, style, range));
                }
                row_width += cells;
            }
            column += cells;
        }
        offset = end;
    }
    rows.push(row);
    rows
}

fn push_diff_rows(
    document: &mut Document<Style>,
    rows: Vec<Vec<Span<Style>>>,
    boundary: Break,
    theme: Theme,
    width: usize,
) {
    let last = rows.len().saturating_sub(1);
    for (index, body) in rows.into_iter().enumerate() {
        let fill = Style::default().bg(theme.card_bg());
        let mut spans = vec![
            Span::decoration(
                TOOL_GUIDE_GLYPH,
                fill.fg(theme.card_guide()).bg(theme.root_bg),
            ),
            Span::decoration("  ", fill),
        ];
        spans.extend(body);
        let mut remaining = width;
        let mut clipped = Vec::new();
        for mut span in spans {
            if remaining == 0 {
                break;
            }
            let mut used = 0;
            let text = span
                .text
                .graphemes(true)
                .take_while(|grapheme| {
                    let cells = display_width(grapheme);
                    if used + cells > remaining {
                        return false;
                    }
                    used += cells;
                    true
                })
                .collect::<String>();
            if text.is_empty() {
                continue;
            }
            if let Some(range) = &mut span.source
                && span.copy_mode == CopyMode::Exact
            {
                range.end = range.start + text.chars().count();
            }
            span.text = text;
            clipped.push(span);
            remaining -= used;
        }
        if remaining > 0 {
            clipped.push(Span::decoration(" ".repeat(remaining), fill));
        }
        document.push_line(
            Line { spans: clipped },
            if index == last {
                boundary
            } else {
                Break::SoftWrap
            },
        );
    }
}
