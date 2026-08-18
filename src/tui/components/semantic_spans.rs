//! Shared semantic-span rendering, terminal-safe text filtering, and
//! small tool-argument helpers used by multiple tool-card renderers.

use ratatui::style::{Color, Modifier, Style};
use unicode_segmentation::UnicodeSegmentation;

use crate::tui::{
    surface,
    measure::display_width,
    theme::Theme,
    timeline::ToolView,
    transcript_render::{Break, SemanticLine, SemanticSpan},
};

pub(super) const TOOL_GUIDE_GLYPH: &str = surface::ACCENT_BAR_GLYPH;

pub(super) const COMPACT_SHELL_BODY_LINES: usize = 20;

pub(super) fn clip_semantic_spans(
    segments: Vec<SemanticSpan<Style>>,
    width: usize,
) -> Vec<SemanticSpan<Style>> {
    let mut remaining = width;
    let mut clipped = Vec::new();
    for segment in segments {
        if remaining == 0 {
            break;
        }
        if display_width(&segment.text) <= remaining {
            remaining = remaining.saturating_sub(display_width(&segment.text));
            clipped.push(segment);
            continue;
        }
        let text = truncate_display_width(&segment.text, remaining);
        let prefix = text.strip_suffix('…').unwrap_or(&text);
        if !prefix.is_empty() {
            clipped.push(if segment.copy {
                SemanticSpan::source_with_join(prefix, segment.style, segment.copy_join)
            } else {
                SemanticSpan::decoration(prefix, segment.style)
            });
        }
        if text.ends_with('…') {
            clipped.push(SemanticSpan::decoration("…", segment.style));
        }
        break;
    }
    clipped
}

pub(super) fn render_output_section(
    title: &str,
    text: &str,
    text_style: Style,
    theme: Theme,
    width: usize,
    expanded_output: bool,
) -> Vec<SemanticLine<Style>> {
    let mut lines = Vec::new();
    lines.push(render_card_line(
        &[],
        Style::default().bg(theme.card_bg()),
        theme,
        width,
    ));
    lines.push(render_card_line(
        &[(
            title.to_string(),
            root_muted_style(theme)
                .bg(theme.card_bg())
                .add_modifier(Modifier::BOLD),
        )],
        Style::default().bg(theme.card_bg()),
        theme,
        width,
    ));
    lines.push(render_card_line(
        &[],
        Style::default().bg(theme.card_bg()),
        theme,
        width,
    ));
    lines.extend(render_limited_text_lines(
        text,
        text_style,
        theme,
        width,
        expanded_output,
    ));
    lines.push(render_card_line(
        &[],
        Style::default().bg(theme.card_bg()),
        theme,
        width,
    ));
    lines
}

pub(super) fn render_limited_text_lines(
    text: &str,
    text_style: Style,
    theme: Theme,
    width: usize,
    expanded_output: bool,
) -> Vec<SemanticLine<Style>> {
    let mut lines = Vec::new();
    for (idx, raw) in text.lines().enumerate() {
        if !expanded_output && idx >= max_body_lines() {
            lines.push(render_card_line(
                &[(
                    "… output clipped in TUI".to_string(),
                    root_muted_style(theme).bg(theme.card_bg()),
                )],
                Style::default().bg(theme.card_bg()),
                theme,
                width,
            ));
            break;
        }
        let line = if raw.is_empty() { " " } else { raw };
        let segments = ansi_sgr_segments(line, text_style.bg(theme.card_bg()));
        lines.push(render_source_card_line_with_boundary(
            &segments,
            Style::default().bg(theme.card_bg()),
            theme,
            width,
            Break::HardBreak,
        ));
    }
    lines
}

pub(super) fn render_tail_limited_text_lines(
    text: &str,
    text_style: Style,
    theme: Theme,
    width: usize,
    expanded_output: bool,
) -> Vec<SemanticLine<Style>> {
    let body = text.lines().collect::<Vec<_>>();
    let is_clipped = !expanded_output && body.len() > COMPACT_SHELL_BODY_LINES;
    let body = if is_clipped {
        &body[body.len() - COMPACT_SHELL_BODY_LINES..]
    } else {
        &body[..]
    };

    let mut lines = Vec::new();
    for raw in body {
        let line = if raw.is_empty() { " " } else { raw };
        let segments = ansi_sgr_segments(line, text_style.bg(theme.card_bg()));
        lines.push(render_source_card_line_with_boundary(
            &segments,
            Style::default().bg(theme.card_bg()),
            theme,
            width,
            Break::HardBreak,
        ));
    }
    lines
}

pub(super) fn terminal_safe_text(text: &str) -> String {
    ansi_sgr_segments(text, Style::default())
        .into_iter()
        .map(|(text, _)| text)
        .collect()
}

/// Split shell/tool output into styled segments for TUI cells.
///
/// Contract: segment text never contains control characters. SGR becomes style;
/// other CSI/OSC/C0 and truncated escapes are dropped so VT state cannot escape
/// into the ratatui write path.

pub(super) fn ansi_sgr_segments(text: &str, base_style: Style) -> Vec<(String, Style)> {
    // Progress bars overwrite with CR; only the suffix after the last CR is visible.
    let text = text.rsplit('\r').next().unwrap_or(text);

    let mut segments = Vec::new();
    let mut current_style = base_style;
    let mut current_text = String::new();
    let mut chars = text.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch == '\u{1b}' {
            match chars.peek().copied() {
                Some('[') => {
                    chars.next();
                    match consume_csi_sequence(&mut chars) {
                        CsiOutcome::Sgr(sequence) => {
                            if !current_text.is_empty() {
                                segments.push((std::mem::take(&mut current_text), current_style));
                            }
                            current_style =
                                apply_sgr_sequence(&sequence, base_style, current_style);
                        }
                        CsiOutcome::Other | CsiOutcome::Incomplete => {}
                    }
                }
                Some(']') => {
                    chars.next();
                    let _ = consume_osc_sequence(&mut chars);
                }
                Some(_) => {
                    // Two-byte / short ESC form: drop introducer and the next byte.
                    chars.next();
                }
                None => {
                    // Truncated ESC at end of stream — never emit it into a cell.
                }
            }
            continue;
        }

        if ch == '\t' {
            current_text.push(' ');
            continue;
        }

        if ch.is_control() {
            continue;
        }

        current_text.push(ch);
    }

    if !current_text.is_empty() {
        segments.push((current_text, current_style));
    }

    if segments.is_empty() {
        segments.push((String::new(), base_style));
    }
    segments
}

pub(super) enum CsiOutcome {
    Sgr(String),
    Other,
    Incomplete,
}

/// Consume CSI parameter/intermediate bytes through the final byte (`@`..=`~`).

pub(super) fn consume_csi_sequence(chars: &mut std::iter::Peekable<std::str::Chars<'_>>) -> CsiOutcome {
    let mut sequence = String::new();
    let mut saw_final = false;
    let mut final_byte = '\0';

    for next in chars.by_ref() {
        match next {
            '\u{20}'..='\u{3f}' => sequence.push(next),
            '\u{40}'..='\u{7e}' => {
                final_byte = next;
                saw_final = true;
                break;
            }
            _ => {
                // Malformed CSI: drop what we consumed; do not emit ESC.
                return CsiOutcome::Other;
            }
        }
    }

    if !saw_final {
        return CsiOutcome::Incomplete;
    }

    if final_byte == 'm' {
        // SGR params are digits/semicolons; strip private/intermediate junk.
        let params: String = sequence
            .chars()
            .filter(|ch| ch.is_ascii_digit() || *ch == ';')
            .collect();
        CsiOutcome::Sgr(params)
    } else {
        CsiOutcome::Other
    }
}

/// Consume OSC through BEL or ST (`ESC \`). Incomplete OSC is dropped.

pub(super) fn consume_osc_sequence(chars: &mut std::iter::Peekable<std::str::Chars<'_>>) -> bool {
    while let Some(next) = chars.next() {
        if next == '\u{07}' {
            return true;
        }
        if next == '\u{1b}' {
            if chars.peek() == Some(&'\\') {
                chars.next();
                return true;
            }
            // Nested/truncated ESC inside OSC — stop without leaking.
            return false;
        }
    }
    false
}

pub(super) fn apply_sgr_sequence(sequence: &str, base_style: Style, mut style: Style) -> Style {
    let codes: Vec<u16> = if sequence.is_empty() {
        vec![0]
    } else {
        sequence
            .split(';')
            .map(|part| part.parse::<u16>().unwrap_or(0))
            .collect()
    };

    let mut index = 0;
    while index < codes.len() {
        match codes[index] {
            0 => style = base_style,
            1 => style = style.add_modifier(Modifier::BOLD),
            3 => style = style.add_modifier(Modifier::ITALIC),
            4 => style = style.add_modifier(Modifier::UNDERLINED),
            22 => style = style.remove_modifier(Modifier::BOLD),
            23 => style = style.remove_modifier(Modifier::ITALIC),
            24 => style = style.remove_modifier(Modifier::UNDERLINED),
            30..=37 => style = style.fg(ansi_basic_color(codes[index] - 30, false)),
            39 => {
                style = match base_style.fg {
                    Some(color) => style.fg(color),
                    None => style,
                }
            }
            90..=97 => style = style.fg(ansi_basic_color(codes[index] - 90, true)),
            38 if codes.get(index + 1) == Some(&5) => {
                if let Some(color_index) = codes.get(index + 2).copied() {
                    style = style.fg(ansi_256_color(color_index));
                    index += 2;
                }
            }
            38 if codes.get(index + 1) == Some(&2) => {
                if let (Some(r), Some(g), Some(b)) = (
                    codes.get(index + 2).copied(),
                    codes.get(index + 3).copied(),
                    codes.get(index + 4).copied(),
                ) {
                    style = style.fg(Color::Rgb(r as u8, g as u8, b as u8));
                    index += 4;
                }
            }
            _ => {}
        }
        index += 1;
    }

    style
}

pub(super) fn ansi_basic_color(index: u16, bright: bool) -> Color {
    let colors = if bright {
        [
            Color::Rgb(128, 128, 128),
            Color::Rgb(255, 85, 85),
            Color::Rgb(80, 250, 123),
            Color::Rgb(241, 250, 140),
            Color::Rgb(98, 114, 164),
            Color::Rgb(255, 121, 198),
            Color::Rgb(139, 233, 253),
            Color::Rgb(248, 248, 242),
        ]
    } else {
        [
            Color::Rgb(0, 0, 0),
            Color::Rgb(205, 49, 49),
            Color::Rgb(13, 188, 121),
            Color::Rgb(229, 229, 16),
            Color::Rgb(36, 114, 200),
            Color::Rgb(188, 63, 188),
            Color::Rgb(17, 168, 205),
            Color::Rgb(229, 229, 229),
        ]
    };
    colors[index as usize]
}

pub(super) fn ansi_256_color(index: u16) -> Color {
    match index {
        0..=7 => ansi_basic_color(index, false),
        8..=15 => ansi_basic_color(index - 8, true),
        16..=231 => {
            let value = index - 16;
            let r = value / 36;
            let g = (value % 36) / 6;
            let b = value % 6;
            Color::Rgb(
                ansi_256_component(r),
                ansi_256_component(g),
                ansi_256_component(b),
            )
        }
        232..=255 => {
            let level = 8 + ((index - 232) * 10) as u8;
            Color::Rgb(level, level, level)
        }
        _ => Color::Reset,
    }
}

pub(super) fn ansi_256_component(value: u16) -> u8 {
    if value == 0 {
        0
    } else {
        (55 + value * 40) as u8
    }
}

pub(super) fn tool_arguments(tool: &ToolView) -> Option<serde_json::Value> {
    tool.arguments
        .as_deref()
        .and_then(|text| serde_json::from_str::<serde_json::Value>(text).ok())
}

pub(super) fn tool_output_data(tool: &ToolView) -> Option<serde_json::Value> {
    let output = tool.output.as_deref()?;
    let value = serde_json::from_str::<serde_json::Value>(output).ok()?;
    value.get("data").cloned().or(Some(value))
}

pub(super) fn output_title<'a>(label: &'a str, truncated: Option<&serde_json::Value>) -> &'a str {
    if truncated
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
    {
        match label {
            "stdout" => "stdout · truncated",
            "stderr" => "stderr · truncated",
            "diff" => "diff · truncated",
            _ => label,
        }
    } else {
        label
    }
}

pub(super) fn root_text_style(theme: Theme) -> Style {
    Style::default().fg(theme.text).bg(theme.root_bg)
}

pub(super) fn root_muted_style(theme: Theme) -> Style {
    Style::default().fg(theme.muted_text).bg(theme.root_bg)
}

pub(super) fn max_body_lines() -> usize {
    120
}

pub(super) fn format_with_optional_fields(prefix: &str, subject: &str, fields: Vec<String>) -> String {
    if fields.is_empty() {
        format!("{prefix} {subject}")
    } else {
        format!("{prefix} {subject} [{}]", fields.join(", "))
    }
}

pub(super) fn value_str<'a>(args: Option<&'a serde_json::Value>, key: &str) -> Option<&'a str> {
    args.and_then(|value| value.get(key))
        .and_then(serde_json::Value::as_str)
}

pub(super) fn value_u64(args: Option<&serde_json::Value>, key: &str) -> Option<u64> {
    args.and_then(|value| value.get(key))
        .and_then(serde_json::Value::as_u64)
}

pub(super) fn fallback_tail(summary: &str) -> &str {
    summary
        .split_once(' ')
        .map(|(_, tail)| tail.trim())
        .filter(|tail| !tail.is_empty())
        .unwrap_or(summary)
}

pub(super) fn one_line_snippet(text: &str) -> String {
    // Collapse newlines/whitespace into single spaces, then trim.
    let mut out = String::with_capacity(text.len().min(140));
    let mut last_was_space = false;
    for ch in text.chars() {
        let is_space = ch.is_whitespace();
        if is_space {
            if !last_was_space {
                out.push(' ');
                last_was_space = true;
            }
            continue;
        }
        last_was_space = false;
        out.push(ch);
        if out.len() >= 240 {
            break;
        }
    }
    out.trim().to_string()
}

pub fn truncate_display_width(text: &str, width: usize) -> String {
    if width == 0 {
        return String::new();
    }

    if display_width(text) <= width {
        return text.to_string();
    }

    let ellipsis = "…";
    let ellipsis_width = display_width(ellipsis);
    if width <= ellipsis_width {
        return ellipsis.chars().take(1).collect();
    }

    let mut out = String::new();
    let mut used = 0usize;

    for grapheme in text.graphemes(true) {
        let grapheme_width = display_width(grapheme);
        if used + grapheme_width + ellipsis_width > width {
            break;
        }
        out.push_str(grapheme);
        used = used.saturating_add(grapheme_width);
    }

    out.push('…');
    out
}


pub(crate) fn render_card_line(
    segments: &[(String, Style)],
    fill_style: Style,
    theme: Theme,
    width: usize,
) -> SemanticLine<Style> {
    let semantic_segments = segments
        .iter()
        .map(|(text, style)| SemanticSpan::decoration(text.clone(), *style))
        .collect::<Vec<_>>();
    render_card_line_with_guide(
        &semantic_segments,
        fill_style,
        theme.card_guide(),
        theme,
        width,
        Break::SoftWrap,
    )
}

pub(super) fn render_source_card_line_with_boundary(
    segments: &[(String, Style)],
    fill_style: Style,
    theme: Theme,
    width: usize,
    boundary: Break,
) -> SemanticLine<Style> {
    let semantic_segments = segments
        .iter()
        .map(|(text, style)| SemanticSpan::source(text.clone(), *style))
        .collect::<Vec<_>>();
    render_card_line_with_guide(
        &semantic_segments,
        fill_style,
        theme.card_guide(),
        theme,
        width,
        boundary,
    )
}

pub(super) fn render_card_line_with_guide(
    segments: &[SemanticSpan<Style>],
    fill_style: Style,
    guide_color: ratatui::style::Color,
    theme: Theme,
    width: usize,
    boundary: Break,
) -> SemanticLine<Style> {
    if width == 0 {
        return SemanticLine::default();
    }

    let guide_width = display_width(TOOL_GUIDE_GLYPH);
    let prefix_width = guide_width.saturating_add(2);
    let guide_style = Style::default().fg(guide_color).bg(theme.root_bg);
    if width <= guide_width {
        return SemanticLine {
            spans: vec![SemanticSpan::decoration(TOOL_GUIDE_GLYPH, guide_style)],
            boundary,
        };
    }

    let leading_pad_style = fill_style;

    let mut spans = vec![
        SemanticSpan::decoration(TOOL_GUIDE_GLYPH, guide_style),
        SemanticSpan::decoration("  ", leading_pad_style),
    ];
    let mut remaining = width.saturating_sub(prefix_width);

    for segment in clip_semantic_spans(segments.to_vec(), remaining) {
        let used = display_width(&segment.text);
        spans.push(segment);
        remaining = remaining.saturating_sub(used);
    }

    if remaining > 0 {
        spans.push(SemanticSpan::decoration(" ".repeat(remaining), fill_style));
    }

    SemanticLine { spans, boundary }
}
