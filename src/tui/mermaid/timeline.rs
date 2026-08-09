use crate::tui::measure::display_width;

use super::{
    MermaidRenderSpan, MermaidSourceSpan, render_line_count_within_limits, source_within_limits,
};

#[derive(Debug)]
enum Item {
    Title(Label),
    Section(Label),
    Events { period: Label, events: Vec<Label> },
}

#[derive(Debug)]
struct Label {
    text: String,
    span: MermaidSourceSpan,
}

pub(super) fn render(source: &str, width: usize) -> Option<Vec<Vec<MermaidRenderSpan>>> {
    let items = parse(source)?;
    let mut lines = Vec::new();
    for item in items {
        match item {
            Item::Title(title) => {
                lines.push(vec![MermaidRenderSpan::decoration("╭─ "), span(&title)])
            }
            Item::Section(section) => {
                lines.push(vec![MermaidRenderSpan::decoration("├─ "), span(&section)])
            }
            Item::Events { period, events } => {
                for (index, event) in events.into_iter().enumerate() {
                    let mut line = vec![MermaidRenderSpan::decoration("│  ")];
                    if index == 0 {
                        line.extend([span(&period), MermaidRenderSpan::decoration(" ─ ")]);
                    } else {
                        line.push(MermaidRenderSpan::decoration("   └─ "));
                    }
                    line.push(span(&event));
                    lines.push(line);
                }
            }
        }
    }
    lines.push(vec![MermaidRenderSpan::decoration("╰─")]);
    (render_line_count_within_limits(lines.len()) && fits(&lines, width)).then_some(lines)
}

fn parse(source: &str) -> Option<Vec<Item>> {
    if !source_within_limits(source) || source.contains('\r') {
        return None;
    }
    let mut items = Vec::new();
    let mut offset = source.lines().next()?.chars().count() + 1;
    let mut in_events = false;
    let mut title_seen = false;
    for line in source.split('\n').skip(1) {
        let line_len = line.chars().count();
        if line.contains('\t') || line.chars().last().is_some_and(char::is_whitespace) {
            return None;
        }
        let leading = line.chars().count() - line.trim_start().chars().count();
        let base = offset + leading;
        let line = line.trim_start();
        if line.is_empty() || line.starts_with("%%") {
            offset += line_len + 1;
            continue;
        }
        if let Some(value) = line.strip_prefix("title ") {
            if title_seen || !items.is_empty() || value.is_empty() {
                return None;
            }
            title_seen = true;
            items.push(Item::Title(label(value, base + 6)?));
        } else if let Some(value) = line.strip_prefix("section ") {
            if value.is_empty() || value.contains(':') {
                return None;
            }
            in_events = false;
            items.push(Item::Section(label(value, base + 8)?));
        } else if line.starts_with(':') {
            if !in_events {
                return None;
            }
            let mut segments = parse_colon_segments(line, base)?;
            if !segments.first().is_some_and(|segment| segment.is_none()) {
                return None;
            }
            let appended = segments.drain(1..).map(Option::unwrap).collect::<Vec<_>>();
            if appended.is_empty() {
                return None;
            }
            match items.last_mut()? {
                Item::Events { events, .. } => events.extend(appended),
                _ => return None,
            }
        } else {
            let mut segments = parse_colon_segments(line, base)?;
            let period = segments.first_mut()?.take()?;
            let parsed_events = segments.drain(1..).map(Option::unwrap).collect::<Vec<_>>();
            if parsed_events.is_empty() {
                return None;
            }
            in_events = true;
            items.push(Item::Events {
                period,
                events: parsed_events,
            });
        }
        offset += line_len + 1;
    }
    (!items.is_empty()).then_some(items)
}

fn label(value: &str, start: usize) -> Option<Label> {
    if value.contains(['\n', '<', '>']) || value.is_empty() {
        return None;
    }
    Some(Label {
        text: value.to_string(),
        span: MermaidSourceSpan::new(start, start + value.chars().count()),
    })
}

fn parse_colon_segments(line: &str, base: usize) -> Option<Vec<Option<Label>>> {
    if line.ends_with(':')
        && line[..line.len() - 1]
            .chars()
            .last()
            .is_some_and(char::is_whitespace)
    {
        return None;
    }
    let boundaries = line
        .char_indices()
        .filter_map(|(byte, character)| {
            (character == ':'
                && line[byte + 1..]
                    .chars()
                    .next()
                    .is_some_and(char::is_whitespace))
            .then_some(byte)
        })
        .collect::<Vec<_>>();
    if boundaries.is_empty() {
        return None;
    }
    let mut labels = Vec::new();
    let mut start_byte = 0;
    for end_byte in boundaries.into_iter().chain(std::iter::once(line.len())) {
        let segment = &line[start_byte..end_byte];
        let trimmed = segment.trim();
        if trimmed.is_empty() {
            labels.push(None);
        } else {
            let leading = segment.chars().count() - segment.trim_start().chars().count();
            let segment_start = base + line[..start_byte].chars().count() + leading;
            labels.push(Some(label(trimmed, segment_start)?));
        }
        start_byte = end_byte.saturating_add(1);
    }
    if labels.iter().skip(1).any(Option::is_none) {
        return None;
    }
    Some(labels)
}

fn span(label: &Label) -> MermaidRenderSpan {
    MermaidRenderSpan::source(label.text.clone(), label.span, false)
}

fn fits(lines: &[Vec<MermaidRenderSpan>], width: usize) -> bool {
    lines.iter().all(|line| {
        line.iter()
            .map(|span| display_width(&span.text))
            .sum::<usize>()
            <= width
    })
}
