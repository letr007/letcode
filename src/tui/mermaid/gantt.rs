use super::gantt_ir as ir;
use super::{
    MermaidRenderSpan, MermaidSourceSpan, render_line_count_within_limits, source_within_limits,
};
use crate::tui::measure::display_width;

const STATUSES: [&str; 4] = ["done", "active", "crit", "milestone"];

pub(super) fn render(source: &str, width: usize) -> Option<Vec<Vec<MermaidRenderSpan>>> {
    let diagram = parse(source)?;
    let mut lines = Vec::new();
    for item in &diagram.items {
        match item {
            ir::Item::Config(config) => lines.push(vec![
                MermaidRenderSpan::decoration(format!("{} ", config.key)),
                span(&config.value),
            ]),
            ir::Item::Section(label) => {
                lines.push(vec![MermaidRenderSpan::decoration("section "), span(label)])
            }
            ir::Item::Task(task) => {
                let mut line = vec![span(&task.name)];
                if let Some(status) = &task.status {
                    line.extend([
                        MermaidRenderSpan::decoration(" ["),
                        span(status),
                        MermaidRenderSpan::decoration("]"),
                    ]);
                }
                if let Some(id) = &task.id {
                    line.extend([MermaidRenderSpan::decoration(" "), span(id)]);
                }
                line.extend([MermaidRenderSpan::decoration(" : "), span(&task.timing)]);
                lines.push(line);
            }
        }
    }
    (render_line_count_within_limits(lines.len())
        && lines
            .iter()
            .all(|line| line.iter().map(|s| display_width(&s.text)).sum::<usize>() <= width))
    .then_some(lines)
}

fn span(label: &ir::Label) -> MermaidRenderSpan {
    MermaidRenderSpan::source(label.text.clone(), label.span, false)
}

fn parse(source: &str) -> Option<ir::Diagram> {
    if !source_within_limits(source) || source.contains('\r') {
        return None;
    }
    let mut lines = source.lines();
    if lines.next()? != "gantt" {
        return None;
    }

    let mut items = Vec::new();
    let mut offset = "gantt".chars().count() + 1;
    for raw in source.lines().skip(1) {
        let trimmed = raw.trim();
        let leading = raw.chars().count() - raw.trim_start().chars().count();
        let base = offset + leading;
        if trimmed.is_empty() || trimmed.starts_with("%%") {
            offset += raw.chars().count() + 1;
            continue;
        }

        let (key, value) = split_directive(trimmed);
        match key {
            "title" | "dateFormat" | "axisFormat" => {
                if value.is_empty() {
                    return None;
                }
                let at = find_char(trimmed, value)?;
                items.push(ir::Item::Config(ir::Config {
                    key: match key {
                        "title" => "title",
                        "dateFormat" => "dateFormat",
                        "axisFormat" => "axisFormat",
                        _ => unreachable!(),
                    },
                    value: ir::Label {
                        text: value.to_string(),
                        span: MermaidSourceSpan::new(base + at, base + at + value.chars().count()),
                    },
                }));
            }
            "section" => {
                if value.is_empty() {
                    return None;
                }
                let at = find_char(trimmed, value)?;
                items.push(ir::Item::Section(ir::Label {
                    text: value.to_string(),
                    span: MermaidSourceSpan::new(base + at, base + at + value.chars().count()),
                }));
            }
            "excludes" | "todayMarker" | "weekday" => return None,
            _ => items.push(ir::Item::Task(parse_task(trimmed, base)?)),
        }
        offset += raw.chars().count() + 1;
    }
    (!items.is_empty()).then_some(ir::Diagram { items })
}

fn parse_task(line: &str, base: usize) -> Option<ir::Task> {
    let colon_byte = line.find(':')?;
    let name = line[..colon_byte].trim();
    let rest = line[colon_byte + 1..].trim();
    if name.is_empty() || rest.is_empty() {
        return None;
    }

    let fields = rest.split(',').map(str::trim).collect::<Vec<_>>();
    if fields.iter().any(|field| field.is_empty()) || !(2..=4).contains(&fields.len()) {
        return None;
    }
    let mut index = 0;
    let field_start_byte = colon_byte + 1;
    let status = if STATUSES.contains(&fields[0]) {
        let label = label_in_from(line, fields[0], base, field_start_byte)?;
        index += 1;
        Some(label)
    } else {
        None
    };
    let remaining = fields.len() - index;
    let id = if remaining == 3 {
        let text = fields[index];
        if !valid_id(text) {
            return None;
        }
        index += 1;
        Some(label_in_from(line, text, base, field_start_byte)?)
    } else {
        None
    };
    if fields.len() - index != 2
        || !valid_start(fields[index])
        || !valid_end_or_duration(fields[index + 1])
    {
        return None;
    }

    let timing_start_byte = line[colon_byte + 1..].find(fields[index])? + colon_byte + 1;
    let timing_end_byte = line.rfind(fields[index + 1])? + fields[index + 1].len();
    let timing_text = line[timing_start_byte..timing_end_byte].trim();
    let timing_at = char_index(line, timing_start_byte)
        + line[timing_start_byte..timing_end_byte]
            .chars()
            .take_while(|c| c.is_whitespace())
            .count();
    let name_at = find_char(line, name)?;
    Some(ir::Task {
        name: ir::Label {
            text: name.to_string(),
            span: MermaidSourceSpan::new(base + name_at, base + name_at + name.chars().count()),
        },
        status,
        id,
        timing: ir::Label {
            text: timing_text.to_string(),
            span: MermaidSourceSpan::new(
                base + timing_at,
                base + timing_at + timing_text.chars().count(),
            ),
        },
    })
}

fn split_directive(line: &str) -> (&str, &str) {
    let split = line.find(char::is_whitespace).unwrap_or(line.len());
    (&line[..split], line[split..].trim())
}

fn label_in_from(line: &str, text: &str, base: usize, start_byte: usize) -> Option<ir::Label> {
    let byte = start_byte + line[start_byte..].find(text)?;
    let at = char_index(line, byte);
    Some(ir::Label {
        text: text.to_string(),
        span: MermaidSourceSpan::new(base + at, base + at + text.chars().count()),
    })
}

fn valid_id(value: &str) -> bool {
    !value.is_empty()
        && value
            .chars()
            .all(|c| c.is_alphanumeric() || c == '_' || c == '-')
}

fn valid_start(value: &str) -> bool {
    valid_date(value) || value.strip_prefix("after ").is_some_and(valid_id)
}

fn valid_end_or_duration(value: &str) -> bool {
    valid_date(value) || value.strip_prefix("until ").is_some_and(valid_id) || is_duration(value)
}

fn valid_date(value: &str) -> bool {
    let parts = value.split('-').collect::<Vec<_>>();
    parts.len() == 3
        && parts[0].len() == 4
        && parts[1].len() == 2
        && parts[2].len() == 2
        && parts
            .iter()
            .all(|part| part.chars().all(|c| c.is_ascii_digit()))
}

fn is_duration(value: &str) -> bool {
    let digit_count = value.chars().take_while(|c| c.is_ascii_digit()).count();
    digit_count > 0 && matches!(&value[digit_count..], "ms" | "s" | "m" | "h" | "d" | "w")
}

fn find_char(haystack: &str, needle: &str) -> Option<usize> {
    haystack.find(needle).map(|byte| char_index(haystack, byte))
}

fn char_index(value: &str, byte: usize) -> usize {
    value[..byte].chars().count()
}
