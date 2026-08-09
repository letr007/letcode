//! Sequence diagram parsing and rendering.

use std::collections::HashMap;

use crate::tui::measure::display_width;

use super::{MermaidRenderSpan, MermaidSourceSpan, sequence_ir as ir};

pub(crate) fn render(source: &str, width: usize) -> Option<Vec<Vec<MermaidRenderSpan>>> {
    let sequence = parse(source)?;
    let mut lines = Vec::new();
    render_items(&sequence, &sequence.items, 0, &mut lines)?;
    lines
        .iter()
        .all(|line| {
            line.iter()
                .map(|span| display_width(&span.text))
                .sum::<usize>()
                <= width
        })
        .then_some(lines)
}

fn render_items(
    sequence: &ir::MermaidSequence,
    items: &[ir::MermaidSequenceItem],
    indent: usize,
    lines: &mut Vec<Vec<MermaidRenderSpan>>,
) -> Option<()> {
    for item in items {
        match item {
            ir::MermaidSequenceItem::Message(message) => {
                let from = sequence.participants.get(&message.from)?;
                let to = sequence.participants.get(&message.to)?;
                lines.push(vec![
                    MermaidRenderSpan::decoration(" ".repeat(indent)),
                    MermaidRenderSpan::source(
                        from.label.clone(),
                        MermaidSourceSpan::new(from.start, from.end),
                        false,
                    ),
                    MermaidRenderSpan::decoration(if message.dashed {
                        " ╌╌▶ "
                    } else {
                        " ──▶ "
                    }),
                    MermaidRenderSpan::source(
                        to.label.clone(),
                        MermaidSourceSpan::new(to.start, to.end),
                        false,
                    ),
                    MermaidRenderSpan::decoration("  "),
                    MermaidRenderSpan::source(
                        message.label.text.clone(),
                        MermaidSourceSpan::new(message.label.start, message.label.end),
                        false,
                    ),
                ]);
            }
            ir::MermaidSequenceItem::Block(block) => {
                lines.push(vec![
                    MermaidRenderSpan::decoration(" ".repeat(indent)),
                    MermaidRenderSpan::decoration(format!("{} ", block.kind.keyword())),
                    MermaidRenderSpan::source(
                        block.label.text.clone(),
                        MermaidSourceSpan::new(block.label.start, block.label.end),
                        false,
                    ),
                ]);
                for (index, branch) in block.branches.iter().enumerate() {
                    if index > 0 {
                        lines.push(vec![
                            MermaidRenderSpan::decoration(" ".repeat(indent)),
                            MermaidRenderSpan::decoration("else"),
                            MermaidRenderSpan::decoration(" "),
                        ]);
                        if let Some(label) = &branch.label {
                            if let Some(line) = lines.last_mut() {
                                line.push(MermaidRenderSpan::source(
                                    label.text.clone(),
                                    MermaidSourceSpan::new(label.start, label.end),
                                    false,
                                ));
                            }
                        }
                    }
                    render_items(sequence, &branch.items, indent + 2, lines)?;
                }
                lines.push(vec![
                    MermaidRenderSpan::decoration(" ".repeat(indent)),
                    MermaidRenderSpan::decoration("end"),
                ]);
            }
        }
    }
    Some(())
}

fn parse(source: &str) -> Option<ir::MermaidSequence> {
    if source.contains('\r') {
        return None;
    }
    let lines = source
        .lines()
        .enumerate()
        .scan(0usize, |offset, (_index, text)| {
            let line = ParsedLine {
                base: *offset + text.chars().count() - text.trim_start().chars().count(),
                text: text.trim(),
            };
            *offset += text.chars().count() + 1;
            Some(line)
        })
        .collect::<Vec<_>>();
    if lines.first()?.text != "sequenceDiagram" {
        return None;
    }
    let mut participants = HashMap::new();
    let mut cursor = 1;
    let items = parse_items(&lines, &mut cursor, &mut participants, true, false)?;
    if cursor != lines.len() || participants.is_empty() || !contains_message(&items) {
        return None;
    }
    Some(ir::MermaidSequence {
        participants,
        items,
    })
}

struct ParsedLine<'a> {
    base: usize,
    text: &'a str,
}

fn parse_items<'a>(
    lines: &[ParsedLine<'a>],
    cursor: &mut usize,
    participants: &mut HashMap<String, ir::MermaidNode>,
    allow_participant: bool,
    in_block: bool,
) -> Option<Vec<ir::MermaidSequenceItem>> {
    let mut items = Vec::new();
    while *cursor < lines.len() {
        let line = &lines[*cursor];
        if line.text.is_empty() {
            *cursor += 1;
            continue;
        }
        if line.text == "end" || line.text == "else" || line.text.starts_with("else ") {
            if in_block {
                break;
            }
            return None;
        }
        let (keyword, rest) = line
            .text
            .split_once(char::is_whitespace)
            .map_or((line.text, ""), |(keyword, rest)| (keyword, rest.trim()));
        if let Some(kind) = match keyword {
            "loop" => Some(ir::MermaidBlockKind::Loop),
            "alt" => Some(ir::MermaidBlockKind::Alt),
            "opt" => Some(ir::MermaidBlockKind::Opt),
            _ => None,
        } {
            if rest.is_empty() {
                return None;
            }
            let label_start =
                line.base + keyword.chars().count() + line.text[keyword.len()..].chars().count()
                    - rest.chars().count();
            let label = ir::MermaidLabel {
                text: rest.to_string(),
                start: label_start,
                end: label_start + rest.chars().count(),
            };
            *cursor += 1;
            let branches = parse_block(lines, cursor, participants, kind)?;
            items.push(ir::MermaidSequenceItem::Block(ir::MermaidBlock {
                kind,
                label,
                branches,
            }));
        } else if keyword == "participant" && allow_participant {
            parse_participant(line, participants)?;
            *cursor += 1;
        } else if keyword == "participant" {
            return None;
        } else if matches!(keyword, "loop" | "alt" | "opt" | "else" | "end") {
            return None;
        } else {
            let message = parse_message(line, participants)?;
            items.push(ir::MermaidSequenceItem::Message(message));
            *cursor += 1;
        }
    }
    Some(items)
}

fn parse_block<'a>(
    lines: &[ParsedLine<'a>],
    cursor: &mut usize,
    participants: &mut HashMap<String, ir::MermaidNode>,
    kind: ir::MermaidBlockKind,
) -> Option<Vec<ir::MermaidBranch>> {
    let mut branches = Vec::new();
    let mut label = None;
    let mut seen_else = false;
    loop {
        let items = parse_items(lines, cursor, participants, false, true)?;
        if items.is_empty() {
            return None;
        }
        branches.push(ir::MermaidBranch { label, items });
        if *cursor >= lines.len() {
            return None;
        }
        let line = &lines[*cursor];
        if line.text == "end" {
            *cursor += 1;
            return if kind == ir::MermaidBlockKind::Alt || branches.len() == 1 {
                Some(branches)
            } else {
                None
            };
        }
        if kind != ir::MermaidBlockKind::Alt || seen_else || !line.text.starts_with("else") {
            return None;
        }
        seen_else = true;
        let rest = line.text.strip_prefix("else").unwrap();
        if !rest.is_empty() && !rest.starts_with(char::is_whitespace) {
            return None;
        }
        let leading = rest.chars().take_while(|ch| ch.is_whitespace()).count();
        let rest = rest.trim();
        label = if rest.is_empty() {
            None
        } else {
            let start = line.base + "else".chars().count() + leading;
            Some(ir::MermaidLabel {
                text: rest.to_string(),
                start,
                end: start + rest.chars().count(),
            })
        };
        *cursor += 1;
    }
}

fn parse_participant(
    line: &ParsedLine<'_>,
    participants: &mut HashMap<String, ir::MermaidNode>,
) -> Option<()> {
    let rest = line.text.strip_prefix("participant ")?;
    let (id, label) = rest.split_once(" as ")?;
    let id = id.trim();
    let label = label.trim();
    if !valid_id(id) || label.is_empty() || participants.contains_key(id) {
        return None;
    }
    let separator = rest.find(" as ")?;
    let label_segment = &rest[separator + " as ".len()..];
    let label_leading = label_segment
        .chars()
        .take_while(|ch| ch.is_whitespace())
        .count();
    let start = line.base
        + "participant ".chars().count()
        + rest[..separator + " as ".len()].chars().count()
        + label_leading;
    participants.insert(
        id.to_string(),
        ir::MermaidNode {
            label: label.to_string(),
            start,
            end: start + label.chars().count(),
        },
    );
    Some(())
}

fn parse_message(
    line: &ParsedLine<'_>,
    participants: &HashMap<String, ir::MermaidNode>,
) -> Option<ir::MermaidMessage> {
    let colon = line.text.find(':')?;
    let route = line.text[..colon].trim();
    let label = line.text[colon + 1..].trim();
    let (arrow_at, arrow) = ["-->>", "->>", "-->", "->"]
        .into_iter()
        .filter_map(|arrow| route.find(arrow).map(|index| (index, arrow)))
        .min_by_key(|(index, _)| *index)?;
    let from = route[..arrow_at].trim();
    let to = route[arrow_at + arrow.len()..].trim();
    if !valid_id(from)
        || !valid_id(to)
        || label.is_empty()
        || !participants.contains_key(from)
        || !participants.contains_key(to)
    {
        return None;
    }
    let label_byte = colon + 1;
    let label_leading = line.text[label_byte..]
        .chars()
        .take_while(|ch| ch.is_whitespace())
        .count();
    let start = line.base + line.text[..label_byte].chars().count() + label_leading;
    Some(ir::MermaidMessage {
        from: from.to_string(),
        to: to.to_string(),
        label: ir::MermaidLabel {
            text: label.to_string(),
            start,
            end: start + label.chars().count(),
        },
        dashed: arrow.starts_with("--"),
    })
}

fn contains_message(items: &[ir::MermaidSequenceItem]) -> bool {
    items.iter().any(|item| match item {
        ir::MermaidSequenceItem::Message(_) => true,
        ir::MermaidSequenceItem::Block(block) => block
            .branches
            .iter()
            .any(|branch| contains_message(&branch.items)),
    })
}

fn valid_id(value: &str) -> bool {
    !value.is_empty()
        && value
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
}
