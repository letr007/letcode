use super::er_ir as ir;
use super::{
    MermaidRenderSpan, MermaidSourceSpan, render_line_count_within_limits, source_within_limits,
};
use crate::tui::measure::display_width;
use std::collections::HashSet;

const CARDINALITIES: [&str; 7] = ["||", "o|", "|o", "o{", "}o", "|{", "}|"];

pub(super) fn render(source: &str, width: usize) -> Option<Vec<Vec<MermaidRenderSpan>>> {
    let diagram = parse(source)?;
    let mut lines = Vec::new();
    for entity in &diagram.entities {
        lines.push(vec![
            MermaidRenderSpan::decoration("╭─ "),
            span(&entity.name),
        ]);
        for attribute in &entity.attributes {
            lines.push(vec![MermaidRenderSpan::decoration("│ "), span(attribute)]);
        }
        lines.push(vec![MermaidRenderSpan::decoration("╰─")]);
    }
    for relation in &diagram.relations {
        lines.push(vec![
            span(&relation.from),
            span(&relation.from_cardinality),
            MermaidRenderSpan::decoration(relation.connector),
            span(&relation.to_cardinality),
            span(&relation.to),
            MermaidRenderSpan::decoration(" : "),
            span(&relation.label),
        ]);
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
    if lines.next()? != "erDiagram" {
        return None;
    }

    let mut entities: Vec<ir::Entity> = Vec::new();
    let mut relations = Vec::new();
    let mut current: Option<usize> = None;
    let mut offset = "erDiagram".chars().count() + 1;
    for raw in source.lines().skip(1) {
        let trimmed = raw.trim();
        let leading = raw.chars().count() - raw.trim_start().chars().count();
        let base = offset + leading;
        if trimmed.is_empty() || trimmed.starts_with("%%") {
            offset += raw.chars().count() + 1;
            continue;
        }

        if trimmed == "}" {
            if current.take().is_none() {
                return None;
            }
        } else if let Some(index) = current {
            if trimmed.contains(['{', '}']) {
                return None;
            }
            entities[index].attributes.push(ir::Label {
                text: trimmed.to_string(),
                span: MermaidSourceSpan::new(base, base + trimmed.chars().count()),
            });
        } else if trimmed.ends_with('{') {
            let name = trimmed.strip_suffix('{')?.trim();
            if !valid_name(name)
                || entities
                    .iter()
                    .any(|entity: &ir::Entity| entity.name.text == name)
            {
                return None;
            }
            let at = find_char(trimmed, name)?;
            entities.push(ir::Entity {
                name: ir::Label {
                    text: name.to_string(),
                    span: MermaidSourceSpan::new(base + at, base + at + name.chars().count()),
                },
                attributes: Vec::new(),
            });
            current = Some(entities.len() - 1);
        } else {
            relations.push(parse_relation(trimmed, base)?);
        }
        offset += raw.chars().count() + 1;
    }

    if current.is_some() || entities.is_empty() || relations.is_empty() {
        return None;
    }
    let names = entities
        .iter()
        .map(|entity| entity.name.text.as_str())
        .collect::<HashSet<_>>();
    if relations.iter().any(|relation| {
        !names.contains(relation.from.text.as_str()) || !names.contains(relation.to.text.as_str())
    }) {
        return None;
    }
    Some(ir::Diagram {
        entities,
        relations,
    })
}

fn parse_relation(line: &str, base: usize) -> Option<ir::Relation> {
    let colon_byte = line.find(':')?;
    let label = line[colon_byte + 1..].trim();
    if label.is_empty() {
        return None;
    }
    let left = line[..colon_byte].trim();
    let mut words = left.split_whitespace();
    let from = words.next()?;
    let token = words.next()?;
    let to = words.next()?;
    if words.next().is_some() || !valid_name(from) || !valid_name(to) {
        return None;
    }
    let (from_cardinality, connector, to_cardinality) = parse_token(token)?;

    let from_at = find_char(line, from)?;
    let token_byte = line.find(token)?;
    let token_at = char_index(line, token_byte);
    let to_at = find_char(line, to)?;
    let label_byte = colon_byte + 1 + line[colon_byte + 1..].find(label)?;
    let label_at = char_index(line, label_byte);
    Some(ir::Relation {
        from: ir::Label {
            text: from.to_string(),
            span: MermaidSourceSpan::new(base + from_at, base + from_at + from.chars().count()),
        },
        from_cardinality: ir::Label {
            text: from_cardinality.to_string(),
            span: MermaidSourceSpan::new(
                base + token_at,
                base + token_at + from_cardinality.chars().count(),
            ),
        },
        connector,
        to_cardinality: ir::Label {
            text: to_cardinality.to_string(),
            span: MermaidSourceSpan::new(
                base + token_at + token.chars().count() - to_cardinality.chars().count(),
                base + token_at + token.chars().count(),
            ),
        },
        to: ir::Label {
            text: to.to_string(),
            span: MermaidSourceSpan::new(base + to_at, base + to_at + to.chars().count()),
        },
        label: ir::Label {
            text: label.to_string(),
            span: MermaidSourceSpan::new(base + label_at, base + label_at + label.chars().count()),
        },
    })
}

fn parse_token(token: &str) -> Option<(&'static str, &'static str, &'static str)> {
    for left in CARDINALITIES {
        for connector in ["--", ".."] {
            for right in CARDINALITIES {
                if token == format!("{left}{connector}{right}") {
                    return Some((left, connector, right));
                }
            }
        }
    }
    None
}

fn find_char(haystack: &str, needle: &str) -> Option<usize> {
    haystack.find(needle).map(|byte| char_index(haystack, byte))
}

fn char_index(value: &str, byte: usize) -> usize {
    value[..byte].chars().count()
}

fn valid_name(value: &str) -> bool {
    !value.is_empty()
        && value
            .chars()
            .all(|c| c.is_alphanumeric() || c == '_' || c == '-')
}
