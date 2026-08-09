use super::class_ir as ir;
use super::{
    MermaidRenderSpan, MermaidSourceSpan, render_line_count_within_limits, source_within_limits,
};
use crate::tui::measure::display_width;
use std::collections::HashSet;

const CONNECTORS: [&str; 6] = ["<|--", "..|>", "*--", "o--", "..>", "-->"];

pub(super) fn render(source: &str, width: usize) -> Option<Vec<Vec<MermaidRenderSpan>>> {
    let diagram = parse(source)?;
    let mut lines = Vec::new();
    for class in &diagram.classes {
        lines.push(vec![
            MermaidRenderSpan::decoration("╭─ "),
            span(&class.name),
        ]);
        for member in &class.members {
            lines.push(vec![MermaidRenderSpan::decoration("│ "), span(member)]);
        }
        lines.push(vec![MermaidRenderSpan::decoration("╰─")]);
    }
    for relation in &diagram.relations {
        let mut line = vec![
            span(&relation.from),
            MermaidRenderSpan::decoration(format!(" {} ", relation.connector)),
            span(&relation.to),
        ];
        if let Some(label) = &relation.label {
            line.extend([MermaidRenderSpan::decoration(" : "), span(label)]);
        }
        lines.push(line);
    }
    (render_line_count_within_limits(lines.len()) && fits(&lines, width)).then_some(lines)
}

fn span(label: &ir::Label) -> MermaidRenderSpan {
    MermaidRenderSpan::source(label.text.clone(), label.span, false)
}

fn fits(lines: &[Vec<MermaidRenderSpan>], width: usize) -> bool {
    lines
        .iter()
        .all(|line| line.iter().map(|s| display_width(&s.text)).sum::<usize>() <= width)
}

fn parse(source: &str) -> Option<ir::Diagram> {
    if !source_within_limits(source) || source.contains('\r') {
        return None;
    }
    let mut lines = source.lines();
    if lines.next()? != "classDiagram" {
        return None;
    }

    let mut classes: Vec<ir::Class> = Vec::new();
    let mut relations = Vec::new();
    let mut current: Option<usize> = None;
    let mut offset = "classDiagram".chars().count() + 1;
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
            if trimmed.contains(['{', '}'])
                || trimmed.starts_with("<<")
                || trimmed.starts_with("class ")
                || contains_connector(trimmed)
            {
                return None;
            }
            classes[index].members.push(ir::Label {
                text: trimmed.to_string(),
                span: MermaidSourceSpan::new(base, base + trimmed.chars().count()),
            });
        } else if let Some(rest) = trimmed.strip_prefix("class ") {
            let (name, block) = rest
                .strip_suffix('{')
                .map_or((rest.trim(), false), |name| (name.trim(), true));
            if !valid_name(name)
                || classes
                    .iter()
                    .any(|class: &ir::Class| class.name.text == name)
            {
                return None;
            }
            let name_start = base + find_char(trimmed, name)?;
            classes.push(ir::Class {
                name: ir::Label {
                    text: name.to_string(),
                    span: MermaidSourceSpan::new(name_start, name_start + name.chars().count()),
                },
                members: Vec::new(),
            });
            current = block.then_some(classes.len() - 1);
        } else if contains_connector(trimmed) {
            relations.push(parse_relation(trimmed, base)?);
        } else if let Some((name, member)) = trimmed.split_once(':') {
            let name = name.trim();
            let member = member.trim();
            if !valid_name(name) || member.is_empty() || member.contains(['{', '}']) {
                return None;
            }
            let name_at = base + find_char(trimmed, name)?;
            let member_at = base + find_char(trimmed, member)?;
            let index =
                if let Some(index) = classes.iter().position(|class| class.name.text == name) {
                    index
                } else {
                    classes.push(ir::Class {
                        name: ir::Label {
                            text: name.to_string(),
                            span: MermaidSourceSpan::new(name_at, name_at + name.chars().count()),
                        },
                        members: Vec::new(),
                    });
                    classes.len() - 1
                };
            classes[index].members.push(ir::Label {
                text: member.to_string(),
                span: MermaidSourceSpan::new(member_at, member_at + member.chars().count()),
            });
        } else {
            return None;
        }
        offset += raw.chars().count() + 1;
    }

    if current.is_some() || classes.is_empty() {
        return None;
    }
    let names = classes
        .iter()
        .map(|class| class.name.text.as_str())
        .collect::<HashSet<_>>();
    if relations.iter().any(|relation| {
        !names.contains(relation.from.text.as_str()) || !names.contains(relation.to.text.as_str())
    }) {
        return None;
    }
    Some(ir::Diagram { classes, relations })
}

fn parse_relation(line: &str, base: usize) -> Option<ir::Relation> {
    let (connector_byte, connector) = CONNECTORS
        .iter()
        .filter_map(|token| line.find(token).map(|at| (at, *token)))
        .min_by_key(|(at, _)| *at)?;
    let from = line[..connector_byte].trim();
    let after_byte = connector_byte + connector.len();
    let after = &line[after_byte..];
    let (to, label) = after
        .split_once(':')
        .map_or((after.trim(), None), |(to, label)| {
            (to.trim(), Some(label.trim()))
        });
    if !valid_name(from) || !valid_name(to) || label.is_some_and(str::is_empty) {
        return None;
    }

    let from_at = find_char(line, from)?;
    let to_byte = after_byte + after.find(to)?;
    let to_at = char_index(line, to_byte);
    let label = label.map(|label| {
        let label_byte = after_byte + after.rfind(label).expect("label is in relation tail");
        let label_at = char_index(line, label_byte);
        ir::Label {
            text: label.to_string(),
            span: MermaidSourceSpan::new(base + label_at, base + label_at + label.chars().count()),
        }
    });
    Some(ir::Relation {
        from: ir::Label {
            text: from.to_string(),
            span: MermaidSourceSpan::new(base + from_at, base + from_at + from.chars().count()),
        },
        to: ir::Label {
            text: to.to_string(),
            span: MermaidSourceSpan::new(base + to_at, base + to_at + to.chars().count()),
        },
        label,
        connector,
    })
}

fn contains_connector(value: &str) -> bool {
    CONNECTORS.iter().any(|connector| value.contains(connector))
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
