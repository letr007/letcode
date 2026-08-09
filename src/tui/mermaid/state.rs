use super::state_ir as ir;
use super::{
    MermaidRenderSpan, MermaidSourceSpan, render_line_count_within_limits, source_within_limits,
};
use crate::tui::measure::display_width;
use std::collections::HashMap;

const MAX_DEPTH: usize = 4;

pub(super) fn render(source: &str, width: usize) -> Option<Vec<Vec<MermaidRenderSpan>>> {
    let diagram = parse(source)?;
    let mut lines = Vec::new();
    for item in &diagram.items {
        match item {
            ir::Item::State(state) => lines.push(vec![
                MermaidRenderSpan::decoration("  ".repeat(state.depth)),
                span(&state.label),
                MermaidRenderSpan::decoration(if state.composite { " {" } else { "" }),
            ]),
            ir::Item::Transition(transition) => {
                let mut line = vec![
                    MermaidRenderSpan::decoration("  ".repeat(transition.depth)),
                    span(&transition.from),
                    MermaidRenderSpan::decoration(" --> "),
                    span(&transition.to),
                ];
                if let Some(label) = &transition.label {
                    line.extend([MermaidRenderSpan::decoration(" : "), span(label)]);
                }
                lines.push(line);
            }
            ir::Item::Close(depth) => lines.push(vec![
                MermaidRenderSpan::decoration("  ".repeat(*depth)),
                MermaidRenderSpan::decoration("}"),
            ]),
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
    let header = lines.next()?;
    if !matches!(header, "stateDiagram" | "stateDiagram-v2") {
        return None;
    }

    let mut items = Vec::new();
    let mut states = HashMap::<String, String>::new();
    let mut stack = Vec::<String>::new();
    let mut offset = header.chars().count() + 1;
    for raw in source.lines().skip(1) {
        let trimmed = raw.trim();
        let leading = raw.chars().count() - raw.trim_start().chars().count();
        let base = offset + leading;
        if trimmed.is_empty() || trimmed.starts_with("%%") {
            offset += raw.chars().count() + 1;
            continue;
        }
        if unsupported(trimmed) {
            return None;
        }

        if trimmed == "}" {
            if stack.pop().is_none() {
                return None;
            }
            items.push(ir::Item::Close(stack.len()));
        } else if let Some(rest) = trimmed.strip_prefix("state ") {
            let declaration = parse_state_declaration(rest)?;
            if stack.len() >= MAX_DEPTH || states.contains_key(declaration.id) {
                return None;
            }
            let label_at = base + find_char(trimmed, declaration.label)?;
            states.insert(declaration.id.to_string(), declaration.label.to_string());
            items.push(ir::Item::State(ir::State {
                label: ir::Label {
                    text: declaration.label.to_string(),
                    span: MermaidSourceSpan::new(
                        label_at,
                        label_at + declaration.label.chars().count(),
                    ),
                },
                composite: declaration.composite,
                depth: stack.len(),
            }));
            if declaration.composite {
                stack.push(declaration.id.to_string());
            }
        } else if trimmed.ends_with('{') {
            return None;
        } else {
            let transition = parse_transition(trimmed, base, stack.len())?;
            for endpoint in [&transition.from.text, &transition.to.text] {
                if endpoint != "[*]" && !states.contains_key(endpoint) {
                    states.insert(endpoint.clone(), endpoint.clone());
                }
            }
            items.push(ir::Item::Transition(transition));
        }
        offset += raw.chars().count() + 1;
    }

    if !stack.is_empty() || items.is_empty() {
        return None;
    }
    Some(ir::Diagram { items })
}

struct StateDeclaration<'a> {
    id: &'a str,
    label: &'a str,
    composite: bool,
}

fn parse_state_declaration(rest: &str) -> Option<StateDeclaration<'_>> {
    let rest = rest.trim();
    if let Some(block) = rest.strip_suffix('{') {
        let id = block.trim();
        if !valid_id(id) {
            return None;
        }
        return Some(StateDeclaration {
            id,
            label: id,
            composite: true,
        });
    }
    if let Some(quoted) = rest.strip_prefix('"') {
        let quote_end = quoted.find('"')?;
        let label = &quoted[..quote_end];
        let tail = quoted[quote_end + 1..].trim();
        let id = tail.strip_prefix("as ")?.trim();
        if label.is_empty() || !valid_id(id) {
            return None;
        }
        return Some(StateDeclaration {
            id,
            label,
            composite: false,
        });
    }
    if !valid_id(rest) {
        return None;
    }
    Some(StateDeclaration {
        id: rest,
        label: rest,
        composite: false,
    })
}

fn parse_transition(line: &str, base: usize, depth: usize) -> Option<ir::Transition> {
    let arrow_byte = line.find("-->")?;
    if line[arrow_byte + 3..].contains("-->") {
        return None;
    }
    let from = line[..arrow_byte].trim();
    let after_byte = arrow_byte + 3;
    let after = &line[after_byte..];
    let (to, label) = after
        .split_once(':')
        .map_or((after.trim(), None), |(to, label)| {
            (to.trim(), Some(label.trim()))
        });
    if !valid_endpoint(from) || !valid_endpoint(to) || label.is_some_and(str::is_empty) {
        return None;
    }

    let from_at = find_char(line, from)?;
    let to_byte = after_byte + after.find(to)?;
    let to_at = char_index(line, to_byte);
    let label = label.map(|label| {
        let label_byte = after_byte + after.rfind(label).expect("label is in transition tail");
        let label_at = char_index(line, label_byte);
        ir::Label {
            text: label.to_string(),
            span: MermaidSourceSpan::new(base + label_at, base + label_at + label.chars().count()),
        }
    });
    Some(ir::Transition {
        from: ir::Label {
            text: from.to_string(),
            span: MermaidSourceSpan::new(base + from_at, base + from_at + from.chars().count()),
        },
        to: ir::Label {
            text: to.to_string(),
            span: MermaidSourceSpan::new(base + to_at, base + to_at + to.chars().count()),
        },
        label,
        depth,
    })
}

fn unsupported(line: &str) -> bool {
    line == "--"
        || line.starts_with("note ")
        || line.starts_with("direction ")
        || line.starts_with("classDef ")
        || line.starts_with("class ")
        || line.starts_with("fork ")
        || line.starts_with("join ")
        || line.contains("<<fork>>")
        || line.contains("<<join>>")
}

fn valid_endpoint(value: &str) -> bool {
    value == "[*]" || valid_id(value)
}

fn valid_id(value: &str) -> bool {
    !value.is_empty()
        && value
            .chars()
            .all(|c| c.is_alphanumeric() || c == '_' || c == '-')
}

fn find_char(haystack: &str, needle: &str) -> Option<usize> {
    haystack.find(needle).map(|byte| char_index(haystack, byte))
}

fn char_index(value: &str, byte: usize) -> usize {
    value[..byte].chars().count()
}
