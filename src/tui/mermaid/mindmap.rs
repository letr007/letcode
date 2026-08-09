use crate::tui::measure::display_width;

use super::{
    MermaidRenderSpan, MermaidSourceSpan, render_line_count_within_limits, source_within_limits,
};

#[derive(Debug)]
struct Node {
    label: String,
    span: MermaidSourceSpan,
    children: Vec<usize>,
}

pub(super) fn render(source: &str, width: usize) -> Option<Vec<Vec<MermaidRenderSpan>>> {
    let nodes = parse(source)?;
    let mut lines = Vec::new();
    render_node(&nodes, 0, &[], true, &mut lines);
    (render_line_count_within_limits(lines.len()) && fits(&lines, width)).then_some(lines)
}

fn parse(source: &str) -> Option<Vec<Node>> {
    if !source_within_limits(source) || source.contains('\r') {
        return None;
    }
    let mut nodes = Vec::new();
    let mut stack = Vec::<(usize, usize)>::new();
    let mut offset = source.lines().next()?.chars().count() + 1;
    let mut root_indent = None;

    for line in source.split('\n').skip(1) {
        let line_len = line.chars().count();
        if line.contains('\t') || line.chars().last().is_some_and(char::is_whitespace) {
            return None;
        }
        let indent = line
            .chars()
            .take_while(|character| *character == ' ')
            .count();
        let trimmed = line.get(indent..)?;
        if trimmed.is_empty() || trimmed.starts_with("%%") {
            offset += line_len + 1;
            continue;
        }
        let root = *root_indent.get_or_insert(indent);
        if nodes.is_empty() && indent != root {
            return None;
        }
        if !nodes.is_empty() && indent <= root {
            return None;
        }

        let (label, span) = parse_label(trimmed, offset + indent)?;
        let index = nodes.len();
        nodes.push(Node {
            label,
            span,
            children: Vec::new(),
        });
        while stack
            .last()
            .is_some_and(|(parent_indent, _)| *parent_indent >= indent)
        {
            stack.pop();
        }
        if let Some((_, parent)) = stack.last() {
            nodes[*parent].children.push(index);
        } else if index != 0 {
            return None;
        }
        stack.push((indent, index));
        offset += line_len + 1;
    }
    (!nodes.is_empty()).then_some(nodes)
}

fn parse_label(text: &str, start: usize) -> Option<(String, MermaidSourceSpan)> {
    if text.contains(":::")
        || text.contains("::icon(")
        || text.starts_with("icon ")
        || text.starts_with("icon:")
        || text.contains('`')
        || text.contains(['<', '>'])
    {
        return None;
    }
    for (open, close, reversed) in [
        ("((", "))", false),
        ("{{", "}}", false),
        ("[", "]", false),
        ("(", ")", false),
        ("))", "((", true),
        (")", "(", true),
    ] {
        let (id, raw_label, label_chars) = if reversed {
            let Some(open_byte) = text.find(open) else {
                continue;
            };
            let Some(body) = text.strip_suffix(close) else {
                continue;
            };
            (
                &text[..open_byte],
                &body[open_byte + open.len()..],
                text[..open_byte].chars().count() + open.chars().count(),
            )
        } else {
            let Some(body) = text.strip_suffix(close) else {
                continue;
            };
            let Some(open_byte) = body.find(open) else {
                continue;
            };
            (
                &body[..open_byte],
                &body[open_byte + open.len()..],
                text[..open_byte].chars().count() + open.chars().count(),
            )
        };
        if id.is_empty()
            || !id.chars().all(is_identifier_character)
            || raw_label.is_empty()
            || raw_label
                .chars()
                .any(|character| matches!(character, '[' | ']' | '(' | ')' | '{' | '}'))
        {
            return None;
        }
        let (label, quote_offset) = if let Some(quoted) = raw_label
            .strip_prefix('"')
            .and_then(|label| label.strip_suffix('"'))
        {
            if quoted.is_empty() || quoted.contains('"') {
                return None;
            }
            (quoted, 1)
        } else if raw_label.contains('"') {
            return None;
        } else {
            (raw_label, 0)
        };
        if !label_is_supported(label) {
            return None;
        }
        let label_start = start + label_chars + quote_offset;
        return Some((
            label.to_string(),
            MermaidSourceSpan::new(label_start, label_start + label.chars().count()),
        ));
    }
    if text
        .chars()
        .any(|character| matches!(character, '[' | ']' | '(' | ')' | '{' | '}' | '#'))
        || !label_is_supported(text)
    {
        return None;
    }
    Some((
        text.to_string(),
        MermaidSourceSpan::new(start, start + text.chars().count()),
    ))
}

fn is_identifier_character(character: char) -> bool {
    character.is_alphanumeric() || matches!(character, '_' | '-')
}

fn label_is_supported(label: &str) -> bool {
    !label.is_empty()
        && !label.contains(":::")
        && !label.contains("::icon(")
        && !label.contains(['\n', '<', '>'])
}

fn render_node(
    nodes: &[Node],
    index: usize,
    ancestors_last: &[bool],
    is_last: bool,
    lines: &mut Vec<Vec<MermaidRenderSpan>>,
) {
    let node = &nodes[index];
    let mut line = Vec::new();
    if index == 0 {
        line.push(MermaidRenderSpan::source(
            node.label.clone(),
            node.span,
            false,
        ));
    } else {
        let mut prefix = String::new();
        for ancestor_last in ancestors_last {
            prefix.push_str(if *ancestor_last { "   " } else { "│  " });
        }
        line.push(MermaidRenderSpan::decoration(prefix));
        line.push(MermaidRenderSpan::decoration(if is_last {
            "└─ "
        } else {
            "├─ "
        }));
        line.push(MermaidRenderSpan::source(
            node.label.clone(),
            node.span,
            false,
        ));
    }
    lines.push(line);

    let mut child_ancestors = ancestors_last.to_vec();
    if index != 0 {
        child_ancestors.push(is_last);
    }
    for (child_index, child) in node.children.iter().enumerate() {
        render_node(
            nodes,
            *child,
            &child_ancestors,
            child_index + 1 == node.children.len(),
            lines,
        );
    }
}

fn fits(lines: &[Vec<MermaidRenderSpan>], width: usize) -> bool {
    lines.iter().all(|line| {
        line.iter()
            .map(|span| display_width(&span.text))
            .sum::<usize>()
            <= width
    })
}
