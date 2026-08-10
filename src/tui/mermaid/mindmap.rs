use crate::tui::measure::display_width;

use super::{
    MermaidRenderSpan, MermaidSourceSpan,
    canvas::{MermaidCanvas, MermaidCanvasLabel},
    render_line_count_within_limits,
    routing::{RouteGrid, route_glyph},
    source_within_limits,
};

const COLUMN_GAP: usize = 5;
const LEAF_ROW_GAP: usize = 2;

#[derive(Debug)]
struct Node {
    label: String,
    span: MermaidSourceSpan,
    children: Vec<usize>,
}

pub(super) fn render(source: &str, width: usize) -> Option<Vec<Vec<MermaidRenderSpan>>> {
    let nodes = parse(source)?;
    let mut depths = vec![0; nodes.len()];
    assign_depths(&nodes, 0, 0, &mut depths);
    let depth_count = depths.iter().copied().max()?.checked_add(1)?;
    let mut column_widths = vec![0usize; depth_count];
    for (node, depth) in nodes.iter().zip(&depths) {
        column_widths[*depth] = column_widths[*depth].max(display_width(&node.label));
    }
    let mut columns = vec![0usize; depth_count];
    for depth in 1..depth_count {
        columns[depth] = columns[depth - 1]
            .checked_add(column_widths[depth - 1])?
            .checked_add(COLUMN_GAP)?;
    }
    let graph_width = columns.last()?.checked_add(*column_widths.last()?)?;
    if graph_width == 0 || graph_width > width {
        return None;
    }

    let mut rows = vec![0; nodes.len()];
    let mut next_leaf_row = 0;
    assign_rows(&nodes, 0, &mut next_leaf_row, &mut rows);
    let graph_height = rows.iter().copied().max()?.checked_add(1)?;
    if !render_line_count_within_limits(graph_height) {
        return None;
    }

    let mut routes = RouteGrid::new();
    for (index, node) in nodes.iter().enumerate() {
        if node.children.is_empty() {
            continue;
        }
        let parent_end = columns[depths[index]].checked_add(display_width(&node.label))?;
        let child_col = columns[depths[index].checked_add(1)?];
        if node.children.len() == 1 {
            let child = node.children[0];
            routes.connect((parent_end, rows[index]), (child_col - 1, rows[child]));
            continue;
        }

        let branch_col = child_col.checked_sub(3)?;
        routes.connect((parent_end, rows[index]), (branch_col, rows[index]));
        let first_row = rows[*node.children.first()?];
        let last_row = rows[*node.children.last()?];
        routes.connect((branch_col, first_row), (branch_col, last_row));
        for child in &node.children {
            routes.connect((branch_col, rows[*child]), (child_col - 1, rows[*child]));
        }
    }

    let mut canvas = MermaidCanvas {
        rows: Vec::new(),
        labels: Vec::new(),
    };
    for row in 0..graph_height {
        canvas.ensure_row(row, graph_width);
    }
    for ((col, row), mask) in routes.iter() {
        canvas.put(*col, *row, route_glyph(*mask));
    }
    for (index, node) in nodes.into_iter().enumerate() {
        canvas.labels.push(MermaidCanvasLabel {
            row: rows[index],
            col: columns[depths[index]],
            text: node.label,
            source: node.span,
        });
    }
    Some(canvas.render())
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

fn assign_depths(nodes: &[Node], index: usize, depth: usize, depths: &mut [usize]) {
    depths[index] = depth;
    for child in &nodes[index].children {
        assign_depths(nodes, *child, depth + 1, depths);
    }
}

fn assign_rows(nodes: &[Node], index: usize, next_leaf_row: &mut usize, rows: &mut [usize]) {
    if nodes[index].children.is_empty() {
        rows[index] = *next_leaf_row;
        *next_leaf_row += LEAF_ROW_GAP;
        return;
    }
    for child in &nodes[index].children {
        assign_rows(nodes, *child, next_leaf_row, rows);
    }
    let first = rows[nodes[index].children[0]];
    let last = rows[*nodes[index].children.last().unwrap()];
    rows[index] = (first + last) / 2;
}
