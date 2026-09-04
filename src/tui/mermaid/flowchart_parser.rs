//! Mermaid source spans use Unicode scalar offsets into the local diagram source.

use std::collections::{HashMap, HashSet};

use super::flowchart_ir::{
    MermaidDirection, MermaidEdge, MermaidEdgeStyle, MermaidGraph, MermaidLabel, MermaidNode,
    MermaidShape,
};

pub(crate) fn parse(source: &str) -> Option<MermaidGraph> {
    if source.contains('\r') {
        return None;
    }
    let mut nodes = HashMap::new();
    let mut explicit_nodes = HashSet::new();
    let mut edges = Vec::new();
    let mut direction = None;
    let mut line_offset = 0usize;
    let mut subgraph_depth = 0usize;
    for (line_index, line) in source.lines().enumerate() {
        let leading = line.chars().count() - line.trim_start().chars().count();
        let trimmed = line.trim();
        if line_index == 0 {
            let mut tokens = trimmed
                .split(|ch: char| ch == ';' || ch.is_whitespace())
                .filter(|s| !s.is_empty());
            let keyword = tokens.next()?;
            let dir = tokens.next()?;
            direction = Some(match (keyword, dir) {
                ("graph" | "flowchart", "TD" | "TB") => MermaidDirection::Td,
                ("graph" | "flowchart", "BT") => MermaidDirection::Bu,
                ("graph" | "flowchart", "LR") => MermaidDirection::Lr,
                ("graph" | "flowchart", "RL") => MermaidDirection::Rl,
                _ => return None,
            });
        } else if !trimmed.is_empty() && !trimmed.starts_with("%%") {
            if trimmed == "end" {
                if subgraph_depth == 0 {
                    return None;
                }
                subgraph_depth -= 1;
            } else if trimmed.starts_with("subgraph")
                && (trimmed == "subgraph" || trimmed[8..].starts_with(char::is_whitespace))
            {
                subgraph_depth += 1;
            } else if subgraph_depth > 0 && trimmed.starts_with("direction ") {
                parse_local_direction(trimmed)?;
                // Local subgraph direction is accepted even though the terminal layout
                // currently uses the parent graph direction for the flattened topology.
            } else {
                let base_line = line_offset + leading;
                let mut segment_start = 0usize;
                for (idx, byte) in trimmed.bytes().enumerate() {
                    if byte == b';' {
                        let segment = &trimmed[segment_start..idx];
                        if !segment.trim().is_empty() && !is_structural_statement(segment) {
                            parse_statement(
                                segment,
                                base_line + trimmed[..segment_start].chars().count(),
                                &mut nodes,
                                &mut explicit_nodes,
                                &mut edges,
                            )?;
                        }
                        segment_start = idx + 1;
                    }
                }
                let segment = &trimmed[segment_start..];
                if !segment.trim().is_empty() && !is_structural_statement(segment) {
                    parse_statement(
                        segment,
                        base_line + trimmed[..segment_start].chars().count(),
                        &mut nodes,
                        &mut explicit_nodes,
                        &mut edges,
                    )?;
                }
            }
        }
        line_offset += line.chars().count() + 1;
    }
    let direction = direction?;
    if subgraph_depth != 0
        || edges.is_empty()
        || edges
            .iter()
            .any(|edge| !nodes.contains_key(&edge.from) || !nodes.contains_key(&edge.to))
    {
        return None;
    }
    Some(MermaidGraph {
        direction,
        nodes,
        edges,
    })
}

const SHAPES: &[(&str, &str, MermaidShape)] = &[
    ("[[", "]]", MermaidShape::Subroutine),
    ("[(", ")]", MermaidShape::Cylinder),
    ("((", "))", MermaidShape::Circle),
    ("{{", "}}", MermaidShape::Hexagon),
    ("[/", "/]", MermaidShape::Parallelogram),
    ("[\\", "\\]", MermaidShape::Parallelogram),
    ("[", "]", MermaidShape::Rectangle),
    ("(", ")", MermaidShape::RoundRect),
    ("{", "}", MermaidShape::Diamond),
    (">", "]", MermaidShape::Stadium),
];

const MAX_EXPANDED_EDGES: usize = 64;

fn parse_statement(
    segment: &str,
    base: usize,
    nodes: &mut HashMap<String, MermaidNode>,
    explicit_nodes: &mut HashSet<String>,
    edges: &mut Vec<MermaidEdge>,
) -> Option<()> {
    let (from_group, rest, rest_base) = parse_endpoint_group(segment, base)?;
    if rest.trim().is_empty() {
        for endpoint in from_group {
            insert_node(nodes, explicit_nodes, endpoint)?;
        }
        return Some(());
    }

    let mut pending_nodes = from_group;
    let mut pending_edges = Vec::new();
    let (mut style, mut arrow, mut reverse_arrow, mut label, mut rest, mut rest_base) =
        parse_connector(rest, rest_base)?;
    let mut left_group = pending_nodes.clone();

    loop {
        let (right_group, next_rest, next_rest_base) = parse_endpoint_group(rest, rest_base)?;
        pending_nodes.extend(right_group.iter().cloned());
        let edge_count = left_group.len().checked_mul(right_group.len())?;
        if pending_edges.len().checked_add(edge_count)? > MAX_EXPANDED_EDGES {
            return None;
        }
        for (from, _, _) in &left_group {
            for (to, _, _) in &right_group {
                pending_edges.push(MermaidEdge {
                    from: from.clone(),
                    to: to.clone(),
                    label: label.clone(),
                    style,
                    arrow,
                    reverse_arrow,
                });
            }
        }

        rest = next_rest;
        rest_base = next_rest_base;
        if rest.trim().is_empty() {
            break;
        }
        (style, arrow, reverse_arrow, label, rest, rest_base) = parse_connector(rest, rest_base)?;
        left_group = right_group;
    }

    for endpoint in pending_nodes {
        insert_node(nodes, explicit_nodes, endpoint)?;
    }
    edges.extend(pending_edges);
    Some(())
}

fn parse_endpoint_group(
    segment: &str,
    base: usize,
) -> Option<(Vec<(String, MermaidNode, bool)>, &str, usize)> {
    let (first_id, first_node, first_explicit, mut rest, mut rest_base) =
        parse_endpoint_prefix(segment, base)?;
    let mut endpoints = vec![(first_id, first_node, first_explicit)];
    loop {
        let leading = rest.chars().take_while(|ch| ch.is_whitespace()).count();
        let trimmed = rest.trim_start();
        if !trimmed.starts_with('&') {
            return Some((endpoints, rest, rest_base));
        }
        rest = &trimmed[1..];
        rest_base += leading + 1;
        let (id, node, explicit, next_rest, next_base) = parse_endpoint_prefix(rest, rest_base)?;
        endpoints.push((id, node, explicit));
        rest = next_rest;
        rest_base = next_base;
    }
}

fn parse_endpoint_prefix(
    segment: &str,
    base: usize,
) -> Option<(String, MermaidNode, bool, &str, usize)> {
    let leading = segment.chars().take_while(|ch| ch.is_whitespace()).count();
    let trimmed = segment.trim_start();
    if trimmed.is_empty() {
        return None;
    }
    let tbase = base + leading;
    for (opener, closer, shape) in SHAPES {
        if let Some(open) = trimmed.find(opener) {
            let id = trimmed[..open].trim();
            if !valid_id(id) {
                continue;
            }
            let body_start = open + opener.len();
            let rel = trimmed[body_start..].find(closer)?;
            let close = body_start + rel;
            let label_seg = &trimmed[body_start..close];
            let label_leading = label_seg
                .chars()
                .take_while(|ch| ch.is_whitespace())
                .count();
            let raw = label_seg.trim();
            let quoted = raw.starts_with('"') || raw.ends_with('"');
            let raw = if quoted {
                raw.strip_prefix('"')?.strip_suffix('"')?
            } else {
                raw
            };
            if raw.is_empty() {
                continue;
            }
            let id_chars = trimmed[..open].chars().count();
            let start = tbase
                + id_chars
                + opener.chars().count()
                + label_leading
                + usize::from(label_seg.trim_start().starts_with('"'));
            let (label, atomic) = parse_label(raw, true)?;
            let node = MermaidNode {
                label,
                start,
                end: start + raw.chars().count(),
                atomic,
                shape: *shape,
            };
            let rest = &trimmed[close + closer.len()..];
            let rest_base = tbase + trimmed[..close + closer.len()].chars().count();
            return Some((id.to_string(), node, true, rest, rest_base));
        }
    }
    let id_end = bare_id_end(trimmed);
    let id = &trimmed[..id_end];
    if !valid_id(id) {
        return None;
    }
    Some((
        id.to_string(),
        MermaidNode {
            label: id.to_string(),
            start: tbase,
            end: tbase + id.chars().count(),
            atomic: false,
            shape: MermaidShape::Rectangle,
        },
        false,
        &trimmed[id_end..],
        tbase + id.chars().count(),
    ))
}

fn parse_connector(
    segment: &str,
    base: usize,
) -> Option<(
    MermaidEdgeStyle,
    bool,
    bool,
    Option<MermaidLabel>,
    &str,
    usize,
)> {
    let leading = segment.chars().take_while(|ch| ch.is_whitespace()).count();
    let segment = segment.trim_start();
    let base = base + leading;
    const TOKENS: &[(&str, MermaidEdgeStyle, bool, bool)] = &[
        ("<-->", MermaidEdgeStyle::Solid, true, true),
        ("-.->", MermaidEdgeStyle::Dashed, true, false),
        ("--->", MermaidEdgeStyle::Solid, true, false),
        ("-->", MermaidEdgeStyle::Solid, true, false),
        ("==>", MermaidEdgeStyle::Thick, true, false),
        ("-.-", MermaidEdgeStyle::Dashed, false, false),
        ("---", MermaidEdgeStyle::Solid, false, false),
        ("===", MermaidEdgeStyle::Thick, false, false),
    ];
    for (token, style, arrow, reverse_arrow) in TOKENS {
        if let Some(rest) = segment.strip_prefix(token) {
            let (label, rest, rest_base) = parse_pipe_label(rest, base + token.chars().count())?;
            return Some((*style, *arrow, *reverse_arrow, label, rest, rest_base));
        }
    }
    let (style, extension) = if segment.starts_with("--") {
        (MermaidEdgeStyle::Solid, '-')
    } else if segment.starts_with("-.") {
        (MermaidEdgeStyle::Dashed, '.')
    } else if segment.starts_with("==") {
        (MermaidEdgeStyle::Thick, '=')
    } else {
        return None;
    };
    if segment[2..].starts_with(extension) {
        return None;
    }
    let rest = segment[2..].trim_start();
    let rest_base = base + 2 + segment[2..].chars().count() - rest.chars().count();
    const CLOSERS: &[(&str, MermaidEdgeStyle, bool, bool)] = &[
        ("--->", MermaidEdgeStyle::Solid, true, false),
        ("-->", MermaidEdgeStyle::Solid, true, false),
        (".->", MermaidEdgeStyle::Dashed, true, false),
        ("==>", MermaidEdgeStyle::Thick, true, false),
        ("---", MermaidEdgeStyle::Solid, false, false),
        ("-.-", MermaidEdgeStyle::Dashed, false, false),
        ("===", MermaidEdgeStyle::Thick, false, false),
    ];
    for (closer, cstyle, carrow, creverse_arrow) in CLOSERS {
        if let Some(pos) = rest.find(closer) {
            if rest[..pos].ends_with(extension) {
                return None;
            }
            let raw = rest[..pos].trim();
            if !raw.is_empty() && *cstyle == style {
                let leading =
                    rest[..pos].chars().count() - rest[..pos].trim_start().chars().count();
                let start = rest_base + leading;
                return Some((
                    style,
                    *carrow,
                    *creverse_arrow,
                    Some({
                        let (text, atomic) = parse_label(raw, false)?;
                        MermaidLabel {
                            text,
                            start,
                            end: start + raw.chars().count(),
                            atomic,
                        }
                    }),
                    &rest[pos + closer.len()..],
                    rest_base + rest[..pos + closer.len()].chars().count(),
                ));
            }
        }
    }
    None
}

fn parse_pipe_label(segment: &str, base: usize) -> Option<(Option<MermaidLabel>, &str, usize)> {
    let leading = segment.chars().take_while(|ch| ch.is_whitespace()).count();
    let trimmed = segment.trim_start();
    let tbase = base + leading;
    if !trimmed.starts_with('|') {
        return Some((None, trimmed, tbase));
    }
    let close = trimmed[1..].find('|')? + 1;
    let raw = &trimmed[1..close];
    let lt = raw.trim();
    if lt.is_empty() {
        return None;
    }
    let start = tbase + 1 + raw.chars().count() - raw.trim_start().chars().count();
    let (text, atomic) = parse_label(lt, false)?;
    Some((
        Some(MermaidLabel {
            text,
            start,
            end: start + lt.chars().count(),
            atomic,
        }),
        &trimmed[close + 1..],
        tbase + trimmed[..close + 1].chars().count(),
    ))
}

// Terminal rendering preserves the topology while intentionally ignoring CSS styling.
fn is_structural_statement(segment: &str) -> bool {
    let trimmed = segment.trim_start();
    ["classDef", "class", "style"].iter().any(|keyword| {
        trimmed
            .strip_prefix(keyword)
            .is_some_and(|rest| rest.starts_with(char::is_whitespace))
    })
}

fn parse_label(raw: &str, multiline: bool) -> Option<(String, bool)> {
    let (text, atomic) = super::render_math_label(raw)?;
    let text = text
        .replace("<br />", "\n")
        .replace("<br/>", "\n")
        .replace("<br>", "\n");
    let transformed = text.contains('\n');
    let text = if multiline {
        text
    } else {
        text.split('\n').collect::<Vec<_>>().join(" / ")
    };
    Some((text, atomic || transformed))
}

fn insert_node(
    nodes: &mut HashMap<String, MermaidNode>,
    explicit_nodes: &mut HashSet<String>,
    (id, node, explicit): (String, MermaidNode, bool),
) -> Option<()> {
    if explicit {
        explicit_nodes.insert(id.clone());
        nodes.insert(id, node);
    } else if !explicit_nodes.contains(&id) {
        nodes.insert(id, node);
    }
    Some(())
}
fn parse_local_direction(line: &str) -> Option<MermaidDirection> {
    match line.strip_prefix("direction ")?.trim() {
        "TD" | "TB" => Some(MermaidDirection::Td),
        "BT" => Some(MermaidDirection::Bu),
        "LR" => Some(MermaidDirection::Lr),
        "RL" => Some(MermaidDirection::Rl),
        _ => None,
    }
}

fn bare_id_end(value: &str) -> usize {
    for (index, ch) in value.char_indices() {
        if ch.is_whitespace() || ch == '&' {
            return index;
        }
        let rest = &value[index..];
        if rest.starts_with("--") || rest.starts_with("-.") || rest.starts_with("==") {
            return index;
        }
    }
    value.len()
}

fn valid_id(value: &str) -> bool {
    !value.is_empty()
        && value
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '_' || ch == '-')
}
