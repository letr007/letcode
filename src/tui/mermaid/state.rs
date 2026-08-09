use super::state_ir as ir;
use super::{
    MermaidRenderSpan, MermaidSourceSpan, canvas, render_line_count_within_limits,
    source_within_limits,
};
use crate::tui::measure::display_width;
use std::collections::{HashMap, VecDeque};

const MAX_DEPTH: usize = 4;
const START_STATE: &str = "\0start";
const END_STATE: &str = "\0end";

fn endpoint_key(endpoint: &str, from: bool) -> &str {
    if endpoint == "[*]" {
        if from { START_STATE } else { END_STATE }
    } else {
        endpoint
    }
}

pub(super) fn render(source: &str, width: usize) -> Option<Vec<Vec<MermaidRenderSpan>>> {
    let diagram = parse(source)?;
    let composite = diagram.items.iter().any(|item| {
        matches!(item, ir::Item::State(state) if state.composite)
            || matches!(item, ir::Item::Transition(transition) if transition.depth > 0)
    });
    if !composite && let Some(canvas) = layout(&diagram, width) {
        let lines = canvas.render();
        if render_line_count_within_limits(lines.len()) {
            return Some(lines);
        }
    }
    render_boxed_transitions(&diagram, width).or_else(|| render_linear(&diagram, width))
}

fn layout(diagram: &ir::Diagram, width: usize) -> Option<canvas::MermaidCanvas> {
    let declared = diagram
        .items
        .iter()
        .filter_map(|item| match item {
            ir::Item::State(state) => Some((state.id.as_str(), &state.label)),
            _ => None,
        })
        .collect::<HashMap<_, _>>();
    let transitions = diagram
        .items
        .iter()
        .filter_map(|item| match item {
            ir::Item::Transition(transition) => Some(transition),
            _ => None,
        })
        .collect::<Vec<_>>();
    if transitions.is_empty() || transitions.len() > 48 {
        return None;
    }

    let mut ids = Vec::new();
    for transition in &transitions {
        for endpoint in [
            endpoint_key(&transition.from.text, true),
            endpoint_key(&transition.to.text, false),
        ] {
            if !ids.contains(&endpoint) {
                ids.push(endpoint);
            }
        }
    }
    if ids.len() > 24 {
        return None;
    }
    let index = ids
        .iter()
        .enumerate()
        .map(|(index, id)| (*id, index))
        .collect::<HashMap<_, _>>();
    let mut indegree = vec![0usize; ids.len()];
    let mut outgoing = vec![Vec::new(); ids.len()];
    for transition in &transitions {
        let from = index[endpoint_key(&transition.from.text, true)];
        let to = index[endpoint_key(&transition.to.text, false)];
        if from == to {
            return None;
        }
        outgoing[from].push(to);
        indegree[to] += 1;
    }
    let mut queue = indegree
        .iter()
        .enumerate()
        .filter_map(|(index, degree)| (*degree == 0).then_some(index))
        .collect::<VecDeque<_>>();
    let mut depths = vec![0usize; ids.len()];
    let mut visited = 0;
    while let Some(from) = queue.pop_front() {
        visited += 1;
        for to in &outgoing[from] {
            depths[*to] = depths[*to].max(depths[from] + 1);
            indegree[*to] -= 1;
            if indegree[*to] == 0 {
                queue.push_back(*to);
            }
        }
    }
    if visited != ids.len()
        || transitions.iter().any(|transition| {
            depths[index[endpoint_key(&transition.to.text, false)]]
                != depths[index[endpoint_key(&transition.from.text, true)]] + 1
        })
    {
        return None;
    }

    let max_depth = depths.iter().copied().max().unwrap_or(0);
    let mut layers = vec![Vec::new(); max_depth + 1];
    for (node, depth) in depths.iter().copied().enumerate() {
        layers[depth].push(node);
    }
    let node_width = |id: &str| {
        if matches!(id, START_STATE | END_STATE) {
            1
        } else {
            let width = declared
                .get(id)
                .map_or_else(|| display_width(id), |label| display_width(&label.text))
                .max(1)
                + 4;
            width + usize::from(width.is_multiple_of(2))
        }
    };
    let layer_widths = layers
        .iter()
        .map(|layer| {
            layer
                .iter()
                .map(|node| node_width(ids[*node]))
                .sum::<usize>()
                + 4 * layer.len().saturating_sub(1)
        })
        .collect::<Vec<_>>();
    let graph_width = layer_widths.iter().copied().max().unwrap_or(0);
    if graph_width == 0 || graph_width > width {
        return None;
    }

    #[derive(Clone, Copy)]
    struct Placed {
        row: usize,
        col: usize,
        width: usize,
        height: usize,
    }
    let layer_heights = layers
        .iter()
        .map(|layer| {
            if layer
                .iter()
                .any(|node| !matches!(ids[*node], START_STATE | END_STATE))
            {
                3
            } else {
                1
            }
        })
        .collect::<Vec<_>>();
    let mut row_starts = vec![0usize; layers.len()];
    for layer in 1..layers.len() {
        let edges = transitions
            .iter()
            .filter(|transition| {
                depths[index[endpoint_key(&transition.from.text, true)]] == layer - 1
            })
            .count();
        row_starts[layer] = row_starts[layer - 1] + layer_heights[layer - 1] + edges * 2 + 2;
    }

    let mut canvas = canvas::MermaidCanvas {
        rows: Vec::new(),
        labels: Vec::new(),
    };
    let mut placements = HashMap::new();
    for (layer, nodes) in layers.iter().enumerate() {
        let mut col = (graph_width - layer_widths[layer]) / 2;
        for node in nodes {
            let id = ids[*node];
            if matches!(id, START_STATE | END_STATE) {
                canvas.put(
                    col,
                    row_starts[layer],
                    if id == END_STATE { '◎' } else { '●' },
                );
                placements.insert(
                    *node,
                    Placed {
                        row: row_starts[layer],
                        col,
                        width: 1,
                        height: 1,
                    },
                );
                col += 5;
                continue;
            }
            let label = declared.get(id);
            let text = label.map_or(id, |label| label.text.as_str());
            let box_width = display_width(text).max(1) + 4;
            let box_width = box_width + usize::from(box_width.is_multiple_of(2));
            canvas.blit(
                row_starts[layer],
                col,
                &format!(
                    "┌{}┐\n│ {} │\n└{}┘",
                    "─".repeat(box_width - 2),
                    text,
                    "─".repeat(box_width - 2)
                ),
            );
            if let Some(label) = label {
                canvas.labels.push(canvas::MermaidCanvasLabel {
                    row: row_starts[layer] + 1,
                    col: col + 2,
                    text: label.text.clone(),
                    source: label.span,
                });
            } else {
                let source = transitions.iter().find_map(|transition| {
                    [&transition.from, &transition.to]
                        .into_iter()
                        .find(|endpoint| endpoint.text == id)
                })?;
                canvas.labels.push(canvas::MermaidCanvasLabel {
                    row: row_starts[layer] + 1,
                    col: col + 2,
                    text: id.to_string(),
                    source: source.span,
                });
            }
            placements.insert(
                *node,
                Placed {
                    row: row_starts[layer],
                    col,
                    width: box_width,
                    height: 3,
                },
            );
            col += box_width + 4;
        }
    }

    let mut routes = HashMap::<(usize, usize), u8>::new();
    let mut layer_offsets = HashMap::<usize, usize>::new();
    for transition in &transitions {
        let from_index = index[endpoint_key(&transition.from.text, true)];
        let to_index = index[endpoint_key(&transition.to.text, false)];
        let from = placements[&from_index];
        let to = placements[&to_index];
        let layer = depths[from_index];
        let offset = layer_offsets.entry(layer).or_default();
        let label_row = from.row + from.height + *offset * 2;
        let route_row = label_row + 1;
        *offset += 1;
        let from_col = from.col + from.width / 2;
        let to_col = to.col + to.width / 2;
        connect_route(
            &mut routes,
            (from_col, from.row + from.height),
            (from_col, route_row),
        );
        connect_route(&mut routes, (from_col, route_row), (to_col, route_row));
        connect_route(&mut routes, (to_col, route_row), (to_col, to.row - 1));
        canvas.put(to_col, to.row.saturating_sub(1), '▼');
        if let Some(label) = &transition.label {
            let label_width = display_width(&label.text);
            canvas.labels.push(canvas::MermaidCanvasLabel {
                row: label_row,
                col: (graph_width - label_width) / 2,
                text: label.text.clone(),
                source: label.span,
            });
        }
    }
    for (&(col, row), &mask) in &routes {
        if !matches!(
            canvas.rows.get(row).and_then(|line| line.get(col)),
            Some(canvas::MermaidCell::Char('▼'))
        ) {
            canvas.put(col, row, route_glyph(mask));
        }
    }
    Some(canvas)
}

const ROUTE_UP: u8 = 1;
const ROUTE_RIGHT: u8 = 2;
const ROUTE_DOWN: u8 = 4;
const ROUTE_LEFT: u8 = 8;

fn connect_route(
    routes: &mut HashMap<(usize, usize), u8>,
    from: (usize, usize),
    to: (usize, usize),
) {
    if from == to {
        return;
    }
    if from.1 == to.1 {
        let (lo, hi) = if from.0 < to.0 {
            (from.0, to.0)
        } else {
            (to.0, from.0)
        };
        for col in lo..hi {
            *routes.entry((col, from.1)).or_default() |= ROUTE_RIGHT;
            *routes.entry((col + 1, from.1)).or_default() |= ROUTE_LEFT;
        }
    } else {
        let (lo, hi) = if from.1 < to.1 {
            (from.1, to.1)
        } else {
            (to.1, from.1)
        };
        for row in lo..hi {
            *routes.entry((from.0, row)).or_default() |= ROUTE_DOWN;
            *routes.entry((from.0, row + 1)).or_default() |= ROUTE_UP;
        }
    }
}

fn route_glyph(mask: u8) -> char {
    match mask {
        1 | 4 | 5 => '│',
        2 | 8 | 10 => '─',
        6 => '┌',
        12 => '┐',
        3 => '└',
        9 => '┘',
        7 => '├',
        13 => '┤',
        14 => '┬',
        11 => '┴',
        15 => '┼',
        _ => '┼',
    }
}

fn render_boxed_transitions(
    diagram: &ir::Diagram,
    width: usize,
) -> Option<Vec<Vec<MermaidRenderSpan>>> {
    let labels = diagram
        .items
        .iter()
        .filter_map(|item| match item {
            ir::Item::State(state) => Some((state.id.as_str(), &state.label)),
            _ => None,
        })
        .collect::<HashMap<_, _>>();
    let mut lines = Vec::new();
    for item in &diagram.items {
        match item {
            ir::Item::State(state) if state.composite => {
                lines.push(vec![
                    MermaidRenderSpan::decoration("  ".repeat(state.depth)),
                    MermaidRenderSpan::decoration("┌─ "),
                    span(&state.label),
                    MermaidRenderSpan::decoration(" { ─┐"),
                ]);
            }
            ir::Item::State(_) => {}
            ir::Item::Close(depth) => lines.push(vec![
                MermaidRenderSpan::decoration("  ".repeat(*depth)),
                MermaidRenderSpan::decoration("└────┘"),
            ]),
            ir::Item::Transition(transition) => {
                let indent = "  ".repeat(transition.depth);
                let from = if transition.from.text == "[*]" {
                    None
                } else {
                    Some(
                        labels
                            .get(transition.from.text.as_str())
                            .copied()
                            .unwrap_or(&transition.from),
                    )
                };
                let to = if transition.to.text == "[*]" {
                    None
                } else {
                    Some(
                        labels
                            .get(transition.to.text.as_str())
                            .copied()
                            .unwrap_or(&transition.to),
                    )
                };
                let mut line = vec![MermaidRenderSpan::decoration(indent)];
                if let Some(from) = from {
                    line.extend([
                        MermaidRenderSpan::decoration("[ "),
                        span(from),
                        MermaidRenderSpan::decoration(" ]"),
                    ]);
                } else {
                    line.push(MermaidRenderSpan::decoration("●"));
                }
                line.push(MermaidRenderSpan::decoration(" ──→ "));
                if let Some(to) = to {
                    line.extend([
                        MermaidRenderSpan::decoration("[ "),
                        span(to),
                        MermaidRenderSpan::decoration(" ]"),
                    ]);
                } else {
                    line.push(MermaidRenderSpan::decoration("◎"));
                }
                if let Some(label) = &transition.label {
                    line.extend([MermaidRenderSpan::decoration("  "), span(label)]);
                }
                lines.push(line);
            }
        }
    }
    (render_line_count_within_limits(lines.len())
        && lines.iter().all(|line| {
            line.iter()
                .map(|span| display_width(&span.text))
                .sum::<usize>()
                <= width
        }))
    .then_some(lines)
}

fn render_linear(diagram: &ir::Diagram, width: usize) -> Option<Vec<Vec<MermaidRenderSpan>>> {
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
        && lines.iter().all(|line| {
            line.iter()
                .map(|span| display_width(&span.text))
                .sum::<usize>()
                <= width
        }))
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
    let mut declared = std::collections::HashSet::<String>::new();
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
            if stack.len() >= MAX_DEPTH || !declared.insert(declaration.id.to_string()) {
                return None;
            }
            let label_at = base + find_char(trimmed, declaration.label)?;
            states.insert(declaration.id.to_string(), declaration.label.to_string());
            items.push(ir::Item::State(ir::State {
                id: declaration.id.to_string(),
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
        if id.is_empty() {
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
    if rest.is_empty() {
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
    value == "[*]" || (!value.is_empty() && !value.contains(['{', '}']))
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
