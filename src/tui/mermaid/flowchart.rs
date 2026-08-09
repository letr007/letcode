//! Flowchart-specific layout, routing, and rendering.

use std::collections::HashMap;

use crate::tui::measure::display_width;

use super::{MermaidRenderSpan, MermaidSourceSpan, canvas, flowchart_ir as ir, routing};

const MERMAID_NODE_GAP: usize = 4;
const MERMAID_LAYER_HEIGHT: usize = 3;
const MERMAID_EDGE_HEIGHT: usize = 3;
const MERMAID_ROUTE_UP: u8 = routing::ROUTE_UP;
const MERMAID_ROUTE_DOWN: u8 = routing::ROUTE_DOWN;
type MermaidRouteGrid = routing::RouteGrid;
type PlacementMap<'a> = HashMap<&'a str, MermaidPlaced>;

#[derive(Debug, Clone, Copy)]
struct MermaidPlaced {
    row: usize,
    col: usize,
    width: usize,
    height: usize,
}

pub(crate) fn render(source: &str, width: usize) -> Option<Vec<Vec<MermaidRenderSpan>>> {
    let graph = super::flowchart_parser::parse(source)?;
    let canvas = match graph.direction {
        ir::MermaidDirection::Td | ir::MermaidDirection::Bu => layout(&graph, width),
        ir::MermaidDirection::Lr | ir::MermaidDirection::Rl => layout_horizontal(&graph, width),
    };
    if let Some(canvas) = canvas {
        return Some(canvas.render());
    }
    render_linear(&graph, width)
}

fn render_linear(graph: &ir::MermaidGraph, width: usize) -> Option<Vec<Vec<MermaidRenderSpan>>> {
    let mut lines = Vec::new();
    let mut rendered_nodes = std::collections::HashSet::new();
    for edge in &graph.edges {
        let from = graph.nodes.get(&edge.from)?;
        let to = graph.nodes.get(&edge.to)?;
        let line = match edge.style {
            ir::MermaidEdgeStyle::Solid => "─",
            ir::MermaidEdgeStyle::Dashed => "╌",
            ir::MermaidEdgeStyle::Thick => "═",
        };
        let mut spans = vec![MermaidRenderSpan::decoration(match graph.direction {
            ir::MermaidDirection::Td => "↓ ",
            ir::MermaidDirection::Bu => "↑ ",
            ir::MermaidDirection::Lr | ir::MermaidDirection::Rl => "",
        })];
        spans.push(source_span(&from.label, from.start, from.end));
        if let Some(label) = &edge.label {
            spans.push(MermaidRenderSpan::decoration(format!(" {line}{line}")));
            spans.push(source_span(&label.text, label.start, label.end));
            spans.push(MermaidRenderSpan::decoration(format!(
                " {line}{line}{} ",
                if edge.arrow { '▶' } else { ' ' }
            )));
        } else {
            spans.push(MermaidRenderSpan::decoration(format!(
                " {line}{line}{} ",
                if edge.arrow { '▶' } else { ' ' }
            )));
        }
        spans.push(source_span(&to.label, to.start, to.end));
        lines.push(spans);
        rendered_nodes.insert(edge.from.as_str());
        rendered_nodes.insert(edge.to.as_str());
    }
    let mut isolated = graph
        .nodes
        .iter()
        .filter(|(id, _)| !rendered_nodes.contains(id.as_str()))
        .collect::<Vec<_>>();
    isolated.sort_by_key(|(_, node)| node.start);
    for (_, node) in isolated {
        lines.push(vec![source_span(&node.label, node.start, node.end)]);
    }
    fits(&lines, width).then_some(lines)
}

fn source_span(text: &str, start: usize, end: usize) -> MermaidRenderSpan {
    MermaidRenderSpan::source(text.to_string(), MermaidSourceSpan::new(start, end), false)
}

fn fits(lines: &[Vec<MermaidRenderSpan>], width: usize) -> bool {
    lines.iter().all(|line| {
        line.iter()
            .map(|span| display_width(&span.text))
            .sum::<usize>()
            <= width
    })
}

fn layout(graph: &ir::MermaidGraph, width: usize) -> Option<canvas::MermaidCanvas> {
    let layers = mermaid_layers(graph)?;
    if layers.len() > 24
        || graph.nodes.len() > 48
        || graph.edges.len() as f64 / graph.nodes.len().max(1) as f64 > 2.5
    {
        return None;
    }
    for layer in &layers {
        let required = layer
            .iter()
            .map(|id| mermaid_node_width(&graph.nodes[id]))
            .sum::<usize>()
            + MERMAID_NODE_GAP * layer.len().saturating_sub(1);
        if required > width {
            return None;
        }
    }
    if mermaid_crossings(graph, &layers) > 8 {
        return None;
    }
    Some(render_mermaid_canvas(graph, &layers))
}

fn layout_horizontal(graph: &ir::MermaidGraph, width: usize) -> Option<canvas::MermaidCanvas> {
    let layers = mermaid_layers(graph)?;
    if layers.len() > 16 || graph.nodes.len() > 32 || mermaid_crossings(graph, &layers) > 8 {
        return None;
    }
    let layer_of = layers
        .iter()
        .enumerate()
        .flat_map(|(layer, nodes)| nodes.iter().map(move |id| (id.as_str(), layer)))
        .collect::<HashMap<_, _>>();
    if graph.edges.iter().any(|edge| {
        layer_of
            .get(edge.from.as_str())
            .zip(layer_of.get(edge.to.as_str()))
            .is_none_or(|(from, to)| *to != *from + 1)
    }) {
        return None;
    }

    let column_widths = layers
        .iter()
        .map(|layer| {
            layer
                .iter()
                .map(|id| mermaid_node_width(&graph.nodes[id]))
                .max()
                .unwrap_or(0)
        })
        .collect::<Vec<_>>();
    let gaps = layers
        .windows(2)
        .enumerate()
        .map(|(layer, _)| {
            let label = graph
                .edges
                .iter()
                .filter(|edge| layer_of[edge.from.as_str()] == layer)
                .filter_map(|edge| edge.label.as_ref())
                .map(|label| display_width(&label.text))
                .max()
                .unwrap_or(0);
            (label + 4).max(8)
        })
        .collect::<Vec<_>>();
    let graph_width = column_widths.iter().sum::<usize>() + gaps.iter().sum::<usize>();
    if graph_width == 0 || graph_width > width {
        return None;
    }
    let graph_height = layers
        .iter()
        .map(|layer| layer.len() * 5usize - 2)
        .max()
        .unwrap_or(0);
    let mut starts = Vec::with_capacity(layers.len());
    let mut col = 0;
    for layer in 0..layers.len() {
        starts.push(col);
        col += column_widths[layer];
        if let Some(gap) = gaps.get(layer) {
            col += gap;
        }
    }
    if matches!(graph.direction, ir::MermaidDirection::Rl) {
        for (layer, start) in starts.iter_mut().enumerate() {
            *start = graph_width - *start - column_widths[layer];
        }
    }

    let mut canvas = canvas::MermaidCanvas {
        rows: Vec::new(),
        labels: Vec::new(),
    };
    let mut placements = HashMap::new();
    for (layer, nodes) in layers.iter().enumerate() {
        let layer_height = nodes.len() * 5 - 2;
        let mut row = (graph_height - layer_height) / 2;
        for id in nodes {
            let node = &graph.nodes[id];
            let node_width = mermaid_node_width(node);
            let node_col = starts[layer] + (column_widths[layer] - node_width) / 2;
            for (offset, line) in render_mermaid_node_shape(node).iter().enumerate() {
                canvas.blit(row + offset, node_col, line);
            }
            canvas.labels.push(canvas::MermaidCanvasLabel {
                row: row + 1,
                col: node_col + (node_width - display_width(&node.label)) / 2,
                text: node.label.clone(),
                source: MermaidSourceSpan::new(node.start, node.end),
            });
            placements.insert(
                id.as_str(),
                MermaidPlaced {
                    row,
                    col: node_col,
                    width: node_width,
                    height: 3,
                },
            );
            row += 5;
        }
    }
    route_horizontal_edges(&mut canvas, graph, &placements, &layer_of)?;
    Some(canvas)
}

fn route_horizontal_edges(
    canvas: &mut canvas::MermaidCanvas,
    graph: &ir::MermaidGraph,
    placements: &PlacementMap<'_>,
    layer_of: &HashMap<&str, usize>,
) -> Option<()> {
    let left_to_right = matches!(graph.direction, ir::MermaidDirection::Lr);
    let mut routes = MermaidRouteGrid::new();
    let mut tracks = HashMap::<usize, routing::TrackAllocator>::new();
    let mut arrows = Vec::new();
    for edge in &graph.edges {
        let Some(from) = placements.get(edge.from.as_str()) else {
            continue;
        };
        let Some(to) = placements.get(edge.to.as_str()) else {
            continue;
        };
        let from_row = from.row + 1;
        let to_row = to.row + 1;
        let source_exit = if left_to_right {
            (from.col + from.width, from_row)
        } else {
            (from.col.saturating_sub(1), from_row)
        };
        let target_entry = if left_to_right {
            (to.col.saturating_sub(1), to_row)
        } else {
            (to.col + to.width, to_row)
        };
        let lo = source_exit.0.min(target_entry.0);
        let hi = source_exit.0.max(target_entry.0);
        let preferred = (source_exit.0 + target_entry.0) / 2;
        let channel = tracks
            .entry(layer_of[edge.from.as_str()])
            .or_default()
            .allocate(preferred, lo.saturating_add(1), hi.saturating_sub(1))?;
        routes.connect(source_exit, (channel, from_row));
        routes.connect((channel, from_row), (channel, to_row));
        routes.connect((channel, to_row), target_entry);
        if edge.arrow {
            arrows.push((target_entry, if left_to_right { '▶' } else { '◀' }));
        }
        if let Some(label) = &edge.label {
            let label_width = display_width(&label.text).max(1);
            let (row, col) = if from_row != to_row {
                let row = (from_row + to_row) / 2;
                let left = channel.saturating_sub(label_width + 1);
                if mermaid_route_span_is_free(&routes, row, left, label_width) {
                    (row, left)
                } else {
                    (row, channel + 1)
                }
            } else {
                let col = lo + 1 + (hi - lo - 1).saturating_sub(label_width) / 2;
                (from_row, col)
            };
            canvas.labels.push(canvas::MermaidCanvasLabel {
                row,
                col,
                text: label.text.clone(),
                source: MermaidSourceSpan::new(label.start, label.end),
            });
        }
    }
    for (&(col, row), &mask) in routes.iter() {
        canvas.put(col, row, mermaid_route_glyph(mask));
    }
    for ((col, row), arrow) in arrows {
        canvas.put(col, row, arrow);
    }
    Some(())
}

fn render_mermaid_canvas(
    graph: &ir::MermaidGraph,
    layers: &[Vec<String>],
) -> canvas::MermaidCanvas {
    let layer_widths = layers
        .iter()
        .map(|layer| {
            layer
                .iter()
                .map(|id| mermaid_node_width(&graph.nodes[id]))
                .sum::<usize>()
                + MERMAID_NODE_GAP * layer.len().saturating_sub(1)
        })
        .collect::<Vec<_>>();
    let graph_width = layer_widths.iter().copied().max().unwrap_or(0);
    let mut canvas = canvas::MermaidCanvas {
        rows: Vec::new(),
        labels: Vec::new(),
    };
    let mut placements = HashMap::new();
    for (li, layer) in layers.iter().enumerate() {
        let row = li * (MERMAID_LAYER_HEIGHT + MERMAID_EDGE_HEIGHT);
        let mut col = (graph_width / 2).saturating_sub(layer_widths[li] / 2);
        for id in layer {
            let node = &graph.nodes[id];
            let width = mermaid_node_width(node);
            placements.insert(
                id.as_str(),
                MermaidPlaced {
                    row,
                    col,
                    width,
                    height: 3,
                },
            );
            for (index, line) in render_mermaid_node_shape(node).iter().enumerate() {
                canvas.blit(row + index, col, line);
            }
            let label_col = col + (width - display_width(&node.label)) / 2;
            canvas.labels.push(canvas::MermaidCanvasLabel {
                row: row + 1,
                col: label_col,
                text: node.label.clone(),
                source: MermaidSourceSpan::new(node.start, node.end),
            });
            col += width + MERMAID_NODE_GAP;
        }
    }
    route_mermaid_edges(&mut canvas, graph, &placements);
    canvas
}

fn render_mermaid_node_shape(node: &ir::MermaidNode) -> Vec<String> {
    let inner = display_width(&node.label).max(1) + 2;
    let _ = node.shape;
    vec![
        format!("╭{}╮", "─".repeat(inner)),
        format!("│ {} │", node.label),
        format!("╰{}╯", "─".repeat(inner)),
    ]
}

fn mermaid_node_width(node: &ir::MermaidNode) -> usize {
    display_width(&node.label).max(1) + 4
}

fn connect_mermaid_route(routes: &mut MermaidRouteGrid, from: (usize, usize), to: (usize, usize)) {
    routes.connect(from, to);
}

fn mermaid_route_glyph(mask: u8) -> char {
    routing::route_glyph(mask)
}

fn route_mermaid_edges(
    canvas: &mut canvas::MermaidCanvas,
    graph: &ir::MermaidGraph,
    placements: &PlacementMap<'_>,
) {
    let mut outgoing: HashMap<&str, Vec<&ir::MermaidEdge>> = HashMap::new();
    let mut incoming: HashMap<&str, Vec<&ir::MermaidEdge>> = HashMap::new();
    for edge in &graph.edges {
        outgoing.entry(edge.from.as_str()).or_default().push(edge);
        incoming.entry(edge.to.as_str()).or_default().push(edge);
    }
    let mut fork_rows = HashMap::new();
    for (from_id, edges) in &outgoing {
        if edges.len() < 2 {
            continue;
        }
        let Some(from) = placements.get(from_id) else {
            continue;
        };
        let from_row = from.row + from.height - 1;
        let Some(min_to_row) = edges
            .iter()
            .filter_map(|edge| placements.get(edge.to.as_str()).map(|to| to.row))
            .min()
        else {
            continue;
        };
        if let Some(row) = pick_beam_row(from_row, min_to_row, placements) {
            fork_rows.insert(*from_id, row);
        }
    }
    let mut merge_rows = HashMap::new();
    for (to_id, edges) in &incoming {
        if edges.len() < 2 {
            continue;
        }
        let Some(to) = placements.get(to_id) else {
            continue;
        };
        let Some(max_from_row) = edges
            .iter()
            .filter_map(|edge| {
                placements
                    .get(edge.from.as_str())
                    .map(|from| from.row + from.height - 1)
            })
            .max()
        else {
            continue;
        };
        if let Some(row) = pick_beam_row(max_from_row, to.row, placements) {
            merge_rows.insert(*to_id, row);
        }
    }
    let mut routes = MermaidRouteGrid::new();
    let mut arrows = Vec::new();
    let mut labels = Vec::new();
    for edge in &graph.edges {
        let Some(from) = placements.get(edge.from.as_str()) else {
            continue;
        };
        let Some(to) = placements.get(edge.to.as_str()) else {
            continue;
        };
        let from_row = from.row + from.height - 1;
        let to_row = to.row;
        if to_row <= from_row + 1 {
            continue;
        }
        let from_col = from.col + from.width / 2;
        let to_col = to.col + to.width / 2;
        let departure = fork_rows
            .get(edge.from.as_str())
            .copied()
            .unwrap_or(from_row + 1);
        let arrival = merge_rows
            .get(edge.to.as_str())
            .copied()
            .unwrap_or(to_row.saturating_sub(2));
        if departure > arrival {
            continue;
        }
        let forked = outgoing
            .get(edge.from.as_str())
            .is_some_and(|edges| edges.len() > 1);
        let preferred_col = if forked {
            to_col
        } else if incoming
            .get(edge.to.as_str())
            .is_some_and(|edges| edges.len() > 1)
        {
            from_col
        } else {
            (from_col + to_col) / 2
        };
        let channel = avoid_column(preferred_col, departure, arrival, placements);
        let source_exit = (from_col, from_row + 1);
        let target_entry = (to_col, to_row - 1);
        *routes.entry(source_exit).or_default() |= MERMAID_ROUTE_UP;
        *routes.entry(target_entry).or_default() |= MERMAID_ROUTE_DOWN;
        connect_mermaid_route(&mut routes, source_exit, (from_col, departure));
        connect_mermaid_route(&mut routes, (from_col, departure), (channel, departure));
        connect_mermaid_route(&mut routes, (channel, departure), (channel, arrival));
        connect_mermaid_route(&mut routes, (channel, arrival), (to_col, arrival));
        connect_mermaid_route(&mut routes, (to_col, arrival), target_entry);
        if edge.arrow {
            arrows.push(target_entry);
        }
        labels.push((edge, departure, arrival, from_col, channel, to_col, forked));
    }
    let mut route_cells = routes.iter().collect::<Vec<_>>();
    route_cells.sort_by_key(|((col, row), _)| (*row, *col));
    for (&(col, row), &mask) in route_cells {
        canvas.put(col, row, mermaid_route_glyph(mask));
    }
    for (col, row) in arrows {
        canvas.put(col, row, 'v');
    }
    for (edge, departure, arrival, from_col, channel, to_col, forked) in labels {
        place_edge_label(
            canvas, &routes, edge, departure, arrival, from_col, channel, to_col, forked,
        );
    }
}

fn mermaid_row_in_box(row: usize, placements: &PlacementMap<'_>) -> bool {
    placements
        .values()
        .any(|p| row >= p.row && row < p.row + p.height)
}
fn pick_beam_row(from_row: usize, to_row: usize, placements: &PlacementMap<'_>) -> Option<usize> {
    if from_row + 1 >= to_row {
        return None;
    }
    let candidates = (from_row + 1..to_row)
        .filter(|row| !mermaid_row_in_box(*row, placements))
        .collect::<Vec<_>>();
    if candidates.is_empty() {
        return None;
    }
    let middle = (from_row + to_row) / 2;
    candidates
        .into_iter()
        .min_by_key(|row| row.abs_diff(middle))
}
fn avoid_column(
    column: usize,
    first_row: usize,
    last_row: usize,
    placements: &PlacementMap<'_>,
) -> usize {
    if first_row > last_row {
        return column;
    }
    let blocked = |col: usize| {
        placements.values().any(|p| {
            col >= p.col
                && col < p.col + p.width
                && first_row < p.row + p.height
                && last_row >= p.row
        })
    };
    if !blocked(column) {
        return column;
    }
    for distance in 1..64 {
        if column >= distance && !blocked(column - distance) {
            return column - distance;
        }
        if !blocked(column + distance) {
            return column + distance;
        }
    }
    column
}
fn centered_mermaid_label_col(a: usize, b: usize, width: usize) -> Option<usize> {
    let lo = a.min(b);
    let hi = a.max(b);
    let Some(interior_start) = lo.checked_add(1) else {
        return None;
    };
    let interior = hi.saturating_sub(interior_start);
    if width > interior {
        return None;
    }
    Some(interior_start + (interior - width) / 2)
}
fn mermaid_route_span_is_free(
    routes: &MermaidRouteGrid,
    row: usize,
    col: usize,
    width: usize,
) -> bool {
    (col..col + width).all(|c| !routes.contains_key(&(c, row)))
}
#[allow(clippy::too_many_arguments)]
fn place_edge_label(
    canvas: &mut canvas::MermaidCanvas,
    routes: &MermaidRouteGrid,
    edge: &ir::MermaidEdge,
    departure: usize,
    arrival: usize,
    from_col: usize,
    channel: usize,
    to_col: usize,
    forked: bool,
) {
    let Some(label) = &edge.label else {
        return;
    };
    let width = display_width(&label.text).max(1);
    let mut position = if forked {
        centered_mermaid_label_col(from_col, channel, width).map(|col| (departure, col))
    } else {
        None
    };
    if position.is_none() {
        position = centered_mermaid_label_col(channel, to_col, width).map(|col| (arrival, col));
    }
    if position.is_none() {
        position = centered_mermaid_label_col(from_col, channel, width).map(|col| (departure, col));
    }
    if position.is_none() {
        let row = (departure + arrival) / 2;
        for gap in 2..64 {
            if let Some(col) = channel.checked_sub(width + gap - 1)
                && mermaid_route_span_is_free(routes, row, col, width)
            {
                position = Some((row, col));
                break;
            }
            let col = channel + gap;
            if mermaid_route_span_is_free(routes, row, col, width) {
                position = Some((row, col));
                break;
            }
        }
    }
    let (row, col) = position.unwrap_or((departure, channel + 2));
    canvas.blit(row, col, &label.text);
    canvas.labels.push(canvas::MermaidCanvasLabel {
        row,
        col,
        text: label.text.clone(),
        source: MermaidSourceSpan::new(label.start, label.end),
    });
}

fn mermaid_layers(graph: &ir::MermaidGraph) -> Option<Vec<Vec<String>>> {
    let mut indeg: HashMap<String, usize> = graph.nodes.keys().map(|id| (id.clone(), 0)).collect();
    for edge in &graph.edges {
        *indeg.get_mut(&edge.to)? += 1;
    }
    let mut layers = Vec::new();
    let mut remaining = graph.nodes.len();
    while remaining > 0 {
        let mut layer = indeg
            .iter()
            .filter(|(_, degree)| **degree == 0)
            .map(|(id, _)| id.clone())
            .collect::<Vec<_>>();
        layer.sort();
        if layer.is_empty() {
            return None;
        }
        for id in &layer {
            for edge in graph.edges.iter().filter(|edge| &edge.from == id) {
                if let Some(degree) = indeg.get_mut(&edge.to) {
                    *degree = degree.saturating_sub(1);
                }
            }
            indeg.remove(id);
        }
        remaining -= layer.len();
        layers.push(layer);
    }
    Some(layers)
}
fn mermaid_crossings(graph: &ir::MermaidGraph, layers: &[Vec<String>]) -> usize {
    let mut layer_of = HashMap::new();
    let mut col_of = HashMap::new();
    for (li, layer) in layers.iter().enumerate() {
        for (ci, id) in layer.iter().enumerate() {
            layer_of.insert(id.as_str(), li);
            col_of.insert(id.as_str(), ci);
        }
    }
    let mut total = 0usize;
    for _ in layers.windows(2) {
        let edges = graph
            .edges
            .iter()
            .filter_map(|edge| {
                let from = *layer_of.get(edge.from.as_str())?;
                let to = *layer_of.get(edge.to.as_str())?;
                (from + 1 == to).then_some((col_of[edge.from.as_str()], col_of[edge.to.as_str()]))
            })
            .collect::<Vec<_>>();
        for (index, edge) in edges.iter().enumerate() {
            for other in &edges[index + 1..] {
                if (edge.0 < other.0 && edge.1 > other.1) || (edge.0 > other.0 && edge.1 < other.1)
                {
                    total += 1;
                }
            }
        }
    }
    total
}
