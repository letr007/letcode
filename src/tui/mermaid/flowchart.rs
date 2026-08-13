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
        let mut lines = canvas.render();
        apply_atomic_labels(&graph, &mut lines);
        if fits(&lines, width) {
            return Some(lines);
        }
    }
    render_linear(&graph, width)
}

fn apply_atomic_labels(graph: &ir::MermaidGraph, lines: &mut [Vec<MermaidRenderSpan>]) {
    for span in lines.iter_mut().flatten() {
        let Some(source) = span.source else {
            continue;
        };
        span.atomic = graph
            .nodes
            .values()
            .any(|node| node.start == source.start && node.end == source.end && node.atomic)
            || graph
                .edges
                .iter()
                .filter_map(|edge| edge.label.as_ref())
                .any(|label| {
                    label.start == source.start && label.end == source.end && label.atomic
                });
    }
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
        spans.push(source_span(&from.label, from.start, from.end, from.atomic));
        if let Some(label) = &edge.label {
            spans.push(MermaidRenderSpan::decoration(format!(" {line}{line}")));
            spans.push(source_span(
                &label.text,
                label.start,
                label.end,
                label.atomic,
            ));
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
        spans.push(source_span(&to.label, to.start, to.end, to.atomic));
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
        lines.push(vec![source_span(
            &node.label,
            node.start,
            node.end,
            node.atomic,
        )]);
    }
    fits(&lines, width).then_some(lines)
}

fn source_span(text: &str, start: usize, end: usize, atomic: bool) -> MermaidRenderSpan {
    MermaidRenderSpan::source(text.to_string(), MermaidSourceSpan::new(start, end), atomic)
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
    for (layer, column_width) in column_widths.iter().enumerate().take(layers.len()) {
        starts.push(col);
        col += column_width;
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

#[derive(Debug, Clone, Copy)]
struct HorizontalBeam {
    col: usize,
    first_row: usize,
    last_row: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HorizontalBeamKind {
    Source,
    Target,
}

#[derive(Debug, Clone, Copy)]
struct HorizontalBeamCandidate<'a> {
    kind: HorizontalBeamKind,
    id: &'a str,
    corridor: usize,
    preferred: usize,
    min_col: usize,
    max_col: usize,
    first_row: usize,
    last_row: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum HorizontalRouteOwner<'a> {
    Source(&'a str),
    Target(&'a str),
    Edge(usize),
}

#[derive(Debug, Clone, Copy)]
struct HorizontalRouteSegment<'a> {
    from: (usize, usize),
    to: (usize, usize),
    owner: HorizontalRouteOwner<'a>,
}

#[derive(Debug, Clone, Copy)]
struct HorizontalLabelRoute<'a> {
    edge: &'a ir::MermaidEdge,
    departure: usize,
    arrival: usize,
    source_col: usize,
    channel: usize,
    target_col: usize,
}

fn horizontal_source_exit(
    placement: &MermaidPlaced,
    row: usize,
    left_to_right: bool,
) -> (usize, usize) {
    if left_to_right {
        (placement.col + placement.width, row)
    } else {
        (placement.col.saturating_sub(1), row)
    }
}

fn horizontal_target_entry(
    placement: &MermaidPlaced,
    row: usize,
    left_to_right: bool,
) -> (usize, usize) {
    if left_to_right {
        (placement.col.saturating_sub(1), row)
    } else {
        (placement.col + placement.width, row)
    }
}

fn horizontal_channel_bounds(a: usize, b: usize) -> Option<(usize, usize)> {
    let lo = a.min(b).checked_add(1)?;
    let hi = a.max(b).checked_sub(1)?;
    (lo <= hi).then_some((lo, hi))
}

fn midpoint_toward(a: usize, b: usize, toward: usize) -> usize {
    let lo = a.min(b);
    let hi = a.max(b);
    let distance = hi - lo;
    let mut midpoint = lo + distance / 2;
    if distance % 2 == 1 && toward == hi {
        midpoint += 1;
    }
    midpoint
}

fn push_horizontal_segment<'a>(
    segments: &mut Vec<HorizontalRouteSegment<'a>>,
    from: (usize, usize),
    to: (usize, usize),
    owner: HorizontalRouteOwner<'a>,
) {
    if from != to {
        segments.push(HorizontalRouteSegment { from, to, owner });
    }
}

fn horizontal_routes_do_not_conflict(segments: &[HorizontalRouteSegment<'_>]) -> bool {
    let mut vertices = HashMap::new();
    let mut units = HashMap::new();
    for segment in segments {
        if segment.from.0 != segment.to.0 && segment.from.1 != segment.to.1 {
            return false;
        }
        let mut claim = |from: (usize, usize), to: (usize, usize)| {
            for point in [from, to] {
                if let Some(owner) = vertices.get(&point)
                    && *owner != segment.owner
                {
                    return false;
                }
                vertices.entry(point).or_insert(segment.owner);
            }
            let unit = if from <= to { (from, to) } else { (to, from) };
            if let Some(owner) = units.get(&unit)
                && *owner != segment.owner
            {
                return false;
            }
            units.insert(unit, segment.owner);
            true
        };
        if segment.from.1 == segment.to.1 {
            let row = segment.from.1;
            let lo = segment.from.0.min(segment.to.0);
            let hi = segment.from.0.max(segment.to.0);
            for col in lo..hi {
                if !claim((col, row), (col + 1, row)) {
                    return false;
                }
            }
        } else {
            let col = segment.from.0;
            let lo = segment.from.1.min(segment.to.1);
            let hi = segment.from.1.max(segment.to.1);
            for row in lo..hi {
                if !claim((col, row), (col, row + 1)) {
                    return false;
                }
            }
        }
    }
    true
}

fn route_horizontal_edges(
    canvas: &mut canvas::MermaidCanvas,
    graph: &ir::MermaidGraph,
    placements: &PlacementMap<'_>,
    layer_of: &HashMap<&str, usize>,
) -> Option<()> {
    let left_to_right = matches!(graph.direction, ir::MermaidDirection::Lr);
    let mut outgoing: HashMap<&str, Vec<&ir::MermaidEdge>> = HashMap::new();
    let mut incoming: HashMap<&str, Vec<&ir::MermaidEdge>> = HashMap::new();
    for edge in &graph.edges {
        outgoing.entry(edge.from.as_str()).or_default().push(edge);
        incoming.entry(edge.to.as_str()).or_default().push(edge);
    }
    if graph.edges.iter().any(|edge| {
        outgoing
            .get(edge.from.as_str())
            .is_some_and(|edges| edges.len() >= 2)
            && incoming
                .get(edge.to.as_str())
                .is_some_and(|edges| edges.len() >= 2)
    }) {
        return None;
    }

    let mut candidates = Vec::new();
    for (&from_id, edges) in &outgoing {
        if edges.len() < 2 || edges.iter().any(|edge| edge.label.is_some()) {
            continue;
        }
        let corridor = *layer_of.get(from_id)?;
        let from = placements.get(from_id)?;
        let from_row = from.row + 1;
        let source_exit = horizontal_source_exit(from, from_row, left_to_right);
        let mut min_col = 0usize;
        let mut max_col = usize::MAX;
        let mut first_row = from_row;
        let mut last_row = from_row;
        let mut target_side = if left_to_right { usize::MAX } else { 0 };
        for edge in edges {
            if *layer_of.get(edge.from.as_str())? != corridor {
                return None;
            }
            let to = placements.get(edge.to.as_str())?;
            let to_row = to.row + 1;
            let target_entry = horizontal_target_entry(to, to_row, left_to_right);
            let (edge_min, edge_max) = horizontal_channel_bounds(source_exit.0, target_entry.0)?;
            min_col = min_col.max(edge_min);
            max_col = max_col.min(edge_max);
            first_row = first_row.min(to_row);
            last_row = last_row.max(to_row);
            if left_to_right {
                target_side = target_side.min(to.col);
            } else {
                target_side = target_side.max(to.col + to.width - 1);
            }
        }
        if min_col > max_col {
            return None;
        }
        candidates.push(HorizontalBeamCandidate {
            kind: HorizontalBeamKind::Source,
            id: from_id,
            corridor,
            preferred: midpoint_toward(source_exit.0, target_side, target_side),
            min_col,
            max_col,
            first_row,
            last_row,
        });
    }
    for (&to_id, edges) in &incoming {
        if edges.len() < 2 || edges.iter().any(|edge| edge.label.is_some()) {
            continue;
        }
        let corridor = *layer_of.get(edges[0].from.as_str())?;
        if edges
            .iter()
            .any(|edge| layer_of.get(edge.from.as_str()).copied() != Some(corridor))
        {
            return None;
        }
        let to = placements.get(to_id)?;
        let to_row = to.row + 1;
        let target_entry = horizontal_target_entry(to, to_row, left_to_right);
        let mut min_col = 0usize;
        let mut max_col = usize::MAX;
        let mut first_row = to_row;
        let mut last_row = to_row;
        let mut source_side = if left_to_right { 0 } else { usize::MAX };
        for edge in edges {
            let from = placements.get(edge.from.as_str())?;
            let from_row = from.row + 1;
            let source_exit = horizontal_source_exit(from, from_row, left_to_right);
            let (edge_min, edge_max) = horizontal_channel_bounds(source_exit.0, target_entry.0)?;
            min_col = min_col.max(edge_min);
            max_col = max_col.min(edge_max);
            first_row = first_row.min(from_row);
            last_row = last_row.max(from_row);
            if left_to_right {
                source_side = source_side.max(from.col + from.width - 1);
            } else {
                source_side = source_side.min(from.col);
            }
        }
        if min_col > max_col {
            return None;
        }
        candidates.push(HorizontalBeamCandidate {
            kind: HorizontalBeamKind::Target,
            id: to_id,
            corridor,
            preferred: midpoint_toward(source_side, target_entry.0, source_side),
            min_col,
            max_col,
            first_row,
            last_row,
        });
    }
    candidates.sort_by(|a, b| {
        let kind_order = |kind| match (left_to_right, kind) {
            (true, HorizontalBeamKind::Source) | (false, HorizontalBeamKind::Target) => 0,
            _ => 1,
        };
        a.corridor
            .cmp(&b.corridor)
            .then_with(|| kind_order(a.kind).cmp(&kind_order(b.kind)))
            .then_with(|| a.id.cmp(b.id))
    });

    let mut tracks = HashMap::<usize, routing::TrackAllocator>::new();
    let mut source_beams = HashMap::<&str, HorizontalBeam>::new();
    let mut target_beams = HashMap::<&str, HorizontalBeam>::new();
    for candidate in candidates {
        let col = tracks.entry(candidate.corridor).or_default().allocate(
            candidate.preferred,
            candidate.min_col,
            candidate.max_col,
        )?;
        let beam = HorizontalBeam {
            col,
            first_row: candidate.first_row,
            last_row: candidate.last_row,
        };
        match candidate.kind {
            HorizontalBeamKind::Source => {
                source_beams.insert(candidate.id, beam);
            }
            HorizontalBeamKind::Target => {
                target_beams.insert(candidate.id, beam);
            }
        }
    }

    let mut segments = Vec::new();
    let mut beam_segments = Vec::new();
    for (&id, beam) in &source_beams {
        push_horizontal_segment(
            &mut beam_segments,
            (beam.col, beam.first_row),
            (beam.col, beam.last_row),
            HorizontalRouteOwner::Source(id),
        );
    }
    for (&id, beam) in &target_beams {
        push_horizontal_segment(
            &mut beam_segments,
            (beam.col, beam.first_row),
            (beam.col, beam.last_row),
            HorizontalRouteOwner::Target(id),
        );
    }

    let mut arrows = Vec::new();
    let mut label_routes = Vec::new();
    for (edge_index, edge) in graph.edges.iter().enumerate() {
        let from = placements.get(edge.from.as_str())?;
        let to = placements.get(edge.to.as_str())?;
        let from_row = from.row + 1;
        let to_row = to.row + 1;
        let source_exit = horizontal_source_exit(from, from_row, left_to_right);
        let target_entry = horizontal_target_entry(to, to_row, left_to_right);
        let source_beam = source_beams.get(edge.from.as_str());
        let target_beam = target_beams.get(edge.to.as_str());
        let source_grouped = outgoing
            .get(edge.from.as_str())
            .is_some_and(|edges| edges.len() >= 2);
        let target_grouped = incoming
            .get(edge.to.as_str())
            .is_some_and(|edges| edges.len() >= 2);
        let owner = if source_grouped {
            HorizontalRouteOwner::Source(edge.from.as_str())
        } else if target_grouped {
            HorizontalRouteOwner::Target(edge.to.as_str())
        } else {
            HorizontalRouteOwner::Edge(edge_index)
        };
        let channel = match (source_beam, target_beam) {
            (Some(_), Some(_)) => return None,
            (Some(source), None) => {
                push_horizontal_segment(&mut segments, source_exit, (source.col, from_row), owner);
                push_horizontal_segment(&mut segments, (source.col, to_row), target_entry, owner);
                source.col
            }
            (None, Some(target)) => {
                let source_joint = (target.col, from_row);
                let target_joint = (target.col, to_row);
                if source_joint == target_joint {
                    push_horizontal_segment(&mut segments, source_exit, target_entry, owner);
                } else {
                    push_horizontal_segment(&mut segments, source_exit, source_joint, owner);
                    push_horizontal_segment(&mut segments, target_joint, target_entry, owner);
                }
                target.col
            }
            (None, None) => {
                let (min_col, max_col) = horizontal_channel_bounds(source_exit.0, target_entry.0)?;
                let preferred = source_exit.0;
                let channel = tracks
                    .entry(*layer_of.get(edge.from.as_str())?)
                    .or_default()
                    .allocate(preferred, min_col, max_col)?;
                push_horizontal_segment(&mut segments, source_exit, (channel, from_row), owner);
                push_horizontal_segment(
                    &mut segments,
                    (channel, from_row),
                    (channel, to_row),
                    owner,
                );
                push_horizontal_segment(&mut segments, (channel, to_row), target_entry, owner);
                channel
            }
        };
        if edge.arrow {
            arrows.push((target_entry, if left_to_right { '▶' } else { '◀' }));
        }
        if edge.label.is_some() {
            label_routes.push(HorizontalLabelRoute {
                edge,
                departure: from_row,
                arrival: to_row,
                source_col: source_exit.0,
                channel,
                target_col: target_entry.0,
            });
        }
    }
    segments.extend(beam_segments);
    if !horizontal_routes_do_not_conflict(&segments) {
        return None;
    }

    let mut routes = MermaidRouteGrid::new();
    for segment in segments {
        routes.connect(segment.from, segment.to);
    }
    let mut edge_labels = Vec::new();
    for route in label_routes {
        let label = route.edge.label.as_ref()?;
        let label_width = display_width(&label.text).max(1);
        let (row, col) = if route.departure != route.arrival {
            let row = (route.departure + route.arrival) / 2;
            let left = route.channel.saturating_sub(label_width + 1);
            if mermaid_route_span_is_free(&routes, row, left, label_width) {
                (row, left)
            } else {
                (row, route.channel + 1)
            }
        } else {
            let lo = route.source_col.min(route.target_col);
            let hi = route.source_col.max(route.target_col);
            (
                route.departure,
                lo + 1 + (hi - lo - 1).saturating_sub(label_width) / 2,
            )
        };
        edge_labels.push(canvas::MermaidCanvasLabel {
            row,
            col,
            text: label.text.clone(),
            source: MermaidSourceSpan::new(label.start, label.end),
        });
    }
    for (&(col, row), &mask) in routes.iter() {
        canvas.put(col, row, mermaid_route_glyph(mask));
    }
    for ((col, row), arrow) in arrows {
        canvas.put(col, row, arrow);
    }
    canvas.labels.extend(edge_labels);
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
        canvas.put(col, row, '▼');
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
    let interior_start = lo.checked_add(1)?;
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

pub(super) fn mermaid_layers(graph: &ir::MermaidGraph) -> Option<Vec<Vec<String>>> {
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
pub(super) fn mermaid_crossings(graph: &ir::MermaidGraph, layers: &[Vec<String>]) -> usize {
    let mut layer_of = HashMap::new();
    let mut col_of = HashMap::new();
    for (li, layer) in layers.iter().enumerate() {
        for (ci, id) in layer.iter().enumerate() {
            layer_of.insert(id.as_str(), li);
            col_of.insert(id.as_str(), ci);
        }
    }
    let mut total = 0usize;
    for (corridor, _) in layers.windows(2).enumerate() {
        let edges = graph
            .edges
            .iter()
            .filter_map(|edge| {
                let from = *layer_of.get(edge.from.as_str())?;
                let to = *layer_of.get(edge.to.as_str())?;
                (from == corridor && to == corridor + 1)
                    .then_some((col_of[edge.from.as_str()], col_of[edge.to.as_str()]))
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

#[cfg(test)]
mod tests {
    use super::{HorizontalRouteOwner, HorizontalRouteSegment, horizontal_routes_do_not_conflict};

    #[test]
    fn horizontal_route_ownership_allows_same_net_junctions_and_rejects_foreign_contacts() {
        let owner = HorizontalRouteOwner::Source("S");
        let mut segments = vec![
            HorizontalRouteSegment {
                from: (2, 0),
                to: (2, 2),
                owner,
            },
            HorizontalRouteSegment {
                from: (0, 1),
                to: (2, 1),
                owner,
            },
        ];
        assert!(horizontal_routes_do_not_conflict(&segments));

        segments.push(HorizontalRouteSegment {
            from: (2, 1),
            to: (4, 1),
            owner: HorizontalRouteOwner::Edge(0),
        });
        assert!(!horizontal_routes_do_not_conflict(&segments));
    }
}
