use super::class_ir as ir;
use super::{
    MermaidRenderSpan, MermaidSourceSpan, canvas, render_line_count_within_limits,
    source_within_limits,
};
use crate::tui::measure::display_width;
use std::collections::{HashMap, HashSet, VecDeque};

const CONNECTORS: [&str; 14] = [
    "<|--", "--|>", "..|>", "<|..", "*--", "--*", "o--", "--o", "..>", "<..", "-->", "<--", "..",
    "--",
];

pub(super) fn render(source: &str, width: usize) -> Option<Vec<Vec<MermaidRenderSpan>>> {
    let diagram = parse(source)?;
    if let Some(canvas) = layout(&diagram, width) {
        let lines = canvas.render();
        if render_line_count_within_limits(lines.len()) {
            return Some(lines);
        }
    }
    render_stacked(&diagram, width).or_else(|| render_linear(&diagram, width))
}

#[derive(Debug, Clone, Copy)]
struct PlacedClass {
    row: usize,
    col: usize,
    width: usize,
    height: usize,
}

type RouteGrid = HashMap<(usize, usize), u8>;

const ROUTE_UP: u8 = 1;
const ROUTE_RIGHT: u8 = 2;
const ROUTE_DOWN: u8 = 4;
const ROUTE_LEFT: u8 = 8;

fn layout(diagram: &ir::Diagram, width: usize) -> Option<canvas::MermaidCanvas> {
    if diagram.classes.len() > 24 || diagram.relations.len() > 48 {
        return None;
    }
    let layers = class_layers(diagram)?;
    let layer_of = layers
        .iter()
        .enumerate()
        .flat_map(|(layer, classes)| classes.iter().map(move |index| (*index, layer)))
        .collect::<HashMap<_, _>>();
    if diagram.relations.iter().any(|relation| {
        let from = class_index(diagram, &relation.from.text);
        let to = class_index(diagram, &relation.to.text);
        from.zip(to)
            .is_none_or(|(from, to)| layer_of[&to] != layer_of[&from] + 1)
    }) {
        return None;
    }

    let boxes = diagram
        .classes
        .iter()
        .map(class_box_size)
        .collect::<Vec<_>>();
    let layer_widths = layers
        .iter()
        .map(|layer| {
            layer.iter().map(|index| boxes[*index].0).sum::<usize>()
                + 4 * layer.len().saturating_sub(1)
        })
        .collect::<Vec<_>>();
    let graph_width = layer_widths.iter().copied().max().unwrap_or(0).max(
        diagram
            .relations
            .iter()
            .filter_map(|relation| relation.label.as_ref())
            .map(|label| display_width(&label.text) + 2)
            .max()
            .unwrap_or(0),
    );
    if graph_width == 0 || graph_width > width {
        return None;
    }

    let mut canvas = canvas::MermaidCanvas {
        rows: Vec::new(),
        labels: Vec::new(),
    };
    let mut placements = HashMap::new();
    let mut row = 0;
    for (layer_index, layer) in layers.iter().enumerate() {
        let layer_height = layer.iter().map(|index| boxes[*index].1).max().unwrap_or(0);
        let mut col = (graph_width - layer_widths[layer_index]) / 2;
        for index in layer {
            let (box_width, box_height) = boxes[*index];
            let placement = PlacedClass {
                row,
                col,
                width: box_width,
                height: box_height,
            };
            draw_class_box(&mut canvas, &diagram.classes[*index], placement);
            placements.insert(*index, placement);
            col += box_width + 4;
        }
        if layer_index + 1 < layers.len() {
            let edge_count = diagram
                .relations
                .iter()
                .filter(|relation| {
                    class_index(diagram, &relation.from.text)
                        .is_some_and(|index| layer_of[&index] == layer_index)
                })
                .count();
            row += layer_height + edge_count * 2 + 2;
        }
    }
    route_relations(&mut canvas, diagram, &placements, &layer_of, graph_width)?;
    Some(canvas)
}

fn class_layers(diagram: &ir::Diagram) -> Option<Vec<Vec<usize>>> {
    let mut indegree = vec![0usize; diagram.classes.len()];
    let mut outgoing = vec![Vec::new(); diagram.classes.len()];
    for relation in &diagram.relations {
        let from = class_index(diagram, &relation.from.text)?;
        let to = class_index(diagram, &relation.to.text)?;
        outgoing[from].push(to);
        indegree[to] += 1;
    }
    let mut queue = indegree
        .iter()
        .enumerate()
        .filter_map(|(index, degree)| (*degree == 0).then_some(index))
        .collect::<VecDeque<_>>();
    let mut depths = vec![0usize; diagram.classes.len()];
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
    if visited != diagram.classes.len() {
        return None;
    }
    let mut layers = vec![Vec::new(); depths.iter().copied().max().unwrap_or(0) + 1];
    for (index, depth) in depths.into_iter().enumerate() {
        layers[depth].push(index);
    }
    Some(layers)
}

fn class_index(diagram: &ir::Diagram, name: &str) -> Option<usize> {
    diagram
        .classes
        .iter()
        .position(|class| class.name.text == name)
}

fn class_box_size(class: &ir::Class) -> (usize, usize) {
    let inner = std::iter::once(display_width(&class.name.text))
        .chain(
            class
                .members
                .iter()
                .map(|member| display_width(&member.text)),
        )
        .max()
        .unwrap_or(1)
        .max(1)
        + 2;
    let split = member_split(class);
    let separators = usize::from(!class.members.is_empty()) + usize::from(split.is_some());
    let width = inner + 2;
    (
        width + usize::from(width.is_multiple_of(2)),
        3 + class.members.len() + separators,
    )
}

fn member_split(class: &ir::Class) -> Option<usize> {
    let split = class
        .members
        .iter()
        .position(|member| member.text.contains('('))?;
    (split > 0).then_some(split)
}

fn draw_class_box(canvas: &mut canvas::MermaidCanvas, class: &ir::Class, placement: PlacedClass) {
    let inner = placement.width - 2;
    canvas.blit(
        placement.row,
        placement.col,
        &format!("┌{}┐", "─".repeat(inner)),
    );
    draw_box_label(canvas, placement.row + 1, placement, &class.name, true);
    if class.members.is_empty() {
        canvas.blit(
            placement.row + 2,
            placement.col,
            &format!("└{}┘", "─".repeat(inner)),
        );
        return;
    }

    let mut row = placement.row + 2;
    canvas.blit(row, placement.col, &format!("├{}┤", "─".repeat(inner)));
    row += 1;
    let split = member_split(class);
    for (index, member) in class.members.iter().enumerate() {
        if split == Some(index) {
            canvas.blit(row, placement.col, &format!("├{}┤", "─".repeat(inner)));
            row += 1;
        }
        draw_box_label(canvas, row, placement, member, false);
        row += 1;
    }
    canvas.blit(row, placement.col, &format!("└{}┘", "─".repeat(inner)));
}

fn draw_box_label(
    canvas: &mut canvas::MermaidCanvas,
    row: usize,
    placement: PlacedClass,
    label: &ir::Label,
    centered: bool,
) {
    canvas.put(placement.col, row, '│');
    canvas.put(placement.col + placement.width - 1, row, '│');
    let label_width = display_width(&label.text);
    let col = if centered {
        placement.col + 1 + (placement.width - 2 - label_width) / 2
    } else {
        placement.col + 2
    };
    canvas.labels.push(canvas::MermaidCanvasLabel {
        row,
        col,
        text: label.text.clone(),
        source: label.span,
    });
}

fn route_relations(
    canvas: &mut canvas::MermaidCanvas,
    diagram: &ir::Diagram,
    placements: &HashMap<usize, PlacedClass>,
    layer_of: &HashMap<usize, usize>,
    width: usize,
) -> Option<()> {
    let mut relations = diagram.relations.iter().collect::<Vec<_>>();
    relations.sort_by_key(|relation| {
        class_index(diagram, &relation.from.text)
            .map(|index| (layer_of[&index], index))
            .unwrap_or((usize::MAX, usize::MAX))
    });
    let layer_bottoms = placements.iter().fold(
        HashMap::<usize, usize>::new(),
        |mut bottoms, (index, placed)| {
            bottoms
                .entry(layer_of[index])
                .and_modify(|bottom| *bottom = (*bottom).max(placed.row + placed.height))
                .or_insert(placed.row + placed.height);
            bottoms
        },
    );
    let mut offsets = HashMap::<usize, usize>::new();
    let mut routes = RouteGrid::new();
    let mut markers = Vec::new();
    for relation in relations {
        let from_index = class_index(diagram, &relation.from.text)?;
        let to_index = class_index(diagram, &relation.to.text)?;
        let from = placements[&from_index];
        let to = placements[&to_index];
        let layer = layer_of[&from_index];
        let offset = offsets.entry(layer).or_default();
        let label_row = layer_bottoms[&layer] + *offset * 2;
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
        markers.push((relation, from, to));
        if let Some(label) = &relation.label {
            let label_width = display_width(&label.text);
            canvas.labels.push(canvas::MermaidCanvasLabel {
                row: label_row,
                col: (width - label_width) / 2,
                text: label.text.clone(),
                source: label.span,
            });
        }
    }
    for (&(col, row), &mask) in &routes {
        canvas.put(col, row, route_glyph(mask));
    }
    for (relation, from, to) in markers {
        let from_col = from.col + from.width / 2;
        let to_col = to.col + to.width / 2;
        let (from_marker, to_marker) = relation_markers(relation.connector);
        if let Some(marker) = from_marker {
            canvas.put(from_col, from.row + from.height - 1, marker);
        }
        if let Some(marker) = to_marker {
            canvas.put(to_col, to.row, marker);
        }
    }
    Some(())
}

fn relation_markers(connector: &str) -> (Option<char>, Option<char>) {
    match connector {
        "<|--" | "<|.." => (Some('△'), None),
        "--|>" | "..|>" => (None, Some('△')),
        "*--" => (Some('◆'), None),
        "--*" => (None, Some('◆')),
        "o--" => (Some('◇'), None),
        "--o" => (None, Some('◇')),
        "..>" | "-->" => (None, Some('▼')),
        "<.." | "<--" => (Some('▲'), None),
        _ => (None, None),
    }
}

fn connect_route(routes: &mut RouteGrid, from: (usize, usize), to: (usize, usize)) {
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

fn render_stacked(diagram: &ir::Diagram, width: usize) -> Option<Vec<Vec<MermaidRenderSpan>>> {
    let sizes = diagram
        .classes
        .iter()
        .map(class_box_size)
        .collect::<Vec<_>>();
    if sizes.iter().any(|(class_width, _)| *class_width > width) {
        return None;
    }
    let mut canvas = canvas::MermaidCanvas {
        rows: Vec::new(),
        labels: Vec::new(),
    };
    let mut row = 0;
    for (class, (class_width, class_height)) in diagram.classes.iter().zip(&sizes) {
        draw_class_box(
            &mut canvas,
            class,
            PlacedClass {
                row,
                col: 0,
                width: *class_width,
                height: *class_height,
            },
        );
        row += class_height + 1;
    }
    let mut lines = canvas.render();
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
            current.take()?;
        } else if let Some(index) = current {
            if trimmed.contains(['{', '}'])
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
    let from_side = line[..connector_byte].trim();
    let after_byte = connector_byte + connector.len();
    let after = &line[after_byte..];
    let (to_side, label) = after
        .split_once(':')
        .map_or((after.trim(), None), |(to, label)| {
            (to.trim(), Some(label.trim()))
        });
    let from = relation_endpoint(from_side, false)?;
    let to = relation_endpoint(to_side, true)?;
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

fn relation_endpoint(value: &str, leading_cardinality: bool) -> Option<&str> {
    let value = value.trim();
    if leading_cardinality && value.starts_with('"') {
        let end = value[1..].find('"')? + 2;
        return Some(value[end..].trim());
    }
    if !leading_cardinality && value.ends_with('"') {
        return value.split_whitespace().next();
    }
    Some(value)
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
