use super::er_ir as ir;
use super::{
    MermaidRenderSpan, MermaidSourceSpan, canvas, render_line_count_within_limits, routing,
    source_within_limits,
};
use crate::tui::measure::display_width;
use std::collections::{HashMap, HashSet, VecDeque};

const CARDINALITIES: [&str; 7] = ["||", "o|", "|o", "o{", "}o", "|{", "}|"];

pub(super) fn render(source: &str, width: usize) -> Option<Vec<Vec<MermaidRenderSpan>>> {
    let diagram = parse(source)?;
    if let Some(canvas) = layout(&diagram, width) {
        let lines = canvas.render();
        if render_line_count_within_limits(lines.len()) {
            return Some(lines);
        }
    }
    render_linear(&diagram, width)
}

fn render_linear(diagram: &ir::Diagram, width: usize) -> Option<Vec<Vec<MermaidRenderSpan>>> {
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

#[derive(Clone, Copy)]
struct PlacedEntity {
    row: usize,
    col: usize,
    width: usize,
    height: usize,
}

struct LayerRows {
    starts: Vec<usize>,
    bottoms: Vec<usize>,
}

fn layout(diagram: &ir::Diagram, width: usize) -> Option<canvas::MermaidCanvas> {
    if diagram.entities.len() > 24 || diagram.relations.len() > 48 {
        return None;
    }
    let index = diagram
        .entities
        .iter()
        .enumerate()
        .map(|(index, entity)| (entity.name.text.as_str(), index))
        .collect::<HashMap<_, _>>();
    let mut indegree = vec![0usize; diagram.entities.len()];
    let mut outgoing = vec![Vec::new(); diagram.entities.len()];
    for relation in &diagram.relations {
        let from = index[relation.from.text.as_str()];
        let to = index[relation.to.text.as_str()];
        outgoing[from].push(to);
        indegree[to] += 1;
    }
    let mut queue = indegree
        .iter()
        .enumerate()
        .filter_map(|(index, degree)| (*degree == 0).then_some(index))
        .collect::<VecDeque<_>>();
    let mut topo_order = Vec::with_capacity(diagram.entities.len());
    while let Some(from) = queue.pop_front() {
        topo_order.push(from);
        for to in &outgoing[from] {
            indegree[*to] -= 1;
            if indegree[*to] == 0 {
                queue.push_back(*to);
            }
        }
    }
    if topo_order.len() != diagram.entities.len() {
        return None;
    }

    let mut earliest_layers = vec![0usize; diagram.entities.len()];
    for &from in &topo_order {
        for to in &outgoing[from] {
            earliest_layers[*to] = earliest_layers[*to].max(earliest_layers[from] + 1);
        }
    }
    let max_depth = earliest_layers.iter().copied().max().unwrap_or(0);
    let mut depths = vec![max_depth; diagram.entities.len()];
    for &entity in topo_order.iter().rev() {
        if let Some(min_successor_depth) = outgoing[entity].iter().map(|to| depths[*to]).min() {
            depths[entity] = min_successor_depth.saturating_sub(1);
        }
    }
    if diagram.relations.iter().any(|relation| {
        depths[index[relation.to.text.as_str()]] != depths[index[relation.from.text.as_str()]] + 1
    }) {
        return None;
    }
    let mut layers = vec![Vec::new(); depths.iter().copied().max().unwrap_or(0) + 1];
    for (entity, depth) in depths.iter().copied().enumerate() {
        layers[depth].push(entity);
    }
    let sizes = diagram
        .entities
        .iter()
        .map(|entity| {
            let inner = std::iter::once(display_width(&entity.name.text))
                .chain(
                    entity
                        .attributes
                        .iter()
                        .map(|attribute| display_width(&attribute.text)),
                )
                .max()
                .unwrap_or(1)
                .max(1)
                + 2;
            (inner + 2, entity.attributes.len() + 4)
        })
        .collect::<Vec<_>>();
    let layer_widths = layers
        .iter()
        .map(|layer| {
            layer.iter().map(|entity| sizes[*entity].0).sum::<usize>()
                + 6 * layer.len().saturating_sub(1)
        })
        .collect::<Vec<_>>();
    let graph_width = layer_widths.iter().copied().max().unwrap_or(0);
    if graph_width == 0 || graph_width > width {
        return None;
    }

    let mut canvas = canvas::MermaidCanvas {
        rows: Vec::new(),
        labels: Vec::new(),
    };
    let mut placements = HashMap::new();
    let mut layer_starts = Vec::with_capacity(layers.len());
    let mut layer_bottoms = Vec::with_capacity(layers.len());
    let mut row = 0;
    for (layer_index, layer) in layers.iter().enumerate() {
        layer_starts.push(row);
        let layer_height = layer
            .iter()
            .map(|entity| sizes[*entity].1)
            .max()
            .unwrap_or(0);
        let mut col = (graph_width - layer_widths[layer_index]) / 2;
        for entity in layer {
            let (entity_width, entity_height) = sizes[*entity];
            let placed = PlacedEntity {
                row,
                col,
                width: entity_width,
                height: entity_height,
            };
            draw_entity(&mut canvas, &diagram.entities[*entity], placed);
            placements.insert(*entity, placed);
            col += entity_width + 6;
        }
        layer_bottoms.push(row + layer_height);
        if layer_index + 1 < layers.len() {
            let edge_count = diagram
                .relations
                .iter()
                .filter(|relation| depths[index[relation.from.text.as_str()]] == layer_index)
                .count();
            row += layer_height + edge_count * 2 + 2;
        }
    }
    route_entity_relations(
        &mut canvas,
        diagram,
        &index,
        &depths,
        graph_width,
        &placements,
        &LayerRows {
            starts: layer_starts,
            bottoms: layer_bottoms,
        },
    )?;
    Some(canvas)
}

fn draw_entity(canvas: &mut canvas::MermaidCanvas, entity: &ir::Entity, placed: PlacedEntity) {
    let inner = placed.width - 2;
    canvas.blit(placed.row, placed.col, &format!("┌{}┐", "─".repeat(inner)));
    canvas.put(placed.col, placed.row + 1, '│');
    canvas.put(placed.col + placed.width - 1, placed.row + 1, '│');
    canvas.labels.push(canvas::MermaidCanvasLabel {
        row: placed.row + 1,
        col: placed.col + 1 + (inner - display_width(&entity.name.text)) / 2,
        text: entity.name.text.clone(),
        source: entity.name.span,
    });
    canvas.blit(
        placed.row + 2,
        placed.col,
        &format!("├{}┤", "─".repeat(inner)),
    );
    let mut row = placed.row + 3;
    for attribute in &entity.attributes {
        canvas.put(placed.col, row, '│');
        canvas.put(placed.col + placed.width - 1, row, '│');
        canvas.labels.push(canvas::MermaidCanvasLabel {
            row,
            col: placed.col + 2,
            text: attribute.text.clone(),
            source: attribute.span,
        });
        row += 1;
    }
    canvas.blit(row, placed.col, &format!("└{}┘", "─".repeat(inner)));
}

fn route_entity_relations(
    canvas: &mut canvas::MermaidCanvas,
    diagram: &ir::Diagram,
    index: &HashMap<&str, usize>,
    depths: &[usize],
    graph_width: usize,
    placements: &HashMap<usize, PlacedEntity>,
    layer_rows: &LayerRows,
) -> Option<()> {
    let mut outgoing = HashMap::<usize, Vec<usize>>::new();
    let mut incoming = HashMap::<usize, Vec<usize>>::new();
    for (relation_index, relation) in diagram.relations.iter().enumerate() {
        let from = index[relation.from.text.as_str()];
        let to = index[relation.to.text.as_str()];
        outgoing.entry(from).or_default().push(relation_index);
        incoming.entry(to).or_default().push(relation_index);
    }

    let mut tracks = HashMap::<usize, routing::TrackAllocator>::new();
    let mut routes = routing::RouteGrid::new();
    for (relation_index, relation) in diagram.relations.iter().enumerate() {
        let from_index = index[relation.from.text.as_str()];
        let to_index = index[relation.to.text.as_str()];
        let from = placements[&from_index];
        let to = placements[&to_index];
        let layer = depths[from_index];
        let min_label_row = layer_rows.bottoms[layer] + 1;
        let max_label_row = layer_rows.starts.get(layer + 1)?.saturating_sub(3);
        let label_row = tracks.entry(layer).or_default().reserve(
            min_label_row,
            min_label_row,
            max_label_row.checked_add(1)?,
            2,
        )?;
        let route_row = label_row.checked_add(1)?;
        let from_ordinal = outgoing[&from_index]
            .iter()
            .position(|index| *index == relation_index)?;
        let to_ordinal = incoming[&to_index]
            .iter()
            .position(|index| *index == relation_index)?;
        let from_col = port_col(from, from_ordinal, outgoing[&from_index].len())?;
        let to_col = port_col(to, to_ordinal, incoming[&to_index].len())?;
        let source_port = (from_col, from.row + from.height);
        let target_port = (to_col, to.row - 1);
        routes.connect(source_port, (from_col, route_row));
        routes.connect((from_col, route_row), (to_col, route_row));
        routes.connect((to_col, route_row), target_port);

        canvas.put(
            source_port.0,
            source_port.1,
            cardinality_icon(&relation.from_cardinality.text),
        );
        canvas.put(
            target_port.0,
            target_port.1,
            cardinality_icon(&relation.to_cardinality.text),
        );

        let label_width = display_width(&relation.label.text).max(1);
        if label_width > graph_width {
            return None;
        }
        let lo = from_col.min(to_col);
        let hi = from_col.max(to_col);
        let col = if hi.saturating_sub(lo) >= label_width {
            lo + (hi - lo - label_width) / 2
        } else {
            (graph_width - label_width) / 2
        };
        canvas.labels.push(canvas::MermaidCanvasLabel {
            row: label_row,
            col,
            text: relation.label.text.clone(),
            source: relation.label.span,
        });
    }

    let mut cells = routes.iter().collect::<Vec<_>>();
    cells.sort_by_key(|((col, row), _)| (*row, *col));
    for (&(col, row), &mask) in cells {
        if !matches!(
            canvas.rows.get(row).and_then(|line| line.get(col)),
            Some(canvas::MermaidCell::Char('1' | '○' | '┤' | '◇' | '•'))
        ) {
            canvas.put(col, row, routing::route_glyph(mask));
        }
    }
    Some(())
}

fn port_col(placed: PlacedEntity, ordinal: usize, count: usize) -> Option<usize> {
    let span = placed.width.saturating_sub(2);
    if count == 0 || count > span || ordinal >= count {
        return None;
    }
    Some(placed.col + ((ordinal + 1) * (span + 1) / (count + 1)))
}

fn cardinality_icon(cardinality: &str) -> char {
    match cardinality {
        "||" => '1',
        "o|" | "|o" => '○',
        "|{" | "}|" => '┤',
        "o{" | "}o" => '◇',
        _ => '•',
    }
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
            current.take()?;
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

#[cfg(test)]
mod tests {
    use super::{PlacedEntity, port_col};

    #[test]
    fn full_port_capacity_uses_each_interior_column_once() {
        let placed = PlacedEntity {
            row: 0,
            col: 4,
            width: 7,
            height: 3,
        };
        let ports = (0..5)
            .map(|ordinal| port_col(placed, ordinal, 5).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(ports, vec![5, 6, 7, 8, 9]);
    }
}
