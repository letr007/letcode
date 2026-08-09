use super::er_ir as ir;
use super::{
    MermaidRenderSpan, MermaidSourceSpan, canvas, render_line_count_within_limits,
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
    let mut depths = vec![0usize; diagram.entities.len()];
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
    if visited != diagram.entities.len() {
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
    let mut row = 0;
    for (layer_index, layer) in layers.iter().enumerate() {
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
        if layer_index + 1 < layers.len() {
            let edge_count = diagram
                .relations
                .iter()
                .filter(|relation| depths[index[relation.from.text.as_str()]] == layer_index)
                .count();
            row += layer_height + edge_count * 2 + 2;
        }
    }
    route_entity_relations(&mut canvas, diagram, &index, &depths, &placements);
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
    placements: &HashMap<usize, PlacedEntity>,
) {
    let layer_bottoms = placements
        .iter()
        .fold(HashMap::new(), |mut bottoms, (entity, placed)| {
            bottoms
                .entry(depths[*entity])
                .and_modify(|bottom: &mut usize| {
                    *bottom = (*bottom).max(placed.row + placed.height)
                })
                .or_insert(placed.row + placed.height);
            bottoms
        });
    let mut offsets = HashMap::<usize, usize>::new();
    for relation in &diagram.relations {
        let from_index = index[relation.from.text.as_str()];
        let to_index = index[relation.to.text.as_str()];
        let from = placements[&from_index];
        let to = placements[&to_index];
        let layer = depths[from_index];
        let offset = offsets.entry(layer).or_default();
        let label_row = layer_bottoms[&layer] + *offset * 2;
        let route_row = label_row + 1;
        *offset += 1;
        let from_col = from.col + from.width / 2;
        let to_col = to.col + to.width / 2;
        for row in from.row + from.height..=route_row {
            canvas.put(from_col, row, '│');
        }
        let (lo, hi) = if from_col < to_col {
            (from_col, to_col)
        } else {
            (to_col, from_col)
        };
        for col in lo..=hi {
            canvas.put(col, route_row, '─');
        }
        for row in route_row..to.row {
            canvas.put(to_col, row, '│');
        }
        canvas.put(
            from_col,
            from.row + from.height - 1,
            cardinality_icon(&relation.from_cardinality.text),
        );
        canvas.put(
            to_col,
            to.row,
            cardinality_icon(&relation.to_cardinality.text),
        );
        let label_width = display_width(&relation.label.text);
        canvas.labels.push(canvas::MermaidCanvasLabel {
            row: label_row,
            col: ((from_col + to_col) / 2).saturating_sub(label_width / 2),
            text: relation.label.text.clone(),
            source: relation.label.span,
        });
    }
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
            if current.take().is_none() {
                return None;
            }
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
