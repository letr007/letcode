//! Mermaid gitGraph parsing and terminal rendering.

use std::collections::HashSet;

use crate::tui::measure::display_width;

use super::{
    MermaidRenderSpan, MermaidSourceSpan, canvas, gitgraph_ir as ir,
    render_line_count_within_limits, source_within_limits,
};

const LANE_GAP: usize = 3;
const MIN_LANE_WIDTH: usize = 7;

pub(super) fn render(source: &str, width: usize) -> Option<Vec<Vec<MermaidRenderSpan>>> {
    let diagram = parse(source)?;
    let label_width = diagram
        .items
        .iter()
        .map(item_label_width)
        .chain(
            diagram
                .branches
                .iter()
                .map(|branch| display_width(&branch.name)),
        )
        .max()
        .unwrap_or(0)
        .max(display_width("commit"));
    let lane_widths = diagram
        .branches
        .iter()
        .map(|branch| MIN_LANE_WIDTH.max(display_width(&branch.name) + 2))
        .collect::<Vec<_>>();
    let track_width = lane_widths
        .iter()
        .enumerate()
        .try_fold(0usize, |total, (index, lane)| {
            total.checked_add(*lane).and_then(|sum| {
                (index == 0)
                    .then_some(sum)
                    .or_else(|| sum.checked_add(LANE_GAP))
            })
        })?;
    let label_start = track_width.checked_add(3)?;
    let graph_width = label_start.checked_add(label_width)?;
    if graph_width == 0 || graph_width > width {
        return None;
    }

    let starts = lane_starts(&lane_widths, 0);
    let centers = lane_widths
        .iter()
        .zip(&starts)
        .map(|(lane, start)| start + lane / 2)
        .collect::<Vec<_>>();
    let branch_index = diagram
        .branches
        .iter()
        .enumerate()
        .map(|(index, branch)| (branch.name.as_str(), index))
        .collect::<std::collections::HashMap<_, _>>();

    let mut canvas = canvas::MermaidCanvas {
        rows: Vec::new(),
        labels: Vec::new(),
    };
    canvas.ensure_row(0, graph_width);
    for (index, branch) in diagram.branches.iter().enumerate() {
        let text_width = display_width(&branch.name);
        let col = centers[index].saturating_sub(text_width / 2);
        if let Some(span) = branch.span {
            canvas.labels.push(canvas::MermaidCanvasLabel {
                row: 0,
                col,
                text: branch.name.clone(),
                source: span,
            });
        } else {
            canvas.blit(0, col, &branch.name);
        }
    }

    let mut current = "main".to_string();
    let mut active = vec![false; diagram.branches.len()];
    active[*branch_index.get("main")?] = true;
    for (row, item) in diagram.items.iter().enumerate() {
        let row = row + 1;
        draw_tracks(&mut canvas, row, graph_width, &centers, &active);
        match item {
            ir::Item::Commit(commit) => {
                let index = *branch_index.get(current.as_str())?;
                canvas.put(centers[index], row, '●');
                draw_command_prefix(&mut canvas, row, label_start, "commit");
                if let Some(id) = &commit.id {
                    canvas.labels.push(canvas::MermaidCanvasLabel {
                        row,
                        col: label_start + display_width("commit "),
                        text: id.text.clone(),
                        source: id.span,
                    });
                }
            }
            ir::Item::Branch(branch) => {
                let from = *branch_index.get(current.as_str())?;
                let to = *branch_index.get(branch.text.as_str())?;
                connect_branch(&mut canvas, row, centers[from], centers[to]);
                active[to] = true;
                draw_command_label(&mut canvas, row, label_start, "branch ", branch);
                current = branch.text.clone();
            }
            ir::Item::Checkout(branch) => {
                let index = *branch_index.get(branch.text.as_str())?;
                canvas.put(centers[index], row, '↳');
                draw_command_label(&mut canvas, row, label_start, "checkout ", branch);
                current = branch.text.clone();
            }
            ir::Item::Merge(branch) => {
                let from = *branch_index.get(branch.text.as_str())?;
                let to = *branch_index.get(current.as_str())?;
                connect_merge(&mut canvas, row, centers[from], centers[to]);
                draw_command_label(&mut canvas, row, label_start, "merge ", branch);
            }
        }
    }

    let lines = canvas.render();
    (render_line_count_within_limits(lines.len())
        && lines.iter().all(|line| {
            line.iter()
                .map(|part| display_width(&part.text))
                .sum::<usize>()
                <= width
        }))
    .then_some(lines)
}

fn draw_tracks(
    canvas: &mut canvas::MermaidCanvas,
    row: usize,
    width: usize,
    centers: &[usize],
    active: &[bool],
) {
    canvas.ensure_row(row, width);
    for (center, active) in centers.iter().zip(active) {
        if *active {
            canvas.put(*center, row, '│');
        }
    }
}

fn connect_branch(canvas: &mut canvas::MermaidCanvas, row: usize, from: usize, to: usize) {
    let (left, right) = (from.min(to), from.max(to));
    for col in left..=right {
        let glyph = if col == from {
            if from < to { '├' } else { '┤' }
        } else if col == to {
            if from < to { '┐' } else { '┌' }
        } else {
            '─'
        };
        canvas.put(col, row, glyph);
    }
}

fn connect_merge(canvas: &mut canvas::MermaidCanvas, row: usize, from: usize, to: usize) {
    let (left, right) = (from.min(to), from.max(to));
    for col in left..=right {
        let glyph = if col == from {
            '○'
        } else if col == to {
            '●'
        } else if from < to && col + 1 == to {
            '▶'
        } else if from > to && col == to + 1 {
            '◀'
        } else {
            '─'
        };
        canvas.put(col, row, glyph);
    }
}

fn draw_command_prefix(canvas: &mut canvas::MermaidCanvas, row: usize, col: usize, command: &str) {
    canvas.blit(row, col, &format!("{command} "));
}

fn draw_command_label(
    canvas: &mut canvas::MermaidCanvas,
    row: usize,
    col: usize,
    prefix: &str,
    label: &ir::Label,
) {
    canvas.blit(row, col, prefix);
    canvas.labels.push(canvas::MermaidCanvasLabel {
        row,
        col: col + display_width(prefix),
        text: label.text.clone(),
        source: label.span,
    });
}

fn item_label_width(item: &ir::Item) -> usize {
    match item {
        ir::Item::Commit(commit) => commit.id.as_ref().map_or(display_width("commit"), |id| {
            display_width("commit ") + display_width(&id.text)
        }),
        ir::Item::Branch(label) => display_width("branch ") + display_width(&label.text),
        ir::Item::Checkout(label) => display_width("checkout ") + display_width(&label.text),
        ir::Item::Merge(label) => display_width("merge ") + display_width(&label.text),
    }
}

fn lane_starts(widths: &[usize], start: usize) -> Vec<usize> {
    let mut result = Vec::with_capacity(widths.len());
    let mut cursor = start;
    for (index, width) in widths.iter().enumerate() {
        result.push(cursor);
        cursor += width;
        if index + 1 < widths.len() {
            cursor += LANE_GAP;
        }
    }
    result
}

fn parse(source: &str) -> Option<ir::Diagram> {
    if !source_within_limits(source) || source.contains('\r') {
        return None;
    }
    let mut lines = source.split('\n');
    if lines.next()? != "gitGraph" {
        return None;
    }

    let mut branches = vec![ir::Branch {
        name: "main".to_string(),
        span: None,
    }];
    let mut known = HashSet::from([String::from("main")]);
    let mut commit_ids = HashSet::new();
    let mut current = String::from("main");
    let mut items = Vec::new();
    let mut commits = 0usize;
    let mut offset = "gitGraph".chars().count() + 1;
    for raw in lines {
        let line_len = raw.chars().count();
        if raw.contains('\t') || raw.chars().last().is_some_and(char::is_whitespace) {
            return None;
        }
        let leading = raw.chars().count() - raw.trim_start().chars().count();
        let base = offset + leading;
        let trimmed = raw.trim();
        if trimmed.is_empty() || trimmed.starts_with("%%") {
            offset += line_len + 1;
            continue;
        }
        let (command, rest) = trimmed
            .split_once(char::is_whitespace)
            .map_or((trimmed, ""), |(command, rest)| (command, rest.trim()));
        match command {
            "commit" => {
                let id = parse_commit_id(rest, trimmed, base)?;
                if let Some(id) = &id {
                    if !commit_ids.insert(id.text.clone()) {
                        return None;
                    }
                }
                items.push(ir::Item::Commit(ir::Commit { id }));
                commits += 1;
            }
            "branch" => {
                let label = parse_name(rest, trimmed, base)?;
                if !known.insert(label.text.clone()) {
                    return None;
                }
                branches.push(ir::Branch {
                    name: label.text.clone(),
                    span: Some(label.span),
                });
                current = label.text.clone();
                items.push(ir::Item::Branch(label));
            }
            "checkout" => {
                let label = parse_name(rest, trimmed, base)?;
                if !known.contains(&label.text) {
                    return None;
                }
                current = label.text.clone();
                items.push(ir::Item::Checkout(label));
            }
            "merge" => {
                let label = parse_name(rest, trimmed, base)?;
                if !known.contains(&label.text) || label.text == current {
                    return None;
                }
                items.push(ir::Item::Merge(label));
            }
            _ => return None,
        }
        offset += line_len + 1;
    }
    if commits == 0 {
        return None;
    }
    Some(ir::Diagram { branches, items })
}

fn parse_commit_id(rest: &str, line: &str, base: usize) -> Option<Option<ir::Label>> {
    if rest.is_empty() {
        return Some(None);
    }
    let value = rest.strip_prefix("id:")?.trim();
    if value.is_empty() {
        return None;
    }
    let (text, quote_offset) = if let Some(quoted) = value.strip_prefix('"') {
        let end = quoted.strip_suffix('"')?;
        if end.is_empty() || end.contains('"') {
            return None;
        }
        (end, 1)
    } else {
        if !valid_name(value) {
            return None;
        }
        (value, 0)
    };
    let id_byte = line.find("id:")?;
    let value_byte = id_byte + 3 + line[id_byte + 3..].find(value)?;
    let start = base + line[..value_byte].chars().count() + quote_offset;
    Some(Some(ir::Label {
        text: text.to_string(),
        span: MermaidSourceSpan::new(start, start + text.chars().count()),
    }))
}

fn parse_name(rest: &str, line: &str, base: usize) -> Option<ir::Label> {
    if rest.is_empty() || rest.split_whitespace().count() != 1 || !valid_name(rest) {
        return None;
    }
    let argument_byte = line.find(char::is_whitespace)?;
    let value_byte = argument_byte + line[argument_byte..].find(rest)?;
    let start = base + line[..value_byte].chars().count();
    Some(ir::Label {
        text: rest.to_string(),
        span: MermaidSourceSpan::new(start, start + rest.chars().count()),
    })
}

fn valid_name(value: &str) -> bool {
    !value.is_empty()
        && value
            .chars()
            .all(|ch| ch.is_alphanumeric() || matches!(ch, '_' | '-' | '/' | '.'))
}
