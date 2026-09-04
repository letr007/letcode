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
                canvas.put(centers[index], row, commit_glyph(commit.commit_type));
                draw_command_prefix(&mut canvas, row, label_start, "commit");
                let mut col = label_start + display_width("commit ");
                if let Some(id) = &commit.id {
                    draw_source_label(&mut canvas, row, col, id);
                    col += display_width(&id.text);
                }
                if let Some(tag) = &commit.tag {
                    let prefix = if commit.id.is_some() {
                        " tag: "
                    } else {
                        "tag: "
                    };
                    draw_metadata_label(&mut canvas, row, col, prefix, tag);
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
            ir::Item::Merge(merge) => {
                let from = *branch_index.get(merge.branch.text.as_str())?;
                let to = *branch_index.get(current.as_str())?;
                connect_merge(
                    &mut canvas,
                    row,
                    centers[from],
                    centers[to],
                    commit_glyph(merge.commit_type),
                );
                draw_command_label(&mut canvas, row, label_start, "merge ", &merge.branch);
                let mut col =
                    label_start + display_width("merge ") + display_width(&merge.branch.text);
                if let Some(id) = &merge.id {
                    col += draw_metadata_label(&mut canvas, row, col, " id: ", id);
                }
                if let Some(tag) = &merge.tag {
                    draw_metadata_label(&mut canvas, row, col, " tag: ", tag);
                }
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

fn connect_merge(
    canvas: &mut canvas::MermaidCanvas,
    row: usize,
    from: usize,
    to: usize,
    target_glyph: char,
) {
    let (left, right) = (from.min(to), from.max(to));
    for col in left..=right {
        let glyph = if col == from {
            '○'
        } else if col == to {
            target_glyph
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

fn commit_glyph(commit_type: ir::CommitType) -> char {
    match commit_type {
        ir::CommitType::Normal => '●',
        ir::CommitType::Reverse => '⊗',
        ir::CommitType::Highlight => '■',
    }
}

fn draw_command_prefix(canvas: &mut canvas::MermaidCanvas, row: usize, col: usize, command: &str) {
    canvas.blit(row, col, &format!("{command} "));
}

fn draw_source_label(
    canvas: &mut canvas::MermaidCanvas,
    row: usize,
    col: usize,
    label: &ir::Label,
) {
    canvas.labels.push(canvas::MermaidCanvasLabel {
        row,
        col,
        text: label.text.clone(),
        source: label.span,
    });
}

fn draw_metadata_label(
    canvas: &mut canvas::MermaidCanvas,
    row: usize,
    col: usize,
    prefix: &str,
    label: &ir::Label,
) -> usize {
    canvas.blit(row, col, prefix);
    let value_col = col + display_width(prefix);
    draw_source_label(canvas, row, value_col, label);
    display_width(prefix) + display_width(&label.text)
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
        ir::Item::Commit(commit) => {
            display_width("commit")
                + commit
                    .id
                    .as_ref()
                    .map_or(0, |id| display_width(" ") + display_width(&id.text))
                + if commit.id.is_none() && commit.tag.is_some() {
                    1
                } else {
                    0
                }
                + commit.tag.as_ref().map_or(0, |tag| {
                    display_width(if commit.id.is_some() {
                        " tag: "
                    } else {
                        "tag: "
                    }) + display_width(&tag.text)
                })
        }
        ir::Item::Branch(label) => display_width("branch ") + display_width(&label.text),
        ir::Item::Checkout(label) => display_width("checkout ") + display_width(&label.text),
        ir::Item::Merge(merge) => {
            display_width("merge ")
                + display_width(&merge.branch.text)
                + merge
                    .id
                    .as_ref()
                    .map_or(0, |id| display_width(" id: ") + display_width(&id.text))
                + merge
                    .tag
                    .as_ref()
                    .map_or(0, |tag| display_width(" tag: ") + display_width(&tag.text))
        }
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
        let command_end = trimmed.find(char::is_whitespace).unwrap_or(trimmed.len());
        let whitespace = &trimmed[command_end..];
        let rest_base = base
            + trimmed[..command_end].chars().count()
            + whitespace
                .chars()
                .count()
                .saturating_sub(whitespace.trim_start().chars().count());
        let (command, rest) = trimmed
            .split_once(char::is_whitespace)
            .map_or((trimmed, ""), |(command, rest)| (command, rest.trim()));
        match command {
            "commit" => {
                let metadata = parse_metadata(rest, rest_base)?;
                if let Some(id) = &metadata.id {
                    if !commit_ids.insert(id.text.clone()) {
                        return None;
                    }
                }
                items.push(ir::Item::Commit(ir::Commit {
                    id: metadata.id,
                    tag: metadata.tag,
                    commit_type: metadata.commit_type.unwrap_or(ir::CommitType::Normal),
                }));
                commits += 1;
            }
            "branch" => {
                let label = parse_name(rest, rest_base)?;
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
                let label = parse_name(rest, rest_base)?;
                if !known.contains(&label.text) {
                    return None;
                }
                current = label.text.clone();
                items.push(ir::Item::Checkout(label));
            }
            "merge" => {
                let merge = parse_merge(rest, rest_base)?;
                if !known.contains(&merge.branch.text) || merge.branch.text == current {
                    return None;
                }
                items.push(ir::Item::Merge(merge));
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

fn parse_merge(rest: &str, base: usize) -> Option<ir::Merge> {
    let (branch_name, metadata) = rest
        .split_once(char::is_whitespace)
        .map_or((rest, ""), |(branch, metadata)| (branch, metadata.trim()));
    let branch = parse_name(branch_name, base)?;
    let metadata_base = base + branch_name.chars().count();
    let metadata_base = metadata_base
        + rest[branch_name.len()..]
            .chars()
            .count()
            .saturating_sub(rest[branch_name.len()..].trim_start().chars().count());
    let metadata = parse_metadata(metadata, metadata_base)?;
    Some(ir::Merge {
        branch,
        id: metadata.id,
        tag: metadata.tag,
        commit_type: metadata.commit_type.unwrap_or(ir::CommitType::Normal),
    })
}

fn parse_metadata(rest: &str, base: usize) -> Option<CommitMetadata> {
    let mut metadata = CommitMetadata::default();
    let mut cursor = 0usize;
    while cursor < rest.len() {
        while rest[cursor..]
            .chars()
            .next()
            .is_some_and(char::is_whitespace)
        {
            cursor += rest[cursor..].chars().next()?.len_utf8();
        }
        if cursor == rest.len() {
            break;
        }
        let key_start = cursor;
        while cursor < rest.len()
            && !rest[cursor..].chars().next()?.is_whitespace()
            && rest.as_bytes()[cursor] != b':'
        {
            cursor += rest[cursor..].chars().next()?.len_utf8();
        }
        let key = &rest[key_start..cursor];
        while cursor < rest.len() && rest[cursor..].chars().next()?.is_whitespace() {
            cursor += rest[cursor..].chars().next()?.len_utf8();
        }
        if key.is_empty() || rest.as_bytes().get(cursor) != Some(&b':') {
            return None;
        }
        cursor += 1;
        while cursor < rest.len() && rest[cursor..].chars().next()?.is_whitespace() {
            cursor += rest[cursor..].chars().next()?.len_utf8();
        }
        let value_start = cursor;
        let (value, value_span_start, quoted) = if rest.as_bytes().get(cursor) == Some(&b'"') {
            cursor += 1;
            let content_start = cursor;
            let end = content_start + rest[content_start..].find('"')?;
            let value = &rest[content_start..end];
            if value.is_empty() {
                return None;
            }
            cursor = end + 1;
            (value, content_start, true)
        } else {
            while cursor < rest.len() && !rest[cursor..].chars().next()?.is_whitespace() {
                cursor += rest[cursor..].chars().next()?.len_utf8();
            }
            let value = &rest[value_start..cursor];
            if value.is_empty() {
                return None;
            }
            (value, value_start, false)
        };
        if cursor < rest.len() && !rest[cursor..].chars().next()?.is_whitespace() {
            return None;
        }
        let label = || ir::Label {
            text: value.to_string(),
            span: MermaidSourceSpan::new(
                base + rest[..value_span_start].chars().count(),
                base + rest[..value_span_start].chars().count() + value.chars().count(),
            ),
        };
        match key {
            "id" => {
                if metadata.id.is_some() || (!quoted && !valid_name(value)) {
                    return None;
                }
                metadata.id = Some(label());
            }
            "tag" => {
                if metadata.tag.is_some() || (!quoted && !valid_name(value)) {
                    return None;
                }
                metadata.tag = Some(label());
            }
            "type" => {
                if quoted || metadata.commit_type.is_some() {
                    return None;
                }
                metadata.commit_type = Some(match value {
                    "NORMAL" => ir::CommitType::Normal,
                    "REVERSE" => ir::CommitType::Reverse,
                    "HIGHLIGHT" => ir::CommitType::Highlight,
                    _ => return None,
                });
            }
            _ => return None,
        }
    }
    Some(metadata)
}

#[derive(Default)]
struct CommitMetadata {
    id: Option<ir::Label>,
    tag: Option<ir::Label>,
    commit_type: Option<ir::CommitType>,
}

fn parse_name(rest: &str, base: usize) -> Option<ir::Label> {
    if rest.is_empty() || rest.split_whitespace().count() != 1 || !valid_name(rest) {
        return None;
    }
    Some(ir::Label {
        text: rest.to_string(),
        span: MermaidSourceSpan::new(base, base + rest.chars().count()),
    })
}

fn valid_name(value: &str) -> bool {
    !value.is_empty()
        && value
            .chars()
            .all(|ch| ch.is_alphanumeric() || matches!(ch, '_' | '-' | '/' | '.'))
}
