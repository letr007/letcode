//! Sequence diagram parsing and rendering.

use std::collections::HashMap;

use crate::tui::measure::display_width;

use super::{
    MermaidRenderSpan, MermaidSourceSpan, canvas, render_line_count_within_limits,
    sequence_ir as ir,
};

pub(crate) fn render(source: &str, width: usize) -> Option<Vec<Vec<MermaidRenderSpan>>> {
    let sequence = parse(source)?;
    validate_activations(&sequence.items, &mut HashMap::new())?;
    if let Some(canvas) = layout(&sequence, width) {
        let lines = canvas.render();
        if render_line_count_within_limits(lines.len()) {
            return Some(lines);
        }
    }

    let mut lines = Vec::new();
    let mut message_number = 1;
    render_linear_items(
        &sequence,
        &sequence.items,
        0,
        &mut lines,
        sequence.autonumber,
        &mut message_number,
    )?;
    (render_line_count_within_limits(lines.len())
        && lines.iter().all(|line| {
            line.iter()
                .map(|span| display_width(&span.text))
                .sum::<usize>()
                <= width
        }))
    .then_some(lines)
}

fn validate_activations(
    items: &[ir::MermaidSequenceItem],
    active: &mut HashMap<String, usize>,
) -> Option<()> {
    for item in items {
        match item {
            ir::MermaidSequenceItem::Message(message) => {
                if message.activate {
                    let count = active.entry(message.to.clone()).or_default();
                    *count = count.checked_add(1)?;
                }
                if message.deactivate {
                    deactivate_participant(active, &message.from)?;
                }
            }
            ir::MermaidSequenceItem::Block(block) => {
                let incoming = active.clone();
                let mut outgoing = None;
                for branch in &block.branches {
                    let mut branch_active = incoming.clone();
                    validate_activations(&branch.items, &mut branch_active)?;
                    if outgoing
                        .as_ref()
                        .is_some_and(|expected| expected != &branch_active)
                    {
                        return None;
                    }
                    outgoing = Some(branch_active);
                }
                *active = outgoing.unwrap_or(incoming);
            }
            ir::MermaidSequenceItem::Activation(activation) => {
                if activation.active {
                    let count = active.entry(activation.participant.clone()).or_default();
                    *count = count.checked_add(1)?;
                } else {
                    deactivate_participant(active, &activation.participant)?;
                }
            }
            ir::MermaidSequenceItem::Note(_) => {}
        }
    }
    Some(())
}

fn layout(sequence: &ir::MermaidSequence, width: usize) -> Option<canvas::MermaidCanvas> {
    if sequence.participants.len() > 16 || has_self_message(&sequence.items) {
        return None;
    }
    let mut participants = sequence.participants.iter().collect::<Vec<_>>();
    participants.sort_by_key(|(_, participant)| participant.start);
    let message_width = max_message_width(&sequence.items).max(1);
    let number_width = if sequence.autonumber {
        message_count(&sequence.items)
            .max(1)
            .to_string()
            .chars()
            .count()
            + 1
    } else {
        0
    };
    let block_width = max_block_width(&sequence.items);
    let label_widths = participants
        .iter()
        .map(|(_, participant)| display_width(&participant.label).max(1))
        .collect::<Vec<_>>();
    let mut centers = vec![label_widths[0].div_ceil(2) + 1 + number_width];
    for index in 1..participants.len() {
        let labels_gap = label_widths[index - 1].div_ceil(2) + label_widths[index].div_ceil(2) + 4;
        centers.push(centers[index - 1] + (message_width + 6).max(labels_gap).max(12));
    }
    let mut diagram_width =
        centers.last().copied()? + label_widths.last().copied()?.div_ceil(2) + 1;
    diagram_width = diagram_width.max(block_width + 3);
    if diagram_width > width {
        return None;
    }

    let columns = participants
        .iter()
        .zip(&centers)
        .map(|((id, _), center)| (id.as_str(), *center))
        .collect::<HashMap<_, _>>();
    let mut canvas = canvas::MermaidCanvas {
        rows: Vec::new(),
        labels: Vec::new(),
    };
    canvas.ensure_row(0, diagram_width);
    for ((_, participant), center) in participants.iter().zip(&centers) {
        canvas.labels.push(canvas::MermaidCanvasLabel {
            row: 0,
            col: center.saturating_sub(display_width(&participant.label) / 2),
            text: participant.label.clone(),
            source: MermaidSourceSpan::new(participant.start, participant.end),
        });
    }
    draw_lifelines(&mut canvas, 1, diagram_width, &centers);
    let mut row = 2;
    let mut message_number = 1;
    let mut active = HashMap::new();
    draw_sequence_items(
        &mut canvas,
        &sequence.items,
        &columns,
        &centers,
        diagram_width,
        &mut row,
        sequence.autonumber,
        &mut message_number,
        &mut active,
    )?;
    Some(canvas)
}

fn draw_sequence_items(
    canvas: &mut canvas::MermaidCanvas,
    items: &[ir::MermaidSequenceItem],
    columns: &HashMap<&str, usize>,
    centers: &[usize],
    width: usize,
    row: &mut usize,
    autonumber: bool,
    message_number: &mut usize,
    active: &mut HashMap<String, usize>,
) -> Option<()> {
    for item in items {
        match item {
            ir::MermaidSequenceItem::Message(message) => {
                let from = *columns.get(message.from.as_str())?;
                let to = *columns.get(message.to.as_str())?;
                if message.activate {
                    let count = active.entry(message.to.clone()).or_default();
                    *count = count.checked_add(1)?;
                }
                draw_lifelines_active(canvas, *row, width, centers, columns, active);
                if autonumber {
                    canvas.blit(*row, 0, &format!("{} ", *message_number));
                    *message_number += 1;
                }
                let (left, right) = if from < to { (from, to) } else { (to, from) };
                let line = if message.dashed { '╌' } else { '─' };
                for col in left + 1..right {
                    canvas.put(col, *row, line);
                }
                canvas.put(
                    if from < to { right - 1 } else { left + 1 },
                    *row,
                    if from < to { '▶' } else { '◀' },
                );
                let label_width = display_width(&message.label.text);
                let available = right.saturating_sub(left + 2);
                if label_width > available {
                    return None;
                }
                let label_start = if from < to { left + 1 } else { left + 2 };
                canvas.labels.push(canvas::MermaidCanvasLabel {
                    row: *row,
                    col: label_start + (available - label_width) / 2,
                    text: message.label.text.clone(),
                    source: MermaidSourceSpan::new(message.label.start, message.label.end),
                });
                if message.deactivate {
                    deactivate_participant(active, &message.from)?;
                }
                *row += 1;
                draw_lifelines_active(canvas, *row, width, centers, columns, active);
                *row += 1;
            }
            ir::MermaidSequenceItem::Block(block) => {
                draw_frame_row(
                    canvas,
                    *row,
                    width,
                    centers,
                    '┌',
                    '┐',
                    &format!("{} ", block.kind.keyword()),
                    Some(&block.label),
                );
                *row += 1;
                let incoming = active.clone();
                let mut outgoing = None;
                for (index, branch) in block.branches.iter().enumerate() {
                    if index > 0 {
                        draw_frame_row(
                            canvas,
                            *row,
                            width,
                            centers,
                            '├',
                            '┤',
                            &format!("{} ", block.kind.branch_keyword()),
                            branch.label.as_ref(),
                        );
                        *row += 1;
                    }
                    let mut branch_active = incoming.clone();
                    draw_sequence_items(
                        canvas,
                        &branch.items,
                        columns,
                        centers,
                        width,
                        row,
                        autonumber,
                        message_number,
                        &mut branch_active,
                    )?;
                    if outgoing
                        .as_ref()
                        .is_some_and(|expected| expected != &branch_active)
                    {
                        return None;
                    }
                    outgoing = Some(branch_active);
                }
                *active = outgoing.unwrap_or(incoming);
                draw_frame_row(canvas, *row, width, centers, '└', '┘', "", None);
                *row += 1;
            }
            ir::MermaidSequenceItem::Activation(activation) => {
                let center = *columns.get(activation.participant.as_str())?;
                if activation.active {
                    let count = active.entry(activation.participant.clone()).or_default();
                    *count = count.checked_add(1)?;
                } else {
                    deactivate_participant(active, &activation.participant)?;
                }
                draw_lifelines_active(canvas, *row, width, centers, columns, active);
                canvas.put(center, *row, if activation.active { '▐' } else { '│' });
                *row += 1;
            }
            ir::MermaidSequenceItem::Note(note) => {
                let first = *columns.get(note.participants.first()?.as_str())?;
                let last = *columns.get(note.participants.last()?.as_str())?;
                draw_lifelines_active(canvas, *row, width, centers, columns, active);
                let note_width = display_width(&note.label.text);
                let (start, end) = match note.position {
                    ir::MermaidNotePosition::Right => (first + 1, width.saturating_sub(2)),
                    ir::MermaidNotePosition::Left => (2, first.saturating_sub(1)),
                    ir::MermaidNotePosition::Over if first == last => {
                        let start = first.saturating_sub(note_width / 2 + 2);
                        (start, start + note_width + 4)
                    }
                    ir::MermaidNotePosition::Over => (first.min(last), last.max(first)),
                };
                if end <= start || note_width + 4 > end - start {
                    return None;
                }
                let label_col = start + 2;
                canvas.blit(*row, start, "╭─ ");
                canvas.blit(*row, label_col + note_width, " ─╮");
                canvas.labels.push(canvas::MermaidCanvasLabel {
                    row: *row,
                    col: label_col,
                    text: note.label.text.clone(),
                    source: MermaidSourceSpan::new(note.label.start, note.label.end),
                });
                *row += 1;
            }
        }
    }
    Some(())
}

#[allow(clippy::too_many_arguments)]
fn draw_frame_row(
    canvas: &mut canvas::MermaidCanvas,
    row: usize,
    width: usize,
    centers: &[usize],
    left: char,
    right: char,
    prefix: &str,
    label: Option<&ir::MermaidLabel>,
) {
    draw_lifelines(canvas, row, width, centers);
    for col in 0..width {
        canvas.put(col, row, '─');
    }
    canvas.put(0, row, left);
    canvas.put(width - 1, row, right);
    if !prefix.is_empty() {
        canvas.blit(row, 2, prefix);
    }
    if let Some(label) = label {
        canvas.labels.push(canvas::MermaidCanvasLabel {
            row,
            col: 2 + display_width(prefix),
            text: label.text.clone(),
            source: MermaidSourceSpan::new(label.start, label.end),
        });
    }
}

fn draw_lifelines(canvas: &mut canvas::MermaidCanvas, row: usize, width: usize, centers: &[usize]) {
    canvas.ensure_row(row, width);
    for center in centers {
        canvas.put(*center, row, '│');
    }
}

fn deactivate_participant(active: &mut HashMap<String, usize>, participant: &str) -> Option<()> {
    let count = active.get(participant).copied()?;
    match count {
        0 => None,
        1 => {
            active.remove(participant);
            Some(())
        }
        _ => {
            active.insert(participant.to_string(), count - 1);
            Some(())
        }
    }
}

fn draw_lifelines_active(
    canvas: &mut canvas::MermaidCanvas,
    row: usize,
    width: usize,
    centers: &[usize],
    columns: &HashMap<&str, usize>,
    active: &HashMap<String, usize>,
) {
    draw_lifelines(canvas, row, width, centers);
    for (participant, center) in columns {
        if active.get(*participant).copied().unwrap_or(0) > 0 {
            canvas.put(*center, row, '█');
        }
    }
}

fn max_message_width(items: &[ir::MermaidSequenceItem]) -> usize {
    items
        .iter()
        .map(|item| match item {
            ir::MermaidSequenceItem::Message(message) => display_width(&message.label.text),
            ir::MermaidSequenceItem::Block(block) => block
                .branches
                .iter()
                .map(|branch| max_message_width(&branch.items))
                .max()
                .unwrap_or(0),
            ir::MermaidSequenceItem::Activation(_) => 0,
            ir::MermaidSequenceItem::Note(note) => display_width(&note.label.text) + 4,
        })
        .max()
        .unwrap_or(0)
}

fn max_block_width(items: &[ir::MermaidSequenceItem]) -> usize {
    items
        .iter()
        .map(|item| match item {
            ir::MermaidSequenceItem::Message(_) => 0,
            ir::MermaidSequenceItem::Block(block) => {
                let header =
                    display_width(block.kind.keyword()) + display_width(&block.label.text) + 1;
                let branches = block
                    .branches
                    .iter()
                    .map(|branch| {
                        let label = branch
                            .label
                            .as_ref()
                            .map_or(0, |label| display_width(&label.text));
                        (display_width(block.kind.branch_keyword()) + 1 + label)
                            .max(max_block_width(&branch.items))
                    })
                    .max()
                    .unwrap_or(0);
                header.max(branches)
            }
            ir::MermaidSequenceItem::Activation(activation) => {
                display_width("activate ") + display_width(&activation.label.text)
            }
            ir::MermaidSequenceItem::Note(note) => display_width(&note.label.text) + 8,
        })
        .max()
        .unwrap_or(0)
}

fn has_self_message(items: &[ir::MermaidSequenceItem]) -> bool {
    items.iter().any(|item| match item {
        ir::MermaidSequenceItem::Message(message) => message.from == message.to,
        ir::MermaidSequenceItem::Block(block) => block
            .branches
            .iter()
            .any(|branch| has_self_message(&branch.items)),
        ir::MermaidSequenceItem::Activation(_) | ir::MermaidSequenceItem::Note(_) => false,
    })
}

fn render_linear_items(
    sequence: &ir::MermaidSequence,
    items: &[ir::MermaidSequenceItem],
    indent: usize,
    lines: &mut Vec<Vec<MermaidRenderSpan>>,
    autonumber: bool,
    message_number: &mut usize,
) -> Option<()> {
    for item in items {
        match item {
            ir::MermaidSequenceItem::Message(message) => {
                let from = sequence.participants.get(&message.from)?;
                let to = sequence.participants.get(&message.to)?;
                let mut line = vec![MermaidRenderSpan::decoration(" ".repeat(indent))];
                if autonumber {
                    line.push(MermaidRenderSpan::decoration(format!(
                        "{} ",
                        *message_number
                    )));
                    *message_number += 1;
                }
                line.extend([
                    MermaidRenderSpan::source(
                        from.label.clone(),
                        MermaidSourceSpan::new(from.start, from.end),
                        from.atomic,
                    ),
                    MermaidRenderSpan::decoration(if message.dashed {
                        " ╌╌▶ "
                    } else {
                        " ──▶ "
                    }),
                    MermaidRenderSpan::source(
                        to.label.clone(),
                        MermaidSourceSpan::new(to.start, to.end),
                        to.atomic,
                    ),
                    MermaidRenderSpan::decoration("  "),
                    MermaidRenderSpan::source(
                        message.label.text.clone(),
                        MermaidSourceSpan::new(message.label.start, message.label.end),
                        message.label.atomic,
                    ),
                ]);
                if message.activate || message.deactivate {
                    line.push(MermaidRenderSpan::decoration(if message.activate {
                        "  (activate)"
                    } else {
                        "  (deactivate)"
                    }));
                }
                lines.push(line);
            }
            ir::MermaidSequenceItem::Block(block) => {
                lines.push(vec![
                    MermaidRenderSpan::decoration(" ".repeat(indent)),
                    MermaidRenderSpan::decoration(format!("{} ", block.kind.keyword())),
                    MermaidRenderSpan::source(
                        block.label.text.clone(),
                        MermaidSourceSpan::new(block.label.start, block.label.end),
                        block.label.atomic,
                    ),
                ]);
                for (index, branch) in block.branches.iter().enumerate() {
                    if index > 0 {
                        lines.push(vec![
                            MermaidRenderSpan::decoration(" ".repeat(indent)),
                            MermaidRenderSpan::decoration(block.kind.branch_keyword()),
                            MermaidRenderSpan::decoration(" "),
                        ]);
                        if let Some(label) = &branch.label
                            && let Some(line) = lines.last_mut()
                        {
                            line.push(MermaidRenderSpan::source(
                                label.text.clone(),
                                MermaidSourceSpan::new(label.start, label.end),
                                label.atomic,
                            ));
                        }
                    }
                    render_linear_items(
                        sequence,
                        &branch.items,
                        indent + 2,
                        lines,
                        autonumber,
                        message_number,
                    )?;
                }
                lines.push(vec![
                    MermaidRenderSpan::decoration(" ".repeat(indent)),
                    MermaidRenderSpan::decoration("end"),
                ]);
            }
            ir::MermaidSequenceItem::Activation(activation) => {
                lines.push(vec![
                    MermaidRenderSpan::decoration(" ".repeat(indent)),
                    MermaidRenderSpan::decoration(if activation.active {
                        "activate "
                    } else {
                        "deactivate "
                    }),
                    MermaidRenderSpan::source(
                        activation.label.text.clone(),
                        MermaidSourceSpan::new(activation.label.start, activation.label.end),
                        activation.label.atomic,
                    ),
                ]);
            }
            ir::MermaidSequenceItem::Note(note) => {
                let position = match note.position {
                    ir::MermaidNotePosition::Right => "right of ",
                    ir::MermaidNotePosition::Left => "left of ",
                    ir::MermaidNotePosition::Over => "over ",
                };
                lines.push(vec![
                    MermaidRenderSpan::decoration(" ".repeat(indent)),
                    MermaidRenderSpan::decoration(format!("note {position}")),
                    MermaidRenderSpan::decoration(note.participants.join(",")),
                    MermaidRenderSpan::decoration(": "),
                    MermaidRenderSpan::source(
                        note.label.text.clone(),
                        MermaidSourceSpan::new(note.label.start, note.label.end),
                        note.label.atomic,
                    ),
                ]);
            }
        }
    }
    Some(())
}

fn parse(source: &str) -> Option<ir::MermaidSequence> {
    if source.contains('\r') {
        return None;
    }
    let lines = source
        .lines()
        .enumerate()
        .scan(0usize, |offset, (_index, text)| {
            let line = ParsedLine {
                base: *offset + text.chars().count() - text.trim_start().chars().count(),
                text: text.trim(),
            };
            *offset += text.chars().count() + 1;
            Some(line)
        })
        .collect::<Vec<_>>();
    if lines.first()?.text != "sequenceDiagram" {
        return None;
    }
    let mut participants = HashMap::new();
    let mut cursor = 1;
    let mut autonumber = false;
    let items = parse_items(
        &lines,
        &mut cursor,
        &mut participants,
        true,
        false,
        &mut autonumber,
    )?;
    if cursor != lines.len() || participants.is_empty() || !contains_message(&items) {
        return None;
    }
    Some(ir::MermaidSequence {
        participants,
        items,
        autonumber,
    })
}

struct ParsedLine<'a> {
    base: usize,
    text: &'a str,
}

fn parse_items<'a>(
    lines: &[ParsedLine<'a>],
    cursor: &mut usize,
    participants: &mut HashMap<String, ir::MermaidNode>,
    allow_participant: bool,
    in_block: bool,
    autonumber: &mut bool,
) -> Option<Vec<ir::MermaidSequenceItem>> {
    let mut items = Vec::new();
    while *cursor < lines.len() {
        let line = &lines[*cursor];
        if line.text.is_empty() {
            *cursor += 1;
            continue;
        }
        if line.text == "end"
            || line.text == "else"
            || line.text.starts_with("else ")
            || line.text == "and"
            || line.text.starts_with("and ")
        {
            if in_block {
                break;
            }
            return None;
        }
        let (keyword, rest) = line
            .text
            .split_once(char::is_whitespace)
            .map_or((line.text, ""), |(keyword, rest)| (keyword, rest.trim()));
        if keyword == "autonumber" {
            if !allow_participant
                || in_block
                || !items.is_empty()
                || !rest.is_empty()
                || *autonumber
            {
                return None;
            }
            *autonumber = true;
            *cursor += 1;
            continue;
        }
        if let Some(kind) = match keyword {
            "loop" => Some(ir::MermaidBlockKind::Loop),
            "alt" => Some(ir::MermaidBlockKind::Alt),
            "opt" => Some(ir::MermaidBlockKind::Opt),
            "rect" => Some(ir::MermaidBlockKind::Rect),
            "par" => Some(ir::MermaidBlockKind::Par),
            _ => None,
        } {
            if rest.is_empty() {
                return None;
            }
            let label = parsed_label(line, keyword, rest)?;
            *cursor += 1;
            let branches = parse_block(lines, cursor, participants, kind, autonumber)?;
            items.push(ir::MermaidSequenceItem::Block(ir::MermaidBlock {
                kind,
                label,
                branches,
            }));
        } else if matches!(keyword, "participant" | "actor") && allow_participant {
            parse_participant(line, participants, keyword)?;
            *cursor += 1;
        } else if matches!(keyword, "activate" | "deactivate") {
            let participant = rest;
            if !valid_id(participant) || !participants.contains_key(participant) {
                return None;
            }
            let label = parsed_label(line, keyword, participant)?;
            items.push(ir::MermaidSequenceItem::Activation(ir::MermaidActivation {
                participant: participant.to_string(),
                label,
                active: keyword == "activate",
            }));
            *cursor += 1;
        } else if matches!(keyword, "Note" | "note") {
            let note = parse_note(line, participants)?;
            items.push(ir::MermaidSequenceItem::Note(note));
            *cursor += 1;
        } else if matches!(
            keyword,
            "participant"
                | "actor"
                | "loop"
                | "alt"
                | "opt"
                | "rect"
                | "par"
                | "else"
                | "and"
                | "end"
        ) {
            return None;
        } else {
            let message = parse_message(line, participants)?;
            items.push(ir::MermaidSequenceItem::Message(message));
            *cursor += 1;
        }
    }
    Some(items)
}

fn parse_block<'a>(
    lines: &[ParsedLine<'a>],
    cursor: &mut usize,
    participants: &mut HashMap<String, ir::MermaidNode>,
    kind: ir::MermaidBlockKind,
    autonumber: &mut bool,
) -> Option<Vec<ir::MermaidBranch>> {
    let mut branches = Vec::new();
    let mut label = None;
    let mut seen_branch = false;
    loop {
        let items = parse_items(lines, cursor, participants, false, true, autonumber)?;
        if items.is_empty() {
            return None;
        }
        branches.push(ir::MermaidBranch { label, items });
        if *cursor >= lines.len() {
            return None;
        }
        let line = &lines[*cursor];
        if line.text == "end" {
            *cursor += 1;
            return if (kind == ir::MermaidBlockKind::Alt || kind == ir::MermaidBlockKind::Par)
                || branches.len() == 1
            {
                Some(branches)
            } else {
                None
            };
        }
        let branch_keyword = kind.branch_keyword();
        if kind != ir::MermaidBlockKind::Alt && kind != ir::MermaidBlockKind::Par
            || kind == ir::MermaidBlockKind::Alt && seen_branch
            || !line.text.starts_with(branch_keyword)
        {
            return None;
        }
        seen_branch = true;
        let rest = line.text.strip_prefix(branch_keyword)?;
        if !rest.is_empty() && !rest.starts_with(char::is_whitespace) {
            return None;
        }
        let rest = rest.trim();
        label = if rest.is_empty() {
            None
        } else {
            Some(parsed_label(line, branch_keyword, rest)?)
        };
        *cursor += 1;
    }
}

fn parsed_label(line: &ParsedLine<'_>, prefix: &str, raw_text: &str) -> Option<ir::MermaidLabel> {
    let after_prefix = line.text.strip_prefix(prefix)?;
    let leading = after_prefix
        .chars()
        .take_while(|ch| ch.is_whitespace())
        .count();
    let start = line.base + prefix.chars().count() + leading;
    let raw_len = raw_text.chars().count();
    let (text, atomic) = parse_label(raw_text)?;
    Some(ir::MermaidLabel {
        text,
        start,
        end: start + raw_len,
        atomic,
    })
}

fn parse_participant(
    line: &ParsedLine<'_>,
    participants: &mut HashMap<String, ir::MermaidNode>,
    keyword: &str,
) -> Option<()> {
    let after_keyword = line.text.strip_prefix(keyword)?;
    if !after_keyword.starts_with(char::is_whitespace) {
        return None;
    }
    let declaration_start = line.base
        + keyword.chars().count()
        + after_keyword
            .chars()
            .take_while(|ch| ch.is_whitespace())
            .count();
    let rest = after_keyword.trim();
    if rest.is_empty() {
        return None;
    }

    let (id, label, label_start) = if let Some(separator) = rest.find(" as ") {
        let id = rest[..separator].trim();
        let label_segment = &rest[separator + " as ".len()..];
        let label = label_segment.trim();
        let label_leading = label_segment
            .chars()
            .take_while(|ch| ch.is_whitespace())
            .count();
        let label_start =
            declaration_start + rest[..separator + " as ".len()].chars().count() + label_leading;
        (id, label, label_start)
    } else {
        (rest, rest, declaration_start)
    };

    if !valid_id(id) || label.is_empty() || participants.contains_key(id) {
        return None;
    }
    let raw_label_len = label.chars().count();
    let (label, atomic) = parse_label(label)?;
    participants.insert(
        id.to_string(),
        ir::MermaidNode {
            label,
            start: label_start,
            end: label_start + raw_label_len,
            atomic,
        },
    );
    Some(())
}

fn parse_message(
    line: &ParsedLine<'_>,
    participants: &HashMap<String, ir::MermaidNode>,
) -> Option<ir::MermaidMessage> {
    let colon = line.text.find(':')?;
    let route = line.text[..colon].trim();
    let label = line.text[colon + 1..].trim();
    let (arrow_at, arrow) = ["-->>", "->>", "-->", "->"]
        .into_iter()
        .filter_map(|arrow| route.find(arrow).map(|index| (index, arrow)))
        .min_by_key(|(index, _)| *index)?;
    let from = route[..arrow_at].trim();
    let raw_to = route[arrow_at + arrow.len()..].trim();
    let (activate, deactivate, to) = if participants.contains_key(raw_to) {
        (false, false, raw_to)
    } else {
        match raw_to.as_bytes().first() {
            Some(b'+') => (true, false, &raw_to[1..]),
            Some(b'-') => (false, true, &raw_to[1..]),
            _ => (false, false, raw_to),
        }
    };
    if matches!(to.as_bytes().first(), Some(b'+' | b'-'))
        || !valid_id(from)
        || !valid_id(to)
        || label.is_empty()
        || !participants.contains_key(from)
        || !participants.contains_key(to)
    {
        return None;
    }
    let label_byte = colon + 1;
    let label_leading = line.text[label_byte..]
        .chars()
        .take_while(|ch| ch.is_whitespace())
        .count();
    let start = line.base + line.text[..label_byte].chars().count() + label_leading;
    let (label, atomic) = parse_label(label)?;
    Some(ir::MermaidMessage {
        from: from.to_string(),
        to: to.to_string(),
        label: ir::MermaidLabel {
            text: label,
            start,
            end: start + line.text[colon + 1..].trim().chars().count(),
            atomic,
        },
        dashed: arrow.starts_with("--"),
        activate,
        deactivate,
    })
}

fn parse_note(
    line: &ParsedLine<'_>,
    participants: &HashMap<String, ir::MermaidNode>,
) -> Option<ir::MermaidNote> {
    let rest = line
        .text
        .strip_prefix("Note ")
        .or_else(|| line.text.strip_prefix("note "))?;
    let (position, target) = if let Some(target) = rest.strip_prefix("right of ") {
        (ir::MermaidNotePosition::Right, target)
    } else if let Some(target) = rest.strip_prefix("left of ") {
        (ir::MermaidNotePosition::Left, target)
    } else if let Some(target) = rest.strip_prefix("over ") {
        (ir::MermaidNotePosition::Over, target)
    } else {
        return None;
    };
    let colon = target.find(':')?;
    let raw_participants = target[..colon].trim();
    let ids = raw_participants
        .split(',')
        .map(str::trim)
        .collect::<Vec<_>>();
    if ids.is_empty()
        || ids
            .iter()
            .any(|id| !valid_id(id) || !participants.contains_key(*id))
        || (position != ir::MermaidNotePosition::Over && ids.len() != 1)
        || (position == ir::MermaidNotePosition::Over && ids.len() > 2)
    {
        return None;
    }
    let raw_label = target[colon + 1..].trim();
    if raw_label.is_empty() {
        return None;
    }
    let label_leading = target[colon + 1..]
        .chars()
        .take_while(|ch| ch.is_whitespace())
        .count();
    let line_colon = line.text.find(':')?;
    let label_start = line.base + line.text[..line_colon + 1].chars().count() + label_leading;
    let raw_len = raw_label.chars().count();
    let (text, atomic) = parse_label(raw_label)?;
    Some(ir::MermaidNote {
        position,
        participants: ids.into_iter().map(str::to_string).collect(),
        label: ir::MermaidLabel {
            text,
            start: label_start,
            end: label_start + raw_len,
            atomic,
        },
    })
}

fn message_count(items: &[ir::MermaidSequenceItem]) -> usize {
    items
        .iter()
        .map(|item| match item {
            ir::MermaidSequenceItem::Message(_) => 1,
            ir::MermaidSequenceItem::Block(block) => block
                .branches
                .iter()
                .map(|branch| message_count(&branch.items))
                .sum(),
            ir::MermaidSequenceItem::Activation(_) | ir::MermaidSequenceItem::Note(_) => 0,
        })
        .sum()
}

fn contains_message(items: &[ir::MermaidSequenceItem]) -> bool {
    items.iter().any(|item| match item {
        ir::MermaidSequenceItem::Message(_) => true,
        ir::MermaidSequenceItem::Block(block) => block
            .branches
            .iter()
            .any(|branch| contains_message(&branch.items)),
        ir::MermaidSequenceItem::Activation(_) | ir::MermaidSequenceItem::Note(_) => false,
    })
}

fn valid_id(value: &str) -> bool {
    !value.is_empty()
        && value
            .chars()
            .all(|ch| ch.is_alphanumeric() || ch == '_' || ch == '-')
}

fn parse_label(label: &str) -> Option<(String, bool)> {
    super::render_math_label(label)
}
