use crate::tui::measure::display_width;

use super::{
    MermaidRenderSpan, MermaidSourceSpan,
    canvas::{MermaidCanvas, MermaidCanvasLabel},
    render_line_count_within_limits, source_within_limits,
};

const AXIS_MARGIN: usize = 2;
const PERIOD_GAP: usize = 4;

#[derive(Debug)]
enum Item {
    Title(Label),
    Section(Label),
    Events { period: Label, events: Vec<Label> },
}

#[derive(Debug)]
struct Label {
    text: String,
    span: MermaidSourceSpan,
}

pub(super) fn render(source: &str, width: usize) -> Option<Vec<Vec<MermaidRenderSpan>>> {
    let items = parse(source)?;
    let mut lines = Vec::new();
    let mut periods = Vec::new();
    for item in items {
        match item {
            Item::Title(title) => {
                lines.push(vec![MermaidRenderSpan::decoration("╭─ "), span(&title)])
            }
            Item::Section(section) => {
                if flush_periods(&mut lines, &mut periods, width)? {
                    lines.push(Vec::new());
                }
                lines.push(vec![MermaidRenderSpan::decoration("├─ "), span(&section)])
            }
            Item::Events { period, events } => periods.push((period, events)),
        }
    }
    flush_periods(&mut lines, &mut periods, width)?;
    (!lines.is_empty() && render_line_count_within_limits(lines.len()) && fits(&lines, width))
        .then_some(lines)
}

fn flush_periods(
    lines: &mut Vec<Vec<MermaidRenderSpan>>,
    periods: &mut Vec<(Label, Vec<Label>)>,
    width: usize,
) -> Option<bool> {
    if periods.is_empty() {
        return Some(false);
    }
    lines.extend(render_periods(periods, width)?);
    periods.clear();
    Some(true)
}

fn render_periods(
    periods: &[(Label, Vec<Label>)],
    width: usize,
) -> Option<Vec<Vec<MermaidRenderSpan>>> {
    let card_widths = periods
        .iter()
        .map(|(_, events)| {
            events
                .iter()
                .map(|event| display_width(&event.text))
                .max()
                .unwrap_or(0)
                .max(1)
                .checked_add(4)
        })
        .collect::<Option<Vec<_>>>()?;
    let card_heights = periods
        .iter()
        .map(|(_, events)| events.len().checked_mul(2)?.checked_add(1))
        .collect::<Option<Vec<_>>>()?;
    let slot_widths = periods
        .iter()
        .zip(&card_widths)
        .map(|((period, _), card_width)| display_width(&period.text).max(*card_width))
        .collect::<Vec<_>>();
    let slots_width = slot_widths
        .iter()
        .try_fold(0usize, |total, slot| total.checked_add(*slot))?;
    let gaps_width = PERIOD_GAP.checked_mul(periods.len().saturating_sub(1))?;
    let graph_width = AXIS_MARGIN
        .checked_mul(2)?
        .checked_add(slots_width)?
        .checked_add(gaps_width)?;
    if graph_width == 0 || graph_width > width {
        return None;
    }
    let max_card_height = card_heights.iter().copied().max()?;
    let axis_row = max_card_height.checked_add(1)?;
    let period_row = axis_row.checked_add(1)?;
    let mut canvas = MermaidCanvas {
        rows: Vec::new(),
        labels: Vec::new(),
    };
    let mut anchors = Vec::with_capacity(periods.len());
    let mut slot_col = AXIS_MARGIN;
    for (index, (period, events)) in periods.iter().enumerate() {
        let slot_width = slot_widths[index];
        let card_width = card_widths[index];
        let card_height = card_heights[index];
        let inner_width = card_width.checked_sub(2)?;
        let card_col = slot_col.checked_add((slot_width - card_width) / 2)?;
        let card_row = max_card_height.checked_sub(card_height)?;
        let anchor = card_col.checked_add(card_width / 2)?;
        anchors.push(anchor);

        canvas.blit(
            card_row,
            card_col,
            &format!("┌{}┐", "─".repeat(inner_width)),
        );
        for (event_index, event) in events.iter().enumerate() {
            let row = card_row.checked_add(event_index.checked_mul(2)?.checked_add(1)?)?;
            canvas.blit(row, card_col, &format!("│{}│", " ".repeat(inner_width)));
            let event_width = display_width(&event.text);
            canvas.labels.push(MermaidCanvasLabel {
                row,
                col: card_col
                    .checked_add(1)?
                    .checked_add((inner_width - event_width) / 2)?,
                text: event.text.clone(),
                source: event.span,
            });
            if event_index + 1 < events.len() {
                canvas.blit(
                    row.checked_add(1)?,
                    card_col,
                    &format!("├{}┤", "─".repeat(inner_width)),
                );
            }
        }
        let bottom_row = card_row.checked_add(card_height.checked_sub(1)?)?;
        canvas.blit(
            bottom_row,
            card_col,
            &format!("└{}┘", "─".repeat(inner_width)),
        );
        canvas.put(anchor, bottom_row, '┬');
        for row in bottom_row.checked_add(1)?..axis_row {
            canvas.put(anchor, row, '│');
        }

        let period_width = display_width(&period.text);
        canvas.labels.push(MermaidCanvasLabel {
            row: period_row,
            col: slot_col.checked_add((slot_width - period_width) / 2)?,
            text: period.text.clone(),
            source: period.span,
        });
        slot_col = slot_col.checked_add(slot_width)?;
        if index + 1 < periods.len() {
            slot_col = slot_col.checked_add(PERIOD_GAP)?;
        }
    }
    canvas.blit(axis_row, 0, &"─".repeat(graph_width));
    for anchor in anchors {
        canvas.put(anchor, axis_row, '┴');
    }
    canvas.ensure_row(period_row, graph_width);
    Some(canvas.render())
}

fn parse(source: &str) -> Option<Vec<Item>> {
    if !source_within_limits(source) || source.contains('\r') {
        return None;
    }
    let mut items = Vec::new();
    let mut offset = source.lines().next()?.chars().count() + 1;
    let mut in_events = false;
    let mut title_seen = false;
    for line in source.split('\n').skip(1) {
        let line_len = line.chars().count();
        if line.contains('\t') || line.chars().last().is_some_and(char::is_whitespace) {
            return None;
        }
        let leading = line.chars().count() - line.trim_start().chars().count();
        let base = offset + leading;
        let line = line.trim_start();
        if line.is_empty() || line.starts_with("%%") {
            offset += line_len + 1;
            continue;
        }
        if let Some(value) = line.strip_prefix("title ") {
            if title_seen || !items.is_empty() || value.is_empty() {
                return None;
            }
            title_seen = true;
            items.push(Item::Title(label(value, base + 6)?));
        } else if let Some(value) = line.strip_prefix("section ") {
            if value.is_empty() || value.contains(':') {
                return None;
            }
            in_events = false;
            items.push(Item::Section(label(value, base + 8)?));
        } else if line.starts_with(':') {
            if !in_events {
                return None;
            }
            let mut segments = parse_colon_segments(line, base)?;
            if !segments.first().is_some_and(|segment| segment.is_none()) {
                return None;
            }
            let appended = segments.drain(1..).map(Option::unwrap).collect::<Vec<_>>();
            if appended.is_empty() {
                return None;
            }
            match items.last_mut()? {
                Item::Events { events, .. } => events.extend(appended),
                _ => return None,
            }
        } else {
            let mut segments = parse_colon_segments(line, base)?;
            let period = segments.first_mut()?.take()?;
            let parsed_events = segments.drain(1..).map(Option::unwrap).collect::<Vec<_>>();
            if parsed_events.is_empty() {
                return None;
            }
            in_events = true;
            items.push(Item::Events {
                period,
                events: parsed_events,
            });
        }
        offset += line_len + 1;
    }
    (!items.is_empty()).then_some(items)
}

fn label(value: &str, start: usize) -> Option<Label> {
    if value.contains(['\n', '<', '>']) || value.is_empty() {
        return None;
    }
    Some(Label {
        text: value.to_string(),
        span: MermaidSourceSpan::new(start, start + value.chars().count()),
    })
}

fn parse_colon_segments(line: &str, base: usize) -> Option<Vec<Option<Label>>> {
    if line.ends_with(':')
        && line[..line.len() - 1]
            .chars()
            .last()
            .is_some_and(char::is_whitespace)
    {
        return None;
    }
    let boundaries = line
        .char_indices()
        .filter_map(|(byte, character)| {
            (character == ':'
                && line[byte + 1..]
                    .chars()
                    .next()
                    .is_some_and(char::is_whitespace))
            .then_some(byte)
        })
        .collect::<Vec<_>>();
    if boundaries.is_empty() {
        return None;
    }
    let mut labels = Vec::new();
    let mut start_byte = 0;
    for end_byte in boundaries.into_iter().chain(std::iter::once(line.len())) {
        let segment = &line[start_byte..end_byte];
        let trimmed = segment.trim();
        if trimmed.is_empty() {
            labels.push(None);
        } else {
            let leading = segment.chars().count() - segment.trim_start().chars().count();
            let segment_start = base + line[..start_byte].chars().count() + leading;
            labels.push(Some(label(trimmed, segment_start)?));
        }
        start_byte = end_byte.saturating_add(1);
    }
    if labels.iter().skip(1).any(Option::is_none) {
        return None;
    }
    Some(labels)
}

fn span(label: &Label) -> MermaidRenderSpan {
    MermaidRenderSpan::source(label.text.clone(), label.span, false)
}

fn fits(lines: &[Vec<MermaidRenderSpan>], width: usize) -> bool {
    lines.iter().all(|line| {
        line.iter()
            .map(|span| display_width(&span.text))
            .sum::<usize>()
            <= width
    })
}
