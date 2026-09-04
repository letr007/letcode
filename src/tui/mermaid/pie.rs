//! Mermaid pie parsing and segmented proportion-bar rendering.

use std::collections::HashSet;

use crate::tui::measure::display_width;

use super::{
    MermaidRenderSpan, MermaidSourceSpan, pie_ir as ir, render_line_count_within_limits,
    source_within_limits,
};

const SEGMENT_GLYPHS: [char; 8] = ['█', '▓', '▒', '░', '▉', '▋', '▌', '▐'];
const MAX_BAR_WIDTH: usize = 64;

pub(super) fn render(source: &str, width: usize) -> Option<Vec<Vec<MermaidRenderSpan>>> {
    let diagram = parse(source)?;
    if diagram.slices.len() > SEGMENT_GLYPHS.len() || width < diagram.slices.len() {
        return None;
    }

    let max_label_width = diagram
        .slices
        .iter()
        .map(|slice| display_width(&slice.label.text))
        .max()
        .unwrap_or(0);
    let percentages = diagram
        .slices
        .iter()
        .map(|slice| format_percentage(slice.value, total(&diagram.slices)))
        .collect::<Vec<_>>();
    let legend_width = diagram
        .slices
        .iter()
        .zip(&percentages)
        .map(|(slice, percentage)| {
            2 + max_label_width
                + 2
                + display_width(percentage)
                + if diagram.show_data {
                    3 + display_width(&slice.raw_value.text)
                } else {
                    0
                }
        })
        .max()
        .unwrap_or(0);
    let title_width = diagram
        .title
        .as_ref()
        .map_or(0, |title| display_width(&title.text));
    if legend_width > width || title_width > width {
        return None;
    }

    let bar_width = width.min(MAX_BAR_WIDTH);
    let segment_widths = allocate_segments(&diagram.slices, bar_width)?;
    let mut lines = Vec::new();
    if let Some(title) = &diagram.title {
        lines.push(vec![span(title)]);
        lines.push(Vec::new());
    }
    lines.push(vec![MermaidRenderSpan::decoration(
        diagram
            .slices
            .iter()
            .enumerate()
            .zip(segment_widths)
            .map(|((index, _), segment_width)| {
                SEGMENT_GLYPHS[index].to_string().repeat(segment_width)
            })
            .collect::<String>(),
    )]);
    lines.push(Vec::new());

    for (index, (slice, percentage)) in diagram.slices.iter().zip(percentages).enumerate() {
        let mut line = vec![
            MermaidRenderSpan::decoration(format!("{} ", SEGMENT_GLYPHS[index])),
            span(&slice.label),
            MermaidRenderSpan::decoration(
                " ".repeat(max_label_width.saturating_sub(display_width(&slice.label.text)) + 2),
            ),
            MermaidRenderSpan::decoration(percentage),
        ];
        if diagram.show_data {
            line.push(MermaidRenderSpan::decoration(" · "));
            line.push(span(&slice.raw_value));
        }
        lines.push(line);
    }

    (render_line_count_within_limits(lines.len())
        && lines.iter().all(|line| {
            line.iter()
                .map(|part| display_width(&part.text))
                .sum::<usize>()
                <= width
        }))
    .then_some(lines)
}

fn total(slices: &[ir::Slice]) -> f64 {
    slices.iter().map(|slice| slice.value).sum()
}

fn allocate_segments(slices: &[ir::Slice], width: usize) -> Option<Vec<usize>> {
    if slices.is_empty() || width < slices.len() {
        return None;
    }
    let total = total(slices);
    if !total.is_finite() || total <= 0.0 {
        return None;
    }
    let exact = slices
        .iter()
        .map(|slice| slice.value / total * width as f64)
        .collect::<Vec<_>>();
    let mut allocated = exact
        .iter()
        .map(|value| (*value as usize).max(1))
        .collect::<Vec<_>>();
    while allocated.iter().sum::<usize>() > width {
        let index = allocated
            .iter()
            .enumerate()
            .filter(|(_, value)| **value > 1)
            .max_by(|(left, left_value), (right, right_value)| {
                (**left_value as f64 - exact[*left])
                    .partial_cmp(&(**right_value as f64 - exact[*right]))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })?
            .0;
        allocated[index] -= 1;
    }
    while allocated.iter().sum::<usize>() < width {
        let index = allocated
            .iter()
            .enumerate()
            .max_by(|(left, left_value), (right, right_value)| {
                (exact[*left] - **left_value as f64)
                    .partial_cmp(&(exact[*right] - **right_value as f64))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })?
            .0;
        allocated[index] += 1;
    }
    Some(allocated)
}

fn format_percentage(value: f64, total: f64) -> String {
    let percentage = value / total * 100.0;
    if (percentage - percentage.round()).abs() < 0.05 {
        format!("{:.0}%", percentage)
    } else {
        format!("{percentage:.1}%")
    }
}

fn span(label: &ir::Label) -> MermaidRenderSpan {
    MermaidRenderSpan::source(label.text.clone(), label.span, false)
}

fn parse(source: &str) -> Option<ir::Diagram> {
    if !source_within_limits(source) || source.contains('\r') {
        return None;
    }
    let mut lines = source.split('\n');
    let header = lines.next()?;
    let (show_data, mut title) = parse_header(header)?;
    let mut slices = Vec::new();
    let mut labels = HashSet::new();
    let mut offset = header.chars().count() + 1;
    for raw in lines {
        let line_len = raw.chars().count();
        if raw.contains('\t') || raw.chars().last().is_some_and(char::is_whitespace) {
            return None;
        }
        let leading = raw.chars().count() - raw.trim_start().chars().count();
        let base = offset + leading;
        let line = raw.trim();
        if line.is_empty() || line.starts_with("%%") {
            offset += line_len + 1;
            continue;
        }
        if let Some(value) = line.strip_prefix("title ") {
            if title.is_some() || !slices.is_empty() || value.is_empty() {
                return None;
            }
            title = Some(label(value, base + "title ".chars().count())?);
        } else {
            let slice = parse_slice(line, base)?;
            if !labels.insert(slice.label.text.clone()) {
                return None;
            }
            slices.push(slice);
        }
        offset += line_len + 1;
    }
    if slices.is_empty() {
        return None;
    }
    Some(ir::Diagram {
        title,
        show_data,
        slices,
    })
}

fn parse_header(header: &str) -> Option<(bool, Option<ir::Label>)> {
    let rest = header.strip_prefix("pie")?;
    if !rest.is_empty() && !rest.starts_with(char::is_whitespace) {
        return None;
    }
    let leading = rest.chars().take_while(|ch| ch.is_whitespace()).count();
    let mut rest = rest.trim_start();
    let mut consumed = "pie".chars().count() + leading;
    let show_data = if rest == "showData" || rest.starts_with("showData ") {
        rest = rest.strip_prefix("showData")?;
        let spacing = rest.chars().take_while(|ch| ch.is_whitespace()).count();
        rest = rest.trim_start();
        consumed += "showData".chars().count() + spacing;
        true
    } else {
        false
    };
    if rest.is_empty() {
        return Some((show_data, None));
    }
    let title_text = rest.strip_prefix("title ")?;
    if title_text.is_empty() {
        return None;
    }
    let start = consumed + "title ".chars().count();
    Some((show_data, Some(label(title_text, start)?)))
}

fn parse_slice(line: &str, base: usize) -> Option<ir::Slice> {
    let quoted = line.strip_prefix('"')?;
    let close = quoted.find('"')?;
    let label_text = &quoted[..close];
    if label_text.is_empty() || label_text.contains('"') {
        return None;
    }
    let after_quote = &quoted[close + 1..];
    let colon_byte = after_quote.find(':')?;
    if !after_quote[..colon_byte].trim().is_empty() {
        return None;
    }
    let value_segment = &after_quote[colon_byte + 1..];
    let value_text = value_segment.trim();
    if value_text.is_empty() || value_text.split_whitespace().count() != 1 {
        return None;
    }
    if !valid_decimal(value_text) {
        return None;
    }
    let value = value_text.parse::<f64>().ok()?;
    if !value.is_finite() || value <= 0.0 {
        return None;
    }
    let value_leading = value_segment.chars().count() - value_segment.trim_start().chars().count();
    let label_start = base + 1;
    let value_start = base
        + 1
        + quoted[..close + 1].chars().count()
        + after_quote[..colon_byte + 1].chars().count()
        + value_leading;
    Some(ir::Slice {
        label: label(label_text, label_start)?,
        value,
        raw_value: label(value_text, value_start)?,
    })
}

fn valid_decimal(value: &str) -> bool {
    if let Some((integer, fraction)) = value.split_once('.') {
        !integer.is_empty()
            && !fraction.is_empty()
            && !fraction.contains('.')
            && integer.chars().all(|ch| ch.is_ascii_digit())
            && fraction.chars().all(|ch| ch.is_ascii_digit())
    } else {
        !value.is_empty() && value.chars().all(|ch| ch.is_ascii_digit())
    }
}

fn label(text: &str, start: usize) -> Option<ir::Label> {
    if text.is_empty() || text.contains(['\n', '<', '>']) {
        return None;
    }
    Some(ir::Label {
        text: text.to_string(),
        span: MermaidSourceSpan::new(start, start + text.chars().count()),
    })
}
