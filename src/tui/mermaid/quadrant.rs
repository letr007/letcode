use crate::tui::measure::display_width;

use super::{
    MermaidRenderSpan, MermaidSourceSpan,
    canvas::{MermaidCanvas, MermaidCanvasLabel},
    render_line_count_within_limits, source_within_limits,
};

mod ir {
    pub(super) use super::super::quadrant_ir::*;
}

const MAX_LAYOUT_WIDTH: usize = 512;
const MIN_PLOT_WIDTH: usize = 7;
const MIN_PLOT_HEIGHT: usize = 9;

pub(super) fn render(source: &str, width: usize) -> Option<Vec<Vec<MermaidRenderSpan>>> {
    let diagram = parse(source)?;
    let graph_width = width.min(MAX_LAYOUT_WIDTH);
    if graph_width < MIN_PLOT_WIDTH {
        return None;
    }

    let mut canvas = MermaidCanvas {
        rows: Vec::new(),
        labels: Vec::new(),
    };
    let mut row = 0;
    if let Some(title) = &diagram.title {
        canvas.ensure_row(row, graph_width);
        place_centered(&mut canvas, row, graph_width, title)?;
        row += 1;
    }

    canvas.ensure_row(row, graph_width);
    place_inline(
        &mut canvas,
        row,
        graph_width,
        "y-axis ",
        &diagram.y_axis.left,
        " ↑ ",
        &diagram.y_axis.right,
    )?;
    row += 1;
    canvas.ensure_row(row, graph_width);
    place_inline(
        &mut canvas,
        row,
        graph_width,
        "x-axis ",
        &diagram.x_axis.left,
        " ─────→ ",
        &diagram.x_axis.right,
    )?;
    row += 1;

    canvas.ensure_row(row, graph_width);
    place_quadrants(
        &mut canvas,
        row,
        graph_width,
        2,
        1,
        &diagram.quadrants[1],
        &diagram.quadrants[0],
    )?;
    let plot_top = row + 1;
    let plot_height = MIN_PLOT_HEIGHT.saturating_add(diagram.points.len().min(256));
    let plot_bottom = plot_top.checked_add(plot_height.checked_sub(1)?)?;
    let axis_row = plot_top + plot_height / 2;
    let axis_col = graph_width / 2;
    let plot_min_col = 1;
    let plot_max_col = graph_width - 2;
    let plot_min_row = plot_top + 1;
    let plot_max_row = plot_bottom - 1;

    for plot_row in plot_top..=plot_bottom {
        canvas.ensure_row(plot_row, graph_width);
        canvas.put(axis_col, plot_row, '│');
    }
    canvas.blit(axis_row, 0, &"─".repeat(graph_width));
    canvas.put(axis_col, axis_row, '┼');

    let usable_width = plot_max_col - plot_min_col;
    let usable_height = plot_max_row - plot_min_row;
    let mut occupied = Vec::new();
    let mut markers = Vec::new();
    for point in &diagram.points {
        let col =
            plot_min_col + ((usable_width as f64 * point.x).round() as usize).min(usable_width);
        let row =
            plot_max_row - ((usable_height as f64 * point.y).round() as usize).min(usable_height);
        if occupied.iter().any(|&(left, top, right, bottom)| {
            left <= col && col <= right && top <= row && row <= bottom
        }) {
            return None;
        }
        canvas.put(col, row, '●');
        markers.push((col, row));

        let label_width = display_width(&point.label.text);
        if label_width == 0 {
            return None;
        }
        let candidates = [
            (1isize, 0isize),
            (-(label_width as isize), 0),
            (1, -1),
            (-(label_width as isize), -1),
            (1, 1),
            (-(label_width as isize), 1),
            (1, -2),
            (-(label_width as isize), -2),
            (1, 2),
            (-(label_width as isize), 2),
        ];
        let mut placed = None;
        for (offset_col, offset_row) in candidates {
            let left = col as isize + offset_col;
            let top = row as isize + offset_row;
            if left < plot_min_col as isize
                || top < plot_min_row as isize
                || left + label_width as isize > plot_max_col as isize
                || top > plot_max_row as isize
            {
                continue;
            }
            let left = left as usize;
            let top = top as usize;
            let right = left + label_width - 1;
            let bottom = top;
            if top <= axis_row && axis_row <= bottom
                || (left..=right).contains(&axis_col)
                || occupied
                    .iter()
                    .any(|&(old_left, old_top, old_right, old_bottom)| {
                        left <= old_right
                            && old_left <= right
                            && top <= old_bottom
                            && old_top <= bottom
                    })
                || markers.iter().any(|&(marker_col, marker_row)| {
                    (marker_col != col || marker_row != row)
                        && (left..=right).contains(&marker_col)
                        && (top..=bottom).contains(&marker_row)
                })
            {
                continue;
            }
            placed = Some((left, top, right, bottom));
            break;
        }
        let (left, top, right, bottom) = placed?;
        canvas.labels.push(MermaidCanvasLabel {
            row: top,
            col: left,
            text: point.label.text.clone(),
            source: point.label.span,
        });
        occupied.push((left, top, right, bottom));
    }

    let bottom_quadrant_row = plot_bottom.checked_add(1)?;
    canvas.ensure_row(bottom_quadrant_row, graph_width);
    place_quadrants(
        &mut canvas,
        bottom_quadrant_row,
        graph_width,
        3,
        4,
        &diagram.quadrants[2],
        &diagram.quadrants[3],
    )?;

    let lines = canvas.render();
    (render_line_count_within_limits(lines.len())
        && lines.iter().all(|line| {
            line.iter()
                .map(|span| display_width(&span.text))
                .sum::<usize>()
                <= width
        }))
    .then_some(lines)
}

fn place_centered(
    canvas: &mut MermaidCanvas,
    row: usize,
    width: usize,
    label: &ir::Label,
) -> Option<()> {
    let label_width = display_width(&label.text);
    if label_width > width {
        return None;
    }
    canvas.labels.push(MermaidCanvasLabel {
        row,
        col: (width - label_width) / 2,
        text: label.text.clone(),
        source: label.span,
    });
    Some(())
}

fn place_inline(
    canvas: &mut MermaidCanvas,
    row: usize,
    width: usize,
    prefix: &str,
    first: &ir::Label,
    connector: &str,
    second: &ir::Label,
) -> Option<()> {
    let first_width = display_width(&first.text);
    let second_width = display_width(&second.text);
    let first_col = display_width(prefix);
    let second_col = first_col
        .checked_add(first_width)?
        .checked_add(display_width(connector))?;
    if second_col.checked_add(second_width)? > width {
        return None;
    }
    canvas.blit(row, 0, prefix);
    canvas.blit(row, first_col + first_width, connector);
    canvas.labels.extend([
        MermaidCanvasLabel {
            row,
            col: first_col,
            text: first.text.clone(),
            source: first.span,
        },
        MermaidCanvasLabel {
            row,
            col: second_col,
            text: second.text.clone(),
            source: second.span,
        },
    ]);
    Some(())
}

fn place_quadrants(
    canvas: &mut MermaidCanvas,
    row: usize,
    width: usize,
    left_number: usize,
    right_number: usize,
    left: &ir::Label,
    right: &ir::Label,
) -> Option<()> {
    let left_prefix = format!("[Q{left_number}] ");
    let right_prefix = format!("[Q{right_number}] ");
    let left_width = display_width(&left_prefix) + display_width(&left.text);
    let right_width = display_width(&right_prefix) + display_width(&right.text);
    if left_width + right_width + 2 > width {
        return None;
    }
    let left_col = 1;
    let right_col = width - right_width - 1;
    canvas.blit(row, left_col, &left_prefix);
    canvas.blit(row, right_col, &right_prefix);
    canvas.labels.extend([
        MermaidCanvasLabel {
            row,
            col: left_col + display_width(&left_prefix),
            text: left.text.clone(),
            source: left.span,
        },
        MermaidCanvasLabel {
            row,
            col: right_col + display_width(&right_prefix),
            text: right.text.clone(),
            source: right.span,
        },
    ]);
    Some(())
}

fn parse(source: &str) -> Option<ir::Diagram> {
    if !source_within_limits(source) || source.contains('\r') {
        return None;
    }
    let mut lines = source.split('\n');
    if lines.next()? != "quadrantChart" {
        return None;
    }

    let mut title = None;
    let mut x_axis = None;
    let mut y_axis = None;
    let mut quadrants: [Option<ir::Label>; 4] = [None, None, None, None];
    let mut points = Vec::new();
    let mut offset = "quadrantChart".chars().count() + 1;
    let mut points_seen = false;
    for raw in lines {
        let line_len = raw.chars().count();
        if raw.contains('\t') || raw.chars().last().is_some_and(char::is_whitespace) {
            return None;
        }
        let leading = line_len - raw.trim_start().chars().count();
        let base = offset + leading;
        let line = raw.trim();
        if line.is_empty() || line.starts_with("%%") {
            offset += line_len + 1;
            continue;
        }
        if let Some(value) = line.strip_prefix("title ") {
            if title.is_some() || points_seen || value.trim().is_empty() {
                return None;
            }
            let leading = value.chars().count() - value.trim_start().chars().count();
            title = Some(label(
                value.trim(),
                base + "title ".chars().count() + leading,
            )?);
        } else if let Some(value) = line.strip_prefix("x-axis ") {
            if x_axis.is_some() || points_seen {
                return None;
            }
            x_axis = Some(parse_axis(value, base + "x-axis ".chars().count())?);
        } else if let Some(value) = line.strip_prefix("y-axis ") {
            if y_axis.is_some() || points_seen {
                return None;
            }
            y_axis = Some(parse_axis(value, base + "y-axis ".chars().count())?);
        } else if let Some(rest) = line.strip_prefix("quadrant-") {
            if points_seen {
                return None;
            }
            let (number, value) = rest.split_once(' ')?;
            let index = match number {
                "1" => 0,
                "2" => 1,
                "3" => 2,
                "4" => 3,
                _ => return None,
            };
            if quadrants[index].is_some() || value.trim().is_empty() {
                return None;
            }
            let value_start = base + "quadrant-".chars().count() + number.chars().count() + 1;
            let value_leading = value.chars().count() - value.trim_start().chars().count();
            quadrants[index] = Some(label(value.trim(), value_start + value_leading)?);
        } else {
            points_seen = true;
            points.push(parse_point(line, base)?);
        }
        offset += line_len + 1;
    }

    Some(ir::Diagram {
        title,
        x_axis: x_axis?,
        y_axis: y_axis?,
        quadrants: [
            quadrants[0].take()?,
            quadrants[1].take()?,
            quadrants[2].take()?,
            quadrants[3].take()?,
        ],
        points: (!points.is_empty()).then_some(points)?,
    })
}

fn parse_axis(value: &str, base: usize) -> Option<ir::Axis> {
    if value.matches("-->").count() != 1 {
        return None;
    }
    let (left, right) = value.split_once(" --> ")?;
    let left = left.trim();
    let right = right.trim();
    if left.is_empty() || right.is_empty() {
        return None;
    }
    let left_leading = value.chars().take_while(|ch| ch.is_whitespace()).count();
    let arrow_byte = value.find(" --> ")?;
    let right_byte = arrow_byte + " --> ".len();
    let right_leading = value[right_byte..]
        .chars()
        .take_while(|ch| ch.is_whitespace())
        .count();
    let right_start = value[..right_byte].chars().count() + right_leading;
    Some(ir::Axis {
        left: label(left, base + left_leading)?,
        right: label(right, base + right_start)?,
    })
}

fn parse_point(line: &str, base: usize) -> Option<ir::Point> {
    let (raw_label, raw_point) = line.split_once(": [")?;
    let point_text = raw_point.strip_suffix(']')?;
    if point_text.contains(['[', ']']) {
        return None;
    }
    let label_text = raw_label.trim();
    if label_text.is_empty() {
        return None;
    }
    let values = point_text.split(',').collect::<Vec<_>>();
    if values.len() != 2 {
        return None;
    }
    let x = coordinate(values[0])?;
    let y = coordinate(values[1])?;
    let label_start = base + raw_label.chars().count() - raw_label.trim_start().chars().count();
    Some(ir::Point {
        label: label(label_text, label_start)?,
        x,
        y,
    })
}

fn coordinate(value: &str) -> Option<f64> {
    let value = value.trim();
    if value.is_empty() {
        return None;
    }
    let value = value.parse::<f64>().ok()?;
    value
        .is_finite()
        .then_some(value)
        .filter(|value| (0.0..=1.0).contains(value))
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
