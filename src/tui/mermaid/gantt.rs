use super::gantt_ir as ir;
use super::{
    MermaidRenderSpan, MermaidSourceSpan, canvas, render_line_count_within_limits,
    source_within_limits,
};
use crate::tui::measure::display_width;
use std::collections::{HashMap, HashSet};

const STATUSES: [&str; 4] = ["done", "active", "crit", "milestone"];
const MIN_TIMELINE_WIDTH: usize = 11;

pub(super) fn render(source: &str, width: usize) -> Option<Vec<Vec<MermaidRenderSpan>>> {
    let diagram = parse(source)?;
    let tasks = diagram
        .items
        .iter()
        .filter_map(|item| match item {
            ir::Item::Task(task) => Some(task),
            _ => None,
        })
        .collect::<Vec<_>>();
    validate_task_dependencies(&tasks)?;
    if let Some(positioned) = position_tasks(&tasks)
        && let Some(canvas) = layout(&diagram, width, &tasks, &positioned)
    {
        let lines = canvas.render();
        if render_line_count_within_limits(lines.len()) {
            return Some(lines);
        }
    }
    render_linear(&diagram, width)
}

fn render_linear(diagram: &ir::Diagram, width: usize) -> Option<Vec<Vec<MermaidRenderSpan>>> {
    let mut lines = Vec::new();
    for item in &diagram.items {
        match item {
            ir::Item::Config(config) => lines.push(vec![
                MermaidRenderSpan::decoration(format!("{} ", config.key)),
                span(&config.value),
            ]),
            ir::Item::Section(label) => {
                lines.push(vec![MermaidRenderSpan::decoration("section "), span(label)])
            }
            ir::Item::Task(task) => {
                let mut line = vec![span(&task.name)];
                if let Some(status) = &task.status {
                    line.extend([
                        MermaidRenderSpan::decoration(" ["),
                        span(status),
                        MermaidRenderSpan::decoration("]"),
                    ]);
                }
                if let Some(id) = &task.id {
                    line.extend([MermaidRenderSpan::decoration(" "), span(id)]);
                }
                line.extend([MermaidRenderSpan::decoration(" : "), span(&task.timing)]);
                lines.push(line);
            }
        }
    }
    (render_line_count_within_limits(lines.len())
        && lines
            .iter()
            .all(|line| line.iter().map(|s| display_width(&s.text)).sum::<usize>() <= width))
    .then_some(lines)
}

struct PositionedTask {
    start: i64,
    end: i64,
}

fn validate_task_dependencies(tasks: &[&ir::Task]) -> Option<()> {
    let mut ids = HashMap::new();
    for (index, task) in tasks.iter().enumerate() {
        if let Some(id) = &task.id
            && ids.insert(id.text.as_str(), index).is_some()
        {
            return None;
        }
    }
    let mut dependencies = vec![Vec::new(); tasks.len()];
    for (index, task) in tasks.iter().enumerate() {
        let (start, end) = task.timing.text.split_once(',')?;
        for token in [start.trim(), end.trim()] {
            let referenced = token
                .strip_prefix("after ")
                .or_else(|| token.strip_prefix("until "));
            if let Some(id) = referenced {
                dependencies[index].push(*ids.get(id.trim())?);
            }
        }
    }
    let mut state = vec![0u8; tasks.len()];
    fn visit(index: usize, dependencies: &[Vec<usize>], state: &mut [u8]) -> Option<()> {
        match state[index] {
            1 => return None,
            2 => return Some(()),
            _ => {}
        }
        state[index] = 1;
        for dependency in &dependencies[index] {
            visit(*dependency, dependencies, state)?;
        }
        state[index] = 2;
        Some(())
    }
    for index in 0..tasks.len() {
        visit(index, &dependencies, &mut state)?;
    }
    Some(())
}

fn position_tasks(tasks: &[&ir::Task]) -> Option<Vec<PositionedTask>> {
    let mut ids = HashMap::new();
    for (index, task) in tasks.iter().enumerate() {
        if let Some(id) = &task.id
            && ids.insert(id.text.as_str(), index).is_some()
        {
            return None;
        }
    }
    let mut resolved = vec![None; tasks.len()];
    for index in 0..tasks.len() {
        resolve_task(index, tasks, &ids, &mut resolved, &mut HashSet::new())?;
    }
    tasks
        .iter()
        .enumerate()
        .map(|(index, _task)| {
            let (start, end) = resolved[index]?;
            Some(PositionedTask { start, end })
        })
        .collect()
}

fn layout(
    diagram: &ir::Diagram,
    width: usize,
    tasks: &[&ir::Task],
    positioned: &[PositionedTask],
) -> Option<canvas::MermaidCanvas> {
    if diagram.items.iter().any(|item| {
        matches!(item, ir::Item::Config(config) if config.key == "axisFormat")
            || matches!(item, ir::Item::Config(config) if config.key == "dateFormat" && config.value.text != "YYYY-MM-DD")
    }) {
        return None;
    }

    if tasks.is_empty() || tasks.len() != positioned.len() {
        return None;
    }

    let min_day = positioned.iter().map(|task| task.start).min()?;
    let max_day = positioned.iter().map(|task| task.end).max()?;
    let timeline_days = usize::try_from(max_day.checked_sub(min_day)?).ok()?;
    if timeline_days == 0 {
        return None;
    }

    let header_width = diagram
        .items
        .iter()
        .filter_map(|item| match item {
            ir::Item::Config(config) if config.key == "title" => {
                Some(6 + display_width(&config.value.text))
            }
            ir::Item::Section(label) => Some(8 + display_width(&label.text)),
            _ => None,
        })
        .max()
        .unwrap_or(0);
    if header_width > width {
        return None;
    }
    let left_width = tasks
        .iter()
        .map(|task| task_text_width(task))
        .max()
        .unwrap_or(0);
    let timeline_start = left_width.checked_add(3)?;
    if timeline_start.checked_add(MIN_TIMELINE_WIDTH)? > width {
        return None;
    }
    let timeline_width = width.checked_sub(timeline_start)?;
    let total_width = timeline_start.checked_add(timeline_width)?;

    let mut result = canvas::MermaidCanvas {
        rows: Vec::new(),
        labels: Vec::new(),
    };
    let mut row = 0;
    for item in &diagram.items {
        match item {
            ir::Item::Config(config) if config.key == "title" => {
                result.ensure_row(row, total_width);
                result.blit(row, 0, "title ");
                result.labels.push(canvas::MermaidCanvasLabel {
                    row,
                    col: 6,
                    text: config.value.text.clone(),
                    source: config.value.span,
                });
                row += 1;
            }
            ir::Item::Section(label) => {
                result.ensure_row(row, total_width);
                result.blit(row, 0, "section ");
                result.labels.push(canvas::MermaidCanvasLabel {
                    row,
                    col: 8,
                    text: label.text.clone(),
                    source: label.span,
                });
                row += 1;
            }
            ir::Item::Task(task) => {
                let index = tasks
                    .iter()
                    .position(|candidate| std::ptr::eq(*candidate, task))?;
                let task_position = &positioned[index];
                result.ensure_row(row, total_width);
                draw_task_text(&mut result, row, task);
                let start = timeline_start
                    + timeline_boundary_col(
                        task_position.start,
                        min_day,
                        timeline_days,
                        timeline_width,
                    )?;
                let end = timeline_start
                    + timeline_boundary_col(
                        task_position.end,
                        min_day,
                        timeline_days,
                        timeline_width,
                    )?;
                let marker = status_marker(task.status.as_ref().map(|status| status.text.as_str()));
                for col in start..end.max(start + 1).min(total_width) {
                    result.put(col, row, marker);
                }
                row += 1;
            }
            _ => {}
        }
    }

    result.ensure_row(row, total_width);
    let axis = format_axis(min_day, max_day, timeline_width);
    result.blit(row, timeline_start, &axis);
    Some(result)
}

fn resolve_task(
    index: usize,
    tasks: &[&ir::Task],
    ids: &HashMap<&str, usize>,
    resolved: &mut [Option<(i64, i64)>],
    visiting: &mut HashSet<usize>,
) -> Option<(i64, i64)> {
    if let Some(value) = resolved[index] {
        return Some(value);
    }
    if !visiting.insert(index) {
        return None;
    }
    let (start_token, end_token) = tasks[index].timing.text.split_once(",")?;
    let start_token = start_token.trim();
    let end_token = end_token.trim();
    let start = if let Some(date) = parse_date(start_token) {
        date
    } else if let Some(id) = start_token.strip_prefix("after ") {
        let predecessor = *ids.get(id.trim())?;
        resolve_task(predecessor, tasks, ids, resolved, visiting)?.1
    } else {
        return None;
    };
    let mut end = if let Some(date) = parse_date(end_token) {
        date
    } else if let Some(id) = end_token.strip_prefix("until ") {
        let predecessor = *ids.get(id.trim())?;
        resolve_task(predecessor, tasks, ids, resolved, visiting)?.1
    } else if let Some(days) = duration_days(end_token) {
        start.checked_add(days)?
    } else {
        return None;
    };
    visiting.remove(&index);
    if end < start {
        return None;
    }
    if end == start {
        end = start + 1;
    }
    resolved[index] = Some((start, end));
    Some((start, end))
}

fn task_text_width(task: &ir::Task) -> usize {
    display_width(&task.name.text)
}

fn timeline_boundary_col(
    day: i64,
    min_day: i64,
    timeline_days: usize,
    timeline_width: usize,
) -> Option<usize> {
    let offset = usize::try_from(day.checked_sub(min_day)?).ok()?;
    offset
        .checked_mul(timeline_width)?
        .checked_div(timeline_days)
}

fn draw_task_text(canvas: &mut canvas::MermaidCanvas, row: usize, task: &ir::Task) {
    canvas.labels.push(canvas::MermaidCanvasLabel {
        row,
        col: 0,
        text: task.name.text.clone(),
        source: task.name.span,
    });
}

fn status_marker(status: Option<&str>) -> char {
    match status {
        Some("done") => '=',
        Some("active") => '#',
        Some("crit") => '!',
        Some("milestone") => 'M',
        _ => '-',
    }
}

fn format_axis(start: i64, end: i64, width: usize) -> String {
    let start_text = short_date(start);
    let end_text = short_date(end);
    let mut axis = vec![' '; width];
    for (index, ch) in start_text.chars().enumerate().take(width) {
        axis[index] = ch;
    }
    let end_col = width.saturating_sub(end_text.chars().count());
    if end_col >= start_text.chars().count() {
        for (index, ch) in end_text.chars().enumerate() {
            if end_col + index < width {
                axis[end_col + index] = ch;
            }
        }
    }
    axis.into_iter().collect()
}

fn short_date(days: i64) -> String {
    let (_, month, day) = civil_from_days(days);
    format!("{month:02}-{day:02}")
}

fn parse_date(value: &str) -> Option<i64> {
    if !valid_date(value) {
        return None;
    }
    let mut parts = value.split('-');
    let year = parts.next()?.parse::<i64>().ok()?;
    let month = parts.next()?.parse::<i64>().ok()?;
    let day = parts.next()?.parse::<i64>().ok()?;
    let days = days_from_civil(year, month, day);
    (civil_from_days(days) == (year, month, day)).then_some(days)
}

fn duration_days(value: &str) -> Option<i64> {
    let digit_count = value.chars().take_while(|c| c.is_ascii_digit()).count();
    if digit_count == 0 || !matches!(&value[digit_count..], "d" | "w") {
        return None;
    }
    let amount = value[..digit_count].parse::<i64>().ok()?;
    amount.checked_mul(if &value[digit_count..] == "w" { 7 } else { 1 })
}

fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let year = year - i64::from(month <= 2);
    let era = (if year >= 0 { year } else { year - 399 }) / 400;
    let year_of_era = year - era * 400;
    let month_prime = month + if month > 2 { -3 } else { 9 };
    let day_of_year = (153 * month_prime + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146_097 + day_of_era - 719_468
}

fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let days = days + 719_468;
    let era = (if days >= 0 { days } else { days - 146_096 }) / 146_097;
    let day_of_era = days - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_prime = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_prime + 2) / 5 + 1;
    let month = month_prime + if month_prime < 10 { 3 } else { -9 };
    (year + i64::from(month <= 2), month, day)
}

fn span(label: &ir::Label) -> MermaidRenderSpan {
    MermaidRenderSpan::source(label.text.clone(), label.span, false)
}

fn parse(source: &str) -> Option<ir::Diagram> {
    if !source_within_limits(source) || source.contains('\r') {
        return None;
    }
    let mut lines = source.lines();
    if lines.next()? != "gantt" {
        return None;
    }

    let mut items = Vec::new();
    let mut offset = "gantt".chars().count() + 1;
    for raw in source.lines().skip(1) {
        let trimmed = raw.trim();
        let leading = raw.chars().count() - raw.trim_start().chars().count();
        let base = offset + leading;
        if trimmed.is_empty() || trimmed.starts_with("%%") {
            offset += raw.chars().count() + 1;
            continue;
        }

        let (key, value) = split_directive(trimmed);
        match key {
            "title" | "dateFormat" | "axisFormat" => {
                if value.is_empty() {
                    return None;
                }
                let at = find_char(trimmed, value)?;
                items.push(ir::Item::Config(ir::Config {
                    key: match key {
                        "title" => "title",
                        "dateFormat" => "dateFormat",
                        "axisFormat" => "axisFormat",
                        _ => unreachable!(),
                    },
                    value: ir::Label {
                        text: value.to_string(),
                        span: MermaidSourceSpan::new(base + at, base + at + value.chars().count()),
                    },
                }));
            }
            "section" => {
                if value.is_empty() {
                    return None;
                }
                let at = find_char(trimmed, value)?;
                items.push(ir::Item::Section(ir::Label {
                    text: value.to_string(),
                    span: MermaidSourceSpan::new(base + at, base + at + value.chars().count()),
                }));
            }
            "excludes" | "todayMarker" | "weekday" => return None,
            _ => items.push(ir::Item::Task(parse_task(trimmed, base)?)),
        }
        offset += raw.chars().count() + 1;
    }
    (!items.is_empty()).then_some(ir::Diagram { items })
}

fn parse_task(line: &str, base: usize) -> Option<ir::Task> {
    let colon_byte = line.find(':')?;
    let name = line[..colon_byte].trim();
    let rest = line[colon_byte + 1..].trim();
    if name.is_empty() || rest.is_empty() {
        return None;
    }

    let fields = rest.split(',').map(str::trim).collect::<Vec<_>>();
    if fields.iter().any(|field| field.is_empty()) || !(2..=4).contains(&fields.len()) {
        return None;
    }
    let mut index = 0;
    let field_start_byte = colon_byte + 1;
    let status = if STATUSES.contains(&fields[0]) {
        let label = label_in_from(line, fields[0], base, field_start_byte)?;
        index += 1;
        Some(label)
    } else {
        None
    };
    let remaining = fields.len() - index;
    let id = if remaining == 3 {
        let text = fields[index];
        if !valid_id(text) {
            return None;
        }
        index += 1;
        Some(label_in_from(line, text, base, field_start_byte)?)
    } else {
        None
    };
    if fields.len() - index != 2
        || !valid_start(fields[index])
        || !valid_end_or_duration(fields[index + 1])
    {
        return None;
    }

    let timing_start_byte = line[colon_byte + 1..].find(fields[index])? + colon_byte + 1;
    let timing_end_byte = line.rfind(fields[index + 1])? + fields[index + 1].len();
    let timing_text = line[timing_start_byte..timing_end_byte].trim();
    let timing_at = char_index(line, timing_start_byte)
        + line[timing_start_byte..timing_end_byte]
            .chars()
            .take_while(|c| c.is_whitespace())
            .count();
    let name_at = find_char(line, name)?;
    Some(ir::Task {
        name: ir::Label {
            text: name.to_string(),
            span: MermaidSourceSpan::new(base + name_at, base + name_at + name.chars().count()),
        },
        status,
        id,
        timing: ir::Label {
            text: timing_text.to_string(),
            span: MermaidSourceSpan::new(
                base + timing_at,
                base + timing_at + timing_text.chars().count(),
            ),
        },
    })
}

fn split_directive(line: &str) -> (&str, &str) {
    let split = line.find(char::is_whitespace).unwrap_or(line.len());
    (&line[..split], line[split..].trim())
}

fn label_in_from(line: &str, text: &str, base: usize, start_byte: usize) -> Option<ir::Label> {
    let byte = start_byte + line[start_byte..].find(text)?;
    let at = char_index(line, byte);
    Some(ir::Label {
        text: text.to_string(),
        span: MermaidSourceSpan::new(base + at, base + at + text.chars().count()),
    })
}

fn valid_id(value: &str) -> bool {
    !value.is_empty()
        && value
            .chars()
            .all(|c| c.is_alphanumeric() || c == '_' || c == '-')
}

fn valid_start(value: &str) -> bool {
    valid_date(value) || value.strip_prefix("after ").is_some_and(valid_id)
}

fn valid_end_or_duration(value: &str) -> bool {
    valid_date(value) || value.strip_prefix("until ").is_some_and(valid_id) || is_duration(value)
}

fn valid_date(value: &str) -> bool {
    let parts = value.split('-').collect::<Vec<_>>();
    parts.len() == 3
        && parts[0].len() == 4
        && parts[1].len() == 2
        && parts[2].len() == 2
        && parts
            .iter()
            .all(|part| part.chars().all(|c| c.is_ascii_digit()))
}

fn is_duration(value: &str) -> bool {
    let digit_count = value.chars().take_while(|c| c.is_ascii_digit()).count();
    digit_count > 0 && matches!(&value[digit_count..], "ms" | "s" | "m" | "h" | "d" | "w")
}

fn find_char(haystack: &str, needle: &str) -> Option<usize> {
    haystack.find(needle).map(|byte| char_index(haystack, byte))
}

fn char_index(value: &str, byte: usize) -> usize {
    value[..byte].chars().count()
}
