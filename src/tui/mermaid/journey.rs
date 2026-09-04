//! Mermaid journey parsing and terminal rendering.

use crate::tui::measure::display_width;

use super::{
    MermaidRenderSpan, MermaidSourceSpan, journey_ir as ir, render_line_count_within_limits,
    source_within_limits,
};

pub(super) fn render(source: &str, width: usize) -> Option<Vec<Vec<MermaidRenderSpan>>> {
    let diagram = parse(source)?;
    let mut lines = Vec::new();
    if let Some(title) = &diagram.title {
        lines.push(vec![
            MermaidRenderSpan::decoration("╭─ "),
            span(title),
            MermaidRenderSpan::decoration(" ─╮"),
        ]);
    }

    for section in &diagram.sections {
        let task_widths = section.tasks.iter().map(task_width).collect::<Vec<_>>();
        let inline_width = task_widths
            .iter()
            .try_fold(0usize, |total, task| total.checked_add(*task))?
            .checked_add(section.tasks.len().saturating_sub(1).checked_mul(3)?)?
            .checked_add(3)?;
        let section_width = (display_width("╭─ ") + display_width(&section.label.text) + 2)
            .max(
                task_widths
                    .iter()
                    .copied()
                    .max()
                    .unwrap_or(0)
                    .checked_add(3)?,
            )
            .max((inline_width <= width).then_some(inline_width).unwrap_or(0));
        if section_width > width {
            return None;
        }
        lines.push(section_header(section, section_width));
        if inline_width <= width {
            lines.extend(inline_tasks(section, &task_widths, section_width));
        } else {
            for (index, task) in section.tasks.iter().enumerate() {
                lines.push(box_line(section_width, task_widths[index], '┌', '┐', '─'));
                lines.push(task_line(task, section_width));
                lines.push(box_line(section_width, task_widths[index], '└', '┘', '─'));
                if index + 1 < section.tasks.len() {
                    lines.push(connector_line(section_width, task_widths[index]));
                }
            }
        }
        lines.push(section_footer(section_width));
    }

    if lines.is_empty()
        || !render_line_count_within_limits(lines.len())
        || lines.iter().any(|line| {
            line.iter()
                .map(|part| display_width(&part.text))
                .sum::<usize>()
                > width
        })
    {
        return None;
    }
    Some(lines)
}

fn section_header(section: &ir::Section, width: usize) -> Vec<MermaidRenderSpan> {
    let mut line = vec![MermaidRenderSpan::decoration("╭─ "), span(&section.label)];
    let used = line
        .iter()
        .map(|part| display_width(&part.text))
        .sum::<usize>();
    line.push(MermaidRenderSpan::decoration(format!(
        "{}╮",
        "─".repeat(width.saturating_sub(used + 1))
    )));
    line
}

fn section_footer(width: usize) -> Vec<MermaidRenderSpan> {
    vec![MermaidRenderSpan::decoration(format!(
        "╰{}╯",
        "─".repeat(width.saturating_sub(2))
    ))]
}

fn task_width(task: &ir::Task) -> usize {
    display_width(&task.label.text)
        .checked_add(participant_width(&task.participants))
        .and_then(|width| width.checked_add(11))
        .unwrap_or(usize::MAX)
}

fn score_marks(score: &str) -> String {
    let score = score.parse::<usize>().unwrap_or(0);
    format!("{}{}", "●".repeat(score), "○".repeat(5 - score))
}

fn inline_tasks(
    section: &ir::Section,
    widths: &[usize],
    width: usize,
) -> Vec<Vec<MermaidRenderSpan>> {
    let mut top = vec![MermaidRenderSpan::decoration("│ ")];
    let mut middle = vec![MermaidRenderSpan::decoration("│ ")];
    let mut bottom = vec![MermaidRenderSpan::decoration("│ ")];
    for (index, task) in section.tasks.iter().enumerate() {
        if index > 0 {
            top.push(MermaidRenderSpan::decoration("   "));
            middle.push(MermaidRenderSpan::decoration("─▶─"));
            bottom.push(MermaidRenderSpan::decoration("   "));
        }
        let inner = widths[index].saturating_sub(2);
        top.push(MermaidRenderSpan::decoration(format!(
            "┌{}┐",
            "─".repeat(inner)
        )));
        let mut content = vec![MermaidRenderSpan::decoration("│ ")];
        content.push(span(&task.label));
        content.push(MermaidRenderSpan::decoration(" "));
        content.push(score_span(&task.score));
        content.push(MermaidRenderSpan::decoration(" "));
        for (participant_index, participant) in task.participants.iter().enumerate() {
            if participant_index > 0 {
                content.push(MermaidRenderSpan::decoration(", "));
            }
            content.push(span(participant));
        }
        content.push(MermaidRenderSpan::decoration(" │"));
        middle.extend(content);
        bottom.push(MermaidRenderSpan::decoration(format!(
            "└{}┘",
            "─".repeat(inner)
        )));
    }
    let used = middle
        .iter()
        .map(|part| display_width(&part.text))
        .sum::<usize>();
    if used < width {
        middle.push(MermaidRenderSpan::decoration(" ".repeat(width - used - 1)));
    }
    for line in [&mut top, &mut bottom] {
        let used = line
            .iter()
            .map(|part| display_width(&part.text))
            .sum::<usize>();
        if used < width {
            line.push(MermaidRenderSpan::decoration(" ".repeat(width - used - 1)));
        }
    }
    top.push(MermaidRenderSpan::decoration("│"));
    middle.push(MermaidRenderSpan::decoration("│"));
    bottom.push(MermaidRenderSpan::decoration("│"));
    vec![top, middle, bottom]
}

fn box_line(
    width: usize,
    task_width: usize,
    left: char,
    right: char,
    fill: char,
) -> Vec<MermaidRenderSpan> {
    let mut text = format!(
        "│ {}{}{}",
        left,
        fill.to_string().repeat(task_width - 2),
        right
    );
    let used = display_width(&text);
    if used < width {
        text.push_str(&" ".repeat(width - used - 1));
    }
    text.push('│');
    vec![MermaidRenderSpan::decoration(text)]
}

fn task_line(task: &ir::Task, width: usize) -> Vec<MermaidRenderSpan> {
    let mut line = vec![MermaidRenderSpan::decoration("│ │ "), span(&task.label)];
    line.push(MermaidRenderSpan::decoration(" "));
    line.push(score_span(&task.score));
    line.push(MermaidRenderSpan::decoration(" "));
    for (index, participant) in task.participants.iter().enumerate() {
        if index > 0 {
            line.push(MermaidRenderSpan::decoration(", "));
        }
        line.push(span(participant));
    }
    line.push(MermaidRenderSpan::decoration(" │"));
    let used = line
        .iter()
        .map(|part| display_width(&part.text))
        .sum::<usize>();
    if used >= width {
        return line;
    }
    line.push(MermaidRenderSpan::decoration(" ".repeat(width - used - 1)));
    line.push(MermaidRenderSpan::decoration("│"));
    line
}

fn connector_line(width: usize, task_width: usize) -> Vec<MermaidRenderSpan> {
    let center = task_width / 2 + 2;
    let mut text = String::new();
    text.push('│');
    text.push_str(&" ".repeat(center.saturating_sub(1)));
    text.push('▼');
    let used = display_width(&text);
    if used + 1 < width {
        text.push_str(&" ".repeat(width - used - 1));
    }
    text.push('│');
    vec![MermaidRenderSpan::decoration(text)]
}

fn score_span(score: &ir::Label) -> MermaidRenderSpan {
    MermaidRenderSpan::source(score_marks(&score.text), score.span, true)
}

fn participant_width(participants: &[ir::Label]) -> usize {
    participants
        .iter()
        .map(|participant| display_width(&participant.text))
        .sum::<usize>()
        + 2 * participants.len().saturating_sub(1)
}

fn span(label: &ir::Label) -> MermaidRenderSpan {
    MermaidRenderSpan::source(label.text.clone(), label.span, false)
}

fn parse(source: &str) -> Option<ir::Diagram> {
    if !source_within_limits(source) || source.contains('\r') {
        return None;
    }
    let mut lines = source.split('\n');
    if lines.next()? != "journey" {
        return None;
    }

    let mut title = None;
    let mut sections = Vec::new();
    let mut offset = "journey".chars().count() + 1;
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
        if let Some(value) = trimmed.strip_prefix("title ") {
            if title.is_some() || !sections.is_empty() || value.is_empty() {
                return None;
            }
            let start = base + 6;
            title = Some(label(value, start)?);
        } else if let Some(value) = trimmed.strip_prefix("section ") {
            if value.is_empty() || value.contains(':') {
                return None;
            }
            let start = base + 8;
            sections.push(ir::Section {
                label: label(value, start)?,
                tasks: Vec::new(),
            });
        } else {
            let section = sections.last_mut()?;
            section.tasks.push(parse_task(trimmed, base)?);
        }
        offset += line_len + 1;
    }
    if sections.is_empty() || sections.iter().any(|section| section.tasks.is_empty()) {
        return None;
    }
    Some(ir::Diagram { title, sections })
}

fn parse_task(line: &str, base: usize) -> Option<ir::Task> {
    let parts = line.split(':').collect::<Vec<_>>();
    if parts.len() != 3 {
        return None;
    }
    let task_text = parts[0].trim();
    let score_text = parts[1].trim();
    let participants_text = parts[2].trim();
    if task_text.is_empty() || participants_text.is_empty() {
        return None;
    }
    let score = score_text.parse::<u8>().ok()?;
    if !(1..=5).contains(&score)
        || score_text.chars().count() != 1
        || score_text.chars().any(|ch| !ch.is_ascii_digit())
    {
        return None;
    }

    let colons = line
        .match_indices(':')
        .map(|(at, _)| at)
        .collect::<Vec<_>>();
    let task_segment = &line[..colons[0]];
    let score_segment = &line[colons[0] + 1..colons[1]];
    let participants_segment = &line[colons[1] + 1..];
    let task_at = task_segment.chars().count() - task_segment.trim_start().chars().count();
    let score_at = line[..colons[0]].chars().count() + 1 + score_segment.chars().count()
        - score_segment.trim_start().chars().count();
    let participants_at =
        line[..colons[1]].chars().count() + 1 + participants_segment.chars().count()
            - participants_segment.trim_start().chars().count();
    let participants = participants_text
        .split(',')
        .scan(0usize, |cursor, value| {
            let trimmed = value.trim();
            let leading = value.chars().count() - value.trim_start().chars().count();
            let at = base + participants_at + *cursor + leading;
            *cursor += value.chars().count() + 1;
            Some(label(trimmed, at))
        })
        .collect::<Option<Vec<_>>>()?;
    if participants.is_empty() {
        return None;
    }
    Some(ir::Task {
        label: label(task_text, base + task_at)?,
        score: label(score_text, base + score_at)?,
        participants,
    })
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
