//! Shell/command tool card output rendering.

use ratatui::style::{Modifier, Style};

use super::semantic_spans::*;
use crate::tui::{
    measure::wrap_text_to_width,
    theme::Theme,
    timeline::ToolView,
    transcript_render::{Break, SemanticLine, SemanticSpan},
};

pub(super) fn render_shell_output_lines(
    tool: &ToolView,
    theme: Theme,
    width: usize,
    expanded_output: bool,
) -> Vec<SemanticLine<Style>> {
    let Some(data) = tool_output_data(tool) else {
        return Vec::new();
    };

    if let Some(error) = data.get("error").and_then(serde_json::Value::as_str) {
        return render_output_section(
            "error",
            error,
            theme.error_style().bg(theme.root_bg),
            theme,
            width,
            expanded_output,
        );
    }

    let stdout = data
        .get("stdout")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");
    let stderr = data
        .get("stderr")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");
    if stdout.trim().is_empty() && stderr.trim().is_empty() {
        return Vec::new();
    }

    let show_expand_hint = !expanded_output
        && [stdout, stderr]
            .into_iter()
            .any(|output| output.lines().count() > COMPACT_SHELL_BODY_LINES);
    let mut lines = render_shell_card_header_lines(tool, theme, width);
    if !stdout.trim().is_empty() || !stderr.trim().is_empty() {
        mark_last_source_boundary(&mut lines, Break::BlockBreak);
    }
    if !stdout.trim().is_empty() {
        lines.extend(render_shell_output_section(
            output_title("stdout", data.get("stdout_truncated")),
            stdout,
            root_text_style(theme),
            theme,
            width,
            expanded_output,
        ));
    }
    if !stderr.trim().is_empty() {
        if !stdout.trim().is_empty() {
            mark_last_source_boundary(&mut lines, Break::BlockBreak);
        }
        if lines.len() > 4 {
            lines.push(render_card_line(
                &[],
                Style::default().bg(theme.card_bg()),
                theme,
                width,
            ));
        }
        lines.extend(render_shell_output_section(
            output_title("stderr", data.get("stderr_truncated")),
            stderr,
            root_text_style(theme),
            theme,
            width,
            expanded_output,
        ));
    }
    if show_expand_hint {
        lines.push(render_card_line(
            &[(
                "… click to expand for details".to_string(),
                root_muted_style(theme).bg(theme.card_bg()),
            )],
            Style::default().bg(theme.card_bg()),
            theme,
            width,
        ));
        lines.push(render_card_line(
            &[],
            Style::default().bg(theme.card_bg()),
            theme,
            width,
        ));
    }
    lines
}

pub(super) fn mark_last_source_boundary(lines: &mut [SemanticLine<Style>], boundary: Break) {
    if let Some(line) = lines
        .iter_mut()
        .rev()
        .find(|line| line.spans.iter().any(|span| span.copy))
    {
        line.boundary = boundary;
    }
}

pub(super) fn render_shell_card_header_lines(
    tool: &ToolView,
    theme: Theme,
    width: usize,
) -> Vec<SemanticLine<Style>> {
    let mut lines = Vec::new();
    let command = shell_command(tool);
    let title = shell_card_title(tool, command.as_deref());

    lines.push(render_card_line(
        &[],
        Style::default().bg(theme.card_bg()),
        theme,
        width,
    ));
    lines.push(render_card_line(
        &[(format!("# {title}"), shell_card_title_style(theme))],
        Style::default().bg(theme.card_bg()),
        theme,
        width,
    ));
    lines.push(render_card_line(
        &[],
        Style::default().bg(theme.card_bg()),
        theme,
        width,
    ));

    if let Some(command) = command {
        let command_width = shell_card_content_width(width).saturating_sub(2).max(1);
        for (index, wrapped) in wrap_text_to_width(&command, command_width)
            .into_iter()
            .enumerate()
        {
            let prompt = if index == 0 { "$ " } else { "  " };
            lines.push(render_card_line_with_guide(
                &[
                    SemanticSpan::decoration(prompt, shell_card_command_style(theme)),
                    SemanticSpan::source(wrapped, shell_card_command_style(theme)),
                ],
                Style::default().bg(theme.card_bg()),
                theme.card_guide(),
                theme,
                width,
                Break::SoftWrap,
            ));
        }
        lines.push(render_card_line(
            &[],
            Style::default().bg(theme.card_bg()),
            theme,
            width,
        ));
    }

    lines
}

pub(super) fn render_shell_output_section(
    title: &str,
    text: &str,
    text_style: Style,
    theme: Theme,
    width: usize,
    expanded_output: bool,
) -> Vec<SemanticLine<Style>> {
    let mut lines = Vec::new();
    lines.push(render_card_line(
        &[(
            title.to_string(),
            root_muted_style(theme)
                .bg(theme.card_bg())
                .add_modifier(Modifier::BOLD),
        )],
        Style::default().bg(theme.card_bg()),
        theme,
        width,
    ));
    lines.push(render_card_line(
        &[],
        Style::default().bg(theme.card_bg()),
        theme,
        width,
    ));
    lines.extend(render_tail_limited_text_lines(
        text,
        text_style,
        theme,
        width,
        expanded_output,
    ));
    lines.push(render_card_line(
        &[],
        Style::default().bg(theme.card_bg()),
        theme,
        width,
    ));
    lines
}

pub(super) fn shell_command(tool: &ToolView) -> Option<String> {
    tool_arguments(tool)
        .as_ref()
        .and_then(|args| value_str(Some(args), "command"))
        .map(ToOwned::to_owned)
}

pub(super) fn shell_card_title(tool: &ToolView, command: Option<&str>) -> String {
    let summary = tool.summary.trim();
    if summary.is_empty() || summary.starts_with("exit ") || summary.starts_with("run ") {
        command
            .map(sentence_case_command_goal)
            .unwrap_or_else(|| "Runs command".to_string())
    } else {
        summary.to_string()
    }
}

pub(super) fn sentence_case_command_goal(command: &str) -> String {
    let command = command.trim();
    if command.is_empty() {
        return "Runs command".to_string();
    }
    let first = command.split_whitespace().next().unwrap_or("command");
    match first {
        "cargo" if command.contains("test") => "Runs test suite".to_string(),
        "cargo" if command.contains("fmt") => "Formats code".to_string(),
        "git" if command.contains("status") => "Shows working tree status".to_string(),
        "git" if command.contains("diff") => "Shows current diff".to_string(),
        _ => format!("Runs {first}"),
    }
}

pub(super) fn shell_card_title_style(theme: Theme) -> Style {
    Style::default()
        .fg(theme.accent)
        .bg(theme.card_bg())
        .add_modifier(Modifier::BOLD)
}

pub(super) fn shell_card_command_style(theme: Theme) -> Style {
    Style::default().fg(theme.text).bg(theme.card_bg())
}
