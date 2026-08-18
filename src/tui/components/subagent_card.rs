//! Subagent (delegated child) tool card rendering.

use ratatui::style::{Modifier, Style};

use super::semantic_spans::*;
use crate::agent::{agent_name_for_subagent_tool, is_subagent_tool_name};
use crate::subagent::StructuredSubagentResult;
use crate::tui::{
    measure::{display_width, wrap_text_to_width},
    theme::Theme,
    timeline::{ToolExecutionStatus, ToolView},
    transcript_render::{Break, CopyJoin, SemanticLine, SemanticSpan},
};

pub(super) fn render_subagent_lines(
    tool: &ToolView,
    theme: Theme,
    width: usize,
    frame: usize,
    expanded_output: bool,
) -> Vec<SemanticLine<Style>> {
    if width == 0 {
        return Vec::new();
    }

    let data = tool_output_data(tool);
    let structured = data
        .as_ref()
        .and_then(|data| data.get("structured_result"))
        .cloned()
        .and_then(|value| serde_json::from_value::<StructuredSubagentResult>(value).ok())
        .filter(|result| !result.malformed);
    let localized_status = crate::tui::state::TuiState::default().translator();
    let status = data
        .as_ref()
        .and_then(|data| data.get("status").and_then(serde_json::Value::as_str))
        .or_else(|| structured.as_ref().map(|result| result.status.as_str()))
        .map(str::to_owned)
        .unwrap_or_else(|| subagent_status_label(tool.status, &localized_status));
    let child_id = data
        .as_ref()
        .and_then(|data| data.get("child_session_id"))
        .and_then(serde_json::Value::as_str)
        .or_else(|| {
            structured
                .as_ref()
                .map(|result| result.child_session_id.as_str())
        })
        .map(|id| truncate_display_width(id, 16))
        .unwrap_or_else(|| "child".into());
    let summary = data
        .as_ref()
        .and_then(|data| data.get("summary"))
        .and_then(serde_json::Value::as_str)
        .map(one_line_snippet)
        .filter(|summary| !summary.is_empty())
        .or_else(|| {
            structured
                .as_ref()
                .map(|result| one_line_snippet(&result.summary))
                .filter(|summary| !summary.is_empty())
        })
        .or_else(|| subagent_task(tool))
        .unwrap_or_else(|| one_line_snippet(&tool.summary));
    let agent_name = data
        .as_ref()
        .and_then(|data| data.get("agent_name"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or_else(|| subagent_name_from_tool(&tool.name));
    let mut state_flags = data.as_ref().map(subagent_state_flags).unwrap_or_default();
    if let Some(failure_kind) = data
        .as_ref()
        .and_then(|data| data.get("failure_kind"))
        .or_else(|| {
            data.as_ref()
                .and_then(|data| data.get("structured_result"))
                .and_then(|result| result.get("failure_kind"))
        })
        .and_then(serde_json::Value::as_str)
        .and_then(|kind| match kind {
            "hard" => Some("hard"),
            "logical" => Some("logical"),
            _ => None,
        })
    {
        state_flags.push(failure_kind);
    }
    let state_suffix = if state_flags.is_empty() {
        String::new()
    } else {
        format!(" [{}]", state_flags.join("/"))
    };

    let status_label = if matches!(
        tool.status,
        ToolExecutionStatus::Pending | ToolExecutionStatus::Running
    ) {
        format!(
            "{} {}",
            PROCESS_FRAMES[frame % PROCESS_FRAMES.len()],
            status_label(
                map_tool_status(tool.status),
                &crate::tui::state::TuiState::default().translator()
            )
        )
    } else {
        status
    };
    let status_color = match tool.status {
        ToolExecutionStatus::Pending => theme.warning,
        ToolExecutionStatus::Running => theme.warning,
        ToolExecutionStatus::Cancelled => theme.error,
        ToolExecutionStatus::Succeeded => theme.assistant,
        ToolExecutionStatus::Failed => theme.error,
    };

    let status_style = root_status_style(status_color, theme);
    let text_style = root_text_style(theme);
    let muted = root_muted_style(theme);
    let mut lines = Vec::new();
    let has_structured_details = structured
        .as_ref()
        .is_some_and(structured_subagent_has_details);
    lines.push(render_card_line_with_guide(
        &[
            SemanticSpan::decoration(format!("{status_label}{state_suffix}"), status_style),
            SemanticSpan::decoration(" ", text_style),
            SemanticSpan::decoration(
                agent_name.to_string(),
                text_style.add_modifier(Modifier::BOLD),
            ),
            SemanticSpan::decoration(" ", text_style),
            SemanticSpan::source(summary, text_style),
            SemanticSpan::decoration(" · ", muted),
            SemanticSpan::decoration(format!("/child {child_id}"), muted),
        ],
        Style::default().bg(theme.root_bg),
        theme.card_guide(),
        theme,
        width,
        if has_structured_details {
            Break::HardBreak
        } else {
            Break::End
        },
    ));

    let Some(structured) = structured else {
        return lines;
    };
    if !has_structured_details {
        lines[0].boundary = Break::End;
        return lines;
    }

    let activity = subagent_activity_summary(&structured);
    if !expanded_output {
        let activity_label = if activity.is_empty() {
            "details · expand".to_string()
        } else {
            format!("{activity} · expand")
        };
        lines.push(render_subagent_compact_line(
            &activity_label,
            muted,
            theme,
            width,
            Break::End,
        ));
        return lines;
    }

    lines.push(render_subagent_compact_line(
        "details · collapse",
        muted,
        theme,
        width,
        Break::HardBreak,
    ));

    let run_id = data
        .as_ref()
        .and_then(|data| data.get("run_id"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or(structured.run_id.as_str());
    if !run_id.is_empty() {
        lines.push(render_subagent_compact_line(
            &format!("run {run_id}"),
            muted,
            theme,
            width,
            Break::HardBreak,
        ));
    }
    render_subagent_wrapped_field(
        &mut lines,
        "summary",
        std::slice::from_ref(&structured.summary),
        muted,
        theme,
        width,
    );
    for (label, values) in [
        ("blocker", &structured.blockers),
        ("finding", &structured.findings),
        ("next_step", &structured.next_steps),
        ("validation", &structured.validation),
        ("changed", &structured.files_changed),
        ("read", &structured.files_read),
        ("command", &structured.commands_run),
    ] {
        render_subagent_wrapped_field(&mut lines, label, values, muted, theme, width);
    }
    if let Some(last) = lines.last_mut() {
        last.boundary = Break::End;
    }
    lines
}

pub(super) fn render_subagent_compact_line(
    text: &str,
    muted: Style,
    theme: Theme,
    width: usize,
    boundary: Break,
) -> SemanticLine<Style> {
    render_card_line_with_guide(
        &[SemanticSpan::decoration(text.to_string(), muted)],
        Style::default().bg(theme.root_bg),
        theme.card_guide(),
        theme,
        width,
        boundary,
    )
}

pub(super) fn render_subagent_wrapped_field(
    lines: &mut Vec<SemanticLine<Style>>,
    label: &str,
    values: &[String],
    muted: Style,
    theme: Theme,
    width: usize,
) {
    let label_text = format!("{label}: ");
    let label_width = display_width(&label_text);
    let content_width = width
        .saturating_sub(display_width(TOOL_GUIDE_GLYPH).saturating_add(2))
        .max(1);
    let value_width = content_width.saturating_sub(label_width).max(1);
    for value in values {
        let wrapped = wrap_text_to_width(value, value_width);
        for (index, chunk) in wrapped.into_iter().enumerate() {
            let prefix = if index == 0 {
                label_text.clone()
            } else {
                " ".repeat(label_width)
            };
            let mut segments = vec![SemanticSpan::decoration(prefix, muted)];
            if !chunk.is_empty() {
                segments.push(SemanticSpan::source_with_join(
                    chunk,
                    muted,
                    CopyJoin::Space,
                ));
            }
            lines.push(render_card_line_with_guide(
                &segments,
                Style::default().bg(theme.root_bg),
                theme.card_guide(),
                theme,
                width,
                Break::HardBreak,
            ));
        }
    }
}

pub(super) fn structured_subagent_has_details(result: &StructuredSubagentResult) -> bool {
    !result.summary.trim().is_empty()
        || !result.blockers.is_empty()
        || !result.findings.is_empty()
        || !result.next_steps.is_empty()
        || !result.validation.is_empty()
        || !result.files_changed.is_empty()
        || !result.files_read.is_empty()
        || !result.commands_run.is_empty()
}

pub(super) fn subagent_activity_summary(result: &StructuredSubagentResult) -> String {
    [
        ("read", result.files_read.len()),
        ("changed", result.files_changed.len()),
        ("commands", result.commands_run.len()),
        ("checks", result.validation.len()),
    ]
    .into_iter()
    .filter(|(_, count)| *count > 0)
    .map(|(label, count)| format!("{label} {count}"))
    .collect::<Vec<_>>()
    .join(" · ")
}

pub(super) fn subagent_task(tool: &ToolView) -> Option<String> {
    tool_arguments(tool)
        .as_ref()
        .and_then(|args| value_str(Some(args), "task"))
        .map(one_line_snippet)
        .filter(|task| !task.is_empty())
}

pub(super) fn is_subagent_tool(name: &str) -> bool {
    is_subagent_tool_name(name)
}

pub(super) fn subagent_name_from_tool(name: &str) -> &str {
    agent_name_for_subagent_tool(name).expect("tool card received unknown subagent tool")
}

pub(super) fn subagent_status_label(
    status: ToolExecutionStatus,
    translator: &crate::tui::i18n::Translator,
) -> String {
    translator.t(match status {
        ToolExecutionStatus::Pending => "status.preparing",
        ToolExecutionStatus::Running => "status.running",
        ToolExecutionStatus::Cancelled => "status.cancelled",
        ToolExecutionStatus::Succeeded => "status.completed",
        ToolExecutionStatus::Failed => "status.failed",
    })
}

pub(super) fn subagent_state_flags(data: &serde_json::Value) -> Vec<&'static str> {
    let mut flags = Vec::new();
    if data.get("active").and_then(serde_json::Value::as_bool) == Some(true) {
        flags.push("active");
    }
    if data
        .get("unreconciled")
        .and_then(serde_json::Value::as_bool)
        == Some(true)
    {
        flags.push("unreconciled");
    }
    if data.get("reconciled").and_then(serde_json::Value::as_bool) == Some(true) {
        flags.push("reconciled");
    }
    if data.get("reusable").and_then(serde_json::Value::as_bool) == Some(true) {
        flags.push("reusable");
    }
    if data
        .get("structured_result")
        .and_then(|result| result.get("malformed"))
        .and_then(serde_json::Value::as_bool)
        == Some(true)
    {
        flags.push("malformed");
    }
    flags
}

