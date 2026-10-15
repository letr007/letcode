//! Subagent (delegated child) tool card rendering.

use ratatui::style::{Modifier, Style};

use super::semantic_spans::*;
use crate::agent::{agent_name_for_subagent_tool, is_subagent_tool_name};
use crate::subagent::StructuredSubagentResult;
use crate::tui::{
    i18n::Translator,
    measure::{display_width, wrap_text_to_width_with_offsets},
    theme::Theme,
    timeline::{ToolExecutionStatus, ToolView},
    transcript_render::{Break, CopyJoin, Document, SemanticLine, SemanticSpan},
};

pub(super) fn render_subagent_lines(
    tool: &ToolView,
    theme: Theme,
    width: usize,
    frame: usize,
    expanded_output: bool,
    translator: &Translator,
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
    let reported_status = data
        .as_ref()
        .and_then(|data| data.get("status").and_then(serde_json::Value::as_str))
        .map(str::to_owned);
    let verdict = structured
        .as_ref()
        .map(|result| result.status.as_str())
        .filter(|verdict| !verdict.is_empty());
    let status = reported_status
        .as_deref()
        .or(verdict)
        .map(|raw| subagent_status_text(raw, translator))
        .unwrap_or_else(|| subagent_status_label(tool.status, translator));
    // The run status is host-reported; the child's own verdict is separate and
    // stays visible when it differs.
    let status = match (verdict, reported_status.as_deref()) {
        (Some(verdict), Some(reported)) if verdict != reported => {
            format!("{status} · {}", subagent_status_text(verdict, translator))
        }
        _ => status,
    };
    let child_id = data
        .as_ref()
        .and_then(|data| data.get("child_session_id"))
        .and_then(serde_json::Value::as_str)
        .or_else(|| {
            structured
                .as_ref()
                .map(|result| result.child_session_id.as_str())
        })
        .filter(|id| !id.is_empty())
        .map(|id| truncate_display_width(id, 16));
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
        .unwrap_or_else(|| agent_name_for_subagent_tool(&tool.name).unwrap_or("Subagent"));
    let waiting = data
        .as_ref()
        .and_then(|data| data.get("waiting"))
        .and_then(serde_json::Value::as_bool)
        == Some(true)
        || tool.name == crate::tool_names::TOOL_AGENT_WAIT;
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
        let labels = state_flags
            .iter()
            .map(|flag| subagent_state_flag_label(flag, translator))
            .collect::<Vec<_>>()
            .join("/");
        format!(" [{labels}]")
    };

    let status_label = if matches!(
        tool.status,
        ToolExecutionStatus::Pending | ToolExecutionStatus::Running
    ) {
        let label = if waiting {
            translator.t("status.waiting")
        } else {
            status_label(map_tool_status(tool.status), translator)
        };
        format!("{} {label}", PROCESS_FRAMES[frame % PROCESS_FRAMES.len()])
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
    let mut header = vec![
        SemanticSpan::decoration(format!("{status_label}{state_suffix}"), status_style),
        SemanticSpan::decoration(" ", text_style),
        SemanticSpan::decoration(
            agent_name.to_string(),
            text_style.add_modifier(Modifier::BOLD),
        ),
        SemanticSpan::decoration(" ", text_style),
        SemanticSpan::source(summary, text_style),
    ];
    if let Some(child_id) = child_id {
        header.push(SemanticSpan::decoration(
            format!(" · /child {child_id}"),
            muted,
        ));
    } else if tool.name == crate::tool_names::TOOL_AGENT_WAIT
        && let Some(run_id) = tool_arguments(tool)
            .as_ref()
            .and_then(|args| args.get("run_id"))
            .and_then(serde_json::Value::as_str)
    {
        header.push(SemanticSpan::source(format!(" · {run_id}"), muted));
    }
    lines.push(render_card_line_with_guide(
        &header,
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

    let activity = subagent_activity_summary(&structured, translator);
    if !expanded_output {
        let expand = translator.t("subagent.expand");
        let activity_label = if activity.is_empty() {
            format!("{} · {expand}", translator.t("subagent.details"))
        } else {
            format!("{activity} · {expand}")
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
        &format!(
            "{} · {}",
            translator.t("subagent.details"),
            translator.t("subagent.collapse")
        ),
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
            &translator.t_fmt("subagent.run", &[("id", run_id)]),
            muted,
            theme,
            width,
            Break::HardBreak,
        ));
    }
    render_subagent_wrapped_field(
        &mut lines,
        &translator.t("subagent.field_summary"),
        std::slice::from_ref(&structured.summary),
        muted,
        theme,
        width,
    );
    for (key, values) in [
        ("subagent.field_blocker", &structured.blockers),
        ("subagent.field_finding", &structured.findings),
        ("subagent.field_next_step", &structured.next_steps),
        ("subagent.field_validation", &structured.validation),
        ("subagent.field_changed", &structured.files_changed),
        ("subagent.field_read", &structured.files_read),
        ("subagent.field_command", &structured.commands_run),
    ] {
        render_subagent_wrapped_field(&mut lines, &translator.t(key), values, muted, theme, width);
    }
    if let Some(last) = lines.last_mut() {
        last.boundary = Break::End;
    }
    lines
}

pub(super) fn render_subagent_send_document(
    tool: &ToolView,
    theme: Theme,
    width: usize,
    frame: usize,
    expanded: bool,
    translator: &Translator,
) -> Document<Style> {
    let mut document = Document::default();
    if width == 0 {
        return document;
    }
    let args = tool_arguments(tool);
    let data = tool_output_data(tool);
    let target = args
        .as_ref()
        .and_then(|args| args.get("session_id"))
        .or_else(|| data.as_ref().and_then(|data| data.get("session_id")))
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default();
    let message = args
        .as_ref()
        .and_then(|args| args.get("message"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default();
    let queued = tool.status == ToolExecutionStatus::Succeeded
        && data.as_ref().and_then(|data| data.get("accepted"))
            == Some(&serde_json::Value::Bool(true));
    let state = if queued {
        translator.t("status.queued")
    } else {
        subagent_status_label(tool.status, translator)
    };
    let active = matches!(
        tool.status,
        ToolExecutionStatus::Pending | ToolExecutionStatus::Running
    );
    let state = if active {
        format!("{} {state}", PROCESS_FRAMES[frame % PROCESS_FRAMES.len()])
    } else {
        state
    };
    let accent = if queued {
        theme.warning
    } else if matches!(
        tool.status,
        ToolExecutionStatus::Failed | ToolExecutionStatus::Cancelled
    ) {
        theme.error
    } else {
        theme.notice
    };
    document.push_semantic_line(render_card_line_with_guide(
        &[
            SemanticSpan::decoration(state, root_status_style(accent, theme)),
            SemanticSpan::decoration(
                format!(" {} ", translator.t("subagent.send")),
                root_text_style(theme).add_modifier(Modifier::BOLD),
            ),
            SemanticSpan::source(target, root_text_style(theme)),
        ],
        Style::default().bg(theme.root_bg),
        theme.card_guide(),
        theme,
        width,
        Break::HardBreak,
    ));
    let hint = translator.t(if expanded {
        "subagent.collapse"
    } else {
        "subagent.expand"
    });
    document.push_semantic_line(render_subagent_compact_line(
        &hint,
        root_muted_style(theme),
        theme,
        width,
        Break::HardBreak,
    ));
    if expanded {
        super::delegation_card::push_wrapped_card_line(
            &mut document,
            "",
            message,
            root_text_style(theme),
            theme,
            width,
        );
        if matches!(
            tool.status,
            ToolExecutionStatus::Failed | ToolExecutionStatus::Cancelled
        ) {
            super::delegation_card::push_wrapped_card_line(
                &mut document,
                "",
                tool.output.as_deref().unwrap_or(&tool.summary),
                root_status_style(theme.error, theme),
                theme,
                width,
            );
        }
    } else {
        document.push_semantic_line(render_card_line_with_guide(
            &[SemanticSpan::source(
                one_line_snippet(message),
                root_text_style(theme),
            )],
            Style::default().bg(theme.root_bg),
            theme.card_guide(),
            theme,
            width,
            Break::End,
        ));
    }
    document.finish();
    debug_assert!(document.validate());
    document
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
    let label_chars = label_text.chars().collect::<Vec<_>>();
    let content_width = width
        .saturating_sub(display_width(TOOL_GUIDE_GLYPH).saturating_add(2))
        .max(1);
    for value in values {
        let text = format!("{label_text}{value}");
        let chars = text.chars().collect::<Vec<_>>();
        let wrapped = wrap_text_to_width_with_offsets(&text, content_width);
        for (index, chunk) in wrapped.iter().enumerate() {
            let mut segments = Vec::new();
            let prefix_end = chunk.source_end_char.min(label_chars.len());
            if chunk.source_start_char < prefix_end {
                segments.push(SemanticSpan::decoration(
                    label_chars[chunk.source_start_char..prefix_end]
                        .iter()
                        .collect::<String>(),
                    muted,
                ));
            }
            let value_start = chunk.source_start_char.max(label_chars.len());
            if value_start < chunk.source_end_char {
                segments.push(SemanticSpan::source_with_join(
                    chars[value_start..chunk.source_end_char]
                        .iter()
                        .collect::<String>(),
                    muted,
                    CopyJoin::Concat,
                ));
            }
            let boundary = if index + 1 == wrapped.len()
                || wrapped[index + 1].source_start_char > chunk.source_end_char
            {
                Break::HardBreak
            } else {
                Break::SoftWrap
            };
            lines.push(render_card_line_with_guide(
                &segments,
                Style::default().bg(theme.root_bg),
                theme.card_guide(),
                theme,
                width,
                boundary,
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

pub(super) fn subagent_activity_summary(
    result: &StructuredSubagentResult,
    translator: &Translator,
) -> String {
    [
        ("subagent.activity_read", result.files_read.len()),
        ("subagent.activity_changed", result.files_changed.len()),
        ("subagent.activity_commands", result.commands_run.len()),
        ("subagent.activity_checks", result.validation.len()),
    ]
    .into_iter()
    .filter(|(_, count)| *count > 0)
    .map(|(key, count)| format!("{} {count}", translator.t(key)))
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

pub(super) fn subagent_status_text(raw: &str, translator: &Translator) -> String {
    translator.t(match raw {
        "preparing" => "status.preparing",
        "running" => "status.running",
        "cancelled" => "status.cancelled",
        "completed" => "status.completed",
        "failed" => "status.failed",
        "budget_exhausted" => "status.budget_exhausted",
        "timed_out" => "status.timed_out",
        "approval" => "status.approval",
        "approved" => "status.approved",
        "denied" => "status.denied",
        "error" => "status.error",
        "interrupted" => "status.interrupted",
        other => return other.to_string(),
    })
}

fn subagent_state_flag_label(flag: &str, translator: &Translator) -> String {
    translator.t(match flag {
        "background" => "status.background",
        "active" => "status.active",
        "malformed" => "status.malformed",
        "hard" => "status.hard",
        "logical" => "status.logical",
        other => return other.to_string(),
    })
}

pub(super) fn subagent_state_flags(data: &serde_json::Value) -> Vec<&'static str> {
    let mut flags = Vec::new();
    if data.get("background").and_then(serde_json::Value::as_bool) == Some(true) {
        flags.push("background");
    }
    if data.get("active").and_then(serde_json::Value::as_bool) == Some(true)
        && data.get("waiting").and_then(serde_json::Value::as_bool) != Some(true)
    {
        flags.push("active");
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tui::i18n::Language;

    #[test]
    fn every_reported_subagent_status_is_localized() {
        let translator = Translator::new(Language::ZhCn);
        let terminal = [
            crate::subagent::SubagentStatus::Running,
            crate::subagent::SubagentStatus::Completed,
            crate::subagent::SubagentStatus::Failed,
            crate::subagent::SubagentStatus::BudgetExhausted,
            crate::subagent::SubagentStatus::Cancelled,
            crate::subagent::SubagentStatus::TimedOut,
        ]
        .map(|status| status.as_str());
        let projected = [
            "preparing",
            "approval",
            "approved",
            "denied",
            "error",
            "interrupted",
        ];
        for raw in terminal.into_iter().chain(projected) {
            let text = subagent_status_text(raw, &translator);
            assert_ne!(text, raw, "{raw} is not localized");
        }
    }

    #[test]
    fn unknown_subagent_status_passes_through() {
        let translator = Translator::new(Language::ZhCn);
        assert_eq!(
            subagent_status_text("changes_requested", &translator),
            "changes_requested"
        );
    }

    #[test]
    fn every_state_flag_is_localized() {
        let translator = Translator::new(Language::ZhCn);
        for flag in ["background", "active", "malformed", "hard", "logical"] {
            assert_ne!(subagent_state_flag_label(flag, &translator), flag);
        }
    }
}
