use ratatui::style::{Modifier, Style};

use crate::subagent::StructuredSubagentResult;
use crate::tui::{theme::Theme, transcript_render::Document};

use super::{composer::one_line_snippet, delegation_card::push_wrapped_card_line};

pub(crate) fn render_structured_subagent_result_document(
    result: &StructuredSubagentResult,
    theme: Theme,
    width: usize,
) -> Document<Style> {
    let mut document = Document::default();
    if width == 0 {
        return document;
    }

    let accent = match result.status.as_str() {
        "completed" => theme.success,
        "failed" | "cancelled" | "timed_out" | "budget_exhausted" => theme.error,
        _ => theme.notice,
    };
    let title_style = Style::default()
        .fg(accent)
        .bg(theme.root_bg)
        .add_modifier(Modifier::BOLD);
    let content_style = Style::default().fg(theme.text).bg(theme.root_bg);
    let detail_style = Style::default().fg(theme.muted_text).bg(theme.root_bg);

    push_wrapped_card_line(
        &mut document,
        &format!("{}  Subagent", result.status),
        "",
        title_style,
        theme,
        width,
    );
    push_wrapped_card_line(
        &mut document,
        "",
        &one_line_snippet(&result.summary, width.saturating_sub(3)),
        content_style,
        theme,
        width,
    );

    let counts = [
        ("read", result.files_read.len()),
        ("changed", result.files_changed.len()),
        ("commands", result.commands_run.len()),
        ("checks", result.validation.len()),
    ];
    if counts.iter().any(|(_, count)| *count > 0) {
        let metadata = counts
            .iter()
            .filter(|(_, count)| *count > 0)
            .map(|(label, count)| format!("{label} {count}"))
            .collect::<Vec<_>>()
            .join(" · ");
        push_wrapped_card_line(&mut document, &metadata, "", detail_style, theme, width);
    }

    let details = [
        ("Blockers", &result.blockers),
        ("Findings", &result.findings),
        ("Next steps", &result.next_steps),
        ("Validation", &result.validation),
        ("Files changed", &result.files_changed),
        ("Files read", &result.files_read),
        ("Commands", &result.commands_run),
    ];
    for (label, values) in details
        .into_iter()
        .filter(|(_, values)| !values.is_empty())
        .take(3)
    {
        push_wrapped_card_line(
            &mut document,
            &format!("{label} {} · ", values.len()),
            &one_line_snippet(&values[0], width.saturating_sub(label.len() + 9).max(1)),
            detail_style,
            theme,
            width,
        );
    }

    document.finish();
    debug_assert!(document.validate());
    document
}
