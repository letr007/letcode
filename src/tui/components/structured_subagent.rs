use ratatui::style::{Color, Modifier, Style};

use crate::subagent::StructuredSubagentResult;
use crate::tui::{
    measure::{display_width, wrap_text_to_width_with_offsets},
    surface,
    theme::Theme,
    transcript_render::{Break, Document, Line, SourceRange, Span},
};

use super::composer::one_line_snippet;

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

    push_card_content(
        &mut document,
        &format!("{}  Subagent", result.status),
        title_style,
        accent,
        theme,
        width,
    );
    push_card_content(
        &mut document,
        &one_line_snippet(&result.summary, width.saturating_sub(3)),
        content_style,
        accent,
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
        push_card_content(&mut document, &metadata, detail_style, accent, theme, width);
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
        let detail = format!(
            "{label} {} · {}",
            values.len(),
            one_line_snippet(&values[0], width.saturating_sub(label.len() + 9).max(1))
        );
        push_card_content(&mut document, &detail, detail_style, accent, theme, width);
    }

    document.finish();
    debug_assert!(document.validate());
    document
}

fn push_card_content(
    document: &mut Document<Style>,
    text: &str,
    text_style: Style,
    accent: Color,
    theme: Theme,
    width: usize,
) {
    let bar_style = Style::default().fg(accent).bg(theme.root_bg);
    if width == 1 {
        document.push_line(
            Line {
                spans: vec![Span::decoration(surface::ACCENT_BAR_GLYPH, bar_style)],
            },
            Break::SoftWrap,
        );
        return;
    }

    let content_width = width.saturating_sub(2).max(1);
    let block = document.add_source(text);
    let chunks = wrap_text_to_width_with_offsets(text, content_width);
    for (index, chunk) in chunks.iter().enumerate() {
        let mut spans = vec![
            Span::decoration(surface::ACCENT_BAR_GLYPH, bar_style),
            Span::decoration(" ", text_style),
        ];
        if chunk.source_start_char < chunk.source_end_char {
            spans.push(Span::source(
                chunk.text.clone(),
                text_style,
                SourceRange::new(block, chunk.source_start_char, chunk.source_end_char),
            ));
        }
        let used = spans.iter().map(|span| display_width(&span.text)).sum();
        if width > used {
            spans.push(Span::decoration(" ".repeat(width - used), text_style));
        }
        document.push_line(Line { spans }, chunk_boundary(&chunks, index));
    }
}

fn chunk_boundary(chunks: &[crate::tui::measure::WrappedChunk], index: usize) -> Break {
    let Some(current) = chunks.get(index) else {
        return Break::SoftWrap;
    };
    let Some(next) = chunks.get(index + 1) else {
        return Break::SoftWrap;
    };
    if next.source_start_char > current.source_end_char {
        Break::HardBreak
    } else {
        Break::SoftWrap
    }
}
