//! Read-only presentation of host-authored Historian reports in child sessions.
use crate::context_history::{HistoryFactCategory, HistoryTier};
use crate::historian::{HistorianReport, UsageUpdate};
use crate::tui::{
    i18n::Translator,
    markdown::{MarkdownRenderOptions, render_markdown_document},
    measure::wrap_text_to_width_with_offsets,
    theme::Theme,
    transcript_render::{Break, Document, Line, SourceRange, Span},
};
use ratatui::style::{Modifier, Style};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ReportView {
    #[default]
    Compact,
    Detailed,
    Anchor,
    Raw,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ReportOptions {
    pub view: ReportView,
    pub sources: bool,
}

pub(crate) fn parse_report(text: &str) -> Option<HistorianReport> {
    serde_json::from_str(text).ok()
}

pub(crate) fn render_report(
    report: &HistorianReport,
    options: ReportOptions,
    theme: Theme,
    width: usize,
    tr: &Translator,
) -> Document<Style> {
    let mut doc = Document::default();
    if width == 0 {
        return doc;
    }
    if width == 1 {
        doc.push_line(
            Line {
                spans: vec![Span::decoration("…", theme.app_style())],
            },
            Break::End,
        );
        return doc;
    }
    let title = Style::default()
        .fg(theme.accent)
        .bg(theme.root_bg)
        .add_modifier(Modifier::BOLD);
    let section = Style::default()
        .fg(theme.text)
        .bg(theme.root_bg)
        .add_modifier(Modifier::BOLD);
    let muted = Style::default().fg(theme.muted_text).bg(theme.root_bg);
    let normal = Style::default().fg(theme.text).bg(theme.root_bg);
    text(
        &mut doc,
        &format!("Historian · {}", tr.t("historian.prepared")),
        title,
        width,
    );
    let p = &report.publication;
    text(
        &mut doc,
        &tr.t_fmt(
            "historian.overview",
            &[
                ("items", &p.source_ids.len().to_string()),
                ("episodes", &p.compartments.len().to_string()),
                ("facts", &p.facts.len().to_string()),
                ("withdrawals", &p.withdrawn_fact_ids.len().to_string()),
            ],
        ),
        normal,
        width,
    );
    text(
        &mut doc,
        &format!(
            "{} · {:.1}s",
            report.model,
            report.elapsed_ms as f64 / 1000.0
        ),
        muted,
        width,
    );
    let view_key = match options.view {
        ReportView::Compact => "historian.compact",
        ReportView::Detailed => "historian.detailed",
        ReportView::Anchor => "historian.anchor",
        ReportView::Raw => "historian.raw",
    };
    text(
        &mut doc,
        &tr.t_fmt("historian.view", &[("view", &tr.t(view_key))]),
        title,
        width,
    );
    text(&mut doc, &tr.t("historian.controls"), muted, width);
    gap(&mut doc);
    if options.view == ReportView::Raw {
        // Pretty-print the persisted host report, not an invented reconstruction
        // of the original provider wire response.
        text(
            &mut doc,
            &serde_json::to_string_pretty(report).expect("serializable report"),
            normal,
            width,
        );
        doc.finish();
        return doc;
    }
    let tier = match options.view {
        ReportView::Detailed => HistoryTier::Detailed,
        ReportView::Anchor => HistoryTier::Anchor,
        _ => HistoryTier::Compact,
    };
    for c in &p.compartments {
        text(&mut doc, &c.title, section, width);
        text(
            &mut doc,
            &tr.t_fmt(
                "historian.episode_meta",
                &[
                    ("sources", &c.source_ids.len().to_string()),
                    ("importance", &c.importance.to_string()),
                ],
            ),
            muted,
            width,
        );
        let body = c.text(tier);
        if body.is_empty() {
            text(&mut doc, &tr.t("historian.title_anchor"), muted, width);
        } else {
            doc.append(render_markdown_document(
                body,
                theme,
                MarkdownRenderOptions::new(width),
            ));
        }
        if options.sources {
            text(&mut doc, &format!("compartment:{}", c.id), muted, width);
            text(&mut doc, &c.source_ids.join(", "), muted, width);
        }
        gap(&mut doc);
    }
    if !p.facts.is_empty() || !p.withdrawn_fact_ids.is_empty() {
        text(&mut doc, &tr.t("historian.fact_changes"), section, width);
        for fact in &p.facts {
            let category = match fact.category {
                HistoryFactCategory::ProjectRules => "historian.project_rules",
                HistoryFactCategory::Architecture => "historian.architecture",
                HistoryFactCategory::Constraints => "historian.constraints",
                HistoryFactCategory::ConfigValues => "historian.config_values",
                HistoryFactCategory::Naming => "historian.naming",
            };
            let change = if fact.supersedes.is_empty() {
                "historian.added"
            } else {
                "historian.revised"
            };
            text(
                &mut doc,
                &format!("{} · {}", tr.t(change), tr.t(category)),
                title,
                width,
            );
            doc.append(render_markdown_document(
                &fact.text,
                theme,
                MarkdownRenderOptions::new(width),
            ));
            if !fact.supersedes.is_empty() {
                text(
                    &mut doc,
                    &tr.t_fmt(
                        "historian.replaces",
                        &[("count", &fact.supersedes.len().to_string())],
                    ),
                    muted,
                    width,
                );
                if options.sources {
                    text(&mut doc, &fact.supersedes.join(", "), muted, width);
                }
            }
            if options.sources {
                text(
                    &mut doc,
                    &format!("evidence:fact:{}", fact.id),
                    muted,
                    width,
                );
                text(&mut doc, &fact.source_ids.join(", "), muted, width);
            }
            gap(&mut doc);
        }
        if !p.withdrawn_fact_ids.is_empty() {
            text(
                &mut doc,
                &tr.t_fmt(
                    "historian.withdrawn",
                    &[("count", &p.withdrawn_fact_ids.len().to_string())],
                ),
                Style::default().fg(theme.warning).bg(theme.root_bg),
                width,
            );
            if options.sources {
                text(&mut doc, &p.withdrawn_fact_ids.join(", "), muted, width);
            }
            gap(&mut doc);
        }
    }
    text(&mut doc, &tr.t("historian.usage"), section, width);
    let latest_usage = report
        .usage
        .iter()
        .rev()
        .find(|e| matches!(e, UsageUpdate::Usage { .. }));
    let value = |v: Option<u64>| {
        v.map(|n| n.to_string())
            .unwrap_or_else(|| tr.t("historian.unreported"))
    };
    if let Some(UsageUpdate::Usage {
        input_tokens,
        output_tokens,
        cached_input_tokens,
        ..
    }) = latest_usage
    {
        text(
            &mut doc,
            &tr.t_fmt(
                "historian.tokens",
                &[
                    ("input", &input_tokens.to_string()),
                    ("output", &output_tokens.to_string()),
                ],
            ),
            normal,
            width,
        );
        text(
            &mut doc,
            &tr.t_fmt(
                "historian.cached",
                &[("tokens", &value(*cached_input_tokens))],
            ),
            muted,
            width,
        );
    } else {
        text(&mut doc, &tr.t("historian.unreported"), muted, width);
    }
    if let Some(UsageUpdate::Cache {
        read_tokens,
        write_tokens,
        ..
    }) = report
        .usage
        .iter()
        .rev()
        .find(|e| matches!(e, UsageUpdate::Cache { .. }))
    {
        text(
            &mut doc,
            &tr.t_fmt(
                "historian.cache",
                &[
                    ("read", &read_tokens.to_string()),
                    ("write", &write_tokens.to_string()),
                ],
            ),
            muted,
            width,
        );
    }
    if options.sources {
        gap(&mut doc);
        text(
            &mut doc,
            &format!(
                "session:{}/{}",
                report.source_session_id, report.source_branch_id
            ),
            muted,
            width,
        );
        text(&mut doc, &format!("publication:{}", p.id), muted, width);
    }
    doc.finish();
    debug_assert!(doc.validate());
    doc
}

fn gap(doc: &mut Document<Style>) {
    doc.push_line(Line { spans: vec![] }, Break::HardBreak);
}

fn text(doc: &mut Document<Style>, value: &str, style: Style, width: usize) {
    let block = doc.add_source(value);
    let chunks = wrap_text_to_width_with_offsets(value, width);
    for (index, chunk) in chunks.iter().enumerate() {
        let boundary = if chunks
            .get(index + 1)
            .is_none_or(|next| next.source_start_char > chunk.source_end_char)
        {
            Break::HardBreak
        } else {
            Break::SoftWrap
        };
        let spans = if chunk.text.is_empty() {
            // Empty visual lines (for example from a title with a leading or
            // trailing newline) have no source range. A zero-length source range
            // is intentionally rejected by Document::validate.
            Vec::new()
        } else {
            vec![Span::source(
                chunk.text.clone(),
                style,
                SourceRange::new(block, chunk.source_start_char, chunk.source_end_char),
            )]
        };
        doc.push_line(Line { spans }, boundary);
    }
}

#[cfg(test)]
#[path = "../../../tests/internal/tui/components/historian_report/tests.rs"]
pub(crate) mod tests;
