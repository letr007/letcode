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
    text(&mut doc, &tr.t("historian.application_note"), muted, width);
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
    text(&mut doc, &tr.t("historian.usage_note"), muted, width);
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
        doc.push_line(
            Line {
                spans: vec![Span::source(
                    chunk.text.clone(),
                    style,
                    SourceRange::new(block, chunk.source_start_char, chunk.source_end_char),
                )],
            },
            boundary,
        );
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::context_history::{HistoryCompartment, HistoryFact, HistoryPublication};
    use crate::historian::HistorianReportKind;
    use crate::tui::{i18n::Language, measure::display_width, transcript_ratatui};

    pub(crate) fn sample_report() -> HistorianReport {
        HistorianReport {
            kind: HistorianReportKind::Prepared,
            publication: HistoryPublication {
                id: "publication-example".into(), project_path: None, external_fact_ids: vec!["old-fact".into()],
                source_ids: vec!["raw:12".into(), "raw:14".into()],
                compartments: vec![HistoryCompartment {
                    id: "episode-example".into(), title: "清理摘要块 · transcript".into(),
                    source_ids: vec!["raw:12".into(), "raw:14".into()], importance: 80,
                    detailed: "Detailed-only rationale\n\nA second paragraph preserves the exact constraint.".into(),
                    compact: "Compact-only result".into(), anchor: "Anchor-only clue".into(),
                }],
                facts: vec![HistoryFact {id:"fact-example".into(),category:HistoryFactCategory::Architecture,
                    text:"Use footer status without blank summary blocks.".into(),source_ids:vec!["raw:14".into()],supersedes:vec!["old-fact".into()]}],
                withdrawn_fact_ids: vec![],
            },
            source_session_id: "parent-example".into(), source_branch_id: "main".into(), model:"provider/model".into(),elapsed_ms:8400,
            usage: vec![
                UsageUpdate::Usage { input_tokens:100,output_tokens:10,total_tokens:110,reasoning_tokens:None,cached_input_tokens:None },
                UsageUpdate::Usage { input_tokens:120,output_tokens:20,total_tokens:140,reasoning_tokens:None,cached_input_tokens:None },
            ],
        }
    }

    pub(crate) fn plain(doc: &Document<Style>) -> String {
        transcript_ratatui::document_to_ratatui(doc)
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn historian_report_views_preserve_sources_and_distinguish_generation_from_application() {
        let report = sample_report();
        let encoded = serde_json::to_string(&report).unwrap();
        assert_eq!(parse_report(&encoded), Some(report.clone()));
        assert!(parse_report("## History prepared\nLegacy report").is_none());
        assert!(parse_report(&encoded.replace("historian_report_v1", "unknown_report")).is_none());
        let tr = Translator::new(Language::En);
        let compact = plain(&render_report(
            &report,
            ReportOptions::default(),
            Theme::dark(),
            100,
            &tr,
        ));
        assert!(compact.contains("Report prepared") && compact.contains("tracked separately"));
        assert!(compact.contains("Compact-only result") && !compact.contains("Detailed-only"));
        assert!(!compact.contains("raw:12") && !compact.contains("old-fact"));
        assert!(compact.contains("Input 120 tokens") && !compact.contains("Input 220"));
        assert!(compact.contains("Cached input: Not reported"));
        for (view, expected) in [
            (ReportView::Detailed, "Detailed-only"),
            (ReportView::Anchor, "Anchor-only"),
        ] {
            let doc = render_report(
                &report,
                ReportOptions {
                    view,
                    sources: true,
                },
                Theme::dark(),
                100,
                &tr,
            );
            assert!(doc.validate());
            let output = plain(&doc);
            assert!(output.contains(expected) && !output.contains("Compact-only"));
            assert!(
                output.contains("raw:12")
                    && output.contains("old-fact")
                    && output.contains("session:parent-example/main")
            );
        }
        let raw = plain(&render_report(
            &report,
            ReportOptions {
                view: ReportView::Raw,
                sources: false,
            },
            Theme::dark(),
            100,
            &tr,
        ));
        assert!(
            raw.contains("historian_report_v1")
                && raw.contains("input_tokens")
                && raw.contains("Detailed-only")
        );
    }

    #[test]
    fn historian_report_layout_handles_narrow_widths_empty_sections_and_unicode() {
        let mut report = sample_report();
        report.publication.compartments[0].anchor.clear();
        report.publication.facts.clear();
        report.usage.clear();
        for language in [Language::En, Language::ZhCn] {
            let tr = Translator::new(language);
            for width in [1, 8, 20, 80] {
                for view in [
                    ReportView::Compact,
                    ReportView::Detailed,
                    ReportView::Anchor,
                    ReportView::Raw,
                ] {
                    let doc = render_report(
                        &report,
                        ReportOptions {
                            view,
                            sources: true,
                        },
                        Theme::dark(),
                        width,
                        &tr,
                    );
                    assert!(doc.validate(), "{width} {view:?}");
                    // A single wide glyph occupies two cells even at width one.
                    for line in transcript_ratatui::document_to_ratatui(&doc) {
                        assert!(
                            display_width(&line.to_string()) <= width.max(2),
                            "{width}: {line}"
                        );
                    }
                }
            }
            let doc = render_report(
                &report,
                ReportOptions {
                    view: ReportView::Anchor,
                    sources: false,
                },
                Theme::dark(),
                100,
                &tr,
            );
            let output = plain(&doc);
            assert!(output.contains(&tr.t("historian.title_anchor")));
            assert!(output.contains(&tr.t("historian.unreported")));
            assert!(!output.contains(&tr.t("historian.fact_changes")));
        }
    }
}
