use super::*;
use crate::context_history::{HistoryCompartment, HistoryFact, HistoryPublication};
use crate::historian::HistorianReportKind;
use crate::tui::{i18n::Language, measure::display_width, transcript_ratatui};

pub(crate) fn sample_report() -> HistorianReport {
    HistorianReport {
        kind: HistorianReportKind::Prepared,
        publication: HistoryPublication {
            id: "publication-example".into(),
            project_path: None,
            external_fact_ids: vec!["old-fact".into()],
            source_ids: vec!["raw:12".into(), "raw:14".into()],
            compartments: vec![HistoryCompartment {
                id: "episode-example".into(),
                title: "清理摘要块 · transcript".into(),
                source_ids: vec!["raw:12".into(), "raw:14".into()],
                importance: 80,
                detailed:
                    "Detailed-only rationale\n\nA second paragraph preserves the exact constraint."
                        .into(),
                compact: "Compact-only result".into(),
                anchor: "Anchor-only clue".into(),
            }],
            facts: vec![HistoryFact {
                id: "fact-example".into(),
                category: HistoryFactCategory::Architecture,
                text: "Use footer status without blank summary blocks.".into(),
                source_ids: vec!["raw:14".into()],
                supersedes: vec!["old-fact".into()],
            }],
            withdrawn_fact_ids: vec![],
        },
        source_session_id: "parent-example".into(),
        source_branch_id: "main".into(),
        model: "provider/model".into(),
        elapsed_ms: 8400,
        usage: vec![
            UsageUpdate::Usage {
                input_tokens: 100,
                output_tokens: 10,
                total_tokens: 110,
                reasoning_tokens: None,
                cached_input_tokens: None,
            },
            UsageUpdate::Usage {
                input_tokens: 120,
                output_tokens: 20,
                total_tokens: 140,
                reasoning_tokens: None,
                cached_input_tokens: None,
            },
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
fn historian_report_views_preserve_sources_and_usage() {
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
    assert!(compact.contains("Report prepared"));
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
fn historian_report_handles_empty_visual_chunks_without_invalid_source_ranges() {
    let mut report = sample_report();
    report.publication.compartments[0].title = "\nTitle with a blank line\n".into();
    let doc = render_report(
        &report,
        ReportOptions::default(),
        Theme::dark(),
        80,
        &Translator::new(Language::En),
    );

    assert!(doc.validate());
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
