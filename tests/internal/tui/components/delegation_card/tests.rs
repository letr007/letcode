use super::render_delegation_card_document;
use crate::tui::{
    i18n::{Language, Translator},
    measure::display_width,
    theme::Theme,
    timeline::DelegationView,
    transcript_ratatui::document_to_ratatui,
};

fn view() -> DelegationView {
    DelegationView {
        objective: "完成结构化委派块渲染".into(),
        success_criteria: vec!["门禁全绿".into(), "零回归".into()],
        owned_paths: vec!["src/tui".into()],
        allowed_paths: Vec::new(),
        forbidden_paths: vec!["docs".into()],
        model: Some("NewAPI/deepseek-flash".into()),
        target_child_session_id: Some("child-1".into()),
    }
}

#[test]
fn delegation_card_respects_display_width_with_cjk_content() {
    for width in [0usize, 1, 2, 3, 4, 10, 18, 32, 80] {
        let document =
            render_delegation_card_document(&view(), Theme::dark(), width, &translator());
        assert!(document.validate());
        for line in document_to_ratatui(&document) {
            let rendered = line.to_string();
            assert!(
                display_width(&rendered) <= width,
                "line width exceeds {width}: {rendered:?}"
            );
        }
    }
}

#[test]
fn delegation_card_renders_delegated_values() {
    let document = render_delegation_card_document(&view(), Theme::dark(), 80, &translator());
    let copied = document
        .lines
        .iter()
        .flat_map(|line| line.spans.iter())
        .filter(|span| span.source.is_some())
        .map(|span| span.text.as_str())
        .collect::<String>();
    assert!(copied.contains("完成结构化委派块渲染"));
    assert!(copied.contains("NewAPI/deepseek-flash"));
    assert!(
        !document
            .source_blocks
            .iter()
            .any(|block| block.source == translator().t("subagent.delegation"))
    );
    assert!(!copied.contains(&translator().t("subagent.field_objective")));
}

fn translator() -> Translator {
    Translator::new(Language::ZhCn)
}
