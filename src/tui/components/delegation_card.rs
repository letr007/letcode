//! Delegation input card rendered inside a child session view.

use ratatui::style::{Modifier, Style};

use crate::tui::{
    i18n::Translator,
    measure::{display_width, wrap_text_to_width_with_offsets},
    surface,
    theme::Theme,
    timeline::DelegationView,
    transcript_render::{Break, Document, Line, SourceRange, Span},
};

const GUTTER_WIDTH: usize = 3;

pub fn render_delegation_card_document(
    delegation: &DelegationView,
    theme: Theme,
    width: usize,
    translator: &Translator,
) -> Document<Style> {
    let mut document = Document::default();
    let width = width.saturating_sub(GUTTER_WIDTH);
    if width == 0 {
        return document;
    }

    push_delegation_line(
        &mut document,
        "",
        &translator.t("subagent.delegation"),
        heading_style(theme),
        theme,
        width,
    );
    push_delegation_field(
        &mut document,
        &translator.t("subagent.field_objective"),
        std::slice::from_ref(&delegation.objective),
        theme,
        width,
    );
    push_delegation_field(
        &mut document,
        &translator.t("subagent.field_success_criteria"),
        &delegation.success_criteria,
        theme,
        width,
    );
    push_delegation_field(
        &mut document,
        &translator.t("subagent.field_owned_paths"),
        &delegation.owned_paths,
        theme,
        width,
    );
    push_delegation_field(
        &mut document,
        &translator.t("subagent.field_allowed_paths"),
        &delegation.allowed_paths,
        theme,
        width,
    );
    push_delegation_field(
        &mut document,
        &translator.t("subagent.field_forbidden_paths"),
        &delegation.forbidden_paths,
        theme,
        width,
    );
    push_delegation_field(
        &mut document,
        &translator.t("subagent.field_model"),
        delegation.model.as_slice(),
        theme,
        width,
    );
    push_delegation_field(
        &mut document,
        &translator.t("subagent.field_takeover"),
        delegation.target_child_session_id.as_slice(),
        theme,
        width,
    );

    document.finish();
    debug_assert!(document.validate());
    document
}

fn push_delegation_field(
    document: &mut Document<Style>,
    label: &str,
    values: &[String],
    theme: Theme,
    width: usize,
) {
    let mut values = values.iter().filter(|value| !value.is_empty());
    let Some(first) = values.next() else {
        return;
    };
    push_delegation_line(
        document,
        &format!("{label}  "),
        first,
        value_style(theme),
        theme,
        width,
    );
    for value in values {
        push_delegation_line(document, "· ", value, value_style(theme), theme, width);
    }
}

fn push_delegation_line(
    document: &mut Document<Style>,
    prefix: &str,
    text: &str,
    text_style: Style,
    theme: Theme,
    width: usize,
) {
    let prefix_chars = prefix.chars().collect::<Vec<_>>();
    let text_chars = text.chars().collect::<Vec<_>>();
    let full = prefix_chars
        .iter()
        .chain(text_chars.iter())
        .collect::<String>();
    let block = document.add_source(full.clone());
    let chunks = wrap_text_to_width_with_offsets(&full, width);
    let guide = Style::default().fg(theme.card_guide()).bg(theme.root_bg);
    let fill = Style::default().bg(theme.root_bg);

    for (index, chunk) in chunks.iter().enumerate() {
        let mut spans = vec![
            Span::decoration(surface::ACCENT_BAR_GLYPH, guide),
            Span::decoration("  ", fill),
        ];
        let prefix_len = prefix_chars.len();
        let start = chunk.source_start_char;
        let end = chunk.source_end_char;
        let prefix_end = prefix_len.min(end);
        if start < prefix_end {
            spans.push(Span::source(
                prefix_chars[start..prefix_end].iter().collect::<String>(),
                label_style(theme),
                SourceRange::new(block, start, prefix_end),
            ));
        }
        let text_start = start.max(prefix_len);
        if text_start < end {
            spans.push(Span::source(
                text_chars[text_start - prefix_len..end - prefix_len]
                    .iter()
                    .collect::<String>(),
                text_style,
                SourceRange::new(block, text_start, end),
            ));
        }
        let used = spans.iter().map(|span| display_width(&span.text)).sum();
        if width > used {
            spans.push(Span::decoration(" ".repeat(width - used), fill));
        }
        document.push_line(
            Line { spans },
            if index + 1 == chunks.len() {
                Break::HardBreak
            } else {
                Break::SoftWrap
            },
        );
    }
}

fn heading_style(theme: Theme) -> Style {
    Style::default()
        .fg(theme.notice)
        .bg(theme.root_bg)
        .add_modifier(Modifier::BOLD)
}

fn label_style(theme: Theme) -> Style {
    Style::default().fg(theme.muted_text).bg(theme.root_bg)
}

fn value_style(theme: Theme) -> Style {
    Style::default().fg(theme.text).bg(theme.root_bg)
}

#[cfg(test)]
mod tests {
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
        for width in [10usize, 18, 32, 80] {
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
    }

    fn translator() -> Translator {
        Translator::new(Language::ZhCn)
    }
}
