//! Compact chat-style cards for the sticky reviewer child view.
//!
//! Only used when viewing the `reviewer` child session — turns the raw approval
//! prompt / JSON reply into request + decision cards.

use ratatui::style::{Color, Modifier, Style};
use serde::Deserialize;
use serde_json::Value;

use crate::tui::{
    measure::{display_width, wrap_text_to_width_with_offsets},
    surface,
    theme::Theme,
    transcript_render::{Break, Document, Line, SourceRange, Span},
};

use super::composer::one_line_snippet;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ReviewRequestCard {
    pub tool: String,
    pub class: Option<String>,
    pub directive: Option<String>,
    pub summary: String,
    pub goal: Option<String>,
    pub can_allow_always: bool,
    pub preview: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ReviewDecisionCard {
    pub decision: String,
    pub risk: Option<String>,
    pub rationale: String,
}

pub(crate) fn looks_like_review_request(text: &str) -> bool {
    let trimmed = text.trim_start();
    trimmed.starts_with("Approve or deny this tool permission request.")
        && trimmed.contains("\nTool: ")
}

pub(crate) fn parse_review_request(text: &str) -> Option<ReviewRequestCard> {
    if !looks_like_review_request(text) {
        return None;
    }
    let tool = field_after(text, "Tool: ")?;
    let class = field_after(text, "Class: ");
    let directive = field_after(text, "Execution directive: ")
        .filter(|value| !value.eq_ignore_ascii_case("none"));
    let summary = field_after(text, "Summary: ").unwrap_or_else(|| tool.clone());
    let preview = field_after(text, "Preview: ").filter(|value| value != "(none)");
    let can_allow_always = field_after(text, "can_allow_always: ")
        .map(|value| value.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let goal = section_after(text, "User goal:\n")
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty() && value != "(not provided)");
    Some(ReviewRequestCard {
        tool,
        class,
        directive,
        summary,
        goal,
        can_allow_always,
        preview,
    })
}

pub(crate) fn parse_review_decision(text: &str) -> Option<ReviewDecisionCard> {
    let candidate = extract_json_object(text)?;
    let parsed: ReviewerJson = serde_json::from_str(candidate).ok()?;
    let decision = match parsed.decision.trim().to_ascii_lowercase().as_str() {
        "allow_always" | "always" => "allow_once".to_string(),
        "allow_once" | "allow" | "once" | "deny" | "reject" => {
            parsed.decision.trim().to_ascii_lowercase()
        }
        _ => return None,
    };
    let rationale = parsed
        .rationale
        .map(|text| text.trim().to_string())
        .filter(|text| !text.is_empty())
        .unwrap_or_else(|| "no rationale".into());
    Some(ReviewDecisionCard {
        decision,
        risk: parsed
            .risk
            .map(|text| text.trim().to_ascii_lowercase())
            .filter(|text| !text.is_empty()),
        rationale,
    })
}

pub(crate) fn render_review_request_card_document(
    card: &ReviewRequestCard,
    theme: Theme,
    width: usize,
) -> Document<Style> {
    let mut document = Document::default();
    if width == 0 {
        return document;
    }
    let accent = theme.notice;
    let title_style = Style::default()
        .fg(accent)
        .bg(theme.root_bg)
        .add_modifier(Modifier::BOLD);
    let content_style = Style::default().fg(theme.text).bg(theme.root_bg);
    let detail_style = Style::default().fg(theme.muted_text).bg(theme.root_bg);

    let class = card.class.as_deref().unwrap_or("tool");
    push_card_line(
        &mut document,
        &format!("request  {} · {}", card.tool, class),
        title_style,
        accent,
        theme,
        width,
    );
    push_card_line(
        &mut document,
        &one_line_snippet(&card.summary, width.saturating_sub(3)),
        content_style,
        accent,
        theme,
        width,
    );

    let mut meta = Vec::new();
    if card.can_allow_always {
        meta.push("always ok".to_string());
    } else {
        meta.push("once only".to_string());
    }
    if let Some(directive) = card.directive.as_deref() {
        meta.push(directive.to_string());
    }
    if let Some(preview) = card.preview.as_deref() {
        meta.push(one_line_snippet(preview, 40));
    }
    push_card_line(
        &mut document,
        &meta.join(" · "),
        detail_style,
        accent,
        theme,
        width,
    );

    if let Some(goal) = card.goal.as_deref() {
        push_card_line(
            &mut document,
            &format!("goal  {}", one_line_snippet(goal, width.saturating_sub(8))),
            detail_style,
            accent,
            theme,
            width,
        );
    }

    document.finish();
    debug_assert!(document.validate());
    document
}

pub(crate) fn render_review_decision_card_document(
    card: &ReviewDecisionCard,
    theme: Theme,
    width: usize,
) -> Document<Style> {
    let mut document = Document::default();
    if width == 0 {
        return document;
    }

    let (label, accent) = match card.decision.as_str() {
        "allow_once" | "allow" | "once" | "allow_always" | "always" => {
            ("allow once", theme.success)
        }
        "deny" | "reject" => ("deny", theme.error),
        other => (other, theme.notice),
    };
    let title_style = Style::default()
        .fg(accent)
        .bg(theme.root_bg)
        .add_modifier(Modifier::BOLD);
    let content_style = Style::default().fg(theme.text).bg(theme.root_bg);
    let detail_style = Style::default().fg(theme.muted_text).bg(theme.root_bg);

    let title = match card.risk.as_deref() {
        Some(risk) => format!("decide   {label} · {risk}"),
        None => format!("decide   {label}"),
    };
    push_card_line(&mut document, &title, title_style, accent, theme, width);
    push_card_line(
        &mut document,
        &one_line_snippet(&card.rationale, width.saturating_sub(3)),
        content_style,
        accent,
        theme,
        width,
    );
    // Keep a muted second line so request/decision cards share the same density.
    push_card_line(
        &mut document,
        "reviewer",
        detail_style,
        accent,
        theme,
        width,
    );

    document.finish();
    debug_assert!(document.validate());
    document
}

fn push_card_line(
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
    for chunk in &chunks {
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
        document.push_line(Line { spans }, Break::SoftWrap);
    }
}

fn field_after(text: &str, key: &str) -> Option<String> {
    let start = text.find(key)? + key.len();
    let rest = &text[start..];
    let end = rest.find('\n').unwrap_or(rest.len());
    let value = rest[..end].trim();
    (!value.is_empty()).then(|| value.to_string())
}

fn section_after(text: &str, key: &str) -> Option<String> {
    let start = text.find(key)? + key.len();
    let rest = &text[start..];
    let end = rest
        .find("\n\nTool: ")
        .or_else(|| rest.find("\nTool: "))
        .unwrap_or(rest.len());
    Some(rest[..end].to_string())
}

#[derive(Debug, Deserialize)]
struct ReviewerJson {
    decision: String,
    #[serde(default)]
    risk: Option<String>,
    #[serde(default)]
    rationale: Option<String>,
}

fn extract_json_object(raw: &str) -> Option<&str> {
    let trimmed = raw.trim();
    if trimmed.starts_with('{')
        && let Ok(Value::Object(_)) = serde_json::from_str::<Value>(trimmed)
    {
        return Some(trimmed);
    }
    let start = trimmed.find('{')?;
    let end = trimmed.rfind('}')?;
    if end <= start {
        return None;
    }
    let slice = &trimmed[start..=end];
    matches!(serde_json::from_str::<Value>(slice), Ok(Value::Object(_))).then_some(slice)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_REQUEST: &str = "Approve or deny this tool permission request.\n\
         \n\
         User goal:\nExecute commands to test auto-approval\n\
         \n\
         Tool: shell__exec\n\
         Class: command\n\
         Execution directive: read-only\n\
         Summary: shell__exec python3 -c \"...\"\n\
         Preview: (none)\n\
         can_allow_always: false\n\
         Arguments:\n{\n  \"command\": \"python3\"\n}\n\
         \n\
         Reply with ONLY JSON:\n\
         {\"decision\":\"allow_once|deny\",\"risk\":\"low|medium|high\",\"rationale\":\"...\"}\n";

    #[test]
    fn parses_review_request_fields() {
        let card = parse_review_request(SAMPLE_REQUEST).expect("request");
        assert_eq!(card.tool, "shell__exec");
        assert_eq!(card.class.as_deref(), Some("command"));
        assert_eq!(card.directive.as_deref(), Some("read-only"));
        assert!(!card.can_allow_always);
        assert_eq!(
            card.goal.as_deref(),
            Some("Execute commands to test auto-approval")
        );
    }

    #[test]
    fn invalid_allow_always_decision_is_rendered_as_allow_once() {
        let card =
            parse_review_decision(r#"{"decision":"allow_always","risk":"low","rationale":"safe"}"#)
                .expect("decision");
        assert_eq!(card.decision, "allow_once");
    }

    #[test]
    fn parses_decision_json_from_assistant_text() {
        let card = parse_review_decision(
            "here\n{\"decision\":\"allow_once\",\"risk\":\"low\",\"rationale\":\"safe https fetch\"}\n",
        )
        .expect("decision");
        assert_eq!(card.decision, "allow_once");
        assert_eq!(card.risk.as_deref(), Some("low"));
        assert!(card.rationale.contains("safe https"));
    }
}
