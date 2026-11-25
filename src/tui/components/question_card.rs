//! Question-response tool card rendering.

use ratatui::style::{Modifier, Style};
use serde_json::Value;

use super::semantic_spans::*;
use crate::tui::{
    measure::wrap_text_to_width,
    theme::Theme,
    timeline::{ToolExecutionStatus, ToolView},
    transcript_render::{Break, SemanticLine},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct QuestionResponseCard {
    header: Option<String>,
    question: String,
    answers: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct QuestionResponseCards {
    cards: Vec<QuestionResponseCard>,
}

pub(super) fn render_question_response_lines(
    tool: &ToolView,
    theme: Theme,
    width: usize,
) -> Vec<SemanticLine<Style>> {
    if tool.status != ToolExecutionStatus::Succeeded || width <= 2 {
        return Vec::new();
    }

    if let Some(responses) = question_response_cards(tool) {
        return render_question_cards(&responses, theme, width);
    }

    let data = tool_output_data(tool);
    let message = data
        .as_ref()
        .and_then(|data| data.get("message"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|message| !message.is_empty());
    let Some(message) = message else {
        return Vec::new();
    };

    let mut lines = question_card_header_lines(theme, width);
    append_question_card_text(
        &mut lines,
        message,
        question_answer_style(theme),
        theme,
        width,
    );
    finish_question_card(lines, theme, width)
}

pub(super) fn question_response_cards(tool: &ToolView) -> Option<QuestionResponseCards> {
    let data = tool_output_data(tool)?;
    let arguments = tool_arguments(tool)?;
    let questions = arguments.get("questions")?.as_array()?;
    let answers = data.get("answers")?.as_array()?;
    if questions.len() != answers.len() || questions.is_empty() {
        return None;
    }
    let cards = questions
        .iter()
        .zip(answers)
        .map(|(question_value, answers)| {
            let question = question_value.get("question")?.as_str()?.trim();
            if question.is_empty() {
                return None;
            }
            Some(QuestionResponseCard {
                header: question_value
                    .get("header")
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|header| !header.is_empty())
                    .map(str::to_string),
                question: question.to_string(),
                answers: question_answer_strings(answers)?,
            })
        })
        .collect::<Option<Vec<_>>>()?;
    Some(QuestionResponseCards { cards })
}

pub(super) fn question_answer_strings(value: &Value) -> Option<Vec<String>> {
    value
        .as_array()?
        .iter()
        .map(Value::as_str)
        .map(|answer| {
            answer
                .map(str::trim)
                .filter(|answer| !answer.is_empty())
                .map(str::to_string)
        })
        .collect()
}

pub(super) fn render_question_cards(
    responses: &QuestionResponseCards,
    theme: Theme,
    width: usize,
) -> Vec<SemanticLine<Style>> {
    let mut lines = question_card_header_lines(theme, width);
    for (index, card) in responses.cards.iter().enumerate() {
        if let Some(header) = card.header.as_deref() {
            append_question_card_text(
                &mut lines,
                header,
                question_header_style(theme),
                theme,
                width,
            );
        }
        append_question_card_text(
            &mut lines,
            &card.question,
            question_text_style(theme),
            theme,
            width,
        );
        lines.push(question_card_decoration_line(
            "",
            question_answer_style(theme),
            theme,
            width,
        ));
        let answer = if card.answers.is_empty() {
            "(no answer)".to_string()
        } else {
            card.answers.join(" · ")
        };
        append_question_card_text(
            &mut lines,
            &answer,
            question_answer_style(theme),
            theme,
            width,
        );
        if index + 1 < responses.cards.len() {
            lines.push(question_card_decoration_line(
                "",
                question_text_style(theme),
                theme,
                width,
            ));
        }
    }
    finish_question_card(lines, theme, width)
}

pub(super) fn finish_question_card(
    mut lines: Vec<SemanticLine<Style>>,
    theme: Theme,
    width: usize,
) -> Vec<SemanticLine<Style>> {
    lines.push(question_card_decoration_line(
        "",
        question_text_style(theme),
        theme,
        width,
    ));
    lines
}

pub(super) fn question_card_header_lines(theme: Theme, width: usize) -> Vec<SemanticLine<Style>> {
    vec![
        question_card_decoration_line("", question_text_style(theme), theme, width),
        question_card_decoration_line("# User response", question_title_style(theme), theme, width),
        question_card_decoration_line("", question_text_style(theme), theme, width),
    ]
}

pub(super) fn append_question_card_text(
    lines: &mut Vec<SemanticLine<Style>>,
    text: &str,
    style: Style,
    theme: Theme,
    width: usize,
) {
    let content_width = shell_card_content_width(width);
    for source_line in text.split('\n') {
        let wrapped = wrap_text_to_width(source_line, content_width);
        let last = wrapped.len().saturating_sub(1);
        lines.extend(wrapped.into_iter().enumerate().map(|(index, line)| {
            question_card_line_with_boundary(
                &line,
                style,
                theme,
                width,
                if index == last {
                    Break::HardBreak
                } else {
                    Break::SoftWrap
                },
            )
        }));
    }
}

pub(super) fn question_card_line_with_boundary(
    text: &str,
    style: Style,
    theme: Theme,
    width: usize,
    boundary: Break,
) -> SemanticLine<Style> {
    render_source_card_line_with_boundary(
        &[(text.to_string(), style)],
        Style::default().bg(theme.element_bg),
        theme,
        width,
        boundary,
    )
}

pub(super) fn question_card_decoration_line(
    text: &str,
    style: Style,
    theme: Theme,
    width: usize,
) -> SemanticLine<Style> {
    render_card_line(
        &[(text.to_string(), style)],
        Style::default().bg(theme.element_bg),
        theme,
        width,
    )
}

pub(super) fn question_text_style(theme: Theme) -> Style {
    Style::default().fg(theme.muted_text).bg(theme.element_bg)
}

pub(super) fn question_header_style(theme: Theme) -> Style {
    Style::default().fg(theme.muted_text).bg(theme.element_bg)
}

pub(super) fn question_title_style(theme: Theme) -> Style {
    Style::default()
        .fg(theme.accent)
        .bg(theme.element_bg)
        .add_modifier(Modifier::BOLD)
}

pub(super) fn question_answer_style(theme: Theme) -> Style {
    Style::default().fg(theme.text).bg(theme.element_bg)
}
