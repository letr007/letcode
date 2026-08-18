use ratatui::style::{Color, Modifier, Style};
#[cfg(test)]
use ratatui::text::Line;
use serde_json::Value;
use unicode_segmentation::UnicodeSegmentation;

use crate::agent::{agent_name_for_subagent_tool, is_subagent_tool_name};
use crate::subagent::StructuredSubagentResult;
use crate::tui::{
    measure::{display_width, wrap_text_to_width},
    presentation::{
        PresentationPolicy, ToolPresentation, ToolPresentationStatus, ToolTextPresentationContext,
    },
    surface,
    theme::Theme,
    timeline::{PermissionPromptStatus, PermissionView, ToolExecutionStatus, ToolView},
    transcript_render::{Break, CopyJoin, Document, SemanticLine, SemanticSpan},
};

const TOOL_GUIDE_GLYPH: &str = surface::ACCENT_BAR_GLYPH;
const COMPACT_SHELL_BODY_LINES: usize = 20;
const DIFF_CARD_HEADER_ARROW: &str = "←";
const PROCESS_FRAMES: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolCardStatus {
    Pending,
    Approved,
    Running,
    Cancelled,
    Succeeded,
    Failed,
    Denied,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCardDetails {
    pub title: String,
    pub status: ToolCardStatus,
    pub summary: String,
    pub call_id: Option<String>,
    /// Compact single-line args snippet.
    pub arguments: Option<String>,
    /// Compact single-line output/error snippet.
    pub output: Option<String>,
    /// Additional compact fields to preserve safety context.
    pub fields: Vec<(String, String)>,
}

impl ToolCardDetails {
    pub fn new(
        title: impl Into<String>,
        status: ToolCardStatus,
        summary: impl Into<String>,
    ) -> Self {
        Self {
            title: title.into(),
            status,
            summary: summary.into(),
            call_id: None,
            arguments: None,
            output: None,
            fields: Vec::new(),
        }
    }
}

/// Pure view logic: derive a compact card model for a ToolView.
/// Returns None when PresentationPolicy wants the tool hidden.
pub fn tool_card_details(tool: &ToolView, policy: &PresentationPolicy) -> Option<ToolCardDetails> {
    let status = match tool.status {
        ToolExecutionStatus::Pending => ToolPresentationStatus::Pending,
        ToolExecutionStatus::Running => ToolPresentationStatus::Running,
        ToolExecutionStatus::Cancelled => ToolPresentationStatus::Failed,
        ToolExecutionStatus::Succeeded => ToolPresentationStatus::Succeeded,
        ToolExecutionStatus::Failed => ToolPresentationStatus::Failed,
    };

    let mut ctx = ToolTextPresentationContext::new(&tool.name, status);
    ctx.arguments = tool.arguments.clone();
    ctx.output = tool.output.clone();

    let presentation = policy.tool_presentation_text(&ctx);
    if presentation == ToolPresentation::Hidden {
        return None;
    }

    let mut details = ToolCardDetails::new(
        tool.name.clone(),
        map_tool_status(tool.status),
        tool.summary.clone(),
    );

    // Hide verbose fields by default; reveal only a tiny snippet when it materially helps.
    match tool.status {
        ToolExecutionStatus::Pending => {}
        ToolExecutionStatus::Running => {
            details.arguments = tool
                .arguments
                .as_deref()
                .map(one_line_snippet)
                .filter(|s| !s.is_empty());
        }
        ToolExecutionStatus::Cancelled => {
            details.output = tool
                .output
                .as_deref()
                .map(one_line_snippet)
                .filter(|s| !s.is_empty());
        }
        ToolExecutionStatus::Failed => {
            details.output = tool
                .output
                .as_deref()
                .map(one_line_snippet)
                .filter(|s| !s.is_empty());
        }
        ToolExecutionStatus::Succeeded => {
            // Keep succeeded cards compact by default.
        }
    }

    // Always show call id for audit trail.
    details.call_id = Some(tool.call_id.clone());

    Some(details)
}

#[cfg(test)]
pub fn permission_card_details(permission: &PermissionView) -> ToolCardDetails {
    let mut details = ToolCardDetails::new(
        permission.tool_name.clone(),
        match permission.status {
            PermissionPromptStatus::Pending => ToolCardStatus::Pending,
            PermissionPromptStatus::Approved => ToolCardStatus::Approved,
            PermissionPromptStatus::Denied => ToolCardStatus::Denied,
        },
        permission.summary.clone(),
    );

    // Always show call id for audit/safety context.
    details.call_id = Some(permission.call_id.clone());

    // Preserve compact safety context without dumping raw JSON/content payloads.
    details.arguments = permission
        .arguments
        .as_deref()
        .map(|args| permission_arguments_summary(&permission.tool_name, args))
        .filter(|s| !s.is_empty());

    if let Some(why) = permission
        .rationale
        .as_deref()
        .map(one_line_snippet)
        .filter(|s| !s.is_empty())
    {
        details.fields.push(("why".into(), why));
    }

    // For denial, include a compact resolution reason.
    if let Some(reason) = permission
        .resolution_reason
        .as_deref()
        .map(one_line_snippet)
        .filter(|s| !s.is_empty())
    {
        details.fields.push(("resolution".into(), reason));
    }
    details
}

/// Render a compact tool card into pre-wrapped transcript lines.
///
/// The caller is responsible for inserting blank spacer lines between timeline items.
#[cfg(test)]
pub(crate) fn render_tool_card_lines(
    tool: &ToolView,
    theme: Theme,
    width: usize,
) -> Vec<Line<'static>> {
    render_tool_card_lines_with_frame(tool, theme, width, 0, false)
}

#[cfg(test)]
pub(crate) fn render_tool_card_lines_with_frame(
    tool: &ToolView,
    theme: Theme,
    width: usize,
    frame: usize,
    expanded_output: bool,
) -> Vec<Line<'static>> {
    crate::tui::transcript_ratatui::document_to_ratatui(&render_tool_card_document(
        tool,
        theme,
        width,
        frame,
        expanded_output,
    ))
}

/// Renderer-neutral tool document. The existing card layout remains visual-only;
/// its semantic text is registered as leaves before the Ratatui bridge consumes it.
pub fn render_tool_card_document(
    tool: &ToolView,
    theme: Theme,
    width: usize,
    frame: usize,
    expanded_output: bool,
) -> Document<Style> {
    let policy = PresentationPolicy;
    let mut document = Document::default();
    if tool_card_details(tool, &policy).is_none() {
        return document;
    }
    let lines = if is_subagent_tool(&tool.name) {
        render_subagent_lines(tool, theme, width, frame, expanded_output)
    } else {
        let body = render_tool_body_lines(tool, theme, width, expanded_output);
        if body.is_empty() {
            vec![render_tool_trace_line(tool, theme, width, frame)]
        } else {
            body
        }
    };
    for line in lines {
        document.push_semantic_line(line);
    }
    document.finish();
    debug_assert!(document.validate());
    document
}

#[cfg(test)]
pub(crate) fn render_permission_card_lines(
    permission: &PermissionView,
    theme: Theme,
    width: usize,
) -> Vec<Line<'static>> {
    crate::tui::transcript_ratatui::document_to_ratatui(&render_permission_card_document(
        permission, theme, width,
    ))
}

pub fn render_permission_card_document(
    permission: &PermissionView,
    theme: Theme,
    width: usize,
) -> Document<Style> {
    let mut document = Document::default();
    if width == 0 {
        return document;
    }

    let translator = crate::tui::i18n::Translator::new(crate::tui::i18n::Language::En);
    let status = status_label(
        match permission.status {
            PermissionPromptStatus::Pending => ToolCardStatus::Pending,
            PermissionPromptStatus::Approved => ToolCardStatus::Approved,
            PermissionPromptStatus::Denied => ToolCardStatus::Denied,
        },
        &translator,
    );
    let status_style = root_status_style(permission_accent(permission.status, theme), theme);
    let text_style = root_text_style(theme);
    let muted_style = root_muted_style(theme);
    let summary = permission
        .arguments
        .as_deref()
        .map(|args| permission_arguments_summary(&permission.tool_name, args))
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| permission.summary.clone());
    let reason = permission
        .resolution_reason
        .as_deref()
        .or(permission.rationale.as_deref())
        .map(one_line_snippet)
        .filter(|s| !s.is_empty());

    let mut segments = vec![
        SemanticSpan::decoration(status, status_style),
        SemanticSpan::decoration(" ", text_style),
    ];
    if let Some(origin) = permission
        .origin_label
        .as_deref()
        .map(str::trim)
        .filter(|label| !label.is_empty())
    {
        segments.push(SemanticSpan::decoration(
            origin.to_string(),
            muted_style.add_modifier(Modifier::BOLD),
        ));
        segments.push(SemanticSpan::decoration(" · ", muted_style));
    }
    segments.push(SemanticSpan::decoration(
        permission.tool_name.clone(),
        text_style.add_modifier(Modifier::BOLD),
    ));
    segments.push(SemanticSpan::decoration(" ", text_style));
    segments.push(SemanticSpan::source(summary, text_style));
    if let Some(reason) = reason {
        segments.push(SemanticSpan::decoration(" · ", muted_style));
        segments.push(SemanticSpan::source_with_join(
            reason,
            muted_style,
            CopyJoin::Space,
        ));
    }

    document.push_semantic_line(render_card_line_with_guide(
        &segments,
        Style::default().bg(theme.root_bg),
        permission_accent(permission.status, theme),
        theme,
        width,
        Break::End,
    ));
    debug_assert!(document.validate());
    document
}

fn render_tool_trace_line(
    tool: &ToolView,
    theme: Theme,
    width: usize,
    frame: usize,
) -> SemanticLine<Style> {
    if width == 0 {
        return SemanticLine::default();
    }

    let active = matches!(
        tool.status,
        ToolExecutionStatus::Pending | ToolExecutionStatus::Running
    );
    let hide_terminal_auto_continue_glyph = tool.name == "workflow__auto_continue" && !active;
    let glyph = if active {
        PROCESS_FRAMES[frame % PROCESS_FRAMES.len()]
    } else {
        "→"
    };
    let prefix = if hide_terminal_auto_continue_glyph {
        "  ".to_owned()
    } else {
        format!("  {glyph} ")
    };
    let arrow_style = tool_trace_arrow_style(tool.status, theme);
    let text_style = tool_trace_text_style(tool.status, theme);
    let status_suffix = if tool.name == crate::tool_names::TOOL_QUESTION {
        ""
    } else {
        match tool.status {
            ToolExecutionStatus::Pending => " …",
            ToolExecutionStatus::Running => " …",
            ToolExecutionStatus::Cancelled => " · cancelled",
            ToolExecutionStatus::Succeeded => "",
            ToolExecutionStatus::Failed => " · failed",
        }
    };
    let text_budget = width.saturating_sub(display_width(&prefix));
    let mut spans = vec![SemanticSpan::decoration("  ", theme.app_style())];
    if !hide_terminal_auto_continue_glyph {
        spans.push(SemanticSpan::decoration(format!("{glyph} "), arrow_style));
    }
    spans.extend(tool_trace_segments(tool, text_style));
    if !status_suffix.is_empty() {
        spans.push(SemanticSpan::decoration(status_suffix, text_style));
    }
    let mut line = SemanticLine {
        spans,
        boundary: Break::End,
    };
    line.spans = clip_semantic_spans(line.spans, text_budget);
    line
}

fn tool_trace_segments(tool: &ToolView, style: Style) -> Vec<SemanticSpan<Style>> {
    if tool.status == ToolExecutionStatus::Pending && tool.arguments.is_none() {
        return vec![SemanticSpan::decoration(
            pending_tool_trace_label(&tool.name),
            style,
        )];
    }

    let args = tool
        .arguments
        .as_deref()
        .and_then(|text| serde_json::from_str::<Value>(text).ok());
    let args = args.as_ref();
    match tool.name.as_str() {
        "fs__read" => {
            let path = value_str(args, "path").unwrap_or_else(|| fallback_tail(&tool.summary));
            let mut segments = trace_action_and_value("Read ", path, style);
            let mut fields = Vec::new();
            if let Some(offset) = value_u64(args, "offset") {
                fields.push(format!("offset={offset}"));
            }
            if let Some(limit) = value_u64(args, "limit") {
                fields.push(format!("limit={limit}"));
            }
            if !fields.is_empty() {
                segments.push(SemanticSpan::decoration(
                    format!(" [{}]", fields.join(", ")),
                    style,
                ));
            }
            segments
        }
        "fs__list" => trace_action_and_value(
            "List ",
            value_str(args, "path").unwrap_or_else(|| fallback_tail(&tool.summary)),
            style,
        ),
        "fs__write" => trace_action_and_value(
            "Write ",
            value_str(args, "path").unwrap_or_else(|| fallback_tail(&tool.summary)),
            style,
        ),
        "fs__append" => trace_action_and_value(
            "Append ",
            value_str(args, "path").unwrap_or_else(|| fallback_tail(&tool.summary)),
            style,
        ),
        "shell__exec" => trace_action_and_value(
            "Run ",
            value_str(args, "command").unwrap_or(tool.summary.as_str()),
            style,
        ),
        "search__rg" => {
            let pattern = value_str(args, "pattern").unwrap_or("pattern");
            let path = value_str(args, "path").unwrap_or(".");
            let mut segments = vec![
                SemanticSpan::decoration("Search ", style),
                SemanticSpan::source(pattern, style),
                SemanticSpan::decoration(" in ", style),
                SemanticSpan::source_with_join(path, style, CopyJoin::Space),
            ];
            // After completion, surface a compact result detail such as
            // "42 matches · 3 files · folded" in the same bracketed style as
            // fs__read's offset/limit.
            if tool.status == ToolExecutionStatus::Succeeded && !tool.summary.is_empty() {
                segments.push(SemanticSpan::decoration(
                    format!(" [{}]", tool.summary),
                    style,
                ));
            }
            segments
        }
        "web__fetch" => trace_action_and_value(
            "Fetch ",
            value_str(args, "url").unwrap_or_else(|| fallback_tail(&tool.summary)),
            style,
        ),
        _ => vec![SemanticSpan::decoration(tool_trace_label(tool), style)],
    }
}

fn trace_action_and_value(action: &str, value: &str, style: Style) -> Vec<SemanticSpan<Style>> {
    vec![
        SemanticSpan::decoration(action, style),
        SemanticSpan::source(value, style),
    ]
}

fn clip_semantic_spans(
    segments: Vec<SemanticSpan<Style>>,
    width: usize,
) -> Vec<SemanticSpan<Style>> {
    let mut remaining = width;
    let mut clipped = Vec::new();
    for segment in segments {
        if remaining == 0 {
            break;
        }
        if display_width(&segment.text) <= remaining {
            remaining = remaining.saturating_sub(display_width(&segment.text));
            clipped.push(segment);
            continue;
        }
        let text = truncate_display_width(&segment.text, remaining);
        let prefix = text.strip_suffix('…').unwrap_or(&text);
        if !prefix.is_empty() {
            clipped.push(if segment.copy {
                SemanticSpan::source_with_join(prefix, segment.style, segment.copy_join)
            } else {
                SemanticSpan::decoration(prefix, segment.style)
            });
        }
        if text.ends_with('…') {
            clipped.push(SemanticSpan::decoration("…", segment.style));
        }
        break;
    }
    clipped
}

fn render_tool_body_lines(
    tool: &ToolView,
    theme: Theme,
    width: usize,
    expanded_output: bool,
) -> Vec<SemanticLine<Style>> {
    if width == 0 || tool.status == ToolExecutionStatus::Pending {
        return Vec::new();
    }

    if tool.status == ToolExecutionStatus::Running && tool.name != "shell__exec" {
        return Vec::new();
    }

    match tool.name.as_str() {
        crate::tool_names::TOOL_QUESTION => {
            if tool.status == ToolExecutionStatus::Succeeded {
                render_question_response_lines(tool, theme, width)
            } else if tool.status == ToolExecutionStatus::Failed {
                render_generic_output_lines(tool, theme, width, expanded_output)
            } else {
                Vec::new()
            }
        }
        "fs__write" => render_write_diff_lines(tool, theme, width),
        "fs__append" => render_append_diff_lines(tool, theme, width),
        "shell__exec" => render_shell_output_lines(tool, theme, width, expanded_output),
        "edit__apply_patch" => render_edit_diff_lines(tool, theme, width),
        _ => {
            if tool.status == ToolExecutionStatus::Failed {
                render_generic_output_lines(tool, theme, width, expanded_output)
            } else {
                Vec::new()
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct QuestionResponseCard {
    header: Option<String>,
    question: String,
    answers: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct QuestionResponseCards {
    cards: Vec<QuestionResponseCard>,
    truncated: bool,
}

const QUESTION_CARD_MAX_LINES: usize = 24;
const QUESTION_CARD_TEXT_MAX_CHARS: usize = 512;

fn render_question_response_lines(
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
    let remaining = question_card_content_limit().saturating_sub(lines.len());
    let truncated = append_question_card_text(
        &mut lines,
        message,
        question_answer_style(theme),
        theme,
        width,
        remaining,
    );
    finish_question_card(lines, truncated, theme, width)
}

fn question_response_cards(tool: &ToolView) -> Option<QuestionResponseCards> {
    let data = tool_output_data(tool)?;
    let arguments = tool_arguments(tool)?;
    let questions = arguments.get("questions")?.as_array()?;
    let answers = data.get("answers")?.as_array()?;
    if questions.len() != answers.len() || questions.is_empty() {
        return None;
    }
    let truncated = questions.len() > question_card_line_limit();
    let cards = questions
        .iter()
        .zip(answers)
        .take(question_card_line_limit())
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
    Some(QuestionResponseCards { cards, truncated })
}

fn question_answer_strings(value: &Value) -> Option<Vec<String>> {
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

fn render_question_cards(
    responses: &QuestionResponseCards,
    theme: Theme,
    width: usize,
) -> Vec<SemanticLine<Style>> {
    let mut lines = question_card_header_lines(theme, width);
    let content_limit = question_card_content_limit();
    let mut truncated = responses.truncated;
    for (index, card) in responses.cards.iter().enumerate() {
        if lines.len() >= content_limit {
            truncated = true;
            break;
        }
        if let Some(header) = card.header.as_deref() {
            if lines.len() >= content_limit {
                truncated = true;
                break;
            }
            lines.push(question_card_line(
                header,
                question_header_style(theme),
                theme,
                width,
            ));
        }
        let remaining = content_limit.saturating_sub(lines.len());
        let question_truncated = append_question_card_text(
            &mut lines,
            &card.question,
            question_text_style(theme),
            theme,
            width,
            remaining,
        );
        truncated |= question_truncated;
        if lines.len() >= content_limit {
            truncated = true;
            break;
        }
        let answer = if card.answers.is_empty() {
            "(no answer)".to_string()
        } else {
            card.answers.join(" · ")
        };
        if lines.len() >= content_limit {
            truncated = true;
            break;
        }
        lines.push(question_card_line(
            "",
            question_answer_style(theme),
            theme,
            width,
        ));
        let remaining = content_limit.saturating_sub(lines.len());
        let answer_truncated = append_question_card_text(
            &mut lines,
            &answer,
            question_answer_style(theme),
            theme,
            width,
            remaining,
        );
        truncated |= answer_truncated;
        if index + 1 < responses.cards.len() && lines.len() < content_limit {
            lines.push(question_card_line(
                "",
                question_text_style(theme),
                theme,
                width,
            ));
        }
    }
    finish_question_card(lines, truncated, theme, width)
}

fn finish_question_card(
    mut lines: Vec<SemanticLine<Style>>,
    truncated: bool,
    theme: Theme,
    width: usize,
) -> Vec<SemanticLine<Style>> {
    let content_limit = question_card_content_limit();
    if truncated {
        if lines.len() >= content_limit {
            lines.pop();
        }
        lines.push(question_card_decoration_line(
            "… response truncated",
            question_header_style(theme),
            theme,
            width,
        ));
    }
    lines.push(question_card_decoration_line(
        "",
        question_text_style(theme),
        theme,
        width,
    ));
    lines
}

fn question_card_header_lines(theme: Theme, width: usize) -> Vec<SemanticLine<Style>> {
    vec![
        question_card_decoration_line("", question_text_style(theme), theme, width),
        question_card_decoration_line("# User response", question_title_style(theme), theme, width),
        question_card_decoration_line("", question_text_style(theme), theme, width),
    ]
}

fn append_question_card_text(
    lines: &mut Vec<SemanticLine<Style>>,
    text: &str,
    style: Style,
    theme: Theme,
    width: usize,
    limit: usize,
) -> bool {
    if width <= 2 || limit == 0 {
        return true;
    }
    let content_width = shell_card_content_width(width);
    let (text, text_truncated) = question_card_text(text);
    let wrapped = wrap_text_to_width(&text, content_width);
    let wrapped_truncated = wrapped.len() > limit;
    let visible_count = wrapped.len().min(limit);
    lines.extend(
        wrapped
            .into_iter()
            .take(limit)
            .enumerate()
            .map(|(index, line)| {
                question_card_line_with_boundary(
                    &line,
                    style,
                    theme,
                    width,
                    if index + 1 == visible_count {
                        Break::HardBreak
                    } else {
                        Break::SoftWrap
                    },
                )
            }),
    );
    text_truncated || wrapped_truncated
}

fn question_card_line(text: &str, style: Style, theme: Theme, width: usize) -> SemanticLine<Style> {
    question_card_line_with_boundary(text, style, theme, width, Break::HardBreak)
}

fn question_card_line_with_boundary(
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

fn question_card_decoration_line(
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

fn question_text_style(theme: Theme) -> Style {
    Style::default()
        .fg(theme.text)
        .bg(theme.element_bg)
        .add_modifier(Modifier::BOLD)
}

fn question_header_style(theme: Theme) -> Style {
    Style::default().fg(theme.muted_text).bg(theme.element_bg)
}

fn question_title_style(theme: Theme) -> Style {
    Style::default()
        .fg(theme.accent)
        .bg(theme.element_bg)
        .add_modifier(Modifier::BOLD)
}

fn question_answer_style(theme: Theme) -> Style {
    Style::default().fg(theme.user).bg(theme.element_bg)
}

fn question_card_line_limit() -> usize {
    max_body_lines().min(QUESTION_CARD_MAX_LINES)
}

fn question_card_content_limit() -> usize {
    question_card_line_limit().saturating_sub(1)
}

fn question_card_text(text: &str) -> (String, bool) {
    let mut chars = text.chars();
    let visible = chars
        .by_ref()
        .take(QUESTION_CARD_TEXT_MAX_CHARS)
        .collect::<String>();
    if chars.next().is_some() {
        (format!("{visible}…"), true)
    } else {
        (visible, false)
    }
}

fn render_subagent_lines(
    tool: &ToolView,
    theme: Theme,
    width: usize,
    frame: usize,
    expanded_output: bool,
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
    let localized_status = crate::tui::state::TuiState::default().translator();
    let status = data
        .as_ref()
        .and_then(|data| data.get("status").and_then(serde_json::Value::as_str))
        .or_else(|| structured.as_ref().map(|result| result.status.as_str()))
        .map(str::to_owned)
        .unwrap_or_else(|| subagent_status_label(tool.status, &localized_status));
    let child_id = data
        .as_ref()
        .and_then(|data| data.get("child_session_id"))
        .and_then(serde_json::Value::as_str)
        .or_else(|| {
            structured
                .as_ref()
                .map(|result| result.child_session_id.as_str())
        })
        .map(|id| truncate_display_width(id, 16))
        .unwrap_or_else(|| "child".into());
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
        .unwrap_or_else(|| subagent_name_from_tool(&tool.name));
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
        format!(" [{}]", state_flags.join("/"))
    };

    let status_label = if matches!(
        tool.status,
        ToolExecutionStatus::Pending | ToolExecutionStatus::Running
    ) {
        format!(
            "{} {}",
            PROCESS_FRAMES[frame % PROCESS_FRAMES.len()],
            status_label(
                map_tool_status(tool.status),
                &crate::tui::state::TuiState::default().translator()
            )
        )
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
    lines.push(render_card_line_with_guide(
        &[
            SemanticSpan::decoration(format!("{status_label}{state_suffix}"), status_style),
            SemanticSpan::decoration(" ", text_style),
            SemanticSpan::decoration(
                agent_name.to_string(),
                text_style.add_modifier(Modifier::BOLD),
            ),
            SemanticSpan::decoration(" ", text_style),
            SemanticSpan::source(summary, text_style),
            SemanticSpan::decoration(" · ", muted),
            SemanticSpan::decoration(format!("/child {child_id}"), muted),
        ],
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

    let activity = subagent_activity_summary(&structured);
    if !expanded_output {
        let activity_label = if activity.is_empty() {
            "details · expand".to_string()
        } else {
            format!("{activity} · expand")
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
        "details · collapse",
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
            &format!("run {run_id}"),
            muted,
            theme,
            width,
            Break::HardBreak,
        ));
    }
    render_subagent_wrapped_field(
        &mut lines,
        "summary",
        std::slice::from_ref(&structured.summary),
        muted,
        theme,
        width,
    );
    for (label, values) in [
        ("blocker", &structured.blockers),
        ("finding", &structured.findings),
        ("next_step", &structured.next_steps),
        ("validation", &structured.validation),
        ("changed", &structured.files_changed),
        ("read", &structured.files_read),
        ("command", &structured.commands_run),
    ] {
        render_subagent_wrapped_field(&mut lines, label, values, muted, theme, width);
    }
    if let Some(last) = lines.last_mut() {
        last.boundary = Break::End;
    }
    lines
}

fn render_subagent_compact_line(
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

fn render_subagent_wrapped_field(
    lines: &mut Vec<SemanticLine<Style>>,
    label: &str,
    values: &[String],
    muted: Style,
    theme: Theme,
    width: usize,
) {
    let label_text = format!("{label}: ");
    let label_width = display_width(&label_text);
    let content_width = width
        .saturating_sub(display_width(TOOL_GUIDE_GLYPH).saturating_add(2))
        .max(1);
    let value_width = content_width.saturating_sub(label_width).max(1);
    for value in values {
        let wrapped = wrap_text_to_width(value, value_width);
        for (index, chunk) in wrapped.into_iter().enumerate() {
            let prefix = if index == 0 {
                label_text.clone()
            } else {
                " ".repeat(label_width)
            };
            let mut segments = vec![SemanticSpan::decoration(prefix, muted)];
            if !chunk.is_empty() {
                segments.push(SemanticSpan::source_with_join(
                    chunk,
                    muted,
                    CopyJoin::Space,
                ));
            }
            lines.push(render_card_line_with_guide(
                &segments,
                Style::default().bg(theme.root_bg),
                theme.card_guide(),
                theme,
                width,
                Break::HardBreak,
            ));
        }
    }
}

fn structured_subagent_has_details(result: &StructuredSubagentResult) -> bool {
    !result.summary.trim().is_empty()
        || !result.blockers.is_empty()
        || !result.findings.is_empty()
        || !result.next_steps.is_empty()
        || !result.validation.is_empty()
        || !result.files_changed.is_empty()
        || !result.files_read.is_empty()
        || !result.commands_run.is_empty()
}

fn subagent_activity_summary(result: &StructuredSubagentResult) -> String {
    [
        ("read", result.files_read.len()),
        ("changed", result.files_changed.len()),
        ("commands", result.commands_run.len()),
        ("checks", result.validation.len()),
    ]
    .into_iter()
    .filter(|(_, count)| *count > 0)
    .map(|(label, count)| format!("{label} {count}"))
    .collect::<Vec<_>>()
    .join(" · ")
}

fn subagent_task(tool: &ToolView) -> Option<String> {
    tool_arguments(tool)
        .as_ref()
        .and_then(|args| value_str(Some(args), "task"))
        .map(one_line_snippet)
        .filter(|task| !task.is_empty())
}

fn is_subagent_tool(name: &str) -> bool {
    is_subagent_tool_name(name)
}

fn subagent_name_from_tool(name: &str) -> &str {
    agent_name_for_subagent_tool(name).expect("tool card received unknown subagent tool")
}

fn subagent_status_label(
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

fn subagent_state_flags(data: &serde_json::Value) -> Vec<&'static str> {
    let mut flags = Vec::new();
    if data.get("active").and_then(serde_json::Value::as_bool) == Some(true) {
        flags.push("active");
    }
    if data
        .get("unreconciled")
        .and_then(serde_json::Value::as_bool)
        == Some(true)
    {
        flags.push("unreconciled");
    }
    if data.get("reconciled").and_then(serde_json::Value::as_bool) == Some(true) {
        flags.push("reconciled");
    }
    if data.get("reusable").and_then(serde_json::Value::as_bool) == Some(true) {
        flags.push("reusable");
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

fn render_write_diff_lines(
    tool: &ToolView,
    theme: Theme,
    width: usize,
) -> Vec<SemanticLine<Style>> {
    let Some(args) = tool_arguments(tool) else {
        return Vec::new();
    };
    let path = args
        .get("path")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("file");
    let content = args
        .get("content")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");

    let mut diff = String::new();
    if content.is_empty() {
        diff.push_str("+\n");
    } else {
        for line in content.lines() {
            diff.push('+');
            diff.push_str(line);
            diff.push('\n');
        }
    }

    render_diff_block(
        diff_card_header_title("Write", &[path.to_string()]),
        &diff,
        None,
        theme,
        width,
    )
}

fn render_append_diff_lines(
    tool: &ToolView,
    theme: Theme,
    width: usize,
) -> Vec<SemanticLine<Style>> {
    let Some(args) = tool_arguments(tool) else {
        return Vec::new();
    };
    let path = args
        .get("path")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("file");
    let content = args
        .get("content")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");

    let mut diff = String::new();
    if content.is_empty() {
        diff.push_str("+\n");
    } else {
        for line in content.lines() {
            diff.push('+');
            diff.push_str(line);
            diff.push('\n');
        }
    }

    render_diff_block(
        diff_card_header_title("Append", &[path.to_string()]),
        &diff,
        None,
        theme,
        width,
    )
}

fn render_shell_output_lines(
    tool: &ToolView,
    theme: Theme,
    width: usize,
    expanded_output: bool,
) -> Vec<SemanticLine<Style>> {
    let Some(data) = tool_output_data(tool) else {
        return Vec::new();
    };

    if let Some(error) = data.get("error").and_then(serde_json::Value::as_str) {
        return render_output_section(
            "error",
            error,
            theme.error_style().bg(theme.root_bg),
            theme,
            width,
            expanded_output,
        );
    }

    let stdout = data
        .get("stdout")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");
    let stderr = data
        .get("stderr")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");
    if stdout.trim().is_empty() && stderr.trim().is_empty() {
        return Vec::new();
    }

    let show_expand_hint = !expanded_output
        && [stdout, stderr]
            .into_iter()
            .any(|output| output.lines().count() > COMPACT_SHELL_BODY_LINES);
    let mut lines = render_shell_card_header_lines(tool, theme, width);
    if !stdout.trim().is_empty() || !stderr.trim().is_empty() {
        mark_last_source_boundary(&mut lines, Break::BlockBreak);
    }
    if !stdout.trim().is_empty() {
        lines.extend(render_shell_output_section(
            output_title("stdout", data.get("stdout_truncated")),
            stdout,
            root_text_style(theme),
            theme,
            width,
            expanded_output,
        ));
    }
    if !stderr.trim().is_empty() {
        if !stdout.trim().is_empty() {
            mark_last_source_boundary(&mut lines, Break::BlockBreak);
        }
        if lines.len() > 4 {
            lines.push(render_card_line(
                &[],
                Style::default().bg(theme.card_bg()),
                theme,
                width,
            ));
        }
        lines.extend(render_shell_output_section(
            output_title("stderr", data.get("stderr_truncated")),
            stderr,
            root_text_style(theme),
            theme,
            width,
            expanded_output,
        ));
    }
    if show_expand_hint {
        lines.push(render_card_line(
            &[(
                "… click to expand for details".to_string(),
                root_muted_style(theme).bg(theme.card_bg()),
            )],
            Style::default().bg(theme.card_bg()),
            theme,
            width,
        ));
        lines.push(render_card_line(
            &[],
            Style::default().bg(theme.card_bg()),
            theme,
            width,
        ));
    }
    lines
}

fn mark_last_source_boundary(lines: &mut [SemanticLine<Style>], boundary: Break) {
    if let Some(line) = lines
        .iter_mut()
        .rev()
        .find(|line| line.spans.iter().any(|span| span.copy))
    {
        line.boundary = boundary;
    }
}

fn render_shell_card_header_lines(
    tool: &ToolView,
    theme: Theme,
    width: usize,
) -> Vec<SemanticLine<Style>> {
    let mut lines = Vec::new();
    let command = shell_command(tool);
    let title = shell_card_title(tool, command.as_deref());

    lines.push(render_card_line(
        &[],
        Style::default().bg(theme.card_bg()),
        theme,
        width,
    ));
    lines.push(render_card_line(
        &[(format!("# {title}"), shell_card_title_style(theme))],
        Style::default().bg(theme.card_bg()),
        theme,
        width,
    ));
    lines.push(render_card_line(
        &[],
        Style::default().bg(theme.card_bg()),
        theme,
        width,
    ));

    if let Some(command) = command {
        let command_width = shell_card_content_width(width).saturating_sub(2).max(1);
        for (index, wrapped) in wrap_text_to_width(&command, command_width)
            .into_iter()
            .enumerate()
        {
            let prompt = if index == 0 { "$ " } else { "  " };
            lines.push(render_card_line_with_guide(
                &[
                    SemanticSpan::decoration(prompt, shell_card_command_style(theme)),
                    SemanticSpan::source(wrapped, shell_card_command_style(theme)),
                ],
                Style::default().bg(theme.card_bg()),
                theme.card_guide(),
                theme,
                width,
                Break::SoftWrap,
            ));
        }
        lines.push(render_card_line(
            &[],
            Style::default().bg(theme.card_bg()),
            theme,
            width,
        ));
    }

    lines
}

fn render_shell_output_section(
    title: &str,
    text: &str,
    text_style: Style,
    theme: Theme,
    width: usize,
    expanded_output: bool,
) -> Vec<SemanticLine<Style>> {
    let mut lines = Vec::new();
    lines.push(render_card_line(
        &[(
            title.to_string(),
            root_muted_style(theme)
                .bg(theme.card_bg())
                .add_modifier(Modifier::BOLD),
        )],
        Style::default().bg(theme.card_bg()),
        theme,
        width,
    ));
    lines.push(render_card_line(
        &[],
        Style::default().bg(theme.card_bg()),
        theme,
        width,
    ));
    lines.extend(render_tail_limited_text_lines(
        text,
        text_style,
        theme,
        width,
        expanded_output,
    ));
    lines.push(render_card_line(
        &[],
        Style::default().bg(theme.card_bg()),
        theme,
        width,
    ));
    lines
}

fn shell_command(tool: &ToolView) -> Option<String> {
    tool_arguments(tool)
        .as_ref()
        .and_then(|args| value_str(Some(args), "command"))
        .map(ToOwned::to_owned)
}

fn shell_card_title(tool: &ToolView, command: Option<&str>) -> String {
    let summary = tool.summary.trim();
    if summary.is_empty() || summary.starts_with("exit ") || summary.starts_with("run ") {
        command
            .map(sentence_case_command_goal)
            .unwrap_or_else(|| "Runs command".to_string())
    } else {
        summary.to_string()
    }
}

fn sentence_case_command_goal(command: &str) -> String {
    let command = command.trim();
    if command.is_empty() {
        return "Runs command".to_string();
    }
    let first = command.split_whitespace().next().unwrap_or("command");
    match first {
        "cargo" if command.contains("test") => "Runs test suite".to_string(),
        "cargo" if command.contains("fmt") => "Formats code".to_string(),
        "git" if command.contains("status") => "Shows working tree status".to_string(),
        "git" if command.contains("diff") => "Shows current diff".to_string(),
        _ => format!("Runs {first}"),
    }
}

fn shell_card_content_width(width: usize) -> usize {
    width
        .saturating_sub(display_width(TOOL_GUIDE_GLYPH) + 2)
        .max(1)
}

fn render_edit_diff_lines(tool: &ToolView, theme: Theme, width: usize) -> Vec<SemanticLine<Style>> {
    let Some(args) = tool_arguments(tool) else {
        return Vec::new();
    };
    let Some(edits) = args.get("edits").and_then(serde_json::Value::as_array) else {
        return Vec::new();
    };

    let mut diff = String::new();
    for edit in edits {
        let find = edit
            .get("find")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");
        let replace = edit
            .get("replace")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");

        if find.is_empty() && replace.is_empty() {
            continue;
        }

        for line in find.lines() {
            diff.push('-');
            diff.push_str(&terminal_safe_text(line));
            diff.push('\n');
        }
        for line in replace.lines() {
            diff.push('+');
            diff.push_str(&terminal_safe_text(line));
            diff.push('\n');
        }
    }

    if diff.trim().is_empty() {
        Vec::new()
    } else {
        let paths = edits
            .iter()
            .filter_map(|edit| edit.get("path").and_then(serde_json::Value::as_str))
            .map(terminal_safe_text)
            .collect::<Vec<_>>();
        render_diff_block(
            diff_card_header_title("Patch", &paths),
            &diff,
            None,
            theme,
            width,
        )
    }
}

fn render_output_section(
    title: &str,
    text: &str,
    text_style: Style,
    theme: Theme,
    width: usize,
    expanded_output: bool,
) -> Vec<SemanticLine<Style>> {
    let mut lines = Vec::new();
    lines.push(render_card_line(
        &[],
        Style::default().bg(theme.card_bg()),
        theme,
        width,
    ));
    lines.push(render_card_line(
        &[(
            title.to_string(),
            root_muted_style(theme)
                .bg(theme.card_bg())
                .add_modifier(Modifier::BOLD),
        )],
        Style::default().bg(theme.card_bg()),
        theme,
        width,
    ));
    lines.push(render_card_line(
        &[],
        Style::default().bg(theme.card_bg()),
        theme,
        width,
    ));
    lines.extend(render_limited_text_lines(
        text,
        text_style,
        theme,
        width,
        expanded_output,
    ));
    lines.push(render_card_line(
        &[],
        Style::default().bg(theme.card_bg()),
        theme,
        width,
    ));
    lines
}

fn render_diff_block(
    title: String,
    diff: &str,
    truncated: Option<&serde_json::Value>,
    theme: Theme,
    width: usize,
) -> Vec<SemanticLine<Style>> {
    if width == 0 {
        return Vec::new();
    }

    let mut lines = Vec::new();
    let header = if truncated
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
    {
        format!("{title} · truncated")
    } else {
        title
    };
    lines.push(render_diff_card_spacer_line(theme, width));
    lines.push(render_diff_card_header_line(&header, theme, width));
    lines.push(render_diff_card_spacer_line(theme, width));

    let mut body = diff
        .lines()
        .filter(|line| !is_diff_file_header_line(line))
        .take(max_body_lines().saturating_add(1))
        .collect::<Vec<_>>();
    if body.len() > max_body_lines() {
        body.pop();
        body.push("… output clipped in TUI");
    }

    if diff_uses_side_by_side_layout(width)
        && body.iter().any(|line| {
            (line.starts_with('+') || line.starts_with('-')) && !is_diff_file_header_line(line)
        })
    {
        lines.extend(render_side_by_side_diff_body(&body, theme, width));
    } else {
        let mut state = DiffLineNumbers::default();
        for line in body {
            let (old_no, new_no) = state.next(line);
            lines.push(render_diff_card_body_line(
                old_no,
                new_no,
                line,
                diff_line_style(line, theme),
                theme,
                width,
            ));
        }
    }
    lines.push(render_diff_card_spacer_line(theme, width));
    lines
}

const DIFF_SIDE_BY_SIDE_SEPARATOR: &str = " │ ";
const DIFF_SIDE_BY_SIDE_MIN_PANEL_WIDTH: usize = 20;

fn diff_uses_side_by_side_layout(width: usize) -> bool {
    let content_width = shell_card_content_width(width);
    content_width
        >= DIFF_SIDE_BY_SIDE_MIN_PANEL_WIDTH * 2 + display_width(DIFF_SIDE_BY_SIDE_SEPARATOR)
}

fn render_side_by_side_diff_body(
    body: &[&str],
    theme: Theme,
    width: usize,
) -> Vec<SemanticLine<Style>> {
    let content_width = shell_card_content_width(width);
    let panel_width = content_width.saturating_sub(display_width(DIFF_SIDE_BY_SIDE_SEPARATOR)) / 2;
    let mut lines = Vec::new();
    let mut state = DiffLineNumbers::default();
    let mut index = 0;

    while index < body.len() {
        let line = body[index];
        if line.starts_with('-') && !is_diff_file_header_line(line) {
            let mut removed = Vec::new();
            while index < body.len()
                && body[index].starts_with('-')
                && !is_diff_file_header_line(body[index])
            {
                let (old_no, _) = state.next(body[index]);
                removed.push((old_no, body[index]));
                index += 1;
            }

            let mut added = Vec::new();
            while index < body.len()
                && body[index].starts_with('+')
                && !is_diff_file_header_line(body[index])
            {
                let (_, new_no) = state.next(body[index]);
                added.push((new_no, body[index]));
                index += 1;
            }

            for row in 0..removed.len().max(added.len()) {
                lines.push(render_side_by_side_diff_line(
                    removed.get(row).copied(),
                    added.get(row).copied(),
                    panel_width,
                    theme,
                    width,
                ));
            }
            continue;
        }

        if line.starts_with('+') && !is_diff_file_header_line(line) {
            let mut added = Vec::new();
            while index < body.len()
                && body[index].starts_with('+')
                && !is_diff_file_header_line(body[index])
            {
                let (_, new_no) = state.next(body[index]);
                added.push((new_no, body[index]));
                index += 1;
            }
            for added in added {
                lines.push(render_side_by_side_diff_line(
                    None,
                    Some(added),
                    panel_width,
                    theme,
                    width,
                ));
            }
            continue;
        }

        let (old_no, new_no) = state.next(line);
        if line.starts_with(' ') {
            lines.push(render_side_by_side_diff_line(
                Some((old_no, line)),
                Some((new_no, line)),
                panel_width,
                theme,
                width,
            ));
        } else {
            lines.push(render_diff_card_body_line(
                old_no,
                new_no,
                line,
                diff_line_style(line, theme),
                theme,
                width,
            ));
        }
        index += 1;
    }

    lines
}

fn render_side_by_side_diff_line(
    removed: Option<(Option<usize>, &str)>,
    added: Option<(Option<usize>, &str)>,
    panel_width: usize,
    theme: Theme,
    width: usize,
) -> SemanticLine<Style> {
    let mut segments = render_diff_side(removed, panel_width, theme);
    segments.push(SemanticSpan::decoration(
        DIFF_SIDE_BY_SIDE_SEPARATOR,
        diff_meta_style(theme),
    ));
    segments.extend(render_diff_side(added, panel_width, theme));
    render_card_line_with_guide(
        &segments,
        diff_header_fill_style(theme),
        theme.card_guide(),
        theme,
        width,
        Break::HardBreak,
    )
}

fn render_diff_side(
    entry: Option<(Option<usize>, &str)>,
    width: usize,
    theme: Theme,
) -> Vec<SemanticSpan<Style>> {
    let Some((number, content)) = entry else {
        return vec![SemanticSpan::decoration(
            " ".repeat(width),
            diff_header_fill_style(theme),
        )];
    };

    let gutter_style = Style::default().fg(theme.muted_text).bg(theme.card_bg());
    let gutter_pad_style = Style::default().bg(theme.card_bg());
    let content_style = diff_line_style(content, theme);
    let pad_style = Style::default().bg(content_style.bg.unwrap_or(theme.card_bg()));
    let (marker, body, marker_style) = diff_marker_and_body(content, theme);
    let clipped_notice = content == "… output clipped in TUI";
    let marker_span = if clipped_notice {
        SemanticSpan::decoration(marker, marker_style)
    } else {
        SemanticSpan::source(marker, marker_style)
    };
    let body_span = if clipped_notice {
        SemanticSpan::decoration(body, content_style)
    } else {
        SemanticSpan::source(body, content_style)
    };
    let mut spans = clip_semantic_spans(
        vec![
            SemanticSpan::decoration(diff_line_number(number), gutter_style),
            SemanticSpan::decoration(" ", gutter_pad_style),
            marker_span,
            SemanticSpan::decoration(" ", pad_style),
            body_span,
        ],
        width,
    );
    let used = spans
        .iter()
        .map(|span| display_width(&span.text))
        .sum::<usize>();
    if used < width {
        spans.push(SemanticSpan::decoration(
            " ".repeat(width - used),
            pad_style,
        ));
    }
    spans
}

fn render_limited_text_lines(
    text: &str,
    text_style: Style,
    theme: Theme,
    width: usize,
    expanded_output: bool,
) -> Vec<SemanticLine<Style>> {
    let mut lines = Vec::new();
    for (idx, raw) in text.lines().enumerate() {
        if !expanded_output && idx >= max_body_lines() {
            lines.push(render_card_line(
                &[(
                    "… output clipped in TUI".to_string(),
                    root_muted_style(theme).bg(theme.card_bg()),
                )],
                Style::default().bg(theme.card_bg()),
                theme,
                width,
            ));
            break;
        }
        let line = if raw.is_empty() { " " } else { raw };
        let segments = ansi_sgr_segments(line, text_style.bg(theme.card_bg()));
        lines.push(render_source_card_line_with_boundary(
            &segments,
            Style::default().bg(theme.card_bg()),
            theme,
            width,
            Break::HardBreak,
        ));
    }
    lines
}

fn render_tail_limited_text_lines(
    text: &str,
    text_style: Style,
    theme: Theme,
    width: usize,
    expanded_output: bool,
) -> Vec<SemanticLine<Style>> {
    let body = text.lines().collect::<Vec<_>>();
    let is_clipped = !expanded_output && body.len() > COMPACT_SHELL_BODY_LINES;
    let body = if is_clipped {
        &body[body.len() - COMPACT_SHELL_BODY_LINES..]
    } else {
        &body[..]
    };

    let mut lines = Vec::new();
    for raw in body {
        let line = if raw.is_empty() { " " } else { raw };
        let segments = ansi_sgr_segments(line, text_style.bg(theme.card_bg()));
        lines.push(render_source_card_line_with_boundary(
            &segments,
            Style::default().bg(theme.card_bg()),
            theme,
            width,
            Break::HardBreak,
        ));
    }
    lines
}

fn terminal_safe_text(text: &str) -> String {
    ansi_sgr_segments(text, Style::default())
        .into_iter()
        .map(|(text, _)| text)
        .collect()
}

/// Split shell/tool output into styled segments for TUI cells.
///
/// Contract: segment text never contains control characters. SGR becomes style;
/// other CSI/OSC/C0 and truncated escapes are dropped so VT state cannot escape
/// into the ratatui write path.
fn ansi_sgr_segments(text: &str, base_style: Style) -> Vec<(String, Style)> {
    // Progress bars overwrite with CR; only the suffix after the last CR is visible.
    let text = text.rsplit('\r').next().unwrap_or(text);

    let mut segments = Vec::new();
    let mut current_style = base_style;
    let mut current_text = String::new();
    let mut chars = text.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch == '\u{1b}' {
            match chars.peek().copied() {
                Some('[') => {
                    chars.next();
                    match consume_csi_sequence(&mut chars) {
                        CsiOutcome::Sgr(sequence) => {
                            if !current_text.is_empty() {
                                segments.push((std::mem::take(&mut current_text), current_style));
                            }
                            current_style =
                                apply_sgr_sequence(&sequence, base_style, current_style);
                        }
                        CsiOutcome::Other | CsiOutcome::Incomplete => {}
                    }
                }
                Some(']') => {
                    chars.next();
                    let _ = consume_osc_sequence(&mut chars);
                }
                Some(_) => {
                    // Two-byte / short ESC form: drop introducer and the next byte.
                    chars.next();
                }
                None => {
                    // Truncated ESC at end of stream — never emit it into a cell.
                }
            }
            continue;
        }

        if ch == '\t' {
            current_text.push(' ');
            continue;
        }

        if ch.is_control() {
            continue;
        }

        current_text.push(ch);
    }

    if !current_text.is_empty() {
        segments.push((current_text, current_style));
    }

    if segments.is_empty() {
        segments.push((String::new(), base_style));
    }
    segments
}

enum CsiOutcome {
    Sgr(String),
    Other,
    Incomplete,
}

/// Consume CSI parameter/intermediate bytes through the final byte (`@`..=`~`).
fn consume_csi_sequence(chars: &mut std::iter::Peekable<std::str::Chars<'_>>) -> CsiOutcome {
    let mut sequence = String::new();
    let mut saw_final = false;
    let mut final_byte = '\0';

    for next in chars.by_ref() {
        match next {
            '\u{20}'..='\u{3f}' => sequence.push(next),
            '\u{40}'..='\u{7e}' => {
                final_byte = next;
                saw_final = true;
                break;
            }
            _ => {
                // Malformed CSI: drop what we consumed; do not emit ESC.
                return CsiOutcome::Other;
            }
        }
    }

    if !saw_final {
        return CsiOutcome::Incomplete;
    }

    if final_byte == 'm' {
        // SGR params are digits/semicolons; strip private/intermediate junk.
        let params: String = sequence
            .chars()
            .filter(|ch| ch.is_ascii_digit() || *ch == ';')
            .collect();
        CsiOutcome::Sgr(params)
    } else {
        CsiOutcome::Other
    }
}

/// Consume OSC through BEL or ST (`ESC \`). Incomplete OSC is dropped.
fn consume_osc_sequence(chars: &mut std::iter::Peekable<std::str::Chars<'_>>) -> bool {
    while let Some(next) = chars.next() {
        if next == '\u{07}' {
            return true;
        }
        if next == '\u{1b}' {
            if chars.peek() == Some(&'\\') {
                chars.next();
                return true;
            }
            // Nested/truncated ESC inside OSC — stop without leaking.
            return false;
        }
    }
    false
}

fn apply_sgr_sequence(sequence: &str, base_style: Style, mut style: Style) -> Style {
    let codes: Vec<u16> = if sequence.is_empty() {
        vec![0]
    } else {
        sequence
            .split(';')
            .map(|part| part.parse::<u16>().unwrap_or(0))
            .collect()
    };

    let mut index = 0;
    while index < codes.len() {
        match codes[index] {
            0 => style = base_style,
            1 => style = style.add_modifier(Modifier::BOLD),
            3 => style = style.add_modifier(Modifier::ITALIC),
            4 => style = style.add_modifier(Modifier::UNDERLINED),
            22 => style = style.remove_modifier(Modifier::BOLD),
            23 => style = style.remove_modifier(Modifier::ITALIC),
            24 => style = style.remove_modifier(Modifier::UNDERLINED),
            30..=37 => style = style.fg(ansi_basic_color(codes[index] - 30, false)),
            39 => {
                style = match base_style.fg {
                    Some(color) => style.fg(color),
                    None => style,
                }
            }
            90..=97 => style = style.fg(ansi_basic_color(codes[index] - 90, true)),
            38 if codes.get(index + 1) == Some(&5) => {
                if let Some(color_index) = codes.get(index + 2).copied() {
                    style = style.fg(ansi_256_color(color_index));
                    index += 2;
                }
            }
            38 if codes.get(index + 1) == Some(&2) => {
                if let (Some(r), Some(g), Some(b)) = (
                    codes.get(index + 2).copied(),
                    codes.get(index + 3).copied(),
                    codes.get(index + 4).copied(),
                ) {
                    style = style.fg(Color::Rgb(r as u8, g as u8, b as u8));
                    index += 4;
                }
            }
            _ => {}
        }
        index += 1;
    }

    style
}

fn ansi_basic_color(index: u16, bright: bool) -> Color {
    let colors = if bright {
        [
            Color::Rgb(128, 128, 128),
            Color::Rgb(255, 85, 85),
            Color::Rgb(80, 250, 123),
            Color::Rgb(241, 250, 140),
            Color::Rgb(98, 114, 164),
            Color::Rgb(255, 121, 198),
            Color::Rgb(139, 233, 253),
            Color::Rgb(248, 248, 242),
        ]
    } else {
        [
            Color::Rgb(0, 0, 0),
            Color::Rgb(205, 49, 49),
            Color::Rgb(13, 188, 121),
            Color::Rgb(229, 229, 16),
            Color::Rgb(36, 114, 200),
            Color::Rgb(188, 63, 188),
            Color::Rgb(17, 168, 205),
            Color::Rgb(229, 229, 229),
        ]
    };
    colors[index as usize]
}

fn ansi_256_color(index: u16) -> Color {
    match index {
        0..=7 => ansi_basic_color(index, false),
        8..=15 => ansi_basic_color(index - 8, true),
        16..=231 => {
            let value = index - 16;
            let r = value / 36;
            let g = (value % 36) / 6;
            let b = value % 6;
            Color::Rgb(
                ansi_256_component(r),
                ansi_256_component(g),
                ansi_256_component(b),
            )
        }
        232..=255 => {
            let level = 8 + ((index - 232) * 10) as u8;
            Color::Rgb(level, level, level)
        }
        _ => Color::Reset,
    }
}

fn ansi_256_component(value: u16) -> u8 {
    if value == 0 {
        0
    } else {
        (55 + value * 40) as u8
    }
}

fn render_generic_output_lines(
    tool: &ToolView,
    theme: Theme,
    width: usize,
    expanded_output: bool,
) -> Vec<SemanticLine<Style>> {
    let Some(output) = tool.output.as_deref() else {
        return Vec::new();
    };

    let parsed = serde_json::from_str::<serde_json::Value>(output).ok();
    if let Some(error) = parsed
        .as_ref()
        .and_then(|value| value.get("error"))
        .and_then(|error| error.get("message"))
        .and_then(serde_json::Value::as_str)
    {
        return render_output_section(
            "error",
            error,
            theme.error_style().bg(theme.root_bg),
            theme,
            width,
            expanded_output,
        );
    }

    let body = parsed
        .and_then(|value| value.get("data").cloned().or(Some(value)))
        .map(|value| serde_json::to_string_pretty(&value).unwrap_or_else(|_| value.to_string()))
        .unwrap_or_else(|| output.to_string());

    if body.trim().is_empty() {
        return Vec::new();
    }

    render_output_section(
        "output",
        &body,
        root_text_style(theme),
        theme,
        width,
        expanded_output,
    )
}

fn tool_arguments(tool: &ToolView) -> Option<serde_json::Value> {
    tool.arguments
        .as_deref()
        .and_then(|text| serde_json::from_str::<serde_json::Value>(text).ok())
}

fn permission_arguments_summary(tool_name: &str, arguments: &str) -> String {
    let Some(args) = serde_json::from_str::<serde_json::Value>(arguments).ok() else {
        return one_line_snippet(arguments);
    };

    match tool_name {
        "fs__write" => format!("Write {}", value_str(Some(&args), "path").unwrap_or("file")),
        "fs__append" => format!(
            "Append {}",
            value_str(Some(&args), "path").unwrap_or("file")
        ),
        "edit__apply_patch" => {
            let edits = args
                .get("edits")
                .and_then(serde_json::Value::as_array)
                .map(Vec::len)
                .unwrap_or(0);
            format!("Patch {edits} edits")
        }
        "shell__exec" => format!(
            "Run {}",
            value_str(Some(&args), "command").unwrap_or("command")
        ),
        _ => one_line_snippet(arguments),
    }
}

fn tool_output_data(tool: &ToolView) -> Option<serde_json::Value> {
    let output = tool.output.as_deref()?;
    let value = serde_json::from_str::<serde_json::Value>(output).ok()?;
    value.get("data").cloned().or(Some(value))
}

fn output_title<'a>(label: &'a str, truncated: Option<&serde_json::Value>) -> &'a str {
    if truncated
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
    {
        match label {
            "stdout" => "stdout · truncated",
            "stderr" => "stderr · truncated",
            "diff" => "diff · truncated",
            _ => label,
        }
    } else {
        label
    }
}

fn diff_line_style(line: &str, theme: Theme) -> Style {
    if line.starts_with("diff --git") || line.starts_with("index ") {
        diff_meta_style(theme).add_modifier(Modifier::BOLD)
    } else if line.starts_with("+++") || line.starts_with("---") {
        diff_meta_style(theme)
    } else if line.starts_with('+') {
        Style::default().fg(theme.text).bg(theme.diff_add_bg)
    } else if line.starts_with('-') {
        Style::default().fg(theme.text).bg(theme.diff_delete_bg)
    } else if line.starts_with("@@") {
        Style::default().fg(theme.user).bg(theme.diff_hunk_bg)
    } else {
        Style::default().fg(theme.text).bg(theme.card_bg())
    }
}

fn is_diff_file_header_line(line: &str) -> bool {
    line.starts_with("---") || line.starts_with("+++")
}

fn diff_meta_style(theme: Theme) -> Style {
    Style::default().fg(theme.muted_text).bg(theme.card_bg())
}

fn render_diff_card_header_line(title: &str, theme: Theme, width: usize) -> SemanticLine<Style> {
    let text = format!(" {DIFF_CARD_HEADER_ARROW} {title}");
    render_card_line(
        &[(text, diff_header_style(theme))],
        diff_header_fill_style(theme),
        theme,
        width,
    )
}

fn render_diff_card_spacer_line(theme: Theme, width: usize) -> SemanticLine<Style> {
    render_card_line(&[], diff_header_fill_style(theme), theme, width)
}

fn render_diff_card_body_line(
    old_no: Option<usize>,
    new_no: Option<usize>,
    content: &str,
    content_style: Style,
    theme: Theme,
    width: usize,
) -> SemanticLine<Style> {
    let gutter_style = Style::default().fg(theme.muted_text).bg(theme.card_bg());
    let number = diff_line_number(new_no.or(old_no));
    let bg = content_style.bg.unwrap_or(theme.card_bg());
    let pad_style = Style::default().bg(bg);
    let gutter_pad_style = Style::default().bg(theme.card_bg());
    let (marker, body, marker_style) = diff_marker_and_body(content, theme);
    let clipped_notice = content == "… output clipped in TUI";
    let marker_span = if clipped_notice {
        SemanticSpan::decoration(marker, marker_style)
    } else {
        SemanticSpan::source(marker, marker_style)
    };
    let body_span = if clipped_notice {
        SemanticSpan::decoration(body, content_style)
    } else {
        SemanticSpan::source(body, content_style)
    };
    let segments = vec![
        SemanticSpan::decoration("", gutter_pad_style),
        SemanticSpan::decoration(number, gutter_style),
        SemanticSpan::decoration(" ", gutter_pad_style),
        marker_span,
        SemanticSpan::decoration(" ", pad_style),
        body_span,
    ];
    render_card_line_with_guide(
        &segments,
        content_style,
        theme.card_guide(),
        theme,
        width,
        Break::HardBreak,
    )
}

fn diff_marker_and_body(content: &str, theme: Theme) -> (String, String, Style) {
    match content.chars().next() {
        Some('+') if !content.starts_with("+++") => (
            "+".to_string(),
            content.chars().skip(1).collect(),
            Style::default().fg(theme.success).bg(theme.card_bg()),
        ),
        Some('-') if !content.starts_with("---") => (
            "-".to_string(),
            content.chars().skip(1).collect(),
            Style::default().fg(theme.error).bg(theme.card_bg()),
        ),
        _ => (
            " ".to_string(),
            content.to_string(),
            Style::default().fg(theme.muted_text).bg(theme.card_bg()),
        ),
    }
}

pub(crate) fn render_card_line(
    segments: &[(String, Style)],
    fill_style: Style,
    theme: Theme,
    width: usize,
) -> SemanticLine<Style> {
    let semantic_segments = segments
        .iter()
        .map(|(text, style)| SemanticSpan::decoration(text.clone(), *style))
        .collect::<Vec<_>>();
    render_card_line_with_guide(
        &semantic_segments,
        fill_style,
        theme.card_guide(),
        theme,
        width,
        Break::SoftWrap,
    )
}

fn render_source_card_line_with_boundary(
    segments: &[(String, Style)],
    fill_style: Style,
    theme: Theme,
    width: usize,
    boundary: Break,
) -> SemanticLine<Style> {
    let semantic_segments = segments
        .iter()
        .map(|(text, style)| SemanticSpan::source(text.clone(), *style))
        .collect::<Vec<_>>();
    render_card_line_with_guide(
        &semantic_segments,
        fill_style,
        theme.card_guide(),
        theme,
        width,
        boundary,
    )
}

fn render_card_line_with_guide(
    segments: &[SemanticSpan<Style>],
    fill_style: Style,
    guide_color: ratatui::style::Color,
    theme: Theme,
    width: usize,
    boundary: Break,
) -> SemanticLine<Style> {
    if width == 0 {
        return SemanticLine::default();
    }

    let guide_width = display_width(TOOL_GUIDE_GLYPH);
    let prefix_width = guide_width.saturating_add(2);
    let guide_style = Style::default().fg(guide_color).bg(theme.root_bg);
    if width <= guide_width {
        return SemanticLine {
            spans: vec![SemanticSpan::decoration(TOOL_GUIDE_GLYPH, guide_style)],
            boundary,
        };
    }

    let leading_pad_style = fill_style;

    let mut spans = vec![
        SemanticSpan::decoration(TOOL_GUIDE_GLYPH, guide_style),
        SemanticSpan::decoration("  ", leading_pad_style),
    ];
    let mut remaining = width.saturating_sub(prefix_width);

    for segment in clip_semantic_spans(segments.to_vec(), remaining) {
        let used = display_width(&segment.text);
        spans.push(segment);
        remaining = remaining.saturating_sub(used);
    }

    if remaining > 0 {
        spans.push(SemanticSpan::decoration(" ".repeat(remaining), fill_style));
    }

    SemanticLine { spans, boundary }
}

fn diff_header_style(theme: Theme) -> Style {
    Style::default().fg(theme.muted_text).bg(theme.card_bg())
}

fn diff_header_fill_style(theme: Theme) -> Style {
    Style::default().bg(theme.card_bg())
}

fn shell_card_title_style(theme: Theme) -> Style {
    Style::default()
        .fg(theme.accent)
        .bg(theme.card_bg())
        .add_modifier(Modifier::BOLD)
}

fn shell_card_command_style(theme: Theme) -> Style {
    Style::default().fg(theme.text).bg(theme.card_bg())
}

fn diff_line_number(number: Option<usize>) -> String {
    match number {
        Some(value) => format!("{:>3}", value),
        None => "   ".to_string(),
    }
}

#[derive(Default)]
struct DiffLineNumbers {
    old_next: Option<usize>,
    new_next: Option<usize>,
}

impl DiffLineNumbers {
    fn next(&mut self, line: &str) -> (Option<usize>, Option<usize>) {
        if let Some((old_start, new_start)) = parse_hunk_header(line) {
            self.old_next = Some(old_start);
            self.new_next = Some(new_start);
            return (None, None);
        }

        if line.starts_with("diff --git")
            || line.starts_with("index ")
            || line.starts_with("---")
            || line.starts_with("+++")
        {
            self.old_next = None;
            self.new_next = None;
            return (None, None);
        }

        match line.chars().next() {
            Some('+') => {
                let next = self.new_next.get_or_insert(1);
                let current = *next;
                *next += 1;
                (None, Some(current))
            }
            Some('-') => {
                let next = self.old_next.get_or_insert(1);
                let current = *next;
                *next += 1;
                (Some(current), None)
            }
            Some(' ') => {
                let old = {
                    let next = self.old_next.get_or_insert(1);
                    let current = *next;
                    *next += 1;
                    current
                };
                let new = {
                    let next = self.new_next.get_or_insert(1);
                    let current = *next;
                    *next += 1;
                    current
                };
                (Some(old), Some(new))
            }
            _ => (None, None),
        }
    }
}

fn parse_hunk_header(line: &str) -> Option<(usize, usize)> {
    if !line.starts_with("@@") {
        return None;
    }

    let mut parts = line.split_whitespace();
    let _ = parts.next()?;
    let old = parts.next()?;
    let new = parts.next()?;
    Some((parse_hunk_range_start(old)?, parse_hunk_range_start(new)?))
}

fn parse_hunk_range_start(part: &str) -> Option<usize> {
    let trimmed = part.strip_prefix(['-', '+'])?;
    trimmed
        .split(',')
        .next()
        .and_then(|value| value.parse::<usize>().ok())
}

fn diff_card_header_title(label: &str, paths: &[String]) -> String {
    match paths.first() {
        Some(first) if paths.len() > 1 => format!("{label} {} +{}", first, paths.len() - 1),
        Some(first) => format!("{label} {first}"),
        None => label.to_string(),
    }
}

fn root_text_style(theme: Theme) -> Style {
    Style::default().fg(theme.text).bg(theme.root_bg)
}

fn root_muted_style(theme: Theme) -> Style {
    Style::default().fg(theme.muted_text).bg(theme.root_bg)
}

fn max_body_lines() -> usize {
    120
}

fn tool_trace_label(tool: &ToolView) -> String {
    if tool.status == ToolExecutionStatus::Pending && tool.arguments.is_none() {
        return pending_tool_trace_label(&tool.name);
    }

    let args = tool
        .arguments
        .as_deref()
        .and_then(|text| serde_json::from_str::<serde_json::Value>(text).ok());
    let args = args.as_ref();

    match tool.name.as_str() {
        crate::tool_names::TOOL_QUESTION => match tool.status {
            ToolExecutionStatus::Pending => "Ask a question".into(),
            ToolExecutionStatus::Running => "Waiting for answer".into(),
            ToolExecutionStatus::Cancelled => "Question cancelled".into(),
            ToolExecutionStatus::Succeeded => "Question answered".into(),
            ToolExecutionStatus::Failed => "Question failed".into(),
        },
        "fs__read" => {
            let path = value_str(args, "path").unwrap_or_else(|| fallback_tail(&tool.summary));
            let mut fields = Vec::new();
            if let Some(offset) = value_u64(args, "offset") {
                fields.push(format!("offset={offset}"));
            }
            if let Some(limit) = value_u64(args, "limit") {
                fields.push(format!("limit={limit}"));
            }
            format_with_optional_fields("Read", path, fields)
        }
        "fs__list" => format!(
            "List {}",
            value_str(args, "path").unwrap_or_else(|| fallback_tail(&tool.summary))
        ),
        "fs__write" => format!(
            "Write {}",
            value_str(args, "path").unwrap_or_else(|| fallback_tail(&tool.summary))
        ),
        "fs__append" => format!(
            "Append {}",
            value_str(args, "path").unwrap_or_else(|| fallback_tail(&tool.summary))
        ),
        "fs__mkdir" => format!(
            "Make dir {}",
            value_str(args, "path").unwrap_or_else(|| fallback_tail(&tool.summary))
        ),
        "shell__exec" => format!(
            "Run {}",
            value_str(args, "command").unwrap_or(tool.summary.as_str())
        ),
        "search__rg" => {
            let pattern = value_str(args, "pattern").unwrap_or("pattern");
            let path = value_str(args, "path").unwrap_or(".");
            format!(
                "Search {:?} in {}",
                truncate_display_width(pattern, 60),
                path
            )
        }
        "web__fetch" => format!(
            "Fetch {}",
            value_str(args, "url").unwrap_or_else(|| fallback_tail(&tool.summary))
        ),
        "git__status" => "Git status".into(),
        "git__diff" => "Git diff".into(),
        "git__log" => "Git log".into(),
        "skill" => format!(
            "Skill {:?}",
            truncate_display_width(value_str(args, "name").unwrap_or("skill"), 60)
        ),
        "edit__apply_patch" => "Apply patch".into(),
        "workflow__todos" => "Update todos".into(),
        "workflow__auto_continue" => match args
            .and_then(|value| value.get("enabled"))
            .and_then(serde_json::Value::as_bool)
        {
            Some(enabled) => format!("⚙ auto-continue {enabled}"),
            None => "⚙ auto-continue".into(),
        },
        "code__ast_search" => {
            let path = value_str(args, "path").unwrap_or(".");
            format!("AST search in {path}")
        }
        "code__ast_replace_preview" => {
            let path = value_str(args, "path").unwrap_or(".");
            format!("AST replace preview in {path}")
        }
        "util__echo" => "Echo".into(),
        _ => format!(
            "{} {}",
            sentence_case_tool_name(&tool.name),
            fallback_tail(&tool.summary)
        ),
    }
}

fn pending_tool_trace_label(name: &str) -> String {
    match name {
        crate::tool_names::TOOL_QUESTION => "Ask a question".into(),
        "git__status" => "Git status".into(),
        "git__diff" => "Git diff".into(),
        "git__log" => "Git log".into(),
        "skill" => "Skill".into(),
        "edit__apply_patch" => "Apply patch".into(),
        "workflow__todos" => "Update todos".into(),
        "workflow__auto_continue" => "⚙ auto-continue".into(),
        "fs__read" => "Read".into(),
        "fs__list" => "List".into(),
        "fs__write" => "Write".into(),
        "fs__append" => "Append".into(),
        "fs__mkdir" => "Make dir".into(),
        "shell__exec" => "Run command".into(),
        "search__rg" => "Search".into(),
        "web__fetch" => "Fetch URL".into(),
        "code__ast_search" => "AST search".into(),
        "code__ast_replace_preview" => "AST replace preview".into(),
        "util__echo" => "Echo".into(),
        _ => sentence_case_tool_name(name),
    }
}

fn format_with_optional_fields(prefix: &str, subject: &str, fields: Vec<String>) -> String {
    if fields.is_empty() {
        format!("{prefix} {subject}")
    } else {
        format!("{prefix} {subject} [{}]", fields.join(", "))
    }
}

fn value_str<'a>(args: Option<&'a serde_json::Value>, key: &str) -> Option<&'a str> {
    args.and_then(|value| value.get(key))
        .and_then(serde_json::Value::as_str)
}

fn value_u64(args: Option<&serde_json::Value>, key: &str) -> Option<u64> {
    args.and_then(|value| value.get(key))
        .and_then(serde_json::Value::as_u64)
}

fn fallback_tail(summary: &str) -> &str {
    summary
        .split_once(' ')
        .map(|(_, tail)| tail.trim())
        .filter(|tail| !tail.is_empty())
        .unwrap_or(summary)
}

fn sentence_case_tool_name(name: &str) -> String {
    let label = name.replace('_', " ");
    let mut chars = label.chars();
    let Some(first) = chars.next() else {
        return "Tool".into();
    };
    format!("{}{}", first.to_uppercase(), chars.as_str())
}

fn one_line_snippet(text: &str) -> String {
    // Collapse newlines/whitespace into single spaces, then trim.
    let mut out = String::with_capacity(text.len().min(140));
    let mut last_was_space = false;
    for ch in text.chars() {
        let is_space = ch.is_whitespace();
        if is_space {
            if !last_was_space {
                out.push(' ');
                last_was_space = true;
            }
            continue;
        }
        last_was_space = false;
        out.push(ch);
        if out.len() >= 240 {
            break;
        }
    }
    out.trim().to_string()
}

fn map_tool_status(status: ToolExecutionStatus) -> ToolCardStatus {
    match status {
        ToolExecutionStatus::Pending => ToolCardStatus::Pending,
        ToolExecutionStatus::Running => ToolCardStatus::Running,
        ToolExecutionStatus::Cancelled => ToolCardStatus::Cancelled,
        ToolExecutionStatus::Succeeded => ToolCardStatus::Succeeded,
        ToolExecutionStatus::Failed => ToolCardStatus::Failed,
    }
}

fn permission_accent(status: PermissionPromptStatus, theme: Theme) -> ratatui::style::Color {
    match status {
        PermissionPromptStatus::Pending => theme.approval,
        PermissionPromptStatus::Approved => theme.success,
        PermissionPromptStatus::Denied => theme.error,
    }
}

fn status_label(status: ToolCardStatus, translator: &crate::tui::i18n::Translator) -> String {
    translator.t(match status {
        ToolCardStatus::Pending => "status.pending",
        ToolCardStatus::Approved => "status.approved",
        ToolCardStatus::Running => "status.running",
        ToolCardStatus::Cancelled => "status.cancelled",
        ToolCardStatus::Succeeded => "status.succeeded",
        ToolCardStatus::Failed => "status.failed",
        ToolCardStatus::Denied => "status.denied",
    })
}

fn tool_trace_arrow_style(status: ToolExecutionStatus, theme: Theme) -> ratatui::style::Style {
    let color = match status {
        ToolExecutionStatus::Pending => theme.warning,
        ToolExecutionStatus::Running => theme.warning,
        ToolExecutionStatus::Cancelled => theme.error,
        ToolExecutionStatus::Succeeded => theme.assistant,
        ToolExecutionStatus::Failed => theme.error,
    };

    ratatui::style::Style::default()
        .fg(color)
        .bg(theme.root_bg)
        .add_modifier(ratatui::style::Modifier::BOLD)
}

fn tool_trace_text_style(status: ToolExecutionStatus, theme: Theme) -> ratatui::style::Style {
    let color = match status {
        ToolExecutionStatus::Pending => theme.warning,
        ToolExecutionStatus::Running => theme.warning,
        ToolExecutionStatus::Cancelled => theme.error,
        ToolExecutionStatus::Succeeded => theme.muted_text,
        ToolExecutionStatus::Failed => theme.error,
    };

    ratatui::style::Style::default().fg(color).bg(theme.root_bg)
}

fn root_status_style(color: ratatui::style::Color, theme: Theme) -> ratatui::style::Style {
    ratatui::style::Style::default()
        .fg(color)
        .bg(theme.root_bg)
        .add_modifier(ratatui::style::Modifier::BOLD)
}

pub fn truncate_display_width(text: &str, width: usize) -> String {
    if width == 0 {
        return String::new();
    }

    if display_width(text) <= width {
        return text.to_string();
    }

    let ellipsis = "…";
    let ellipsis_width = display_width(ellipsis);
    if width <= ellipsis_width {
        return ellipsis.chars().take(1).collect();
    }

    let mut out = String::new();
    let mut used = 0usize;

    for grapheme in text.graphemes(true) {
        let grapheme_width = display_width(grapheme);
        if used + grapheme_width + ellipsis_width > width {
            break;
        }
        out.push_str(grapheme);
        used = used.saturating_add(grapheme_width);
    }

    out.push('…');
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tui::timeline::{PermissionPromptStatus, ToolExecutionStatus};
    use serde_json::json;

    fn question_tool(arguments: Option<serde_json::Value>, data: serde_json::Value) -> ToolView {
        ToolView {
            call_id: "question-1".into(),
            name: crate::tool_names::TOOL_QUESTION.into(),
            summary: "Question 2 fields".into(),
            arguments: arguments.map(|value| value.to_string()),
            output: Some(json!({"ok": true, "data": data}).to_string()),
            status: ToolExecutionStatus::Succeeded,
        }
    }

    fn rendered_question(tool: &ToolView, width: usize) -> Vec<String> {
        render_tool_card_lines(tool, Theme::dark(), width)
            .into_iter()
            .map(|line| line.to_string())
            .collect()
    }

    fn plain_question_lines(tool: &ToolView, width: usize) -> Vec<String> {
        let prefix = format!("{TOOL_GUIDE_GLYPH}  ");
        rendered_question(tool, width)
            .into_iter()
            .map(|line| {
                line.strip_prefix(&prefix)
                    .unwrap_or(&line)
                    .trim_end()
                    .to_string()
            })
            .collect()
    }

    fn assert_question_padding_frame(tool: &ToolView, width: usize) {
        let lines = render_tool_card_lines(tool, Theme::dark(), width);
        let surface = Theme::dark().element_bg;
        for line in [
            lines.first().expect("top padding"),
            lines.last().expect("bottom padding"),
        ] {
            assert_eq!(line.spans[0].content, TOOL_GUIDE_GLYPH);
            assert!(
                line.spans[0].style
                    == Style::default()
                        .fg(Theme::dark().card_guide())
                        .bg(Theme::dark().root_bg)
                    && line.spans[1].content == "  "
                    && line.spans[1].style.bg == Some(surface)
                    && line.spans[2..]
                        .iter()
                        .all(|span| span.style.bg == Some(surface))
            );
            assert_eq!(display_width(&line.to_string()), width);
        }
    }

    #[test]
    fn width_zero_returns_empty_string() {
        assert_eq!(truncate_display_width("abcdef", 0), "");
    }

    #[test]
    fn cjk_text_truncates_on_display_cells() {
        assert_eq!(truncate_display_width("你好吗", 5), "你好…");
    }

    #[test]
    fn truncation_preserves_extended_graphemes() {
        assert_eq!(truncate_display_width("e\u{301}xy", 2), "e\u{301}…");
        assert_eq!(truncate_display_width("👩‍💻xy", 3), "👩‍💻…");
    }

    #[test]
    fn non_successful_question_does_not_render_the_response_card() {
        let mut tool = question_tool(
            Some(
                json!({"questions": [{"header": "Mode", "question": "Which mode should we use?", "options": [], "multiple": false}]}),
            ),
            json!({"answers": [["Fast"]]}),
        );
        tool.status = ToolExecutionStatus::Running;

        let rendered = rendered_question(&tool, 80).join("\n");
        assert!(rendered.contains("Waiting for answer"), "{rendered}");
        assert!(!rendered.contains("# User response"), "{rendered}");
        assert!(
            !rendered.contains("Which mode should we use?"),
            "{rendered}"
        );
        assert!(!rendered.contains("Fast"), "{rendered}");
    }

    #[test]
    fn question_card_wraps_without_overflow_at_narrow_width() {
        let tool = question_tool(
            Some(
                json!({"questions": [{"question": "A long question that must wrap safely in a narrow terminal", "header": "Mode", "options": [], "multiple": false}]}),
            ),
            json!({"answers": [["A custom answer that must also wrap safely"]]}),
        );
        let lines = rendered_question(&tool, 24);

        assert!(
            lines.iter().any(|line| line.contains("custom")),
            "{lines:?}"
        );
        assert!(
            lines.iter().all(|line| display_width(line) <= 24),
            "{lines:?}"
        );
    }

    #[test]
    fn malformed_question_output_falls_back_to_a_human_trace_without_field_noise() {
        let tool = question_tool(None, json!({"answers": "bad"}));
        let rendered = rendered_question(&tool, 60).join("\n");

        assert!(rendered.contains("Question answered"), "{rendered}");
        assert!(!rendered.contains("fields"), "{rendered}");
    }

    #[test]
    fn question_label_aware_wrapping_preserves_long_ascii_and_cjk_text() {
        let ascii = "The complete ASCII question must survive a long header on the first line";
        let cjk = "完整的中文问题与回答不能因为标签而丢失";
        let tool = question_tool(
            Some(json!({"questions": [
                {"header": "A deliberately long header label", "question": ascii, "options": [], "multiple": false},
                {"header": "中文标签很长", "question": cjk, "options": [], "multiple": false}
            ]})),
            json!({"answers": [["The complete ASCII answer must also survive wrapping"], [cjk]]}),
        );
        let rendered = rendered_question(&tool, 32)
            .iter()
            .map(String::as_str)
            .collect::<String>();
        let compact = rendered
            .chars()
            .filter(|ch| !ch.is_whitespace() && *ch != TOOL_GUIDE_GLYPH.chars().next().unwrap())
            .collect::<String>();

        assert!(
            compact.contains(
                &ascii
                    .chars()
                    .filter(|ch| !ch.is_whitespace())
                    .collect::<String>()
            ),
            "{rendered}"
        );
        assert!(compact.contains(cjk), "{rendered}");
        assert!(
            compact.contains("ThecompleteASCIIanswermustalsosurvivewrapping"),
            "{rendered}"
        );
    }

    #[test]
    fn question_card_is_safe_at_extremely_narrow_widths() {
        let tool = question_tool(
            Some(
                json!({"questions": [{"header": "Mode", "question": "你好", "options": [], "multiple": false}]}),
            ),
            json!({"answers": [["好的"]]}),
        );
        for width in 0..=2 {
            assert!(rendered_question(&tool, width).len() <= 1);
        }
    }

    #[test]
    fn subagent_card_falls_back_for_missing_or_invalid_structured_results() {
        for structured_result in [
            serde_json::json!("not an object"),
            serde_json::json!({"status": "completed", "summary": "partial"}),
        ] {
            let tool = ToolView {
                call_id: "run-fallback".into(),
                name: "agent__explore".into(),
                summary: "explorer completed".into(),
                arguments: None,
                output: Some(
                    serde_json::json!({
                        "data": {
                            "agent_name": "explorer",
                            "status": "completed",
                            "summary": "fallback summary",
                            "child_session_id": "child-fallback",
                            "structured_result": structured_result
                        }
                    })
                    .to_string(),
                ),
                status: ToolExecutionStatus::Succeeded,
            };

            let rendered = render_tool_card_lines(&tool, Theme::dark(), 120)
                .into_iter()
                .map(|line| line.to_string())
                .collect::<Vec<_>>();
            assert_eq!(rendered.len(), 1, "{rendered:?}");
            assert!(
                rendered[0].contains("completed explorer fallback summary"),
                "{rendered:?}"
            );
            assert!(!rendered[0].contains("run run-fallback"), "{rendered:?}");
        }
    }

    #[test]
    fn subagent_card_collapsed_and_expanded_render_all_structured_items() {
        let tool = ToolView {
            call_id: "run-expanded".into(),
            name: "agent__explore".into(),
            summary: "explorer completed".into(),
            arguments: None,
            output: Some(
                serde_json::json!({
                    "data": {
                        "agent_name": "explorer",
                        "status": "completed",
                        "summary": "short card summary",
                        "child_session_id": "child-expanded",
                        "structured_result": {
                            "status": "completed",
                            "summary": "The complete expanded summary survives across transcript rows.",
                            "malformed": false,
                            "blockers": ["blocker one", "blocker two"],
                            "findings": ["finding one", "finding two"],
                            "next_steps": ["next step one", "next step two"],
                            "validation": ["validation one", "validation two"],
                            "files_changed": ["changed one", "changed two"],
                            "files_read": ["read one", "read two"],
                            "commands_run": ["command one", "command two"],
                            "run_id": "run-expanded",
                            "child_session_id": "child-expanded"
                        }
                    }
                })
                .to_string(),
            ),
            status: ToolExecutionStatus::Succeeded,
        };

        let collapsed = render_tool_card_lines_with_frame(&tool, Theme::dark(), 80, 0, false)
            .into_iter()
            .map(|line| line.to_string())
            .collect::<Vec<_>>();
        let expanded = render_tool_card_lines_with_frame(&tool, Theme::dark(), 80, 0, true)
            .into_iter()
            .map(|line| line.to_string())
            .collect::<Vec<_>>();
        let collapsed_text = collapsed.join("\n");
        let expanded_text = expanded.join("\n");

        assert_ne!(collapsed, expanded);
        assert!(collapsed_text.contains("expand"), "{collapsed_text}");
        assert!(!collapsed_text.contains("blocker one"), "{collapsed_text}");
        assert!(!collapsed_text.contains("command two"), "{collapsed_text}");
        assert!(
            expanded_text
                .contains("The complete expanded summary survives across transcript rows."),
            "{expanded_text}"
        );
        for item in [
            "blocker one",
            "blocker two",
            "finding one",
            "finding two",
            "next step one",
            "next step two",
            "validation one",
            "validation two",
            "changed one",
            "changed two",
            "read one",
            "read two",
            "command one",
            "command two",
        ] {
            assert!(
                expanded_text.contains(item),
                "missing {item}: {expanded_text}"
            );
        }
    }

    #[test]
    fn subagent_card_summary_only_expands_and_remains_copyable() {
        let tool = ToolView {
            call_id: "run-summary-only".into(),
            name: "agent__explore".into(),
            summary: "explorer completed".into(),
            arguments: None,
            output: Some(
                serde_json::json!({
                    "data": {
                        "agent_name": "explorer",
                        "status": "completed",
                        "summary": "fallback summary",
                        "child_session_id": "child-summary-only",
                        "structured_result": {
                            "status": "completed",
                            "summary": "A summary-only structured result expands in full.",
                            "malformed": false,
                            "blockers": [],
                            "findings": [],
                            "next_steps": [],
                            "validation": [],
                            "files_changed": [],
                            "files_read": [],
                            "commands_run": [],
                            "run_id": "run-summary-only",
                            "child_session_id": "child-summary-only"
                        }
                    }
                })
                .to_string(),
            ),
            status: ToolExecutionStatus::Succeeded,
        };

        let collapsed = render_tool_card_lines_with_frame(&tool, Theme::dark(), 80, 0, false)
            .into_iter()
            .map(|line| line.to_string())
            .collect::<Vec<_>>();
        let expanded_document = render_tool_card_document(&tool, Theme::dark(), 80, 0, true);
        let expanded = render_tool_card_lines_with_frame(&tool, Theme::dark(), 80, 0, true)
            .into_iter()
            .map(|line| line.to_string())
            .collect::<Vec<_>>();

        assert!(
            collapsed
                .join("\n")
                .contains("completed explorer fallback summary")
        );
        assert!(collapsed.join("\n").contains("expand"));
        assert!(
            expanded
                .join("\n")
                .contains("A summary-only structured result expands in full.")
        );
        assert!(
            expanded_document
                .lines
                .iter()
                .flat_map(|line| &line.spans)
                .any(|span| {
                    span.source.is_some() && span.text.contains("A summary-only structured result")
                })
        );
        assert!(expanded_document.validate());
    }

    #[test]
    fn subagent_card_keeps_structured_details_width_safe_and_truncated() {
        let tool = ToolView {
            call_id: "run-narrow".into(),
            name: "agent__fixer".into(),
            summary: "fixer completed".into(),
            arguments: None,
            output: Some(
                serde_json::json!({
                    "data": {
                        "run_id": "run-narrow-with-a-long-identifier",
                        "child_session_id": "child-narrow-with-a-long-identifier",
                        "agent_name": "fixer",
                        "status": "completed",
                        "summary": "a very long summary that must be clipped",
                        "structured_result": {
                            "status": "completed",
                            "summary": "a very long summary that must be clipped",
                            "malformed": false,
                            "findings": ["a very long finding that must be clipped"],
                            "files_read": ["src/one.rs", "src/two.rs"],
                            "files_changed": ["src/changed.rs"],
                            "commands_run": ["cargo test --all-targets"],
                            "validation": ["cargo test passed"],
                            "blockers": [],
                            "next_steps": ["a very long next step that must be clipped"],
                            "run_id": "run-narrow-with-a-long-identifier",
                            "child_session_id": "child-narrow-with-a-long-identifier"
                        }
                    }
                })
                .to_string(),
            ),
            status: ToolExecutionStatus::Succeeded,
        };

        for expanded_output in [false, true] {
            let rendered =
                render_tool_card_lines_with_frame(&tool, Theme::dark(), 32, 0, expanded_output);
            assert!(rendered.iter().any(|line| line.to_string().contains('…')));
            assert!(
                rendered
                    .iter()
                    .all(|line| display_width(&line.to_string()) <= 32),
                "{rendered:?}"
            );
        }
    }

    #[test]
    fn subagent_card_distinguishes_hard_and_logical_failures() {
        for (status, failure_kind, summary) in [
            ("failed", "hard", "provider connection failed"),
            ("failed", "logical", "out-of-scope changes detected"),
            ("timed_out", "hard", "provider timed out"),
            ("cancelled", "logical", "task was cancelled by the delegate"),
        ] {
            let tool = ToolView {
                call_id: "run-failure".into(),
                name: "agent__fixer".into(),
                summary: summary.into(),
                arguments: None,
                output: Some(
                    serde_json::json!({
                        "data": {
                            "agent_name": "fixer",
                            "status": status,
                            "failure_kind": failure_kind,
                            "summary": summary,
                            "child_session_id": "child-session-1234567890"
                        }
                    })
                    .to_string(),
                ),
                status: ToolExecutionStatus::Failed,
            };

            let rendered = render_tool_card_lines(&tool, Theme::dark(), 140)
                .into_iter()
                .map(|line| line.to_string())
                .collect::<Vec<_>>();

            assert_eq!(rendered.len(), 1, "{rendered:?}");
            assert!(
                rendered[0].contains(&format!("{status} [{failure_kind}] fixer {summary}")),
                "{}",
                rendered[0]
            );
        }
    }

    #[test]
    fn permission_card_lines_use_composer_style_status_guide() {
        let permission = PermissionView {
            call_id: "perm-fill".into(),
            tool_name: "shell__exec".into(),
            summary: "echo ok".into(),
            arguments: Some("echo ok".into()),
            rationale: None,
            origin_label: None,
            can_allow_always: false,
            grant_summary: None,
            status: PermissionPromptStatus::Pending,
            resolution_reason: None,
        };

        let theme = Theme::dark();
        let width = 72usize;
        let lines = render_permission_card_lines(&permission, theme, width);
        assert!(!lines.is_empty());

        for line in &lines {
            assert!(display_width(&line.to_string()) <= width, "{line:?}");
            let guide = line.spans.first().expect("line has guide span");
            assert_eq!(guide.content.as_ref(), TOOL_GUIDE_GLYPH);
            assert_eq!(guide.style.fg, Some(theme.approval));
            assert_eq!(guide.style.bg, Some(theme.root_bg));
        }
        assert_eq!(lines.len(), 1);
        let rendered = lines[0].to_string();
        assert!(
            (rendered.contains("待处理") || rendered.contains("pending"))
                && rendered.contains("shell__exec echo ok"),
            "{rendered}"
        );
    }

    #[test]
    fn tool_card_render_truncates_snippet_by_display_width_cjk() {
        let theme = Theme::dark();
        let tool = ToolView {
            call_id: "call-cjk".into(),
            name: "shell__exec".into(),
            summary: "run".into(),
            arguments: Some("你好吗你好吗你好吗".into()),
            output: None,
            status: ToolExecutionStatus::Running,
        };
        let width = 24usize;
        let lines = render_tool_card_lines(&tool, theme, width);
        assert!(!lines.is_empty());
        for line in &lines {
            let w = display_width(&line.to_string());
            assert!(w <= width, "line width {w} > {width}: {}", line);
        }
    }

    fn segment_texts(segments: &[(String, Style)]) -> Vec<&str> {
        segments.iter().map(|(text, _)| text.as_str()).collect()
    }

    fn assert_segments_have_no_controls(segments: &[(String, Style)]) {
        for (text, _) in segments {
            assert!(
                !text.chars().any(|ch| ch.is_control()),
                "control leaked into segment text: {text:?}"
            );
        }
    }

    #[test]
    fn shell_output_drops_non_sgr_csi_from_visible_text() {
        let base = root_text_style(Theme::dark()).bg(Theme::dark().card_bg());
        let cases = [
            ("keep\u{1b}[2Kdone", "keepdone"),
            ("up\u{1b}[1Aline", "upline"),
            ("clear\u{1b}[2Jscreen", "clearscreen"),
            ("hide\u{1b}[?25lcursor", "hidecursor"),
        ];
        for (input, expected) in cases {
            let segments = ansi_sgr_segments(input, base);
            assert_segments_have_no_controls(&segments);
            assert_eq!(
                segment_texts(&segments).join(""),
                expected,
                "input={input:?}"
            );
        }
    }

    #[test]
    fn shell_output_strips_osc8_hyperlink_wrappers() {
        let base = root_text_style(Theme::dark()).bg(Theme::dark().card_bg());
        let segments = ansi_sgr_segments(
            "\u{1b}]8;;http://example.test\u{07}link\u{1b}]8;;\u{07}",
            base,
        );
        assert_segments_have_no_controls(&segments);
        assert_eq!(segment_texts(&segments), vec!["link"]);
    }

    #[test]
    fn shell_output_drops_truncated_escape_without_leaking_esc() {
        let base = root_text_style(Theme::dark()).bg(Theme::dark().card_bg());
        let truncated = ansi_sgr_segments("plain\u{1b}[3", base);
        assert_segments_have_no_controls(&truncated);
        assert_eq!(segment_texts(&truncated), vec!["plain"]);

        let completed = ansi_sgr_segments("plain\u{1b}[31mred", base);
        assert_segments_have_no_controls(&completed);
        assert_eq!(segment_texts(&completed), vec!["plain", "red"]);
        assert_eq!(completed[1].1.fg, Some(Color::Rgb(205, 49, 49)));
    }

    #[test]
    fn diff_uses_single_column_layout_at_narrow_width_without_overflow() {
        let tool = ToolView {
            call_id: "call-narrow-edit".into(),
            name: "edit__apply_patch".into(),
            summary: "patched".into(),
            arguments: Some(
                json!({
                    "edits": [{
                        "path": "src/main.rs",
                        "find": "old line",
                        "replace": "new line",
                        "replace_all": false
                    }]
                })
                .to_string(),
            ),
            output: None,
            status: ToolExecutionStatus::Succeeded,
        };
        let lines = render_tool_card_lines(&tool, Theme::dark(), 40);
        let rendered = lines
            .iter()
            .map(Line::to_string)
            .collect::<Vec<_>>()
            .join("\n");

        assert!(
            !rendered.contains(DIFF_SIDE_BY_SIDE_SEPARATOR),
            "{rendered}"
        );
        assert!(rendered.contains("- old line"), "{rendered}");
        assert!(rendered.contains("+ new line"), "{rendered}");
        assert!(
            lines
                .iter()
                .all(|line| display_width(&line.to_string()) <= 40),
            "{rendered}"
        );
    }

    #[test]
    fn patch_diff_sanitizes_terminal_controls_in_side_by_side_layout() {
        let tool = ToolView {
            call_id: "call-safe-edit".into(),
            name: "edit__apply_patch".into(),
            summary: "patched".into(),
            arguments: Some(
                json!({
                    "edits": [{
                        "path": "src/\u{1b}[2Kmain.rs",
                        "find": "\told value\u{1b}[2K",
                        "replace": "\tnew value\u{1b}]8;;https://example.test\u{07}",
                        "replace_all": false
                    }]
                })
                .to_string(),
            ),
            output: None,
            status: ToolExecutionStatus::Succeeded,
        };
        let width = 100;
        let lines = render_tool_card_lines(&tool, Theme::dark(), width);
        let rendered = lines
            .iter()
            .map(Line::to_string)
            .collect::<Vec<_>>()
            .join("\n");

        assert!(rendered.contains(DIFF_SIDE_BY_SIDE_SEPARATOR), "{rendered}");
        assert!(rendered.contains("old value"), "{rendered}");
        assert!(rendered.contains("new value"), "{rendered}");
        assert!(
            lines
                .iter()
                .all(|line| !line.to_string().chars().any(char::is_control)),
            "control leaked into patch card: {rendered:?}"
        );
        assert!(
            lines
                .iter()
                .all(|line| display_width(&line.to_string()) <= width),
            "{rendered}"
        );
    }

    #[test]
    fn permission_denied_shows_resolution_as_error_snippet() {
        let permission = PermissionView {
            call_id: "perm-1".into(),
            tool_name: "shell__exec".into(),
            summary: "dangerous".into(),
            arguments: None,
            rationale: None,
            origin_label: None,
            can_allow_always: false,
            grant_summary: None,
            status: PermissionPromptStatus::Denied,
            resolution_reason: Some("not allowed by policy".into()),
        };

        let details = permission_card_details(&permission);
        assert_eq!(details.status, ToolCardStatus::Denied);
        assert!(
            details
                .fields
                .iter()
                .any(|(k, v)| k == "resolution" && v.contains("not allowed"))
        );
    }

    #[test]
    fn permission_cards_show_pending_approved_denied_and_context_fields() {
        let theme = Theme::dark();

        let pending = PermissionView {
            call_id: "perm-p".into(),
            tool_name: "shell__exec".into(),
            summary: "rm -rf /".into(),
            arguments: Some("rm -rf /".into()),
            rationale: Some("requested by user".into()),
            origin_label: None,
            can_allow_always: false,
            grant_summary: None,
            status: PermissionPromptStatus::Pending,
            resolution_reason: None,
        };
        let approved = PermissionView {
            call_id: "perm-a".into(),
            tool_name: "shell__exec".into(),
            summary: "touch file".into(),
            arguments: Some("touch a.txt".into()),
            rationale: Some("needed".into()),
            origin_label: None,
            can_allow_always: false,
            grant_summary: None,
            status: PermissionPromptStatus::Approved,
            resolution_reason: None,
        };
        let denied = PermissionView {
            call_id: "perm-d".into(),
            tool_name: "shell__exec".into(),
            summary: "format disk".into(),
            arguments: Some("mkfs".into()),
            rationale: Some("unsafe".into()),
            origin_label: None,
            can_allow_always: false,
            grant_summary: None,
            status: PermissionPromptStatus::Denied,
            resolution_reason: Some("policy".into()),
        };

        let p = render_permission_card_lines(&pending, theme, 80)
            .iter()
            .map(|l| l.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        let a = render_permission_card_lines(&approved, theme, 80)
            .iter()
            .map(|l| l.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        let d = render_permission_card_lines(&denied, theme, 80)
            .iter()
            .map(|l| l.to_string())
            .collect::<Vec<_>>()
            .join("\n");

        assert!(p.contains("待处理") || p.contains("pending"), "{p}");
        assert!(a.contains("已批准") || a.contains("approved"), "{a}");
        assert!(d.contains("已拒绝") || d.contains("denied"), "{d}");

        assert!(p.contains("shell__exec"), "{p}");
        assert!(p.contains("rm -rf /"), "{p}");
        assert!(p.contains("requested by user"), "{p}");
        assert!(d.contains("policy"), "{d}");
        assert!(!p.contains("call"), "{p}");
        assert!(!p.contains("perm-p"), "{p}");
        assert!(!p.contains("args"), "{p}");
        assert!(!p.contains("why"), "{p}");

        let auto = PermissionView {
            call_id: "perm-auto".into(),
            tool_name: "fs__write".into(),
            summary: "fs__write a.txt".into(),
            arguments: Some(r#"{"path":"a.txt"}"#.into()),
            rationale: None,
            origin_label: Some("reviewer".into()),
            can_allow_always: false,
            grant_summary: None,
            status: PermissionPromptStatus::Approved,
            resolution_reason: Some("safe edit".into()),
        };
        let auto_line = render_permission_card_lines(&auto, theme, 80)
            .iter()
            .map(|l| l.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(auto_line.contains("reviewer"), "{auto_line}");
        assert!(auto_line.contains("fs__write"), "{auto_line}");
        assert!(auto_line.contains("safe edit"), "{auto_line}");
    }

    #[test]
    fn permission_card_lines_are_width_safe_at_narrow_and_normal_widths() {
        let theme = Theme::dark();

        let denied = PermissionView {
            call_id: "perm-w".into(),
            tool_name: "shell__exec".into(),
            summary: "danger".into(),
            arguments: Some("--flag ".repeat(20)),
            rationale: Some("because ".repeat(30)),
            origin_label: None,
            can_allow_always: false,
            grant_summary: None,
            status: PermissionPromptStatus::Denied,
            resolution_reason: Some("resolution reason ".repeat(20)),
        };

        for width in [12usize, 18, 26, 44, 80] {
            let lines = render_permission_card_lines(&denied, theme, width);
            assert!(!lines.is_empty());
            for line in &lines {
                let w = display_width(&line.to_string());
                assert!(
                    w <= width,
                    "permission line width {w} > {width}: {:?}",
                    line.to_string()
                );
            }
        }
    }

    #[test]
    fn permission_pending_and_approved_are_width_safe_with_args_and_rationale() {
        let theme = Theme::dark();

        let pending = PermissionView {
            call_id: "perm-pw".into(),
            tool_name: "shell__exec".into(),
            summary: "pending".into(),
            arguments: Some("arg ".repeat(60)),
            rationale: Some("why ".repeat(80)),
            origin_label: None,
            can_allow_always: false,
            grant_summary: None,
            status: PermissionPromptStatus::Pending,
            resolution_reason: None,
        };
        let approved = PermissionView {
            call_id: "perm-aw".into(),
            tool_name: "shell__exec".into(),
            summary: "approved".into(),
            arguments: Some("arg ".repeat(60)),
            rationale: Some("why ".repeat(80)),
            origin_label: None,
            can_allow_always: false,
            grant_summary: None,
            status: PermissionPromptStatus::Approved,
            resolution_reason: None,
        };

        for width in [12usize, 18, 26, 44, 80] {
            for p in [&pending, &approved] {
                let lines = render_permission_card_lines(p, theme, width);
                assert!(!lines.is_empty());
                for line in &lines {
                    let w = display_width(&line.to_string());
                    assert!(
                        w <= width,
                        "permission line width {w} > {width}: {:?}",
                        line.to_string()
                    );
                }
            }
        }
    }

    #[test]
    fn tool_document_registers_shell_semantic_leaves() {
        let tool = ToolView {
            call_id: "shell-1".into(),
            name: "shell__exec".into(),
            summary: "run echo".into(),
            arguments: Some(json!({"command": "echo hello"}).to_string()),
            output: Some(json!({"data": {"stdout": "hello", "stderr": "warn"}}).to_string()),
            status: ToolExecutionStatus::Succeeded,
        };
        let document = render_tool_card_document(&tool, Theme::dark(), 80, 0, false);
        let copied = document
            .lines
            .iter()
            .flat_map(|line| line.spans.iter())
            .filter(|span| span.source.is_some())
            .map(|span| span.text.as_str())
            .collect::<String>();
        assert!(copied.contains("echo hello"));
        assert!(copied.contains("hello"));
        assert!(copied.contains("warn"));
        assert!(!copied.contains("$ "));
        assert!(!copied.contains("stdout"));
        assert!(!copied.contains("stderr"));
        assert!(document.validate());
    }

    #[test]
    fn permission_document_excludes_fixed_labels() {
        let permission = PermissionView {
            call_id: "perm-1".into(),
            tool_name: "shell__exec".into(),
            summary: "Run echo hello".into(),
            arguments: None,
            rationale: Some("needed for validation".into()),
            origin_label: None,
            can_allow_always: false,
            grant_summary: None,
            status: PermissionPromptStatus::Approved,
            resolution_reason: None,
        };
        let document = render_permission_card_document(&permission, Theme::dark(), 80);
        let copied = document
            .lines
            .iter()
            .flat_map(|line| line.spans.iter())
            .filter(|span| span.source.is_some())
            .map(|span| span.text.as_str())
            .collect::<String>();
        assert!(copied.contains("Run echo hello"));
        assert!(copied.contains("needed for validation"));
        assert!(!copied.contains("approved"));
        assert!(!copied.contains("shell__exec"));
        assert!(document.validate());
    }

    #[test]
    fn clipped_semantic_spans_preserve_copy_join() {
        let style = Style::default();
        let clipped = clip_semantic_spans(
            vec![SemanticSpan::source_with_join(
                "reason text",
                style,
                CopyJoin::Space,
            )],
            7,
        );

        assert_eq!(clipped[0].text, "reason");
        assert_eq!(clipped[0].copy_join, CopyJoin::Space);
        assert!(clipped[0].copy);
    }

    #[test]
    fn shell_document_marks_semantic_section_boundaries() {
        let tool = ToolView {
            call_id: "shell-boundaries".into(),
            name: "shell__exec".into(),
            summary: "run".into(),
            arguments: Some(json!({"command": "printf test"}).to_string()),
            output: Some(
                json!({"data": {"stdout": "out one\nout two", "stderr": "err one\nerr two"}})
                    .to_string(),
            ),
            status: ToolExecutionStatus::Succeeded,
        };
        let document = render_tool_card_document(&tool, Theme::dark(), 80, 0, false);
        let source_lines = document
            .lines
            .iter()
            .enumerate()
            .filter(|(_, line)| line.spans.iter().any(|span| span.source.is_some()))
            .map(|(index, line)| {
                (
                    line.spans
                        .iter()
                        .filter(|span| span.source.is_some())
                        .map(|span| span.text.as_str())
                        .collect::<String>(),
                    document.break_after(index),
                )
            })
            .collect::<Vec<_>>();

        assert_eq!(
            source_lines[0],
            ("printf test".into(), Some(Break::BlockBreak))
        );
        assert_eq!(source_lines[1], ("out one".into(), Some(Break::HardBreak)));
        assert_eq!(source_lines[2], ("out two".into(), Some(Break::BlockBreak)));
        assert_eq!(source_lines[3], ("err one".into(), Some(Break::HardBreak)));
        assert_eq!(source_lines[4].0, "err two");
    }
}
