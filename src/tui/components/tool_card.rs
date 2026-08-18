use ratatui::style::{Modifier, Style};
#[cfg(test)]
use ratatui::style::Color;
#[cfg(test)]
use ratatui::text::Line;
use serde_json::Value;

use super::semantic_spans::*;
pub(crate) use super::semantic_spans::truncate_display_width;
use super::{diff_render::*, question_card::*, shell_output::*, subagent_card::*};
use crate::tui::{
    measure::display_width,
    presentation::{
        PresentationPolicy, ToolPresentation, ToolPresentationStatus, ToolTextPresentationContext,
    },
    theme::Theme,
    timeline::{PermissionPromptStatus, PermissionView, ToolExecutionStatus, ToolView},
    transcript_render::{Break, CopyJoin, Document, SemanticLine, SemanticSpan},
};

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
                SemanticSpan::source(terminal_safe_text(pattern), style),
                SemanticSpan::decoration(" in ", style),
                SemanticSpan::source_with_join(terminal_safe_text(path), style, CopyJoin::Space),
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
        SemanticSpan::source(terminal_safe_text(value), style),
    ]
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

fn sentence_case_tool_name(name: &str) -> String {
    let label = name.replace('_', " ");
    let mut chars = label.chars();
    let Some(first) = chars.next() else {
        return "Tool".into();
    };
    format!("{}{}", first.to_uppercase(), chars.as_str())
}

pub(super) fn map_tool_status(status: ToolExecutionStatus) -> ToolCardStatus {
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

pub(super) fn status_label(status: ToolCardStatus, translator: &crate::tui::i18n::Translator) -> String {
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

pub(super) fn root_status_style(color: ratatui::style::Color, theme: Theme) -> ratatui::style::Style {
    ratatui::style::Style::default()
        .fg(color)
        .bg(theme.root_bg)
        .add_modifier(ratatui::style::Modifier::BOLD)
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
    fn write_diff_sanitizes_terminal_controls_across_layouts() {
        // fs__write renders its full content as a diff (render_write_diff_lines);
        // the body previously bypassed ansi_sgr_segments and could tear the TUI
        // when the written text carried ANSI/control characters. The on-disk
        // content is untouched — filtering is presentation-only.
        let tool = ToolView {
            call_id: "call-safe-write".into(),
            name: "fs__write".into(),
            summary: "wrote file".into(),
            arguments: Some(
                json!({
                    "path": "notes/\u{1b}[2Kreadme.txt",
                    "content": "line one\n\tline two\u{1b}[2K\ncolor\u{1b}[31m red\u{1b}[0m\n"
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

        assert!(
            lines
                .iter()
                .all(|line| !line.to_string().chars().any(char::is_control)),
            "control leaked into write card: {rendered:?}"
        );
        assert!(rendered.contains("line one"), "{rendered}");
        assert!(rendered.contains("line two"), "{rendered}");
        assert!(rendered.contains("red"), "{rendered}");
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
