#[cfg(test)]
use ratatui::style::Color;
use ratatui::style::{Modifier, Style};
#[cfg(test)]
use ratatui::text::Line;
use serde_json::Value;

pub use super::semantic_spans::ToolCardStatus;
pub(crate) use super::semantic_spans::truncate_display_width;
use super::semantic_spans::*;
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
        &crate::tui::i18n::Translator::new(crate::tui::i18n::Language::En),
    ))
}

pub(crate) fn tool_is_active(tool: &ToolView) -> bool {
    match tool.status {
        ToolExecutionStatus::Pending | ToolExecutionStatus::Running => true,
        ToolExecutionStatus::Failed | ToolExecutionStatus::Cancelled => false,
        ToolExecutionStatus::Succeeded => {
            crate::agent::is_subagent_tool_name(&tool.name)
                && tool
                    .output
                    .as_deref()
                    .and_then(|output| serde_json::from_str::<serde_json::Value>(output).ok())
                    .and_then(|value| {
                        value
                            .get("data")
                            .unwrap_or(&value)
                            .get("active")
                            .and_then(serde_json::Value::as_bool)
                    })
                    == Some(true)
        }
    }
}

/// Renderer-neutral compact summary for a contiguous tool group.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ToolGroupStats {
    pub call_count: usize,
    pub failed_count: usize,
    pub cancelled_count: usize,
    pub active: bool,
}

pub fn render_tool_group_document(
    latest: &ToolView,
    stats: ToolGroupStats,
    theme: Theme,
    width: usize,
    frame: usize,
    translator: &crate::tui::i18n::Translator,
) -> Document<Style> {
    let mut document = Document::default();
    if width == 0 || stats.call_count == 0 {
        return document;
    }
    let glyph = if stats.active {
        PROCESS_FRAMES[frame % PROCESS_FRAMES.len()]
    } else {
        "→"
    };
    let aggregate_status = if stats.active {
        ToolExecutionStatus::Running
    } else if stats.failed_count > 0 {
        ToolExecutionStatus::Failed
    } else if stats.cancelled_count > 0 {
        ToolExecutionStatus::Cancelled
    } else {
        ToolExecutionStatus::Succeeded
    };
    let style = tool_trace_text_style(aggregate_status, theme);
    let mut header_spans = vec![
        SemanticSpan::decoration("  ", theme.app_style()),
        SemanticSpan::decoration(
            format!("{glyph} "),
            tool_trace_arrow_style(aggregate_status, theme),
        ),
        SemanticSpan::decoration(
            format!("Tools · {}", stats.call_count),
            style.add_modifier(Modifier::BOLD),
        ),
    ];
    if stats.failed_count > 0 {
        header_spans.push(SemanticSpan::decoration(
            format!(
                " · {} {}",
                stats.failed_count,
                translator.t("status.failed")
            ),
            style,
        ));
    }
    if stats.cancelled_count > 0 {
        header_spans.push(SemanticSpan::decoration(
            format!(
                " · {} {}",
                stats.cancelled_count,
                translator.t("status.cancelled")
            ),
            style,
        ));
    }
    document.push_semantic_line(SemanticLine {
        spans: clip_semantic_spans(header_spans, width),
        boundary: Break::HardBreak,
    });

    let mut trace_spans = vec![SemanticSpan::decoration("  ", theme.app_style())];
    trace_spans.extend(tool_trace_segments(latest, style));
    document.push_semantic_line(SemanticLine {
        spans: clip_semantic_spans(trace_spans, width),
        boundary: Break::End,
    });
    document.finish();
    debug_assert!(document.validate());
    document
}

pub fn render_tool_card_document(
    tool: &ToolView,
    theme: Theme,
    width: usize,
    frame: usize,
    expanded_output: bool,
    translator: &crate::tui::i18n::Translator,
) -> Document<Style> {
    let policy = PresentationPolicy;
    let mut document = Document::default();
    if tool_card_details(tool, &policy).is_none() {
        return document;
    }
    if matches!(
        tool.name.as_str(),
        crate::tool_names::TOOL_AGENT_SEND | crate::tool_names::TOOL_PEER_SEND
    ) {
        return render_subagent_send_document(
            tool,
            theme,
            width,
            frame,
            expanded_output,
            translator,
        );
    }
    let lines = if is_subagent_tool(&tool.name) || tool.name == crate::tool_names::TOOL_AGENT_WAIT {
        render_subagent_lines(tool, theme, width, frame, expanded_output, translator)
    } else {
        let body = render_tool_body_lines(tool, theme, width, frame, expanded_output);
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
    frame: usize,
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
        "fs__write" => render_write_diff_lines(tool, theme, width, expanded_output),
        "fs__append" => render_append_diff_lines(tool, theme, width, expanded_output),
        "shell__exec" => render_shell_output_lines(tool, theme, width, frame, expanded_output),
        "edit__apply_patch" => render_edit_diff_lines(tool, theme, width, expanded_output),
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
        "git" => format!("Git {}", value_str(args, "action").unwrap_or("status")),
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
        "git" => "Git".into(),
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

fn permission_accent(status: PermissionPromptStatus, theme: Theme) -> ratatui::style::Color {
    match status {
        PermissionPromptStatus::Pending => theme.approval,
        PermissionPromptStatus::Approved => theme.success,
        PermissionPromptStatus::Denied => theme.error,
    }
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

#[cfg(test)]
#[path = "../../../tests/internal/tui/components/tool_card/tests.rs"]
mod tests;
