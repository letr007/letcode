use ratatui::{
    style::{Modifier, Style},
    text::{Line, Span},
};

use crate::tui::{
    measure::{display_width, wrap_text_to_width},
    presentation::{
        PresentationPolicy, ToolPresentation, ToolPresentationStatus, ToolTextPresentationContext,
    },
    surface,
    theme::Theme,
    timeline::{PermissionPromptStatus, PermissionView, ToolExecutionStatus, ToolView},
};

const TOOL_GUIDE_GLYPH: &str = surface::ACCENT_BAR_GLYPH;
const TOOL_CARD_GUIDE: ratatui::style::Color = ratatui::style::Color::Rgb(76, 80, 96);
const DIFF_CARD_BG: ratatui::style::Color = ratatui::style::Color::Rgb(30, 30, 32);
const DIFF_CARD_GUTTER: ratatui::style::Color = ratatui::style::Color::Rgb(112, 118, 134);
const DIFF_CARD_GUTTER_BG: ratatui::style::Color = ratatui::style::Color::Rgb(30, 30, 32);
const DIFF_CARD_TEXT: ratatui::style::Color = ratatui::style::Color::Rgb(222, 226, 236);
const DIFF_CARD_META: ratatui::style::Color = ratatui::style::Color::Rgb(143, 151, 170);
const DIFF_CARD_ADD_SIGN: ratatui::style::Color = ratatui::style::Color::Rgb(107, 211, 145);
const DIFF_CARD_DELETE_SIGN: ratatui::style::Color = ratatui::style::Color::Rgb(239, 126, 139);
const DIFF_CARD_ADD_BG: ratatui::style::Color = ratatui::style::Color::Rgb(22, 45, 32);
const DIFF_CARD_DELETE_BG: ratatui::style::Color = ratatui::style::Color::Rgb(54, 32, 42);
const DIFF_CARD_HUNK_BG: ratatui::style::Color = ratatui::style::Color::Rgb(31, 40, 60);
const DIFF_CARD_HEADER_ARROW: &str = "←";
const PROCESS_FRAMES: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolCardStatus {
    Pending,
    Approved,
    Running,
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
pub fn render_tool_card_lines(tool: &ToolView, theme: Theme, width: usize) -> Vec<Line<'static>> {
    render_tool_card_lines_with_frame(tool, theme, width, 0)
}

pub fn render_tool_card_lines_with_frame(
    tool: &ToolView,
    theme: Theme,
    width: usize,
    frame: usize,
) -> Vec<Line<'static>> {
    let policy = PresentationPolicy;
    if tool_card_details(tool, &policy).is_none() {
        return Vec::new();
    }

    if is_subagent_tool(&tool.name) {
        return render_subagent_lines(tool, theme, width, frame);
    }

    let body = render_tool_body_lines(tool, theme, width);
    if body.is_empty() {
        vec![render_tool_trace_line(tool, theme, width, frame)]
    } else {
        body
    }
}

pub fn render_permission_card_lines(
    permission: &PermissionView,
    theme: Theme,
    width: usize,
) -> Vec<Line<'static>> {
    if width == 0 {
        return Vec::new();
    }

    let status = status_label(match permission.status {
        PermissionPromptStatus::Pending => ToolCardStatus::Pending,
        PermissionPromptStatus::Approved => ToolCardStatus::Approved,
        PermissionPromptStatus::Denied => ToolCardStatus::Denied,
    });
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
        (status.to_string(), status_style),
        (" ".to_string(), text_style),
        (
            permission.tool_name.clone(),
            text_style.add_modifier(Modifier::BOLD),
        ),
        (" ".to_string(), text_style),
        (summary, text_style),
    ];
    if let Some(reason) = reason {
        segments.push((" · ".to_string(), muted_style));
        segments.push((reason, muted_style));
    }

    vec![render_card_line_with_guide(
        &segments,
        Style::default().bg(theme.root_bg),
        permission_accent(permission.status, theme),
        theme,
        width,
    )]
}

fn render_tool_trace_line(
    tool: &ToolView,
    theme: Theme,
    width: usize,
    frame: usize,
) -> Line<'static> {
    if width == 0 {
        return Line::from("");
    }

    let glyph = if matches!(
        tool.status,
        ToolExecutionStatus::Pending | ToolExecutionStatus::Running
    ) {
        PROCESS_FRAMES[frame % PROCESS_FRAMES.len()]
    } else {
        "→"
    };
    let prefix = format!("  {glyph} ");
    let arrow_style = tool_trace_arrow_style(tool.status, theme);
    let text_style = tool_trace_text_style(tool.status, theme);
    let status_suffix = match tool.status {
        ToolExecutionStatus::Pending => " …",
        ToolExecutionStatus::Running => " …",
        ToolExecutionStatus::Succeeded => "",
        ToolExecutionStatus::Failed => " · failed",
    };
    let text_budget = width.saturating_sub(display_width(&prefix));
    let text = truncate_display_width(
        &format!("{}{}", tool_trace_label(tool), status_suffix),
        text_budget,
    );

    Line::from(vec![
        Span::styled("  ", theme.app_style()),
        Span::styled(format!("{glyph} "), arrow_style),
        Span::styled(text, text_style),
    ])
}

fn render_tool_body_lines(tool: &ToolView, theme: Theme, width: usize) -> Vec<Line<'static>> {
    if width == 0
        || matches!(
            tool.status,
            ToolExecutionStatus::Pending | ToolExecutionStatus::Running
        )
    {
        return Vec::new();
    }

    match tool.name.as_str() {
        "fs__write" => render_write_diff_lines(tool, theme, width),
        "fs__append" => render_append_diff_lines(tool, theme, width),
        "shell__exec" => render_shell_output_lines(tool, theme, width),
        "edit__apply_patch" => render_edit_diff_lines(tool, theme, width),
        _ => Vec::new(),
    }
}

fn render_subagent_lines(
    tool: &ToolView,
    theme: Theme,
    width: usize,
    frame: usize,
) -> Vec<Line<'static>> {
    if width == 0 {
        return Vec::new();
    }

    let data = tool_output_data(tool);
    let status = data
        .as_ref()
        .and_then(|data| data.get("status").and_then(serde_json::Value::as_str))
        .unwrap_or_else(|| subagent_status_label(tool.status));
    let child = data
        .as_ref()
        .and_then(|data| data.get("child_session_id"))
        .and_then(serde_json::Value::as_str)
        .map(|id| format!("/child {}", truncate_display_width(id, 16)))
        .unwrap_or_else(|| "/child".into());
    let summary = data
        .as_ref()
        .and_then(|data| data.get("summary"))
        .and_then(serde_json::Value::as_str)
        .map(one_line_snippet)
        .filter(|summary| !summary.is_empty())
        .or_else(|| subagent_task(tool))
        .unwrap_or_else(|| one_line_snippet(&tool.summary));
    let agent_name = data
        .as_ref()
        .and_then(|data| data.get("agent_name"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or_else(|| subagent_name_from_tool(&tool.name));

    let status_label = if matches!(
        tool.status,
        ToolExecutionStatus::Pending | ToolExecutionStatus::Running
    ) {
        format!(
            "{} {}",
            PROCESS_FRAMES[frame % PROCESS_FRAMES.len()],
            status_label(map_tool_status(tool.status))
        )
    } else {
        status.to_string()
    };
    let status_color = match tool.status {
        ToolExecutionStatus::Pending => theme.warning,
        ToolExecutionStatus::Running => theme.warning,
        ToolExecutionStatus::Succeeded => theme.notice,
        ToolExecutionStatus::Failed => theme.error,
    };

    let status_style = root_status_style(status_color, theme);
    let text_style = root_text_style(theme);
    let muted = root_muted_style(theme);

    vec![render_card_line(
        &[
            (status_label, status_style),
            (" ".to_string(), text_style),
            (
                agent_name.to_string(),
                text_style.add_modifier(Modifier::BOLD),
            ),
            (" ".to_string(), text_style),
            (summary, text_style),
            (" · ".to_string(), muted),
            (child, muted),
        ],
        Style::default().bg(theme.root_bg),
        theme,
        width,
    )]
}

fn subagent_task(tool: &ToolView) -> Option<String> {
    tool_arguments(tool)
        .as_ref()
        .and_then(|args| value_str(Some(args), "task"))
        .map(one_line_snippet)
        .filter(|task| !task.is_empty())
}

fn is_subagent_tool(name: &str) -> bool {
    matches!(name, "agent__explore" | "agent__fixer")
}

fn subagent_name_from_tool(name: &str) -> &str {
    match name {
        "agent__explore" => "explorer",
        "agent__fixer" => "fixer",
        _ => "subagent",
    }
}

fn subagent_status_label(status: ToolExecutionStatus) -> &'static str {
    match status {
        ToolExecutionStatus::Pending => "preparing",
        ToolExecutionStatus::Running => "running",
        ToolExecutionStatus::Succeeded => "completed",
        ToolExecutionStatus::Failed => "failed",
    }
}

fn render_write_diff_lines(tool: &ToolView, theme: Theme, width: usize) -> Vec<Line<'static>> {
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

fn render_append_diff_lines(tool: &ToolView, theme: Theme, width: usize) -> Vec<Line<'static>> {
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

fn render_shell_output_lines(tool: &ToolView, theme: Theme, width: usize) -> Vec<Line<'static>> {
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

    let mut lines = render_shell_card_header_lines(tool, theme, width);
    if !stdout.trim().is_empty() {
        lines.extend(render_shell_output_section(
            output_title("stdout", data.get("stdout_truncated")),
            stdout,
            root_text_style(theme),
            theme,
            width,
        ));
    }
    if !stderr.trim().is_empty() {
        if lines.len() > 4 {
            lines.push(render_card_line(
                &[],
                Style::default().bg(DIFF_CARD_BG),
                theme,
                width,
            ));
        }
        lines.extend(render_shell_output_section(
            output_title("stderr", data.get("stderr_truncated")),
            stderr,
            theme.error_style().bg(theme.root_bg),
            theme,
            width,
        ));
    }
    lines
}

fn render_shell_card_header_lines(
    tool: &ToolView,
    theme: Theme,
    width: usize,
) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    let command = shell_command(tool);
    let title = shell_card_title(tool, command.as_deref());

    lines.push(render_card_line(
        &[],
        Style::default().bg(DIFF_CARD_BG),
        theme,
        width,
    ));
    lines.push(render_card_line(
        &[(format!("# {title}"), shell_card_title_style())],
        Style::default().bg(DIFF_CARD_BG),
        theme,
        width,
    ));
    lines.push(render_card_line(
        &[],
        Style::default().bg(DIFF_CARD_BG),
        theme,
        width,
    ));

    if let Some(command) = command {
        for (index, wrapped) in
            wrap_text_to_width(&format!("$ {command}"), shell_card_content_width(width))
                .into_iter()
                .enumerate()
        {
            let prefix = if index == 0 { "" } else { "  " };
            lines.push(render_card_line(
                &[(format!("{prefix}{wrapped}"), shell_card_command_style())],
                Style::default().bg(DIFF_CARD_BG),
                theme,
                width,
            ));
        }
        lines.push(render_card_line(
            &[],
            Style::default().bg(DIFF_CARD_BG),
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
) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    lines.push(render_card_line(
        &[(
            title.to_string(),
            root_muted_style(theme)
                .bg(DIFF_CARD_BG)
                .add_modifier(Modifier::BOLD),
        )],
        Style::default().bg(DIFF_CARD_BG),
        theme,
        width,
    ));
    lines.push(render_card_line(
        &[],
        Style::default().bg(DIFF_CARD_BG),
        theme,
        width,
    ));
    lines.extend(render_limited_text_lines(text, text_style, theme, width));
    lines.push(render_card_line(
        &[],
        Style::default().bg(DIFF_CARD_BG),
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
            .map(|command| sentence_case_command_goal(command))
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

fn render_edit_diff_lines(tool: &ToolView, theme: Theme, width: usize) -> Vec<Line<'static>> {
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
            diff.push_str(line);
            diff.push('\n');
        }
        for line in replace.lines() {
            diff.push('+');
            diff.push_str(line);
            diff.push('\n');
        }
    }

    if diff.trim().is_empty() {
        Vec::new()
    } else {
        let paths = edits
            .iter()
            .filter_map(|edit| edit.get("path").and_then(serde_json::Value::as_str))
            .map(ToOwned::to_owned)
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
) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    lines.push(render_card_line(
        &[],
        Style::default().bg(DIFF_CARD_BG),
        theme,
        width,
    ));
    lines.push(render_card_line(
        &[(
            format!("{title}"),
            root_muted_style(theme)
                .bg(DIFF_CARD_BG)
                .add_modifier(Modifier::BOLD),
        )],
        Style::default().bg(DIFF_CARD_BG),
        theme,
        width,
    ));
    lines.push(render_card_line(
        &[],
        Style::default().bg(DIFF_CARD_BG),
        theme,
        width,
    ));
    lines.extend(render_limited_text_lines(text, text_style, theme, width));
    lines.push(render_card_line(
        &[],
        Style::default().bg(DIFF_CARD_BG),
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
) -> Vec<Line<'static>> {
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

    let max_lines = max_body_lines();
    let mut state = DiffLineNumbers::default();
    for (idx, line) in diff
        .lines()
        .filter(|line| !is_diff_file_header_line(line))
        .enumerate()
    {
        if idx >= max_lines {
            lines.push(render_diff_card_body_line(
                None,
                None,
                "… output clipped in TUI",
                diff_meta_style(),
                theme,
                width,
            ));
            break;
        }
        let (old_no, new_no) = state.next(line);
        lines.push(render_diff_card_body_line(
            old_no,
            new_no,
            line,
            diff_line_style(line),
            theme,
            width,
        ));
    }
    lines.push(render_diff_card_spacer_line(theme, width));
    lines
}

fn render_limited_text_lines(
    text: &str,
    text_style: Style,
    theme: Theme,
    width: usize,
) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    let max_lines = max_body_lines();
    for (idx, raw) in text.lines().enumerate() {
        if idx >= max_lines {
            lines.push(render_card_line(
                &[(
                    "… output clipped in TUI".to_string(),
                    root_muted_style(theme).bg(DIFF_CARD_BG),
                )],
                Style::default().bg(DIFF_CARD_BG),
                theme,
                width,
            ));
            break;
        }
        let line = if raw.is_empty() { " " } else { raw };
        lines.push(render_card_line(
            &[(line.to_string(), text_style.bg(DIFF_CARD_BG))],
            Style::default().bg(DIFF_CARD_BG),
            theme,
            width,
        ));
    }
    lines
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

fn diff_line_style(line: &str) -> Style {
    if line.starts_with("diff --git") || line.starts_with("index ") {
        diff_meta_style().add_modifier(Modifier::BOLD)
    } else if line.starts_with("+++") || line.starts_with("---") {
        diff_meta_style()
    } else if line.starts_with('+') {
        Style::default().fg(DIFF_CARD_TEXT).bg(DIFF_CARD_ADD_BG)
    } else if line.starts_with('-') {
        Style::default().fg(DIFF_CARD_TEXT).bg(DIFF_CARD_DELETE_BG)
    } else if line.starts_with("@@") {
        Style::default()
            .fg(ratatui::style::Color::Rgb(161, 198, 255))
            .bg(DIFF_CARD_HUNK_BG)
    } else {
        Style::default().fg(DIFF_CARD_TEXT).bg(DIFF_CARD_BG)
    }
}

fn is_diff_file_header_line(line: &str) -> bool {
    line.starts_with("---") || line.starts_with("+++")
}

fn diff_meta_style() -> Style {
    Style::default().fg(DIFF_CARD_META).bg(DIFF_CARD_BG)
}

fn render_diff_card_header_line(title: &str, theme: Theme, width: usize) -> Line<'static> {
    let text = format!(" {DIFF_CARD_HEADER_ARROW} {title}");
    render_card_line(
        &[(text, diff_header_style())],
        diff_header_fill_style(),
        theme,
        width,
    )
}

fn render_diff_card_spacer_line(theme: Theme, width: usize) -> Line<'static> {
    render_card_line(&[], diff_header_fill_style(), theme, width)
}

fn render_diff_card_body_line(
    old_no: Option<usize>,
    new_no: Option<usize>,
    content: &str,
    content_style: Style,
    theme: Theme,
    width: usize,
) -> Line<'static> {
    let gutter_style = Style::default()
        .fg(DIFF_CARD_GUTTER)
        .bg(DIFF_CARD_GUTTER_BG);
    let number = diff_line_number(new_no.or(old_no));
    let bg = content_style.bg.unwrap_or(DIFF_CARD_BG);
    let (marker, body, marker_style) = diff_marker_and_body(content);
    let pad_style = Style::default().bg(bg);
    let gutter_pad_style = Style::default().bg(DIFF_CARD_GUTTER_BG);
    render_card_line(
        &[
            ("".to_string(), gutter_pad_style),
            (number, gutter_style),
            (" ".to_string(), gutter_pad_style),
            (marker, marker_style),
            (" ".to_string(), pad_style),
            (body, content_style),
        ],
        content_style,
        theme,
        width,
    )
}

fn diff_marker_and_body(content: &str) -> (String, String, Style) {
    match content.chars().next() {
        Some('+') if !content.starts_with("+++") => (
            "+".to_string(),
            content.chars().skip(1).collect(),
            Style::default()
                .fg(DIFF_CARD_ADD_SIGN)
                .bg(DIFF_CARD_GUTTER_BG),
        ),
        Some('-') if !content.starts_with("---") => (
            "-".to_string(),
            content.chars().skip(1).collect(),
            Style::default()
                .fg(DIFF_CARD_DELETE_SIGN)
                .bg(DIFF_CARD_GUTTER_BG),
        ),
        _ => (
            " ".to_string(),
            content.to_string(),
            Style::default()
                .fg(DIFF_CARD_GUTTER)
                .bg(DIFF_CARD_GUTTER_BG),
        ),
    }
}

pub(crate) fn render_card_line(
    segments: &[(String, Style)],
    fill_style: Style,
    theme: Theme,
    width: usize,
) -> Line<'static> {
    render_card_line_with_guide(segments, fill_style, TOOL_CARD_GUIDE, theme, width)
}

fn render_card_line_with_guide(
    segments: &[(String, Style)],
    fill_style: Style,
    guide_color: ratatui::style::Color,
    theme: Theme,
    width: usize,
) -> Line<'static> {
    if width == 0 {
        return Line::from("");
    }

    let guide_width = display_width(TOOL_GUIDE_GLYPH);
    let prefix_width = guide_width.saturating_add(2);
    let guide_style = Style::default().fg(guide_color).bg(theme.root_bg);
    if width <= guide_width {
        return Line::from(Span::styled(TOOL_GUIDE_GLYPH, guide_style));
    }

    let leading_pad_style = if fill_style.bg == Some(theme.root_bg) {
        fill_style
    } else {
        Style::default().bg(DIFF_CARD_BG)
    };

    let mut spans = vec![
        Span::styled(TOOL_GUIDE_GLYPH.to_string(), guide_style),
        Span::styled("  ".to_string(), leading_pad_style),
    ];
    let mut remaining = width.saturating_sub(prefix_width);

    for (text, style) in segments {
        if remaining == 0 {
            break;
        }
        let clipped = truncate_display_width(text, remaining);
        let used = display_width(&clipped);
        if used == 0 {
            continue;
        }
        spans.push(Span::styled(clipped, *style));
        remaining = remaining.saturating_sub(used);
    }

    if remaining > 0 {
        spans.push(Span::styled(" ".repeat(remaining), fill_style));
    }

    Line::from(spans)
}

fn diff_header_style() -> Style {
    Style::default().fg(DIFF_CARD_META).bg(DIFF_CARD_BG)
}

fn diff_header_fill_style() -> Style {
    Style::default().bg(DIFF_CARD_BG)
}

fn shell_card_title_style() -> Style {
    Style::default()
        .fg(ratatui::style::Color::Rgb(160, 170, 210))
        .bg(DIFF_CARD_BG)
}

fn shell_card_command_style() -> Style {
    Style::default().fg(DIFF_CARD_TEXT).bg(DIFF_CARD_BG)
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
        "git__status" => "Git status".into(),
        "git__diff" => "Git diff".into(),
        "git__log" => "Git log".into(),
        "edit__apply_patch" => "Apply patch".into(),
        "workflow__todos" => "Update todos".into(),
        "workflow__auto_continue" => "Update auto-continue".into(),
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
        "git__status" => "Git status".into(),
        "git__diff" => "Git diff".into(),
        "git__log" => "Git log".into(),
        "edit__apply_patch" => "Apply patch".into(),
        "workflow__todos" => "Update todos".into(),
        "workflow__auto_continue" => "Update auto-continue".into(),
        "fs__read" => "Read".into(),
        "fs__list" => "List".into(),
        "fs__write" => "Write".into(),
        "fs__append" => "Append".into(),
        "fs__mkdir" => "Make dir".into(),
        "shell__exec" => "Run command".into(),
        "search__rg" => "Search".into(),
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

fn status_label(status: ToolCardStatus) -> &'static str {
    match status {
        ToolCardStatus::Pending => "pending",
        ToolCardStatus::Approved => "approved",
        ToolCardStatus::Running => "running",
        ToolCardStatus::Succeeded => "succeeded",
        ToolCardStatus::Failed => "failed",
        ToolCardStatus::Denied => "denied",
    }
}

fn tool_trace_arrow_style(status: ToolExecutionStatus, theme: Theme) -> ratatui::style::Style {
    let color = match status {
        ToolExecutionStatus::Pending => theme.warning,
        ToolExecutionStatus::Running => theme.warning,
        ToolExecutionStatus::Succeeded => theme.notice,
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
        ToolExecutionStatus::Succeeded => theme.notice,
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

    for ch in text.chars() {
        let ch_width = unicode_width::UnicodeWidthChar::width(ch).unwrap_or(0);
        if used + ch_width + ellipsis_width > width {
            break;
        }
        out.push(ch);
        used = used.saturating_add(ch_width);
    }

    out.push('…');
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tui::timeline::{PermissionPromptStatus, ToolExecutionStatus};

    #[test]
    fn width_zero_returns_empty_string() {
        assert_eq!(truncate_display_width("abcdef", 0), "");
    }

    #[test]
    fn exact_fit_returns_original_text() {
        assert_eq!(truncate_display_width("abcd", 4), "abcd");
    }

    #[test]
    fn cjk_text_truncates_on_display_cells() {
        assert_eq!(truncate_display_width("你好吗", 5), "你好…");
    }

    #[test]
    fn tool_card_details_hide_verbose_fields_by_default_on_success() {
        let tool = ToolView {
            call_id: "call-1".into(),
            name: "shell__exec".into(),
            summary: "cargo check".into(),
            arguments: Some("--really-long-arg ".repeat(50)),
            output: Some("lots of output\n".repeat(50)),
            status: ToolExecutionStatus::Succeeded,
        };

        let policy = PresentationPolicy;
        let details = tool_card_details(&tool, &policy).expect("not hidden");

        assert_eq!(details.status, ToolCardStatus::Succeeded);
        assert_eq!(details.call_id.as_deref(), Some("call-1"));
        assert!(
            details.arguments.is_none(),
            "args should be hidden on success"
        );
        assert!(
            details.output.is_none(),
            "output should be hidden on success"
        );
    }

    #[test]
    fn quiet_success_command_tools_still_render_compact_cards() {
        let tool = ToolView {
            call_id: "call-q".into(),
            name: "shell__exec".into(),
            summary: "cargo check".into(),
            arguments: Some("cargo check".into()),
            output: Some("\n".into()),
            status: ToolExecutionStatus::Succeeded,
        };

        let theme = Theme::dark();
        let lines = render_tool_card_lines(&tool, theme, 60);
        let rendered = lines
            .iter()
            .map(|l| l.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(rendered.contains("  → Run cargo check"), "{rendered}");
        assert!(!rendered.contains("call-q"), "{rendered}");
        assert!(!rendered.contains("succeeded"), "{rendered}");
    }

    #[test]
    fn agent_explore_running_renders_single_compact_parent_line() {
        let tool = ToolView {
            call_id: "run-1".into(),
            name: "agent__explore".into(),
            summary: "explorer running · child-sessio".into(),
            arguments: Some(serde_json::json!({"task":"inspect src/tui"}).to_string()),
            output: None,
            status: ToolExecutionStatus::Running,
        };

        let rendered = render_tool_card_lines_with_frame(&tool, Theme::dark(), 96, 0)
            .into_iter()
            .map(|line| line.to_string())
            .collect::<Vec<_>>();

        assert_eq!(rendered.len(), 1, "{rendered:?}");
        assert!(
            rendered[0].contains("running explorer inspect src/tui · /child"),
            "{}",
            rendered[0]
        );
    }

    #[test]
    fn agent_explore_success_uses_compact_summary_without_full_body() {
        let tool = ToolView {
            call_id: "run-1".into(),
            name: "agent__explore".into(),
            summary: "explorer completed · child-sessio".into(),
            arguments: None,
            output: Some(
                serde_json::json!({
                    "data": {
                        "agent_name": "Explorer",
                        "status": "completed",
                        "summary": "checked src/tui/timeline.rs and found one follow-up",
                        "full_summary": "checked src/tui/timeline.rs and found one follow-up\nextra hidden detail",
                        "child_session_id": "child-session-1234567890"
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
            rendered[0].contains(
                "completed Explorer checked src/tui/timeline.rs and found one follow-up · /child child-session-1…"
            ),
            "{}",
            rendered[0]
        );
        assert!(
            !rendered[0].contains("extra hidden detail"),
            "{}",
            rendered[0]
        );
        assert!(!rendered[0].contains("full_summary"), "{}", rendered[0]);
    }

    #[test]
    fn agent_fixer_success_uses_same_compact_summary_style() {
        let tool = ToolView {
            call_id: "run-2".into(),
            name: "agent__fixer".into(),
            summary: "fixer completed · child-sessio".into(),
            arguments: None,
            output: Some(
                serde_json::json!({
                    "data": {
                        "agent_name": "fixer",
                        "status": "completed",
                        "summary": "implemented the requested agent wiring",
                        "full_summary": "implemented the requested agent wiring\nhidden child detail",
                        "child_session_id": "child-session-1234567890"
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
            rendered[0].contains(
                "completed fixer implemented the requested agent wiring · /child child-session-1…"
            ),
            "{}",
            rendered[0]
        );
        assert!(
            !rendered[0].contains("hidden child detail"),
            "{}",
            rendered[0]
        );
    }

    #[test]
    fn workflow_control_tools_hide_quiet_success_trace_lines() {
        let tool = ToolView {
            call_id: "call-workflow".into(),
            name: "workflow__todos".into(),
            summary: "workflow__todos completed".into(),
            arguments: Some(serde_json::json!({"items": []}).to_string()),
            output: Some("{}".into()),
            status: ToolExecutionStatus::Succeeded,
        };

        let lines = render_tool_card_lines(&tool, Theme::dark(), 60);

        assert!(lines.is_empty());
    }

    #[test]
    fn workflow_pending_trace_renders_compact_label() {
        let tool = ToolView {
            call_id: "call-workflow-pending".into(),
            name: "workflow__todos".into(),
            summary: "preparing input".into(),
            arguments: None,
            output: None,
            status: ToolExecutionStatus::Pending,
        };

        let rendered = render_tool_card_lines(&tool, Theme::dark(), 80)
            .into_iter()
            .map(|line| line.to_string())
            .collect::<Vec<_>>();

        assert_eq!(rendered.len(), 1, "{rendered:?}");
        assert!(rendered[0].contains("Update todos …"), "{}", rendered[0]);
        assert!(!rendered[0].contains("preparing input"), "{}", rendered[0]);
    }

    #[test]
    fn workflow_running_trace_renders_compact_label() {
        let tool = ToolView {
            call_id: "call-workflow-running".into(),
            name: "workflow__auto_continue".into(),
            summary: "workflow__auto_continue".into(),
            arguments: Some(serde_json::json!({"enabled":true,"max_continuations":2}).to_string()),
            output: None,
            status: ToolExecutionStatus::Running,
        };

        let rendered = render_tool_card_lines(&tool, Theme::dark(), 80)
            .into_iter()
            .map(|line| line.to_string())
            .collect::<Vec<_>>();

        assert_eq!(rendered.len(), 1, "{rendered:?}");
        assert!(
            rendered[0].contains("Update auto-continue …"),
            "{}",
            rendered[0]
        );
    }

    #[test]
    fn read_file_trace_hides_details_but_keeps_useful_summary() {
        let tool = ToolView {
            call_id: "call-read".into(),
            name: "fs__read".into(),
            summary: "fs__read src/tui/runner.rs".into(),
            arguments: Some(
                serde_json::json!({"path":"src/tui/runner.rs","offset":390,"limit":120})
                    .to_string(),
            ),
            output: Some("large raw output that should never be shown".into()),
            status: ToolExecutionStatus::Succeeded,
        };

        let theme = Theme::dark();
        let width = 96usize;
        let lines = render_tool_card_lines(&tool, theme, width);
        assert_eq!(lines.len(), 1);
        let rendered = lines[0].to_string();

        assert_eq!(
            rendered,
            "  → Read src/tui/runner.rs [offset=390, limit=120]"
        );
        assert!(!rendered.contains("call-read"), "{rendered}");
        assert!(!rendered.contains("large raw output"), "{rendered}");
    }

    #[test]
    fn pending_trace_uses_neutral_label_without_fake_arguments() {
        let tool = ToolView {
            call_id: "call-pending".into(),
            name: "shell__exec".into(),
            summary: "preparing input".into(),
            arguments: None,
            output: None,
            status: ToolExecutionStatus::Pending,
        };

        let rendered = render_tool_card_lines(&tool, Theme::dark(), 80)
            .into_iter()
            .map(|line| line.to_string())
            .collect::<Vec<_>>();

        assert_eq!(rendered.len(), 1, "{rendered:?}");
        assert!(rendered[0].contains("Run command …"), "{}", rendered[0]);
        assert!(!rendered[0].contains("preparing input"), "{}", rendered[0]);
    }

    #[test]
    fn permission_card_lines_use_composer_style_status_guide() {
        let permission = PermissionView {
            call_id: "perm-fill".into(),
            tool_name: "shell__exec".into(),
            summary: "echo ok".into(),
            arguments: Some("echo ok".into()),
            rationale: None,
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
            rendered.contains("pending shell__exec echo ok"),
            "{rendered}"
        );
    }

    #[test]
    fn quiet_success_read_like_tools_can_be_hidden() {
        let tool = ToolView {
            call_id: "call-r".into(),
            name: "fs__read".into(),
            summary: "read".into(),
            arguments: Some("src/main.rs".into()),
            output: Some("\n".into()),
            status: ToolExecutionStatus::Succeeded,
        };
        let theme = Theme::dark();
        let lines = render_tool_card_lines(&tool, theme, 80);
        assert!(lines.is_empty(), "expected hidden read-like quiet success");
    }

    #[test]
    fn tool_card_details_surface_failed_state_with_output_snippet() {
        let tool = ToolView {
            call_id: "call-2".into(),
            name: "shell__exec".into(),
            summary: "run cargo test".into(),
            arguments: Some("cargo test".into()),
            output: Some("error: failed to compile\nmore...".into()),
            status: ToolExecutionStatus::Failed,
        };

        let policy = PresentationPolicy;
        let details = tool_card_details(&tool, &policy).expect("not hidden");
        assert_eq!(details.status, ToolCardStatus::Failed);
        assert_eq!(details.call_id.as_deref(), Some("call-2"));
        assert!(
            details
                .output
                .as_deref()
                .is_some_and(|s| s.contains("error:"))
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

    #[test]
    fn shell_output_renders_stdout_and_stderr_sections() {
        let theme = Theme::dark();
        let tool = ToolView {
            call_id: "call-shell".into(),
            name: "shell__exec".into(),
            summary: "exit 1 · stdout 1 lines · stderr 1 lines".into(),
            arguments: Some(serde_json::json!({"command":"cargo test"}).to_string()),
            output: Some(
                serde_json::json!({
                    "ok": false,
                    "tool": "shell__exec",
                    "data": {
                        "status": 1,
                        "success": false,
                        "stdout": "running tests\n",
                        "stdout_truncated": false,
                        "stderr": "error: failed\n",
                        "stderr_truncated": false
                    }
                })
                .to_string(),
            ),
            status: ToolExecutionStatus::Failed,
        };

        let rendered = render_tool_card_lines(&tool, theme, 80)
            .iter()
            .map(|line| line.to_string())
            .collect::<Vec<_>>()
            .join("\n");

        assert!(rendered.contains("# Runs test suite"), "{rendered}");
        assert!(rendered.contains("$ cargo test"), "{rendered}");
        assert!(!rendered.contains("→ Run cargo test"), "{rendered}");
        assert!(rendered.contains("stdout"), "{rendered}");
        assert!(rendered.contains("running tests"), "{rendered}");
        assert!(rendered.contains("stderr"), "{rendered}");
        assert!(rendered.contains("error: failed"), "{rendered}");

        let lines = render_tool_card_lines(&tool, theme, 80);
        assert!(lines.len() > 1, "{rendered}");
        for line in lines.iter().skip(1) {
            let guide = line.spans.first().expect("body line has guide");
            assert_eq!(guide.content.as_ref(), TOOL_GUIDE_GLYPH);
            assert_eq!(guide.style.fg, Some(TOOL_CARD_GUIDE));
            assert_eq!(guide.style.bg, Some(theme.root_bg));
            assert_eq!(display_width(&line.to_string()), 80, "{line:?}");
        }
    }

    #[test]
    fn apply_patch_renders_inline_diff_from_arguments() {
        let theme = Theme::dark();
        let tool = ToolView {
            call_id: "call-edit".into(),
            name: "edit__apply_patch".into(),
            summary: "patched 1 files · 1 edits".into(),
            arguments: Some(
                serde_json::json!({
                    "edits": [{
                        "path": "src/main.rs",
                        "find": "old line",
                        "replace": "new line",
                        "replace_all": false
                    }]
                })
                .to_string(),
            ),
            output: Some(
                serde_json::json!({
                    "ok": true,
                    "tool": "edit__apply_patch",
                    "data": {"files_changed": 1, "edits_applied": 1}
                })
                .to_string(),
            ),
            status: ToolExecutionStatus::Succeeded,
        };

        let rendered = render_tool_card_lines(&tool, theme, 80)
            .iter()
            .map(|line| line.to_string())
            .collect::<Vec<_>>()
            .join("\n");

        assert!(rendered.contains("← Patch src/main.rs"), "{rendered}");
        assert!(!rendered.contains("--- src/main.rs"), "{rendered}");
        assert!(!rendered.contains("+++ src/main.rs"), "{rendered}");
        assert!(rendered.contains("- old line"), "{rendered}");
        assert!(rendered.contains("+ new line"), "{rendered}");
    }

    #[test]
    fn write_file_renders_written_content_as_diff() {
        let theme = Theme::dark();
        let tool = ToolView {
            call_id: "call-write".into(),
            name: "fs__write".into(),
            summary: "wrote 11 bytes to tool-write-test.txt".into(),
            arguments: Some(
                serde_json::json!({
                    "path": "tool-write-test.txt",
                    "content": "test write\n"
                })
                .to_string(),
            ),
            output: Some(
                serde_json::json!({
                    "ok": true,
                    "tool": "fs__write",
                    "data": {"path": "tool-write-test.txt", "bytes_written": 11}
                })
                .to_string(),
            ),
            status: ToolExecutionStatus::Succeeded,
        };

        let rendered = render_tool_card_lines(&tool, theme, 80)
            .iter()
            .map(|line| line.to_string())
            .collect::<Vec<_>>()
            .join("\n");

        assert!(
            rendered.contains("← Write tool-write-test.txt"),
            "{rendered}"
        );
        assert!(!rendered.contains("--- tool-write-test.txt"), "{rendered}");
        assert!(!rendered.contains("+++ tool-write-test.txt"), "{rendered}");
        assert!(rendered.contains("+ test write"), "{rendered}");
    }

    #[test]
    fn append_file_renders_appended_content_as_diff() {
        let theme = Theme::dark();
        let tool = ToolView {
            call_id: "call-append".into(),
            name: "fs__append".into(),
            summary: "appended 120 bytes to tool-write-test.txt".into(),
            arguments: Some(
                serde_json::json!({
                    "path": "tool-write-test.txt",
                    "content": "追加一行\n再追加一行\n"
                })
                .to_string(),
            ),
            output: Some(
                serde_json::json!({
                    "ok": true,
                    "tool": "fs__append",
                    "data": {"path": "tool-write-test.txt", "bytes_appended": 120}
                })
                .to_string(),
            ),
            status: ToolExecutionStatus::Succeeded,
        };

        let rendered = render_tool_card_lines(&tool, theme, 80)
            .iter()
            .map(|line| line.to_string())
            .collect::<Vec<_>>()
            .join("\n");

        assert!(
            rendered.contains("← Append tool-write-test.txt"),
            "{rendered}"
        );
        assert!(!rendered.contains("--- tool-write-test.txt"), "{rendered}");
        assert!(!rendered.contains("+++ tool-write-test.txt"), "{rendered}");
        assert!(rendered.contains("+ 追加一行"), "{rendered}");
        assert!(rendered.contains("+ 再追加一行"), "{rendered}");
    }

    #[test]
    fn git_diff_success_renders_compact_trace_without_diff_card() {
        let theme = Theme::dark();
        let tool = ToolView {
            call_id: "call-git-diff".into(),
            name: "git__diff".into(),
            summary: "git diff src/lib.rs".into(),
            arguments: Some(serde_json::json!({"path":"src/lib.rs"}).to_string()),
            output: Some(
                serde_json::json!({
                    "ok": true,
                    "tool": "git__diff",
                    "data": {
                        "stdout": "diff --git a/src/lib.rs b/src/lib.rs\nindex 123..456 100644\n--- a/src/lib.rs\n+++ b/src/lib.rs\n@@ -1 +1 @@\n-old\n+new\n",
                        "stdout_truncated": false
                    }
                })
                .to_string(),
            ),
            status: ToolExecutionStatus::Succeeded,
        };

        let lines = render_tool_card_lines(&tool, theme, 84);
        let rendered = lines
            .iter()
            .map(|line| line.to_string())
            .collect::<Vec<_>>()
            .join("\n");

        assert_eq!(lines.len(), 1, "{rendered}");
        assert!(rendered.contains("→ Git diff"), "{rendered}");
        assert!(!rendered.contains("← Diff src/lib.rs"), "{rendered}");
        assert!(!rendered.contains("--- a/src/lib.rs"), "{rendered}");
        assert!(!rendered.contains("+++ b/src/lib.rs"), "{rendered}");
        assert!(!rendered.contains("+ new"), "{rendered}");
        assert!(display_width(&rendered) <= 84, "{rendered}");
    }

    #[test]
    fn permission_denied_shows_resolution_as_error_snippet() {
        let permission = PermissionView {
            call_id: "perm-1".into(),
            tool_name: "shell__exec".into(),
            summary: "dangerous".into(),
            arguments: None,
            rationale: None,
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
            status: PermissionPromptStatus::Pending,
            resolution_reason: None,
        };
        let approved = PermissionView {
            call_id: "perm-a".into(),
            tool_name: "shell__exec".into(),
            summary: "touch file".into(),
            arguments: Some("touch a.txt".into()),
            rationale: Some("needed".into()),
            status: PermissionPromptStatus::Approved,
            resolution_reason: None,
        };
        let denied = PermissionView {
            call_id: "perm-d".into(),
            tool_name: "shell__exec".into(),
            summary: "format disk".into(),
            arguments: Some("mkfs".into()),
            rationale: Some("unsafe".into()),
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

        assert!(p.contains("pending"), "{p}");
        assert!(a.contains("approved"), "{a}");
        assert!(d.contains("denied"), "{d}");

        assert!(p.contains("shell__exec"), "{p}");
        assert!(p.contains("rm -rf /"), "{p}");
        assert!(p.contains("requested by user"), "{p}");
        assert!(d.contains("policy"), "{d}");
        assert!(!p.contains("call"), "{p}");
        assert!(!p.contains("perm-p"), "{p}");
        assert!(!p.contains("args"), "{p}");
        assert!(!p.contains("why"), "{p}");
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
            status: PermissionPromptStatus::Pending,
            resolution_reason: None,
        };
        let approved = PermissionView {
            call_id: "perm-aw".into(),
            tool_name: "shell__exec".into(),
            summary: "approved".into(),
            arguments: Some("arg ".repeat(60)),
            rationale: Some("why ".repeat(80)),
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
}
