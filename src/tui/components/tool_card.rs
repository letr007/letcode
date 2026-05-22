use ratatui::{
    Frame,
    layout::Rect,
    style::{Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Clear, Paragraph, Wrap},
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

const PERMISSION_HINT: &str = "[a] approve  [d] deny";

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

    // Preserve compact safety context: args + rationale when present.
    details.arguments = permission
        .arguments
        .as_deref()
        .map(one_line_snippet)
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
    let policy = PresentationPolicy;
    let Some(details) = tool_card_details(tool, &policy) else {
        return Vec::new();
    };
    render_details_lines(
        &details,
        tool_accent(tool.status, theme),
        theme.element_bg,
        theme,
        width,
    )
}

pub fn render_permission_card_lines(
    permission: &PermissionView,
    theme: Theme,
    width: usize,
) -> Vec<Line<'static>> {
    let details = permission_card_details(permission);
    let accent = permission_accent(permission.status, theme);
    render_details_lines(&details, accent, theme.elevated_bg, theme, width)
}

/// Render the *pending* permission prompt as a focused elevated panel.
///
/// This is separate from the transcript permission timeline item and is shown while waiting for
/// approval/denial.
pub fn render_pending_permission_prompt(
    frame: &mut Frame<'_>,
    permission: &PermissionView,
    area: Rect,
    theme: Theme,
) {
    if area.is_empty() {
        return;
    }

    if area.height < 3 || area.width < 24 {
        let line = Line::from(vec![
            Span::styled("PERMISSION ", theme.approval_style()),
            Span::styled(permission.tool_name.clone(), inline_elevated(theme)),
            Span::styled(": ", inline_elevated(theme)),
            Span::styled(permission.summary.clone(), inline_elevated(theme)),
            Span::styled("  ", inline_elevated(theme)),
            Span::styled(PERMISSION_HINT, theme.approval_style()),
        ]);
        frame.render_widget(
            Paragraph::new(line)
                .style(theme.elevated_style())
                .wrap(Wrap { trim: true }),
            area,
        );
        return;
    }

    frame.render_widget(Clear, area);
    frame.render_widget(Block::new().style(theme.elevated_style()), area);
    render_accent_bar(
        frame,
        area,
        surface::accent_style(
            theme,
            surface::SurfaceEmphasis::Approval,
            surface::SurfaceKind::Elevated,
        )
        .add_modifier(Modifier::BOLD),
    );

    let mut lines = vec![
        Line::from(vec![
            Span::styled("Permission required", elevated_title_style(theme)),
            Span::styled("  ", inline_elevated(theme)),
            Span::styled(permission.call_id.clone(), elevated_muted(theme)),
        ]),
        Line::from(vec![
            Span::styled("tool ", elevated_muted(theme)),
            Span::styled(permission.tool_name.clone(), theme.approval_style()),
            Span::styled(" — ", inline_elevated(theme)),
            Span::styled(permission.summary.clone(), inline_elevated(theme)),
        ]),
        Line::from(Span::styled(PERMISSION_HINT, theme.approval_style())),
    ];

    if let Some(args) = permission.arguments.as_deref().filter(|s| !s.is_empty()) {
        lines.push(kv_line(
            "args",
            args,
            theme.approval,
            theme.elevated_bg,
            elevated_muted(theme),
            inline_elevated(theme),
            area.width.max(1) as usize,
        ));
    }
    if let Some(why) = permission.rationale.as_deref().filter(|s| !s.is_empty()) {
        lines.push(kv_line(
            "why",
            why,
            theme.approval,
            theme.elevated_bg,
            elevated_muted(theme),
            inline_elevated(theme),
            area.width.max(1) as usize,
        ));
    }

    // Leave breathing room at the left for the accent bar.
    let content_area = inset_left(area, 3);
    let paragraph = Paragraph::new(Text::from(lines))
        .style(theme.elevated_style())
        .wrap(Wrap { trim: false });
    frame.render_widget(paragraph, content_area);
}

fn render_details_lines(
    details: &ToolCardDetails,
    accent: ratatui::style::Color,
    bg: ratatui::style::Color,
    theme: Theme,
    width: usize,
) -> Vec<Line<'static>> {
    let mut lines = Vec::new();

    let title_style = if bg == theme.elevated_bg {
        elevated_title_style(theme)
    } else {
        element_title_style(theme)
    };
    let label_style = if bg == theme.elevated_bg {
        elevated_muted(theme)
    } else {
        element_muted_style(theme)
    };
    let value_style = if bg == theme.elevated_bg {
        inline_elevated(theme)
    } else {
        theme.element_style()
    };

    // 1) Header line: tool name + compact status label
    push_wrapped_card_line(
        &mut lines,
        &format!("# {}", details.title),
        accent,
        bg,
        title_style,
        width,
    );
    push_wrapped_card_line(
        &mut lines,
        &format!("{} · {}", status_label(details.status), details.summary),
        accent,
        bg,
        value_style,
        width,
    );

    // 2) Optional tiny detail rows.
    // Intentionally keep these extremely short; no raw multi-line payload dumping.
    if let Some(call_id) = details.call_id.as_deref() {
        push_card_single_line_kv(
            &mut lines,
            "call",
            call_id,
            accent,
            bg,
            label_style,
            value_style,
            width,
        );
    }

    if let Some(args) = details.arguments.as_deref() {
        push_card_single_line_kv(
            &mut lines,
            "args",
            args,
            accent,
            bg,
            label_style,
            value_style,
            width,
        );
    }

    if let Some(out) = details.output.as_deref() {
        let label = match details.status {
            ToolCardStatus::Failed | ToolCardStatus::Denied => "err",
            _ => "out",
        };
        push_card_single_line_kv(
            &mut lines,
            label,
            out,
            accent,
            bg,
            label_style,
            value_style,
            width,
        );
    }

    for (label, value) in &details.fields {
        push_card_single_line_kv(
            &mut lines,
            label,
            value,
            accent,
            bg,
            label_style,
            value_style,
            width,
        );
    }

    lines
}

fn push_wrapped_card_line(
    lines: &mut Vec<Line<'static>>,
    content: &str,
    accent: ratatui::style::Color,
    bg: ratatui::style::Color,
    value_style: ratatui::style::Style,
    width: usize,
) {
    let content_width = card_content_width(width).max(1);
    for wrapped in wrap_text_to_width(content, content_width) {
        let mut line = Line::from(vec![
            Span::styled(surface::ACCENT_BAR_GLYPH, card_bar_style(accent, bg)),
            Span::styled("  ", value_style),
            Span::styled(wrapped, value_style),
        ]);
        pad_card_line_to_width(&mut line, width, value_style);
        lines.push(line);
    }
}

fn push_card_single_line_kv(
    lines: &mut Vec<Line<'static>>,
    label: &str,
    value: &str,
    accent: ratatui::style::Color,
    bg: ratatui::style::Color,
    label_style: ratatui::style::Style,
    value_style: ratatui::style::Style,
    width: usize,
) {
    lines.push(kv_line(
        label,
        value,
        accent,
        bg,
        label_style,
        value_style,
        width,
    ));
}

fn kv_line(
    label: &str,
    value: &str,
    accent: ratatui::style::Color,
    bg: ratatui::style::Color,
    label_style: ratatui::style::Style,
    value_style: ratatui::style::Style,
    width: usize,
) -> Line<'static> {
    // Width-safe single-row key/value line.
    // Layout intent: "┃  {label} {value}" with aggressive truncation on tiny widths.
    if width == 0 {
        return Line::from("");
    }

    let bar = surface::ACCENT_BAR_GLYPH;
    let bar_w = display_width(bar);
    if width <= bar_w {
        return Line::from(Span::styled(bar, card_bar_style(accent, bg)));
    }

    let mut remaining = width.saturating_sub(bar_w);

    // Prefer 2 spaces of padding, but degrade when narrow.
    let pad = "  ";
    let pad_w = display_width(pad);
    let pad_take = remaining.min(pad_w);
    let pad_str = " ".repeat(pad_take);
    remaining = remaining.saturating_sub(pad_take);

    // Keep labels compact; they must never extend beyond width.
    // Default max label budget (display cells) for aesthetics.
    let max_label_cells = 9usize;
    let label_budget = remaining.min(max_label_cells);
    let label_str = if label_budget == 0 {
        String::new()
    } else {
        truncate_display_width(label, label_budget)
    };
    let label_w = display_width(&label_str);
    remaining = remaining.saturating_sub(label_w);

    // Add a separating space if we have both label and remaining capacity.
    let sep = if !label_str.is_empty() && remaining > 0 {
        " "
    } else {
        ""
    };
    let sep_w = display_width(sep);
    remaining = remaining.saturating_sub(sep_w);

    let mut clipped = one_line_snippet(value);
    clipped = if remaining == 0 {
        String::new()
    } else {
        truncate_display_width(&clipped, remaining)
    };

    let mut line = Line::from(vec![
        Span::styled(bar, card_bar_style(accent, bg)),
        Span::styled(pad_str, label_style),
        Span::styled(label_str, label_style),
        Span::styled(sep.to_string(), label_style),
        Span::styled(clipped, value_style),
    ]);
    pad_card_line_to_width(&mut line, width, value_style);
    line
}

fn card_content_width(width: usize) -> usize {
    // Card line shape is `┃  content ` at normal widths. Reserve one cell for
    // the accent, two for left padding, and one for right fill so the card reads
    // as a full-width block instead of text highlighted on the root background.
    width.saturating_sub(4)
}

fn pad_card_line_to_width(
    line: &mut Line<'static>,
    width: usize,
    fill_style: ratatui::style::Style,
) {
    let used = display_width(&line.to_string());
    if width > used {
        line.spans
            .push(Span::styled(" ".repeat(width - used), fill_style));
    }
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
        ToolExecutionStatus::Running => ToolCardStatus::Running,
        ToolExecutionStatus::Succeeded => ToolCardStatus::Succeeded,
        ToolExecutionStatus::Failed => ToolCardStatus::Failed,
    }
}

fn tool_accent(status: ToolExecutionStatus, theme: Theme) -> ratatui::style::Color {
    match status {
        ToolExecutionStatus::Running => theme.warning,
        ToolExecutionStatus::Succeeded => theme.success,
        ToolExecutionStatus::Failed => theme.error,
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

fn card_bar_style(
    accent: ratatui::style::Color,
    bg: ratatui::style::Color,
) -> ratatui::style::Style {
    ratatui::style::Style::default().fg(accent).bg(bg)
}

fn render_accent_bar(frame: &mut Frame<'_>, area: Rect, style: Style) {
    if area.is_empty() {
        return;
    }

    let bar_area = Rect::new(area.x, area.y, 1.min(area.width), area.height);
    let lines =
        vec![Line::from(Span::styled(surface::ACCENT_BAR_GLYPH, style)); area.height as usize];
    frame.render_widget(Paragraph::new(Text::from(lines)).style(style), bar_area);
}

fn inset_left(area: Rect, amount: u16) -> Rect {
    let inset = amount.min(area.width);
    Rect::new(
        area.x + inset,
        area.y,
        area.width.saturating_sub(inset),
        area.height,
    )
}

fn element_title_style(theme: Theme) -> ratatui::style::Style {
    ratatui::style::Style::default()
        .fg(theme.text)
        .bg(theme.element_bg)
        .add_modifier(ratatui::style::Modifier::BOLD)
}

fn element_muted_style(theme: Theme) -> ratatui::style::Style {
    ratatui::style::Style::default()
        .fg(theme.muted_text)
        .bg(theme.element_bg)
}

fn inline_elevated(theme: Theme) -> ratatui::style::Style {
    ratatui::style::Style::default()
        .fg(theme.text)
        .bg(theme.elevated_bg)
}

fn elevated_title_style(theme: Theme) -> ratatui::style::Style {
    ratatui::style::Style::default()
        .fg(theme.text)
        .bg(theme.elevated_bg)
        .add_modifier(ratatui::style::Modifier::BOLD)
}

fn elevated_muted(theme: Theme) -> ratatui::style::Style {
    ratatui::style::Style::default()
        .fg(theme.muted_text)
        .bg(theme.elevated_bg)
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
            name: "bash".into(),
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
            name: "run_command".into(),
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
        assert!(rendered.contains("# run_command"), "{rendered}");
        assert!(rendered.contains("succeeded"), "{rendered}");
        assert!(rendered.contains("call"), "{rendered}");
        assert!(rendered.contains("call-q"), "{rendered}");
    }

    #[test]
    fn tool_card_lines_fill_available_width_with_card_background() {
        let tool = ToolView {
            call_id: "call-fill".into(),
            name: "run_command".into(),
            summary: "echo ok".into(),
            arguments: Some("echo ok".into()),
            output: Some("\n".into()),
            status: ToolExecutionStatus::Succeeded,
        };

        let theme = Theme::dark();
        let width = 72usize;
        let lines = render_tool_card_lines(&tool, theme, width);
        assert!(!lines.is_empty());

        for line in &lines {
            assert_eq!(display_width(&line.to_string()), width, "{line:?}");
            let fill = line.spans.last().expect("line has fill span");
            assert!(fill.content.chars().all(|ch| ch == ' '), "{line:?}");
            assert_eq!(fill.style.bg, Some(theme.element_bg));
        }
    }

    #[test]
    fn quiet_success_read_like_tools_can_be_hidden() {
        let tool = ToolView {
            call_id: "call-r".into(),
            name: "read_file".into(),
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
            name: "bash".into(),
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
            name: "bash".into(),
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
    fn permission_denied_shows_resolution_as_error_snippet() {
        let permission = PermissionView {
            call_id: "perm-1".into(),
            tool_name: "bash".into(),
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
            tool_name: "bash".into(),
            summary: "rm -rf /".into(),
            arguments: Some("rm -rf /".into()),
            rationale: Some("requested by user".into()),
            status: PermissionPromptStatus::Pending,
            resolution_reason: None,
        };
        let approved = PermissionView {
            call_id: "perm-a".into(),
            tool_name: "bash".into(),
            summary: "touch file".into(),
            arguments: Some("touch a.txt".into()),
            rationale: Some("needed".into()),
            status: PermissionPromptStatus::Approved,
            resolution_reason: None,
        };
        let denied = PermissionView {
            call_id: "perm-d".into(),
            tool_name: "bash".into(),
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

        assert!(p.contains("call"), "{p}");
        assert!(p.contains("perm-p"), "{p}");
        assert!(p.contains("args"), "{p}");
        assert!(p.contains("why"), "{p}");
        // Label may be truncated on narrow widths.
        assert!(d.contains("resol") || d.contains("resolution"), "{d}");
    }

    #[test]
    fn permission_card_lines_are_width_safe_at_narrow_and_normal_widths() {
        let theme = Theme::dark();

        let denied = PermissionView {
            call_id: "perm-w".into(),
            tool_name: "bash".into(),
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
            tool_name: "bash".into(),
            summary: "pending".into(),
            arguments: Some("arg ".repeat(60)),
            rationale: Some("why ".repeat(80)),
            status: PermissionPromptStatus::Pending,
            resolution_reason: None,
        };
        let approved = PermissionView {
            call_id: "perm-aw".into(),
            tool_name: "bash".into(),
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
