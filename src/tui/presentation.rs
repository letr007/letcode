use super::components::transcript::TranscriptRenderCache;
use super::state::{SelectionAnchor, TranscriptClickTarget, TuiState};
use super::timeline::TimelineItem;
use super::transcript_render::Interaction;
use crate::agent::is_subagent_tool_name;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TuiPresentationState {
    pub transcript_render_cache: TranscriptRenderCache,
    pub frame_hyperlink_cells: Vec<super::transcript_ratatui::HyperlinkCell>,
    pub last_transcript_total_rows: Option<usize>,
    pub last_transcript_area: ratatui::layout::Rect,
    pub last_transcript_scroll_top: u16,
}

impl TuiPresentationState {
    /// 将终端坐标映射到 transcript 的点击目标。
    pub fn transcript_click_target(
        &self,
        state: &TuiState,
        terminal_col: u16,
        terminal_row: u16,
    ) -> Option<TranscriptClickTarget> {
        let area = self.last_transcript_area;
        if terminal_col < area.left()
            || terminal_col >= area.right()
            || terminal_row < area.top()
            || terminal_row >= area.bottom()
            || area.width == 0
            || area.height == 0
        {
            return None;
        }

        let absolute_row =
            (terminal_row - area.y) as usize + self.last_transcript_scroll_top as usize;
        let item_index = match self
            .transcript_render_cache
            .row_starts()
            .binary_search(&absolute_row)
        {
            Ok(index) => index,
            Err(index) => index.saturating_sub(1),
        };
        let entry = self.transcript_render_cache.entries().get(item_index)?;
        let rendered_line_offset = absolute_row
            .saturating_sub(*self.transcript_render_cache.row_starts().get(item_index)?);
        let line = entry.document.lines.get(rendered_line_offset)?;
        let local_col = terminal_col - area.x;
        let mut visual_col = 0u16;
        for span in &line.spans {
            let span_width = crate::tui::measure::display_width(&span.text) as u16;
            if local_col >= visual_col && local_col < visual_col.saturating_add(span_width) {
                if let Some(Interaction::OpenUrl(url)) = &span.interaction {
                    return crate::tui::transcript_ratatui::safe_hyperlink_url(url)
                        .then(|| TranscriptClickTarget::OpenUrl(url.clone()));
                }
                break;
            }
            visual_col = visual_col.saturating_add(span_width);
        }

        match state.active_timeline().items().get(item_index) {
            Some(TimelineItem::Tool(tool)) => {
                Some(TranscriptClickTarget::ToolCard(tool.call_id.clone()))
            }
            Some(TimelineItem::AutoReview(decision)) => Some(TranscriptClickTarget::ToolCard(
                format!("auto-review:{}", decision.call_id),
            )),
            _ => None,
        }
    }

    pub fn map_mouse_to_anchor(
        &self,
        terminal_col: u16,
        terminal_row: u16,
    ) -> Option<SelectionAnchor> {
        let area = self.last_transcript_area;
        if terminal_col < area.left()
            || terminal_col >= area.right()
            || terminal_row < area.top()
            || terminal_row >= area.bottom()
            || area.width == 0
            || area.height == 0
        {
            return None;
        }

        let viewport_row = terminal_row - area.y;
        let absolute_row = viewport_row as usize + self.last_transcript_scroll_top as usize;
        let cache = &self.transcript_render_cache;
        if cache.row_starts().is_empty() {
            return None;
        }

        let item_index = match cache.row_starts().binary_search(&absolute_row) {
            Ok(idx) => idx,
            Err(idx) => idx.saturating_sub(1),
        };
        if item_index >= cache.entries().len() {
            return None;
        }

        let item_start_row = cache.row_starts()[item_index];
        let entry = &cache.entries()[item_index];
        let rendered_line_offset = absolute_row.saturating_sub(item_start_row);
        let line = entry.document.lines.get(rendered_line_offset)?;

        let local_col = terminal_col - area.x;
        let mut visual_col = 0u16;
        let mut visual_char = 0usize;
        for span in &line.spans {
            let span_width = crate::tui::measure::display_width(&span.text) as u16;
            let span_chars = span.text.chars().count();
            if local_col >= visual_col && local_col < visual_col.saturating_add(span_width) {
                let range = span.source?;
                let within =
                    column_to_char_offset(&span.text, local_col - visual_col).min(span_chars);
                let _source = entry.document.source_blocks.get(range.block_index)?;
                return Some(SelectionAnchor {
                    item_index,
                    rendered_line_offset,
                    char_offset: visual_char + within,
                });
            }
            visual_col = visual_col.saturating_add(span_width);
            visual_char += span_chars;
        }
        None
    }
}

/// Convert a display-cell coordinate into the leading Unicode-scalar offset of
/// the hit extended grapheme cluster. Selection extraction expands endpoint
/// boundaries, so both forward and reverse gestures include the hit graphemes.
fn column_to_char_offset(text: &str, target_col: u16) -> usize {
    use unicode_segmentation::UnicodeSegmentation;

    let mut width = 0usize;
    let mut offset = 0usize;
    for grapheme in text.graphemes(true) {
        let grapheme_width = crate::tui::measure::display_width(grapheme);
        if width + grapheme_width > target_col as usize {
            return offset;
        }
        width += grapheme_width;
        offset += grapheme.chars().count();
    }
    offset
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolPresentation {
    Hidden,
    Inline,
    CompactCard,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolPresentationStatus {
    Pending,
    Running,
    Succeeded,
    Failed,
}

/// Render-facing context for TUI timeline tool items.
///
/// The TUI currently stores tool arguments/output as already-formatted text in the
/// timeline view models (not structured JSON). This context lets PresentationPolicy
/// own the single source of truth for whether a tool should be shown and how much
/// detail is reasonable by default.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolTextPresentationContext {
    pub name: String,
    pub status: ToolPresentationStatus,
    pub arguments: Option<String>,
    pub output: Option<String>,
}

impl ToolTextPresentationContext {
    pub fn new(name: impl Into<String>, status: ToolPresentationStatus) -> Self {
        Self {
            name: name.into(),
            status,
            arguments: None,
            output: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PresentationPolicy;

impl PresentationPolicy {
    pub fn tool_presentation_text(
        &self,
        context: &ToolTextPresentationContext,
    ) -> ToolPresentation {
        tool_presentation_impl(
            &context.name,
            context.status,
            is_quiet_success_text(context.output.as_deref()),
        )
    }
}

fn tool_presentation_impl(
    tool_name: &str,
    status: ToolPresentationStatus,
    is_quiet_success: bool,
) -> ToolPresentation {
    use crate::permission::ToolPermissionClass;

    // A completed user question is part of the conversation's durable decision trail.
    // It must remain visible even when a generic low-risk tool result would be quiet.
    if tool_name == crate::tool_names::TOOL_QUESTION {
        return ToolPresentation::CompactCard;
    }

    if is_workflow_control_tool(tool_name) {
        return ToolPresentation::CompactCard;
    }

    let class = crate::permission::classify_tool(tool_name);

    if is_subagent_tool_name(tool_name) {
        return ToolPresentation::CompactCard;
    }

    match status {
        ToolPresentationStatus::Pending => ToolPresentation::CompactCard,
        ToolPresentationStatus::Running => match class {
            ToolPermissionClass::Read | ToolPermissionClass::Preview => ToolPresentation::Inline,
            ToolPermissionClass::Write
            | ToolPermissionClass::Command
            | ToolPermissionClass::Unknown => ToolPresentation::CompactCard,
        },
        ToolPresentationStatus::Succeeded => {
            // Safety/audit trail: never hide write/command/unknown tool executions.
            // Quiet success hiding is only allowed for low-risk read/preview tools.
            if is_quiet_success {
                match class {
                    ToolPermissionClass::Read | ToolPermissionClass::Preview => {
                        ToolPresentation::Hidden
                    }
                    ToolPermissionClass::Write
                    | ToolPermissionClass::Command
                    | ToolPermissionClass::Unknown => ToolPresentation::CompactCard,
                }
            } else {
                match class {
                    ToolPermissionClass::Read | ToolPermissionClass::Preview => {
                        ToolPresentation::Inline
                    }
                    ToolPermissionClass::Write
                    | ToolPermissionClass::Command
                    | ToolPermissionClass::Unknown => ToolPresentation::CompactCard,
                }
            }
        }
        ToolPresentationStatus::Failed => ToolPresentation::CompactCard,
    }
}

fn is_workflow_control_tool(tool_name: &str) -> bool {
    matches!(tool_name, "workflow__todos" | "workflow__auto_continue")
}

fn is_quiet_success_text(output: Option<&str>) -> bool {
    let Some(output) = output else {
        return true;
    };
    output.trim().is_empty()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn completed_question_is_never_hidden_as_a_quiet_success() {
        let policy = PresentationPolicy;
        let context =
            ToolTextPresentationContext::new("question", ToolPresentationStatus::Succeeded);

        assert_eq!(
            policy.tool_presentation_text(&context),
            ToolPresentation::CompactCard
        );
    }

    #[test]
    fn quiet_success_text_write_like_tools_are_never_hidden() {
        let policy = PresentationPolicy;
        let mut ctx =
            ToolTextPresentationContext::new("shell__exec", ToolPresentationStatus::Succeeded);
        ctx.output = Some("\n".into());
        assert_eq!(
            policy.tool_presentation_text(&ctx),
            ToolPresentation::CompactCard
        );
    }
}
