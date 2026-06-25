use ratatui::layout::{Constraint, Direction, Layout, Rect};

use crate::tui::measure::wrapped_row_count;
use crate::tui::state::TuiState;
use crate::user_content::UserImageAttachment;

use super::super::surface;
use super::slash_panel;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct WorkspaceLayoutMetrics {
    pub transcript_viewport_height: u16,
    pub slash_panel_height: u16,
    pub composer_height: u16,
}

pub fn workspace_area(area: Rect) -> Rect {
    if area.width <= surface::OUTER_PAD_X * 2 + 4 {
        return area;
    }

    Rect::new(
        area.x + surface::OUTER_PAD_X,
        area.y,
        area.width.saturating_sub(surface::OUTER_PAD_X * 2),
        area.height.saturating_sub(surface::SESSION_PAD_BOTTOM),
    )
}

pub fn composer_height(
    total_height: u16,
    input: &str,
    attachments: &[UserImageAttachment],
    width: usize,
) -> u16 {
    match total_height {
        0..=2 => 0,
        3..=5 => 1,
        6..=8 => 3,
        _ => {
            let text_width = width.max(1).saturating_sub(
                surface::ACCENT_BAR_WIDTH as usize + surface::PROMPT_INNER_PAD_X as usize * 2,
            );
            let rows = wrapped_row_count(input, text_width.max(1)) as u16;
            let clamped = rows.clamp(surface::TEXTAREA_MIN_ROWS, surface::TEXTAREA_MAX_ROWS);
            let attachment_rows = u16::try_from(attachments.len()).unwrap_or(u16::MAX);
            (surface::PROMPT_INNER_PAD_TOP
                + attachment_rows
                + clamped
                + surface::PROMPT_METADATA_PAD_TOP
                + 1
                + surface::PROMPT_INNER_PAD_BOTTOM)
                .min(total_height.saturating_sub(2))
        }
    }
}

pub fn child_read_only_composer_height(total_height: u16) -> u16 {
    match total_height {
        0..=2 => 0,
        3..=6 => 1,
        _ => 4,
    }
}

pub fn slash_panel_height(state: &TuiState) -> u16 {
    slash_panel::slash_panel_reserved_height(state)
}

pub fn approval_composer_height(total_height: u16) -> u16 {
    match total_height {
        0..=2 => 0,
        3..=6 => 1,
        7..=10 => 3,
        _ => 5,
    }
}

pub fn split_workspace_layout(area: Rect, metrics: WorkspaceLayoutMetrics) -> [Rect; 5] {
    let gap_height = if area.height >= 7 {
        surface::CONTENT_GAP
    } else {
        0
    };

    Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(metrics.transcript_viewport_height),
            Constraint::Length(gap_height),
            Constraint::Length(metrics.slash_panel_height),
            Constraint::Length(metrics.composer_height),
            Constraint::Length(1),
        ])
        .split(area)
        .as_ref()
        .try_into()
        .unwrap_or([
            Rect::new(area.x, area.y, area.width, area.height),
            Rect::new(area.x, area.y, 0, 0),
            Rect::new(area.x, area.y, 0, 0),
            Rect::new(area.x, area.y, 0, 0),
            Rect::new(area.x, area.y, 0, 0),
        ])
}

pub fn workspace_metrics(
    area: Rect,
    input: &str,
    attachments: &[UserImageAttachment],
    has_permission: bool,
    is_read_only_child_view: bool,
    slash_panel_height: u16,
) -> WorkspaceLayoutMetrics {
    let slash_panel_height = if has_permission {
        0
    } else {
        slash_panel_height
    };
    let composer_height = if has_permission {
        approval_composer_height(area.height)
    } else if is_read_only_child_view && input.is_empty() {
        child_read_only_composer_height(area.height)
    } else {
        composer_height(area.height, input, attachments, area.width as usize)
            .min(area.height.saturating_sub(1))
    };
    let gap_height = if area.height >= 7 {
        surface::CONTENT_GAP
    } else {
        0
    };

    WorkspaceLayoutMetrics {
        transcript_viewport_height: area
            .height
            .saturating_sub(slash_panel_height)
            .saturating_sub(composer_height)
            .saturating_sub(gap_height)
            .saturating_sub(1),
        slash_panel_height,
        composer_height,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_reserves_composer_height_below_transcript() {
        let area = Rect::new(2, 0, 120, 40);
        let input = "hello\nworld\n你好";
        let metrics = workspace_metrics(area, input, &[], false, false, 0);

        let [transcript, gap, slash, composer, footer] = split_workspace_layout(area, metrics);

        assert_eq!(transcript.height, metrics.transcript_viewport_height);
        assert_eq!(gap.height, surface::CONTENT_GAP);
        assert_eq!(slash.height, 0);
        assert_eq!(composer.height, metrics.composer_height);
        assert_eq!(footer.height, 1);

        assert_eq!(gap.y, transcript.y + transcript.height);
        assert_eq!(slash.y, gap.y + gap.height);
        assert_eq!(composer.y, slash.y + slash.height);
        assert_eq!(footer.y, composer.y + composer.height);
        assert_eq!(footer.y + footer.height, area.y + area.height);
    }

    #[test]
    fn pending_permission_uses_composer_takeover_height() {
        let area = Rect::new(0, 0, 100, 24);
        let metrics = workspace_metrics(area, "", &[], true, false, 0);

        let [transcript, gap, slash, composer, footer] = split_workspace_layout(area, metrics);

        assert_eq!(gap.height, surface::CONTENT_GAP);
        assert_eq!(slash.height, 0);
        assert_eq!(composer.height, approval_composer_height(area.height));
        assert_eq!(composer.height, metrics.composer_height);
        assert_eq!(footer.height, 1);
        assert_eq!(transcript.height, metrics.transcript_viewport_height);
        assert_eq!(footer.y + footer.height, area.y + area.height);
    }

    #[test]
    fn slash_panel_height_increases_composer_space() {
        let area = Rect::new(0, 0, 100, 24);
        let base = workspace_metrics(area, "/", &[], false, false, 0);
        let with_panel = workspace_metrics(area, "/", &[], false, false, 4);

        assert_eq!(with_panel.composer_height, base.composer_height);
        assert_eq!(with_panel.slash_panel_height, 4);
        assert!(with_panel.transcript_viewport_height < base.transcript_viewport_height);
    }

    #[test]
    fn expert_panel_height_matches_slash_panel_behavior() {
        let area = Rect::new(0, 0, 100, 24);
        let base = workspace_metrics(area, "@or", &[], false, false, 0);
        let with_panel = workspace_metrics(area, "@or", &[], false, false, 4);

        assert_eq!(with_panel.composer_height, base.composer_height);
        assert_eq!(with_panel.slash_panel_height, 4);
        assert!(with_panel.transcript_viewport_height < base.transcript_viewport_height);
    }

    #[test]
    fn split_workspace_layout_places_slash_panel_above_composer() {
        let area = Rect::new(0, 0, 100, 24);
        let metrics = workspace_metrics(area, "/", &[], false, false, 4);

        let [transcript, gap, slash, composer, footer] = split_workspace_layout(area, metrics);

        assert_eq!(slash.height, 4);
        assert_eq!(slash.y, gap.y + gap.height);
        assert_eq!(composer.y, slash.y + slash.height);
        assert_eq!(footer.y, composer.y + composer.height);
        assert_eq!(footer.y + footer.height, transcript.y + area.height);
    }

    #[test]
    fn child_read_only_view_uses_compact_composer_height() {
        let area = Rect::new(0, 0, 100, 24);
        let metrics = workspace_metrics(area, "", &[], false, true, 0);

        let [transcript, gap, slash, composer, footer] = split_workspace_layout(area, metrics);

        assert_eq!(
            composer.height,
            child_read_only_composer_height(area.height)
        );
        assert_eq!(composer.height, 4);
        assert_eq!(slash.height, 0);
        assert_eq!(gap.height, surface::CONTENT_GAP);
        assert_eq!(transcript.height, metrics.transcript_viewport_height);
        assert_eq!(footer.y, composer.y + composer.height);
        assert_eq!(footer.y + footer.height, area.y + area.height);
    }

    #[test]
    fn composer_height_grows_for_attachment_strip() {
        let area = Rect::new(0, 0, 100, 24);
        let no_attachments = workspace_metrics(area, "hello", &[], false, false, 0);
        let with_attachments = workspace_metrics(
            area,
            "hello",
            &[
                UserImageAttachment {
                    id: "img-1".into(),
                    label: "clipboard".into(),
                    mime: "image/png".into(),
                    data_url: "data:image/png;base64,AAAA".into(),
                },
                UserImageAttachment {
                    id: "img-2".into(),
                    label: "diagram.png".into(),
                    mime: "image/png".into(),
                    data_url: "data:image/png;base64,BBBB".into(),
                },
            ],
            false,
            false,
            0,
        );

        assert!(with_attachments.composer_height > no_attachments.composer_height);
    }
}
