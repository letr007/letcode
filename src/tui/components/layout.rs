use ratatui::layout::{Constraint, Direction, Layout, Rect};

use crate::tui::measure::wrapped_row_count;

use super::super::surface;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct WorkspaceLayoutMetrics {
    pub transcript_viewport_height: u16,
    pub composer_height: u16,
    pub permission_height: u16,
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

pub fn composer_height(total_height: u16, input: &str, width: usize) -> u16 {
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
            (surface::PROMPT_INNER_PAD_TOP
                + clamped
                + surface::PROMPT_METADATA_PAD_TOP
                + 1
                + surface::PROMPT_INNER_PAD_BOTTOM)
                .min(total_height.saturating_sub(2))
        }
    }
}

pub fn permission_height(total_height: u16) -> u16 {
    match total_height {
        0..=2 => 0,
        3..=6 => 1,
        7..=10 => 3,
        _ => 5,
    }
}

pub fn split_workspace_layout(
    area: Rect,
    metrics: WorkspaceLayoutMetrics,
    has_permission: bool,
) -> [Rect; 4] {
    let second_height = if has_permission {
        metrics.permission_height
    } else if area.height >= 7 {
        surface::CONTENT_GAP
    } else {
        0
    };

    Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(0),
            Constraint::Length(second_height),
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
        ])
}

pub fn workspace_metrics(area: Rect, input: &str, has_permission: bool) -> WorkspaceLayoutMetrics {
    let composer_height = composer_height(area.height, input, area.width as usize);
    WorkspaceLayoutMetrics {
        transcript_viewport_height: area
            .height
            .saturating_sub(composer_height)
            .saturating_sub(1)
            .saturating_sub(if has_permission {
                permission_height(area.height)
            } else {
                0
            })
            .saturating_sub(if area.height >= 7 {
                surface::CONTENT_GAP
            } else {
                0
            }),
        composer_height,
        permission_height: if has_permission {
            permission_height(area.height)
        } else {
            0
        },
    }
}
