use ratatui::layout::Rect;

use crate::tui::measure::display_width;
use crate::tui::state::{ComposerToken, TuiState};

use super::super::surface;
use super::{composer::composer_textarea_width, slash_panel};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct WorkspaceLayoutMetrics {
    pub transcript_viewport_height: u16,
    pub slash_panel_height: u16,
    pub composer_height: u16,
}

pub const SIDEBAR_WIDTH: u16 = 42;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SidebarLayout {
    pub main: Rect,
    pub sidebar: Option<Rect>,
    pub overlay: bool,
}

pub fn split_sidebar_layout(area: Rect, visible: bool) -> SidebarLayout {
    if !visible || area.width == 0 || area.height == 0 {
        return SidebarLayout {
            main: area,
            sidebar: None,
            overlay: false,
        };
    }

    let sidebar_width = SIDEBAR_WIDTH.min(area.width);
    if area.width > 120 {
        SidebarLayout {
            main: Rect::new(area.x, area.y, area.width - sidebar_width, area.height),
            sidebar: Some(Rect::new(
                area.right().saturating_sub(sidebar_width),
                area.y,
                sidebar_width,
                area.height,
            )),
            overlay: false,
        }
    } else {
        SidebarLayout {
            main: area,
            sidebar: Some(Rect::new(
                area.right().saturating_sub(sidebar_width),
                area.y,
                sidebar_width,
                area.height,
            )),
            overlay: true,
        }
    }
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
    tokens: &[ComposerToken],
    width: usize,
) -> u16 {
    match total_height {
        0..=2 => 0,
        3..=5 => 1,
        6..=8 => 3,
        _ => {
            let text_width = composer_textarea_width(u16::try_from(width).unwrap_or(u16::MAX));
            let rows = composer_inline_row_count(tokens, input, text_width.max(1)) as u16;
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

fn composer_inline_row_count(tokens: &[ComposerToken], input: &str, width: usize) -> usize {
    let markers = input
        .chars()
        .filter(|ch| *ch == crate::tui::state::COMPOSER_ATTACHMENT_MARKER)
        .count();
    assert_eq!(
        markers,
        tokens.len(),
        "composer token markers must match tokens"
    );
    let mut tokens = tokens.iter();
    let mut image_index = 0usize;
    let mut rows = 1usize;
    let mut column = 0usize;
    let mut ended_by_exact_fill = false;
    for ch in input.chars() {
        if ch == crate::tui::state::COMPOSER_ATTACHMENT_MARKER {
            let token = tokens.next().expect("composer marker has matching token");
            let token_width = display_width(&token.display_text(image_index));
            if column > 0 && column + token_width > width {
                rows += 1;
                column = 0;
            }
            column += token_width;
            if matches!(token, ComposerToken::Image(_)) {
                image_index += 1;
            }
        } else if ch == '\n' {
            rows += 1;
            column = 0;
        } else {
            let ch_width = display_width(&ch.to_string());
            if ch_width > 0 && column > 0 && column + ch_width > width {
                rows += 1;
                column = 0;
            }
            column += ch_width;
        }
        ended_by_exact_fill = ch != '\n' && column >= width;
    }
    rows + usize::from(ended_by_exact_fill)
}

pub fn child_read_only_composer_height(total_height: u16) -> u16 {
    match total_height {
        0..=2 => 0,
        3..=6 => 1,
        7 => 4,
        _ => 5,
    }
}

pub fn slash_panel_height(state: &TuiState) -> u16 {
    slash_panel::slash_panel_reserved_height(state)
}

pub fn approval_composer_height(total_height: u16) -> u16 {
    match total_height {
        0..=2 => 0,
        3..=6 => 2,
        7..=10 => 5,
        11..=16 => 7,
        _ => 9,
    }
}

const QUESTION_SHELL_ROWS: u16 = 4;
const QUESTION_MIN_COMPOSER_ROWS: u16 = 6;

pub fn question_composer_height(total_height: u16) -> u16 {
    question_composer_height_for_content(total_height, 2)
}

/// Height for the connected question surface: content, its padding/cap, and its action row.
/// The caller supplies the already display-width-aware number of content rows.
pub fn question_composer_height_for_content(total_height: u16, content_rows: usize) -> u16 {
    let requested = u16::try_from(content_rows)
        .unwrap_or(u16::MAX)
        .saturating_add(QUESTION_SHELL_ROWS)
        .max(QUESTION_MIN_COMPOSER_ROWS);
    // The connected prompt shares the workspace with the content gap and global footer.
    // Never request more rows than that stack can physically contain.
    requested.min(total_height.saturating_sub(question_workspace_overhead(total_height)))
}

fn question_workspace_overhead(total_height: u16) -> u16 {
    1 + u16::from(total_height >= 7) * surface::CONTENT_GAP
}

pub fn split_workspace_layout(area: Rect, metrics: WorkspaceLayoutMetrics) -> [Rect; 5] {
    let gap_height = if area.height >= 7 {
        surface::CONTENT_GAP
    } else {
        0
    };

    let footer_height = u16::from(area.height > 0);
    let available = area.height.saturating_sub(footer_height);
    let fixed_height = gap_height.saturating_add(metrics.slash_panel_height);
    let composer_height = metrics
        .composer_height
        .min(available.saturating_sub(fixed_height));
    let transcript_height = metrics.transcript_viewport_height.min(
        available
            .saturating_sub(fixed_height)
            .saturating_sub(composer_height),
    );
    let gap_height = gap_height.min(available.saturating_sub(transcript_height));
    let slash_height = metrics.slash_panel_height.min(
        available
            .saturating_sub(transcript_height)
            .saturating_sub(gap_height),
    );
    let composer_height = composer_height.min(
        available
            .saturating_sub(transcript_height)
            .saturating_sub(gap_height)
            .saturating_sub(slash_height),
    );
    let mut y = area.y;
    let next = |height: u16, y: &mut u16| {
        let rect = Rect::new(area.x, *y, area.width, height);
        *y = y.saturating_add(height);
        rect
    };
    let transcript = next(transcript_height, &mut y);
    let gap = next(gap_height, &mut y);
    let slash = next(slash_height, &mut y);
    let composer = next(composer_height, &mut y);
    let footer = Rect::new(
        area.x,
        area.bottom().saturating_sub(footer_height),
        area.width,
        footer_height,
    );
    [transcript, gap, slash, composer, footer]
}

pub fn workspace_metrics(
    area: Rect,
    input: &str,
    tokens: &[ComposerToken],
    has_permission: bool,
    has_question: bool,
    is_read_only_child_view: bool,
    slash_panel_height: u16,
) -> WorkspaceLayoutMetrics {
    let slash_panel_height = if has_permission || has_question {
        0
    } else {
        slash_panel_height
    };
    let composer_height = if has_permission {
        approval_composer_height(area.height)
    } else if has_question {
        question_composer_height(area.height)
    } else if is_read_only_child_view && input.is_empty() {
        child_read_only_composer_height(area.height)
    } else {
        composer_height(area.height, input, tokens, area.width as usize)
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
    use crate::user_content::UserImageAttachment;

    #[test]
    fn sidebar_layout_splits_wide_and_overlays_narrow_workspaces() {
        let wide = split_sidebar_layout(Rect::new(0, 0, 160, 30), true);
        assert_eq!(wide.main.width, 118);
        assert_eq!(wide.sidebar.expect("sidebar").width, 42);
        assert!(!wide.overlay);

        let narrow = split_sidebar_layout(Rect::new(0, 0, 100, 30), true);
        assert_eq!(narrow.main.width, 100);
        assert_eq!(narrow.sidebar.expect("sidebar").x, 58);
        assert!(narrow.overlay);
    }

    #[test]
    fn inline_attachment_row_count_keeps_tokens_atomic_at_narrow_widths() {
        let attachment = UserImageAttachment {
            id: "img-1".into(),
            label: "clipboard".into(),
            mime: "image/png".into(),
            data_url: "data:image/png;base64,AAAA".into(),
        };

        assert_eq!(
            composer_inline_row_count(&[ComposerToken::Image(attachment)], "ab\u{fffc}", 5),
            3
        );
        assert_eq!(composer_inline_row_count(&[], "abcd", 4), 2);
        assert_eq!(composer_inline_row_count(&[], "abcd\n", 4), 2);
        assert_eq!(composer_inline_row_count(&[], "abcde", 4), 2);
    }

    #[test]
    fn pending_permission_uses_composer_takeover_height() {
        let area = Rect::new(0, 0, 100, 24);
        let metrics = workspace_metrics(area, "", &[], true, false, false, 0);

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
    fn pending_question_uses_bottom_panel_height_and_hides_slash_panel() {
        let area = Rect::new(0, 0, 100, 24);
        let metrics = workspace_metrics(area, "/per", &[], false, true, false, 4);

        let [_transcript, gap, slash, composer, footer] = split_workspace_layout(area, metrics);

        assert_eq!(slash.height, 0);
        assert_eq!(composer.height, question_composer_height(area.height));
        assert_eq!(composer.height, 6);
        assert_eq!(gap.height, surface::CONTENT_GAP);
        assert_eq!(footer.y, composer.y + composer.height);
    }

    #[test]
    fn question_layout_never_overflows_short_workspaces() {
        for height in 7..=10 {
            let area = Rect::new(0, 0, 80, height);
            let metrics = workspace_metrics(area, "", &[], false, true, false, 0);
            let [transcript, gap, slash, composer, footer] = split_workspace_layout(area, metrics);

            assert!(composer.height + gap.height + footer.height <= area.height);
            assert_eq!(slash.height, 0);
            assert_eq!(footer.y + footer.height, area.y + area.height);
            assert_eq!(composer.y + composer.height, footer.y);
            assert_eq!(transcript.y + transcript.height, gap.y);
        }
    }

    #[test]
    fn question_composer_grows_with_content_without_taking_unused_workspace() {
        let workspace_height = 30;
        let short = question_composer_height_for_content(workspace_height, 5);
        let detailed = question_composer_height_for_content(workspace_height, 13);
        let overflowing = question_composer_height_for_content(workspace_height, 80);

        assert_eq!(short, 9);
        assert_eq!(detailed, 17);
        assert!(detailed > short);
        assert_eq!(overflowing, 28);
    }

    #[test]
    fn height_seven_terminal_keeps_question_composer_above_the_global_footer() {
        let terminal = Rect::new(0, 0, 80, 7);
        let workspace = workspace_area(terminal);
        let metrics = workspace_metrics(workspace, "", &[], false, true, false, 0);
        let [_transcript, _gap, _slash, composer, footer] =
            split_workspace_layout(workspace, metrics);

        assert_eq!(workspace.height, 6);
        assert_eq!(composer.height, 5);
        assert_eq!(composer.bottom(), footer.y);
    }

    #[test]
    fn child_read_only_composer_grows_only_when_a_transcript_row_remains() {
        let compact = workspace_metrics(Rect::new(0, 0, 80, 7), "", &[], false, false, true, 0);
        let centered = workspace_metrics(Rect::new(0, 0, 80, 8), "", &[], false, false, true, 0);

        assert_eq!(compact.composer_height, 4);
        assert_eq!(compact.transcript_viewport_height, 1);
        assert_eq!(centered.composer_height, 5);
        assert_eq!(centered.transcript_viewport_height, 1);
    }
}
