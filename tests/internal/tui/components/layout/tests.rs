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
    let metrics = workspace_metrics(area, "", &[], true, false, false, 0, false);

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
    let metrics = workspace_metrics(area, "/per", &[], false, true, false, 4, false);

    let [_transcript, gap, slash, composer, footer] = split_workspace_layout(area, metrics);

    assert_eq!(slash.height, 0);
    assert_eq!(
        composer.height,
        question_composer_height(area.height, false)
    );
    assert_eq!(composer.height, 6);
    assert_eq!(gap.height, surface::CONTENT_GAP);
    assert_eq!(footer.y, composer.y + composer.height);
}

#[test]
fn question_layout_never_overflows_short_workspaces() {
    for height in 7..=10 {
        let area = Rect::new(0, 0, 80, height);
        let metrics = workspace_metrics(area, "", &[], false, true, false, 0, false);
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
    let short = question_composer_height_for_content(workspace_height, 5, false);
    let detailed = question_composer_height_for_content(workspace_height, 13, false);
    let overflowing = question_composer_height_for_content(workspace_height, 80, false);

    assert_eq!(short, 9);
    assert_eq!(detailed, 17);
    assert!(detailed > short);
    assert_eq!(overflowing, 28);
}

#[test]
fn height_seven_terminal_keeps_question_composer_above_the_global_footer() {
    let terminal = Rect::new(0, 0, 80, 7);
    let workspace = workspace_area(terminal);
    let metrics = workspace_metrics(workspace, "", &[], false, true, false, 0, false);
    let [_transcript, _gap, _slash, composer, footer] = split_workspace_layout(workspace, metrics);

    assert_eq!(workspace.height, 6);
    assert_eq!(composer.height, 5);
    assert_eq!(composer.bottom(), footer.y);
}

#[test]
fn wireframe_composer_reserves_an_extra_row_for_card_spacing() {
    let area = Rect::new(0, 0, 80, 12);
    let plain = workspace_metrics(area, "", &[], false, false, false, 0, false);
    let wireframe = workspace_metrics(area, "", &[], false, false, false, 0, true);

    assert_eq!(wireframe.composer_height, plain.composer_height + 1);
    assert_eq!(
        wireframe.transcript_viewport_height,
        plain.transcript_viewport_height - 1
    );
}

#[test]
fn child_read_only_composer_grows_only_when_a_transcript_row_remains() {
    let compact = workspace_metrics(
        Rect::new(0, 0, 80, 7),
        "",
        &[],
        false,
        false,
        true,
        0,
        false,
    );
    let centered = workspace_metrics(
        Rect::new(0, 0, 80, 8),
        "",
        &[],
        false,
        false,
        true,
        0,
        false,
    );

    assert_eq!(compact.composer_height, 4);
    assert_eq!(compact.transcript_viewport_height, 1);
    assert_eq!(centered.composer_height, 5);
    assert_eq!(centered.transcript_viewport_height, 1);
}
