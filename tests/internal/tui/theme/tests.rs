use super::*;

#[test]
fn rainbow_cycles_accent_without_changing_error_semantics() {
    let first = Theme::for_name(ThemeName::Rainbow, 0, None);
    let next = Theme::for_name(ThemeName::Rainbow, 3, None);

    assert_ne!(first.accent, next.accent);
    assert_ne!(first.user, next.user);
    assert_ne!(first.assistant, next.assistant);
    assert_ne!(first.border, next.border);
    assert_eq!(first.error, next.error);
    assert_eq!(first.warning, next.warning);
    assert_eq!(first.success, next.success);
    assert_eq!(Theme::for_name(ThemeName::Dark, 0, None), Theme::dark());
}

fn channels(color: Color) -> Rgb {
    let Color::Rgb(red, green, blue) = color else {
        panic!("expected an rgb surface, got {color:?}");
    };
    (red, green, blue)
}

fn channel_sum(color: Color) -> u32 {
    let (red, green, blue) = channels(color);
    u32::from(red) + u32::from(green) + u32::from(blue)
}

#[test]
fn plain_theme_leaves_the_screen_to_the_terminal() {
    let plain = Theme::for_name(ThemeName::Plain, 0, None);
    let dark = Theme::dark();

    assert_eq!(plain.root_bg, Color::Reset);
    assert_eq!(plain.text, dark.text);
    assert_eq!(plain.accent, dark.accent);
    assert_eq!(plain.on_accent, dark.on_accent);
    assert_eq!(plain.diff_add_bg, dark.diff_add_bg);
    assert_eq!(plain.diff_delete_bg, dark.diff_delete_bg);
}

#[test]
fn plain_theme_lifts_panels_above_a_detected_background() {
    let background: Rgb = (26, 27, 38);
    let plain = Theme::plain_for(Some(background));
    let background = channel_sum(Color::Rgb(background.0, background.1, background.2));

    for surface in [plain.surface_bg, plain.element_bg, plain.elevated_bg] {
        assert!(
            channel_sum(surface) > background,
            "{surface:?} should sit above the terminal background"
        );
    }
    assert!(channel_sum(plain.surface_bg) < channel_sum(plain.element_bg));
    assert!(channel_sum(plain.element_bg) < channel_sum(plain.elevated_bg));
}

#[test]
fn plain_assumes_a_dark_background_when_there_is_none_to_use() {
    let assumed = Theme::plain_for(None);

    assert!(channel_sum(assumed.element_bg) > channel_sum(Color::Rgb(24, 25, 34)));
    assert_eq!(Theme::plain_for(Some((255, 255, 255))), assumed);
    assert_eq!(Theme::plain_for(Some((250, 250, 250))), assumed);
    assert_eq!(Theme::plain_for(Some((120, 120, 120))), assumed);
}

#[test]
fn plain_keeps_the_background_hue_in_its_panels() {
    let background: Rgb = (26, 27, 38);
    let plain = Theme::plain_for(Some(background));

    let cast = background.2 - background.0;

    for surface in [plain.surface_bg, plain.element_bg, plain.elevated_bg] {
        let channels = channels(surface);
        assert!(
            channels.2 - channels.0 >= cast,
            "{channels:?} should keep the background's blue cast"
        );
    }
}

#[test]
fn glass_theme_leaves_every_surface_to_the_terminal() {
    let glass = Theme::for_name(ThemeName::Glass, 0, Some((26, 27, 38)));

    for surface in [
        glass.root_bg,
        glass.surface_bg,
        glass.element_bg,
        glass.elevated_bg,
    ] {
        assert_eq!(surface, Color::Reset);
    }
    assert!(!glass.card_frame);
    assert!(!glass.paints_panels());
    assert!(Theme::plain_for(None).paints_panels());
}

#[test]
fn wireframe_theme_keeps_transparent_surfaces_and_enables_card_frames() {
    let wireframe = Theme::for_name(ThemeName::Wireframe, 0, None);

    assert!(wireframe.card_frame);
    assert_eq!(wireframe.root_bg, Color::Reset);
    assert_eq!(wireframe.element_bg, Color::Reset);
    assert_eq!(wireframe.border, Color::Rgb(0x70, 0x80, 0x9a));
    assert_eq!(wireframe.diff_add_bg, Theme::dark().diff_add_bg);
    assert_eq!(wireframe.diff_delete_bg, Theme::dark().diff_delete_bg);
    assert_eq!(wireframe.diff_hunk_bg, Theme::dark().diff_hunk_bg);
}

#[test]
fn canvas_stays_concrete_without_a_painted_root() {
    assert_eq!(Theme::dark().canvas(), Theme::dark().root_bg);
    assert_eq!(Theme::plain_for(None).canvas(), Color::Rgb(0, 0, 0));
    assert_eq!(Theme::glass().canvas(), Color::Rgb(0, 0, 0));
}

#[test]
fn plain_lifts_panels_off_a_black_background() {
    let plain = Theme::plain_for(Some((0, 0, 0)));

    for surface in [plain.surface_bg, plain.element_bg, plain.elevated_bg] {
        assert!(
            channel_sum(surface) > 0,
            "{surface:?} should not stay black"
        );
    }
}
