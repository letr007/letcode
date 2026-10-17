use super::*;

#[test]
fn parses_bel_and_st_replies() {
    assert_eq!(
        parse_reply(b"\x1b]11;rgb:1a1b/1a1b/2626\x1b\\"),
        Some((26, 26, 38))
    );
    assert_eq!(parse_reply(b"\x1b]11;rgb:11/11/11\x07"), Some((17, 17, 17)));
}

#[test]
fn parses_component_widths_and_alpha() {
    assert_eq!(parse_color_spec("rgb:f/e/d"), Some((255, 238, 221)));
    assert_eq!(parse_color_spec("rgb:00/80/ff"), Some((0, 128, 255)));
    assert_eq!(
        parse_color_spec("rgba:1111/1111/1111/ffff"),
        Some((17, 17, 17))
    );
}

#[test]
fn rejects_incomplete_or_malformed_replies() {
    assert_eq!(parse_reply(b"\x1b]10;rgb:eeee/eeee/eeee\x1b\\"), None);
    assert_eq!(parse_reply(b"\x1b]11;rgb:1111/1111/1111"), None);
    assert_eq!(parse_color_spec("rgb:fffff/0/0"), None);
    assert_eq!(parse_color_spec("rgb:11/11"), None);
    assert_eq!(parse_color_spec("rgb:11/11/11/11"), None);
    assert_eq!(parse_color_spec("rgb:gg/11/11"), None);
    assert_eq!(parse_color_spec("nope:11/11/11"), None);
}

#[test]
fn finds_the_reply_after_unrelated_bytes() {
    assert_eq!(
        parse_reply(b"xy\x1b]11;rgb:0000/0000/0000\x07z"),
        Some((0, 0, 0))
    );
}
