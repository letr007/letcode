use super::vt;
use crossterm::event::{Event, KeyCode, KeyModifiers, MouseButton, MouseEventKind};

#[test]
fn escape_and_shift_tab_preserve_keyboard_actions() {
    assert!(
        matches!(vt::parse_vt_chunks(&[b"\x1b"]).as_slice(), [Event::Key(key)] if key.code == KeyCode::Esc)
    );
    assert!(
        matches!(vt::parse_vt_chunks(&[b"\x1b[Z"]).as_slice(), [Event::Key(key)] if key.code == KeyCode::BackTab)
    );
    assert!(
        matches!(vt::parse_vt_chunks(&[b"\x16"]).as_slice(), [Event::Key(key)] if key.code == KeyCode::Char('v') && key.modifiers == KeyModifiers::CONTROL)
    );
}

#[test]
fn bracketed_paste_is_one_event_and_preserves_large_unicode_text() {
    let text = format!("{}\r\n🙂", "a".repeat(999_997));
    assert_eq!(text.chars().count(), 1_000_000);
    let bytes = format!("\x1b[200~{text}\x1b[201~").into_bytes();
    let events = vt::parse_vt_chunks(&bytes.chunks(4096).collect::<Vec<_>>());
    assert_eq!(events, vec![Event::Paste(text)]);
}

#[test]
fn parser_does_not_emit_a_partial_paste_before_its_end_marker() {
    let mut parser = vt::parser();
    assert!(parser.feed(b"\x1b[200~partial", true).is_empty());
    assert!(parser.flush().is_empty());
    assert_eq!(
        parser.feed(b" text\x1b[201~", true),
        vec![Event::Paste("partial text".to_owned())]
    );
}

#[test]
fn paste_marker_split_across_an_empty_poll_does_not_leak_escape_keys() {
    let mut parser = vt::parser();
    assert!(parser.feed(b"\x1b[20", true).is_empty());
    assert!(parser.flush().is_empty());
    assert_eq!(
        parser.feed(b"0~line\r\n\xf0\x9f\x99\x82\x1b[201~", true),
        vec![Event::Paste("line\r\n🙂".to_owned())]
    );
}

#[test]
fn split_utf8_and_command_characters_are_preserved() {
    let command = "<script>& && $HOME \\ \"quoted\" 🙂";
    let mut parser = vt::parser();
    let mut events = Vec::new();
    events.extend(parser.feed(b"\x1b[200~<script>& && $HOME \\ ", true));
    events.extend(parser.feed(b"\"quoted\" \xf0", true));
    events.extend(parser.feed(b"\x9f\x99\x82\x1b[201~", true));
    assert_eq!(events, vec![Event::Paste(command.to_owned())]);
}

#[test]
fn key_mouse_and_modifier_sequences_convert_to_crossterm() {
    let events = vt::parse_vt_chunks(&[b"\x1b[A\x1b[1;5C\x1b[<0;4;5M\x1b[<32;5;6M\x1b[<3;5;6m"]);
    assert!(matches!(events[0], Event::Key(key) if key.code == KeyCode::Up));
    assert!(
        matches!(events[1], Event::Key(key) if key.code == KeyCode::Right && key.modifiers == KeyModifiers::CONTROL)
    );
    assert!(
        matches!(events[2], Event::Mouse(mouse) if mouse.kind == MouseEventKind::Down(MouseButton::Left) && mouse.column == 3 && mouse.row == 4)
    );
    assert!(
        matches!(events[3], Event::Mouse(mouse) if mouse.kind == MouseEventKind::Drag(MouseButton::Left) && mouse.column == 4 && mouse.row == 5)
    );
    assert!(
        matches!(events[4], Event::Mouse(mouse) if mouse.kind == MouseEventKind::Up(MouseButton::Left) && mouse.column == 4 && mouse.row == 5)
    );
}

#[test]
fn alt_v_escape_prefix_survives_split_chunks() {
    for chunks in [
        vec![&b"\x1bv"[..]],
        vec![&b"\x1b"[..], &b"v"[..]],
        vec![&b"\x1bV"[..]],
    ] {
        assert!(
            matches!(vt::parse_vt_chunks(&chunks).as_slice(), [Event::Key(key)] if matches!(key.code, KeyCode::Char('v' | 'V')) && key.modifiers == KeyModifiers::ALT),
            "{chunks:?}"
        );
    }
}

#[test]
fn utf16_surrogate_pairs_are_encoded_across_input_chunks() {
    let mut decoder = vt::Utf16Decoder::default();
    assert!(decoder.push(0xd83d).is_empty());
    assert_eq!(decoder.push(0xde42), "🙂".as_bytes());
}
