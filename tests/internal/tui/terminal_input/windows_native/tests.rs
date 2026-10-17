use super::*;

fn test_input() -> std::mem::ManuallyDrop<NativeInput> {
    std::mem::ManuallyDrop::new(NativeInput {
        handle: std::ptr::null_mut(),
        output_handle: std::ptr::null_mut(),
        original_mode: 0,
        parser: Parser::new(),
        raw_utf16: Utf16Decoder::default(),
        native_surrogate: None,
        pending: VecDeque::new(),
        mouse_buttons: 0,
        raw_sequence: RawSequence::None,
        window_origin: (0, 0),
        window_size: (80, 24),
    })
}

fn key_record(unicode: u16, vk: u16, down: bool, modifiers: u32) -> INPUT_RECORD {
    INPUT_RECORD {
        EventType: KEY_EVENT as u16,
        Event: INPUT_RECORD_0 {
            KeyEvent: KEY_EVENT_RECORD {
                bKeyDown: i32::from(down),
                wRepeatCount: 1,
                wVirtualKeyCode: vk,
                wVirtualScanCode: 0,
                uChar: KEY_EVENT_RECORD_0 {
                    UnicodeChar: unicode,
                },
                dwControlKeyState: modifiers,
            },
        },
    }
}

#[test]
fn native_character_commits_and_clipboard_shortcuts_are_preserved() {
    let mut input = test_input();
    input.process_records(&[
        key_record(
            '@' as u16,
            b'Q' as u16,
            true,
            RIGHT_ALT_PRESSED | LEFT_CTRL_PRESSED,
        ),
        key_record('é' as u16, VK_MENU, false, 0),
        key_record(0x16, b'V' as u16, true, LEFT_CTRL_PRESSED),
        key_record(0x16, b'V' as u16, false, LEFT_CTRL_PRESSED),
        key_record(0x1b, VK_ESCAPE, true, 0),
        key_record(0x1b, VK_ESCAPE, false, 0),
    ]);
    let expected = [
        (KeyCode::Char('@'), KeyModifiers::NONE),
        (KeyCode::Char('é'), KeyModifiers::NONE),
        (KeyCode::Char('v'), KeyModifiers::CONTROL),
        (KeyCode::Esc, KeyModifiers::NONE),
    ];
    for (code, modifiers) in expected {
        assert_eq!(
            input.pending.pop_front(),
            Some(Event::Key(KeyEvent::new(code, modifiers)))
        );
    }
    assert!(input.pending.is_empty());
}

#[test]
fn native_paste_records_keep_newlines_and_surrogates_in_one_event() {
    let mut input = test_input();
    let text = "line\r\n🙂\r\n/quit";
    let records = format!("\x1b[200~{text}\x1b[201~")
        .encode_utf16()
        .flat_map(|unit| [key_record(unit, 0, true, 0), key_record(unit, 0, false, 0)])
        .collect::<Vec<_>>();
    for chunk in records.chunks(3) {
        input.process_records(chunk);
        // Empty polls cannot flush any part of an active paste.
        if input.parser.paste_in_progress() {
            assert!(input.parser.flush().is_empty());
        }
    }
    assert_eq!(
        input.pending.drain(..).collect::<Vec<_>>(),
        vec![Event::Paste(text.into())]
    );
}

#[test]
fn native_mouse_transitions_share_window_relative_coordinates() {
    let mut input = test_input();
    input.window_origin = (10, 20);
    for (buttons, flags) in [
        (FROM_LEFT_1ST_BUTTON_PRESSED, 0),
        (FROM_LEFT_1ST_BUTTON_PRESSED, MOUSE_MOVED),
        (0, 0),
    ] {
        unsafe {
            input.process_mouse(&MOUSE_EVENT_RECORD {
                dwMousePosition: COORD { X: 14, Y: 25 },
                dwButtonState: buttons,
                dwControlKeyState: 0,
                dwEventFlags: flags,
            });
        }
    }
    for kind in [
        MouseEventKind::Down(MouseButton::Left),
        MouseEventKind::Drag(MouseButton::Left),
        MouseEventKind::Up(MouseButton::Left),
    ] {
        assert!(
            matches!(input.pending.pop_front(), Some(Event::Mouse(event)) if event.kind == kind && event.column == 4 && event.row == 5)
        );
    }
}

#[test]
fn native_records_deliver_alt_v_for_clipboard_paste() {
    let mut input = test_input();
    input.process_records(&[
        key_record(b'v' as u16, b'V' as u16, true, LEFT_ALT_PRESSED),
        key_record(b'v' as u16, b'V' as u16, false, LEFT_ALT_PRESSED),
    ]);
    assert_eq!(
        input.pending.pop_front(),
        Some(Event::Key(KeyEvent::new(
            KeyCode::Char('v'),
            KeyModifiers::ALT
        )))
    );
    assert!(input.pending.is_empty());
}

#[test]
fn native_records_preserve_key_mouse_and_resize_events() {
    let mut input = NativeInput {
        handle: std::ptr::null_mut(),
        output_handle: std::ptr::null_mut(),
        original_mode: 0,
        parser: Parser::new(),
        raw_utf16: Utf16Decoder::default(),
        native_surrogate: None,
        pending: VecDeque::new(),
        mouse_buttons: 0,
        raw_sequence: RawSequence::None,
        window_origin: (0, 0),
        window_size: (80, 24),
    };
    let records = [
        INPUT_RECORD {
            EventType: KEY_EVENT as u16,
            Event: INPUT_RECORD_0 {
                KeyEvent: KEY_EVENT_RECORD {
                    bKeyDown: 1,
                    wRepeatCount: 1,
                    wVirtualKeyCode: VK_F1,
                    wVirtualScanCode: 0,
                    uChar: KEY_EVENT_RECORD_0 { UnicodeChar: 0 },
                    dwControlKeyState: LEFT_CTRL_PRESSED,
                },
            },
        },
        INPUT_RECORD {
            EventType: KEY_EVENT as u16,
            Event: INPUT_RECORD_0 {
                KeyEvent: KEY_EVENT_RECORD {
                    bKeyDown: 1,
                    wRepeatCount: 2,
                    wVirtualKeyCode: VK_RETURN,
                    wVirtualScanCode: 0,
                    uChar: KEY_EVENT_RECORD_0 {
                        UnicodeChar: b'\r' as u16,
                    },
                    dwControlKeyState: SHIFT_PRESSED,
                },
            },
        },
        INPUT_RECORD {
            EventType: KEY_EVENT as u16,
            Event: INPUT_RECORD_0 {
                KeyEvent: KEY_EVENT_RECORD {
                    bKeyDown: 1,
                    wRepeatCount: 2,
                    wVirtualKeyCode: b'X' as u16,
                    wVirtualScanCode: 0,
                    uChar: KEY_EVENT_RECORD_0 {
                        UnicodeChar: b'x' as u16,
                    },
                    dwControlKeyState: LEFT_ALT_PRESSED,
                },
            },
        },
        INPUT_RECORD {
            EventType: MOUSE_EVENT as u16,
            Event: INPUT_RECORD_0 {
                MouseEvent: MOUSE_EVENT_RECORD {
                    dwMousePosition: COORD { X: 4, Y: 5 },
                    dwButtonState: FROM_LEFT_1ST_BUTTON_PRESSED,
                    dwControlKeyState: SHIFT_PRESSED,
                    dwEventFlags: 0,
                },
            },
        },
        INPUT_RECORD {
            EventType: WINDOW_BUFFER_SIZE_EVENT as u16,
            Event: INPUT_RECORD_0 {
                WindowBufferSizeEvent: WINDOW_BUFFER_SIZE_RECORD {
                    dwSize: COORD { X: 80, Y: 24 },
                },
            },
        },
    ];
    input.process_records(&records);

    assert!(matches!(
        input.pending.pop_front(),
        Some(Event::Key(event))
            if event.code == KeyCode::F(1)
                && event.modifiers == KeyModifiers::CONTROL
                && event.kind == KeyEventKind::Press
    ));
    assert!(matches!(
        input.pending.pop_front(),
        Some(Event::Key(event))
            if event.code == KeyCode::Enter
                && event.modifiers == KeyModifiers::SHIFT
    ));
    assert!(matches!(
        input.pending.pop_front(),
        Some(Event::Key(event))
            if event.code == KeyCode::Enter
                && event.modifiers == KeyModifiers::SHIFT
    ));
    assert!(matches!(
        input.pending.pop_front(),
        Some(Event::Key(event))
            if event.code == KeyCode::Char('x')
                && event.modifiers == KeyModifiers::ALT
    ));
    assert!(matches!(
        input.pending.pop_front(),
        Some(Event::Key(event))
            if event.code == KeyCode::Char('x')
                && event.modifiers == KeyModifiers::ALT
    ));
    assert!(matches!(
        input.pending.pop_front(),
        Some(Event::Mouse(event))
            if event.kind == MouseEventKind::Down(MouseButton::Left)
                && event.column == 4
                && event.row == 5
                && event.modifiers == KeyModifiers::SHIFT
    ));
    assert_eq!(input.pending.pop_front(), Some(Event::Resize(80, 24)));
    std::mem::forget(input);
}
