#[cfg(windows)]
use std::collections::VecDeque;
use std::io;
use std::time::Duration;
#[cfg(windows)]
use std::time::Instant;

use crossterm::event::Event;

#[cfg(any(test, windows))]
mod vt {
    use super::*;
    use crossterm::event::{
        KeyCode as CrosstermKeyCode, KeyEvent, KeyEventKind, KeyEventState, KeyModifiers,
        MouseButton, MouseEvent, MouseEventKind,
    };
    use termwiz::input::{
        InputEvent, InputParser, KeyCode, KeyEvent as TermwizKeyEvent, Modifiers, MouseButtons,
        MouseEvent as TermwizMouseEvent,
    };

    pub(crate) struct Parser {
        parser: InputParser,
        mouse_buttons: MouseButtons,
        paste_active: bool,
        paste_start_match: usize,
        paste_end_match: usize,
    }

    const PASTE_START: &[u8] = b"\x1b[200~";
    const PASTE_END: &[u8] = b"\x1b[201~";

    impl Parser {
        pub(crate) fn new() -> Self {
            Self {
                parser: InputParser::new(),
                mouse_buttons: MouseButtons::NONE,
                paste_active: false,
                paste_start_match: 0,
                paste_end_match: 0,
            }
        }

        #[allow(dead_code)]
        pub(crate) fn feed(&mut self, bytes: &[u8], maybe_more: bool) -> Vec<Event> {
            self.observe_bytes(bytes);
            self.feed_observed(bytes, maybe_more)
        }

        pub(crate) fn observe_bytes(&mut self, bytes: &[u8]) {
            self.observe_paste_markers(bytes);
        }

        pub(crate) fn feed_observed(&mut self, bytes: &[u8], maybe_more: bool) -> Vec<Event> {
            self.parser
                .parse_as_vec(bytes, maybe_more)
                .into_iter()
                .filter_map(|event| self.convert_event(event))
                .collect()
        }

        pub(crate) fn paste_in_progress(&self) -> bool {
            self.paste_active || self.paste_start_match != 0
        }

        pub(crate) fn flush(&mut self) -> Vec<Event> {
            if self.paste_active || self.paste_start_match > 1 {
                return Vec::new();
            }
            self.paste_start_match = 0;
            self.parser
                .parse_as_vec(&[], false)
                .into_iter()
                .filter_map(|event| self.convert_event(event))
                .collect()
        }

        fn observe_paste_markers(&mut self, bytes: &[u8]) {
            for &byte in bytes {
                if self.paste_active {
                    if byte == PASTE_END[self.paste_end_match] {
                        self.paste_end_match += 1;
                        if self.paste_end_match == PASTE_END.len() {
                            self.paste_active = false;
                            self.paste_end_match = 0;
                        }
                    } else {
                        self.paste_end_match = usize::from(byte == PASTE_END[0]);
                    }
                    continue;
                }

                if byte == PASTE_START[self.paste_start_match] {
                    self.paste_start_match += 1;
                    if self.paste_start_match == PASTE_START.len() {
                        self.paste_active = true;
                        self.paste_start_match = 0;
                    }
                } else {
                    self.paste_start_match = usize::from(byte == PASTE_START[0]);
                }
            }
        }

        fn convert_event(&mut self, event: InputEvent) -> Option<Event> {
            match event {
                InputEvent::Key(key) => convert_key(key).map(Event::Key),
                InputEvent::Mouse(mouse) => self.convert_mouse(mouse),
                InputEvent::PixelMouse(_) => None,
                InputEvent::Resized { cols, rows } => Some(Event::Resize(
                    cols.min(u16::MAX as usize) as u16,
                    rows.min(u16::MAX as usize) as u16,
                )),
                InputEvent::Paste(text) => Some(Event::Paste(text)),
                InputEvent::Wake => None,
            }
        }

        fn convert_mouse(&mut self, event: TermwizMouseEvent) -> Option<Event> {
            let (kind, buttons) = mouse_kind(event.mouse_buttons, self.mouse_buttons.clone());
            self.mouse_buttons = buttons;
            Some(Event::Mouse(MouseEvent {
                kind,
                column: event.x.saturating_sub(1),
                row: event.y.saturating_sub(1),
                modifiers: convert_modifiers(event.modifiers),
            }))
        }
    }

    fn convert_key(event: TermwizKeyEvent) -> Option<KeyEvent> {
        let code = match event.key {
            KeyCode::Char(character) => CrosstermKeyCode::Char(character),
            KeyCode::Backspace => CrosstermKeyCode::Backspace,
            KeyCode::Tab => CrosstermKeyCode::Tab,
            KeyCode::Enter => CrosstermKeyCode::Enter,
            KeyCode::Escape => CrosstermKeyCode::Esc,
            KeyCode::PageUp | KeyCode::KeyPadPageUp => CrosstermKeyCode::PageUp,
            KeyCode::PageDown | KeyCode::KeyPadPageDown => CrosstermKeyCode::PageDown,
            KeyCode::End | KeyCode::KeyPadEnd => CrosstermKeyCode::End,
            KeyCode::Home | KeyCode::KeyPadHome => CrosstermKeyCode::Home,
            KeyCode::LeftArrow => CrosstermKeyCode::Left,
            KeyCode::RightArrow => CrosstermKeyCode::Right,
            KeyCode::UpArrow => CrosstermKeyCode::Up,
            KeyCode::DownArrow | KeyCode::ApplicationDownArrow => CrosstermKeyCode::Down,
            KeyCode::ApplicationLeftArrow => CrosstermKeyCode::Left,
            KeyCode::ApplicationRightArrow => CrosstermKeyCode::Right,
            KeyCode::ApplicationUpArrow => CrosstermKeyCode::Up,
            KeyCode::Insert => CrosstermKeyCode::Insert,
            KeyCode::Delete => CrosstermKeyCode::Delete,
            KeyCode::Function(number) if (1..=24).contains(&number) => CrosstermKeyCode::F(number),
            KeyCode::CapsLock => CrosstermKeyCode::CapsLock,
            KeyCode::ScrollLock => CrosstermKeyCode::ScrollLock,
            KeyCode::NumLock => CrosstermKeyCode::NumLock,
            KeyCode::PrintScreen => CrosstermKeyCode::PrintScreen,
            KeyCode::Pause => CrosstermKeyCode::Pause,
            KeyCode::Menu => CrosstermKeyCode::Menu,
            KeyCode::KeyPadBegin => CrosstermKeyCode::KeypadBegin,
            KeyCode::MediaNextTrack => {
                CrosstermKeyCode::Media(crossterm::event::MediaKeyCode::TrackNext)
            }
            KeyCode::MediaPrevTrack => {
                CrosstermKeyCode::Media(crossterm::event::MediaKeyCode::TrackPrevious)
            }
            KeyCode::MediaStop => CrosstermKeyCode::Media(crossterm::event::MediaKeyCode::Stop),
            KeyCode::MediaPlayPause => {
                CrosstermKeyCode::Media(crossterm::event::MediaKeyCode::PlayPause)
            }
            KeyCode::VolumeMute => {
                CrosstermKeyCode::Media(crossterm::event::MediaKeyCode::MuteVolume)
            }
            KeyCode::VolumeDown => {
                CrosstermKeyCode::Media(crossterm::event::MediaKeyCode::LowerVolume)
            }
            KeyCode::VolumeUp => {
                CrosstermKeyCode::Media(crossterm::event::MediaKeyCode::RaiseVolume)
            }
            KeyCode::LeftShift => {
                CrosstermKeyCode::Modifier(crossterm::event::ModifierKeyCode::LeftShift)
            }
            KeyCode::RightShift => {
                CrosstermKeyCode::Modifier(crossterm::event::ModifierKeyCode::RightShift)
            }
            KeyCode::LeftControl => {
                CrosstermKeyCode::Modifier(crossterm::event::ModifierKeyCode::LeftControl)
            }
            KeyCode::RightControl => {
                CrosstermKeyCode::Modifier(crossterm::event::ModifierKeyCode::RightControl)
            }
            KeyCode::LeftAlt | KeyCode::LeftMenu => {
                CrosstermKeyCode::Modifier(crossterm::event::ModifierKeyCode::LeftAlt)
            }
            KeyCode::RightAlt | KeyCode::RightMenu => {
                CrosstermKeyCode::Modifier(crossterm::event::ModifierKeyCode::RightAlt)
            }
            KeyCode::LeftWindows => {
                CrosstermKeyCode::Modifier(crossterm::event::ModifierKeyCode::LeftSuper)
            }
            KeyCode::RightWindows => {
                CrosstermKeyCode::Modifier(crossterm::event::ModifierKeyCode::RightSuper)
            }
            KeyCode::Shift => {
                CrosstermKeyCode::Modifier(crossterm::event::ModifierKeyCode::LeftShift)
            }
            KeyCode::Control => {
                CrosstermKeyCode::Modifier(crossterm::event::ModifierKeyCode::LeftControl)
            }
            KeyCode::Alt => CrosstermKeyCode::Modifier(crossterm::event::ModifierKeyCode::LeftAlt),
            KeyCode::Super => {
                CrosstermKeyCode::Modifier(crossterm::event::ModifierKeyCode::LeftSuper)
            }
            KeyCode::Hyper => {
                CrosstermKeyCode::Modifier(crossterm::event::ModifierKeyCode::LeftHyper)
            }
            KeyCode::Meta => {
                CrosstermKeyCode::Modifier(crossterm::event::ModifierKeyCode::LeftMeta)
            }
            KeyCode::Clear
            | KeyCode::Cancel
            | KeyCode::Select
            | KeyCode::Print
            | KeyCode::Execute
            | KeyCode::Help
            | KeyCode::Applications
            | KeyCode::Sleep
            | KeyCode::Numpad0
            | KeyCode::Numpad1
            | KeyCode::Numpad2
            | KeyCode::Numpad3
            | KeyCode::Numpad4
            | KeyCode::Numpad5
            | KeyCode::Numpad6
            | KeyCode::Numpad7
            | KeyCode::Numpad8
            | KeyCode::Numpad9
            | KeyCode::Multiply
            | KeyCode::Add
            | KeyCode::Separator
            | KeyCode::Subtract
            | KeyCode::Decimal
            | KeyCode::Divide
            | KeyCode::BrowserBack
            | KeyCode::BrowserForward
            | KeyCode::BrowserRefresh
            | KeyCode::BrowserStop
            | KeyCode::BrowserSearch
            | KeyCode::BrowserFavorites
            | KeyCode::BrowserHome
            | KeyCode::Copy
            | KeyCode::Cut
            | KeyCode::Paste
            | KeyCode::InternalPasteStart
            | KeyCode::InternalPasteEnd
            | KeyCode::Function(_) => return None,
        };

        let modifiers = convert_modifiers(event.modifiers);
        let code = if code == CrosstermKeyCode::Tab && modifiers.contains(KeyModifiers::SHIFT) {
            CrosstermKeyCode::BackTab
        } else {
            code
        };
        Some(KeyEvent {
            code,
            modifiers,
            kind: KeyEventKind::Press,
            state: KeyEventState::empty(),
        })
    }

    fn convert_modifiers(modifiers: Modifiers) -> KeyModifiers {
        let mut result = KeyModifiers::empty();
        if modifiers.contains(Modifiers::SHIFT) {
            result.insert(KeyModifiers::SHIFT);
        }
        if modifiers.contains(Modifiers::CTRL) {
            result.insert(KeyModifiers::CONTROL);
        }
        if modifiers.contains(Modifiers::ALT) {
            result.insert(KeyModifiers::ALT);
        }
        if modifiers.contains(Modifiers::SUPER) {
            result.insert(KeyModifiers::SUPER);
        }
        result
    }

    fn mouse_kind(buttons: MouseButtons, previous: MouseButtons) -> (MouseEventKind, MouseButtons) {
        if buttons.contains(MouseButtons::VERT_WHEEL) {
            return (
                if buttons.contains(MouseButtons::WHEEL_POSITIVE) {
                    MouseEventKind::ScrollUp
                } else {
                    MouseEventKind::ScrollDown
                },
                previous,
            );
        }
        if buttons.contains(MouseButtons::HORZ_WHEEL) {
            return (
                if buttons.contains(MouseButtons::WHEEL_POSITIVE) {
                    MouseEventKind::ScrollRight
                } else {
                    MouseEventKind::ScrollLeft
                },
                previous,
            );
        }

        let button = if buttons.contains(MouseButtons::LEFT) {
            Some(MouseButton::Left)
        } else if buttons.contains(MouseButtons::RIGHT) {
            Some(MouseButton::Right)
        } else if buttons.contains(MouseButtons::MIDDLE) {
            Some(MouseButton::Middle)
        } else {
            None
        };
        let previous_button = if previous.contains(MouseButtons::LEFT) {
            Some(MouseButton::Left)
        } else if previous.contains(MouseButtons::RIGHT) {
            Some(MouseButton::Right)
        } else if previous.contains(MouseButtons::MIDDLE) {
            Some(MouseButton::Middle)
        } else {
            None
        };

        match (button, previous_button) {
            (Some(button), None) => (MouseEventKind::Down(button), buttons),
            (Some(button), Some(previous)) if button == previous => {
                (MouseEventKind::Drag(button), buttons)
            }
            (Some(button), Some(_)) => (MouseEventKind::Down(button), buttons),
            (None, Some(previous)) => (MouseEventKind::Up(previous), MouseButtons::NONE),
            (None, None) => (MouseEventKind::Moved, MouseButtons::NONE),
        }
    }

    #[cfg(test)]
    pub(crate) fn parse_vt_chunks(chunks: &[&[u8]]) -> Vec<Event> {
        let mut parser = Parser::new();
        let mut events = Vec::new();
        for chunk in chunks {
            events.extend(parser.feed(chunk, true));
        }
        events.extend(parser.flush());
        events
    }

    #[cfg(test)]
    pub(crate) fn parser() -> Parser {
        Parser::new()
    }

    #[cfg(any(test, windows))]
    #[derive(Default)]
    pub(crate) struct Utf16Decoder {
        high_surrogate: Option<u16>,
    }

    #[cfg(any(test, windows))]
    impl Utf16Decoder {
        #[allow(dead_code)]
        pub(crate) fn push(&mut self, unit: u16) -> Vec<u8> {
            let mut bytes = Vec::new();
            self.push_into(unit, &mut bytes);
            bytes
        }

        pub(crate) fn push_into(&mut self, unit: u16, bytes: &mut Vec<u8>) {
            if let Some(high) = self.high_surrogate.take() {
                if (0xdc00..=0xdfff).contains(&unit) {
                    let scalar =
                        0x1_0000 + (((high as u32 - 0xd800) << 10) | (unit as u32 - 0xdc00));
                    encode_char(char::from_u32(scalar).expect("valid UTF-16 pair"), bytes);
                    return;
                }
                encode_char('\u{fffd}', bytes);
            }

            if (0xd800..=0xdbff).contains(&unit) {
                self.high_surrogate = Some(unit);
            } else if let Some(character) = char::from_u32(unit as u32) {
                encode_char(character, bytes);
            } else {
                encode_char('\u{fffd}', bytes);
            }
        }

        #[allow(dead_code)]
        pub(crate) fn finish(&mut self) -> Vec<u8> {
            let mut bytes = Vec::new();
            self.finish_into(&mut bytes);
            bytes
        }

        pub(crate) fn finish_into(&mut self, bytes: &mut Vec<u8>) {
            if self.high_surrogate.take().is_some() {
                bytes.extend_from_slice("\u{fffd}".as_bytes());
            }
        }
    }

    #[cfg(any(test, windows))]
    fn encode_char(character: char, bytes: &mut Vec<u8>) {
        let mut buffer = [0; 4];
        bytes.extend_from_slice(character.encode_utf8(&mut buffer).as_bytes());
    }
}

#[cfg(windows)]
mod windows_native {
    use super::*;
    use crate::tui::terminal_input::vt::{Parser, Utf16Decoder};
    use crossterm::event::{
        KeyCode, KeyEvent, KeyEventKind, KeyEventState, KeyModifiers, ModifierKeyCode, MouseButton,
        MouseEvent, MouseEventKind,
    };
    use windows_sys::Win32::Foundation::{
        GetLastError, HANDLE, INVALID_HANDLE_VALUE, WAIT_FAILED, WAIT_OBJECT_0, WAIT_TIMEOUT,
    };
    use windows_sys::Win32::System::Console::{
        CONSOLE_MODE, CONSOLE_SCREEN_BUFFER_INFO, ENABLE_VIRTUAL_TERMINAL_INPUT,
        FROM_LEFT_1ST_BUTTON_PRESSED, FROM_LEFT_2ND_BUTTON_PRESSED, GetConsoleMode,
        GetConsoleScreenBufferInfo, GetStdHandle, INPUT_RECORD, KEY_EVENT, KEY_EVENT_RECORD,
        LEFT_ALT_PRESSED, LEFT_CTRL_PRESSED, MOUSE_EVENT, MOUSE_EVENT_RECORD, MOUSE_HWHEELED,
        MOUSE_MOVED, MOUSE_WHEELED, RIGHT_ALT_PRESSED, RIGHT_CTRL_PRESSED,
        RIGHTMOST_BUTTON_PRESSED, ReadConsoleInputW, SHIFT_PRESSED, STD_INPUT_HANDLE,
        STD_OUTPUT_HANDLE, SetConsoleMode, WINDOW_BUFFER_SIZE_EVENT,
    };
    #[cfg(test)]
    use windows_sys::Win32::System::Console::{
        COORD, INPUT_RECORD_0, KEY_EVENT_RECORD_0, WINDOW_BUFFER_SIZE_RECORD,
    };
    use windows_sys::Win32::System::Threading::WaitForSingleObject;
    use windows_sys::Win32::UI::Input::KeyboardAndMouse::*;

    const MAX_RECORDS: usize = 4096;
    const LEFT_BUTTON: u8 = 1;
    const RIGHT_BUTTON: u8 = 2;
    const MIDDLE_BUTTON: u8 = 4;

    #[derive(Clone, Copy)]
    enum RawSequence {
        None,
        Escape,
        Csi,
        Ss3,
    }

    #[derive(Clone, Copy)]
    struct NativeSurrogate {
        high: u16,
        virtual_key: u16,
        modifiers: KeyModifiers,
        repeat: u16,
    }

    pub(crate) struct NativeInput {
        handle: HANDLE,
        output_handle: HANDLE,
        original_mode: CONSOLE_MODE,
        parser: Parser,
        raw_utf16: Utf16Decoder,
        native_surrogate: Option<NativeSurrogate>,
        pending: VecDeque<Event>,
        mouse_buttons: u8,
        raw_sequence: RawSequence,
        window_origin: (i16, i16),
        window_size: (u16, u16),
    }

    impl NativeInput {
        pub(crate) fn new() -> io::Result<Self> {
            let handle = unsafe { GetStdHandle(STD_INPUT_HANDLE) };
            if handle.is_null() || handle == INVALID_HANDLE_VALUE {
                return Err(io::Error::last_os_error());
            }

            let mut original_mode = 0;
            if unsafe { GetConsoleMode(handle, &mut original_mode) } == 0 {
                return Err(io::Error::last_os_error());
            }
            if unsafe { SetConsoleMode(handle, original_mode | ENABLE_VIRTUAL_TERMINAL_INPUT) } == 0
            {
                return Err(io::Error::last_os_error());
            }

            let output_handle = unsafe { GetStdHandle(STD_OUTPUT_HANDLE) };
            let (window_origin, window_size) =
                console_window_metrics(output_handle).unwrap_or(((0, 0), (0, 0)));

            Ok(Self {
                handle,
                output_handle,
                original_mode,
                parser: Parser::new(),
                raw_utf16: Utf16Decoder::default(),
                native_surrogate: None,
                pending: VecDeque::new(),
                mouse_buttons: 0,
                raw_sequence: RawSequence::None,
                window_origin,
                window_size,
            })
        }

        pub(crate) fn read(&mut self, timeout: Duration) -> io::Result<Option<Event>> {
            let deadline = Instant::now()
                .checked_add(timeout)
                .unwrap_or_else(Instant::now);

            loop {
                if let Some(event) = self.pending.pop_front() {
                    return Ok(Some(event));
                }

                let remaining = deadline.saturating_duration_since(Instant::now());
                let wait_ms = remaining.as_millis().min(u32::MAX as u128) as u32;
                let wait_result = unsafe { WaitForSingleObject(self.handle, wait_ms) };
                match wait_result {
                    WAIT_OBJECT_0 => {}
                    WAIT_TIMEOUT => {
                        if !timeout.is_zero() && matches!(self.raw_sequence, RawSequence::Escape) {
                            self.pending.extend(self.parser.flush());
                            self.raw_sequence = RawSequence::None;
                        }
                        return Ok(self.pending.pop_front());
                    }
                    WAIT_FAILED => {
                        return Err(io::Error::from_raw_os_error(
                            unsafe { GetLastError() } as i32
                        ));
                    }
                    _ => {
                        return Err(io::Error::new(
                            io::ErrorKind::Other,
                            "unexpected console wait result",
                        ));
                    }
                }

                let mut records = vec![INPUT_RECORD::default(); MAX_RECORDS];
                let mut count = 0;
                if unsafe {
                    ReadConsoleInputW(
                        self.handle,
                        records.as_mut_ptr(),
                        records.len() as u32,
                        &mut count,
                    )
                } == 0
                {
                    return Err(io::Error::last_os_error());
                }
                self.process_records(&records[..count as usize]);
                if timeout.is_zero() || Instant::now() >= deadline {
                    return Ok(self.pending.pop_front());
                }
            }
        }

        fn process_records(&mut self, records: &[INPUT_RECORD]) {
            let mut raw_bytes = Vec::new();
            for record in records {
                match record.EventType as u32 {
                    KEY_EVENT => unsafe {
                        self.process_key(&record.Event.KeyEvent, &mut raw_bytes)
                    },
                    MOUSE_EVENT => unsafe {
                        self.finish_raw_surrogate(&mut raw_bytes);
                        self.flush_raw(&mut raw_bytes);
                        self.finish_native_surrogate();
                        self.process_mouse(&record.Event.MouseEvent);
                    },
                    WINDOW_BUFFER_SIZE_EVENT => {
                        self.finish_raw_surrogate(&mut raw_bytes);
                        self.flush_raw(&mut raw_bytes);
                        self.finish_native_surrogate();
                        self.refresh_window_metrics();
                        if self.window_size != (0, 0) {
                            self.pending
                                .push_back(Event::Resize(self.window_size.0, self.window_size.1));
                        }
                    }
                    _ => {}
                }
            }
            self.flush_raw(&mut raw_bytes);
        }

        fn finish_raw_surrogate(&mut self, raw_bytes: &mut Vec<u8>) {
            let start = raw_bytes.len();
            self.raw_utf16.finish_into(raw_bytes);
            self.parser.observe_bytes(&raw_bytes[start..]);
        }

        fn flush_raw(&mut self, raw_bytes: &mut Vec<u8>) {
            if raw_bytes.is_empty() {
                return;
            }
            self.pending
                .extend(self.parser.feed_observed(raw_bytes, true));
            raw_bytes.clear();
        }

        unsafe fn process_key(&mut self, record: &KEY_EVENT_RECORD, raw_bytes: &mut Vec<u8>) {
            let unicode = unsafe { record.uChar.UnicodeChar };
            // Windows commits Alt+numpad characters on release of Alt itself.
            if record.bKeyDown == 0 {
                if matches!(record.wVirtualKeyCode, VK_MENU | VK_LMENU | VK_RMENU) && unicode != 0 {
                    let mut committed = *record;
                    committed.dwControlKeyState &= !(LEFT_ALT_PRESSED | RIGHT_ALT_PRESSED);
                    self.process_native_unicode(&committed, unicode);
                }
                return;
            }
            // VT input is represented by character-only records. Physical keys
            // retain their virtual key, including Shift+Enter and Ctrl+V.
            let raw_record = unicode != 0
                && ((record.wVirtualKeyCode == 0 && record.wVirtualScanCode == 0)
                    || self.parser.paste_in_progress()
                    || !matches!(self.raw_sequence, RawSequence::None));

            if raw_record {
                self.finish_native_surrogate();
                if unicode != 0 {
                    let start = raw_bytes.len();
                    for _ in 0..record.wRepeatCount.max(1) {
                        self.raw_utf16.push_into(unicode, raw_bytes);
                    }
                    let bytes = &raw_bytes[start..];
                    self.parser.observe_bytes(bytes);
                    self.advance_raw_sequence(bytes);
                }
                return;
            }

            self.finish_raw_surrogate(raw_bytes);
            self.flush_raw(raw_bytes);
            if unicode != 0 {
                if record.bKeyDown != 0 {
                    self.process_native_unicode(record, unicode);
                }
                return;
            }

            self.finish_native_surrogate();
            let Some(code) = native_key_code(record.wVirtualKeyCode) else {
                return;
            };
            let modifiers = native_modifiers(record.dwControlKeyState);
            let code = if code == KeyCode::Tab && modifiers.contains(KeyModifiers::SHIFT) {
                KeyCode::BackTab
            } else {
                code
            };
            let kind = if record.bKeyDown != 0 {
                KeyEventKind::Press
            } else {
                KeyEventKind::Release
            };
            let repeats = if record.bKeyDown != 0 {
                record.wRepeatCount.max(1)
            } else {
                1
            };
            for _ in 0..repeats {
                self.pending.push_back(Event::Key(KeyEvent {
                    code,
                    modifiers,
                    kind,
                    state: KeyEventState::empty(),
                }));
            }
        }

        fn advance_raw_sequence(&mut self, bytes: &[u8]) {
            for &byte in bytes {
                self.raw_sequence = match self.raw_sequence {
                    RawSequence::None if byte == 0x1b => RawSequence::Escape,
                    RawSequence::Escape if byte == b'[' => RawSequence::Csi,
                    RawSequence::Escape if byte == b'O' => RawSequence::Ss3,
                    RawSequence::Escape => RawSequence::None,
                    RawSequence::Csi if (0x40..=0x7e).contains(&byte) => RawSequence::None,
                    RawSequence::Csi => RawSequence::Csi,
                    RawSequence::Ss3 => RawSequence::None,
                    RawSequence::None => RawSequence::None,
                };
            }
        }

        fn process_native_unicode(&mut self, record: &KEY_EVENT_RECORD, unit: u16) {
            let mut modifiers = native_modifiers(record.dwControlKeyState);
            // AltGr is reported as right Alt + left Ctrl, even when it produces text.
            if unit >= 0x20
                && record.dwControlKeyState & (RIGHT_ALT_PRESSED | LEFT_CTRL_PRESSED)
                    == (RIGHT_ALT_PRESSED | LEFT_CTRL_PRESSED)
            {
                modifiers.remove(KeyModifiers::CONTROL | KeyModifiers::ALT);
            }
            let repeat = record.wRepeatCount.max(1);
            if let Some(previous) = self.native_surrogate.take() {
                if (0xdc00..=0xdfff).contains(&unit) {
                    let scalar = 0x1_0000
                        + (((previous.high as u32 - 0xd800) << 10) | (unit as u32 - 0xdc00));
                    self.emit_native_char(
                        record.wVirtualKeyCode,
                        char::from_u32(scalar).unwrap_or('\u{fffd}'),
                        previous.modifiers,
                        previous.repeat,
                    );
                    return;
                }
                self.emit_native_char(
                    previous.virtual_key,
                    '\u{fffd}',
                    previous.modifiers,
                    previous.repeat,
                );
            }
            if (0xd800..=0xdbff).contains(&unit) {
                self.native_surrogate = Some(NativeSurrogate {
                    high: unit,
                    virtual_key: record.wVirtualKeyCode,
                    modifiers,
                    repeat,
                });
            } else {
                self.emit_native_char(
                    record.wVirtualKeyCode,
                    char::from_u32(unit as u32).unwrap_or('\u{fffd}'),
                    modifiers,
                    repeat,
                );
            }
        }

        fn finish_native_surrogate(&mut self) {
            if let Some(previous) = self.native_surrogate.take() {
                self.emit_native_char(
                    previous.virtual_key,
                    '\u{fffd}',
                    previous.modifiers,
                    previous.repeat,
                );
            }
        }

        fn emit_native_char(
            &mut self,
            virtual_key: u16,
            character: char,
            modifiers: KeyModifiers,
            repeat: u16,
        ) {
            let code = native_unicode_key_code(virtual_key, character);
            let code = if code == KeyCode::Tab && modifiers.contains(KeyModifiers::SHIFT) {
                KeyCode::BackTab
            } else {
                code
            };
            for _ in 0..repeat {
                self.pending.push_back(Event::Key(KeyEvent {
                    code,
                    modifiers,
                    kind: KeyEventKind::Press,
                    state: KeyEventState::empty(),
                }));
            }
        }

        unsafe fn process_mouse(&mut self, record: &MOUSE_EVENT_RECORD) {
            self.refresh_window_metrics();
            let current = native_mouse_buttons(record.dwButtonState);
            let modifiers = native_modifiers(record.dwControlKeyState);
            let position = record.dwMousePosition;

            if record.dwEventFlags & MOUSE_WHEELED != 0 {
                let delta = (record.dwButtonState >> 16) as i16;
                self.pending.push_back(Event::Mouse(MouseEvent {
                    kind: if delta >= 0 {
                        MouseEventKind::ScrollUp
                    } else {
                        MouseEventKind::ScrollDown
                    },
                    column: self.relative_x(position.X),
                    row: self.relative_y(position.Y),
                    modifiers,
                }));
            } else if record.dwEventFlags & MOUSE_HWHEELED != 0 {
                let delta = (record.dwButtonState >> 16) as i16;
                self.pending.push_back(Event::Mouse(MouseEvent {
                    kind: if delta >= 0 {
                        MouseEventKind::ScrollRight
                    } else {
                        MouseEventKind::ScrollLeft
                    },
                    column: self.relative_x(position.X),
                    row: self.relative_y(position.Y),
                    modifiers,
                }));
            } else if record.dwEventFlags & MOUSE_MOVED != 0 {
                self.pending.push_back(Event::Mouse(MouseEvent {
                    kind: if current == 0 {
                        MouseEventKind::Moved
                    } else {
                        mouse_button(current)
                            .map(MouseEventKind::Drag)
                            .unwrap_or(MouseEventKind::Moved)
                    },
                    column: self.relative_x(position.X),
                    row: self.relative_y(position.Y),
                    modifiers,
                }));
            } else {
                let changed = self.mouse_buttons ^ current;
                for mask in [LEFT_BUTTON, RIGHT_BUTTON, MIDDLE_BUTTON] {
                    if changed & mask != 0 {
                        self.pending.push_back(Event::Mouse(MouseEvent {
                            kind: if current & mask != 0 {
                                MouseEventKind::Down(mouse_button(mask).unwrap())
                            } else {
                                MouseEventKind::Up(mouse_button(mask).unwrap())
                            },
                            column: self.relative_x(position.X),
                            row: self.relative_y(position.Y),
                            modifiers,
                        }));
                    }
                }
            }
            self.mouse_buttons = current;
        }

        fn refresh_window_metrics(&mut self) {
            if let Some((origin, size)) = console_window_metrics(self.output_handle) {
                self.window_origin = origin;
                self.window_size = size;
            }
        }

        fn relative_x(&self, x: i16) -> u16 {
            x.saturating_sub(self.window_origin.0).max(0) as u16
        }

        fn relative_y(&self, y: i16) -> u16 {
            y.saturating_sub(self.window_origin.1).max(0) as u16
        }
    }

    impl Drop for NativeInput {
        fn drop(&mut self) {
            if unsafe { SetConsoleMode(self.handle, self.original_mode) } == 0 {
                tracing::error!(
                    error = %io::Error::last_os_error(),
                    "failed to restore Windows console input mode"
                );
            }
        }
    }

    fn console_window_metrics(handle: HANDLE) -> Option<((i16, i16), (u16, u16))> {
        if handle.is_null() || handle == INVALID_HANDLE_VALUE {
            return None;
        }
        let mut info = CONSOLE_SCREEN_BUFFER_INFO::default();
        if unsafe { GetConsoleScreenBufferInfo(handle, &mut info) } == 0 {
            return None;
        }
        let width = info.srWindow.Right.saturating_sub(info.srWindow.Left) as u16 + 1;
        let height = info.srWindow.Bottom.saturating_sub(info.srWindow.Top) as u16 + 1;
        Some(((info.srWindow.Left, info.srWindow.Top), (width, height)))
    }

    fn native_modifiers(state: u32) -> KeyModifiers {
        let mut modifiers = KeyModifiers::empty();
        if state & SHIFT_PRESSED != 0 {
            modifiers.insert(KeyModifiers::SHIFT);
        }
        if state & (LEFT_CTRL_PRESSED | RIGHT_CTRL_PRESSED) != 0 {
            modifiers.insert(KeyModifiers::CONTROL);
        }
        if state & (LEFT_ALT_PRESSED | RIGHT_ALT_PRESSED) != 0 {
            modifiers.insert(KeyModifiers::ALT);
        }
        modifiers
    }

    fn native_unicode_key_code(virtual_key: u16, character: char) -> KeyCode {
        match native_key_code(virtual_key) {
            Some(KeyCode::Enter)
            | Some(KeyCode::Tab)
            | Some(KeyCode::Backspace)
            | Some(KeyCode::Esc) => native_key_code(virtual_key).unwrap(),
            _ if ('\u{1}'..='\u{1a}').contains(&character) => {
                KeyCode::Char((b'a' + character as u8 - 1) as char)
            }
            _ => KeyCode::Char(character),
        }
    }

    fn native_mouse_buttons(state: u32) -> u8 {
        let mut buttons = 0;
        if state & FROM_LEFT_1ST_BUTTON_PRESSED != 0 {
            buttons |= LEFT_BUTTON;
        }
        if state & RIGHTMOST_BUTTON_PRESSED != 0 {
            buttons |= RIGHT_BUTTON;
        }
        if state & FROM_LEFT_2ND_BUTTON_PRESSED != 0 {
            buttons |= MIDDLE_BUTTON;
        }
        buttons
    }

    fn mouse_button(mask: u8) -> Option<MouseButton> {
        if mask & LEFT_BUTTON != 0 {
            Some(MouseButton::Left)
        } else if mask & RIGHT_BUTTON != 0 {
            Some(MouseButton::Right)
        } else if mask & MIDDLE_BUTTON != 0 {
            Some(MouseButton::Middle)
        } else {
            None
        }
    }

    #[cfg(test)]
    mod tests {
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
    }

    fn native_key_code(virtual_key: u16) -> Option<KeyCode> {
        let code = match virtual_key {
            VK_BACK => KeyCode::Backspace,
            VK_TAB => KeyCode::Tab,
            VK_RETURN => KeyCode::Enter,
            VK_ESCAPE => KeyCode::Esc,
            VK_PRIOR => KeyCode::PageUp,
            VK_NEXT => KeyCode::PageDown,
            VK_END => KeyCode::End,
            VK_HOME => KeyCode::Home,
            VK_LEFT => KeyCode::Left,
            VK_RIGHT => KeyCode::Right,
            VK_UP => KeyCode::Up,
            VK_DOWN => KeyCode::Down,
            VK_INSERT => KeyCode::Insert,
            VK_DELETE => KeyCode::Delete,
            VK_CAPITAL => KeyCode::CapsLock,
            VK_SCROLL => KeyCode::ScrollLock,
            VK_NUMLOCK => KeyCode::NumLock,
            VK_SNAPSHOT => KeyCode::PrintScreen,
            VK_PAUSE => KeyCode::Pause,
            VK_APPS => KeyCode::Menu,
            VK_LSHIFT => KeyCode::Modifier(ModifierKeyCode::LeftShift),
            VK_RSHIFT => KeyCode::Modifier(ModifierKeyCode::RightShift),
            VK_LCONTROL => KeyCode::Modifier(ModifierKeyCode::LeftControl),
            VK_RCONTROL => KeyCode::Modifier(ModifierKeyCode::RightControl),
            VK_LMENU => KeyCode::Modifier(ModifierKeyCode::LeftAlt),
            VK_RMENU => KeyCode::Modifier(ModifierKeyCode::RightAlt),
            VK_LWIN => KeyCode::Modifier(ModifierKeyCode::LeftSuper),
            VK_RWIN => KeyCode::Modifier(ModifierKeyCode::RightSuper),
            VK_F1..=VK_F24 => KeyCode::F((virtual_key - VK_F1 + 1) as u8),
            VK_NUMPAD0..=VK_NUMPAD9 => {
                KeyCode::Char((b'0' + (virtual_key - VK_NUMPAD0) as u8) as char)
            }
            VK_MULTIPLY => KeyCode::Char('*'),
            VK_ADD => KeyCode::Char('+'),
            VK_SUBTRACT => KeyCode::Char('-'),
            VK_DECIMAL => KeyCode::Char('.'),
            VK_DIVIDE => KeyCode::Char('/'),
            VK_BROWSER_BACK => KeyCode::Media(crossterm::event::MediaKeyCode::TrackPrevious),
            VK_BROWSER_FORWARD => KeyCode::Media(crossterm::event::MediaKeyCode::TrackNext),
            VK_VOLUME_MUTE => KeyCode::Media(crossterm::event::MediaKeyCode::MuteVolume),
            VK_VOLUME_DOWN => KeyCode::Media(crossterm::event::MediaKeyCode::LowerVolume),
            VK_VOLUME_UP => KeyCode::Media(crossterm::event::MediaKeyCode::RaiseVolume),
            VK_MEDIA_NEXT_TRACK => KeyCode::Media(crossterm::event::MediaKeyCode::TrackNext),
            VK_MEDIA_PREV_TRACK => KeyCode::Media(crossterm::event::MediaKeyCode::TrackPrevious),
            VK_MEDIA_STOP => KeyCode::Media(crossterm::event::MediaKeyCode::Stop),
            VK_MEDIA_PLAY_PAUSE => KeyCode::Media(crossterm::event::MediaKeyCode::PlayPause),
            _ => return None,
        };
        Some(code)
    }
}

pub struct TerminalInput {
    #[cfg(windows)]
    native: windows_native::NativeInput,
}

impl TerminalInput {
    pub fn new() -> io::Result<Self> {
        #[cfg(windows)]
        {
            return Ok(Self {
                native: windows_native::NativeInput::new()?,
            });
        }

        #[cfg(not(windows))]
        {
            Ok(Self {})
        }
    }

    pub fn read(&mut self, timeout: Duration) -> io::Result<Option<Event>> {
        #[cfg(windows)]
        {
            return self.native.read(timeout);
        }

        #[cfg(not(windows))]
        {
            if crossterm::event::poll(timeout)? {
                crossterm::event::read().map(Some)
            } else {
                Ok(None)
            }
        }
    }
}

#[cfg(test)]
mod tests {
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
        let events =
            vt::parse_vt_chunks(&[b"\x1b[A\x1b[1;5C\x1b[<0;4;5M\x1b[<32;5;6M\x1b[<3;5;6m"]);
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
    fn utf16_surrogate_pairs_are_encoded_across_input_chunks() {
        let mut decoder = vt::Utf16Decoder::default();
        assert!(decoder.push(0xd83d).is_empty());
        assert_eq!(decoder.push(0xde42), "🙂".as_bytes());
    }
}
