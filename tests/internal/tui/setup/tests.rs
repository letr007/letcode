use super::*;
use ratatui::{Terminal, backend::TestBackend};

fn draw(setup: &SetupState, width: u16, height: u16) -> String {
    let mut state = TuiState::new("pending", "pending", "default");
    state.set_language(Some(crate::tui::i18n::Language::En));
    let backend = TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend).expect("terminal");
    terminal
        .draw(|frame| render_setup(frame, &state, setup))
        .expect("render");
    terminal
        .backend()
        .buffer()
        .content()
        .iter()
        .map(|cell| cell.symbol())
        .collect()
}

#[test]
fn setup_screen_renders_fields_and_masks_api_key() {
    let setup = SetupState {
        api_key: "secret-key".into(),
        model: "gpt-5.5".into(),
        ..SetupState::default()
    };
    let rendered = draw(&setup, 100, 30);

    assert!(rendered.contains("Set up letcode"), "{rendered}");
    assert!(rendered.contains("Base URL"), "{rendered}");
    assert!(rendered.contains("API key"), "{rendered}");
    assert!(rendered.contains("gpt-5.5"), "{rendered}");
    assert!(rendered.contains("••••••••••"), "{rendered}");
    assert!(!rendered.contains(surface::ACCENT_BAR_GLYPH), "{rendered}");
    assert!(!rendered.contains("secret-key"), "{rendered}");
}

#[test]
fn setup_screen_has_a_compact_fallback() {
    let setup = SetupState {
        active_field: SetupField::ApiKey,
        api_key: "secret-key".into(),
        ..SetupState::default()
    };
    let rendered = draw(&setup, 32, 10);
    assert!(rendered.contains("Set up letcode"), "{rendered}");
    assert!(rendered.contains("Resize"), "{rendered}");
    assert!(rendered.contains("••••••••••"), "{rendered}");
    assert!(!rendered.contains(surface::ACCENT_BAR_GLYPH), "{rendered}");
    assert!(!rendered.contains("secret-key"), "{rendered}");
}
