use super::*;
use ratatui::{Terminal, backend::TestBackend};

fn draw_panel(state: &TuiState, width: u16, height: u16) -> String {
    let backend = TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend).expect("terminal");

    terminal
        .draw(|frame| {
            render_slash_panel(frame, state, Rect::new(0, 0, width, height), Theme::dark())
        })
        .expect("draw");

    terminal
        .backend()
        .buffer()
        .content()
        .iter()
        .map(|cell| cell.symbol())
        .collect()
}

#[test]
fn completion_panel_is_hidden_in_child_read_only_view() {
    let mut state = TuiState::default();
    state.set_input("/fa");
    let parent = draw_panel(&state, 72, 6);
    assert!(parent.contains("/fast"), "{parent}");

    state.replace_child_timeline_from_records(
        &[],
        "parent-session",
        "child-session",
        "explorer",
        0,
        1,
        1,
    );
    state.set_input("/fa");

    let rendered = draw_panel(&state, 72, 6);
    assert!(!rendered.contains("/fast"), "{rendered}");
}
