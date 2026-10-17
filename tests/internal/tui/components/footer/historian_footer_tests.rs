use super::*;
fn text(state: &TuiState) -> String {
    footer_hint_spans(state, crate::tui::Theme::dark(), 80)
        .iter()
        .map(|s| s.content.as_ref())
        .collect()
}
#[test]
fn background_icon_does_not_enable_the_blocking_animation() {
    let mut state = TuiState::default();
    state.session_id = Some("s".into());
    state.historian_status = Some(("s".into(), true, false));
    assert!(text(&state).contains("󰁯"));
    assert!(state.active_compaction_animation_start_frame().is_none());
    state.compaction_active = true;
    assert!(!text(&state).contains("󰁯"));
    assert!(state.active_compaction_animation_start_frame().is_some());
    assert!(text(&state).contains(&state.t("runtime.compacting_context")));
    state.compaction_active = false;
    state.session_id = Some("another-session".into());
    assert!(!text(&state).contains("󰁯"));
}
