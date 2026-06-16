use crossterm::event::{KeyCode, KeyEvent, KeyModifiers, MouseEvent, MouseEventKind};

use super::state::{DialogKind, TuiState};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InputAction {
    Insert(char),
    Backspace,
    DialogInsert(char),
    DialogBackspace,
    DialogNext,
    DialogPrev,
    DialogAccept,
    DialogCancel,
    SlashPanelNext,
    SlashPanelPrev,
    SlashPanelAccept,
    SlashPanelDismiss,
    Submit,
    ScrollUp,
    ScrollDown,
    ScrollPageUp,
    ScrollPageDown,
    ScrollToBottom,
    MouseScrollUp,
    MouseScrollDown,
    MouseClick,
    CycleReasoningEffort,
    ChildPrefix,
    ChildNext,
    ChildPrev,
    ChildParent,
    ApprovePermission,
    DenyPermission,
    Interrupt,
    Quit,
    Tick,
    NoOp,
}

pub fn map_key_event(state: &TuiState, key: KeyEvent) -> InputAction {
    if key.modifiers.contains(KeyModifiers::CONTROL) && matches!(key.code, KeyCode::Char('c')) {
        return InputAction::Quit;
    }

    if key.modifiers.contains(KeyModifiers::CONTROL) && matches!(key.code, KeyCode::Char('t')) {
        return InputAction::CycleReasoningEffort;
    }

    if (key.modifiers.contains(KeyModifiers::CONTROL) && matches!(key.code, KeyCode::Char('x')))
        || matches!(key.code, KeyCode::Char(''))
    {
        return InputAction::ChildPrefix;
    }

    if state.is_read_only_child_view() && !has_non_shift_modifiers(key.modifiers) {
        match key.code {
            KeyCode::Up => return InputAction::ChildParent,
            KeyCode::Left => return InputAction::ChildPrev,
            KeyCode::Right => return InputAction::ChildNext,
            KeyCode::Char('h') | KeyCode::Char('H') => return InputAction::ChildPrev,
            KeyCode::Char('j') | KeyCode::Char('J') => return InputAction::ScrollDown,
            KeyCode::Char('k') | KeyCode::Char('K') => return InputAction::ScrollUp,
            KeyCode::Char('l') | KeyCode::Char('L') => return InputAction::ChildNext,
            _ => {}
        }
    }

    if state.child_navigation_prefix {
        return match key.code {
            KeyCode::Down => InputAction::ChildNext,
            KeyCode::Left => InputAction::ChildPrev,
            KeyCode::Right => InputAction::ChildNext,
            KeyCode::Up => InputAction::ChildParent,
            KeyCode::Esc => InputAction::NoOp,
            _ => InputAction::NoOp,
        };
    }

    if key.modifiers.contains(KeyModifiers::ALT) {
        return match key.code {
            KeyCode::Right => InputAction::ChildNext,
            KeyCode::Left => InputAction::ChildPrev,
            KeyCode::Up => InputAction::ChildParent,
            _ => InputAction::NoOp,
        };
    }

    if state.pending_permission.is_some() {
        return match key.code {
            KeyCode::Up => InputAction::ScrollUp,
            KeyCode::Down => InputAction::ScrollDown,
            KeyCode::PageUp => InputAction::ScrollPageUp,
            KeyCode::PageDown => InputAction::ScrollPageDown,
            KeyCode::End => InputAction::ScrollToBottom,
            KeyCode::Char('y') | KeyCode::Char('Y') | KeyCode::Char('a') | KeyCode::Char('A') => {
                InputAction::ApprovePermission
            }
            KeyCode::Char('n')
            | KeyCode::Char('N')
            | KeyCode::Char('d')
            | KeyCode::Char('D')
            | KeyCode::Esc => InputAction::DenyPermission,
            KeyCode::Enter => InputAction::NoOp,
            _ => InputAction::NoOp,
        };
    }

    if state.dialog_is_open() {
        let search_dialog = state
            .dialog()
            .map(|dialog| {
                matches!(
                    dialog.kind,
                    DialogKind::ModelPicker | DialogKind::SessionPicker
                )
            })
            .unwrap_or(false);

        return if search_dialog {
            match key.code {
                KeyCode::Up => InputAction::DialogPrev,
                KeyCode::Down => InputAction::DialogNext,
                KeyCode::Enter => InputAction::DialogAccept,
                KeyCode::Esc => InputAction::DialogCancel,
                KeyCode::Backspace => InputAction::DialogBackspace,
                KeyCode::Char(ch) if !has_non_shift_modifiers(key.modifiers) => {
                    InputAction::DialogInsert(ch)
                }
                _ => InputAction::NoOp,
            }
        } else {
            match key.code {
                KeyCode::Up | KeyCode::Char('k') | KeyCode::Char('K') => InputAction::DialogPrev,
                KeyCode::Down | KeyCode::Char('j') | KeyCode::Char('J') => InputAction::DialogNext,
                KeyCode::Enter => InputAction::DialogAccept,
                KeyCode::Esc => InputAction::DialogCancel,
                _ => InputAction::NoOp,
            }
        };
    }

    if state.slash_panel_is_open() {
        return match key.code {
            KeyCode::Up => InputAction::SlashPanelPrev,
            KeyCode::Down => InputAction::SlashPanelNext,
            KeyCode::Tab => InputAction::SlashPanelAccept,
            KeyCode::Esc => InputAction::SlashPanelDismiss,
            KeyCode::Enter => InputAction::Submit,
            KeyCode::Backspace => InputAction::Backspace,
            KeyCode::Char(ch) if !has_non_shift_modifiers(key.modifiers) => InputAction::Insert(ch),
            _ => InputAction::NoOp,
        };
    }

    match key.code {
        KeyCode::Esc if matches!(state.phase, super::state::AppPhase::Running) => {
            InputAction::Interrupt
        }
        KeyCode::Up => InputAction::ScrollUp,
        KeyCode::Down => InputAction::ScrollDown,
        KeyCode::PageUp => InputAction::ScrollPageUp,
        KeyCode::PageDown => InputAction::ScrollPageDown,
        KeyCode::End => InputAction::ScrollToBottom,
        KeyCode::Enter => InputAction::Submit,
        KeyCode::Backspace => InputAction::Backspace,
        KeyCode::Char(ch) if !has_non_shift_modifiers(key.modifiers) => InputAction::Insert(ch),
        _ => InputAction::NoOp,
    }
}

pub fn apply_edit_action(state: &mut TuiState, action: &InputAction) -> bool {
    if state.is_read_only_child_view() {
        return false;
    }

    match action {
        InputAction::Insert(ch) => {
            state.input_buffer.push(*ch);
            state.sync_input_phase();
            state.sync_slash_panel();
            true
        }
        InputAction::Backspace => {
            if state.input_buffer.pop().is_some() {
                state.sync_input_phase();
                state.sync_slash_panel();
                true
            } else {
                false
            }
        }
        _ => false,
    }
}

pub fn map_mouse_event(_state: &TuiState, mouse: MouseEvent) -> InputAction {
    match mouse.kind {
        MouseEventKind::ScrollUp => InputAction::MouseScrollUp,
        MouseEventKind::ScrollDown => InputAction::MouseScrollDown,
        MouseEventKind::Down(_) | MouseEventKind::Drag(_) | MouseEventKind::Moved => {
            InputAction::MouseClick
        }
        _ => InputAction::NoOp,
    }
}

fn has_non_shift_modifiers(modifiers: KeyModifiers) -> bool {
    modifiers.intersects(KeyModifiers::ALT | KeyModifiers::CONTROL | KeyModifiers::SUPER)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tui::{AppPhase, PermissionRequestEvent};

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    #[test]
    fn edit_actions_update_input_buffer() {
        let mut state = TuiState::default();

        assert!(apply_edit_action(&mut state, &InputAction::Insert('h')));
        assert!(apply_edit_action(&mut state, &InputAction::Insert('i')));
        assert_eq!(state.input_buffer, "hi");
        assert_eq!(state.phase, AppPhase::Editing);

        assert!(apply_edit_action(&mut state, &InputAction::Backspace));
        assert_eq!(state.input_buffer, "h");

        assert!(apply_edit_action(&mut state, &InputAction::Backspace));
        assert_eq!(state.input_buffer, "");
        assert_eq!(state.phase, AppPhase::Idle);
        assert!(!apply_edit_action(&mut state, &InputAction::Backspace));
    }

    #[test]
    fn enter_maps_to_submit_when_not_waiting_for_permission() {
        let state = TuiState::default();

        assert_eq!(
            map_key_event(&state, key(KeyCode::Enter)),
            InputAction::Submit
        );
    }

    #[test]
    fn scroll_keys_map_without_conflicting_with_input_text() {
        let state = TuiState::default();

        assert_eq!(
            map_key_event(&state, key(KeyCode::Up)),
            InputAction::ScrollUp
        );
        assert_eq!(
            map_key_event(&state, key(KeyCode::Down)),
            InputAction::ScrollDown
        );
        assert_eq!(
            map_key_event(&state, key(KeyCode::PageUp)),
            InputAction::ScrollPageUp
        );
        assert_eq!(
            map_key_event(&state, key(KeyCode::PageDown)),
            InputAction::ScrollPageDown
        );
        assert_eq!(
            map_key_event(&state, key(KeyCode::End)),
            InputAction::ScrollToBottom
        );
    }

    #[test]
    fn only_ctrl_c_quits_without_a_permission_prompt() {
        let mut empty_state = TuiState::default();
        assert_eq!(
            map_key_event(&empty_state, key(KeyCode::Esc)),
            InputAction::NoOp
        );
        assert_eq!(
            map_key_event(&empty_state, key(KeyCode::Char('q'))),
            InputAction::Insert('q')
        );
        assert_eq!(
            map_key_event(
                &empty_state,
                KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL)
            ),
            InputAction::Quit
        );

        empty_state.set_input("q");
        assert_eq!(
            map_key_event(&empty_state, key(KeyCode::Char('q'))),
            InputAction::Insert('q')
        );
    }

    #[test]
    fn ctrl_t_cycles_reasoning_effort() {
        let state = TuiState::default();

        assert_eq!(
            map_key_event(
                &state,
                KeyEvent::new(KeyCode::Char('t'), KeyModifiers::CONTROL)
            ),
            InputAction::CycleReasoningEffort
        );
    }

    #[test]
    fn ctrl_x_maps_to_child_prefix() {
        let state = TuiState::default();

        assert_eq!(
            map_key_event(
                &state,
                KeyEvent::new(KeyCode::Char('x'), KeyModifiers::CONTROL)
            ),
            InputAction::ChildPrefix
        );

        assert_eq!(
            map_key_event(&state, key(KeyCode::Char('\u{0018}'))),
            InputAction::ChildPrefix
        );
    }

    #[test]
    fn ctrl_x_prefix_enters_child_navigation_mode() {
        let mut state = TuiState::default();
        state.child_navigation_prefix = true;

        assert_eq!(
            map_key_event(&state, key(KeyCode::Down)),
            InputAction::ChildNext
        );
        assert_eq!(
            map_key_event(&state, key(KeyCode::Up)),
            InputAction::ChildParent
        );
    }

    #[test]
    fn child_view_arrow_keys_navigate_without_prefix() {
        let mut state = TuiState::default();
        state.replace_child_timeline_from_records(
            &[],
            "parent-session",
            "child-session",
            "explorer",
            0,
            1,
        );

        assert_eq!(
            map_key_event(&state, key(KeyCode::Up)),
            InputAction::ChildParent
        );
        assert_eq!(
            map_key_event(&state, key(KeyCode::Left)),
            InputAction::ChildPrev
        );
        assert_eq!(
            map_key_event(&state, key(KeyCode::Right)),
            InputAction::ChildNext
        );
        assert_eq!(
            map_key_event(&state, key(KeyCode::Down)),
            InputAction::ScrollDown
        );
    }

    #[test]
    fn child_view_hjkl_matches_navigation_and_scroll_roles() {
        let mut state = TuiState::default();
        state.replace_child_timeline_from_records(
            &[],
            "parent-session",
            "child-session",
            "explorer",
            0,
            1,
        );

        assert_eq!(
            map_key_event(&state, key(KeyCode::Char('h'))),
            InputAction::ChildPrev
        );
        assert_eq!(
            map_key_event(&state, key(KeyCode::Char('j'))),
            InputAction::ScrollDown
        );
        assert_eq!(
            map_key_event(&state, key(KeyCode::Char('k'))),
            InputAction::ScrollUp
        );
        assert_eq!(
            map_key_event(&state, key(KeyCode::Char('l'))),
            InputAction::ChildNext
        );
    }

    #[test]
    fn mouse_events_map_to_scroll_and_click_actions() {
        let state = TuiState::default();

        assert_eq!(
            map_mouse_event(
                &state,
                MouseEvent {
                    kind: MouseEventKind::ScrollUp,
                    column: 0,
                    row: 0,
                    modifiers: KeyModifiers::NONE,
                }
            ),
            InputAction::MouseScrollUp
        );
        assert_eq!(
            map_mouse_event(
                &state,
                MouseEvent {
                    kind: MouseEventKind::ScrollDown,
                    column: 0,
                    row: 0,
                    modifiers: KeyModifiers::NONE,
                }
            ),
            InputAction::MouseScrollDown
        );
        assert_eq!(
            map_mouse_event(
                &state,
                MouseEvent {
                    kind: MouseEventKind::Down(crossterm::event::MouseButton::Left),
                    column: 0,
                    row: 0,
                    modifiers: KeyModifiers::NONE,
                }
            ),
            InputAction::MouseClick
        );
    }

    #[test]
    fn alt_arrow_keys_map_to_child_navigation_actions() {
        let state = TuiState::default();

        assert_eq!(
            map_key_event(&state, KeyEvent::new(KeyCode::Right, KeyModifiers::ALT)),
            InputAction::ChildNext
        );
        assert_eq!(
            map_key_event(&state, KeyEvent::new(KeyCode::Left, KeyModifiers::ALT)),
            InputAction::ChildPrev
        );
        assert_eq!(
            map_key_event(&state, KeyEvent::new(KeyCode::Up, KeyModifiers::ALT)),
            InputAction::ChildParent
        );
    }

    #[test]
    fn esc_interrupts_running_turn_without_quitting() {
        let mut state = TuiState::default();
        state.phase = AppPhase::Running;

        assert_eq!(
            map_key_event(&state, key(KeyCode::Esc)),
            InputAction::Interrupt
        );
    }

    #[test]
    fn permission_prompt_maps_approve_and_deny_actions() {
        let mut state = TuiState::default();
        state.pending_permission = Some(crate::tui::PermissionView::from_request(
            PermissionRequestEvent::new("call-1", "shell__exec", "ls"),
        ));

        assert_eq!(
            map_key_event(&state, key(KeyCode::Char('y'))),
            InputAction::ApprovePermission
        );
        assert_eq!(
            map_key_event(&state, key(KeyCode::Char('a'))),
            InputAction::ApprovePermission
        );
        assert_eq!(
            map_key_event(&state, key(KeyCode::Char('Y'))),
            InputAction::ApprovePermission
        );
        assert_eq!(
            map_key_event(&state, key(KeyCode::Enter)),
            InputAction::NoOp
        );
        assert_eq!(
            map_key_event(&state, key(KeyCode::Char('n'))),
            InputAction::DenyPermission
        );
        assert_eq!(
            map_key_event(&state, key(KeyCode::Char('d'))),
            InputAction::DenyPermission
        );
        assert_eq!(
            map_key_event(&state, key(KeyCode::Char('N'))),
            InputAction::DenyPermission
        );
        assert_eq!(
            map_key_event(&state, key(KeyCode::Esc)),
            InputAction::DenyPermission
        );
        assert_eq!(
            map_key_event(&state, key(KeyCode::Char('x'))),
            InputAction::NoOp
        );
    }

    #[test]
    fn permission_picker_does_not_accept_filter_text() {
        let mut state = TuiState::default();
        state.open_dialog(crate::tui::state::DialogState::new(
            DialogKind::PermissionPicker,
            "Permission mode",
            None,
            vec![crate::tui::state::DialogItem::new(
                "default", "Default", None,
            )],
        ));

        assert_eq!(
            map_key_event(&state, key(KeyCode::Char('d'))),
            InputAction::NoOp
        );
        assert_eq!(
            map_key_event(&state, key(KeyCode::Down)),
            InputAction::DialogNext
        );
    }

    #[test]
    fn scroll_actions_still_work_while_permission_prompt_is_pending() {
        let mut state = TuiState::default();
        state.pending_permission = Some(crate::tui::PermissionView::from_request(
            PermissionRequestEvent::new("call-1", "shell__exec", "ls"),
        ));

        assert_eq!(
            map_key_event(&state, key(KeyCode::Up)),
            InputAction::ScrollUp
        );
        assert_eq!(
            map_key_event(&state, key(KeyCode::Down)),
            InputAction::ScrollDown
        );
        assert_eq!(
            map_key_event(&state, key(KeyCode::PageUp)),
            InputAction::ScrollPageUp
        );
        assert_eq!(
            map_key_event(&state, key(KeyCode::PageDown)),
            InputAction::ScrollPageDown
        );
        assert_eq!(
            map_key_event(&state, key(KeyCode::End)),
            InputAction::ScrollToBottom
        );
    }

    #[test]
    fn slash_panel_remaps_navigation_keys() {
        let mut state = TuiState::default();
        state.set_input("/p");

        assert_eq!(
            map_key_event(&state, key(KeyCode::Up)),
            InputAction::SlashPanelPrev
        );
        assert_eq!(
            map_key_event(&state, key(KeyCode::Down)),
            InputAction::SlashPanelNext
        );
        assert_eq!(
            map_key_event(&state, key(KeyCode::Tab)),
            InputAction::SlashPanelAccept
        );
        assert_eq!(
            map_key_event(&state, key(KeyCode::Esc)),
            InputAction::SlashPanelDismiss
        );
    }
}
