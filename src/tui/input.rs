use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use super::state::TuiState;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InputAction {
    Insert(char),
    Backspace,
    Submit,
    ScrollUp,
    ScrollDown,
    ScrollPageUp,
    ScrollPageDown,
    ScrollToBottom,
    ApprovePermission,
    DenyPermission,
    Quit,
    Tick,
    NoOp,
}

pub fn map_key_event(state: &TuiState, key: KeyEvent) -> InputAction {
    if key.modifiers.contains(KeyModifiers::CONTROL) && matches!(key.code, KeyCode::Char('c')) {
        return InputAction::Quit;
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

    match key.code {
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
    match action {
        InputAction::Insert(ch) => {
            state.input_buffer.push(*ch);
            state.sync_input_phase();
            true
        }
        InputAction::Backspace => {
            if state.input_buffer.pop().is_some() {
                state.sync_input_phase();
                true
            } else {
                false
            }
        }
        _ => false,
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
    fn permission_prompt_maps_approve_and_deny_actions() {
        let mut state = TuiState::default();
        state.pending_permission = Some(crate::tui::PermissionView::from_request(
            PermissionRequestEvent::new("call-1", "bash", "ls"),
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
    fn scroll_actions_still_work_while_permission_prompt_is_pending() {
        let mut state = TuiState::default();
        state.pending_permission = Some(crate::tui::PermissionView::from_request(
            PermissionRequestEvent::new("call-1", "bash", "ls"),
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
}
