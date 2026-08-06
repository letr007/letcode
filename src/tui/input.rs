use crossterm::event::{KeyCode, KeyEvent, KeyModifiers, MouseEvent, MouseEventKind};

use super::state::{COMPOSER_ATTACHMENT_MARKER, DialogKind, TuiState};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InputAction {
    Insert(char),
    Paste(String),
    PasteFromClipboard,
    InsertNewline,
    Backspace,
    Delete,
    MoveCursorLeft,
    MoveCursorRight,
    MoveCursorHome,
    MoveCursorEnd,
    DialogInsert(char),
    DialogPaste(String),
    DialogBackspace,
    DialogNext,
    DialogPrev,
    DialogAccept,
    DialogToggle,
    DialogCancel,
    SlashPanelNext,
    SlashPanelPrev,
    SlashPanelAccept,
    SlashPanelDismiss,
    Submit,
    HistoryPrev,
    HistoryNext,
    ScrollUp,
    ScrollDown,
    ScrollPageUp,
    ScrollPageDown,
    ScrollToBottom,
    MouseScrollUp,
    MouseScrollDown,
    MouseSelectionStart(u16, u16),
    MouseSelectionDrag(u16, u16),
    /// Third flag is true when Ctrl/Cmd is held — used to activate underlined links.
    MouseSelectionEnd(u16, u16, bool),
    CopySelection,
    ClearSelection,
    CycleReasoningEffort,
    ChildPrefix,
    ChildFirst,
    ChildNext,
    ChildPrev,
    ChildParent,
    QuestionPrevTab,
    QuestionNextTab,
    QuestionPrevOption,
    QuestionNextOption,
    QuestionPickOption(u8),
    QuestionActivate,
    QuestionSubmit,
    QuestionCancel,
    QuestionInsert(char),
    QuestionPaste(String),
    QuestionBackspace,
    QuestionDelete,
    QuestionMoveCursorLeft,
    QuestionMoveCursorRight,
    QuestionMoveCursorHome,
    QuestionMoveCursorEnd,
    ApprovePermission,
    ApprovePermissionAlways,
    DenyPermission,
    Interrupt,
    Quit,
    Tick,
    NoOp,
}

pub fn map_key_event(state: &TuiState, key: KeyEvent) -> InputAction {
    // Ctrl+C：有选择时复制，无选择时退出
    if key.modifiers.contains(KeyModifiers::CONTROL) && matches!(key.code, KeyCode::Char('c')) {
        if state.text_selection.is_some() {
            return InputAction::CopySelection;
        } else {
            return InputAction::Quit;
        }
    }

    if let Some(question) = state.pending_question.as_ref() {
        if question.editing_custom {
            return match key.code {
                KeyCode::Char('v')
                    if key.modifiers.contains(KeyModifiers::CONTROL)
                        || key.modifiers.contains(KeyModifiers::SUPER) =>
                {
                    InputAction::PasteFromClipboard
                }
                KeyCode::Backspace => InputAction::QuestionBackspace,
                KeyCode::Delete => InputAction::QuestionDelete,
                KeyCode::Left => InputAction::QuestionMoveCursorLeft,
                KeyCode::Right => InputAction::QuestionMoveCursorRight,
                KeyCode::Home => InputAction::QuestionMoveCursorHome,
                KeyCode::End => InputAction::QuestionMoveCursorEnd,
                KeyCode::Char('a') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    InputAction::QuestionMoveCursorHome
                }
                KeyCode::Char('e') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    InputAction::QuestionMoveCursorEnd
                }
                KeyCode::Enter => InputAction::QuestionActivate,
                KeyCode::Esc => InputAction::QuestionCancel,
                KeyCode::Char(ch) if !has_non_shift_modifiers(key.modifiers) => {
                    InputAction::QuestionInsert(ch)
                }
                _ => InputAction::NoOp,
            };
        }

        return match key.code {
            KeyCode::Left | KeyCode::Char('h') | KeyCode::Char('H') => InputAction::QuestionPrevTab,
            KeyCode::Right | KeyCode::Char('l') | KeyCode::Char('L') | KeyCode::Tab => {
                InputAction::QuestionNextTab
            }
            KeyCode::BackTab => InputAction::QuestionPrevTab,
            KeyCode::Up | KeyCode::Char('k') | KeyCode::Char('K') => {
                InputAction::QuestionPrevOption
            }
            KeyCode::Down | KeyCode::Char('j') | KeyCode::Char('J') => {
                InputAction::QuestionNextOption
            }
            KeyCode::Char(ch) if ('1'..='9').contains(&ch) => {
                InputAction::QuestionPickOption(ch as u8 - b'0')
            }
            KeyCode::Enter => {
                if question.is_confirm_tab() {
                    InputAction::QuestionSubmit
                } else {
                    InputAction::QuestionActivate
                }
            }
            KeyCode::Esc => InputAction::QuestionCancel,
            _ => InputAction::NoOp,
        };
    }

    // Ctrl+V / Cmd+V: 主动从系统剪贴板读取并插入，避免依赖终端把内容逐字符“灌进来”。
    if (key.modifiers.contains(KeyModifiers::CONTROL)
        || key.modifiers.contains(KeyModifiers::SUPER))
        && matches!(key.code, KeyCode::Char('v'))
    {
        return InputAction::PasteFromClipboard;
    }

    if state.pending_permission.is_some() {
        return match key.code {
            KeyCode::Up => InputAction::ScrollUp,
            KeyCode::Down => InputAction::ScrollDown,
            KeyCode::PageUp => InputAction::ScrollPageUp,
            KeyCode::PageDown => InputAction::ScrollPageDown,
            KeyCode::End => InputAction::ScrollToBottom,
            KeyCode::Char('y') | KeyCode::Char('Y') | KeyCode::Char('o') | KeyCode::Char('O') => {
                InputAction::ApprovePermission
            }
            KeyCode::Char('a') | KeyCode::Char('A')
                if state
                    .pending_permission
                    .as_ref()
                    .is_some_and(|permission| permission.can_allow_always) =>
            {
                InputAction::ApprovePermissionAlways
            }
            KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Char('d') | KeyCode::Char('D') => {
                InputAction::DenyPermission
            }
            KeyCode::Esc => InputAction::Interrupt,
            _ => InputAction::NoOp,
        };
    }

    if key.modifiers.contains(KeyModifiers::CONTROL) && matches!(key.code, KeyCode::Char('t')) {
        return InputAction::CycleReasoningEffort;
    }

    if (key.modifiers.contains(KeyModifiers::CONTROL) && matches!(key.code, KeyCode::Char('x')))
        || matches!(key.code, KeyCode::Char(''))
    {
        return InputAction::ChildPrefix;
    }

    if state.is_read_only_child_view()
        && state.input_buffer.is_empty()
        && !has_non_shift_modifiers(key.modifiers)
    {
        match key.code {
            KeyCode::Up => return InputAction::ChildParent,
            KeyCode::Down => return InputAction::ScrollDown,
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
            KeyCode::Down => InputAction::ChildFirst,
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

    if state.dialog_is_open() {
        let search_dialog = state
            .dialog()
            .map(|dialog| {
                matches!(
                    dialog.kind,
                    DialogKind::ModelPicker
                        | DialogKind::SessionPicker
                        | DialogKind::HistoryTree
                        | DialogKind::ContextPicker
                        | DialogKind::McpPicker
                        | DialogKind::McpToolsPicker
                        | DialogKind::SkillPicker
                )
            })
            .unwrap_or(false);

        return if search_dialog {
            match key.code {
                KeyCode::Up => InputAction::DialogPrev,
                KeyCode::Down => InputAction::DialogNext,
                KeyCode::Enter => InputAction::DialogAccept,
                KeyCode::Char(' ')
                    if state
                        .dialog()
                        .is_some_and(|dialog| dialog.kind == DialogKind::McpPicker) =>
                {
                    InputAction::DialogToggle
                }
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
            KeyCode::Delete => InputAction::Delete,
            KeyCode::Left => InputAction::MoveCursorLeft,
            KeyCode::Right => InputAction::MoveCursorRight,
            KeyCode::Home => InputAction::MoveCursorHome,
            KeyCode::End => InputAction::MoveCursorEnd,
            KeyCode::Char('a') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                InputAction::MoveCursorHome
            }
            KeyCode::Char('e') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                InputAction::MoveCursorEnd
            }
            KeyCode::Char(ch) if !has_non_shift_modifiers(key.modifiers) => InputAction::Insert(ch),
            _ => InputAction::NoOp,
        };
    }

    // Esc：清除选择（如果有），否则中断执行
    if matches!(key.code, KeyCode::Esc) {
        if state.text_selection.is_some() {
            return InputAction::ClearSelection;
        }
    }

    match key.code {
        // Whether an interrupt is meaningful depends on session activity, not
        // the projected UI phase. A recoverable session transport error can leave work
        // active while the projection is in `Error`.
        KeyCode::Esc => InputAction::Interrupt,
        KeyCode::Up => InputAction::HistoryPrev,
        KeyCode::Down => InputAction::HistoryNext,
        KeyCode::PageUp => InputAction::ScrollPageUp,
        KeyCode::PageDown => InputAction::ScrollPageDown,
        KeyCode::Home => InputAction::MoveCursorHome,
        KeyCode::End if key.modifiers.contains(KeyModifiers::CONTROL) => {
            InputAction::ScrollToBottom
        }
        KeyCode::End => InputAction::MoveCursorEnd,
        KeyCode::Left => InputAction::MoveCursorLeft,
        KeyCode::Right => InputAction::MoveCursorRight,
        KeyCode::Delete => InputAction::Delete,
        KeyCode::Enter if key.modifiers.contains(KeyModifiers::SHIFT) => InputAction::InsertNewline,
        KeyCode::Enter => InputAction::Submit,
        KeyCode::Backspace => InputAction::Backspace,
        KeyCode::Char('a') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            InputAction::MoveCursorHome
        }
        KeyCode::Char('e') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            InputAction::MoveCursorEnd
        }
        KeyCode::Char(ch) if !has_non_shift_modifiers(key.modifiers) => InputAction::Insert(ch),
        _ => InputAction::NoOp,
    }
}

pub fn apply_edit_action(state: &mut TuiState, action: &InputAction) -> bool {
    if state.is_read_only_child_view()
        && state.input_buffer.is_empty()
        && !matches!(action, InputAction::Insert('/'))
    {
        return false;
    }

    match action {
        InputAction::Insert(ch) => insert_at_cursor(state, *ch),
        InputAction::Paste(text) => insert_text_at_cursor(state, text),
        InputAction::InsertNewline => insert_at_cursor(state, '\n'),
        InputAction::Backspace => backspace_at_cursor(state),
        InputAction::Delete => delete_at_cursor(state),
        InputAction::MoveCursorLeft => move_cursor_left(state),
        InputAction::MoveCursorRight => move_cursor_right(state),
        InputAction::MoveCursorHome => move_cursor_home(state),
        InputAction::MoveCursorEnd => move_cursor_end(state),
        _ => false,
    }
}

pub fn map_paste_event(state: &TuiState, text: String) -> InputAction {
    if state
        .pending_question
        .as_ref()
        .is_some_and(|question| question.editing_custom)
    {
        return InputAction::QuestionPaste(text);
    }

    if state.dialog_is_open() {
        let search_dialog = state
            .dialog()
            .map(|dialog| {
                matches!(
                    dialog.kind,
                    DialogKind::ModelPicker
                        | DialogKind::SessionPicker
                        | DialogKind::HistoryTree
                        | DialogKind::ContextPicker
                        | DialogKind::McpPicker
                        | DialogKind::McpToolsPicker
                        | DialogKind::SkillPicker
                )
            })
            .unwrap_or(false);
        if search_dialog {
            return InputAction::DialogPaste(text);
        }
    }

    InputAction::Paste(text)
}

pub fn map_mouse_event(_state: &TuiState, mouse: MouseEvent) -> InputAction {
    use crossterm::event::MouseButton;

    match mouse.kind {
        MouseEventKind::ScrollUp => InputAction::MouseScrollUp,
        MouseEventKind::ScrollDown => InputAction::MouseScrollDown,

        // 左键按下：开始选择
        MouseEventKind::Down(MouseButton::Left) => {
            InputAction::MouseSelectionStart(mouse.column, mouse.row)
        }

        // 左键拖拽：更新选择范围
        MouseEventKind::Drag(MouseButton::Left) => {
            InputAction::MouseSelectionDrag(mouse.column, mouse.row)
        }

        // 左键松开：结束选择；Ctrl/Cmd+click 才激活链接跳转
        MouseEventKind::Up(MouseButton::Left) => {
            let activate_link = mouse
                .modifiers
                .intersects(KeyModifiers::CONTROL | KeyModifiers::SUPER);
            InputAction::MouseSelectionEnd(mouse.column, mouse.row, activate_link)
        }

        _ => InputAction::NoOp,
    }
}

fn has_non_shift_modifiers(modifiers: KeyModifiers) -> bool {
    modifiers.intersects(KeyModifiers::ALT | KeyModifiers::CONTROL | KeyModifiers::SUPER)
}

fn insert_at_cursor(state: &mut TuiState, ch: char) -> bool {
    if ch == COMPOSER_ATTACHMENT_MARKER {
        return false;
    }
    state.input_cursor = clamp_to_char_boundary(&state.input_buffer, state.input_cursor);
    state.input_buffer.insert(state.input_cursor, ch);
    state.input_cursor += ch.len_utf8();
    state.sync_input_phase();
    state.sync_slash_panel();
    true
}

fn insert_text_at_cursor(state: &mut TuiState, text: &str) -> bool {
    if text.is_empty() {
        return false;
    }
    let text = text.replace(COMPOSER_ATTACHMENT_MARKER, "");
    if text.is_empty() {
        return false;
    }
    state.input_cursor = clamp_to_char_boundary(&state.input_buffer, state.input_cursor);
    state.input_buffer.insert_str(state.input_cursor, &text);
    state.input_cursor += text.len();
    state.sync_input_phase();
    state.sync_slash_panel();
    true
}

fn backspace_at_cursor(state: &mut TuiState) -> bool {
    state.input_cursor = clamp_to_char_boundary(&state.input_buffer, state.input_cursor);
    let Some(previous) = previous_char_boundary(&state.input_buffer, state.input_cursor) else {
        return false;
    };
    if state.input_buffer[previous..state.input_cursor]
        .chars()
        .next()
        .is_some_and(|ch| ch == COMPOSER_ATTACHMENT_MARKER)
    {
        return state.remove_composer_token_at_marker(previous);
    }
    state.input_buffer.drain(previous..state.input_cursor);
    state.input_cursor = previous;
    state.sync_input_phase();
    state.sync_slash_panel();
    true
}

fn delete_at_cursor(state: &mut TuiState) -> bool {
    state.input_cursor = clamp_to_char_boundary(&state.input_buffer, state.input_cursor);
    let Some(next) = next_char_boundary(&state.input_buffer, state.input_cursor) else {
        return false;
    };
    if state.input_buffer[state.input_cursor..next]
        .chars()
        .next()
        .is_some_and(|ch| ch == COMPOSER_ATTACHMENT_MARKER)
    {
        return state.remove_composer_token_at_marker(state.input_cursor);
    }
    state.input_buffer.drain(state.input_cursor..next);
    state.sync_input_phase();
    state.sync_slash_panel();
    true
}

fn move_cursor_left(state: &mut TuiState) -> bool {
    state.input_cursor = clamp_to_char_boundary(&state.input_buffer, state.input_cursor);
    let Some(previous) = previous_char_boundary(&state.input_buffer, state.input_cursor) else {
        return false;
    };
    state.input_cursor = previous;
    true
}

fn move_cursor_right(state: &mut TuiState) -> bool {
    state.input_cursor = clamp_to_char_boundary(&state.input_buffer, state.input_cursor);
    let Some(next) = next_char_boundary(&state.input_buffer, state.input_cursor) else {
        return false;
    };
    state.input_cursor = next;
    true
}

fn move_cursor_home(state: &mut TuiState) -> bool {
    if state.input_cursor == 0 {
        return false;
    }
    state.input_cursor = 0;
    true
}

fn move_cursor_end(state: &mut TuiState) -> bool {
    if state.input_cursor == state.input_buffer.len() {
        return false;
    }
    state.input_cursor = state.input_buffer.len();
    true
}

fn previous_char_boundary(text: &str, cursor: usize) -> Option<usize> {
    if cursor == 0 {
        return None;
    }
    text[..cursor].char_indices().last().map(|(index, _)| index)
}

fn next_char_boundary(text: &str, cursor: usize) -> Option<usize> {
    if cursor >= text.len() {
        return None;
    }
    text[cursor..]
        .chars()
        .next()
        .map(|ch| cursor + ch.len_utf8())
}

fn clamp_to_char_boundary(text: &str, cursor: usize) -> usize {
    let mut cursor = cursor.min(text.len());
    while cursor > 0 && !text.is_char_boundary(cursor) {
        cursor -= 1;
    }
    cursor
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tui::{AppPhase, PermissionRequestEvent};
    use crate::user_content::UserImageAttachment;

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    fn attachment(id: &str) -> UserImageAttachment {
        UserImageAttachment {
            id: id.into(),
            label: format!("{id}.png"),
            mime: "image/png".into(),
            data_url: "data:image/png;base64,AAAA".into(),
        }
    }

    #[test]
    fn edit_actions_update_input_buffer() {
        let mut state = TuiState::default();

        assert!(apply_edit_action(&mut state, &InputAction::Insert('h')));
        assert!(apply_edit_action(&mut state, &InputAction::Insert('i')));
        assert_eq!(state.input_buffer, "hi");
        assert_eq!(state.input_cursor, 2);
        assert_eq!(state.phase, AppPhase::Editing);

        assert!(apply_edit_action(&mut state, &InputAction::MoveCursorLeft));
        assert!(apply_edit_action(&mut state, &InputAction::Insert('!')));
        assert_eq!(state.input_buffer, "h!i");
        assert_eq!(state.input_cursor, 2);

        assert!(apply_edit_action(&mut state, &InputAction::Delete));
        assert_eq!(state.input_buffer, "h!");

        assert!(apply_edit_action(&mut state, &InputAction::Backspace));
        assert_eq!(state.input_buffer, "h");

        assert!(apply_edit_action(&mut state, &InputAction::Backspace));
        assert_eq!(state.input_buffer, "");
        assert_eq!(state.phase, AppPhase::Idle);
        assert!(!apply_edit_action(&mut state, &InputAction::Backspace));
    }

    #[test]
    fn paste_action_inserts_full_text_at_cursor() {
        let mut state = TuiState::default();
        state.input_buffer = "helo".into();
        state.input_cursor = 2;

        assert!(apply_edit_action(
            &mut state,
            &InputAction::Paste("ll\nworld".into())
        ));
        assert_eq!(state.input_buffer, "hell\nworldlo");
        assert_eq!(state.input_cursor, 2 + "ll\nworld".len());
    }

    #[test]
    fn external_marker_text_is_ignored_without_touching_attachments() {
        let mut state = TuiState::default();
        state.add_composer_attachment(attachment("one"));
        let original = state.input_buffer.clone();

        assert!(!apply_edit_action(
            &mut state,
            &InputAction::Insert(COMPOSER_ATTACHMENT_MARKER)
        ));
        assert!(apply_edit_action(
            &mut state,
            &InputAction::Paste(format!("x{COMPOSER_ATTACHMENT_MARKER}y"))
        ));
        assert_eq!(state.input_buffer, format!("{original}xy"));
        assert_eq!(state.composer_tokens[0].image().unwrap().id, "one");
        state.assert_composer_token_invariant();

        state.set_input(format!("a{COMPOSER_ATTACHMENT_MARKER}b"));
        assert_eq!(state.input_buffer, "ab");
        assert!(state.composer_tokens.is_empty());
    }

    #[test]
    fn attachment_tokens_are_atomic_inline_editing_units() {
        let mut state = TuiState::default();
        state.set_input("before after");
        state.input_cursor = "before ".len();
        state.add_composer_attachment(attachment("one"));

        assert_eq!(state.input_buffer, format!("before {}after", '\u{fffc}'));
        assert_eq!(state.composer_tokens.len(), 1);

        assert!(apply_edit_action(&mut state, &InputAction::Insert('X')));
        assert_eq!(state.input_buffer, format!("before {}Xafter", '\u{fffc}'));

        assert!(apply_edit_action(&mut state, &InputAction::MoveCursorLeft));
        assert!(apply_edit_action(&mut state, &InputAction::Backspace));
        assert_eq!(state.input_buffer, "before Xafter");
        assert!(state.composer_tokens.is_empty());
    }

    #[test]
    fn arrow_navigation_can_select_and_delete_attachment_tokens() {
        let mut state = TuiState::default();
        state.set_input("hello");
        state.input_cursor = 0;
        state.add_composer_attachment(attachment("one"));
        state.add_composer_attachment(attachment("two"));
        state.input_cursor = 0;

        assert!(apply_edit_action(&mut state, &InputAction::Delete));
        assert_eq!(state.composer_tokens.len(), 1);
        assert_eq!(state.composer_tokens[0].image().unwrap().id, "two");
    }

    #[test]
    fn skill_tokens_are_atomic_and_removed_with_backspace() {
        let mut state = TuiState::default();
        state.set_input("review");
        assert!(state.add_composer_skill("rust-audit".into()));

        assert!(apply_edit_action(&mut state, &InputAction::Backspace));
        assert_eq!(state.input_buffer, "review");
        assert!(state.composer_tokens.is_empty());
        assert!(state.composer_content().selected_skills.is_empty());
    }

    #[test]
    fn ctrl_or_cmd_v_maps_to_clipboard_paste_action() {
        let state = TuiState::default();

        assert_eq!(
            map_key_event(
                &state,
                KeyEvent::new(KeyCode::Char('v'), KeyModifiers::CONTROL)
            ),
            InputAction::PasteFromClipboard
        );
        assert_eq!(
            map_key_event(
                &state,
                KeyEvent::new(KeyCode::Char('v'), KeyModifiers::SUPER)
            ),
            InputAction::PasteFromClipboard
        );
    }

    #[test]
    fn paste_event_targets_search_dialog_query() {
        let mut state = TuiState::default();
        state.open_dialog(crate::tui::state::DialogState::new(
            DialogKind::ModelPicker,
            "Pick model",
            None,
            vec![crate::tui::state::DialogItem::new("gpt", "GPT", None)],
        ));

        assert_eq!(
            map_paste_event(&state, "hello".into()),
            InputAction::DialogPaste("hello".into())
        );
    }

    #[test]
    fn mcp_tools_picker_accepts_search_input_and_paste() {
        let mut state = TuiState::default();
        state.open_dialog(crate::tui::state::DialogState::new(
            DialogKind::McpToolsPicker,
            "Tools · local",
            None,
            vec![crate::tui::state::DialogItem::new("search", "Search", None)],
        ));

        assert_eq!(
            map_key_event(&state, key(KeyCode::Char('s'))),
            InputAction::DialogInsert('s')
        );
        assert_eq!(
            map_key_event(&state, key(KeyCode::Char(' '))),
            InputAction::DialogInsert(' ')
        );
        assert_eq!(
            map_paste_event(&state, "query".into()),
            InputAction::DialogPaste("query".into())
        );
    }

    #[test]
    fn space_remains_search_text_outside_the_primary_mcp_picker() {
        let mut state = TuiState::default();
        state.open_dialog(crate::tui::state::DialogState::new(
            DialogKind::ModelPicker,
            "Pick model",
            None,
            vec![crate::tui::state::DialogItem::new("gpt", "GPT", None)],
        ));

        assert_eq!(
            map_key_event(&state, key(KeyCode::Char(' '))),
            InputAction::DialogInsert(' ')
        );
    }

    #[test]
    fn primary_mcp_picker_space_maps_to_toggle() {
        let mut state = TuiState::default();
        state.open_dialog(crate::tui::state::DialogState::new(
            DialogKind::McpPicker,
            "MCP Servers",
            None,
            vec![crate::tui::state::DialogItem::new("docs", "docs", None)],
        ));

        assert_eq!(
            map_key_event(&state, key(KeyCode::Char(' '))),
            InputAction::DialogToggle
        );
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
    fn normal_composer_maps_history_and_cursor_keys() {
        let state = TuiState::default();

        assert_eq!(
            map_key_event(&state, key(KeyCode::Up)),
            InputAction::HistoryPrev
        );
        assert_eq!(
            map_key_event(&state, key(KeyCode::Down)),
            InputAction::HistoryNext
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
            InputAction::MoveCursorEnd
        );
        assert_eq!(
            map_key_event(&state, KeyEvent::new(KeyCode::End, KeyModifiers::CONTROL)),
            InputAction::ScrollToBottom
        );
        assert_eq!(
            map_key_event(&state, key(KeyCode::Left)),
            InputAction::MoveCursorLeft
        );
        assert_eq!(
            map_key_event(&state, key(KeyCode::Right)),
            InputAction::MoveCursorRight
        );
        assert_eq!(
            map_key_event(&state, key(KeyCode::Home)),
            InputAction::MoveCursorHome
        );
        assert_eq!(
            map_key_event(
                &state,
                KeyEvent::new(KeyCode::Char('e'), KeyModifiers::CONTROL)
            ),
            InputAction::MoveCursorEnd
        );
    }

    #[test]
    fn modified_enter_inserts_newline() {
        let state = TuiState::default();

        assert_eq!(
            map_key_event(&state, KeyEvent::new(KeyCode::Enter, KeyModifiers::SHIFT)),
            InputAction::InsertNewline
        );
    }

    #[test]
    fn idle_escape_is_an_interrupt_candidate_and_only_ctrl_c_quits() {
        let mut empty_state = TuiState::default();
        assert_eq!(
            map_key_event(&empty_state, key(KeyCode::Esc)),
            InputAction::Interrupt
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
            InputAction::ChildFirst
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
    fn ctrl_x_down_enters_first_child_navigation() {
        let mut state = TuiState::default();
        state.child_navigation_prefix = true;

        assert_eq!(
            map_key_event(&state, key(KeyCode::Down)),
            InputAction::ChildFirst
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
    fn child_view_command_entry_allows_typing_after_slash() {
        let mut state = TuiState::default();
        state.replace_child_timeline_from_records(
            &[],
            "parent-session",
            "child-session",
            "explorer",
            0,
            1,
            1,
        );

        assert!(apply_edit_action(&mut state, &InputAction::Insert('/')));
        assert_eq!(state.input_buffer, "/");
        assert_eq!(
            map_key_event(&state, key(KeyCode::Char('h'))),
            InputAction::Insert('h')
        );
        assert_eq!(
            map_key_event(&state, key(KeyCode::Char('l'))),
            InputAction::Insert('l')
        );
    }

    #[test]
    fn mouse_events_map_to_scroll_and_selection_actions() {
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
                    column: 5,
                    row: 10,
                    modifiers: KeyModifiers::NONE,
                }
            ),
            InputAction::MouseSelectionStart(5, 10)
        );
        assert_eq!(
            map_mouse_event(
                &state,
                MouseEvent {
                    kind: MouseEventKind::Drag(crossterm::event::MouseButton::Left),
                    column: 15,
                    row: 12,
                    modifiers: KeyModifiers::NONE,
                }
            ),
            InputAction::MouseSelectionDrag(15, 12)
        );
        assert_eq!(
            map_mouse_event(
                &state,
                MouseEvent {
                    kind: MouseEventKind::Up(crossterm::event::MouseButton::Left),
                    column: 20,
                    row: 15,
                    modifiers: KeyModifiers::NONE,
                }
            ),
            InputAction::MouseSelectionEnd(20, 15, false)
        );
        assert_eq!(
            map_mouse_event(
                &state,
                MouseEvent {
                    kind: MouseEventKind::Up(crossterm::event::MouseButton::Left),
                    column: 20,
                    row: 15,
                    modifiers: KeyModifiers::CONTROL,
                }
            ),
            InputAction::MouseSelectionEnd(20, 15, true)
        );
        assert_eq!(
            map_mouse_event(
                &state,
                MouseEvent {
                    kind: MouseEventKind::Up(crossterm::event::MouseButton::Left),
                    column: 20,
                    row: 15,
                    modifiers: KeyModifiers::SUPER,
                }
            ),
            InputAction::MouseSelectionEnd(20, 15, true)
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
    fn permission_prompt_maps_approve_deny_and_interrupt_actions() {
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
            InputAction::NoOp
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
            InputAction::Interrupt
        );
        assert_eq!(
            map_key_event(&state, key(KeyCode::Char('x'))),
            InputAction::NoOp
        );
    }

    #[test]
    fn permission_prompt_maps_always_only_when_request_allows_it() {
        let mut state = TuiState::default();
        let mut request = PermissionRequestEvent::new("call-1", "shell__exec", "cargo test");
        request.can_allow_always = true;
        state.pending_permission = Some(crate::tui::PermissionView::from_request(request));

        assert_eq!(
            map_key_event(&state, key(KeyCode::Char('a'))),
            InputAction::ApprovePermissionAlways
        );
        assert_eq!(
            map_key_event(&state, key(KeyCode::Char('o'))),
            InputAction::ApprovePermission
        );
    }

    #[test]
    fn pending_question_maps_navigation_and_selection_actions() {
        let mut state = TuiState::default();
        state.pending_question = Some(crate::tui::state::PendingQuestionState::new(
            crate::tool::QuestionRequest {
                questions: vec![
                    crate::tool::QuestionSpec {
                        question: "Choose one".into(),
                        header: "Mode".into(),
                        options: vec![crate::tool::QuestionOption {
                            label: "Fast".into(),
                            description: "Fast path".into(),
                        }],
                        multiple: false,
                    },
                    crate::tool::QuestionSpec {
                        question: "Choose tone".into(),
                        header: "Tone".into(),
                        options: vec![crate::tool::QuestionOption {
                            label: "Warm".into(),
                            description: "Warm path".into(),
                        }],
                        multiple: false,
                    },
                ],
            },
            None,
        ));

        assert_eq!(
            map_key_event(&state, key(KeyCode::Right)),
            InputAction::QuestionNextTab
        );
        assert_eq!(
            map_key_event(&state, key(KeyCode::Down)),
            InputAction::QuestionNextOption
        );
        assert_eq!(
            map_key_event(&state, key(KeyCode::Char('1'))),
            InputAction::QuestionPickOption(1)
        );
        assert_eq!(
            map_key_event(&state, key(KeyCode::Enter)),
            InputAction::QuestionActivate
        );
        assert_eq!(
            map_key_event(&state, key(KeyCode::Esc)),
            InputAction::QuestionCancel
        );
    }

    #[test]
    fn pending_question_confirm_tab_maps_enter_to_submit() {
        let mut state = TuiState::default();
        let mut question = crate::tui::state::PendingQuestionState::new(
            crate::tool::QuestionRequest {
                questions: vec![
                    crate::tool::QuestionSpec {
                        question: "Choose one".into(),
                        header: "Mode".into(),
                        options: vec![crate::tool::QuestionOption {
                            label: "Fast".into(),
                            description: "Fast path".into(),
                        }],
                        multiple: false,
                    },
                    crate::tool::QuestionSpec {
                        question: "Choose tone".into(),
                        header: "Tone".into(),
                        options: vec![crate::tool::QuestionOption {
                            label: "Warm".into(),
                            description: "Warm path".into(),
                        }],
                        multiple: false,
                    },
                ],
            },
            None,
        );
        question.active_tab = 2;
        state.pending_question = Some(question);

        assert_eq!(
            map_key_event(&state, key(KeyCode::Enter)),
            InputAction::QuestionSubmit
        );
    }

    #[test]
    fn pending_question_edit_mode_keeps_tab_keys_for_text_only() {
        let mut state = TuiState::default();
        let mut question = crate::tui::state::PendingQuestionState::new(
            crate::tool::QuestionRequest {
                questions: vec![crate::tool::QuestionSpec {
                    question: "Choose one".into(),
                    header: "Mode".into(),
                    options: vec![crate::tool::QuestionOption {
                        label: "Fast".into(),
                        description: "Fast path".into(),
                    }],
                    multiple: false,
                }],
            },
            None,
        );
        question.active_row = 1;
        question.begin_custom_edit();
        state.pending_question = Some(question);

        assert_eq!(
            map_key_event(&state, key(KeyCode::Left)),
            InputAction::QuestionMoveCursorLeft
        );
        assert_eq!(
            map_key_event(&state, key(KeyCode::Right)),
            InputAction::QuestionMoveCursorRight
        );
        assert_eq!(
            map_key_event(&state, key(KeyCode::Enter)),
            InputAction::QuestionActivate
        );
        assert_eq!(map_key_event(&state, key(KeyCode::Tab)), InputAction::NoOp);
    }

    #[test]
    fn pending_question_custom_edit_allows_clipboard_paste_shortcut() {
        let mut state = TuiState::default();
        let mut question = crate::tui::state::PendingQuestionState::new(
            crate::tool::QuestionRequest {
                questions: vec![crate::tool::QuestionSpec {
                    question: "Choose one".into(),
                    header: "Mode".into(),
                    options: vec![crate::tool::QuestionOption {
                        label: "Fast".into(),
                        description: "Fast path".into(),
                    }],
                    multiple: false,
                }],
            },
            None,
        );
        question.active_row = 1;
        question.begin_custom_edit();
        state.pending_question = Some(question);

        assert_eq!(
            map_key_event(
                &state,
                KeyEvent::new(KeyCode::Char('v'), KeyModifiers::CONTROL)
            ),
            InputAction::PasteFromClipboard
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
    fn pending_permission_blocks_child_navigation_shortcuts() {
        let mut state = TuiState::default();
        state.replace_child_timeline_from_records(
            &[],
            "parent-session",
            "child-session",
            "explorer",
            0,
            1,
            1,
        );
        state.pending_permission = Some(crate::tui::PermissionView::from_request(
            PermissionRequestEvent::new("call-1", "shell__exec", "ls"),
        ));
        state.child_navigation_prefix = true;

        assert_eq!(map_key_event(&state, key(KeyCode::Left)), InputAction::NoOp);
        assert_eq!(
            map_key_event(&state, KeyEvent::new(KeyCode::Right, KeyModifiers::ALT)),
            InputAction::NoOp
        );
        assert_eq!(
            map_key_event(
                &state,
                KeyEvent::new(KeyCode::Char('x'), KeyModifiers::CONTROL)
            ),
            InputAction::NoOp
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
