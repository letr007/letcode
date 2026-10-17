use crossterm::event::{KeyCode, KeyEvent, KeyModifiers, MouseEvent, MouseEventKind};

use super::help::HelpAction;
use super::state::{COMPOSER_ATTACHMENT_MARKER, DialogKind, TuiState};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InputAction {
    HistorianView(super::components::historian_report::ReportView),
    HistorianSources,
    Insert(char),
    Paste(String),
    PasteLongText(String),
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
    DialogToggleSessionScope,
    DialogAccept,
    DialogToggle,
    DialogDelete,
    DialogCancel,
    Help(HelpAction),
    ConfigEditInsert(char),
    ConfigEditPaste(String),
    ConfigEditBackspace,
    ConfigEditDelete,
    ConfigEditLeft,
    ConfigEditRight,
    ConfigEditHome,
    ConfigEditEnd,
    ConfigEditConfirm,
    ConfigEditCancel,
    ConfigSave,
    ConfigCollapse,
    ConfigListAppend,
    ConfigListRemove,
    SlashPanelNext,
    SlashPanelPrev,
    SlashPanelAccept,
    SlashPanelDismiss,
    Submit,
    RemoveLastQueuedPrompt,
    HistoryPrev,
    HistoryNext,
    ScrollUp,
    ScrollDown,
    ScrollPageUp,
    ScrollPageDown,
    ScrollToBottom,
    MouseScrollUp,
    MouseScrollDown,
    SidebarScrollUp,
    SidebarScrollDown,
    ToggleSidebarContext,
    ToggleSidebarMcp,
    ToggleSidebarTodos,
    CopySessionId,
    MouseSelectionStart(u16, u16),
    MouseSelectionDrag(u16, u16),
    /// Third flag is true when Ctrl/Cmd is held — used to activate underlined links.
    MouseSelectionEnd(u16, u16, bool),
    ScrollbarDragStart(u16, u16),
    ScrollbarDragMove(u16, u16),
    ScrollbarDragEnd,
    CopySelection,
    ClearSelection,
    CycleReasoningEffort,
    ToggleSidebar,
    ShowModel,
    ShowPermission,
    ShowReasoning,
    ShowThoughts,
    ShowAgents,
    ShowContext,
    ShowMcp,
    ShowSkills,
    ShowHelp,
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
    PermissionPrevOption,
    PermissionNextOption,
    PermissionActivate,
    ApprovePermission,
    ApprovePermissionAlways,
    DenyPermission,
    Interrupt,
    Quit,
    Tick,
    NoOp,
}

fn config_editor_key_action(state: &TuiState, key: KeyEvent) -> InputAction {
    if state
        .dialog()
        .is_some_and(|dialog| dialog.config_close_selected.is_some())
    {
        return match key.code {
            KeyCode::Up => InputAction::DialogPrev,
            KeyCode::Down => InputAction::DialogNext,
            KeyCode::Enter => InputAction::DialogAccept,
            KeyCode::Esc => InputAction::DialogCancel,
            _ => InputAction::NoOp,
        };
    }
    if state.config_edit.is_some() {
        return match key.code {
            KeyCode::Enter => InputAction::ConfigEditConfirm,
            KeyCode::Esc => InputAction::ConfigEditCancel,
            KeyCode::Backspace => InputAction::ConfigEditBackspace,
            KeyCode::Delete => InputAction::ConfigEditDelete,
            KeyCode::Left => InputAction::ConfigEditLeft,
            KeyCode::Right => InputAction::ConfigEditRight,
            KeyCode::Home => InputAction::ConfigEditHome,
            KeyCode::End => InputAction::ConfigEditEnd,
            KeyCode::Char(ch) if !has_non_shift_modifiers(key.modifiers) => {
                InputAction::ConfigEditInsert(ch)
            }
            _ => InputAction::NoOp,
        };
    }
    if let Some(dialog) = state
        .dialog()
        .filter(|dialog| dialog.config_expanded.is_some())
    {
        let editable_list = dialog.config_detail_target.is_none()
            && dialog
                .config_expanded
                .and_then(|index| dialog.config_fields.get(index))
                .is_some_and(|field| matches!(field, super::state::ConfigFieldRef::List(_)));
        return match key.code {
            KeyCode::Up => InputAction::DialogPrev,
            KeyCode::Down => InputAction::DialogNext,
            KeyCode::Enter => InputAction::DialogAccept,
            KeyCode::Left | KeyCode::Esc => InputAction::ConfigCollapse,
            KeyCode::Char('a') if editable_list => InputAction::ConfigListAppend,
            KeyCode::Char('d') | KeyCode::Delete if editable_list => InputAction::ConfigListRemove,
            KeyCode::Char('s') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                InputAction::ConfigSave
            }
            _ => InputAction::NoOp,
        };
    }
    match key.code {
        KeyCode::Up => InputAction::DialogPrev,
        KeyCode::Down => InputAction::DialogNext,
        KeyCode::Right | KeyCode::Enter => InputAction::DialogAccept,
        KeyCode::Esc => InputAction::DialogCancel,
        KeyCode::Char('s') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            InputAction::ConfigSave
        }
        KeyCode::Backspace => InputAction::DialogBackspace,
        KeyCode::Char(ch) if !has_non_shift_modifiers(key.modifiers) => {
            InputAction::DialogInsert(ch)
        }
        _ => InputAction::NoOp,
    }
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

    if state.pending_question.is_none()
        && state.pending_permission.is_none()
        && let Some(help) = state
            .dialog()
            .filter(|dialog| dialog.kind == DialogKind::Help)
            .and_then(|dialog| dialog.help.as_ref())
    {
        return match key.code {
            _ if has_non_shift_modifiers(key.modifiers) => InputAction::NoOp,
            KeyCode::Esc | KeyCode::Char('q') => InputAction::DialogCancel,
            KeyCode::Tab | KeyCode::BackTab => InputAction::Help(HelpAction::ToggleContents),
            KeyCode::Left => InputAction::Help(HelpAction::FocusContents),
            KeyCode::Right | KeyCode::Enter => InputAction::Help(HelpAction::FocusDocument),
            KeyCode::Up | KeyCode::Char('k') => InputAction::Help(if help.contents_focused {
                HelpAction::PreviousChapter
            } else {
                HelpAction::ScrollUp(1)
            }),
            KeyCode::Down | KeyCode::Char('j') => InputAction::Help(if help.contents_focused {
                HelpAction::NextChapter
            } else {
                HelpAction::ScrollDown(1)
            }),
            KeyCode::PageUp => InputAction::Help(HelpAction::PageUp),
            KeyCode::PageDown => InputAction::Help(HelpAction::PageDown),
            KeyCode::Home => InputAction::Help(if help.contents_focused {
                HelpAction::SelectChapter(0)
            } else {
                HelpAction::Top
            }),
            KeyCode::End => InputAction::Help(if help.contents_focused {
                HelpAction::SelectChapter(help.chapters.len().saturating_sub(1))
            } else {
                HelpAction::Bottom
            }),
            _ => InputAction::NoOp,
        };
    }

    if let Some(question) = state.pending_question.as_ref() {
        if question.editing_custom {
            return match key.code {
                _ if is_clipboard_paste_key(key) => InputAction::PasteFromClipboard,
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
    if is_clipboard_paste_key(key) {
        return InputAction::PasteFromClipboard;
    }

    if state.pending_permission.is_some() {
        // 审批选项横排：水平键移动高亮，Enter 确认；上下键留给 transcript 滚动。
        return match key.code {
            KeyCode::Left | KeyCode::Char('h') | KeyCode::Char('H') => {
                InputAction::PermissionPrevOption
            }
            KeyCode::Right | KeyCode::Char('l') | KeyCode::Char('L') => {
                InputAction::PermissionNextOption
            }
            KeyCode::Up | KeyCode::Char('k') | KeyCode::Char('K') => InputAction::ScrollUp,
            KeyCode::Down | KeyCode::Char('j') | KeyCode::Char('J') => InputAction::ScrollDown,
            KeyCode::PageUp => InputAction::ScrollPageUp,
            KeyCode::PageDown => InputAction::ScrollPageDown,
            KeyCode::End => InputAction::ScrollToBottom,
            KeyCode::Enter => InputAction::PermissionActivate,
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

    if ((key.modifiers.contains(KeyModifiers::CONTROL) && matches!(key.code, KeyCode::Char('x')))
        || matches!(key.code, KeyCode::Char('')))
        && !state.dialog_is_open()
        && !state.slash_panel_is_open()
    {
        return InputAction::ChildPrefix;
    }

    if state.is_historian_child_view()
        && state.input_buffer.is_empty()
        && !state.dialog_is_open()
        && !state.child_navigation_prefix
        && !has_non_shift_modifiers(key.modifiers)
    {
        use super::components::historian_report::ReportView;
        match key.code {
            KeyCode::Char('1') => return InputAction::HistorianView(ReportView::Compact),
            KeyCode::Char('2') => return InputAction::HistorianView(ReportView::Detailed),
            KeyCode::Char('3') => return InputAction::HistorianView(ReportView::Anchor),
            KeyCode::Char('0') => return InputAction::HistorianView(ReportView::Raw),
            KeyCode::Char('s') => return InputAction::HistorianSources,
            _ => {}
        }
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
            KeyCode::Char('b') | KeyCode::Char('B') => InputAction::ToggleSidebar,
            KeyCode::Char('m') | KeyCode::Char('M') => InputAction::ShowModel,
            KeyCode::Char('p') | KeyCode::Char('P') => InputAction::ShowPermission,
            KeyCode::Char('r') | KeyCode::Char('R') => InputAction::ShowReasoning,
            KeyCode::Char('t') | KeyCode::Char('T') => InputAction::ShowThoughts,
            KeyCode::Char('a') | KeyCode::Char('A') => InputAction::ShowAgents,
            KeyCode::Char('c') | KeyCode::Char('C') => InputAction::ShowContext,
            KeyCode::Char('i') | KeyCode::Char('I') => InputAction::ShowMcp,
            KeyCode::Char('s') | KeyCode::Char('S') => InputAction::ShowSkills,
            KeyCode::Char('h') | KeyCode::Char('H') | KeyCode::Char('?') => InputAction::ShowHelp,
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
        if state
            .dialog()
            .is_some_and(|dialog| dialog.kind == DialogKind::ConfigEditor)
        {
            return config_editor_key_action(state, key);
        }
        let search_dialog = state
            .dialog()
            .is_some_and(|dialog| dialog.kind.is_searchable());
        let session_picker = state
            .dialog()
            .is_some_and(|dialog| dialog.kind == DialogKind::SessionPicker);

        return if search_dialog {
            match key.code {
                KeyCode::Up => InputAction::DialogPrev,
                KeyCode::Down => InputAction::DialogNext,
                KeyCode::Left | KeyCode::Right if session_picker => {
                    InputAction::DialogToggleSessionScope
                }
                KeyCode::Enter => InputAction::DialogAccept,
                KeyCode::Char(' ')
                    if state.dialog().is_some_and(|dialog| {
                        matches!(
                            dialog.kind,
                            DialogKind::McpPicker | DialogKind::ExpertModelPicker(_)
                        )
                    }) =>
                {
                    InputAction::DialogToggle
                }
                KeyCode::Esc => InputAction::DialogCancel,
                KeyCode::Backspace => InputAction::DialogBackspace,
                KeyCode::Char('d')
                    if session_picker && key.modifiers.contains(KeyModifiers::CONTROL) =>
                {
                    InputAction::DialogDelete
                }
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

    if matches!(key.code, KeyCode::Backspace)
        && key.modifiers.contains(KeyModifiers::CONTROL)
        && !state.is_read_only_child_view()
    {
        return InputAction::RemoveLastQueuedPrompt;
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
    if matches!(key.code, KeyCode::Esc) && state.text_selection.is_some() {
        return InputAction::ClearSelection;
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

fn is_clipboard_paste_key(key: KeyEvent) -> bool {
    ((key.modifiers.contains(KeyModifiers::CONTROL) || key.modifiers.contains(KeyModifiers::SUPER))
        && matches!(key.code, KeyCode::Char('v' | 'V')))
        || (key.modifiers.contains(KeyModifiers::ALT)
            && matches!(key.code, KeyCode::Char('v' | 'V')))
        || (key.code == KeyCode::Insert && key.modifiers == KeyModifiers::SHIFT)
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
        InputAction::PasteLongText(text) => {
            let text = text.replace(COMPOSER_ATTACHMENT_MARKER, "");
            if text.is_empty() {
                false
            } else {
                state.add_composer_pasted_text(text);
                true
            }
        }
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
    if state.pending_question.is_none()
        && state.pending_permission.is_none()
        && state
            .dialog()
            .is_some_and(|dialog| dialog.kind == DialogKind::Help)
    {
        return InputAction::NoOp;
    }

    // Terminal paste commonly uses CR, while the system clipboard uses CRLF.
    // Normalize once per paste; embedded newlines never become Submit actions.
    let text = text.replace("\r\n", "\n").replace('\r', "\n");
    if state
        .pending_question
        .as_ref()
        .is_some_and(|question| question.editing_custom)
    {
        return InputAction::QuestionPaste(text);
    }

    if let Some(dialog) = state.dialog()
        && dialog.kind == DialogKind::ConfigEditor
    {
        if state.config_edit.is_some() {
            return InputAction::ConfigEditPaste(text);
        }
        if dialog.config_expanded.is_some() || dialog.config_close_selected.is_some() {
            return InputAction::NoOp;
        }
    }

    if state.dialog_is_open() {
        let search_dialog = state
            .dialog()
            .is_some_and(|dialog| dialog.kind.is_searchable());
        if search_dialog {
            return InputAction::DialogPaste(text);
        }
    }

    let text = text.replace(COMPOSER_ATTACHMENT_MARKER, "");
    if long_text_paste(&text) {
        InputAction::PasteLongText(text)
    } else {
        InputAction::Paste(text)
    }
}

fn long_text_paste(text: &str) -> bool {
    pasted_text_line_count(text) >= 3 || text.trim().chars().count() > 150
}

fn pasted_text_line_count(text: &str) -> usize {
    text.split('\n').count().max(1)
}

fn scrollbar_thumb_row(state: &TuiState, column: u16, row: u16) -> Option<usize> {
    let area = state.last_scrollbar_area;
    if column < area.x || column >= area.right() || row < area.y || row >= area.bottom() {
        return None;
    }
    let track_row = usize::from(row - area.y);
    state
        .transcript_scrollbar()
        .filter(|geometry| geometry.contains_row(track_row))
        .map(|_| track_row)
}

pub fn map_mouse_event(state: &TuiState, mouse: MouseEvent) -> InputAction {
    use crossterm::event::MouseButton;

    if state.pending_question.is_none()
        && state.pending_permission.is_none()
        && state
            .dialog()
            .is_some_and(|dialog| dialog.kind == DialogKind::Help)
    {
        return match mouse.kind {
            MouseEventKind::ScrollUp => InputAction::Help(HelpAction::ScrollUp(3)),
            MouseEventKind::ScrollDown => InputAction::Help(HelpAction::ScrollDown(3)),
            _ => InputAction::NoOp,
        };
    }

    let over_sidebar = state.sidebar_visible(state.last_terminal_width)
        && state.last_sidebar_area.width > 0
        && mouse.column >= state.last_sidebar_area.x
        && mouse.column < state.last_sidebar_area.right()
        && mouse.row >= state.last_sidebar_area.y
        && mouse.row < state.last_sidebar_area.bottom();

    match mouse.kind {
        MouseEventKind::Down(MouseButton::Left)
            if state
                .last_sidebar_context_header
                .contains((mouse.column, mouse.row).into()) =>
        {
            InputAction::ToggleSidebarContext
        }
        MouseEventKind::Down(MouseButton::Left)
            if state
                .last_sidebar_mcp_header
                .contains((mouse.column, mouse.row).into()) =>
        {
            InputAction::ToggleSidebarMcp
        }
        MouseEventKind::Down(MouseButton::Left)
            if state
                .last_sidebar_todos_header
                .contains((mouse.column, mouse.row).into()) =>
        {
            InputAction::ToggleSidebarTodos
        }
        MouseEventKind::Down(MouseButton::Left)
            if state
                .last_sidebar_session_id
                .contains((mouse.column, mouse.row).into()) =>
        {
            InputAction::CopySessionId
        }
        MouseEventKind::ScrollUp if over_sidebar => InputAction::SidebarScrollUp,
        MouseEventKind::ScrollDown if over_sidebar => InputAction::SidebarScrollDown,
        MouseEventKind::ScrollUp => InputAction::MouseScrollUp,
        MouseEventKind::ScrollDown => InputAction::MouseScrollDown,

        MouseEventKind::Down(MouseButton::Left)
            if scrollbar_thumb_row(state, mouse.column, mouse.row).is_some() =>
        {
            InputAction::ScrollbarDragStart(mouse.column, mouse.row)
        }
        MouseEventKind::Drag(MouseButton::Left) if state.transcript_scrollbar_drag.is_some() => {
            InputAction::ScrollbarDragMove(mouse.column, mouse.row)
        }
        MouseEventKind::Up(MouseButton::Left) if state.transcript_scrollbar_drag.is_some() => {
            InputAction::ScrollbarDragEnd
        }

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
#[path = "../../tests/internal/tui/input/tests.rs"]
mod tests;
