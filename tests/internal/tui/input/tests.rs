use super::*;
use crate::tui::{AppPhase, PermissionRequestEvent};
use crate::user_content::UserImageAttachment;
use crossterm::event::MouseButton;

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
fn paste_normalizes_line_endings_without_losing_blank_lines() {
    let state = TuiState::default();
    assert_eq!(
        map_paste_event(&state, "first\r\n\r\nsecond\rthird\n".into()),
        InputAction::PasteLongText("first\n\nsecond\nthird\n".into()),
    );
    assert_eq!(
        map_paste_event(&state, "a\rb".into()),
        InputAction::Paste("a\nb".into())
    );
}

#[test]
fn clipboard_shortcuts_accept_shift_insert_uppercase_v_and_alt_v() {
    let state = TuiState::default();
    for key in [
        KeyEvent::new(KeyCode::Insert, KeyModifiers::SHIFT),
        KeyEvent::new(
            KeyCode::Char('V'),
            KeyModifiers::CONTROL | KeyModifiers::SHIFT,
        ),
        KeyEvent::new(
            KeyCode::Char('v'),
            KeyModifiers::CONTROL | KeyModifiers::SHIFT,
        ),
        KeyEvent::new(KeyCode::Char('v'), KeyModifiers::ALT),
        KeyEvent::new(KeyCode::Char('V'), KeyModifiers::ALT),
    ] {
        assert_eq!(map_key_event(&state, key), InputAction::PasteFromClipboard);
    }
    assert_ne!(
        map_key_event(&state, KeyEvent::new(KeyCode::Insert, KeyModifiers::NONE)),
        InputAction::PasteFromClipboard
    );
    assert_ne!(
        map_key_event(&state, KeyEvent::new(KeyCode::Char('x'), KeyModifiers::ALT)),
        InputAction::PasteFromClipboard
    );
}

#[test]
fn backspace_keeps_composer_editing() {
    for phase in [AppPhase::Editing, AppPhase::Running] {
        let mut state = TuiState::default();
        state.phase = phase;
        state.input_buffer = "a🙂b".into();
        state.input_cursor = "a🙂".len();

        let action = map_key_event(&state, key(KeyCode::Backspace));
        assert_eq!(action, InputAction::Backspace);
        assert!(apply_edit_action(&mut state, &action));
        assert_eq!(state.input_buffer, "ab");
        assert_eq!(state.input_cursor, 1);
        assert_eq!(state.phase, phase);
    }

    let mut state = TuiState::default();
    let action = map_key_event(&state, key(KeyCode::Backspace));
    assert_eq!(action, InputAction::Backspace);
    assert!(!apply_edit_action(&mut state, &action));
    assert!(state.input_buffer.is_empty());
}

#[test]
fn ctrl_backspace_maps_to_queue_removal_without_editing_the_draft() {
    for phase in [AppPhase::Idle, AppPhase::Editing, AppPhase::Running] {
        for draft in ["", "draft"] {
            let mut state = TuiState::default();
            state.phase = phase;
            state.input_buffer = draft.into();
            state.input_cursor = draft.len();

            let action = map_key_event(
                &state,
                KeyEvent::new(KeyCode::Backspace, KeyModifiers::CONTROL),
            );
            assert_eq!(action, InputAction::RemoveLastQueuedPrompt);
            assert!(!apply_edit_action(&mut state, &action));
            assert_eq!(state.input_buffer, draft);
            assert_eq!(state.input_cursor, draft.len());
            assert_eq!(state.phase, phase);
        }
    }
}

#[test]
fn queue_removal_requires_control_backspace() {
    let state = TuiState::default();
    for modifiers in [KeyModifiers::NONE, KeyModifiers::SHIFT, KeyModifiers::SUPER] {
        assert_eq!(
            map_key_event(&state, KeyEvent::new(KeyCode::Backspace, modifiers)),
            InputAction::Backspace
        );
    }
    for event in [
        KeyEvent::new(KeyCode::Backspace, KeyModifiers::ALT),
        KeyEvent::new(
            KeyCode::Backspace,
            KeyModifiers::CONTROL | KeyModifiers::ALT,
        ),
        KeyEvent::new(KeyCode::Delete, KeyModifiers::CONTROL),
        KeyEvent::new(KeyCode::Char('h'), KeyModifiers::CONTROL),
        key(KeyCode::Char('\u{8}')),
    ] {
        assert_ne!(
            map_key_event(&state, event),
            InputAction::RemoveLastQueuedPrompt
        );
    }
}

#[test]
fn ctrl_backspace_preserves_prefix_and_read_only_child_editing() {
    let mut state = TuiState::default();
    state.child_navigation_prefix = true;
    for modifiers in [KeyModifiers::NONE, KeyModifiers::CONTROL] {
        assert_eq!(
            map_key_event(&state, KeyEvent::new(KeyCode::Backspace, modifiers)),
            InputAction::NoOp
        );
    }

    state.child_navigation_prefix = false;
    state.replace_child_timeline_from_records(
        &[],
        "parent-session",
        "child-session",
        "explorer",
        0,
        1,
        1,
    );
    assert!(state.is_read_only_child_view());
    for draft in ["", "/help"] {
        for modifiers in [KeyModifiers::NONE, KeyModifiers::CONTROL] {
            state.input_buffer = draft.into();
            state.input_cursor = draft.len();
            let action = map_key_event(&state, KeyEvent::new(KeyCode::Backspace, modifiers));
            assert_eq!(action, InputAction::Backspace);
            assert_eq!(apply_edit_action(&mut state, &action), !draft.is_empty());
            assert_eq!(
                state.input_buffer,
                if draft.is_empty() { "" } else { "/hel" }
            );
        }
    }
}

#[test]
fn ctrl_backspace_preserves_the_slash_completion_draft() {
    let mut state = TuiState::default();
    state.set_input("/");
    assert!(state.slash_panel_is_open());
    let action = map_key_event(
        &state,
        KeyEvent::new(KeyCode::Backspace, KeyModifiers::CONTROL),
    );
    assert_eq!(action, InputAction::RemoveLastQueuedPrompt);
    assert!(!apply_edit_action(&mut state, &action));
    assert_eq!(state.input_buffer, "/");
    assert_eq!(state.input_cursor, 1);
    assert!(state.slash_panel_is_open());
    assert_eq!(
        map_key_event(&state, key(KeyCode::Tab)),
        InputAction::SlashPanelAccept
    );

    let action = map_key_event(&state, key(KeyCode::Backspace));
    assert_eq!(action, InputAction::Backspace);
    assert!(apply_edit_action(&mut state, &action));
    assert!(state.input_buffer.is_empty());
    assert!(!state.slash_panel_is_open());
}

#[test]
fn long_text_paste_becomes_an_atomic_composer_token() {
    let state = TuiState::default();
    let pasted = "one\ntwo\nthree".to_string();
    assert_eq!(
        map_paste_event(&state, pasted.clone()),
        InputAction::PasteLongText(pasted.clone())
    );

    let mut state = TuiState::default();
    assert!(apply_edit_action(
        &mut state,
        &InputAction::PasteLongText(pasted.clone())
    ));
    assert_eq!(state.input_buffer, COMPOSER_ATTACHMENT_MARKER.to_string());
    assert_eq!(state.composer_content().text, pasted);
    assert!(apply_edit_action(&mut state, &InputAction::Backspace));
    assert!(state.input_buffer.is_empty());
    assert!(state.composer_tokens.is_empty());
}

#[test]
fn pasted_text_token_preserves_surrounding_text_order() {
    let mut state = TuiState::default();
    state.set_input("before after");
    state.input_cursor = "before ".len();
    assert!(apply_edit_action(
        &mut state,
        &InputAction::PasteLongText("one\ntwo\nthree".into())
    ));
    assert_eq!(state.composer_content().text, "before one\ntwo\nthreeafter");
}

#[test]
fn short_text_paste_remains_inline_text() {
    let state = TuiState::default();
    assert_eq!(
        map_paste_event(&state, "one\ntwo".into()),
        InputAction::Paste("one\ntwo".into())
    );
    assert!(matches!(
        map_paste_event(&state, "one\ntwo\n".into()),
        InputAction::PasteLongText(_)
    ));
}

#[test]
fn long_text_paste_strips_reserved_composer_markers() {
    let mut state = TuiState::default();
    assert!(apply_edit_action(
        &mut state,
        &InputAction::PasteLongText(format!("one\ntwo\nthree{COMPOSER_ATTACHMENT_MARKER}"))
    ));
    assert_eq!(state.composer_content().text, "one\ntwo\nthree");
    state.assert_composer_token_invariant();
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
fn ctrl_x_b_toggles_sidebar() {
    let mut state = TuiState::default();
    state.child_navigation_prefix = true;
    assert_eq!(
        map_key_event(&state, key(KeyCode::Char('b'))),
        InputAction::ToggleSidebar
    );
}

#[test]
fn ctrl_x_common_command_shortcuts_map_to_local_actions() {
    let mut state = TuiState::default();
    state.child_navigation_prefix = true;

    for (ch, action) in [
        ('m', InputAction::ShowModel),
        ('p', InputAction::ShowPermission),
        ('r', InputAction::ShowReasoning),
        ('t', InputAction::ShowThoughts),
        ('a', InputAction::ShowAgents),
        ('c', InputAction::ShowContext),
        ('i', InputAction::ShowMcp),
        ('s', InputAction::ShowSkills),
        ('h', InputAction::ShowHelp),
        ('?', InputAction::ShowHelp),
    ] {
        assert_eq!(map_key_event(&state, key(KeyCode::Char(ch))), action);
    }
}

fn help_state() -> TuiState {
    let mut state = TuiState::default();
    let mut dialog =
        crate::tui::state::DialogState::new(DialogKind::Help, "Help", None, Vec::new());
    dialog.help = Some(crate::tui::help::HelpState::new(state.language()));
    state.open_dialog(dialog);
    state
}

#[test]
fn help_keys_follow_contents_and_document_focus() {
    let mut state = help_state();
    for contents_focused in [true, false] {
        state
            .dialog_mut()
            .unwrap()
            .help
            .as_mut()
            .unwrap()
            .contents_focused = contents_focused;
        for (code, action) in [
            (KeyCode::Esc, InputAction::DialogCancel),
            (KeyCode::Char('q'), InputAction::DialogCancel),
            (KeyCode::Tab, InputAction::Help(HelpAction::ToggleContents)),
            (
                KeyCode::BackTab,
                InputAction::Help(HelpAction::ToggleContents),
            ),
            (KeyCode::Left, InputAction::Help(HelpAction::FocusContents)),
            (KeyCode::Right, InputAction::Help(HelpAction::FocusDocument)),
            (KeyCode::Enter, InputAction::Help(HelpAction::FocusDocument)),
            (KeyCode::PageUp, InputAction::Help(HelpAction::PageUp)),
            (KeyCode::PageDown, InputAction::Help(HelpAction::PageDown)),
            (
                KeyCode::Home,
                InputAction::Help(if contents_focused {
                    HelpAction::SelectChapter(0)
                } else {
                    HelpAction::Top
                }),
            ),
            (
                KeyCode::End,
                InputAction::Help(if contents_focused {
                    HelpAction::SelectChapter(
                        state
                            .dialog()
                            .unwrap()
                            .help
                            .as_ref()
                            .unwrap()
                            .chapters
                            .len()
                            - 1,
                    )
                } else {
                    HelpAction::Bottom
                }),
            ),
        ] {
            assert_eq!(map_key_event(&state, key(code)), action);
        }
        for code in [KeyCode::Up, KeyCode::Char('k')] {
            assert_eq!(
                map_key_event(&state, key(code)),
                InputAction::Help(if contents_focused {
                    HelpAction::PreviousChapter
                } else {
                    HelpAction::ScrollUp(1)
                })
            );
        }
        for code in [KeyCode::Down, KeyCode::Char('j')] {
            assert_eq!(
                map_key_event(&state, key(code)),
                InputAction::Help(if contents_focused {
                    HelpAction::NextChapter
                } else {
                    HelpAction::ScrollDown(1)
                })
            );
        }
    }
}

#[test]
fn help_blocks_composer_keys_and_paste_but_preserves_ctrl_c() {
    let state = help_state();
    for event in [
        key(KeyCode::Char('x')),
        key(KeyCode::Backspace),
        key(KeyCode::Delete),
        KeyEvent::new(KeyCode::Char('q'), KeyModifiers::CONTROL),
        KeyEvent::new(KeyCode::Char('t'), KeyModifiers::CONTROL),
        KeyEvent::new(KeyCode::Char('x'), KeyModifiers::CONTROL),
        KeyEvent::new(KeyCode::Char('v'), KeyModifiers::CONTROL),
        KeyEvent::new(KeyCode::Char('v'), KeyModifiers::SUPER),
        KeyEvent::new(KeyCode::Insert, KeyModifiers::SHIFT),
    ] {
        assert_eq!(map_key_event(&state, event), InputAction::NoOp);
    }
    assert_eq!(
        map_paste_event(&state, "pasted\ntext".into()),
        InputAction::NoOp
    );
    assert_eq!(
        map_key_event(
            &state,
            KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL)
        ),
        InputAction::Quit
    );
}

#[test]
fn help_keys_take_precedence_over_read_only_child_navigation() {
    let mut state = help_state();
    let dialog = state.dialog().unwrap().clone();
    state.replace_child_timeline_from_records(
        &[],
        "parent-session",
        "child-session",
        "explorer",
        0,
        1,
        1,
    );
    state.open_dialog(dialog);
    assert!(state.is_read_only_child_view());
    assert!(state.input_buffer.is_empty());
    assert_eq!(
        map_key_event(&state, key(KeyCode::Left)),
        InputAction::Help(HelpAction::FocusContents)
    );
    assert_eq!(
        map_key_event(&state, key(KeyCode::Right)),
        InputAction::Help(HelpAction::FocusDocument)
    );
    state
        .dialog_mut()
        .unwrap()
        .help
        .as_mut()
        .unwrap()
        .contents_focused = true;
    assert_eq!(
        map_key_event(&state, key(KeyCode::Up)),
        InputAction::Help(HelpAction::PreviousChapter)
    );
    state
        .dialog_mut()
        .unwrap()
        .help
        .as_mut()
        .unwrap()
        .contents_focused = false;
    for code in [KeyCode::Up, KeyCode::Char('k')] {
        assert_eq!(
            map_key_event(&state, key(code)),
            InputAction::Help(HelpAction::ScrollUp(1))
        );
    }
    for code in [KeyCode::Down, KeyCode::Char('j')] {
        assert_eq!(
            map_key_event(&state, key(code)),
            InputAction::Help(HelpAction::ScrollDown(1))
        );
    }
}

#[test]
fn help_mouse_events_scroll_the_document_and_block_background_interaction() {
    let state = help_state();
    for (kind, action) in [
        (
            MouseEventKind::ScrollUp,
            InputAction::Help(HelpAction::ScrollUp(3)),
        ),
        (
            MouseEventKind::ScrollDown,
            InputAction::Help(HelpAction::ScrollDown(3)),
        ),
        (MouseEventKind::Down(MouseButton::Left), InputAction::NoOp),
        (MouseEventKind::Drag(MouseButton::Left), InputAction::NoOp),
        (MouseEventKind::Up(MouseButton::Left), InputAction::NoOp),
        (MouseEventKind::Moved, InputAction::NoOp),
    ] {
        assert_eq!(
            map_mouse_event(
                &state,
                MouseEvent {
                    kind,
                    column: 1,
                    row: 1,
                    modifiers: KeyModifiers::NONE,
                }
            ),
            action
        );
    }
}

#[test]
fn ctrl_x_prefix_preserves_child_navigation_arrows() {
    let mut state = TuiState::default();
    state.child_navigation_prefix = true;

    assert_eq!(
        map_key_event(&state, key(KeyCode::Down)),
        InputAction::ChildFirst
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
        map_key_event(&state, key(KeyCode::Up)),
        InputAction::ChildParent
    );
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
fn config_editor_keys_and_paste_follow_the_current_mode() {
    use crate::tui::state::{ConfigEditState, ConfigFieldRef, DialogState};

    let mut state = TuiState::default();
    let mut dialog = DialogState::new(DialogKind::ConfigEditor, "Configuration", None, vec![]);
    dialog.config_fields = vec![ConfigFieldRef::Choice(vec!["protocol".into()])];
    dialog.config_expanded = Some(0);
    state.open_dialog(dialog);
    for modifiers in [KeyModifiers::NONE, KeyModifiers::CONTROL] {
        assert_eq!(
            map_key_event(&state, KeyEvent::new(KeyCode::Backspace, modifiers)),
            InputAction::NoOp
        );
    }
    assert_eq!(
        map_key_event(&state, key(KeyCode::Char('a'))),
        InputAction::NoOp
    );
    assert_eq!(
        map_key_event(&state, key(KeyCode::Char('d'))),
        InputAction::NoOp
    );
    assert_eq!(map_paste_event(&state, "text".into()), InputAction::NoOp);
    assert_eq!(
        map_key_event(
            &state,
            KeyEvent::new(KeyCode::Char('s'), KeyModifiers::CONTROL)
        ),
        InputAction::ConfigSave
    );

    state.dialog_mut().expect("dialog").config_fields[0] =
        ConfigFieldRef::List(vec!["command".into()]);
    assert_eq!(
        map_key_event(&state, key(KeyCode::Char('a'))),
        InputAction::ConfigListAppend
    );
    assert_eq!(
        map_key_event(&state, key(KeyCode::Delete)),
        InputAction::ConfigListRemove
    );
    state.dialog_mut().expect("dialog").config_detail_target = Some(0);
    assert_eq!(
        map_key_event(&state, key(KeyCode::Char('a'))),
        InputAction::NoOp
    );
    assert_eq!(
        map_key_event(&state, key(KeyCode::Delete)),
        InputAction::NoOp
    );

    state.config_edit = Some(ConfigEditState {
        field: ConfigFieldRef::ListItem(vec!["command".into()], 0),
        buffer: String::new(),
        cursor: 0,
    });
    for modifiers in [KeyModifiers::NONE, KeyModifiers::CONTROL] {
        assert_eq!(
            map_key_event(&state, KeyEvent::new(KeyCode::Backspace, modifiers)),
            InputAction::ConfigEditBackspace
        );
    }
    assert_eq!(
        map_key_event(&state, key(KeyCode::Enter)),
        InputAction::ConfigEditConfirm
    );
    assert_eq!(
        map_key_event(&state, key(KeyCode::Esc)),
        InputAction::ConfigEditCancel
    );
    assert_eq!(map_key_event(&state, key(KeyCode::Down)), InputAction::NoOp);
    assert_eq!(
        map_key_event(
            &state,
            KeyEvent::new(KeyCode::Char('s'), KeyModifiers::CONTROL)
        ),
        InputAction::NoOp
    );
    assert_eq!(
        map_paste_event(&state, "text".into()),
        InputAction::ConfigEditPaste("text".into())
    );

    state.config_edit = None;
    state.dialog_mut().expect("dialog").config_expanded = None;
    state.dialog_mut().expect("dialog").config_close_selected = Some(0);
    for modifiers in [KeyModifiers::NONE, KeyModifiers::CONTROL] {
        assert_eq!(
            map_key_event(&state, KeyEvent::new(KeyCode::Backspace, modifiers)),
            InputAction::NoOp
        );
    }
    assert_eq!(
        map_key_event(&state, key(KeyCode::Enter)),
        InputAction::DialogAccept
    );
    assert_eq!(
        map_key_event(&state, key(KeyCode::Esc)),
        InputAction::DialogCancel
    );
    assert_eq!(
        map_key_event(&state, key(KeyCode::Down)),
        InputAction::DialogNext
    );
    assert_eq!(
        map_key_event(&state, key(KeyCode::Char('a'))),
        InputAction::NoOp
    );
    assert_eq!(map_paste_event(&state, "text".into()), InputAction::NoOp);
}

#[test]
fn expert_model_picker_uses_space_to_toggle_and_enter_or_escape_to_navigate() {
    let mut state = TuiState::default();
    state.open_dialog(crate::tui::state::DialogState::new(
        DialogKind::ExpertModelPicker("explorer".into()),
        "Expert models",
        None,
        vec![crate::tui::state::DialogItem::new(
            "provider/model",
            "Model",
            None,
        )],
    ));

    assert_eq!(
        map_key_event(&state, key(KeyCode::Char(' '))),
        InputAction::DialogToggle
    );
    assert_eq!(
        map_key_event(&state, key(KeyCode::Enter)),
        InputAction::DialogAccept
    );
    assert_eq!(
        map_key_event(&state, key(KeyCode::Esc)),
        InputAction::DialogCancel
    );
}

#[test]
fn dialog_searchability_is_shared_by_key_and_paste_mapping() {
    let cases = [
        (DialogKind::ModelPicker, true),
        (DialogKind::ExpertModelPicker("explorer".into()), true),
        (DialogKind::SessionPicker, true),
        (DialogKind::HistoryTree, true),
        (DialogKind::ContextPicker, true),
        (DialogKind::McpPicker, true),
        (DialogKind::McpToolsPicker, true),
        (DialogKind::SkillPicker, true),
        (DialogKind::AgentPicker, false),
        (DialogKind::PermissionPicker, false),
        (DialogKind::ReasoningPicker, false),
        (DialogKind::ThoughtsPicker, false),
        (DialogKind::ToolsPicker, false),
        (DialogKind::ThemePicker, false),
        (DialogKind::FakePicker, false),
        (DialogKind::LanguagePicker, false),
        (DialogKind::ConfigEditor, true),
        (DialogKind::ContextDetail, false),
    ];

    for (kind, searchable) in cases {
        let mut state = TuiState::default();
        state.open_dialog(crate::tui::state::DialogState::new(
            kind,
            "Dialog",
            None,
            vec![crate::tui::state::DialogItem::new("item", "Item", None)],
        ));

        assert_eq!(
            map_key_event(&state, key(KeyCode::Char('x'))),
            if searchable {
                InputAction::DialogInsert('x')
            } else {
                InputAction::NoOp
            }
        );
        for modifiers in [KeyModifiers::NONE, KeyModifiers::CONTROL] {
            assert_eq!(
                map_key_event(&state, KeyEvent::new(KeyCode::Backspace, modifiers)),
                if searchable {
                    InputAction::DialogBackspace
                } else {
                    InputAction::NoOp
                }
            );
        }
        assert_eq!(
            map_paste_event(&state, "x".into()),
            if searchable {
                InputAction::DialogPaste("x".into())
            } else {
                InputAction::Paste("x".into())
            }
        );
    }
}

#[test]
fn session_picker_horizontal_keys_switch_scope() {
    let mut state = TuiState::default();
    state.open_dialog(crate::tui::state::DialogState::new(
        DialogKind::SessionPicker,
        "Sessions",
        None,
        vec![crate::tui::state::DialogItem::new(
            "session-1",
            "Work",
            None,
        )],
    ));

    assert_eq!(
        map_key_event(&state, key(KeyCode::Left)),
        InputAction::DialogToggleSessionScope
    );
    assert_eq!(
        map_key_event(&state, key(KeyCode::Right)),
        InputAction::DialogToggleSessionScope
    );
}

#[test]
fn session_picker_ctrl_d_requests_delete() {
    let mut state = TuiState::default();
    state.open_dialog(crate::tui::state::DialogState::new(
        DialogKind::SessionPicker,
        "Sessions",
        None,
        vec![crate::tui::state::DialogItem::new(
            "session-1",
            "Work",
            None,
        )],
    ));

    assert_eq!(
        map_key_event(
            &state,
            KeyEvent::new(KeyCode::Char('d'), KeyModifiers::CONTROL)
        ),
        InputAction::DialogDelete
    );
}

#[test]
fn other_searchable_pickers_leave_horizontal_keys_unbound() {
    for kind in [
        DialogKind::ModelPicker,
        DialogKind::HistoryTree,
        DialogKind::ContextPicker,
    ] {
        let mut state = TuiState::default();
        state.open_dialog(crate::tui::state::DialogState::new(
            kind.clone(),
            "Dialog",
            None,
            vec![crate::tui::state::DialogItem::new("item", "Item", None)],
        ));

        assert_eq!(
            map_key_event(&state, key(KeyCode::Left)),
            InputAction::NoOp,
            "{kind:?} must not swallow the horizontal keys"
        );
        assert_eq!(
            map_key_event(&state, key(KeyCode::Right)),
            InputAction::NoOp,
            "{kind:?} must not swallow the horizontal keys"
        );
    }
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
fn enter_maps_to_submit_when_not_waiting_for_permission() {
    let state = TuiState::default();

    assert_eq!(
        map_key_event(&state, key(KeyCode::Enter)),
        InputAction::Submit
    );
}

#[test]
fn permission_prompt_maps_approve_deny_and_interrupt_actions() {
    let mut state = TuiState::default();
    state.pending_permission = Some(crate::tui::PermissionView::from_request(
        PermissionRequestEvent::new("call-1", "shell__exec", "ls"),
    ));
    for modifiers in [KeyModifiers::NONE, KeyModifiers::CONTROL] {
        assert_eq!(
            map_key_event(&state, KeyEvent::new(KeyCode::Backspace, modifiers)),
            InputAction::NoOp
        );
    }

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
fn permission_prompt_maps_horizontal_keys_to_highlight_movement() {
    let mut state = TuiState::default();
    state.pending_permission = Some(crate::tui::PermissionView::from_request(
        PermissionRequestEvent::new("call-1", "shell__exec", "ls"),
    ));

    assert_eq!(
        map_key_event(&state, key(KeyCode::Left)),
        InputAction::PermissionPrevOption
    );
    assert_eq!(
        map_key_event(&state, key(KeyCode::Char('h'))),
        InputAction::PermissionPrevOption
    );
    assert_eq!(
        map_key_event(&state, key(KeyCode::Right)),
        InputAction::PermissionNextOption
    );
    assert_eq!(
        map_key_event(&state, key(KeyCode::Char('l'))),
        InputAction::PermissionNextOption
    );
    assert_eq!(
        map_key_event(&state, key(KeyCode::Enter)),
        InputAction::PermissionActivate
    );
}

#[test]
fn permission_prompt_keeps_vertical_keys_scrolling_the_transcript() {
    let mut state = TuiState::default();
    state.pending_permission = Some(crate::tui::PermissionView::from_request(
        PermissionRequestEvent::new("call-1", "shell__exec", "ls"),
    ));

    assert_eq!(
        map_key_event(&state, key(KeyCode::Up)),
        InputAction::ScrollUp
    );
    assert_eq!(
        map_key_event(&state, key(KeyCode::Char('k'))),
        InputAction::ScrollUp
    );
    assert_eq!(
        map_key_event(&state, key(KeyCode::Down)),
        InputAction::ScrollDown
    );
    assert_eq!(
        map_key_event(&state, key(KeyCode::Char('j'))),
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
    for modifiers in [KeyModifiers::NONE, KeyModifiers::CONTROL] {
        assert_eq!(
            map_key_event(&state, KeyEvent::new(KeyCode::Backspace, modifiers)),
            InputAction::NoOp
        );
    }

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
fn pending_question_confirm_tab_maps_vertical_keys_to_scroll_actions() {
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
                multiple: true,
            }],
        },
        None,
    );
    question.active_tab = 1;
    state.pending_question = Some(question);

    assert_eq!(
        map_key_event(&state, key(KeyCode::Up)),
        InputAction::QuestionPrevOption
    );
    assert_eq!(
        map_key_event(&state, key(KeyCode::Char('j'))),
        InputAction::QuestionNextOption
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
    question.active_row.focus(1);
    question.begin_custom_edit();
    state.pending_question = Some(question);
    for modifiers in [KeyModifiers::NONE, KeyModifiers::CONTROL] {
        assert_eq!(
            map_key_event(&state, KeyEvent::new(KeyCode::Backspace, modifiers)),
            InputAction::QuestionBackspace
        );
    }

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
    question.active_row.focus(1);
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
fn mouse_on_the_scrollbar_thumb_drags_instead_of_selecting() {
    let mut state = TuiState::default();
    state.set_transcript_scrollbar_for_test(
        ratatui::layout::Rect::new(40, 0, 1, 10),
        ratatui::layout::Rect::new(0, 0, 40, 10),
        100,
        0,
    );
    let mouse = |kind, column, row| MouseEvent {
        kind,
        column,
        row,
        modifiers: KeyModifiers::NONE,
    };

    assert_eq!(
        map_mouse_event(
            &state,
            mouse(MouseEventKind::Down(MouseButton::Left), 40, 0)
        ),
        InputAction::ScrollbarDragStart(40, 0)
    );
    assert_eq!(
        map_mouse_event(
            &state,
            mouse(MouseEventKind::Down(MouseButton::Left), 40, 5)
        ),
        InputAction::MouseSelectionStart(40, 5)
    );
    assert_eq!(
        map_mouse_event(
            &state,
            mouse(MouseEventKind::Down(MouseButton::Left), 39, 0)
        ),
        InputAction::MouseSelectionStart(39, 0)
    );

    state.transcript_scrollbar_drag = Some(4);
    assert_eq!(
        map_mouse_event(
            &state,
            mouse(MouseEventKind::Drag(MouseButton::Left), 40, 7)
        ),
        InputAction::ScrollbarDragMove(40, 7)
    );
    assert_eq!(
        map_mouse_event(&state, mouse(MouseEventKind::Up(MouseButton::Left), 40, 7)),
        InputAction::ScrollbarDragEnd
    );
}

#[test]
fn sidebar_session_id_click_routes_to_copy_action() {
    let mut state = TuiState::default();
    state.last_sidebar_session_id = ratatui::layout::Rect::new(103, 2, 35, 1);

    let mouse = MouseEvent {
        kind: MouseEventKind::Down(MouseButton::Left),
        column: 110,
        row: 2,
        modifiers: KeyModifiers::NONE,
    };
    assert_eq!(map_mouse_event(&state, mouse), InputAction::CopySessionId);
}

#[test]
fn sidebar_section_header_click_routes_to_toggle_action() {
    let mut state = TuiState::default();
    state.last_terminal_width = 140;
    state.last_sidebar_area = ratatui::layout::Rect::new(100, 1, 40, 20);
    state.last_sidebar_context_header = ratatui::layout::Rect::new(103, 7, 35, 1);
    state.last_sidebar_mcp_header = ratatui::layout::Rect::new(103, 12, 35, 1);
    state.last_sidebar_todos_header = ratatui::layout::Rect::new(103, 16, 35, 1);

    for (row, expected) in [
        (7, InputAction::ToggleSidebarContext),
        (12, InputAction::ToggleSidebarMcp),
        (16, InputAction::ToggleSidebarTodos),
    ] {
        let mouse = MouseEvent {
            kind: MouseEventKind::Down(MouseButton::Left),
            column: 110,
            row,
            modifiers: KeyModifiers::NONE,
        };
        assert_eq!(map_mouse_event(&state, mouse), expected);
    }
}

#[test]
fn mouse_wheel_over_sidebar_routes_to_sidebar_scroll() {
    let mut state = TuiState::default();
    state.last_terminal_width = 140;
    state.last_sidebar_area = ratatui::layout::Rect::new(100, 1, 40, 20);
    let mouse = MouseEvent {
        kind: MouseEventKind::ScrollDown,
        column: 120,
        row: 10,
        modifiers: KeyModifiers::NONE,
    };

    assert_eq!(
        map_mouse_event(&state, mouse),
        InputAction::SidebarScrollDown
    );
}

#[test]
fn scroll_actions_still_work_while_permission_prompt_is_pending() {
    let mut state = TuiState::default();
    state.pending_permission = Some(crate::tui::PermissionView::from_request(
        PermissionRequestEvent::new("call-1", "shell__exec", "ls"),
    ));

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

    assert_eq!(
        map_key_event(&state, key(KeyCode::Left)),
        InputAction::PermissionPrevOption
    );
    assert_eq!(
        map_key_event(&state, key(KeyCode::Char('m'))),
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
