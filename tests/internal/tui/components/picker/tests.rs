use super::*;
use ratatui::{Terminal, backend::TestBackend};

#[test]
fn mcp_tools_picker_hides_raw_offline_diagnostics() {
    let theme = Theme::dark();
    let area = Rect::new(0, 0, 100, 30);
    let diagnostic = "Offline · connection refused at https://private.example";
    let dialog = DialogState::new(
        DialogKind::McpToolsPicker,
        "Tools · local",
        Some(diagnostic.into()),
        Vec::new(),
    );
    let mut state = TuiState::default();
    state.set_language(Some(crate::tui::i18n::Language::En));
    let backend = TestBackend::new(area.width, area.height);
    let mut terminal = Terminal::new(backend).expect("terminal");

    terminal
        .draw(|frame| render_picker(frame, &mut state, area, theme, &dialog))
        .expect("draw");

    let rendered = terminal
        .backend()
        .buffer()
        .content()
        .iter()
        .map(|cell| cell.symbol())
        .collect::<String>();
    assert!(rendered.contains("Offline"));
    assert!(!rendered.contains("connection refused"));
    assert!(!rendered.contains("private.example"));
    assert!(rendered.contains("No tools discovered for this server"));
    assert!(rendered.contains("Esc back"));
}

#[test]
fn visible_entries_keep_the_selected_expert_inside_a_height_limited_window() {
    let items = (0..6)
        .map(|index| {
            DialogItem::new(format!("expert-{index}"), format!("expert-{index}"), None)
                .with_section("Experts")
                .with_right_detail("deepseek/deepseek-v4 · zhipu/glm-5")
        })
        .collect();
    let mut dialog = DialogState::new(DialogKind::AgentPicker, "Experts", None, items);
    dialog.selected = 5;

    let entries = visible_picker_entries(&dialog, 6, 62);
    let rows: u16 = entries.iter().map(PickerEntry::rows).sum();
    assert!(rows <= 6, "{rows}");
    assert!(entries.iter().any(|entry| {
        matches!(entry, PickerEntry::Item { index, .. } if *index == dialog.selected)
    }));
}

#[test]
fn agent_picker_wraps_long_model_lists_without_truncating() {
    let theme = Theme::dark();
    let area = Rect::new(0, 0, PICKER_MIN_WIDTH, 30);
    let dialog = DialogState::new(
        DialogKind::AgentPicker,
        "Experts",
        None,
        vec![
            DialogItem::new("explorer", "explorer", None)
                .with_section("Experts")
                .with_right_detail("deepseek/deepseek-v4 · zhipu/glm-5 · moonshot/kimi-k2"),
        ],
    );
    let mut state = TuiState::default();
    state.set_language(Some(crate::tui::i18n::Language::En));
    let backend = TestBackend::new(area.width, area.height);
    let mut terminal = Terminal::new(backend).expect("terminal");

    terminal
        .draw(|frame| render_picker(frame, &mut state, area, theme, &dialog))
        .expect("draw");

    let rendered = terminal
        .backend()
        .buffer()
        .content()
        .iter()
        .map(|cell| cell.symbol())
        .collect::<String>();
    assert!(rendered.contains("explorer"), "{rendered}");
    assert!(rendered.contains("deepseek/deepseek-v4"), "{rendered}");
    assert!(rendered.contains("zhipu/glm-5"), "{rendered}");
    assert!(rendered.contains("moonshot/kimi-k2"), "{rendered}");
    assert!(!rendered.contains('…'), "{rendered}");
}

fn config_state(field: ConfigFieldRef) -> TuiState {
    let mut state = TuiState::default();
    state.set_language(Some(crate::tui::i18n::Language::En));
    let mut dialog = DialogState::new(
        DialogKind::ConfigEditor,
        "Configuration",
        None,
        vec![DialogItem::new("field", "Field", Some("value".into())).with_section("Settings")],
    );
    dialog.config_fields.push(field);
    state.open_dialog(dialog);
    state
}

fn config_buffer(state: &mut TuiState, width: u16, height: u16) -> ratatui::buffer::Buffer {
    let mut terminal = Terminal::new(TestBackend::new(width, height)).expect("terminal");
    let dialog = state.dialog().expect("config dialog").clone();
    let theme = state.theme();
    terminal
        .draw(|frame| render_picker(frame, state, frame.area(), theme, &dialog))
        .expect("draw");
    terminal.backend().buffer().clone()
}

fn buffer_rows(buffer: &ratatui::buffer::Buffer) -> Vec<String> {
    (buffer.area.y..buffer.area.bottom())
        .map(|y| {
            let mut row = String::new();
            let mut x = buffer.area.x;
            while x < buffer.area.right() {
                let symbol = buffer[(x, y)].symbol();
                row.push_str(symbol);
                x += display_width(symbol).max(1) as u16;
            }
            row
        })
        .collect()
}

fn config_footer_text(state: &TuiState, width: u16) -> String {
    let mut terminal = Terminal::new(TestBackend::new(width, 1)).expect("terminal");
    terminal
        .draw(|frame| render_config_footer(frame, frame.area(), state.theme(), state))
        .expect("draw");
    buffer_rows(terminal.backend().buffer()).join("\n")
}

#[test]
fn config_render_detail_keeps_selection_and_editing_visible() {
    for (width, height) in [(100, 24), (40, 14)] {
        let path = vec!["items".into()];
        let mut state = config_state(ConfigFieldRef::List(path.clone()));
        let dialog = state.dialog_mut().expect("config dialog");
        dialog.config_expanded = Some(0);
        dialog.config_detail_items = (0..32)
            .map(|index| DialogItem::new(index.to_string(), format!("item-{index:02}"), None))
            .collect();
        dialog.config_detail_selected = 31;
        let rows = buffer_rows(&config_buffer(&mut state, width, height));
        assert!(rows.iter().any(|row| row.contains("● item-31")), "{rows:?}");
        state.config_edit = Some(ConfigEditState {
            field: ConfigFieldRef::ListItem(path, 31),
            buffer: "EDITED-LAST".into(),
            cursor: "EDITED-LAST".len(),
        });
        let rows = buffer_rows(&config_buffer(&mut state, width, height));
        assert!(
            rows.iter().any(|row| row.contains("EDITED-LAST▏")),
            "{rows:?}"
        );
    }
}

#[test]
fn config_render_unicode_edit_keeps_the_cursor_and_input_visible() {
    for list_item in [false, true] {
        for width in [40, 100] {
            let path = vec!["value".into()];
            let field = if list_item {
                ConfigFieldRef::List(path.clone())
            } else {
                ConfigFieldRef::Field(path.clone())
            };
            let mut state = config_state(field.clone());
            state.dialog_mut().expect("config dialog").items[0].label =
                "A very long field name that must not hide the input cursor".into();
            let edit_field = if list_item {
                let dialog = state.dialog_mut().expect("config dialog");
                dialog.config_expanded = Some(0);
                dialog.config_detail_items = vec![DialogItem::new("0", "item", None)];
                ConfigFieldRef::ListItem(path, 0)
            } else {
                field
            };
            let buffer = format!("{}新输入TAIL", "中🙂e\u{301}👩‍💻".repeat(30));
            state.config_edit = Some(ConfigEditState {
                field: edit_field,
                cursor: buffer.len() - "TAIL".len(),
                buffer,
            });
            let rows = buffer_rows(&config_buffer(&mut state, width, 24));
            assert!(
                rows.iter().any(|row| row.contains("新输入▏TAIL")),
                "{width}: {rows:?}"
            );
            let edit = state.config_edit.as_mut().expect("edit");
            edit.cursor = edit.buffer.len();
            let rows = buffer_rows(&config_buffer(&mut state, width, 24));
            assert!(
                rows.iter().any(|row| row.contains("TAIL▏")),
                "{width}: {rows:?}"
            );
        }
    }
}

#[test]
fn config_render_text_footer_only_confirms_or_cancels() {
    for field in [
        ConfigFieldRef::Field(vec!["prompt".into()]),
        ConfigFieldRef::ListItem(vec!["items".into()], 0),
    ] {
        let mut state = config_state(field.clone());
        state.config_edit = Some(ConfigEditState {
            field,
            buffer: "value".into(),
            cursor: 5,
        });
        let text = config_footer_text(&state, 96);
        assert!(text.contains("Enter confirm"), "{text}");
        assert!(text.contains("Esc cancel"), "{text}");
        for invalid in ["Ctrl-S", "↑/↓", "a add", "d remove"] {
            assert!(!text.contains(invalid), "{text}");
        }
    }
}

#[test]
fn config_render_multiline_edit_keeps_the_cursor_without_changing_the_buffer() {
    let field = ConfigFieldRef::Field(vec!["prompt".into()]);
    let mut state = config_state(field.clone());
    state.config_edit = Some(ConfigEditState {
        field,
        buffer: "first\nsecond".into(),
        cursor: "first\nsecond".len(),
    });
    let rows = buffer_rows(&config_buffer(&mut state, 80, 24));
    assert!(rows.iter().any(|row| row.contains("second▏")), "{rows:?}");
    assert_eq!(
        state.config_edit.as_ref().expect("edit").buffer,
        "first\nsecond"
    );
}

#[test]
fn config_render_nested_footer_returns_to_the_parent() {
    let mut state = config_state(ConfigFieldRef::Field(vec!["prompt".into()]));
    state.dialog_mut().expect("dialog").config_path = vec!["global".into()];
    let text = config_footer_text(&state, 96);
    assert!(text.contains("Esc back"), "{text}");
    assert!(!text.contains("Esc close"), "{text}");
}

#[test]
fn config_render_expanded_footer_matches_the_current_mode() {
    for (field, target, list_content) in [
        (ConfigFieldRef::Choice(vec!["mode".into()]), None, false),
        (ConfigFieldRef::List(vec!["items".into()]), Some(0), false),
        (ConfigFieldRef::List(vec!["items".into()]), None, true),
    ] {
        let mut state = config_state(field);
        let dialog = state.dialog_mut().expect("config dialog");
        dialog.config_expanded = Some(0);
        dialog.config_detail_target = target;
        let text = config_footer_text(&state, 96);
        for hint in [
            "Ctrl-S save",
            "Esc collapse",
            if list_content {
                "Enter edit"
            } else {
                "Enter select"
            },
        ] {
            assert!(text.contains(hint), "{text}");
        }
        for hint in ["a add", "d remove"] {
            assert_eq!(text.contains(hint), list_content, "{text}");
        }
    }
}

#[test]
fn zh_cn_language_picker_marks_current_language() {
    let theme = Theme::dark();
    let area = Rect::new(0, 0, 100, 30);
    let dialog = DialogState::new(
        DialogKind::LanguagePicker,
        "选择语言",
        None,
        vec![
            DialogItem::new("en", "English", None),
            DialogItem::new("zh-CN", "简体中文", None),
        ],
    );
    let mut state = TuiState::default();
    state.set_language(Some(crate::tui::i18n::Language::ZhCn));
    let backend = TestBackend::new(area.width, area.height);
    let mut terminal = Terminal::new(backend).expect("terminal");

    terminal
        .draw(|frame| render_picker(frame, &mut state, area, theme, &dialog))
        .expect("draw");

    let rendered = terminal
        .backend()
        .buffer()
        .content()
        .iter()
        .map(|cell| cell.symbol())
        .collect::<String>();
    assert!(rendered.contains("English"), "{rendered}");
    assert!(rendered.contains("● 简"), "{rendered}");
}
