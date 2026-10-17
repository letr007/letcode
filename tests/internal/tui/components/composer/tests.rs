use super::*;
use crate::tui::{PermissionRequestEvent, SessionEvent};
use crate::user_content::UserImageAttachment;
use ratatui::style::Color;
use ratatui::{Terminal, backend::TestBackend};

fn test_attachment(id: &str, label: &str) -> UserImageAttachment {
    UserImageAttachment {
        id: id.into(),
        label: label.into(),
        mime: "image/png".into(),
        data_url: "data:image/png;base64,AAAA".into(),
    }
}

fn draw_to_string(state: &mut TuiState, width: u16, height: u16) -> String {
    if state.language.is_none() {
        state.set_language(Some(crate::tui::i18n::Language::En));
    }
    let backend = TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend).expect("terminal");

    terminal
        .draw(|frame| {
            let area = Rect::new(0, 0, width, height);
            render_composer(frame, state, area, Theme::dark());
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

fn draw_rows(state: &mut TuiState, width: u16, height: u16) -> Vec<String> {
    if state.language.is_none() {
        state.set_language(Some(crate::tui::i18n::Language::En));
    }
    let backend = TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend).expect("terminal");

    terminal
        .draw(|frame| {
            let area = Rect::new(0, 0, width, height);
            render_composer(frame, state, area, Theme::dark());
        })
        .expect("draw");

    let buffer = terminal.backend().buffer();
    (0..height)
        .map(|y| {
            (0..width)
                .filter_map(|x| buffer.cell((x, y)).map(|cell| cell.symbol()))
                .collect::<String>()
        })
        .collect()
}

fn leading_bar_color(state: &mut TuiState, width: u16, height: u16) -> Option<Color> {
    let backend = TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend).expect("terminal");

    terminal
        .draw(|frame| {
            let area = Rect::new(0, 0, width, height);
            render_composer(frame, state, area, Theme::dark());
        })
        .expect("draw");

    terminal.backend().buffer().cell((0, 0)).map(|cell| cell.fg)
}

fn composer_cell_style(
    state: &mut TuiState,
    width: u16,
    height: u16,
    x: u16,
    y: u16,
) -> Option<(String, Color, Color)> {
    let backend = TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend).expect("terminal");

    terminal
        .draw(|frame| {
            let area = Rect::new(0, 0, width, height);
            render_composer(frame, state, area, Theme::dark());
        })
        .expect("draw");

    terminal
        .backend()
        .buffer()
        .cell((x, y))
        .map(|cell| (cell.symbol().to_string(), cell.fg, cell.bg))
}

#[test]
fn composer_cursor_never_returns_column_equal_to_width() {
    let width = 4;
    let cursor = composer_cursor_position("abcd", width, 4);
    assert!(cursor.column < width, "{cursor:?}");

    let cursor = composer_cursor_position("你", 2, "你".len());
    assert!(cursor.column < 2, "{cursor:?}");
}

#[test]
fn composer_cursor_pulses_in_every_theme() {
    for theme in [Theme::dark(), Theme::plain_for(None)] {
        let colors = cursor_backgrounds(theme);

        assert!(colors.len() > 2, "光标应该会脉动: {colors:?}");
        assert!(!colors.contains(&Color::Reset), "有面板时光标始终有底色");
    }

    let colors = cursor_backgrounds(Theme::glass());

    assert!(
        colors.contains(&Color::Reset),
        "空拍要交给终端背景: {colors:?}"
    );
    assert!(
        colors.iter().any(|color| *color != Color::Reset),
        "亮拍要画出来: {colors:?}"
    );
}

fn cursor_backgrounds(theme: Theme) -> Vec<Color> {
    let mut colors = Vec::new();
    for frame in 0..64 {
        let bg = composer_cursor_pulse(theme, frame).bg;
        if !colors.contains(&bg) {
            colors.push(bg);
        }
    }
    colors
}

#[test]
fn long_single_line_cjk_wraps_into_multiple_rows() {
    // Each CJK char is width 2.
    // width 6 => 3 chars per row.
    let input = "你好世界你好"; // 6 chars => 2 wrapped rows.
    let metrics = composer_metrics(input, 6, input.len());
    // Exact-fill on the final row advances the cursor to a new empty visual row.
    assert_eq!(metrics.row_count, 3);
    assert_eq!(metrics.cursor, ComposerCursor { row: 2, column: 0 });

    // Mixed widths.
    let mixed = "ab你cd你ef"; // widths: 1+1+2+1+1+2+1+1 = 10
    let metrics = composer_metrics(mixed, 4, mixed.len());
    assert!(metrics.row_count >= 3, "{metrics:?}");
}

#[test]
fn composer_inline_lines_wrap_wide_char_before_overflow() {
    let mut state = TuiState::default();
    state.set_input("abc你d");

    let lines = composer_inline_lines(&state, 4, Theme::dark())
        .into_iter()
        .map(|line| {
            line.spans
                .into_iter()
                .map(|span| span.content.into_owned())
                .collect::<String>()
        })
        .collect::<Vec<_>>();

    assert_eq!(lines, vec!["abc", "你d"]);
}

#[test]
fn pasted_text_token_renders_as_a_compact_chip() {
    let mut state = TuiState::default();
    state.add_composer_pasted_text("one\ntwo\nthree".into());

    let rendered = composer_inline_lines(&state, 80, Theme::dark())
        .into_iter()
        .flat_map(|line| line.spans)
        .map(|span| span.content.into_owned())
        .collect::<String>();

    assert_eq!(rendered, "[Pasted ~3 lines]");
    assert_eq!(state.composer_content().text, "one\ntwo\nthree");
}

#[test]
fn narrow_attachment_tokens_reserve_the_same_rows_as_the_renderer() {
    let attachment = test_attachment("img-1", "clipboard");
    let mut state = TuiState::default();
    state.set_input("ab");
    state.input_cursor = 2;
    state.add_composer_attachment(attachment.clone());

    assert_eq!(composer_inline_lines(&state, 5, Theme::dark()).len(), 3);
    assert_eq!(composer_metrics_with_attachments(&state, 5).row_count, 3);
}

#[test]
fn composer_metrics_keep_cursor_at_the_inline_attachment_boundary() {
    let mut state = TuiState::default();
    state.set_input("测试");
    state.add_composer_attachment(test_attachment("img-1", "clipboard"));
    state.input_cursor = "测试".len();

    let metrics = composer_metrics_with_attachments(&state, 80);
    assert_eq!(metrics.cursor.row, 0);
    assert_eq!(metrics.cursor.column, display_width("测试"));
}

#[test]
fn composer_metrics_wrap_attachment_tokens_atomically() {
    let mut state = TuiState::default();
    state.set_input("ab");
    state.add_composer_attachment(test_attachment("img-1", "clipboard"));
    state.add_composer_attachment(test_attachment("img-2", "clipboard"));
    state.input_cursor = "ab".len();

    let metrics = composer_metrics_with_attachments(&state, 14);
    assert!(metrics.row_count >= 2, "{metrics:?}");
    assert_eq!(metrics.cursor, ComposerCursor { row: 0, column: 2 });
}

#[test]
fn render_composer_tiny_does_not_panic_on_minuscule_areas() {
    let backend = TestBackend::new(2, 1);
    let mut terminal = Terminal::new(backend).expect("terminal");
    let mut state = TuiState::default();
    state.input_buffer = "你".into();

    terminal
        .draw(|frame| {
            let area = Rect::new(0, 0, 2, 1);
            render_composer(frame, &mut state, area, Theme::dark());
        })
        .expect("draw");
}

#[test]
fn pending_permission_takes_over_composer_surface() {
    let mut state = TuiState::default();
    let mut request =
        PermissionRequestEvent::new("call-1", "shell__exec", "cargo test --workspace");
    request.arguments = Some("cargo test --workspace".into());
    state.apply_event(SessionEvent::PermissionRequested(request));

    let rendered = draw_to_string(&mut state, 80, 8);

    assert!(
        rendered.contains("Approve tool") || rendered.contains("Run command"),
        "{rendered}"
    );
    assert!(
        rendered.contains('允')
            || rendered.contains("Allow once")
            || rendered.contains("allow once"),
        "{rendered}"
    );
    assert!(
        rendered.contains('拒') || rendered.contains("Reject") || rendered.contains("reject"),
        "{rendered}"
    );
    assert!(!rendered.contains("message letcode"), "{rendered}");
    assert!(!rendered.contains("args"), "{rendered}");
}

#[test]
fn pending_permission_footer_uses_connected_action_row() {
    let mut state = TuiState::default();
    let mut request =
        PermissionRequestEvent::new("call-1", "shell__exec", "cargo test --workspace");
    request.arguments = Some(r#"{"command":"cargo test --workspace"}"#.into());
    request.rationale = Some("project tests require approval".into());
    state.apply_event(SessionEvent::PermissionRequested(request));

    let rendered = draw_to_string(&mut state, 100, 10);

    assert!(
        rendered.contains('模') || rendered.contains("Patterns"),
        "{rendered}"
    );
    assert!(
        rendered.contains('参') || rendered.contains("Values"),
        "{rendered}"
    );
    assert!(
        rendered.contains('允') || rendered.contains("Allow once"),
        "{rendered}"
    );
    assert!(
        rendered.contains('拒') || rendered.contains("Reject"),
        "{rendered}"
    );
    assert!(!rendered.contains("Allow always"), "{rendered}");
    assert!(rendered.contains("y/o"), "{rendered}");
    assert!(rendered.contains("n/d"), "{rendered}");
    assert!(rendered.contains("esc"), "{rendered}");
    assert!(!rendered.contains("fullscreen"), "{rendered}");
}

#[test]
fn pending_permission_highlight_follows_the_selected_choice() {
    let mut state = TuiState::default();
    let mut request =
        PermissionRequestEvent::new("call-1", "shell__exec", "cargo test --workspace");
    request.arguments = Some(r#"{"command":"cargo test --workspace"}"#.into());
    request.can_allow_always = true;
    state.apply_event(SessionEvent::PermissionRequested(request));
    state.move_permission_choice_next();

    let rendered = draw_to_string(&mut state, 100, 10);

    assert!(rendered.contains("› Allow always"), "{rendered}");
    assert!(!rendered.contains("› Allow once"), "{rendered}");
    assert!(rendered.contains("←→"), "{rendered}");
    assert!(rendered.contains("enter"), "{rendered}");
}

#[test]
fn pending_permission_shows_subagent_origin_when_present() {
    let mut state = TuiState::default();
    let mut request =
        PermissionRequestEvent::new("call-1", "shell__exec", "cargo test --workspace");
    request.arguments = Some(r#"{"command":"cargo test --workspace"}"#.into());
    request.origin_label = Some("fixer".into());
    state.apply_event(SessionEvent::PermissionRequested(request));

    let rendered = draw_to_string(&mut state, 80, 8);

    assert!(rendered.contains("fixer"), "{rendered}");
    assert!(rendered.contains("cargo test --workspace"), "{rendered}");
}

fn color_distance(a: Color, b: Color) -> u32 {
    match (a, b) {
        (Color::Rgb(ar, ag, ab), Color::Rgb(br, bg, bb)) => {
            u32::from(ar.abs_diff(br)) + u32::from(ag.abs_diff(bg)) + u32::from(ab.abs_diff(bb))
        }
        _ => u32::MAX,
    }
}

#[test]
fn child_view_renders_read_only_status_bar_instead_of_input_composer() {
    let mut state = TuiState::new("gpt-5.5", "GPT-5.5", "default");
    state.set_provider_label("CLI Proxy API");
    state.replace_child_timeline_from_records(
        &[crate::transcript::TranscriptRecord {
            session_id: "child-session".into(),
            sequence: 1,
            timestamp_ms: 0,
            context_branch_id: None,
            event: crate::transcript::TranscriptEvent::SessionStarted {
                model: "gpt-5.5-mini".into(),
            },
        }],
        "parent-session",
        "child-session-1234567890",
        "fixer",
        1,
        3,
        1,
    );

    let rendered = draw_to_string(&mut state, 100, 8);

    assert!(rendered.contains("fixer"), "{rendered}");
    assert!(rendered.contains("2/3"), "{rendered}");
    assert!(rendered.contains("gpt-5.5-mini"), "{rendered}");
    assert!(rendered.contains("Parent"), "{rendered}");
    assert!(rendered.contains("Prev"), "{rendered}");
    assert!(rendered.contains("Next"), "{rendered}");
    assert!(!rendered.contains("Read-only child view"), "{rendered}");
    assert!(!rendered.contains("child-session-1234567890"), "{rendered}");
    assert!(!rendered.contains("model gpt-5.5-mini"), "{rendered}");
    assert!(!rendered.contains("records"), "{rendered}");
    assert!(!rendered.contains("parent-session"), "{rendered}");
    assert!(!rendered.contains("message letcode"), "{rendered}");
}

#[test]
fn wireframe_composer_uses_a_continuous_box() {
    let mut state = TuiState::new("gpt-5.5", "GPT-5.5", "default");
    state.set_language(Some(crate::tui::i18n::Language::En));
    let backend = TestBackend::new(60, 6);
    let mut terminal = Terminal::new(backend).expect("terminal");
    terminal
        .draw(|frame| {
            render_composer(
                frame,
                &mut state,
                Rect::new(0, 0, 60, 6),
                Theme::wireframe(),
            );
        })
        .expect("draw");

    let buffer = terminal.backend().buffer();
    let rows = (0..6)
        .map(|y| {
            (0..60)
                .filter_map(|x| buffer.cell((x, y)).map(|cell| cell.symbol()))
                .collect::<String>()
        })
        .collect::<Vec<_>>();

    assert!(
        rows[0].starts_with("┌") && rows[0].ends_with("┐"),
        "{rows:?}"
    );
    assert!(
        rows[1].starts_with("│") && rows[1].ends_with("│") && !rows[1].contains("message"),
        "{rows:?}"
    );
    assert!(
        rows[2].starts_with("│") && rows[2].ends_with("│") && rows[2].contains("message"),
        "{rows:?}"
    );
    assert!(
        rows[5].starts_with("└") && rows[5].ends_with("┘"),
        "{rows:?}"
    );
}

#[test]
fn prompt_cap_bands_need_painted_panels() {
    let mut state = TuiState::new("gpt-5.5", "GPT-5.5", "default");
    state.set_language(Some(crate::tui::i18n::Language::En));

    let cap_glyphs = |state: &mut TuiState, theme: Theme| {
        let backend = TestBackend::new(60, 5);
        let mut terminal = Terminal::new(backend).expect("terminal");
        terminal
            .draw(|frame| render_composer(frame, state, Rect::new(0, 0, 60, 5), theme))
            .expect("draw");
        terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|cell| cell.symbol())
            .collect::<String>()
    };

    assert!(
        cap_glyphs(&mut state, Theme::plain_for(None)).contains(surface::PROMPT_BOTTOM_CAP_GLYPH),
        "有面板填充的主题继续用半块字形描盒子边缘"
    );

    let glass = cap_glyphs(&mut state, Theme::glass());
    assert!(!glass.contains(surface::PROMPT_BOTTOM_CAP_GLYPH), "{glass}");
    assert!(glass.contains(surface::PROMPT_BOTTOM_LEFT_GLYPH), "{glass}");
}

#[test]
fn child_read_only_panel_centers_status_between_top_and_cap() {
    let mut state = TuiState::new("gpt-5.5", "GPT-5.5", "default");
    state.replace_child_timeline_from_records(
        &[],
        "parent-session",
        "child-session",
        "explorer",
        0,
        1,
        1,
    );

    let rows = draw_rows(&mut state, 100, 5);

    assert!(
        rows[0].contains(surface::PROMPT_TOP_LEFT_GLYPH)
            || rows[0].contains(surface::PROMPT_TOP_CAP_GLYPH),
        "{rows:?}"
    );
    assert!(!rows[1].contains("explorer"), "{rows:?}");
    assert!(rows[2].contains("explorer"), "{rows:?}");
    assert!(rows[2].contains("Parent"), "{rows:?}");
    assert!(!rows[3].contains("explorer"), "{rows:?}");
    assert!(
        rows[4].contains(surface::PROMPT_BOTTOM_LEFT_GLYPH)
            || rows[4].contains(surface::PROMPT_BOTTOM_CAP_GLYPH),
        "{rows:?}"
    );
}

#[test]
fn child_view_with_command_input_renders_composer_instead_of_read_only_panel() {
    let mut state = TuiState::new("gpt-5.5", "GPT-5.5", "default");
    state.replace_child_timeline_from_records(
        &[],
        "parent-session",
        "child-session",
        "explorer",
        0,
        1,
        1,
    );
    state.set_input("/tools".to_string());

    let rendered = draw_to_string(&mut state, 100, 8);

    assert!(rendered.contains("/tools"), "{rendered}");
    assert!(!rendered.contains("Read-only child view"), "{rendered}");
}

#[test]
fn pending_settings_use_dimmed_target_values_without_status_copy() {
    let mut state = TuiState::new("old/model", "Old", "default");
    state.set_provider_label("provider");
    state.set_reasoning_effort_label(Some("medium".into()));
    state.set_pending_model("new/model", "New");
    state.set_pending_reasoning_effort("high");
    state.set_pending_permission_mode("safe");

    let backend = TestBackend::new(100, 8);
    let mut terminal = Terminal::new(backend).expect("terminal");
    terminal
        .draw(|frame| {
            render_composer(frame, &mut state, Rect::new(0, 0, 100, 8), Theme::dark());
        })
        .expect("draw");

    let buffer = terminal.backend().buffer();
    let row_cells = (0..100)
        .filter_map(|x| buffer.cell((x, 6)))
        .collect::<Vec<_>>();
    let row = row_cells
        .iter()
        .map(|cell| cell.symbol())
        .collect::<String>();
    assert!(row.contains("New · provider · high · safe"), "{row}");
    assert!(!row.contains("pending"), "{row}");
    assert!(!row.contains("Old"), "{row}");
    assert!(!row.contains("medium"), "{row}");

    for value in ["New", "high", "safe"] {
        let start = row_cells
            .windows(value.chars().count())
            .position(|cells| cells.iter().map(|cell| cell.symbol()).collect::<String>() == value)
            .expect("pending value is visible");
        let cell = row_cells[start];
        assert_eq!(cell.fg, Theme::dark().dim_text, "{value}");
        assert!(cell.modifier.contains(Modifier::DIM), "{value}");
    }
}

#[test]
fn slash_panel_is_hidden_while_permission_prompt_is_pending() {
    let mut state = TuiState::default();
    state.set_input("/per");
    let request = PermissionRequestEvent::new("call-1", "shell__exec", "cargo test --workspace");
    state.apply_event(SessionEvent::PermissionRequested(request));

    let rendered = draw_to_string(&mut state, 100, 12);
    assert!(
        !rendered.contains("Show current permission mode"),
        "{rendered}"
    );
}
