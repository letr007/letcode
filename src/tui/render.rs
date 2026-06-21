use ratatui::{
    Frame,
    layout::{Alignment, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Paragraph},
};

use super::{
    components::{composer, dialog, footer, layout, slash_panel, transcript},
    state::TuiState,
    surface,
    theme::Theme,
};

const WELCOME_ART_LEFT: &[&str] = &[
    "▄          ▄  ",
    "█    █▀▀█ ▀█▀▀",
    "█    █▀▀▀  █  ",
    "▀▀▀▀ ▀▀▀▀  ▀▀▀",
];
const WELCOME_ART_RIGHT: &[&str] = &[
    "             ▄     ",
    "█▀▀▀ █▀▀█ █▀▀█ █▀▀█",
    "█    █  █ █  █ █▀▀▀",
    "▀▀▀▀ ▀▀▀▀ ▀▀▀▀ ▀▀▀▀",
];
/// Render the TUI from the current state using ratatui widgets only.
///
/// Rendering may refresh viewport bookkeeping, but it never invokes tools, resolves permissions,
/// persists transcripts, or mutates runtime/business state.
pub fn render(frame: &mut Frame<'_>, state: &mut TuiState) {
    let theme = Theme::dark();
    let area = frame.area();

    if area.is_empty() {
        return;
    }

    // Root background.
    frame.render_widget(Block::new().style(theme.app_style()), area);

    let workspace = layout::workspace_area(area);
    if workspace.height == 0 {
        // If bottom padding collapses the workspace, still render a 1-row footer.
        footer::render_footer(frame, state, area, theme);
        return;
    }
    if workspace.height == 1 {
        footer::render_footer(frame, state, workspace, theme);
        return;
    }

    if state.show_dashboard() {
        render_dashboard(frame, state, workspace, theme);
        dialog::render_dialog(frame, state, area, theme);
        return;
    }

    let metrics = layout::workspace_metrics(
        workspace,
        &state.input_buffer,
        state.pending_permission.is_some(),
        state.is_read_only_child_view(),
        layout::slash_panel_height(state),
    );
    let [
        transcript_area,
        _gap_area,
        slash_panel_area,
        composer_area,
        footer_area,
    ] = layout::split_workspace_layout(workspace, metrics);

    if state.active_timeline().items().is_empty() {
        render_welcome(frame, transcript_area, theme);
    } else {
        transcript::render_transcript(frame, state, transcript_area, theme);
    }

    slash_panel::render_slash_panel(frame, state, slash_panel_area, theme);
    composer::render_composer(frame, state, composer_area, theme);
    footer::render_footer(frame, state, footer_area, theme);
    dialog::render_dialog(frame, state, area, theme);
}

fn render_dashboard(frame: &mut Frame<'_>, state: &TuiState, area: Rect, theme: Theme) {
    let footer_area = Rect::new(
        area.x,
        area.y + area.height.saturating_sub(1),
        area.width,
        1,
    );
    let content_area = Rect::new(area.x, area.y, area.width, area.height.saturating_sub(1));

    if content_area.height == 0 {
        footer::render_footer(frame, state, footer_area, theme);
        return;
    }

    let prompt_width = content_area
        .width
        .min(surface::WELCOME_PROMPT_MAX_WIDTH)
        .max(1);
    let prompt_height = layout::composer_height(
        content_area.height,
        &state.input_buffer,
        prompt_width as usize,
    )
    .clamp(1, content_area.height);
    let slash_height =
        layout::slash_panel_height(state).min(content_area.height.saturating_sub(prompt_height));
    let logo_height: u16 = if content_area.width >= 52 && content_area.height >= 4 {
        4
    } else {
        1
    };
    let logo_gap: u16 = if content_area.height >= 12 { 2 } else { 1 };
    let prompt_gap: u16 = if slash_height > 0 { 1 } else { 0 };
    let hint_height: u16 = if content_area.height >= 10 { 1 } else { 0 };
    let hint_gap: u16 = if hint_height > 0 { 1 } else { 0 };
    let stack_height = logo_height
        .saturating_add(logo_gap)
        .saturating_add(prompt_height)
        .saturating_add(prompt_gap)
        .saturating_add(slash_height)
        .saturating_add(hint_gap)
        .saturating_add(hint_height)
        .min(content_area.height);
    let stack_y = content_area.y
        + content_area
            .height
            .saturating_sub(stack_height)
            .saturating_div(2);

    let logo_area = Rect::new(area.x, stack_y, area.width, logo_height);
    render_welcome(frame, logo_area, theme);

    let prompt_y = stack_y.saturating_add(logo_height).saturating_add(logo_gap);
    let prompt_x = content_area.x
        + content_area
            .width
            .saturating_sub(prompt_width)
            .saturating_div(2);
    let prompt_area = Rect::new(prompt_x, prompt_y, prompt_width, prompt_height);
    composer::render_composer(frame, state, prompt_area, theme);

    if slash_height > 0 {
        let slash_area = Rect::new(
            prompt_x,
            prompt_y
                .saturating_add(prompt_height)
                .saturating_add(prompt_gap),
            prompt_width,
            slash_height,
        );
        slash_panel::render_slash_panel(frame, state, slash_area, theme);
    }

    if hint_height > 0 {
        let hint_y = prompt_y
            .saturating_add(prompt_height)
            .saturating_add(prompt_gap)
            .saturating_add(slash_height)
            .saturating_add(hint_gap);
        render_dashboard_hint(
            frame,
            state,
            Rect::new(prompt_x, hint_y, prompt_width, 1),
            theme,
        );
    }

    footer::render_footer(frame, state, footer_area, theme);
}

fn render_dashboard_hint(frame: &mut Frame<'_>, state: &TuiState, area: Rect, theme: Theme) {
    if area.is_empty() || state.slash_panel_is_open() {
        return;
    }

    let line = Line::from(vec![
        Span::styled("/resume", dashboard_hint_key_style(theme)),
        Span::styled(" sessions   ", dashboard_hint_style(theme)),
        Span::styled("/help", dashboard_hint_key_style(theme)),
        Span::styled(" commands", dashboard_hint_style(theme)),
    ]);
    frame.render_widget(
        Paragraph::new(line)
            .style(theme.app_style())
            .alignment(Alignment::Right),
        area,
    );
}

fn render_welcome(frame: &mut Frame<'_>, area: Rect, theme: Theme) {
    if area.is_empty() {
        return;
    }

    if area.width >= 52 && area.height >= 4 {
        let lines: Vec<Line<'static>> = WELCOME_ART_LEFT
            .iter()
            .zip(WELCOME_ART_RIGHT.iter())
            .map(|(left, right)| {
                Line::from(vec![
                    Span::styled(format!("{left} "), wordmark_shadow_style(theme)),
                    Span::styled((*right).to_string(), wordmark_style(theme)),
                ])
            })
            .collect();
        let lines_height = lines.len() as u16;
        let title_y = area.y + area.height.saturating_sub(lines_height).saturating_div(2);
        frame.render_widget(
            Paragraph::new(lines)
                .style(theme.app_style())
                .alignment(Alignment::Center),
            Rect::new(area.x, title_y, area.width, lines_height),
        );
        return;
    }

    let title = if area.width >= 14 { "LETCODE" } else { "LC" };
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(title, wordmark_style(theme))))
            .style(theme.app_style())
            .alignment(Alignment::Center),
        area,
    );
}

fn wordmark_style(theme: Theme) -> Style {
    Style::default()
        .fg(theme.text)
        .bg(theme.root_bg)
        .add_modifier(Modifier::BOLD)
}

fn wordmark_shadow_style(theme: Theme) -> Style {
    Style::default()
        .fg(theme.notice)
        .bg(theme.root_bg)
        .add_modifier(Modifier::DIM)
}

fn dashboard_hint_style(theme: Theme) -> Style {
    Style::default().fg(theme.muted_text).bg(theme.root_bg)
}

fn dashboard_hint_key_style(theme: Theme) -> Style {
    Style::default()
        .fg(theme.text)
        .bg(theme.root_bg)
        .add_modifier(Modifier::BOLD)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tui::surface;
    use crate::tui::{
        AppEvent, AssistantDeltaEvent, ErrorEvent, PermissionRequestEvent, ToolFinishedEvent,
        ToolOutcome, ToolStartedEvent, UserMessageEvent,
    };
    use ratatui::{Terminal, backend::TestBackend, layout::Position};

    fn draw_to_string(state: &mut TuiState, width: u16, height: u16) -> String {
        let backend = TestBackend::new(width, height);
        let mut terminal = Terminal::new(backend).expect("test terminal is created");

        terminal
            .draw(|frame| render(frame, state))
            .expect("render succeeds");

        terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|cell| cell.symbol())
            .collect()
    }

    #[test]
    fn empty_welcome_view_renders_wordmark_without_panic() {
        let mut state = TuiState::new("gpt-5.5", "gpt-5.5", "default");

        let rendered = draw_to_string(&mut state, 80, 20);
        assert!(
            rendered.contains("█    █▀▀█ ▀█▀▀") || rendered.contains("LETCODE"),
            "{rendered}"
        );
        assert!(rendered.contains("/resume sessions"), "{rendered}");

        let tiny = draw_to_string(&mut state, 10, 2);
        assert!(!tiny.is_empty());
    }

    #[test]
    fn active_empty_session_uses_normal_workspace_layout() {
        let mut state = TuiState::new("gpt-5.5", "gpt-5.5", "default");
        state.mark_session_active();

        let rendered = draw_to_string(&mut state, 100, 24);

        assert!(rendered.contains("message letcode"), "{rendered}");
        assert!(!rendered.contains("/resume sessions"), "{rendered}");
    }

    #[test]
    fn user_and_assistant_timeline_content_appears() {
        let mut state = TuiState::default();
        state.apply_event(AppEvent::UserMessage(UserMessageEvent::new("hello tui")));
        state.apply_event(AppEvent::AssistantDelta(AssistantDeltaEvent::new(
            "hi there",
        )));
        state.apply_event(AppEvent::AssistantDone { message_id: None });

        let rendered = draw_to_string(&mut state, 90, 20);

        assert!(rendered.contains("hello tui"), "{rendered}");
        assert!(rendered.contains("hi there"), "{rendered}");
        assert!(rendered.contains(surface::ACCENT_BAR_GLYPH), "{rendered}");
        assert!(!rendered.contains("streaming"), "{rendered}");
    }

    #[test]
    fn pending_permission_prompt_displays_hint_and_tool_summary() {
        let mut state = TuiState::default();
        let mut request = PermissionRequestEvent::new("call-1", "shell__exec", "cargo test all");
        request.arguments = Some("cargo test".into());
        request.rationale = Some("tests need confirmation".into());
        state.apply_event(AppEvent::PermissionRequested(request));

        let rendered = draw_to_string(&mut state, 96, 24);

        assert!(
            rendered.contains("Approve tool") || rendered.contains("Run command"),
            "{rendered}"
        );
        assert!(rendered.contains("allow once"), "{rendered}");
        assert!(rendered.contains("reject"), "{rendered}");
        assert!(rendered.contains("cargo test all"), "{rendered}");
        assert!(!rendered.contains("message letcode"), "{rendered}");
        assert!(!rendered.contains("args"), "{rendered}");
    }

    #[test]
    fn footer_uses_compact_help_hint_without_duplicate_metadata() {
        let mut state = TuiState::new("gpt-5.5-mini", "gpt-5.5-mini", "safe");
        state.set_token_usage(crate::tui::state::ModelTokenUsage {
            used_tokens: 50_000,
            context_window_tokens: 100_000,
            input_tokens: 40_000,
            output_tokens: 10_000,
            cached_tokens: 20_000,
        });
        state.set_footer("Ready", Some("detail text".into()));

        let rendered = draw_to_string(&mut state, 100, 16);

        assert!(!rendered.contains("model gpt-5.5-mini"), "{rendered}");
        assert!(
            rendered.contains("██████████ ↑40.0k ↓10.0k 50% · /help commands"),
            "{rendered}"
        );
        assert!(!rendered.contains("exit to quit"), "{rendered}");
    }

    #[test]
    fn slash_panel_renders_above_composer_in_full_view() {
        let mut state = TuiState::default();
        state.set_input("/per");

        let rendered = draw_to_string(&mut state, 100, 20);

        assert!(
            rendered.contains("Show or switch permission mode"),
            "{rendered}"
        );
        assert!(rendered.contains("/per"), "{rendered}");
        assert!(!rendered.contains("prompt ·"), "{rendered}");
    }

    #[test]
    fn child_view_scroll_redraw_keeps_read_only_status_bar() {
        let mut state = TuiState::new("gpt-5.5", "GPT-5.5", "default");
        state.replace_child_timeline_from_records(
            &[crate::transcript::TranscriptRecord {
                session_id: "child-session".into(),
                sequence: 1,
                timestamp_ms: 0,
                event: crate::transcript::TranscriptEvent::SessionStarted {
                    model: "gpt-5.5-mini".into(),
                },
            }],
            "parent-session",
            "child-session-1234567890",
            "explorer",
            0,
            1,
        );

        let before = draw_to_string(&mut state, 100, 18);
        state.scroll_transcript_down(1);
        let after = draw_to_string(&mut state, 100, 18);

        assert!(before.contains("explorer"), "{before}");
        assert!(after.contains("explorer"), "{after}");
        assert!(after.contains("gpt-5.5-mini"), "{after}");
        assert!(after.contains("Parent"), "{after}");
        assert!(!after.contains("Read-only child view"), "{after}");
        assert!(!after.contains("child-session-1234567890"), "{after}");
        assert!(!after.contains("records"), "{after}");
        assert!(!after.contains("parent-session"), "{after}");
        assert!(!after.contains("message letcode"), "{after}");
    }

    #[test]
    fn dialog_overlay_renders_title_and_items() {
        let mut state = TuiState::new("gpt-5.5", "GPT-5.5", "default");
        state.open_dialog(crate::tui::state::DialogState::new(
            crate::tui::state::DialogKind::ModelPicker,
            "Switch model",
            Some("Select a model".into()),
            vec![
                crate::tui::state::DialogItem::new(
                    "gpt-5.5",
                    "GPT-5.5",
                    Some("gpt-5.5 · current".into()),
                ),
                crate::tui::state::DialogItem::new(
                    "gpt-5.5-mini",
                    "GPT-5.5 Mini",
                    Some("gpt-5.5-mini".into()),
                ),
            ],
        ));

        let rendered = draw_to_string(&mut state, 100, 24);

        assert!(rendered.contains("Switch model"), "{rendered}");
        assert!(rendered.contains("GPT-5.5 Mini"), "{rendered}");
        assert!(rendered.contains("Search"), "{rendered}");
        assert!(rendered.contains("Recent"), "{rendered}");
    }

    #[test]
    fn dialog_overlay_does_not_leave_dashboard_composer_cursor_visible() {
        let backend = TestBackend::new(100, 24);
        let mut terminal = Terminal::new(backend).expect("test terminal is created");
        let mut state = TuiState::new("gpt-5.5", "GPT-5.5", "default");
        state.open_dialog(crate::tui::state::DialogState::new(
            crate::tui::state::DialogKind::ModelPicker,
            "Select model",
            None,
            vec![crate::tui::state::DialogItem::new(
                "gpt-5.5",
                "GPT-5.5",
                Some("gpt-5.5".into()),
            )],
        ));

        terminal
            .draw(|frame| render(frame, &mut state))
            .expect("render succeeds");

        assert_eq!(
            terminal.get_cursor_position().expect("cursor position"),
            Position::ORIGIN
        );
    }

    #[test]
    fn permission_dialog_uses_picker_style() {
        let mut state = TuiState::new("gpt-5.5", "GPT-5.5", "default");
        let mut dialog = crate::tui::state::DialogState::new(
            crate::tui::state::DialogKind::PermissionPicker,
            "Permission mode",
            Some("Select how much freedom the agent has when using tools".into()),
            vec![
                crate::tui::state::DialogItem::new(
                    "safe",
                    "Safe",
                    Some("Ask before all tools".into()),
                ),
                crate::tui::state::DialogItem::new(
                    "default",
                    "Default",
                    Some("Allow read/preview, ask for risky tools".into()),
                ),
                crate::tui::state::DialogItem::new(
                    "solo",
                    "Solo",
                    Some("Allow write and command tools without asking".into()),
                ),
            ],
        );
        dialog.selected = 1;
        state.open_dialog(dialog);

        let rendered = draw_to_string(&mut state, 100, 24);

        assert!(rendered.contains("Permission mode"), "{rendered}");
        assert!(rendered.contains("Select how much freedom"), "{rendered}");
        assert!(!rendered.contains("Search"), "{rendered}");
        assert!(rendered.contains("Default"), "{rendered}");
        assert!(rendered.contains("Allow read/preview"), "{rendered}");
        assert!(rendered.contains("esc"), "{rendered}");
    }

    #[test]
    fn model_dialog_scrolls_to_selected_item() {
        let mut state = TuiState::new("model-00", "Model 00", "default");
        let items = (0..20)
            .map(|index| {
                crate::tui::state::DialogItem::new(
                    format!("model-{index:02}"),
                    format!("Model {index:02}"),
                    Some(format!("provider-{index:02}")),
                )
            })
            .collect::<Vec<_>>();
        let mut dialog = crate::tui::state::DialogState::new(
            crate::tui::state::DialogKind::ModelPicker,
            "Select model",
            None,
            items,
        );
        dialog.selected = 14;
        state.open_dialog(dialog);

        let rendered = draw_to_string(&mut state, 100, 20);

        assert!(rendered.contains("Model 14"), "{rendered}");
        assert!(!rendered.contains("Model 01"), "{rendered}");
    }

    #[test]
    fn session_dialog_scrolls_to_selected_item() {
        let mut state = TuiState::default();
        let items = (0..20)
            .map(|index| {
                crate::tui::state::DialogItem::new(
                    format!("session-{index:02}"),
                    format!("Session {index:02}"),
                    Some(format!("detail-{index:02}")),
                )
                .with_section("Today")
            })
            .collect::<Vec<_>>();
        let mut dialog = crate::tui::state::DialogState::new(
            crate::tui::state::DialogKind::SessionPicker,
            "Sessions",
            None,
            items,
        );
        dialog.selected = 14;
        state.open_dialog(dialog);

        let rendered = draw_to_string(&mut state, 100, 20);

        assert!(rendered.contains("Session 14"), "{rendered}");
        assert!(!rendered.contains("Session 01"), "{rendered}");
    }

    #[test]
    fn tool_cards_and_errors_use_structured_timeline_fields() {
        let mut state = TuiState::default();
        let mut started = ToolStartedEvent::new("tool-7", "shell__exec", "run cargo check");
        started.arguments = Some("cargo check".into());
        state.apply_event(AppEvent::ToolStarted(started));
        let mut finished = ToolFinishedEvent::new(
            "tool-7",
            "shell__exec",
            "run cargo check",
            ToolOutcome::Failure,
        );
        finished.output = Some("compiler said no".into());
        state.apply_event(AppEvent::ToolFinished(finished));
        let mut error = ErrorEvent::new("render problem");
        error.details = Some("missing widget area".into());
        state.apply_event(AppEvent::Error(error));

        let rendered = draw_to_string(&mut state, 100, 24);

        assert!(rendered.contains("→"), "{rendered}");
        assert!(rendered.contains("Run"), "{rendered}");
        assert!(rendered.contains("cargo check"), "{rendered}");
        assert!(!rendered.contains("compiler said no"), "{rendered}");
        assert!(rendered.contains("error"), "{rendered}");
        assert!(rendered.contains("render problem"), "{rendered}");
    }
}
