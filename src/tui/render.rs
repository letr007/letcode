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

    let metrics = layout::workspace_metrics(
        workspace,
        &state.input_buffer,
        state.pending_permission.is_some(),
        layout::slash_panel_height(state),
    );
    let [
        transcript_area,
        _gap_area,
        slash_panel_area,
        composer_area,
        footer_area,
    ] = layout::split_workspace_layout(workspace, metrics);

    if state.timeline.items().is_empty() {
        render_welcome(frame, transcript_area, theme);
    } else {
        transcript::render_transcript(frame, state, transcript_area, theme);
    }

    slash_panel::render_slash_panel(frame, state, slash_panel_area, theme);
    composer::render_composer(frame, state, composer_area, theme);
    footer::render_footer(frame, state, footer_area, theme);
    dialog::render_dialog(frame, state, area, theme);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tui::surface;
    use crate::tui::{
        AppEvent, AssistantDeltaEvent, ErrorEvent, PermissionRequestEvent, ToolFinishedEvent,
        ToolOutcome, ToolStartedEvent, UserMessageEvent,
    };
    use ratatui::{Terminal, backend::TestBackend};

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

        let tiny = draw_to_string(&mut state, 10, 2);
        assert!(!tiny.is_empty());
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
            used_tokens: 12_345,
            context_window_tokens: 4_000_000,
        });
        state.set_footer("Ready", Some("detail text".into()));

        let rendered = draw_to_string(&mut state, 100, 16);

        assert!(!rendered.contains("model gpt-5.5-mini"), "{rendered}");
        assert!(!rendered.contains("12.3K/4.0M (0%)"), "{rendered}");
        assert!(!rendered.contains("exit to quit"), "{rendered}");
        assert!(rendered.contains("/help"), "{rendered}");
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
