use super::{
    TokenBudgetSegment, compaction_indicator_spans, footer_hint_spans, footer_scanner_cells,
    footer_status_spans, output_rate_graph_glyph, output_token_rate_style, render_footer,
    token_budget_bar_spans, token_budget_cache_hit_percent, token_budget_segment_units,
    token_budget_spans,
};
use crate::{
    session::RetryLifecycleEvent,
    tui::{AppPhase, TuiState},
};
use ratatui::{Terminal, backend::TestBackend, layout::Rect};

#[test]
fn footer_places_git_branch_before_context_branch() {
    let mut state = TuiState::default();
    state.set_git_branch(Some("feature/branch".into()));
    state.set_current_context_branch("context-2");

    let rendered = footer_hint_spans(&state, crate::tui::Theme::dark(), 80)
        .iter()
        .map(|span| span.content.as_ref())
        .collect::<String>();

    assert!(
        rendered.starts_with(" feature/branch · 󰙅 context-2"),
        "{rendered}"
    );
}

#[test]
fn footer_preserves_context_when_long_git_branch_is_truncated() {
    let mut state = TuiState::default();
    state.set_git_branch(Some("feature/超长分支名称-for-footer-budget".into()));
    state.set_current_context_branch("root");

    let spans = footer_hint_spans(&state, crate::tui::Theme::dark(), 18);
    let line = ratatui::text::Line::from(spans);
    let rendered = line.to_string();

    assert!(line.width() <= 18, "{rendered}");
    assert!(rendered.contains("󰙅 root"), "{rendered}");
    assert!(rendered.contains('…'), "{rendered}");
}

#[test]
fn footer_prioritizes_token_status_over_metadata_and_help() {
    let mut state = TuiState::default();
    state.set_git_branch(Some("feature/long-branch".into()));
    state.set_current_context_branch("root");
    state.model_token_usage = Some(crate::tui::state::ModelTokenUsage {
        input_tokens: 600,
        output_tokens: 100,
        cached_tokens: 200,
        used_tokens: 700,
        context_window_tokens: 1_000,
        cache_report: None,
        prompt_composition: Vec::new(),
    });

    let status = token_budget_spans(
        state.model_token_usage.as_ref().expect("token usage"),
        crate::tui::Theme::dark(),
    );
    let status_width = ratatui::text::Line::from(status.clone()).width();
    let rendered = footer_hint_spans(&state, crate::tui::Theme::dark(), status_width)
        .iter()
        .map(|span| span.content.as_ref())
        .collect::<String>();
    let expected = status
        .iter()
        .map(|span| span.content.as_ref())
        .collect::<String>();

    assert_eq!(rendered, expected);
    assert!(!rendered.contains(" "), "{rendered}");
    assert!(!rendered.contains("󰙅 "), "{rendered}");
    assert!(!rendered.contains("/help"), "{rendered}");
}

#[test]
fn footer_keeps_last_cache_rate_until_provider_usage_arrives() {
    let mut state = TuiState::default();
    state.set_token_usage(crate::tui::state::ModelTokenUsage {
        input_tokens: 120,
        output_tokens: 20,
        cached_tokens: 60,
        used_tokens: 140,
        context_window_tokens: 1_000,
        cache_report: None,
        prompt_composition: Vec::new(),
    });
    state.apply_live_token_usage(crate::tui::state::ModelTokenUsage {
        input_tokens: 300,
        output_tokens: 0,
        cached_tokens: 0,
        used_tokens: 300,
        context_window_tokens: 1_000,
        cache_report: None,
        prompt_composition: Vec::new(),
    });

    let pending = footer_hint_spans(&state, crate::tui::Theme::dark(), 80)
        .iter()
        .map(|span| span.content.as_ref())
        .collect::<String>();
    assert!(pending.contains("50.00%"), "{pending}");

    state.set_token_usage(crate::tui::state::ModelTokenUsage {
        input_tokens: 300,
        output_tokens: 10,
        cached_tokens: 60,
        used_tokens: 310,
        context_window_tokens: 1_000,
        cache_report: None,
        prompt_composition: Vec::new(),
    });
    let committed = footer_hint_spans(&state, crate::tui::Theme::dark(), 80)
        .iter()
        .map(|span| span.content.as_ref())
        .collect::<String>();
    assert!(committed.contains("20.00%"), "{committed}");
    assert!(!committed.contains("50.00%"), "{committed}");
}

#[test]
fn footer_appends_output_token_rate_after_context_status() {
    let mut state = TuiState::default();
    state.model_token_usage = Some(crate::tui::state::ModelTokenUsage {
        input_tokens: 600,
        output_tokens: 100,
        cached_tokens: 0,
        used_tokens: 700,
        context_window_tokens: 1_000,
        cache_report: None,
        prompt_composition: Vec::new(),
    });
    state.push_output_rate_graph_sample(Some(60));

    let rendered = footer_hint_spans(&state, crate::tui::Theme::dark(), 80)
        .iter()
        .map(|span| span.content.as_ref())
        .collect::<String>();

    assert!(rendered.contains("~70% · ⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀ 60t/s"), "{rendered}");
}

#[test]
fn footer_shows_output_token_rate_before_provider_usage_arrives() {
    let mut state = TuiState::default();
    state.push_output_rate_graph_sample(Some(60));

    let rendered = footer_hint_spans(&state, crate::tui::Theme::dark(), 80)
        .iter()
        .map(|span| span.content.as_ref())
        .collect::<String>();

    assert!(rendered.contains(" · ⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀ 60t/s"), "{rendered}");
}

#[test]
fn output_token_rate_color_scales_from_slow_to_fast() {
    let theme = crate::tui::Theme::dark();

    assert_eq!(output_token_rate_style(19, theme).fg, Some(theme.error));
    assert_eq!(output_token_rate_style(20, theme).fg, Some(theme.approval));
    assert_eq!(output_token_rate_style(39, theme).fg, Some(theme.approval));
    assert_eq!(output_token_rate_style(40, theme).fg, Some(theme.warning));
    assert_eq!(output_token_rate_style(79, theme).fg, Some(theme.warning));
    assert_eq!(output_token_rate_style(80, theme).fg, Some(theme.success));
}

#[test]
fn output_rate_graph_glyphs_encode_dot_heights() {
    assert_eq!(output_rate_graph_glyph(1, 1), '⣀');
    assert_eq!(output_rate_graph_glyph(2, 2), '⣤');
    assert_eq!(output_rate_graph_glyph(3, 3), '⣶');
    assert_eq!(output_rate_graph_glyph(4, 4), '⣿');
    assert_eq!(output_rate_graph_glyph(3, 4), '⣾');
}

#[test]
fn footer_uses_token_usage_from_active_child_view() {
    let mut state = TuiState::default();
    state.set_token_usage(crate::tui::state::ModelTokenUsage {
        used_tokens: 100,
        context_window_tokens: 1_000,
        input_tokens: 100,
        output_tokens: 0,
        cached_tokens: 0,
        cache_report: None,
        prompt_composition: Vec::new(),
    });
    state.replace_child_timeline_from_records(
        &[],
        "parent-session",
        "child-session",
        "explorer",
        0,
        1,
        1,
    );
    state.apply_child_session_event(
        "child-session",
        crate::tui::SessionEvent::TokenUsage(crate::session::TokenUsageEvent::with_breakdown(
            700, 1_000, 600, 100, 0,
        )),
    );

    let rendered = footer_hint_spans(&state, crate::tui::Theme::dark(), 80)
        .iter()
        .map(|span| span.content.as_ref())
        .collect::<String>();

    assert!(rendered.contains("↑600"), "{rendered}");
    assert!(rendered.contains("↓100"), "{rendered}");
    assert!(rendered.contains("~70%"), "{rendered}");
    assert!(!rendered.contains("~10%"), "{rendered}");
}

#[test]
fn footer_truncates_long_metadata_without_exceeding_budget() {
    let mut state = TuiState::default();
    state.set_git_branch(Some(
        "feature/a-very-long-branch-name-that-must-be-clipped".into(),
    ));
    state.set_current_context_branch("context-with-a-long-name");

    let spans = footer_hint_spans(&state, crate::tui::Theme::dark(), 24);
    let line = ratatui::text::Line::from(spans);
    let rendered = line.to_string();

    assert!(line.width() <= 24, "{rendered}");
    assert!(rendered.contains('…'), "{rendered}");
}

#[test]
fn footer_phase_indicator_tracks_the_visible_child_session() {
    let mut state = TuiState::default();
    state.phase = AppPhase::Completed;
    state.replace_child_timeline_from_records(
        &[],
        "parent-session",
        "child-session",
        "explorer",
        0,
        1,
        1,
    );
    state.set_child_view_phase_for_test(AppPhase::WaitingForPermission);

    let rendered = footer_status_spans(&state, crate::tui::Theme::dark())
        .iter()
        .map(|span| span.content.as_ref())
        .collect::<String>();

    assert_eq!(state.active_phase(), AppPhase::WaitingForPermission);
    assert_eq!(rendered, "▲");
}

#[test]
fn footer_phase_indicator_returns_to_parent_session_state() {
    let mut state = TuiState::default();
    state.phase = AppPhase::Running;
    state.replace_child_timeline_from_records(
        &[],
        "parent-session",
        "child-session",
        "explorer",
        0,
        1,
        1,
    );
    state.set_child_view_phase_for_test(AppPhase::Completed);
    assert_eq!(state.active_phase(), AppPhase::Completed);

    state.transcript_view = crate::tui::state::TranscriptViewState::Parent;

    assert_eq!(state.active_phase(), AppPhase::Running);
    assert_eq!(
        footer_status_spans(&state, crate::tui::Theme::dark()).len(),
        8
    );
}

#[test]
fn narrow_footer_keeps_phase_indicator_visible() {
    let mut state = TuiState::default();
    state.set_git_branch(Some("feature/extremely-long-branch-name".into()));
    state.set_current_context_branch("context-long");
    let backend = TestBackend::new(12, 1);
    let mut terminal = Terminal::new(backend).expect("terminal");

    terminal
        .draw(|frame| {
            render_footer(
                frame,
                &state,
                Rect::new(0, 0, 12, 1),
                crate::tui::Theme::dark(),
            )
        })
        .expect("render footer");

    let row = (0..12)
        .map(|x| {
            terminal
                .backend()
                .buffer()
                .cell((x, 0))
                .expect("footer cell")
                .symbol()
        })
        .collect::<String>();
    assert!(row.contains('◆'), "{row}");
}

#[test]
fn footer_hides_git_branch_outside_a_repository() {
    let mut state = TuiState::default();
    state.set_git_branch(None);
    state.set_current_context_branch("root");

    let rendered = footer_hint_spans(&state, crate::tui::Theme::dark(), 80)
        .iter()
        .map(|span| span.content.as_ref())
        .collect::<String>();

    assert!(rendered.starts_with("󰙅 root"), "{rendered}");
    assert!(!rendered.contains(" "), "{rendered}");
}

#[test]
fn token_budget_units_keep_cache_input_and_output_segments() {
    assert_eq!(
        token_budget_segment_units(100, 10, [20, 30, 10]),
        [16, 24, 8]
    );
}

#[test]
fn token_budget_units_keep_low_usage_visible() {
    assert_eq!(
        token_budget_segment_units(400_000, 10, [0, 15_900, 1_400]),
        [0, 3, 0]
    );
}

#[test]
fn token_budget_units_make_tiny_used_total_visible_once() {
    assert_eq!(
        token_budget_segment_units(1_000_000, 10, [0, 10, 0]),
        [0, 1, 0]
    );
}

#[test]
fn token_budget_uses_snapshot_usage_for_percent_and_output_bar() {
    let usage = crate::tui::state::ModelTokenUsage {
        used_tokens: 228_000,
        context_window_tokens: 1_000_000,
        input_tokens: 200_000,
        output_tokens: 1_300_000,
        cached_tokens: 0,
        cache_report: None,
        prompt_composition: Vec::new(),
    };

    let rendered = token_budget_spans(&usage, crate::tui::Theme::dark())
        .iter()
        .map(|span| span.content.as_ref())
        .collect::<String>();

    assert!(rendered.contains("↓1.3m"), "{rendered}");
    assert!(rendered.contains("~23%"), "{rendered}");
    assert!(!rendered.contains("100%"), "{rendered}");
    assert_eq!(
        token_budget_segment_units(1_000_000, 10, [0, 200_000, 28_000]),
        [0, 16, 2]
    );
}

#[test]
fn empty_token_budget_cells_keep_ink_only_on_painted_panels() {
    let empty_bar = |theme: crate::tui::Theme| {
        token_budget_bar_spans(
            3,
            [
                (0, TokenBudgetSegment::Cache),
                (0, TokenBudgetSegment::Input),
                (0, TokenBudgetSegment::Output),
            ],
            theme,
        )
    };

    let plain = empty_bar(crate::tui::Theme::plain_for(None));
    assert!(plain.iter().all(|span| span.content.as_ref() == "█"));

    let glass = empty_bar(crate::tui::Theme::glass());
    assert!(
        glass.iter().all(|span| span.content.as_ref() == " "),
        "{glass:?}"
    );

    let used_bar = |theme: crate::tui::Theme| {
        token_budget_bar_spans(
            3,
            [
                (3, TokenBudgetSegment::Cache),
                (0, TokenBudgetSegment::Input),
                (0, TokenBudgetSegment::Output),
            ],
            theme,
        )
    };
    assert_eq!(
        used_bar(crate::tui::Theme::plain_for(None))[0]
            .content
            .as_ref(),
        "▍"
    );

    let glass_used = used_bar(crate::tui::Theme::glass());
    assert_eq!(glass_used[0].content.as_ref(), "▍");
    assert!(
        glass_used[1..]
            .iter()
            .all(|span| span.content.as_ref() == " ")
    );
}

#[test]
fn scanner_trail_fades_without_a_painted_panel() {
    for theme in [
        crate::tui::Theme::dark(),
        crate::tui::Theme::plain_for(None),
        crate::tui::Theme::glass(),
    ] {
        let mut colors: Vec<ratatui::style::Color> = Vec::new();
        for frame in 0..8 {
            for (_, color) in footer_scanner_cells(frame, theme) {
                if !colors.contains(&color) {
                    colors.push(color);
                }
            }
        }

        assert!(colors.len() > 2, "拖尾应该保留渐变: {colors:?}");
    }
}

#[test]
fn token_budget_cache_hit_percent_is_hidden_without_cached_tokens() {
    assert_eq!(token_budget_cache_hit_percent(40_000, 0), None);
    assert_eq!(token_budget_cache_hit_percent(0, 20_000), None);
}

#[test]
fn token_budget_cache_hit_percent_uses_cached_over_input_tokens() {
    assert_eq!(token_budget_cache_hit_percent(40_000, 20_000), Some(50.0));
    assert_eq!(token_budget_cache_hit_percent(40_000, 80_000), Some(100.0));
    assert_eq!(
        token_budget_cache_hit_percent(3, 1),
        Some(33.33333333333333)
    );
}

#[test]
fn token_budget_cache_hit_percent_renders_two_decimal_places() {
    let usage = crate::tui::state::ModelTokenUsage {
        used_tokens: 3,
        context_window_tokens: 10,
        input_tokens: 3,
        output_tokens: 0,
        cached_tokens: 1,
        cache_report: None,
        prompt_composition: Vec::new(),
    };

    let rendered = token_budget_spans(&usage, crate::tui::Theme::dark())
        .iter()
        .map(|span| span.content.as_ref())
        .collect::<String>();

    assert!(rendered.contains("33.33%"), "{rendered}");
}

#[test]
fn projected_token_usage_marks_percent_as_estimated() {
    let usage = crate::tui::state::ModelTokenUsage {
        used_tokens: 50_000,
        context_window_tokens: 100_000,
        input_tokens: 50_000,
        output_tokens: 0,
        cached_tokens: 0,
        cache_report: None,
        prompt_composition: Vec::new(),
    };

    let rendered = super::token_budget_spans(&usage, crate::tui::Theme::dark())
        .iter()
        .map(|span| span.content.as_ref())
        .collect::<String>();

    assert!(rendered.contains("~50%"), "{rendered}");
}
