use super::*;
use ratatui::{Terminal, backend::TestBackend};

#[test]
fn sidebar_renders_compact_session_and_model_details() {
    let mut state = TuiState::default();
    state.set_language(Some(crate::tui::i18n::Language::En));
    state.set_git_branch(Some("main".into()));
    state.session_id = Some("session-1234567890".into());
    state.model_label = "Test Model".into();
    let backend = TestBackend::new(42, 18);
    let mut terminal = Terminal::new(backend).expect("terminal");
    terminal
        .draw(|frame| render_sidebar(frame, &mut state, frame.area(), Theme::dark()))
        .expect("draw");
    let rendered = terminal
        .backend()
        .buffer()
        .content()
        .iter()
        .map(|cell| cell.symbol())
        .collect::<String>();
    assert!(rendered.contains("session-1234567890"), "{rendered}");
    assert!(rendered.contains("Test Model"), "{rendered}");
    assert!(rendered.contains("Git       main"), "{rendered}");
    assert_eq!(
        rendered.matches("session-1234567890").count(),
        1,
        "{rendered}"
    );
}

#[test]
fn sidebar_registers_the_session_id_row_only_when_present() {
    let mut state = TuiState::default();
    state.session_id = Some("session-1234567890".into());
    let backend = TestBackend::new(42, 18);
    let mut terminal = Terminal::new(backend).expect("terminal");
    terminal
        .draw(|frame| render_sidebar(frame, &mut state, frame.area(), Theme::dark()))
        .expect("draw");

    let area = state.last_sidebar_session_id;
    assert!(area.width > 0 && area.height > 0);
    let buffer = terminal.backend().buffer();
    let text = (area.x..area.right())
        .filter_map(|column| buffer.cell((column, area.y)))
        .map(|cell| cell.symbol())
        .collect::<String>();
    assert!(text.contains("session-1234567890"), "{text}");

    state.session_id = None;
    terminal
        .draw(|frame| render_sidebar(frame, &mut state, frame.area(), Theme::dark()))
        .expect("draw");
    assert_eq!(state.last_sidebar_session_id, Rect::default());
}

#[test]
fn sidebar_renders_detailed_context_usage_breakdown() {
    let mut state = TuiState::default();
    state.set_language(Some(crate::tui::i18n::Language::En));
    state.set_token_usage(crate::tui::state::ModelTokenUsage {
        used_tokens: 70_000,
        context_window_tokens: 128_000,
        input_tokens: 64_000,
        output_tokens: 6_000,
        cached_tokens: 24_000,
        prompt_composition: vec![
            crate::agent::PromptCompositionEntry {
                category: "system".into(),
                estimated_tokens: 18_000,
                segments: 2,
            },
            crate::agent::PromptCompositionEntry {
                category: "skills".into(),
                estimated_tokens: 12_000,
                segments: 2,
            },
            crate::agent::PromptCompositionEntry {
                category: "tools".into(),
                estimated_tokens: 20_000,
                segments: 1,
            },
            crate::agent::PromptCompositionEntry {
                category: "messages".into(),
                estimated_tokens: 20_000,
                segments: 3,
            },
        ],
        cache_report: Some(crate::agent::CacheUsageReport {
            configured: true,
            hint_serialized: true,
            retention_sent: None,
            stable_prefix_segments: 2,
            stable_prompt_tokens: 42_000,
            volatile_prompt_tokens: 22_000,
            cacheable_prefix_tokens: 40_000,
            stable_after_boundary_tokens: 2_000,
            local_prefix_fingerprint: None,
            routing_key: None,
            actual_cached_tokens: Some(24_000),
        }),
    });
    let backend = TestBackend::new(42, 28);
    let mut terminal = Terminal::new(backend).expect("terminal");
    terminal
        .draw(|frame| render_sidebar(frame, &mut state, frame.area(), Theme::dark()))
        .expect("draw");
    let rendered = terminal
        .backend()
        .buffer()
        .content()
        .iter()
        .map(|cell| cell.symbol())
        .collect::<String>();

    for expected in [
        "Node",
        "Context",
        "64.0k / 128.0k · 50%",
        "System prompt",
        "16.5k",
        "Skills",
        "11.0k",
        "Tools",
        "18.3k",
        "Messages",
        "18.3k",
        "Remaining",
        "64.0k",
    ] {
        assert!(
            rendered.contains(expected),
            "missing {expected:?}: {rendered}"
        );
    }
    assert!(!rendered.contains("Output"), "{rendered}");
}

#[test]
fn context_breakdown_excludes_current_output_tokens() {
    let mut state = TuiState::default();
    state.set_language(Some(crate::tui::i18n::Language::En));
    state.set_token_usage(crate::tui::state::ModelTokenUsage {
        used_tokens: 12_000,
        context_window_tokens: 20_000,
        input_tokens: 10_000,
        output_tokens: 2_000,
        cached_tokens: 0,
        cache_report: None,
        prompt_composition: vec![crate::agent::PromptCompositionEntry {
            category: "messages".into(),
            estimated_tokens: 10_000,
            segments: 1,
        }],
    });
    let backend = TestBackend::new(42, 22);
    let mut terminal = Terminal::new(backend).expect("terminal");
    terminal
        .draw(|frame| render_sidebar(frame, &mut state, frame.area(), Theme::dark()))
        .expect("draw");
    let rendered = terminal
        .backend()
        .buffer()
        .content()
        .iter()
        .map(|cell| cell.symbol())
        .collect::<String>();

    assert!(rendered.contains("Messages        10.0k"), "{rendered}");
    assert!(!rendered.contains("Output"), "{rendered}");
    assert!(rendered.contains("10.0k / 20.0k · 50%"), "{rendered}");
    let bar_cells = terminal
        .backend()
        .buffer()
        .content()
        .chunks(42)
        .filter(|row| row.iter().any(|cell| "█▉▊▋▌▍▎▏".contains(cell.symbol())))
        .max_by_key(|row| {
            row.iter()
                .filter(|cell| "█▉▊▋▌▍▎▏".contains(cell.symbol()))
                .count()
        })
        .expect("context bar");
    assert_eq!(
        bar_cells
            .iter()
            .filter(|cell| cell.symbol() == "█" && cell.fg != Theme::dark().elevated_bg)
            .count(),
        19
    );
    assert_eq!(
        bar_cells
            .iter()
            .filter(|cell| "▉▊▋▌▍▎▏".contains(cell.symbol()))
            .count(),
        1
    );
    assert!(rendered.contains("Remaining       10.0k"), "{rendered}");
}

#[test]
fn sidebar_wraps_long_context_summary_and_todo_content() {
    let mut state = TuiState::default();
    state.set_language(Some(crate::tui::i18n::Language::En));
    state.current_context_branch = "history-13".into();
    state.mcp_discovery = crate::tui::state::McpDiscoveryState::Ready;
    state.set_token_usage(crate::tui::state::ModelTokenUsage {
        used_tokens: 84_100,
        context_window_tokens: 1_000_000,
        input_tokens: 84_100,
        output_tokens: 0,
        cached_tokens: 0,
        cache_report: None,
        prompt_composition: Vec::new(),
    });
    state.latest_todo = Some(crate::tui::timeline::TodoView {
        items: vec![crate::agent::TodoItem {
            id: "wrap".into(),
            content: "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789".into(),
            status: crate::agent::TodoStatus::InProgress,
        }],
        auto_continue: crate::agent::AutoContinueState::default(),
    });
    let backend = TestBackend::new(32, 24);
    let mut terminal = Terminal::new(backend).expect("terminal");
    terminal
        .draw(|frame| render_sidebar(frame, &mut state, frame.area(), Theme::dark()))
        .expect("draw");
    let rows = (0..terminal.backend().buffer().area.height)
        .map(|y| {
            terminal
                .backend()
                .buffer()
                .content()
                .iter()
                .skip(y as usize * 32)
                .take(32)
                .map(|cell| cell.symbol())
                .collect::<String>()
        })
        .collect::<Vec<_>>();

    let context_row = rows
        .iter()
        .position(|row| row.contains("history-13"))
        .expect("context branch row");
    assert!(rows[context_row].contains("Node"), "{rows:?}");
    assert!(
        rows.iter().any(|row| row.contains("84.1k / 1.0m")),
        "{rows:?}"
    );
    let todo_start = rows
        .iter()
        .position(|row| row.contains("Todos  1"))
        .expect("todo heading");
    let todo_rows = rows
        .iter()
        .skip(todo_start + 1)
        .take(2)
        .cloned()
        .collect::<Vec<_>>();
    assert_eq!(todo_rows.len(), 2, "{rows:?}");
    assert_eq!(
        todo_rows
            .iter()
            .flat_map(|row| row.chars().filter(|ch| ch.is_ascii_alphanumeric()))
            .collect::<String>(),
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    );
    assert!(!todo_rows.iter().any(|row| row.contains('…')));
}

#[test]
fn context_details_leave_room_for_mcp_and_todos() {
    let mut state = TuiState::default();
    state.set_language(Some(crate::tui::i18n::Language::En));
    state.set_mcp_servers(vec![crate::mcp::McpServerCatalogEntry {
        name: "docs".into(),
        enabled: true,
        status: crate::mcp::McpServerStatus::Online { tool_count: 2 },
    }]);
    state.set_token_usage(crate::tui::state::ModelTokenUsage {
        used_tokens: 10_000,
        context_window_tokens: 20_000,
        input_tokens: 9_000,
        output_tokens: 1_000,
        cached_tokens: 0,
        cache_report: None,
        prompt_composition: (0..8)
            .map(|index| crate::agent::PromptCompositionEntry {
                category: format!("other-{index}"),
                estimated_tokens: 1_000,
                segments: 1,
            })
            .collect(),
    });
    state.latest_todo = Some(crate::tui::timeline::TodoView {
        items: vec![crate::agent::TodoItem {
            id: "keep".into(),
            content: "important todo".into(),
            status: crate::agent::TodoStatus::Pending,
        }],
        auto_continue: crate::agent::AutoContinueState::default(),
    });
    let backend = TestBackend::new(42, 20);
    let mut terminal = Terminal::new(backend).expect("terminal");
    terminal
        .draw(|frame| render_sidebar(frame, &mut state, frame.area(), Theme::dark()))
        .expect("draw");
    let rendered = terminal
        .backend()
        .buffer()
        .content()
        .iter()
        .map(|cell| cell.symbol())
        .collect::<String>();

    assert!(state.sidebar_max_scroll > 0);
    state.scroll_sidebar_to_bottom();
    terminal
        .draw(|frame| render_sidebar(frame, &mut state, frame.area(), Theme::dark()))
        .expect("draw scrolled sidebar");
    let rendered = terminal
        .backend()
        .buffer()
        .content()
        .iter()
        .map(|cell| cell.symbol())
        .collect::<String>();
    assert!(rendered.contains("MCP"), "{rendered}");
    assert!(rendered.contains("docs"), "{rendered}");
    assert!(rendered.contains("2 tools"), "{rendered}");
    let rows = (0..terminal.backend().buffer().area.height)
        .map(|y| {
            terminal
                .backend()
                .buffer()
                .content()
                .iter()
                .skip(y as usize * 42)
                .take(42)
                .map(|cell| cell.symbol())
                .collect::<String>()
        })
        .collect::<Vec<_>>();
    let mcp_row = rows
        .iter()
        .position(|row| row.contains("MCP"))
        .expect("MCP heading");
    assert!(
        rows[mcp_row.saturating_sub(1)]
            .trim_matches(|ch: char| ch == '▎' || ch.is_whitespace())
            .is_empty(),
        "{rows:?}"
    );
    assert!(rendered.contains("Todos  1"), "{rendered}");
    assert!(rendered.contains("important todo"), "{rendered}");
}

#[test]
fn mcp_list_respects_width_and_height_limits() {
    let mut state = TuiState::default();
    state.set_language(Some(crate::tui::i18n::Language::En));
    state.set_mcp_servers(
        (0..6)
            .map(|index| crate::mcp::McpServerCatalogEntry {
                name: format!("very-long-mcp-server-name-{index}"),
                enabled: true,
                status: crate::mcp::McpServerStatus::Online { tool_count: 12 },
            })
            .collect(),
    );
    let backend = TestBackend::new(24, 10);
    let mut terminal = Terminal::new(backend).expect("terminal");
    terminal
        .draw(|frame| render_sidebar(frame, &mut state, frame.area(), Theme::dark()))
        .expect("draw");
    let rows = (0..terminal.backend().buffer().area.height)
        .map(|y| {
            terminal
                .backend()
                .buffer()
                .content()
                .iter()
                .skip(y as usize * 24)
                .take(24)
                .map(|cell| cell.symbol())
                .collect::<String>()
        })
        .collect::<Vec<_>>();

    assert!(
        rows.iter().filter(|row| row.contains("12 tools")).count() <= 2,
        "{rows:?}"
    );
    assert!(rows.iter().all(|row| display_width(row) <= 24), "{rows:?}");
    assert!(rows.iter().any(|row| row.contains("Ctrl-X B")), "{rows:?}");
    assert!(
        rows.iter().all(|row| !row.contains("Scroll panel")),
        "{rows:?}"
    );
}

#[test]
fn sidebar_collapses_context_mcp_and_todo_sections() {
    let mut state = TuiState::default();
    state.set_language(Some(crate::tui::i18n::Language::En));
    state.sidebar_context_expanded = false;
    state.sidebar_mcp_expanded = false;
    state.sidebar_todos_expanded = false;
    state.set_token_usage(crate::tui::state::ModelTokenUsage {
        used_tokens: 10_000,
        context_window_tokens: 20_000,
        input_tokens: 10_000,
        output_tokens: 0,
        cached_tokens: 0,
        cache_report: None,
        prompt_composition: vec![crate::agent::PromptCompositionEntry {
            category: "messages".into(),
            estimated_tokens: 10_000,
            segments: 1,
        }],
    });
    state.set_mcp_servers(vec![crate::mcp::McpServerCatalogEntry {
        name: "docs".into(),
        enabled: true,
        status: crate::mcp::McpServerStatus::Online { tool_count: 2 },
    }]);
    state.latest_todo = Some(crate::tui::timeline::TodoView {
        items: vec![crate::agent::TodoItem {
            id: "todo".into(),
            content: "hidden todo".into(),
            status: crate::agent::TodoStatus::Pending,
        }],
        auto_continue: crate::agent::AutoContinueState::default(),
    });
    let backend = TestBackend::new(42, 24);
    let mut terminal = Terminal::new(backend).expect("terminal");
    terminal
        .draw(|frame| render_sidebar(frame, &mut state, frame.area(), Theme::dark()))
        .expect("draw");
    let rendered = terminal
        .backend()
        .buffer()
        .content()
        .iter()
        .map(|cell| cell.symbol())
        .collect::<String>();

    assert!(rendered.contains("Node"), "{rendered}");
    assert!(rendered.contains("▶ Context"), "{rendered}");
    assert!(rendered.contains("▶ MCP"), "{rendered}");
    assert!(rendered.contains("▶ Todos"), "{rendered}");
    assert!(!rendered.contains("System prompt"), "{rendered}");
    assert!(!rendered.contains("docs"), "{rendered}");
    assert!(!rendered.contains("hidden todo"), "{rendered}");
}

#[test]
fn sidebar_scroll_reveals_overflow_content() {
    let mut state = TuiState::default();
    state.sidebar_scroll = 3;
    state.set_mcp_servers(
        (0..8)
            .map(|index| crate::mcp::McpServerCatalogEntry {
                name: format!("server-{index}"),
                enabled: true,
                status: crate::mcp::McpServerStatus::Online {
                    tool_count: index + 1,
                },
            })
            .collect(),
    );
    state.latest_todo = Some(crate::tui::timeline::TodoView {
        items: (0..5)
            .map(|index| crate::agent::TodoItem {
                id: format!("todo-{index}"),
                content: format!("long scrolling todo item {index}"),
                status: crate::agent::TodoStatus::Pending,
            })
            .collect(),
        auto_continue: crate::agent::AutoContinueState::default(),
    });
    let backend = TestBackend::new(42, 16);
    let mut terminal = Terminal::new(backend).expect("terminal");
    terminal
        .draw(|frame| render_sidebar(frame, &mut state, frame.area(), Theme::dark()))
        .expect("draw");

    assert!(state.sidebar_max_scroll > 0);
    assert_eq!(state.sidebar_scroll, 3.min(state.sidebar_max_scroll));
}

#[test]
fn sidebar_renders_mcp_server_list() {
    let mut state = TuiState::default();
    state.set_language(Some(crate::tui::i18n::Language::En));
    state.set_mcp_servers(vec![
        crate::mcp::McpServerCatalogEntry {
            name: "docs".into(),
            enabled: true,
            status: crate::mcp::McpServerStatus::Online { tool_count: 3 },
        },
        crate::mcp::McpServerCatalogEntry {
            name: "broken".into(),
            enabled: true,
            status: crate::mcp::McpServerStatus::Offline {
                message: "unreachable".into(),
            },
        },
    ]);
    let backend = TestBackend::new(42, 18);
    let mut terminal = Terminal::new(backend).expect("terminal");
    terminal
        .draw(|frame| render_sidebar(frame, &mut state, frame.area(), Theme::dark()))
        .expect("draw");
    let rendered = terminal
        .backend()
        .buffer()
        .content()
        .iter()
        .map(|cell| cell.symbol())
        .collect::<String>();

    for expected in ["MCP", "docs", "3 tools", "broken", "Offline"] {
        assert!(
            rendered.contains(expected),
            "missing {expected:?}: {rendered}"
        );
    }
}

#[test]
fn context_bar_fills_used_share_and_leaves_capacity_visible() {
    let composition = vec![
        crate::agent::PromptCompositionEntry {
            category: "system".into(),
            estimated_tokens: 4_000,
            segments: 1,
        },
        crate::agent::PromptCompositionEntry {
            category: "messages".into(),
            estimated_tokens: 6_000,
            segments: 1,
        },
    ];
    let line = context_bar_line(20, &composition, 10_000, 10_000, 20_000, Theme::dark());
    assert_eq!(line.width(), 20);
    assert_eq!(line.spans.len(), 20);
    assert_eq!(
        line.spans
            .iter()
            .filter(|span| span.style.fg == Some(Theme::dark().elevated_bg))
            .count(),
        10
    );
}

#[test]
fn context_bar_keeps_small_positive_categories_visible() {
    let units = spread_bar_boundaries(
        allocate_bar_units(&[7_600, 7_300, 4_100, 668_800, 4_600], 43 * 8),
        43 * 8,
        8,
    );

    assert_eq!(units.iter().sum::<usize>(), 43 * 8);
    assert!(units.iter().all(|units| *units >= 1), "{units:?}");
    assert!(units[3] > units[0]);
    let boundaries = units
        .iter()
        .scan(0usize, |total, units| {
            *total += *units;
            Some(*total)
        })
        .take(units.len() - 1)
        .filter(|boundary| *boundary % 8 != 0)
        .map(|boundary| boundary / 8)
        .collect::<Vec<_>>();
    assert_eq!(
        boundaries
            .iter()
            .copied()
            .collect::<std::collections::HashSet<_>>()
            .len(),
        boundaries.len(),
        "{units:?}"
    );
}

#[test]
fn context_bar_small_categories_use_two_color_cells() {
    let theme = Theme::dark();
    let segments = [
        (7_600, theme.notice),
        (7_300, theme.warning),
        (4_100, theme.success),
        (668_800, theme.user),
        (4_600, theme.error),
    ];
    let line = context_bar_spans(43, 43, &segments, theme);

    assert_eq!(line.width(), 43);
    for pair in segments.windows(2) {
        assert!(
            line.spans.iter().any(|span| {
                span.style.fg == Some(pair[0].1)
                    && span.style.bg == Some(pair[1].1)
                    && "▉▊▋▌▍▎▏".contains(span.content.as_ref())
            }),
            "missing boundary for {pair:?}"
        );
    }
}

#[test]
fn empty_context_bar_keeps_its_track_only_on_painted_panels() {
    for theme in [Theme::dark(), Theme::plain_for(None)] {
        let line = context_bar_spans(4, 0, &[], theme);

        assert_eq!(line.width(), 4);
        assert!(line.spans.iter().all(|span| span.content.as_ref() == "█"));
        assert!(
            line.spans
                .iter()
                .all(|span| span.style.fg == Some(theme.elevated_bg))
        );
    }

    let glass = context_bar_spans(4, 0, &[], Theme::glass());
    assert_eq!(glass.width(), 4);
    assert!(glass.spans.iter().all(|span| span.content.as_ref() == " "));
    assert!(glass.spans.iter().all(|span| span.style.fg.is_none()));
}

#[test]
fn context_bar_falls_back_to_used_color_without_composition() {
    let theme = Theme::dark();
    let line = context_bar_spans(4, 2, &[], theme);

    assert_eq!(line.width(), 4);
    assert_eq!(line.spans[0].style.fg, Some(theme.accent));
    assert_eq!(line.spans[1].style.fg, Some(theme.accent));
    assert_eq!(line.spans[2].content.as_ref(), "█");
    assert_eq!(line.spans[2].style.fg, Some(theme.elevated_bg));
    assert_eq!(line.spans[3].content.as_ref(), "█");
    assert_eq!(line.spans[3].style.fg, Some(theme.elevated_bg));
}

#[test]
fn context_bar_cell_uses_dominant_run_when_three_colors_share_a_cell() {
    let theme = Theme::dark();
    let cell = [
        Some(theme.notice),
        Some(theme.warning),
        Some(theme.success),
        Some(theme.success),
        Some(theme.success),
        Some(theme.user),
        Some(theme.user),
        Some(theme.user),
    ];
    let span = context_bar_cell(&cell, theme);

    assert_eq!(span.content.as_ref(), "█");
    assert_eq!(span.style.fg, Some(theme.user));
    assert_eq!(span.style.bg, Some(theme.element_bg));
}

#[test]
fn context_bar_handles_more_categories_than_subcells() {
    let theme = Theme::dark();
    let colors = [
        theme.user,
        theme.accent,
        theme.approval,
        theme.assistant,
        theme.notice,
        theme.warning,
        theme.success,
        theme.error,
    ];
    let segments = (0..12)
        .map(|index| (1, colors[index % colors.len()]))
        .collect::<Vec<_>>();
    let line = context_bar_spans(1, 1, &segments, theme);

    assert_eq!(line.width(), 1);
    assert_eq!(line.spans.len(), 1);
}

#[test]
fn tool_iteration_adds_prior_output_only_after_it_becomes_input() {
    let before = context_bar_line(
        40,
        &[crate::agent::PromptCompositionEntry {
            category: "messages".into(),
            estimated_tokens: 100,
            segments: 1,
        }],
        100,
        100,
        200,
        Theme::dark(),
    );
    let after = context_bar_line(
        40,
        &[crate::agent::PromptCompositionEntry {
            category: "messages".into(),
            estimated_tokens: 120,
            segments: 1,
        }],
        120,
        120,
        200,
        Theme::dark(),
    );
    let filled_cells = |line: &Line<'_>| {
        line.spans
            .iter()
            .filter(|span| span.style.fg != Some(Theme::dark().elevated_bg))
            .count()
    };

    assert_eq!(filled_cells(&before), 20);
    assert_eq!(filled_cells(&after), 24);
}

#[test]
fn sidebar_labels_align_by_terminal_display_width() {
    assert_eq!(display_width(&padded_label("模型", 10)), 10);
    assert_eq!(display_width(&padded_label("Git", 10)), 10);
    assert_eq!(display_width(&padded_label("服务商", 10)), 10);
}
