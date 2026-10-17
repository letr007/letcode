use super::*;
use crate::tui::state::{SelectionAnchor, TextSelection};

#[test]
fn test_selection_anchor_ordering() {
    let anchor1 = SelectionAnchor {
        item_index: 0,
        rendered_line_offset: 0,
        char_offset: 5,
    };
    let anchor2 = SelectionAnchor {
        item_index: 0,
        rendered_line_offset: 0,
        char_offset: 10,
    };
    let anchor3 = SelectionAnchor {
        item_index: 1,
        rendered_line_offset: 0,
        char_offset: 0,
    };

    assert!(anchor1 < anchor2);
    assert!(anchor2 < anchor3);
    assert!(anchor1 < anchor3);
}

#[test]
fn test_selection_normalize() {
    let selection = TextSelection {
        start: SelectionAnchor {
            item_index: 1,
            rendered_line_offset: 0,
            char_offset: 0,
        },
        end: SelectionAnchor {
            item_index: 0,
            rendered_line_offset: 0,
            char_offset: 5,
        },
    };

    let (start, end) = selection.normalize();
    assert_eq!(start.item_index, 0);
    assert_eq!(end.item_index, 1);
}

#[test]
fn separate_source_blocks_preserve_visible_line_breaks() {
    use crate::tui::{
        components::transcript::TranscriptRenderCacheEntry,
        transcript_render::{Break, Document, Line, SourceRange, Span},
    };

    let mut state = TuiState::default();
    let mut document = Document::<ratatui::style::Style>::default();
    let first = document.add_source("first");
    let second = document.add_source("second");
    document.push_line(
        Line {
            spans: vec![Span::source(
                "first",
                ratatui::style::Style::default(),
                SourceRange::new(first, 0, 5),
            )],
        },
        Break::BlockBreak,
    );
    document.push_line(
        Line {
            spans: vec![Span::source(
                "second",
                ratatui::style::Style::default(),
                SourceRange::new(second, 0, 6),
            )],
        },
        Break::End,
    );
    state
        .transcript_render_cache
        .set_entries_for_test(vec![TranscriptRenderCacheEntry {
            revision: None,
            document,
        }]);
    state.text_selection = Some(TextSelection {
        start: SelectionAnchor {
            item_index: 0,
            rendered_line_offset: 0,
            char_offset: 0,
        },
        end: SelectionAnchor {
            item_index: 0,
            rendered_line_offset: 1,
            char_offset: 6,
        },
    });

    assert_eq!(extract_selected_text(&state), "first\nsecond");
}

#[test]
fn atomic_source_is_copied_once_across_wrapped_rows() {
    use crate::tui::{
        components::transcript::TranscriptRenderCacheEntry,
        transcript_render::{Break, Document, Line, SourceRange, Span},
    };

    let mut state = TuiState::default();
    let mut document = Document::<ratatui::style::Style>::default();
    let source = "graph TD\nA[Start]\nB[Finish]";
    let block = document.add_source(source);
    let source_end = source.chars().count();
    document.push_line(
        Line {
            spans: vec![Span::source_atomic(
                "Start",
                ratatui::style::Style::default(),
                SourceRange::new(block, 0, source_end),
            )],
        },
        Break::HardBreak,
    );
    document.push_line(
        Line {
            spans: vec![Span::source_atomic(
                "Finish",
                ratatui::style::Style::default(),
                SourceRange::new(block, 0, source_end),
            )],
        },
        Break::End,
    );
    state
        .transcript_render_cache
        .set_entries_for_test(vec![TranscriptRenderCacheEntry {
            revision: None,
            document,
        }]);
    state.text_selection = Some(TextSelection {
        start: SelectionAnchor {
            item_index: 0,
            rendered_line_offset: 0,
            char_offset: 0,
        },
        end: SelectionAnchor {
            item_index: 0,
            rendered_line_offset: 1,
            char_offset: 6,
        },
    });

    assert_eq!(extract_selected_text(&state), source);
}

#[test]
fn mixed_text_and_multiple_atomic_sources_copy_in_both_directions() {
    use crate::tui::{
        components::transcript::TranscriptRenderCacheEntry,
        transcript_render::{Break, Document, Line, SourceRange, Span},
    };

    let mut document = Document::<ratatui::style::Style>::default();
    let before = document.add_source("before ");
    let alpha = document.add_source(r"\alpha^2");
    let between = document.add_source(" and ");
    let beta = document.add_source(r"\beta");
    let after = document.add_source(" after");
    let alpha_end = document.source_blocks[alpha].source.chars().count();
    let beta_end = document.source_blocks[beta].source.chars().count();
    document.push_line(
        Line {
            spans: vec![
                Span::source(
                    "before ",
                    ratatui::style::Style::default(),
                    SourceRange::new(before, 0, 7),
                ),
                Span::source_atomic(
                    "α²",
                    ratatui::style::Style::default(),
                    SourceRange::new(alpha, 0, alpha_end),
                ),
                Span::source(
                    " and ",
                    ratatui::style::Style::default(),
                    SourceRange::new(between, 0, 5),
                ),
                Span::source_atomic(
                    "β",
                    ratatui::style::Style::default(),
                    SourceRange::new(beta, 0, beta_end),
                ),
                Span::source(
                    " after",
                    ratatui::style::Style::default(),
                    SourceRange::new(after, 0, 6),
                ),
            ],
        },
        Break::End,
    );
    assert!(document.validate());
    let line_end = document.lines[0]
        .spans
        .iter()
        .map(|span| span.text.chars().count())
        .sum();
    let expected = r"before \alpha^2 and \beta after";

    for (start, end) in [(0, line_end), (line_end, 0)] {
        let mut state = TuiState::default();
        state
            .transcript_render_cache
            .set_entries_for_test(vec![TranscriptRenderCacheEntry {
                revision: None,
                document: document.clone(),
            }]);
        state.text_selection = Some(TextSelection {
            start: SelectionAnchor {
                item_index: 0,
                rendered_line_offset: 0,
                char_offset: start,
            },
            end: SelectionAnchor {
                item_index: 0,
                rendered_line_offset: 0,
                char_offset: end,
            },
        });

        assert_eq!(extract_selected_text(&state), expected);
    }

    let mut state = TuiState::default();
    state
        .transcript_render_cache
        .set_entries_for_test(vec![TranscriptRenderCacheEntry {
            revision: None,
            document,
        }]);
    state.text_selection = Some(TextSelection {
        start: SelectionAnchor {
            item_index: 0,
            rendered_line_offset: 0,
            char_offset: 7,
        },
        end: SelectionAnchor {
            item_index: 0,
            rendered_line_offset: 0,
            char_offset: 7,
        },
    });
    assert_eq!(extract_selected_text(&state), r"\alpha^2");
}

#[test]
fn copies_only_source_backed_spans_across_cjk_and_chrome() {
    use crate::tui::{
        components::transcript::TranscriptRenderCacheEntry,
        transcript_render::{Break, Document, Line, SourceRange, Span},
    };

    let mut state = TuiState::default();
    let mut document = Document::<ratatui::style::Style>::default();
    let block = document.add_source("你好 world");
    document.push_line(
        Line {
            spans: vec![
                Span::decoration("┃  ", ratatui::style::Style::default()),
                Span::source(
                    "你好 ",
                    ratatui::style::Style::default(),
                    SourceRange::new(block, 0, 3),
                ),
                Span::decoration("[queued] ", ratatui::style::Style::default()),
                Span::source(
                    "world",
                    ratatui::style::Style::default(),
                    SourceRange::new(block, 3, 8),
                ),
            ],
        },
        Break::End,
    );
    state
        .transcript_render_cache
        .set_entries_for_test(vec![TranscriptRenderCacheEntry {
            revision: None,
            document,
        }]);
    state.text_selection = Some(TextSelection {
        start: SelectionAnchor {
            item_index: 0,
            rendered_line_offset: 0,
            char_offset: 0,
        },
        end: SelectionAnchor {
            item_index: 0,
            rendered_line_offset: 0,
            char_offset: 20,
        },
    });

    assert_eq!(extract_selected_text(&state), "你好 world");
}

#[test]
fn soft_wrap_does_not_insert_newline_and_reverse_selection_normalizes() {
    use crate::tui::{
        components::transcript::TranscriptRenderCacheEntry,
        transcript_render::{Break, Document, Line, SourceRange, Span},
    };

    let mut state = TuiState::default();
    let mut document = Document::<ratatui::style::Style>::default();
    let block = document.add_source("repeat repeat");
    document.push_line(
        Line {
            spans: vec![Span::source(
                "repeat ",
                ratatui::style::Style::default(),
                SourceRange::new(block, 0, 7),
            )],
        },
        Break::SoftWrap,
    );
    document.push_line(
        Line {
            spans: vec![Span::source(
                "repeat",
                ratatui::style::Style::default(),
                SourceRange::new(block, 7, 13),
            )],
        },
        Break::End,
    );
    state
        .transcript_render_cache
        .set_entries_for_test(vec![TranscriptRenderCacheEntry {
            revision: None,
            document,
        }]);
    state.text_selection = Some(TextSelection {
        start: SelectionAnchor {
            item_index: 0,
            rendered_line_offset: 1,
            char_offset: 6,
        },
        end: SelectionAnchor {
            item_index: 0,
            rendered_line_offset: 0,
            char_offset: 0,
        },
    });

    assert_eq!(extract_selected_text(&state), "repeat repeat");
}

#[test]
fn release_at_a_grapheme_start_includes_that_grapheme_in_both_directions() {
    use crate::tui::{
        components::transcript::TranscriptRenderCacheEntry,
        transcript_render::{Break, Document, Line, SourceRange, Span},
    };

    let mut document = Document::<ratatui::style::Style>::default();
    let block = document.add_source("a你e\u{301}👩‍💻");
    document.push_line(
        Line {
            spans: vec![Span::source(
                "a你e\u{301}👩‍💻",
                ratatui::style::Style::default(),
                SourceRange::new(block, 0, 7),
            )],
        },
        Break::End,
    );

    for (start, end, expected) in [(0, 1, "a你"), (1, 0, "a你")] {
        let mut state = TuiState::default();
        state
            .transcript_render_cache
            .set_entries_for_test(vec![TranscriptRenderCacheEntry {
                revision: None,
                document: document.clone(),
            }]);
        state.text_selection = Some(TextSelection {
            start: SelectionAnchor {
                item_index: 0,
                rendered_line_offset: 0,
                char_offset: start,
            },
            end: SelectionAnchor {
                item_index: 0,
                rendered_line_offset: 0,
                char_offset: end,
            },
        });
        assert_eq!(extract_selected_text(&state), expected);
    }
}

#[test]
fn wrapped_table_cells_copy_without_extra_separator_spaces() {
    use crate::tui::{
        components::transcript::TranscriptRenderCacheEntry,
        markdown::{MarkdownRenderOptions, render_markdown_document},
    };

    let document = render_markdown_document(
        "| 名称 | 说明 |\n| --- | --- |\n| 组件 | 这是一个很长的说明文字 |",
        crate::tui::theme::Theme::dark(),
        MarkdownRenderOptions::new(16),
    );
    assert!(document.validate(), "{document:?}");
    let last_line = document.lines.len() - 1;

    let mut state = TuiState::default();
    state
        .transcript_render_cache
        .set_entries_for_test(vec![TranscriptRenderCacheEntry {
            revision: None,
            document,
        }]);
    state.text_selection = Some(TextSelection {
        start: SelectionAnchor {
            item_index: 0,
            rendered_line_offset: 0,
            char_offset: 0,
        },
        end: SelectionAnchor {
            item_index: 0,
            rendered_line_offset: last_line,
            char_offset: usize::MAX,
        },
    });

    let copied = extract_selected_text(&state);
    assert!(copied.contains("这是一个很长的说明文字"), "{copied}");
    assert!(!copied.contains("  "), "{copied}");
}
