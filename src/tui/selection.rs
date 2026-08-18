use std::collections::HashSet;

use super::{
    state::TuiState,
    transcript_render::{CopyJoin, CopyMode, inclusive_grapheme_bounds},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SourceSliceKey {
    item_index: usize,
    block_index: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SourceSliceAcc {
    key: SourceSliceKey,
    start: usize,
    end: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct AtomicSourceSliceKey {
    item_index: usize,
    block_index: usize,
    start: usize,
    end: usize,
}

/// 提取选择范围内的文本内容。
///
/// 关键点：复制来源是 `TranscriptRenderCacheEntry::{source_blocks,line_origins}`，
/// 而不是渲染后的 `Line<'static>`。因此：
/// - 视觉 soft-wrap 不会泄漏成真实换行；
/// - card 边框、padding、badge、separator 等装饰字符不会进剪贴板；
/// - 同一 source block 的多条渲染行会合并为一次切片，自然保留原文中的 `\n`。
#[cfg(test)]
pub fn extract_selected_text(
    state: &TuiState,
    presentation: &crate::tui::presentation::TuiPresentationState,
) -> String {
    extract_selected_text_with_cache(state, &presentation.transcript_render_cache)
}

pub fn extract_selected_text_with_cache(
    state: &TuiState,
    cache: &crate::tui::components::transcript::TranscriptRenderCache,
) -> String {
    let Some(selection) = &state.text_selection else {
        return String::new();
    };

    let (start, end) = selection.normalize();
    let mut result = String::new();
    let mut pending_newlines = 0usize;
    let mut previous: Option<SourceSliceAcc> = None;
    let mut copied_atomic = HashSet::new();

    for item_idx in start.item_index..=end.item_index {
        let mut copied_in_item = false;
        let Some(entry) = cache.entries().get(item_idx) else {
            break;
        };
        let line_start = if item_idx == start.item_index {
            start.rendered_line_offset.min(entry.document.lines.len())
        } else {
            0
        };
        let line_end = if item_idx == end.item_index {
            (end.rendered_line_offset + 1).min(entry.document.lines.len())
        } else {
            entry.document.lines.len()
        };

        for line_idx in line_start..line_end {
            let selected_start =
                if item_idx == start.item_index && line_idx == start.rendered_line_offset {
                    start.char_offset
                } else {
                    0
                };
            let selected_end = if item_idx == end.item_index && line_idx == end.rendered_line_offset
            {
                end.char_offset
            } else {
                usize::MAX
            };
            let slices = span_ranges_for_selection(
                &entry.document.lines[line_idx],
                selected_start,
                selected_end,
            );
            let mut copied_on_line = false;
            for (span_start, overlap_start, overlap_end, range, copy_mode, copy_join) in slices {
                let (slice_start, slice_end) = if copy_mode == CopyMode::Atomic {
                    (range.start, range.end)
                } else {
                    (
                        range.start + overlap_start.saturating_sub(span_start),
                        range.start + overlap_end.saturating_sub(span_start),
                    )
                };
                let atomic_key = AtomicSourceSliceKey {
                    item_index: item_idx,
                    block_index: range.block_index,
                    start: range.start,
                    end: range.end,
                };
                if copy_mode == CopyMode::Atomic && !copied_atomic.insert(atomic_key) {
                    continue;
                }
                let slice = SourceSliceAcc {
                    key: SourceSliceKey {
                        item_index: item_idx,
                        block_index: range.block_index,
                    },
                    start: slice_start,
                    end: slice_end,
                };
                if slice.start >= slice.end {
                    continue;
                }
                let separated_by_newline = pending_newlines > 0;
                if !result.is_empty() {
                    result.extend(std::iter::repeat_n('\n', pending_newlines));
                }
                pending_newlines = 0;

                let source = &entry.document.source_blocks[slice.key.block_index].source;
                if previous.is_some_and(|last| last.key == slice.key && slice.start <= last.end) {
                    let start = previous.unwrap().end.min(slice.end);
                    if start < slice.end {
                        result.push_str(&slice_chars(source, start, slice.end));
                    }
                } else {
                    if !result.is_empty() && !separated_by_newline && copy_join == CopyJoin::Space {
                        result.push(' ');
                    }
                    result.push_str(&slice_chars(source, slice.start, slice.end));
                }
                previous = Some(slice);
                copied_on_line = true;
                copied_in_item = true;
            }
            if copied_on_line
                && matches!(
                    entry.document.break_after(line_idx),
                    Some(
                        crate::tui::transcript_render::Break::HardBreak
                            | crate::tui::transcript_render::Break::BlockBreak
                    )
                )
            {
                pending_newlines = pending_newlines.saturating_add(1);
                previous = None;
            }
        }
        if item_idx < end.item_index && !result.is_empty() && copied_in_item {
            pending_newlines = pending_newlines.max(1);
            previous = None;
        }
    }

    result
}

fn slice_chars(text: &str, start: usize, end: usize) -> String {
    text.chars()
        .skip(start)
        .take(end.saturating_sub(start))
        .collect()
}

fn span_ranges_for_selection(
    line: &crate::tui::transcript_render::Line<ratatui::style::Style>,
    selected_start: usize,
    selected_end: usize,
) -> Vec<(
    usize,
    usize,
    usize,
    crate::tui::transcript_render::SourceRange,
    CopyMode,
    CopyJoin,
)> {
    let mut visual_offset = 0usize;
    line.spans
        .iter()
        .filter_map(|span| {
            let span_start = visual_offset;
            let span_end = span_start + span.text.chars().count();
            visual_offset = span_end;
            let range = span.source?;
            if selected_end < span_start || selected_start >= span_end {
                return None;
            }
            let local_start = selected_start
                .saturating_sub(span_start)
                .min(span.text.chars().count());
            let local_end = selected_end
                .saturating_sub(span_start)
                .min(span.text.chars().count());
            let (local_start, local_end) =
                inclusive_grapheme_bounds(&span.text, local_start, local_end);
            (local_start < local_end).then_some((
                span_start,
                span_start + local_start,
                span_start + local_end,
                range,
                span.copy_mode,
                span.copy_join,
            ))
        })
        .collect()
}

#[cfg(test)]
mod tests {
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
        let mut presentation = crate::tui::presentation::TuiPresentationState::default();
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
        presentation
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

        assert_eq!(
            extract_selected_text(&state, &presentation),
            "first\nsecond"
        );
    }

    #[test]
    fn atomic_source_is_copied_once_across_wrapped_rows() {
        use crate::tui::{
            components::transcript::TranscriptRenderCacheEntry,
            transcript_render::{Break, Document, Line, SourceRange, Span},
        };

        let mut state = TuiState::default();
        let mut presentation = crate::tui::presentation::TuiPresentationState::default();
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
        presentation
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

        assert_eq!(extract_selected_text(&state, &presentation), source);
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
            let mut presentation = crate::tui::presentation::TuiPresentationState::default();
            presentation
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

            assert_eq!(extract_selected_text(&state, &presentation), expected);
        }

        let mut state = TuiState::default();
        let mut presentation = crate::tui::presentation::TuiPresentationState::default();
        presentation
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
        assert_eq!(extract_selected_text(&state, &presentation), r"\alpha^2");
    }

    #[test]
    fn copies_only_source_backed_spans_across_cjk_and_chrome() {
        use crate::tui::{
            components::transcript::TranscriptRenderCacheEntry,
            transcript_render::{Break, Document, Line, SourceRange, Span},
        };

        let mut state = TuiState::default();
        let mut presentation = crate::tui::presentation::TuiPresentationState::default();
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
        presentation
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

        assert_eq!(extract_selected_text(&state, &presentation), "你好 world");
    }

    #[test]
    fn soft_wrap_does_not_insert_newline_and_reverse_selection_normalizes() {
        use crate::tui::{
            components::transcript::TranscriptRenderCacheEntry,
            transcript_render::{Break, Document, Line, SourceRange, Span},
        };

        let mut state = TuiState::default();
        let mut presentation = crate::tui::presentation::TuiPresentationState::default();
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
        presentation
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

        assert_eq!(
            extract_selected_text(&state, &presentation),
            "repeat repeat"
        );
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
            let mut presentation = crate::tui::presentation::TuiPresentationState::default();
            presentation
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
            assert_eq!(extract_selected_text(&state, &presentation), expected);
        }
    }
}
