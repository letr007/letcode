use super::*;

#[test]
fn splits_preserve_trailing_empty_line() {
    assert_eq!(split_lines_preserving_trailing("a\n"), vec!["a", ""]);
    assert_eq!(split_lines_preserving_trailing(""), vec![""]);
}

#[test]
fn wraps_using_display_cell_width() {
    assert_eq!(wrap_text_to_width("a你b", 2), vec!["a", "你", "b"]);
    assert_eq!(wrap_text_to_width("a你b", 3), vec!["a你", "b"]);
    for width in [1, 2, 3] {
        let source = "a你b";
        for chunk in wrap_text_to_width_with_offsets(source, width) {
            assert!(display_width(&chunk.text) <= width);
            assert_eq!(
                chunk.text,
                source
                    .chars()
                    .skip(chunk.source_start_char)
                    .take(chunk.source_end_char - chunk.source_start_char)
                    .collect::<String>()
            );
        }
    }
}

#[test]
fn cursor_moves_to_next_row_when_last_chunk_fills_width_exactly() {
    // ASCII exact fill.
    assert_eq!(
        end_cursor_visual_position("abcd", 4),
        CursorVisualPosition { row: 1, column: 0 }
    );

    // Wide char exact fill (CJK width=2).
    assert_eq!(
        end_cursor_visual_position("你", 2),
        CursorVisualPosition { row: 1, column: 0 }
    );

    // Mixed exact fill.
    assert_eq!(
        end_cursor_visual_position("a你", 3),
        CursorVisualPosition { row: 1, column: 0 }
    );
}

#[test]
fn wrapping_does_not_split_extended_grapheme_clusters() {
    assert_eq!(wrap_text_to_width("e\u{301}x", 1), vec!["e\u{301}", "x"]);
    assert_eq!(wrap_text_to_width("👩‍💻x", 2), vec!["👩‍💻", "x"]);
}

#[test]
fn transcript_scroll_math_matches_bottom_relative_semantics() {
    assert_eq!(max_scroll(3, 10), 0);
    assert_eq!(resolved_scroll_offset(20, 5, 2, true), 15);
    assert_eq!(resolved_scroll_offset(20, 5, 18, false), 0);
    assert_eq!(resolved_scroll_offset(20, 5, 4, false), 11);

    assert!(is_at_bottom(20, 5, 0));
    assert!(is_at_bottom(3, 10, 4));
    assert!(!is_at_bottom(20, 5, 14));
}

#[test]
fn transcript_scroll_math_handles_large_documents() {
    let total_rows = 100_000usize;
    let viewport_rows = 30;
    let max = max_scroll(total_rows, viewport_rows);

    assert_eq!(max, total_rows - viewport_rows as usize);
    assert_eq!(
        resolved_scroll_offset(total_rows, viewport_rows, 0, true),
        max
    );
    assert_eq!(
        resolved_scroll_offset(total_rows, viewport_rows, max, false),
        0
    );
    assert_eq!(
        resolved_scroll_offset(total_rows, viewport_rows, 10_000, false),
        max - 10_000
    );
}
