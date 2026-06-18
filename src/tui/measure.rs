use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CursorVisualPosition {
    pub row: usize,
    pub column: usize,
}

pub fn display_width(text: &str) -> usize {
    UnicodeWidthStr::width(text)
}

/// Compute max scroll (top-relative) given total transcript rows and viewport height.
///
/// This math is intentionally UI-neutral so state/input logic doesn't depend on component modules.
pub fn max_scroll(total_rows: usize, viewport_rows: u16) -> u16 {
    u16::try_from(total_rows.saturating_sub(viewport_rows as usize)).unwrap_or(u16::MAX)
}

/// Convert bottom-relative scroll offset (0 = bottom/follow) into top-relative row offset.
pub fn resolved_scroll_offset(
    total_rows: usize,
    viewport_rows: u16,
    scroll_offset: u16,
    auto_scroll: bool,
) -> u16 {
    let max_scroll = max_scroll(total_rows, viewport_rows);
    let bottom_offset = if auto_scroll {
        0
    } else {
        scroll_offset.min(max_scroll)
    };
    max_scroll.saturating_sub(bottom_offset)
}

/// Whether we're at the transcript bottom using bottom-relative scroll offset.
pub fn is_at_bottom(total_rows: usize, viewport_rows: u16, scroll_offset: u16) -> bool {
    max_scroll(total_rows, viewport_rows) == 0 || scroll_offset == 0
}

pub fn split_lines_preserving_trailing(text: &str) -> Vec<&str> {
    text.split('\n').collect()
}

pub fn wrap_text_to_width(text: &str, width: usize) -> Vec<String> {
    let mut rows = Vec::new();

    for line in split_lines_preserving_trailing(text) {
        rows.extend(wrap_line_to_width(line, width));
    }

    if rows.is_empty() {
        rows.push(String::new());
    }

    rows
}

pub fn wrapped_row_count(text: &str, width: usize) -> usize {
    wrap_text_to_width(text, width).len().max(1)
}

pub fn cursor_visual_position(
    text: &str,
    width: usize,
    cursor_byte_index: usize,
) -> CursorVisualPosition {
    let cursor_byte_index = clamp_to_char_boundary(text, cursor_byte_index.min(text.len()));
    end_cursor_visual_position(&text[..cursor_byte_index], width)
}

pub fn end_cursor_visual_position(text: &str, width: usize) -> CursorVisualPosition {
    let lines = split_lines_preserving_trailing(text);
    let mut row = 0usize;
    let mut column = 0usize;

    for (line_index, line) in lines.iter().enumerate() {
        let wrapped = wrap_line_to_width(line, width);
        if line_index + 1 == lines.len() {
            row += wrapped.len().saturating_sub(1);
            column = wrapped
                .last()
                .map(|chunk| display_width(chunk))
                .unwrap_or_default();

            // Deterministic exact-width boundary: when the last visual chunk exactly
            // consumes the available width, the terminal cursor would be positioned
            // at the beginning of the next wrapped row.
            if width > 0 && column >= width {
                row = row.saturating_add(1);
                column = 0;
            }
        } else {
            row += wrapped.len();
            column = 0;
        }
    }

    CursorVisualPosition { row, column }
}

fn clamp_to_char_boundary(text: &str, index: usize) -> usize {
    if text.is_char_boundary(index) {
        return index;
    }

    let mut index = index;
    while index > 0 && !text.is_char_boundary(index) {
        index -= 1;
    }
    index
}

fn wrap_line_to_width(line: &str, width: usize) -> Vec<String> {
    if width == 0 || line.is_empty() {
        return vec![line.to_string()];
    }

    let mut rows = Vec::new();
    let mut current = String::new();
    let mut current_width = 0usize;
    let mut emitted_tiny_width_placeholder = false;

    for ch in line.chars() {
        let ch_width = UnicodeWidthChar::width(ch).unwrap_or(0);

        if ch_width == 0 {
            current.push(ch);
            continue;
        }

        if ch_width > width && current.is_empty() {
            if !emitted_tiny_width_placeholder {
                rows.push(String::new());
                emitted_tiny_width_placeholder = true;
            }
            continue;
        }

        if current_width > 0 && current_width.saturating_add(ch_width) > width {
            rows.push(current);
            current = String::new();
            current_width = 0;
        }

        current.push(ch);
        current_width = current_width.saturating_add(ch_width);
    }

    if current.is_empty() {
        if rows.is_empty() || !rows.last().is_some_and(String::is_empty) {
            rows.push(String::new());
        }
    } else {
        rows.push(current);
    }

    rows
}

#[cfg(test)]
mod tests {
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
    }

    #[test]
    fn counts_wrapped_rows_and_cursor_position() {
        assert_eq!(wrapped_row_count("hello\nworld", 10), 2);
        assert_eq!(
            end_cursor_visual_position("你你a\n", 2),
            CursorVisualPosition { row: 3, column: 0 }
        );
    }

    #[test]
    fn cursor_position_supports_arbitrary_byte_offsets() {
        assert_eq!(
            cursor_visual_position("hello", 10, 2),
            CursorVisualPosition { row: 0, column: 2 }
        );
        assert_eq!(
            cursor_visual_position("ab你cd", 4, 5),
            CursorVisualPosition { row: 1, column: 0 }
        );
        assert_eq!(
            cursor_visual_position("hi\nthere", 10, 3),
            CursorVisualPosition { row: 1, column: 0 }
        );
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
    fn wrapping_and_cursor_position_handle_mixed_ascii_and_cjk() {
        // width 4: "ab你" (1+1+2) fills exactly, so cursor moves to next row.
        assert_eq!(
            end_cursor_visual_position("ab你", 4),
            CursorVisualPosition { row: 1, column: 0 }
        );

        // width 4: next char starts new wrapped row.
        assert_eq!(
            end_cursor_visual_position("ab你c", 4),
            CursorVisualPosition { row: 1, column: 1 }
        );
    }

    #[test]
    fn tiny_widths_emit_placeholder_rows_instead_of_wide_rows() {
        assert_eq!(wrap_text_to_width("你a", 1), vec!["", "a"]);
        assert_eq!(wrap_text_to_width("你", 1), vec![""]);
        assert_eq!(wrapped_row_count("你", 1), 1);
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
}
