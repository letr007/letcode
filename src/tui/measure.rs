use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct CursorVisualPosition {
    pub row: usize,
    pub column: usize,
}

pub fn display_width(text: &str) -> usize {
    UnicodeWidthStr::width(text)
}

/// Compute max scroll (top-relative) given total transcript rows and viewport height.
///
/// This math is intentionally UI-neutral so state/input logic doesn't depend on component modules.
pub fn max_scroll(total_rows: usize, viewport_rows: u16) -> usize {
    total_rows.saturating_sub(viewport_rows as usize)
}

/// Convert bottom-relative scroll offset (0 = bottom/follow) into top-relative row offset.
pub fn resolved_scroll_offset(
    total_rows: usize,
    viewport_rows: u16,
    scroll_offset: usize,
    auto_scroll: bool,
) -> usize {
    let max_scroll = max_scroll(total_rows, viewport_rows);
    let bottom_offset = if auto_scroll {
        0
    } else {
        scroll_offset.min(max_scroll)
    };
    max_scroll.saturating_sub(bottom_offset)
}

/// Whether we're at the transcript bottom using bottom-relative scroll offset.
#[cfg(test)]
pub fn is_at_bottom(total_rows: usize, viewport_rows: u16, scroll_offset: usize) -> bool {
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

/// 与 `wrap_text_to_width` 相同的视觉换行，但同时返回每个视觉 chunk 在原文
/// 中的字符区间 `[start, end)`，用于把渲染后的选择/复制坐标映射回源文本。
///
/// 区间以 Unicode 字符数（非字节）为单位；多行原文中的 `\n` 不属于任何 chunk，
/// 与之相邻的 chunk 区间不会越界到下一行。
pub fn wrap_text_to_width_with_offsets(text: &str, width: usize) -> Vec<WrappedChunk> {
    let chunks = wrap_text_with_offsets(text, width);
    if chunks.is_empty() {
        vec![WrappedChunk {
            text: String::new(),
            source_start_char: 0,
            source_end_char: 0,
        }]
    } else {
        chunks
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WrappedChunk {
    pub text: String,
    pub source_start_char: usize,
    pub source_end_char: usize,
}

fn wrap_text_with_offsets(text: &str, width: usize) -> Vec<WrappedChunk> {
    let mut chunks = Vec::new();
    let mut char_cursor = 0usize;
    let segments = text.split('\n').collect::<Vec<&str>>();
    let segment_count = segments.len();
    for (line_idx, line) in segments.iter().enumerate() {
        let line_start_char = char_cursor;
        let wrapped = wrap_line_to_width(line, width);
        if line.is_empty() {
            chunks.push(WrappedChunk {
                text: String::new(),
                source_start_char: line_start_char,
                source_end_char: line_start_char,
            });
        } else {
            let mut pos = line_start_char;
            for chunk in &wrapped {
                let n = chunk.chars().count();
                chunks.push(WrappedChunk {
                    text: chunk.clone(),
                    source_start_char: pos,
                    source_end_char: pos + n,
                });
                pos += n;
            }
        }
        // 前进到下一段起点：本段字符 + 1 个 '\n' 分隔符（如果后续还有内容）
        char_cursor = line_start_char + line.chars().count();
        if line_idx + 1 < segment_count {
            char_cursor += 1;
        }
    }
    chunks
}

pub fn wrapped_row_count(text: &str, width: usize) -> usize {
    wrap_text_to_width(text, width).len().max(1)
}

#[cfg(test)]
pub(crate) fn cursor_visual_position(
    text: &str,
    width: usize,
    cursor_byte_index: usize,
) -> CursorVisualPosition {
    let cursor_byte_index = clamp_to_char_boundary(text, cursor_byte_index.min(text.len()));
    end_cursor_visual_position(&text[..cursor_byte_index], width)
}

#[cfg(test)]
pub(crate) fn end_cursor_visual_position(text: &str, width: usize) -> CursorVisualPosition {
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

#[cfg(test)]
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

    for grapheme in line.graphemes(true) {
        let grapheme_width = display_width(grapheme);

        if grapheme_width > width && current.is_empty() {
            if !emitted_tiny_width_placeholder {
                rows.push(String::new());
                emitted_tiny_width_placeholder = true;
            }
            continue;
        }

        if current_width > 0 && current_width.saturating_add(grapheme_width) > width {
            rows.push(current);
            current = String::new();
            current_width = 0;
        }

        current.push_str(grapheme);
        current_width = current_width.saturating_add(grapheme_width);
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
}
