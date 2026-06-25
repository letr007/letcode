use super::state::TuiState;

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

/// 提取选择范围内的文本内容。
///
/// 关键点：复制来源是 `TranscriptRenderCacheEntry::{source_blocks,line_origins}`，
/// 而不是渲染后的 `Line<'static>`。因此：
/// - 视觉 soft-wrap 不会泄漏成真实换行；
/// - card 边框、padding、badge、separator 等装饰字符不会进剪贴板；
/// - 同一 source block 的多条渲染行会合并为一次切片，自然保留原文中的 `\n`。
pub fn extract_selected_text(state: &TuiState) -> String {
    let Some(selection) = &state.text_selection else {
        return String::new();
    };

    let (start, end) = selection.normalize();
    let cache = &state.transcript_render_cache;
    let mut slices: Vec<SourceSliceAcc> = Vec::new();

    for item_idx in start.item_index..=end.item_index {
        if item_idx >= cache.entries().len() {
            break;
        }

        let entry = &cache.entries()[item_idx];
        if entry.lines.is_empty() {
            continue;
        }

        let line_start = if item_idx == start.item_index {
            start.rendered_line_offset.min(entry.lines.len())
        } else {
            0
        };
        let line_end = if item_idx == end.item_index {
            (end.rendered_line_offset + 1).min(entry.lines.len())
        } else {
            entry.lines.len()
        };
        if line_start >= line_end {
            continue;
        }

        for line_idx in line_start..line_end {
            let origin = &entry.line_origins[line_idx];
            let Some(block_index) = origin.block_index else {
                continue;
            };

            let selected_start =
                if item_idx == start.item_index && line_idx == start.rendered_line_offset {
                    start.char_offset.min(origin.content_char_len)
                } else {
                    0
                };
            let selected_end = if item_idx == end.item_index && line_idx == end.rendered_line_offset
            {
                end.char_offset.min(origin.content_char_len)
            } else {
                origin.content_char_len
            };

            if selected_start >= selected_end {
                continue;
            }

            let abs_start = origin.content_char_offset.saturating_add(selected_start);
            let abs_end = origin.content_char_offset.saturating_add(selected_end);
            let key = SourceSliceKey {
                item_index: item_idx,
                block_index,
            };

            match slices.last_mut() {
                Some(last) if last.key == key => {
                    last.start = last.start.min(abs_start);
                    last.end = last.end.max(abs_end);
                }
                _ => slices.push(SourceSliceAcc {
                    key,
                    start: abs_start,
                    end: abs_end,
                }),
            }
        }
    }

    let mut result = String::new();
    for slice in slices {
        let entry = &cache.entries()[slice.key.item_index];
        if slice.key.block_index >= entry.source_blocks.len() {
            continue;
        }
        let source = &entry.source_blocks[slice.key.block_index].source;
        let start = slice.start.min(source.chars().count());
        let end = slice.end.min(source.chars().count());
        if start >= end {
            continue;
        }
        if !result.is_empty() {
            result.push('\n');
        }
        result.push_str(&slice_chars(source, start, end));
    }

    result
}

fn slice_chars(text: &str, start: usize, end: usize) -> String {
    text.chars()
        .skip(start)
        .take(end.saturating_sub(start))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tui::state::{SelectionAnchor, TextSelection};

    #[test]
    fn test_extract_empty_selection() {
        let state = TuiState::default();
        assert_eq!(extract_selected_text(&state), "");
    }

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
}
