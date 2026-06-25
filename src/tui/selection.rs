use super::state::TuiState;

/// 提取选择范围内的文本内容
pub fn extract_selected_text(state: &TuiState) -> String {
    let Some(selection) = &state.text_selection else {
        return String::new();
    };

    let (start, end) = selection.normalize();
    let cache = &state.transcript_render_cache;
    let mut result = String::new();

    // 遍历涉及的所有 TimelineItems
    for item_idx in start.item_index..=end.item_index {
        if item_idx >= cache.entries().len() {
            break;
        }

        let lines = &cache.entries()[item_idx].lines;
        let lines_len = lines.len();
        if lines_len == 0 {
            continue;
        }

        // 确定该 Item 内的行范围（边界 clamp，防止 stale anchor 越界 panic）
        let line_start = if item_idx == start.item_index {
            start.rendered_line_offset.min(lines_len)
        } else {
            0
        };

        let line_end = if item_idx == end.item_index {
            (end.rendered_line_offset + 1).min(lines_len)
        } else {
            lines_len
        };
        if line_start >= line_end {
            continue;
        }

        // 提取并裁剪文本
        for (local_line_idx, line) in lines[line_start..line_end].iter().enumerate() {
            let global_line_idx = line_start + local_line_idx;
            let line_text = line.to_string();

            let text = if item_idx == start.item_index
                && global_line_idx == start.rendered_line_offset
            {
                // 第一行：从 start.char_offset 开始
                let chars: Vec<char> = line_text.chars().collect();
                let trimmed: String = chars[start.char_offset.min(chars.len())..]
                    .iter()
                    .collect();

                if item_idx == end.item_index && global_line_idx == end.rendered_line_offset {
                    // 同时也是最后一行
                    let chars: Vec<char> = trimmed.chars().collect();
                    chars[..end
                        .char_offset
                        .saturating_sub(start.char_offset)
                        .min(chars.len())]
                        .iter()
                        .collect()
                } else {
                    trimmed
                }
            } else if item_idx == end.item_index && global_line_idx == end.rendered_line_offset {
                // 最后一行：到 end.char_offset 结束
                let chars: Vec<char> = line_text.chars().collect();
                chars[..end.char_offset.min(chars.len())].iter().collect()
            } else {
                // 中间行：完整复制
                line_text
            };

            result.push_str(&text);
            if global_line_idx < line_end - 1 || item_idx < end.item_index {
                result.push('\n');
            }
        }
    }

    result
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
