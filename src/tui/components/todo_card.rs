use ratatui::style::{Modifier, Style};
#[cfg(test)]
use ratatui::text::Line;

use crate::{
    agent::{TodoItem, TodoStatus},
    tui::{
        measure::{display_width, wrap_text_to_width_with_offsets},
        surface,
        theme::Theme,
        timeline::TodoView,
        transcript_render::{Break, Document, Line as RenderLine, SourceRange, Span},
    },
};

/// Legacy visual API. Its output is always the renderer bridge of the semantic
/// document, so it cannot introduce or discard provenance before projection.
#[cfg(test)]
pub(crate) fn render_todo_card_lines(
    todo: &TodoView,
    theme: Theme,
    width: usize,
) -> Vec<Line<'static>> {
    crate::tui::transcript_ratatui::document_to_ratatui(&render_todo_card_document(
        todo, theme, width,
    ))
}

pub fn render_todo_card_document(todo: &TodoView, theme: Theme, width: usize) -> Document<Style> {
    let mut document = Document::default();
    if width == 0 {
        return document;
    }

    push_todo_decoration(&mut document, "", theme, width);
    push_todo_decoration(&mut document, "# Todos", theme, width);
    push_todo_decoration(&mut document, "", theme, width);

    let items: Vec<TodoItem> = if todo.items.is_empty() {
        vec![TodoItem {
            id: String::new(),
            content: "No tasks".into(),
            status: TodoStatus::Pending,
        }]
    } else {
        todo.items.clone()
    };
    for item in &items {
        push_todo_item(&mut document, item, width, theme);
    }
    push_todo_decoration(&mut document, "", theme, width);
    document.finish();
    debug_assert!(document.validate());
    document
}

fn push_todo_decoration(document: &mut Document<Style>, text: &str, theme: Theme, width: usize) {
    let mut spans = vec![
        Span::decoration(surface::ACCENT_BAR_GLYPH, guide_style(theme)),
        Span::decoration("  ", fill_style(theme)),
    ];
    if !text.is_empty() {
        let style = if text.starts_with('#') {
            title_style(theme)
        } else {
            text_style(theme)
        };
        spans.push(Span::decoration(text, style));
    }
    let used = spans.iter().map(|span| display_width(&span.text)).sum();
    if width > used {
        spans.push(Span::decoration(
            " ".repeat(width - used),
            fill_style(theme),
        ));
    }
    document.push_line(RenderLine { spans }, Break::SoftWrap);
}

fn push_todo_item(document: &mut Document<Style>, item: &TodoItem, width: usize, theme: Theme) {
    let marker = status_marker(&item.status);
    let marker_width = display_width(marker) + 1;
    let content_width = width.saturating_sub(3 + marker_width).max(1);
    let block = document.add_source(item.content.clone());
    let chunks = wrap_text_to_width_with_offsets(&item.content, content_width);
    let chunks = if chunks.is_empty() {
        vec![crate::tui::measure::WrappedChunk {
            text: String::new(),
            source_start_char: 0,
            source_end_char: 0,
        }]
    } else {
        chunks
    };

    let chunk_count = chunks.len();
    for (index, chunk) in chunks.into_iter().enumerate() {
        let mut spans = vec![
            Span::decoration(surface::ACCENT_BAR_GLYPH, guide_style(theme)),
            Span::decoration("  ", fill_style(theme)),
        ];
        if index == 0 {
            spans.push(Span::decoration(
                format!("{marker} "),
                item_style(&item.status, theme),
            ));
        }
        if chunk.source_start_char < chunk.source_end_char {
            spans.push(Span::source(
                chunk.text,
                item_style(&item.status, theme),
                SourceRange::new(block, chunk.source_start_char, chunk.source_end_char),
            ));
        }
        let used = spans.iter().map(|span| display_width(&span.text)).sum();
        if width > used {
            spans.push(Span::decoration(
                " ".repeat(width - used),
                fill_style(theme),
            ));
        }
        document.push_line(
            RenderLine { spans },
            if index + 1 == chunk_count {
                Break::HardBreak
            } else {
                Break::SoftWrap
            },
        );
    }
}

fn guide_style(theme: Theme) -> Style {
    Style::default().fg(theme.card_guide()).bg(theme.root_bg)
}

fn title_style(theme: Theme) -> Style {
    Style::default()
        .fg(theme.approval)
        .bg(theme.element_bg)
        .add_modifier(Modifier::BOLD)
}

fn status_marker(status: &TodoStatus) -> &'static str {
    match status {
        TodoStatus::Pending => "[ ]",
        TodoStatus::InProgress => "[•]",
        TodoStatus::Blocked => "[!]",
        TodoStatus::Completed => "[✓]",
        TodoStatus::Cancelled => "[×]",
    }
}

fn text_style(theme: Theme) -> Style {
    Style::default().fg(theme.text).bg(theme.element_bg)
}

fn fill_style(theme: Theme) -> Style {
    Style::default().bg(theme.element_bg)
}

fn item_style(status: &TodoStatus, theme: Theme) -> Style {
    let color = match status {
        TodoStatus::Pending | TodoStatus::Completed => theme.text,
        TodoStatus::InProgress => theme.approval,
        TodoStatus::Blocked | TodoStatus::Cancelled => theme.error,
    };
    let style = Style::default().fg(color).bg(theme.element_bg);
    match status {
        TodoStatus::InProgress => style.add_modifier(Modifier::BOLD),
        _ => style,
    }
}

#[cfg(test)]
mod tests {
    use super::{render_todo_card_document, render_todo_card_lines};
    use crate::{
        agent::{AutoContinueState, TodoItem, TodoStatus},
        tui::{measure::display_width, theme::Theme, timeline::TodoView},
    };
    use ratatui::style::Modifier;

    #[test]
    fn todo_card_respects_display_width_with_cjk_content() {
        let todo = TodoView {
            items: vec![
                TodoItem {
                    id: "t1".into(),
                    content: "修复宽度并保持布局稳定".into(),
                    status: TodoStatus::InProgress,
                },
                TodoItem {
                    id: "t2".into(),
                    content: "验证已完成项目显示".into(),
                    status: TodoStatus::Completed,
                },
            ],
            auto_continue: AutoContinueState { enabled: true },
        };

        for width in [18usize, 24, 36] {
            let lines = render_todo_card_lines(&todo, Theme::dark(), width);
            assert!(!lines.is_empty());
            for line in &lines {
                let rendered = line.to_string();
                let measured = display_width(&rendered);
                assert!(
                    measured <= width,
                    "line width {measured} > {width}: {rendered:?}"
                );
            }
        }
    }

    #[test]
    fn projected_todo_excludes_checkbox_and_header() {
        let todo = TodoView {
            items: vec![TodoItem {
                id: "t".into(),
                content: "完成 emoji 👩‍💻".into(),
                status: TodoStatus::InProgress,
            }],
            auto_continue: AutoContinueState::default(),
        };
        let document = render_todo_card_document(&todo, Theme::dark(), 80);
        let copied = document
            .lines
            .iter()
            .flat_map(|line| line.spans.iter())
            .filter(|span| span.source.is_some())
            .map(|span| span.text.as_str())
            .collect::<String>();
        assert_eq!(copied, "完成 emoji 👩‍💻");
    }
}
