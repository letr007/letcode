use ratatui::{style::Style, text::Line};

use crate::{
    agent::{TodoItem, TodoStatus},
    tui::{measure::wrap_text_to_width, theme::Theme, timeline::TodoView},
};

use super::tool_card::{render_card_line, truncate_display_width};

pub fn render_todo_card_lines(todo: &TodoView, theme: Theme, width: usize) -> Vec<Line<'static>> {
    if width == 0 {
        return Vec::new();
    }

    let mut lines = vec![
        render_blank_line(theme, width),
        render_title_line(theme, width),
        render_blank_line(theme, width),
    ];

    if todo.items.is_empty() {
        push_checklist_item(
            &mut lines,
            &TodoItem {
                id: String::new(),
                content: "No tasks".into(),
                status: TodoStatus::Pending,
            },
            width,
            theme,
        );
        lines.push(render_blank_line(theme, width));
        return lines;
    }

    for item in &todo.items {
        push_checklist_item(&mut lines, item, width, theme);
    }
    lines.push(render_blank_line(theme, width));

    lines
}

fn render_title_line(theme: Theme, width: usize) -> Line<'static> {
    render_card_line(
        &[("# Todos".to_string(), text_style(theme))],
        fill_style(theme),
        theme,
        width,
    )
}

fn render_blank_line(theme: Theme, width: usize) -> Line<'static> {
    render_card_line(&[], fill_style(theme), theme, width)
}

fn item_summary(item: &TodoItem) -> String {
    format!("{} {}", status_marker(&item.status), item.content)
}

fn status_marker(status: &TodoStatus) -> &'static str {
    match status {
        TodoStatus::Pending => "[ ]",
        TodoStatus::InProgress => "[~]",
        TodoStatus::Blocked => "[!]",
        TodoStatus::Completed => "[✓]",
        TodoStatus::Cancelled => "[×]",
    }
}

fn push_checklist_item(
    lines: &mut Vec<Line<'static>>,
    item: &TodoItem,
    width: usize,
    theme: Theme,
) {
    let row = item_summary(item);
    let content_width = width;
    let wrapped = wrap_text_to_width(&row, content_width);

    for chunk in wrapped {
        let segments = vec![(
            truncate_display_width(&chunk, content_width),
            item_style(&item.status, theme),
        )];
        lines.push(render_card_line(&segments, fill_style(theme), theme, width));
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
        TodoStatus::Pending | TodoStatus::InProgress | TodoStatus::Completed => theme.text,
        TodoStatus::Blocked | TodoStatus::Cancelled => theme.error,
    };
    Style::default().fg(color).bg(theme.element_bg)
}

#[cfg(test)]
mod tests {
    use super::render_todo_card_lines;
    use crate::{
        agent::{AutoContinueState, TodoItem, TodoStatus},
        tui::{measure::display_width, theme::Theme, timeline::TodoView},
    };

    #[test]
    fn todo_card_renders_sections_and_progress() {
        let todo = TodoView {
            items: vec![
                TodoItem {
                    id: "t1".into(),
                    content: "Inspect transcript width".into(),
                    status: TodoStatus::InProgress,
                },
                TodoItem {
                    id: "t2".into(),
                    content: "Write todo card tests".into(),
                    status: TodoStatus::Pending,
                },
                TodoItem {
                    id: "t3".into(),
                    content: "Ship transcript integration".into(),
                    status: TodoStatus::Completed,
                },
                TodoItem {
                    id: "t4".into(),
                    content: "Wait on upstream event rename".into(),
                    status: TodoStatus::Blocked,
                },
            ],
            auto_continue: AutoContinueState {
                enabled: true,
                max_continuations: 3,
            },
        };

        let lines = render_todo_card_lines(&todo, Theme::dark(), 56);
        let joined = lines
            .iter()
            .map(|line| line.to_string())
            .collect::<Vec<_>>()
            .join("\n");

        let first = lines.first().expect("top padding").to_string();
        let last = lines.last().expect("bottom padding").to_string();
        assert!(!first.contains("# Todos"));
        assert!(!first.contains("[~]"));
        assert!(!last.contains("# Todos"));
        assert!(!last.contains("[!]"));
        assert!(joined.contains("# Todos"));
        assert!(joined.contains("[~] Inspect transcript width"));
        assert!(joined.contains("[ ] Write todo card tests"));
        assert!(joined.contains("[✓] Ship transcript integration"));
        assert!(joined.contains("[!] Wait on upstream event rename"));
        assert!(!joined.contains("auto on"));
        assert!(!joined.contains("current"));
    }

    #[test]
    fn todo_card_empty_state_is_compact() {
        let todo = TodoView {
            items: Vec::new(),
            auto_continue: AutoContinueState::default(),
        };

        let joined = render_todo_card_lines(&todo, Theme::dark(), 40)
            .into_iter()
            .map(|line| line.to_string())
            .collect::<Vec<_>>()
            .join("\n");

        assert!(joined.contains("# Todos"));
        assert!(joined.contains("[ ] No tasks"));
        assert!(!joined.contains("auto off"));
    }

    #[test]
    fn todo_card_completed_and_cancelled_items_render_clearly() {
        let todo = TodoView {
            items: vec![
                TodoItem {
                    id: "t1".into(),
                    content: "Wrap CJK-safe labels".into(),
                    status: TodoStatus::Completed,
                },
                TodoItem {
                    id: "t2".into(),
                    content: "Drop deprecated branch".into(),
                    status: TodoStatus::Cancelled,
                },
            ],
            auto_continue: AutoContinueState::default(),
        };

        let joined = render_todo_card_lines(&todo, Theme::dark(), 52)
            .into_iter()
            .map(|line| line.to_string())
            .collect::<Vec<_>>()
            .join("\n");

        assert!(joined.contains("# Todos"));
        assert!(joined.contains("[✓] Wrap CJK-safe labels"));
        assert!(joined.contains("[×] Drop deprecated branch"));
    }

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
            auto_continue: AutoContinueState {
                enabled: true,
                max_continuations: 2,
            },
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
}
