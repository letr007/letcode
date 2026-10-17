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
