use super::*;

fn todo(id: &str, content: &str, status: TodoStatus) -> TodoItem {
    TodoItem {
        id: id.into(),
        content: content.into(),
        status,
    }
}

#[test]
fn compaction_reminder_lists_open_todos_only() {
    let state = WorkflowState {
        todos: vec![
            todo("open-assembly", "wire parser", TodoStatus::InProgress),
            todo("waiting-upstream", "await review", TodoStatus::Blocked),
            todo("finished-item", "scaffold module", TodoStatus::Completed),
            todo("dropped-item", "abandoned spike", TodoStatus::Cancelled),
        ],
        auto_continue: AutoContinueState::default(),
    };

    let reminder = state.render_compaction_reminder().expect("reminder");
    assert!(
        reminder.contains("- open-assembly [in_progress]: wire parser"),
        "{reminder}"
    );
    assert!(
        reminder.contains("- waiting-upstream [blocked]: await review"),
        "{reminder}"
    );
    assert!(!reminder.contains("finished-item"), "{reminder}");
    assert!(!reminder.contains("abandoned spike"), "{reminder}");
}

#[test]
fn compaction_reminder_reports_auto_continue() {
    let state = WorkflowState {
        todos: Vec::new(),
        auto_continue: AutoContinueState { enabled: true },
    };

    let reminder = state.render_compaction_reminder().expect("reminder");
    assert!(reminder.contains("Auto-continue is enabled."), "{reminder}");
}

#[test]
fn compaction_reminder_is_absent_without_active_workflow_state() {
    assert!(
        WorkflowState::default()
            .render_compaction_reminder()
            .is_none()
    );

    let finished = WorkflowState {
        todos: vec![todo(
            "finished-item",
            "scaffold module",
            TodoStatus::Completed,
        )],
        auto_continue: AutoContinueState::default(),
    };
    assert!(finished.render_compaction_reminder().is_none());
}
