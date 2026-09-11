use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum TodoStatus {
    #[serde(rename = "pending")]
    Pending,
    #[serde(rename = "in_progress")]
    InProgress,
    #[serde(rename = "blocked")]
    Blocked,
    #[serde(rename = "completed")]
    Completed,
    #[serde(rename = "cancelled")]
    Cancelled,
}

impl TodoStatus {
    pub(crate) fn is_unfinished(&self) -> bool {
        matches!(self, Self::Pending | Self::InProgress)
    }

    fn label(&self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::InProgress => "in_progress",
            Self::Blocked => "blocked",
            Self::Completed => "completed",
            Self::Cancelled => "cancelled",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TodoItem {
    pub id: String,
    pub content: String,
    pub status: TodoStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct AutoContinueState {
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub(crate) struct WorkflowState {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub todos: Vec<TodoItem>,
    #[serde(default, skip_serializing_if = "is_default_auto_continue")]
    pub auto_continue: AutoContinueState,
}

impl WorkflowState {
    pub(crate) fn is_empty(&self) -> bool {
        self.todos.is_empty() && !self.auto_continue.enabled
    }

    /// Model-facing note appended after a context compaction. The todo list and
    /// the auto-continue flag live in the runtime snapshot, not in transcript
    /// text, so a summarized history would otherwise drop them.
    pub(crate) fn render_compaction_reminder(&self) -> Option<String> {
        let open = self
            .todos
            .iter()
            .filter(|todo| !matches!(todo.status, TodoStatus::Completed | TodoStatus::Cancelled))
            .map(|todo| format!("- {} [{}]: {}", todo.id, todo.status.label(), todo.content))
            .collect::<Vec<_>>();
        if open.is_empty() && !self.auto_continue.enabled {
            return None;
        }
        let mut reminder = String::from(
            "[workflow-state] Earlier conversation was summarized into the history above. This session state is still active:\n",
        );
        if !open.is_empty() {
            reminder.push_str("Open todos:\n");
            reminder.push_str(&open.join("\n"));
            reminder.push('\n');
        }
        if self.auto_continue.enabled {
            reminder.push_str("Auto-continue is enabled.\n");
        }
        reminder.push_str("Continue the current task; do not repeat work that is already done.");
        Some(reminder)
    }
}

fn is_default_auto_continue(state: &AutoContinueState) -> bool {
    !state.enabled
}

#[cfg(test)]
mod tests {
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
}
