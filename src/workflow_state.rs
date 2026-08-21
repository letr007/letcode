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
}

fn is_default_auto_continue(state: &AutoContinueState) -> bool {
    !state.enabled
}
