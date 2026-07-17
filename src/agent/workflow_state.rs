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
    pub(super) fn is_unfinished(&self) -> bool {
        matches!(self, Self::Pending | Self::InProgress)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TodoItem {
    pub id: String,
    pub content: String,
    pub status: TodoStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AutoContinueState {
    pub enabled: bool,
    pub max_continuations: usize,
}

impl AutoContinueState {
    const DEFAULT_MAX_CONTINUATIONS: usize = 3;
    pub(super) const ABSOLUTE_MAX_CONTINUATIONS: usize = 16;
}

impl Default for AutoContinueState {
    fn default() -> Self {
        Self {
            enabled: false,
            max_continuations: Self::DEFAULT_MAX_CONTINUATIONS,
        }
    }
}
