//! Inbound commands from any frontend into the session backend.
//!
//! Keep this enum free of presentation types (no ratatui, no layout). CLI, TUI,
//! and future GUI should all map user intent onto these variants.

use crate::command::ChildNavigation;
use crate::permission::PermissionMode;
use crate::request_builder::ModelReasoningEffort;
use crate::user_content::UserMessageSubmission;

/// A frontend request to mutate or query the active session.
///
/// This is the stable FE→BE command surface. TUI historically called the same
/// shape `RuntimeCommand`; that name remains as a compatibility alias.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionCommand {
    SubmitPrompt(UserMessageSubmission),
    DelegateSubagent { agent_name: String, task: String },
    Compact,
    ShowBranchTree,
    ListBranches,
    ViewChild(ChildNavigation),
    ViewParent,
    SetPermissionMode(PermissionMode),
    SetModel(String),
    SetReasoningEffort(ModelReasoningEffort),
    ResumeSession(String),
    NewSession,
    ToggleMcpServer(String),
    Interrupt,
}
