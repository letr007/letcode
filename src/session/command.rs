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
    DelegateSubagent {
        agent_name: String,
        task: String,
    },
    Compact,
    ShowHistoryTree,
    Undo,
    Redo,
    NavigateHistory {
        target_entry_id: String,
    },
    ViewChild {
        navigation: ChildNavigation,
        /// Optional anchor for next/prev navigation within the child list.
        anchor_child_session_id: Option<String>,
    },
    ViewParent,
    SetPermissionMode(PermissionMode),
    SetModel(String),
    SetExpertModel {
        agent_name: String,
        model_id: String,
    },
    ToggleFastMode,
    SetReasoningEffort(ModelReasoningEffort),
    ResumeSession(String),
    NewSession,
    ToggleMcpServer(String),
    Interrupt,
}

impl SessionCommand {
    /// Map a shared slash/line [`crate::command::CommandIntent`] into a session
    /// command when the intent is backend-owned. Presentation-only intents
    /// (`Help`, browse UIs, local display toggles) return `None`.
    pub fn from_command_intent(intent: crate::command::CommandIntent) -> Option<Self> {
        use crate::command::CommandIntent;
        use crate::user_content::UserMessageSubmission;

        match intent {
            CommandIntent::Prompt(text) => {
                // Stable FE→BE shape: id + content (text/attachments live on content).
                let id = format!(
                    "cmd-prompt-{}",
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_nanos())
                        .unwrap_or(0)
                );
                Some(Self::SubmitPrompt(UserMessageSubmission::new(
                    id,
                    crate::user_content::UserMessageContent::from(text),
                )))
            }
            CommandIntent::Delegate { agent_name, task } => {
                Some(Self::DelegateSubagent { agent_name, task })
            }
            CommandIntent::Compact => Some(Self::Compact),
            CommandIntent::Tree => Some(Self::ShowHistoryTree),
            CommandIntent::Undo => Some(Self::Undo),
            CommandIntent::Redo => Some(Self::Redo),
            CommandIntent::Child(nav) => Some(Self::ViewChild {
                navigation: nav,
                anchor_child_session_id: None,
            }),
            CommandIntent::Parent => Some(Self::ViewParent),
            CommandIntent::PermissionSet(mode) => Some(Self::SetPermissionMode(mode)),
            CommandIntent::ModelSet(model) => Some(Self::SetModel(model)),
            CommandIntent::FastToggle => Some(Self::ToggleFastMode),
            CommandIntent::ReasoningSet(effort) => Some(Self::SetReasoningEffort(effort)),
            CommandIntent::Resume(id) => Some(Self::ResumeSession(id)),
            CommandIntent::NewSession => Some(Self::NewSession),
            CommandIntent::Help
            | CommandIntent::Exit
            | CommandIntent::PermissionShow
            | CommandIntent::ModelShow
            | CommandIntent::AgentsShow
            | CommandIntent::ReasoningShow
            | CommandIntent::ToolOutputSet(_)
            | CommandIntent::TranscriptScrollbarSet(_)
            | CommandIntent::Theme(_)
            | CommandIntent::ResumeShow
            | CommandIntent::ContextBrowse
            | CommandIntent::McpBrowse
            | CommandIntent::SkillBrowse => None,
        }
    }
}
