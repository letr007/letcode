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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ActiveTurnCommandDisposition {
    QueuePrompt,
    Immediate,
    Defer,
    Reject,
    Interrupt,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionCommand {
    SubmitPrompt(UserMessageSubmission),
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
    SetExpertAllowedModels {
        agent_name: String,
        model_ids: Vec<String>,
    },
    ToggleFastMode,
    SetReasoningEffort(ModelReasoningEffort),
    SetFakeClient(Option<crate::fake::FakeClient>),
    ResumeSession(String),
    NewSession,
    ToggleMcpServer(String),
    Interrupt,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CommandSchedule {
    pub(crate) idle: bool,
    pub(crate) active_turn: ActiveTurnCommandDisposition,
}

impl SessionCommand {
    pub(crate) fn schedule(&self) -> CommandSchedule {
        let (idle, active_turn) = match self {
            Self::SubmitPrompt(_) => (false, ActiveTurnCommandDisposition::QueuePrompt),
            Self::ViewChild { .. } | Self::ViewParent => {
                (true, ActiveTurnCommandDisposition::Immediate)
            }
            Self::SetPermissionMode(_)
            | Self::ToggleFastMode
            | Self::SetReasoningEffort(_)
            | Self::SetFakeClient(_) => (true, ActiveTurnCommandDisposition::Defer),
            Self::SetModel(_) | Self::SetExpertAllowedModels { .. } | Self::ToggleMcpServer(_) => {
                (false, ActiveTurnCommandDisposition::Defer)
            }
            Self::Interrupt => (false, ActiveTurnCommandDisposition::Interrupt),
            Self::Compact | Self::ResumeSession(_) | Self::NewSession => {
                (false, ActiveTurnCommandDisposition::Reject)
            }
            Self::ShowHistoryTree | Self::Undo | Self::Redo | Self::NavigateHistory { .. } => {
                (true, ActiveTurnCommandDisposition::Reject)
            }
        };
        CommandSchedule { idle, active_turn }
    }

    pub(crate) fn active_turn_disposition(&self) -> ActiveTurnCommandDisposition {
        self.schedule().active_turn
    }

    /// Map a shared slash/line [`crate::command::CommandIntent`] into a session
    /// command when the intent is backend-owned. Presentation-only intents
    /// (`Help`, browse UIs, local display toggles) return `None`.
    pub fn from_command_intent(intent: crate::command::CommandIntent) -> Option<Self> {
        use crate::command::CommandIntent;
        use crate::user_content::UserMessageSubmission;

        match intent {
            CommandIntent::Language(_) => None,
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
            CommandIntent::Compact => Some(Self::Compact),
            CommandIntent::Tree => Some(Self::ShowHistoryTree),
            CommandIntent::Undo => Some(Self::Undo),
            CommandIntent::Redo => Some(Self::Redo),
            CommandIntent::Child(nav) => Some(Self::ViewChild {
                navigation: nav,
                anchor_child_session_id: None,
            }),
            CommandIntent::PermissionSet(mode) => Some(Self::SetPermissionMode(mode)),
            CommandIntent::ModelSet(model) => Some(Self::SetModel(model)),
            CommandIntent::FastToggle => Some(Self::ToggleFastMode),
            CommandIntent::ReasoningSet(effort) => Some(Self::SetReasoningEffort(effort)),
            CommandIntent::Fake(crate::command::FakeCommand::Set(client)) => {
                Some(Self::SetFakeClient(client))
            }
            CommandIntent::Resume(id) => Some(Self::ResumeSession(id)),
            CommandIntent::NewSession => Some(Self::NewSession),
            CommandIntent::Help
            | CommandIntent::Exit
            | CommandIntent::PermissionShow
            | CommandIntent::ModelShow
            | CommandIntent::AgentsShow
            | CommandIntent::ReasoningShow
            | CommandIntent::ThoughtsShow
            | CommandIntent::ThoughtsSet(_)
            | CommandIntent::ToolsShow
            | CommandIntent::ToolsSet(_)
            | CommandIntent::TranscriptScrollbarSet(_)
            | CommandIntent::PanelSet(_)
            | CommandIntent::Theme(_)
            | CommandIntent::Fake(crate::command::FakeCommand::Show)
            | CommandIntent::ResumeShow
            | CommandIntent::ContextBrowse
            | CommandIntent::McpBrowse
            | CommandIntent::SkillBrowse
            | CommandIntent::ConfigShow => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fake_set_maps_to_backend_and_show_stays_local() {
        for client in [
            crate::fake::FakeClient::Auto,
            crate::fake::FakeClient::Codex,
            crate::fake::FakeClient::Anthropic,
        ] {
            assert_eq!(
                SessionCommand::from_command_intent(crate::command::CommandIntent::Fake(
                    crate::command::FakeCommand::Set(Some(client))
                )),
                Some(SessionCommand::SetFakeClient(Some(client)))
            );
        }
        assert_eq!(
            SessionCommand::from_command_intent(crate::command::CommandIntent::Fake(
                crate::command::FakeCommand::Show
            )),
            None
        );
        assert_eq!(
            SessionCommand::SetFakeClient(Some(crate::fake::FakeClient::Codex))
                .active_turn_disposition(),
            crate::session::ActiveTurnCommandDisposition::Defer
        );
    }
}
