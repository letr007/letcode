//! Session coordinator: grows into the home for session-owned command execution.
//!
//! Phase H owns **idle** commands that do not start turns or need TUI-private
//! transport (child anchors, MCP catalog UI, resume/new-session orchestration).
//! The TUI still hosts the async control loop; it delegates idle work here.

use std::sync::{Arc, Mutex};

use anyhow::Result;
use async_openai::config::Config;

use crate::agent::Agent;
use crate::session::branch_query::{format_branch_listing, load_context_branches};
use crate::session::command::SessionCommand;
use crate::session::event::{ErrorEvent, NoticeEvent};
use crate::session::runner::RunnerEvent;
use crate::session::settings::{apply_model, apply_permission_mode, apply_reasoning_effort};
use crate::transcript::TranscriptRecorder;
use tokio::sync::mpsc;

/// Outcome of attempting to run a [`SessionCommand`] as idle session work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdleDispatch {
    /// Command fully handled (events emitted / agent mutated as needed).
    Handled,
    /// Command requires turn execution or frontend-private orchestration.
    NotIdle,
}

/// Session-owned coordinator entry point for multi-frontend backends.
///
/// Today this is a namespace for idle dispatch; later it absorbs more of the
/// TUI-hosted runner control loop and AgentRunner turn ownership.
#[derive(Debug, Default)]
pub struct SessionCoordinator;

impl SessionCoordinator {
    /// Execute an idle session command, emitting [`RunnerEvent`]s for the
    /// frontend bridge. Returns [`IdleDispatch::NotIdle`] when the command is
    /// outside the current coordinator surface.
    pub fn dispatch_idle_command<C: Config>(
        command: SessionCommand,
        agent: &mut Agent<C>,
        transcript: &Arc<Mutex<TranscriptRecorder>>,
        event_tx: &mpsc::UnboundedSender<RunnerEvent>,
    ) -> Result<IdleDispatch> {
        match command {
            SessionCommand::ShowBranchTree => {
                match load_context_branches(transcript) {
                    Ok(branches) => {
                        let _ = event_tx.send(RunnerEvent::ContextBranchesLoaded { branches });
                    }
                    Err(error) => {
                        let _ = event_tx.send(RunnerEvent::Error(ErrorEvent::new(format!(
                            "failed to load context tree: {error}"
                        ))));
                    }
                }
                Ok(IdleDispatch::Handled)
            }
            SessionCommand::ListBranches => {
                match load_context_branches(transcript) {
                    Ok(branches) => {
                        let message = format_branch_listing(&branches);
                        let _ = event_tx.send(RunnerEvent::Notice(NoticeEvent::info(message)));
                    }
                    Err(error) => {
                        let _ = event_tx.send(RunnerEvent::Error(ErrorEvent::new(format!(
                            "failed to list context branches: {error}"
                        ))));
                    }
                }
                Ok(IdleDispatch::Handled)
            }
            SessionCommand::SetPermissionMode(mode) => {
                if let Err(error) = apply_permission_mode(agent, transcript, mode) {
                    let _ = event_tx.send(RunnerEvent::Error(ErrorEvent::new(format!(
                        "failed to set permission mode: {error}"
                    ))));
                }
                Ok(IdleDispatch::Handled)
            }
            SessionCommand::SetModel(model) => {
                if let Err(error) = apply_model(agent, transcript, model) {
                    let _ = event_tx.send(RunnerEvent::Error(ErrorEvent::new(format!(
                        "failed to set model: {error}"
                    ))));
                }
                Ok(IdleDispatch::Handled)
            }
            SessionCommand::SetReasoningEffort(effort) => {
                if let Err(error) = apply_reasoning_effort(agent, effort) {
                    let _ = event_tx
                        .send(RunnerEvent::Notice(NoticeEvent::info(error.to_string())));
                }
                Ok(IdleDispatch::Handled)
            }
            SessionCommand::SubmitPrompt(_)
            | SessionCommand::DelegateSubagent { .. }
            | SessionCommand::Compact
            | SessionCommand::ViewChild(_)
            | SessionCommand::ViewParent
            | SessionCommand::ResumeSession(_)
            | SessionCommand::NewSession
            | SessionCommand::ToggleMcpServer(_)
            | SessionCommand::Interrupt => Ok(IdleDispatch::NotIdle),
        }
    }

    /// Whether this command is currently handled as idle work by the coordinator.
    pub fn is_idle_command(command: &SessionCommand) -> bool {
        matches!(
            command,
            SessionCommand::ShowBranchTree
                | SessionCommand::ListBranches
                | SessionCommand::SetPermissionMode(_)
                | SessionCommand::SetModel(_)
                | SessionCommand::SetReasoningEffort(_)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_openai::{Client, config::OpenAIConfig};
    use crate::permission::PermissionMode;
    use crate::request_builder::ModelReasoningEffort;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_transcript() -> Arc<Mutex<TranscriptRecorder>> {
        let base_dir = std::env::temp_dir().join(format!(
            "letcode-session-coordinator-test-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("time ok")
                .as_nanos()
        ));
        Arc::new(Mutex::new(
            TranscriptRecorder::create(&base_dir).expect("create transcript"),
        ))
    }

    fn test_agent() -> Agent<OpenAIConfig> {
        let config = OpenAIConfig::new()
            .with_api_base("http://127.0.0.1:9/v1")
            .with_api_key("test-key");
        Agent::new(Client::with_config(config), "gpt-5.5", 1, 1)
    }

    #[test]
    fn idle_classifier_and_dispatch_cover_settings_and_branches() {
        assert!(SessionCoordinator::is_idle_command(
            &SessionCommand::ListBranches
        ));
        assert!(!SessionCoordinator::is_idle_command(
            &SessionCommand::Compact
        ));

        let transcript = temp_transcript();
        let mut agent = test_agent();
        let (tx, mut rx) = mpsc::unbounded_channel();

        let outcome = SessionCoordinator::dispatch_idle_command(
            SessionCommand::SetPermissionMode(PermissionMode::Safe),
            &mut agent,
            &transcript,
            &tx,
        )
        .expect("dispatch");
        assert_eq!(outcome, IdleDispatch::Handled);
        assert_eq!(agent.permission_mode(), PermissionMode::Safe);
        assert!(rx.try_recv().is_err(), "success path emits no events");

        let outcome = SessionCoordinator::dispatch_idle_command(
            SessionCommand::ListBranches,
            &mut agent,
            &transcript,
            &tx,
        )
        .expect("list");
        assert_eq!(outcome, IdleDispatch::Handled);
        match rx.try_recv().expect("notice") {
            RunnerEvent::Notice(notice) => assert!(!notice.message.is_empty()),
            other => panic!("expected notice, got {other:?}"),
        }

        let outcome = SessionCoordinator::dispatch_idle_command(
            SessionCommand::Compact,
            &mut agent,
            &transcript,
            &tx,
        )
        .expect("not idle");
        assert_eq!(outcome, IdleDispatch::NotIdle);

        let _ = ModelReasoningEffort::None;
    }
}
