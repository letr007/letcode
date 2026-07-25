//! Session coordinator: grows into the home for session-owned command execution.
//!
//! Owns idle commands that do not start turns. The TUI still hosts the async
//! control loop and delegates idle work here.

use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use async_openai::config::Config;

use crate::agent::Agent;
use crate::session::branch_query::{format_branch_listing, load_context_branches};
use crate::session::child_view::{
    current_session_records, project_child_session_view, project_parent_session_view,
    sessions_dir_from_transcript,
};
use crate::session::command::SessionCommand;
use crate::session::event::{ErrorEvent, NoticeEvent};
use crate::session::restore::restored_messages_from_protocol_frames;
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
#[derive(Debug, Default)]
pub struct SessionCoordinator;

impl SessionCoordinator {
    /// Execute an idle session command, emitting [`RunnerEvent`]s for the
    /// frontend bridge. Returns [`IdleDispatch::NotIdle`] when the command is
    /// outside the current coordinator surface.
    ///
    /// `sessions_dir` is required for child/parent view commands; when `None`,
    /// those commands resolve the directory from the live transcript path.
    pub fn dispatch_idle_command<C: Config>(
        command: SessionCommand,
        agent: &mut Agent<C>,
        transcript: &Arc<Mutex<TranscriptRecorder>>,
        event_tx: &mpsc::UnboundedSender<RunnerEvent>,
        sessions_dir: Option<&Path>,
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
                    let _ =
                        event_tx.send(RunnerEvent::Notice(NoticeEvent::info(error.to_string())));
                }
                Ok(IdleDispatch::Handled)
            }
            SessionCommand::ViewParent => {
                Self::emit_view_parent(transcript, event_tx, sessions_dir);
                Ok(IdleDispatch::Handled)
            }
            SessionCommand::ViewChild {
                navigation,
                anchor_child_session_id,
            } => {
                Self::emit_view_child(
                    transcript,
                    event_tx,
                    sessions_dir,
                    navigation,
                    anchor_child_session_id.as_deref(),
                );
                Ok(IdleDispatch::Handled)
            }
            SessionCommand::SubmitPrompt(_)
            | SessionCommand::DelegateSubagent { .. }
            | SessionCommand::Compact
            | SessionCommand::ResumeSession(_)
            | SessionCommand::NewSession
            | SessionCommand::ToggleMcpServer(_)
            | SessionCommand::Interrupt => Ok(IdleDispatch::NotIdle),
        }
    }

    /// Emit parent transcript view events without requiring a mutable agent.
    ///
    /// Safe to call while a turn holds `&mut Agent` (navigation-only path).
    pub fn emit_view_parent(
        transcript: &Arc<Mutex<TranscriptRecorder>>,
        event_tx: &mpsc::UnboundedSender<RunnerEvent>,
        sessions_dir: Option<&Path>,
    ) {
        let dir = match sessions_dir.map(Path::to_path_buf) {
            Some(dir) => Ok(dir),
            None => sessions_dir_from_transcript(transcript),
        };
        match dir.and_then(|dir| project_parent_session_view(transcript, dir)) {
            Ok(projected) => {
                let snapshot = projected.snapshot;
                let _ = event_tx.send(RunnerEvent::SessionResumed {
                    session_id: snapshot.session_id,
                    branch_id: snapshot.branch_id,
                    messages: restored_messages_from_protocol_frames(&snapshot.protocol_frames),
                    records: snapshot.records,
                    evidence_count: projected.evidence_count,
                    model_id: snapshot.latest_model,
                    token_usage: None,
                    runtime_context: projected.runtime_context,
                });
            }
            Err(error) => {
                let _ = event_tx.send(RunnerEvent::Error(ErrorEvent::new(format!(
                    "failed to view parent transcript: {error}"
                ))));
            }
        }
    }

    /// Emit child transcript view events without requiring a mutable agent.
    ///
    /// Safe to call while a turn holds `&mut Agent` (navigation-only path).
    /// Returns the selected child session id when a child view was emitted.
    pub fn emit_view_child(
        transcript: &Arc<Mutex<TranscriptRecorder>>,
        event_tx: &mpsc::UnboundedSender<RunnerEvent>,
        sessions_dir: Option<&Path>,
        navigation: crate::command::ChildNavigation,
        anchor_child_session_id: Option<&str>,
    ) -> Option<String> {
        let dir = match sessions_dir.map(Path::to_path_buf) {
            Some(dir) => Ok(dir),
            None => sessions_dir_from_transcript(transcript),
        };
        match dir.and_then(|dir| {
            let (parent_session_id, parent_records) = current_session_records(transcript)?;
            project_child_session_view(
                dir,
                parent_session_id,
                &parent_records,
                navigation,
                anchor_child_session_id,
            )
        }) {
            Ok(None) => {
                let _ = event_tx.send(RunnerEvent::Notice(NoticeEvent::info(
                    "No child subagent transcripts for this session",
                )));
                None
            }
            Ok(Some(view)) => {
                let child_session_id = view.child_session_id.clone();
                let _ = event_tx.send(RunnerEvent::ChildSessionViewed {
                    parent_session_id: view.parent_session_id,
                    child_session_id: view.child_session_id,
                    agent_name: view.agent_name,
                    index: view.index,
                    total: view.total,
                    pool_ordinal: view.pool_ordinal,
                    records: view.records,
                    runtime_context: view.runtime_context,
                });
                Some(child_session_id)
            }
            Err(error) => {
                let _ = event_tx.send(RunnerEvent::Error(ErrorEvent::new(format!(
                    "failed to view child transcript: {error}"
                ))));
                None
            }
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
                | SessionCommand::ViewParent
                | SessionCommand::ViewChild { .. }
        )
    }

    /// Exhaustive ownership table for migration tracking.
    pub fn ownership(command: &SessionCommand) -> CommandOwnership {
        match command {
            SessionCommand::ShowBranchTree
            | SessionCommand::ListBranches
            | SessionCommand::SetPermissionMode(_)
            | SessionCommand::SetModel(_)
            | SessionCommand::SetReasoningEffort(_)
            | SessionCommand::ViewParent
            | SessionCommand::ViewChild { .. } => CommandOwnership::IdleCoordinator,
            SessionCommand::SubmitPrompt(_)
            | SessionCommand::DelegateSubagent { .. }
            | SessionCommand::Compact
            | SessionCommand::ResumeSession(_)
            | SessionCommand::NewSession
            | SessionCommand::ToggleMcpServer(_)
            | SessionCommand::Interrupt => CommandOwnership::FrontendHosted,
        }
    }
}

/// Where a [`SessionCommand`] is executed today.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandOwnership {
    /// Fully handled by [`SessionCoordinator::dispatch_idle_command`].
    IdleCoordinator,
    /// Still executed by the TUI runner loop and/or CLI-specific paths.
    FrontendHosted,
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
    fn ownership_table_is_exhaustive_and_matches_idle_classifier() {
        let samples = [
            SessionCommand::SubmitPrompt(crate::user_content::UserMessageSubmission::new(
                "id",
                crate::user_content::UserMessageContent::from("hi"),
            )),
            SessionCommand::DelegateSubagent {
                agent_name: "explorer".into(),
                task: "x".into(),
            },
            SessionCommand::Compact,
            SessionCommand::ShowBranchTree,
            SessionCommand::ListBranches,
            SessionCommand::ViewChild {
                navigation: crate::command::ChildNavigation::Next,
                anchor_child_session_id: None,
            },
            SessionCommand::ViewParent,
            SessionCommand::SetPermissionMode(PermissionMode::Safe),
            SessionCommand::SetModel("m".into()),
            SessionCommand::SetReasoningEffort(ModelReasoningEffort::Low),
            SessionCommand::ResumeSession("abc".into()),
            SessionCommand::NewSession,
            SessionCommand::ToggleMcpServer("s".into()),
            SessionCommand::Interrupt,
        ];
        for command in samples {
            let owned = SessionCoordinator::ownership(&command);
            assert_eq!(
                owned == CommandOwnership::IdleCoordinator,
                SessionCoordinator::is_idle_command(&command),
                "ownership mismatch for {command:?}"
            );
        }
    }

    #[test]
    fn idle_classifier_and_dispatch_cover_settings_and_branches() {
        assert!(SessionCoordinator::is_idle_command(
            &SessionCommand::ListBranches
        ));
        assert!(SessionCoordinator::is_idle_command(
            &SessionCommand::ViewParent
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
            None,
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
            None,
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
            None,
        )
        .expect("not idle");
        assert_eq!(outcome, IdleDispatch::NotIdle);
    }

    #[test]
    fn emit_view_child_without_children_sends_notice() {
        let transcript = temp_transcript();
        let (tx, mut rx) = mpsc::unbounded_channel();

        let selected = SessionCoordinator::emit_view_child(
            &transcript,
            &tx,
            None,
            crate::command::ChildNavigation::First,
            None,
        );
        assert_eq!(selected, None);
        match rx.try_recv().expect("notice") {
            RunnerEvent::Notice(notice) => {
                assert!(notice.message.contains("No child subagent transcripts"));
            }
            other => panic!("expected notice, got {other:?}"),
        }
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn emit_view_parent_sends_session_resumed() {
        let transcript = temp_transcript();
        let (tx, mut rx) = mpsc::unbounded_channel();

        SessionCoordinator::emit_view_parent(&transcript, &tx, None);
        match rx.try_recv().expect("session resumed") {
            RunnerEvent::SessionResumed { session_id, .. } => {
                assert!(!session_id.is_empty());
            }
            other => panic!("expected SessionResumed, got {other:?}"),
        }
        assert!(rx.try_recv().is_err());
    }
}
