//! Session coordinator: grows into the home for session-owned command execution.
//!
//! Owns idle commands that do not start turns. The TUI still hosts the async
//! control loop and delegates idle work here.

use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use async_openai::config::Config;

use crate::agent::Agent;
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
            SessionCommand::ShowHistoryTree => {
                let entries = (|| {
                    let recorder = transcript
                        .lock()
                        .map_err(|_| anyhow::anyhow!("transcript recorder poisoned"))?;
                    let records = crate::transcript::read_records(recorder.path())?;
                    Ok::<_, anyhow::Error>(
                        crate::transcript::transcript_projection::project_session_history_tree(
                            &records,
                        ),
                    )
                })();
                match entries {
                    Ok(entries) => {
                        let _ = event_tx.send(RunnerEvent::SessionHistoryLoaded { entries });
                    }
                    Err(error) => {
                        let _ = event_tx.send(RunnerEvent::Error(ErrorEvent::new(format!(
                            "failed to load session history: {error}"
                        ))));
                    }
                }
                Ok(IdleDispatch::Handled)
            }
            SessionCommand::NavigateHistory { target_entry_id } => {
                match Self::entry_sequence(&target_entry_id) {
                    Ok(target_sequence) => Self::navigate_history(
                        agent,
                        transcript,
                        event_tx,
                        target_sequence,
                        crate::transcript::HistoryNavigationOperation::Navigate,
                        Vec::new(),
                    ),
                    Err(error) => {
                        let _ =
                            event_tx.send(RunnerEvent::Error(ErrorEvent::new(error.to_string())));
                    }
                }
                Ok(IdleDispatch::Handled)
            }
            SessionCommand::Undo => {
                Self::navigate_undo(agent, transcript, event_tx);
                Ok(IdleDispatch::Handled)
            }
            SessionCommand::Redo => {
                Self::navigate_redo(agent, transcript, event_tx);
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

    fn navigate_undo<C: Config>(
        agent: &mut Agent<C>,
        transcript: &Arc<Mutex<TranscriptRecorder>>,
        event_tx: &mpsc::UnboundedSender<RunnerEvent>,
    ) {
        let result = (|| -> Result<(u64, Vec<u64>)> {
            let recorder = transcript
                .lock()
                .map_err(|_| anyhow::anyhow!("transcript recorder poisoned"))?;
            let records = crate::transcript::read_records(recorder.path())?;
            let entries =
                crate::transcript::transcript_projection::project_session_history_tree(&records);
            let navigation =
                crate::transcript::transcript_projection::history_navigation_state(&records);
            let current = match navigation.as_ref() {
                Some(navigation) => navigation.target_sequence,
                None => {
                    let snapshot =
                        crate::transcript::transcript_projection::build_session_context_snapshot(
                            recorder.session_id().to_string(),
                            records.clone(),
                            crate::transcript::transcript_projection::SessionContextCursor {
                                branch_id: recorder.current_context_branch_id().map(str::to_string),
                                leaf_sequence: None,
                            },
                        )?;
                    let visible_sequences = snapshot
                        .records
                        .iter()
                        .map(|record| record.sequence)
                        .collect::<std::collections::BTreeSet<_>>();
                    entries
                        .iter()
                        .rev()
                        .find(|entry| visible_sequences.contains(&entry.sequence))
                        .map(|entry| entry.sequence)
                        .ok_or_else(|| anyhow::anyhow!("no session history entry to undo"))?
                }
            };
            if current == 0 {
                anyhow::bail!("already at the start of session history");
            }
            let mut turn_root = entries
                .iter()
                .find(|entry| entry.sequence == current)
                .ok_or_else(|| anyhow::anyhow!("current history target is unavailable"))?;
            while turn_root.kind
                != crate::transcript::transcript_projection::SessionHistoryEntryKind::User
            {
                let parent_id = turn_root
                    .parent_id
                    .as_deref()
                    .ok_or_else(|| anyhow::anyhow!("history parent is unavailable"))?;
                turn_root = entries
                    .iter()
                    .find(|entry| entry.id == parent_id)
                    .ok_or_else(|| anyhow::anyhow!("history parent is unavailable"))?;
            }
            let target = turn_root
                .parent_id
                .as_deref()
                .map(Self::entry_sequence)
                .transpose()?
                .unwrap_or(0);
            let mut redo_stack = navigation.map_or_else(Vec::new, |state| state.redo_stack.clone());
            redo_stack.push(current);
            Ok((target, redo_stack))
        })();
        match result {
            Ok((target, redo_stack)) => Self::navigate_history(
                agent,
                transcript,
                event_tx,
                target,
                crate::transcript::HistoryNavigationOperation::Undo,
                redo_stack,
            ),
            Err(error) => {
                let _ = event_tx.send(RunnerEvent::Notice(NoticeEvent::info(error.to_string())));
            }
        }
    }

    fn navigate_redo<C: Config>(
        agent: &mut Agent<C>,
        transcript: &Arc<Mutex<TranscriptRecorder>>,
        event_tx: &mpsc::UnboundedSender<RunnerEvent>,
    ) {
        let result = (|| -> Result<(u64, Vec<u64>)> {
            let recorder = transcript
                .lock()
                .map_err(|_| anyhow::anyhow!("transcript recorder poisoned"))?;
            let records = crate::transcript::read_records(recorder.path())?;
            let mut state =
                crate::transcript::transcript_projection::history_navigation_state(&records)
                    .ok_or_else(|| anyhow::anyhow!("no history entry available to redo"))?;
            let target = state
                .redo_stack
                .pop()
                .ok_or_else(|| anyhow::anyhow!("no history entry available to redo"))?;
            // Ensure a corrupt navigation event cannot make redo guess a sibling.
            if target != 0
                && !crate::transcript::transcript_projection::project_session_history_tree(&records)
                    .iter()
                    .any(|entry| entry.sequence == target)
            {
                anyhow::bail!("redo target is unavailable");
            }
            Ok((target, state.redo_stack))
        })();
        match result {
            Ok((target, redo_stack)) => Self::navigate_history(
                agent,
                transcript,
                event_tx,
                target,
                crate::transcript::HistoryNavigationOperation::Redo,
                redo_stack,
            ),
            Err(error) => {
                let _ = event_tx.send(RunnerEvent::Notice(NoticeEvent::info(error.to_string())));
            }
        }
    }

    fn entry_sequence(entry_id: &str) -> Result<u64> {
        entry_id
            .strip_prefix("entry-")
            .ok_or_else(|| anyhow::anyhow!("invalid history entry id '{entry_id}'"))?
            .parse()
            .map_err(Into::into)
    }

    fn navigate_history<C: Config>(
        agent: &mut Agent<C>,
        transcript: &Arc<Mutex<TranscriptRecorder>>,
        event_tx: &mpsc::UnboundedSender<RunnerEvent>,
        target_sequence: u64,
        operation: crate::transcript::HistoryNavigationOperation,
        redo_stack: Vec<u64>,
    ) {
        let result = (|| -> Result<_> {
            let mut recorder = transcript
                .lock()
                .map_err(|_| anyhow::anyhow!("transcript recorder poisoned"))?;
            let records = crate::transcript::read_records(recorder.path())?;
            let entries =
                crate::transcript::transcript_projection::project_session_history_tree(&records);
            let parent_branch_id = if target_sequence == 0 {
                crate::transcript::ROOT_CONTEXT_BRANCH_ID.into()
            } else {
                entries
                    .iter()
                    .find(|entry| entry.sequence == target_sequence)
                    .map(|entry| entry.branch_id.clone())
                    .ok_or_else(|| {
                        anyhow::anyhow!("history target {target_sequence} is unavailable")
                    })?
            };
            let branch_sequence = records
                .iter()
                .map(|record| record.sequence)
                .max()
                .unwrap_or(0)
                .checked_add(1)
                .ok_or_else(|| anyhow::anyhow!("transcript sequence overflow"))?;
            let branch_id = format!("history-{branch_sequence}");
            let mut candidate_records = records.clone();
            for (offset, event) in [
                crate::transcript::TranscriptEvent::ContextBranchCreated {
                    branch_id: branch_id.clone(),
                    parent_branch_id: parent_branch_id.clone(),
                    base_sequence: target_sequence,
                    label: None,
                },
                crate::transcript::TranscriptEvent::ContextCheckout {
                    branch_id: branch_id.clone(),
                    leaf_sequence: target_sequence,
                },
                crate::transcript::TranscriptEvent::HistoryNavigation {
                    operation,
                    target_sequence,
                    redo_stack: redo_stack.clone(),
                    redo_target_sequence: None,
                },
            ]
            .into_iter()
            .enumerate()
            {
                candidate_records.push(crate::transcript::TranscriptRecord {
                    session_id: recorder.session_id().to_string(),
                    sequence: branch_sequence + offset as u64,
                    timestamp_ms: 0,
                    context_branch_id: None,
                    event,
                });
            }
            let sessions_dir = recorder
                .path()
                .parent()
                .ok_or_else(|| anyhow::anyhow!("transcript path has no parent directory"))?;
            let resolved =
                crate::transcript::transcript_projection::project_runtime_restore_snapshot(
                    recorder.session_id().to_string(),
                    candidate_records.clone(),
                    crate::transcript::transcript_projection::SessionContextCursor {
                        branch_id: Some(branch_id.clone()),
                        leaf_sequence: None,
                    },
                    &[],
                )?;
            let children =
                crate::transcript::list_child_sessions_for_parent(sessions_dir, &resolved.records);
            let snapshot =
                crate::transcript::transcript_projection::project_runtime_restore_snapshot(
                    recorder.session_id().to_string(),
                    candidate_records,
                    crate::transcript::transcript_projection::SessionContextCursor {
                        branch_id: Some(branch_id.clone()),
                        leaf_sequence: None,
                    },
                    &children,
                )?;
            let runtime_context =
                crate::runtime_context::RuntimeActiveContext::try_from(&snapshot.snapshot)?;
            let (protocol_frames, runtime_snapshot) = agent.validate_runtime_snapshot_restore(
                snapshot.protocol_frames.clone(),
                snapshot.snapshot.clone(),
            )?;
            recorder.record_history_navigation_transaction(
                branch_id.clone(),
                parent_branch_id,
                target_sequence,
                operation,
                redo_stack,
            )?;
            crate::transcript::sync_recorder_branch(&mut recorder, &branch_id);
            Ok((snapshot, runtime_context, protocol_frames, runtime_snapshot))
        })();
        match result {
            Ok((snapshot, runtime_context, protocol_frames, runtime_snapshot)) => {
                agent.install_validated_runtime_snapshot(protocol_frames, runtime_snapshot);
                if let Some(model) = &snapshot.latest_model {
                    agent.set_model(model.clone());
                }
                agent.restore_turn_sequence(snapshot.max_turn_id);
                let _ = event_tx.send(RunnerEvent::SessionResumed {
                    session_id: snapshot.session_id,
                    branch_id: snapshot.branch_id,
                    messages: restored_messages_from_protocol_frames(&snapshot.protocol_frames),
                    records: snapshot.records,
                    evidence_count: 0,
                    model_id: snapshot.latest_model,
                    token_usage: None,
                    runtime_context,
                });
            }
            Err(error) => {
                let message = format!("failed to navigate session history: {error}");
                let event = match operation {
                    crate::transcript::HistoryNavigationOperation::Undo
                    | crate::transcript::HistoryNavigationOperation::Redo => {
                        RunnerEvent::Notice(NoticeEvent::info(message))
                    }
                    crate::transcript::HistoryNavigationOperation::Navigate => {
                        RunnerEvent::Error(ErrorEvent::new(message))
                    }
                };
                let _ = event_tx.send(event);
            }
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
            SessionCommand::ShowHistoryTree
                | SessionCommand::Undo
                | SessionCommand::Redo
                | SessionCommand::NavigateHistory { .. }
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
            SessionCommand::ShowHistoryTree
            | SessionCommand::Undo
            | SessionCommand::Redo
            | SessionCommand::NavigateHistory { .. }
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
    use crate::permission::PermissionMode;
    use crate::request_builder::ModelReasoningEffort;
    use async_openai::{Client, config::OpenAIConfig};
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
            SessionCommand::ShowHistoryTree,
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
            &SessionCommand::ShowHistoryTree
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
            SessionCommand::ShowHistoryTree,
            &mut agent,
            &transcript,
            &tx,
            None,
        )
        .expect("list");
        assert_eq!(outcome, IdleDispatch::Handled);
        match rx.try_recv().expect("history") {
            RunnerEvent::SessionHistoryLoaded { entries } => assert!(entries.is_empty()),
            other => panic!("expected session history, got {other:?}"),
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
    fn unavailable_undo_and_redo_emit_info_notices() {
        for command in [SessionCommand::Undo, SessionCommand::Redo] {
            let transcript = temp_transcript();
            let mut agent = test_agent();
            let (tx, mut rx) = mpsc::unbounded_channel();

            assert_eq!(
                SessionCoordinator::dispatch_idle_command(
                    command,
                    &mut agent,
                    &transcript,
                    &tx,
                    None,
                )
                .expect("dispatch"),
                IdleDispatch::Handled
            );
            match rx.try_recv().expect("notice") {
                RunnerEvent::Notice(notice) => {
                    assert_eq!(notice.kind, crate::session::event::NoticeKind::Info)
                }
                other => panic!("expected info notice, got {other:?}"),
            }
            assert!(rx.try_recv().is_err());
        }
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

    #[test]
    fn navigation_install_is_infallible_after_precommit_validation() {
        let transcript = temp_transcript();
        {
            let mut recorder = transcript.lock().expect("recorder");
            recorder.record_user_message("first").expect("first user");
        }
        let mut agent = test_agent();
        let (tx, mut rx) = mpsc::unbounded_channel();
        assert_eq!(
            SessionCoordinator::dispatch_idle_command(
                SessionCommand::NavigateHistory {
                    target_entry_id: "entry-0".into(),
                },
                &mut agent,
                &transcript,
                &tx,
                None,
            )
            .expect("dispatch"),
            IdleDispatch::Handled
        );
        assert!(matches!(
            rx.try_recv().expect("navigation result"),
            RunnerEvent::SessionResumed { .. }
        ));
        assert_eq!(
            agent.runtime_snapshot_for_test().active_context.branch_id,
            "history-2"
        );
    }

    #[test]
    fn history_tree_lists_abandoned_siblings() {
        let transcript = temp_transcript();
        let mut agent = test_agent();
        let (tx, mut rx) = mpsc::unbounded_channel();
        {
            let mut recorder = transcript.lock().expect("recorder");
            recorder.record_user_message("first").expect("first user");
            recorder
                .record_assistant_message("first answer")
                .expect("first answer");
            recorder.record_user_message("second").expect("second user");
        }
        dispatch_navigation(SessionCommand::Undo, &mut agent, &transcript, &tx, &mut rx);
        {
            let mut recorder = transcript.lock().expect("recorder");
            recorder
                .record_user_message("alternate second")
                .expect("alternate user");
        }

        assert_eq!(
            SessionCoordinator::dispatch_idle_command(
                SessionCommand::ShowHistoryTree,
                &mut agent,
                &transcript,
                &tx,
                None,
            )
            .expect("dispatch"),
            IdleDispatch::Handled
        );
        match rx.try_recv().expect("history") {
            RunnerEvent::SessionHistoryLoaded { entries } => {
                assert!(entries.iter().any(|entry| entry.label == "second"));
                assert!(
                    entries
                        .iter()
                        .any(|entry| entry.label == "alternate second")
                );
            }
            other => panic!("expected history, got {other:?}"),
        }
    }

    #[test]
    fn history_navigation_includes_child_sessions_like_resume_projection() {
        let transcript = temp_transcript();
        let (sessions_dir, parent_session_id) = {
            let recorder = transcript.lock().expect("recorder");
            (
                recorder
                    .path()
                    .parent()
                    .expect("sessions directory")
                    .to_path_buf(),
                recorder.session_id().to_string(),
            )
        };
        let child_dir = crate::transcript::child_sessions_dir(&sessions_dir);
        let mut child = TranscriptRecorder::create(&child_dir).expect("child recorder");
        let child_session_id = child.session_id().to_string();
        child
            .record_user_message("inspect history")
            .expect("record child message");
        drop(child);
        {
            let mut recorder = transcript.lock().expect("recorder");
            recorder
                .record_subagent_started(
                    "run-1",
                    &parent_session_id,
                    "turn-1",
                    &child_session_id,
                    "explorer",
                    "inspect history",
                    1,
                )
                .expect("record child");
            recorder
                .record_user_message("continue with the child result")
                .expect("record parent message");
        }

        let mut agent = test_agent();
        let (tx, mut rx) = mpsc::unbounded_channel();
        assert_eq!(
            crate::session::project_parent_session_view(&transcript, &sessions_dir)
                .expect("resume projection")
                .snapshot
                .snapshot
                .child_sessions
                .len(),
            1
        );
        dispatch_navigation(
            SessionCommand::NavigateHistory {
                target_entry_id: "entry-2".into(),
            },
            &mut agent,
            &transcript,
            &tx,
            &mut rx,
        );
        assert_eq!(agent.runtime_snapshot_for_test().child_sessions.len(), 1);
    }

    fn dispatch_navigation(
        command: SessionCommand,
        agent: &mut Agent<OpenAIConfig>,
        transcript: &Arc<Mutex<TranscriptRecorder>>,
        tx: &mpsc::UnboundedSender<RunnerEvent>,
        rx: &mut mpsc::UnboundedReceiver<RunnerEvent>,
    ) {
        assert_eq!(
            SessionCoordinator::dispatch_idle_command(command, agent, transcript, tx, None)
                .expect("dispatch navigation"),
            IdleDispatch::Handled
        );
        match rx.try_recv().expect("navigation result") {
            RunnerEvent::SessionResumed { .. } => {}
            other => panic!("expected history navigation to resume session, got {other:?}"),
        }
    }

    #[test]
    fn durable_navigation_reopens_with_multilevel_redo_and_preserves_siblings_on_new_prompt() {
        let transcript = temp_transcript();
        let path = transcript.lock().expect("recorder").path().to_path_buf();
        {
            let mut recorder = transcript.lock().expect("recorder");
            recorder
                .record_session_started("gpt-test")
                .expect("started");
            recorder.record_user_message("first").expect("first user");
            recorder
                .record_assistant_message("first answer")
                .expect("first answer");
            recorder.record_user_message("second").expect("second user");
            recorder
                .record_assistant_message("second answer")
                .expect("second answer");
        }
        let mut agent = test_agent();
        let (tx, mut rx) = mpsc::unbounded_channel();

        dispatch_navigation(SessionCommand::Undo, &mut agent, &transcript, &tx, &mut rx);
        dispatch_navigation(SessionCommand::Undo, &mut agent, &transcript, &tx, &mut rx);
        let records = crate::transcript::read_records(&path).expect("records after undo");
        assert_eq!(
            crate::transcript::transcript_projection::history_navigation_state(&records),
            Some(
                crate::transcript::transcript_projection::HistoryNavigationState {
                    target_sequence: 0,
                    redo_stack: vec![5, 3],
                }
            )
        );
        assert!(matches!(
            records[8].event,
            crate::transcript::TranscriptEvent::ContextBranchCreated { .. }
        ));
        assert!(matches!(
            records[9].event,
            crate::transcript::TranscriptEvent::ContextCheckout { .. }
        ));
        assert!(matches!(
            records[10].event,
            crate::transcript::TranscriptEvent::HistoryNavigation { .. }
        ));

        let first_navigation = &records[5..8];
        let transaction_id = match &first_navigation[0].event {
            crate::transcript::TranscriptEvent::ContextBranchCreated { .. } => {
                first_navigation[0].sequence
            }
            other => panic!("expected branch creation, got {other:?}"),
        };
        assert_eq!(transaction_id, 6);

        let (sessions_dir, session_id) = {
            let recorder = transcript.lock().expect("recorder");
            (
                recorder
                    .path()
                    .parent()
                    .expect("sessions directory")
                    .to_path_buf(),
                recorder.session_id().to_string(),
            )
        };
        let reopened = Arc::new(Mutex::new(
            TranscriptRecorder::open_existing(&sessions_dir, &session_id).expect("reopen"),
        ));
        dispatch_navigation(SessionCommand::Redo, &mut agent, &reopened, &tx, &mut rx);
        dispatch_navigation(SessionCommand::Redo, &mut agent, &reopened, &tx, &mut rx);

        {
            let mut recorder = reopened.lock().expect("reopened recorder");
            recorder
                .record_user_message("new branch")
                .expect("new prompt");
        }
        let records = crate::transcript::read_records(&path).expect("final records");
        assert_eq!(
            crate::transcript::transcript_projection::history_navigation_state(&records),
            None,
            "a new prompt invalidates persisted redo"
        );
        let entries =
            crate::transcript::transcript_projection::project_session_history_tree(&records);
        assert!(entries.iter().any(|entry| entry.label == "second"));
        assert!(entries.iter().any(|entry| entry.label == "new branch"));
        assert!(entries.iter().any(|entry| {
            entry.label == "new branch" && entry.parent_id.as_deref() == Some("entry-5")
        }));
    }
}
