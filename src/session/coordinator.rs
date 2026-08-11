//! Session coordinator: grows into the home for session-owned command execution.
//!
//! Owns idle commands that do not start turns. The TUI still hosts the async
//! control loop and delegates idle work here.

#[cfg(test)]
use std::cell::Cell;
use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::Result;

use crate::agent::Agent;
use crate::session::child_view::{
    current_session_records, project_child_session_view, project_parent_session_view,
    sessions_dir_from_transcript,
};
use crate::session::command::SessionCommand;
use crate::session::event::{ErrorEvent, NoticeEvent};
use crate::session::restore::{
    apply_prepared_restored_route, apply_restored_permission_mode, prepare_restored_model_route,
    restored_messages_from_protocol_frames,
};
use crate::session::runner::SessionTransportEvent;
use crate::session::settings::{apply_permission_mode, apply_reasoning_effort};
use crate::transcript::TranscriptRecorder;

#[derive(Debug)]
struct NavigationError {
    error: anyhow::Error,
    fast_mode_auto_disabled: bool,
}

impl NavigationError {
    fn before_fast_mode_reconciliation(error: anyhow::Error) -> Self {
        Self {
            error,
            fast_mode_auto_disabled: false,
        }
    }
}

impl From<anyhow::Error> for NavigationError {
    fn from(error: anyhow::Error) -> Self {
        Self::before_fast_mode_reconciliation(error)
    }
}
use tokio::sync::mpsc;

/// Outcome of attempting to run a [`SessionCommand`] as idle session work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdleDispatch {
    /// Command fully handled (events emitted / agent mutated as needed).
    Handled,
    /// History navigation committed successfully.
    HistoryNavigated,
    /// Command requires turn execution or frontend-private orchestration.
    NotIdle,
}

/// Session-owned coordinator entry point for multi-frontend backends.
#[derive(Debug, Default)]
pub struct SessionCoordinator;

#[cfg(test)]
thread_local! {
    static FAIL_HISTORY_NAVIGATION_COMMIT: Cell<bool> = const { Cell::new(false) };
}

impl SessionCoordinator {
    /// Execute an idle session command, emitting [`SessionTransportEvent`]s for the
    /// frontend bridge. Returns [`IdleDispatch::NotIdle`] when the command is
    /// outside the current coordinator surface.
    ///
    /// `sessions_dir` is required for child/parent view commands; when `None`,
    /// those commands resolve the directory from the live transcript path.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn dispatch_idle_command(
        command: SessionCommand,
        agent: &mut Agent<async_openai::config::OpenAIConfig>,
        transcript: &Arc<Mutex<TranscriptRecorder>>,
        event_tx: &mpsc::UnboundedSender<SessionTransportEvent>,
        sessions_dir: Option<&Path>,
    ) -> Result<IdleDispatch> {
        Self::dispatch_idle_command_with_history_prepare(
            command,
            agent,
            transcript,
            event_tx,
            sessions_dir,
            |_| Ok(()),
        )
    }

    pub(crate) fn dispatch_idle_command_with_history_prepare<F>(
        command: SessionCommand,
        agent: &mut Agent<async_openai::config::OpenAIConfig>,
        transcript: &Arc<Mutex<TranscriptRecorder>>,
        event_tx: &mpsc::UnboundedSender<SessionTransportEvent>,
        sessions_dir: Option<&Path>,
        mut prepare_history: F,
    ) -> Result<IdleDispatch>
    where
        F: FnMut(&crate::transcript::transcript_projection::RuntimeRestoreSnapshot) -> Result<()>,
    {
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
                        let _ =
                            event_tx.send(SessionTransportEvent::SessionHistoryLoaded { entries });
                    }
                    Err(error) => {
                        let _ = event_tx.send(SessionTransportEvent::Error(ErrorEvent::new(
                            format!("failed to load session history: {error}"),
                        )));
                    }
                }
                Ok(IdleDispatch::Handled)
            }
            SessionCommand::NavigateHistory { target_entry_id } => {
                let navigated = match Self::entry_sequence(&target_entry_id) {
                    Ok(target_sequence) => Self::navigate_history(
                        agent,
                        transcript,
                        event_tx,
                        target_sequence,
                        crate::transcript::HistoryNavigationOperation::Navigate,
                        Vec::new(),
                        &mut prepare_history,
                    ),
                    Err(error) => {
                        let _ = event_tx.send(SessionTransportEvent::Error(ErrorEvent::new(
                            error.to_string(),
                        )));
                        false
                    }
                };
                Ok(if navigated {
                    IdleDispatch::HistoryNavigated
                } else {
                    IdleDispatch::Handled
                })
            }
            SessionCommand::Undo => Ok(
                if Self::navigate_undo(agent, transcript, event_tx, &mut prepare_history) {
                    IdleDispatch::HistoryNavigated
                } else {
                    IdleDispatch::Handled
                },
            ),
            SessionCommand::Redo => Ok(
                if Self::navigate_redo(agent, transcript, event_tx, &mut prepare_history) {
                    IdleDispatch::HistoryNavigated
                } else {
                    IdleDispatch::Handled
                },
            ),
            SessionCommand::SetPermissionMode(mode) => {
                if let Err(error) = apply_permission_mode(agent, transcript, mode) {
                    let message = format!("failed to set permission mode: {error}");
                    let _ = event_tx.send(SessionTransportEvent::SettingChangeFailed {
                        command: SessionCommand::SetPermissionMode(mode),
                    });
                    let _ = event_tx.send(SessionTransportEvent::Error(ErrorEvent::new(message)));
                } else {
                    let _ = event_tx.send(SessionTransportEvent::PermissionModeChanged {
                        mode: mode.to_string(),
                    });
                }
                Ok(IdleDispatch::Handled)
            }
            SessionCommand::SetModel(_) | SessionCommand::SetExpertModel { .. } => {
                Ok(IdleDispatch::NotIdle)
            }
            SessionCommand::ToggleFastMode => {
                let Some(fast_mode) = agent.fast_mode() else {
                    let _ = event_tx.send(SessionTransportEvent::Notice(NoticeEvent::info(
                        "Fast mode unavailable",
                    )));
                    return Ok(IdleDispatch::Handled);
                };
                match fast_mode.toggle(agent.model()) {
                    Ok(toggle) => {
                        let (enabled, notice) = match toggle {
                            crate::fast_mode::FastModeToggle::Enabled => {
                                (true, "Fast mode enabled")
                            }
                            crate::fast_mode::FastModeToggle::Disabled => {
                                (false, "Fast mode disabled")
                            }
                            crate::fast_mode::FastModeToggle::Unavailable => {
                                (false, "Fast mode unavailable for current model")
                            }
                        };
                        let _ = event_tx.send(SessionTransportEvent::FastModeChanged { enabled });
                        let _ =
                            event_tx.send(SessionTransportEvent::Notice(NoticeEvent::info(notice)));
                    }
                    Err(error) => {
                        let _ = event_tx.send(SessionTransportEvent::Error(ErrorEvent::new(
                            format!("failed to toggle fast mode: {error}"),
                        )));
                    }
                }
                Ok(IdleDispatch::Handled)
            }
            SessionCommand::SetReasoningEffort(effort) => {
                if let Err(error) = apply_reasoning_effort(agent, effort.clone()) {
                    let message = error.to_string();
                    let _ = event_tx.send(SessionTransportEvent::SettingChangeFailed {
                        command: SessionCommand::SetReasoningEffort(effort),
                    });
                    let _ =
                        event_tx.send(SessionTransportEvent::Notice(NoticeEvent::info(message)));
                } else {
                    let _ = event_tx.send(SessionTransportEvent::ReasoningEffortChanged { effort });
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

    fn navigate_undo<F>(
        agent: &mut Agent<async_openai::config::OpenAIConfig>,
        transcript: &Arc<Mutex<TranscriptRecorder>>,
        event_tx: &mpsc::UnboundedSender<SessionTransportEvent>,
        prepare_history: &mut F,
    ) -> bool
    where
        F: FnMut(&crate::transcript::transcript_projection::RuntimeRestoreSnapshot) -> Result<()>,
    {
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
                prepare_history,
            ),
            Err(error) => {
                let _ = event_tx.send(SessionTransportEvent::Notice(NoticeEvent::info(
                    error.to_string(),
                )));
                false
            }
        }
    }

    fn navigate_redo<F>(
        agent: &mut Agent<async_openai::config::OpenAIConfig>,
        transcript: &Arc<Mutex<TranscriptRecorder>>,
        event_tx: &mpsc::UnboundedSender<SessionTransportEvent>,
        prepare_history: &mut F,
    ) -> bool
    where
        F: FnMut(&crate::transcript::transcript_projection::RuntimeRestoreSnapshot) -> Result<()>,
    {
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
                prepare_history,
            ),
            Err(error) => {
                let _ = event_tx.send(SessionTransportEvent::Notice(NoticeEvent::info(
                    error.to_string(),
                )));
                false
            }
        }
    }

    fn emit_fast_mode_auto_disabled(event_tx: &mpsc::UnboundedSender<SessionTransportEvent>) {
        let _ = event_tx.send(SessionTransportEvent::FastModeChanged { enabled: false });
        let _ = event_tx.send(SessionTransportEvent::Notice(NoticeEvent::info(
            "Fast mode auto-disabled: current model is unavailable",
        )));
    }

    fn entry_sequence(entry_id: &str) -> Result<u64> {
        entry_id
            .strip_prefix("entry-")
            .ok_or_else(|| anyhow::anyhow!("invalid history entry id '{entry_id}'"))?
            .parse()
            .map_err(Into::into)
    }

    fn navigate_history<F>(
        agent: &mut Agent<async_openai::config::OpenAIConfig>,
        transcript: &Arc<Mutex<TranscriptRecorder>>,
        event_tx: &mpsc::UnboundedSender<SessionTransportEvent>,
        target_sequence: u64,
        operation: crate::transcript::HistoryNavigationOperation,
        redo_stack: Vec<u64>,
        prepare_history: &mut F,
    ) -> bool
    where
        F: FnMut(&crate::transcript::transcript_projection::RuntimeRestoreSnapshot) -> Result<()>,
    {
        let result = (|| -> std::result::Result<_, NavigationError> {
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
            prepare_history(&snapshot)?;
            let (protocol_frames, runtime_snapshot) = agent.validate_runtime_snapshot_restore(
                snapshot.protocol_frames.clone(),
                snapshot.snapshot.clone(),
            )?;
            let route = prepare_restored_model_route(agent, snapshot.latest_model.as_deref())?;
            let fast_mode_model = route
                .as_ref()
                .map_or_else(|| agent.model(), |route| route.target_model());
            let prepared_fast_mode_disable = agent
                .prepare_fast_mode_auto_disable(fast_mode_model)
                .map_err(|error| NavigationError {
                    error,
                    fast_mode_auto_disabled: false,
                })?;
            let fast_mode_auto_disabled = prepared_fast_mode_disable.is_some();
            #[cfg(test)]
            if FAIL_HISTORY_NAVIGATION_COMMIT.with(|fail| fail.replace(false)) {
                return Err(NavigationError {
                    error: anyhow::anyhow!("injected history navigation commit failure"),
                    fast_mode_auto_disabled: false,
                });
            }
            recorder.preflight_history_navigation_transaction(
                branch_id.clone(),
                parent_branch_id.clone(),
                target_sequence,
                operation,
                redo_stack.clone(),
            )?;
            if let Some(prepared_fast_mode_disable) = prepared_fast_mode_disable {
                prepared_fast_mode_disable
                    .commit()
                    .map_err(|error| NavigationError {
                        error,
                        fast_mode_auto_disabled: false,
                    })?;
            }
            recorder
                .record_history_navigation_transaction(
                    branch_id.clone(),
                    parent_branch_id,
                    target_sequence,
                    operation,
                    redo_stack,
                )
                .map_err(|error| NavigationError {
                    error,
                    fast_mode_auto_disabled,
                })?;
            crate::transcript::sync_recorder_branch(&mut recorder, &branch_id);
            Ok((
                snapshot,
                runtime_context,
                protocol_frames,
                runtime_snapshot,
                route,
                fast_mode_auto_disabled,
            ))
        })();
        match result {
            Ok((
                snapshot,
                runtime_context,
                protocol_frames,
                runtime_snapshot,
                route,
                fast_mode_auto_disabled,
            )) => {
                apply_prepared_restored_route(agent, route);
                apply_restored_permission_mode(agent, snapshot.latest_permission_mode.as_deref());
                agent.install_validated_runtime_snapshot(protocol_frames, runtime_snapshot);
                agent.restore_turn_sequence(snapshot.max_turn_id);
                if fast_mode_auto_disabled {
                    Self::emit_fast_mode_auto_disabled(event_tx);
                }
                let expert_models =
                    crate::transcript::restore_latest_expert_models(&snapshot.records);
                let _ = event_tx.send(SessionTransportEvent::SessionResumed {
                    session_id: snapshot.session_id,
                    branch_id: snapshot.branch_id,
                    messages: restored_messages_from_protocol_frames(&snapshot.protocol_frames),
                    records: snapshot.records,
                    evidence_count: 0,
                    model_id: Some(agent.route_display_name()),
                    token_usage: None,
                    runtime_context,
                    expert_models,
                });
                true
            }
            Err(error) => {
                if error.fast_mode_auto_disabled {
                    Self::emit_fast_mode_auto_disabled(event_tx);
                }
                let message = format!("failed to navigate session history: {}", error.error);
                let event = if error.fast_mode_auto_disabled {
                    SessionTransportEvent::Error(ErrorEvent::new(message))
                } else {
                    match operation {
                        crate::transcript::HistoryNavigationOperation::Undo
                        | crate::transcript::HistoryNavigationOperation::Redo => {
                            SessionTransportEvent::Notice(NoticeEvent::info(message))
                        }
                        crate::transcript::HistoryNavigationOperation::Navigate => {
                            SessionTransportEvent::Error(ErrorEvent::new(message))
                        }
                    }
                };
                let _ = event_tx.send(event);
                false
            }
        }
    }

    /// Emit parent transcript view events without requiring a mutable agent.
    ///
    /// Safe to call while a turn holds `&mut Agent` (navigation-only path).
    pub(crate) fn emit_view_parent(
        transcript: &Arc<Mutex<TranscriptRecorder>>,
        event_tx: &mpsc::UnboundedSender<SessionTransportEvent>,
        sessions_dir: Option<&Path>,
    ) {
        let dir = match sessions_dir.map(Path::to_path_buf) {
            Some(dir) => Ok(dir),
            None => sessions_dir_from_transcript(transcript),
        };
        match dir.and_then(|dir| project_parent_session_view(transcript, dir)) {
            Ok(projected) => {
                let snapshot = projected.snapshot;
                let _ = event_tx.send(SessionTransportEvent::ParentSessionViewed {
                    session_id: snapshot.session_id,
                    branch_id: snapshot.branch_id,
                    records: snapshot.records,
                    model_id: snapshot.latest_model,
                    token_usage: None,
                    runtime_context: projected.runtime_context,
                });
            }
            Err(error) => {
                let _ = event_tx.send(SessionTransportEvent::Error(ErrorEvent::new(format!(
                    "failed to view parent transcript: {error}"
                ))));
            }
        }
    }

    /// Emit child transcript view events without requiring a mutable agent.
    ///
    /// Safe to call while a turn holds `&mut Agent` (navigation-only path).
    /// Returns the selected child session id when a child view was emitted.
    pub(crate) fn emit_view_child(
        transcript: &Arc<Mutex<TranscriptRecorder>>,
        event_tx: &mpsc::UnboundedSender<SessionTransportEvent>,
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
                let _ = event_tx.send(SessionTransportEvent::Notice(NoticeEvent::info(
                    "No child subagent transcripts for this session",
                )));
                None
            }
            Ok(Some(view)) => {
                let child_session_id = view.child_session_id.clone();
                let _ = event_tx.send(SessionTransportEvent::ChildSessionViewed {
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
                let _ = event_tx.send(SessionTransportEvent::Error(ErrorEvent::new(format!(
                    "failed to view child transcript: {error}"
                ))));
                None
            }
        }
    }

    /// Whether this command is currently handled as idle work by the coordinator.
    #[cfg(test)]
    pub fn is_idle_command(command: &SessionCommand) -> bool {
        matches!(
            command,
            SessionCommand::ShowHistoryTree
                | SessionCommand::Undo
                | SessionCommand::Redo
                | SessionCommand::NavigateHistory { .. }
                | SessionCommand::SetPermissionMode(_)
                | SessionCommand::ToggleFastMode
                | SessionCommand::SetReasoningEffort(_)
                | SessionCommand::ViewParent
                | SessionCommand::ViewChild { .. }
        )
    }

    /// Exhaustive ownership table for migration tracking.
    #[cfg(test)]
    pub fn ownership(command: &SessionCommand) -> CommandOwnership {
        match command {
            SessionCommand::ShowHistoryTree
            | SessionCommand::Undo
            | SessionCommand::Redo
            | SessionCommand::NavigateHistory { .. }
            | SessionCommand::SetPermissionMode(_)
            | SessionCommand::ToggleFastMode
            | SessionCommand::SetReasoningEffort(_)
            | SessionCommand::ViewParent
            | SessionCommand::ViewChild { .. } => CommandOwnership::IdleCoordinator,
            SessionCommand::SubmitPrompt(_)
            | SessionCommand::DelegateSubagent { .. }
            | SessionCommand::Compact
            | SessionCommand::SetModel(_)
            | SessionCommand::SetExpertModel { .. }
            | SessionCommand::ResumeSession(_)
            | SessionCommand::NewSession
            | SessionCommand::ToggleMcpServer(_)
            | SessionCommand::Interrupt => CommandOwnership::FrontendHosted,
        }
    }
}

/// Where a [`SessionCommand`] is executed today.
#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandOwnership {
    /// Fully handled by [`SessionCoordinator::dispatch_idle_command`].
    IdleCoordinator,
    /// Still executed by the session executor loop and/or CLI-specific paths.
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
    fn fast_mode_toggle_persistence_failure_emits_error_without_state_change() {
        let transcript = temp_transcript();
        let mut agent = test_agent();
        let fast_mode_path = std::env::temp_dir().join(format!(
            "letcode-session-coordinator-fast-mode-file-{}.toml",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("time ok")
                .as_nanos()
        ));
        std::fs::write(&fast_mode_path, "not a config file").expect("create blocking file");
        let fast_mode = crate::fast_mode::FastMode::load(&fast_mode_path, false);
        agent.set_fast_mode(fast_mode);
        let (tx, mut rx) = mpsc::unbounded_channel();

        assert_eq!(
            SessionCoordinator::dispatch_idle_command(
                SessionCommand::ToggleFastMode,
                &mut agent,
                &transcript,
                &tx,
                None,
            )
            .expect("toggle dispatch"),
            IdleDispatch::Handled
        );
        assert!(!agent.fast_mode_enabled());
        match rx.try_recv().expect("persistence error") {
            SessionTransportEvent::Error(error) => {
                assert!(error.message.contains("failed to toggle fast mode"));
            }
            other => panic!("expected fast mode error, got {other:?}"),
        }
        assert!(rx.try_recv().is_err());
    }

    fn history_navigation_with_unsupported_model() -> (
        Arc<Mutex<TranscriptRecorder>>,
        Agent<OpenAIConfig>,
        std::path::PathBuf,
    ) {
        let transcript = temp_transcript();
        {
            let mut recorder = transcript.lock().expect("recorder");
            recorder.record_user_message("first").expect("first user");
            recorder
                .record_model_changed("gpt-5.5", "claude-4")
                .expect("model changed");
            recorder.record_user_message("second").expect("second user");
        }
        let mut agent = test_agent();
        let fast_mode_dir = std::env::temp_dir().join(format!(
            "letcode-session-coordinator-navigation-fast-mode-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("time ok")
                .as_nanos()
        ));
        let fast_mode_config_path = fast_mode_dir.join("letcode.toml");
        std::fs::create_dir_all(&fast_mode_dir).expect("create Fast Mode config directory");
        std::fs::write(
            &fast_mode_config_path,
            r#"active_provider = "primary"

[providers.primary]
base_url = "https://primary.invalid/v1"
api_key = "primary-key"
protocol = "responses"
[providers.primary.models."gpt-5.5"]
"#,
        )
        .expect("write Fast Mode config");
        let fast_mode = crate::fast_mode::FastMode::load(&fast_mode_config_path, true);
        agent.set_fast_mode(fast_mode);
        (transcript, agent, fast_mode_config_path)
    }

    #[test]
    fn navigation_fast_mode_persistence_failure_prevents_transaction() {
        let (transcript, mut agent, fast_mode_path) = history_navigation_with_unsupported_model();
        std::fs::write(&fast_mode_path, "not a config file").expect("block persistence");
        let before = {
            let recorder = transcript.lock().expect("recorder");
            crate::transcript::read_records(recorder.path()).expect("read initial records")
        };
        let (tx, mut rx) = mpsc::unbounded_channel();

        SessionCoordinator::dispatch_idle_command(
            SessionCommand::NavigateHistory {
                target_entry_id: "entry-3".into(),
            },
            &mut agent,
            &transcript,
            &tx,
            None,
        )
        .expect("dispatch");

        assert_eq!(agent.model(), "gpt-5.5");
        assert!(agent.fast_mode_enabled());
        let after = {
            let recorder = transcript.lock().expect("recorder");
            crate::transcript::read_records(recorder.path()).expect("read final records")
        };
        assert_eq!(after.len(), before.len());
        assert!(matches!(
            rx.try_recv().expect("navigation error"),
            SessionTransportEvent::Error(error) if error.message.contains("failed to navigate session history")
        ));
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn navigation_commit_failure_keeps_fast_mode_unchanged() {
        let (transcript, mut agent, _fast_mode_dir) = history_navigation_with_unsupported_model();
        let before = {
            let recorder = transcript.lock().expect("recorder");
            crate::transcript::read_records(recorder.path()).expect("read initial records")
        };
        FAIL_HISTORY_NAVIGATION_COMMIT.with(|fail| fail.set(true));
        let (tx, mut rx) = mpsc::unbounded_channel();

        SessionCoordinator::dispatch_idle_command(
            SessionCommand::NavigateHistory {
                target_entry_id: "entry-3".into(),
            },
            &mut agent,
            &transcript,
            &tx,
            None,
        )
        .expect("dispatch");

        assert_eq!(agent.model(), "gpt-5.5");
        assert!(agent.fast_mode_enabled());
        let after = {
            let recorder = transcript.lock().expect("recorder");
            crate::transcript::read_records(recorder.path()).expect("read final records")
        };
        assert_eq!(after.len(), before.len());
        assert!(matches!(
            rx.try_recv().expect("navigation error"),
            SessionTransportEvent::Error(error)
                if error.message.contains("injected history navigation commit failure")
        ));
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn history_prepare_failure_prevents_navigation_commit() {
        let transcript = temp_transcript();
        {
            let mut recorder = transcript.lock().expect("recorder");
            recorder.record_user_message("first").expect("first user");
        }
        let before = {
            let recorder = transcript.lock().expect("recorder");
            crate::transcript::read_records(recorder.path()).expect("records before navigation")
        };
        let mut agent = test_agent();
        let (tx, mut rx) = mpsc::unbounded_channel();

        SessionCoordinator::dispatch_idle_command_with_history_prepare(
            SessionCommand::NavigateHistory {
                target_entry_id: "entry-0".into(),
            },
            &mut agent,
            &transcript,
            &tx,
            None,
            |_| anyhow::bail!("expert factory unavailable"),
        )
        .expect("dispatch");

        let after = {
            let recorder = transcript.lock().expect("recorder");
            crate::transcript::read_records(recorder.path()).expect("records after navigation")
        };
        assert_eq!(after.len(), before.len());
        assert_eq!(
            after
                .iter()
                .map(|record| record.sequence)
                .collect::<Vec<_>>(),
            before
                .iter()
                .map(|record| record.sequence)
                .collect::<Vec<_>>()
        );
        assert!(matches!(
            rx.try_recv().expect("navigation error"),
            SessionTransportEvent::Error(error)
                if error.message.contains("expert factory unavailable")
        ));
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
            IdleDispatch::HistoryNavigated
        );
        assert!(matches!(
            rx.try_recv().expect("navigation result"),
            SessionTransportEvent::SessionResumed { .. }
        ));
        assert_eq!(
            agent.runtime_snapshot_for_test().active_context.branch_id,
            "history-2"
        );
    }

    fn dispatch_navigation(
        command: SessionCommand,
        agent: &mut Agent<OpenAIConfig>,
        transcript: &Arc<Mutex<TranscriptRecorder>>,
        tx: &mpsc::UnboundedSender<SessionTransportEvent>,
        rx: &mut mpsc::UnboundedReceiver<SessionTransportEvent>,
    ) {
        assert_eq!(
            SessionCoordinator::dispatch_idle_command(command, agent, transcript, tx, None)
                .expect("dispatch navigation"),
            IdleDispatch::HistoryNavigated
        );
        match rx.try_recv().expect("navigation result") {
            SessionTransportEvent::SessionResumed { .. } => {}
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
