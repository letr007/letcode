//! Session engine command and control scheduling.

use super::*;

/// Backend-private command payload for the session executor.
///
/// This is intentionally crate-private: frontend code submits
/// [`SessionCommand`] through [`SessionEngineIngress`] instead.
#[derive(Debug)]
pub(crate) enum SessionEngineCommand {
    Prompt(crate::user_content::UserMessageSubmission),
    DelegateSubagent {
        agent_name: String,
        task: String,
    },
    BackgroundSubagentCompleted {
        parent_session_id: String,
        parent_tool_call_id: Option<String>,
        result: Result<crate::subagent::SubagentRunSummary, String>,
    },
    ContinueSession,
    Compact,
    ShowHistoryTree,
    Undo,
    Redo,
    NavigateHistory {
        target_entry_id: String,
    },
    ViewChild {
        navigation: crate::command::ChildNavigation,
        anchor_child_session_id: Option<String>,
    },
    ViewParent,
    SetPermissionMode(crate::permission::PermissionMode),
    SetModel(String),
    SetExpertModel {
        agent_name: String,
        model_id: String,
    },
    SetExpertAllowedModels {
        agent_name: String,
        model_ids: Vec<String>,
    },
    ToggleFastMode,
    SetReasoningEffort(crate::request_builder::ModelReasoningEffort),
    ResumeSession(String),
    NewSession,
    ToggleMcpServer(String),
    /// Toggle the anchored bootstrap experiment for this session.
    AnchoredToggle,
    #[cfg(test)]
    InspectHistory(tokio::sync::oneshot::Sender<Vec<crate::request_builder::HistoryItem>>),
}

impl SessionEngineCommand {
    pub(crate) fn from_session_command(command: SessionCommand) -> Self {
        match command {
            SessionCommand::SubmitPrompt(prompt) => Self::Prompt(prompt),
            SessionCommand::DelegateSubagent { agent_name, task } => {
                Self::DelegateSubagent { agent_name, task }
            }
            SessionCommand::Compact => Self::Compact,
            SessionCommand::ShowHistoryTree => Self::ShowHistoryTree,
            SessionCommand::Undo => Self::Undo,
            SessionCommand::Redo => Self::Redo,
            SessionCommand::NavigateHistory { target_entry_id } => {
                Self::NavigateHistory { target_entry_id }
            }
            SessionCommand::ViewChild {
                navigation,
                anchor_child_session_id,
            } => Self::ViewChild {
                navigation,
                anchor_child_session_id,
            },
            SessionCommand::ViewParent => Self::ViewParent,
            SessionCommand::SetPermissionMode(mode) => Self::SetPermissionMode(mode),
            SessionCommand::SetModel(model) => Self::SetModel(model),
            SessionCommand::SetExpertModel {
                agent_name,
                model_id,
            } => Self::SetExpertModel {
                agent_name,
                model_id,
            },
            SessionCommand::SetExpertAllowedModels {
                agent_name,
                model_ids,
            } => Self::SetExpertAllowedModels {
                agent_name,
                model_ids,
            },
            SessionCommand::ToggleFastMode => Self::ToggleFastMode,
            SessionCommand::AnchoredToggle => Self::AnchoredToggle,
            SessionCommand::SetReasoningEffort(effort) => Self::SetReasoningEffort(effort),
            SessionCommand::ResumeSession(session_id) => Self::ResumeSession(session_id),
            SessionCommand::NewSession => Self::NewSession,
            SessionCommand::ToggleMcpServer(server_name) => Self::ToggleMcpServer(server_name),
            SessionCommand::Interrupt => unreachable!("interrupt has its own ingress intent"),
        }
    }
}

/// Ordered controls consumed by the session executor.
#[derive(Debug)]
pub(crate) enum SessionEngineControl {
    Command(SessionEngineCommand),
    Interrupt,
    Shutdown,
}

/// Map private session transport commands that the session coordinator owns as idle work.
pub(crate) fn session_engine_command_as_session_command(
    command: &SessionEngineCommand,
) -> Option<crate::session::SessionCommand> {
    match command {
        SessionEngineCommand::ShowHistoryTree => {
            Some(crate::session::SessionCommand::ShowHistoryTree)
        }
        SessionEngineCommand::Undo => Some(crate::session::SessionCommand::Undo),
        SessionEngineCommand::Redo => Some(crate::session::SessionCommand::Redo),
        SessionEngineCommand::NavigateHistory { target_entry_id } => {
            Some(crate::session::SessionCommand::NavigateHistory {
                target_entry_id: target_entry_id.clone(),
            })
        }
        SessionEngineCommand::SetPermissionMode(mode) => {
            Some(crate::session::SessionCommand::SetPermissionMode(*mode))
        }
        SessionEngineCommand::AnchoredToggle => {
            Some(crate::session::SessionCommand::AnchoredToggle)
        }
        SessionEngineCommand::ToggleFastMode => {
            Some(crate::session::SessionCommand::ToggleFastMode)
        }
        SessionEngineCommand::SetReasoningEffort(effort) => Some(
            crate::session::SessionCommand::SetReasoningEffort(effort.clone()),
        ),
        SessionEngineCommand::ViewParent => Some(crate::session::SessionCommand::ViewParent),
        SessionEngineCommand::ViewChild {
            navigation,
            anchor_child_session_id,
        } => Some(crate::session::SessionCommand::ViewChild {
            navigation: *navigation,
            anchor_child_session_id: anchor_child_session_id.clone(),
        }),
        SessionEngineCommand::DelegateSubagent { agent_name, task } => {
            Some(crate::session::SessionCommand::DelegateSubagent {
                agent_name: agent_name.clone(),
                task: task.clone(),
            })
        }
        SessionEngineCommand::BackgroundSubagentCompleted { .. }
        | SessionEngineCommand::ContinueSession => None,
        SessionEngineCommand::Compact => Some(crate::session::SessionCommand::Compact),
        SessionEngineCommand::SetModel(model) => {
            Some(crate::session::SessionCommand::SetModel(model.clone()))
        }
        SessionEngineCommand::SetExpertModel {
            agent_name,
            model_id,
        } => Some(crate::session::SessionCommand::SetExpertModel {
            agent_name: agent_name.clone(),
            model_id: model_id.clone(),
        }),
        SessionEngineCommand::SetExpertAllowedModels {
            agent_name,
            model_ids,
        } => Some(crate::session::SessionCommand::SetExpertAllowedModels {
            agent_name: agent_name.clone(),
            model_ids: model_ids.clone(),
        }),
        SessionEngineCommand::ResumeSession(session_id) => Some(
            crate::session::SessionCommand::ResumeSession(session_id.clone()),
        ),
        SessionEngineCommand::NewSession => Some(crate::session::SessionCommand::NewSession),
        SessionEngineCommand::ToggleMcpServer(server_name) => Some(
            crate::session::SessionCommand::ToggleMcpServer(server_name.clone()),
        ),
        SessionEngineCommand::Prompt(_) => None,
        #[cfg(test)]
        SessionEngineCommand::InspectHistory(_) => None,
    }
}

pub(crate) fn send_setting_change_failed(
    session_transport_tx: &mpsc::UnboundedSender<SessionTransportEvent>,
    command: crate::session::SessionCommand,
    message: impl Into<String>,
) {
    let message = message.into();
    let _ = session_transport_tx.send(SessionTransportEvent::SettingChangeFailed { command });
    let _ = session_transport_tx.send(SessionTransportEvent::Error(ErrorEvent::new(message)));
}

pub(crate) fn session_engine_command_as_idle_session_command(
    command: &SessionEngineCommand,
) -> Option<crate::session::SessionCommand> {
    match session_engine_command_as_session_command(command)? {
        command @ (crate::session::SessionCommand::ShowHistoryTree
        | crate::session::SessionCommand::Undo
        | crate::session::SessionCommand::Redo
        | crate::session::SessionCommand::NavigateHistory { .. }
        | crate::session::SessionCommand::SetPermissionMode(_)
        | crate::session::SessionCommand::ToggleFastMode
        | crate::session::SessionCommand::SetReasoningEffort(_)
        | crate::session::SessionCommand::ViewParent
        | crate::session::SessionCommand::ViewChild { .. }) => Some(command),
        crate::session::SessionCommand::SubmitPrompt(_)
        | crate::session::SessionCommand::DelegateSubagent { .. }
        | crate::session::SessionCommand::Compact
        | crate::session::SessionCommand::SetModel(_)
        | crate::session::SessionCommand::SetExpertModel { .. }
        | crate::session::SessionCommand::SetExpertAllowedModels { .. }
        | crate::session::SessionCommand::ResumeSession(_)
        | crate::session::SessionCommand::NewSession
        | crate::session::SessionCommand::ToggleMcpServer(_)
        | crate::session::SessionCommand::AnchoredToggle
        | crate::session::SessionCommand::Interrupt => None,
    }
}

pub(crate) enum ActiveSessionOperation<T> {
    Interrupted,
    Shutdown,
    Completed(T),
    RunnerEvent(SessionTransportEvent),
    Command(Option<SessionEngineCommand>),
}

pub(crate) fn handle_active_turn_command(
    command: SessionEngineCommand,
    parked_commands: &mut VecDeque<SessionEngineCommand>,
    session_transport_tx: &mpsc::UnboundedSender<SessionTransportEvent>,
) {
    if matches!(
        command,
        SessionEngineCommand::BackgroundSubagentCompleted { .. }
    ) {
        enqueue_deferred_command(parked_commands, command);
        return;
    }

    let disposition = session_engine_command_as_session_command(&command)
        .map(|command| command.active_turn_disposition())
        .unwrap_or(crate::session::ActiveTurnCommandDisposition::Defer);
    match disposition {
        crate::session::ActiveTurnCommandDisposition::Defer => {
            park_active_turn_command(parked_commands, command, session_transport_tx);
        }
        crate::session::ActiveTurnCommandDisposition::Reject => {
            let _ = session_transport_tx.send(SessionTransportEvent::Notice(NoticeEvent::info(
                "Turn still running",
            )));
        }
        crate::session::ActiveTurnCommandDisposition::QueuePrompt
        | crate::session::ActiveTurnCommandDisposition::Immediate
        | crate::session::ActiveTurnCommandDisposition::Interrupt => {
            unreachable!("active turn command is handled before the deferred fallback")
        }
    }
}

pub(crate) enum ManualCompactionOperation<T> {
    Interrupted,
    Shutdown,
    Completed(T),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum QueuedSessionEngineControlSignal {
    NoSignal,
    Interrupt,
    Shutdown,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum DeferredCommandKey {
    PermissionMode,
    Model,
    ExpertModel(String),
    ExpertAllowedModels(String),
    ReasoningEffort,
}

fn deferred_command_key(command: &SessionEngineCommand) -> Option<DeferredCommandKey> {
    match command {
        SessionEngineCommand::SetPermissionMode(_) => Some(DeferredCommandKey::PermissionMode),
        SessionEngineCommand::SetModel(_) => Some(DeferredCommandKey::Model),
        SessionEngineCommand::SetExpertModel { agent_name, .. } => {
            Some(DeferredCommandKey::ExpertModel(agent_name.clone()))
        }
        SessionEngineCommand::SetExpertAllowedModels { agent_name, .. } => {
            Some(DeferredCommandKey::ExpertAllowedModels(agent_name.clone()))
        }
        SessionEngineCommand::SetReasoningEffort(_) => Some(DeferredCommandKey::ReasoningEffort),
        SessionEngineCommand::Prompt(_)
        | SessionEngineCommand::DelegateSubagent { .. }
        | SessionEngineCommand::BackgroundSubagentCompleted { .. }
        | SessionEngineCommand::ContinueSession
        | SessionEngineCommand::Compact
        | SessionEngineCommand::ShowHistoryTree
        | SessionEngineCommand::Undo
        | SessionEngineCommand::Redo
        | SessionEngineCommand::NavigateHistory { .. }
        | SessionEngineCommand::ViewChild { .. }
        | SessionEngineCommand::ViewParent
        | SessionEngineCommand::ToggleFastMode
        | SessionEngineCommand::AnchoredToggle
        | SessionEngineCommand::ResumeSession(_)
        | SessionEngineCommand::NewSession
        | SessionEngineCommand::ToggleMcpServer(_) => None,
        #[cfg(test)]
        SessionEngineCommand::InspectHistory(_) => None,
    }
}

pub(crate) fn enqueue_deferred_command(
    commands: &mut VecDeque<SessionEngineCommand>,
    command: SessionEngineCommand,
) {
    if let Some(key) = deferred_command_key(&command) {
        let batch_start = commands
            .iter()
            .rposition(|queued| deferred_command_key(queued).is_none())
            .map_or(0, |index| index + 1);
        if let Some(index) = commands
            .iter()
            .enumerate()
            .skip(batch_start)
            .rev()
            .find_map(|(index, queued)| {
                (deferred_command_key(queued).as_ref() == Some(&key)).then_some(index)
            })
        {
            commands.remove(index);
        }
    }
    commands.push_back(command);
}

pub(crate) fn park_active_turn_command(
    parked_commands: &mut VecDeque<SessionEngineCommand>,
    command: SessionEngineCommand,
    session_transport_tx: &mpsc::UnboundedSender<SessionTransportEvent>,
) {
    enqueue_deferred_command(parked_commands, command);
    let _ = session_transport_tx.send(SessionTransportEvent::Notice(NoticeEvent::info(
        "Change queued for after the current turn",
    )));
}

pub(crate) fn flush_parked_commands(
    deferred_commands: &mut VecDeque<SessionEngineCommand>,
    parked_commands: &mut VecDeque<SessionEngineCommand>,
) {
    let mut remaining = std::mem::take(deferred_commands);
    while let Some(command) = parked_commands.pop_front() {
        enqueue_deferred_command(deferred_commands, command);
    }
    while let Some(command) = remaining.pop_front() {
        enqueue_deferred_command(deferred_commands, command);
    }
}

fn drain_queued_session_controls(
    control_rx: &mut mpsc::UnboundedReceiver<SessionEngineControl>,
    deferred_commands: &mut VecDeque<SessionEngineCommand>,
) -> QueuedSessionEngineControlSignal {
    let mut interrupted = false;
    loop {
        match control_rx.try_recv() {
            Ok(SessionEngineControl::Command(command)) => {
                enqueue_deferred_command(deferred_commands, command)
            }
            Ok(SessionEngineControl::Interrupt) => interrupted = true,
            Ok(SessionEngineControl::Shutdown) | Err(mpsc::error::TryRecvError::Disconnected) => {
                return QueuedSessionEngineControlSignal::Shutdown;
            }
            Err(mpsc::error::TryRecvError::Empty) => {
                return if interrupted {
                    QueuedSessionEngineControlSignal::Interrupt
                } else {
                    QueuedSessionEngineControlSignal::NoSignal
                };
            }
        }
    }
}

pub(crate) async fn next_idle_session_command(
    control_rx: &mut mpsc::UnboundedReceiver<SessionEngineControl>,
    deferred_commands: &mut VecDeque<SessionEngineCommand>,
) -> Option<SessionEngineCommand> {
    loop {
        if !deferred_commands.is_empty() {
            match drain_queued_session_controls(control_rx, deferred_commands) {
                QueuedSessionEngineControlSignal::Shutdown => return None,
                // An idle interrupt is stale only when it appears before the
                // next command in the FIFO stream.
                QueuedSessionEngineControlSignal::Interrupt
                | QueuedSessionEngineControlSignal::NoSignal => {}
            }

            return deferred_commands.pop_front();
        }

        match control_rx.recv().await? {
            SessionEngineControl::Command(command) => return Some(command),
            SessionEngineControl::Interrupt => {}
            SessionEngineControl::Shutdown => return None,
        }
    }
}

pub(crate) async fn select_active_session_operation<T, F>(
    control_rx: &mut mpsc::UnboundedReceiver<SessionEngineControl>,
    deferred_commands: &mut VecDeque<SessionEngineCommand>,
    operation: Pin<&mut F>,
) -> ActiveSessionOperation<T>
where
    F: Future<Output = T> + ?Sized,
{
    select_active_session_operation_with_events(control_rx, deferred_commands, operation, None)
        .await
}

pub(crate) async fn select_active_session_operation_with_events<T, F>(
    control_rx: &mut mpsc::UnboundedReceiver<SessionEngineControl>,
    deferred_commands: &mut VecDeque<SessionEngineCommand>,
    mut operation: Pin<&mut F>,
    mut runner_event_rx: Option<&mut mpsc::UnboundedReceiver<SessionTransportEvent>>,
) -> ActiveSessionOperation<T>
where
    F: Future<Output = T> + ?Sized,
{
    loop {
        // Keep commands queued while looking through the FIFO stream for an
        // already-arrived interrupt. This retains cancellation priority for a
        // live operation without losing commands that preceded the interrupt.
        match drain_queued_session_controls(control_rx, deferred_commands) {
            QueuedSessionEngineControlSignal::Interrupt => {
                return ActiveSessionOperation::Interrupted;
            }
            QueuedSessionEngineControlSignal::Shutdown => return ActiveSessionOperation::Shutdown,
            QueuedSessionEngineControlSignal::NoSignal => {}
        }

        if let Some(command) = deferred_commands.pop_front() {
            return ActiveSessionOperation::Command(Some(command));
        }

        tokio::select! {
            biased;
            control = control_rx.recv() => match control {
                Some(SessionEngineControl::Interrupt) => {
                    return match drain_queued_session_controls(control_rx, deferred_commands) {
                        QueuedSessionEngineControlSignal::Shutdown => ActiveSessionOperation::Shutdown,
                        QueuedSessionEngineControlSignal::Interrupt | QueuedSessionEngineControlSignal::NoSignal => {
                            ActiveSessionOperation::Interrupted
                        }
                    };
                }
                Some(SessionEngineControl::Command(command)) => {
                    enqueue_deferred_command(deferred_commands, command)
                }
                Some(SessionEngineControl::Shutdown) | None => {
                    return ActiveSessionOperation::Shutdown;
                }
            },
            event = async {
                match runner_event_rx.as_deref_mut() {
                    Some(receiver) => receiver.recv().await,
                    None => std::future::pending().await,
                }
            } => match event {
                Some(event) => return ActiveSessionOperation::RunnerEvent(event),
                None => runner_event_rx = None,
            },
            result = operation.as_mut() => return ActiveSessionOperation::Completed(result),
        }
    }
}

/// Forwards all runner events that are already queued when its future settles,
/// except its private completion marker. The engine emits the sole public Done
/// after this drain.
pub(crate) fn forward_queued_runner_events(
    runner_event_rx: &mut mpsc::UnboundedReceiver<SessionTransportEvent>,
    session_transport_tx: &mpsc::UnboundedSender<SessionTransportEvent>,
) {
    loop {
        match runner_event_rx.try_recv() {
            Ok(SessionTransportEvent::Done) => {}
            Ok(event) => {
                let _ = session_transport_tx.send(event);
            }
            Err(mpsc::error::TryRecvError::Empty | mpsc::error::TryRecvError::Disconnected) => {
                break;
            }
        }
    }
}

pub(crate) async fn select_manual_compaction_operation<T, F>(
    control_rx: &mut mpsc::UnboundedReceiver<SessionEngineControl>,
    deferred_commands: &mut VecDeque<SessionEngineCommand>,
    mut operation: Pin<&mut F>,
) -> ManualCompactionOperation<T>
where
    F: Future<Output = T> + ?Sized,
{
    loop {
        match drain_queued_session_controls(control_rx, deferred_commands) {
            QueuedSessionEngineControlSignal::Interrupt => {
                return ManualCompactionOperation::Interrupted;
            }
            QueuedSessionEngineControlSignal::Shutdown => {
                return ManualCompactionOperation::Shutdown;
            }
            QueuedSessionEngineControlSignal::NoSignal => {}
        }

        tokio::select! {
            biased;
            control = control_rx.recv() => match control {
                Some(SessionEngineControl::Command(command)) => {
                    enqueue_deferred_command(deferred_commands, command)
                }
                Some(SessionEngineControl::Interrupt) => {
                    return match drain_queued_session_controls(control_rx, deferred_commands) {
                        QueuedSessionEngineControlSignal::Shutdown => ManualCompactionOperation::Shutdown,
                        QueuedSessionEngineControlSignal::Interrupt | QueuedSessionEngineControlSignal::NoSignal => {
                            ManualCompactionOperation::Interrupted
                        }
                    };
                }
                Some(SessionEngineControl::Shutdown) | None => return ManualCompactionOperation::Shutdown,
            },
            result = operation.as_mut() => return ManualCompactionOperation::Completed(result),
        }
    }
}

/// Polls the parent run until every running subagent has settled (its
/// completion teardown ran) or the bounded settle window expires.
///
/// Callers must signal `cancel_active()` before calling this. During the settle
/// window the parent run remains live and polled: this lets the inline
/// subagent future complete `complete_started_run`, record its cancelled
/// terminal state, and release its active slot. Commands received during this
/// wait are retained for the outer loop rather than being consumed or lost.
///
/// Returns whether shutdown was requested while waiting.
pub(crate) async fn wait_for_subagent_cancel_settle<T, F>(
    control_rx: &mut mpsc::UnboundedReceiver<SessionEngineControl>,
    deferred_commands: &mut VecDeque<SessionEngineCommand>,
    mut run: Pin<&mut F>,
    subagent_runtime: &SubagentPool,
) -> bool
where
    F: Future<Output = T> + ?Sized,
{
    let timeout = tokio::time::sleep(SUBAGENT_CANCEL_SETTLE_TIMEOUT);
    tokio::pin!(timeout);
    let mut cancel_tick = tokio::time::interval(SUBAGENT_CANCEL_SETTLE_POLL_INTERVAL);
    let mut shutdown = false;

    loop {
        if !subagent_runtime.is_running() {
            return shutdown;
        }

        tokio::select! {
            biased;
            _ = &mut timeout => {
                // Bounded fallback: the caller will drop the parent run, which
                // force-cancels any child that could not settle cooperatively.
                return shutdown;
            }
            control = control_rx.recv() => match control {
                Some(SessionEngineControl::Interrupt) => {
                    // Keep waiting; the first interrupt already initiated cancellation.
                }
                Some(SessionEngineControl::Shutdown) | None => {
                    shutdown = true;
                }
                Some(SessionEngineControl::Command(command)) => {
                    enqueue_deferred_command(deferred_commands, command);
                }
            },
            _ = cancel_tick.tick() => {
                // Re-signal in case the parent turn briefly recovered and
                // attempted to start another subagent during cancellation.
                subagent_runtime.cancel_active();
            }
            _ = run.as_mut() => {
                // The parent turn completed before the pool observed guard
                // release. Its future is settled, so no further child work can
                // be driven by this run.
                return shutdown;
            }
        }
    }
}
