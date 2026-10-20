//! Session engine command and control scheduling.

use super::*;

/// Backend-private command payload for the session executor.
///
/// This is intentionally crate-private: frontend code submits
/// [`SessionCommand`] through [`SessionEngineIngress`] instead.
#[derive(Debug)]
pub(crate) enum SessionEngineCommand {
    Prompt(crate::user_content::UserMessageSubmission),
    CancelQueuedPrompt(String),
    CancelPrompt(String),
    PeerMessage(crate::peer::PeerMessage),
    BackgroundSubagentCompleted {
        generation: Option<u64>,
        parent_session_id: String,
        parent_tool_call_id: Option<String>,
        result: Result<crate::subagent::SubagentRunSummary, String>,
    },
    SwarmMessageReady {
        root_session_id: String,
        generation: u64,
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
    SetExpertAllowedModels {
        agent_name: String,
        model_ids: Vec<String>,
    },
    ToggleFastMode,
    SetReasoningEffort(crate::request_builder::ModelReasoningEffort),
    SetFakeClient(Option<crate::fake::FakeClient>),
    ResumeSession(String),
    NewSession,
    ToggleMcpServer(String),
    #[cfg(test)]
    InspectHistory(tokio::sync::oneshot::Sender<Vec<crate::request_builder::HistoryItem>>),
}

impl SessionEngineCommand {
    pub(crate) fn from_session_command(command: SessionCommand) -> Self {
        match command {
            SessionCommand::SubmitPrompt(prompt) => Self::Prompt(prompt),
            SessionCommand::CancelQueuedPrompt(id) => Self::CancelQueuedPrompt(id),
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
            SessionCommand::SetExpertAllowedModels {
                agent_name,
                model_ids,
            } => Self::SetExpertAllowedModels {
                agent_name,
                model_ids,
            },
            SessionCommand::ToggleFastMode => Self::ToggleFastMode,
            SessionCommand::SetReasoningEffort(effort) => Self::SetReasoningEffort(effort),
            SessionCommand::SetFakeClient(client) => Self::SetFakeClient(client),
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
        SessionEngineCommand::ToggleFastMode => {
            Some(crate::session::SessionCommand::ToggleFastMode)
        }
        SessionEngineCommand::SetReasoningEffort(effort) => Some(
            crate::session::SessionCommand::SetReasoningEffort(effort.clone()),
        ),
        SessionEngineCommand::SetFakeClient(client) => {
            Some(crate::session::SessionCommand::SetFakeClient(*client))
        }
        SessionEngineCommand::ViewParent => Some(crate::session::SessionCommand::ViewParent),
        SessionEngineCommand::ViewChild {
            navigation,
            anchor_child_session_id,
        } => Some(crate::session::SessionCommand::ViewChild {
            navigation: navigation.clone(),
            anchor_child_session_id: anchor_child_session_id.clone(),
        }),
        SessionEngineCommand::BackgroundSubagentCompleted { .. }
        | SessionEngineCommand::ContinueSession => None,
        SessionEngineCommand::Compact => Some(crate::session::SessionCommand::Compact),
        SessionEngineCommand::SetModel(model) => {
            Some(crate::session::SessionCommand::SetModel(model.clone()))
        }
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
        SessionEngineCommand::CancelQueuedPrompt(id) => Some(
            crate::session::SessionCommand::CancelQueuedPrompt(id.clone()),
        ),
        SessionEngineCommand::Prompt(_)
        | SessionEngineCommand::CancelPrompt(_)
        | SessionEngineCommand::PeerMessage(_)
        | SessionEngineCommand::SwarmMessageReady { .. } => None,
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
    session_engine_command_as_session_command(command).filter(|command| command.schedule().idle)
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
    transcript: &Arc<StdMutex<TranscriptRecorder>>,
    subagent_runtime: &SubagentPool,
    background_deliveries: &mut std::collections::HashMap<String, BackgroundDelivery>,
    change_notice_sent: &mut bool,
) {
    if let SessionEngineCommand::BackgroundSubagentCompleted {
        generation,
        parent_session_id,
        parent_tool_call_id,
        result,
    } = &command
    {
        if *generation != subagent_runtime.swarm_generation(parent_session_id) {
            return;
        }
        // Only the model delivery waits for this operation to finish.
        observe_background_subagent_completion(
            transcript,
            subagent_runtime,
            background_deliveries,
            session_transport_tx,
            parent_session_id,
            parent_tool_call_id.clone(),
            result,
        );
        enqueue_deferred_command(parked_commands, command);
        return;
    }

    // Reading a transcript does not touch the agent, but history navigation
    // rewrites it, so it waits for the turn in every phase that parks commands.
    match &command {
        SessionEngineCommand::Undo | SessionEngineCommand::Redo => {
            let _ = session_transport_tx.send(SessionTransportEvent::Notice(
                NoticeEvent::info("history navigation is unavailable while a turn is active")
                    .with_message_id(NoticeId::HistoryNavigationUnavailable),
            ));
            return;
        }
        SessionEngineCommand::ShowHistoryTree | SessionEngineCommand::NavigateHistory { .. } => {
            let _ = session_transport_tx.send(SessionTransportEvent::Error(ErrorEvent::new(
                "history navigation is unavailable while a turn is active",
            )));
            return;
        }
        _ => {}
    }

    let disposition = session_engine_command_as_session_command(&command)
        .map(|command| command.active_turn_disposition())
        .unwrap_or(crate::session::ActiveTurnCommandDisposition::Defer);
    match disposition {
        crate::session::ActiveTurnCommandDisposition::Defer => {
            park_active_turn_command(
                parked_commands,
                command,
                session_transport_tx,
                change_notice_sent,
            );
        }
        crate::session::ActiveTurnCommandDisposition::Reject => {
            if matches!(command, SessionEngineCommand::Compact) {
                let _ = session_transport_tx.send(SessionTransportEvent::ManualCompactionStarted);
            }
            let _ = session_transport_tx.send(SessionTransportEvent::Notice(
                NoticeEvent::info("Turn still running").with_message_id(NoticeId::TurnRunning),
            ));
        }
        crate::session::ActiveTurnCommandDisposition::QueuePrompt
        | crate::session::ActiveTurnCommandDisposition::Immediate
        | crate::session::ActiveTurnCommandDisposition::Interrupt => {
            unreachable!("active turn command is handled before the deferred fallback")
        }
    }
}

/// Transcript navigation request carried out while compaction is still running.
pub(crate) enum ManualCompactionNavigation {
    ViewChild {
        navigation: crate::command::ChildNavigation,
        anchor_child_session_id: Option<String>,
    },
    ViewParent,
}

pub(crate) enum ManualCompactionOperation<T> {
    Interrupted,
    Shutdown,
    Completed(T),
    /// Serviced while the compaction future stays pinned: reading a transcript
    /// does not touch the agent, so manual compaction no longer has to finish
    /// before the child view can be opened.
    Navigation(ManualCompactionNavigation),
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
    ExpertAllowedModels(String),
    ReasoningEffort,
    FakeClient,
}

fn deferred_command_key(command: &SessionEngineCommand) -> Option<DeferredCommandKey> {
    match command {
        SessionEngineCommand::SetPermissionMode(_) => Some(DeferredCommandKey::PermissionMode),
        SessionEngineCommand::SetModel(_) => Some(DeferredCommandKey::Model),
        SessionEngineCommand::SetExpertAllowedModels { agent_name, .. } => {
            Some(DeferredCommandKey::ExpertAllowedModels(agent_name.clone()))
        }
        SessionEngineCommand::SetReasoningEffort(_) => Some(DeferredCommandKey::ReasoningEffort),
        SessionEngineCommand::SetFakeClient(_) => Some(DeferredCommandKey::FakeClient),
        SessionEngineCommand::Prompt(_)
        | SessionEngineCommand::CancelQueuedPrompt(_)
        | SessionEngineCommand::CancelPrompt(_)
        | SessionEngineCommand::PeerMessage(_)
        | SessionEngineCommand::BackgroundSubagentCompleted { .. }
        | SessionEngineCommand::SwarmMessageReady { .. }
        | SessionEngineCommand::ContinueSession
        | SessionEngineCommand::Compact
        | SessionEngineCommand::ShowHistoryTree
        | SessionEngineCommand::Undo
        | SessionEngineCommand::Redo
        | SessionEngineCommand::NavigateHistory { .. }
        | SessionEngineCommand::ViewChild { .. }
        | SessionEngineCommand::ViewParent
        | SessionEngineCommand::ToggleFastMode
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
    if matches!(command, SessionEngineCommand::CancelPrompt(_)) {
        commands.push_front(command);
        return;
    }
    if let SessionEngineCommand::CancelQueuedPrompt(id) = &command {
        cancel_queued_prompt(commands, id);
    }
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

pub(super) fn cancel_queued_prompt(
    commands: &mut VecDeque<SessionEngineCommand>,
    id: &str,
) -> bool {
    let before = commands.len();
    commands.retain(
        |command| !matches!(command, SessionEngineCommand::Prompt(prompt) if prompt.id == id),
    );
    commands.len() != before
}

pub(super) fn discard_queued_peer_messages(commands: &mut VecDeque<SessionEngineCommand>) {
    commands.retain(|command| !matches!(command, SessionEngineCommand::PeerMessage(_)));
}

pub(crate) fn park_active_turn_command(
    parked_commands: &mut VecDeque<SessionEngineCommand>,
    command: SessionEngineCommand,
    session_transport_tx: &mpsc::UnboundedSender<SessionTransportEvent>,
    change_notice_sent: &mut bool,
) {
    let first_parked = !*change_notice_sent;
    enqueue_deferred_command(parked_commands, command);
    if first_parked {
        *change_notice_sent = true;
        let _ = session_transport_tx.send(SessionTransportEvent::Notice(
            NoticeEvent::info("Change queued for after the current turn")
                .with_message_id(NoticeId::ChangeQueued),
        ));
    }
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
            Ok(SessionEngineControl::Interrupt) => {
                discard_queued_peer_messages(deferred_commands);
                interrupted = true;
            }
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

#[derive(Debug)]
pub(crate) enum IdleSessionOperation {
    Command(SessionEngineCommand),
    Interrupt,
}

pub(crate) async fn next_idle_session_operation(
    control_rx: &mut mpsc::UnboundedReceiver<SessionEngineControl>,
    deferred_commands: &mut VecDeque<SessionEngineCommand>,
) -> Option<IdleSessionOperation> {
    if deferred_commands.is_empty() {
        match control_rx.recv().await? {
            SessionEngineControl::Command(command @ SessionEngineCommand::PeerMessage(_)) => {
                enqueue_deferred_command(deferred_commands, command);
            }
            SessionEngineControl::Command(command) => {
                return Some(IdleSessionOperation::Command(command));
            }
            SessionEngineControl::Interrupt => return Some(IdleSessionOperation::Interrupt),
            SessionEngineControl::Shutdown => return None,
        }
    }
    match drain_queued_session_controls(control_rx, deferred_commands) {
        QueuedSessionEngineControlSignal::Shutdown => return None,
        QueuedSessionEngineControlSignal::Interrupt => {
            return Some(IdleSessionOperation::Interrupt);
        }
        QueuedSessionEngineControlSignal::NoSignal => {}
    }
    deferred_commands
        .pop_front()
        .map(IdleSessionOperation::Command)
}

#[cfg(test)]
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
                    discard_queued_peer_messages(deferred_commands);
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

/// Transcript navigation stays available while manual compaction runs. These are
/// exactly the commands [`crate::session::SessionCommand::active_turn_disposition`]
/// marks `Immediate`; everything else keeps waiting for the compaction outcome.
fn manual_compaction_navigation(
    command: &SessionEngineCommand,
) -> Option<ManualCompactionNavigation> {
    match command {
        SessionEngineCommand::ViewChild {
            navigation,
            anchor_child_session_id,
        } => Some(ManualCompactionNavigation::ViewChild {
            navigation: navigation.clone(),
            anchor_child_session_id: anchor_child_session_id.clone(),
        }),
        SessionEngineCommand::ViewParent => Some(ManualCompactionNavigation::ViewParent),
        _ => None,
    }
}

struct DrainedManualCompactionControls {
    shutdown: bool,
    interrupted: bool,
    navigation: Option<ManualCompactionNavigation>,
}

/// Same FIFO deferral as [`drain_queued_session_controls`], except that queued
/// navigation is returned instead of parked, so a request that arrived before
/// the loop was entered is still serviced during compaction.
fn drain_manual_compaction_controls(
    control_rx: &mut mpsc::UnboundedReceiver<SessionEngineControl>,
    deferred_commands: &mut VecDeque<SessionEngineCommand>,
) -> DrainedManualCompactionControls {
    let mut drained = DrainedManualCompactionControls {
        shutdown: false,
        interrupted: false,
        navigation: None,
    };
    loop {
        match control_rx.try_recv() {
            Ok(SessionEngineControl::Command(command)) => {
                match manual_compaction_navigation(&command) {
                    Some(navigation) if drained.navigation.is_none() => {
                        drained.navigation = Some(navigation)
                    }
                    _ => enqueue_deferred_command(deferred_commands, command),
                }
            }
            Ok(SessionEngineControl::Interrupt) => {
                discard_queued_peer_messages(deferred_commands);
                drained.interrupted = true;
            }
            Ok(SessionEngineControl::Shutdown) | Err(mpsc::error::TryRecvError::Disconnected) => {
                drained.shutdown = true;
                return drained;
            }
            Err(mpsc::error::TryRecvError::Empty) => return drained,
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
        let drained = drain_manual_compaction_controls(control_rx, deferred_commands);
        if drained.shutdown {
            return ManualCompactionOperation::Shutdown;
        }
        if drained.interrupted {
            return ManualCompactionOperation::Interrupted;
        }
        if let Some(navigation) = drained.navigation {
            return ManualCompactionOperation::Navigation(navigation);
        }

        tokio::select! {
            biased;
            control = control_rx.recv() => match control {
                Some(SessionEngineControl::Command(command)) => {
                    match manual_compaction_navigation(&command) {
                        Some(navigation) => {
                            return ManualCompactionOperation::Navigation(navigation)
                        }
                        None => enqueue_deferred_command(deferred_commands, command),
                    }
                }
                Some(SessionEngineControl::Interrupt) => {
                    discard_queued_peer_messages(deferred_commands);
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

pub(crate) struct SubagentCancelSettle {
    pub(crate) shutdown: bool,
    pub(crate) settled: bool,
    pub(crate) remaining_run_ids: Vec<String>,
}

pub(crate) async fn wait_for_subagent_cancel_settle<T, F>(
    control_rx: &mut mpsc::UnboundedReceiver<SessionEngineControl>,
    deferred_commands: &mut VecDeque<SessionEngineCommand>,
    mut run: Pin<&mut F>,
    subagent_runtime: &SubagentPool,
    settle_timeout: Duration,
) -> SubagentCancelSettle
where
    F: Future<Output = T> + ?Sized,
{
    let timeout = tokio::time::sleep(settle_timeout);
    tokio::pin!(timeout);
    let mut cancel_tick = tokio::time::interval(SUBAGENT_CANCEL_SETTLE_POLL_INTERVAL);
    let mut shutdown = false;
    let mut run_completed = false;
    let mut control_open = true;

    loop {
        if !subagent_runtime.is_running() {
            return SubagentCancelSettle {
                shutdown,
                settled: true,
                remaining_run_ids: Vec::new(),
            };
        }

        tokio::select! {
            biased;
            _ = &mut timeout => {
                return SubagentCancelSettle {
                    shutdown, settled: !subagent_runtime.is_running(),
                    remaining_run_ids: subagent_runtime.active_jobs().into_iter().map(|job| job.run_id).collect(),
                };
            }
            control = control_rx.recv(), if control_open => match control {
                Some(SessionEngineControl::Interrupt) => {
                    discard_queued_peer_messages(deferred_commands);
                }
                Some(SessionEngineControl::Shutdown) => shutdown = true,
                None => {
                    control_open = false;
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
            _ = run.as_mut(), if !run_completed => run_completed = true,
        }
    }
}

#[cfg(test)]
#[path = "../../../tests/internal/session/engine/control/tests.rs"]
mod tests;
