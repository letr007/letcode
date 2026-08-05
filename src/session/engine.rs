//! Typed transport handles for the incremental session-engine boundary.
//!
//! The engine owns command ingress and event egress. During the staged
//! migration, the TUI runner still owns agent and transcript lifetimes while it
//! consumes crate-private transitional endpoints.

use std::fmt;

use notify::{Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use tokio::sync::mpsc;

use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::{
    Arc, Mutex as StdMutex,
    atomic::{AtomicBool, Ordering},
};
use std::time::Duration;

use anyhow::{Result, anyhow};
use async_openai::config::Config;
use serde_json::json;
use tokio::task::JoinHandle;

use crate::agent::{
    Agent, AgentEvent, ConfiguredPrimaryRouteFactory, ManualCompactionOutcome, PrimaryRouteFactory,
    SubagentInvocation,
};
use crate::agent_event_journal::persist_agent_event;
use crate::config::{AppConfig, ModelRoute, ProviderConfig, RetryConfig};
use crate::mcp;
use crate::runtime_context::RuntimeActiveContext;
use crate::session::{
    AgentRunner, ErrorEvent, NoticeEvent, RuntimeContextDisposition, RuntimeContextUpdatedEvent,
    SessionCommand, SessionEvent, SessionTransportEvent, TokenUsageEvent, session_started_event,
    unfinished_current_active_turn_tool_calls,
};
use crate::subagent::SubagentPool;
use crate::tool::{ToolHandler, normalize_subagent_input};
use crate::transcript::{
    ROOT_CONTEXT_BRANCH_ID, TranscriptEvent, TranscriptRecorder, read_records,
    read_records_allow_partial_tail, remove_empty_session_file, sync_recorder_branch,
    transcript_projection,
};

/// Maximum time to keep polling the parent run after signalling subagent
/// cancellation so the subagent's completion teardown (cancelled terminal
/// record, guard release) can run to completion. Bounded so a stuck subagent
/// cannot block the engine forever.
const SUBAGENT_CANCEL_SETTLE_TIMEOUT: Duration = Duration::from_secs(3);

/// Poll interval while waiting for subagent cancellation to settle. Keeps the
/// parent run polled (so the in-flight subagent future can settle) without
/// busy-spinning.
const SUBAGENT_CANCEL_SETTLE_POLL_INTERVAL: Duration = Duration::from_millis(50);

/// Error returned when the session engine no longer accepts frontend input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SessionEngineIngressError;

impl fmt::Display for SessionEngineIngressError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("session engine is no longer available")
    }
}

impl std::error::Error for SessionEngineIngressError {}

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
    ToggleFastMode,
    SetReasoningEffort(crate::request_builder::ModelReasoningEffort),
    ResumeSession(String),
    NewSession,
    ToggleMcpServer(String),
    #[cfg(test)]
    InspectHistory(tokio::sync::oneshot::Sender<Vec<crate::request_builder::HistoryItem>>),
}

impl SessionEngineCommand {
    fn from_session_command(command: SessionCommand) -> Self {
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
            SessionCommand::ToggleFastMode => Self::ToggleFastMode,
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

/// Frontend-owned handle for submitting session commands and lifecycle intent.
#[derive(Clone, Debug)]
pub struct SessionEngineIngress {
    control_tx: mpsc::UnboundedSender<SessionEngineControl>,
}

impl SessionEngineIngress {
    /// Submit a frontend-neutral session command.
    pub fn submit(&self, command: SessionCommand) -> Result<(), SessionEngineIngressError> {
        if matches!(command, SessionCommand::Interrupt) {
            return self.request_interrupt();
        }
        self.send_control(SessionEngineControl::Command(
            SessionEngineCommand::from_session_command(command),
        ))
    }

    /// Request cancellation without frontend-specific execution metadata.
    pub fn request_interrupt(&self) -> Result<(), SessionEngineIngressError> {
        self.send_control(SessionEngineControl::Interrupt)
    }

    /// Request orderly session-engine shutdown.
    pub fn shutdown(&self) -> Result<(), SessionEngineIngressError> {
        self.send_control(SessionEngineControl::Shutdown)
    }

    fn send_control(&self, control: SessionEngineControl) -> Result<(), SessionEngineIngressError> {
        self.control_tx
            .send(control)
            .map_err(|_| SessionEngineIngressError)
    }

    #[cfg(test)]
    pub(crate) fn submit_transitional(
        &self,
        command: SessionEngineCommand,
    ) -> Result<(), SessionEngineIngressError> {
        self.send_control(SessionEngineControl::Command(command))
    }
}

/// Frontend-owned event stream emitted by the session engine.
pub(crate) struct SessionEngineEventEgress {
    event_rx: mpsc::UnboundedReceiver<SessionTransportEvent>,
}

impl SessionEngineEventEgress {
    pub(crate) fn into_receiver(self) -> mpsc::UnboundedReceiver<SessionTransportEvent> {
        self.event_rx
    }
}

/// Session-owned boundary between frontend intent and backend execution.
///
/// A started engine owns the agent, transcript, MCP discovery, and execution
/// loop. The frontend receives only its command ingress and event egress.
pub struct SessionEngine {
    #[cfg(test)]
    control_rx: Option<mpsc::UnboundedReceiver<SessionEngineControl>>,
    #[cfg(test)]
    event_tx: Option<mpsc::UnboundedSender<SessionTransportEvent>>,
    ingress: Option<SessionEngineIngress>,
    event_rx: Option<mpsc::UnboundedReceiver<SessionTransportEvent>>,
    engine_task: Option<JoinHandle<()>>,
    mcp_discovery_task: Option<JoinHandle<()>>,
    reload_watcher: Option<RecommendedWatcher>,
    transcript: Option<Arc<StdMutex<TranscriptRecorder>>>,
}

/// Backend-only startup settings for an interactive session engine.
#[derive(Debug, Clone)]
pub struct SessionEngineConfig {
    pub sessions_dir: PathBuf,
    /// Routes keyed by their provider-qualified display name (`provider/model`).
    pub model_routes: indexmap::IndexMap<String, ModelRoute>,
    /// Whether each route has a non-empty credential configured.
    pub route_api_key_configured: indexmap::IndexMap<String, bool>,
    /// Expert routes keyed by role name; roles without an entry use the active primary route.
    pub expert_model_routes: indexmap::IndexMap<String, ModelRoute>,
    /// Legacy model-only expert assignments keyed by role name. Their provider
    /// follows successful primary-route changes while their model id is retained.
    pub legacy_expert_models: indexmap::IndexMap<String, String>,
    /// Provider catalog used to reconstruct expert route factories after configuration updates.
    pub providers: indexmap::IndexMap<String, crate::config::ProviderConfig>,
    pub global_retry: crate::config::RetryConfig,
    /// Provider-specific API-key remediation keyed by provider name.
    pub provider_api_key_hints: indexmap::IndexMap<String, String>,
    /// Fallback remediation if a provider-specific hint is unavailable.
    pub api_key_hint: String,
    pub mcp_config_path: PathBuf,
    pub mcp_config: indexmap::IndexMap<String, crate::config::McpServerConfig>,
}

/// Initial presentation data projected while the engine takes ownership.
#[derive(Debug, Clone)]
pub struct SessionEngineProjection {
    pub session_id: String,
    pub session_title: Option<String>,
    pub model_id: String,
    pub model_label: String,
    pub permission_mode_label: String,
    pub fast_mode_enabled: bool,
    pub api_key_configured: bool,
}

impl SessionEngine {
    #[cfg(test)]
    pub(crate) fn new() -> (Self, SessionEngineIngress, SessionEngineEventEgress) {
        let (control_tx, control_rx) = mpsc::unbounded_channel();
        let (event_tx, event_rx) = mpsc::unbounded_channel();
        let ingress = SessionEngineIngress { control_tx };
        (
            Self {
                control_rx: Some(control_rx),
                event_tx: Some(event_tx),
                ingress: None,
                event_rx: None,
                engine_task: None,
                mcp_discovery_task: None,
                reload_watcher: None,
                transcript: None,
            },
            ingress,
            SessionEngineEventEgress { event_rx },
        )
    }

    /// Start the backend control loop and transfer all execution resources into it.
    pub fn start(
        agent: Agent<async_openai::config::OpenAIConfig>,
        transcript: Arc<StdMutex<TranscriptRecorder>>,
        model_label: String,
        config: SessionEngineConfig,
    ) -> Result<(Self, SessionEngineProjection)> {
        // `main` synchronizes the initial scope before startup; subsequent session
        // switches are synchronized by this engine's control loop.
        let model_id = agent.route_display_name();
        let api_key_configured = route_has_api_key(&config.route_api_key_configured, &model_id);
        let permission_mode_label = agent.permission_mode().to_string();
        let fast_mode_enabled = agent.fast_mode_enabled();
        let (session_id, session_title) = initial_session_metadata(&transcript)?;
        let projection = SessionEngineProjection {
            session_id,
            session_title,
            model_id,
            model_label,
            permission_mode_label,
            fast_mode_enabled,
            api_key_configured,
        };
        let (control_tx, control_rx) = mpsc::unbounded_channel();
        let (event_tx, event_rx) = mpsc::unbounded_channel();
        let title_event_tx = event_tx.clone();
        let ingress = SessionEngineIngress { control_tx };
        let (reload_tx, reload_rx) = mpsc::unbounded_channel();
        let config_path = config.mcp_config_path.clone();
        let reload_watcher = create_config_watcher(&config_path, reload_tx)?;
        let (mcp_tools_tx, mcp_tools_rx) = mpsc::unbounded_channel();
        let discovery_config = config.mcp_config.clone();
        let mcp_discovery_task = tokio::spawn(async move {
            let result = mcp::discover_servers(&discovery_config).await;
            let _ = mcp_tools_tx.send(result);
        });
        let subagent_runtime = SubagentPool::new();
        let task = tokio::spawn(run_engine_loop(
            agent,
            Arc::clone(&transcript),
            config.sessions_dir,
            config.model_routes,
            config.route_api_key_configured,
            config.expert_model_routes,
            config.legacy_expert_models,
            config.providers,
            config.global_retry,
            config.provider_api_key_hints,
            config.api_key_hint,
            config.mcp_config_path,
            config.mcp_config,
            mcp_tools_rx,
            reload_rx,
            control_rx,
            event_tx.clone(),
            title_event_tx,
            subagent_runtime,
        ));
        Ok((
            Self {
                #[cfg(test)]
                control_rx: None,
                #[cfg(test)]
                event_tx: None,
                ingress: Some(ingress),
                event_rx: Some(event_rx),
                engine_task: Some(task),
                mcp_discovery_task: Some(mcp_discovery_task),
                reload_watcher: Some(reload_watcher),
                transcript: Some(transcript),
            },
            projection,
        ))
    }

    /// Transfer the frontend command ingress to the TUI.
    pub fn take_ingress(&mut self) -> SessionEngineIngress {
        self.ingress
            .take()
            .expect("session engine command ingress already taken")
    }

    pub(crate) fn take_event_egress(&mut self) -> SessionEngineEventEgress {
        SessionEngineEventEgress {
            event_rx: self
                .event_rx
                .take()
                .expect("session engine event egress already taken"),
        }
    }

    /// Request backend termination while retaining ownership for a later join.
    #[cfg(test)]
    pub fn request_shutdown(&self) -> Result<(), SessionEngineIngressError> {
        self.ingress
            .as_ref()
            .ok_or(SessionEngineIngressError)?
            .shutdown()
    }

    /// Join backend-owned tasks and run transcript cleanup.
    ///
    /// Cleanup is attempted even if either task panics. Any join or cleanup
    /// failure is returned after all owned resources have been reconciled.
    pub async fn join(mut self) -> Result<()> {
        let mut failure = None;
        // Stop filesystem callbacks before waiting for the engine and discovery
        // tasks so shutdown cannot enqueue work into a finished session.
        self.reload_watcher.take();

        if let Some(task) = self.engine_task.take() {
            if let Err(error) = task.await {
                failure = Some(anyhow!("session engine task failed: {error}"));
            }
        }
        if let Some(task) = self.mcp_discovery_task.take() {
            if !task.is_finished() {
                task.abort();
            }
            if let Err(error) = task.await {
                if !error.is_cancelled() && failure.is_none() {
                    failure = Some(anyhow!("MCP discovery task failed: {error}"));
                }
            }
        }
        if let Some(transcript) = self.transcript.take() {
            let cleanup = (|| -> Result<()> {
                let path = transcript
                    .lock()
                    .map_err(|_| anyhow!("transcript recorder poisoned"))?
                    .path()
                    .to_path_buf();
                remove_empty_session_file(path).map(|_| ())
            })();
            if let Err(error) = cleanup {
                if failure.is_none() {
                    failure = Some(error);
                }
            }
        }

        match failure {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    /// Compatibility lifecycle convenience for callers that have not yet
    /// separated shutdown request from joining.
    #[cfg(test)]
    pub async fn shutdown(self) -> Result<()> {
        self.request_shutdown()?;
        self.join().await
    }

    #[cfg(test)]
    pub(crate) async fn recv_control(&mut self) -> Option<SessionEngineControl> {
        self.control_rx
            .as_mut()
            .expect("test engine control receiver unavailable")
            .recv()
            .await
    }

    #[cfg(test)]
    pub(crate) fn try_recv_control(
        &mut self,
    ) -> Result<SessionEngineControl, mpsc::error::TryRecvError> {
        self.control_rx
            .as_mut()
            .expect("test engine control receiver unavailable")
            .try_recv()
    }

    #[cfg(test)]
    pub(crate) fn event_sender(&self) -> mpsc::UnboundedSender<SessionTransportEvent> {
        self.event_tx
            .as_ref()
            .expect("test engine event sender unavailable")
            .clone()
    }

    /// Transfer internal control and event endpoints to the session executor.
    #[cfg(test)]
    pub(crate) fn into_session_executor_parts(
        self,
    ) -> (
        mpsc::UnboundedReceiver<SessionEngineControl>,
        mpsc::UnboundedSender<SessionTransportEvent>,
    ) {
        (
            self.control_rx
                .expect("test engine control receiver unavailable"),
            self.event_tx.expect("test engine event sender unavailable"),
        )
    }
}

/// Map private session transport commands that the session coordinator owns as idle work.
fn session_engine_command_as_idle_session_command(
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
        SessionEngineCommand::ViewParent => Some(crate::session::SessionCommand::ViewParent),
        SessionEngineCommand::ViewChild {
            navigation,
            anchor_child_session_id,
        } => Some(crate::session::SessionCommand::ViewChild {
            navigation: *navigation,
            anchor_child_session_id: anchor_child_session_id.clone(),
        }),
        SessionEngineCommand::Prompt(_)
        | SessionEngineCommand::DelegateSubagent { .. }
        | SessionEngineCommand::Compact
        | SessionEngineCommand::SetModel(_)
        | SessionEngineCommand::SetExpertModel { .. }
        | SessionEngineCommand::ResumeSession(_)
        | SessionEngineCommand::NewSession
        | SessionEngineCommand::ToggleMcpServer(_) => None,
        #[cfg(test)]
        SessionEngineCommand::InspectHistory(_) => None,
    }
}

pub(crate) enum ActiveSessionOperation<T> {
    Interrupted,
    Shutdown,
    Completed(T),
    RunnerEvent(SessionTransportEvent),
    Command(Option<SessionEngineCommand>),
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

fn drain_queued_session_controls(
    control_rx: &mut mpsc::UnboundedReceiver<SessionEngineControl>,
    deferred_commands: &mut VecDeque<SessionEngineCommand>,
) -> QueuedSessionEngineControlSignal {
    let mut interrupted = false;
    loop {
        match control_rx.try_recv() {
            Ok(SessionEngineControl::Command(command)) => deferred_commands.push_back(command),
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
                Some(SessionEngineControl::Command(command)) => deferred_commands.push_back(command),
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
                Some(SessionEngineControl::Command(command)) => deferred_commands.push_back(command),
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
                    deferred_commands.push_back(command);
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

fn apply_config_reload(
    agent: &mut Agent<async_openai::config::OpenAIConfig>,
    config_path: &std::path::Path,
    model_routes: &mut indexmap::IndexMap<String, ModelRoute>,
    route_api_key_configured: &mut indexmap::IndexMap<String, bool>,
    expert_model_routes: &mut indexmap::IndexMap<String, ModelRoute>,
    legacy_expert_models: &mut indexmap::IndexMap<String, String>,
    providers: &mut indexmap::IndexMap<String, ProviderConfig>,
    global_retry: &mut RetryConfig,
    provider_api_key_hints: &mut indexmap::IndexMap<String, String>,
    event_tx: &mpsc::UnboundedSender<SessionTransportEvent>,
) -> Result<()> {
    let config = AppConfig::load_from_path(config_path)?;
    let previous_active_route = agent
        .primary_route()
        .cloned()
        .ok_or_else(|| anyhow!("active agent route is unavailable during configuration reload"))?;
    let active_route = config.active_route();
    let provider = config.provider_for_route(&active_route)?;
    let primary_factory = ConfiguredPrimaryRouteFactory::new(
        config.providers.clone(),
        config.global.retry.clone(),
    );
    let prepared = primary_factory.prepare_route(active_route.clone())?;

    let next_model_routes = config
        .providers
        .iter()
        .flat_map(|(provider_name, provider)| {
            provider.models.keys().map(move |model| {
                let route = ModelRoute::new(provider_name, model);
                (route.display_name(), route)
            })
        })
        .collect::<indexmap::IndexMap<_, _>>();
    let next_route_api_key_configured = config
        .providers
        .iter()
        .flat_map(|(provider_name, provider)| {
            provider.models.keys().map(move |model| {
                let route = ModelRoute::new(provider_name, model);
                (route.display_name(), !provider.api_key.trim().is_empty())
            })
        })
        .collect::<indexmap::IndexMap<_, _>>();
    let next_expert_model_routes = crate::delegation::supported_agent_names()
        .filter_map(|name| {
            config
                .model_route_for(name)
                .cloned()
                .map(|route| (name.to_string(), route))
        })
        .collect::<indexmap::IndexMap<_, _>>();
    let next_legacy_expert_models = crate::delegation::supported_agent_names()
        .filter(|name| config.agents.follows_active_provider(name))
        .filter_map(|name| {
            config
                .model_route_for(name)
                .map(|route| (name.to_string(), route.model.clone()))
        })
        .collect::<indexmap::IndexMap<_, _>>();
    let next_provider_api_key_hints = config
        .providers
        .keys()
        .map(|name| {
            (
                name.clone(),
                format!(
                    "Set providers.{name}.api_key in {} or set {}.",
                    config.config_path.display(),
                    crate::config::provider_api_key_env_var(name)
                ),
            )
        })
        .collect::<indexmap::IndexMap<_, _>>();
    let expert_factory = crate::subagent::ExpertRouteFactory::new(
        next_expert_model_routes
            .iter()
            .map(|(name, route)| (name.clone(), route.clone())),
        &config.providers,
        &config.global.retry,
    )?;
    let next_providers = config.providers.clone();
    let next_global_retry = config.global.retry.clone();
    let next_agent_retry = provider
        .retry
        .clone()
        .unwrap_or_else(|| next_global_retry.clone());
    let next_parallelism = config
        .tools
        .parallelism
        .iter()
        .map(|(name, mode)| (name.clone(), *mode))
        .collect::<std::collections::BTreeMap<_, _>>();

    let previous_expert_routes = crate::delegation::supported_agent_names()
        .map(|name| {
            (
                name.to_string(),
                effective_expert_route(
                    expert_model_routes,
                    legacy_expert_models,
                    &previous_active_route,
                    name,
                ),
            )
        })
        .collect::<indexmap::IndexMap<_, _>>();
    let next_expert_routes = crate::delegation::supported_agent_names()
        .map(|name| {
            (
                name.to_string(),
                effective_expert_route(
                    &next_expert_model_routes,
                    &next_legacy_expert_models,
                    &active_route,
                    name,
                ),
            )
        })
        .collect::<indexmap::IndexMap<_, _>>();

    let providers_runtime_unchanged = providers_runtime_eq(providers, &next_providers);
    let maps_unchanged = *model_routes == next_model_routes
        && *route_api_key_configured == next_route_api_key_configured
        && *expert_model_routes == next_expert_model_routes
        && *legacy_expert_models == next_legacy_expert_models
        && *provider_api_key_hints == next_provider_api_key_hints
        && *global_retry == next_global_retry;
    let settings_unchanged = agent.compaction_config() == &config.global.compaction
        && agent.tool_timeout_secs() == config.global.tool_timeout_secs
        && agent.retry_config() == &next_agent_retry
        && agent.tool_parallelism_overrides() == &next_parallelism;
    let route_unchanged = previous_active_route == active_route;
    let previous_provider = providers.get(&previous_active_route.provider);
    let client_unchanged = route_unchanged
        && previous_provider.is_some_and(|previous| {
            previous.api_key == provider.api_key
                && previous.base_url == provider.base_url
                && previous.protocol == provider.protocol
        });
    let next_model_protocols = provider
        .models
        .iter()
        .map(|(id, model)| (id.clone(), model.protocol))
        .collect::<HashMap<_, _>>();
    let next_model_catalog = provider
        .models
        .iter()
        .map(|(id, model)| (id.clone(), model.request_metadata()))
        .collect::<HashMap<_, _>>();
    let catalog_unchanged = agent.default_protocol() == provider.protocol
        && agent.model_protocols() == &next_model_protocols
        && agent.model_catalog() == &next_model_catalog;
    let expert_routes_unchanged = previous_expert_routes == next_expert_routes;

    // Self-writes (model/fast-mode/MCP persist) and duplicate watcher events often
    // land here with no reloadable runtime delta — stay silent and keep usage anchors.
    if providers_runtime_unchanged
        && maps_unchanged
        && settings_unchanged
        && client_unchanged
        && catalog_unchanged
        && expert_routes_unchanged
    {
        return Ok(());
    }

    // Fallible mutator first; remaining updates below are infallible.
    agent.set_tool_parallelism(next_parallelism)?;
    if agent.compaction_config() != &config.global.compaction {
        agent.set_compaction_config(config.global.compaction.clone());
    }
    if agent.tool_timeout_secs() != config.global.tool_timeout_secs {
        agent.set_tool_timeout_secs(config.global.tool_timeout_secs);
    }
    if agent.retry_config() != &next_agent_retry {
        agent.set_retry_config(next_agent_retry);
    }
    agent.set_primary_route_factory(Arc::new(primary_factory));
    agent.set_subagent_child_factory(Arc::new(expert_factory));
    if !client_unchanged {
        // Rebuilding the client/route clears provider usage anchors intentionally.
        prepared.into_install().apply(agent);
    } else if !catalog_unchanged {
        agent.set_default_protocol(provider.protocol);
        agent.set_model_protocols(next_model_protocols);
        agent.set_model_catalog(next_model_catalog);
    }

    if !route_unchanged {
        let _ = event_tx.send(SessionTransportEvent::ModelChanged {
            model_id: active_route.display_name(),
        });
    }
    *model_routes = next_model_routes;
    *route_api_key_configured = next_route_api_key_configured;
    *expert_model_routes = next_expert_model_routes;
    *legacy_expert_models = next_legacy_expert_models;
    *provider_api_key_hints = next_provider_api_key_hints;
    *providers = next_providers;
    *global_retry = next_global_retry;
    for (agent_name, route) in next_expert_routes {
        if previous_expert_routes.get(&agent_name) != Some(&route) {
            let _ = event_tx.send(SessionTransportEvent::ExpertModelChanged {
                agent_name,
                model_id: route.display_name(),
            });
        }
    }
    let _ = event_tx.send(SessionTransportEvent::Notice(NoticeEvent::info(
        "configuration reloaded (supported runtime fields only; MCP, permissions, Fast Mode, max_iterations/max_tool_calls unchanged)",
    )));
    Ok(())
}

/// Compare reloadable provider fields, ignoring `default_model` which is often
/// rewritten by in-session model switches that already updated the live agent.
fn providers_runtime_eq(
    left: &indexmap::IndexMap<String, ProviderConfig>,
    right: &indexmap::IndexMap<String, ProviderConfig>,
) -> bool {
    if left.len() != right.len() {
        return false;
    }
    left.iter().all(|(name, left_provider)| {
        right.get(name).is_some_and(|right_provider| {
            left_provider.base_url == right_provider.base_url
                && left_provider.api_key == right_provider.api_key
                && left_provider.protocol == right_provider.protocol
                && left_provider.retry == right_provider.retry
                && left_provider.models == right_provider.models
        })
    })
}

fn create_config_watcher(
    config_path: &std::path::Path,
    reload_tx: mpsc::UnboundedSender<()>,
) -> Result<RecommendedWatcher> {
    let target = std::fs::canonicalize(config_path).unwrap_or_else(|_| config_path.to_path_buf());
    let watch_dir = target
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."))
        .to_path_buf();
    let mut watcher = notify::recommended_watcher(move |event: notify::Result<Event>| {
        let Ok(event) = event else {
            // Transient watcher errors should not force a reload storm.
            return;
        };
        if matches!(
            event.kind,
            EventKind::Create(_) | EventKind::Modify(_) | EventKind::Remove(_)
        ) && event
            .paths
            .iter()
            .any(|path| path.file_name() == target.file_name())
        {
            let _ = reload_tx.send(());
        }
    })?;
    watcher.watch(&watch_dir, RecursiveMode::NonRecursive)?;
    Ok(watcher)
}

fn route_has_api_key(
    route_api_key_configured: &indexmap::IndexMap<String, bool>,
    route_display_name: &str,
) -> bool {
    route_api_key_configured
        .get(route_display_name)
        .copied()
        .unwrap_or(false)
}

fn route_api_key_hint(
    route_display_name: &str,
    provider_api_key_hints: &indexmap::IndexMap<String, String>,
    fallback_hint: &str,
) -> String {
    let provider = route_display_name
        .split_once('/')
        .map(|(provider, _)| provider)
        .unwrap_or("selected");
    provider_api_key_hints
        .get(provider)
        .cloned()
        .unwrap_or_else(|| fallback_hint.to_string())
}

fn active_route_has_api_key(
    agent: &Agent<async_openai::config::OpenAIConfig>,
    route_api_key_configured: &indexmap::IndexMap<String, bool>,
) -> bool {
    route_has_api_key(route_api_key_configured, &agent.route_display_name())
}

fn effective_expert_route(
    expert_model_routes: &indexmap::IndexMap<String, ModelRoute>,
    legacy_expert_models: &indexmap::IndexMap<String, String>,
    primary_route: &ModelRoute,
    agent_name: &str,
) -> ModelRoute {
    expert_model_routes
        .get(agent_name)
        .cloned()
        .or_else(|| {
            legacy_expert_models
                .get(agent_name)
                .map(|model| ModelRoute::new(primary_route.provider.clone(), model))
        })
        .unwrap_or_else(|| primary_route.clone())
}

fn expert_routes_after_primary_switch(
    expert_model_routes: &indexmap::IndexMap<String, ModelRoute>,
    legacy_expert_models: &indexmap::IndexMap<String, String>,
    primary_route: &ModelRoute,
) -> indexmap::IndexMap<String, ModelRoute> {
    let mut routes = expert_model_routes.clone();
    for (agent_name, model) in legacy_expert_models {
        routes.insert(
            agent_name.clone(),
            ModelRoute::new(primary_route.provider.clone(), model.clone()),
        );
    }
    routes
}

fn delegated_route_display_name(
    agent: &Agent<async_openai::config::OpenAIConfig>,
    expert_model_routes: &indexmap::IndexMap<String, ModelRoute>,
    agent_name: &str,
) -> String {
    expert_model_routes
        .get(agent_name)
        .map_or_else(|| agent.route_display_name(), ModelRoute::display_name)
}

fn delegated_route_for_takeover(
    agent: &Agent<async_openai::config::OpenAIConfig>,
    expert_model_routes: &indexmap::IndexMap<String, ModelRoute>,
    sessions_dir: &std::path::Path,
    parent_transcript: &Arc<StdMutex<TranscriptRecorder>>,
    agent_name: &str,
    target_child_session_id: Option<&str>,
) -> Result<String> {
    let Some(target_child_session_id) = target_child_session_id else {
        return Ok(delegated_route_display_name(
            agent,
            expert_model_routes,
            agent_name,
        ));
    };
    let parent_records = parent_transcript
        .lock()
        .map_err(|_| anyhow::anyhow!("transcript recorder poisoned"))
        .and_then(|recorder| read_records(recorder.path()).map_err(Into::into))?;
    let child = crate::subagent::SubagentPool::child_sessions(sessions_dir, &parent_records)
        .into_iter()
        .find(|child| child.child_session_id == target_child_session_id)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "takeover failed: child_session_id `{target_child_session_id}` is not a known child of this parent"
            )
        })?;
    if child.agent_name != agent_name {
        anyhow::bail!(
            "takeover failed: child `{target_child_session_id}` is agent `{}`, expected `{agent_name}`",
            child.agent_name
        );
    }
    let child_records = read_records_allow_partial_tail(
        crate::transcript::child_sessions_dir(sessions_dir)
            .join(format!("{target_child_session_id}.jsonl")),
    )?;
    crate::transcript::restore_latest_model(&child_records).ok_or_else(|| {
        anyhow::anyhow!(
            "takeover failed: child `{target_child_session_id}` has no recorded model route"
        )
    })
}

fn missing_api_key_error(api_key_hint: &str) -> ErrorEvent {
    ErrorEvent::new(format!(
        "API key is not set for the selected provider. {}",
        api_key_hint
    ))
}

fn send_missing_api_key_error(
    session_transport_tx: &mpsc::UnboundedSender<SessionTransportEvent>,
    route_display_name: &str,
    provider_api_key_hints: &indexmap::IndexMap<String, String>,
    api_key_hint: &str,
) {
    let hint = route_api_key_hint(route_display_name, provider_api_key_hints, api_key_hint);
    let _ = session_transport_tx.send(SessionTransportEvent::Error(missing_api_key_error(&hint)));
    let _ = session_transport_tx.send(SessionTransportEvent::Done);
}

fn current_runtime_context(
    transcript: &Arc<StdMutex<TranscriptRecorder>>,
) -> Result<RuntimeActiveContext> {
    let (session_id, records, branch_id) = {
        let recorder = transcript
            .lock()
            .map_err(|_| anyhow::anyhow!("transcript recorder poisoned"))?;
        (
            recorder.session_id().to_string(),
            read_records(recorder.path())?,
            recorder
                .current_context_branch_id()
                .unwrap_or(crate::transcript::ROOT_CONTEXT_BRANCH_ID)
                .to_string(),
        )
    };
    runtime_context_from_records(&records, &session_id, Some(&branch_id))
}

fn record_manual_compaction_error(
    transcript: &Arc<StdMutex<TranscriptRecorder>>,
    message: String,
) -> ErrorEvent {
    let message = match transcript
        .lock()
        .map_err(|_| anyhow!("transcript recorder poisoned"))
        .and_then(|mut recorder| recorder.record_error(message.clone()))
    {
        Ok(()) => message,
        Err(error) => {
            format!("{message} (additionally failed to record transcript error: {error})")
        }
    };
    ErrorEvent::new(message)
}

fn runtime_context_from_records(
    records: &[crate::transcript::TranscriptRecord],
    session_id: &str,
    branch_id: Option<&str>,
) -> Result<RuntimeActiveContext> {
    let snapshot = transcript_projection::project_runtime_restore_snapshot(
        session_id.to_string(),
        records.to_vec(),
        transcript_projection::SessionContextCursor {
            branch_id: branch_id.map(str::to_string),
            leaf_sequence: None,
        },
        &[],
    )?
    .snapshot;
    RuntimeActiveContext::try_from(&snapshot)
}

fn sessions_dir_for_transcript(transcript: &Arc<StdMutex<TranscriptRecorder>>) -> Result<PathBuf> {
    let recorder = transcript
        .lock()
        .map_err(|_| anyhow::anyhow!("transcript recorder poisoned"))?;
    recorder
        .path()
        .parent()
        .map(ToOwned::to_owned)
        .ok_or_else(|| anyhow::anyhow!("transcript path has no parent directory"))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct InterruptRequest {
    pub(crate) parent_tool_calls: Vec<(String, String)>,
    pub(crate) visible_child_session_id: Option<String>,
}

pub(crate) fn derive_interrupt_request(
    transcript: &Arc<StdMutex<TranscriptRecorder>>,
    subagent_runtime: &SubagentPool,
) -> InterruptRequest {
    let active_child_session_id = subagent_runtime
        .active_child()
        .map(|child| child.child_session_id);
    let parent_tool_calls = unfinished_current_active_turn_tool_calls(transcript);

    InterruptRequest {
        parent_tool_calls,
        visible_child_session_id: active_child_session_id,
    }
}

pub(crate) fn send_subagent_interrupted(
    session_transport_tx: &mpsc::UnboundedSender<SessionTransportEvent>,
    child_session_id: Option<String>,
) {
    if let Some(child_session_id) = child_session_id {
        let _ = session_transport_tx.send(SessionTransportEvent::ChildSessionEvent {
            child_session_id,
            agent_name: None,
            parent_tool_call_id: None,
            event: SessionEvent::Interrupted,
        });
    }
    let _ = session_transport_tx.send(SessionTransportEvent::Interrupted);
}

pub(crate) fn record_interrupt_transcript(
    transcript: &Arc<StdMutex<TranscriptRecorder>>,
    interrupt: &InterruptRequest,
) {
    let mut recorder = match transcript.lock() {
        Ok(recorder) => recorder,
        Err(_) => return,
    };
    let cursor = transcript_projection::SessionContextCursor {
        branch_id: Some(
            recorder
                .current_context_branch_id()
                .unwrap_or(ROOT_CONTEXT_BRANCH_ID)
                .to_string(),
        ),
        leaf_sequence: None,
    };

    let turn_id = match read_records(recorder.path()).and_then(|records| {
        transcript_projection::active_turn_id_at_context_cursor(records, cursor)
    }) {
        Ok(turn_id) => turn_id,
        Err(_) => return,
    };

    for (call_id, name) in &interrupt.parent_tool_calls {
        let _ = recorder.record_tool_call_cancelled(call_id.clone(), name.clone());
    }

    if let Some(turn_id) = turn_id {
        let _ = recorder.record_turn_interrupted(Some(turn_id));
    }
}

pub(crate) fn rehydrate_agent_from_transcript<C>(
    agent: &mut Agent<C>,
    transcript: &Arc<StdMutex<TranscriptRecorder>>,
) -> Result<()>
where
    C: Config,
{
    let path = transcript
        .lock()
        .map_err(|_| anyhow::anyhow!("transcript recorder poisoned"))?
        .path()
        .to_path_buf();
    let records = read_records(&path)?;
    let session_id = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .ok_or_else(|| anyhow::anyhow!("invalid transcript path: {}", path.display()))?
        .to_string();
    let snapshot = crate::session::project_runtime_restore_snapshot_with_children(
        &session_id,
        records,
        transcript_projection::SessionContextCursor {
            branch_id: None,
            leaf_sequence: None,
        },
        &sessions_dir_for_transcript(transcript)?,
    )?;
    let branch_id = snapshot.branch_id.clone();
    let max_turn_id = snapshot.max_turn_id;
    let mut recorder = transcript
        .lock()
        .map_err(|_| anyhow::anyhow!("transcript recorder poisoned"))?;
    agent.restore_runtime_snapshot(snapshot.protocol_frames, snapshot.snapshot)?;
    agent.restore_turn_sequence(max_turn_id);
    sync_recorder_branch(&mut recorder, &branch_id);
    Ok(())
}

pub(crate) fn manual_compaction_session_token_usage<C>(agent: &Agent<C>) -> Result<TokenUsageEvent>
where
    C: Config,
{
    let usage = agent.session_token_usage()?;
    Ok(TokenUsageEvent::with_breakdown(
        usage.used_tokens,
        usage.context_window_tokens,
        usage.input_tokens,
        0,
        0,
    ))
}

pub(crate) async fn run_manual_compaction<C>(
    agent: &mut Agent<C>,
    transcript: &Arc<StdMutex<TranscriptRecorder>>,
    session_transport_tx: &mpsc::UnboundedSender<SessionTransportEvent>,
    control_rx: &mut mpsc::UnboundedReceiver<SessionEngineControl>,
    deferred_commands: &mut VecDeque<SessionEngineCommand>,
) -> bool
where
    C: Config + Clone,
{
    let transcript = Arc::clone(transcript);
    let snapshot_transcript = Arc::clone(&transcript);
    agent.set_runtime_snapshot_provider(Arc::new(move || {
        let transcript = snapshot_transcript
            .lock()
            .map_err(|_| anyhow::anyhow!("transcript recorder poisoned"))?;
        let records = read_records(transcript.path())?;
        Ok(
            crate::transcript::transcript_projection::project_runtime_restore_snapshot(
                transcript.session_id().to_string(),
                records,
                crate::transcript::transcript_projection::SessionContextCursor {
                    branch_id: transcript.current_context_branch_id().map(str::to_string),
                    leaf_sequence: None,
                },
                &[],
            )?
            .snapshot,
        )
    }));
    let event_transcript = Arc::clone(&transcript);
    let event_session_transport_tx = session_transport_tx.clone();
    // Persistence is the compaction transaction boundary. A cancellation that
    // arrives after it must retain the record.
    let compaction_persisted = Arc::new(AtomicBool::new(false));
    let event_compaction_persisted = Arc::clone(&compaction_persisted);
    let on_event = move |event| {
        let transcript = Arc::clone(&event_transcript);
        let session_transport_tx = event_session_transport_tx.clone();
        let compaction_persisted = Arc::clone(&event_compaction_persisted);
        async move {
            match event {
                AgentEvent::ContextCompactionStarted { .. } => {
                    let _ = session_transport_tx.send(SessionTransportEvent::CompactionStarted);
                }
                AgentEvent::ContextCompactionNoProgress(no_progress) => {
                    let _ =
                        session_transport_tx.send(SessionTransportEvent::CompactionNoProgress {
                            blockers: no_progress
                                .blockers
                                .into_iter()
                                .map(|blocker| blocker.label().to_string())
                                .collect(),
                        });
                }
                AgentEvent::ContextCompactionFailed { .. } => {
                    let _ = session_transport_tx.send(SessionTransportEvent::CompactionFailed);
                }
                AgentEvent::ContextCompacted(event) => {
                    let summary = event.summary.clone();
                    let mut recorder = transcript
                        .lock()
                        .map_err(|_| anyhow::anyhow!("transcript recorder poisoned"))?;
                    persist_agent_event(&mut recorder, &AgentEvent::ContextCompacted(event))?;
                    drop(recorder);
                    compaction_persisted.store(true, Ordering::Release);
                    // Persistence acknowledges the transaction. A closed
                    // presentation channel cannot roll it back.
                    let _ = session_transport_tx.send(SessionTransportEvent::CompactionCommitted {
                        summary: Some(summary),
                    });
                }
                AgentEvent::ContextCompactionDelta { delta } => {
                    let _ = session_transport_tx
                        .send(SessionTransportEvent::CompactionPreviewDelta { delta });
                }
                _ => {}
            }
            Ok(())
        }
    };
    let mut on_start = || Ok(());
    let mut on_delta = |_delta: &str| Ok(());
    // Drop the compaction future before reporting cancellation so a late
    // durable acknowledgement from a cancelled attempt cannot reach the UI.
    let compaction_result = {
        let compact = agent.compact_session_stream_async(on_event, &mut on_start, &mut on_delta);
        tokio::pin!(compact);
        select_manual_compaction_operation(control_rx, deferred_commands, compact.as_mut()).await
    };

    let shutdown = matches!(compaction_result, ManualCompactionOperation::Shutdown);
    match compaction_result {
        ManualCompactionOperation::Interrupted | ManualCompactionOperation::Shutdown => {
            // Manual compaction is not a model turn: do not write
            // TurnInterrupted. Restore the mutable agent from durable state so
            // the next command starts cleanly.
            let rehydrated = match rehydrate_agent_from_transcript(agent, &transcript) {
                Ok(()) => true,
                Err(error) => {
                    let _ =
                        session_transport_tx.send(SessionTransportEvent::Error(ErrorEvent::new(
                            format!("failed to restore cancelled compaction context: {error}"),
                        )));
                    false
                }
            };
            if compaction_persisted.load(Ordering::Acquire) {
                // The durable callback won before cancellation. The candidate
                // may not have been installed in memory yet, so rehydration is
                // authoritative.
                if rehydrated {
                    match manual_compaction_session_token_usage(agent) {
                        Ok(token_usage) => {
                            let _ = session_transport_tx
                                .send(SessionTransportEvent::SessionTokenUsage(token_usage));
                        }
                        Err(error) => {
                            let _ = session_transport_tx.send(SessionTransportEvent::Error(
                                ErrorEvent::new(format!(
                                    "failed to refresh committed compacted token usage: {error}"
                                )),
                            ));
                        }
                    }
                }
                match current_runtime_context(&transcript) {
                    Ok(context) => {
                        let _ = session_transport_tx.send(
                            SessionTransportEvent::RuntimeContextUpdated(
                                RuntimeContextUpdatedEvent {
                                    context,
                                    disposition: RuntimeContextDisposition::Advance,
                                },
                            ),
                        );
                    }
                    Err(error) => {
                        let _ = session_transport_tx.send(SessionTransportEvent::Error(
                            ErrorEvent::new(format!(
                                "failed to refresh committed compacted context: {error}"
                            )),
                        ));
                    }
                }
            } else {
                let _ = session_transport_tx.send(SessionTransportEvent::CompactionFailed);
            }
        }
        ManualCompactionOperation::Completed(Ok(ManualCompactionOutcome::Compacted { .. })) => {
            match manual_compaction_session_token_usage(agent) {
                Ok(token_usage) => {
                    let _ = session_transport_tx
                        .send(SessionTransportEvent::SessionTokenUsage(token_usage));
                }
                Err(error) => {
                    let event = record_manual_compaction_error(
                        &transcript,
                        format!("failed to refresh compacted token usage: {error}"),
                    );
                    let _ = session_transport_tx.send(SessionTransportEvent::Error(event));
                }
            }
            match current_runtime_context(&transcript) {
                Ok(context) => {
                    let _ = session_transport_tx.send(
                        SessionTransportEvent::RuntimeContextUpdated(RuntimeContextUpdatedEvent {
                            context,
                            disposition: RuntimeContextDisposition::Advance,
                        }),
                    );
                }
                Err(error) => {
                    let event = record_manual_compaction_error(
                        &transcript,
                        format!("failed to refresh compacted context: {error}"),
                    );
                    let _ = session_transport_tx.send(SessionTransportEvent::Error(event));
                }
            }
        }
        ManualCompactionOperation::Completed(Ok(ManualCompactionOutcome::NoProgress(_))) => {}
        ManualCompactionOperation::Completed(Err(error)) => {
            let event = record_manual_compaction_error(
                &transcript,
                format!("failed to compact context: {error}"),
            );
            let _ = session_transport_tx.send(SessionTransportEvent::Error(event));
        }
    }

    let _ = session_transport_tx.send(SessionTransportEvent::Done);
    shutdown
}

fn session_title_from_records(records: &[crate::transcript::TranscriptRecord]) -> Option<String> {
    records.iter().rev().find_map(|record| match &record.event {
        TranscriptEvent::SessionTitle { title } => Some(title.clone()),
        _ => None,
    })
}

pub(crate) fn initial_session_metadata(
    transcript: &Arc<StdMutex<TranscriptRecorder>>,
) -> Result<(String, Option<String>)> {
    let recorder = transcript
        .lock()
        .map_err(|_| anyhow::anyhow!("transcript recorder poisoned"))?;
    Ok((
        recorder.session_id().to_string(),
        session_title_from_records(&read_records(recorder.path())?),
    ))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct VisibleChildViewState {
    record_count: usize,
    index: usize,
    total: usize,
}

async fn refresh_visible_child_session_view(
    transcript: &Arc<StdMutex<TranscriptRecorder>>,
    session_transport_tx: &mpsc::UnboundedSender<SessionTransportEvent>,
    sessions_dir: &std::path::Path,
    visible_child_session_id: &mut Option<String>,
    visible_child_view_state: &mut Option<VisibleChildViewState>,
) {
    let Some(child_session_id) = visible_child_session_id.as_deref() else {
        return;
    };
    let records = match crate::transcript::read_child_session_records_allow_partial_tail(
        sessions_dir,
        child_session_id,
    ) {
        Ok(records) => records,
        Err(error) => {
            let _ = session_transport_tx.send(SessionTransportEvent::Error(ErrorEvent::new(
                format!("failed to refresh child transcript: {error}"),
            )));
            return;
        }
    };

    let (parent_session_id, parent_records) =
        match crate::session::current_session_records(transcript) {
            Ok(current) => current,
            Err(error) => {
                let _ = session_transport_tx.send(SessionTransportEvent::Error(ErrorEvent::new(
                    format!("failed to refresh child transcript: {error}"),
                )));
                return;
            }
        };
    let children = crate::session::list_child_sessions_for_view(sessions_dir, &parent_records);
    let Some((index, child)) = children
        .iter()
        .enumerate()
        .find(|(_, child)| child.child_session_id == child_session_id)
    else {
        return;
    };
    let view_state = VisibleChildViewState {
        record_count: records.len(),
        index,
        total: children.len(),
    };
    if visible_child_view_state.is_some_and(|state| state == view_state) {
        return;
    }

    let runtime_context = match runtime_context_from_records(&records, child_session_id, None) {
        Ok(context) => context,
        Err(error) => {
            let _ = session_transport_tx.send(SessionTransportEvent::Error(ErrorEvent::new(
                format!("failed to refresh child transcript context: {error}"),
            )));
            return;
        }
    };
    *visible_child_view_state = Some(view_state);
    let _ = session_transport_tx.send(SessionTransportEvent::ChildSessionViewed {
        parent_session_id,
        child_session_id: child.child_session_id.clone(),
        agent_name: child.agent_name.clone(),
        index,
        total: children.len(),
        pool_ordinal: child.pool_ordinal,
        records,
        runtime_context,
    });
}

#[allow(clippy::too_many_arguments)]
async fn run_engine_loop(
    agent: Agent<async_openai::config::OpenAIConfig>,
    transcript: Arc<StdMutex<TranscriptRecorder>>,
    sessions_dir: PathBuf,
    model_routes: indexmap::IndexMap<String, ModelRoute>,
    route_api_key_configured: indexmap::IndexMap<String, bool>,
    expert_model_routes: indexmap::IndexMap<String, ModelRoute>,
    legacy_expert_models: indexmap::IndexMap<String, String>,
    providers: indexmap::IndexMap<String, crate::config::ProviderConfig>,
    global_retry: crate::config::RetryConfig,
    provider_api_key_hints: indexmap::IndexMap<String, String>,
    api_key_hint: String,
    mcp_config_path: PathBuf,
    mcp_config: indexmap::IndexMap<String, crate::config::McpServerConfig>,
    mcp_tools_rx: mpsc::UnboundedReceiver<Vec<mcp::McpServerDiscovery>>,
    mut reload_rx: mpsc::UnboundedReceiver<()>,
    mut control_rx: mpsc::UnboundedReceiver<SessionEngineControl>,
    session_transport_tx: mpsc::UnboundedSender<SessionTransportEvent>,
    title_event_tx: mpsc::UnboundedSender<SessionTransportEvent>,
    subagent_runtime: SubagentPool,
) {
    let transcript = transcript;
    let mut agent = agent;
    let mut mcp_tools_rx = Some(mcp_tools_rx);
    let mut mcp_config = mcp_config;
    let mut model_routes = model_routes;
    let mut route_api_key_configured = route_api_key_configured;
    let mut expert_model_routes = expert_model_routes;
    let mut legacy_expert_models = legacy_expert_models;
    let mut providers = providers;
    let mut global_retry = global_retry;
    let mut provider_api_key_hints = provider_api_key_hints;
    let mut mcp_registered_tools: HashMap<String, Vec<String>> = HashMap::new();
    let subagent_runtime = subagent_runtime;
    let mut deferred_commands = VecDeque::new();
    let mut visible_child_session_id = None;
    let mut visible_child_view_state = None;
    let mut child_refresh = tokio::time::interval(std::time::Duration::from_millis(250));
    child_refresh.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    loop {
        tokio::select! {
            biased;
            reload = reload_rx.recv() => {
                if reload.is_none() {
                    break;
                }
                while reload_rx.try_recv().is_ok() {}
                if let Err(error) = apply_config_reload(
                    &mut agent,
                    &mcp_config_path,
                    &mut model_routes,
                    &mut route_api_key_configured,
                    &mut expert_model_routes,
                    &mut legacy_expert_models,
                    &mut providers,
                    &mut global_retry,
                    &mut provider_api_key_hints,
                    &session_transport_tx,
                ) {
                    let _ = session_transport_tx.send(SessionTransportEvent::Error(ErrorEvent::new(
                        format!("failed to reload configuration: {error}"),
                    )));
                }
            }
            command = next_idle_session_command(&mut control_rx, &mut deferred_commands) => {
                let Some(command) = command else {
                    break;
                };

                if let SessionEngineCommand::SetExpertModel {
                    agent_name,
                    model_id,
                } = &command
                {
                    if !crate::delegation::supported_agent_names().any(|name| name == agent_name) {
                        let _ = session_transport_tx.send(SessionTransportEvent::Error(
                            ErrorEvent::new(format!("unknown expert: {agent_name}")),
                        ));
                        continue;
                    }
                    let Some(route) = model_routes.get(model_id).cloned() else {
                        let _ = session_transport_tx.send(SessionTransportEvent::Error(
                            ErrorEvent::new(format!("unknown model: {model_id}")),
                        ));
                        continue;
                    };
                    if let Err(error) = crate::config::persist_expert_model_route(
                        &mcp_config_path,
                        agent_name,
                        &route,
                    ) {
                        let _ = session_transport_tx.send(SessionTransportEvent::Error(
                            ErrorEvent::new(format!("failed to set expert model: {error}")),
                        ));
                        continue;
                    }
                    if let Err(error) = apply_config_reload(
                        &mut agent,
                        &mcp_config_path,
                        &mut model_routes,
                        &mut route_api_key_configured,
                        &mut expert_model_routes,
                        &mut legacy_expert_models,
                        &mut providers,
                        &mut global_retry,
                        &mut provider_api_key_hints,
                        &session_transport_tx,
                    ) {
                        let _ = session_transport_tx.send(SessionTransportEvent::Error(
                            ErrorEvent::new(format!("failed to reload configuration: {error}")),
                        ));
                    }
                    continue;
                }

                if let SessionEngineCommand::SetModel(model) = &command {
                    let Some(route) = model_routes.get(model).cloned() else {
                        let _ = session_transport_tx.send(SessionTransportEvent::Error(
                            ErrorEvent::new(format!("unknown model: {model}")),
                        ));
                        continue;
                    };
                    let model_id = route.display_name();
                    let prepared_route = match agent.prepare_primary_route(route.clone()) {
                        Ok(prepared_route) => prepared_route,
                        Err(error) => {
                            let _ = session_transport_tx.send(SessionTransportEvent::Error(
                                ErrorEvent::new(format!("failed to set model: {error}")),
                            ));
                            continue;
                        }
                    };
                    match crate::session::persist_and_apply_model_route_with(
                        &mut agent,
                        &transcript,
                        route.clone(),
                        prepared_route,
                        |route| crate::config::persist_primary_model_route(&mcp_config_path, route),
                    ) {
                        Ok(fast_mode_auto_disabled) => {
                            if fast_mode_auto_disabled {
                                let _ = session_transport_tx.send(
                                    SessionTransportEvent::FastModeChanged { enabled: false },
                                );
                                let _ = session_transport_tx.send(SessionTransportEvent::Notice(
                                    NoticeEvent::info(
                                        "Fast mode auto-disabled: current model is unavailable",
                                    ),
                                ));
                            }
                            let updated_expert_model_routes = expert_routes_after_primary_switch(
                                &expert_model_routes,
                                &legacy_expert_models,
                                &route,
                            );
                            let factory = crate::subagent::ExpertRouteFactory::new(
                                updated_expert_model_routes
                                    .iter()
                                    .map(|(name, route)| (name.clone(), route.clone())),
                                &providers,
                                &global_retry,
                            )
                            .expect("configured expert routes remain constructible after a primary switch");
                            agent.set_subagent_child_factory(Arc::new(factory));
                            expert_model_routes = updated_expert_model_routes;
                            let _ = session_transport_tx.send(SessionTransportEvent::ModelChanged {
                                model_id: model_id.clone(),
                            });
                            if !route_has_api_key(&route_api_key_configured, &model_id) {
                                let _ = session_transport_tx.send(SessionTransportEvent::Error(
                                    missing_api_key_error(&route_api_key_hint(
                                        &model_id,
                                        &provider_api_key_hints,
                                        &api_key_hint,
                                    )),
                                ));
                            }
                        }
                        Err(error) => {
                            if error.fast_mode_auto_disabled() {
                                let _ = session_transport_tx.send(
                                    SessionTransportEvent::FastModeChanged { enabled: false },
                                );
                                let _ = session_transport_tx.send(SessionTransportEvent::Notice(
                                    NoticeEvent::info(
                                        "Fast mode auto-disabled: current model is unavailable",
                                    ),
                                ));
                            }
                            let _ = session_transport_tx.send(SessionTransportEvent::Error(
                                ErrorEvent::new(format!("failed to set model: {error}")),
                            ));
                        }
                    }
                    continue;
                }

                if let Some(session_command) = session_engine_command_as_idle_session_command(&command) {
                    if let SessionEngineCommand::ViewChild { .. } = &command {
                        visible_child_session_id = crate::session::SessionCoordinator::emit_view_child(
                            &transcript,
                            &session_transport_tx,
                            Some(sessions_dir.as_path()),
                            match &command {
                                SessionEngineCommand::ViewChild { navigation, .. } => *navigation,
                                _ => unreachable!("view-child command was matched above"),
                            },
                            match &command {
                                SessionEngineCommand::ViewChild { anchor_child_session_id, .. } => {
                                    anchor_child_session_id.as_deref()
                                }
                                _ => unreachable!("view-child command was matched above"),
                            },
                        );
                        visible_child_view_state = None;
                    } else {
                        let _ = crate::session::SessionCoordinator::dispatch_idle_command(
                            session_command,
                            &mut agent,
                            &transcript,
                            &session_transport_tx,
                            Some(sessions_dir.as_path()),
                        );
                        if matches!(command, SessionEngineCommand::ViewParent) {
                            visible_child_session_id = None;
                            visible_child_view_state = None;
                        }
                    }
                    continue;
                }

                let prompt = match command {
                    SessionEngineCommand::ToggleMcpServer(server_name) => {
                        let Some(server_config) = mcp_config.get(&server_name).cloned() else {
                            let _ = session_transport_tx.send(SessionTransportEvent::McpServerUpdating {
                                name: server_name.clone(),
                                updating: false,
                            });
                            let _ = session_transport_tx.send(SessionTransportEvent::McpDiagnostic(format!(
                                "MCP server '{server_name}' is no longer configured"
                            )));
                            continue;
                        };
                        let enabled = !server_config.enabled;
                        let persisted_config = match crate::config::persist_mcp_server_enabled(
                            &mcp_config_path,
                            &server_name,
                            enabled,
                        ) {
                            Ok(config) => config,
                            Err(error) => {
                                let _ = session_transport_tx.send(SessionTransportEvent::McpServerUpdating {
                                    name: server_name.clone(),
                                    updating: false,
                                });
                                let _ = session_transport_tx.send(SessionTransportEvent::McpDiagnostic(format!(
                                    "failed to persist MCP server '{server_name}': {error}"
                                )));
                                continue;
                            }
                        };
                        mcp_config.insert(server_name.clone(), persisted_config);
                        if !enabled {
                            for tool_name in mcp_registered_tools
                                .remove(&server_name)
                                .unwrap_or_default()
                            {
                                agent.unregister_tool(&tool_name);
                            }
                            let _ = session_transport_tx.send(SessionTransportEvent::McpServerUpdated(
                                mcp::McpServerCatalogEntry {
                                    name: server_name,
                                    enabled: false,
                                    status: mcp::McpServerStatus::Disabled,
                                },
                            ));
                            continue;
                        }

                        let mut one_server = indexmap::IndexMap::new();
                        one_server.insert(
                            server_name.clone(),
                            mcp_config
                                .get(&server_name)
                                .expect("configured MCP server should remain present")
                                .clone(),
                        );
                        let discovery = mcp::discover_servers(&one_server)
                            .await
                            .into_iter()
                            .next()
                            .expect("single MCP server discovery should return one result");
                        let mut server = discovery.server;
                        let mut catalog_tools = Vec::new();
                        match server.status {
                            mcp::McpServerStatus::Online { .. } => {
                                let mut registered = Vec::new();
                                for tool in discovery.tools {
                                    let tool_name = tool.name().to_string();
                                    let catalog_entry = tool.catalog_entry();
                                    if let Err(error) = agent.try_register_tool(tool) {
                                        let _ = session_transport_tx.send(SessionTransportEvent::McpDiagnostic(format!(
                                            "failed to register MCP tool '{tool_name}': {error}"
                                        )));
                                    } else {
                                        registered.push(tool_name);
                                        catalog_tools.push(catalog_entry);
                                    }
                                }
                                server.status = mcp::McpServerStatus::Online {
                                    tool_count: registered.len(),
                                };
                                mcp_registered_tools.insert(server_name, registered);
                            }
                            mcp::McpServerStatus::Offline { ref message } => {
                                let _ = session_transport_tx.send(SessionTransportEvent::McpDiagnostic(format!(
                                    "MCP server '{}' is offline: {message}",
                                    server.name
                                )));
                            }
                            mcp::McpServerStatus::Disabled => unreachable!("enabled server was discovered"),
                        }
                        let _ = session_transport_tx.send(SessionTransportEvent::McpServerToolsUpdated {
                            name: server.name.clone(),
                            tools: catalog_tools,
                        });
                        let _ = session_transport_tx.send(SessionTransportEvent::McpServerUpdated(server));
                        continue;
                    }
                    SessionEngineCommand::Prompt(prompt) => prompt,
                    SessionEngineCommand::ShowHistoryTree
                    | SessionEngineCommand::Undo
                    | SessionEngineCommand::Redo
                    | SessionEngineCommand::NavigateHistory { .. }
                    | SessionEngineCommand::SetPermissionMode(_)
                    | SessionEngineCommand::SetModel(_)
                    | SessionEngineCommand::SetExpertModel { .. }
                    | SessionEngineCommand::ToggleFastMode
                    | SessionEngineCommand::SetReasoningEffort(_)
                    | SessionEngineCommand::ViewChild { .. }
                    | SessionEngineCommand::ViewParent => {
                        // Idle commands are handled above via SessionCoordinator.
                        continue;
                    }
                    SessionEngineCommand::DelegateSubagent { agent_name, task } => {
                        let parent_session_id = match transcript.lock() {
                            Ok(recorder) => recorder.session_id().to_string(),
                            Err(_) => {
                                let _ = session_transport_tx.send(SessionTransportEvent::Error(ErrorEvent::new(
                                    "transcript recorder poisoned",
                                )));
                                let _ = session_transport_tx.send(SessionTransportEvent::Done);
                                continue;
                            }
                        };

                        let invocation = match normalize_subagent_input(
                            &format!("agent__{agent_name}"),
                            &json!({ "task": task }),
                        ) {
                            Ok(input) => SubagentInvocation {
                                prompt: input.objective.clone(),
                                input,
                                parent_tool_call_id: None,
                            },
                            Err(error) => {
                                let _ = session_transport_tx.send(SessionTransportEvent::Error(ErrorEvent::new(
                                    format!("{error:#}"),
                                )));
                                let _ = session_transport_tx.send(SessionTransportEvent::Done);
                                continue;
                            }
                        };
                        let route_display_name = match delegated_route_for_takeover(
                            &agent,
                            &expert_model_routes,
                            &sessions_dir,
                            &transcript,
                            &agent_name,
                            invocation.input.target_child_session_id.as_deref(),
                        ) {
                            Ok(route_display_name) => route_display_name,
                            Err(error) => {
                                let _ = session_transport_tx.send(SessionTransportEvent::Error(
                                    ErrorEvent::new(error.to_string()),
                                ));
                                continue;
                            }
                        };
                        if !route_has_api_key(&route_api_key_configured, &route_display_name) {
                            send_missing_api_key_error(
                                &session_transport_tx,
                                &route_display_name,
                                &provider_api_key_hints,
                                &api_key_hint,
                            );
                            continue;
                        }

                        let (
                            interrupted,
                            child_started,
                            interrupted_child_session_id,
                            shutdown,
                        ) = {
                            let delegate = subagent_runtime.run_named_governed(
                                &agent,
                                &agent_name,
                                invocation,
                                sessions_dir.clone(),
                                parent_session_id,
                                format!(
                                    "turn-{}",
                                    std::time::SystemTime::now()
                                        .duration_since(std::time::UNIX_EPOCH)
                                        .unwrap_or_default()
                                        .as_millis()
                                ),
                                Some(transcript.clone()),
                                Some(crate::session::subagent_event_sender(session_transport_tx.clone())),
                            );

                            tokio::pin!(delegate);
                            let mut interrupted = false;
                            let mut child_started = false;
                            let mut interrupted_child_session_id = None;
                            let mut shutdown = false;

                            loop {
                                match select_active_session_operation(
                                    &mut control_rx,
                                    &mut deferred_commands,
                                    delegate.as_mut(),
                                )
                                .await
                                {
                                    outcome @ (ActiveSessionOperation::Interrupted
                                    | ActiveSessionOperation::Shutdown) => {
                                        shutdown = matches!(outcome, ActiveSessionOperation::Shutdown);
                                        let interrupt = derive_interrupt_request(
                                            &transcript,
                                            &subagent_runtime,
                                        );
                                        child_started = subagent_runtime.is_running();
                                        interrupted = true;
                                        interrupted_child_session_id = interrupt
                                            .visible_child_session_id
                                            .clone();
                                        if child_started {
                                            subagent_runtime.cancel_active();
                                        }
                                        record_interrupt_transcript(&transcript, &interrupt);
                                        if child_started {
                                            let settle_shutdown = wait_for_subagent_cancel_settle(
                                                &mut control_rx,
                                                &mut deferred_commands,
                                                delegate.as_mut(),
                                                &subagent_runtime,
                                            )
                                            .await;
                                            shutdown |= settle_shutdown;
                                        }
                                        break;
                                    }
                                    ActiveSessionOperation::Completed(result) => {
                                        match result {
                                            Ok(_) => {
                                                let _ = session_transport_tx.send(SessionTransportEvent::Done);
                                            }
                                            Err(error) => {
                                                let _ = session_transport_tx.send(SessionTransportEvent::Error(
                                                    ErrorEvent::new(format!("{error:#}")),
                                                ));
                                                let _ = session_transport_tx.send(SessionTransportEvent::Done);
                                            }
                                        }
                                        break;
                                    }
                                    ActiveSessionOperation::Command(Some(
                                        SessionEngineCommand::Prompt(prompt),
                                    )) => {
                                        deferred_commands.push_front(SessionEngineCommand::Prompt(prompt));
                                        let _ = session_transport_tx.send(SessionTransportEvent::AssistantDone {
                                            message_id: None,
                                        });
                                        break;
                                    }
                                    ActiveSessionOperation::Command(Some(
                                        SessionEngineCommand::ViewChild {
                                            navigation,
                                            anchor_child_session_id,
                                        },
                                    )) => {
                                        visible_child_session_id =
                                            crate::session::SessionCoordinator::emit_view_child(
                                                &transcript,
                                                &session_transport_tx,
                                                Some(sessions_dir.as_path()),
                                                navigation,
                                                anchor_child_session_id.as_deref(),
                                            );
                                        visible_child_view_state = None;
                                    }
                                    ActiveSessionOperation::Command(Some(
                                        SessionEngineCommand::ViewParent,
                                    )) => {
                                        crate::session::SessionCoordinator::emit_view_parent(
                                            &transcript,
                                            &session_transport_tx,
                                            Some(sessions_dir.as_path()),
                                        );
                                        visible_child_session_id = None;
                                        visible_child_view_state = None;
                                    }
                                    ActiveSessionOperation::Command(Some(
                                        SessionEngineCommand::Undo | SessionEngineCommand::Redo,
                                    )) => {
                                        let _ = session_transport_tx.send(SessionTransportEvent::Notice(
                                            NoticeEvent::info(
                                                "history navigation is unavailable while a turn is active",
                                            ),
                                        ));
                                    }
                                    ActiveSessionOperation::Command(Some(
                                        SessionEngineCommand::ShowHistoryTree
                                        | SessionEngineCommand::NavigateHistory { .. },
                                    )) => {
                                        let _ = session_transport_tx.send(SessionTransportEvent::Error(ErrorEvent::new(
                                            "history navigation is unavailable while a turn is active",
                                        )));
                                    }
                                    ActiveSessionOperation::RunnerEvent(_) => {
                                        unreachable!("event-aware selection is not used for delegates")
                                    }
                                    ActiveSessionOperation::Command(Some(command)) => {
                                        deferred_commands.push_front(command);
                                        let _ = session_transport_tx.send(SessionTransportEvent::Notice(
                                            NoticeEvent::info("Turn still running · navigation only"),
                                        ));
                                    }
                                    ActiveSessionOperation::Command(None) => break,
                                }
                            }

                            (
                                interrupted,
                                child_started,
                                interrupted_child_session_id,
                                shutdown,
                            )
                        };

                        if interrupted {
                            if child_started {
                                if let Err(error) =
                                    rehydrate_agent_from_transcript(&mut agent, &transcript)
                                {
                                    let _ = session_transport_tx.send(SessionTransportEvent::Error(ErrorEvent::new(format!(
                                        "failed to restore interrupted session context: {error}"
                                    ))));
                                }
                            }
                            send_subagent_interrupted(&session_transport_tx, interrupted_child_session_id);
                        }
                        if shutdown {
                            deferred_commands.clear();
                            break;
                        }
                        continue;
                    }
                    SessionEngineCommand::Compact => {
                        if !active_route_has_api_key(&agent, &route_api_key_configured) {
                            let route_display_name = agent.route_display_name();
                            send_missing_api_key_error(
                                &session_transport_tx,
                                &route_display_name,
                                &provider_api_key_hints,
                                &api_key_hint,
                            );
                            continue;
                        }
                        if subagent_runtime.is_running() {
                            let _ = session_transport_tx.send(SessionTransportEvent::Notice(NoticeEvent::info(
                                "Wait for the active subagent to finish before compacting context",
                            )));
                            let _ = session_transport_tx.send(SessionTransportEvent::Done);
                            continue;
                        }

                        let shutdown = run_manual_compaction(
                            &mut agent,
                            &transcript,
                            &session_transport_tx,
                            &mut control_rx,
                            &mut deferred_commands,
                        )
                        .await;
                        if shutdown {
                            deferred_commands.clear();
                            break;
                        }
                        continue;
                    }
                    #[cfg(test)]
                    SessionEngineCommand::InspectHistory(reply) => {
                        let _ = reply.send(agent.history_for_test().to_vec());
                        continue;
                    }
                    SessionEngineCommand::ResumeSession(prefix) => {
                        if subagent_runtime.is_running() {
                            let _ = session_transport_tx.send(SessionTransportEvent::Error(
                                ErrorEvent::new(
                                    "Wait for the active subagent to finish before resuming another session",
                                ),
                            ));
                            continue;
                        }

                        let session_id = match crate::session::resolve_session_prefix(
                            &sessions_dir,
                            &prefix,
                        ) {
                            Ok(session_id) => session_id,
                            Err(error) => {
                                let _ = session_transport_tx.send(SessionTransportEvent::Error(ErrorEvent::new(
                                    error.to_string(),
                                )));
                                continue;
                            }
                        };
                        let prepared = match crate::session::prepare_resume_package(
                            &sessions_dir,
                            session_id,
                        ) {
                            Ok(prepared) => prepared,
                            Err(error) => {
                                let _ = session_transport_tx.send(SessionTransportEvent::Error(ErrorEvent::new(format!(
                                    "failed to prepare resume: {error}"
                                ))));
                                continue;
                            }
                        };
                        let runtime_context =
                            match RuntimeActiveContext::try_from(&prepared.snapshot.snapshot) {
                            Ok(context) => context,
                            Err(error) => {
                                let _ = session_transport_tx.send(SessionTransportEvent::Error(ErrorEvent::new(format!(
                                    "failed to validate restored session context: {error}"
                                ))));
                                continue;
                            }
                        };
                        let resumed_event_session_id = prepared.session_id.clone();
                        let resumed_event_branch_id = prepared.snapshot.branch_id.clone();
                        let resumed_event_messages =
                            crate::session::restore::restored_messages_from_protocol_frames(
                                &prepared.snapshot.protocol_frames,
                            );
                        let resumed_event_records = prepared.snapshot.records.clone();
                        let resumed_event_evidence_count = prepared.snapshot.snapshot.evidence.len();
                        let (fast_mode_auto_disabled, token_usage) = match crate::session::install_prepared_routed_resume_for_agent(
                            &mut agent,
                            &transcript,
                            prepared,
                        ) {
                            Ok(result) => result,
                            Err(error) => {
                                if error.fast_mode_auto_disabled {
                                    let _ = session_transport_tx.send(SessionTransportEvent::FastModeChanged { enabled: false });
                                    let _ = session_transport_tx.send(SessionTransportEvent::Notice(NoticeEvent::info(
                                        "Fast mode auto-disabled: current model is unavailable",
                                    )));
                                }
                                let _ = session_transport_tx.send(SessionTransportEvent::Error(ErrorEvent::new(format!(
                                    "failed to install resumed session: {error}"
                                ))));
                                continue;
                            }
                        };
                        if fast_mode_auto_disabled {
                            let _ = session_transport_tx.send(SessionTransportEvent::FastModeChanged { enabled: false });
                            let _ = session_transport_tx.send(SessionTransportEvent::Notice(NoticeEvent::info(
                                "Fast mode auto-disabled: current model is unavailable",
                            )));
                        }
                        let _ = session_transport_tx.send(SessionTransportEvent::SessionResumed {
                            session_id: resumed_event_session_id,
                            branch_id: resumed_event_branch_id,
                            messages: resumed_event_messages,
                            records: resumed_event_records,
                            evidence_count: resumed_event_evidence_count,
                            model_id: Some(agent.route_display_name()),
                            token_usage: Some(token_usage),
                            runtime_context,
                        });
                        continue;
                    }
                    SessionEngineCommand::NewSession => {
                        if subagent_runtime.is_running() {
                            let _ = session_transport_tx.send(SessionTransportEvent::Notice(NoticeEvent::info(
                                "Wait for the active subagent to finish before starting a new session",
                            )));
                            continue;
                        }

                        let model = agent.route_display_name();
                        let prepared = match crate::session::prepare_new_session_package(
                            &sessions_dir,
                            model,
                        ) {
                            Ok(prepared) => prepared,
                            Err(error) => {
                                let _ = session_transport_tx.send(SessionTransportEvent::Error(ErrorEvent::new(format!(
                                    "failed to create session transcript: {error}"
                                ))));
                                continue;
                            }
                        };
                        let started_event = session_started_event(&prepared);
                        let new_path = prepared.recorder.path().to_path_buf();
                        if let Err(error) =
                            crate::session::install_prepared_new_session_for_agent(
                                &mut agent,
                                &transcript,
                                prepared,
                            )
                        {
                            let _ = remove_empty_session_file(&new_path);
                            let _ = session_transport_tx.send(SessionTransportEvent::Error(ErrorEvent::new(format!(
                                "failed to install new session: {error}"
                            ))));
                            continue;
                        }
                        let _ = session_transport_tx.send(started_event);
                        continue;
                    }
                };

                let _ = session_transport_tx.send(SessionTransportEvent::QueuedPromptAccepted {
                    prompt: prompt.clone(),
                });

                if !active_route_has_api_key(&agent, &route_api_key_configured) {
                    let route_display_name = agent.route_display_name();
                    send_missing_api_key_error(
                        &session_transport_tx,
                        &route_display_name,
                        &provider_api_key_hints,
                        &api_key_hint,
                    );
                    continue;
                }

                let (runner_event_tx, mut runner_event_rx) = mpsc::unbounded_channel();
                let runner = AgentRunner::<async_openai::config::OpenAIConfig>::with_transcript(
                    runner_event_tx,
                    transcript.clone(),
                )
                    .with_session_title_event_sender(title_event_tx.clone())
                    .with_subagent_runtime(
                        subagent_runtime.clone(),
                        sessions_dir.clone(),
                        expert_model_routes.clone(),
                        route_api_key_configured.clone(),
                        provider_api_key_hints.clone(),
                        api_key_hint.clone(),
                    );
                let (interrupted, shutdown) = {
                    let run = runner.run_prompt(&mut agent, prompt);
                    tokio::pin!(run);
                    let mut interrupted = None;
                    let mut shutdown = false;

                    loop {
                        match select_active_session_operation_with_events(
                            &mut control_rx,
                            &mut deferred_commands,
                            run.as_mut(),
                            Some(&mut runner_event_rx),
                        )
                        .await
                        {
                            outcome @ (ActiveSessionOperation::Interrupted
                            | ActiveSessionOperation::Shutdown) => {
                                let is_shutdown =
                                    matches!(outcome, ActiveSessionOperation::Shutdown);
                                // Capture the interrupt request while the subagent is
                                // still active so the visible child session can be
                                // reported. Then signal cancellation and poll the run
                                // until the in-flight subagent's completion teardown
                                // (cancelled terminal record, guard release) settles.
                                let interrupt = derive_interrupt_request(
                                    &transcript,
                                    &subagent_runtime,
                                );
                                interrupted = Some(interrupt);
                                if subagent_runtime.is_running() {
                                    subagent_runtime.cancel_active();
                                    let settle_shutdown = wait_for_subagent_cancel_settle(
                                        &mut control_rx,
                                        &mut deferred_commands,
                                        run.as_mut(),
                                        &subagent_runtime,
                                    )
                                    .await;
                                    shutdown = is_shutdown || settle_shutdown;
                                } else {
                                    shutdown = is_shutdown;
                                }
                                break;
                            }
                            ActiveSessionOperation::RunnerEvent(SessionTransportEvent::Done) => {
                                // Runner completion is internal until its future has
                                // settled. This keeps external Done authoritative.
                            }
                            ActiveSessionOperation::RunnerEvent(event) => {
                                let _ = session_transport_tx.send(event);
                            }
                            ActiveSessionOperation::Completed(_) => {
                                forward_queued_runner_events(
                                    &mut runner_event_rx,
                                    &session_transport_tx,
                                );
                                let _ = session_transport_tx.send(SessionTransportEvent::Done);
                                break;
                            }
                            ActiveSessionOperation::Command(command) => match command {
                                Some(SessionEngineCommand::Prompt(prompt)) => {
                                    deferred_commands.push_front(SessionEngineCommand::Prompt(prompt));
                                    let _ = session_transport_tx.send(SessionTransportEvent::AssistantDone {
                                        message_id: None,
                                    });
                                    break;
                                }
                                Some(SessionEngineCommand::ViewChild {
                                    navigation,
                                    anchor_child_session_id,
                                }) => {
                                    visible_child_session_id =
                                        crate::session::SessionCoordinator::emit_view_child(
                                            &transcript,
                                            &session_transport_tx,
                                            Some(sessions_dir.as_path()),
                                            navigation,
                                            anchor_child_session_id.as_deref(),
                                        );
                                    visible_child_view_state = None;
                                }
                                Some(SessionEngineCommand::ViewParent) => {
                                    crate::session::SessionCoordinator::emit_view_parent(
                                        &transcript,
                                        &session_transport_tx,
                                        Some(sessions_dir.as_path()),
                                    );
                                    visible_child_session_id = None;
                                    visible_child_view_state = None;
                                }
                                Some(SessionEngineCommand::Undo) | Some(SessionEngineCommand::Redo) => {
                                    let _ = session_transport_tx.send(SessionTransportEvent::Notice(
                                        NoticeEvent::info(
                                            "history navigation is unavailable while a turn is active",
                                        ),
                                    ));
                                }
                                Some(SessionEngineCommand::ShowHistoryTree)
                                | Some(SessionEngineCommand::NavigateHistory { .. }) => {
                                    let _ = session_transport_tx.send(SessionTransportEvent::Error(ErrorEvent::new(
                                        "history navigation is unavailable while a turn is active",
                                    )));
                                }
                                Some(command) => {
                                    deferred_commands.push_front(command);
                                    let _ = session_transport_tx.send(SessionTransportEvent::Notice(
                                        NoticeEvent::info("Turn still running · navigation only"),
                                    ));
                                }
                                None => break,
                            },
                        }
                    }

                    (interrupted, shutdown)
                };

                if let Some(interrupt) = interrupted {
                    subagent_runtime.cancel_active();
                    record_interrupt_transcript(&transcript, &interrupt);
                    if let Err(error) = rehydrate_agent_from_transcript(&mut agent, &transcript) {
                        let _ = session_transport_tx.send(SessionTransportEvent::Error(ErrorEvent::new(format!(
                            "failed to restore interrupted session context: {error}"
                        ))));
                    }
                    send_subagent_interrupted(&session_transport_tx, interrupt.visible_child_session_id);
                }
                if shutdown {
                    deferred_commands.clear();
                    break;
                }
            }
            _ = child_refresh.tick(), if visible_child_session_id.is_some() => {
                refresh_visible_child_session_view(
                    &transcript,
                    &session_transport_tx,
                    &sessions_dir,
                    &mut visible_child_session_id,
                    &mut visible_child_view_state,
                ).await;
            }
            discovery = async {
                mcp_tools_rx
                    .as_mut()
                    .expect("MCP discovery receiver should exist when select branch is enabled")
                    .recv()
                    .await
            }, if mcp_tools_rx.is_some() => {
                let Some(discovery) = discovery else {
                    mcp_tools_rx = None;
                    continue;
                };
                mcp_tools_rx = None;

                let mut servers = Vec::with_capacity(discovery.len());
                for server_discovery in discovery {
                    let mut server = server_discovery.server;
                    let mut catalog_tools = Vec::new();
                    if let mcp::McpServerStatus::Offline { message } = &server.status {
                        let _ = session_transport_tx.send(SessionTransportEvent::McpDiagnostic(format!(
                            "MCP server '{}' is offline: {message}",
                            server.name
                        )));
                    }
                    let mut registered = Vec::new();
                    for tool in server_discovery.tools {
                        let tool_name = tool.name().to_string();
                        let catalog_entry = tool.catalog_entry();
                        if let Err(error) = agent.try_register_tool(tool) {
                            let _ = session_transport_tx.send(SessionTransportEvent::McpDiagnostic(format!(
                                "failed to register MCP tool '{tool_name}': {error}"
                            )));
                        } else {
                            registered.push(tool_name);
                            catalog_tools.push(catalog_entry);
                        }
                    }
                    if matches!(server.status, mcp::McpServerStatus::Online { .. }) {
                        server.status = mcp::McpServerStatus::Online {
                            tool_count: registered.len(),
                        };
                        mcp_registered_tools.insert(server.name.clone(), registered);
                    }
                    let _ = session_transport_tx.send(SessionTransportEvent::McpServerToolsUpdated {
                        name: server.name.clone(),
                        tools: catalog_tools,
                    });
                    servers.push(server);
                }
                let _ = session_transport_tx.send(SessionTransportEvent::McpToolsDiscovered(servers));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_openai::{Client, config::OpenAIConfig};
    use std::fs;
    use std::sync::Arc;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    fn temp_sessions_dir() -> PathBuf {
        std::env::temp_dir().join(format!(
            "letcode-session-engine-test-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system time is valid")
                .as_nanos()
        ))
    }

    fn parent_transcript(sessions_dir: &std::path::Path) -> Arc<StdMutex<TranscriptRecorder>> {
        Arc::new(StdMutex::new(
            TranscriptRecorder::create(sessions_dir).expect("create parent transcript"),
        ))
    }

    fn add_child(
        transcript: &Arc<StdMutex<TranscriptRecorder>>,
        sessions_dir: &std::path::Path,
        run_id: &str,
        pool_ordinal: u32,
    ) -> String {
        let child_dir = crate::transcript::child_sessions_dir(sessions_dir);
        let mut child = TranscriptRecorder::create(child_dir).expect("create child transcript");
        let child_session_id = child.session_id().to_string();
        child
            .record_user_message("child transcript")
            .expect("record child message");
        drop(child);

        let mut parent = transcript.lock().expect("parent transcript");
        let parent_session_id = parent.session_id().to_string();
        parent
            .record_subagent_started(
                run_id,
                parent_session_id,
                "turn-1",
                &child_session_id,
                "explorer",
                "inspect child",
                pool_ordinal,
            )
            .expect("record child start");
        child_session_id
    }

    fn child_view_event(event: SessionTransportEvent) -> (String, usize, usize) {
        match event {
            SessionTransportEvent::ChildSessionViewed {
                child_session_id,
                index,
                total,
                ..
            } => (child_session_id, index, total),
            other => panic!("expected child session view, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn visible_child_refresh_emits_when_total_changes_without_new_child_records() {
        let sessions_dir = temp_sessions_dir();
        let transcript = parent_transcript(&sessions_dir);
        let visible_child_session_id = add_child(&transcript, &sessions_dir, "run-1", 1);
        let (session_transport_tx, mut session_transport_rx) = mpsc::unbounded_channel();
        let mut visible_child = Some(visible_child_session_id.clone());
        let mut view_state = None;

        refresh_visible_child_session_view(
            &transcript,
            &session_transport_tx,
            &sessions_dir,
            &mut visible_child,
            &mut view_state,
        )
        .await;
        assert_eq!(
            child_view_event(session_transport_rx.try_recv().expect("initial child view")),
            (visible_child_session_id.clone(), 0, 1)
        );

        add_child(&transcript, &sessions_dir, "run-2", 2);
        refresh_visible_child_session_view(
            &transcript,
            &session_transport_tx,
            &sessions_dir,
            &mut visible_child,
            &mut view_state,
        )
        .await;
        assert_eq!(
            child_view_event(
                session_transport_rx
                    .try_recv()
                    .expect("refreshed child view")
            ),
            (visible_child_session_id, 0, 2)
        );
    }

    #[tokio::test]
    async fn visible_child_refresh_emits_when_index_changes_without_new_child_records() {
        let sessions_dir = temp_sessions_dir();
        let transcript = parent_transcript(&sessions_dir);
        let visible_child_session_id = add_child(&transcript, &sessions_dir, "run-1", 2);
        let (session_transport_tx, mut session_transport_rx) = mpsc::unbounded_channel();
        let mut visible_child = Some(visible_child_session_id.clone());
        let mut view_state = None;

        refresh_visible_child_session_view(
            &transcript,
            &session_transport_tx,
            &sessions_dir,
            &mut visible_child,
            &mut view_state,
        )
        .await;
        assert_eq!(
            child_view_event(session_transport_rx.try_recv().expect("initial child view")),
            (visible_child_session_id.clone(), 0, 1)
        );

        add_child(&transcript, &sessions_dir, "run-2", 1);
        refresh_visible_child_session_view(
            &transcript,
            &session_transport_tx,
            &sessions_dir,
            &mut visible_child,
            &mut view_state,
        )
        .await;
        assert_eq!(
            child_view_event(
                session_transport_rx
                    .try_recv()
                    .expect("refreshed child view")
            ),
            (visible_child_session_id, 1, 2)
        );
    }

    #[tokio::test]
    async fn visible_child_refresh_suppresses_identical_projection() {
        let sessions_dir = temp_sessions_dir();
        let transcript = parent_transcript(&sessions_dir);
        let visible_child_session_id = add_child(&transcript, &sessions_dir, "run-1", 1);
        let (session_transport_tx, mut session_transport_rx) = mpsc::unbounded_channel();
        let mut visible_child = Some(visible_child_session_id);
        let mut view_state = None;

        refresh_visible_child_session_view(
            &transcript,
            &session_transport_tx,
            &sessions_dir,
            &mut visible_child,
            &mut view_state,
        )
        .await;
        assert!(matches!(
            session_transport_rx.try_recv(),
            Ok(SessionTransportEvent::ChildSessionViewed { .. })
        ));

        refresh_visible_child_session_view(
            &transcript,
            &session_transport_tx,
            &sessions_dir,
            &mut visible_child,
            &mut view_state,
        )
        .await;
        assert!(session_transport_rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn command_ingress_preserves_fifo_order() {
        let (mut engine, ingress, _egress) = SessionEngine::new();
        ingress
            .submit(SessionCommand::SetModel("first".into()))
            .expect("engine accepts first command");
        ingress
            .submit(SessionCommand::SetModel("second".into()))
            .expect("engine accepts second command");

        assert!(matches!(
            engine.recv_control().await,
            Some(SessionEngineControl::Command(SessionEngineCommand::SetModel(model))) if model == "first"
        ));
        assert!(matches!(
            engine.recv_control().await,
            Some(SessionEngineControl::Command(SessionEngineCommand::SetModel(model))) if model == "second"
        ));
    }

    #[tokio::test]
    async fn command_ingress_maps_expert_model_selection() {
        let (mut engine, ingress, _egress) = SessionEngine::new();
        ingress
            .submit(SessionCommand::SetExpertModel {
                agent_name: "explorer".into(),
                model_id: "expert/shared".into(),
            })
            .expect("engine accepts expert model command");

        assert!(matches!(
            engine.recv_control().await,
            Some(SessionEngineControl::Command(SessionEngineCommand::SetExpertModel {
                agent_name,
                model_id,
            })) if agent_name == "explorer" && model_id == "expert/shared"
        ));
    }

    #[tokio::test]
    async fn interrupt_and_shutdown_are_frontend_neutral_intents() {
        let (mut engine, ingress, _egress) = SessionEngine::new();
        ingress
            .request_interrupt()
            .expect("engine accepts interrupt intent");
        ingress.shutdown().expect("engine accepts shutdown intent");

        assert!(matches!(
            engine.recv_control().await,
            Some(SessionEngineControl::Interrupt)
        ));
        assert!(matches!(
            engine.recv_control().await,
            Some(SessionEngineControl::Shutdown)
        ));
    }

    #[tokio::test]
    async fn closing_command_ingress_stops_the_engine_stream() {
        let (mut engine, ingress, _egress) = SessionEngine::new();
        drop(ingress);

        assert!(engine.recv_control().await.is_none());
    }

    #[test]
    fn closing_event_egress_closes_the_engine_sender_without_panicking() {
        let (engine, _ingress, egress) = SessionEngine::new();
        let event_tx = engine.event_sender();
        drop(egress);

        assert!(event_tx.send(SessionTransportEvent::Done).is_err());
    }

    #[test]
    fn active_route_credential_lookup_uses_the_provider_qualified_route() {
        let route_api_key_configured = indexmap::IndexMap::from([
            ("primary/shared".into(), true),
            ("expert/shared".into(), false),
        ]);
        let mut agent = Agent::new(
            async_openai::Client::with_config(
                async_openai::config::OpenAIConfig::new()
                    .with_api_base("http://127.0.0.1:9/v1")
                    .with_api_key("test-key"),
            ),
            "shared",
            1,
            1,
        );
        agent.set_primary_route(ModelRoute::new("expert", "shared"));

        assert!(!active_route_has_api_key(&agent, &route_api_key_configured));

        agent.set_primary_route(ModelRoute::new("primary", "shared"));
        assert!(active_route_has_api_key(&agent, &route_api_key_configured));
    }

    #[test]
    fn direct_expert_execution_uses_the_selected_expert_provider_credential() {
        let route_api_key_configured = indexmap::IndexMap::from([
            ("primary/shared".into(), true),
            ("expert/shared".into(), false),
        ]);
        let mut agent = Agent::new(
            async_openai::Client::with_config(
                async_openai::config::OpenAIConfig::new()
                    .with_api_base("http://127.0.0.1:9/v1")
                    .with_api_key("primary-key"),
            ),
            "shared",
            1,
            1,
        );
        agent.set_primary_route(ModelRoute::new("expert", "shared"));

        assert!(
            !active_route_has_api_key(&agent, &route_api_key_configured),
            "direct expert delegation must not inherit the primary provider credential"
        );
        assert_eq!(
            delegated_route_display_name(
                &agent,
                &indexmap::IndexMap::from([(
                    "explorer".into(),
                    ModelRoute::new("expert", "shared"),
                )]),
                "explorer",
            ),
            "expert/shared"
        );
        assert_eq!(
            route_api_key_hint(
                &agent.route_display_name(),
                &indexmap::IndexMap::from([("expert".into(), "Set EXPERT_API_KEY.".into())]),
                "Set <PROVIDER>_API_KEY.",
            ),
            "Set EXPERT_API_KEY."
        );
    }

    #[test]
    fn delegated_credential_lookup_uses_the_selected_expert_route_or_primary_fallback() {
        let mut agent = Agent::new(
            async_openai::Client::with_config(
                async_openai::config::OpenAIConfig::new()
                    .with_api_base("http://127.0.0.1:9/v1")
                    .with_api_key("primary-key"),
            ),
            "shared",
            1,
            1,
        );
        agent.set_primary_route(ModelRoute::new("primary", "shared"));
        let routes =
            indexmap::IndexMap::from([("explorer".into(), ModelRoute::new("expert", "shared"))]);
        let credentials = indexmap::IndexMap::from([
            ("primary/shared".into(), true),
            ("expert/shared".into(), false),
        ]);

        let explorer_route = delegated_route_display_name(&agent, &routes, "explorer");
        assert_eq!(explorer_route, "expert/shared");
        assert!(!route_has_api_key(&credentials, &explorer_route));
        assert_eq!(
            route_api_key_hint(
                &explorer_route,
                &indexmap::IndexMap::from([("expert".into(), "Set EXPERT_API_KEY.".into())]),
                "Set <PROVIDER>_API_KEY.",
            ),
            "Set EXPERT_API_KEY."
        );

        let general_route = delegated_route_display_name(&agent, &routes, "general");
        assert_eq!(general_route, "primary/shared");
        assert!(route_has_api_key(&credentials, &general_route));
    }

    #[test]
    fn legacy_expert_routes_follow_primary_switches_but_explicit_routes_stay_fixed() {
        let routes = indexmap::IndexMap::from([
            ("explorer".into(), ModelRoute::new("primary", "legacy")),
            ("fixer".into(), ModelRoute::new("expert", "fixed")),
        ]);
        let legacy_models = indexmap::IndexMap::from([("explorer".into(), "legacy".into())]);

        assert_eq!(
            expert_routes_after_primary_switch(
                &routes,
                &legacy_models,
                &ModelRoute::new("secondary", "primary-model"),
            ),
            indexmap::IndexMap::from([
                ("explorer".into(), ModelRoute::new("secondary", "legacy")),
                ("fixer".into(), ModelRoute::new("expert", "fixed")),
            ])
        );
    }

    #[test]
    fn takeover_credential_lookup_uses_the_historical_child_route() {
        let sessions_dir = temp_sessions_dir();
        let parent = parent_transcript(&sessions_dir);
        let child_session_id = add_child(&parent, &sessions_dir, "run-1", 1);
        let child_path = crate::transcript::child_sessions_dir(&sessions_dir)
            .join(format!("{child_session_id}.jsonl"));
        let mut child = TranscriptRecorder::open(
            crate::transcript::child_sessions_dir(&sessions_dir),
            child_session_id.clone(),
        )
        .expect("open child transcript");
        child
            .record_session_started("expert/shared")
            .expect("record child route");
        drop(child);
        assert!(child_path.exists());

        let agent = Agent::new(
            async_openai::Client::with_config(async_openai::config::OpenAIConfig::new()),
            "shared",
            1,
            1,
        );
        let route = delegated_route_for_takeover(
            &agent,
            &indexmap::IndexMap::from([("explorer".into(), ModelRoute::new("primary", "shared"))]),
            &sessions_dir,
            &parent,
            "explorer",
            Some(&child_session_id),
        )
        .expect("historical child route resolves");
        let credentials = indexmap::IndexMap::from([
            ("primary/shared".into(), true),
            ("expert/shared".into(), false),
        ]);

        assert_eq!(route, "expert/shared");
        assert!(
            !route_has_api_key(&credentials, &route),
            "takeover must validate its historical child provider, not the current expert route"
        );
    }

    #[test]
    fn missing_api_key_error_emits_error_then_exactly_one_done() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        send_missing_api_key_error(
            &tx,
            "expert/shared",
            &indexmap::IndexMap::from([("expert".into(), "Set EXPERT_API_KEY.".into())]),
            "fallback",
        );

        assert!(matches!(
            rx.try_recv(),
            Ok(SessionTransportEvent::Error(error))
                if error.message == "API key is not set for the selected provider. Set EXPERT_API_KEY."
        ));
        assert!(matches!(rx.try_recv(), Ok(SessionTransportEvent::Done)));
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn route_credential_lookup_and_hint_use_the_selected_provider() {
        let route_api_key_configured = indexmap::IndexMap::from([
            ("primary/shared".into(), true),
            ("expert/shared".into(), false),
        ]);

        assert!(route_has_api_key(
            &route_api_key_configured,
            "primary/shared"
        ));
        assert!(!route_has_api_key(
            &route_api_key_configured,
            "expert/shared"
        ));
        assert!(!route_has_api_key(
            &route_api_key_configured,
            "unknown/shared"
        ));
        assert_eq!(
            route_api_key_hint(
                "expert/shared",
                &indexmap::IndexMap::from([("expert".into(), "Set EXPERT_API_KEY.".into())]),
                "Set <PROVIDER>_API_KEY.",
            ),
            "Set EXPERT_API_KEY."
        );
    }

    #[test]
    fn initial_projection_preserves_distinct_model_label() {
        let projection = SessionEngineProjection {
            session_id: "session".into(),
            session_title: None,
            model_id: "provider/model-id".into(),
            model_label: "Provider Model Label".into(),
            permission_mode_label: "ask".into(),
            fast_mode_enabled: false,
            api_key_configured: true,
        };

        assert_ne!(projection.model_id, projection.model_label);
        assert_eq!(projection.model_label, "Provider Model Label");
    }

    #[test]
    fn reload_failure_preserves_engine_owned_state() {
        let old_path = std::env::temp_dir().join(format!(
            "letcode-engine-reload-old-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system time is valid")
                .as_nanos()
        ));
        let bad_path = old_path.with_file_name("letcode-engine-reload-bad.toml");
        let old_contents = r#"
            active_provider = "primary"

            [providers.primary]
            base_url = "https://example.invalid/v1"
            api_key = "old-key"
            protocol = "responses"

            [providers.primary.models.old]
            "#;
        let bad_contents = r#"
            active_provider = "primary"

            [providers.primary]
            api_key = "new-key"
            protocol = "responses"

            [providers.primary.models.new]
            "#;
        fs::write(&old_path, old_contents).expect("write old config");
        fs::write(&bad_path, bad_contents).expect("write invalid reload config");
        let old_config = AppConfig::load_from_path(&old_path).expect("old config should load");
        let route = ModelRoute::new("primary", "old");
        let mut agent = Agent::new(Client::with_config(OpenAIConfig::new()), "old", 1, 1);
        agent.set_primary_route(route.clone());

        let mut model_routes = indexmap::IndexMap::from([(route.display_name(), route.clone())]);
        let mut route_api_key_configured = indexmap::IndexMap::from([(route.display_name(), true)]);
        let mut expert_model_routes =
            indexmap::IndexMap::from([(String::from("explorer"), route.clone())]);
        let mut legacy_expert_models =
            indexmap::IndexMap::from([(String::from("explorer"), String::from("old"))]);
        let mut providers = old_config.providers.clone();
        let mut global_retry = old_config.global.retry.clone();
        let mut provider_api_key_hints =
            indexmap::IndexMap::from([(String::from("primary"), String::from("old hint"))]);
        let old_model_routes = model_routes.clone();
        let old_route_api_key_configured = route_api_key_configured.clone();
        let old_expert_model_routes = expert_model_routes.clone();
        let old_legacy_expert_models = legacy_expert_models.clone();
        let old_providers = providers.clone();
        let old_global_retry = global_retry.clone();
        let old_provider_api_key_hints = provider_api_key_hints.clone();
        let (event_tx, mut event_rx) = mpsc::unbounded_channel();

        assert!(
            apply_config_reload(
                &mut agent,
                &bad_path,
                &mut model_routes,
                &mut route_api_key_configured,
                &mut expert_model_routes,
                &mut legacy_expert_models,
                &mut providers,
                &mut global_retry,
                &mut provider_api_key_hints,
                &event_tx,
            )
            .is_err()
        );

        assert_eq!(agent.primary_route(), Some(&route));
        assert_eq!(model_routes, old_model_routes);
        assert_eq!(route_api_key_configured, old_route_api_key_configured);
        assert_eq!(expert_model_routes, old_expert_model_routes);
        assert_eq!(legacy_expert_models, old_legacy_expert_models);
        assert_eq!(
            providers.keys().collect::<Vec<_>>(),
            old_providers.keys().collect::<Vec<_>>()
        );
        assert_eq!(
            providers["primary"].api_key,
            old_providers["primary"].api_key
        );
        assert_eq!(
            providers["primary"].models.keys().collect::<Vec<_>>(),
            old_providers["primary"].models.keys().collect::<Vec<_>>()
        );
        assert_eq!(global_retry, old_global_retry);
        assert_eq!(provider_api_key_hints, old_provider_api_key_hints);
        assert!(event_rx.try_recv().is_err());

        let _ = fs::remove_file(old_path);
        let _ = fs::remove_file(bad_path);
    }

    #[test]
    fn reload_factories_construct_and_prepare_configured_routes() {
        let path = std::env::temp_dir().join(format!(
            "letcode-engine-reload-factory-{}.toml",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system time is valid")
                .as_nanos()
        ));
        fs::write(
            &path,
            r#"
            active_provider = "primary"

            [providers.primary]
            base_url = "https://example.invalid/v1"
            api_key = "config-key"
            protocol = "responses"

            [providers.primary.models.primary-model]

            [providers.primary.models.expert-model]

            [agents.explorer]
            provider = "primary"
            model = "expert-model"
            "#,
        )
        .expect("write factory config");
        let config = AppConfig::load_from_path(&path).expect("factory config should load");
        let expert_route = ModelRoute::new("primary", "expert-model");
        let expert_factory = crate::subagent::ExpertRouteFactory::new(
            [(String::from("explorer"), expert_route.clone())],
            &config.providers,
            &config.global.retry,
        )
        .expect("expert factory should construct");
        let _prepared_expert = <crate::subagent::ExpertRouteFactory as PrimaryRouteFactory<
            OpenAIConfig,
        >>::prepare_route(&expert_factory, expert_route)
        .expect("expert factory should prepare configured route");

        let primary_factory = ConfiguredPrimaryRouteFactory::new(
            config.providers.clone(),
            config.global.retry.clone(),
        );
        let _prepared_primary = primary_factory
            .prepare_route(ModelRoute::new("primary", "primary-model"))
            .expect("primary factory should prepare configured route");

        let _ = fs::remove_file(path);
    }

    #[test]
    fn reload_applies_configured_active_route_change() {
        let path = std::env::temp_dir().join(format!(
            "letcode-engine-reload-active-route-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system time is valid")
                .as_nanos()
        ));
        fs::write(
            &path,
            r#"
            active_provider = "primary"

            [providers.primary]
            base_url = "https://primary.example.invalid/v1"
            api_key = "primary-key"
            protocol = "responses"
            default_model = "primary-model"

            [providers.primary.models.primary-model]

            [providers.secondary]
            base_url = "https://secondary.example.invalid/v1"
            api_key = "secondary-key"
            protocol = "responses"
            default_model = "secondary-model"

            [providers.secondary.models.secondary-model]
            "#,
        )
        .expect("write initial active route config");

        let primary_route = ModelRoute::new("primary", "primary-model");
        let secondary_route = ModelRoute::new("secondary", "secondary-model");
        let mut agent = Agent::new(
            Client::with_config(OpenAIConfig::new()),
            primary_route.model.clone(),
            1,
            1,
        );
        agent.set_primary_route(primary_route.clone());
        let initial_config = AppConfig::load_from_path(&path).expect("initial config should load");
        let mut model_routes = indexmap::IndexMap::new();
        let mut route_api_key_configured = indexmap::IndexMap::new();
        let mut expert_model_routes = indexmap::IndexMap::new();
        let mut legacy_expert_models = indexmap::IndexMap::new();
        let mut providers = initial_config.providers.clone();
        let mut global_retry = initial_config.global.retry.clone();
        let mut provider_api_key_hints = indexmap::IndexMap::new();
        let (event_tx, mut event_rx) = mpsc::unbounded_channel();

        fs::write(
            &path,
            r#"
            active_provider = "secondary"

            [providers.primary]
            base_url = "https://primary.example.invalid/v1"
            api_key = "primary-key"
            protocol = "responses"
            default_model = "primary-model"

            [providers.primary.models.primary-model]

            [providers.secondary]
            base_url = "https://secondary.example.invalid/v1"
            api_key = "secondary-key"
            protocol = "responses"
            default_model = "secondary-model"

            [providers.secondary.models.secondary-model]
            "#,
        )
        .expect("write updated active route config");

        apply_config_reload(
            &mut agent,
            &path,
            &mut model_routes,
            &mut route_api_key_configured,
            &mut expert_model_routes,
            &mut legacy_expert_models,
            &mut providers,
            &mut global_retry,
            &mut provider_api_key_hints,
            &event_tx,
        )
        .expect("active route config should reload");

        assert_eq!(agent.primary_route(), Some(&secondary_route));
        assert!(matches!(
            event_rx.try_recv(),
            Ok(SessionTransportEvent::ModelChanged { model_id })
                if model_id == "secondary/secondary-model"
        ));

        let _ = fs::remove_file(path);
    }

    #[test]
    fn reload_emits_effective_expert_route_changes() {
        let path = std::env::temp_dir().join(format!(
            "letcode-engine-reload-expert-{}.toml",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system time is valid")
                .as_nanos()
        ));
        let config = |agent_entry: &str| {
            format!(
                r#"
                active_provider = "primary"

                [providers.primary]
                base_url = "https://example.invalid/v1"
                api_key = "config-key"
                protocol = "responses"

                [providers.primary.models.primary-model]
                [providers.primary.models.expert-old]
                [providers.primary.models.expert-new]
                {agent_entry}
                "#
            )
        };
        fs::write(
            &path,
            config(
                r#"
                [agents.explorer]
                provider = "primary"
                model = "expert-old"
                "#,
            ),
        )
        .expect("write initial expert config");

        let primary_route = ModelRoute::new("primary", "primary-model");
        let old_route = ModelRoute::new("primary", "expert-old");
        let mut agent = Agent::new(
            Client::with_config(OpenAIConfig::new()),
            "primary-model",
            1,
            1,
        );
        agent.set_primary_route(primary_route.clone());
        let mut model_routes = indexmap::IndexMap::new();
        let mut route_api_key_configured = indexmap::IndexMap::new();
        let mut expert_model_routes =
            indexmap::IndexMap::from([(String::from("explorer"), old_route)]);
        let mut legacy_expert_models = indexmap::IndexMap::new();
        let initial_config = AppConfig::load_from_path(&path).expect("initial config should load");
        let mut providers = initial_config.providers.clone();
        let mut global_retry = initial_config.global.retry.clone();
        let mut provider_api_key_hints = indexmap::IndexMap::new();
        let (event_tx, mut event_rx) = mpsc::unbounded_channel();

        fs::write(
            &path,
            config(
                r#"
                [agents.explorer]
                provider = "primary"
                model = "expert-new"
                "#,
            ),
        )
        .expect("write edited expert config");
        apply_config_reload(
            &mut agent,
            &path,
            &mut model_routes,
            &mut route_api_key_configured,
            &mut expert_model_routes,
            &mut legacy_expert_models,
            &mut providers,
            &mut global_retry,
            &mut provider_api_key_hints,
            &event_tx,
        )
        .expect("edited expert config should reload");
        assert!(matches!(
            event_rx.try_recv(),
            Ok(SessionTransportEvent::ExpertModelChanged { agent_name, model_id })
                if agent_name == "explorer" && model_id == "primary/expert-new"
        ));
        while event_rx.try_recv().is_ok() {}

        fs::write(&path, config("")).expect("write removed expert config");
        apply_config_reload(
            &mut agent,
            &path,
            &mut model_routes,
            &mut route_api_key_configured,
            &mut expert_model_routes,
            &mut legacy_expert_models,
            &mut providers,
            &mut global_retry,
            &mut provider_api_key_hints,
            &event_tx,
        )
        .expect("removed expert config should reload");
        assert!(matches!(
            event_rx.try_recv(),
            Ok(SessionTransportEvent::ExpertModelChanged { agent_name, model_id })
                if agent_name == "explorer" && model_id == "primary/primary-model"
        ));

        let _ = fs::remove_file(path);
    }

    #[test]
    fn reload_is_silent_when_runtime_fields_are_unchanged() {
        let path = std::env::temp_dir().join(format!(
            "letcode-engine-reload-noop-{}.toml",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system time is valid")
                .as_nanos()
        ));
        let contents = r#"
            active_provider = "primary"

            [providers.primary]
            base_url = "https://example.invalid/v1"
            api_key = "config-key"
            protocol = "responses"
            default_model = "primary-model"

            [providers.primary.models.primary-model]
            "#;
        fs::write(&path, contents).expect("write config");

        let route = ModelRoute::new("primary", "primary-model");
        let mut agent = Agent::new(
            Client::with_config(OpenAIConfig::new()),
            route.model.clone(),
            1,
            1,
        );
        agent.set_primary_route(route.clone());
        let config = AppConfig::load_from_path(&path).expect("config should load");
        let mut model_routes = indexmap::IndexMap::from([(route.display_name(), route.clone())]);
        let mut route_api_key_configured =
            indexmap::IndexMap::from([(route.display_name(), true)]);
        let mut expert_model_routes = indexmap::IndexMap::new();
        let mut legacy_expert_models = indexmap::IndexMap::new();
        let mut providers = config.providers.clone();
        let mut global_retry = config.global.retry.clone();
        let mut provider_api_key_hints = indexmap::IndexMap::from([(
            String::from("primary"),
            format!(
                "Set providers.primary.api_key in {} or set {}.",
                config.config_path.display(),
                crate::config::provider_api_key_env_var("primary")
            ),
        )]);
        let (event_tx, mut event_rx) = mpsc::unbounded_channel();

        apply_config_reload(
            &mut agent,
            &path,
            &mut model_routes,
            &mut route_api_key_configured,
            &mut expert_model_routes,
            &mut legacy_expert_models,
            &mut providers,
            &mut global_retry,
            &mut provider_api_key_hints,
            &event_tx,
        )
        .expect("warming reload should succeed");
        while event_rx.try_recv().is_ok() {}

        apply_config_reload(
            &mut agent,
            &path,
            &mut model_routes,
            &mut route_api_key_configured,
            &mut expert_model_routes,
            &mut legacy_expert_models,
            &mut providers,
            &mut global_retry,
            &mut provider_api_key_hints,
            &event_tx,
        )
        .expect("identical second reload should succeed");
        assert!(
            event_rx.try_recv().is_err(),
            "noop reload must stay silent"
        );

        let _ = fs::remove_file(path);
    }

    #[test]
    fn reload_preserves_usage_anchor_when_only_default_model_diverges() {
        let path = std::env::temp_dir().join(format!(
            "letcode-engine-reload-anchor-{}.toml",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system time is valid")
                .as_nanos()
        ));
        fs::write(
            &path,
            r#"
            active_provider = "primary"

            [providers.primary]
            base_url = "https://example.invalid/v1"
            api_key = "config-key"
            protocol = "responses"
            default_model = "model-b"

            [providers.primary.models.model-a]
            [providers.primary.models.model-b]
            "#,
        )
        .expect("write config");

        let file_config = AppConfig::load_from_path(&path).expect("config should load");
        let provider = file_config
            .providers
            .get("primary")
            .expect("primary provider");
        let route = ModelRoute::new("primary", "model-b");
        let mut agent = Agent::new(
            Client::with_config(OpenAIConfig::new()),
            route.model.clone(),
            1,
            1,
        );
        agent.set_primary_route(route.clone());
        agent.set_default_protocol(provider.protocol);
        agent.set_model_protocols(
            provider
                .models
                .iter()
                .map(|(id, model)| (id.clone(), model.protocol))
                .collect(),
        );
        agent.set_model_catalog(
            provider
                .models
                .iter()
                .map(|(id, model)| (id.clone(), model.request_metadata()))
                .collect(),
        );
        let usage = crate::agent::TokenUsageEstimate {
            used_tokens: 42,
            context_window_tokens: 1000,
            input_tokens: 40,
            output_tokens: 2,
            cached_tokens: 0,
        };
        agent.install_provider_usage_anchor_for_test(usage.clone());

        let mut stale_providers = file_config.providers.clone();
        stale_providers
            .get_mut("primary")
            .expect("primary provider")
            .default_model = "model-a".into();
        let mut model_routes = file_config
            .providers
            .iter()
            .flat_map(|(provider_name, provider)| {
                provider.models.keys().map(move |model| {
                    let route = ModelRoute::new(provider_name, model);
                    (route.display_name(), route)
                })
            })
            .collect::<indexmap::IndexMap<_, _>>();
        let mut route_api_key_configured = model_routes
            .keys()
            .cloned()
            .map(|name| (name, true))
            .collect::<indexmap::IndexMap<_, _>>();
        let mut expert_model_routes = indexmap::IndexMap::new();
        let mut legacy_expert_models = indexmap::IndexMap::new();
        let mut global_retry = file_config.global.retry.clone();
        let mut provider_api_key_hints = indexmap::IndexMap::from([(
            String::from("primary"),
            format!(
                "Set providers.primary.api_key in {} or set {}.",
                file_config.config_path.display(),
                crate::config::provider_api_key_env_var("primary")
            ),
        )]);
        let (event_tx, mut event_rx) = mpsc::unbounded_channel();

        apply_config_reload(
            &mut agent,
            &path,
            &mut model_routes,
            &mut route_api_key_configured,
            &mut expert_model_routes,
            &mut legacy_expert_models,
            &mut stale_providers,
            &mut global_retry,
            &mut provider_api_key_hints,
            &event_tx,
        )
        .expect("default_model-only divergence should be a silent noop");

        assert_eq!(agent.provider_usage_anchor_for_test(), Some(usage));
        assert!(
            event_rx.try_recv().is_err(),
            "self-echo default_model sync must not emit reload notice"
        );
        assert_eq!(
            stale_providers["primary"].default_model, "model-a",
            "noop path must not rewrite engine maps"
        );

        let _ = fs::remove_file(path);
    }

    #[test]
    fn reload_updates_retry_without_clearing_usage_anchor() {
        let path = std::env::temp_dir().join(format!(
            "letcode-engine-reload-retry-{}.toml",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system time is valid")
                .as_nanos()
        ));
        fs::write(
            &path,
            r#"
            active_provider = "primary"

            [global.retry]
            enabled = false
            max_attempts = 1

            [providers.primary]
            base_url = "https://example.invalid/v1"
            api_key = "config-key"
            protocol = "responses"
            default_model = "primary-model"

            [providers.primary.models.primary-model]
            "#,
        )
        .expect("write retry config");

        let config = AppConfig::load_from_path(&path).expect("config should load");
        let provider = config.providers.get("primary").expect("primary provider");
        let route = ModelRoute::new("primary", "primary-model");
        let mut agent = Agent::new(
            Client::with_config(OpenAIConfig::new()),
            route.model.clone(),
            1,
            1,
        );
        agent.set_primary_route(route.clone());
        agent.set_default_protocol(provider.protocol);
        agent.set_model_protocols(
            provider
                .models
                .iter()
                .map(|(id, model)| (id.clone(), model.protocol))
                .collect(),
        );
        agent.set_model_catalog(
            provider
                .models
                .iter()
                .map(|(id, model)| (id.clone(), model.request_metadata()))
                .collect(),
        );
        let usage = crate::agent::TokenUsageEstimate {
            used_tokens: 7,
            context_window_tokens: 1000,
            input_tokens: 7,
            output_tokens: 0,
            cached_tokens: 0,
        };
        agent.install_provider_usage_anchor_for_test(usage.clone());
        assert!(agent.retry_config().enabled);

        let mut model_routes = indexmap::IndexMap::from([(route.display_name(), route.clone())]);
        let mut route_api_key_configured =
            indexmap::IndexMap::from([(route.display_name(), true)]);
        let mut expert_model_routes = indexmap::IndexMap::new();
        let mut legacy_expert_models = indexmap::IndexMap::new();
        let mut providers = config.providers.clone();
        let mut global_retry = RetryConfig::default();
        let mut provider_api_key_hints = indexmap::IndexMap::from([(
            String::from("primary"),
            format!(
                "Set providers.primary.api_key in {} or set {}.",
                config.config_path.display(),
                crate::config::provider_api_key_env_var("primary")
            ),
        )]);
        let (event_tx, mut event_rx) = mpsc::unbounded_channel();

        apply_config_reload(
            &mut agent,
            &path,
            &mut model_routes,
            &mut route_api_key_configured,
            &mut expert_model_routes,
            &mut legacy_expert_models,
            &mut providers,
            &mut global_retry,
            &mut provider_api_key_hints,
            &event_tx,
        )
        .expect("retry-only reload should succeed");

        assert!(!agent.retry_config().enabled);
        assert_eq!(agent.provider_usage_anchor_for_test(), Some(usage));
        assert!(matches!(
            event_rx.try_recv(),
            Ok(SessionTransportEvent::Notice(_))
        ));

        let _ = fs::remove_file(path);
    }

    #[tokio::test]
    async fn config_watcher_filters_unrelated_files_and_notifies_target_events() {
        let directory = std::env::temp_dir().join(format!(
            "letcode-engine-watcher-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system time is valid")
                .as_nanos()
        ));
        fs::create_dir_all(&directory).expect("create watcher directory");
        let config_path = directory.join("letcode.toml");
        let unrelated_path = directory.join("unrelated.toml");
        fs::write(&config_path, "active_provider = \"primary\"\n").expect("write target");
        let (reload_tx, mut reload_rx) = mpsc::unbounded_channel();
        let watcher = create_config_watcher(&config_path, reload_tx).expect("watch config parent");
        // macOS FSEvents can deliver the pre-watch create/write after the watcher
        // starts; drain that noise before asserting filter behavior.
        tokio::time::sleep(Duration::from_millis(100)).await;
        while reload_rx.try_recv().is_ok() {}

        fs::write(&unrelated_path, "unrelated = true\n").expect("write unrelated file");
        assert!(
            tokio::time::timeout(Duration::from_millis(250), reload_rx.recv())
                .await
                .is_err(),
            "unrelated file changes must not request reload"
        );

        fs::write(&config_path, "active_provider = \"primary\"\n").expect("write target file");
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(2), reload_rx.recv())
                .await
                .expect("target event should arrive before timeout")
                .expect("watcher channel should remain open"),
            ()
        );

        drop(watcher);
        let _ = fs::remove_dir_all(directory);
    }

    #[tokio::test]
    async fn join_without_started_tasks_completes_after_ingress_transfer() {
        let (mut engine, ingress, _egress) = SessionEngine::new();
        engine.ingress = Some(ingress);

        engine
            .request_shutdown()
            .expect("shutdown request succeeds");
        engine
            .join()
            .await
            .expect("join completes without a backend task");
    }
}
