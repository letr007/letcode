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

use anyhow::{Result, anyhow, bail};
#[cfg(test)]
use serde_json::json;
use tokio::task::JoinHandle;

use crate::agent::{Agent, AgentEvent, ConfiguredPrimaryRouteFactory, ManualCompactionOutcome};
use crate::agent_event_journal::persist_agent_event;
use crate::config::{AppConfig, ModelRoute, ProviderConfig, SubmitWhileRunning};
use crate::mcp;
use crate::runtime_context::RuntimeActiveContext;
use crate::session::runner::{ModelCatalogEntry, ModelCatalogReasoning, ModelCatalogUpdatedEvent};
use crate::session::{
    AgentRunner, ErrorEvent, NoticeEvent, NoticeId, RuntimeContextDisposition,
    RuntimeContextUpdatedEvent, SessionCommand, SessionEvent, SessionTransportEvent,
    TokenUsageEvent,
};
use crate::subagent::{SubagentJob, SubagentPool, SubagentStatus};
use crate::tool::ToolHandler;

mod config_reload;
mod control;

use crate::transcript::{
    ChildSessionSummary, TranscriptEvent, TranscriptRecorder, read_records,
    remove_empty_session_file, sync_recorder_branch, transcript_projection,
};
pub(crate) use config_reload::*;
pub(crate) use control::*;

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

    pub fn cancel_prompt(
        &self,
        submission_id: impl Into<String>,
    ) -> Result<(), SessionEngineIngressError> {
        self.send_control(SessionEngineControl::Command(
            SessionEngineCommand::CancelPrompt(submission_id.into()),
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
    pub workspace_dir: Option<PathBuf>,
    pub submit_while_running: crate::config::SubmitWhileRunning,
    /// Routes keyed by their provider-qualified display name (`provider/model`).
    pub model_routes: indexmap::IndexMap<String, ModelRoute>,
    /// Whether each route has a non-empty credential configured.
    pub route_api_key_configured: indexmap::IndexMap<String, bool>,
    /// Global default route used only when starting a new session.
    pub new_session_default_route: ModelRoute,
    pub new_session_subagents_policy: crate::config::SubagentsConfig,
    /// Resolved expert default routes derived from the configuration.
    pub expert_default_routes: indexmap::IndexMap<String, ModelRoute>,
    /// Provider-qualified routes allowed for per-invocation expert selection and takeover.
    pub expert_allowed_models: indexmap::IndexMap<String, Vec<ModelRoute>>,
    /// Reasoning level each expert declares in `[agents.<expert>]`.
    pub expert_reasoning_efforts:
        indexmap::IndexMap<String, crate::request_builder::ModelReasoningEffort>,
    /// Provider catalog used to reconstruct expert route factories after configuration updates.
    pub providers: indexmap::IndexMap<String, crate::config::ProviderConfig>,
    pub global_retry: crate::config::RetryConfig,
    /// Provider-specific API-key remediation keyed by provider name.
    pub provider_api_key_hints: indexmap::IndexMap<String, String>,
    /// Fallback remediation if a provider-specific hint is unavailable.
    pub api_key_hint: String,
    pub mcp_config_path: PathBuf,
    pub mcp_config: indexmap::IndexMap<String, crate::config::McpServerConfig>,
    pub runtime_catalog: crate::model_runtime::ResolvedRuntimeCatalog,
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
        let ingress = SessionEngineIngress {
            control_tx: control_tx.clone(),
        };
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
        mut agent: Agent,
        transcript: Arc<StdMutex<TranscriptRecorder>>,
        model_label: String,
        config: SessionEngineConfig,
    ) -> Result<(Self, SessionEngineProjection)> {
        // Bind the initial session before any turn can prepare history work.
        rehydrate_agent_from_transcript(&mut agent, &transcript)?;
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
        let ingress = SessionEngineIngress {
            control_tx: control_tx.clone(),
        };
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
        let child_live_text = crate::session::child_view::ChildLiveText::default();
        let (child_event_tx, child_event_rx) = mpsc::unbounded_channel();
        spawn_child_event_forwarder(child_event_rx, event_tx.clone());
        // The Jev backend records its reviews in a `reviewer` child session through
        // the same pool the expert backend uses. It is resolved here, where the
        // reviewer route is known and a client failure still reaches the caller
        // that reports it.
        let jev_reviewer = reviewer_jev_config(&agent, &config.providers)
            .map(|jev| {
                crate::session::jev_review::JevReviewer::new(
                    jev,
                    Arc::clone(&transcript),
                    Some(child_event_tx.clone()),
                    subagent_runtime.clone(),
                    config.sessions_dir.clone(),
                )
                .map(|reviewer| {
                    std::sync::Arc::new(reviewer)
                        as std::sync::Arc<dyn crate::agent::AutoReviewService>
                })
            })
            .transpose()?;
        let task = tokio::spawn(run_engine_loop(
            agent,
            Arc::clone(&transcript),
            config,
            jev_reviewer,
            mcp_tools_rx,
            reload_rx,
            control_rx,
            control_tx.clone(),
            event_tx.clone(),
            title_event_tx,
            subagent_runtime,
            child_live_text,
            child_event_tx,
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

    /// Join backend-owned tasks and run transcript cleanup.
    ///
    /// Cleanup is attempted even if either task panics. Any join or cleanup
    /// failure is returned after all owned resources have been reconciled.
    pub async fn join(mut self) -> Result<()> {
        let mut failure = None;
        // Stop filesystem callbacks before waiting for the engine and discovery
        // tasks so shutdown cannot enqueue work into a finished session.
        self.reload_watcher.take();

        if let Some(task) = self.engine_task.take()
            && let Err(error) = task.await
        {
            failure = Some(anyhow!("session engine task failed: {error}"));
        }
        if let Some(task) = self.mcp_discovery_task.take() {
            if !task.is_finished() {
                task.abort();
            }
            if let Err(error) = task.await
                && !error.is_cancelled()
                && failure.is_none()
            {
                failure = Some(anyhow!("MCP discovery task failed: {error}"));
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
            if let Err(error) = cleanup
                && failure.is_none()
            {
                failure = Some(error);
            }
        }

        match failure {
            Some(error) => Err(error),
            None => Ok(()),
        }
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
        control_tx: mpsc::UnboundedSender<SessionEngineControl>,
    ) -> (
        mpsc::UnboundedReceiver<SessionEngineControl>,
        mpsc::UnboundedSender<SessionEngineControl>,
        mpsc::UnboundedSender<SessionTransportEvent>,
    ) {
        (
            self.control_rx
                .expect("test engine control receiver unavailable"),
            control_tx,
            self.event_tx.expect("test engine event sender unavailable"),
        )
    }
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
            recorder.records().to_vec(),
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
    transcript_projection::project_runtime_active_context(
        session_id,
        records,
        transcript_projection::SessionContextCursor {
            branch_id: branch_id.map(str::to_string),
            leaf_sequence: None,
        },
    )
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
    pub(crate) active_runs: Vec<SubagentJob>,
    pub(crate) turn_id: Option<u64>,
    pub(crate) transcript_revision: u64,
    pub(crate) branch_id: Option<String>,
}

pub(crate) fn derive_interrupt_request(
    transcript: &Arc<StdMutex<TranscriptRecorder>>,
    subagent_runtime: &SubagentPool,
) -> Result<InterruptRequest> {
    let active_runs = subagent_runtime.active_jobs();
    let (transcript_revision, branch_id, turn_id, parent_tool_calls) = transcript
        .lock()
        .map_err(|_| anyhow!("transcript recorder poisoned"))?
        .interrupt_transcript_plan()?;

    Ok(InterruptRequest {
        parent_tool_calls,
        active_runs,
        turn_id,
        transcript_revision,
        branch_id,
    })
}

fn send_rehydrated_runtime_context(
    session_transport_tx: &mpsc::UnboundedSender<SessionTransportEvent>,
    agent: &Agent,
) -> Result<()> {
    let context = agent.runtime_context()?;
    let _ = session_transport_tx.send(SessionTransportEvent::RuntimeContextUpdated(
        RuntimeContextUpdatedEvent {
            context,
            disposition: RuntimeContextDisposition::Advance,
        },
    ));
    Ok(())
}

pub(crate) fn emit_child_terminal(
    session_transport_tx: &mpsc::UnboundedSender<SessionTransportEvent>,
    child_session_id: &str,
) {
    let _ = session_transport_tx.send(SessionTransportEvent::ChildSessionEvent {
        child_session_id: child_session_id.to_string(),
        agent_name: None,
        parent_tool_call_id: None,
        event: SessionEvent::Interrupted,
    });
}

/// Emits the child-scope terminal for a run whose executor was dropped by cancellation or timeout.
pub(crate) fn emit_dropped_child_terminal(
    session_transport_tx: &mpsc::UnboundedSender<SessionTransportEvent>,
    summary: &crate::subagent::SubagentRunSummary,
) {
    if matches!(
        summary.status,
        SubagentStatus::Cancelled | SubagentStatus::TimedOut
    ) {
        emit_child_terminal(session_transport_tx, &summary.child_session_id);
    }
}

/// Emits a child-scope terminal for every run whose executor was dropped before publishing one.
fn emit_dropped_child_terminals(
    session_transport_tx: &mpsc::UnboundedSender<SessionTransportEvent>,
    subagent_runtime: &SubagentPool,
    runs: &[SubagentJob],
    background_deliveries: &mut HashMap<String, BackgroundDelivery>,
) {
    for run in runs {
        // The historian runtime publishes its own terminal.
        if run.agent_name == "historian" {
            continue;
        }
        let dropped = subagent_runtime
            .completed_result(&run.run_id)
            .is_some_and(|summary| {
                matches!(
                    summary.status,
                    SubagentStatus::Cancelled | SubagentStatus::TimedOut
                )
            });
        if dropped
            && background_deliveries
                .entry(run.run_id.clone())
                .or_default()
                .mark_notified()
        {
            emit_child_terminal(session_transport_tx, &run.child_session_id);
        }
    }
}

pub(crate) fn send_subagent_interrupted(
    session_transport_tx: &mpsc::UnboundedSender<SessionTransportEvent>,
) {
    let _ = session_transport_tx.send(SessionTransportEvent::Interrupted);
}

pub(crate) fn record_interrupt_transcript(
    transcript: &Arc<StdMutex<TranscriptRecorder>>,
    interrupt: &InterruptRequest,
) -> Result<()> {
    let mut recorder = transcript
        .lock()
        .map_err(|_| anyhow!("transcript recorder poisoned"))?;
    let active_turn_id = recorder.active_turn_id()?;
    let identity_matches = recorder.current_context_branch_id().map(str::to_string)
        == interrupt.branch_id
        && active_turn_id == interrupt.turn_id;
    if !identity_matches {
        return Err(anyhow!(
            "interrupt transcript plan targets a different branch or turn: expected branch {:?}, turn {:?}; found branch {:?}, turn {:?}",
            interrupt.branch_id,
            interrupt.turn_id,
            recorder.current_context_branch_id(),
            active_turn_id,
        ));
    }
    let parent_tool_calls = if recorder.sequence == interrupt.transcript_revision {
        interrupt.parent_tool_calls.clone()
    } else {
        recorder.unfinished_tool_calls_in_active_turn()?
    };

    let mut events = parent_tool_calls
        .iter()
        .map(|(call_id, name)| {
            (
                TranscriptEvent::ToolCallCancelled {
                    call_id: call_id.clone(),
                    name: name.clone(),
                },
                interrupt.branch_id.clone(),
            )
        })
        .collect::<Vec<_>>();
    if let Some(turn_id) = interrupt.turn_id {
        events.push((
            TranscriptEvent::TurnInterrupted {
                turn_id: Some(turn_id),
            },
            interrupt.branch_id.clone(),
        ));
    }
    if events.is_empty() {
        return Ok(());
    }
    recorder.append_transaction(events)
}

pub(crate) fn close_failed_turn(
    agent: &mut Agent,
    transcript: &Arc<StdMutex<TranscriptRecorder>>,
    subagent_runtime: &SubagentPool,
) -> Result<()> {
    derive_interrupt_request(transcript, subagent_runtime)
        .and_then(|interrupt| record_interrupt_transcript(transcript, &interrupt))
        .and_then(|()| rehydrate_agent_from_transcript(agent, transcript))
}

pub(crate) fn rehydrate_agent_from_transcript(
    agent: &mut Agent,
    transcript: &Arc<StdMutex<TranscriptRecorder>>,
) -> Result<()>
where
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
    agent.restore_runtime_snapshot(snapshot.snapshot)?;
    agent.restore_turn_sequence(max_turn_id);
    sync_recorder_branch(&mut recorder, &branch_id);
    Ok(())
}

pub(crate) fn manual_compaction_session_token_usage(agent: &Agent) -> Result<TokenUsageEvent>
where
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

pub(crate) async fn run_manual_compaction(
    agent: &mut Agent,
    transcript: &Arc<StdMutex<TranscriptRecorder>>,
    session_transport_tx: &mpsc::UnboundedSender<SessionTransportEvent>,
    sessions_dir: &std::path::Path,
    control_rx: &mut mpsc::UnboundedReceiver<SessionEngineControl>,
    deferred_commands: &mut VecDeque<SessionEngineCommand>,
    visible_child_session_id: &mut Option<String>,
    visible_child_view_state: &mut Option<VisibleChildViewState>,
    child_live_text: &crate::session::child_view::ChildLiveText,
) -> bool
where
{
    let transcript = Arc::clone(transcript);
    // main/runner already install an equivalent transcript→snapshot provider.
    // Only fill it in when a direct Agent caller left the slot empty.
    if !agent.has_runtime_snapshot_provider() {
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
    }
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
                event @ (AgentEvent::HistoryPublished { .. }
                | AgentEvent::HistoryApplied { .. }) => {
                    let applied = matches!(&event, AgentEvent::HistoryApplied { .. });
                    let mut recorder = transcript
                        .lock()
                        .map_err(|_| anyhow::anyhow!("transcript recorder poisoned"))?;
                    persist_agent_event(&mut recorder, &event)?;
                    drop(recorder);
                    if applied {
                        compaction_persisted.store(true, Ordering::Release);
                        let _ = session_transport_tx
                            .send(SessionTransportEvent::CompactionCommitted { summary: None });
                    }
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
    // Drop the compaction future before reporting cancellation so a late
    // durable acknowledgement from a cancelled attempt cannot reach the UI.
    let compaction_result = {
        let compact = agent.compact_session_stream_async(on_event, &mut on_start);
        tokio::pin!(compact);
        loop {
            match select_manual_compaction_operation(
                control_rx,
                deferred_commands,
                compact.as_mut(),
            )
            .await
            {
                // Navigation only reads the transcript, so it is serviced without
                // disturbing the pinned compaction future or the agent borrow.
                ManualCompactionOperation::Navigation(navigation) => match navigation {
                    ManualCompactionNavigation::ViewChild {
                        navigation,
                        anchor_child_session_id,
                    } => {
                        apply_engine_view_child(
                            &transcript,
                            session_transport_tx,
                            Some(sessions_dir),
                            navigation,
                            anchor_child_session_id.as_deref(),
                            visible_child_session_id,
                            visible_child_view_state,
                            child_live_text,
                        );
                    }
                    ManualCompactionNavigation::ViewParent => {
                        crate::session::SessionCoordinator::emit_view_parent(
                            &transcript,
                            session_transport_tx,
                            Some(sessions_dir),
                        );
                        *visible_child_session_id = None;
                        *visible_child_view_state = None;
                    }
                },
                terminal => break terminal,
            }
        }
    };

    let shutdown = matches!(compaction_result, ManualCompactionOperation::Shutdown);
    match compaction_result {
        ManualCompactionOperation::Navigation(_) => {
            unreachable!("navigation is serviced while compaction stays pinned")
        }
        ManualCompactionOperation::Interrupted | ManualCompactionOperation::Shutdown => {
            if let Some(historian) = &agent.historian_runtime {
                historian.cancel();
            }
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
                let _ = session_transport_tx.send(SessionTransportEvent::CompactionCancelled);
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
pub(crate) struct VisibleChildViewState {
    record_count: usize,
    index: usize,
    total: usize,
}

fn apply_engine_view_child(
    transcript: &std::sync::Arc<std::sync::Mutex<crate::transcript::TranscriptRecorder>>,
    session_transport_tx: &tokio::sync::mpsc::UnboundedSender<
        crate::session::SessionTransportEvent,
    >,
    sessions_dir: Option<&std::path::Path>,
    navigation: crate::command::ChildNavigation,
    anchor_child_session_id: Option<&str>,
    visible_child_session_id: &mut Option<String>,
    visible_child_view_state: &mut Option<VisibleChildViewState>,
    child_live_text: &crate::session::child_view::ChildLiveText,
) {
    if navigation == crate::command::ChildNavigation::Toggle && visible_child_session_id.is_some() {
        crate::session::SessionCoordinator::emit_view_parent(
            transcript,
            session_transport_tx,
            sessions_dir,
        );
        *visible_child_session_id = None;
        *visible_child_view_state = None;
    } else {
        let effective_nav = match navigation {
            crate::command::ChildNavigation::Toggle => crate::command::ChildNavigation::First,
            other => other,
        };
        if let Some(child_session_id) = crate::session::SessionCoordinator::emit_view_child(
            transcript,
            session_transport_tx,
            sessions_dir,
            effective_nav,
            anchor_child_session_id,
            child_live_text,
        ) {
            *visible_child_session_id = Some(child_session_id);
            *visible_child_view_state = None;
        }
    }
}

/// Size and modification time of a journal file, read without opening it.
///
/// Journals only grow apart from tail repair, which moves either field anyway,
/// so an unchanged fingerprint means there is nothing new to read.
/// `TranscriptFileFingerprint` is the content digest of the same file, computed
/// from its bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct JournalFingerprint {
    len: u64,
    modified: Option<std::time::SystemTime>,
}

fn journal_fingerprint(path: &std::path::Path) -> Option<JournalFingerprint> {
    let metadata = std::fs::metadata(path).ok()?;
    Some(JournalFingerprint {
        len: metadata.len(),
        modified: metadata.modified().ok(),
    })
}

#[derive(Debug, Clone)]
struct VisibleChildViewResolution {
    parent_session_id: String,
    index: usize,
    total: usize,
    child: ChildSessionSummary,
    member_session_ids: Vec<String>,
}

#[derive(Debug, Clone)]
struct VisibleChildViewCache {
    parent_journal: Option<JournalFingerprint>,
    member_journals: Vec<(String, Option<JournalFingerprint>)>,
    resolution: VisibleChildViewResolution,
}

fn resolve_visible_child_view(
    transcript: &Arc<StdMutex<TranscriptRecorder>>,
    sessions_dir: &std::path::Path,
    child_session_id: &str,
) -> Result<Option<VisibleChildViewResolution>> {
    let parent_session_id = transcript
        .lock()
        .map_err(|_| anyhow::anyhow!("transcript recorder poisoned"))?
        .session_id()
        .to_string();
    let children =
        crate::session::child_view::project_session_members(sessions_dir, &parent_session_id)?;
    Ok(children
        .iter()
        .enumerate()
        .find(|(_, child)| child.child_session_id == child_session_id)
        .map(|(index, child)| VisibleChildViewResolution {
            parent_session_id,
            index,
            total: children.len(),
            child: child.clone(),
            member_session_ids: children
                .iter()
                .map(|child| child.child_session_id.clone())
                .collect(),
        }))
}

async fn refresh_visible_child_session_view(
    transcript: &Arc<StdMutex<TranscriptRecorder>>,
    session_transport_tx: &mpsc::UnboundedSender<SessionTransportEvent>,
    sessions_dir: &std::path::Path,
    visible_child_session_id: &mut Option<String>,
    visible_child_view_state: &mut Option<VisibleChildViewState>,
    visible_child_view_cache: &mut Option<VisibleChildViewCache>,
    child_live_text: &crate::session::child_view::ChildLiveText,
) {
    let Some(child_session_id) = visible_child_session_id.as_deref() else {
        return;
    };
    let member_journals: Vec<_> = visible_child_view_cache
        .as_ref()
        .map(|cache| {
            cache
                .member_journals
                .iter()
                .map(|(session_id, _)| {
                    (
                        session_id.clone(),
                        journal_fingerprint(&crate::transcript::child_session_path(
                            sessions_dir,
                            session_id,
                        )),
                    )
                })
                .collect()
        })
        .unwrap_or_default();
    let parent_journal = transcript
        .lock()
        .ok()
        .and_then(|recorder| journal_fingerprint(recorder.path()));
    let settled = visible_child_view_cache.as_ref().is_some_and(|cache| {
        cache.resolution.child.child_session_id == child_session_id
            && cache.member_journals == member_journals
            && cache.parent_journal == parent_journal
    });
    // A cleared view state is still owed a full pass.
    if settled && visible_child_view_state.is_some() {
        return;
    }
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

    let resolution = match visible_child_view_cache.as_ref().filter(|cache| {
        cache.resolution.child.child_session_id == child_session_id
            && cache.parent_journal == parent_journal
            && cache.member_journals == member_journals
    }) {
        Some(cache) => Some(cache.resolution.clone()),
        None => match resolve_visible_child_view(transcript, sessions_dir, child_session_id) {
            Ok(resolution) => resolution,
            Err(error) => {
                let _ = session_transport_tx.send(SessionTransportEvent::Error(ErrorEvent::new(
                    format!("failed to refresh child transcript: {error}"),
                )));
                return;
            }
        },
    };
    let Some(resolution) = resolution else {
        return;
    };
    let view_state = VisibleChildViewState {
        record_count: records.len(),
        index: resolution.index,
        total: resolution.total,
    };
    *visible_child_view_cache = Some(VisibleChildViewCache {
        parent_journal,
        member_journals: resolution
            .member_session_ids
            .iter()
            .map(|session_id| {
                let fingerprint = member_journals
                    .iter()
                    .find(|(id, _)| id == session_id)
                    .and_then(|(_, fingerprint)| *fingerprint);
                (session_id.clone(), fingerprint)
            })
            .collect(),
        resolution: resolution.clone(),
    });
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
    let in_progress_assistant_text = if resolution.child.status == "running" {
        child_live_text.get(&resolution.child.child_session_id)
    } else {
        child_live_text.clear(&resolution.child.child_session_id);
        None
    };
    let _ = session_transport_tx.send(SessionTransportEvent::ChildSessionViewed {
        parent_session_id: resolution.parent_session_id,
        child_session_id: resolution.child.child_session_id,
        agent_name: resolution.child.agent_name,
        index: resolution.index,
        total: resolution.total,
        pool_ordinal: resolution.child.pool_ordinal,
        records,
        runtime_context,
        in_progress_assistant_text,
    });
}

fn spawn_child_event_forwarder(
    mut child_event_rx: mpsc::UnboundedReceiver<SessionTransportEvent>,
    session_transport_tx: mpsc::UnboundedSender<SessionTransportEvent>,
) {
    tokio::spawn(async move {
        while let Some(event) = child_event_rx.recv().await {
            if session_transport_tx.send(event).is_err() {
                break;
            }
        }
    });
}

/// Typesafe Jev settings when the route the reviewer resolves to names a
/// provider that declares `reviewer = "jev"`.
fn reviewer_jev_config(
    agent: &Agent,
    providers: &indexmap::IndexMap<String, ProviderConfig>,
) -> Option<crate::config::JevReviewConfig> {
    // A route that cannot be resolved here is left to the chat backend, which
    // reports its own resolution failure as the denial reason.
    let route = crate::agent::AgentFactory::resolve_subagent_route(
        agent,
        &crate::agent::AgentTemplate::reviewer(),
        None,
        false,
    )
    .ok()?;
    crate::config::jev_review_for_route(providers, &route)
}

async fn replace_mcp_session(
    sessions: &mut HashMap<String, mcp::McpSession>,
    name: &str,
    session: Option<mcp::McpSession>,
) {
    if let Some(previous) = sessions.remove(name) {
        previous.shutdown().await;
    }
    if let Some(session) = session {
        sessions.insert(name.to_string(), session);
    }
}

fn refresh_peer_session(
    runtime: &crate::peer::PeerRuntime,
    transcript: &Arc<StdMutex<TranscriptRecorder>>,
    resumed: bool,
) -> Result<()> {
    let (session_id, title) = initial_session_metadata(transcript)?;
    if resumed {
        runtime.resume_session(session_id, title)
    } else {
        runtime.update_session(session_id, title)
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_engine_loop(
    agent: Agent,
    transcript: Arc<StdMutex<TranscriptRecorder>>,
    mut config: SessionEngineConfig,
    jev_reviewer: Option<std::sync::Arc<dyn crate::agent::AutoReviewService>>,
    mcp_tools_rx: mpsc::UnboundedReceiver<Vec<mcp::McpServerDiscovery>>,
    mut reload_rx: mpsc::UnboundedReceiver<()>,
    mut control_rx: mpsc::UnboundedReceiver<SessionEngineControl>,
    control_tx: mpsc::UnboundedSender<SessionEngineControl>,
    session_transport_tx: mpsc::UnboundedSender<SessionTransportEvent>,
    title_event_tx: mpsc::UnboundedSender<SessionTransportEvent>,
    subagent_runtime: SubagentPool,
    child_live_text: crate::session::child_view::ChildLiveText,
    child_event_tx: mpsc::UnboundedSender<SessionTransportEvent>,
) {
    let transcript = transcript;
    let mut agent = agent;
    let peer_control_tx = control_tx.clone();
    let peer_runtime = match async {
        let (session_id, title) = initial_session_metadata(&transcript)?;
        let config_dir = config
            .mcp_config_path
            .parent()
            .ok_or_else(|| anyhow!("configuration path has no parent directory"))?;
        let runtime = crate::peer::PeerRuntime::start(
            config_dir,
            config.workspace_dir.clone(),
            session_id,
            title,
            move |message| {
                let _ = peer_control_tx.send(SessionEngineControl::Command(
                    SessionEngineCommand::PeerMessage(message),
                ));
            },
        )
        .await?;
        crate::tool::peer::register(&mut agent, runtime.client())?;
        Ok::<_, anyhow::Error>(runtime)
    }
    .await
    {
        Ok(runtime) => runtime,
        Err(error) => {
            let _ = session_transport_tx.send(SessionTransportEvent::Error(ErrorEvent::new(
                format!("failed to start peer messaging: {error}"),
            )));
            return;
        }
    };
    let peer_client = peer_runtime.client();
    let (peer_title_tx, mut peer_title_rx) = mpsc::unbounded_channel();
    let mut mcp_tools_rx = Some(mcp_tools_rx);
    // The auto-review service reads both maps from outside the engine loop, so
    // they become shared handles here and leave the configuration value empty.
    let route_api_key_configured = Arc::new(StdMutex::new(std::mem::take(
        &mut config.route_api_key_configured,
    )));
    let provider_api_key_hints = Arc::new(StdMutex::new(std::mem::take(
        &mut config.provider_api_key_hints,
    )));
    let mut mcp_registered_tools: HashMap<String, Vec<String>> = HashMap::new();
    let mut mcp_sessions: HashMap<String, mcp::McpSession> = HashMap::new();
    let subagent_runtime = subagent_runtime;
    let auto_review_service: std::sync::Arc<dyn crate::agent::AutoReviewService> =
        match jev_reviewer {
            Some(service) => service,
            None => std::sync::Arc::new(crate::session::auto_review::StickyAutoReviewer::new(
                subagent_runtime.clone(),
                config.sessions_dir.clone(),
                Arc::clone(&transcript),
                Some(child_event_tx.clone()),
                Arc::clone(&route_api_key_configured),
                Arc::clone(&provider_api_key_hints),
                config.api_key_hint.clone(),
            )),
        };
    agent.set_auto_review_service(Some(std::sync::Arc::clone(&auto_review_service)));
    agent.historian_runtime = Some(Arc::new(
        crate::session::historian::HistorianRuntime::new(
            subagent_runtime.clone(),
            config.sessions_dir.clone(),
            Arc::clone(&transcript),
            session_transport_tx.clone(),
        )
        .with_live_text(child_live_text.clone()),
    ));
    let mut memory_worker = crate::project_memory::MemoryWorker::default();
    let memory_refresh_period = std::time::Duration::from_secs(30);
    let mut memory_refresh = tokio::time::interval_at(
        tokio::time::Instant::now() + memory_refresh_period,
        memory_refresh_period,
    );
    memory_refresh.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut deferred_commands = VecDeque::new();
    let mut parked_commands = VecDeque::new();
    let mut boundary_prompts = VecDeque::new();
    let mut change_notice_sent = false;
    let mut background_deliveries: HashMap<String, BackgroundDelivery> = HashMap::new();
    let mut visible_child_session_id = None;
    let mut visible_child_view_state = None;
    let mut visible_child_view_cache = None;
    let mut child_refresh = tokio::time::interval(std::time::Duration::from_millis(250));
    child_refresh.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    loop {
        tokio::select! {
            biased;
            Some(event) = peer_title_rx.recv() => {
                if let SessionTransportEvent::SessionTitleUpdated { session_id, title } = &event
                    && peer_client.info().address.session_id == *session_id
                    && let Err(error) = peer_runtime.update_session(session_id.clone(), Some(title.clone()))
                {
                    let _ = session_transport_tx.send(SessionTransportEvent::Error(ErrorEvent::new(
                        format!("failed to update peer session title: {error}"),
                    )));
                    break;
                }
                let _ = title_event_tx.send(event);
            }
            reload = reload_rx.recv() => {
                if reload.is_none() {
                    break;
                }
                while reload_rx.try_recv().is_ok() {}
                if let Some(historian) = &agent.historian_runtime { historian.cancel(); }
                let previous_expert_default_routes = config.expert_default_routes.clone();
                let previous_expert_allowed_models = config.expert_allowed_models.clone();
                let reload_config_path = config.mcp_config_path.clone();
                let reload_config = tokio::task::spawn_blocking(move || {
                    AppConfig::load_from_path(&reload_config_path)
                })
                .await;
                let reload_result = match reload_config {
                    Ok(Ok(reload_config)) => {
                        let mut route_keys = route_api_key_configured
                            .lock()
                            .unwrap_or_else(|error| error.into_inner());
                        let mut hints = provider_api_key_hints
                            .lock()
                            .unwrap_or_else(|error| error.into_inner());
                        apply_config_reload(
                            &mut agent,
                            &mut config,
                            reload_config,
                            &mut route_keys,
                            &mut hints,
                            &session_transport_tx,
                        )
                    }
                    Ok(Err(error)) => Err(error),
                    Err(error) => Err(anyhow!("configuration reload task failed: {error}")),
                };
                if let Err(error) = reload_result {
                    let _ = session_transport_tx.send(SessionTransportEvent::Error(ErrorEvent::new(
                        format!("failed to reload configuration: {error}"),
                    )));
                } else if reviewer_policy_changed(
                    &previous_expert_default_routes,
                    &config.expert_default_routes,
                    &previous_expert_allowed_models,
                    &config.expert_allowed_models,
                ) {
                    auto_review_service.clear_sticky();
                }
            }
            idle_operation = next_idle_session_operation(&mut control_rx, &mut deferred_commands) => {
                let Some(idle_operation) = idle_operation else {
                    break;
                };
                let command = match idle_operation {
                    IdleSessionOperation::Interrupt => {
                        subagent_runtime.cancel_active();
                        continue;
                    }
                    IdleSessionOperation::Command(command) => command,
                };

                if let SessionEngineCommand::SetExpertAllowedModels {
                    agent_name,
                    model_ids,
                } = &command
                {
                    if !crate::delegation::supported_agent_names().any(|name| name == agent_name) {
                        send_setting_change_failed(
                            &session_transport_tx,
                            crate::session::SessionCommand::SetExpertAllowedModels {
                                agent_name: agent_name.clone(),
                                model_ids: model_ids.clone(),
                            },
                            format!("unknown expert: {agent_name}"),
                        );
                        continue;
                    }
                    let mut routes = Vec::with_capacity(model_ids.len());
                    let mut seen = std::collections::HashSet::new();
                    let mut invalid = None;
                    for model_id in model_ids {
                        let Some(route) = config.model_routes.get(model_id).cloned() else {
                            invalid = Some(format!("unknown model: {model_id}"));
                            break;
                        };
                        if !seen.insert(route.display_name()) {
                            invalid = Some(format!("duplicate model: {}", route.display_name()));
                            break;
                        }
                        routes.push(route);
                    }
                    if let Some(error) = invalid {
                        send_setting_change_failed(
                            &session_transport_tx,
                            crate::session::SessionCommand::SetExpertAllowedModels {
                                agent_name: agent_name.clone(),
                                model_ids: model_ids.clone(),
                            },
                            error,
                        );
                        continue;
                    }
                    let mut updated_allowed_models = config.expert_allowed_models.clone();
                    updated_allowed_models.insert(agent_name.clone(), routes.clone());
                    let expert_factory = match crate::subagent::ExpertRouteFactory::new_with_policies(
                        crate::delegation::supported_agent_names().map(|name| {
                            (
                                name.to_string(),
                                config.expert_default_routes.get(name).cloned(),
                                updated_allowed_models.get(name).cloned().unwrap_or_default(),
                                config.expert_reasoning_efforts.get(name).cloned(),
                            )
                        }),
                        &config.providers,
                        &config.global_retry,
                    ) {
                        Ok(factory) => factory.with_runtime_catalog(config.runtime_catalog.clone()),
                        Err(error) => {
                            send_setting_change_failed(
                                &session_transport_tx,
                                crate::session::SessionCommand::SetExpertAllowedModels {
                                    agent_name: agent_name.clone(),
                                    model_ids: model_ids.clone(),
                                },
                                format!("failed to rebuild expert route factory: {error}"),
                            );
                            continue;
                        }
                    };
                    if let Err(error) = crate::config::persist_expert_allowed_models(
                        &config.mcp_config_path,
                        agent_name,
                        &routes,
                    ) {
                        send_setting_change_failed(
                            &session_transport_tx,
                            crate::session::SessionCommand::SetExpertAllowedModels {
                                agent_name: agent_name.clone(),
                                model_ids: model_ids.clone(),
                            },
                            format!("failed to persist expert allowed models: {error}"),
                        );
                        continue;
                    }
                    agent.set_subagent_child_factory(Arc::new(expert_factory));
                    config.expert_allowed_models = updated_allowed_models;
                    if agent_name == "historian"
                        && let Some(historian) = &agent.historian_runtime { historian.cancel(); }
                    if agent_name == "reviewer" {
                        auto_review_service.clear_sticky();
                    }
                    let _ = session_transport_tx.send(SessionTransportEvent::ExpertAllowedModelsChanged {
                        agent_name: agent_name.clone(),
                        model_ids: routes.iter().map(ModelRoute::display_name).collect(),
                    });
                    continue;
                }

                if let SessionEngineCommand::SetModel(model) = &command {
                    let Some(route) = config.model_routes.get(model).cloned() else {
                        send_setting_change_failed(
                            &session_transport_tx,
                            crate::session::SessionCommand::SetModel(model.clone()),
                            format!("unknown model: {model}"),
                        );
                        continue;
                    };
                    let model_id = route.display_name();
                    let resolved_route = match agent.resolved_model_route_for(&route) {
                        Some(resolved_route) => resolved_route,
                        None => {
                            send_setting_change_failed(
                                &session_transport_tx,
                                crate::session::SessionCommand::SetModel(model.clone()),
                                format!("resolved runtime route is unavailable: {}", route.display_name()),
                            );
                            continue;
                        }
                    };
                    let prepared_route = match agent.prepare_primary_route(route.clone()) {
                        Ok(prepared_route) => prepared_route,
                        Err(error) => {
                            send_setting_change_failed(
                                &session_transport_tx,
                                crate::session::SessionCommand::SetModel(model.clone()),
                                format!("failed to set model: {error}"),
                            );
                            continue;
                        }
                    };
                    match crate::session::settings::apply_model_route_with_authority(
                        &mut agent,
                        &transcript,
                        route.clone(),
                        resolved_route,
                        prepared_route,
                    ) {
                        Ok(fast_mode_auto_disabled) => {
                            if fast_mode_auto_disabled {
                                let _ = session_transport_tx.send(
                                    SessionTransportEvent::FastModeChanged { enabled: false },
                                );
                                let _ = session_transport_tx.send(SessionTransportEvent::Notice(
                                    NoticeEvent::info(
                                        "Fast mode auto-disabled: current model is unavailable",
                                    )
                                    .with_message_id(NoticeId::FastModeAutoDisabled),
                                ));
                            }
                            let _ = session_transport_tx.send(SessionTransportEvent::ModelChanged {
                                model_id: model_id.clone(),
                            });
                            if let Some(effort) = agent.reasoning_effort() {
                                let _ = session_transport_tx.send(
                                    SessionTransportEvent::ReasoningEffortChanged { effort },
                                );
                            }
                            let route_api_keys = route_api_key_configured
                                .lock()
                                .unwrap_or_else(|error| error.into_inner());
                            if !route_has_api_key(&route_api_keys, &model_id) {
                                let provider_hints = provider_api_key_hints
                                    .lock()
                                    .unwrap_or_else(|error| error.into_inner());
                                let _ = session_transport_tx.send(SessionTransportEvent::Error(
                                    missing_api_key_error(&route_api_key_hint(
                                        &model_id,
                                        &provider_hints,
                                        &config.api_key_hint,
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
                                    )
                                    .with_message_id(NoticeId::FastModeAutoDisabled),
                                ));
                            }
                            send_setting_change_failed(
                                &session_transport_tx,
                                crate::session::SessionCommand::SetModel(model.clone()),
                                format!("failed to set model: {error}"),
                            );
                        }
                    }
                    continue;
                }

                if let Some(session_command) = session_engine_command_as_idle_session_command(&command) {
                    if let SessionEngineCommand::ViewChild { navigation, anchor_child_session_id } = &command {
                        apply_engine_view_child(
                            &transcript,
                            &session_transport_tx,
                            Some(config.sessions_dir.as_path()),
                            navigation.clone(),
                            anchor_child_session_id.as_deref(),
                            &mut visible_child_session_id,
                            &mut visible_child_view_state,
                            &child_live_text,
                        );
                    } else {
                        let history_navigation = matches!(
                            command,
                            SessionEngineCommand::Undo
                                | SessionEngineCommand::Redo
                                | SessionEngineCommand::NavigateHistory { .. }
                        );
                        if history_navigation
                            && let Some(historian) = &agent.historian_runtime { historian.cancel(); }
                        let dispatch_result = crate::session::SessionCoordinator::dispatch_idle_command(
                            session_command,
                            &mut agent,
                            &transcript,
                            &session_transport_tx,
                            Some(config.sessions_dir.as_path()),
                            &child_live_text,
                        );
                        if let Err(error) = dispatch_result {
                            let _ = session_transport_tx.send(SessionTransportEvent::Error(
                                ErrorEvent::new(format!("failed to dispatch session command: {error}")),
                            ));
                            continue;
                        }
                        if matches!(command, SessionEngineCommand::ViewParent) {
                            visible_child_session_id = None;
                            visible_child_view_state = None;
                        }
                    }
                    continue;
                }

                if let SessionEngineCommand::BackgroundSubagentCompleted {
                    generation,
                    parent_session_id,
                    parent_tool_call_id,
                    result,
                } = command
                {
                    if generation != subagent_runtime.swarm_generation(&parent_session_id) { continue; }
                    if !observe_background_subagent_completion(
                        &transcript,
                        &subagent_runtime,
                        &mut background_deliveries,
                        &session_transport_tx,
                        &parent_session_id,
                        parent_tool_call_id,
                        &result,
                    ) {
                        continue;
                    }
                    let delivered_run = result.as_ref().ok().map(|result| result.run_id.clone());
                    let prompt = match result {
                        Ok(result) => {
                            if background_deliveries.get(&result.run_id).is_some_and(|delivery| delivery.delivered) {
                                continue;
                            }
                            if let Err(error) = agent.install_background_subagent_result(&result) {
                                let _ = session_transport_tx.send(SessionTransportEvent::Error(
                                    ErrorEvent::new(format!(
                                        "failed to install background subagent result: {error}"
                                    )),
                                ));
                                continue;
                            }
                            format_background_subagent_completion(&result)
                        }
                        Err(error) => format!(
                            "A background subagent failed before producing a structured result.\n\n{error}\n\nContinue the user's task and account for this failure."
                        ),
                    };
                    if let Err(error) = agent
                        .begin_internal_continuation_turn()
                        .and_then(|()| {
                            transcript
                                .lock()
                                .map_err(|_| anyhow!("transcript recorder poisoned"))
                                .and_then(|mut recorder| {
                                    recorder.record_internal_continuation(
                                        prompt.clone(),
                                        crate::transcript::InternalContinuationSource::SubagentCompletion,
                                    )
                                })
                                .and_then(|()| agent.append_internal_continuation(prompt))
                        })
                    {
                        let _ = session_transport_tx.send(SessionTransportEvent::Error(
                            ErrorEvent::new(format!(
                                "failed to record background subagent continuation: {error}"
                            )),
                        ));
                        continue;
                    }
                    deferred_commands.push_front(SessionEngineCommand::ContinueSession);
                    if let Some(run_id) = delivered_run {
                        background_deliveries.entry(run_id).or_default().mark_delivered();
                    }
                    continue;
                }

                let prompt = match command {
                    SessionEngineCommand::ToggleMcpServer(server_name) => {
                        let Some(server_config) = config.mcp_config.get(&server_name).cloned() else {
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
                            &config.mcp_config_path,
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
                        config.mcp_config.insert(server_name.clone(), persisted_config);
                        if !enabled {
                            for tool_name in mcp_registered_tools
                                .remove(&server_name)
                                .unwrap_or_default()
                            {
                                agent.unregister_tool(&tool_name);
                            }
                            replace_mcp_session(&mut mcp_sessions, &server_name, None).await;
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
                            config.mcp_config
                                .get(&server_name)
                                .expect("configured MCP server should remain present")
                                .clone(),
                        );
                        let mcp::McpServerDiscovery {
                            mut server,
                            session,
                            tools,
                        } = mcp::discover_servers(&one_server)
                            .await
                            .into_iter()
                            .next()
                            .expect("single MCP server discovery should return one result");
                        let mut catalog_tools = Vec::new();
                        match server.status {
                            mcp::McpServerStatus::Online { .. } => {
                                let mut registered = Vec::new();
                                for tool in tools {
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
                        replace_mcp_session(&mut mcp_sessions, &server.name, session).await;
                        let _ = session_transport_tx.send(SessionTransportEvent::McpServerToolsUpdated {
                            name: server.name.clone(),
                            tools: catalog_tools,
                        });
                        let _ = session_transport_tx.send(SessionTransportEvent::McpServerUpdated(server));
                        continue;
                    }
                    SessionEngineCommand::SwarmMessageReady { root_session_id, generation } => {
                        if subagent_runtime.swarm_generation(&root_session_id) != Some(generation) { continue; }
                        if !transcript.lock().is_ok_and(|recorder| recorder.session_id() == root_session_id) { continue; }
                        let message = match subagent_runtime.take_member_message(&root_session_id, &root_session_id) {
                            Ok(Some(message)) => message,
                            Ok(None) => continue,
                            Err(error) => { let _ = session_transport_tx.send(SessionTransportEvent::Error(ErrorEvent::new(error.to_string()))); continue; }
                        };
                        crate::user_content::UserMessageSubmission::new(crate::subagent::generate_message_id(), crate::user_content::UserMessageContent::from(message))
                    }
                    SessionEngineCommand::Prompt(prompt) => prompt,
                    SessionEngineCommand::PeerMessage(message) => {
                        if message.to != peer_client.info().address {
                            continue;
                        }
                        let current_session_id = match transcript.lock() {
                            Ok(recorder) => recorder.session_id().to_string(),
                            Err(_) => {
                                let _ = session_transport_tx.send(SessionTransportEvent::Error(
                                    ErrorEvent::new("transcript recorder poisoned"),
                                ));
                                break;
                            }
                        };
                        if message.to.session_id != current_session_id {
                            continue;
                        }
                        crate::user_content::UserMessageSubmission::new(
                            crate::subagent::generate_message_id(),
                            crate::user_content::UserMessageContent::from(format!(
                                "[Peer message from {}. The source is routing metadata.]\n{}",
                                serde_json::to_string(&message.from).expect("peer address serializes"),
                                message.message,
                            )),
                        )
                    }
                    SessionEngineCommand::CancelQueuedPrompt(id) => {
                        cancel_queued_prompt(&mut deferred_commands, &id);
                        continue;
                    }
                    SessionEngineCommand::CancelPrompt(id) => {
                        if cancel_queued_prompt(&mut deferred_commands, &id) {
                            let _ = session_transport_tx.send(SessionTransportEvent::PromptCancelled {
                                submission_id: id,
                            });
                        }
                        continue;
                    }
                    SessionEngineCommand::ContinueSession => {
                        crate::user_content::UserMessageSubmission::new(
                            "internal-continuation",
                            crate::user_content::UserMessageContent::default(),
                        )
                    }
                    SessionEngineCommand::BackgroundSubagentCompleted { .. } => {
                        unreachable!("background completion was handled above")
                    }
                    SessionEngineCommand::ShowHistoryTree
                    | SessionEngineCommand::Undo
                    | SessionEngineCommand::Redo
                    | SessionEngineCommand::NavigateHistory { .. }
                    | SessionEngineCommand::SetPermissionMode(_)
                    | SessionEngineCommand::SetModel(_)
                    | SessionEngineCommand::SetExpertAllowedModels { .. }
                    | SessionEngineCommand::ToggleFastMode
                    | SessionEngineCommand::SetReasoningEffort(_)
                    | SessionEngineCommand::SetFakeClient(_)
                    | SessionEngineCommand::ViewChild { .. }
                    | SessionEngineCommand::ViewParent => {
                        // Idle commands are handled above via SessionCoordinator.
                        continue;
                    }
                    SessionEngineCommand::Compact => {
                        let _ = session_transport_tx.send(SessionTransportEvent::ManualCompactionStarted);
                        let active_route_has_credentials = {
                            let route_api_keys = route_api_key_configured
                                .lock()
                                .unwrap_or_else(|error| error.into_inner());
                            active_route_has_api_key(&agent, &route_api_keys)
                        };
                        if !active_route_has_credentials {
                            let route_display_name = agent.route_display_name();
                            {
                                let provider_hints = provider_api_key_hints
                                    .lock()
                                    .unwrap_or_else(|error| error.into_inner());
                                send_missing_api_key_error(
                                    &session_transport_tx,
                                    &route_display_name,
                                    &provider_hints,
                                    &config.api_key_hint,
                                );
                            }
                            continue;
                        }
                        if subagent_runtime.is_running() {
                            let _ = session_transport_tx.send(SessionTransportEvent::Notice(
                                NoticeEvent::info(
                                    "Wait for the active subagent to finish before compacting context",
                                )
                                .with_message_id(NoticeId::WaitForSubagentBeforeCompaction),
                            ));
                            let _ = session_transport_tx.send(SessionTransportEvent::Done);
                            continue;
                        }

                        let shutdown = run_manual_compaction(
                            &mut agent,
                            &transcript,
                            &session_transport_tx,
                            &config.sessions_dir,
                            &mut control_rx,
                            &mut deferred_commands,
                            &mut visible_child_session_id,
                            &mut visible_child_view_state,
                            &child_live_text,
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
                        if let Some(historian) = &agent.historian_runtime { historian.cancel(); }

                        let session_id = match crate::session::resolve_session_prefix(
                            &config.sessions_dir,
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
                        let current_session_id = transcript
                            .lock()
                            .ok()
                            .map(|recorder| recorder.session_id().to_string());
                        if current_session_id.as_deref() == Some(session_id.as_str())
                            && subagent_runtime.is_running()
                        {
                            let _ = session_transport_tx.send(SessionTransportEvent::Error(
                                ErrorEvent::new(
                                    "Wait for the active subagent to finish before reloading the current session",
                                ),
                            ));
                            continue;
                        }
                        if current_session_id.as_deref() == Some(session_id.as_str())
                            && agent.subagents_policy().mode == crate::config::SubagentMode::Swarm {
                            subagent_runtime.close_swarm_sessions();
                            if !subagent_runtime.wait_until_idle().await {
                                let _ = session_transport_tx.send(SessionTransportEvent::Error(ErrorEvent::new("failed to settle swarm before reloading the current session")));
                                continue;
                            }
                            let restored = (|| -> Result<SessionTransportEvent> {
                                rehydrate_agent_from_transcript(&mut agent, &transcript)?;
                                let recorder = transcript.lock().map_err(|_| anyhow!("transcript recorder poisoned"))?;
                                let records = read_records(recorder.path())?;
                                let snapshot = crate::session::restore::project_runtime_restore_snapshot_with_children(
                                    session_id.clone(), records.clone(),
                                    crate::transcript::transcript_projection::SessionContextCursor { branch_id: recorder.current_context_branch_id().map(str::to_string), leaf_sequence: None }, &config.sessions_dir)?;
                                drop(recorder);
                                let runtime_context = RuntimeActiveContext::try_from(&snapshot.snapshot)?;
                                subagent_runtime.resume_swarm(&config.sessions_dir, &records)?;
                                let runner = AgentRunner::with_transcript(session_transport_tx.clone(), transcript.clone())
                                    .with_child_live_text(child_live_text.clone())
                                    .with_subagent_runtime(subagent_runtime.clone(), config.sessions_dir.clone(),
                                        route_api_key_configured.lock().unwrap_or_else(|error| error.into_inner()).clone(),
                                        provider_api_key_hints.lock().unwrap_or_else(|error| error.into_inner()).clone(),
                                        config.api_key_hint.clone(), Some(control_tx.clone()), Some(session_transport_tx.clone()));
                                runner.prepare_subagent_runtime(&mut agent)?;
                                auto_review_service.clear_sticky();
                                Ok(SessionTransportEvent::SessionResumed {
                                    session_id: session_id.clone(), branch_id: snapshot.branch_id,
                                    messages: crate::session::restore::restored_messages_from_protocol_frames(&snapshot.snapshot.active_protocol_frames()),
                                    records, evidence_count: snapshot.snapshot.evidence.len(), model_id: Some(agent.route_display_name()), token_usage: None,
                                    runtime_context,
                                    expert_models: config.expert_default_routes.iter().map(|(name, route)| (name.clone(), route.display_name())).collect(),
                                })
                            })();
                            match restored {
                                Ok(event) => {
                                    if let Err(error) = refresh_peer_session(&peer_runtime, &transcript, true) {
                                        let _ = session_transport_tx.send(SessionTransportEvent::Error(ErrorEvent::new(error.to_string())));
                                        break;
                                    }
                                    let _ = session_transport_tx.send(event);
                                }
                                Err(error) => {
                                    subagent_runtime.close_swarm_sessions();
                                    let _ = session_transport_tx.send(SessionTransportEvent::Error(ErrorEvent::new(format!("failed to reload current swarm session: {error}"))));
                                }
                            }
                            continue;
                        }
                        let prepared = match crate::session::prepare_resume_package(
                            &config.sessions_dir,
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
                        let prepared_install =
                            match crate::session::restore::prepare_routed_resume_install(
                                &agent,
                                &transcript,
                                prepared,
                                Some(&config.new_session_default_route),
                            ) {
                                Ok(prepared) => prepared,
                                Err(error) => {
                                    let _ = session_transport_tx.send(SessionTransportEvent::Error(
                                        ErrorEvent::new(format!(
                                            "failed to prepare resumed session: {error}"
                                        )),
                                    ));
                                    continue;
                                }
                            };
                        let resumed_event_session_id = prepared_install.session().session_id.clone();
                        let resumed_event_branch_id = prepared_install.session().snapshot.branch_id.clone();
                        let resumed_event_messages =
                            crate::session::restore::restored_messages_from_protocol_frames(
                                &prepared_install.session().snapshot.snapshot.active_protocol_frames(),
                            );
                        let resumed_event_records = prepared_install.session().snapshot.records.clone();
                        let resumed_event_evidence_count =
                            prepared_install.session().snapshot.snapshot.evidence.len();
                        let fast_mode_auto_disabled = prepared_install.fast_mode_auto_disabled();
                        let resumed_permission_mode = prepared_install
                            .session()
                            .snapshot
                            .latest_permission_mode
                            .is_some();
                        let token_usage = {
                            subagent_runtime.close_swarm_sessions();
                            if !subagent_runtime.wait_until_idle().await {
                                let _ = session_transport_tx.send(SessionTransportEvent::Error(
                                    ErrorEvent::new("failed to cancel active subagents before resuming session"),
                                ));
                                continue;
                            }
                            prepared_install.commit(&mut agent, &transcript).1
                        };
                        if agent.subagents_policy().mode == crate::config::SubagentMode::Swarm {
                            if let Err(error) = subagent_runtime.resume_swarm(&config.sessions_dir, &resumed_event_records) {
                                let _ = session_transport_tx.send(SessionTransportEvent::Error(ErrorEvent::new(format!("failed to open resumed swarm: {error}"))));
                                continue;
                            }
                            let runner = AgentRunner::with_transcript(session_transport_tx.clone(), transcript.clone())
                                .with_child_live_text(child_live_text.clone())
                                .with_subagent_runtime(subagent_runtime.clone(), config.sessions_dir.clone(),
                                    route_api_key_configured.lock().unwrap_or_else(|error| error.into_inner()).clone(),
                                    provider_api_key_hints.lock().unwrap_or_else(|error| error.into_inner()).clone(),
                                    config.api_key_hint.clone(), Some(control_tx.clone()), Some(session_transport_tx.clone()));
                            if let Err(error) = runner.prepare_subagent_runtime(&mut agent) {
                                subagent_runtime.close_swarm_sessions();
                                let _ = session_transport_tx.send(SessionTransportEvent::Error(ErrorEvent::new(format!("failed to prepare resumed swarm: {error}"))));
                                continue;
                            }
                        }
                        if fast_mode_auto_disabled {
                            let _ = session_transport_tx.send(SessionTransportEvent::FastModeChanged { enabled: false });
                            let _ = session_transport_tx.send(SessionTransportEvent::Notice(
                                NoticeEvent::info(
                                    "Fast mode auto-disabled: current model is unavailable",
                                )
                                .with_message_id(NoticeId::FastModeAutoDisabled),
                            ));
                        }
                        let restored_fake_client =
                            crate::transcript::restore_latest_fake_client(&resumed_event_records);
                        agent.set_fake_client(restored_fake_client);
                        let _ = session_transport_tx.send(
                            SessionTransportEvent::FakeClientChanged {
                                client: restored_fake_client,
                            },
                        );
                        if let Some(historian) = &agent.historian_runtime { historian.cancel(); }
                        auto_review_service.clear_sticky();
                        // Resuming can install a recorded permission mode, and that mode may
                        // differ from the one this process runs with. Frontends build their
                        // resume report from `SessionResumed`, so the mode is announced before
                        // it — otherwise a client would show the pre-resume mode until the
                        // next edit. A session that recorded no mode keeps the live one.
                        if resumed_permission_mode {
                            let _ = session_transport_tx.send(
                                SessionTransportEvent::PermissionModeChanged {
                                    mode: agent.permission_mode().to_string(),
                                },
                            );
                        }
                        if let Err(error) = refresh_peer_session(&peer_runtime, &transcript, true) {
                            let _ = session_transport_tx.send(SessionTransportEvent::Error(ErrorEvent::new(error.to_string())));
                            break;
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
                            expert_models: config.expert_default_routes
                                .iter()
                                .map(|(name, route)| (name.clone(), route.display_name()))
                                .collect(),
                        });
                        if let Some(effort) = agent.reasoning_effort() {
                            let _ = session_transport_tx
                                .send(SessionTransportEvent::ReasoningEffortChanged { effort });
                        }
                        continue;
                    }
                    SessionEngineCommand::NewSession => {
                        if let Some(historian) = &agent.historian_runtime { historian.cancel(); }
                        let prepared_new_route = if let Some(current_route) = agent.primary_route().cloned() {
                            match agent.prepare_primary_route(current_route.clone()) {
                                Ok(route) => Ok((current_route, route)),
                                Err(current_error) => match agent
                                    .prepare_primary_route(config.new_session_default_route.clone())
                                {
                                    Ok(route) => Ok((config.new_session_default_route.clone(), route)),
                                    Err(default_error) => Err(anyhow!(
                                        "failed to prepare the current model ({current_error}); fallback default model also failed ({default_error})"
                                    )),
                                },
                            }
                        } else {
                            agent
                                .prepare_primary_route(config.new_session_default_route.clone())
                                .map(|route| (config.new_session_default_route.clone(), route))
                        };
                        let (new_session_route, prepared_route) = match prepared_new_route {
                            Ok(prepared) => prepared,
                            Err(error) => {
                                let _ = session_transport_tx.send(SessionTransportEvent::Error(
                                    ErrorEvent::new(format!(
                                        "failed to prepare a model for a new session: {error}"
                                    )),
                                ));
                                continue;
                            }
                        };
                        let prepared = match crate::session::prepare_new_session_package(
                            &config.sessions_dir,
                            new_session_route.display_name(),
                            config.workspace_dir.as_deref(),
                            &config.new_session_subagents_policy,
                        ) {
                            Ok(prepared) => prepared,
                            Err(error) => {
                                let _ = session_transport_tx.send(SessionTransportEvent::Error(ErrorEvent::new(format!(
                                    "failed to create session transcript: {error}"
                                ))));
                                continue;
                            }
                        };
                        let prepared_install =
                            match crate::session::lifecycle::prepare_new_session_install_with_route(
                                &agent,
                                &transcript,
                                prepared,
                                Some(prepared_route),
                            ) {
                                Ok(prepared) => prepared,
                                Err(error) => {
                                    let _ = session_transport_tx.send(SessionTransportEvent::Error(
                                        ErrorEvent::new(format!(
                                            "failed to prepare new session: {error}"
                                        )),
                                    ));
                                    continue;
                                }
                            };
                        let started_event = SessionTransportEvent::SessionStarted {
                            session_id: prepared_install.session().session_id.clone(),
                            records: prepared_install.session().snapshot.records.clone(),
                            runtime_context: prepared_install.session().runtime_context.clone(),
                            expert_models: config.expert_default_routes
                                .iter()
                                .map(|(name, route)| (name.clone(), route.display_name()))
                                .collect(),
                        };
                        subagent_runtime.close_swarm_sessions();
                        if !subagent_runtime.wait_until_idle().await {
                            let _ = session_transport_tx.send(SessionTransportEvent::Error(
                                ErrorEvent::new("failed to cancel active subagents before starting new session"),
                            ));
                            let _ = remove_empty_session_file(prepared_install.new_path());
                            continue;
                        }
                        let new_session_fake_client = agent.fake_client();
                        prepared_install.commit(&mut agent, &transcript);
                        agent.set_fake_client(new_session_fake_client);
                        let inherited_reasoning_effort = agent.reasoning_effort();
                        if let Err(error) = transcript
                            .lock()
                            .map_err(|_| anyhow!("transcript recorder poisoned"))
                            .and_then(|mut recorder| {
                                recorder.record_fake_client_changed(None, new_session_fake_client)?;
                                if let Some(effort) = inherited_reasoning_effort.clone() {
                                    recorder.record_reasoning_effort_changed(
                                        agent.route_display_name(),
                                        effort,
                                    )?;
                                }
                                Ok(())
                            })
                        {
                            let _ = session_transport_tx.send(SessionTransportEvent::Error(
                                ErrorEvent::new(format!(
                                    "failed to record inherited session settings: {error}"
                                )),
                            ));
                        }
                        let _ = session_transport_tx.send(
                            SessionTransportEvent::FakeClientChanged {
                                client: new_session_fake_client,
                            },
                        );
                        let new_session_model_id = agent.route_display_name();
                        if let Some(historian) = &agent.historian_runtime { historian.cancel(); }
                        auto_review_service.clear_sticky();
                        if let Err(error) = refresh_peer_session(&peer_runtime, &transcript, false) {
                            let _ = session_transport_tx.send(SessionTransportEvent::Error(ErrorEvent::new(error.to_string())));
                            break;
                        }
                        let _ = session_transport_tx.send(started_event);
                        let _ = session_transport_tx.send(SessionTransportEvent::ModelChanged {
                            model_id: new_session_model_id,
                        });
                        if let Some(effort) = inherited_reasoning_effort {
                            let _ = session_transport_tx
                                .send(SessionTransportEvent::ReasoningEffortChanged { effort });
                        }
                        continue;
                    }
                };

                let _ = session_transport_tx.send(SessionTransportEvent::QueuedPromptAccepted {
                    prompt: prompt.clone(),
                });

                let active_route_has_credentials = {
                    let route_api_keys = route_api_key_configured
                        .lock()
                        .unwrap_or_else(|error| error.into_inner());
                    active_route_has_api_key(&agent, &route_api_keys)
                };
                if !active_route_has_credentials {
                    if let Ok(recorder) = transcript.lock() {
                        let root = recorder.session_id().to_string();
                        drop(recorder);
                        subagent_runtime.fail_member_messages(&root, &root, "route credentials unavailable", false);
                    }
                    let route_display_name = agent.route_display_name();
                    {
                        let provider_hints = provider_api_key_hints
                            .lock()
                            .unwrap_or_else(|error| error.into_inner());
                        send_missing_api_key_error(
                            &session_transport_tx,
                            &route_display_name,
                            &provider_hints,
                            &config.api_key_hint,
                        );
                    }
                    continue;
                }

                let _ = session_transport_tx.send(SessionTransportEvent::TurnStarted);

                let turn_continuation_queue = Arc::new(StdMutex::new(
                    crate::agent::TurnContinuationQueue::default(),
                ));
                let (runner_event_tx, mut runner_event_rx) = mpsc::unbounded_channel();
                let runner = AgentRunner::with_transcript(
                    runner_event_tx,
                    transcript.clone(),
                )
                    .with_session_title_event_sender(peer_title_tx.clone())
                    .with_child_live_text(child_live_text.clone())
                    .with_subagent_runtime(
                        subagent_runtime.clone(),
                        config.sessions_dir.clone(),
                        route_api_key_configured
                            .lock()
                            .unwrap_or_else(|error| error.into_inner())
                            .clone(),
                        provider_api_key_hints
                            .lock()
                            .unwrap_or_else(|error| error.into_inner())
                            .clone(),
                        config.api_key_hint.clone(),
                        Some(control_tx.clone()),
                        Some(session_transport_tx.clone()),
                    );
                let historian_control = agent.historian_runtime.clone();
                let active_submission_id = prompt.id.clone();
                let (
                    interrupted,
                    shutdown,
                    interrupt_failure,
                    turn_failure,
                    boundary_preempted,
                    inserted_prompt,
                ) = {
                    let run: std::pin::Pin<
                        Box<dyn std::future::Future<Output = Result<String>> + Send + '_>,
                    > =
                        if prompt.content.is_empty() {
                            Box::pin(runner.run_existing_history_with_continuations(
                                &mut agent,
                                Arc::clone(&turn_continuation_queue),
                            ))
                        } else {
                            Box::pin(runner.run_prompt_with_continuations(
                                &mut agent,
                                prompt,
                                Arc::clone(&turn_continuation_queue),
                            ))
                        };
                    tokio::pin!(run);
                    let mut interrupted = None;
                    let mut shutdown = false;
                    let mut interrupt_failure = None;
                    let mut turn_failure = None;
                    let mut boundary_preempted = false;
                    let mut inserted_prompt = None;

                    loop {
                        let operation = tokio::select! {
                            Some(event) = peer_title_rx.recv() => {
                                if let SessionTransportEvent::SessionTitleUpdated { session_id, title } = &event
                                    && peer_client.info().address.session_id == *session_id
                                    && let Err(error) = peer_runtime.update_session(session_id.clone(), Some(title.clone()))
                                {
                                    interrupt_failure = Some(format!("failed to update peer session title: {error}"));
                                    break;
                                }
                                let _ = title_event_tx.send(event);
                                continue;
                            }
                            operation = select_active_session_operation_with_events(
                                &mut control_rx,
                                &mut deferred_commands,
                                run.as_mut(),
                                Some(&mut runner_event_rx),
                            ) => operation,
                        };
                        let targeted_cancel = matches!(&operation,
                            ActiveSessionOperation::Command(Some(SessionEngineCommand::CancelPrompt(id)))
                                if *id == active_submission_id);
                        let operation = match operation {
                            ActiveSessionOperation::Command(Some(SessionEngineCommand::CancelPrompt(id))) => {
                                if targeted_cancel {
                                    ActiveSessionOperation::Interrupted
                                } else {
                                    let parked = cancel_queued_prompt(&mut parked_commands, &id);
                                    let deferred = cancel_queued_prompt(&mut deferred_commands, &id);
                                    let boundary = cancel_queued_prompt(&mut boundary_prompts, &id);
                                    if parked || deferred || boundary {
                                        let _ = session_transport_tx.send(SessionTransportEvent::PromptCancelled {
                                            submission_id: id,
                                        });
                                    }
                                    continue;
                                }
                            }
                            operation => operation,
                        };
                        match operation {
                            outcome @ (ActiveSessionOperation::Interrupted
                            | ActiveSessionOperation::Shutdown) => {
                                let is_shutdown =
                                    matches!(outcome, ActiveSessionOperation::Shutdown);
                                if let Some(historian) = &historian_control { historian.cancel(); }
                                // Capture the interrupt request while the subagent is
                                // still active so the visible child session can be
                                // reported. Then signal cancellation and poll the run
                                // until the in-flight subagent's completion teardown
                                // (cancelled terminal record, guard release) settles.
                                let interrupt = match derive_interrupt_request(
                                    &transcript,
                                    &subagent_runtime,
                                ) {
                                    Ok(interrupt) => interrupt,
                                    Err(error) => {
                                        interrupt_failure = Some(format!(
                                            "failed to derive interrupt transcript plan: {error}"
                                        ));
                                        shutdown = true;
                                        break;
                                    }
                                };
                                if !targeted_cancel {
                                    discard_queued_peer_messages(&mut parked_commands);
                                    discard_queued_peer_messages(&mut boundary_prompts);
                                }
                                interrupted = Some(interrupt);
                                if is_shutdown { subagent_runtime.cancel_active(); }
                                if is_shutdown && subagent_runtime.is_running() {
                                    let settlement = wait_for_subagent_cancel_settle(
                                        &mut control_rx,
                                        &mut deferred_commands,
                                        run.as_mut(),
                                        &subagent_runtime,
                                        SUBAGENT_CANCEL_SETTLE_TIMEOUT,
                                    )
                                    .await;
                                    shutdown = is_shutdown || settlement.shutdown;
                                    if !settlement.settled {
                                        let residual_runs = settlement.remaining_run_ids;
                                        let detail = if residual_runs.is_empty() { "pending member recovery owners".to_string() } else { format!("run_ids: {}", residual_runs.join(", ")) };
                                        let _ = session_transport_tx.send(SessionTransportEvent::Error(ErrorEvent::new(format!("subagent cancellation did not settle within the bounded wait; remaining {detail}"))));
                                    }
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
                                let at_boundary = config.submit_while_running
                                    == SubmitWhileRunning::Insert
                                    && matches!(&event, SessionTransportEvent::ToolBatchFinished)
                                    && !boundary_prompts.is_empty();
                                let _ = session_transport_tx.send(event);
                                if at_boundary
                                    && let Some(prompt) = boundary_prompts.pop_front()
                                {
                                    inserted_prompt = Some(prompt);
                                    boundary_preempted = true;
                                    let _ = session_transport_tx.send(
                                        SessionTransportEvent::AssistantDone { message_id: None },
                                    );
                                    break;
                                }
                            }
                            ActiveSessionOperation::Completed(result) => {
                                turn_failure = result.err().map(|error| error.to_string());
                                forward_queued_runner_events(
                                    &mut runner_event_rx,
                                    &session_transport_tx,
                                );
                                let _ = session_transport_tx.send(SessionTransportEvent::Done);
                                break;
                            }
                            ActiveSessionOperation::Command(command) => match command {
                                Some(command @ SessionEngineCommand::SwarmMessageReady { .. }) => {
                                    parked_commands.push_back(command);
                                }
                                Some(command @ (SessionEngineCommand::Prompt(_)
                                    | SessionEngineCommand::PeerMessage(_)))
                                    if config.submit_while_running == SubmitWhileRunning::Insert =>
                                {
                                    boundary_prompts.push_back(command);
                                }
                                Some(command @ (SessionEngineCommand::Prompt(_)
                                    | SessionEngineCommand::PeerMessage(_))) => {
                                    parked_commands.push_back(command);
                                }
                                Some(SessionEngineCommand::CancelQueuedPrompt(id)) => {
                                    cancel_queued_prompt(&mut parked_commands, &id);
                                    cancel_queued_prompt(&mut deferred_commands, &id);
                                    cancel_queued_prompt(&mut boundary_prompts, &id);
                                }
                                Some(SessionEngineCommand::ViewChild {
                                    navigation,
                                    anchor_child_session_id,
                                }) => {
                                    apply_engine_view_child(
                                        &transcript,
                                        &session_transport_tx,
                                        Some(config.sessions_dir.as_path()),
                                        navigation,
                                        anchor_child_session_id.as_deref(),
                                        &mut visible_child_session_id,
                                        &mut visible_child_view_state,
                                        &child_live_text,
                                    );
                                }
                                Some(SessionEngineCommand::ViewParent) => {
                                    crate::session::SessionCoordinator::emit_view_parent(
                                        &transcript,
                                        &session_transport_tx,
                                        Some(config.sessions_dir.as_path()),
                                    );
                                    visible_child_session_id = None;
                                    visible_child_view_state = None;
                                }
                                Some(SessionEngineCommand::BackgroundSubagentCompleted {
                                    generation,
                                    parent_session_id,
                                    parent_tool_call_id,
                                    result,
                                }) => {
                                    if generation != subagent_runtime.swarm_generation(&parent_session_id) { continue; }
                                    if !observe_background_subagent_completion(
                                        &transcript,
                                        &subagent_runtime,
                                        &mut background_deliveries,
                                        &session_transport_tx,
                                        &parent_session_id,
                                        parent_tool_call_id,
                                        &result,
                                    ) {
                                        continue;
                                    }
                                    let delivered_run = result.as_ref().ok().map(|result| result.run_id.clone());
                                    let (text, continuation) = match result {
                                        Ok(result) => {
                                            if background_deliveries.get(&result.run_id).is_some_and(|delivery| delivery.delivered) {
                                                continue;
                                            }
                                            (
                                                format_background_subagent_completion(&result),
                                                crate::agent::PendingTurnContinuation {
                                                    result: Some(result),
                                                },
                                            )
                                        }
                                        Err(error) => (
                                            format!(
                                                "A background subagent failed before producing a structured result.\n\n{error}\n\nContinue the user's task and account for this failure."
                                            ),
                                            crate::agent::PendingTurnContinuation { result: None },
                                        ),
                                    };
                                    match transcript
                                        .lock()
                                        .map_err(|_| anyhow!("transcript recorder poisoned"))
                                        .and_then(|mut recorder| {
                                            recorder.record_internal_continuation(
                                                text,
                                                crate::transcript::InternalContinuationSource::SubagentCompletion,
                                            )
                                        })
                                        .and_then(|()| {
                                            turn_continuation_queue
                                                .lock()
                                                .map_err(|_| anyhow!("turn continuation queue poisoned"))
                                                .map(|mut queue| queue.push(continuation))
                                        }) {
                                        Ok(()) => {
                                            if let Some(run_id) = delivered_run {
                                                background_deliveries.entry(run_id).or_default().mark_delivered();
                                            }
                                        }
                                        Err(error) => {
                                            let _ = session_transport_tx.send(
                                                SessionTransportEvent::Error(ErrorEvent::new(format!(
                                                    "failed to queue background subagent completion: {error}"
                                                ))),
                                            );
                                        }
                                    }
                                }
                                Some(command) => {
                                    handle_active_turn_command(
                                        command,
                                        &mut parked_commands,
                                        &session_transport_tx,
                                        &transcript,
                                        &subagent_runtime,
                                        &mut background_deliveries,
                                        &mut change_notice_sent,
                                    );
                                }
                                None => break,
                            },
                        }
                    }

                    (
                        interrupted,
                        shutdown,
                        interrupt_failure,
                        turn_failure,
                        boundary_preempted,
                        inserted_prompt,
                    )
                };

                if let Some(error) = interrupt_failure {
                    let _ = session_transport_tx.send(SessionTransportEvent::Error(ErrorEvent::new(error)));
                    deferred_commands.clear();
                    parked_commands.clear();
                    boundary_prompts.clear();
                    break;
                }
                if boundary_preempted {
                    if let Err(error) = derive_interrupt_request(&transcript, &subagent_runtime)
                        .and_then(|interrupt| record_interrupt_transcript(&transcript, &interrupt))
                    {
                        let _ = session_transport_tx.send(SessionTransportEvent::Error(
                            ErrorEvent::new(format!(
                                "failed to close the preempted turn before queued prompt: {error}"
                            )),
                        ));
                        deferred_commands.clear();
                        parked_commands.clear();
                        boundary_prompts.clear();
                        break;
                    }
                    if let Err(error) = rehydrate_agent_from_transcript(&mut agent, &transcript) {
                        let _ = session_transport_tx.send(SessionTransportEvent::Error(
                            ErrorEvent::new(format!(
                                "failed to restore background completion before queued prompt: {error}"
                            )),
                        ));
                        deferred_commands.clear();
                        parked_commands.clear();
                        boundary_prompts.clear();
                        break;
                    }
                }
                if let Some(interrupt) = interrupted {
                    let persisted_interrupt = match record_interrupt_transcript(&transcript, &interrupt) {
                        Ok(()) => true,
                        Err(error) => {
                            let replanned = match
                                derive_interrupt_request(&transcript, &subagent_runtime)
                            {
                                Ok(replanned) => replanned,
                                Err(replan_error) => {
                                    let _ = session_transport_tx.send(SessionTransportEvent::Error(
                                        ErrorEvent::new(format!(
                                            "failed to replan interrupt transcript: {replan_error}"
                                        )),
                                    ));
                                    deferred_commands.clear();
                                    parked_commands.clear();
                                    break;
                                }
                            };
                            if replanned.branch_id != interrupt.branch_id
                                || replanned.turn_id != interrupt.turn_id
                            {
                                let _ = session_transport_tx.send(SessionTransportEvent::Error(
                                    ErrorEvent::new(
                                        "interrupt transcript plan changed branch or turn while refreshing",
                                    ),
                                ));
                                false
                            } else {
                                match record_interrupt_transcript(&transcript, &replanned) {
                                    Ok(()) => true,
                                    Err(replan_error) => {
                                        let _ = session_transport_tx.send(SessionTransportEvent::Error(
                                            ErrorEvent::new(format!(
                                                "failed to record interrupt transcript: {error}; replan failed: {replan_error}"
                                            )),
                                        ));
                                        false
                                    }
                                }
                            }
                        }
                    };
                    if persisted_interrupt {
                        if let Err(error) = rehydrate_agent_from_transcript(&mut agent, &transcript) {
                            let _ = session_transport_tx.send(SessionTransportEvent::Error(ErrorEvent::new(format!(
                                "failed to restore interrupted session context: {error}"
                            ))));
                            deferred_commands.clear();
                            parked_commands.clear();
                            break;
                        } else {
                            if let Err(error) =
                                send_rehydrated_runtime_context(&session_transport_tx, &agent)
                            {
                                let _ = session_transport_tx.send(SessionTransportEvent::Error(
                                    ErrorEvent::new(format!(
                                        "failed to emit interrupted runtime context: {error}"
                                    )),
                                ));
                                deferred_commands.clear();
                                parked_commands.clear();
                                break;
                            }
                            emit_dropped_child_terminals(
                                &session_transport_tx,
                                &subagent_runtime,
                                &interrupt.active_runs,
                                &mut background_deliveries,
                            );
                            send_subagent_interrupted(&session_transport_tx);
                        }
                    } else {
                        deferred_commands.clear();
                        parked_commands.clear();
                        break;
                    }
                }
                if shutdown {
                    deferred_commands.clear();
                    parked_commands.clear();
                    boundary_prompts.clear();
                    break;
                }
                if let Some(failure) = &turn_failure
                    && let Ok(recorder) = transcript.lock() {
                    let root = recorder.session_id().to_string();
                    drop(recorder);
                    subagent_runtime.fail_member_messages(&root, &root, failure, false);
                }
                if let Some(failure) = turn_failure
                    && agent.has_incomplete_tool_call_groups()
                {
                    tracing::warn!(
                        error = %failure,
                        "failed turn left unfinished tool calls; closing the turn"
                    );
                    match close_failed_turn(&mut agent, &transcript, &subagent_runtime) {
                        Ok(()) => {
                            if let Err(error) =
                                send_rehydrated_runtime_context(&session_transport_tx, &agent)
                            {
                                let _ = session_transport_tx.send(SessionTransportEvent::Error(
                                    ErrorEvent::new(format!(
                                        "failed to emit context after the failed turn: {error}"
                                    )),
                                ));
                                deferred_commands.clear();
                                parked_commands.clear();
                                break;
                            }
                        }
                        Err(error) => {
                            let _ = session_transport_tx.send(SessionTransportEvent::Error(
                                ErrorEvent::new(format!(
                                    "failed to close the failed turn: {error}"
                                )),
                            ));
                            deferred_commands.clear();
                            parked_commands.clear();
                            break;
                        }
                    }
                }
                // A terminal turn is now durable. Record its journal frontier;
                // the periodic worker batches pending turns without adding a model
                // call to every turn boundary.
                match transcript.lock() {
                    Ok(recorder) => {
                        if let Err(error) = crate::project_memory::enroll(&recorder) {
                            tracing::warn!(error = %error, "could not update project memory source");
                        }
                    }
                    Err(_) => tracing::warn!("could not update poisoned project memory source"),
                }
                flush_parked_commands(&mut deferred_commands, &mut parked_commands);
                if let Some(prompt) = inserted_prompt {
                    deferred_commands.push_front(prompt);
                } else if !boundary_preempted {
                    while let Some(prompt) = boundary_prompts.pop_back() {
                        deferred_commands.push_front(prompt);
                    }
                }
                change_notice_sent = false;
            }
            _ = memory_refresh.tick() => {
                if let Err(error) = memory_worker.tick(&agent).await {
                    tracing::warn!(error = %error, "project memory worker unavailable");
                    let _ = session_transport_tx.send(SessionTransportEvent::Notice(
                        NoticeEvent::info(
                            "Background memory processing is unavailable; see the application log",
                        )
                        .with_message_id(NoticeId::BackgroundMemoryUnavailable),
                    ));
                }
            }
            _ = child_refresh.tick(), if visible_child_session_id.is_some() => {
                refresh_visible_child_session_view(
                    &transcript,
                    &session_transport_tx,
                    &config.sessions_dir,
                    &mut visible_child_session_id,
                    &mut visible_child_view_state,
                    &mut visible_child_view_cache,
                    &child_live_text,
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
                    let mcp::McpServerDiscovery {
                        mut server,
                        session,
                        tools,
                    } = server_discovery;
                    let mut catalog_tools = Vec::new();
                    if let mcp::McpServerStatus::Offline { message } = &server.status {
                        let _ = session_transport_tx.send(SessionTransportEvent::McpDiagnostic(format!(
                            "MCP server '{}' is offline: {message}",
                            server.name
                        )));
                    }
                    let mut registered = Vec::new();
                    for tool in tools {
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
                    replace_mcp_session(&mut mcp_sessions, &server.name, session).await;
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
    if agent.subagents_policy().mode == crate::config::SubagentMode::Swarm {
        subagent_runtime.close_swarm_sessions();
        if !subagent_runtime.wait_until_idle().await {
            let _ = session_transport_tx.send(SessionTransportEvent::Error(ErrorEvent::new(
                "subagent runs remain active after shutdown cancellation",
            )));
        }
    }
    for (_, session) in std::mem::take(&mut mcp_sessions) {
        session.shutdown().await;
    }
    if let Err(error) = memory_worker.shutdown().await {
        tracing::warn!(error = %error, "project memory worker shutdown failed");
    }
}

#[derive(Default)]
pub(crate) struct BackgroundDelivery {
    notified: bool,
    delivered: bool,
}

impl BackgroundDelivery {
    /// Marks the run notified; returns whether this call was the first.
    pub(crate) fn mark_notified(&mut self) -> bool {
        !std::mem::replace(&mut self.notified, true)
    }

    pub(crate) fn mark_delivered(&mut self) {
        self.delivered = true;
    }
}

/// Report a terminal background run once; returns whether it is deliverable.
pub(crate) fn observe_background_subagent_completion(
    transcript: &Arc<StdMutex<TranscriptRecorder>>,
    subagent_runtime: &SubagentPool,
    background_deliveries: &mut HashMap<String, BackgroundDelivery>,
    session_transport_tx: &mpsc::UnboundedSender<SessionTransportEvent>,
    parent_session_id: &str,
    parent_tool_call_id: Option<String>,
    result: &Result<crate::subagent::SubagentRunSummary, String>,
) -> bool {
    if subagent_runtime.member_scope_stopped(parent_session_id) {
        return false;
    }
    let Ok(summary) = result else {
        return true;
    };
    if subagent_runtime.is_foregrounded(&summary.run_id) {
        return false;
    }
    let current_session_id = transcript
        .lock()
        .ok()
        .map(|recorder| recorder.session_id().to_string());
    if current_session_id.as_deref() != Some(parent_session_id) {
        return false;
    }
    // A cancelled background run still needs the terminal its dropped executor never published.
    if summary.status == SubagentStatus::Cancelled {
        if background_deliveries
            .entry(summary.run_id.clone())
            .or_default()
            .mark_notified()
        {
            emit_child_terminal(session_transport_tx, &summary.child_session_id);
        }
        return false;
    }
    if subagent_runtime.member_scope_stopped(&summary.child_session_id) {
        return false;
    }
    if background_deliveries
        .entry(summary.run_id.clone())
        .or_default()
        .mark_notified()
    {
        if summary.status == SubagentStatus::TimedOut {
            emit_child_terminal(session_transport_tx, &summary.child_session_id);
        }
        let _ = session_transport_tx.send(SessionTransportEvent::BackgroundSubagentCompleted {
            parent_tool_call_id,
            result: summary.clone(),
        });
    }
    true
}

pub(crate) fn format_background_subagent_completion(
    result: &crate::subagent::SubagentRunSummary,
) -> String {
    let structured = serde_json::to_string_pretty(&result.structured_result)
        .unwrap_or_else(|_| result.summary.clone());
    format!(
        "A background subagent has completed.\n\nagent: {}\nrun_id: {}\nchild_session_id: {}\nstatus: {}\nfailure_kind: {}\n\n{}\n\nContinue the user's task using this result. Do not repeat work already completed.",
        result.agent_name,
        result.run_id,
        result.child_session_id,
        result.status.as_str(),
        result
            .failure_kind
            .map(|kind| kind.as_str())
            .unwrap_or("none"),
        structured
    )
}

#[cfg(test)]
#[path = "../../tests/internal/session/engine/tests.rs"]
mod tests;
