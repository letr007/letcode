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
use serde_json::json;
use tokio::task::JoinHandle;

use crate::agent::{
    Agent, AgentEvent, ConfiguredPrimaryRouteFactory, ManualCompactionOutcome, SubagentInvocation,
};
use crate::agent_event_journal::persist_agent_event;
use crate::config::{AppConfig, ModelRoute, ProviderConfig, RetryConfig};
use crate::mcp;
use crate::runtime_context::RuntimeActiveContext;
use crate::session::runner::{ModelCatalogEntry, ModelCatalogReasoning, ModelCatalogUpdatedEvent};
use crate::session::{
    AgentRunner, ErrorEvent, NoticeEvent, RuntimeContextDisposition, RuntimeContextUpdatedEvent,
    SessionCommand, SessionEvent, SessionTransportEvent, TokenUsageEvent,
};
use crate::subagent::SubagentPool;
use crate::tool::{ToolHandler, normalize_subagent_input};

mod config_reload;
mod control;

use crate::transcript::{
    ChildSessionSummary, TranscriptEvent, TranscriptRecorder, read_records,
    read_records_allow_partial_tail, remove_empty_session_file, sync_recorder_branch,
    transcript_projection,
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
    /// Global default route used only when starting a new session.
    pub new_session_default_route: ModelRoute,
    /// Global expert defaults used only when starting a new session.
    pub new_session_default_expert_routes: indexmap::IndexMap<String, ModelRoute>,
    /// Expert routes for the current session.
    pub expert_model_routes: indexmap::IndexMap<String, ModelRoute>,
    /// Provider-qualified routes allowed for per-invocation expert selection and takeover.
    pub expert_allowed_models: indexmap::IndexMap<String, Vec<ModelRoute>>,
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
        // The Jev backend records its reviews in a `reviewer` child session through
        // the same pool the expert backend uses. It is resolved here, where the
        // reviewer route is known and a client failure still reaches the caller
        // that reports it.
        let jev_reviewer = reviewer_jev_config(&agent, &config.providers)
            .map(|jev| {
                crate::session::jev_review::JevReviewer::new(
                    jev,
                    Arc::clone(&transcript),
                    Some(event_tx.clone()),
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
            config.sessions_dir,
            config.model_routes,
            config.route_api_key_configured,
            config.new_session_default_route,
            config.new_session_default_expert_routes,
            config.expert_model_routes,
            config.expert_allowed_models,
            config.legacy_expert_models,
            config.providers,
            config.global_retry,
            config.provider_api_key_hints,
            config.api_key_hint,
            config.mcp_config_path,
            config.mcp_config,
            config.runtime_catalog,
            jev_reviewer,
            mcp_tools_rx,
            reload_rx,
            control_rx,
            control_tx.clone(),
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

fn delegated_route_display_name(
    agent: &Agent,
    expert_model_routes: &indexmap::IndexMap<String, ModelRoute>,
    agent_name: &str,
) -> String {
    expert_model_routes
        .get(agent_name)
        .map_or_else(|| agent.route_display_name(), ModelRoute::display_name)
}

fn delegated_route_for_takeover(
    agent: &Agent,
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
        .and_then(|recorder| read_records(recorder.path()))?;
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
    pub(crate) turn_id: Option<u64>,
    pub(crate) transcript_revision: u64,
    pub(crate) branch_id: Option<String>,
}

pub(crate) fn derive_interrupt_request(
    transcript: &Arc<StdMutex<TranscriptRecorder>>,
    subagent_runtime: &SubagentPool,
) -> Result<InterruptRequest> {
    let active_child_session_id = subagent_runtime
        .active_child()
        .map(|child| child.child_session_id);
    let (transcript_revision, branch_id, turn_id, parent_tool_calls) = transcript
        .lock()
        .map_err(|_| anyhow!("transcript recorder poisoned"))?
        .interrupt_transcript_plan()?;

    Ok(InterruptRequest {
        parent_tool_calls,
        visible_child_session_id: active_child_session_id,
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
                        *visible_child_session_id =
                            crate::session::SessionCoordinator::emit_view_child(
                                &transcript,
                                session_transport_tx,
                                Some(sessions_dir),
                                navigation,
                                anchor_child_session_id.as_deref(),
                            );
                        *visible_child_view_state = None;
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
pub(crate) struct VisibleChildViewState {
    record_count: usize,
    index: usize,
    total: usize,
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

/// Where the visible child sits among its parent's children.
#[derive(Debug, Clone)]
struct VisibleChildViewResolution {
    parent_session_id: String,
    index: usize,
    total: usize,
    child: ChildSessionSummary,
}

/// What the poll keeps from the last pass that reached a resolution. It stays
/// valid while both journals keep the same size and modification time, because
/// an unchanged parent journal lists the same children in the same order: a
/// child's transcript file is created before its parent records the start, and a
/// live parent holds the writer lock that archiving a session family needs.
///
/// It is written before the runtime context is projected, so once a view has
/// been sent, a projection that fails is not retried until one of the two
/// journals moves.
#[derive(Debug, Clone)]
struct VisibleChildViewCache {
    child_journal: Option<JournalFingerprint>,
    parent_journal: Option<JournalFingerprint>,
    resolution: VisibleChildViewResolution,
}

fn resolve_visible_child_view(
    transcript: &Arc<StdMutex<TranscriptRecorder>>,
    sessions_dir: &std::path::Path,
    child_session_id: &str,
) -> Result<Option<VisibleChildViewResolution>> {
    let (parent_session_id, parent_path) = {
        let recorder = transcript
            .lock()
            .map_err(|_| anyhow::anyhow!("transcript recorder poisoned"))?;
        (
            recorder.session_id().to_string(),
            recorder.path().to_path_buf(),
        )
    };
    // Same shaped scan the navigation path uses: the poll only needs the child
    // list, and decoding every parent record costs several times more (measured
    // 70 ms against 12 ms on a 21 MB journal).
    let children = SubagentPool::child_sessions_from_summaries(
        transcript_projection::project_child_session_summaries_from_file(
            &crate::transcript::child_sessions_dir(sessions_dir),
            &parent_path,
        )?,
    );
    Ok(children
        .iter()
        .enumerate()
        .find(|(_, child)| child.child_session_id == child_session_id)
        .map(|(index, child)| VisibleChildViewResolution {
            parent_session_id,
            index,
            total: children.len(),
            child: child.clone(),
        }))
}

async fn refresh_visible_child_session_view(
    transcript: &Arc<StdMutex<TranscriptRecorder>>,
    session_transport_tx: &mpsc::UnboundedSender<SessionTransportEvent>,
    sessions_dir: &std::path::Path,
    visible_child_session_id: &mut Option<String>,
    visible_child_view_state: &mut Option<VisibleChildViewState>,
    visible_child_view_cache: &mut Option<VisibleChildViewCache>,
) {
    let Some(child_session_id) = visible_child_session_id.as_deref() else {
        return;
    };
    let child_journal = journal_fingerprint(&crate::transcript::child_session_path(
        sessions_dir,
        child_session_id,
    ));
    let parent_journal = transcript
        .lock()
        .ok()
        .and_then(|recorder| journal_fingerprint(recorder.path()));
    let settled = visible_child_view_cache.as_ref().is_some_and(|cache| {
        cache.resolution.child.child_session_id == child_session_id
            && cache.child_journal == child_journal
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
        child_journal,
        parent_journal,
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
    let _ = session_transport_tx.send(SessionTransportEvent::ChildSessionViewed {
        parent_session_id: resolution.parent_session_id,
        child_session_id: resolution.child.child_session_id,
        agent_name: resolution.child.agent_name,
        index: resolution.index,
        total: resolution.total,
        pool_ordinal: resolution.child.pool_ordinal,
        records,
        runtime_context,
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

#[allow(clippy::too_many_arguments)]
async fn run_engine_loop(
    agent: Agent,
    transcript: Arc<StdMutex<TranscriptRecorder>>,
    sessions_dir: PathBuf,
    model_routes: indexmap::IndexMap<String, ModelRoute>,
    route_api_key_configured: indexmap::IndexMap<String, bool>,
    new_session_default_route: ModelRoute,
    new_session_default_expert_routes: indexmap::IndexMap<String, ModelRoute>,
    expert_model_routes: indexmap::IndexMap<String, ModelRoute>,
    expert_allowed_models: indexmap::IndexMap<String, Vec<ModelRoute>>,
    legacy_expert_models: indexmap::IndexMap<String, String>,
    providers: indexmap::IndexMap<String, crate::config::ProviderConfig>,
    global_retry: crate::config::RetryConfig,
    provider_api_key_hints: indexmap::IndexMap<String, String>,
    api_key_hint: String,
    mcp_config_path: PathBuf,
    mcp_config: indexmap::IndexMap<String, crate::config::McpServerConfig>,
    mut runtime_catalog: crate::model_runtime::ResolvedRuntimeCatalog,
    jev_reviewer: Option<std::sync::Arc<dyn crate::agent::AutoReviewService>>,
    mcp_tools_rx: mpsc::UnboundedReceiver<Vec<mcp::McpServerDiscovery>>,
    mut reload_rx: mpsc::UnboundedReceiver<()>,
    mut control_rx: mpsc::UnboundedReceiver<SessionEngineControl>,
    control_tx: mpsc::UnboundedSender<SessionEngineControl>,
    session_transport_tx: mpsc::UnboundedSender<SessionTransportEvent>,
    title_event_tx: mpsc::UnboundedSender<SessionTransportEvent>,
    subagent_runtime: SubagentPool,
) {
    let transcript = transcript;
    let mut agent = agent;
    let mut mcp_tools_rx = Some(mcp_tools_rx);
    let mut mcp_config = mcp_config;
    let mut model_routes = model_routes;
    let route_api_key_configured = Arc::new(StdMutex::new(route_api_key_configured));
    let mut new_session_default_route = new_session_default_route;
    let mut new_session_default_expert_routes = new_session_default_expert_routes;
    let mut expert_model_routes = expert_model_routes;
    let mut expert_allowed_models = expert_allowed_models;
    let mut legacy_expert_models = legacy_expert_models;
    let mut providers = providers;
    let mut global_retry = global_retry;
    let provider_api_key_hints = Arc::new(StdMutex::new(provider_api_key_hints));
    let mut mcp_registered_tools: HashMap<String, Vec<String>> = HashMap::new();
    let subagent_runtime = subagent_runtime;
    let auto_review_service: std::sync::Arc<dyn crate::agent::AutoReviewService> =
        match jev_reviewer {
            Some(service) => service,
            None => std::sync::Arc::new(crate::session::auto_review::StickyAutoReviewer::new(
                subagent_runtime.clone(),
                sessions_dir.clone(),
                Arc::clone(&transcript),
                Some(session_transport_tx.clone()),
                Arc::clone(&route_api_key_configured),
                Arc::clone(&provider_api_key_hints),
                api_key_hint.clone(),
            )),
        };
    agent.set_auto_review_service(Some(std::sync::Arc::clone(&auto_review_service)));
    agent.historian_runtime = Some(Arc::new(crate::session::historian::HistorianRuntime::new(
        subagent_runtime.clone(),
        sessions_dir.clone(),
        Arc::clone(&transcript),
        session_transport_tx.clone(),
    )));
    let mut memory_worker = crate::project_memory::MemoryWorker::default();
    let memory_refresh_period = std::time::Duration::from_secs(30);
    let mut memory_refresh = tokio::time::interval_at(
        tokio::time::Instant::now() + memory_refresh_period,
        memory_refresh_period,
    );
    memory_refresh.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut deferred_commands = VecDeque::new();
    let mut parked_commands = VecDeque::new();
    let mut delivered_background_runs = std::collections::HashSet::new();
    let mut visible_child_session_id = None;
    let mut visible_child_view_state = None;
    let mut visible_child_view_cache = None;
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
                if let Some(historian) = &agent.historian_runtime { historian.cancel(); }
                let previous_primary_route = agent.primary_route().cloned();
                let previous_expert_model_routes = expert_model_routes.clone();
                let previous_expert_allowed_models = expert_allowed_models.clone();
                if let Err(error) = apply_config_reload(
                    &mut agent,
                    &mcp_config_path,
                    &mut model_routes,
                    &mut route_api_key_configured
                        .lock()
                        .unwrap_or_else(|error| error.into_inner()),
                    &mut expert_model_routes,
                    &mut new_session_default_expert_routes,
                    &mut expert_allowed_models,
                    &mut legacy_expert_models,
                    &mut providers,
                    &mut global_retry,
                    &mut provider_api_key_hints
                        .lock()
                        .unwrap_or_else(|error| error.into_inner()),
                    &mut new_session_default_route,
                    &mut runtime_catalog,
                    &session_transport_tx,
                ) {
                    let _ = session_transport_tx.send(SessionTransportEvent::Error(ErrorEvent::new(
                        format!("failed to reload configuration: {error}"),
                    )));
                } else if reviewer_policy_changed(
                    previous_primary_route.as_ref(),
                    agent.primary_route(),
                    &previous_expert_model_routes,
                    &expert_model_routes,
                    &previous_expert_allowed_models,
                    &expert_allowed_models,
                ) {
                    auto_review_service.clear_sticky();
                }
            }
            command = next_idle_session_command(&mut control_rx, &mut deferred_commands) => {
                let Some(command) = command else {
                    break;
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
                        let Some(route) = model_routes.get(model_id).cloned() else {
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
                    let mut updated_allowed_models = expert_allowed_models.clone();
                    updated_allowed_models.insert(agent_name.clone(), routes.clone());
                    let expert_factory = match crate::subagent::ExpertRouteFactory::new_with_policies(
                        crate::delegation::supported_agent_names().map(|name| {
                            (
                                name.to_string(),
                                expert_model_routes.get(name).cloned(),
                                updated_allowed_models.get(name).cloned().unwrap_or_default(),
                            )
                        }),
                        &providers,
                        &global_retry,
                    ) {
                        Ok(factory) => factory.with_runtime_catalog(runtime_catalog.clone()),
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
                        &mcp_config_path,
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
                    expert_allowed_models = updated_allowed_models;
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
                    let mut updated_expert_model_routes = expert_model_routes.clone();
                    updated_expert_model_routes.insert(agent_name.clone(), route.clone());
                    let expert_factory = match crate::subagent::ExpertRouteFactory::new_with_policies(
                        crate::delegation::supported_agent_names().map(|name| {
                            (
                                name.to_string(),
                                updated_expert_model_routes.get(name).cloned(),
                                expert_allowed_models
                                    .get(name)
                                    .cloned()
                                    .unwrap_or_default(),
                            )
                        }),
                        &providers,
                        &global_retry,
                    ) {
                        Ok(factory) => factory.with_runtime_catalog(runtime_catalog.clone()),
                        Err(error) => {
                            let _ = session_transport_tx.send(SessionTransportEvent::Error(
                                ErrorEvent::new(format!("failed to set expert model: {error}")),
                            ));
                            continue;
                        }
                    };
                    if let Err(error) = transcript
                        .lock()
                        .map_err(|_| anyhow!("transcript recorder poisoned"))
                        .and_then(|mut recorder| {
                            recorder.record_expert_model_changed(
                                agent_name.clone(),
                                route.display_name(),
                            )
                        })
                    {
                        let _ = session_transport_tx.send(SessionTransportEvent::Error(
                            ErrorEvent::new(format!("failed to set expert model: {error}")),
                        ));
                        continue;
                    }
                    agent.set_subagent_child_factory(Arc::new(expert_factory));
                    expert_model_routes = updated_expert_model_routes;
                    if agent_name == "historian"
                        && let Some(historian) = &agent.historian_runtime { historian.cancel(); }
                    if agent_name == "reviewer" {
                        auto_review_service.clear_sticky();
                    }
                    let _ = session_transport_tx.send(SessionTransportEvent::ExpertModelChanged {
                        agent_name: agent_name.clone(),
                        model_id: route.display_name(),
                    });
                    continue;
                }

                if let SessionEngineCommand::SetModel(model) = &command {
                    let Some(route) = model_routes.get(model).cloned() else {
                        send_setting_change_failed(
                            &session_transport_tx,
                            crate::session::SessionCommand::SetModel(model.clone()),
                            format!("unknown model: {model}"),
                        );
                        continue;
                    };
                    let model_id = route.display_name();
                    let updated_expert_model_routes = match expert_routes_after_primary_switch(
                        &expert_model_routes,
                        &legacy_expert_models,
                        &providers,
                        &route,
                    ) {
                        Ok(routes) => routes,
                        Err(error) => {
                            send_setting_change_failed(
                                &session_transport_tx,
                                crate::session::SessionCommand::SetModel(model.clone()),
                                format!(
                                    "failed to set model because expert routes could not be updated: {error}"
                                ),
                            );
                            continue;
                        }
                    };
                    let expert_factory = match crate::subagent::ExpertRouteFactory::new_with_policies(
                        crate::delegation::supported_agent_names().map(|name| {
                            (
                                name.to_string(),
                                updated_expert_model_routes.get(name).cloned(),
                                expert_allowed_models
                                    .get(name)
                                    .cloned()
                                    .unwrap_or_default(),
                            )
                        }),
                        &providers,
                        &global_retry,
                    ) {
                        Ok(factory) => factory.with_runtime_catalog(runtime_catalog.clone()),
                        Err(error) => {
                            send_setting_change_failed(
                                &session_transport_tx,
                                crate::session::SessionCommand::SetModel(model.clone()),
                                format!(
                                    "failed to set model because expert routes could not be rebuilt: {error}"
                                ),
                            );
                            continue;
                        }
                    };
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
                                    ),
                                ));
                            }
                            if agent
                                .fake_client()
                                .is_some_and(|client| !client.supports_protocol(agent.active_protocol()))
                            {
                                agent
                                    .set_fake_client(None)
                                    .expect("disabling fake mode is always supported");
                                let _ = session_transport_tx.send(
                                    SessionTransportEvent::FakeClientChanged { client: None },
                                );
                                let _ = session_transport_tx.send(SessionTransportEvent::Notice(
                                    NoticeEvent::info(
                                        "Fake mode disabled: unsupported by the selected model protocol",
                                    ),
                                ));
                            }
                            agent.set_subagent_child_factory(Arc::new(expert_factory));
                            expert_model_routes = updated_expert_model_routes;
                            // A sticky reviewer session records its actual route. Drop it after a
                            // primary-route change so the next review starts with the new policy.
                            if let Some(historian) = &agent.historian_runtime { historian.cancel(); }
                        auto_review_service.clear_sticky();
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
                        let history_navigation = matches!(
                            command,
                            SessionEngineCommand::Undo
                                | SessionEngineCommand::Redo
                                | SessionEngineCommand::NavigateHistory { .. }
                        );
                        if history_navigation
                            && let Some(historian) = &agent.historian_runtime { historian.cancel(); }
                        let prepared_history_factory = std::cell::RefCell::new(None);
                        let prepared_history_routes = std::cell::RefCell::new(None);
                        let current_primary_route = agent.primary_route().cloned();
                        let dispatch_result =
                            crate::session::SessionCoordinator::dispatch_idle_command_with_history_prepare(
                                session_command,
                                &mut agent,
                                &transcript,
                                &session_transport_tx,
                                Some(sessions_dir.as_path()),
                                |snapshot| {
                                    let restored_expert_models =
                                        crate::transcript::restore_latest_expert_models(
                                            &snapshot.records,
                                        );
                                    let mut restored_routes =
                                        config_default_expert_routes_for_primary(
                                            &new_session_default_expert_routes,
                                            &legacy_expert_models,
                                            snapshot.latest_model.as_deref()
                                                .map(ModelRoute::parse)
                                                .transpose()?
                                                .as_ref()
                                                .or(current_primary_route.as_ref())
                                                .unwrap_or(&new_session_default_route),
                                        );
                                    for (agent_name, route) in restored_expert_models {
                                        restored_routes.insert(
                                            agent_name,
                                            ModelRoute::parse(&route).map_err(|error| {
                                                anyhow!(
                                                    "failed to restore expert model '{route}': {error}"
                                                )
                                            })?,
                                        );
                                    }
                                    let factory =
                                        crate::subagent::ExpertRouteFactory::new_with_policies(
                                            crate::delegation::supported_agent_names().map(|name| {
                                                (
                                                    name.to_string(),
                                                    restored_routes.get(name).cloned(),
                                                    expert_allowed_models
                                                        .get(name)
                                                        .cloned()
                                                        .unwrap_or_default(),
                                                )
                                            }),
                                            &providers,
                                            &global_retry,
                                        )?
                                        .with_runtime_catalog(runtime_catalog.clone());
                                    *prepared_history_routes.borrow_mut() = Some(restored_routes);
                                    *prepared_history_factory.borrow_mut() = Some(factory);
                                    Ok(())
                                },
                            );
                        if let Err(error) = dispatch_result {
                            let _ = session_transport_tx.send(SessionTransportEvent::Error(
                                ErrorEvent::new(format!("failed to dispatch session command: {error}")),
                            ));
                            continue;
                        }
                        if history_navigation
                            && matches!(
                                dispatch_result,
                                Ok(crate::session::IdleDispatch::HistoryNavigated)
                            )
                            && let (Some(factory), Some(routes)) = (
                                prepared_history_factory.into_inner(),
                                prepared_history_routes.into_inner(),
                            ) {
                                expert_model_routes = routes;
                                agent.set_subagent_child_factory(Arc::new(factory));
                                if let Some(historian) = &agent.historian_runtime { historian.cancel(); }
                        auto_review_service.clear_sticky();
                            }
                        if matches!(command, SessionEngineCommand::ViewParent) {
                            visible_child_session_id = None;
                            visible_child_view_state = None;
                        }
                    }
                    continue;
                }

                if let SessionEngineCommand::BackgroundSubagentCompleted {
                    parent_session_id,
                    parent_tool_call_id,
                    result,
                } = command
                {
                    let current_session_id = transcript
                        .lock()
                        .ok()
                        .map(|recorder| recorder.session_id().to_string());
                    if current_session_id.as_deref() != Some(parent_session_id.as_str()) {
                        continue;
                    }
                    if result.as_ref().is_ok_and(|result| {
                        result.status == crate::subagent::SubagentStatus::Cancelled
                            || subagent_runtime.is_foregrounded(&result.run_id)
                    }) {
                        continue;
                    }
                    let delivered_run = result.as_ref().ok().map(|result| result.run_id.clone());
                    let prompt = match result {
                        Ok(result) => {
                            if delivered_background_runs.contains(&result.run_id) {
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
                            let _ = session_transport_tx.send(
                                SessionTransportEvent::BackgroundSubagentCompleted {
                                    parent_tool_call_id,
                                    result: result.clone(),
                                },
                            );
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
                        delivered_background_runs.insert(run_id);
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
                    | SessionEngineCommand::SetExpertModel { .. }
                    | SessionEngineCommand::SetExpertAllowedModels { .. }
                    | SessionEngineCommand::ToggleFastMode
                    | SessionEngineCommand::SetReasoningEffort(_)
                    | SessionEngineCommand::SetFakeClient(_)
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
                                model: None,
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
                        let route_has_credentials = {
                            let route_api_keys = route_api_key_configured
                                .lock()
                                .unwrap_or_else(|error| error.into_inner());
                            route_has_api_key(&route_api_keys, &route_display_name)
                                || delegated_route_display_name(
                                    &agent,
                                    &expert_model_routes,
                                    &agent_name,
                                ) == route_display_name
                        };
                        if !route_has_credentials {
                            {
                                let provider_hints = provider_api_key_hints
                                    .lock()
                                    .unwrap_or_else(|error| error.into_inner());
                                send_missing_api_key_error(
                                    &session_transport_tx,
                                    &route_display_name,
                                    &provider_hints,
                                    &api_key_hint,
                                );
                            }
                            continue;
                        }

                        let (
                            interrupted,
                            interrupted_child_session_id,
                            shutdown,
                            interrupt_failure,
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
                            let mut interrupted_child_session_id = None;
                            let mut shutdown = false;
                            let mut interrupt_failure = None;

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
                                        let interrupt = match derive_interrupt_request(
                                            &transcript,
                                            &subagent_runtime,
                                        ) {
                                            Ok(interrupt) => interrupt,
                                            Err(error) => {
                                                subagent_runtime.cancel_active();
                                                let settle_shutdown = wait_for_subagent_cancel_settle(
                                                    &mut control_rx,
                                                    &mut deferred_commands,
                                                    delegate.as_mut(),
                                                    &subagent_runtime,
                                                )
                                                .await;
                                                shutdown = true;
                                                let _ = settle_shutdown;
                                                interrupt_failure = Some(format!(
                                                    "failed to derive interrupt transcript plan: {error}"
                                                ));
                                                break;
                                            }
                                        };
                                        interrupted = true;
                                        interrupted_child_session_id = interrupt
                                            .visible_child_session_id
                                            .clone();
                                        subagent_runtime.cancel_active();
                                        if let Err(error) =
                                            record_interrupt_transcript(&transcript, &interrupt)
                                        {
                                            let settle_shutdown = wait_for_subagent_cancel_settle(
                                                &mut control_rx,
                                                &mut deferred_commands,
                                                delegate.as_mut(),
                                                &subagent_runtime,
                                            )
                                            .await;
                                            shutdown |= settle_shutdown;
                                            interrupt_failure = Some(format!(
                                                "failed to record interrupt transcript: {error}"
                                            ));
                                            interrupted = false;
                                            break;
                                        }
                                        let settle_shutdown = wait_for_subagent_cancel_settle(
                                            &mut control_rx,
                                            &mut deferred_commands,
                                            delegate.as_mut(),
                                            &subagent_runtime,
                                        )
                                        .await;
                                        shutdown |= settle_shutdown;
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
                                        handle_active_turn_command(
                                            command,
                                            &mut parked_commands,
                                            &session_transport_tx,
                                        );
                                    }
                                    ActiveSessionOperation::Command(None) => break,
                                }
                            }

                            (
                                interrupted,
                                interrupted_child_session_id,
                                shutdown,
                                interrupt_failure,
                            )
                        };

                        if let Some(error) = interrupt_failure {
                            let _ = session_transport_tx
                                .send(SessionTransportEvent::Error(ErrorEvent::new(error)));
                            deferred_commands.clear();
                            parked_commands.clear();
                            break;
                        }
                        if interrupted {
                            match rehydrate_agent_from_transcript(&mut agent, &transcript) {
                                Ok(()) => {
                                    if let Err(error) =
                                        send_rehydrated_runtime_context(&session_transport_tx, &agent)
                                    {
                                        let _ = session_transport_tx.send(
                                            SessionTransportEvent::Error(ErrorEvent::new(format!(
                                                "failed to emit interrupted runtime context: {error}"
                                            ))),
                                        );
                                        deferred_commands.clear();
                                        parked_commands.clear();
                                        break;
                                    }
                                    send_subagent_interrupted(
                                        &session_transport_tx,
                                        interrupted_child_session_id,
                                    );
                                }
                                Err(error) => {
                                    let _ = session_transport_tx.send(SessionTransportEvent::Error(
                                        ErrorEvent::new(format!(
                                            "failed to restore interrupted session context: {error}"
                                        )),
                                    ));
                                    deferred_commands.clear();
                                    parked_commands.clear();
                                    break;
                                }
                            }
                        }
                        if shutdown {
                            deferred_commands.clear();
                            parked_commands.clear();
                            break;
                        }
                        flush_parked_commands(&mut deferred_commands, &mut parked_commands);
                        continue;
                    }
                    SessionEngineCommand::Compact => {
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
                                    &api_key_hint,
                                );
                            }
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
                            &sessions_dir,
                            &mut control_rx,
                            &mut deferred_commands,
                            &mut visible_child_session_id,
                            &mut visible_child_view_state,
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
                        let restored_expert_models =
                            crate::transcript::restore_latest_expert_models(&prepared.snapshot.records);
                        let mut resumed_expert_model_routes =
                            config_default_expert_routes_for_primary(
                                &new_session_default_expert_routes,
                                &legacy_expert_models,
                                &new_session_default_route,
                            );
                        let mut expert_restore_error = None;
                        for (agent_name, route) in restored_expert_models {
                            match ModelRoute::parse(&route) {
                                Ok(route) => {
                                    resumed_expert_model_routes.insert(agent_name, route);
                                }
                                Err(error) => {
                                    expert_restore_error = Some(format!(
                                        "failed to restore expert model for '{agent_name}': {error}"
                                    ));
                                    break;
                                }
                            }
                        }
                        if let Some(error) = expert_restore_error {
                            let _ = session_transport_tx.send(SessionTransportEvent::Error(
                                ErrorEvent::new(error),
                            ));
                            continue;
                        }
                        let expert_factory = match crate::subagent::ExpertRouteFactory::new_with_policies(
                            crate::delegation::supported_agent_names().map(|name| {
                                (
                                    name.to_string(),
                                    resumed_expert_model_routes.get(name).cloned(),
                                    expert_allowed_models
                                        .get(name)
                                        .cloned()
                                        .unwrap_or_default(),
                                )
                            }),
                            &providers,
                            &global_retry,
                        ) {
                            Ok(factory) => factory.with_runtime_catalog(runtime_catalog.clone()),
                            Err(error) => {
                                let _ = session_transport_tx.send(SessionTransportEvent::Error(
                                    ErrorEvent::new(format!(
                                        "failed to configure expert models for the resumed session: {error}"
                                    )),
                                ));
                                continue;
                            }
                        };
                        let prepared_install =
                            match crate::session::restore::prepare_routed_resume_install(
                                &agent,
                                &transcript,
                                prepared,
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
                            subagent_runtime.cancel_active_run_ids();
                            if !subagent_runtime.wait_until_idle().await {
                                let _ = session_transport_tx.send(SessionTransportEvent::Error(
                                    ErrorEvent::new("failed to cancel active subagents before resuming session"),
                                ));
                                continue;
                            }
                            prepared_install.commit(&mut agent, &transcript).1
                        };
                        if fast_mode_auto_disabled {
                            let _ = session_transport_tx.send(SessionTransportEvent::FastModeChanged { enabled: false });
                            let _ = session_transport_tx.send(SessionTransportEvent::Notice(NoticeEvent::info(
                                "Fast mode auto-disabled: current model is unavailable",
                            )));
                        }
                        let recorded_fake_client =
                            crate::transcript::restore_latest_fake_client(&resumed_event_records);
                        let restored_fake_client = recorded_fake_client
                            .filter(|client| client.supports_protocol(agent.active_protocol()));
                        agent
                            .set_fake_client(restored_fake_client)
                            .expect("restoring fake mode must validate against the active protocol");
                        let _ = session_transport_tx.send(
                            SessionTransportEvent::FakeClientChanged {
                                client: restored_fake_client,
                            },
                        );
                        if restored_fake_client.is_none() && recorded_fake_client.is_some() {
                            let _ = session_transport_tx.send(SessionTransportEvent::Notice(
                                NoticeEvent::info(
                                    "Fake mode disabled: unsupported by the resumed model protocol",
                                ),
                            ));
                        }
                        expert_model_routes = resumed_expert_model_routes;
                        agent.set_subagent_child_factory(Arc::new(expert_factory));
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
                        let _ = session_transport_tx.send(SessionTransportEvent::SessionResumed {
                            session_id: resumed_event_session_id,
                            branch_id: resumed_event_branch_id,
                            messages: resumed_event_messages,
                            records: resumed_event_records,
                            evidence_count: resumed_event_evidence_count,
                            model_id: Some(agent.route_display_name()),
                            token_usage: Some(token_usage),
                            runtime_context,
                            expert_models: expert_model_routes
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
                                    .prepare_primary_route(new_session_default_route.clone())
                                {
                                    Ok(route) => Ok((new_session_default_route.clone(), route)),
                                    Err(default_error) => Err(anyhow!(
                                        "failed to prepare the current model ({current_error}); fallback default model also failed ({default_error})"
                                    )),
                                },
                            }
                        } else {
                            agent
                                .prepare_primary_route(new_session_default_route.clone())
                                .map(|route| (new_session_default_route.clone(), route))
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
                        let mut new_session_expert_model_routes =
                            config_default_expert_routes_for_primary(
                                &new_session_default_expert_routes,
                                &legacy_expert_models,
                                &new_session_route,
                            );
                        for (agent_name, route) in &expert_model_routes {
                            if providers
                                .get(&route.provider)
                                .is_some_and(|provider| provider.has_model(&route.model))
                            {
                                new_session_expert_model_routes
                                    .insert(agent_name.clone(), route.clone());
                            }
                        }
                        let expert_factory = match crate::subagent::ExpertRouteFactory::new_with_policies(
                            crate::delegation::supported_agent_names().map(|name| {
                                (
                                    name.to_string(),
                                    new_session_expert_model_routes.get(name).cloned(),
                                    expert_allowed_models
                                        .get(name)
                                        .cloned()
                                        .unwrap_or_default(),
                                )
                            }),
                            &providers,
                            &global_retry,
                        ) {
                            Ok(factory) => factory.with_runtime_catalog(runtime_catalog.clone()),
                            Err(error) => {
                                let _ = session_transport_tx.send(SessionTransportEvent::Error(
                                    ErrorEvent::new(format!(
                                        "failed to configure expert models for the new session: {error}"
                                    )),
                                ));
                                continue;
                            }
                        };
                        let prepared = match crate::session::prepare_new_session_package(
                            &sessions_dir,
                            new_session_route.display_name(),
                        ) {
                            Ok(prepared) => prepared,
                            Err(error) => {
                                let _ = session_transport_tx.send(SessionTransportEvent::Error(ErrorEvent::new(format!(
                                    "failed to create session transcript: {error}"
                                ))));
                                continue;
                            }
                        };
                        let mut prepared = prepared;
                        if let Err(error) = (|| -> Result<()> {
                            for (agent_name, route) in &new_session_expert_model_routes {
                                prepared.recorder.record_expert_model_changed(
                                    agent_name.clone(),
                                    route.display_name(),
                                )?;
                            }
                            Ok(())
                        })() {
                            let _ = remove_empty_session_file(prepared.recorder.path());
                            let _ = session_transport_tx.send(SessionTransportEvent::Error(
                                ErrorEvent::new(format!(
                                    "failed to record expert models for the new session: {error}"
                                )),
                            ));
                            continue;
                        }
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
                            expert_models: new_session_expert_model_routes
                                .iter()
                                .map(|(name, route)| (name.clone(), route.display_name()))
                                .collect(),
                        };
                        subagent_runtime.cancel_active_run_ids();
                        if !subagent_runtime.wait_until_idle().await {
                            let _ = session_transport_tx.send(SessionTransportEvent::Error(
                                ErrorEvent::new("failed to cancel active subagents before starting new session"),
                            ));
                            let _ = remove_empty_session_file(prepared_install.new_path());
                            continue;
                        }
                        let inherited_fake_client = agent.fake_client();
                        prepared_install.commit(&mut agent, &transcript);
                        let new_session_fake_client = inherited_fake_client
                            .filter(|client| client.supports_protocol(agent.active_protocol()));
                        agent
                            .set_fake_client(new_session_fake_client)
                            .expect("inheriting fake mode must validate against the new model protocol");
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
                        expert_model_routes = new_session_expert_model_routes;
                        agent.set_subagent_child_factory(Arc::new(expert_factory));
                        if let Some(historian) = &agent.historian_runtime { historian.cancel(); }
                        auto_review_service.clear_sticky();
                        for (agent_name, route) in &expert_model_routes {
                            let _ = session_transport_tx.send(
                                SessionTransportEvent::ExpertModelChanged {
                                    agent_name: agent_name.clone(),
                                    model_id: route.display_name(),
                                },
                            );
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
                    let route_display_name = agent.route_display_name();
                    {
                        let provider_hints = provider_api_key_hints
                            .lock()
                            .unwrap_or_else(|error| error.into_inner());
                        send_missing_api_key_error(
                            &session_transport_tx,
                            &route_display_name,
                            &provider_hints,
                            &api_key_hint,
                        );
                    }
                    continue;
                }

                let turn_continuation_queue = Arc::new(StdMutex::new(
                    crate::agent::TurnContinuationQueue::default(),
                ));
                let (runner_event_tx, mut runner_event_rx) = mpsc::unbounded_channel();
                let runner = AgentRunner::with_transcript(
                    runner_event_tx,
                    transcript.clone(),
                )
                    .with_session_title_event_sender(title_event_tx.clone())
                    .with_subagent_runtime(
                        subagent_runtime.clone(),
                        sessions_dir.clone(),
                        expert_model_routes.clone(),
                        route_api_key_configured
                            .lock()
                            .unwrap_or_else(|error| error.into_inner())
                            .clone(),
                        provider_api_key_hints
                            .lock()
                            .unwrap_or_else(|error| error.into_inner())
                            .clone(),
                        api_key_hint.clone(),
                        Some(control_tx.clone()),
                        Some(session_transport_tx.clone()),
                    );
                let historian_control = agent.historian_runtime.clone();
                let (interrupted, shutdown, interrupt_failure) = {
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
                                interrupted = Some(interrupt);
                                if is_shutdown && subagent_runtime.is_running() {
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
                                    if let Ok(mut queue) = turn_continuation_queue.lock() {
                                        queue.mark_user_prompt_queued();
                                    }
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
                                Some(SessionEngineCommand::BackgroundSubagentCompleted {
                                    parent_session_id,
                                    parent_tool_call_id,
                                    result,
                                }) => {
                                    let current_session_id = transcript
                                        .lock()
                                        .ok()
                                        .map(|recorder| recorder.session_id().to_string());
                                    if current_session_id.as_deref()
                                        != Some(parent_session_id.as_str())
                                    {
                                        continue;
                                    }
                                    if result.as_ref().is_ok_and(|result| {
                                        result.status == crate::subagent::SubagentStatus::Cancelled
                                            || subagent_runtime.is_foregrounded(&result.run_id)
                                    }) {
                                        continue;
                                    }
                                    let delivered_run = result.as_ref().ok().map(|result| result.run_id.clone());
                                    let (text, continuation) = match result {
                                        Ok(result) => {
                                            if delivered_background_runs.contains(&result.run_id) {
                                                continue;
                                            }
                                            let _ = session_transport_tx.send(
                                                SessionTransportEvent::BackgroundSubagentCompleted {
                                                    parent_tool_call_id,
                                                    result: result.clone(),
                                                },
                                            );
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
                                                delivered_background_runs.insert(run_id);
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
                                Some(SessionEngineCommand::ShowHistoryTree)
                                | Some(SessionEngineCommand::NavigateHistory { .. }) => {
                                    let _ = session_transport_tx.send(SessionTransportEvent::Error(ErrorEvent::new(
                                        "history navigation is unavailable while a turn is active",
                                    )));
                                }
                                Some(command) => {
                                    handle_active_turn_command(
                                        command,
                                        &mut parked_commands,
                                        &session_transport_tx,
                                    );
                                }
                                None => break,
                            },
                        }
                    }

                    (interrupted, shutdown, interrupt_failure)
                };

                if let Some(error) = interrupt_failure {
                    let _ = session_transport_tx.send(SessionTransportEvent::Error(ErrorEvent::new(error)));
                    deferred_commands.clear();
                    parked_commands.clear();
                    break;
                }
                if turn_continuation_queue
                    .lock()
                    .map(|queue| queue.preempted_by_user_prompt())
                    .unwrap_or(false)
                {
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
                            send_subagent_interrupted(
                                &session_transport_tx,
                                interrupt.visible_child_session_id,
                            );
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
                    break;
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
            }
            _ = memory_refresh.tick() => {
                if let Err(error) = memory_worker.tick(&agent).await {
                    tracing::warn!(error = %error, "project memory worker unavailable");
                    let _ = session_transport_tx.send(SessionTransportEvent::Notice(NoticeEvent::info(
                        "Background memory processing is unavailable; see the application log",
                    )));
                }
            }
            _ = child_refresh.tick(), if visible_child_session_id.is_some() => {
                refresh_visible_child_session_view(
                    &transcript,
                    &session_transport_tx,
                    &sessions_dir,
                    &mut visible_child_session_id,
                    &mut visible_child_view_state,
                    &mut visible_child_view_cache,
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
    if let Err(error) = memory_worker.shutdown().await {
        tracing::warn!(error = %error, "project memory worker shutdown failed");
    }
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
mod tests {
    use super::*;
    use crate::agent::PrimaryRouteFactory;
    use crate::request_builder::ModelReasoningEffort;
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

    /// Windows 路径里的反斜杠在 TOML 基本字符串中会被当成转义序列，先转义再插值。
    fn toml_path(path: &std::path::Path) -> String {
        path.display().to_string().replace('\\', "\\\\")
    }

    #[test]
    fn config_templates_keep_windows_paths_parseable() {
        let path =
            std::path::Path::new("C:\\Users\\runneradmin\\AppData\\Local\\Temp\\letcode\\sessions");
        let rendered = format!("sessions_dir = \"{}\"\n", toml_path(path));

        let parsed: toml::Value =
            toml::from_str(&rendered).expect("windows paths must stay parseable");
        assert_eq!(
            parsed["sessions_dir"].as_str(),
            Some(path.to_string_lossy().as_ref())
        );
    }

    fn parent_transcript(sessions_dir: &std::path::Path) -> Arc<StdMutex<TranscriptRecorder>> {
        Arc::new(StdMutex::new(
            TranscriptRecorder::create(sessions_dir).expect("create parent transcript"),
        ))
    }

    /// Agent whose reviewer policy is the given route.
    fn reviewer_route_agent(config: &AppConfig, route: ModelRoute) -> Agent {
        let mut agent = Agent::new("current", 1, 1);
        agent.set_subagent_child_factory(Arc::new(
            crate::subagent::ExpertRouteFactory::new_with_policies(
                [("reviewer".to_string(), Some(route), Vec::new())],
                &config.providers,
                &config.global.retry,
            )
            .expect("reviewer route factory"),
        ));
        agent
    }

    #[test]
    fn reviewer_route_selects_the_backend_from_its_provider() {
        let directory = tempfile::tempdir().unwrap();
        let config_path = directory.path().join("letcode.toml");
        fs::write(
            &config_path,
            r#"
active_provider = "chat"
[providers.chat]
protocol = "responses"
default_model = "current"
[providers.chat.auth]
type = "bearer"
credential = "chat-key"
[providers.chat.endpoints]
base_url = "http://127.0.0.1:1"
[providers.chat.models.current]

[providers.typesafe]
protocol = "responses"
default_model = "jev-latest"
reviewer = "jev"
[providers.typesafe.auth]
type = "bearer"
credential = "typesafe-key"
[providers.typesafe.endpoints]
base_url = "https://api.typesafe.ai"
[providers.typesafe.models."jev-latest"]
"#,
        )
        .unwrap();
        let config = AppConfig::load_from_path(&config_path).expect("config");

        let marked = reviewer_route_agent(&config, ModelRoute::new("typesafe", "jev-latest"));
        let jev = reviewer_jev_config(&marked, &config.providers).expect("jev backend");
        assert_eq!(jev.provider, "typesafe");
        assert_eq!(jev.base_url, "https://api.typesafe.ai");
        assert_eq!(jev.model, "jev-latest");
        assert_eq!(jev.credential, "typesafe-key");

        let unmarked = reviewer_route_agent(&config, ModelRoute::new("chat", "current"));
        assert!(reviewer_jev_config(&unmarked, &config.providers).is_none());
    }

    #[tokio::test]
    async fn new_session_inherits_current_model_expert_fake_and_reasoning_settings() {
        tokio::time::timeout(Duration::from_secs(10), async {
            let directory = tempfile::tempdir().unwrap();
            let config_path = directory.path().join("letcode.toml");
            fs::write(
                &config_path,
                r#"
active_provider = "test"
[providers.test]
protocol = "responses"
default_model = "default"
[providers.test.auth]
type = "bearer"
credential = "test-key"
[providers.test.endpoints]
base_url = "http://127.0.0.1:1"
[providers.test.models.current]
[providers.test.models.current.capabilities]
reasoning = true
generation = { reasoning = true }
[providers.test.models.current.generation]
reasoning_efforts = ["high"]
[providers.test.models.default]
[providers.test.models.expert]
"#,
            )
            .unwrap();
            let config = AppConfig::load_from_path(&config_path).unwrap();
            let current_route = ModelRoute::new("test", "current");
            let primary_factory =
                Arc::new(ConfiguredPrimaryRouteFactory::new_with_runtime_catalog(
                    config.providers.clone(),
                    config.global.retry.clone(),
                    config.runtime_catalog.clone(),
                ));
            let mut agent = Agent::new("current", 1, 1);
            agent.apply_prepared_route(
                primary_factory
                    .prepare_route(current_route.clone())
                    .unwrap(),
            );
            agent.set_primary_route_factory(primary_factory);
            agent
                .set_fake_client(Some(crate::fake::FakeClient::Codex))
                .unwrap();
            agent
                .set_reasoning_effort(ModelReasoningEffort::High)
                .unwrap();
            let transcript = parent_transcript(&config.global.sessions_dir);
            transcript
                .lock()
                .unwrap()
                .record_session_started(current_route.display_name())
                .unwrap();
            crate::configure_agent_runtime_snapshot_provider(&mut agent, &transcript);

            let (mut engine, _) = SessionEngine::start(
                agent,
                transcript.clone(),
                "Current".into(),
                SessionEngineConfig {
                    sessions_dir: config.global.sessions_dir.clone(),
                    model_routes: indexmap::IndexMap::from([
                        ("test/current".into(), current_route.clone()),
                        ("test/default".into(), ModelRoute::new("test", "default")),
                    ]),
                    route_api_key_configured: indexmap::IndexMap::from([
                        ("test/current".into(), true),
                        ("test/default".into(), true),
                        ("test/expert".into(), true),
                    ]),
                    new_session_default_route: ModelRoute::new("test", "default"),
                    new_session_default_expert_routes: indexmap::IndexMap::from([
                        ("explorer".into(), ModelRoute::new("test", "default")),
                        ("reviewer".into(), ModelRoute::new("test", "default")),
                    ]),
                    expert_model_routes: indexmap::IndexMap::from([
                        ("explorer".into(), ModelRoute::new("test", "expert")),
                        ("reviewer".into(), ModelRoute::new("test", "missing")),
                    ]),
                    expert_allowed_models: indexmap::IndexMap::new(),
                    legacy_expert_models: indexmap::IndexMap::new(),
                    providers: config.providers.clone(),
                    global_retry: config.global.retry.clone(),
                    provider_api_key_hints: indexmap::IndexMap::new(),
                    api_key_hint: String::new(),
                    mcp_config_path: config.config_path.clone(),
                    mcp_config: config.mcp.clone(),
                    runtime_catalog: config.runtime_catalog.clone(),
                },
            )
            .unwrap();
            let ingress = engine.take_ingress();
            let mut events = engine.take_event_egress().into_receiver();
            ingress.submit(SessionCommand::NewSession).unwrap();

            let mut started = None;
            let mut fake = None;
            let mut effort = None;
            while started.is_none() || fake.is_none() || effort.is_none() {
                match tokio::time::timeout(Duration::from_secs(5), events.recv())
                    .await
                    .unwrap()
                    .unwrap()
                {
                    SessionTransportEvent::SessionStarted { expert_models, .. } => {
                        started = Some(expert_models)
                    }
                    SessionTransportEvent::FakeClientChanged { client } => fake = Some(client),
                    SessionTransportEvent::ReasoningEffortChanged { effort: value } => {
                        effort = Some(value)
                    }
                    _ => {}
                }
            }
            let started = started.unwrap();
            assert_eq!(started.get("explorer").unwrap(), "test/expert");
            assert_eq!(started.get("reviewer").unwrap(), "test/default");
            assert_eq!(fake, Some(Some(crate::fake::FakeClient::Codex)));
            assert_eq!(effort, Some(ModelReasoningEffort::High));

            let records =
                crate::transcript::read_records(transcript.lock().unwrap().path()).unwrap();
            assert_eq!(
                crate::transcript::restore_latest_fake_client(&records),
                Some(crate::fake::FakeClient::Codex)
            );
            assert_eq!(
                crate::transcript::restore_latest_reasoning_effort(&records, "test/current",),
                Some(ModelReasoningEffort::High)
            );
            ingress.shutdown().unwrap();
            engine.join().await.unwrap();
        })
        .await
        .expect("new-session inheritance timed out");
    }

    #[tokio::test]
    async fn fresh_session_can_prepare_history_without_a_restore() {
        tokio::time::timeout(Duration::from_secs(10), async {
            let directory = tempfile::tempdir().unwrap();
            let config_path = directory.path().join("letcode.toml");
            fs::write(
                &config_path,
                r#"
active_provider = "test"
[providers.test]
protocol = "responses"
default_model = "model"
[providers.test.auth]
type = "bearer"
credential = "test-key"
[providers.test.endpoints]
base_url = "http://127.0.0.1:1"
[providers.test.models.model]
"#,
            )
            .unwrap();
            let config = AppConfig::load_from_path(&config_path).unwrap();
            let mut recorder = TranscriptRecorder::create(&config.global.sessions_dir).unwrap();
            recorder.record_session_started("test/model").unwrap();
            let transcript = Arc::new(StdMutex::new(recorder));
            let mut agent = Agent::new("model", 1, 1);
            agent.set_primary_route(ModelRoute::new("test", "model"));
            agent.set_resolved_model_route(Some(Arc::new(
                config
                    .runtime_catalog
                    .route("test", "model")
                    .unwrap()
                    .clone(),
            )));
            crate::configure_agent_runtime_snapshot_provider(&mut agent, &transcript);
            let settings = crate::session_engine_config(&config, Default::default(), String::new());
            let (mut engine, _) =
                SessionEngine::start(agent, transcript, "model".into(), settings).unwrap();
            let ingress = engine.take_ingress();
            let mut events = engine.take_event_egress().into_receiver();
            ingress.submit(SessionCommand::Compact).unwrap();
            loop {
                match events.recv().await.expect("engine event stream") {
                    SessionTransportEvent::CompactionNoProgress { .. } => break,
                    event @ (SessionTransportEvent::CompactionFailed
                    | SessionTransportEvent::Error(_)) => {
                        panic!("empty history must report no progress: {event:?}");
                    }
                    _ => {}
                }
            }
            ingress.shutdown().unwrap();
            engine.join().await.unwrap();
        })
        .await
        .expect("new-session history preparation timed out");
    }

    #[tokio::test]
    async fn resume_reports_the_recorded_permission_mode_before_session_resumed() {
        tokio::time::timeout(Duration::from_secs(10), async {
            let directory = tempfile::tempdir().unwrap();
            let config_path = directory.path().join("letcode.toml");
            fs::write(
                &config_path,
                r#"
active_provider = "test"
[providers.test]
protocol = "responses"
default_model = "model"
[providers.test.auth]
type = "bearer"
credential = "test-key"
[providers.test.endpoints]
base_url = "http://127.0.0.1:1"
[providers.test.models.model]
"#,
            )
            .unwrap();
            let config = AppConfig::load_from_path(&config_path).unwrap();
            let sessions_dir = config.global.sessions_dir.clone();

            let mut resumed = TranscriptRecorder::create(&sessions_dir).unwrap();
            resumed.record_session_started("test/model").unwrap();
            resumed
                .record_permission_mode_changed("default", "yolo")
                .unwrap();
            let resumed_session_id = resumed.session_id().to_string();
            drop(resumed);

            let mut recorder = TranscriptRecorder::create(&sessions_dir).unwrap();
            recorder.record_session_started("test/model").unwrap();
            let transcript = Arc::new(StdMutex::new(recorder));
            let route = ModelRoute::new("test", "model");
            let primary_factory =
                Arc::new(ConfiguredPrimaryRouteFactory::new_with_runtime_catalog(
                    config.providers.clone(),
                    config.global.retry.clone(),
                    config.runtime_catalog.clone(),
                ));
            let mut agent = Agent::new("model", 1, 1);
            agent.apply_prepared_route(primary_factory.prepare_route(route).unwrap());
            agent.set_primary_route_factory(primary_factory);
            // The live process runs in a different mode than the resumed session
            // recorded, so the announced mode can only come from the transcript.
            assert_eq!(agent.permission_mode().to_string(), "default");
            crate::configure_agent_runtime_snapshot_provider(&mut agent, &transcript);
            let settings = crate::session_engine_config(&config, Default::default(), String::new());
            let (mut engine, _) =
                SessionEngine::start(agent, transcript, "model".into(), settings).unwrap();
            let ingress = engine.take_ingress();
            let mut events = engine.take_event_egress().into_receiver();

            ingress
                .submit_transitional(SessionEngineCommand::ResumeSession(
                    resumed_session_id.clone(),
                ))
                .unwrap();

            let mut mode_index = None;
            let mut resumed_index = None;
            let mut index = 0;
            while resumed_index.is_none() {
                let event = tokio::time::timeout(Duration::from_secs(5), events.recv())
                    .await
                    .expect("resume must report the resumed session")
                    .expect("engine event stream");
                match event {
                    SessionTransportEvent::PermissionModeChanged { mode } => {
                        assert_eq!(mode, "yolo", "restored mode is announced as-is");
                        mode_index = Some(index);
                    }
                    SessionTransportEvent::SessionResumed { session_id, .. } => {
                        assert_eq!(session_id, resumed_session_id);
                        resumed_index = Some(index);
                    }
                    SessionTransportEvent::Error(error) => panic!("resume failed: {error:?}"),
                    _ => {}
                }
                index += 1;
            }
            let mode_index = mode_index.expect("the recorded permission mode is reported");
            let resumed_index = resumed_index.expect("SessionResumed was reported");
            assert!(
                mode_index < resumed_index,
                "the restored mode must precede SessionResumed: mode at {mode_index}, resumed at {resumed_index}"
            );

            ingress.shutdown().unwrap();
            engine.join().await.unwrap();
        })
        .await
        .expect("resume of a recorded permission mode timed out");
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
    async fn child_view_refresh_skips_unchanged_journals_and_follows_the_moving_ones() {
        let sessions_dir = temp_sessions_dir();
        let parent = parent_transcript(&sessions_dir);
        add_child(&parent, &sessions_dir, "run-1", 1);
        let visible_child = add_child(&parent, &sessions_dir, "run-2", 2);

        let (tx, mut rx) = mpsc::unbounded_channel();
        let mut visible_child_session_id = Some(visible_child.clone());
        let mut view_state = None;
        let mut cache = None;

        refresh_visible_child_session_view(
            &parent,
            &tx,
            &sessions_dir,
            &mut visible_child_session_id,
            &mut view_state,
            &mut cache,
        )
        .await;
        assert_eq!(
            child_view_event(rx.try_recv().expect("the first pass reports the view")),
            (visible_child.clone(), 1, 2)
        );
        assert!(
            cache.is_some(),
            "a completed pass is kept for the next tick"
        );

        refresh_visible_child_session_view(
            &parent,
            &tx,
            &sessions_dir,
            &mut visible_child_session_id,
            &mut view_state,
            &mut cache,
        )
        .await;
        assert!(
            rx.try_recv().is_err(),
            "journals that have not moved must not be reported again"
        );

        let mut child = TranscriptRecorder::open(
            crate::transcript::child_sessions_dir(&sessions_dir),
            visible_child.clone(),
        )
        .expect("open the visible child transcript");
        child
            .record_user_message("more child work")
            .expect("record the new child message");
        drop(child);

        refresh_visible_child_session_view(
            &parent,
            &tx,
            &sessions_dir,
            &mut visible_child_session_id,
            &mut view_state,
            &mut cache,
        )
        .await;
        let (_, index, total) =
            child_view_event(rx.try_recv().expect("a grown child journal is reported"));
        assert_eq!((index, total), (1, 2), "the sibling count has not changed");

        add_child(&parent, &sessions_dir, "run-3", 3);
        refresh_visible_child_session_view(
            &parent,
            &tx,
            &sessions_dir,
            &mut visible_child_session_id,
            &mut view_state,
            &mut cache,
        )
        .await;
        let (child_session_id, index, total) = child_view_event(
            rx.try_recv()
                .expect("a moved parent journal is re-resolved"),
        );
        assert_eq!(
            (child_session_id.as_str(), index, total),
            (visible_child.as_str(), 1, 3)
        );

        view_state = None;
        refresh_visible_child_session_view(
            &parent,
            &tx,
            &sessions_dir,
            &mut visible_child_session_id,
            &mut view_state,
            &mut cache,
        )
        .await;
        assert_eq!(
            child_view_event(
                rx.try_recv()
                    .expect("a cleared view state is answered with a full pass")
            )
            .2,
            3
        );
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

    #[test]
    fn direct_expert_execution_uses_the_selected_expert_provider_credential() {
        let route_api_key_configured = indexmap::IndexMap::from([
            ("primary/shared".into(), true),
            ("expert/shared".into(), false),
        ]);
        let mut agent = Agent::new("shared", 1, 1);
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

        let agent = Agent::new("shared", 1, 1);
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
            protocol = "responses"
            default_model = "old"
            [providers.primary.auth]
            type = "bearer"
            credential = "old-key"
            [providers.primary.endpoints]
            base_url = "https://example.invalid/v1"

            [providers.primary.models.old]
            [providers.primary.models.new]
            "#;
        let bad_contents = r#"
            active_provider = "primary"

            [providers.primary]
            unknown = true
            protocol = "responses"
            default_model = "new"
            [providers.primary.auth]
            type = "bearer"
            credential = "new-key"
            [providers.primary.endpoints]
            base_url = "https://example.invalid/v1"

            [providers.primary.models.new]
            "#;
        fs::write(&old_path, old_contents).expect("write old config");
        fs::write(&bad_path, bad_contents).expect("write invalid reload config");
        let old_config = AppConfig::load_from_path(&old_path).expect("old config should load");
        let route = ModelRoute::new("primary", "old");
        let mut agent = Agent::new("old", 1, 1);
        agent.set_primary_route(route.clone());

        let mut model_routes = indexmap::IndexMap::from([(route.display_name(), route.clone())]);
        let mut route_api_key_configured = indexmap::IndexMap::from([(route.display_name(), true)]);
        let mut expert_model_routes =
            indexmap::IndexMap::from([(String::from("explorer"), route.clone())]);
        let mut new_session_default_expert_routes = expert_model_routes.clone();
        let mut expert_allowed_models = crate::delegation::supported_agent_names()
            .map(|name| (name.to_string(), Vec::new()))
            .collect();
        let mut legacy_expert_models =
            indexmap::IndexMap::from([(String::from("explorer"), String::from("old"))]);
        let mut providers = old_config.providers.clone();
        let mut global_retry = old_config.global.retry.clone();
        let mut provider_api_key_hints =
            indexmap::IndexMap::from([(String::from("primary"), String::from("old hint"))]);
        let mut new_session_default_route = route.clone();
        let mut runtime_catalog = old_config.runtime_catalog.clone();
        let old_model_routes = model_routes.clone();
        let old_route_api_key_configured = route_api_key_configured.clone();
        let old_expert_model_routes = expert_model_routes.clone();
        let old_legacy_expert_models = legacy_expert_models.clone();
        let old_providers = providers.clone();
        let old_global_retry = global_retry.clone();
        let old_provider_api_key_hints = provider_api_key_hints.clone();
        let old_new_session_default_route = new_session_default_route.clone();
        let (event_tx, mut event_rx) = mpsc::unbounded_channel();

        assert!(
            apply_config_reload(
                &mut agent,
                &bad_path,
                &mut model_routes,
                &mut route_api_key_configured,
                &mut expert_model_routes,
                &mut new_session_default_expert_routes,
                &mut expert_allowed_models,
                &mut legacy_expert_models,
                &mut providers,
                &mut global_retry,
                &mut provider_api_key_hints,
                &mut new_session_default_route,
                &mut runtime_catalog,
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
        assert_eq!(new_session_default_route, old_new_session_default_route);
        assert_eq!(
            runtime_catalog.fingerprint(),
            old_config.runtime_catalog.fingerprint()
        );
        assert!(event_rx.try_recv().is_err());

        let _ = fs::remove_file(old_path);
        let _ = fs::remove_file(bad_path);
    }

    #[test]
    fn config_reload_preserves_session_reasoning_effort() {
        let path = std::env::temp_dir().join(format!(
            "letcode-engine-reload-non-runtime-write-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system time is valid")
                .as_nanos()
        ));
        fs::write(
            &path,
            r#"
            active_provider = "primary"

            [mcp.alpha]
            type = "local"
            command = ["alpha"]
            enabled = true

            [providers.primary]
            protocol = "responses"
            default_model = "primary-model"
            [providers.primary.auth]
            type = "bearer"
            credential = "primary-key"
            [providers.primary.endpoints]
            base_url = "https://primary.example.invalid/v1"

            [providers.primary.models.primary-model]
            [providers.primary.models.primary-model.capabilities]
            reasoning = true
            [providers.primary.models.primary-model.capabilities.generation]
            reasoning = true
            [providers.primary.models.primary-model.generation]
            reasoning_effort = "high"
            "#,
        )
        .expect("write initial config");

        let config = AppConfig::load_from_path(&path).expect("initial config should load");
        let route = config.active_route();
        let provider = config
            .provider_for_route(&route)
            .expect("active provider is configured");
        let mut agent = Agent::new(route.model.clone(), 1, 1);
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
        agent
            .set_reasoning_effort(ModelReasoningEffort::High)
            .expect("set session reasoning effort");
        agent.set_compaction_config(config.global.compaction.clone());
        agent.set_tool_timeout_secs(config.global.tool_timeout_secs);
        agent
            .set_tool_parallelism(
                config
                    .tools
                    .parallelism
                    .iter()
                    .map(|(name, mode)| (name.clone(), *mode)),
            )
            .expect("tool parallelism");
        agent.set_retry_config(
            provider
                .retry
                .clone()
                .unwrap_or_else(|| config.global.retry.clone()),
        );

        let mut model_routes = indexmap::IndexMap::from([(route.display_name(), route.clone())]);
        let mut route_api_key_configured = indexmap::IndexMap::from([(route.display_name(), true)]);
        let mut expert_model_routes = indexmap::IndexMap::new();
        let mut new_session_default_expert_routes = indexmap::IndexMap::new();
        let mut expert_allowed_models = crate::delegation::supported_agent_names()
            .map(|name| (name.to_string(), Vec::new()))
            .collect();
        let mut legacy_expert_models = indexmap::IndexMap::new();
        let mut providers = config.providers.clone();
        let mut global_retry = config.global.retry.clone();
        let mut provider_api_key_hints = config
            .providers
            .keys()
            .map(|name| {
                (
                    name.clone(),
                    format!(
                        "Set [providers.{name}.auth].credential in {} or set {} environment variable.",
                        config.config_path.display(),
                        crate::config::provider_api_key_env_var(name)
                    ),
                )
            })
            .collect();
        let mut new_session_default_route = route;
        let mut runtime_catalog = config.runtime_catalog.clone();
        let (event_tx, mut event_rx) = mpsc::unbounded_channel();

        crate::config::persist_mcp_server_enabled(&path, "alpha", false)
            .expect("persist non-reloadable global setting");
        apply_config_reload(
            &mut agent,
            &path,
            &mut model_routes,
            &mut route_api_key_configured,
            &mut expert_model_routes,
            &mut new_session_default_expert_routes,
            &mut expert_allowed_models,
            &mut legacy_expert_models,
            &mut providers,
            &mut global_retry,
            &mut provider_api_key_hints,
            &mut new_session_default_route,
            &mut runtime_catalog,
            &event_tx,
        )
        .expect("non-reloadable config write should preserve session settings");

        assert_eq!(
            agent.reasoning_effort(),
            Some(ModelReasoningEffort::High),
            "configuration reload must preserve the session reasoning selection"
        );
        assert!(event_rx.try_recv().is_err());
        let _ = fs::remove_file(path);
    }

    #[test]
    fn reload_keeps_current_route_when_it_leaves_the_catalog() {
        let path = std::env::temp_dir().join(format!(
            "letcode-engine-reload-removed-current-route-{}",
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
            protocol = "responses"
            default_model = "primary-model"
            [providers.primary.auth]
            type = "bearer"
            credential = "primary-key"
            [providers.primary.endpoints]
            base_url = "https://primary.example.invalid/v1"

            [providers.primary.models.primary-model]
            [providers.primary.models.session-model]
            "#,
        )
        .expect("write initial config");

        let current_route = ModelRoute::new("primary", "session-model");
        let default_route = ModelRoute::new("primary", "primary-model");
        let mut agent = Agent::new(current_route.model.clone(), 1, 1);
        agent.set_primary_route(current_route.clone());
        let initial_config = AppConfig::load_from_path(&path).expect("initial config should load");
        let mut model_routes = indexmap::IndexMap::new();
        let mut route_api_key_configured = indexmap::IndexMap::new();
        let mut expert_model_routes = indexmap::IndexMap::new();
        let mut new_session_default_expert_routes = indexmap::IndexMap::new();
        let mut expert_allowed_models = crate::delegation::supported_agent_names()
            .map(|name| (name.to_string(), Vec::new()))
            .collect();
        let mut legacy_expert_models = indexmap::IndexMap::new();
        let mut providers = initial_config.providers.clone();
        let mut global_retry = initial_config.global.retry.clone();
        let mut provider_api_key_hints = indexmap::IndexMap::new();
        let mut new_session_default_route = current_route.clone();
        let mut runtime_catalog = initial_config.runtime_catalog.clone();
        let (event_tx, mut event_rx) = mpsc::unbounded_channel();

        fs::write(
            &path,
            r#"
            active_provider = "primary"

            [providers.primary]
            protocol = "responses"
            default_model = "primary-model"
            [providers.primary.auth]
            type = "bearer"
            credential = "primary-key"
            [providers.primary.endpoints]
            base_url = "https://primary.example.invalid/v1"

            [providers.primary.models.primary-model]
            "#,
        )
        .expect("remove current session model from config");

        apply_config_reload(
            &mut agent,
            &path,
            &mut model_routes,
            &mut route_api_key_configured,
            &mut expert_model_routes,
            &mut new_session_default_expert_routes,
            &mut expert_allowed_models,
            &mut legacy_expert_models,
            &mut providers,
            &mut global_retry,
            &mut provider_api_key_hints,
            &mut new_session_default_route,
            &mut runtime_catalog,
            &event_tx,
        )
        .expect("catalog reload should keep the current route alive");

        assert_eq!(agent.primary_route(), Some(&current_route));
        assert_eq!(new_session_default_route, default_route);
        assert_eq!(
            route_api_key_configured.get(&current_route.display_name()),
            Some(&true),
            "the retained current route keeps its credential state"
        );
        let events = std::iter::from_fn(|| event_rx.try_recv().ok()).collect::<Vec<_>>();
        assert!(events.iter().any(|event| matches!(
            event,
            SessionTransportEvent::Notice(notice)
                if notice.message.contains("is no longer in the configured model catalog")
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            SessionTransportEvent::ModelCatalogUpdated(catalog)
                if catalog.models.iter().all(|model| model.id != current_route.display_name())
        )));
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, SessionTransportEvent::ModelChanged { .. }))
        );

        apply_config_reload(
            &mut agent,
            &path,
            &mut model_routes,
            &mut route_api_key_configured,
            &mut expert_model_routes,
            &mut new_session_default_expert_routes,
            &mut expert_allowed_models,
            &mut legacy_expert_models,
            &mut providers,
            &mut global_retry,
            &mut provider_api_key_hints,
            &mut new_session_default_route,
            &mut runtime_catalog,
            &event_tx,
        )
        .expect("duplicate removed-route reload should be a no-op");
        assert!(event_rx.try_recv().is_err());

        let _ = fs::remove_file(path);
    }

    #[test]
    fn reload_keeps_current_route_when_its_provider_leaves_the_catalog() {
        let path = std::env::temp_dir().join(format!(
            "letcode-engine-reload-removed-current-provider-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system time is valid")
                .as_nanos()
        ));
        let initial = r#"
            active_provider = "primary"
            [providers.primary]
            protocol = "responses"
            default_model = "session-model"
            [providers.primary.auth]
            type = "bearer"
            credential = "primary-key"
            [providers.primary.endpoints]
            base_url = "https://primary.example.invalid/v1"
            [providers.primary.models.session-model]
            [providers.secondary]
            protocol = "responses"
            default_model = "secondary-model"
            [providers.secondary.auth]
            type = "bearer"
            credential = "secondary-key"
            [providers.secondary.endpoints]
            base_url = "https://secondary.example.invalid/v1"
            [providers.secondary.models.secondary-model]
        "#;
        fs::write(&path, initial).expect("write initial config");
        let initial_config = AppConfig::load_from_path(&path).expect("load initial config");
        let current_route = ModelRoute::new("primary", "session-model");
        let next_default = ModelRoute::new("secondary", "secondary-model");
        let provider = initial_config.provider_for_route(&current_route).unwrap();
        let mut agent = Agent::new(current_route.model.clone(), 1, 1);
        agent.set_primary_route(current_route.clone());
        let mut model_routes = indexmap::IndexMap::from([
            (current_route.display_name(), current_route.clone()),
            (next_default.display_name(), next_default.clone()),
        ]);
        let mut route_api_key_configured = indexmap::IndexMap::from([
            (current_route.display_name(), true),
            (next_default.display_name(), true),
        ]);
        let mut expert_model_routes = indexmap::IndexMap::new();
        let mut new_session_default_expert_routes = indexmap::IndexMap::new();
        let mut expert_allowed_models = crate::delegation::supported_agent_names()
            .map(|name| (name.to_string(), Vec::new()))
            .collect();
        let mut legacy_expert_models = indexmap::IndexMap::new();
        let mut providers = initial_config.providers.clone();
        let mut global_retry = initial_config.global.retry.clone();
        let mut provider_api_key_hints = indexmap::IndexMap::new();
        let mut new_session_default_route = current_route.clone();
        let mut runtime_catalog = initial_config.runtime_catalog.clone();
        let (event_tx, mut event_rx) = mpsc::unbounded_channel();

        let changed = r#"
            active_provider = "secondary"
            [providers.secondary]
            protocol = "responses"
            default_model = "secondary-model"
            [providers.secondary.auth]
            type = "bearer"
            credential = "secondary-key"
            [providers.secondary.endpoints]
            base_url = "https://secondary.example.invalid/v1"
            [providers.secondary.models.secondary-model]
        "#;
        fs::write(&path, changed).expect("remove current provider");

        apply_config_reload(
            &mut agent,
            &path,
            &mut model_routes,
            &mut route_api_key_configured,
            &mut expert_model_routes,
            &mut new_session_default_expert_routes,
            &mut expert_allowed_models,
            &mut legacy_expert_models,
            &mut providers,
            &mut global_retry,
            &mut provider_api_key_hints,
            &mut new_session_default_route,
            &mut runtime_catalog,
            &event_tx,
        )
        .expect("removed provider reload keeps live route");

        assert_eq!(agent.primary_route(), Some(&current_route));
        assert_eq!(new_session_default_route, next_default);
        assert!(providers.get("primary").is_some_and(|provider| {
            provider.has_model("session-model") && !provider.api_key.is_empty()
        }));
        assert!(!model_routes.contains_key(&current_route.display_name()));
        assert_eq!(
            route_api_key_configured.get(&current_route.display_name()),
            Some(&true)
        );
        while event_rx.try_recv().is_ok() {}
        apply_config_reload(
            &mut agent,
            &path,
            &mut model_routes,
            &mut route_api_key_configured,
            &mut expert_model_routes,
            &mut new_session_default_expert_routes,
            &mut expert_allowed_models,
            &mut legacy_expert_models,
            &mut providers,
            &mut global_retry,
            &mut provider_api_key_hints,
            &mut new_session_default_route,
            &mut runtime_catalog,
            &event_tx,
        )
        .expect("duplicate removed-provider reload should be a no-op");
        assert!(event_rx.try_recv().is_err());
        assert_eq!(
            route_api_key_configured.get(&current_route.display_name()),
            Some(&true)
        );
        let _ = fs::remove_file(path);
    }

    #[test]
    fn reload_keeps_current_expert_route_when_it_leaves_the_catalog() {
        let path = std::env::temp_dir().join(format!(
            "letcode-engine-reload-removed-current-expert-route-{}",
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
            protocol = "responses"
            default_model = "primary-model"
            [providers.primary.auth]
            type = "bearer"
            credential = "primary-key"
            [providers.primary.endpoints]
            base_url = "https://primary.example.invalid/v1"

            [providers.primary.models.primary-model]
            [providers.primary.models.expert-model]
            "#,
        )
        .expect("write initial config");

        let primary_route = ModelRoute::new("primary", "primary-model");
        let expert_route = ModelRoute::new("primary", "expert-model");
        let mut agent = Agent::new(primary_route.model.clone(), 1, 1);
        agent.set_primary_route(primary_route.clone());
        let initial_config = AppConfig::load_from_path(&path).expect("initial config should load");
        let mut model_routes = indexmap::IndexMap::new();
        let mut route_api_key_configured = indexmap::IndexMap::new();
        let mut expert_model_routes =
            indexmap::IndexMap::from([("explorer".into(), expert_route.clone())]);
        let mut new_session_default_expert_routes = indexmap::IndexMap::new();
        let mut expert_allowed_models = crate::delegation::supported_agent_names()
            .map(|name| (name.to_string(), Vec::new()))
            .collect();
        let mut legacy_expert_models = indexmap::IndexMap::new();
        let mut providers = initial_config.providers.clone();
        let mut global_retry = initial_config.global.retry.clone();
        let mut provider_api_key_hints = indexmap::IndexMap::new();
        let mut new_session_default_route = primary_route.clone();
        let mut runtime_catalog = initial_config.runtime_catalog.clone();
        let (event_tx, mut event_rx) = mpsc::unbounded_channel();

        fs::write(
            &path,
            r#"
            active_provider = "primary"

            [providers.primary]
            protocol = "responses"
            default_model = "primary-model"
            [providers.primary.auth]
            type = "bearer"
            credential = "primary-key"
            [providers.primary.endpoints]
            base_url = "https://primary.example.invalid/v1"

            [providers.primary.models.primary-model]
            "#,
        )
        .expect("remove current expert model from config");

        apply_config_reload(
            &mut agent,
            &path,
            &mut model_routes,
            &mut route_api_key_configured,
            &mut expert_model_routes,
            &mut new_session_default_expert_routes,
            &mut expert_allowed_models,
            &mut legacy_expert_models,
            &mut providers,
            &mut global_retry,
            &mut provider_api_key_hints,
            &mut new_session_default_route,
            &mut runtime_catalog,
            &event_tx,
        )
        .expect("catalog reload should keep the current expert route alive");

        assert_eq!(expert_model_routes.get("explorer"), Some(&expert_route));
        assert!(
            providers
                .get("primary")
                .is_some_and(|provider| provider.has_model("expert-model"))
        );
        assert!(
            !model_routes.contains_key(&expert_route.display_name()),
            "removed expert route must not remain selectable in the global catalog"
        );
        assert_eq!(
            route_api_key_configured.get(&expert_route.display_name()),
            Some(&true),
            "the retained session route keeps the credential state of its live provider"
        );
        while event_rx.try_recv().is_ok() {}
        apply_config_reload(
            &mut agent,
            &path,
            &mut model_routes,
            &mut route_api_key_configured,
            &mut expert_model_routes,
            &mut new_session_default_expert_routes,
            &mut expert_allowed_models,
            &mut legacy_expert_models,
            &mut providers,
            &mut global_retry,
            &mut provider_api_key_hints,
            &mut new_session_default_route,
            &mut runtime_catalog,
            &event_tx,
        )
        .expect("duplicate removed-expert reload should be a no-op");
        assert!(event_rx.try_recv().is_err());

        let _ = fs::remove_file(path);
    }

    #[test]
    fn reload_runtime_change_updates_fingerprint_and_reinstalls_current_route() {
        let path = std::env::temp_dir().join(format!(
            "letcode-engine-reload-runtime-change-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system time is valid")
                .as_nanos()
        ));
        let initial = r#"
            active_provider = "primary"
            [providers.primary]
            protocol = "responses"
            default_model = "model"
            [providers.primary.auth]
            type = "bearer"
            credential = "key"
            [providers.primary.endpoints]
            base_url = "https://old.example.invalid/v1"
            [providers.primary.models.model]
        "#;
        fs::write(&path, initial).expect("write initial config");
        let initial_config = AppConfig::load_from_path(&path).expect("load initial config");
        let route = initial_config.active_route();
        let provider = initial_config.active_provider().1;
        let mut agent = Agent::new(route.model.clone(), 1, 1);
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
        agent.install_provider_usage_anchor_for_test(crate::agent::TokenUsageEstimate {
            used_tokens: 10,
            context_window_tokens: 1_000,
            input_tokens: 8,
            output_tokens: 2,
            cached_tokens: 0,
        });

        let mut model_routes = indexmap::IndexMap::from([(route.display_name(), route.clone())]);
        let mut route_api_key_configured = indexmap::IndexMap::from([(route.display_name(), true)]);
        let mut expert_model_routes = indexmap::IndexMap::new();
        let mut new_session_default_expert_routes = indexmap::IndexMap::new();
        let mut expert_allowed_models = crate::delegation::supported_agent_names()
            .map(|name| (name.to_string(), Vec::new()))
            .collect();
        let mut legacy_expert_models = indexmap::IndexMap::new();
        let mut providers = initial_config.providers.clone();
        let mut global_retry = initial_config.global.retry.clone();
        let mut provider_api_key_hints = indexmap::IndexMap::new();
        let mut new_session_default_route = route.clone();
        let mut runtime_catalog = initial_config.runtime_catalog.clone();
        let (event_tx, _event_rx) = mpsc::unbounded_channel();

        let changed = initial.replace(
            "https://old.example.invalid/v1",
            "https://new.example.invalid/v1",
        );
        fs::write(&path, changed).expect("write changed config");
        let expected = AppConfig::load_from_path(&path)
            .expect("load changed config")
            .runtime_catalog
            .fingerprint()
            .clone();

        apply_config_reload(
            &mut agent,
            &path,
            &mut model_routes,
            &mut route_api_key_configured,
            &mut expert_model_routes,
            &mut new_session_default_expert_routes,
            &mut expert_allowed_models,
            &mut legacy_expert_models,
            &mut providers,
            &mut global_retry,
            &mut provider_api_key_hints,
            &mut new_session_default_route,
            &mut runtime_catalog,
            &event_tx,
        )
        .expect("runtime change reload succeeds");

        assert_eq!(runtime_catalog.fingerprint(), &expected);
        assert_eq!(agent.primary_route(), Some(&route));
        assert!(
            agent.provider_usage_anchor_for_test().is_none(),
            "reinstalling a changed route clears provider-local usage state"
        );
        let _ = fs::remove_file(path);
    }

    #[test]
    fn default_model_only_reload_updates_defaults_without_reinstalling_active_route() {
        let path = std::env::temp_dir().join(format!(
            "letcode-engine-reload-default-only-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system time is valid")
                .as_nanos()
        ));
        let initial = r#"
            active_provider = "primary"
            [providers.primary]
            protocol = "responses"
            default_model = "a"
            [providers.primary.auth]
            type = "bearer"
            credential = "key"
            [providers.primary.endpoints]
            base_url = "https://example.invalid/v1"
            [providers.primary.models.a]
            [providers.primary.models.b]
        "#;
        fs::write(&path, initial).expect("write initial config");
        let initial_config = AppConfig::load_from_path(&path).expect("load initial config");
        let active_route = ModelRoute::new("primary", "a");
        let next_default = ModelRoute::new("primary", "b");
        let provider = initial_config.active_provider().1;
        let mut agent = Agent::new(active_route.model.clone(), 1, 1);
        agent.set_primary_route(active_route.clone());
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
            used_tokens: 10,
            context_window_tokens: 1_000,
            input_tokens: 8,
            output_tokens: 2,
            cached_tokens: 0,
        };
        agent.install_provider_usage_anchor_for_test(usage);

        let mut model_routes = initial_config
            .providers
            .iter()
            .flat_map(|(provider_name, provider)| {
                provider.models.keys().map(move |model| {
                    let route = ModelRoute::new(provider_name, model);
                    (route.display_name(), route)
                })
            })
            .collect();
        let mut route_api_key_configured = indexmap::IndexMap::from([
            (active_route.display_name(), true),
            (next_default.display_name(), true),
        ]);
        let mut expert_model_routes = indexmap::IndexMap::new();
        let mut new_session_default_expert_routes = indexmap::IndexMap::new();
        let mut expert_allowed_models = crate::delegation::supported_agent_names()
            .map(|name| (name.to_string(), Vec::new()))
            .collect();
        let mut legacy_expert_models = indexmap::IndexMap::new();
        let mut providers = initial_config.providers.clone();
        let mut global_retry = initial_config.global.retry.clone();
        let mut provider_api_key_hints = indexmap::IndexMap::new();
        let mut new_session_default_route = active_route.clone();
        let mut runtime_catalog = initial_config.runtime_catalog.clone();
        let (event_tx, mut event_rx) = mpsc::unbounded_channel();

        fs::write(
            &path,
            initial.replace("default_model = \"a\"", "default_model = \"b\""),
        )
        .expect("write default-only config");
        let expected = AppConfig::load_from_path(&path)
            .expect("load default-only config")
            .runtime_catalog
            .fingerprint()
            .clone();

        apply_config_reload(
            &mut agent,
            &path,
            &mut model_routes,
            &mut route_api_key_configured,
            &mut expert_model_routes,
            &mut new_session_default_expert_routes,
            &mut expert_allowed_models,
            &mut legacy_expert_models,
            &mut providers,
            &mut global_retry,
            &mut provider_api_key_hints,
            &mut new_session_default_route,
            &mut runtime_catalog,
            &event_tx,
        )
        .expect("default-only reload succeeds");

        assert_eq!(runtime_catalog.fingerprint(), &expected);
        assert_eq!(new_session_default_route, next_default);
        assert_eq!(agent.primary_route(), Some(&active_route));
        assert_eq!(agent.provider_usage_anchor_for_test(), Some(usage));
        let events = std::iter::from_fn(|| event_rx.try_recv().ok()).collect::<Vec<_>>();
        assert!(
            events
                .iter()
                .any(|event| matches!(event, SessionTransportEvent::ModelCatalogUpdated(_)))
        );
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, SessionTransportEvent::ModelChanged { .. }))
        );
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
            protocol = "responses"
            default_model = "primary-model"
            [providers.primary.auth]
            type = "bearer"
            credential = "primary-key"
            [providers.primary.endpoints]
            base_url = "https://primary.example.invalid/v1"

            [providers.primary.models.primary-model]

            [providers.secondary]
            protocol = "responses"
            default_model = "secondary-model"
            [providers.secondary.auth]
            type = "bearer"
            credential = "secondary-key"
            [providers.secondary.endpoints]
            base_url = "https://secondary.example.invalid/v1"

            [providers.secondary.models.secondary-model]
            "#,
        )
        .expect("write initial active route config");

        let primary_route = ModelRoute::new("primary", "primary-model");
        let secondary_route = ModelRoute::new("secondary", "secondary-model");
        let mut agent = Agent::new(primary_route.model.clone(), 1, 1);
        agent.set_primary_route(primary_route.clone());
        let initial_config = AppConfig::load_from_path(&path).expect("initial config should load");
        let mut model_routes = indexmap::IndexMap::new();
        let mut route_api_key_configured = indexmap::IndexMap::new();
        let mut expert_model_routes = indexmap::IndexMap::new();
        let mut new_session_default_expert_routes = indexmap::IndexMap::new();
        let mut expert_allowed_models = crate::delegation::supported_agent_names()
            .map(|name| (name.to_string(), Vec::new()))
            .collect();
        let mut legacy_expert_models = indexmap::IndexMap::new();
        let mut providers = initial_config.providers.clone();
        let mut global_retry = initial_config.global.retry.clone();
        let mut provider_api_key_hints = indexmap::IndexMap::new();
        let mut new_session_default_route = primary_route.clone();
        let mut runtime_catalog = initial_config.runtime_catalog.clone();
        let (event_tx, mut event_rx) = mpsc::unbounded_channel();

        fs::write(
            &path,
            r#"
            active_provider = "secondary"

            [providers.primary]
            protocol = "responses"
            default_model = "primary-model"
            [providers.primary.auth]
            type = "bearer"
            credential = "primary-key"
            [providers.primary.endpoints]
            base_url = "https://primary.example.invalid/v1"

            [providers.primary.models.primary-model]

            [providers.secondary]
            protocol = "responses"
            default_model = "secondary-model"
            [providers.secondary.auth]
            type = "bearer"
            credential = "secondary-key"
            [providers.secondary.endpoints]
            base_url = "https://secondary.example.invalid/v1"

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
            &mut new_session_default_expert_routes,
            &mut expert_allowed_models,
            &mut legacy_expert_models,
            &mut providers,
            &mut global_retry,
            &mut provider_api_key_hints,
            &mut new_session_default_route,
            &mut runtime_catalog,
            &event_tx,
        )
        .expect("active route config should reload");

        assert_eq!(agent.primary_route(), Some(&primary_route));
        assert_eq!(new_session_default_route, secondary_route);
        assert!(matches!(
            event_rx.try_recv(),
            Ok(SessionTransportEvent::ModelCatalogUpdated(_))
        ));
        assert!(event_rx.try_recv().is_err());

        let _ = fs::remove_file(path);
    }

    const STUB_ANSWER: &str = "the session is still usable";

    struct HoldingTool;

    #[async_trait::async_trait]
    impl ToolHandler for HoldingTool {
        fn name(&self) -> &str {
            "test__hold"
        }

        fn description(&self) -> &str {
            "Holds an assistant tool call batch open"
        }

        fn parameters(&self) -> serde_json::Value {
            json!({ "type": "object", "properties": {} })
        }

        fn permission_class(&self) -> crate::permission::ToolPermissionClass {
            crate::permission::ToolPermissionClass::Read
        }

        async fn execute(&self, _args: serde_json::Value) -> Result<serde_json::Value> {
            std::future::pending().await
        }
    }

    async fn serve_responses_stub(listener: tokio::net::TcpListener) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let mut served = 0usize;
        while let Ok(Ok((mut stream, _))) =
            tokio::time::timeout(Duration::from_secs(5), listener.accept()).await
        {
            let mut request = Vec::new();
            let mut chunk = [0u8; 4096];
            let (header_end, content_length) = loop {
                let read = stream.read(&mut chunk).await.expect("stub request read");
                assert!(read > 0, "stub request closed before headers");
                request.extend_from_slice(&chunk[..read]);
                let Some(at) = request.windows(4).position(|part| part == b"\r\n\r\n") else {
                    continue;
                };
                let header_end = at + 4;
                let headers = std::str::from_utf8(&request[..header_end]).unwrap();
                let content_length = headers
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse::<usize>().unwrap())
                    })
                    .expect("responses request has content length");
                break (header_end, content_length);
            };
            while request.len() < header_end + content_length {
                let read = stream.read(&mut chunk).await.expect("stub body read");
                assert!(read > 0, "stub request closed before body");
                request.extend_from_slice(&chunk[..read]);
            }

            let body = if served == 0 {
                stub_tool_call_sse()
            } else {
                stub_text_sse()
            };
            served += 1;
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            stream
                .write_all(response.as_bytes())
                .await
                .expect("stub response write");
        }
    }

    fn stub_tool_call_sse() -> String {
        [
            json!({"type":"response.output_item.added","item":{"type":"function_call","id":"item-1","call_id":"call-1","name":"test__hold"}}),
            json!({"type":"response.function_call_arguments.done","item_id":"item-1","arguments":"{}"}),
            json!({"type":"response.output_item.done","item":{"type":"function_call","id":"item-1","call_id":"call-1","name":"test__hold","arguments":"{}"}}),
            json!({"type":"response.completed","response":{"id":"resp-1","status":"completed","usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15}}}),
        ]
        .iter()
        .map(|event| format!("data: {event}\n\n"))
        .collect()
    }

    fn stub_text_sse() -> String {
        [
            json!({"type":"response.output_text.delta","delta":STUB_ANSWER}),
            json!({"type":"response.completed","response":{"id":"resp-2","status":"completed","usage":{"input_tokens":12,"output_tokens":6,"total_tokens":18}}}),
        ]
        .iter()
        .map(|event| format!("data: {event}\n\n"))
        .collect()
    }

    fn open_tool_call_batch_recorded(
        transcript: &Arc<StdMutex<TranscriptRecorder>>,
    ) -> Result<bool> {
        let path = transcript
            .lock()
            .map_err(|_| anyhow!("transcript recorder poisoned"))?
            .path()
            .to_path_buf();
        let records = read_records(&path)?;
        Ok(records.iter().any(|record| {
            matches!(
                &record.event,
                TranscriptEvent::AssistantTurn(turn) if !turn.calls.is_empty()
            )
        }))
    }

    fn observed_error(error: &ErrorEvent) -> String {
        match error.details.as_deref().filter(|detail| !detail.is_empty()) {
            Some(detail) => format!("{} ({detail})", error.message),
            None => error.message.clone(),
        }
    }

    #[tokio::test]
    async fn a_recorded_background_result_is_delivered_once_and_continues_the_session() {
        tokio::time::timeout(Duration::from_secs(30), async {
            let directory = tempfile::tempdir().unwrap();
            let config_path = directory.path().join("letcode.toml");
            let sessions_dir = directory.path().join("sessions");
            fs::write(
                &config_path,
                format!(
                    r#"
active_provider = "test"

[global]
sessions_dir = "{}"

[providers.test]
protocol = "responses"
default_model = "model"
[providers.test.auth]
type = "bearer"
credential = "test-key"
[providers.test.endpoints]
base_url = "http://127.0.0.1:1"
[providers.test.models.model]
# Retains the pre-recorded history, so historian work stays out of the turn.
context_window = 128000
effective_input_limit_tokens = 64000
[providers.test.models.model.capabilities]
tools = true
[providers.test.models.model.capabilities.generation]
max_output_tokens = true
[providers.test.models.model.generation]
max_output_tokens = 4096
"#,
                    toml_path(&sessions_dir)
                ),
            )
            .unwrap();
            let config = AppConfig::load_from_path(&config_path).expect("config");

            let mut recorder = TranscriptRecorder::create(&sessions_dir).unwrap();
            recorder.record_session_started("test/model").unwrap();
            recorder.record_user_message("earlier prompt").unwrap();
            recorder.record_assistant_message("earlier answer").unwrap();
            let session_id = recorder.session_id().to_string();
            let transcript = Arc::new(StdMutex::new(recorder));

            // The subagent pool records the result before it sends the command.
            transcript
                .lock()
                .unwrap()
                .record_subagent_result_structured(
                    "run-1",
                    &session_id,
                    "run-1",
                    "child-1",
                    "explorer",
                    "completed",
                    "the child finished",
                    None,
                )
                .unwrap();

            let route = ModelRoute::new("test", "model");
            let primary_factory =
                Arc::new(ConfiguredPrimaryRouteFactory::new_with_runtime_catalog(
                    config.providers.clone(),
                    config.global.retry.clone(),
                    config.runtime_catalog.clone(),
                ));
            let mut agent = Agent::new("model", None, None);
            agent.apply_prepared_route(primary_factory.prepare_route(route).unwrap());
            agent.set_primary_route_factory(primary_factory);
            crate::configure_agent_runtime_snapshot_provider(&mut agent, &transcript);

            let settings = crate::session_engine_config(&config, Default::default(), String::new());
            let (mut engine, _) =
                SessionEngine::start(agent, transcript.clone(), "model".into(), settings).unwrap();
            let ingress = engine.take_ingress();
            let mut events = engine.take_event_egress().into_receiver();

            let completion = || SessionEngineCommand::BackgroundSubagentCompleted {
                parent_session_id: session_id.clone(),
                parent_tool_call_id: Some("call-1".into()),
                result: Ok(crate::subagent::SubagentRunSummary {
                    run_id: "run-1".into(),
                    child_session_id: "child-1".into(),
                    agent_name: "explorer".into(),
                    status: crate::subagent::SubagentStatus::Completed,
                    failure_kind: None,
                    summary: "the child finished".into(),
                    structured_result: crate::subagent::StructuredSubagentResult {
                        status: "completed".into(),
                        summary: "the child finished".into(),
                        malformed: false,
                        findings: Vec::new(),
                        files_read: Vec::new(),
                        files_changed: Vec::new(),
                        commands_run: Vec::new(),
                        validation: Vec::new(),
                        blockers: Vec::new(),
                        next_steps: Vec::new(),
                        run_id: "run-1".into(),
                        child_session_id: "child-1".into(),
                        raw_excerpt: None,
                    },
                }),
            };

            ingress.submit_transitional(completion()).unwrap();
            let delivery = tokio::time::timeout(Duration::from_secs(10), async {
                loop {
                    match events.recv().await {
                        Some(SessionTransportEvent::BackgroundSubagentCompleted { .. }) => {
                            return true;
                        }
                        Some(_) => {}
                        None => return false,
                    }
                }
            })
            .await
            .expect("the delivered event arrives");
            assert!(delivery, "a recorded result must still be delivered");

            let mut continued = false;
            for _ in 0..100 {
                let records = read_records(transcript.lock().unwrap().path()).unwrap();
                continued = records.iter().any(|record| {
                    matches!(
                        &record.event,
                        TranscriptEvent::InternalContinuation {
                            source:
                                crate::transcript::InternalContinuationSource::SubagentCompletion,
                            ..
                        }
                    )
                });
                if continued {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
            assert!(continued, "the completion must queue a continuation turn");

            ingress.submit_transitional(completion()).unwrap();
            let duplicate = tokio::time::timeout(Duration::from_millis(500), events.recv()).await;
            assert!(
                !matches!(
                    duplicate,
                    Ok(Some(
                        SessionTransportEvent::BackgroundSubagentCompleted { .. }
                    ))
                ),
                "the same run must not be delivered twice"
            );

            ingress.shutdown().unwrap();
            engine.join().await.unwrap();
        })
        .await
        .expect("background completion flow timed out");
    }

    #[tokio::test]
    async fn a_background_result_arriving_during_an_active_turn_is_delivered_and_continues() {
        tokio::time::timeout(Duration::from_secs(30), async {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
                .await
                .unwrap();
            let address = listener.local_addr().unwrap();
            let stub = tokio::spawn(serve_responses_stub(listener));

            let directory = tempfile::tempdir().unwrap();
            let config_path = directory.path().join("letcode.toml");
            let sessions_dir = directory.path().join("sessions");
            fs::write(
                &config_path,
                format!(
                    r#"
active_provider = "test"

[global]
sessions_dir = "{}"

[providers.test]
protocol = "responses"
default_model = "model"
[providers.test.auth]
type = "bearer"
credential = "test-key"
[providers.test.endpoints]
base_url = "http://{address}"
[providers.test.models.model]
# Retains the pre-recorded history, so historian work stays out of the turn.
context_window = 128000
effective_input_limit_tokens = 64000
[providers.test.models.model.capabilities]
tools = true
[providers.test.models.model.capabilities.generation]
max_output_tokens = true
[providers.test.models.model.generation]
max_output_tokens = 4096
"#,
                    toml_path(&sessions_dir)
                ),
            )
            .unwrap();
            let config = AppConfig::load_from_path(&config_path).expect("config");

            let mut recorder = TranscriptRecorder::create(&sessions_dir).unwrap();
            recorder.record_session_started("test/model").unwrap();
            // A recorded user message skips title generation, so the stub only serves turns.
            recorder.record_user_message("earlier prompt").unwrap();
            recorder.record_assistant_message("earlier answer").unwrap();
            let session_id = recorder.session_id().to_string();
            let transcript = Arc::new(StdMutex::new(recorder));

            let route = ModelRoute::new("test", "model");
            let primary_factory =
                Arc::new(ConfiguredPrimaryRouteFactory::new_with_runtime_catalog(
                    config.providers.clone(),
                    config.global.retry.clone(),
                    config.runtime_catalog.clone(),
                ));
            let mut agent = Agent::new("model", None, None);
            agent.apply_prepared_route(primary_factory.prepare_route(route).unwrap());
            agent.set_primary_route_factory(primary_factory);
            agent
                .try_register_tool(HoldingTool)
                .expect("register the holding tool");
            crate::configure_agent_runtime_snapshot_provider(&mut agent, &transcript);

            let settings = crate::session_engine_config(&config, Default::default(), String::new());
            let (mut engine, _) =
                SessionEngine::start(agent, transcript.clone(), "model".into(), settings).unwrap();
            let ingress = engine.take_ingress();
            let mut events = engine.take_event_egress().into_receiver();

            let mut errors = Vec::new();
            ingress
                .submit(SessionCommand::SubmitPrompt(
                    crate::user_content::UserMessageSubmission::new(
                        "prompt-1",
                        crate::user_content::UserMessageContent::new("first prompt", Vec::new()),
                    ),
                ))
                .unwrap();

            // The held tool call keeps the turn running for the rest of the test.
            let barrier = tokio::time::Instant::now() + Duration::from_secs(10);
            let mut barrier_error = None;
            loop {
                match open_tool_call_batch_recorded(&transcript) {
                    Ok(true) => break,
                    Ok(false) => {}
                    Err(error) => barrier_error = Some(error.to_string()),
                }
                if tokio::time::Instant::now() >= barrier {
                    panic!(
                        "the stub tool call never reached the transcript: {barrier_error:?}; errors: {errors:?}"
                    );
                }
                while let Ok(event) = events.try_recv() {
                    if let SessionTransportEvent::Error(error) = event {
                        errors.push(observed_error(&error));
                    }
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }

            let structured_result = crate::subagent::StructuredSubagentResult {
                status: "completed".into(),
                summary: "the child finished".into(),
                malformed: false,
                findings: Vec::new(),
                files_read: Vec::new(),
                files_changed: Vec::new(),
                commands_run: Vec::new(),
                validation: Vec::new(),
                blockers: Vec::new(),
                next_steps: Vec::new(),
                run_id: "run-1".into(),
                child_session_id: "child-1".into(),
                raw_excerpt: None,
            };

            // The subagent pool records the result before it sends the command.
            transcript
                .lock()
                .unwrap()
                .record_subagent_result_structured(
                    "run-1",
                    &session_id,
                    "run-1",
                    "child-1",
                    "explorer",
                    "completed",
                    "the child finished",
                    Some(structured_result.clone()),
                )
                .unwrap();

            ingress
                .submit_transitional(SessionEngineCommand::BackgroundSubagentCompleted {
                    parent_session_id: session_id.clone(),
                    parent_tool_call_id: Some("call-1".into()),
                    result: Ok(crate::subagent::SubagentRunSummary {
                        run_id: "run-1".into(),
                        child_session_id: "child-1".into(),
                        agent_name: "explorer".into(),
                        status: crate::subagent::SubagentStatus::Completed,
                        failure_kind: None,
                        summary: "the child finished".into(),
                        structured_result,
                    }),
                })
                .unwrap();

            let delivery = tokio::time::timeout(Duration::from_secs(10), async {
                loop {
                    match events.recv().await {
                        Some(SessionTransportEvent::BackgroundSubagentCompleted { .. }) => {
                            return true;
                        }
                        Some(_) => {}
                        None => return false,
                    }
                }
            })
            .await
            .expect("the delivered event arrives while the turn is active");
            assert!(
                delivery,
                "a result recorded while the turn is active must still be delivered"
            );

            let mut continued = false;
            for _ in 0..100 {
                let records = read_records(transcript.lock().unwrap().path()).unwrap();
                continued = records.iter().any(|record| {
                    matches!(
                        &record.event,
                        TranscriptEvent::InternalContinuation {
                            source: crate::transcript::InternalContinuationSource::SubagentCompletion,
                            ..
                        }
                    )
                });
                if continued {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
            assert!(
                continued,
                "the completion must queue a continuation turn while the turn is active"
            );

            ingress.shutdown().unwrap();
            engine.join().await.unwrap();
            stub.abort();
        })
        .await
        .expect("background completion during an active turn timed out");
    }

    #[tokio::test]
    async fn a_prompt_queued_behind_a_preempted_turn_keeps_the_session_usable() {
        tokio::time::timeout(Duration::from_secs(30), async {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
                .await
                .unwrap();
            let address = listener.local_addr().unwrap();
            let stub = tokio::spawn(serve_responses_stub(listener));

            let directory = tempfile::tempdir().unwrap();
            let config_path = directory.path().join("letcode.toml");
            let sessions_dir = directory.path().join("sessions");
            fs::write(
                &config_path,
                format!(
                    r#"
active_provider = "test"

[global]
sessions_dir = "{}"

[providers.test]
protocol = "responses"
default_model = "model"
[providers.test.auth]
type = "bearer"
credential = "test-key"
[providers.test.endpoints]
base_url = "http://{address}"
[providers.test.models.model]
# Retains the pre-recorded history, so historian work stays out of the turn.
context_window = 128000
effective_input_limit_tokens = 64000
[providers.test.models.model.capabilities]
tools = true
[providers.test.models.model.capabilities.generation]
max_output_tokens = true
[providers.test.models.model.generation]
max_output_tokens = 4096
"#,
                    toml_path(&sessions_dir)
                ),
            )
            .unwrap();
            let config = AppConfig::load_from_path(&config_path).expect("config");

            let mut recorder = TranscriptRecorder::create(&sessions_dir).unwrap();
            recorder.record_session_started("test/model").unwrap();
            // A recorded user message skips title generation, so the stub only serves turns.
            recorder.record_user_message("earlier prompt").unwrap();
            recorder.record_assistant_message("earlier answer").unwrap();
            let transcript = Arc::new(StdMutex::new(recorder));

            let route = ModelRoute::new("test", "model");
            let primary_factory =
                Arc::new(ConfiguredPrimaryRouteFactory::new_with_runtime_catalog(
                    config.providers.clone(),
                    config.global.retry.clone(),
                    config.runtime_catalog.clone(),
                ));
            let mut agent = Agent::new("model", None, None);
            agent.apply_prepared_route(primary_factory.prepare_route(route).unwrap());
            agent.set_primary_route_factory(primary_factory);
            agent
                .try_register_tool(HoldingTool)
                .expect("register the holding tool");
            crate::configure_agent_runtime_snapshot_provider(&mut agent, &transcript);

            let settings = crate::session_engine_config(&config, Default::default(), String::new());
            let (mut engine, _) =
                SessionEngine::start(agent, transcript.clone(), "model".into(), settings).unwrap();
            let ingress = engine.take_ingress();
            let mut events = engine.take_event_egress().into_receiver();

            let mut errors = Vec::new();
            let mut answers = Vec::new();
            ingress
                .submit(SessionCommand::SubmitPrompt(
                    crate::user_content::UserMessageSubmission::new(
                        "prompt-1",
                        crate::user_content::UserMessageContent::new("first prompt", Vec::new()),
                    ),
                ))
                .unwrap();

            let barrier = tokio::time::Instant::now() + Duration::from_secs(10);
            let mut barrier_error = None;
            loop {
                match open_tool_call_batch_recorded(&transcript) {
                    Ok(true) => break,
                    Ok(false) => {}
                    Err(error) => barrier_error = Some(error.to_string()),
                }
                if tokio::time::Instant::now() >= barrier {
                    panic!(
                        "the stub tool call never reached the transcript: {barrier_error:?}; errors: {errors:?}"
                    );
                }
                while let Ok(event) = events.try_recv() {
                    if let SessionTransportEvent::Error(error) = event {
                        errors.push(observed_error(&error));
                    }
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }

            ingress
                .submit(SessionCommand::SubmitPrompt(
                    crate::user_content::UserMessageSubmission::new(
                        "prompt-2",
                        crate::user_content::UserMessageContent::new("second prompt", Vec::new()),
                    ),
                ))
                .unwrap();

            let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
            loop {
                let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
                if remaining.is_zero() {
                    break;
                }
                let Ok(Some(event)) = tokio::time::timeout(remaining, events.recv()).await else {
                    break;
                };
                match event {
                    SessionTransportEvent::Error(error) => {
                        let locked = error
                            .message
                            .contains("assistant tool call group is incomplete");
                        errors.push(observed_error(&error));
                        if locked {
                            break;
                        }
                    }
                    SessionTransportEvent::AssistantDelta(delta) => {
                        let answered = delta.delta.contains(STUB_ANSWER);
                        answers.push(delta.delta);
                        if answered {
                            break;
                        }
                    }
                    _ => {}
                }
            }

            assert!(
                !errors.iter().any(|message| message
                    .contains("assistant tool call group is incomplete")),
                "preempting a running turn must not leave the session locked; observed errors: {errors:?}"
            );
            assert!(
                answers.iter().any(|answer| answer.contains(STUB_ANSWER)),
                "the prompt queued behind the preempted turn must open its own turn; observed errors: {errors:?}"
            );

            ingress.shutdown().unwrap();
            engine.join().await.unwrap();
            stub.abort();
        })
        .await
        .expect("preemption of a running turn timed out");
    }
}
