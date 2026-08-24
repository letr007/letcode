use anyhow::{Context, Result, anyhow, bail};
use async_openai::config::Config;
use futures_util::FutureExt;
use serde_json::Value;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::oneshot;
use tokio::time::{Duration, timeout};

use crate::agent::{Agent, AgentFactory, AgentTemplate, SubagentInvocation};
use crate::config::ModelRoute;
use crate::subagent_events::{SubagentEventSender, emit_error, emit_status, run_child_prompt};
use crate::tool::{NormalizedSubagentInput, SubagentPathScope};
use crate::transcript::transcript_projection;
use crate::transcript::{
    ChildSessionSummary, TranscriptEvent, TranscriptRecorder, child_sessions_dir,
    list_child_sessions_for_parent, read_records_allow_partial_tail, sort_child_session_summaries,
};

use super::result::{
    SubagentFailureKind, SubagentRunSummary, SubagentStatus, build_completed_summary,
    build_runtime_summary, classify_failure_status,
};

type BoxExecFuture = Pin<Box<dyn Future<Output = Result<String>> + Send>>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubagentRunGovernance {
    pub timeout_secs: Option<u64>,
    pub max_tool_calls: Option<usize>,
    pub model: Option<ModelRoute>,
    pub input: NormalizedSubagentInput,
}

impl SubagentRunGovernance {
    fn from_template_and_input(
        template: &AgentTemplate,
        input: NormalizedSubagentInput,
        model: Option<ModelRoute>,
    ) -> Self {
        Self {
            timeout_secs: input.effective_timeout_secs(template.timeout_secs),
            max_tool_calls: input.effective_max_tool_calls(template.max_tool_calls),
            model,
            input,
        }
    }
}

#[derive(Clone)]
pub struct SubagentPool {
    /// Active and completed runs keyed by stable run id.
    state: Arc<Mutex<SubagentPoolState>>,
    /// Wakes waiters whenever the active or completed run set changes.
    changed: Arc<tokio::sync::Notify>,
    /// Monotonic ordinal issuer for stable TUI pool numbers (1-based).
    next_ordinal: Arc<Mutex<u32>>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct SubagentJob {
    pub active: bool,
    pub run_id: String,
    pub child_session_id: String,
    pub agent_name: String,
    pub status: String,
    pub summary: String,
    pub pool_ordinal: u32,
}

pub struct ForegroundRunGuard {
    state: Arc<Mutex<SubagentPoolState>>,
    run_id: String,
    retained: bool,
}

impl ForegroundRunGuard {
    fn new(state: Arc<Mutex<SubagentPoolState>>, run_id: String) -> Self {
        Self {
            state,
            run_id,
            retained: false,
        }
    }

    pub fn retain(&mut self) {
        self.retained = true;
    }
}

impl Drop for ForegroundRunGuard {
    fn drop(&mut self) {
        if self.retained {
            return;
        }
        if let Ok(mut state) = self.state.lock() {
            state.foregrounded_by_run.remove(&self.run_id);
        }
    }
}

impl SubagentJob {
    pub fn from_result(result: SubagentRunSummary) -> Self {
        Self {
            active: false,
            run_id: result.run_id,
            child_session_id: result.child_session_id,
            agent_name: result.agent_name,
            status: result.status.as_str().into(),
            summary: result.summary,
            pool_ordinal: 0,
        }
    }
}

#[derive(Default)]
struct SubagentPoolState {
    active_by_run: std::collections::HashMap<String, ActiveSlot>,
    completed_by_run: std::collections::HashMap<String, SubagentRunSummary>,
    foregrounded_by_run: std::collections::HashSet<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum RunPathAccess {
    Read(Vec<PathBuf>),
    Write(Vec<PathBuf>),
}

#[derive(Debug)]
struct ActiveSlot {
    cancel: Option<oneshot::Sender<()>>,
    child: ChildSessionSummary,
    path_access: RunPathAccess,
    takeover_child_session_id: Option<String>,
}

struct RunReservation {
    state: Arc<Mutex<SubagentPoolState>>,
    changed: Arc<tokio::sync::Notify>,
    run_id: String,
    activated: bool,
}

impl RunReservation {
    fn activate(mut self, cancel: oneshot::Sender<()>, child: ChildSessionSummary) -> Result<()> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow!("subagent pool lock poisoned"))?;
        let slot = state
            .active_by_run
            .get_mut(&self.run_id)
            .ok_or_else(|| anyhow!("subagent run reservation disappeared"))?;
        slot.cancel = Some(cancel);
        slot.child = child;
        self.activated = true;
        drop(state);
        self.changed.notify_waiters();
        Ok(())
    }
}

impl Drop for RunReservation {
    fn drop(&mut self) {
        if self.activated {
            return;
        }
        if let Ok(mut state) = self.state.lock() {
            state.active_by_run.remove(&self.run_id);
        }
        self.changed.notify_waiters();
    }
}

struct DropTerminalContext {
    child_transcript: Arc<Mutex<TranscriptRecorder>>,
    parent_transcript: Option<Arc<Mutex<TranscriptRecorder>>>,
    parent_session_id: String,
    parent_turn_id: String,
    child_session_id: String,
    agent_name: String,
}

struct ActiveRunGuard {
    state: Arc<Mutex<SubagentPoolState>>,
    changed: Arc<tokio::sync::Notify>,
    run_id: String,
    terminal_context: DropTerminalContext,
    terminal: bool,
}

impl ActiveRunGuard {
    fn new(
        state: Arc<Mutex<SubagentPoolState>>,
        changed: Arc<tokio::sync::Notify>,
        run_id: String,
        terminal_context: DropTerminalContext,
    ) -> Self {
        Self {
            state,
            changed,
            run_id,
            terminal_context,
            terminal: false,
        }
    }

    fn complete(&mut self, summary: SubagentRunSummary) {
        if let Ok(mut state) = self.state.lock() {
            state.completed_by_run.insert(self.run_id.clone(), summary);
            state.active_by_run.remove(&self.run_id);
        }
        self.terminal = true;
        self.changed.notify_waiters();
    }

    fn clear_cancel(&self) {
        if let Ok(mut state) = self.state.lock()
            && let Some(slot) = state.active_by_run.get_mut(&self.run_id)
        {
            slot.cancel.take();
        }
    }
}

impl Drop for ActiveRunGuard {
    fn drop(&mut self) {
        if self.terminal {
            return;
        }
        let context = &self.terminal_context;
        let mut summary = build_runtime_summary(
            &self.run_id,
            &context.child_session_id,
            &context.agent_name,
            SubagentStatus::Cancelled,
            format!("{} cancelled", context.agent_name),
        );
        if let Err(error) = record_child_completion(
            &context.child_transcript,
            &summary,
            &context.parent_session_id,
            &context.parent_turn_id,
        ) {
            set_hard_failure(
                &mut summary,
                format!("failed to record dropped child subagent completion: {error}"),
            );
        }
        if let Err(error) = record_parent_result(
            &context.parent_transcript,
            &summary,
            &context.parent_session_id,
            &context.parent_turn_id,
        ) {
            set_hard_failure(
                &mut summary,
                format!("failed to record dropped parent subagent result: {error}"),
            );
            let _ = record_child_completion(
                &context.child_transcript,
                &summary,
                &context.parent_session_id,
                &context.parent_turn_id,
            );
        }
        if let Ok(mut state) = self.state.lock() {
            state.completed_by_run.insert(self.run_id.clone(), summary);
            state.active_by_run.remove(&self.run_id);
        }
        self.changed.notify_waiters();
    }
}

impl Default for SubagentPool {
    fn default() -> Self {
        Self::new()
    }
}

impl SubagentPool {
    pub fn new() -> Self {
        Self {
            state: Arc::new(Mutex::new(SubagentPoolState::default())),
            changed: Arc::new(tokio::sync::Notify::new()),
            next_ordinal: Arc::new(Mutex::new(1)),
        }
    }

    pub fn cancel_active(&self) -> bool {
        let Ok(mut state) = self.state.lock() else {
            return false;
        };
        let mut cancelled = false;
        for slot in state.active_by_run.values_mut() {
            if let Some(tx) = slot.cancel.take() {
                let _ = tx.send(());
                cancelled = true;
            }
        }
        cancelled
    }

    pub fn cancel_run(&self, run_id: &str) -> bool {
        let Ok(mut state) = self.state.lock() else {
            return false;
        };
        let Some(slot) = state.active_by_run.get_mut(run_id) else {
            return false;
        };
        let Some(tx) = slot.cancel.take() else {
            return false;
        };
        tx.send(()).is_ok()
    }

    pub fn is_running(&self) -> bool {
        self.state
            .lock()
            .map(|state| !state.active_by_run.is_empty())
            .unwrap_or(false)
    }

    pub fn active_child(&self) -> Option<ChildSessionSummary> {
        let state = self.state.lock().ok()?;
        state
            .active_by_run
            .values()
            .map(|slot| slot.child.clone())
            .min_by_key(|child| (child.pool_ordinal, child.child_session_id.clone()))
    }

    pub fn active_jobs(&self) -> Vec<SubagentJob> {
        let Ok(state) = self.state.lock() else {
            return Vec::new();
        };
        let mut jobs = state
            .active_by_run
            .iter()
            .map(|(run_id, slot)| SubagentJob {
                active: true,
                run_id: run_id.clone(),
                child_session_id: slot.child.child_session_id.clone(),
                agent_name: slot.child.agent_name.clone(),
                status: slot.child.status.clone(),
                summary: slot.child.summary.clone(),
                pool_ordinal: slot.child.pool_ordinal,
            })
            .collect::<Vec<_>>();
        jobs.sort_by_key(|job| (job.pool_ordinal, job.run_id.clone()));
        jobs
    }

    pub fn claim_foreground(&self, run_id: &str) -> Result<Option<ForegroundRunGuard>> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow!("subagent pool lock poisoned"))?;
        if state.completed_by_run.contains_key(run_id)
            || !state.active_by_run.contains_key(run_id)
            || !state.foregrounded_by_run.insert(run_id.to_string())
        {
            return Ok(None);
        }
        Ok(Some(ForegroundRunGuard::new(
            Arc::clone(&self.state),
            run_id.to_string(),
        )))
    }

    pub fn release_foreground(&self, run_id: &str) {
        if let Ok(mut state) = self.state.lock() {
            state.foregrounded_by_run.remove(run_id);
        }
    }

    pub fn is_foregrounded(&self, run_id: &str) -> bool {
        self.state
            .lock()
            .map(|state| state.foregrounded_by_run.contains(run_id))
            .unwrap_or(false)
    }

    pub fn completed_result(&self, run_id: &str) -> Option<SubagentRunSummary> {
        self.state
            .lock()
            .ok()
            .and_then(|state| state.completed_by_run.get(run_id).cloned())
    }

    pub async fn wait_for_result(&self, run_id: &str) -> Result<Option<SubagentRunSummary>> {
        loop {
            if let Some(result) = self.completed_result(run_id) {
                return Ok(Some(result));
            }
            let running = self
                .state
                .lock()
                .map_err(|_| anyhow!("subagent pool lock poisoned"))?
                .active_by_run
                .contains_key(run_id);
            if !running {
                return Ok(None);
            }
            let notified = self.changed.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if let Some(result) = self.completed_result(run_id) {
                return Ok(Some(result));
            }
            let running = self
                .state
                .lock()
                .map_err(|_| anyhow!("subagent pool lock poisoned"))?
                .active_by_run
                .contains_key(run_id);
            if !running {
                return Ok(None);
            }
            notified.await;
        }
    }

    fn reserve_run(
        &self,
        path_access: RunPathAccess,
        takeover_child_session_id: Option<&str>,
    ) -> Result<RunReservation> {
        let run_id = generate_run_id();
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow!("subagent pool lock poisoned"))?;
        ensure_path_access_available_in_map(&state.active_by_run, &path_access)?;
        if let Some(target) = takeover_child_session_id
            && state
                .active_by_run
                .values()
                .any(|slot| slot.takeover_child_session_id.as_deref() == Some(target))
        {
            bail!("takeover failed: child `{target}` already has an active takeover");
        }
        state.active_by_run.insert(
            run_id.clone(),
            ActiveSlot {
                cancel: None,
                child: ChildSessionSummary {
                    parent_session_id: String::new(),
                    parent_run_id: String::new(),
                    child_session_id: String::new(),
                    agent_name: String::new(),
                    status: "starting".into(),
                    summary: String::new(),
                    timestamp_ms: current_timestamp_ms(),
                    pool_ordinal: 0,
                },
                path_access,
                takeover_child_session_id: takeover_child_session_id.map(str::to_string),
            },
        );
        drop(state);
        self.changed.notify_waiters();
        Ok(RunReservation {
            state: Arc::clone(&self.state),
            changed: Arc::clone(&self.changed),
            run_id,
            activated: false,
        })
    }

    pub fn child_sessions(
        sessions_dir: impl AsRef<Path>,
        parent_records: &[crate::transcript::TranscriptRecord],
    ) -> Vec<ChildSessionSummary> {
        let mut children = list_child_sessions_for_parent(sessions_dir, parent_records);
        // Synthesize stable ordinals for legacy rows (pool_ordinal == 0).
        let mut next = children
            .iter()
            .map(|child| child.pool_ordinal)
            .filter(|ordinal| *ordinal > 0)
            .max()
            .unwrap_or(0);
        // Assign in current sort order for legacy stability within a process.
        let mut legacy: Vec<_> = children
            .iter()
            .enumerate()
            .filter(|(_, child)| child.pool_ordinal == 0)
            .map(|(idx, _)| idx)
            .collect();
        // Sort legacy indices by timestamp then id to assign deterministic ordinals.
        legacy.sort_by(|&left, &right| {
            children[left]
                .timestamp_ms
                .cmp(&children[right].timestamp_ms)
                .then_with(|| {
                    children[left]
                        .child_session_id
                        .cmp(&children[right].child_session_id)
                })
        });
        for idx in legacy {
            next += 1;
            children[idx].pool_ordinal = next;
        }
        sort_child_session_summaries(&mut children);
        children
    }

    fn allocate_ordinal_from_children(&self, existing: &[ChildSessionSummary]) -> u32 {
        let max_existing = existing
            .iter()
            .map(|child| child.pool_ordinal)
            .max()
            .unwrap_or(0);
        let max_active = self
            .state
            .lock()
            .ok()
            .and_then(|state| {
                state
                    .active_by_run
                    .values()
                    .map(|slot| slot.child.pool_ordinal)
                    .max()
            })
            .unwrap_or(0);
        let mut next = self
            .next_ordinal
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let max_known = max_existing.max(max_active);
        if *next <= max_known {
            *next = max_known.saturating_add(1);
        }
        let ordinal = *next;
        *next = next.saturating_add(1);
        ordinal
    }

    pub async fn run_named_governed<C: Config + Clone + Send + Sync + 'static>(
        &self,
        parent: &Agent<C>,
        agent_name: &str,
        invocation: SubagentInvocation,
        sessions_dir: impl AsRef<Path>,
        parent_session_id: String,
        parent_turn_id: String,
        parent_transcript: Option<Arc<Mutex<TranscriptRecorder>>>,
        event_sender: Option<SubagentEventSender<C>>,
    ) -> Result<SubagentRunSummary> {
        self.complete_started_run(self.start_named_governed(
            parent,
            agent_name,
            invocation,
            sessions_dir,
            parent_session_id,
            parent_turn_id,
            parent_transcript,
            event_sender,
        )?)
        .await
    }

    pub fn start_named_governed<C: Config + Clone + Send + Sync + 'static>(
        &self,
        parent: &Agent<C>,
        agent_name: &str,
        invocation: SubagentInvocation,
        sessions_dir: impl AsRef<Path>,
        parent_session_id: String,
        parent_turn_id: String,
        parent_transcript: Option<Arc<Mutex<TranscriptRecorder>>>,
        event_sender: Option<SubagentEventSender<C>>,
    ) -> Result<StartedSubagentRun<C>> {
        let template = AgentTemplate::from_name(agent_name)
            .ok_or_else(|| anyhow!("unknown subagent template: {agent_name}"))?;
        let governance = SubagentRunGovernance::from_template_and_input(
            &template,
            invocation.input.clone(),
            invocation.model.clone(),
        );
        let task = invocation.prompt;
        let run = self.start_run(
            parent,
            template,
            task.clone(),
            governance,
            sessions_dir.as_ref().to_path_buf(),
            parent_session_id,
            parent_turn_id,
            parent_transcript,
            event_sender
                .map(|sender| sender.with_parent_tool_call_id(invocation.parent_tool_call_id)),
            invocation.input.target_child_session_id.clone(),
        )?;
        let receipt = run
            .guard
            .state
            .lock()
            .ok()
            .and_then(|state| {
                state
                    .active_by_run
                    .get(&run.run_id)
                    .map(|slot| slot.child.clone())
            })
            .ok_or_else(|| anyhow!("started subagent slot is unavailable"))?;
        Ok(StartedSubagentRun {
            run_id: run.run_id.clone(),
            receipt,
            run,
            task,
        })
    }

    pub async fn complete_started_run<C: Config + Clone + Send + Sync + 'static>(
        &self,
        started: StartedSubagentRun<C>,
    ) -> Result<SubagentRunSummary> {
        complete_started_run(
            started.run,
            started.task,
            |agent, prompt, transcript, event_sender, child_session_id, agent_name| {
                async move {
                    run_child_prompt(
                        agent,
                        prompt,
                        transcript,
                        event_sender,
                        child_session_id,
                        Some(agent_name),
                    )
                    .await
                }
                .boxed()
            },
        )
        .await
    }

    pub async fn run_with_executor<C, F>(
        &self,
        parent: &Agent<C>,
        template: AgentTemplate,
        task: String,
        governance: SubagentRunGovernance,
        sessions_dir: PathBuf,
        parent_session_id: String,
        parent_turn_id: String,
        parent_transcript: Option<Arc<Mutex<TranscriptRecorder>>>,
        event_sender: Option<SubagentEventSender<C>>,
        takeover_child_session_id: Option<String>,
        exec: F,
    ) -> Result<SubagentRunSummary>
    where
        C: Config + Clone + Send + Sync + 'static,
        F: FnOnce(
                Agent<C>,
                String,
                Arc<Mutex<TranscriptRecorder>>,
                Option<SubagentEventSender<C>>,
                String,
                String,
            ) -> BoxExecFuture
            + Send
            + 'static,
    {
        let running = self.start_run(
            parent,
            template,
            task.clone(),
            governance,
            sessions_dir,
            parent_session_id,
            parent_turn_id,
            parent_transcript,
            event_sender,
            takeover_child_session_id,
        )?;
        complete_started_run(running, task, exec).await
    }

    fn start_run<C>(
        &self,
        parent: &Agent<C>,
        template: AgentTemplate,
        task: String,
        governance: SubagentRunGovernance,
        sessions_dir: PathBuf,
        parent_session_id: String,
        parent_turn_id: String,
        parent_transcript: Option<Arc<Mutex<TranscriptRecorder>>>,
        event_sender: Option<SubagentEventSender<C>>,
        takeover_child_session_id: Option<String>,
    ) -> Result<StartedRun<C>>
    where
        C: Config + Clone + Send + Sync + 'static,
    {
        let scope = SubagentPathScope::from_input(&governance.input)?;
        let path_access = run_path_access(&template, &governance.input, scope.as_ref())?;
        let mut reservation =
            self.reserve_run(path_access.clone(), takeover_child_session_id.as_deref())?;

        let effective_timeout_secs = governance.timeout_secs.or(template.timeout_secs);
        let effective_max_tool_calls = governance.max_tool_calls.or(template.max_tool_calls);
        if takeover_child_session_id.is_some() && governance.model.is_some() {
            bail!("model override cannot be used when taking over a child session");
        }

        let existing_children = {
            let parent_records = parent_transcript
                .as_ref()
                .and_then(|recorder| recorder.lock().ok())
                .and_then(|recorder| read_records_allow_partial_tail(recorder.path()).ok())
                .unwrap_or_default();
            Self::child_sessions(&sessions_dir, &parent_records)
        };

        let setup = (|| -> Result<(
            String,
            String,
            u32,
            Arc<Mutex<TranscriptRecorder>>,
            Agent<C>,
        )> {
            let run_id = reservation.run_id.clone();
            let child_dir = child_sessions_dir(&sessions_dir);

            if let Some(target_id) = takeover_child_session_id.as_ref() {
                let target = existing_children
                    .iter()
                    .find(|child| child.child_session_id == *target_id)
                    .ok_or_else(|| {
                        anyhow!(
                            "takeover failed: child_session_id `{target_id}` is not a known child of this parent"
                        )
                    })?;
                if target.agent_name != template.name {
                    return Err(anyhow!(
                        "takeover failed: child `{}` is agent `{}`, expected `{}`",
                        target_id,
                        target.agent_name,
                        template.name
                    ));
                }
                let status = target.status.as_str();
                let terminal = matches!(
                    status,
                    "completed"
                        | "failed"
                        | "budget_exhausted"
                        | "cancelled"
                        | "timed_out"
                        | "error"
                        | "errored"
                );
                if !terminal {
                    return Err(anyhow!(
                        "takeover failed: child `{}` status is `{status}` (only terminal sessions can be taken over)",
                        target_id
                    ));
                }

                let pool_ordinal = if target.pool_ordinal > 0 {
                    target.pool_ordinal
                } else {
                    self.allocate_ordinal_from_children(&existing_children)
                };

                let mut child_recorder = TranscriptRecorder::open(&child_dir, target_id.clone())?;
                let child_records = read_records_allow_partial_tail(child_recorder.path())?;
                let recorded_route = crate::transcript::restore_latest_model(&child_records)
                    .ok_or_else(|| {
                        anyhow!("takeover failed: child `{target_id}` has no recorded model route")
                    })?;
                let recorded_route = ModelRoute::parse(&recorded_route).with_context(|| {
                    format!(
                        "takeover failed: child `{target_id}` recorded an invalid model route"
                    )
                })?;
                let mut child_agent = AgentFactory::create_child_with_route_and_max_tool_calls(
                    parent,
                    &template,
                    Some(recorded_route),
                    true,
                    effective_max_tool_calls,
                )?;
                child_agent.set_subagent_path_scope(scope.clone().map(Arc::new));
                child_agent.set_context_scope_state(child_recorder.context_scope_state());
                let snapshot = transcript_projection::project_runtime_restore_snapshot(
                    target_id.clone(),
                    child_records,
                    transcript_projection::SessionContextCursor {
                        branch_id: None,
                        leaf_sequence: None,
                    },
                    &[],
                )?;
                child_recorder.adopt_legacy_linear_branch(&snapshot.branch_id)?;
                let runtime_snapshot = child_agent
                    .validate_runtime_snapshot_restore(snapshot.snapshot)?;

                child_recorder.record_subagent_lifecycle(
                    run_id.clone(),
                    parent_session_id.clone(),
                    parent_turn_id.clone(),
                    template.name.clone(),
                    SubagentStatus::Running.as_str(),
                    Some(format!("takeover: {task}")),
                )?;
                // Re-record started with ordinal so projections stay stable across takeover.
                child_recorder.record_subagent_started(
                    run_id.clone(),
                    parent_session_id.clone(),
                    parent_turn_id.clone(),
                    target_id.clone(),
                    template.name.clone(),
                    task.clone(),
                    pool_ordinal,
                )?;

                child_agent.install_validated_runtime_snapshot(runtime_snapshot);
                child_agent.restore_turn_sequence(snapshot.max_turn_id);

                Ok((
                    run_id,
                    target_id.clone(),
                    pool_ordinal,
                    Arc::new(Mutex::new(child_recorder)),
                    child_agent,
                ))
            } else {
                let pool_ordinal = self.allocate_ordinal_from_children(&existing_children);
                let mut child_recorder = TranscriptRecorder::create(&child_dir)?;
                let mut child_agent = AgentFactory::create_child_with_route_and_max_tool_calls(
                    parent,
                    &template,
                    governance.model.clone(),
                    false,
                    effective_max_tool_calls,
                )?;
                child_agent.set_subagent_path_scope(scope.clone().map(Arc::new));
                child_agent.set_context_scope_state(child_recorder.context_scope_state());
                child_recorder.record_session_started(child_agent.route_display_name())?;
                let child_session_id = child_recorder.session_id().to_string();
                child_recorder.record_subagent_lifecycle(
                    run_id.clone(),
                    parent_session_id.clone(),
                    parent_turn_id.clone(),
                    template.name.clone(),
                    SubagentStatus::Running.as_str(),
                    Some(task.clone()),
                )?;
                child_recorder.record_subagent_started(
                    run_id.clone(),
                    parent_session_id.clone(),
                    parent_turn_id.clone(),
                    child_session_id.clone(),
                    template.name.clone(),
                    task.clone(),
                    pool_ordinal,
                )?;
                Ok((
                    run_id,
                    child_session_id,
                    pool_ordinal,
                    Arc::new(Mutex::new(child_recorder)),
                    child_agent,
                ))
            }
        })();

        let (run_id, child_session_id, pool_ordinal, child_transcript, child_agent) = setup?;

        if let Err(error) = record_parent_started(
            &parent_transcript,
            &run_id,
            &parent_session_id,
            &parent_turn_id,
            &child_session_id,
            &template.name,
            &task,
            pool_ordinal,
        ) {
            let mut summary = build_runtime_summary(
                &run_id,
                &child_session_id,
                &template.name,
                SubagentStatus::Failed,
                String::new(),
            );
            set_hard_failure(
                &mut summary,
                format!("failed to record parent subagent start: {error}"),
            );
            let _ = record_child_completion(
                &child_transcript,
                &summary,
                &parent_session_id,
                &parent_turn_id,
            );
            if let Some(parent_path) = parent_transcript
                .as_ref()
                .and_then(|recorder| recorder.lock().ok())
                .map(|recorder| recorder.path().to_path_buf())
                && crate::transcript::repair_partial_tail(&parent_path).is_ok()
                && let Some(parent_dir) = parent_path.parent()
                && let Ok(mut recorder) = TranscriptRecorder::open(parent_dir, &parent_session_id)
            {
                let _ = recorder.record_subagent_started(
                    run_id.clone(),
                    parent_session_id.clone(),
                    parent_turn_id.clone(),
                    child_session_id.clone(),
                    template.name.clone(),
                    task.clone(),
                    pool_ordinal,
                );
                let _ = recorder.record_subagent_result_structured(
                    run_id.clone(),
                    parent_session_id.clone(),
                    parent_turn_id.clone(),
                    child_session_id.clone(),
                    template.name.clone(),
                    summary.status.as_str().to_string(),
                    summary.summary.clone(),
                    Some(summary.structured_result.clone()),
                );
            }
            if let Ok(mut state) = self.state.lock() {
                state.completed_by_run.insert(run_id.clone(), summary);
                state.active_by_run.remove(&run_id);
            }
            self.changed.notify_waiters();
            reservation.activated = true;
            return Err(error);
        }

        let (cancel_tx, cancel_rx) = oneshot::channel();
        let summary = ChildSessionSummary {
            parent_session_id: parent_session_id.clone(),
            parent_run_id: parent_turn_id.clone(),
            child_session_id: child_session_id.clone(),
            agent_name: template.name.clone(),
            status: SubagentStatus::Running.as_str().into(),
            summary: task.clone(),
            timestamp_ms: current_timestamp_ms(),
            pool_ordinal,
        };

        reservation.activate(cancel_tx, summary)?;

        Ok(StartedRun {
            guard: ActiveRunGuard::new(
                Arc::clone(&self.state),
                Arc::clone(&self.changed),
                run_id.clone(),
                DropTerminalContext {
                    child_transcript: Arc::clone(&child_transcript),
                    parent_transcript: parent_transcript.clone(),
                    parent_session_id: parent_session_id.clone(),
                    parent_turn_id: parent_turn_id.clone(),
                    child_session_id: child_session_id.clone(),
                    agent_name: template.name.clone(),
                },
            ),
            path_access,
            run_id,
            child_session_id,
            agent_name: template.name.clone(),
            timeout_secs: effective_timeout_secs,
            governance,
            parent_session_id,
            parent_turn_id,
            parent_transcript,
            event_sender,
            child_transcript,
            child_agent,
            cancel_rx,
        })
    }
}

fn run_path_access(
    template: &AgentTemplate,
    input: &NormalizedSubagentInput,
    scope: Option<&SubagentPathScope>,
) -> Result<RunPathAccess> {
    if template.can_write {
        if input.owned_paths.is_empty() {
            bail!(
                "writable subagent `{}` requires non-empty owned_paths for concurrent file locking",
                template.name
            );
        }
        let roots = scope
            .map(|scope| scope.owned_roots().to_vec())
            .unwrap_or_default();
        return Ok(RunPathAccess::Write(roots));
    }

    let roots = if input.allowed_paths.is_empty() && input.owned_paths.is_empty() {
        vec![crate::tool::workspace_root_for_subagent_lock()?]
    } else {
        let mut paths = input.allowed_paths.clone();
        paths.extend(input.owned_paths.clone());
        canonical_lock_roots(&paths)?
    };
    Ok(RunPathAccess::Read(roots))
}

fn validate_write_lock_coverage(
    requested: &RunPathAccess,
    observed_changed_paths: &[String],
) -> Vec<String> {
    let RunPathAccess::Write(roots) = requested else {
        return Vec::new();
    };
    observed_changed_paths
        .iter()
        .filter(|path| {
            crate::tool::canonical_subagent_observed_path(path)
                .map(|path| {
                    !roots
                        .iter()
                        .any(|root| path == *root || path.starts_with(root))
                })
                .unwrap_or(true)
        })
        .cloned()
        .collect()
}

fn canonical_lock_roots(paths: &[String]) -> Result<Vec<PathBuf>> {
    let mut roots = Vec::new();
    for path in paths {
        let root = crate::tool::canonical_subagent_lock_root(path)?;
        if !roots.contains(&root) {
            roots.push(root);
        }
    }
    Ok(roots)
}

fn ensure_path_access_available_in_map(
    map: &std::collections::HashMap<String, ActiveSlot>,
    requested: &RunPathAccess,
) -> Result<()> {
    for (run_id, slot) in map {
        if path_access_conflicts(requested, &slot.path_access) {
            bail!(
                "subagent path lock conflict with run_id={} child_session_id={} agent_name={}",
                run_id,
                slot.child.child_session_id,
                slot.child.agent_name
            );
        }
    }
    Ok(())
}

fn path_access_conflicts(left: &RunPathAccess, right: &RunPathAccess) -> bool {
    match (left, right) {
        (RunPathAccess::Read(_), RunPathAccess::Read(_)) => false,
        (RunPathAccess::Read(reads), RunPathAccess::Write(writes))
        | (RunPathAccess::Write(writes), RunPathAccess::Read(reads)) => {
            path_sets_overlap(reads, writes)
        }
        (RunPathAccess::Write(left), RunPathAccess::Write(right)) => path_sets_overlap(left, right),
    }
}

fn path_sets_overlap(left: &[PathBuf], right: &[PathBuf]) -> bool {
    left.iter().any(|left| {
        right
            .iter()
            .any(|right| left == right || left.starts_with(right) || right.starts_with(left))
    })
}

pub struct StartedSubagentRun<C: Config> {
    run_id: String,
    receipt: ChildSessionSummary,
    run: StartedRun<C>,
    task: String,
}

impl<C: Config> StartedSubagentRun<C> {
    pub fn run_id(&self) -> &str {
        &self.run_id
    }

    pub fn receipt(&self) -> &ChildSessionSummary {
        &self.receipt
    }
}

struct StartedRun<C: Config> {
    guard: ActiveRunGuard,
    path_access: RunPathAccess,
    run_id: String,
    child_session_id: String,
    agent_name: String,
    timeout_secs: Option<u64>,
    governance: SubagentRunGovernance,
    parent_session_id: String,
    parent_turn_id: String,
    parent_transcript: Option<Arc<Mutex<TranscriptRecorder>>>,
    event_sender: Option<SubagentEventSender<C>>,
    child_transcript: Arc<Mutex<TranscriptRecorder>>,
    child_agent: Agent<C>,
    cancel_rx: oneshot::Receiver<()>,
}

async fn complete_started_run<C, F>(
    started: StartedRun<C>,
    task: String,
    exec: F,
) -> Result<SubagentRunSummary>
where
    C: Config + Clone + Send + Sync + 'static,
    F: FnOnce(
            Agent<C>,
            String,
            Arc<Mutex<TranscriptRecorder>>,
            Option<SubagentEventSender<C>>,
            String,
            String,
        ) -> BoxExecFuture
        + Send
        + 'static,
{
    let StartedRun {
        mut guard,
        path_access,
        run_id,
        child_session_id,
        agent_name,
        timeout_secs,
        governance,
        parent_session_id,
        parent_turn_id,
        parent_transcript,
        event_sender,
        child_transcript,
        child_agent,
        cancel_rx,
    } = started;

    let execution = exec(
        child_agent,
        task,
        Arc::clone(&child_transcript),
        event_sender.clone(),
        child_session_id.clone(),
        agent_name.clone(),
    );
    let summary = tokio::select! {
        result = async {
            match timeout_secs {
                Some(timeout_secs) => match timeout(Duration::from_secs(timeout_secs), execution).await {
                    Ok(Ok(message)) => build_completed_summary(
                        &run_id,
                        &child_session_id,
                        &agent_name,
                        message,
                    ),
                    Ok(Err(error)) => build_runtime_summary(
                        &run_id,
                        &child_session_id,
                        &agent_name,
                        classify_failure_status(&error.to_string()),
                        error.to_string(),
                    ),
                    Err(_) => build_runtime_summary(
                        &run_id,
                        &child_session_id,
                        &agent_name,
                        SubagentStatus::TimedOut,
                        format!("{} timed out after {timeout_secs}s", agent_name),
                    ),
                },
                None => match execution.await {
                    Ok(message) => build_completed_summary(
                        &run_id,
                        &child_session_id,
                        &agent_name,
                        message,
                    ),
                    Err(error) => build_runtime_summary(
                        &run_id,
                        &child_session_id,
                        &agent_name,
                        classify_failure_status(&error.to_string()),
                        error.to_string(),
                    ),
                },
            }
        } => result,
        _ = cancel_rx => build_runtime_summary(
            &run_id,
            &child_session_id,
            &agent_name,
            SubagentStatus::Cancelled,
            format!("{} cancelled", agent_name),
        )
    };

    let mut summary = summary;
    let observed_changed_paths = observed_changed_paths_from_child_transcript(&child_transcript);
    enforce_write_scope(&mut summary, &governance, &observed_changed_paths);
    let unlocked_changes = validate_write_lock_coverage(&path_access, &observed_changed_paths);
    if !unlocked_changes.is_empty() {
        let message = format!(
            "changes outside acquired file locks detected: {}",
            unlocked_changes.join(", ")
        );
        summary.status = SubagentStatus::Failed;
        summary.failure_kind = Some(SubagentFailureKind::Logical);
        summary.summary = message.clone();
        summary.structured_result.status = SubagentStatus::Failed.as_str().into();
        summary.structured_result.summary = message.clone();
        summary.structured_result.blockers.push(message);
    }

    guard.clear_cancel();

    if let Err(error) = record_child_completion(
        &child_transcript,
        &summary,
        &parent_session_id,
        &parent_turn_id,
    ) {
        let message = format!("failed to record child subagent completion: {error}");
        emit_error(&event_sender, message.clone());
        set_hard_failure(&mut summary, message);
    }

    let parent_record_result = record_parent_result(
        &parent_transcript,
        &summary,
        &parent_session_id,
        &parent_turn_id,
    );
    if let Err(error) = parent_record_result {
        let message = format!("failed to record parent subagent result: {error}");
        emit_error(&event_sender, message.clone());
        set_hard_failure(&mut summary, message.clone());
        if let Err(child_error) = record_child_completion(
            &child_transcript,
            &summary,
            &parent_session_id,
            &parent_turn_id,
        ) {
            emit_error(
                &event_sender,
                format!(
                    "failed to record reconciled child subagent completion after parent failure: {child_error}"
                ),
            );
        }
    }
    guard.complete(summary.clone());
    if summary.status != SubagentStatus::Completed {
        emit_status(
            &event_sender,
            format!(
                "{} {} · {} · /child to inspect {}",
                summary.agent_name,
                summary.status.as_str(),
                summary.summary,
                short_session_id(&summary.child_session_id)
            ),
        );
    }
    Ok(summary)
}

fn set_hard_failure(summary: &mut SubagentRunSummary, message: String) {
    summary.status = SubagentStatus::Failed;
    summary.failure_kind = Some(SubagentFailureKind::Hard);
    summary.summary = message.clone();
    summary.structured_result.status = SubagentStatus::Failed.as_str().into();
    summary.structured_result.summary = message.clone();
    summary.structured_result.blockers.push(message);
}

fn record_parent_started(
    transcript: &Option<Arc<Mutex<TranscriptRecorder>>>,
    run_id: &str,
    parent_session_id: &str,
    parent_turn_id: &str,
    child_session_id: &str,
    agent_name: &str,
    summary: &str,
    pool_ordinal: u32,
) -> Result<()> {
    let Some(transcript) = transcript else {
        return Ok(());
    };
    let mut recorder = transcript
        .lock()
        .map_err(|_| anyhow!("parent transcript recorder poisoned"))?;
    recorder.record_subagent_started(
        run_id.to_string(),
        parent_session_id.to_string(),
        parent_turn_id.to_string(),
        child_session_id.to_string(),
        agent_name.to_string(),
        summary.to_string(),
        pool_ordinal,
    )
}

fn record_parent_result(
    transcript: &Option<Arc<Mutex<TranscriptRecorder>>>,
    summary: &SubagentRunSummary,
    parent_session_id: &str,
    parent_turn_id: &str,
) -> Result<()> {
    let Some(transcript) = transcript else {
        return Ok(());
    };
    let mut recorder = transcript
        .lock()
        .map_err(|_| anyhow!("parent transcript recorder poisoned"))?;
    recorder.record_subagent_result_structured(
        summary.run_id.clone(),
        parent_session_id.to_string(),
        parent_turn_id.to_string(),
        summary.child_session_id.clone(),
        summary.agent_name.clone(),
        summary.status.as_str().to_string(),
        summary.summary.clone(),
        Some(summary.structured_result.clone()),
    )
}

fn record_child_completion(
    transcript: &Arc<Mutex<TranscriptRecorder>>,
    summary: &SubagentRunSummary,
    parent_session_id: &str,
    parent_turn_id: &str,
) -> Result<()> {
    let mut recorder = transcript
        .lock()
        .map_err(|_| anyhow!("child transcript recorder poisoned"))?;
    recorder.record_subagent_lifecycle(
        summary.run_id.clone(),
        parent_session_id.to_string(),
        parent_turn_id.to_string(),
        summary.agent_name.clone(),
        summary.status.as_str().to_string(),
        Some(summary.summary.clone()),
    )
}

fn short_session_id(session_id: &str) -> &str {
    session_id.get(..12).unwrap_or(session_id)
}

pub(crate) fn generate_run_id() -> String {
    static NEXT_RUN_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let millis = current_timestamp_ms();
    let suffix = NEXT_RUN_ID.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    format!("subagent-{millis}-{suffix}")
}

fn current_timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn enforce_write_scope(
    summary: &mut SubagentRunSummary,
    governance: &SubagentRunGovernance,
    observed_changed_paths: &[String],
) {
    if summary.agent_name != "fixer" || !governance.input.has_write_scope() {
        return;
    }

    let mut changed_paths = summary.structured_result.files_changed.clone();
    for path in observed_changed_paths {
        if !changed_paths.contains(path) {
            changed_paths.push(path.clone());
        }
    }

    let offenders = changed_paths
        .iter()
        .filter(|path| !governance.input.permits_write_path(path))
        .cloned()
        .collect::<Vec<_>>();

    if offenders.is_empty() {
        return;
    }

    let message = format!("out-of-scope changes detected: {}", offenders.join(", "));
    summary.status = SubagentStatus::Failed;
    summary.failure_kind = Some(SubagentFailureKind::Logical);
    summary.summary = message.clone();
    summary.structured_result.status = SubagentStatus::Failed.as_str().to_string();
    summary.structured_result.summary = message.clone();
    summary.structured_result.blockers.push(message);
}

fn observed_changed_paths_from_child_transcript(
    transcript: &Arc<Mutex<TranscriptRecorder>>,
) -> Vec<String> {
    let path = match transcript.lock() {
        Ok(recorder) => recorder.path().to_path_buf(),
        Err(_) => return Vec::new(),
    };
    let Ok(records) = read_records_allow_partial_tail(path) else {
        return Vec::new();
    };

    let mut paths = Vec::new();
    for record in records {
        match record.event {
            TranscriptEvent::Evidence {
                evidence_kind: crate::evidence::EvidenceKind::Change,
                source: crate::evidence::EvidenceSource::File { path, .. },
                ..
            } => push_unique_path(&mut paths, path),
            TranscriptEvent::ToolCallFinished { name, output, .. }
                if matches!(
                    name.as_str(),
                    "fs__write" | "fs__append" | "fs__mkdir" | "edit__apply_patch"
                ) =>
            {
                collect_tool_output_paths(&mut paths, &output)
            }
            _ => {}
        }
    }
    paths
}

fn collect_tool_output_paths(paths: &mut Vec<String>, output: &crate::tool::ToolResult) {
    if let Some(path) = output
        .data
        .as_ref()
        .and_then(|data| data.get("path"))
        .and_then(Value::as_str)
    {
        push_unique_path(paths, path.to_string());
    }
    if let Some(edits) = output
        .data
        .as_ref()
        .and_then(|data| data.get("edits"))
        .and_then(Value::as_array)
    {
        for edit in edits {
            if let Some(path) = edit.get("path").and_then(Value::as_str) {
                push_unique_path(paths, path.to_string());
            }
        }
    }
}

fn push_unique_path(paths: &mut Vec<String>, path: String) {
    if !paths.contains(&path) {
        paths.push(path);
    }
}
