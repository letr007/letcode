use anyhow::{Context, Result, anyhow, bail};
use futures_util::FutureExt;
use serde_json::Value;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::oneshot;
use tokio::time::{Duration, Instant, timeout_at};

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

#[path = "messaging.rs"]
mod messaging;
pub(crate) use messaging::{MessageDelegateFactory, MessageHost};

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

#[derive(Debug)]
pub(crate) struct SubagentCapacityExceeded {
    pub(crate) limit: usize,
}

impl std::fmt::Display for SubagentCapacityExceeded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "swarm concurrency limit reached ({}); no run was started; wait for an active member to finish",
            self.limit
        )
    }
}

impl std::error::Error for SubagentCapacityExceeded {}

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
    swarms: std::collections::HashMap<String, SwarmAuthority>,
}

impl SubagentPoolState {
    fn swarm_occupancy(&self, root: &str) -> usize {
        let mut sessions = std::collections::HashSet::new();
        let mut starting = 0;
        for slot in self
            .active_by_run
            .values()
            .filter(|slot| slot.swarm_root.as_deref() == Some(root))
        {
            let target = if slot.child.child_session_id.is_empty() {
                slot.takeover_child_session_id.as_deref()
            } else {
                Some(slot.child.child_session_id.as_str())
            };
            if let Some(target) = target {
                sessions.insert(target);
            } else {
                starting += 1;
            }
        }
        if let Some(authority) = self.swarms.get(root) {
            sessions.extend(authority.mailboxes.iter().filter_map(|(target, mailbox)| {
                (target != root && mailbox.waking).then_some(target.as_str())
            }));
        }
        sessions.len() + starting
    }

    fn swarm_scope_is_stopping(&self, root: &str, session: &str) -> bool {
        let mut ancestor = Some(session);
        while let Some(session) = ancestor {
            if self.active_by_run.values().any(|slot| {
                slot.swarm_root.as_deref() == Some(root)
                    && slot.child.child_session_id == session
                    && slot.cancellation_requested
            }) || self
                .swarms
                .get(root)
                .and_then(|authority| authority.mailboxes.get(session))
                .is_some_and(|mailbox| mailbox.wake_cancelled)
            {
                return true;
            }
            ancestor = self
                .swarms
                .get(root)
                .and_then(|authority| authority.members.get(session))
                .map(|member| member.parent_session_id.as_str());
        }
        false
    }
}

#[derive(Default)]
struct SwarmAuthority {
    policy: Option<crate::config::SubagentsConfig>,
    members: std::collections::BTreeMap<String, super::SwarmMember>,
    next_ordinal: u32,
    mailboxes: std::collections::HashMap<String, messaging::Mailbox>,
    message_host: Option<Arc<MessageHost>>,
    generation: u64,
}

impl SwarmAuthority {
    fn merge_member(&mut self, member: &super::SwarmMember) -> Result<()> {
        if let Some(existing) = self.members.get(&member.child_session_id) {
            anyhow::ensure!(existing == member, "conflicting live swarm member identity");
        } else {
            anyhow::ensure!(
                !self
                    .members
                    .values()
                    .any(|existing| existing.pool_ordinal == member.pool_ordinal),
                "conflicting live swarm ordinal"
            );
            self.members
                .insert(member.child_session_id.clone(), member.clone());
        }
        self.next_ordinal = self.next_ordinal.max(member.pool_ordinal.saturating_add(1));
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum RunPathAccess {
    Read(Vec<PathBuf>),
    Write(Vec<PathBuf>),
}

#[derive(Debug)]
struct ActiveSlot {
    cancel: Option<oneshot::Sender<()>>,
    cancellation_requested: bool,
    child: ChildSessionSummary,
    path_access: RunPathAccess,
    takeover_child_session_id: Option<String>,
    swarm_root: Option<String>,
    parent_run_id: Option<String>,
}

struct SwarmAdmission<'a> {
    root_session_id: &'a str,
    parent_session_id: &'a str,
    depth: u32,
    policy: &'a crate::config::SubagentsConfig,
    message_resume: bool,
}

struct RunReservation {
    state: Arc<Mutex<SubagentPoolState>>,
    changed: Arc<tokio::sync::Notify>,
    run_id: String,
    activated: bool,
}

impl RunReservation {
    fn activate(
        mut self,
        cancel: oneshot::Sender<()>,
        child: ChildSessionSummary,
        activation: Option<(&Arc<Mutex<TranscriptRecorder>>, super::SwarmMember)>,
        timeout_secs: Option<u64>,
    ) -> Result<Option<Instant>> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow!("subagent pool lock poisoned"))?;
        if let Some((recorder, member)) = activation {
            let authority = state
                .swarms
                .entry(member.root_session_id.clone())
                .or_default();
            if let Some(existing) = authority.members.get(&member.child_session_id) {
                anyhow::ensure!(existing == &member, "conflicting swarm activation identity");
            }
            anyhow::ensure!(
                !authority
                    .members
                    .values()
                    .any(
                        |existing| existing.child_session_id != member.child_session_id
                            && existing.pool_ordinal == member.pool_ordinal
                    ),
                "conflicting swarm activation ordinal"
            );
            let mut recorder = recorder
                .lock()
                .map_err(|_| anyhow!("child transcript recorder poisoned"))?;
            if let Err(error) =
                recorder.record_subagent_activation(self.run_id.clone(), member.clone())
            {
                let records = read_records_allow_partial_tail(recorder.path())?;
                if super::swarm::caller_identity(&records)?.as_ref() == Some(&member) {
                    authority.merge_member(&member)?;
                }
                return Err(error);
            }
            authority.merge_member(&member)?;
        }
        let deadline = timeout_secs
            .map(|secs| {
                Instant::now()
                    .checked_add(Duration::from_secs(secs))
                    .ok_or_else(|| {
                        anyhow!("swarm run timeout exceeds the supported deadline range")
                    })
            })
            .transpose()?;
        let slot = state
            .active_by_run
            .get_mut(&self.run_id)
            .ok_or_else(|| anyhow!("subagent run reservation disappeared"))?;
        slot.cancel = Some(cancel);
        slot.child = child;
        if slot.cancellation_requested
            && let Some(cancel) = slot.cancel.take()
        {
            let _ = cancel.send(());
        }
        self.activated = true;
        drop(state);
        self.changed.notify_waiters();
        Ok(deadline)
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
    runtime: SubagentPool,
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
        runtime: SubagentPool,
    ) -> Self {
        Self {
            runtime,
            state,
            changed,
            run_id,
            terminal_context,
            terminal: false,
        }
    }

    fn complete(&mut self, summary: SubagentRunSummary) {
        if let Ok(mut state) = self.state.lock() {
            if summary.status != SubagentStatus::Completed {
                cancel_descendants(&mut state, &self.run_id);
            }
            if let Some(root) = state
                .active_by_run
                .get(&self.run_id)
                .and_then(|slot| slot.swarm_root.clone())
            {
                state
                    .swarms
                    .get_mut(&root)
                    .expect("swarm authority")
                    .mailboxes
                    .entry(summary.child_session_id.clone())
                    .or_default()
                    .waking = true;
            }
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
        self.runtime.stop_swarm_members(Some(&self.run_id));
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
            cancel_descendants(&mut state, &self.run_id);
            if let Some(root) = state
                .active_by_run
                .get(&self.run_id)
                .and_then(|slot| slot.swarm_root.clone())
                && let Some(mailbox) = state
                    .swarms
                    .get_mut(&root)
                    .and_then(|authority| authority.mailboxes.get_mut(&context.child_session_id))
            {
                mailbox.waking = false;
            }
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

    pub(crate) fn swarm_snapshot(
        &self,
        sessions_dir: &Path,
        records: &[crate::transcript::TranscriptRecord],
    ) -> Result<super::swarm::SwarmSnapshot> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow!("subagent pool lock poisoned"))?;
        let caller = records
            .first()
            .ok_or_else(|| anyhow!("missing swarm session records"))?;
        let records = super::swarm::read_session(
            sessions_dir,
            &caller.session_id,
            super::swarm::caller_identity(records)?.is_some(),
        )?;
        let mut snapshot = super::swarm::project(sessions_dir, &records)?;
        let authority = state
            .swarms
            .entry(snapshot.root.session_id.clone())
            .or_default();
        if let Some(policy) = &authority.policy {
            anyhow::ensure!(policy == &snapshot.policy, "conflicting swarm root policy");
        }
        authority.policy = Some(snapshot.policy.clone());
        for member in snapshot.members.values() {
            authority.merge_member(member)?;
        }
        for run in &mut snapshot.runs {
            let run_id = run.run_id.as_deref().expect("member run id");
            if let Some(result) = state.completed_by_run.get(run_id)
                && result.child_session_id == run.session_id
            {
                run.status = result.status.as_str().into();
                run.active = false;
            } else if let Some(slot) = state.active_by_run.get(run_id)
                && slot.child.child_session_id == run.session_id
            {
                run.status = slot.child.status.clone();
                run.active = true;
            }
        }
        Ok(snapshot)
    }

    fn activate_run(
        &self,
        reservation: RunReservation,
        cancel: oneshot::Sender<()>,
        child: ChildSessionSummary,
        child_transcript: &Arc<Mutex<TranscriptRecorder>>,
        parent_transcript: &Option<Arc<Mutex<TranscriptRecorder>>>,
        member: Option<super::SwarmMember>,
        timeout_secs: Option<u64>,
    ) -> Result<Option<Instant>> {
        let run_id = reservation.run_id.clone();
        let activation = reservation.activate(
            cancel,
            child.clone(),
            member.map(|member| (child_transcript, member)),
            timeout_secs,
        );
        if let Err(error) = &activation {
            let mut summary = build_runtime_summary(
                &run_id,
                &child.child_session_id,
                &child.agent_name,
                SubagentStatus::Failed,
                format!("failed to activate subagent: {error:#}"),
            );
            let child_outcome = (|| -> Result<()> {
                let mut recorder = child_transcript
                    .lock()
                    .map_err(|_| anyhow!("child transcript recorder poisoned"))?;
                let path = recorder.path().to_path_buf();
                let reopened = TranscriptRecorder::open_after_partial_tail_repair(
                    path.parent()
                        .ok_or_else(|| anyhow!("child transcript has no directory"))?,
                    &child.child_session_id,
                )?;
                *recorder = reopened;
                drop(recorder);
                record_child_completion(
                    child_transcript,
                    &summary,
                    &child.parent_session_id,
                    &child.parent_run_id,
                )
            })();
            if let Err(cleanup_error) = child_outcome {
                let message = format!(
                    "{}; failed to record child activation failure: {cleanup_error:#}",
                    summary.summary
                );
                set_hard_failure(&mut summary, message);
            }
            if let Err(cleanup_error) = record_parent_result(
                parent_transcript,
                &summary,
                &child.parent_session_id,
                &child.parent_run_id,
            ) {
                let message = format!(
                    "{}; failed to record parent activation failure: {cleanup_error:#}",
                    summary.summary
                );
                set_hard_failure(&mut summary, message);
            }
            let message = summary.summary.clone();
            self.state
                .lock()
                .map_err(|_| anyhow!("{message}; subagent pool lock poisoned"))?
                .completed_by_run
                .insert(run_id, summary);
            self.changed.notify_waiters();
            bail!("{message}");
        }
        activation
    }

    fn allocate_swarm_ordinal(&self, root_session_id: &str) -> Result<u32> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow!("subagent pool lock poisoned"))?;
        let authority = state.swarms.entry(root_session_id.to_string()).or_default();
        let ordinal = authority.next_ordinal.max(1);
        authority.next_ordinal = ordinal
            .checked_add(1)
            .ok_or_else(|| anyhow!("swarm ordinal exhausted"))?;
        Ok(ordinal)
    }

    pub fn cancel_active(&self) -> bool {
        !self.cancel_active_run_ids().is_empty()
    }

    pub fn cancel_active_run_ids(&self) -> Vec<String> {
        self.stop_swarm_members(None);
        let Ok(mut state) = self.state.lock() else {
            return Vec::new();
        };
        let mut run_ids = Vec::with_capacity(state.active_by_run.len());
        for (run_id, slot) in &mut state.active_by_run {
            run_ids.push(run_id.clone());
            slot.cancellation_requested = true;
            if let Some(tx) = slot.cancel.take() {
                let _ = tx.send(());
            }
        }
        if !run_ids.is_empty() {
            self.changed.notify_waiters();
        }
        run_ids
    }

    pub async fn wait_until_idle(&self) -> bool {
        let deadline = tokio::time::sleep(Duration::from_secs(3));
        tokio::pin!(deadline);
        loop {
            if !self.is_running() {
                return true;
            }
            let notified = self.changed.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if !self.is_running() {
                return true;
            }
            tokio::select! {
                _ = &mut deadline => return !self.is_running(),
                _ = notified => {}
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn starting_reservation_for_test(&self) -> StartingReservationForTest {
        let reservation = self
            .reserve_run(RunPathAccess::Read(Vec::new()), None)
            .expect("test reservation");
        StartingReservationForTest {
            reservation: Some(reservation),
        }
    }

    #[cfg(test)]
    pub(crate) fn record_completed_for_test(&self, summary: SubagentRunSummary) {
        let mut state = self.state.lock().expect("subagent pool lock");
        state
            .completed_by_run
            .insert(summary.run_id.clone(), summary);
    }

    pub fn cancel_run(&self, run_id: &str) -> bool {
        let swarm = self.state.lock().ok().and_then(|state| {
            state
                .active_by_run
                .get(run_id)
                .map(|slot| slot.swarm_root.is_some())
        });
        if swarm == Some(true) {
            self.stop_swarm_members(Some(run_id));
            return true;
        }
        let Ok(mut state) = self.state.lock() else {
            return false;
        };
        let Some(slot) = state.active_by_run.get_mut(run_id) else {
            return false;
        };
        slot.cancellation_requested = true;
        match slot.cancel.take() {
            Some(tx) => tx.send(()).is_ok(),
            None => true,
        }
    }

    pub fn is_running(&self) -> bool {
        self.state
            .lock()
            .map(|state| {
                !state.active_by_run.is_empty()
                    || state.swarms.values().any(|authority| {
                        authority.mailboxes.iter().any(|(target, mailbox)| {
                            authority.members.contains_key(target) && mailbox.waking
                        })
                    })
            })
            .unwrap_or(false)
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

    #[cfg(test)]
    fn reserve_run(
        &self,
        path_access: RunPathAccess,
        takeover_child_session_id: Option<&str>,
    ) -> Result<RunReservation> {
        self.reserve_governed_run(path_access, takeover_child_session_id, None)
    }

    fn reserve_governed_run(
        &self,
        path_access: RunPathAccess,
        takeover_child_session_id: Option<&str>,
        admission: Option<SwarmAdmission<'_>>,
    ) -> Result<RunReservation> {
        let run_id = generate_run_id();
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow!("subagent pool lock poisoned"))?;
        let mut parent_run_id = None;
        let mut swarm_root = None;
        let mut parent_session_id = String::new();
        if let Some(admission) = admission {
            anyhow::ensure!(
                admission.depth as usize <= admission.policy.max_depth,
                "swarm maximum delegation depth exhausted"
            );
            anyhow::ensure!(
                !state.swarm_scope_is_stopping(
                    admission.root_session_id,
                    admission.parent_session_id
                ),
                "swarm ancestor run is stopping"
            );
            if !admission.message_resume && admission.parent_session_id != admission.root_session_id
            {
                let (run, slot) = state
                    .active_by_run
                    .iter()
                    .find(|(_, slot)| {
                        slot.child.child_session_id == admission.parent_session_id
                            && slot.swarm_root.as_deref() == Some(admission.root_session_id)
                    })
                    .ok_or_else(|| anyhow!("swarm parent run is no longer active"))?;
                anyhow::ensure!(!slot.cancellation_requested, "swarm parent run is stopping");
                parent_run_id = Some(run.clone());
            }
            let occupied = state.swarm_occupancy(admission.root_session_id);
            let authority = state
                .swarms
                .entry(admission.root_session_id.to_string())
                .or_default();
            if let Some(policy) = &authority.policy {
                anyhow::ensure!(policy == admission.policy, "conflicting swarm root policy");
            }
            authority.policy = Some(admission.policy.clone());
            anyhow::ensure!(
                !authority
                    .mailboxes
                    .get(admission.parent_session_id)
                    .is_some_and(|mailbox| mailbox.stopped),
                "swarm parent scope is stopping"
            );
            if let Some(target) = takeover_child_session_id {
                let mailbox = authority.mailboxes.entry(target.to_string()).or_default();
                anyhow::ensure!(
                    !mailbox.stopped && (!admission.message_resume || !mailbox.wake_cancelled),
                    "swarm target is stopping"
                );
                anyhow::ensure!(
                    admission.message_resume || !mailbox.waking,
                    "swarm target already has a resume owner"
                );
            }
            if let Some(target) = takeover_child_session_id {
                anyhow::ensure!(
                    authority.members.contains_key(target),
                    "takeover failed: swarm child is not activated"
                );
            }
            let wake_reserved = admission.message_resume
                && takeover_child_session_id.is_some_and(|target| {
                    authority
                        .mailboxes
                        .get(target)
                        .is_some_and(|mailbox| mailbox.waking)
                });
            if !wake_reserved && occupied >= admission.policy.max_concurrent {
                return Err(SubagentCapacityExceeded {
                    limit: admission.policy.max_concurrent,
                }
                .into());
            }
            swarm_root = Some(admission.root_session_id.to_string());
            parent_session_id = admission.parent_session_id.to_string();
        }
        ensure_path_access_available_in_map(&state.active_by_run, &path_access)?;
        if let Some(target) = takeover_child_session_id
            && state.active_by_run.values().any(|slot| {
                slot.takeover_child_session_id.as_deref() == Some(target)
                    || slot.child.child_session_id == target
            })
        {
            bail!("takeover failed: child `{target}` already has an active takeover");
        }
        state.active_by_run.insert(
            run_id.clone(),
            ActiveSlot {
                cancel: None,
                cancellation_requested: false,
                child: ChildSessionSummary {
                    parent_session_id,
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
                swarm_root,
                parent_run_id,
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
        Self::child_sessions_from_summaries(list_child_sessions_for_parent(
            sessions_dir,
            parent_records,
        ))
    }

    pub(crate) fn child_sessions_from_summaries(
        mut children: Vec<ChildSessionSummary>,
    ) -> Vec<ChildSessionSummary> {
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

    pub fn start_named_governed(
        &self,
        parent: &Agent,
        agent_name: &str,
        invocation: SubagentInvocation,
        sessions_dir: impl AsRef<Path>,
        parent_session_id: String,
        parent_turn_id: String,
        parent_transcript: Option<Arc<Mutex<TranscriptRecorder>>>,
        event_sender: Option<SubagentEventSender>,
    ) -> Result<StartedSubagentRun> {
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
            false,
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

    pub async fn complete_started_run(
        &self,
        started: StartedSubagentRun,
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

    pub(crate) async fn complete_started_run_with_executor<F>(
        &self,
        started: StartedSubagentRun,
        exec: F,
    ) -> Result<SubagentRunSummary>
    where
        F: FnOnce(
                Agent,
                String,
                Arc<Mutex<TranscriptRecorder>>,
                Option<SubagentEventSender>,
                String,
                String,
            ) -> BoxExecFuture
            + Send
            + 'static,
    {
        complete_started_run(started.run, started.task, exec).await
    }

    pub async fn run_with_executor<F>(
        &self,
        parent: &Agent,
        template: AgentTemplate,
        task: String,
        governance: SubagentRunGovernance,
        sessions_dir: PathBuf,
        parent_session_id: String,
        parent_turn_id: String,
        parent_transcript: Option<Arc<Mutex<TranscriptRecorder>>>,
        event_sender: Option<SubagentEventSender>,
        takeover_child_session_id: Option<String>,
        exec: F,
    ) -> Result<SubagentRunSummary>
    where
        F: FnOnce(
                Agent,
                String,
                Arc<Mutex<TranscriptRecorder>>,
                Option<SubagentEventSender>,
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
            false,
        )?;
        complete_started_run(running, task, exec).await
    }

    fn start_run(
        &self,
        parent: &Agent,
        template: AgentTemplate,
        task: String,
        governance: SubagentRunGovernance,
        sessions_dir: PathBuf,
        parent_session_id: String,
        parent_turn_id: String,
        parent_transcript: Option<Arc<Mutex<TranscriptRecorder>>>,
        event_sender: Option<SubagentEventSender>,
        takeover_child_session_id: Option<String>,
        message_resume: bool,
    ) -> Result<StartedRun> {
        let mut governance = governance;
        let mut template = template;
        let effective_max_tool_calls = governance.max_tool_calls.or(template.max_tool_calls);
        if takeover_child_session_id.is_some() && governance.model.is_some() {
            bail!("model override cannot be used when taking over a child session");
        }

        anyhow::ensure!(
            parent_transcript.is_some()
                || parent.subagents_policy().mode != crate::config::SubagentMode::Swarm,
            "swarm delegation requires a parent transcript"
        );
        let recorded_parent = parent_transcript
            .as_ref()
            .map(|recorder| {
                let recorder = recorder
                    .lock()
                    .map_err(|_| anyhow!("transcript recorder poisoned"))?;
                read_records_allow_partial_tail(recorder.path())
            })
            .transpose()?;
        let recorded_policy = recorded_parent
            .as_ref()
            .map(|records| crate::transcript::restore_subagents_policy(records))
            .transpose()?;
        let recorded_member = recorded_parent
            .as_ref()
            .map(|records| super::swarm::caller_identity(records))
            .transpose()?
            .flatten();
        let swarm_snapshot = if recorded_member.is_some()
            || recorded_policy
                .as_ref()
                .is_some_and(|policy| policy.mode == crate::config::SubagentMode::Swarm)
        {
            let records = recorded_parent.as_ref().expect("recorded swarm policy");
            anyhow::ensure!(
                records
                    .first()
                    .is_some_and(|record| record.session_id == parent_session_id),
                "swarm parent session mismatch"
            );
            Some(self.swarm_snapshot(&sessions_dir, records)?)
        } else {
            None
        };
        let swarm_parent = swarm_snapshot.as_ref().map(|snapshot| {
            (
                snapshot.root.session_id.clone(),
                snapshot
                    .members
                    .get(&parent_session_id)
                    .map_or(0, |member| member.depth),
            )
        });

        let policy = swarm_snapshot.as_ref().map_or_else(
            || recorded_policy.unwrap_or_else(|| parent.subagents_policy().clone()),
            |snapshot| snapshot.policy.clone(),
        );
        let effective_timeout_secs = governance
            .timeout_secs
            .or(template.timeout_secs)
            .or_else(|| swarm_parent.as_ref().map(|_| policy.run_timeout_secs));
        if swarm_parent.is_some() {
            policy.validate()?;
            if let Some(secs) = effective_timeout_secs {
                anyhow::ensure!(secs > 0, "swarm run timeout must be positive");
                anyhow::ensure!(
                    Instant::now()
                        .checked_add(Duration::from_secs(secs))
                        .is_some(),
                    "swarm run timeout exceeds the supported deadline range"
                );
            }
            anyhow::ensure!(
                !template.can_write || parent.permits_delegation_write(),
                "read-only caller cannot delegate writable work"
            );
            template = template.for_mode(crate::config::SubagentMode::Swarm);
        }
        if crate::delegation::DELEGATION_EXPERTS
            .iter()
            .any(|expert| expert.agent_name == template.name)
        {
            let contract = if swarm_parent.is_some() {
                "可在递归深度和并发上限内委派；只读角色只能委派只读任务；保持父作用域并简洁报告结果。"
            } else {
                "不要递归委派；保持在给定范围内，并简洁报告发现或实现结果。"
            };
            template.system_prompt = format!(
                "{}\n\n委派约定：{}\n\n{}",
                template.system_prompt,
                contract,
                crate::subagent::REPORT_PROMPT
            );
        }
        if swarm_parent.is_some()
            && let Some(target) = &takeover_child_session_id
        {
            let snapshot = swarm_snapshot
                .as_ref()
                .ok_or_else(|| anyhow!("takeover requires swarm ancestry records"))?;
            let member = snapshot
                .members
                .get(target)
                .ok_or_else(|| anyhow!("unknown swarm takeover target"))?;
            anyhow::ensure!(
                member.parent_session_id == parent_session_id,
                "takeover target is not a child of this parent"
            );
            let records = super::swarm::read_session(&sessions_dir, target, true)?;
            let original = records
                .iter()
                .find_map(|record| match &record.event {
                    TranscriptEvent::SubagentDelegation { input, .. } => Some(input),
                    _ => None,
                })
                .ok_or_else(|| anyhow!("takeover failed: missing original delegation scope"))?;
            let supplied = &governance.input;
            if !supplied.allowed_paths.is_empty()
                || !supplied.owned_paths.is_empty()
                || !supplied.forbidden_paths.is_empty()
            {
                anyhow::ensure!(
                    SubagentPathScope::from_input(supplied)?
                        == SubagentPathScope::from_input(original)?,
                    "takeover cannot change the original delegation scope"
                );
            }
            governance.input.allowed_paths = original.allowed_paths.clone();
            governance.input.owned_paths = original.owned_paths.clone();
            governance.input.forbidden_paths = original.forbidden_paths.clone();
        }
        let mut scope = SubagentPathScope::from_input(&governance.input)?;
        if swarm_parent.is_some()
            && let Some(parent_scope) = parent.delegation_scope()
        {
            scope = Some(match scope {
                Some(scope) => scope.narrowed_from(parent_scope)?,
                None => parent_scope.clone(),
            });
        }
        let path_access = run_path_access(&template, &governance.input, scope.as_ref())?;
        let mut reservation = self.reserve_governed_run(
            path_access.clone(),
            takeover_child_session_id.as_deref(),
            swarm_parent
                .as_ref()
                .map(|(root, depth)| -> Result<_> {
                    Ok(SwarmAdmission {
                        root_session_id: root,
                        parent_session_id: &parent_session_id,
                        depth: depth
                            .checked_add(1)
                            .ok_or_else(|| anyhow!("swarm depth overflow"))?,
                        policy: &policy,
                        message_resume,
                    })
                })
                .transpose()?,
        )?;

        let mut existing_children = if swarm_snapshot.is_some() {
            let recorder = parent_transcript
                .as_ref()
                .expect("swarm parent recorder")
                .lock()
                .map_err(|_| anyhow!("transcript recorder poisoned"))?;
            Self::child_sessions(
                &sessions_dir,
                &read_records_allow_partial_tail(recorder.path())?,
            )
        } else {
            parent_transcript
                .as_ref()
                .and_then(|recorder| recorder.lock().ok())
                .map(|recorder| recorder.path().to_path_buf())
                .and_then(|parent_path| {
                    transcript_projection::project_child_session_summaries_from_file(
                        &child_sessions_dir(&sessions_dir),
                        &parent_path,
                    )
                    .ok()
                })
                .map(SubagentPool::child_sessions_from_summaries)
                .unwrap_or_default()
        };
        if let Some(snapshot) = &swarm_snapshot {
            let jobs = snapshot.latest_jobs();
            existing_children.retain_mut(|child| {
                let Some(job) = jobs
                    .iter()
                    .find(|job| job.session_id == child.child_session_id)
                else {
                    return false;
                };
                child.status = job.status.clone();
                child.agent_name = job.role.clone();
                child.pool_ordinal = job.pool_ordinal.expect("member ordinal");
                true
            });
        }

        let setup = (|| -> Result<(String, String, u32, Arc<Mutex<TranscriptRecorder>>, Agent)> {
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

                if let Some((root, _)) = &swarm_parent {
                    let state = self
                        .state
                        .lock()
                        .map_err(|_| anyhow!("subagent pool lock poisoned"))?;
                    anyhow::ensure!(
                        state
                            .swarms
                            .get(root)
                            .is_some_and(|authority| authority.members.contains_key(target_id)),
                        "takeover failed: swarm child is not activated"
                    );
                }
                let pool_ordinal = if target.pool_ordinal > 0 {
                    target.pool_ordinal
                } else {
                    self.allocate_ordinal_from_children(&existing_children)
                };

                let child_transcript = if let Some((root, _)) = &swarm_parent {
                    self.member_transcript(root, target_id, &sessions_dir)?
                } else {
                    Arc::new(Mutex::new(TranscriptRecorder::open(
                        &child_dir,
                        target_id.clone(),
                    )?))
                };
                let mut child_recorder = child_transcript
                    .lock()
                    .map_err(|_| anyhow!("child transcript recorder poisoned"))?;
                let child_records = read_records_allow_partial_tail(child_recorder.path())?;
                let subagents_policy = crate::transcript::restore_subagents_policy(&child_records)?;
                let recorded_route = crate::transcript::restore_latest_model(&child_records)
                    .ok_or_else(|| {
                        anyhow!("takeover failed: child `{target_id}` has no recorded model route")
                    })?;
                let recorded_route = ModelRoute::parse(&recorded_route).with_context(|| {
                    format!("takeover failed: child `{target_id}` recorded an invalid model route")
                })?;
                let mut child_agent = if message_resume {
                    AgentFactory::create_resumed_child(
                        parent,
                        &template,
                        recorded_route,
                        effective_max_tool_calls,
                    )?
                } else {
                    AgentFactory::create_child_with_route_and_max_tool_calls(
                        parent,
                        &template,
                        Some(recorded_route),
                        true,
                        effective_max_tool_calls,
                    )?
                };
                if swarm_parent.is_some() {
                    anyhow::ensure!(subagents_policy == policy, "takeover swarm policy mismatch");
                }
                child_agent.set_subagents_policy(subagents_policy);
                if swarm_parent.is_some()
                    && let Some(mode) =
                        crate::transcript::restore_latest_permission_mode(&child_records)
                {
                    child_agent.set_permission_mode(
                        crate::permission::PermissionMode::parse(&mode)
                            .ok_or_else(|| anyhow!("invalid recorded member permission mode"))?,
                    );
                }
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
                let runtime_snapshot =
                    child_agent.validate_runtime_snapshot_restore(snapshot.snapshot)?;

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

                child_recorder
                    .record_subagent_delegation(run_id.clone(), governance.input.clone())?;
                child_agent.install_validated_runtime_snapshot(runtime_snapshot);
                child_agent.restore_turn_sequence(snapshot.max_turn_id);
                drop(child_recorder);
                Ok((
                    run_id,
                    target_id.clone(),
                    pool_ordinal,
                    child_transcript,
                    child_agent,
                ))
            } else {
                let pool_ordinal = if let Some((root, _)) = &swarm_parent {
                    self.allocate_swarm_ordinal(root)?
                } else {
                    self.allocate_ordinal_from_children(&existing_children)
                };
                let mut child_recorder = TranscriptRecorder::create(&child_dir)?;
                let mut child_agent = AgentFactory::create_child_with_route_and_max_tool_calls(
                    parent,
                    &template,
                    governance.model.clone(),
                    false,
                    effective_max_tool_calls,
                )?;
                child_agent.set_subagents_policy(policy.clone());
                child_agent.set_subagent_path_scope(scope.clone().map(Arc::new));
                child_agent.set_context_scope_state(child_recorder.context_scope_state());
                child_recorder.record_session_started(child_agent.route_display_name())?;
                child_recorder.record_subagents_policy(child_agent.subagents_policy())?;
                if swarm_parent.is_some() {
                    child_recorder.record_permission_mode_changed(
                        "default",
                        child_agent.permission_mode().as_str(),
                    )?;
                }
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
                child_recorder
                    .record_subagent_delegation(run_id.clone(), governance.input.clone())?;
                let snapshot = transcript_projection::project_runtime_restore_snapshot(
                    child_session_id.clone(),
                    read_records_allow_partial_tail(child_recorder.path())?,
                    transcript_projection::SessionContextCursor {
                        branch_id: None,
                        leaf_sequence: None,
                    },
                    &[],
                )?;
                child_agent.restore_runtime_snapshot(snapshot.snapshot)?;
                Ok((
                    run_id,
                    child_session_id,
                    pool_ordinal,
                    Arc::new(Mutex::new(child_recorder)),
                    child_agent,
                ))
            }
        })();

        let (run_id, child_session_id, pool_ordinal, child_transcript, mut child_agent) = setup?;
        if swarm_parent.is_some() {
            child_agent
                .set_subagent_delegate(parent.child_delegate(Arc::clone(&child_transcript))?);
        }
        if message_resume {
            child_agent.mark_member_message_start();
        }

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
                && let Some(parent_dir) = parent_path.parent()
                && let Ok(mut recorder) = TranscriptRecorder::open_after_partial_tail_repair(
                    parent_dir,
                    &parent_session_id,
                )
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

        let member = swarm_parent
            .as_ref()
            .map(|(root_session_id, parent_depth)| super::SwarmMember {
                root_session_id: root_session_id.clone(),
                parent_session_id: parent_session_id.clone(),
                child_session_id: child_session_id.clone(),
                depth: parent_depth + 1,
                agent_name: template.name.clone(),
                pool_ordinal,
            });
        if let Some((root, _)) = &swarm_parent {
            let mut state = self
                .state
                .lock()
                .map_err(|_| anyhow!("subagent pool lock poisoned"))?;
            let mailbox = state
                .swarms
                .get_mut(root)
                .expect("swarm authority")
                .mailboxes
                .entry(child_session_id.clone())
                .or_default();
            mailbox
                .execution_sender
                .get_or_insert_with(|| event_sender.clone());
            mailbox.transcript = Arc::downgrade(&child_transcript);
        }
        let deadline = self.activate_run(
            reservation,
            cancel_tx,
            summary,
            &child_transcript,
            &parent_transcript,
            member,
            swarm_parent.as_ref().and(effective_timeout_secs),
        )?;

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
                self.clone(),
            ),
            path_access,
            run_id,
            child_session_id,
            agent_name: template.name.clone(),
            timeout_secs: effective_timeout_secs,
            deadline,
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

fn cancel_descendants(state: &mut SubagentPoolState, parent_run_id: &str) {
    if let Some((root, session)) = state.active_by_run.get(parent_run_id).and_then(|slot| {
        slot.swarm_root
            .as_ref()
            .map(|root| (root.clone(), slot.child.child_session_id.clone()))
    }) {
        let authority = &state.swarms[&root];
        let mut pending = vec![session.clone()];
        let mut descendants = std::collections::HashSet::new();
        while let Some(parent) = pending.pop() {
            for member in authority
                .members
                .values()
                .filter(|member| member.parent_session_id == parent)
            {
                if descendants.insert(member.child_session_id.clone()) {
                    pending.push(member.child_session_id.clone());
                }
            }
        }
        for target in &descendants {
            if let Some(mailbox) = state
                .swarms
                .get_mut(&root)
                .and_then(|authority| authority.mailboxes.get_mut(target))
                && mailbox.waking
            {
                mailbox.wake_cancelled = true;
            }
        }
        for slot in state
            .active_by_run
            .values_mut()
            .filter(|slot| slot.swarm_root.as_ref() == Some(&root))
        {
            if descendants.contains(&slot.child.child_session_id)
                || slot
                    .takeover_child_session_id
                    .as_ref()
                    .is_some_and(|target| descendants.contains(target))
                || (slot.child.child_session_id.is_empty()
                    && (slot.child.parent_session_id == session
                        || descendants.contains(&slot.child.parent_session_id)))
            {
                slot.cancellation_requested = true;
                if let Some(cancel) = slot.cancel.take() {
                    let _ = cancel.send(());
                }
            }
        }
    }
    let mut pending = vec![parent_run_id.to_string()];
    while let Some(parent) = pending.pop() {
        for (run_id, slot) in &mut state.active_by_run {
            if slot.parent_run_id.as_deref() == Some(parent.as_str()) {
                slot.cancellation_requested = true;
                if let Some(cancel) = slot.cancel.take() {
                    let _ = cancel.send(());
                }
                pending.push(run_id.clone());
            }
        }
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

    // Tool-free internal workers consume a supplied snapshot, not repository
    // files. They must not acquire a workspace read lock against active fixers.
    if template.max_tool_calls == Some(0) && input.max_tool_calls.unwrap_or(0) == 0 {
        return Ok(RunPathAccess::Read(Vec::new()));
    }
    let roots = match scope.map(SubagentPathScope::read_roots) {
        Some(roots) if !roots.is_empty() => roots,
        _ => vec![crate::tool::workspace_root_for_subagent_lock()?],
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

#[cfg(test)]
pub(crate) struct StartingReservationForTest {
    reservation: Option<RunReservation>,
}

#[cfg(test)]
impl StartingReservationForTest {
    pub(crate) fn run_id(&self) -> &str {
        &self.reservation.as_ref().expect("reservation held").run_id
    }

    pub(crate) fn activate(mut self) -> oneshot::Receiver<()> {
        let reservation = self.reservation.take().expect("reservation held");
        let (cancel_tx, cancel_rx) = oneshot::channel();
        reservation
            .activate(
                cancel_tx,
                ChildSessionSummary {
                    parent_session_id: String::new(),
                    parent_run_id: String::new(),
                    child_session_id: String::new(),
                    agent_name: String::new(),
                    status: "starting".into(),
                    summary: String::new(),
                    timestamp_ms: current_timestamp_ms(),
                    pool_ordinal: 0,
                },
                None,
                None,
            )
            .expect("test activation");
        cancel_rx
    }
}

pub struct StartedSubagentRun {
    run_id: String,
    receipt: ChildSessionSummary,
    run: StartedRun,
    task: String,
}

impl StartedSubagentRun {
    pub fn run_id(&self) -> &str {
        &self.run_id
    }

    pub fn receipt(&self) -> &ChildSessionSummary {
        &self.receipt
    }
}

struct StartedRun {
    guard: ActiveRunGuard,
    path_access: RunPathAccess,
    run_id: String,
    child_session_id: String,
    agent_name: String,
    timeout_secs: Option<u64>,
    deadline: Option<Instant>,
    governance: SubagentRunGovernance,
    parent_session_id: String,
    parent_turn_id: String,
    parent_transcript: Option<Arc<Mutex<TranscriptRecorder>>>,
    event_sender: Option<SubagentEventSender>,
    child_transcript: Arc<Mutex<TranscriptRecorder>>,
    child_agent: Agent,
    cancel_rx: oneshot::Receiver<()>,
}

async fn complete_started_run<F>(
    started: StartedRun,
    task: String,
    exec: F,
) -> Result<SubagentRunSummary>
where
    F: FnOnce(
            Agent,
            String,
            Arc<Mutex<TranscriptRecorder>>,
            Option<SubagentEventSender>,
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
        deadline,
        governance,
        parent_session_id,
        parent_turn_id,
        parent_transcript,
        event_sender,
        child_transcript,
        child_agent,
        mut cancel_rx,
    } = started;

    let effective_scope = (child_agent.subagents_policy().mode
        == crate::config::SubagentMode::Swarm)
        .then(|| child_agent.delegation_scope().cloned())
        .flatten();
    let cancellation_requested = cancel_rx.try_recv().is_ok();
    let summary = if cancellation_requested {
        build_runtime_summary(
            &run_id,
            &child_session_id,
            &agent_name,
            SubagentStatus::Cancelled,
            format!("{} cancelled", agent_name),
        )
    } else {
        let execution_child_transcript = Arc::clone(&child_transcript);
        let execution_event_sender = event_sender.clone();
        let execution_child_session_id = child_session_id.clone();
        let execution_agent_name = agent_name.clone();
        let mut exec = Some(exec);
        let execution = async move {
            exec.take()
                .expect("subagent executor polled more than once")(
                child_agent,
                task,
                execution_child_transcript,
                execution_event_sender,
                execution_child_session_id,
                execution_agent_name,
            )
            .await
        };
        tokio::select! {
        biased;
        _ = &mut cancel_rx => build_runtime_summary(
            &run_id,
            &child_session_id,
            &agent_name,
            SubagentStatus::Cancelled,
            format!("{} cancelled", agent_name),
        ),
        result = async {
            match timeout_secs {
                Some(timeout_secs) => match timeout_at(deadline.unwrap_or_else(|| Instant::now() + Duration::from_secs(timeout_secs)), async {
                    if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                        std::future::pending::<()>().await;
                    }
                    execution.await
                }).await {
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
        }
    };

    let mut summary = summary;
    let observed_changed_paths = observed_changed_paths_from_child_transcript(&child_transcript);
    enforce_write_scope(
        &mut summary,
        &governance,
        effective_scope.as_ref(),
        &observed_changed_paths,
    );
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
    settle_pending_messages(&guard, &mut summary);

    if let Err(error) = record_child_completion(
        &child_transcript,
        &summary,
        &parent_session_id,
        &parent_turn_id,
    ) {
        let message = format!(
            "{}; failed to record child subagent completion: {error}",
            summary.summary
        );
        emit_error(&event_sender, message.clone());
        set_hard_failure(&mut summary, message);
        settle_pending_messages(&guard, &mut summary);
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
        settle_pending_messages(&guard, &mut summary);
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
    let runtime = guard.runtime.clone();
    let message_root = runtime.state.lock().ok().and_then(|state| {
        state
            .active_by_run
            .get(&run_id)
            .and_then(|slot| slot.swarm_root.clone())
    });
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
    drop(guard);
    drop(child_transcript);
    if let Some(root) = message_root {
        runtime.release_message_owner(&root, &child_session_id);
    }
    Ok(summary)
}

fn settle_pending_messages(guard: &ActiveRunGuard, summary: &mut SubagentRunSummary) {
    let count = guard.runtime.settle_message_run(&guard.run_id, summary);
    if count > 0 {
        let detail = format!(
            "{count} queued message(s) failed after {}",
            summary.status.as_str()
        );
        summary.summary = format!("{}; {detail}", summary.summary);
        summary.structured_result.summary = summary.summary.clone();
        summary.structured_result.blockers.push(detail);
    }
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
    effective_scope: Option<&SubagentPathScope>,
    observed_changed_paths: &[String],
) {
    if !matches!(summary.agent_name.as_str(), "fixer" | "general")
        || !governance.input.has_write_scope()
    {
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
        .filter(|path| match effective_scope {
            Some(scope) => crate::tool::canonical_subagent_observed_path(path)
                .map(|path| !scope.permits_write([path.as_path()]))
                .unwrap_or(true),
            None => !governance.input.permits_write_path(path),
        })
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

#[cfg(test)]
#[path = "../../tests/internal/subagent/pool/tests.rs"]
mod tests;
