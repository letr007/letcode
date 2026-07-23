use anyhow::{Result, anyhow};
use async_openai::config::Config;
use futures_util::FutureExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::oneshot;
use tokio::time::{Duration, timeout};

use crate::agent::{
    Agent, AgentFactory, AgentTemplate, ConversationMessage, ConversationRole, SubagentInvocation,
};
use crate::request_builder::HistoryItem;
use crate::subagent_events::{SubagentEventSender, emit_error, emit_status, run_child_prompt};
use crate::tool::NormalizedSubagentInput;
use crate::transcript::{
    ChildSessionSummary, TranscriptEvent, TranscriptRecorder, child_sessions_dir,
    list_child_sessions_for_parent, read_records_allow_partial_tail, sort_child_session_summaries,
};
use crate::transcript::transcript_projection;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubagentStatus {
    Running,
    Completed,
    Failed,
    BudgetExhausted,
    Cancelled,
    TimedOut,
}

impl SubagentStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::BudgetExhausted => "budget_exhausted",
            Self::Cancelled => "cancelled",
            Self::TimedOut => "timed_out",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubagentRunSummary {
    pub run_id: String,
    pub child_session_id: String,
    pub agent_name: String,
    pub status: SubagentStatus,
    pub summary: String,
    pub structured_result: StructuredSubagentResult,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StructuredSubagentResult {
    pub status: String,
    pub summary: String,
    pub malformed: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub findings: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub files_read: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub files_changed: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub commands_run: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub validation: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub blockers: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub next_steps: Vec<String>,
    pub run_id: String,
    pub child_session_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_excerpt: Option<String>,
}

impl StructuredSubagentResult {
    fn from_model_output(
        raw: &str,
        fallback_status: SubagentStatus,
        run_id: &str,
        child_session_id: &str,
    ) -> Self {
        let candidate = extract_json_candidate(raw).unwrap_or(raw.trim());
        match serde_json::from_str::<Value>(candidate) {
            Ok(Value::Object(map)) => {
                let value = Value::Object(map);
                let findings = list_field(&value, "findings");
                let summary = string_field(&value, "summary")
                    .filter(|text| !text.is_empty())
                    .or_else(|| findings.first().cloned())
                    .unwrap_or_else(|| excerpt(raw));
                Self {
                    status: string_field(&value, "status")
                        .filter(|text| !text.is_empty())
                        .unwrap_or_else(|| fallback_status.as_str().to_string()),
                    summary,
                    malformed: false,
                    findings,
                    files_read: list_field(&value, "files_read"),
                    files_changed: list_field(&value, "files_changed"),
                    commands_run: list_field(&value, "commands_run"),
                    validation: validation_field(&value),
                    blockers: list_field(&value, "blockers"),
                    next_steps: list_field(&value, "next_steps"),
                    run_id: run_id.to_string(),
                    child_session_id: child_session_id.to_string(),
                    raw_excerpt: None,
                }
            }
            _ => Self {
                status: fallback_status.as_str().to_string(),
                summary: excerpt(raw),
                malformed: true,
                findings: Vec::new(),
                files_read: Vec::new(),
                files_changed: Vec::new(),
                commands_run: Vec::new(),
                validation: Vec::new(),
                blockers: Vec::new(),
                next_steps: Vec::new(),
                run_id: run_id.to_string(),
                child_session_id: child_session_id.to_string(),
                raw_excerpt: Some(excerpt(raw)),
            },
        }
    }

    fn from_runtime_status(
        status: SubagentStatus,
        summary: String,
        run_id: &str,
        child_session_id: &str,
    ) -> Self {
        Self {
            status: status.as_str().to_string(),
            blockers: matches!(
                status,
                SubagentStatus::Failed
                    | SubagentStatus::BudgetExhausted
                    | SubagentStatus::Cancelled
                    | SubagentStatus::TimedOut
            )
            .then(|| vec![summary.clone()])
            .unwrap_or_default(),
            summary,
            malformed: false,
            findings: Vec::new(),
            files_read: Vec::new(),
            files_changed: Vec::new(),
            commands_run: Vec::new(),
            validation: Vec::new(),
            next_steps: Vec::new(),
            run_id: run_id.to_string(),
            child_session_id: child_session_id.to_string(),
            raw_excerpt: None,
        }
    }
}

type BoxExecFuture = Pin<Box<dyn Future<Output = Result<String>> + Send>>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubagentRunGovernance {
    pub timeout_secs: Option<u64>,
    pub max_tool_calls: Option<usize>,
    pub input: NormalizedSubagentInput,
}

impl SubagentRunGovernance {
    fn from_template_and_input(template: &AgentTemplate, input: NormalizedSubagentInput) -> Self {
        Self {
            timeout_secs: input.effective_timeout_secs(template.timeout_secs),
            max_tool_calls: input.effective_max_tool_calls(template.max_tool_calls),
            input,
        }
    }
}

#[derive(Clone)]
pub struct SubagentPool {
    /// agent_name -> active child (one running slot per role).
    active_by_agent: Arc<Mutex<std::collections::HashMap<String, ActiveSlot>>>,
    /// Monotonic ordinal issuer for stable TUI pool numbers (1-based).
    next_ordinal: Arc<Mutex<u32>>,
}

#[derive(Debug)]
struct ActiveSlot {
    cancel: Option<oneshot::Sender<()>>,
    child: ChildSessionSummary,
    run_id: String,
}

struct ActiveRunGuard {
    active_by_agent: Arc<Mutex<std::collections::HashMap<String, ActiveSlot>>>,
    agent_name: String,
    run_id: String,
}

impl ActiveRunGuard {
    fn new(
        active_by_agent: Arc<Mutex<std::collections::HashMap<String, ActiveSlot>>>,
        agent_name: String,
        run_id: String,
    ) -> Self {
        Self {
            active_by_agent,
            agent_name,
            run_id,
        }
    }

    fn clear_cancel(&self) {
        if let Ok(mut map) = self.active_by_agent.lock() {
            if let Some(slot) = map.get_mut(&self.agent_name) {
                if slot.run_id == self.run_id {
                    slot.cancel.take();
                }
            }
        }
    }
}

impl Drop for ActiveRunGuard {
    fn drop(&mut self) {
        if let Ok(mut map) = self.active_by_agent.lock() {
            if let Some(slot) = map.get(&self.agent_name) {
                if slot.run_id == self.run_id {
                    map.remove(&self.agent_name);
                }
            }
        }
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
            active_by_agent: Arc::new(Mutex::new(std::collections::HashMap::new())),
            next_ordinal: Arc::new(Mutex::new(1)),
        }
    }

    pub fn cancel_active(&self) -> bool {
        let Ok(mut map) = self.active_by_agent.lock() else {
            return false;
        };
        let mut cancelled = false;
        for slot in map.values_mut() {
            if let Some(tx) = slot.cancel.take() {
                let _ = tx.send(());
                cancelled = true;
            }
        }
        cancelled
    }

    pub fn cancel_agent(&self, agent_name: &str) -> bool {
        let Ok(mut map) = self.active_by_agent.lock() else {
            return false;
        };
        let Some(slot) = map.get_mut(agent_name) else {
            return false;
        };
        if let Some(tx) = slot.cancel.take() {
            let _ = tx.send(());
            true
        } else {
            false
        }
    }

    pub fn is_running(&self) -> bool {
        self.active_by_agent
            .lock()
            .map(|map| !map.is_empty())
            .unwrap_or(false)
    }

    pub fn is_agent_running(&self, agent_name: &str) -> bool {
        self.active_by_agent
            .lock()
            .map(|map| map.contains_key(agent_name))
            .unwrap_or(false)
    }

    pub fn active_child(&self) -> Option<ChildSessionSummary> {
        let map = self.active_by_agent.lock().ok()?;
        map.values()
            .map(|slot| slot.child.clone())
            .min_by_key(|child| (child.pool_ordinal, child.child_session_id.clone()))
    }

    pub fn active_children(&self) -> Vec<ChildSessionSummary> {
        let Ok(map) = self.active_by_agent.lock() else {
            return Vec::new();
        };
        let mut children: Vec<_> = map.values().map(|slot| slot.child.clone()).collect();
        sort_child_session_summaries(&mut children);
        children
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
        let mut next = self
            .next_ordinal
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if *next <= max_existing {
            *next = max_existing.saturating_add(1);
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
        let template = AgentTemplate::from_name(agent_name)
            .ok_or_else(|| anyhow!("unknown subagent template: {agent_name}"))?;
        let governance =
            SubagentRunGovernance::from_template_and_input(&template, invocation.input.clone());
        self.run_with_executor(
            parent,
            template,
            invocation.prompt,
            governance,
            sessions_dir.as_ref().to_path_buf(),
            parent_session_id,
            parent_turn_id,
            parent_transcript,
            event_sender,
            invocation.input.target_child_session_id.clone(),
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
        {
            let busy = self
                .active_by_agent
                .lock()
                .map_err(|_| anyhow!("subagent pool lock poisoned"))?
                .contains_key(template.name.as_str());
            if busy {
                return Err(anyhow!(self.busy_error_message_for(template.name.as_str())));
            }
        }

        let effective_timeout_secs = governance.timeout_secs.or(template.timeout_secs);
        let effective_max_tool_calls = governance.max_tool_calls.or(template.max_tool_calls);

        let mut child_agent = AgentFactory::create_child_with_max_tool_calls(
            parent,
            &template,
            effective_max_tool_calls,
        );

        let existing_children = {
            let parent_records = parent_transcript
                .as_ref()
                .and_then(|recorder| recorder.lock().ok())
                .and_then(|recorder| {
                    read_records_allow_partial_tail(recorder.path()).ok()
                })
                .unwrap_or_default();
            Self::child_sessions(&sessions_dir, &parent_records)
        };

        let setup = (|| -> Result<(String, String, u32, Arc<Mutex<TranscriptRecorder>>)> {
            let run_id = generate_run_id();
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
                child_agent.set_context_scope_state(child_recorder.context_scope_state());
                let child_records =
                    read_records_allow_partial_tail(child_recorder.path()).unwrap_or_default();
                let snapshot = transcript_projection::project_runtime_restore_snapshot(
                    target_id.clone(),
                    child_records,
                    transcript_projection::SessionContextCursor {
                        branch_id: None,
                        leaf_sequence: None,
                    },
                    &[],
                )?;
                let restored =
                    restored_messages_from_protocol_frames(&snapshot.protocol_frames);
                if !restored.is_empty() {
                    child_agent.restore_transcript_messages(restored);
                }

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

                Ok((
                    run_id,
                    target_id.clone(),
                    pool_ordinal,
                    Arc::new(Mutex::new(child_recorder)),
                ))
            } else {
                let pool_ordinal = self.allocate_ordinal_from_children(&existing_children);
                let mut child_recorder = TranscriptRecorder::create(&child_dir)?;
                child_agent.set_context_scope_state(child_recorder.context_scope_state());
                child_recorder.record_session_started(child_agent.model().to_string())?;
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
                ))
            }
        })();

        let (run_id, child_session_id, pool_ordinal, child_transcript) = match setup {
            Ok(values) => values,
            Err(error) => return Err(error),
        };

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
            return Err(error);
        }

        emit_status(
            &event_sender,
            format!(
                "{} running · #{} · run {}",
                template.name, pool_ordinal, run_id
            ),
        );

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

        {
            let mut map = self
                .active_by_agent
                .lock()
                .map_err(|_| anyhow!("subagent pool lock poisoned"))?;
            if map.contains_key(template.name.as_str()) {
                drop(map);
                return Err(anyhow!(self.busy_error_message_for(template.name.as_str())));
            }
            map.insert(
                template.name.clone(),
                ActiveSlot {
                    cancel: Some(cancel_tx),
                    child: summary,
                    run_id: run_id.clone(),
                },
            );
        }

        Ok(StartedRun {
            guard: ActiveRunGuard::new(
                Arc::clone(&self.active_by_agent),
                template.name.clone(),
                run_id.clone(),
            ),
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

    fn busy_error_message_for(&self, agent_name: &str) -> String {
        if let Ok(map) = self.active_by_agent.lock() {
            if let Some(slot) = map.get(agent_name) {
                return format!(
                    "subagent role `{agent_name}` is busy: only one active run per role is allowed; wait for completion or cancel (run_id={}, child_session_id={}, pool_ordinal={})",
                    slot.run_id, slot.child.child_session_id, slot.child.pool_ordinal
                );
            }
        }
        format!(
            "subagent role `{agent_name}` is busy: only one active run per role is allowed; wait for completion or cancel"
        )
    }

    fn busy_error_message(&self) -> String {
        let children = self.active_children();
        if children.is_empty() {
            return "subagent pool is busy: wait for completion or cancel the active child".into();
        }
        let detail = children
            .into_iter()
            .map(|child| {
                format!(
                    "{}#{}:{}",
                    child.agent_name, child.pool_ordinal, child.child_session_id
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "subagent pool has active children [{detail}]; per-role single-active is enforced and queueing is unsupported"
        )
    }

}

struct StartedRun<C: Config> {
    guard: ActiveRunGuard,
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
        guard,
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

    guard.clear_cancel();

    if let Err(error) = record_child_completion(
        &child_transcript,
        &summary,
        &parent_session_id,
        &parent_turn_id,
    ) {
        emit_error(
            &event_sender,
            format!("failed to record child subagent completion: {error}"),
        );
    }

    let parent_record_result = record_parent_result(
        &parent_transcript,
        &summary,
        &parent_session_id,
        &parent_turn_id,
    );
    if let Err(error) = parent_record_result {
        emit_error(
            &event_sender,
            format!("failed to record parent subagent result: {error}"),
        );
        return Err(error);
    }
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
    Ok(summary)
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

fn restored_messages_from_protocol_frames(
    protocol_frames: &[crate::protocol_frames::ProtocolFrame],
) -> Vec<ConversationMessage> {
    crate::protocol_frames::history_items_from_frames(protocol_frames)
        .into_iter()
        .filter_map(|item| match item {
            HistoryItem::ContextSummary { text } => Some(ConversationMessage {
                role: ConversationRole::Summary,
                content: text,
            }),
            HistoryItem::UserMessage { content } => Some(ConversationMessage {
                role: ConversationRole::User,
                content: content.display_text(),
            }),
            HistoryItem::InternalContinuation { text } => Some(ConversationMessage {
                role: ConversationRole::User,
                content: text,
            }),
            HistoryItem::AssistantText { text } => Some(ConversationMessage {
                role: ConversationRole::Assistant,
                content: text,
            }),
            HistoryItem::AssistantToolCalls { text, .. } => text.map(|content| ConversationMessage {
                role: ConversationRole::Assistant,
                content,
            }),
            HistoryItem::ToolOutput { .. } => None,
        })
        .collect()
}

fn short_session_id(session_id: &str) -> &str {
    session_id.get(..12).unwrap_or(session_id)
}

fn generate_run_id() -> String {
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

fn build_completed_summary(
    run_id: &str,
    child_session_id: &str,
    agent_name: &str,
    message: String,
) -> SubagentRunSummary {
    let structured_result = StructuredSubagentResult::from_model_output(
        &message,
        SubagentStatus::Completed,
        run_id,
        child_session_id,
    );
    let status = map_structured_status(&structured_result.status);
    SubagentRunSummary {
        run_id: run_id.to_string(),
        child_session_id: child_session_id.to_string(),
        agent_name: agent_name.to_string(),
        status,
        summary: structured_result.summary.clone(),
        structured_result,
    }
}

fn build_runtime_summary(
    run_id: &str,
    child_session_id: &str,
    agent_name: &str,
    status: SubagentStatus,
    summary: String,
) -> SubagentRunSummary {
    let structured_result = StructuredSubagentResult::from_runtime_status(
        status,
        summary.clone(),
        run_id,
        child_session_id,
    );
    SubagentRunSummary {
        run_id: run_id.to_string(),
        child_session_id: child_session_id.to_string(),
        agent_name: agent_name.to_string(),
        status,
        summary,
        structured_result,
    }
}

fn extract_json_candidate(raw: &str) -> Option<&str> {
    let trimmed = raw.trim();
    if trimmed.starts_with("```") {
        let body = trimmed
            .trim_start_matches("```")
            .trim_start_matches("json")
            .trim();
        return body.strip_suffix("```").map(str::trim);
    }
    if trimmed.starts_with('{') && trimmed.ends_with('}') {
        return Some(trimmed);
    }
    let start = trimmed.find('{')?;
    let end = trimmed.rfind('}')?;
    (end > start).then_some(trimmed[start..=end].trim())
}

fn string_field(value: &Value, key: &str) -> Option<String> {
    match value.get(key) {
        Some(Value::String(text)) => Some(text.trim().to_string()).filter(|text| !text.is_empty()),
        Some(Value::Number(number)) => Some(number.to_string()),
        Some(Value::Bool(flag)) => Some(flag.to_string()),
        _ => None,
    }
}

fn list_field(value: &Value, key: &str) -> Vec<String> {
    match value.get(key) {
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(|item| match item {
                Value::String(text) => {
                    Some(text.trim().to_string()).filter(|text| !text.is_empty())
                }
                Value::Number(number) => Some(number.to_string()),
                Value::Bool(flag) => Some(flag.to_string()),
                Value::Object(map) => map
                    .get("path")
                    .or_else(|| map.get("file"))
                    .or_else(|| map.get("command"))
                    .or_else(|| map.get("summary"))
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .map(str::to_string)
                    .filter(|text| !text.is_empty()),
                _ => None,
            })
            .collect(),
        Some(Value::String(text)) => {
            let trimmed = text.trim();
            if trimmed.is_empty() {
                Vec::new()
            } else {
                vec![trimmed.to_string()]
            }
        }
        _ => Vec::new(),
    }
}

fn validation_field(value: &Value) -> Vec<String> {
    match value.get("validation") {
        Some(Value::Array(items)) => items.iter().filter_map(validation_item_summary).collect(),
        Some(Value::String(text)) => {
            let trimmed = text.trim();
            if trimmed.is_empty() {
                Vec::new()
            } else {
                vec![trimmed.to_string()]
            }
        }
        _ => Vec::new(),
    }
}

fn validation_item_summary(item: &Value) -> Option<String> {
    match item {
        Value::String(text) => Some(text.trim().to_string()).filter(|text| !text.is_empty()),
        Value::Object(map) => {
            let command = map
                .get("command")
                .or_else(|| map.get("name"))
                .or_else(|| map.get("summary"))
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|text| !text.is_empty())
                .unwrap_or("validation");
            let outcome = map
                .get("status")
                .or_else(|| map.get("result"))
                .or_else(|| map.get("outcome"))
                .or_else(|| map.get("state"))
                .and_then(|value| match value {
                    Value::String(text) => Some(text.trim().to_string()),
                    Value::Bool(flag) => Some(if *flag { "passed" } else { "failed" }.into()),
                    Value::Number(number) => Some(number.to_string()),
                    _ => None,
                });
            Some(match outcome {
                Some(outcome) if !outcome.is_empty() => format!("{command} {outcome}"),
                _ => command.to_string(),
            })
        }
        Value::Bool(flag) => Some(if *flag { "passed" } else { "failed" }.into()),
        Value::Number(number) => Some(number.to_string()),
        _ => None,
    }
}

fn excerpt(raw: &str) -> String {
    let compact = raw.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut chars = compact.chars();
    let excerpt = chars.by_ref().take(240).collect::<String>();
    if chars.next().is_some() {
        format!("{excerpt}…")
    } else if excerpt.is_empty() {
        "subagent produced empty output".into()
    } else {
        excerpt
    }
}

fn legacy_task_input(task: String) -> NormalizedSubagentInput {
    NormalizedSubagentInput {
        objective: task,
        success_criteria: Vec::new(),
        allowed_paths: Vec::new(),
        forbidden_paths: Vec::new(),
        owned_paths: Vec::new(),
        timeout_secs: None,
        max_tool_calls: None,
        target_child_session_id: None,
    }
}

fn classify_failure_status(message: &str) -> SubagentStatus {
    if message.contains("stopped: too many tool calls") {
        SubagentStatus::BudgetExhausted
    } else {
        SubagentStatus::Failed
    }
}

fn map_structured_status(status: &str) -> SubagentStatus {
    match status.trim().to_ascii_lowercase().as_str() {
        "completed" | "succeeded" | "success" => SubagentStatus::Completed,
        "cancelled" | "canceled" => SubagentStatus::Cancelled,
        "timed_out" | "timed out" | "timeout" => SubagentStatus::TimedOut,
        "budget_exhausted" | "budget exhausted" => SubagentStatus::BudgetExhausted,
        "failed" | "error" | "blocked" => SubagentStatus::Failed,
        _ => SubagentStatus::Completed,
    }
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
mod tests {
    use super::*;
    use crate::transcript::read_records;
    use crate::tui::RunnerEvent;
    use async_openai::Client;
    use async_openai::config::OpenAIConfig;
    use std::sync::atomic::AtomicU64;
    use tokio::sync::Barrier;
    use tokio::time::sleep;

    static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);

    fn test_agent() -> Agent<OpenAIConfig> {
        Agent::new(Client::with_config(OpenAIConfig::new()), "gpt-test", 2, 4)
    }

    fn temp_sessions_dir() -> PathBuf {
        let id = NEXT_TEMP_ID.fetch_add(1, Ordering::SeqCst);
        std::env::temp_dir().join(format!("{}-{id}", generate_run_id()))
    }

    fn test_governance() -> SubagentRunGovernance {
        SubagentRunGovernance {
            timeout_secs: None,
            max_tool_calls: None,
            input: NormalizedSubagentInput {
                objective: "test".into(),
                success_criteria: Vec::new(),
                allowed_paths: Vec::new(),
                forbidden_paths: Vec::new(),
                owned_paths: Vec::new(),
                timeout_secs: None,
                max_tool_calls: None,
        target_child_session_id: None,
            },
        }
    }

    async fn wait_until<F>(mut condition: F)
    where
        F: FnMut() -> bool,
    {
        for _ in 0..50 {
            if condition() {
                return;
            }
            sleep(Duration::from_millis(10)).await;
        }
        assert!(condition(), "condition was not met before timeout");
    }

    #[test]
    fn agent_factory_creates_scoped_child_without_changing_default_scope() {
        let agent = test_agent();
        assert_eq!(agent.tool_scope().as_str(), "full_access");

        let child = AgentFactory::create_child(&agent, &AgentTemplate::explorer());
        assert_eq!(child.tool_scope().as_str(), "read_only_explorer");
        assert_eq!(child.permission_mode().as_str(), "default");
        assert_eq!(child.model(), agent.model());
    }

    #[test]
    fn child_agents_do_not_expose_recursive_subagent_tools() {
        let agent = test_agent();
        let child = AgentFactory::create_child(&agent, &AgentTemplate::fixer());
        let tool_names = child
            .tool_definitions_for_test()
            .into_iter()
            .map(|spec| spec.name)
            .collect::<Vec<_>>();

        assert!(!tool_names.iter().any(|name| name == "agent__explore"));
        assert!(!tool_names.iter().any(|name| name == "agent__fixer"));
    }

    #[test]
    fn agent_factory_uses_configured_subagent_model_override() {
        let mut agent = test_agent();
        agent.set_subagent_model_override("explorer", "gpt-explorer");
        agent.set_subagent_model_override("fixer", "gpt-fixer");

        let explorer = AgentFactory::create_child(&agent, &AgentTemplate::explorer());
        let fixer = AgentFactory::create_child(&agent, &AgentTemplate::fixer());

        assert_eq!(explorer.model(), "gpt-explorer");
        assert_eq!(fixer.model(), "gpt-fixer");
    }

    #[test]
    fn generate_run_id_is_unique_within_process() {
        let first = generate_run_id();
        let second = generate_run_id();

        assert_ne!(first, second);
    }

    #[test]
    fn explorer_child_shares_parent_permission_mode() {
        let mut agent = test_agent();
        agent.set_permission_mode(crate::permission::PermissionMode::Safe);
        let child = AgentFactory::create_child(&agent, &AgentTemplate::explorer());

        assert_eq!(agent.permission_mode().as_str(), "safe");
        assert_eq!(child.permission_mode().as_str(), "safe");
    }

    #[test]
    fn structured_result_parser_accepts_json_object_output() {
        let result = StructuredSubagentResult::from_model_output(
            r#"{"status":"completed","summary":"done","findings":["a"],"files_read":["src/agent.rs"],"files_changed":["src/subagent.rs"],"commands_run":["cargo test"],"validation":["passed"],"blockers":[],"next_steps":["report"]}"#,
            SubagentStatus::Completed,
            "run-1",
            "child-1",
        );

        assert!(!result.malformed);
        assert_eq!(result.status, "completed");
        assert_eq!(result.summary, "done");
        assert_eq!(result.files_read, vec!["src/agent.rs"]);
        assert_eq!(result.files_changed, vec!["src/subagent.rs"]);
        assert_eq!(result.commands_run, vec!["cargo test"]);
        assert_eq!(result.validation, vec!["passed"]);
        assert_eq!(result.run_id, "run-1");
        assert_eq!(result.child_session_id, "child-1");
    }

    #[test]
    fn structured_result_parser_marks_non_json_output_as_malformed() {
        let result = StructuredSubagentResult::from_model_output(
            "completed after inspecting src/subagent.rs",
            SubagentStatus::Completed,
            "run-1",
            "child-1",
        );

        assert!(result.malformed);
        assert_eq!(result.status, "completed");
        assert!(result.summary.contains("completed after inspecting"));
        assert_eq!(result.raw_excerpt.as_deref(), Some(result.summary.as_str()));
    }

    #[test]
    fn structured_result_parser_preserves_object_shaped_validation_outcomes() {
        let result = StructuredSubagentResult::from_model_output(
            r#"{"status":"completed","summary":"done","validation":[{"command":"cargo test","result":"failed"},{"command":"cargo fmt","result":"not_run"}]}"#,
            SubagentStatus::Completed,
            "run-1",
            "child-1",
        );

        assert_eq!(
            result.validation,
            vec!["cargo test failed", "cargo fmt not_run"]
        );
    }

    #[test]
    fn runtime_failure_summary_is_not_marked_malformed() {
        let summary = build_runtime_summary(
            "run-1",
            "child-1",
            "fixer",
            SubagentStatus::TimedOut,
            "fixer timed out after 10s".into(),
        );

        assert_eq!(summary.status, SubagentStatus::TimedOut);
        assert!(!summary.structured_result.malformed);
        assert_eq!(summary.structured_result.status, "timed_out");
        assert_eq!(
            summary.structured_result.blockers,
            vec!["fixer timed out after 10s"]
        );
    }

    #[test]
    fn tool_call_budget_failures_are_promoted_to_budget_exhausted_status() {
        let summary = build_runtime_summary(
            "run-1",
            "child-1",
            "fixer",
            classify_failure_status("stopped: too many tool calls (2 requested, max 1)"),
            "stopped: too many tool calls (2 requested, max 1)".into(),
        );

        assert_eq!(summary.status, SubagentStatus::BudgetExhausted);
        assert_eq!(summary.structured_result.status, "budget_exhausted");
        assert!(summary.summary.contains("too many tool calls"));
    }

    #[test]
    fn model_reported_non_success_status_changes_outer_run_status() {
        let summary = build_completed_summary(
            "run-2",
            "child-2",
            "fixer",
            r#"{"status":"cancelled","summary":"user cancelled"}"#.into(),
        );

        assert_eq!(summary.status, SubagentStatus::Cancelled);
        assert_eq!(summary.structured_result.status, "cancelled");
    }

    #[tokio::test]
    async fn fixer_out_of_scope_changes_are_visible_and_fail_the_run() {
        let runtime = SubagentPool::new();
        let mut governance = test_governance();
        governance.input.allowed_paths = vec!["src/owned.rs".into()];
        governance.input.owned_paths = vec!["src/owned.rs".into()];

        let summary = runtime
            .run_with_executor(
                &test_agent(),
                AgentTemplate::fixer(),
                "apply fix".into(),
                governance,
                temp_sessions_dir(),
                "parent-session".into(),
                "turn-1".into(),
                None,
                None,
                |_agent, _task, _transcript, _runner_tx, _child_session_id, _agent_name| {
                    async move {
                        Ok(
                            r#"{"status":"completed","summary":"changed files","files_changed":["src/outside.rs"]}"#
                                .into(),
                        )
                    }
                    .boxed()
                },
            None,
            )
            .await
            .expect("run returns governed summary");

        assert_eq!(summary.status, SubagentStatus::Failed);
        assert_eq!(summary.structured_result.status, "failed");
        assert!(summary.summary.contains("out-of-scope changes detected"));
        assert!(
            summary
                .structured_result
                .blockers
                .iter()
                .any(|blocker| blocker.contains("src/outside.rs"))
        );
    }

    #[tokio::test]
    async fn observed_child_write_effects_enforce_scope_even_when_files_changed_missing() {
        let runtime = SubagentPool::new();
        let mut governance = test_governance();
        governance.input.allowed_paths = vec!["src/owned.rs".into()];

        let summary = runtime
            .run_with_executor(
                &test_agent(),
                AgentTemplate::fixer(),
                "apply fix".into(),
                governance,
                temp_sessions_dir(),
                "parent-session".into(),
                "turn-1".into(),
                None,
                None,
                |_agent, _task, transcript, _runner_tx, _child_session_id, _agent_name| {
                    async move {
                        transcript
                            .lock()
                            .expect("lock child transcript")
                            .record_tool_call_finished(
                                "call-1",
                                "fs__write",
                                true,
                                crate::tool::ToolResult::ok(
                                    "fs__write",
                                    serde_json::json!({"path":"src/outside.rs"}),
                                ),
                            )
                            .expect("record child write");
                        Ok(r#"{"status":"completed","summary":"done"}"#.into())
                    }
                    .boxed()
                },
            None,
            )
            .await
            .expect("run returns summary");

        assert_eq!(summary.status, SubagentStatus::Failed);
        assert!(summary.summary.contains("src/outside.rs"));
    }

    #[tokio::test]
    async fn governance_timeout_overrides_template_default() {
        let runtime = SubagentPool::new();
        let mut governance = test_governance();
        governance.timeout_secs = Some(1);

        let summary = runtime
            .run_with_executor(
                &test_agent(),
                AgentTemplate::fixer(),
                "apply fix".into(),
                governance,
                temp_sessions_dir(),
                "parent-session".into(),
                "turn-1".into(),
                None,
                None,
                |_agent, _task, _transcript, _runner_tx, _child_session_id, _agent_name| {
                    async move {
                        tokio::time::sleep(Duration::from_secs(60)).await;
                        Ok("late".into())
                    }
                    .boxed()
                },
            None,
            )
            .await
            .expect("timeout returns summary");

        assert_eq!(summary.status, SubagentStatus::TimedOut);
        assert_eq!(summary.structured_result.status, "timed_out");
    }

    #[tokio::test]
    async fn max_concurrency_guard_rejects_second_run() {
        let runtime = SubagentPool::new();
        let agent = test_agent();
        let sessions_dir = temp_sessions_dir();
        let barrier = Arc::new(Barrier::new(2));

        let first_runtime = runtime.clone();
        let first_barrier = Arc::clone(&barrier);
        let first = tokio::spawn(async move {
            first_runtime
                .run_with_executor(
                    &agent,
                    AgentTemplate::explorer(),
                    "inspect".into(),
                    test_governance(),
                    sessions_dir,
                    "parent-session".into(),
                    "turn-1".into(),
                    None,
                    None,
                    move |_agent,
                          _task,
                          _transcript,
                          _runner_tx,
                          _child_session_id,
                          _agent_name| {
                        async move {
                            first_barrier.wait().await;
                            tokio::time::sleep(Duration::from_millis(50)).await;
                            Ok("done".into())
                        }
                        .boxed()
                    },
            None,
                )
                .await
        });

        barrier.wait().await;

        let second = runtime
            .run_with_executor(
                &test_agent(),
                AgentTemplate::explorer(),
                "inspect again".into(),
                test_governance(),
                temp_sessions_dir(),
                "parent-session".into(),
                "turn-2".into(),
                None,
                None,
                |_agent, _task, _transcript, _runner_tx, _child_session_id, _agent_name| {
                    async move { Ok("done".into()) }.boxed()
                },
            None,
            )
            .await
            .expect_err("second run should be rejected");

        let error = second.to_string();
        assert!(error.contains("single-slot busy"), "{error}");
        assert!(
            error.contains("parallel execution and queueing are unsupported"),
            "{error}"
        );
        assert!(error.contains("agent=explorer"), "{error}");
        assert!(error.contains("run_id="), "{error}");
        assert!(error.contains("child_session_id="), "{error}");
        let first_summary = first.await.expect("join first").expect("first ok");
        assert_eq!(first_summary.status, SubagentStatus::Completed);

        let next = runtime
            .run_with_executor(
                &test_agent(),
                AgentTemplate::explorer(),
                "inspect after completion".into(),
                test_governance(),
                temp_sessions_dir(),
                "parent-session".into(),
                "turn-3".into(),
                None,
                None,
                |_agent, _task, _transcript, _runner_tx, _child_session_id, _agent_name| {
                    async move { Ok("done".into()) }.boxed()
                },
            None,
            )
            .await
            .expect("slot is reusable after completion");
        assert_eq!(next.status, SubagentStatus::Completed);
    }

    #[tokio::test]
    async fn cancel_active_records_cancelled_and_releases_guard() {
        let runtime = SubagentPool::new();
        let agent = test_agent();
        let sessions_dir = temp_sessions_dir();
        let barrier = Arc::new(Barrier::new(2));

        let run_runtime = runtime.clone();
        let run_barrier = Arc::clone(&barrier);
        let run = tokio::spawn(async move {
            run_runtime
                .run_with_executor(
                    &agent,
                    AgentTemplate::explorer(),
                    "inspect".into(),
                    test_governance(),
                    sessions_dir,
                    "parent-session".into(),
                    "turn-1".into(),
                    None,
                    None,
                    move |_agent,
                          _task,
                          _transcript,
                          _runner_tx,
                          _child_session_id,
                          _agent_name| {
                        async move {
                            run_barrier.wait().await;
                            std::future::pending::<Result<String>>().await
                        }
                        .boxed()
                    },
            None,
                )
                .await
        });

        barrier.wait().await;
        assert!(runtime.cancel_active());

        let summary = run.await.expect("join run").expect("run summary");
        assert_eq!(summary.status, SubagentStatus::Cancelled);

        let next = runtime
            .run_with_executor(
                &test_agent(),
                AgentTemplate::explorer(),
                "inspect again".into(),
                test_governance(),
                temp_sessions_dir(),
                "parent-session".into(),
                "turn-2".into(),
                None,
                None,
                |_agent, _task, _transcript, _runner_tx, _child_session_id, _agent_name| {
                    async move { Ok("done".into()) }.boxed()
                },
            None,
            )
            .await
            .expect("second run succeeds after cancellation");
        assert_eq!(next.status, SubagentStatus::Completed);
    }

    #[tokio::test]
    async fn active_child_uses_running_status_while_subagent_is_active() {
        let runtime = SubagentPool::new();
        let agent = test_agent();
        let sessions_dir = temp_sessions_dir();
        let barrier = Arc::new(Barrier::new(2));

        let run_runtime = runtime.clone();
        let run_barrier = Arc::clone(&barrier);
        let run = tokio::spawn(async move {
            run_runtime
                .run_with_executor(
                    &agent,
                    AgentTemplate::explorer(),
                    "inspect src".into(),
                    test_governance(),
                    sessions_dir,
                    "parent-session".into(),
                    "turn-1".into(),
                    None,
                    None,
                    move |_agent,
                          _task,
                          _transcript,
                          _runner_tx,
                          _child_session_id,
                          _agent_name| {
                        async move {
                            run_barrier.wait().await;
                            tokio::time::sleep(Duration::from_millis(50)).await;
                            Ok("done".into())
                        }
                        .boxed()
                    },
            None,
                )
                .await
        });

        wait_until(|| runtime.active_child().is_some()).await;
        let active_child = runtime.active_child().expect("active child available");
        assert_eq!(active_child.status, "running");
        assert_eq!(active_child.summary, "inspect src");

        barrier.wait().await;
        let _ = run.await.expect("join run").expect("run summary");
        assert!(runtime.active_child().is_none());
    }

    #[tokio::test]
    async fn parent_transcript_records_running_lifecycle_and_terminal_result_only() {
        let runtime = SubagentPool::new();
        let agent = test_agent();
        let sessions_dir = temp_sessions_dir();
        let parent_dir = temp_sessions_dir();
        let parent_recorder = Arc::new(Mutex::new(
            TranscriptRecorder::create(&parent_dir).expect("create parent recorder"),
        ));
        let parent_session_id = parent_recorder
            .lock()
            .expect("lock parent recorder")
            .session_id()
            .to_string();

        let run_summary = runtime
            .run_with_executor(
                &agent,
                AgentTemplate::explorer(),
                "inspect src/subagent.rs".into(),
                test_governance(),
                sessions_dir,
                parent_session_id.clone(),
                "turn-1".into(),
                Some(Arc::clone(&parent_recorder)),
                None,
                |_agent, _task, _transcript, _runner_tx, _child_session_id, _agent_name| {
                    async move { Ok("completed summary".into()) }.boxed()
                },
            None,
            )
            .await
            .expect("run succeeds");

        let parent_records = read_records(parent_dir.join(format!("{}.jsonl", parent_session_id)))
            .expect("read parent records");

        assert_eq!(run_summary.status, SubagentStatus::Completed);
        assert_eq!(parent_records.len(), 3);
        match &parent_records[0].event {
            crate::transcript::TranscriptEvent::SubagentStarted {
                run_id,
                child_session_id,
                summary,
                pool_ordinal: _,
                ..
            } => {
                assert_eq!(run_id, &run_summary.run_id);
                assert_eq!(child_session_id, &run_summary.child_session_id);
                assert_eq!(summary, "inspect src/subagent.rs");
            }
            other => panic!("unexpected parent event: {other:?}"),
        }
        match &parent_records[1].event {
            crate::transcript::TranscriptEvent::SubagentResult {
                status,
                summary,
                child_session_id,
                ..
            } => {
                assert_eq!(status, "completed");
                assert_eq!(summary, "completed summary");
                assert_eq!(child_session_id, &run_summary.child_session_id);
            }
            other => panic!("unexpected parent event: {other:?}"),
        }
        match &parent_records[2].event {
            crate::transcript::TranscriptEvent::Evidence {
                source, summary, ..
            } => {
                assert_eq!(summary, "completed summary");
                assert!(matches!(
                    source,
                    crate::evidence::EvidenceSource::Subagent {
                        run_id,
                        child_session_id,
                        ..
                    } if run_id == &run_summary.run_id && child_session_id == &run_summary.child_session_id
                ));
            }
            other => panic!("unexpected parent event: {other:?}"),
        }
    }

    #[tokio::test]
    async fn child_transcript_session_started_records_actual_child_model() {
        let runtime = SubagentPool::new();
        let mut agent = test_agent();
        agent.set_subagent_model_override("explorer", "gpt-explorer");
        let sessions_dir = temp_sessions_dir();

        let summary = runtime
            .run_with_executor(
                &agent,
                AgentTemplate::explorer(),
                "inspect src/subagent.rs".into(),
                test_governance(),
                sessions_dir.clone(),
                "parent-session".into(),
                "turn-1".into(),
                None,
                None,
                |_agent, _task, _transcript, _runner_tx, _child_session_id, _agent_name| {
                    async move { Ok("completed summary".into()) }.boxed()
                },
            None,
            )
            .await
            .expect("run succeeds");

        let child_records = read_records(
            child_sessions_dir(&sessions_dir).join(format!("{}.jsonl", summary.child_session_id)),
        )
        .expect("read child records");

        match &child_records[0].event {
            crate::transcript::TranscriptEvent::SessionStarted { model } => {
                assert_eq!(model, "gpt-explorer");
            }
            other => panic!("unexpected child event: {other:?}"),
        }
    }

    #[tokio::test]
    async fn timeout_records_timed_out_and_releases_guard() {
        let runtime = SubagentPool::new();
        let mut template = AgentTemplate::explorer();
        template.timeout_secs = Some(1);

        let summary = runtime
            .run_with_executor(
                &test_agent(),
                template,
                "inspect".into(),
                test_governance(),
                temp_sessions_dir(),
                "parent-session".into(),
                "turn-1".into(),
                None,
                None,
                |_agent, _task, _transcript, _runner_tx, _child_session_id, _agent_name| {
                    async move {
                        tokio::time::sleep(Duration::from_secs(60)).await;
                        Ok("late".into())
                    }
                    .boxed()
                },
            None,
            )
            .await
            .expect("timeout returns summary");
        assert_eq!(summary.status, SubagentStatus::TimedOut);

        let next = runtime
            .run_with_executor(
                &test_agent(),
                AgentTemplate::explorer(),
                "inspect again".into(),
                test_governance(),
                temp_sessions_dir(),
                "parent-session".into(),
                "turn-2".into(),
                None,
                None,
                |_agent, _task, _transcript, _runner_tx, _child_session_id, _agent_name| {
                    async move { Ok("done".into()) }.boxed()
                },
            None,
            )
            .await
            .expect("second run succeeds after timeout");
        assert_eq!(next.status, SubagentStatus::Completed);
    }

    #[tokio::test]
    async fn failed_and_timed_out_subagents_do_not_emit_global_error_events() {
        let runtime = SubagentPool::new();
        let (_tx, mut rx) = tokio::sync::mpsc::unbounded_channel();

        let failed = runtime
            .run_with_executor(
                &test_agent(),
                AgentTemplate::fixer(),
                "fail task".into(),
                test_governance(),
                temp_sessions_dir(),
                "parent-session".into(),
                "turn-1".into(),
                None,
                Some(crate::tui::runner::subagent_event_sender::<OpenAIConfig>(
                    _tx.clone(),
                )),
                |_agent, _task, _transcript, _runner_tx, _child_session_id, _agent_name| {
                    async move { Err(anyhow!("child tool denied")) }.boxed()
                },
            None,
            )
            .await
            .expect("failed subagent still returns summary");
        assert_eq!(failed.status, SubagentStatus::Failed);

        let mut saw_terminal_status = false;
        while let Ok(event) = rx.try_recv() {
            match event {
                RunnerEvent::Notice(notice) => {
                    if notice.message.contains("failed") {
                        saw_terminal_status = true;
                    }
                }
                RunnerEvent::Error(error) => {
                    panic!("unexpected global error event: {}", error.message);
                }
                _ => {}
            }
        }
        assert!(
            saw_terminal_status,
            "expected terminal status event for failed subagent"
        );

        let runtime = SubagentPool::new();
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let mut template = AgentTemplate::fixer();
        template.timeout_secs = Some(1);
        let timed_out = runtime
            .run_with_executor(
                &test_agent(),
                template,
                "timeout task".into(),
                test_governance(),
                temp_sessions_dir(),
                "parent-session".into(),
                "turn-2".into(),
                None,
                Some(crate::tui::runner::subagent_event_sender::<OpenAIConfig>(
                    tx,
                )),
                |_agent, _task, _transcript, _runner_tx, _child_session_id, _agent_name| {
                    async move {
                        tokio::time::sleep(Duration::from_secs(60)).await;
                        Ok("late".into())
                    }
                    .boxed()
                },
            None,
            )
            .await
            .expect("timed out subagent still returns summary");
        assert_eq!(timed_out.status, SubagentStatus::TimedOut);

        let mut saw_terminal_status = false;
        while let Ok(event) = rx.try_recv() {
            match event {
                RunnerEvent::Notice(notice) => {
                    if notice.message.contains("timed_out") || notice.message.contains("timed out")
                    {
                        saw_terminal_status = true;
                    }
                }
                RunnerEvent::Error(error) => {
                    panic!("unexpected global error event: {}", error.message);
                }
                _ => {}
            }
        }
        assert!(
            saw_terminal_status,
            "expected terminal status event for timed out subagent"
        );
    }

    #[tokio::test]
    async fn dropped_run_future_releases_concurrency_guard() {
        let runtime = SubagentPool::new();
        let agent = test_agent();
        let sessions_dir = temp_sessions_dir();
        let barrier = Arc::new(Barrier::new(2));

        let run_runtime = runtime.clone();
        let run_barrier = Arc::clone(&barrier);
        let run = tokio::spawn(async move {
            run_runtime
                .run_with_executor(
                    &agent,
                    AgentTemplate::explorer(),
                    "inspect".into(),
                    test_governance(),
                    sessions_dir,
                    "parent-session".into(),
                    "turn-1".into(),
                    None,
                    None,
                    move |_agent,
                          _task,
                          _transcript,
                          _runner_tx,
                          _child_session_id,
                          _agent_name| {
                        async move {
                            run_barrier.wait().await;
                            std::future::pending::<Result<String>>().await
                        }
                        .boxed()
                    },
            None,
                )
                .await
        });

        barrier.wait().await;
        run.abort();
        assert!(
            run.await
                .expect_err("run task should be aborted")
                .is_cancelled()
        );

        let next = runtime
            .run_with_executor(
                &test_agent(),
                AgentTemplate::explorer(),
                "inspect again".into(),
                test_governance(),
                temp_sessions_dir(),
                "parent-session".into(),
                "turn-2".into(),
                None,
                None,
                |_agent, _task, _transcript, _runner_tx, _child_session_id, _agent_name| {
                    async move { Ok("done".into()) }.boxed()
                },
            None,
            )
            .await
            .expect("second run succeeds after aborted caller");
        assert_eq!(next.status, SubagentStatus::Completed);
    }
}
