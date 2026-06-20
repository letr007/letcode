use anyhow::{Result, anyhow};
use async_openai::config::Config;
use futures_util::FutureExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::oneshot;
use tokio::time::{Duration, timeout};

use crate::agent::{Agent, AgentFactory, AgentTemplate, SubagentInvocation};
use crate::tool::NormalizedSubagentInput;
use crate::transcript::{
    ChildSessionSummary, TranscriptEvent, TranscriptRecorder, child_sessions_dir, read_records,
};
use crate::tui::events::ErrorEvent;
use crate::tui::runner::{AgentRunner, RunnerEvent, RunnerEventSender};

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
pub struct SubagentRuntime {
    running: Arc<AtomicBool>,
    active_cancel: Arc<Mutex<Option<oneshot::Sender<()>>>>,
    active_child: Arc<Mutex<Option<ChildSessionSummary>>>,
}

struct ActiveRunGuard {
    running: Arc<AtomicBool>,
    active_cancel: Arc<Mutex<Option<oneshot::Sender<()>>>>,
    active_child: Arc<Mutex<Option<ChildSessionSummary>>>,
}

impl ActiveRunGuard {
    fn new(
        running: Arc<AtomicBool>,
        active_cancel: Arc<Mutex<Option<oneshot::Sender<()>>>>,
        active_child: Arc<Mutex<Option<ChildSessionSummary>>>,
    ) -> Self {
        Self {
            running,
            active_cancel,
            active_child,
        }
    }

    fn clear_cancel(&self) {
        if let Ok(mut sender) = self.active_cancel.lock() {
            sender.take();
        }
    }
}

impl Drop for ActiveRunGuard {
    fn drop(&mut self) {
        if let Ok(mut sender) = self.active_cancel.lock() {
            sender.take();
        }
        if let Ok(mut active_child) = self.active_child.lock() {
            active_child.take();
        }
        self.running.store(false, Ordering::SeqCst);
    }
}

impl Default for SubagentRuntime {
    fn default() -> Self {
        Self::new()
    }
}

impl SubagentRuntime {
    pub fn new() -> Self {
        Self {
            running: Arc::new(AtomicBool::new(false)),
            active_cancel: Arc::new(Mutex::new(None)),
            active_child: Arc::new(Mutex::new(None)),
        }
    }

    pub fn cancel_active(&self) -> bool {
        self.active_cancel
            .lock()
            .ok()
            .and_then(|mut sender| sender.take())
            .map(|sender| sender.send(()).is_ok())
            .unwrap_or(false)
    }

    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    pub fn active_child(&self) -> Option<ChildSessionSummary> {
        self.active_child
            .lock()
            .ok()
            .and_then(|active_child| active_child.clone())
    }

    pub async fn run_explorer<C: Config + Clone + Send + Sync + 'static>(
        &self,
        parent: &Agent<C>,
        task: String,
        sessions_dir: impl AsRef<Path>,
        parent_session_id: String,
        parent_turn_id: String,
        parent_transcript: Option<Arc<Mutex<TranscriptRecorder>>>,
        runner_tx: Option<RunnerEventSender>,
    ) -> Result<SubagentRunSummary> {
        self.run_explorer_governed(
            parent,
            SubagentInvocation {
                prompt: task.clone(),
                input: legacy_task_input(task),
            },
            sessions_dir,
            parent_session_id,
            parent_turn_id,
            parent_transcript,
            runner_tx,
        )
        .await
    }

    pub async fn run_explorer_governed<C: Config + Clone + Send + Sync + 'static>(
        &self,
        parent: &Agent<C>,
        invocation: SubagentInvocation,
        sessions_dir: impl AsRef<Path>,
        parent_session_id: String,
        parent_turn_id: String,
        parent_transcript: Option<Arc<Mutex<TranscriptRecorder>>>,
        runner_tx: Option<RunnerEventSender>,
    ) -> Result<SubagentRunSummary> {
        self.run_named_governed(
            parent,
            "explorer",
            invocation,
            sessions_dir,
            parent_session_id,
            parent_turn_id,
            parent_transcript,
            runner_tx,
        )
        .await
    }

    pub async fn run_fixer<C: Config + Clone + Send + Sync + 'static>(
        &self,
        parent: &Agent<C>,
        task: String,
        sessions_dir: impl AsRef<Path>,
        parent_session_id: String,
        parent_turn_id: String,
        parent_transcript: Option<Arc<Mutex<TranscriptRecorder>>>,
        runner_tx: Option<RunnerEventSender>,
    ) -> Result<SubagentRunSummary> {
        self.run_fixer_governed(
            parent,
            SubagentInvocation {
                prompt: task.clone(),
                input: legacy_task_input(task),
            },
            sessions_dir,
            parent_session_id,
            parent_turn_id,
            parent_transcript,
            runner_tx,
        )
        .await
    }

    pub async fn run_fixer_governed<C: Config + Clone + Send + Sync + 'static>(
        &self,
        parent: &Agent<C>,
        invocation: SubagentInvocation,
        sessions_dir: impl AsRef<Path>,
        parent_session_id: String,
        parent_turn_id: String,
        parent_transcript: Option<Arc<Mutex<TranscriptRecorder>>>,
        runner_tx: Option<RunnerEventSender>,
    ) -> Result<SubagentRunSummary> {
        self.run_named_governed(
            parent,
            "fixer",
            invocation,
            sessions_dir,
            parent_session_id,
            parent_turn_id,
            parent_transcript,
            runner_tx,
        )
        .await
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
        runner_tx: Option<RunnerEventSender>,
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
            runner_tx,
            |agent, prompt, transcript, runner_tx, child_session_id, agent_name| {
                async move {
                    if agent_name == "fixer" {
                        run_child_agent_with_permissions(
                            agent,
                            prompt,
                            transcript,
                            runner_tx,
                            child_session_id,
                            agent_name,
                        )
                        .await
                    } else {
                        run_child_agent(agent, prompt, transcript, runner_tx, child_session_id)
                            .await
                    }
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
        runner_tx: Option<RunnerEventSender>,
        exec: F,
    ) -> Result<SubagentRunSummary>
    where
        C: Config + Clone + Send + Sync + 'static,
        F: FnOnce(
                Agent<C>,
                String,
                Arc<Mutex<TranscriptRecorder>>,
                Option<RunnerEventSender>,
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
            runner_tx,
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
        runner_tx: Option<RunnerEventSender>,
    ) -> Result<StartedRun<C>>
    where
        C: Config + Clone + Send + Sync + 'static,
    {
        if self.running.swap(true, Ordering::SeqCst) {
            return Err(anyhow!("subagent is already running"));
        }

        let effective_timeout_secs = governance.timeout_secs.or(template.timeout_secs);
        let effective_max_tool_calls = governance.max_tool_calls.or(template.max_tool_calls);

        let child_agent = AgentFactory::create_child_with_max_tool_calls(
            parent,
            &template,
            effective_max_tool_calls,
        );

        let setup = (|| -> Result<(String, String, Arc<Mutex<TranscriptRecorder>>)> {
            let run_id = generate_run_id();
            let child_dir = child_sessions_dir(&sessions_dir);
            let mut child_recorder = TranscriptRecorder::create(&child_dir)?;
            child_recorder.record_session_started(child_agent.model().to_string())?;
            child_recorder.record_subagent_lifecycle(
                run_id.clone(),
                parent_session_id.clone(),
                parent_turn_id.clone(),
                template.name.clone(),
                SubagentStatus::Running.as_str(),
                Some(task.clone()),
            )?;
            let child_session_id = child_recorder.session_id().to_string();
            Ok((
                run_id,
                child_session_id,
                Arc::new(Mutex::new(child_recorder)),
            ))
        })();

        let (run_id, child_session_id, child_transcript) = match setup {
            Ok(values) => values,
            Err(error) => {
                self.running.store(false, Ordering::SeqCst);
                return Err(error);
            }
        };
        if let Err(error) = record_parent_lifecycle(
            &parent_transcript,
            &run_id,
            &parent_session_id,
            &parent_turn_id,
            &template.name,
            SubagentStatus::Running,
            Some(task.clone()),
        ) {
            self.running.store(false, Ordering::SeqCst);
            return Err(error);
        }

        if let Some(sender) = &runner_tx {
            let _ = sender.send(RunnerEvent::Status(format!(
                "{} running · run {}",
                template.name, run_id
            )));
        }

        let (cancel_tx, cancel_rx) = oneshot::channel();
        if let Ok(mut active_cancel) = self.active_cancel.lock() {
            *active_cancel = Some(cancel_tx);
        }

        if let Ok(mut active_child) = self.active_child.lock() {
            *active_child = Some(ChildSessionSummary {
                parent_session_id: parent_session_id.clone(),
                parent_run_id: parent_turn_id.clone(),
                child_session_id: child_session_id.clone(),
                agent_name: template.name.clone(),
                status: SubagentStatus::Running.as_str().into(),
                summary: task.clone(),
                timestamp_ms: current_timestamp_ms(),
            });
        }

        Ok(StartedRun {
            guard: ActiveRunGuard::new(
                Arc::clone(&self.running),
                Arc::clone(&self.active_cancel),
                Arc::clone(&self.active_child),
            ),
            run_id,
            child_session_id,
            agent_name: template.name.clone(),
            timeout_secs: effective_timeout_secs,
            governance,
            parent_session_id,
            parent_turn_id,
            parent_transcript,
            runner_tx,
            child_transcript,
            child_agent,
            cancel_rx,
        })
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
    runner_tx: Option<RunnerEventSender>,
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
            Option<RunnerEventSender>,
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
        runner_tx,
        child_transcript,
        child_agent,
        cancel_rx,
    } = started;

    let execution = exec(
        child_agent,
        task,
        Arc::clone(&child_transcript),
        runner_tx.clone(),
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
    ) && let Some(sender) = &runner_tx
    {
        let _ = sender.send(RunnerEvent::Error(ErrorEvent::new(format!(
            "failed to record child subagent completion: {error}"
        ))));
    }

    let parent_record_result = record_parent_result(
        &parent_transcript,
        &summary,
        &parent_session_id,
        &parent_turn_id,
    );
    if let Err(error) = parent_record_result {
        if let Some(sender) = &runner_tx {
            let _ = sender.send(RunnerEvent::Error(ErrorEvent::new(format!(
                "failed to record parent subagent result: {error}"
            ))));
        }
        return Err(error);
    }
    if let Some(sender) = &runner_tx {
        let _ = sender.send(RunnerEvent::Status(format!(
            "{} {} · {} · /child to inspect {}",
            summary.agent_name,
            summary.status.as_str(),
            summary.summary,
            short_session_id(&summary.child_session_id)
        )));
    }
    Ok(summary)
}

async fn run_child_agent<C: Config + Clone + Send + Sync + 'static>(
    mut agent: Agent<C>,
    prompt: String,
    transcript: Arc<Mutex<TranscriptRecorder>>,
    runner_tx: Option<RunnerEventSender>,
    child_session_id: String,
) -> Result<String> {
    let runner: AgentRunner<C> = if let Some(runner_tx) = runner_tx {
        AgentRunner::child_streaming_with_transcript(transcript, runner_tx, child_session_id)
    } else {
        AgentRunner::silent_with_transcript(transcript)
    };
    runner.run_prompt(&mut agent, prompt).await
}

async fn run_child_agent_with_permissions<C: Config + Clone + Send + Sync + 'static>(
    mut agent: Agent<C>,
    prompt: String,
    transcript: Arc<Mutex<TranscriptRecorder>>,
    runner_tx: Option<RunnerEventSender>,
    child_session_id: String,
    agent_name: String,
) -> Result<String> {
    let runner: AgentRunner<C> = if let Some(runner_tx) = runner_tx {
        AgentRunner::child_streaming_with_permission_passthrough(
            transcript,
            runner_tx,
            child_session_id,
            agent_name,
        )
    } else {
        AgentRunner::silent_with_transcript(transcript)
    };
    runner.run_prompt(&mut agent, prompt).await
}

fn record_parent_lifecycle(
    transcript: &Option<Arc<Mutex<TranscriptRecorder>>>,
    run_id: &str,
    parent_session_id: &str,
    parent_turn_id: &str,
    agent_name: &str,
    status: SubagentStatus,
    detail: Option<String>,
) -> Result<()> {
    let Some(transcript) = transcript else {
        return Ok(());
    };
    let mut recorder = transcript
        .lock()
        .map_err(|_| anyhow!("parent transcript recorder poisoned"))?;
    recorder.record_subagent_lifecycle(
        run_id.to_string(),
        parent_session_id.to_string(),
        parent_turn_id.to_string(),
        agent_name.to_string(),
        status.as_str().to_string(),
        detail,
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

fn generate_run_id() -> String {
    static NEXT_RUN_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let millis = current_timestamp_ms();
    let suffix = NEXT_RUN_ID.fetch_add(1, Ordering::SeqCst);
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
    let Ok(records) = read_records(path) else {
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
    fn explorer_child_uses_default_permission_even_when_parent_is_safe() {
        let mut agent = test_agent();
        agent.set_permission_mode(crate::permission::PermissionMode::Safe);
        let child = AgentFactory::create_child(&agent, &AgentTemplate::explorer());

        assert_eq!(agent.permission_mode().as_str(), "safe");
        assert_eq!(child.permission_mode().as_str(), "default");
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
        let runtime = SubagentRuntime::new();
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
        let runtime = SubagentRuntime::new();
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
            )
            .await
            .expect("run returns summary");

        assert_eq!(summary.status, SubagentStatus::Failed);
        assert!(summary.summary.contains("src/outside.rs"));
    }

    #[tokio::test]
    async fn governance_timeout_overrides_template_default() {
        let runtime = SubagentRuntime::new();
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
            )
            .await
            .expect("timeout returns summary");

        assert_eq!(summary.status, SubagentStatus::TimedOut);
        assert_eq!(summary.structured_result.status, "timed_out");
    }

    #[tokio::test]
    async fn max_concurrency_guard_rejects_second_run() {
        let runtime = SubagentRuntime::new();
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
            )
            .await
            .expect_err("second run should be rejected");

        assert!(second.to_string().contains("already running"));
        let _ = first.await.expect("join first").expect("first ok");
    }

    #[tokio::test]
    async fn cancel_active_records_cancelled_and_releases_guard() {
        let runtime = SubagentRuntime::new();
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
            )
            .await
            .expect("second run succeeds after cancellation");
        assert_eq!(next.status, SubagentStatus::Completed);
    }

    #[tokio::test]
    async fn active_child_uses_running_status_while_subagent_is_active() {
        let runtime = SubagentRuntime::new();
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
        let runtime = SubagentRuntime::new();
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
            )
            .await
            .expect("run succeeds");

        let parent_records = read_records(parent_dir.join(format!("{}.jsonl", parent_session_id)))
            .expect("read parent records");

        assert_eq!(run_summary.status, SubagentStatus::Completed);
        assert_eq!(parent_records.len(), 3);
        match &parent_records[0].event {
            crate::transcript::TranscriptEvent::SubagentLifecycle { status, detail, .. } => {
                assert_eq!(status, "running");
                assert_eq!(detail.as_deref(), Some("inspect src/subagent.rs"));
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
        let runtime = SubagentRuntime::new();
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
        let runtime = SubagentRuntime::new();
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
            )
            .await
            .expect("second run succeeds after timeout");
        assert_eq!(next.status, SubagentStatus::Completed);
    }

    #[tokio::test]
    async fn failed_and_timed_out_subagents_do_not_emit_global_error_events() {
        let runtime = SubagentRuntime::new();
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
                Some(_tx.clone()),
                |_agent, _task, _transcript, _runner_tx, _child_session_id, _agent_name| {
                    async move { Err(anyhow!("child tool denied")) }.boxed()
                },
            )
            .await
            .expect("failed subagent still returns summary");
        assert_eq!(failed.status, SubagentStatus::Failed);

        let mut saw_terminal_status = false;
        while let Ok(event) = rx.try_recv() {
            match event {
                RunnerEvent::Status(message) => {
                    if message.contains("failed") {
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

        let runtime = SubagentRuntime::new();
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
                Some(tx),
                |_agent, _task, _transcript, _runner_tx, _child_session_id, _agent_name| {
                    async move {
                        tokio::time::sleep(Duration::from_secs(60)).await;
                        Ok("late".into())
                    }
                    .boxed()
                },
            )
            .await
            .expect("timed out subagent still returns summary");
        assert_eq!(timed_out.status, SubagentStatus::TimedOut);

        let mut saw_terminal_status = false;
        while let Ok(event) = rx.try_recv() {
            match event {
                RunnerEvent::Status(message) => {
                    if message.contains("timed_out") || message.contains("timed out") {
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
        let runtime = SubagentRuntime::new();
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
            )
            .await
            .expect("second run succeeds after aborted caller");
        assert_eq!(next.status, SubagentStatus::Completed);
    }
}
