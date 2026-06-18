use anyhow::{Result, anyhow};
use async_openai::config::Config;
use futures_util::FutureExt;
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

use crate::agent::{Agent, AgentFactory, AgentTemplate};
use crate::transcript::{ChildSessionSummary, TranscriptRecorder, child_sessions_dir};
use crate::tui::events::ErrorEvent;
use crate::tui::runner::{AgentRunner, RunnerEvent, RunnerEventSender};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubagentStatus {
    Running,
    Completed,
    Failed,
    Cancelled,
    TimedOut,
}

impl SubagentStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Failed => "failed",
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
}

type BoxExecFuture = Pin<Box<dyn Future<Output = Result<String>> + Send>>;

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
        let template = AgentTemplate::explorer();
        self.run_with_executor(
            parent,
            template,
            task,
            sessions_dir.as_ref().to_path_buf(),
            parent_session_id,
            parent_turn_id,
            parent_transcript,
            runner_tx,
            |agent, prompt, transcript, _runner_tx, _agent_name| {
                async move { run_child_agent(agent, prompt, transcript).await }.boxed()
            },
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
        let template = AgentTemplate::fixer();
        self.run_with_executor(
            parent,
            template,
            task,
            sessions_dir.as_ref().to_path_buf(),
            parent_session_id,
            parent_turn_id,
            parent_transcript,
            runner_tx,
            |agent, prompt, transcript, runner_tx, agent_name| {
                async move {
                    run_child_agent_with_permissions(
                        agent, prompt, transcript, runner_tx, agent_name,
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
            ) -> BoxExecFuture
            + Send
            + 'static,
    {
        let running = self.start_run(
            parent,
            template,
            task.clone(),
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

        let setup = (|| -> Result<(String, String, Arc<Mutex<TranscriptRecorder>>)> {
            let run_id = generate_run_id();
            let child_dir = child_sessions_dir(&sessions_dir);
            let mut child_recorder = TranscriptRecorder::create(&child_dir)?;
            child_recorder.record_session_started(parent.model().to_string())?;
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
            timeout_secs: template.timeout_secs,
            parent_session_id,
            parent_turn_id,
            parent_transcript,
            runner_tx,
            child_transcript,
            child_agent: AgentFactory::create_child(parent, &template),
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
        agent_name.clone(),
    );
    let summary = tokio::select! {
        result = async {
            match timeout_secs {
                Some(timeout_secs) => match timeout(Duration::from_secs(timeout_secs), execution).await {
                    Ok(Ok(message)) => SubagentRunSummary {
                        run_id: run_id.clone(),
                        child_session_id: child_session_id.clone(),
                        agent_name: agent_name.clone(),
                        status: SubagentStatus::Completed,
                        summary: message,
                    },
                    Ok(Err(error)) => SubagentRunSummary {
                        run_id: run_id.clone(),
                        child_session_id: child_session_id.clone(),
                        agent_name: agent_name.clone(),
                        status: SubagentStatus::Failed,
                        summary: error.to_string(),
                    },
                    Err(_) => SubagentRunSummary {
                        run_id: run_id.clone(),
                        child_session_id: child_session_id.clone(),
                        agent_name: agent_name.clone(),
                        status: SubagentStatus::TimedOut,
                        summary: format!("{} timed out after {timeout_secs}s", agent_name),
                    },
                },
                None => match execution.await {
                    Ok(message) => SubagentRunSummary {
                        run_id: run_id.clone(),
                        child_session_id: child_session_id.clone(),
                        agent_name: agent_name.clone(),
                        status: SubagentStatus::Completed,
                        summary: message,
                    },
                    Err(error) => SubagentRunSummary {
                        run_id: run_id.clone(),
                        child_session_id: child_session_id.clone(),
                        agent_name: agent_name.clone(),
                        status: SubagentStatus::Failed,
                        summary: error.to_string(),
                    },
                },
            }
        } => result,
        _ = cancel_rx => {
            SubagentRunSummary {
                run_id: run_id.clone(),
                child_session_id: child_session_id.clone(),
                agent_name: agent_name.clone(),
                status: SubagentStatus::Cancelled,
                summary: format!("{} cancelled", agent_name),
            }
        }
    };

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

async fn run_child_agent<C: Config + Send + Sync + 'static>(
    mut agent: Agent<C>,
    prompt: String,
    transcript: Arc<Mutex<TranscriptRecorder>>,
) -> Result<String> {
    let runner: AgentRunner<C> = AgentRunner::silent_with_transcript(transcript);
    runner.run_prompt(&mut agent, prompt).await
}

async fn run_child_agent_with_permissions<C: Config + Send + Sync + 'static>(
    mut agent: Agent<C>,
    prompt: String,
    transcript: Arc<Mutex<TranscriptRecorder>>,
    runner_tx: Option<RunnerEventSender>,
    agent_name: String,
) -> Result<String> {
    let runner: AgentRunner<C> = if let Some(runner_tx) = runner_tx {
        AgentRunner::silent_with_permission_passthrough(transcript, runner_tx, agent_name)
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
    recorder.record_subagent_result(
        summary.run_id.clone(),
        parent_session_id.to_string(),
        parent_turn_id.to_string(),
        summary.child_session_id.clone(),
        summary.agent_name.clone(),
        summary.status.as_str().to_string(),
        summary.summary.clone(),
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
    let millis = current_timestamp_ms();
    format!("subagent-{millis}")
}

fn current_timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcript::read_records;
    use async_openai::Client;
    use async_openai::config::OpenAIConfig;
    use tokio::sync::Barrier;
    use tokio::time::sleep;

    fn test_agent() -> Agent<OpenAIConfig> {
        Agent::new(Client::with_config(OpenAIConfig::new()), "gpt-test", 2, 4)
    }

    fn temp_sessions_dir() -> PathBuf {
        std::env::temp_dir().join(generate_run_id())
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
    fn explorer_child_uses_default_permission_even_when_parent_is_safe() {
        let mut agent = test_agent();
        agent.set_permission_mode(crate::permission::PermissionMode::Safe);
        let child = AgentFactory::create_child(&agent, &AgentTemplate::explorer());

        assert_eq!(agent.permission_mode().as_str(), "safe");
        assert_eq!(child.permission_mode().as_str(), "default");
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
                    sessions_dir,
                    "parent-session".into(),
                    "turn-1".into(),
                    None,
                    None,
                    move |_agent, _task, _transcript, _runner_tx, _agent_name| {
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
                temp_sessions_dir(),
                "parent-session".into(),
                "turn-2".into(),
                None,
                None,
                |_agent, _task, _transcript, _runner_tx, _agent_name| {
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
                    sessions_dir,
                    "parent-session".into(),
                    "turn-1".into(),
                    None,
                    None,
                    move |_agent, _task, _transcript, _runner_tx, _agent_name| {
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
                temp_sessions_dir(),
                "parent-session".into(),
                "turn-2".into(),
                None,
                None,
                |_agent, _task, _transcript, _runner_tx, _agent_name| {
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
                    sessions_dir,
                    "parent-session".into(),
                    "turn-1".into(),
                    None,
                    None,
                    move |_agent, _task, _transcript, _runner_tx, _agent_name| {
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
                sessions_dir,
                parent_session_id.clone(),
                "turn-1".into(),
                Some(Arc::clone(&parent_recorder)),
                None,
                |_agent, _task, _transcript, _runner_tx, _agent_name| {
                    async move { Ok("completed summary".into()) }.boxed()
                },
            )
            .await
            .expect("run succeeds");

        let parent_records = read_records(parent_dir.join(format!("{}.jsonl", parent_session_id)))
            .expect("read parent records");

        assert_eq!(run_summary.status, SubagentStatus::Completed);
        assert_eq!(parent_records.len(), 2);
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
    }

    #[tokio::test]
    async fn timeout_records_timed_out_and_releases_guard() {
        let runtime = SubagentRuntime::new();
        let mut template = AgentTemplate::explorer();
        template.timeout_secs = Some(0);

        let summary = runtime
            .run_with_executor(
                &test_agent(),
                template,
                "inspect".into(),
                temp_sessions_dir(),
                "parent-session".into(),
                "turn-1".into(),
                None,
                None,
                |_agent, _task, _transcript, _runner_tx, _agent_name| {
                    async move { std::future::pending::<Result<String>>().await }.boxed()
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
                temp_sessions_dir(),
                "parent-session".into(),
                "turn-2".into(),
                None,
                None,
                |_agent, _task, _transcript, _runner_tx, _agent_name| {
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
                temp_sessions_dir(),
                "parent-session".into(),
                "turn-1".into(),
                None,
                Some(_tx.clone()),
                |_agent, _task, _transcript, _runner_tx, _agent_name| {
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
        template.timeout_secs = Some(0);
        let timed_out = runtime
            .run_with_executor(
                &test_agent(),
                template,
                "timeout task".into(),
                temp_sessions_dir(),
                "parent-session".into(),
                "turn-2".into(),
                None,
                Some(tx),
                |_agent, _task, _transcript, _runner_tx, _agent_name| {
                    async move { std::future::pending::<Result<String>>().await }.boxed()
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
                    sessions_dir,
                    "parent-session".into(),
                    "turn-1".into(),
                    None,
                    None,
                    move |_agent, _task, _transcript, _runner_tx, _agent_name| {
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
                temp_sessions_dir(),
                "parent-session".into(),
                "turn-2".into(),
                None,
                None,
                |_agent, _task, _transcript, _runner_tx, _agent_name| {
                    async move { Ok("done".into()) }.boxed()
                },
            )
            .await
            .expect("second run succeeds after aborted caller");
        assert_eq!(next.status, SubagentStatus::Completed);
    }
}
