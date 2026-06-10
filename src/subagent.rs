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
use crate::transcript::{TranscriptRecorder, child_sessions_dir};
use crate::tui::events::ErrorEvent;
use crate::tui::runner::{AgentRunner, RunnerEvent, RunnerEventSender};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubagentStatus {
    Started,
    Completed,
    Failed,
    Cancelled,
    TimedOut,
}

impl SubagentStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Started => "started",
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
}

struct ActiveRunGuard {
    running: Arc<AtomicBool>,
    active_cancel: Arc<Mutex<Option<oneshot::Sender<()>>>>,
}

impl ActiveRunGuard {
    fn new(
        running: Arc<AtomicBool>,
        active_cancel: Arc<Mutex<Option<oneshot::Sender<()>>>>,
    ) -> Self {
        Self {
            running,
            active_cancel,
        }
    }
}

impl Drop for ActiveRunGuard {
    fn drop(&mut self) {
        if let Ok(mut sender) = self.active_cancel.lock() {
            sender.take();
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
            |agent, prompt, transcript| {
                async move { run_child_agent(agent, prompt, transcript).await }.boxed()
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
        F: FnOnce(Agent<C>, String, Arc<Mutex<TranscriptRecorder>>) -> BoxExecFuture
            + Send
            + 'static,
    {
        if self.running.swap(true, Ordering::SeqCst) {
            return Err(anyhow!("explorer subagent is already running"));
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
                SubagentStatus::Started.as_str(),
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
        record_parent_lifecycle(
            &parent_transcript,
            &run_id,
            &parent_session_id,
            &parent_turn_id,
            &template.name,
            SubagentStatus::Started,
            Some(task.clone()),
        );

        if let Some(sender) = &runner_tx {
            let _ = sender.send(RunnerEvent::Notice(format!(
                "Explorer started · run {}",
                run_id
            )));
        }

        let (cancel_tx, cancel_rx) = oneshot::channel();
        if let Ok(mut active_cancel) = self.active_cancel.lock() {
            *active_cancel = Some(cancel_tx);
        }

        let _active_run =
            ActiveRunGuard::new(Arc::clone(&self.running), Arc::clone(&self.active_cancel));
        let template_name = template.name.clone();
        let child_agent = AgentFactory::create_child(parent, &template);
        let timeout_secs = template.timeout_secs;

        let execution = exec(child_agent, task, Arc::clone(&child_transcript));
        let summary = tokio::select! {
            result = timeout(Duration::from_secs(timeout_secs), execution) => {
                match result {
                    Ok(Ok(message)) => SubagentRunSummary {
                        run_id: run_id.clone(),
                        child_session_id: child_session_id.clone(),
                        agent_name: template_name.clone(),
                        status: SubagentStatus::Completed,
                        summary: message,
                    },
                    Ok(Err(error)) => SubagentRunSummary {
                        run_id: run_id.clone(),
                        child_session_id: child_session_id.clone(),
                        agent_name: template_name.clone(),
                        status: SubagentStatus::Failed,
                        summary: error.to_string(),
                    },
                    Err(_) => SubagentRunSummary {
                        run_id: run_id.clone(),
                        child_session_id: child_session_id.clone(),
                        agent_name: template_name.clone(),
                        status: SubagentStatus::TimedOut,
                        summary: format!("explorer timed out after {timeout_secs}s"),
                    },
                }
            }
            _ = cancel_rx => {
                SubagentRunSummary {
                    run_id: run_id.clone(),
                    child_session_id: child_session_id.clone(),
                    agent_name: template_name.clone(),
                    status: SubagentStatus::Cancelled,
                    summary: "explorer cancelled".into(),
                }
            }
        };

        record_child_completion(
            &child_transcript,
            &summary,
            &parent_session_id,
            &parent_turn_id,
        );
        record_parent_lifecycle(
            &parent_transcript,
            &summary.run_id,
            &parent_session_id,
            &parent_turn_id,
            &summary.agent_name,
            summary.status,
            Some(summary.summary.clone()),
        );
        record_parent_result(
            &parent_transcript,
            &summary,
            &parent_session_id,
            &parent_turn_id,
        );
        if let Some(sender) = &runner_tx {
            let _ = sender.send(RunnerEvent::Notice(format!(
                "Explorer {} · {} · /child to inspect {}",
                summary.status.as_str(),
                summary.summary,
                short_session_id(&summary.child_session_id)
            )));
            if matches!(
                summary.status,
                SubagentStatus::Failed | SubagentStatus::TimedOut
            ) {
                let _ = sender.send(RunnerEvent::Error(ErrorEvent::new(summary.summary.clone())));
            }
        }

        Ok(summary)
    }
}

async fn run_child_agent<C: Config + Send + Sync + 'static>(
    mut agent: Agent<C>,
    prompt: String,
    transcript: Arc<Mutex<TranscriptRecorder>>,
) -> Result<String> {
    let runner: AgentRunner<C> = AgentRunner::silent_with_transcript(transcript);
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
) {
    if let Some(transcript) = transcript
        && let Ok(mut recorder) = transcript.lock()
    {
        let _ = recorder.record_subagent_lifecycle(
            run_id.to_string(),
            parent_session_id.to_string(),
            parent_turn_id.to_string(),
            agent_name.to_string(),
            status.as_str().to_string(),
            detail,
        );
    }
}

fn record_parent_result(
    transcript: &Option<Arc<Mutex<TranscriptRecorder>>>,
    summary: &SubagentRunSummary,
    parent_session_id: &str,
    parent_turn_id: &str,
) {
    if let Some(transcript) = transcript
        && let Ok(mut recorder) = transcript.lock()
    {
        let _ = recorder.record_subagent_result(
            summary.run_id.clone(),
            parent_session_id.to_string(),
            parent_turn_id.to_string(),
            summary.child_session_id.clone(),
            summary.agent_name.clone(),
            summary.status.as_str().to_string(),
            summary.summary.clone(),
        );
    }
}

fn record_child_completion(
    transcript: &Arc<Mutex<TranscriptRecorder>>,
    summary: &SubagentRunSummary,
    parent_session_id: &str,
    parent_turn_id: &str,
) {
    if let Ok(mut recorder) = transcript.lock() {
        let _ = recorder.record_subagent_lifecycle(
            summary.run_id.clone(),
            parent_session_id.to_string(),
            parent_turn_id.to_string(),
            summary.agent_name.clone(),
            summary.status.as_str().to_string(),
            Some(summary.summary.clone()),
        );
    }
}

fn short_session_id(session_id: &str) -> &str {
    session_id.get(..12).unwrap_or(session_id)
}

fn generate_run_id() -> String {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    format!("subagent-{millis}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_openai::Client;
    use async_openai::config::OpenAIConfig;
    use tokio::sync::Barrier;

    fn test_agent() -> Agent<OpenAIConfig> {
        Agent::new(Client::with_config(OpenAIConfig::new()), "gpt-test", 2, 4)
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
        let sessions_dir = std::env::temp_dir().join(generate_run_id());
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
                    move |_agent, _task, _transcript| {
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
                std::env::temp_dir().join(generate_run_id()),
                "parent-session".into(),
                "turn-2".into(),
                None,
                None,
                |_agent, _task, _transcript| async move { Ok("done".into()) }.boxed(),
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
        let sessions_dir = std::env::temp_dir().join(generate_run_id());
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
                    move |_agent, _task, _transcript| {
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
                std::env::temp_dir().join(generate_run_id()),
                "parent-session".into(),
                "turn-2".into(),
                None,
                None,
                |_agent, _task, _transcript| async move { Ok("done".into()) }.boxed(),
            )
            .await
            .expect("second run succeeds after cancellation");
        assert_eq!(next.status, SubagentStatus::Completed);
    }

    #[tokio::test]
    async fn timeout_records_timed_out_and_releases_guard() {
        let runtime = SubagentRuntime::new();
        let mut template = AgentTemplate::explorer();
        template.timeout_secs = 0;

        let summary = runtime
            .run_with_executor(
                &test_agent(),
                template,
                "inspect".into(),
                std::env::temp_dir().join(generate_run_id()),
                "parent-session".into(),
                "turn-1".into(),
                None,
                None,
                |_agent, _task, _transcript| {
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
                std::env::temp_dir().join(generate_run_id()),
                "parent-session".into(),
                "turn-2".into(),
                None,
                None,
                |_agent, _task, _transcript| async move { Ok("done".into()) }.boxed(),
            )
            .await
            .expect("second run succeeds after timeout");
        assert_eq!(next.status, SubagentStatus::Completed);
    }

    #[tokio::test]
    async fn dropped_run_future_releases_concurrency_guard() {
        let runtime = SubagentRuntime::new();
        let agent = test_agent();
        let sessions_dir = std::env::temp_dir().join(generate_run_id());
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
                    move |_agent, _task, _transcript| {
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
                std::env::temp_dir().join(generate_run_id()),
                "parent-session".into(),
                "turn-2".into(),
                None,
                None,
                |_agent, _task, _transcript| async move { Ok("done".into()) }.boxed(),
            )
            .await
            .expect("second run succeeds after aborted caller");
        assert_eq!(next.status, SubagentStatus::Completed);
    }
}
