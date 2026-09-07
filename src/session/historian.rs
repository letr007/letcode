//! Session-owned, single-flight Historian. Uses the same pool and one-shot
//! execution boundary as reviewer; only the host publishes validated artifacts.
use crate::agent::{Agent, SubagentInvocation};
use crate::context_history::HistoryPublication;
use crate::session::runner::{
    SessionTransportEvent, SessionTransportEventSender, subagent_event_sender,
};
use crate::subagent::{SubagentPool, SubagentStatus};
use crate::tool::NormalizedSubagentInput;
use crate::transcript::{TranscriptEvent, TranscriptRecorder};
use anyhow::{Result, anyhow, ensure};
use futures_util::FutureExt;
use std::sync::{Arc, Mutex};
use tokio::sync::oneshot;

#[derive(Clone)]
pub(crate) struct HistoryWork {
    pub id: String,
    pub session_id: String,
    pub branch_id: String,
    pub revision: u64,
    pub source_ids: Vec<String>,
    pub prompt: String,
    pub project_path: String,
    pub external_fact_ids: Vec<String>,
}
struct Pending {
    run_id: String,
    session_id: String,
    branch_id: String,
    revision: u64,
    receiver: oneshot::Receiver<Result<HistoryPublication>>,
    cancelled: Arc<Mutex<bool>>,
}
pub(crate) struct HistorianRuntime {
    pool: SubagentPool,
    sessions_dir: std::path::PathBuf,
    transcript: Arc<Mutex<TranscriptRecorder>>,
    event_tx: SessionTransportEventSender,
    pending: Mutex<Option<Pending>>,
}
impl HistorianRuntime {
    pub(crate) fn new(
        pool: SubagentPool,
        sessions_dir: std::path::PathBuf,
        transcript: Arc<Mutex<TranscriptRecorder>>,
        event_tx: SessionTransportEventSender,
    ) -> Self {
        Self {
            pool,
            sessions_dir,
            transcript,
            event_tx,
            pending: Mutex::new(None),
        }
    }
    pub(crate) async fn project_facts(
        &self,
        project: &str,
        session: &str,
    ) -> Result<Vec<crate::evidence::EvidenceRecord>> {
        let sessions_dir = self.sessions_dir.clone();
        let project = project.to_owned();
        let session = session.to_owned();
        // Journal reads and replay must not block the session task that forwards
        // runner events and handles interrupts. The worker only returns data;
        // dropping this await cannot install facts into a subsequent turn.
        tokio::task::spawn_blocking(move || {
            crate::memory::recall_project_facts(&sessions_dir, &project, &session)
        })
        .await?
    }

    pub(crate) fn is_running(&self) -> bool {
        self.pending.lock().is_ok_and(|pending| pending.is_some())
    }
    pub(crate) fn cancel(&self) {
        if let Ok(mut pending) = self.pending.lock() {
            if let Some(job) = pending.take() {
                if let Ok(mut cancelled) = job.cancelled.lock() {
                    *cancelled = true;
                }
                self.pool.cancel_run(&job.run_id);
            }
        }
        self.emit(false, false);
    }
    fn emit(&self, running: bool, failed: bool) {
        let _ = self.event_tx.send(SessionTransportEvent::HistorianStatus {
            session_id: self
                .transcript
                .lock()
                .map(|r| r.session_id().to_string())
                .unwrap_or_default(),
            running,
            failed,
        });
    }
    pub(crate) fn start(&self, parent: &Agent, work: HistoryWork) -> Result<()> {
        let mut pending = self
            .pending
            .lock()
            .map_err(|_| anyhow!("historian state poisoned"))?;
        if pending.is_some() {
            return Ok(());
        }
        let summary = format!("Organize {} history items", work.source_ids.len());
        let input = NormalizedSubagentInput {
            objective: summary.clone(),
            success_criteria: vec!["Three-tier history with complete source coverage".into()],
            allowed_paths: vec![],
            forbidden_paths: vec![],
            owned_paths: vec![],
            timeout_secs: Some(600),
            max_tool_calls: Some(0),
            model: None,
            target_child_session_id: None,
            background: true,
        };
        let invocation = SubagentInvocation {
            input,
            model: None,
            // Pool lifecycle records contain the task label, not the source
            // transcript. The executor receives the full input separately.
            prompt: summary,
            parent_tool_call_id: None,
        };
        let started = self.pool.start_named_governed(
            parent,
            "historian",
            invocation,
            &self.sessions_dir,
            work.session_id.clone(),
            format!("history-{}", work.id),
            Some(self.transcript.clone()),
            Some(subagent_event_sender(self.event_tx.clone())),
        )?;
        let run_id = started.run_id().to_string();
        let (tx, rx) = oneshot::channel();
        let cancelled = Arc::new(Mutex::new(false));
        *pending = Some(Pending {
            run_id,
            session_id: work.session_id.clone(),
            branch_id: work.branch_id.clone(),
            revision: work.revision,
            receiver: rx,
            cancelled: cancelled.clone(),
        });
        let pool = self.pool.clone();
        let recorder = self.transcript.clone();
        let event_tx = self.event_tx.clone();
        self.emit(true, false);
        tokio::spawn(async move {
            let produced = Arc::new(Mutex::new(None));
            let produced_child = produced.clone();
            let source_ids = work.source_ids.clone();
            let publication_id = work.id.clone();
            let project_path = work.project_path.clone();
            let external_fact_ids = work.external_fact_ids.clone();
            let source_session_id = work.session_id.clone();
            let source_branch_id = work.branch_id.clone();
            let prompt = work.prompt.clone();
            let result = pool.complete_started_run_with_executor(started, move |agent, _, child, _, _, _| {
                async move {
                    child.lock().map_err(|_| anyhow!("historian child transcript poisoned"))?.record_user_message(format!("Historian · {} history items\n\nModel: {}", source_ids.len(), agent.model()))?;
                    let started_at = std::time::Instant::now();
                    let (raw, usage) = agent.run_historian_text(&prompt).await?;
                    let mut publication = crate::historian::parse_publication(&publication_id, &source_ids, &raw)?;
                    publication.project_path = Some(project_path);
                    publication.external_fact_ids = external_fact_ids;
                    let report = crate::historian::HistorianReport {
                        kind: crate::historian::HistorianReportKind::Prepared,
                        publication: publication.clone(),
                        source_session_id,
                        source_branch_id,
                        model: agent.model().to_string(),
                        elapsed_ms: started_at.elapsed().as_millis().try_into().unwrap_or(u64::MAX),
                        usage,
                    };
                    child.lock().map_err(|_| anyhow!("historian child transcript poisoned"))?.record_assistant_message(serde_json::to_string(&report)?)?;
                    let summary = serde_json::json!({"status":"completed","summary":format!("Prepared {} history episodes and {} facts",publication.compartments.len(),publication.facts.len())}).to_string();
                    *produced_child.lock().map_err(|_| anyhow!("historian result poisoned"))? = Some(publication);
                    Ok(summary)
                }.boxed()
            }).await;
            let result = (|| -> Result<HistoryPublication> {
                let summary = result?;
                ensure!(
                    summary.status == SubagentStatus::Completed,
                    "historian ended with {}: {}",
                    summary.status.as_str(),
                    summary.summary
                );
                let publication = produced
                    .lock()
                    .map_err(|_| anyhow!("historian result poisoned"))?
                    .take()
                    .ok_or_else(|| anyhow!("historian produced no publication"))?;
                // Serialize cancellation acknowledgement with durable publication:
                // cancellation either wins first, or waits for the committed record.
                let cancelled_guard = cancelled
                    .lock()
                    .map_err(|_| anyhow!("historian cancellation state poisoned"))?;
                ensure!(!*cancelled_guard, "historian cancelled before publication");
                let mut recorder = recorder
                    .lock()
                    .map_err(|_| anyhow!("historian transcript poisoned"))?;
                ensure!(
                    recorder.session_id() == work.session_id,
                    "historian session changed"
                );
                ensure!(
                    recorder
                        .current_context_branch_id()
                        .unwrap_or(crate::transcript::ROOT_CONTEXT_BRANCH_ID)
                        == work.branch_id,
                    "historian branch changed"
                );
                recorder.record_history_event(
                    TranscriptEvent::HistoryPublished(publication.clone()),
                    work.revision,
                )?;
                Ok(publication)
            })();
            let failed = result.is_err();
            if let Err(error) = &result {
                tracing::warn!(error=%error,"historian background work failed");
            }
            if cancelled.lock().is_ok_and(|value| !*value) {
                let _ = event_tx.send(SessionTransportEvent::HistorianStatus {
                    session_id: work.session_id.clone(),
                    running: false,
                    failed,
                });
            }
            let _ = tx.send(result);
        });
        Ok(())
    }
    pub(crate) fn poll(
        &self,
        session: &str,
        branch: &str,
        revision: u64,
    ) -> Result<Option<HistoryPublication>> {
        let mut guard = self
            .pending
            .lock()
            .map_err(|_| anyhow!("historian state poisoned"))?;
        let Some(job) = guard.as_mut() else {
            return Ok(None);
        };
        if job.session_id != session || job.branch_id != branch || job.revision != revision {
            let id = job.run_id.clone();
            if let Ok(mut cancelled) = job.cancelled.lock() {
                *cancelled = true;
            }
            self.pool.cancel_run(&id);
            *guard = None;
            return Ok(None);
        }
        match job.receiver.try_recv() {
            Ok(result) => {
                *guard = None;
                result.map(Some)
            }
            Err(oneshot::error::TryRecvError::Empty) => Ok(None),
            Err(oneshot::error::TryRecvError::Closed) => {
                *guard = None;
                Err(anyhow!("historian task closed without a result"))
            }
        }
    }
}
impl Drop for HistorianRuntime {
    fn drop(&mut self) {
        if let Ok(pending) = self.pending.get_mut() {
            if let Some(job) = pending.take() {
                if let Ok(mut cancelled) = job.cancelled.lock() {
                    *cancelled = true;
                }
                self.pool.cancel_run(&job.run_id);
            }
        }
    }
}

#[cfg(test)]
mod project_fact_tests {
    use super::*;

    fn historian(
        sessions_dir: &std::path::Path,
        transcript_dir: &std::path::Path,
    ) -> HistorianRuntime {
        let recorder = TranscriptRecorder::create(transcript_dir).unwrap();
        let (event_tx, _event_rx) = tokio::sync::mpsc::unbounded_channel();
        HistorianRuntime::new(
            SubagentPool::new(),
            sessions_dir.to_owned(),
            Arc::new(Mutex::new(recorder)),
            event_tx,
        )
    }

    #[test]
    fn project_fact_recall_yields_while_waiting_for_journal_worker() {
        let root = tempfile::tempdir().unwrap();
        let transcript_dir = tempfile::tempdir().unwrap();
        crate::memory::project_fact_tests::write_session(root.path(), "a", "project", vec![]);
        crate::memory::project_fact_tests::write_session(root.path(), "b", "other", vec![]);
        let expected =
            crate::memory::recall_project_facts(root.path(), "project", "current").unwrap();
        assert!(!expected.is_empty());
        let historian = historian(root.path(), transcript_dir.path());
        let runtime = tokio::runtime::Builder::new_current_thread()
            .max_blocking_threads(1)
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(async {
            // Hold the journal worker queue so pending recall is deterministic,
            // independent of machine speed or the size of the fixture.
            let (release, wait) = std::sync::mpsc::channel();
            let worker = tokio::task::spawn_blocking(move || {
                wait.recv_timeout(std::time::Duration::from_secs(5))
            });
            let mut recall = Box::pin(historian.project_facts("project", "current"));
            let immediate = recall.as_mut().now_or_never();
            release.send(()).unwrap();
            assert!(
                immediate.is_none(),
                "journal replay must run off the async executor"
            );
            assert_eq!(recall.await.unwrap(), expected);
            worker.await.unwrap().unwrap();
        });
    }

    #[tokio::test]
    async fn project_fact_recall_propagates_journal_read_errors() {
        let root = tempfile::NamedTempFile::new().unwrap();
        let transcript_dir = tempfile::tempdir().unwrap();
        let historian = historian(root.path(), transcript_dir.path());
        let error = historian
            .project_facts("project", "current")
            .await
            .unwrap_err();
        assert!(
            error.downcast_ref::<std::io::Error>().is_some(),
            "{error:#}"
        );
    }
}
