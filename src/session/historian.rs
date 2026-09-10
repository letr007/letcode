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
    pub input: crate::user_content::UserMessageContent,
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

    pub(crate) fn is_running(&self) -> bool {
        self.pending.lock().is_ok_and(|pending| pending.is_some())
    }
    pub(crate) fn cancel(&self) {
        if let Ok(mut pending) = self.pending.lock()
            && let Some(job) = pending.take()
        {
            if let Ok(mut cancelled) = job.cancelled.lock() {
                *cancelled = true;
            }
            self.pool.cancel_run(&job.run_id);
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
            let input = work.input.clone();
            let result = pool.complete_started_run_with_executor(started, move |agent, _, child, _, _, _| {
                async move {
                    child.lock().map_err(|_| anyhow!("historian child transcript poisoned"))?.record_user_message(format!("Historian · {} history items\n\nModel: {}", source_ids.len(), agent.model()))?;
                    let started_at = std::time::Instant::now();
                    let (raw, usage) = agent.run_historian(&input).await?;
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
                    let summary = serde_json::json!({"status":"completed","summary":format!("Prepared {} history episodes",publication.compartments.len())}).to_string();
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
        if let Ok(pending) = self.pending.get_mut()
            && let Some(job) = pending.take()
        {
            if let Ok(mut cancelled) = job.cancelled.lock() {
                *cancelled = true;
            }
            self.pool.cancel_run(&job.run_id);
        }
    }
}
