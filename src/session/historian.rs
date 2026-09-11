//! Session-owned, single-flight Historian. Uses the same pool and one-shot
//! execution boundary as reviewer; only the host publishes validated artifacts.
use crate::agent::{Agent, SubagentInvocation};
use crate::context_history::HistoryPublication;
use crate::model_runtime::ModelFailure;
use crate::session::runner::{
    SessionTransportEvent, SessionTransportEventSender, subagent_event_sender,
};
use crate::session::{AssistantDeltaEvent, SessionEvent};
use crate::subagent::{SubagentPool, SubagentStatus};
use crate::tool::NormalizedSubagentInput;
use crate::transcript::{HistorianExchangeEvent, TranscriptEvent, TranscriptRecorder};
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
    work_id: String,
    source_len: usize,
    session_id: String,
    branch_id: String,
    revision: u64,
    receiver: oneshot::Receiver<Result<HistoryPublication>>,
    cancelled: Arc<Mutex<bool>>,
}

/// Terminal outcome of the most recent Historian attempt. The host keeps this so
/// an identical source prefix is not re-dispatched at every later request
/// boundary, and so an oversized prefix can shrink instead of failing forever.
#[derive(Clone)]
pub(crate) struct HistorianFailure {
    pub work_id: String,
    pub source_len: usize,
    pub message: String,
    pub oversized: bool,
}

pub(crate) struct HistorianRuntime {
    pool: SubagentPool,
    sessions_dir: std::path::PathBuf,
    transcript: Arc<Mutex<TranscriptRecorder>>,
    event_tx: SessionTransportEventSender,
    pending: Mutex<Option<Pending>>,
    /// Last terminal failure, cleared by the next successful publication.
    failure: Mutex<Option<HistorianFailure>>,
    /// Largest source prefix this session has not yet seen rejected as oversized.
    /// Only ever reduced within a session, so discovery work is not repeated.
    pass_limit: Mutex<Option<usize>>,
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
            failure: Mutex::new(None),
            pass_limit: Mutex::new(None),
        }
    }

    pub(crate) fn failure(&self) -> Option<HistorianFailure> {
        self.failure.lock().ok().and_then(|failure| failure.clone())
    }

    pub(crate) fn pass_limit(&self) -> Option<usize> {
        self.pass_limit.lock().ok().and_then(|limit| *limit)
    }

    pub(crate) fn failed_for(&self, work_id: &str) -> bool {
        self.failure
            .lock()
            .ok()
            .and_then(|failure| failure.as_ref().map(|failure| failure.work_id == work_id))
            .unwrap_or(false)
    }

    fn record_failure(&self, failure: HistorianFailure) {
        if let Ok(mut slot) = self.failure.lock() {
            *slot = Some(failure);
        }
    }

    fn clear_failure(&self) {
        if let Ok(mut slot) = self.failure.lock() {
            *slot = None;
        }
    }

    /// Halve the accepted prefix length after an oversized rejection. Takes the
    /// minimum so repeated observations of one failure stay idempotent.
    pub(crate) fn reduce_pass_limit(&self, source_len: usize) {
        let reduced = (source_len / 2).max(1);
        if let Ok(mut limit) = self.pass_limit.lock() {
            *limit = Some(limit.map_or(reduced, |current| current.min(reduced)));
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
        let exchange_run_id = run_id.clone();
        let (tx, rx) = oneshot::channel();
        let cancelled = Arc::new(Mutex::new(false));
        *pending = Some(Pending {
            run_id,
            work_id: work.id.clone(),
            source_len: work.source_ids.len(),
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
            let delta_tx = event_tx.clone();
            let result = pool.complete_started_run_with_executor(started, move |agent, _, child, _, child_session_id, _| {
                async move {
                    child.lock().map_err(|_| anyhow!("historian child transcript poisoned"))?.record_user_message(format!("Historian · {} history items\n\nModel: {}", source_ids.len(), agent.model()))?;
                    let started_at = std::time::Instant::now();
                    // Provisional text is surfaced on the existing child-session
                    // channel. A retried attempt starts a new observation, so the
                    // child view treats it as a separate message.
                    let delta_sender = delta_tx.clone();
                    let delta_child = child_session_id.clone();
                    let on_delta = move |delta: &str| {
                        let _ = delta_sender.send(SessionTransportEvent::ChildSessionEvent {
                            child_session_id: delta_child.clone(),
                            agent_name: Some("historian".to_string()),
                            parent_tool_call_id: None,
                            event: SessionEvent::AssistantDelta(AssistantDeltaEvent::new(delta)),
                        });
                        std::future::ready(Ok::<(), ModelFailure>(()))
                    };
                    let payload_bytes = serde_json::to_string(&input)
                        .map(|payload| payload.len() as u64)
                        .unwrap_or_default();
                    let outcome = async {
                        let (raw, usage) = match agent.run_historian(&input, on_delta).await {
                            Ok(exchange) => exchange,
                            Err(error) => {
                                record_exchange(&child, &exchange_run_id, source_ids.len(), payload_bytes, &[], Exchange::Failed(error.to_string()));
                                return Err(error);
                            }
                        };
                        let mut publication = match crate::historian::parse_publication(&publication_id, &source_ids, &raw) {
                            Ok(publication) => publication,
                            Err(error) => {
                                record_exchange(&child, &exchange_run_id, source_ids.len(), payload_bytes, &usage, Exchange::Rejected { error: error.to_string(), response: &raw });
                                return Err(error);
                            }
                        };
                        record_exchange(&child, &exchange_run_id, source_ids.len(), payload_bytes, &usage, Exchange::Prepared);
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
                        let report_json = serde_json::to_string(&report)?;
                        child.lock().map_err(|_| anyhow!("historian child transcript poisoned"))?.record_assistant_message(report_json.clone())?;
                        for event in [
                            SessionEvent::AssistantDone { message_id: None },
                            SessionEvent::AssistantDelta(AssistantDeltaEvent::new(report_json)),
                        ] {
                            let _ = delta_tx.send(SessionTransportEvent::ChildSessionEvent {
                                child_session_id: child_session_id.clone(),
                                agent_name: Some("historian".to_string()),
                                parent_tool_call_id: None,
                                event,
                            });
                        }
                        let summary = serde_json::json!({"status":"completed","summary":format!("Prepared {} history episodes",publication.compartments.len())}).to_string();
                        *produced_child.lock().map_err(|_| anyhow!("historian result poisoned"))? = Some(publication);
                        Ok(summary)
                    }
                    .await;
                    // Finalize the child view stream for every outcome; the host
                    // reports failure separately through HistorianStatus.
                    let _ = delta_tx.send(SessionTransportEvent::ChildSessionEvent {
                        child_session_id: child_session_id.clone(),
                        agent_name: Some("historian".to_string()),
                        parent_tool_call_id: None,
                        event: SessionEvent::AssistantDone { message_id: None },
                    });
                    let _ = delta_tx.send(SessionTransportEvent::ChildSessionEvent {
                        child_session_id,
                        agent_name: Some("historian".to_string()),
                        parent_tool_call_id: None,
                        event: SessionEvent::Done,
                    });
                    outcome
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
                let work_id = job.work_id.clone();
                let source_len = job.source_len;
                *guard = None;
                match result {
                    Ok(publication) => {
                        self.clear_failure();
                        Ok(Some(publication))
                    }
                    Err(error) => {
                        let message = format!("{error:#}");
                        self.record_failure(HistorianFailure {
                            work_id,
                            source_len,
                            oversized: crate::historian::is_context_overflow(&message),
                            message,
                        });
                        Err(error)
                    }
                }
            }
            Err(oneshot::error::TryRecvError::Empty) => Ok(None),
            Err(oneshot::error::TryRecvError::Closed) => {
                *guard = None;
                Err(anyhow!("historian task closed without a result"))
            }
        }
    }
}

const EXCHANGE_RESPONSE_CHARS: usize = 32_768;

enum Exchange<'a> {
    Prepared,
    Rejected { error: String, response: &'a str },
    Failed(String),
}

/// Diagnostic failures are logged, never propagated.
fn record_exchange(
    child: &Arc<Mutex<TranscriptRecorder>>,
    run_id: &str,
    source_count: usize,
    payload_bytes: u64,
    usage: &[crate::historian::UsageUpdate],
    exchange: Exchange<'_>,
) {
    let (outcome, error, response) = match exchange {
        Exchange::Prepared => ("prepared", None, None),
        Exchange::Rejected { error, response } => ("rejected", Some(error), Some(response)),
        Exchange::Failed(error) => ("failed", Some(error), None),
    };
    let tokens = usage.iter().rev().find_map(|update| match update {
        crate::historian::UsageUpdate::Usage {
            input_tokens,
            output_tokens,
            ..
        } => Some((*input_tokens, *output_tokens)),
        crate::historian::UsageUpdate::Cache { .. } => None,
    });
    let event = HistorianExchangeEvent {
        run_id: run_id.to_string(),
        source_count,
        payload_bytes,
        input_tokens: tokens.map(|(input, _)| input),
        output_tokens: tokens.map(|(_, output)| output),
        response_chars: response.map(|text| text.chars().count() as u64),
        outcome: outcome.to_string(),
        error,
        response: response.map(exchange_response),
    };
    let recorded = child
        .lock()
        .map_err(|_| anyhow!("historian child transcript poisoned"))
        .and_then(|mut recorder| recorder.record_historian_exchange(event));
    if let Err(error) = recorded {
        tracing::warn!(error = %error, "historian exchange diagnostics not recorded");
    }
}

fn exchange_response(raw: &str) -> String {
    if raw.chars().count() <= EXCHANGE_RESPONSE_CHARS {
        return raw.to_string();
    }
    format!(
        "{}…[response truncated]…{}",
        crate::historian::raw_excerpt(raw),
        crate::historian::raw_tail(raw)
    )
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

#[cfg(test)]
mod tests {
    use super::*;

    fn runtime(dir: &std::path::Path) -> HistorianRuntime {
        let recorder = TranscriptRecorder::create(dir).expect("recorder");
        let (event_tx, _event_rx) = tokio::sync::mpsc::unbounded_channel();
        HistorianRuntime::new(
            SubagentPool::new(),
            dir.to_path_buf(),
            Arc::new(Mutex::new(recorder)),
            event_tx,
        )
    }

    #[test]
    fn pass_limit_only_shrinks_as_oversized_rejections_repeat() {
        let dir = tempfile::tempdir().expect("temp dir");
        let runtime = runtime(dir.path());
        assert_eq!(runtime.pass_limit(), None);
        runtime.reduce_pass_limit(600);
        assert_eq!(runtime.pass_limit(), Some(300));
        runtime.reduce_pass_limit(600);
        assert_eq!(runtime.pass_limit(), Some(300));
        runtime.reduce_pass_limit(80);
        assert_eq!(runtime.pass_limit(), Some(40));
        runtime.reduce_pass_limit(1);
        assert_eq!(runtime.pass_limit(), Some(1));
    }

    #[test]
    fn failure_tracking_is_scoped_to_the_failed_work_identity() {
        let dir = tempfile::tempdir().expect("temp dir");
        let runtime = runtime(dir.path());
        assert!(!runtime.failed_for("work-1"));
        runtime.record_failure(HistorianFailure {
            work_id: "work-1".into(),
            source_len: 600,
            message: "cancelled".into(),
            oversized: false,
        });
        assert!(runtime.failed_for("work-1"));
        assert!(!runtime.failed_for("work-2"));
        assert_eq!(
            runtime.failure().map(|failure| failure.source_len),
            Some(600)
        );
        runtime.clear_failure();
        assert!(runtime.failure().is_none());
    }
}
