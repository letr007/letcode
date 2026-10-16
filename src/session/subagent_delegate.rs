use std::path::PathBuf;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use anyhow::{Result, anyhow, bail};
use futures_util::future::BoxFuture;
use serde_json::json;

use crate::agent::{
    Agent, SubagentDelegate, SubagentInvocation, subagent_tool_name_for_agent_name,
};
use crate::session::child_view::ChildLiveText;
use crate::session::engine::{SessionEngineCommand, SessionEngineControl};
use crate::subagent::{SubagentFailureKind, SubagentJob, SubagentPool, SubagentStatus};
use crate::tool::ToolResult;
use crate::transcript::{TranscriptEvent, TranscriptRecorder};

use super::events::SessionTransportEventSender;
use super::formatting::compact_subagent_summary;
use super::subagent_event_sender;

type ChildCompletions =
    Arc<Mutex<Vec<std::result::Result<crate::subagent::SubagentRunSummary, String>>>>;

#[derive(Clone)]
pub(super) struct RunnerSubagentDelegate {
    pub(super) child_completions: Option<ChildCompletions>,
    pub(super) runtime: SubagentPool,
    pub(super) sessions_dir: PathBuf,
    pub(super) transcript: Arc<Mutex<TranscriptRecorder>>,
    pub(super) event_tx: Option<SessionTransportEventSender>,
    pub(super) background_event_tx: Option<SessionTransportEventSender>,
    pub(super) child_live_text: Option<ChildLiveText>,
    #[cfg(test)]
    pub(super) background_child_started_tx: Option<tokio::sync::mpsc::UnboundedSender<String>>,
    pub(super) route_api_key_configured: indexmap::IndexMap<String, bool>,
    pub(super) provider_api_key_hints: indexmap::IndexMap<String, String>,
    pub(super) api_key_hint: String,
    pub(super) background_control_tx:
        Option<tokio::sync::mpsc::UnboundedSender<SessionEngineControl>>,
}

impl RunnerSubagentDelegate {
    fn hard_failure_result(
        tool_name: &str,
        agent_name: &str,
        child_session_id: Option<String>,
        summary: String,
    ) -> ToolResult {
        let error_message = summary.clone();
        let data = json!({
            "agent_name": agent_name,
            "child_session_id": child_session_id,
            "status": SubagentStatus::Failed.as_str(),
            "failure_kind": SubagentFailureKind::Hard.as_str(),
            "summary": compact_subagent_summary(&summary),
            "full_summary": summary,
            "active": false,
        });
        ToolResult::err_with_data(tool_name, error_message, data)
    }

    fn capacity_result(
        tool_name: &str,
        agent_name: Option<&str>,
        session_id: Option<String>,
        error: &crate::subagent::SubagentCapacityExceeded,
    ) -> ToolResult {
        ToolResult::err_with_data(
            tool_name,
            error.to_string(),
            json!({
                "agent_name": agent_name,
                "session_id": session_id,
                "status": "not_started",
                "started": false,
                "active": false,
                "accepted": false,
                "reason": "concurrency_limit",
                "max_concurrent": error.limit,
                "summary": error.to_string(),
            }),
        )
    }

    fn route_display_name(
        &self,
        parent: &Agent,
        agent_name: &str,
        invocation: &SubagentInvocation,
    ) -> Result<String> {
        let template = crate::agent::AgentTemplate::from_name(agent_name)
            .ok_or_else(|| anyhow!("unknown subagent template: {agent_name}"))?;
        if let Some(route) = &invocation.model {
            return crate::agent::AgentFactory::resolve_subagent_route(
                parent,
                &template,
                Some(route),
                false,
            )
            .map(|route| route.display_name());
        }
        let Some(target_child_session_id) = invocation.input.target_child_session_id.as_deref()
        else {
            return crate::agent::AgentFactory::resolve_subagent_route(
                parent, &template, None, false,
            )
            .map(|route| route.display_name());
        };
        let parent_records = self
            .transcript
            .lock()
            .map_err(|_| anyhow!("transcript recorder poisoned"))
            .and_then(|recorder| crate::transcript::read_records(recorder.path()))?;
        let child = SubagentPool::child_sessions(&self.sessions_dir, &parent_records)
            .into_iter()
            .find(|child| child.child_session_id == target_child_session_id)
            .ok_or_else(|| {
                anyhow!(
                    "takeover failed: child_session_id `{target_child_session_id}` is not a known child of this parent"
                )
            })?;
        if child.agent_name != agent_name {
            bail!(
                "takeover failed: child `{target_child_session_id}` is agent `{}`, expected `{agent_name}`",
                child.agent_name
            );
        }
        let child_records = crate::transcript::read_records_allow_partial_tail(
            crate::transcript::child_sessions_dir(&self.sessions_dir)
                .join(format!("{target_child_session_id}.jsonl")),
        )?;
        let recorded_route =
            crate::transcript::restore_latest_model(&child_records).ok_or_else(|| {
                anyhow!(
                    "takeover failed: child `{target_child_session_id}` has no recorded model route"
                )
            })?;
        let recorded_route = crate::config::ModelRoute::parse(&recorded_route)?;
        crate::agent::AgentFactory::resolve_subagent_route(
            parent,
            &template,
            Some(&recorded_route),
            true,
        )
        .map(|route| route.display_name())
    }

    fn missing_api_key_result(
        &self,
        tool_name: &str,
        agent_name: &str,
        route_display_name: String,
    ) -> ToolResult {
        let provider = route_display_name
            .split_once('/')
            .map(|(provider, _)| provider)
            .unwrap_or("selected");
        let hint = self
            .provider_api_key_hints
            .get(provider)
            .cloned()
            .unwrap_or_else(|| self.api_key_hint.clone());
        let summary = format!("API key is not set for the selected provider. {hint}");
        let data = json!({
            "agent_name": agent_name,
            "route": route_display_name,
            "status": SubagentStatus::Failed.as_str(),
            "failure_kind": SubagentFailureKind::Hard.as_str(),
            "summary": summary,
            "full_summary": summary,
            "active": false,
        });
        ToolResult::err_with_data(tool_name, summary, data)
    }

    fn job(&self, run_id: &str) -> Result<Option<SubagentJob>> {
        let parent_records = self.parent_records()?;
        if crate::transcript::restore_subagents_policy(&parent_records)?.mode
            == crate::config::SubagentMode::Swarm
        {
            let snapshot = self
                .runtime
                .swarm_snapshot(&self.sessions_dir, &parent_records)?;
            let caller = parent_records
                .first()
                .ok_or_else(|| anyhow!("missing caller session"))?
                .session_id
                .as_str();
            let Some(run) = snapshot
                .runs
                .iter()
                .find(|run| run.run_id.as_deref() == Some(run_id))
            else {
                return Ok(None);
            };
            let mut ancestor = snapshot
                .members
                .get(&run.session_id)
                .map(|member| member.parent_session_id.as_str());
            let mut owned = false;
            while let Some(parent) = ancestor {
                if parent == caller {
                    owned = true;
                    break;
                }
                ancestor = snapshot
                    .members
                    .get(parent)
                    .map(|member| member.parent_session_id.as_str());
            }
            if !owned {
                return Ok(None);
            }
            return Ok(Some(SubagentJob {
                active: run.active,
                run_id: run_id.into(),
                child_session_id: run.session_id.clone(),
                agent_name: run.role.clone(),
                status: run.status.clone(),
                summary: self
                    .runtime
                    .completed_result(run_id)
                    .map_or_else(String::new, |result| result.summary),
                pool_ordinal: run.pool_ordinal.unwrap_or(0),
            }));
        }
        if let Some(result) = self.runtime.completed_result(run_id) {
            let pool_ordinal = pool_ordinal_for_run(&parent_records, run_id);
            if pool_ordinal > 0 {
                let mut job = SubagentJob::from_result(result);
                job.pool_ordinal = pool_ordinal;
                return Ok(Some(job));
            }
        }
        if let Some(job) = self.runtime.active_jobs().into_iter().find(|job| {
            job.run_id == run_id && pool_ordinal_for_run(&parent_records, &job.run_id) > 0
        }) {
            return Ok(Some(job));
        }
        Ok(
            crate::transcript::project_subagent_jobs(&self.sessions_dir, &parent_records)?
                .into_iter()
                .find(|job| job.run_id == run_id),
        )
    }

    fn parent_records(&self) -> Result<Vec<crate::transcript::TranscriptRecord>> {
        self.transcript
            .lock()
            .map_err(|_| anyhow!("transcript recorder poisoned"))
            .and_then(|recorder| {
                crate::transcript::read_records_allow_partial_tail(recorder.path())
            })
    }
}

fn pool_ordinal_for_run(records: &[crate::transcript::TranscriptRecord], run_id: &str) -> u32 {
    records
        .iter()
        .find_map(|record| match &record.event {
            TranscriptEvent::SubagentStarted {
                run_id: recorded,
                pool_ordinal,
                ..
            } if recorded == run_id => Some(*pool_ordinal),
            _ => None,
        })
        .unwrap_or(0)
}

fn control_run_id<'a>(tool_name: &str, args: &'a serde_json::Value) -> Result<&'a str> {
    args.get("run_id")
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|run_id| !run_id.is_empty())
        .ok_or_else(|| anyhow!("{tool_name} requires non-empty run_id"))
}

fn swarm_status_selector(args: &serde_json::Value) -> Result<(&str, &str)> {
    let object = args
        .as_object()
        .ok_or_else(|| anyhow!("agent__status requires an object"))?;
    anyhow::ensure!(
        object.len() == 1,
        "agent__status requires exactly one of run_id or session_id"
    );
    let (key, value) = object.iter().next().expect("one selector");
    anyhow::ensure!(
        key == "run_id" || key == "session_id",
        "agent__status requires run_id or session_id"
    );
    let value = value
        .as_str()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| anyhow!("agent__status requires a non-empty {key}"))?;
    Ok((key, value))
}

fn merge_live_jobs(
    jobs: &mut Vec<SubagentJob>,
    live: Vec<SubagentJob>,
    parent_records: &[crate::transcript::TranscriptRecord],
) {
    for live_job in live {
        if pool_ordinal_for_run(parent_records, &live_job.run_id) == 0 {
            continue;
        }
        if let Some(job) = jobs.iter_mut().find(|job| job.run_id == live_job.run_id) {
            *job = live_job;
        } else {
            jobs.push(live_job);
        }
    }
    jobs.sort_by_key(|job| (job.pool_ordinal, job.run_id.clone()));
}

impl SubagentDelegate for RunnerSubagentDelegate {
    fn prepare(&self, agent: &Agent) -> Result<()> {
        if agent.subagents_policy().mode != crate::config::SubagentMode::Swarm {
            return Ok(());
        }
        let records = self.parent_records()?;
        if crate::subagent::swarm::caller_identity(&records)?.is_some() {
            return Ok(());
        }
        let event_tx = self
            .background_event_tx
            .clone()
            .or_else(|| self.event_tx.clone());
        let event_sender =
            event_tx.map(|tx| subagent_event_sender((tx, self.child_live_text.clone())));
        let Some(control_tx) = self.background_control_tx.clone() else {
            return Ok(());
        };
        let sessions_dir = self.sessions_dir.clone();
        let route_keys = self.route_api_key_configured.clone();
        let hints = self.provider_api_key_hints.clone();
        let hint = self.api_key_hint.clone();
        let event_tx = self.event_tx.clone();
        let background_event_tx = self.background_event_tx.clone();
        let live_text = self.child_live_text.clone();
        let make_delegate: crate::subagent::MessageDelegateFactory =
            Arc::new(move |runtime, transcript| {
                Ok(Arc::new(RunnerSubagentDelegate {
                    child_completions: Some(Arc::new(Mutex::new(Vec::new()))),
                    runtime,
                    sessions_dir: sessions_dir.clone(),
                    transcript,
                    event_tx: event_tx.clone(),
                    background_event_tx: background_event_tx.clone(),
                    child_live_text: live_text.clone(),
                    #[cfg(test)]
                    background_child_started_tx: None,
                    route_api_key_configured: route_keys.clone(),
                    provider_api_key_hints: hints.clone(),
                    api_key_hint: hint.clone(),
                    background_control_tx: None,
                }))
            });
        let generation = self.runtime.swarm_snapshot(&self.sessions_dir, &records)?;
        let generation = self
            .runtime
            .swarm_generation(&generation.root.session_id)
            .expect("restored authority");
        self.runtime.install_message_host(
            &records,
            crate::subagent::MessageHost {
                seed: Mutex::new(crate::agent::AgentFactory::launch_seed(agent)),
                sessions_dir: self.sessions_dir.clone(),
                root_transcript: self.transcript.clone(),
                delegate: make_delegate,
                event_sender,
                wake_root: Arc::new(move |root_session_id| {
                    control_tx
                        .send(SessionEngineControl::Command(
                            SessionEngineCommand::SwarmMessageReady {
                                root_session_id,
                                generation,
                            },
                        ))
                        .map_err(|_| anyhow!("root execution channel closed"))
                }),
            },
        )
    }

    fn has_user_message(&self) -> Result<bool> {
        let records = self.parent_records()?;
        let Some(first) = records.first() else {
            return Ok(false);
        };
        if crate::transcript::restore_subagents_policy(&records)?.mode
            != crate::config::SubagentMode::Swarm
        {
            return Ok(false);
        }
        let identity = crate::subagent::swarm::caller_identity(&records)?;
        let root = identity
            .as_ref()
            .map_or(first.session_id.as_str(), |member| {
                member.root_session_id.as_str()
            });
        self.runtime.member_message_pending(root, &first.session_id)
    }

    fn take_user_message(&self) -> Result<Option<String>> {
        let records = self.parent_records()?;
        let Some(first) = records.first() else {
            return Ok(None);
        };
        if crate::transcript::restore_subagents_policy(&records)?.mode
            != crate::config::SubagentMode::Swarm
        {
            return Ok(None);
        }
        let identity = crate::subagent::swarm::caller_identity(&records)?;
        let root = identity
            .as_ref()
            .map_or(first.session_id.as_str(), |member| {
                member.root_session_id.as_str()
            });
        self.runtime.take_member_message(root, &first.session_id)
    }

    fn send<'a>(
        &'a self,
        parent: &'a Agent,
        args: &'a serde_json::Value,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<ToolResult>> + Send + 'a>> {
        Box::pin(async move {
            self.prepare(parent)?;
            let object = args
                .as_object()
                .ok_or_else(|| anyhow!("agent__send requires an object"))?;
            anyhow::ensure!(
                object.len() == 2,
                "agent__send requires session_id and message"
            );
            let target = args["session_id"]
                .as_str()
                .filter(|target| !target.trim().is_empty())
                .ok_or_else(|| anyhow!("session_id must not be empty"))?;
            let message = args["message"]
                .as_str()
                .ok_or_else(|| anyhow!("message must be a string"))?;
            let pending = match self.runtime.enqueue_member_message(
                &self.sessions_dir,
                &self.parent_records()?,
                target,
                message,
            ) {
                Ok(pending) => pending,
                Err(error) => {
                    if let Some(capacity) =
                        error.downcast_ref::<crate::subagent::SubagentCapacityExceeded>()
                    {
                        return Ok(Self::capacity_result(
                            crate::tool_names::TOOL_AGENT_SEND,
                            None,
                            Some(target.to_string()),
                            capacity,
                        ));
                    }
                    return Err(error);
                }
            };
            Ok(ToolResult::ok(
                crate::tool_names::TOOL_AGENT_SEND,
                json!({"session_id": target, "accepted":true,"queued":pending}),
            ))
        })
    }

    fn for_child(
        &self,
        transcript: Arc<Mutex<TranscriptRecorder>>,
    ) -> Result<Arc<dyn SubagentDelegate>> {
        let mut delegate = self.clone();
        delegate.transcript = transcript;
        delegate.child_completions = (self.background_control_tx.is_some()
            || self.child_completions.is_some())
        .then(|| Arc::new(Mutex::new(Vec::new())));
        delegate.background_control_tx = None;
        Ok(Arc::new(delegate))
    }

    fn can_delegate(&self) -> bool {
        self.parent_records()
            .and_then(|records| {
                if crate::transcript::restore_subagents_policy(&records)?.mode
                    != crate::config::SubagentMode::Swarm
                {
                    return Ok(true);
                }
                self.runtime
                    .swarm_can_delegate(&self.sessions_dir, &records)
            })
            .unwrap_or(false)
    }

    fn take_completed(&self) -> Result<Vec<crate::subagent::SubagentRunSummary>> {
        let Some(queue) = &self.child_completions else {
            return Ok(Vec::new());
        };
        let caller = self
            .transcript
            .lock()
            .map_err(|_| anyhow!("transcript recorder poisoned"))?
            .session_id()
            .to_string();
        if self.runtime.member_scope_stopped(&caller) {
            return Ok(Vec::new());
        }
        let mut queue = queue
            .lock()
            .map_err(|_| anyhow!("child completion queue poisoned"))?;
        let mut completed = Vec::new();
        for result in queue.drain(..) {
            let result = result.map_err(|error| anyhow!("background subagent failed: {error}"))?;
            if self.runtime.is_foregrounded(&result.run_id) {
                continue;
            }
            self.transcript
                .lock()
                .map_err(|_| anyhow!("transcript recorder poisoned"))?
                .record_internal_continuation(
                    crate::session::engine::format_background_subagent_completion(&result),
                    crate::transcript::InternalContinuationSource::SubagentCompletion,
                )?;
            completed.push(result);
        }
        Ok(completed)
    }

    fn control<'a>(
        &'a self,
        tool_name: &'a str,
        args: &'a serde_json::Value,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<ToolResult>> + Send + 'a>> {
        Box::pin(async move {
            match tool_name {
                crate::tool_names::TOOL_AGENT_JOBS => {
                    let parent_records = self.parent_records()?;
                    if crate::transcript::restore_subagents_policy(&parent_records)?.mode
                        == crate::config::SubagentMode::Swarm
                    {
                        let snapshot = self
                            .runtime
                            .swarm_snapshot(&self.sessions_dir, &parent_records)?;
                        return Ok(ToolResult::ok(
                            tool_name,
                            json!({"jobs": snapshot.latest_jobs()}),
                        ));
                    }
                    let mut jobs = crate::transcript::project_subagent_jobs(
                        &self.sessions_dir,
                        &parent_records,
                    )?;
                    merge_live_jobs(&mut jobs, self.runtime.active_jobs(), &parent_records);
                    Ok(ToolResult::ok(tool_name, json!({"jobs": jobs})))
                }
                crate::tool_names::TOOL_AGENT_STATUS => {
                    let records = self.parent_records()?;
                    if crate::transcript::restore_subagents_policy(&records)?.mode
                        == crate::config::SubagentMode::Swarm
                    {
                        let (key, value) = swarm_status_selector(args)?;
                        let snapshot = self.runtime.swarm_snapshot(&self.sessions_dir, &records)?;
                        let job = if key == "run_id" {
                            snapshot
                                .runs
                                .iter()
                                .find(|job| job.run_id.as_deref() == Some(value))
                                .cloned()
                        } else {
                            snapshot
                                .latest_jobs()
                                .into_iter()
                                .find(|job| job.session_id == value)
                        };
                        return match job {
                            Some(job) => Ok(ToolResult::ok(tool_name, serde_json::to_value(job)?)),
                            None => Ok(ToolResult::err(
                                tool_name,
                                format!("unknown swarm {key}: {value}"),
                            )),
                        };
                    }
                    let run_id = control_run_id(tool_name, args)?;
                    let job = self.job(run_id)?;
                    match job {
                        Some(job) => Ok(ToolResult::ok(tool_name, serde_json::to_value(job)?)),
                        None => Ok(ToolResult::err(
                            tool_name,
                            format!("unknown subagent run_id: {run_id}"),
                        )),
                    }
                }
                crate::tool_names::TOOL_AGENT_WAIT => {
                    let run_id = control_run_id(tool_name, args)?;
                    let Some(job) = self.job(run_id)? else {
                        return Ok(ToolResult::err(
                            tool_name,
                            format!("unknown subagent run_id: {run_id}"),
                        ));
                    };
                    if !job.active {
                        return Ok(ToolResult::err(
                            tool_name,
                            format!("subagent run is already terminal: {run_id}"),
                        ));
                    }
                    let Some(mut foreground) = self.runtime.claim_foreground(run_id)? else {
                        return Ok(ToolResult::err(
                            tool_name,
                            format!("subagent run is already terminal or foregrounded: {run_id}"),
                        ));
                    };
                    let result = self.runtime.wait_for_result(run_id).await?;
                    match result {
                        Some(result) => {
                            foreground.retain();
                            Ok(ToolResult::ok(
                                tool_name,
                                json!({
                                    "run_id": result.run_id,
                                    "child_session_id": result.child_session_id,
                                    "agent_name": result.agent_name,
                                    "status": result.status.as_str(),
                                    "failure_kind": result.failure_kind.map(|kind| kind.as_str()),
                                    "summary": result.summary,
                                    "structured_result": result.structured_result,
                                    "active": false,
                                }),
                            ))
                        }
                        None => {
                            self.runtime.release_foreground(run_id);
                            Ok(ToolResult::err(
                                tool_name,
                                format!("subagent run ended without a terminal result: {run_id}"),
                            ))
                        }
                    }
                }
                crate::tool_names::TOOL_AGENT_CANCEL => {
                    let run_id = control_run_id(tool_name, args)?;
                    if self.job(run_id)?.is_some() && self.runtime.cancel_run(run_id) {
                        Ok(ToolResult::ok(
                            tool_name,
                            json!({"run_id": run_id, "cancellation_requested": true}),
                        ))
                    } else if let Some(job) = self.job(run_id)? {
                        Ok(ToolResult::ok(
                            tool_name,
                            json!({"run_id": run_id, "cancellation_requested": false, "job": job}),
                        ))
                    } else {
                        Ok(ToolResult::err(
                            tool_name,
                            format!("unknown subagent run_id: {run_id}"),
                        ))
                    }
                }
                _ => Ok(ToolResult::err(
                    tool_name,
                    format!("unknown subagent control tool: {tool_name}"),
                )),
            }
        })
    }

    fn run_named(
        &self,
        parent: &Agent,
        agent_name: &str,
        invocation: SubagentInvocation,
    ) -> BoxFuture<'static, Result<ToolResult>> {
        let ready = |result| -> BoxFuture<'static, Result<ToolResult>> {
            Box::pin(std::future::ready(Ok(result)))
        };
        let prepared = (|| -> Result<BoxFuture<'static, Result<ToolResult>>> {
            self.prepare(parent)?;
            let tool_name = subagent_tool_name_for_agent_name(agent_name)
                .expect("runner dispatched unknown subagent agent name");
            let route_display_name = match self.route_display_name(parent, agent_name, &invocation)
            {
                Ok(route) => route,
                Err(error) => {
                    return Ok(ready(Self::hard_failure_result(
                        tool_name,
                        agent_name,
                        invocation.input.target_child_session_id,
                        error.to_string(),
                    )));
                }
            };
            if !self
                .route_api_key_configured
                .get(&route_display_name)
                .copied()
                .unwrap_or(false)
            {
                return Ok(ready(self.missing_api_key_result(
                    tool_name,
                    agent_name,
                    route_display_name,
                )));
            }
            let target_child_session_id = invocation.input.target_child_session_id.clone();
            let parent_session_id = match self.transcript.lock() {
                Ok(recorder) => recorder.session_id().to_string(),
                Err(_) => {
                    let summary = "transcript recorder poisoned".to_string();
                    let data = json!({
                        "agent_name": agent_name,
                        "child_session_id": target_child_session_id,
                        "status": SubagentStatus::Failed.as_str(),
                        "failure_kind": SubagentFailureKind::Hard.as_str(),
                        "summary": summary,
                        "full_summary": summary,
                        "active": false,
                    });
                    return Ok(ready(ToolResult::err_with_data(tool_name, summary, data)));
                }
            };
            let parent_turn_id = format!(
                "turn-{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis()
            );
            let background = invocation.input.background;
            if background
                && self.background_control_tx.is_none()
                && self.child_completions.is_none()
            {
                return Ok(ready(ToolResult::err(
                    tool_name,
                    "background execution is unavailable in this runtime",
                )));
            }
            let parent_tool_call_id = invocation.parent_tool_call_id.clone();
            let started = self.runtime.start_named_governed(
                parent,
                agent_name,
                invocation,
                self.sessions_dir.clone(),
                parent_session_id.clone(),
                parent_turn_id,
                Some(self.transcript.clone()),
                if background {
                    self.background_event_tx.clone()
                } else {
                    self.event_tx.clone()
                }
                .map(|event_tx| subagent_event_sender((event_tx, self.child_live_text.clone()))),
            );
            let started = match started {
                Ok(started) => started,
                Err(error) => {
                    if let Some(capacity) =
                        error.downcast_ref::<crate::subagent::SubagentCapacityExceeded>()
                    {
                        return Ok(ready(Self::capacity_result(
                            tool_name,
                            Some(agent_name),
                            target_child_session_id,
                            capacity,
                        )));
                    }
                    return Ok(ready(Self::hard_failure_result(
                        tool_name,
                        agent_name,
                        target_child_session_id,
                        error.to_string(),
                    )));
                }
            };

            if background {
                #[cfg(test)]
                if let Some(started_tx) = &self.background_child_started_tx {
                    let _ = started_tx.send(started.receipt().child_session_id.clone());
                }
                let run_id = started.run_id().to_string();
                let receipt = started.receipt().clone();
                let runtime = self.runtime.clone();
                let generation = runtime.swarm_generation(&parent_session_id);
                let control_tx = self.background_control_tx.clone();
                let child_completions = self.child_completions.clone();
                return Ok(Box::pin(async move {
                    tokio::spawn(async move {
                        let result = runtime
                            .complete_started_run(started)
                            .await
                            .map_err(|error| error.to_string());
                        if let Some(queue) = child_completions {
                            if let Ok(mut queue) = queue.lock() {
                                queue.push(result);
                            }
                        } else if let Some(control_tx) = control_tx {
                            let _ = control_tx.send(SessionEngineControl::Command(
                                SessionEngineCommand::BackgroundSubagentCompleted {
                                    generation,
                                    parent_session_id,
                                    parent_tool_call_id,
                                    result,
                                },
                            ));
                        }
                    });
                    Ok(ToolResult::ok(
                        tool_name,
                        json!({
                            "run_id": run_id,
                            "child_session_id": receipt.child_session_id,
                            "agent_name": receipt.agent_name,
                            "status": SubagentStatus::Running.as_str(),
                            "summary": receipt.summary,
                            "active": true,
                            "background": true,
                        }),
                    ))
                }));
            }

            let runtime = self.runtime.clone();
            let agent_name = agent_name.to_string();
            Ok(Box::pin(async move {
                let summary = runtime.complete_started_run(started).await;
                let summary = match summary {
                    Ok(summary) => summary,
                    Err(error) => {
                        return Ok(Self::hard_failure_result(
                            tool_name,
                            &agent_name,
                            target_child_session_id,
                            error.to_string(),
                        ));
                    }
                };

                let status = summary.status;
                let failure_kind = summary.failure_kind;
                let summary_text = summary.summary.clone();
                let compact_summary = compact_subagent_summary(&summary.summary);

                let data = json!({
                    "run_id": summary.run_id,
                    "child_session_id": summary.child_session_id,
                    "agent_name": summary.agent_name,
                    "status": status.as_str(),
                    "failure_kind": failure_kind.map(|kind| kind.as_str()),
                    "summary": compact_summary,
                    "full_summary": summary.summary,
                    "structured_result": summary.structured_result,
                    "active": false,
                });

                if status == SubagentStatus::Completed {
                    Ok(ToolResult::ok(tool_name, data))
                } else {
                    Ok(ToolResult::err_with_data(tool_name, summary_text, data))
                }
            }))
        })();
        Box::pin(async move { prepared?.await })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn swarm_discovery_restores_tree_identity_and_keeps_run_history() {
        use crate::config::{SubagentMode, SubagentsConfig};
        use crate::subagent::SwarmMember;
        use crate::transcript::{child_sessions_dir, read_records};
        let dir = tempfile::tempdir().unwrap();
        let policy = SubagentsConfig {
            mode: SubagentMode::Swarm,
            ..Default::default()
        };
        let mut root = TranscriptRecorder::create(dir.path()).unwrap();
        root.record_session_started("gpt-test").unwrap();
        root.record_subagents_policy(&policy).unwrap();
        let root_id = root.session_id().to_string();
        let root_path = root.path().to_path_buf();

        fn record_run(
            parent: &mut TranscriptRecorder,
            child: &mut TranscriptRecorder,
            member: &SwarmMember,
            run: &str,
            status: &str,
        ) {
            let parent_id = parent.session_id().to_string();
            for recorder in [&mut *parent, &mut *child] {
                recorder
                    .record_subagent_started(
                        run,
                        &parent_id,
                        "turn-1",
                        &member.child_session_id,
                        &member.agent_name,
                        "private task",
                        member.pool_ordinal,
                    )
                    .unwrap();
            }
            child
                .record_subagent_activation(run.to_string(), member.clone())
                .unwrap();
            child
                .record_subagent_lifecycle(
                    run,
                    &parent_id,
                    "turn-1",
                    &member.agent_name,
                    status,
                    Some("private detail".into()),
                )
                .unwrap();
            parent
                .record_subagent_result_structured(
                    run,
                    &parent_id,
                    "turn-1",
                    &member.child_session_id,
                    &member.agent_name,
                    status,
                    "private summary",
                    None,
                )
                .unwrap();
        }
        let mut children: Vec<TranscriptRecorder> = Vec::new();
        let mut members: Vec<SwarmMember> = Vec::new();
        for ordinal in 1..=3 {
            let mut child = TranscriptRecorder::create(child_sessions_dir(dir.path())).unwrap();
            child.record_session_started("gpt-test").unwrap();
            child.record_subagents_policy(&policy).unwrap();
            let member = SwarmMember {
                root_session_id: root_id.clone(),
                parent_session_id: if ordinal == 3 {
                    members[0].child_session_id.clone()
                } else {
                    root_id.clone()
                },
                child_session_id: child.session_id().to_string(),
                depth: if ordinal == 3 { 2 } else { 1 },
                agent_name: "explorer".into(),
                pool_ordinal: ordinal,
            };
            if ordinal == 3 {
                record_run(
                    &mut children[0],
                    &mut child,
                    &member,
                    "grandchild-run",
                    "completed",
                );
            } else {
                record_run(
                    &mut root,
                    &mut child,
                    &member,
                    &format!("child-{ordinal}-run"),
                    "completed",
                );
            }
            members.push(member);
            children.push(child);
        }
        record_run(
            &mut root,
            &mut children[0],
            &members[0],
            "takeover-run",
            "failed",
        );
        let archived_id = members[2].child_session_id.clone();
        let archived_path =
            crate::transcript::archive_index::child_transcript_path(dir.path(), &archived_id);
        std::fs::create_dir_all(archived_path.parent().unwrap()).unwrap();
        let grandchild = children.pop().unwrap();
        let grandchild_path = grandchild.path().to_path_buf();
        drop(grandchild);
        std::fs::write(
            &archived_path,
            zstd::stream::encode_all(std::fs::File::open(&grandchild_path).unwrap(), 1).unwrap(),
        )
        .unwrap();
        std::fs::remove_file(grandchild_path).unwrap();
        use std::io::Write;
        let child_path = children[1].path().to_path_buf();
        drop(children);
        std::fs::OpenOptions::new()
            .append(true)
            .open(child_path)
            .unwrap()
            .write_all(b"{\"kind\":")
            .unwrap();

        let mut setup = TranscriptRecorder::create(child_sessions_dir(dir.path())).unwrap();
        setup.record_session_started("gpt-test").unwrap();
        setup.record_subagents_policy(&policy).unwrap();
        root.record_subagent_started(
            "setup-run",
            &root_id,
            "turn-1",
            setup.session_id(),
            "explorer",
            "private setup",
            99,
        )
        .unwrap();
        drop(setup);
        let runtime = SubagentPool::new();
        let transcript = Arc::new(Mutex::new(root));
        let delegate = Arc::new(RunnerSubagentDelegate {
            child_completions: None,
            runtime: runtime.clone(),
            sessions_dir: dir.path().to_path_buf(),
            transcript: transcript.clone(),
            event_tx: None,
            background_event_tx: None,
            child_live_text: None,
            background_child_started_tx: None,
            route_api_key_configured: Default::default(),
            provider_api_key_hints: Default::default(),
            api_key_hint: String::new(),
            background_control_tx: None,
        });
        for mode in [SubagentMode::Expert, SubagentMode::Swarm] {
            let mut agent = Agent::new("gpt-test", 2, 4);
            agent.set_subagents_policy(SubagentsConfig {
                mode,
                ..Default::default()
            });
            agent.set_subagent_delegate(delegate.clone());
            assert_eq!(
                agent
                    .tool_definitions_for_test()
                    .iter()
                    .any(|tool| tool.name == crate::tool_names::TOOL_AGENT_SEND),
                mode == SubagentMode::Swarm
            );
            let status = agent
                .tool_definitions_for_test()
                .into_iter()
                .find(|tool| tool.name == crate::tool_names::TOOL_AGENT_STATUS)
                .unwrap();
            assert_eq!(status.strict, mode == SubagentMode::Expert);
            if mode == SubagentMode::Expert {
                assert_eq!(
                    status.parameters,
                    json!({"type": "object", "properties": {"run_id": {"type": "string", "minLength": 1}}, "required": ["run_id"], "additionalProperties": false})
                );
            } else {
                assert!(status.parameters["properties"].get("session_id").is_some());
                assert_eq!(status.parameters["oneOf"].as_array().unwrap().len(), 2);
            }
        }
        let jobs = delegate
            .control(crate::tool_names::TOOL_AGENT_JOBS, &json!({}))
            .await
            .unwrap()
            .data
            .unwrap();
        assert_eq!(jobs["jobs"].as_array().unwrap().len(), 4);
        assert_eq!(jobs["jobs"][0]["session_id"], root_id);
        assert_eq!(jobs["jobs"][0]["role"], "root");
        assert_eq!(jobs["jobs"][0]["status"], "idle");
        for field in ["run_id", "child_session_id", "pool_ordinal"] {
            assert!(jobs["jobs"][0].get(field).is_none());
        }
        assert_eq!(jobs["jobs"][3]["depth"], 2);
        let child_records = read_records(crate::transcript::child_session_path(
            dir.path(),
            &members[0].child_session_id,
        ))
        .unwrap();
        assert_eq!(
            runtime
                .swarm_snapshot(dir.path(), &child_records)
                .unwrap()
                .latest_jobs()
                .len(),
            4
        );
        transcript
            .lock()
            .unwrap()
            .record_turn_started(crate::agent::TurnStartedEvent { turn_id: 1 })
            .unwrap();
        let active_root = delegate
            .control(
                crate::tool_names::TOOL_AGENT_STATUS,
                &json!({"session_id": root_id}),
            )
            .await
            .unwrap()
            .data
            .unwrap();
        assert_eq!(active_root["status"], "active");
        assert_eq!(active_root["active"], true);
        transcript
            .lock()
            .unwrap()
            .record_turn_interrupted(Some(1))
            .unwrap();
        assert!(!jobs.to_string().contains("private"));
        for field in ["summary", "full_summary", "structured_result", "task"] {
            assert!(!jobs.to_string().contains(field));
        }
        for (selector, expected) in [
            (json!({"run_id": "child-1-run"}), "completed"),
            (json!({"run_id": "takeover-run"}), "failed"),
            (json!({"session_id": members[0].child_session_id}), "failed"),
            (json!({"session_id": root_id}), "idle"),
            (json!({"session_id": archived_id}), "completed"),
        ] {
            let status = delegate
                .control(crate::tool_names::TOOL_AGENT_STATUS, &selector)
                .await
                .unwrap();
            assert!(status.ok);
            assert_eq!(status.data.unwrap()["status"], expected);
        }
        for args in [
            json!({}),
            json!({"run_id":"x", "session_id":"y"}),
            json!({"session_id":" "}),
            json!({"run_id":null}),
            json!({"other":"x"}),
        ] {
            assert!(
                delegate
                    .control(crate::tool_names::TOOL_AGENT_STATUS, &args)
                    .await
                    .is_err()
            );
        }
        let child_delegate = delegate
            .for_child(Arc::new(Mutex::new(
                TranscriptRecorder::open(
                    child_sessions_dir(dir.path()),
                    &members[0].child_session_id,
                )
                .unwrap(),
            )))
            .unwrap();
        for tool in [
            crate::tool_names::TOOL_AGENT_WAIT,
            crate::tool_names::TOOL_AGENT_CANCEL,
        ] {
            let result = child_delegate
                .control(tool, &json!({"run_id": "child-2-run"}))
                .await
                .unwrap();
            assert!(!result.ok);
        }
        drop(child_delegate);
        let old_records = read_records(&root_path).unwrap();
        assert_eq!(
            runtime
                .swarm_snapshot(dir.path(), &old_records)
                .unwrap()
                .members
                .len(),
            3
        );
        let mut parent = Agent::new("gpt-test", 2, 4);
        parent.set_subagents_policy(policy.clone());
        parent.set_subagent_delegate(delegate.clone());
        let invocation = || SubagentInvocation {
            prompt: "inspect".into(),
            model: None,
            parent_tool_call_id: None,
            input: crate::tool::NormalizedSubagentInput {
                objective: "inspect".into(),
                success_criteria: Vec::new(),
                allowed_paths: Vec::new(),
                forbidden_paths: Vec::new(),
                owned_paths: Vec::new(),
                timeout_secs: None,
                max_tool_calls: None,
                model: None,
                target_child_session_id: None,
                background: false,
            },
        };
        let started = runtime
            .clone()
            .start_named_governed(
                &parent,
                "explorer",
                invocation(),
                dir.path(),
                root_id.clone(),
                "turn-2".into(),
                Some(transcript.clone()),
                None,
            )
            .unwrap();
        assert_eq!(started.receipt().pool_ordinal, 4);
        let run_id = started.run_id().to_string();
        let snapshot = runtime.swarm_snapshot(dir.path(), &old_records).unwrap();
        assert!(runtime.active_jobs().iter().any(|job| job.run_id == run_id));
        assert_eq!(snapshot.members.len(), 4);
        runtime.cancel_run(&run_id);
        runtime.complete_started_run(started).await.unwrap();
        let records = read_records(&root_path).unwrap();
        let snapshot = runtime.swarm_snapshot(dir.path(), &records).unwrap();
        assert_eq!(snapshot.members.len(), 4);
        assert_eq!(
            SubagentPool::new()
                .swarm_snapshot(dir.path(), &records)
                .unwrap()
                .members,
            snapshot.members
        );

        let mut other = TranscriptRecorder::create(dir.path()).unwrap();
        other.record_subagents_policy(&policy).unwrap();
        let other_records = read_records(other.path()).unwrap();
        assert!(
            !delegate
                .control(
                    crate::tool_names::TOOL_AGENT_STATUS,
                    &json!({"session_id": other.session_id()})
                )
                .await
                .unwrap()
                .ok
        );
        assert_eq!(
            runtime
                .swarm_snapshot(dir.path(), &other_records)
                .unwrap()
                .latest_jobs()
                .len(),
            1
        );

        drop(parent);
        drop(delegate);
        drop(transcript);
        let prepared = crate::session::prepare_resume_package(dir.path(), &root_id).unwrap();
        drop(prepared);
        let child_path =
            crate::transcript::child_session_path(dir.path(), &members[0].child_session_id);
        let original = std::fs::read_to_string(&child_path).unwrap();
        let live_path = other.path().to_path_buf();
        let live_content = std::fs::read(&live_path).unwrap();
        for (field, value) in [
            ("root_session_id", json!("foreign-root")),
            ("parent_session_id", json!(members[1].child_session_id)),
            ("depth", json!(9)),
            ("agent_name", json!("oracle")),
            ("pool_ordinal", json!(2)),
        ] {
            let corrupted: String = original
                .lines()
                .map(|line| {
                    let mut record: serde_json::Value = serde_json::from_str(line).unwrap();
                    if record["kind"] == "subagent_activated" {
                        record["member"][field] = value.clone();
                    }
                    format!("{record}\n")
                })
                .collect();
            std::fs::write(&child_path, corrupted).unwrap();
            assert!(
                crate::session::prepare_resume_package(dir.path(), &root_id).is_err(),
                "{field}"
            );
            assert_eq!(
                crate::transcript::restore_subagents_policy(&read_records(&root_path).unwrap())
                    .unwrap(),
                policy
            );
            assert_eq!(std::fs::read(&live_path).unwrap(), live_content);
        }
        std::fs::write(child_path, original).unwrap();
    }

    #[test]
    fn capacity_result_reports_not_started_without_a_run_failure() {
        for tool in [
            crate::tool_names::TOOL_AGENT_EXPLORE,
            crate::tool_names::TOOL_AGENT_SEND,
        ] {
            let result = RunnerSubagentDelegate::capacity_result(
                tool,
                Some("explorer"),
                Some("child".into()),
                &crate::subagent::SubagentCapacityExceeded { limit: 1 },
            );
            assert!(!result.ok);
            assert!(result.error.unwrap().message.contains("no run was started"));
            let data = result.data.unwrap();
            assert_eq!(data["status"], "not_started");
            assert_eq!(data["started"], false);
            assert_eq!(data["active"], false);
            assert_eq!(data["accepted"], false);
            assert_eq!(data["max_concurrent"], 1);
            assert!(data.get("failure_kind").is_none());
        }
    }

    #[test]
    fn hard_failure_result_preserves_compact_full_and_error_payloads() {
        let summary = "first line\n".to_string() + &"detail ".repeat(40);
        let result = RunnerSubagentDelegate::hard_failure_result(
            "agent__explore",
            "explorer",
            Some("child-123".to_string()),
            summary.clone(),
        );

        assert!(!result.ok);
        assert_eq!(result.tool, "agent__explore");
        assert_eq!(
            result.error.as_ref().map(|error| &error.message),
            Some(&summary)
        );
        assert_eq!(
            result.error.as_ref().map(|error| error.recoverable),
            Some(true)
        );

        let data = result.data.expect("hard failure data");
        assert_eq!(data["agent_name"], "explorer");
        assert_eq!(data["child_session_id"], "child-123");
        assert_eq!(data["status"], "failed");
        assert_eq!(data["failure_kind"], "hard");
        assert_eq!(data["summary"], compact_subagent_summary(&summary));
        assert_eq!(data["full_summary"], summary);
        assert_eq!(data["active"], false);
    }
}
