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
#[path = "../../tests/internal/session/subagent_delegate/tests.rs"]
mod tests;
