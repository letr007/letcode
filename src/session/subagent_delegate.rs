use std::path::PathBuf;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use anyhow::{Result, anyhow, bail};
use serde_json::json;

use crate::agent::{
    Agent, SubagentDelegate, SubagentInvocation, subagent_tool_name_for_agent_name,
};
use crate::subagent::{SubagentFailureKind, SubagentPool, SubagentStatus};
use crate::tool::ToolResult;
use crate::transcript::TranscriptRecorder;

use super::events::SessionTransportEventSender;
use super::formatting::compact_subagent_summary;
use super::subagent_event_sender;

pub(super) struct RunnerSubagentDelegate {
    pub(super) runtime: SubagentPool,
    pub(super) sessions_dir: PathBuf,
    pub(super) transcript: Arc<Mutex<TranscriptRecorder>>,
    pub(super) event_tx: Option<SessionTransportEventSender>,
    pub(super) route_api_key_configured: indexmap::IndexMap<String, bool>,
    pub(super) retained_session_routes: std::collections::HashSet<String>,
    pub(super) provider_api_key_hints: indexmap::IndexMap<String, String>,
    pub(super) api_key_hint: String,
}

impl RunnerSubagentDelegate {
    fn route_display_name(
        &self,
        parent: &Agent<async_openai::config::OpenAIConfig>,
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
            "unreconciled": false,
            "reconciled": false,
            "reusable": false,
        });
        ToolResult::err_with_data(tool_name, summary, data)
    }
}

impl SubagentDelegate<async_openai::config::OpenAIConfig> for RunnerSubagentDelegate {
    fn run_named<'a>(
        &'a self,
        parent: &'a Agent<async_openai::config::OpenAIConfig>,
        agent_name: &'a str,
        invocation: SubagentInvocation,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<ToolResult>> + Send + 'a>> {
        Box::pin(async move {
            let tool_name = subagent_tool_name_for_agent_name(agent_name)
                .expect("runner dispatched unknown subagent agent name");
            let route_display_name = match self.route_display_name(parent, agent_name, &invocation)
            {
                Ok(route) => route,
                Err(error) => {
                    let summary = error.to_string();
                    let data = json!({
                        "agent_name": agent_name,
                        "child_session_id": invocation.input.target_child_session_id,
                        "status": SubagentStatus::Failed.as_str(),
                        "failure_kind": SubagentFailureKind::Hard.as_str(),
                        "summary": compact_subagent_summary(&summary),
                        "full_summary": summary,
                        "active": false,
                        "unreconciled": false,
                        "reconciled": false,
                        "reusable": false,
                    });
                    return Ok(ToolResult::err_with_data(tool_name, summary, data));
                }
            };
            if !self
                .route_api_key_configured
                .get(&route_display_name)
                .copied()
                .unwrap_or_else(|| self.retained_session_routes.contains(&route_display_name))
            {
                return Ok(self.missing_api_key_result(tool_name, agent_name, route_display_name));
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
                        "unreconciled": false,
                        "reconciled": false,
                        "reusable": false,
                    });
                    return Ok(ToolResult::err_with_data(tool_name, summary, data));
                }
            };
            let parent_turn_id = format!(
                "turn-{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis()
            );
            let summary = self
                .runtime
                .run_named_governed(
                    parent,
                    agent_name,
                    invocation,
                    self.sessions_dir.clone(),
                    parent_session_id,
                    parent_turn_id,
                    Some(self.transcript.clone()),
                    self.event_tx.clone().map(subagent_event_sender),
                )
                .await;

            let summary = match summary {
                Ok(summary) => summary,
                Err(error) => {
                    let summary = error.to_string();
                    let data = json!({
                        "agent_name": agent_name,
                        "child_session_id": target_child_session_id,
                        "status": SubagentStatus::Failed.as_str(),
                        "failure_kind": SubagentFailureKind::Hard.as_str(),
                        "summary": compact_subagent_summary(&summary),
                        "full_summary": summary,
                        "active": false,
                        "unreconciled": false,
                        "reconciled": false,
                        "reusable": false,
                    });
                    return Ok(ToolResult::err_with_data(tool_name, summary, data));
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
                "unreconciled": status == SubagentStatus::Completed,
                "reconciled": false,
                "reusable": false,
            });

            if status == SubagentStatus::Completed {
                Ok(ToolResult::ok(tool_name, data))
            } else {
                Ok(ToolResult::err_with_data(tool_name, summary_text, data))
            }
        })
    }
}
