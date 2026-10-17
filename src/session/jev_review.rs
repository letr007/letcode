//! Experimental approval backend: Typesafe Jev.
//!
//! Jev answers a `choice` question against a JSON `state` and returns
//! probabilities over the criteria instead of prose, so this backend posts to
//! `/v1/systemone` directly rather than going through the chat routes. It is
//! selected by the provider the reviewer route names, and answers the same
//! `execute` / `ask_user` / `refuse` contract as the reviewer expert. Each
//! decision is recorded in the same sticky `reviewer` child session the expert
//! backend uses, so both backends look the same when that session is opened.

use std::collections::{BTreeMap, HashMap};
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use futures_util::FutureExt;
use serde::Deserialize;
use serde_json::{Map, Value, json};

use crate::agent::{
    Agent, AgentTemplate, AutoReviewOutcome, AutoReviewResolution, AutoReviewService,
    REVIEW_INSTRUCTIONS, REVIEW_OUTCOME_CRITERIA,
};
use crate::config::JevReviewConfig;
use crate::permission::PermissionRequest;
use crate::session::auto_review::{
    build_review_prompt, emit_auto_resolution, record_auto_decision,
};
use crate::session::runner::{SessionTransportEventSender, subagent_event_sender};
use crate::subagent::{SubagentPool, SubagentRunGovernance};
use crate::tool::NormalizedSubagentInput;
use crate::transcript::{
    TranscriptEvent, TranscriptRecord, TranscriptRecorder, read_records_allow_partial_tail,
};

/// The reviewer asks Jev exactly one question per call.
const QUESTION: &str = "approval";
/// How much of the session's user messages the reviewer sees.
const USER_MESSAGE_LIMIT: usize = 3;
/// Bound on the executor's own narration, in characters.
const EXECUTOR_CONTEXT_LIMIT: usize = 800;
/// Bound on text copied out of an error response or rationale.
const TEXT_LIMIT: usize = 400;
/// File-name markers whose contents do not belong in an unattended decision.
const SECRET_PATH_MARKERS: [&str; 6] = [".env", "id_rsa", ".pem", ".ssh/", "credentials", ".p12"];

pub(crate) struct JevReviewer {
    config: JevReviewConfig,
    client: reqwest::Client,
    parent_transcript: Arc<Mutex<TranscriptRecorder>>,
    event_tx: Option<SessionTransportEventSender>,
    pool: SubagentPool,
    sessions_dir: PathBuf,
    /// The `reviewer` child session approvals are recorded in. Reviews are
    /// serialized by `review_gate`, so the sticky id is read and written under
    /// that same lock.
    child_session_id: Arc<Mutex<Option<String>>>,
    review_gate: Arc<tokio::sync::Mutex<()>>,
}

impl JevReviewer {
    pub(crate) fn new(
        config: JevReviewConfig,
        parent_transcript: Arc<Mutex<TranscriptRecorder>>,
        event_tx: Option<SessionTransportEventSender>,
        pool: SubagentPool,
        sessions_dir: PathBuf,
    ) -> Result<Self> {
        let mut builder =
            reqwest::Client::builder().timeout(Duration::from_secs(config.timeout_secs));
        if is_loopback_endpoint(&config.endpoint) || is_loopback_endpoint(&config.base_url) {
            builder = builder.no_proxy();
        }
        let client = builder
            .build()
            .context("failed to build the Jev HTTP client")?;
        Ok(Self {
            config,
            client,
            parent_transcript,
            event_tx,
            pool,
            sessions_dir,
            child_session_id: Arc::new(Mutex::new(None)),
            review_gate: Arc::new(tokio::sync::Mutex::new(())),
        })
    }

    fn sticky_child_id(&self) -> Option<String> {
        self.child_session_id
            .lock()
            .ok()
            .and_then(|state| state.clone())
    }

    fn set_sticky_child_id(&self, child_session_id: &str) {
        if let Ok(mut state) = self.child_session_id.lock() {
            *state = Some(child_session_id.to_string());
        }
    }

    /// What the reviewer is shown. Every field here is gathered by this process
    /// except `executor_context`, which is the requesting agent's own narration.
    fn build_state(&self, parent: &Agent, request: &PermissionRequest) -> Value {
        let records = self.parent_records();
        let user_messages = user_messages(&records, USER_MESSAGE_LIMIT);
        let mut state = Map::new();
        state.insert(
            "user_goal".into(),
            json!(user_messages.last().cloned().unwrap_or_default()),
        );
        state.insert("tool".into(), json!(request.tool));
        state.insert("class".into(), json!(request.class.as_str()));
        state.insert("summary".into(), json!(request.summary));
        state.insert("arguments".into(), request.args.clone());
        if !user_messages.is_empty() {
            state.insert("user_messages".into(), json!(user_messages));
        }
        state.insert("facts".into(), facts(request));
        state.insert("session_history".into(), session_history(&records, request));
        if let Some(text) = parent.last_visible_assistant_text() {
            let text = truncate(text.trim(), EXECUTOR_CONTEXT_LIMIT);
            if !text.is_empty() {
                state.insert("executor_context".into(), json!(text));
            }
        }
        Value::Object(state)
    }

    fn parent_records(&self) -> Vec<TranscriptRecord> {
        let Ok(recorder) = self.parent_transcript.lock() else {
            return Vec::new();
        };
        read_records_allow_partial_tail(recorder.path()).unwrap_or_default()
    }
}

impl AutoReviewService for JevReviewer {
    fn review<'a>(
        &'a self,
        parent: &'a Agent,
        request: PermissionRequest,
        _user_goal: Option<String>,
    ) -> Pin<Box<dyn Future<Output = Result<AutoReviewResolution>> + Send + 'a>> {
        Box::pin(async move {
            let _review_guard = self.review_gate.lock().await;

            if self.config.credential.trim().is_empty() {
                bail!("{}", missing_credential_rationale(&self.config));
            }

            let state = self.build_state(parent, &request);
            let goal = state
                .get("user_goal")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|goal| !goal.is_empty());
            // The executor context travels in `state` for this backend, so the
            // recorded request card stays the same as it was before.
            let prompt = build_review_prompt(&request, goal, None);

            let template = AgentTemplate::reviewer();
            let takeover = self.sticky_child_id();
            let parent_session_id = self
                .parent_transcript
                .lock()
                .map_err(|_| anyhow!("transcript recorder poisoned"))?
                .session_id()
                .to_string();
            let parent_turn_id = format!(
                "jev-review-{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis()
            );
            let event_sender = self.event_tx.clone().map(subagent_event_sender);
            let input = NormalizedSubagentInput {
                objective: prompt.clone(),
                success_criteria: vec![
                    "Return only JSON with decision, risk, and rationale.".into(),
                ],
                allowed_paths: Vec::new(),
                forbidden_paths: Vec::new(),
                owned_paths: Vec::new(),
                // This child makes one HTTP call and records it; it runs no tools,
                // so it takes no workspace lock and the pool's bound is the HTTP
                // timeout this backend is configured with.
                timeout_secs: Some(self.config.timeout_secs),
                max_tool_calls: Some(0),
                model: None,
                target_child_session_id: takeover.clone(),
                background: false,
            };
            let governance = SubagentRunGovernance {
                timeout_secs: input.effective_timeout_secs(template.timeout_secs),
                max_tool_calls: input.effective_max_tool_calls(template.max_tool_calls),
                model: None,
                input,
            };

            let captured: Arc<Mutex<Option<(AutoReviewOutcome, String)>>> =
                Arc::new(Mutex::new(None));
            let captured_for_executor = Arc::clone(&captured);
            let client = self.client.clone();
            let config = self.config.clone();
            let summary = self
                .pool
                .run_with_executor(
                    parent,
                    template,
                    prompt.clone(),
                    governance,
                    self.sessions_dir.clone(),
                    parent_session_id,
                    parent_turn_id,
                    Some(Arc::clone(&self.parent_transcript)),
                    event_sender,
                    takeover,
                    move |_agent,
                          prompt,
                          transcript,
                          _event_sender,
                          _child_session_id,
                          _agent_name| {
                        let captured = Arc::clone(&captured_for_executor);
                        let client = client.clone();
                        let config = config.clone();
                        async move {
                            {
                                let mut recorder = transcript
                                    .lock()
                                    .map_err(|_| anyhow!("transcript recorder poisoned"))?;
                                recorder.record_user_message(prompt.clone())?;
                            }
                            let answer = ask_jev(&client, &config, state).await?;
                            let Some(outcome) = outcome_for(&answer.choice) else {
                                bail!(
                                    "jev answered an unknown decision: {}",
                                    truncate(answer.choice.trim(), TEXT_LIMIT)
                                );
                            };
                            let rationale = describe(&answer);
                            let reply = review_reply(outcome, &rationale);
                            {
                                let mut recorder = transcript
                                    .lock()
                                    .map_err(|_| anyhow!("transcript recorder poisoned"))?;
                                recorder.record_assistant_message(reply.clone())?;
                            }
                            *captured
                                .lock()
                                .map_err(|_| anyhow!("jev review capture poisoned"))? =
                                Some((outcome, rationale));
                            Ok(reply)
                        }
                        .boxed()
                    },
                )
                .await;

            let summary = summary.map_err(|error| anyhow!("jev review failed: {error:#}"))?;
            let child_session_id = summary.child_session_id.clone();
            self.set_sticky_child_id(&child_session_id);
            let (outcome, rationale) = captured
                .lock()
                .map_err(|_| anyhow!("jev review capture poisoned"))?
                .take()
                // The run ended without reaching a verdict (timeout or
                // cancellation), so the pool's account of it is the reason.
                .ok_or_else(|| anyhow!("jev review failed: {}", summary.summary))?;

            let risk = risk_for(outcome);
            if outcome != AutoReviewOutcome::Ask {
                record_auto_decision(
                    &self.parent_transcript,
                    &request,
                    outcome,
                    &rationale,
                    Some(risk),
                    Some(child_session_id.as_str()),
                )?;
                emit_auto_resolution(
                    self.event_tx.as_ref(),
                    &request,
                    outcome,
                    &rationale,
                    Some(risk),
                    Some(child_session_id.as_str()),
                );
            }
            Ok(AutoReviewResolution {
                outcome,
                reason: rationale,
            })
        })
    }

    /// Drops the reviewer session so the next approval starts a new one, as the
    /// expert backend does when the reviewer policy changes.
    fn clear_sticky(&self) {
        if let Ok(mut state) = self.child_session_id.lock() {
            *state = None;
        }
    }
}

async fn ask_jev(
    client: &reqwest::Client,
    config: &JevReviewConfig,
    state: Value,
) -> Result<Answer> {
    let credential = config.credential.trim();
    if credential.is_empty() {
        bail!("{}", missing_credential_rationale(config));
    }
    let url = &config.endpoint;
    let body = json!({
        "model": config.model,
        "state": state,
        "questions": {
            QUESTION: {
                "type": "choice",
                "instructions": REVIEW_INSTRUCTIONS,
                "criteria": criteria_object(),
            }
        },
    });
    let response = client
        .post(url)
        .header("authorization", format!("Bearer {credential}"))
        .json(&body)
        .send()
        .await
        .context("jev request failed")?;
    let status = response.status();
    if !status.is_success() {
        let text = response.text().await.unwrap_or_default();
        bail!(
            "jev answered {status}: {}",
            truncate(text.trim(), TEXT_LIMIT)
        );
    }
    let body: SystemOneResponse = response
        .json()
        .await
        .context("jev returned a malformed body")?;
    body.answers
        .get(QUESTION)
        .map(Answer::from)
        .ok_or_else(|| anyhow!("jev answered no '{QUESTION}' question"))
}

fn missing_credential_rationale(config: &JevReviewConfig) -> String {
    format!(
        "jev review is unavailable: set [providers.{}.auth] credential to a Typesafe key",
        config.provider
    )
}

/// The criteria object Jev receives. Order follows the shared contract.
fn criteria_object() -> Value {
    let mut criteria = Map::new();
    for (outcome, criterion) in REVIEW_OUTCOME_CRITERIA {
        criteria.insert(outcome.into(), json!(criterion));
    }
    Value::Object(criteria)
}

/// What the process can establish about this call without asking a model.
///
/// Fields are omitted rather than sent as `false`: a missing key means "not
/// determined", while `false` would claim a certainty this code does not have.
fn facts(request: &PermissionRequest) -> Value {
    let mut facts = Map::new();
    if let Some(access) =
        crate::tool::external_workspace_access_for_tool(&request.tool, &request.args)
    {
        facts.insert("outside_workspace_paths".into(), json!(access.paths));
    }
    if let Some(text) = command_text(&request.args)
        && SECRET_PATH_MARKERS
            .iter()
            .any(|marker| text.contains(marker))
    {
        facts.insert("matches_secret_path_marker".into(), json!(true));
    }
    Value::Object(facts)
}

fn command_text(args: &Value) -> Option<&str> {
    args.get("command").and_then(Value::as_str)
}

/// Counts this exact call earlier in the session and how often it completed.
///
/// The current call is excluded: whether it appears in the transcript already
/// depends on where the caller records it relative to the permission check.
fn session_history(records: &[TranscriptRecord], request: &PermissionRequest) -> Value {
    let signature = signature_of(&request.args);
    let current = request.call_id.as_deref();
    let mut identical = 0usize;
    let mut succeeded = 0usize;
    let mut reviewed: HashMap<&str, bool> = HashMap::new();
    let mut calls = 0usize;
    for record in records {
        match &record.event {
            TranscriptEvent::ToolCallStarted {
                call_id,
                name,
                args,
            } => {
                if Some(call_id.as_str()) == current {
                    continue;
                }
                calls += 1;
                let same = name == &request.tool && signature_of(args) == signature;
                if same {
                    identical += 1;
                }
                reviewed.insert(call_id.as_str(), same);
            }
            TranscriptEvent::ToolCallFinished { call_id, ok, .. } => {
                let counted = *ok && reviewed.get(call_id.as_str()).copied().unwrap_or(false);
                succeeded += usize::from(counted);
            }
            _ => {}
        }
    }
    json!({
        "prior_calls_in_session": calls,
        "prior_identical_calls": identical,
        "prior_successful_runs_of_this_call": succeeded,
    })
}

fn signature_of(args: &Value) -> String {
    args.to_string()
}

fn user_messages(records: &[TranscriptRecord], limit: usize) -> Vec<String> {
    let mut messages = Vec::new();
    for record in records.iter().rev() {
        if let TranscriptEvent::UserMessage { content, .. } = &record.event {
            let text = content.text.trim();
            if !text.is_empty() {
                messages.push(text.to_string());
            }
            if messages.len() == limit {
                break;
            }
        }
    }
    messages.reverse();
    messages
}

fn outcome_for(choice: &str) -> Option<AutoReviewOutcome> {
    match choice.trim().to_ascii_lowercase().as_str() {
        "execute" => Some(AutoReviewOutcome::AllowOnce),
        "ask_user" => Some(AutoReviewOutcome::Ask),
        "refuse" => Some(AutoReviewOutcome::Deny),
        _ => None,
    }
}

/// Jev reports no severity, so the recorded risk follows the decision.
fn risk_for(outcome: AutoReviewOutcome) -> &'static str {
    match outcome {
        AutoReviewOutcome::AllowOnce | AutoReviewOutcome::AllowAlways => "low",
        AutoReviewOutcome::Ask => "medium",
        AutoReviewOutcome::Deny => "high",
    }
}

/// The verdict as it is recorded in the reviewer child session: the same JSON
/// shape the reviewer expert replies with, so the child view renders a decision
/// card for both backends.
fn review_reply(outcome: AutoReviewOutcome, rationale: &str) -> String {
    let decision = match outcome {
        AutoReviewOutcome::AllowOnce => "execute",
        AutoReviewOutcome::AllowAlways => "allow_always",
        AutoReviewOutcome::Ask => "ask_user",
        AutoReviewOutcome::Deny => "refuse",
    };
    json!({
        "decision": decision,
        "risk": risk_for(outcome),
        "rationale": rationale,
    })
    .to_string()
}

/// The distribution, so a low-confidence decision is visible in the transcript.
fn describe(answer: &Answer) -> String {
    let mut parts = Vec::new();
    for (outcome, probability) in &answer.probabilities {
        parts.push(format!("{outcome} {probability:.2}"));
    }
    let distribution = if parts.is_empty() {
        String::new()
    } else {
        format!("; {}", parts.join(" / "))
    };
    format!(
        "jev: {} (confidence {:.2}{distribution})",
        answer.choice.trim(),
        answer.confidence
    )
}

fn truncate(text: &str, limit: usize) -> String {
    if text.chars().count() <= limit {
        return text.to_string();
    }
    let mut out = text
        .chars()
        .take(limit.saturating_sub(1))
        .collect::<String>();
    out.push('…');
    out
}

fn is_loopback_endpoint(base_url: &str) -> bool {
    reqwest::Url::parse(base_url).is_ok_and(|url| {
        url.host_str().is_some_and(|host| {
            // `host_str` keeps the brackets an IPv6 literal is written with.
            let host = host.trim_start_matches('[').trim_end_matches(']');
            host.eq_ignore_ascii_case("localhost")
                || host
                    .parse::<std::net::IpAddr>()
                    .is_ok_and(|address| address.is_loopback())
        })
    })
}

#[derive(Debug, Default, Clone)]
struct Answer {
    choice: String,
    confidence: f64,
    probabilities: BTreeMap<String, f64>,
}

#[derive(Debug, Deserialize)]
struct SystemOneResponse {
    #[serde(default)]
    answers: BTreeMap<String, RawAnswer>,
}

#[derive(Debug, Deserialize)]
struct RawAnswer {
    #[serde(default)]
    choice: String,
    #[serde(default)]
    confidence: f64,
    #[serde(default)]
    probabilities: BTreeMap<String, f64>,
}

impl From<&RawAnswer> for Answer {
    fn from(raw: &RawAnswer) -> Self {
        Self {
            choice: raw.choice.clone(),
            confidence: raw.confidence,
            probabilities: raw.probabilities.clone(),
        }
    }
}

#[cfg(test)]
#[path = "../../tests/internal/session/jev_review/tests.rs"]
mod tests;
