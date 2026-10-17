//! Sticky reviewer expert for `PermissionMode::Auto`.
//!
//! Reuses [`SubagentPool`] + `AgentTemplate::reviewer` with one child session per
//! parent session. Writes permission decisions into the parent transcript.

use std::sync::{Arc, Mutex};

use anyhow::{Result, anyhow, bail};
use serde::Deserialize;
use serde_json::Value;

use crate::agent::{
    Agent, AgentTemplate, AutoReviewOutcome, AutoReviewResolution, AutoReviewService,
    REVIEW_DECISION_VOCABULARY,
};
use crate::permission::PermissionRequest;
use crate::session::event::PermissionResolutionEvent;
use crate::session::runner::{
    SessionTransportEvent, SessionTransportEventSender, subagent_event_sender,
};
use crate::subagent::{SubagentPool, SubagentRunGovernance, SubagentRunSummary, SubagentStatus};
use crate::tool::NormalizedSubagentInput;
use crate::transcript::{TranscriptEvent, TranscriptRecorder, read_records_allow_partial_tail};
use futures_util::FutureExt;

const REVIEWER_AGENT_NAME: &str = "reviewer";

#[derive(Clone)]
pub(crate) struct StickyAutoReviewer {
    inner: Arc<Mutex<StickyAutoReviewerState>>,
    review_gate: Arc<tokio::sync::Mutex<()>>,
    pool: SubagentPool,
    sessions_dir: std::path::PathBuf,
    parent_transcript: Arc<Mutex<TranscriptRecorder>>,
    event_tx: Option<SessionTransportEventSender>,
    route_api_key_configured: Arc<Mutex<indexmap::IndexMap<String, bool>>>,
    provider_api_key_hints: Arc<Mutex<indexmap::IndexMap<String, String>>>,
    api_key_hint: String,
}

struct StickyAutoReviewerState {
    child_session_id: Option<String>,
}

impl StickyAutoReviewer {
    pub fn new(
        pool: SubagentPool,
        sessions_dir: std::path::PathBuf,
        parent_transcript: Arc<Mutex<TranscriptRecorder>>,
        event_tx: Option<SessionTransportEventSender>,
        route_api_key_configured: Arc<Mutex<indexmap::IndexMap<String, bool>>>,
        provider_api_key_hints: Arc<Mutex<indexmap::IndexMap<String, String>>>,
        api_key_hint: String,
    ) -> Self {
        Self {
            inner: Arc::new(Mutex::new(StickyAutoReviewerState {
                child_session_id: None,
            })),
            review_gate: Arc::new(tokio::sync::Mutex::new(())),
            pool,
            sessions_dir,
            parent_transcript,
            event_tx,
            route_api_key_configured,
            provider_api_key_hints,
            api_key_hint,
        }
    }

    fn resolve_route(
        &self,
        parent: &Agent,
        template: &AgentTemplate,
        takeover_child_session_id: Option<&str>,
    ) -> Result<crate::config::ModelRoute> {
        let recorded_route = takeover_child_session_id
            .map(|child_session_id| -> Result<_> {
                let records = read_records_allow_partial_tail(
                    crate::transcript::child_sessions_dir(&self.sessions_dir)
                        .join(format!("{child_session_id}.jsonl")),
                )?;
                let route = crate::transcript::restore_latest_model(&records).ok_or_else(|| {
                    anyhow!(
                        "takeover failed: child `{child_session_id}` has no recorded model route"
                    )
                })?;
                crate::config::ModelRoute::parse(&route).map_err(|error| {
                    anyhow!(
                        "takeover failed: child `{child_session_id}` recorded an invalid model route: {error}"
                    )
                })
            })
            .transpose()?;
        crate::agent::AgentFactory::resolve_subagent_route(
            parent,
            template,
            recorded_route.as_ref(),
            recorded_route.is_some(),
        )
    }

    fn route_has_api_key(&self, parent: &Agent, route: &crate::config::ModelRoute) -> Result<bool> {
        self.route_api_key_configured
            .lock()
            .map_err(|_| anyhow!("route credential state poisoned"))
            .map(|configured| {
                configured
                    .get(&route.display_name())
                    .copied()
                    .unwrap_or_else(|| {
                        crate::agent::AgentFactory::resolve_subagent_route(
                            parent,
                            &AgentTemplate::reviewer(),
                            None,
                            false,
                        )
                        .is_ok_and(|current| current == *route)
                    })
            })
    }

    fn missing_api_key_rationale(&self, route: &crate::config::ModelRoute) -> String {
        let hint = self
            .provider_api_key_hints
            .lock()
            .ok()
            .and_then(|hints| hints.get(&route.provider).cloned())
            .unwrap_or_else(|| self.api_key_hint.clone());
        format!(
            "auto-review unavailable: API key is not set for reviewer route '{}'. {hint}",
            route.display_name()
        )
    }

    #[cfg(test)]
    fn sticky_child_id(&self) -> String {
        self.inner
            .lock()
            .ok()
            .and_then(|state| state.child_session_id.clone())
            .unwrap_or_default()
    }

    fn latest_user_goal(&self) -> Option<String> {
        let recorder = self.parent_transcript.lock().ok()?;
        let records = read_records_allow_partial_tail(recorder.path()).ok()?;
        records
            .into_iter()
            .rev()
            .find_map(|record| match record.event {
                TranscriptEvent::UserMessage { content, .. } => {
                    let text = content.text.trim();
                    (!text.is_empty()).then(|| text.to_string())
                }
                _ => None,
            })
    }

    fn record_decision(
        &self,
        request: &PermissionRequest,
        parsed: &ParsedReview,
        child_session_id: &str,
    ) -> Result<()> {
        record_auto_decision(
            &self.parent_transcript,
            request,
            parsed.outcome,
            &parsed.rationale,
            parsed.risk.as_deref(),
            (!child_session_id.is_empty()).then_some(child_session_id),
        )
    }
}

impl AutoReviewService for StickyAutoReviewer {
    fn review<'a>(
        &'a self,
        parent: &'a Agent,
        request: PermissionRequest,
        user_goal: Option<String>,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<AutoReviewResolution>> + Send + 'a>,
    > {
        Box::pin(async move {
            let _review_guard = self.review_gate.lock().await;

            let template = AgentTemplate::reviewer();
            let takeover = self
                .inner
                .lock()
                .map_err(|_| anyhow!("auto-reviewer state poisoned"))?
                .child_session_id
                .clone();
            let route = self.resolve_route(parent, &template, takeover.as_deref())?;
            if !self.route_has_api_key(parent, &route)? {
                bail!("{}", self.missing_api_key_rationale(&route));
            }

            let goal = user_goal.or_else(|| self.latest_user_goal());
            let prompt = expert_review_prompt(parent, &request, goal.as_deref());

            let parent_session_id = self
                .parent_transcript
                .lock()
                .map_err(|_| anyhow!("transcript recorder poisoned"))?
                .session_id()
                .to_string();
            let parent_turn_id = format!(
                "auto-review-{}",
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
                timeout_secs: template.timeout_secs,
                max_tool_calls: template.max_tool_calls,
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

            // Custom executor: SilentDeny permissions (no human UI) and no parent
            // tool-call projection — reviewer is not a delegated agent__* tool.
            let full_output = Arc::new(Mutex::new(None));
            let full_output_for_executor = Arc::clone(&full_output);
            let summary = match self
                .pool
                .run_with_executor(
                    parent,
                    template,
                    prompt,
                    governance,
                    self.sessions_dir.clone(),
                    parent_session_id,
                    parent_turn_id,
                    Some(Arc::clone(&self.parent_transcript)),
                    event_sender,
                    takeover,
                    move |agent,
                          prompt,
                          transcript,
                          _event_sender,
                          _child_session_id,
                          _agent_name| {
                        let full_output = Arc::clone(&full_output_for_executor);
                        async move {
                            {
                                let mut recorder = transcript
                                    .lock()
                                    .map_err(|_| anyhow!("transcript recorder poisoned"))?;
                                recorder.record_user_message(prompt.clone())?;
                            }
                            let output = agent
                                .run_resolved_text_oneshot(&prompt)
                                .await
                                .map_err(|error| anyhow!("auto-review failed: {error:#}"))?;
                            *full_output
                                .lock()
                                .map_err(|_| anyhow!("auto-review output capture poisoned"))? =
                                Some(output.clone());
                            Ok(output)
                        }
                        .boxed()
                    },
                )
                .await
            {
                Ok(summary) => summary,
                Err(error) => return Err(anyhow!("auto-review failed: {error:#}")),
            };

            let child_session_id = summary.child_session_id.clone();
            {
                let mut state = self
                    .inner
                    .lock()
                    .map_err(|_| anyhow!("auto-reviewer state poisoned"))?;
                state.child_session_id = Some(child_session_id.clone());
            }

            let full_output = full_output
                .lock()
                .map_err(|_| anyhow!("auto-review output capture poisoned"))?;
            if summary.status != SubagentStatus::Completed {
                bail!(
                    "auto-review did not complete: {}",
                    one_line(&summary.summary, 160)
                );
            }
            let parsed =
                parse_reviewer_output(&summary, full_output.as_deref(), request.can_allow_always)?;
            if parsed.outcome == AutoReviewOutcome::Ask {
                // Not a decision: the call goes to the user through the same prompt
                // `default` mode uses, and that path records the answer. The reviewer's
                // reasoning stays in its own child transcript.
                tracing::debug!(
                    call_id = request.call_id.as_deref().unwrap_or_default(),
                    rationale = %parsed.rationale,
                    "auto-review left the call to the user"
                );
                return Ok(parsed.into_resolution());
            }

            self.record_decision(&request, &parsed, &child_session_id)?;
            self.emit_resolution(&request, &parsed, &child_session_id);

            Ok(parsed.into_resolution())
        })
    }

    fn clear_sticky(&self) {
        if let Ok(mut state) = self.inner.lock() {
            state.child_session_id = None;
        }
    }
}

impl StickyAutoReviewer {
    fn emit_resolution(
        &self,
        request: &PermissionRequest,
        parsed: &ParsedReview,
        child_session_id: &str,
    ) {
        emit_auto_resolution(
            self.event_tx.as_ref(),
            request,
            parsed.outcome,
            &parsed.rationale,
            parsed.risk.as_deref(),
            (!child_session_id.is_empty()).then_some(child_session_id),
        );
    }
}

/// Records a decided outcome in the parent transcript.
///
/// `ask` never reaches here: it is handed back to the requester for one round of
/// explanation, and to the user when that round settles nothing.
pub(crate) fn record_auto_decision(
    parent_transcript: &Mutex<TranscriptRecorder>,
    request: &PermissionRequest,
    outcome: AutoReviewOutcome,
    rationale: &str,
    risk: Option<&str>,
    reviewer_child_session_id: Option<&str>,
) -> Result<()> {
    let mut recorder = parent_transcript
        .lock()
        .map_err(|_| anyhow!("transcript recorder poisoned"))?;
    recorder.record_permission_decision_full(
        request.call_id.clone(),
        request.tool.clone(),
        request.args.clone(),
        outcome
            .approval()
            .is_some_and(|approval| approval.allowed()),
        Some(rationale.to_string()),
        Some("auto".into()),
        Some(outcome.label().into()),
        risk.map(str::to_string),
        reviewer_child_session_id.map(str::to_string),
    )?;
    Ok(())
}

/// Announces a decided outcome to the session transport.
pub(crate) fn emit_auto_resolution(
    event_tx: Option<&SessionTransportEventSender>,
    request: &PermissionRequest,
    outcome: AutoReviewOutcome,
    rationale: &str,
    risk: Option<&str>,
    reviewer_child_session_id: Option<&str>,
) {
    let Some(event_tx) = event_tx else {
        return;
    };
    let call_id = request
        .call_id
        .clone()
        .unwrap_or_else(|| request.tool.clone());
    let decision = match outcome {
        AutoReviewOutcome::Deny => crate::session::PermissionDecision::Denied,
        _ => crate::session::PermissionDecision::Approved,
    };
    let _ = event_tx.send(SessionTransportEvent::PermissionResolved(
        PermissionResolutionEvent {
            call_id,
            decision,
            reason: Some(rationale.to_string()),
            tool_name: Some(request.tool.clone()),
            summary: Some(request.summary.clone()),
            origin_label: Some(REVIEWER_AGENT_NAME.into()),
            approval: Some(outcome.label().into()),
            risk: risk.map(str::to_string),
            reviewer_child_session_id: reviewer_child_session_id.map(str::to_string),
        },
    ));
}

struct ParsedReview {
    outcome: AutoReviewOutcome,
    rationale: String,
    risk: Option<String>,
}

impl ParsedReview {
    fn into_resolution(self) -> AutoReviewResolution {
        AutoReviewResolution {
            outcome: self.outcome,
            reason: self.rationale,
        }
    }
}

/// The reviewer expert's prompt, carrying the requesting agent's own most
/// recent visible narration next to the goal it is judged against.
fn expert_review_prompt(
    parent: &Agent,
    request: &PermissionRequest,
    user_goal: Option<&str>,
) -> String {
    let executor_context = parent.last_visible_assistant_text();
    build_review_prompt(request, user_goal, executor_context.as_deref())
}

pub(crate) fn build_review_prompt(
    request: &PermissionRequest,
    user_goal: Option<&str>,
    executor_context: Option<&str>,
) -> String {
    let args = serde_json::to_string_pretty(&request.args).unwrap_or_else(|_| "{}".into());
    let goal = user_goal.unwrap_or("(not provided)");
    let preview = request.preview.as_deref().unwrap_or("(none)");
    // Absent narration leaves the prompt as it was before the requester said
    // anything visible.
    let executor_context = executor_context
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .map(|text| format!("Requester's most recent visible message:\n{text}\n\n"))
        .unwrap_or_default();
    format!(
        "Approve or deny this tool permission request.\n\
         \n\
         User goal:\n{goal}\n\
         \n\
         {executor_context}Tool: {}\n\
         Class: {}\n\
         Summary: {}\n\
         Preview: {preview}\n\
         can_allow_always: {}\n\
         Arguments:\n{args}\n\
         \n\
         Reply with ONLY JSON:\n\
         {{\"decision\":\"{REVIEW_DECISION_VOCABULARY}\",\"risk\":\"low|medium|high\",\"rationale\":\"...\"}}\n\
         Respect the user's goal and the agent's autonomy. Refuse only when the call clearly conflicts with the user's intent or has unacceptable risk; choose ask_user whenever the state does not settle it.",
        request.tool,
        request.class.as_str(),
        request.summary,
        request.can_allow_always,
    )
}

#[derive(Debug, Deserialize)]
struct ReviewerJson {
    decision: String,
    #[serde(default)]
    risk: Option<String>,
    #[serde(default)]
    rationale: Option<String>,
}

fn parse_reviewer_output(
    summary: &SubagentRunSummary,
    full_output: Option<&str>,
    can_allow_always: bool,
) -> Result<ParsedReview> {
    let raw_candidates = [
        full_output,
        summary.structured_result.raw_excerpt.as_deref(),
        Some(summary.summary.as_str()),
        summary
            .structured_result
            .findings
            .first()
            .map(String::as_str),
    ];
    for candidate in raw_candidates.into_iter().flatten() {
        if let Some(parsed) = try_parse_reviewer_json(candidate, can_allow_always) {
            return Ok(parsed);
        }
    }
    Err(anyhow!(
        "auto-review returned unparseable output: {}",
        one_line(&summary.summary, 160)
    ))
}

fn try_parse_reviewer_json(raw: &str, can_allow_always: bool) -> Option<ParsedReview> {
    let candidate = extract_json_object(raw)?;
    let parsed: ReviewerJson = serde_json::from_str(candidate).ok()?;
    let rationale = parsed
        .rationale
        .filter(|text| !text.trim().is_empty())
        .unwrap_or_else(|| "no rationale provided".into());
    let decision = parsed.decision.trim().to_ascii_lowercase();
    let outcome = match decision.as_str() {
        "execute" | "run" | "allow_once" | "allow" | "once" => AutoReviewOutcome::AllowOnce,
        "allow_always" | "always" if can_allow_always => AutoReviewOutcome::AllowAlways,
        "allow_always" | "always" => AutoReviewOutcome::AllowOnce,
        "ask_user" | "ask" => AutoReviewOutcome::Ask,
        "refuse" | "deny" | "reject" => AutoReviewOutcome::Deny,
        _ => return None,
    };
    Some(ParsedReview {
        outcome,
        rationale,
        risk: parsed.risk,
    })
}

fn extract_json_object(raw: &str) -> Option<&str> {
    let trimmed = raw.trim();
    if trimmed.starts_with('{')
        && let Ok(Value::Object(_)) = serde_json::from_str::<Value>(trimmed)
    {
        return Some(trimmed);
    }
    let start = trimmed.find('{')?;
    let end = trimmed.rfind('}')?;
    if end <= start {
        return None;
    }
    let slice = &trimmed[start..=end];
    matches!(serde_json::from_str::<Value>(slice), Ok(Value::Object(_))).then_some(slice)
}

fn one_line(text: &str, max_chars: usize) -> String {
    let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() <= max_chars {
        return flat;
    }
    let mut out = flat
        .chars()
        .take(max_chars.saturating_sub(1))
        .collect::<String>();
    out.push('…');
    out
}

#[cfg(test)]
#[path = "../../tests/internal/session/auto_review/tests.rs"]
mod tests;
