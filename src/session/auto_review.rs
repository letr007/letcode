//! Sticky reviewer expert for `PermissionMode::Auto`.
//!
//! Reuses [`SubagentPool`] + `AgentTemplate::reviewer` with one child session per
//! parent session. Writes permission decisions into the parent transcript.

use std::sync::{Arc, Mutex};

use anyhow::{Result, anyhow};
use async_openai::config::OpenAIConfig;
use serde::Deserialize;
use serde_json::Value;

use crate::agent::{Agent, AgentTemplate, AutoReviewResolution, AutoReviewService};
use crate::permission::PermissionRequest;
use crate::session::event::PermissionResolutionEvent;
use crate::session::runner::{
    PermissionResponse, SessionTransportEvent, SessionTransportEventSender, subagent_event_sender,
};
#[cfg(test)]
use crate::subagent::SubagentStatus;
use crate::subagent::{SubagentPool, SubagentRunGovernance, SubagentRunSummary};
use crate::subagent_events::run_child_prompt;
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

    pub fn clear_sticky_session(&self) {
        self.clear_sticky();
    }

    fn resolve_route(
        &self,
        parent: &Agent<OpenAIConfig>,
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

    fn route_has_api_key(
        &self,
        parent: &Agent<OpenAIConfig>,
        route: &crate::config::ModelRoute,
    ) -> Result<bool> {
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

    fn emit(&self, event: SessionTransportEvent) {
        if let Some(tx) = &self.event_tx {
            let _ = tx.send(event);
        }
    }

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
        outcome: &ParsedReview,
        child_session_id: &str,
    ) -> Result<()> {
        let mut recorder = self
            .parent_transcript
            .lock()
            .map_err(|_| anyhow!("transcript recorder poisoned"))?;
        recorder.record_permission_decision_full(
            request.call_id.clone(),
            request.tool.clone(),
            request.args.clone(),
            outcome.response.allowed(),
            Some(outcome.rationale.clone()),
            Some("auto".into()),
            Some(outcome.approval_label().into()),
            outcome.risk.clone(),
            (!child_session_id.is_empty()).then(|| child_session_id.to_string()),
        )?;
        Ok(())
    }
}

impl AutoReviewService<OpenAIConfig> for StickyAutoReviewer {
    fn review<'a>(
        &'a self,
        parent: &'a Agent<OpenAIConfig>,
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
                let rationale = self.missing_api_key_rationale(&route);
                let denied = ParsedReview {
                    response: PermissionResponse::Deny,
                    risk: Some("high".into()),
                    rationale,
                };
                self.record_decision(&request, &denied, "")?;
                self.emit_resolution(&request, &denied, "");
                return Ok(denied.into_resolution());
            }

            let goal = user_goal.or_else(|| self.latest_user_goal());
            let prompt = build_review_prompt(&request, goal.as_deref());

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
                          event_sender,
                          child_session_id,
                          _agent_name| {
                        let full_output = Arc::clone(&full_output_for_executor);
                        async move {
                            let output = run_child_prompt(
                                agent,
                                prompt,
                                transcript,
                                event_sender,
                                child_session_id,
                                None,
                            )
                            .await?;
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
                Err(error) => {
                    let parsed = ParsedReview {
                        response: PermissionResponse::Deny,
                        rationale: format!("auto-review failed: {error:#}"),
                        risk: Some("high".into()),
                    };
                    let child_id = self.sticky_child_id();
                    self.record_decision(&request, &parsed, &child_id)?;
                    self.emit_resolution(&request, &parsed, &child_id);
                    return Ok(parsed.into_resolution());
                }
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
            let parsed =
                parse_reviewer_output(&summary, full_output.as_deref(), request.can_allow_always);

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
        let call_id = request
            .call_id
            .clone()
            .unwrap_or_else(|| request.tool.clone());
        let decision = match parsed.response {
            PermissionResponse::Deny => crate::session::PermissionDecision::Denied,
            _ => crate::session::PermissionDecision::Approved,
        };
        self.emit(SessionTransportEvent::PermissionResolved(
            PermissionResolutionEvent {
                call_id,
                decision,
                reason: Some(parsed.rationale.clone()),
                tool_name: Some(request.tool.clone()),
                summary: Some(request.summary.clone()),
                origin_label: Some(REVIEWER_AGENT_NAME.into()),
                approval: Some(parsed.approval_label().into()),
                risk: parsed.risk.clone(),
                reviewer_child_session_id: (!child_session_id.is_empty())
                    .then(|| child_session_id.to_string()),
            },
        ));
    }
}

struct ParsedReview {
    response: PermissionResponse,
    rationale: String,
    risk: Option<String>,
}

impl ParsedReview {
    fn approval_label(&self) -> &'static str {
        match self.response {
            PermissionResponse::AllowOnce => "once",
            PermissionResponse::AllowAlways => "always",
            PermissionResponse::Deny => "deny",
        }
    }

    fn into_resolution(self) -> AutoReviewResolution {
        AutoReviewResolution {
            approval: match self.response {
                PermissionResponse::AllowOnce => crate::permission::PermissionApproval::AllowOnce,
                PermissionResponse::AllowAlways => {
                    crate::permission::PermissionApproval::AllowAlways
                }
                PermissionResponse::Deny => crate::permission::PermissionApproval::Deny,
            },
            reason: self.rationale,
        }
    }
}

fn build_review_prompt(request: &PermissionRequest, user_goal: Option<&str>) -> String {
    let args = serde_json::to_string_pretty(&request.args).unwrap_or_else(|_| "{}".into());
    let goal = user_goal.unwrap_or("(not provided)");
    let preview = request.preview.as_deref().unwrap_or("(none)");
    format!(
        "Approve or deny this tool permission request.\n\
         \n\
         User goal:\n{goal}\n\
         \n\
         Tool: {}\n\
         Class: {}\n\
         Execution directive: {}\n\
         Summary: {}\n\
         Preview: {preview}\n\
         can_allow_always: {}\n\
         Arguments:\n{args}\n\
         \n\
         Reply with ONLY JSON:\n\
         {{\"decision\":\"allow_once|deny\",\"risk\":\"low|medium|high\",\"rationale\":\"...\"}}\n\
         Respect the user's goal and the agent's autonomy. Deny only when the call clearly conflicts with the user's intent or has unacceptable risk.",
        request.tool,
        request.class.as_str(),
        request.directive.as_str(),
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
) -> ParsedReview {
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
            return parsed;
        }
    }
    ParsedReview {
        response: PermissionResponse::Deny,
        rationale: format!(
            "auto-review returned unparseable output: {}",
            one_line(&summary.summary, 160)
        ),
        risk: Some("high".into()),
    }
}

fn try_parse_reviewer_json(raw: &str, can_allow_always: bool) -> Option<ParsedReview> {
    let candidate = extract_json_object(raw)?;
    let parsed: ReviewerJson = serde_json::from_str(candidate).ok()?;
    let rationale = parsed
        .rationale
        .filter(|text| !text.trim().is_empty())
        .unwrap_or_else(|| "no rationale provided".into());
    let decision = parsed.decision.trim().to_ascii_lowercase();
    let response = match decision.as_str() {
        "allow_once" | "allow" | "once" => PermissionResponse::AllowOnce,
        "allow_always" | "always" if can_allow_always => PermissionResponse::AllowAlways,
        "allow_always" | "always" => PermissionResponse::AllowOnce,
        "deny" | "reject" => PermissionResponse::Deny,
        _ => return None,
    };
    Some(ParsedReview {
        response,
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
mod tests {
    use super::*;
    use async_openai::Client;

    fn test_model_config(protocol: crate::config::ApiProtocol) -> crate::config::ModelConfig {
        crate::config::ModelConfig {
            display_name: None,
            anthropic_thinking: Default::default(),
            anthropic_betas: Vec::new(),
            cache_control: false,
            protocol,
            context_window: None,
            effective_input_limit_tokens: None,
            max_output_tokens: None,
            supports_tools: false,
            supports_reasoning: false,
            reasoning_effort: None,
            reasoning_efforts: Vec::new(),
            reasoning_summary: None,
            text_verbosity: None,
            temperature: None,
            top_p: None,
            prompt_cache: crate::config::PromptCacheConfig::default(),
            parallel_tool_calls: false,
        }
    }

    fn test_provider(model: &str) -> crate::config::ProviderConfig {
        let protocol = crate::config::ApiProtocol::Completions;
        crate::config::ProviderConfig {
            base_url: "http://127.0.0.1:9/v1".into(),
            auth_mode: crate::config::ProviderAuthMode::ApiKey,
            api_key: "reviewer-key".into(),
            protocol,
            default_model: model.into(),
            retry: None,
            models: indexmap::IndexMap::from([(model.into(), test_model_config(protocol))]),
        }
    }

    fn permission_request() -> PermissionRequest {
        PermissionRequest {
            call_id: Some("call-review".into()),
            tool: "shell__exec".into(),
            args: serde_json::json!({"command": "pwd"}),
            class: crate::permission::ToolPermissionClass::Command,
            directive: crate::permission::ExecutionDirective::ReadOnly,
            summary: "Run a read-only command".into(),
            preview: Some("pwd".into()),
            can_allow_always: false,
            grant_summary: None,
        }
    }

    fn completed_summary(summary: &str) -> SubagentRunSummary {
        SubagentRunSummary {
            run_id: "r".into(),
            child_session_id: "c".into(),
            agent_name: "reviewer".into(),
            status: SubagentStatus::Completed,
            failure_kind: None,
            summary: summary.into(),
            structured_result: crate::subagent::StructuredSubagentResult {
                status: "completed".into(),
                summary: summary.into(),
                malformed: false,
                findings: Vec::new(),
                files_read: Vec::new(),
                files_changed: Vec::new(),
                commands_run: Vec::new(),
                validation: Vec::new(),
                blockers: Vec::new(),
                next_steps: Vec::new(),
                run_id: "r".into(),
                child_session_id: "c".into(),
                raw_excerpt: None,
            },
        }
    }

    #[test]
    fn unparseable_output_denies() {
        let denied = parse_reviewer_output(&completed_summary("I allow this"), None, true);
        assert!(matches!(denied.response, PermissionResponse::Deny));
        assert!(
            denied
                .rationale
                .starts_with("auto-review returned unparseable output")
        );
    }

    #[test]
    fn allow_always_is_downgraded_when_request_cannot_create_a_grant() {
        let parsed = parse_reviewer_output(
            &completed_summary(r#"{"decision":"allow_always","risk":"low","rationale":"safe"}"#),
            None,
            false,
        );
        assert!(matches!(parsed.response, PermissionResponse::AllowOnce));
    }

    #[test]
    fn full_reviewer_output_is_parsed_before_truncation_fallbacks() {
        let rationale = "safe ".repeat(80);
        let full_output =
            format!(r#"{{"decision":"allow_once","risk":"low","rationale":"{rationale}"}}"#);
        assert!(full_output.chars().count() > 240);

        let parsed = parse_reviewer_output(
            &completed_summary("truncated reviewer output"),
            Some(&full_output),
            true,
        );

        assert!(matches!(parsed.response, PermissionResponse::AllowOnce));
        assert_eq!(parsed.rationale, rationale);
    }

    #[tokio::test]
    async fn missing_reviewer_route_credential_denies_before_starting_a_child() {
        let dir = std::env::temp_dir().join(format!(
            "letcode-auto-review-credential-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).expect("sessions dir");
        let transcript = Arc::new(Mutex::new(
            TranscriptRecorder::create(&dir).expect("transcript"),
        ));
        let route = crate::config::ModelRoute::new("reviewer-provider", "reviewer-model");
        let providers =
            indexmap::IndexMap::from([(route.provider.clone(), test_provider(&route.model))]);
        let factory = Arc::new(
            crate::subagent::ExpertRouteFactory::new_with_policies(
                [(REVIEWER_AGENT_NAME.into(), Some(route.clone()), Vec::new())],
                &providers,
                &crate::config::RetryConfig::default(),
            )
            .expect("reviewer route factory"),
        );
        let mut parent = Agent::new(
            Client::with_config(OpenAIConfig::new()),
            "parent-model",
            1,
            1,
        );
        parent.set_primary_route(crate::config::ModelRoute::new("primary", "parent-model"));
        parent.set_subagent_child_factory(factory);
        let reviewer = StickyAutoReviewer::new(
            SubagentPool::new(),
            dir.clone(),
            Arc::clone(&transcript),
            None,
            Arc::new(Mutex::new(indexmap::IndexMap::from([(
                route.display_name(),
                false,
            )]))),
            Arc::new(Mutex::new(indexmap::IndexMap::from([(
                route.provider.clone(),
                "Set REVIEWER_API_KEY".into(),
            )]))),
            "Set the provider API key".into(),
        );

        let resolution = reviewer
            .review(&parent, permission_request(), None)
            .await
            .expect("credential failure should become a denial");

        assert!(matches!(
            resolution.approval,
            crate::permission::PermissionApproval::Deny
        ));
        assert!(
            resolution
                .reason
                .contains("reviewer-provider/reviewer-model")
        );
        assert!(resolution.reason.contains("Set REVIEWER_API_KEY"));
        assert!(reviewer.sticky_child_id().is_empty());
        assert!(
            !crate::transcript::child_sessions_dir(&dir).exists(),
            "credential preflight must happen before child transcript creation"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn clearing_sticky_session_allows_new_reviewer_route_after_policy_change() {
        let dir = std::env::temp_dir().join(format!(
            "letcode-auto-review-route-change-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).expect("sessions dir");
        let transcript = Arc::new(Mutex::new(
            TranscriptRecorder::create(&dir).expect("transcript"),
        ));
        let reviewer = StickyAutoReviewer::new(
            SubagentPool::new(),
            dir.clone(),
            Arc::clone(&transcript),
            None,
            Arc::new(Mutex::new(indexmap::IndexMap::new())),
            Arc::new(Mutex::new(indexmap::IndexMap::new())),
            "Set the provider API key".into(),
        );
        let child_dir = crate::transcript::child_sessions_dir(&dir);
        let mut child = TranscriptRecorder::create(&child_dir).expect("child transcript");
        let child_session_id = child.session_id().to_string();
        child
            .record_session_started("old-provider/reviewer-model")
            .expect("record child route");
        reviewer
            .inner
            .lock()
            .expect("reviewer state")
            .child_session_id = Some(child_session_id.clone());

        let new_route = crate::config::ModelRoute::new("new-provider", "reviewer-model");
        let providers = indexmap::IndexMap::from([(
            new_route.provider.clone(),
            test_provider(&new_route.model),
        )]);
        let factory = Arc::new(
            crate::subagent::ExpertRouteFactory::new_with_policies(
                [(
                    REVIEWER_AGENT_NAME.into(),
                    Some(new_route.clone()),
                    Vec::new(),
                )],
                &providers,
                &crate::config::RetryConfig::default(),
            )
            .expect("reviewer route factory"),
        );
        let mut parent = Agent::new(
            Client::with_config(OpenAIConfig::new()),
            "parent-model",
            1,
            1,
        );
        parent.set_primary_route(crate::config::ModelRoute::new("primary", "parent-model"));
        parent.set_subagent_child_factory(factory);

        let stale =
            reviewer.resolve_route(&parent, &AgentTemplate::reviewer(), Some(&child_session_id));
        assert!(
            stale
                .expect_err("old reviewer route must fail current policy")
                .to_string()
                .contains("historical model route 'old-provider/reviewer-model' is not allowed")
        );

        reviewer.clear_sticky_session();
        assert_eq!(
            reviewer
                .resolve_route(&parent, &AgentTemplate::reviewer(), None)
                .expect("cleared reviewer should resolve the new default route"),
            new_route
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn review_gate_serializes_auto_reviews() {
        let dir = std::env::temp_dir().join(format!(
            "letcode-auto-review-gate-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).expect("sessions dir");
        let transcript = Arc::new(Mutex::new(
            TranscriptRecorder::create(&dir).expect("transcript"),
        ));
        let reviewer = StickyAutoReviewer::new(
            SubagentPool::new(),
            dir.clone(),
            Arc::clone(&transcript),
            None,
            Arc::new(Mutex::new(indexmap::IndexMap::new())),
            Arc::new(Mutex::new(indexmap::IndexMap::new())),
            "Set the provider API key".into(),
        );

        let first = reviewer.review_gate.lock().await;
        assert!(
            tokio::time::timeout(
                std::time::Duration::from_millis(20),
                reviewer.review_gate.lock()
            )
            .await
            .is_err(),
            "a concurrent review must wait for the active review"
        );
        drop(first);
        let _second = tokio::time::timeout(
            std::time::Duration::from_secs(1),
            reviewer.review_gate.lock(),
        )
        .await
        .expect("queued review should proceed after the gate is released");
        let _ = std::fs::remove_dir_all(dir);
    }
}
