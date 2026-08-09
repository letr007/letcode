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
use crate::session::event::{NoticeEvent, NoticeKind, PermissionResolutionEvent};
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
        }
    }

    pub fn clear_sticky_session(&self) {
        self.clear_sticky();
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

            self.emit(SessionTransportEvent::Notice(NoticeEvent::new(
                format!("Auto-reviewing {}…", request.tool),
                NoticeKind::Info,
            )));

            let goal = user_goal.or_else(|| self.latest_user_goal());
            let prompt = build_review_prompt(&request, goal.as_deref());
            let takeover = self
                .inner
                .lock()
                .map_err(|_| anyhow!("auto-reviewer state poisoned"))?
                .child_session_id
                .clone();
            let template = AgentTemplate::reviewer();

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
                target_child_session_id: takeover.clone(),
            };
            let governance = SubagentRunGovernance {
                timeout_secs: input.effective_timeout_secs(template.timeout_secs),
                max_tool_calls: input.effective_max_tool_calls(template.max_tool_calls),
                input,
            };

            // Custom executor: SilentDeny permissions (no human UI) and no parent
            // tool-call projection — reviewer is not a delegated agent__* tool.
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
                    |agent, prompt, transcript, event_sender, child_session_id, _agent_name| {
                        async move {
                            run_child_prompt(
                                agent,
                                prompt,
                                transcript,
                                event_sender,
                                child_session_id,
                                None,
                            )
                            .await
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
                    self.emit_resolution(&request, &parsed);
                    return Ok(parsed.into_resolution(child_id));
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

            let parsed = parse_reviewer_output(&summary, request.can_allow_always);

            self.record_decision(&request, &parsed, &child_session_id)?;
            self.emit_resolution(&request, &parsed);
            self.emit(SessionTransportEvent::Notice(NoticeEvent::new(
                format!(
                    "Auto-review {}: {}",
                    parsed.approval_label(),
                    one_line(&parsed.rationale, 120)
                ),
                match parsed.response {
                    PermissionResponse::Deny => NoticeKind::RecoverableError,
                    _ => NoticeKind::Success,
                },
            )));

            Ok(parsed.into_resolution(child_session_id))
        })
    }

    fn clear_sticky(&self) {
        if let Ok(mut state) = self.inner.lock() {
            state.child_session_id = None;
        }
    }
}

impl StickyAutoReviewer {
    fn emit_resolution(&self, request: &PermissionRequest, parsed: &ParsedReview) {
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

    fn into_resolution(self, reviewer_child_session_id: String) -> AutoReviewResolution {
        let approval_label = self.approval_label();
        AutoReviewResolution {
            approval: match self.response {
                PermissionResponse::AllowOnce => crate::permission::PermissionApproval::AllowOnce,
                PermissionResponse::AllowAlways => {
                    crate::permission::PermissionApproval::AllowAlways
                }
                PermissionResponse::Deny => crate::permission::PermissionApproval::Deny,
            },
            reason: self.rationale,
            risk: self.risk,
            approval_label,
            reviewer_child_session_id,
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

fn parse_reviewer_output(summary: &SubagentRunSummary, can_allow_always: bool) -> ParsedReview {
    let raw_candidates = [
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
        let denied = parse_reviewer_output(&completed_summary("I allow this"), true);
        assert!(matches!(denied.response, PermissionResponse::Deny));
    }

    #[test]
    fn allow_always_is_downgraded_when_request_cannot_create_a_grant() {
        let parsed = parse_reviewer_output(
            &completed_summary(r#"{"decision":"allow_always","risk":"low","rationale":"safe"}"#),
            false,
        );
        assert!(matches!(parsed.response, PermissionResponse::AllowOnce));
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
