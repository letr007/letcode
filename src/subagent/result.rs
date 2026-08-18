use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubagentStatus {
    Running,
    Completed,
    Failed,
    BudgetExhausted,
    Cancelled,
    TimedOut,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubagentFailureKind {
    Hard,
    Logical,
}

impl SubagentFailureKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Hard => "hard",
            Self::Logical => "logical",
        }
    }
}

impl SubagentStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::BudgetExhausted => "budget_exhausted",
            Self::Cancelled => "cancelled",
            Self::TimedOut => "timed_out",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubagentRunSummary {
    pub run_id: String,
    pub child_session_id: String,
    pub agent_name: String,
    pub status: SubagentStatus,
    pub failure_kind: Option<SubagentFailureKind>,
    pub summary: String,
    pub structured_result: StructuredSubagentResult,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StructuredSubagentResult {
    pub status: String,
    pub summary: String,
    pub malformed: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub findings: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub files_read: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub files_changed: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub commands_run: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub validation: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub blockers: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub next_steps: Vec<String>,
    pub run_id: String,
    pub child_session_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_excerpt: Option<String>,
}

impl StructuredSubagentResult {
    pub(crate) fn from_model_output(
        raw: &str,
        fallback_status: SubagentStatus,
        run_id: &str,
        child_session_id: &str,
    ) -> Self {
        let candidate = extract_json_candidate(raw).unwrap_or(raw.trim());
        match serde_json::from_str::<Value>(candidate) {
            Ok(Value::Object(map)) => {
                let value = Value::Object(map);
                let findings = list_field(&value, "findings");
                let summary = string_field(&value, "summary")
                    .filter(|text| !text.is_empty())
                    .or_else(|| findings.first().cloned())
                    .unwrap_or_else(|| excerpt(raw));
                Self {
                    status: string_field(&value, "status")
                        .filter(|text| !text.is_empty())
                        .unwrap_or_else(|| fallback_status.as_str().to_string()),
                    summary,
                    malformed: false,
                    findings,
                    files_read: list_field(&value, "files_read"),
                    files_changed: list_field(&value, "files_changed"),
                    commands_run: list_field(&value, "commands_run"),
                    validation: validation_field(&value),
                    blockers: list_field(&value, "blockers"),
                    next_steps: list_field(&value, "next_steps"),
                    run_id: run_id.to_string(),
                    child_session_id: child_session_id.to_string(),
                    raw_excerpt: None,
                }
            }
            _ => Self {
                status: fallback_status.as_str().to_string(),
                summary: excerpt(raw),
                malformed: true,
                findings: Vec::new(),
                files_read: Vec::new(),
                files_changed: Vec::new(),
                commands_run: Vec::new(),
                validation: Vec::new(),
                blockers: Vec::new(),
                next_steps: Vec::new(),
                run_id: run_id.to_string(),
                child_session_id: child_session_id.to_string(),
                raw_excerpt: Some(excerpt(raw)),
            },
        }
    }

    pub(crate) fn from_runtime_status(
        status: SubagentStatus,
        summary: String,
        run_id: &str,
        child_session_id: &str,
    ) -> Self {
        Self {
            status: status.as_str().to_string(),
            blockers: matches!(
                status,
                SubagentStatus::Failed
                    | SubagentStatus::BudgetExhausted
                    | SubagentStatus::Cancelled
                    | SubagentStatus::TimedOut
            )
            .then(|| vec![summary.clone()])
            .unwrap_or_default(),
            summary,
            malformed: false,
            findings: Vec::new(),
            files_read: Vec::new(),
            files_changed: Vec::new(),
            commands_run: Vec::new(),
            validation: Vec::new(),
            next_steps: Vec::new(),
            run_id: run_id.to_string(),
            child_session_id: child_session_id.to_string(),
            raw_excerpt: None,
        }
    }
}

pub(crate) fn build_completed_summary(
    run_id: &str,
    child_session_id: &str,
    agent_name: &str,
    message: String,
) -> SubagentRunSummary {
    let structured_result = StructuredSubagentResult::from_model_output(
        &message,
        SubagentStatus::Completed,
        run_id,
        child_session_id,
    );
    let status = map_structured_status(&structured_result.status);
    let failure_kind =
        (status != SubagentStatus::Completed).then_some(SubagentFailureKind::Logical);
    SubagentRunSummary {
        run_id: run_id.to_string(),
        child_session_id: child_session_id.to_string(),
        agent_name: agent_name.to_string(),
        status,
        failure_kind,
        summary: structured_result.summary.clone(),
        structured_result,
    }
}

pub(crate) fn build_runtime_summary(
    run_id: &str,
    child_session_id: &str,
    agent_name: &str,
    status: SubagentStatus,
    summary: String,
) -> SubagentRunSummary {
    let structured_result = StructuredSubagentResult::from_runtime_status(
        status,
        summary.clone(),
        run_id,
        child_session_id,
    );
    let failure_kind = (status != SubagentStatus::Completed).then_some(SubagentFailureKind::Hard);
    SubagentRunSummary {
        run_id: run_id.to_string(),
        child_session_id: child_session_id.to_string(),
        agent_name: agent_name.to_string(),
        status,
        failure_kind,
        summary,
        structured_result,
    }
}

fn extract_json_candidate(raw: &str) -> Option<&str> {
    let trimmed = raw.trim();
    if trimmed.starts_with("```") {
        let body = trimmed
            .trim_start_matches("```")
            .trim_start_matches("json")
            .trim();
        return body.strip_suffix("```").map(str::trim);
    }
    if trimmed.starts_with('{') && trimmed.ends_with('}') {
        return Some(trimmed);
    }
    let start = trimmed.find('{')?;
    let end = trimmed.rfind('}')?;
    (end > start).then_some(trimmed[start..=end].trim())
}

pub fn looks_like_structured_subagent_output(raw: &str) -> bool {
    let compact = raw.trim_start();
    (compact.starts_with('{') || compact.starts_with("```"))
        && compact.contains("\"status\"")
        && compact.contains("\"summary\"")
}

pub fn try_parse_structured_subagent_result(raw: &str) -> Option<StructuredSubagentResult> {
    let candidate = extract_json_candidate(raw)?;
    let value = serde_json::from_str::<Value>(candidate).ok()?;
    let object = structured_result_object(&value)?;
    let status = string_field(&Value::Object(object.clone()), "status")?;
    let summary = string_field(&Value::Object(object.clone()), "summary").or_else(|| {
        object
            .get("summary")
            .and_then(|summary| summary.get("conclusion"))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|summary| !summary.is_empty())
            .map(str::to_string)
    })?;
    const LIST_FIELDS: [&str; 7] = [
        "blockers",
        "findings",
        "next_steps",
        "validation",
        "files_read",
        "files_changed",
        "commands_run",
    ];
    if !LIST_FIELDS.iter().any(|field| object.contains_key(*field)) {
        return None;
    }

    let object = Value::Object(object.clone());
    Some(StructuredSubagentResult {
        status,
        summary,
        malformed: false,
        findings: list_field(&object, "findings"),
        files_read: list_field(&object, "files_read"),
        files_changed: list_field(&object, "files_changed"),
        commands_run: list_field(&object, "commands_run"),
        validation: validation_field(&object),
        blockers: list_field(&object, "blockers"),
        next_steps: list_field(&object, "next_steps"),
        run_id: String::new(),
        child_session_id: String::new(),
        raw_excerpt: None,
    })
}

fn structured_result_object(value: &Value) -> Option<&serde_json::Map<String, Value>> {
    [
        value.as_object(),
        value.get("data").and_then(Value::as_object),
        value.get("result").and_then(Value::as_object),
        value.get("structured_result").and_then(Value::as_object),
    ]
    .into_iter()
    .flatten()
    .find(|object| {
        object.get("status").and_then(Value::as_str).is_some() && object.contains_key("summary")
    })
}

fn string_field(value: &Value, key: &str) -> Option<String> {
    match value.get(key) {
        Some(Value::String(text)) => Some(text.trim().to_string()).filter(|text| !text.is_empty()),
        Some(Value::Number(number)) => Some(number.to_string()),
        Some(Value::Bool(flag)) => Some(flag.to_string()),
        _ => None,
    }
}

fn list_field(value: &Value, key: &str) -> Vec<String> {
    match value.get(key) {
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(|item| match item {
                Value::String(text) => {
                    Some(text.trim().to_string()).filter(|text| !text.is_empty())
                }
                Value::Number(number) => Some(number.to_string()),
                Value::Bool(flag) => Some(flag.to_string()),
                Value::Object(map) => map
                    .get("path")
                    .or_else(|| map.get("file"))
                    .or_else(|| map.get("command"))
                    .or_else(|| map.get("evidence"))
                    .or_else(|| map.get("summary"))
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .map(str::to_string)
                    .filter(|text| !text.is_empty()),
                _ => None,
            })
            .collect(),
        Some(Value::String(text)) => {
            let trimmed = text.trim();
            if trimmed.is_empty() {
                Vec::new()
            } else {
                vec![trimmed.to_string()]
            }
        }
        _ => Vec::new(),
    }
}

fn validation_field(value: &Value) -> Vec<String> {
    match value.get("validation") {
        Some(Value::Array(items)) => items.iter().filter_map(validation_item_summary).collect(),
        Some(Value::String(text)) => {
            let trimmed = text.trim();
            if trimmed.is_empty() {
                Vec::new()
            } else {
                vec![trimmed.to_string()]
            }
        }
        _ => Vec::new(),
    }
}

fn validation_item_summary(item: &Value) -> Option<String> {
    match item {
        Value::String(text) => Some(text.trim().to_string()).filter(|text| !text.is_empty()),
        Value::Object(map) => {
            let command = map
                .get("command")
                .or_else(|| map.get("name"))
                .or_else(|| map.get("summary"))
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|text| !text.is_empty())
                .unwrap_or("validation");
            let outcome = map
                .get("status")
                .or_else(|| map.get("result"))
                .or_else(|| map.get("outcome"))
                .or_else(|| map.get("state"))
                .and_then(|value| match value {
                    Value::String(text) => Some(text.trim().to_string()),
                    Value::Bool(flag) => Some(if *flag { "passed" } else { "failed" }.into()),
                    Value::Number(number) => Some(number.to_string()),
                    _ => None,
                });
            Some(match outcome {
                Some(outcome) if !outcome.is_empty() => format!("{command} {outcome}"),
                _ => command.to_string(),
            })
        }
        Value::Bool(flag) => Some(if *flag { "passed" } else { "failed" }.into()),
        Value::Number(number) => Some(number.to_string()),
        _ => None,
    }
}

fn excerpt(raw: &str) -> String {
    let compact = raw.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut chars = compact.chars();
    let excerpt = chars.by_ref().take(240).collect::<String>();
    if chars.next().is_some() {
        format!("{excerpt}…")
    } else if excerpt.is_empty() {
        "subagent produced empty output".into()
    } else {
        excerpt
    }
}

pub(crate) fn classify_failure_status(message: &str) -> SubagentStatus {
    if message.contains("stopped: too many tool calls") {
        SubagentStatus::BudgetExhausted
    } else {
        SubagentStatus::Failed
    }
}

fn map_structured_status(status: &str) -> SubagentStatus {
    match status.trim().to_ascii_lowercase().as_str() {
        "completed" | "succeeded" | "success" => SubagentStatus::Completed,
        "cancelled" | "canceled" => SubagentStatus::Cancelled,
        "timed_out" | "timed out" | "timeout" => SubagentStatus::TimedOut,
        "budget_exhausted" | "budget exhausted" => SubagentStatus::BudgetExhausted,
        "failed" | "error" | "blocked" => SubagentStatus::Failed,
        _ => SubagentStatus::Completed,
    }
}
