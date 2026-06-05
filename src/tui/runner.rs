use std::sync::{Arc, Mutex};

use anyhow::{Result, anyhow};
use async_openai::config::Config;
use serde_json::Value;
use tokio::sync::{mpsc, oneshot};

use crate::agent::{Agent, AgentEvent, ConversationMessage};
use crate::permission::PermissionRequest;
use crate::tool::ToolResult;
use crate::tool_format::format_tool_call;
use crate::transcript::{TranscriptRecord, TranscriptRecorder};

use super::events::{
    AppEvent, AssistantDeltaEvent, AutoContinueChangedEvent, ErrorEvent, PermissionRequestEvent,
    PermissionResolutionEvent, ReasoningDeltaEvent, ReasoningDoneEvent, TodoSnapshotEvent,
    TokenUsageEvent, ToolFinishedEvent, ToolOutcome, ToolStartedEvent, UserMessageEvent,
};

pub type RunnerEventSender = mpsc::UnboundedSender<RunnerEvent>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PermissionResponse {
    Approve,
    Deny,
}

impl PermissionResponse {
    pub fn allowed(self) -> bool {
        matches!(self, Self::Approve)
    }
}

#[derive(Debug, Clone)]
pub struct RunnerPermissionRequest {
    sender: Arc<Mutex<Option<oneshot::Sender<PermissionResponse>>>>,
}

impl RunnerPermissionRequest {
    pub fn new(sender: oneshot::Sender<PermissionResponse>) -> Self {
        Self {
            sender: Arc::new(Mutex::new(Some(sender))),
        }
    }

    pub fn respond(&self, response: PermissionResponse) -> Result<()> {
        let sender = self
            .sender
            .lock()
            .map_err(|_| anyhow!("permission request lock poisoned"))?
            .take()
            .ok_or_else(|| anyhow!("permission request already resolved"))?;

        sender
            .send(response)
            .map_err(|_| anyhow!("permission response receiver dropped"))
    }

    pub fn approve(&self) -> Result<()> {
        self.respond(PermissionResponse::Approve)
    }

    pub fn deny(&self) -> Result<()> {
        self.respond(PermissionResponse::Deny)
    }
}

#[derive(Debug, Clone)]
pub enum RunnerEvent {
    UserMessage(UserMessageEvent),
    ReasoningDelta(ReasoningDeltaEvent),
    ReasoningDone(ReasoningDoneEvent),
    AssistantDelta(AssistantDeltaEvent),
    AssistantDone {
        message_id: Option<String>,
    },
    TokenUsage(TokenUsageEvent),
    ToolStarted(ToolStartedEvent),
    ToolFinished(ToolFinishedEvent),
    TodoSnapshot(TodoSnapshotEvent),
    AutoContinueChanged(AutoContinueChangedEvent),
    PermissionRequested {
        event: PermissionRequestEvent,
        handle: RunnerPermissionRequest,
    },
    PermissionResolved(PermissionResolutionEvent),
    SessionResumed {
        session_id: String,
        messages: Vec<ConversationMessage>,
        records: Vec<TranscriptRecord>,
        evidence_count: usize,
    },
    SessionStarted {
        session_id: String,
    },
    Error(ErrorEvent),
    Done,
}

impl RunnerEvent {
    pub fn app_event(&self) -> Option<AppEvent> {
        match self {
            Self::UserMessage(event) => Some(AppEvent::UserMessage(event.clone())),
            Self::ReasoningDelta(event) => Some(AppEvent::ReasoningDelta(event.clone())),
            Self::ReasoningDone(event) => Some(AppEvent::ReasoningDone(event.clone())),
            Self::AssistantDelta(event) => Some(AppEvent::AssistantDelta(event.clone())),
            Self::AssistantDone { message_id } => Some(AppEvent::AssistantDone {
                message_id: message_id.clone(),
            }),
            Self::TokenUsage(event) => Some(AppEvent::TokenUsage(*event)),
            Self::ToolStarted(event) => Some(AppEvent::ToolStarted(event.clone())),
            Self::ToolFinished(event) => Some(AppEvent::ToolFinished(event.clone())),
            Self::TodoSnapshot(event) => Some(AppEvent::TodoSnapshot(event.clone())),
            Self::AutoContinueChanged(event) => Some(AppEvent::AutoContinueChanged(event.clone())),
            Self::PermissionRequested { event, .. } => {
                Some(AppEvent::PermissionRequested(event.clone()))
            }
            Self::PermissionResolved(event) => Some(AppEvent::PermissionResolved(event.clone())),
            Self::SessionResumed { .. } | Self::SessionStarted { .. } => None,
            Self::Error(event) => Some(AppEvent::Error(event.clone())),
            Self::Done => Some(AppEvent::Done),
        }
    }
}

pub struct AgentRunner<C: Config> {
    event_tx: RunnerEventSender,
    transcript: Option<Arc<Mutex<TranscriptRecorder>>>,
    _config: std::marker::PhantomData<C>,
}

impl<C: Config> AgentRunner<C> {
    pub fn new(event_tx: RunnerEventSender) -> Self {
        Self {
            event_tx,
            transcript: None,
            _config: std::marker::PhantomData,
        }
    }

    pub fn with_transcript(
        event_tx: RunnerEventSender,
        transcript: Arc<Mutex<TranscriptRecorder>>,
    ) -> Self {
        Self {
            event_tx,
            transcript: Some(transcript),
            _config: std::marker::PhantomData,
        }
    }

    pub async fn run_prompt(
        &self,
        agent: &mut Agent<C>,
        prompt: impl Into<String>,
    ) -> Result<String> {
        let prompt = prompt.into();
        let user_event = UserMessageEvent::new(prompt.clone());
        self.emit(RunnerEvent::UserMessage(user_event))?;
        self.record(|recorder| recorder.record_user_message(prompt.clone()))
            .or_else(|error| self.finish_with_error(error))?;

        let sender = self.event_tx.clone();
        let response = agent
            .run_stream_async(
                &prompt,
                move |delta| {
                    let sender = sender.clone();
                    let delta = delta.to_string();
                    async move {
                        sender
                            .send(RunnerEvent::AssistantDelta(AssistantDeltaEvent::new(delta)))
                            .map_err(|_| anyhow!("runner event channel closed"))
                    }
                },
                {
                    let sender = self.event_tx.clone();
                    let transcript = self.transcript.clone();
                    move |event| {
                        let sender = sender.clone();
                        let transcript = transcript.clone();
                        async move {
                            match event {
                                AgentEvent::TokenUsageUpdated {
                                    used_tokens,
                                    context_window_tokens,
                                } => {
                                    sender
                                        .send(RunnerEvent::TokenUsage(TokenUsageEvent::new(
                                            used_tokens,
                                            context_window_tokens,
                                        )))
                                        .map_err(|_| anyhow!("runner event channel closed"))?;
                                }
                                AgentEvent::EvidenceRecorded(evidence) => {
                                    record_transcript(&transcript, |recorder| {
                                        recorder.record_evidence_record(evidence.clone())
                                    })?;
                                }
                                AgentEvent::ReasoningDelta { item_id, delta } => {
                                    sender
                                        .send(RunnerEvent::ReasoningDelta(
                                            ReasoningDeltaEvent::new(item_id, delta),
                                        ))
                                        .map_err(|_| anyhow!("runner event channel closed"))?;
                                }
                                AgentEvent::ReasoningDone { item_id, text } => {
                                    record_transcript(&transcript, |recorder| {
                                        recorder.record_reasoning_message(text.clone())
                                    })?;
                                    sender
                                        .send(RunnerEvent::ReasoningDone(ReasoningDoneEvent::new(
                                            item_id, text,
                                        )))
                                        .map_err(|_| anyhow!("runner event channel closed"))?;
                                }
                                AgentEvent::ToolCallStarted {
                                    call_id,
                                    name,
                                    args,
                                } => {
                                    let started = tool_started_event(call_id, name, args);
                                    record_transcript(&transcript, |recorder| {
                                        recorder.record_tool_call_started(
                                            started.call_id.clone(),
                                            started.name.clone(),
                                            parse_arguments(&started.arguments),
                                        )
                                    })?;
                                    sender
                                        .send(RunnerEvent::ToolStarted(started))
                                        .map_err(|_| anyhow!("runner event channel closed"))?;
                                }
                                AgentEvent::ToolCallFinished {
                                    call_id,
                                    name,
                                    ok,
                                    output,
                                } => {
                                    let finished =
                                        tool_finished_event(call_id, name, ok, output.clone());
                                    record_transcript(&transcript, |recorder| {
                                        recorder.record_tool_call_finished(
                                            finished.call_id.clone(),
                                            finished.name.clone(),
                                            ok,
                                            output.clone(),
                                        )
                                    })?;
                                    sender
                                        .send(RunnerEvent::ToolFinished(finished))
                                        .map_err(|_| anyhow!("runner event channel closed"))?;
                                }
                                AgentEvent::TodoSnapshotUpdated { items } => {
                                    record_transcript(&transcript, |recorder| {
                                        recorder.record_todo_snapshot(items.clone())
                                    })?;
                                    sender
                                        .send(RunnerEvent::TodoSnapshot(TodoSnapshotEvent::new(
                                            items,
                                        )))
                                        .map_err(|_| anyhow!("runner event channel closed"))?;
                                }
                                AgentEvent::AutoContinueChanged { state } => {
                                    record_transcript(&transcript, |recorder| {
                                        recorder.record_auto_continue_changed(state.clone())
                                    })?;
                                    sender
                                        .send(RunnerEvent::AutoContinueChanged(
                                            AutoContinueChangedEvent::new(state),
                                        ))
                                        .map_err(|_| anyhow!("runner event channel closed"))?;
                                }
                            }

                            Ok(())
                        }
                    }
                },
                {
                    let sender = self.event_tx.clone();
                    let transcript = self.transcript.clone();
                    move |request| {
                        let sender = sender.clone();
                        let transcript = transcript.clone();
                        async move {
                            let request_event = permission_request_event(&request);
                            let (response_tx, response_rx) = oneshot::channel();
                            let handle = RunnerPermissionRequest::new(response_tx);
                            sender
                                .send(RunnerEvent::PermissionRequested {
                                    event: request_event.clone(),
                                    handle,
                                })
                                .map_err(|_| anyhow!("runner event channel closed"))?;

                            let response = response_rx
                                .await
                                .map_err(|_| anyhow!("permission response sender dropped"))?;
                            let resolution = permission_resolution_event(&request, response);
                            record_transcript(&transcript, |recorder| {
                                recorder.record_permission_decision_details(
                                    request.call_id.clone(),
                                    request.tool.clone(),
                                    request.args.clone(),
                                    response.allowed(),
                                    resolution.reason.clone(),
                                )
                            })?;
                            sender
                                .send(RunnerEvent::PermissionResolved(resolution))
                                .map_err(|_| anyhow!("runner event channel closed"))?;

                            Ok(response.allowed())
                        }
                    }
                },
            )
            .await;

        match response {
            Ok(message) => {
                self.record(|recorder| recorder.record_assistant_message(message.clone()))
                    .or_else(|error| self.finish_with_error(error))?;
                self.emit(RunnerEvent::AssistantDone { message_id: None })?;
                self.emit(RunnerEvent::Done)?;
                Ok(message)
            }
            Err(error) => {
                let event = ErrorEvent::new(error.to_string());
                if let Err(record_error) =
                    self.record(|recorder| recorder.record_error(error.to_string()))
                {
                    let composite_message = format!(
                        "{} (additionally failed to record transcript error: {})",
                        error, record_error
                    );
                    self.finish_with_error(anyhow!(composite_message.clone()))?;
                    return Err(anyhow!(composite_message));
                }
                self.emit(RunnerEvent::Error(event))?;
                self.emit(RunnerEvent::Done)?;
                Err(error)
            }
        }
    }

    pub fn record_model_changed(&self, previous_model: &str, new_model: &str) -> Result<()> {
        self.record(|recorder| {
            recorder.record_model_changed(previous_model.to_string(), new_model.to_string())
        })
    }

    pub fn record_permission_mode_changed(
        &self,
        previous_mode: &str,
        new_mode: &str,
    ) -> Result<()> {
        self.record(|recorder| {
            recorder.record_permission_mode_changed(previous_mode.to_string(), new_mode.to_string())
        })
    }

    fn emit(&self, event: RunnerEvent) -> Result<()> {
        self.event_tx
            .send(event)
            .map_err(|_| anyhow!("runner event channel closed"))
    }

    fn record<F>(&self, f: F) -> Result<()>
    where
        F: FnOnce(&mut TranscriptRecorder) -> Result<()>,
    {
        record_transcript(&self.transcript, f)
    }

    fn finish_with_error(&self, error: anyhow::Error) -> Result<()> {
        let event = ErrorEvent::new(error.to_string());
        self.emit(RunnerEvent::Error(event))?;
        self.emit(RunnerEvent::Done)?;
        Err(error)
    }
}

fn record_transcript<F>(transcript: &Option<Arc<Mutex<TranscriptRecorder>>>, f: F) -> Result<()>
where
    F: FnOnce(&mut TranscriptRecorder) -> Result<()>,
{
    let Some(transcript) = transcript else {
        return Ok(());
    };

    let mut recorder = transcript
        .lock()
        .map_err(|_| anyhow!("transcript recorder poisoned"))?;
    f(&mut recorder)
}

fn tool_started_event(call_id: String, name: String, args: Value) -> ToolStartedEvent {
    let summary = format_tool_call(&name, &args);
    ToolStartedEvent {
        call_id,
        name,
        summary,
        arguments: Some(args.to_string()),
    }
}

fn tool_finished_event(
    call_id: String,
    name: String,
    ok: bool,
    output: ToolResult,
) -> ToolFinishedEvent {
    let summary = output_summary(&output).unwrap_or_else(|| name.clone());
    ToolFinishedEvent {
        call_id,
        name,
        summary,
        outcome: if ok {
            ToolOutcome::Success
        } else {
            ToolOutcome::Failure
        },
        output: Some(output_json(&output).to_string()),
    }
}

fn permission_request_event(request: &PermissionRequest) -> PermissionRequestEvent {
    let mut event = PermissionRequestEvent::new(
        request
            .call_id
            .clone()
            .unwrap_or_else(|| request.tool.clone()),
        request.tool.clone(),
        request.summary.clone(),
    );
    event.arguments = Some(request.args.to_string());
    event.rationale = Some(format!("{} permission requires approval", request.class));
    event
}

fn permission_resolution_event(
    request: &PermissionRequest,
    response: PermissionResponse,
) -> PermissionResolutionEvent {
    let call_id = request
        .call_id
        .clone()
        .unwrap_or_else(|| request.tool.clone());
    match response {
        PermissionResponse::Approve => PermissionResolutionEvent::approved(call_id),
        PermissionResponse::Deny => PermissionResolutionEvent::denied(
            call_id,
            Some("Denied by user from TUI permission prompt".into()),
        ),
    }
}

fn parse_arguments(arguments: &Option<String>) -> Value {
    arguments
        .as_deref()
        .and_then(|text| serde_json::from_str(text).ok())
        .unwrap_or(Value::Null)
}

fn output_summary(output: &ToolResult) -> Option<String> {
    if let Some(error) = &output.error {
        return Some(error.message.clone());
    }

    let data = output.data.as_ref()?;
    Some(match output.tool.as_str() {
        "util__echo" => summarize_echo(data),
        "fs__list" => summarize_array_count(data, "entries", "entries"),
        "fs__read" => summarize_read_file(data),
        "fs__write" => summarize_bytes(data, "bytes_written", "wrote"),
        "fs__append" => summarize_bytes(data, "bytes_appended", "appended"),
        "fs__mkdir" => summarize_path_action(data, "created"),
        "search__rg" => summarize_array_count(data, "matches", "matches"),
        "shell__exec" | "git__status" | "git__diff" | "git__log" => summarize_command(data),
        "edit__apply_patch" => summarize_apply_patch(data),
        "code__ast_search" => summarize_array_count(data, "matches", "matches"),
        "code__ast_replace_preview" => summarize_array_count(data, "replacements", "replacements"),
        "workflow__todos" => summarize_todos(data),
        "workflow__auto_continue" => summarize_auto_continue(data),
        _ => summarize_generic(data),
    })
}

fn summarize_todos(data: &Value) -> String {
    let count = data
        .get("items")
        .and_then(Value::as_array)
        .map(Vec::len)
        .unwrap_or(0);
    format!("updated {count} todos")
}

fn summarize_auto_continue(data: &Value) -> String {
    let enabled = data
        .get("enabled")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let max = data
        .get("max_continuations")
        .and_then(Value::as_u64)
        .map(|value| format!(" · max {value}"))
        .unwrap_or_default();
    if enabled {
        format!("enabled auto-continue{max}")
    } else {
        "disabled auto-continue".into()
    }
}

fn summarize_echo(data: &Value) -> String {
    let chars = data
        .get("result")
        .and_then(Value::as_str)
        .map(str::chars)
        .map(Iterator::count)
        .unwrap_or(0);
    format!("returned {chars} chars")
}

fn summarize_array_count(data: &Value, key: &str, label: &str) -> String {
    let count = data
        .get(key)
        .and_then(Value::as_array)
        .map(Vec::len)
        .unwrap_or(0);
    let truncated = data
        .get("truncated")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if truncated {
        format!("{count} {label} shown · truncated")
    } else {
        format!("{count} {label}")
    }
}

fn summarize_read_file(data: &Value) -> String {
    let path = data.get("path").and_then(Value::as_str).unwrap_or("file");
    let lines = data.get("lines_read").and_then(Value::as_u64).unwrap_or(0);
    let start = data.get("start_line").and_then(Value::as_u64);
    let end = data.get("end_line").and_then(Value::as_u64);
    let suffix = if data
        .get("truncated")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        " · has more"
    } else {
        ""
    };

    match (start, end) {
        (Some(start), Some(end)) => format!("read {path}:{start}-{end} ({lines} lines){suffix}"),
        _ => format!("read {path} ({lines} lines){suffix}"),
    }
}

fn summarize_bytes(data: &Value, key: &str, verb: &str) -> String {
    let path = data.get("path").and_then(Value::as_str).unwrap_or("file");
    let bytes = data.get(key).and_then(Value::as_u64).unwrap_or(0);
    format!("{verb} {bytes} bytes to {path}")
}

fn summarize_path_action(data: &Value, action: &str) -> String {
    let path = data.get("path").and_then(Value::as_str).unwrap_or("path");
    format!("{action} {path}")
}

fn summarize_command(data: &Value) -> String {
    if let Some(error) = data.get("error").and_then(Value::as_str) {
        return error.to_string();
    }

    let status = data
        .get("status")
        .and_then(Value::as_i64)
        .map(|status| format!("exit {status}"))
        .unwrap_or_else(|| "completed".to_string());
    let stdout = output_line_count(data, "stdout", "stdout_truncated");
    let stderr = output_line_count(data, "stderr", "stderr_truncated");
    let mut parts = vec![status];
    if let Some(stdout) = stdout {
        parts.push(format!("stdout {stdout}"));
    }
    if let Some(stderr) = stderr {
        parts.push(format!("stderr {stderr}"));
    }
    parts.join(" · ")
}

fn output_line_count(data: &Value, key: &str, truncated_key: &str) -> Option<String> {
    let text = data.get(key).and_then(Value::as_str)?;
    if text.trim().is_empty() {
        return None;
    }
    let count = text.lines().count().max(1);
    let suffix = if data
        .get(truncated_key)
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        "+"
    } else {
        ""
    };
    Some(format!("{count}{suffix} lines"))
}

fn summarize_apply_patch(data: &Value) -> String {
    let files = data
        .get("files_changed")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let edits = data
        .get("edits_applied")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    format!("patched {files} files · {edits} edits")
}

fn summarize_generic(data: &Value) -> String {
    match data {
        Value::Array(items) => format!("{} items", items.len()),
        Value::Object(fields) => format!("{} fields", fields.len()),
        Value::String(text) => format!("returned {} chars", text.chars().count()),
        Value::Null => "completed".into(),
        _ => "completed".into(),
    }
}

fn output_json(output: &ToolResult) -> Value {
    serde_json::to_value(output)
        .unwrap_or_else(|_| Value::String("<unserializable tool output>".into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{AutoContinueState, TodoItem, TodoStatus};
    use crate::transcript::TranscriptRecorder;
    use crate::tui::events::{AppEvent, PermissionDecision};
    use async_openai::{Client, config::OpenAIConfig};
    use serde_json::json;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[tokio::test]
    async fn permission_request_handle_delivers_approval() {
        let (tx, rx) = oneshot::channel();
        let handle = RunnerPermissionRequest::new(tx);

        handle.approve().expect("approve succeeds");

        assert_eq!(
            rx.await.expect("receiver gets response"),
            PermissionResponse::Approve
        );
    }

    #[test]
    fn permission_resolution_event_maps_denial() {
        let request = PermissionRequest {
            call_id: Some("call-7".into()),
            tool: "shell__exec".into(),
            args: Value::Null,
            class: crate::permission::ToolPermissionClass::Command,
            summary: "shell__exec cargo test".into(),
            preview: None,
        };

        let resolution = permission_resolution_event(&request, PermissionResponse::Deny);

        assert_eq!(resolution.call_id, "call-7");
        assert_eq!(resolution.decision, PermissionDecision::Denied);
        assert!(resolution.reason.is_some());
    }

    #[test]
    fn tool_output_summary_avoids_dumping_json_payloads() {
        let output = ToolResult::ok(
            "util__echo",
            serde_json::json!({ "result": "已调用工具。" }),
        );

        assert_eq!(output_summary(&output).as_deref(), Some("returned 6 chars"));

        let read = ToolResult::ok(
            "fs__read",
            serde_json::json!({
                "path": "src/main.rs",
                "start_line": 10,
                "end_line": 20,
                "lines_read": 11,
                "truncated": true
            }),
        );

        assert_eq!(
            output_summary(&read).as_deref(),
            Some("read src/main.rs:10-20 (11 lines) · has more")
        );
    }

    #[test]
    fn command_summary_reports_counts_not_output_text() {
        let output = ToolResult::ok(
            "shell__exec",
            serde_json::json!({
                "command": "cargo test",
                "status": 0,
                "success": true,
                "stdout": "line one\nline two\n",
                "stdout_truncated": false,
                "stderr": "warning\n",
                "stderr_truncated": true
            }),
        );

        assert_eq!(
            output_summary(&output).as_deref(),
            Some("exit 0 · stdout 2 lines · stderr 1+ lines")
        );
    }

    #[test]
    fn workflow_control_tools_have_compact_summaries() {
        let todos = ToolResult::ok(
            "workflow__todos",
            serde_json::json!({
                "items": [
                    {"id": "t1", "content": "one", "status": "pending"},
                    {"id": "t2", "content": "two", "status": "completed"}
                ]
            }),
        );
        assert_eq!(output_summary(&todos).as_deref(), Some("updated 2 todos"));

        let auto_continue = ToolResult::ok(
            "workflow__auto_continue",
            serde_json::json!({"enabled": true, "max_continuations": 2}),
        );
        assert_eq!(
            output_summary(&auto_continue).as_deref(),
            Some("enabled auto-continue · max 2")
        );
    }

    #[tokio::test]
    async fn transcript_failure_emits_error_and_done() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let runner = AgentRunner::with_transcript(tx, poisoned_transcript());
        let config = OpenAIConfig::new()
            .with_api_base("https://api.openai.com/v1")
            .with_api_key("test-key");
        let client = Client::with_config(config);
        let mut agent = Agent::new(client, "gpt-5.5", 1, 1);

        let error = runner
            .run_prompt(&mut agent, "hello")
            .await
            .expect_err("transcript failure should error");

        assert!(error.to_string().contains("transcript recorder poisoned"));

        assert!(matches!(
            rx.recv().await,
            Some(RunnerEvent::UserMessage(UserMessageEvent { content, .. })) if content == "hello"
        ));
        assert!(matches!(
            rx.recv().await,
            Some(RunnerEvent::Error(ErrorEvent { message, .. })) if message.contains("transcript recorder poisoned")
        ));
        assert!(matches!(rx.recv().await, Some(RunnerEvent::Done)));
    }

    #[test]
    fn runner_can_record_model_and_permission_provenance_events() {
        let base_dir = std::env::temp_dir().join(format!(
            "letcode-runner-provenance-test-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("time ok")
                .as_nanos()
        ));
        let recorder = TranscriptRecorder::create(&base_dir).expect("create transcript");
        let session_id = recorder.session_id().to_string();
        let transcript = Arc::new(Mutex::new(recorder));
        let (tx, _rx) = mpsc::unbounded_channel();
        let runner = AgentRunner::<OpenAIConfig>::with_transcript(tx, transcript);

        runner
            .record_model_changed("gpt-5.5", "gpt-5.5-mini")
            .expect("record model");
        runner
            .record_permission_mode_changed("default", "safe")
            .expect("record permission");

        let records = crate::transcript::read_records(base_dir.join(format!("{session_id}.jsonl")))
            .expect("read records");
        assert_eq!(records.len(), 2);
        let first = serde_json::to_value(&records[0]).expect("serialize");
        assert_eq!(first.get("kind"), Some(&json!("model_changed")));
        let second = serde_json::to_value(&records[1]).expect("serialize");
        assert_eq!(second.get("kind"), Some(&json!("permission_mode_changed")));
    }

    #[test]
    fn todo_runner_events_map_to_app_events() {
        let todo_event = RunnerEvent::TodoSnapshot(TodoSnapshotEvent::new(vec![TodoItem {
            id: "t1".into(),
            content: "inspect".into(),
            status: TodoStatus::Pending,
        }]));
        assert!(matches!(
            todo_event.app_event(),
            Some(AppEvent::TodoSnapshot(_))
        ));

        let auto_event =
            RunnerEvent::AutoContinueChanged(AutoContinueChangedEvent::new(AutoContinueState {
                enabled: true,
                max_continuations: 2,
            }));
        assert!(matches!(
            auto_event.app_event(),
            Some(AppEvent::AutoContinueChanged(_))
        ));
    }

    fn poisoned_transcript() -> Arc<Mutex<TranscriptRecorder>> {
        let base_dir = std::env::temp_dir().join(format!(
            "letcode-group8-runner-test-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("time ok")
                .as_nanos()
        ));
        let recorder = TranscriptRecorder::create(&base_dir).expect("create transcript recorder");
        let transcript = Arc::new(Mutex::new(recorder));
        let cloned = Arc::clone(&transcript);

        let join = std::thread::spawn(move || {
            let _guard = cloned.lock().expect("lock transcript");
            panic!("poison transcript mutex for test");
        });
        let _ = join.join();

        transcript
    }
}
