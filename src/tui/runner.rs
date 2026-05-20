use std::sync::{Arc, Mutex};

use anyhow::{Result, anyhow};
use async_openai::config::Config;
use serde_json::Value;
use tokio::sync::{mpsc, oneshot};

use crate::agent::{Agent, AgentEvent};
use crate::permission::PermissionRequest;
use crate::tool::ToolResult;
use crate::tool_format::format_tool_call;
use crate::transcript::TranscriptRecorder;

use super::events::{
    AppEvent, AssistantDeltaEvent, ErrorEvent, PermissionRequestEvent, PermissionResolutionEvent,
    ToolFinishedEvent, ToolOutcome, ToolStartedEvent, UserMessageEvent,
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
    AssistantDelta(AssistantDeltaEvent),
    AssistantDone {
        message_id: Option<String>,
    },
    ToolStarted(ToolStartedEvent),
    ToolFinished(ToolFinishedEvent),
    PermissionRequested {
        event: PermissionRequestEvent,
        handle: RunnerPermissionRequest,
    },
    PermissionResolved(PermissionResolutionEvent),
    Error(ErrorEvent),
    Done,
}

impl RunnerEvent {
    pub fn app_event(&self) -> Option<AppEvent> {
        match self {
            Self::UserMessage(event) => Some(AppEvent::UserMessage(event.clone())),
            Self::AssistantDelta(event) => Some(AppEvent::AssistantDelta(event.clone())),
            Self::AssistantDone { message_id } => Some(AppEvent::AssistantDone {
                message_id: message_id.clone(),
            }),
            Self::ToolStarted(event) => Some(AppEvent::ToolStarted(event.clone())),
            Self::ToolFinished(event) => Some(AppEvent::ToolFinished(event.clone())),
            Self::PermissionRequested { event, .. } => {
                Some(AppEvent::PermissionRequested(event.clone()))
            }
            Self::PermissionResolved(event) => Some(AppEvent::PermissionResolved(event.clone())),
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
                                            output,
                                        )
                                    })?;
                                    sender
                                        .send(RunnerEvent::ToolFinished(finished))
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
    output
        .data
        .as_ref()
        .map(Value::to_string)
        .or_else(|| output.error.as_ref().map(|error| error.message.clone()))
}

fn output_json(output: &ToolResult) -> Value {
    serde_json::to_value(output)
        .unwrap_or_else(|_| Value::String("<unserializable tool output>".into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcript::TranscriptRecorder;
    use crate::tui::events::PermissionDecision;
    use async_openai::{Client, config::OpenAIConfig};
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
            tool: "run_command".into(),
            args: Value::Null,
            class: crate::permission::ToolPermissionClass::Command,
            summary: "run_command cargo test".into(),
            preview: None,
        };

        let resolution = permission_resolution_event(&request, PermissionResponse::Deny);

        assert_eq!(resolution.call_id, "call-7");
        assert_eq!(resolution.decision, PermissionDecision::Denied);
        assert!(resolution.reason.is_some());
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
