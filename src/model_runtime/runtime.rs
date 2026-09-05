use super::{
    ContentPart, FailureKind, FailurePhase, HttpMethod, ModelEvent, ModelFailure, ModelMessage,
    ModelRequestInput, PreparedHttpRequest, ResolvedModelRoute, RetryHint, RuntimeRetryConfig,
    TerminalStatus, TransportResponse,
};
use async_trait::async_trait;
use futures_util::Stream;
use std::collections::{BTreeMap, HashMap};
use std::pin::Pin;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::Duration;

const MAX_PROVIDER_ERROR_BODY_BYTES: usize = 512 * 1024;

/// Provider-neutral boundary used by the runtime. Production uses the resolved
/// provider transport; tests can inject deterministic responses without a
/// network server.
#[async_trait]
pub trait ModelTransport: Send + Sync {
    async fn send_prepared(
        &self,
        route: &ResolvedModelRoute,
        request: PreparedHttpRequest,
    ) -> Result<super::TransportResponse, ModelFailure>;
}

#[derive(Clone, Default)]
pub struct ResolvedProviderTransport;

pub(crate) struct TurnLocalResponsesTransport {
    session: Arc<tokio::sync::Mutex<Option<crate::model_runtime::websocket::TurnLocalWsSession>>>,
    previous_response_id: Arc<tokio::sync::Mutex<Option<String>>>,
    force_full: Arc<tokio::sync::Mutex<bool>>,
    next_prompt_unit_start: Arc<tokio::sync::Mutex<Option<usize>>>,
    poisoned_response: Arc<AtomicBool>,
    force_http: Arc<AtomicBool>,
}

impl TurnLocalResponsesTransport {
    pub(crate) fn new() -> Self {
        Self {
            session: Arc::new(tokio::sync::Mutex::new(None)),
            previous_response_id: Arc::new(tokio::sync::Mutex::new(None)),
            force_full: Arc::new(tokio::sync::Mutex::new(false)),
            next_prompt_unit_start: Arc::new(tokio::sync::Mutex::new(None)),
            poisoned_response: Arc::new(AtomicBool::new(false)),
            force_http: Arc::new(AtomicBool::new(false)),
        }
    }

    pub(crate) async fn set_next_prompt_unit_start(&self, prompt_unit_start: usize) {
        *self.next_prompt_unit_start.lock().await = Some(prompt_unit_start);
    }

    pub(crate) async fn reset_chain(&self) {
        *self.previous_response_id.lock().await = None;
        *self.next_prompt_unit_start.lock().await = None;
        *self.force_full.lock().await = true;
    }
}

#[async_trait]
impl ModelTransport for TurnLocalResponsesTransport {
    async fn send_prepared(
        &self,
        route: &ResolvedModelRoute,
        request: PreparedHttpRequest,
    ) -> Result<TransportResponse, ModelFailure> {
        if !route.websocket {
            return Err(
                ModelFailure::new(FailurePhase::Transport, FailureKind::InvalidRequest)
                    .with_code("websocket_disabled"),
            );
        }
        if self.force_http.load(Ordering::Acquire) {
            return ResolvedProviderTransport
                .send_prepared(route, request)
                .await;
        }
        if self.poisoned_response.swap(false, Ordering::AcqRel) {
            *self.session.lock().await = None;
        }
        let mut session_guard = self.session.lock().await;
        if session_guard.is_none() {
            // Incremental continuation relies on state cached by this connection.
            self.reset_chain().await;
            let builder = route.transport.request(
                reqwest::Method::POST,
                &request.url,
                &route.provider,
                &route.auth,
                &route.headers,
                &route.query,
            )?;
            let builder = request
                .protocol_headers
                .iter()
                .filter(|(name, _)| {
                    !matches!(
                        name.to_ascii_lowercase().as_str(),
                        "accept" | "content-type"
                    )
                })
                .fold(builder, |builder, (name, value)| {
                    builder.header(name, value)
                });
            *session_guard = Some(
                route
                    .transport
                    .open_websocket(&route.protocol_id, builder, &route.auth)
                    .await?,
            );
        }
        let Some(session) = session_guard.as_mut() else {
            return Err(
                ModelFailure::new(FailurePhase::Transport, FailureKind::Internal)
                    .with_code("websocket_session_missing"),
            );
        };
        let previous_response_id = self.previous_response_id.lock().await.clone();
        let can_continue = !*self.force_full.lock().await && previous_response_id.is_some();
        let incremental_prompt_unit_start = if can_continue {
            *self.next_prompt_unit_start.lock().await
        } else {
            None
        };
        let frame_previous_response_id = can_continue
            .then_some(previous_response_id.as_deref())
            .flatten();
        let frame = route.binding.websocket_frame(
            &request,
            frame_previous_response_id,
            incremental_prompt_unit_start,
        )?;
        if let Err(error) = session
            .send_text(
                frame,
                &[route.auth.credential.as_deref().unwrap_or_default()],
            )
            .await
        {
            *session_guard = None;
            return Err(error);
        }
        drop(session_guard);
        let stream = websocket_response_stream(
            self.session.clone(),
            self.previous_response_id.clone(),
            self.force_full.clone(),
            self.next_prompt_unit_start.clone(),
            self.poisoned_response.clone(),
            self.force_http.clone(),
            route.auth.credential.clone().unwrap_or_default(),
        );
        Ok(TransportResponse::from_responses_websocket_stream(
            200,
            BTreeMap::new(),
            stream,
        ))
    }
}

struct WebsocketResponseStream {
    inner: Pin<Box<dyn Stream<Item = Result<Vec<u8>, ModelFailure>> + Send>>,
    terminal: Arc<AtomicBool>,
    poisoned: Arc<AtomicBool>,
}

impl Stream for WebsocketResponseStream {
    type Item = Result<Vec<u8>, ModelFailure>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        self.inner.as_mut().poll_next(cx)
    }
}

impl Drop for WebsocketResponseStream {
    fn drop(&mut self) {
        if !self.terminal.load(Ordering::Acquire) {
            self.poisoned.store(true, Ordering::Release);
        }
    }
}

fn websocket_response_stream(
    session: Arc<tokio::sync::Mutex<Option<crate::model_runtime::websocket::TurnLocalWsSession>>>,
    previous_response_id: Arc<tokio::sync::Mutex<Option<String>>>,
    force_full: Arc<tokio::sync::Mutex<bool>>,
    next_prompt_unit_start: Arc<tokio::sync::Mutex<Option<usize>>>,
    poisoned: Arc<AtomicBool>,
    force_http: Arc<AtomicBool>,
    secret: String,
) -> WebsocketResponseStream {
    let terminal = Arc::new(AtomicBool::new(false));
    let terminal_for_stream = terminal.clone();
    let inner = futures_util::stream::unfold(
        (session, previous_response_id, secret, false),
        move |(session, previous_response_id, secret, ended)| {
            let force_full = force_full.clone();
            let next_prompt_unit_start = next_prompt_unit_start.clone();
            let terminal_for_stream = terminal_for_stream.clone();
            let force_http = force_http.clone();
            async move {
                if ended {
                    return None;
                }
                let mut guard = session.lock().await;
                let socket = guard.as_mut()?;
                let result = socket.next_text(&[secret.as_str()]).await;
                drop(guard);
                match result {
                    Ok(text) => {
                        let value = serde_json::from_slice::<serde_json::Value>(&text).ok();
                        let event_type = value
                            .as_ref()
                            .and_then(|value| value.get("type"))
                            .and_then(serde_json::Value::as_str)
                            .unwrap_or_default();
                        let is_successful_terminal =
                            matches!(event_type, "response.completed" | "response.incomplete");
                        let terminal_response_id = is_successful_terminal
                            .then(|| {
                                value
                                    .as_ref()
                                    .and_then(|value| value.get("response"))
                                    .and_then(|response| response.get("id"))
                                    .and_then(serde_json::Value::as_str)
                                    .map(str::to_owned)
                            })
                            .flatten();
                        if is_successful_terminal {
                            *previous_response_id.lock().await = terminal_response_id.clone();
                        }
                        if event_type == "error" {
                            let code = value
                                .as_ref()
                                .and_then(|value| value.get("error"))
                                .and_then(|error| error.get("code"))
                                .and_then(serde_json::Value::as_str)
                                .or_else(|| {
                                    value
                                        .as_ref()
                                        .and_then(|value| value.get("code"))
                                        .and_then(serde_json::Value::as_str)
                                });
                            if code == Some("previous_response_not_found") {
                                *previous_response_id.lock().await = None;
                                *force_full.lock().await = true;
                            }
                        }
                        if is_successful_terminal {
                            *force_full.lock().await = terminal_response_id.is_none();
                            *next_prompt_unit_start.lock().await = None;
                        }
                        let is_terminal = is_successful_terminal
                            || matches!(event_type, "response.failed" | "error");
                        if is_terminal {
                            terminal_for_stream.store(true, Ordering::Release);
                        }
                        Some((
                            Ok(text),
                            (session, previous_response_id, secret, is_terminal),
                        ))
                    }
                    Err(error) => {
                        *session.lock().await = None;
                        if error.code.as_deref() == Some("websocket_message_too_big") {
                            force_http.store(true, Ordering::Release);
                        }
                        Some((Err(error), (session, previous_response_id, secret, true)))
                    }
                }
            }
        },
    );
    WebsocketResponseStream {
        inner: Box::pin(inner),
        terminal,
        poisoned,
    }
}

#[async_trait]
impl ModelTransport for ResolvedProviderTransport {
    async fn send_prepared(
        &self,
        route: &ResolvedModelRoute,
        request: PreparedHttpRequest,
    ) -> Result<super::TransportResponse, ModelFailure> {
        let method = match request.method {
            HttpMethod::Get => reqwest::Method::GET,
            HttpMethod::Post => reqwest::Method::POST,
        };
        let builder = route.transport.request(
            method,
            &request.url,
            &route.provider,
            &route.auth,
            &route.headers,
            &route.query,
        )?;
        let builder = request
            .protocol_headers
            .iter()
            .fold(builder, |builder, (name, value)| {
                builder.header(name, value)
            });
        route
            .transport
            .send(builder.body(request.body), &route.auth)
            .await
    }
}

#[async_trait]
pub trait ModelEventObserver: Send {
    async fn observe(&mut self, event: &ModelEvent) -> Result<(), ModelFailure>;
}

struct TextOneshotObserver<'a, F> {
    text: String,
    on_delta: &'a mut F,
    rejected_event: bool,
}

#[async_trait]
impl<F, Fut> ModelEventObserver for TextOneshotObserver<'_, F>
where
    F: FnMut(&str) -> Fut + Send,
    Fut: std::future::Future<Output = Result<(), ModelFailure>> + Send,
{
    async fn observe(&mut self, event: &ModelEvent) -> Result<(), ModelFailure> {
        match event {
            ModelEvent::TextDelta { text } => {
                if let Err(error) = (self.on_delta)(text).await {
                    self.rejected_event = true;
                    return Err(error);
                }
                self.text.push_str(text);
            }
            ModelEvent::ToolStarted { .. }
            | ModelEvent::ToolArgumentsDelta { .. }
            | ModelEvent::ToolDone { .. } => {
                self.rejected_event = true;
                return Err(runtime_invalid("oneshot emitted a tool call"));
            }
            _ => {}
        }
        Ok(())
    }
}

struct NoopObserver;

#[async_trait]
impl ModelEventObserver for NoopObserver {
    async fn observe(&mut self, _event: &ModelEvent) -> Result<(), ModelFailure> {
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct AttemptSideEffects {
    pub text: bool,
    pub reasoning: bool,
    pub pending_tool: bool,
    pub completed_tool: bool,
}

impl AttemptSideEffects {
    pub fn observable(&self) -> bool {
        self.text || self.reasoning || self.pending_tool || self.completed_tool
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompletedToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelAttemptSnapshot {
    pub events: Vec<ModelEvent>,
    pub assistant: ModelMessage,
    pub completed_tools: Vec<CompletedToolCall>,
    pub pending_tools: Vec<PendingToolCall>,
    pub side_effects: AttemptSideEffects,
    pub response_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelAttemptResult {
    pub snapshot: ModelAttemptSnapshot,
    pub terminal: TerminalStatus,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelAttemptFailure {
    pub failure: ModelFailure,
    pub partial: ModelAttemptSnapshot,
}

#[derive(Clone)]
pub struct ModelRuntime {
    transport: Arc<dyn ModelTransport>,
    responses_websocket: bool,
}

impl Default for ModelRuntime {
    fn default() -> Self {
        Self::new(Arc::new(ResolvedProviderTransport))
    }
}

impl ModelRuntime {
    pub fn new(transport: Arc<dyn ModelTransport>) -> Self {
        Self {
            transport,
            responses_websocket: false,
        }
    }

    pub(crate) fn new_responses_websocket(transport: Arc<dyn ModelTransport>) -> Self {
        Self {
            transport,
            responses_websocket: true,
        }
    }

    /// Text deltas are provisional until completion. Before retrying, `on_retry`
    /// must discard any preview from the failed attempt.
    pub async fn execute_text_oneshot<F, Fut, R, Rfut>(
        &self,
        route: &ResolvedModelRoute,
        input: &ModelRequestInput,
        mut on_delta: F,
        mut on_retry: R,
    ) -> Result<String, ModelFailure>
    where
        F: FnMut(&str) -> Fut + Send,
        Fut: std::future::Future<Output = Result<(), ModelFailure>> + Send,
        R: FnMut() -> Rfut + Send,
        Rfut: std::future::Future<Output = Result<(), ModelFailure>> + Send,
    {
        let retry = route.retry.clone().unwrap_or_else(default_retry_config);
        let request = route.binding.prepare_request(input)?;
        let mut attempt = 1;
        loop {
            let mut observer = TextOneshotObserver {
                text: String::new(),
                on_delta: &mut on_delta,
                rejected_event: false,
            };
            match self
                .execute_prepared_attempt(route, request.clone(), &mut observer)
                .await
            {
                Ok(result) => {
                    if result.terminal != TerminalStatus::Completed
                        || !result.snapshot.completed_tools.is_empty()
                    {
                        return Err(runtime_invalid("oneshot requires text completion"));
                    }
                    return Ok(observer.text);
                }
                Err(error)
                    if !observer.rejected_event
                        && retryable_failure(&error.failure)
                        && retry.enabled
                        && attempt < retry.max_attempts =>
                {
                    let delay = retry_delay(&retry, attempt, error.failure.retry_hint);
                    tracing::warn!(
                        provider = %route.provider,
                        model = %route.model,
                        next_attempt = attempt + 1,
                        max_attempts = retry.max_attempts,
                        delay_secs = delay.as_secs(),
                        error = %error.failure,
                        detail = %error.failure.detail(),
                        "retrying text oneshot request"
                    );
                    on_retry().await?;
                    tokio::time::sleep(delay).await;
                    attempt += 1;
                }
                Err(error) => return Err(error.failure),
            }
        }
    }

    pub async fn execute_attempt(
        &self,
        route: &ResolvedModelRoute,
        input: &ModelRequestInput,
    ) -> Result<ModelAttemptResult, ModelAttemptFailure> {
        self.execute_attempt_with(route, input, &mut NoopObserver)
            .await
    }

    pub async fn execute_attempt_with(
        &self,
        route: &ResolvedModelRoute,
        input: &ModelRequestInput,
        observer: &mut dyn ModelEventObserver,
    ) -> Result<ModelAttemptResult, ModelAttemptFailure> {
        let request =
            route
                .binding
                .prepare_request(input)
                .map_err(|failure| ModelAttemptFailure {
                    failure,
                    partial: ModelAttemptSnapshot::default(),
                })?;
        self.execute_prepared_attempt(route, request, observer)
            .await
    }

    async fn execute_prepared_attempt(
        &self,
        route: &ResolvedModelRoute,
        request: PreparedHttpRequest,
        observer: &mut dyn ModelEventObserver,
    ) -> Result<ModelAttemptResult, ModelAttemptFailure> {
        let owned_secrets = route_error_secrets(route);
        let secret_refs = owned_secrets.iter().map(String::as_str).collect::<Vec<_>>();
        let mut response =
            self.transport
                .send_prepared(route, request)
                .await
                .map_err(|failure| ModelAttemptFailure {
                    failure,
                    partial: ModelAttemptSnapshot::default(),
                })?;
        if !(200..300).contains(&response.status) {
            let status = response.status;
            let retry_hint = response.retry_hint();
            let detail = read_provider_error_detail(&mut response, &secret_refs).await;
            let mut failure = http_status_failure(status, retry_hint);
            if !detail.is_empty() {
                failure = failure.with_detail_redacted(detail, &secret_refs);
            }
            return Err(ModelAttemptFailure {
                failure,
                partial: ModelAttemptSnapshot::default(),
            });
        }
        let mut decoder = if self.responses_websocket
            && response.uses_responses_websocket_events()
            && route.protocol_id.as_str() == "responses"
        {
            route.binding.new_websocket_decoder()
        } else {
            route.binding.new_decoder()
        };
        let mut accumulator = AttemptAccumulator::default();
        while let Some(chunk) = response.next_chunk().await {
            let chunk = chunk.map_err(|failure| accumulator.failed(failure))?;
            let events = decoder
                .push(&chunk)
                .map_err(|failure| accumulator.failed(failure))?;
            observe_events(observer, &mut accumulator, events).await?;
        }
        let events = decoder
            .finish()
            .map_err(|failure| accumulator.failed(failure))?;
        observe_events(observer, &mut accumulator, events).await?;
        accumulator.finish()
    }
}

async fn read_provider_error_detail(
    response: &mut super::TransportResponse,
    secrets: &[&str],
) -> String {
    let mut body = Vec::new();
    let mut truncated = false;
    let mut read_error = None;
    while let Some(chunk) = response.next_chunk().await {
        match chunk {
            Ok(chunk) => {
                let remaining = MAX_PROVIDER_ERROR_BODY_BYTES.saturating_sub(body.len());
                if chunk.len() > remaining {
                    body.extend_from_slice(&chunk[..remaining]);
                    truncated = true;
                    break;
                }
                body.extend_from_slice(&chunk);
            }
            Err(failure) => {
                read_error = Some(failure);
                break;
            }
        }
    }

    let mut detail = String::from_utf8_lossy(&body).into_owned();
    if truncated {
        detail.push_str(&format!(
            "\n[provider response body truncated after {MAX_PROVIDER_ERROR_BODY_BYTES} bytes]"
        ));
    }
    if let Some(failure) = read_error {
        if !detail.is_empty() {
            detail.push('\n');
        }
        detail.push_str("[failed to finish reading provider response body: ");
        detail.push_str(&failure.to_string());
        if !failure.detail().is_empty() {
            detail.push_str(": ");
            detail.push_str(failure.detail());
        }
        detail.push(']');
    }
    ModelFailure::new(FailurePhase::Transport, FailureKind::Http)
        .with_detail_redacted(detail, secrets)
        .detail()
        .to_owned()
}

fn route_error_secrets(route: &ResolvedModelRoute) -> Vec<String> {
    route
        .auth
        .credential
        .as_deref()
        .filter(|value| !value.is_empty())
        .map(|credential| vec![credential.to_owned()])
        .unwrap_or_default()
}

async fn observe_events(
    observer: &mut dyn ModelEventObserver,
    accumulator: &mut AttemptAccumulator,
    events: Vec<ModelEvent>,
) -> Result<(), ModelAttemptFailure> {
    for event in events {
        observer
            .observe(&event)
            .await
            .map_err(|failure| accumulator.failed(failure))?;
        accumulator
            .consume_observed(event)
            .map_err(|failure| accumulator.failed(failure))?;
        if let Some(failure) = accumulator.failure.clone() {
            return Err(accumulator.failed(failure));
        }
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum AssistantPartKey {
    Reasoning(String),
    Text,
    Tool(String),
}

#[derive(Default)]
struct AttemptAccumulator {
    events: Vec<ModelEvent>,
    order: Vec<AssistantPartKey>,
    text: String,
    reasoning: HashMap<String, (String, Option<super::OpaqueReplayState>)>,
    reasoning_text_components: HashMap<String, String>,
    reasoning_summary_components: BTreeMap<(String, u64), String>,
    tools: HashMap<String, PendingToolCall>,
    completed_tool_arguments: HashMap<String, serde_json::Value>,
    side_effects: AttemptSideEffects,
    terminal: Option<TerminalStatus>,
    failure: Option<ModelFailure>,
    response_id: Option<String>,
}

impl AttemptAccumulator {
    fn consume_observed(&mut self, event: ModelEvent) -> Result<(), ModelFailure> {
        if self.terminal.is_some() || self.failure.is_some() {
            return Err(runtime_invalid("event received after terminal"));
        }
        match &event {
            ModelEvent::TextDelta { text } => {
                self.push_order(AssistantPartKey::Text);
                self.text.push_str(text);
                self.side_effects.text = true;
            }
            ModelEvent::ReasoningStarted { item_id } => {
                self.push_order(AssistantPartKey::Reasoning(item_id.clone()));
                self.reasoning.entry(item_id.clone()).or_default();
                self.side_effects.reasoning = true;
            }
            ModelEvent::ReasoningDelta { item_id, text } => {
                self.push_order(AssistantPartKey::Reasoning(item_id.clone()));
                self.reasoning_text_components
                    .entry(item_id.clone())
                    .or_default()
                    .push_str(text);
                self.update_reasoning_content(item_id);
                self.side_effects.reasoning = true;
            }
            ModelEvent::ReasoningSummaryDelta {
                item_id,
                summary_index,
                text,
            } => {
                self.push_order(AssistantPartKey::Reasoning(item_id.clone()));
                self.reasoning_summary_components
                    .entry((item_id.clone(), *summary_index))
                    .or_default()
                    .push_str(text);
                self.update_reasoning_content(item_id);
                self.side_effects.reasoning = true;
            }
            ModelEvent::ReasoningSummaryDone {
                item_id,
                summary_index,
                text,
            } => {
                self.push_order(AssistantPartKey::Reasoning(item_id.clone()));
                self.reasoning_summary_components
                    .insert((item_id.clone(), *summary_index), text.clone());
                self.update_reasoning_content(item_id);
                self.side_effects.reasoning = true;
            }
            ModelEvent::ReasoningDone {
                item_id,
                text,
                replay,
            } => {
                self.push_order(AssistantPartKey::Reasoning(item_id.clone()));
                self.reasoning
                    .insert(item_id.clone(), (text.clone(), replay.clone()));
                self.side_effects.reasoning = true;
            }
            ModelEvent::ToolStarted { id, name } => {
                self.push_order(AssistantPartKey::Tool(id.clone()));
                self.tools.entry(id.clone()).or_insert(PendingToolCall {
                    id: id.clone(),
                    name: name.clone(),
                    arguments: String::new(),
                });
                self.side_effects.pending_tool = true;
            }
            ModelEvent::ToolArgumentsDelta { id, delta } => {
                let tool = self
                    .tools
                    .get_mut(id)
                    .ok_or_else(|| runtime_invalid("tool arguments before tool start"))?;
                tool.arguments.push_str(delta);
                self.side_effects.pending_tool = true;
            }
            ModelEvent::ToolDone {
                id,
                name,
                arguments,
            } => {
                let tool = self
                    .tools
                    .get(id)
                    .ok_or_else(|| runtime_invalid("tool completion before tool start"))?;
                if tool.name != *name {
                    return Err(runtime_invalid("tool name changed"));
                }
                self.completed_tool_arguments
                    .insert(id.clone(), arguments.clone());
                self.side_effects.pending_tool = true;
                self.side_effects.completed_tool = true;
            }
            ModelEvent::ResponseMetadata { response_id } => {
                if let Some(previous) = &self.response_id
                    && previous != response_id
                {
                    return Err(runtime_invalid("response id changed"));
                }
                self.response_id = Some(response_id.clone());
            }
            ModelEvent::Terminal { status } => self.terminal = Some(*status),
            ModelEvent::Failure(failure) => self.failure = Some(failure.clone()),
            ModelEvent::Usage { .. } | ModelEvent::Cache { .. } => {}
        }
        self.events.push(event);
        Ok(())
    }

    fn update_reasoning_content(&mut self, item_id: &str) {
        let summary = self
            .reasoning_summary_components
            .iter()
            .filter(|((component_item_id, _), _)| component_item_id == item_id)
            .map(|(_, text)| text.as_str())
            .filter(|text| !text.is_empty())
            .collect::<Vec<_>>()
            .join("\n\n");
        let reasoning_text = self
            .reasoning_text_components
            .get(item_id)
            .map(String::as_str)
            .filter(|text| !text.is_empty());
        let text = match (summary.is_empty(), reasoning_text) {
            (false, Some(reasoning_text)) => format!("{summary}\n\n{reasoning_text}"),
            (false, None) => summary,
            (true, Some(reasoning_text)) => reasoning_text.to_string(),
            (true, None) => String::new(),
        };
        self.reasoning.entry(item_id.to_string()).or_default().0 = text;
    }

    fn push_order(&mut self, key: AssistantPartKey) {
        if !self.order.contains(&key) {
            self.order.push(key);
        }
    }

    fn snapshot(&self) -> ModelAttemptSnapshot {
        let mut content = Vec::new();
        let mut completed_tools = Vec::new();
        let mut pending_tools = Vec::new();
        for key in &self.order {
            match key {
                AssistantPartKey::Reasoning(item_id) => {
                    if let Some((text, replay)) = self.reasoning.get(item_id) {
                        content.push(ContentPart::Reasoning {
                            item_id: item_id.clone(),
                            text: text.clone(),
                            replay: replay.clone(),
                        });
                    }
                }
                AssistantPartKey::Text => {
                    if !self.text.is_empty() {
                        content.push(ContentPart::Text(self.text.clone()));
                    }
                }
                AssistantPartKey::Tool(id) => {
                    if let Some(tool) = self.tools.get(id) {
                        if let Some(arguments) = self.completed_tool_arguments.get(id) {
                            content.push(ContentPart::ToolCall {
                                id: id.clone(),
                                name: tool.name.clone(),
                                arguments: arguments.clone(),
                            });
                            completed_tools.push(CompletedToolCall {
                                id: id.clone(),
                                name: tool.name.clone(),
                                arguments: arguments.clone(),
                            });
                        } else {
                            pending_tools.push(tool.clone());
                        }
                    }
                }
            }
        }
        ModelAttemptSnapshot {
            events: self.events.clone(),
            assistant: ModelMessage {
                role: super::MessageRole::Assistant,
                content,
            },
            completed_tools,
            pending_tools,
            side_effects: self.side_effects.clone(),
            response_id: self.response_id.clone(),
        }
    }

    fn failed(&self, failure: ModelFailure) -> ModelAttemptFailure {
        ModelAttemptFailure {
            failure,
            partial: self.snapshot(),
        }
    }

    fn finish(self) -> Result<ModelAttemptResult, ModelAttemptFailure> {
        if let Some(failure) = self.failure.clone() {
            return Err(self.failed(failure));
        }
        let terminal = self
            .terminal
            .ok_or_else(|| self.failed(runtime_invalid("missing terminal event")))?;
        let snapshot = self.snapshot();
        if !snapshot.pending_tools.is_empty() {
            return Err(ModelAttemptFailure {
                failure: runtime_invalid("terminal contains incomplete tools"),
                partial: snapshot,
            });
        }
        let terminal = match terminal {
            TerminalStatus::Completed if !snapshot.completed_tools.is_empty() => {
                TerminalStatus::ToolUse
            }
            TerminalStatus::ToolUse if snapshot.completed_tools.is_empty() => {
                return Err(ModelAttemptFailure {
                    failure: runtime_invalid("tool terminal contains no tools"),
                    partial: snapshot,
                });
            }
            TerminalStatus::Length
            | TerminalStatus::ContentFilter
            | TerminalStatus::Refusal
            | TerminalStatus::Pause
            | TerminalStatus::Incomplete => {
                return Err(ModelAttemptFailure {
                    failure: ModelFailure::new(FailurePhase::Finish, FailureKind::InvalidRequest)
                        .with_code("non_success_terminal")
                        .with_detail(format!("terminal status: {terminal:?}")),
                    partial: snapshot,
                });
            }
            TerminalStatus::Completed | TerminalStatus::ToolUse => terminal,
        };
        Ok(ModelAttemptResult { snapshot, terminal })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TurnLimits {
    pub max_iterations: usize,
    pub max_tool_calls: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TurnResult {
    pub final_attempt: ModelAttemptResult,
    pub iterations: usize,
    pub tool_calls: usize,
    pub recovery_attempts: usize,
}

#[async_trait]
pub trait TurnDriver: Send {
    async fn prepare_iteration(
        &mut self,
        iteration: usize,
    ) -> Result<ModelRequestInput, ModelFailure>;

    fn bypass_iteration_limit(&self) -> bool {
        false
    }

    fn bypass_tool_limit(&self) -> bool {
        false
    }

    async fn decorate_request(
        &mut self,
        request: PreparedHttpRequest,
    ) -> Result<PreparedHttpRequest, ModelFailure> {
        Ok(request)
    }

    async fn attempt_started(
        &mut self,
        _iteration: usize,
        _attempt: usize,
    ) -> Result<(), ModelFailure> {
        Ok(())
    }

    async fn commit_before_first_send(&mut self, _iteration: usize) -> Result<(), ModelFailure> {
        Ok(())
    }

    async fn attempt_finished(
        &mut self,
        _iteration: usize,
        _attempt: usize,
        _outcome: &AttemptOutcome,
    ) -> Result<(), ModelFailure> {
        Ok(())
    }

    async fn retry_scheduled(
        &mut self,
        _iteration: usize,
        _attempt: usize,
        _next_attempt: usize,
        _delay: Duration,
        _failure: &ModelFailure,
    ) -> Result<(), ModelFailure> {
        Ok(())
    }

    async fn retry_started(
        &mut self,
        _iteration: usize,
        _attempt: usize,
    ) -> Result<(), ModelFailure> {
        Ok(())
    }

    async fn observe_event(&mut self, event: &ModelEvent) -> Result<(), ModelFailure>;

    async fn persist_assistant(&mut self, assistant: &ModelMessage) -> Result<(), ModelFailure>;

    async fn execute_tools(&mut self, tools: &[CompletedToolCall]) -> Result<(), ModelFailure>;

    async fn recover_iteration(
        &mut self,
        partial: &ModelAttemptSnapshot,
        failure: &ModelFailure,
    ) -> Result<(), ModelFailure>;

    async fn after_assistant_persisted(
        &mut self,
        _result: &ModelAttemptResult,
    ) -> Result<TurnContinuationDecision, ModelFailure> {
        Ok(TurnContinuationDecision::Finalize)
    }

    async fn finalize(&mut self, result: &ModelAttemptResult) -> Result<(), ModelFailure>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnContinuationDecision {
    Continue,
    Finalize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttemptOutcome {
    Completed {
        terminal: TerminalStatus,
        side_effects: AttemptSideEffects,
    },
    Failed {
        failure: ModelFailure,
        side_effects: AttemptSideEffects,
    },
}

pub struct TurnOrchestrator {
    runtime: ModelRuntime,
    limits: TurnLimits,
}

impl TurnOrchestrator {
    pub fn new(runtime: ModelRuntime, limits: TurnLimits) -> Self {
        Self { runtime, limits }
    }

    pub async fn run(
        &self,
        route: &ResolvedModelRoute,
        driver: &mut dyn TurnDriver,
    ) -> Result<TurnResult, ModelFailure> {
        let retry = route.retry.clone().unwrap_or_else(default_retry_config);
        let mut iteration = 0;
        let mut total_tools: usize = 0;
        let mut recovery_attempts = 0;
        loop {
            if !driver.bypass_iteration_limit() && iteration >= self.limits.max_iterations {
                return Err(
                    ModelFailure::new(FailurePhase::Finish, FailureKind::InvalidRequest)
                        .with_code("max_iterations"),
                );
            }
            let input = driver.prepare_iteration(iteration).await?;
            let request = route.binding.prepare_request(&input)?;
            let request = driver.decorate_request(request).await?;
            let result = self
                .execute_with_physical_retries(route, &request, driver, &retry, iteration)
                .await;
            let result = match result {
                Ok(result) => result,
                Err(error)
                    if error.partial.side_effects.observable()
                        && retryable_failure(&error.failure)
                        && retry.enabled
                        && recovery_attempts < retry.max_recovery_attempts =>
                {
                    driver
                        .recover_iteration(&error.partial, &error.failure)
                        .await?;
                    recovery_attempts += 1;
                    iteration += 1;
                    continue;
                }
                Err(error) => return Err(error.failure),
            };
            if result.terminal == TerminalStatus::ToolUse {
                let next_tool_total =
                    total_tools.saturating_add(result.snapshot.completed_tools.len());
                if !driver.bypass_tool_limit()
                    && self
                        .limits
                        .max_tool_calls
                        .is_some_and(|limit| next_tool_total > limit)
                {
                    return Err(ModelFailure::new(
                        FailurePhase::Finish,
                        FailureKind::InvalidRequest,
                    )
                    .with_code("max_tool_calls"));
                }
                total_tools = next_tool_total;
            }
            driver.persist_assistant(&result.snapshot.assistant).await?;
            if result.terminal == TerminalStatus::ToolUse {
                driver
                    .execute_tools(&result.snapshot.completed_tools)
                    .await?;
                iteration += 1;
                continue;
            }
            if driver.after_assistant_persisted(&result).await?
                == TurnContinuationDecision::Continue
            {
                iteration += 1;
                continue;
            }
            driver.finalize(&result).await?;
            return Ok(TurnResult {
                final_attempt: result,
                iterations: iteration + 1,
                tool_calls: total_tools,
                recovery_attempts,
            });
        }
    }

    async fn execute_with_physical_retries(
        &self,
        route: &ResolvedModelRoute,
        request: &PreparedHttpRequest,
        driver: &mut dyn TurnDriver,
        retry: &RuntimeRetryConfig,
        iteration: usize,
    ) -> Result<ModelAttemptResult, ModelAttemptFailure> {
        let mut attempt = 1;
        loop {
            if let Err(failure) = driver.attempt_started(iteration, attempt).await {
                return Err(ModelAttemptFailure {
                    failure,
                    partial: ModelAttemptSnapshot::default(),
                });
            }
            if attempt == 1
                && let Err(failure) = driver.commit_before_first_send(iteration).await
            {
                return Err(ModelAttemptFailure {
                    failure,
                    partial: ModelAttemptSnapshot::default(),
                });
            }
            let mut observer = DriverObserver { driver };
            let result = self
                .runtime
                .execute_prepared_attempt(route, request.clone(), &mut observer)
                .await;
            let outcome = match &result {
                Ok(result) => AttemptOutcome::Completed {
                    terminal: result.terminal,
                    side_effects: result.snapshot.side_effects.clone(),
                },
                Err(error) => AttemptOutcome::Failed {
                    failure: error.failure.clone(),
                    side_effects: error.partial.side_effects.clone(),
                },
            };
            if let Err(failure) = driver.attempt_finished(iteration, attempt, &outcome).await {
                return Err(ModelAttemptFailure {
                    failure,
                    partial: result
                        .as_ref()
                        .map(|result| result.snapshot.clone())
                        .unwrap_or_else(|error| error.partial.clone()),
                });
            }
            match result {
                Ok(result) => return Ok(result),
                Err(error)
                    if !error.partial.side_effects.observable()
                        && retryable_failure(&error.failure)
                        && retry.enabled
                        && attempt < retry.max_attempts =>
                {
                    let delay = retry_delay(retry, attempt, error.failure.retry_hint);
                    if let Err(failure) = driver
                        .retry_scheduled(iteration, attempt, attempt + 1, delay, &error.failure)
                        .await
                    {
                        return Err(ModelAttemptFailure {
                            failure,
                            partial: error.partial,
                        });
                    }
                    tokio::time::sleep(delay).await;
                    attempt += 1;
                    if let Err(failure) = driver.retry_started(iteration, attempt).await {
                        return Err(ModelAttemptFailure {
                            failure,
                            partial: error.partial,
                        });
                    }
                }
                Err(error) => return Err(error),
            }
        }
    }
}

struct DriverObserver<'a> {
    driver: &'a mut dyn TurnDriver,
}

#[async_trait]
impl ModelEventObserver for DriverObserver<'_> {
    async fn observe(&mut self, event: &ModelEvent) -> Result<(), ModelFailure> {
        self.driver.observe_event(event).await
    }
}

fn retryable_failure(failure: &ModelFailure) -> bool {
    failure.retry_hint != RetryHint::Never
}

fn retry_delay(config: &RuntimeRetryConfig, attempt: usize, hint: RetryHint) -> Duration {
    if let RetryHint::RetryAfterSeconds(seconds) = hint {
        return Duration::from_secs(seconds);
    }
    let exponent = if config.exponential_backoff {
        i32::try_from(attempt.saturating_sub(1)).unwrap_or(i32::MAX)
    } else {
        0
    };
    let delay = (config.initial_delay_secs as f64)
        * if config.exponential_backoff {
            (config.backoff_multiplier as f64).powi(exponent)
        } else {
            1.0
        };
    let base = if delay.is_finite() && delay < u64::MAX as f64 {
        delay.round() as u64
    } else {
        u64::MAX
    };
    let jitter = if !config.exponential_backoff || config.jitter_secs == 0 {
        0
    } else {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| {
                u64::from(duration.subsec_nanos()) % config.jitter_secs.saturating_add(1)
            })
            .unwrap_or(0)
    };
    Duration::from_secs(base.saturating_add(jitter))
}

fn default_retry_config() -> RuntimeRetryConfig {
    RuntimeRetryConfig {
        enabled: false,
        max_attempts: 1,
        max_recovery_attempts: 0,
        initial_delay_secs: 0,
        exponential_backoff: false,
        backoff_multiplier: 1.0,
        jitter_secs: 0,
    }
}

fn http_status_failure(status: u16, retry_hint: RetryHint) -> ModelFailure {
    let kind = match status {
        401 | 403 => FailureKind::Authentication,
        429 => FailureKind::RateLimited,
        _ => FailureKind::Http,
    };
    let retryable = reqwest::StatusCode::from_u16(status)
        .ok()
        .is_some_and(crate::retry::is_retryable_http_status);
    ModelFailure::new(FailurePhase::Transport, kind)
        .with_status(status)
        .with_code("http_status")
        .with_retry_hint(if retryable {
            retry_hint
        } else {
            RetryHint::Never
        })
}

fn runtime_invalid(detail: &str) -> ModelFailure {
    ModelFailure::new(FailurePhase::Finish, FailureKind::MalformedResponse)
        .with_code("invalid_runtime_event_sequence")
        .with_detail(detail)
}

impl Default for ModelAttemptSnapshot {
    fn default() -> Self {
        Self {
            events: Vec::new(),
            assistant: ModelMessage {
                role: super::MessageRole::Assistant,
                content: Vec::new(),
            },
            completed_tools: Vec::new(),
            pending_tools: Vec::new(),
            side_effects: AttemptSideEffects::default(),
            response_id: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_runtime::{ProtocolRegistry, RuntimeConfig, TransportResponse};
    use futures_util::{SinkExt, StreamExt};
    use std::collections::{BTreeMap, VecDeque};
    use std::sync::Mutex;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;
    use tokio_tungstenite::{accept_async, accept_hdr_async, tungstenite::Message};

    #[derive(Default)]
    struct QueueTransport {
        responses: Mutex<VecDeque<Result<TransportResponse, ModelFailure>>>,
    }

    #[async_trait]
    impl ModelTransport for QueueTransport {
        async fn send_prepared(
            &self,
            _route: &ResolvedModelRoute,
            _request: PreparedHttpRequest,
        ) -> Result<TransportResponse, ModelFailure> {
            self.responses
                .lock()
                .unwrap()
                .pop_front()
                .unwrap_or_else(|| Err(runtime_invalid("missing mock response")))
        }
    }

    fn route(protocol: &str) -> ResolvedModelRoute {
        let protocol_settings = if protocol == "anthropic" {
            "\n[providers.vendor.models.model.protocol_settings]\nanthropic_thinking = { mode = \"disabled\" }"
        } else {
            ""
        };
        let config = format!(
            r#"active_provider = "vendor"
[providers.vendor]
protocol = "{protocol}"
default_model = "model"
flavor = "standard"
[providers.vendor.auth]
type = "none"
[providers.vendor.endpoints]
base_url = "https://example.invalid/v1"
[providers.vendor.retry]
enabled = true
max_attempts = 3
max_recovery_attempts = 1
initial_delay_secs = 1
exponential_backoff = false
backoff_multiplier = 1.0
jitter_secs = 0
[providers.vendor.models.model]
[providers.vendor.models.model.capabilities]
tools = true
reasoning = true
[providers.vendor.models.model.capabilities.generation]
reasoning = true
{protocol_settings}
"#
        );
        RuntimeConfig::from_toml(&config)
            .unwrap()
            .resolve(&ProtocolRegistry::builtins())
            .unwrap()
            .route("vendor", "model")
            .unwrap()
            .clone()
    }

    fn response(status: u16, chunks: Vec<impl AsRef<[u8]>>) -> TransportResponse {
        TransportResponse::from_chunks(
            status,
            BTreeMap::new(),
            chunks
                .into_iter()
                .map(|chunk| chunk.as_ref().to_vec())
                .collect(),
        )
    }

    fn oneshot_response(text: &str, failure: Option<ModelFailure>) -> TransportResponse {
        let delta = format!(
            "data: {}\n\n",
            serde_json::json!({"type":"response.output_text.delta", "delta":text})
        );
        let end = match failure {
            Some(failure) => Err(failure),
            None => Ok(b"data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\"}}\n\n".to_vec()),
        };
        TransportResponse::from_results(200, BTreeMap::new(), vec![Ok(delta.into_bytes()), end])
    }

    fn oneshot_stream_failure() -> ModelFailure {
        ModelFailure::new(FailurePhase::Transport, FailureKind::Http)
            .with_status(200)
            .with_code("response_chunk_failed")
            .with_retry_hint(RetryHint::RetryAfterSeconds(0))
            .with_detail("stream interrupted")
    }

    #[tokio::test]
    async fn text_oneshot_resets_partial_preview_before_retry() {
        let route = route("responses");
        let transport = Arc::new(QueueTransport {
            responses: Mutex::new(VecDeque::from([
                Ok(oneshot_response(
                    "discarded",
                    Some(oneshot_stream_failure()),
                )),
                Ok(oneshot_response("summary", None)),
            ])),
        });
        let runtime = ModelRuntime::new(transport.clone());
        let events = Mutex::new(Vec::new());
        let output = runtime
            .execute_text_oneshot(
                &route,
                &ModelRequestInput::new("model", Vec::new()),
                |delta| {
                    events.lock().unwrap().push(delta.to_owned());
                    std::future::ready(Ok(()))
                },
                || {
                    events.lock().unwrap().push("reset".into());
                    std::future::ready(Ok(()))
                },
            )
            .await
            .unwrap();
        assert_eq!(output, "summary");
        assert_eq!(*events.lock().unwrap(), ["discarded", "reset", "summary"]);
        assert!(transport.responses.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn text_oneshot_respects_retry_limits_and_preserves_provider_failure() {
        for (enabled, max_attempts, hint, expected_attempts) in [
            (true, 2, RetryHint::RetryAfterSeconds(0), 2),
            (false, 3, RetryHint::RetryAfterSeconds(0), 1),
            (true, 1, RetryHint::RetryAfterSeconds(0), 1),
            (true, 3, RetryHint::Never, 1),
        ] {
            let mut route = route("responses");
            let retry = route.retry.as_mut().unwrap();
            retry.enabled = enabled;
            retry.max_attempts = max_attempts;
            let failure = oneshot_stream_failure().with_retry_hint(hint);
            let transport = Arc::new(QueueTransport {
                responses: Mutex::new(
                    (0..3)
                        .map(|_| Ok(oneshot_response("partial", Some(failure.clone()))))
                        .collect(),
                ),
            });
            let runtime = ModelRuntime::new(transport.clone());
            let mut resets = 0;
            let error = runtime
                .execute_text_oneshot(
                    &route,
                    &ModelRequestInput::new("model", Vec::new()),
                    |_| std::future::ready(Ok(())),
                    || {
                        resets += 1;
                        std::future::ready(Ok(()))
                    },
                )
                .await
                .unwrap_err();
            assert_eq!(error, failure);
            assert_eq!(resets, expected_attempts - 1);
            assert_eq!(
                transport.responses.lock().unwrap().len(),
                3 - expected_attempts
            );
        }
    }

    #[tokio::test]
    async fn text_oneshot_retries_before_output_but_rejects_tools() {
        for emits_tool in [false, true] {
            let route = route("responses");
            let first = if emits_tool {
                response(200, vec![b"data: {\"type\":\"response.output_item.added\",\"item\":{\"type\":\"function_call\",\"id\":\"item-1\",\"call_id\":\"call-1\",\"name\":\"search\"}}\n\n".as_slice()])
            } else {
                TransportResponse::from_results(
                    200,
                    BTreeMap::new(),
                    vec![Err(oneshot_stream_failure())],
                )
            };
            let transport = Arc::new(QueueTransport {
                responses: Mutex::new(VecDeque::from([
                    Ok(first),
                    Ok(oneshot_response("summary", None)),
                ])),
            });
            let runtime = ModelRuntime::new(transport.clone());
            let mut resets = 0;
            let output = runtime
                .execute_text_oneshot(
                    &route,
                    &ModelRequestInput::new("model", Vec::new()),
                    |_| std::future::ready(Ok(())),
                    || {
                        resets += 1;
                        std::future::ready(Ok(()))
                    },
                )
                .await;
            if emits_tool {
                assert_eq!(output.unwrap_err().retry_hint, RetryHint::Never);
                assert_eq!(resets, 0);
                assert_eq!(transport.responses.lock().unwrap().len(), 1);
            } else {
                assert_eq!(output.unwrap(), "summary");
                assert_eq!(resets, 1);
            }
        }
    }

    #[tokio::test]
    async fn text_oneshot_does_not_retry_callback_failures() {
        for fail_on_reset in [false, true] {
            let route = route("responses");
            let callback_error = oneshot_stream_failure().with_code("callback_failed");
            let transport = Arc::new(QueueTransport {
                responses: Mutex::new(VecDeque::from([
                    Ok(oneshot_response("partial", Some(oneshot_stream_failure()))),
                    Ok(oneshot_response("unused", None)),
                ])),
            });
            let runtime = ModelRuntime::new(transport.clone());
            let error = runtime
                .execute_text_oneshot(
                    &route,
                    &ModelRequestInput::new("model", Vec::new()),
                    |_| {
                        std::future::ready(if fail_on_reset {
                            Ok(())
                        } else {
                            Err(callback_error.clone())
                        })
                    },
                    || std::future::ready(Err(callback_error.clone())),
                )
                .await
                .unwrap_err();
            assert_eq!(error, callback_error);
            assert_eq!(transport.responses.lock().unwrap().len(), 1);
        }
    }

    #[tokio::test]
    #[allow(clippy::result_large_err)]
    async fn websocket_transport_replays_full_input_without_previous_id_after_not_found() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let mut socket = accept_hdr_async(
                stream,
                |request: &tokio_tungstenite::tungstenite::http::Request<()>, response| {
                    assert_eq!(request.headers()["x-turn-header"], "present");
                    assert!(request.headers().get("content-type").is_none());
                    Ok(response)
                },
            )
            .await
            .unwrap();
            for (round, expected_previous, expected_input_len) in [
                (1, None, 2usize),
                (2, Some("resp-1"), 1usize),
                (3, None, 2usize),
            ] {
                let Message::Text(text) = socket.next().await.unwrap().unwrap() else {
                    panic!("expected a text request frame");
                };
                let value: serde_json::Value = serde_json::from_str(&text).unwrap();
                assert_eq!(value["type"], "response.create");
                assert_eq!(value["instructions"], "system instructions");
                assert_eq!(value["tools"][0]["name"], "lookup");
                assert_eq!(value["input"].as_array().unwrap().len(), expected_input_len);
                assert_eq!(
                    value
                        .get("previous_response_id")
                        .and_then(serde_json::Value::as_str),
                    expected_previous
                );
                let response = if round == 2 {
                    r#"{"type":"error","code":"previous_response_not_found","message":"stale response"}"#
                        .to_owned()
                } else {
                    format!(
                        r#"{{"type":"response.completed","response":{{"id":"resp-{round}","status":"completed"}}}}"#
                    )
                };
                socket.send(Message::Text(response.into())).await.unwrap();
            }
        });

        let config = format!(
            r#"active_provider = "vendor"
[providers.vendor]
protocol = "responses"
default_model = "model"
flavor = "standard"
[providers.vendor.auth]
type = "none"
[providers.vendor.endpoints]
base_url = "http://{address}"
[providers.vendor.models.model]
[providers.vendor.models.model.transport]
websocket = true
[providers.vendor.models.model.capabilities]
reasoning = true
tools = true
"#
        );
        let route = RuntimeConfig::from_toml(&config)
            .unwrap()
            .resolve(&ProtocolRegistry::builtins())
            .unwrap()
            .route("vendor", "model")
            .unwrap()
            .clone();
        let mut input = ModelRequestInput::new(
            "model",
            vec![
                ModelMessage::text(super::super::MessageRole::User, "assistant history"),
                ModelMessage::text(super::super::MessageRole::User, "tool output"),
            ],
        );
        input.segments = vec![super::super::ControlSegment::system("system instructions")];
        input.segment_origins = vec!["system".into()];
        input.message_origins = vec!["history".into(), "tool".into()];
        input.tools = vec![super::super::ToolDefinition::new(
            "lookup",
            "look something up",
            serde_json::json!({"type":"object"}),
        )];
        let mut request = route.binding.prepare_request(&input).unwrap();
        request
            .protocol_headers
            .insert("x-turn-header".into(), "present".into());
        let transport = TurnLocalResponsesTransport::new();
        let first = transport
            .send_prepared(&route, request.clone())
            .await
            .unwrap();
        drain_response(first).await;
        transport.set_next_prompt_unit_start(2).await;
        let second = transport
            .send_prepared(&route, request.clone())
            .await
            .unwrap();
        drain_response(second).await;
        let third = transport.send_prepared(&route, request).await.unwrap();
        drain_response(third).await;
        assert!(!*transport.force_full.lock().await);
        assert_eq!(*transport.next_prompt_unit_start.lock().await, None);
        server.await.unwrap();
    }

    async fn drain_response(mut response: TransportResponse) {
        while let Some(chunk) = response.next_chunk().await {
            chunk.unwrap();
        }
    }

    #[tokio::test]
    async fn websocket_message_too_big_uses_http_on_next_attempt() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let mut socket = accept_async(stream).await.unwrap();
            let Message::Text(_) = socket.next().await.unwrap().unwrap() else {
                panic!("expected websocket request frame");
            };
            socket
                .send(Message::Close(Some(
                    tokio_tungstenite::tungstenite::protocol::CloseFrame {
                        code: tokio_tungstenite::tungstenite::protocol::frame::coding::CloseCode::Size,
                        reason: "message too big".into(),
                    },
                )))
                .await
                .unwrap();

            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = Vec::new();
            let mut chunk = [0u8; 4096];
            let (header_end, content_length) = loop {
                let read = stream.read(&mut chunk).await.unwrap();
                assert!(read > 0, "HTTP request closed before headers");
                request.extend_from_slice(&chunk[..read]);
                let Some(header_end) = request.windows(4).position(|part| part == b"\r\n\r\n")
                else {
                    continue;
                };
                let header_end = header_end + 4;
                let headers = std::str::from_utf8(&request[..header_end]).unwrap();
                let content_length = headers
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse::<usize>().unwrap())
                    })
                    .unwrap();
                break (header_end, content_length);
            };
            while request.len() < header_end + content_length {
                let read = stream.read(&mut chunk).await.unwrap();
                assert!(read > 0, "HTTP request closed before body");
                request.extend_from_slice(&chunk[..read]);
            }
            let request_line = std::str::from_utf8(&request)
                .unwrap()
                .lines()
                .next()
                .unwrap();
            assert!(matches!(
                request_line,
                "POST /responses HTTP/1.1" | "POST /responses? HTTP/1.1"
            ));
            let body = "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp-http\",\"status\":\"completed\"}}\n\n";
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            stream.write_all(response.as_bytes()).await.unwrap();
        });

        let mut route = route_for_local_server(address);
        route.retry = Some(RuntimeRetryConfig {
            enabled: true,
            max_attempts: 2,
            max_recovery_attempts: 0,
            initial_delay_secs: 0,
            exponential_backoff: false,
            backoff_multiplier: 1.0,
            jitter_secs: 0,
        });
        let input = ModelRequestInput::new(
            "model",
            vec![ModelMessage::text(super::super::MessageRole::User, "hello")],
        );
        let transport = Arc::new(TurnLocalResponsesTransport::new());
        let runtime = ModelRuntime::new_responses_websocket(transport.clone());
        let output = runtime
            .execute_text_oneshot(&route, &input, |_| async { Ok(()) }, || async { Ok(()) })
            .await
            .unwrap();
        assert!(output.is_empty());
        assert!(transport.force_http.load(Ordering::Acquire));
        server.await.unwrap();
    }

    #[tokio::test]
    async fn websocket_missing_terminal_response_id_forces_full_next_request() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let mut socket = accept_async(stream).await.unwrap();
            for (round, expected_previous, expected_input_len) in
                [(1, None, 2usize), (2, None, 2usize)]
            {
                let Message::Text(text) = socket.next().await.unwrap().unwrap() else {
                    panic!("expected a text request frame");
                };
                let value: serde_json::Value = serde_json::from_str(&text).unwrap();
                assert_eq!(value["input"].as_array().unwrap().len(), expected_input_len);
                assert_eq!(
                    value
                        .get("previous_response_id")
                        .and_then(serde_json::Value::as_str),
                    expected_previous
                );
                let response = if round == 1 {
                    r#"{"type":"response.completed","response":{"status":"completed"}}"#.to_owned()
                } else {
                    r#"{"type":"response.completed","response":{"id":"resp-2","status":"completed"}}"#
                        .to_owned()
                };
                socket.send(Message::Text(response.into())).await.unwrap();
            }
        });
        let route = route_for_local_server(address);
        let request = local_responses_request(&route);
        let transport = TurnLocalResponsesTransport::new();
        let first = transport
            .send_prepared(&route, request.clone())
            .await
            .unwrap();
        drain_response(first).await;
        assert!(*transport.force_full.lock().await);
        assert_eq!(*transport.previous_response_id.lock().await, None);
        transport.set_next_prompt_unit_start(1).await;
        let second = transport.send_prepared(&route, request).await.unwrap();
        drain_response(second).await;
        assert!(!*transport.force_full.lock().await);
        assert_eq!(
            transport.previous_response_id.lock().await.as_deref(),
            Some("resp-2")
        );
        server.await.unwrap();
    }

    #[tokio::test]
    async fn websocket_incremental_boundary_survives_frame_failure_until_send_succeeds() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let mut socket = accept_async(stream).await.unwrap();
            for (expected_input_len, expected_previous) in
                [(2usize, None), (1usize, Some("resp-1"))]
            {
                let Message::Text(text) = socket.next().await.unwrap().unwrap() else {
                    panic!("expected a text request frame");
                };
                let value: serde_json::Value = serde_json::from_str(&text).unwrap();
                assert_eq!(value["input"].as_array().unwrap().len(), expected_input_len);
                assert_eq!(
                    value
                        .get("previous_response_id")
                        .and_then(serde_json::Value::as_str),
                    expected_previous
                );
                socket
                    .send(Message::Text(
                        r#"{"type":"response.completed","response":{"id":"resp-1","status":"completed"}}"#.into(),
                    ))
                    .await
                    .unwrap();
            }
        });
        let route = route_for_local_server(address);
        let request = local_responses_request(&route);
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&request.body).unwrap()["input"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
        let transport = TurnLocalResponsesTransport::new();
        let first = transport
            .send_prepared(&route, request.clone())
            .await
            .unwrap();
        drain_response(first).await;
        transport.set_next_prompt_unit_start(99).await;
        assert!(
            transport
                .send_prepared(&route, request.clone())
                .await
                .is_err()
        );
        assert_eq!(*transport.next_prompt_unit_start.lock().await, Some(99));
        transport.set_next_prompt_unit_start(1).await;
        let second = transport.send_prepared(&route, request).await.unwrap();
        drain_response(second).await;
        assert!(!*transport.force_full.lock().await);
        assert_eq!(*transport.next_prompt_unit_start.lock().await, None);
        server.await.unwrap();
    }

    #[tokio::test]
    async fn websocket_response_drop_poison_reconnects_before_next_send() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let mut first_socket = accept_async(stream).await.unwrap();
            let Message::Text(_) = first_socket.next().await.unwrap().unwrap() else {
                panic!("expected first request frame");
            };
            first_socket
                .send(Message::Text(
                    r#"{"type":"response.output_text.delta","delta":"partial"}"#.into(),
                ))
                .await
                .unwrap();

            let (stream, _) = listener.accept().await.unwrap();
            let mut second_socket = accept_async(stream).await.unwrap();
            let Message::Text(text) = second_socket.next().await.unwrap().unwrap() else {
                panic!("expected reconnected request frame");
            };
            let value: serde_json::Value = serde_json::from_str(&text).unwrap();
            assert_eq!(value["input"].as_array().unwrap().len(), 2);
            assert!(value.get("previous_response_id").is_none());
            second_socket
                .send(Message::Text(
                    r#"{"type":"response.completed","response":{"id":"resp-2","status":"completed"}}"#.into(),
                ))
                .await
                .unwrap();
        });
        let route = route_for_local_server(address);
        let request = local_responses_request(&route);
        let transport = TurnLocalResponsesTransport::new();
        let mut partial = transport
            .send_prepared(&route, request.clone())
            .await
            .unwrap();
        assert!(partial.next_chunk().await.unwrap().is_ok());
        drop(partial);
        let next = transport.send_prepared(&route, request).await.unwrap();
        drain_response(next).await;
        server.await.unwrap();
    }

    #[tokio::test]
    async fn websocket_reconnect_starts_full_context_then_resumes_incremental_input() {
        for drop_partial in [false, true] {
            tokio::time::timeout(Duration::from_secs(5), async {
                let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
                let route = route_for_local_server(listener.local_addr().unwrap());
                let server = async {
                    let (stream, _) = listener.accept().await.unwrap();
                    let mut socket = accept_async(stream).await.unwrap();
                    assert!(matches!(socket.next().await, Some(Ok(Message::Text(_)))));
                    socket
                        .send(Message::Text(
                            r#"{"type":"response.completed","response":{"id":"resp-old","status":"completed"}}"#.into(),
                        ))
                        .await
                        .unwrap();
                    let Message::Text(text) = socket.next().await.unwrap().unwrap() else {
                        panic!("expected incremental request");
                    };
                    let value: serde_json::Value = serde_json::from_str(&text).unwrap();
                    assert_eq!(value["previous_response_id"], "resp-old");
                    assert_eq!(value["input"].as_array().unwrap().len(), 1);
                    if drop_partial {
                        socket
                            .send(Message::Text(
                                r#"{"type":"response.output_text.delta","delta":"partial"}"#.into(),
                            ))
                            .await
                            .unwrap();
                    } else {
                        socket.send(Message::Close(None)).await.unwrap();
                    }
                    let (stream, _) = listener.accept().await.unwrap();
                    let mut reconnected = accept_async(stream).await.unwrap();
                    for (previous, input_len) in [(None, 2), (Some("resp-new"), 1)] {
                        let Message::Text(text) = reconnected.next().await.unwrap().unwrap() else {
                            panic!("expected request on new connection");
                        };
                        let value: serde_json::Value = serde_json::from_str(&text).unwrap();
                        assert_eq!(value.get("previous_response_id").and_then(|v| v.as_str()), previous);
                        assert_eq!(value["input"].as_array().unwrap().len(), input_len);
                        reconnected
                            .send(Message::Text(
                                r#"{"type":"response.completed","response":{"id":"resp-new","status":"completed"}}"#.into(),
                            ))
                            .await
                            .unwrap();
                    }
                };
                let client = async {
                    let request = local_responses_request(&route);
                    let transport = TurnLocalResponsesTransport::new();
                    drain_response(transport.send_prepared(&route, request.clone()).await.unwrap()).await;
                    transport.set_next_prompt_unit_start(1).await;
                    let mut interrupted = transport.send_prepared(&route, request.clone()).await.unwrap();
                    let chunk = interrupted.next_chunk().await.unwrap();
                    if drop_partial {
                        assert!(chunk.is_ok());
                    } else {
                        let error = chunk.unwrap_err();
                        assert_eq!(error.code.as_deref(), Some("websocket_closed_before_terminal"));
                        assert_eq!(error.retry_hint, RetryHint::Retryable);
                    }
                    drop(interrupted);
                    drain_response(transport.send_prepared(&route, request.clone()).await.unwrap()).await;
                    transport.set_next_prompt_unit_start(1).await;
                    drain_response(transport.send_prepared(&route, request).await.unwrap()).await;
                };
                tokio::join!(server, client);
            })
            .await
            .expect("WebSocket reconnect test timed out");
        }
    }

    fn route_for_local_server(address: std::net::SocketAddr) -> ResolvedModelRoute {
        let config = format!(
            r#"active_provider = "vendor"
[providers.vendor]
protocol = "responses"
default_model = "model"
flavor = "standard"
[providers.vendor.auth]
type = "none"
[providers.vendor.endpoints]
base_url = "http://{address}"
[providers.vendor.models.model]
[providers.vendor.models.model.transport]
websocket = true
[providers.vendor.models.model.capabilities]
reasoning = true
"#
        );
        RuntimeConfig::from_toml(&config)
            .unwrap()
            .resolve(&ProtocolRegistry::builtins())
            .unwrap()
            .route("vendor", "model")
            .unwrap()
            .clone()
    }

    fn local_responses_request(route: &ResolvedModelRoute) -> PreparedHttpRequest {
        let mut input = ModelRequestInput::new(
            "model",
            vec![
                ModelMessage::text(super::super::MessageRole::User, "first"),
                ModelMessage::text(super::super::MessageRole::User, "second"),
            ],
        );
        input.message_origins = vec!["first".into(), "second".into()];
        route.binding.prepare_request(&input).unwrap()
    }

    #[tokio::test]
    async fn non_success_status_preserves_provider_detail_before_decoding() {
        let route = route("responses");
        let transport = Arc::new(QueueTransport {
            responses: Mutex::new(VecDeque::from([Ok(TransportResponse::from_chunks(
                429,
                BTreeMap::from([("retry-after".into(), "7".into())]),
                vec![b"not protocol data".to_vec()],
            ))])),
        });
        let runtime = ModelRuntime::new(transport);
        let error = runtime
            .execute_attempt(&route, &ModelRequestInput::new("model", Vec::new()))
            .await
            .unwrap_err();
        assert_eq!(error.failure.kind, FailureKind::RateLimited);
        assert_eq!(error.failure.status, Some(429));
        assert_eq!(error.failure.retry_hint, RetryHint::RetryAfterSeconds(7));
        assert_eq!(error.failure.detail(), "not protocol data");
        assert!(!error.partial.side_effects.observable());
    }

    #[tokio::test]
    async fn client_error_detail_is_preserved_without_becoming_retryable() {
        let route = route("responses");
        let transport = Arc::new(QueueTransport {
            responses: Mutex::new(VecDeque::from([Ok(response(
                400,
                vec![br#"{"error":{"message":"input[3].role is invalid","unknown":{"expected":"user"}}}"#],
            ))])),
        });
        let runtime = ModelRuntime::new(transport);
        let error = runtime
            .execute_attempt(&route, &ModelRequestInput::new("model", Vec::new()))
            .await
            .unwrap_err();
        assert_eq!(error.failure.status, Some(400));
        assert_eq!(error.failure.retry_hint, RetryHint::Never);
        assert!(error.failure.detail().contains("input[3].role is invalid"));
        assert!(error.failure.detail().contains("\"expected\":\"user\""));
    }

    #[tokio::test]
    async fn attempt_accumulates_atomic_assistant_and_side_effects_in_event_order() {
        let route = route("responses");
        let body = concat!(
            "event: response.output_item.added\n",
            "data: {\"type\":\"response.output_item.added\",\"item\":{\"type\":\"reasoning\",\"id\":\"r1\"}}\n\n",
            "event: response.reasoning_text.delta\n",
            "data: {\"type\":\"response.reasoning_text.delta\",\"item_id\":\"r1\",\"delta\":\"plan\"}\n\n",
            "event: response.output_item.done\n",
            "data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"reasoning\",\"id\":\"r1\",\"content\":[{\"type\":\"reasoning_text\",\"text\":\"plan\"}]}}\n\n",
            "event: response.output_text.delta\n",
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"hello\"}\n\n",
            "event: response.completed\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp-1\",\"status\":\"completed\"}}\n\n"
        );
        let runtime = ModelRuntime::new(Arc::new(QueueTransport {
            responses: Mutex::new(VecDeque::from([Ok(response(200, vec![body.as_bytes()]))])),
        }));
        let result = runtime
            .execute_attempt(&route, &ModelRequestInput::new("model", Vec::new()))
            .await
            .unwrap();
        assert_eq!(result.terminal, TerminalStatus::Completed);
        assert_eq!(result.snapshot.response_id.as_deref(), Some("resp-1"));
        assert!(result.snapshot.side_effects.observable());
        assert!(matches!(
            result.snapshot.assistant.content.as_slice(),
            [ContentPart::Reasoning { text, .. }, ContentPart::Text(value)]
                if text == "plan" && value == "hello"
        ));
    }

    #[test]
    fn accumulator_separates_reasoning_summary_parts() {
        let mut accumulator = AttemptAccumulator::default();
        for event in [
            ModelEvent::ReasoningSummaryDelta {
                item_id: "r1".into(),
                summary_index: 0,
                text: "first".into(),
            },
            ModelEvent::ReasoningSummaryDone {
                item_id: "r1".into(),
                summary_index: 0,
                text: "first".into(),
            },
            ModelEvent::ReasoningSummaryDelta {
                item_id: "r1".into(),
                summary_index: 1,
                text: "second".into(),
            },
        ] {
            accumulator.consume_observed(event).unwrap();
        }
        assert!(matches!(
            accumulator.snapshot().assistant.content.as_slice(),
            [ContentPart::Reasoning { text, .. }] if text == "first\n\nsecond"
        ));
    }

    #[test]
    fn accumulator_preserves_mixed_reasoning_parts_before_done() {
        for events in [
            vec![
                ModelEvent::ReasoningSummaryDelta {
                    item_id: "r1".into(),
                    summary_index: 0,
                    text: "summary".into(),
                },
                ModelEvent::ReasoningDelta {
                    item_id: "r1".into(),
                    text: "raw reasoning".into(),
                },
            ],
            vec![
                ModelEvent::ReasoningDelta {
                    item_id: "r1".into(),
                    text: "raw reasoning".into(),
                },
                ModelEvent::ReasoningSummaryDelta {
                    item_id: "r1".into(),
                    summary_index: 0,
                    text: "summary".into(),
                },
            ],
        ] {
            let mut accumulator = AttemptAccumulator::default();
            for event in events {
                accumulator.consume_observed(event).unwrap();
            }
            assert!(matches!(
                accumulator.snapshot().assistant.content.as_slice(),
                [ContentPart::Reasoning { text, .. }]
                    if text == "summary\n\nraw reasoning"
            ));
        }
    }

    #[test]
    fn accumulator_preserves_tool_order_and_normalizes_completed_with_tools() {
        let mut accumulator = AttemptAccumulator::default();
        for (id, name) in [("b", "second"), ("a", "first")] {
            accumulator
                .consume_observed(ModelEvent::ToolStarted {
                    id: id.into(),
                    name: name.into(),
                })
                .unwrap();
            accumulator
                .consume_observed(ModelEvent::ToolDone {
                    id: id.into(),
                    name: name.into(),
                    arguments: serde_json::json!({}),
                })
                .unwrap();
        }
        accumulator
            .consume_observed(ModelEvent::Terminal {
                status: TerminalStatus::Completed,
            })
            .unwrap();
        let result = accumulator.finish().unwrap();
        assert_eq!(result.terminal, TerminalStatus::ToolUse);
        assert_eq!(result.snapshot.completed_tools[0].id, "b");
        assert_eq!(result.snapshot.completed_tools[1].id, "a");
    }

    #[test]
    fn partial_failure_preserves_reasoning_text_replay_and_pending_tools() {
        let replay = super::super::OpaqueReplayState::new(
            "test.replay",
            1,
            super::super::ReplayProducer {
                scope: super::super::ReplayScope::Protocol,
                protocol_id: super::super::ProtocolId::new("responses").unwrap(),
                profile_identity: None,
                route_identity: None,
            },
            serde_json::json!({"opaque":true}),
        );
        let mut accumulator = AttemptAccumulator::default();
        accumulator
            .consume_observed(ModelEvent::ReasoningDone {
                item_id: "r".into(),
                text: "plan".into(),
                replay: Some(replay.clone()),
            })
            .unwrap();
        accumulator
            .consume_observed(ModelEvent::TextDelta { text: "hi".into() })
            .unwrap();
        accumulator
            .consume_observed(ModelEvent::ToolStarted {
                id: "call".into(),
                name: "search".into(),
            })
            .unwrap();
        let failure = accumulator.failed(
            ModelFailure::new(FailurePhase::Transport, FailureKind::Http)
                .with_retry_hint(RetryHint::Retryable),
        );
        assert_eq!(failure.partial.pending_tools[0].id, "call");
        assert!(matches!(
            failure.partial.assistant.content.as_slice(),
            [ContentPart::Reasoning { replay: Some(value), .. }, ContentPart::Text(text)]
                if value == &replay && text == "hi"
        ));
    }

    #[derive(Default)]
    struct TestDriver {
        requests: VecDeque<ModelRequestInput>,
        observed: Vec<ModelEvent>,
        persisted: Vec<ModelMessage>,
        tool_batches: Vec<Vec<CompletedToolCall>>,
        recoveries: Vec<ModelAttemptSnapshot>,
        attempts_started: Vec<(usize, usize)>,
        attempts_finished: Vec<(usize, usize, AttemptOutcome)>,
        retries: Vec<(usize, usize, usize, Duration)>,
        commits: Vec<usize>,
        continuation_decisions: VecDeque<TurnContinuationDecision>,
        finalized: usize,
    }

    #[async_trait]
    impl TurnDriver for TestDriver {
        async fn prepare_iteration(
            &mut self,
            _iteration: usize,
        ) -> Result<ModelRequestInput, ModelFailure> {
            self.requests
                .pop_front()
                .ok_or_else(|| runtime_invalid("missing test request"))
        }

        async fn attempt_started(
            &mut self,
            iteration: usize,
            attempt: usize,
        ) -> Result<(), ModelFailure> {
            self.attempts_started.push((iteration, attempt));
            Ok(())
        }

        async fn commit_before_first_send(&mut self, iteration: usize) -> Result<(), ModelFailure> {
            self.commits.push(iteration);
            Ok(())
        }

        async fn attempt_finished(
            &mut self,
            iteration: usize,
            attempt: usize,
            outcome: &AttemptOutcome,
        ) -> Result<(), ModelFailure> {
            self.attempts_finished
                .push((iteration, attempt, outcome.clone()));
            Ok(())
        }

        async fn retry_scheduled(
            &mut self,
            iteration: usize,
            attempt: usize,
            next_attempt: usize,
            delay: Duration,
            _failure: &ModelFailure,
        ) -> Result<(), ModelFailure> {
            self.retries.push((iteration, attempt, next_attempt, delay));
            Ok(())
        }

        async fn observe_event(&mut self, event: &ModelEvent) -> Result<(), ModelFailure> {
            self.observed.push(event.clone());
            Ok(())
        }

        async fn persist_assistant(
            &mut self,
            assistant: &ModelMessage,
        ) -> Result<(), ModelFailure> {
            self.persisted.push(assistant.clone());
            Ok(())
        }

        async fn execute_tools(&mut self, tools: &[CompletedToolCall]) -> Result<(), ModelFailure> {
            self.tool_batches.push(tools.to_vec());
            Ok(())
        }

        async fn recover_iteration(
            &mut self,
            partial: &ModelAttemptSnapshot,
            _failure: &ModelFailure,
        ) -> Result<(), ModelFailure> {
            self.recoveries.push(partial.clone());
            Ok(())
        }

        async fn after_assistant_persisted(
            &mut self,
            _result: &ModelAttemptResult,
        ) -> Result<TurnContinuationDecision, ModelFailure> {
            Ok(self
                .continuation_decisions
                .pop_front()
                .unwrap_or(TurnContinuationDecision::Finalize))
        }

        async fn finalize(&mut self, _result: &ModelAttemptResult) -> Result<(), ModelFailure> {
            self.finalized += 1;
            Ok(())
        }
    }

    #[tokio::test]
    async fn orchestrator_retries_before_side_effects() {
        let route = route("responses");
        let success = concat!(
            "event: response.output_text.delta\n",
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"ok\"}\n\n",
            "event: response.completed\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\"}}\n\n"
        );
        let runtime = ModelRuntime::new(Arc::new(QueueTransport {
            responses: Mutex::new(VecDeque::from([
                Err(
                    ModelFailure::new(FailurePhase::Transport, FailureKind::Timeout)
                        .with_retry_hint(RetryHint::Retryable),
                ),
                Ok(response(200, vec![success.as_bytes()])),
            ])),
        }));
        let mut driver = TestDriver {
            requests: VecDeque::from([ModelRequestInput::new("model", Vec::new())]),
            ..TestDriver::default()
        };
        let result = TurnOrchestrator::new(
            runtime,
            TurnLimits {
                max_iterations: 2,
                max_tool_calls: Some(0),
            },
        )
        .run(&route, &mut driver)
        .await
        .unwrap();
        assert_eq!(result.iterations, 1);
        assert_eq!(driver.persisted.len(), 1);
        assert_eq!(driver.recoveries.len(), 0);
        assert_eq!(driver.attempts_started, vec![(0, 1), (0, 2)]);
        assert_eq!(driver.attempts_finished.len(), 2);
        assert_eq!(driver.retries, vec![(0, 1, 2, Duration::from_secs(1))]);
        assert_eq!(driver.commits, vec![0]);
        assert_eq!(driver.finalized, 1);
    }

    #[tokio::test]
    async fn orchestrator_recovers_after_observed_side_effects() {
        let route = route("responses");
        let partial = concat!(
            "event: response.output_text.delta\n",
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"partial\"}\n\n"
        );
        let success = concat!(
            "event: response.output_text.delta\n",
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"done\"}\n\n",
            "event: response.completed\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\"}}\n\n"
        );
        let runtime = ModelRuntime::new(Arc::new(QueueTransport {
            responses: Mutex::new(VecDeque::from([
                Ok(TransportResponse::from_results(
                    200,
                    BTreeMap::new(),
                    vec![
                        Ok(partial.as_bytes().to_vec()),
                        Err(
                            ModelFailure::new(FailurePhase::Transport, FailureKind::Http)
                                .with_retry_hint(RetryHint::Retryable),
                        ),
                    ],
                )),
                Ok(response(200, vec![success.as_bytes()])),
            ])),
        }));
        let mut driver = TestDriver {
            requests: VecDeque::from([
                ModelRequestInput::new("model", Vec::new()),
                ModelRequestInput::new("model", Vec::new()),
            ]),
            ..TestDriver::default()
        };
        let result = TurnOrchestrator::new(
            runtime,
            TurnLimits {
                max_iterations: 3,
                max_tool_calls: Some(0),
            },
        )
        .run(&route, &mut driver)
        .await
        .unwrap();
        assert_eq!(result.iterations, 2);
        assert_eq!(result.recovery_attempts, 1);
        assert_eq!(driver.recoveries.len(), 1);
        assert!(matches!(
            driver.recoveries[0].assistant.content.as_slice(),
            [ContentPart::Text(text)] if text == "partial"
        ));
    }

    #[tokio::test]
    async fn no_tool_reply_can_continue_before_finalization() {
        let route = route("responses");
        let response_body = |text: &str| {
            format!(
                "event: response.output_text.delta\ndata: {{\"type\":\"response.output_text.delta\",\"delta\":\"{text}\"}}\n\nevent: response.completed\ndata: {{\"type\":\"response.completed\",\"response\":{{\"status\":\"completed\"}}}}\n\n"
            )
        };
        let first = response_body("first");
        let second = response_body("second");
        let runtime = ModelRuntime::new(Arc::new(QueueTransport {
            responses: Mutex::new(VecDeque::from([
                Ok(response(200, vec![first.as_bytes()])),
                Ok(response(200, vec![second.as_bytes()])),
            ])),
        }));
        let mut driver = TestDriver {
            requests: VecDeque::from([
                ModelRequestInput::new("model", Vec::new()),
                ModelRequestInput::new("model", Vec::new()),
            ]),
            continuation_decisions: VecDeque::from([
                TurnContinuationDecision::Continue,
                TurnContinuationDecision::Finalize,
            ]),
            ..TestDriver::default()
        };
        let result = TurnOrchestrator::new(
            runtime,
            TurnLimits {
                max_iterations: 3,
                max_tool_calls: Some(0),
            },
        )
        .run(&route, &mut driver)
        .await
        .unwrap();
        assert_eq!(result.iterations, 2);
        assert_eq!(driver.persisted.len(), 2);
        assert_eq!(driver.finalized, 1);
    }

    #[tokio::test]
    async fn tool_budget_is_checked_before_assistant_persistence() {
        let route = route("responses");
        let body = concat!(
            "event: response.output_item.added\n",
            "data: {\"type\":\"response.output_item.added\",\"item\":{\"type\":\"function_call\",\"id\":\"item-1\",\"call_id\":\"call-1\",\"name\":\"search\"}}\n\n",
            "event: response.function_call_arguments.done\n",
            "data: {\"type\":\"response.function_call_arguments.done\",\"item_id\":\"item-1\",\"arguments\":\"{}\"}\n\n",
            "event: response.output_item.done\n",
            "data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"function_call\",\"id\":\"item-1\",\"call_id\":\"call-1\",\"name\":\"search\",\"arguments\":\"{}\"}}\n\n",
            "event: response.completed\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\"}}\n\n"
        );
        let runtime = ModelRuntime::new(Arc::new(QueueTransport {
            responses: Mutex::new(VecDeque::from([Ok(response(200, vec![body.as_bytes()]))])),
        }));
        let mut driver = TestDriver {
            requests: VecDeque::from([ModelRequestInput::new("model", Vec::new())]),
            ..TestDriver::default()
        };
        let error = TurnOrchestrator::new(
            runtime,
            TurnLimits {
                max_iterations: 2,
                max_tool_calls: Some(0),
            },
        )
        .run(&route, &mut driver)
        .await
        .unwrap_err();
        assert_eq!(error.code.as_deref(), Some("max_tool_calls"));
        assert!(driver.persisted.is_empty());
        assert!(driver.tool_batches.is_empty());
    }

    #[test]
    fn retry_delay_includes_bounded_jitter() {
        let config = RuntimeRetryConfig {
            enabled: true,
            max_attempts: 2,
            max_recovery_attempts: 0,
            initial_delay_secs: 3,
            exponential_backoff: false,
            backoff_multiplier: 1.0,
            jitter_secs: 2,
        };
        let delay = retry_delay(&config, 1, RetryHint::Retryable);
        assert_eq!(delay, Duration::from_secs(3));
        let mut exponential = config.clone();
        exponential.exponential_backoff = true;
        let delay = retry_delay(&exponential, 1, RetryHint::Retryable);
        assert!((3..=5).contains(&delay.as_secs()));
        assert_eq!(
            retry_delay(&config, 1, RetryHint::RetryAfterSeconds(7)),
            Duration::from_secs(7)
        );
    }

    #[test]
    fn resolved_route_carries_effective_retry_configuration() {
        let route = route("responses");
        assert_eq!(route.retry.as_ref().unwrap().max_attempts, 3);
    }
}
