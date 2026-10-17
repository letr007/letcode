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
            // The request keeps the headers it was prepared with, so a
            // request decorator that targeted the WebSocket path has its
            // headers ride this HTTP fallback as well.
            return ResolvedProviderTransport
                .send_prepared(route, request)
                .await;
        }
        if self.poisoned_response.swap(false, Ordering::AcqRel) {
            *self.session.lock().await = None;
        }
        let mut session_guard = self.session.lock().await;
        if session_guard
            .as_ref()
            .is_some_and(|session| session.is_closed())
        {
            *session_guard = None;
        }
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
    /// Boxed so `Result<_, ModelAttemptFailure>` stays small on every call path;
    /// only the failure path pays for the allocation.
    pub partial: Box<ModelAttemptSnapshot>,
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
        on_delta: F,
        on_retry: R,
    ) -> Result<String, ModelFailure>
    where
        F: FnMut(&str) -> Fut + Send,
        Fut: std::future::Future<Output = Result<(), ModelFailure>> + Send,
        R: FnMut() -> Rfut + Send,
        Rfut: std::future::Future<Output = Result<(), ModelFailure>> + Send,
    {
        self.execute_text_oneshot_with_usage(route, input, on_delta, on_retry)
            .await
            .map(|(text, _)| text)
    }

    /// Retain provider-reported accounting from each attempt, including retries.
    /// Only usage/cache events are returned, never hidden reasoning or prompt text.
    pub async fn execute_text_oneshot_with_usage<F, Fut, R, Rfut>(
        &self,
        route: &ResolvedModelRoute,
        input: &ModelRequestInput,
        on_delta: F,
        on_retry: R,
    ) -> Result<(String, Vec<ModelEvent>), ModelFailure>
    where
        F: FnMut(&str) -> Fut + Send,
        Fut: std::future::Future<Output = Result<(), ModelFailure>> + Send,
        R: FnMut() -> Rfut + Send,
        Rfut: std::future::Future<Output = Result<(), ModelFailure>> + Send,
    {
        let request = route.binding.prepare_request(input)?;
        self.execute_prepared_text_oneshot_with_usage(route, request, on_delta, on_retry)
            .await
    }

    /// Runs a request the caller prepared, and possibly decorated, itself.
    pub async fn execute_prepared_text_oneshot<F, Fut, R, Rfut>(
        &self,
        route: &ResolvedModelRoute,
        request: PreparedHttpRequest,
        on_delta: F,
        on_retry: R,
    ) -> Result<String, ModelFailure>
    where
        F: FnMut(&str) -> Fut + Send,
        Fut: std::future::Future<Output = Result<(), ModelFailure>> + Send,
        R: FnMut() -> Rfut + Send,
        Rfut: std::future::Future<Output = Result<(), ModelFailure>> + Send,
    {
        self.execute_prepared_text_oneshot_with_usage(route, request, on_delta, on_retry)
            .await
            .map(|(text, _)| text)
    }

    /// Runs a request the caller prepared, and possibly decorated, itself.
    /// Accounting, retry classification and terminal validation are identical
    /// to [`Self::execute_text_oneshot_with_usage`].
    pub async fn execute_prepared_text_oneshot_with_usage<F, Fut, R, Rfut>(
        &self,
        route: &ResolvedModelRoute,
        request: PreparedHttpRequest,
        mut on_delta: F,
        mut on_retry: R,
    ) -> Result<(String, Vec<ModelEvent>), ModelFailure>
    where
        F: FnMut(&str) -> Fut + Send,
        Fut: std::future::Future<Output = Result<(), ModelFailure>> + Send,
        R: FnMut() -> Rfut + Send,
        Rfut: std::future::Future<Output = Result<(), ModelFailure>> + Send,
    {
        let mut usage_events = Vec::new();
        let retry = route.retry.clone().unwrap_or_else(default_retry_config);
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
                    usage_events.extend(result.snapshot.events.into_iter().filter(|event| {
                        matches!(event, ModelEvent::Usage { .. } | ModelEvent::Cache { .. })
                    }));
                    return Ok((observer.text, usage_events));
                }
                Err(error)
                    if !observer.rejected_event
                        && retryable_failure(&error.failure)
                        && retry.enabled
                        && attempt < retry.max_attempts =>
                {
                    usage_events.extend(
                        error
                            .partial
                            .events
                            .iter()
                            .filter(|event| {
                                matches!(event, ModelEvent::Usage { .. } | ModelEvent::Cache { .. })
                            })
                            .cloned(),
                    );
                    let delay = overload_fixed_retry_delay(route, &retry, &error.failure)
                        .unwrap_or_else(|| retry_delay(&retry, attempt, error.failure.retry_hint));
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
                    partial: Box::new(ModelAttemptSnapshot::default()),
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
                    partial: Box::new(ModelAttemptSnapshot::default()),
                })?;
        if !(200..300).contains(&response.status) {
            let status = response.status;
            let retry_hint = response.retry_hint();
            let detail = read_provider_error_detail(&mut response, &secret_refs).await;
            let mut failure = route
                .binding
                .decode_http_error(status, detail.as_bytes(), retry_hint)
                .unwrap_or_else(|| http_status_failure(status, retry_hint));
            if !detail.is_empty() && failure.detail().is_empty() {
                failure = failure.with_detail_redacted(detail, &secret_refs);
            }
            return Err(ModelAttemptFailure {
                failure,
                partial: Box::new(ModelAttemptSnapshot::default()),
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
            let chunk = match chunk {
                Ok(chunk) => chunk,
                Err(failure) => {
                    // Some gateways close after the terminal event without terminating the
                    // chunked body, so the decoder may already hold the whole response.
                    return match decoder.finish() {
                        Ok(events) => {
                            observe_events(observer, &mut accumulator, events).await?;
                            accumulator.finish()
                        }
                        Err(_) => Err(accumulator.failed(failure)),
                    };
                }
            };
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
            partial: Box::new(self.snapshot()),
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
                partial: Box::new(snapshot),
            });
        }
        let terminal = match terminal {
            TerminalStatus::Completed if !snapshot.completed_tools.is_empty() => {
                TerminalStatus::ToolUse
            }
            TerminalStatus::ToolUse if snapshot.completed_tools.is_empty() => {
                return Err(ModelAttemptFailure {
                    failure: runtime_invalid("tool terminal contains no tools"),
                    partial: Box::new(snapshot),
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
                    partial: Box::new(snapshot),
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

    async fn reconcile_pending_tools(&mut self) -> Result<(), ModelFailure> {
        Ok(())
    }

    fn take_recovered_tool_calls(&mut self) -> usize {
        0
    }

    async fn abort_iteration(
        &mut self,
        _partial: &ModelAttemptSnapshot,
        _failure: &ModelFailure,
    ) -> Result<(), ModelFailure> {
        self.reconcile_pending_tools().await
    }

    async fn reject_tool_batch(
        &mut self,
        _partial: &ModelAttemptSnapshot,
        _failure: &ModelFailure,
    ) -> Result<(), ModelFailure> {
        Ok(())
    }

    async fn persist_assistant(&mut self, assistant: &ModelMessage) -> Result<(), ModelFailure>;

    async fn persist_partial_assistant(
        &mut self,
        assistant: &ModelMessage,
    ) -> Result<(), ModelFailure> {
        self.persist_assistant(assistant).await
    }

    async fn execute_tools(&mut self, tools: &[CompletedToolCall]) -> Result<(), ModelFailure>;

    fn user_turn_handoff(&self) -> bool {
        false
    }

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
                    let recovered_tools = driver.take_recovered_tool_calls();
                    let next_tool_total = total_tools.saturating_add(recovered_tools);
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
                    recovery_attempts += 1;
                    iteration += 1;
                    continue;
                }
                Err(error) => {
                    let mut failure = error.failure;
                    if let Err(cleanup) = driver.abort_iteration(&error.partial, &failure).await {
                        failure = failure_with_cleanup(failure, cleanup);
                    }
                    return Err(failure);
                }
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
                    let failure =
                        ModelFailure::new(FailurePhase::Finish, FailureKind::InvalidRequest)
                            .with_code("max_tool_calls");
                    if let Err(cleanup) = driver.reject_tool_batch(&result.snapshot, &failure).await
                    {
                        return Err(failure_with_cleanup(failure, cleanup));
                    }
                    return Err(failure);
                }
                total_tools = next_tool_total;
            }
            if let Err(mut error) = driver.persist_assistant(&result.snapshot.assistant).await {
                if let Err(cleanup) = driver.abort_iteration(&result.snapshot, &error).await {
                    error = failure_with_cleanup(error, cleanup);
                }
                return Err(error);
            }
            if result.terminal == TerminalStatus::ToolUse {
                if let Err(mut error) = driver.execute_tools(&result.snapshot.completed_tools).await
                {
                    if let Err(cleanup) = driver.abort_iteration(&result.snapshot, &error).await {
                        error = failure_with_cleanup(error, cleanup);
                    }
                    return Err(error);
                }
                if driver.user_turn_handoff() {
                    driver.finalize(&result).await?;
                    return Ok(TurnResult {
                        final_attempt: result,
                        iterations: iteration + 1,
                        tool_calls: total_tools,
                        recovery_attempts,
                    });
                }
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
                    partial: Box::new(ModelAttemptSnapshot::default()),
                });
            }
            if attempt == 1
                && let Err(failure) = driver.commit_before_first_send(iteration).await
            {
                return Err(ModelAttemptFailure {
                    failure,
                    partial: Box::new(ModelAttemptSnapshot::default()),
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
                    partial: Box::new(
                        result
                            .as_ref()
                            .map(|result| result.snapshot.clone())
                            .unwrap_or_else(|error| error.partial.as_ref().clone()),
                    ),
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
                    let delay = overload_fixed_retry_delay(route, retry, &error.failure)
                        .unwrap_or_else(|| retry_delay(retry, attempt, error.failure.retry_hint));
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

fn fixed_retry_delay(config: &RuntimeRetryConfig) -> Option<Duration> {
    (config.initial_delay_secs > 0).then(|| Duration::from_secs(config.initial_delay_secs))
}

fn overload_fixed_retry_delay(
    route: &ResolvedModelRoute,
    config: &RuntimeRetryConfig,
    failure: &ModelFailure,
) -> Option<Duration> {
    if failure.code.as_deref() != Some("server_is_overloaded")
        || !crate::fast_mode::is_fast_capable_model(&route.model)
    {
        return None;
    }
    fixed_retry_delay(config)
}

pub(crate) fn failure_with_cleanup(failure: ModelFailure, cleanup: ModelFailure) -> ModelFailure {
    let detail = failure.detail().to_owned();
    let cleanup_detail = if cleanup.detail().is_empty() {
        cleanup.to_string()
    } else {
        cleanup.detail().to_owned()
    };
    let combined = if detail.is_empty() {
        format!("pending tool cleanup failed: {cleanup_detail}")
    } else {
        format!("{detail}; pending tool cleanup failed: {cleanup_detail}")
    };
    failure.with_detail(combined)
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
#[path = "../../tests/internal/model_runtime/runtime/tests.rs"]
mod tests;
