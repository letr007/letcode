use super::{
    ContentPart, FailureKind, FailurePhase, HttpMethod, ModelEvent, ModelFailure, ModelMessage,
    ModelRequestInput, PreparedHttpRequest, ResolvedModelRoute, RetryHint, RuntimeRetryConfig,
    TerminalStatus, TransportResponse, UserMessageSubmission,
};
use async_trait::async_trait;
use futures_util::Stream;
use std::collections::{BTreeMap, HashMap};
use std::pin::Pin;
use std::sync::{
    Arc, Mutex as StdMutex,
    atomic::{AtomicBool, AtomicU8, Ordering},
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

#[derive(Debug, Clone)]
pub struct ResponseSteerRequest {
    pub submission: UserMessageSubmission,
}

const STEER_UNSUPPORTED: u8 = 0;
const STEER_UNAVAILABLE: u8 = 1;
const STEER_AVAILABLE: u8 = 2;
const STEER_PENDING: u8 = 3;
const STEER_DISABLED: u8 = 4;

#[derive(Clone, Debug)]
pub(crate) struct ResponseSteerHandle {
    state: Arc<AtomicU8>,
    available: Arc<tokio::sync::Notify>,
    returned_submissions: Arc<StdMutex<Vec<UserMessageSubmission>>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResponseSteerDecision {
    Accepted,
    Deferred,
    Unsupported,
}

impl ResponseSteerHandle {
    pub(crate) fn new(supported: bool) -> Self {
        Self {
            state: Arc::new(AtomicU8::new(if supported {
                STEER_UNAVAILABLE
            } else {
                STEER_UNSUPPORTED
            })),
            available: Arc::new(tokio::sync::Notify::new()),
            returned_submissions: Arc::new(StdMutex::new(Vec::new())),
        }
    }

    pub(crate) async fn wait_available(&self) {
        loop {
            let notified = self.available.notified();
            if self.state.load(Ordering::Acquire) == STEER_AVAILABLE {
                return;
            }
            notified.await;
        }
    }

    pub(crate) fn has_returned_submissions(&self) -> bool {
        !self.returned_submissions.lock().unwrap().is_empty()
    }

    pub(crate) fn take_returned_submissions(&self) -> Vec<UserMessageSubmission> {
        std::mem::take(&mut *self.returned_submissions.lock().unwrap())
    }

    fn return_submission(&self, submission: UserMessageSubmission) {
        self.steer_failed();
        self.returned_submissions.lock().unwrap().push(submission);
    }

    pub(crate) fn try_claim(&self) -> ResponseSteerDecision {
        match self.state.load(Ordering::Acquire) {
            STEER_UNSUPPORTED => ResponseSteerDecision::Unsupported,
            STEER_AVAILABLE => self
                .state
                .compare_exchange(
                    STEER_AVAILABLE,
                    STEER_PENDING,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .map(|_| ResponseSteerDecision::Accepted)
                .unwrap_or(ResponseSteerDecision::Deferred),
            _ => ResponseSteerDecision::Deferred,
        }
    }

    fn response_created(&self) {
        if self
            .state
            .compare_exchange(
                STEER_UNAVAILABLE,
                STEER_AVAILABLE,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
        {
            self.available.notify_one();
        }
    }

    fn steer_committed(&self) {
        let _ = self.state.compare_exchange(
            STEER_PENDING,
            STEER_UNAVAILABLE,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
    }

    pub(crate) fn steer_failed(&self) {
        let _ = self.state.compare_exchange(
            STEER_PENDING,
            STEER_UNAVAILABLE,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
    }

    pub(crate) fn defer_claim(&self) {
        self.steer_failed();
    }

    fn begin_terminal(&self) -> bool {
        loop {
            match self.state.load(Ordering::Acquire) {
                STEER_AVAILABLE => {
                    if self
                        .state
                        .compare_exchange(
                            STEER_AVAILABLE,
                            STEER_UNAVAILABLE,
                            Ordering::AcqRel,
                            Ordering::Acquire,
                        )
                        .is_ok()
                    {
                        return false;
                    }
                }
                STEER_PENDING => return true,
                _ => return false,
            }
        }
    }

    fn terminal(&self) {
        let _ = self.state.compare_exchange(
            STEER_AVAILABLE,
            STEER_UNAVAILABLE,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
    }

    fn disable(&self) {
        loop {
            match self.state.load(Ordering::Acquire) {
                STEER_UNSUPPORTED => return,
                state => {
                    if self
                        .state
                        .compare_exchange(
                            state,
                            STEER_DISABLED,
                            Ordering::AcqRel,
                            Ordering::Acquire,
                        )
                        .is_ok()
                    {
                        return;
                    }
                }
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn availability(&self) -> bool {
        self.state.load(Ordering::Acquire) == STEER_AVAILABLE
    }

    #[cfg(test)]
    pub(crate) fn pending(&self) -> bool {
        self.state.load(Ordering::Acquire) == STEER_PENDING
    }
}

pub(crate) struct TurnLocalResponsesTransport {
    session: Arc<tokio::sync::Mutex<Option<crate::model_runtime::websocket::TurnLocalWsSession>>>,
    previous_response_id: Arc<tokio::sync::Mutex<Option<String>>>,
    force_full: Arc<tokio::sync::Mutex<bool>>,
    next_prompt_unit_start: Arc<tokio::sync::Mutex<Option<usize>>>,
    poisoned_response: Arc<AtomicBool>,
    force_http: Arc<AtomicBool>,
    steer_handle: Arc<StdMutex<Option<ResponseSteerHandle>>>,
    steer_receiver:
        Arc<StdMutex<Option<tokio::sync::mpsc::UnboundedReceiver<ResponseSteerRequest>>>>,
    cached_successor: Arc<tokio::sync::Mutex<Option<Vec<u8>>>>,
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
            steer_handle: Arc::new(StdMutex::new(None)),
            steer_receiver: Arc::new(StdMutex::new(None)),
            cached_successor: Arc::new(tokio::sync::Mutex::new(None)),
        }
    }

    pub(crate) async fn set_steer_receiver(
        &self,
        receiver: tokio::sync::mpsc::UnboundedReceiver<ResponseSteerRequest>,
    ) {
        *self.steer_receiver.lock().unwrap() = Some(receiver);
    }

    pub(crate) fn set_steer_handle(&self, handle: ResponseSteerHandle) {
        *self.steer_handle.lock().unwrap() = Some(handle);
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
        let cached_successor = self.cached_successor.lock().await.take();
        if cached_successor.is_none() {
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
        }
        let steer_receiver = self.steer_receiver.clone();
        let steer_handle = self.steer_handle.lock().unwrap().clone();
        drop(session_guard);
        let stream = websocket_response_stream(
            self.session.clone(),
            self.previous_response_id.clone(),
            self.force_full.clone(),
            self.next_prompt_unit_start.clone(),
            self.poisoned_response.clone(),
            self.force_http.clone(),
            self.cached_successor.clone(),
            route.binding.clone(),
            steer_handle,
            steer_receiver,
            cached_successor,
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

fn unknown_steer_failure(mut error: ModelFailure, steer_submitted: bool) -> ModelFailure {
    if steer_submitted {
        error = error
            .with_code("websocket_steer_outcome_unknown")
            .with_retry_hint(RetryHint::Never);
    }
    error
}

fn is_successful_steer_terminal(event_type: &str, value: Option<&serde_json::Value>) -> bool {
    event_type == "response.completed"
        || (event_type == "response.incomplete"
            && value
                .and_then(|value| value.get("response"))
                .and_then(|response| response.get("incomplete_details"))
                .and_then(|details| details.get("reason"))
                .and_then(serde_json::Value::as_str)
                == Some("steered"))
}

#[derive(Debug, Clone)]
struct PendingSteer {
    submission: UserMessageSubmission,
    target_response_id: Option<String>,
    server_steer_id: Option<String>,
}

fn steer_details(
    value: &serde_json::Value,
) -> Result<(&serde_json::Value, String, Option<String>), ModelFailure> {
    let steer = value.get("steer").ok_or_else(|| {
        ModelFailure::new(FailurePhase::Decode, FailureKind::MalformedResponse)
            .with_code("steer_event_missing_steer")
    })?;
    let previous_response_id = steer
        .get("previous_response_id")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| {
            ModelFailure::new(FailurePhase::Decode, FailureKind::MalformedResponse)
                .with_code("steer_event_missing_previous_response_id")
        })?
        .to_owned();
    let server_steer_id = steer
        .get("id")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned);
    Ok((steer, previous_response_id, server_steer_id))
}

fn bind_accepted_steer(
    pending_steers: &mut [PendingSteer],
    value: &serde_json::Value,
) -> Result<(), ModelFailure> {
    let (_, previous_response_id, server_steer_id) = steer_details(value)?;
    let server_steer_id = server_steer_id.ok_or_else(|| {
        ModelFailure::new(FailurePhase::Decode, FailureKind::MalformedResponse)
            .with_code("steer_accepted_missing_id")
    })?;
    let pending = pending_steers
        .iter_mut()
        .find(|pending| {
            (pending.server_steer_id.is_none()
                || pending.server_steer_id.as_deref() == Some(server_steer_id.as_str()))
                && pending.target_response_id.as_deref() == Some(previous_response_id.as_str())
        })
        .ok_or_else(|| {
            ModelFailure::new(FailurePhase::Decode, FailureKind::MalformedResponse)
                .with_code("steer_accepted_without_pending_submission")
        })?;
    pending.server_steer_id = Some(server_steer_id);
    Ok(())
}

fn take_pending_steer(
    pending_steers: &mut Vec<PendingSteer>,
    value: &serde_json::Value,
) -> Result<UserMessageSubmission, ModelFailure> {
    let (_, previous_response_id, server_steer_id) = steer_details(value)?;
    let event_type = value
        .get("type")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default();
    if event_type == "response.steer.failed"
        && value
            .get("steer")
            .and_then(|steer| steer.get("input"))
            .is_none()
    {
        return Err(
            ModelFailure::new(FailurePhase::Decode, FailureKind::MalformedResponse)
                .with_code("steer_failed_missing_input"),
        );
    }
    if event_type != "response.steer.failed" && server_steer_id.is_none() {
        return Err(
            ModelFailure::new(FailurePhase::Decode, FailureKind::MalformedResponse)
                .with_code("steer_pending_missing_id"),
        );
    }
    let index = if let Some(server_steer_id) = server_steer_id {
        pending_steers
            .iter()
            .position(|pending| {
                pending.server_steer_id.as_deref() == Some(server_steer_id.as_str())
                    && pending.target_response_id.as_deref() == Some(previous_response_id.as_str())
            })
            .or_else(|| {
                pending_steers.iter().position(|pending| {
                    pending.target_response_id.as_deref() == Some(previous_response_id.as_str())
                })
            })
    } else {
        pending_steers.iter().position(|pending| {
            pending.target_response_id.as_deref() == Some(previous_response_id.as_str())
        })
    };
    let Some(index) = index else {
        return Err(
            ModelFailure::new(FailurePhase::Decode, FailureKind::MalformedResponse)
                .with_code("steer_outcome_without_pending_submission"),
        );
    };
    Ok(pending_steers.remove(index).submission)
}

struct ReceiverLease {
    slot: Arc<StdMutex<Option<tokio::sync::mpsc::UnboundedReceiver<ResponseSteerRequest>>>>,
    receiver: Option<tokio::sync::mpsc::UnboundedReceiver<ResponseSteerRequest>>,
}

impl ReceiverLease {
    fn take(
        slot: Arc<StdMutex<Option<tokio::sync::mpsc::UnboundedReceiver<ResponseSteerRequest>>>>,
    ) -> Self {
        let receiver = slot.lock().unwrap().take();
        Self { slot, receiver }
    }
}

impl Drop for ReceiverLease {
    fn drop(&mut self) {
        if let Some(receiver) = self.receiver.take() {
            let mut slot = self.slot.lock().unwrap();
            if slot.is_none() {
                *slot = Some(receiver);
            }
        }
    }
}

async fn take_steer_request(
    steer_receiver: &Arc<
        StdMutex<Option<tokio::sync::mpsc::UnboundedReceiver<ResponseSteerRequest>>>,
    >,
) -> Result<ResponseSteerRequest, ModelFailure> {
    let mut lease = ReceiverLease::take(steer_receiver.clone());
    lease
        .receiver
        .as_mut()
        .ok_or_else(|| {
            ModelFailure::new(FailurePhase::Finish, FailureKind::MalformedResponse)
                .with_code("steer_receiver_missing")
        })?
        .recv()
        .await
        .ok_or_else(|| {
            ModelFailure::new(FailurePhase::Finish, FailureKind::MalformedResponse)
                .with_code("steer_receiver_closed")
        })
}

fn websocket_response_stream(
    session: Arc<tokio::sync::Mutex<Option<crate::model_runtime::websocket::TurnLocalWsSession>>>,
    previous_response_id: Arc<tokio::sync::Mutex<Option<String>>>,
    force_full: Arc<tokio::sync::Mutex<bool>>,
    next_prompt_unit_start: Arc<tokio::sync::Mutex<Option<usize>>>,
    poisoned: Arc<AtomicBool>,
    force_http: Arc<AtomicBool>,
    cached_successor: Arc<tokio::sync::Mutex<Option<Vec<u8>>>>,
    binding: Arc<dyn super::ProtocolBinding>,
    steer_handle: Option<ResponseSteerHandle>,
    steer_receiver: Arc<
        StdMutex<Option<tokio::sync::mpsc::UnboundedReceiver<ResponseSteerRequest>>>,
    >,
    cached_first: Option<Vec<u8>>,
    secret: String,
) -> WebsocketResponseStream {
    let terminal = Arc::new(AtomicBool::new(false));
    let terminal_for_stream = terminal.clone();
    let steer_handle = steer_handle.map(Arc::new);
    let inner = futures_util::stream::unfold(
        (
            session,
            previous_response_id,
            force_full,
            next_prompt_unit_start,
            cached_successor,
            binding,
            steer_receiver,
            cached_first,
            None::<String>,
            Vec::<PendingSteer>::new(),
            false,
            false,
            secret,
        ),
        move |(
            session,
            previous_response_id,
            force_full,
            next_prompt_unit_start,
            cached_successor,
            binding,
            steer_receiver,
            mut cached_first,
            mut current_response_id,
            mut pending_steers,
            mut steer_accepted,
            ended,
            secret,
        )| {
            let terminal_for_stream = terminal_for_stream.clone();
            let force_http = force_http.clone();
            let steer_handle = steer_handle.clone();
            async move {
                if ended {
                    return None;
                }
                let session_handle = session.clone();
                let mut guard = session.lock().await;
                let socket = guard.as_mut()?;
                loop {
                    let mut lease = ReceiverLease::take(steer_receiver.clone());
                    let raw_event = if let Some(first) = cached_first.take() {
                        Ok(crate::model_runtime::websocket::WsReadEvent::Text(first))
                    } else if let Some(receiver) = lease.receiver.as_mut() {
                        socket
                            .next_text_or_control(receiver, &[secret.as_str()])
                            .await
                    } else {
                        socket
                            .next_text(&[secret.as_str()])
                            .await
                            .map(crate::model_runtime::websocket::WsReadEvent::Text)
                    };
                    drop(lease);
                    let raw = match raw_event {
                        Ok(crate::model_runtime::websocket::WsReadEvent::Steer(request)) => {
                            let response_id = current_response_id.clone();
                            pending_steers.push(PendingSteer {
                                submission: request.submission,
                                target_response_id: response_id.clone(),
                                server_steer_id: None,
                            });
                            if let (Some(response_id), Some(request)) =
                                (response_id, pending_steers.last().cloned())
                            {
                                let frame = match binding
                                    .websocket_steer_frame(&response_id, &request.submission)
                                {
                                    Ok(frame) => frame,
                                    Err(error) => {
                                        *guard = None;
                                        return Some((
                                            Err(unknown_steer_failure(error, true)),
                                            (
                                                session_handle.clone(),
                                                previous_response_id,
                                                force_full,
                                                next_prompt_unit_start,
                                                cached_successor,
                                                binding,
                                                steer_receiver,
                                                cached_first,
                                                current_response_id,
                                                pending_steers,
                                                steer_accepted,
                                                true,
                                                secret,
                                            ),
                                        ));
                                    }
                                };
                                if let Err(error) =
                                    socket.send_text(frame, &[secret.as_str()]).await
                                {
                                    *guard = None;
                                    return Some((
                                        Err(unknown_steer_failure(error, true)),
                                        (
                                            session_handle.clone(),
                                            previous_response_id,
                                            force_full,
                                            next_prompt_unit_start,
                                            cached_successor,
                                            binding,
                                            steer_receiver,
                                            cached_first,
                                            current_response_id,
                                            pending_steers,
                                            true,
                                            true,
                                            secret,
                                        ),
                                    ));
                                }
                            }
                            continue;
                        }
                        Ok(crate::model_runtime::websocket::WsReadEvent::Text(text)) => text,
                        Err(error) => {
                            *guard = None;
                            if error.code.as_deref() == Some("websocket_message_too_big") {
                                force_http.store(true, Ordering::Release);
                            }
                            if let Some(handle) = &steer_handle {
                                handle.disable();
                            }
                            let submitted = steer_accepted || !pending_steers.is_empty();
                            return Some((
                                Err(unknown_steer_failure(error, submitted)),
                                (
                                    session_handle.clone(),
                                    previous_response_id,
                                    force_full,
                                    next_prompt_unit_start,
                                    cached_successor,
                                    binding,
                                    steer_receiver,
                                    cached_first,
                                    current_response_id,
                                    pending_steers,
                                    steer_accepted,
                                    true,
                                    secret,
                                ),
                            ));
                        }
                    };
                    let value = serde_json::from_slice::<serde_json::Value>(&raw).ok();
                    let event_type = value
                        .as_ref()
                        .and_then(|v| v.get("type"))
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or_default();
                    if event_type == "response.created" {
                        if let Some(handle) = &steer_handle {
                            handle.response_created();
                        }
                        current_response_id = value
                            .as_ref()
                            .and_then(|v| v.get("response"))
                            .and_then(|v| v.get("id"))
                            .and_then(serde_json::Value::as_str)
                            .map(str::to_owned);
                        if let (Some(response_id), Some(request)) =
                            (current_response_id.as_deref(), pending_steers.last_mut())
                        {
                            if request.target_response_id.is_none() {
                                request.target_response_id = Some(response_id.to_owned());
                            }
                            let frame = match binding
                                .websocket_steer_frame(response_id, &request.submission)
                            {
                                Ok(frame) => frame,
                                Err(error) => {
                                    *guard = None;
                                    return Some((
                                        Err(unknown_steer_failure(error, true)),
                                        (
                                            session_handle.clone(),
                                            previous_response_id,
                                            force_full,
                                            next_prompt_unit_start,
                                            cached_successor,
                                            binding,
                                            steer_receiver,
                                            cached_first,
                                            current_response_id,
                                            pending_steers,
                                            steer_accepted,
                                            true,
                                            secret,
                                        ),
                                    ));
                                }
                            };
                            if let Err(error) = socket.send_text(frame, &[secret.as_str()]).await {
                                *guard = None;
                                return Some((
                                    Err(unknown_steer_failure(error, true)),
                                    (
                                        session_handle.clone(),
                                        previous_response_id,
                                        force_full,
                                        next_prompt_unit_start,
                                        cached_successor,
                                        binding,
                                        steer_receiver,
                                        cached_first,
                                        current_response_id,
                                        pending_steers,
                                        true,
                                        true,
                                        secret,
                                    ),
                                ));
                            }
                        }
                    }
                    if matches!(
                        event_type,
                        "response.steer.accepted" | "response.steer.pending"
                    ) {
                        let accepted = value.as_ref().ok_or_else(|| {
                            ModelFailure::new(FailurePhase::Decode, FailureKind::MalformedResponse)
                                .with_code("steer_accepted_malformed")
                        });
                        if let Err(error) = accepted
                            .and_then(|value| bind_accepted_steer(&mut pending_steers, value))
                        {
                            *guard = None;
                            return Some((
                                Err(error),
                                (
                                    session_handle,
                                    previous_response_id,
                                    force_full,
                                    next_prompt_unit_start,
                                    cached_successor,
                                    binding,
                                    steer_receiver,
                                    cached_first,
                                    current_response_id,
                                    pending_steers,
                                    steer_accepted,
                                    true,
                                    secret,
                                ),
                            ));
                        }
                        // Both acknowledgements retain server ownership. Only a
                        // successor response.created commits the submission.
                        steer_accepted = true;
                        continue;
                    }
                    let successful_terminal =
                        is_successful_steer_terminal(event_type, value.as_ref());
                    let is_failed_terminal =
                        event_type == "response.failed" || event_type == "error";
                    let is_terminal = matches!(
                        event_type,
                        "response.completed" | "response.incomplete" | "response.failed" | "error"
                    );
                    let terminal_claimed = is_terminal
                        && steer_handle
                            .as_ref()
                            .is_some_and(|handle| handle.begin_terminal());
                    let claimed_request_pending = terminal_claimed && pending_steers.is_empty();
                    if claimed_request_pending {
                        let request = match take_steer_request(&steer_receiver).await {
                            Ok(request) => request,
                            Err(error) => {
                                *guard = None;
                                return Some((
                                    Err(error),
                                    (
                                        session_handle.clone(),
                                        previous_response_id,
                                        force_full,
                                        next_prompt_unit_start,
                                        cached_successor,
                                        binding,
                                        steer_receiver,
                                        cached_first,
                                        current_response_id,
                                        pending_steers,
                                        steer_accepted,
                                        true,
                                        secret,
                                    ),
                                ));
                            }
                        };
                        let request = PendingSteer {
                            submission: request.submission,
                            target_response_id: current_response_id.clone(),
                            server_steer_id: None,
                        };
                        if is_failed_terminal {
                            let submission = request.submission.clone();
                            pending_steers.push(request);
                            if let Some(handle) = &steer_handle {
                                handle.steer_failed();
                            }
                            *previous_response_id.lock().await = None;
                            *force_full.lock().await = true;
                            *next_prompt_unit_start.lock().await = None;
                            *cached_successor.lock().await = None;
                            *guard = None;
                            let synthetic = serde_json::json!({
                                "type": "response.steer.failed",
                                "steer_submission_id": submission.id,
                                "input": submission.content,
                            })
                            .to_string()
                            .into_bytes();
                            terminal_for_stream.store(true, Ordering::Release);
                            return Some((
                                Ok(synthetic),
                                (
                                    session_handle,
                                    previous_response_id,
                                    force_full,
                                    next_prompt_unit_start,
                                    cached_successor,
                                    binding,
                                    steer_receiver,
                                    cached_first,
                                    current_response_id,
                                    pending_steers,
                                    steer_accepted,
                                    true,
                                    secret,
                                ),
                            ));
                        }
                        pending_steers.push(request);
                        let Some(response_id) = current_response_id.as_deref() else {
                            *guard = None;
                            return Some((
                                Err(unknown_steer_failure(
                                    ModelFailure::new(
                                        FailurePhase::Decode,
                                        FailureKind::MalformedResponse,
                                    )
                                    .with_code("steer_terminal_missing_response_id"),
                                    true,
                                )),
                                (
                                    session_handle,
                                    previous_response_id,
                                    force_full,
                                    next_prompt_unit_start,
                                    cached_successor,
                                    binding,
                                    steer_receiver,
                                    cached_first,
                                    current_response_id,
                                    pending_steers,
                                    steer_accepted,
                                    true,
                                    secret,
                                ),
                            ));
                        };
                        let request = pending_steers.last().expect("just pushed");
                        let frame =
                            match binding.websocket_steer_frame(response_id, &request.submission) {
                                Ok(frame) => frame,
                                Err(error) => {
                                    *guard = None;
                                    return Some((
                                        Err(unknown_steer_failure(error, true)),
                                        (
                                            session_handle,
                                            previous_response_id,
                                            force_full,
                                            next_prompt_unit_start,
                                            cached_successor,
                                            binding,
                                            steer_receiver,
                                            cached_first,
                                            current_response_id,
                                            pending_steers,
                                            steer_accepted,
                                            true,
                                            secret,
                                        ),
                                    ));
                                }
                            };
                        if let Err(error) = socket.send_text(frame, &[secret.as_str()]).await {
                            *guard = None;
                            return Some((
                                Err(unknown_steer_failure(error, true)),
                                (
                                    session_handle,
                                    previous_response_id,
                                    force_full,
                                    next_prompt_unit_start,
                                    cached_successor,
                                    binding,
                                    steer_receiver,
                                    cached_first,
                                    current_response_id,
                                    pending_steers,
                                    true,
                                    true,
                                    secret,
                                ),
                            ));
                        }
                    }
                    let terminal_response_id = successful_terminal
                        .then(|| {
                            value
                                .as_ref()
                                .and_then(|v| v.get("response"))
                                .and_then(|v| v.get("id"))
                                .and_then(serde_json::Value::as_str)
                                .map(str::to_owned)
                        })
                        .flatten();
                    if successful_terminal {
                        *previous_response_id.lock().await = terminal_response_id.clone();
                        *force_full.lock().await = terminal_response_id.is_none();
                        *next_prompt_unit_start.lock().await = None;
                    }
                    if event_type == "error"
                        && value
                            .as_ref()
                            .and_then(|v| v.get("error"))
                            .and_then(|v| v.get("code"))
                            .and_then(serde_json::Value::as_str)
                            .or_else(|| {
                                value
                                    .as_ref()
                                    .and_then(|v| v.get("code"))
                                    .and_then(serde_json::Value::as_str)
                            })
                            == Some("previous_response_not_found")
                    {
                        *previous_response_id.lock().await = None;
                        *force_full.lock().await = true;
                        *next_prompt_unit_start.lock().await = None;
                    }
                    if successful_terminal && (!pending_steers.is_empty() || steer_accepted) {
                        let original_terminal = raw.clone();
                        loop {
                            let mut lease = ReceiverLease::take(steer_receiver.clone());
                            let successor_event = if let Some(receiver) = lease.receiver.as_mut() {
                                socket
                                    .next_text_or_control(receiver, &[secret.as_str()])
                                    .await
                            } else {
                                socket
                                    .next_text(&[secret.as_str()])
                                    .await
                                    .map(crate::model_runtime::websocket::WsReadEvent::Text)
                            };
                            drop(lease);
                            let successor = match successor_event {
                                Ok(crate::model_runtime::websocket::WsReadEvent::Steer(
                                    request,
                                )) => {
                                    let response_id = current_response_id.clone();
                                    pending_steers.push(PendingSteer {
                                        submission: request.submission,
                                        target_response_id: response_id.clone(),
                                        server_steer_id: None,
                                    });
                                    if let (Some(response_id), Some(request)) =
                                        (response_id, pending_steers.last().cloned())
                                    {
                                        let frame = match binding.websocket_steer_frame(
                                            &response_id,
                                            &request.submission,
                                        ) {
                                            Ok(frame) => frame,
                                            Err(error) => {
                                                *guard = None;
                                                return Some((
                                                    Err(unknown_steer_failure(error, true)),
                                                    (
                                                        session_handle.clone(),
                                                        previous_response_id,
                                                        force_full,
                                                        next_prompt_unit_start,
                                                        cached_successor,
                                                        binding,
                                                        steer_receiver,
                                                        cached_first,
                                                        current_response_id,
                                                        pending_steers,
                                                        steer_accepted,
                                                        true,
                                                        secret,
                                                    ),
                                                ));
                                            }
                                        };
                                        if let Err(error) =
                                            socket.send_text(frame, &[secret.as_str()]).await
                                        {
                                            *guard = None;
                                            return Some((
                                                Err(unknown_steer_failure(error, true)),
                                                (
                                                    session_handle.clone(),
                                                    previous_response_id,
                                                    force_full,
                                                    next_prompt_unit_start,
                                                    cached_successor,
                                                    binding,
                                                    steer_receiver,
                                                    cached_first,
                                                    current_response_id,
                                                    pending_steers,
                                                    true,
                                                    true,
                                                    secret,
                                                ),
                                            ));
                                        }
                                    }
                                    continue;
                                }
                                Ok(crate::model_runtime::websocket::WsReadEvent::Text(
                                    successor,
                                )) => successor,
                                Err(error) => {
                                    *guard = None;
                                    return Some((
                                        Err(unknown_steer_failure(error, true)),
                                        (
                                            session_handle.clone(),
                                            previous_response_id,
                                            force_full,
                                            next_prompt_unit_start,
                                            cached_successor,
                                            binding,
                                            steer_receiver,
                                            cached_first,
                                            current_response_id,
                                            pending_steers,
                                            steer_accepted,
                                            true,
                                            secret,
                                        ),
                                    ));
                                }
                            };
                            let successor_value =
                                serde_json::from_slice::<serde_json::Value>(&successor).ok();
                            let successor_type = successor_value
                                .as_ref()
                                .and_then(|value| value.get("type"))
                                .and_then(serde_json::Value::as_str)
                                .unwrap_or_default();
                            if matches!(
                                successor_type,
                                "response.steer.accepted" | "response.steer.pending"
                            ) {
                                let accepted = successor_value.as_ref().ok_or_else(|| {
                                    ModelFailure::new(
                                        FailurePhase::Decode,
                                        FailureKind::MalformedResponse,
                                    )
                                    .with_code("steer_accepted_malformed")
                                });
                                if let Err(error) = accepted.and_then(|value| {
                                    bind_accepted_steer(&mut pending_steers, value)
                                }) {
                                    *guard = None;
                                    return Some((
                                        Err(error),
                                        (
                                            session_handle,
                                            previous_response_id,
                                            force_full,
                                            next_prompt_unit_start,
                                            cached_successor,
                                            binding,
                                            steer_receiver,
                                            cached_first,
                                            current_response_id,
                                            pending_steers,
                                            steer_accepted,
                                            true,
                                            secret,
                                        ),
                                    ));
                                }
                                steer_accepted = true;
                                continue;
                            }
                            if successor_type == "response.steer.failed" {
                                let submission = match successor_value
                                    .as_ref()
                                    .ok_or_else(|| {
                                        ModelFailure::new(
                                            FailurePhase::Decode,
                                            FailureKind::MalformedResponse,
                                        )
                                    })
                                    .and_then(|value| {
                                        take_pending_steer(&mut pending_steers, value)
                                    }) {
                                    Ok(submission) => submission,
                                    Err(error) => {
                                        *previous_response_id.lock().await = None;
                                        *force_full.lock().await = true;
                                        *next_prompt_unit_start.lock().await = None;
                                        *cached_successor.lock().await = None;
                                        *guard = None;
                                        return Some((
                                            Err(error),
                                            (
                                                session_handle.clone(),
                                                previous_response_id,
                                                force_full,
                                                next_prompt_unit_start,
                                                cached_successor,
                                                binding,
                                                steer_receiver,
                                                cached_first,
                                                current_response_id,
                                                pending_steers,
                                                steer_accepted,
                                                true,
                                                secret,
                                            ),
                                        ));
                                    }
                                };
                                if let Some(handle) = &steer_handle {
                                    handle.return_submission(submission);
                                    steer_accepted = false;
                                    break;
                                }
                                *previous_response_id.lock().await = None;
                                *force_full.lock().await = true;
                                *next_prompt_unit_start.lock().await = None;
                                *cached_successor.lock().await = None;
                                *guard = None;
                                let synthetic = serde_json::json!({
                                    "type": successor_type,
                                    "steer_submission_id": submission.id,
                                    "input": submission.content,
                                })
                                .to_string()
                                .into_bytes();
                                return Some((
                                    Ok(synthetic),
                                    (
                                        session_handle,
                                        previous_response_id,
                                        force_full,
                                        next_prompt_unit_start,
                                        cached_successor,
                                        binding,
                                        steer_receiver,
                                        cached_first,
                                        current_response_id,
                                        pending_steers,
                                        steer_accepted,
                                        true,
                                        secret,
                                    ),
                                ));
                            }
                            if successor_type == "response.failed" || successor_type == "error" {
                                *guard = None;
                                return Some((
                                    Err(unknown_steer_failure(
                                        ModelFailure::new(
                                            FailurePhase::Finish,
                                            FailureKind::MalformedResponse,
                                        )
                                        .with_detail(
                                            format!(
                                                "steer response terminated with {successor_type}"
                                            ),
                                        ),
                                        true,
                                    )),
                                    (
                                        session_handle,
                                        previous_response_id,
                                        force_full,
                                        next_prompt_unit_start,
                                        cached_successor,
                                        binding,
                                        steer_receiver,
                                        cached_first,
                                        current_response_id,
                                        pending_steers,
                                        steer_accepted,
                                        true,
                                        secret,
                                    ),
                                ));
                            }
                            if successor_type == "response.created" {
                                let commit = match pending_steers.pop() {
                                    Some(request) => request.submission,
                                    None => {
                                        *guard = None;
                                        return Some((
                                            Err(ModelFailure::new(
                                                FailurePhase::Decode,
                                                FailureKind::MalformedResponse,
                                            )
                                            .with_code(
                                                "steer_successor_without_pending_submission",
                                            )),
                                            (
                                                session_handle.clone(),
                                                previous_response_id,
                                                force_full,
                                                next_prompt_unit_start,
                                                cached_successor,
                                                binding,
                                                steer_receiver,
                                                cached_first,
                                                current_response_id,
                                                pending_steers,
                                                steer_accepted,
                                                true,
                                                secret,
                                            ),
                                        ));
                                    }
                                };
                                *cached_successor.lock().await = Some(successor);
                                if let Some(handle) = &steer_handle {
                                    handle.steer_committed();
                                }
                                let synthetic = serde_json::json!({
                                    "type": "response.steer.commit",
                                    "steer_submission_id": commit.id,
                                    "input": commit.content,
                                })
                                .to_string()
                                .into_bytes();
                                return Some((
                                    Ok(synthetic),
                                    (
                                        session_handle,
                                        previous_response_id,
                                        force_full,
                                        next_prompt_unit_start,
                                        cached_successor,
                                        binding,
                                        steer_receiver,
                                        Some(original_terminal),
                                        current_response_id,
                                        pending_steers,
                                        false,
                                        false,
                                        secret,
                                    ),
                                ));
                            }
                        }
                    }
                    if event_type == "response.steer.failed" {
                        if let Some(handle) = &steer_handle {
                            handle.steer_failed();
                        }
                        let submission = match value
                            .as_ref()
                            .ok_or_else(|| {
                                ModelFailure::new(
                                    FailurePhase::Decode,
                                    FailureKind::MalformedResponse,
                                )
                            })
                            .and_then(|value| take_pending_steer(&mut pending_steers, value))
                        {
                            Ok(submission) => submission,
                            Err(error) => {
                                *previous_response_id.lock().await = None;
                                *force_full.lock().await = true;
                                *next_prompt_unit_start.lock().await = None;
                                *cached_successor.lock().await = None;
                                *guard = None;
                                return Some((
                                    Err(error),
                                    (
                                        session_handle.clone(),
                                        previous_response_id,
                                        force_full,
                                        next_prompt_unit_start,
                                        cached_successor,
                                        binding,
                                        steer_receiver,
                                        cached_first,
                                        current_response_id,
                                        pending_steers,
                                        steer_accepted,
                                        true,
                                        secret,
                                    ),
                                ));
                            }
                        };
                        if let Some(handle) = &steer_handle {
                            handle.return_submission(submission);
                            steer_accepted = false;
                            continue;
                        }
                        *previous_response_id.lock().await = None;
                        *force_full.lock().await = true;
                        *next_prompt_unit_start.lock().await = None;
                        *cached_successor.lock().await = None;
                        *guard = None;
                        let synthetic = serde_json::json!({
                            "type": event_type,
                            "steer_submission_id": submission.id,
                            "input": submission.content,
                        })
                        .to_string()
                        .into_bytes();
                        terminal_for_stream.store(true, Ordering::Release);
                        return Some((
                            Ok(synthetic),
                            (
                                session_handle,
                                previous_response_id,
                                force_full,
                                next_prompt_unit_start,
                                cached_successor,
                                binding,
                                steer_receiver,
                                cached_first,
                                current_response_id,
                                pending_steers,
                                steer_accepted,
                                true,
                                secret,
                            ),
                        ));
                    }
                    if is_failed_terminal && (steer_accepted || !pending_steers.is_empty()) {
                        *guard = None;
                        return Some((
                            Err(unknown_steer_failure(
                                ModelFailure::new(
                                    FailurePhase::Finish,
                                    FailureKind::MalformedResponse,
                                )
                                .with_detail(format!(
                                    "steer response terminated with {event_type}"
                                )),
                                true,
                            )),
                            (
                                session_handle,
                                previous_response_id,
                                force_full,
                                next_prompt_unit_start,
                                cached_successor,
                                binding,
                                steer_receiver,
                                cached_first,
                                current_response_id,
                                pending_steers,
                                steer_accepted,
                                true,
                                secret,
                            ),
                        ));
                    }
                    if is_terminal {
                        terminal_for_stream.store(true, Ordering::Release);
                        if let Some(handle) = &steer_handle {
                            handle.terminal();
                        }
                    }
                    let ended = is_terminal;
                    let next = (
                        session_handle,
                        previous_response_id,
                        force_full,
                        next_prompt_unit_start,
                        cached_successor,
                        binding,
                        steer_receiver,
                        cached_first,
                        current_response_id,
                        pending_steers,
                        steer_accepted,
                        ended,
                        secret,
                    );
                    drop(guard);
                    return Some((Ok(raw), next));
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
            let mut failure = route
                .binding
                .decode_http_error(status, detail.as_bytes(), retry_hint)
                .unwrap_or_else(|| http_status_failure(status, retry_hint));
            if !detail.is_empty() && failure.detail().is_empty() {
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
        if self.failure.is_some()
            || (self.terminal.is_some()
                && !matches!(
                    event,
                    ModelEvent::SteerCommit { .. }
                        | ModelEvent::SteerPending { .. }
                        | ModelEvent::SteerFailed { .. }
                ))
        {
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
            ModelEvent::SteerCommit { .. }
            | ModelEvent::SteerPending { .. }
            | ModelEvent::SteerFailed { .. } => {}
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
        if self.events.iter().any(|event| {
            matches!(
                event,
                ModelEvent::SteerPending { .. } | ModelEvent::SteerFailed { .. }
            )
        }) {
            return Err(ModelAttemptFailure {
                failure: runtime_invalid("response steer was not committed"),
                partial: snapshot,
            });
        }
        if !snapshot.pending_tools.is_empty() {
            return Err(ModelAttemptFailure {
                failure: runtime_invalid("terminal contains incomplete tools"),
                partial: snapshot,
            });
        }
        let terminal = match terminal {
            TerminalStatus::Completed | TerminalStatus::Steered
                if !snapshot.completed_tools.is_empty() =>
            {
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
            TerminalStatus::Completed | TerminalStatus::ToolUse | TerminalStatus::Steered => {
                terminal
            }
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

    async fn before_tools(&mut self) -> Result<(), ModelFailure> {
        Ok(())
    }

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
                    if error.partial.events.iter().any(|event| {
                        matches!(
                            event,
                            ModelEvent::SteerPending { .. } | ModelEvent::SteerFailed { .. }
                        )
                    }) =>
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
                    iteration += 1;
                    continue;
                }
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
                if let Err(mut error) = driver.before_tools().await {
                    if let Err(cleanup) = driver.abort_iteration(&result.snapshot, &error).await {
                        error = failure_with_cleanup(error, cleanup);
                    }
                    return Err(error);
                }
                if let Err(mut error) = driver.execute_tools(&result.snapshot.completed_tools).await
                {
                    if let Err(cleanup) = driver.abort_iteration(&result.snapshot, &error).await {
                        error = failure_with_cleanup(error, cleanup);
                    }
                    return Err(error);
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
                let partial = result
                    .as_ref()
                    .map(|result| result.snapshot.clone())
                    .unwrap_or_else(|error| error.partial.clone());
                return Err(ModelAttemptFailure { failure, partial });
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

    #[derive(Default)]
    struct RecordingObserver {
        events: Vec<ModelEvent>,
    }

    #[async_trait]
    impl ModelEventObserver for RecordingObserver {
        async fn observe(&mut self, event: &ModelEvent) -> Result<(), ModelFailure> {
            self.events.push(event.clone());
            Ok(())
        }
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

    #[tokio::test]
    async fn websocket_completed_before_accepted_commits_before_terminal_and_uses_cached_successor()
    {
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let mut socket = accept_async(stream).await.unwrap();
            let Message::Text(_) = socket.next().await.unwrap().unwrap() else {
                panic!("expected initial response.create");
            };
            socket
                .send(Message::Text(
                    r#"{"type":"response.created","response":{"id":"resp-1","status":"in_progress"}}"#
                        .into(),
                ))
                .await
                .unwrap();
            let Message::Text(steer) = socket.next().await.unwrap().unwrap() else {
                panic!("expected response.steer");
            };
            let steer: serde_json::Value = serde_json::from_str(&steer).unwrap();
            assert_eq!(steer["type"], "response.steer");
            assert_eq!(steer["previous_response_id"], "resp-1");
            assert_eq!(steer["input"][0]["role"], "user");
            assert_eq!(steer["input"][0]["content"][0]["type"], "input_text");
            socket
                .send(Message::Text(
                    r#"{"type":"response.completed","response":{"id":"resp-1","status":"completed"}}"#.into(),
                ))
                .await
                .unwrap();
            socket
                .send(Message::Text(
                    r#"{"type":"response.steer.accepted","steer":{"id":"server-steer-1","previous_response_id":"resp-1"}}"#.into(),
                ))
                .await
                .unwrap();
            socket
                .send(Message::Text(
                    r#"{"type":"response.created","response":{"id":"resp-2","status":"in_progress"}}"#.into(),
                ))
                .await
                .unwrap();
            socket
                .send(Message::Text(
                    r#"{"type":"response.output_text.delta","delta":"successor"}"#.into(),
                ))
                .await
                .unwrap();
            socket
                .send(Message::Text(
                    r#"{"type":"response.completed","response":{"id":"resp-2","status":"completed"}}"#.into(),
                ))
                .await
                .unwrap();
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
strategy = "astra"
[providers.vendor.models.model.transport]
websocket = true
[providers.vendor.models.model.capabilities]
reasoning = true
tools = true
[providers.vendor.models.model.capabilities.generation]
reasoning = true
"#
        );
        let route = RuntimeConfig::from_toml(&config)
            .unwrap()
            .resolve(&ProtocolRegistry::builtins())
            .unwrap()
            .route("vendor", "model")
            .unwrap()
            .clone();
        let input = ModelRequestInput::new("model", vec![]);
        let (steer_tx, steer_rx) = tokio::sync::mpsc::unbounded_channel();
        let transport = Arc::new(TurnLocalResponsesTransport::new());
        transport.set_steer_receiver(steer_rx).await;
        steer_tx
            .send(ResponseSteerRequest {
                submission: UserMessageSubmission::new(
                    "steer-1",
                    crate::user_content::UserMessageContent::from("steer now"),
                ),
            })
            .unwrap();
        let runtime = ModelRuntime::new_responses_websocket(transport.clone());
        let mut first_observer = RecordingObserver::default();
        let first = runtime
            .execute_attempt_with(&route, &input, &mut first_observer)
            .await
            .expect("first attempt should complete without event-after-terminal");
        assert!(matches!(
            first.terminal,
            TerminalStatus::Completed | TerminalStatus::Steered
        ));
        let commit_index = first_observer
            .events
            .iter()
            .position(|event| matches!(event, ModelEvent::SteerCommit { .. }))
            .expect("commit event must be observed");
        let terminal_index = first_observer
            .events
            .iter()
            .position(|event| matches!(event, ModelEvent::Terminal { .. }))
            .expect("terminal event must be observed");
        assert!(
            commit_index < terminal_index,
            "commit must precede terminal"
        );
        assert!(!first_observer.events.iter().any(|event| matches!(
            event,
            ModelEvent::SteerPending { .. } | ModelEvent::SteerFailed { .. }
        )));

        let mut second_observer = RecordingObserver::default();
        let second = runtime
            .execute_attempt_with(&route, &input, &mut second_observer)
            .await
            .expect("cached successor attempt should complete");
        assert!(matches!(
            second.terminal,
            TerminalStatus::Completed | TerminalStatus::Steered
        ));
        assert!(
            second_observer.events.iter().any(
                |event| matches!(event, ModelEvent::TextDelta { text } if text == "successor")
            )
        );
        server.await.unwrap();
    }

    #[tokio::test]
    async fn websocket_claim_wins_terminal_and_consumes_queued_steer() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let address = listener.local_addr().unwrap();
        let (terminal_tx, terminal_rx) = tokio::sync::oneshot::channel();
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let mut socket = accept_async(stream).await.unwrap();
            let Message::Text(_) = socket.next().await.unwrap().unwrap() else {
                panic!("expected initial response.create");
            };
            socket
                .send(Message::Text(
                    r#"{"type":"response.created","response":{"id":"resp-1","status":"in_progress"}}"#.into(),
                ))
                .await
                .unwrap();
            terminal_rx.await.unwrap();
            socket
                .send(Message::Text(
                    r#"{"type":"response.completed","response":{"id":"resp-1","status":"completed"}}"#.into(),
                ))
                .await
                .unwrap();
            let Message::Text(steer) = socket.next().await.unwrap().unwrap() else {
                panic!("expected response.steer after terminal");
            };
            let steer: serde_json::Value = serde_json::from_str(&steer).unwrap();
            assert_eq!(steer["type"], "response.steer");
            assert_eq!(steer["previous_response_id"], "resp-1");
            socket
                .send(Message::Text(
                    r#"{"type":"response.steer.accepted","steer":{"id":"server-steer-1","previous_response_id":"resp-1"}}"#.into(),
                ))
                .await
                .unwrap();
            socket
                .send(Message::Text(
                    r#"{"type":"response.created","response":{"id":"resp-2","status":"in_progress"}}"#.into(),
                ))
                .await
                .unwrap();
            socket
                .send(Message::Text(
                    r#"{"type":"response.completed","response":{"id":"resp-2","status":"completed"}}"#.into(),
                ))
                .await
                .unwrap();
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
strategy = "astra"
[providers.vendor.models.model.transport]
websocket = true
[providers.vendor.models.model.capabilities]
reasoning = true
[providers.vendor.models.model.capabilities.generation]
reasoning = true
"#
        );
        let route = RuntimeConfig::from_toml(&config)
            .unwrap()
            .resolve(&ProtocolRegistry::builtins())
            .unwrap()
            .route("vendor", "model")
            .unwrap()
            .clone();
        let request = local_responses_request(&route);
        let (steer_tx, steer_rx) = tokio::sync::mpsc::unbounded_channel();
        let handle = ResponseSteerHandle::new(true);
        let transport = Arc::new(TurnLocalResponsesTransport::new());
        transport.set_steer_receiver(steer_rx).await;
        transport.set_steer_handle(handle.clone());
        let mut response = transport.send_prepared(&route, request).await.unwrap();
        assert!(response.next_chunk().await.unwrap().is_ok());
        assert!(handle.availability());
        assert_eq!(handle.try_claim(), ResponseSteerDecision::Accepted);
        steer_tx
            .send(ResponseSteerRequest {
                submission: UserMessageSubmission::new(
                    "race-steer",
                    crate::user_content::UserMessageContent::from("continue"),
                ),
            })
            .unwrap();
        terminal_tx.send(()).unwrap();
        let mut saw_commit = false;
        while let Some(chunk) = response.next_chunk().await {
            let chunk = chunk.unwrap();
            saw_commit |= String::from_utf8_lossy(&chunk).contains("response.steer.commit");
        }
        assert!(saw_commit, "claimed steer must commit after terminal");
        assert!(!handle.pending());
        server.await.unwrap();
    }

    #[tokio::test]
    async fn websocket_disconnect_after_steer_is_unknown_and_not_retryable() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let mut socket = accept_async(stream).await.unwrap();
            let _ = socket.next().await.unwrap();
            socket
                .send(Message::Text(
                    r#"{"type":"response.created","response":{"id":"resp-1","status":"in_progress"}}"#.into(),
                ))
                .await
                .unwrap();
            let _ = socket.next().await.unwrap();
            drop(socket);
        });
        let route = route_for_local_server(address);
        let request = local_responses_request(&route);
        let (steer_tx, steer_rx) = tokio::sync::mpsc::unbounded_channel();
        let transport = TurnLocalResponsesTransport::new();
        transport.set_steer_receiver(steer_rx).await;
        let mut response = transport.send_prepared(&route, request).await.unwrap();
        steer_tx
            .send(ResponseSteerRequest {
                submission: UserMessageSubmission::new(
                    "steer-unknown",
                    crate::user_content::UserMessageContent::from("continue"),
                ),
            })
            .unwrap();
        let mut failure = None;
        while let Some(chunk) = response.next_chunk().await {
            if let Err(error) = chunk {
                failure = Some(error);
                break;
            }
        }
        let failure = failure.expect("disconnect must surface an error");
        assert_eq!(
            failure.code.as_deref(),
            Some("websocket_steer_outcome_unknown")
        );
        assert_eq!(failure.retry_hint, RetryHint::Never);
        server.await.unwrap();
    }

    #[tokio::test]
    async fn websocket_failed_without_engine_returns_input_and_starts_full_create() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            for (index, outcome) in [(0usize, "failed"), (1usize, "failed")] {
                let (stream, _) = listener.accept().await.unwrap();
                let mut socket = accept_async(stream).await.unwrap();
                let Message::Text(create) = socket.next().await.unwrap().unwrap() else {
                    panic!("expected response.create");
                };
                let create: serde_json::Value = serde_json::from_str(&create).unwrap();
                assert_eq!(create["type"], "response.create");
                assert!(create.get("previous_response_id").is_none());
                socket
                    .send(Message::Text(
                        format!(
                            r#"{{"type":"response.created","response":{{"id":"resp-{index}","status":"in_progress"}}}}"#
                        )
                        .into(),
                    ))
                    .await
                    .unwrap();
                let Message::Text(steer) = socket.next().await.unwrap().unwrap() else {
                    panic!("expected response.steer");
                };
                let steer: serde_json::Value = serde_json::from_str(&steer).unwrap();
                assert_eq!(steer["type"], "response.steer");
                socket
                    .send(Message::Text(
                        format!(
                            r#"{{"type":"response.completed","response":{{"id":"resp-{index}","status":"completed"}}}}"#
                        )
                        .into(),
                    ))
                    .await
                    .unwrap();
                socket
                    .send(Message::Text(
                        format!(
                            r#"{{"type":"response.steer.{outcome}","steer":{{"id":"server-steer-{index}","previous_response_id":"resp-{index}","input":"continue"}}}}"#
                        )
                        .into(),
                    ))
                    .await
                    .unwrap();
            }
        });
        let route = route_for_local_server(address);
        let request = local_responses_request(&route);
        let (steer_tx, steer_rx) = tokio::sync::mpsc::unbounded_channel();
        let transport = TurnLocalResponsesTransport::new();
        transport.set_steer_receiver(steer_rx).await;
        for index in 0..2 {
            let expected_outcome = "failed";
            let response = transport
                .send_prepared(&route, request.clone())
                .await
                .unwrap();
            steer_tx
                .send(ResponseSteerRequest {
                    submission: UserMessageSubmission::new(
                        format!("steer-{index}"),
                        crate::user_content::UserMessageContent::from("continue"),
                    ),
                })
                .unwrap();
            let mut response = response;
            let mut saw_outcome = false;
            while let Some(chunk) = response.next_chunk().await {
                let chunk = chunk.unwrap();
                saw_outcome |= String::from_utf8_lossy(&chunk)
                    .contains(&format!("response.steer.{expected_outcome}"));
            }
            assert!(saw_outcome, "{index}: steer outcome must end the attempt");
            assert!(*transport.force_full.lock().await);
        }
        server.await.unwrap();
    }

    #[test]
    fn steer_terminal_classification_excludes_failed_and_unsteered_incomplete() {
        assert!(is_successful_steer_terminal("response.completed", None));
        assert!(is_successful_steer_terminal(
            "response.incomplete",
            Some(&serde_json::json!({
                "response": {"incomplete_details": {"reason": "steered"}}
            }))
        ));
        assert!(!is_successful_steer_terminal(
            "response.incomplete",
            Some(&serde_json::json!({
                "response": {"incomplete_details": {"reason": "cancelled"}}
            }))
        ));
        assert!(!is_successful_steer_terminal("response.failed", None));
        assert!(!is_successful_steer_terminal("error", None));
    }

    #[test]
    fn accepted_server_id_binds_and_pending_returns_local_submission() {
        let mut pending = vec![
            PendingSteer {
                submission: UserMessageSubmission::new(
                    "local-1",
                    crate::user_content::UserMessageContent::from("first"),
                ),
                target_response_id: Some("resp-1".into()),
                server_steer_id: None,
            },
            PendingSteer {
                submission: UserMessageSubmission::new(
                    "local-2",
                    crate::user_content::UserMessageContent::from("second"),
                ),
                target_response_id: Some("resp-1".into()),
                server_steer_id: None,
            },
        ];
        bind_accepted_steer(
            &mut pending,
            &serde_json::json!({
                "type": "response.steer.accepted",
                "steer": {"id": "server-2", "previous_response_id": "resp-1"}
            }),
        )
        .unwrap();
        let submission = take_pending_steer(
            &mut pending,
            &serde_json::json!({
                "type": "response.steer.pending",
                "steer": {"id": "server-2", "previous_response_id": "resp-1"}
            }),
        )
        .unwrap();
        assert_eq!(submission.id, "local-1");
        assert_eq!(pending[0].submission.id, "local-2");
        let failed_submission = take_pending_steer(
            &mut pending,
            &serde_json::json!({
                "type": "response.steer.failed",
                "steer": {
                    "previous_response_id": "resp-1",
                    "input": "second"
                }
            }),
        )
        .unwrap();
        assert_eq!(failed_submission.id, "local-2");
    }

    #[tokio::test]
    async fn steer_waiter_wakes_for_created_response_and_successor() {
        let handle = ResponseSteerHandle::new(true);
        tokio::time::timeout(Duration::from_secs(1), async {
            let waiting = handle.wait_available();
            let create = async {
                tokio::task::yield_now().await;
                handle.response_created();
            };
            tokio::join!(waiting, create);
            assert_eq!(handle.try_claim(), ResponseSteerDecision::Accepted);
            handle.steer_committed();
            let waiting = handle.wait_available();
            let create = async {
                tokio::task::yield_now().await;
                handle.response_created();
            };
            tokio::join!(waiting, create);
            assert_eq!(handle.try_claim(), ResponseSteerDecision::Accepted);
        })
        .await
        .unwrap();
    }

    #[test]
    fn steer_handle_allows_one_atomic_pending_claim_and_reopens_after_commit() {
        let handle = ResponseSteerHandle::new(true);
        assert_eq!(handle.try_claim(), ResponseSteerDecision::Deferred);
        handle.response_created();
        assert!(handle.availability());
        assert_eq!(handle.try_claim(), ResponseSteerDecision::Accepted);
        assert!(handle.pending());
        assert_eq!(handle.try_claim(), ResponseSteerDecision::Deferred);
        handle.steer_committed();
        assert!(!handle.pending());
        assert!(!handle.availability());
        handle.response_created();
        assert!(handle.availability());
        assert_eq!(handle.try_claim(), ResponseSteerDecision::Accepted);
    }

    #[test]
    fn steer_handle_disables_availability_when_forced_to_http() {
        let handle = ResponseSteerHandle::new(true);
        handle.response_created();
        handle.disable();
        assert!(!handle.availability());
        assert_eq!(handle.try_claim(), ResponseSteerDecision::Deferred);
    }

    #[test]
    fn steer_handle_terminal_and_claim_have_one_linearized_state() {
        let handle = ResponseSteerHandle::new(true);
        handle.response_created();
        assert_eq!(handle.try_claim(), ResponseSteerDecision::Accepted);
        assert!(handle.begin_terminal());
        assert!(handle.pending());
        assert_eq!(handle.try_claim(), ResponseSteerDecision::Deferred);
        handle.steer_failed();
        assert!(!handle.pending());
        assert!(!handle.availability());

        let handle = ResponseSteerHandle::new(true);
        handle.response_created();
        assert!(!handle.begin_terminal());
        assert_eq!(handle.try_claim(), ResponseSteerDecision::Deferred);
    }

    #[test]
    fn unknown_steer_outcome_is_not_retryable() {
        let failure = unknown_steer_failure(
            ModelFailure::new(FailurePhase::Transport, FailureKind::Http),
            true,
        );
        assert_eq!(
            failure.code.as_deref(),
            Some("websocket_steer_outcome_unknown")
        );
        assert_eq!(failure.retry_hint, RetryHint::Never);
    }

    async fn drain_response(mut response: TransportResponse) {
        while let Some(chunk) = response.next_chunk().await {
            chunk.unwrap();
        }
    }

    #[tokio::test]
    async fn websocket_incomplete_max_output_tokens_finishes_before_socket_close() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let mut socket = accept_async(stream).await.unwrap();
            let Message::Text(_) = socket.next().await.unwrap().unwrap() else {
                panic!("expected websocket request frame");
            };
            socket
                .send(Message::Text(
                    r#"{"type":"response.created","response":{"id":"resp-limit","status":"in_progress"}}"#.into(),
                ))
                .await
                .unwrap();
            socket
                .send(Message::Text(
                    r#"{"type":"response.incomplete","response":{"id":"resp-limit","status":"incomplete","incomplete_details":{"reason":"max_output_tokens"}}}"#.into(),
                ))
                .await
                .unwrap();
            tokio::time::sleep(Duration::from_millis(500)).await;
        });

        let route = route_for_local_server(address);
        let handle = ResponseSteerHandle::new(true);
        let transport = TurnLocalResponsesTransport::new();
        transport.set_steer_handle(handle.clone());
        let runtime = ModelRuntime::new_responses_websocket(Arc::new(transport));
        let failure = tokio::time::timeout(
            Duration::from_millis(100),
            runtime.execute_attempt(&route, &ModelRequestInput::new("model", Vec::new())),
        )
        .await
        .expect("incomplete response must not wait for websocket close")
        .expect_err("max_output_tokens is a non-success terminal");
        assert_eq!(
            failure.failure.code.as_deref(),
            Some("non_success_terminal")
        );
        assert_eq!(handle.try_claim(), ResponseSteerDecision::Deferred);
        server.await.unwrap();
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
    fn accumulator_keeps_reverse_tool_done_events_in_model_start_order() {
        let mut accumulator = AttemptAccumulator::default();
        for (id, name) in [("a", "first"), ("b", "second")] {
            accumulator
                .consume_observed(ModelEvent::ToolStarted {
                    id: id.into(),
                    name: name.into(),
                })
                .unwrap();
        }
        for (id, name) in [("b", "second"), ("a", "first")] {
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
                status: TerminalStatus::ToolUse,
            })
            .unwrap();
        let result = accumulator.finish().unwrap();
        assert_eq!(
            result
                .snapshot
                .completed_tools
                .iter()
                .map(|tool| tool.id.as_str())
                .collect::<Vec<_>>(),
            vec!["a", "b"]
        );
    }

    #[test]
    fn accumulator_normalizes_steered_with_tools_to_tool_use() {
        let mut accumulator = AttemptAccumulator::default();
        accumulator
            .consume_observed(ModelEvent::ToolStarted {
                id: "call".into(),
                name: "search".into(),
            })
            .unwrap();
        accumulator
            .consume_observed(ModelEvent::ToolDone {
                id: "call".into(),
                name: "search".into(),
                arguments: serde_json::json!({}),
            })
            .unwrap();
        accumulator
            .consume_observed(ModelEvent::Terminal {
                status: TerminalStatus::Steered,
            })
            .unwrap();
        let result = accumulator.finish().unwrap();
        assert_eq!(result.terminal, TerminalStatus::ToolUse);
        assert_eq!(result.snapshot.completed_tools.len(), 1);
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
