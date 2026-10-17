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
        None => Ok(
            b"data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\"}}\n\n"
                .to_vec(),
        ),
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

fn anthropic_stream_body(terminal: bool) -> String {
    let mut body = String::from(concat!(
        "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":3}}}\n\n",
        "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
        "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n",
    ));
    if terminal {
        body.push_str(concat!(
                "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
                "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":1}}\n\n",
                "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
            ));
    }
    body
}

fn unterminated_chunk_close() -> ModelFailure {
    ModelFailure::new(FailurePhase::Transport, FailureKind::Http)
        .with_status(200)
        .with_code("response_chunk_failed")
        .with_retry_hint(RetryHint::Retryable)
        .with_detail("error decoding response body")
}

fn anthropic_stream_attempt(terminal: bool) -> Arc<QueueTransport> {
    Arc::new(QueueTransport {
        responses: Mutex::new(VecDeque::from([Ok(TransportResponse::from_results(
            200,
            BTreeMap::new(),
            vec![
                Ok(anthropic_stream_body(terminal).into_bytes()),
                Err(unterminated_chunk_close()),
            ],
        ))])),
    })
}

async fn drain_response(mut response: TransportResponse) {
    while let Some(chunk) = response.next_chunk().await {
        chunk.unwrap();
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

    async fn persist_assistant(&mut self, assistant: &ModelMessage) -> Result<(), ModelFailure> {
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
fn gpt_overload_uses_fixed_interval_without_exponential_growth() {
    let mut route = route("responses");
    route.model = "gpt-5.5".into();
    let config = RuntimeRetryConfig {
        enabled: true,
        max_attempts: 5,
        max_recovery_attempts: 0,
        initial_delay_secs: 2,
        exponential_backoff: true,
        backoff_multiplier: 2.0,
        jitter_secs: 1,
    };
    let failure = ModelFailure::new(FailurePhase::Decode, FailureKind::Http)
        .with_code("server_is_overloaded")
        .with_retry_hint(RetryHint::Retryable);
    assert_eq!(
        overload_fixed_retry_delay(&route, &config, &failure),
        Some(Duration::from_secs(2))
    );
    route.model = "claude-4".into();
    assert_eq!(overload_fixed_retry_delay(&route, &config, &failure), None);
}

#[test]
fn resolved_route_carries_effective_retry_configuration() {
    let route = route("responses");
    assert_eq!(route.retry.as_ref().unwrap().max_attempts, 3);
}

mod integration_tests {
    use super::*;

    #[tokio::test]
    async fn completed_stream_survives_an_unterminated_chunked_close() {
        let route = route("anthropic");
        let runtime = ModelRuntime::new(anthropic_stream_attempt(true));
        let result = runtime
            .execute_attempt(&route, &ModelRequestInput::new("model", Vec::new()))
            .await
            .expect("a completed stream must not fail on the chunked close");
        assert_eq!(result.terminal, TerminalStatus::Completed);
        assert!(result.snapshot.side_effects.text);
    }

    #[tokio::test]
    async fn stream_without_terminal_still_fails_on_the_chunk_error() {
        let route = route("anthropic");
        let runtime = ModelRuntime::new(anthropic_stream_attempt(false));
        let failure = runtime
            .execute_attempt(&route, &ModelRequestInput::new("model", Vec::new()))
            .await
            .expect_err("a stream without message_stop must still fail");
        assert_eq!(
            failure.failure.code.as_deref(),
            Some("response_chunk_failed")
        );
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
                ModelMessage::text(crate::model_runtime::MessageRole::User, "assistant history"),
                ModelMessage::text(crate::model_runtime::MessageRole::User, "tool output"),
            ],
        );
        input.segments = vec![crate::model_runtime::ControlSegment::system(
            "system instructions",
        )];
        input.segment_origins = vec!["system".into()];
        input.message_origins = vec!["history".into(), "tool".into()];
        input.tools = vec![crate::model_runtime::ToolDefinition::new(
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
            vec![ModelMessage::text(
                crate::model_runtime::MessageRole::User,
                "hello",
            )],
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
        for (drop_partial, close_idle) in [(false, false), (true, false), (false, true)] {
            tokio::time::timeout(Duration::from_secs(5), async {
                let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
                let route = route_for_local_server(listener.local_addr().unwrap());
                let (idle_tx, idle_rx) = tokio::sync::oneshot::channel();
                let (pong_tx, pong_rx) = tokio::sync::oneshot::channel();
                let (close_tx, close_rx) = tokio::sync::oneshot::channel();
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
                    if close_idle {
                        idle_rx.await.unwrap();
                        socket.send(Message::Ping(vec![1, 2, 3].into())).await.unwrap();
                        assert_eq!(socket.next().await.unwrap().unwrap(), Message::Pong(vec![1, 2, 3].into()));
                        pong_tx.send(()).unwrap();
                        close_rx.await.unwrap();
                        socket.send(Message::Close(None)).await.unwrap();
                    } else {
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
                    if close_idle {
                        idle_tx.send(()).unwrap();
                        pong_rx.await.unwrap();
                        close_tx.send(()).unwrap();
                        loop {
                            if transport.session.lock().await.as_ref().unwrap().is_closed() {
                                break;
                            }
                            tokio::task::yield_now().await;
                        }
                    } else {
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
                    }
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
}
