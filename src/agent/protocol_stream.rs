use super::*;
use crate::langfuse_trace;
use crate::user_content::UserMessageContent;
use tracing::Instrument;

const STREAM_INTERRUPT_MESSAGE: &str = "Model stream interrupted";
const STREAM_INTERRUPT_ACTION: &str = "Continuing with a fresh model iteration";

fn llm_request_telemetry(
    logical_request_id: &str,
    turn_id: u64,
    iteration: usize,
    attempt: usize,
    model: &str,
    protocol: ApiProtocol,
    build: &crate::request_builder::BuildResult,
    tool_call_count_before: usize,
    tool_definitions_count: usize,
    observation: AdjacentRequestObservation,
) -> LlmRequestTelemetry {
    LlmRequestTelemetry::prepared_from_build(
        logical_request_id.into(),
        turn_id,
        iteration,
        attempt,
        model.into(),
        protocol,
        build,
        tool_call_count_before,
        tool_definitions_count,
        observation,
    )
}

async fn emit_attempt_terminal<E, Efut>(
    error_class: LlmRequestErrorClass,
    prepared: &LlmRequestTelemetry,
    iteration_span: &tracing::Span,
    on_event: &mut E,
) -> Result<()>
where
    E: FnMut(AgentEvent) -> Efut,
    Efut: Future<Output = Result<()>>,
{
    let telemetry = prepared.failed(error_class);
    on_event(AgentEvent::LlmRequestTelemetry(telemetry.clone())).await?;
    langfuse_trace::record_llm_request_telemetry(iteration_span, &telemetry);
    Ok(())
}

async fn emit_attempt_interrupted<E, Efut>(
    error_class: LlmRequestErrorClass,
    prepared: &LlmRequestTelemetry,
    iteration_span: &tracing::Span,
    on_event: &mut E,
) -> Result<()>
where
    E: FnMut(AgentEvent) -> Efut,
    Efut: Future<Output = Result<()>>,
{
    let telemetry = prepared.interrupted(error_class);
    on_event(AgentEvent::LlmRequestTelemetry(telemetry.clone())).await?;
    langfuse_trace::record_llm_request_telemetry(iteration_span, &telemetry);
    Ok(())
}

enum ResponseStreamRequest {
    Typed(async_openai::types::responses::CreateResponse),
    Compatible(Value),
}

enum CompletionStreamRequest {
    Typed(async_openai::types::chat::CreateChatCompletionRequest),
    Compatible(Value),
}

#[derive(Debug)]
pub(super) enum ChatStreamCreationError {
    Setup(String),
    Transport(reqwest::Error),
    Status {
        status: reqwest::StatusCode,
        message: String,
    },
}

impl std::fmt::Display for ChatStreamCreationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Setup(error) => write!(f, "failed to create streamed chat completion: {error}"),
            Self::Transport(error) => {
                write!(f, "failed to create streamed chat completion: {error}")
            }
            Self::Status { status, message } => {
                write!(
                    f,
                    "chat completions request failed with status {status}: {message}"
                )
            }
        }
    }
}

impl std::error::Error for ChatStreamCreationError {}

fn should_retry_chat_stream_creation(
    config: &crate::config::RetryConfig,
    attempt: usize,
    error: &ChatStreamCreationError,
) -> bool {
    match error {
        ChatStreamCreationError::Setup(_) => false,
        ChatStreamCreationError::Transport(error) => {
            should_retry_reqwest_error(config, attempt, error)
        }
        ChatStreamCreationError::Status { status, .. } => {
            should_retry_http_status(config, attempt, *status)
        }
    }
}

async fn create_response_stream<C: Config>(
    client: &Client<C>,
    request: &ResponseStreamRequest,
) -> Result<async_openai::types::stream::StreamResponse<Value>, OpenAIError> {
    match request {
        ResponseStreamRequest::Typed(request) => {
            client.responses().create_stream_byot(request.clone()).await
        }
        ResponseStreamRequest::Compatible(request) => {
            client.responses().create_stream_byot(request.clone()).await
        }
    }
}

/// Ignores the one validated provider side-band event and otherwise preserves
/// the payload for strict SDK deserialization.
pub(super) fn project_response_stream_event(
    raw: &Value,
) -> std::result::Result<Option<ResponseStreamEvent>, serde_json::Error> {
    let Some(projected) = project_response_stream_event_value(raw)? else {
        return Ok(None);
    };
    serde_json::from_value(projected).map(Some)
}

fn project_response_stream_event_value(
    raw: &Value,
) -> std::result::Result<Option<Value>, serde_json::Error> {
    if raw.get("type").and_then(Value::as_str) == Some("response.metadata") {
        let valid_extension = raw.get("response_id").is_some_and(Value::is_string)
            && raw
                .get("sequence_number")
                .is_some_and(|value| value.as_u64().is_some())
            && raw.get("metadata").is_some_and(Value::is_object);
        if valid_extension {
            return Ok(None);
        }
        return Err(serde_json::Error::io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "malformed response.metadata stream event",
        )));
    }

    let mut projected = raw.clone();
    if let Some(response) = projected.get_mut("response").and_then(Value::as_object_mut) {
        response.remove("reasoning");
    }
    Ok(Some(projected))
}

pub(super) fn is_ignorable_response_lifecycle_event(raw: &Value) -> bool {
    matches!(
        raw.get("type").and_then(Value::as_str),
        Some("response.created" | "response.in_progress")
    ) && raw
        .get("response")
        .and_then(Value::as_object)
        .is_some_and(|response| !response.contains_key("model"))
}

async fn emit_prepared_request_metadata<E, Efut>(
    build: &crate::request_builder::BuildResult,
    on_event: &mut E,
) -> Result<()>
where
    E: FnMut(AgentEvent) -> Efut,
    Efut: Future<Output = Result<()>>,
{
    let cache_report = CacheUsageReport::from_build(build);
    on_event(AgentEvent::TokenUsageUpdated {
        used_tokens: build.budget.estimated_request_tokens,
        context_window_tokens: build.budget.context_window_tokens,
        input_tokens: build.budget.estimated_request_tokens,
        output_tokens: 0,
        cached_tokens: 0,
        cache_report: Some(cache_report.clone()),
    })
    .await?;
    if build.budget.truncated {
        debug!(
            original_history_items = build.budget.original_history_items,
            retained_history_items = build.budget.retained_history_items,
            dropped_history_items = build.budget.dropped_history_items,
            context_window_tokens = build.budget.context_window_tokens,
            input_budget_tokens = build.budget.input_budget_tokens,
            estimated_request_tokens = build.budget.estimated_request_tokens,
            prompt_segments = build.prompt_plan.segments.len(),
            prompt_stable_prefix_hash = build.prompt_plan.stable_prefix_hash(),
            "request history truncated to fit budget"
        );
    }
    Ok(())
}

/// Prepares the sole provider request for an outer LLM iteration.
async fn prepare_protocol_stream_request<C, E, Efut>(
    agent: &mut Agent<C>,
    protocol: ApiProtocol,
    turn_prelude: &[PromptMessage],
    protected_start_index: &mut usize,
    tool_definitions: &[crate::request_builder::ToolSpec],
    on_event: &mut E,
) -> Result<crate::agent::compaction::PreparedRequestBuild>
where
    C: Config + Clone,
    E: FnMut(AgentEvent) -> Efut + Send,
    Efut: Future<Output = Result<()>> + Send,
{
    prepare_canonical_protocol_stream_request(
        agent,
        protocol,
        turn_prelude,
        protected_start_index,
        tool_definitions,
        on_event,
    )
    .await
}

/// Canonical preparation path with deterministic preflight and checkpoint admission.
async fn prepare_canonical_protocol_stream_request<C, E, Efut>(
    agent: &mut Agent<C>,
    protocol: ApiProtocol,
    turn_prelude: &[PromptMessage],
    protected_start_index: &mut usize,
    tool_definitions: &[crate::request_builder::ToolSpec],
    on_event: &mut E,
) -> Result<crate::agent::compaction::PreparedRequestBuild>
where
    C: Config + Clone,
    E: FnMut(AgentEvent) -> Efut + Send,
    Efut: Future<Output = Result<()>> + Send,
{
    let prepared = match compaction::prepare_request_build(
        agent,
        protocol,
        turn_prelude,
        *protected_start_index,
        tool_definitions,
        on_event,
    )
    .await
    {
        Ok(prepared) => prepared,
        Err(error) if compaction::is_recognized_request_budget_overflow(&error) => {
            return compact_for_request_pressure(
                agent,
                protocol,
                turn_prelude,
                protected_start_index,
                tool_definitions,
                on_event,
            )
            .await;
        }
        Err(error) => return Err(error),
    };
    if prepared.build.budget.request_classification().safe {
        return Ok(prepared);
    }
    compact_for_request_pressure(
        agent,
        protocol,
        turn_prelude,
        protected_start_index,
        tool_definitions,
        on_event,
    )
    .await
}

#[cfg(test)]
pub(super) async fn prepare_canonical_protocol_stream_request_for_test<C, E, Efut>(
    agent: &mut Agent<C>,
    protocol: ApiProtocol,
    turn_prelude: &[PromptMessage],
    protected_start_index: &mut usize,
    tool_definitions: &[crate::request_builder::ToolSpec],
    on_event: &mut E,
) -> Result<crate::agent::compaction::PreparedRequestBuild>
where
    C: Config + Clone,
    E: FnMut(AgentEvent) -> Efut + Send,
    Efut: Future<Output = Result<()>> + Send,
{
    prepare_canonical_protocol_stream_request(
        agent,
        protocol,
        turn_prelude,
        protected_start_index,
        tool_definitions,
        on_event,
    )
    .await
}

async fn compact_for_request_pressure<C, E, Efut>(
    agent: &mut Agent<C>,
    protocol: ApiProtocol,
    turn_prelude: &[PromptMessage],
    protected_start_index: &mut usize,
    tool_definitions: &[crate::request_builder::ToolSpec],
    on_event: &mut E,
) -> Result<crate::agent::compaction::PreparedRequestBuild>
where
    C: Config + Clone,
    E: FnMut(AgentEvent) -> Efut + Send,
    Efut: Future<Output = Result<()>> + Send,
{
    let frontier = PressureCompactionFrontier {
        frame_count: agent.protocol_frames.len(),
        protocol_prefix_digest: protocol_prefix_digest(&agent.protocol_frames),
    };
    agent.turn.pressure_compaction.mark_attempted(frontier)?;
    // Consume the frontier before every fallible validation or selection step.
    // `compact_for_request_pressure` emits Started before validating so a
    // malformed protocol receives the same failed lifecycle terminal.
    let successor = compaction::compact_for_request_pressure(
        agent,
        protocol,
        turn_prelude,
        tool_definitions,
        on_event,
    )
    .await?;
    *protected_start_index = successor.protected_start_index;
    // The compaction transaction preflighted this exact cold successor before
    // acknowledging and committing its candidate; only invariant checks remain.
    let successor_preview = &successor.epoch_preview;
    ensure!(
        matches!(successor_preview.transition, ActiveEpochTransition::Cold),
        "pressure compaction successor did not start a cold epoch"
    );
    let classification = successor.build.budget.request_classification();
    ensure!(
        classification.safe,
        "pressure compaction successor remains unsafe (request {}, watermark {}, hard limit {}, truncated {})",
        successor.build.budget.estimated_request_tokens,
        classification.high_watermark,
        classification.hard_request_limit,
        successor.build.budget.truncated,
    );
    Ok(successor)
}
pub(super) async fn run_responses_stream_async<C, F, E, A, Dfut, Efut, Afut>(
    agent: &mut Agent<C>,
    user_content: UserMessageContent,
    user_input: &str,
    mut on_delta: F,
    mut on_event: E,
    mut approve: A,
) -> Result<String>
where
    C: Config + Clone,
    F: FnMut(&str) -> Dfut,
    E: FnMut(AgentEvent) -> Efut + Send,
    A: FnMut(PermissionRequest) -> Afut,
    Dfut: Future<Output = Result<()>>,
    Efut: Future<Output = Result<()>> + Send,
    Afut: Future<Output = Result<PermissionApproval>>,
{
    let turn_prelude = agent.try_prepare_turn_prelude(user_input)?;
    let mut protected_start_index = agent.history.len();
    let previous_turn_start_index = agent.turn.current_turn_start_index;
    agent.turn.current_turn_start_index = Some(protected_start_index);
    if let Err(error) = agent.append_history_item(HistoryItem::user_content(user_content)) {
        agent.turn.current_turn_start_index = previous_turn_start_index;
        return Err(error);
    }
    Agent::<C>::emit_audit_event(
        &mut on_event,
        AgentEvent::TurnStarted(agent.turn_started_event()),
        "turn_started",
    )
    .await;
    debug!(
        user_input_len = user_input.len(),
        history_len = agent.history.len(),
        "user message added to history"
    );

    let turn_id = agent.turn.turn_id;
    let turn_span = langfuse_trace::llm_turn_span(
        turn_id,
        "responses",
        &agent.model,
        agent.max_iterations,
        agent.max_tool_calls,
        user_input.chars().count(),
        agent.history.len(),
    );
    let mut final_text = String::new();
    let mut tool_call_count = 0;
    let mut continuation_count = 0;

    let result = async {
        let mut iteration_count = 0;
        'agent_iteration: loop {
            ensure_iteration_budget(agent.max_iterations, iteration_count)?;
            let iteration = iteration_count;
            iteration_count += 1;
        debug!(
            iteration,
            model = %agent.model,
            history_len = agent.history.len(),
            tool_call_count,
            max_tool_calls = agent.max_tool_calls,
            "creating streamed response"
        );

        let tool_definitions = agent.tool_definitions();
        let prepared = prepare_protocol_stream_request(
            agent,
            ApiProtocol::Responses,
            &turn_prelude,
            &mut protected_start_index,
            &tool_definitions,
            &mut on_event,
        )
        .await?;
        protected_start_index = prepared.protected_start_index;
        let epoch_preview = prepared.epoch_preview;
        let build = prepared.build;
        if agent.turn.frozen_evidence.is_none() {
            agent.turn.frozen_evidence = Some(FrozenTurnEvidence {
                message: build.selected_evidence_message.clone(),
                selected_ids: build.selected_evidence_ids.clone(),
            });
        }
        let logical_observation = agent.preview_final_logical_request(&build);
        let cache_report = CacheUsageReport::from_build(&build);
        let iteration_span = langfuse_trace::llm_iteration_span(
            turn_id,
            "responses",
            &agent.model,
            iteration,
            build.budget.retained_history_items,
            tool_call_count,
            tool_definitions.len(),
        );
        emit_prepared_request_metadata(&build, &mut on_event).await?;
        let logical_request_id = format!("turn-{turn_id}-iteration-{iteration}");

        let response_request = match build.request.clone() {
            BuiltRequest::Responses(request) => ResponseStreamRequest::Typed(request),
            BuiltRequest::ResponsesCompatible(request) => ResponseStreamRequest::Compatible(request),
            BuiltRequest::Completions(_) | BuiltRequest::CompletionsCompatible(_) => {
                return Err(anyhow!("request builder returned non-responses request"));
            }
        };

        let mut attempt = 1;
        let (response, mut turn_text, completed_reasoning_ids, prepared_telemetry) = 'retry_response_stream: loop {
            let mut prepared_telemetry = llm_request_telemetry(
                &logical_request_id, turn_id, iteration, attempt, &agent.model,
                ApiProtocol::Responses, &build, tool_call_count, tool_definitions.len(), logical_observation,
            );
            if attempt > 1 {
                // Adjacent LCP is a logical-request scalar, persisted only by
                // the physical attempt that establishes this baseline.
                prepared_telemetry.adjacent_lcp_units = None;
                prepared_telemetry.adjacent_lcp_bytes = None;
                prepared_telemetry.adjacent_lcp_estimated_tokens = None;
                prepared_telemetry.first_breaker = None;
            }
            on_event(AgentEvent::LlmRequestTelemetry(prepared_telemetry.clone())).await?;
            langfuse_trace::record_llm_request_telemetry(&iteration_span, &prepared_telemetry);
            if attempt == 1 {
                // Commit only after all prepared telemetry callbacks succeed,
                // immediately before the first physical transport send.
                agent.commit_final_logical_request(&build);
                agent.commit_active_epoch(epoch_preview.clone());
            }
            let mut stream = match create_response_stream(
                &agent.client,
                &response_request,
            )
            .await
            {
                Ok(stream) => stream,
                Err(error)
                    if should_retry_openai_stream_creation(
                        &agent.retry_config,
                        attempt,
                        &error,
                    ) =>
                {
                    let delay = retry_delay(&agent.retry_config, attempt);
                    warn!(
                        attempt,
                        max_attempts = agent.retry_config.max_attempts,
                        delay_ms = delay.as_millis(),
                        error = %error,
                        "retrying streamed response creation"
                    );
                    emit_attempt_terminal(
                        LlmRequestErrorClass::RequestCreation,
                        &prepared_telemetry,
                        &iteration_span,
                        &mut on_event,
                    )
                    .await?;
                    tokio::time::sleep(delay).await;
                    attempt += 1;
                    continue 'retry_response_stream;
                }
                Err(error) => {
                    emit_attempt_terminal(
                        LlmRequestErrorClass::RequestCreation,
                        &prepared_telemetry,
                        &iteration_span,
                        &mut on_event,
                    )
                    .await?;
                    return Err(anyhow!(error).context(request_creation_failure_context(
                        "streamed response",
                        &agent.model,
                        agent.active_model_metadata(),
                        &build.budget,
                    )));
                }
            };

            let mut completed_response: Option<Response> = None;
            let mut completed_reasoning_ids = HashSet::new();
            let mut emitted_pending_tool_calls = HashSet::new();
            let mut pending_tool_calls = BTreeMap::new();
            let mut turn_text = String::new();
            let mut stream_had_side_effect = false;

            while let Some(event) = stream.next().await {
                let raw = match event {
                    Ok(event) => event,
                    Err(error)
                        if !stream_had_side_effect
                            && should_retry_openai_stream_read(
                                &agent.retry_config,
                                attempt,
                                &error,
                            ) =>
                    {
                        let delay = retry_delay(&agent.retry_config, attempt);
                        warn!(
                            attempt,
                            max_attempts = agent.retry_config.max_attempts,
                            delay_ms = delay.as_millis(),
                            error = %error,
                            "retrying streamed response read before side effects"
                        );
                        emit_attempt_terminal(LlmRequestErrorClass::StreamRead, &prepared_telemetry, &iteration_span, &mut on_event).await?;
                        tokio::time::sleep(delay).await;
                        attempt += 1;
                        continue 'retry_response_stream;
                    }
                    Err(error) if stream_had_side_effect => {
                        warn!(
                            protocol = "responses",
                            phase = "stream_read",
                            error = %error,
                            text_len = turn_text.len(),
                            tool_count = pending_tool_calls.len(),
                            "recovering interrupted responses stream after side effects"
                        );
                        emit_attempt_interrupted(LlmRequestErrorClass::StreamRead, &prepared_telemetry, &iteration_span, &mut on_event).await?;
                        recover_stream_interrupt(
                            agent,
                            &turn_text,
                            &pending_tool_calls,
                            "responses",
                            "stream_read",
                            &mut on_event,
                        )
                        .await?;
                        continue 'agent_iteration;
                    }
                    Err(error) => {
                        emit_attempt_terminal(LlmRequestErrorClass::StreamRead, &prepared_telemetry, &iteration_span, &mut on_event).await?;
                        return Err(error.into());
                    }
                };
                let event = match project_response_stream_event(&raw) {
                    Ok(Some(event)) => event,
                    Ok(None) => continue,
                    Err(error) if is_ignorable_response_lifecycle_event(&raw) => {
                        warn!(error = %error, "ignored response lifecycle stream event without model");
                        continue;
                    }
                    Err(error) => {
                        if stream_had_side_effect {
                            warn!(
                                protocol = "responses",
                                phase = "event_projection",
                                error = %error,
                                text_len = turn_text.len(),
                                tool_count = pending_tool_calls.len(),
                                "recovering responses stream after event projection failure following side effects"
                            );
                            emit_attempt_interrupted(LlmRequestErrorClass::ProtocolValidation, &prepared_telemetry, &iteration_span, &mut on_event).await?;
                            recover_stream_interrupt(
                                agent,
                                &turn_text,
                                &pending_tool_calls,
                                "responses",
                                "event_projection",
                                &mut on_event,
                            )
                            .await?;
                            continue 'agent_iteration;
                        }
                        emit_attempt_terminal(LlmRequestErrorClass::ProtocolValidation, &prepared_telemetry, &iteration_span, &mut on_event).await?;
                        return Err(anyhow!(error).context("failed to deserialize responses stream event"));
                    }
                };

                match event {
                    ResponseStreamEvent::ResponseOutputTextDelta(event) => {
                        stream_had_side_effect = true;
                        trace!(delta_len = event.delta.len(), "received text delta");
                        on_delta(&event.delta).await?;
                        turn_text.push_str(&event.delta);
                        final_text.push_str(&event.delta);
                    }
                    ResponseStreamEvent::ResponseReasoningSummaryTextDelta(event) => {
                        stream_had_side_effect = true;
                        on_event(AgentEvent::ReasoningDelta {
                            item_id: event.item_id,
                            delta: event.delta,
                        })
                        .await?;
                    }
                    ResponseStreamEvent::ResponseReasoningSummaryTextDone(event) => {
                        stream_had_side_effect = true;
                        completed_reasoning_ids.insert(event.item_id.clone());
                        on_event(AgentEvent::ReasoningDone {
                            item_id: event.item_id,
                            text: event.text,
                        })
                        .await?;
                    }
                    ResponseStreamEvent::ResponseOutputItemAdded(event) => {
                        if let OutputItem::FunctionCall(call) = event.item {
                            if emit_tool_call_pending_if_ready(
                                &mut emitted_pending_tool_calls,
                                &call.call_id,
                                &call.name,
                                &mut on_event,
                            )
                            .await?
                            {
                                stream_had_side_effect = true;
                                pending_tool_calls.insert(call.call_id.clone(), call.name.clone());
                            }
                        }
                    }
                    ResponseStreamEvent::ResponseCompleted(event) => {
                        debug!(
                            response_id = %event.response.id,
                            output_items = event.response.output.len(),
                            "streamed response completed"
                        );
                        if let Some(usage) = &event.response.usage {
                            on_event(token_usage_event_from_response_usage(
                                usage,
                                build.budget.context_window_tokens,
                                &cache_report,
                            ))
                            .await?;
                        }
                        completed_response = Some(event.response);
                    }
                    ResponseStreamEvent::ResponseFailed(event) => {
                        if stream_had_side_effect {
                            warn!(
                                protocol = "responses",
                                phase = "response_failed",
                                text_len = turn_text.len(),
                                tool_count = pending_tool_calls.len(),
                                response = ?event.response,
                                "recovering failed responses stream after side effects"
                            );
                            emit_attempt_interrupted(LlmRequestErrorClass::ProviderTerminal, &prepared_telemetry, &iteration_span, &mut on_event).await?;
                            recover_stream_interrupt(
                                agent,
                                &turn_text,
                                &pending_tool_calls,
                                "responses",
                                "response_failed",
                                &mut on_event,
                            )
                            .await?;
                            continue 'agent_iteration;
                        }
                        error!(response = ?event.response, "response failed");
                        emit_attempt_terminal(LlmRequestErrorClass::ProviderTerminal, &prepared_telemetry, &iteration_span, &mut on_event).await?;
                        return Err(anyhow!("response failed: {:#?}", event.response));
                    }
                    ResponseStreamEvent::ResponseIncomplete(event) => {
                        if stream_had_side_effect {
                            warn!(
                                protocol = "responses",
                                phase = "response_incomplete",
                                text_len = turn_text.len(),
                                tool_count = pending_tool_calls.len(),
                                response = ?event.response,
                                "recovering incomplete responses stream after side effects"
                            );
                            emit_attempt_interrupted(LlmRequestErrorClass::ProviderTerminal, &prepared_telemetry, &iteration_span, &mut on_event).await?;
                            recover_stream_interrupt(
                                agent,
                                &turn_text,
                                &pending_tool_calls,
                                "responses",
                                "response_incomplete",
                                &mut on_event,
                            )
                            .await?;
                            continue 'agent_iteration;
                        }
                        warn!(response = ?event.response, "response incomplete");
                        emit_attempt_terminal(LlmRequestErrorClass::ProviderTerminal, &prepared_telemetry, &iteration_span, &mut on_event).await?;
                        return Err(anyhow!("response incomplete: {:#?}", event.response));
                    }
                    _ => {}
                }
            }

            let response = match completed_response {
                Some(response) => response,
                None if !stream_had_side_effect
                    && can_retry_attempt(&agent.retry_config, attempt) =>
                {
                    let delay = retry_delay(&agent.retry_config, attempt);
                    warn!(
                        attempt,
                        max_attempts = agent.retry_config.max_attempts,
                        delay_ms = delay.as_millis(),
                        "retrying streamed response after early end before side effects"
                    );
                    emit_attempt_terminal(LlmRequestErrorClass::StreamRead, &prepared_telemetry, &iteration_span, &mut on_event).await?;
                    tokio::time::sleep(delay).await;
                    attempt += 1;
                    continue 'retry_response_stream;
                }
                None if stream_had_side_effect => {
                    warn!(
                        protocol = "responses",
                        phase = "early_end",
                        text_len = turn_text.len(),
                        tool_count = pending_tool_calls.len(),
                        "recovering responses stream end without response.completed after side effects"
                    );
                    emit_attempt_interrupted(LlmRequestErrorClass::StreamRead, &prepared_telemetry, &iteration_span, &mut on_event).await?;
                    recover_stream_interrupt(
                        agent,
                        &turn_text,
                        &pending_tool_calls,
                        "responses",
                        "early_end",
                        &mut on_event,
                    )
                    .await?;
                    continue 'agent_iteration;
                }
                None => {
                    emit_attempt_terminal(LlmRequestErrorClass::StreamRead, &prepared_telemetry, &iteration_span, &mut on_event).await?;
                    return Err(anyhow!("stream ended without response.completed"));
                }
            };
            break 'retry_response_stream (response, turn_text, completed_reasoning_ids, prepared_telemetry);
        };

        for (index, item) in response.output.iter().enumerate() {
            if let OutputItem::Reasoning(reasoning) = item {
                let item_id = reasoning
                    .id
                    .clone()
                    .unwrap_or_else(|| format!("reasoning-{iteration}-{index}"));
                if completed_reasoning_ids.contains(&item_id) {
                    continue;
                }

                let text = reasoning_summary_text(item);
                if !text.is_empty() {
                    on_event(AgentEvent::ReasoningDone { item_id, text }).await?;
                }
            }
        }

        let tool_calls = response
            .output
            .iter()
            .filter_map(|item| match item {
                OutputItem::FunctionCall(call) => Some(HistoryToolCall {
                    call_id: call.call_id.clone(),
                    name: call.name.clone(),
                    arguments_json: call.arguments.clone(),
                }),
                _ => None,
            })
            .collect::<Vec<_>>();

        let response_usage = response.usage.as_ref().map(|usage| TokenUsageEstimate {
            used_tokens: usage.total_tokens as u64,
            context_window_tokens: build.budget.context_window_tokens,
            input_tokens: usage.input_tokens as u64,
            output_tokens: usage.output_tokens as u64,
            cached_tokens: usage.input_tokens_details.cached_tokens as u64,
        });
        let completed_telemetry = prepared_telemetry.completed(
            response_usage,
            Some(response.id.clone()),
            if response.usage.is_some() {
                ProviderUsageCompleteness::Complete
            } else {
                ProviderUsageCompleteness::UsageMissing
            },
        );
        on_event(AgentEvent::LlmRequestTelemetry(completed_telemetry.clone())).await?;
        langfuse_trace::record_llm_request_telemetry(&iteration_span, &completed_telemetry);

        agent.ensure_tool_call_budget(tool_call_count, tool_calls.len())?;
        tool_call_count += tool_calls.len();

        if tool_calls.is_empty() {
            if turn_text.is_empty() {
                turn_text = response
                    .output_text()
                    .unwrap_or_else(|| "No response content".to_string());
                final_text.push_str(&turn_text);
            }

            agent
                .append_history_item(HistoryItem::assistant(turn_text.clone()))?;
            on_event(AgentEvent::AssistantMessage {
                content: turn_text.clone(),
            })
            .await?;

            langfuse_trace::finish_llm_iteration_span(
                &iteration_span,
                turn_text.chars().count(),
                0,
                response.output.len(),
                None,
            );
            drop(iteration_span);

            if agent
                .continue_or_finalize_no_tool_reply(
                    &mut on_event,
                    tool_call_count,
                    &mut continuation_count,
                )
                .await?
            {
                continue;
            }

            info!(
                output_chars = final_text.chars().count(),
                history_len = agent.history.len(),
                "final answer completed"
            );

            return Ok(final_text);
        }

        langfuse_trace::finish_llm_iteration_span(
            &iteration_span,
            turn_text.chars().count(),
            tool_calls.len(),
            response.output.len(),
            None,
        );
        drop(iteration_span);

        agent.append_assistant_tool_calls(&turn_text, &tool_calls)?;
        on_event(AgentEvent::AssistantToolCallBatch {
            text: (!turn_text.is_empty()).then(|| turn_text.clone()),
            calls: tool_calls.clone(),
        })
        .await?;

        debug!(
            iteration,
            tool_calls = tool_calls.len(),
            tool_call_count,
            history_len = agent.history.len(),
            "response tool calls appended to history"
        );

        for call in tool_calls {
            info!(tool_name = %call.name, call_id = %call.call_id, "tool call requested");
            debug!(
                tool_name = %call.name,
                call_id = %call.call_id,
                arguments = %call.arguments_json,
                "tool call arguments"
            );

            agent
                .execute_tool_call_and_record(&call, &mut on_event, &mut approve)
                .await?;
        }
        on_event(AgentEvent::ToolCallBatchFinished).await?;
    }
    }
    .instrument(turn_span.clone())
    .await;
    langfuse_trace::finish_llm_turn_span(
        &turn_span,
        &result,
        tool_call_count,
        continuation_count,
        agent.history.len(),
    );
    agent.turn.pressure_compaction.reset_for_turn_end();
    result
}

pub(super) async fn run_oai_comp_stream_async<C, F, E, A, Dfut, Efut, Afut>(
    agent: &mut Agent<C>,
    user_content: UserMessageContent,
    user_input: &str,
    mut on_delta: F,
    mut on_event: E,
    mut approve: A,
) -> Result<String>
where
    C: Config + Clone,
    F: FnMut(&str) -> Dfut,
    E: FnMut(AgentEvent) -> Efut + Send,
    A: FnMut(PermissionRequest) -> Afut,
    Dfut: Future<Output = Result<()>>,
    Efut: Future<Output = Result<()>> + Send,
    Afut: Future<Output = Result<PermissionApproval>>,
{
    let turn_prelude = agent.try_prepare_turn_prelude(user_input)?;
    let mut protected_start_index = agent.history.len();
    let previous_turn_start_index = agent.turn.current_turn_start_index;
    agent.turn.current_turn_start_index = Some(protected_start_index);
    if let Err(error) = agent.append_history_item(HistoryItem::user_content(user_content)) {
        agent.turn.current_turn_start_index = previous_turn_start_index;
        return Err(error);
    }
    Agent::<C>::emit_audit_event(
        &mut on_event,
        AgentEvent::TurnStarted(agent.turn_started_event()),
        "turn_started",
    )
    .await;
    debug!(
        user_input_len = user_input.len(),
        history_len = agent.history.len(),
        "user message added to history"
    );

    let turn_id = agent.turn.turn_id;
    let turn_span = langfuse_trace::llm_turn_span(
        turn_id,
        "chat_completions",
        &agent.model,
        agent.max_iterations,
        agent.max_tool_calls,
        user_input.chars().count(),
        agent.history.len(),
    );
    let mut final_text = String::new();
    let mut tool_call_count = 0;
    let mut continuation_count = 0;

    let result = async {
        let mut iteration_count = 0;
        'agent_iteration: loop {
            ensure_iteration_budget(agent.max_iterations, iteration_count)?;
            let iteration = iteration_count;
            iteration_count += 1;
        debug!(
            iteration,
            model = %agent.model,
            history_len = agent.history.len(),
            tool_call_count,
            max_tool_calls = agent.max_tool_calls,
            "creating streamed chat completion"
        );

        let tool_definitions = agent.tool_definitions();
        let prepared = prepare_protocol_stream_request(
            agent,
            ApiProtocol::Completions,
            &turn_prelude,
            &mut protected_start_index,
            &tool_definitions,
            &mut on_event,
        )
        .await?;
        protected_start_index = prepared.protected_start_index;
        let epoch_preview = prepared.epoch_preview;
        let build = prepared.build;
        if agent.turn.frozen_evidence.is_none() {
            agent.turn.frozen_evidence = Some(FrozenTurnEvidence {
                message: build.selected_evidence_message.clone(),
                selected_ids: build.selected_evidence_ids.clone(),
            });
        }
        let logical_observation = agent.preview_final_logical_request(&build);
        let cache_report = CacheUsageReport::from_build(&build);
        let iteration_span = langfuse_trace::llm_iteration_span(
            turn_id,
            "chat_completions",
            &agent.model,
            iteration,
            build.budget.retained_history_items,
            tool_call_count,
            tool_definitions.len(),
        );
        emit_prepared_request_metadata(&build, &mut on_event).await?;
        let logical_request_id = format!("turn-{turn_id}-iteration-{iteration}");
        let completion_request = match build.request.clone() {
            BuiltRequest::Completions(request) => CompletionStreamRequest::Typed(request),
            BuiltRequest::CompletionsCompatible(request) => CompletionStreamRequest::Compatible(request),
            BuiltRequest::Responses(_) | BuiltRequest::ResponsesCompatible(_) => {
                return Err(anyhow!("request builder returned non-completions request"));
            }
        };

        let mut attempt = 1;
        'retry_chat_stream: loop {
            let mut prepared_telemetry = llm_request_telemetry(
                &logical_request_id, turn_id, iteration, attempt, &agent.model,
                ApiProtocol::Completions, &build, tool_call_count, tool_definitions.len(), logical_observation,
            );
            if attempt > 1 {
                // Retries share the logical request; only its primary attempt
                // records adjacent LCP scalars in the durable transcript.
                prepared_telemetry.adjacent_lcp_units = None;
                prepared_telemetry.adjacent_lcp_bytes = None;
                prepared_telemetry.adjacent_lcp_estimated_tokens = None;
                prepared_telemetry.first_breaker = None;
            }
            on_event(AgentEvent::LlmRequestTelemetry(prepared_telemetry.clone())).await?;
            langfuse_trace::record_llm_request_telemetry(&iteration_span, &prepared_telemetry);
            if attempt == 1 {
                agent.commit_final_logical_request(&build);
                agent.commit_active_epoch(epoch_preview.clone());
            }
            let response = match &completion_request {
                CompletionStreamRequest::Typed(request) => {
                    send_compatible_chat_completion_stream(
                        &agent.client,
                        request,
                    )
                    .await
                }
                CompletionStreamRequest::Compatible(request) => {
                    send_compatible_chat_completion_stream(
                        &agent.client,
                        request,
                    )
                    .await
                }
            };
            let response = match response {
                Ok(response) => response,
                Err(error)
                    if should_retry_chat_stream_creation(&agent.retry_config, attempt, &error) =>
                {
                    emit_attempt_terminal(
                        LlmRequestErrorClass::RequestCreation, &prepared_telemetry, &iteration_span, &mut on_event,
                    ).await?;
                    let delay = retry_delay(&agent.retry_config, attempt);
                    tokio::time::sleep(delay).await;
                    attempt += 1;
                    continue 'retry_chat_stream;
                }
                Err(error) => {
                    emit_attempt_terminal(
                        LlmRequestErrorClass::RequestCreation,
                        &prepared_telemetry,
                        &iteration_span,
                        &mut on_event,
                    )
                    .await?;
                    return Err(error).with_context(|| request_creation_failure_context(
                        "streamed chat completion",
                        &agent.model,
                        agent.active_model_metadata(),
                        &build.budget,
                    ));
                }
            };
            let mut byte_stream = response.bytes_stream();
            let mut sse_buffer = String::new();
            let mut turn_text = String::new();
            let mut tool_calls: BTreeMap<usize, ChatCompletionMessageToolCall> = BTreeMap::new();
            let mut emitted_pending_tool_calls = HashSet::new();
            let mut pending_tool_calls: BTreeMap<String, String> = BTreeMap::new();
            let mut finish_reasons: Vec<FinishReason> = Vec::new();
            let mut reasoning =
                InlineReasoningExtractor::new(format!("chat-reasoning-{iteration}"));
            let mut native_reasoning =
                NativeReasoningAccumulator::new(format!("chat-native-reasoning-{iteration}"));
            let mut provider_usage: Option<TokenUsageEstimate> = None;
            let mut provider_cache_details_present = false;
            let mut provider_response_id: Option<String> = None;

            let mut stream_had_side_effect = false;
            while let Some(chunk) = byte_stream.next().await {
                let chunk = match chunk {
                    Ok(chunk) => chunk,
                    Err(error)
                        if !stream_had_side_effect
                            && should_retry_reqwest_error(&agent.retry_config, attempt, &error) =>
                    {
                        let delay = retry_delay(&agent.retry_config, attempt);
                        warn!(
                            attempt,
                            max_attempts = agent.retry_config.max_attempts,
                            delay_ms = delay.as_millis(),
                            error = %error,
                            "retrying chat completions stream read before side effects"
                        );
                        emit_attempt_terminal(
                            LlmRequestErrorClass::StreamRead, &prepared_telemetry,
                            &iteration_span, &mut on_event,
                        )
                        .await?;
                        tokio::time::sleep(delay).await;
                        attempt += 1;
                        continue 'retry_chat_stream;
                    }
                    Err(error) if stream_had_side_effect => {
                        warn!(
                            protocol = "chat_completions",
                            phase = "stream_read",
                            error = %error,
                            text_len = turn_text.len(),
                            tool_count = pending_tool_calls.len(),
                            "recovering interrupted chat stream after side effects"
                        );
                        emit_attempt_interrupted(LlmRequestErrorClass::StreamRead, &prepared_telemetry, &iteration_span, &mut on_event).await?;
                        recover_stream_interrupt(
                            agent,
                            &turn_text,
                            &pending_tool_calls,
                            "chat_completions",
                            "stream_read",
                            &mut on_event,
                        )
                        .await?;
                        continue 'agent_iteration;
                    }
                    Err(error) => {
                        emit_attempt_terminal(LlmRequestErrorClass::StreamRead, &prepared_telemetry, &iteration_span, &mut on_event).await?;
                        return Err(error.into());
                    }
                };
                append_sse_chunk(&mut sse_buffer, &chunk);
                let events = drain_sse_data_events(&mut sse_buffer);
                for event in events {
                    let Some(data) = event else {
                        continue;
                    };
                    let response: CompatibleChatCompletionStreamResponse = match serde_json::from_str(&data) {
                        Ok(response) => response,
                        Err(error)
                            if !stream_had_side_effect
                                && can_retry_attempt(&agent.retry_config, attempt)
                                && is_retryable_json_deserialize_error(&error, &data) =>
                        {
                            let delay = retry_delay(&agent.retry_config, attempt);
                            warn!(
                                protocol = "chat_completions",
                                phase = "event_parse",
                                attempt,
                                max_attempts = agent.retry_config.max_attempts,
                                delay_ms = delay.as_millis(),
                                error = %error,
                                "retrying chat completions stream after transient event parse failure before side effects"
                            );
                            emit_attempt_terminal(LlmRequestErrorClass::ProtocolValidation, &prepared_telemetry, &iteration_span, &mut on_event).await?;
                            tokio::time::sleep(delay).await;
                            attempt += 1;
                            continue 'retry_chat_stream;
                        }
                        Err(error) if stream_had_side_effect => {
                            warn!(
                                protocol = "chat_completions",
                                phase = "event_parse",
                                error = %error,
                                text_len = turn_text.len(),
                                tool_count = pending_tool_calls.len(),
                                "recovering chat stream after event parse failure following side effects"
                            );
                            emit_attempt_interrupted(LlmRequestErrorClass::ProtocolValidation, &prepared_telemetry, &iteration_span, &mut on_event).await?;
                            recover_stream_interrupt(
                                agent,
                                &turn_text,
                                &pending_tool_calls,
                                "chat_completions",
                                "event_parse",
                                &mut on_event,
                            )
                            .await?;
                            continue 'agent_iteration;
                        }
                        Err(error) => {
                            emit_attempt_terminal(LlmRequestErrorClass::ProtocolValidation, &prepared_telemetry, &iteration_span, &mut on_event).await?;
                            return Err(error).with_context(|| {
                                format!("failed to parse chat completions stream event: {data}")
                            });
                        }
                    };
                    if let Some(usage) = &response.usage {
                        stream_had_side_effect = true;
                        on_event(token_usage_event_from_completion_usage(
                            usage,
                            build.budget.context_window_tokens,
                            &cache_report,
                        ))
                        .await?;
                        provider_cache_details_present = usage.prompt_tokens_details.as_ref().and_then(|details| details.cached_tokens).is_some();
                        provider_usage = Some(TokenUsageEstimate { used_tokens: usage.total_tokens as u64, context_window_tokens: build.budget.context_window_tokens, input_tokens: usage.prompt_tokens as u64, output_tokens: usage.completion_tokens as u64, cached_tokens: usage.prompt_tokens_details.as_ref().and_then(|details| details.cached_tokens).unwrap_or(0) as u64 });
                        prepared_telemetry.usage = provider_usage;
                        prepared_telemetry.usage_completeness = completion_usage_completeness(provider_usage, provider_cache_details_present);
                    }
                    provider_response_id = response.id.clone().or(provider_response_id);
                    for choice in response.choices {
                        if choice.index != 0 {
                            emit_attempt_terminal(LlmRequestErrorClass::ProtocolValidation, &prepared_telemetry, &iteration_span, &mut on_event).await?;
                            return Err(anyhow!(
                                "completions returned unexpected choice index {}; only n=1/index 0 is supported",
                                choice.index
                            ));
                        }

                        if let Some(delta) = choice.delta {
                            if let Some(reasoning_delta) = delta.reasoning_delta() {
                                if let Some(event) = native_reasoning.push(reasoning_delta) {
                                    stream_had_side_effect = true;
                                    on_event(event).await?;
                                }
                            }

                            if let Some(content_delta) = delta.content {
                                trace!(delta_len = content_delta.len(), "received chat text delta");
                                for part in reasoning.push(&content_delta) {
                                    match part {
                                        StreamTextPart::Visible(text) => {
                                            stream_had_side_effect = true;
                                            on_delta(&text).await?;
                                            turn_text.push_str(&text);
                                            final_text.push_str(&text);
                                        }
                                        StreamTextPart::ReasoningDelta { item_id, delta } => {
                                            stream_had_side_effect = true;
                                            on_event(AgentEvent::ReasoningDelta { item_id, delta })
                                                .await?;
                                        }
                                        StreamTextPart::ReasoningDone { item_id, text } => {
                                            stream_had_side_effect = true;
                                            on_event(AgentEvent::ReasoningDone { item_id, text })
                                                .await?;
                                        }
                                    }
                                }
                            }

                            if let Some(chunks) = delta.tool_calls {
                                for chunk in chunks {
                                    let index = chunk.index as usize;
                                    merge_chat_tool_call_chunk(&mut tool_calls, chunk);
                                    if let Some(call) = tool_calls.get(&index) {
                                        if emit_tool_call_pending_if_ready(
                                            &mut emitted_pending_tool_calls,
                                            &call.id,
                                            &call.function.name,
                                            &mut on_event,
                                        )
                                        .await?
                                        {
                                            stream_had_side_effect = true;
                                            pending_tool_calls.insert(
                                                call.id.clone(),
                                                call.function.name.clone(),
                                            );
                                        }
                                    }
                                }
                            }
                        }

                        if let Some(reason) = choice.finish_reason {
                            finish_reasons.push(reason);
                        }
                    }
                }
            }

            let events = finish_sse_data_events(&mut sse_buffer);
            for event in events {
                let Some(data) = event else {
                    continue;
                };
                let response: CompatibleChatCompletionStreamResponse = match serde_json::from_str(&data) {
                    Ok(response) => response,
                    Err(error)
                        if !stream_had_side_effect
                            && can_retry_attempt(&agent.retry_config, attempt)
                            && is_retryable_json_deserialize_error(&error, &data) =>
                    {
                        let delay = retry_delay(&agent.retry_config, attempt);
                        warn!(
                            protocol = "chat_completions",
                            phase = "finish_event_parse",
                            attempt,
                            max_attempts = agent.retry_config.max_attempts,
                            delay_ms = delay.as_millis(),
                            error = %error,
                            "retrying chat completions stream after transient final event parse failure before side effects"
                        );
                        emit_attempt_terminal(LlmRequestErrorClass::ProtocolValidation, &prepared_telemetry, &iteration_span, &mut on_event).await?;
                        tokio::time::sleep(delay).await;
                        attempt += 1;
                        continue 'retry_chat_stream;
                    }
                    Err(error) if stream_had_side_effect => {
                        warn!(
                            protocol = "chat_completions",
                            phase = "finish_event_parse",
                            error = %error,
                            text_len = turn_text.len(),
                            tool_count = pending_tool_calls.len(),
                            "recovering chat stream after final event parse failure following side effects"
                        );
                        emit_attempt_interrupted(LlmRequestErrorClass::ProtocolValidation, &prepared_telemetry, &iteration_span, &mut on_event).await?;
                        recover_stream_interrupt(
                            agent,
                            &turn_text,
                            &pending_tool_calls,
                            "chat_completions",
                            "finish_event_parse",
                            &mut on_event,
                        )
                        .await?;
                        continue 'agent_iteration;
                    }
                    Err(error) => {
                        emit_attempt_terminal(LlmRequestErrorClass::ProtocolValidation, &prepared_telemetry, &iteration_span, &mut on_event).await?;
                        return Err(error).with_context(|| {
                            format!("failed to parse chat completions stream event: {data}")
                        });
                    }
                };
                if let Some(usage) = &response.usage {
                    stream_had_side_effect = true;
                    on_event(token_usage_event_from_completion_usage(
                        usage,
                        build.budget.context_window_tokens,
                        &cache_report,
                    ))
                    .await?;
                    provider_cache_details_present = usage.prompt_tokens_details.as_ref().and_then(|details| details.cached_tokens).is_some();
                    provider_usage = Some(TokenUsageEstimate { used_tokens: usage.total_tokens as u64, context_window_tokens: build.budget.context_window_tokens, input_tokens: usage.prompt_tokens as u64, output_tokens: usage.completion_tokens as u64, cached_tokens: usage.prompt_tokens_details.as_ref().and_then(|details| details.cached_tokens).unwrap_or(0) as u64 });
                    prepared_telemetry.usage = provider_usage;
                    prepared_telemetry.usage_completeness = completion_usage_completeness(provider_usage, provider_cache_details_present);
                }
                provider_response_id = response.id.clone().or(provider_response_id);
                for choice in response.choices {
                    if choice.index != 0 {
                        emit_attempt_terminal(LlmRequestErrorClass::ProtocolValidation, &prepared_telemetry, &iteration_span, &mut on_event).await?;
                        return Err(anyhow!(
                            "completions returned unexpected choice index {}; only n=1/index 0 is supported",
                            choice.index
                        ));
                    }

                    if let Some(delta) = choice.delta {
                        if let Some(reasoning_delta) = delta.reasoning_delta() {
                            if let Some(event) = native_reasoning.push(reasoning_delta) {
                                stream_had_side_effect = true;
                                on_event(event).await?;
                            }
                        }

                        if let Some(content_delta) = delta.content {
                            trace!(delta_len = content_delta.len(), "received chat text delta");
                            for part in reasoning.push(&content_delta) {
                                match part {
                                    StreamTextPart::Visible(text) => {
                                        stream_had_side_effect = true;
                                        on_delta(&text).await?;
                                        turn_text.push_str(&text);
                                        final_text.push_str(&text);
                                    }
                                    StreamTextPart::ReasoningDelta { item_id, delta } => {
                                        stream_had_side_effect = true;
                                        on_event(AgentEvent::ReasoningDelta { item_id, delta })
                                            .await?;
                                    }
                                    StreamTextPart::ReasoningDone { item_id, text } => {
                                        stream_had_side_effect = true;
                                        on_event(AgentEvent::ReasoningDone { item_id, text })
                                            .await?;
                                    }
                                }
                            }
                        }

                        if let Some(chunks) = delta.tool_calls {
                            for chunk in chunks {
                                let index = chunk.index as usize;
                                merge_chat_tool_call_chunk(&mut tool_calls, chunk);
                                if let Some(call) = tool_calls.get(&index) {
                                    if emit_tool_call_pending_if_ready(
                                        &mut emitted_pending_tool_calls,
                                        &call.id,
                                        &call.function.name,
                                        &mut on_event,
                                    )
                                    .await?
                                    {
                                        stream_had_side_effect = true;
                                        pending_tool_calls
                                            .insert(call.id.clone(), call.function.name.clone());
                                    }
                                }
                            }
                        }
                    }

                    if let Some(reason) = choice.finish_reason {
                        finish_reasons.push(reason);
                    }
                }
            }

            for part in reasoning.finish() {
                match part {
                    StreamTextPart::Visible(text) => {
                        stream_had_side_effect = true;
                        on_delta(&text).await?;
                        turn_text.push_str(&text);
                        final_text.push_str(&text);
                    }
                    StreamTextPart::ReasoningDelta { item_id, delta } => {
                        stream_had_side_effect = true;
                        on_event(AgentEvent::ReasoningDelta { item_id, delta }).await?;
                    }
                    StreamTextPart::ReasoningDone { item_id, text } => {
                        stream_had_side_effect = true;
                        on_event(AgentEvent::ReasoningDone { item_id, text }).await?;
                    }
                }
            }
            if let Some(event) = native_reasoning.finish() {
                stream_had_side_effect = true;
                on_event(event).await?;
            }

            let has_tool_calls = !tool_calls.is_empty();
            if finish_reasons.is_empty()
                && !stream_had_side_effect
                && can_retry_attempt(&agent.retry_config, attempt)
            {
                let delay = retry_delay(&agent.retry_config, attempt);
                warn!(
                    attempt,
                    max_attempts = agent.retry_config.max_attempts,
                    delay_ms = delay.as_millis(),
                    "retrying chat completions stream after early end before side effects"
                );
                emit_attempt_terminal(LlmRequestErrorClass::StreamRead, &prepared_telemetry, &iteration_span, &mut on_event).await?;
                tokio::time::sleep(delay).await;
                attempt += 1;
                continue 'retry_chat_stream;
            }
            if finish_reasons.is_empty() && (stream_had_side_effect || has_tool_calls) {
                warn!(
                    protocol = "chat_completions",
                    phase = "finish_reason_validation",
                    text_len = turn_text.len(),
                    tool_count = pending_tool_calls.len(),
                    "recovering interrupted chat stream after missing finish state"
                );
                emit_attempt_interrupted(LlmRequestErrorClass::ProtocolValidation, &prepared_telemetry, &iteration_span, &mut on_event).await?;
                recover_stream_interrupt(
                    agent,
                    &turn_text,
                    &pending_tool_calls,
                    "chat_completions",
                    "finish_reason_validation",
                    &mut on_event,
                )
                .await?;
                continue 'agent_iteration;
            }
            if let Err(error) = validate_chat_finish_reasons(&finish_reasons, has_tool_calls) {
                if !pending_tool_calls.is_empty() {
                    if finish_reasons.iter().any(|reason| {
                        matches!(reason, FinishReason::Length | FinishReason::ContentFilter)
                    }) {
                        emit_pending_tool_call_cancellations(&pending_tool_calls, &mut on_event)
                            .await?;
                        emit_attempt_terminal(LlmRequestErrorClass::ProtocolValidation, &prepared_telemetry, &iteration_span, &mut on_event).await?;
                        return Err(error);
                    }
                    warn!(
                        protocol = "chat_completions",
                        phase = "finish_reason_validation",
                        error = %error,
                        text_len = turn_text.len(),
                        tool_count = pending_tool_calls.len(),
                        "recovering interrupted chat stream with incomplete pending tool call"
                    );
                    emit_attempt_interrupted(LlmRequestErrorClass::ProtocolValidation, &prepared_telemetry, &iteration_span, &mut on_event).await?;
                    recover_stream_interrupt(
                        agent,
                        &turn_text,
                        &pending_tool_calls,
                        "chat_completions",
                        "finish_reason_validation",
                        &mut on_event,
                    )
                    .await?;
                    continue 'agent_iteration;
                }
                emit_attempt_terminal(LlmRequestErrorClass::ProtocolValidation, &prepared_telemetry, &iteration_span, &mut on_event).await?;
                return Err(error);
            }
            let finish_reasons_label = format!("{finish_reasons:?}");

            if !has_tool_calls {
                let completed_telemetry = prepared_telemetry.completed(
                    provider_usage,
                    provider_response_id,
                    completion_usage_completeness(provider_usage, provider_cache_details_present),
                );
                on_event(AgentEvent::LlmRequestTelemetry(completed_telemetry.clone())).await?;
                langfuse_trace::record_llm_request_telemetry(&iteration_span, &completed_telemetry);
                if final_text.is_empty() {
                    final_text = "No response content".to_string();
                }

                agent
                    .append_history_item(HistoryItem::assistant(turn_text.clone()))?;
                on_event(AgentEvent::AssistantMessage {
                    content: turn_text.clone(),
                })
                .await?;

                langfuse_trace::finish_llm_iteration_span(
                    &iteration_span,
                    turn_text.chars().count(),
                    0,
                    0,
                    Some(&finish_reasons_label),
                );
                drop(iteration_span);

                if agent
                    .continue_or_finalize_no_tool_reply(
                        &mut on_event,
                        tool_call_count,
                        &mut continuation_count,
                    )
                    .await?
                {
                    continue 'agent_iteration;
                }

                info!(
                    output_chars = final_text.chars().count(),
                    history_len = agent.history.len(),
                    "final chat completion answer completed"
                );

                return Ok(final_text);
            }

            let tool_calls = compact_indexed_chat_tool_calls(tool_calls);
            if let Err(error) = validate_chat_tool_calls(&tool_calls) {
                if stream_had_side_effect || !pending_tool_calls.is_empty() {
                    warn!(
                        protocol = "chat_completions",
                        phase = "tool_call_validation",
                        error = %error,
                        text_len = turn_text.len(),
                        tool_count = pending_tool_calls.len(),
                        "recovering interrupted chat stream with incomplete tool call"
                    );
                    emit_attempt_interrupted(LlmRequestErrorClass::ProtocolValidation, &prepared_telemetry, &iteration_span, &mut on_event).await?;
                    recover_stream_interrupt(
                        agent,
                        &turn_text,
                        &pending_tool_calls,
                        "chat_completions",
                        "tool_call_validation",
                        &mut on_event,
                    )
                    .await?;
                    continue 'agent_iteration;
                }
                emit_attempt_terminal(LlmRequestErrorClass::ProtocolValidation, &prepared_telemetry, &iteration_span, &mut on_event).await?;
                return Err(error);
            }
            let tool_calls = tool_calls
                .into_iter()
                .map(|call| HistoryToolCall {
                    call_id: call.id,
                    name: call.function.name,
                    arguments_json: call.function.arguments,
                })
                .collect::<Vec<_>>();

            let completed_telemetry = prepared_telemetry.completed(
                provider_usage,
                provider_response_id,
                completion_usage_completeness(provider_usage, provider_cache_details_present),
            );
            on_event(AgentEvent::LlmRequestTelemetry(completed_telemetry.clone())).await?;
            langfuse_trace::record_llm_request_telemetry(&iteration_span, &completed_telemetry);

            agent.ensure_tool_call_budget(tool_call_count, tool_calls.len())?;

            tool_call_count += tool_calls.len();
            langfuse_trace::finish_llm_iteration_span(
                &iteration_span,
                turn_text.chars().count(),
                tool_calls.len(),
                tool_calls.len(),
                Some(&finish_reasons_label),
            );
            drop(iteration_span);
            agent.append_assistant_tool_calls(&turn_text, &tool_calls)?;
            on_event(AgentEvent::AssistantToolCallBatch {
                text: (!turn_text.is_empty()).then(|| turn_text.clone()),
                calls: tool_calls.clone(),
            })
            .await?;

            for call in tool_calls {
                info!(tool_name = %call.name, call_id = %call.call_id, "chat tool call requested");
                debug!(
                    tool_name = %call.name,
                    call_id = %call.call_id,
                    arguments = %call.arguments_json,
                    "chat tool call arguments"
                );

                agent
                    .execute_tool_call_and_record(&call, &mut on_event, &mut approve)
                    .await?;
            }
            on_event(AgentEvent::ToolCallBatchFinished).await?;
            break 'retry_chat_stream;
        }
    }
    }
    .instrument(turn_span.clone())
    .await;
    langfuse_trace::finish_llm_turn_span(
        &turn_span,
        &result,
        tool_call_count,
        continuation_count,
        agent.history.len(),
    );
    agent.turn.pressure_compaction.reset_for_turn_end();
    result
}

pub(super) async fn send_compatible_chat_completion_stream<C: Config>(
    client: &Client<C>,
    request: &impl Serialize,
) -> std::result::Result<reqwest::Response, ChatStreamCreationError> {
    let config = client.config();
    let url = config.url("/chat/completions");
    let http = reqwest_client_for_url(&url)
        .map_err(|error| ChatStreamCreationError::Setup(error.to_string()))?;

    let response = match http
        .post(url.clone())
        .query(&config.query())
        .headers(config.headers())
        .json(request)
        .send()
        .await
    {
        Ok(response) => response,
        Err(error) => return Err(ChatStreamCreationError::Transport(error)),
    };

    let status = response.status();
    if status.is_success() {
        return Ok(response);
    }

    let message = response
        .text()
        .await
        .unwrap_or_else(|error| format!("failed to read error body: {error}"));
    Err(ChatStreamCreationError::Status { status, message })
}

fn request_creation_failure_context(
    operation: &str,
    model: &str,
    metadata: crate::request_builder::ModelRequestMetadata,
    budget: &crate::request_builder::BudgetReport,
) -> String {
    format!(
        "failed to create {operation} (model={model}, estimated_request_tokens={}, input_budget_tokens={}, context_window_tokens={}, effective_input_limit_tokens={}, prelude_tokens={}, protected_tokens={}, retained_history_tokens={}, tools_tokens={}, evidence_tokens={}, retained_history_items={}, dropped_history_items={}, selected_evidence_items={}, dropped_evidence_items={})",
        budget.estimated_request_tokens,
        budget.input_budget_tokens,
        budget.context_window_tokens,
        metadata
            .effective_input_limit_tokens()
            .map(|value| value.to_string())
            .unwrap_or_else(|| "none".into()),
        budget.estimated_prelude_tokens,
        budget.estimated_protected_tokens,
        budget.estimated_retained_history_tokens,
        budget.estimated_tools_tokens,
        budget.estimated_evidence_tokens,
        budget.retained_history_items,
        budget.dropped_history_items,
        budget.selected_evidence_items,
        budget.dropped_evidence_items,
    )
}

fn reqwest_client_for_url(url: &str) -> Result<reqwest::Client> {
    let builder = reqwest::Client::builder();
    let builder = if is_loopback_url(url) {
        builder.no_proxy()
    } else {
        builder
    };
    builder
        .build()
        .context("failed to build chat completions HTTP client")
}

fn is_loopback_url(url: &str) -> bool {
    reqwest::Url::parse(url)
        .ok()
        .and_then(|url| url.host_str().map(str::to_owned))
        .is_some_and(|host| {
            host.eq_ignore_ascii_case("localhost")
                || host
                    .parse::<std::net::IpAddr>()
                    .is_ok_and(|addr| addr.is_loopback())
        })
}

#[derive(Debug, Deserialize)]
pub(super) struct CompatibleChatCompletionStreamResponse {
    pub(super) id: Option<String>,
    pub(super) choices: Vec<CompatibleChatChoiceStream>,
    pub(super) usage: Option<CompletionUsage>,
}

pub(super) fn token_usage_event_from_response_usage(
    usage: &ResponseUsage,
    context_window_tokens: u64,
    cache_report: &CacheUsageReport,
) -> AgentEvent {
    let cached_tokens = usage.input_tokens_details.cached_tokens as u64;
    AgentEvent::TokenUsageUpdated {
        used_tokens: usage.total_tokens as u64,
        context_window_tokens,
        input_tokens: usage.input_tokens as u64,
        output_tokens: usage.output_tokens as u64,
        cached_tokens,
        cache_report: Some(cache_report.with_actual_cached_tokens(cached_tokens)),
    }
}

pub(super) fn token_usage_event_from_completion_usage(
    usage: &CompletionUsage,
    context_window_tokens: u64,
    cache_report: &CacheUsageReport,
) -> AgentEvent {
    let cached_tokens = usage
        .prompt_tokens_details
        .as_ref()
        .and_then(|details| details.cached_tokens)
        .unwrap_or(0) as u64;
    let cache_report = match usage
        .prompt_tokens_details
        .as_ref()
        .and_then(|details| details.cached_tokens)
    {
        Some(_) => cache_report.with_actual_cached_tokens(cached_tokens),
        None => cache_report.clone(),
    };
    AgentEvent::TokenUsageUpdated {
        used_tokens: usage.total_tokens as u64,
        context_window_tokens,
        input_tokens: usage.prompt_tokens as u64,
        output_tokens: usage.completion_tokens as u64,
        cached_tokens,
        cache_report: Some(cache_report),
    }
}

fn completion_usage_completeness(
    usage: Option<TokenUsageEstimate>,
    cache_details_present: bool,
) -> ProviderUsageCompleteness {
    if usage.is_none() {
        ProviderUsageCompleteness::UsageMissing
    } else if cache_details_present {
        ProviderUsageCompleteness::Complete
    } else {
        ProviderUsageCompleteness::CacheDetailsMissing
    }
}

#[derive(Debug, Deserialize)]
pub(super) struct CompatibleChatChoiceStream {
    pub(super) index: u32,
    pub(super) delta: Option<CompatibleChatCompletionStreamResponseDelta>,
    pub(super) finish_reason: Option<FinishReason>,
}

#[derive(Debug, Deserialize)]
pub(super) struct CompatibleChatCompletionStreamResponseDelta {
    pub(super) content: Option<String>,
    pub(super) tool_calls: Option<Vec<ChatCompletionMessageToolCallChunk>>,
    pub(super) reasoning_content: Option<CompatibleReasoningDelta>,
    pub(super) reasoning: Option<CompatibleReasoningDelta>,
    pub(super) thinking: Option<CompatibleReasoningDelta>,
}

impl CompatibleChatCompletionStreamResponseDelta {
    pub(super) fn reasoning_delta(&self) -> Option<String> {
        [
            self.reasoning_content.as_ref(),
            self.reasoning.as_ref(),
            self.thinking.as_ref(),
        ]
        .into_iter()
        .flatten()
        .find_map(|reasoning| reasoning.to_text().filter(|text| !text.is_empty()))
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub(super) enum CompatibleReasoningDelta {
    Text(String),
    Object {
        content: Option<String>,
        text: Option<String>,
        summary: Option<String>,
    },
    Array(Vec<CompatibleReasoningDelta>),
}

impl CompatibleReasoningDelta {
    fn to_text(&self) -> Option<String> {
        match self {
            Self::Text(text) => Some(text.clone()),
            Self::Object {
                content,
                text,
                summary,
            } => content
                .as_ref()
                .or(text.as_ref())
                .or(summary.as_ref())
                .cloned(),
            Self::Array(parts) => {
                let text = parts.iter().filter_map(Self::to_text).collect::<String>();
                (!text.is_empty()).then_some(text)
            }
        }
    }
}

#[derive(Debug, Clone)]
struct NativeReasoningAccumulator {
    item_id: String,
    text: String,
}

impl NativeReasoningAccumulator {
    fn new(item_id: impl Into<String>) -> Self {
        Self {
            item_id: item_id.into(),
            text: String::new(),
        }
    }

    fn push(&mut self, delta: String) -> Option<AgentEvent> {
        if delta.is_empty() {
            return None;
        }
        self.text.push_str(&delta);
        Some(AgentEvent::ReasoningDelta {
            item_id: self.item_id.clone(),
            delta,
        })
    }

    fn finish(self) -> Option<AgentEvent> {
        (!self.text.is_empty()).then_some(AgentEvent::ReasoningDone {
            item_id: self.item_id,
            text: self.text,
        })
    }
}

pub(super) fn append_sse_chunk(buffer: &mut String, chunk: &[u8]) {
    buffer.push_str(&String::from_utf8_lossy(chunk));
}

pub(super) fn drain_sse_data_events(buffer: &mut String) -> Vec<Option<String>> {
    let mut events = Vec::new();
    while let Some((index, len)) = find_sse_event_boundary(buffer) {
        let raw = buffer[..index].to_string();
        buffer.drain(..index + len);
        if let Some(event) = parse_sse_data_event(&raw) {
            events.push(event);
        }
    }
    events
}

fn finish_sse_data_events(buffer: &mut String) -> Vec<Option<String>> {
    let mut events = drain_sse_data_events(buffer);
    if !buffer.trim().is_empty() {
        let raw = std::mem::take(buffer);
        if let Some(event) = parse_sse_data_event(&raw) {
            events.push(event);
        }
    }
    events
}

async fn recover_stream_interrupt<C, E, Efut>(
    agent: &mut Agent<C>,
    turn_text: &str,
    pending_tool_calls: &BTreeMap<String, String>,
    protocol: &str,
    phase: &str,
    on_event: &mut E,
) -> Result<()>
where
    C: Config + Clone,
    E: FnMut(AgentEvent) -> Efut,
    Efut: Future<Output = Result<()>>,
{
    emit_pending_tool_call_cancellations(pending_tool_calls, on_event).await?;

    if !turn_text.is_empty() {
        agent.append_history_item(HistoryItem::assistant(turn_text.to_string()))?;
        on_event(AgentEvent::AssistantMessage {
            content: turn_text.to_string(),
        })
        .await?;
    }

    let detail = if pending_tool_calls.is_empty() {
        if turn_text.is_empty() {
            format!(
                "The {protocol} stream was interrupted during {phase}. The next model iteration will continue from the latest turn state."
            )
        } else {
            format!(
                "The {protocol} stream was interrupted during {phase}. Partial assistant output was preserved."
            )
        }
    } else {
        format!(
            "The {protocol} stream was interrupted during {phase}. {} incomplete tool call(s) were cancelled{}.",
            pending_tool_calls.len(),
            if turn_text.is_empty() {
                ""
            } else {
                "; partial assistant output was preserved"
            }
        )
    };
    on_event(AgentEvent::ModelStreamIssue {
        message: STREAM_INTERRUPT_MESSAGE.to_string(),
        detail: Some(detail),
        action: STREAM_INTERRUPT_ACTION.to_string(),
    })
    .await?;

    let text = build_stream_interrupt_continuation(turn_text, pending_tool_calls);
    agent.append_history_item(HistoryItem::internal_continuation(text.clone()))?;
    on_event(AgentEvent::InternalContinuation {
        text,
        source: crate::transcript::InternalContinuationSource::StreamRecovery,
    })
    .await?;

    Ok(())
}

fn build_stream_interrupt_continuation(
    turn_text: &str,
    pending_tool_calls: &BTreeMap<String, String>,
) -> String {
    let mut message = String::from(
        "The previous model stream was interrupted before completion. Continue the same task from the latest state. Do not repeat assistant text that has already been shown to the user.",
    );
    if !turn_text.is_empty() {
        message.push_str(" Continue from the partial assistant text already in history.");
    }
    if !pending_tool_calls.is_empty() {
        message.push_str(
            " Any streamed tool calls from the interrupted response were cancelled. If tool work is still needed, emit complete fresh tool calls before requesting execution.",
        );
    }
    message
}

async fn emit_pending_tool_call_cancellations<E, Efut>(
    pending_tool_calls: &BTreeMap<String, String>,
    on_event: &mut E,
) -> Result<()>
where
    E: FnMut(AgentEvent) -> Efut,
    Efut: Future<Output = Result<()>>,
{
    for (call_id, name) in pending_tool_calls {
        on_event(AgentEvent::ToolCallCancelled {
            call_id: call_id.clone(),
            name: name.clone(),
        })
        .await?;
    }
    Ok(())
}

fn ensure_iteration_budget(limit: Option<usize>, iteration_count: usize) -> Result<()> {
    if let Some(limit) = limit
        && iteration_count >= limit
    {
        return Err(anyhow!(
            "stopped: too many agent iterations (max {})",
            limit
        ));
    }
    Ok(())
}

fn find_sse_event_boundary(buffer: &str) -> Option<(usize, usize)> {
    match (buffer.find("\n\n"), buffer.find("\r\n\r\n")) {
        (Some(lf), Some(crlf)) if crlf < lf => Some((crlf, 4)),
        (Some(lf), _) => Some((lf, 2)),
        (None, Some(crlf)) => Some((crlf, 4)),
        (None, None) => None,
    }
}

fn parse_sse_data_event(raw: &str) -> Option<Option<String>> {
    let data = raw
        .lines()
        .filter_map(|line| line.strip_prefix("data:"))
        .map(str::trim_start)
        .collect::<Vec<_>>()
        .join("\n");

    if data.is_empty() {
        return None;
    }
    if data.trim() == "[DONE]" {
        return Some(None);
    }
    Some(Some(data))
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_openai::config::OpenAIConfig;
    use serde_json::{Value, json};
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    fn prepared_build_agent() -> Agent<OpenAIConfig> {
        let client = Client::with_config(
            OpenAIConfig::new()
                .with_api_base("https://api.openai.com/v1")
                .with_api_key("test"),
        );
        let mut agent = Agent::new(client, "group-16", 4, 4);
        agent.prelude = vec![PromptMessage::system("GROUP-16 stable prelude")];
        agent.set_model_catalog(HashMap::from([(
            "group-16".into(),
            crate::request_builder::ModelRequestMetadata {
                context_window: Some(2_048),
                effective_input_limit_tokens: Some(1_280),
                max_output_tokens: Some(512),
                supports_tools: true,
                prompt_cache: crate::config::PromptCacheConfig {
                    enabled: true,
                    retention: Some(crate::config::PromptCacheRetention::InMemory),
                    namespace: Some("group-16".into()),
                },
                ..Default::default()
            },
        )]));
        agent
            .replace_history(vec![
                HistoryItem::user(format!("GROUP-16-DROPPED-SENTINEL {}", "old ".repeat(800))),
                HistoryItem::assistant("GROUP-16 old response"),
                HistoryItem::user("GROUP-16 volatile runtime material"),
            ])
            .expect("legal test history");
        agent
    }

    #[test]
    fn valid_top_level_response_metadata_extension_is_ignored() {
        let event = json!({
            "type": "response.metadata",
            "response_id": "resp_123",
            "sequence_number": 7,
            "metadata": { "provider": "test" },
        });

        assert!(
            project_response_stream_event(&event)
                .expect("valid provider extension")
                .is_none()
        );
    }

    #[test]
    fn malformed_top_level_response_metadata_extension_fails_closed() {
        for event in [
            json!({ "type": "response.metadata", "sequence_number": 7, "metadata": {} }),
            json!({ "type": "response.metadata", "response_id": 7, "sequence_number": 7, "metadata": {} }),
            json!({ "type": "response.metadata", "response_id": "resp_123", "sequence_number": 7.5, "metadata": {} }),
            json!({ "type": "response.metadata", "response_id": "resp_123", "sequence_number": -1, "metadata": {} }),
            json!({ "type": "response.metadata", "response_id": "resp_123", "sequence_number": 7, "metadata": [] }),
        ] {
            assert!(project_response_stream_event(&event).is_err(), "{event}");
        }
    }

    #[test]
    fn unknown_response_stream_event_remains_a_strict_error() {
        let event = json!({ "type": "response.provider_extension", "value": true });
        assert!(project_response_stream_event(&event).is_err());
    }

    #[test]
    fn projection_removes_only_nested_reasoning_and_retains_nested_metadata() {
        let event = json!({
            "type": "response.created",
            "response": {
                "reasoning": { "opaque": true },
                "metadata": { "request": "preserve" },
            },
        });

        let projected = project_response_stream_event_value(&event)
            .expect("nested projection")
            .expect("normal event is not ignored");
        assert!(projected["response"].get("reasoning").is_none());
        assert_eq!(
            projected["response"]["metadata"],
            json!({ "request": "preserve" })
        );
    }

    struct Group16Tool;

    #[async_trait::async_trait]
    impl crate::tool::ToolHandler for Group16Tool {
        fn name(&self) -> &'static str {
            "group_16_tool"
        }

        fn description(&self) -> &'static str {
            "Deterministic GROUP-16 tool"
        }

        fn parameters(&self) -> Value {
            json!({"type": "object", "properties": {}})
        }

        async fn execute(&self, _args: Value) -> Result<Value> {
            Ok(json!({"status": "complete"}))
        }
    }

    fn prepared_group_16_agent() -> Agent<OpenAIConfig> {
        let mut agent = prepared_build_agent();
        agent.tools = crate::tool::ToolRegistry::new();
        agent.register_tool(Group16Tool);
        agent
    }

    fn checkpoint_candidate_for(
        agent: &Agent<OpenAIConfig>,
    ) -> crate::transcript::PreparedLogicalCheckpoint {
        use crate::protocol_frames::{ProtocolFrame, ProtocolFrameItem};
        use crate::runtime_context::{RuntimeFrameProvenance, RuntimeSource, SourceSpan};
        use crate::transcript::LogicalCheckpointEventV1;

        let previous_segment_id = agent
            .runtime_snapshot
            .current_segment_id
            .expect("live segment");
        let boundary_sequence = agent
            .runtime_snapshot
            .leaf_sequence
            .expect("journal frontier");
        let checkpoint_id = "phase1c-checkpoint";
        let event = LogicalCheckpointEventV1 {
            schema_version: 1,
            checkpoint_id: checkpoint_id.into(),
            turn_id: agent.turn.turn_id,
            previous_segment_id,
            segment_id: previous_segment_id + 1,
            previous_checkpoint_id: None,
            boundary_sequence,
            context_scope_revision: agent.runtime_snapshot.context_scope_revision,
            covered_source_spans: Vec::new(),
            retained_items: Vec::new(),
        };
        let leaf = boundary_sequence + 1;
        let mut frames = vec![
            ProtocolFrame::derived(ProtocolFrameItem::AssistantToolCalls {
                text: Some("retained checkpoint tool group".into()),
                calls: vec![crate::protocol_frames::ProtocolToolCall {
                    call_id: "phase1c-retained-call".into(),
                    name: "group_16_tool".into(),
                    arguments_json: r#"{"checkpoint":true}"#.into(),
                }],
            }),
            ProtocolFrame::derived(ProtocolFrameItem::ToolOutput {
                call_id: "phase1c-retained-call".into(),
                output_json: r#"{"checkpoint_output":"retained"}"#.into(),
            }),
            ProtocolFrame::derived(ProtocolFrameItem::ContextSummary {
                text: crate::transcript::render_checkpoint_v1(&event).expect("summary renders"),
            }),
            ProtocolFrame::derived(ProtocolFrameItem::InternalContinuation {
                text: crate::transcript::render_checkpoint_continuation_v1(&event),
            }),
        ];
        for (index, (frame, source_id)) in frames
            .iter_mut()
            .zip([
                "phase1c-checkpoint:retained-call",
                "phase1c-checkpoint:retained-output",
                "phase1c-checkpoint:summary",
                "phase1c-checkpoint:continuation",
            ])
            .enumerate()
        {
            frame.history_index = index;
            frame.source_provenance = Some(
                RuntimeFrameProvenance::new(RuntimeSource::Transcript)
                    .with_source_id(source_id)
                    .with_span(SourceSpan::new(leaf, leaf).expect("valid source span")),
            );
        }
        let mut snapshot =
            RuntimeSnapshot::new(agent.runtime_snapshot.active_context.branch_id.clone());
        snapshot.current_turn_id = Some(agent.turn.turn_id);
        snapshot.current_segment_id = Some(event.segment_id);
        snapshot.leaf_sequence = Some(leaf);
        snapshot.latest_model = Some(agent.model.clone());
        snapshot.frames = frames
            .iter()
            .enumerate()
            .map(|(index, frame)| {
                let mut runtime = runtime_frame_from_protocol_frame(frame, index as u32);
                runtime.provenance = frame.source_provenance.clone().expect("suffix provenance");
                runtime
            })
            .collect();
        for (frame, runtime) in frames.iter_mut().zip(&snapshot.frames) {
            frame.runtime_frame_id = Some(runtime.id);
        }
        crate::transcript::PreparedLogicalCheckpoint {
            expected_journal_frontier: boundary_sequence,
            expected_branch_id: agent.runtime_snapshot.active_context.branch_id.clone(),
            event,
            projected_snapshot: snapshot,
            projected_protocol_frames: frames,
            projected_workflow: Some(crate::transcript::CheckpointWorkflowProjection {
                todos: agent.turn.workflow.todos.clone(),
                auto_continue: agent.turn.workflow.auto_continue.clone(),
            }),
        }
    }

    fn authorized_checkpoint_agent(
        protocol: ApiProtocol,
    ) -> (Agent<OpenAIConfig>, Vec<PromptMessage>) {
        let mut agent = prepared_group_16_agent();
        agent.default_protocol = protocol;
        let metadata = agent.model_catalog.get_mut("group-16").expect("test model");
        metadata.context_window = Some(16_384);
        metadata.effective_input_limit_tokens = Some(12_000);
        agent
            .replace_history(vec![HistoryItem::user(format!(
                "phase1c unsafe protected context {}",
                "history ".repeat(16_000)
            ))])
            .expect("legal unsafe history");
        agent.set_logical_checkpoint_config(crate::config::LogicalCheckpointConfig {
            enabled: true,
            automatic: true,
            max_automatic_per_turn: 1,
            ..Default::default()
        });
        let prelude = agent.prepare_turn_prelude("phase1c unsafe request");
        agent.turn.current_turn_start_index = Some(0);
        agent.runtime_snapshot.current_turn_id = Some(agent.turn.turn_id);
        agent.runtime_snapshot.current_segment_id = Some(1);
        agent.runtime_snapshot.leaf_sequence = Some(9);
        agent.runtime_snapshot.latest_model = Some(agent.model.clone());
        let candidate = checkpoint_candidate_for(&agent);
        agent.set_logical_checkpoint_candidate_provider(Arc::new(move || Ok(candidate.clone())));
        (agent, prelude)
    }

    fn warm_append_checkpoint_agent(
        protocol: ApiProtocol,
        max_automatic_per_turn: u8,
    ) -> (Agent<OpenAIConfig>, Vec<PromptMessage>) {
        let mut agent = prepared_group_16_agent();
        agent.default_protocol = protocol;
        let metadata = agent.model_catalog.get_mut("group-16").expect("test model");
        metadata.context_window = Some(16_384);
        metadata.effective_input_limit_tokens = Some(12_000);
        agent
            .replace_history(vec![HistoryItem::user(format!(
                "phase1c warm active epoch {}",
                "history ".repeat(2_000)
            ))])
            .expect("legal warm history");
        agent.prelude.push(PromptMessage::system(format!(
            "stable {}",
            "prelude ".repeat(1_600)
        )));
        agent.set_logical_checkpoint_config(crate::config::LogicalCheckpointConfig {
            enabled: true,
            automatic: true,
            max_automatic_per_turn,
            ..Default::default()
        });
        let prelude = agent.prepare_turn_prelude("phase1c warm append request");
        agent.turn.current_turn_start_index = Some(agent.history.len());
        agent.runtime_snapshot.current_turn_id = Some(agent.turn.turn_id);
        agent.runtime_snapshot.current_segment_id = Some(1);
        agent.runtime_snapshot.leaf_sequence = Some(9);
        agent.runtime_snapshot.latest_model = Some(agent.model.clone());
        (agent, prelude)
    }

    async fn commit_safe_active_epoch(
        agent: &mut Agent<OpenAIConfig>,
        protocol: ApiProtocol,
        prelude: &[PromptMessage],
        tools: &[crate::request_builder::ToolSpec],
    ) {
        let mut on_event = |_| std::future::ready(Ok(()));
        let prepared = prepare_canonical_protocol_stream_request(
            agent,
            protocol,
            prelude,
            &mut 0,
            tools,
            &mut on_event,
        )
        .await
        .expect("safe warm request prepares");
        let preview = prepared.epoch_preview;
        assert!(matches!(preview.transition, ActiveEpochTransition::Cold));
        agent.commit_active_epoch(preview);
        assert!(agent.active_epoch.is_some(), "warm epoch is committed");
    }

    fn append_complete_overflow_tool_group(agent: &mut Agent<OpenAIConfig>, sequence: u8) -> u64 {
        use crate::protocol_frames::{ProtocolFrame, ProtocolFrameItem};
        use crate::runtime_context::{RuntimeFrameProvenance, RuntimeSource, SourceSpan};

        let mut append = |item, source_id: String, source_sequence| {
            let mut frame = ProtocolFrame::derived(item);
            frame.source_provenance = Some(
                RuntimeFrameProvenance::new(RuntimeSource::Transcript)
                    .with_source_id(source_id)
                    .with_span(
                        SourceSpan::new(source_sequence, source_sequence).expect("source span"),
                    ),
            );
            agent.append_protocol_frame(frame)
        };
        let payload = "overflow ".repeat(2_000);
        let calls = ["a", "b"].map(|suffix| HistoryToolCall {
            call_id: format!("phase1c-overflow-{sequence}-{suffix}"),
            name: "group_16_tool".into(),
            arguments_json: format!(r#"{{"payload":"{payload}"}}"#),
        });
        append(
            ProtocolFrameItem::AssistantToolCalls {
                text: Some("phase1c complete overflowing tool group".into()),
                calls: calls.to_vec(),
            },
            format!("phase1c-overflow-{sequence}:calls"),
            10 + u64::from(sequence) * 3,
        )
        .expect("tool group is protocol-valid");
        for (index, call) in calls.into_iter().enumerate() {
            append(
                ProtocolFrameItem::ToolOutput {
                    call_id: call.call_id,
                    output_json:
                        r#"{"ok":true,"tool":"group_16_tool","data":{"status":"complete"}}"#.into(),
                },
                format!("phase1c-overflow-{sequence}:output-{index}"),
                11 + u64::from(sequence) * 3 + index as u64,
            )
            .expect("tool output closes the complete group");
        }
        drop(append);
        agent.turn.automatic_checkpoint.begin_complete_boundary()
    }

    fn logical_checkpoint_events(events: &[AgentEvent]) -> usize {
        events
            .iter()
            .filter(|event| matches!(event, AgentEvent::LogicalCheckpoint { .. }))
            .count()
    }

    fn serialized_request_bytes(build: &crate::request_builder::BuildResult) -> Vec<u8> {
        match &build.request {
            BuiltRequest::Responses(request) => serde_json::to_vec(request),
            BuiltRequest::ResponsesCompatible(request)
            | BuiltRequest::CompletionsCompatible(request) => serde_json::to_vec(request),
            BuiltRequest::Completions(request) => serde_json::to_vec(request),
        }
        .expect("provider request serializes")
    }

    fn assert_complete_checkpoint_successor_request(
        protocol: ApiProtocol,
        build: &crate::request_builder::BuildResult,
    ) {
        let request = String::from_utf8(serialized_request_bytes(build)).expect("request is JSON");
        let call = request
            .find("phase1c-retained-call")
            .expect("retained tool call id is serialized");
        let output = request
            .find("checkpoint_output")
            .expect("matching retained tool output is serialized");
        assert!(call < output, "tool call must precede its matching output");
        assert!(request.contains("group_16_tool"));
        assert!(request.contains("phase1c-checkpoint"));
        assert!(!request.contains("phase1c unsafe protected context"));
        match protocol {
            ApiProtocol::Responses => {
                assert!(request.contains("function_call"));
                assert!(request.contains("function_call_output"));
            }
            ApiProtocol::Completions => {
                assert!(request.contains("tool_calls"));
                assert!(request.contains("tool_call_id"));
            }
        }
    }

    async fn assert_unsafe_preparation_without_boundary_is_side_effect_free(protocol: ApiProtocol) {
        let mut agent = prepared_group_16_agent();
        let prelude = agent.prepare_turn_prelude("unsafe preparation");
        let tools = agent.tool_definitions();
        let provider_calls = Arc::new(AtomicUsize::new(0));
        agent.set_logical_checkpoint_candidate_provider({
            let provider_calls = Arc::clone(&provider_calls);
            Arc::new(move || {
                provider_calls.fetch_add(1, Ordering::SeqCst);
                Err(anyhow!(
                    "candidate provider must not run without a boundary"
                ))
            })
        });
        let history = agent.history.clone();
        let frames = agent.protocol_frames.clone();
        let snapshot = agent.runtime_snapshot.clone();
        let active_epoch = agent.active_epoch.clone();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured_events = Arc::clone(&events);
        let mut on_event = move |event| {
            captured_events.lock().expect("event lock").push(event);
            std::future::ready(Ok(()))
        };

        let result = prepare_canonical_protocol_stream_request(
            &mut agent,
            protocol,
            &prelude,
            &mut 0,
            &tools,
            &mut on_event,
        )
        .await;
        let Err(error) = result else {
            panic!("the intentionally truncated candidate needs a fresh boundary");
        };

        assert!(
            error
                .to_string()
                .contains("without a fresh complete tool boundary")
        );
        assert_eq!(agent.history, history);
        assert_eq!(agent.protocol_frames, frames);
        assert_eq!(agent.runtime_snapshot, snapshot);
        assert_eq!(agent.active_epoch, active_epoch);
        assert!(events.lock().expect("event lock").is_empty());
        assert_eq!(provider_calls.load(Ordering::SeqCst), 0);
        assert!(
            agent
                .turn
                .automatic_checkpoint
                .current_boundary_generation
                .is_none()
        );
    }

    #[test]
    fn phase1c_complete_boundary_is_single_use_and_never_exists_for_partial_or_restored_work() {
        let mut scheduler = AutomaticCheckpointSchedulerState::default();
        assert_eq!(
            scheduler.consume_complete_boundary(),
            None,
            "first preparation has no boundary"
        );
        assert_eq!(
            scheduler.current_boundary_generation, None,
            "no-tool work has no boundary"
        );

        // Restored turns deliberately begin with the default scheduler state.
        let restored = AutomaticCheckpointSchedulerState::default();
        assert_eq!(restored.current_boundary_generation, None);

        let boundary = scheduler.begin_complete_boundary();
        assert_eq!(scheduler.current_boundary_generation, Some(boundary));
        assert_eq!(scheduler.consume_complete_boundary(), Some(boundary));
        assert_eq!(
            scheduler.consume_complete_boundary(),
            None,
            "a preparation consumes the capability"
        );

        // Starting a partial group does not mint a boundary; only its completed
        // ToolCallBatchFinished path is allowed to call begin_complete_boundary.
        scheduler.current_boundary_generation = None;
        assert_eq!(scheduler.consume_complete_boundary(), None);
        let completed = scheduler.begin_complete_boundary();
        assert_eq!(scheduler.consume_complete_boundary(), Some(completed));
        assert_eq!(scheduler.last_consumed_boundary, Some(completed));
    }

    #[tokio::test]
    async fn checkpoint_authorization_matrix_rejects_without_mutation() {
        enum Case {
            PolicyDisabled,
            AutomaticDisabled,
            MaxCommits,
            Attempted,
            MissingProvider,
        }

        for case in [
            Case::PolicyDisabled,
            Case::AutomaticDisabled,
            Case::MaxCommits,
            Case::Attempted,
            Case::MissingProvider,
        ] {
            let mut agent = prepared_group_16_agent();
            let prelude = agent.prepare_turn_prelude("phase1c authorization matrix");
            let tools = agent.tool_definitions();
            agent.set_logical_checkpoint_config(crate::config::LogicalCheckpointConfig {
                enabled: true,
                automatic: true,
                max_automatic_per_turn: 1,
                ..Default::default()
            });
            let boundary = agent.turn.automatic_checkpoint.begin_complete_boundary();
            match case {
                Case::PolicyDisabled => agent.automatic_checkpoint_policy.enabled = false,
                Case::AutomaticDisabled => agent.logical_checkpoint_control.set_config(
                    crate::config::LogicalCheckpointConfig {
                        enabled: true,
                        automatic: false,
                        ..Default::default()
                    },
                ),
                Case::MaxCommits => agent.turn.automatic_checkpoint.commits = 1,
                Case::Attempted => agent.turn.automatic_checkpoint.mark_attempted(boundary),
                Case::MissingProvider => {}
            }
            let history = agent.history.clone();
            let frames = agent.protocol_frames.clone();
            let snapshot = agent.runtime_snapshot.clone();
            let events = Arc::new(Mutex::new(Vec::new()));
            let captured = Arc::clone(&events);
            let mut on_event = move |event| {
                captured.lock().expect("event lock").push(event);
                std::future::ready(Ok(()))
            };
            let _run = agent.logical_checkpoint_control.begin_run();

            assert!(
                prepare_canonical_protocol_stream_request(
                    &mut agent,
                    ApiProtocol::Responses,
                    &prelude,
                    &mut 0,
                    &tools,
                    &mut on_event,
                )
                .await
                .is_err()
            );
            assert_eq!(agent.history, history);
            assert_eq!(agent.protocol_frames, frames);
            assert_eq!(agent.runtime_snapshot, snapshot);
            assert_eq!(
                logical_checkpoint_events(&events.lock().expect("event lock")),
                0
            );
        }
    }

    #[tokio::test]
    async fn active_epoch_prepared_callback_failure_does_not_commit_epoch() {
        let mut agent = prepared_build_agent();
        let metadata = agent
            .model_catalog
            .get_mut("group-16")
            .expect("model metadata");
        metadata.context_window = Some(32_000);
        metadata.effective_input_limit_tokens = Some(30_000);
        let prepared_metadata_seen = Arc::new(AtomicUsize::new(0));
        let seen = Arc::clone(&prepared_metadata_seen);
        let result = run_responses_stream_async(
            &mut agent,
            UserMessageContent::new("callback failure", Vec::new()),
            "callback failure",
            |_| std::future::ready(Ok(())),
            move |event| {
                if matches!(event, AgentEvent::TokenUsageUpdated { .. }) {
                    seen.fetch_add(1, Ordering::SeqCst);
                    return std::future::ready(Err(anyhow!("prepared callback failed")));
                }
                std::future::ready(Ok(()))
            },
            |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
        )
        .await;

        let error = result.expect_err("prepared callback must stop the send");
        assert!(
            error.to_string().contains("prepared callback failed"),
            "unexpected stream failure: {error:#}"
        );
        assert_eq!(prepared_metadata_seen.load(Ordering::SeqCst), 1);
        assert!(
            agent.active_epoch.is_none(),
            "preview must not commit before prepared metadata succeeds"
        );
    }

    async fn assert_same_build_preflight_and_request(protocol: ApiProtocol) {
        let mut agent = prepared_group_16_agent();
        let prelude = agent.prepare_turn_prelude("GROUP-16 volatile runtime material");
        let tools = agent.tool_definitions();
        let protected_start_index = agent.history.len() - 1;
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured_events = Arc::clone(&events);
        let mut on_event = move |event| {
            captured_events.lock().expect("event lock").push(event);
            std::future::ready(Ok(()))
        };

        // This is the only preparation/build in this lane. All assertions below use it.
        let prepared = compaction::prepare_request_build(
            &mut agent,
            protocol,
            &prelude,
            protected_start_index,
            &tools,
            &mut on_event,
        )
        .await
        .expect("prepared request build");
        let build = prepared.build;
        let _span = tracing::info_span!("group_16_same_build");
        emit_prepared_request_metadata(&build, &mut on_event)
            .await
            .expect("preflight metadata");

        let event = events
            .lock()
            .expect("event lock")
            .pop()
            .expect("preflight token event");
        let AgentEvent::TokenUsageUpdated {
            used_tokens,
            context_window_tokens,
            input_tokens,
            output_tokens,
            cached_tokens,
            cache_report,
        } = event
        else {
            panic!("expected preflight token event");
        };
        assert_eq!(used_tokens, build.budget.estimated_request_tokens);
        assert_eq!(input_tokens, build.budget.estimated_request_tokens);
        assert_eq!(context_window_tokens, build.budget.context_window_tokens);
        assert_eq!(output_tokens, 0);
        assert_eq!(cached_tokens, 0);
        assert_eq!(cache_report, Some(CacheUsageReport::from_build(&build)));
        assert_eq!(
            cache_report.and_then(|report| report.actual_cached_tokens),
            None
        );

        let token_report = build.prompt_plan.token_report();
        assert_eq!(
            build.budget.plan_total_prompt_tokens,
            token_report.total_prompt_tokens
        );
        assert_eq!(
            build.budget.plan_stable_prompt_tokens,
            token_report.stable_prompt_tokens
        );
        assert_eq!(
            build.budget.plan_volatile_prompt_tokens,
            token_report.volatile_prompt_tokens
        );
        assert_eq!(
            build.budget.plan_cacheable_prefix_tokens,
            token_report.cacheable_prefix_tokens
        );
        assert_eq!(
            build.budget.plan_stable_after_boundary_tokens,
            token_report.stable_after_boundary_tokens
        );
        assert_eq!(build.prompt_plan.protocol, protocol);
        assert!(
            build.budget.dropped_history_items > 0,
            "fixture must truncate"
        );
        assert!(matches!(
            (protocol, &build.request),
            (ApiProtocol::Responses, BuiltRequest::Responses(_))
                | (ApiProtocol::Responses, BuiltRequest::ResponsesCompatible(_))
                | (ApiProtocol::Completions, BuiltRequest::Completions(_))
                | (
                    ApiProtocol::Completions,
                    BuiltRequest::CompletionsCompatible(_)
                )
        ));

        let request = match &build.request {
            BuiltRequest::Responses(request) => serde_json::to_value(request),
            BuiltRequest::ResponsesCompatible(request) => Ok(request.clone()),
            BuiltRequest::Completions(request) => serde_json::to_value(request),
            BuiltRequest::CompletionsCompatible(request) => Ok(request.clone()),
        }
        .expect("same build request serializes");
        let request_json = serde_json::to_string(&request).expect("request json");
        assert!(!request_json.contains("GROUP-16-DROPPED-SENTINEL"));
        // A segment can render to several Responses items (for example, an
        // assistant tool-call segment), and provider JSON is not a text
        // stream. Compare the protocol-rendered representation instead.
        let serialized_plan = crate::request_builder::canonical_cache_input(
            "group-16",
            protocol,
            "group-16",
            &build.prompt_plan.segments,
            &tools,
            true,
        );
        let rendered_items = match protocol {
            ApiProtocol::Responses => &request["input"],
            ApiProtocol::Completions => &request["messages"],
        };
        assert_eq!(
            rendered_items, &serialized_plan["items"],
            "provider request must serialize the same selected prompt plan"
        );
        assert_eq!(
            request["tools"], serialized_plan["tools"],
            "provider request must serialize the same tools"
        );
        assert_eq!(
            request["parallel_tool_calls"], serialized_plan["input_shape"]["parallel_tool_calls"],
            "provider request must preserve the rendered input shape"
        );
        assert_eq!(request["model"], "group-16");
        assert!(request_json.contains("GROUP-16 volatile runtime material"));
        assert!(request_json.contains("group_16_tool"));

        let cache = CacheUsageReport::from_build(&build);
        assert!(cache.hint_serialized);
        assert_eq!(
            cache.local_prefix_fingerprint,
            build.cache.local_prefix_fingerprint
        );
        assert_eq!(cache.routing_key, build.cache.routing_key);
        assert_eq!(
            request.get("prompt_cache_key").and_then(Value::as_str),
            build.cache.routing_key.as_deref()
        );
        match protocol {
            ApiProtocol::Responses => {
                assert_eq!(
                    cache.retention_sent,
                    Some(crate::config::PromptCacheRetention::InMemory)
                );
                assert_eq!(request["prompt_cache_retention"], "in_memory");
            }
            ApiProtocol::Completions => {
                assert_eq!(cache.retention_sent, None);
                assert!(request.get("prompt_cache_retention").is_none());
            }
        }
    }

    #[tokio::test]
    async fn group_16_responses_preflight_and_provider_request_share_one_build_result() {
        assert_same_build_preflight_and_request(ApiProtocol::Responses).await;
    }

    #[tokio::test]
    async fn group_16_chat_preflight_and_provider_request_share_one_build_result() {
        assert_same_build_preflight_and_request(ApiProtocol::Completions).await;
    }

    fn cache_report() -> CacheUsageReport {
        CacheUsageReport {
            configured: true,
            hint_serialized: true,
            retention_sent: None,
            stable_prefix_segments: 1,
            stable_prompt_tokens: 100,
            volatile_prompt_tokens: 20,
            cacheable_prefix_tokens: 80,
            stable_after_boundary_tokens: 20,
            local_prefix_fingerprint: Some("prefix".into()),
            routing_key: Some("route".into()),
            actual_cached_tokens: None,
        }
    }

    #[test]
    fn response_usage_converter_reports_provider_cached_tokens() {
        let usage: ResponseUsage = serde_json::from_value(serde_json::json!({
            "input_tokens": 100,
            "input_tokens_details": { "cached_tokens": 80 },
            "output_tokens": 20,
            "output_tokens_details": { "reasoning_tokens": 0 },
            "total_tokens": 120,
        }))
        .expect("response usage fixture");

        let AgentEvent::TokenUsageUpdated {
            cached_tokens,
            cache_report,
            ..
        } = token_usage_event_from_response_usage(&usage, 1_000, &cache_report())
        else {
            panic!("expected token usage event");
        };
        assert_eq!(cached_tokens, 80);
        assert_eq!(
            cache_report.and_then(|report| report.actual_cached_tokens),
            Some(cached_tokens)
        );
    }

    #[test]
    fn chat_usage_converter_reports_provider_cached_tokens() {
        let usage: CompletionUsage = serde_json::from_value(serde_json::json!({
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "prompt_tokens_details": { "cached_tokens": 80 },
        }))
        .expect("completion usage fixture");

        let AgentEvent::TokenUsageUpdated {
            cached_tokens,
            cache_report,
            ..
        } = token_usage_event_from_completion_usage(&usage, 1_000, &cache_report())
        else {
            panic!("expected token usage event");
        };
        assert_eq!(cached_tokens, 80);
        assert_eq!(
            cache_report.and_then(|report| report.actual_cached_tokens),
            Some(cached_tokens)
        );
    }

    fn assert_actual_report_only_adds_provider_cache_hit(
        event: AgentEvent,
        prepared: &CacheUsageReport,
    ) {
        let AgentEvent::TokenUsageUpdated { cache_report, .. } = event else {
            panic!("expected token usage event");
        };
        let actual = cache_report.expect("actual cache report");
        assert_eq!(actual.actual_cached_tokens, Some(80));
        let mut without_actual = actual;
        without_actual.actual_cached_tokens = None;
        assert_eq!(&without_actual, prepared);
    }

    #[test]
    fn responses_and_chat_actual_reports_only_differ_from_prepared_by_provider_cache_hit() {
        let prepared = cache_report();
        let response_usage: ResponseUsage = serde_json::from_value(serde_json::json!({
            "input_tokens": 100, "input_tokens_details": { "cached_tokens": 80 },
            "output_tokens": 20, "output_tokens_details": { "reasoning_tokens": 0 }, "total_tokens": 120,
        }))
        .expect("response usage fixture");
        let chat_usage: CompletionUsage = serde_json::from_value(serde_json::json!({
            "prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120,
            "prompt_tokens_details": { "cached_tokens": 80 },
        }))
        .expect("chat usage fixture");

        assert_actual_report_only_adds_provider_cache_hit(
            token_usage_event_from_response_usage(&response_usage, 1_000, &prepared),
            &prepared,
        );
        assert_actual_report_only_adds_provider_cache_hit(
            token_usage_event_from_completion_usage(&chat_usage, 1_000, &prepared),
            &prepared,
        );
    }

    #[test]
    fn no_prefix_actual_report_keeps_provider_cache_hit_independent() {
        let prepared = CacheUsageReport {
            configured: true,
            hint_serialized: false,
            retention_sent: None,
            stable_prefix_segments: 0,
            stable_prompt_tokens: 0,
            volatile_prompt_tokens: 100,
            cacheable_prefix_tokens: 0,
            stable_after_boundary_tokens: 0,
            local_prefix_fingerprint: None,
            routing_key: None,
            actual_cached_tokens: None,
        };
        let usage: CompletionUsage = serde_json::from_value(serde_json::json!({
            "prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120,
            "prompt_tokens_details": { "cached_tokens": 80 },
        }))
        .expect("chat usage fixture");
        assert_actual_report_only_adds_provider_cache_hit(
            token_usage_event_from_completion_usage(&usage, 1_000, &prepared),
            &prepared,
        );
    }

    #[test]
    fn iteration_budget_is_unbounded_when_limit_omitted() {
        ensure_iteration_budget(None, 0).expect("omitted limit should not fail");
        ensure_iteration_budget(None, usize::MAX).expect("omitted limit should remain unbounded");
    }

    #[test]
    fn iteration_budget_fails_only_when_explicit_limit_exceeded() {
        ensure_iteration_budget(Some(2), 0).expect("first iteration should pass");
        ensure_iteration_budget(Some(2), 1).expect("second iteration should pass");
        let error =
            ensure_iteration_budget(Some(2), 2).expect_err("explicit limit should fail-fast");
        assert!(
            error
                .to_string()
                .contains("too many agent iterations (max 2)")
        );
    }
}
