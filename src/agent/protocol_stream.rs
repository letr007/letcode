use super::*;
use crate::langfuse_trace;
use crate::user_content::UserMessageContent;
use tracing::Instrument;

const STREAM_INTERRUPT_MESSAGE: &str = "Model stream interrupted";
const STREAM_INTERRUPT_ACTION: &str = "Continuing with a fresh model iteration";

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
    E: FnMut(AgentEvent) -> Efut,
    A: FnMut(PermissionRequest) -> Afut,
    Dfut: Future<Output = Result<()>>,
    Efut: Future<Output = Result<()>>,
    Afut: Future<Output = Result<bool>>,
{
    let turn_prelude = agent.prepare_turn_prelude(user_input);
    let mut protected_start_index = agent.history.len();
    agent.history.push(HistoryItem::user_content(user_content));
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
        let iteration_span = langfuse_trace::llm_iteration_span(
            turn_id,
            "responses",
            &agent.model,
            iteration,
            agent.history.len(),
            tool_call_count,
            tool_definitions.len(),
        );
        protected_start_index = compaction::preflight_compact_context(
            agent,
            &turn_prelude,
            protected_start_index,
            &tool_definitions,
            &mut on_event,
        )
        .await?;
        let context_view = agent.context_view_for_request()?;
        let build = build_request(RequestBuilderInput {
            protocol: ApiProtocol::Responses,
            model_id: &agent.model,
            model: agent.active_model_metadata(),
            prelude: &turn_prelude,
            history: &agent.history,
            protected_start_index,
            tools: &tool_definitions,
            evidence: &agent.evidence,
            context_view: context_view.as_ref(),
        })?;
        on_event(AgentEvent::TokenUsageUpdated {
            used_tokens: build.budget.estimated_request_tokens,
            context_window_tokens: build.budget.context_window_tokens,
            input_tokens: build.budget.estimated_request_tokens,
            output_tokens: 0,
            cached_tokens: 0,
        })
        .await?;
        langfuse_trace::record_llm_request_budget(&iteration_span, &build.budget);
        if build.budget.truncated {
            debug!(
                model = %agent.model,
                original_history_items = build.budget.original_history_items,
                retained_history_items = build.budget.retained_history_items,
                dropped_history_items = build.budget.dropped_history_items,
                context_window_tokens = build.budget.context_window_tokens,
                input_budget_tokens = build.budget.input_budget_tokens,
                estimated_request_tokens = build.budget.estimated_request_tokens,
                "request history truncated to fit budget"
            );
        }

        let BuiltRequest::Responses(request) = build.request else {
            return Err(anyhow!("request builder returned non-responses request"));
        };

        let mut attempt = 1;
        let (response, mut turn_text, completed_reasoning_ids) = 'retry_response_stream: loop {
            let mut stream = match agent
                .client
                .responses()
                .create_stream(request.clone())
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
                    tokio::time::sleep(delay).await;
                    attempt += 1;
                    continue 'retry_response_stream;
                }
                Err(error) => {
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
                let event = match event {
                    Ok(event) => event,
                    Err(error) if is_ignorable_response_lifecycle_deserialize_error(&error) => {
                        warn!(error = %error, "ignored malformed response lifecycle stream event");
                        continue;
                    }
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
                    Err(error) => return Err(error.into()),
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
                            langfuse_trace::record_llm_usage(
                                &iteration_span,
                                usage.input_tokens as u64,
                                usage.output_tokens as u64,
                                usage.input_tokens_details.cached_tokens as u64,
                                usage.total_tokens as u64,
                            );
                            on_event(token_usage_event_from_response_usage(
                                usage,
                                build.budget.context_window_tokens,
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
                None => return Err(anyhow!("stream ended without response.completed")),
            };
            break 'retry_response_stream (response, turn_text, completed_reasoning_ids);
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
                .history
                .push(HistoryItem::assistant(turn_text.clone()));

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

        agent.append_assistant_tool_calls(&turn_text, &tool_calls);

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
    E: FnMut(AgentEvent) -> Efut,
    A: FnMut(PermissionRequest) -> Afut,
    Dfut: Future<Output = Result<()>>,
    Efut: Future<Output = Result<()>>,
    Afut: Future<Output = Result<bool>>,
{
    let turn_prelude = agent.prepare_turn_prelude(user_input);
    let mut protected_start_index = agent.history.len();
    agent.history.push(HistoryItem::user_content(user_content));
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
        let iteration_span = langfuse_trace::llm_iteration_span(
            turn_id,
            "chat_completions",
            &agent.model,
            iteration,
            agent.history.len(),
            tool_call_count,
            tool_definitions.len(),
        );
        protected_start_index = compaction::preflight_compact_context(
            agent,
            &turn_prelude,
            protected_start_index,
            &tool_definitions,
            &mut on_event,
        )
        .await?;
        let context_view = agent.context_view_for_request()?;

        let build = build_request(RequestBuilderInput {
            protocol: ApiProtocol::Completions,
            model_id: &agent.model,
            model: agent.active_model_metadata(),
            prelude: &turn_prelude,
            history: &agent.history,
            protected_start_index,
            tools: &tool_definitions,
            evidence: &agent.evidence,
            context_view: context_view.as_ref(),
        })?;
        on_event(AgentEvent::TokenUsageUpdated {
            used_tokens: build.budget.estimated_request_tokens,
            context_window_tokens: build.budget.context_window_tokens,
            input_tokens: build.budget.estimated_request_tokens,
            output_tokens: 0,
            cached_tokens: 0,
        })
        .await?;
        langfuse_trace::record_llm_request_budget(&iteration_span, &build.budget);
        if build.budget.truncated {
            debug!(
                model = %agent.model,
                original_history_items = build.budget.original_history_items,
                retained_history_items = build.budget.retained_history_items,
                dropped_history_items = build.budget.dropped_history_items,
                context_window_tokens = build.budget.context_window_tokens,
                input_budget_tokens = build.budget.input_budget_tokens,
                estimated_request_tokens = build.budget.estimated_request_tokens,
                "request history truncated to fit budget"
            );
        }
        let BuiltRequest::Completions(request) = build.request else {
            return Err(anyhow!("request builder returned non-completions request"));
        };

        let mut attempt = 1;
        'retry_chat_stream: loop {
            let response = send_compatible_chat_completion_stream(
                &agent.client,
                &request,
                &agent.retry_config,
                &mut attempt,
            )
            .await
            .with_context(|| {
                request_creation_failure_context(
                    "streamed chat completion",
                    &agent.model,
                    agent.active_model_metadata(),
                    &build.budget,
                )
            })?;
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
                    Err(error) => return Err(error.into()),
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
                            return Err(error).with_context(|| {
                                format!("failed to parse chat completions stream event: {data}")
                            });
                        }
                    };
                    if let Some(usage) = &response.usage {
                        stream_had_side_effect = true;
                        langfuse_trace::record_llm_usage(
                            &iteration_span,
                            usage.prompt_tokens as u64,
                            usage.completion_tokens as u64,
                            usage
                                .prompt_tokens_details
                                .as_ref()
                                .and_then(|details| details.cached_tokens)
                                .unwrap_or(0) as u64,
                            usage.total_tokens as u64,
                        );
                        on_event(token_usage_event_from_completion_usage(
                            usage,
                            build.budget.context_window_tokens,
                        ))
                        .await?;
                    }
                    for choice in response.choices {
                        if choice.index != 0 {
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
                        return Err(error).with_context(|| {
                            format!("failed to parse chat completions stream event: {data}")
                        });
                    }
                };
                if let Some(usage) = &response.usage {
                    stream_had_side_effect = true;
                    langfuse_trace::record_llm_usage(
                        &iteration_span,
                        usage.prompt_tokens as u64,
                        usage.completion_tokens as u64,
                        usage
                            .prompt_tokens_details
                            .as_ref()
                            .and_then(|details| details.cached_tokens)
                            .unwrap_or(0) as u64,
                        usage.total_tokens as u64,
                    );
                    on_event(token_usage_event_from_completion_usage(
                        usage,
                        build.budget.context_window_tokens,
                    ))
                    .await?;
                }
                for choice in response.choices {
                    if choice.index != 0 {
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
                return Err(error);
            }
            let finish_reasons_label = format!("{finish_reasons:?}");

            if !has_tool_calls {
                if final_text.is_empty() {
                    final_text = "No response content".to_string();
                }

                agent
                    .history
                    .push(HistoryItem::assistant(turn_text.clone()));

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
            agent.append_assistant_tool_calls(&turn_text, &tool_calls);

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
    result
}

pub(super) fn is_ignorable_response_lifecycle_deserialize_error(error: &OpenAIError) -> bool {
    let OpenAIError::JSONDeserialize(source, content) = error else {
        return false;
    };

    source.to_string().contains("missing field `model`")
        && serde_json::from_str::<Value>(content)
            .ok()
            .and_then(|value| value.get("type").and_then(Value::as_str).map(str::to_owned))
            .as_deref()
            .is_some_and(|event_type| {
                matches!(event_type, "response.created" | "response.in_progress")
            })
}

pub(super) async fn send_compatible_chat_completion_stream<C: Config>(
    client: &Client<C>,
    request: &impl Serialize,
    retry_config: &RetryConfig,
    attempt: &mut usize,
) -> Result<reqwest::Response> {
    let config = client.config();
    let url = config.url("/chat/completions");
    let http = reqwest_client_for_url(&url)?;

    loop {
        let response = match http
            .post(url.clone())
            .query(&config.query())
            .headers(config.headers())
            .json(request)
            .send()
            .await
        {
            Ok(response) => response,
            Err(error) if should_retry_reqwest_error(retry_config, *attempt, &error) => {
                let delay = retry_delay(retry_config, *attempt);
                warn!(
                    attempt = *attempt,
                    max_attempts = retry_config.max_attempts,
                    delay_ms = delay.as_millis(),
                    error = %error,
                    "retrying chat completions stream creation"
                );
                tokio::time::sleep(delay).await;
                *attempt += 1;
                continue;
            }
            Err(error) => {
                return Err(error).context("failed to create streamed chat completion");
            }
        };

        let status = response.status();
        if status.is_success() {
            return Ok(response);
        }

        if should_retry_http_status(retry_config, *attempt, status) {
            let delay = retry_delay_from_headers(retry_config, *attempt, response.headers());
            warn!(
                attempt = *attempt,
                max_attempts = retry_config.max_attempts,
                delay_ms = delay.as_millis(),
                status = %status,
                "retrying chat completions stream creation after transient status"
            );
            tokio::time::sleep(delay).await;
            *attempt += 1;
            continue;
        }

        let message = response
            .text()
            .await
            .unwrap_or_else(|error| format!("failed to read error body: {error}"));
        bail!("chat completions request failed with status {status}: {message}");
    }
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
    pub(super) choices: Vec<CompatibleChatChoiceStream>,
    pub(super) usage: Option<CompletionUsage>,
}

pub(super) fn token_usage_event_from_response_usage(
    usage: &ResponseUsage,
    context_window_tokens: u64,
) -> AgentEvent {
    AgentEvent::TokenUsageUpdated {
        used_tokens: usage.total_tokens as u64,
        context_window_tokens,
        input_tokens: usage.input_tokens as u64,
        output_tokens: usage.output_tokens as u64,
        cached_tokens: usage.input_tokens_details.cached_tokens as u64,
    }
}

pub(super) fn token_usage_event_from_completion_usage(
    usage: &CompletionUsage,
    context_window_tokens: u64,
) -> AgentEvent {
    AgentEvent::TokenUsageUpdated {
        used_tokens: usage.total_tokens as u64,
        context_window_tokens,
        input_tokens: usage.prompt_tokens as u64,
        output_tokens: usage.completion_tokens as u64,
        cached_tokens: usage
            .prompt_tokens_details
            .as_ref()
            .and_then(|details| details.cached_tokens)
            .unwrap_or(0) as u64,
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
        agent
            .history
            .push(HistoryItem::assistant(turn_text.to_string()));
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

    agent.history.push(HistoryItem::internal_continuation(
        build_stream_interrupt_continuation(turn_text, pending_tool_calls),
    ));

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
    use super::ensure_iteration_budget;

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
