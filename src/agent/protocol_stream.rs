use super::*;
use crate::user_content::UserMessageContent;

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

    let mut final_text = String::new();
    let mut tool_call_count = 0;
    let mut continuation_count = 0;

    for iteration in 0..agent.max_iterations {
        debug!(
            iteration,
            model = %agent.model,
            history_len = agent.history.len(),
            tool_call_count,
            max_tool_calls = agent.max_tool_calls,
            "creating streamed response"
        );

        let tool_definitions = agent.tool_definitions();
        protected_start_index = compaction::preflight_compact_context(
            agent,
            &turn_prelude,
            protected_start_index,
            &tool_definitions,
            &mut on_event,
        )
        .await?;
        let build = build_request(RequestBuilderInput {
            protocol: ApiProtocol::Responses,
            model_id: &agent.model,
            model: agent.active_model_metadata(),
            prelude: &turn_prelude,
            history: &agent.history,
            protected_start_index,
            tools: &tool_definitions,
            evidence: &agent.evidence,
        })?;
        on_event(AgentEvent::TokenUsageUpdated {
            used_tokens: build.budget.estimated_request_tokens,
            context_window_tokens: build.budget.context_window_tokens,
            input_tokens: build.budget.estimated_request_tokens,
            output_tokens: 0,
            cached_tokens: 0,
        })
        .await?;
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
                Err(error) => return Err(error.into()),
            };

            let mut completed_response: Option<Response> = None;
            let mut completed_reasoning_ids = HashSet::new();
            let mut pending_tool_calls = HashSet::new();
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
                            stream_had_side_effect = true;
                            emit_tool_call_pending_if_ready(
                                &mut pending_tool_calls,
                                &call.call_id,
                                &call.name,
                                &mut on_event,
                            )
                            .await?;
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
                            ))
                            .await?;
                        }
                        completed_response = Some(event.response);
                    }
                    ResponseStreamEvent::ResponseFailed(event) => {
                        error!(response = ?event.response, "response failed");
                        return Err(anyhow!("response failed: {:#?}", event.response));
                    }
                    ResponseStreamEvent::ResponseIncomplete(event) => {
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

    Err(anyhow!(
        "stopped: too many agent iterations (max {})",
        agent.max_iterations
    ))
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

    let mut final_text = String::new();
    let mut tool_call_count = 0;
    let mut continuation_count = 0;

    'agent_iteration: for iteration in 0..agent.max_iterations {
        debug!(
            iteration,
            model = %agent.model,
            history_len = agent.history.len(),
            tool_call_count,
            max_tool_calls = agent.max_tool_calls,
            "creating streamed chat completion"
        );

        let tool_definitions = agent.tool_definitions();
        protected_start_index = compaction::preflight_compact_context(
            agent,
            &turn_prelude,
            protected_start_index,
            &tool_definitions,
            &mut on_event,
        )
        .await?;

        let build = build_request(RequestBuilderInput {
            protocol: ApiProtocol::Completions,
            model_id: &agent.model,
            model: agent.active_model_metadata(),
            prelude: &turn_prelude,
            history: &agent.history,
            protected_start_index,
            tools: &tool_definitions,
            evidence: &agent.evidence,
        })?;
        on_event(AgentEvent::TokenUsageUpdated {
            used_tokens: build.budget.estimated_request_tokens,
            context_window_tokens: build.budget.context_window_tokens,
            input_tokens: build.budget.estimated_request_tokens,
            output_tokens: 0,
            cached_tokens: 0,
        })
        .await?;
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
            .await?;
            let mut byte_stream = response.bytes_stream();
            let mut sse_buffer = String::new();
            let mut turn_text = String::new();
            let mut tool_calls: BTreeMap<usize, ChatCompletionMessageToolCall> = BTreeMap::new();
            let mut pending_tool_calls = HashSet::new();
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
                    Err(error) => return Err(error.into()),
                };
                append_sse_chunk(&mut sse_buffer, &chunk);
                let events = drain_sse_data_events(&mut sse_buffer);
                for event in events {
                    let Some(data) = event else {
                        continue;
                    };
                    let response: CompatibleChatCompletionStreamResponse =
                        serde_json::from_str(&data).with_context(|| {
                            format!("failed to parse chat completions stream event: {data}")
                        })?;
                    if let Some(usage) = &response.usage {
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
                                        stream_had_side_effect = true;
                                        emit_tool_call_pending_if_ready(
                                            &mut pending_tool_calls,
                                            &call.id,
                                            &call.function.name,
                                            &mut on_event,
                                        )
                                        .await?;
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
                let response: CompatibleChatCompletionStreamResponse = serde_json::from_str(&data)
                    .with_context(|| {
                        format!("failed to parse chat completions stream event: {data}")
                    })?;
                if let Some(usage) = &response.usage {
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
                                    stream_had_side_effect = true;
                                    emit_tool_call_pending_if_ready(
                                        &mut pending_tool_calls,
                                        &call.id,
                                        &call.function.name,
                                        &mut on_event,
                                    )
                                    .await?;
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
            validate_chat_finish_reasons(&finish_reasons, has_tool_calls)?;

            if !has_tool_calls {
                if final_text.is_empty() {
                    final_text = "No response content".to_string();
                }

                agent
                    .history
                    .push(HistoryItem::assistant(turn_text.clone()));

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
            validate_chat_tool_calls(&tool_calls)?;
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

    Err(anyhow!(
        "stopped: too many agent iterations (max {})",
        agent.max_iterations
    ))
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
    let http = reqwest::Client::new();

    loop {
        let response = match http
            .post(config.url("/chat/completions"))
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
