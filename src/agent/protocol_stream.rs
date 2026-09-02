use super::*;
use crate::model_runtime::projection::model_request_from_prompt_plan;
use crate::model_runtime::runtime::{
    AttemptOutcome, CompletedToolCall, ModelAttemptResult, ModelAttemptSnapshot, ModelRuntime,
    TurnContinuationDecision, TurnDriver, TurnLimits, TurnOrchestrator,
};
use crate::model_runtime::{ModelEvent, ModelFailure, ModelMessage, ModelRequestInput};
use crate::user_content::UserMessageContent;
use std::collections::BTreeMap;

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

fn resolved_protocol(route: &crate::model_runtime::ResolvedModelRoute) -> Result<ApiProtocol> {
    match route.protocol_id.as_str() {
        "responses" => Ok(ApiProtocol::Responses),
        "completions" => Ok(ApiProtocol::Completions),
        "anthropic" => Ok(ApiProtocol::Anthropic),
        value => Err(anyhow!("unsupported resolved protocol '{value}'")),
    }
}

fn runtime_failure(
    phase: crate::model_runtime::FailurePhase,
    error: impl std::fmt::Display,
) -> crate::model_runtime::ModelFailure {
    crate::model_runtime::ModelFailure::new(phase, crate::model_runtime::FailureKind::Internal)
        .with_detail(error.to_string())
}

fn failure_error_class(failure: &ModelFailure) -> LlmRequestErrorClass {
    match failure.phase {
        crate::model_runtime::FailurePhase::Prepare | crate::model_runtime::FailurePhase::Bind => {
            LlmRequestErrorClass::RequestCreation
        }
        crate::model_runtime::FailurePhase::Transport => LlmRequestErrorClass::StreamRead,
        crate::model_runtime::FailurePhase::Decode | crate::model_runtime::FailurePhase::Finish => {
            if matches!(
                failure.kind,
                crate::model_runtime::FailureKind::Authentication
                    | crate::model_runtime::FailureKind::RateLimited
                    | crate::model_runtime::FailureKind::Http
            ) {
                LlmRequestErrorClass::ProviderTerminal
            } else {
                LlmRequestErrorClass::ProtocolValidation
            }
        }
    }
}

/// Executes a normal Agent turn through the installed provider-neutral route.
pub(super) async fn run_resolved_turn_async<F, E, A, Dfut, Efut, Afut>(
    agent: &mut Agent,
    user_content: UserMessageContent,
    user_input: &str,
    mut on_delta: F,
    mut on_event: E,
    mut approve: A,
) -> Result<String>
where
    F: FnMut(&str) -> Dfut + Send,
    E: FnMut(AgentEvent) -> Efut + Send,
    A: FnMut(PermissionRequest) -> Afut + Send,
    Dfut: Future<Output = Result<()>> + Send,
    Efut: Future<Output = Result<()>> + Send,
    Afut: Future<Output = Result<PermissionApproval>> + Send,
{
    let route = agent
        .resolved_model_route()
        .cloned()
        .ok_or_else(|| anyhow!("normal Agent turn requires an installed resolved model route"))?;
    let protocol = resolved_protocol(&route)?;
    let turn_prelude =
        agent.try_prepare_turn_prelude_with_skills(user_input, &user_content.selected_skills)?;
    let protected_start_index = agent.active_history_items().len();
    let previous_turn_start_index = agent.turn.current_turn_start_index;
    agent.turn.current_turn_start_index = Some(protected_start_index);
    if !user_content.has_no_parts()
        && let Err(error) = agent.append_history_item(HistoryItem::user_content(user_content))
    {
        agent.turn.current_turn_start_index = previous_turn_start_index;
        return Err(error);
    }
    Agent::emit_audit_event(
        &mut on_event,
        AgentEvent::TurnStarted(agent.turn_started_event()),
        "turn_started",
    )
    .await;

    let limits = TurnLimits {
        max_iterations: agent.max_iterations.unwrap_or(usize::MAX),
        max_tool_calls: agent.max_tool_calls,
    };
    let fake_decorator = agent.fake_client().and_then(|client| {
        let profile = match route.protocol_id.as_str() {
            "responses" => crate::fake::FakeClient::Codex,
            "anthropic" => crate::fake::FakeClient::Anthropic,
            _ => return None,
        };
        agent.fake_turn_context(profile).and_then(|context| {
            crate::model_runtime::decorator::FakeRequestDecorator::new(
                client,
                &route.protocol_id,
                context,
            )
            .ok()
        })
    });
    let ws_transport = (route.protocol_id.as_str() == "responses" && route.transport.websocket())
        .then(|| {
            std::sync::Arc::new(crate::model_runtime::runtime::TurnLocalResponsesTransport::new())
        });
    let runtime = ws_transport
        .as_ref()
        .map(|transport| ModelRuntime::new_responses_websocket(transport.clone()))
        .unwrap_or_default();
    let mut driver = ResolvedTurnDriver {
        agent,
        route: route.clone(),
        protocol,
        turn_prelude: &turn_prelude,
        protected_start_index,
        on_delta: &mut on_delta,
        on_event: &mut on_event,
        approve: &mut approve,
        tool_call_count: 0,
        continuation_count: 0,
        final_text: String::new(),
        prepared: None,
        usage: None,
        response_id: None,
        cache_read_tokens: None,
        cache_write_tokens: None,
        cache_details_present: false,
        current_attempt: 1,
        scheduled_retry: None,
        fake_decorator,
        ws_transport,
        ws_previous: None,
    };
    TurnOrchestrator::new(runtime, limits)
        .run(&route, &mut driver)
        .await
        .map_err(anyhow::Error::new)?;
    Ok(if driver.final_text.is_empty() {
        "No response content".to_string()
    } else {
        driver.final_text
    })
}

struct PreparedResolvedIteration {
    build: crate::request_builder::BuildResult,
    epoch_preview: crate::agent::ActiveEpochPreview,
    telemetry: LlmRequestTelemetry,
    observation: crate::request_builder::LogicalRequestObservation,
    inspection: Option<crate::model_runtime::PreparedRequestInspection>,
}

struct WsRequestSnapshot {
    inspection: crate::model_runtime::PreparedRequestInspection,
    assistant_frontier: u64,
}

struct ResolvedTurnDriver<'a, F, E, A> {
    agent: &'a mut Agent,
    route: std::sync::Arc<crate::model_runtime::ResolvedModelRoute>,
    protocol: ApiProtocol,
    turn_prelude: &'a [PromptMessage],
    protected_start_index: usize,
    on_delta: &'a mut F,
    on_event: &'a mut E,
    approve: &'a mut A,
    tool_call_count: usize,
    continuation_count: usize,
    final_text: String,
    prepared: Option<PreparedResolvedIteration>,
    usage: Option<TokenUsageEstimate>,
    response_id: Option<String>,
    cache_read_tokens: Option<u64>,
    cache_write_tokens: Option<u64>,
    cache_details_present: bool,
    current_attempt: usize,
    scheduled_retry: Option<LlmRetryLifecycle>,
    fake_decorator: Option<crate::model_runtime::decorator::FakeRequestDecorator>,
    ws_transport:
        Option<std::sync::Arc<crate::model_runtime::runtime::TurnLocalResponsesTransport>>,
    ws_previous: Option<WsRequestSnapshot>,
}

#[async_trait::async_trait]
impl<'a, F, E, A, Dfut, Efut, Afut> TurnDriver for ResolvedTurnDriver<'a, F, E, A>
where
    F: FnMut(&str) -> Dfut + Send,
    E: FnMut(AgentEvent) -> Efut + Send,
    A: FnMut(PermissionRequest) -> Afut + Send,
    Dfut: Future<Output = Result<()>> + Send,
    Efut: Future<Output = Result<()>> + Send,
    Afut: Future<Output = Result<PermissionApproval>> + Send,
{
    async fn prepare_iteration(
        &mut self,
        iteration: usize,
    ) -> std::result::Result<ModelRequestInput, ModelFailure> {
        let tools = self.agent.tool_definitions();
        let prepared = prepare_protocol_stream_request(
            self.agent,
            self.protocol,
            self.turn_prelude,
            &mut self.protected_start_index,
            &tools,
            self.on_event,
        )
        .await
        .map_err(|error| runtime_failure(crate::model_runtime::FailurePhase::Prepare, error))?;
        self.protected_start_index = prepared.protected_start_index;
        let input = model_request_from_prompt_plan(
            &self.route,
            &self.agent.active_model_metadata(),
            &prepared.build.prompt_plan,
            &tools,
        )
        .map_err(|error| runtime_failure(crate::model_runtime::FailurePhase::Prepare, error))?;
        let planned_observation = self.agent.preview_final_logical_request(&prepared.build);
        let telemetry = llm_request_telemetry(
            &format!("turn-{}-iteration-{iteration}", self.agent.turn.turn_id),
            self.agent.turn.turn_id,
            iteration,
            1,
            &self.route.model_override,
            self.protocol,
            &prepared.build,
            self.tool_call_count,
            tools.len(),
            planned_observation,
        );
        self.usage = None;
        self.response_id = None;
        self.cache_read_tokens = None;
        self.cache_write_tokens = None;
        self.cache_details_present = false;
        self.current_attempt = 1;
        self.prepared = Some(PreparedResolvedIteration {
            build: prepared.build,
            epoch_preview: prepared.epoch_preview,
            telemetry,
            observation: crate::request_builder::LogicalRequestObservation {
                cohort: crate::request_builder::LogicalRequestCohort {
                    request_shape_digest: String::new(),
                },
                units: Vec::new(),
            },
            inspection: None,
        });
        Ok(input)
    }

    fn bypass_iteration_limit(&self) -> bool {
        self.agent.turn.auto_continue_active
    }

    fn bypass_tool_limit(&self) -> bool {
        self.agent.turn.auto_continue_active
    }

    async fn decorate_request(
        &mut self,
        request: crate::model_runtime::PreparedHttpRequest,
    ) -> std::result::Result<crate::model_runtime::PreparedHttpRequest, ModelFailure> {
        let request = match &self.fake_decorator {
            Some(decorator) => decorator.decorate(&self.route.protocol_id, request)?,
            None => request,
        };
        if let Some(prepared) = self.prepared.as_mut() {
            let stable_request =
                if let Some(stable_end) = prepared.build.prompt_plan.stable_prefix_end {
                    let mut stable_plan = prepared.build.prompt_plan.clone();
                    stable_plan.segments.truncate(stable_end + 1);
                    stable_plan.recompute_cache_metadata();
                    let stable_input = model_request_from_prompt_plan(
                        &self.route,
                        &self.agent.active_model_metadata(),
                        &stable_plan,
                        &self.agent.tool_definitions(),
                    )
                    .map_err(|error| {
                        runtime_failure(crate::model_runtime::FailurePhase::Prepare, error)
                    })?;
                    let stable_request = self.route.binding.prepare_request(&stable_input)?;
                    let stable_request = match &self.fake_decorator {
                        Some(decorator) => {
                            decorator.decorate(&self.route.protocol_id, stable_request)?
                        }
                        None => stable_request,
                    };
                    Some(stable_request)
                } else {
                    None
                };
            let inspection = self
                .route
                .binding
                .inspect_prepared_request(&request, stable_request.as_ref())?;
            prepared.observation = crate::request_builder::observe_prepared_model_request(
                &inspection,
                &prepared.build.prompt_plan,
            )
            .map_err(|error| runtime_failure(crate::model_runtime::FailurePhase::Prepare, error))?;
            prepared.inspection = Some(inspection.clone());
            let adjacent = self
                .agent
                .preview_logical_observation(prepared.observation.clone());
            prepared.telemetry.cache_configured = self.route.cache.enabled;
            prepared.telemetry.cache_hint_serialized = inspection.cache.hint_serialized;
            prepared.telemetry.cache_retention_sent =
                inspection.cache.retention_sent.map(|value| match value {
                    crate::model_runtime::CacheRetention::InMemory => {
                        crate::config::PromptCacheRetention::InMemory
                    }
                    crate::model_runtime::CacheRetention::TwentyFourHours => {
                        crate::config::PromptCacheRetention::TwentyFourHours
                    }
                });
            prepared.telemetry.cache_stable_prefix_segments = prepared
                .build
                .prompt_plan
                .stable_prefix_end
                .map_or(0, |index| index + 1);
            prepared.telemetry.local_prefix_fingerprint = inspection.cache.local_prefix_fingerprint;
            prepared.telemetry.routing_key = inspection.cache.routing_key;
            prepared.telemetry.adjacent_lcp_units =
                adjacent.cohort_comparable.then_some(adjacent.lcp_units);
            prepared.telemetry.adjacent_lcp_bytes =
                adjacent.cohort_comparable.then_some(adjacent.lcp_bytes);
            prepared.telemetry.adjacent_lcp_estimated_tokens = adjacent
                .cohort_comparable
                .then_some(adjacent.lcp_estimated_tokens);
            prepared.telemetry.current_unit_count = adjacent.current_unit_count;
            prepared.telemetry.first_breaker = adjacent.first_breaker;
            prepared.telemetry.cohort_comparable = adjacent.cohort_comparable;
            prepared.telemetry.cohort_changed = adjacent.cohort_changed;
        }
        if let Some(ws_transport) = &self.ws_transport {
            let current_inspection = self
                .prepared
                .as_ref()
                .and_then(|prepared| prepared.inspection.as_ref());
            let current_plan = self
                .prepared
                .as_ref()
                .map(|prepared| &prepared.build.prompt_plan);
            let boundary = match (&self.ws_previous, current_inspection, current_plan) {
                (Some(previous), Some(current_inspection), Some(current_plan)) => {
                    websocket_incremental_prompt_unit_start(
                        current_inspection,
                        current_plan,
                        previous,
                    )
                }
                _ => None,
            };
            if let Some(prompt_unit_start) = boundary {
                ws_transport
                    .set_next_prompt_unit_start(prompt_unit_start)
                    .await;
            } else {
                ws_transport.reset_chain().await;
            }
        }
        Ok(request)
    }

    async fn attempt_started(
        &mut self,
        _iteration: usize,
        attempt: usize,
    ) -> std::result::Result<(), ModelFailure> {
        let Some(prepared) = &self.prepared else {
            return Ok(());
        };
        self.current_attempt = attempt;
        self.usage = None;
        self.response_id = None;
        self.cache_read_tokens = None;
        self.cache_write_tokens = None;
        self.cache_details_present = false;
        let mut telemetry = prepared.telemetry.clone();
        telemetry.attempt = attempt;
        if attempt > 1 {
            telemetry.adjacent_lcp_units = None;
            telemetry.adjacent_lcp_bytes = None;
            telemetry.adjacent_lcp_estimated_tokens = None;
            telemetry.first_breaker = None;
        }
        (self.on_event)(AgentEvent::LlmRequestTelemetry(telemetry))
            .await
            .map_err(|error| runtime_failure(crate::model_runtime::FailurePhase::Transport, error))
    }

    async fn commit_before_first_send(
        &mut self,
        _iteration: usize,
    ) -> std::result::Result<(), ModelFailure> {
        let Some(prepared) = &self.prepared else {
            return Ok(());
        };
        self.agent
            .commit_resolved_active_epoch(
                prepared.epoch_preview.clone(),
                prepared.observation.clone(),
            )
            .map_err(|error| runtime_failure(crate::model_runtime::FailurePhase::Prepare, error))?;
        self.agent
            .commit_logical_observation(prepared.observation.clone());
        Ok(())
    }

    async fn attempt_finished(
        &mut self,
        _iteration: usize,
        _attempt: usize,
        outcome: &AttemptOutcome,
    ) -> std::result::Result<(), ModelFailure> {
        let Some(prepared) = &self.prepared else {
            return Ok(());
        };
        let mut base = prepared.telemetry.clone();
        base.attempt = self.current_attempt;
        let telemetry = match outcome {
            AttemptOutcome::Completed { .. } => return Ok(()),
            AttemptOutcome::Failed {
                failure,
                side_effects,
            } => {
                let error_class = failure_error_class(failure);
                if side_effects.observable() {
                    base.interrupted(error_class)
                } else {
                    base.failed(error_class)
                }
            }
        };
        (self.on_event)(AgentEvent::LlmRequestTelemetry(telemetry))
            .await
            .map_err(|error| runtime_failure(crate::model_runtime::FailurePhase::Transport, error))
    }

    async fn retry_scheduled(
        &mut self,
        _iteration: usize,
        _attempt: usize,
        next_attempt: usize,
        delay: std::time::Duration,
        failure: &ModelFailure,
    ) -> std::result::Result<(), ModelFailure> {
        if failure.code.as_deref() == Some("previous_response_not_found")
            && let Some(ws_transport) = &self.ws_transport
        {
            self.ws_previous = None;
            ws_transport.reset_chain().await;
        }
        let retry = LlmRetryLifecycle {
            attempt: next_attempt,
            max_attempts: self
                .route
                .retry
                .as_ref()
                .map_or(1, |retry| retry.max_attempts),
            delay_secs: delay.as_secs(),
            error: failure.to_string(),
        };
        (self.on_event)(AgentEvent::LlmRetryScheduled(retry.clone()))
            .await
            .map_err(|error| {
                runtime_failure(crate::model_runtime::FailurePhase::Transport, error)
            })?;
        self.scheduled_retry = Some(retry);
        Ok(())
    }

    async fn retry_started(
        &mut self,
        _iteration: usize,
        _attempt: usize,
    ) -> std::result::Result<(), ModelFailure> {
        if let Some(retry) = self.scheduled_retry.take() {
            (self.on_event)(AgentEvent::LlmRetryStarted(retry))
                .await
                .map_err(|error| {
                    runtime_failure(crate::model_runtime::FailurePhase::Transport, error)
                })?;
        }
        Ok(())
    }

    async fn observe_event(&mut self, event: &ModelEvent) -> std::result::Result<(), ModelFailure> {
        match event {
            ModelEvent::TextDelta { text } => {
                (self.on_delta)(text).await.map_err(|error| {
                    runtime_failure(crate::model_runtime::FailurePhase::Transport, error)
                })?;
                self.final_text.push_str(text);
            }
            ModelEvent::ReasoningDelta { item_id, text } => {
                (self.on_event)(AgentEvent::ReasoningDelta {
                    item_id: item_id.clone(),
                    delta: text.clone(),
                })
                .await
                .map_err(|error| {
                    runtime_failure(crate::model_runtime::FailurePhase::Transport, error)
                })?;
            }
            ModelEvent::ReasoningDone { item_id, text, .. } => {
                (self.on_event)(AgentEvent::ReasoningDone {
                    item_id: item_id.clone(),
                    text: text.clone(),
                })
                .await
                .map_err(|error| {
                    runtime_failure(crate::model_runtime::FailurePhase::Transport, error)
                })?;
            }
            ModelEvent::ToolStarted { id, name } => {
                (self.on_event)(AgentEvent::ToolCallPending {
                    call_id: id.clone(),
                    name: name.clone(),
                })
                .await
                .map_err(|error| {
                    runtime_failure(crate::model_runtime::FailurePhase::Transport, error)
                })?;
            }
            ModelEvent::Usage {
                input_tokens,
                output_tokens,
                total_tokens,
                cached_input_tokens,
                ..
            } => {
                let Some(prepared) = &self.prepared else {
                    return Ok(());
                };
                let cached_tokens = cached_input_tokens.unwrap_or(0);
                self.cache_details_present |= cached_input_tokens.is_some();
                let usage = TokenUsageEstimate {
                    used_tokens: *total_tokens,
                    context_window_tokens: prepared.build.budget.context_window_tokens,
                    input_tokens: *input_tokens,
                    output_tokens: *output_tokens,
                    cached_tokens: self.cache_read_tokens.unwrap_or(cached_tokens),
                };
                self.usage = Some(usage);
            }
            ModelEvent::ResponseMetadata { response_id } => {
                self.response_id = Some(response_id.clone());
            }
            ModelEvent::Cache {
                read_tokens,
                write_tokens,
                ..
            } => {
                self.cache_read_tokens = Some(*read_tokens);
                self.cache_write_tokens = Some(*write_tokens);
                self.cache_details_present = true;
                if let Some(usage) = self.usage.as_mut() {
                    usage.cached_tokens = *read_tokens;
                }
            }
            _ => {}
        }
        Ok(())
    }

    async fn persist_assistant(
        &mut self,
        assistant: &ModelMessage,
    ) -> std::result::Result<(), ModelFailure> {
        let mut text = String::new();
        let mut reasoning = String::new();
        let mut replay = None;
        let mut calls = Vec::new();
        for part in &assistant.content {
            match part {
                crate::model_runtime::ContentPart::Text(value) => text.push_str(value),
                crate::model_runtime::ContentPart::Reasoning {
                    text: value,
                    replay: value_replay,
                    ..
                } => {
                    reasoning.push_str(value);
                    replay = value_replay.clone();
                }
                crate::model_runtime::ContentPart::ToolCall {
                    id,
                    name,
                    arguments,
                } => calls.push(HistoryToolCall {
                    call_id: id.clone(),
                    name: name.clone(),
                    arguments_json: arguments.to_string(),
                }),
                _ => {}
            }
        }
        self.agent
            .append_history_item(HistoryItem::AssistantTurn {
                text: (!text.is_empty()).then_some(text.clone()),
                reasoning_content: (!reasoning.is_empty()).then_some(reasoning.clone()),
                replay: replay.clone(),
                calls: calls.clone(),
            })
            .map_err(|error| runtime_failure(crate::model_runtime::FailurePhase::Finish, error))?;
        if let Some(ws_transport) = &self.ws_transport {
            let frontier = self
                .agent
                .active_protocol_frames()
                .iter()
                .rev()
                .find_map(|frame| {
                    matches!(
                        frame.item,
                        crate::protocol_frames::ProtocolFrameItem::AssistantTurn { .. }
                    )
                    .then(|| {
                        frame
                            .source_provenance
                            .as_ref()
                            .and_then(|provenance| provenance.source_span)
                            .map(|span| span.end_sequence)
                    })
                    .flatten()
                });
            if let (Some(prepared), Some(frontier)) = (&self.prepared, frontier) {
                if let Some(inspection) = &prepared.inspection {
                    self.ws_previous = Some(WsRequestSnapshot {
                        inspection: inspection.clone(),
                        assistant_frontier: frontier,
                    });
                }
            } else {
                self.ws_previous = None;
                ws_transport.reset_chain().await;
            }
        }
        if let Some(prepared) = &self.prepared {
            if let Some(usage) = self.usage {
                let mut cache_report = CacheUsageReport::from_build(&prepared.build);
                if self.cache_details_present {
                    cache_report = cache_report.with_actual_cached_tokens(usage.cached_tokens);
                }
                cache_report.configured = prepared.telemetry.cache_configured;
                cache_report.hint_serialized = prepared.telemetry.cache_hint_serialized;
                cache_report.retention_sent = prepared.telemetry.cache_retention_sent;
                cache_report.stable_prefix_segments =
                    prepared.telemetry.cache_stable_prefix_segments;
                cache_report.local_prefix_fingerprint =
                    prepared.telemetry.local_prefix_fingerprint.clone();
                cache_report.routing_key = prepared.telemetry.routing_key.clone();
                (self.on_event)(AgentEvent::TokenUsageUpdated {
                    used_tokens: usage.used_tokens,
                    context_window_tokens: usage.context_window_tokens,
                    input_tokens: usage.input_tokens,
                    output_tokens: usage.output_tokens,
                    cached_tokens: usage.cached_tokens,
                    cache_report: Some(cache_report),
                })
                .await
                .map_err(|error| {
                    runtime_failure(crate::model_runtime::FailurePhase::Finish, error)
                })?;
            }
            let mut completed = prepared.telemetry.completed(
                self.usage,
                self.response_id.clone(),
                if self.usage.is_none() {
                    ProviderUsageCompleteness::UsageMissing
                } else if self.cache_details_present {
                    ProviderUsageCompleteness::Complete
                } else {
                    ProviderUsageCompleteness::CacheDetailsMissing
                },
            );
            completed.attempt = self.current_attempt;
            completed.cache_write_tokens = self.cache_write_tokens;
            (self.on_event)(AgentEvent::LlmRequestTelemetry(completed))
                .await
                .map_err(|error| {
                    runtime_failure(crate::model_runtime::FailurePhase::Finish, error)
                })?;
        }
        if let Some(usage) = self.usage {
            self.agent.install_provider_usage_anchor(usage);
        }
        if calls.is_empty() {
            if !text.is_empty() {
                (self.on_event)(AgentEvent::AssistantMessage { content: text })
                    .await
                    .map_err(|error| {
                        runtime_failure(crate::model_runtime::FailurePhase::Finish, error)
                    })?;
            }
        } else {
            (self.on_event)(AgentEvent::AssistantToolCallBatch {
                text: (!text.is_empty()).then_some(text),
                reasoning_content: (!reasoning.is_empty()).then_some(reasoning),
                reasoning_wire: replay.and_then(|value| value.payload_json()),
                calls,
            })
            .await
            .map_err(|error| runtime_failure(crate::model_runtime::FailurePhase::Finish, error))?;
        }
        Ok(())
    }

    async fn execute_tools(
        &mut self,
        tools: &[CompletedToolCall],
    ) -> std::result::Result<(), ModelFailure> {
        let calls = tools
            .iter()
            .map(|tool| HistoryToolCall {
                call_id: tool.id.clone(),
                name: tool.name.clone(),
                arguments_json: tool.arguments.to_string(),
            })
            .collect::<Vec<_>>();
        self.tool_call_count = self.tool_call_count.saturating_add(calls.len());
        self.agent
            .execute_tool_calls_and_record(&calls, self.on_event, self.approve)
            .await
            .map_err(|error| runtime_failure(crate::model_runtime::FailurePhase::Finish, error))?;
        (self.on_event)(AgentEvent::ToolCallBatchFinished)
            .await
            .map_err(|error| runtime_failure(crate::model_runtime::FailurePhase::Finish, error))?;
        (self.on_event)(AgentEvent::TurnContinuationBoundary)
            .await
            .map_err(|error| runtime_failure(crate::model_runtime::FailurePhase::Finish, error))?;
        let _ = self
            .agent
            .drain_turn_continuations(self.on_event)
            .await
            .map_err(|error| runtime_failure(crate::model_runtime::FailurePhase::Finish, error))?;
        Ok(())
    }

    async fn recover_iteration(
        &mut self,
        partial: &ModelAttemptSnapshot,
        _failure: &ModelFailure,
    ) -> std::result::Result<(), ModelFailure> {
        if let Some(ws_transport) = &self.ws_transport {
            self.ws_previous = None;
            ws_transport.reset_chain().await;
        }
        let pending = partial
            .events
            .iter()
            .filter_map(|event| match event {
                ModelEvent::ToolStarted { id, name } => Some((id.clone(), name.clone())),
                _ => None,
            })
            .collect::<std::collections::BTreeMap<_, _>>();
        emit_pending_tool_call_cancellations(&pending, self.on_event)
            .await
            .map_err(|error| runtime_failure(crate::model_runtime::FailurePhase::Finish, error))?;
        let mut text = String::new();
        let mut reasoning = String::new();
        let mut replay = None;
        for part in &partial.assistant.content {
            match part {
                crate::model_runtime::ContentPart::Text(value) => text.push_str(value),
                crate::model_runtime::ContentPart::Reasoning {
                    text: value,
                    replay: value_replay,
                    ..
                } => {
                    reasoning.push_str(value);
                    replay = value_replay.clone();
                }
                _ => {}
            }
        }
        if !text.is_empty() || !reasoning.is_empty() || replay.is_some() {
            self.agent
                .append_history_item(HistoryItem::AssistantTurn {
                    text: (!text.is_empty()).then_some(text.clone()),
                    reasoning_content: (!reasoning.is_empty()).then_some(reasoning.clone()),
                    replay,
                    calls: Vec::new(),
                })
                .map_err(|error| {
                    runtime_failure(crate::model_runtime::FailurePhase::Finish, error)
                })?;
            if !text.is_empty() {
                (self.on_event)(AgentEvent::AssistantMessage {
                    content: text.clone(),
                })
                .await
                .map_err(|error| {
                    runtime_failure(crate::model_runtime::FailurePhase::Finish, error)
                })?;
            }
        }
        let continuation = build_stream_interrupt_continuation(&text, &pending);
        self.agent
            .append_history_item(HistoryItem::internal_continuation(continuation.clone()))
            .map_err(|error| runtime_failure(crate::model_runtime::FailurePhase::Finish, error))?;
        (self.on_event)(AgentEvent::ModelStreamIssue {
            message: STREAM_INTERRUPT_MESSAGE.into(),
            detail: Some("The model stream was interrupted; partial output was preserved.".into()),
            action: STREAM_INTERRUPT_ACTION.into(),
        })
        .await
        .map_err(|error| runtime_failure(crate::model_runtime::FailurePhase::Finish, error))?;
        (self.on_event)(AgentEvent::InternalContinuation {
            text: continuation,
            source: crate::transcript::InternalContinuationSource::StreamRecovery,
        })
        .await
        .map_err(|error| runtime_failure(crate::model_runtime::FailurePhase::Finish, error))?;
        Ok(())
    }

    async fn after_assistant_persisted(
        &mut self,
        _result: &ModelAttemptResult,
    ) -> std::result::Result<TurnContinuationDecision, ModelFailure> {
        (self.on_event)(AgentEvent::TurnContinuationBoundary)
            .await
            .map_err(|error| runtime_failure(crate::model_runtime::FailurePhase::Finish, error))?;
        if self
            .agent
            .drain_turn_continuations(self.on_event)
            .await
            .map_err(|error| runtime_failure(crate::model_runtime::FailurePhase::Finish, error))?
        {
            return Ok(TurnContinuationDecision::Continue);
        }
        let mut count = self.continuation_count;
        if self
            .agent
            .continue_or_finalize_no_tool_reply(self.on_event, self.tool_call_count, &mut count)
            .await
            .map_err(|error| runtime_failure(crate::model_runtime::FailurePhase::Finish, error))?
        {
            self.continuation_count = count;
            Ok(TurnContinuationDecision::Continue)
        } else {
            Ok(TurnContinuationDecision::Finalize)
        }
    }

    async fn finalize(
        &mut self,
        _result: &ModelAttemptResult,
    ) -> std::result::Result<(), ModelFailure> {
        Ok(())
    }
}

fn websocket_incremental_prompt_unit_start(
    current_inspection: &crate::model_runtime::PreparedRequestInspection,
    current_plan: &crate::request_builder::prompt_plan::PromptPlan,
    previous: &WsRequestSnapshot,
) -> Option<usize> {
    let previous_len = previous.inspection.prompt_units.len();
    if current_inspection.request_shape != previous.inspection.request_shape
        || current_inspection.prompt_units.len() <= previous_len
        || current_inspection.prompt_units[..previous_len] != previous.inspection.prompt_units[..]
    {
        return None;
    }

    let mut last_span_end = previous.assistant_frontier;
    let mut first_new_index = None;
    for (offset, unit) in current_inspection.prompt_units[previous_len..]
        .iter()
        .enumerate()
    {
        let segment_id = (unit.semantic_segment_ids.len() == 1)
            .then(|| unit.semantic_segment_ids[0].as_str())?;
        let segment = current_plan.segment(segment_id)?;
        let span = segment.source.provenance.source_span?;
        let is_assistant_output = matches!(
            segment.role,
            crate::request_builder::prompt_plan::PromptSegmentRole::Assistant
        ) && matches!(
            &segment.content,
            crate::request_builder::prompt_plan::PromptSegmentContent::Text { .. }
                | crate::request_builder::prompt_plan::PromptSegmentContent::AssistantToolCalls { .. }
        );
        let is_new_client_input = matches!(
            &segment.content,
            crate::request_builder::prompt_plan::PromptSegmentContent::ToolOutput { .. }
        ) || (matches!(
            &segment.content,
            crate::request_builder::prompt_plan::PromptSegmentContent::Text { .. }
        ) && segment.source.source_label.as_deref()
            == Some("internal_continuation"));

        if span.end_sequence <= previous.assistant_frontier {
            if first_new_index.is_some() || !is_assistant_output {
                return None;
            }
            continue;
        }
        if span.start_sequence <= previous.assistant_frontier
            || span.start_sequence <= last_span_end
            || !is_new_client_input
        {
            return None;
        }
        if first_new_index.is_none() {
            first_new_index = Some(previous_len + offset);
        }
        last_span_end = span.end_sequence;
    }
    first_new_index
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod websocket_incremental_tests {
    use super::*;
    use crate::model_runtime::{PreparedPromptUnitInspection, PreparedRequestInspection};
    use crate::request_builder::prompt_plan::{
        PromptPlan, PromptSegment, PromptSegmentContent, PromptSegmentProtection,
        PromptSegmentRetention, PromptSegmentRole, PromptSegmentSource, PromptSegmentStability,
        PromptTokenEstimate,
    };
    use crate::runtime_context::{
        PromptContributorKind, RuntimeFrameProvenance, RuntimeSource, SourceSpan,
    };

    fn inspection(units: Vec<(&str, &[&str])>) -> PreparedRequestInspection {
        PreparedRequestInspection {
            request_shape: vec![1],
            prompt_units: units
                .into_iter()
                .map(|(kind, ids)| PreparedPromptUnitInspection {
                    identity: serde_json::json!({"type": kind}).to_string().into_bytes(),
                    semantic_segment_ids: ids.iter().map(|id| (*id).to_string()).collect(),
                })
                .collect(),
            cache: Default::default(),
        }
    }

    fn segment(
        id: &str,
        role: PromptSegmentRole,
        source_span: Option<SourceSpan>,
        source_label: Option<&str>,
        content: PromptSegmentContent,
    ) -> PromptSegment {
        PromptSegment {
            id: id.into(),
            order: 0,
            role,
            contributor_id: id.into(),
            source: PromptSegmentSource {
                order: 0,
                contributor_kind: PromptContributorKind::CurrentTurn,
                provenance: RuntimeFrameProvenance {
                    source: RuntimeSource::Transcript,
                    label: None,
                    source_span,
                    source_id: None,
                },
                source_key: None,
                source_label: source_label.map(str::to_owned),
            },
            stability: PromptSegmentStability::Volatile,
            retention: PromptSegmentRetention::Required,
            protection: PromptSegmentProtection::default(),
            cache: crate::request_builder::prompt_plan::PromptCacheMetadata {
                cache_eligible: false,
                boundary: None,
                prefix_hash: None,
            },
            tokens: PromptTokenEstimate {
                estimated_input_tokens: None,
                budget_input_tokens: None,
                actual_input_tokens: None,
            },
            text: id.into(),
            content,
        }
    }

    fn plan_with_assistant_and_tool(
        assistant_span: Option<SourceSpan>,
        tool_span: Option<SourceSpan>,
    ) -> PromptPlan {
        PromptPlan {
            model_id: "test".into(),
            contributors: Vec::new(),
            segments: vec![
                segment(
                    "assistant",
                    PromptSegmentRole::Assistant,
                    assistant_span,
                    None,
                    PromptSegmentContent::AssistantToolCalls {
                        text: Some("calling lookup".into()),
                        reasoning_content: None,
                        replay: None,
                        calls: vec![crate::request_builder::HistoryToolCall {
                            call_id: "call".into(),
                            name: "lookup".into(),
                            arguments_json: "{}".into(),
                        }],
                    },
                ),
                segment(
                    "tool",
                    PromptSegmentRole::Tool,
                    tool_span,
                    None,
                    PromptSegmentContent::ToolOutput {
                        call_id: "call".into(),
                        output_json: "{}".into(),
                        images: Vec::new(),
                    },
                ),
            ],
            stable_prefix_end: None,
            kernel_end_exclusive: 0,
            envelope_end_exclusive: 0,
        }
    }

    fn previous_snapshot() -> WsRequestSnapshot {
        WsRequestSnapshot {
            inspection: inspection(vec![("instructions", &["system"]), ("message", &["user"])]),
            assistant_frontier: 10,
        }
    }

    #[test]
    fn websocket_incremental_boundary_accepts_only_new_tool_output() {
        let current = inspection(vec![
            ("instructions", &["system"]),
            ("message", &["user"]),
            ("message", &["assistant"]),
            ("message", &["tool"]),
        ]);
        assert_eq!(
            websocket_incremental_prompt_unit_start(
                &current,
                &plan_with_assistant_and_tool(
                    Some(SourceSpan::new(10, 10).unwrap()),
                    Some(SourceSpan::new(11, 12).unwrap()),
                ),
                &previous_snapshot(),
            ),
            Some(3)
        );
    }

    #[test]
    fn websocket_incremental_boundary_skips_repeated_assistant_units() {
        let current = inspection(vec![
            ("instructions", &["system"]),
            ("message", &["user"]),
            ("message", &["assistant"]),
            ("message", &["assistant"]),
            ("message", &["tool"]),
        ]);
        assert_eq!(
            websocket_incremental_prompt_unit_start(
                &current,
                &plan_with_assistant_and_tool(
                    Some(SourceSpan::new(10, 10).unwrap()),
                    Some(SourceSpan::new(11, 12).unwrap()),
                ),
                &previous_snapshot(),
            ),
            Some(4)
        );
    }

    #[test]
    fn websocket_incremental_boundary_resets_for_unsafe_prefix_or_missing_span() {
        let unsafe_prefix = inspection(vec![
            ("instructions", &["changed"]),
            ("message", &["user"]),
            ("message", &["assistant"]),
            ("message", &["tool"]),
        ]);
        assert_eq!(
            websocket_incremental_prompt_unit_start(
                &unsafe_prefix,
                &plan_with_assistant_and_tool(
                    Some(SourceSpan::new(10, 10).unwrap()),
                    Some(SourceSpan::new(11, 12).unwrap()),
                ),
                &previous_snapshot(),
            ),
            None
        );

        let current = inspection(vec![
            ("instructions", &["system"]),
            ("message", &["user"]),
            ("message", &["assistant"]),
            ("message", &["tool"]),
        ]);
        assert_eq!(
            websocket_incremental_prompt_unit_start(
                &current,
                &plan_with_assistant_and_tool(Some(SourceSpan::new(10, 10).unwrap()), None,),
                &previous_snapshot(),
            ),
            None
        );

        let crossing = inspection(vec![
            ("instructions", &["system"]),
            ("message", &["user"]),
            ("message", &["assistant"]),
            ("message", &["tool"]),
        ]);
        assert_eq!(
            websocket_incremental_prompt_unit_start(
                &crossing,
                &plan_with_assistant_and_tool(
                    Some(SourceSpan::new(10, 11).unwrap()),
                    Some(SourceSpan::new(12, 13).unwrap()),
                ),
                &previous_snapshot(),
            ),
            None
        );

        let old_after_new = inspection(vec![
            ("instructions", &["system"]),
            ("message", &["user"]),
            ("message", &["assistant"]),
            ("message", &["tool"]),
            ("message", &["assistant"]),
        ]);
        assert_eq!(
            websocket_incremental_prompt_unit_start(
                &old_after_new,
                &plan_with_assistant_and_tool(
                    Some(SourceSpan::new(10, 10).unwrap()),
                    Some(SourceSpan::new(11, 12).unwrap()),
                ),
                &previous_snapshot(),
            ),
            None
        );

        let multi_origin = inspection(vec![
            ("instructions", &["system"]),
            ("message", &["user"]),
            ("message", &["assistant", "other"]),
            ("message", &["tool"]),
        ]);
        assert_eq!(
            websocket_incremental_prompt_unit_start(
                &multi_origin,
                &plan_with_assistant_and_tool(
                    Some(SourceSpan::new(10, 10).unwrap()),
                    Some(SourceSpan::new(11, 12).unwrap()),
                ),
                &previous_snapshot(),
            ),
            None
        );
    }
}

async fn prepare_protocol_stream_request<E, Efut>(
    agent: &mut Agent,
    protocol: ApiProtocol,
    turn_prelude: &[PromptMessage],
    protected_start_index: &mut usize,
    tool_definitions: &[crate::request_builder::ToolSpec],
    on_event: &mut E,
) -> Result<crate::agent::compaction::PreparedRequestBuild>
where
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

async fn prepare_canonical_protocol_stream_request<E, Efut>(
    agent: &mut Agent,
    protocol: ApiProtocol,
    turn_prelude: &[PromptMessage],
    protected_start_index: &mut usize,
    tool_definitions: &[crate::request_builder::ToolSpec],
    on_event: &mut E,
) -> Result<crate::agent::compaction::PreparedRequestBuild>
where
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
    let classification = prepared.build.budget.request_classification();
    let Some(projected_usage) = agent.projected_token_usage() else {
        return Ok(prepared);
    };
    let under_soft = !prepared.build.budget.truncated
        && projected_usage.used_tokens < classification.high_watermark;
    if under_soft {
        return Ok(prepared);
    }
    if agent.turn.pressure_compaction.compacted_this_turn && classification.safe {
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
pub(super) async fn prepare_canonical_protocol_stream_request_for_test<E, Efut>(
    agent: &mut Agent,
    protocol: ApiProtocol,
    turn_prelude: &[PromptMessage],
    protected_start_index: &mut usize,
    tool_definitions: &[crate::request_builder::ToolSpec],
    on_event: &mut E,
) -> Result<crate::agent::compaction::PreparedRequestBuild>
where
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

async fn compact_for_request_pressure<E, Efut>(
    agent: &mut Agent,
    protocol: ApiProtocol,
    turn_prelude: &[PromptMessage],
    protected_start_index: &mut usize,
    tool_definitions: &[crate::request_builder::ToolSpec],
    on_event: &mut E,
) -> Result<crate::agent::compaction::PreparedRequestBuild>
where
    E: FnMut(AgentEvent) -> Efut + Send,
    Efut: Future<Output = Result<()>> + Send,
{
    let frames = agent.active_protocol_frames();
    let frontier = PressureCompactionFrontier {
        frame_count: frames.len(),
        protocol_prefix_digest: protocol_prefix_digest(&frames),
    };
    agent.turn.pressure_compaction.mark_attempted(frontier)?;
    let successor = compaction::compact_for_request_pressure(
        agent,
        protocol,
        turn_prelude,
        tool_definitions,
        on_event,
    )
    .await?;
    agent.turn.pressure_compaction.compacted_this_turn = true;
    *protected_start_index = successor.protected_start_index;
    Ok(successor)
}

fn build_oneshot_text_request(
    model_id: &str,
    model: ModelRequestMetadata,
    prelude: &[PromptMessage],
    user_text: &str,
) -> Result<crate::request_builder::BuildResult> {
    let mut snapshot = RuntimeSnapshot::new("helper-oneshot");
    let frames = crate::protocol_frames::history_items_to_frames(&[HistoryItem::user(user_text)]);
    for (ordinal, frame) in frames.into_iter().enumerate() {
        let stable_key = frame.stable_prompt_key();
        let runtime_frame = RuntimeFrame::new(
            RuntimeFrameKind::User,
            FrameVisibility::Active,
            RuntimeFrameProvenance::new(RuntimeSource::Derived),
            RuntimeFrameIdSeed {
                frame_kind: RuntimeFrameKind::User,
                source: RuntimeSource::Derived,
                ordinal: ordinal as u32,
                stable_key: &stable_key,
                source_span: None,
            },
        )
        .with_protocol(frame.item);
        snapshot
            .compaction
            .protected_frame_ids
            .push(runtime_frame.id);
        snapshot.push_frame(runtime_frame);
    }
    build_request_with_policy(
        RequestBuilderInput {
            model_id,
            model,
            prelude,
            snapshot: &snapshot,
            tools: &[],
        },
        None,
        Some(ProtectedContextPolicy { reserve_tokens: 0 }),
    )
}

pub(super) fn preflight_resolved_oneshot_text_request(
    route: &crate::model_runtime::ResolvedModelRoute,
    mut model: ModelRequestMetadata,
    prelude: &[PromptMessage],
    user_text: &str,
) -> Result<crate::request_builder::BuildResult> {
    model.supports_reasoning = false;
    model.reasoning_effort = None;
    model.reasoning_summary = None;
    model.supports_tools = false;
    model.parallel_tool_calls = false;
    model.fast_mode = false;
    let build =
        build_oneshot_text_request(&route.model_override, model.clone(), prelude, user_text)?;
    let input = model_request_from_prompt_plan(route, &model, &build.prompt_plan, &[])
        .map_err(anyhow::Error::msg)?;
    route
        .binding
        .prepare_request(&input)
        .map_err(anyhow::Error::new)?;
    Ok(build)
}

pub(super) async fn stream_resolved_oneshot_text_async<F, Fut>(
    route: &crate::model_runtime::ResolvedModelRoute,
    mut model: ModelRequestMetadata,
    prelude: &[PromptMessage],
    user_text: &str,
    mut on_delta: F,
) -> Result<String>
where
    F: FnMut(&str) -> Fut + Send,
    Fut: Future<Output = Result<()>> + Send,
{
    model.supports_reasoning = false;
    model.reasoning_effort = None;
    model.reasoning_summary = None;
    model.supports_tools = false;
    model.parallel_tool_calls = false;
    model.fast_mode = false;
    let build =
        build_oneshot_text_request(&route.model_override, model.clone(), prelude, user_text)?;
    let input = model_request_from_prompt_plan(route, &model, &build.prompt_plan, &[])
        .map_err(anyhow::Error::msg)?;
    ModelRuntime::default()
        .execute_text_oneshot(route, &input, move |delta| {
            let future = on_delta(delta);
            async move {
                future.await.map_err(|error| {
                    runtime_failure(crate::model_runtime::FailurePhase::Finish, error)
                })
            }
        })
        .await
        .map_err(anyhow::Error::new)
}

pub(super) async fn execute_resolved_text_oneshot(
    route: &crate::model_runtime::ResolvedModelRoute,
    model: ModelRequestMetadata,
    prelude: &[PromptMessage],
    user_text: &str,
) -> Result<String> {
    stream_resolved_oneshot_text_async(route, model, prelude, user_text, |_| {
        std::future::ready(Ok(()))
    })
    .await
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
