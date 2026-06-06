use anyhow::{Result, anyhow};
use async_openai::Client;
use async_openai::config::Config;
use async_openai::types::chat::{
    ChatCompletionMessageToolCall, ChatCompletionMessageToolCallChunk, FinishReason,
};
use async_openai::types::responses::{OutputItem, Response, ResponseStreamEvent};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::future::Future;
use tracing::{debug, error, info, trace, warn};

use crate::config::ApiProtocol;
use crate::evidence::{EvidenceDraft, EvidenceRecord, require_unique_evidence_id};
use crate::permission::{
    ExecutionDirective, PermissionDecision, PermissionMode, PermissionPolicy, PermissionRequest,
    restricted_by_directive_with_class,
};
use crate::request_builder::{
    BuiltRequest, HistoryItem, HistoryToolCall, ModelRequestMetadata, PromptMessage,
    RequestBuilderInput, build_request,
};
use crate::tool::{ToolHandler, ToolRegistry, ToolResult};
use crate::tool_format::format_tool_call;

#[derive(Debug, Clone)]
pub enum AgentEvent {
    TokenUsageUpdated {
        used_tokens: u64,
        context_window_tokens: u64,
    },
    ReasoningDelta {
        item_id: String,
        delta: String,
    },
    ReasoningDone {
        item_id: String,
        text: String,
    },
    ToolCallStarted {
        call_id: String,
        name: String,
        args: Value,
    },
    ToolCallFinished {
        call_id: String,
        name: String,
        ok: bool,
        output: ToolResult,
    },
    TodoSnapshotUpdated {
        items: Vec<TodoItem>,
    },
    AutoContinueChanged {
        state: AutoContinueState,
    },
    EvidenceRecorded(EvidenceRecord),
}

#[derive(Debug, Clone)]
pub enum ConversationRole {
    User,
    Assistant,
}

#[derive(Debug, Clone)]
pub struct ConversationMessage {
    pub role: ConversationRole,
    pub content: String,
}

const DEFAULT_AGENT_PRELUDE: &str = r#"You are a coding agent operating inside a local repository.
Work from the actual project state. Inspect relevant files before changing code. Prefer the smallest correct change that follows existing patterns.
Use tools deliberately: read/search before editing, edit only intended files, and run the validation that fits the task after changes when it is relevant.
When requirements are ambiguous or risky, ask a concise clarifying question. Do not hide errors with fallbacks; fail fast and explain the actionable cause.
Keep responses concise. Summarize changed files and validation results when code was modified."#;

const ENGINEERING_WORKFLOW_PRELUDE: &str = r#"This turn is an engineering workflow task.
Stay single-agent. Do not delegate work or introduce multi-agent or subtask orchestration.
For non-trivial work, keep a short working plan, track the steps you complete, and surface any remaining work or blockers before you stop."#;

pub struct Agent<C: Config> {
    pub client: Client<C>,
    model: String,
    default_protocol: ApiProtocol,
    model_protocols: HashMap<String, ApiProtocol>,
    model_catalog: HashMap<String, ModelRequestMetadata>,
    prelude: Vec<PromptMessage>,
    history: Vec<HistoryItem>,
    evidence: Vec<EvidenceRecord>,
    tools: ToolRegistry,
    permission_policy: PermissionPolicy,
    current_turn: Option<WorkflowTurnState>,
    todos: Vec<TodoItem>,
    auto_continue: AutoContinueState,
    max_iterations: usize,
    max_tool_calls: usize,
}

impl<C: Config> Agent<C> {
    pub fn new(
        client: Client<C>,
        model: impl Into<String>,
        max_iterations: usize,
        max_tool_calls: usize,
    ) -> Self {
        Self {
            client,
            model: model.into(),
            default_protocol: ApiProtocol::Responses,
            model_protocols: HashMap::new(),
            model_catalog: HashMap::new(),
            prelude: default_agent_prelude(),
            history: vec![],
            evidence: vec![],
            tools: ToolRegistry::default_tools(),
            permission_policy: PermissionPolicy::default(),
            current_turn: None,
            todos: Vec::new(),
            auto_continue: AutoContinueState::default(),
            max_iterations: max_iterations,
            max_tool_calls,
        }
    }

    pub fn set_model_catalog(&mut self, catalog: HashMap<String, ModelRequestMetadata>) {
        self.model_catalog = catalog;
    }

    pub fn set_default_protocol(&mut self, protocol: ApiProtocol) {
        self.default_protocol = protocol;
    }

    pub fn set_model_protocols(&mut self, protocols: HashMap<String, ApiProtocol>) {
        self.model_protocols = protocols;
    }

    fn active_protocol(&self) -> ApiProtocol {
        self.model_protocols
            .get(&self.model)
            .copied()
            .unwrap_or(self.default_protocol)
    }

    fn active_model_metadata(&self) -> ModelRequestMetadata {
        self.model_catalog
            .get(&self.model)
            .copied()
            .unwrap_or(ModelRequestMetadata {
                context_window: None,
                max_output_tokens: None,
                // Backward compatible default: historically tools were always advertised.
                // If a model isn't in the catalog, we assume tools are supported.
                supports_tools: true,
                supports_reasoning: false,
                ..Default::default()
            })
    }

    pub fn permission_mode(&self) -> PermissionMode {
        self.permission_policy.mode()
    }

    pub fn set_permission_mode(&mut self, mode: PermissionMode) {
        self.permission_policy.set_mode(mode);
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn set_model(&mut self, model: impl Into<String>) {
        self.model = model.into();
    }

    #[allow(dead_code)]
    pub fn restore_transcript_messages(&mut self, messages: Vec<ConversationMessage>) {
        self.history = messages
            .into_iter()
            .map(|message| match message.role {
                ConversationRole::User => HistoryItem::user(message.content),
                ConversationRole::Assistant => HistoryItem::assistant(message.content),
            })
            .collect();
    }

    #[allow(dead_code)]
    pub fn restore_evidence(&mut self, evidence: Vec<EvidenceRecord>) -> Result<()> {
        let mut restored = Vec::with_capacity(evidence.len());
        for record in evidence {
            require_unique_evidence_id(&restored, &record.id)?;
            restored.push(record);
        }
        self.evidence = restored;
        Ok(())
    }

    pub fn restore_session_context(
        &mut self,
        messages: Vec<ConversationMessage>,
        evidence: Vec<EvidenceRecord>,
    ) -> Result<()> {
        let mut restored_evidence = Vec::with_capacity(evidence.len());
        for record in evidence {
            require_unique_evidence_id(&restored_evidence, &record.id)?;
            restored_evidence.push(record);
        }
        let restored_history = messages
            .into_iter()
            .map(|message| match message.role {
                ConversationRole::User => HistoryItem::user(message.content),
                ConversationRole::Assistant => HistoryItem::assistant(message.content),
            })
            .collect();

        self.history = restored_history;
        self.evidence = restored_evidence;
        Ok(())
    }

    pub fn add_evidence(&mut self, evidence: EvidenceRecord) -> Result<()> {
        require_unique_evidence_id(&self.evidence, &evidence.id)?;
        self.evidence.push(evidence);
        Ok(())
    }

    #[allow(dead_code)]
    pub fn evidence(&self) -> &[EvidenceRecord] {
        &self.evidence
    }

    #[allow(dead_code)]
    pub fn register_tool<T>(&mut self, tool: T)
    where
        T: ToolHandler + 'static,
    {
        self.tools.register(tool);
    }

    pub fn try_register_tool<T>(&mut self, tool: T) -> Result<()>
    where
        T: ToolHandler + 'static,
    {
        self.tools.try_register(tool)
    }

    #[allow(dead_code)]
    pub async fn run(&mut self, user_input: &str) -> Result<String> {
        self.run_stream(user_input, |_| Ok(()), |_| Ok(()), |_| Ok(true))
            .await
    }

    pub async fn run_stream_async<F, E, A, Dfut, Efut, Afut>(
        &mut self,
        user_input: &str,
        on_delta: F,
        on_event: E,
        approve: A,
    ) -> Result<String>
    where
        F: FnMut(&str) -> Dfut,
        E: FnMut(AgentEvent) -> Efut,
        A: FnMut(PermissionRequest) -> Afut,
        Dfut: Future<Output = Result<()>>,
        Efut: Future<Output = Result<()>>,
        Afut: Future<Output = Result<bool>>,
    {
        match self.active_protocol() {
            ApiProtocol::Responses => {
                self.run_responses_stream_async(user_input, on_delta, on_event, approve)
                    .await
            }
            ApiProtocol::Completions => {
                self.run_oai_comp_stream_async(user_input, on_delta, on_event, approve)
                    .await
            }
        }
    }

    async fn run_responses_stream_async<F, E, A, Dfut, Efut, Afut>(
        &mut self,
        user_input: &str,
        mut on_delta: F,
        mut on_event: E,
        mut approve: A,
    ) -> Result<String>
    where
        F: FnMut(&str) -> Dfut,
        E: FnMut(AgentEvent) -> Efut,
        A: FnMut(PermissionRequest) -> Afut,
        Dfut: Future<Output = Result<()>>,
        Efut: Future<Output = Result<()>>,
        Afut: Future<Output = Result<bool>>,
    {
        let turn_prelude = self.prepare_turn_prelude(user_input);
        let protected_start_index = self.history.len();
        self.history.push(HistoryItem::user(user_input));
        debug!(
            user_input_len = user_input.len(),
            history_len = self.history.len(),
            "user message added to history"
        );

        let mut final_text = String::new();
        let mut tool_call_count = 0;
        let mut continuation_count = 0;

        for iteration in 0..self.max_iterations {
            let mut completed_reasoning_ids = HashSet::new();
            let mut turn_text = String::new();
            debug!(
                iteration,
                model = %self.model,
                history_len = self.history.len(),
                tool_call_count,
                max_tool_calls = self.max_tool_calls,
                "creating streamed response"
            );

            let tool_definitions = self.tools.specs();
            let build = build_request(RequestBuilderInput {
                protocol: ApiProtocol::Responses,
                model_id: &self.model,
                model: self.active_model_metadata(),
                prelude: &turn_prelude,
                history: &self.history,
                protected_start_index,
                tools: &tool_definitions,
                evidence: &self.evidence,
            })?;
            on_event(AgentEvent::TokenUsageUpdated {
                used_tokens: build.budget.estimated_request_tokens,
                context_window_tokens: build.budget.context_window_tokens,
            })
            .await?;
            if build.budget.truncated {
                debug!(
                    model = %self.model,
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

            let mut stream = self.client.responses().create_stream(request).await?;
            let mut completed_response: Option<Response> = None;

            while let Some(event) = stream.next().await {
                match event? {
                    ResponseStreamEvent::ResponseOutputTextDelta(event) => {
                        trace!(delta_len = event.delta.len(), "received text delta");
                        on_delta(&event.delta).await?;
                        turn_text.push_str(&event.delta);
                        final_text.push_str(&event.delta);
                    }
                    ResponseStreamEvent::ResponseReasoningSummaryTextDelta(event) => {
                        on_event(AgentEvent::ReasoningDelta {
                            item_id: event.item_id,
                            delta: event.delta,
                        })
                        .await?;
                    }
                    ResponseStreamEvent::ResponseReasoningSummaryTextDone(event) => {
                        completed_reasoning_ids.insert(event.item_id.clone());
                        on_event(AgentEvent::ReasoningDone {
                            item_id: event.item_id,
                            text: event.text,
                        })
                        .await?;
                    }
                    ResponseStreamEvent::ResponseCompleted(event) => {
                        debug!(
                            response_id = %event.response.id,
                            output_items = event.response.output.len(),
                            "streamed response completed"
                        );
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

            let response = completed_response
                .ok_or_else(|| anyhow!("stream ended without response.completed"))?;

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

            if tool_call_count + tool_calls.len() > self.max_tool_calls {
                return Err(anyhow!(
                    "stopped: too many tool calls ({} requested, max {})",
                    tool_call_count + tool_calls.len(),
                    self.max_tool_calls
                ));
            }

            tool_call_count += tool_calls.len();

            if tool_calls.is_empty() {
                if turn_text.is_empty() {
                    turn_text = response
                        .output_text()
                        .unwrap_or_else(|| "No response content".to_string());
                    final_text.push_str(&turn_text);
                }

                self.history.push(HistoryItem::assistant(turn_text.clone()));

                if self
                    .continue_after_no_tool_reply(&mut on_event, &mut continuation_count)
                    .await?
                {
                    continue;
                }

                info!(
                    output_chars = final_text.chars().count(),
                    history_len = self.history.len(),
                    "final answer completed"
                );

                return Ok(final_text);
            }

            self.history.push(HistoryItem::AssistantToolCalls {
                text: if turn_text.is_empty() {
                    None
                } else {
                    Some(turn_text.clone())
                },
                calls: tool_calls.clone(),
            });

            debug!(
                iteration,
                tool_calls = tool_calls.len(),
                tool_call_count,
                history_len = self.history.len(),
                "response tool calls appended to history"
            );

            for call in tool_calls {
                info!(
                    tool_name = %call.name,
                    call_id = %call.call_id,
                    "tool call requested"
                );
                debug!(
                    tool_name = %call.name,
                    call_id = %call.call_id,
                    arguments = %call.arguments_json,
                    "tool call arguments"
                );

                let output = self
                    .execute_tool_call(&call, &mut on_event, &mut approve)
                    .await?;

                debug!(
                    tool_name = %call.name,
                    call_id = %call.call_id,
                    output = ?output,
                    "tool call completed"
                );

                let output_json = serde_json::to_string(&output)?;

                self.history.push(HistoryItem::ToolOutput {
                    call_id: call.call_id.clone(),
                    output_json,
                });

                debug!(
                    history_len = self.history.len(),
                    "tool output appended to history"
                );

                let evidence = self.remember_tool_evidence(&call, &output)?;
                on_event(AgentEvent::EvidenceRecorded(evidence)).await?;
            }
        }

        Err(anyhow!(
            "stopped: too many agent iterations (max {})",
            self.max_iterations
        ))
    }

    async fn run_oai_comp_stream_async<F, E, A, Dfut, Efut, Afut>(
        &mut self,
        user_input: &str,
        mut on_delta: F,
        mut on_event: E,
        mut approve: A,
    ) -> Result<String>
    where
        F: FnMut(&str) -> Dfut,
        E: FnMut(AgentEvent) -> Efut,
        A: FnMut(PermissionRequest) -> Afut,
        Dfut: Future<Output = Result<()>>,
        Efut: Future<Output = Result<()>>,
        Afut: Future<Output = Result<bool>>,
    {
        let turn_prelude = self.prepare_turn_prelude(user_input);
        let protected_start_index = self.history.len();
        self.history.push(HistoryItem::user(user_input));
        debug!(
            user_input_len = user_input.len(),
            history_len = self.history.len(),
            "user message added to history"
        );

        let mut final_text = String::new();
        let mut tool_call_count = 0;
        let mut continuation_count = 0;

        for iteration in 0..self.max_iterations {
            debug!(
                iteration,
                model = %self.model,
                history_len = self.history.len(),
                tool_call_count,
                max_tool_calls = self.max_tool_calls,
                "creating streamed chat completion"
            );

            let tool_definitions = self.tools.specs();
            let build = build_request(RequestBuilderInput {
                protocol: ApiProtocol::Completions,
                model_id: &self.model,
                model: self.active_model_metadata(),
                prelude: &turn_prelude,
                history: &self.history,
                protected_start_index,
                tools: &tool_definitions,
                evidence: &self.evidence,
            })?;
            on_event(AgentEvent::TokenUsageUpdated {
                used_tokens: build.budget.estimated_request_tokens,
                context_window_tokens: build.budget.context_window_tokens,
            })
            .await?;
            if build.budget.truncated {
                debug!(
                    model = %self.model,
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

            let mut stream = self.client.chat().create_stream(request).await?;
            let mut turn_text = String::new();
            let mut tool_calls: BTreeMap<usize, ChatCompletionMessageToolCall> = BTreeMap::new();
            let mut finish_reasons: Vec<FinishReason> = Vec::new();
            let mut reasoning =
                InlineReasoningExtractor::new(format!("chat-reasoning-{iteration}"));

            while let Some(event) = stream.next().await {
                let response = event?;
                for choice in response.choices {
                    if choice.index != 0 {
                        return Err(anyhow!(
                            "completions returned unexpected choice index {}; only n=1/index 0 is supported",
                            choice.index
                        ));
                    }

                    if let Some(delta) = choice.delta.content {
                        trace!(delta_len = delta.len(), "received chat text delta");
                        for part in reasoning.push(&delta) {
                            match part {
                                StreamTextPart::Visible(text) => {
                                    on_delta(&text).await?;
                                    turn_text.push_str(&text);
                                    final_text.push_str(&text);
                                }
                                StreamTextPart::ReasoningDelta { item_id, delta } => {
                                    on_event(AgentEvent::ReasoningDelta { item_id, delta }).await?;
                                }
                                StreamTextPart::ReasoningDone { item_id, text } => {
                                    on_event(AgentEvent::ReasoningDone { item_id, text }).await?;
                                }
                            }
                        }
                    }

                    if let Some(chunks) = choice.delta.tool_calls {
                        for chunk in chunks {
                            merge_chat_tool_call_chunk(&mut tool_calls, chunk);
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
                        on_delta(&text).await?;
                        turn_text.push_str(&text);
                        final_text.push_str(&text);
                    }
                    StreamTextPart::ReasoningDelta { item_id, delta } => {
                        on_event(AgentEvent::ReasoningDelta { item_id, delta }).await?;
                    }
                    StreamTextPart::ReasoningDone { item_id, text } => {
                        on_event(AgentEvent::ReasoningDone { item_id, text }).await?;
                    }
                }
            }

            let has_tool_calls = !tool_calls.is_empty();
            validate_chat_finish_reasons(&finish_reasons, has_tool_calls)?;

            if !has_tool_calls {
                if final_text.is_empty() {
                    final_text = "No response content".to_string();
                }

                self.history.push(HistoryItem::assistant(turn_text.clone()));

                if self
                    .continue_after_no_tool_reply(&mut on_event, &mut continuation_count)
                    .await?
                {
                    continue;
                }

                info!(
                    output_chars = final_text.chars().count(),
                    history_len = self.history.len(),
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

            if tool_call_count + tool_calls.len() > self.max_tool_calls {
                return Err(anyhow!(
                    "stopped: too many tool calls ({} requested, max {})",
                    tool_call_count + tool_calls.len(),
                    self.max_tool_calls
                ));
            }

            tool_call_count += tool_calls.len();
            self.history.push(HistoryItem::AssistantToolCalls {
                text: if turn_text.is_empty() {
                    None
                } else {
                    Some(turn_text.clone())
                },
                calls: tool_calls.clone(),
            });

            for call in tool_calls {
                info!(
                    tool_name = %call.name,
                    call_id = %call.call_id,
                    "chat tool call requested"
                );
                debug!(
                    tool_name = %call.name,
                    call_id = %call.call_id,
                    arguments = %call.arguments_json,
                    "chat tool call arguments"
                );

                let output = self
                    .execute_tool_call(&call, &mut on_event, &mut approve)
                    .await?;

                let output_json = serde_json::to_string(&output)?;
                self.history.push(HistoryItem::ToolOutput {
                    call_id: call.call_id.clone(),
                    output_json,
                });

                let evidence = self.remember_tool_evidence(&call, &output)?;
                on_event(AgentEvent::EvidenceRecorded(evidence)).await?;
            }
        }

        Err(anyhow!(
            "stopped: too many agent iterations (max {})",
            self.max_iterations
        ))
    }

    async fn execute_tool_call<E, A, Efut, Afut>(
        &mut self,
        call: &HistoryToolCall,
        on_event: &mut E,
        approve: &mut A,
    ) -> Result<ToolResult>
    where
        E: FnMut(AgentEvent) -> Efut,
        A: FnMut(PermissionRequest) -> Afut,
        Efut: Future<Output = Result<()>>,
        Afut: Future<Output = Result<bool>>,
    {
        let output = match serde_json::from_str::<Value>(&call.arguments_json) {
            Ok(args) => {
                let directive = self
                    .current_turn
                    .as_ref()
                    .map(|turn| turn.directive)
                    .unwrap_or(ExecutionDirective::None);

                let permission_class = self.tools.permission_class(&call.name);

                if let Some(message) = restricted_by_directive_with_class(
                    &call.name,
                    &args,
                    permission_class,
                    directive,
                ) {
                    let output = ToolResult::err(&call.name, message);
                    on_event(AgentEvent::ToolCallFinished {
                        call_id: call.call_id.clone(),
                        name: call.name.clone(),
                        ok: output.ok,
                        output: output.clone(),
                    })
                    .await?;
                    return Ok(output);
                }

                let permission_decision = self.permission_policy.check_class_with_directive(
                    &call.name,
                    &args,
                    permission_class,
                    directive,
                );
                let should_execute = if is_workflow_control_tool(&call.name) {
                    true
                } else {
                    match permission_decision {
                        PermissionDecision::Allow => true,
                        PermissionDecision::Ask => {
                            approve(PermissionRequest {
                                call_id: Some(call.call_id.clone()),
                                tool: call.name.clone(),
                                args: args.clone(),
                                class: permission_class,
                                summary: format_tool_call(&call.name, &args),
                                preview: None,
                            })
                            .await?
                        }
                        PermissionDecision::Deny => false,
                    }
                };

                if should_execute {
                    on_event(AgentEvent::ToolCallStarted {
                        call_id: call.call_id.clone(),
                        name: call.name.clone(),
                        args: args.clone(),
                    })
                    .await?;

                    let output = self.tools.call(&call.name, args.clone()).await;

                    if output.ok {
                        self.apply_control_tool_state(&call.name, &args, on_event)
                            .await?;
                    }

                    on_event(AgentEvent::ToolCallFinished {
                        call_id: call.call_id.clone(),
                        name: call.name.clone(),
                        ok: output.ok,
                        output: output.clone(),
                    })
                    .await?;

                    output
                } else {
                    let output = if matches!(permission_decision, PermissionDecision::Deny) {
                        ToolResult::err(&call.name, "permission denied by current mode")
                    } else {
                        ToolResult::err(&call.name, "user denied permission")
                    };
                    on_event(AgentEvent::ToolCallFinished {
                        call_id: call.call_id.clone(),
                        name: call.name.clone(),
                        ok: output.ok,
                        output: output.clone(),
                    })
                    .await?;
                    output
                }
            }
            Err(err) => {
                warn!(
                    tool_name = %call.name,
                    call_id = %call.call_id,
                    error = %err,
                    raw_arguments = %call.arguments_json,
                    "invalid tool call JSON arguments"
                );
                ToolResult::err(
                    &call.name,
                    format!(
                        "invalid JSON arguments: {err}; raw: {}",
                        call.arguments_json
                    ),
                )
            }
        };

        Ok(output)
    }

    fn remember_tool_evidence(
        &mut self,
        call: &HistoryToolCall,
        output: &ToolResult,
    ) -> Result<EvidenceRecord> {
        let args = serde_json::from_str::<Value>(&call.arguments_json).unwrap_or(Value::Null);
        let draft =
            EvidenceDraft::from_tool_result(call.call_id.clone(), call.name.clone(), args, output);
        let sequence = self.next_evidence_sequence();
        let id = draft
            .id
            .clone()
            .unwrap_or_else(|| format!("ev-agent-{sequence:06}"));
        let record = draft.into_record(id, sequence, 0)?;
        self.add_evidence(record.clone())?;
        Ok(record)
    }

    fn next_evidence_sequence(&self) -> u64 {
        self.evidence
            .iter()
            .map(|record| record.sequence)
            .max()
            .unwrap_or(0)
            .saturating_add(1)
    }

    pub async fn run_stream<F, E, A>(
        &mut self,
        user_input: &str,
        mut on_delta: F,
        mut on_event: E,
        mut approve: A,
    ) -> Result<String>
    where
        F: FnMut(&str) -> Result<()>,
        E: FnMut(AgentEvent) -> Result<()>,
        A: FnMut(PermissionRequest) -> Result<bool>,
    {
        self.run_stream_async(
            user_input,
            |delta| std::future::ready(on_delta(delta)),
            |event| std::future::ready(on_event(event)),
            |request| std::future::ready(approve(request)),
        )
        .await
    }

    fn prepare_turn_prelude(&mut self, user_input: &str) -> Vec<PromptMessage> {
        let turn = WorkflowTurnState::from_user_input(user_input);
        self.current_turn = Some(turn.clone());
        self.todos.clear();
        self.auto_continue = AutoContinueState::default();

        let mut turn_prelude = self.prelude.clone();
        if let Some(message) = turn.developer_context_message() {
            turn_prelude.push(message);
        }
        turn_prelude
    }

    async fn apply_control_tool_state<E, Efut>(
        &mut self,
        tool_name: &str,
        args: &Value,
        on_event: &mut E,
    ) -> Result<()>
    where
        E: FnMut(AgentEvent) -> Efut,
        Efut: Future<Output = Result<()>>,
    {
        match tool_name {
            "workflow__todos" => {
                let payload: WorkflowTodosPayload = serde_json::from_value(args.clone())?;
                self.todos = payload.items;
                on_event(AgentEvent::TodoSnapshotUpdated {
                    items: self.todos.clone(),
                })
                .await?;
            }
            "workflow__auto_continue" => {
                let payload: WorkflowAutoContinuePayload = serde_json::from_value(args.clone())?;
                self.auto_continue.enabled = payload.enabled;
                if let Some(max_continuations) = payload.max_continuations {
                    if max_continuations > AutoContinueState::ABSOLUTE_MAX_CONTINUATIONS {
                        return Err(anyhow!(
                            "max_continuations {max_continuations} exceeds maximum {}",
                            AutoContinueState::ABSOLUTE_MAX_CONTINUATIONS
                        ));
                    }
                    self.auto_continue.max_continuations = max_continuations;
                }
                on_event(AgentEvent::AutoContinueChanged {
                    state: self.auto_continue.clone(),
                })
                .await?;
            }
            _ => {}
        }

        Ok(())
    }

    async fn continue_after_no_tool_reply<E, Efut>(
        &mut self,
        _on_event: &mut E,
        continuation_count: &mut usize,
    ) -> Result<bool>
    where
        E: FnMut(AgentEvent) -> Efut,
        Efut: Future<Output = Result<()>>,
    {
        let Some(remaining_unfinished) = self.remaining_unfinished_todos() else {
            return Ok(false);
        };

        if !self.auto_continue.enabled {
            return Ok(false);
        }

        if *continuation_count >= self.auto_continue.max_continuations {
            return Err(anyhow!(
                "stopped: auto-continue limit reached (max {}, {} unfinished todo item{})",
                self.auto_continue.max_continuations,
                remaining_unfinished,
                if remaining_unfinished == 1 { "" } else { "s" }
            ));
        }

        *continuation_count += 1;
        self.history.push(HistoryItem::user(
            "Continue the current task internally. Do not repeat finished work. Focus on unfinished todo items and stop when they are complete or blocked.",
        ));
        Ok(true)
    }

    fn remaining_unfinished_todos(&self) -> Option<usize> {
        if self
            .todos
            .iter()
            .any(|todo| todo.status == TodoStatus::Blocked)
        {
            return None;
        }

        let unfinished = self
            .todos
            .iter()
            .filter(|todo| todo.status.is_unfinished())
            .count();
        (unfinished > 0).then_some(unfinished)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum TodoStatus {
    #[serde(rename = "pending")]
    Pending,
    #[serde(rename = "in_progress")]
    InProgress,
    #[serde(rename = "blocked")]
    Blocked,
    #[serde(rename = "completed")]
    Completed,
    #[serde(rename = "cancelled")]
    Cancelled,
}

impl TodoStatus {
    fn is_unfinished(&self) -> bool {
        matches!(self, Self::Pending | Self::InProgress)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TodoItem {
    pub id: String,
    pub content: String,
    pub status: TodoStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AutoContinueState {
    pub enabled: bool,
    pub max_continuations: usize,
}

impl AutoContinueState {
    const DEFAULT_MAX_CONTINUATIONS: usize = 3;
    const ABSOLUTE_MAX_CONTINUATIONS: usize = 8;
}

impl Default for AutoContinueState {
    fn default() -> Self {
        Self {
            enabled: false,
            max_continuations: Self::DEFAULT_MAX_CONTINUATIONS,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
struct WorkflowTodosPayload {
    items: Vec<TodoItem>,
}

#[derive(Debug, Clone, Deserialize)]
struct WorkflowAutoContinuePayload {
    enabled: bool,
    #[serde(default)]
    max_continuations: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TurnIntent {
    Lightweight,
    Engineering,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ValidationReminder {
    None,
    Focused,
    Targeted,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct WorkflowTurnState {
    intent: TurnIntent,
    validation: ValidationReminder,
    directive: ExecutionDirective,
}

impl WorkflowTurnState {
    fn from_user_input(user_input: &str) -> Self {
        let intent = classify_turn_intent(user_input);
        let validation = detect_validation_reminder(user_input, intent);
        let directive = detect_execution_directive(user_input);
        Self {
            intent,
            validation,
            directive,
        }
    }

    fn developer_context_message(&self) -> Option<PromptMessage> {
        if self.intent == TurnIntent::Lightweight {
            return None;
        }

        let mut text = ENGINEERING_WORKFLOW_PRELUDE.to_string();
        match self.validation {
            ValidationReminder::None => {}
            ValidationReminder::Focused => {
                text.push_str(
                    "\nIf you make code changes, run focused validation for the files or behavior you touched. If validation is not practical, say so explicitly.",
                );
            }
            ValidationReminder::Targeted => {
                text.push_str(
                    "\nPlan to run the most relevant targeted validation for this task, such as the affected tests, build, or lint command. If you skip validation, say why explicitly.",
                );
            }
        }

        match self.directive {
            ExecutionDirective::None => {}
            ExecutionDirective::ReadOnly => {
                text.push_str(
                    "\nThis turn is read-only. Do not modify files or run non-read-only commands.",
                );
            }
            ExecutionDirective::PlanOnly => {
                text.push_str(
                    "\nThis turn is plan-only. Produce analysis and planning only. Do not modify files or run non-read-only commands.",
                );
            }
            ExecutionDirective::AnalyzeOnly => {
                text.push_str(
                    "\nThis turn is analyze-only. Inspect and explain only. Do not modify files or run non-read-only commands.",
                );
            }
            ExecutionDirective::DoNotEdit => {
                text.push_str(
                    "\nThis turn has an explicit do-not-edit directive. Do not modify files or run non-read-only commands.",
                );
            }
        }

        Some(PromptMessage::developer(text))
    }
}

fn detect_execution_directive(user_input: &str) -> ExecutionDirective {
    let normalized = normalize_for_intent(user_input);

    if contains_any(&normalized, &["read-only", "read only", "readonly", "只读"]) {
        ExecutionDirective::ReadOnly
    } else if contains_any(
        &normalized,
        &[
            "plan-only",
            "plan only",
            "planning only",
            "only plan",
            "just plan",
            "只做计划",
        ],
    ) {
        ExecutionDirective::PlanOnly
    } else if contains_any(
        &normalized,
        &[
            "analyze-only",
            "analyze only",
            "analysis only",
            "only analyze",
            "only analyse",
            "只分析",
        ],
    ) {
        ExecutionDirective::AnalyzeOnly
    } else if contains_any(
        &normalized,
        &[
            "do not edit",
            "don't edit",
            "dont edit",
            "no edits",
            "不要修改",
        ],
    ) {
        ExecutionDirective::DoNotEdit
    } else {
        ExecutionDirective::None
    }
}

fn classify_turn_intent(user_input: &str) -> TurnIntent {
    let normalized = normalize_for_intent(user_input);

    if contains_engineering_signal(&normalized) {
        TurnIntent::Engineering
    } else {
        TurnIntent::Lightweight
    }
}

fn detect_validation_reminder(user_input: &str, intent: TurnIntent) -> ValidationReminder {
    if intent == TurnIntent::Lightweight {
        return ValidationReminder::None;
    }

    let normalized = normalize_for_intent(user_input);
    if contains_any(
        &normalized,
        &[
            "cargo test",
            "cargo check",
            "cargo clippy",
            "test ",
            "tests ",
            "build ",
            "compile",
            "lint",
        ],
    ) {
        ValidationReminder::Targeted
    } else if contains_any(
        &normalized,
        &[
            "fix",
            "implement",
            "add",
            "update",
            "modify",
            "refactor",
            "rename",
            "remove",
            "create",
            "write",
            "edit",
            "patch",
            "bug",
            "failing",
            "regression",
        ],
    ) {
        ValidationReminder::Focused
    } else {
        ValidationReminder::None
    }
}

fn contains_engineering_signal(normalized: &str) -> bool {
    contains_any(
        normalized,
        &[
            "fix",
            "implement",
            "add",
            "update",
            "modify",
            "refactor",
            "rename",
            "remove",
            "create",
            "write",
            "edit",
            "patch",
            "debug",
            "investigate",
            "trace",
            "root cause",
            "complex analysis",
            "full analysis",
            "workflow",
            "codebase",
            "repository",
            "repo",
            "project",
            "module",
            "crate",
            "src/",
            "cargo ",
            "test ",
            "tests ",
            "build ",
            "compile",
            "lint",
            "multi-step",
            "step by step",
            "plan",
            "pipeline",
            "across",
            "multiple files",
            "复杂任务",
            "复杂分析",
            "工程",
            "实现",
            "修改",
            "修复",
            "重构",
            "调试",
            "排查",
            "计划",
            "当前项目",
        ],
    )
}

fn normalize_for_intent(user_input: &str) -> String {
    user_input.to_ascii_lowercase()
}

fn contains_any(text: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| text.contains(needle))
}

fn is_workflow_control_tool(tool_name: &str) -> bool {
    matches!(tool_name, "workflow__todos" | "workflow__auto_continue")
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_openai::config::OpenAIConfig;

    fn test_agent() -> Agent<OpenAIConfig> {
        let client = Client::with_config(
            OpenAIConfig::new()
                .with_api_base("https://api.openai.com/v1")
                .with_api_key("test"),
        );
        Agent::new(client, "m1", 4, 4)
    }

    #[test]
    fn model_switch_uses_new_metadata_for_next_request_build() {
        let client = Client::with_config(
            OpenAIConfig::new()
                .with_api_base("https://api.openai.com/v1")
                .with_api_key("test"),
        );
        let mut agent = Agent::new(client, "m1", 1, 1);

        let mut catalog = HashMap::new();
        catalog.insert(
            "m1".to_string(),
            ModelRequestMetadata {
                context_window: Some(2048),
                max_output_tokens: Some(256),
                supports_tools: true,
                supports_reasoning: false,
                ..Default::default()
            },
        );
        catalog.insert(
            "m2".to_string(),
            ModelRequestMetadata {
                context_window: Some(128_000),
                max_output_tokens: Some(256),
                supports_tools: true,
                supports_reasoning: false,
                ..Default::default()
            },
        );
        agent.set_model_catalog(catalog);

        // Simulate first user message.
        agent.history.push(HistoryItem::user("hello"));
        let b1 = build_request(RequestBuilderInput {
            protocol: ApiProtocol::Responses,
            model_id: agent.model(),
            model: agent.active_model_metadata(),
            prelude: &agent.prelude,
            history: &agent.history,
            protected_start_index: agent.history.len().saturating_sub(1),
            tools: &[],
            evidence: &[],
        })
        .expect("request builds");
        assert_eq!(b1.budget.context_window_tokens, 2048.max(1024));

        // Switch model and build again.
        agent.set_model("m2");
        let b2 = build_request(RequestBuilderInput {
            protocol: ApiProtocol::Responses,
            model_id: agent.model(),
            model: agent.active_model_metadata(),
            prelude: &agent.prelude,
            history: &agent.history,
            protected_start_index: agent.history.len().saturating_sub(1),
            tools: &[],
            evidence: &[],
        })
        .expect("request builds");
        assert!(b2.budget.context_window_tokens > b1.budget.context_window_tokens);
    }

    #[test]
    fn inline_reasoning_extractor_splits_think_tags_from_visible_text() {
        let mut extractor = InlineReasoningExtractor::new("r-1");

        let mut parts = extractor.push("hello <thi");
        parts.extend(extractor.push("nk>plan</think> answer"));
        parts.extend(extractor.finish());

        assert_eq!(
            parts,
            vec![
                StreamTextPart::Visible("hello ".into()),
                StreamTextPart::ReasoningDelta {
                    item_id: "r-1".into(),
                    delta: "plan".into(),
                },
                StreamTextPart::ReasoningDone {
                    item_id: "r-1".into(),
                    text: "plan".into(),
                },
                StreamTextPart::Visible(" answer".into()),
            ]
        );
    }

    #[test]
    fn compact_indexed_chat_tool_calls_does_not_synthesize_missing_indices() {
        let mut indexed = BTreeMap::new();
        let mut call = ChatCompletionMessageToolCall::default();
        call.id = "call-1".into();
        call.function.name = "fs__write".into();
        call.function.arguments = r#"{"path":"a.txt","content":"ok"}"#.into();
        indexed.insert(1, call);

        let compacted = compact_indexed_chat_tool_calls(indexed);

        assert_eq!(compacted.len(), 1);
        assert_eq!(compacted[0].id, "call-1");
        assert_eq!(compacted[0].function.name, "fs__write");
        validate_chat_tool_calls(&compacted).expect("valid sparse-index tool call");
    }

    #[test]
    fn chat_tool_call_chunk_empty_name_does_not_overwrite_real_name() {
        let mut indexed = BTreeMap::new();
        for raw in [
            serde_json::json!({
                "index": 0,
                "id": "call-1",
                "type": "function",
                "function": {"name": "fs__write", "arguments": ""}
            }),
            serde_json::json!({
                "index": 0,
                "function": {"name": "", "arguments": "{\"path\":"}
            }),
            serde_json::json!({
                "index": 0,
                "function": {"name": "", "arguments": "\"a.txt\",\"content\":\"ok\"}"}
            }),
        ] {
            let chunk: ChatCompletionMessageToolCallChunk =
                serde_json::from_value(raw).expect("chunk deserializes");
            merge_chat_tool_call_chunk(&mut indexed, chunk);
        }

        let compacted = compact_indexed_chat_tool_calls(indexed);

        assert_eq!(compacted.len(), 1);
        assert_eq!(compacted[0].id, "call-1");
        assert_eq!(compacted[0].function.name, "fs__write");
        assert_eq!(
            compacted[0].function.arguments,
            r#"{"path":"a.txt","content":"ok"}"#
        );
        validate_chat_tool_calls(&compacted).expect("valid streamed tool call");
    }

    #[test]
    fn classifies_lightweight_and_engineering_turns() {
        assert_eq!(
            classify_turn_intent("Explain how Rust ownership works."),
            TurnIntent::Lightweight
        );
        assert_eq!(
            classify_turn_intent("Explain what this function does."),
            TurnIntent::Lightweight
        );
        assert_eq!(
            classify_turn_intent(
                "Fix the failing tests in src/agent.rs and update the implementation."
            ),
            TurnIntent::Engineering
        );
    }

    #[test]
    fn auto_continue_defaults_to_disabled() {
        let agent = test_agent();

        assert_eq!(agent.auto_continue, AutoContinueState::default());
        assert!(agent.todos.is_empty());
    }

    #[tokio::test]
    async fn workflow_auto_continue_tool_enables_bounded_state() {
        let mut agent = test_agent();
        let call = HistoryToolCall {
            call_id: "call-auto".into(),
            name: "workflow__auto_continue".into(),
            arguments_json: r#"{"enabled":true,"max_continuations":2}"#.into(),
        };

        let output = agent
            .execute_tool_call(&call, &mut |_| std::future::ready(Ok(())), &mut |_| {
                std::future::ready(Ok(true))
            })
            .await
            .expect("control tool should succeed");

        assert!(output.ok);
        assert_eq!(agent.auto_continue.enabled, true);
        assert_eq!(agent.auto_continue.max_continuations, 2);
    }

    #[tokio::test]
    async fn workflow_todos_tool_updates_todo_state() {
        let mut agent = test_agent();
        let call = HistoryToolCall {
            call_id: "call-todos".into(),
            name: "workflow__todos".into(),
            arguments_json: r#"{"items":[{"id":"t1","content":"first","status":"pending"},{"id":"t2","content":"done","status":"completed"}]}"#.into(),
        };

        agent
            .execute_tool_call(&call, &mut |_| std::future::ready(Ok(())), &mut |_| {
                std::future::ready(Ok(true))
            })
            .await
            .expect("todo control tool should succeed");

        assert_eq!(agent.todos.len(), 2);
        assert_eq!(agent.todos[0].status, TodoStatus::Pending);
        assert_eq!(agent.todos[1].status, TodoStatus::Completed);
    }

    #[tokio::test]
    async fn unfinished_todos_trigger_bounded_internal_continuation() {
        let mut agent = test_agent();
        agent.auto_continue = AutoContinueState {
            enabled: true,
            max_continuations: 2,
        };
        agent.todos = vec![TodoItem {
            id: "t1".into(),
            content: "keep going".into(),
            status: TodoStatus::InProgress,
        }];
        let mut continuation_count = 0;

        let should_continue = agent
            .continue_after_no_tool_reply(
                &mut |_| std::future::ready(Ok(())),
                &mut continuation_count,
            )
            .await
            .expect("continuation decision succeeds");

        assert!(should_continue);
        assert_eq!(continuation_count, 1);
        assert!(matches!(
            agent.history.last(),
            Some(HistoryItem::UserText { .. })
        ));
    }

    #[tokio::test]
    async fn completed_or_blocked_todos_stop_auto_continuation() {
        let mut agent = test_agent();
        agent.auto_continue.enabled = true;
        let mut continuation_count = 0;

        agent.todos = vec![TodoItem {
            id: "done".into(),
            content: "done".into(),
            status: TodoStatus::Completed,
        }];
        assert!(
            !agent
                .continue_after_no_tool_reply(
                    &mut |_| std::future::ready(Ok(())),
                    &mut continuation_count
                )
                .await
                .expect("completed todos should stop")
        );

        agent.todos = vec![TodoItem {
            id: "blocked".into(),
            content: "blocked".into(),
            status: TodoStatus::Blocked,
        }];
        assert!(
            !agent
                .continue_after_no_tool_reply(
                    &mut |_| std::future::ready(Ok(())),
                    &mut continuation_count
                )
                .await
                .expect("blocked todos should stop")
        );
    }

    #[tokio::test]
    async fn continuation_bound_is_runtime_enforced() {
        let mut agent = test_agent();
        agent.auto_continue = AutoContinueState {
            enabled: true,
            max_continuations: 1,
        };
        agent.todos = vec![TodoItem {
            id: "t1".into(),
            content: "still pending".into(),
            status: TodoStatus::Pending,
        }];
        let mut continuation_count = 1;

        let error = agent
            .continue_after_no_tool_reply(
                &mut |_| std::future::ready(Ok(())),
                &mut continuation_count,
            )
            .await
            .expect_err("limit should fail fast");

        assert!(error.to_string().contains("auto-continue limit reached"));
        assert_eq!(continuation_count, 1);
    }

    #[test]
    fn engineering_turn_prelude_adds_workflow_context_and_validation_reminder() {
        let client = Client::with_config(
            OpenAIConfig::new()
                .with_api_base("https://api.openai.com/v1")
                .with_api_key("test"),
        );
        let mut agent = Agent::new(client, "m1", 1, 1);

        let turn_prelude =
            agent.prepare_turn_prelude("Implement the fix in src/agent.rs and run cargo test.");

        assert_eq!(
            agent.current_turn.as_ref().map(|turn| turn.intent),
            Some(TurnIntent::Engineering)
        );
        assert_eq!(
            agent.current_turn.as_ref().map(|turn| turn.directive),
            Some(ExecutionDirective::None)
        );
        assert_eq!(turn_prelude.len(), agent.prelude.len() + 1);
        let workflow_message = &turn_prelude[turn_prelude.len() - 1];
        assert_eq!(
            workflow_message.role,
            crate::request_builder::PromptRole::Developer
        );
        assert!(workflow_message.text.contains("engineering workflow task"));
        assert!(workflow_message.text.contains("single-agent"));
        assert!(workflow_message.text.contains("targeted validation"));
    }

    #[test]
    fn lightweight_turn_prelude_stays_unmodified() {
        let client = Client::with_config(
            OpenAIConfig::new()
                .with_api_base("https://api.openai.com/v1")
                .with_api_key("test"),
        );
        let mut agent = Agent::new(client, "m1", 1, 1);

        let turn_prelude = agent.prepare_turn_prelude("Summarize what this tool does.");

        assert_eq!(
            agent.current_turn.as_ref().map(|turn| turn.intent),
            Some(TurnIntent::Lightweight)
        );
        assert_eq!(turn_prelude, agent.prelude);
    }

    #[test]
    fn detects_explicit_execution_directives() {
        assert_eq!(
            detect_execution_directive("Read-only: inspect src/permission.rs and summarize it."),
            ExecutionDirective::ReadOnly
        );
        assert_eq!(
            detect_execution_directive("Plan only. Do not edit anything yet."),
            ExecutionDirective::PlanOnly
        );
        assert_eq!(
            detect_execution_directive("Analyze only and explain the failure."),
            ExecutionDirective::AnalyzeOnly
        );
        assert_eq!(
            detect_execution_directive("Please investigate, but do not edit files."),
            ExecutionDirective::DoNotEdit
        );
    }

    #[tokio::test]
    async fn execute_tool_call_blocks_write_tools_under_read_only_directive() {
        let client = Client::with_config(
            OpenAIConfig::new()
                .with_api_base("https://api.openai.com/v1")
                .with_api_key("test"),
        );
        let mut agent = Agent::new(client, "m1", 1, 1);
        agent.current_turn = Some(WorkflowTurnState::from_user_input(
            "Read-only: inspect and report only.",
        ));

        let call = HistoryToolCall {
            call_id: "call-1".into(),
            name: "fs__write".into(),
            arguments_json: r#"{"path":"a.txt","content":"x"}"#.into(),
        };
        let mut events = Vec::new();

        let output = agent
            .execute_tool_call(
                &call,
                &mut |event| {
                    events.push(event);
                    std::future::ready(Ok(()))
                },
                &mut |_| std::future::ready(Ok(true)),
            )
            .await
            .expect("tool call should complete with visible error");

        assert!(!output.ok);
        assert!(
            output
                .error
                .as_ref()
                .expect("error payload")
                .message
                .contains("read_only directive")
        );
        assert!(matches!(
            events.as_slice(),
            [AgentEvent::ToolCallFinished { .. }]
        ));
    }

    #[tokio::test]
    async fn execute_tool_call_blocks_non_read_only_commands_under_read_only_directive() {
        let client = Client::with_config(
            OpenAIConfig::new()
                .with_api_base("https://api.openai.com/v1")
                .with_api_key("test"),
        );
        let mut agent = Agent::new(client, "m1", 1, 1);
        agent.current_turn = Some(WorkflowTurnState::from_user_input(
            "Read only. Analyze and report.",
        ));

        let call = HistoryToolCall {
            call_id: "call-2".into(),
            name: "shell__exec".into(),
            arguments_json: r#"{"command":"cargo test permission::tests"}"#.into(),
        };

        let output = agent
            .execute_tool_call(&call, &mut |_| std::future::ready(Ok(())), &mut |_| {
                std::future::ready(Ok(true))
            })
            .await
            .expect("tool call should complete with visible error");

        assert!(!output.ok);
        assert!(
            output
                .error
                .as_ref()
                .expect("error payload")
                .message
                .contains("not read-only compatible")
        );
    }

    #[tokio::test]
    async fn execute_tool_call_emits_finished_event_for_policy_denial() {
        let client = Client::with_config(
            OpenAIConfig::new()
                .with_api_base("https://api.openai.com/v1")
                .with_api_key("test"),
        );
        let mut agent = Agent::new(client, "m1", 1, 1);
        let call = HistoryToolCall {
            call_id: "call-denied".into(),
            name: "shell__exec".into(),
            arguments_json: r#"{"command":"rm -rf target"}"#.into(),
        };
        let mut events = Vec::new();

        let output = agent
            .execute_tool_call(
                &call,
                &mut |event| {
                    events.push(event);
                    std::future::ready(Ok(()))
                },
                &mut |_| std::future::ready(Ok(true)),
            )
            .await
            .expect("policy denial should be reported as tool output");

        assert!(!output.ok);
        assert!(matches!(
            events.as_slice(),
            [AgentEvent::ToolCallFinished { ok: false, .. }]
        ));
    }
}

fn default_agent_prelude() -> Vec<PromptMessage> {
    vec![PromptMessage::developer(DEFAULT_AGENT_PRELUDE)]
}

fn reasoning_summary_text(item: &OutputItem) -> String {
    match item {
        OutputItem::Reasoning(reasoning) => reasoning
            .summary
            .iter()
            .map(|part| match part {
                async_openai::types::responses::SummaryPart::SummaryText(content) => {
                    content.text.clone()
                }
            })
            .filter(|text| !text.trim().is_empty())
            .collect::<Vec<_>>()
            .join("\n\n"),
        _ => String::new(),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum StreamTextPart {
    Visible(String),
    ReasoningDelta { item_id: String, delta: String },
    ReasoningDone { item_id: String, text: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InlineReasoningMode {
    Visible,
    Reasoning,
}

#[derive(Debug, Clone)]
struct InlineReasoningExtractor {
    item_id: String,
    mode: InlineReasoningMode,
    buffer: String,
    reasoning_text: String,
}

impl InlineReasoningExtractor {
    fn new(item_id: impl Into<String>) -> Self {
        Self {
            item_id: item_id.into(),
            mode: InlineReasoningMode::Visible,
            buffer: String::new(),
            reasoning_text: String::new(),
        }
    }

    fn push(&mut self, text: &str) -> Vec<StreamTextPart> {
        self.buffer.push_str(text);
        self.drain(false)
    }

    fn finish(&mut self) -> Vec<StreamTextPart> {
        self.drain(true)
    }

    fn drain(&mut self, finishing: bool) -> Vec<StreamTextPart> {
        let mut parts = Vec::new();

        loop {
            match self.mode {
                InlineReasoningMode::Visible => {
                    if let Some((start, len)) = find_open_reasoning_tag(&self.buffer) {
                        let visible = self.buffer[..start].to_string();
                        if !visible.is_empty() {
                            parts.push(StreamTextPart::Visible(visible));
                        }
                        self.buffer.drain(..start + len);
                        self.mode = InlineReasoningMode::Reasoning;
                        continue;
                    }

                    let emit_len = if finishing {
                        self.buffer.len()
                    } else {
                        safe_emit_len_without_partial_tag(&self.buffer, OPEN_REASONING_TAGS)
                    };
                    if emit_len == 0 {
                        break;
                    }
                    let visible = self.buffer[..emit_len].to_string();
                    self.buffer.drain(..emit_len);
                    parts.push(StreamTextPart::Visible(visible));
                }
                InlineReasoningMode::Reasoning => {
                    if let Some((start, len)) = find_close_reasoning_tag(&self.buffer) {
                        let delta = self.buffer[..start].to_string();
                        if !delta.is_empty() {
                            self.reasoning_text.push_str(&delta);
                            parts.push(StreamTextPart::ReasoningDelta {
                                item_id: self.item_id.clone(),
                                delta,
                            });
                        }
                        self.buffer.drain(..start + len);
                        parts.push(StreamTextPart::ReasoningDone {
                            item_id: self.item_id.clone(),
                            text: self.reasoning_text.clone(),
                        });
                        self.mode = InlineReasoningMode::Visible;
                        continue;
                    }

                    let emit_len = if finishing {
                        self.buffer.len()
                    } else {
                        safe_emit_len_without_partial_tag(&self.buffer, CLOSE_REASONING_TAGS)
                    };
                    if emit_len == 0 {
                        break;
                    }
                    let delta = self.buffer[..emit_len].to_string();
                    self.buffer.drain(..emit_len);
                    self.reasoning_text.push_str(&delta);
                    parts.push(StreamTextPart::ReasoningDelta {
                        item_id: self.item_id.clone(),
                        delta,
                    });
                }
            }
        }

        if finishing && matches!(self.mode, InlineReasoningMode::Reasoning) {
            parts.push(StreamTextPart::ReasoningDone {
                item_id: self.item_id.clone(),
                text: self.reasoning_text.clone(),
            });
            self.mode = InlineReasoningMode::Visible;
        }

        parts
    }
}

const OPEN_REASONING_TAGS: &[&str] = &["<think>", "<thinking>"];
const CLOSE_REASONING_TAGS: &[&str] = &["</think>", "</thinking>"];

fn find_open_reasoning_tag(text: &str) -> Option<(usize, usize)> {
    find_earliest_tag(text, OPEN_REASONING_TAGS)
}

fn find_close_reasoning_tag(text: &str) -> Option<(usize, usize)> {
    find_earliest_tag(text, CLOSE_REASONING_TAGS)
}

fn find_earliest_tag(text: &str, tags: &[&str]) -> Option<(usize, usize)> {
    tags.iter()
        .filter_map(|tag| text.find(tag).map(|index| (index, tag.len())))
        .min_by_key(|(index, _)| *index)
}

fn safe_emit_len_without_partial_tag(text: &str, tags: &[&str]) -> usize {
    for hold in (1..=max_tag_len(tags).saturating_sub(1)).rev() {
        if text.len() >= hold {
            let suffix_start = next_char_boundary(text, text.len() - hold);
            let suffix = &text[suffix_start..];
            if tags.iter().any(|tag| tag.starts_with(suffix)) {
                return suffix_start;
            }
        }
    }
    text.len()
}

fn max_tag_len(tags: &[&str]) -> usize {
    tags.iter().map(|tag| tag.len()).max().unwrap_or(0)
}

fn next_char_boundary(text: &str, index: usize) -> usize {
    if text.is_char_boundary(index) {
        return index;
    }
    text.char_indices()
        .map(|(i, _)| i)
        .find(|i| *i > index)
        .unwrap_or(text.len())
}

fn validate_chat_finish_reasons(reasons: &[FinishReason], has_tool_calls: bool) -> Result<()> {
    if reasons.is_empty() {
        return Err(anyhow!(
            "completions stream ended without finish_reason; cannot determine completion status"
        ));
    }

    for reason in reasons {
        match (reason, has_tool_calls) {
            (FinishReason::Stop, false) => {}
            (FinishReason::ToolCalls, true) | (FinishReason::FunctionCall, true) => {}
            (FinishReason::Length, _) => {
                return Err(anyhow!(
                    "completions response incomplete: finish_reason=length"
                ));
            }
            (FinishReason::ContentFilter, _) => {
                return Err(anyhow!(
                    "completions response filtered: finish_reason=content_filter"
                ));
            }
            (reason, _) => {
                return Err(anyhow!(
                    "unexpected completions finish_reason {:?} for {} response",
                    reason,
                    if has_tool_calls { "tool-call" } else { "text" }
                ));
            }
        }
    }

    Ok(())
}

fn validate_chat_tool_calls(tool_calls: &[ChatCompletionMessageToolCall]) -> Result<()> {
    for (index, call) in tool_calls.iter().enumerate() {
        if call.id.trim().is_empty() {
            return Err(anyhow!(
                "invalid completions tool call at index {index}: missing id"
            ));
        }
        if call.function.name.trim().is_empty() {
            return Err(anyhow!(
                "invalid completions tool call at index {index}: missing function name"
            ));
        }
        if call.function.arguments.trim().is_empty() {
            return Err(anyhow!(
                "invalid completions tool call at index {index}: missing function arguments"
            ));
        }
    }

    Ok(())
}

fn compact_indexed_chat_tool_calls(
    tool_calls: BTreeMap<usize, ChatCompletionMessageToolCall>,
) -> Vec<ChatCompletionMessageToolCall> {
    tool_calls.into_values().collect()
}

fn merge_chat_tool_call_chunk(
    tool_calls: &mut BTreeMap<usize, ChatCompletionMessageToolCall>,
    chunk: ChatCompletionMessageToolCallChunk,
) {
    let index = chunk.index as usize;
    let tool_call = tool_calls.entry(index).or_default();
    if let Some(id) = chunk.id.filter(|id| !id.trim().is_empty()) {
        tool_call.id = id;
    }
    if let Some(function) = chunk.function {
        if let Some(name) = function.name.filter(|name| !name.trim().is_empty()) {
            tool_call.function.name = name;
        }
        if let Some(arguments) = function.arguments {
            tool_call.function.arguments.push_str(&arguments);
        }
    }
}
