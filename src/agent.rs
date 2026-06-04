use anyhow::{Result, anyhow};
use async_openai::Client;
use async_openai::config::Config;
use async_openai::types::chat::{ChatCompletionMessageToolCall, FinishReason};
use async_openai::types::responses::{OutputItem, Response, ResponseStreamEvent};
use futures_util::StreamExt;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::future::Future;
use tracing::{debug, error, info, trace, warn};

use crate::config::ApiProtocol;
use crate::evidence::{EvidenceDraft, EvidenceRecord, require_unique_evidence_id};
use crate::permission::{
    PermissionDecision, PermissionMode, PermissionPolicy, PermissionRequest, classify_tool,
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
Use tools deliberately: read/search before editing, edit only intended files, and run focused validation after changes.
When requirements are ambiguous or risky, ask a concise clarifying question. Do not hide errors with fallbacks; fail fast and explain the actionable cause.
Keep responses concise. Summarize changed files and validation results when code was modified."#;

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
        let protected_start_index = self.history.len();
        self.history.push(HistoryItem::user(user_input));
        debug!(
            user_input_len = user_input.len(),
            history_len = self.history.len(),
            "user message added to history"
        );

        let mut text = String::new();
        let mut tool_call_count = 0;

        for iteration in 0..self.max_iterations {
            let mut completed_reasoning_ids = HashSet::new();
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
                prelude: &self.prelude,
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
                        text.push_str(&event.delta);
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
                if text.is_empty() {
                    text = response
                        .output_text()
                        .unwrap_or_else(|| "No response content".to_string());
                }

                self.history.push(HistoryItem::assistant(text.clone()));

                info!(
                    output_chars = text.chars().count(),
                    history_len = self.history.len(),
                    "final answer completed"
                );

                return Ok(text);
            }

            self.history.push(HistoryItem::AssistantToolCalls {
                text: if text.is_empty() {
                    None
                } else {
                    Some(text.clone())
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
                let evidence = self.remember_tool_evidence(&call, &output)?;
                on_event(AgentEvent::EvidenceRecorded(evidence)).await?;

                debug!(
                    tool_name = %call.name,
                    call_id = %call.call_id,
                    output = ?output,
                    "tool call completed"
                );

                let output = serde_json::to_string(&output)?;

                self.history.push(HistoryItem::ToolOutput {
                    call_id: call.call_id,
                    output_json: output,
                });

                debug!(
                    history_len = self.history.len(),
                    "tool output appended to history"
                );
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
        let protected_start_index = self.history.len();
        self.history.push(HistoryItem::user(user_input));
        debug!(
            user_input_len = user_input.len(),
            history_len = self.history.len(),
            "user message added to history"
        );

        let mut final_text = String::new();
        let mut tool_call_count = 0;

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
                prelude: &self.prelude,
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
            let mut tool_calls: Vec<ChatCompletionMessageToolCall> = Vec::new();
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
                            let index = chunk.index as usize;
                            while tool_calls.len() <= index {
                                tool_calls.push(ChatCompletionMessageToolCall::default());
                            }
                            let tool_call = &mut tool_calls[index];
                            if let Some(id) = chunk.id {
                                tool_call.id = id;
                            }
                            if let Some(function) = chunk.function {
                                if let Some(name) = function.name {
                                    tool_call.function.name = name;
                                }
                                if let Some(arguments) = function.arguments {
                                    tool_call.function.arguments.push_str(&arguments);
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

            validate_chat_finish_reasons(&finish_reasons, !tool_calls.is_empty())?;

            if tool_calls.is_empty() {
                if final_text.is_empty() {
                    final_text = "No response content".to_string();
                }

                self.history.push(HistoryItem::assistant(turn_text.clone()));

                info!(
                    output_chars = final_text.chars().count(),
                    history_len = self.history.len(),
                    "final chat completion answer completed"
                );

                return Ok(final_text);
            }

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
                let evidence = self.remember_tool_evidence(&call, &output)?;
                on_event(AgentEvent::EvidenceRecorded(evidence)).await?;

                let output = serde_json::to_string(&output)?;
                self.history.push(HistoryItem::ToolOutput {
                    call_id: call.call_id,
                    output_json: output,
                });
            }
        }

        Err(anyhow!(
            "stopped: too many agent iterations (max {})",
            self.max_iterations
        ))
    }

    async fn execute_tool_call<E, A, Efut, Afut>(
        &self,
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
                let permission_decision = self.permission_policy.check(&call.name, &args);
                let should_execute = match permission_decision {
                    PermissionDecision::Allow => true,
                    PermissionDecision::Ask => {
                        approve(PermissionRequest {
                            call_id: Some(call.call_id.clone()),
                            tool: call.name.clone(),
                            args: args.clone(),
                            class: classify_tool(&call.name),
                            summary: format_tool_call(&call.name, &args),
                            preview: None,
                        })
                        .await?
                    }
                    PermissionDecision::Deny => false,
                };

                if should_execute {
                    on_event(AgentEvent::ToolCallStarted {
                        call_id: call.call_id.clone(),
                        name: call.name.clone(),
                        args: args.clone(),
                    })
                    .await?;

                    let output = self.tools.call(&call.name, args).await;

                    on_event(AgentEvent::ToolCallFinished {
                        call_id: call.call_id.clone(),
                        name: call.name.clone(),
                        ok: output.ok,
                        output: output.clone(),
                    })
                    .await?;

                    output
                } else if matches!(permission_decision, PermissionDecision::Deny) {
                    ToolResult::err(&call.name, "permission denied by current mode")
                } else {
                    ToolResult::err(&call.name, "user denied permission")
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_openai::config::OpenAIConfig;

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
            },
        );
        catalog.insert(
            "m2".to_string(),
            ModelRequestMetadata {
                context_window: Some(128_000),
                max_output_tokens: Some(256),
                supports_tools: true,
                supports_reasoning: false,
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
