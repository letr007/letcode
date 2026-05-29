use anyhow::{Result, anyhow};
use async_openai::Client;
use async_openai::config::Config;
use async_openai::types::responses::{
    CreateResponse, EasyInputContent, EasyInputMessage, FunctionCallOutput,
    FunctionCallOutputItemParam, InputItem, Item, MessageType, OutputItem, Response,
    ResponseStreamEvent, Role,
};
use futures_util::StreamExt;
use serde_json::Value;
use std::collections::HashSet;
use std::future::Future;
use tracing::{debug, error, info, trace, warn};

use crate::permission::{
    PermissionDecision, PermissionMode, PermissionPolicy, PermissionRequest, classify_tool,
};
use crate::tool::{ToolHandler, ToolRegistry, ToolResult};
use crate::tool_format::format_tool_call;

#[derive(Debug, Clone)]
pub enum AgentEvent {
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

pub struct Agent<C: Config> {
    pub client: Client<C>,
    model: String,
    history: Vec<InputItem>,
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
            history: vec![],
            tools: ToolRegistry::default_tools(),
            permission_policy: PermissionPolicy::default(),
            max_iterations: max_iterations,
            max_tool_calls,
        }
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

    pub fn restore_transcript_messages(&mut self, messages: Vec<ConversationMessage>) {
        self.history = messages
            .into_iter()
            .map(|message| match message.role {
                ConversationRole::User => user_message(message.content),
                ConversationRole::Assistant => assistant_message(message.content),
            })
            .collect();
    }

    #[allow(dead_code)]
    pub fn register_tool<T>(&mut self, tool: T)
    where
        T: ToolHandler + 'static,
    {
        self.tools.register(tool);
    }

    #[allow(dead_code)]
    pub async fn run(&mut self, user_input: &str) -> Result<String> {
        self.run_stream(user_input, |_| Ok(()), |_| Ok(()), |_| Ok(true))
            .await
    }

    pub async fn run_stream_async<F, E, A, Dfut, Efut, Afut>(
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
        self.history.push(user_message(user_input));
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

            let request = CreateResponse {
                model: Some(self.model.clone()),
                input: self.history.clone().into(),
                previous_response_id: None,
                tools: Some(self.tools.definitions()),
                parallel_tool_calls: Some(false),
                ..Default::default()
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
                    OutputItem::FunctionCall(call) => Some(call.clone()),
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

            self.history
                .extend(response.output.iter().cloned().map(InputItem::from));

            debug!(
                iteration,
                tool_calls = tool_calls.len(),
                tool_call_count,
                history_len = self.history.len(),
                "response output appended to history"
            );

            if tool_calls.is_empty() {
                if text.is_empty() {
                    text = response
                        .output_text()
                        .unwrap_or_else(|| "No response content".to_string());
                }

                info!(
                    output_chars = text.chars().count(),
                    history_len = self.history.len(),
                    "final answer completed"
                );

                return Ok(text);
            }

            for call in tool_calls {
                info!(
                    tool_name = %call.name,
                    call_id = %call.call_id,
                    "tool call requested"
                );
                debug!(
                    tool_name = %call.name,
                    call_id = %call.call_id,
                    arguments = %call.arguments,
                    "tool call arguments"
                );

                let output = match serde_json::from_str::<Value>(&call.arguments) {
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
                            raw_arguments = %call.arguments,
                            "invalid tool call JSON arguments"
                        );
                        ToolResult::err(
                            &call.name,
                            format!("invalid JSON arguments: {err}; raw: {}", call.arguments),
                        )
                    }
                };

                debug!(
                    tool_name = %call.name,
                    call_id = %call.call_id,
                    output = ?output,
                    "tool call completed"
                );

                let output = serde_json::to_string(&output)?;

                self.history.push(InputItem::Item(Item::FunctionCallOutput(
                    FunctionCallOutputItemParam {
                        call_id: call.call_id,
                        output: FunctionCallOutput::Text(output),
                        id: None,
                        status: None,
                    },
                )));

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

fn user_message(content: impl Into<String>) -> InputItem {
    InputItem::EasyMessage(EasyInputMessage {
        r#type: MessageType::Message,
        role: Role::User,
        content: EasyInputContent::Text(content.into()),
        phase: None,
    })
}

fn assistant_message(content: impl Into<String>) -> InputItem {
    InputItem::EasyMessage(EasyInputMessage {
        r#type: MessageType::Message,
        role: Role::Assistant,
        content: EasyInputContent::Text(content.into()),
        phase: None,
    })
}
