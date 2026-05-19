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
use tracing::{debug, error, info, trace, warn};

use crate::permission::{
    PermissionDecision, PermissionMode, PermissionPolicy, PermissionRequest, classify_tool,
};
use crate::tool::{ToolHandler, ToolRegistry, ToolResult};

#[derive(Debug, Clone)]
pub enum AgentEvent {
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
        self.history.push(user_message(user_input));
        debug!(
            user_input_len = user_input.len(),
            history_len = self.history.len(),
            "user message added to history"
        );

        let mut text = String::new();
        let mut tool_call_count = 0;

        for iteration in 0..self.max_iterations {
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
                        on_delta(&event.delta)?;
                        text.push_str(&event.delta);
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
                            PermissionDecision::Ask => approve(PermissionRequest {
                                tool: call.name.clone(),
                                args: args.clone(),
                                class: classify_tool(&call.name),
                                summary: format_tool_summary(&call.name, &args),
                                preview: None,
                            })?,
                            PermissionDecision::Deny => false,
                        };

                        if should_execute {
                            on_event(AgentEvent::ToolCallStarted {
                                call_id: call.call_id.clone(),
                                name: call.name.clone(),
                                args: args.clone(),
                            })?;

                            let output = self.tools.call(&call.name, args).await;

                            on_event(AgentEvent::ToolCallFinished {
                                call_id: call.call_id.clone(),
                                name: call.name.clone(),
                                ok: output.ok,
                                output: output.clone(),
                            })?;

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
}

fn format_tool_summary(name: &str, args: &Value) -> String {
    match name {
        "list_dir" | "read_file" | "write_file" | "append_file" | "mkdir" => args
            .get("path")
            .and_then(Value::as_str)
            .map(|path| format!("{name} {path}"))
            .unwrap_or_else(|| format!("{name} {args}")),
        "rg" => {
            let pattern = args.get("pattern").and_then(Value::as_str).unwrap_or("");
            let path = args.get("path").and_then(Value::as_str).unwrap_or(".");
            format!("rg {:?} in {}", truncate_summary(pattern, 60), path)
        }
        "git_status" => "git status".to_string(),
        "git_diff" => {
            let staged = args.get("staged").and_then(Value::as_bool).unwrap_or(false);
            let path = args.get("path").and_then(Value::as_str).unwrap_or("");
            let staged_flag = if staged { " --cached" } else { "" };
            format!("git diff{} {}", staged_flag, path)
                .trim()
                .to_string()
        }
        "git_log" => {
            let max_count = args.get("max_count").and_then(Value::as_u64).unwrap_or(10);
            format!("git log -{}", max_count)
        }
        "apply_patch" => {
            let edits = args
                .get("edits")
                .and_then(Value::as_array)
                .map(Vec::len)
                .unwrap_or(0);
            format!(
                "apply_patch {} edit{}",
                edits,
                if edits == 1 { "" } else { "s" }
            )
        }
        "ast_search" => {
            let pattern = args.get("pattern").and_then(Value::as_str).unwrap_or("");
            let path = args.get("path").and_then(Value::as_str).unwrap_or(".");
            format!("ast_search {:?} in {}", truncate_summary(pattern, 60), path)
        }
        "ast_replace_preview" => {
            let pattern = args.get("pattern").and_then(Value::as_str).unwrap_or("");
            let path = args.get("path").and_then(Value::as_str).unwrap_or(".");
            format!(
                "ast_replace_preview {:?} in {}",
                truncate_summary(pattern, 60),
                path
            )
        }
        "run_command" => {
            let command = args.get("command").and_then(Value::as_str).unwrap_or("");
            format!("run_command {}", truncate_summary(command, 120))
        }
        "echo" => args
            .get("text")
            .and_then(Value::as_str)
            .map(|text| format!("echo {:?}", truncate_summary(text, 60)))
            .unwrap_or_else(|| format!("echo {args}")),
        _ => format!("{name} {args}"),
    }
}

fn truncate_summary(text: &str, max_chars: usize) -> String {
    let mut truncated = text.chars().take(max_chars).collect::<String>();
    if text.chars().count() > max_chars {
        truncated.push('…');
    }
    truncated
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
