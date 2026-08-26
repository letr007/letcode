use async_openai::types::chat::{
    ChatCompletionMessageToolCall, ChatCompletionMessageToolCalls,
    ChatCompletionRequestAssistantMessage, ChatCompletionRequestAssistantMessageContent,
    ChatCompletionRequestDeveloperMessage, ChatCompletionRequestDeveloperMessageContent,
    ChatCompletionRequestMessage, ChatCompletionRequestMessageContentPartImage,
    ChatCompletionRequestMessageContentPartText, ChatCompletionRequestSystemMessage,
    ChatCompletionRequestSystemMessageContent, ChatCompletionRequestToolMessage,
    ChatCompletionRequestToolMessageContent, ChatCompletionRequestUserMessage,
    ChatCompletionRequestUserMessageContent, ChatCompletionRequestUserMessageContentPart,
    ChatCompletionStreamOptions, ChatCompletionTool, ChatCompletionTools,
    CreateChatCompletionRequest, FunctionCall, FunctionObject, ImageUrl,
    ServiceTier as ChatServiceTier, Verbosity as ChatVerbosity,
};
use async_openai::types::responses::{
    CreateResponse, EasyInputContent, EasyInputMessage, FunctionCallOutput,
    FunctionCallOutputItemParam, FunctionTool, FunctionToolCall, ImageDetail, InputContent,
    InputImageContent, InputItem, InputMessage, InputRole, InputTextContent, Item, MessageItem,
    MessageType, OutputStatus, PromptCacheRetention as OpenAiPromptCacheRetention, Reasoning,
    ReasoningEffort as OpenAiReasoningEffort, ReasoningItem, ReasoningItemContent,
    ReasoningSummary as ResponseReasoningSummary, ReasoningTextContent, ResponseTextParam, Role,
    ServiceTier as ResponseServiceTier, TextResponseFormatConfiguration, Tool,
    Verbosity as ResponseVerbosity,
};

use serde_json::Value;

use crate::config::{ApiProtocol, PromptCacheRetention};
use crate::user_content::{UserImageAttachment, UserMessageContent, UserMessagePart};

use super::prompt_plan::{PromptPlan, PromptSegment, PromptSegmentContent, PromptSegmentRole};
use super::{
    ModelReasoningEffort, ModelReasoningSummary, ModelRequestMetadata, ModelTextVerbosity,
    PromptMessage, PromptRole, ProviderRequestStrategy, ToolSpec, cache_request_fields,
};

pub(super) fn response_instructions(segments: &[PromptSegment]) -> Option<String> {
    let instructions = segments
        .iter()
        .filter(|segment| segment.role == PromptSegmentRole::System)
        .filter_map(|segment| match &segment.content {
            PromptSegmentContent::Text { text } if !text.is_empty() => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();
    (!instructions.is_empty()).then(|| instructions.join("\n\n"))
}

pub(super) fn build_responses_request(
    strategy: ProviderRequestStrategy,
    model_id: &str,
    model: ModelRequestMetadata,
    prompt_plan: &PromptPlan,
    tools: &[ToolSpec],
) -> CreateResponse {
    let input = prompt_plan
        .segments
        .iter()
        .flat_map(|segment| prompt_segment_to_response_inputs(segment, strategy))
        .collect::<Vec<_>>();
    let instructions = response_instructions(&prompt_plan.segments);
    let cache = cache_request_fields(
        strategy,
        ApiProtocol::Responses,
        model_id,
        &model.prompt_cache,
        prompt_plan,
        tools,
        model.supports_tools,
        model.parallel_tool_calls,
    );
    let tools = if model.supports_tools {
        Some(tools.iter().map(tool_to_response_tool).collect())
    } else {
        None
    };
    let parallel_tool_calls = model.supports_tools.then_some(model.parallel_tool_calls);

    CreateResponse {
        model: Some(model_id.to_string()),
        input: input.into(),
        instructions,
        max_output_tokens: model.max_output_tokens.and_then(u64_to_u32),
        previous_response_id: None,
        reasoning: response_reasoning(model.clone()),
        temperature: model.temperature,
        text: response_text(model.clone()),
        tools,
        parallel_tool_calls,
        stream: Some(true),
        top_p: model.top_p,
        prompt_cache_key: cache.key,
        prompt_cache_retention: cache.retention.map(openai_cache_retention),
        service_tier: model.fast_mode.then_some(ResponseServiceTier::Priority),
        ..Default::default()
    }
}

pub(super) fn build_anthropic_request(
    model_id: &str,
    model: ModelRequestMetadata,
    prompt_plan: &PromptPlan,
    tools: &[ToolSpec],
) -> Value {
    let mut system = Vec::new();
    let mut messages = Vec::new();
    let mut segment_message_index = Vec::with_capacity(prompt_plan.segments.len());

    let push_message = |role: &str, mut content: Vec<Value>, messages: &mut Vec<Value>| {
        if content.is_empty() {
            content.push(anthropic_text_block(""));
        }
        messages.push(serde_json::json!({ "role": role, "content": content }));
        messages.len() - 1
    };

    for segment in &prompt_plan.segments {
        match (&segment.role, &segment.content) {
            (
                PromptSegmentRole::System | PromptSegmentRole::Developer,
                PromptSegmentContent::Text { text },
            ) if !text.is_empty() => {
                system.push(anthropic_text_block(text));
                segment_message_index.push(None);
            }
            (_, PromptSegmentContent::UserContent { content }) => {
                let index = push_message(
                    "user",
                    content
                        .parts()
                        .into_iter()
                        .map(anthropic_user_part)
                        .collect(),
                    &mut messages,
                );
                segment_message_index.push(Some(index));
            }
            (PromptSegmentRole::User, PromptSegmentContent::Text { text }) => {
                let index = push_message("user", vec![anthropic_text_block(text)], &mut messages);
                segment_message_index.push(Some(index));
            }
            (
                PromptSegmentRole::Assistant,
                PromptSegmentContent::AssistantToolCalls {
                    text,
                    reasoning_wire,
                    calls,
                    ..
                },
            ) => {
                let mut content = Vec::new();
                if let Some(wire) = reasoning_wire
                    .as_deref()
                    .and_then(|wire| serde_json::from_str::<Vec<Value>>(wire).ok())
                {
                    content.extend(wire.into_iter().filter_map(anthropic_thinking_block));
                }
                if let Some(text) = text.clone().filter(|text| !text.is_empty()) {
                    content.push(anthropic_text_block(&text));
                }
                content.extend(calls.iter().map(|call| {
                    serde_json::json!({
                        "type": "tool_use",
                        "id": call.call_id,
                        "name": call.name,
                        "input": serde_json::from_str::<Value>(&call.arguments_json)
                            .unwrap_or_else(|_| Value::Object(Default::default())),
                    })
                }));
                let index = push_message("assistant", content, &mut messages);
                segment_message_index.push(Some(index));
            }
            (
                PromptSegmentRole::Tool,
                PromptSegmentContent::ToolOutput {
                    call_id,
                    output_json,
                    images,
                },
            ) => {
                let block = serde_json::json!({
                    "type": "tool_result",
                    "tool_use_id": call_id,
                    "content": anthropic_tool_result_content(output_json, images),
                    "is_error": false,
                });
                if let Some(last) = messages.last_mut()
                    && last["role"] == "user"
                    && last["content"].as_array().is_some_and(|blocks| {
                        blocks.iter().all(|block| block["type"] == "tool_result")
                    })
                {
                    last["content"]
                        .as_array_mut()
                        .expect("checked array")
                        .push(block);
                    segment_message_index.push(Some(messages.len() - 1));
                } else {
                    let index = push_message("user", vec![block], &mut messages);
                    segment_message_index.push(Some(index));
                }
            }
            (PromptSegmentRole::Assistant, PromptSegmentContent::Text { text }) => {
                let index =
                    push_message("assistant", vec![anthropic_text_block(text)], &mut messages);
                segment_message_index.push(Some(index));
            }
            _ => {
                let index = push_message(
                    anthropic_role(segment.role),
                    vec![anthropic_text_block(&segment.text)],
                    &mut messages,
                );
                segment_message_index.push(Some(index));
            }
        }
    }

    let mut request = serde_json::json!({
        "model": model_id,
        "max_tokens": model.max_output_tokens.unwrap_or_else(|| model.output_reserve_tokens()),
        "messages": messages,
        "stream": true,
    });
    if !system.is_empty() {
        request["system"] = Value::Array(system.clone());
    }
    if model.supports_tools && !tools.is_empty() {
        request["tools"] = Value::Array(
            tools
                .iter()
                .map(|tool| {
                    serde_json::json!({
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.parameters,
                    })
                })
                .collect(),
        );
    }

    apply_anthropic_thinking(&mut request, &model);
    if model.cache_control {
        if let Some(last) = system.last_mut() {
            last["cache_control"] = serde_json::json!({ "type": "ephemeral" });
            if let Some(system_value) = request.get_mut("system") {
                *system_value = Value::Array(system.clone());
            }
        }
        if let Some(tools) = request.get_mut("tools").and_then(Value::as_array_mut)
            && let Some(last) = tools.last_mut()
        {
            last["cache_control"] = serde_json::json!({ "type": "ephemeral" });
        }
        if let Some(segment_index) = prompt_plan.stable_prefix_end
            && let Some(message_index) = segment_message_index.get(segment_index).copied().flatten()
            && let Some(message) = messages.get_mut(message_index)
            && let Some(blocks) = message.get_mut("content").and_then(Value::as_array_mut)
            && let Some(last) = blocks.last_mut()
        {
            last["cache_control"] = serde_json::json!({ "type": "ephemeral" });
            request["messages"] = Value::Array(messages.clone());
        }
    }

    request
}

fn apply_anthropic_thinking(request: &mut Value, model: &ModelRequestMetadata) {
    match model.anthropic_thinking.mode {
        crate::request_builder::AnthropicThinkingMode::Disabled => {}
        crate::request_builder::AnthropicThinkingMode::Adaptive => {
            request["thinking"] = serde_json::json!({ "type": "adaptive" });
            let effort = model
                .reasoning_effort
                .as_ref()
                .unwrap_or(&ModelReasoningEffort::Low);
            request["output_config"] = serde_json::json!({ "effort": effort.as_str() });
        }
        crate::request_builder::AnthropicThinkingMode::Budget => {
            let budget = model.anthropic_thinking.budget_tokens.unwrap_or(1024);
            request["thinking"] = serde_json::json!({
                "type": "enabled",
                "budget_tokens": budget,
            });
        }
    }
}

fn anthropic_thinking_block(block: Value) -> Option<Value> {
    if block.get("type")?.as_str()? != "thinking" {
        return None;
    }
    let thinking = block.get("thinking")?.as_str()?.to_string();
    let signature = block
        .get("signature")
        .and_then(Value::as_str)
        .filter(|signature| !signature.is_empty())
        .map(ToString::to_string);
    let mut block = serde_json::json!({ "type": "thinking", "thinking": thinking });
    if let Some(signature) = signature {
        block["signature"] = Value::String(signature);
    }
    Some(block)
}

fn anthropic_role(role: PromptSegmentRole) -> &'static str {
    match role {
        PromptSegmentRole::System | PromptSegmentRole::Developer | PromptSegmentRole::Tool => {
            "user"
        }
        PromptSegmentRole::User => "user",
        PromptSegmentRole::Assistant => "assistant",
    }
}

fn anthropic_text_block(text: impl Into<String>) -> Value {
    serde_json::json!({ "type": "text", "text": text.into() })
}

fn anthropic_image_block(attachment: &crate::user_content::UserImageAttachment) -> Value {
    let data_url = attachment.data_url.trim_start_matches("data:");
    let (media_type, data) = data_url
        .split_once(";base64,")
        .unwrap_or(("image/png", data_url));
    serde_json::json!({
        "type": "image",
        "source": { "type": "base64", "media_type": media_type, "data": data },
    })
}

fn anthropic_user_part(part: crate::user_content::UserMessagePart) -> Value {
    match part {
        crate::user_content::UserMessagePart::Text { text } => anthropic_text_block(text),
        crate::user_content::UserMessagePart::Image { attachment } => {
            anthropic_image_block(&attachment)
        }
    }
}

fn anthropic_tool_result_content(
    output_json: &str,
    images: &[crate::user_content::UserImageAttachment],
) -> Value {
    if images.is_empty() {
        return Value::String(output_json.to_string());
    }
    let mut content = vec![anthropic_text_block(output_json)];
    content.extend(images.iter().map(anthropic_image_block));
    Value::Array(content)
}

fn openai_cache_retention(value: PromptCacheRetention) -> OpenAiPromptCacheRetention {
    match value {
        PromptCacheRetention::InMemory => OpenAiPromptCacheRetention::InMemory,
        PromptCacheRetention::TwentyFourHours => OpenAiPromptCacheRetention::Hours24,
    }
}

fn response_reasoning(model: ModelRequestMetadata) -> Option<Reasoning> {
    if !model.supports_reasoning {
        return None;
    }
    let effort = model
        .reasoning_effort
        .filter(|effort| !effort.requires_compatible_request())
        .map(openai_reasoning_effort);
    let summary = model.reasoning_summary.map(response_reasoning_summary);
    (effort.is_some() || summary.is_some()).then_some(Reasoning { effort, summary })
}

fn response_text(model: ModelRequestMetadata) -> Option<ResponseTextParam> {
    model.text_verbosity.map(|verbosity| ResponseTextParam {
        format: TextResponseFormatConfiguration::Text,
        verbosity: Some(response_verbosity(verbosity)),
    })
}

pub(super) fn prompt_segment_to_response_inputs(
    segment: &PromptSegment,
    strategy: ProviderRequestStrategy,
) -> Vec<InputItem> {
    match (&segment.role, &segment.content) {
        (PromptSegmentRole::System, _) => Vec::new(),
        (PromptSegmentRole::Developer, PromptSegmentContent::Text { text }) => {
            vec![response_text_message(Role::Developer, text.clone())]
        }
        (PromptSegmentRole::User, PromptSegmentContent::UserContent { content }) => {
            vec![response_user_message(content.clone())]
        }
        (PromptSegmentRole::User, PromptSegmentContent::Text { text }) => {
            vec![response_text_message(Role::User, text.clone())]
        }
        (PromptSegmentRole::Assistant, PromptSegmentContent::Text { text }) => {
            vec![response_text_message(Role::Assistant, text.clone())]
        }
        (
            PromptSegmentRole::Assistant,
            PromptSegmentContent::AssistantToolCalls {
                text,
                reasoning_content,
                calls,
                ..
            },
        ) => {
            let mut input = Vec::new();
            if strategy.is_deepseek_v4() {
                input.push(InputItem::Item(Item::Reasoning(ReasoningItem {
                    id: None,
                    summary: Vec::new(),
                    content: Some(vec![ReasoningItemContent::ReasoningText(
                        ReasoningTextContent {
                            text: reasoning_content.clone().unwrap_or_default(),
                        },
                    )]),
                    encrypted_content: None,
                    status: None,
                })));
            }
            if let Some(text) = text.clone().filter(|text| !text.is_empty()) {
                input.push(response_text_message(Role::Assistant, text));
            }
            input.extend(calls.iter().cloned().map(|call| {
                InputItem::Item(Item::FunctionCall(FunctionToolCall {
                    arguments: call.arguments_json,
                    call_id: call.call_id,
                    namespace: None,
                    name: call.name,
                    id: None,
                    status: None::<OutputStatus>,
                }))
            }));
            input
        }
        (
            PromptSegmentRole::Tool,
            PromptSegmentContent::ToolOutput {
                call_id,
                output_json,
                images,
            },
        ) => {
            let output = if images.is_empty() {
                FunctionCallOutput::Text(output_json.clone())
            } else {
                let mut content = vec![InputContent::InputText(InputTextContent {
                    text: output_json.clone(),
                })];
                content.extend(images.iter().cloned().map(response_image_part));
                FunctionCallOutput::Content(content)
            };
            vec![InputItem::Item(Item::FunctionCallOutput(
                FunctionCallOutputItemParam {
                    call_id: call_id.clone(),
                    output,
                    id: None,
                    status: None,
                },
            ))]
        }
        _ => vec![response_text_message(
            role_to_response_role(segment.role),
            segment.text.clone(),
        )],
    }
}

fn role_to_response_role(role: PromptSegmentRole) -> Role {
    match role {
        PromptSegmentRole::System => Role::System,
        PromptSegmentRole::Developer => Role::Developer,
        PromptSegmentRole::User => Role::User,
        PromptSegmentRole::Assistant => Role::Assistant,
        PromptSegmentRole::Tool => Role::Developer,
    }
}

pub(super) fn prompt_segment_to_chat_message(
    segment: &PromptSegment,
) -> ChatCompletionRequestMessage {
    match (&segment.role, &segment.content) {
        (PromptSegmentRole::System, PromptSegmentContent::Text { text }) => {
            prelude_to_chat_message(PromptMessage::system(text.clone()))
        }
        (PromptSegmentRole::Developer, PromptSegmentContent::Text { text }) => {
            prelude_to_chat_message(PromptMessage::developer(text.clone()))
        }
        (PromptSegmentRole::User, PromptSegmentContent::UserContent { content }) => {
            ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage {
                content: user_content_to_chat_content(content.clone()),
                name: None,
            })
        }
        (PromptSegmentRole::User, PromptSegmentContent::Text { text }) => {
            ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage {
                content: ChatCompletionRequestUserMessageContent::Text(text.clone()),
                name: None,
            })
        }
        (PromptSegmentRole::Assistant, PromptSegmentContent::Text { text }) => {
            chat_assistant_text(text.clone())
        }
        (
            PromptSegmentRole::Assistant,
            PromptSegmentContent::AssistantToolCalls { text, calls, .. },
        ) => ChatCompletionRequestMessage::Assistant(ChatCompletionRequestAssistantMessage {
            content: text
                .clone()
                .map(ChatCompletionRequestAssistantMessageContent::Text),
            refusal: None,
            name: None,
            audio: None,
            tool_calls: Some(
                calls
                    .iter()
                    .cloned()
                    .map(|call| {
                        ChatCompletionMessageToolCalls::Function(ChatCompletionMessageToolCall {
                            id: call.call_id,
                            function: FunctionCall {
                                name: call.name,
                                arguments: call.arguments_json,
                            },
                        })
                    })
                    .collect(),
            ),
            ..Default::default()
        }),
        (
            PromptSegmentRole::Tool,
            PromptSegmentContent::ToolOutput {
                call_id,
                output_json,
                ..
            },
        ) => ChatCompletionRequestMessage::Tool(ChatCompletionRequestToolMessage {
            content: ChatCompletionRequestToolMessageContent::Text(output_json.clone()),
            tool_call_id: call_id.clone(),
        }),
        (PromptSegmentRole::Assistant, _) => chat_assistant_text(segment.text.clone()),
        (PromptSegmentRole::System, _) => {
            prelude_to_chat_message(PromptMessage::system(segment.text.clone()))
        }
        (PromptSegmentRole::Developer, _) | (PromptSegmentRole::Tool, _) => {
            prelude_to_chat_message(PromptMessage::developer(segment.text.clone()))
        }
        (PromptSegmentRole::User, _) => {
            ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage {
                content: ChatCompletionRequestUserMessageContent::Text(segment.text.clone()),
                name: None,
            })
        }
    }
}

fn response_text_message(role: Role, text: String) -> InputItem {
    InputItem::EasyMessage(EasyInputMessage {
        r#type: MessageType::Message,
        role,
        content: EasyInputContent::Text(text),
        phase: None,
    })
}

fn response_user_message(content: UserMessageContent) -> InputItem {
    InputItem::Item(Item::Message(MessageItem::Input(InputMessage {
        role: InputRole::User,
        content: user_content_to_response_content(content),
        status: None,
    })))
}

pub(super) fn tool_to_response_tool(tool: &ToolSpec) -> Tool {
    Tool::Function(FunctionTool {
        name: tool.name.clone(),
        description: Some(tool.description.clone()),
        parameters: Some(tool.parameters.clone()),
        strict: Some(tool.strict),
        defer_loading: None,
    })
}

pub(super) fn apply_chat_reasoning_content(
    request: &mut serde_json::Value,
    prompt_plan: &PromptPlan,
) {
    let messages = request["messages"]
        .as_array_mut()
        .expect("chat completion request has messages");
    for (message, segment) in messages.iter_mut().zip(&prompt_plan.segments) {
        if let PromptSegmentContent::AssistantToolCalls {
            reasoning_content, ..
        } = &segment.content
        {
            message["reasoning_content"] =
                serde_json::Value::String(reasoning_content.clone().unwrap_or_default());
        }
    }
}

fn deepseek_reasoning_effort(value: &ModelReasoningEffort) -> Option<String> {
    match value {
        ModelReasoningEffort::None => None,
        ModelReasoningEffort::Minimal | ModelReasoningEffort::Low => Some("low".into()),
        ModelReasoningEffort::Medium | ModelReasoningEffort::High | ModelReasoningEffort::Xhigh => {
            Some("high".into())
        }
        ModelReasoningEffort::Max => Some("max".into()),
        ModelReasoningEffort::Custom(value) => Some(value.clone()),
    }
}

fn normalize_deepseek_chat_message_roles(request: &mut serde_json::Value) {
    if let Some(messages) = request["messages"].as_array_mut() {
        for message in messages {
            if message["role"] == "developer" {
                message["role"] = serde_json::Value::String("system".into());
            }
        }
    }
}

pub(super) fn apply_deepseek_chat_compat(
    request: &mut serde_json::Value,
    model: &ModelRequestMetadata,
    prompt_plan: &PromptPlan,
) {
    normalize_deepseek_chat_message_roles(request);

    if let Some(max_tokens) = request.get("max_completion_tokens").cloned() {
        request["max_tokens"] = max_tokens;
        request
            .as_object_mut()
            .expect("chat request is an object")
            .remove("max_completion_tokens");
    }

    request
        .as_object_mut()
        .expect("chat request is an object")
        .remove("verbosity");
    request
        .as_object_mut()
        .expect("chat request is an object")
        .remove("prompt_cache_key");
    request
        .as_object_mut()
        .expect("chat request is an object")
        .remove("service_tier");

    if model.supports_reasoning {
        match model.reasoning_effort.as_ref() {
            Some(ModelReasoningEffort::None) => {
                request["thinking"] = serde_json::json!({"type": "disabled"});
                request
                    .as_object_mut()
                    .expect("chat request is an object")
                    .remove("reasoning_effort");
            }
            Some(effort) => {
                request["thinking"] = serde_json::json!({"type": "enabled"});
                request["reasoning_effort"] = serde_json::Value::String(
                    deepseek_reasoning_effort(effort).unwrap_or_else(|| "high".into()),
                );
            }
            None => {
                request["thinking"] = serde_json::json!({"type": "enabled"});
            }
        }
    }

    apply_chat_reasoning_content(request, prompt_plan);
}

pub(super) fn build_completions_request(
    strategy: ProviderRequestStrategy,
    model_id: &str,
    model: ModelRequestMetadata,
    prompt_plan: &PromptPlan,
    tools: &[ToolSpec],
) -> anyhow::Result<CreateChatCompletionRequest> {
    if prompt_plan.segments.iter().any(|segment| {
        matches!(
            &segment.content,
            PromptSegmentContent::ToolOutput { images, .. } if !images.is_empty()
        )
    }) {
        anyhow::bail!(
            "Chat Completions does not support image content in tool outputs; use a Responses API provider route"
        );
    }
    let messages = prompt_plan
        .segments
        .iter()
        .map(prompt_segment_to_chat_message)
        .collect::<Vec<_>>();
    let cache = cache_request_fields(
        strategy,
        ApiProtocol::Completions,
        model_id,
        &model.prompt_cache,
        prompt_plan,
        tools,
        model.supports_tools,
        model.parallel_tool_calls,
    );
    let tools = if model.supports_tools {
        Some(tools.iter().map(tool_to_chat_tool).collect())
    } else {
        None
    };
    let parallel_tool_calls = model.supports_tools.then_some(model.parallel_tool_calls);

    Ok(CreateChatCompletionRequest {
        model: model_id.to_string(),
        messages,
        max_completion_tokens: model.max_output_tokens.and_then(u64_to_u32),
        reasoning_effort: model
            .supports_reasoning
            .then_some(model.reasoning_effort)
            .flatten()
            .filter(|effort| !effort.requires_compatible_request())
            .map(openai_reasoning_effort),
        stream: Some(true),
        stream_options: Some(ChatCompletionStreamOptions {
            include_usage: Some(true),
            include_obfuscation: None,
        }),
        n: Some(1),
        temperature: model.temperature,
        top_p: model.top_p,
        tools,
        parallel_tool_calls,
        verbosity: model.text_verbosity.map(chat_verbosity),
        prompt_cache_key: cache.key,
        service_tier: model.fast_mode.then_some(ChatServiceTier::Priority),
        ..Default::default()
    })
}

fn u64_to_u32(value: u64) -> Option<u32> {
    u32::try_from(value).ok()
}

fn openai_reasoning_effort(effort: ModelReasoningEffort) -> OpenAiReasoningEffort {
    match effort {
        ModelReasoningEffort::None => OpenAiReasoningEffort::None,
        ModelReasoningEffort::Minimal => OpenAiReasoningEffort::Minimal,
        ModelReasoningEffort::Low => OpenAiReasoningEffort::Low,
        ModelReasoningEffort::Medium => OpenAiReasoningEffort::Medium,
        ModelReasoningEffort::High => OpenAiReasoningEffort::High,
        ModelReasoningEffort::Xhigh => OpenAiReasoningEffort::Xhigh,
        ModelReasoningEffort::Max | ModelReasoningEffort::Custom(_) => {
            unreachable!("compatible efforts are serialized through a compatible request")
        }
    }
}

fn response_reasoning_summary(summary: ModelReasoningSummary) -> ResponseReasoningSummary {
    match summary {
        ModelReasoningSummary::Auto => ResponseReasoningSummary::Auto,
        ModelReasoningSummary::Concise => ResponseReasoningSummary::Concise,
        ModelReasoningSummary::Detailed => ResponseReasoningSummary::Detailed,
    }
}

fn response_verbosity(verbosity: ModelTextVerbosity) -> ResponseVerbosity {
    match verbosity {
        ModelTextVerbosity::Low => ResponseVerbosity::Low,
        ModelTextVerbosity::Medium => ResponseVerbosity::Medium,
        ModelTextVerbosity::High => ResponseVerbosity::High,
    }
}

fn chat_verbosity(verbosity: ModelTextVerbosity) -> ChatVerbosity {
    match verbosity {
        ModelTextVerbosity::Low => ChatVerbosity::Low,
        ModelTextVerbosity::Medium => ChatVerbosity::Medium,
        ModelTextVerbosity::High => ChatVerbosity::High,
    }
}

fn prelude_to_chat_message(message: PromptMessage) -> ChatCompletionRequestMessage {
    match message.role {
        PromptRole::System => {
            ChatCompletionRequestMessage::System(ChatCompletionRequestSystemMessage {
                content: ChatCompletionRequestSystemMessageContent::Text(message.text),
                name: None,
            })
        }
        PromptRole::Developer => {
            ChatCompletionRequestMessage::Developer(ChatCompletionRequestDeveloperMessage {
                content: ChatCompletionRequestDeveloperMessageContent::Text(message.text),
                name: None,
            })
        }
    }
}

fn chat_assistant_text(text: String) -> ChatCompletionRequestMessage {
    ChatCompletionRequestMessage::Assistant(ChatCompletionRequestAssistantMessage {
        content: Some(ChatCompletionRequestAssistantMessageContent::Text(text)),
        ..Default::default()
    })
}

pub(super) fn tool_to_chat_tool(tool: &ToolSpec) -> ChatCompletionTools {
    ChatCompletionTools::Function(ChatCompletionTool {
        function: FunctionObject {
            name: tool.name.clone(),
            description: Some(tool.description.clone()),
            parameters: Some(tool.parameters.clone()),
            strict: Some(tool.strict),
        },
    })
}

fn user_content_to_response_content(content: UserMessageContent) -> Vec<InputContent> {
    content
        .parts()
        .into_iter()
        .map(|part| match part {
            UserMessagePart::Text { text } => InputContent::InputText(InputTextContent { text }),
            UserMessagePart::Image { attachment } => response_image_part(attachment),
        })
        .collect()
}

fn response_image_part(attachment: UserImageAttachment) -> InputContent {
    InputContent::InputImage(InputImageContent {
        detail: ImageDetail::Auto,
        image_url: Some(attachment.data_url),
        file_id: None,
    })
}

fn user_content_to_chat_content(
    content: UserMessageContent,
) -> ChatCompletionRequestUserMessageContent {
    let parts = content.parts();
    if !parts
        .iter()
        .any(|part| matches!(part, UserMessagePart::Image { .. }))
    {
        return ChatCompletionRequestUserMessageContent::Text(content.text);
    }

    let parts = parts
        .into_iter()
        .map(|part| match part {
            UserMessagePart::Text { text } => ChatCompletionRequestUserMessageContentPart::Text(
                ChatCompletionRequestMessageContentPartText { text },
            ),
            UserMessagePart::Image { attachment } => chat_image_part(attachment),
        })
        .collect();
    ChatCompletionRequestUserMessageContent::Array(parts)
}

fn chat_image_part(attachment: UserImageAttachment) -> ChatCompletionRequestUserMessageContentPart {
    ChatCompletionRequestUserMessageContentPart::ImageUrl(
        ChatCompletionRequestMessageContentPartImage {
            image_url: ImageUrl {
                url: attachment.data_url,
                detail: None,
            },
        },
    )
}
