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
    ReasoningEffort as OpenAiReasoningEffort, ReasoningSummary as ResponseReasoningSummary,
    ResponseTextParam, Role, ServiceTier as ResponseServiceTier, TextResponseFormatConfiguration,
    Tool, Verbosity as ResponseVerbosity,
};

use crate::config::{ApiProtocol, PromptCacheRetention};
use crate::user_content::{UserImageAttachment, UserMessageContent, UserMessagePart};

use super::prompt_plan::{PromptPlan, PromptSegment, PromptSegmentContent, PromptSegmentRole};
use super::{
    ModelReasoningEffort, ModelReasoningSummary, ModelRequestMetadata, ModelTextVerbosity,
    PromptMessage, PromptRole, ToolSpec, cache_request_fields,
};

pub(super) fn build_responses_request(
    model_id: &str,
    model: ModelRequestMetadata,
    prompt_plan: &PromptPlan,
    tools: &[ToolSpec],
) -> CreateResponse {
    let input = prompt_plan
        .segments
        .iter()
        .flat_map(prompt_segment_to_response_inputs)
        .collect::<Vec<_>>();
    let cache = cache_request_fields(
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

pub(super) fn prompt_segment_to_response_inputs(segment: &PromptSegment) -> Vec<InputItem> {
    match (&segment.role, &segment.content) {
        (PromptSegmentRole::System, PromptSegmentContent::Text { text }) => {
            vec![response_text_message(Role::System, text.clone())]
        }
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
            PromptSegmentContent::AssistantToolCalls { text, calls },
        ) => {
            let mut input = text
                .clone()
                .filter(|text| !text.is_empty())
                .map(|text| vec![response_text_message(Role::Assistant, text)])
                .unwrap_or_default();
            input.extend(
                calls
                    .iter()
                    .cloned()
                    .map(|call| {
                        InputItem::Item(Item::FunctionCall(FunctionToolCall {
                            arguments: call.arguments_json,
                            call_id: call.call_id,
                            namespace: None,
                            name: call.name,
                            id: None,
                            status: None::<OutputStatus>,
                        }))
                    })
                    .collect::<Vec<_>>(),
            );
            input
        }
        (
            PromptSegmentRole::Tool,
            PromptSegmentContent::ToolOutput {
                call_id,
                output_json,
            },
        ) => {
            vec![InputItem::Item(Item::FunctionCallOutput(
                FunctionCallOutputItemParam {
                    call_id: call_id.clone(),
                    output: FunctionCallOutput::Text(output_json.clone()),
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
            PromptSegmentContent::AssistantToolCalls { text, calls },
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
            function_call: None,
        }),
        (
            PromptSegmentRole::Tool,
            PromptSegmentContent::ToolOutput {
                call_id,
                output_json,
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

pub(super) fn build_completions_request(
    model_id: &str,
    model: ModelRequestMetadata,
    prompt_plan: &PromptPlan,
    tools: &[ToolSpec],
) -> CreateChatCompletionRequest {
    let messages = prompt_plan
        .segments
        .iter()
        .map(prompt_segment_to_chat_message)
        .collect::<Vec<_>>();
    let cache = cache_request_fields(
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

    CreateChatCompletionRequest {
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
    }
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
        refusal: None,
        name: None,
        audio: None,
        tool_calls: None,
        function_call: None,
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
