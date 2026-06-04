use anyhow::Result;
use async_openai::types::chat::{
    ChatCompletionMessageToolCall, ChatCompletionMessageToolCalls,
    ChatCompletionRequestDeveloperMessage, ChatCompletionRequestDeveloperMessageContent,
    ChatCompletionRequestAssistantMessage, ChatCompletionRequestAssistantMessageContent,
    ChatCompletionRequestMessage, ChatCompletionRequestToolMessage,
    ChatCompletionRequestToolMessageContent, ChatCompletionRequestUserMessage,
    ChatCompletionRequestUserMessageContent, ChatCompletionRequestSystemMessage,
    ChatCompletionRequestSystemMessageContent, ChatCompletionTool, ChatCompletionTools,
    CreateChatCompletionRequest, FunctionCall, FunctionObject,
};
use async_openai::types::responses::{
    CreateResponse, EasyInputContent, EasyInputMessage, FunctionCallOutput,
    FunctionCallOutputItemParam, FunctionTool, FunctionToolCall, InputItem, Item, MessageType,
    OutputStatus, Role, Tool,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::config::ApiProtocol;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ModelRequestMetadata {
    pub context_window: Option<u64>,
    pub max_output_tokens: Option<u64>,
    pub supports_tools: bool,
    pub supports_reasoning: bool,
}

impl ModelRequestMetadata {
    pub fn context_window_tokens(self) -> u64 {
        self.context_window
            .filter(|v| *v > 0)
            .unwrap_or(DEFAULT_FALLBACK_CONTEXT_WINDOW_TOKENS)
            .max(MIN_CONTEXT_WINDOW_TOKENS)
    }

    pub fn output_reserve_tokens(self) -> u64 {
        self.max_output_tokens
            .filter(|v| *v > 0)
            .unwrap_or(DEFAULT_FALLBACK_OUTPUT_RESERVE_TOKENS)
            .max(MIN_OUTPUT_RESERVE_TOKENS)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PromptRole {
    System,
    Developer,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptMessage {
    pub role: PromptRole,
    pub text: String,
}

impl PromptMessage {
    pub fn developer(text: impl Into<String>) -> Self {
        Self {
            role: PromptRole::Developer,
            text: text.into(),
        }
    }

    pub fn system(text: impl Into<String>) -> Self {
        Self {
            role: PromptRole::System,
            text: text.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HistoryToolCall {
    pub call_id: String,
    pub name: String,
    pub arguments_json: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum HistoryItem {
    UserText { text: String },
    AssistantText { text: String },
    AssistantToolCalls {
        text: Option<String>,
        calls: Vec<HistoryToolCall>,
    },
    ToolOutput {
        call_id: String,
        output_json: String,
    },
}

impl HistoryItem {
    pub fn user(text: impl Into<String>) -> Self {
        Self::UserText { text: text.into() }
    }

    pub fn assistant(text: impl Into<String>) -> Self {
        Self::AssistantText { text: text.into() }
    }
}

#[derive(Debug, Clone)]
pub struct RequestBuilderInput<'a> {
    pub protocol: ApiProtocol,
    pub model_id: &'a str,
    pub model: ModelRequestMetadata,
    pub prelude: &'a [PromptMessage],
    pub history: &'a [HistoryItem],
    pub protected_start_index: usize,
    pub tools: &'a [ToolSpec],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BudgetReport {
    pub context_window_tokens: u64,
    pub input_budget_tokens: u64,
    pub estimated_request_tokens: u64,
    pub estimated_prelude_tokens: u64,
    pub estimated_protected_tokens: u64,
    pub estimated_retained_history_tokens: u64,
    pub estimated_tools_tokens: u64,
    pub original_history_items: usize,
    pub retained_history_items: usize,
    pub dropped_history_items: usize,
    pub truncated: bool,
}

#[derive(Debug, Clone)]
pub enum BuiltRequest {
    Responses(CreateResponse),
    Completions(CreateChatCompletionRequest),
}

#[derive(Debug, Clone)]
pub struct BuildResult {
    pub request: BuiltRequest,
    pub budget: BudgetReport,
}

const MIN_CONTEXT_WINDOW_TOKENS: u64 = 1024;
const DEFAULT_FALLBACK_CONTEXT_WINDOW_TOKENS: u64 = 8 * 1024;
const MIN_OUTPUT_RESERVE_TOKENS: u64 = 128;
const DEFAULT_FALLBACK_OUTPUT_RESERVE_TOKENS: u64 = 1024;
const SAFETY_OVERHEAD_TOKENS: u64 = 256;

pub fn build_request(input: RequestBuilderInput<'_>) -> Result<BuildResult> {
    let (history, budget) = retain_history(
        input.prelude,
        input.history,
        input.protected_start_index,
        input.model,
        input.tools,
    );
    let request = match input.protocol {
        ApiProtocol::Responses => BuiltRequest::Responses(build_responses_request(
            input.model_id,
            input.model,
            input.prelude,
            &history,
            input.tools,
        )),
        ApiProtocol::Completions => BuiltRequest::Completions(build_completions_request(
            input.model_id,
            input.model,
            input.prelude,
            &history,
            input.tools,
        )),
    };

    Ok(BuildResult { request, budget })
}

fn retain_history(
    prelude: &[PromptMessage],
    history: &[HistoryItem],
    protected_start_index: usize,
    model: ModelRequestMetadata,
    tools: &[ToolSpec],
) -> (Vec<HistoryItem>, BudgetReport) {
    let history_len = history.len();
    let protected_start = protected_start_index.min(history_len);
    let (older, protected) = history.split_at(protected_start);

    let prelude_tokens = estimate_prelude_tokens(prelude);
    let protected_tokens = estimate_history_tokens(protected);
    let context_window = model.context_window_tokens();
    let tools_tokens = if model.supports_tools {
        estimate_tools_tokens(tools)
    } else {
        0
    };
    let input_budget = context_window
        .saturating_sub(model.output_reserve_tokens())
        .saturating_sub(SAFETY_OVERHEAD_TOKENS)
        .saturating_sub(tools_tokens)
        .max(1);

    let mut retained_older = Vec::new();
    let mut retained_older_tokens = 0_u64;

    let fixed_tokens = prelude_tokens.saturating_add(protected_tokens);

    if fixed_tokens < input_budget {
        for item in older.iter().rev() {
            let cost = estimate_history_item_tokens(item);
            let next = fixed_tokens
                .saturating_add(retained_older_tokens)
                .saturating_add(cost);
            if next > input_budget {
                break;
            }
            retained_older.push(item.clone());
            retained_older_tokens = retained_older_tokens.saturating_add(cost);
        }
        retained_older.reverse();
        drop_leading_orphan_tool_items(&mut retained_older);
        retained_older_tokens = estimate_history_tokens(&retained_older);
    }

    let mut retained = Vec::with_capacity(retained_older.len() + protected.len());
    retained.extend(retained_older.iter().cloned());
    retained.extend(protected.iter().cloned());

    let retained_history_items = retained.len();
    let dropped_history_items = history_len.saturating_sub(retained_history_items);
    let estimated_request_tokens = protected_tokens
        .saturating_add(prelude_tokens)
        .saturating_add(retained_older_tokens)
        .saturating_add(tools_tokens);

    (
        retained,
        BudgetReport {
            context_window_tokens: context_window,
            input_budget_tokens: input_budget,
            estimated_request_tokens,
            estimated_prelude_tokens: prelude_tokens,
            estimated_protected_tokens: protected_tokens,
            estimated_retained_history_tokens: retained_older_tokens,
            estimated_tools_tokens: tools_tokens,
            original_history_items: history_len,
            retained_history_items,
            dropped_history_items,
            truncated: dropped_history_items > 0,
        },
    )
}

fn drop_leading_orphan_tool_items(items: &mut Vec<HistoryItem>) {
    let Some(first_valid) = items
        .iter()
        .position(|item| matches!(item, HistoryItem::UserText { .. } | HistoryItem::AssistantText { .. }))
    else {
        items.clear();
        return;
    };
    if first_valid > 0 {
        items.drain(..first_valid);
    }
}

fn build_responses_request(
    model_id: &str,
    model: ModelRequestMetadata,
    prelude: &[PromptMessage],
    history: &[HistoryItem],
    tools: &[ToolSpec],
) -> CreateResponse {
    let input = prelude
        .iter()
        .cloned()
        .map(prelude_to_response_input)
        .chain(history
        .iter()
        .cloned()
        .flat_map(history_to_response_inputs))
        .collect::<Vec<_>>();
    let tools = if model.supports_tools {
        Some(tools.iter().map(tool_to_response_tool).collect())
    } else {
        None
    };
    let parallel_tool_calls = model.supports_tools.then_some(false);

    CreateResponse {
        model: Some(model_id.to_string()),
        input: input.into(),
        previous_response_id: None,
        tools,
        parallel_tool_calls,
        ..Default::default()
    }
}

fn prelude_to_response_input(message: PromptMessage) -> InputItem {
    let role = match message.role {
        PromptRole::System => Role::System,
        PromptRole::Developer => Role::Developer,
    };
    response_text_message(role, message.text)
}

fn history_to_response_inputs(item: HistoryItem) -> Vec<InputItem> {
    match item {
        HistoryItem::UserText { text } => vec![response_text_message(Role::User, text)],
        HistoryItem::AssistantText { text } => vec![response_text_message(Role::Assistant, text)],
        HistoryItem::AssistantToolCalls { calls, .. } => {
            calls
                .into_iter()
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
                .collect()
        }
        HistoryItem::ToolOutput { call_id, output_json } => {
            vec![InputItem::Item(Item::FunctionCallOutput(FunctionCallOutputItemParam {
                call_id,
                output: FunctionCallOutput::Text(output_json),
                id: None,
                status: None,
            }))]
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

fn tool_to_response_tool(tool: &ToolSpec) -> Tool {
    Tool::Function(FunctionTool {
        name: tool.name.clone(),
        description: Some(tool.description.clone()),
        parameters: Some(tool.parameters.clone()),
        strict: Some(true),
        defer_loading: None,
    })
}

fn build_completions_request(
    model_id: &str,
    model: ModelRequestMetadata,
    prelude: &[PromptMessage],
    history: &[HistoryItem],
    tools: &[ToolSpec],
) -> CreateChatCompletionRequest {
    let messages = prelude
        .iter()
        .cloned()
        .map(prelude_to_chat_message)
        .chain(history.iter().cloned().map(history_to_chat_message))
        .collect::<Vec<_>>();
    let tools = if model.supports_tools {
        Some(tools.iter().map(tool_to_chat_tool).collect())
    } else {
        None
    };
    let parallel_tool_calls = model.supports_tools.then_some(false);

    CreateChatCompletionRequest {
        model: model_id.to_string(),
        messages,
        stream: Some(true),
        n: Some(1),
        tools,
        parallel_tool_calls,
        ..Default::default()
    }
}

fn prelude_to_chat_message(message: PromptMessage) -> ChatCompletionRequestMessage {
    match message.role {
        PromptRole::System => ChatCompletionRequestMessage::System(
            ChatCompletionRequestSystemMessage {
                content: ChatCompletionRequestSystemMessageContent::Text(message.text),
                name: None,
            },
        ),
        PromptRole::Developer => ChatCompletionRequestMessage::Developer(
            ChatCompletionRequestDeveloperMessage {
                content: ChatCompletionRequestDeveloperMessageContent::Text(message.text),
                name: None,
            },
        ),
    }
}

fn history_to_chat_message(item: HistoryItem) -> ChatCompletionRequestMessage {
    match item {
        HistoryItem::UserText { text } => ChatCompletionRequestMessage::User(
            ChatCompletionRequestUserMessage {
                content: ChatCompletionRequestUserMessageContent::Text(text),
                name: None,
            },
        ),
        HistoryItem::AssistantText { text } => chat_assistant_text(text),
        HistoryItem::AssistantToolCalls { text, calls } => {
            ChatCompletionRequestMessage::Assistant(ChatCompletionRequestAssistantMessage {
                content: text.map(ChatCompletionRequestAssistantMessageContent::Text),
                refusal: None,
                name: None,
                audio: None,
                tool_calls: Some(
                    calls
                        .into_iter()
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
            })
        }
        HistoryItem::ToolOutput { call_id, output_json } => {
            ChatCompletionRequestMessage::Tool(ChatCompletionRequestToolMessage {
                content: ChatCompletionRequestToolMessageContent::Text(output_json),
                tool_call_id: call_id,
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

fn tool_to_chat_tool(tool: &ToolSpec) -> ChatCompletionTools {
    ChatCompletionTools::Function(ChatCompletionTool {
        function: FunctionObject {
            name: tool.name.clone(),
            description: Some(tool.description.clone()),
            parameters: Some(tool.parameters.clone()),
            strict: None,
        },
    })
}

fn estimate_history_item_tokens(item: &HistoryItem) -> u64 {
    let json_len = serde_json::to_string(item).map(|s| s.len()).unwrap_or(0);
    ((json_len as u64 + 2) / 3).saturating_add(8)
}

fn estimate_history_tokens(items: &[HistoryItem]) -> u64 {
    items.iter().map(estimate_history_item_tokens).sum()
}

fn estimate_prelude_tokens(items: &[PromptMessage]) -> u64 {
    items
        .iter()
        .map(|item| {
            let json_len = serde_json::to_string(item).map(|s| s.len()).unwrap_or(0);
            ((json_len as u64 + 2) / 3).saturating_add(8)
        })
        .sum()
}

fn estimate_tools_tokens(tools: &[ToolSpec]) -> u64 {
    if tools.is_empty() {
        return 0;
    }
    let json_len = serde_json::to_string(tools).map(|s| s.len()).unwrap_or(0);
    ((json_len as u64 + 2) / 3).saturating_add(16)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn metadata(context_window: u64) -> ModelRequestMetadata {
        ModelRequestMetadata {
            context_window: Some(context_window),
            max_output_tokens: Some(256),
            supports_tools: true,
            supports_reasoning: false,
        }
    }

    #[test]
    fn builds_responses_request_from_unified_history() {
        let history = vec![HistoryItem::user("hello"), HistoryItem::assistant("hi")];
        let result = build_request(RequestBuilderInput {
            protocol: ApiProtocol::Responses,
            model_id: "gpt-test",
            model: metadata(8192),
            prelude: &[],
            history: &history,
            protected_start_index: 0,
            tools: &[],
        })
        .expect("request builds");

        let BuiltRequest::Responses(request) = result.request else {
            panic!("expected responses request");
        };
        let json = serde_json::to_string(&request).expect("request serializes");
        assert!(json.contains("hello"));
        assert!(json.contains("hi"));
    }

    #[test]
    fn builds_completions_request_from_unified_history() {
        let history = vec![HistoryItem::user("hello")];
        let result = build_request(RequestBuilderInput {
            protocol: ApiProtocol::Completions,
            model_id: "chat-test",
            model: metadata(8192),
            prelude: &[],
            history: &history,
            protected_start_index: 0,
            tools: &[],
        })
        .expect("request builds");

        let BuiltRequest::Completions(request) = result.request else {
            panic!("expected completions request");
        };
        assert_eq!(request.model, "chat-test");
        assert_eq!(request.messages.len(), 1);
        assert_eq!(request.stream, Some(true));
    }

    #[test]
    fn responses_prelude_is_stable_prefix_before_history() {
        let prelude = vec![
            PromptMessage::system("stable system"),
            PromptMessage::developer("stable developer"),
        ];
        let history = vec![HistoryItem::user("current user")];

        let result = build_request(RequestBuilderInput {
            protocol: ApiProtocol::Responses,
            model_id: "gpt-test",
            model: metadata(8192),
            prelude: &prelude,
            history: &history,
            protected_start_index: 0,
            tools: &[],
        })
        .expect("request builds");

        let BuiltRequest::Responses(request) = result.request else {
            panic!("expected responses request");
        };
        let json = serde_json::to_value(&request).expect("request serializes");
        let input = json["input"].as_array().expect("input should be array");
        assert_eq!(input[0]["role"], "system");
        assert_eq!(input[1]["role"], "developer");
        assert_eq!(input[2]["role"], "user");
        assert!(result.budget.estimated_prelude_tokens > 0);
    }

    #[test]
    fn completions_prelude_is_stable_prefix_before_history() {
        let prelude = vec![
            PromptMessage::system("stable system"),
            PromptMessage::developer("stable developer"),
        ];
        let history = vec![HistoryItem::user("current user")];

        let result = build_request(RequestBuilderInput {
            protocol: ApiProtocol::Completions,
            model_id: "chat-test",
            model: metadata(8192),
            prelude: &prelude,
            history: &history,
            protected_start_index: 0,
            tools: &[],
        })
        .expect("request builds");

        let BuiltRequest::Completions(request) = result.request else {
            panic!("expected completions request");
        };
        assert_eq!(request.messages.len(), 3);
        assert!(matches!(request.messages[0], ChatCompletionRequestMessage::System(_)));
        assert!(matches!(request.messages[1], ChatCompletionRequestMessage::Developer(_)));
        assert!(matches!(request.messages[2], ChatCompletionRequestMessage::User(_)));
        assert!(result.budget.estimated_prelude_tokens > 0);
    }

    #[test]
    fn truncates_oldest_history_but_keeps_protected_items() {
        let long = "x".repeat(10_000);
        let history = vec![
            HistoryItem::user("old"),
            HistoryItem::assistant(long),
            HistoryItem::user("current"),
        ];
        let result = build_request(RequestBuilderInput {
            protocol: ApiProtocol::Responses,
            model_id: "gpt-test",
            model: metadata(1200),
            prelude: &[],
            history: &history,
            protected_start_index: 2,
            tools: &[],
        })
        .expect("request builds");

        assert!(result.budget.truncated);
        let BuiltRequest::Responses(request) = result.request else {
            panic!("expected responses request");
        };
        let json = serde_json::to_string(&request).expect("request serializes");
        assert!(json.contains("current"));
    }

    #[test]
    fn tool_schema_size_counts_toward_budget() {
        let long = "x".repeat(6000);
        let history = vec![
            HistoryItem::user("old"),
            HistoryItem::assistant(long),
            HistoryItem::user("current"),
        ];
        let tools = vec![ToolSpec {
            name: "big_tool".to_string(),
            description: "big".to_string(),
            parameters: json!({
                "type": "object",
                "properties": { "payload": { "type": "string", "description": "x".repeat(8000) } },
                "required": ["payload"],
                "additionalProperties": false
            }),
        }];

        let without_tools = build_request(RequestBuilderInput {
            protocol: ApiProtocol::Responses,
            model_id: "gpt-test",
            model: metadata(4096),
            prelude: &[],
            history: &history,
            protected_start_index: 2,
            tools: &[],
        })
        .expect("request builds");
        let with_tools = build_request(RequestBuilderInput {
            protocol: ApiProtocol::Responses,
            model_id: "gpt-test",
            model: metadata(4096),
            prelude: &[],
            history: &history,
            protected_start_index: 2,
            tools: &tools,
        })
        .expect("request builds");

        assert!(with_tools.budget.estimated_tools_tokens > 0);
        assert!(with_tools.budget.retained_history_items <= without_tools.budget.retained_history_items);
    }
}
