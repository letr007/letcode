use anyhow::Result;
use async_openai::types::chat::{
    ChatCompletionMessageToolCall, ChatCompletionMessageToolCalls,
    ChatCompletionRequestAssistantMessage, ChatCompletionRequestAssistantMessageContent,
    ChatCompletionRequestDeveloperMessage, ChatCompletionRequestDeveloperMessageContent,
    ChatCompletionRequestMessage, ChatCompletionRequestSystemMessage,
    ChatCompletionRequestSystemMessageContent, ChatCompletionRequestToolMessage,
    ChatCompletionRequestToolMessageContent, ChatCompletionRequestUserMessage,
    ChatCompletionRequestUserMessageContent, ChatCompletionTool, ChatCompletionTools,
    CreateChatCompletionRequest, FunctionCall, FunctionObject, Verbosity as ChatVerbosity,
};
use async_openai::types::responses::{
    CreateResponse, EasyInputContent, EasyInputMessage, FunctionCallOutput,
    FunctionCallOutputItemParam, FunctionTool, FunctionToolCall, InputItem, Item, MessageType,
    OutputStatus, Reasoning, ReasoningEffort as OpenAiReasoningEffort,
    ReasoningSummary as ResponseReasoningSummary, ResponseTextParam, Role,
    TextResponseFormatConfiguration, Tool, Verbosity as ResponseVerbosity,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::config::ApiProtocol;
use crate::evidence::{EvidenceRecord, estimate_evidence_tokens, evidence_context_message};

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ModelRequestMetadata {
    pub context_window: Option<u64>,
    pub max_output_tokens: Option<u64>,
    pub supports_tools: bool,
    pub supports_reasoning: bool,
    pub reasoning_effort: Option<ModelReasoningEffort>,
    pub reasoning_summary: Option<ModelReasoningSummary>,
    pub text_verbosity: Option<ModelTextVerbosity>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelReasoningEffort {
    None,
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelReasoningSummary {
    Auto,
    Concise,
    Detailed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelTextVerbosity {
    Low,
    Medium,
    High,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub parameters: Value,
    #[serde(default = "default_tool_strict")]
    pub strict: bool,
}

fn default_tool_strict() -> bool {
    true
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
    UserText {
        text: String,
    },
    InternalContinuation {
        text: String,
    },
    AssistantText {
        text: String,
    },
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

    pub fn internal_continuation(text: impl Into<String>) -> Self {
        Self::InternalContinuation { text: text.into() }
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
    pub evidence: &'a [EvidenceRecord],
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
    pub estimated_evidence_tokens: u64,
    pub original_history_items: usize,
    pub retained_history_items: usize,
    pub dropped_history_items: usize,
    pub selected_evidence_items: usize,
    pub dropped_evidence_items: usize,
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
    #[allow(dead_code)]
    pub selected_evidence_ids: Vec<String>,
}

const MIN_CONTEXT_WINDOW_TOKENS: u64 = 1024;
const DEFAULT_FALLBACK_CONTEXT_WINDOW_TOKENS: u64 = 8 * 1024;
const MIN_OUTPUT_RESERVE_TOKENS: u64 = 128;
const DEFAULT_FALLBACK_OUTPUT_RESERVE_TOKENS: u64 = 1024;
const SAFETY_OVERHEAD_TOKENS: u64 = 256;

pub fn build_request(input: RequestBuilderInput<'_>) -> Result<BuildResult> {
    validate_model_metadata(input.model)?;
    let context_window = input.model.context_window_tokens();
    let tools_tokens = if input.model.supports_tools {
        estimate_tools_tokens(input.tools)
    } else {
        0
    };
    let input_budget = context_window
        .saturating_sub(input.model.output_reserve_tokens())
        .saturating_sub(SAFETY_OVERHEAD_TOKENS)
        .saturating_sub(tools_tokens)
        .max(1);
    let protected_start = input.protected_start_index.min(input.history.len());
    let protected_tokens = estimate_history_tokens(&input.history[protected_start..]);
    let prelude_tokens = estimate_prelude_tokens(input.prelude);
    ensure_protected_context_within_budget(input_budget, prelude_tokens, protected_tokens, 0)?;
    let evidence_room =
        input_budget.saturating_sub(protected_tokens.saturating_add(prelude_tokens));
    let evidence_budget = evidence_budget_tokens(context_window).min(evidence_room);
    let current_query = current_user_query(input.history, input.protected_start_index);
    let (mut evidence_message, mut selected_evidence_ids, mut dropped_evidence_items) =
        if evidence_budget > 0 {
            evidence_context_message(input.evidence, &current_query, evidence_budget)
        } else {
            (None, Vec::new(), input.evidence.len())
        };
    let mut estimated_evidence_tokens = evidence_message
        .as_deref()
        .map(estimate_evidence_tokens)
        .unwrap_or(0);
    if protected_tokens
        .saturating_add(prelude_tokens)
        .saturating_add(estimated_evidence_tokens)
        > input_budget
    {
        evidence_message = None;
        selected_evidence_ids.clear();
        dropped_evidence_items = input.evidence.len();
        estimated_evidence_tokens = 0;
    }

    let (history, budget) = retain_history(
        input.prelude,
        input.history,
        input.protected_start_index,
        input.model,
        input.tools,
        EvidenceBudgetReport {
            estimated_evidence_tokens,
            selected_evidence_items: selected_evidence_ids.len(),
            dropped_evidence_items,
        },
    );
    let request = match input.protocol {
        ApiProtocol::Responses => BuiltRequest::Responses(build_responses_request(
            input.model_id,
            input.model,
            input.prelude,
            &history,
            evidence_message.as_deref(),
            input.tools,
        )),
        ApiProtocol::Completions => BuiltRequest::Completions(build_completions_request(
            input.model_id,
            input.model,
            input.prelude,
            &history,
            evidence_message.as_deref(),
            input.tools,
        )),
    };

    Ok(BuildResult {
        request,
        budget,
        selected_evidence_ids,
    })
}

fn ensure_protected_context_within_budget(
    input_budget: u64,
    prelude_tokens: u64,
    protected_tokens: u64,
    evidence_tokens: u64,
) -> Result<()> {
    let fixed_tokens = prelude_tokens
        .saturating_add(protected_tokens)
        .saturating_add(evidence_tokens);
    if fixed_tokens > input_budget {
        anyhow::bail!(
            "protected current context exceeds input budget: protected/current context tokens ({fixed_tokens}) exceed budget ({input_budget}); prelude={prelude_tokens}, protected={protected_tokens}, evidence={evidence_tokens}"
        );
    }
    Ok(())
}

fn validate_model_metadata(model: ModelRequestMetadata) -> Result<()> {
    if let Some(max_output_tokens) = model.max_output_tokens {
        if max_output_tokens > u32::MAX as u64 {
            anyhow::bail!("model.max_output_tokens must be at most {}", u32::MAX);
        }
    }
    if let Some(temperature) = model.temperature {
        validate_f32_range("model.temperature", temperature, 0.0, 2.0)?;
    }
    if let Some(top_p) = model.top_p {
        validate_f32_range("model.top_p", top_p, 0.0, 1.0)?;
    }
    Ok(())
}

fn validate_f32_range(label: &str, value: f32, min: f32, max: f32) -> Result<()> {
    if !value.is_finite() || value < min || value > max {
        anyhow::bail!("{label} must be between {min} and {max}");
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct EvidenceBudgetReport {
    estimated_evidence_tokens: u64,
    selected_evidence_items: usize,
    dropped_evidence_items: usize,
}

fn retain_history(
    prelude: &[PromptMessage],
    history: &[HistoryItem],
    protected_start_index: usize,
    model: ModelRequestMetadata,
    tools: &[ToolSpec],
    evidence_budget: EvidenceBudgetReport,
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

    let fixed_tokens = prelude_tokens
        .saturating_add(protected_tokens)
        .saturating_add(evidence_budget.estimated_evidence_tokens);

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
        .saturating_add(evidence_budget.estimated_evidence_tokens)
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
            estimated_evidence_tokens: evidence_budget.estimated_evidence_tokens,
            original_history_items: history_len,
            retained_history_items,
            dropped_history_items,
            selected_evidence_items: evidence_budget.selected_evidence_items,
            dropped_evidence_items: evidence_budget.dropped_evidence_items,
            truncated: dropped_history_items > 0,
        },
    )
}

fn current_user_query(history: &[HistoryItem], protected_start_index: usize) -> String {
    history
        .iter()
        .skip(protected_start_index.min(history.len()))
        .rev()
        .find_map(|item| match item {
            HistoryItem::UserText { text } => Some(text.clone()),
            _ => None,
        })
        .or_else(|| {
            history.iter().rev().find_map(|item| match item {
                HistoryItem::UserText { text } => Some(text.clone()),
                _ => None,
            })
        })
        .unwrap_or_default()
}

fn evidence_budget_tokens(context_window_tokens: u64) -> u64 {
    context_window_tokens
        .saturating_mul(15)
        .saturating_div(100)
        .clamp(512, 3_000)
}

fn drop_leading_orphan_tool_items(items: &mut Vec<HistoryItem>) {
    let Some(first_valid) = items.iter().position(|item| {
        matches!(
            item,
            HistoryItem::UserText { .. }
                | HistoryItem::InternalContinuation { .. }
                | HistoryItem::AssistantText { .. }
        )
    }) else {
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
    evidence_message: Option<&str>,
    tools: &[ToolSpec],
) -> CreateResponse {
    let mut input = prelude
        .iter()
        .cloned()
        .map(prelude_to_response_input)
        .collect::<Vec<_>>();
    append_history_with_evidence_response(&mut input, history, evidence_message);
    let tools = if model.supports_tools {
        Some(tools.iter().map(tool_to_response_tool).collect())
    } else {
        None
    };
    let parallel_tool_calls = model.supports_tools.then_some(false);

    CreateResponse {
        model: Some(model_id.to_string()),
        input: input.into(),
        max_output_tokens: model.max_output_tokens.and_then(u64_to_u32),
        previous_response_id: None,
        reasoning: response_reasoning(model),
        temperature: model.temperature,
        text: response_text(model),
        tools,
        parallel_tool_calls,
        top_p: model.top_p,
        ..Default::default()
    }
}

fn response_reasoning(model: ModelRequestMetadata) -> Option<Reasoning> {
    if !model.supports_reasoning {
        return None;
    }
    let effort = model.reasoning_effort.map(openai_reasoning_effort);
    let summary = model.reasoning_summary.map(response_reasoning_summary);
    (effort.is_some() || summary.is_some()).then_some(Reasoning { effort, summary })
}

fn response_text(model: ModelRequestMetadata) -> Option<ResponseTextParam> {
    model.text_verbosity.map(|verbosity| ResponseTextParam {
        format: TextResponseFormatConfiguration::Text,
        verbosity: Some(response_verbosity(verbosity)),
    })
}

fn prelude_to_response_input(message: PromptMessage) -> InputItem {
    let role = match message.role {
        PromptRole::System => Role::System,
        PromptRole::Developer => Role::Developer,
    };
    response_text_message(role, message.text)
}

fn append_history_with_evidence_response(
    input: &mut Vec<InputItem>,
    history: &[HistoryItem],
    evidence_message: Option<&str>,
) {
    let evidence_insert_index = evidence_message.and_then(|_| last_user_history_index(history));
    for (index, item) in history.iter().cloned().enumerate() {
        if evidence_insert_index == Some(index) {
            input.push(response_text_message(
                Role::Developer,
                evidence_message.expect("evidence exists").to_string(),
            ));
        }
        input.extend(history_to_response_inputs(item));
    }
    if evidence_message.is_some() && evidence_insert_index.is_none() {
        input.push(response_text_message(
            Role::Developer,
            evidence_message.expect("evidence exists").to_string(),
        ));
    }
}

fn history_to_response_inputs(item: HistoryItem) -> Vec<InputItem> {
    match item {
        HistoryItem::UserText { text } | HistoryItem::InternalContinuation { text } => {
            vec![response_text_message(Role::User, text)]
        }
        HistoryItem::AssistantText { text } => vec![response_text_message(Role::Assistant, text)],
        HistoryItem::AssistantToolCalls { calls, .. } => calls
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
            .collect(),
        HistoryItem::ToolOutput {
            call_id,
            output_json,
        } => {
            vec![InputItem::Item(Item::FunctionCallOutput(
                FunctionCallOutputItemParam {
                    call_id,
                    output: FunctionCallOutput::Text(output_json),
                    id: None,
                    status: None,
                },
            ))]
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
        strict: Some(tool.strict),
        defer_loading: None,
    })
}

fn build_completions_request(
    model_id: &str,
    model: ModelRequestMetadata,
    prelude: &[PromptMessage],
    history: &[HistoryItem],
    evidence_message: Option<&str>,
    tools: &[ToolSpec],
) -> CreateChatCompletionRequest {
    let mut messages = prelude
        .iter()
        .cloned()
        .map(prelude_to_chat_message)
        .collect::<Vec<_>>();
    append_history_with_evidence_chat(&mut messages, history, evidence_message);
    let tools = if model.supports_tools {
        Some(tools.iter().map(tool_to_chat_tool).collect())
    } else {
        None
    };
    let parallel_tool_calls = model.supports_tools.then_some(false);

    CreateChatCompletionRequest {
        model: model_id.to_string(),
        messages,
        max_completion_tokens: model.max_output_tokens.and_then(u64_to_u32),
        reasoning_effort: model
            .supports_reasoning
            .then_some(model.reasoning_effort)
            .flatten()
            .map(openai_reasoning_effort),
        stream: Some(true),
        n: Some(1),
        temperature: model.temperature,
        top_p: model.top_p,
        tools,
        parallel_tool_calls,
        verbosity: model.text_verbosity.map(chat_verbosity),
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

fn append_history_with_evidence_chat(
    messages: &mut Vec<ChatCompletionRequestMessage>,
    history: &[HistoryItem],
    evidence_message: Option<&str>,
) {
    let evidence_insert_index = evidence_message.and_then(|_| last_user_history_index(history));
    for (index, item) in history.iter().cloned().enumerate() {
        if evidence_insert_index == Some(index) {
            messages.push(prelude_to_chat_message(PromptMessage::developer(
                evidence_message.expect("evidence exists"),
            )));
        }
        messages.push(history_to_chat_message(item));
    }
    if evidence_message.is_some() && evidence_insert_index.is_none() {
        messages.push(prelude_to_chat_message(PromptMessage::developer(
            evidence_message.expect("evidence exists"),
        )));
    }
}

fn last_user_history_index(history: &[HistoryItem]) -> Option<usize> {
    history.iter().rposition(|item| {
        matches!(
            item,
            HistoryItem::UserText { .. } | HistoryItem::InternalContinuation { .. }
        )
    })
}

fn history_to_chat_message(item: HistoryItem) -> ChatCompletionRequestMessage {
    match item {
        HistoryItem::UserText { text } | HistoryItem::InternalContinuation { text } => {
            ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage {
                content: ChatCompletionRequestUserMessageContent::Text(text),
                name: None,
            })
        }
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
                            ChatCompletionMessageToolCalls::Function(
                                ChatCompletionMessageToolCall {
                                    id: call.call_id,
                                    function: FunctionCall {
                                        name: call.name,
                                        arguments: call.arguments_json,
                                    },
                                },
                            )
                        })
                        .collect(),
                ),
                function_call: None,
            })
        }
        HistoryItem::ToolOutput {
            call_id,
            output_json,
        } => ChatCompletionRequestMessage::Tool(ChatCompletionRequestToolMessage {
            content: ChatCompletionRequestToolMessageContent::Text(output_json),
            tool_call_id: call_id,
        }),
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
            strict: Some(tool.strict),
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
    use crate::evidence::{EvidenceKind, EvidenceRecord, EvidenceSource};
    use serde_json::json;

    fn metadata(context_window: u64) -> ModelRequestMetadata {
        ModelRequestMetadata {
            context_window: Some(context_window),
            max_output_tokens: Some(256),
            supports_tools: true,
            supports_reasoning: false,
            ..Default::default()
        }
    }

    fn evidence(id: &str, summary: &str, path: &str, sequence: u64) -> EvidenceRecord {
        EvidenceRecord {
            id: id.to_string(),
            sequence,
            timestamp_ms: 0,
            evidence_kind: EvidenceKind::FileExcerpt,
            title: format!("read {path}"),
            summary: summary.to_string(),
            detail: None,
            source: EvidenceSource::File {
                path: path.to_string(),
                start_line: Some(1),
                end_line: Some(3),
            },
            tags: vec![path.to_string()],
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
            evidence: &[],
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
    fn responses_request_includes_model_generation_parameters() {
        let result = build_request(RequestBuilderInput {
            protocol: ApiProtocol::Responses,
            model_id: "gpt-test",
            model: ModelRequestMetadata {
                context_window: Some(8192),
                max_output_tokens: Some(2048),
                supports_tools: true,
                supports_reasoning: true,
                reasoning_effort: Some(ModelReasoningEffort::High),
                reasoning_summary: Some(ModelReasoningSummary::Auto),
                text_verbosity: Some(ModelTextVerbosity::Low),
                temperature: Some(0.2),
                top_p: Some(0.8),
            },
            prelude: &[],
            history: &[HistoryItem::user("hello")],
            protected_start_index: 0,
            tools: &[],
            evidence: &[],
        })
        .expect("request builds");

        let BuiltRequest::Responses(request) = result.request else {
            panic!("expected responses request");
        };
        let json = serde_json::to_value(&request).expect("request serializes");

        assert_eq!(json["max_output_tokens"], 2048);
        assert_eq!(json["reasoning"]["effort"], "high");
        assert_eq!(json["reasoning"]["summary"], "auto");
        assert_eq!(json["text"]["verbosity"], "low");
        assert_json_f64_close(&json["temperature"], 0.2);
        assert_json_f64_close(&json["top_p"], 0.8);
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
            evidence: &[],
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
    fn completions_request_includes_model_generation_parameters() {
        let result = build_request(RequestBuilderInput {
            protocol: ApiProtocol::Completions,
            model_id: "chat-test",
            model: ModelRequestMetadata {
                context_window: Some(8192),
                max_output_tokens: Some(2048),
                supports_tools: true,
                supports_reasoning: true,
                reasoning_effort: Some(ModelReasoningEffort::Minimal),
                reasoning_summary: Some(ModelReasoningSummary::Detailed),
                text_verbosity: Some(ModelTextVerbosity::High),
                temperature: Some(0.3),
                top_p: Some(0.7),
            },
            prelude: &[],
            history: &[HistoryItem::user("hello")],
            protected_start_index: 0,
            tools: &[],
            evidence: &[],
        })
        .expect("request builds");

        let BuiltRequest::Completions(request) = result.request else {
            panic!("expected completions request");
        };
        let json = serde_json::to_value(&request).expect("request serializes");

        assert_eq!(json["max_completion_tokens"], 2048);
        assert_eq!(json["reasoning_effort"], "minimal");
        assert_eq!(json["verbosity"], "high");
        assert_json_f64_close(&json["temperature"], 0.3);
        assert_json_f64_close(&json["top_p"], 0.7);
        assert!(json.get("reasoning_summary").is_none());
    }

    fn assert_json_f64_close(value: &serde_json::Value, expected: f64) {
        let actual = value.as_f64().expect("value should be a number");
        assert!(
            (actual - expected).abs() < 0.000_001,
            "{actual} != {expected}"
        );
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
            evidence: &[],
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
            evidence: &[],
        })
        .expect("request builds");

        let BuiltRequest::Completions(request) = result.request else {
            panic!("expected completions request");
        };
        assert_eq!(request.messages.len(), 3);
        assert!(matches!(
            request.messages[0],
            ChatCompletionRequestMessage::System(_)
        ));
        assert!(matches!(
            request.messages[1],
            ChatCompletionRequestMessage::Developer(_)
        ));
        assert!(matches!(
            request.messages[2],
            ChatCompletionRequestMessage::User(_)
        ));
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
            evidence: &[],
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
            strict: true,
        }];

        let without_tools = build_request(RequestBuilderInput {
            protocol: ApiProtocol::Responses,
            model_id: "gpt-test",
            model: metadata(4096),
            prelude: &[],
            history: &history,
            protected_start_index: 2,
            tools: &[],
            evidence: &[],
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
            evidence: &[],
        })
        .expect("request builds");

        assert!(with_tools.budget.estimated_tools_tokens > 0);
        assert!(
            with_tools.budget.retained_history_items <= without_tools.budget.retained_history_items
        );
    }

    #[test]
    fn selected_evidence_is_injected_before_current_user_for_both_protocols() {
        let history = vec![
            HistoryItem::user("old question"),
            HistoryItem::assistant("old answer"),
            HistoryItem::user("What did src/evidence.rs say?"),
        ];
        let evidence = vec![evidence(
            "ev-1",
            "src/evidence.rs defines compact evidence records",
            "src/evidence.rs",
            1,
        )];

        let responses = build_request(RequestBuilderInput {
            protocol: ApiProtocol::Responses,
            model_id: "gpt-test",
            model: metadata(8192),
            prelude: &[],
            history: &history,
            protected_start_index: 2,
            tools: &[],
            evidence: &evidence,
        })
        .expect("request builds");
        let BuiltRequest::Responses(request) = responses.request else {
            panic!("expected responses request");
        };
        let json = serde_json::to_value(&request).expect("request serializes");
        let input = json["input"].as_array().expect("input array");
        assert_eq!(input[0]["role"], "user");
        assert_eq!(input[1]["role"], "assistant");
        assert_eq!(input[2]["role"], "developer");
        assert_eq!(input[3]["role"], "user");
        assert!(
            input[2]["content"]
                .as_str()
                .unwrap_or_default()
                .contains("ev-1")
        );
        assert_eq!(responses.selected_evidence_ids, vec!["ev-1"]);
        assert_eq!(responses.budget.selected_evidence_items, 1);

        let completions = build_request(RequestBuilderInput {
            protocol: ApiProtocol::Completions,
            model_id: "chat-test",
            model: metadata(8192),
            prelude: &[],
            history: &history,
            protected_start_index: 2,
            tools: &[],
            evidence: &evidence,
        })
        .expect("request builds");
        let BuiltRequest::Completions(request) = completions.request else {
            panic!("expected completions request");
        };
        assert!(matches!(
            request.messages[0],
            ChatCompletionRequestMessage::User(_)
        ));
        assert!(matches!(
            request.messages[1],
            ChatCompletionRequestMessage::Assistant(_)
        ));
        assert!(matches!(
            request.messages[2],
            ChatCompletionRequestMessage::Developer(_)
        ));
        assert!(matches!(
            request.messages[3],
            ChatCompletionRequestMessage::User(_)
        ));
        let json = serde_json::to_string(&request.messages[2]).expect("message serializes");
        assert!(json.contains("Relevant evidence"));
        assert!(json.contains("ev-1"));
    }

    #[test]
    fn evidence_is_dropped_when_current_turn_leaves_no_context_room() {
        let model = metadata(1024);
        let input_budget = model
            .context_window_tokens()
            .saturating_sub(model.output_reserve_tokens())
            .saturating_sub(SAFETY_OVERHEAD_TOKENS)
            .max(1);
        let exact_fit = (0..10_000)
            .map(|len| "x".repeat(len))
            .find(|text| {
                estimate_history_item_tokens(&HistoryItem::user(text.clone())) == input_budget
            })
            .expect("should find exact fit for input budget");
        let history = vec![HistoryItem::user(exact_fit)];
        let evidence = vec![evidence(
            "ev-1",
            "src/evidence.rs defines compact evidence records",
            "src/evidence.rs",
            1,
        )];

        let result = build_request(RequestBuilderInput {
            protocol: ApiProtocol::Responses,
            model_id: "gpt-test",
            model,
            prelude: &[],
            history: &history,
            protected_start_index: 0,
            tools: &[],
            evidence: &evidence,
        })
        .expect("request builds");

        assert!(result.selected_evidence_ids.is_empty());
        assert_eq!(result.budget.dropped_evidence_items, 1);
    }

    #[test]
    fn oversized_optional_evidence_is_dropped_instead_of_failing_protected_context() {
        let model = metadata(1024);
        let input_budget = model
            .context_window_tokens()
            .saturating_sub(model.output_reserve_tokens())
            .saturating_sub(SAFETY_OVERHEAD_TOKENS)
            .max(1);
        let near_fit = (0..10_000)
            .map(|len| "x".repeat(len))
            .find(|text| {
                estimate_history_item_tokens(&HistoryItem::user(text.clone()))
                    == input_budget.saturating_sub(1)
            })
            .expect("should find near fit for input budget");
        let history = vec![HistoryItem::user(near_fit)];
        let evidence = vec![evidence(
            "ev-1",
            "x ".repeat(200).as_str(),
            "src/evidence.rs",
            1,
        )];

        let result = build_request(RequestBuilderInput {
            protocol: ApiProtocol::Responses,
            model_id: "gpt-test",
            model,
            prelude: &[],
            history: &history,
            protected_start_index: 0,
            tools: &[],
            evidence: &evidence,
        })
        .expect("optional evidence should be dropped instead of failing protected context");

        assert!(result.selected_evidence_ids.is_empty());
        assert_eq!(result.budget.dropped_evidence_items, 1);
    }

    #[test]
    fn returns_error_when_protected_current_turn_exceeds_input_budget() {
        let history = vec![
            HistoryItem::user("old context"),
            HistoryItem::assistant("old reply"),
            HistoryItem::user("x".repeat(20_000)),
        ];

        let err = build_request(RequestBuilderInput {
            protocol: ApiProtocol::Responses,
            model_id: "gpt-test",
            model: metadata(1024),
            prelude: &[],
            history: &history,
            protected_start_index: 2,
            tools: &[],
            evidence: &[],
        })
        .expect_err("protected current turn should fail fast");

        let message = err.to_string();
        assert!(message.contains("protected"));
        assert!(message.contains("current"));
        assert!(message.contains("context"));
        assert!(message.contains("budget"));
    }
}
