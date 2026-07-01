use anyhow::Result;
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
    Verbosity as ChatVerbosity,
};
use async_openai::types::responses::{
    CreateResponse, EasyInputContent, EasyInputMessage, FunctionCallOutput,
    FunctionCallOutputItemParam, FunctionTool, FunctionToolCall, ImageDetail, InputContent,
    InputImageContent, InputItem, InputMessage, InputRole, InputTextContent, Item, MessageItem,
    MessageType, OutputStatus, Reasoning, ReasoningEffort as OpenAiReasoningEffort,
    ReasoningSummary as ResponseReasoningSummary, ResponseTextParam, Role,
    TextResponseFormatConfiguration, Tool, Verbosity as ResponseVerbosity,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeSet;

use crate::config::ApiProtocol;
use crate::context_view::{
    ContextBlock, ContextBlockKind, ContextBlockRetention, ContextBlockSource,
    ContextViewProjection, ContextViewStatus, FoldedOutputMetadata, ProtectedReason,
};
use crate::evidence::{EvidenceRecord, estimate_evidence_tokens, evidence_context_message};
use crate::user_content::{UserImageAttachment, UserMessageContent, UserMessagePart};

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ModelRequestMetadata {
    pub context_window: Option<u64>,
    pub effective_input_limit_tokens: Option<u64>,
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

    pub fn effective_input_limit_tokens(self) -> Option<u64> {
        self.effective_input_limit_tokens.filter(|v| *v > 0)
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
    ContextSummary {
        text: String,
    },
    UserMessage {
        content: UserMessageContent,
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
    pub fn context_summary(text: impl Into<String>) -> Self {
        Self::ContextSummary { text: text.into() }
    }

    pub fn user(text: impl Into<String>) -> Self {
        Self::UserMessage {
            content: UserMessageContent::new(text, Vec::new()),
        }
    }

    pub fn user_content(content: UserMessageContent) -> Self {
        Self::UserMessage { content }
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
    pub context_view: Option<&'a ContextViewProjection>,
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

#[derive(Debug, Default)]
struct ContextViewPromptSections {
    prelude: Vec<PromptMessage>,
    history_prefix: Vec<HistoryItem>,
}

const MIN_CONTEXT_WINDOW_TOKENS: u64 = 1024;
const DEFAULT_FALLBACK_CONTEXT_WINDOW_TOKENS: u64 = 8 * 1024;
const MIN_OUTPUT_RESERVE_TOKENS: u64 = 128;
const DEFAULT_FALLBACK_OUTPUT_RESERVE_TOKENS: u64 = 1024;
const SAFETY_OVERHEAD_TOKENS: u64 = 256;

pub fn effective_input_budget_tokens(model: ModelRequestMetadata, tools: &[ToolSpec]) -> u64 {
    let tools_tokens = if model.supports_tools {
        estimate_tools_tokens(tools)
    } else {
        0
    };
    effective_input_budget_tokens_for_tool_tokens(model, tools_tokens)
}

fn effective_input_budget_tokens_for_tool_tokens(
    model: ModelRequestMetadata,
    tools_tokens: u64,
) -> u64 {
    let configured_input_budget = model
        .context_window_tokens()
        .saturating_sub(model.output_reserve_tokens())
        .saturating_sub(SAFETY_OVERHEAD_TOKENS);
    let capped_input_budget = match model.effective_input_limit_tokens() {
        Some(cap) => configured_input_budget.min(cap.max(1)),
        None => configured_input_budget,
    };
    capped_input_budget.saturating_sub(tools_tokens).max(1)
}

pub fn build_request(input: RequestBuilderInput<'_>) -> Result<BuildResult> {
    let mut effective_prelude = input.prelude.to_vec();
    let mut effective_history = input.history.to_vec();
    let mut effective_protected_start_index = input.protected_start_index;

    if let Some(context_view) = input.context_view {
        let sections = assemble_context_view_sections(
            context_view,
            input.history,
            input.protected_start_index,
        );
        effective_prelude.extend(sections.prelude);
        effective_protected_start_index =
            effective_protected_start_index.saturating_add(sections.history_prefix.len());
        effective_history.splice(0..0, sections.history_prefix);
    }

    validate_model_metadata(input.model)?;
    let context_window = input.model.context_window_tokens();
    let tools_tokens = if input.model.supports_tools {
        estimate_tools_tokens(input.tools)
    } else {
        0
    };
    let input_budget = effective_input_budget_tokens_for_tool_tokens(input.model, tools_tokens);
    let protected_start = effective_protected_start_index.min(effective_history.len());
    let protected_tokens = estimate_history_tokens(&effective_history[protected_start..]);
    let prelude_tokens = estimate_prelude_tokens(&effective_prelude);
    ensure_protected_context_within_budget(input_budget, prelude_tokens, protected_tokens, 0)?;
    let evidence_room =
        input_budget.saturating_sub(protected_tokens.saturating_add(prelude_tokens));
    let evidence_budget = evidence_budget_tokens(context_window).min(evidence_room);
    let current_query = current_user_query(&effective_history, effective_protected_start_index);
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
        &effective_prelude,
        &effective_history,
        effective_protected_start_index,
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
            &effective_prelude,
            &history,
            evidence_message.as_deref(),
            input.tools,
        )),
        ApiProtocol::Completions => BuiltRequest::Completions(build_completions_request(
            input.model_id,
            input.model,
            &effective_prelude,
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

fn assemble_context_view_sections(
    context_view: &ContextViewProjection,
    history: &[HistoryItem],
    protected_start_index: usize,
) -> ContextViewPromptSections {
    let mut sections = ContextViewPromptSections::default();
    let sorted_blocks = sorted_context_blocks(context_view);

    let protected_blocks = sorted_blocks
        .iter()
        .filter(|(id, block)| block.is_protected() && !is_resolved(context_view, id))
        .map(|(id, block)| (*id, *block))
        .collect::<Vec<_>>();
    if !protected_blocks.is_empty() {
        sections.prelude.push(PromptMessage::developer(format!(
            "[Context: Hard Context]\n{}",
            protected_blocks
                .iter()
                .map(|(id, block)| format_protected_context_block_line(id, block))
                .collect::<Vec<_>>()
                .join("\n")
        )));
    }

    let protected_ids = protected_blocks
        .iter()
        .map(|(id, _)| id.as_str().to_string())
        .collect::<BTreeSet<_>>();
    let pinned_blocks = sorted_blocks
        .iter()
        .filter(|(id, _)| {
            is_pinned_visible(context_view, id) && !protected_ids.contains(id.as_str())
        })
        .map(|(id, block)| (*id, *block))
        .collect::<Vec<_>>();
    if !pinned_blocks.is_empty() {
        sections.prelude.push(PromptMessage::developer(format!(
            "[Context: Pinned Context]\n{}",
            pinned_blocks
                .iter()
                .map(|(id, block)| format_context_block_line(id, block, false))
                .collect::<Vec<_>>()
                .join("\n")
        )));
    }

    if let Some(tail_section) = build_protected_tail_section(history, protected_start_index) {
        sections
            .history_prefix
            .push(HistoryItem::context_summary(tail_section));
    }

    let visible_index_blocks = sorted_blocks
        .iter()
        .filter(|(id, block)| include_in_context_index(context_view, id, block))
        .map(|(id, block)| format_context_block_line(id, block, false))
        .collect::<Vec<_>>();
    if !visible_index_blocks.is_empty() {
        sections
            .history_prefix
            .push(HistoryItem::context_summary(format!(
                "[Context: Index]\n{}",
                visible_index_blocks.join("\n")
            )));
    }

    let summaries = context_view
        .summary_artifacts
        .iter()
        .map(format_summary_artifact)
        .collect::<Vec<_>>();
    if !summaries.is_empty() {
        sections
            .history_prefix
            .push(HistoryItem::context_summary(format!(
                "[Context: Summaries]\n{}",
                summaries.join("\n")
            )));
    }

    let folded = sorted_context_blocks(context_view)
        .into_iter()
        .filter(|(id, block)| {
            block.folded_output_id.is_some()
                && (is_normally_visible(context_view, id) || is_opened(context_view, id))
        })
        .filter_map(|(_, block)| {
            block
                .folded_output_id
                .as_deref()
                .and_then(|output_id| context_view.folded_outputs.get(output_id))
        })
        .map(format_folded_placeholder)
        .collect::<Vec<_>>();
    if !folded.is_empty() {
        sections
            .history_prefix
            .push(HistoryItem::context_summary(format!(
                "[Context: Folded Outputs]\n{}",
                folded.join("\n")
            )));
    }

    if let Some(open_id) = context_view.view_state.open_detail_block_id()
        && let Some(block) = context_view.blocks.get(open_id)
        && view_status(context_view, open_id) != ContextViewStatus::RemovedFromView
        && !is_resolved(context_view, open_id)
    {
        sections
            .history_prefix
            .push(HistoryItem::context_summary(format!(
                "[Context: Opened Details]\n{}\nDetail: {}",
                format_context_block_line(open_id, block, false),
                excerpt(&block.detail, 1200)
            )));
    }

    sections
}

fn is_opened(
    context_view: &ContextViewProjection,
    block_id: &crate::context_view::ContextBlockId,
) -> bool {
    context_view.view_state.open_detail_block_id() == Some(block_id)
}

fn view_status(
    context_view: &ContextViewProjection,
    block_id: &crate::context_view::ContextBlockId,
) -> ContextViewStatus {
    context_view
        .view_state
        .status(block_id)
        .unwrap_or(ContextViewStatus::Visible)
}

fn is_normally_visible(
    context_view: &ContextViewProjection,
    block_id: &crate::context_view::ContextBlockId,
) -> bool {
    matches!(
        view_status(context_view, block_id),
        ContextViewStatus::Visible | ContextViewStatus::Pinned
    )
}

fn is_resolved(
    context_view: &ContextViewProjection,
    block_id: &crate::context_view::ContextBlockId,
) -> bool {
    view_status(context_view, block_id) == ContextViewStatus::Resolved
}

fn is_pinned_visible(
    context_view: &ContextViewProjection,
    block_id: &crate::context_view::ContextBlockId,
) -> bool {
    view_status(context_view, block_id) == ContextViewStatus::Pinned
}

fn include_in_context_index(
    context_view: &ContextViewProjection,
    block_id: &crate::context_view::ContextBlockId,
    block: &ContextBlock,
) -> bool {
    if is_resolved(context_view, block_id) {
        return false;
    }
    if block.retention_class() == ContextBlockRetention::Debug {
        return is_pinned_visible(context_view, block_id) || is_opened(context_view, block_id);
    }
    block.is_protected() || is_normally_visible(context_view, block_id)
}

fn sorted_context_blocks(
    context_view: &ContextViewProjection,
) -> Vec<(&crate::context_view::ContextBlockId, &ContextBlock)> {
    let mut blocks = context_view.blocks.iter().collect::<Vec<_>>();
    blocks.sort_by(|(left_id, left), (right_id, right)| {
        left.source_start_sequence
            .or(left.available_sequence)
            .unwrap_or(u64::MAX)
            .cmp(
                &right
                    .source_start_sequence
                    .or(right.available_sequence)
                    .unwrap_or(u64::MAX),
            )
            .then_with(|| left_id.as_str().cmp(right_id.as_str()))
    });
    blocks
}

fn build_protected_tail_section(
    history: &[HistoryItem],
    protected_start_index: usize,
) -> Option<String> {
    let protected = &history[protected_start_index.min(history.len())..];
    let tail = protected.iter().rev().take(4).collect::<Vec<_>>();
    if tail.is_empty() {
        return None;
    }
    let mut lines = tail
        .into_iter()
        .rev()
        .map(format_history_item_line)
        .collect::<Vec<_>>();
    lines.retain(|line| !line.is_empty());
    if lines.is_empty() {
        None
    } else {
        Some(format!("[Context: Active Tail]\n{}", lines.join("\n")))
    }
}

fn format_history_item_line(item: &HistoryItem) -> String {
    match item {
        HistoryItem::ContextSummary { text } => format!("- summary: {}", excerpt(text, 240)),
        HistoryItem::UserMessage { content } => {
            format!("- user: {}", excerpt(&content.display_text(), 240))
        }
        HistoryItem::InternalContinuation { text } => {
            format!("- continuation: {}", excerpt(text, 240))
        }
        HistoryItem::AssistantText { text } => format!("- assistant: {}", excerpt(text, 240)),
        HistoryItem::AssistantToolCalls { calls, .. } => format!(
            "- tool_calls: {}",
            calls
                .iter()
                .map(|call| call.name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ),
        HistoryItem::ToolOutput { call_id, .. } => format!("- tool_output: {call_id}"),
    }
}

fn format_context_block_line(
    block_id: &crate::context_view::ContextBlockId,
    block: &ContextBlock,
    include_protected: bool,
) -> String {
    let mut tags = Vec::new();
    tags.push(format!("id={}", block_id.as_str()));
    tags.push(format!("kind={}", context_block_kind_label(block.kind)));
    tags.push(format!(
        "retention={}",
        context_block_retention_label(block.retention_class())
    ));
    if include_protected && !block.protected_reasons.is_empty() {
        tags.push(format!(
            "protected={}",
            block
                .protected_reasons
                .iter()
                .map(|reason| protected_reason_label(*reason))
                .collect::<Vec<_>>()
                .join(",")
        ));
    }
    tags.push(format!("source={}", format_block_source(&block.source)));
    format!(
        "- [{}] {} :: {}",
        tags.join(" "),
        block.title,
        excerpt(&block.detail, 240)
    )
}

fn format_protected_context_block_line(
    block_id: &crate::context_view::ContextBlockId,
    block: &ContextBlock,
) -> String {
    let mut tags = Vec::new();
    tags.push(format!("id={}", block_id.as_str()));
    tags.push(format!("kind={}", context_block_kind_label(block.kind)));
    tags.push(format!(
        "retention={}",
        context_block_retention_label(block.retention_class())
    ));
    if !block.protected_reasons.is_empty() {
        tags.push(format!(
            "protected={}",
            block
                .protected_reasons
                .iter()
                .map(|reason| protected_reason_label(*reason))
                .collect::<Vec<_>>()
                .join(",")
        ));
    }
    tags.push(format!("source={}", format_block_source(&block.source)));
    format!("- [{}] {} :: {}", tags.join(" "), block.title, block.detail)
}

fn format_summary_artifact(artifact: &crate::context_view::SummaryArtifact) -> String {
    format!(
        "- id={} node={} version={} kind={} source_node={} source_block={} span={}..{} :: {}",
        artifact.artifact_id,
        artifact.node_id,
        artifact.version,
        artifact.artifact_kind,
        artifact.source_node_id.as_deref().unwrap_or("-"),
        artifact.source_block_id.as_deref().unwrap_or("-"),
        artifact
            .source_start_sequence
            .map(|value| value.to_string())
            .unwrap_or_else(|| "-".into()),
        artifact
            .source_end_sequence
            .map(|value| value.to_string())
            .unwrap_or_else(|| "-".into()),
        excerpt(&artifact.summary, 240)
    )
}

fn format_folded_placeholder(metadata: &FoldedOutputMetadata) -> String {
    format!(
        "- output_id={} tool={} stream={} status={} size={} lines={} command={} ",
        metadata.output_id,
        metadata.tool_name.as_deref().unwrap_or("-"),
        metadata.stream.as_deref().unwrap_or("-"),
        folded_status(metadata),
        metadata.byte_count,
        metadata.line_count,
        metadata.shell_command.as_deref().unwrap_or("-")
    )
}

fn format_block_source(source: &ContextBlockSource) -> String {
    match source {
        ContextBlockSource::TranscriptSpan {
            start_sequence,
            end_sequence,
        } => format!("transcript:{start_sequence}..{end_sequence}"),
        ContextBlockSource::SummaryArtifact { artifact_id } => format!("summary:{artifact_id}"),
        ContextBlockSource::FoldedOutput { output_id } => format!("folded:{output_id}"),
    }
}

fn context_block_kind_label(kind: ContextBlockKind) -> &'static str {
    match kind {
        ContextBlockKind::HardConstraint => "hard_constraint",
        ContextBlockKind::CurrentUserRequirement => "current_user_requirement",
        ContextBlockKind::UnresolvedError => "unresolved_error",
        ContextBlockKind::Permission => "permission",
        ContextBlockKind::FileWriteFact => "file_write_fact",
        ContextBlockKind::TestResult => "test_result",
        ContextBlockKind::CommitHash => "commit_hash",
        ContextBlockKind::ToolOutput => "tool_output",
        ContextBlockKind::Note => "note",
        ContextBlockKind::ReasoningNote => "reasoning_note",
    }
}

fn context_block_retention_label(retention: ContextBlockRetention) -> &'static str {
    match retention {
        ContextBlockRetention::Critical => "critical",
        ContextBlockRetention::Protected => "protected",
        ContextBlockRetention::Working => "working",
        ContextBlockRetention::Debug => "debug",
    }
}

fn protected_reason_label(reason: ProtectedReason) -> &'static str {
    match reason {
        ProtectedReason::HardConstraint => "hard_constraint",
        ProtectedReason::CurrentUserRequirement => "current_user_requirement",
        ProtectedReason::UnresolvedError => "unresolved_error",
        ProtectedReason::Permission => "permission",
        ProtectedReason::FileWriteFact => "file_write_fact",
        ProtectedReason::TestResult => "test_result",
        ProtectedReason::CommitHash => "commit_hash",
    }
}

fn folded_status(metadata: &FoldedOutputMetadata) -> String {
    match (metadata.exit_status, metadata.tool_ok) {
        (Some(status), Some(ok)) => format!("status={status},ok={ok}"),
        (Some(status), None) => format!("status={status}"),
        (None, Some(ok)) => format!("ok={ok}"),
        (None, None) => "unknown".into(),
    }
}

fn excerpt(text: &str, max_chars: usize) -> String {
    let compact = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut value = compact.chars().take(max_chars).collect::<String>();
    if compact.chars().count() > max_chars {
        value.push('…');
    }
    value
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
    if let Some(effective_input_limit_tokens) = model.effective_input_limit_tokens {
        if effective_input_limit_tokens == 0 {
            anyhow::bail!("model.effective_input_limit_tokens must be greater than 0");
        }
    }
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
    let input_budget = effective_input_budget_tokens_for_tool_tokens(model, tools_tokens);

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
    }

    let mut retained = Vec::with_capacity(retained_older.len() + protected.len());
    retained.extend(retained_older.iter().cloned());
    retained.extend(protected.iter().cloned());
    sanitize_tool_call_pairs(&mut retained);

    let retained_history_items = retained.len();
    let dropped_history_items = history_len.saturating_sub(retained_history_items);
    let retained_tokens = estimate_history_tokens(&retained);
    let estimated_request_tokens = prelude_tokens
        .saturating_add(evidence_budget.estimated_evidence_tokens)
        .saturating_add(retained_tokens)
        .saturating_add(tools_tokens);

    (
        retained,
        BudgetReport {
            context_window_tokens: context_window,
            input_budget_tokens: input_budget,
            estimated_request_tokens,
            estimated_prelude_tokens: prelude_tokens,
            estimated_protected_tokens: protected_tokens,
            estimated_retained_history_tokens: retained_tokens,
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
            HistoryItem::UserMessage { content } => Some(content.text.clone()),
            _ => None,
        })
        .or_else(|| {
            history.iter().rev().find_map(|item| match item {
                HistoryItem::UserMessage { content } => Some(content.text.clone()),
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

#[derive(Debug)]
struct PendingToolCallGroup {
    start_index: usize,
    pending_call_ids: Vec<String>,
}

fn sanitize_tool_call_pairs(items: &mut Vec<HistoryItem>) {
    let original = std::mem::take(items);
    let mut sanitized = Vec::with_capacity(original.len());
    let mut pending_group: Option<PendingToolCallGroup> = None;

    for item in original {
        match item {
            HistoryItem::AssistantToolCalls { calls, text } => {
                discard_incomplete_tool_call_group(&mut sanitized, &mut pending_group);
                if calls.is_empty() {
                    if let Some(text) = text {
                        sanitized.push(HistoryItem::AssistantText { text });
                    }
                    continue;
                }
                let pending_call_ids = calls
                    .iter()
                    .map(|call| call.call_id.clone())
                    .collect::<Vec<_>>();
                let start_index = sanitized.len();
                sanitized.push(HistoryItem::AssistantToolCalls { text, calls });
                pending_group = Some(PendingToolCallGroup {
                    start_index,
                    pending_call_ids,
                });
            }
            HistoryItem::ToolOutput {
                call_id,
                output_json,
            } => {
                let Some(group) = pending_group.as_mut() else {
                    continue;
                };
                let Some(position) = group.pending_call_ids.iter().position(|id| id == &call_id)
                else {
                    continue;
                };
                group.pending_call_ids.remove(position);
                sanitized.push(HistoryItem::ToolOutput {
                    call_id,
                    output_json,
                });
                if group.pending_call_ids.is_empty() {
                    pending_group = None;
                }
            }
            other => {
                discard_incomplete_tool_call_group(&mut sanitized, &mut pending_group);
                sanitized.push(other);
            }
        }
    }

    discard_incomplete_tool_call_group(&mut sanitized, &mut pending_group);
    *items = sanitized;
}

fn discard_incomplete_tool_call_group(
    sanitized: &mut Vec<HistoryItem>,
    pending_group: &mut Option<PendingToolCallGroup>,
) {
    let Some(group) = pending_group.take() else {
        return;
    };
    if group.pending_call_ids.is_empty() {
        return;
    }

    let replacement_text = match sanitized.get(group.start_index).cloned() {
        Some(HistoryItem::AssistantToolCalls {
            text: Some(text), ..
        }) => Some(text),
        _ => None,
    };
    sanitized.truncate(group.start_index);
    if let Some(text) = replacement_text {
        sanitized.push(HistoryItem::AssistantText { text });
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
        HistoryItem::ContextSummary { text } => vec![response_text_message(
            Role::Developer,
            format!("以下是当前会话的结构化摘要：\n\n{text}"),
        )],
        HistoryItem::UserMessage { content } => vec![response_user_message(content)],
        HistoryItem::InternalContinuation { text } => {
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

fn response_user_message(content: UserMessageContent) -> InputItem {
    InputItem::Item(Item::Message(MessageItem::Input(InputMessage {
        role: InputRole::User,
        content: user_content_to_response_content(content),
        status: None,
    })))
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

pub(crate) fn last_user_history_index(history: &[HistoryItem]) -> Option<usize> {
    history.iter().rposition(|item| {
        matches!(
            item,
            HistoryItem::UserMessage { .. } | HistoryItem::InternalContinuation { .. }
        )
    })
}

fn history_to_chat_message(item: HistoryItem) -> ChatCompletionRequestMessage {
    match item {
        HistoryItem::ContextSummary { text } => {
            ChatCompletionRequestMessage::Developer(ChatCompletionRequestDeveloperMessage {
                content: ChatCompletionRequestDeveloperMessageContent::Text(format!(
                    "以下是当前会话的结构化摘要：\n\n{text}"
                )),
                name: None,
            })
        }
        HistoryItem::UserMessage { content } => {
            ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage {
                content: user_content_to_chat_content(content),
                name: None,
            })
        }
        HistoryItem::InternalContinuation { text } => {
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

pub(crate) fn estimate_history_item_tokens(item: &HistoryItem) -> u64 {
    let json_len = serde_json::to_string(item).map(|s| s.len()).unwrap_or(0);
    ((json_len as u64 + 2) / 3).saturating_add(8)
}

fn user_content_to_response_content(content: UserMessageContent) -> Vec<InputContent> {
    let mut parts = Vec::with_capacity(1 + content.attachments.len());
    if !content.text.is_empty() {
        parts.push(InputContent::InputText(InputTextContent {
            text: content.text,
        }));
    }
    parts.extend(content.attachments.into_iter().map(response_image_part));
    parts
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
    if content.attachments.is_empty() {
        return ChatCompletionRequestUserMessageContent::Text(content.text);
    }

    let parts = content
        .parts()
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

pub(crate) fn estimate_history_tokens(items: &[HistoryItem]) -> u64 {
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
    use crate::agent::{ToolExecutionSummaryEvent, ValidationAdvisory};
    use crate::context_tree::ContextNodeStatus;
    use crate::context_view::{
        ContextBlockId, ContextViewStatus, DEFAULT_FOLDED_OUTPUT_THRESHOLD_BYTES,
        project_context_view,
    };
    use crate::evidence::{EvidenceKind, EvidenceRecord, EvidenceSource};
    use crate::tool::ToolResult;
    use crate::transcript::transcript_projection::{
        project_context_tree, project_context_view as project_restored_context_view,
        project_session_restore_snapshot,
    };
    use crate::transcript::{TranscriptEvent, TranscriptRecord};
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

    fn metadata_with_effective_input_limit(
        context_window: u64,
        effective_input_limit_tokens: u64,
    ) -> ModelRequestMetadata {
        ModelRequestMetadata {
            effective_input_limit_tokens: Some(effective_input_limit_tokens),
            ..metadata(context_window)
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

    fn transcript_record(sequence: u64, event: TranscriptEvent) -> TranscriptRecord {
        TranscriptRecord {
            session_id: "s".into(),
            sequence,
            timestamp_ms: 0,
            context_branch_id: None,
            event,
        }
    }

    fn sample_context_view(open_detail: bool) -> crate::context_view::ContextViewProjection {
        let mut records = vec![
            transcript_record(
                1,
                TranscriptEvent::UserMessage {
                    content: UserMessageContent::from("Do not drop hard constraints"),
                },
            ),
            transcript_record(
                2,
                TranscriptEvent::AssistantMessage {
                    content: "Pinned context note".into(),
                },
            ),
            transcript_record(
                3,
                TranscriptEvent::ContextViewOperationMetadata {
                    operation: "pin".into(),
                    node_id: None,
                    block_id: Some("block-seq-2-note".into()),
                    detail: None,
                },
            ),
            transcript_record(
                4,
                TranscriptEvent::ToolCallStarted {
                    call_id: "call-1".into(),
                    name: "shell__exec".into(),
                    args: json!({"command": "cargo test"}),
                },
            ),
            transcript_record(
                5,
                TranscriptEvent::ToolCallFinished {
                    call_id: "call-1".into(),
                    name: "shell__exec".into(),
                    ok: true,
                    output: crate::tool::ToolResult::ok(
                        "shell__exec",
                        json!({
                            "status": 0,
                            "stdout": "x".repeat(5000),
                            "stdout_truncated": false,
                            "stderr": "",
                            "stderr_truncated": false
                        }),
                    ),
                },
            ),
            transcript_record(
                6,
                TranscriptEvent::ContextSummaryArtifactMetadata {
                    node_id: "node-a".into(),
                    artifact_id: "sum-1".into(),
                    artifact_kind: "summary".into(),
                    version: Some(1),
                    summary: Some("Summary text".into()),
                    source_node_id: Some("node-a".into()),
                    source_block_id: Some("block-seq-2-note".into()),
                    source_start_sequence: Some(2),
                    source_end_sequence: Some(2),
                },
            ),
        ];
        if open_detail {
            records.push(transcript_record(
                7,
                TranscriptEvent::ContextViewOperationMetadata {
                    operation: "open_detail".into(),
                    node_id: None,
                    block_id: Some("block-seq-2-note".into()),
                    detail: None,
                },
            ));
        }
        project_context_view(&records).expect("context view projection")
    }

    fn request_json(result: BuildResult) -> String {
        match result.request {
            BuiltRequest::Responses(request) => serde_json::to_string(&request).expect("serialize"),
            BuiltRequest::Completions(request) => {
                serde_json::to_string(&request).expect("serialize")
            }
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
            context_view: None,
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
                effective_input_limit_tokens: None,
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
            context_view: None,
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
            context_view: None,
        })
        .expect("request builds");

        let BuiltRequest::Completions(request) = result.request else {
            panic!("expected completions request");
        };
        assert_eq!(request.model, "chat-test");
        assert_eq!(request.messages.len(), 1);
        assert_eq!(request.stream, Some(true));
        assert_eq!(
            request
                .stream_options
                .as_ref()
                .and_then(|options| options.include_usage),
            Some(true)
        );
    }

    #[test]
    fn completions_request_serializes_multimodal_user_message_parts() {
        let history = vec![HistoryItem::user_content(UserMessageContent::new(
            "describe this image",
            vec![UserImageAttachment {
                id: "img-1".into(),
                label: "screen.png".into(),
                mime: "image/png".into(),
                data_url: "data:image/png;base64,AAAA".into(),
            }],
        ))];
        let result = build_request(RequestBuilderInput {
            protocol: ApiProtocol::Completions,
            model_id: "chat-test",
            model: metadata(8192),
            prelude: &[],
            history: &history,
            protected_start_index: 0,
            tools: &[],
            evidence: &[],
            context_view: None,
        })
        .expect("request builds");

        let BuiltRequest::Completions(request) = result.request else {
            panic!("expected completions request");
        };
        let json = serde_json::to_value(&request).expect("request serializes");
        let content = &json["messages"][0]["content"];
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[0]["text"], "describe this image");
        assert_eq!(content[1]["type"], "image_url");
        assert_eq!(content[1]["image_url"]["url"], "data:image/png;base64,AAAA");
    }

    #[test]
    fn responses_request_serializes_multimodal_user_message_parts() {
        let history = vec![HistoryItem::user_content(UserMessageContent::new(
            "describe this image",
            vec![UserImageAttachment {
                id: "img-1".into(),
                label: "screen.png".into(),
                mime: "image/png".into(),
                data_url: "data:image/png;base64,AAAA".into(),
            }],
        ))];
        let result = build_request(RequestBuilderInput {
            protocol: ApiProtocol::Responses,
            model_id: "resp-test",
            model: metadata(8192),
            prelude: &[],
            history: &history,
            protected_start_index: 0,
            tools: &[],
            evidence: &[],
            context_view: None,
        })
        .expect("request builds");

        let BuiltRequest::Responses(request) = result.request else {
            panic!("expected responses request");
        };
        let json = serde_json::to_value(&request).expect("request serializes");
        let content = &json["input"][0]["content"];
        assert_eq!(json["input"][0]["type"], "message");
        assert_eq!(json["input"][0]["role"], "user");
        assert_eq!(content[0]["type"], "input_text");
        assert_eq!(content[0]["text"], "describe this image");
        assert_eq!(content[1]["type"], "input_image");
        assert_eq!(content[1]["image_url"], "data:image/png;base64,AAAA");
    }

    #[test]
    fn completions_request_includes_model_generation_parameters() {
        let result = build_request(RequestBuilderInput {
            protocol: ApiProtocol::Completions,
            model_id: "chat-test",
            model: ModelRequestMetadata {
                context_window: Some(8192),
                effective_input_limit_tokens: None,
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
            context_view: None,
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
            context_view: None,
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
    fn context_summary_is_encoded_as_developer_message_for_both_protocols() {
        let history = vec![HistoryItem::context_summary("目标\n- 修复 compaction")];

        let responses = build_request(RequestBuilderInput {
            protocol: ApiProtocol::Responses,
            model_id: "gpt-test",
            model: metadata(8192),
            prelude: &[],
            history: &history,
            protected_start_index: 0,
            tools: &[],
            evidence: &[],
            context_view: None,
        })
        .expect("responses request builds");
        let BuiltRequest::Responses(response_request) = responses.request else {
            panic!("expected responses request");
        };
        let response_json = serde_json::to_string(&response_request).expect("serialize response");
        assert!(response_json.contains("developer"));
        assert!(response_json.contains("以下是当前会话的结构化摘要"));

        let completions = build_request(RequestBuilderInput {
            protocol: ApiProtocol::Completions,
            model_id: "gpt-test",
            model: metadata(8192),
            prelude: &[],
            history: &history,
            protected_start_index: 0,
            tools: &[],
            evidence: &[],
            context_view: None,
        })
        .expect("completions request builds");
        let BuiltRequest::Completions(chat_request) = completions.request else {
            panic!("expected completions request");
        };
        assert!(matches!(
            chat_request.messages[0],
            ChatCompletionRequestMessage::Developer(_)
        ));
        let chat_json = serde_json::to_string(&chat_request.messages[0]).expect("serialize chat");
        assert!(chat_json.contains("以下是当前会话的结构化摘要"));
    }

    #[test]
    fn orphan_tool_outputs_are_dropped_when_building_chat_request() {
        let history = vec![
            HistoryItem::context_summary("旧工具调用已总结"),
            HistoryItem::ToolOutput {
                call_id: "call-orphan".into(),
                output_json: r#"{"ok":true}"#.into(),
            },
            HistoryItem::user("continue"),
        ];

        let result = build_request(RequestBuilderInput {
            protocol: ApiProtocol::Completions,
            model_id: "chat-test",
            model: metadata(8192),
            prelude: &[],
            history: &history,
            protected_start_index: 2,
            tools: &[],
            evidence: &[],
            context_view: None,
        })
        .expect("request builds");

        let BuiltRequest::Completions(request) = result.request else {
            panic!("expected chat completions request");
        };
        assert!(
            !request
                .messages
                .iter()
                .any(|message| matches!(message, ChatCompletionRequestMessage::Tool(_)))
        );
    }

    #[test]
    fn complete_tool_call_output_pairs_are_kept_when_building_chat_request() {
        let history = vec![
            HistoryItem::AssistantToolCalls {
                text: None,
                calls: vec![HistoryToolCall {
                    call_id: "call-read".into(),
                    name: "fs__read".into(),
                    arguments_json: r#"{"path":"src/main.rs"}"#.into(),
                }],
            },
            HistoryItem::ToolOutput {
                call_id: "call-read".into(),
                output_json: r#"{"ok":true}"#.into(),
            },
            HistoryItem::user("continue"),
        ];

        let result = build_request(RequestBuilderInput {
            protocol: ApiProtocol::Completions,
            model_id: "chat-test",
            model: metadata(8192),
            prelude: &[],
            history: &history,
            protected_start_index: 2,
            tools: &[],
            evidence: &[],
            context_view: None,
        })
        .expect("request builds");

        let BuiltRequest::Completions(request) = result.request else {
            panic!("expected chat completions request");
        };
        assert!(
            request
                .messages
                .iter()
                .any(|message| matches!(message, ChatCompletionRequestMessage::Assistant(_)))
        );
        assert!(
            request
                .messages
                .iter()
                .any(|message| matches!(message, ChatCompletionRequestMessage::Tool(_)))
        );
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
            context_view: None,
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
            context_view: None,
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
            context_view: None,
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
            context_view: None,
        })
        .expect("request builds");

        assert!(with_tools.budget.estimated_tools_tokens > 0);
        assert!(
            with_tools.budget.retained_history_items <= without_tools.budget.retained_history_items
        );
    }

    #[test]
    fn effective_input_limit_bounds_retained_history_budget() {
        let old_context = "x".repeat(6_000);
        let history = vec![
            HistoryItem::user("old question"),
            HistoryItem::assistant(old_context),
            HistoryItem::user("current question"),
        ];

        let uncapped = build_request(RequestBuilderInput {
            protocol: ApiProtocol::Responses,
            model_id: "gpt-test",
            model: metadata(32_000),
            prelude: &[],
            history: &history,
            protected_start_index: 2,
            tools: &[],
            evidence: &[],
            context_view: None,
        })
        .expect("uncapped request builds");
        let capped = build_request(RequestBuilderInput {
            protocol: ApiProtocol::Responses,
            model_id: "gpt-test",
            model: metadata_with_effective_input_limit(32_000, 900),
            prelude: &[],
            history: &history,
            protected_start_index: 2,
            tools: &[],
            evidence: &[],
            context_view: None,
        })
        .expect("effective-input-limited request builds");

        assert_eq!(capped.budget.input_budget_tokens, 900);
        assert!(capped.budget.truncated);
        assert!(uncapped.budget.retained_history_items > capped.budget.retained_history_items);
    }

    #[test]
    fn effective_input_limit_counts_tool_schema_tokens() {
        let history = vec![HistoryItem::user("current")];
        let tools = vec![ToolSpec {
            name: "read_file".to_string(),
            description: "read a file".to_string(),
            parameters: json!({
                "type": "object",
                "properties": { "path": { "type": "string" } },
                "required": ["path"],
                "additionalProperties": false
            }),
            strict: true,
        }];

        let capped = build_request(RequestBuilderInput {
            protocol: ApiProtocol::Responses,
            model_id: "gpt-test",
            model: metadata_with_effective_input_limit(32_000, 2_000),
            prelude: &[],
            history: &history,
            protected_start_index: 0,
            tools: &tools,
            evidence: &[],
            context_view: None,
        })
        .expect("effective-input-limited request with tools builds");

        assert!(capped.budget.estimated_tools_tokens > 0);
        assert_eq!(
            capped.budget.input_budget_tokens + capped.budget.estimated_tools_tokens,
            2_000
        );
        assert!(capped.budget.estimated_request_tokens <= 2_000);
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
            context_view: None,
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
            context_view: None,
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
            context_view: None,
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
            context_view: None,
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
            context_view: None,
        })
        .expect_err("protected current turn should fail fast");

        let message = err.to_string();
        assert!(message.contains("protected"));
        assert!(message.contains("current"));
        assert!(message.contains("context"));
        assert!(message.contains("budget"));
    }

    #[test]
    fn returns_error_when_protected_current_turn_exceeds_effective_input_limit() {
        let history = vec![HistoryItem::user("x".repeat(20_000))];

        let err = build_request(RequestBuilderInput {
            protocol: ApiProtocol::Responses,
            model_id: "gpt-test",
            model: metadata_with_effective_input_limit(32_000, 300),
            prelude: &[],
            history: &history,
            protected_start_index: 0,
            tools: &[],
            evidence: &[],
            context_view: None,
        })
        .expect_err("effective-input-limited protected current turn should fail fast");

        let message = err.to_string();
        assert!(message.contains("protected/current context tokens"));
        assert!(message.contains("exceed budget (300)"));
    }

    #[test]
    fn rejects_zero_effective_input_limit_metadata() {
        let err = build_request(RequestBuilderInput {
            protocol: ApiProtocol::Responses,
            model_id: "gpt-test",
            model: ModelRequestMetadata {
                effective_input_limit_tokens: Some(0),
                ..metadata(32_000)
            },
            prelude: &[],
            history: &[HistoryItem::user("current")],
            protected_start_index: 0,
            tools: &[],
            evidence: &[],
            context_view: None,
        })
        .expect_err("zero effective input limit should fail fast");

        assert!(
            err.to_string()
                .contains("model.effective_input_limit_tokens must be greater than 0")
        );
    }

    #[test]
    fn none_context_view_preserves_request_shape() {
        let history = vec![HistoryItem::user("hello"), HistoryItem::assistant("hi")];
        let baseline = build_request(RequestBuilderInput {
            protocol: ApiProtocol::Responses,
            model_id: "gpt-test",
            model: metadata(8192),
            prelude: &[],
            history: &history,
            protected_start_index: 0,
            tools: &[],
            evidence: &[],
            context_view: None,
        })
        .expect("request builds");
        let repeat = build_request(RequestBuilderInput {
            protocol: ApiProtocol::Responses,
            model_id: "gpt-test",
            model: metadata(8192),
            prelude: &[],
            history: &history,
            protected_start_index: 0,
            tools: &[],
            evidence: &[],
            context_view: None,
        })
        .expect("request rebuilds");

        let baseline_json = request_json(baseline);
        let repeat_json = request_json(repeat);
        assert_eq!(baseline_json, repeat_json);
        assert!(!baseline_json.contains("[Context:"));
    }

    #[test]
    fn context_view_prompt_sections_are_deterministic() {
        let history = vec![
            HistoryItem::assistant("previous"),
            HistoryItem::user("current user"),
        ];
        let context_view = sample_context_view(false);
        let first = request_json(
            build_request(RequestBuilderInput {
                protocol: ApiProtocol::Responses,
                model_id: "gpt-test",
                model: metadata(8192),
                prelude: &[],
                history: &history,
                protected_start_index: 1,
                tools: &[],
                evidence: &[],
                context_view: Some(&context_view),
            })
            .expect("request builds"),
        );
        let second = request_json(
            build_request(RequestBuilderInput {
                protocol: ApiProtocol::Responses,
                model_id: "gpt-test",
                model: metadata(8192),
                prelude: &[],
                history: &history,
                protected_start_index: 1,
                tools: &[],
                evidence: &[],
                context_view: Some(&context_view),
            })
            .expect("request rebuilds"),
        );
        assert_eq!(first, second);
    }

    #[test]
    fn context_view_sections_appear_in_required_order() {
        let history = vec![
            HistoryItem::assistant("previous"),
            HistoryItem::user("current user"),
        ];
        let context_view = sample_context_view(true);
        let sections = assemble_context_view_sections(&context_view, &history, 1);
        let mut combined = sections
            .prelude
            .iter()
            .map(|message| message.text.as_str())
            .collect::<Vec<_>>();
        combined.extend(
            sections
                .history_prefix
                .iter()
                .filter_map(|item| match item {
                    HistoryItem::ContextSummary { text } => Some(text.as_str()),
                    _ => None,
                }),
        );
        combined.push("current user");
        let combined = combined.join("\n");
        let mut cursor = 0usize;
        for needle in [
            "[Context: Hard Context]",
            "[Context: Pinned Context]",
            "[Context: Active Tail]",
            "[Context: Index]",
            "[Context: Summaries]",
            "[Context: Folded Outputs]",
            "[Context: Opened Details]",
            "current user",
        ] {
            let next = combined[cursor..].find(needle).expect("section present") + cursor;
            cursor = next + needle.len();
        }
    }

    #[test]
    fn opened_detail_only_changes_suffix_after_stable_context_prefix() {
        let history = vec![
            HistoryItem::assistant("previous"),
            HistoryItem::user("current user"),
        ];
        let closed_json = request_json(
            build_request(RequestBuilderInput {
                protocol: ApiProtocol::Responses,
                model_id: "gpt-test",
                model: metadata(8192),
                prelude: &[],
                history: &history,
                protected_start_index: 1,
                tools: &[],
                evidence: &[],
                context_view: Some(&sample_context_view(false)),
            })
            .expect("closed request builds"),
        );
        let open_json = request_json(
            build_request(RequestBuilderInput {
                protocol: ApiProtocol::Responses,
                model_id: "gpt-test",
                model: metadata(8192),
                prelude: &[],
                history: &history,
                protected_start_index: 1,
                tools: &[],
                evidence: &[],
                context_view: Some(&sample_context_view(true)),
            })
            .expect("open request builds"),
        );
        let marker = "[Context: Opened Details]";
        let folded_marker = "[Context: Folded Outputs]";
        let stable_end = open_json.find(marker).expect("opened marker present");
        let closed_end = closed_json
            .find(folded_marker)
            .expect("folded marker present")
            + folded_marker.len();
        assert_eq!(&closed_json[..closed_end], &open_json[..closed_end]);
        assert!(open_json[stable_end..].contains(marker));
    }

    #[test]
    fn protected_current_oversize_still_fails_with_context_view_present() {
        let history = vec![
            HistoryItem::user("old"),
            HistoryItem::user("x".repeat(20_000)),
        ];
        let context_view = sample_context_view(true);
        let err = build_request(RequestBuilderInput {
            protocol: ApiProtocol::Responses,
            model_id: "gpt-test",
            model: metadata(1024),
            prelude: &[],
            history: &history,
            protected_start_index: 1,
            tools: &[],
            evidence: &[],
            context_view: Some(&context_view),
        })
        .expect_err("protected current turn should still fail");
        assert!(
            err.to_string()
                .contains("protected current context exceeds input budget")
        );
    }

    #[test]
    fn hard_context_includes_full_protected_detail_without_truncation() {
        let long_detail = format!("HARD-CONTEXT-START {} HARD-CONTEXT-END", "x".repeat(600));
        let context_view = project_context_view(&[transcript_record(
            1,
            TranscriptEvent::UserMessage {
                content: UserMessageContent::from(long_detail.clone()),
            },
        )])
        .expect("context view projection");

        let result = build_request(RequestBuilderInput {
            protocol: ApiProtocol::Responses,
            model_id: "gpt-test",
            model: metadata(8192),
            prelude: &[],
            history: &[HistoryItem::user("current user")],
            protected_start_index: 0,
            tools: &[],
            evidence: &[],
            context_view: Some(&context_view),
        })
        .expect("request builds");

        let json = request_json(result);
        assert!(json.contains("[Context: Hard Context]"));
        assert!(json.contains(&long_detail));
    }

    #[test]
    fn archived_and_removed_blocks_are_suppressed_from_context_sections() {
        let context_view = project_context_view(&[
            transcript_record(
                1,
                TranscriptEvent::AssistantMessage {
                    content: "visible note".into(),
                },
            ),
            transcript_record(
                2,
                TranscriptEvent::AssistantMessage {
                    content: "archived note detail".into(),
                },
            ),
            transcript_record(
                3,
                TranscriptEvent::ContextViewOperationMetadata {
                    operation: "archive".into(),
                    node_id: None,
                    block_id: Some("block-seq-2-note".into()),
                    detail: None,
                },
            ),
            transcript_record(
                4,
                TranscriptEvent::AssistantMessage {
                    content: "removed note detail".into(),
                },
            ),
            transcript_record(
                5,
                TranscriptEvent::ContextViewOperationMetadata {
                    operation: "remove_from_view".into(),
                    node_id: None,
                    block_id: Some("block-seq-4-note".into()),
                    detail: None,
                },
            ),
        ])
        .expect("context view projection");

        let sections =
            assemble_context_view_sections(&context_view, &[HistoryItem::user("current")], 0);
        let combined = sections
            .prelude
            .iter()
            .map(|message| message.text.as_str())
            .chain(
                sections
                    .history_prefix
                    .iter()
                    .filter_map(|item| match item {
                        HistoryItem::ContextSummary { text } => Some(text.as_str()),
                        _ => None,
                    }),
            )
            .collect::<Vec<_>>()
            .join("\n");

        assert!(combined.contains("visible note"));
        assert!(!combined.contains("archived note detail"));
        assert!(!combined.contains("removed note detail"));
    }

    #[test]
    fn resolved_unresolved_errors_are_suppressed_from_context_sections() {
        let context_view = project_context_view(&[
            transcript_record(
                1,
                TranscriptEvent::Error {
                    message: "context view projection unavailable".into(),
                },
            ),
            transcript_record(
                2,
                TranscriptEvent::ContextViewOperationMetadata {
                    operation: "resolve".into(),
                    node_id: None,
                    block_id: Some("block-seq-1-error".into()),
                    detail: None,
                },
            ),
        ])
        .expect("resolved context view projection");

        let sections =
            assemble_context_view_sections(&context_view, &[HistoryItem::user("current")], 0);
        let combined = sections
            .prelude
            .iter()
            .map(|message| message.text.as_str())
            .chain(
                sections
                    .history_prefix
                    .iter()
                    .filter_map(|item| match item {
                        HistoryItem::ContextSummary { text } => Some(text.as_str()),
                        _ => None,
                    }),
            )
            .collect::<Vec<_>>()
            .join("\n");

        assert!(!combined.contains("context view projection unavailable"));
        assert!(!combined.contains("unresolved_error"));
    }

    #[test]
    fn reasoning_debug_notes_are_hidden_from_context_index_unless_opened() {
        let context_view = project_context_view(&[
            transcript_record(
                1,
                TranscriptEvent::ReasoningMessage {
                    content: "scratch reasoning trace".into(),
                },
            ),
            transcript_record(
                2,
                TranscriptEvent::AssistantMessage {
                    content: "durable assistant note".into(),
                },
            ),
        ])
        .expect("context view projection");

        let sections =
            assemble_context_view_sections(&context_view, &[HistoryItem::user("current")], 0);
        let combined = sections
            .history_prefix
            .iter()
            .filter_map(|item| match item {
                HistoryItem::ContextSummary { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");

        assert!(combined.contains("durable assistant note"));
        assert!(!combined.contains("scratch reasoning trace"));

        let opened_context_view = project_context_view(&[
            transcript_record(
                1,
                TranscriptEvent::ReasoningMessage {
                    content: "scratch reasoning trace".into(),
                },
            ),
            transcript_record(
                2,
                TranscriptEvent::ContextViewOperationMetadata {
                    operation: "open_detail".into(),
                    node_id: None,
                    block_id: Some("block-seq-1-reasoning-note".into()),
                    detail: None,
                },
            ),
        ])
        .expect("opened context view projection");
        let opened_sections = assemble_context_view_sections(
            &opened_context_view,
            &[HistoryItem::user("current")],
            0,
        );
        let opened_combined = opened_sections
            .history_prefix
            .iter()
            .filter_map(|item| match item {
                HistoryItem::ContextSummary { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");

        assert!(opened_combined.contains("[Context: Opened Details]"));
        assert!(opened_combined.contains("scratch reasoning trace"));
    }

    #[test]
    fn folded_placeholders_respect_archive_and_remove_visibility() {
        let context_view = project_context_view(&[
            transcript_record(
                1,
                TranscriptEvent::ToolCallStarted {
                    call_id: "call-1".into(),
                    name: "shell__exec".into(),
                    args: json!({"command": "cargo test"}),
                },
            ),
            transcript_record(
                2,
                TranscriptEvent::ToolCallFinished {
                    call_id: "call-1".into(),
                    name: "shell__exec".into(),
                    ok: true,
                    output: crate::tool::ToolResult::ok(
                        "shell__exec",
                        json!({
                            "status": 0,
                            "stdout": "a".repeat(5000),
                            "stdout_truncated": false,
                            "stderr": "b".repeat(5000),
                            "stderr_truncated": false
                        }),
                    ),
                },
            ),
            transcript_record(
                3,
                TranscriptEvent::ContextViewOperationMetadata {
                    operation: "archive".into(),
                    node_id: None,
                    block_id: Some("block-seq-2-folded-output-folded-output-seq-2-stdout".into()),
                    detail: None,
                },
            ),
            transcript_record(
                4,
                TranscriptEvent::ContextViewOperationMetadata {
                    operation: "remove_from_view".into(),
                    node_id: None,
                    block_id: Some("block-seq-2-folded-output-folded-output-seq-2-stderr".into()),
                    detail: None,
                },
            ),
        ])
        .expect("context view projection");

        let sections =
            assemble_context_view_sections(&context_view, &[HistoryItem::user("current")], 0);
        let combined = sections
            .history_prefix
            .iter()
            .filter_map(|item| match item {
                HistoryItem::ContextSummary { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");

        assert!(!combined.contains("folded-output-seq-2-stdout"));
        assert!(!combined.contains("folded-output-seq-2-stderr"));
    }

    #[test]
    fn restored_context_view_prompt_preserves_protected_context_and_hides_soft_deleted_blocks() {
        let large_stdout = "stdout-body-"
            .repeat((DEFAULT_FOLDED_OUTPUT_THRESHOLD_BYTES / "stdout-body-".len()) + 32);
        let large_stderr = "stderr-body-"
            .repeat((DEFAULT_FOLDED_OUTPUT_THRESHOLD_BYTES / "stderr-body-".len()) + 32);
        let records = vec![
            transcript_record(
                1,
                TranscriptEvent::SessionStarted {
                    model: "gpt-test".into(),
                },
            ),
            transcript_record(
                2,
                TranscriptEvent::UserMessage {
                    content: UserMessageContent::from(
                        "MUST keep raw transcript events append-only; do not purge requirements",
                    ),
                },
            ),
            transcript_record(
                3,
                TranscriptEvent::ContextNodeCreated {
                    node_id: "child".into(),
                    parent_node_id: Some("root".into()),
                    label: Some("Restored child".into()),
                    purpose: Some("Replay projected context tree".into()),
                    block_ref: None,
                    source_ref: None,
                },
            ),
            transcript_record(
                4,
                TranscriptEvent::ContextNodeLifecycle {
                    node_id: "root".into(),
                    status: ContextNodeStatus::Inactive,
                },
            ),
            transcript_record(
                5,
                TranscriptEvent::ContextNodeLifecycle {
                    node_id: "child".into(),
                    status: ContextNodeStatus::Active,
                },
            ),
            transcript_record(
                6,
                TranscriptEvent::ToolExecutionSummary(ToolExecutionSummaryEvent {
                    turn_id: 1,
                    call_id: "call-write".into(),
                    name: "fs__write".into(),
                    status: "executed".into(),
                    rejection: None,
                    effect_kind: "write".into(),
                    primary_path: Some("src/lib.rs".into()),
                    command: None,
                }),
            ),
            transcript_record(
                7,
                TranscriptEvent::PermissionDecision {
                    call_id: Some("call-shell".into()),
                    tool: "shell__exec".into(),
                    args: json!({"command": "cargo test --quiet"}),
                    allowed: false,
                    reason: Some("Denied from restored permission prompt".into()),
                },
            ),
            transcript_record(
                8,
                TranscriptEvent::Error {
                    message: "invariant violation: raw event missing".into(),
                },
            ),
            transcript_record(
                9,
                TranscriptEvent::ValidationAdvisory(ValidationAdvisory {
                    write_effects: 1,
                    validation_effects: 1,
                    failed_validation_effects: 1,
                    message: "cargo test failed".into(),
                }),
            ),
            transcript_record(
                10,
                TranscriptEvent::AssistantMessage {
                    content: "commit a270dda is current base".into(),
                },
            ),
            transcript_record(
                11,
                TranscriptEvent::AssistantMessage {
                    content: "soft archived note".into(),
                },
            ),
            transcript_record(
                12,
                TranscriptEvent::ContextViewOperationMetadata {
                    operation: "archive".into(),
                    node_id: None,
                    block_id: Some("block-seq-11-note".into()),
                    detail: None,
                },
            ),
            transcript_record(
                13,
                TranscriptEvent::AssistantMessage {
                    content: "soft removed note".into(),
                },
            ),
            transcript_record(
                14,
                TranscriptEvent::ContextViewOperationMetadata {
                    operation: "remove_from_view".into(),
                    node_id: None,
                    block_id: Some("block-seq-13-note".into()),
                    detail: None,
                },
            ),
            transcript_record(
                15,
                TranscriptEvent::ToolCallStarted {
                    call_id: "call-shell".into(),
                    name: "shell__exec".into(),
                    args: json!({"command": "cargo test --quiet"}),
                },
            ),
            transcript_record(
                16,
                TranscriptEvent::ToolCallFinished {
                    call_id: "call-shell".into(),
                    name: "shell__exec".into(),
                    ok: false,
                    output: ToolResult::err_with_data(
                        "shell__exec",
                        "command failed",
                        json!({
                            "status": 101,
                            "stdout": large_stdout,
                            "stdout_truncated": false,
                            "stderr": large_stderr,
                            "stderr_truncated": false,
                        }),
                    ),
                },
            ),
            transcript_record(
                17,
                TranscriptEvent::ContextSummaryArtifactMetadata {
                    node_id: "child".into(),
                    artifact_id: "summary-1".into(),
                    artifact_kind: "summary".into(),
                    version: Some(1),
                    summary: Some("child summary artifact".into()),
                    source_node_id: Some("child".into()),
                    source_block_id: Some("block-seq-10-note".into()),
                    source_start_sequence: Some(10),
                    source_end_sequence: Some(10),
                },
            ),
        ];
        let original_len = records.len();

        let snapshot =
            project_session_restore_snapshot("s".into(), records.clone(), None).expect("snapshot");
        let tree = project_context_tree(&records).expect("context tree");
        assert_eq!(tree.active_node_id().map(|id| id.as_str()), Some("child"));

        let projection = project_restored_context_view(&records).expect("context view");
        assert_eq!(
            projection
                .view_state
                .status(&ContextBlockId::new("block-seq-11-note").expect("archived block id")),
            Some(ContextViewStatus::Archived)
        );
        assert_eq!(
            projection
                .view_state
                .status(&ContextBlockId::new("block-seq-13-note").expect("removed block id")),
            Some(ContextViewStatus::RemovedFromView)
        );

        assert!(!snapshot.history.is_empty());
        let current_history = vec![HistoryItem::user("continue from restored context")];

        let result = build_request(RequestBuilderInput {
            protocol: ApiProtocol::Responses,
            model_id: "gpt-test",
            model: metadata(32_768),
            prelude: &[],
            history: &current_history,
            protected_start_index: 0,
            tools: &[],
            evidence: &snapshot.evidence,
            context_view: Some(&projection),
        })
        .expect("request builds from restored projection");
        let json = request_json(result);

        assert!(json.contains("[Context: Hard Context]"));
        assert!(
            json.contains("MUST keep raw transcript events append-only; do not purge requirements")
        );
        assert!(json.contains("Permission denied"));
        assert!(json.contains("src/lib.rs"));
        assert!(json.contains("cargo test failed"));
        assert!(json.contains("a270dda"));
        assert!(json.contains("invariant violation"));
        assert!(json.contains("[Context: Folded Outputs]"));
        assert!(json.contains("folded-output-seq-16-stdout"));
        assert!(json.contains("folded-output-seq-16-stderr"));
        assert!(json.contains("tool=shell__exec"));
        assert!(json.contains("stream=stdout"));
        assert!(json.contains("stream=stderr"));
        assert!(json.contains("command=cargo test --quiet"));
        assert!(!json.contains("soft archived note"));
        assert!(!json.contains("soft removed note"));
        assert!(!json.contains(&large_stdout));
        assert!(!json.contains(&large_stderr));
        assert_eq!(records.len(), original_len);
    }

    #[test]
    fn legacy_session_restore_builds_prompt_without_context_metadata() {
        let records = vec![
            transcript_record(
                1,
                TranscriptEvent::SessionStarted {
                    model: "gpt-test".into(),
                },
            ),
            transcript_record(
                2,
                TranscriptEvent::UserMessage {
                    content: UserMessageContent::from("legacy user one"),
                },
            ),
            transcript_record(
                3,
                TranscriptEvent::AssistantMessage {
                    content: "legacy assistant".into(),
                },
            ),
            transcript_record(
                4,
                TranscriptEvent::UserMessage {
                    content: UserMessageContent::from("legacy user two"),
                },
            ),
        ];

        let snapshot =
            project_session_restore_snapshot("s".into(), records.clone(), None).expect("snapshot");
        let tree = project_context_tree(&records).expect("legacy tree defaults to root");
        assert_eq!(tree.root_node_id().as_str(), "root");
        assert_eq!(tree.active_node_id().map(|id| id.as_str()), Some("root"));
        let projection = project_restored_context_view(&records).expect("legacy context view");

        let result = build_request(RequestBuilderInput {
            protocol: ApiProtocol::Responses,
            model_id: "gpt-test",
            model: metadata(8192),
            prelude: &[],
            history: &snapshot.history,
            protected_start_index: snapshot.history.len().saturating_sub(1),
            tools: &[],
            evidence: &snapshot.evidence,
            context_view: Some(&projection),
        })
        .expect("legacy request builds");
        let json = request_json(result);

        assert!(json.contains("legacy user one"));
        assert!(json.contains("legacy assistant"));
        assert!(json.contains("legacy user two"));
        assert!(!json.contains("context_node_created"));
        assert!(!json.contains("context_branch_created"));
    }
}
