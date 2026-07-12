use anyhow::Result;
#[path = "prompt_plan.rs"]
pub(crate) mod prompt_plan;
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
    MessageType, OutputStatus, PromptCacheRetention as OpenAiPromptCacheRetention, Reasoning,
    ReasoningEffort as OpenAiReasoningEffort, ReasoningSummary as ResponseReasoningSummary,
    ResponseTextParam, Role, TextResponseFormatConfiguration, Tool, Verbosity as ResponseVerbosity,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeSet;

use crate::config::{ApiProtocol, PromptCacheConfig, PromptCacheRetention};
use crate::context_view::{
    ContextBlock, ContextBlockKind, ContextBlockRetention, ContextBlockSource,
    ContextViewProjection, ContextViewStatus, FoldedOutputMetadata, ProtectedReason,
};
use crate::evidence::EvidenceRecord;
use crate::protocol_frames::{
    ProtocolFrame, ProtocolFrameItem, history_items_from_frames, history_items_to_frames,
    validate_history_items_complete,
};
pub use crate::protocol_frames::{
    ProtocolItem as HistoryItem, ProtocolToolCall as HistoryToolCall,
};
use crate::runtime_context::{
    FrameVisibility, RuntimeFrame, RuntimeFrameIdSeed, RuntimeFrameKind, RuntimeFrameProvenance,
    RuntimeSnapshot, RuntimeSource,
};
use crate::user_content::{UserImageAttachment, UserMessageContent, UserMessagePart};
use prompt_plan::{
    PlannedPrompt, PromptPlan, PromptPlanner, PromptPlannerInput, PromptSegmentContent,
    PromptSegmentRole,
};
#[cfg(test)]
use prompt_plan::{PromptPlanBuildInput, build_prompt_plan};

#[derive(Debug, Clone, PartialEq, Default)]
pub struct ModelRequestMetadata {
    pub context_window: Option<u64>,
    pub effective_input_limit_tokens: Option<u64>,
    pub max_output_tokens: Option<u64>,
    pub supports_tools: bool,
    pub supports_reasoning: bool,
    pub reasoning_effort: Option<ModelReasoningEffort>,
    pub reasoning_efforts: Vec<ModelReasoningEffort>,
    pub reasoning_summary: Option<ModelReasoningSummary>,
    pub text_verbosity: Option<ModelTextVerbosity>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub prompt_cache: PromptCacheConfig,
}

pub const DEFAULT_REASONING_EFFORTS: [ModelReasoningEffort; 6] = [
    ModelReasoningEffort::None,
    ModelReasoningEffort::Minimal,
    ModelReasoningEffort::Low,
    ModelReasoningEffort::Medium,
    ModelReasoningEffort::High,
    ModelReasoningEffort::Xhigh,
];

impl ModelRequestMetadata {
    pub fn context_window_tokens(&self) -> u64 {
        self.context_window
            .filter(|v| *v > 0)
            .unwrap_or(DEFAULT_FALLBACK_CONTEXT_WINDOW_TOKENS)
            .max(MIN_CONTEXT_WINDOW_TOKENS)
    }

    pub fn output_reserve_tokens(&self) -> u64 {
        self.max_output_tokens
            .filter(|v| *v > 0)
            .unwrap_or(DEFAULT_FALLBACK_OUTPUT_RESERVE_TOKENS)
            .max(MIN_OUTPUT_RESERVE_TOKENS)
    }

    pub fn effective_input_limit_tokens(&self) -> Option<u64> {
        self.effective_input_limit_tokens.filter(|v| *v > 0)
    }

    pub fn selectable_reasoning_efforts(&self) -> Vec<ModelReasoningEffort> {
        if !self.supports_reasoning {
            return Vec::new();
        }

        if self.reasoning_efforts.is_empty() {
            return DEFAULT_REASONING_EFFORTS.to_vec();
        }

        self.reasoning_efforts.clone()
    }

    pub fn allows_reasoning_effort(&self, effort: ModelReasoningEffort) -> bool {
        self.selectable_reasoning_efforts().contains(&effort)
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
    Max,
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

/// The producer-defined stability classification; never infer it from prompt text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PromptMessageOrigin {
    StaticPrelude,
    SkillCatalog,
    RuntimeClock,
    WorkflowTurn,
    UnreconciledSubagentContext,
    RuntimeContextView,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptMessage {
    pub role: PromptRole,
    pub text: String,
    #[serde(default = "default_prompt_message_origin")]
    pub origin: PromptMessageOrigin,
}

fn default_prompt_message_origin() -> PromptMessageOrigin {
    PromptMessageOrigin::StaticPrelude
}

impl PromptMessage {
    pub fn developer(text: impl Into<String>) -> Self {
        Self {
            role: PromptRole::Developer,
            text: text.into(),
            origin: PromptMessageOrigin::StaticPrelude,
        }
    }

    pub fn system(text: impl Into<String>) -> Self {
        Self {
            role: PromptRole::System,
            text: text.into(),
            origin: PromptMessageOrigin::StaticPrelude,
        }
    }

    pub fn developer_with_origin(text: impl Into<String>, origin: PromptMessageOrigin) -> Self {
        Self {
            role: PromptRole::Developer,
            text: text.into(),
            origin,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RequestBuilderInput<'a> {
    pub protocol: ApiProtocol,
    pub model_id: &'a str,
    pub model: ModelRequestMetadata,
    pub prelude: &'a [PromptMessage],
    pub snapshot: &'a RuntimeSnapshot,
    pub tools: &'a [ToolSpec],
}

/// Compatibility-only material accepted by tests and legacy callers. New
/// production paths must use [`RequestBuilderInput`] and a RuntimeSnapshot.
#[derive(Debug, Clone)]
pub struct LegacyRequestBuilderInput<'a> {
    pub protocol: ApiProtocol,
    pub model_id: &'a str,
    pub model: ModelRequestMetadata,
    pub prelude: &'a [PromptMessage],
    pub history: &'a [HistoryItem],
    pub protected_start_index: usize,
    pub tools: &'a [ToolSpec],
    pub evidence: &'a [EvidenceRecord],
    pub history_adapter: Option<&'a HistoryAdapterProjection>,
    pub context_view: Option<&'a ContextViewProjection>,
}

#[derive(Debug, Clone)]
pub(crate) struct SelectedPromptRequestInput<'a> {
    pub protocol: ApiProtocol,
    pub model_id: &'a str,
    pub model: ModelRequestMetadata,
    pub tools: &'a [ToolSpec],
    pub prompt_plan: PromptPlan,
    pub budget: BudgetReport,
    pub selected_evidence_ids: Vec<String>,
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
    pub estimated_required_fallback_tokens: u64,
    pub original_history_items: usize,
    pub retained_history_items: usize,
    pub dropped_history_items: usize,
    pub selected_evidence_items: usize,
    pub dropped_evidence_items: usize,
    pub truncated: bool,
    pub plan_total_prompt_tokens: u64,
    pub plan_stable_prompt_tokens: u64,
    pub plan_volatile_prompt_tokens: u64,
    pub plan_cacheable_prefix_tokens: u64,
    pub plan_stable_after_boundary_tokens: u64,
}

#[derive(Debug, Clone)]
pub enum BuiltRequest {
    Responses(CreateResponse),
    ResponsesCompatible(Value),
    Completions(CreateChatCompletionRequest),
    CompletionsCompatible(Value),
}

#[derive(Debug, Clone)]
pub struct BuildResult {
    pub request: BuiltRequest,
    pub budget: BudgetReport,
    #[allow(dead_code)]
    pub prompt_plan: PromptPlan,
    #[allow(dead_code)]
    pub selected_evidence_ids: Vec<String>,
    pub cache: PromptCacheReport,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PromptCacheReport {
    pub local_prefix_segments: usize,
    pub configured: bool,
    pub hint_serialized: bool,
    pub retention_sent: Option<PromptCacheRetention>,
    pub local_prefix_fingerprint: Option<String>,
    pub routing_key: Option<String>,
}

#[derive(Debug, Default)]
pub(crate) struct HistoryAdapterProjection {
    prelude: Vec<PromptMessage>,
    history_prefix: Vec<ProtocolFrame>,
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
    let PlannedPrompt {
        prompt_plan,
        budget,
        selected_evidence_ids,
    } = PromptPlanner::plan(PromptPlannerInput {
        protocol: input.protocol,
        model: input.model.clone(),
        model_id: input.model_id,
        prelude: input.prelude,
        snapshot: input.snapshot,
        tools: input.tools,
    })?;
    build_request_from_selected_prompt(SelectedPromptRequestInput {
        protocol: input.protocol,
        model_id: input.model_id,
        model: input.model,
        tools: input.tools,
        prompt_plan,
        budget,
        selected_evidence_ids,
    })
}

/// Compatibility seam for callers that have not yet materialized runtime authority.
/// It is deliberately the only path that accepts independent history, evidence, or
/// context-view projection material.
pub fn build_request_from_legacy(input: LegacyRequestBuilderInput<'_>) -> Result<BuildResult> {
    let mut prelude = input.prelude.to_vec();
    let mut frames = history_items_to_frames(input.history);
    let compatibility_adapter = input.context_view.map(|context_view| {
        context_view_history_adapter(context_view, input.history, input.protected_start_index)
    });
    let mut protected_start_index = input.protected_start_index;
    if let Some(sections) = input.history_adapter.or(compatibility_adapter.as_ref()) {
        prelude.extend(sections.prelude.clone());
        protected_start_index = protected_start_index.saturating_add(sections.history_prefix.len());
        frames.splice(0..0, sections.history_prefix.clone());
    }

    let mut snapshot = RuntimeSnapshot::new("legacy-request-builder");
    snapshot.set_evidence(input.evidence.to_vec());
    for (ordinal, frame) in frames.iter().enumerate() {
        let stable_key = frame.stable_prompt_key();
        let runtime_frame = RuntimeFrame::new(
            runtime_frame_kind(&frame.item),
            FrameVisibility::Active,
            crate::runtime_context::RuntimeFrameProvenance::new(RuntimeSource::Derived),
            RuntimeFrameIdSeed {
                frame_kind: runtime_frame_kind(&frame.item),
                source: RuntimeSource::Derived,
                ordinal: ordinal as u32,
                stable_key: &stable_key,
                source_span: None,
            },
        )
        .with_protocol(frame.item.clone());
        if ordinal >= protected_start_index {
            snapshot
                .compaction
                .protected_frame_ids
                .push(runtime_frame.id);
        }
        snapshot.push_frame(runtime_frame);
    }
    build_request(RequestBuilderInput {
        protocol: input.protocol,
        model_id: input.model_id,
        model: input.model,
        prelude: &prelude,
        snapshot: &snapshot,
        tools: input.tools,
    })
}

fn runtime_frame_kind(item: &ProtocolFrameItem) -> RuntimeFrameKind {
    match item {
        ProtocolFrameItem::ContextSummary { .. } => RuntimeFrameKind::Summary,
        ProtocolFrameItem::UserMessage { .. } => RuntimeFrameKind::User,
        ProtocolFrameItem::InternalContinuation { .. } => RuntimeFrameKind::Reasoning,
        ProtocolFrameItem::AssistantText { .. } => RuntimeFrameKind::Assistant,
        ProtocolFrameItem::AssistantToolCalls { .. } => RuntimeFrameKind::ToolCall,
        ProtocolFrameItem::ToolOutput { .. } => RuntimeFrameKind::ToolOutput,
    }
}

pub(crate) fn build_request_from_selected_prompt(
    mut input: SelectedPromptRequestInput<'_>,
) -> Result<BuildResult> {
    validate_prompt_plan_protocol(input.protocol, &input.prompt_plan)?;
    let plan_tokens = input.prompt_plan.token_report();
    input.budget.plan_total_prompt_tokens = plan_tokens.total_prompt_tokens;
    input.budget.plan_stable_prompt_tokens = plan_tokens.stable_prompt_tokens;
    input.budget.plan_volatile_prompt_tokens = plan_tokens.volatile_prompt_tokens;
    input.budget.plan_cacheable_prefix_tokens = plan_tokens.cacheable_prefix_tokens;
    input.budget.plan_stable_after_boundary_tokens = plan_tokens.stable_after_boundary_tokens;
    input.budget.estimated_request_tokens = plan_tokens
        .total_prompt_tokens
        .saturating_add(input.budget.estimated_tools_tokens);
    if input.budget.estimated_request_tokens
        > input
            .budget
            .input_budget_tokens
            .saturating_add(input.budget.estimated_tools_tokens)
    {
        anyhow::bail!("final prompt and tools exceed selected input budget");
    }
    let request = match input.protocol {
        ApiProtocol::Responses => {
            let request = build_responses_request(
                input.model_id,
                input.model.clone(),
                &input.prompt_plan,
                input.tools,
            );
            if input.model.reasoning_effort == Some(ModelReasoningEffort::Max) {
                let mut request = serde_json::to_value(request)
                    .expect("CreateResponse should always serialize to JSON");
                let fields = request
                    .as_object_mut()
                    .expect("CreateResponse should serialize to an object");
                let reasoning = fields
                    .entry("reasoning")
                    .or_insert_with(|| serde_json::json!({}));
                let reasoning = reasoning
                    .as_object_mut()
                    .expect("reasoning configuration should serialize as an object");
                reasoning.insert("effort".into(), Value::String("max".into()));
                BuiltRequest::ResponsesCompatible(request)
            } else {
                BuiltRequest::Responses(request)
            }
        }
        ApiProtocol::Completions => {
            let request = build_completions_request(
                input.model_id,
                input.model.clone(),
                &input.prompt_plan,
                input.tools,
            );
            if input.model.reasoning_effort == Some(ModelReasoningEffort::Max) {
                let mut request = serde_json::to_value(request)
                    .expect("CreateChatCompletionRequest should always serialize to JSON");
                request["reasoning_effort"] = Value::String("max".into());
                BuiltRequest::CompletionsCompatible(request)
            } else {
                BuiltRequest::Completions(request)
            }
        }
    };

    let cache = prompt_cache_report(
        input.protocol,
        input.model_id,
        &input.model.prompt_cache,
        &input.prompt_plan,
        input.tools,
        input.model.supports_tools,
    );
    Ok(BuildResult {
        request,
        budget: input.budget,
        prompt_plan: input.prompt_plan,
        selected_evidence_ids: input.selected_evidence_ids,
        cache,
    })
}

pub(crate) fn context_view_history_adapter(
    context_view: &ContextViewProjection,
    history: &[HistoryItem],
    protected_start_index: usize,
) -> HistoryAdapterProjection {
    let mut sections = HistoryAdapterProjection::default();
    let sorted_blocks = sorted_context_blocks(context_view);

    let protected_blocks = sorted_blocks
        .iter()
        .filter(|(id, block)| {
            !context_view.is_compacted(id) && block.is_protected() && !is_resolved(context_view, id)
        })
        .map(|(id, block)| (*id, *block))
        .collect::<Vec<_>>();
    if !protected_blocks.is_empty() {
        sections.prelude.push(PromptMessage::developer_with_origin(
            format!(
                "[Context: Hard Context]\n{}",
                protected_blocks
                    .iter()
                    .map(|(id, block)| format_protected_context_block_line(id, block))
                    .collect::<Vec<_>>()
                    .join("\n")
            ),
            PromptMessageOrigin::RuntimeContextView,
        ));
    }

    let protected_ids = protected_blocks
        .iter()
        .map(|(id, _)| id.as_str().to_string())
        .collect::<BTreeSet<_>>();
    let pinned_blocks = sorted_blocks
        .iter()
        .filter(|(id, _)| {
            !context_view.is_compacted(id)
                && is_pinned_visible(context_view, id)
                && !protected_ids.contains(id.as_str())
        })
        .map(|(id, block)| (*id, *block))
        .collect::<Vec<_>>();
    if !pinned_blocks.is_empty() {
        sections.prelude.push(PromptMessage::developer_with_origin(
            format!(
                "[Context: Pinned Context]\n{}",
                pinned_blocks
                    .iter()
                    .map(|(id, block)| format_context_block_line(id, block, false))
                    .collect::<Vec<_>>()
                    .join("\n")
            ),
            PromptMessageOrigin::RuntimeContextView,
        ));
    }

    if let Some(tail_section) = build_protected_tail_section(history, protected_start_index) {
        sections
            .history_prefix
            .push(ProtocolFrame::derived(ProtocolFrameItem::ContextSummary {
                text: tail_section,
            }));
    }

    let visible_index_blocks = sorted_blocks
        .iter()
        .filter(|(id, block)| include_in_context_index(context_view, id, block))
        .map(|(id, block)| format_context_block_line(id, block, false))
        .collect::<Vec<_>>();
    if !visible_index_blocks.is_empty() {
        sections
            .history_prefix
            .push(ProtocolFrame::derived(ProtocolFrameItem::ContextSummary {
                text: format!("[Context: Index]\n{}", visible_index_blocks.join("\n")),
            }));
    }

    let summaries = context_view
        .summary_artifacts
        .iter()
        .map(format_summary_artifact)
        .collect::<Vec<_>>();
    if !summaries.is_empty() {
        sections
            .history_prefix
            .push(ProtocolFrame::derived(ProtocolFrameItem::ContextSummary {
                text: format!("[Context: Summaries]\n{}", summaries.join("\n")),
            }));
    }

    let folded = sorted_context_blocks(context_view)
        .into_iter()
        .filter(|(id, block)| {
            !context_view.is_compacted(id)
                && block.folded_output_id.is_some()
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
            .push(ProtocolFrame::derived(ProtocolFrameItem::ContextSummary {
                text: format!("[Context: Folded Outputs]\n{}", folded.join("\n")),
            }));
    }

    if let Some(open_id) = context_view.view_state.open_detail_block_id()
        && let Some(block) = context_view.blocks.get(open_id)
        && !context_view.is_compacted(open_id)
        && view_status(context_view, open_id) != ContextViewStatus::RemovedFromView
        && !is_resolved(context_view, open_id)
    {
        sections
            .history_prefix
            .push(ProtocolFrame::derived(ProtocolFrameItem::ContextSummary {
                text: format!(
                    "[Context: Opened Details]\n{}\nDetail: {}",
                    format_context_block_line(open_id, block, false),
                    excerpt(&block.detail, 1200)
                ),
            }));
    }

    sections
}

/// Materializes provider-visible context solely from the runtime snapshot.
fn runtime_context_history_adapter(
    snapshot: &RuntimeSnapshot,
    history: &[HistoryItem],
    protected_start_index: usize,
) -> HistoryAdapterProjection {
    let mut sections = if snapshot.context_view == ContextViewProjection::default() {
        HistoryAdapterProjection::default()
    } else {
        context_view_history_adapter(&snapshot.context_view, history, protected_start_index)
    };
    for frame in &mut sections.history_prefix {
        frame.source_provenance = Some(RuntimeFrameProvenance::new(match &frame.item {
            ProtocolFrameItem::ContextSummary { text }
                if text.starts_with("[Context: Summaries]") =>
            {
                RuntimeSource::SummaryArtifact
            }
            ProtocolFrameItem::ContextSummary { text }
                if text.starts_with("[Context: Folded Outputs]") =>
            {
                RuntimeSource::FoldedOutput
            }
            _ => RuntimeSource::ContextView,
        }));
    }
    // A live snapshot can carry materialized context frames before a view
    // projection exists (for example during a runtime rebuild). Render those
    // frames directly rather than silently losing provider-visible context.
    if snapshot.context_view == ContextViewProjection::default() {
        for frame in snapshot.frames.iter().filter(|frame| {
            frame_is_provider_visible(snapshot, frame)
                && frame.protocol.is_none()
                && matches!(
                    frame.kind,
                    RuntimeFrameKind::ContextBlock | RuntimeFrameKind::Summary
                )
        }) {
            let Some(summary) = frame.summary.as_deref() else {
                continue;
            };
            sections.history_prefix.push(ProtocolFrame {
                runtime_frame_id: Some(frame.id),
                source_provenance: Some(frame.provenance.clone()),
                history_index: usize::MAX,
                item: ProtocolFrameItem::ContextSummary {
                    text: format!("[Context: Runtime Material]\n{summary}"),
                },
            });
        }
        for folded in snapshot.folded_outputs.iter().filter(|folded| {
            folded.source_span.is_none_or(|span| {
                !snapshot
                    .compaction
                    .retired_source_spans
                    .iter()
                    .any(|retired| retired.overlaps(span))
            })
        }) {
            let mut provenance = RuntimeFrameProvenance::new(RuntimeSource::FoldedOutput);
            provenance.source_id = Some(folded.output_id.clone());
            sections.history_prefix.push(ProtocolFrame {
                runtime_frame_id: None,
                source_provenance: Some(provenance),
                history_index: usize::MAX,
                item: ProtocolFrameItem::ContextSummary {
                    text: format!(
                        "[Context: Folded Outputs]\n- output_id={} tool={} call_id={}",
                        folded.output_id,
                        folded.tool_name.as_deref().unwrap_or("-"),
                        folded.call_id.as_deref().unwrap_or("-")
                    ),
                },
            });
        }
    }
    // Standard projection contributors are represented by the dedicated sections
    // above. Everything else is the generic provider-visible contributor channel.
    for contributor in snapshot.prompt_contributors.iter().filter(|contributor| {
        !matches!(
            contributor.contributor_id.as_str(),
            "context-view-active"
                | "evidence"
                | "summary-artifacts"
                | "folded-outputs"
                | "child-sessions"
        ) && contributor.kind != crate::runtime_context::PromptContributorKind::SkillMaterial
    }) {
        if contributor.provenance.source_span.is_some_and(|span| {
            snapshot
                .compaction
                .retired_source_spans
                .iter()
                .any(|retired| retired.overlaps(span))
        }) {
            continue;
        }
        let text = contributor
            .frame_ids
            .iter()
            .filter_map(|id| {
                snapshot
                    .frames
                    .iter()
                    .find(|frame| frame.id == *id)
                    .filter(|frame| frame_is_provider_visible(snapshot, frame))
                    .and_then(|frame| frame.summary.as_deref())
            })
            .collect::<Vec<_>>()
            .join("\n");
        if text.is_empty() {
            continue;
        }
        sections.history_prefix.push(ProtocolFrame {
            runtime_frame_id: None,
            source_provenance: Some(contributor.provenance.clone()),
            history_index: usize::MAX,
            item: ProtocolFrameItem::ContextSummary {
                text: format!(
                    "[Context: {}]\n{}",
                    contributor
                        .label
                        .as_deref()
                        .unwrap_or(&contributor.contributor_id),
                    text
                ),
            },
        });
    }
    sections
}

fn provider_visible_protocol_frames(snapshot: &RuntimeSnapshot) -> Vec<ProtocolFrame> {
    snapshot
        .frames
        .iter()
        .filter(|frame| frame_is_provider_visible(snapshot, frame))
        .filter_map(|frame| {
            frame.protocol.clone().map(|item| ProtocolFrame {
                runtime_frame_id: Some(frame.id),
                source_provenance: Some(frame.provenance.clone()),
                history_index: 0,
                item,
            })
        })
        .enumerate()
        .map(|(history_index, mut frame)| {
            frame.history_index = history_index;
            frame
        })
        .collect()
}

fn frame_is_provider_visible(snapshot: &RuntimeSnapshot, frame: &RuntimeFrame) -> bool {
    frame.visibility == FrameVisibility::Active
        && !snapshot.compaction.compacted_frame_ids.contains(&frame.id)
        && frame.provenance.source_span.is_none_or(|span| {
            !snapshot
                .compaction
                .retired_source_spans
                .iter()
                .any(|retired| retired.overlaps(span))
        })
}

fn protected_start_index_for_snapshot(
    snapshot: &RuntimeSnapshot,
    frames: &[ProtocolFrame],
) -> usize {
    frames
        .iter()
        .position(|frame| {
            frame
                .runtime_frame_id
                .is_some_and(|id| snapshot.compaction.protected_frame_ids.contains(&id))
        })
        .unwrap_or(frames.len())
}

fn assemble_context_view_sections(
    context_view: &ContextViewProjection,
    history: &[HistoryItem],
    protected_start_index: usize,
) -> HistoryAdapterProjection {
    context_view_history_adapter(context_view, history, protected_start_index)
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
    if context_view.is_compacted(block_id) {
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
    history: &[ProtocolFrame],
    protected_start_index: usize,
    model: ModelRequestMetadata,
    tools: &[ToolSpec],
    evidence_budget: EvidenceBudgetReport,
    required_fallback_tokens: u64,
) -> (Vec<ProtocolFrame>, BudgetReport) {
    let history_len = history.len();
    let protected_start = protected_start_index.min(history_len);
    let (older, protected) = history.split_at(protected_start);

    let prelude_tokens = estimate_prelude_tokens(prelude);
    let protected_tokens = estimate_protocol_frame_tokens(protected);
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
        .saturating_add(evidence_budget.estimated_evidence_tokens)
        .saturating_add(required_fallback_tokens);

    if fixed_tokens < input_budget {
        for unit in retention_units(older).into_iter().rev() {
            let cost = estimate_protocol_frame_tokens(unit);
            let next = fixed_tokens
                .saturating_add(retained_older_tokens)
                .saturating_add(cost);
            if next > input_budget {
                break;
            }
            retained_older.extend(unit.iter().cloned().rev());
            retained_older_tokens = retained_older_tokens.saturating_add(cost);
        }
        retained_older.reverse();
    }

    let mut retained = Vec::with_capacity(retained_older.len() + protected.len());
    retained.extend(retained_older.iter().cloned());
    retained.extend(protected.iter().cloned());
    let retained_history_items = retained.len();
    let dropped_history_items = history_len.saturating_sub(retained_history_items);
    let retained_tokens = estimate_protocol_frame_tokens(&retained);
    let estimated_request_tokens = prelude_tokens
        .saturating_add(evidence_budget.estimated_evidence_tokens)
        .saturating_add(required_fallback_tokens)
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
            estimated_required_fallback_tokens: required_fallback_tokens,
            original_history_items: history_len,
            retained_history_items,
            dropped_history_items,
            selected_evidence_items: evidence_budget.selected_evidence_items,
            dropped_evidence_items: evidence_budget.dropped_evidence_items,
            truncated: dropped_history_items > 0,
            plan_total_prompt_tokens: 0,
            plan_stable_prompt_tokens: 0,
            plan_volatile_prompt_tokens: 0,
            plan_cacheable_prefix_tokens: 0,
            plan_stable_after_boundary_tokens: 0,
        },
    )
}

/// A tool-call batch and all of its outputs are retained atomically.
fn retention_units(frames: &[ProtocolFrame]) -> Vec<&[ProtocolFrame]> {
    let transcript = validate_history_items_complete(&history_items_from_frames(frames), None)
        .expect("history was validated before retention");
    let mut group_end_by_start = std::collections::BTreeMap::new();
    for group in transcript.tool_call_groups {
        let end = group
            .tool_output_indexes
            .iter()
            .copied()
            .max()
            .unwrap_or(group.assistant_index);
        group_end_by_start.insert(group.assistant_index, end);
    }
    let mut units = Vec::new();
    let mut index = 0;
    while index < frames.len() {
        let end = group_end_by_start.get(&index).copied().unwrap_or(index);
        units.push(&frames[index..=end]);
        index = end + 1;
    }
    units
}

fn expand_protected_start_to_group(
    history: &[HistoryItem],
    protected_start: usize,
) -> Result<usize> {
    let transcript = validate_history_items_complete(history, Some(protected_start))?;
    Ok(transcript
        .tool_call_groups
        .iter()
        .fold(protected_start, |start, group| {
            let group_end = group
                .tool_output_indexes
                .iter()
                .copied()
                .max()
                .unwrap_or(group.assistant_index);
            if group.assistant_index < start && group_end >= start {
                group.assistant_index
            } else {
                start
            }
        }))
}

fn estimate_protocol_frame_tokens(frames: &[ProtocolFrame]) -> u64 {
    frames
        .iter()
        .map(|frame| estimate_history_item_tokens(&frame.to_history_item()))
        .sum()
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
    );
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
        reasoning: response_reasoning(model.clone()),
        temperature: model.temperature,
        text: response_text(model.clone()),
        tools,
        parallel_tool_calls,
        top_p: model.top_p,
        prompt_cache_key: cache.key,
        prompt_cache_retention: cache.retention.map(openai_cache_retention),
        ..Default::default()
    }
}

struct CacheRequestFields {
    key: Option<String>,
    retention: Option<PromptCacheRetention>,
}

fn cache_request_fields(
    protocol: ApiProtocol,
    model_id: &str,
    config: &PromptCacheConfig,
    plan: &PromptPlan,
    tools: &[ToolSpec],
    supports_tools: bool,
) -> CacheRequestFields {
    if !config.enabled || plan.cacheable_prefix_len() == 0 {
        return CacheRequestFields {
            key: None,
            retention: None,
        };
    }
    let namespace = config
        .namespace
        .as_deref()
        .expect("enabled prompt cache has normalized namespace");
    let key = routing_key(namespace, protocol, model_id, tools, supports_tools);
    CacheRequestFields {
        key: Some(key),
        retention: (protocol == ApiProtocol::Responses)
            .then_some(config.retention)
            .flatten(),
    }
}

fn prompt_cache_report(
    protocol: ApiProtocol,
    model_id: &str,
    config: &PromptCacheConfig,
    plan: &PromptPlan,
    tools: &[ToolSpec],
    supports_tools: bool,
) -> PromptCacheReport {
    let prefix = plan.cacheable_prefix_len();
    if !config.enabled || prefix == 0 {
        return PromptCacheReport {
            local_prefix_segments: prefix,
            configured: config.enabled,
            ..Default::default()
        };
    }
    let namespace = config
        .namespace
        .as_deref()
        .expect("enabled prompt cache has normalized namespace");
    let canonical_input = canonical_cache_input(
        namespace,
        protocol,
        model_id,
        &plan.segments[..prefix],
        tools,
        supports_tools,
    );
    let routing_key = routing_key_from_canonical_input(&canonical_input);
    let local_prefix_fingerprint =
        format!("ppf-v1-{}", sha256_hex(&canonical_bytes(&canonical_input)));
    PromptCacheReport {
        local_prefix_segments: prefix,
        configured: true,
        hint_serialized: true,
        retention_sent: if protocol == ApiProtocol::Responses {
            config.retention
        } else {
            None
        },
        local_prefix_fingerprint: Some(local_prefix_fingerprint),
        routing_key: Some(routing_key),
    }
}

fn routing_key(
    namespace: &str,
    protocol: ApiProtocol,
    model_id: &str,
    tools: &[ToolSpec],
    supports_tools: bool,
) -> String {
    let canonical_input =
        canonical_cache_input(namespace, protocol, model_id, &[], tools, supports_tools);
    routing_key_from_canonical_input(&canonical_input)
}

/// Provider-visible cache identity. Values are serialized through the same
/// protocol conversion helpers used to construct the final request.
pub(crate) fn canonical_cache_input(
    namespace: &str,
    protocol: ApiProtocol,
    model_id: &str,
    prefix: &[prompt_plan::PromptSegment],
    tools: &[ToolSpec],
    supports_tools: bool,
) -> Value {
    let (items, provider_tools, parallel_tool_calls) = match protocol {
        ApiProtocol::Responses => (
            serde_json::to_value(
                prefix
                    .iter()
                    .flat_map(prompt_segment_to_response_inputs)
                    .collect::<Vec<_>>(),
            )
            .expect("response input is serializable"),
            serde_json::to_value(
                supports_tools.then(|| tools.iter().map(tool_to_response_tool).collect::<Vec<_>>()),
            )
            .expect("response tools are serializable"),
            serde_json::to_value(supports_tools.then_some(false))
                .expect("parallel tool calls is serializable"),
        ),
        ApiProtocol::Completions => (
            serde_json::to_value(
                prefix
                    .iter()
                    .map(prompt_segment_to_chat_message)
                    .collect::<Vec<_>>(),
            )
            .expect("chat messages are serializable"),
            serde_json::to_value(
                supports_tools.then(|| tools.iter().map(tool_to_chat_tool).collect::<Vec<_>>()),
            )
            .expect("chat tools are serializable"),
            serde_json::to_value(supports_tools.then_some(false))
                .expect("parallel tool calls is serializable"),
        ),
    };
    serde_json::json!({
        "namespace": namespace,
        "shape_version": 1,
        "protocol": protocol,
        "model": model_id,
        "items": items,
        "tools": provider_tools,
        "input_shape": { "parallel_tool_calls": parallel_tool_calls },
    })
}

fn routing_key_from_canonical_input(input: &Value) -> String {
    let Value::Object(values) = input else {
        unreachable!("canonical cache input is an object");
    };
    let routing_input = serde_json::json!({
        "namespace": values["namespace"],
        "shape_version": values["shape_version"],
        "protocol": values["protocol"],
        "model": values["model"],
        "tools": values["tools"],
        "input_shape": values["input_shape"],
    });
    format!(
        "lc-pc-v1-{}",
        &sha256_hex(&canonical_bytes(&routing_input))[..32]
    )
}

fn openai_cache_retention(value: PromptCacheRetention) -> OpenAiPromptCacheRetention {
    match value {
        PromptCacheRetention::InMemory => OpenAiPromptCacheRetention::InMemory,
        PromptCacheRetention::TwentyFourHours => OpenAiPromptCacheRetention::Hours24,
    }
}

fn canonical_bytes(value: &Value) -> Vec<u8> {
    fn append(out: &mut Vec<u8>, tag: u8, bytes: &[u8]) {
        out.push(tag);
        out.extend_from_slice(&(bytes.len() as u64).to_be_bytes());
        out.extend_from_slice(bytes);
    }
    fn visit(out: &mut Vec<u8>, value: &Value) {
        match value {
            Value::Null => append(out, b'n', b""),
            Value::Bool(value) => append(out, b'b', if *value { b"1" } else { b"0" }),
            Value::Number(value) => append(out, b'#', value.to_string().as_bytes()),
            Value::String(value) => append(out, b's', value.as_bytes()),
            Value::Array(values) => {
                append(out, b'[', &(values.len() as u64).to_be_bytes());
                for value in values {
                    visit(out, value);
                }
            }
            Value::Object(values) => {
                append(out, b'{', &(values.len() as u64).to_be_bytes());
                let mut keys = values.keys().collect::<Vec<_>>();
                keys.sort();
                for key in keys {
                    append(out, b'k', key.as_bytes());
                    visit(out, &values[key]);
                }
            }
        }
    }
    let mut out = Vec::new();
    visit(&mut out, value);
    out
}

/// Small local SHA-256 implementation to keep fingerprinting dependency-free.
fn sha256_hex(input: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut data = input.to_vec();
    let bits = (data.len() as u64) * 8;
    data.push(0x80);
    while data.len() % 64 != 56 {
        data.push(0);
    }
    data.extend_from_slice(&bits.to_be_bytes());
    let mut h = [
        0x6a09e667u32,
        0xbb67ae85,
        0x3c6ef372,
        0xa54ff53a,
        0x510e527f,
        0x9b05688c,
        0x1f83d9ab,
        0x5be0cd19,
    ];
    for chunk in data.chunks_exact(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes(chunk[i * 4..i * 4 + 4].try_into().unwrap());
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh) =
            (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (x, y) in h.iter_mut().zip([a, b, c, d, e, f, g, hh]) {
            *x = x.wrapping_add(y);
        }
    }
    h.iter().map(|word| format!("{word:08x}")).collect()
}

fn response_reasoning(model: ModelRequestMetadata) -> Option<Reasoning> {
    if !model.supports_reasoning {
        return None;
    }
    let effort = model
        .reasoning_effort
        .filter(|effort| *effort != ModelReasoningEffort::Max)
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

fn prelude_to_response_input(message: PromptMessage) -> InputItem {
    let role = match message.role {
        PromptRole::System => Role::System,
        PromptRole::Developer => Role::Developer,
    };
    response_text_message(role, message.text)
}

fn prompt_segment_to_response_inputs(segment: &prompt_plan::PromptSegment) -> Vec<InputItem> {
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

fn prompt_segment_to_chat_message(
    segment: &prompt_plan::PromptSegment,
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

fn validate_prompt_plan_protocol(protocol: ApiProtocol, prompt_plan: &PromptPlan) -> Result<()> {
    if prompt_plan.protocol != protocol {
        anyhow::bail!(
            "selected prompt plan protocol mismatch: request={protocol:?} prompt_plan={:?}",
            prompt_plan.protocol
        );
    }
    Ok(())
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
        HistoryItem::AssistantToolCalls { text, calls } => {
            let mut input = text
                .filter(|text| !text.is_empty())
                .map(|text| vec![response_text_message(Role::Assistant, text)])
                .unwrap_or_default();
            input.extend(
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
                    .collect::<Vec<_>>(),
            );
            input
        }
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
    );
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
            .filter(|effort| *effort != ModelReasoningEffort::Max)
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
        ModelReasoningEffort::Max => unreachable!("max is serialized through a compatible request"),
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

    #[test]
    fn history_aliases_share_protocol_item_type_and_json_shape() {
        let legacy = HistoryItem::AssistantToolCalls {
            text: Some("working".into()),
            calls: vec![HistoryToolCall {
                call_id: "call-1".into(),
                name: "fs__read".into(),
                arguments_json: r#"{"path":"src/main.rs"}"#.into(),
            }],
        };
        let canonical: crate::protocol_frames::ProtocolItem = legacy.clone();
        let legacy_again: HistoryItem = canonical.clone();

        assert_eq!(legacy_again, legacy);
        assert_eq!(
            serde_json::to_value(&legacy).expect("legacy item serializes"),
            serde_json::to_value(&canonical).expect("canonical item serializes")
        );
        assert_eq!(
            serde_json::to_string(&canonical).expect("canonical item serializes"),
            r#"{"kind":"assistant_tool_calls","text":"working","calls":[{"call_id":"call-1","name":"fs__read","arguments_json":"{\"path\":\"src/main.rs\"}"}]}"#
        );
    }

    #[test]
    fn sha256_matches_nist_vectors_and_canonical_json_is_key_order_independent() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            canonical_bytes(&serde_json::json!({"b": [2, 1], "a": {"z": true, "x": null}})),
            canonical_bytes(&serde_json::json!({"a": {"x": null, "z": true}, "b": [2, 1]}))
        );
    }

    #[test]
    fn canonical_cache_input_uses_exact_serialized_protocol_shape() {
        let plan = build_prompt_plan(PromptPlanBuildInput {
            protocol: ApiProtocol::Responses,
            model_id: "gpt-test",
            prelude: &[PromptMessage::system("stable")],
            snapshot: &RuntimeSnapshot::new("canonical-cache-test"),
            selected_frames: &[],
            protected_suffix_len: 0,
            evidence_message: None,
            selected_evidence_ids: &[],
        });
        let tools = [ToolSpec {
            name: "read".into(),
            description: "Read a file".into(),
            parameters: json!({"type": "object", "properties": {"path": {"type": "string"}}}),
            strict: true,
        }];
        let model = ModelRequestMetadata {
            supports_tools: true,
            prompt_cache: PromptCacheConfig {
                enabled: true,
                retention: None,
                namespace: Some("test".into()),
            },
            ..Default::default()
        };

        let responses_request = build_responses_request("gpt-test", model.clone(), &plan, &tools);
        assert_eq!(responses_request.parallel_tool_calls, Some(false));
        let responses =
            serde_json::to_value(responses_request).expect("responses request serializes");
        let responses_canonical = canonical_cache_input(
            "test",
            ApiProtocol::Responses,
            "gpt-test",
            &plan.segments,
            &tools,
            true,
        );
        assert_eq!(responses_canonical["items"], responses["input"]);
        assert_eq!(responses_canonical["tools"], responses["tools"]);
        assert_eq!(
            responses_canonical["input_shape"]["parallel_tool_calls"],
            false
        );
        assert_eq!(responses["parallel_tool_calls"], false);

        let chat_plan = PromptPlan {
            protocol: ApiProtocol::Completions,
            ..plan.clone()
        };
        let chat_request = build_completions_request("gpt-test", model, &chat_plan, &tools);
        assert_eq!(chat_request.parallel_tool_calls, Some(false));
        let chat = serde_json::to_value(chat_request).expect("chat request serializes");
        let chat_canonical = canonical_cache_input(
            "test",
            ApiProtocol::Completions,
            "gpt-test",
            &chat_plan.segments,
            &tools,
            true,
        );
        assert_eq!(chat_canonical["items"], chat["messages"]);
        assert_eq!(chat_canonical["tools"], chat["tools"]);
        assert_eq!(chat_canonical["input_shape"]["parallel_tool_calls"], false);
        assert_eq!(chat["parallel_tool_calls"], false);
    }
    use crate::agent::{ToolExecutionSummaryEvent, ValidationAdvisory};
    use crate::context_tree::ContextNodeStatus;
    use crate::context_view::{
        ContextBlockId, ContextViewStatus, DEFAULT_FOLDED_OUTPUT_THRESHOLD_BYTES,
        project_context_view,
    };
    use crate::evidence::{EvidenceKind, EvidenceRecord, EvidenceSource};
    use crate::protocol_frames::history_items_from_frames;
    use crate::runtime_context::RuntimeChildSession;
    use crate::tool::ToolResult;
    use crate::transcript::transcript_projection::{
        project_context_tree, project_context_view as project_restored_context_view,
        project_session_restore_snapshot, restore_session_history_projection,
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

    fn adapter_summary_texts(adapter: &HistoryAdapterProjection) -> Vec<String> {
        history_items_from_frames(&adapter.history_prefix)
            .into_iter()
            .filter_map(|item| match item {
                HistoryItem::ContextSummary { text } => Some(text),
                _ => None,
            })
            .collect()
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
            BuiltRequest::ResponsesCompatible(request)
            | BuiltRequest::CompletionsCompatible(request) => {
                serde_json::to_string(&request).expect("serialize")
            }
            BuiltRequest::Completions(request) => {
                serde_json::to_string(&request).expect("serialize")
            }
        }
    }

    fn cache_config(retention: Option<PromptCacheRetention>) -> PromptCacheConfig {
        PromptCacheConfig {
            enabled: true,
            retention,
            namespace: Some("cache-test".into()),
        }
    }

    fn cache_test_result(
        protocol: ApiProtocol,
        prompt_cache: PromptCacheConfig,
        tools: &[ToolSpec],
    ) -> BuildResult {
        let mut model = metadata(8192);
        model.prompt_cache = prompt_cache;
        build_request_from_legacy(LegacyRequestBuilderInput {
            protocol,
            model_id: "cache-model",
            model,
            prelude: &[PromptMessage::system("stable instructions")],
            history: &[
                HistoryItem::assistant("prior answer"),
                HistoryItem::user("current question"),
            ],
            protected_start_index: 1,
            tools,
            evidence: &[],
            history_adapter: None,
            context_view: None,
        })
        .expect("cache test request builds")
    }

    #[test]
    fn configured_reasoning_efforts_constrain_selectable_levels() {
        let metadata = ModelRequestMetadata {
            supports_reasoning: true,
            reasoning_efforts: vec![
                ModelReasoningEffort::None,
                ModelReasoningEffort::Low,
                ModelReasoningEffort::Max,
            ],
            ..Default::default()
        };

        assert!(metadata.allows_reasoning_effort(ModelReasoningEffort::Low));
        assert!(metadata.allows_reasoning_effort(ModelReasoningEffort::Max));
        assert!(!metadata.allows_reasoning_effort(ModelReasoningEffort::High));
    }

    fn request_value(result: &BuildResult) -> Value {
        match &result.request {
            BuiltRequest::Responses(request) => serde_json::to_value(request),
            BuiltRequest::ResponsesCompatible(request)
            | BuiltRequest::CompletionsCompatible(request) => Ok(request.clone()),
            BuiltRequest::Completions(request) => serde_json::to_value(request),
        }
        .expect("request serializes")
    }

    fn without_cache_fields(mut request: Value) -> Value {
        let fields = request
            .as_object_mut()
            .expect("serialized request is an object");
        fields.remove("prompt_cache_key");
        fields.remove("prompt_cache_retention");
        request
    }

    #[test]
    fn prompt_cache_serialization_and_omission_follow_protocol_and_prefix() {
        for (protocol, retention, expected_retention) in [
            (
                ApiProtocol::Responses,
                Some(PromptCacheRetention::InMemory),
                Some("in_memory"),
            ),
            (
                ApiProtocol::Responses,
                Some(PromptCacheRetention::TwentyFourHours),
                Some("24h"),
            ),
            (ApiProtocol::Responses, None, None),
            (
                ApiProtocol::Completions,
                Some(PromptCacheRetention::InMemory),
                None,
            ),
        ] {
            let result = cache_test_result(protocol, cache_config(retention), &[]);
            let request = request_value(&result);
            let key = request["prompt_cache_key"]
                .as_str()
                .expect("enabled stable cache serializes a key");
            assert!(key.starts_with("lc-pc-v1-"));
            assert_eq!(key.len(), 41);
            assert!(key.bytes().all(|byte| byte.is_ascii()));
            assert!(
                key[9..]
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            );
            match expected_retention {
                Some(retention) => assert_eq!(request["prompt_cache_retention"], retention),
                None => assert!(request.get("prompt_cache_retention").is_none()),
            }
            assert_eq!(result.cache.hint_serialized, true);
            assert_eq!(
                result.cache.retention_sent,
                retention.filter(|_| protocol == ApiProtocol::Responses)
            );
        }

        let disabled = cache_test_result(ApiProtocol::Responses, PromptCacheConfig::default(), &[]);
        let disabled_request = request_value(&disabled);
        assert!(disabled_request.get("prompt_cache_key").is_none());
        assert!(disabled_request.get("prompt_cache_retention").is_none());
        assert!(disabled.cache.local_prefix_segments > 0);
        assert!(!disabled.cache.configured);
        assert!(!disabled.cache.hint_serialized);
        assert_eq!(disabled.cache.retention_sent, None);
        assert_eq!(disabled.cache.local_prefix_fingerprint, None);
        assert_eq!(disabled.cache.routing_key, None);

        let mut model = metadata(8192);
        model.prompt_cache = cache_config(Some(PromptCacheRetention::InMemory));
        let no_prefix = build_request_from_legacy(LegacyRequestBuilderInput {
            protocol: ApiProtocol::Responses,
            model_id: "cache-model",
            model,
            prelude: &[],
            history: &[HistoryItem::user("current question")],
            protected_start_index: 0,
            tools: &[],
            evidence: &[],
            history_adapter: None,
            context_view: None,
        })
        .expect("zero-prefix request builds");
        let no_prefix_request = request_value(&no_prefix);
        assert!(no_prefix_request.get("prompt_cache_key").is_none());
        assert!(no_prefix_request.get("prompt_cache_retention").is_none());
        assert_eq!(no_prefix.cache.local_prefix_segments, 0);
        assert!(no_prefix.cache.configured);
        assert!(!no_prefix.cache.hint_serialized);
        assert_eq!(no_prefix.cache.local_prefix_fingerprint, None);
        assert_eq!(no_prefix.cache.routing_key, None);
    }

    #[test]
    fn prompt_cache_is_a_provider_noop_and_budget_reports_match_final_plan() {
        let tools = [
            ToolSpec {
                name: "read".into(),
                description: "Read a file".into(),
                parameters: json!({"type": "object", "properties": {"path": {"type": "string"}}}),
                strict: true,
            },
            ToolSpec {
                name: "write".into(),
                description: "Write a file".into(),
                parameters: json!({"type": "object", "properties": {"path": {"type": "string"}}}),
                strict: true,
            },
        ];
        for protocol in [ApiProtocol::Responses, ApiProtocol::Completions] {
            let disabled = cache_test_result(protocol, PromptCacheConfig::default(), &tools);
            let enabled = cache_test_result(protocol, cache_config(None), &tools);

            assert_eq!(
                without_cache_fields(request_value(&disabled)),
                without_cache_fields(request_value(&enabled)),
                "cache controls must not change provider content"
            );
            assert_eq!(disabled.prompt_plan, enabled.prompt_plan);
            assert_eq!(
                disabled.selected_evidence_ids,
                enabled.selected_evidence_ids
            );
            assert_eq!(disabled.budget, enabled.budget);

            let report = enabled.prompt_plan.token_report();
            assert_eq!(
                enabled.budget.plan_total_prompt_tokens,
                report.total_prompt_tokens
            );
            assert_eq!(
                enabled.budget.plan_stable_prompt_tokens,
                report.stable_prompt_tokens
            );
            assert_eq!(
                enabled.budget.plan_volatile_prompt_tokens,
                report.volatile_prompt_tokens
            );
            assert_eq!(
                enabled.budget.plan_cacheable_prefix_tokens,
                report.cacheable_prefix_tokens
            );
            assert_eq!(
                enabled.budget.plan_stable_after_boundary_tokens,
                report.stable_after_boundary_tokens
            );
            assert_eq!(
                enabled.budget.estimated_request_tokens,
                report.total_prompt_tokens + enabled.budget.estimated_tools_tokens
            );
            assert_eq!(
                enabled.budget.estimated_tools_tokens,
                estimate_tools_tokens(&tools)
            );
            assert_eq!(
                enabled.budget.input_budget_tokens,
                effective_input_budget_tokens(enabled_model_metadata(), &tools)
            );
            assert_eq!(
                enabled.cache.local_prefix_segments,
                enabled.prompt_plan.cacheable_prefix_len()
            );
            let namespace = "cache-test";
            let canonical = canonical_cache_input(
                namespace,
                protocol,
                "cache-model",
                &enabled.prompt_plan.segments[..enabled.cache.local_prefix_segments],
                &tools,
                true,
            );
            let request = request_value(&enabled);
            let rendered = match protocol {
                ApiProtocol::Responses => &request["input"],
                ApiProtocol::Completions => &request["messages"],
            };
            assert_eq!(
                canonical["items"],
                Value::Array(
                    rendered.as_array().expect("request items")
                        [..enabled.cache.local_prefix_segments]
                        .to_vec()
                )
            );
        }
    }

    fn enabled_model_metadata() -> ModelRequestMetadata {
        metadata(8192)
    }

    #[test]
    fn prompt_cache_fingerprints_and_routing_keys_follow_identity_boundaries() {
        let tools = [
            ToolSpec {
                name: "read".into(),
                description: "Read".into(),
                parameters: json!({"type": "object"}),
                strict: true,
            },
            ToolSpec {
                name: "write".into(),
                description: "Write".into(),
                parameters: json!({"type": "object"}),
                strict: true,
            },
        ];
        let base = cache_test_result(ApiProtocol::Responses, cache_config(None), &tools);
        let base_report = base.cache.clone();
        let report = |namespace: &str,
                      protocol: ApiProtocol,
                      model: &str,
                      plan: &PromptPlan,
                      tool_specs: &[ToolSpec],
                      supports_tools: bool| {
            prompt_cache_report(
                protocol,
                model,
                &PromptCacheConfig {
                    enabled: true,
                    retention: None,
                    namespace: Some(namespace.into()),
                },
                plan,
                tool_specs,
                supports_tools,
            )
        };
        let base_again = report(
            "cache-test",
            ApiProtocol::Responses,
            "cache-model",
            &base.prompt_plan,
            &tools,
            true,
        );
        assert_eq!(base_report, base_again);
        assert!(
            base_report
                .local_prefix_fingerprint
                .as_deref()
                .is_some_and(|fingerprint| fingerprint.starts_with("ppf-v1-"))
        );

        let mut changed_stable = base.prompt_plan.clone();
        changed_stable.segments[0].text = "changed stable instructions".into();
        changed_stable.segments[0].content = PromptSegmentContent::Text {
            text: "changed stable instructions".into(),
        };
        let mut changed_role = base.prompt_plan.clone();
        changed_role.segments[0].role = PromptSegmentRole::Developer;
        let mut changed_schema = tools.clone();
        changed_schema[0].parameters = json!({"type": "object", "additionalProperties": false});
        let mut changed_image = base.prompt_plan.clone();
        changed_image.segments[0].role = PromptSegmentRole::User;
        changed_image.segments[0].content = PromptSegmentContent::UserContent {
            content: UserMessageContent::new(
                "image",
                vec![UserImageAttachment {
                    id: "image-1".into(),
                    label: "image.png".into(),
                    mime: "image/png".into(),
                    data_url: "data:image/png;base64,AAAA".into(),
                }],
            ),
        };
        let mut changed_suffix = base.prompt_plan.clone();
        let suffix = changed_suffix.segments.last_mut().expect("volatile suffix");
        suffix.text = "changed volatile suffix".into();
        suffix.content = PromptSegmentContent::Text {
            text: "changed volatile suffix".into(),
        };
        let mut stable_after_boundary = base.prompt_plan.clone();
        stable_after_boundary
            .segments
            .push(base.prompt_plan.segments[0].clone());
        let after_boundary = stable_after_boundary
            .segments
            .last_mut()
            .expect("appended segment");
        after_boundary.text = "changed stable after boundary".into();
        after_boundary.content = PromptSegmentContent::Text {
            text: "changed stable after boundary".into(),
        };

        for changed in [
            report(
                "cache-test",
                ApiProtocol::Responses,
                "cache-model",
                &changed_stable,
                &tools,
                true,
            ),
            report(
                "cache-test",
                ApiProtocol::Responses,
                "cache-model",
                &changed_role,
                &tools,
                true,
            ),
            report(
                "cache-test",
                ApiProtocol::Completions,
                "cache-model",
                &base.prompt_plan,
                &tools,
                true,
            ),
            report(
                "other",
                ApiProtocol::Responses,
                "cache-model",
                &base.prompt_plan,
                &tools,
                true,
            ),
            report(
                "cache-test",
                ApiProtocol::Responses,
                "other-model",
                &base.prompt_plan,
                &tools,
                true,
            ),
            report(
                "cache-test",
                ApiProtocol::Responses,
                "cache-model",
                &base.prompt_plan,
                &changed_schema,
                true,
            ),
            report(
                "cache-test",
                ApiProtocol::Responses,
                "cache-model",
                &base.prompt_plan,
                &tools[..1],
                true,
            ),
            report(
                "cache-test",
                ApiProtocol::Responses,
                "cache-model",
                &base.prompt_plan,
                &tools.iter().cloned().rev().collect::<Vec<_>>(),
                true,
            ),
            report(
                "cache-test",
                ApiProtocol::Responses,
                "cache-model",
                &base.prompt_plan,
                &tools,
                false,
            ),
            report(
                "cache-test",
                ApiProtocol::Responses,
                "cache-model",
                &changed_image,
                &tools,
                true,
            ),
        ] {
            assert_ne!(
                changed.local_prefix_fingerprint,
                base_report.local_prefix_fingerprint
            );
        }

        for unchanged in [
            report(
                "cache-test",
                ApiProtocol::Responses,
                "cache-model",
                &changed_suffix,
                &tools,
                true,
            ),
            report(
                "cache-test",
                ApiProtocol::Responses,
                "cache-model",
                &stable_after_boundary,
                &tools,
                true,
            ),
        ] {
            assert_eq!(
                unchanged.local_prefix_fingerprint,
                base_report.local_prefix_fingerprint
            );
            assert_eq!(unchanged.routing_key, base_report.routing_key);
        }

        assert_eq!(
            report(
                "cache-test",
                ApiProtocol::Responses,
                "cache-model",
                &changed_stable,
                &tools,
                true
            )
            .routing_key,
            base_report.routing_key
        );
        assert_ne!(
            report(
                "other",
                ApiProtocol::Responses,
                "cache-model",
                &base.prompt_plan,
                &tools,
                true
            )
            .routing_key,
            base_report.routing_key
        );
        assert_ne!(
            report(
                "cache-test",
                ApiProtocol::Completions,
                "cache-model",
                &base.prompt_plan,
                &tools,
                true
            )
            .routing_key,
            base_report.routing_key
        );
        assert_ne!(
            report(
                "cache-test",
                ApiProtocol::Responses,
                "other-model",
                &base.prompt_plan,
                &tools,
                true
            )
            .routing_key,
            base_report.routing_key
        );
        assert_ne!(
            report(
                "cache-test",
                ApiProtocol::Responses,
                "cache-model",
                &base.prompt_plan,
                &changed_schema,
                true
            )
            .routing_key,
            base_report.routing_key
        );
        assert_ne!(
            report(
                "cache-test",
                ApiProtocol::Responses,
                "cache-model",
                &base.prompt_plan,
                &tools[..1],
                true
            )
            .routing_key,
            base_report.routing_key
        );
        assert_ne!(
            report(
                "cache-test",
                ApiProtocol::Responses,
                "cache-model",
                &base.prompt_plan,
                &tools.iter().cloned().rev().collect::<Vec<_>>(),
                true
            )
            .routing_key,
            base_report.routing_key
        );
        assert_ne!(
            report(
                "cache-test",
                ApiProtocol::Responses,
                "cache-model",
                &base.prompt_plan,
                &tools,
                false
            )
            .routing_key,
            base_report.routing_key
        );
    }

    #[test]
    fn selected_prompt_rebuild_preserves_responses_request_and_metadata() {
        let prelude = vec![PromptMessage::system("system")];
        let history = vec![
            HistoryItem::assistant("older assistant"),
            HistoryItem::user("latest user"),
        ];
        let evidence = vec![evidence("ev-1", "summary", "src/main.rs", 1)];
        let original = build_request_from_legacy(LegacyRequestBuilderInput {
            protocol: ApiProtocol::Responses,
            model_id: "gpt-test",
            model: metadata(8192),
            prelude: &prelude,
            history: &history,
            protected_start_index: 1,
            tools: &[],
            evidence: &evidence,
            history_adapter: None,
            context_view: None,
        })
        .expect("request builds");

        let rebuilt = build_request_from_selected_prompt(SelectedPromptRequestInput {
            protocol: ApiProtocol::Responses,
            model_id: "gpt-test",
            model: metadata(8192),
            tools: &[],
            prompt_plan: original.prompt_plan.clone(),
            budget: original.budget,
            selected_evidence_ids: original.selected_evidence_ids.clone(),
        })
        .expect("selected prompt rebuilds");

        assert_eq!(
            request_json(original.clone()),
            request_json(rebuilt.clone())
        );
        assert_eq!(rebuilt.budget, original.budget);
        assert_eq!(rebuilt.prompt_plan, original.prompt_plan);
        assert_eq!(
            rebuilt.selected_evidence_ids,
            original.selected_evidence_ids
        );
    }

    #[test]
    fn builds_responses_request_from_unified_history() {
        let history = vec![HistoryItem::user("hello"), HistoryItem::assistant("hi")];
        let result = build_request_from_legacy(LegacyRequestBuilderInput {
            protocol: ApiProtocol::Responses,
            model_id: "gpt-test",
            model: metadata(8192),
            prelude: &[],
            history: &history,
            protected_start_index: 0,
            tools: &[],
            evidence: &[],
            history_adapter: None,
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
        let result = build_request_from_legacy(LegacyRequestBuilderInput {
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
                ..ModelRequestMetadata::default()
            },
            prelude: &[],
            history: &[HistoryItem::user("hello")],
            protected_start_index: 0,
            tools: &[],
            evidence: &[],
            history_adapter: None,
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
        let result = build_request_from_legacy(LegacyRequestBuilderInput {
            protocol: ApiProtocol::Completions,
            model_id: "chat-test",
            model: metadata(8192),
            prelude: &[],
            history: &history,
            protected_start_index: 0,
            tools: &[],
            evidence: &[],
            history_adapter: None,
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
        let result = build_request_from_legacy(LegacyRequestBuilderInput {
            protocol: ApiProtocol::Completions,
            model_id: "chat-test",
            model: metadata(8192),
            prelude: &[],
            history: &history,
            protected_start_index: 0,
            tools: &[],
            evidence: &[],
            history_adapter: None,
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
        let result = build_request_from_legacy(LegacyRequestBuilderInput {
            protocol: ApiProtocol::Responses,
            model_id: "resp-test",
            model: metadata(8192),
            prelude: &[],
            history: &history,
            protected_start_index: 0,
            tools: &[],
            evidence: &[],
            history_adapter: None,
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
    fn responses_request_serializes_max_reasoning_effort_through_compatible_payload() {
        let result = build_request_from_legacy(LegacyRequestBuilderInput {
            protocol: ApiProtocol::Responses,
            model_id: "gpt-5.6-terra",
            model: ModelRequestMetadata {
                supports_reasoning: true,
                reasoning_effort: Some(ModelReasoningEffort::Max),
                reasoning_summary: Some(ModelReasoningSummary::Auto),
                ..metadata(8192)
            },
            prelude: &[],
            history: &[HistoryItem::user("hello")],
            protected_start_index: 0,
            tools: &[],
            evidence: &[],
            history_adapter: None,
            context_view: None,
        })
        .expect("request builds");

        let BuiltRequest::ResponsesCompatible(request) = result.request else {
            panic!("expected compatible responses request");
        };
        assert_eq!(request["reasoning"]["effort"], "max");
        assert_eq!(request["reasoning"]["summary"], "auto");
    }

    #[test]
    fn responses_request_serializes_max_reasoning_effort_without_summary() {
        let result = build_request_from_legacy(LegacyRequestBuilderInput {
            protocol: ApiProtocol::Responses,
            model_id: "gpt-5.6-terra",
            model: ModelRequestMetadata {
                supports_reasoning: true,
                reasoning_effort: Some(ModelReasoningEffort::Max),
                ..metadata(8192)
            },
            prelude: &[],
            history: &[HistoryItem::user("hello")],
            protected_start_index: 0,
            tools: &[],
            evidence: &[],
            history_adapter: None,
            context_view: None,
        })
        .expect("request builds");

        let BuiltRequest::ResponsesCompatible(request) = result.request else {
            panic!("expected compatible responses request");
        };
        assert_eq!(request["reasoning"]["effort"], "max");
        assert!(request["reasoning"].get("summary").is_none());
    }

    #[test]
    fn completions_request_serializes_max_reasoning_effort_through_compatible_payload() {
        let result = build_request_from_legacy(LegacyRequestBuilderInput {
            protocol: ApiProtocol::Completions,
            model_id: "gpt-5.6-terra",
            model: ModelRequestMetadata {
                supports_reasoning: true,
                reasoning_effort: Some(ModelReasoningEffort::Max),
                ..metadata(8192)
            },
            prelude: &[],
            history: &[HistoryItem::user("hello")],
            protected_start_index: 0,
            tools: &[],
            evidence: &[],
            history_adapter: None,
            context_view: None,
        })
        .expect("request builds");

        let BuiltRequest::CompletionsCompatible(request) = result.request else {
            panic!("expected compatible chat completions request");
        };
        assert_eq!(request["reasoning_effort"], "max");
    }

    #[test]
    fn completions_request_includes_model_generation_parameters() {
        let result = build_request_from_legacy(LegacyRequestBuilderInput {
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
                ..ModelRequestMetadata::default()
            },
            prelude: &[],
            history: &[HistoryItem::user("hello")],
            protected_start_index: 0,
            tools: &[],
            evidence: &[],
            history_adapter: None,
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

        let result = build_request_from_legacy(LegacyRequestBuilderInput {
            protocol: ApiProtocol::Responses,
            model_id: "gpt-test",
            model: metadata(8192),
            prelude: &prelude,
            history: &history,
            protected_start_index: 0,
            tools: &[],
            evidence: &[],
            history_adapter: None,
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

        let responses = build_request_from_legacy(LegacyRequestBuilderInput {
            protocol: ApiProtocol::Responses,
            model_id: "gpt-test",
            model: metadata(8192),
            prelude: &[],
            history: &history,
            protected_start_index: 0,
            tools: &[],
            evidence: &[],
            history_adapter: None,
            context_view: None,
        })
        .expect("responses request builds");
        let BuiltRequest::Responses(response_request) = responses.request else {
            panic!("expected responses request");
        };
        let response_json = serde_json::to_string(&response_request).expect("serialize response");
        assert!(response_json.contains("developer"));
        assert!(response_json.contains("以下是当前会话的结构化摘要"));

        let completions = build_request_from_legacy(LegacyRequestBuilderInput {
            protocol: ApiProtocol::Completions,
            model_id: "gpt-test",
            model: metadata(8192),
            prelude: &[],
            history: &history,
            protected_start_index: 0,
            tools: &[],
            evidence: &[],
            history_adapter: None,
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
    fn orphan_tool_outputs_fail_fast_when_building_chat_request() {
        let history = vec![
            HistoryItem::context_summary("旧工具调用已总结"),
            HistoryItem::ToolOutput {
                call_id: "call-orphan".into(),
                output_json: r#"{"ok":true}"#.into(),
            },
            HistoryItem::user("continue"),
        ];

        let error = build_request_from_legacy(LegacyRequestBuilderInput {
            protocol: ApiProtocol::Completions,
            model_id: "chat-test",
            model: metadata(8192),
            prelude: &[],
            history: &history,
            protected_start_index: 2,
            tools: &[],
            evidence: &[],
            history_adapter: None,
            context_view: None,
        })
        .expect_err("orphan tool output must fail");
        assert!(error.to_string().contains("orphan tool output"));
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

        let result = build_request_from_legacy(LegacyRequestBuilderInput {
            protocol: ApiProtocol::Completions,
            model_id: "chat-test",
            model: metadata(8192),
            prelude: &[],
            history: &history,
            protected_start_index: 2,
            tools: &[],
            evidence: &[],
            history_adapter: None,
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

        let result = build_request_from_legacy(LegacyRequestBuilderInput {
            protocol: ApiProtocol::Completions,
            model_id: "chat-test",
            model: metadata(8192),
            prelude: &prelude,
            history: &history,
            protected_start_index: 0,
            tools: &[],
            evidence: &[],
            history_adapter: None,
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
        let result = build_request_from_legacy(LegacyRequestBuilderInput {
            protocol: ApiProtocol::Responses,
            model_id: "gpt-test",
            model: metadata(1200),
            prelude: &[],
            history: &history,
            protected_start_index: 2,
            tools: &[],
            evidence: &[],
            history_adapter: None,
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
    fn truncation_retains_or_drops_complete_tool_call_batches_atomically_for_both_providers() {
        let history = vec![
            HistoryItem::user("old ordinary turn"),
            HistoryItem::assistant("x".repeat(10_000)),
            HistoryItem::AssistantToolCalls {
                text: None,
                calls: vec![
                    HistoryToolCall {
                        call_id: "batch-a".into(),
                        name: "fs__read".into(),
                        arguments_json: r#"{"path":"a"}"#.into(),
                    },
                    HistoryToolCall {
                        call_id: "batch-b".into(),
                        name: "fs__read".into(),
                        arguments_json: r#"{"path":"b"}"#.into(),
                    },
                ],
            },
            HistoryItem::ToolOutput {
                call_id: "batch-a".into(),
                output_json: r#"{"body":"output-a"}"#.into(),
            },
            HistoryItem::ToolOutput {
                call_id: "batch-b".into(),
                output_json: r#"{"body":"output-b"}"#.into(),
            },
            HistoryItem::user("current turn"),
        ];

        for protocol in [ApiProtocol::Responses, ApiProtocol::Completions] {
            let fit = build_request_from_legacy(LegacyRequestBuilderInput {
                protocol,
                model_id: "gpt-test",
                model: metadata_with_effective_input_limit(32_000, 300),
                prelude: &[],
                history: &history,
                protected_start_index: 5,
                tools: &[],
                evidence: &[],
                history_adapter: None,
                context_view: None,
            })
            .expect("batch fits after old history is dropped");
            assert_eq!(fit.budget.retained_history_items, 4);
            let fit_json: serde_json::Value =
                serde_json::from_str(&request_json(fit)).expect("request JSON");
            match protocol {
                ApiProtocol::Responses => {
                    let items = fit_json["input"].as_array().expect("responses input");
                    assert_eq!(items.len(), 5);
                    assert_eq!(items[0]["type"], "function_call");
                    assert_eq!(items[0]["call_id"], "batch-a");
                    assert_eq!(items[1]["type"], "function_call");
                    assert_eq!(items[1]["call_id"], "batch-b");
                    assert_eq!(items[2]["type"], "function_call_output");
                    assert_eq!(items[2]["call_id"], "batch-a");
                    assert_eq!(items[3]["type"], "function_call_output");
                    assert_eq!(items[3]["call_id"], "batch-b");
                    assert_eq!(items[4]["role"], "user");
                    assert_eq!(items[4]["content"][0]["text"], "current turn");
                }
                ApiProtocol::Completions => {
                    let messages = fit_json["messages"].as_array().expect("chat messages");
                    assert_eq!(messages.len(), 4);
                    assert_eq!(messages[0]["role"], "assistant");
                    assert_eq!(messages[0]["tool_calls"][0]["id"], "batch-a");
                    assert_eq!(messages[0]["tool_calls"][1]["id"], "batch-b");
                    assert_eq!(messages[1]["role"], "tool");
                    assert_eq!(messages[1]["tool_call_id"], "batch-a");
                    assert_eq!(messages[2]["role"], "tool");
                    assert_eq!(messages[2]["tool_call_id"], "batch-b");
                    assert_eq!(messages[3]["role"], "user");
                    assert_eq!(messages[3]["content"], "current turn");
                }
            }
            assert!(
                !request_json(
                    build_request_from_legacy(LegacyRequestBuilderInput {
                        protocol,
                        model_id: "gpt-test",
                        model: metadata_with_effective_input_limit(32_000, 150),
                        prelude: &[],
                        history: &history,
                        protected_start_index: 5,
                        tools: &[],
                        evidence: &[],
                        history_adapter: None,
                        context_view: None,
                    })
                    .expect("current turn fits")
                )
                .contains("batch-a")
            );
        }
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

        let without_tools = build_request_from_legacy(LegacyRequestBuilderInput {
            protocol: ApiProtocol::Responses,
            model_id: "gpt-test",
            model: metadata(4096),
            prelude: &[],
            history: &history,
            protected_start_index: 2,
            tools: &[],
            evidence: &[],
            history_adapter: None,
            context_view: None,
        })
        .expect("request builds");
        let with_tools = build_request_from_legacy(LegacyRequestBuilderInput {
            protocol: ApiProtocol::Responses,
            model_id: "gpt-test",
            model: metadata(4096),
            prelude: &[],
            history: &history,
            protected_start_index: 2,
            tools: &tools,
            evidence: &[],
            history_adapter: None,
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

        let uncapped = build_request_from_legacy(LegacyRequestBuilderInput {
            protocol: ApiProtocol::Responses,
            model_id: "gpt-test",
            model: metadata(32_000),
            prelude: &[],
            history: &history,
            protected_start_index: 2,
            tools: &[],
            evidence: &[],
            history_adapter: None,
            context_view: None,
        })
        .expect("uncapped request builds");
        let capped = build_request_from_legacy(LegacyRequestBuilderInput {
            protocol: ApiProtocol::Responses,
            model_id: "gpt-test",
            model: metadata_with_effective_input_limit(32_000, 900),
            prelude: &[],
            history: &history,
            protected_start_index: 2,
            tools: &[],
            evidence: &[],
            history_adapter: None,
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

        let capped = build_request_from_legacy(LegacyRequestBuilderInput {
            protocol: ApiProtocol::Responses,
            model_id: "gpt-test",
            model: metadata_with_effective_input_limit(32_000, 2_000),
            prelude: &[],
            history: &history,
            protected_start_index: 0,
            tools: &tools,
            evidence: &[],
            history_adapter: None,
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

        let responses = build_request_from_legacy(LegacyRequestBuilderInput {
            protocol: ApiProtocol::Responses,
            model_id: "gpt-test",
            model: metadata(8192),
            prelude: &[],
            history: &history,
            protected_start_index: 2,
            tools: &[],
            evidence: &evidence,
            history_adapter: None,
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

        let completions = build_request_from_legacy(LegacyRequestBuilderInput {
            protocol: ApiProtocol::Completions,
            model_id: "chat-test",
            model: metadata(8192),
            prelude: &[],
            history: &history,
            protected_start_index: 2,
            tools: &[],
            evidence: &evidence,
            history_adapter: None,
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

        let result = build_request_from_legacy(LegacyRequestBuilderInput {
            protocol: ApiProtocol::Responses,
            model_id: "gpt-test",
            model,
            prelude: &[],
            history: &history,
            protected_start_index: 0,
            tools: &[],
            evidence: &evidence,
            history_adapter: None,
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

        let result = build_request_from_legacy(LegacyRequestBuilderInput {
            protocol: ApiProtocol::Responses,
            model_id: "gpt-test",
            model,
            prelude: &[],
            history: &history,
            protected_start_index: 0,
            tools: &[],
            evidence: &evidence,
            history_adapter: None,
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

        let err = build_request_from_legacy(LegacyRequestBuilderInput {
            protocol: ApiProtocol::Responses,
            model_id: "gpt-test",
            model: metadata(1024),
            prelude: &[],
            history: &history,
            protected_start_index: 2,
            tools: &[],
            evidence: &[],
            history_adapter: None,
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

        let err = build_request_from_legacy(LegacyRequestBuilderInput {
            protocol: ApiProtocol::Responses,
            model_id: "gpt-test",
            model: metadata_with_effective_input_limit(32_000, 300),
            prelude: &[],
            history: &history,
            protected_start_index: 0,
            tools: &[],
            evidence: &[],
            history_adapter: None,
            context_view: None,
        })
        .expect_err("effective-input-limited protected current turn should fail fast");

        let message = err.to_string();
        assert!(message.contains("protected/current context tokens"));
        assert!(message.contains("exceed budget (300)"));
    }

    #[test]
    fn rejects_zero_effective_input_limit_metadata() {
        let err = build_request_from_legacy(LegacyRequestBuilderInput {
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
            history_adapter: None,
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
        let baseline = build_request_from_legacy(LegacyRequestBuilderInput {
            protocol: ApiProtocol::Responses,
            model_id: "gpt-test",
            model: metadata(8192),
            prelude: &[],
            history: &history,
            protected_start_index: 0,
            tools: &[],
            evidence: &[],
            history_adapter: None,
            context_view: None,
        })
        .expect("request builds");
        let repeat = build_request_from_legacy(LegacyRequestBuilderInput {
            protocol: ApiProtocol::Responses,
            model_id: "gpt-test",
            model: metadata(8192),
            prelude: &[],
            history: &history,
            protected_start_index: 0,
            tools: &[],
            evidence: &[],
            history_adapter: None,
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
            build_request_from_legacy(LegacyRequestBuilderInput {
                protocol: ApiProtocol::Responses,
                model_id: "gpt-test",
                model: metadata(8192),
                prelude: &[],
                history: &history,
                protected_start_index: 1,
                tools: &[],
                evidence: &[],
                history_adapter: None,
                context_view: Some(&context_view),
            })
            .expect("request builds"),
        );
        let second = request_json(
            build_request_from_legacy(LegacyRequestBuilderInput {
                protocol: ApiProtocol::Responses,
                model_id: "gpt-test",
                model: metadata(8192),
                prelude: &[],
                history: &history,
                protected_start_index: 1,
                tools: &[],
                evidence: &[],
                history_adapter: None,
                context_view: Some(&context_view),
            })
            .expect("request rebuilds"),
        );
        assert_eq!(first, second);
    }

    #[test]
    fn runtime_session_state_stops_stable_prefix_and_changes_both_provider_requests() {
        fn snapshot(runtime_material: &str) -> RuntimeSnapshot {
            let frame = |kind, source, ordinal, stable_key, item| {
                RuntimeFrame::new(
                    kind,
                    FrameVisibility::Active,
                    RuntimeFrameProvenance::new(source),
                    RuntimeFrameIdSeed {
                        frame_kind: kind,
                        source,
                        ordinal,
                        stable_key,
                        source_span: None,
                    },
                )
                .with_protocol(item)
            };

            let mut snapshot = RuntimeSnapshot::new("runtime-provenance-cache-test");
            snapshot.push_frame(frame(
                RuntimeFrameKind::Summary,
                RuntimeSource::SessionState,
                1,
                "child-session-runtime-material",
                ProtocolFrameItem::ContextSummary {
                    text: format!("[Context: Child Sessions]\n- {runtime_material}"),
                },
            ));
            snapshot.push_frame(frame(
                RuntimeFrameKind::Assistant,
                RuntimeSource::Transcript,
                2,
                "older-ordinary-transcript",
                ProtocolFrameItem::AssistantText {
                    text: "older ordinary transcript".into(),
                },
            ));
            snapshot
        }

        let prelude = [PromptMessage::system("stable static prelude")];
        for protocol in [ApiProtocol::Responses, ApiProtocol::Completions] {
            let first = build_request(RequestBuilderInput {
                protocol,
                model_id: "gpt-test",
                model: metadata(8192),
                prelude: &prelude,
                snapshot: &snapshot("child status: running"),
                tools: &[],
            })
            .expect("request builds");
            let first_prefix = first
                .prompt_plan
                .stable_prefix_hash()
                .expect("stable static prelude has fingerprint")
                .to_string();
            assert_eq!(first.prompt_plan.stable_prefix_end, Some(0));
            assert_eq!(
                first.prompt_plan.segments[1].stability,
                prompt_plan::PromptSegmentStability::Volatile
            );
            assert_eq!(
                first.prompt_plan.segments[1].source.provenance.source,
                RuntimeSource::SessionState
            );
            assert!(
                first.prompt_plan.segments[2].stability
                    == prompt_plan::PromptSegmentStability::Stable
            );
            assert!(!first.prompt_plan.segments[2].cache.cache_eligible);
            let runtime_tokens = first.prompt_plan.segments[1]
                .tokens
                .estimated_input_tokens
                .expect("runtime material has token estimate");
            assert!(first.prompt_plan.token_report().volatile_prompt_tokens >= runtime_tokens);
            assert!(
                first
                    .prompt_plan
                    .token_report()
                    .stable_after_boundary_tokens
                    > 0
            );
            let first_json = request_json(first);

            let changed = build_request(RequestBuilderInput {
                protocol,
                model_id: "gpt-test",
                model: metadata(8192),
                prelude: &prelude,
                snapshot: &snapshot("child status: complete"),
                tools: &[],
            })
            .expect("changed request builds");
            assert_eq!(
                changed.prompt_plan.stable_prefix_hash(),
                Some(first_prefix.as_str())
            );
            let changed_json = request_json(changed);
            assert_ne!(first_json, changed_json);
            assert!(changed_json.contains("child status: complete"));
        }
    }

    #[test]
    fn context_view_adapter_hard_and_pinned_preludes_are_volatile() {
        let context_view = sample_context_view(false);
        let adapter = context_view_history_adapter(&context_view, &[], 0);
        assert_eq!(adapter.prelude.len(), 2);
        assert!(
            adapter.prelude[0]
                .text
                .starts_with("[Context: Hard Context]")
        );
        assert!(
            adapter.prelude[1]
                .text
                .starts_with("[Context: Pinned Context]")
        );
        assert!(
            adapter
                .prelude
                .iter()
                .all(|message| { message.origin == PromptMessageOrigin::RuntimeContextView })
        );

        let plan_for = |prelude: &[PromptMessage]| {
            build_prompt_plan(PromptPlanBuildInput {
                protocol: ApiProtocol::Responses,
                model_id: "gpt-test",
                prelude,
                snapshot: &RuntimeSnapshot::new("context-origin-test"),
                selected_frames: &[],
                protected_suffix_len: 0,
                evidence_message: None,
                selected_evidence_ids: &[],
            })
        };
        let context_first = plan_for(&adapter.prelude);
        assert_eq!(context_first.cacheable_prefix_len(), 0);
        assert_eq!(context_first.stable_prefix_end, None);

        let mut prelude = vec![PromptMessage::system("stable system")];
        prelude.extend(adapter.prelude.clone());
        let plan = plan_for(&prelude);
        assert_eq!(plan.cacheable_prefix_len(), 1);
        assert_eq!(
            plan.segments[1].stability,
            prompt_plan::PromptSegmentStability::Volatile
        );
        assert_eq!(
            plan.segments[2].stability,
            prompt_plan::PromptSegmentStability::Volatile
        );
        let report = plan.token_report();
        let stable_tokens = plan.segments[0].tokens.estimated_input_tokens.unwrap();
        let volatile_tokens = plan.segments[1..]
            .iter()
            .map(|segment| segment.tokens.estimated_input_tokens.unwrap())
            .sum::<u64>();
        assert_eq!(report.total_prompt_tokens, stable_tokens + volatile_tokens);
        assert_eq!(report.stable_prompt_tokens, stable_tokens);
        assert_eq!(report.volatile_prompt_tokens, volatile_tokens);
        assert_eq!(report.cacheable_prefix_tokens, stable_tokens);
        assert_eq!(report.stable_after_boundary_tokens, 0);
        assert_eq!(report.first_volatile_index, Some(1));

        let mut changed_prelude = prelude;
        changed_prelude[1].text.push_str(" changed");
        let changed = plan_for(&changed_prelude);
        assert_eq!(plan.stable_prefix_hash(), changed.stable_prefix_hash());
    }

    #[test]
    fn explicit_history_adapter_matches_context_view_compatibility_path() {
        let history = vec![
            HistoryItem::assistant("previous"),
            HistoryItem::user("current user"),
        ];
        let context_view = sample_context_view(true);
        let adapter = context_view_history_adapter(&context_view, &history, 1);

        let compatibility = request_json(
            build_request_from_legacy(LegacyRequestBuilderInput {
                protocol: ApiProtocol::Responses,
                model_id: "gpt-test",
                model: metadata(8192),
                prelude: &[],
                history: &history,
                protected_start_index: 1,
                tools: &[],
                evidence: &[],
                history_adapter: None,
                context_view: Some(&context_view),
            })
            .expect("compatibility request builds"),
        );
        let explicit = request_json(
            build_request_from_legacy(LegacyRequestBuilderInput {
                protocol: ApiProtocol::Responses,
                model_id: "gpt-test",
                model: metadata(8192),
                prelude: &[],
                history: &history,
                protected_start_index: 1,
                tools: &[],
                evidence: &[],
                history_adapter: Some(&adapter),
                context_view: None,
            })
            .expect("adapter request builds"),
        );

        assert_eq!(explicit, compatibility);
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
        let summary_texts = adapter_summary_texts(&sections);
        combined.extend(summary_texts.iter().map(String::as_str));
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
            build_request_from_legacy(LegacyRequestBuilderInput {
                protocol: ApiProtocol::Responses,
                model_id: "gpt-test",
                model: metadata(8192),
                prelude: &[],
                history: &history,
                protected_start_index: 1,
                tools: &[],
                evidence: &[],
                history_adapter: None,
                context_view: Some(&sample_context_view(false)),
            })
            .expect("closed request builds"),
        );
        let open_json = request_json(
            build_request_from_legacy(LegacyRequestBuilderInput {
                protocol: ApiProtocol::Responses,
                model_id: "gpt-test",
                model: metadata(8192),
                prelude: &[],
                history: &history,
                protected_start_index: 1,
                tools: &[],
                evidence: &[],
                history_adapter: None,
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
        let err = build_request_from_legacy(LegacyRequestBuilderInput {
            protocol: ApiProtocol::Responses,
            model_id: "gpt-test",
            model: metadata(1024),
            prelude: &[],
            history: &history,
            protected_start_index: 1,
            tools: &[],
            evidence: &[],
            history_adapter: None,
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

        let result = build_request_from_legacy(LegacyRequestBuilderInput {
            protocol: ApiProtocol::Responses,
            model_id: "gpt-test",
            model: metadata(8192),
            prelude: &[],
            history: &[HistoryItem::user("current user")],
            protected_start_index: 0,
            tools: &[],
            evidence: &[],
            history_adapter: None,
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
            .chain(adapter_summary_texts(&sections).iter().map(String::as_str))
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
            .chain(adapter_summary_texts(&sections).iter().map(String::as_str))
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
        let combined = adapter_summary_texts(&sections).join("\n");

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
        let opened_combined = adapter_summary_texts(&opened_sections).join("\n");

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
        let combined = adapter_summary_texts(&sections).join("\n");

        assert!(!combined.contains("folded-output-seq-2-stdout"));
        assert!(!combined.contains("folded-output-seq-2-stderr"));
    }

    #[test]
    fn compacted_projection_excludes_old_raw_context_and_folded_placeholders() {
        let records = vec![
            transcript_record(
                1,
                TranscriptEvent::AssistantMessage {
                    content: "old raw note should disappear".into(),
                },
            ),
            transcript_record(
                2,
                TranscriptEvent::ToolCallStarted {
                    call_id: "call-1".into(),
                    name: "shell__exec".into(),
                    args: json!({"command": "cargo test"}),
                },
            ),
            transcript_record(
                3,
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
                4,
                TranscriptEvent::ContextCompaction(crate::agent::ContextCompactionEvent {
                    outcome: "succeeded".into(),
                    summary: "compacted summary survives".into(),
                    tail_start_index: 3,
                    original_history_items: 3,
                    retained_history_items: 1,
                    retired_source_spans: vec![crate::agent::ContextCompactionSourceSpan {
                        start_sequence: 1,
                        end_sequence: 3,
                    }],
                    frame_identity_bindings: Vec::new(),
                    detail: None,
                }),
            ),
            transcript_record(
                5,
                TranscriptEvent::UserMessage {
                    content: UserMessageContent::from("current tail stays"),
                },
            ),
        ];
        let context_view = project_context_view(&records).expect("context view projection");
        let history = restore_session_history_projection(&records);

        let json = request_json(
            build_request_from_legacy(LegacyRequestBuilderInput {
                protocol: ApiProtocol::Responses,
                model_id: "gpt-test",
                model: metadata(8192),
                prelude: &[],
                history: &history,
                protected_start_index: history.len().saturating_sub(1),
                tools: &[],
                evidence: &[],
                history_adapter: None,
                context_view: Some(&context_view),
            })
            .expect("request builds"),
        );

        assert!(json.contains("compacted summary survives"));
        assert!(json.contains("current tail stays"));
        assert!(!json.contains("old raw note should disappear"));
        assert!(!json.contains("folded-output-seq-3-stdout"));
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
            project_session_restore_snapshot("s".into(), records.clone()).expect("snapshot");
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

        let result = build_request_from_legacy(LegacyRequestBuilderInput {
            protocol: ApiProtocol::Responses,
            model_id: "gpt-test",
            model: metadata(32_768),
            prelude: &[],
            history: &current_history,
            protected_start_index: 0,
            tools: &[],
            evidence: &snapshot.evidence,
            history_adapter: None,
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
            project_session_restore_snapshot("s".into(), records.clone()).expect("snapshot");
        let tree = project_context_tree(&records).expect("legacy tree defaults to root");
        assert_eq!(tree.root_node_id().as_str(), "root");
        assert_eq!(tree.active_node_id().map(|id| id.as_str()), Some("root"));
        let projection = project_restored_context_view(&records).expect("legacy context view");

        let result = build_request_from_legacy(LegacyRequestBuilderInput {
            protocol: ApiProtocol::Responses,
            model_id: "gpt-test",
            model: metadata(8192),
            prelude: &[],
            history: &snapshot.history,
            protected_start_index: snapshot.history.len().saturating_sub(1),
            tools: &[],
            evidence: &snapshot.evidence,
            history_adapter: None,
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

    #[test]
    fn both_providers_render_mixed_skill_source_and_fallback_once_in_legal_order() {
        fn frame(
            kind: RuntimeFrameKind,
            visibility: FrameVisibility,
            ordinal: u32,
            item: ProtocolFrameItem,
        ) -> RuntimeFrame {
            RuntimeFrame::new(
                kind,
                visibility,
                RuntimeFrameProvenance::new(RuntimeSource::Transcript),
                RuntimeFrameIdSeed {
                    frame_kind: kind,
                    source: RuntimeSource::Transcript,
                    ordinal,
                    stable_key: "mixed-skill-protocol",
                    source_span: None,
                },
            )
            .with_protocol(item)
        }

        let mut snapshot = RuntimeSnapshot::new("main");
        snapshot.push_frame(frame(
            RuntimeFrameKind::Assistant,
            FrameVisibility::Active,
            0,
            ProtocolFrameItem::AssistantText {
                text: "OLD-UNRELATED-HISTORY".repeat(2_000),
            },
        ));
        snapshot.push_frame(frame(
            RuntimeFrameKind::ToolCall,
            FrameVisibility::Retired,
            1,
            ProtocolFrameItem::AssistantToolCalls {
                text: None,
                calls: vec![HistoryToolCall {
                    call_id: "retired-skill".into(),
                    name: "skill".into(),
                    arguments_json: "{}".into(),
                }],
            },
        ));
        snapshot.push_frame(frame(
            RuntimeFrameKind::ToolOutput,
            FrameVisibility::Retired,
            2,
            ProtocolFrameItem::ToolOutput {
                call_id: "retired-skill".into(),
                output_json: r#"{"ok":true,"tool":"skill","data":{"name":"retired","content":"RETIRED-SKILL-BODY"}}"#.into(),
            },
        ));
        snapshot.push_frame(frame(
            RuntimeFrameKind::ToolCall,
            FrameVisibility::Active,
            3,
            ProtocolFrameItem::AssistantToolCalls {
                text: None,
                calls: vec![HistoryToolCall {
                    call_id: "active-skill".into(),
                    name: "skill".into(),
                    arguments_json: "{}".into(),
                }],
            },
        ));
        snapshot.push_frame(frame(
            RuntimeFrameKind::ToolOutput,
            FrameVisibility::Active,
            4,
            ProtocolFrameItem::ToolOutput {
                call_id: "active-skill".into(),
                output_json: r#"{"ok":true,"tool":"skill","data":{"name":"active","content":"ACTIVE-SKILL-BODY"}}"#.into(),
            },
        ));
        snapshot.push_frame(frame(
            RuntimeFrameKind::User,
            FrameVisibility::Active,
            5,
            ProtocolFrameItem::UserMessage {
                content: UserMessageContent::from("continue"),
            },
        ));
        crate::skills::reconcile_loaded_skill_material(&mut snapshot)
            .expect("reconciles persisted skill material");
        let retired_source_id = snapshot
            .prompt_contributors
            .iter()
            .find(|contributor| contributor.contributor_id == "skill-material:retired-skill")
            .expect("retired skill contributor")
            .source_frame_ids[0];
        let source = snapshot
            .frames
            .iter_mut()
            .find(|frame| frame.id == retired_source_id)
            .expect("retired skill source");
        let ProtocolFrameItem::ToolOutput { output_json, .. } = source.protocol.as_mut().unwrap()
        else {
            panic!("retired skill source is a tool output");
        };
        *output_json = r#"{"_compaction":{"pruned":true,"reason":"tool output pruned by compaction.prune","original_chars":9999,"tool":"skill"}}"#.into();
        let source_json: serde_json::Value =
            serde_json::from_str(output_json).expect("structural compaction marker");
        assert_eq!(source_json["_compaction"]["pruned"], true);
        crate::skills::reconcile_loaded_skill_material(&mut snapshot)
            .expect("preserves detached skill material after pruning");

        for protocol in [ApiProtocol::Responses, ApiProtocol::Completions] {
            let result = build_request(RequestBuilderInput {
                protocol,
                model_id: "gpt-test",
                model: metadata_with_effective_input_limit(32_000, 350),
                prelude: &[],
                snapshot: &snapshot,
                tools: &[],
            })
            .expect("request builds");
            let json = request_json(result);

            assert!(json.contains("RETIRED-SKILL-BODY"));
            assert!(json.contains("ACTIVE-SKILL-BODY"));
            assert_eq!(json.matches("RETIRED-SKILL-BODY").count(), 1);
            assert_eq!(json.matches("ACTIVE-SKILL-BODY").count(), 1);
            assert!(!json.contains("tool output pruned by compaction.prune"));
            assert!(!json.contains("OLD-UNRELATED-HISTORY"));
            match protocol {
                ApiProtocol::Responses => {
                    let input = serde_json::from_str::<serde_json::Value>(&json)
                        .expect("responses request JSON")["input"]
                        .as_array()
                        .expect("responses input array")
                        .clone();
                    assert_eq!(input[0]["role"], "developer");
                    assert_eq!(input[1]["type"], "function_call");
                    assert_eq!(input[2]["type"], "function_call_output");
                    assert_eq!(input[3]["role"], "user");
                }
                ApiProtocol::Completions => {
                    let messages = serde_json::from_str::<serde_json::Value>(&json)
                        .expect("chat request JSON")["messages"]
                        .as_array()
                        .expect("chat messages array")
                        .clone();
                    assert_eq!(messages[0]["role"], "developer");
                    assert_eq!(messages[1]["role"], "assistant");
                    assert_eq!(messages[2]["role"], "tool");
                    assert_eq!(messages[3]["role"], "user");
                }
            }
        }
    }

    #[test]
    fn group_16_both_provider_requests_share_canonical_surviving_context() {
        let snapshot = crate::context_tools::group_16_runtime_snapshot();
        crate::protocol_frames::validate_history_items_complete(
            &snapshot.active_history_items(),
            None,
        )
        .expect("canonical protocol frames remain complete");

        for protocol in [ApiProtocol::Responses, ApiProtocol::Completions] {
            let json = request_json(
                build_request(RequestBuilderInput {
                    protocol,
                    model_id: "gpt-test",
                    model: metadata(8192),
                    prelude: &[],
                    snapshot: &snapshot,
                    tools: &[],
                })
                .expect("canonical request builds"),
            );

            for surviving in [
                "CANONICAL ACTIVE TITLE",
                "CANONICAL ACTIVE CONTENT CURRENT-TAIL-SENTINEL",
                "PINNED ACTIVE TITLE",
                "ACTIVE-FOLDED-SENTINEL",
                "SURVIVING-PROTOCOL-SENTINEL",
            ] {
                assert!(json.contains(surviving), "{protocol:?}: {json}");
            }
            for retired in [
                "RETIRED-RAW-SENTINEL",
                "RETIRED-FOLDED-SENTINEL",
                "COMPACTED FOLDED TITLE",
            ] {
                assert!(!json.contains(retired), "{protocol:?}: {json}");
            }
            let request: serde_json::Value = serde_json::from_str(&json).expect("request JSON");
            match protocol {
                ApiProtocol::Responses => {
                    assert!(request["input"].as_array().is_some_and(|items| {
                        items.iter().any(|item| {
                            item["type"] == "function_call_output"
                                && item["call_id"] == "current-call"
                        })
                    }));
                }
                ApiProtocol::Completions => {
                    assert!(
                        request["messages"]
                            .as_array()
                            .is_some_and(|items| items.iter().any(|item| item["role"] == "tool"
                                && item["tool_call_id"] == "current-call"))
                    );
                }
            }
        }
    }

    #[test]
    fn planner_is_pure_deterministic_and_preserves_protected_tool_groups() {
        let frame = |kind, ordinal, item| {
            RuntimeFrame::new(
                kind,
                FrameVisibility::Active,
                RuntimeFrameProvenance::new(RuntimeSource::Transcript),
                RuntimeFrameIdSeed {
                    frame_kind: kind,
                    source: RuntimeSource::Transcript,
                    ordinal,
                    stable_key: "planner-purity",
                    source_span: None,
                },
            )
            .with_protocol(item)
        };
        let tool_call = frame(
            RuntimeFrameKind::ToolCall,
            0,
            ProtocolFrameItem::AssistantToolCalls {
                text: None,
                calls: vec![HistoryToolCall {
                    call_id: "call-1".into(),
                    name: "read".into(),
                    arguments_json: "{}".into(),
                }],
            },
        );
        let tool_output = frame(
            RuntimeFrameKind::ToolOutput,
            1,
            ProtocolFrameItem::ToolOutput {
                call_id: "call-1".into(),
                output_json: r#"{"ok":true}"#.into(),
            },
        );
        let user = frame(
            RuntimeFrameKind::User,
            2,
            ProtocolFrameItem::UserMessage {
                content: UserMessageContent::from("current request"),
            },
        );
        let mut snapshot = RuntimeSnapshot::new("planner-purity");
        snapshot.compaction.protected_frame_ids.push(tool_output.id);
        snapshot.push_frame(tool_call);
        snapshot.push_frame(tool_output);
        snapshot.push_frame(user);
        let mut retired = frame(
            RuntimeFrameKind::Assistant,
            3,
            ProtocolFrameItem::AssistantText {
                text: "RETIRED-PLANNER-FRAME".into(),
            },
        );
        retired.visibility = FrameVisibility::Retired;
        snapshot.push_frame(retired);
        let before = snapshot.clone();
        let input = PromptPlannerInput {
            protocol: ApiProtocol::Responses,
            model: metadata(8192),
            model_id: "gpt-test",
            prelude: &[PromptMessage::system("system")],
            snapshot: &snapshot,
            tools: &[],
        };

        let first = PromptPlanner::plan(input.clone()).expect("planner succeeds");
        let second = PromptPlanner::plan(input).expect("planner repeats");

        assert_eq!(snapshot, before);
        assert_eq!(first.prompt_plan.segments, second.prompt_plan.segments);
        assert_eq!(first.budget, second.budget);
        assert_eq!(first.selected_evidence_ids, second.selected_evidence_ids);
        assert_eq!(
            first.prompt_plan.stable_prefix_hash(),
            second.prompt_plan.stable_prefix_hash()
        );
        assert!(first.prompt_plan.segments.iter().any(|segment| {
            matches!(
                segment.content,
                PromptSegmentContent::AssistantToolCalls { ref calls, .. }
                    if calls.iter().any(|call| call.call_id == "call-1")
            )
        }));
        assert!(first.prompt_plan.segments.iter().any(|segment| {
            matches!(
                segment.content,
                PromptSegmentContent::ToolOutput { ref call_id, .. } if call_id == "call-1"
            )
        }));
        assert!(
            first
                .prompt_plan
                .segments
                .iter()
                .all(|segment| segment.text != "RETIRED-PLANNER-FRAME")
        );
    }

    #[test]
    fn build_request_matches_direct_planner_then_serializer_for_both_protocols() {
        let mut snapshot = RuntimeSnapshot::new("planner-serializer-equivalence");
        snapshot.push_frame(
            RuntimeFrame::new(
                RuntimeFrameKind::User,
                FrameVisibility::Active,
                RuntimeFrameProvenance::new(RuntimeSource::Transcript),
                RuntimeFrameIdSeed {
                    frame_kind: RuntimeFrameKind::User,
                    source: RuntimeSource::Transcript,
                    ordinal: 0,
                    stable_key: "planner-serializer-equivalence",
                    source_span: None,
                },
            )
            .with_protocol(ProtocolFrameItem::UserMessage {
                content: UserMessageContent::from("hello"),
            }),
        );
        let prelude = [PromptMessage::system("system")];
        for protocol in [ApiProtocol::Responses, ApiProtocol::Completions] {
            let model = metadata(8192);
            let planned = PromptPlanner::plan(PromptPlannerInput {
                protocol,
                model: model.clone(),
                model_id: "gpt-test",
                prelude: &prelude,
                snapshot: &snapshot,
                tools: &[],
            })
            .expect("planner succeeds");
            let direct = build_request_from_selected_prompt(SelectedPromptRequestInput {
                protocol,
                model_id: "gpt-test",
                model: model.clone(),
                tools: &[],
                prompt_plan: planned.prompt_plan,
                budget: planned.budget,
                selected_evidence_ids: planned.selected_evidence_ids,
            })
            .expect("serializer succeeds");
            let built = build_request(RequestBuilderInput {
                protocol,
                model_id: "gpt-test",
                model,
                prelude: &prelude,
                snapshot: &snapshot,
                tools: &[],
            })
            .expect("builder succeeds");
            assert_eq!(request_json(direct), request_json(built));
        }
    }

    #[test]
    fn child_session_metadata_does_not_change_provider_prompt_or_subagent_evidence_context() {
        let mut snapshot = RuntimeSnapshot::new("child-session-metadata");
        snapshot.push_frame(
            RuntimeFrame::new(
                RuntimeFrameKind::User,
                FrameVisibility::Active,
                RuntimeFrameProvenance::new(RuntimeSource::Transcript),
                RuntimeFrameIdSeed {
                    frame_kind: RuntimeFrameKind::User,
                    source: RuntimeSource::Transcript,
                    ordinal: 0,
                    stable_key: "child-session-metadata-user",
                    source_span: None,
                },
            )
            .with_protocol(ProtocolFrameItem::UserMessage {
                content: UserMessageContent::from("continue delegated work with src/subagent.rs"),
            }),
        );
        let mut subagent_evidence = evidence(
            "subagent-evidence",
            "SUBAGENT-EVIDENCE-SENTINEL",
            "src/subagent.rs",
            1,
        );
        subagent_evidence.source = EvidenceSource::Subagent {
            run_id: "run-1".into(),
            child_session_id: "child-1".into(),
            source_session_id: "child-session-1".into(),
            parent_tool: "agent__implementer".into(),
            parent_turn_id: Some("turn-1".into()),
            parent_session_id: Some("parent-session".into()),
        };
        snapshot.set_evidence(vec![subagent_evidence]);
        let mut with_child_session = snapshot.clone();
        with_child_session.push_child_session(RuntimeChildSession {
            parent_run_id: "run-1".into(),
            child_session_id: "child-1".into(),
            agent_name: "implementer".into(),
            status: "completed".into(),
            summary: "CHILD-SESSION-METADATA-SENTINEL".into(),
            timestamp_ms: 1,
        });
        let prelude = [PromptMessage::developer_with_origin(
            "UNRECONCILED-SUBAGENT-CONTEXT-SENTINEL",
            PromptMessageOrigin::UnreconciledSubagentContext,
        )];

        for protocol in [ApiProtocol::Responses, ApiProtocol::Completions] {
            let without_child_session = build_request(RequestBuilderInput {
                protocol,
                model_id: "gpt-test",
                model: metadata(8192),
                prelude: &prelude,
                snapshot: &snapshot,
                tools: &[],
            })
            .expect("request without child session builds");
            let with_child_session_request = build_request(RequestBuilderInput {
                protocol,
                model_id: "gpt-test",
                model: metadata(8192),
                prelude: &prelude,
                snapshot: &with_child_session,
                tools: &[],
            })
            .expect("request with child session builds");

            assert_eq!(
                request_value(&without_child_session),
                request_value(&with_child_session_request),
                "{protocol:?} provider request must ignore child session metadata"
            );
            assert_eq!(
                without_child_session.prompt_plan, with_child_session_request.prompt_plan,
                "{protocol:?} prompt plan must ignore child session metadata"
            );
            assert_eq!(
                without_child_session.selected_evidence_ids,
                with_child_session_request.selected_evidence_ids
            );
            let request = request_json(with_child_session_request);
            assert!(request.contains("SUBAGENT-EVIDENCE-SENTINEL"));
            assert!(request.contains("UNRECONCILED-SUBAGENT-CONTEXT-SENTINEL"));
            assert!(!request.contains("CHILD-SESSION-METADATA-SENTINEL"));
        }
    }
}
