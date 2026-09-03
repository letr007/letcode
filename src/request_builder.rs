use anyhow::Result;
#[path = "request_builder/history_budget.rs"]
mod history_budget;
#[path = "request_builder/prompt_plan.rs"]
pub(crate) mod prompt_plan;
#[path = "request_builder/runtime_projection.rs"]
mod runtime_projection;
use crate::config::PromptCacheConfig;
#[cfg(test)]
use crate::protocol_frames::ProtocolFrameItem;
pub(crate) use crate::protocol_frames::history_items_from_frames;
use crate::protocol_frames::{ProtocolFrame, validate_history_items_complete};
pub use crate::protocol_frames::{
    ProtocolItem as HistoryItem, ProtocolToolCall as HistoryToolCall,
};
use crate::request_builder::prompt_plan::{
    PlannedPrompt, PromptPlan, PromptPlanner, PromptPlannerInput, PromptSegmentContent,
    PromptSegmentRole,
};
#[cfg(test)]
use crate::runtime_context::RuntimeFrameKind;
#[cfg(test)]
use crate::runtime_context::{
    FrameVisibility, RuntimeFrame, RuntimeFrameProvenance, RuntimeSource,
};
use crate::runtime_context::{PromptContributorKind, RuntimeSnapshot};

#[cfg(test)]
use crate::user_content::{UserImageAttachment, UserMessageContent};
#[cfg(test)]
use crate::{
    evidence::EvidenceRecord, protocol_frames::history_items_to_frames,
    runtime_context::RuntimeFrameIdSeed,
};
use serde_json::Value;

#[cfg(test)]
use crate::request_builder::prompt_plan::{PromptPlanBuildInput, build_prompt_plan};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum AnthropicThinkingMode {
    #[default]
    Disabled,
    Adaptive,
    Budget,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct AnthropicThinkingConfig {
    #[serde(default)]
    pub mode: AnthropicThinkingMode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub budget_tokens: Option<u64>,
}

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
    pub parallel_tool_calls: bool,
    pub fast_mode: bool,
    pub anthropic_thinking: AnthropicThinkingConfig,
    pub anthropic_betas: Vec<String>,
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
            let mut efforts = DEFAULT_REASONING_EFFORTS.to_vec();
            if let Some(effort) = &self.reasoning_effort
                && !efforts.contains(effort)
            {
                efforts.push(effort.clone());
            }
            return efforts;
        }

        self.reasoning_efforts.clone()
    }

    pub fn allows_reasoning_effort(&self, effort: &ModelReasoningEffort) -> bool {
        self.selectable_reasoning_efforts().contains(effort)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum ModelReasoningEffort {
    None,
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
    Max,
    Custom(String),
}

impl ModelReasoningEffort {
    pub fn as_str(&self) -> &str {
        match self {
            Self::None => "none",
            Self::Minimal => "minimal",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Xhigh => "xhigh",
            Self::Max => "max",
            Self::Custom(value) => value,
        }
    }
}

impl Serialize for ModelReasoningEffort {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for ModelReasoningEffort {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Ok(match value.as_str() {
            "none" => Self::None,
            "minimal" => Self::Minimal,
            "low" => Self::Low,
            "medium" => Self::Medium,
            "high" => Self::High,
            "xhigh" => Self::Xhigh,
            "max" => Self::Max,
            _ => Self::Custom(value),
        })
    }
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
    SkillMaterial,
    RuntimeClock,
    WorkflowTurn,
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
    #[cfg(test)]
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
    pub model_id: &'a str,
    pub model: ModelRequestMetadata,
    pub prelude: &'a [PromptMessage],
    pub snapshot: &'a RuntimeSnapshot,
    pub tools: &'a [ToolSpec],
}

/// Configuration-normalized policy for provider-only current-turn pressure relief.
/// The dynamic default reserves 20% of the input budget, bounded to 256..65,536
/// tokens; zero explicitly selects reactive-only pressure handling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ProtectedContextPolicy {
    pub reserve_tokens: u64,
}

impl ProtectedContextPolicy {
    pub(crate) fn from_configured_reserve(configured: Option<u64>, input_budget: u64) -> Self {
        Self {
            reserve_tokens: configured
                .map(|reserve| reserve.min(input_budget))
                .unwrap_or_else(|| {
                    input_budget
                        .saturating_div(5)
                        .clamp(256, 65_536)
                        .min(input_budget)
                }),
        }
    }

    fn enabled(self) -> bool {
        self.reserve_tokens > 0
    }
}

/// Provider-visible evidence rendering fixed for one logical turn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FrozenEvidence {
    pub message: Option<String>,
    pub selected_ids: Vec<String>,
}

/// Test-only fixture input that is converted to a runtime snapshot.
#[cfg(test)]
#[derive(Debug, Clone)]
pub(crate) struct TestRequestBuilderInput<'a> {
    pub model_id: &'a str,
    pub model: ModelRequestMetadata,
    pub prelude: &'a [PromptMessage],
    pub history: &'a [HistoryItem],
    pub protected_start_index: usize,
    pub tools: &'a [ToolSpec],
    pub evidence: &'a [EvidenceRecord],
}

#[derive(Debug, Clone)]
pub(crate) struct SelectedPromptRequestInput {
    pub prompt_plan: PromptPlan,
    pub budget: BudgetReport,
    pub selected_evidence_ids: Vec<String>,
    pub selected_evidence_message: Option<String>,
}

/// Fixed reserve for the response budget when admitting a request.
pub const EMERGENCY_RESERVE_TOKENS: u64 = 2_048;

/// Final request admission derived from the assembled prompt budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RequestBudgetClassification {
    pub prompt_limit: u64,
    pub reserve: u64,
    pub hard_request_limit: u64,
    pub high_watermark: u64,
    pub safe: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BudgetReport {
    pub context_window_tokens: u64,
    pub input_budget_tokens: u64,
    pub estimated_request_tokens: u64,
    pub estimated_prelude_tokens: u64,
    pub estimated_protected_tokens: u64,
    pub protected_safe_ceiling_tokens: u64,
    pub protected_reserve_tokens: u64,
    pub estimated_unaddressable_protected_tokens: u64,
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

impl BudgetReport {
    /// Classifies the final assembled request.
    ///
    /// - `safe`: hard admission gate (request tokens <= hard_request_limit)
    /// - `high_watermark`: soft advisory used only to *trigger* pressure compact
    pub fn request_classification(self) -> RequestBudgetClassification {
        let prompt_limit = self.input_budget_tokens;
        let reserve = EMERGENCY_RESERVE_TOKENS.min(prompt_limit);
        let hard_request_limit = prompt_limit.saturating_add(self.estimated_tools_tokens);
        // high_watermark remains as a soft advisory for telemetry/auto-checkpoint
        // hints. Admission itself is single-threshold: under the hard limit.
        let high_watermark = prompt_limit
            .saturating_sub(reserve)
            .saturating_add(self.estimated_tools_tokens);
        RequestBudgetClassification {
            prompt_limit,
            reserve,
            hard_request_limit,
            high_watermark,
            safe: self.estimated_request_tokens <= hard_request_limit,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BuildResult {
    pub budget: BudgetReport,
    #[allow(dead_code)]
    pub prompt_plan: PromptPlan,
    #[allow(dead_code)]
    pub selected_evidence_ids: Vec<String>,
    pub selected_evidence_message: Option<String>,
}

/// Process-local, complete logical units used only to compare adjacent final
/// requests. They are deliberately never serialized or attached to telemetry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LogicalRequestObservation {
    pub cohort: LogicalRequestCohort,
    pub units: Vec<LogicalRequestUnit>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LogicalRequestCohort {
    /// Digest of the serialized request after its provider input/messages are
    /// removed. This is process-local; no provider route identity is available
    /// here beyond non-sensitive fields serialized into the request itself.
    pub request_shape_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LogicalRequestUnit {
    pub category: LogicalRequestUnitCategory,
    pub estimated_tokens: u64,
    pub byte_count: u64,
    pub digest: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LogicalRequestUnitCategory {
    StableKernel,
    RuntimeContext,
    History,
    User,
    Assistant,
    ToolCall,
    ToolOutput,
    Evidence,
    Composite,
}

/// The first non-shared boundary between two comparable provider requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LogicalRequestBreaker {
    CurrentUnit(LogicalRequestUnitCategory),
    RemovedSuffix,
}

impl LogicalRequestBreaker {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::CurrentUnit(category) => category.as_str(),
            Self::RemovedSuffix => "removed_suffix",
        }
    }
}

impl LogicalRequestUnitCategory {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::StableKernel => "stable_kernel",
            Self::RuntimeContext => "runtime_context",
            Self::History => "history",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::ToolCall => "tool_call",
            Self::ToolOutput => "tool_output",
            Self::Evidence => "evidence",
            Self::Composite => "composite",
        }
    }
}

/// Converts protocol-owned prepared-wire inspection into semantic adjacent-request units.
/// The request builder never parses protocol fields from the wire body.
pub(crate) fn observe_prepared_model_request(
    inspection: &crate::model_runtime::PreparedRequestInspection,
    prompt_plan: &PromptPlan,
) -> Result<LogicalRequestObservation> {
    let semantic_categories = prompt_plan
        .segments
        .iter()
        .map(|segment| (segment.id.as_str(), prompt_segment_category(segment)))
        .collect::<std::collections::HashMap<_, _>>();
    let units = inspection
        .prompt_units
        .iter()
        .map(|unit| {
            let categories = unit
                .semantic_segment_ids
                .iter()
                .map(|id| {
                    semantic_categories
                        .get(id.as_str())
                        .copied()
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "prepared request references unknown prompt segment '{id}'"
                            )
                        })
                })
                .collect::<Result<Vec<_>>>()?;
            let category = match categories.first().copied() {
                None => anyhow::bail!("prepared request unit has no semantic prompt origin"),
                Some(category) if categories.iter().all(|candidate| *candidate == category) => {
                    category
                }
                Some(_) => LogicalRequestUnitCategory::Composite,
            };
            Ok(LogicalRequestUnit {
                category,
                estimated_tokens: (unit.identity.len() as u64).div_ceil(4),
                byte_count: unit.identity.len() as u64,
                digest: sha256_hex(&unit.identity),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(LogicalRequestObservation {
        cohort: LogicalRequestCohort {
            request_shape_digest: sha256_hex(&inspection.request_shape),
        },
        units,
    })
}

pub(crate) fn observe_logical_request(build: &BuildResult) -> LogicalRequestObservation {
    observe_prompt_segments(&build.prompt_plan.segments)
}

fn observe_prompt_segments(segments: &[prompt_plan::PromptSegment]) -> LogicalRequestObservation {
    let units = segments
        .iter()
        .map(|segment| {
            let serialized = serde_json::to_vec(&serde_json::json!({
                "role": segment.role,
                "content": segment.content,
                "source": segment.source.contributor_kind,
            }))
            .expect("prompt segment serializes");
            LogicalRequestUnit {
                category: prompt_segment_category(segment),
                estimated_tokens: segment.tokens.budget_input_tokens.unwrap_or(0),
                byte_count: serialized.len() as u64,
                digest: sha256_hex(&serialized),
            }
        })
        .collect();
    let shape = segments
        .iter()
        .map(|segment| {
            serde_json::json!({
                "role": segment.role,
                "content_kind": match &segment.content {
                    PromptSegmentContent::Text { .. } => "text",
                    PromptSegmentContent::UserContent { .. } => "user_content",
                    PromptSegmentContent::AssistantToolCalls { .. } => "assistant_tool_calls",
                    PromptSegmentContent::ToolOutput { .. } => "tool_output",
                },
            })
        })
        .collect::<Vec<_>>();
    LogicalRequestObservation {
        cohort: LogicalRequestCohort {
            request_shape_digest: sha256_hex(
                &serde_json::to_vec(&shape).expect("prompt shape serializes"),
            ),
        },
        units,
    }
}

#[cfg(test)]
pub(crate) fn provider_unit_prefix_digest(build: &BuildResult, segment_count: usize) -> String {
    let end = segment_count.min(build.prompt_plan.segments.len());
    observe_prompt_segments(&build.prompt_plan.segments[..end])
        .units
        .into_iter()
        .fold(String::new(), |mut digest, unit| {
            digest.push_str(&unit.digest);
            digest
        })
}

fn prompt_segment_category(segment: &prompt_plan::PromptSegment) -> LogicalRequestUnitCategory {
    match segment.source.contributor_kind {
        PromptContributorKind::SystemPrelude
        | PromptContributorKind::DeveloperPrelude
        | PromptContributorKind::SkillMaterial => LogicalRequestUnitCategory::StableKernel,
        PromptContributorKind::RuntimeContext
        | PromptContributorKind::ContextMaterial
        | PromptContributorKind::ContextIndex => LogicalRequestUnitCategory::RuntimeContext,
        PromptContributorKind::Evidence => LogicalRequestUnitCategory::Evidence,
        PromptContributorKind::TranscriptFrame
        | PromptContributorKind::CurrentTurn
        | PromptContributorKind::Other => match &segment.content {
            PromptSegmentContent::AssistantToolCalls { calls, .. } if !calls.is_empty() => {
                LogicalRequestUnitCategory::ToolCall
            }
            PromptSegmentContent::ToolOutput { .. } => LogicalRequestUnitCategory::ToolOutput,
            _ => match segment.role {
                PromptSegmentRole::User => LogicalRequestUnitCategory::User,
                PromptSegmentRole::Assistant => LogicalRequestUnitCategory::Assistant,
                PromptSegmentRole::System | PromptSegmentRole::Developer => {
                    LogicalRequestUnitCategory::History
                }
                PromptSegmentRole::Tool => LogicalRequestUnitCategory::ToolOutput,
            },
        },
    }
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

#[cfg(test)]
pub fn build_request(input: RequestBuilderInput<'_>) -> Result<BuildResult> {
    build_request_with_policy(input, None, None)
}

pub(crate) fn build_request_with_policy(
    input: RequestBuilderInput<'_>,
    frozen_evidence: Option<&FrozenEvidence>,
    policy: Option<ProtectedContextPolicy>,
) -> Result<BuildResult> {
    build_request_with_frozen_and_policy(input, frozen_evidence, policy)
}

#[cfg(test)]
pub(crate) fn build_request_with_frozen(
    input: RequestBuilderInput<'_>,
    frozen_evidence: Option<&FrozenEvidence>,
) -> Result<BuildResult> {
    build_request_with_frozen_and_policy(input, frozen_evidence, None)
}

fn build_request_with_frozen_and_policy(
    input: RequestBuilderInput<'_>,
    frozen_evidence: Option<&FrozenEvidence>,
    policy: Option<ProtectedContextPolicy>,
) -> Result<BuildResult> {
    let protected_context_policy = policy.unwrap_or_else(|| {
        ProtectedContextPolicy::from_configured_reserve(
            None,
            effective_input_budget_tokens(input.model.clone(), input.tools),
        )
    });
    let planner_input = PromptPlannerInput {
        model: input.model.clone(),
        model_id: input.model_id,
        prelude: input.prelude,
        snapshot: input.snapshot,
        tools: input.tools,
        frozen_evidence,
        protected_context_policy,
    };
    let PlannedPrompt {
        prompt_plan,
        budget,
        selected_evidence_ids,
        selected_evidence_message,
    } = PromptPlanner::plan(planner_input)?;
    let prompt_plan = prompt_plan::canonicalize_prompt_plan(prompt_plan);
    build_request_from_selected_prompt(SelectedPromptRequestInput {
        prompt_plan,
        budget,
        selected_evidence_ids,
        selected_evidence_message,
    })
}

/// Test-only fixture adapter. It always delegates to the canonical runtime path.
#[cfg(test)]
pub(crate) fn build_test_request(input: TestRequestBuilderInput<'_>) -> Result<BuildResult> {
    let prelude = input.prelude.to_vec();
    let frames = history_items_to_frames(input.history);
    let protected_start_index = input.protected_start_index;

    let mut snapshot = RuntimeSnapshot::new("test-request-builder");
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
        model_id: input.model_id,
        model: input.model,
        prelude: &prelude,
        snapshot: &snapshot,
        tools: input.tools,
    })
}

#[cfg(test)]
fn runtime_frame_kind(item: &ProtocolFrameItem) -> RuntimeFrameKind {
    match item {
        ProtocolFrameItem::ContextSummary { .. } => RuntimeFrameKind::Summary,
        ProtocolFrameItem::UserMessage { .. } => RuntimeFrameKind::User,
        ProtocolFrameItem::InternalContinuation { .. } => RuntimeFrameKind::Reasoning,
        ProtocolFrameItem::AssistantTurn { calls, .. } if calls.is_empty() => {
            RuntimeFrameKind::Assistant
        }
        ProtocolFrameItem::AssistantTurn { .. } => RuntimeFrameKind::ToolCall,
        ProtocolFrameItem::ToolOutput { .. } => RuntimeFrameKind::ToolOutput,
    }
}

pub(crate) fn build_request_from_selected_prompt(
    mut input: SelectedPromptRequestInput,
) -> Result<BuildResult> {
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
    Ok(BuildResult {
        budget: input.budget,
        prompt_plan: input.prompt_plan,
        selected_evidence_ids: input.selected_evidence_ids,
        selected_evidence_message: input.selected_evidence_message,
    })
}

pub(crate) fn provider_visible_protocol_frames(snapshot: &RuntimeSnapshot) -> Vec<ProtocolFrame> {
    runtime_projection::provider_visible_protocol_frames(snapshot)
}

fn protected_start_index_for_snapshot(
    snapshot: &RuntimeSnapshot,
    frames: &[ProtocolFrame],
) -> usize {
    runtime_projection::protected_start_index_for_snapshot(snapshot, frames)
}

fn ensure_protected_context_within_budget(
    input_budget: u64,
    prelude_tokens: u64,
    protected_tokens: u64,
    evidence_tokens: u64,
) -> Result<()> {
    history_budget::ensure_protected_context_within_budget(
        input_budget,
        prelude_tokens,
        protected_tokens,
        evidence_tokens,
    )
}

fn validate_model_metadata(model: ModelRequestMetadata) -> Result<()> {
    history_budget::validate_model_metadata(model)
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
    protected_tokens: u64,
    model: ModelRequestMetadata,
    tools: &[ToolSpec],
    evidence_budget: EvidenceBudgetReport,
    required_fallback_tokens: u64,
) -> (Vec<ProtocolFrame>, BudgetReport) {
    history_budget::retain_history(
        prelude,
        history,
        protected_start_index,
        protected_tokens,
        model,
        tools,
        evidence_budget,
        required_fallback_tokens,
    )
}

fn current_user_query(history: &[HistoryItem], protected_start_index: usize) -> String {
    history_budget::current_user_query(history, protected_start_index)
}

fn evidence_budget_tokens(context_window_tokens: u64) -> u64 {
    history_budget::evidence_budget_tokens(context_window_tokens)
}

/// Small local SHA-256 implementation to keep fingerprinting dependency-free.
pub(crate) fn sha256_hex(input: &[u8]) -> String {
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

pub(crate) fn estimate_history_item_tokens(item: &HistoryItem) -> u64 {
    if let HistoryItem::ToolOutput {
        call_id,
        output_json,
        images,
    } = item
        && !images.is_empty()
    {
        let compact_text = format!(
            "{call_id}\n{output_json}\n{}",
            images
                .iter()
                .map(crate::user_content::UserImageAttachment::prompt_plan_placeholder)
                .collect::<Vec<_>>()
                .join("\n")
        );
        let text_tokens = (compact_text.len() as u64).div_ceil(3).saturating_add(8);
        let visual_tokens = images
            .iter()
            .map(crate::user_content::UserImageAttachment::visual_token_charge)
            .sum::<u64>();
        return text_tokens.saturating_add(visual_tokens);
    }

    if let HistoryItem::UserMessage { content } = item
        && !content.attachments.is_empty()
    {
        // Image data URLs are provider transport data, not prompt text. Match
        // the prompt plan's compact attachment markers, then separately
        // account for provider auto-detail visual input by image dimensions.
        let compact_item = HistoryItem::user(content.prompt_plan_text());
        let json_len = serde_json::to_string(&compact_item)
            .map(|serialized| serialized.len())
            .unwrap_or(0);
        let text_tokens = (json_len as u64).div_ceil(3).saturating_add(8);
        let visual_tokens = content
            .attachments
            .iter()
            .map(crate::user_content::UserImageAttachment::visual_token_charge)
            .sum::<u64>();
        return text_tokens.saturating_add(visual_tokens);
    }

    let json_len = serde_json::to_string(item).map(|s| s.len()).unwrap_or(0);
    (json_len as u64).div_ceil(3).saturating_add(8)
}

pub(crate) fn estimate_history_tokens(items: &[HistoryItem]) -> u64 {
    items.iter().map(estimate_history_item_tokens).sum()
}

fn estimate_prelude_tokens(items: &[PromptMessage]) -> u64 {
    items
        .iter()
        .map(|item| {
            let json_len = serde_json::to_string(item).map(|s| s.len()).unwrap_or(0);
            (json_len as u64).div_ceil(3).saturating_add(8)
        })
        .sum()
}

fn estimate_tools_tokens(tools: &[ToolSpec]) -> u64 {
    if tools.is_empty() {
        return 0;
    }
    let json_len = serde_json::to_string(tools).map(|s| s.len()).unwrap_or(0);
    (json_len as u64).div_ceil(3).saturating_add(16)
}

#[cfg(test)]
#[path = "request_builder/tests.rs"]
mod tests;
