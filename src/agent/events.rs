use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{AdjacentRequestObservation, AutoContinueState, TodoItem};
use crate::config::ApiProtocol;
use crate::evidence::EvidenceRecord;
use crate::request_builder::HistoryToolCall;
use crate::tool::{ToolOutputStream, ToolResult};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValidationAdvisory {
    pub write_effects: usize,
    pub validation_effects: usize,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub failed_validation_effects: usize,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TurnStartedEvent {
    pub turn_id: u64,
    pub intent: String,
    pub directive: String,
    pub validation_reminder: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TurnFinalizedEvent {
    pub turn_id: u64,
    pub outcome: String,
    pub tool_call_count: usize,
    pub continuation_count: usize,
    pub write_effects: usize,
    pub validation_effects: usize,
    pub failed_validation_effects: usize,
    pub validation_advisory_emitted: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolExecutionSummaryEvent {
    pub turn_id: u64,
    pub call_id: String,
    pub name: String,
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rejection: Option<String>,
    pub effect_kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub primary_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompactionFileOperations {
    pub read_files: Vec<String>,
    pub modified_files: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompactionCheckpoint {
    /// The exact first actionable item from the checkpoint's Next Steps section.
    #[serde(default)]
    pub next_action: String,
    /// An internal, provider-facing instruction installed after the summary.
    pub continuation: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub split_turn_handoff: Option<String>,
    #[serde(default)]
    pub file_operations: CompactionFileOperations,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextCompactionEvent {
    pub summary: String,
    /// Stable identity of the first entry retained from the pre-compaction
    /// projection. New events require this append-only anchor instead of a live
    /// history index.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_kept_entry_id: Option<String>,
    /// Legacy replay boundary. New events omit this field and use
    /// `first_kept_entry_id`; existing JSONL records remain deserializable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tail_start_index: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checkpoint: Option<CompactionCheckpoint>,
}

impl ContextCompactionEvent {
    /// Constructs a legacy index-based event for compatibility callers.
    pub fn succeeded(summary: impl Into<String>, tail_start_index: usize) -> Self {
        Self {
            summary: summary.into(),
            first_kept_entry_id: None,
            tail_start_index: Some(tail_start_index),
            checkpoint: None,
        }
    }

    /// Constructs a legacy index-based event for compatibility callers.
    #[cfg(test)]
    pub fn checkpointed(
        summary: impl Into<String>,
        tail_start_index: usize,
        checkpoint: CompactionCheckpoint,
    ) -> Self {
        Self {
            summary: summary.into(),
            first_kept_entry_id: None,
            tail_start_index: Some(tail_start_index),
            checkpoint: Some(checkpoint),
        }
    }

    /// Constructs a modern append-only event. `None` compacts the complete
    /// safe projection and retains no raw suffix.
    pub fn succeeded_at(summary: impl Into<String>, first_kept_entry_id: Option<String>) -> Self {
        assert!(
            first_kept_entry_id
                .as_deref()
                .is_none_or(|entry_id| !entry_id.trim().is_empty()),
            "modern context compaction first_kept_entry_id must not be empty"
        );
        Self {
            summary: summary.into(),
            first_kept_entry_id,
            tail_start_index: None,
            checkpoint: None,
        }
    }
}

/// Why an ephemeral compaction attempt was started. This is intentionally not
/// part of the durable compaction record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompactionTrigger {
    Manual,
    RequestPressure,
}

/// Sanitized reasons that no safe historical prefix can be selected.
///
/// These labels are safe to render directly: they never carry transcript,
/// runtime identity, provider, filesystem, or tool data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompactionBlocker {
    IncompleteToolGroup,
    MissingSourceProvenance,
    NoHistoricalItems,
    NoSafeBoundary,
    ProtectedContext,
}

impl CompactionBlocker {
    pub const fn label(self) -> &'static str {
        match self {
            Self::IncompleteToolGroup => "incomplete_tool_group",
            Self::MissingSourceProvenance => "missing_source_provenance",
            Self::NoHistoricalItems => "no_historical_items",
            Self::NoSafeBoundary => "no_safe_boundary",
            Self::ProtectedContext => "protected_context",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompactionNoProgress {
    pub trigger: CompactionTrigger,
    /// Always sorted and deduplicated by the selector.
    pub blockers: Vec<CompactionBlocker>,
}

/// The non-error terminal outcome of a compaction attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompactionAttemptOutcome {
    Compacted { retained_items: usize },
    NoProgress(CompactionNoProgress),
}

/// Backwards-compatible name for the manual attempt outcome.
pub type ManualCompactionOutcome = CompactionAttemptOutcome;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenUsageEstimate {
    pub used_tokens: u64,
    pub context_window_tokens: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cached_tokens: u64,
}

/// Whether provider usage and its cache-detail subobject were actually
/// supplied. `cached_tokens == 0` remains compatible with existing callers,
/// while this status keeps an absent detail distinct from an explicit zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderUsageCompleteness {
    Complete,
    UsageMissing,
    CacheDetailsMissing,
}

impl ProviderUsageCompleteness {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Complete => "complete",
            Self::UsageMissing => "usage_missing",
            Self::CacheDetailsMissing => "cache_details_missing",
        }
    }
}

/// Cache telemetry for the exact request represented by a token-usage event.
/// Configuration and serialized hints describe request intent; only
/// `actual_cached_tokens` describes a provider-reported cache hit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheUsageReport {
    pub configured: bool,
    pub hint_serialized: bool,
    pub retention_sent: Option<crate::config::PromptCacheRetention>,
    pub stable_prefix_segments: usize,
    pub stable_prompt_tokens: u64,
    pub volatile_prompt_tokens: u64,
    pub cacheable_prefix_tokens: u64,
    pub stable_after_boundary_tokens: u64,
    pub local_prefix_fingerprint: Option<String>,
    pub routing_key: Option<String>,
    pub actual_cached_tokens: Option<u64>,
}

pub type PromptCompositionEntry = crate::request_builder::prompt_plan::PromptCompositionEntry;

/// Safe, scalar-only provenance for one attempt of a logical model request.
/// Prompt, tool, evidence, and provider payloads deliberately do not cross this boundary.
#[derive(Debug, Clone)]
pub struct LlmRequestTelemetry {
    pub logical_request_id: String,
    pub turn_id: u64,
    pub iteration: usize,
    pub attempt: usize,
    pub phase: LlmRequestTelemetryPhase,
    pub model: String,
    pub protocol: ApiProtocol,
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
    pub selected_evidence_ids: Vec<String>,
    pub evidence_fingerprint: String,
    pub truncated: bool,
    pub prompt_segment_count: usize,
    pub prompt_contributor_count: usize,
    pub prompt_composition: Vec<PromptCompositionEntry>,
    pub prompt_stable_prefix_hash: Option<String>,
    pub cache_first_volatile_index: Option<usize>,
    pub cache_configured: bool,
    pub cache_hint_serialized: bool,
    pub cache_retention_sent: Option<crate::config::PromptCacheRetention>,
    pub cache_stable_prefix_segments: usize,
    pub cache_stable_prompt_tokens: u64,
    pub cache_volatile_prompt_tokens: u64,
    pub cacheable_prefix_tokens: u64,
    pub cache_stable_after_boundary_tokens: u64,
    pub local_prefix_fingerprint: Option<String>,
    pub routing_key: Option<String>,
    pub tool_call_count_before: usize,
    pub tool_definitions_count: usize,
    pub adjacent_lcp_units: Option<usize>,
    pub adjacent_lcp_bytes: Option<u64>,
    pub adjacent_lcp_estimated_tokens: Option<u64>,
    pub current_unit_count: usize,
    pub first_breaker: Option<crate::request_builder::LogicalRequestBreaker>,
    pub cohort_comparable: bool,
    pub cohort_changed: bool,
    pub usage: Option<TokenUsageEstimate>,
    pub usage_completeness: ProviderUsageCompleteness,
    pub cache_write_tokens: Option<u64>,
    pub provider_response_id: Option<String>,
    /// Categorical only. Never contains an error body or provider message.
    pub error_class: Option<LlmRequestErrorClass>,
}

impl LlmRequestTelemetry {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prepared_from_build(
        logical_request_id: String,
        turn_id: u64,
        iteration: usize,
        attempt: usize,
        model: String,
        protocol: ApiProtocol,
        build: &crate::request_builder::BuildResult,
        tool_call_count_before: usize,
        tool_definitions_count: usize,
        observation: AdjacentRequestObservation,
    ) -> Self {
        let budget = build.budget;
        let cache = CacheUsageReport::from_build(build);
        let plan = build.prompt_plan.token_report();
        Self {
            logical_request_id,
            turn_id,
            iteration,
            attempt,
            phase: LlmRequestTelemetryPhase::Prepared,
            model,
            protocol,
            context_window_tokens: budget.context_window_tokens,
            input_budget_tokens: budget.input_budget_tokens,
            estimated_request_tokens: budget.estimated_request_tokens,
            estimated_prelude_tokens: budget.estimated_prelude_tokens,
            estimated_protected_tokens: budget.estimated_protected_tokens,
            protected_safe_ceiling_tokens: budget.protected_safe_ceiling_tokens,
            protected_reserve_tokens: budget.protected_reserve_tokens,
            estimated_unaddressable_protected_tokens: budget
                .estimated_unaddressable_protected_tokens,
            estimated_retained_history_tokens: budget.estimated_retained_history_tokens,
            estimated_tools_tokens: budget.estimated_tools_tokens,
            estimated_evidence_tokens: budget.estimated_evidence_tokens,
            estimated_required_fallback_tokens: budget.estimated_required_fallback_tokens,
            original_history_items: budget.original_history_items,
            retained_history_items: budget.retained_history_items,
            dropped_history_items: budget.dropped_history_items,
            selected_evidence_items: budget.selected_evidence_items,
            dropped_evidence_items: budget.dropped_evidence_items,
            selected_evidence_ids: build.selected_evidence_ids.clone(),
            evidence_fingerprint: evidence_fingerprint(build.selected_evidence_message.as_deref()),
            truncated: budget.truncated,
            prompt_segment_count: build.prompt_plan.segments.len(),
            prompt_contributor_count: build.prompt_plan.contributors.len(),
            prompt_composition: build.prompt_plan.composition(budget.estimated_tools_tokens),
            prompt_stable_prefix_hash: build.prompt_plan.stable_prefix_hash().map(str::to_owned),
            cache_first_volatile_index: plan.first_volatile_index,
            cache_configured: cache.configured,
            cache_hint_serialized: cache.hint_serialized,
            cache_retention_sent: cache.retention_sent,
            cache_stable_prefix_segments: cache.stable_prefix_segments,
            cache_stable_prompt_tokens: cache.stable_prompt_tokens,
            cache_volatile_prompt_tokens: cache.volatile_prompt_tokens,
            cacheable_prefix_tokens: cache.cacheable_prefix_tokens,
            cache_stable_after_boundary_tokens: cache.stable_after_boundary_tokens,
            local_prefix_fingerprint: cache.local_prefix_fingerprint,
            routing_key: cache.routing_key,
            tool_call_count_before,
            tool_definitions_count,
            adjacent_lcp_units: observation
                .cohort_comparable
                .then_some(observation.lcp_units),
            adjacent_lcp_bytes: observation
                .cohort_comparable
                .then_some(observation.lcp_bytes),
            adjacent_lcp_estimated_tokens: observation
                .cohort_comparable
                .then_some(observation.lcp_estimated_tokens),
            current_unit_count: observation.current_unit_count,
            first_breaker: observation.first_breaker,
            cohort_comparable: observation.cohort_comparable,
            cohort_changed: observation.cohort_changed,
            usage: None,
            usage_completeness: ProviderUsageCompleteness::UsageMissing,
            cache_write_tokens: None,
            provider_response_id: None,
            error_class: None,
        }
    }

    pub(crate) fn completed(
        &self,
        usage: Option<TokenUsageEstimate>,
        provider_response_id: Option<String>,
        usage_completeness: ProviderUsageCompleteness,
    ) -> Self {
        let mut telemetry = self.clone();
        telemetry.phase = LlmRequestTelemetryPhase::Completed;
        telemetry.usage = usage;
        telemetry.provider_response_id = provider_response_id;
        telemetry.usage_completeness = usage_completeness;
        telemetry
    }

    pub(crate) fn failed(&self, error_class: LlmRequestErrorClass) -> Self {
        let mut telemetry = self.clone();
        telemetry.phase = LlmRequestTelemetryPhase::Failed;
        telemetry.error_class = Some(error_class);
        telemetry
    }

    pub(crate) fn interrupted(&self, error_class: LlmRequestErrorClass) -> Self {
        let mut telemetry = self.clone();
        telemetry.phase = LlmRequestTelemetryPhase::Interrupted;
        telemetry.error_class = Some(error_class);
        telemetry
    }
}

fn evidence_fingerprint(message: Option<&str>) -> String {
    let mut bytes = b"frozen-evidence-v1\0".to_vec();
    match message {
        Some(message) => {
            bytes.extend_from_slice(b"some\0");
            bytes.extend_from_slice(message.as_bytes());
        }
        None => bytes.extend_from_slice(b"none"),
    }
    format!("fte-v1-{}", crate::request_builder::sha256_hex(&bytes))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlmRequestTelemetryPhase {
    Prepared,
    Completed,
    Failed,
    Interrupted,
}

impl LlmRequestTelemetryPhase {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Prepared => "prepared",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Interrupted => "interrupted",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlmRetryLifecycle {
    pub attempt: usize,
    pub max_attempts: usize,
    pub delay_secs: u64,
    pub error: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlmRequestErrorClass {
    RequestCreation,
    StreamRead,
    ProtocolValidation,
    ProviderTerminal,
}

impl LlmRequestErrorClass {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::RequestCreation => "request_creation",
            Self::StreamRead => "stream_read",
            Self::ProtocolValidation => "protocol_validation",
            Self::ProviderTerminal => "provider_terminal",
        }
    }
}

impl CacheUsageReport {
    pub(crate) fn from_build(build: &crate::request_builder::BuildResult) -> Self {
        Self {
            configured: build.cache.configured,
            hint_serialized: build.cache.hint_serialized,
            retention_sent: build.cache.retention_sent,
            stable_prefix_segments: build.cache.local_prefix_segments,
            stable_prompt_tokens: build.budget.plan_stable_prompt_tokens,
            volatile_prompt_tokens: build.budget.plan_volatile_prompt_tokens,
            cacheable_prefix_tokens: build.budget.plan_cacheable_prefix_tokens,
            stable_after_boundary_tokens: build.budget.plan_stable_after_boundary_tokens,
            local_prefix_fingerprint: build.cache.local_prefix_fingerprint.clone(),
            routing_key: build.cache.routing_key.clone(),
            actual_cached_tokens: None,
        }
    }

    pub(crate) fn with_actual_cached_tokens(&self, actual_cached_tokens: u64) -> Self {
        let mut report = self.clone();
        report.actual_cached_tokens = Some(actual_cached_tokens);
        report
    }
}

#[derive(Debug, Clone)]
pub enum AgentEvent {
    TurnStarted(TurnStartedEvent),
    ContextCompactionStarted {
        #[allow(dead_code)]
        // Event payload retained for consumers that distinguish compaction triggers.
        trigger: CompactionTrigger,
    },
    ContextCompactionNoProgress(CompactionNoProgress),
    /// A technical failure occurred after `ContextCompactionStarted`. Details
    /// remain in the returned `Result`, never in this diagnostic event.
    ContextCompactionFailed {
        #[allow(dead_code)]
        // Event payload retained for consumers that distinguish compaction triggers.
        trigger: CompactionTrigger,
    },
    ContextCompactionDelta {
        delta: String,
    },
    TokenUsageUpdated {
        used_tokens: u64,
        context_window_tokens: u64,
        input_tokens: u64,
        output_tokens: u64,
        cached_tokens: u64,
        cache_report: Option<CacheUsageReport>,
    },
    LlmRequestTelemetry(LlmRequestTelemetry),
    LlmRetryScheduled(LlmRetryLifecycle),
    LlmRetryStarted(LlmRetryLifecycle),
    ReasoningDelta {
        item_id: String,
        delta: String,
    },
    ReasoningDone {
        item_id: String,
        text: String,
    },
    ModelStreamIssue {
        message: String,
        detail: Option<String>,
        action: String,
    },
    AssistantMessage {
        content: String,
    },
    AssistantToolCallBatch {
        text: Option<String>,
        reasoning_content: Option<String>,
        reasoning_wire: Option<String>,
        calls: Vec<HistoryToolCall>,
    },
    InternalContinuation {
        text: String,
        source: crate::transcript::InternalContinuationSource,
    },
    ToolCallPending {
        call_id: String,
        name: String,
    },
    ToolCallStarted {
        call_id: String,
        name: String,
        args: Value,
    },
    ToolCallCancelled {
        call_id: String,
        name: String,
    },
    ToolCallFinished {
        call_id: String,
        name: String,
        ok: bool,
        output: ToolResult,
    },
    ToolOutputDelta {
        call_id: String,
        stream: ToolOutputStream,
        chunk: String,
    },
    ToolCallBatchFinished,
    TodoSnapshotUpdated {
        items: Vec<TodoItem>,
    },
    AutoContinueChanged {
        state: AutoContinueState,
    },
    /// Global Fast Mode was persistently changed while preparing a request.
    /// This is a runtime projection only and is intentionally not journaled.
    FastModeChanged {
        enabled: bool,
    },
    TurnContinuationBoundary,
    AutoContinuationScheduled {
        continuation_count: usize,
        remaining_unfinished: usize,
    },
    ValidationAdvisory(ValidationAdvisory),
    ToolExecutionSummary(ToolExecutionSummaryEvent),
    ContextCompacted(ContextCompactionEvent),
    TurnFinalized(TurnFinalizedEvent),
    EvidenceRecorded(EvidenceRecord),
}

fn is_zero(value: &usize) -> bool {
    *value == 0
}
