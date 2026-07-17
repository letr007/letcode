use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::agent::{
    AutoContinueState, ContextCompactionEvent, TodoItem, ToolExecutionSummaryEvent,
    TurnFinalizedEvent, TurnStartedEvent, ValidationAdvisory,
};
use crate::context_tree::{ContextBlockRef, ContextNodeStatus, ContextSourceRef};
use crate::evidence::{EvidenceKind, EvidenceSource};
use crate::request_builder::HistoryToolCall;
use crate::tool::ToolResult;
use crate::user_content::UserMessageContent;

use super::{InternalContinuationSource, LogicalCheckpointEventV1};

fn default_usage_completeness() -> String {
    // Records written before completeness existed cannot distinguish an absent
    // provider usage object from an older schema.
    "legacy_unknown".into()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptRecord {
    pub session_id: String,
    pub sequence: u64,
    pub timestamp_ms: u128,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_branch_id: Option<String>,
    #[serde(flatten)]
    pub event: TranscriptEvent,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TranscriptEvent {
    SessionStarted {
        model: String,
    },
    SessionTitle {
        title: String,
    },
    SubagentLifecycle {
        run_id: String,
        parent_session_id: String,
        parent_run_id: String,
        agent_name: String,
        status: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        detail: Option<String>,
    },
    SubagentResult {
        run_id: String,
        parent_session_id: String,
        parent_run_id: String,
        child_session_id: String,
        agent_name: String,
        status: String,
        summary: String,
    },
    ModelChanged {
        previous_model: String,
        new_model: String,
    },
    ContextBranchCreated {
        branch_id: String,
        parent_branch_id: String,
        base_sequence: u64,
        #[serde(skip_serializing_if = "Option::is_none")]
        label: Option<String>,
    },
    ContextBranchSummary {
        branch_id: String,
        leaf_sequence: u64,
        summary: String,
    },
    ContextCheckout {
        branch_id: String,
        leaf_sequence: u64,
    },
    ContextExperimentStarted {
        branch_id: String,
        parent_branch_id: String,
        base_sequence: u64,
    },
    ContextNodeCreated {
        node_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        parent_node_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        label: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        purpose: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        block_ref: Option<ContextBlockRef>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        source_ref: Option<ContextSourceRef>,
    },
    ContextNodeLifecycle {
        node_id: String,
        status: ContextNodeStatus,
    },
    ContextViewOperationMetadata {
        operation: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        node_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        block_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        detail: Option<String>,
    },
    ContextSummaryArtifactMetadata {
        node_id: String,
        artifact_id: String,
        artifact_kind: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        version: Option<u32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        summary: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        source_node_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        source_block_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        source_start_sequence: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        source_end_sequence: Option<u64>,
    },
    FoldedOutputMetadata {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        node_id: Option<String>,
        output_id: String,
        output_kind: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        call_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tool_name: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        stream: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        content: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        byte_count: Option<usize>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        line_count: Option<usize>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        truncated: Option<bool>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        shell_command: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        source_start_sequence: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        source_end_sequence: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tool_ok: Option<bool>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        exit_status: Option<i32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        provider_metadata: Option<Value>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        provider_fold_eligible: Option<bool>,
    },
    UserMessage {
        content: UserMessageContent,
    },
    TurnStarted(TurnStartedEvent),
    LlmRequestTelemetry {
        version: u32,
        logical_request_id: String,
        turn_id: u64,
        iteration: usize,
        attempt: usize,
        phase: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        error_class: Option<String>,
        model: String,
        protocol: String,
        context_window_tokens: u64,
        input_budget_tokens: u64,
        estimated_request_tokens: u64,
        estimated_prelude_tokens: u64,
        estimated_protected_tokens: u64,
        #[serde(default)]
        protected_safe_ceiling_tokens: u64,
        #[serde(default)]
        protected_reserve_tokens: u64,
        #[serde(default)]
        estimated_foldable_protected_tokens: u64,
        #[serde(default)]
        estimated_provider_folded_protected_tokens: u64,
        #[serde(default)]
        estimated_unaddressable_protected_tokens: u64,
        #[serde(default)]
        provider_folded_output_count: usize,
        estimated_retained_history_tokens: u64,
        estimated_tools_tokens: u64,
        estimated_evidence_tokens: u64,
        estimated_required_fallback_tokens: u64,
        original_history_items: usize,
        retained_history_items: usize,
        dropped_history_items: usize,
        selected_evidence_items: usize,
        dropped_evidence_items: usize,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        selected_evidence_ids: Vec<String>,
        #[serde(default, skip_serializing_if = "String::is_empty")]
        evidence_fingerprint: String,
        truncated: bool,
        prompt_segment_count: usize,
        prompt_contributor_count: usize,
        #[serde(skip_serializing_if = "Option::is_none")]
        prompt_stable_prefix_hash: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_first_volatile_index: Option<usize>,
        plan_total_prompt_tokens: u64,
        plan_stable_prompt_tokens: u64,
        plan_volatile_prompt_tokens: u64,
        plan_cacheable_prefix_tokens: u64,
        plan_stable_after_boundary_tokens: u64,
        cache_configured: bool,
        cache_hint_serialized: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_retention_sent: Option<String>,
        cache_stable_prefix_segments: usize,
        cache_stable_prompt_tokens: u64,
        cache_volatile_prompt_tokens: u64,
        cacheable_prefix_tokens: u64,
        cache_stable_after_boundary_tokens: u64,
        tool_call_count_before: usize,
        tool_definitions_count: usize,
        #[serde(skip_serializing_if = "Option::is_none")]
        local_prefix_fingerprint: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        routing_key: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_cached_tokens: Option<u64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_input_tokens: Option<u64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_output_tokens: Option<u64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_total_tokens: Option<u64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_response_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        adjacent_lcp_units: Option<usize>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        adjacent_lcp_bytes: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        adjacent_lcp_estimated_tokens: Option<u64>,
        #[serde(default)]
        current_unit_count: usize,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        first_breaker: Option<String>,
        #[serde(default)]
        cohort_comparable: bool,
        #[serde(default)]
        cohort_changed: bool,
        #[serde(default = "default_usage_completeness")]
        usage_completeness: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_write_tokens: Option<u64>,
    },
    AssistantMessage {
        content: String,
    },
    ReasoningMessage {
        content: String,
    },
    AssistantToolCallBatch {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        text: Option<String>,
        calls: Vec<HistoryToolCall>,
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
    ToolCallCancelled {
        call_id: String,
        name: String,
    },
    PermissionDecision {
        #[serde(skip_serializing_if = "Option::is_none")]
        call_id: Option<String>,
        tool: String,
        args: Value,
        allowed: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
    },
    PermissionModeChanged {
        previous_mode: String,
        new_mode: String,
    },
    TodoSnapshot {
        items: Vec<TodoItem>,
    },
    AutoContinueChanged {
        state: AutoContinueState,
    },
    AutoContinuationScheduled {
        continuation_count: usize,
        remaining_unfinished: usize,
    },
    InternalContinuation {
        text: String,
        #[serde(default)]
        source: InternalContinuationSource,
    },
    ValidationAdvisory(ValidationAdvisory),
    ToolExecutionSummary(ToolExecutionSummaryEvent),
    ContextCompaction(ContextCompactionEvent),
    LogicalCheckpoint(LogicalCheckpointEventV1),
    TurnFinalized(TurnFinalizedEvent),
    TurnInterrupted {
        #[serde(skip_serializing_if = "Option::is_none")]
        turn_id: Option<u64>,
    },
    Evidence {
        id: String,
        evidence_kind: EvidenceKind,
        title: String,
        summary: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        detail: Option<String>,
        source: EvidenceSource,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        tags: Vec<String>,
    },
    ContextExperimentReturned {
        branch_id: String,
        parent_branch_id: String,
        base_sequence: u64,
        outcome: String,
        summary: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        next_action: Option<String>,
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        had_writes: bool,
    },
    Error {
        message: String,
    },
    #[serde(other)]
    Unknown,
}
