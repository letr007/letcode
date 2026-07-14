use anyhow::{Context, Result, anyhow, bail, ensure};
use async_openai::Client;
use async_openai::config::Config;
use async_openai::error::OpenAIError;
use async_openai::types::chat::{
    ChatCompletionMessageToolCall, ChatCompletionMessageToolCallChunk, CompletionUsage,
    FinishReason,
};
use async_openai::types::responses::{OutputItem, Response, ResponseStreamEvent, ResponseUsage};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::future::Future;
use std::pin::Pin;
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::{debug, error, info, trace, warn};

use crate::config::{ApiProtocol, CompactionConfig, LogicalCheckpointConfig, RetryConfig};
use crate::evidence::{EvidenceDraft, EvidenceRecord, EvidenceSource, require_unique_evidence_id};
use crate::permission::{
    ExecutionDirective, PermissionApproval, PermissionDecision, PermissionMode, PermissionRequest,
    PermissionSessionState, ToolScope, restricted_by_directive_with_class,
};
use crate::request_builder::{
    BuiltRequest, HistoryItem, HistoryToolCall, ModelReasoningEffort, ModelRequestMetadata,
    PromptMessage, PromptMessageOrigin, ProtectedContextPolicy, RequestBuilderInput,
    build_request_with_policy, effective_input_budget_tokens, estimate_history_item_tokens,
};
use crate::retry::{
    can_retry_attempt, is_retryable_json_deserialize_error, retry_delay, should_retry_http_status,
    should_retry_openai_stream_creation, should_retry_openai_stream_read,
    should_retry_reqwest_error,
};
use crate::runtime_context::{
    FrameVisibility, RuntimeFrame, RuntimeFrameIdSeed, RuntimeFrameKind, RuntimeFrameProvenance,
    RuntimeSnapshot, RuntimeSource,
};
use crate::skills::{
    SkillCard, SkillRegistry, SkillResourceListTool, SkillResourceReadTool, SkillTool,
    reconcile_loaded_skill_material,
};
use crate::tool::{
    NormalizedSubagentInput, QuestionCallback, QuestionRequest, QuestionResponse,
    ToolExecutionContext, ToolHandler, ToolOutputStream, ToolRegistry, ToolResult,
    external_workspace_access_for_tool, normalize_subagent_input, subagent_parameters_schema,
};
use crate::tool_format::format_tool_call;
use crate::tool_names;
use crate::transcript::{
    ActiveContextExperiment, ContextScopeState, LogicalCheckpointEventV1, ROOT_CONTEXT_BRANCH_ID,
};
use crate::user_content::UserMessageContent;

#[path = "agent/automatic_checkpoint.rs"]
mod automatic_checkpoint;
#[path = "agent/compaction.rs"]
mod compaction;
#[path = "agent/evidence_memory.rs"]
mod evidence_memory;
#[path = "agent/logical_checkpoint.rs"]
mod logical_checkpoint;
#[path = "agent/protocol_stream.rs"]
mod protocol_stream;
#[path = "agent/tool_execution.rs"]
mod tool_execution;

#[cfg(test)]
use compaction::{
    compaction_history_char_budget, default_preserve_recent_budget, describe_history_item,
    render_bounded_compaction_history, render_compaction_prompt, select_compaction_segments,
};
#[cfg(test)]
use protocol_stream::{
    CompatibleChatCompletionStreamResponse, CompatibleChatCompletionStreamResponseDelta,
    append_sse_chunk, drain_sse_data_events, is_ignorable_response_lifecycle_event,
    project_response_stream_event, send_compatible_chat_completion_stream,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ToolExecutionStatus {
    Executed,
    Rejected,
    TimedOut,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ToolExecutionRejection {
    InvalidJsonArguments,
    DirectiveBlocked,
    ToolScopeDenied,
    PermissionDeniedByPolicy,
    PermissionDeniedByUser,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ToolEffectKind {
    Read,
    Write,
    Command,
    Validation,
    WorkflowControl,
    Diagnostic,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ToolEffects {
    pub kind: ToolEffectKind,
    pub primary_path: Option<String>,
    pub edited_paths: Vec<String>,
    pub command: Option<String>,
}

#[derive(Debug, Clone)]
pub(crate) struct ToolExecutionRecord {
    pub call_id: String,
    pub tool_name: String,
    pub arguments: Option<Value>,
    #[allow(dead_code)]
    pub permission_class: crate::permission::ToolPermissionClass,
    #[allow(dead_code)]
    pub directive: ExecutionDirective,
    #[allow(dead_code)]
    pub status: ToolExecutionStatus,
    #[allow(dead_code)]
    pub rejection: Option<ToolExecutionRejection>,
    pub output: ToolResult,
    pub effects: ToolEffects,
}

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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextCompactionSourceSpan {
    pub start_sequence: u64,
    pub end_sequence: u64,
}

/// Durable runtime identity assigned to a frame created before compaction.
/// The key is deliberately opaque so this public journal API does not expose
/// runtime-only frame enums.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextCompactionFrameBinding {
    pub key: String,
    pub frame_id: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextCompactionEvent {
    #[serde(default = "default_compaction_outcome")]
    pub outcome: String,
    pub summary: String,
    pub tail_start_index: usize,
    pub original_history_items: usize,
    pub retained_history_items: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub retired_source_spans: Vec<ContextCompactionSourceSpan>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub frame_identity_bindings: Vec<ContextCompactionFrameBinding>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ManualCompactionOutcome {
    Compacted { retained_items: usize },
    NothingToCompact,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenUsageEstimate {
    pub used_tokens: u64,
    pub context_window_tokens: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cached_tokens: u64,
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
    pub estimated_foldable_protected_tokens: u64,
    pub estimated_provider_folded_protected_tokens: u64,
    pub estimated_unaddressable_protected_tokens: u64,
    pub provider_folded_output_count: usize,
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
    pub usage: Option<TokenUsageEstimate>,
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
            estimated_foldable_protected_tokens: budget.estimated_foldable_protected_tokens,
            estimated_provider_folded_protected_tokens: budget
                .estimated_provider_folded_protected_tokens,
            estimated_unaddressable_protected_tokens: budget
                .estimated_unaddressable_protected_tokens,
            provider_folded_output_count: budget.provider_folded_output_count,
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
            usage: None,
            provider_response_id: None,
            error_class: None,
        }
    }

    pub(crate) fn completed(
        &self,
        usage: Option<TokenUsageEstimate>,
        provider_response_id: Option<String>,
    ) -> Self {
        let mut telemetry = self.clone();
        telemetry.phase = LlmRequestTelemetryPhase::Completed;
        telemetry.usage = usage;
        telemetry.provider_response_id = provider_response_id;
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
    ContextCompactionStarted,
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
    AutoContinuationScheduled {
        continuation_count: usize,
        remaining_unfinished: usize,
    },
    ValidationAdvisory(ValidationAdvisory),
    ToolExecutionSummary(ToolExecutionSummaryEvent),
    ContextCompacted(ContextCompactionEvent),
    LogicalCheckpoint {
        expected_journal_frontier: u64,
        expected_branch_id: String,
        event: LogicalCheckpointEventV1,
    },
    TurnFinalized(TurnFinalizedEvent),
    EvidenceRecorded(EvidenceRecord),
}

pub trait SubagentDelegate<C: Config>: Send + Sync {
    fn run_named<'a>(
        &'a self,
        parent: &'a Agent<C>,
        agent_name: &'a str,
        invocation: SubagentInvocation,
    ) -> Pin<Box<dyn Future<Output = Result<ToolResult>> + Send + 'a>>;

    #[allow(dead_code)]
    fn capability_contracts(&self) -> Vec<SubagentCapabilityContract> {
        AgentTemplate::catalog()
            .into_iter()
            .map(|template| template.capability_contract())
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubagentInvocation {
    pub input: NormalizedSubagentInput,
    pub prompt: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubagentCapabilityContract {
    pub name: String,
    pub purpose: String,
    pub tool_scope: ToolScope,
    pub permission_mode: PermissionMode,
    pub can_write: bool,
    pub can_delegate: bool,
    pub default_timeout_secs: Option<u64>,
    pub default_max_tool_calls: Option<usize>,
    pub input_expectations: String,
    pub expected_result_shape: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SubagentCatalogEntry {
    pub agent_name: &'static str,
    pub tool_name: &'static str,
    pub task_description: &'static str,
    pub tool_description: &'static str,
    pub read_only: bool,
}

pub(crate) const SUBAGENT_CATALOG: &[SubagentCatalogEntry] = &[
    SubagentCatalogEntry {
        agent_name: "explorer",
        tool_name: tool_names::TOOL_AGENT_EXPLORE,
        task_description: "交给 explorer 子代理执行的聚焦只读调研任务",
        tool_description: "将限定范围的只读仓库调研任务委派给 explorer 子代理，并返回摘要。",
        read_only: true,
    },
    SubagentCatalogEntry {
        agent_name: "fixer",
        tool_name: tool_names::TOOL_AGENT_FIXER,
        task_description: "交给 fixer 子代理执行的聚焦实现或修复任务",
        tool_description: "将限定范围的实现或修复任务委派给 fixer 子代理，并返回摘要。",
        read_only: false,
    },
    SubagentCatalogEntry {
        agent_name: "oracle",
        tool_name: tool_names::TOOL_AGENT_ORACLE,
        task_description: "交给 oracle 子代理执行的根因分析、风险判断或验证建议任务",
        tool_description: "将限定范围的根因分析、风险判断或验证建议任务委派给 oracle 子代理，并返回摘要。",
        read_only: true,
    },
    SubagentCatalogEntry {
        agent_name: "designer",
        tool_name: tool_names::TOOL_AGENT_DESIGNER,
        task_description: "交给 designer 子代理执行的设计、方案整理或接口梳理任务",
        tool_description: "将限定范围的设计、方案整理或接口梳理任务委派给 designer 子代理，并返回摘要。",
        read_only: true,
    },
    SubagentCatalogEntry {
        agent_name: "librarian",
        tool_name: tool_names::TOOL_AGENT_LIBRARIAN,
        task_description: "交给 librarian 子代理执行的资料整理、证据检索或上下文归档任务",
        tool_description: "将限定范围的仓库资料整理、证据检索或上下文归档任务委派给 librarian 子代理，并返回摘要。",
        read_only: true,
    },
    SubagentCatalogEntry {
        agent_name: "general",
        tool_name: tool_names::TOOL_AGENT_GENERAL,
        task_description: "交给 general 子代理执行的限定范围只读通用辅助任务",
        tool_description: "将限定范围的只读通用辅助任务委派给 general 子代理，并返回摘要。",
        read_only: true,
    },
];

#[derive(Debug, Clone)]
pub enum ConversationRole {
    User,
    Assistant,
    Summary,
}

#[derive(Debug, Clone)]
pub struct ConversationMessage {
    pub role: ConversationRole,
    pub content: String,
}

const DEFAULT_AGENT_PRELUDE: &str = r#"You are a coding agent operating inside a local repository.
Work from the actual project state. Inspect relevant files before changing code. Prefer the smallest correct change that follows existing patterns.
Use tools deliberately: read/search before editing, edit only intended files, and run the validation that fits the task after changes when it is relevant.
Use `memory__recall` selectively when the current task likely overlaps prior investigation, failed approaches, returned context experiments, or files with meaningful history. Prefer filtering by relevant file paths and failed/blocked outcomes when debugging; do not treat recall as a mandatory first step for every task.
For non-trivial work, act like a workflow manager first: decide whether the task needs a specialist lane before you start doing the work yourself. Small tasks may still be handled directly when delegation overhead is not worth it.
Direct execution is for trivial, single-file, clearly bounded work, or when delegation overhead clearly exceeds the benefit. Otherwise, keep a short plan, choose the right specialist or direct path intentionally, reconcile delegated results, and finish with the clearest verified outcome.
Stay within scope. Do not refactor, reformat, rename, or modify unrelated code unless necessary; if broader changes are needed, explain why.
When tools, edits, or validation fail, inspect the error before retrying. Do not hide failures with broad fallbacks or skipped validation; fail fast and explain the actionable cause.
Use context efficiently: search before reading large files, read only relevant sections, avoid dumping long outputs, and summarize state for long tasks.
When requirements are ambiguous or risky, ask a concise clarifying question.
Keep responses concise. Summarize changed files and validation results when code was modified."#;

const ENGINEERING_WORKFLOW_PRELUDE: &str = r#"This turn is an engineering workflow task.
Strengthen the workflow-manager role: for any non-trivial coding task, first decide whether a specialist lane is needed before proceeding directly.
Use direct execution only for trivial, single-file, clearly bounded work, or when delegation overhead clearly exceeds the benefit. Keep simple tasks simple and avoid unnecessary orchestration.
Choose specialists intentionally: explorer for broad or unknown code search; fixer for bounded implementation work and multi-file mechanical edits; oracle for root-cause analysis, risk review, or critical evaluation; designer for UI/UX decisions; librarian for external docs or library/framework behavior; general for bounded read-only auxiliary work.
Reuse prior specialist work when it matches: prefer completed or reconciled sessions from the session history or job board before launching overlapping work. Never reuse cancelled or errored sessions as authoritative results.
Delegate bounded work when it improves quality, speed, or context hygiene, especially for low-level or read-heavy tasks that would otherwise pollute the main agent context.
Keep delegation controlled: avoid recursive delegation, avoid unnecessary multi-agent orchestration, preserve a clear parent agent narrative, reconcile child results, and surface remaining blockers or targeted validation gaps before you stop."#;
const SESSION_TITLE_PRELUDE: &str = r#"Generate a concise session title for the user's first message.
Return only the title text.
Do not use quotes, bullets, markdown, prefixes, or explanations.
Keep it specific and under 80 characters."#;
const CONTEXT_COMPACTION_PRELUDE: &str = r#"你正在为同一会话生成结构化上下文摘要，供后续模型继续工作。

输出要求：
- 只输出摘要正文，不要加前言、后记、代码块或解释。
- 严格使用以下 section 结构与顺序：
  目标
  约束与偏好
  进展
    已完成
    进行中
    受阻
  关键决策
  下一步
  关键上下文
  相关文件
- 保留并逐字引用重要的路径、命令、错误信息、标识符、接口名、配置键、测试名。
- 不要提及“压缩”“摘要过程”“上下文窗口”“tail”等过程性描述。
- 如果某 section 暂无内容，写“无”。
- 进展部分使用简洁项目符号。
- 相关文件尽量写成“路径 — 作用/状态”的形式。"#;
const NO_HISTORICAL_ITEMS_FOR_COMPACTION: &str =
    "no historical items available for context compaction";
const NO_OLDER_ITEMS_AFTER_TAIL: &str = "no older items remain after preserving recent tail";
const COMPACTION_TOOL_OUTPUT_CHAR_CAP: usize = 2_000;
const COMPACTION_HISTORY_MIN_CHAR_BUDGET: usize = 768;
const COMPACTION_HISTORY_MAX_CHAR_BUDGET: usize = 64_000;
const COMPACTION_PRUNE_MIN_OUTPUT_CHARS: usize = 20_000;
const COMPACTION_TOOL_OUTPUT_TRUNCATION_MARKER: &str = "… [tool output truncated for compaction]";
const COMPACTION_HISTORY_TRUNCATION_MARKER: &str =
    "… [older history omitted to keep compaction prompt bounded]";
const COMPACTION_PRUNED_MARKER: &str = "tool output pruned by compaction.prune";
const MAX_SKILL_CARDS_IN_PRELUDE: usize = 64;

pub(crate) type RuntimeSnapshotProvider = Arc<dyn Fn() -> Result<RuntimeSnapshot> + Send + Sync>;
pub(crate) type LogicalCheckpointCandidateProvider =
    Arc<dyn Fn() -> Result<crate::transcript::PreparedLogicalCheckpoint> + Send + Sync>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogicalCheckpointRequestOutcome {
    Queued,
    AlreadyQueued,
    Disabled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LogicalCheckpointRequestState {
    Idle,
    Pending(LogicalCheckpointRequestOwner),
    InFlight(LogicalCheckpointRequestOwner),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LogicalCheckpointRequestOwner {
    Manual,
    Automatic { boundary_id: u64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LogicalCheckpointLease {
    run_id: u64,
    request_id: u64,
    ownership: LogicalCheckpointRequestOwner,
}

#[derive(Clone)]
pub struct LogicalCheckpointControl {
    state: Arc<Mutex<LogicalCheckpointControlState>>,
}

#[derive(Debug)]
struct LogicalCheckpointControlState {
    enabled: bool,
    request: LogicalCheckpointRequestState,
    request_run_id: Option<u64>,
    active_run_id: Option<u64>,
    next_run_id: u64,
    next_request_id: u64,
    request_id: Option<u64>,
    automatic_enabled: bool,
}

impl LogicalCheckpointControl {
    pub fn request(&self) -> LogicalCheckpointRequestOutcome {
        let mut state = self
            .state
            .lock()
            .expect("logical checkpoint control poisoned");
        if !state.enabled {
            return LogicalCheckpointRequestOutcome::Disabled;
        }
        match state.request {
            LogicalCheckpointRequestState::Idle => {
                state.request =
                    LogicalCheckpointRequestState::Pending(LogicalCheckpointRequestOwner::Manual);
                state.request_run_id = state.active_run_id;
                state.request_id = Some(Self::next_request_id(&mut state));
                LogicalCheckpointRequestOutcome::Queued
            }
            LogicalCheckpointRequestState::Pending(LogicalCheckpointRequestOwner::Automatic {
                ..
            }) => {
                state.request =
                    LogicalCheckpointRequestState::Pending(LogicalCheckpointRequestOwner::Manual);
                state.request_id = Some(Self::next_request_id(&mut state));
                LogicalCheckpointRequestOutcome::Queued
            }
            LogicalCheckpointRequestState::Pending(_)
            | LogicalCheckpointRequestState::InFlight(_) => {
                LogicalCheckpointRequestOutcome::AlreadyQueued
            }
        }
    }

    pub(crate) fn request_automatic(&self, boundary_id: u64) -> LogicalCheckpointRequestOutcome {
        let mut state = self
            .state
            .lock()
            .expect("logical checkpoint control poisoned");
        if !state.enabled || !state.automatic_enabled || state.active_run_id.is_none() {
            return LogicalCheckpointRequestOutcome::Disabled;
        }
        match state.request {
            LogicalCheckpointRequestState::Idle => {
                state.request = LogicalCheckpointRequestState::Pending(
                    LogicalCheckpointRequestOwner::Automatic { boundary_id },
                );
                state.request_run_id = state.active_run_id;
                state.request_id = Some(Self::next_request_id(&mut state));
                LogicalCheckpointRequestOutcome::Queued
            }
            LogicalCheckpointRequestState::Pending(_)
            | LogicalCheckpointRequestState::InFlight(_) => {
                LogicalCheckpointRequestOutcome::AlreadyQueued
            }
        }
    }

    fn next_request_id(state: &mut LogicalCheckpointControlState) -> u64 {
        state.next_request_id = state
            .next_request_id
            .checked_add(1)
            .expect("logical checkpoint request id overflow");
        state.next_request_id
    }

    pub(crate) fn take_pending(&self) -> Option<LogicalCheckpointLease> {
        let mut state = self
            .state
            .lock()
            .expect("logical checkpoint control poisoned");
        if !state.enabled {
            state.request = LogicalCheckpointRequestState::Idle;
            state.request_run_id = None;
            return None;
        }
        match state.request {
            LogicalCheckpointRequestState::Pending(ownership) => {
                let run_id = state.request_run_id?;
                let request_id = state.request_id?;
                state.request = LogicalCheckpointRequestState::InFlight(ownership);
                Some(LogicalCheckpointLease {
                    run_id,
                    request_id,
                    ownership,
                })
            }
            LogicalCheckpointRequestState::Idle | LogicalCheckpointRequestState::InFlight(_) => {
                None
            }
        }
    }

    pub(crate) fn clear(&self) {
        let mut state = self
            .state
            .lock()
            .expect("logical checkpoint control poisoned");
        state.request = LogicalCheckpointRequestState::Idle;
        state.request_run_id = None;
        state.request_id = None;
    }

    fn clear_lease(&self, lease: LogicalCheckpointLease) {
        let mut state = self
            .state
            .lock()
            .expect("logical checkpoint control poisoned");
        if state.request_run_id == Some(lease.run_id) && state.request_id == Some(lease.request_id)
        {
            state.request = LogicalCheckpointRequestState::Idle;
            state.request_run_id = None;
            state.request_id = None;
        }
    }

    pub(crate) fn begin_run(&self) -> LogicalCheckpointRunGuard {
        let run_id = {
            let mut state = self
                .state
                .lock()
                .expect("logical checkpoint control poisoned");
            state.next_run_id = state
                .next_run_id
                .checked_add(1)
                .expect("logical checkpoint run id overflow");
            let run_id = state.next_run_id;
            state.active_run_id = Some(run_id);
            // A request made immediately before the stream began belongs to this
            // run; later requests are tagged by request().
            if matches!(state.request, LogicalCheckpointRequestState::Pending(_))
                && state.request_run_id.is_none()
            {
                state.request_run_id = Some(run_id);
            }
            run_id
        };
        LogicalCheckpointRunGuard {
            control: self.clone(),
            run_id,
        }
    }

    fn set_config(&self, config: LogicalCheckpointConfig) {
        let mut state = self
            .state
            .lock()
            .expect("logical checkpoint control poisoned");
        state.enabled = config.enabled;
        state.automatic_enabled = config.enabled && config.automatic;
        if !config.enabled {
            state.request = LogicalCheckpointRequestState::Idle;
            state.request_run_id = None;
            state.request_id = None;
        }
    }

    fn set_enabled(&self, enabled: bool) {
        self.set_config(LogicalCheckpointConfig {
            enabled,
            ..LogicalCheckpointConfig::default()
        });
    }
}

/// Owns a single streamed turn's checkpoint request.  It is deliberately held
/// across all awaits in a protocol stream, so cancellation cannot strand a
/// pending or in-flight request for the next turn.
pub(crate) struct LogicalCheckpointRunGuard {
    control: LogicalCheckpointControl,
    run_id: u64,
}

impl Drop for LogicalCheckpointRunGuard {
    fn drop(&mut self) {
        let mut state = self
            .control
            .state
            .lock()
            .expect("logical checkpoint control poisoned");
        if state.active_run_id == Some(self.run_id) {
            if state.request_run_id == Some(self.run_id) {
                state.request = LogicalCheckpointRequestState::Idle;
                state.request_run_id = None;
                state.request_id = None;
            }
            state.active_run_id = None;
        }
    }
}

impl LogicalCheckpointControl {
    #[cfg(test)]
    fn disabled_for_test() -> Self {
        Self {
            state: Arc::new(Mutex::new(LogicalCheckpointControlState {
                enabled: false,
                request: LogicalCheckpointRequestState::Idle,
                request_run_id: None,
                active_run_id: None,
                next_run_id: 0,
                next_request_id: 0,
                request_id: None,
                automatic_enabled: false,
            })),
        }
    }
}

pub struct Agent<C: Config> {
    pub client: Client<C>,
    model: String,
    subagent_model_overrides: HashMap<String, String>,
    default_protocol: ApiProtocol,
    model_protocols: HashMap<String, ApiProtocol>,
    model_catalog: HashMap<String, ModelRequestMetadata>,
    prelude: Vec<PromptMessage>,
    protocol_frames: Vec<crate::protocol_frames::ProtocolFrame>,
    history: Vec<HistoryItem>,
    runtime_snapshot: RuntimeSnapshot,
    tools: ToolRegistry,
    skill_registry: Option<Arc<SkillRegistry>>,
    skill_cards: Vec<SkillCard>,
    subagent_delegate: Option<Arc<dyn SubagentDelegate<C>>>,
    question_handler: Option<QuestionCallback>,
    permission_session: Arc<Mutex<PermissionSessionState>>,
    compaction_config: CompactionConfig,
    automatic_checkpoint_policy: automatic_checkpoint::AutoCheckpointPolicy,
    retry_config: RetryConfig,
    tool_timeout_secs: Option<u64>,
    needs_compaction: bool,
    turn: TurnRuntimeState,
    next_turn_id: u64,
    max_iterations: Option<usize>,
    max_tool_calls: Option<usize>,
    context_scope_state: Arc<std::sync::Mutex<ContextScopeState>>,
    runtime_snapshot_provider: Option<RuntimeSnapshotProvider>,
    logical_checkpoint_candidate_provider: Option<LogicalCheckpointCandidateProvider>,
    context_experiment_restore_point: Option<ContextExperimentRestorePoint>,
    logical_checkpoint_control: LogicalCheckpointControl,
}

#[derive(Debug, Clone)]
struct ContextExperimentRestorePoint {
    scope: ActiveContextExperiment,
    protocol_frames: Vec<crate::protocol_frames::ProtocolFrame>,
    runtime_snapshot: RuntimeSnapshot,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentTemplate {
    pub name: String,
    pub purpose: String,
    pub system_prompt: String,
    pub tool_scope: ToolScope,
    pub permission_mode: PermissionMode,
    pub can_write: bool,
    pub can_delegate: bool,
    pub timeout_secs: Option<u64>,
    pub max_tool_calls: Option<usize>,
    pub input_expectations: String,
    pub expected_result_shape: String,
}

impl AgentTemplate {
    fn read_only(name: &str, purpose: &str, system_prompt: &str) -> Self {
        Self {
            name: name.into(),
            purpose: purpose.into(),
            system_prompt: system_prompt.into(),
            tool_scope: ToolScope::ReadOnlyExplorer,
            permission_mode: PermissionMode::Default,
            can_write: false,
            can_delegate: false,
            timeout_secs: None,
            max_tool_calls: None,
            input_expectations: "需要明确的 task 或 objective；可选 success_criteria、allowed_paths、forbidden_paths、owned_paths。runtime 超时和工具预算由配置继承，不应在普通委派里填写。".into(),
            expected_result_shape: "JSON object with run_id, child_session_id, agent_name, status, summary.".into(),
        }
    }

    pub fn explorer() -> Self {
        Self::read_only(
            "explorer",
            "只读仓库探索",
            concat!(
                "你是一个只读的 explorer 子代理。请围绕分配给你的任务调查本地项目，仓库，文件夹等、给出结论，",
                "并且只能使用只读工具。不要编辑文件，不要运行具备写能力的命令，也不要继续委派。"
            ),
        )
    }
    pub fn fixer() -> Self {
        Self {
            name: "fixer".into(),
            purpose: "修复/构建者代理".into(),
            system_prompt: concat!(
                "你是一个可读可写的修复者子代理。根据主代理给出的方向和要求，使用合理的工具，按照意图进行实现。",
                "请严格按照主代理的要求来进行实现，而非自己想当然的做法。仅做主代理要求做的部分，不做分外的事。",
                "你可以使用绝大多数工具，但请按照要求来。"
            )
            .into(),
            tool_scope: ToolScope::FullAccess,
            permission_mode: PermissionMode::Default,
            can_write: true,
            can_delegate: false,
            timeout_secs: None,
            max_tool_calls: None,
            input_expectations: "需要明确的 task 或 objective；可选 success_criteria、allowed_paths、forbidden_paths、owned_paths。runtime 超时和工具预算由配置继承，不应在普通委派里填写。".into(),
            expected_result_shape: "JSON object with run_id, child_session_id, agent_name, status, summary.".into(),
        }
    }

    pub fn oracle() -> Self {
        Self::read_only(
            "oracle",
            "只读根因与风险分析",
            concat!(
                "你是 oracle 子代理。专注于只读分析、根因判断、方案权衡、风险识别与验证建议。",
                "不要修改文件，不要运行具备写能力的命令，不要继续委派。输出应帮助主代理做决策，而不是代替 fixer 实现修改。"
            ),
        )
    }

    pub fn designer() -> Self {
        Self::read_only(
            "designer",
            "只读设计与方案整理",
            concat!(
                "你是 designer 子代理。专注于阅读现有实现、梳理接口、提出小而清晰的设计方案、命名建议与变更边界。",
                "不要修改文件，不要运行具备写能力的命令，不要继续委派。"
            ),
        )
    }

    pub fn librarian() -> Self {
        Self::read_only(
            "librarian",
            "只读资料整理与证据归档",
            concat!(
                "你是 librarian 子代理。专注于检索本仓库中的相关文件、证据、历史上下文、接口位置与约束，",
                "给出紧凑且可追溯的引用。不要修改文件，不要运行具备写能力的命令，不要继续委派。"
            ),
        )
    }

    pub fn general() -> Self {
        Self::read_only(
            "general",
            "只读通用问题助手",
            concat!(
                "你是 general 子代理。用于边界明确但不属于其他专家的只读辅助任务，例如梳理奇怪输出、归纳现象、总结仓库事实。",
                "保持只读，不要实现修改，不要替代 fixer，不要继续委派。"
            ),
        )
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "explorer" => Some(Self::explorer()),
            "fixer" => Some(Self::fixer()),
            "oracle" => Some(Self::oracle()),
            "designer" => Some(Self::designer()),
            "librarian" => Some(Self::librarian()),
            "general" => Some(Self::general()),
            _ => None,
        }
    }

    pub fn catalog() -> Vec<Self> {
        vec![
            Self::explorer(),
            Self::fixer(),
            Self::oracle(),
            Self::designer(),
            Self::librarian(),
            Self::general(),
        ]
    }

    pub fn capability_contract(&self) -> SubagentCapabilityContract {
        SubagentCapabilityContract {
            name: self.name.clone(),
            purpose: self.purpose.clone(),
            tool_scope: self.tool_scope,
            permission_mode: self.permission_mode,
            can_write: self.can_write,
            can_delegate: self.can_delegate,
            default_timeout_secs: self.timeout_secs,
            default_max_tool_calls: self.max_tool_calls,
            input_expectations: self.input_expectations.clone(),
            expected_result_shape: self.expected_result_shape.clone(),
        }
    }
}

pub struct AgentFactory;

impl AgentFactory {
    pub fn create_child<C: Config + Clone>(
        parent: &Agent<C>,
        template: &AgentTemplate,
    ) -> Agent<C> {
        Self::create_child_with_max_tool_calls(parent, template, None)
    }

    pub fn create_child_with_max_tool_calls<C: Config + Clone>(
        parent: &Agent<C>,
        template: &AgentTemplate,
        max_tool_calls_override: Option<usize>,
    ) -> Agent<C> {
        let model = parent
            .subagent_model_override(&template.name)
            .unwrap_or(parent.model())
            .to_string();
        let mut prelude = parent.prelude.clone();
        prelude.push(PromptMessage::developer(template.system_prompt.clone()));

        Agent {
            client: parent.client.clone(),
            model: model.clone(),
            subagent_model_overrides: parent.subagent_model_overrides.clone(),
            default_protocol: parent.default_protocol,
            model_protocols: parent.model_protocols.clone(),
            model_catalog: parent.model_catalog.clone(),
            prelude,
            protocol_frames: Vec::new(),
            history: Vec::new(),
            runtime_snapshot: Agent::<C>::fresh_runtime_snapshot(&model),
            tools: parent.tools.scoped(template.tool_scope).without_tools(&[
                tool_names::TOOL_MEMORY_RECALL,
                tool_names::TOOL_AGENT_RECONCILE,
            ]),
            skill_registry: parent.skill_registry.clone(),
            skill_cards: parent.skill_cards.clone(),
            subagent_delegate: None,
            question_handler: None,
            permission_session: Arc::clone(&parent.permission_session),
            compaction_config: parent.compaction_config.clone(),
            automatic_checkpoint_policy: parent.automatic_checkpoint_policy,
            retry_config: parent.retry_config.clone(),
            tool_timeout_secs: parent.tool_timeout_secs,
            needs_compaction: false,
            turn: TurnRuntimeState::default(),
            next_turn_id: 0,
            max_iterations: parent.max_iterations,
            max_tool_calls: capped_tool_call_limit(
                parent.max_tool_calls,
                max_tool_calls_override.or(template.max_tool_calls),
            ),
            context_scope_state: Arc::new(std::sync::Mutex::new(ContextScopeState::default())),
            runtime_snapshot_provider: None,
            logical_checkpoint_candidate_provider: None,
            context_experiment_restore_point: None,
            logical_checkpoint_control: LogicalCheckpointControl {
                state: Arc::new(Mutex::new(LogicalCheckpointControlState {
                    enabled: false,
                    request: LogicalCheckpointRequestState::Idle,
                    request_run_id: None,
                    active_run_id: None,
                    next_run_id: 0,
                    next_request_id: 0,
                    request_id: None,
                    automatic_enabled: false,
                })),
            },
        }
    }
}

impl<C: Config> Agent<C> {
    pub fn new(
        client: Client<C>,
        model: impl Into<String>,
        max_iterations: impl Into<Option<usize>>,
        max_tool_calls: impl Into<Option<usize>>,
    ) -> Self {
        let max_iterations = max_iterations.into();
        let max_tool_calls = max_tool_calls.into();
        let model = model.into();
        Self {
            client,
            model: model.clone(),
            subagent_model_overrides: HashMap::new(),
            default_protocol: ApiProtocol::Responses,
            model_protocols: HashMap::new(),
            model_catalog: HashMap::new(),
            prelude: default_agent_prelude(),
            protocol_frames: vec![],
            history: vec![],
            runtime_snapshot: Self::fresh_runtime_snapshot(&model),
            tools: ToolRegistry::default_tools(),
            skill_registry: None,
            skill_cards: Vec::new(),
            subagent_delegate: None,
            question_handler: None,
            permission_session: Arc::new(Mutex::new(PermissionSessionState::default())),
            compaction_config: CompactionConfig::default(),
            automatic_checkpoint_policy: automatic_checkpoint::AutoCheckpointPolicy::from_config(
                LogicalCheckpointConfig::default(),
            ),
            retry_config: RetryConfig::default(),
            tool_timeout_secs: Some(60),
            needs_compaction: false,
            turn: TurnRuntimeState::default(),
            next_turn_id: 0,
            max_iterations,
            max_tool_calls,
            context_scope_state: Arc::new(std::sync::Mutex::new(ContextScopeState::default())),
            runtime_snapshot_provider: None,
            logical_checkpoint_candidate_provider: None,
            context_experiment_restore_point: None,
            logical_checkpoint_control: LogicalCheckpointControl {
                state: Arc::new(Mutex::new(LogicalCheckpointControlState {
                    enabled: false,
                    request: LogicalCheckpointRequestState::Idle,
                    request_run_id: None,
                    active_run_id: None,
                    next_run_id: 0,
                    next_request_id: 0,
                    request_id: None,
                    automatic_enabled: false,
                })),
            },
        }
    }

    pub fn set_model_catalog(&mut self, catalog: HashMap<String, ModelRequestMetadata>) {
        self.model_catalog = catalog;
    }

    pub fn set_default_protocol(&mut self, protocol: ApiProtocol) {
        self.default_protocol = protocol;
    }

    pub fn set_model_protocols(&mut self, protocols: HashMap<String, ApiProtocol>) {
        self.model_protocols = protocols;
    }

    pub fn set_compaction_config(&mut self, config: CompactionConfig) {
        self.compaction_config = config;
    }

    pub fn set_logical_checkpoint_config(&mut self, config: LogicalCheckpointConfig) {
        self.automatic_checkpoint_policy =
            automatic_checkpoint::AutoCheckpointPolicy::from_config(config);
        self.logical_checkpoint_control.set_config(config);
    }

    pub fn logical_checkpoint_control(&self) -> LogicalCheckpointControl {
        self.logical_checkpoint_control.clone()
    }

    pub fn request_logical_checkpoint(&self) -> LogicalCheckpointRequestOutcome {
        self.logical_checkpoint_control.request()
    }

    pub(crate) fn clear_logical_checkpoint_request(&self) {
        self.logical_checkpoint_control.clear();
    }

    pub fn set_tool_timeout_secs(&mut self, timeout_secs: Option<u64>) {
        self.tool_timeout_secs = timeout_secs;
    }

    pub fn set_retry_config(&mut self, config: RetryConfig) {
        self.retry_config = config;
    }

    fn active_protocol(&self) -> ApiProtocol {
        self.protocol_for_model(&self.model)
    }

    fn protocol_for_model(&self, model_id: &str) -> ApiProtocol {
        self.model_protocols
            .get(model_id)
            .cloned()
            .unwrap_or(self.default_protocol)
    }

    pub(crate) fn active_model_metadata(&self) -> ModelRequestMetadata {
        self.model_metadata_for(&self.model)
    }

    fn model_metadata_for(&self, model_id: &str) -> ModelRequestMetadata {
        self.model_catalog
            .get(model_id)
            .cloned()
            .unwrap_or(ModelRequestMetadata {
                context_window: None,
                max_output_tokens: None,
                // Backward compatible default: historically tools were always advertised.
                // If a model isn't in the catalog, we assume tools are supported.
                supports_tools: true,
                supports_reasoning: false,
                ..Default::default()
            })
    }

    #[cfg(test)]
    fn current_turn(&self) -> &WorkflowTurnState {
        &self.turn.policy
    }

    #[cfg(test)]
    fn current_turn_id(&self) -> u64 {
        self.turn.turn_id
    }

    #[cfg(test)]
    fn todos(&self) -> &[TodoItem] {
        &self.turn.workflow.todos
    }

    #[cfg(test)]
    fn auto_continue(&self) -> &AutoContinueState {
        &self.turn.workflow.auto_continue
    }

    #[cfg(test)]
    pub(crate) fn tool_definitions_for_test(&self) -> Vec<crate::request_builder::ToolSpec> {
        self.tool_definitions()
    }

    pub fn permission_mode(&self) -> PermissionMode {
        self.permission_session
            .lock()
            .expect("permission session poisoned")
            .mode()
    }

    pub fn set_permission_mode(&mut self, mode: PermissionMode) {
        self.permission_session
            .lock()
            .expect("permission session poisoned")
            .set_mode(mode);
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn reasoning_effort(&self) -> Option<ModelReasoningEffort> {
        self.active_model_metadata().reasoning_effort
    }

    pub fn session_token_usage(&self) -> Result<TokenUsageEstimate> {
        self.candidate_session_token_usage(&self.model, &self.runtime_snapshot)
    }

    /// Estimate a prospective session request without changing this agent's
    /// selected model, history, runtime snapshot, or turn state.
    pub(crate) fn candidate_session_token_usage(
        &self,
        model_id: &str,
        runtime_snapshot: &RuntimeSnapshot,
    ) -> Result<TokenUsageEstimate> {
        let model = self.model_metadata_for(model_id);
        let policy = ProtectedContextPolicy::from_configured_reserve(
            self.compaction_config.protected_reserve_tokens,
            effective_input_budget_tokens(model.clone(), &self.tool_definitions()),
        );
        let build = build_request_with_policy(
            RequestBuilderInput {
                protocol: self.protocol_for_model(model_id),
                model_id,
                model,
                prelude: &[],
                snapshot: runtime_snapshot,
                tools: &self.tool_definitions(),
            },
            None,
            Some(policy),
        )?;

        Ok(TokenUsageEstimate {
            used_tokens: build.budget.estimated_request_tokens,
            context_window_tokens: build.budget.context_window_tokens,
            input_tokens: build.budget.estimated_request_tokens,
            output_tokens: 0,
            cached_tokens: 0,
        })
    }

    pub fn tool_scope(&self) -> ToolScope {
        self.tools.scope()
    }

    pub(crate) fn max_tool_calls_limit(&self) -> Option<usize> {
        self.max_tool_calls
    }

    #[cfg(test)]
    pub(crate) fn max_iterations_limit(&self) -> Option<usize> {
        self.max_iterations
    }

    pub fn subagent_model_override(&self, agent_name: &str) -> Option<&str> {
        self.subagent_model_overrides
            .get(agent_name)
            .map(String::as_str)
    }

    pub fn set_model(&mut self, model: impl Into<String>) {
        self.model = model.into();
        self.runtime_snapshot.latest_model = Some(self.model.clone());
    }

    pub fn set_subagent_model_override(
        &mut self,
        agent_name: impl Into<String>,
        model: impl Into<String>,
    ) {
        self.subagent_model_overrides
            .insert(agent_name.into(), model.into());
    }

    pub fn set_reasoning_effort(&mut self, effort: ModelReasoningEffort) -> Result<()> {
        let mut metadata = self.active_model_metadata();
        let selectable = metadata.selectable_reasoning_efforts();
        if selectable.is_empty() {
            bail!(
                "model '{}' does not support configurable reasoning",
                self.model
            );
        }
        if !metadata.allows_reasoning_effort(&effort) {
            let available = selectable
                .into_iter()
                .map(|effort| format!("{effort:?}").to_ascii_lowercase())
                .collect::<Vec<_>>()
                .join(", ");
            bail!(
                "reasoning effort '{effort:?}' is not supported by model '{}'; available values: {available}",
                self.model
            );
        }
        metadata.reasoning_effort = Some(effort);
        self.model_catalog.insert(self.model.clone(), metadata);
        Ok(())
    }

    #[allow(dead_code)]
    pub fn restore_transcript_messages(&mut self, messages: Vec<ConversationMessage>) {
        self.history = messages
            .into_iter()
            .map(|message| match message.role {
                ConversationRole::User => HistoryItem::user(message.content),
                ConversationRole::Assistant => HistoryItem::assistant(message.content),
                ConversationRole::Summary => HistoryItem::context_summary(message.content),
            })
            .collect();
        self.rebuild_protocol_state_from_history()
            .expect("restored transcript messages should remain protocol-compatible");
    }

    #[allow(dead_code)]
    pub fn restore_evidence(&mut self, evidence: Vec<EvidenceRecord>) -> Result<()> {
        Self::validate_evidence_ids(&evidence)?;
        self.runtime_snapshot.set_evidence(evidence);
        Ok(())
    }

    pub fn restore_session_context(
        &mut self,
        messages: Vec<ConversationMessage>,
        evidence: Vec<EvidenceRecord>,
        max_turn_id: u64,
    ) -> Result<()> {
        let history = messages
            .into_iter()
            .filter_map(|message| match message.role {
                ConversationRole::User => Some(HistoryItem::user(message.content)),
                ConversationRole::Assistant => Some(HistoryItem::assistant(message.content)),
                ConversationRole::Summary => Some(HistoryItem::context_summary(message.content)),
            })
            .collect();
        self.restore_session_history(history, evidence, max_turn_id)
    }

    pub fn restore_session_history(
        &mut self,
        history: Vec<HistoryItem>,
        evidence: Vec<EvidenceRecord>,
        max_turn_id: u64,
    ) -> Result<()> {
        Self::validate_evidence_ids(&evidence)?;

        let transcript = crate::protocol_frames::analyze_history_items(&history, None)?;
        let mut runtime_snapshot = self.rebuilt_runtime_snapshot_from_protocol_frames(
            &transcript.frames,
            self.protocol_frames.len(),
            &self.history,
        )?;
        runtime_snapshot.current_turn_id = Some(max_turn_id);
        runtime_snapshot.set_evidence(evidence);
        let protocol_frames = runtime_snapshot.active_protocol_frames();
        let candidate_history = crate::protocol_frames::history_items_from_frames(&protocol_frames);
        crate::protocol_frames::analyze_history_items(&candidate_history, None)?;
        validate_runtime_snapshot_correspondence(&candidate_history, &runtime_snapshot)?;

        self.protocol_frames = protocol_frames;
        self.history = candidate_history;
        self.runtime_snapshot = runtime_snapshot;
        self.next_turn_id = max_turn_id;
        self.needs_compaction = false;
        self.turn = TurnRuntimeState::default();
        Ok(())
    }

    /// Discard all state that belongs to the current session before creating a
    /// new one. Unlike compatibility rebuilds used by restore and checkout,
    /// this deliberately does not preserve runtime snapshot metadata.
    pub fn reset_for_new_session(&mut self) {
        self.protocol_frames.clear();
        self.history.clear();
        self.runtime_snapshot = Self::fresh_runtime_snapshot(&self.model);
        self.needs_compaction = false;
        self.turn = TurnRuntimeState::default();
        self.next_turn_id = 0;
        self.context_experiment_restore_point = None;
        if let Ok(mut permissions) = self.permission_session.lock() {
            permissions.clear_grants();
        }
    }

    pub fn restore_runtime_snapshot(
        &mut self,
        protocol_frames: Vec<crate::protocol_frames::ProtocolFrame>,
        mut runtime_snapshot: RuntimeSnapshot,
    ) -> Result<()> {
        reconcile_loaded_skill_material(&mut runtime_snapshot)?;
        Self::validate_evidence_ids(&runtime_snapshot.evidence)?;

        // Validate the complete candidate before replacing any live state. A failed
        // restore must leave the running agent untouched.
        let history = crate::protocol_frames::history_items_from_frames(&protocol_frames);
        crate::protocol_frames::analyze_history_items(&history, None)?;
        validate_runtime_snapshot_correspondence(&history, &runtime_snapshot)?;
        validate_protocol_frame_correspondence(&protocol_frames, &runtime_snapshot)?;
        let protocol_frames = runtime_snapshot.active_protocol_frames();

        let restored_turn_id = runtime_snapshot.current_turn_id.unwrap_or_default();
        if runtime_snapshot.latest_model.is_none() {
            runtime_snapshot.latest_model = Some(self.model.clone());
        }
        self.turn = TurnRuntimeState::default();
        self.protocol_frames = protocol_frames;
        self.history = history;
        self.runtime_snapshot = runtime_snapshot;
        self.next_turn_id = self.next_turn_id.max(restored_turn_id);
        self.needs_compaction = false;
        Ok(())
    }

    pub fn restore_turn_sequence(&mut self, max_turn_id: u64) {
        self.next_turn_id = self.next_turn_id.max(max_turn_id);
    }

    /// Commit a snapshot for a wholly new session.  Unlike ordinary restores,
    /// the turn sequence must not retain an id from the abandoned session.
    pub fn restore_new_session_runtime_snapshot(
        &mut self,
        protocol_frames: Vec<crate::protocol_frames::ProtocolFrame>,
        runtime_snapshot: RuntimeSnapshot,
        max_turn_id: u64,
    ) -> Result<()> {
        self.restore_runtime_snapshot(protocol_frames, runtime_snapshot)?;
        self.permission_session
            .lock()
            .map_err(|_| anyhow!("permission session poisoned"))?
            .clear_grants();
        self.next_turn_id = max_turn_id;
        self.context_experiment_restore_point = None;
        Ok(())
    }

    pub fn add_evidence(&mut self, evidence: EvidenceRecord) -> Result<()> {
        let mut candidate = self.runtime_snapshot.evidence.clone();
        require_unique_evidence_id(&candidate, &evidence.id)?;
        candidate.push(evidence);
        self.runtime_snapshot.set_evidence(candidate);
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn history_for_test(&self) -> &[HistoryItem] {
        &self.history
    }

    #[cfg(test)]
    pub(crate) fn protocol_frames_for_test(&self) -> &[crate::protocol_frames::ProtocolFrame] {
        &self.protocol_frames
    }

    #[cfg(test)]
    pub(crate) fn runtime_snapshot_for_test(&self) -> &RuntimeSnapshot {
        &self.runtime_snapshot
    }

    #[allow(dead_code)]
    pub fn evidence(&self) -> &[EvidenceRecord] {
        self.runtime_snapshot.evidence.as_slice()
    }

    #[allow(dead_code)]
    pub fn register_tool<T>(&mut self, tool: T)
    where
        T: ToolHandler + 'static,
    {
        self.tools.register(tool);
    }

    pub fn try_register_tool<T>(&mut self, tool: T) -> Result<()>
    where
        T: ToolHandler + 'static,
    {
        self.tools.try_register(tool)
    }

    pub fn register_skill_registry(&mut self, registry: Arc<SkillRegistry>) -> Result<()> {
        self.skill_cards = registry.cards();
        if self.skill_cards.len() > MAX_SKILL_CARDS_IN_PRELUDE {
            bail!(
                "too many skills discovered: {} exceeds maximum {}",
                self.skill_cards.len(),
                MAX_SKILL_CARDS_IN_PRELUDE
            );
        }
        self.skill_registry = Some(registry.clone());
        if registry.is_empty() {
            Ok(())
        } else {
            self.try_register_tool(SkillTool::new(registry.clone()))?;
            self.try_register_tool(SkillResourceListTool::new(registry.clone()))?;
            self.try_register_tool(SkillResourceReadTool::new(registry))
        }
    }

    pub fn set_subagent_delegate(&mut self, delegate: Arc<dyn SubagentDelegate<C>>) {
        self.subagent_delegate = Some(delegate);
    }

    pub fn set_context_scope_state(
        &mut self,
        context_scope_state: Arc<std::sync::Mutex<ContextScopeState>>,
    ) {
        self.context_scope_state = context_scope_state;
    }

    pub(crate) fn set_runtime_snapshot_provider(&mut self, provider: RuntimeSnapshotProvider) {
        self.runtime_snapshot_provider = Some(provider);
    }

    pub(crate) fn clear_runtime_snapshot_provider(&mut self) {
        self.runtime_snapshot_provider = None;
    }

    pub(crate) fn set_logical_checkpoint_candidate_provider(
        &mut self,
        provider: LogicalCheckpointCandidateProvider,
    ) {
        self.logical_checkpoint_candidate_provider = Some(provider);
    }

    pub(crate) fn clear_logical_checkpoint_candidate_provider(&mut self) {
        self.logical_checkpoint_candidate_provider = None;
    }

    pub(super) fn refresh_runtime_snapshot_from_provider(&mut self) -> Result<()> {
        let Some(provider) = &self.runtime_snapshot_provider else {
            return Ok(());
        };
        let mut projected = provider().context("failed to project runtime snapshot for refresh")?;
        Self::validate_evidence_ids(&projected.evidence)?;
        validate_runtime_snapshot_correspondence(&self.history, &projected)?;
        validate_runtime_snapshot_correspondence(&self.history, &self.runtime_snapshot)?;
        let live_protocol_frames = self.runtime_snapshot.active_protocol_frames();
        let provider_id_remap = projected
            .active_protocol_frames()
            .iter()
            .zip(&live_protocol_frames)
            .map(|(provider_frame, live_frame)| {
                (
                    provider_frame
                        .runtime_frame_id
                        .expect("validated provider frame id"),
                    live_frame
                        .runtime_frame_id
                        .expect("validated live frame id"),
                )
            })
            .collect::<HashMap<_, _>>();
        remap_runtime_snapshot_frame_ids(&mut projected, &provider_id_remap);
        // Providers commonly project only live protocol frames. Preserve durable
        // runtime/session metadata when that projection intentionally omits it,
        // and enrich matching live frames without replacing their identity.
        let durable_by_id = self
            .runtime_snapshot
            .frames
            .iter()
            .map(|frame| (frame.id, frame))
            .collect::<HashMap<_, _>>();
        for frame in &mut projected.frames {
            if let Some(durable) = durable_by_id.get(&frame.id) {
                if frame.protocol.is_some() {
                    merge_runtime_provenance(&mut frame.provenance, &durable.provenance);
                }
            }
        }
        let projected_ids = projected
            .frames
            .iter()
            .map(|frame| frame.id)
            .collect::<HashSet<_>>();
        projected.frames.extend(
            self.runtime_snapshot
                .frames
                .iter()
                .filter(|frame| !projected_ids.contains(&frame.id))
                .cloned(),
        );
        for child in &self.runtime_snapshot.child_sessions {
            if !projected
                .child_sessions
                .iter()
                .any(|existing| existing.child_session_id == child.child_session_id)
            {
                projected.child_sessions.push(child.clone());
            }
        }
        for contributor in &self.runtime_snapshot.prompt_contributors {
            if !projected
                .prompt_contributors
                .iter()
                .any(|existing| existing.contributor_id == contributor.contributor_id)
            {
                projected.prompt_contributors.push(contributor.clone());
            }
        }
        if projected.session_id.is_none() {
            projected.session_id = self.runtime_snapshot.session_id.clone();
        }
        if projected.latest_model.is_none() {
            projected.latest_model = self.runtime_snapshot.latest_model.clone();
        }
        if projected.leaf_sequence.is_none() {
            projected.leaf_sequence = self.runtime_snapshot.leaf_sequence;
        }
        if projected.current_turn_id.is_none() {
            projected.current_turn_id = self.runtime_snapshot.current_turn_id;
        }
        projected.compaction.explicit_protected_frame_ids.extend(
            self.runtime_snapshot
                .compaction
                .explicit_protected_frame_ids
                .iter()
                .copied(),
        );
        projected.compaction.explicit_protected_frame_ids.sort();
        projected.compaction.explicit_protected_frame_ids.dedup();
        projected.recompute_protected_frame_ids();
        projected.compaction.compacted_frame_ids.extend(
            self.runtime_snapshot
                .compaction
                .compacted_frame_ids
                .iter()
                .copied(),
        );
        projected.compaction.compacted_frame_ids.sort();
        projected.compaction.compacted_frame_ids.dedup();
        projected.compaction.retired_source_spans.extend(
            self.runtime_snapshot
                .compaction
                .retired_source_spans
                .iter()
                .copied(),
        );
        projected.compaction.retired_source_spans =
            merge_runtime_source_spans(projected.compaction.retired_source_spans.iter().copied());
        reconcile_loaded_skill_material(&mut projected)?;
        projected.validate_references()?;
        validate_runtime_snapshot_correspondence(&self.history, &projected)?;
        let protocol_frames = projected.active_protocol_frames();
        validate_protocol_frame_correspondence(&protocol_frames, &projected)?;
        self.runtime_snapshot = projected;
        self.protocol_frames = protocol_frames;
        self.history = crate::protocol_frames::history_items_from_frames(&self.protocol_frames);
        Ok(())
    }

    /// Replace the active runtime with the provider's canonical projection.
    /// Unlike refresh, a context scope transition must not retain frames,
    /// contributors, or protocol identity from the outgoing scope.
    fn replace_runtime_snapshot_from_provider(&mut self) -> Result<()> {
        let provider = self.runtime_snapshot_provider.as_ref().ok_or_else(|| {
            anyhow!("successful context scope transition requires a runtime snapshot provider")
        })?;
        let mut snapshot = provider().context("failed to project replacement runtime snapshot")?;
        reconcile_loaded_skill_material(&mut snapshot)?;
        let protocol_frames = snapshot.active_protocol_frames();
        let history = crate::protocol_frames::history_items_from_frames(&protocol_frames);
        crate::protocol_frames::analyze_history_items(&history, None)?;
        validate_runtime_snapshot_correspondence(&history, &snapshot)?;
        validate_protocol_frame_correspondence(&protocol_frames, &snapshot)?;

        Self::validate_evidence_ids(&snapshot.evidence)?;
        let restored_turn_id = snapshot.current_turn_id.unwrap_or_default();

        self.turn = TurnRuntimeState::default();
        self.protocol_frames = protocol_frames;
        self.history = history;
        self.runtime_snapshot = snapshot;
        self.next_turn_id = self.next_turn_id.max(restored_turn_id);
        self.needs_compaction = false;
        Ok(())
    }

    pub fn clear_context_experiment_restore_point(&mut self) {
        self.context_experiment_restore_point = None;
    }

    pub(super) fn history_items(&self) -> Vec<HistoryItem> {
        crate::protocol_frames::history_items_from_frames(&self.protocol_frames)
    }

    pub(super) fn append_history_item(&mut self, item: HistoryItem) -> Result<()> {
        self.append_protocol_frame(crate::protocol_frames::ProtocolFrame::derived(
            protocol_frame_item_from_history_item(&item),
        ))
    }

    pub(super) fn replace_history(&mut self, history: Vec<HistoryItem>) -> Result<()> {
        let old_history = self.history.clone();
        let transcript = crate::protocol_frames::analyze_history_items(
            &history,
            self.turn.current_turn_start_index,
        )?;
        let previous_protocol_frame_count = self.protocol_frames.len();
        let protocol_frames = transcript.frames;
        let runtime_snapshot = self.rebuilt_runtime_snapshot_from_protocol_frames(
            &protocol_frames,
            previous_protocol_frame_count,
            &old_history,
        )?;
        self.history = history;
        self.protocol_frames = protocol_frames;
        self.runtime_snapshot = runtime_snapshot;
        Ok(())
    }

    /// Applies a compaction plan by runtime identity.  This deliberately avoids
    /// rebuilding from compatibility history: retired frames retain their IDs and
    /// provenance, while the active protocol/history caches are derived afterward.
    fn apply_runtime_compaction(
        &mut self,
        selection: &compaction::CompactionSelection,
        summary: String,
    ) -> Result<()> {
        let snapshot = self.prepare_runtime_compaction(selection, summary)?;
        let protocol_frames = snapshot.active_protocol_frames();
        let history = crate::protocol_frames::history_items_from_frames(&protocol_frames);
        crate::protocol_frames::analyze_history_items(
            &history,
            self.turn.current_turn_start_index,
        )?;
        self.runtime_snapshot = snapshot;
        self.protocol_frames = protocol_frames;
        self.history = history;
        Ok(())
    }

    fn prepare_runtime_compaction(
        &self,
        selection: &compaction::CompactionSelection,
        summary: String,
    ) -> Result<RuntimeSnapshot> {
        self.prepare_runtime_compaction_from_snapshot(&self.runtime_snapshot, selection, summary)
    }

    fn prepare_runtime_compaction_from_snapshot(
        &self,
        source_snapshot: &RuntimeSnapshot,
        selection: &compaction::CompactionSelection,
        summary: String,
    ) -> Result<RuntimeSnapshot> {
        let protocol_selected = selection
            .retired_frame_ids
            .iter()
            .copied()
            .collect::<HashSet<_>>();
        let selected = protocol_selected
            .iter()
            .copied()
            .chain(selection.dependent_frame_ids.iter().copied())
            .collect::<HashSet<_>>();
        ensure!(!selected.is_empty(), "compaction selection has no frames");
        let active = source_snapshot.active_protocol_frames();
        let active_ids = active
            .iter()
            .filter_map(|frame| frame.runtime_frame_id)
            .collect::<HashSet<_>>();
        ensure!(
            protocol_selected.is_subset(&active_ids),
            "compaction selection references non-active runtime frames"
        );
        let protocol_selected_for_classification = selection
            .retired_frame_ids
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>();
        let retained_spans = compaction::retained_compaction_spans(
            source_snapshot,
            &protocol_selected_for_classification,
            &selection.retired_source_spans,
        )?;
        ensure!(
            selection
                .retired_source_spans
                .iter()
                .all(|retired| retained_spans.iter().all(|retained| {
                    !(retired.start_sequence <= retained.end_sequence
                        && retained.start_sequence <= retired.end_sequence)
                })),
            "compaction retirement spans overlap retained runtime state"
        );

        let mut snapshot = source_snapshot.clone();
        for frame in &mut snapshot.frames {
            if selected.contains(&frame.id) {
                frame.visibility = FrameVisibility::Retired;
            }
        }
        if let Some(frame) = snapshot.frames.iter_mut().find(|frame| {
            frame.visibility == FrameVisibility::Active
                && matches!(
                    frame.protocol,
                    Some(crate::protocol_frames::ProtocolFrameItem::ContextSummary { .. })
                )
        }) {
            frame.protocol = Some(crate::protocol_frames::ProtocolFrameItem::ContextSummary {
                text: summary.clone(),
            });
            frame.summary = Some(summary);
        } else {
            let protocol = crate::protocol_frames::ProtocolFrameItem::ContextSummary {
                text: summary.clone(),
            };
            let mut summary_frame = RuntimeFrame::new(
                RuntimeFrameKind::Summary,
                FrameVisibility::Active,
                RuntimeFrameProvenance::new(RuntimeSource::SummaryArtifact),
                RuntimeFrameIdSeed {
                    frame_kind: RuntimeFrameKind::Summary,
                    source: RuntimeSource::SummaryArtifact,
                    ordinal: snapshot.compaction.compacted_frame_ids.len() as u32,
                    stable_key: "runtime-compaction-summary",
                    source_span: None,
                },
            );
            summary_frame.protocol = Some(protocol);
            summary_frame.summary = Some(summary);
            let first_retained = active
                .iter()
                .find_map(|frame| frame.runtime_frame_id.filter(|id| !selected.contains(id)));
            let insertion = first_retained
                .and_then(|id| snapshot.frames.iter().position(|frame| frame.id == id))
                .unwrap_or(snapshot.frames.len());
            snapshot.frames.insert(insertion, summary_frame);
        }
        snapshot
            .compaction
            .compacted_frame_ids
            .extend(selected.iter().copied());
        snapshot.compaction.compacted_frame_ids.sort();
        snapshot.compaction.compacted_frame_ids.dedup();
        snapshot
            .compaction
            .retired_source_spans
            .extend(selection.retired_source_spans.iter().copied());
        snapshot.compaction.retired_source_spans =
            merge_runtime_source_spans(snapshot.compaction.retired_source_spans.iter().copied());
        snapshot
            .context_view
            .apply_retired_spans(&snapshot.compaction.retired_source_spans);
        snapshot.active_context.open_detail_block_id =
            snapshot.context_view.provider_open_detail_block_id();
        snapshot.active_context.visible_block_ids =
            snapshot.context_view.provider_visible_block_ids();
        snapshot.active_context.pinned_block_ids =
            snapshot.context_view.provider_pinned_block_ids();
        snapshot.validate_references()?;
        Ok(snapshot)
    }

    fn commit_prepared_runtime_compaction(
        &mut self,
        snapshot: RuntimeSnapshot,
        protocol_frames: Vec<crate::protocol_frames::ProtocolFrame>,
        history: Vec<HistoryItem>,
    ) {
        self.runtime_snapshot = snapshot;
        self.protocol_frames = protocol_frames;
        self.history = history;
        self.needs_compaction = false;
    }

    fn sync_protocol_caches_from_runtime_snapshot(&mut self) -> Result<()> {
        self.protocol_frames = self.runtime_snapshot.active_protocol_frames();
        self.history = crate::protocol_frames::history_items_from_frames(&self.protocol_frames);
        crate::protocol_frames::analyze_history_items(
            &self.history,
            self.turn.current_turn_start_index,
        )?;
        Ok(())
    }

    fn append_protocol_frame(
        &mut self,
        mut frame: crate::protocol_frames::ProtocolFrame,
    ) -> Result<()> {
        self.ensure_protocol_frame_append_allowed(&frame.item)?;
        frame.history_index = self.protocol_frames.len();
        let mut candidate_frames = self.protocol_frames.clone();
        candidate_frames.push(frame);
        self.validate_protocol_frames_candidate(&candidate_frames)?;
        let previous_protocol_frame_count = self.protocol_frames.len();
        let previous_protocol_frames =
            std::mem::replace(&mut self.protocol_frames, candidate_frames);
        if let Err(error) =
            self.refresh_history_cache_from_protocol_frames(previous_protocol_frame_count)
        {
            self.protocol_frames = previous_protocol_frames;
            return Err(error);
        }
        Ok(())
    }

    fn validate_protocol_frames_candidate(
        &self,
        frames: &[crate::protocol_frames::ProtocolFrame],
    ) -> Result<()> {
        let history = crate::protocol_frames::history_items_from_frames(frames);
        crate::protocol_frames::analyze_history_items(
            &history,
            self.turn.current_turn_start_index,
        )?;
        Ok(())
    }

    fn ensure_protocol_frame_append_allowed(
        &self,
        next_item: &crate::protocol_frames::ProtocolFrameItem,
    ) -> Result<()> {
        let transcript = crate::protocol_frames::analyze_history_items(
            &self.history,
            self.turn.current_turn_start_index,
        )?;
        if transcript.has_incomplete_tool_call_groups()
            && !matches!(
                next_item,
                crate::protocol_frames::ProtocolFrameItem::ToolOutput { .. }
            )
        {
            bail!(
                "cannot append {:?} while assistant tool call group is incomplete",
                next_item
            );
        }
        Ok(())
    }

    fn rebuild_protocol_state_from_history(&mut self) -> Result<()> {
        let transcript = crate::protocol_frames::analyze_history_items(
            &self.history,
            self.turn.current_turn_start_index,
        )?;
        let previous_protocol_frame_count = self.protocol_frames.len();
        self.protocol_frames = transcript.frames;
        self.refresh_history_cache_from_protocol_frames(previous_protocol_frame_count)?;
        self.validate_protocol_frames()
    }

    fn refresh_history_cache_from_protocol_frames(
        &mut self,
        previous_protocol_frame_count: usize,
    ) -> Result<()> {
        let old_history = self.history.clone();
        let history = crate::protocol_frames::history_items_from_frames(&self.protocol_frames);
        let runtime_snapshot = self.rebuilt_runtime_snapshot_from_protocol_frames(
            &self.protocol_frames,
            previous_protocol_frame_count,
            &old_history,
        )?;
        self.history = history;
        self.runtime_snapshot = runtime_snapshot;
        for (frame, projected) in self
            .protocol_frames
            .iter_mut()
            .zip(self.runtime_snapshot.active_protocol_frames())
        {
            frame.runtime_frame_id = projected.runtime_frame_id;
        }
        Ok(())
    }

    fn validate_protocol_frames(&self) -> Result<()> {
        crate::protocol_frames::analyze_history_items(
            &self.history,
            self.turn.current_turn_start_index,
        )?;
        Ok(())
    }

    fn fresh_runtime_snapshot(model: &str) -> RuntimeSnapshot {
        RuntimeSnapshot::new(ROOT_CONTEXT_BRANCH_ID).with_latest_model(model.to_string())
    }

    fn validate_evidence_ids(evidence: &[EvidenceRecord]) -> Result<()> {
        let mut candidate = Vec::with_capacity(evidence.len());
        for record in evidence {
            require_unique_evidence_id(&candidate, &record.id)?;
            candidate.push(record.clone());
        }
        Ok(())
    }

    fn rebuilt_runtime_snapshot_from_protocol_frames(
        &self,
        protocol_frames: &[crate::protocol_frames::ProtocolFrame],
        _previous_protocol_frame_count: usize,
        _old_history: &[HistoryItem],
    ) -> Result<RuntimeSnapshot> {
        // Runtime snapshots contain considerably more than protocol history. Keep
        // the restored context tree, view, compaction, child and contributor data;
        // only replace the leading protocol-derived frame portion.
        let mut snapshot = self.runtime_snapshot.clone();
        let old_frames = std::mem::take(&mut snapshot.frames);
        // Protocol caches retain their runtime IDs.  Never recover identity by
        // payload equality, call id, or a storage offset: equal messages are
        // distinct frames and metadata may be interspersed with protocol data.
        let old_by_id = old_frames
            .iter()
            .map(|frame| (frame.id, frame))
            .collect::<HashMap<_, _>>();
        let preserved_frames = old_frames
            .iter()
            // Retired protocol frames are durable compaction records, not cache
            // entries. Keep their exact identity, payload, provenance and span.
            .filter(|frame| {
                frame.protocol.is_none() || frame.visibility == FrameVisibility::Retired
            })
            .cloned()
            .collect::<Vec<_>>();
        snapshot.frames.clear();
        if snapshot.latest_model.is_none() {
            snapshot.latest_model = Some(self.model.clone());
        }
        if self.turn.turn_id != 0 || snapshot.current_turn_id.is_none() {
            snapshot.current_turn_id = Some(self.turn.turn_id);
        }
        for (ordinal, frame) in protocol_frames.iter().enumerate() {
            let runtime_frame = frame
                .runtime_frame_id
                .and_then(|id| old_by_id.get(&id).cloned())
                .cloned()
                .map(|mut existing| {
                    existing.protocol = Some(frame.item.clone());
                    existing
                })
                .unwrap_or_else(|| runtime_frame_from_protocol_frame(frame, ordinal as u32));
            snapshot.push_frame(runtime_frame);
        }
        snapshot.frames.extend(preserved_frames);
        let transcript = crate::protocol_frames::analyze_history_items(
            &crate::protocol_frames::history_items_from_frames(protocol_frames),
            self.turn.current_turn_start_index,
        )?;
        let current_turn_start = self
            .turn
            .current_turn_start_index
            .unwrap_or(protocol_frames.len())
            .min(protocol_frames.len());
        // A live turn is an atomic request boundary, even when it contains no
        // tool calls.  Protect every protocol frame from its start onward.
        let mut turn_protected_frame_ids = Vec::new();
        turn_protected_frame_ids.extend(
            snapshot.frames[..protocol_frames.len()]
                .iter()
                .skip(current_turn_start)
                .map(|frame| frame.id),
        );
        turn_protected_frame_ids.extend(
            transcript
                .protected_history_indexes()
                .into_iter()
                .filter_map(|index| snapshot.frames.get(index).map(|frame| frame.id))
                .collect::<Vec<_>>(),
        );
        turn_protected_frame_ids.sort();
        turn_protected_frame_ids.dedup();
        let active_frame_ids = snapshot
            .frames
            .iter()
            .map(|frame| frame.id)
            .collect::<HashSet<_>>();
        turn_protected_frame_ids.retain(|id| active_frame_ids.contains(id));
        snapshot.set_turn_protected_frame_ids(turn_protected_frame_ids);
        validate_runtime_snapshot_correspondence(
            &crate::protocol_frames::history_items_from_frames(protocol_frames),
            &snapshot,
        )?;
        Ok(snapshot)
    }

    pub fn set_context_experiment_restore_point(
        &mut self,
        scope: ActiveContextExperiment,
        protocol_frames: Vec<crate::protocol_frames::ProtocolFrame>,
        runtime_snapshot: RuntimeSnapshot,
    ) {
        self.context_experiment_restore_point = Some(ContextExperimentRestorePoint {
            scope,
            protocol_frames,
            runtime_snapshot,
        });
    }

    fn tool_execution_context_for(
        &self,
        tool_name: &str,
        allow_outside_workspace: bool,
    ) -> Result<ToolExecutionContext> {
        let mut context = if allow_outside_workspace {
            ToolExecutionContext::outside_workspace_granted()
        } else {
            ToolExecutionContext::default()
        };
        context.question_handler = self.question_handler.clone();

        if !is_context_tool_name(tool_name) {
            return Ok(context);
        }

        context.runtime_snapshot = Some(Arc::new(self.runtime_snapshot.clone()));
        Ok(context)
    }

    pub fn session_title_agent(&self) -> Agent<C>
    where
        C: Clone,
    {
        Agent {
            client: self.client.clone(),
            model: self.model.clone(),
            subagent_model_overrides: HashMap::new(),
            default_protocol: self.default_protocol,
            model_protocols: self.model_protocols.clone(),
            model_catalog: self.model_catalog.clone(),
            prelude: vec![PromptMessage::developer(SESSION_TITLE_PRELUDE)],
            protocol_frames: Vec::new(),
            history: Vec::new(),
            runtime_snapshot: Self::fresh_runtime_snapshot(&self.model),
            tools: ToolRegistry::new(),
            skill_registry: None,
            skill_cards: Vec::new(),
            subagent_delegate: None,
            question_handler: None,
            permission_session: Arc::new(Mutex::new(PermissionSessionState::default())),
            compaction_config: CompactionConfig::default(),
            automatic_checkpoint_policy: automatic_checkpoint::AutoCheckpointPolicy::from_config(
                LogicalCheckpointConfig::default(),
            ),
            retry_config: self.retry_config.clone(),
            tool_timeout_secs: self.tool_timeout_secs,
            needs_compaction: false,
            turn: TurnRuntimeState::default(),
            next_turn_id: 0,
            max_iterations: Some(1),
            max_tool_calls: Some(0),
            context_scope_state: Arc::new(std::sync::Mutex::new(ContextScopeState::default())),
            runtime_snapshot_provider: None,
            logical_checkpoint_candidate_provider: None,
            context_experiment_restore_point: None,
            logical_checkpoint_control: LogicalCheckpointControl {
                state: Arc::new(Mutex::new(LogicalCheckpointControlState {
                    enabled: false,
                    request: LogicalCheckpointRequestState::Idle,
                    request_run_id: None,
                    active_run_id: None,
                    next_run_id: 0,
                    next_request_id: 0,
                    request_id: None,
                    automatic_enabled: false,
                })),
            },
        }
    }

    pub async fn generate_session_title(&mut self, user_input: &str) -> Result<String>
    where
        C: Clone,
    {
        let raw = self
            .run_stream(
                user_input,
                |_| Ok(()),
                |_| Ok(()),
                |_| Ok(PermissionApproval::Deny),
            )
            .await?;
        normalize_session_title(&raw)
    }

    #[allow(dead_code)]
    pub async fn run(&mut self, user_input: &str) -> Result<String>
    where
        C: Clone,
    {
        self.run_stream(
            user_input,
            |_| Ok(()),
            |_| Ok(()),
            |_| Ok(PermissionApproval::AllowOnce),
        )
        .await
    }

    pub async fn run_stream_async<F, E, A, Dfut, Efut, Afut>(
        &mut self,
        user_input: &str,
        on_delta: F,
        on_event: E,
        approve: A,
    ) -> Result<String>
    where
        F: FnMut(&str) -> Dfut,
        E: FnMut(AgentEvent) -> Efut,
        A: FnMut(PermissionRequest) -> Afut,
        Dfut: Future<Output = Result<()>>,
        Efut: Future<Output = Result<()>>,
        Afut: Future<Output = Result<PermissionApproval>>,
        C: Clone,
    {
        self.run_stream_content_with_interactions_async(
            UserMessageContent::new(user_input.to_string(), Vec::new()),
            on_delta,
            on_event,
            approve,
            |request| async move {
                Err(anyhow!(
                    "question tool requires an interactive runtime; received {} question(s)",
                    request.questions.len()
                ))
            },
        )
        .await
    }

    pub async fn run_stream_content_async<F, E, A, Dfut, Efut, Afut>(
        &mut self,
        user_content: UserMessageContent,
        on_delta: F,
        on_event: E,
        approve: A,
    ) -> Result<String>
    where
        F: FnMut(&str) -> Dfut,
        E: FnMut(AgentEvent) -> Efut,
        A: FnMut(PermissionRequest) -> Afut,
        Dfut: Future<Output = Result<()>>,
        Efut: Future<Output = Result<()>>,
        Afut: Future<Output = Result<PermissionApproval>>,
        C: Clone,
    {
        self.run_stream_content_with_interactions_async(
            user_content,
            on_delta,
            on_event,
            approve,
            |request| async move {
                Err(anyhow!(
                    "question tool requires an interactive runtime; received {} question(s)",
                    request.questions.len()
                ))
            },
        )
        .await
    }

    pub async fn run_stream_content_with_interactions_async<F, E, A, Q, Dfut, Efut, Afut, Qfut>(
        &mut self,
        user_content: UserMessageContent,
        on_delta: F,
        on_event: E,
        approve: A,
        ask_question: Q,
    ) -> Result<String>
    where
        F: FnMut(&str) -> Dfut,
        E: FnMut(AgentEvent) -> Efut,
        A: FnMut(PermissionRequest) -> Afut,
        Q: FnMut(QuestionRequest) -> Qfut + Send + 'static,
        Dfut: Future<Output = Result<()>>,
        Efut: Future<Output = Result<()>>,
        Afut: Future<Output = Result<PermissionApproval>>,
        Qfut: Future<Output = Result<QuestionResponse>> + Send + 'static,
        C: Clone,
    {
        let mut question_handler_guard =
            QuestionHandlerGuard::install(self, Some(Self::wrap_question_handler(ask_question)));

        let user_input = user_content.text.clone();
        let result = match question_handler_guard.agent().active_protocol() {
            ApiProtocol::Responses => {
                protocol_stream::run_responses_stream_async(
                    question_handler_guard.agent(),
                    user_content.clone(),
                    &user_input,
                    on_delta,
                    on_event,
                    approve,
                )
                .await
            }
            ApiProtocol::Completions => {
                protocol_stream::run_oai_comp_stream_async(
                    question_handler_guard.agent(),
                    user_content,
                    &user_input,
                    on_delta,
                    on_event,
                    approve,
                )
                .await
            }
        };
        if let Err(error) = &result {
            question_handler_guard
                .agent()
                .note_context_overflow_error(error);
        }
        result
    }

    fn wrap_question_handler<Q, Qfut>(ask_question: Q) -> QuestionCallback
    where
        Q: FnMut(QuestionRequest) -> Qfut + Send + 'static,
        Qfut: Future<Output = Result<QuestionResponse>> + Send + 'static,
    {
        let ask_question = Arc::new(tokio::sync::Mutex::new(ask_question));
        Arc::new(move |request: QuestionRequest| {
            let ask_question = Arc::clone(&ask_question);
            Box::pin(async move {
                let mut callback = ask_question.lock().await;
                (*callback)(request).await
            })
        })
    }

    fn note_context_overflow_error(&mut self, error: &anyhow::Error) {
        if is_context_overflow_error(error) {
            self.needs_compaction = true;
        }
    }

    async fn execute_tool_call<E, A, Efut, Afut>(
        &mut self,
        call: &HistoryToolCall,
        on_event: &mut E,
        approve: &mut A,
    ) -> Result<ToolExecutionRecord>
    where
        E: FnMut(AgentEvent) -> Efut,
        A: FnMut(PermissionRequest) -> Afut,
        Efut: Future<Output = Result<()>>,
        Afut: Future<Output = Result<PermissionApproval>>,
    {
        tool_execution::execute_tool_call(self, call, on_event, approve).await
    }

    async fn execute_subagent_tool(&self, tool_name: &str, args: &Value) -> ToolResult {
        let input = match normalize_subagent_input(tool_name, args) {
            Ok(input) => input,
            Err(error) => return ToolResult::err(tool_name, error.to_string()),
        };

        let Some(delegate) = self.subagent_delegate.clone() else {
            return ToolResult::err(
                tool_name,
                format!("{tool_name} is unavailable outside a subagent-capable runtime"),
            );
        };

        let task = self.render_subagent_prompt(tool_name, &input);
        let invocation = SubagentInvocation {
            input,
            prompt: task,
        };
        let Some(agent_name) = agent_name_for_subagent_tool(tool_name) else {
            return ToolResult::err(tool_name, format!("unknown subagent tool: {tool_name}"));
        };

        let result = delegate.run_named(self, agent_name, invocation).await;

        match result {
            Ok(result) => result,
            Err(error) => ToolResult::err(tool_name, error.to_string()),
        }
    }

    fn render_subagent_prompt(&self, tool_name: &str, input: &NormalizedSubagentInput) -> String {
        format!(
            "{}\n\nReturn only a single JSON object with fields: status, summary, findings, files_read, files_changed, commands_run, validation, blockers, next_steps.",
            input.render_for_delegate(tool_name)
        )
    }

    fn tool_definitions(&self) -> Vec<crate::request_builder::ToolSpec> {
        let mut specs = self.tools.specs();
        // ToolRegistry retains a pair of legacy subagent handlers for validation and
        // scope compatibility. Catalog tools are advertised only when their delegate
        // is executable, exactly as they are checked at execution time.
        specs.retain(|spec| !is_subagent_tool_name(&spec.name));
        specs.extend(
            subagent_tool_specs()
                .into_iter()
                .filter(|spec| is_executable_tool(self, &spec.name)),
        );
        specs
    }

    fn ensure_tool_call_budget(&self, current_count: usize, requested_count: usize) -> Result<()> {
        let total_count = current_count + requested_count;
        if let Some(limit) = self.max_tool_calls
            && total_count > limit
        {
            return Err(anyhow!(
                "stopped: too many tool calls ({} requested, max {})",
                total_count,
                limit
            ));
        }

        Ok(())
    }

    fn append_assistant_tool_calls(
        &mut self,
        turn_text: &str,
        tool_calls: &[HistoryToolCall],
    ) -> Result<()> {
        self.validate_assistant_tool_call_batch(tool_calls)?;
        self.append_history_item(HistoryItem::AssistantToolCalls {
            text: if turn_text.is_empty() {
                None
            } else {
                Some(turn_text.to_string())
            },
            calls: tool_calls.to_vec(),
        })
        .map_err(|error| anyhow!("assistant tool calls should remain protocol-compatible: {error}"))
    }

    fn validate_assistant_tool_call_batch(&self, tool_calls: &[HistoryToolCall]) -> Result<()> {
        if tool_calls.len() > 1
            && tool_calls.iter().any(|call| {
                matches!(
                    call.name.as_str(),
                    tool_names::TOOL_CONTEXT_CHECKPOINT | tool_names::TOOL_CONTEXT_RETURN
                )
            })
        {
            let batched_context_tool = tool_calls
                .iter()
                .find(|call| {
                    matches!(
                        call.name.as_str(),
                        tool_names::TOOL_CONTEXT_CHECKPOINT | tool_names::TOOL_CONTEXT_RETURN
                    )
                })
                .expect("batched context tool must exist when batch validation fails");

            bail!(
                "{} cannot be batched with other tool calls in the same assistant tool-call group",
                batched_context_tool.name
            );
        }
        Ok(())
    }

    async fn execute_tool_call_and_record<E, A, Efut, Afut>(
        &mut self,
        call: &HistoryToolCall,
        on_event: &mut E,
        approve: &mut A,
    ) -> Result<()>
    where
        E: FnMut(AgentEvent) -> Efut,
        A: FnMut(PermissionRequest) -> Afut,
        Efut: Future<Output = Result<()>>,
        Afut: Future<Output = Result<PermissionApproval>>,
    {
        let record = self.execute_tool_call(call, on_event, approve).await?;

        debug!(
            tool_name = %call.name,
            call_id = %call.call_id,
            output = ?record.output,
            effects = ?record.effects,
            "tool call completed"
        );

        let output_json = serde_json::to_string(&record.output)?;
        self.append_history_item(HistoryItem::ToolOutput {
            call_id: call.call_id.clone(),
            output_json,
        })?;
        reconcile_loaded_skill_material(&mut self.runtime_snapshot)?;

        if record.output.ok {
            match call.name.as_str() {
                tool_names::TOOL_CONTEXT_CHECKPOINT => {
                    self.finalize_context_checkpoint_after_recording()?;
                }
                tool_names::TOOL_CONTEXT_RETURN => {
                    self.finalize_context_return_after_recording()?;
                }
                _ if is_context_tool_name(&call.name)
                    && record
                        .output
                        .data
                        .as_ref()
                        .and_then(|data| data.get("pending_recording"))
                        .and_then(Value::as_bool)
                        == Some(true) =>
                {
                    self.refresh_runtime_snapshot_from_provider()?;
                }
                _ => {}
            }
        }

        debug!(
            history_len = self.history.len(),
            "tool output appended to history"
        );

        let evidence = self.remember_tool_evidence(&record)?;
        on_event(AgentEvent::EvidenceRecorded(evidence)).await?;

        if is_cancelled_subagent_record(&record) {
            return Err(anyhow!("{} cancelled", record.tool_name));
        }

        Ok(())
    }

    fn remember_tool_evidence(&mut self, record: &ToolExecutionRecord) -> Result<EvidenceRecord> {
        evidence_memory::remember_tool_evidence(self, record)
    }

    fn next_evidence_sequence(&self) -> u64 {
        evidence_memory::next_evidence_sequence(self)
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
        A: FnMut(PermissionRequest) -> Result<PermissionApproval>,
        C: Clone,
    {
        self.run_stream_async(
            user_input,
            |delta| std::future::ready(on_delta(delta)),
            |event| std::future::ready(on_event(event)),
            |request| std::future::ready(approve(request)),
        )
        .await
    }

    pub async fn run_stream_with_interactions<F, E, A, Q>(
        &mut self,
        user_input: &str,
        mut on_delta: F,
        mut on_event: E,
        mut approve: A,
        mut ask_question: Q,
    ) -> Result<String>
    where
        F: FnMut(&str) -> Result<()>,
        E: FnMut(AgentEvent) -> Result<()>,
        A: FnMut(PermissionRequest) -> Result<PermissionApproval>,
        Q: FnMut(QuestionRequest) -> Result<QuestionResponse> + Send + 'static,
        C: Clone,
    {
        self.run_stream_content_with_interactions_async(
            UserMessageContent::new(user_input.to_string(), Vec::new()),
            |delta| std::future::ready(on_delta(delta)),
            |event| std::future::ready(on_event(event)),
            |request| std::future::ready(approve(request)),
            move |request| std::future::ready(ask_question(request)),
        )
        .await
    }

    pub async fn compact_session_async<E, Efut>(
        &mut self,
        on_event: E,
    ) -> Result<ManualCompactionOutcome>
    where
        E: FnMut(AgentEvent) -> Efut,
        Efut: Future<Output = Result<()>>,
        C: Clone,
    {
        compaction::compact_session_stream_async(self, on_event, || Ok(()), |_| Ok(())).await
    }

    pub async fn compact_session_stream_async<E, Efut, S, D>(
        &mut self,
        on_event: E,
        on_start: S,
        on_delta: D,
    ) -> Result<ManualCompactionOutcome>
    where
        E: FnMut(AgentEvent) -> Efut,
        Efut: Future<Output = Result<()>>,
        S: FnMut() -> Result<()> + Send,
        D: FnMut(&str) -> Result<()> + Send,
        C: Clone,
    {
        compaction::compact_session_stream_async(self, on_event, on_start, on_delta).await
    }

    async fn run_oai_comp_stream_async<F, E, A, Dfut, Efut, Afut>(
        &mut self,
        user_input: &str,
        on_delta: F,
        on_event: E,
        approve: A,
    ) -> Result<String>
    where
        F: FnMut(&str) -> Dfut,
        E: FnMut(AgentEvent) -> Efut,
        A: FnMut(PermissionRequest) -> Afut,
        Dfut: Future<Output = Result<()>>,
        Efut: Future<Output = Result<()>>,
        Afut: Future<Output = Result<PermissionApproval>>,
        C: Clone,
    {
        protocol_stream::run_oai_comp_stream_async(
            self,
            UserMessageContent::new(user_input.to_string(), Vec::new()),
            user_input,
            on_delta,
            on_event,
            approve,
        )
        .await
    }

    async fn preflight_compact_context<E, Efut>(
        &mut self,
        turn_prelude: &[PromptMessage],
        protected_start_index: usize,
        tool_definitions: &[crate::request_builder::ToolSpec],
        on_event: &mut E,
    ) -> Result<usize>
    where
        E: FnMut(AgentEvent) -> Efut,
        Efut: Future<Output = Result<()>>,
        C: Clone,
    {
        compaction::preflight_compact_context(
            self,
            turn_prelude,
            protected_start_index,
            tool_definitions,
            on_event,
        )
        .await
    }

    fn prune_old_tool_outputs(&mut self, preserve_recent_budget: u64) -> Result<()> {
        compaction::prune_old_tool_outputs(self, preserve_recent_budget)
    }

    fn prepare_turn_prelude(&mut self, user_input: &str) -> Vec<PromptMessage> {
        let turn = WorkflowTurnState::from_user_input(user_input);
        self.next_turn_id = self.next_turn_id.saturating_add(1);
        self.turn = TurnRuntimeState::new(self.next_turn_id, turn.clone());
        self.runtime_snapshot.current_turn_id = Some(self.next_turn_id);

        let mut turn_prelude = self.prelude.clone();
        turn_prelude.push(runtime_context_message());
        if let Some(message) = self.skill_prelude_message() {
            turn_prelude.push(message);
        }
        if let Some(message) = turn.developer_context_message() {
            turn_prelude.push(message);
        }
        if let Some(message) = self.unreconciled_subagent_context_message() {
            turn_prelude.push(message);
        }
        turn_prelude
    }

    fn skill_prelude_message(&self) -> Option<PromptMessage> {
        if self.skill_cards.is_empty() {
            return None;
        }

        let mut text = String::from(
            "Available local skills:\nLoad relevant skills with the `skill` tool when needed. Do not load skills speculatively. Skills do not change permissions or expand tool scope.",
        );
        for card in &self.skill_cards {
            text.push_str(&format!(
                "\n- {} — {} (source: {})",
                card.name, card.description, card.location
            ));
        }

        Some(PromptMessage::developer_with_origin(
            text,
            PromptMessageOrigin::SkillCatalog,
        ))
    }

    async fn apply_control_tool_state<E, Efut>(
        &mut self,
        tool_name: &str,
        args: &Value,
        on_event: &mut E,
    ) -> Result<()>
    where
        E: FnMut(AgentEvent) -> Efut,
        Efut: Future<Output = Result<()>>,
    {
        match tool_name {
            "workflow__todos" => {
                let payload: WorkflowTodosPayload = serde_json::from_value(args.clone())?;
                on_event(AgentEvent::TodoSnapshotUpdated {
                    items: payload.items.clone(),
                })
                .await?;
                self.turn.workflow.todos = payload.items;
            }
            "workflow__auto_continue" => {
                let payload: WorkflowAutoContinuePayload = serde_json::from_value(args.clone())?;
                let mut next_state = self.turn.workflow.auto_continue.clone();
                next_state.enabled = payload.enabled;
                if let Some(max_continuations) = payload.max_continuations {
                    if max_continuations > AutoContinueState::ABSOLUTE_MAX_CONTINUATIONS {
                        return Err(anyhow!(
                            "max_continuations {max_continuations} exceeds maximum {}",
                            AutoContinueState::ABSOLUTE_MAX_CONTINUATIONS
                        ));
                    }
                    next_state.max_continuations = max_continuations;
                }
                on_event(AgentEvent::AutoContinueChanged {
                    state: next_state.clone(),
                })
                .await?;
                self.turn.workflow.auto_continue = next_state;
            }
            _ => {}
        }

        Ok(())
    }

    fn validate_context_control_tool(&self, tool_name: &str) -> Result<()> {
        let state = self
            .context_scope_state
            .lock()
            .map_err(|_| anyhow!("context scope state poisoned"))?;
        match tool_name {
            tool_names::TOOL_CONTEXT_CHECKPOINT => {
                ensure!(
                    state.active_experiment.is_none(),
                    "context__checkpoint cannot start a nested experiment while another experiment is active"
                );
            }
            tool_names::TOOL_CONTEXT_RETURN => {
                ensure!(
                    state.active_experiment.is_some(),
                    "context__return requires an active context experiment"
                );
            }
            _ => {}
        }
        Ok(())
    }

    fn finalize_context_checkpoint_after_recording(&mut self) -> Result<()> {
        crate::protocol_frames::validate_history_items_complete(&self.history, None).map_err(
            |error| {
                anyhow!(
                    "context__checkpoint restore point requires a complete assistant tool-call group: {error}"
                )
            },
        )?;
        let experiment = self
            .context_scope_state
            .lock()
            .map_err(|_| anyhow!("context scope state poisoned"))?
            .active_experiment
            .clone()
            .ok_or_else(|| anyhow!("context__checkpoint did not activate a context experiment"))?;
        let restore_point = ContextExperimentRestorePoint {
            scope: experiment,
            protocol_frames: self.protocol_frames.clone(),
            runtime_snapshot: self.runtime_snapshot.clone(),
        };
        self.replace_runtime_snapshot_from_provider()?;
        self.context_experiment_restore_point = Some(restore_point);
        Ok(())
    }

    fn finalize_context_return_after_recording(&mut self) -> Result<()> {
        self.context_experiment_restore_point
            .clone()
            .ok_or_else(|| anyhow!("context__return restore point is missing"))?;
        self.replace_runtime_snapshot_from_provider()?;
        self.context_experiment_restore_point = None;
        Ok(())
    }

    fn finalize_turn_decision(&self, continuation_count: usize) -> FinalizeDecision {
        let Some(remaining_unfinished) = self.remaining_unfinished_todos() else {
            return FinalizeDecision::Finish;
        };

        if !self.turn.workflow.auto_continue.enabled {
            return FinalizeDecision::Finish;
        }

        if continuation_count >= self.turn.workflow.auto_continue.max_continuations {
            return FinalizeDecision::StopWithError {
                message: format!(
                    "stopped: auto-continue limit reached (max {}, {} unfinished todo item{})",
                    self.turn.workflow.auto_continue.max_continuations,
                    remaining_unfinished,
                    if remaining_unfinished == 1 { "" } else { "s" }
                ),
            };
        }

        if self
            .turn
            .last_continuation_todos
            .as_ref()
            .is_some_and(|previous| previous == &self.turn.workflow.todos)
        {
            return FinalizeDecision::StopWithError {
                message: format!(
                    "stopped: auto-continue made no todo progress ({} unfinished todo item{})",
                    remaining_unfinished,
                    if remaining_unfinished == 1 { "" } else { "s" }
                ),
            };
        }

        FinalizeDecision::Continue {
            remaining_unfinished,
        }
    }

    async fn continue_after_no_tool_reply<E, Efut>(
        &mut self,
        on_event: &mut E,
        continuation_count: &mut usize,
    ) -> Result<bool>
    where
        E: FnMut(AgentEvent) -> Efut,
        Efut: Future<Output = Result<()>>,
    {
        match self.finalize_turn_decision(*continuation_count) {
            FinalizeDecision::Finish => Ok(false),
            FinalizeDecision::StopWithError { message } => Err(anyhow!(message)),
            FinalizeDecision::Continue {
                remaining_unfinished,
            } => {
                *continuation_count += 1;
                self.turn.counters.continuations = *continuation_count;
                self.turn.last_continuation_todos = Some(self.turn.workflow.todos.clone());
                let text = "Continue the current task internally. Do not repeat finished work. Focus on unfinished todo items and stop when they are complete or blocked.".to_string();
                self.append_history_item(HistoryItem::internal_continuation(text.clone()))?;
                on_event(AgentEvent::InternalContinuation {
                    text,
                    source: crate::transcript::InternalContinuationSource::AutoContinue,
                })
                .await?;
                on_event(AgentEvent::AutoContinuationScheduled {
                    continuation_count: *continuation_count,
                    remaining_unfinished,
                })
                .await?;
                Ok(true)
            }
        }
    }

    async fn continue_or_finalize_no_tool_reply<E, Efut>(
        &mut self,
        on_event: &mut E,
        tool_call_count: usize,
        continuation_count: &mut usize,
    ) -> Result<bool>
    where
        E: FnMut(AgentEvent) -> Efut,
        Efut: Future<Output = Result<()>>,
    {
        self.ensure_no_open_context_experiment_before_finalization()?;

        if self
            .continue_after_no_tool_reply(on_event, continuation_count)
            .await?
        {
            return Ok(true);
        }

        let validation_advisory_emitted = self.emit_validation_advisory_if_needed(on_event).await?;

        Self::emit_audit_event(
            on_event,
            AgentEvent::TurnFinalized(self.turn_finalized_event(
                "completed",
                tool_call_count,
                *continuation_count,
                validation_advisory_emitted,
            )),
            "turn_finalized",
        )
        .await;

        self.finish_current_turn()?;

        Ok(false)
    }

    async fn emit_validation_advisory_if_needed<E, Efut>(&self, on_event: &mut E) -> Result<bool>
    where
        E: FnMut(AgentEvent) -> Efut,
        Efut: Future<Output = Result<()>>,
    {
        let Some(advisory) = self.pending_validation_advisory() else {
            return Ok(false);
        };

        on_event(AgentEvent::ValidationAdvisory(advisory)).await?;
        Ok(true)
    }

    async fn emit_audit_event<E, Efut>(
        on_event: &mut E,
        event: AgentEvent,
        event_kind: &'static str,
    ) where
        E: FnMut(AgentEvent) -> Efut,
        Efut: Future<Output = Result<()>>,
    {
        if let Err(error) = on_event(event).await {
            warn!(
                error = %error,
                event_kind,
                "audit event handler failed; continuing agent turn"
            );
        }
    }

    fn turn_started_event(&self) -> TurnStartedEvent {
        TurnStartedEvent {
            turn_id: self.turn.turn_id,
            intent: self.turn.policy.intent.as_str().to_string(),
            directive: self.turn.policy.directive.as_str().to_string(),
            validation_reminder: self.turn.policy.validation.as_str().to_string(),
        }
    }

    fn turn_finalized_event(
        &self,
        outcome: &str,
        tool_call_count: usize,
        continuation_count: usize,
        validation_advisory_emitted: bool,
    ) -> TurnFinalizedEvent {
        TurnFinalizedEvent {
            turn_id: self.turn.turn_id,
            outcome: outcome.to_string(),
            tool_call_count,
            continuation_count,
            write_effects: self.turn.counters.write_effects,
            validation_effects: self.turn.counters.validation_effects,
            failed_validation_effects: self.turn.counters.failed_validation_effects,
            validation_advisory_emitted,
        }
    }

    fn finish_current_turn(&mut self) -> Result<()> {
        self.logical_checkpoint_control.clear();
        self.turn.automatic_checkpoint.reset_for_turn_end();
        self.turn.current_turn_start_index = None;
        self.runtime_snapshot.current_turn_id = None;
        self.runtime_snapshot = self.rebuilt_runtime_snapshot_from_protocol_frames(
            &self.protocol_frames,
            self.protocol_frames.len(),
            &self.history,
        )?;
        self.runtime_snapshot.current_turn_id = None;
        Ok(())
    }

    fn tool_execution_summary_event(
        &self,
        record: &ToolExecutionRecord,
    ) -> ToolExecutionSummaryEvent {
        ToolExecutionSummaryEvent {
            turn_id: self.turn.turn_id,
            call_id: record.call_id.clone(),
            name: record.tool_name.clone(),
            status: record.status.as_str().to_string(),
            rejection: record
                .rejection
                .map(|rejection| rejection.as_str().to_string()),
            effect_kind: record.effects.kind.as_str().to_string(),
            primary_path: record.effects.primary_path.clone(),
            command: record.effects.command.clone(),
        }
    }

    fn pending_validation_advisory(&self) -> Option<ValidationAdvisory> {
        let write_effects =
            self.turn.counters.write_effects + self.turn.counters.child_write_effects;
        let validation_effects =
            self.turn.counters.validation_effects + self.turn.counters.child_validation_effects;
        let failed_validation_effects = self.turn.counters.failed_validation_effects
            + self.turn.counters.child_failed_validation_effects;

        (write_effects > 0 && (validation_effects == 0 || failed_validation_effects > 0)).then(|| {
            let message = if failed_validation_effects > 0 {
                "This turn made write changes, including delegated child work, and validation ran but failed. Review the failed validation output before relying on the changes; at least one validation failed."
            } else {
                "This turn made write changes, including delegated child work, without running validation. Review and run the most relevant checks if needed."
            };

            ValidationAdvisory {
                write_effects,
                validation_effects,
                failed_validation_effects,
                message: message.into(),
            }
        })
    }

    fn unreconciled_subagent_context_message(&self) -> Option<PromptMessage> {
        let jobs = self.pending_subagent_jobs();
        if jobs.is_empty() {
            return None;
        }
        let mut text = String::from(
            "Pending child subagent results from earlier turns:\nUse agent__reconcile to explicitly record accepted, rejected, or conflict decisions before relying on them.",
        );
        for job in jobs {
            text.push_str(&format!(
                "\n- {} [{}] {} — {} (child {})",
                job.agent_name, job.status, job.run_id, job.summary, job.child_session_id
            ));
        }
        Some(PromptMessage::developer_with_origin(
            text,
            PromptMessageOrigin::UnreconciledSubagentContext,
        ))
    }

    fn pending_subagent_jobs(&self) -> Vec<PendingSubagentJob> {
        let mut jobs = BTreeMap::<String, PendingSubagentJob>::new();
        let mut reconciled = HashSet::new();

        for evidence in &self.runtime_snapshot.evidence {
            let EvidenceSource::Subagent {
                run_id,
                child_session_id,
                parent_tool,
                ..
            } = &evidence.source
            else {
                continue;
            };

            if evidence
                .tags
                .iter()
                .any(|tag| tag == "subagent_reconciliation" || tag == "reconciled")
            {
                reconciled.insert(run_id.clone());
                continue;
            }

            if evidence.tags.iter().any(|tag| tag == "subagent_result") {
                let status = evidence
                    .detail
                    .as_deref()
                    .and_then(|detail| {
                        serde_json::from_str::<crate::subagent::StructuredSubagentResult>(detail)
                            .ok()
                    })
                    .map(|structured| structured.status)
                    .unwrap_or_else(|| "completed".into());
                jobs.insert(
                    run_id.clone(),
                    PendingSubagentJob {
                        run_id: run_id.clone(),
                        child_session_id: child_session_id.clone(),
                        agent_name: parent_tool.trim_start_matches("agent__").to_string(),
                        status,
                        summary: evidence.summary.clone(),
                    },
                );
            }
        }

        jobs.into_iter()
            .filter_map(|(run_id, job)| (!reconciled.contains(&run_id)).then_some(job))
            .collect()
    }

    fn remaining_unfinished_todos(&self) -> Option<usize> {
        if self
            .turn
            .workflow
            .todos
            .iter()
            .any(|todo| todo.status == TodoStatus::Blocked)
        {
            return None;
        }

        let unfinished = self
            .turn
            .workflow
            .todos
            .iter()
            .filter(|todo| todo.status.is_unfinished())
            .count();
        (unfinished > 0).then_some(unfinished)
    }

    fn record_tool_effects(&mut self, record: &ToolExecutionRecord) {
        if is_subagent_tool_name(&record.tool_name) {
            self.record_subagent_effects(record);
        }
        match record.effects.kind {
            ToolEffectKind::Write => {
                self.mark_active_context_experiment_write_observed();
                self.turn.counters.write_effects =
                    self.turn.counters.write_effects.saturating_add(1);
            }
            ToolEffectKind::Validation => {
                self.turn.counters.validation_effects =
                    self.turn.counters.validation_effects.saturating_add(1);
            }
            ToolEffectKind::Diagnostic if is_failed_validation_attempt(record) => {
                self.turn.counters.failed_validation_effects = self
                    .turn
                    .counters
                    .failed_validation_effects
                    .saturating_add(1);
            }
            _ => {}
        }
    }

    fn record_subagent_effects(&mut self, record: &ToolExecutionRecord) {
        let Some(structured) = record
            .output
            .data
            .as_ref()
            .and_then(|data| data.get("structured_result"))
            .cloned()
            .and_then(|value| {
                serde_json::from_value::<crate::subagent::StructuredSubagentResult>(value).ok()
            })
        else {
            return;
        };

        self.turn.counters.child_write_effects = self
            .turn
            .counters
            .child_write_effects
            .saturating_add(structured.files_changed.len());
        let (ran_validation_effects, failed_validation_effects) =
            classify_child_validation_entries(&structured.validation);
        self.turn.counters.child_validation_effects = self
            .turn
            .counters
            .child_validation_effects
            .saturating_add(ran_validation_effects);
        self.turn.counters.child_failed_validation_effects = self
            .turn
            .counters
            .child_failed_validation_effects
            .saturating_add(failed_validation_effects);
    }

    fn ensure_no_open_context_experiment_before_finalization(&self) -> Result<()> {
        let state = self
            .context_scope_state
            .lock()
            .map_err(|_| anyhow!("context scope state poisoned"))?;
        if let Some(experiment) = state.active_experiment.as_ref() {
            bail!(
                "cannot finalize turn while context experiment '{}' is active; return it explicitly with context__return or continue the experiment",
                experiment.branch_id
            );
        }
        Ok(())
    }

    fn mark_active_context_experiment_write_observed(&mut self) {
        if let Ok(mut state) = self.context_scope_state.lock()
            && let Some(experiment) = state.active_experiment.as_mut()
        {
            experiment.writes_observed = true;
            if let Some(restore_point) = self.context_experiment_restore_point.as_mut()
                && restore_point.scope.branch_id == experiment.branch_id
            {
                restore_point.scope.writes_observed = true;
            }
        }
    }
}

struct QuestionHandlerGuard<'a, C: Config> {
    agent: &'a mut Agent<C>,
    previous: Option<QuestionCallback>,
}

impl<'a, C: Config> QuestionHandlerGuard<'a, C> {
    fn install(agent: &'a mut Agent<C>, replacement: Option<QuestionCallback>) -> Self {
        let previous = agent.question_handler.take();
        agent.question_handler = replacement;
        Self { agent, previous }
    }

    fn agent(&mut self) -> &mut Agent<C> {
        self.agent
    }
}

impl<C: Config> Drop for QuestionHandlerGuard<'_, C> {
    fn drop(&mut self) {
        self.agent.question_handler = self.previous.take();
    }
}

fn permission_class_for_tool_call(
    tools: &ToolRegistry,
    tool_name: &str,
) -> crate::permission::ToolPermissionClass {
    subagent_tool_permission_class(tool_name).unwrap_or_else(|| tools.permission_class(tool_name))
}

/// A tool is executable only if a registry handler exists, or it is a catalogued
/// subagent tool backed by an installed delegate. Keep this shared predicate in
/// step with tool advertisement so a model cannot request a virtual tool that
/// will only fail after approval.
fn is_executable_tool<C: Config>(agent: &Agent<C>, tool_name: &str) -> bool {
    match subagent_catalog_entry_by_tool_name(tool_name) {
        Some(_) => agent.subagent_delegate.is_some(),
        None => agent.tools.contains(tool_name),
    }
}

fn subagent_tool_permission_class(
    tool_name: &str,
) -> Option<crate::permission::ToolPermissionClass> {
    let entry = subagent_catalog_entry_by_tool_name(tool_name)?;
    Some(if entry.read_only {
        crate::permission::ToolPermissionClass::Preview
    } else {
        crate::permission::ToolPermissionClass::Write
    })
}

fn is_read_only_subagent_tool_name(name: &str) -> bool {
    subagent_catalog_entry_by_tool_name(name)
        .map(|entry| entry.read_only)
        .unwrap_or(false)
}

pub(crate) fn is_subagent_tool_name(name: &str) -> bool {
    agent_name_for_subagent_tool(name).is_some()
}

pub(crate) fn is_context_tool_name(name: &str) -> bool {
    matches!(
        name,
        tool_names::TOOL_CONTEXT_LIST
            | tool_names::TOOL_CONTEXT_SEARCH
            | tool_names::TOOL_CONTEXT_GREP
            | tool_names::TOOL_CONTEXT_OPEN
            | tool_names::TOOL_CONTEXT_SUMMARIZE
            | tool_names::TOOL_CONTEXT_PIN
            | tool_names::TOOL_CONTEXT_ARCHIVE
            | tool_names::TOOL_CONTEXT_REMOVE
            | tool_names::TOOL_CONTEXT_RESOLVE
    )
}

pub(crate) fn agent_name_for_subagent_tool(tool_name: &str) -> Option<&'static str> {
    subagent_catalog_entry_by_tool_name(tool_name).map(|entry| entry.agent_name)
}

pub(crate) fn subagent_tool_name_for_agent_name(agent_name: &str) -> Option<&'static str> {
    subagent_catalog_entry_by_agent_name(agent_name).map(|entry| entry.tool_name)
}

pub(crate) fn subagent_catalog_entry_by_tool_name(
    tool_name: &str,
) -> Option<&'static SubagentCatalogEntry> {
    SUBAGENT_CATALOG
        .iter()
        .find(|entry| entry.tool_name == tool_name)
}

pub(crate) fn subagent_catalog_entry_by_agent_name(
    agent_name: &str,
) -> Option<&'static SubagentCatalogEntry> {
    SUBAGENT_CATALOG
        .iter()
        .find(|entry| entry.agent_name == agent_name)
}

fn subagent_tool_specs() -> Vec<crate::request_builder::ToolSpec> {
    SUBAGENT_CATALOG
        .iter()
        .map(|entry| crate::request_builder::ToolSpec {
            name: entry.tool_name.to_string(),
            description: entry.tool_description.to_string(),
            parameters: subagent_parameters_schema(entry.task_description),
            strict: true,
        })
        .collect()
}

fn classify_child_validation_entries(entries: &[String]) -> (usize, usize) {
    let mut ran = 0usize;
    let mut failed = 0usize;

    for entry in entries {
        let lower = entry.to_ascii_lowercase();
        let not_run = [
            "not_run",
            "not run",
            "no_validation",
            "no validation",
            "did not run",
            "skipped",
        ]
        .iter()
        .any(|needle| lower.contains(needle));
        if not_run {
            continue;
        }

        let failed_entry = [
            "failed",
            "fail",
            "error",
            "timed_out",
            "cancelled",
            "blocked",
        ]
        .iter()
        .any(|needle| lower.contains(needle));
        let passed_entry = ["passed", "success", "succeeded", "ok"]
            .iter()
            .any(|needle| lower.contains(needle));

        if failed_entry {
            ran = ran.saturating_add(1);
            failed = failed.saturating_add(1);
        } else if passed_entry {
            ran = ran.saturating_add(1);
        }
    }

    (ran, failed)
}

impl ToolExecutionRecord {
    fn new(
        call: &HistoryToolCall,
        arguments: Option<Value>,
        permission_class: crate::permission::ToolPermissionClass,
        directive: ExecutionDirective,
        status: ToolExecutionStatus,
        rejection: Option<ToolExecutionRejection>,
        output: ToolResult,
    ) -> Self {
        let effects = ToolEffects::derive(&call.name, arguments.as_ref(), &output);
        Self {
            call_id: call.call_id.clone(),
            tool_name: call.name.clone(),
            arguments,
            permission_class,
            directive,
            status,
            rejection,
            output,
            effects,
        }
    }
}

impl ToolEffects {
    fn derive(tool_name: &str, arguments: Option<&Value>, output: &ToolResult) -> Self {
        let primary_path = arguments
            .and_then(argument_path)
            .or_else(|| output_string(output, "path"));
        let command = arguments
            .and_then(|args| value_string(args, "command"))
            .or_else(|| output_string(output, "command"));
        let edited_paths = output_edited_paths(output);

        let kind = if !output.ok {
            ToolEffectKind::Diagnostic
        } else {
            if is_read_only_subagent_tool_name(tool_name) {
                ToolEffectKind::Read
            } else {
                match tool_name {
                    "fs__read"
                    | "fs__list"
                    | "skill"
                    | "search__rg"
                    | "code__ast_search"
                    | "git__status"
                    | "git__diff"
                    | "git__log"
                    | "code__ast_replace_preview" => ToolEffectKind::Read,
                    "agent__fixer" | "fs__write" | "fs__append" | "fs__mkdir"
                    | "edit__apply_patch" => ToolEffectKind::Write,
                    "shell__exec" if command.as_deref().is_some_and(is_validation_command_text) => {
                        if shell_command_succeeded(output) {
                            ToolEffectKind::Validation
                        } else {
                            ToolEffectKind::Diagnostic
                        }
                    }
                    "shell__exec" => ToolEffectKind::Command,
                    "workflow__todos"
                    | "workflow__auto_continue"
                    | "context__checkpoint"
                    | "context__return"
                    | "agent__reconcile" => ToolEffectKind::WorkflowControl,
                    _ => ToolEffectKind::Unknown,
                }
            }
        };

        Self {
            kind,
            primary_path,
            edited_paths,
            command,
        }
    }
}

fn argument_path(args: &Value) -> Option<String> {
    value_string(args, "path")
        .or_else(|| value_string(args, "file_path"))
        .or_else(|| value_string(args, "filePath"))
}

fn value_string(value: &Value, key: &str) -> Option<String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

fn output_string(output: &ToolResult, key: &str) -> Option<String> {
    output
        .data
        .as_ref()?
        .get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

fn output_edited_paths(output: &ToolResult) -> Vec<String> {
    output
        .data
        .as_ref()
        .and_then(|data| data.get("edits"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|edit| edit.get("path").and_then(Value::as_str))
        .map(str::to_string)
        .collect()
}

fn required_tool_output_string(output: &ToolResult, field: &str) -> Result<String> {
    let data = output
        .data
        .as_ref()
        .ok_or_else(|| anyhow!("tool output is missing data"))?;
    let value = data
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("tool output field '{field}' must be a string"))?;
    let value = value.trim();
    ensure!(
        !value.is_empty(),
        "tool output field '{field}' must not be empty"
    );
    Ok(value.to_string())
}

fn optional_tool_output_string(output: &ToolResult, field: &str) -> Result<Option<String>> {
    let Some(data) = output.data.as_ref() else {
        return Ok(None);
    };
    match data.get(field) {
        Some(Value::String(value)) => {
            let value = value.trim();
            ensure!(
                !value.is_empty(),
                "tool output field '{field}' must not be empty when provided"
            );
            Ok(Some(value.to_string()))
        }
        Some(Value::Null) | None => Ok(None),
        Some(_) => Err(anyhow!(
            "tool output field '{field}' must be string or null"
        )),
    }
}

fn shell_command_succeeded(output: &ToolResult) -> bool {
    if !output.ok {
        return false;
    }

    let Some(data) = output.data.as_ref() else {
        return true;
    };

    if let Some(status) = data.get("status").and_then(Value::as_i64) {
        if status != 0 {
            return false;
        }
    }

    if let Some(success) = data.get("success").and_then(Value::as_bool) {
        if !success {
            return false;
        }
    }

    !data.get("error").is_some()
}

fn is_failed_validation_attempt(record: &ToolExecutionRecord) -> bool {
    record.tool_name == "shell__exec"
        && record.status == ToolExecutionStatus::Executed
        && record
            .effects
            .command
            .as_deref()
            .is_some_and(is_validation_command_text)
        && !shell_command_succeeded(&record.output)
}

fn is_zero(value: &usize) -> bool {
    *value == 0
}

fn normalize_session_title(raw: &str) -> Result<String> {
    let first_line = raw
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .unwrap_or_default();
    let collapsed = first_line.split_whitespace().collect::<Vec<_>>().join(" ");
    let stripped = strip_wrapping_title_quotes(collapsed.trim());
    if stripped.is_empty() {
        bail!("session title generation returned empty content");
    }

    let normalized = truncate_chars(stripped, 80).trim().to_string();
    if normalized.is_empty() {
        bail!("session title generation returned empty normalized content");
    }

    Ok(normalized)
}

fn strip_wrapping_title_quotes(mut text: &str) -> &str {
    loop {
        let trimmed = text.trim();
        let next = if let Some(inner) = trimmed
            .strip_prefix('"')
            .and_then(|inner| inner.strip_suffix('"'))
        {
            inner
        } else if let Some(inner) = trimmed
            .strip_prefix('\'')
            .and_then(|inner| inner.strip_suffix('\''))
        {
            inner
        } else if let Some(inner) = trimmed
            .strip_prefix('`')
            .and_then(|inner| inner.strip_suffix('`'))
        {
            inner
        } else if let Some(inner) = trimmed
            .strip_prefix('“')
            .and_then(|inner| inner.strip_suffix('”'))
        {
            inner
        } else if let Some(inner) = trimmed
            .strip_prefix('‘')
            .and_then(|inner| inner.strip_suffix('’'))
        {
            inner
        } else {
            return trimmed;
        };
        text = next;
    }
}

fn truncate_chars(text: &str, max_chars: usize) -> String {
    text.chars().take(max_chars).collect()
}

fn is_validation_command_text(command: &str) -> bool {
    let command = command.trim();
    command == "cargo check"
        || command.starts_with("cargo check ")
        || command == "cargo test"
        || command.starts_with("cargo test ")
        || command == "cargo clippy"
        || command.starts_with("cargo clippy ")
        || command == "cargo fmt --check"
        || command.starts_with("cargo fmt --check ")
        || command == "npm test"
        || command.starts_with("npm test ")
        || command == "pnpm test"
        || command.starts_with("pnpm test ")
        || command == "yarn test"
        || command.starts_with("yarn test ")
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum TodoStatus {
    #[serde(rename = "pending")]
    Pending,
    #[serde(rename = "in_progress")]
    InProgress,
    #[serde(rename = "blocked")]
    Blocked,
    #[serde(rename = "completed")]
    Completed,
    #[serde(rename = "cancelled")]
    Cancelled,
}

impl TodoStatus {
    fn is_unfinished(&self) -> bool {
        matches!(self, Self::Pending | Self::InProgress)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TodoItem {
    pub id: String,
    pub content: String,
    pub status: TodoStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AutoContinueState {
    pub enabled: bool,
    pub max_continuations: usize,
}

impl AutoContinueState {
    const DEFAULT_MAX_CONTINUATIONS: usize = 3;
    const ABSOLUTE_MAX_CONTINUATIONS: usize = 16;
}

impl Default for AutoContinueState {
    fn default() -> Self {
        Self {
            enabled: false,
            max_continuations: Self::DEFAULT_MAX_CONTINUATIONS,
        }
    }
}

pub(crate) fn protocol_frame_item_from_history_item(
    item: &HistoryItem,
) -> crate::protocol_frames::ProtocolFrameItem {
    match item {
        HistoryItem::ContextSummary { text } => {
            crate::protocol_frames::ProtocolFrameItem::ContextSummary { text: text.clone() }
        }
        HistoryItem::UserMessage { content } => {
            crate::protocol_frames::ProtocolFrameItem::UserMessage {
                content: content.clone(),
            }
        }
        HistoryItem::InternalContinuation { text } => {
            crate::protocol_frames::ProtocolFrameItem::InternalContinuation { text: text.clone() }
        }
        HistoryItem::AssistantText { text } => {
            crate::protocol_frames::ProtocolFrameItem::AssistantText { text: text.clone() }
        }
        HistoryItem::AssistantToolCalls { text, calls } => {
            crate::protocol_frames::ProtocolFrameItem::AssistantToolCalls {
                text: text.clone(),
                calls: calls.clone(),
            }
        }
        HistoryItem::ToolOutput {
            call_id,
            output_json,
        } => crate::protocol_frames::ProtocolFrameItem::ToolOutput {
            call_id: call_id.clone(),
            output_json: output_json.clone(),
        },
    }
}

fn runtime_frame_from_protocol_frame(
    frame: &crate::protocol_frames::ProtocolFrame,
    ordinal: u32,
) -> RuntimeFrame {
    let (kind, stable_key, summary) = match &frame.item {
        crate::protocol_frames::ProtocolFrameItem::ContextSummary { text } => (
            RuntimeFrameKind::Summary,
            format!("context-summary:{}", frame.history_index),
            Some(text.clone()),
        ),
        crate::protocol_frames::ProtocolFrameItem::UserMessage { content } => (
            RuntimeFrameKind::User,
            format!("user:{}", frame.history_index),
            Some(content.prompt_plan_text()),
        ),
        crate::protocol_frames::ProtocolFrameItem::InternalContinuation { text } => (
            RuntimeFrameKind::Metadata,
            format!("internal-continuation:{}", frame.history_index),
            Some(text.clone()),
        ),
        crate::protocol_frames::ProtocolFrameItem::AssistantText { text } => (
            RuntimeFrameKind::Assistant,
            format!("assistant:{}", frame.history_index),
            Some(text.clone()),
        ),
        crate::protocol_frames::ProtocolFrameItem::AssistantToolCalls { text, calls } => (
            RuntimeFrameKind::ToolCall,
            format!(
                "assistant-tool-calls:{}:{}",
                frame.history_index,
                calls
                    .iter()
                    .map(|call| call.call_id.as_str())
                    .collect::<Vec<_>>()
                    .join("|")
            ),
            Some(text.clone().unwrap_or_else(|| {
                calls
                    .iter()
                    .map(|call| format!("{}({})", call.name, call.call_id))
                    .collect::<Vec<_>>()
                    .join(", ")
            })),
        ),
        crate::protocol_frames::ProtocolFrameItem::ToolOutput {
            call_id,
            output_json,
        } => (
            RuntimeFrameKind::ToolOutput,
            format!("tool-output:{}", call_id),
            Some(output_json.clone()),
        ),
    };
    let mut runtime_frame = RuntimeFrame::new(
        kind,
        FrameVisibility::Active,
        RuntimeFrameProvenance::new(RuntimeSource::Derived),
        RuntimeFrameIdSeed {
            frame_kind: kind,
            source: RuntimeSource::Derived,
            ordinal,
            stable_key: &stable_key,
            source_span: None,
        },
    );
    runtime_frame.summary = summary;
    runtime_frame.protocol = Some(frame.item.clone());
    runtime_frame
}

/// The protocol-derived portion of a snapshot is an ordered, one-to-one view of
/// protocol history.  Validate it before any snapshot becomes live: accepting a
/// shifted or partial prefix makes compaction retire unrelated transcript data.
fn validate_runtime_snapshot_correspondence(
    history: &[HistoryItem],
    snapshot: &RuntimeSnapshot,
) -> Result<()> {
    snapshot.validate_references()?;
    let frames = snapshot.active_protocol_frames();
    ensure!(
        frames.len() == history.len(),
        "runtime snapshot protocol projection length {} does not match history length {}",
        frames.len(),
        history.len()
    );
    for (index, (frame, item)) in frames.iter().zip(history).enumerate() {
        ensure!(
            frame.item == protocol_frame_item_from_history_item(item),
            "runtime snapshot protocol payload at ordinal {index} does not exactly match history"
        );
        ensure!(
            frame.runtime_frame_id.is_some(),
            "runtime snapshot protocol frame at ordinal {index} has no frame id"
        );
    }
    Ok(())
}

fn validate_protocol_frame_correspondence(
    protocol_frames: &[crate::protocol_frames::ProtocolFrame],
    snapshot: &RuntimeSnapshot,
) -> Result<()> {
    let projected = snapshot.active_protocol_frames();
    ensure!(
        protocol_frames.len() == projected.len(),
        "protocol frame cache length does not match runtime snapshot projection"
    );
    for (index, (cached, runtime)) in protocol_frames.iter().zip(&projected).enumerate() {
        ensure!(
            cached.runtime_frame_id.is_none()
                || cached.runtime_frame_id == runtime.runtime_frame_id,
            "protocol frame cache at ordinal {index} has a runtime id that does not match the runtime snapshot"
        );
        ensure!(
            cached.item == runtime.item,
            "protocol frame cache at ordinal {index} does not exactly match runtime snapshot"
        );
    }
    Ok(())
}

fn remap_runtime_snapshot_frame_ids(
    snapshot: &mut RuntimeSnapshot,
    remap: &HashMap<crate::runtime_context::RuntimeFrameId, crate::runtime_context::RuntimeFrameId>,
) {
    for frame in &mut snapshot.frames {
        if let Some(id) = remap.get(&frame.id) {
            frame.id = *id;
        }
    }
    for id in snapshot
        .compaction
        .protected_frame_ids
        .iter_mut()
        .chain(snapshot.compaction.explicit_protected_frame_ids.iter_mut())
        .chain(snapshot.compaction.turn_protected_frame_ids.iter_mut())
        .chain(snapshot.compaction.compacted_frame_ids.iter_mut())
        .chain(
            snapshot
                .prompt_contributors
                .iter_mut()
                .flat_map(|contributor| {
                    contributor
                        .frame_ids
                        .iter_mut()
                        .chain(contributor.source_frame_ids.iter_mut())
                }),
        )
    {
        if let Some(mapped) = remap.get(id) {
            *id = *mapped;
        }
    }
}

fn merge_runtime_provenance(
    provider: &mut RuntimeFrameProvenance,
    durable: &RuntimeFrameProvenance,
) {
    if provider.source_span.is_none() {
        provider.source_span = durable.source_span;
    }
    if provider.source_id.is_none() {
        provider.source_id = durable.source_id.clone();
    }
    if provider.label.is_none() {
        provider.label = durable.label.clone();
    }
}

fn merge_runtime_source_spans(
    spans: impl IntoIterator<Item = crate::runtime_context::SourceSpan>,
) -> Vec<crate::runtime_context::SourceSpan> {
    let mut spans = spans.into_iter().collect::<Vec<_>>();
    spans.sort();
    let mut merged: Vec<crate::runtime_context::SourceSpan> = Vec::new();
    for span in spans {
        if let Some(last) = merged.last_mut()
            && span.start_sequence <= last.end_sequence.saturating_add(1)
        {
            last.end_sequence = last.end_sequence.max(span.end_sequence);
        } else {
            merged.push(span);
        }
    }
    merged
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TurnRuntimeState {
    turn_id: u64,
    current_turn_start_index: Option<usize>,
    policy: WorkflowTurnState,
    workflow: WorkflowState,
    counters: TurnCounters,
    last_continuation_todos: Option<Vec<TodoItem>>,
    frozen_evidence: Option<FrozenTurnEvidence>,
    automatic_checkpoint: AutomaticCheckpointSchedulerState,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FrozenTurnEvidence {
    message: Option<String>,
    selected_ids: Vec<String>,
}

impl TurnRuntimeState {
    fn new(turn_id: u64, policy: WorkflowTurnState) -> Self {
        Self {
            turn_id,
            current_turn_start_index: None,
            policy,
            workflow: WorkflowState::default(),
            counters: TurnCounters::default(),
            last_continuation_todos: None,
            frozen_evidence: None,
            automatic_checkpoint: AutomaticCheckpointSchedulerState::default(),
        }
    }
}

/// Ephemeral automatic-checkpoint state. It is intentionally absent from all
/// transcript and snapshot projection types: restored turns must not inherit a
/// scheduler decision from a previous process.
#[derive(Debug, Clone, PartialEq, Eq)]
struct AutomaticCheckpointSchedulerState {
    armed: bool,
    commits: u8,
    current_boundary_generation: Option<u64>,
    next_boundary_generation: u64,
    last_attempted_boundary: Option<u64>,
    last_consumed_boundary: Option<u64>,
    suppressed: bool,
}

impl Default for AutomaticCheckpointSchedulerState {
    fn default() -> Self {
        Self {
            armed: true,
            commits: 0,
            current_boundary_generation: None,
            next_boundary_generation: 0,
            last_attempted_boundary: None,
            last_consumed_boundary: None,
            suppressed: false,
        }
    }
}

impl AutomaticCheckpointSchedulerState {
    fn begin_complete_boundary(&mut self) -> u64 {
        let generation = self
            .next_boundary_generation
            .checked_add(1)
            .expect("automatic checkpoint boundary generation overflow");
        self.next_boundary_generation = generation;
        self.current_boundary_generation = Some(generation);
        generation
    }

    /// A boundary is valid for exactly the request preparation immediately
    /// following its completed tool batch.  Consume before any fallible work so
    /// recovery and later iterations cannot inherit it.
    fn consume_complete_boundary(&mut self) -> Option<u64> {
        let boundary = self.current_boundary_generation.take();
        if boundary.is_some() {
            self.last_consumed_boundary = boundary;
        }
        boundary
    }

    fn view_for_boundary(
        &self,
        boundary: Option<u64>,
    ) -> automatic_checkpoint::AutoCheckpointSchedulerView {
        automatic_checkpoint::AutoCheckpointSchedulerView {
            armed: self.armed,
            automatic_commits: self.commits,
            boundary_available: boundary.is_some(),
            boundary_consumed: false,
            boundary_attempted: boundary == self.last_attempted_boundary,
            suppressed: self.suppressed,
        }
    }

    #[cfg(test)]
    fn view(&self) -> automatic_checkpoint::AutoCheckpointSchedulerView {
        let mut view = self.view_for_boundary(self.current_boundary_generation);
        view.boundary_consumed = self.current_boundary_generation == self.last_consumed_boundary;
        view
    }

    fn mark_attempted(&mut self, boundary: u64) {
        self.last_attempted_boundary = Some(boundary);
        self.armed = false;
    }

    fn mark_committed(&mut self, owner: LogicalCheckpointRequestOwner) {
        if matches!(owner, LogicalCheckpointRequestOwner::Automatic { .. }) {
            self.commits = self.commits.saturating_add(1);
        } else if self.current_boundary_generation.is_some() {
            self.last_consumed_boundary = self.current_boundary_generation;
        }
        self.armed = false;
    }

    fn rearm(&mut self) {
        if !self.suppressed {
            self.armed = true;
        }
    }

    fn suppress(&mut self) {
        self.suppressed = true;
        self.armed = false;
    }

    fn reset_for_turn_end(&mut self) {
        *self = Self::default();
    }
}

impl Default for TurnRuntimeState {
    fn default() -> Self {
        Self::new(0, WorkflowTurnState::default())
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct WorkflowState {
    todos: Vec<TodoItem>,
    auto_continue: AutoContinueState,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct TurnCounters {
    continuations: usize,
    write_effects: usize,
    validation_effects: usize,
    failed_validation_effects: usize,
    child_write_effects: usize,
    child_validation_effects: usize,
    child_failed_validation_effects: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PendingSubagentJob {
    run_id: String,
    child_session_id: String,
    agent_name: String,
    status: String,
    summary: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum FinalizeDecision {
    Finish,
    Continue { remaining_unfinished: usize },
    StopWithError { message: String },
}

#[derive(Debug, Clone, Deserialize)]
struct WorkflowTodosPayload {
    items: Vec<TodoItem>,
}

#[derive(Debug, Clone, Deserialize)]
struct WorkflowAutoContinuePayload {
    enabled: bool,
    #[serde(default)]
    max_continuations: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TurnIntent {
    Lightweight,
    Engineering,
}

impl TurnIntent {
    fn as_str(self) -> &'static str {
        match self {
            Self::Lightweight => "lightweight",
            Self::Engineering => "engineering",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ValidationReminder {
    None,
    Focused,
    Targeted,
}

impl ValidationReminder {
    fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Focused => "focused",
            Self::Targeted => "targeted",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct WorkflowTurnState {
    intent: TurnIntent,
    validation: ValidationReminder,
    directive: ExecutionDirective,
}

impl Default for WorkflowTurnState {
    fn default() -> Self {
        Self {
            intent: TurnIntent::Lightweight,
            validation: ValidationReminder::None,
            directive: ExecutionDirective::None,
        }
    }
}

impl WorkflowTurnState {
    fn from_user_input(user_input: &str) -> Self {
        let intent = classify_turn_intent(user_input);
        let validation = detect_validation_reminder(user_input, intent);
        let directive = detect_execution_directive(user_input);
        Self {
            intent,
            validation,
            directive,
        }
    }

    fn developer_context_message(&self) -> Option<PromptMessage> {
        if self.intent == TurnIntent::Lightweight {
            return None;
        }

        let mut text = ENGINEERING_WORKFLOW_PRELUDE.to_string();
        match self.validation {
            ValidationReminder::None => {}
            ValidationReminder::Focused => {
                text.push_str(
                    "\nIf you make code changes, run focused validation for the files or behavior you touched. If validation is not practical, say so explicitly.",
                );
            }
            ValidationReminder::Targeted => {
                text.push_str(
                    "\nPlan to run the most relevant targeted validation for this task, such as the affected tests, build, or lint command. If you skip validation, say why explicitly.",
                );
            }
        }

        match self.directive {
            ExecutionDirective::None => {}
            ExecutionDirective::ReadOnly => {
                text.push_str(
                    "\nThis turn is read-only. Do not modify files or run non-read-only commands.",
                );
            }
            ExecutionDirective::PlanOnly => {
                text.push_str(
                    "\nThis turn is plan-only. Produce analysis and planning only. Do not modify files or run non-read-only commands.",
                );
            }
            ExecutionDirective::AnalyzeOnly => {
                text.push_str(
                    "\nThis turn is analyze-only. Inspect and explain only. Do not modify files or run non-read-only commands.",
                );
            }
            ExecutionDirective::DoNotEdit => {
                text.push_str(
                    "\nThis turn has an explicit do-not-edit directive. Do not modify files or run non-read-only commands.",
                );
            }
        }

        Some(PromptMessage::developer_with_origin(
            text,
            PromptMessageOrigin::WorkflowTurn,
        ))
    }
}

fn detect_execution_directive(user_input: &str) -> ExecutionDirective {
    let normalized = normalize_for_intent(user_input);

    if contains_any(&normalized, &["read-only", "read only", "readonly", "只读"]) {
        ExecutionDirective::ReadOnly
    } else if contains_any(
        &normalized,
        &[
            "plan-only",
            "plan only",
            "planning only",
            "only plan",
            "just plan",
            "只做计划",
        ],
    ) {
        ExecutionDirective::PlanOnly
    } else if contains_any(
        &normalized,
        &[
            "analyze-only",
            "analyze only",
            "analysis only",
            "only analyze",
            "only analyse",
            "只分析",
        ],
    ) {
        ExecutionDirective::AnalyzeOnly
    } else if contains_any(
        &normalized,
        &[
            "do not edit",
            "don't edit",
            "dont edit",
            "no edits",
            "不要修改",
        ],
    ) {
        ExecutionDirective::DoNotEdit
    } else {
        ExecutionDirective::None
    }
}

fn classify_turn_intent(user_input: &str) -> TurnIntent {
    let normalized = normalize_for_intent(user_input);

    if contains_engineering_signal(&normalized) {
        TurnIntent::Engineering
    } else {
        TurnIntent::Lightweight
    }
}

fn detect_validation_reminder(user_input: &str, intent: TurnIntent) -> ValidationReminder {
    if intent == TurnIntent::Lightweight {
        return ValidationReminder::None;
    }

    let normalized = normalize_for_intent(user_input);
    if contains_any(
        &normalized,
        &[
            "cargo test",
            "cargo check",
            "cargo clippy",
            "test ",
            "tests ",
            "build ",
            "compile",
            "lint",
        ],
    ) {
        ValidationReminder::Targeted
    } else if contains_any(
        &normalized,
        &[
            "fix",
            "implement",
            "add",
            "update",
            "modify",
            "refactor",
            "rename",
            "remove",
            "create",
            "write",
            "edit",
            "patch",
            "bug",
            "failing",
            "regression",
        ],
    ) {
        ValidationReminder::Focused
    } else {
        ValidationReminder::None
    }
}

fn contains_engineering_signal(normalized: &str) -> bool {
    contains_any(
        normalized,
        &[
            "fix",
            "implement",
            "add",
            "update",
            "modify",
            "refactor",
            "rename",
            "remove",
            "create",
            "write",
            "edit",
            "patch",
            "debug",
            "investigate",
            "trace",
            "root cause",
            "complex analysis",
            "full analysis",
            "workflow",
            "codebase",
            "repository",
            "repo",
            "project",
            "module",
            "crate",
            "src/",
            "cargo ",
            "test ",
            "tests ",
            "build ",
            "compile",
            "lint",
            "multi-step",
            "step by step",
            "plan",
            "pipeline",
            "across",
            "multiple files",
            "复杂任务",
            "复杂分析",
            "工程",
            "实现",
            "修改",
            "修复",
            "重构",
            "调试",
            "排查",
            "计划",
            "当前项目",
        ],
    )
}

fn normalize_for_intent(user_input: &str) -> String {
    user_input.to_ascii_lowercase()
}

fn contains_any(text: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| text.contains(needle))
}

fn capped_tool_call_limit(
    parent_limit: Option<usize>,
    requested_limit: Option<usize>,
) -> Option<usize> {
    match (parent_limit, requested_limit) {
        (Some(parent), Some(requested)) => Some(parent.min(requested)),
        (Some(parent), None) => Some(parent),
        (None, Some(requested)) => Some(requested),
        (None, None) => None,
    }
}

fn is_workflow_control_tool(tool_name: &str) -> bool {
    matches!(
        tool_name,
        "workflow__todos"
            | "workflow__auto_continue"
            | "context__checkpoint"
            | "context__return"
            | "agent__reconcile"
    )
}

fn is_cancelled_subagent_record(record: &ToolExecutionRecord) -> bool {
    is_subagent_tool_name(&record.tool_name)
        && record
            .output
            .data
            .as_ref()
            .and_then(|data| data.get("status"))
            .and_then(Value::as_str)
            == Some("cancelled")
}

impl ToolExecutionStatus {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Executed => "executed",
            Self::Rejected => "rejected",
            Self::TimedOut => "timed_out",
        }
    }
}

fn default_compaction_outcome() -> String {
    "succeeded".to_string()
}

impl ToolExecutionRejection {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::InvalidJsonArguments => "invalid_json_arguments",
            Self::DirectiveBlocked => "directive_blocked",
            Self::ToolScopeDenied => "tool_scope_denied",
            Self::PermissionDeniedByPolicy => "permission_denied_by_policy",
            Self::PermissionDeniedByUser => "permission_denied_by_user",
        }
    }
}

impl ToolEffectKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Read => "read",
            Self::Write => "write",
            Self::Command => "command",
            Self::Validation => "validation",
            Self::WorkflowControl => "workflow_control",
            Self::Diagnostic => "diagnostic",
            Self::Unknown => "unknown",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent_event_journal::persist_agent_event;
    use crate::context_tree::{ContextNodeStatus, ContextTreeState};
    use crate::context_view::{
        ContextBlock, ContextBlockId, ContextBlockKind, ContextBlockSource, ContextViewProjection,
        FoldedOutputMetadata,
    };
    use crate::protocol_frames::ProtocolFrameItem;
    use crate::request_builder::{LegacyRequestBuilderInput, build_request_from_legacy};
    use crate::runtime_context::{
        PromptContributorKind, PromptContributorPlaceholder, RuntimeChildSession,
        RuntimeFrameIdSeed, RuntimeFrameKind, RuntimeFrameProvenance, RuntimeSource, SourceSpan,
    };
    use crate::transcript::transcript_projection::{project_context_tree, project_context_view};
    use crate::transcript::{
        ROOT_CONTEXT_BRANCH_ID, TranscriptEvent, TranscriptRecord, TranscriptRecorder,
        read_records, restore_runtime_snapshot, restore_session_history,
    };
    use async_openai::config::OpenAIConfig;
    use async_trait::async_trait;
    use serde_json::json;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;
    use tokio::task::JoinHandle;
    use tokio::time::{Duration, sleep};

    #[test]
    fn logical_checkpoint_disable_clears_pending_before_take() {
        let control = LogicalCheckpointControl::disabled_for_test();
        control.set_enabled(true);
        assert_eq!(control.request(), LogicalCheckpointRequestOutcome::Queued);
        control.set_enabled(false);
        assert!(control.take_pending().is_none());
        assert_eq!(control.request(), LogicalCheckpointRequestOutcome::Disabled);
    }

    #[test]
    fn logical_checkpoint_request_and_disable_are_serialized() {
        let control = LogicalCheckpointControl::disabled_for_test();
        control.set_enabled(true);
        let requester = control.clone();
        let disabler = control.clone();
        let barrier = Arc::new(std::sync::Barrier::new(3));
        let request_barrier = Arc::clone(&barrier);
        let disable_barrier = Arc::clone(&barrier);
        let request = std::thread::spawn(move || {
            request_barrier.wait();
            requester.request()
        });
        let disable = std::thread::spawn(move || {
            disable_barrier.wait();
            disabler.set_enabled(false);
        });
        barrier.wait();
        let _ = request.join().expect("request thread");
        disable.join().expect("disable thread");
        assert!(control.take_pending().is_none());
        assert_eq!(control.request(), LogicalCheckpointRequestOutcome::Disabled);
    }

    #[test]
    fn logical_checkpoint_run_guard_clears_pending_and_in_flight_requests_on_drop() {
        let control = LogicalCheckpointControl::disabled_for_test();
        control.set_enabled(true);

        let pending_run = control.begin_run();
        assert_eq!(control.request(), LogicalCheckpointRequestOutcome::Queued);
        drop(pending_run);
        assert_eq!(control.request(), LogicalCheckpointRequestOutcome::Queued);
        control.clear();

        let in_flight_run = control.begin_run();
        assert_eq!(control.request(), LogicalCheckpointRequestOutcome::Queued);
        assert!(control.take_pending().is_some());
        drop(in_flight_run);
        assert_eq!(control.request(), LogicalCheckpointRequestOutcome::Queued);
    }

    #[test]
    fn logical_checkpoint_lease_does_not_clear_a_later_run_request() {
        let control = LogicalCheckpointControl::disabled_for_test();
        control.set_enabled(true);
        let first_run = control.begin_run();
        assert_eq!(control.request(), LogicalCheckpointRequestOutcome::Queued);
        let first_lease = control.take_pending().expect("first lease");
        control.clear_lease(first_lease);
        drop(first_run);

        let second_run = control.begin_run();
        assert_eq!(control.request(), LogicalCheckpointRequestOutcome::Queued);
        control.clear_lease(first_lease);
        assert_eq!(
            control.request(),
            LogicalCheckpointRequestOutcome::AlreadyQueued
        );
        drop(second_run);
    }

    #[test]
    fn automatic_checkpoint_request_requires_active_run_and_never_displaces_manual() {
        let control = LogicalCheckpointControl::disabled_for_test();
        control.set_config(LogicalCheckpointConfig {
            enabled: true,
            automatic: true,
            ..Default::default()
        });
        assert_eq!(
            control.request_automatic(1),
            LogicalCheckpointRequestOutcome::Disabled
        );
        let run = control.begin_run();
        assert_eq!(
            control.request_automatic(1),
            LogicalCheckpointRequestOutcome::Queued
        );
        assert_eq!(control.request(), LogicalCheckpointRequestOutcome::Queued);
        let lease = control.take_pending().expect("manual lease");
        assert_eq!(lease.ownership, LogicalCheckpointRequestOwner::Manual);
        assert_eq!(
            control.request_automatic(2),
            LogicalCheckpointRequestOutcome::AlreadyQueued
        );
        control.clear_lease(lease);
        drop(run);
    }

    #[test]
    fn manual_request_replaces_a_queued_automatic_request_without_counting_as_automatic() {
        let control = LogicalCheckpointControl::disabled_for_test();
        control.set_config(LogicalCheckpointConfig {
            enabled: true,
            automatic: true,
            ..Default::default()
        });
        let run = control.begin_run();
        assert_eq!(
            control.request_automatic(7),
            LogicalCheckpointRequestOutcome::Queued
        );
        assert_eq!(control.request(), LogicalCheckpointRequestOutcome::Queued);
        let lease = control.take_pending().expect("manual replacement lease");
        assert_eq!(lease.ownership, LogicalCheckpointRequestOwner::Manual);
        control.clear_lease(lease);

        let mut scheduler = AutomaticCheckpointSchedulerState::default();
        scheduler.begin_complete_boundary();
        scheduler.mark_committed(LogicalCheckpointRequestOwner::Manual);
        assert_eq!(scheduler.commits, 0);
        scheduler.commits = 1;
        let policy = automatic_checkpoint::AutoCheckpointPolicy {
            enabled: true,
            trigger_reserve_percent: 50,
            max_automatic_per_turn: 1,
        };
        assert_eq!(
            policy.decide(None, true, scheduler.view()),
            automatic_checkpoint::AutoCheckpointDecision::Suppress
        );
        assert_eq!(
            control.request(),
            LogicalCheckpointRequestOutcome::Queued,
            "a manual request remains available after the automatic per-turn limit"
        );
        drop(run);
    }

    #[test]
    fn automatic_checkpoint_lease_identity_prevents_same_run_aba_clear() {
        let control = LogicalCheckpointControl::disabled_for_test();
        control.set_config(LogicalCheckpointConfig {
            enabled: true,
            automatic: true,
            ..Default::default()
        });
        let run = control.begin_run();
        assert_eq!(
            control.request_automatic(1),
            LogicalCheckpointRequestOutcome::Queued
        );
        let first = control.take_pending().expect("first lease");
        control.clear_lease(first);
        assert_eq!(
            control.request_automatic(2),
            LogicalCheckpointRequestOutcome::Queued
        );
        let second = control.take_pending().expect("second lease");
        assert_ne!(first.request_id, second.request_id);
        control.clear_lease(first);
        assert_eq!(
            control.request(),
            LogicalCheckpointRequestOutcome::AlreadyQueued
        );
        control.clear_lease(second);
        drop(run);
    }

    #[test]
    fn automatic_scheduler_state_is_ephemeral_and_resets_for_the_next_turn() {
        let mut state = AutomaticCheckpointSchedulerState::default();
        assert!(state.armed);
        assert!(!state.view().boundary_available);
        assert_eq!(state.begin_complete_boundary(), 1);
        state.mark_attempted(1);
        assert!(state.view().boundary_attempted);
        state.rearm();
        state.mark_committed(LogicalCheckpointRequestOwner::Automatic { boundary_id: 1 });
        state.suppress();
        state.reset_for_turn_end();
        assert_eq!(state, AutomaticCheckpointSchedulerState::default());
    }

    #[test]
    fn automatic_scheduler_new_boundary_allows_trigger_after_disarmed_boundary() {
        let mut state = AutomaticCheckpointSchedulerState::default();
        state.begin_complete_boundary();
        state.mark_attempted(1);
        state.rearm();
        assert!(state.view().boundary_attempted);
        assert_eq!(state.begin_complete_boundary(), 2);
        assert!(state.view().boundary_available);
        assert!(!state.view().boundary_attempted);
        assert!(!state.view().boundary_consumed);
    }

    #[test]
    fn committed_checkpoint_consumes_and_disarms_but_only_automatic_counts() {
        let mut state = AutomaticCheckpointSchedulerState::default();
        state.begin_complete_boundary();
        state.mark_committed(LogicalCheckpointRequestOwner::Manual);
        assert!(!state.armed);
        assert!(state.view().boundary_consumed);
        assert_eq!(state.commits, 0);

        state.begin_complete_boundary();
        state.mark_committed(LogicalCheckpointRequestOwner::Automatic { boundary_id: 2 });
        assert_eq!(state.commits, 1);
    }

    #[test]
    fn logical_checkpoint_config_preserves_nondefault_automatic_policy_for_children() {
        let mut parent = test_agent();
        let config = LogicalCheckpointConfig {
            enabled: true,
            automatic: true,
            trigger_reserve_percent: 73,
            max_automatic_per_turn: 3,
        };
        parent.set_logical_checkpoint_config(config);
        assert_eq!(
            parent.automatic_checkpoint_policy,
            automatic_checkpoint::AutoCheckpointPolicy::from_config(config)
        );

        let child = AgentFactory::create_child(&parent, &AgentTemplate::explorer());
        assert_eq!(
            child.automatic_checkpoint_policy,
            parent.automatic_checkpoint_policy
        );
    }

    fn checkpoint_test_agent() -> (Agent<OpenAIConfig>, Vec<PromptMessage>) {
        let mut agent = test_agent();
        agent.set_model_catalog(HashMap::from([(
            "m1".into(),
            ModelRequestMetadata {
                context_window: Some(32_000),
                max_output_tokens: Some(2_000),
                supports_tools: true,
                ..Default::default()
            },
        )]));
        agent.set_logical_checkpoint_config(LogicalCheckpointConfig {
            enabled: true,
            ..Default::default()
        });
        let prelude = agent.prepare_turn_prelude("continue the active turn");
        agent.history = vec![HistoryItem::user("current request")];
        agent.turn.current_turn_start_index = Some(0);
        agent.runtime_snapshot.current_turn_id = Some(agent.turn.turn_id);
        agent.runtime_snapshot.current_segment_id = Some(1);
        agent.runtime_snapshot.leaf_sequence = Some(9);
        agent.runtime_snapshot.latest_model = Some(agent.model.clone());
        (agent, prelude)
    }

    fn prepared_checkpoint_for(
        agent: &Agent<OpenAIConfig>,
    ) -> crate::transcript::PreparedLogicalCheckpoint {
        prepared_checkpoint_for_lineage(agent, "checkpoint-test", None)
    }

    fn prepared_checkpoint_for_lineage(
        agent: &Agent<OpenAIConfig>,
        checkpoint_id: &str,
        previous_checkpoint_id: Option<&str>,
    ) -> crate::transcript::PreparedLogicalCheckpoint {
        let previous_segment_id = agent
            .runtime_snapshot
            .current_segment_id
            .expect("checkpoint fixture has a live segment");
        let boundary_sequence = agent
            .runtime_snapshot
            .leaf_sequence
            .expect("checkpoint fixture has a journal frontier");
        let segment_id = previous_segment_id + 1;
        let leaf = boundary_sequence + 1;
        let event = LogicalCheckpointEventV1 {
            schema_version: 1,
            checkpoint_id: checkpoint_id.into(),
            turn_id: agent.turn.turn_id,
            previous_segment_id,
            segment_id,
            previous_checkpoint_id: previous_checkpoint_id.map(str::to_string),
            boundary_sequence,
            context_scope_revision: agent.runtime_snapshot.context_scope_revision,
            covered_source_spans: Vec::new(),
            retained_items: Vec::new(),
        };
        let summary = crate::transcript::render_checkpoint_v1(&event).expect("summary renders");
        let continuation = crate::transcript::render_checkpoint_continuation_v1(&event);
        let mut frames = vec![
            crate::protocol_frames::ProtocolFrame::derived(ProtocolFrameItem::ContextSummary {
                text: summary,
            }),
            crate::protocol_frames::ProtocolFrame::derived(
                ProtocolFrameItem::InternalContinuation { text: continuation },
            ),
        ];
        for (index, frame) in frames.iter_mut().enumerate() {
            frame.history_index = index;
        }
        for (frame, source_id) in frames.iter_mut().zip([
            format!("{checkpoint_id}:summary"),
            format!("{checkpoint_id}:continuation"),
        ]) {
            frame.source_provenance = Some(
                RuntimeFrameProvenance::new(RuntimeSource::Transcript)
                    .with_source_id(&source_id)
                    .with_span(SourceSpan::new(leaf, leaf).expect("valid source span")),
            );
        }
        let mut snapshot =
            RuntimeSnapshot::new(agent.runtime_snapshot.active_context.branch_id.clone());
        snapshot.current_turn_id = Some(agent.turn.turn_id);
        snapshot.current_segment_id = Some(segment_id);
        snapshot.leaf_sequence = Some(leaf);
        snapshot.latest_model = Some(agent.model.clone());
        snapshot.frames = frames
            .iter()
            .enumerate()
            .map(|(index, frame)| {
                let mut runtime = runtime_frame_from_protocol_frame(frame, index as u32);
                runtime.provenance = frame
                    .source_provenance
                    .clone()
                    .expect("checkpoint suffix has provenance");
                runtime
            })
            .collect();
        for (frame, runtime) in frames.iter_mut().zip(&snapshot.frames) {
            frame.runtime_frame_id = Some(runtime.id);
        }
        crate::transcript::PreparedLogicalCheckpoint {
            expected_journal_frontier: boundary_sequence,
            expected_branch_id: agent.runtime_snapshot.active_context.branch_id.clone(),
            event,
            projected_snapshot: snapshot,
            projected_protocol_frames: frames,
            projected_workflow: Some(crate::transcript::CheckpointWorkflowProjection {
                todos: agent.turn.workflow.todos.clone(),
                auto_continue: agent.turn.workflow.auto_continue.clone(),
            }),
        }
    }

    #[tokio::test]
    async fn logical_checkpoint_missing_provider_rejects_and_clears_its_request() {
        let (mut agent, prelude) = checkpoint_test_agent();
        let run = agent.logical_checkpoint_control.begin_run();
        assert_eq!(
            agent.request_logical_checkpoint(),
            LogicalCheckpointRequestOutcome::Queued
        );

        let error = logical_checkpoint::commit_pending_at_batch_boundary(
            &mut agent,
            ApiProtocol::Responses,
            &prelude,
            0,
            &mut |_| std::future::ready(Ok(())),
        )
        .await
        .expect_err("missing provider must reject the pending checkpoint");

        assert!(error.to_string().contains("without a candidate provider"));
        assert_eq!(
            agent.request_logical_checkpoint(),
            LogicalCheckpointRequestOutcome::Queued
        );
        drop(run);
    }

    #[tokio::test]
    async fn logical_checkpoint_callback_failure_preserves_live_envelope_and_clears_request() {
        let (mut agent, prelude) = checkpoint_test_agent();
        agent.turn.automatic_checkpoint.begin_complete_boundary();
        let candidate = prepared_checkpoint_for(&agent);
        agent.set_logical_checkpoint_candidate_provider(Arc::new(move || Ok(candidate.clone())));
        let history_before = agent.history.clone();
        let frames_before = agent.protocol_frames.clone();
        let snapshot_before = agent.runtime_snapshot.clone();
        let workflow_before = agent.turn.workflow.clone();
        let start_before = agent.turn.current_turn_start_index;
        let run = agent.logical_checkpoint_control.begin_run();
        assert_eq!(
            agent.request_logical_checkpoint(),
            LogicalCheckpointRequestOutcome::Queued
        );

        let error = logical_checkpoint::commit_pending_at_batch_boundary(
            &mut agent,
            ApiProtocol::Responses,
            &prelude,
            0,
            &mut |_| std::future::ready(Err(anyhow!("durable callback failed"))),
        )
        .await
        .expect_err("durable acknowledgement failure must reject before installation");

        assert!(error.to_string().contains("durable callback failed"));
        assert_eq!(agent.history, history_before);
        assert_eq!(agent.protocol_frames, frames_before);
        assert_eq!(agent.runtime_snapshot, snapshot_before);
        assert_eq!(agent.turn.workflow, workflow_before);
        assert_eq!(agent.turn.current_turn_start_index, start_before);
        assert!(agent.turn.automatic_checkpoint.armed);
        assert_eq!(agent.turn.automatic_checkpoint.commits, 0);
        assert_eq!(
            agent.request_logical_checkpoint(),
            LogicalCheckpointRequestOutcome::Queued
        );
        drop(run);
    }

    #[tokio::test]
    async fn automatic_checkpoint_ack_failure_preserves_envelope_and_releases_lease() {
        let (mut agent, prelude) = checkpoint_test_agent();
        agent.set_logical_checkpoint_config(LogicalCheckpointConfig {
            enabled: true,
            automatic: true,
            ..Default::default()
        });
        let boundary = agent.turn.automatic_checkpoint.begin_complete_boundary();
        agent.turn.automatic_checkpoint.mark_attempted(boundary);
        let scheduler_before = agent.turn.automatic_checkpoint.clone();
        let candidate = prepared_checkpoint_for(&agent);
        agent.set_logical_checkpoint_candidate_provider(Arc::new(move || Ok(candidate.clone())));
        let history_before = agent.history.clone();
        let frames_before = agent.protocol_frames.clone();
        let snapshot_before = agent.runtime_snapshot.clone();
        let workflow_before = agent.turn.workflow.clone();
        let run = agent.logical_checkpoint_control.begin_run();
        assert_eq!(
            agent.logical_checkpoint_control.request_automatic(boundary),
            LogicalCheckpointRequestOutcome::Queued
        );

        logical_checkpoint::commit_pending_at_boundary_with_automatic_token(
            &mut agent,
            ApiProtocol::Responses,
            &prelude,
            0,
            Some(boundary),
            &mut |_| std::future::ready(Err(anyhow!("durable automatic acknowledgement failed"))),
        )
        .await
        .expect_err("failed automatic acknowledgement must not install its candidate");

        assert_eq!(agent.history, history_before);
        assert_eq!(agent.protocol_frames, frames_before);
        assert_eq!(agent.runtime_snapshot, snapshot_before);
        assert_eq!(agent.turn.workflow, workflow_before);
        assert_eq!(agent.turn.automatic_checkpoint, scheduler_before);
        assert_eq!(
            agent
                .logical_checkpoint_control
                .request_automatic(boundary + 1),
            LogicalCheckpointRequestOutcome::Queued,
            "the failed lease is cleared rather than stranding the scheduler"
        );
        drop(run);
    }

    #[tokio::test]
    async fn automatic_checkpoint_owner_mismatch_rejects_without_consuming_manual_budget_or_lease()
    {
        let (mut agent, prelude) = checkpoint_test_agent();
        agent.set_logical_checkpoint_config(LogicalCheckpointConfig {
            enabled: true,
            automatic: true,
            max_automatic_per_turn: 1,
            ..Default::default()
        });
        let candidate = prepared_checkpoint_for(&agent);
        agent.set_logical_checkpoint_candidate_provider(Arc::new(move || Ok(candidate.clone())));
        let run = agent.logical_checkpoint_control.begin_run();
        assert_eq!(
            agent.logical_checkpoint_control.request_automatic(7),
            LogicalCheckpointRequestOutcome::Queued
        );

        let mut logical_events = 0;
        let error = logical_checkpoint::commit_pending_at_boundary_with_automatic_token(
            &mut agent,
            ApiProtocol::Responses,
            &prelude,
            0,
            Some(8),
            &mut |event| {
                logical_events +=
                    usize::from(matches!(event, AgentEvent::LogicalCheckpoint { .. }));
                std::future::ready(Ok(()))
            },
        )
        .await
        .expect_err("a stale automatic boundary must not commit");

        assert!(error.to_string().contains("does not match"));
        assert_eq!(
            logical_events, 0,
            "a stale automatic owner is never persisted"
        );
        assert_eq!(agent.turn.automatic_checkpoint.commits, 0);
        assert_eq!(
            agent.logical_checkpoint_control.request_automatic(9),
            LogicalCheckpointRequestOutcome::Queued,
            "the mismatched lease was cleaned up"
        );
        drop(run);
    }

    #[tokio::test]
    async fn logical_checkpoint_success_installs_exact_prepared_envelope_after_acknowledgement() {
        let (mut agent, prelude) = checkpoint_test_agent();
        agent.turn.automatic_checkpoint.begin_complete_boundary();
        let candidate = prepared_checkpoint_for(&agent);
        let expected_history =
            crate::protocol_frames::history_items_from_frames(&candidate.projected_protocol_frames);
        let expected_snapshot = candidate.projected_snapshot.clone();
        let expected_frames = candidate.projected_protocol_frames.clone();
        agent.set_logical_checkpoint_candidate_provider(Arc::new(move || Ok(candidate.clone())));
        let frozen = FrozenTurnEvidence {
            message: Some("frozen evidence".into()),
            selected_ids: vec!["e-1".into()],
        };
        agent.turn.frozen_evidence = Some(frozen.clone());
        let run = agent.logical_checkpoint_control.begin_run();
        assert_eq!(
            agent.request_logical_checkpoint(),
            LogicalCheckpointRequestOutcome::Queued
        );
        let mut acknowledged = false;

        let protected_start = logical_checkpoint::commit_pending_at_batch_boundary(
            &mut agent,
            ApiProtocol::Responses,
            &prelude,
            0,
            &mut |event| {
                assert!(matches!(event, AgentEvent::LogicalCheckpoint { .. }));
                acknowledged = true;
                std::future::ready(Ok(()))
            },
        )
        .await
        .expect("checkpoint commits")
        .expect("prepared successor supplies protected start");

        assert!(acknowledged);
        assert_eq!(protected_start.protected_start_index, 0);
        assert_eq!(protected_start.owner, LogicalCheckpointRequestOwner::Manual);
        assert!(agent.turn.automatic_checkpoint.view().boundary_consumed);
        assert!(!agent.turn.automatic_checkpoint.armed);
        assert_eq!(agent.turn.automatic_checkpoint.commits, 0);
        assert_eq!(agent.history, expected_history);
        assert_eq!(agent.protocol_frames, expected_frames);
        assert_eq!(agent.runtime_snapshot, expected_snapshot);
        assert_eq!(agent.turn.current_turn_start_index, Some(0));
        assert_eq!(agent.turn.frozen_evidence, Some(frozen));
        assert_eq!(
            agent.runtime_snapshot.current_turn_id,
            Some(agent.turn.turn_id)
        );
        assert_eq!(agent.runtime_snapshot.current_segment_id, Some(2));
        drop(run);
    }

    fn checkpoint_recorder(name: &str) -> Arc<Mutex<TranscriptRecorder>> {
        let directory = std::env::temp_dir().join(format!(
            "letcode-phase3b-{name}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        Arc::new(Mutex::new(
            TranscriptRecorder::create(directory).expect("create checkpoint recorder"),
        ))
    }

    /// Builds a live envelope and its journal predecessor using the same event
    /// persistence boundary as the interactive runners.
    fn transcript_backed_checkpoint_agent(
        recorder: &Arc<Mutex<TranscriptRecorder>>,
    ) -> (Agent<OpenAIConfig>, Vec<PromptMessage>) {
        let (mut agent, prelude) = checkpoint_test_agent();
        let checkpoint_recorder = Arc::clone(recorder);
        let mut recorder_guard = recorder.lock().expect("checkpoint recorder lock");
        recorder_guard
            .record_session_started("m1")
            .expect("session started");
        recorder_guard
            .record_user_message("current request")
            .expect("user message");
        recorder_guard
            .record_turn_started(agent.turn_started_event())
            .expect("turn started");
        recorder_guard
            .record_assistant_message("working")
            .expect("assistant message");
        agent.history.push(HistoryItem::assistant("working"));
        agent.protocol_frames = crate::protocol_frames::history_items_to_frames(&agent.history);
        agent.runtime_snapshot.current_segment_id = Some(0);
        agent.runtime_snapshot.leaf_sequence = Some(
            read_records(recorder_guard.path())
                .expect("read checkpoint predecessor")
                .last()
                .expect("checkpoint predecessor record")
                .sequence,
        );
        agent.runtime_snapshot.latest_model = Some(agent.model.clone());
        drop(recorder_guard);
        agent.set_logical_checkpoint_candidate_provider(Arc::new(move || {
            checkpoint_recorder
                .lock()
                .map_err(|_| anyhow!("checkpoint recorder poisoned"))?
                .prepare_logical_checkpoint()
        }));
        (agent, prelude)
    }

    #[tokio::test]
    async fn phase3b_transcript_checkpoint_acknowledgement_replays_the_installed_successor() {
        let recorder = checkpoint_recorder("acknowledged-successor");
        let (mut agent, prelude) = transcript_backed_checkpoint_agent(&recorder);
        let run = agent.logical_checkpoint_control.begin_run();
        assert_eq!(
            agent.request_logical_checkpoint(),
            LogicalCheckpointRequestOutcome::Queued
        );

        logical_checkpoint::commit_pending_at_batch_boundary(
            &mut agent,
            ApiProtocol::Responses,
            &prelude,
            0,
            &mut |event| {
                let recorder = Arc::clone(&recorder);
                async move {
                    persist_agent_event(
                        &mut recorder.lock().expect("checkpoint recorder lock"),
                        &event,
                    )
                    .map(|_| ())
                }
            },
        )
        .await
        .expect("transcript acknowledgement commits checkpoint");

        let records = {
            let recorder = recorder.lock().expect("checkpoint recorder lock");
            read_records(recorder.path()).expect("read committed transcript")
        };
        assert!(matches!(
            records.last().map(|record| &record.event),
            Some(TranscriptEvent::LogicalCheckpoint(_))
        ));
        let replay = restore_runtime_snapshot(&records).expect("replay checkpoint successor");
        assert_eq!(agent.runtime_snapshot, replay);
        assert_eq!(
            agent.history,
            restore_session_history(&records).expect("replay checkpoint history")
        );
        let replay_protocol =
            crate::transcript::transcript_projection::project_runtime_restore_snapshot(
                recorder
                    .lock()
                    .expect("checkpoint recorder lock")
                    .session_id()
                    .to_string(),
                records.clone(),
                crate::transcript::transcript_projection::SessionContextCursor {
                    branch_id: Some(ROOT_CONTEXT_BRANCH_ID.into()),
                    leaf_sequence: records.last().map(|record| record.sequence),
                },
                &[],
            )
            .expect("replay checkpoint protocol");
        assert_eq!(agent.protocol_frames, replay_protocol.protocol_frames);
        drop(run);
    }

    #[tokio::test]
    async fn phase3b_journal_frontier_race_rejects_without_record_or_installation() {
        let recorder = checkpoint_recorder("frontier-race");
        let (mut agent, prelude) = transcript_backed_checkpoint_agent(&recorder);
        let before_history = agent.history.clone();
        let before_frames = agent.protocol_frames.clone();
        let before_snapshot = agent.runtime_snapshot.clone();
        let run = agent.logical_checkpoint_control.begin_run();
        assert_eq!(
            agent.request_logical_checkpoint(),
            LogicalCheckpointRequestOutcome::Queued
        );

        let error = logical_checkpoint::commit_pending_at_batch_boundary(
            &mut agent,
            ApiProtocol::Responses,
            &prelude,
            0,
            &mut |event| {
                let recorder = Arc::clone(&recorder);
                async move {
                    let mut recorder = recorder.lock().expect("checkpoint recorder lock");
                    recorder
                        .record_assistant_message("racing writer")
                        .expect("race write");
                    persist_agent_event(&mut recorder, &event).map(|_| ())
                }
            },
        )
        .await
        .expect_err("stale recorder frontier rejects acknowledgement");

        assert!(error.to_string().contains("stale"));
        assert_eq!(agent.history, before_history);
        assert_eq!(agent.protocol_frames, before_frames);
        assert_eq!(agent.runtime_snapshot, before_snapshot);
        let records = {
            let recorder = recorder.lock().expect("checkpoint recorder lock");
            read_records(recorder.path()).expect("read transcript")
        };
        assert!(
            !records
                .iter()
                .any(|record| matches!(record.event, TranscriptEvent::LogicalCheckpoint(_)))
        );
        drop(run);
    }

    #[tokio::test]
    async fn phase3b_recorder_cursor_branch_change_after_preparation_rejects_without_installing() {
        let recorder = checkpoint_recorder("cursor-branch-race");
        let (mut agent, prelude) = transcript_backed_checkpoint_agent(&recorder);
        let before_history = agent.history.clone();
        let before_frames = agent.protocol_frames.clone();
        let before_snapshot = agent.runtime_snapshot.clone();
        let run = agent.logical_checkpoint_control.begin_run();
        assert_eq!(
            agent.request_logical_checkpoint(),
            LogicalCheckpointRequestOutcome::Queued
        );

        let error = logical_checkpoint::commit_pending_at_batch_boundary(
            &mut agent,
            ApiProtocol::Responses,
            &prelude,
            0,
            &mut |event| {
                let recorder = Arc::clone(&recorder);
                async move {
                    let mut recorder = recorder.lock().expect("checkpoint recorder lock");
                    // Moving only the recorder cursor must invalidate the prepared
                    // branch envelope even though no journal record is appended.
                    recorder.set_current_context_branch_id(Some("other-branch".into()));
                    persist_agent_event(&mut recorder, &event).map(|_| ())
                }
            },
        )
        .await
        .expect_err("a cursor branch change must reject the prepared candidate");

        assert!(error.to_string().contains("expected branch"));
        assert_eq!(agent.history, before_history);
        assert_eq!(agent.protocol_frames, before_frames);
        assert_eq!(agent.runtime_snapshot, before_snapshot);
        let records = {
            let recorder = recorder.lock().expect("checkpoint recorder lock");
            read_records(recorder.path()).expect("read transcript")
        };
        assert!(
            !records
                .iter()
                .any(|record| matches!(record.event, TranscriptEvent::LogicalCheckpoint(_)))
        );
        drop(run);
    }

    fn assert_request_telemetry_is_terminal_once(events: &[LlmRequestTelemetry]) {
        let mut terminals = HashMap::new();
        for event in events {
            match event.phase {
                LlmRequestTelemetryPhase::Prepared => {
                    assert!(
                        terminals
                            .insert((event.logical_request_id.as_str(), event.attempt), None,)
                            .is_none(),
                        "duplicate prepared event for physical request"
                    );
                }
                LlmRequestTelemetryPhase::Completed
                | LlmRequestTelemetryPhase::Failed
                | LlmRequestTelemetryPhase::Interrupted => {
                    let key = (event.logical_request_id.as_str(), event.attempt);
                    let terminal = terminals
                        .get_mut(&key)
                        .expect("terminal event without prepared event");
                    assert!(
                        terminal.replace(event.phase).is_none(),
                        "duplicate terminal event"
                    );
                }
            }
        }
        assert!(
            terminals.values().all(Option::is_some),
            "prepared event without terminal event"
        );
    }

    fn test_skill_registry() -> Arc<SkillRegistry> {
        Arc::new(
            SkillRegistry::from_entries(vec![crate::skills::SkillEntry {
                name: "rust-audit".into(),
                description: "Inspect Rust code".into(),
                body: "# Private body".into(),
                content:
                    "---\nname: rust-audit\ndescription: Inspect Rust code\n---\n# Private body\n"
                        .into(),
                location: ".letcode/skills".into(),
                path: PathBuf::from("/workspace/.letcode/skills/rust-audit/SKILL.md"),
                base_dir: PathBuf::from("/workspace/.letcode/skills/rust-audit"),
            }])
            .expect("skill registry"),
        )
    }

    fn test_agent() -> Agent<OpenAIConfig> {
        let client = Client::with_config(
            OpenAIConfig::new()
                .with_api_base("https://api.openai.com/v1")
                .with_api_key("test"),
        );
        Agent::new(client, "m1", 4, 4)
    }

    fn test_evidence(id: &str, sequence: u64) -> EvidenceRecord {
        EvidenceRecord {
            id: id.into(),
            sequence,
            timestamp_ms: 0,
            evidence_kind: crate::evidence::EvidenceKind::Decision,
            title: format!("evidence {id}"),
            summary: format!("summary {id}"),
            detail: None,
            source: EvidenceSource::Transcript { sequence },
            tags: Vec::new(),
        }
    }

    fn runtime_frames_for_history(history: &[HistoryItem]) -> Vec<RuntimeFrame> {
        crate::protocol_frames::history_items_to_frames(history)
            .iter()
            .enumerate()
            .map(|(ordinal, frame)| runtime_frame_from_protocol_frame(frame, ordinal as u32))
            .collect()
    }

    fn runtime_snapshot_for_history(
        branch_id: impl Into<String>,
        history: &[HistoryItem],
    ) -> RuntimeSnapshot {
        let mut snapshot = RuntimeSnapshot::new(branch_id);
        snapshot.frames = runtime_frames_for_history(history);
        snapshot
    }

    #[test]
    fn runtime_compaction_applies_repeatedly_with_cumulative_ids_and_retained_frames() {
        let mut agent = test_agent();
        let history = vec![
            HistoryItem::user("first"),
            HistoryItem::assistant("second"),
            HistoryItem::user("retained"),
        ];
        agent.replace_history(history).expect("valid history");
        agent.runtime_snapshot = compaction::test_snapshot_for_history(&agent.history);
        let first_id = agent.runtime_snapshot.frames[0].id;
        let second_id = agent.runtime_snapshot.frames[1].id;
        let retained_id = agent.runtime_snapshot.frames[2].id;
        let first_span = agent.runtime_snapshot.frames[0]
            .provenance
            .source_span
            .unwrap();
        let second_span = agent.runtime_snapshot.frames[1]
            .provenance
            .source_span
            .unwrap();
        let first = compaction::CompactionSelection {
            previous_summary: None,
            head_for_summary: vec![HistoryItem::user("first")],
            tail_items: Vec::new(),
            tail_start_index: 1,
            retired_frame_ids: vec![first_id],
            dependent_frame_ids: Vec::new(),
            retired_source_spans: vec![first_span],
        };
        agent
            .apply_runtime_compaction(&first, "first summary".into())
            .expect("first apply succeeds");
        let summary_id = agent.runtime_snapshot.frames[0].id;
        let second = compaction::CompactionSelection {
            previous_summary: Some("first summary".into()),
            head_for_summary: vec![HistoryItem::assistant("second")],
            tail_items: Vec::new(),
            tail_start_index: 1,
            retired_frame_ids: vec![second_id],
            dependent_frame_ids: Vec::new(),
            retired_source_spans: vec![second_span],
        };
        agent
            .apply_runtime_compaction(&second, "second summary".into())
            .expect("second apply succeeds");

        assert_eq!(
            agent.runtime_snapshot.compaction.compacted_frame_ids,
            vec![first_id, second_id]
        );
        assert_eq!(agent.runtime_snapshot.frames[0].id, summary_id);
        assert!(agent.runtime_snapshot.frames.iter().any(|frame| frame.id == retained_id
            && frame.visibility == FrameVisibility::Active));
    }

    #[test]
    fn runtime_compaction_overlap_failure_is_atomic() {
        let mut agent = test_agent();
        let history = vec![HistoryItem::user("old"), HistoryItem::user("retained")];
        agent.replace_history(history).expect("valid history");
        agent.runtime_snapshot = compaction::test_snapshot_for_history(&agent.history);
        let before = agent.runtime_snapshot.clone();
        let invalid = compaction::CompactionSelection {
            previous_summary: None,
            head_for_summary: vec![HistoryItem::user("old")],
            tail_items: Vec::new(),
            tail_start_index: 1,
            retired_frame_ids: vec![before.frames[0].id],
            dependent_frame_ids: Vec::new(),
            retired_source_spans: vec![SourceSpan::new(1, 2).unwrap()],
        };

        assert!(
            agent
                .apply_runtime_compaction(&invalid, "summary".into())
                .is_err()
        );
        assert_eq!(agent.runtime_snapshot, before);
    }

    #[test]
    fn runtime_snapshot_provider_refresh_retains_durable_metadata() {
        let mut agent = test_agent();
        let history = vec![HistoryItem::user("current")];
        agent
            .replace_history(history.clone())
            .expect("valid history");
        agent.runtime_snapshot = compaction::test_snapshot_for_history(&history);
        let session_frame = RuntimeFrame::new(
            RuntimeFrameKind::Metadata,
            FrameVisibility::Active,
            RuntimeFrameProvenance::new(RuntimeSource::SessionState),
            RuntimeFrameIdSeed {
                frame_kind: RuntimeFrameKind::Metadata,
                source: RuntimeSource::SessionState,
                ordinal: 0,
                stable_key: "durable-session-state",
                source_span: None,
            },
        );
        let session_frame_id = session_frame.id;
        agent.runtime_snapshot.push_frame(session_frame);
        agent
            .runtime_snapshot
            .push_child_session(RuntimeChildSession {
                parent_run_id: "parent".into(),
                child_session_id: "child".into(),
                agent_name: "explorer".into(),
                status: "completed".into(),
                summary: "retained".into(),
                timestamp_ms: 1,
            });
        agent
            .runtime_snapshot
            .push_prompt_contributor(PromptContributorPlaceholder {
                contributor_id: "contributor".into(),
                kind: PromptContributorKind::RuntimeContext,
                label: Some("retained".into()),
                provenance: RuntimeFrameProvenance::new(RuntimeSource::ContextView),
                frame_ids: Vec::new(),
                source_frame_ids: Vec::new(),
            });
        let projected = runtime_snapshot_for_history("main", &history);
        agent.set_runtime_snapshot_provider(Arc::new(move || Ok(projected.clone())));

        agent
            .refresh_runtime_snapshot_from_provider()
            .expect("refresh succeeds");

        assert!(
            agent
                .runtime_snapshot
                .frames
                .iter()
                .any(|frame| frame.id == session_frame_id)
        );
        assert_eq!(
            agent.runtime_snapshot.child_sessions[0].child_session_id,
            "child"
        );
        assert_eq!(
            agent.runtime_snapshot.prompt_contributors[0].contributor_id,
            "contributor"
        );
    }

    #[test]
    fn runtime_snapshot_provider_refresh_accepts_empty_context_projection() {
        let mut agent = test_agent();
        let history = vec![HistoryItem::user("current")];
        agent
            .replace_history(history.clone())
            .expect("valid history");
        agent.runtime_snapshot = runtime_snapshot_for_history("main", &history);
        let records = vec![transcript_record(
            1,
            TranscriptEvent::AssistantMessage {
                content: "stale context".into(),
            },
        )];
        agent
            .runtime_snapshot
            .set_context_view(project_context_view(&records).expect("context view"));
        agent
            .runtime_snapshot
            .set_context_tree(project_context_tree(&records).expect("context tree"));
        let projected = runtime_snapshot_for_history("main", &history);
        agent.set_runtime_snapshot_provider(Arc::new(move || Ok(projected.clone())));

        agent
            .refresh_runtime_snapshot_from_provider()
            .expect("refresh succeeds");

        assert_eq!(
            agent.runtime_snapshot.context_view,
            ContextViewProjection::default()
        );
        assert_eq!(
            agent.runtime_snapshot.context_tree,
            ContextTreeState::with_default_root()
        );
    }

    #[test]
    fn evidence_has_one_runtime_snapshot_authority_and_failed_candidates_are_atomic() {
        let mut agent = test_agent();
        agent
            .add_evidence(test_evidence("ev-live", 1))
            .expect("add live evidence");
        assert_eq!(agent.evidence(), agent.runtime_snapshot.evidence.as_slice());

        let before = agent.runtime_snapshot.evidence.clone();
        assert!(agent.add_evidence(test_evidence("ev-live", 2)).is_err());
        assert_eq!(agent.runtime_snapshot.evidence, before);

        assert!(
            agent
                .restore_evidence(vec![
                    test_evidence("ev-duplicate", 2),
                    test_evidence("ev-duplicate", 3)
                ])
                .is_err()
        );
        assert_eq!(agent.runtime_snapshot.evidence, before);

        assert!(
            agent
                .restore_session_history(
                    Vec::new(),
                    vec![
                        test_evidence("ev-duplicate", 2),
                        test_evidence("ev-duplicate", 3),
                    ],
                    0,
                )
                .is_err()
        );
        assert_eq!(agent.runtime_snapshot.evidence, before);

        let mut invalid_restore = RuntimeSnapshot::new(ROOT_CONTEXT_BRANCH_ID);
        invalid_restore.set_evidence(vec![
            test_evidence("ev-duplicate", 2),
            test_evidence("ev-duplicate", 3),
        ]);
        assert!(
            agent
                .restore_runtime_snapshot(Vec::new(), invalid_restore)
                .is_err()
        );
        assert_eq!(agent.runtime_snapshot.evidence, before);

        let mut replacement = RuntimeSnapshot::new(ROOT_CONTEXT_BRANCH_ID);
        replacement.set_evidence(vec![test_evidence("ev-provider", 4)]);
        agent.set_runtime_snapshot_provider(Arc::new(move || Ok(replacement.clone())));
        agent
            .replace_runtime_snapshot_from_provider()
            .expect("replace provider snapshot");
        assert_eq!(agent.evidence(), agent.runtime_snapshot.evidence.as_slice());
        assert_eq!(agent.evidence()[0].id, "ev-provider");

        let before = agent.runtime_snapshot.evidence.clone();
        let mut invalid = RuntimeSnapshot::new(ROOT_CONTEXT_BRANCH_ID);
        invalid.set_evidence(vec![
            test_evidence("ev-duplicate", 5),
            test_evidence("ev-duplicate", 6),
        ]);
        agent.set_runtime_snapshot_provider(Arc::new(move || Ok(invalid.clone())));
        assert!(agent.replace_runtime_snapshot_from_provider().is_err());
        assert_eq!(agent.runtime_snapshot.evidence, before);
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

    #[tokio::test]
    async fn runtime_snapshot_allows_context_tool_execution_without_context_provider() {
        let mut agent = test_agent();
        let records = vec![
            transcript_record(
                1,
                TranscriptEvent::UserMessage {
                    content: crate::user_content::UserMessageContent::from(
                        "append-only requirement",
                    ),
                },
            ),
            transcript_record(
                2,
                TranscriptEvent::ContextNodeCreated {
                    node_id: "node-a".into(),
                    parent_node_id: Some("root".into()),
                    label: Some("node a".into()),
                    purpose: Some("tool snapshot".into()),
                    block_ref: None,
                    source_ref: None,
                },
            ),
            transcript_record(
                3,
                TranscriptEvent::ContextNodeLifecycle {
                    node_id: "root".into(),
                    status: ContextNodeStatus::Inactive,
                },
            ),
            transcript_record(
                4,
                TranscriptEvent::ContextNodeLifecycle {
                    node_id: "node-a".into(),
                    status: ContextNodeStatus::Active,
                },
            ),
        ];
        agent
            .runtime_snapshot
            .set_context_view(project_context_view(&records).expect("context view"));
        agent
            .runtime_snapshot
            .set_context_tree(project_context_tree(&records).expect("context tree"));

        let call = HistoryToolCall {
            call_id: "call-1".into(),
            name: tool_names::TOOL_CONTEXT_LIST.into(),
            arguments_json: json!({"include_archived":false,"include_removed":false,"limit":null})
                .to_string(),
        };

        let record = tool_execution::execute_tool_call(
            &mut agent,
            &call,
            &mut |_| async { Ok(()) },
            &mut |_| async { Ok(PermissionApproval::Deny) },
        )
        .await
        .expect("context tool executes with injected snapshots");

        assert!(record.output.ok, "{:?}", record.output);
        let nodes = record
            .output
            .data
            .as_ref()
            .and_then(|data| data.get("nodes"))
            .and_then(Value::as_array)
            .expect("nodes array");
        assert!(nodes.iter().any(|node| node["ref_id"] == "node-a"));
    }

    #[tokio::test]
    async fn non_context_tool_execution_does_not_require_snapshot_provider() {
        let mut agent = test_agent();
        let call = HistoryToolCall {
            call_id: "call-echo".into(),
            name: tool_names::TOOL_UTIL_ECHO.into(),
            arguments_json: json!({"text":"hello"}).to_string(),
        };

        let record = tool_execution::execute_tool_call(
            &mut agent,
            &call,
            &mut |_| async { Ok(()) },
            &mut |_| async { Ok(PermissionApproval::Deny) },
        )
        .await
        .expect("non-context tool executes without snapshots");

        assert!(record.output.ok, "{:?}", record.output);
    }

    #[tokio::test]
    async fn context_tool_execution_without_snapshot_provider_uses_runtime_snapshot() {
        let mut agent = test_agent();
        let call = HistoryToolCall {
            call_id: "call-context".into(),
            name: tool_names::TOOL_CONTEXT_LIST.into(),
            arguments_json: json!({"include_archived":false,"include_removed":false,"limit":null})
                .to_string(),
        };

        let record = tool_execution::execute_tool_call(
            &mut agent,
            &call,
            &mut |_| async { Ok(()) },
            &mut |_| async { Ok(PermissionApproval::Deny) },
        )
        .await
        .expect("context tool execution is returned as record");

        assert!(record.output.ok, "{:?}", record.output);
    }

    #[test]
    fn agent_iteration_limit_allows_tool_budget_plus_final_round() {
        let client = Client::with_config(
            OpenAIConfig::new()
                .with_api_base("https://api.openai.com/v1")
                .with_api_key("test"),
        );
        let agent = Agent::new(client, "m1", 64, 128);

        assert_eq!(agent.max_tool_calls_limit(), Some(128));
        assert_eq!(agent.max_iterations_limit(), Some(64));
    }

    #[test]
    fn agent_iteration_limit_preserves_larger_configured_limit() {
        let client = Client::with_config(
            OpenAIConfig::new()
                .with_api_base("https://api.openai.com/v1")
                .with_api_key("test"),
        );
        let agent = Agent::new(client, "m1", 200, 128);

        assert_eq!(agent.max_iterations_limit(), Some(200));
    }

    #[test]
    fn agent_limits_are_unbounded_by_default_when_omitted() {
        let client = Client::with_config(
            OpenAIConfig::new()
                .with_api_base("https://api.openai.com/v1")
                .with_api_key("test"),
        );
        let agent = Agent::new(client, "m1", None, None);

        assert_eq!(agent.max_iterations_limit(), None);
        assert_eq!(agent.max_tool_calls_limit(), None);
    }

    fn complete_http_request_len(request: &[u8]) -> Option<usize> {
        let header_end = request
            .windows(4)
            .position(|window| window == b"\r\n\r\n")?;
        let headers = std::str::from_utf8(&request[..header_end])
            .expect("test client sends UTF-8 HTTP headers");
        let content_length = headers
            .lines()
            .find_map(|header| {
                header
                    .split_once(':')
                    .filter(|(name, _)| name.eq_ignore_ascii_case("content-length"))
                    .map(|(_, value)| {
                        value
                            .trim()
                            .parse::<usize>()
                            .expect("test client sends a numeric content length")
                    })
            })
            .unwrap_or(0);
        Some(header_end + 4 + content_length)
    }

    async fn read_complete_http_request(socket: &mut tokio::net::TcpStream) {
        let mut request = Vec::new();
        loop {
            if complete_http_request_len(&request).is_some_and(|length| request.len() >= length) {
                return;
            }
            let read = socket
                .read_buf(&mut request)
                .await
                .expect("server reads request");
            assert_ne!(read, 0, "test client closed before completing its request");
        }
    }

    async fn spawn_chat_completion_server(
        responses: Vec<&'static str>,
    ) -> (String, Arc<AtomicUsize>, JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("test server should bind");
        let addr = listener.local_addr().expect("test server has local addr");
        let count = Arc::new(AtomicUsize::new(0));
        let server_count = count.clone();
        let handle = tokio::spawn(async move {
            for response in responses {
                let (mut socket, _) = listener.accept().await.expect("server accepts request");
                // A single read can stop anywhere in the headers or request body.  Serving
                // the next scripted response at that point lets a connection close race the
                // client's upload, which made response sequencing intermittent under a busy
                // serial suite.  Consume the complete request before advancing the script.
                read_complete_http_request(&mut socket).await;
                server_count.fetch_add(1, Ordering::SeqCst);
                socket
                    .write_all(response.as_bytes())
                    .await
                    .expect("server writes response");
                socket.shutdown().await.expect("server closes response");
            }
        });
        (format!("http://{addr}"), count, handle)
    }

    fn sse_response(body: String) -> &'static str {
        Box::leak(
            format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            )
            .into_boxed_str(),
        )
    }

    fn responses_tool_batch_sse(calls: Vec<serde_json::Value>) -> &'static str {
        let response = json!({
            "type": "response.completed", "sequence_number": 1,
            "response": {
                "id": "r-tools", "object": "response", "created_at": 1,
                "status": "completed", "background": false, "error": null,
                "incomplete_details": null, "instructions": null, "max_output_tokens": null,
                "model": "m1", "output": calls, "parallel_tool_calls": true,
                "previous_response_id": null, "reasoning": {}, "store": true,
                "temperature": 1, "text": {"format": {"type": "text"}},
                "tool_choice": "auto", "tools": [], "top_p": 1,
                "truncation": "disabled",
                "usage": {"input_tokens": 1, "input_tokens_details": {"cached_tokens": 0}, "output_tokens": 1, "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 2},
                "user": null, "metadata": {}
            }
        });
        sse_response(format!("data: {response}\n\ndata: [DONE]\n\n"))
    }

    fn responses_final_sse(text: &str) -> &'static str {
        let response = json!({
            "type": "response.completed", "sequence_number": 1,
            "response": {
                "id": "r-final", "object": "response", "created_at": 1,
                "status": "completed", "background": false, "error": null,
                "incomplete_details": null, "instructions": null, "max_output_tokens": null,
                "model": "m1", "output": [{"type": "message", "id": "m1", "status": "completed", "role": "assistant", "content": [{"type": "output_text", "text": text, "annotations": []}]}],
                "parallel_tool_calls": true, "previous_response_id": null, "reasoning": {},
                "store": true, "temperature": 1, "text": {"format": {"type": "text"}},
                "tool_choice": "auto", "tools": [], "top_p": 1, "truncation": "disabled",
                "usage": {"input_tokens": 1, "input_tokens_details": {"cached_tokens": 0}, "output_tokens": 1, "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 2},
                "user": null, "metadata": {}
            }
        });
        sse_response(format!("data: {response}\n\ndata: [DONE]\n\n"))
    }

    fn checkpoint_stream_agent(base_url: String, protocol: ApiProtocol) -> Agent<OpenAIConfig> {
        let client = Client::with_config(
            OpenAIConfig::new()
                .with_api_base(base_url)
                .with_api_key("test"),
        );
        let mut agent = Agent::new(client, "m1", 4, 4);
        agent.set_default_protocol(protocol);
        agent.set_model_catalog(HashMap::from([(
            "m1".into(),
            ModelRequestMetadata {
                context_window: Some(32_000),
                max_output_tokens: Some(2_000),
                supports_tools: true,
                ..Default::default()
            },
        )]));
        agent.set_logical_checkpoint_config(LogicalCheckpointConfig {
            enabled: true,
            ..Default::default()
        });
        // The stream owns turn 1; seed the live segment that it continues.
        agent.turn.turn_id = 1;
        agent.runtime_snapshot.current_turn_id = Some(1);
        agent.runtime_snapshot.current_segment_id = Some(1);
        agent.runtime_snapshot.leaf_sequence = Some(9);
        agent.runtime_snapshot.latest_model = Some("m1".into());
        let candidate = prepared_checkpoint_for(&agent);
        agent.set_logical_checkpoint_candidate_provider(Arc::new(move || Ok(candidate.clone())));
        agent
    }

    async fn assert_checkpointed_tool_stream(protocol: ApiProtocol) {
        let response_body = r#"data: {"type":"response.completed","sequence_number":1,"response":{"id":"r-tools","object":"response","created_at":1,"status":"completed","background":false,"error":null,"incomplete_details":null,"instructions":null,"max_output_tokens":null,"model":"m1","output":[{"type":"function_call","id":"f1","call_id":"one","name":"test__replay_guard","arguments":"{}","status":"completed"},{"type":"function_call","id":"f2","call_id":"two","name":"test__replay_guard","arguments":"{}","status":"completed"}],"parallel_tool_calls":true,"previous_response_id":null,"reasoning":{},"store":true,"temperature":1,"text":{"format":{"type":"text"}},"tool_choice":"auto","tools":[],"top_p":1,"truncation":"disabled","usage":{"input_tokens":1,"input_tokens_details":{"cached_tokens":0},"output_tokens":1,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":2},"user":null,"metadata":{}}}

data: [DONE]

"#;
        let response_final = r#"data: {"type":"response.completed","sequence_number":1,"response":{"id":"r-final","object":"response","created_at":1,"status":"completed","background":false,"error":null,"incomplete_details":null,"instructions":null,"max_output_tokens":null,"model":"m1","output":[{"type":"message","id":"m1","status":"completed","role":"assistant","content":[{"type":"output_text","text":"done","annotations":[]}]}],"parallel_tool_calls":true,"previous_response_id":null,"reasoning":{},"store":true,"temperature":1,"text":{"format":{"type":"text"}},"tool_choice":"auto","tools":[],"top_p":1,"truncation":"disabled","usage":{"input_tokens":1,"input_tokens_details":{"cached_tokens":0},"output_tokens":1,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":2},"user":null,"metadata":{}}}

data: [DONE]

"#;
        let chat_tools = r#"data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"one","type":"function","function":{"name":"test__replay_guard","arguments":"{}"}},{"index":1,"id":"two","type":"function","function":{"name":"test__replay_guard","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}

data: [DONE]

"#;
        let chat_final = r#"data: {"choices":[{"index":0,"delta":{"content":"done"},"finish_reason":"stop"}]}

data: [DONE]

"#;
        let bodies = match protocol {
            ApiProtocol::Responses => vec![
                sse_response(response_body.into()),
                sse_response(response_final.into()),
            ],
            ApiProtocol::Completions => vec![
                sse_response(chat_tools.into()),
                sse_response(chat_final.into()),
            ],
        };
        let (base_url, requests, server) = spawn_chat_completion_server(bodies).await;
        let mut agent = checkpoint_stream_agent(base_url, protocol);
        let executions = Arc::new(AtomicUsize::new(0));
        agent.register_tool(ReplayGuardTool(executions.clone()));
        assert_eq!(
            agent.request_logical_checkpoint(),
            LogicalCheckpointRequestOutcome::Queued
        );
        let mut events = Vec::new();
        let result = agent
            .run_stream_async(
                "continue",
                |_| std::future::ready(Ok(())),
                |event| {
                    events.push(match event {
                        AgentEvent::AssistantToolCallBatch { .. } => "batch",
                        AgentEvent::ToolCallBatchFinished => "finished",
                        AgentEvent::LogicalCheckpoint { .. } => "checkpoint",
                        AgentEvent::AssistantMessage { .. } => "final",
                        AgentEvent::ModelStreamIssue { .. } => "recovery",
                        _ => "other",
                    });
                    std::future::ready(Ok(()))
                },
                |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
            )
            .await
            .expect("tool batch and successor request should complete");

        assert_eq!(result, "done");
        assert_eq!(requests.load(Ordering::SeqCst), 2);
        assert_eq!(
            executions.load(Ordering::SeqCst),
            2,
            "each tool runs once, never on the successor request"
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| **event == "checkpoint")
                .count(),
            1
        );
        let batch = events
            .iter()
            .position(|event| *event == "batch")
            .expect("tool batch");
        let finished = events
            .iter()
            .position(|event| *event == "finished")
            .expect("batch finished");
        let checkpoint = events
            .iter()
            .position(|event| *event == "checkpoint")
            .expect("checkpoint");
        let final_message = events
            .iter()
            .position(|event| *event == "final")
            .expect("final reply");
        assert!(
            batch < finished && finished < checkpoint && checkpoint < final_message,
            "{events:?}"
        );
        assert!(!events[..finished].contains(&"checkpoint"));
        assert!(!events.contains(&"recovery"));
        // Finalization clears the per-turn start marker, while the checkpointed
        // successor remains in the same advanced segment.
        assert_eq!(agent.turn.current_turn_start_index, None);
        assert_eq!(agent.runtime_snapshot.current_segment_id, Some(2));
        server.await.expect("server task should finish");
    }

    #[tokio::test]
    async fn logical_checkpoint_responses_stream_commits_only_after_multi_tool_batch() {
        assert_checkpointed_tool_stream(ApiProtocol::Responses).await;
    }

    #[tokio::test]
    async fn logical_checkpoint_chat_stream_commits_only_after_multi_tool_batch() {
        assert_checkpointed_tool_stream(ApiProtocol::Completions).await;
    }

    #[tokio::test]
    async fn phase3b_live_checkpoint_preserves_101_complete_tool_pairs_without_reexecution() {
        let calls = (0..101)
            .map(|index| {
                json!({
                    "type": "function_call",
                    "id": format!("f-{index}"),
                    "call_id": format!("call-{index}"),
                    "name": "test__replay_guard",
                    "arguments": "{}",
                    "status": "completed"
                })
            })
            .collect::<Vec<_>>();
        let tools = json!({
            "type": "response.completed",
            "sequence_number": 1,
            "response": {
                "id": "r-101-tools", "object": "response", "created_at": 1,
                "status": "completed", "background": false, "error": null,
                "incomplete_details": null, "instructions": null,
                "max_output_tokens": null, "model": "m1", "output": calls,
                "parallel_tool_calls": true, "previous_response_id": null,
                "reasoning": {}, "store": true, "temperature": 1,
                "text": {"format": {"type": "text"}}, "tool_choice": "auto",
                "tools": [], "top_p": 1, "truncation": "disabled",
                "usage": {"input_tokens": 1, "input_tokens_details": {"cached_tokens": 0}, "output_tokens": 1, "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 2},
                "user": null, "metadata": {}
            }
        });
        let final_reply = r#"data: {"type":"response.completed","sequence_number":1,"response":{"id":"r-101-final","object":"response","created_at":1,"status":"completed","background":false,"error":null,"incomplete_details":null,"instructions":null,"max_output_tokens":null,"model":"m1","output":[{"type":"message","id":"m1","status":"completed","role":"assistant","content":[{"type":"output_text","text":"done","annotations":[]}]}],"parallel_tool_calls":true,"previous_response_id":null,"reasoning":{},"store":true,"temperature":1,"text":{"format":{"type":"text"}},"tool_choice":"auto","tools":[],"top_p":1,"truncation":"disabled","usage":{"input_tokens":1,"input_tokens_details":{"cached_tokens":0},"output_tokens":1,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":2},"user":null,"metadata":{}}}

data: [DONE]

"#;
        let tool_reply = format!("data: {}\n\ndata: [DONE]\n\n", tools);
        let (base_url, requests, server) = spawn_chat_completion_server(vec![
            sse_response(tool_reply),
            sse_response(final_reply.into()),
        ])
        .await;
        let mut agent = checkpoint_stream_agent(base_url, ApiProtocol::Responses);
        agent.max_tool_calls = Some(128);
        agent.max_iterations = Some(4);
        let executions = Arc::new(AtomicUsize::new(0));
        agent.register_tool(ReplayGuardTool(Arc::clone(&executions)));
        assert_eq!(
            agent.request_logical_checkpoint(),
            LogicalCheckpointRequestOutcome::Queued
        );
        let mut checkpoint_count = 0;
        let mut observed_pair_count = 0;
        let result = agent
            .run_stream_async(
                "continue",
                |_| std::future::ready(Ok(())),
                |event| {
                    if let AgentEvent::AssistantToolCallBatch { ref calls, .. } = event {
                        observed_pair_count += calls.len();
                    }
                    if matches!(event, AgentEvent::LogicalCheckpoint { .. }) {
                        checkpoint_count += 1;
                    }
                    std::future::ready(Ok(()))
                },
                |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
            )
            .await
            .expect("101 complete tool pairs must produce a valid successor request");

        assert_eq!(result, "done");
        assert_eq!(requests.load(Ordering::SeqCst), 2);
        assert_eq!(observed_pair_count, 101);
        assert_eq!(executions.load(Ordering::SeqCst), 101);
        assert_eq!(checkpoint_count, 1);
        assert_eq!(agent.runtime_snapshot.current_segment_id, Some(2));
        assert!(
            crate::protocol_frames::validate_history_items_complete(&agent.history, None).is_ok()
        );
        server.await.expect("server task should finish");
    }

    #[tokio::test]
    async fn phase3b_live_turn_commits_two_successor_segments_after_distinct_batches() {
        let first_batch = vec![json!({
            "type":"function_call", "id":"f1", "call_id":"one",
            "name":"test__replay_guard", "arguments":"{}", "status":"completed"
        })];
        let second_batch = vec![json!({
            "type":"function_call", "id":"f2", "call_id":"two",
            "name":"test__replay_guard", "arguments":"{}", "status":"completed"
        })];
        let (base_url, requests, server) = spawn_chat_completion_server(vec![
            responses_tool_batch_sse(first_batch),
            responses_tool_batch_sse(second_batch),
            responses_final_sse("done"),
        ])
        .await;
        let mut agent = checkpoint_stream_agent(base_url, ApiProtocol::Responses);
        let first = prepared_checkpoint_for_lineage(&agent, "checkpoint-live-1", None);
        let mut after_first =
            checkpoint_stream_agent("http://unused".into(), ApiProtocol::Responses);
        after_first.runtime_snapshot.current_segment_id = Some(2);
        after_first.runtime_snapshot.leaf_sequence = Some(10);
        let second = prepared_checkpoint_for_lineage(
            &after_first,
            "checkpoint-live-2",
            Some("checkpoint-live-1"),
        );
        let candidates = Arc::new(vec![first, second]);
        let candidate_index = Arc::new(AtomicUsize::new(0));
        agent.set_logical_checkpoint_candidate_provider({
            let candidates = Arc::clone(&candidates);
            let candidate_index = Arc::clone(&candidate_index);
            Arc::new(move || {
                candidates
                    .get(candidate_index.fetch_add(1, Ordering::SeqCst))
                    .cloned()
                    .ok_or_else(|| anyhow!("unexpected third checkpoint"))
            })
        });
        let executions = Arc::new(AtomicUsize::new(0));
        agent.register_tool(ReplayGuardTool(Arc::clone(&executions)));
        assert_eq!(
            agent.request_logical_checkpoint(),
            LogicalCheckpointRequestOutcome::Queued
        );
        let checkpoint_control = agent.logical_checkpoint_control.clone();
        let mut checkpoints = Vec::new();
        let mut batches = 0;
        let result = agent
            .run_stream_async(
                "continue",
                |_| std::future::ready(Ok(())),
                |event| {
                    if matches!(event, AgentEvent::AssistantToolCallBatch { .. }) {
                        batches += 1;
                        if batches == 2 {
                            assert_eq!(
                                checkpoint_control.request(),
                                LogicalCheckpointRequestOutcome::Queued
                            );
                        }
                    }
                    if let AgentEvent::LogicalCheckpoint { event, .. } = event {
                        checkpoints.push((
                            event.previous_segment_id,
                            event.segment_id,
                            event.previous_checkpoint_id,
                        ));
                    }
                    std::future::ready(Ok(()))
                },
                |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
            )
            .await
            .expect("two completed batches commit two successors");

        assert_eq!(result, "done");
        assert_eq!(requests.load(Ordering::SeqCst), 3);
        assert_eq!(executions.load(Ordering::SeqCst), 2);
        assert_eq!(
            checkpoints,
            vec![(1, 2, None), (2, 3, Some("checkpoint-live-1".into()))]
        );
        assert_eq!(agent.runtime_snapshot.current_segment_id, Some(3));
        assert!(
            crate::protocol_frames::validate_history_items_complete(&agent.history, None).is_ok()
        );
        server.await.expect("server task should finish");
    }

    #[tokio::test]
    async fn phase3b_live_workflow_controls_are_preserved_by_successor_checkpoint() {
        let calls = vec![
            json!({"type":"function_call", "id":"todos", "call_id":"todos", "name":"workflow__todos", "arguments":"{\"items\":[{\"id\":\"t1\",\"content\":\"ship\",\"status\":\"pending\"}]}", "status":"completed"}),
            json!({"type":"function_call", "id":"continue", "call_id":"continue", "name":"workflow__auto_continue", "arguments":"{\"enabled\":true,\"max_continuations\":2}", "status":"completed"}),
            json!({"type":"function_call", "id":"reset", "call_id":"reset", "name":"workflow__todos", "arguments":"{\"items\":[]}", "status":"completed"}),
        ];
        let (base_url, requests, server) = spawn_chat_completion_server(vec![
            responses_tool_batch_sse(calls),
            responses_final_sse("done"),
        ])
        .await;
        let mut agent = checkpoint_stream_agent(base_url, ApiProtocol::Responses);
        let mut candidate = prepared_checkpoint_for(&agent);
        candidate.projected_workflow = Some(crate::transcript::CheckpointWorkflowProjection {
            todos: Vec::new(),
            auto_continue: AutoContinueState {
                enabled: true,
                max_continuations: 2,
            },
        });
        agent.set_logical_checkpoint_candidate_provider(Arc::new(move || Ok(candidate.clone())));
        assert_eq!(
            agent.request_logical_checkpoint(),
            LogicalCheckpointRequestOutcome::Queued
        );
        let result = agent
            .run_stream_async(
                "continue",
                |_| std::future::ready(Ok(())),
                |_| std::future::ready(Ok(())),
                |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
            )
            .await
            .expect("workflow successor request succeeds");

        assert_eq!(result, "done");
        assert_eq!(requests.load(Ordering::SeqCst), 2);
        let expected = WorkflowState {
            todos: Vec::new(),
            auto_continue: AutoContinueState {
                enabled: true,
                max_continuations: 2,
            },
        };
        assert_eq!(agent.turn.workflow, expected);
        server.await.expect("server task should finish");
    }

    #[tokio::test]
    async fn phase3b_live_checkpoint_eligibility_rejections_are_nonpersistent_and_release_leases() {
        type Configure =
            fn(&mut Agent<OpenAIConfig>, &mut crate::transcript::PreparedLogicalCheckpoint);
        let cases: Vec<(&str, Configure)> = vec![
            ("active context experiment", |agent, _| {
                agent.set_context_scope_state(Arc::new(std::sync::Mutex::new(ContextScopeState {
                    active_experiment: Some(ActiveContextExperiment {
                        branch_id: "experiment".into(),
                        parent_branch_id: ROOT_CONTEXT_BRANCH_ID.into(),
                        base_sequence: 9,
                        writes_observed: false,
                    }),
                })));
            }),
            ("restored unavailable live turn", |agent, _| {
                agent.turn.current_turn_start_index = None;
            }),
            ("model protocol mismatch", |agent, _| {
                agent.set_model_protocols(HashMap::from([("m1".into(), ApiProtocol::Completions)]));
            }),
            ("scope mismatch", |_, candidate| {
                candidate.event.context_scope_revision += 1;
            }),
            ("workflow mismatch", |_, candidate| {
                candidate.projected_workflow =
                    Some(crate::transcript::CheckpointWorkflowProjection {
                        todos: vec![TodoItem {
                            id: "wrong".into(),
                            content: "wrong".into(),
                            status: TodoStatus::Pending,
                        }],
                        auto_continue: AutoContinueState::default(),
                    });
            }),
            ("prospective request overflow", |agent, _| {
                agent.set_model_catalog(HashMap::from([(
                    "m1".into(),
                    ModelRequestMetadata {
                        context_window: Some(1),
                        max_output_tokens: Some(1),
                        supports_tools: true,
                        ..Default::default()
                    },
                )]));
            }),
        ];

        for (name, configure) in cases {
            let (mut agent, prelude) = checkpoint_test_agent();
            let mut candidate = prepared_checkpoint_for(&agent);
            configure(&mut agent, &mut candidate);
            agent
                .set_logical_checkpoint_candidate_provider(Arc::new(move || Ok(candidate.clone())));
            let history = agent.history.clone();
            let frames = agent.protocol_frames.clone();
            let snapshot = agent.runtime_snapshot.clone();
            let workflow = agent.turn.workflow.clone();
            let run = agent.logical_checkpoint_control.begin_run();
            assert_eq!(
                agent.request_logical_checkpoint(),
                LogicalCheckpointRequestOutcome::Queued
            );
            let mut logical_events = 0;
            logical_checkpoint::commit_pending_at_batch_boundary(
                &mut agent,
                ApiProtocol::Responses,
                &prelude,
                0,
                &mut |event| {
                    logical_events +=
                        usize::from(matches!(event, AgentEvent::LogicalCheckpoint { .. }));
                    std::future::ready(Ok(()))
                },
            )
            .await
            .expect_err(name);
            assert_eq!(logical_events, 0, "{name} must not persist a logical event");
            assert_eq!(agent.history, history, "{name}");
            assert_eq!(agent.protocol_frames, frames, "{name}");
            assert_eq!(agent.runtime_snapshot, snapshot, "{name}");
            assert_eq!(agent.turn.workflow, workflow, "{name}");
            assert_eq!(
                agent.request_logical_checkpoint(),
                LogicalCheckpointRequestOutcome::Queued,
                "{name} lease"
            );
            drop(run);
        }
    }

    async fn assert_cancelled_stream_releases_checkpoint_lease(
        protocol: ApiProtocol,
        cancel_at_checkpoint: bool,
    ) {
        let response_tools = r#"data: {"type":"response.completed","sequence_number":1,"response":{"id":"r-tools","object":"response","created_at":1,"status":"completed","background":false,"error":null,"incomplete_details":null,"instructions":null,"max_output_tokens":null,"model":"m1","output":[{"type":"function_call","id":"f1","call_id":"one","name":"test__replay_guard","arguments":"{}","status":"completed"}],"parallel_tool_calls":true,"previous_response_id":null,"reasoning":{},"store":true,"temperature":1,"text":{"format":{"type":"text"}},"tool_choice":"auto","tools":[],"top_p":1,"truncation":"disabled","usage":{"input_tokens":1,"input_tokens_details":{"cached_tokens":0},"output_tokens":1,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":2},"user":null,"metadata":{}}}

data: [DONE]

"#;
        let chat_tools = r#"data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"one","type":"function","function":{"name":"test__replay_guard","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}

data: [DONE]

"#;
        let response_final = r#"data: {"type":"response.completed","sequence_number":1,"response":{"id":"r-final","object":"response","created_at":1,"status":"completed","background":false,"error":null,"incomplete_details":null,"instructions":null,"max_output_tokens":null,"model":"m1","output":[{"type":"message","id":"m1","status":"completed","role":"assistant","content":[{"type":"output_text","text":"clean","annotations":[]}]}],"parallel_tool_calls":true,"previous_response_id":null,"reasoning":{},"store":true,"temperature":1,"text":{"format":{"type":"text"}},"tool_choice":"auto","tools":[],"top_p":1,"truncation":"disabled","usage":{"input_tokens":1,"input_tokens_details":{"cached_tokens":0},"output_tokens":1,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":2},"user":null,"metadata":{}}}

data: [DONE]

"#;
        let chat_final = r#"data: {"choices":[{"index":0,"delta":{"content":"clean"},"finish_reason":"stop"}]}

data: [DONE]

"#;
        let bodies = match protocol {
            ApiProtocol::Responses => vec![
                sse_response(response_tools.into()),
                sse_response(response_final.into()),
            ],
            ApiProtocol::Completions => vec![
                sse_response(chat_tools.into()),
                sse_response(chat_final.into()),
            ],
        };
        let (base_url, requests, server) = spawn_chat_completion_server(bodies).await;
        let mut agent = checkpoint_stream_agent(base_url, protocol);
        let executions = Arc::new(AtomicUsize::new(0));
        agent.register_tool(ReplayGuardTool(Arc::clone(&executions)));
        assert_eq!(
            agent.request_logical_checkpoint(),
            LogicalCheckpointRequestOutcome::Queued
        );
        let entered = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let entered_event = Arc::clone(&entered);
        {
            let first = agent.run_stream_async(
                "cancelled turn",
                |_| std::future::ready(Ok(())),
                move |event| {
                    let stop = if cancel_at_checkpoint {
                        matches!(event, AgentEvent::LogicalCheckpoint { .. })
                    } else {
                        matches!(event, AgentEvent::ToolCallBatchFinished)
                    };
                    let entered = Arc::clone(&entered_event);
                    async move {
                        if stop {
                            entered.store(true, Ordering::SeqCst);
                            std::future::pending::<()>().await;
                        }
                        Ok(())
                    }
                },
                |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
            );
            tokio::pin!(first);
            for _ in 0..50 {
                if entered.load(Ordering::SeqCst) {
                    break;
                }
                tokio::select! {
                    _ = &mut first => panic!("cancelled stream completed before its cancellation point"),
                    _ = sleep(Duration::from_millis(10)) => {}
                }
            }
            assert!(
                entered.load(Ordering::SeqCst),
                "stream did not reach cancellation point"
            );
        }
        assert_eq!(
            agent.request_logical_checkpoint(),
            LogicalCheckpointRequestOutcome::Queued
        );

        let clean = agent
            .run_stream_async(
                "clean turn",
                |_| std::future::ready(Ok(())),
                |_| std::future::ready(Ok(())),
                |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
            )
            .await
            .expect("the successor run must not inherit the cancelled checkpoint lease");
        assert_eq!(clean, "clean");
        assert_eq!(requests.load(Ordering::SeqCst), 2);
        assert_eq!(executions.load(Ordering::SeqCst), 1);
        server.await.expect("server task should finish");
    }

    #[tokio::test]
    async fn phase3b_actual_responses_and_chat_stream_cancellation_releases_clean_next_run() {
        for protocol in [ApiProtocol::Responses, ApiProtocol::Completions] {
            assert_cancelled_stream_releases_checkpoint_lease(protocol, false).await;
            assert_cancelled_stream_releases_checkpoint_lease(protocol, true).await;
        }
    }

    #[tokio::test]
    async fn logical_checkpoint_rejection_matrix_is_nonpersistent_and_releases_its_lease() {
        type CandidateMutation = fn(&mut crate::transcript::PreparedLogicalCheckpoint);
        let cases: Vec<(&str, CandidateMutation)> = vec![
            ("stale frontier", |candidate| {
                candidate.expected_journal_frontier = 8
            }),
            ("frontier overflow", |candidate| {
                candidate.expected_journal_frontier = u64::MAX
            }),
            ("model", |candidate| {
                candidate.projected_snapshot.latest_model = Some("other".into())
            }),
            ("scope", |candidate| {
                candidate.event.context_scope_revision += 1
            }),
            ("branch", |candidate| {
                candidate.expected_branch_id = "other-branch".into();
                candidate.projected_snapshot.active_context.branch_id = "other-branch".into();
            }),
            ("leaf", |candidate| {
                candidate.projected_snapshot.leaf_sequence = Some(11)
            }),
            ("workflow", |candidate| {
                candidate.projected_workflow =
                    Some(crate::transcript::CheckpointWorkflowProjection {
                        todos: vec![TodoItem {
                            id: "different".into(),
                            content: "different".into(),
                            status: TodoStatus::Pending,
                        }],
                        auto_continue: AutoContinueState::default(),
                    })
            }),
            ("suffix", |candidate| {
                candidate.projected_protocol_frames.pop();
            }),
        ];

        for (name, mutate) in cases {
            let (mut agent, prelude) = checkpoint_test_agent();
            let mut candidate = prepared_checkpoint_for(&agent);
            mutate(&mut candidate);
            agent
                .set_logical_checkpoint_candidate_provider(Arc::new(move || Ok(candidate.clone())));
            let history = agent.history.clone();
            let frames = agent.protocol_frames.clone();
            let snapshot = agent.runtime_snapshot.clone();
            let workflow = agent.turn.workflow.clone();
            let run = agent.logical_checkpoint_control.begin_run();
            assert_eq!(
                agent.request_logical_checkpoint(),
                LogicalCheckpointRequestOutcome::Queued
            );
            let mut persisted = false;
            let error = logical_checkpoint::commit_pending_at_batch_boundary(
                &mut agent,
                ApiProtocol::Responses,
                &prelude,
                0,
                &mut |event| {
                    persisted |= matches!(event, AgentEvent::LogicalCheckpoint { .. });
                    std::future::ready(Ok(()))
                },
            )
            .await
            .expect_err(name);
            assert!(!error.to_string().is_empty(), "{name}");
            assert!(!persisted, "{name} must not emit a logical event");
            assert_eq!(agent.history, history, "{name}");
            assert_eq!(agent.protocol_frames, frames, "{name}");
            assert_eq!(agent.runtime_snapshot, snapshot, "{name}");
            assert_eq!(agent.turn.workflow, workflow, "{name}");
            assert_eq!(
                agent.request_logical_checkpoint(),
                LogicalCheckpointRequestOutcome::Queued,
                "{name} lease"
            );
            drop(run);
        }
    }

    #[tokio::test]
    async fn logical_checkpoint_disabled_and_unrequested_controls_are_exact_noops() {
        for enabled in [false, true] {
            let (mut agent, prelude) = checkpoint_test_agent();
            agent.set_logical_checkpoint_config(LogicalCheckpointConfig {
                enabled,
                ..Default::default()
            });
            let candidates = Arc::new(AtomicUsize::new(0));
            let candidate_count = Arc::clone(&candidates);
            agent.set_logical_checkpoint_candidate_provider(Arc::new(move || {
                candidate_count.fetch_add(1, Ordering::SeqCst);
                unreachable!("a no-op control must not ask for a checkpoint candidate")
            }));
            let history = agent.history.clone();
            let frames = agent.protocol_frames.clone();
            let snapshot = agent.runtime_snapshot.clone();
            let workflow = agent.turn.workflow.clone();
            let run = agent.logical_checkpoint_control.begin_run();
            if !enabled {
                assert_eq!(
                    agent.request_logical_checkpoint(),
                    LogicalCheckpointRequestOutcome::Disabled
                );
            }
            let result = logical_checkpoint::commit_pending_at_batch_boundary(
                &mut agent,
                ApiProtocol::Responses,
                &prelude,
                0,
                &mut |_| std::future::ready(Err(anyhow!("a no-op must emit no event"))),
            )
            .await
            .expect("disabled or unrequested control is a no-op");
            assert_eq!(result, None);
            assert_eq!(candidates.load(Ordering::SeqCst), 0);
            assert_eq!(agent.history, history);
            assert_eq!(agent.protocol_frames, frames);
            assert_eq!(agent.runtime_snapshot, snapshot);
            assert_eq!(agent.turn.workflow, workflow);
            drop(run);
        }
    }

    fn test_retry_config() -> RetryConfig {
        RetryConfig {
            enabled: true,
            max_attempts: 3,
            initial_delay_ms: 1,
            max_delay_ms: 5,
            backoff_multiplier: 2.0,
            jitter_ms: 0,
        }
    }

    fn test_tool_call(name: &str, arguments_json: &str) -> HistoryToolCall {
        HistoryToolCall {
            call_id: format!("call-{name}"),
            name: name.into(),
            arguments_json: arguments_json.into(),
        }
    }

    fn test_execution_record(tool_name: &str, output: ToolResult) -> ToolExecutionRecord {
        ToolExecutionRecord {
            call_id: format!("call-{tool_name}"),
            tool_name: tool_name.into(),
            arguments: Some(json!({})),
            permission_class: crate::permission::ToolPermissionClass::Read,
            directive: ExecutionDirective::None,
            status: ToolExecutionStatus::Executed,
            rejection: None,
            output,
            effects: ToolEffects {
                kind: ToolEffectKind::Read,
                primary_path: None,
                edited_paths: vec![],
                command: None,
            },
        }
    }

    struct SleepTool;

    #[async_trait]
    impl ToolHandler for SleepTool {
        fn name(&self) -> &str {
            "test__sleep"
        }

        fn description(&self) -> &str {
            "sleep test tool"
        }

        fn parameters(&self) -> Value {
            json!({
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": false
            })
        }

        async fn execute(&self, _args: Value) -> Result<Value> {
            sleep(Duration::from_millis(1_100)).await;
            Ok(json!({"done": true}))
        }
    }

    struct ReplayGuardTool(Arc<AtomicUsize>);

    #[async_trait]
    impl ToolHandler for ReplayGuardTool {
        fn name(&self) -> &str {
            "test__replay_guard"
        }

        fn description(&self) -> &str {
            "counts executions"
        }

        fn parameters(&self) -> Value {
            json!({"type": "object", "properties": {}, "additionalProperties": false})
        }

        async fn execute(&self, _args: Value) -> Result<Value> {
            self.0.fetch_add(1, Ordering::SeqCst);
            Ok(json!({"executed": true}))
        }
    }

    /// Makes the completed tool batch exceed the protected tail budget while
    /// leaving the checkpoint successor small.  This exercises the actual
    /// request-preparation path rather than injecting a scheduler decision.
    struct ProtectedOverflowTool(Arc<AtomicUsize>);

    #[async_trait]
    impl ToolHandler for ProtectedOverflowTool {
        fn name(&self) -> &str {
            "test__protected_overflow"
        }

        fn description(&self) -> &str {
            "produces a protected-tail overflow"
        }

        fn parameters(&self) -> Value {
            json!({"type": "object", "properties": {}, "additionalProperties": false})
        }

        async fn execute(&self, _args: Value) -> Result<Value> {
            self.0.fetch_add(1, Ordering::SeqCst);
            Ok(json!({"payload": "x ".repeat(50_000)}))
        }
    }

    /// Leaves the protected tail below the hard limit, but above the automatic
    /// high watermark.  This makes the stream exercise the post-fold soft path
    /// instead of the hard-overflow recovery path.
    struct SoftPressureTool(Arc<AtomicUsize>);

    #[async_trait]
    impl ToolHandler for SoftPressureTool {
        fn name(&self) -> &str {
            "test__soft_pressure"
        }

        fn description(&self) -> &str {
            "produces soft protected-tail pressure"
        }

        fn parameters(&self) -> Value {
            json!({"type": "object", "properties": {}, "additionalProperties": false})
        }

        async fn execute(&self, _args: Value) -> Result<Value> {
            self.0.fetch_add(1, Ordering::SeqCst);
            // This sits between the 30k soft high watermark and the protected
            // hard limit for the fixture's actual serialized tool output.
            Ok(json!({"payload": "x ".repeat(45_000)}))
        }
    }

    async fn assert_automatic_soft_pressure_checkpoint(protocol: ApiProtocol) {
        let responses_tools = responses_tool_batch_sse(vec![json!({
            "type": "function_call", "id": "f1", "call_id": "one",
            "name": "test__soft_pressure", "arguments": "{}", "status": "completed"
        })]);
        let chat_tools = sse_response(
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"one\",\"type\":\"function\",\"function\":{\"name\":\"test__soft_pressure\",\"arguments\":\"{}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\ndata: [DONE]\n\n".into(),
        );
        let bodies = match protocol {
            ApiProtocol::Responses => vec![responses_tools, responses_final_sse("done")],
            ApiProtocol::Completions => vec![
                chat_tools,
                sse_response("data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"done\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n".into()),
            ],
        };
        let (base_url, requests, server) = spawn_chat_completion_server(bodies).await;
        let mut agent = checkpoint_stream_agent(base_url, protocol);
        agent.set_model_catalog(HashMap::from([(
            "m1".into(),
            ModelRequestMetadata {
                context_window: Some(32_000),
                max_output_tokens: Some(128),
                supports_tools: true,
                ..Default::default()
            },
        )]));
        agent.set_compaction_config(CompactionConfig {
            protected_reserve_tokens: Some(4_000),
            ..Default::default()
        });
        agent.set_logical_checkpoint_config(LogicalCheckpointConfig {
            enabled: true,
            automatic: true,
            trigger_reserve_percent: 50,
            max_automatic_per_turn: 1,
        });
        let candidate = prepared_checkpoint_for(&agent);
        let candidates = Arc::new(AtomicUsize::new(0));
        agent.set_logical_checkpoint_candidate_provider({
            let candidates = Arc::clone(&candidates);
            Arc::new(move || {
                candidates.fetch_add(1, Ordering::SeqCst);
                Ok(candidate.clone())
            })
        });
        let executions = Arc::new(AtomicUsize::new(0));
        agent.register_tool(SoftPressureTool(Arc::clone(&executions)));
        let mut checkpoints = 0;
        let mut request_telemetry = 0;
        let answer = agent
            .run_stream_async(
                "continue",
                |_| std::future::ready(Ok(())),
                |event| {
                    match event {
                        AgentEvent::LogicalCheckpoint { .. } => checkpoints += 1,
                        AgentEvent::LlmRequestTelemetry(_) => request_telemetry += 1,
                        _ => {}
                    }
                    std::future::ready(Ok(()))
                },
                |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
            )
            .await
            .expect("soft pressure checkpoint rebuilds the successor request");

        assert_eq!(answer, "done");
        assert_eq!(requests.load(Ordering::SeqCst), 2);
        assert_eq!(
            executions.load(Ordering::SeqCst),
            1,
            "tool batch is not replayed"
        );
        assert_eq!(candidates.load(Ordering::SeqCst), 1);
        assert_eq!(checkpoints, 1);
        assert_eq!(
            request_telemetry, 4,
            "the discarded pre-checkpoint build emits neither request metadata nor telemetry"
        );
        assert_eq!(agent.runtime_snapshot.current_segment_id, Some(2));
        server.await.expect("server task should finish");
    }

    #[tokio::test]
    async fn phase3c_actual_responses_and_chat_soft_post_fold_pressure_rebuild_once() {
        for protocol in [ApiProtocol::Responses, ApiProtocol::Completions] {
            assert_automatic_soft_pressure_checkpoint(protocol).await;
        }
    }

    async fn assert_automatic_hard_overflow_checkpoint(protocol: ApiProtocol) {
        let response_tools = r#"data: {"type":"response.completed","sequence_number":1,"response":{"id":"r-tools","object":"response","created_at":1,"status":"completed","background":false,"error":null,"incomplete_details":null,"instructions":null,"max_output_tokens":null,"model":"m1","output":[{"type":"function_call","id":"f1","call_id":"one","name":"test__protected_overflow","arguments":"{}","status":"completed"}],"parallel_tool_calls":true,"previous_response_id":null,"reasoning":{},"store":true,"temperature":1,"text":{"format":{"type":"text"}},"tool_choice":"auto","tools":[],"top_p":1,"truncation":"disabled","usage":{"input_tokens":1,"input_tokens_details":{"cached_tokens":0},"output_tokens":1,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":2},"user":null,"metadata":{}}}

data: [DONE]

"#;
        let chat_tools = r#"data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"one","type":"function","function":{"name":"test__protected_overflow","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}

data: [DONE]

"#;
        let bodies = match protocol {
            ApiProtocol::Responses => vec![
                sse_response(response_tools.into()),
                responses_final_sse("done"),
            ],
            ApiProtocol::Completions => vec![
                sse_response(chat_tools.into()),
                sse_response("data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"done\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n".into()),
            ],
        };
        let (base_url, requests, server) = spawn_chat_completion_server(bodies).await;
        let mut agent = checkpoint_stream_agent(base_url, protocol);
        agent.set_model_catalog(HashMap::from([(
            "m1".into(),
            ModelRequestMetadata {
                context_window: Some(32_000),
                max_output_tokens: Some(128),
                supports_tools: true,
                ..Default::default()
            },
        )]));
        agent.set_compaction_config(CompactionConfig {
            protected_reserve_tokens: Some(4_000),
            ..Default::default()
        });
        agent.set_logical_checkpoint_config(LogicalCheckpointConfig {
            enabled: true,
            automatic: true,
            max_automatic_per_turn: 1,
            ..Default::default()
        });
        let candidate = prepared_checkpoint_for(&agent);
        let candidates = Arc::new(AtomicUsize::new(0));
        agent.set_logical_checkpoint_candidate_provider({
            let candidates = Arc::clone(&candidates);
            Arc::new(move || {
                candidates.fetch_add(1, Ordering::SeqCst);
                Ok(candidate.clone())
            })
        });
        let executions = Arc::new(AtomicUsize::new(0));
        agent.register_tool(ProtectedOverflowTool(Arc::clone(&executions)));
        let mut checkpoints = 0;
        let mut prepared = 0;
        let result = agent
            .run_stream_async(
                "continue",
                |_| std::future::ready(Ok(())),
                |event| {
                    match event {
                        AgentEvent::LogicalCheckpoint { .. } => checkpoints += 1,
                        AgentEvent::LlmRequestTelemetry(_) => prepared += 1,
                        _ => {}
                    }
                    std::future::ready(Ok(()))
                },
                |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
            )
            .await
            .expect("automatic checkpoint rebuilds the protected overflow");

        assert_eq!(result, "done");
        assert_eq!(
            requests.load(Ordering::SeqCst),
            2,
            "one request per iteration"
        );
        assert_eq!(
            executions.load(Ordering::SeqCst),
            1,
            "the batch never replays"
        );
        assert_eq!(candidates.load(Ordering::SeqCst), 1);
        assert_eq!(checkpoints, 1, "one automatic boundary event");
        assert_eq!(
            prepared, 4,
            "only the two provider requests emit prepared and terminal telemetry"
        );
        assert_eq!(agent.runtime_snapshot.current_segment_id, Some(2));
        server.await.expect("server task should finish");
    }

    #[tokio::test]
    async fn phase3c_actual_responses_and_chat_hard_overflow_rebuild_once_without_replay() {
        for protocol in [ApiProtocol::Responses, ApiProtocol::Completions] {
            assert_automatic_hard_overflow_checkpoint(protocol).await;
        }
    }

    #[tokio::test]
    async fn phase3c_actual_two_automatic_boundaries_rearm_below_low_and_commit_successors() {
        let first_batch = vec![json!({
            "type": "function_call", "id": "f1", "call_id": "one",
            "name": "test__soft_pressure", "arguments": "{}", "status": "completed"
        })];
        let second_batch = vec![json!({
            "type": "function_call", "id": "f2", "call_id": "two",
            "name": "test__soft_pressure", "arguments": "{}", "status": "completed"
        })];
        let (base_url, requests, server) = spawn_chat_completion_server(vec![
            responses_tool_batch_sse(first_batch),
            responses_tool_batch_sse(second_batch),
            responses_final_sse("done"),
        ])
        .await;
        let mut agent = checkpoint_stream_agent(base_url, ApiProtocol::Responses);
        agent.set_model_catalog(HashMap::from([(
            "m1".into(),
            ModelRequestMetadata {
                context_window: Some(32_000),
                max_output_tokens: Some(128),
                supports_tools: true,
                ..Default::default()
            },
        )]));
        agent.set_compaction_config(CompactionConfig {
            protected_reserve_tokens: Some(4_000),
            ..Default::default()
        });
        agent.set_logical_checkpoint_config(LogicalCheckpointConfig {
            enabled: true,
            automatic: true,
            trigger_reserve_percent: 50,
            max_automatic_per_turn: 2,
        });
        let first = prepared_checkpoint_for_lineage(&agent, "automatic-1", None);
        let mut successor = checkpoint_stream_agent("http://unused".into(), ApiProtocol::Responses);
        successor.runtime_snapshot.current_segment_id = Some(2);
        successor.runtime_snapshot.leaf_sequence = Some(10);
        let second =
            prepared_checkpoint_for_lineage(&successor, "automatic-2", Some("automatic-1"));
        let candidates = Arc::new(vec![first, second]);
        let candidate_index = Arc::new(AtomicUsize::new(0));
        agent.set_logical_checkpoint_candidate_provider({
            let candidates = Arc::clone(&candidates);
            let candidate_index = Arc::clone(&candidate_index);
            Arc::new(move || {
                candidates
                    .get(candidate_index.fetch_add(1, Ordering::SeqCst))
                    .cloned()
                    .ok_or_else(|| anyhow!("unexpected third automatic checkpoint"))
            })
        });
        let executions = Arc::new(AtomicUsize::new(0));
        agent.register_tool(SoftPressureTool(Arc::clone(&executions)));
        let mut checkpoints = Vec::new();
        let mut request_telemetry = 0;
        let result = agent
            .run_stream_async(
                "continue",
                |_| std::future::ready(Ok(())),
                |event| {
                    match event {
                        AgentEvent::LogicalCheckpoint { event, .. } => checkpoints.push((
                            event.previous_segment_id,
                            event.segment_id,
                            event.previous_checkpoint_id,
                        )),
                        AgentEvent::LlmRequestTelemetry(_) => request_telemetry += 1,
                        _ => {}
                    }
                    std::future::ready(Ok(()))
                },
                |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
            )
            .await
            .expect("each boundary automatically commits its rearmed successor");

        assert_eq!(result, "done");
        assert_eq!(requests.load(Ordering::SeqCst), 3);
        assert_eq!(executions.load(Ordering::SeqCst), 2);
        assert_eq!(candidate_index.load(Ordering::SeqCst), 2);
        assert_eq!(
            checkpoints,
            vec![(1, 2, None), (2, 3, Some("automatic-1".into()))]
        );
        assert_eq!(
            request_telemetry, 6,
            "only three provider requests are observable"
        );
        assert_eq!(agent.runtime_snapshot.current_segment_id, Some(3));
        server.await.expect("server task should finish");
    }

    #[tokio::test]
    async fn phase3c_actual_automatic_checkpoint_cancellation_never_installs_a_successor() {
        let (base_url, requests, server) = spawn_chat_completion_server(vec![
            responses_tool_batch_sse(vec![json!({
                "type": "function_call", "id": "f1", "call_id": "one",
                "name": "test__soft_pressure", "arguments": "{}", "status": "completed"
            })]),
            responses_final_sse("clean"),
        ])
        .await;
        let mut agent = checkpoint_stream_agent(base_url, ApiProtocol::Responses);
        agent.set_model_catalog(HashMap::from([(
            "m1".into(),
            ModelRequestMetadata {
                context_window: Some(32_000),
                max_output_tokens: Some(128),
                supports_tools: true,
                ..Default::default()
            },
        )]));
        agent.set_compaction_config(CompactionConfig {
            protected_reserve_tokens: Some(4_000),
            ..Default::default()
        });
        agent.set_logical_checkpoint_config(LogicalCheckpointConfig {
            enabled: true,
            automatic: true,
            trigger_reserve_percent: 50,
            max_automatic_per_turn: 1,
        });
        agent.register_tool(SoftPressureTool(Arc::new(AtomicUsize::new(0))));
        let entered = Arc::new(std::sync::atomic::AtomicBool::new(false));
        {
            let entered_event = Arc::clone(&entered);
            let stream = agent.run_stream_async(
                "cancel automatic checkpoint",
                |_| std::future::ready(Ok(())),
                move |event| {
                    let entered = Arc::clone(&entered_event);
                    async move {
                        if matches!(event, AgentEvent::LogicalCheckpoint { .. }) {
                            entered.store(true, Ordering::SeqCst);
                            std::future::pending::<()>().await;
                        }
                        Ok(())
                    }
                },
                |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
            );
            tokio::pin!(stream);
            for _ in 0..50 {
                if entered.load(Ordering::SeqCst) {
                    break;
                }
                tokio::select! {
                    _ = &mut stream => panic!("stream completed before automatic callback cancellation"),
                    _ = sleep(Duration::from_millis(10)) => {}
                }
            }
            assert!(
                entered.load(Ordering::SeqCst),
                "automatic checkpoint callback was not reached"
            );
        }
        assert_eq!(agent.runtime_snapshot.current_segment_id, Some(1));

        let result = agent
            .run_stream_async(
                "clean next turn",
                |_| std::future::ready(Ok(())),
                |_| std::future::ready(Ok(())),
                |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
            )
            .await
            .expect("the clean run must not inherit a cancelled automatic checkpoint");
        assert_eq!(result, "clean");
        assert_eq!(requests.load(Ordering::SeqCst), 2);
        assert_eq!(
            agent.turn.automatic_checkpoint,
            AutomaticCheckpointSchedulerState::default()
        );
        server.await.expect("server task should finish");
    }

    #[test]
    fn completed_tool_output_projection_and_restore_never_reexecutes_handler() {
        let records = vec![
            TranscriptRecord {
                session_id: "s".into(),
                sequence: 1,
                timestamp_ms: 0,
                context_branch_id: None,
                event: TranscriptEvent::UserMessage {
                    content: crate::user_content::UserMessageContent::from("resume"),
                },
            },
            TranscriptRecord {
                session_id: "s".into(),
                sequence: 2,
                timestamp_ms: 0,
                context_branch_id: None,
                event: TranscriptEvent::AssistantToolCallBatch {
                    text: None,
                    calls: vec![HistoryToolCall {
                        call_id: "finished".into(),
                        name: "test__replay_guard".into(),
                        arguments_json: "{}".into(),
                    }],
                },
            },
            TranscriptRecord {
                session_id: "s".into(),
                sequence: 3,
                timestamp_ms: 0,
                context_branch_id: None,
                event: TranscriptEvent::ToolCallFinished {
                    call_id: "finished".into(),
                    name: "test__replay_guard".into(),
                    ok: true,
                    output: ToolResult::ok("test__replay_guard", json!({"persisted": true})),
                },
            },
        ];
        let projected = crate::transcript::transcript_projection::project_runtime_restore_snapshot(
            "s".into(),
            records,
            crate::transcript::transcript_projection::SessionContextCursor {
                branch_id: None,
                leaf_sequence: None,
            },
            &[],
        )
        .expect("project completed output");
        let executions = Arc::new(AtomicUsize::new(0));
        let mut agent = test_agent();
        agent.register_tool(ReplayGuardTool(executions.clone()));
        agent
            .restore_runtime_snapshot(projected.protocol_frames, projected.snapshot)
            .expect("restore persisted output");
        assert_eq!(executions.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn context_checkpoint_cannot_nest_inside_active_experiment() {
        let mut agent = test_agent();
        agent.set_context_scope_state(Arc::new(std::sync::Mutex::new(ContextScopeState {
            active_experiment: Some(ActiveContextExperiment {
                branch_id: "branch-1".into(),
                parent_branch_id: ROOT_CONTEXT_BRANCH_ID.into(),
                base_sequence: 4,
                writes_observed: false,
            }),
        })));

        let error = agent
            .validate_context_control_tool(tool_names::TOOL_CONTEXT_CHECKPOINT)
            .expect_err("nested checkpoint should fail");

        assert!(
            error
                .to_string()
                .contains("cannot start a nested experiment")
        );
    }

    #[test]
    fn context_return_requires_active_experiment() {
        let agent = test_agent();
        let error = agent
            .validate_context_control_tool(tool_names::TOOL_CONTEXT_RETURN)
            .expect_err("return without active experiment should fail");

        assert!(
            error
                .to_string()
                .contains("requires an active context experiment")
        );
    }

    #[tokio::test]
    async fn active_context_experiment_blocks_normal_turn_finalization() {
        let mut agent = test_agent();
        agent.set_context_scope_state(Arc::new(std::sync::Mutex::new(ContextScopeState {
            active_experiment: Some(ActiveContextExperiment {
                branch_id: "branch-1".into(),
                parent_branch_id: ROOT_CONTEXT_BRANCH_ID.into(),
                base_sequence: 4,
                writes_observed: false,
            }),
        })));

        let mut events = Vec::new();
        let error = agent
            .continue_or_finalize_no_tool_reply(
                &mut |event| {
                    events.push(event);
                    std::future::ready(Ok(()))
                },
                0,
                &mut 0,
            )
            .await
            .expect_err("active experiment should fail closed");

        assert!(
            error
                .to_string()
                .contains("cannot finalize turn while context experiment 'branch-1' is active")
        );
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, AgentEvent::TurnFinalized(_)))
        );
    }

    #[tokio::test]
    async fn non_shell_tool_timeout_emits_cancelled_and_timed_out_terminal_events() {
        let mut agent = test_agent();
        agent.set_tool_timeout_secs(Some(1));
        agent
            .try_register_tool(SleepTool)
            .expect("register sleep tool");

        let call = test_tool_call("test__sleep", "{}");
        let mut events = Vec::new();
        let record = agent
            .execute_tool_call(
                &call,
                &mut |event| {
                    events.push(event);
                    std::future::ready(Ok(()))
                },
                &mut |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
            )
            .await
            .expect("tool call should return timeout record");

        assert_eq!(record.status, ToolExecutionStatus::TimedOut);
        assert!(!record.output.ok);
        assert_eq!(
            record
                .output
                .data
                .as_ref()
                .and_then(|data| data.get("status"))
                .and_then(Value::as_str),
            Some("timed_out")
        );
        assert!(events.iter().any(|event| matches!(
            event,
            AgentEvent::ToolCallCancelled { call_id, name }
                if call_id == "call-test__sleep" && name == "test__sleep"
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            AgentEvent::ToolCallFinished { ok, output, .. }
                if !ok
                    && output
                        .data
                        .as_ref()
                        .and_then(|data| data.get("status"))
                        .and_then(Value::as_str)
                        == Some("timed_out")
        )));
    }

    #[test]
    fn write_effects_mark_active_context_experiment_before_transcript_audit_replay() {
        let mut agent = test_agent();
        let scope = ActiveContextExperiment {
            branch_id: "branch-1".into(),
            parent_branch_id: ROOT_CONTEXT_BRANCH_ID.into(),
            base_sequence: 4,
            writes_observed: false,
        };
        agent.set_context_scope_state(Arc::new(std::sync::Mutex::new(ContextScopeState {
            active_experiment: Some(scope.clone()),
        })));
        agent.set_runtime_snapshot_provider(Arc::new(|| Ok(RuntimeSnapshot::new("branch-1"))));
        agent.set_context_experiment_restore_point(
            scope,
            Vec::new(),
            RuntimeSnapshot::new(ROOT_CONTEXT_BRANCH_ID),
        );

        let mut record = test_execution_record(
            "fs__write",
            ToolResult::ok("fs__write", json!({"path": "src/lib.rs"})),
        );
        record.effects.kind = ToolEffectKind::Write;
        record.effects.primary_path = Some("src/lib.rs".into());

        agent.record_tool_effects(&record);

        assert!(
            agent
                .context_scope_state
                .lock()
                .expect("scope state lock")
                .active_experiment
                .as_ref()
                .is_some_and(|experiment| experiment.writes_observed)
        );
        assert!(
            agent
                .context_experiment_restore_point
                .as_ref()
                .is_some_and(|restore| restore.scope.writes_observed)
        );
    }

    #[test]
    fn evidence_ids_remain_unique_after_restoring_older_evidence_snapshot() {
        let mut agent = test_agent();
        let first = agent
            .remember_tool_evidence(&test_execution_record(
                "fs__read",
                ToolResult::ok("fs__read", json!({"content": "one"})),
            ))
            .expect("first evidence");
        let older_snapshot = agent.runtime_snapshot.evidence.clone();

        let second = agent
            .remember_tool_evidence(&test_execution_record(
                "fs__read",
                ToolResult::ok("fs__read", json!({"content": "two"})),
            ))
            .expect("second evidence");
        assert_ne!(first.id, second.id);

        agent
            .restore_session_history(agent.history.clone(), older_snapshot, agent.next_turn_id)
            .expect("restore older evidence snapshot");

        let third = agent
            .remember_tool_evidence(&test_execution_record(
                "fs__read",
                ToolResult::ok("fs__read", json!({"content": "three"})),
            ))
            .expect("third evidence after restore");

        assert_ne!(first.id, third.id);
        assert_ne!(second.id, third.id);
    }

    fn large_tool_output_json(field: &str) -> String {
        json!({field: "line ".repeat((COMPACTION_TOOL_OUTPUT_CHAR_CAP + 500) / 5)}).to_string()
    }

    fn prunable_tool_output_json(field: &str) -> String {
        json!({field: "line ".repeat((COMPACTION_PRUNE_MIN_OUTPUT_CHARS + 1_000) / 5)}).to_string()
    }

    fn prune_protect_padding() -> String {
        "padding ".repeat(18_000)
    }

    struct StaticSubagentDelegate {
        result: ToolResult,
    }

    struct CapturingSubagentDelegate {
        result: ToolResult,
        explorer_tasks: Arc<std::sync::Mutex<Vec<String>>>,
        fixer_tasks: Arc<std::sync::Mutex<Vec<String>>>,
    }

    impl SubagentDelegate<OpenAIConfig> for StaticSubagentDelegate {
        fn run_named<'a>(
            &'a self,
            _parent: &'a Agent<OpenAIConfig>,
            _agent_name: &'a str,
            _invocation: SubagentInvocation,
        ) -> Pin<Box<dyn Future<Output = Result<ToolResult>> + Send + 'a>> {
            let result = self.result.clone();
            Box::pin(async move { Ok(result) })
        }
    }

    impl SubagentDelegate<OpenAIConfig> for CapturingSubagentDelegate {
        fn run_named<'a>(
            &'a self,
            _parent: &'a Agent<OpenAIConfig>,
            agent_name: &'a str,
            invocation: SubagentInvocation,
        ) -> Pin<Box<dyn Future<Output = Result<ToolResult>> + Send + 'a>> {
            match agent_name {
                "fixer" => self
                    .fixer_tasks
                    .lock()
                    .expect("fixer capture lock")
                    .push(invocation.prompt),
                _ => self
                    .explorer_tasks
                    .lock()
                    .expect("explorer capture lock")
                    .push(invocation.prompt),
            }
            let result = self.result.clone();
            Box::pin(async move { Ok(result) })
        }
    }

    fn static_delegate(result: ToolResult) -> Arc<dyn SubagentDelegate<OpenAIConfig>> {
        Arc::new(StaticSubagentDelegate { result })
    }

    fn capturing_delegate(
        result: ToolResult,
    ) -> (
        Arc<dyn SubagentDelegate<OpenAIConfig>>,
        Arc<std::sync::Mutex<Vec<String>>>,
        Arc<std::sync::Mutex<Vec<String>>>,
    ) {
        let explorer_tasks = Arc::new(std::sync::Mutex::new(Vec::new()));
        let fixer_tasks = Arc::new(std::sync::Mutex::new(Vec::new()));
        (
            Arc::new(CapturingSubagentDelegate {
                result,
                explorer_tasks: Arc::clone(&explorer_tasks),
                fixer_tasks: Arc::clone(&fixer_tasks),
            }),
            explorer_tasks,
            fixer_tasks,
        )
    }

    #[test]
    fn tool_effects_classify_read_write_validation_command_diagnostic_and_workflow_control() {
        let read = ToolEffects::derive(
            "fs__read",
            Some(&json!({"path": "src/lib.rs"})),
            &ToolResult::ok(
                "fs__read",
                json!({"path": "src/lib.rs", "content": "fn main() {}"}),
            ),
        );
        assert_eq!(read.kind, ToolEffectKind::Read);
        assert_eq!(read.primary_path.as_deref(), Some("src/lib.rs"));
        assert!(read.edited_paths.is_empty());
        assert_eq!(read.command, None);

        let write = ToolEffects::derive(
            "edit__apply_patch",
            None,
            &ToolResult::ok(
                "edit__apply_patch",
                json!({"edits": [{"path": "src/lib.rs"}, {"path": "src/agent.rs"}]}),
            ),
        );
        assert_eq!(write.kind, ToolEffectKind::Write);
        assert_eq!(write.edited_paths, vec!["src/lib.rs", "src/agent.rs"]);

        let validation = ToolEffects::derive(
            "shell__exec",
            Some(&json!({"command": "cargo test transcript"})),
            &ToolResult::ok(
                "shell__exec",
                json!({"command": "cargo test transcript", "status": 0}),
            ),
        );
        assert_eq!(validation.kind, ToolEffectKind::Validation);
        assert_eq!(validation.command.as_deref(), Some("cargo test transcript"));

        let failed_validation = ToolEffects::derive(
            "shell__exec",
            Some(&json!({"command": "cargo test transcript"})),
            &ToolResult::ok(
                "shell__exec",
                json!({"command": "cargo test transcript", "status": 101, "success": false}),
            ),
        );
        assert_eq!(failed_validation.kind, ToolEffectKind::Diagnostic);
        assert_eq!(
            failed_validation.command.as_deref(),
            Some("cargo test transcript")
        );

        let contradictory_failed_validation = ToolEffects::derive(
            "shell__exec",
            Some(&json!({"command": "cargo test transcript"})),
            &ToolResult::ok(
                "shell__exec",
                json!({"command": "cargo test transcript", "status": 101, "success": true}),
            ),
        );
        assert_eq!(
            contradictory_failed_validation.kind,
            ToolEffectKind::Diagnostic
        );

        let checkout = ToolEffects::derive(
            "shell__exec",
            Some(&json!({"command": "git checkout main"})),
            &ToolResult::ok(
                "shell__exec",
                json!({"command": "git checkout main", "status": 0, "success": true}),
            ),
        );
        assert_eq!(checkout.kind, ToolEffectKind::Command);

        let command = ToolEffects::derive(
            "shell__exec",
            Some(&json!({"command": "ls src"})),
            &ToolResult::ok("shell__exec", json!({"command": "ls src", "status": 0})),
        );
        assert_eq!(command.kind, ToolEffectKind::Command);
        assert_eq!(command.command.as_deref(), Some("ls src"));

        let diagnostic = ToolEffects::derive(
            "shell__exec",
            Some(&json!({"command": "cargo test agent::tests::tool"})),
            &ToolResult::err("shell__exec", "command failed"),
        );
        assert_eq!(diagnostic.kind, ToolEffectKind::Diagnostic);
        assert_eq!(
            diagnostic.command.as_deref(),
            Some("cargo test agent::tests::tool")
        );

        let workflow = ToolEffects::derive(
            "workflow__todos",
            Some(&json!({"items": [{"id": "t1", "content": "x", "status": "pending"}]})),
            &ToolResult::ok("workflow__todos", json!({"ok": true})),
        );
        assert_eq!(workflow.kind, ToolEffectKind::WorkflowControl);
    }

    #[test]
    fn agent_tool_definitions_hide_subagent_tools_until_delegate_is_installed() {
        let mut agent = test_agent();
        let specs = agent.tool_definitions();
        assert!(
            specs
                .iter()
                .any(|spec| spec.name == tool_names::TOOL_AGENT_RECONCILE)
        );
        for name in [
            "agent__explore",
            "agent__fixer",
            "agent__oracle",
            "agent__designer",
            "agent__librarian",
            "agent__general",
        ] {
            assert!(
                !specs.iter().any(|spec| spec.name == name),
                "{name} should be hidden"
            );
        }

        agent.set_subagent_delegate(static_delegate(ToolResult::ok(
            "agent__explore",
            json!({
                "run_id": "run-1",
                "child_session_id": "child-session",
                "agent_name": "explorer",
                "status": "completed",
                "summary": "done",
            }),
        )));

        let specs = agent.tool_definitions();
        for name in [
            "agent__explore",
            "agent__fixer",
            "agent__oracle",
            "agent__designer",
            "agent__librarian",
            "agent__general",
        ] {
            assert!(
                specs.iter().any(|spec| spec.name == name),
                "{name} should be exposed"
            );
        }
    }

    #[test]
    fn agent_templates_expose_capability_contracts() {
        let explorer = AgentTemplate::explorer().capability_contract();
        assert_eq!(explorer.name, "explorer");
        assert_eq!(explorer.tool_scope, ToolScope::ReadOnlyExplorer);
        assert_eq!(explorer.permission_mode, PermissionMode::Default);
        assert!(!explorer.can_write);
        assert!(!explorer.can_delegate);
        assert_eq!(explorer.default_max_tool_calls, None);
        assert!(explorer.input_expectations.contains("task 或 objective"));
        assert!(explorer.expected_result_shape.contains("run_id"));

        let fixer = AgentTemplate::fixer().capability_contract();
        assert_eq!(fixer.name, "fixer");
        assert_eq!(fixer.tool_scope, ToolScope::FullAccess);
        assert!(fixer.can_write);
        assert!(!fixer.can_delegate);
        assert_eq!(fixer.default_max_tool_calls, None);

        let readonly_names = ["oracle", "designer", "librarian", "general"];
        for name in readonly_names {
            let template = AgentTemplate::from_name(name).expect("known template");
            let contract = template.capability_contract();
            assert_eq!(contract.name, name);
            assert_eq!(contract.tool_scope, ToolScope::ReadOnlyExplorer);
            assert!(!contract.can_write);
            assert!(!contract.can_delegate);
        }
    }

    #[test]
    fn child_agent_applies_runtime_max_tool_call_override() {
        let agent = test_agent();
        let child = AgentFactory::create_child_with_max_tool_calls(
            &agent,
            &AgentTemplate::fixer(),
            Some(1),
        );

        assert_eq!(child.max_tool_calls_limit(), Some(1));
        let error = child
            .ensure_tool_call_budget(0, 2)
            .expect_err("tool-call budget should be enforced");
        assert!(error.to_string().contains("too many tool calls"));
    }

    #[test]
    fn child_agent_retains_explicit_tool_call_override_with_unbounded_parent() {
        let client = Client::with_config(
            OpenAIConfig::new()
                .with_api_base("https://api.openai.com/v1")
                .with_api_key("test"),
        );
        let agent = Agent::new(client, "m1", None, None);
        let child = AgentFactory::create_child_with_max_tool_calls(
            &agent,
            &AgentTemplate::fixer(),
            Some(2),
        );

        assert_eq!(child.max_tool_calls_limit(), Some(2));
        child
            .ensure_tool_call_budget(0, 2)
            .expect("budget edge should pass");
        let error = child
            .ensure_tool_call_budget(0, 3)
            .expect_err("explicit child budget should still be enforced");
        assert!(error.to_string().contains("max 2"));
    }

    #[test]
    fn child_agent_inherits_parent_tool_call_limit_without_template_budget() {
        let agent = test_agent();
        let child = AgentFactory::create_child(&agent, &AgentTemplate::fixer());

        assert_eq!(child.max_tool_calls_limit(), agent.max_tool_calls_limit());
    }

    #[tokio::test]
    async fn subagent_tool_execution_normalizes_bounded_input_before_delegation() {
        let mut agent = test_agent();
        let (delegate, explorer_tasks, fixer_tasks) = capturing_delegate(ToolResult::ok(
            "agent__fixer",
            json!({
                "run_id": "run-1",
                "child_session_id": "child-session",
                "agent_name": "fixer",
                "status": "completed",
                "summary": "done"
            }),
        ));
        agent.set_subagent_delegate(delegate);

        let output = agent
            .execute_subagent_tool(
                "agent__fixer",
                &json!({
                    "objective": "Implement contract",
                    "success_criteria": ["tests pass"],
                    "allowed_paths": ["src/agent.rs"],
                    "owned_paths": ["src/tool.rs"],
                    "timeout_secs": 30,
                    "max_tool_calls": 5
                }),
            )
            .await;

        assert!(output.ok);
        assert!(explorer_tasks.lock().expect("explorer tasks").is_empty());
        let fixer_tasks = fixer_tasks.lock().expect("fixer tasks");
        assert_eq!(fixer_tasks.len(), 1);
        let prompt = &fixer_tasks[0];
        assert!(prompt.contains("Objective: Implement contract"));
        assert!(prompt.contains("Success criteria:"));
        assert!(prompt.contains("Allowed paths: src/agent.rs"));
        assert!(prompt.contains("Owned paths: src/tool.rs"));
        assert!(prompt.contains("Execution bounds: timeout_secs=30, max_tool_calls=5"));
        assert!(prompt.contains("do not recursively delegate"));
    }

    #[tokio::test]
    async fn subagent_tool_execution_supports_legacy_task_only_input() {
        let mut agent = test_agent();
        let (delegate, explorer_tasks, _fixer_tasks) = capturing_delegate(ToolResult::ok(
            "agent__explore",
            json!({
                "run_id": "run-1",
                "child_session_id": "child-session",
                "agent_name": "explorer",
                "status": "completed",
                "summary": "done"
            }),
        ));
        agent.set_subagent_delegate(delegate);

        let output = agent
            .execute_subagent_tool(
                "agent__explore",
                &json!({"task": "inspect src/subagent.rs"}),
            )
            .await;

        assert!(output.ok);
        let explorer_tasks = explorer_tasks.lock().expect("explorer tasks");
        assert_eq!(explorer_tasks.len(), 1);
        assert!(explorer_tasks[0].contains("Objective: inspect src/subagent.rs"));
        assert!(explorer_tasks[0].contains("Mode: read-only exploration only."));
    }

    #[tokio::test]
    async fn readonly_expert_subagent_tool_execution_routes_through_generic_delegate() {
        let mut agent = test_agent();
        let (delegate, explorer_tasks, fixer_tasks) = capturing_delegate(ToolResult::ok(
            "agent__oracle",
            json!({
                "run_id": "run-1",
                "child_session_id": "child-session",
                "agent_name": "oracle",
                "status": "completed",
                "summary": "done"
            }),
        ));
        agent.set_subagent_delegate(delegate);

        let output = agent
            .execute_subagent_tool("agent__oracle", &json!({"task": "analyze failure mode"}))
            .await;

        assert!(output.ok);
        assert!(fixer_tasks.lock().expect("fixer tasks").is_empty());
        let readonly_tasks = explorer_tasks.lock().expect("readonly tasks");
        assert_eq!(readonly_tasks.len(), 1);
        assert!(readonly_tasks[0].contains("Objective: analyze failure mode"));
    }

    #[tokio::test]
    async fn subagent_tool_execution_returns_validation_error_for_missing_objective() {
        let agent = test_agent();

        let output = agent
            .execute_subagent_tool("agent__explore", &json!({}))
            .await;

        assert!(!output.ok);
        assert!(
            output
                .error
                .as_ref()
                .expect("validation error")
                .message
                .contains("requires a non-empty 'task' or 'objective'")
        );
    }

    #[test]
    fn remembered_subagent_evidence_carries_parent_turn_provenance() {
        let mut agent = test_agent();
        agent.turn.turn_id = 42;
        let record = ToolExecutionRecord {
            call_id: "call-agent__explore".into(),
            tool_name: "agent__explore".into(),
            arguments: Some(json!({"task": "inspect"})),
            permission_class: crate::permission::ToolPermissionClass::Preview,
            directive: crate::permission::ExecutionDirective::None,
            status: ToolExecutionStatus::Executed,
            rejection: None,
            output: ToolResult::ok(
                "agent__explore",
                json!({
                    "run_id": "run-1",
                    "child_session_id": "child-1",
                    "status": "completed",
                    "summary": "done"
                }),
            ),
            effects: ToolEffects {
                kind: ToolEffectKind::Read,
                primary_path: None,
                edited_paths: vec![],
                command: None,
            },
        };

        let evidence = agent
            .remember_tool_evidence(&record)
            .expect("subagent evidence should be recorded");

        match evidence.source {
            EvidenceSource::Subagent {
                run_id,
                child_session_id,
                parent_tool,
                parent_turn_id,
                ..
            } => {
                assert_eq!(run_id, "run-1");
                assert_eq!(child_session_id, "child-1");
                assert_eq!(parent_tool, "agent__explore");
                assert_eq!(parent_turn_id.as_deref(), Some("turn-42"));
            }
            other => panic!("unexpected evidence source: {other:?}"),
        }
    }

    #[tokio::test]
    async fn cancelled_agent_explore_records_tool_output_before_interrupting_turn() {
        let mut agent = test_agent();
        agent.set_subagent_delegate(static_delegate(ToolResult::err_with_data(
            "agent__explore",
            "explorer cancelled",
            json!({
                "run_id": "run-1",
                "child_session_id": "child-session",
                "agent_name": "explorer",
                "status": "cancelled",
                "summary": "explorer cancelled",
            }),
        )));
        let call = test_tool_call("agent__explore", r#"{"task":"inspect"}"#);
        agent
            .append_assistant_tool_calls("", std::slice::from_ref(&call))
            .expect("assistant tool calls should append");
        let mut events = Vec::new();

        let error = agent
            .execute_tool_call_and_record(
                &call,
                &mut |event| {
                    events.push(event);
                    std::future::ready(Ok(()))
                },
                &mut |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
            )
            .await
            .expect_err("cancelled explorer interrupts the turn after recording output");

        assert!(error.to_string().contains("agent__explore cancelled"));
        assert!(matches!(
            agent.history.last(),
            Some(HistoryItem::ToolOutput {
                call_id,
                output_json,
            }) if call_id == "call-agent__explore"
                && output_json.contains("cancelled")
                && output_json.contains("child-session")
        ));
        assert!(events.iter().any(|event| matches!(
            event,
            AgentEvent::ToolCallFinished {
                name,
                ok: false,
                output,
                ..
            } if name == "agent__explore"
                && output
                    .data
                    .as_ref()
                    .and_then(|data| data.get("status"))
                    .and_then(Value::as_str)
                    == Some("cancelled")
        )));
    }

    #[tokio::test]
    async fn cancelled_agent_fixer_records_tool_output_before_interrupting_turn() {
        let mut agent = test_agent();
        agent.set_subagent_delegate(static_delegate(ToolResult::err_with_data(
            "agent__fixer",
            "fixer cancelled",
            json!({
                "run_id": "run-1",
                "child_session_id": "child-session",
                "agent_name": "fixer",
                "status": "cancelled",
                "summary": "fixer cancelled",
            }),
        )));
        let call = test_tool_call("agent__fixer", r#"{"task":"apply requested fix"}"#);
        agent
            .append_assistant_tool_calls("", std::slice::from_ref(&call))
            .expect("assistant tool calls should append");
        let mut events = Vec::new();

        let error = agent
            .execute_tool_call_and_record(
                &call,
                &mut |event| {
                    events.push(event);
                    std::future::ready(Ok(()))
                },
                &mut |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
            )
            .await
            .expect_err("cancelled fixer interrupts the turn after recording output");

        assert!(error.to_string().contains("agent__fixer cancelled"));
        assert!(matches!(
            agent.history.last(),
            Some(HistoryItem::ToolOutput {
                call_id,
                output_json,
            }) if call_id == "call-agent__fixer"
                && output_json.contains("cancelled")
                && output_json.contains("child-session")
        ));
        assert!(events.iter().any(|event| matches!(
            event,
            AgentEvent::ToolCallFinished {
                name,
                ok: false,
                output,
                ..
            } if name == "agent__fixer"
                && output
                    .data
                    .as_ref()
                    .and_then(|data| data.get("status"))
                    .and_then(Value::as_str)
                    == Some("cancelled")
        )));
    }

    #[tokio::test]
    async fn delegated_structured_subagent_results_surface_in_next_turn_prelude() {
        let mut agent = test_agent();
        agent.prepare_turn_prelude("Delegate implementation work");
        agent.set_subagent_delegate(static_delegate(ToolResult::ok(
            "agent__fixer",
            json!({
                "run_id": "run-structured-1",
                "child_session_id": "child-structured-1",
                "agent_name": "fixer",
                "status": "completed",
                "summary": "implemented bounded fix",
                "structured_result": {
                    "status": "completed",
                    "summary": "implemented bounded fix",
                    "malformed": false,
                    "findings": [],
                    "files_read": ["src/agent.rs"],
                    "files_changed": ["src/agent.rs"],
                    "commands_run": ["cargo test subagent --quiet"],
                    "validation": ["cargo test subagent --quiet passed"],
                    "blockers": [],
                    "next_steps": ["reconcile in parent turn"],
                    "run_id": "run-structured-1",
                    "child_session_id": "child-structured-1"
                }
            }),
        )));

        let call = test_tool_call("agent__fixer", r#"{"task":"implement bounded fix"}"#);
        agent
            .append_assistant_tool_calls("", std::slice::from_ref(&call))
            .expect("assistant tool calls should append");
        let mut events = Vec::new();
        agent
            .execute_tool_call_and_record(
                &call,
                &mut |event| {
                    events.push(event);
                    std::future::ready(Ok(()))
                },
                &mut |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
            )
            .await
            .expect("subagent tool execution should succeed");

        assert!(events.iter().any(|event| matches!(
            event,
            AgentEvent::EvidenceRecorded(record)
                if record.tags.iter().any(|tag| tag == "subagent_result")
                    && record.tags.iter().any(|tag| tag == "unreconciled")
        )));

        let jobs = agent.pending_subagent_jobs();
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].agent_name, "fixer");
        assert_eq!(jobs[0].run_id, "run-structured-1");
        assert_eq!(jobs[0].child_session_id, "child-structured-1");
        assert_eq!(jobs[0].summary, "implemented bounded fix");

        let prelude = agent.prepare_turn_prelude("Reconcile child work");
        assert!(prelude.iter().any(|message| {
            message.text.contains("Pending child subagent results")
                && message.text.contains("run-structured-1")
                && message.text.contains("implemented bounded fix")
                && message.text.contains("child-structured-1")
        }));
    }

    #[test]
    fn model_switch_uses_new_metadata_for_next_request_build() {
        let client = Client::with_config(
            OpenAIConfig::new()
                .with_api_base("https://api.openai.com/v1")
                .with_api_key("test"),
        );
        let mut agent = Agent::new(client, "m1", 1, 1);

        let mut catalog = HashMap::new();
        catalog.insert(
            "m1".to_string(),
            ModelRequestMetadata {
                context_window: Some(2048),
                max_output_tokens: Some(256),
                supports_tools: true,
                supports_reasoning: false,
                ..Default::default()
            },
        );
        catalog.insert(
            "m2".to_string(),
            ModelRequestMetadata {
                context_window: Some(128_000),
                max_output_tokens: Some(256),
                supports_tools: true,
                supports_reasoning: false,
                ..Default::default()
            },
        );
        agent.set_model_catalog(catalog);

        // Simulate first user message.
        agent
            .append_history_item(HistoryItem::user("hello"))
            .expect("history append succeeds");
        let history = agent.history_items();
        let b1 = build_request_from_legacy(LegacyRequestBuilderInput {
            protocol: ApiProtocol::Responses,
            model_id: agent.model(),
            model: agent.active_model_metadata(),
            prelude: &agent.prelude,
            history: &history,
            protected_start_index: history.len().saturating_sub(1),
            tools: &[],
            evidence: &[],
            history_adapter: None,
            context_view: None,
        })
        .expect("request builds");
        assert_eq!(b1.budget.context_window_tokens, 2048.max(1024));

        // Switch model and build again.
        agent.set_model("m2");
        let history = agent.history_items();
        let b2 = build_request_from_legacy(LegacyRequestBuilderInput {
            protocol: ApiProtocol::Responses,
            model_id: agent.model(),
            model: agent.active_model_metadata(),
            prelude: &agent.prelude,
            history: &history,
            protected_start_index: history.len().saturating_sub(1),
            tools: &[],
            evidence: &[],
            history_adapter: None,
            context_view: None,
        })
        .expect("request builds");
        assert!(b2.budget.context_window_tokens > b1.budget.context_window_tokens);
    }

    #[test]
    fn inline_reasoning_extractor_splits_think_tags_from_visible_text() {
        let mut extractor = InlineReasoningExtractor::new("r-1");

        let mut parts = extractor.push("hello <thi");
        parts.extend(extractor.push("nk>plan</think> answer"));
        parts.extend(extractor.finish());

        assert_eq!(
            parts,
            vec![
                StreamTextPart::Visible("hello ".into()),
                StreamTextPart::ReasoningDelta {
                    item_id: "r-1".into(),
                    delta: "plan".into(),
                },
                StreamTextPart::ReasoningDone {
                    item_id: "r-1".into(),
                    text: "plan".into(),
                },
                StreamTextPart::Visible(" answer".into()),
            ]
        );
    }

    #[test]
    fn compatible_chat_delta_reads_native_reasoning_fields() {
        for (field, expected) in [
            ("reasoning_content", "plan"),
            ("reasoning", "think"),
            ("thinking", "ponder"),
        ] {
            let raw = serde_json::json!({
                "content": null,
                field: expected,
            });
            let delta: CompatibleChatCompletionStreamResponseDelta =
                serde_json::from_value(raw).expect("delta deserializes");

            assert_eq!(delta.reasoning_delta().as_deref(), Some(expected));
        }
    }

    #[test]
    fn protocol_frames_remain_authoritative_for_history_cache() {
        let mut agent = test_agent();
        agent
            .append_history_item(HistoryItem::user("hello"))
            .expect("user append succeeds");
        agent
            .append_history_item(HistoryItem::AssistantToolCalls {
                text: Some("working".into()),
                calls: vec![test_tool_call("fs__read", r#"{"path":"src/main.rs"}"#)],
            })
            .expect("tool call append succeeds");
        agent
            .append_history_item(HistoryItem::ToolOutput {
                call_id: "call-fs__read".into(),
                output_json: r#"{"ok":true}"#.into(),
            })
            .expect("tool output append succeeds");

        assert_eq!(
            crate::protocol_frames::history_items_from_frames(agent.protocol_frames_for_test()),
            agent.history_for_test()
        );
        assert_eq!(
            agent.runtime_snapshot.frames.len(),
            agent.protocol_frames_for_test().len()
        );
    }

    #[test]
    fn append_history_item_is_atomic_when_protocol_validation_fails() {
        let mut agent = test_agent();
        agent
            .append_history_item(HistoryItem::user("hello"))
            .expect("user append succeeds");

        let history_before = agent.history.clone();
        let frames_before = agent.protocol_frames.clone();
        let snapshot_before = agent.runtime_snapshot.clone();

        let error = agent
            .append_history_item(HistoryItem::ToolOutput {
                call_id: "call-orphan".into(),
                output_json: "{}".into(),
            })
            .expect_err("orphan tool output must fail");

        assert!(error.to_string().contains("orphan tool output"));
        assert_eq!(agent.history, history_before);
        assert_eq!(agent.protocol_frames, frames_before);
        assert_eq!(agent.runtime_snapshot, snapshot_before);
    }

    #[test]
    fn compatible_chat_delta_reads_object_and_array_reasoning() {
        let raw = serde_json::json!({
            "reasoning_content": [
                {"text": "step "},
                {"content": "one"}
            ]
        });
        let delta: CompatibleChatCompletionStreamResponseDelta =
            serde_json::from_value(raw).expect("delta deserializes");

        assert_eq!(delta.reasoning_delta().as_deref(), Some("step one"));
    }

    #[test]
    fn compatible_chat_stream_accepts_terminal_chunk_without_delta() {
        let raw = serde_json::json!({
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1780856440_u64,
            "model": "gpt-5.5",
            "choices": [{
                "index": 0,
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 3060,
                "completion_tokens": 25,
                "total_tokens": 3085
            }
        });

        let response: CompatibleChatCompletionStreamResponse =
            serde_json::from_value(raw).expect("terminal chunk deserializes");

        assert_eq!(response.choices.len(), 1);
        assert!(response.choices[0].delta.is_none());
        assert_eq!(response.choices[0].finish_reason, Some(FinishReason::Stop));
        assert_eq!(
            response.usage.as_ref().map(|usage| usage.prompt_tokens),
            Some(3060)
        );
        assert_eq!(
            response.usage.as_ref().map(|usage| usage.completion_tokens),
            Some(25)
        );
    }

    #[test]
    fn sse_parser_drains_data_events_and_done_marker() {
        let mut buffer = String::new();
        append_sse_chunk(&mut buffer, b"data: {\"choices\":[]}\n\ndata: [DONE]\n\n");

        assert_eq!(
            drain_sse_data_events(&mut buffer),
            vec![Some(r#"{"choices":[]}"#.into()), None]
        );
        assert!(buffer.is_empty());
    }

    #[test]
    fn ignores_non_terminal_lifecycle_events_missing_model() {
        for event_type in ["response.created", "response.in_progress"] {
            let raw = serde_json::json!({
                "type": event_type,
                "sequence_number": 1,
                "response": {
                    "id": "resp_test",
                    "object": "response",
                    "created_at": 1780765723_u64,
                    "status": "in_progress",
                    "background": false,
                    "error": null,
                    "output": []
                }
            });
            assert!(
                is_ignorable_response_lifecycle_event(&raw),
                "{event_type} should be ignored"
            );
        }
    }

    #[test]
    fn does_not_ignore_other_stream_deserialize_errors() {
        let raw = serde_json::json!({
            "type": "response.completed",
            "sequence_number": 1,
            "response": {
                "id": "resp_test",
                "object": "response",
                "created_at": 1780765723_u64,
                "status": "completed",
                "background": false,
                "error": null,
                "output": []
            }
        });
        assert!(!is_ignorable_response_lifecycle_event(&raw));
    }

    #[test]
    fn projects_provider_reasoning_efforts_without_mutating_completed_events() {
        for effort in ["max", "provider-ultra"] {
            let raw = serde_json::json!({
                "type": "response.completed", "sequence_number": 1,
                "response": {
                    "id": "resp_test", "object": "response", "created_at": 1780765723_u64,
                    "status": "completed", "background": false, "error": null,
                    "incomplete_details": null, "instructions": null, "max_output_tokens": null,
                    "model": "m1", "output": [], "parallel_tool_calls": true,
                    "previous_response_id": null, "reasoning": {"effort": effort}, "store": true,
                    "temperature": 1, "text": {"format": {"type": "text"}}, "tool_choice": "auto",
                    "tools": [], "top_p": 1, "truncation": "disabled",
                    "usage": {"input_tokens": 5, "input_tokens_details": {"cached_tokens": 0}, "output_tokens": 3, "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 8},
                    "user": null, "metadata": {}
                }
            });
            let event =
                project_response_stream_event(&raw).expect("completed event should project");
            let ResponseStreamEvent::ResponseCompleted(event) = event else {
                panic!("expected completion")
            };
            assert_eq!(event.response.id, "resp_test");
            assert!(event.response.output.is_empty());
            assert_eq!(event.response.usage.expect("usage").total_tokens, 8);
            assert_eq!(raw["response"]["reasoning"]["effort"], effort);
        }
    }

    #[test]
    fn rejects_malformed_completed_events_after_projection() {
        let raw = serde_json::json!({"type": "response.completed", "response": {"reasoning": {"effort": "max"}}});
        assert!(project_response_stream_event(&raw).is_err());
    }

    #[test]
    fn compact_indexed_chat_tool_calls_does_not_synthesize_missing_indices() {
        let mut indexed = BTreeMap::new();
        let mut call = ChatCompletionMessageToolCall::default();
        call.id = "call-1".into();
        call.function.name = "fs__write".into();
        call.function.arguments = r#"{"path":"a.txt","content":"ok"}"#.into();
        indexed.insert(1, call);

        let compacted = compact_indexed_chat_tool_calls(indexed);

        assert_eq!(compacted.len(), 1);
        assert_eq!(compacted[0].id, "call-1");
        assert_eq!(compacted[0].function.name, "fs__write");
        validate_chat_tool_calls(&compacted).expect("valid sparse-index tool call");
    }

    #[test]
    fn chat_tool_call_chunk_empty_name_does_not_overwrite_real_name() {
        let mut indexed = BTreeMap::new();
        for raw in [
            serde_json::json!({
                "index": 0,
                "id": "call-1",
                "type": "function",
                "function": {"name": "fs__write", "arguments": ""}
            }),
            serde_json::json!({
                "index": 0,
                "function": {"name": "", "arguments": "{\"path\":"}
            }),
            serde_json::json!({
                "index": 0,
                "function": {"name": "", "arguments": "\"a.txt\",\"content\":\"ok\"}"}
            }),
        ] {
            let chunk: ChatCompletionMessageToolCallChunk =
                serde_json::from_value(raw).expect("chunk deserializes");
            merge_chat_tool_call_chunk(&mut indexed, chunk);
        }

        let compacted = compact_indexed_chat_tool_calls(indexed);

        assert_eq!(compacted.len(), 1);
        assert_eq!(compacted[0].id, "call-1");
        assert_eq!(compacted[0].function.name, "fs__write");
        assert_eq!(
            compacted[0].function.arguments,
            r#"{"path":"a.txt","content":"ok"}"#
        );
        validate_chat_tool_calls(&compacted).expect("valid streamed tool call");
    }

    #[test]
    fn classifies_lightweight_and_engineering_turns() {
        assert_eq!(
            classify_turn_intent("Explain how Rust ownership works."),
            TurnIntent::Lightweight
        );
        assert_eq!(
            classify_turn_intent("Explain what this function does."),
            TurnIntent::Lightweight
        );
        assert_eq!(
            classify_turn_intent(
                "Fix the failing tests in src/agent.rs and update the implementation."
            ),
            TurnIntent::Engineering
        );
    }

    #[test]
    fn prepare_turn_prelude_assigns_incrementing_turn_ids() {
        let client = Client::with_config(
            OpenAIConfig::new()
                .with_api_base("https://api.openai.com/v1")
                .with_api_key("test"),
        );
        let mut agent = Agent::new(client, "m1", 1, 1);

        agent.prepare_turn_prelude("first turn");
        assert_eq!(agent.current_turn_id(), 1);

        agent.prepare_turn_prelude("second turn");
        assert_eq!(agent.current_turn_id(), 2);
    }

    #[test]
    fn restore_session_context_seeds_next_turn_id() {
        let mut agent = test_agent();

        agent
            .restore_session_context(Vec::new(), Vec::new(), 7)
            .expect("restore session context");
        agent.prepare_turn_prelude("resumed turn");

        assert_eq!(agent.current_turn_id(), 8);
    }

    #[test]
    fn candidate_session_usage_failure_leaves_live_agent_unchanged() {
        let mut agent = test_agent();
        agent
            .restore_session_history(vec![HistoryItem::user("old session")], Vec::new(), 7)
            .expect("restore old session");
        agent.prepare_turn_prelude("active turn");
        let mut invalid_metadata = ModelRequestMetadata::default();
        invalid_metadata.effective_input_limit_tokens = Some(0);
        agent.set_model_catalog(HashMap::from([(
            String::from("invalid-model"),
            invalid_metadata,
        )]));
        let target_snapshot = runtime_snapshot_for_history(
            ROOT_CONTEXT_BRANCH_ID,
            &[HistoryItem::user("target session")],
        );
        let model = agent.model.clone();
        let history = agent.history.clone();
        let protocol_frames = agent.protocol_frames.clone();
        let runtime_snapshot = agent.runtime_snapshot.clone();
        let turn_id = agent.current_turn_id();
        let next_turn_id = agent.next_turn_id;

        let error = agent
            .candidate_session_token_usage("invalid-model", &target_snapshot)
            .expect_err("invalid target metadata must fail");

        assert!(
            error
                .to_string()
                .contains("effective_input_limit_tokens must be greater than 0")
        );
        assert_eq!(agent.model, model);
        assert_eq!(agent.history, history);
        assert_eq!(agent.protocol_frames, protocol_frames);
        assert_eq!(agent.runtime_snapshot, runtime_snapshot);
        assert_eq!(agent.current_turn_id(), turn_id);
        assert_eq!(agent.next_turn_id, next_turn_id);
    }

    #[test]
    fn restore_runtime_snapshot_keeps_projected_runtime_state_authoritative() {
        let mut agent = test_agent();
        let history = vec![
            HistoryItem::user("resume question"),
            HistoryItem::assistant("resume answer"),
        ];
        let frames = crate::protocol_frames::history_items_to_frames(&history);
        let mut snapshot = RuntimeSnapshot::new("feature")
            .with_session_id("session-1")
            .with_latest_model("m1")
            .with_leaf_sequence(12)
            .with_current_turn_id(7);
        snapshot.frames = runtime_frames_for_history(&history);
        snapshot.set_evidence(vec![EvidenceRecord {
            id: "evidence-1".into(),
            sequence: 1,
            timestamp_ms: 1,
            evidence_kind: crate::evidence::EvidenceKind::Decision,
            title: "Restored evidence".into(),
            summary: "restored evidence".into(),
            detail: None,
            source: EvidenceSource::Transcript { sequence: 1 },
            tags: Vec::new(),
        }]);

        agent
            .restore_runtime_snapshot(frames.clone(), snapshot.clone())
            .expect("restore runtime snapshot");

        assert_eq!(
            agent.protocol_frames_for_test(),
            snapshot.active_protocol_frames().as_slice()
        );
        assert_eq!(
            agent.history_for_test(),
            crate::protocol_frames::history_items_from_frames(&frames).as_slice()
        );
        assert_eq!(agent.runtime_snapshot_for_test(), &snapshot);
        assert_eq!(agent.evidence(), snapshot.evidence.as_slice());
        agent.prepare_turn_prelude("continued turn");
        assert_eq!(agent.current_turn_id(), 8);
    }

    #[test]
    fn compatibility_rebuilds_preserve_restored_turn_id_without_an_active_turn() {
        let mut agent = test_agent();
        let snapshot = RuntimeSnapshot::new(ROOT_CONTEXT_BRANCH_ID).with_current_turn_id(7);
        agent
            .restore_runtime_snapshot(Vec::new(), snapshot)
            .expect("restore runtime snapshot");
        agent.compaction_config.prune = true;
        agent.compaction_config.tail_turns = 1;

        agent
            .replace_history(vec![
                HistoryItem::user("older turn"),
                HistoryItem::AssistantToolCalls {
                    text: None,
                    calls: vec![HistoryToolCall {
                        call_id: "call-read".into(),
                        name: "fs__read".into(),
                        arguments_json: r#"{"path":"src/lib.rs"}"#.into(),
                    }],
                },
                HistoryItem::ToolOutput {
                    call_id: "call-read".into(),
                    output_json: prunable_tool_output_json("stdout"),
                },
                HistoryItem::assistant(prune_protect_padding()),
                HistoryItem::user("recent turn"),
                HistoryItem::assistant("recent reply"),
                HistoryItem::user("current turn"),
            ])
            .expect("compatibility replacement succeeds");
        assert_eq!(agent.runtime_snapshot_for_test().current_turn_id, Some(7));

        agent
            .prune_old_tool_outputs(4_000)
            .expect("pruning succeeds");
        assert_eq!(agent.runtime_snapshot_for_test().current_turn_id, Some(7));

        agent
            .append_history_item(HistoryItem::context_summary("restored summary"))
            .expect("summary append succeeds");
        assert_eq!(agent.runtime_snapshot_for_test().current_turn_id, Some(7));
    }

    #[test]
    fn new_session_reset_discards_restored_runtime_metadata() {
        let mut agent = test_agent();
        let history = vec![HistoryItem::user("old prompt")];
        let mut snapshot = RuntimeSnapshot::new("feature")
            .with_session_id("old-session")
            .with_latest_model("old-model")
            .with_leaf_sequence(12)
            .with_current_turn_id(7);
        snapshot.frames = runtime_frames_for_history(&history);
        agent
            .restore_runtime_snapshot(
                crate::protocol_frames::history_items_to_frames(&history),
                snapshot,
            )
            .expect("restore runtime snapshot");

        agent.reset_for_new_session();

        assert!(agent.history_for_test().is_empty());
        assert!(agent.protocol_frames_for_test().is_empty());
        assert!(agent.evidence().is_empty());
        assert_eq!(
            agent.runtime_snapshot_for_test(),
            &RuntimeSnapshot::new(ROOT_CONTEXT_BRANCH_ID).with_latest_model("m1")
        );
        agent.prepare_turn_prelude("fresh turn");
        assert_eq!(agent.current_turn_id(), 1);
    }

    #[test]
    fn compaction_selection_preserves_recent_tail_and_reuses_previous_summary() {
        let history = vec![
            HistoryItem::context_summary("旧摘要"),
            HistoryItem::user("turn-1 user"),
            HistoryItem::assistant("turn-1 assistant"),
            HistoryItem::user("turn-2 user"),
            HistoryItem::assistant("turn-2 assistant"),
            HistoryItem::user("current user"),
        ];

        let selection = select_compaction_segments(
            &history,
            5,
            &CompactionConfig {
                tail_turns: 1,
                ..CompactionConfig::default()
            },
            4_000,
        )
        .expect("selection succeeds");

        assert_eq!(selection.previous_summary.as_deref(), Some("旧摘要"));
        assert_eq!(selection.head_for_summary.len(), 4);
        assert!(selection.tail_items.is_empty());
        assert_eq!(selection.tail_start_index, 5);
    }

    #[tokio::test]
    async fn manual_compaction_noops_when_history_is_empty() {
        let mut agent = test_agent();

        let outcome = agent
            .compact_session_async(|_| async { Ok(()) })
            .await
            .expect("manual compaction should not fail");

        assert_eq!(outcome, ManualCompactionOutcome::NothingToCompact);
    }

    #[tokio::test]
    async fn manual_compaction_noops_when_only_recent_tail_exists() {
        let mut agent = test_agent();
        agent
            .replace_history(vec![
                HistoryItem::user("short prompt"),
                HistoryItem::assistant("reply"),
            ])
            .expect("history replace succeeds");

        let outcome = agent
            .compact_session_async(|_| async { Ok(()) })
            .await
            .expect("manual compaction should not fail");

        assert_eq!(outcome, ManualCompactionOutcome::NothingToCompact);
        assert_eq!(
            agent.history,
            vec![
                HistoryItem::user("short prompt"),
                HistoryItem::assistant("reply")
            ]
        );
    }

    #[test]
    fn compaction_selection_never_summarizes_protected_current_turn() {
        let history = vec![
            HistoryItem::user("old user"),
            HistoryItem::assistant("old assistant"),
            HistoryItem::user("current user"),
            HistoryItem::assistant("current assistant"),
        ];

        let selection = select_compaction_segments(
            &history,
            2,
            &CompactionConfig {
                tail_turns: 0,
                preserve_recent_tokens: Some(0),
                ..CompactionConfig::default()
            },
            0,
        )
        .expect("selection succeeds");

        assert_eq!(selection.head_for_summary.len(), 2);
        assert!(selection.tail_items.is_empty());
        assert_eq!(
            &history[2..],
            &[
                HistoryItem::user("current user"),
                HistoryItem::assistant("current assistant")
            ]
        );
    }

    #[test]
    fn latest_item_over_budget_does_not_force_tail_retention() {
        let history = vec![
            HistoryItem::user("older user"),
            HistoryItem::assistant("older assistant"),
            HistoryItem::user("x".repeat(15_000)),
        ];

        let selection = select_compaction_segments(
            &history,
            3,
            &CompactionConfig {
                tail_turns: 1,
                ..CompactionConfig::default()
            },
            10,
        )
        .expect("selection succeeds");

        assert!(selection.tail_items.is_empty());
        assert_eq!(selection.head_for_summary.len(), 3);
        assert_eq!(selection.tail_start_index, 3);
    }

    #[test]
    fn oversized_latest_turn_can_keep_suffix_that_fits_budget() {
        let suffix = HistoryItem::assistant("small suffix");
        let history = vec![
            HistoryItem::user("older user"),
            HistoryItem::assistant("older assistant"),
            HistoryItem::user("x".repeat(15_000)),
            suffix.clone(),
        ];

        let selection = select_compaction_segments(
            &history,
            4,
            &CompactionConfig {
                tail_turns: 1,
                ..CompactionConfig::default()
            },
            estimate_history_item_tokens(&suffix),
        )
        .expect("selection succeeds");

        assert_eq!(selection.tail_items, vec![suffix]);
        assert_eq!(selection.head_for_summary.len(), 3);
        assert_eq!(selection.tail_start_index, 3);
    }

    #[test]
    fn compaction_tail_does_not_start_with_orphan_tool_output() {
        let tool_output = HistoryItem::ToolOutput {
            call_id: "call-read".into(),
            output_json: r#"{"ok":true}"#.into(),
        };
        let history = vec![
            HistoryItem::user("older user"),
            HistoryItem::assistant("older assistant"),
            HistoryItem::user("inspect file"),
            HistoryItem::AssistantToolCalls {
                text: None,
                calls: vec![test_tool_call("read", r#"{"path":"src/main.rs"}"#)],
            },
            tool_output.clone(),
        ];

        let selection = select_compaction_segments(
            &history,
            history.len(),
            &CompactionConfig {
                tail_turns: 1,
                ..CompactionConfig::default()
            },
            estimate_history_item_tokens(&tool_output),
        )
        .expect("selection succeeds");

        assert!(matches!(
            selection.tail_items.first(),
            Some(HistoryItem::AssistantToolCalls { .. })
        ));
        assert!(matches!(
            selection.tail_items.get(1),
            Some(HistoryItem::ToolOutput { call_id, .. }) if call_id == "call-read"
        ));
    }

    #[test]
    fn default_preserve_recent_budget_uses_quarter_clamped_range() {
        assert_eq!(default_preserve_recent_budget(1_000), 1_000);
        assert_eq!(default_preserve_recent_budget(12_000), 3_000);
        assert_eq!(default_preserve_recent_budget(100_000), 8_000);
    }

    #[test]
    fn compaction_history_char_budget_scales_with_model_window() {
        let small = compaction_history_char_budget(ModelRequestMetadata {
            context_window: Some(1_024),
            max_output_tokens: Some(128),
            ..ModelRequestMetadata::default()
        });
        let large = compaction_history_char_budget(ModelRequestMetadata {
            context_window: Some(128_000),
            max_output_tokens: Some(4_096),
            ..ModelRequestMetadata::default()
        });

        assert!(small <= 1_000);
        assert!(large > small);
        assert!(large <= COMPACTION_HISTORY_MAX_CHAR_BUDGET);
    }

    #[test]
    fn compaction_history_char_budget_uses_effective_input_limit() {
        let uncapped = compaction_history_char_budget(ModelRequestMetadata {
            context_window: Some(128_000),
            max_output_tokens: Some(4_096),
            ..ModelRequestMetadata::default()
        });
        let capped = compaction_history_char_budget(ModelRequestMetadata {
            context_window: Some(128_000),
            effective_input_limit_tokens: Some(4_000),
            max_output_tokens: Some(4_096),
            ..ModelRequestMetadata::default()
        });

        assert!(capped < uncapped);
        assert!(capped <= 4_000);
    }

    #[test]
    fn render_compaction_prompt_distinguishes_initial_and_incremental_modes() {
        let items = vec![HistoryItem::user("修复 src/agent.rs")];

        let initial = render_compaction_prompt(None, &items, 16_000);
        assert!(initial.contains("生成新的锚定摘要"));
        assert!(!initial.contains("更新已有锚定摘要"));

        let incremental = render_compaction_prompt(Some("已有摘要"), &items, 16_000);
        assert!(incremental.contains("更新已有锚定摘要"));
        assert!(incremental.contains("删除已过时或被推翻的信息"));
    }

    #[test]
    fn render_compaction_tool_output_caps_large_payloads() {
        let rendered = describe_history_item(&HistoryItem::ToolOutput {
            call_id: "call-big".into(),
            output_json: large_tool_output_json("stdout"),
        });

        assert!(rendered.contains(COMPACTION_TOOL_OUTPUT_TRUNCATION_MARKER));
        assert!(rendered.chars().count() < 2_200);
    }

    #[test]
    fn render_compaction_tool_output_strips_media_like_fields() {
        let base64 = "A".repeat(3_000);
        let rendered = describe_history_item(&HistoryItem::ToolOutput {
            call_id: "call-media".into(),
            output_json: json!({
                "image_base64": base64,
                "preview_url": "blob:https://example.invalid/123",
                "stdout": "kept text"
            })
            .to_string(),
        });

        assert!(rendered.contains("stripped media/blob-like field"));
        assert!(rendered.contains("kept text"));
        assert!(!rendered.contains("blob:https://example.invalid/123"));
        assert!(!rendered.contains(&"A".repeat(128)));
    }

    #[test]
    fn render_compaction_prompt_applies_total_history_cap() {
        let items = (0..20)
            .map(|index| HistoryItem::ToolOutput {
                call_id: format!("call-{index}"),
                output_json: large_tool_output_json("stdout"),
            })
            .collect::<Vec<_>>();

        let rendered = render_bounded_compaction_history(&items, 4_000);

        assert!(rendered.contains(COMPACTION_HISTORY_TRUNCATION_MARKER));
        assert!(rendered.chars().count() <= 4_000);
        assert!(rendered.contains("call-19"));
        assert!(!rendered.contains("call-0"));
    }

    #[test]
    fn prune_old_tool_outputs_protects_source_less_payloads() {
        let mut agent = test_agent();
        agent.compaction_config.prune = true;
        agent.compaction_config.tail_turns = 1;
        let prunable_output = prunable_tool_output_json("stdout");
        agent
            .replace_history(vec![
                HistoryItem::user("older turn"),
                HistoryItem::AssistantToolCalls {
                    text: None,
                    calls: vec![HistoryToolCall {
                        call_id: "call-read".into(),
                        name: "fs__read".into(),
                        arguments_json: r#"{"path":"src/lib.rs"}"#.into(),
                    }],
                },
                HistoryItem::ToolOutput {
                    call_id: "call-read".into(),
                    output_json: prunable_output.clone(),
                },
                HistoryItem::assistant(prune_protect_padding()),
                HistoryItem::user("recent turn"),
                HistoryItem::assistant("recent reply"),
                HistoryItem::user("current turn"),
            ])
            .expect("history replace succeeds");

        agent
            .prune_old_tool_outputs(4_000)
            .expect("pruning succeeds");

        let HistoryItem::ToolOutput { output_json, .. } = &agent.history[2] else {
            panic!("expected tool output");
        };
        assert_eq!(output_json, &prunable_output);
        assert_eq!(
            crate::protocol_frames::history_items_from_frames(agent.protocol_frames_for_test()),
            agent.history_for_test()
        );
        assert_eq!(
            agent.runtime_snapshot.frames.len(),
            agent.protocol_frames_for_test().len()
        );
    }

    #[test]
    fn prune_old_tool_outputs_skips_recent_and_skill_payloads() {
        let mut agent = test_agent();
        agent.compaction_config.prune = true;
        agent.compaction_config.tail_turns = 1;
        let skill_output = prunable_tool_output_json("result");
        let recent_output = prunable_tool_output_json("stdout");
        agent
            .replace_history(vec![
                HistoryItem::user("older turn"),
                HistoryItem::AssistantToolCalls {
                    text: None,
                    calls: vec![HistoryToolCall {
                        call_id: "call-skill".into(),
                        name: "skill".into(),
                        arguments_json: r#"{"name":"rust-audit"}"#.into(),
                    }],
                },
                HistoryItem::ToolOutput {
                    call_id: "call-skill".into(),
                    output_json: skill_output.clone(),
                },
                HistoryItem::assistant(prune_protect_padding()),
                HistoryItem::user("recent turn"),
                HistoryItem::AssistantToolCalls {
                    text: None,
                    calls: vec![HistoryToolCall {
                        call_id: "call-recent".into(),
                        name: "fs__read".into(),
                        arguments_json: r#"{"path":"src/main.rs"}"#.into(),
                    }],
                },
                HistoryItem::ToolOutput {
                    call_id: "call-recent".into(),
                    output_json: recent_output.clone(),
                },
                HistoryItem::user("current turn"),
            ])
            .expect("history replace succeeds");

        agent
            .prune_old_tool_outputs(4_000)
            .expect("pruning succeeds");

        let HistoryItem::ToolOutput {
            output_json: skill_after,
            ..
        } = &agent.history[2]
        else {
            panic!("expected skill tool output");
        };
        let HistoryItem::ToolOutput {
            output_json: recent_after,
            ..
        } = &agent.history[6]
        else {
            panic!("expected recent tool output");
        };
        assert_eq!(skill_after, &skill_output);
        assert_eq!(recent_after, &recent_output);
        assert_eq!(
            crate::protocol_frames::history_items_from_frames(agent.protocol_frames_for_test()),
            agent.history_for_test()
        );
        assert_eq!(
            agent.runtime_snapshot.frames.len(),
            agent.protocol_frames_for_test().len()
        );
    }

    #[test]
    fn context_checkpoint_restore_point_keeps_complete_tool_call_group() {
        let mut agent = test_agent();
        let call = test_tool_call(
            tool_names::TOOL_CONTEXT_CHECKPOINT,
            r#"{"label":"alt","reason":"try alternative approach"}"#,
        );
        agent
            .append_assistant_tool_calls("", std::slice::from_ref(&call))
            .expect("checkpoint-only tool batch should append");
        agent
            .append_history_item(HistoryItem::ToolOutput {
                call_id: call.call_id.clone(),
                output_json: json!({
                    "label": "alt",
                    "reason": "try alternative approach"
                })
                .to_string(),
            })
            .expect("tool output append succeeds");
        let scope = ActiveContextExperiment {
            branch_id: "branch-1".into(),
            parent_branch_id: ROOT_CONTEXT_BRANCH_ID.into(),
            base_sequence: 4,
            writes_observed: false,
        };
        agent.set_context_scope_state(Arc::new(std::sync::Mutex::new(ContextScopeState {
            active_experiment: Some(scope.clone()),
        })));

        agent.set_runtime_snapshot_provider(Arc::new(|| Ok(RuntimeSnapshot::new("branch-1"))));

        agent
            .finalize_context_checkpoint_after_recording()
            .expect("checkpoint finalize succeeds");

        let restore = agent
            .context_experiment_restore_point
            .as_ref()
            .expect("restore point stored");
        let restore_history =
            crate::protocol_frames::history_items_from_frames(&restore.protocol_frames);
        crate::protocol_frames::validate_history_items_complete(&restore_history, None)
            .expect("restore history remains protocol-complete");
        assert!(matches!(
            restore_history.last(),
            Some(HistoryItem::ToolOutput { call_id, .. }) if call_id == &call.call_id
        ));
        assert_eq!(
            restore.runtime_snapshot.evidence,
            agent.runtime_snapshot.evidence
        );
        assert_eq!(
            restore.runtime_snapshot.current_turn_id,
            Some(agent.next_turn_id)
        );
    }

    #[tokio::test]
    async fn context_return_records_output_before_restoring_parent_context() {
        let mut agent = test_agent();
        agent
            .append_history_item(HistoryItem::user("hello"))
            .expect("seed history");

        let restore_history = agent.history.clone();
        let restore_turn_id = agent.next_turn_id;
        let scope = ActiveContextExperiment {
            branch_id: "branch-1".into(),
            parent_branch_id: ROOT_CONTEXT_BRANCH_ID.into(),
            base_sequence: 1,
            writes_observed: false,
        };
        agent.set_context_scope_state(Arc::new(std::sync::Mutex::new(ContextScopeState {
            active_experiment: Some(scope.clone()),
        })));
        agent.set_context_experiment_restore_point(
            scope,
            crate::protocol_frames::history_items_to_frames(&restore_history),
            runtime_snapshot_for_history(ROOT_CONTEXT_BRANCH_ID, &restore_history)
                .with_current_turn_id(restore_turn_id),
        );
        let returned_summary =
            HistoryItem::context_summary(crate::transcript::format_context_experiment_return(
                "branch-1",
                "useful",
                "Found the issue",
                Some("Apply fix"),
                false,
            ));
        let parent_history = vec![restore_history[0].clone(), returned_summary];
        agent.set_runtime_snapshot_provider(Arc::new(move || {
            Ok(runtime_snapshot_for_history(
                ROOT_CONTEXT_BRANCH_ID,
                &parent_history,
            ))
        }));

        let call = test_tool_call(
            tool_names::TOOL_CONTEXT_RETURN,
            r#"{"outcome":"useful","summary":"Found the issue","next_action":"Apply fix"}"#,
        );
        agent
            .append_assistant_tool_calls("", std::slice::from_ref(&call))
            .expect("return tool call should append");

        agent
            .execute_tool_call_and_record(&call, &mut |_| std::future::ready(Ok(())), &mut |_| {
                std::future::ready(Ok(PermissionApproval::AllowOnce))
            })
            .await
            .expect("context return should record before restoring");

        crate::protocol_frames::validate_history_items_complete(&agent.history, None)
            .expect("restored history remains protocol-complete");
        assert_eq!(agent.history.len(), 2);
        assert!(matches!(&agent.history[0], HistoryItem::UserMessage { .. }));
        assert!(matches!(
            &agent.history[1],
            HistoryItem::ContextSummary { .. }
        ));
        assert!(agent.history.iter().all(
            |item| !matches!(item, HistoryItem::ToolOutput { call_id, .. } if call_id == &call.call_id)
        ));
        assert!(agent.context_experiment_restore_point.is_none());
    }

    #[test]
    fn context_checkpoint_batched_with_other_tool_call_fails_before_history_mutation() {
        let mut agent = test_agent();
        let calls = vec![
            test_tool_call(
                tool_names::TOOL_CONTEXT_CHECKPOINT,
                r#"{"label":"alt","reason":"try alternative approach"}"#,
            ),
            test_tool_call("fs__read", r#"{"path":"src/main.rs"}"#),
        ];
        let history_before = agent.history.clone();
        let frames_before = agent.protocol_frames.clone();
        let snapshot_before = agent.runtime_snapshot.clone();

        let error = agent
            .append_assistant_tool_calls("", &calls)
            .expect_err("batched checkpoint must fail before history mutation");

        assert!(error.to_string().contains(
            "context__checkpoint cannot be batched with other tool calls in the same assistant tool-call group"
        ));
        assert_eq!(agent.history, history_before);
        assert_eq!(agent.protocol_frames, frames_before);
        assert_eq!(agent.runtime_snapshot, snapshot_before);
    }

    #[test]
    fn context_return_batched_with_sibling_fails_before_history_mutation() {
        let mut agent = test_agent();
        let calls = vec![
            test_tool_call(
                tool_names::TOOL_CONTEXT_RETURN,
                r#"{"outcome":"useful","summary":"done","next_action":null}"#,
            ),
            test_tool_call("fs__read", r#"{"path":"src/main.rs"}"#),
        ];
        let history_before = agent.history.clone();
        let frames_before = agent.protocol_frames.clone();
        let snapshot_before = agent.runtime_snapshot.clone();

        let error = agent
            .append_assistant_tool_calls("", &calls)
            .expect_err("batched context__return must fail before history mutation");

        assert!(error.to_string().contains(
            "context__return cannot be batched with other tool calls in the same assistant tool-call group"
        ));
        assert_eq!(agent.history, history_before);
        assert_eq!(agent.protocol_frames, frames_before);
        assert_eq!(agent.runtime_snapshot, snapshot_before);
    }

    #[tokio::test]
    async fn preflight_does_not_prune_source_less_tool_outputs() {
        let mut agent = test_agent();
        agent.compaction_config.auto = false;
        agent.compaction_config.prune = true;
        agent.compaction_config.tail_turns = 1;
        let prunable_output = prunable_tool_output_json("stdout");
        agent
            .replace_history(vec![
                HistoryItem::user("older turn"),
                HistoryItem::AssistantToolCalls {
                    text: None,
                    calls: vec![HistoryToolCall {
                        call_id: "call-read".into(),
                        name: "fs__read".into(),
                        arguments_json: r#"{"path":"src/lib.rs"}"#.into(),
                    }],
                },
                HistoryItem::ToolOutput {
                    call_id: "call-read".into(),
                    output_json: prunable_output.clone(),
                },
                HistoryItem::assistant(prune_protect_padding()),
                HistoryItem::user("recent turn"),
                HistoryItem::assistant("recent reply"),
                HistoryItem::user("current turn"),
            ])
            .expect("history replace succeeds");

        let protected_start_index = agent.history.len() - 1;
        let turn_prelude = agent.prepare_turn_prelude("current turn");
        let mut on_event = |_| std::future::ready(Ok(()));
        let retained_start = agent
            .preflight_compact_context(&turn_prelude, protected_start_index, &[], &mut on_event)
            .await
            .expect("preflight succeeds");

        assert_eq!(retained_start, protected_start_index);
        let HistoryItem::ToolOutput { output_json, .. } = &agent.history[2] else {
            panic!("expected tool output");
        };
        assert_eq!(output_json, &prunable_output);
    }

    #[tokio::test]
    async fn preflight_compaction_treats_no_older_items_after_tail_as_noop() {
        let mut agent = test_agent();
        agent.compaction_config.auto = true;
        agent.compaction_config.tail_turns = 1;
        agent
            .replace_history(vec![
                HistoryItem::user("recent turn"),
                HistoryItem::assistant("recent reply"),
                HistoryItem::user("current turn"),
            ])
            .expect("history replace succeeds");

        let protected_start_index = agent.history.len() - 1;
        let turn_prelude = agent.prepare_turn_prelude("current turn");
        let mut on_event = |_| std::future::ready(Ok(()));
        let retained_start = agent
            .preflight_compact_context(&turn_prelude, protected_start_index, &[], &mut on_event)
            .await
            .expect("preflight no-op succeeds");

        assert_eq!(retained_start, protected_start_index);
    }

    #[tokio::test]
    async fn preflight_compaction_noops_when_protected_start_index_is_zero() {
        let mut agent = test_agent();
        agent.compaction_config.auto = true;
        agent.needs_compaction = true;
        let turn_prelude = agent.prepare_turn_prelude("current turn");
        let mut on_event = |_| std::future::ready(Ok(()));

        let retained_start = agent
            .preflight_compact_context(&turn_prelude, 0, &[], &mut on_event)
            .await
            .expect("preflight should noop");

        assert_eq!(retained_start, 0);
    }

    #[tokio::test]
    async fn ordinary_request_build_uses_installed_runtime_snapshot_only() {
        let mut agent = test_agent();
        agent.history = vec![HistoryItem::user("EXTERNAL-TRANSCRIPT-CONTENT")];
        agent.runtime_snapshot = runtime_snapshot_for_history(
            ROOT_CONTEXT_BRANCH_ID,
            &[HistoryItem::user("INSTALLED-RUNTIME-SNAPSHOT-CONTENT")],
        );
        let mut on_event = |_| std::future::ready(Ok(()));

        let prepared = compaction::prepare_request_build(
            &mut agent,
            ApiProtocol::Responses,
            &[],
            0,
            &[],
            &mut on_event,
        )
        .await
        .expect("ordinary request builds from the installed runtime snapshot");

        let request = match prepared.build.request {
            BuiltRequest::Responses(request) => serde_json::to_value(request),
            BuiltRequest::ResponsesCompatible(request) => Ok(request),
            BuiltRequest::Completions(_) | BuiltRequest::CompletionsCompatible(_) => {
                panic!("expected responses request")
            }
        };
        let request = request.expect("request serializes");
        let json = serde_json::to_string(&request).expect("request serializes");
        assert!(json.contains("INSTALLED-RUNTIME-SNAPSHOT-CONTENT"));
        assert!(!json.contains("EXTERNAL-TRANSCRIPT-CONTENT"));
    }

    #[tokio::test]
    async fn request_build_refreshes_provider_metadata_before_proactive_protected_folding() {
        let mut agent = test_agent();
        agent.model_catalog.insert(
            "m1".into(),
            ModelRequestMetadata {
                context_window: Some(8_192),
                effective_input_limit_tokens: Some(1_000),
                supports_tools: false,
                ..ModelRequestMetadata::default()
            },
        );
        agent.compaction_config.protected_reserve_tokens = Some(600);
        let raw_output = format!(
            r#"{{"status":0,"stdout":"PROVIDER-REFRESH-RAW-SENTINEL-{}"}}"#,
            "x".repeat(2_200)
        );
        let history = vec![
            HistoryItem::AssistantToolCalls {
                text: None,
                calls: vec![HistoryToolCall {
                    call_id: "provider-refresh-call".into(),
                    name: "shell__exec".into(),
                    arguments_json: "{}".into(),
                }],
            },
            HistoryItem::ToolOutput {
                call_id: "provider-refresh-call".into(),
                output_json: raw_output.clone(),
            },
            HistoryItem::user("continue"),
        ];
        agent
            .replace_history(history.clone())
            .expect("valid history");
        agent.runtime_snapshot = runtime_snapshot_for_history("main", &history);
        let protected_ids = agent
            .runtime_snapshot
            .frames
            .iter()
            .map(|frame| frame.id)
            .collect();
        agent
            .runtime_snapshot
            .set_protected_frame_ids(protected_ids);

        let mut projected = agent.runtime_snapshot.clone();
        let output_id = "provider-refresh-output";
        projected.context_view.folded_outputs.insert(
            output_id.into(),
            FoldedOutputMetadata {
                output_id: output_id.into(),
                node_id: None,
                output_kind: "shell_output".into(),
                call_id: Some("provider-refresh-call".into()),
                tool_name: Some("shell__exec".into()),
                stream: Some("stdout".into()),
                content: raw_output.clone(),
                byte_count: raw_output.len(),
                line_count: 1,
                truncated: false,
                shell_command: None,
                source_start_sequence: None,
                source_end_sequence: None,
                available_sequence: None,
                tool_ok: Some(true),
                exit_status: Some(0),
                provider_metadata: None,
                provider_fold_eligible: true,
            },
        );
        let block_id = ContextBlockId::new("provider-refresh-block").expect("valid block id");
        projected.context_view.blocks.insert(
            block_id.clone(),
            ContextBlock {
                block_id,
                node_id: None,
                kind: ContextBlockKind::ToolOutput,
                title: "provider refreshed output".into(),
                detail: String::new(),
                source: ContextBlockSource::FoldedOutput {
                    output_id: output_id.into(),
                },
                source_start_sequence: None,
                available_sequence: None,
                protected_reasons: Vec::new(),
                folded_output_id: Some(output_id.into()),
            },
        );
        let refreshes = Arc::new(AtomicUsize::new(0));
        let refresh_count = refreshes.clone();
        agent.set_runtime_snapshot_provider(Arc::new(move || {
            refresh_count.fetch_add(1, Ordering::SeqCst);
            Ok(projected.clone())
        }));

        let mut on_event = |_| std::future::ready(Ok(()));
        let prepared = compaction::prepare_request_build(
            &mut agent,
            ApiProtocol::Responses,
            &[],
            0,
            &[],
            &mut on_event,
        )
        .await
        .expect("provider metadata enables proactive folding");

        assert_eq!(refreshes.load(Ordering::SeqCst), 1);
        assert!(prepared.build.budget.provider_folded_output_count > 0);
        assert!(
            prepared
                .build
                .budget
                .estimated_provider_folded_protected_tokens
                > 0
        );
        let request = match prepared.build.request {
            BuiltRequest::Responses(request) => serde_json::to_string(&request),
            BuiltRequest::ResponsesCompatible(request) => serde_json::to_string(&request),
            BuiltRequest::Completions(_) | BuiltRequest::CompletionsCompatible(_) => {
                panic!("expected responses request")
            }
        }
        .expect("request serializes");
        assert!(!request.contains("PROVIDER-REFRESH-RAW-SENTINEL"));
        let ProtocolFrameItem::ToolOutput { output_json, .. } = agent.runtime_snapshot.frames[1]
            .protocol
            .as_ref()
            .expect("tool output protocol frame")
        else {
            panic!("expected tool output")
        };
        assert_eq!(output_json, &raw_output);
    }

    #[tokio::test]
    async fn preflight_compaction_uses_runtime_snapshot() {
        let mut agent = test_agent();
        agent.compaction_config.auto = true;
        agent.needs_compaction = true;
        agent
            .replace_history(vec![
                HistoryItem::user("older turn"),
                HistoryItem::assistant("older reply"),
                HistoryItem::user("current turn"),
            ])
            .expect("history replace succeeds");
        let protected_start_index = agent.history.len() - 1;
        let turn_prelude = agent.prepare_turn_prelude("current turn");
        let mut on_event = |_| std::future::ready(Ok(()));
        let retained_start = agent
            .preflight_compact_context(&turn_prelude, protected_start_index, &[], &mut on_event)
            .await
            .expect("preflight builds from runtime snapshot");
        assert!(retained_start <= protected_start_index);
    }

    #[test]
    fn context_overflow_classifier_is_conservative() {
        let overflow = anyhow!(
            "OpenAI API error: This model's maximum context length is 128000 tokens. Reduce the length of the messages."
        );
        let unrelated = anyhow!("request failed with status 500: upstream timeout");

        assert!(is_context_overflow_error(&overflow));
        assert!(!is_context_overflow_error(&unrelated));
        assert!(is_context_overflow_message(
            "prompt is too long for this model"
        ));
        assert!(!is_context_overflow_message(
            "token usage updated successfully"
        ));
    }

    #[test]
    fn context_overflow_error_marks_next_turn_for_compaction() {
        let mut agent = test_agent();
        let overflow = anyhow!("context length exceeded for this request");
        let unrelated = anyhow!("request failed with status 500: upstream timeout");

        agent.note_context_overflow_error(&unrelated);
        assert!(!agent.needs_compaction);

        agent.note_context_overflow_error(&overflow);
        assert!(agent.needs_compaction);
    }

    #[tokio::test]
    async fn chat_stream_creation_failure_includes_request_budget_diagnostic() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("test port should bind");
        let addr = listener.local_addr().expect("test listener has local addr");
        drop(listener);
        let client = Client::with_config(
            OpenAIConfig::new()
                .with_api_base(format!("http://{addr}"))
                .with_api_key("test"),
        );
        let mut agent = Agent::new(client, "m1", 4, 4);
        agent.set_model_catalog(HashMap::from([(
            "m1".into(),
            ModelRequestMetadata {
                context_window: Some(32_000),
                effective_input_limit_tokens: Some(1_200),
                max_output_tokens: Some(2_000),
                supports_tools: false,
                supports_reasoning: false,
                ..Default::default()
            },
        )]));
        let mut retry_config = test_retry_config();
        retry_config.enabled = false;
        agent.set_retry_config(retry_config);

        let error = agent
            .run_oai_comp_stream_async(
                "hello",
                |_| std::future::ready(Ok(())),
                |_| std::future::ready(Ok(())),
                |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
            )
            .await
            .expect_err("stream creation should fail");
        let message = format!("{error:#}");

        assert!(
            message.contains("failed to create streamed chat completion"),
            "{message}"
        );
        assert!(message.contains("model=m1"), "{message}");
        assert!(message.contains("estimated_request_tokens="), "{message}");
        assert!(message.contains("input_budget_tokens=1200"), "{message}");
        assert!(
            message.contains("effective_input_limit_tokens=1200"),
            "{message}"
        );
        assert!(message.contains("protected_tokens="), "{message}");
    }

    #[tokio::test]
    async fn compatible_chat_stream_sends_one_physical_request() {
        let (base_url, request_count, server) = spawn_chat_completion_server(vec![
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: 14\r\nConnection: close\r\n\r\ndata: [DONE]\n\n",
        ])
        .await;
        let client = Client::with_config(
            OpenAIConfig::new()
                .with_api_base(base_url)
                .with_api_key("test"),
        );
        let request = json!({"model": "m1", "stream": true, "messages": []});
        let response = send_compatible_chat_completion_stream(&client, &request)
            .await
            .expect("request should succeed");

        assert_eq!(response.status(), reqwest::StatusCode::OK);
        assert_eq!(request_count.load(Ordering::SeqCst), 1);
        server.await.expect("server task should finish");
    }

    #[tokio::test]
    async fn responses_completion_over_tool_call_budget_emits_completed_telemetry() {
        let body = r#"data: {"type":"response.completed","sequence_number":1,"response":{"id":"resp-over-budget","object":"response","created_at":1780856440,"status":"completed","background":false,"error":null,"incomplete_details":null,"instructions":null,"max_output_tokens":null,"model":"m1","output":[{"type":"function_call","id":"fc-1","call_id":"call-1","name":"first","arguments":"{}","status":"completed"},{"type":"function_call","id":"fc-2","call_id":"call-2","name":"second","arguments":"{}","status":"completed"}],"parallel_tool_calls":true,"previous_response_id":null,"reasoning":{},"store":true,"temperature":1,"text":{"format":{"type":"text"}},"tool_choice":"auto","tools":[],"top_p":1,"truncation":"disabled","usage":{"input_tokens":5,"input_tokens_details":{"cached_tokens":0},"output_tokens":3,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":8},"user":null,"metadata":{}}}

data: [DONE]

"#;
        let response = Box::leak(
            format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            )
            .into_boxed_str(),
        );
        let (base_url, request_count, server) = spawn_chat_completion_server(vec![response]).await;
        let client = Client::with_config(
            OpenAIConfig::new()
                .with_api_base(base_url)
                .with_api_key("test"),
        );
        let mut agent = Agent::new(client, "m1", 4, 1);
        agent.set_model_catalog(HashMap::from([(
            "m1".into(),
            ModelRequestMetadata {
                context_window: Some(32_000),
                max_output_tokens: Some(2_000),
                supports_tools: true,
                supports_reasoning: false,
                ..Default::default()
            },
        )]));
        let mut audit_telemetry = Vec::new();

        let error = agent
            .run_stream_async(
                "hello",
                |_| std::future::ready(Ok(())),
                |event| {
                    if let AgentEvent::LlmRequestTelemetry(telemetry) = event {
                        audit_telemetry.push(telemetry);
                    }
                    std::future::ready(Ok(()))
                },
                |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
            )
            .await
            .expect_err("over-budget tool calls should fail locally");

        assert!(error.to_string().contains("too many tool calls"));
        assert_eq!(
            audit_telemetry
                .iter()
                .map(|telemetry| telemetry.phase)
                .collect::<Vec<_>>(),
            vec![
                LlmRequestTelemetryPhase::Prepared,
                LlmRequestTelemetryPhase::Completed,
            ]
        );
        let completed = audit_telemetry
            .iter()
            .find(|telemetry| telemetry.phase == LlmRequestTelemetryPhase::Completed)
            .expect("provider completion telemetry");
        assert_eq!(
            completed.provider_response_id.as_deref(),
            Some("resp-over-budget")
        );
        assert_eq!(
            completed.usage.as_ref().map(|usage| usage.used_tokens),
            Some(8)
        );
        assert_request_telemetry_is_terminal_once(&audit_telemetry);
        assert_eq!(request_count.load(Ordering::SeqCst), 1);
        server.await.expect("server task should finish");
    }

    #[tokio::test]
    async fn responses_stream_recovers_from_malformed_event_after_visible_delta() {
        let first_body = r#"data: {"type":"response.output_text.delta","sequence_number":1,"item_id":"msg-1","output_index":0,"content_index":0,"delta":"partial "}

data: {"type":"response.completed","response":{"reasoning":{"effort":"max"}}}

"#;
        let second_body = r#"data: {"type":"response.completed","sequence_number":1,"response":{"id":"resp-recovered","object":"response","created_at":1780856440,"status":"completed","background":false,"error":null,"incomplete_details":null,"instructions":null,"max_output_tokens":null,"model":"m1","output":[{"type":"message","id":"msg-2","status":"completed","role":"assistant","content":[{"type":"output_text","text":"continued","annotations":[]}]}],"parallel_tool_calls":true,"previous_response_id":null,"reasoning":{},"store":true,"temperature":1,"text":{"format":{"type":"text"}},"tool_choice":"auto","tools":[],"top_p":1,"truncation":"disabled","usage":{"input_tokens":5,"input_tokens_details":{"cached_tokens":0},"output_tokens":3,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":8},"user":null,"metadata":{}}}

data: [DONE]

"#;
        let first_response = Box::leak(
            format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{first_body}",
                first_body.len()
            )
            .into_boxed_str(),
        );
        let second_response = Box::leak(
            format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{second_body}",
                second_body.len()
            )
            .into_boxed_str(),
        );
        let (base_url, request_count, server) =
            spawn_chat_completion_server(vec![first_response, second_response]).await;
        let client = Client::with_config(
            OpenAIConfig::new()
                .with_api_base(base_url)
                .with_api_key("test"),
        );
        let mut agent = Agent::new(client, "m1", 4, 4);
        agent.set_model_catalog(HashMap::from([(
            "m1".into(),
            ModelRequestMetadata {
                context_window: Some(32_000),
                max_output_tokens: Some(2_000),
                supports_tools: false,
                supports_reasoning: false,
                ..Default::default()
            },
        )]));
        agent.set_retry_config(test_retry_config());
        let mut deltas = Vec::new();
        let mut stream_issues = Vec::new();
        let mut audit_telemetry = Vec::new();

        let result = agent
            .run_stream_async(
                "hello",
                |delta| {
                    deltas.push(delta.to_string());
                    std::future::ready(Ok(()))
                },
                |event| {
                    match event {
                        AgentEvent::ModelStreamIssue { message, .. } => stream_issues.push(message),
                        AgentEvent::LlmRequestTelemetry(telemetry) => {
                            audit_telemetry.push(telemetry)
                        }
                        _ => {}
                    }
                    std::future::ready(Ok(()))
                },
                |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
            )
            .await
            .expect("malformed event after output should continue with a fresh iteration");

        assert_eq!(result, "partial continued");
        assert_eq!(deltas, vec!["partial "]);
        assert_eq!(stream_issues, vec!["Model stream interrupted"]);
        assert_eq!(
            audit_telemetry
                .iter()
                .map(|telemetry| (telemetry.phase, telemetry.error_class))
                .collect::<Vec<_>>(),
            vec![
                (LlmRequestTelemetryPhase::Prepared, None),
                (
                    LlmRequestTelemetryPhase::Interrupted,
                    Some(LlmRequestErrorClass::ProtocolValidation),
                ),
                (LlmRequestTelemetryPhase::Prepared, None),
                (LlmRequestTelemetryPhase::Completed, None),
            ]
        );
        assert_request_telemetry_is_terminal_once(&audit_telemetry);
        assert_eq!(request_count.load(Ordering::SeqCst), 2);
        server.await.expect("server task should finish");
    }

    #[tokio::test]
    async fn compatible_chat_stream_retries_read_error_before_visible_output() {
        let body = r#"data: {"choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}

data: [DONE]

"#;
        let first_response = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: 100\r\nConnection: close\r\n\r\n";
        let second_response = Box::leak(
            format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            )
            .into_boxed_str(),
        );
        let (base_url, request_count, server) =
            spawn_chat_completion_server(vec![first_response, second_response]).await;
        let client = Client::with_config(
            OpenAIConfig::new()
                .with_api_base(base_url)
                .with_api_key("test"),
        );
        let mut agent = Agent::new(client, "m1", 4, 4);
        agent.set_model_catalog(HashMap::from([(
            "m1".into(),
            ModelRequestMetadata {
                context_window: Some(32_000),
                max_output_tokens: Some(2_000),
                supports_tools: true,
                supports_reasoning: false,
                ..Default::default()
            },
        )]));
        agent.set_retry_config(test_retry_config());
        let mut deltas = Vec::new();
        let mut audit_telemetry = Vec::new();

        let result = agent
            .run_oai_comp_stream_async(
                "hello",
                |delta| {
                    deltas.push(delta.to_string());
                    std::future::ready(Ok(()))
                },
                |event| {
                    if let AgentEvent::LlmRequestTelemetry(telemetry) = event {
                        audit_telemetry.push(telemetry);
                    }
                    std::future::ready(Ok(()))
                },
                |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
            )
            .await
            .expect("pre-output stream read failure should retry");

        assert_eq!(result, "ok");
        assert_eq!(deltas, vec!["ok"]);
        assert_eq!(
            audit_telemetry
                .iter()
                .map(|telemetry| (telemetry.phase, telemetry.attempt))
                .collect::<Vec<_>>(),
            vec![
                (LlmRequestTelemetryPhase::Prepared, 1),
                (LlmRequestTelemetryPhase::Failed, 1),
                (LlmRequestTelemetryPhase::Prepared, 2),
                (LlmRequestTelemetryPhase::Completed, 2),
            ]
        );
        assert_request_telemetry_is_terminal_once(&audit_telemetry);
        assert_eq!(request_count.load(Ordering::SeqCst), 2);
        server.await.expect("server task should finish");
    }

    #[tokio::test]
    async fn compatible_chat_stream_continues_after_streamed_usage_event() {
        let first_body = r#"data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":0,"total_tokens":5}}

"#;
        let second_body = r#"data: {"choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}

data: [DONE]

"#;
        let first_response = Box::leak(
            format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: 1000\r\nConnection: close\r\n\r\n{first_body}"
            )
            .into_boxed_str(),
        );
        let second_response = Box::leak(
            format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{second_body}",
                second_body.len()
            )
            .into_boxed_str(),
        );
        let (base_url, request_count, server) =
            spawn_chat_completion_server(vec![first_response, second_response]).await;
        let client = Client::with_config(
            OpenAIConfig::new()
                .with_api_base(base_url)
                .with_api_key("test"),
        );
        let mut agent = Agent::new(client, "m1", 4, 4);
        agent.set_model_catalog(HashMap::from([(
            "m1".into(),
            ModelRequestMetadata {
                context_window: Some(32_000),
                max_output_tokens: Some(2_000),
                supports_tools: true,
                supports_reasoning: false,
                ..Default::default()
            },
        )]));
        agent.set_retry_config(test_retry_config());
        let mut deltas = Vec::new();
        let mut usage_events = Vec::new();
        let mut stream_issues = Vec::new();

        let result = agent
            .run_oai_comp_stream_async(
                "hello",
                |delta| {
                    deltas.push(delta.to_string());
                    std::future::ready(Ok(()))
                },
                |event| {
                    match event {
                        AgentEvent::TokenUsageUpdated { used_tokens, .. } => {
                            usage_events.push(used_tokens);
                        }
                        AgentEvent::ModelStreamIssue { message, .. } => {
                            stream_issues.push(message);
                        }
                        _ => {}
                    }
                    std::future::ready(Ok(()))
                },
                |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
            )
            .await
            .expect("stream read failure after usage event should continue with a fresh iteration");

        assert_eq!(result, "ok");
        assert_eq!(deltas, vec!["ok"]);
        assert_eq!(stream_issues, vec!["Model stream interrupted"]);
        assert!(
            usage_events.contains(&5),
            "missing streamed usage event: {usage_events:?}"
        );
        assert_eq!(request_count.load(Ordering::SeqCst), 2);
        server.await.expect("server task should finish");
    }

    #[tokio::test]
    async fn compatible_chat_stream_retries_incomplete_json_event_before_visible_output() {
        let first_body = "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"partial\"}\n\n";
        let second_body = r#"data: {"choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}

data: [DONE]

"#;
        let first_response = Box::leak(
            format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n{first_body}"
            )
            .into_boxed_str(),
        );
        let second_response = Box::leak(
            format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{second_body}",
                second_body.len()
            )
            .into_boxed_str(),
        );
        let (base_url, request_count, server) =
            spawn_chat_completion_server(vec![first_response, second_response]).await;
        let client = Client::with_config(
            OpenAIConfig::new()
                .with_api_base(base_url)
                .with_api_key("test"),
        );
        let mut agent = Agent::new(client, "m1", 4, 4);
        agent.set_model_catalog(HashMap::from([(
            "m1".into(),
            ModelRequestMetadata {
                context_window: Some(32_000),
                max_output_tokens: Some(2_000),
                supports_tools: true,
                supports_reasoning: false,
                ..Default::default()
            },
        )]));
        agent.set_retry_config(test_retry_config());
        let mut deltas = Vec::new();

        let result = agent
            .run_oai_comp_stream_async(
                "hello",
                |delta| {
                    deltas.push(delta.to_string());
                    std::future::ready(Ok(()))
                },
                |_| std::future::ready(Ok(())),
                |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
            )
            .await
            .expect("incomplete pre-output json event should retry");

        assert_eq!(result, "ok");
        assert_eq!(deltas, vec!["ok"]);
        assert_eq!(request_count.load(Ordering::SeqCst), 2);
        server.await.expect("server task should finish");
    }

    #[tokio::test]
    async fn compatible_chat_stream_shares_retry_budget_across_creation_and_read() {
        let body = r#"data: {"choices":[{"index":0,"delta":{"content":"too-late"},"finish_reason":"stop"}]}

data: [DONE]

"#;
        let third_response = Box::leak(
            format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            )
            .into_boxed_str(),
        );
        let (base_url, request_count, server) = spawn_chat_completion_server(vec![
            "HTTP/1.1 500 Internal Server Error\r\nContent-Length: 9\r\nConnection: close\r\n\r\ntransient",
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: 100\r\nConnection: close\r\n\r\n",
            third_response,
        ])
        .await;
        let client = Client::with_config(
            OpenAIConfig::new()
                .with_api_base(base_url)
                .with_api_key("test"),
        );
        let mut retry_config = test_retry_config();
        retry_config.max_attempts = 2;
        let mut agent = Agent::new(client, "m1", 4, 4);
        agent.set_model_catalog(HashMap::from([(
            "m1".into(),
            ModelRequestMetadata {
                context_window: Some(32_000),
                max_output_tokens: Some(2_000),
                supports_tools: true,
                supports_reasoning: false,
                ..Default::default()
            },
        )]));
        agent.set_retry_config(retry_config);
        let mut deltas = Vec::new();

        let error = agent
            .run_oai_comp_stream_async(
                "hello",
                |delta| {
                    deltas.push(delta.to_string());
                    std::future::ready(Ok(()))
                },
                |_| std::future::ready(Ok(())),
                |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
            )
            .await
            .expect_err("read retry should not exceed shared max_attempts budget");

        assert!(
            !error.to_string().trim().is_empty(),
            "unexpected empty error message: {error:?}"
        );
        assert!(deltas.is_empty());
        assert_eq!(request_count.load(Ordering::SeqCst), 2);
        server.abort();
    }

    #[tokio::test]
    async fn compatible_chat_stream_recovers_read_error_after_visible_text() {
        let body = r#"data: {"choices":[{"index":0,"delta":{"content":"partial"},"finish_reason":null}]}

"#;
        let first_response = Box::leak(
            format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: 1000\r\nConnection: close\r\n\r\n{body}"
            )
            .into_boxed_str(),
        );
        let second_body = r#"data: {"choices":[{"index":0,"delta":{"content":" continuation"},"finish_reason":"stop"}]}

data: [DONE]

"#;
        let second_response = Box::leak(
            format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{second_body}",
                second_body.len()
            )
            .into_boxed_str(),
        );
        let (base_url, request_count, server) =
            spawn_chat_completion_server(vec![first_response, second_response]).await;
        let client = Client::with_config(
            OpenAIConfig::new()
                .with_api_base(base_url)
                .with_api_key("test"),
        );
        let mut agent = Agent::new(client, "m1", 4, 4);
        agent.set_model_catalog(HashMap::from([(
            "m1".into(),
            ModelRequestMetadata {
                context_window: Some(32_000),
                max_output_tokens: Some(2_000),
                supports_tools: true,
                supports_reasoning: false,
                ..Default::default()
            },
        )]));
        agent.set_retry_config(test_retry_config());
        let mut deltas = Vec::new();
        let mut stream_issues = Vec::new();
        let mut finalized_outcomes = Vec::new();

        let result = agent
            .run_oai_comp_stream_async(
                "hello",
                |delta| {
                    deltas.push(delta.to_string());
                    std::future::ready(Ok(()))
                },
                |event| {
                    match event {
                        AgentEvent::TurnFinalized(event) => finalized_outcomes.push(event.outcome),
                        AgentEvent::ModelStreamIssue { message, .. } => stream_issues.push(message),
                        _ => {}
                    }
                    std::future::ready(Ok(()))
                },
                |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
            )
            .await
            .expect("post-output stream read failure should continue with a fresh iteration");

        let expected = "partial continuation".to_string();
        assert_eq!(result, expected);
        assert_eq!(deltas, vec!["partial", " continuation"]);
        assert_eq!(stream_issues, vec!["Model stream interrupted"]);
        assert_eq!(finalized_outcomes, vec!["completed"]);
        assert_eq!(request_count.load(Ordering::SeqCst), 2);
        assert!(
            agent.history.iter().any(
                |item| matches!(item, HistoryItem::AssistantText { text } if text == "partial")
            )
        );
        server.await.expect("server task should finish");
    }

    #[tokio::test]
    async fn compatible_chat_stream_recovers_missing_finish_reason_after_visible_text() {
        let body = r#"data: {"choices":[{"index":0,"delta":{"content":"partial"},"finish_reason":null}]}

data: [DONE]

"#;
        let first_response = Box::leak(
            format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            )
            .into_boxed_str(),
        );
        let second_body = r#"data: {"choices":[{"index":0,"delta":{"content":" continuation"},"finish_reason":"stop"}]}

data: [DONE]

"#;
        let second_response = Box::leak(
            format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{second_body}",
                second_body.len()
            )
            .into_boxed_str(),
        );
        let (base_url, request_count, server) =
            spawn_chat_completion_server(vec![first_response, second_response]).await;
        let client = Client::with_config(
            OpenAIConfig::new()
                .with_api_base(base_url)
                .with_api_key("test"),
        );
        let mut agent = Agent::new(client, "m1", 4, 4);
        agent.set_model_catalog(HashMap::from([(
            "m1".into(),
            ModelRequestMetadata {
                context_window: Some(32_000),
                max_output_tokens: Some(2_000),
                supports_tools: true,
                supports_reasoning: false,
                ..Default::default()
            },
        )]));
        agent.set_retry_config(test_retry_config());

        let result = agent
            .run_oai_comp_stream_async(
                "hello",
                |_| std::future::ready(Ok(())),
                |_| std::future::ready(Ok(())),
                |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
            )
            .await
            .expect("missing finish_reason after visible text should continue next iteration");

        let expected = "partial continuation".to_string();
        assert_eq!(result, expected);
        assert!(matches!(
            agent.history.first(),
            Some(HistoryItem::UserMessage { .. })
        ));
        assert!(
            agent.history.iter().any(
                |item| matches!(item, HistoryItem::AssistantText { text } if text == "partial")
            )
        );
        assert_eq!(request_count.load(Ordering::SeqCst), 2);
        server.await.expect("server task should finish");
    }

    #[tokio::test]
    async fn compatible_chat_stream_cancels_pending_tool_call_on_invalid_finish_reason() {
        let first_body = r#"data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call-interrupted","type":"function","function":{"name":"shell__exec","arguments":""}}]},"finish_reason":"stop"}]}

data: [DONE]

"#;
        let second_body = r#"data: {"choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}

data: [DONE]

"#;
        let first_response = Box::leak(
            format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{first_body}",
                first_body.len()
            )
            .into_boxed_str(),
        );
        let second_response = Box::leak(
            format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{second_body}",
                second_body.len()
            )
            .into_boxed_str(),
        );
        let (base_url, request_count, server) =
            spawn_chat_completion_server(vec![first_response, second_response]).await;
        let client = Client::with_config(
            OpenAIConfig::new()
                .with_api_base(base_url)
                .with_api_key("test"),
        );
        let mut agent = Agent::new(client, "m1", 4, 4);
        agent.set_model_catalog(HashMap::from([(
            "m1".into(),
            ModelRequestMetadata {
                context_window: Some(32_000),
                max_output_tokens: Some(2_000),
                supports_tools: true,
                supports_reasoning: false,
                ..Default::default()
            },
        )]));
        agent.set_retry_config(test_retry_config());
        let mut cancelled_calls = Vec::new();
        let mut started_calls = Vec::new();
        let mut finished_calls = Vec::new();
        let mut stream_issues = Vec::new();

        let result = agent
            .run_oai_comp_stream_async(
                "hello",
                |_| std::future::ready(Ok(())),
                |event| {
                    match event {
                        AgentEvent::ToolCallCancelled { call_id, name } => {
                            cancelled_calls.push((call_id, name));
                        }
                        AgentEvent::ToolCallStarted { call_id, .. } => started_calls.push(call_id),
                        AgentEvent::ToolCallFinished { call_id, .. } => {
                            finished_calls.push(call_id)
                        }
                        AgentEvent::ModelStreamIssue { message, .. } => stream_issues.push(message),
                        _ => {}
                    }
                    std::future::ready(Ok(()))
                },
                |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
            )
            .await
            .expect("invalid finish reason with pending tool should cancel and continue");

        assert_eq!(result, "ok");
        assert_eq!(
            cancelled_calls,
            vec![("call-interrupted".to_string(), "shell__exec".to_string())]
        );
        assert!(started_calls.is_empty());
        assert!(finished_calls.is_empty());
        assert_eq!(stream_issues, vec!["Model stream interrupted"]);
        assert!(!agent.history.iter().any(|item| matches!(
            item,
            HistoryItem::AssistantToolCalls { calls, .. }
                if calls.iter().any(|call| call.call_id == "call-interrupted")
        )));
        assert_eq!(request_count.load(Ordering::SeqCst), 2);
        server.await.expect("server task should finish");
    }

    #[tokio::test]
    async fn compatible_chat_stream_cancels_pending_tool_call_before_terminal_finish_error() {
        let body = r#"data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call-filtered","type":"function","function":{"name":"shell__exec","arguments":""}}]},"finish_reason":"content_filter"}]}

data: [DONE]

"#;
        let response = Box::leak(
            format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            )
            .into_boxed_str(),
        );
        let (base_url, request_count, server) = spawn_chat_completion_server(vec![response]).await;
        let client = Client::with_config(
            OpenAIConfig::new()
                .with_api_base(base_url)
                .with_api_key("test"),
        );
        let mut agent = Agent::new(client, "m1", 4, 4);
        agent.set_model_catalog(HashMap::from([(
            "m1".into(),
            ModelRequestMetadata {
                context_window: Some(32_000),
                max_output_tokens: Some(2_000),
                supports_tools: true,
                supports_reasoning: false,
                ..Default::default()
            },
        )]));
        agent.set_retry_config(test_retry_config());
        let mut cancelled_calls = Vec::new();
        let mut started_calls = Vec::new();

        let error = agent
            .run_oai_comp_stream_async(
                "hello",
                |_| std::future::ready(Ok(())),
                |event| {
                    match event {
                        AgentEvent::ToolCallCancelled { call_id, name } => {
                            cancelled_calls.push((call_id, name));
                        }
                        AgentEvent::ToolCallStarted { call_id, .. } => started_calls.push(call_id),
                        _ => {}
                    }
                    std::future::ready(Ok(()))
                },
                |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
            )
            .await
            .expect_err("content_filter should remain terminal after cancelling pending tool");

        assert!(error.to_string().contains("finish_reason=content_filter"));
        assert_eq!(
            cancelled_calls,
            vec![("call-filtered".to_string(), "shell__exec".to_string())]
        );
        assert!(started_calls.is_empty());
        assert_eq!(request_count.load(Ordering::SeqCst), 1);
        server.await.expect("server task should finish");
    }

    #[tokio::test]
    async fn compatible_chat_stream_does_not_recover_terminal_finish_reason_errors() {
        for (finish_reason, expected_error) in [
            ("length", "finish_reason=length"),
            ("content_filter", "finish_reason=content_filter"),
        ] {
            let body = format!(
                r#"data: {{"choices":[{{"index":0,"delta":{{"content":"partial"}},"finish_reason":"{finish_reason}"}}]}}

data: [DONE]

"#
            );
            let response = Box::leak(
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                )
                .into_boxed_str(),
            );
            let (base_url, request_count, server) =
                spawn_chat_completion_server(vec![response]).await;
            let client = Client::with_config(
                OpenAIConfig::new()
                    .with_api_base(base_url)
                    .with_api_key("test"),
            );
            let mut agent = Agent::new(client, "m1", 4, 4);
            agent.set_model_catalog(HashMap::from([(
                "m1".into(),
                ModelRequestMetadata {
                    context_window: Some(32_000),
                    max_output_tokens: Some(2_000),
                    supports_tools: true,
                    supports_reasoning: false,
                    ..Default::default()
                },
            )]));
            agent.set_retry_config(test_retry_config());
            let mut deltas = Vec::new();

            let error = agent
                .run_oai_comp_stream_async(
                    "hello",
                    |delta| {
                        deltas.push(delta.to_string());
                        std::future::ready(Ok(()))
                    },
                    |_| std::future::ready(Ok(())),
                    |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
                )
                .await
                .expect_err("terminal finish_reason errors should fail explicitly");

            assert!(
                error.to_string().contains(expected_error),
                "unexpected error for {finish_reason}: {error:?}"
            );
            assert_eq!(deltas, vec!["partial"]);
            assert_eq!(request_count.load(Ordering::SeqCst), 1);
            assert!(
                !agent
                    .history
                    .iter()
                    .any(|item| matches!(item, HistoryItem::AssistantText { .. }))
            );
            server.await.expect("server task should finish");
        }
    }

    #[tokio::test]
    async fn compatible_chat_stream_does_not_retry_bad_request() {
        let (base_url, request_count, server) = spawn_chat_completion_server(vec![
            "HTTP/1.1 400 Bad Request\r\nContent-Length: 11\r\nConnection: close\r\n\r\nbad request",
        ])
        .await;
        let client = Client::with_config(
            OpenAIConfig::new()
                .with_api_base(base_url)
                .with_api_key("test"),
        );
        let request = json!({"model": "m1", "stream": true, "messages": []});
        let error = send_compatible_chat_completion_stream(&client, &request)
            .await
            .expect_err("400 should fail fast");

        assert!(
            error
                .to_string()
                .contains("chat completions request failed with status 400 Bad Request")
        );
        assert_eq!(request_count.load(Ordering::SeqCst), 1);
        server.await.expect("server task should finish");
    }

    #[test]
    fn auto_continue_defaults_to_disabled() {
        let agent = test_agent();

        assert_eq!(agent.auto_continue(), &AutoContinueState::default());
        assert!(agent.todos().is_empty());
    }

    #[tokio::test]
    async fn workflow_auto_continue_tool_enables_bounded_state() {
        let mut agent = test_agent();
        let call = HistoryToolCall {
            call_id: "call-auto".into(),
            name: "workflow__auto_continue".into(),
            arguments_json: r#"{"enabled":true,"max_continuations":2}"#.into(),
        };

        let record = agent
            .execute_tool_call(&call, &mut |_| std::future::ready(Ok(())), &mut |_| {
                std::future::ready(Ok(PermissionApproval::AllowOnce))
            })
            .await
            .expect("control tool should succeed");

        assert!(record.output.ok);
        assert_eq!(agent.auto_continue().enabled, true);
        assert_eq!(agent.auto_continue().max_continuations, 2);
    }

    #[tokio::test]
    async fn execute_tool_call_records_success_status_effects_and_started_finished_events() {
        let mut agent = test_agent();
        let call = test_tool_call(
            "workflow__todos",
            r#"{"items":[{"id":"t1","content":"first","status":"pending"}]}"#,
        );
        let mut events = Vec::new();

        let record = agent
            .execute_tool_call(
                &call,
                &mut |event| {
                    events.push(event);
                    std::future::ready(Ok(()))
                },
                &mut |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
            )
            .await
            .expect("tool call should succeed");

        assert_eq!(record.status, ToolExecutionStatus::Executed);
        assert_eq!(record.rejection, None);
        assert!(record.output.ok);
        assert_eq!(record.effects.kind, ToolEffectKind::WorkflowControl);
        assert!(matches!(
            events.as_slice(),
            [
                AgentEvent::ToolCallStarted { .. },
                AgentEvent::TodoSnapshotUpdated { .. },
                AgentEvent::ToolCallFinished { ok: true, .. },
                AgentEvent::ToolExecutionSummary(ToolExecutionSummaryEvent {
                    status,
                    effect_kind,
                    ..
                })
            ] if status == "executed" && effect_kind == "workflow_control"
        ));
    }

    #[tokio::test]
    async fn workflow_todos_tool_updates_todo_state() {
        let mut agent = test_agent();
        let call = HistoryToolCall {
            call_id: "call-todos".into(),
            name: "workflow__todos".into(),
            arguments_json: r#"{"items":[{"id":"t1","content":"first","status":"pending"},{"id":"t2","content":"done","status":"completed"}]}"#.into(),
        };

        agent
            .execute_tool_call(&call, &mut |_| std::future::ready(Ok(())), &mut |_| {
                std::future::ready(Ok(PermissionApproval::AllowOnce))
            })
            .await
            .expect("todo control tool should succeed");

        assert_eq!(agent.todos().len(), 2);
        assert_eq!(agent.todos()[0].status, TodoStatus::Pending);
        assert_eq!(agent.todos()[1].status, TodoStatus::Completed);
    }

    #[tokio::test]
    async fn workflow_todos_event_failure_does_not_mutate_state() {
        let mut agent = test_agent();
        let previous = vec![TodoItem {
            id: "old".into(),
            content: "old task".into(),
            status: TodoStatus::InProgress,
        }];
        agent.turn.workflow.todos = previous.clone();
        let args = json!({
            "items": [{"id":"new","content":"new task","status":"pending"}]
        });

        let result = agent
            .apply_control_tool_state("workflow__todos", &args, &mut |_| {
                std::future::ready(Err(anyhow!("event sink failed")))
            })
            .await;

        assert!(result.is_err());
        assert_eq!(agent.todos(), previous.as_slice());
    }

    #[tokio::test]
    async fn workflow_auto_continue_event_failure_does_not_mutate_state() {
        let mut agent = test_agent();
        let previous = AutoContinueState {
            enabled: false,
            max_continuations: 3,
        };
        agent.turn.workflow.auto_continue = previous.clone();
        let args = json!({"enabled": true, "max_continuations": 5});

        let result = agent
            .apply_control_tool_state("workflow__auto_continue", &args, &mut |_| {
                std::future::ready(Err(anyhow!("event sink failed")))
            })
            .await;

        assert!(result.is_err());
        assert_eq!(agent.auto_continue(), &previous);
    }

    #[tokio::test]
    async fn solo_external_workspace_read_executes_without_approval() {
        let mut agent = test_agent();
        agent.set_permission_mode(PermissionMode::Solo);
        let outside_path = std::env::temp_dir().join(format!(
            "letcode-outside-agent-read-{}.txt",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system time")
                .as_nanos()
        ));
        std::fs::write(&outside_path, "outside\n").expect("write outside fixture");
        let outside = outside_path.to_string_lossy().to_string();
        let call = HistoryToolCall {
            call_id: "call-outside-read".into(),
            name: "fs__read".into(),
            arguments_json: json!({"path": outside, "offset": 1, "limit": 10}).to_string(),
        };
        let mut permission_requests = Vec::new();

        let record = agent
            .execute_tool_call(&call, &mut |_| std::future::ready(Ok(())), &mut |request| {
                permission_requests.push(request);
                std::future::ready(Ok(PermissionApproval::AllowOnce))
            })
            .await
            .expect("outside read should execute in solo mode");

        assert!(record.output.ok, "{:?}", record.output.error);
        assert!(permission_requests.is_empty());
        assert!(
            record
                .output
                .data
                .as_ref()
                .and_then(|data| data.get("content"))
                .and_then(Value::as_str)
                .is_some_and(|content| content.contains("outside"))
        );

        let _ = std::fs::remove_file(outside_path);
    }

    #[tokio::test]
    async fn default_external_workspace_read_allow_always_grants_only_matching_resource() {
        let mut agent = test_agent();
        let fixture_root = std::env::temp_dir().join(format!(
            "letcode-external-read-grant-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system time")
                .as_nanos()
        ));
        let first_directory = fixture_root.join("first");
        let second_directory = fixture_root.join("second");
        std::fs::create_dir_all(&first_directory).expect("create first fixture directory");
        std::fs::create_dir_all(&second_directory).expect("create second fixture directory");
        let first_path = first_directory.join("read.txt");
        let second_path = second_directory.join("read.txt");
        std::fs::write(&first_path, "first\n").expect("write first fixture");
        std::fs::write(&second_path, "second\n").expect("write second fixture");

        let first_call = HistoryToolCall {
            call_id: "call-external-read-first".into(),
            name: "fs__read".into(),
            arguments_json: json!({"path": first_path, "offset": 1, "limit": 10}).to_string(),
        };
        let repeated_call = HistoryToolCall {
            call_id: "call-external-read-repeated".into(),
            name: "fs__read".into(),
            arguments_json: first_call.arguments_json.clone(),
        };
        let other_directory_call = HistoryToolCall {
            call_id: "call-external-read-other".into(),
            name: "fs__read".into(),
            arguments_json: json!({"path": second_path, "offset": 1, "limit": 10}).to_string(),
        };
        let mut approval_requests = 0;

        let first = agent
            .execute_tool_call(
                &first_call,
                &mut |_| std::future::ready(Ok(())),
                &mut |request| {
                    approval_requests += 1;
                    assert!(request.can_allow_always);
                    std::future::ready(Ok(PermissionApproval::AllowAlways))
                },
            )
            .await
            .expect("first external read should execute after approval");
        assert!(first.output.ok);
        assert_eq!(approval_requests, 1);

        let repeated = agent
            .execute_tool_call(
                &repeated_call,
                &mut |_| std::future::ready(Ok(())),
                &mut |_| {
                    approval_requests += 1;
                    std::future::ready(Ok(PermissionApproval::Deny))
                },
            )
            .await
            .expect("matching grant should execute without approval");
        assert!(repeated.output.ok);
        assert_eq!(approval_requests, 1, "matching grant must bypass approval");

        let other_directory = agent
            .execute_tool_call(
                &other_directory_call,
                &mut |_| std::future::ready(Ok(())),
                &mut |_| {
                    approval_requests += 1;
                    std::future::ready(Ok(PermissionApproval::Deny))
                },
            )
            .await
            .expect("other external directory should request approval");
        assert_eq!(approval_requests, 2);
        assert_eq!(other_directory.status, ToolExecutionStatus::Rejected);
        assert_eq!(
            other_directory.rejection,
            Some(ToolExecutionRejection::PermissionDeniedByUser)
        );

        let _ = std::fs::remove_dir_all(fixture_root);
    }

    #[tokio::test]
    async fn matching_grant_does_not_override_base_policy_denial() {
        let mut agent = test_agent();
        let command = "rm -rf letcode-test-target";
        agent
            .permission_session
            .lock()
            .expect("permission session")
            .grant(crate::permission::PermissionResource::Exact {
                tool: "shell__exec".into(),
                value: command.into(),
            });
        let call = HistoryToolCall {
            call_id: "call-granted-policy-denial".into(),
            name: "shell__exec".into(),
            arguments_json: json!({"command": command}).to_string(),
        };
        let mut approval_requested = false;

        let record = agent
            .execute_tool_call(&call, &mut |_| std::future::ready(Ok(())), &mut |_| {
                approval_requested = true;
                std::future::ready(Ok(PermissionApproval::AllowOnce))
            })
            .await
            .expect("policy denial should produce a rejection record");

        assert!(!approval_requested);
        assert_eq!(record.status, ToolExecutionStatus::Rejected);
        assert_eq!(
            record.rejection,
            Some(ToolExecutionRejection::PermissionDeniedByPolicy)
        );
    }

    #[tokio::test]
    async fn solo_external_workspace_write_executes_without_approval() {
        let mut agent = test_agent();
        agent.set_permission_mode(PermissionMode::Solo);
        let outside_path = std::env::temp_dir().join(format!(
            "letcode-outside-agent-denied-write-{}.txt",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system time")
                .as_nanos()
        ));
        let outside = outside_path.to_string_lossy().to_string();
        let call = HistoryToolCall {
            call_id: "call-outside-write-denied".into(),
            name: "fs__write".into(),
            arguments_json: json!({"path": outside, "content": "denied"}).to_string(),
        };
        let mut permission_requests = Vec::new();
        let mut events = Vec::new();

        let record = agent
            .execute_tool_call(
                &call,
                &mut |event| {
                    events.push(event);
                    std::future::ready(Ok(()))
                },
                &mut |request| {
                    permission_requests.push(request);
                    std::future::ready(Ok(PermissionApproval::Deny))
                },
            )
            .await
            .expect("outside write should execute in solo mode");

        assert!(permission_requests.is_empty());
        assert_eq!(record.status, ToolExecutionStatus::Executed);
        assert!(record.output.ok);
        assert!(outside_path.exists());
        assert!(
            events
                .iter()
                .any(|event| matches!(event, AgentEvent::ToolCallFinished { ok: true, .. }))
        );
        let _ = std::fs::remove_file(outside_path);
    }

    #[tokio::test]
    async fn solo_mode_executes_commands_that_default_mode_denies_by_policy() {
        let mut agent = test_agent();
        agent.set_permission_mode(PermissionMode::Solo);
        let call = HistoryToolCall {
            call_id: "call-solo-deny-risk".into(),
            name: "shell__exec".into(),
            arguments_json: json!({"command": "curl --version"}).to_string(),
        };
        let mut approval_requested = false;

        let record = agent
            .execute_tool_call(&call, &mut |_| std::future::ready(Ok(())), &mut |_| {
                approval_requested = true;
                std::future::ready(Ok(PermissionApproval::Deny))
            })
            .await
            .expect("solo mode should execute command without asking");

        assert!(!approval_requested);
        assert_eq!(record.status, ToolExecutionStatus::Executed);
        assert_ne!(
            record.rejection,
            Some(ToolExecutionRejection::PermissionDeniedByPolicy)
        );
    }

    #[tokio::test]
    async fn unfinished_todos_trigger_bounded_internal_continuation() {
        let mut agent = test_agent();
        agent.prepare_turn_prelude("implement a feature");
        let turn_id = agent.current_turn_id();
        agent.turn.workflow.auto_continue = AutoContinueState {
            enabled: true,
            max_continuations: 2,
        };
        agent.turn.workflow.todos = vec![TodoItem {
            id: "t1".into(),
            content: "keep going".into(),
            status: TodoStatus::InProgress,
        }];
        let mut continuation_count = 0;
        let mut events = Vec::new();

        let should_continue = agent
            .continue_after_no_tool_reply(
                &mut |event| {
                    events.push(event);
                    std::future::ready(Ok(()))
                },
                &mut continuation_count,
            )
            .await
            .expect("continuation decision succeeds");

        assert!(should_continue);
        assert_eq!(continuation_count, 1);
        assert_eq!(agent.current_turn_id(), turn_id);
        assert!(matches!(
            agent.history.last(),
            Some(HistoryItem::InternalContinuation { .. })
        ));
        assert!(matches!(
            events.as_slice(),
            [
                AgentEvent::InternalContinuation {
                    source: crate::transcript::InternalContinuationSource::AutoContinue,
                    ..
                },
                AgentEvent::AutoContinuationScheduled {
                    continuation_count: 1,
                    remaining_unfinished: 1,
                }
            ]
        ));
    }

    #[tokio::test]
    async fn auto_continue_stops_when_todos_do_not_progress() {
        let mut agent = test_agent();
        agent.turn.workflow.auto_continue = AutoContinueState {
            enabled: true,
            max_continuations: 3,
        };
        agent.turn.workflow.todos = vec![TodoItem {
            id: "t1".into(),
            content: "still pending".into(),
            status: TodoStatus::Pending,
        }];
        let mut continuation_count = 0;

        assert!(
            agent
                .continue_after_no_tool_reply(
                    &mut |_| std::future::ready(Ok(())),
                    &mut continuation_count
                )
                .await
                .expect("first continuation should proceed")
        );

        let error = agent
            .continue_after_no_tool_reply(
                &mut |_| std::future::ready(Ok(())),
                &mut continuation_count,
            )
            .await
            .expect_err("unchanged todo snapshot should stop");

        assert!(error.to_string().contains("no todo progress"));
        assert_eq!(continuation_count, 1);
    }

    #[tokio::test]
    async fn completed_or_blocked_todos_stop_auto_continuation() {
        let mut agent = test_agent();
        agent.turn.workflow.auto_continue.enabled = true;
        let mut continuation_count = 0;

        agent.turn.workflow.todos = vec![TodoItem {
            id: "done".into(),
            content: "done".into(),
            status: TodoStatus::Completed,
        }];
        assert!(
            !agent
                .continue_after_no_tool_reply(
                    &mut |_| std::future::ready(Ok(())),
                    &mut continuation_count
                )
                .await
                .expect("completed todos should stop")
        );

        agent.turn.workflow.todos = vec![TodoItem {
            id: "blocked".into(),
            content: "blocked".into(),
            status: TodoStatus::Blocked,
        }];
        assert!(
            !agent
                .continue_after_no_tool_reply(
                    &mut |_| std::future::ready(Ok(())),
                    &mut continuation_count
                )
                .await
                .expect("blocked todos should stop")
        );
    }

    #[tokio::test]
    async fn continuation_bound_is_runtime_enforced() {
        let mut agent = test_agent();
        agent.turn.workflow.auto_continue = AutoContinueState {
            enabled: true,
            max_continuations: 1,
        };
        agent.turn.workflow.todos = vec![TodoItem {
            id: "t1".into(),
            content: "still pending".into(),
            status: TodoStatus::Pending,
        }];
        let mut continuation_count = 1;

        let error = agent
            .continue_after_no_tool_reply(
                &mut |_| std::future::ready(Ok(())),
                &mut continuation_count,
            )
            .await
            .expect_err("limit should fail fast");

        assert!(error.to_string().contains("auto-continue limit reached"));
        assert_eq!(continuation_count, 1);
    }

    #[test]
    fn engineering_turn_prelude_adds_workflow_context_and_validation_reminder() {
        let client = Client::with_config(
            OpenAIConfig::new()
                .with_api_base("https://api.openai.com/v1")
                .with_api_key("test"),
        );
        let mut agent = Agent::new(client, "m1", 1, 1);

        let turn_prelude =
            agent.prepare_turn_prelude("Implement the fix in src/agent.rs and run cargo test.");

        assert_eq!(agent.current_turn().intent, TurnIntent::Engineering);
        assert_eq!(agent.current_turn().directive, ExecutionDirective::None);
        assert_eq!(turn_prelude.len(), agent.prelude.len() + 2);
        let runtime_message = &turn_prelude[turn_prelude.len() - 2];
        assert_eq!(
            runtime_message.role,
            crate::request_builder::PromptRole::Developer
        );
        assert!(runtime_message.text.contains("Runtime context"));
        assert!(runtime_message.text.contains("Current date:"));
        assert!(runtime_message.text.contains("Timezone:"));
        assert!(!runtime_message.text.contains("Current time:"));
        let workflow_message = &turn_prelude[turn_prelude.len() - 1];
        assert_eq!(
            workflow_message.role,
            crate::request_builder::PromptRole::Developer
        );
        assert!(workflow_message.text.contains("engineering workflow task"));
        assert!(workflow_message.text.contains("Delegate bounded work"));
        assert!(workflow_message.text.contains("context hygiene"));
        assert!(workflow_message.text.contains("targeted validation"));
    }

    #[test]
    fn lightweight_turn_prelude_adds_only_runtime_context() {
        let client = Client::with_config(
            OpenAIConfig::new()
                .with_api_base("https://api.openai.com/v1")
                .with_api_key("test"),
        );
        let mut agent = Agent::new(client, "m1", 1, 1);

        let turn_prelude = agent.prepare_turn_prelude("Summarize what this tool does.");

        assert_eq!(agent.current_turn().intent, TurnIntent::Lightweight);
        assert_eq!(turn_prelude.len(), agent.prelude.len() + 1);
        assert_eq!(
            &turn_prelude[..agent.prelude.len()],
            agent.prelude.as_slice()
        );
        let runtime_message = turn_prelude.last().expect("runtime context present");
        assert_eq!(
            runtime_message.role,
            crate::request_builder::PromptRole::Developer
        );
        assert!(runtime_message.text.contains("Runtime context"));
        assert!(runtime_message.text.contains("Current date:"));
        assert!(runtime_message.text.contains("Timezone:"));
        assert!(!runtime_message.text.contains("Current time:"));
    }

    #[test]
    fn normalize_session_title_trims_and_strips_wrapping_quotes() {
        assert_eq!(
            normalize_session_title("  \"Fix startup crash in CI\"  ").expect("normalize title"),
            "Fix startup crash in CI"
        );
        assert_eq!(
            normalize_session_title("`Debug flaky transcript tests`\nextra")
                .expect("normalize title"),
            "Debug flaky transcript tests"
        );
    }

    #[test]
    fn session_title_agent_has_no_tools_or_history() {
        let mut agent = test_agent();
        agent.restore_transcript_messages(vec![ConversationMessage {
            role: ConversationRole::User,
            content: "existing conversation".into(),
        }]);
        let title_agent = agent.session_title_agent();

        assert!(title_agent.history.is_empty());
        assert!(title_agent.runtime_snapshot.evidence.is_empty());
        assert!(title_agent.tools.specs().is_empty());
        assert_eq!(title_agent.model(), agent.model());
    }

    #[test]
    fn turn_prelude_injects_skill_cards_without_skill_body() {
        let mut agent = test_agent();
        agent
            .register_skill_registry(test_skill_registry())
            .expect("register skill registry");

        let turn_prelude = agent.prepare_turn_prelude("Summarize the available tools.");
        let skill_message = turn_prelude
            .iter()
            .find(|message| message.text.contains("Available local skills:"))
            .expect("skill prelude message present");

        assert!(
            skill_message
                .text
                .contains("Load relevant skills with the `skill` tool when needed.")
        );
        assert!(
            skill_message
                .text
                .contains("rust-audit — Inspect Rust code")
        );
        assert!(skill_message.text.contains("source: .letcode/skills"));
        assert!(
            !skill_message
                .text
                .contains("/workspace/.letcode/skills/rust-audit/SKILL.md")
        );
        assert!(!skill_message.text.contains("# Private body"));
        assert!(
            skill_message
                .text
                .contains("Skills do not change permissions or expand tool scope.")
        );
    }

    #[test]
    fn register_skill_registry_registers_skill_resource_tools() {
        let mut agent = test_agent();
        agent
            .register_skill_registry(test_skill_registry())
            .expect("register skill registry");

        let specs = agent.tool_definitions();
        for name in ["skill", "skill__resource_list", "skill__resource_read"] {
            assert!(
                specs.iter().any(|spec| spec.name == name),
                "{name} should be registered"
            );
        }
    }

    #[test]
    fn empty_skill_registry_does_not_register_skill_tool_or_prelude() {
        let mut agent = test_agent();
        agent
            .register_skill_registry(Arc::new(SkillRegistry::default()))
            .expect("register empty skill registry");

        assert!(!agent.tool_definitions().iter().any(|spec| {
            matches!(
                spec.name.as_str(),
                "skill" | "skill__resource_list" | "skill__resource_read"
            )
        }));
        let turn_prelude = agent.prepare_turn_prelude("Summarize this project.");
        assert!(
            !turn_prelude
                .iter()
                .any(|message| message.text.contains("Available local skills:"))
        );
    }

    #[test]
    fn runtime_context_message_contains_date_and_timezone_only() {
        let message = runtime_context_message_from_parts("2026-06-18", "Asia/Shanghai");

        assert_eq!(message.role, crate::request_builder::PromptRole::Developer);
        assert!(message.text.contains("Runtime context:"));
        assert!(message.text.contains("Current date: 2026-06-18"));
        assert!(message.text.contains("Timezone: Asia/Shanghai"));
        assert!(!message.text.contains("Current time:"));
        assert!(!message.text.contains("09:43"));
    }

    #[test]
    fn utc_date_from_unix_days_formats_calendar_dates() {
        assert_eq!(utc_date_from_unix_days(0), "1970-01-01");
        assert_eq!(utc_date_from_unix_days(20_622), "2026-06-18");
    }

    #[test]
    fn detects_explicit_execution_directives() {
        assert_eq!(
            detect_execution_directive("Read-only: inspect src/permission.rs and summarize it."),
            ExecutionDirective::ReadOnly
        );
        assert_eq!(
            detect_execution_directive("Plan only. Do not edit anything yet."),
            ExecutionDirective::PlanOnly
        );
        assert_eq!(
            detect_execution_directive("Analyze only and explain the failure."),
            ExecutionDirective::AnalyzeOnly
        );
        assert_eq!(
            detect_execution_directive("Please investigate, but do not edit files."),
            ExecutionDirective::DoNotEdit
        );
    }

    #[tokio::test]
    async fn execute_tool_call_blocks_write_tools_under_read_only_directive() {
        let client = Client::with_config(
            OpenAIConfig::new()
                .with_api_base("https://api.openai.com/v1")
                .with_api_key("test"),
        );
        let mut agent = Agent::new(client, "m1", 1, 1);
        agent.turn = TurnRuntimeState::new(
            1,
            WorkflowTurnState::from_user_input("Read-only: inspect and report only."),
        );

        let call = HistoryToolCall {
            call_id: "call-1".into(),
            name: "fs__write".into(),
            arguments_json: r#"{"path":"a.txt","content":"x"}"#.into(),
        };
        let mut events = Vec::new();

        let record = agent
            .execute_tool_call(
                &call,
                &mut |event| {
                    events.push(event);
                    std::future::ready(Ok(()))
                },
                &mut |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
            )
            .await
            .expect("tool call should complete with visible error");

        assert!(!record.output.ok);
        assert!(
            record
                .output
                .error
                .as_ref()
                .expect("error payload")
                .message
                .contains("read_only directive")
        );
        assert!(matches!(
            events.as_slice(),
            [
                AgentEvent::ToolCallFinished { .. },
                AgentEvent::ToolExecutionSummary(ToolExecutionSummaryEvent {
                    status,
                    rejection: Some(rejection),
                    effect_kind,
                    ..
                })
            ] if status == "rejected"
                    && rejection == "directive_blocked"
                    && effect_kind == "diagnostic"
        ));
        assert_eq!(record.status, ToolExecutionStatus::Rejected);
        assert_eq!(
            record.rejection,
            Some(ToolExecutionRejection::DirectiveBlocked)
        );
        assert_eq!(record.effects.kind, ToolEffectKind::Diagnostic);
    }

    #[tokio::test]
    async fn execute_tool_call_blocks_non_read_only_commands_under_read_only_directive() {
        let client = Client::with_config(
            OpenAIConfig::new()
                .with_api_base("https://api.openai.com/v1")
                .with_api_key("test"),
        );
        let mut agent = Agent::new(client, "m1", 1, 1);
        agent.turn = TurnRuntimeState::new(
            1,
            WorkflowTurnState::from_user_input("Read only. Analyze and report."),
        );

        let call = HistoryToolCall {
            call_id: "call-2".into(),
            name: "shell__exec".into(),
            arguments_json: r#"{"command":"cargo test permission::tests"}"#.into(),
        };

        let record = agent
            .execute_tool_call(&call, &mut |_| std::future::ready(Ok(())), &mut |_| {
                std::future::ready(Ok(PermissionApproval::AllowOnce))
            })
            .await
            .expect("tool call should complete with visible error");

        assert!(!record.output.ok);
        assert!(
            record
                .output
                .error
                .as_ref()
                .expect("error payload")
                .message
                .contains("not read-only compatible")
        );
    }

    #[tokio::test]
    async fn execute_tool_call_emits_finished_event_for_policy_denial() {
        let client = Client::with_config(
            OpenAIConfig::new()
                .with_api_base("https://api.openai.com/v1")
                .with_api_key("test"),
        );
        let mut agent = Agent::new(client, "m1", 1, 1);
        let call = HistoryToolCall {
            call_id: "call-denied".into(),
            name: "shell__exec".into(),
            arguments_json: r#"{"command":"rm -rf target"}"#.into(),
        };
        let mut events = Vec::new();

        let record = agent
            .execute_tool_call(
                &call,
                &mut |event| {
                    events.push(event);
                    std::future::ready(Ok(()))
                },
                &mut |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
            )
            .await
            .expect("policy denial should be reported as tool output");

        assert!(!record.output.ok);
        assert!(matches!(
            events.as_slice(),
            [
                AgentEvent::ToolCallFinished { ok: false, .. },
                AgentEvent::ToolExecutionSummary(ToolExecutionSummaryEvent {
                    status,
                    rejection: Some(rejection),
                    effect_kind,
                    ..
                })
            ] if status == "rejected"
                    && rejection == "permission_denied_by_policy"
                    && effect_kind == "diagnostic"
        ));
        assert_eq!(record.status, ToolExecutionStatus::Rejected);
        assert_eq!(
            record.rejection,
            Some(ToolExecutionRejection::PermissionDeniedByPolicy)
        );
        assert_eq!(record.effects.kind, ToolEffectKind::Diagnostic);
    }

    #[tokio::test]
    async fn execute_tool_call_invalid_json_emits_finished_event_and_records_rejection() {
        let mut agent = test_agent();
        let call = test_tool_call("fs__write", r#"{"path":"a.txt","content": }"#);
        let mut events = Vec::new();

        let record = agent
            .execute_tool_call(
                &call,
                &mut |event| {
                    events.push(event);
                    std::future::ready(Ok(()))
                },
                &mut |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
            )
            .await
            .expect("invalid json should still produce a record");

        assert_eq!(record.status, ToolExecutionStatus::Rejected);
        assert_eq!(
            record.rejection,
            Some(ToolExecutionRejection::InvalidJsonArguments)
        );
        assert!(!record.output.ok);
        assert_eq!(record.arguments, None);
        assert_eq!(record.effects.kind, ToolEffectKind::Diagnostic);
        assert!(matches!(
            events.as_slice(),
            [
                AgentEvent::ToolCallFinished { ok: false, .. },
                AgentEvent::ToolExecutionSummary(ToolExecutionSummaryEvent {
                    status,
                    rejection: Some(rejection),
                    effect_kind,
                    ..
                })
            ] if status == "rejected"
                && rejection == "invalid_json_arguments"
                && effect_kind == "diagnostic"
        ));
    }

    #[tokio::test]
    async fn audit_event_failures_do_not_fail_tool_execution() {
        let mut agent = test_agent();
        let call = test_tool_call("fs__write", r#"{"path":"a.txt","content": }"#);
        let mut event_count = 0;

        let record = agent
            .execute_tool_call(
                &call,
                &mut |event| {
                    assert!(matches!(
                        event,
                        AgentEvent::ToolCallFinished { .. } | AgentEvent::ToolExecutionSummary(_)
                    ));
                    event_count += 1;
                    if matches!(event, AgentEvent::ToolExecutionSummary(_)) {
                        std::future::ready(Err(anyhow!("audit sink failed")))
                    } else {
                        std::future::ready(Ok(()))
                    }
                },
                &mut |_| std::future::ready(Ok(PermissionApproval::AllowOnce)),
            )
            .await
            .expect("audit failure should not fail tool execution");

        assert_eq!(event_count, 2);
        assert_eq!(record.status, ToolExecutionStatus::Rejected);
        assert_eq!(
            record.rejection,
            Some(ToolExecutionRejection::InvalidJsonArguments)
        );
    }

    #[test]
    fn pending_validation_advisory_only_emits_for_write_without_validation() {
        let mut agent = test_agent();
        assert!(agent.pending_validation_advisory().is_none());

        agent.turn.counters.write_effects = 1;
        let advisory = agent
            .pending_validation_advisory()
            .expect("write without validation should emit advisory");
        assert_eq!(advisory.write_effects, 1);
        assert_eq!(advisory.validation_effects, 0);
        assert_eq!(advisory.failed_validation_effects, 0);
        assert!(advisory.message.contains("without running validation"));

        agent.turn.counters.failed_validation_effects = 1;
        let advisory = agent
            .pending_validation_advisory()
            .expect("failed validation should emit advisory");
        assert_eq!(advisory.write_effects, 1);
        assert_eq!(advisory.validation_effects, 0);
        assert_eq!(advisory.failed_validation_effects, 1);
        assert!(advisory.message.contains("validation ran but failed"));

        agent.turn.counters.validation_effects = 1;
        let advisory = agent
            .pending_validation_advisory()
            .expect("failed validation should continue to emit advisory");
        assert_eq!(advisory.validation_effects, 1);
        assert_eq!(advisory.failed_validation_effects, 1);
        assert!(advisory.message.contains("validation ran but failed"));
    }

    #[test]
    fn pending_validation_advisory_includes_child_write_and_validation_failures() {
        let mut agent = test_agent();
        agent.turn.counters.child_write_effects = 2;
        let advisory = agent
            .pending_validation_advisory()
            .expect("child writes without validation should emit advisory");
        assert_eq!(advisory.write_effects, 2);
        assert_eq!(advisory.validation_effects, 0);
        assert!(advisory.message.contains("delegated child work"));

        agent.turn.counters.child_validation_effects = 1;
        agent.turn.counters.child_failed_validation_effects = 1;
        let advisory = agent
            .pending_validation_advisory()
            .expect("child validation failures should emit advisory");
        assert_eq!(advisory.failed_validation_effects, 1);
        assert!(advisory.message.contains("validation failed"));
    }

    #[test]
    fn prepare_turn_prelude_includes_unreconciled_subagent_results() {
        let mut agent = test_agent();
        agent
            .add_evidence(
                EvidenceDraft {
                    id: Some("ev-1".into()),
                    evidence_kind: crate::evidence::EvidenceKind::Decision,
                    title: "subagent result".into(),
                    summary: "child completed".into(),
                    detail: Some(
                        serde_json::to_string(&crate::subagent::StructuredSubagentResult {
                            status: "completed".into(),
                            summary: "child completed".into(),
                            malformed: false,
                            findings: vec![],
                            files_read: vec![],
                            files_changed: vec![],
                            commands_run: vec![],
                            validation: vec![],
                            blockers: vec![],
                            next_steps: vec![],
                            run_id: "run-1".into(),
                            child_session_id: "child-1".into(),
                            raw_excerpt: None,
                        })
                        .expect("serialize structured result"),
                    ),
                    source: EvidenceSource::Subagent {
                        run_id: "run-1".into(),
                        child_session_id: "child-1".into(),
                        source_session_id: "child-1".into(),
                        parent_tool: "agent__explore".into(),
                        parent_turn_id: Some("turn-1".into()),
                        parent_session_id: None,
                    },
                    tags: vec![
                        "explorer".into(),
                        "subagent_result".into(),
                        "unreconciled".into(),
                    ],
                }
                .into_record("ev-1".into(), 1, 0)
                .expect("build evidence"),
            )
            .expect("add evidence");

        let prelude = agent.prepare_turn_prelude("Implement next step");
        assert!(prelude.iter().any(|message| {
            message.text.contains("Pending child subagent results")
                && message.text.contains("agent__reconcile")
                && message.text.contains("run-1")
                && message.text.contains("child completed")
        }));
    }

    #[test]
    fn pending_subagent_jobs_clear_after_live_reconciliation_evidence() {
        let mut agent = test_agent();
        agent
            .add_evidence(
                EvidenceDraft {
                    id: Some("ev-result".into()),
                    evidence_kind: crate::evidence::EvidenceKind::Decision,
                    title: "subagent result".into(),
                    summary: "child completed".into(),
                    detail: Some(
                        serde_json::to_string(&crate::subagent::StructuredSubagentResult {
                            status: "completed".into(),
                            summary: "child completed".into(),
                            malformed: false,
                            findings: vec![],
                            files_read: vec![],
                            files_changed: vec![],
                            commands_run: vec![],
                            validation: vec![],
                            blockers: vec![],
                            next_steps: vec![],
                            run_id: "run-1".into(),
                            child_session_id: "child-1".into(),
                            raw_excerpt: None,
                        })
                        .expect("serialize structured result"),
                    ),
                    source: EvidenceSource::Subagent {
                        run_id: "run-1".into(),
                        child_session_id: "child-1".into(),
                        source_session_id: "child-1".into(),
                        parent_tool: "agent__explore".into(),
                        parent_turn_id: Some("turn-1".into()),
                        parent_session_id: None,
                    },
                    tags: vec![
                        "explorer".into(),
                        "subagent_result".into(),
                        "unreconciled".into(),
                    ],
                }
                .into_record("ev-result".into(), 1, 0)
                .expect("build evidence"),
            )
            .expect("add result evidence");
        assert_eq!(agent.pending_subagent_jobs().len(), 1);

        let record = ToolExecutionRecord::new(
            &test_tool_call(
                tool_names::TOOL_AGENT_RECONCILE,
                r#"{"run_id":"run-1","child_session_id":"child-1","agent_name":"explorer","decision":"accepted","summary":"accepted child result"}"#,
            ),
            Some(json!({
                "run_id": "run-1",
                "child_session_id": "child-1",
                "agent_name": "explorer",
                "decision": "accepted",
                "summary": "accepted child result"
            })),
            crate::permission::ToolPermissionClass::Preview,
            ExecutionDirective::None,
            ToolExecutionStatus::Executed,
            None,
            ToolResult::ok(
                tool_names::TOOL_AGENT_RECONCILE,
                json!({
                    "run_id": "run-1",
                    "child_session_id": "child-1",
                    "agent_name": "explorer",
                    "decision": "accepted",
                    "summary": "accepted child result",
                    "reconciled": true,
                    "pending_recording": true
                }),
            ),
        );
        agent.record_tool_effects(&record);
        let evidence = agent
            .remember_tool_evidence(&record)
            .expect("record live reconciliation evidence");
        assert!(
            evidence
                .tags
                .iter()
                .any(|tag| tag == "subagent_reconciliation")
        );
        assert!(evidence.tags.iter().any(|tag| tag == "reconciled"));
        assert_eq!(agent.pending_subagent_jobs().len(), 0);
    }

    #[test]
    fn default_prelude_and_engineering_guidance_frame_non_trivial_work_as_orchestration() {
        assert!(DEFAULT_AGENT_PRELUDE.contains("workflow manager first"));
        assert!(
            DEFAULT_AGENT_PRELUDE
                .contains("Direct execution is for trivial, single-file, clearly bounded work")
        );
        assert!(ENGINEERING_WORKFLOW_PRELUDE.contains("specialist lane is needed"));
        assert!(ENGINEERING_WORKFLOW_PRELUDE.contains("explorer for broad or unknown code search"));
        assert!(ENGINEERING_WORKFLOW_PRELUDE.contains("prefer completed or reconciled sessions"));
        assert!(ENGINEERING_WORKFLOW_PRELUDE.contains("Never reuse cancelled or errored sessions"));

        let mut agent = test_agent();
        let prelude = agent.prepare_turn_prelude("Implement a non-trivial feature with validation");
        assert!(
            prelude
                .iter()
                .any(|message| message.text.contains("workflow manager first"))
        );
        assert!(
            prelude
                .iter()
                .any(|message| message.text.contains("specialist lane is needed"))
        );
        assert!(prelude.iter().any(|message| {
            message
                .text
                .contains("explorer for broad or unknown code search")
        }));
    }

    #[tokio::test]
    async fn finalization_does_not_auto_reconcile_unreconciled_subagent_jobs() {
        let mut agent = test_agent();
        agent.prepare_turn_prelude("Follow up on child work");
        agent
            .add_evidence(
                EvidenceDraft {
                    id: Some("ev-1".into()),
                    evidence_kind: crate::evidence::EvidenceKind::Decision,
                    title: "subagent result".into(),
                    summary: "child completed".into(),
                    detail: Some(
                        serde_json::to_string(&crate::subagent::StructuredSubagentResult {
                            status: "completed".into(),
                            summary: "child completed".into(),
                            malformed: false,
                            findings: vec![],
                            files_read: vec![],
                            files_changed: vec![],
                            commands_run: vec![],
                            validation: vec![],
                            blockers: vec![],
                            next_steps: vec![],
                            run_id: "run-1".into(),
                            child_session_id: "child-1".into(),
                            raw_excerpt: None,
                        })
                        .expect("serialize structured result"),
                    ),
                    source: EvidenceSource::Subagent {
                        run_id: "run-1".into(),
                        child_session_id: "child-1".into(),
                        source_session_id: "child-1".into(),
                        parent_tool: "agent__explore".into(),
                        parent_turn_id: Some("turn-1".into()),
                        parent_session_id: None,
                    },
                    tags: vec![
                        "explorer".into(),
                        "subagent_result".into(),
                        "unreconciled".into(),
                    ],
                }
                .into_record("ev-1".into(), 1, 0)
                .expect("build evidence"),
            )
            .expect("add evidence");

        let mut events = Vec::new();
        let continued = agent
            .continue_or_finalize_no_tool_reply(
                &mut |event| {
                    events.push(event);
                    std::future::ready(Ok(()))
                },
                0,
                &mut 0,
            )
            .await
            .expect("finalization succeeds");

        assert!(!continued);
        assert!(!events.iter().any(|event| matches!(
            event,
            AgentEvent::EvidenceRecorded(record)
                if record.tags.iter().any(|tag| tag == "subagent_reconciliation")
        )));
        assert_eq!(agent.pending_subagent_jobs().len(), 1);
    }

    #[test]
    fn child_validation_classification_ignores_not_run_and_counts_object_failures() {
        let (ran, failed) = classify_child_validation_entries(&[
            "cargo test not_run".into(),
            "cargo fmt passed".into(),
            "cargo test failed".into(),
        ]);

        assert_eq!(ran, 2);
        assert_eq!(failed, 1);
    }

    #[test]
    fn turn_lifecycle_events_capture_expected_snapshot_fields() {
        let client = Client::with_config(
            OpenAIConfig::new()
                .with_api_base("https://api.openai.com/v1")
                .with_api_key("test"),
        );
        let mut agent = Agent::new(client, "m1", 1, 1);

        agent.prepare_turn_prelude("Implement fix in src/agent.rs and run cargo test transcript");
        agent.turn.counters.write_effects = 2;
        agent.turn.counters.validation_effects = 1;
        agent.turn.counters.failed_validation_effects = 0;

        let started = agent.turn_started_event();
        assert_eq!(started.turn_id, 1);
        assert_eq!(started.intent, "engineering");
        assert_eq!(started.directive, "none");
        assert_eq!(started.validation_reminder, "targeted");

        let finalized = agent.turn_finalized_event("completed", 3, 1, true);
        assert_eq!(finalized.turn_id, 1);
        assert_eq!(finalized.outcome, "completed");
        assert_eq!(finalized.tool_call_count, 3);
        assert_eq!(finalized.continuation_count, 1);
        assert_eq!(finalized.write_effects, 2);
        assert_eq!(finalized.validation_effects, 1);
        assert!(finalized.validation_advisory_emitted);
    }

    #[test]
    fn tool_execution_summary_event_omits_full_output_and_captures_audit_fields() {
        let mut agent = test_agent();
        agent.prepare_turn_prelude("Implement fix");
        let record = ToolExecutionRecord::new(
            &test_tool_call("shell__exec", r#"{"command":"cargo test transcript"}"#),
            Some(json!({"command": "cargo test transcript", "path": "src/agent.rs"})),
            crate::permission::ToolPermissionClass::Command,
            ExecutionDirective::None,
            ToolExecutionStatus::Executed,
            None,
            ToolResult::ok(
                "shell__exec",
                json!({"command": "cargo test transcript", "status": 0, "stdout": "lots"}),
            ),
        );

        let summary = agent.tool_execution_summary_event(&record);
        assert_eq!(summary.turn_id, 1);
        assert_eq!(summary.call_id, "call-shell__exec");
        assert_eq!(summary.name, "shell__exec");
        assert_eq!(summary.status, "executed");
        assert_eq!(summary.effect_kind, "validation");
        assert_eq!(summary.primary_path.as_deref(), Some("src/agent.rs"));
        assert_eq!(summary.command.as_deref(), Some("cargo test transcript"));
        assert_eq!(summary.rejection, None);
    }
}

fn default_agent_prelude() -> Vec<PromptMessage> {
    vec![PromptMessage::developer(DEFAULT_AGENT_PRELUDE)]
}

fn runtime_context_message() -> PromptMessage {
    runtime_context_message_from_parts(&current_date_label(), &timezone_label())
}

fn runtime_context_message_from_parts(date: &str, timezone: &str) -> PromptMessage {
    PromptMessage::developer_with_origin(
        format!("Runtime context:\n- Current date: {date}\n- Timezone: {timezone}"),
        PromptMessageOrigin::RuntimeClock,
    )
}

fn current_date_label() -> String {
    command_output("date", &["+%Y-%m-%d"]).unwrap_or_else(current_utc_date_label)
}

fn timezone_label() -> String {
    std::env::var("TZ")
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
        .or_else(|| command_output("date", &["+%Z"]))
        .unwrap_or_else(|| "local system timezone".into())
}

fn command_output(command: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(command).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }

    let text = String::from_utf8(output.stdout).ok()?;
    let trimmed = text.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_owned())
    }
}

fn current_utc_date_label() -> String {
    let days = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| (duration.as_secs() / 86_400) as i64)
        .unwrap_or(0);
    utc_date_from_unix_days(days)
}

fn utc_date_from_unix_days(days: i64) -> String {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = mp + if mp < 10 { 3 } else { -9 };
    let year = y + if month <= 2 { 1 } else { 0 };

    format!("{year:04}-{month:02}-{day:02}")
}

fn reasoning_summary_text(item: &OutputItem) -> String {
    match item {
        OutputItem::Reasoning(reasoning) => reasoning
            .summary
            .iter()
            .map(|part| match part {
                async_openai::types::responses::SummaryPart::SummaryText(content) => {
                    content.text.clone()
                }
            })
            .filter(|text| !text.trim().is_empty())
            .collect::<Vec<_>>()
            .join("\n\n"),
        _ => String::new(),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum StreamTextPart {
    Visible(String),
    ReasoningDelta { item_id: String, delta: String },
    ReasoningDone { item_id: String, text: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InlineReasoningMode {
    Visible,
    Reasoning,
}

#[derive(Debug, Clone)]
struct InlineReasoningExtractor {
    item_id: String,
    mode: InlineReasoningMode,
    buffer: String,
    reasoning_text: String,
}

impl InlineReasoningExtractor {
    fn new(item_id: impl Into<String>) -> Self {
        Self {
            item_id: item_id.into(),
            mode: InlineReasoningMode::Visible,
            buffer: String::new(),
            reasoning_text: String::new(),
        }
    }

    fn push(&mut self, text: &str) -> Vec<StreamTextPart> {
        self.buffer.push_str(text);
        self.drain(false)
    }

    fn finish(&mut self) -> Vec<StreamTextPart> {
        self.drain(true)
    }

    fn drain(&mut self, finishing: bool) -> Vec<StreamTextPart> {
        let mut parts = Vec::new();

        loop {
            match self.mode {
                InlineReasoningMode::Visible => {
                    if let Some((start, len)) = find_open_reasoning_tag(&self.buffer) {
                        let visible = self.buffer[..start].to_string();
                        if !visible.is_empty() {
                            parts.push(StreamTextPart::Visible(visible));
                        }
                        self.buffer.drain(..start + len);
                        self.mode = InlineReasoningMode::Reasoning;
                        continue;
                    }

                    let emit_len = if finishing {
                        self.buffer.len()
                    } else {
                        safe_emit_len_without_partial_tag(&self.buffer, OPEN_REASONING_TAGS)
                    };
                    if emit_len == 0 {
                        break;
                    }
                    let visible = self.buffer[..emit_len].to_string();
                    self.buffer.drain(..emit_len);
                    parts.push(StreamTextPart::Visible(visible));
                }
                InlineReasoningMode::Reasoning => {
                    if let Some((start, len)) = find_close_reasoning_tag(&self.buffer) {
                        let delta = self.buffer[..start].to_string();
                        if !delta.is_empty() {
                            self.reasoning_text.push_str(&delta);
                            parts.push(StreamTextPart::ReasoningDelta {
                                item_id: self.item_id.clone(),
                                delta,
                            });
                        }
                        self.buffer.drain(..start + len);
                        parts.push(StreamTextPart::ReasoningDone {
                            item_id: self.item_id.clone(),
                            text: self.reasoning_text.clone(),
                        });
                        self.mode = InlineReasoningMode::Visible;
                        continue;
                    }

                    let emit_len = if finishing {
                        self.buffer.len()
                    } else {
                        safe_emit_len_without_partial_tag(&self.buffer, CLOSE_REASONING_TAGS)
                    };
                    if emit_len == 0 {
                        break;
                    }
                    let delta = self.buffer[..emit_len].to_string();
                    self.buffer.drain(..emit_len);
                    self.reasoning_text.push_str(&delta);
                    parts.push(StreamTextPart::ReasoningDelta {
                        item_id: self.item_id.clone(),
                        delta,
                    });
                }
            }
        }

        if finishing && matches!(self.mode, InlineReasoningMode::Reasoning) {
            parts.push(StreamTextPart::ReasoningDone {
                item_id: self.item_id.clone(),
                text: self.reasoning_text.clone(),
            });
            self.mode = InlineReasoningMode::Visible;
        }

        parts
    }
}

const OPEN_REASONING_TAGS: &[&str] = &["<think>", "<thinking>"];
const CLOSE_REASONING_TAGS: &[&str] = &["</think>", "</thinking>"];

fn find_open_reasoning_tag(text: &str) -> Option<(usize, usize)> {
    find_earliest_tag(text, OPEN_REASONING_TAGS)
}

fn find_close_reasoning_tag(text: &str) -> Option<(usize, usize)> {
    find_earliest_tag(text, CLOSE_REASONING_TAGS)
}

fn find_earliest_tag(text: &str, tags: &[&str]) -> Option<(usize, usize)> {
    tags.iter()
        .filter_map(|tag| text.find(tag).map(|index| (index, tag.len())))
        .min_by_key(|(index, _)| *index)
}

fn safe_emit_len_without_partial_tag(text: &str, tags: &[&str]) -> usize {
    for hold in (1..=max_tag_len(tags).saturating_sub(1)).rev() {
        if text.len() >= hold {
            let suffix_start = next_char_boundary(text, text.len() - hold);
            let suffix = &text[suffix_start..];
            if tags.iter().any(|tag| tag.starts_with(suffix)) {
                return suffix_start;
            }
        }
    }
    text.len()
}

fn max_tag_len(tags: &[&str]) -> usize {
    tags.iter().map(|tag| tag.len()).max().unwrap_or(0)
}

fn next_char_boundary(text: &str, index: usize) -> usize {
    if text.is_char_boundary(index) {
        return index;
    }
    text.char_indices()
        .map(|(i, _)| i)
        .find(|i| *i > index)
        .unwrap_or(text.len())
}

fn validate_chat_finish_reasons(reasons: &[FinishReason], has_tool_calls: bool) -> Result<()> {
    if reasons.is_empty() {
        return Err(anyhow!(
            "completions stream ended without finish_reason; cannot determine completion status"
        ));
    }

    for reason in reasons {
        match (reason, has_tool_calls) {
            (FinishReason::Stop, false) => {}
            (FinishReason::ToolCalls, true) | (FinishReason::FunctionCall, true) => {}
            (FinishReason::Length, _) => {
                return Err(anyhow!(
                    "completions response incomplete: finish_reason=length"
                ));
            }
            (FinishReason::ContentFilter, _) => {
                return Err(anyhow!(
                    "completions response filtered: finish_reason=content_filter"
                ));
            }
            (reason, _) => {
                return Err(anyhow!(
                    "unexpected completions finish_reason {:?} for {} response",
                    reason,
                    if has_tool_calls { "tool-call" } else { "text" }
                ));
            }
        }
    }

    Ok(())
}

fn validate_chat_tool_calls(tool_calls: &[ChatCompletionMessageToolCall]) -> Result<()> {
    for (index, call) in tool_calls.iter().enumerate() {
        if call.id.trim().is_empty() {
            return Err(anyhow!(
                "invalid completions tool call at index {index}: missing id"
            ));
        }
        if call.function.name.trim().is_empty() {
            return Err(anyhow!(
                "invalid completions tool call at index {index}: missing function name"
            ));
        }
        if call.function.arguments.trim().is_empty() {
            return Err(anyhow!(
                "invalid completions tool call at index {index}: missing function arguments"
            ));
        }
    }

    Ok(())
}

fn compact_indexed_chat_tool_calls(
    tool_calls: BTreeMap<usize, ChatCompletionMessageToolCall>,
) -> Vec<ChatCompletionMessageToolCall> {
    tool_calls.into_values().collect()
}

fn is_context_overflow_error(error: &anyhow::Error) -> bool {
    error
        .chain()
        .any(|cause| is_context_overflow_message(&cause.to_string()))
}

fn is_context_overflow_message(message: &str) -> bool {
    let normalized = message.to_ascii_lowercase();
    [
        "maximum context length",
        "context length exceeded",
        "context window exceeded",
        "context overflow",
        "prompt is too long",
        "input is too long",
        "reduce the length of the messages",
        "requested too many tokens",
        "context_window_exceeded",
    ]
    .iter()
    .any(|needle| normalized.contains(needle))
}

fn build_tool_call_name_index(history: &[HistoryItem]) -> HashMap<String, String> {
    let mut call_names = HashMap::new();
    for item in history {
        if let HistoryItem::AssistantToolCalls { calls, .. } = item {
            for call in calls {
                call_names.insert(call.call_id.clone(), call.name.clone());
            }
        }
    }
    call_names
}

async fn emit_tool_call_pending_if_ready<E, Efut>(
    emitted_pending_tool_calls: &mut HashSet<String>,
    call_id: &str,
    name: &str,
    on_event: &mut E,
) -> Result<bool>
where
    E: FnMut(AgentEvent) -> Efut,
    Efut: Future<Output = Result<()>>,
{
    if call_id.trim().is_empty() || name.trim().is_empty() {
        return Ok(false);
    }

    if emitted_pending_tool_calls.insert(call_id.to_string()) {
        on_event(AgentEvent::ToolCallPending {
            call_id: call_id.to_string(),
            name: name.to_string(),
        })
        .await?;
        return Ok(true);
    }

    Ok(false)
}

fn merge_chat_tool_call_chunk(
    tool_calls: &mut BTreeMap<usize, ChatCompletionMessageToolCall>,
    chunk: ChatCompletionMessageToolCallChunk,
) {
    let index = chunk.index as usize;
    let tool_call = tool_calls.entry(index).or_default();
    if let Some(id) = chunk.id.filter(|id| !id.trim().is_empty()) {
        tool_call.id = id;
    }
    if let Some(function) = chunk.function {
        if let Some(name) = function.name.filter(|name| !name.trim().is_empty()) {
            tool_call.function.name = name;
        }
        if let Some(arguments) = function.arguments {
            tool_call.function.arguments.push_str(&arguments);
        }
    }
}
