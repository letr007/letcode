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
    observe_logical_request, rebuild_request_from_plan,
};
use crate::retry::{
    can_retry_attempt, is_retryable_json_deserialize_error, retry_delay, should_retry_http_status,
    should_retry_openai_stream_creation, should_retry_openai_stream_read,
    should_retry_reqwest_error,
};
use crate::runtime_context::{
    FrameVisibility, RuntimeFrame, RuntimeFrameIdSeed, RuntimeFrameKind, RuntimeFrameProvenance,
    RuntimeSnapshot, RuntimeSource, SourceSpan,
};
use crate::skills::{
    SkillCard, SkillRegistry, SkillResourceListTool, SkillResourceReadTool, SkillTool,
    parse_manual_skill_markers, reconcile_loaded_skill_material,
};
use crate::tool::{
    NormalizedSubagentInput, QuestionCallback, QuestionRequest, QuestionResponse,
    ToolExecutionContext, ToolHandler, ToolRegistry, ToolResult,
    external_workspace_access_for_tool, normalize_subagent_input, subagent_parameters_schema,
};
use crate::tool_format::format_tool_call;
use crate::tool_names;
use crate::transcript::{ActiveContextExperiment, ContextScopeState, ROOT_CONTEXT_BRANCH_ID};
use crate::user_content::UserMessageContent;

#[cfg(test)]
use crate::transcript::LogicalCheckpointEventV1;

#[path = "agent/automatic_checkpoint.rs"]
mod automatic_checkpoint;
#[path = "agent/catalog.rs"]
mod catalog;
#[path = "agent/checkpoint_control.rs"]
mod checkpoint_control;
#[path = "agent/compaction.rs"]
mod compaction;
#[path = "agent/events.rs"]
mod events;
#[path = "agent/evidence_memory.rs"]
mod evidence_memory;
#[path = "agent/logical_checkpoint.rs"]
mod logical_checkpoint;
#[path = "agent/protocol_stream.rs"]
mod protocol_stream;
#[path = "agent/tool_execution.rs"]
mod tool_execution;
#[path = "agent/workflow_state.rs"]
mod workflow_state;

pub use catalog::{AgentFactory, AgentTemplate, SubagentCapabilityContract};
pub(crate) use catalog::{
    SUBAGENT_CATALOG, agent_name_for_subagent_tool, is_subagent_tool_name,
    subagent_catalog_entry_by_tool_name, subagent_tool_name_for_agent_name,
};
#[allow(unused_imports)]
pub(crate) use catalog::{SubagentCatalogEntry, subagent_catalog_entry_by_agent_name};
pub(crate) use checkpoint_control::LogicalCheckpointRequestOwner;
pub use checkpoint_control::{LogicalCheckpointControl, LogicalCheckpointRequestOutcome};
use checkpoint_control::{LogicalCheckpointControlState, LogicalCheckpointRequestState};
#[allow(unused_imports)]
pub(crate) use checkpoint_control::{LogicalCheckpointLease, LogicalCheckpointRunGuard};
pub use events::{
    AgentEvent, CONTEXT_COMPACTION_DERIVED_COVERAGE_VERSION, CacheUsageReport,
    CompactionAttemptOutcome, CompactionBlocker, CompactionNoProgress, CompactionTrigger,
    ContextCompactionDerivedCoverage, ContextCompactionDerivedCoverageItem,
    ContextCompactionDerivedKind, ContextCompactionEvent, ContextCompactionFrameBinding,
    ContextCompactionSourceSpan, LlmRequestErrorClass, LlmRequestTelemetry,
    LlmRequestTelemetryPhase, ManualCompactionOutcome, ProviderUsageCompleteness,
    TokenUsageEstimate, ToolExecutionSummaryEvent, TurnFinalizedEvent, TurnStartedEvent,
    ValidationAdvisory,
};
pub use workflow_state::{AutoContinueState, TodoItem, TodoStatus};

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

#[derive(Debug, Clone, Default)]
struct LogicalRequestObservationTracker {
    previous: Option<crate::request_builder::LogicalRequestObservation>,
}

/// Ephemeral authority for the active cache prefix. It deliberately stores
/// only process-local identities, never prompt bytes or transcript data.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ActiveEpoch {
    turn_id: u64,
    request_shape_digest: String,
    kernel_identity: Option<String>,
    envelope_identity: Option<String>,
    observation: crate::request_builder::LogicalRequestObservation,
    committed_plan: crate::request_builder::prompt_plan::PromptPlan,
    protocol_frontier_count: usize,
    protocol_prefix_digest: String,
}

#[derive(Debug, Clone)]
struct ActiveEpochPreview {
    epoch: ActiveEpoch,
    build: crate::request_builder::BuildResult,
    transition: ActiveEpochTransition,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ActiveEpochTransition {
    Cold,
    Append { added: usize },
}

fn protocol_prefix_digest(frames: &[crate::protocol_frames::ProtocolFrame]) -> String {
    let mut bytes = Vec::new();
    for frame in frames {
        let encoded = serde_json::to_vec(&(frame.runtime_frame_id, &frame.item))
            .expect("protocol frame identity serializes");
        bytes.extend_from_slice(&(encoded.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&encoded);
    }
    crate::request_builder::sha256_hex(&bytes)
}

fn normalize_prompt_plan(plan: &mut crate::request_builder::prompt_plan::PromptPlan) {
    for (order, segment) in plan.segments.iter_mut().enumerate() {
        segment.order = order as u32;
        segment.source.order = order as u32;
    }
    for contributor in &mut plan.contributors {
        contributor.segment_ids.clear();
    }
    for segment in &plan.segments {
        if let Some(contributor) = plan
            .contributors
            .iter_mut()
            .find(|contributor| contributor.id == segment.contributor_id)
        {
            contributor.segment_ids.push(segment.id.clone());
        }
    }
    plan.contributors
        .retain(|contributor| !contributor.segment_ids.is_empty());
    for (order, contributor) in plan.contributors.iter_mut().enumerate() {
        contributor.order = order as u32;
    }
    plan.recompute_cache_metadata();
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AdjacentRequestObservation {
    lcp_units: usize,
    lcp_bytes: u64,
    lcp_estimated_tokens: u64,
    current_unit_count: usize,
    first_breaker: Option<crate::request_builder::LogicalRequestBreaker>,
    cohort_comparable: bool,
    cohort_changed: bool,
}

impl LogicalRequestObservationTracker {
    fn preview(
        &self,
        current: crate::request_builder::LogicalRequestObservation,
    ) -> AdjacentRequestObservation {
        let current_unit_count = current.units.len();
        let Some(previous) = &self.previous else {
            return AdjacentRequestObservation {
                lcp_units: 0,
                lcp_bytes: 0,
                lcp_estimated_tokens: 0,
                current_unit_count,
                first_breaker: None,
                cohort_comparable: false,
                cohort_changed: false,
            };
        };
        if previous.cohort != current.cohort {
            return AdjacentRequestObservation {
                lcp_units: 0,
                lcp_bytes: 0,
                lcp_estimated_tokens: 0,
                current_unit_count,
                first_breaker: None,
                cohort_comparable: false,
                cohort_changed: true,
            };
        }
        let mut lcp_units = 0;
        let mut lcp_bytes = 0;
        let mut lcp_estimated_tokens = 0;
        for (previous_unit, current_unit) in previous.units.iter().zip(&current.units) {
            if previous_unit.digest != current_unit.digest {
                break;
            }
            lcp_units += 1;
            lcp_bytes += current_unit.byte_count;
            lcp_estimated_tokens += current_unit.estimated_tokens;
        }
        AdjacentRequestObservation {
            lcp_units,
            lcp_bytes,
            lcp_estimated_tokens,
            current_unit_count,
            first_breaker: if lcp_units == previous.units.len() {
                None
            } else if lcp_units == current.units.len() {
                Some(crate::request_builder::LogicalRequestBreaker::RemovedSuffix)
            } else {
                current.units.get(lcp_units).map(|unit| {
                    crate::request_builder::LogicalRequestBreaker::CurrentUnit(unit.category)
                })
            },
            cohort_comparable: true,
            cohort_changed: false,
        }
    }
}

impl LogicalRequestObservationTracker {
    fn commit(&mut self, current: crate::request_builder::LogicalRequestObservation) {
        self.previous = Some(current);
    }
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
Prefer specialized tools over `shell__exec` when they fit: use `fs__*` for workspace file read/list/write, `search__rg` (or other search tools) for text search, `git__*` for git status/diff/log, and edit tools for patches. Reserve `shell__exec` for builds, tests, process control, and commands that truly need a shell pipeline or OS features the specialized tools cannot express. Do not use `cat`/`sed`/`rg`/`find` via shell as a default substitute for those tools.
When running shell commands, treat high-impact or irreversible operations carefully (for example `rm -rf`, force overwrite, bulk delete): do not run them until the target, scope, and impact are clear. For compound commands joined by `&&`, `||`, `;`, or pipes, keep order, failure behavior, and side-effect scope aligned with intent—do not silently chain steps that should be separate human decisions, and do not let composition run side effects early, widen scope, or change meaning (for example, do not turn “inspect history, then decide whether to push” into `git log --oneline -n 10 && git push`). Commands with loops, long-lived listeners, or retry logic must have a clear exit condition, timeout, or resource limit so they cannot hang indefinitely.
Use `memory__recall` selectively when the current task likely overlaps prior investigation, failed approaches, returned context experiments, or files with meaningful history. Prefer filtering by relevant file paths and failed/blocked outcomes when debugging; do not treat recall as a mandatory first step for every task.
For non-trivial work, act like a workflow manager first: decide whether the task needs a specialist lane before you start doing the work yourself. Small tasks may still be handled directly when delegation overhead is not worth it.
Direct execution is for trivial, single-file, clearly bounded work, or when delegation overhead clearly exceeds the benefit. Otherwise, keep a short plan, choose the right specialist or direct path intentionally, reconcile delegated results, and finish with the clearest verified outcome.
The current runtime permits only one active subagent. Delegates do not queue: while one runs, wait for it to finish or cancel it. Historical child sessions are persisted results and navigation references, not live executions to resume.
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
The current runtime permits only one active subagent. Delegates do not queue: while one runs, wait for it to finish or cancel it. Historical child sessions are persisted results and navigation references, not live executions to resume.
Keep delegation controlled: avoid recursive delegation, avoid unnecessary multi-agent orchestration, preserve a clear parent agent narrative, reconcile child results, and surface remaining blockers or targeted validation gaps before you stop."#;
const SESSION_TITLE_PRELUDE: &str = r#"为用户的第一条消息生成简洁的会话标题，准确概括其主题、意图或任务。
将该消息视为待命名的内容，不要把它当作需要回复的对话。
输出应为描述性标题，而不是对用户消息的直接回应。
只返回标题文本。
不要使用引号、项目符号、Markdown、前缀或解释。
保持具体，且不超过 80 个字符。"#;
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
- 相关文件尽量写成“路径 — 作用/状态”的形式。
- 不得输出保留标记 `[retained-facts:v1]`。"#;
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
    #[cfg(test)]
    automatic_checkpoint_policy: automatic_checkpoint::AutoCheckpointPolicy,
    retry_config: RetryConfig,
    tool_timeout_secs: Option<u64>,
    turn: TurnRuntimeState,
    next_turn_id: u64,
    max_iterations: Option<usize>,
    max_tool_calls: Option<usize>,
    context_scope_state: Arc<std::sync::Mutex<ContextScopeState>>,
    runtime_snapshot_provider: Option<RuntimeSnapshotProvider>,
    logical_checkpoint_candidate_provider: Option<LogicalCheckpointCandidateProvider>,
    context_experiment_restore_point: Option<ContextExperimentRestorePoint>,
    logical_checkpoint_control: LogicalCheckpointControl,
    logical_request_observations: LogicalRequestObservationTracker,
    active_epoch: Option<ActiveEpoch>,
    // Summary agents must never recursively compact their own request. This
    // outlives their turn initialization, which replaces `TurnRuntimeState`.
    pressure_compaction_suppressed: bool,
}

#[derive(Debug, Clone)]
struct ContextExperimentRestorePoint {
    scope: ActiveContextExperiment,
    protocol_frames: Vec<crate::protocol_frames::ProtocolFrame>,
    runtime_snapshot: RuntimeSnapshot,
}

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
            #[cfg(test)]
            automatic_checkpoint_policy: parent.automatic_checkpoint_policy,
            retry_config: parent.retry_config.clone(),
            tool_timeout_secs: parent.tool_timeout_secs,
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
            logical_checkpoint_control: LogicalCheckpointControl::disabled(),
            logical_request_observations: LogicalRequestObservationTracker::default(),
            active_epoch: None,
            pressure_compaction_suppressed: false,
        }
    }
}

impl<C: Config> Agent<C> {
    fn preview_final_logical_request(
        &self,
        build: &crate::request_builder::BuildResult,
    ) -> AdjacentRequestObservation {
        self.logical_request_observations
            .preview(crate::request_builder::observe_logical_request(build))
    }

    fn commit_final_logical_request(&mut self, build: &crate::request_builder::BuildResult) {
        self.logical_request_observations
            .commit(crate::request_builder::observe_logical_request(build));
    }

    fn clear_active_epoch(&mut self) {
        self.active_epoch = None;
    }

    /// Pure active-epoch preview. The returned token is committed only after the
    /// prepared telemetry callback succeeded and immediately before transport.
    fn preview_active_epoch(
        &self,
        protocol: ApiProtocol,
        turn_prelude: &[PromptMessage],
        tools: &[crate::request_builder::ToolSpec],
    ) -> Result<ActiveEpochPreview> {
        crate::protocol_frames::validate_history_items_complete(&self.history, None)?;
        let model = self.active_model_metadata();
        let frozen = self.turn.frozen_evidence.as_ref().map(|evidence| {
            crate::request_builder::FrozenEvidence {
                message: evidence.message.clone(),
                selected_ids: evidence.selected_ids.clone(),
            }
        });
        let policy = ProtectedContextPolicy::from_configured_reserve(
            None,
            effective_input_budget_tokens(model.clone(), tools),
        );
        let built = build_request_with_policy(
            RequestBuilderInput {
                protocol,
                model_id: &self.model,
                model: model.clone(),
                prelude: turn_prelude,
                snapshot: &self.runtime_snapshot,
                tools,
            },
            frozen.as_ref(),
            Some(policy),
        )?;
        let kernel_identity = Some(crate::request_builder::provider_unit_prefix_digest(
            &built,
            built.prompt_plan.kernel_end_exclusive,
        ));
        let envelope_identity = Some(crate::request_builder::provider_unit_prefix_digest(
            &built,
            built.prompt_plan.envelope_end_exclusive,
        ));
        let cold = self.active_epoch.as_ref().is_none_or(|previous| {
            previous.turn_id != self.turn.turn_id
                || previous.request_shape_digest
                    != observe_logical_request(&built).cohort.request_shape_digest
                || previous.kernel_identity != kernel_identity
                || previous.envelope_identity != envelope_identity
        });
        let (plan, transition) = if cold {
            (built.prompt_plan.clone(), ActiveEpochTransition::Cold)
        } else {
            let previous = self.active_epoch.as_ref().expect("warm epoch exists");
            ensure!(
                self.protocol_frames.len() >= previous.protocol_frontier_count,
                "active epoch protocol prefix was truncated"
            );
            ensure!(
                protocol_prefix_digest(&self.protocol_frames[..previous.protocol_frontier_count])
                    == previous.protocol_prefix_digest,
                "active epoch protocol prefix was mutated or reordered"
            );
            let suffix = &self.protocol_frames[previous.protocol_frontier_count..];
            crate::protocol_frames::validate_history_items_complete(
                &crate::protocol_frames::history_items_from_frames(suffix),
                None,
            )
            .context("active epoch protocol suffix is incomplete")?;

            let mut used = HashSet::new();
            let mut appended = Vec::with_capacity(suffix.len());
            for frame in suffix {
                let key = frame
                    .runtime_frame_id
                    .map(|id| id.as_u64().to_string())
                    .or_else(|| Some(frame.stable_prompt_key()))
                    .expect("protocol frame key exists");
                let matches = built
                    .prompt_plan
                    .segments
                    .iter()
                    .filter(|segment| segment.source.source_key.as_deref() == Some(key.as_str()))
                    .collect::<Vec<_>>();
                ensure!(
                    matches.len() == 1,
                    "active epoch suffix frame cannot map uniquely to canonical prompt segment"
                );
                ensure!(
                    used.insert(matches[0].id.clone()),
                    "active epoch suffix maps a prompt segment twice"
                );
                appended.push((*matches[0]).clone());
            }
            let mut committed = previous.committed_plan.clone();
            for segment in &mut committed.segments {
                segment.stability =
                    crate::request_builder::prompt_plan::PromptSegmentStability::Stable;
            }
            for segment in &mut appended {
                segment.stability =
                    crate::request_builder::prompt_plan::PromptSegmentStability::Volatile;
            }
            committed.segments.extend(appended);
            normalize_prompt_plan(&mut committed);
            (
                committed,
                ActiveEpochTransition::Append {
                    added: suffix.len(),
                },
            )
        };
        let rebuilt = rebuild_request_from_plan(&built, model, tools, plan)?;
        let observation = observe_logical_request(&rebuilt);
        if let Some(previous) = &self.active_epoch
            && matches!(transition, ActiveEpochTransition::Append { .. })
        {
            ensure!(
                observation.units.len() >= previous.observation.units.len()
                    && observation.units[..previous.observation.units.len()]
                        == previous.observation.units[..],
                "active epoch committed provider units are not an exact prefix"
            );
            ensure!(
                observation.cohort.request_shape_digest == previous.request_shape_digest,
                "active epoch request shape changed while rebuilding"
            );
        }
        let committed_plan = rebuilt.prompt_plan.clone();
        Ok(ActiveEpochPreview {
            build: rebuilt,
            epoch: ActiveEpoch {
                turn_id: self.turn.turn_id,
                request_shape_digest: observation.cohort.request_shape_digest.clone(),
                kernel_identity,
                envelope_identity,
                committed_plan,
                protocol_frontier_count: self.protocol_frames.len(),
                protocol_prefix_digest: protocol_prefix_digest(&self.protocol_frames),
                observation,
            },
            transition,
        })
    }

    fn commit_active_epoch(&mut self, preview: ActiveEpochPreview) {
        self.active_epoch = Some(preview.epoch);
    }

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
            #[cfg(test)]
            automatic_checkpoint_policy: automatic_checkpoint::AutoCheckpointPolicy::from_config(
                LogicalCheckpointConfig::default(),
            ),
            retry_config: RetryConfig::default(),
            tool_timeout_secs: Some(60),
            turn: TurnRuntimeState::default(),
            next_turn_id: 0,
            max_iterations,
            max_tool_calls,
            context_scope_state: Arc::new(std::sync::Mutex::new(ContextScopeState::default())),
            runtime_snapshot_provider: None,
            logical_checkpoint_candidate_provider: None,
            context_experiment_restore_point: None,
            logical_checkpoint_control: LogicalCheckpointControl::disabled(),
            logical_request_observations: LogicalRequestObservationTracker::default(),
            active_epoch: None,
            pressure_compaction_suppressed: false,
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
            None,
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
        self.clear_active_epoch();
    }

    #[allow(dead_code)]
    pub fn restore_evidence(&mut self, evidence: Vec<EvidenceRecord>) -> Result<()> {
        Self::validate_evidence_ids(&evidence)?;
        self.runtime_snapshot.set_evidence(evidence);
        self.clear_active_epoch();
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
        self.turn = TurnRuntimeState::default();
        self.clear_active_epoch();
        Ok(())
    }

    /// Discard all state that belongs to the current session before creating a
    /// new one. Unlike compatibility rebuilds used by restore and checkout,
    /// this deliberately does not preserve runtime snapshot metadata.
    pub fn reset_for_new_session(&mut self) {
        self.protocol_frames.clear();
        self.history.clear();
        self.runtime_snapshot = Self::fresh_runtime_snapshot(&self.model);
        self.turn = TurnRuntimeState::default();
        self.next_turn_id = 0;
        self.clear_active_epoch();
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
        self.clear_active_epoch();
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

    /// Unregister a dynamically owned tool, such as an MCP tool for a server
    /// that has just been disabled. Default tools are otherwise unchanged.
    pub fn unregister_tool(&mut self, name: &str) -> bool {
        self.tools.remove(name)
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
        self.clear_active_epoch();
        Ok(())
    }

    pub fn clear_context_experiment_restore_point(&mut self) {
        // Retained as a compatibility entry point for legacy-session adopters.
        // Live agents no longer adopt experiment restore points.
    }

    pub(super) fn history_items(&self) -> Vec<HistoryItem> {
        crate::protocol_frames::history_items_from_frames(&self.protocol_frames)
    }

    pub(super) fn append_history_item(&mut self, item: HistoryItem) -> Result<()> {
        // Live protocol history is compaction authority. Attach transcript
        // provenance here so rebuild never has to invent Derived frames without
        // source spans under request-pressure selection.
        let mut frame = crate::protocol_frames::ProtocolFrame::derived(
            protocol_frame_item_from_history_item(&item),
        );
        frame.source_provenance = Some(protocol_item_default_provenance(
            &frame.item,
            next_protocol_source_sequence(self),
        ));
        self.append_protocol_frame(frame)
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
        sync_protocol_frame_provenance_from_snapshot(
            &mut self.protocol_frames,
            &self.runtime_snapshot,
        );
        self.clear_active_epoch();
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
        self.clear_active_epoch();
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
            .chain(selection.co_retired_frame_ids.iter().copied())
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
        // Selection already chose retired frames. Shared source spans between
        // retired prefixes and retained tails are normal and must not fail-fast.

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
        current_turn_start_index: Option<usize>,
    ) {
        self.runtime_snapshot = snapshot;
        self.protocol_frames = protocol_frames;
        self.history = history;
        self.turn.current_turn_start_index = current_turn_start_index;
        self.clear_active_epoch();
    }

    fn rebased_current_turn_start_index_after_compaction(
        &self,
        selection: &compaction::CompactionSelection,
        snapshot: &mut RuntimeSnapshot,
    ) -> Result<Option<usize>> {
        let active_before = self.runtime_snapshot.active_protocol_frames();
        let start = self
            .turn
            .current_turn_start_index
            .unwrap_or(active_before.len())
            .min(active_before.len());
        let retired = selection
            .retired_frame_ids
            .iter()
            .copied()
            .collect::<HashSet<_>>();
        let retained_active_turn_ids = active_before[start..]
            .iter()
            .filter_map(|frame| frame.runtime_frame_id)
            .filter(|id| !retired.contains(id))
            .collect::<HashSet<_>>();
        let protocol_frames = snapshot.active_protocol_frames();
        let rebased = protocol_frames
            .iter()
            .position(|frame| {
                frame
                    .runtime_frame_id
                    .is_some_and(|id| retained_active_turn_ids.contains(&id))
            })
            .unwrap_or(protocol_frames.len());
        let history = crate::protocol_frames::history_items_from_frames(&protocol_frames);
        crate::protocol_frames::analyze_history_items(&history, Some(rebased))?;
        // The summary has no turn identity. Rebuild turn protection solely from
        // retained active-turn identities, rather than protocol-group status.
        let turn_protected = protocol_frames[rebased..]
            .iter()
            .filter_map(|frame| frame.runtime_frame_id)
            .collect();
        snapshot.set_turn_protected_frame_ids(turn_protected);
        snapshot.validate_references()?;
        Ok(self.turn.current_turn_start_index.map(|_| rebased))
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
        // Keep the protocol cache and runtime authority on the same provenance
        // coordinates after every rebuild/heal.
        sync_protocol_frame_provenance_from_snapshot(
            &mut self.protocol_frames,
            &self.runtime_snapshot,
        );
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
                    // Prefer the producer-attached span when a cached frame was
                    // created before provenance was required.
                    if existing.provenance.source_span.is_none() {
                        if let Some(provenance) = frame.source_provenance.clone() {
                            existing.provenance = provenance;
                        }
                    }
                    existing
                })
                .unwrap_or_else(|| runtime_frame_from_protocol_frame(frame, ordinal as u32));
            snapshot.push_frame(runtime_frame);
        }
        snapshot.frames.extend(preserved_frames);
        ensure_active_protocol_source_spans(&mut snapshot);
        crate::protocol_frames::analyze_history_items(
            &crate::protocol_frames::history_items_from_frames(protocol_frames),
            self.turn.current_turn_start_index,
        )?;
        // Normal request construction protects every active-turn frame. The
        // compaction selector has its own narrower retirement blocker view.
        let start = self
            .turn
            .current_turn_start_index
            .unwrap_or(protocol_frames.len())
            .min(protocol_frames.len());
        let active_protocol_frames = snapshot.active_protocol_frames();
        let mut turn_protected_frame_ids = active_protocol_frames[start..]
            .iter()
            .filter_map(|frame| frame.runtime_frame_id)
            .collect::<Vec<_>>();
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
        _scope: ActiveContextExperiment,
        _protocol_frames: Vec<crate::protocol_frames::ProtocolFrame>,
        _runtime_snapshot: RuntimeSnapshot,
    ) {
        // Retained as a compatibility entry point for legacy-session adopters.
        // Live agents no longer adopt experiment restore points.
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
            #[cfg(test)]
            automatic_checkpoint_policy: automatic_checkpoint::AutoCheckpointPolicy::from_config(
                LogicalCheckpointConfig::default(),
            ),
            retry_config: self.retry_config.clone(),
            tool_timeout_secs: self.tool_timeout_secs,
            turn: TurnRuntimeState::default(),
            next_turn_id: 0,
            max_iterations: Some(1),
            max_tool_calls: Some(0),
            context_scope_state: Arc::new(std::sync::Mutex::new(ContextScopeState::default())),
            runtime_snapshot_provider: None,
            logical_checkpoint_candidate_provider: None,
            context_experiment_restore_point: None,
            logical_checkpoint_control: LogicalCheckpointControl::disabled(),
            logical_request_observations: LogicalRequestObservationTracker::default(),
            active_epoch: None,
            pressure_compaction_suppressed: false,
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
        E: FnMut(AgentEvent) -> Efut + Send,
        A: FnMut(PermissionRequest) -> Afut,
        Dfut: Future<Output = Result<()>>,
        Efut: Future<Output = Result<()>> + Send,
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
        E: FnMut(AgentEvent) -> Efut + Send,
        A: FnMut(PermissionRequest) -> Afut,
        Dfut: Future<Output = Result<()>>,
        Efut: Future<Output = Result<()>> + Send,
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
        E: FnMut(AgentEvent) -> Efut + Send,
        A: FnMut(PermissionRequest) -> Afut,
        Q: FnMut(QuestionRequest) -> Qfut + Send + 'static,
        Dfut: Future<Output = Result<()>>,
        Efut: Future<Output = Result<()>> + Send,
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

        if record.output.ok
            && is_context_tool_name(&call.name)
            && record
                .output
                .data
                .as_ref()
                .and_then(|data| data.get("pending_recording"))
                .and_then(Value::as_bool)
                == Some(true)
        {
            self.refresh_runtime_snapshot_from_provider()?;
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
        E: FnMut(AgentEvent) -> Result<()> + Send,
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
        E: FnMut(AgentEvent) -> Result<()> + Send,
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
        E: FnMut(AgentEvent) -> Efut + Send,
        Efut: Future<Output = Result<()>> + Send,
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
        E: FnMut(AgentEvent) -> Efut + Send,
        Efut: Future<Output = Result<()>> + Send,
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
        E: FnMut(AgentEvent) -> Efut + Send,
        A: FnMut(PermissionRequest) -> Afut,
        Dfut: Future<Output = Result<()>>,
        Efut: Future<Output = Result<()>> + Send,
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

    fn prune_old_tool_outputs(&mut self, preserve_recent_budget: u64) -> Result<()> {
        compaction::prune_old_tool_outputs(self, preserve_recent_budget)?;
        self.sync_protocol_caches_from_runtime_snapshot()?;
        self.clear_active_epoch();
        Ok(())
    }

    fn prepare_turn_prelude(&mut self, user_input: &str) -> Vec<PromptMessage> {
        self.try_prepare_turn_prelude(user_input)
            .expect("test/internal turn prelude should resolve selected skills")
    }

    fn try_prepare_turn_prelude(&mut self, user_input: &str) -> Result<Vec<PromptMessage>> {
        let manual_skill_material = self.manual_skill_material_messages(user_input)?;
        self.clear_active_epoch();
        let turn = WorkflowTurnState::from_user_input(user_input);
        self.next_turn_id = self.next_turn_id.saturating_add(1);
        self.turn = TurnRuntimeState::new(self.next_turn_id, turn.clone());
        if self.pressure_compaction_suppressed {
            self.turn.pressure_compaction.suppress();
        }
        self.runtime_snapshot.current_turn_id = Some(self.next_turn_id);

        let mut turn_prelude = self.prelude.clone();
        turn_prelude.push(runtime_context_message());
        if let Some(message) = self.skill_prelude_message() {
            turn_prelude.push(message);
        }
        turn_prelude.extend(manual_skill_material);
        if let Some(message) = turn.developer_context_message() {
            turn_prelude.push(message);
        }
        if let Some(message) = self.unreconciled_subagent_context_message() {
            turn_prelude.push(message);
        }
        Ok(turn_prelude)
    }

    fn manual_skill_material_messages(&self, user_input: &str) -> Result<Vec<PromptMessage>> {
        let names = parse_manual_skill_markers(user_input)?;
        if names.is_empty() {
            return Ok(Vec::new());
        }
        let registry = self
            .skill_registry
            .as_ref()
            .ok_or_else(|| anyhow!("unknown selected skill: {}", names[0]))?;
        registry
            .selected_entries(&names)?
            .into_iter()
            .map(|entry| {
                Ok(PromptMessage::developer_with_origin(
                    entry.content.clone(),
                    PromptMessageOrigin::SkillMaterial,
                ))
            })
            .collect()
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
        self.turn.pressure_compaction.reset_for_turn_end();
        self.turn.current_turn_start_index = None;
        self.runtime_snapshot.current_turn_id = None;
        self.runtime_snapshot = self.rebuilt_runtime_snapshot_from_protocol_frames(
            &self.protocol_frames,
            self.protocol_frames.len(),
            &self.history,
        )?;
        self.runtime_snapshot.current_turn_id = None;
        self.clear_active_epoch();
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
                "\n- {} [{}] {} — {} (child_session_id={}; child transcript navigation only, not a context node_id)",
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
                    "workflow__todos" | "workflow__auto_continue" | "agent__reconcile" => {
                        ToolEffectKind::WorkflowControl
                    }
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
    // Prefer producer provenance. Compatibility frames without provenance get a
    // synthetic transcript span so compaction can still retire them safely.
    // ContextSummary anchors remain spannable-optional (they are not retired).
    let provenance = frame
        .source_provenance
        .clone()
        .unwrap_or_else(|| protocol_item_default_provenance(&frame.item, ordinal as u64 + 1));
    let source = provenance.source;
    let source_span = provenance.source_span;
    let mut runtime_frame = RuntimeFrame::new(
        kind,
        FrameVisibility::Active,
        provenance,
        RuntimeFrameIdSeed {
            frame_kind: kind,
            source,
            ordinal,
            stable_key: &stable_key,
            source_span,
        },
    );
    runtime_frame.summary = summary;
    runtime_frame.protocol = Some(frame.item.clone());
    runtime_frame
}

fn protocol_item_default_provenance(
    item: &crate::protocol_frames::ProtocolFrameItem,
    sequence: u64,
) -> RuntimeFrameProvenance {
    match item {
        crate::protocol_frames::ProtocolFrameItem::ContextSummary { .. } => {
            RuntimeFrameProvenance::new(RuntimeSource::SummaryArtifact)
        }
        _ => RuntimeFrameProvenance::new(RuntimeSource::Transcript).with_span(
            SourceSpan::new(sequence.max(1), sequence.max(1)).expect("singleton source span"),
        ),
    }
}

fn next_protocol_source_sequence<C: Config>(agent: &Agent<C>) -> u64 {
    let from_frames = agent
        .runtime_snapshot
        .frames
        .iter()
        .filter_map(|frame| frame.provenance.source_span.map(|span| span.end_sequence))
        .chain(
            agent
                .protocol_frames
                .iter()
                .filter_map(|frame| {
                    frame
                        .source_provenance
                        .as_ref()
                        .and_then(|provenance| provenance.source_span)
                        .map(|span| span.end_sequence)
                }),
        )
        .max()
        .unwrap_or(0);
    let from_leaf = agent.runtime_snapshot.leaf_sequence.unwrap_or(0);
    from_frames.max(from_leaf).saturating_add(1).max(1)
}

/// Close the compaction invariant for live snapshots: every active protocol
/// frame that can participate in retirement must have a source span.
///
/// Synthetic spans are only assigned when producers left `None`. They never
/// overwrite an existing transcript coordinate. ContextSummary anchors may remain
/// spannable-optional because they are selection base markers, not retired bodies.
pub(super) fn sync_protocol_frame_provenance_from_snapshot(
    protocol_frames: &mut [crate::protocol_frames::ProtocolFrame],
    snapshot: &RuntimeSnapshot,
) {
    for (frame, projected) in protocol_frames
        .iter_mut()
        .zip(snapshot.active_protocol_frames())
    {
        if frame.runtime_frame_id.is_none() {
            frame.runtime_frame_id = projected.runtime_frame_id;
        }
        if frame.source_provenance.is_none() {
            frame.source_provenance = projected.source_provenance.clone();
            continue;
        }
        if let (Some(cached), Some(healed)) = (
            frame.source_provenance.as_mut(),
            projected.source_provenance.as_ref(),
        ) {
            if cached.source_span.is_none() {
                cached.source_span = healed.source_span;
                if cached.source == RuntimeSource::Derived {
                    cached.source = healed.source;
                }
            }
        }
    }
}

pub(super) fn ensure_active_protocol_source_spans(snapshot: &mut RuntimeSnapshot) {
    let mut high = snapshot
        .frames
        .iter()
        .filter_map(|frame| frame.provenance.source_span.map(|span| span.end_sequence))
        .chain(snapshot.leaf_sequence)
        .max()
        .unwrap_or(0);
    for frame in &mut snapshot.frames {
        if frame.visibility != FrameVisibility::Active || frame.protocol.is_none() {
            continue;
        }
        if matches!(
            frame.protocol,
            Some(crate::protocol_frames::ProtocolFrameItem::ContextSummary { .. })
        ) {
            continue;
        }
        if frame.provenance.source_span.is_some() {
            continue;
        }
        high = high.saturating_add(1);
        let span = SourceSpan::new(high, high).expect("singleton source span");
        if frame.provenance.source == RuntimeSource::Derived {
            frame.provenance.source = RuntimeSource::Transcript;
        }
        frame.provenance.source_span = Some(span);
    }
    if snapshot.leaf_sequence.map(|leaf| leaf < high).unwrap_or(true) {
        snapshot.leaf_sequence = Some(high);
    }
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
    pressure_compaction: PressureCompactionState,
    #[cfg(test)]
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
            pressure_compaction: PressureCompactionState::default(),
            #[cfg(test)]
            automatic_checkpoint: AutomaticCheckpointSchedulerState::default(),
        }
    }
}

/// Ephemeral pressure-compaction state. It is absent from transcript and
/// snapshot projections, so a restored turn never inherits an attempted prompt.
#[derive(Debug, Clone, PartialEq, Eq)]
struct PressureCompactionState {
    last_attempted_frontier: Option<PressureCompactionFrontier>,
    suppressed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PressureCompactionFrontier {
    frame_count: usize,
    protocol_prefix_digest: String,
}

impl Default for PressureCompactionState {
    fn default() -> Self {
        Self {
            last_attempted_frontier: None,
            suppressed: false,
        }
    }
}

impl PressureCompactionState {
    fn mark_attempted(&mut self, frontier: PressureCompactionFrontier) -> Result<()> {
        ensure!(!self.suppressed, "pressure compaction is suppressed");
        ensure!(
            self.last_attempted_frontier.as_ref() != Some(&frontier),
            "pressure compaction already attempted for this protocol frontier"
        );
        self.last_attempted_frontier = Some(frontier);
        Ok(())
    }

    fn suppress(&mut self) {
        self.suppressed = true;
    }

    fn reset_for_turn_end(&mut self) {
        *self = Self::default();
    }
}

// Compatibility-only scheduler fixture retained for legacy unit tests. Normal
// production request preparation exclusively uses `PressureCompactionState`.
#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
impl AutomaticCheckpointSchedulerState {
    fn begin_complete_boundary(&mut self) -> u64 {
        self.next_boundary_generation += 1;
        self.current_boundary_generation = Some(self.next_boundary_generation);
        self.next_boundary_generation
    }
    fn consume_complete_boundary(&mut self) -> Option<u64> {
        let boundary = self.current_boundary_generation.take();
        self.last_consumed_boundary = boundary;
        boundary
    }
    fn mark_attempted(&mut self, boundary: u64) {
        self.last_attempted_boundary = Some(boundary);
        self.armed = false;
    }
    fn mark_committed(&mut self, owner: LogicalCheckpointRequestOwner) {
        if matches!(owner, LogicalCheckpointRequestOwner::Automatic { .. }) {
            self.commits += 1;
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
    fn view(&self) -> automatic_checkpoint::AutoCheckpointSchedulerView {
        automatic_checkpoint::AutoCheckpointSchedulerView {
            armed: self.armed,
            automatic_commits: self.commits,
            boundary_available: self.current_boundary_generation.is_some(),
            boundary_consumed: self.current_boundary_generation == self.last_consumed_boundary,
            boundary_attempted: self.current_boundary_generation == self.last_attempted_boundary,
            suppressed: self.suppressed,
        }
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
        "workflow__todos" | "workflow__auto_continue" | "agent__reconcile"
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
mod tests;

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
