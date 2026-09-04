use anyhow::{Context, Result, anyhow, bail, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet, VecDeque};
use std::future::Future;
use std::path::Path;
use std::pin::Pin;
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::{Instrument, debug, warn};

use crate::config::{ApiProtocol, CompactionConfig, ModelRoute, ProviderConfig, RetryConfig};
use crate::evidence::{EvidenceDraft, EvidenceRecord, EvidenceSource, require_unique_evidence_id};
use crate::model_runtime::{ResolvedModelRoute, ResolvedRuntimeCatalog};
#[cfg(test)]
use crate::permission::ToolScope;
use crate::permission::{
    ExecutionDirective, PermissionApproval, PermissionDecision, PermissionMode, PermissionRequest,
    PermissionSessionState, restricted_by_directive_with_class,
};
use crate::request_builder::{
    HistoryItem, HistoryToolCall, ModelReasoningEffort, ModelRequestMetadata, PromptMessage,
    PromptMessageOrigin, ProtectedContextPolicy, RequestBuilderInput, SelectedPromptRequestInput,
    build_request_from_selected_prompt, build_request_with_policy, effective_input_budget_tokens,
    observe_logical_request,
};
use crate::runtime_context::{
    FrameVisibility, RuntimeFrame, RuntimeFrameIdSeed, RuntimeFrameKind, RuntimeFrameProvenance,
    RuntimeSnapshot, RuntimeSource, SourceSpan,
};
use crate::skills::{
    SkillCard, SkillRegistry, SkillResourceListTool, SkillResourceReadTool, SkillTool,
    reconcile_loaded_skill_material as reconcile_snapshot_skill_material,
};
use crate::tool::{
    NormalizedSubagentInput, QuestionCallback, QuestionRequest, QuestionResponse,
    SubagentPathScope, ToolExecutionContext, ToolHandler, ToolParallelism, ToolRegistry,
    ToolResult, external_workspace_access_for_tool, is_delegation_path_scoped_tool,
    normalize_subagent_input, subagent_parameters_schema,
};
use crate::tool_format::format_tool_call;
use crate::tool_names;
use crate::transcript::{ContextScopeState, ROOT_CONTEXT_BRANCH_ID};
use crate::user_content::UserMessageContent;
use indexmap::IndexMap;

#[path = "agent/auto_review.rs"]
mod auto_review;
#[path = "agent/catalog.rs"]
mod catalog;
#[path = "agent/compaction.rs"]
pub(crate) mod compaction;
#[path = "agent/events.rs"]
mod events;
#[path = "agent/evidence_memory.rs"]
mod evidence_memory;
#[path = "agent/history_compact.rs"]
mod history_compact;
#[path = "agent/protocol_stream.rs"]
mod protocol_stream;
#[path = "agent/tool_execution.rs"]
mod tool_execution;
pub(crate) use auto_review::{AutoReviewResolution, AutoReviewService};

use crate::anchored_bootstrap::{AnchoredBootstrap, AnchoredPhase};
pub use crate::workflow_state::{AutoContinueState, TodoItem, TodoStatus};
pub use catalog::{AgentFactory, AgentTemplate, SubagentCapabilityContract};
pub(crate) use catalog::{
    SUBAGENT_CATALOG, agent_name_for_subagent_tool, is_subagent_tool_name,
    subagent_catalog_entry_by_tool_name, subagent_evidence_parent_tool,
    subagent_tool_name_for_agent_name,
};
#[allow(unused_imports)]
pub(crate) use catalog::{SubagentCatalogEntry, subagent_catalog_entry_by_agent_name};
pub use events::{
    AgentEvent, CacheUsageReport, CompactionAttemptOutcome, CompactionBlocker,
    CompactionNoProgress, CompactionTrigger, ContextCompactionEvent, LlmRequestErrorClass,
    LlmRequestTelemetry, LlmRequestTelemetryPhase, LlmRetryLifecycle, ManualCompactionOutcome,
    PromptCompositionEntry, ProviderUsageCompleteness, TokenUsageEstimate,
    ToolExecutionSummaryEvent, TurnFinalizedEvent, TurnStartedEvent, ValidationAdvisory,
};
#[cfg(test)]
pub(crate) use events::{CompactionCheckpoint, CompactionFileOperations};

#[cfg(test)]
use compaction::default_preserve_recent_budget;

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
    DelegationScopeDenied,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ToolCallRecordOutcome {
    Completed,
    Cancelled,
}

#[derive(Debug, Clone, Default)]
struct LogicalRequestObservationTracker {
    previous: Option<crate::request_builder::LogicalRequestObservation>,
}

/// Ephemeral authority for the active cache prefix. It deliberately stores
/// only process-local identities, never prompt bytes or transcript data.
#[derive(Debug, Clone, PartialEq)]
struct ActiveEpoch {
    turn_id: u64,
    request_shape_digest: String,
    observation: crate::request_builder::LogicalRequestObservation,
    committed_plan: crate::request_builder::prompt_plan::PromptPlan,
    protocol_frontier_count: usize,
    protocol_frontier_token: u64,
    protocol_append_generation: u64,
    projection_generation: u64,
    budget: crate::request_builder::BudgetReport,
    selected_evidence_ids: Vec<String>,
    selected_evidence_message: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub(super) enum ColdRequiredReason {
    NoActiveEpoch,
    TurnChanged,
    RequestProjectionChanged,
    ProtocolFrontierChanged,
    TruncatedOrReselected,
    EvidenceChanged,
    UnsupportedAppendShape,
    SuffixValidationFailed,
    BudgetRequiresColdPlan,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub(super) enum ActiveEpochPreparation {
    Warm(ActiveEpochPreview),
    ColdRequired(ColdRequiredReason),
}

#[derive(Debug, Clone)]
pub(super) struct ActiveEpochPreview {
    epoch: ActiveEpoch,
    build: crate::request_builder::BuildResult,
    transition: ActiveEpochTransition,
}

/// Process-local provider-usage baseline for projecting context growth between
/// provider responses. This deliberately remains outside durable projections.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ProviderUsageAnchor {
    usage: TokenUsageEstimate,
    protocol_frontier_count: usize,
    protocol_prefix_digest: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ActiveEpochTransition {
    Cold,
    Append { added: usize },
}

fn protocol_prefix_digest(frames: &[crate::protocol_frames::ProtocolFrame]) -> String {
    // 逐帧折叠成有界指纹串再统一哈希：不再对每帧 item 做大 JSON 全量序列化，也不把
    // 所有帧的序列化累积成一个大 Vec（1M 上下文下这是同步内存/CPU 洪水）。digest 仅供
    // 进程内前后一致性比较（压力 frontier / usage anchor），不落盘；有界指纹保留
    // 字段集合与有序边界，压缩后必变、未变 reload 必稳定。
    let mut fingerprints = Vec::with_capacity(frames.len().saturating_mul(64));
    for frame in frames {
        let identity = frame.bounded_identity_bytes();
        fingerprints.extend_from_slice(&(identity.len() as u64).to_le_bytes());
        fingerprints.extend_from_slice(crate::request_builder::sha256_hex(&identity).as_bytes());
    }
    crate::request_builder::sha256_hex(&fingerprints)
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

pub trait SubagentChildFactory: Send + Sync {
    fn resolve_route(
        &self,
        parent: &Agent,
        template: &AgentTemplate,
        requested_route: Option<&ModelRoute>,
        takeover: bool,
    ) -> Result<ModelRoute>;

    fn create_child(
        &self,
        parent: &Agent,
        template: &AgentTemplate,
        route: &ModelRoute,
        max_tool_calls_override: Option<usize>,
    ) -> Result<Agent>;
}

pub trait PrimaryRouteFactory: Send + Sync {
    fn prepare_route(&self, route: ModelRoute) -> Result<PreparedPrimaryRoute>;
}

/// Provider-catalog factory used at startup and after configuration reload.
///
/// Prepared routes expose the full provider model catalog so subsequent
/// in-session switches do not need a reload to learn sibling model metadata.
pub struct ConfiguredPrimaryRouteFactory {
    providers: IndexMap<String, ProviderConfig>,
    global_retry: RetryConfig,
    runtime_catalog: Option<ResolvedRuntimeCatalog>,
}

impl ConfiguredPrimaryRouteFactory {
    #[allow(dead_code)]
    pub fn new(providers: IndexMap<String, ProviderConfig>, global_retry: RetryConfig) -> Self {
        Self {
            providers,
            global_retry,
            runtime_catalog: None,
        }
    }

    pub fn new_with_runtime_catalog(
        providers: IndexMap<String, ProviderConfig>,
        global_retry: RetryConfig,
        runtime_catalog: ResolvedRuntimeCatalog,
    ) -> Self {
        Self {
            providers,
            global_retry,
            runtime_catalog: Some(runtime_catalog),
        }
    }
}

impl PrimaryRouteFactory for ConfiguredPrimaryRouteFactory {
    fn prepare_route(&self, route: ModelRoute) -> Result<PreparedPrimaryRoute> {
        let provider = self.providers.get(&route.provider).ok_or_else(|| {
            anyhow!(
                "provider '{}' is not defined under [providers]",
                route.provider
            )
        })?;
        if !provider.has_model(&route.model) {
            bail!(
                "model '{}' is not defined under [providers.{}.models]",
                route.model,
                route.provider
            );
        }
        let runtime_route = self
            .runtime_catalog
            .as_ref()
            .and_then(|catalog| catalog.route(&route.provider, &route.model))
            .cloned()
            .map(Arc::new);
        let prepared = PreparedPrimaryRoute::new_with_runtime_route(
            route.clone(),
            provider.protocol,
            provider
                .models
                .iter()
                .map(|(id, model)| (id.clone(), model.protocol))
                .collect(),
            provider
                .models
                .iter()
                .map(|(id, model)| (id.clone(), model.request_metadata()))
                .collect(),
            provider
                .retry
                .clone()
                .unwrap_or_else(|| self.global_retry.clone()),
            runtime_route,
        );
        Ok(match self.runtime_catalog.clone() {
            Some(catalog) => prepared.with_runtime_catalog(catalog),
            None => prepared,
        })
    }
}

pub struct PreparedPrimaryRoute {
    route: ModelRoute,
    default_protocol: ApiProtocol,
    model_protocols: HashMap<String, ApiProtocol>,
    model_catalog: HashMap<String, ModelRequestMetadata>,
    retry_config: RetryConfig,
    runtime_route: Option<Arc<ResolvedModelRoute>>,
    runtime_catalog: Option<ResolvedRuntimeCatalog>,
}

pub(crate) struct PreparedPrimaryRouteInstall {
    route: PreparedPrimaryRoute,
}

#[derive(Clone)]
struct RetainedRoutePreparation {
    runtime_route: Arc<ResolvedModelRoute>,
    route_factory: Arc<dyn PrimaryRouteFactory>,
}

impl PreparedPrimaryRoute {
    #[allow(dead_code)]
    pub fn new(
        route: ModelRoute,
        default_protocol: ApiProtocol,
        model_protocols: HashMap<String, ApiProtocol>,
        model_catalog: HashMap<String, ModelRequestMetadata>,
        retry_config: RetryConfig,
    ) -> Self {
        Self {
            route,
            default_protocol,
            model_protocols,
            model_catalog,
            retry_config,
            runtime_route: None,
            runtime_catalog: None,
        }
    }

    pub fn new_with_runtime_route(
        route: ModelRoute,
        default_protocol: ApiProtocol,
        model_protocols: HashMap<String, ApiProtocol>,
        model_catalog: HashMap<String, ModelRequestMetadata>,
        retry_config: RetryConfig,
        runtime_route: Option<Arc<ResolvedModelRoute>>,
    ) -> Self {
        Self {
            route,
            default_protocol,
            model_protocols,
            model_catalog,
            retry_config,
            runtime_route,
            runtime_catalog: None,
        }
    }

    pub fn with_runtime_catalog(mut self, runtime_catalog: ResolvedRuntimeCatalog) -> Self {
        self.runtime_catalog = Some(runtime_catalog);
        self
    }

    pub(crate) fn with_resolved_authority_from(
        mut self,
        agent: &Agent,
        route: &ModelRoute,
    ) -> Result<Self> {
        let runtime_route = agent.resolved_model_route_for(route).ok_or_else(|| {
            anyhow!(
                "resolved runtime route is unavailable: {}",
                route.display_name()
            )
        })?;
        self.runtime_route = Some(runtime_route);
        self.runtime_catalog = agent.resolved_runtime_catalog.clone();
        Ok(self)
    }

    pub(crate) fn candidate_session_usage_with_composition(
        &self,
        agent: &Agent,
        runtime_snapshot: &RuntimeSnapshot,
    ) -> Result<(TokenUsageEstimate, Vec<PromptCompositionEntry>)> {
        agent.candidate_session_usage_with_route(
            &self.route,
            &self.model_catalog,
            self.runtime_route.as_deref(),
            runtime_snapshot,
        )
    }

    pub(crate) fn into_install(self) -> PreparedPrimaryRouteInstall {
        PreparedPrimaryRouteInstall { route: self }
    }
}

impl PreparedPrimaryRouteInstall {
    pub(crate) fn apply(self, agent: &mut Agent) {
        let PreparedPrimaryRoute {
            route,
            default_protocol,
            model_protocols,
            model_catalog,
            retry_config,
            runtime_route,
            runtime_catalog,
        } = self.route;
        agent.apply_route_with_runtime_route(
            route,
            default_protocol,
            model_protocols,
            model_catalog,
            retry_config,
            runtime_route,
            runtime_catalog,
        );
    }
}

pub trait SubagentDelegate: Send + Sync {
    fn run_named<'a>(
        &'a self,
        parent: &'a Agent,
        agent_name: &'a str,
        invocation: SubagentInvocation,
    ) -> Pin<Box<dyn Future<Output = Result<ToolResult>> + Send + 'a>>;

    fn control<'a>(
        &'a self,
        tool_name: &'a str,
        _args: &'a Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolResult>> + Send + 'a>> {
        Box::pin(async move {
            Ok(ToolResult::err(
                tool_name,
                format!("{tool_name} is unavailable in this subagent runtime"),
            ))
        })
    }

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
    pub model: Option<ModelRoute>,
    pub prompt: String,
    /// Stable identity of the parent subagent tool call, when launched by a tool.
    /// Direct slash delegation intentionally has no parent call.
    pub parent_tool_call_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConversationRole {
    User,
    Assistant,
    Summary,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConversationMessage {
    pub role: ConversationRole,
    pub content: String,
}

const DEFAULT_AGENT_PRELUDE: &str = r#"你是运行在本地仓库中的编程代理。
请基于项目真实状态工作。改代码前先检查相关文件。优先做符合现有模式、且最小正确的改动。
有意识地使用工具：先读/搜再改，只编辑目标文件；改完后在相关时运行与任务匹配的验证。
在适用时优先使用专用工具而非 `shell__exec`：工作区文件读写/列表用 `fs__*`，文本搜索用 `search__rg`（或其他搜索工具），git 状态/diff/log 用 `git__*`，补丁用编辑工具。将 `shell__exec` 留给构建、测试、进程控制，以及确实需要 shell 管道或专用工具无法表达的 OS 能力的命令。不要默认用 shell 里的 `cat`/`sed`/`rg`/`find` 替代上述工具。
运行 shell 命令时，对高影响或不可逆操作要谨慎（例如 `rm -rf`、强制覆盖、批量删除）：在目标、范围和影响明确之前不要执行。对由 `&&`、`||`、`;` 或管道连接的复合命令，保持顺序、失败行为与副作用范围与意图一致——不要静默串联本应分开由人决策的步骤，也不要让组合命令过早产生副作用、扩大范围或改变含义（例如，不要把「先查看历史，再决定是否 push」变成 `git log --oneline -n 10 && git push`）。带循环、长驻监听或重试逻辑的命令必须有明确退出条件、超时或资源限制，以免无限挂起。
当当前任务可能与先前调查、失败路径、回传上下文实验，或有重要历史的文件重叠时，有选择地使用 `memory__recall`。调试时优先按相关文件路径以及失败/受阻结果过滤；不要把 recall 当成每个任务的必做第一步。
工程任务工作流：
若本回合涉及代码修复、重构、调试、多文件改动、需要验证或跨模块的工程任务，先像工作流管理者一样行动：先判断是否需要专家通道，再直接动手。
仅在琐碎、单文件、边界清晰的工作，或委派开销明显超过收益时直接执行；否则保持简短计划，有意识地选择合适专家或直接路径，调和委派结果，并以最清晰且经过验证的结果收尾。
有意识地选择专家：explorer 用于广泛或未知代码搜索；fixer 用于有界实现与多文件机械修改；oracle 用于根因分析、风险审查或关键评估；designer 用于 UI/UX 决策；librarian 用于外部文档或库/框架行为；general 用于有界只读辅助工作。
优先复用先前专家成果：使用会话历史或任务板中已完成或已调和的会话，再启动重叠工作；绝不要把已取消或出错的会话当作权威结果复用。
当委派能提升质量、速度或上下文卫生时，委派有界工作，尤其是会污染主代理上下文的底层或读密集任务。
SubagentPool 支持同角色并发；只读任务共享读锁，fixer 必须声明非空 owned_paths 并取得对应文件或目录子树的写锁。路径重叠的读写或写写任务不会排队，而是明确拒绝，待冲突任务完成或取消后再启动。使用 agent__jobs/status/wait/cancel 管理后台任务；查询和等待不得隐式接管或重复执行。引用子代理时优先使用稳定的池序号（#N），而不是列表位置。两种启动模式：省略 target_child_session_id 以创建新子会话，或设置 target_child_session_id 以接管已结束的子会话并复用其上下文。历史子会话是池记录；只有显式接管才会继续既有会话。
保持委派受控：避免递归委派，避免不必要的多智能体编排，保持清晰的父代理叙事，调和子结果，并在停止前暴露剩余阻塞或有针对性的验证缺口。
保持在范围内。除非必要，不要重构、重排格式、重命名或修改无关代码；若需要更广改动，说明原因。
当工具、编辑或验证失败时，先检查错误再重试。不要用宽泛回退或跳过验证来掩盖失败；快速失败并解释可操作原因。
高效使用上下文：读大文件前先搜索，只读相关部分，避免倾倒长输出，长任务时总结状态。
需求模糊或有风险时，提出简洁澄清问题。
回复保持简洁。修改代码时总结变更文件与验证结果。

渲染输出：
- 当前前端已支持原生 LaTeX 数学公式与 Mermaid 图表渲染，优先使用真正的 Markdown 语法，不要用字符画或 ASCII 框线图模拟。
- 数学公式：行内公式用 `$...$`（或 `\(...\)`），独立公式用 `$$...$$`（或 `\[...\]`）。不要用 Unicode 近似或纯文本拼凑公式。
- 图表：需要使用流程图、时序图、状态图、类图、ER 图、甘特图、思维导图或时间线时，用 ```mermaid 代码块编写，而不是手绘 ASCII 图。
- Mermaid 节点或边标签中的数学公式用 `$$...$$`。"#;

const SESSION_TITLE_PRELUDE: &str = r#"为用户的第一条消息生成简洁的会话标题，准确概括其主题、意图或任务。
将该消息视为待命名的内容，不要把它当作需要回复的对话。
输出应为描述性标题，而不是对用户消息的直接回应。
只返回标题文本。
不要使用引号、项目符号、Markdown、前缀或解释。
保持具体，且不超过 80 个字符。"#;
const CONTEXT_COMPACTION_PRELUDE: &str = r#"你正在为同一会话生成结构化执行检查点，供后续模型从当前状态继续工作。
近期原始消息仍会保留在检查点之后。

输出要求：
- 只输出检查点正文（Markdown），不要加前言、后记、外层代码块或过程解释。
- 严格使用以下 Markdown section 标题与顺序：
  ## Progress
  ### Done
  ### In Progress
  ### Blocked
  ## Key Decisions
  ## Validation
  ## File Operations
  ### Read
  ### Modified
  ## Next Steps
  ## Critical Context
- Progress：明确已完成、正在执行、受阻的事项；In Progress 必须给出当前阶段和未完成工作。
- Key Decisions：记录已解决的用户选择，以及不可恢复或已拒绝、不得重试的方案。
- Validation：记录已运行、通过、失败或尚未运行的验证及其证据。
- File Operations：分别列出累计读过与修改过的文件。只列路径；无内容写「无」。
- Next Steps：首项必须是精确、可立即执行的下一步；不要重新规划已完成工作或重问已解决问题。
- Critical Context：保留精确路径、命令、错误、标识符、接口、配置、待办和专家结论。活动回合被部分折叠时，明确交接当前阶段、精确下一步、未解决工作与不得重复的决定。
- 保留并逐字引用重要的路径、命令、错误信息、标识符、接口名、配置键、测试名。
- 每个 section 均必须存在；无内容写「无」。
- 不得输出保留标记 `[retained-facts:v1]`。"#;
const COMPACTION_TOOL_OUTPUT_CHAR_CAP: usize = 2_000;
const COMPACTION_TOOL_OUTPUT_TRUNCATION_MARKER: &str = "… [工具输出已为压缩而截断]";
const MAX_SKILL_CARDS_IN_PRELUDE: usize = 64;

pub(crate) type RuntimeSnapshotProvider = Arc<dyn Fn() -> Result<RuntimeSnapshot> + Send + Sync>;

#[derive(Debug, Default)]
pub(crate) struct TurnContinuationQueue {
    pending: VecDeque<PendingTurnContinuation>,
    preempted_by_user_prompt: bool,
}

impl TurnContinuationQueue {
    pub(crate) fn push(&mut self, continuation: PendingTurnContinuation) {
        self.pending.push_back(continuation);
    }

    pub(crate) fn mark_user_prompt_queued(&mut self) {
        self.preempted_by_user_prompt = true;
    }

    pub(crate) fn preempted_by_user_prompt(&self) -> bool {
        self.preempted_by_user_prompt
    }

    pub(crate) fn drain_ready(&mut self) -> Vec<PendingTurnContinuation> {
        if self.preempted_by_user_prompt {
            return Vec::new();
        }
        self.pending.drain(..).collect()
    }
}

#[derive(Debug, Clone)]
pub(crate) struct PendingTurnContinuation {
    pub result: Option<crate::subagent::SubagentRunSummary>,
}

pub(crate) type TurnContinuationProvider =
    Arc<dyn Fn() -> Result<Vec<PendingTurnContinuation>> + Send + Sync>;

pub struct Agent {
    model: String,
    primary_route: Option<ModelRoute>,
    subagent_model_overrides: HashMap<String, String>,
    default_protocol: ApiProtocol,
    model_protocols: HashMap<String, ApiProtocol>,
    model_catalog: HashMap<String, ModelRequestMetadata>,
    session_reasoning_efforts: HashMap<String, ModelReasoningEffort>,
    prelude: Vec<PromptMessage>,
    runtime_snapshot: RuntimeSnapshot,
    protocol_append_state: crate::protocol_frames::ProtocolAppendState,
    tools: ToolRegistry,
    skill_registry: Option<Arc<SkillRegistry>>,
    skill_cards: Vec<SkillCard>,
    subagent_delegate: Option<Arc<dyn SubagentDelegate>>,
    subagent_child_factory: Option<Arc<dyn SubagentChildFactory>>,
    primary_route_factory: Option<Arc<dyn PrimaryRouteFactory>>,
    question_handler: Option<QuestionCallback>,
    auto_review_service: Option<Arc<dyn AutoReviewService>>,
    permission_session: Arc<Mutex<PermissionSessionState>>,
    /// Run-local directory authorization for delegated children. Never inherited.
    subagent_path_scope: Option<Arc<SubagentPathScope>>,
    compaction_config: CompactionConfig,
    retry_config: RetryConfig,
    tool_timeout_secs: Option<u64>,
    turn: TurnRuntimeState,
    next_turn_id: u64,
    max_iterations: Option<usize>,
    max_tool_calls: Option<usize>,
    context_scope_state: Arc<std::sync::Mutex<ContextScopeState>>,
    runtime_snapshot_provider: Option<RuntimeSnapshotProvider>,
    turn_continuation_provider: Option<TurnContinuationProvider>,
    logical_request_observations: LogicalRequestObservationTracker,
    active_epoch: Option<ActiveEpoch>,
    provider_usage_anchor: Option<ProviderUsageAnchor>,
    request_projection_generation: u64,
    // Summary agents must never recursively compact their own request. This
    // outlives their turn initialization, which replaces `TurnRuntimeState`.
    pressure_compaction_suppressed: bool,
    fast_mode: Option<Arc<crate::fast_mode::FastMode>>,
    /// Anchored bootstrap experiment state; None = experiment not enabled.
    anchored: Option<AnchoredBootstrap>,
    /// Session-level runtime switch (default on). The `/anchored` command
    /// flips it; the experiment only applies while it is on.
    anchored_override: bool,
    /// Phase bound once per turn by the prelude hook, so the tool catalog and
    /// alias resolution stay stable across iterations of one request.
    anchored_request_phase: Option<AnchoredPhase>,
    fake_client: Option<crate::fake::FakeClient>,
    fake_installation_id: String,
    fake_identity: Option<crate::fake::CodexIdentity>,
    resolved_model_route: Option<Arc<ResolvedModelRoute>>,
    resolved_runtime_catalog: Option<ResolvedRuntimeCatalog>,
    retained_route_preparations: HashMap<String, RetainedRoutePreparation>,
}

impl AgentFactory {
    #[cfg(test)]
    pub fn create_child(parent: &Agent, template: &AgentTemplate) -> Agent {
        Self::create_child_with_max_tool_calls(parent, template, None)
    }

    #[allow(dead_code)]
    pub fn create_child_with_max_tool_calls(
        parent: &Agent,
        template: &AgentTemplate,
        max_tool_calls_override: Option<usize>,
    ) -> Agent {
        Self::create_child_with_route_and_max_tool_calls(
            parent,
            template,
            None,
            false,
            max_tool_calls_override,
        )
        .expect("default child route should be constructible")
    }

    pub fn create_child_with_route_and_max_tool_calls(
        parent: &Agent,
        template: &AgentTemplate,
        requested_route: Option<ModelRoute>,
        takeover: bool,
        max_tool_calls_override: Option<usize>,
    ) -> Result<Agent> {
        if let Some(factory) = &parent.subagent_child_factory {
            let route =
                factory.resolve_route(parent, template, requested_route.as_ref(), takeover)?;
            return factory.create_child(parent, template, &route, max_tool_calls_override);
        }
        if requested_route.is_some() || takeover {
            bail!("subagent model route selection is not configured");
        }

        let model = parent
            .subagent_model_override(&template.name)
            .unwrap_or(parent.model())
            .to_string();
        let inherited_runtime_route = parent
            .subagent_model_override(&template.name)
            .is_none()
            .then(|| parent.resolved_model_route.clone())
            .flatten();
        let mut child = Self::create_child_with_parts(
            parent,
            template,
            model,
            parent.default_protocol,
            parent.model_protocols.clone(),
            parent.model_catalog.clone(),
            parent.retry_config.clone(),
            inherited_runtime_route,
            parent.resolved_runtime_catalog.clone(),
            max_tool_calls_override,
        );
        if parent.subagent_model_override(&template.name).is_none()
            && let Some(route) = parent.primary_route().cloned()
        {
            child.set_primary_route(route);
        }
        Ok(child)
    }

    pub fn resolve_subagent_route(
        parent: &Agent,
        template: &AgentTemplate,
        requested_route: Option<&ModelRoute>,
        takeover: bool,
    ) -> Result<ModelRoute> {
        if let Some(factory) = &parent.subagent_child_factory {
            return factory.resolve_route(parent, template, requested_route, takeover);
        }
        if requested_route.is_some() || takeover {
            bail!("subagent model route selection is not configured");
        }
        parent.primary_route().cloned().ok_or_else(|| {
            anyhow!(
                "parent model route is unavailable for expert '{}'",
                template.name
            )
        })
    }

    pub fn create_prepared_routed_child_with_max_tool_calls(
        parent: &Agent,
        template: &AgentTemplate,
        prepared: PreparedPrimaryRoute,
        max_tool_calls_override: Option<usize>,
    ) -> Agent {
        let PreparedPrimaryRoute {
            route,
            default_protocol,
            model_protocols,
            model_catalog,
            retry_config,
            runtime_route,
            runtime_catalog,
        } = prepared;
        Self::create_routed_child_with_runtime_route(
            parent,
            template,
            route,
            default_protocol,
            model_protocols,
            model_catalog,
            retry_config,
            runtime_route,
            runtime_catalog,
            max_tool_calls_override,
        )
    }

    #[allow(dead_code)]
    pub fn create_routed_child_with_max_tool_calls(
        parent: &Agent,
        template: &AgentTemplate,
        route: ModelRoute,
        default_protocol: ApiProtocol,
        model_protocols: HashMap<String, ApiProtocol>,
        model_catalog: HashMap<String, ModelRequestMetadata>,
        retry_config: RetryConfig,
        max_tool_calls_override: Option<usize>,
    ) -> Agent {
        Self::create_routed_child_with_runtime_route(
            parent,
            template,
            route,
            default_protocol,
            model_protocols,
            model_catalog,
            retry_config,
            None,
            None,
            max_tool_calls_override,
        )
    }

    fn create_routed_child_with_runtime_route(
        parent: &Agent,
        template: &AgentTemplate,
        route: ModelRoute,
        default_protocol: ApiProtocol,
        model_protocols: HashMap<String, ApiProtocol>,
        model_catalog: HashMap<String, ModelRequestMetadata>,
        retry_config: RetryConfig,
        runtime_route: Option<Arc<ResolvedModelRoute>>,
        runtime_catalog: Option<ResolvedRuntimeCatalog>,
        max_tool_calls_override: Option<usize>,
    ) -> Agent {
        let mut child = Self::create_child_with_parts(
            parent,
            template,
            route.model.clone(),
            default_protocol,
            model_protocols,
            model_catalog,
            retry_config,
            runtime_route,
            runtime_catalog,
            max_tool_calls_override,
        );
        child.set_primary_route(route);
        child
    }

    #[allow(clippy::too_many_arguments)]
    fn create_child_with_parts(
        parent: &Agent,
        template: &AgentTemplate,
        model: String,
        default_protocol: ApiProtocol,
        model_protocols: HashMap<String, ApiProtocol>,
        model_catalog: HashMap<String, ModelRequestMetadata>,
        retry_config: RetryConfig,
        runtime_route: Option<Arc<ResolvedModelRoute>>,
        runtime_catalog: Option<ResolvedRuntimeCatalog>,
        max_tool_calls_override: Option<usize>,
    ) -> Agent {
        let mut prelude = parent.prelude.clone();
        prelude.push(PromptMessage::system(template.system_prompt.clone()));

        let mut permission_session = parent
            .permission_session
            .lock()
            .expect("permission session poisoned")
            .fork_without_grants();
        if template.permission_mode != PermissionMode::Default {
            permission_session.set_mode(template.permission_mode);
        }

        Agent {
            model: model.clone(),
            primary_route: None,
            subagent_model_overrides: parent.subagent_model_overrides.clone(),
            default_protocol,
            model_protocols,
            model_catalog,
            session_reasoning_efforts: parent.session_reasoning_efforts.clone(),
            prelude,
            runtime_snapshot: Agent::fresh_runtime_snapshot(&model),
            protocol_append_state: crate::protocol_frames::ProtocolAppendState::empty(),
            tools: parent
                .tools
                .scoped(template.tool_scope)
                .without_tools(&[tool_names::TOOL_MEMORY_RECALL]),
            skill_registry: parent.skill_registry.clone(),
            skill_cards: parent.skill_cards.clone(),
            subagent_delegate: None,
            subagent_child_factory: parent.subagent_child_factory.clone(),
            primary_route_factory: parent.primary_route_factory.clone(),
            question_handler: None,
            auto_review_service: if template.name == "reviewer" {
                None
            } else {
                parent.auto_review_service.clone()
            },
            permission_session: Arc::new(Mutex::new(permission_session)),
            // Scope is installed per invocation by SubagentPool::start_run.
            subagent_path_scope: None,
            compaction_config: parent.compaction_config.clone(),
            retry_config,
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
            turn_continuation_provider: None,
            logical_request_observations: LogicalRequestObservationTracker::default(),
            active_epoch: None,
            provider_usage_anchor: None,
            request_projection_generation: 0,
            pressure_compaction_suppressed: false,
            fast_mode: parent.fast_mode.clone(),
            // Subagents always run with the full catalog and regular context;
            // the anchored bootstrap wraps only the primary session.
            anchored: None,
            anchored_override: true,
            anchored_request_phase: None,
            fake_client: parent.fake_client,
            fake_installation_id: parent.fake_installation_id.clone(),
            fake_identity: parent.fake_identity.clone(),
            resolved_model_route: runtime_route,
            resolved_runtime_catalog: runtime_catalog
                .or_else(|| parent.resolved_runtime_catalog.clone()),
            retained_route_preparations: parent.retained_route_preparations.clone(),
        }
    }
}

impl Agent {
    fn preview_final_logical_request(
        &self,
        build: &crate::request_builder::BuildResult,
    ) -> AdjacentRequestObservation {
        self.logical_request_observations
            .preview(crate::request_builder::observe_logical_request(build))
    }

    fn preview_logical_observation(
        &self,
        observation: crate::request_builder::LogicalRequestObservation,
    ) -> AdjacentRequestObservation {
        self.logical_request_observations.preview(observation)
    }

    fn commit_logical_observation(
        &mut self,
        observation: crate::request_builder::LogicalRequestObservation,
    ) {
        self.logical_request_observations.commit(observation);
    }

    fn clear_active_epoch(&mut self) {
        self.active_epoch = None;
    }

    #[cfg(test)]
    fn resolved_epoch_preview_for_test(
        &self,
        protocol: ApiProtocol,
        turn_prelude: &[PromptMessage],
        tools: &[crate::request_builder::ToolSpec],
    ) -> Result<ActiveEpochPreview> {
        match self.prepare_active_epoch(protocol, turn_prelude, tools)? {
            ActiveEpochPreparation::Warm(preview) => Ok(preview),
            ActiveEpochPreparation::ColdRequired(_) => {
                self.preview_active_epoch(protocol, turn_prelude, tools)
            }
        }
    }

    fn invalidate_request_projection(&mut self) {
        self.request_projection_generation = self.request_projection_generation.saturating_add(1);
        self.clear_active_epoch();
    }

    fn frozen_evidence(&self) -> Option<crate::request_builder::FrozenEvidence> {
        self.turn
            .frozen_evidence
            .as_ref()
            .map(|evidence| crate::request_builder::FrozenEvidence {
                message: evidence.message.clone(),
                selected_ids: evidence.selected_ids.clone(),
            })
    }

    fn effective_frozen_evidence_for_preview(
        &self,
        preview: &ActiveEpochPreview,
    ) -> crate::request_builder::FrozenEvidence {
        self.frozen_evidence()
            .unwrap_or_else(|| crate::request_builder::FrozenEvidence {
                message: preview.build.selected_evidence_message.clone(),
                selected_ids: preview.build.selected_evidence_ids.clone(),
            })
    }

    fn clear_provider_usage_anchor(&mut self) {
        self.provider_usage_anchor = None;
    }

    /// Proc-local state (process identities, baselines, cache-prefix adjacency)
    /// must not bleed into a resumed read-model. Resume clears it wholesale.
    fn clear_resume_proc_local(&mut self) {
        self.clear_active_epoch();
        self.clear_provider_usage_anchor();
        self.logical_request_observations = LogicalRequestObservationTracker::default();
    }

    /// Active protocol frames projected from the runtime snapshot (single source
    /// of truth). Consumers must read protocol history here, not keep a mirror.
    pub(super) fn active_protocol_frames(&self) -> Vec<crate::protocol_frames::ProtocolFrame> {
        self.runtime_snapshot.active_protocol_frames()
    }

    pub(crate) fn runtime_context(&self) -> Result<crate::runtime_context::RuntimeActiveContext> {
        crate::runtime_context::RuntimeActiveContext::try_from(&self.runtime_snapshot)
    }

    pub(crate) fn reconcile_loaded_skill_material(&mut self) -> Result<()> {
        let before = self.runtime_snapshot.clone();
        reconcile_snapshot_skill_material(&mut self.runtime_snapshot)?;
        if self.runtime_snapshot != before {
            self.invalidate_request_projection();
        }
        Ok(())
    }

    fn appended_protocol_frames(
        &self,
        previous: &ActiveEpoch,
    ) -> Option<Vec<crate::protocol_frames::ProtocolFrame>> {
        let appended_ids = self.protocol_append_state.appended_frame_ids(
            previous.protocol_frontier_token,
            previous.protocol_append_generation,
            previous.protocol_frontier_count,
        )?;
        if appended_ids.is_empty() {
            return Some(Vec::new());
        }
        let mut suffix = Vec::with_capacity(appended_ids.len());
        let mut next = 0usize;
        for runtime_frame in &self.runtime_snapshot.frames {
            if next == appended_ids.len() {
                break;
            }
            if runtime_frame.id != appended_ids[next] {
                continue;
            }
            if runtime_frame.visibility != FrameVisibility::Active {
                return None;
            }
            let item = runtime_frame.protocol.clone()?;
            suffix.push(crate::protocol_frames::ProtocolFrame {
                runtime_frame_id: Some(runtime_frame.id),
                source_provenance: Some(runtime_frame.provenance.clone()),
                history_index: previous.protocol_frontier_count + next,
                item,
            });
            next += 1;
        }
        (next == appended_ids.len()).then_some(suffix)
    }

    pub(super) fn active_history_items(&self) -> Vec<HistoryItem> {
        crate::protocol_frames::history_items_from_frames(&self.active_protocol_frames())
    }

    fn install_provider_usage_anchor(&mut self, usage: TokenUsageEstimate) {
        let frames = self.active_protocol_frames();
        self.provider_usage_anchor = Some(ProviderUsageAnchor {
            usage,
            protocol_frontier_count: frames.len(),
            protocol_prefix_digest: protocol_prefix_digest(&frames),
        });
    }

    /// Returns provider usage plus the local estimate for frames appended since
    /// that provider response. A stale frontier deliberately fails open.
    pub(super) fn projected_token_usage(&self) -> Option<TokenUsageEstimate> {
        let anchor = self.provider_usage_anchor.as_ref()?;
        let frames = self.active_protocol_frames();
        if frames.len() < anchor.protocol_frontier_count
            || protocol_prefix_digest(&frames[..anchor.protocol_frontier_count])
                != anchor.protocol_prefix_digest
        {
            return None;
        }

        let trailing_tokens = frames[anchor.protocol_frontier_count..]
            .iter()
            .map(|frame| estimate_trailing_history_item_tokens(&frame.to_history_item()))
            .sum::<u64>();
        Some(TokenUsageEstimate {
            used_tokens: anchor.usage.used_tokens.saturating_add(trailing_tokens),
            context_window_tokens: anchor.usage.context_window_tokens,
            input_tokens: anchor.usage.input_tokens.saturating_add(trailing_tokens),
            output_tokens: anchor.usage.output_tokens,
            cached_tokens: anchor.usage.cached_tokens,
        })
    }

    /// Pure active-epoch preview. The returned token is committed only after the
    /// prepared telemetry callback succeeded and immediately before transport.
    pub(super) fn prepare_active_epoch(
        &self,
        protocol: ApiProtocol,
        _turn_prelude: &[PromptMessage],
        tools: &[crate::request_builder::ToolSpec],
    ) -> Result<ActiveEpochPreparation> {
        let Some(previous) = self.active_epoch.as_ref() else {
            return Ok(ActiveEpochPreparation::ColdRequired(
                ColdRequiredReason::NoActiveEpoch,
            ));
        };
        if previous.turn_id != self.turn.turn_id {
            return Ok(ActiveEpochPreparation::ColdRequired(
                ColdRequiredReason::TurnChanged,
            ));
        }
        if previous.projection_generation != self.request_projection_generation {
            return Ok(ActiveEpochPreparation::ColdRequired(
                ColdRequiredReason::RequestProjectionChanged,
            ));
        }
        if previous.budget.truncated {
            return Ok(ActiveEpochPreparation::ColdRequired(
                ColdRequiredReason::TruncatedOrReselected,
            ));
        }
        let Some(suffix) = self.appended_protocol_frames(previous) else {
            return Ok(ActiveEpochPreparation::ColdRequired(
                ColdRequiredReason::ProtocolFrontierChanged,
            ));
        };
        if suffix.is_empty() {
            return Ok(ActiveEpochPreparation::ColdRequired(
                ColdRequiredReason::UnsupportedAppendShape,
            ));
        }
        if crate::protocol_frames::validate_history_items_complete(
            &crate::protocol_frames::history_items_from_frames(&suffix),
            None,
        )
        .is_err()
        {
            return Ok(ActiveEpochPreparation::ColdRequired(
                ColdRequiredReason::SuffixValidationFailed,
            ));
        }
        let suffix_tokens = suffix
            .iter()
            .map(|frame| {
                crate::request_builder::estimate_history_item_tokens(&frame.to_history_item())
            })
            .sum::<u64>();
        if previous
            .budget
            .estimated_request_tokens
            .saturating_add(suffix_tokens)
            > previous
                .budget
                .input_budget_tokens
                .saturating_add(previous.budget.estimated_tools_tokens)
        {
            return Ok(ActiveEpochPreparation::ColdRequired(
                ColdRequiredReason::BudgetRequiresColdPlan,
            ));
        }
        match self.preview_warm_active_epoch(protocol, tools, previous, &suffix) {
            Ok(preview) => Ok(ActiveEpochPreparation::Warm(preview)),
            Err(_) => Ok(ActiveEpochPreparation::ColdRequired(
                ColdRequiredReason::UnsupportedAppendShape,
            )),
        }
    }

    fn preview_active_epoch(
        &self,
        _protocol: ApiProtocol,
        turn_prelude: &[PromptMessage],
        tools: &[crate::request_builder::ToolSpec],
    ) -> Result<ActiveEpochPreview> {
        let model = self.active_model_metadata();
        let frozen = self.frozen_evidence();
        let policy = ProtectedContextPolicy::from_configured_reserve(
            None,
            effective_input_budget_tokens(model.clone(), tools),
        );
        let history = self.active_history_items();
        crate::protocol_frames::validate_history_items_complete(&history, None)?;
        let planned = build_request_with_policy(
            RequestBuilderInput {
                model_id: &self.model,
                model,
                prelude: turn_prelude,
                snapshot: &self.runtime_snapshot,
                tools,
            },
            frozen.as_ref(),
            Some(policy),
        )?;
        self.active_epoch_preview_from_build(planned, ActiveEpochTransition::Cold)
    }

    fn preview_warm_active_epoch(
        &self,
        _protocol: ApiProtocol,
        _tools: &[crate::request_builder::ToolSpec],
        previous: &ActiveEpoch,
        suffix: &[crate::protocol_frames::ProtocolFrame],
    ) -> Result<ActiveEpochPreview> {
        let suffix_plan = crate::request_builder::prompt_plan::build_prompt_plan_suffix(
            &self.model,
            &self.runtime_snapshot,
            suffix,
            previous.committed_plan.segments.len(),
        );
        let mut plan = previous.committed_plan.clone();
        for segment in &mut plan.segments {
            segment.stability = crate::request_builder::prompt_plan::PromptSegmentStability::Stable;
        }
        for contributor in suffix_plan.contributors {
            if let Some(existing) = plan
                .contributors
                .iter_mut()
                .find(|existing| existing.id == contributor.id)
            {
                existing.segment_ids.extend(contributor.segment_ids);
            } else {
                plan.contributors.push(contributor);
            }
        }
        plan.segments.extend(suffix_plan.segments);
        for (order, contributor) in plan.contributors.iter_mut().enumerate() {
            contributor.order = order as u32;
        }
        plan.recompute_cache_metadata();

        let suffix_tokens = suffix
            .iter()
            .map(|frame| {
                crate::request_builder::estimate_history_item_tokens(&frame.to_history_item())
            })
            .sum::<u64>();
        let mut budget = previous.budget;
        budget.original_history_items = budget.original_history_items.saturating_add(suffix.len());
        budget.retained_history_items = budget.retained_history_items.saturating_add(suffix.len());
        budget.estimated_retained_history_tokens = budget
            .estimated_retained_history_tokens
            .saturating_add(suffix_tokens);
        budget.estimated_protected_tokens = budget
            .estimated_protected_tokens
            .saturating_add(suffix_tokens);
        budget.estimated_unaddressable_protected_tokens = budget
            .estimated_unaddressable_protected_tokens
            .saturating_add(suffix_tokens);
        budget.truncated = false;

        let rebuilt = build_request_from_selected_prompt(SelectedPromptRequestInput {
            prompt_plan: plan,
            budget,
            selected_evidence_ids: previous.selected_evidence_ids.clone(),
            selected_evidence_message: previous.selected_evidence_message.clone(),
        })?;
        let observation = observe_logical_request(&rebuilt);
        if self.resolved_model_route.is_none() {
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
        self.active_epoch_preview_from_build(
            rebuilt,
            ActiveEpochTransition::Append {
                added: suffix.len(),
            },
        )
    }

    fn active_epoch_preview_from_build(
        &self,
        build: crate::request_builder::BuildResult,
        _transition: ActiveEpochTransition,
    ) -> Result<ActiveEpochPreview> {
        let observation = observe_logical_request(&build);
        let committed_plan = build.prompt_plan.clone();
        let budget = build.budget;
        let selected_evidence_ids = build.selected_evidence_ids.clone();
        let selected_evidence_message = build.selected_evidence_message.clone();
        Ok(ActiveEpochPreview {
            build,
            epoch: ActiveEpoch {
                turn_id: self.turn.turn_id,
                request_shape_digest: observation.cohort.request_shape_digest.clone(),
                committed_plan,
                protocol_frontier_count: self.protocol_append_state.frame_count(),
                protocol_frontier_token: self.protocol_append_state.frontier_token(),
                protocol_append_generation: self.protocol_append_state.generation(),
                observation,
                projection_generation: self.request_projection_generation,
                budget,
                selected_evidence_ids,
                selected_evidence_message,
            },
            transition: _transition,
        })
    }

    fn commit_active_epoch(&mut self, preview: ActiveEpochPreview) {
        if self.turn.frozen_evidence.is_none() {
            let frozen = self.effective_frozen_evidence_for_preview(&preview);
            self.turn.frozen_evidence = Some(FrozenTurnEvidence {
                message: frozen.message,
                selected_ids: frozen.selected_ids,
            });
        }
        self.active_epoch = Some(preview.epoch);
    }

    fn commit_resolved_active_epoch(
        &mut self,
        mut preview: ActiveEpochPreview,
        observation: crate::request_builder::LogicalRequestObservation,
    ) -> Result<()> {
        if matches!(preview.transition, ActiveEpochTransition::Append { .. })
            && let Some(previous) = self.active_epoch.as_ref()
        {
            ensure!(
                observation.units.len() >= previous.observation.units.len()
                    && observation.units[..previous.observation.units.len()]
                        == previous.observation.units[..],
                "active epoch committed resolved wire units are not an exact prefix"
            );
            ensure!(
                observation.cohort.request_shape_digest == previous.request_shape_digest,
                "active epoch resolved request shape changed while rebuilding"
            );
        }
        preview.epoch.request_shape_digest = observation.cohort.request_shape_digest.clone();
        preview.epoch.observation = observation;
        self.commit_active_epoch(preview);
        Ok(())
    }

    pub fn new(
        model: impl Into<String>,
        max_iterations: impl Into<Option<usize>>,
        max_tool_calls: impl Into<Option<usize>>,
    ) -> Self {
        let max_iterations = max_iterations.into();
        let max_tool_calls = max_tool_calls.into();
        let model = model.into();
        Self {
            model: model.clone(),
            primary_route: None,
            subagent_model_overrides: HashMap::new(),
            default_protocol: ApiProtocol::Responses,
            model_protocols: HashMap::new(),
            model_catalog: HashMap::new(),
            session_reasoning_efforts: HashMap::new(),
            prelude: default_agent_prelude(),
            runtime_snapshot: Agent::fresh_runtime_snapshot(&model),
            protocol_append_state: crate::protocol_frames::ProtocolAppendState::empty(),
            tools: ToolRegistry::default_tools(),
            skill_registry: None,
            skill_cards: Vec::new(),
            subagent_delegate: None,
            subagent_child_factory: None,
            primary_route_factory: None,
            question_handler: None,
            auto_review_service: None,
            permission_session: Arc::new(Mutex::new(PermissionSessionState::default())),
            subagent_path_scope: None,
            compaction_config: CompactionConfig::default(),
            retry_config: RetryConfig::default(),
            tool_timeout_secs: Some(60),
            fake_client: None,
            fake_installation_id: crate::fake::CodexIdentity::new("letcode").installation_id,
            fake_identity: None,
            resolved_model_route: None,
            resolved_runtime_catalog: None,
            retained_route_preparations: HashMap::new(),
            turn: TurnRuntimeState::default(),
            next_turn_id: 0,
            max_iterations,
            max_tool_calls,
            context_scope_state: Arc::new(std::sync::Mutex::new(ContextScopeState::default())),
            runtime_snapshot_provider: None,
            turn_continuation_provider: None,
            logical_request_observations: LogicalRequestObservationTracker::default(),
            active_epoch: None,
            provider_usage_anchor: None,
            request_projection_generation: 0,
            pressure_compaction_suppressed: false,
            fast_mode: None,
            anchored: None,
            anchored_override: true,
            anchored_request_phase: None,
        }
    }

    /// Loads global configuration instructions, then the nearest repository's
    /// `AGENTS.md` chain. Project instructions are appended after global ones.
    pub fn load_instruction_files_from(
        &mut self,
        config_dir: &Path,
        current_dir: &Path,
    ) -> Result<()> {
        self.load_instruction_file(&config_dir.join("AGENTS.md"))?;
        self.load_workspace_instructions_from(current_dir)
    }

    /// Discovers the nearest repository root and loads its `AGENTS.md` chain.
    pub fn load_workspace_instructions_from(&mut self, current_dir: &Path) -> Result<()> {
        let workspace_root = current_dir
            .ancestors()
            .find(|ancestor| ancestor.join(".git").exists())
            .unwrap_or(current_dir);
        self.load_workspace_instructions(workspace_root, current_dir)
    }

    /// Loads `AGENTS.md` files from the workspace root through the current directory.
    /// Later files are appended after earlier files so deeper instructions take precedence.
    pub fn load_workspace_instructions(
        &mut self,
        workspace_root: &Path,
        current_dir: &Path,
    ) -> Result<()> {
        let workspace_root = workspace_root.canonicalize().with_context(|| {
            format!(
                "failed to resolve workspace root {}",
                workspace_root.display()
            )
        })?;
        let current_dir = current_dir.canonicalize().with_context(|| {
            format!(
                "failed to resolve current directory {}",
                current_dir.display()
            )
        })?;
        let relative_current_dir =
            current_dir.strip_prefix(&workspace_root).with_context(|| {
                format!(
                    "current directory {} is outside workspace root {}",
                    current_dir.display(),
                    workspace_root.display()
                )
            })?;

        let mut directories = vec![workspace_root.clone()];
        let mut directory = workspace_root;
        for component in relative_current_dir.components() {
            directory.push(component);
            directories.push(directory.clone());
        }

        for directory in directories {
            self.load_instruction_file(&directory.join("AGENTS.md"))?;
        }

        Ok(())
    }

    fn load_instruction_file(&mut self, path: &Path) -> Result<()> {
        if !path.is_file() {
            return Ok(());
        }
        let path = path
            .canonicalize()
            .with_context(|| format!("failed to resolve {}", path.display()))?;
        let marker = format!("来自 {} 的指令：\n", path.display());
        if self
            .prelude
            .iter()
            .any(|message| message.text.starts_with(&marker))
        {
            return Ok(());
        }
        let content = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        self.prelude
            .push(PromptMessage::system(format!("{marker}{content}")));
        self.invalidate_request_projection();
        Ok(())
    }

    pub fn set_fast_mode(&mut self, fast_mode: Arc<crate::fast_mode::FastMode>) {
        self.fast_mode = Some(fast_mode);
    }

    pub fn set_fake_installation_id(&mut self, installation_id: impl Into<String>) {
        let installation_id = installation_id.into();
        if installation_id.is_empty() {
            return;
        }
        self.fake_installation_id = installation_id;
        if let Some(identity) = &mut self.fake_identity {
            identity.installation_id = self.fake_installation_id.clone();
        }
    }

    pub(crate) fn set_fake_client(
        &mut self,
        client: Option<crate::fake::FakeClient>,
    ) -> Result<()> {
        if let Some(client) = client {
            let protocol = self.active_protocol();
            if !client.supports_protocol(protocol) {
                bail!(
                    "fake mode '{}' is not supported by the current model protocol '{}'; use {}",
                    client.as_str(),
                    protocol.as_str(),
                    match protocol {
                        ApiProtocol::Responses => "codex or auto",
                        ApiProtocol::Anthropic => "anthropic or auto",
                        ApiProtocol::Completions => "off",
                    }
                );
            }
        }
        self.fake_client = client;
        self.fake_identity =
            client.map(|_| crate::fake::CodexIdentity::new(self.fake_installation_id.clone()));
        Ok(())
    }

    pub(crate) fn fake_client(&self) -> Option<crate::fake::FakeClient> {
        self.fake_client
    }

    pub(crate) fn fake_turn_context(
        &self,
        profile: crate::fake::FakeClient,
    ) -> Option<crate::fake::CodexRequestContext> {
        (self.fake_client == Some(crate::fake::FakeClient::Auto)
            || self.fake_client == Some(profile))
        .then(|| {
            self.fake_identity
                .as_ref()
                .map(|identity| identity.turn_context())
        })
        .flatten()
    }

    /// Enable the anchored bootstrap experiment. Fails fast when the alias
    /// tools or any compaction tool do not exist in the registry — a
    /// composition drift must be visible at startup, not per request.
    /// Human-readable status of the anchored bootstrap experiment for the
    /// `/anchored` slash command: enablement, whitelist match, current phase,
    /// and the phase's catalog shape.
    /// Whether the anchored bootstrap experiment currently applies to this
    /// session: enabled by config, model on the whitelist, and the runtime
    /// switch is on.
    pub(crate) fn anchored_active(&self) -> bool {
        self.anchored_override
            && self
                .anchored
                .as_ref()
                .is_some_and(|anchored| anchored.enabled_for(&self.model))
    }

    /// Flip the runtime switch. Mirrors Fast Mode semantics: on a model outside
    /// the whitelist (or when the experiment is not configured) the switch is
    /// left untouched and reported as unavailable — the experiment simply does
    /// not apply to this model, which is the expected state, not an error.
    pub(crate) fn toggle_anchored(&mut self) -> String {
        let Some(anchored) = &self.anchored else {
            return "anchored: unavailable".to_string();
        };
        if !anchored.enabled_for(&self.model) {
            return "anchored: unavailable for current model".to_string();
        }
        self.anchored_override = !self.anchored_override;
        self.anchored_status()
    }

    /// Concise status line for the `/anchored` slash command toast.
    pub(crate) fn anchored_status(&self) -> String {
        let Some(anchored) = &self.anchored else {
            return "anchored: unavailable".to_string();
        };
        if !anchored.enabled_for(&self.model) {
            return "anchored: unavailable for current model".to_string();
        }
        if !self.anchored_override {
            return "anchored: off".to_string();
        }
        format!(
            "anchored: on ({})",
            anchored.phase(&self.active_history_items()).as_str()
        )
    }

    pub fn set_anchored(&mut self, anchored: Option<AnchoredBootstrap>) -> Result<()> {
        if let Some(anchored) = &anchored {
            let available: std::collections::BTreeSet<String> = self
                .tools
                .specs()
                .iter()
                .map(|spec| spec.name.clone())
                .collect();
            for required in [
                tool_names::TOOL_SHELL_EXEC,
                tool_names::TOOL_EDIT_APPLY_PATCH,
            ] {
                if !available.contains(required) {
                    bail!(
                        "anchored_bootstrap: required tool '{required}' is not registered; the alias pair cannot be assembled"
                    );
                }
            }
            for tool in anchored.compaction_tools() {
                if !available.contains(tool) {
                    bail!(
                        "anchored_bootstrap: compaction_tools entry '{tool}' is not a registered tool"
                    );
                }
            }
        }
        self.anchored = anchored;
        Ok(())
    }

    /// Resolve an anchored-bootstrap alias tool name (bash / str_replace_editor)
    /// to the real registry name. Only aliases produced by a non-promoted
    /// request are resolved — a promoted request's catalog has no alias names,
    /// and a same-named MCP tool must never be hijacked. The phase is the one
    /// bound once per turn by the prelude hook.
    pub(crate) fn resolve_tool_alias(&self, name: &str) -> String {
        if let Some(anchored) = &self.anchored
            && anchored.enabled_for(&self.model)
            && self.anchored_override
            && matches!(
                self.anchored_request_phase,
                Some(AnchoredPhase::Bootstrap) | Some(AnchoredPhase::CompactedFallback)
            )
        {
            return anchored.resolve_tool_name(name);
        }
        name.to_string()
    }

    pub fn fast_mode(&self) -> Option<&Arc<crate::fast_mode::FastMode>> {
        self.fast_mode.as_ref()
    }

    pub fn fast_mode_enabled(&self) -> bool {
        self.fast_mode
            .as_ref()
            .is_some_and(|fast_mode| fast_mode.enabled())
    }

    pub(crate) fn auto_disable_fast_mode_for_model(&self, model_id: &str) -> Result<bool> {
        self.fast_mode.as_ref().map_or(Ok(false), |fast_mode| {
            fast_mode.auto_disable_for_model(model_id)
        })
    }

    pub(crate) fn prepare_fast_mode_auto_disable(
        &self,
        model_id: &str,
    ) -> Result<Option<crate::fast_mode::PreparedFastModeDisable>> {
        self.fast_mode.as_ref().map_or(Ok(None), |fast_mode| {
            fast_mode.prepare_auto_disable_for_model(model_id)
        })
    }

    /// Returns whether request preparation auto-disabled Fast Mode.
    pub(crate) fn prepare_fast_mode_for_request(&mut self) -> Result<bool> {
        self.auto_disable_fast_mode_for_model(&self.model)
    }

    pub fn set_model_catalog(&mut self, catalog: HashMap<String, ModelRequestMetadata>) {
        self.model_catalog = catalog;
        self.invalidate_request_projection();
    }

    pub fn set_default_protocol(&mut self, protocol: ApiProtocol) {
        self.default_protocol = protocol;
        self.invalidate_request_projection();
    }

    pub fn set_model_protocols(&mut self, protocols: HashMap<String, ApiProtocol>) {
        self.model_protocols = protocols;
        self.invalidate_request_projection();
    }

    pub fn set_compaction_config(&mut self, config: CompactionConfig) {
        self.compaction_config = config;
        self.invalidate_request_projection();
    }

    pub(crate) fn compaction_config(&self) -> &CompactionConfig {
        &self.compaction_config
    }

    pub fn set_tool_timeout_secs(&mut self, timeout_secs: Option<u64>) {
        self.tool_timeout_secs = timeout_secs;
    }

    pub(crate) fn tool_timeout_secs(&self) -> Option<u64> {
        self.tool_timeout_secs
    }

    pub fn set_tool_parallelism(
        &mut self,
        parallelism: impl IntoIterator<Item = (String, crate::tool::ToolParallelism)>,
    ) -> Result<()> {
        self.tools.set_parallelism_overrides(parallelism)
    }

    pub(crate) fn tool_parallelism_overrides(
        &self,
    ) -> &std::collections::BTreeMap<String, crate::tool::ToolParallelism> {
        self.tools.parallelism_overrides()
    }

    pub fn set_retry_config(&mut self, config: RetryConfig) {
        self.retry_config = config;
        self.invalidate_request_projection();
    }

    pub(crate) fn retry_config(&self) -> &RetryConfig {
        &self.retry_config
    }

    pub(crate) fn model_catalog(&self) -> &HashMap<String, ModelRequestMetadata> {
        &self.model_catalog
    }

    pub(crate) fn model_protocols(&self) -> &HashMap<String, ApiProtocol> {
        &self.model_protocols
    }

    pub(crate) fn default_protocol(&self) -> ApiProtocol {
        self.default_protocol
    }

    pub(crate) fn active_protocol(&self) -> ApiProtocol {
        self.protocol_for_model(&self.model)
    }

    fn protocol_for_model(&self, model_id: &str) -> ApiProtocol {
        self.model_protocols
            .get(model_id)
            .cloned()
            .unwrap_or(self.default_protocol)
    }

    pub(crate) fn active_model_metadata(&self) -> ModelRequestMetadata {
        let mut metadata = self.model_metadata_for(&self.model);
        if let Some(effort) = self
            .session_reasoning_efforts
            .get(&self.active_reasoning_effort_key())
        {
            metadata.reasoning_effort = Some(effort.clone());
        }
        metadata.fast_mode = self.fast_mode_enabled();
        metadata
    }

    fn active_reasoning_effort_key(&self) -> String {
        self.primary_route
            .as_ref()
            .filter(|route| route.model == self.model)
            .map_or_else(|| self.model.clone(), ModelRoute::display_name)
    }

    fn model_metadata_for(&self, model_id: &str) -> ModelRequestMetadata {
        self.model_catalog
            .get(model_id)
            .cloned()
            .unwrap_or(ModelRequestMetadata {
                context_window: None,
                max_output_tokens: None,
                supports_tools: true,
                supports_reasoning: false,
                parallel_tool_calls: true,
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
        &self.runtime_snapshot.workflow.todos
    }

    #[cfg(test)]
    fn auto_continue(&self) -> &AutoContinueState {
        &self.runtime_snapshot.workflow.auto_continue
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

    pub(crate) fn set_subagent_path_scope(&mut self, scope: Option<Arc<SubagentPathScope>>) {
        self.subagent_path_scope = scope;
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn primary_route(&self) -> Option<&ModelRoute> {
        self.primary_route.as_ref()
    }

    #[allow(dead_code)]
    pub fn resolved_model_route(&self) -> Option<&Arc<ResolvedModelRoute>> {
        self.resolved_model_route.as_ref()
    }

    pub fn route_display_name(&self) -> String {
        self.primary_route()
            .map_or_else(|| self.model().to_string(), ModelRoute::display_name)
    }

    pub fn set_primary_route(&mut self, route: ModelRoute) {
        self.primary_route = Some(route);
        self.invalidate_request_projection();
    }

    #[allow(dead_code)]
    pub(crate) fn set_resolved_model_route(&mut self, route: Option<Arc<ResolvedModelRoute>>) {
        self.resolved_model_route = route;
    }

    pub(crate) fn resolved_runtime_catalog(&self) -> Option<&ResolvedRuntimeCatalog> {
        self.resolved_runtime_catalog.as_ref()
    }

    pub(crate) fn resolved_model_route_for(
        &self,
        route: &ModelRoute,
    ) -> Option<Arc<ResolvedModelRoute>> {
        if self.primary_route.as_ref() == Some(route) {
            return self.resolved_model_route.clone();
        }
        self.resolved_runtime_catalog
            .as_ref()
            .and_then(|catalog| catalog.route(&route.provider, &route.model))
            .cloned()
            .map(Arc::new)
            .or_else(|| {
                self.retained_route_preparations
                    .get(&route.display_name())
                    .map(|retained| Arc::clone(&retained.runtime_route))
            })
    }

    pub(crate) fn set_resolved_runtime_catalog(&mut self, catalog: Option<ResolvedRuntimeCatalog>) {
        self.resolved_runtime_catalog = catalog;
    }

    pub fn reasoning_effort(&self) -> Option<ModelReasoningEffort> {
        self.active_model_metadata().reasoning_effort
    }

    pub(crate) fn clear_session_reasoning_efforts(&mut self) {
        self.session_reasoning_efforts.clear();
    }

    pub fn session_token_usage(&self) -> Result<TokenUsageEstimate> {
        self.candidate_session_token_usage(&self.model, &self.runtime_snapshot)
    }

    pub(crate) fn candidate_session_usage_with_composition(
        &self,
        model_id: &str,
        runtime_snapshot: &RuntimeSnapshot,
    ) -> Result<(TokenUsageEstimate, Vec<PromptCompositionEntry>)> {
        let (build, tools) = self.build_candidate_session_request(model_id, runtime_snapshot)?;
        if let Some(route) = self.resolved_model_route.as_deref() {
            self.validate_resolved_candidate_request(
                route,
                self.active_model_metadata(),
                &tools,
                &build,
            )?;
        }
        let usage = TokenUsageEstimate {
            used_tokens: build.budget.estimated_request_tokens,
            context_window_tokens: build.budget.context_window_tokens,
            input_tokens: build.budget.estimated_request_tokens,
            output_tokens: 0,
            cached_tokens: 0,
        };
        let composition = build
            .prompt_plan
            .composition(build.budget.estimated_tools_tokens);
        Ok((usage, composition))
    }

    /// Estimate a prospective session request without changing this agent's
    /// selected model, history, runtime snapshot, or turn state.
    pub(crate) fn candidate_session_token_usage(
        &self,
        model_id: &str,
        runtime_snapshot: &RuntimeSnapshot,
    ) -> Result<TokenUsageEstimate> {
        self.candidate_session_usage_with_composition(model_id, runtime_snapshot)
            .map(|(usage, _)| usage)
    }

    fn candidate_session_usage_with_route(
        &self,
        route: &ModelRoute,
        model_catalog: &HashMap<String, ModelRequestMetadata>,
        runtime_route: Option<&ResolvedModelRoute>,
        runtime_snapshot: &RuntimeSnapshot,
    ) -> Result<(TokenUsageEstimate, Vec<PromptCompositionEntry>)> {
        let (build, tools) = self.build_candidate_session_request_with_route(
            &route.model,
            model_catalog,
            runtime_snapshot,
        )?;
        if let Some(runtime_route) = runtime_route {
            self.validate_resolved_candidate_request(
                runtime_route,
                model_catalog.get(&route.model).cloned().unwrap_or_default(),
                &tools,
                &build,
            )?;
        }
        let usage = TokenUsageEstimate {
            used_tokens: build.budget.estimated_request_tokens,
            context_window_tokens: build.budget.context_window_tokens,
            input_tokens: build.budget.estimated_request_tokens,
            output_tokens: 0,
            cached_tokens: 0,
        };
        let composition = build
            .prompt_plan
            .composition(build.budget.estimated_tools_tokens);
        Ok((usage, composition))
    }

    fn validate_resolved_candidate_request(
        &self,
        route: &ResolvedModelRoute,
        model: ModelRequestMetadata,
        tools: &[crate::request_builder::ToolSpec],
        build: &crate::request_builder::BuildResult,
    ) -> Result<()> {
        let input = crate::model_runtime::projection::model_request_from_prompt_plan(
            route,
            &model,
            &build.prompt_plan,
            tools,
        )
        .map_err(anyhow::Error::msg)?;
        route
            .binding
            .prepare_request(&input)
            .map_err(|failure| anyhow!(failure.to_string()))?;
        Ok(())
    }

    fn build_candidate_session_request(
        &self,
        model_id: &str,
        runtime_snapshot: &RuntimeSnapshot,
    ) -> Result<(
        crate::request_builder::BuildResult,
        Vec<crate::request_builder::ToolSpec>,
    )> {
        self.build_candidate_session_request_with_route(
            model_id,
            &self.model_catalog,
            runtime_snapshot,
        )
    }

    fn build_candidate_session_request_with_route(
        &self,
        model_id: &str,
        model_catalog: &HashMap<String, ModelRequestMetadata>,
        runtime_snapshot: &RuntimeSnapshot,
    ) -> Result<(
        crate::request_builder::BuildResult,
        Vec<crate::request_builder::ToolSpec>,
    )> {
        let model = model_catalog
            .get(model_id)
            .cloned()
            .unwrap_or(ModelRequestMetadata {
                context_window: None,
                max_output_tokens: None,
                supports_tools: true,
                supports_reasoning: false,
                parallel_tool_calls: true,
                ..Default::default()
            });
        let candidate_history = crate::request_builder::history_items_from_frames(
            &crate::request_builder::provider_visible_protocol_frames(runtime_snapshot),
        );
        let mut tools = self.tools.specs();
        tools.retain(|spec| !is_subagent_tool_name(&spec.name));
        tools.extend(
            subagent_tool_specs()
                .into_iter()
                .filter(|spec| is_executable_tool(self, &spec.name)),
        );
        if self.subagent_delegate.is_some() {
            tools.extend(subagent_control_tool_specs());
        }
        let anchored_phase = self
            .anchored
            .as_ref()
            .filter(|anchored| anchored.enabled_for(model_id) && self.anchored_override)
            .map(|anchored| anchored.phase(&candidate_history));
        if let (Some(anchored), Some(phase)) = (&self.anchored, anchored_phase) {
            tools = anchored.tool_catalog(&phase, tools);
        }
        let policy = ProtectedContextPolicy::from_configured_reserve(
            None,
            effective_input_budget_tokens(model.clone(), &tools),
        );
        let runtime_message = runtime_context_message();
        let skill_message = self.skill_prelude_message();
        let prelude = if let (Some(anchored), Some(phase)) = (&self.anchored, anchored_phase) {
            anchored.prelude(
                &phase,
                &self.prelude,
                Some(runtime_message),
                skill_message,
                None,
                &[],
            )
        } else {
            let mut prelude = self.prelude.clone();
            prelude.push(runtime_message);
            if let Some(message) = skill_message {
                prelude.push(message);
            }
            prelude
        };
        let build = build_request_with_policy(
            RequestBuilderInput {
                model_id,
                model,
                prelude: &prelude,
                snapshot: runtime_snapshot,
                tools: &tools,
            },
            None,
            Some(policy),
        )?;
        Ok((build, tools))
    }

    #[cfg(test)]
    pub fn tool_scope(&self) -> ToolScope {
        self.tools.scope()
    }

    #[cfg(test)]
    pub(crate) fn max_tool_calls_limit(&self) -> Option<usize> {
        self.max_tool_calls
    }

    #[cfg(test)]
    pub(crate) fn max_iterations_limit(&self) -> Option<usize> {
        self.max_iterations
    }

    pub fn set_model(&mut self, model: impl Into<String>) {
        self.model = model.into();
        self.runtime_snapshot.latest_model = Some(self.model.clone());
        self.clear_provider_usage_anchor();
        self.invalidate_request_projection();
    }

    pub(crate) fn set_model_route_authority(
        &mut self,
        route: ModelRoute,
        resolved_route: Arc<ResolvedModelRoute>,
    ) {
        self.primary_route = Some(route.clone());
        self.resolved_model_route = Some(Arc::clone(&resolved_route));
        self.set_model(route.model);
    }

    #[allow(dead_code)]
    pub fn apply_route(
        &mut self,
        route: ModelRoute,
        default_protocol: ApiProtocol,
        model_protocols: HashMap<String, ApiProtocol>,
        model_catalog: HashMap<String, ModelRequestMetadata>,
        retry_config: RetryConfig,
    ) {
        self.apply_route_with_runtime_route(
            route,
            default_protocol,
            model_protocols,
            model_catalog,
            retry_config,
            None,
            None,
        );
    }

    fn apply_route_with_runtime_route(
        &mut self,
        route: ModelRoute,
        default_protocol: ApiProtocol,
        model_protocols: HashMap<String, ApiProtocol>,
        model_catalog: HashMap<String, ModelRequestMetadata>,
        retry_config: RetryConfig,
        runtime_route: Option<Arc<ResolvedModelRoute>>,
        runtime_catalog: Option<ResolvedRuntimeCatalog>,
    ) {
        self.default_protocol = default_protocol;
        self.model_protocols = model_protocols;
        self.model_catalog = model_catalog;
        self.retry_config = retry_config;
        self.primary_route = Some(route.clone());
        self.resolved_model_route = runtime_route;
        self.resolved_runtime_catalog = runtime_catalog;
        self.set_model(route.model);
    }

    #[cfg(test)]
    pub(crate) fn default_protocol_for_test(&self) -> ApiProtocol {
        self.default_protocol
    }

    #[cfg(test)]
    pub(crate) fn retry_config_for_test(&self) -> &RetryConfig {
        &self.retry_config
    }

    pub fn subagent_model_override(&self, agent_name: &str) -> Option<&str> {
        self.subagent_model_overrides
            .get(agent_name)
            .map(String::as_str)
    }

    #[allow(dead_code)]
    pub fn set_subagent_model_override(
        &mut self,
        agent_name: impl Into<String>,
        model: impl Into<String>,
    ) {
        self.subagent_model_overrides
            .insert(agent_name.into(), model.into());
    }

    pub fn set_reasoning_effort(&mut self, effort: ModelReasoningEffort) -> Result<()> {
        let metadata = self.active_model_metadata();
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
        self.session_reasoning_efforts
            .insert(self.active_reasoning_effort_key(), effort);
        self.invalidate_request_projection();
        Ok(())
    }

    #[allow(dead_code)]
    pub fn restore_transcript_messages(&mut self, messages: Vec<ConversationMessage>) {
        let history = messages
            .into_iter()
            .map(|message| match message.role {
                ConversationRole::User => HistoryItem::user(message.content),
                ConversationRole::Assistant => HistoryItem::assistant(message.content),
                ConversationRole::Summary => HistoryItem::context_summary(message.content),
            })
            .collect::<Vec<_>>();
        let frames = crate::protocol_frames::history_items_to_frames(&history);
        self.runtime_snapshot = self
            .rebuilt_runtime_snapshot_from_protocol_frames(&frames, 0, &[])
            .expect("restored transcript messages should remain protocol-compatible");
        self.protocol_append_state = crate::protocol_frames::ProtocolAppendState::from_frames(
            &self.runtime_snapshot.active_protocol_frames(),
            None,
        )
        .expect("restored transcript protocol state is valid");
        self.invalidate_request_projection();
        self.runtime_snapshot.workflow = crate::workflow_state::WorkflowState::default();
        self.clear_resume_proc_local();
    }

    #[allow(dead_code)]
    pub fn restore_evidence(&mut self, evidence: Vec<EvidenceRecord>) -> Result<()> {
        Self::validate_evidence_ids(&evidence)?;
        self.runtime_snapshot.set_evidence(evidence);
        self.invalidate_request_projection();
        Ok(())
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn restore_session_context(
        &mut self,
        messages: Vec<ConversationMessage>,
        evidence: Vec<EvidenceRecord>,
        max_turn_id: u64,
    ) -> Result<()> {
        let history = messages
            .into_iter()
            .map(|message| match message.role {
                ConversationRole::User => HistoryItem::user(message.content),
                ConversationRole::Assistant => HistoryItem::assistant(message.content),
                ConversationRole::Summary => HistoryItem::context_summary(message.content),
            })
            .collect();
        self.restore_session_history(history, evidence, max_turn_id)
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn restore_session_history(
        &mut self,
        history: Vec<HistoryItem>,
        evidence: Vec<EvidenceRecord>,
        max_turn_id: u64,
    ) -> Result<()> {
        Self::validate_evidence_ids(&evidence)?;

        let transcript = crate::protocol_frames::analyze_history_items(&history, None)?;
        let mut runtime_snapshot =
            self.rebuilt_runtime_snapshot_from_protocol_frames(&transcript.frames, 0, &[])?;
        runtime_snapshot.current_turn_id = Some(max_turn_id);
        runtime_snapshot.set_evidence(evidence);
        runtime_snapshot.workflow = crate::workflow_state::WorkflowState::default();

        self.runtime_snapshot = runtime_snapshot;
        self.protocol_append_state = crate::protocol_frames::ProtocolAppendState::from_frames(
            &self.runtime_snapshot.active_protocol_frames(),
            None,
        )?;
        self.invalidate_request_projection();
        self.next_turn_id = max_turn_id;
        self.turn = TurnRuntimeState::default();
        self.clear_resume_proc_local();
        Ok(())
    }

    /// Discard all state that belongs to the current session before creating a
    /// new one. Unlike compatibility rebuilds used by restore and checkout,
    /// this deliberately does not preserve runtime snapshot metadata.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn reset_for_new_session(&mut self) {
        self.runtime_snapshot = Self::fresh_runtime_snapshot(&self.model);
        self.protocol_append_state = crate::protocol_frames::ProtocolAppendState::empty();
        self.turn = TurnRuntimeState::default();
        self.next_turn_id = 0;
        self.clear_resume_proc_local();
        if let Ok(mut permissions) = self.permission_session.lock() {
            permissions.clear_grants();
        }
    }

    /// Validate and normalize a restore package without mutating the live agent.
    ///
    /// Callers that commit durable state must perform this validation before the
    /// commit, then use [`Self::install_validated_runtime_snapshot`] afterwards.
    /// RuntimeSnapshot is the single source of truth: validate its own active
    /// protocol stream, not some external frame list that the caller might
    /// hand in but never install.
    pub fn validate_runtime_snapshot_restore(
        &self,
        mut runtime_snapshot: RuntimeSnapshot,
    ) -> Result<RuntimeSnapshot> {
        reconcile_snapshot_skill_material(&mut runtime_snapshot)?;
        Self::validate_evidence_ids(&runtime_snapshot.evidence)?;
        ensure_active_protocol_source_spans(&mut runtime_snapshot);

        let frames = runtime_snapshot.active_protocol_frames();
        let history = crate::protocol_frames::history_items_from_frames(&frames);
        crate::protocol_frames::analyze_history_items(&history, None)?;
        if runtime_snapshot.latest_model.is_none() {
            runtime_snapshot.latest_model = Some(self.model.clone());
        }
        Ok(runtime_snapshot)
    }

    /// Install a package previously accepted by
    /// [`Self::validate_runtime_snapshot_restore`]. This mutation is infallible.
    pub fn install_validated_runtime_snapshot(&mut self, runtime_snapshot: RuntimeSnapshot) {
        let restored_turn_id = runtime_snapshot.current_turn_id.unwrap_or_default();
        let current_turn_start_index =
            Self::current_turn_start_index_for_snapshot(&runtime_snapshot);
        self.turn = TurnRuntimeState::default();
        self.turn.current_turn_start_index = current_turn_start_index;
        self.runtime_snapshot = runtime_snapshot;
        self.protocol_append_state = crate::protocol_frames::ProtocolAppendState::from_frames(
            &self.runtime_snapshot.active_protocol_frames(),
            current_turn_start_index,
        )
        .expect("validated runtime snapshot protocol state is valid");
        self.next_turn_id = self.next_turn_id.max(restored_turn_id);
        self.clear_resume_proc_local();
    }

    fn current_turn_start_index_for_snapshot(snapshot: &RuntimeSnapshot) -> Option<usize> {
        snapshot.active_protocol_frames().iter().position(|frame| {
            frame
                .runtime_frame_id
                .is_some_and(|id| snapshot.compaction.turn_protected_frame_ids.contains(&id))
        })
    }

    pub fn restore_runtime_snapshot(&mut self, runtime_snapshot: RuntimeSnapshot) -> Result<()> {
        let runtime_snapshot = self.validate_runtime_snapshot_restore(runtime_snapshot)?;
        self.install_validated_runtime_snapshot(runtime_snapshot);
        Ok(())
    }

    pub fn restore_turn_sequence(&mut self, max_turn_id: u64) {
        self.next_turn_id = self.next_turn_id.max(max_turn_id);
    }

    /// Commit a snapshot for a wholly new session.  Unlike ordinary restores,
    /// the turn sequence must not retain an id from the abandoned session.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn restore_new_session_runtime_snapshot(
        &mut self,
        runtime_snapshot: RuntimeSnapshot,
        max_turn_id: u64,
    ) -> Result<()> {
        let runtime_snapshot = self.validate_runtime_snapshot_restore(runtime_snapshot)?;
        self.prepare_new_session_permission_reset()?;
        self.install_new_session_runtime_snapshot(runtime_snapshot, max_turn_id);
        Ok(())
    }

    pub(crate) fn prepare_new_session_permission_reset(&self) -> Result<()> {
        drop(
            self.permission_session
                .lock()
                .map_err(|_| anyhow!("permission session poisoned"))?,
        );
        Ok(())
    }

    pub(crate) fn install_new_session_runtime_snapshot(
        &mut self,
        runtime_snapshot: RuntimeSnapshot,
        max_turn_id: u64,
    ) {
        self.install_validated_runtime_snapshot(runtime_snapshot);
        self.permission_session
            .lock()
            .expect("permission session was validated before new-session install")
            .clear_grants();
        self.next_turn_id = max_turn_id;
    }

    pub fn add_evidence(&mut self, evidence: EvidenceRecord) -> Result<()> {
        require_unique_evidence_id(&self.runtime_snapshot.evidence, &evidence.id)?;
        self.runtime_snapshot.evidence.push(evidence);
        if self.turn.frozen_evidence.is_none() {
            self.invalidate_request_projection();
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn install_provider_usage_anchor_for_test(&mut self, usage: TokenUsageEstimate) {
        self.install_provider_usage_anchor(usage);
    }

    #[cfg(test)]
    pub(crate) fn provider_usage_anchor_for_test(&self) -> Option<TokenUsageEstimate> {
        self.provider_usage_anchor
            .as_ref()
            .map(|anchor| anchor.usage)
    }

    #[cfg(test)]
    pub(crate) fn has_set_logical_request_observation_for_test(&self) -> bool {
        self.logical_request_observations.previous.is_some()
    }

    #[cfg(test)]
    pub(crate) fn runtime_snapshot_for_test(&self) -> &RuntimeSnapshot {
        &self.runtime_snapshot
    }

    #[cfg(test)]
    pub(crate) fn history_for_test(&self) -> Vec<HistoryItem> {
        self.active_history_items()
    }

    #[cfg(test)]
    pub(crate) fn protocol_frames_for_test(&self) -> Vec<crate::protocol_frames::ProtocolFrame> {
        self.active_protocol_frames()
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
        self.invalidate_request_projection();
    }

    pub fn try_register_tool<T>(&mut self, tool: T) -> Result<()>
    where
        T: ToolHandler + 'static,
    {
        self.tools.try_register(tool)?;
        self.invalidate_request_projection();
        Ok(())
    }

    /// Unregister a dynamically owned tool, such as an MCP tool for a server
    /// that has just been disabled. Default tools are otherwise unchanged.
    pub fn unregister_tool(&mut self, name: &str) -> bool {
        let removed = self.tools.remove(name);
        if removed {
            self.invalidate_request_projection();
        }
        removed
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
        self.invalidate_request_projection();
        if registry.is_empty() {
            Ok(())
        } else {
            self.try_register_tool(SkillTool::new(registry.clone()))?;
            self.try_register_tool(SkillResourceListTool::new(registry.clone()))?;
            self.try_register_tool(SkillResourceReadTool::new(registry))
        }
    }

    pub fn set_subagent_delegate(&mut self, delegate: Arc<dyn SubagentDelegate>) {
        self.subagent_delegate = Some(delegate);
    }

    pub fn set_auto_review_service(&mut self, service: Option<Arc<dyn AutoReviewService>>) {
        self.auto_review_service = service;
    }

    pub(crate) async fn resolve_auto_permission(
        &self,
        request: PermissionRequest,
        user_goal: Option<String>,
    ) -> Result<AutoReviewResolution> {
        let Some(service) = self.auto_review_service.as_ref() else {
            bail!("auto permission mode requires a reviewer service");
        };
        service.review(self, request, user_goal).await
    }

    pub fn set_subagent_child_factory(&mut self, factory: Arc<dyn SubagentChildFactory>) {
        self.subagent_child_factory = Some(factory);
    }

    pub fn set_primary_route_factory(&mut self, factory: Arc<dyn PrimaryRouteFactory>) {
        if let (Some(previous_factory), Some(catalog)) = (
            self.primary_route_factory.as_ref(),
            self.resolved_runtime_catalog.as_ref(),
        ) {
            for provider in catalog.providers.values() {
                for route in provider.models.values() {
                    self.retained_route_preparations.insert(
                        format!("{}/{}", route.provider, route.model),
                        RetainedRoutePreparation {
                            runtime_route: Arc::new(route.clone()),
                            route_factory: Arc::clone(previous_factory),
                        },
                    );
                }
            }
        }
        self.primary_route_factory = Some(factory);
    }

    pub fn prepare_primary_route(&self, route: ModelRoute) -> Result<PreparedPrimaryRoute> {
        if self
            .resolved_runtime_catalog
            .as_ref()
            .and_then(|catalog| catalog.route(&route.provider, &route.model))
            .is_some()
        {
            let factory = self
                .primary_route_factory
                .clone()
                .ok_or_else(|| anyhow!("primary route switching is not configured"))?;
            return factory
                .prepare_route(route.clone())?
                .with_resolved_authority_from(self, &route);
        }
        let retained = self
            .retained_route_preparations
            .get(&route.display_name())
            .ok_or_else(|| anyhow!("model route is not available: {}", route.display_name()))?;
        let mut prepared = retained.route_factory.prepare_route(route.clone())?;
        prepared.runtime_route = Some(Arc::clone(&retained.runtime_route));
        prepared.runtime_catalog = self.resolved_runtime_catalog.clone();
        Ok(prepared)
    }

    pub(crate) fn apply_prepared_primary_route(&mut self, route: PreparedPrimaryRouteInstall) {
        route.apply(self);
    }

    pub(crate) fn apply_prepared_route(&mut self, route: PreparedPrimaryRoute) {
        self.apply_prepared_primary_route(route.into_install());
    }

    #[cfg(test)]
    pub(crate) fn switch_primary_route(&mut self, route: ModelRoute) -> Result<()> {
        let prepared = self.prepare_primary_route(route)?;
        self.apply_prepared_route(prepared);
        Ok(())
    }

    pub fn set_context_scope_state(
        &mut self,
        context_scope_state: Arc<std::sync::Mutex<ContextScopeState>>,
    ) {
        self.context_scope_state = context_scope_state;
    }

    pub(crate) fn set_runtime_snapshot_provider(&mut self, provider: RuntimeSnapshotProvider) {
        self.runtime_snapshot_provider = Some(provider);
        self.invalidate_request_projection();
    }

    pub(crate) fn has_runtime_snapshot_provider(&self) -> bool {
        self.runtime_snapshot_provider.is_some()
    }

    pub(crate) fn clear_runtime_snapshot_provider(&mut self) {
        self.runtime_snapshot_provider = None;
        self.invalidate_request_projection();
    }

    pub(crate) fn turn_continuation_provider_guard(
        &mut self,
        provider: TurnContinuationProvider,
    ) -> TurnContinuationProviderGuard<'_> {
        TurnContinuationProviderGuard::install(self, Some(provider))
    }

    async fn drain_turn_continuations<E, Efut>(&mut self, _on_event: &mut E) -> Result<bool>
    where
        E: FnMut(AgentEvent) -> Efut,
        Efut: Future<Output = Result<()>>,
    {
        let Some(provider) = self.turn_continuation_provider.as_ref() else {
            return Ok(false);
        };
        let messages = provider()?;
        self.apply_turn_continuations(messages)
    }

    pub(crate) fn apply_turn_continuations(
        &mut self,
        messages: Vec<PendingTurnContinuation>,
    ) -> Result<bool> {
        if messages.is_empty() {
            return Ok(false);
        }
        self.reload_runtime_snapshot_from_provider()?;
        self.apply_turn_continuation_effects(&messages);
        Ok(true)
    }

    fn apply_turn_continuation_effects(&mut self, messages: &[PendingTurnContinuation]) {
        for message in messages {
            if let Some(result) = &message.result {
                self.record_background_subagent_effects(result);
            }
        }
    }

    // The Agent retains no checkpoint candidate or production control state.
    pub(crate) fn clear_logical_checkpoint_candidate_provider(&mut self) {}

    pub(super) fn reload_runtime_snapshot_from_provider(&mut self) -> Result<()> {
        let provider = self.runtime_snapshot_provider.as_ref().ok_or_else(|| {
            anyhow!("canonical runtime reload requires a runtime snapshot provider")
        })?;
        let mut snapshot = provider().context("failed to project runtime snapshot for reload")?;
        reconcile_snapshot_skill_material(&mut snapshot)?;
        Self::validate_evidence_ids(&snapshot.evidence)?;
        ensure_active_protocol_source_spans(&mut snapshot);
        let protocol_frames = snapshot.active_protocol_frames();
        let history = crate::protocol_frames::history_items_from_frames(&protocol_frames);
        crate::protocol_frames::analyze_history_items(&history, None)?;
        let current_turn_start_index = protocol_frames.iter().position(|frame| {
            frame
                .runtime_frame_id
                .is_some_and(|id| snapshot.compaction.turn_protected_frame_ids.contains(&id))
        });
        let restored_turn_id = snapshot.current_turn_id.unwrap_or_default();

        self.runtime_snapshot = snapshot;
        self.protocol_append_state = crate::protocol_frames::ProtocolAppendState::from_frames(
            &self.runtime_snapshot.active_protocol_frames(),
            current_turn_start_index,
        )?;
        self.turn.current_turn_start_index = current_turn_start_index;
        self.next_turn_id = self.next_turn_id.max(restored_turn_id);
        self.clear_resume_proc_local();
        Ok(())
    }

    /// Replace the active runtime with the provider's canonical projection.
    /// Unlike refresh, a context scope transition must not retain frames,
    /// contributors, or protocol identity from the outgoing scope.
    #[cfg(test)]
    fn replace_runtime_snapshot_from_provider(&mut self) -> Result<()> {
        let provider = self.runtime_snapshot_provider.as_ref().ok_or_else(|| {
            anyhow!("successful context scope transition requires a runtime snapshot provider")
        })?;
        let mut snapshot = provider().context("failed to project replacement runtime snapshot")?;
        reconcile_snapshot_skill_material(&mut snapshot)?;
        ensure_active_protocol_source_spans(&mut snapshot);
        let protocol_frames = snapshot.active_protocol_frames();
        let history = crate::protocol_frames::history_items_from_frames(&protocol_frames);
        crate::protocol_frames::analyze_history_items(&history, None)?;
        // Scope transitions adopt the provider wholesale (history included).
        // No multi-authority payload equality checks: provider is the new live
        // history for this path only.

        Self::validate_evidence_ids(&snapshot.evidence)?;
        let restored_turn_id = snapshot.current_turn_id.unwrap_or_default();

        self.turn = TurnRuntimeState::default();
        self.runtime_snapshot = snapshot;
        self.protocol_append_state = crate::protocol_frames::ProtocolAppendState::from_frames(
            &self.runtime_snapshot.active_protocol_frames(),
            None,
        )?;
        self.next_turn_id = self.next_turn_id.max(restored_turn_id);
        self.clear_resume_proc_local();
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn history_items(&self) -> Vec<HistoryItem> {
        self.active_history_items()
    }

    /// Test-only seed that installs a history as the live snapshot authority.
    #[cfg(test)]
    pub(crate) fn set_history_for_test(&mut self, history: Vec<HistoryItem>) {
        let frames = crate::protocol_frames::history_items_to_frames(&history);
        self.runtime_snapshot = self
            .rebuilt_runtime_snapshot_from_protocol_frames(&frames, 0, &[])
            .expect("test-seeded history is protocol compatible");
        self.protocol_append_state = crate::protocol_frames::ProtocolAppendState::from_frames(
            &self.runtime_snapshot.active_protocol_frames(),
            self.turn.current_turn_start_index,
        )
        .expect("test-seeded protocol state is valid");
    }

    pub(super) fn append_history_item(&mut self, item: HistoryItem) -> Result<()> {
        // RuntimeSnapshot is the single source of truth. Append via a derived
        // ProtocolFrame and rebuild the snapshot, preserving frame identity.
        let mut frame = crate::protocol_frames::ProtocolFrame::derived(
            protocol_frame_item_from_history_item(&item),
        );
        frame.source_provenance = Some(protocol_item_default_provenance(
            &frame.item,
            next_protocol_source_sequence(self),
        ));
        self.append_protocol_frame_to_snapshot(frame, item)
    }

    #[cfg(test)]
    pub(super) fn replace_history(&mut self, history: Vec<HistoryItem>) -> Result<()> {
        crate::protocol_frames::analyze_history_items(
            &history,
            self.turn.current_turn_start_index,
        )?;
        let frames = crate::protocol_frames::history_items_to_frames(&history);
        self.runtime_snapshot =
            self.rebuilt_runtime_snapshot_from_protocol_frames(&frames, 0, &[])?;
        self.protocol_append_state = crate::protocol_frames::ProtocolAppendState::from_frames(
            &self.runtime_snapshot.active_protocol_frames(),
            self.turn.current_turn_start_index,
        )?;
        self.clear_resume_proc_local();
        Ok(())
    }

    /// Snapshot is authoritative; just validate the active stream and heal.
    #[cfg(test)]
    pub(super) fn adopt_snapshot_as_history_seed(&mut self) -> Result<()> {
        crate::protocol_frames::analyze_history_items(
            &self.active_history_items(),
            self.turn.current_turn_start_index,
        )?;
        self.runtime_snapshot.heal_references()?;
        self.clear_provider_usage_anchor();
        Ok(())
    }

    /// Append one protocol frame to the live runtime snapshot. Restore,
    /// replacement, and compaction continue to use the cold rebuild below.
    fn append_protocol_frame_to_snapshot(
        &mut self,
        mut frame: crate::protocol_frames::ProtocolFrame,
        item: HistoryItem,
    ) -> Result<()> {
        let active_protocol_frame_count = self
            .runtime_snapshot
            .frames
            .iter()
            .filter(|runtime_frame| {
                runtime_frame.visibility == FrameVisibility::Active
                    && runtime_frame.protocol.is_some()
            })
            .count();
        let active_protocol_last_id = self
            .runtime_snapshot
            .frames
            .iter()
            .rev()
            .find(|runtime_frame| {
                runtime_frame.visibility == FrameVisibility::Active
                    && runtime_frame.protocol.is_some()
            })
            .map(|runtime_frame| runtime_frame.id);
        if !self.protocol_append_state.is_initialized()
            || self.protocol_append_state.frame_count() != active_protocol_frame_count
            || self.protocol_append_state.last_frame_id() != active_protocol_last_id
        {
            ensure_active_protocol_source_spans(&mut self.runtime_snapshot);
            let active = self.runtime_snapshot.active_protocol_frames();
            self.protocol_append_state = crate::protocol_frames::ProtocolAppendState::from_frames(
                &active,
                self.turn.current_turn_start_index,
            )?;
        }

        let next_item = protocol_frame_item_from_history_item(&item);
        if self.protocol_append_state.has_incomplete_tool_call_groups()
            && !matches!(
                &next_item,
                crate::protocol_frames::ProtocolFrameItem::ToolOutput { call_id, .. }
                    if self
                        .protocol_append_state
                        .incomplete_tool_call_ids()
                        .contains(call_id)
            )
        {
            bail!(
                "cannot append {:?} while assistant tool call group is incomplete",
                next_item
            );
        }

        let history_index = self.protocol_append_state.frame_count();
        frame.item = next_item;
        frame.history_index = history_index;
        let runtime_frame = runtime_frame_from_protocol_frame(&frame, history_index as u32);
        let frame_id = runtime_frame.id;
        self.protocol_append_state.append(
            history_index,
            frame_id,
            &frame.item,
            self.turn.current_turn_start_index,
        )?;
        self.runtime_snapshot.push_frame(runtime_frame);
        if let Some(span) = self
            .runtime_snapshot
            .frames
            .last()
            .and_then(|frame| frame.provenance.source_span)
        {
            self.runtime_snapshot.leaf_sequence = Some(
                self.runtime_snapshot
                    .leaf_sequence
                    .unwrap_or(0)
                    .max(span.end_sequence),
            );
        }
        let mut turn_protected_frame_ids =
            self.protocol_append_state.protected_frame_ids().to_vec();
        turn_protected_frame_ids.sort();
        turn_protected_frame_ids.dedup();
        self.runtime_snapshot
            .set_turn_protected_frame_ids(turn_protected_frame_ids);
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
                    if existing.provenance.source_span.is_none()
                        && let Some(provenance) = frame.source_provenance.clone()
                    {
                        existing.provenance = provenance;
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
        Ok(snapshot)
    }

    fn tool_execution_context_for(
        &self,
        _tool_name: &str,
        allow_outside_workspace: bool,
    ) -> Result<ToolExecutionContext> {
        let mut context = if allow_outside_workspace {
            ToolExecutionContext::outside_workspace_granted()
        } else {
            ToolExecutionContext::default()
        };
        context.question_handler = self.question_handler.clone();
        Ok(context)
    }

    pub(super) fn helper_max_iterations(&self) -> Option<usize> {
        Some(self.retry_config.max_recovery_attempts.saturating_add(1))
    }

    pub fn session_title_agent(&self) -> Agent {
        Agent {
            model: self.model.clone(),
            primary_route: None,
            subagent_model_overrides: HashMap::new(),
            default_protocol: self.default_protocol,
            model_protocols: self.model_protocols.clone(),
            model_catalog: self.model_catalog.clone(),
            session_reasoning_efforts: self.session_reasoning_efforts.clone(),
            prelude: vec![PromptMessage::system(SESSION_TITLE_PRELUDE)],
            runtime_snapshot: Self::fresh_runtime_snapshot(&self.model),
            protocol_append_state: crate::protocol_frames::ProtocolAppendState::empty(),
            tools: ToolRegistry::new(),
            skill_registry: None,
            skill_cards: Vec::new(),
            subagent_delegate: None,
            subagent_child_factory: None,
            primary_route_factory: None,
            question_handler: None,
            auto_review_service: None,
            permission_session: Arc::new(Mutex::new(PermissionSessionState::default())),
            subagent_path_scope: None,
            compaction_config: CompactionConfig::default(),
            retry_config: self.retry_config.clone(),
            tool_timeout_secs: self.tool_timeout_secs,
            turn: TurnRuntimeState::default(),
            next_turn_id: 0,
            // One normal helper iteration plus semantic recovery retries.
            max_iterations: self.helper_max_iterations(),
            max_tool_calls: Some(0),
            context_scope_state: Arc::new(std::sync::Mutex::new(ContextScopeState::default())),
            runtime_snapshot_provider: None,
            turn_continuation_provider: None,
            logical_request_observations: LogicalRequestObservationTracker::default(),
            active_epoch: None,
            provider_usage_anchor: None,
            request_projection_generation: 0,
            pressure_compaction_suppressed: false,
            fast_mode: None,
            // Summary agents never run the anchored bootstrap.
            anchored: None,
            anchored_override: true,
            anchored_request_phase: None,
            fake_client: None,
            fake_installation_id: self.fake_installation_id.clone(),
            fake_identity: None,
            resolved_model_route: self.resolved_model_route.clone(),
            resolved_runtime_catalog: self.resolved_runtime_catalog.clone(),
            retained_route_preparations: self.retained_route_preparations.clone(),
        }
    }

    pub(crate) async fn run_resolved_text_oneshot(&self, user_input: &str) -> Result<String> {
        let route = self
            .resolved_model_route()
            .ok_or_else(|| anyhow!("helper requires an installed resolved model route"))?;
        protocol_stream::execute_resolved_text_oneshot(
            route,
            self.active_model_metadata(),
            &self.prelude,
            user_input,
        )
        .await
    }

    pub async fn generate_session_title(&mut self, user_input: &str) -> Result<String> {
        let route = self
            .resolved_model_route()
            .ok_or_else(|| anyhow!("helper requires an installed resolved model route"))?;
        let raw = protocol_stream::execute_resolved_text_oneshot(
            route,
            self.active_model_metadata(),
            &self.prelude,
            user_input,
        )
        .await
        .map_err(|error| anyhow!("session title generation failed: {error:#}"))?;
        normalize_session_title(&raw)
    }

    #[allow(dead_code)]
    pub async fn run(&mut self, user_input: &str) -> Result<String> {
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
        F: FnMut(&str) -> Dfut + Send,
        E: FnMut(AgentEvent) -> Efut + Send,
        A: FnMut(PermissionRequest) -> Afut + Send,
        Dfut: Future<Output = Result<()>> + Send,
        Efut: Future<Output = Result<()>> + Send,
        Afut: Future<Output = Result<PermissionApproval>> + Send,
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

    #[cfg(test)]
    pub async fn run_stream_content_async<F, E, A, Dfut, Efut, Afut>(
        &mut self,
        user_content: UserMessageContent,
        on_delta: F,
        on_event: E,
        approve: A,
    ) -> Result<String>
    where
        F: FnMut(&str) -> Dfut + Send,
        E: FnMut(AgentEvent) -> Efut + Send,
        A: FnMut(PermissionRequest) -> Afut + Send,
        Dfut: Future<Output = Result<()>> + Send,
        Efut: Future<Output = Result<()>> + Send,
        Afut: Future<Output = Result<PermissionApproval>> + Send,
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
        F: FnMut(&str) -> Dfut + Send,
        E: FnMut(AgentEvent) -> Efut + Send,
        A: FnMut(PermissionRequest) -> Afut + Send,
        Q: FnMut(QuestionRequest) -> Qfut + Send + 'static,
        Dfut: Future<Output = Result<()>> + Send,
        Efut: Future<Output = Result<()>> + Send,
        Afut: Future<Output = Result<PermissionApproval>> + Send,
        Qfut: Future<Output = Result<QuestionResponse>> + Send + 'static,
    {
        let mut question_handler_guard =
            QuestionHandlerGuard::install(self, Some(Self::wrap_question_handler(ask_question)));

        let user_input = user_content.text.clone();

        if question_handler_guard
            .agent()
            .resolved_model_route()
            .is_some()
        {
            return protocol_stream::run_resolved_turn_async(
                question_handler_guard.agent(),
                user_content,
                &user_input,
                on_delta,
                on_event,
                approve,
            )
            .await;
        }

        Err(anyhow!(
            "normal agent turns require an installed resolved model route"
        ))
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

    #[cfg(test)]
    async fn execute_subagent_tool(&self, tool_name: &str, args: &Value) -> ToolResult {
        self.execute_subagent_tool_for_call(tool_name, args, None)
            .await
    }

    #[cfg(test)]
    pub(crate) async fn execute_subagent_control_tool_for_test(
        &self,
        tool_name: &str,
        args: &Value,
    ) -> ToolResult {
        self.execute_subagent_tool_for_call(tool_name, args, None)
            .await
    }

    #[cfg(test)]
    pub(crate) async fn execute_subagent_tool_with_parent_call_for_test(
        &self,
        tool_name: &str,
        args: &Value,
        parent_tool_call_id: String,
    ) -> ToolResult {
        self.execute_subagent_tool_for_call(tool_name, args, Some(parent_tool_call_id))
            .await
    }

    async fn execute_subagent_tool_for_call(
        &self,
        tool_name: &str,
        args: &Value,
        parent_tool_call_id: Option<String>,
    ) -> ToolResult {
        let Some(delegate) = self.subagent_delegate.clone() else {
            return ToolResult::err(
                tool_name,
                format!("{tool_name} is unavailable outside a subagent-capable runtime"),
            );
        };

        if is_subagent_control_tool_name(tool_name) {
            return match delegate.control(tool_name, args).await {
                Ok(result) => result,
                Err(error) => ToolResult::err(tool_name, error.to_string()),
            };
        }

        let input = match normalize_subagent_input(tool_name, args) {
            Ok(input) => input,
            Err(error) => return ToolResult::err(tool_name, error.to_string()),
        };

        let task = self.render_subagent_prompt(tool_name, &input);
        let model = match input.model.as_deref().map(ModelRoute::parse).transpose() {
            Ok(model) => model,
            Err(error) => return ToolResult::err(tool_name, error.to_string()),
        };
        let invocation = SubagentInvocation {
            input,
            model,
            prompt: task,
            parent_tool_call_id,
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
        self.tool_definitions_for(&self.model)
    }

    /// Build the catalog for a specific model id. The primary request path
    /// (tool_definitions) uses the phase bound once per turn by the prelude
    /// hook, so the catalog stays stable across iterations of one request;
    /// estimation paths recompute the phase for the candidate model.
    fn tool_definitions_for(&self, model_id: &str) -> Vec<crate::request_builder::ToolSpec> {
        let mut specs = self.tools.specs();
        // ToolRegistry retains a pair of subagent handlers for validation and
        // scope compatibility. Catalog tools are advertised only when their delegate
        // is executable, exactly as they are checked at execution time.
        specs.retain(|spec| !is_subagent_tool_name(&spec.name));
        specs.extend(
            subagent_tool_specs()
                .into_iter()
                .filter(|spec| is_executable_tool(self, &spec.name)),
        );
        if self.subagent_delegate.is_some() {
            specs.extend(subagent_control_tool_specs());
        }
        if let Some(anchored) = &self.anchored
            && anchored.enabled_for(model_id)
            && self.anchored_override
        {
            let phase = if model_id == self.model {
                self.anchored_request_phase
                    .unwrap_or_else(|| anchored.phase(&self.active_history_items()))
            } else {
                anchored.phase(&self.active_history_items())
            };
            specs = anchored.tool_catalog(&phase, specs);
        }
        specs
    }

    #[allow(dead_code)]
    fn append_assistant_tool_calls(
        &mut self,
        turn_text: &str,
        tool_calls: &[HistoryToolCall],
    ) -> Result<()> {
        self.append_assistant_tool_calls_with_reasoning_content(turn_text, None, None, tool_calls)
    }

    fn append_assistant_tool_calls_with_reasoning_content(
        &mut self,
        turn_text: &str,
        reasoning_content: Option<&str>,
        replay: Option<&str>,
        tool_calls: &[HistoryToolCall],
    ) -> Result<()> {
        self.append_history_item(HistoryItem::AssistantTurn {
            text: (!turn_text.is_empty()).then(|| turn_text.to_string()),
            reasoning_content: reasoning_content.map(ToString::to_string),
            replay: replay.and_then(
                crate::model_runtime::OpaqueReplayState::from_anthropic_thinking_blocks_json,
            ),
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
        match self.record_tool_call_result(call, record, on_event).await? {
            ToolCallRecordOutcome::Completed => Ok(()),
            ToolCallRecordOutcome::Cancelled => Err(anyhow!("{} cancelled", call.name)),
        }
    }

    fn parallel_tool_call_is_ready(&self, call: &HistoryToolCall) -> bool {
        if is_subagent_tool_name(&call.name)
            || self.tools.parallelism(&call.name) != ToolParallelism::Parallel
            || !is_executable_tool(self, &call.name)
            || !self.tools.scope().allows_tool(&call.name)
            // ponytail: scoped children fall back to sequential execution so the
            // single denial path in execute_with_arguments owns DelegationScopeDenied.
            || (self.subagent_path_scope.is_some() && is_delegation_path_scoped_tool(&call.name))
        {
            return false;
        }
        let Ok(args) = serde_json::from_str::<Value>(&call.arguments_json) else {
            return false;
        };
        let permission_class = permission_class_for_tool_call(&self.tools, &call.name);
        if (self.permission_mode() != PermissionMode::Auto
            && restricted_by_directive_with_class(
                &call.name,
                &args,
                permission_class,
                self.turn.policy.directive,
            )
            .is_some())
            || external_workspace_access_for_tool(&call.name, &args).is_some()
        {
            return false;
        }
        let Ok(state) = self.permission_session.lock() else {
            return false;
        };
        let (mode, _, decision, grant_allowed) = state.approval_snapshot(
            crate::tool::permission_resource_for_tool(&call.name, &args).as_ref(),
            &call.name,
            &args,
            permission_class,
            self.turn.policy.directive,
            false,
            crate::permission::is_internal_tool(&call.name),
        );
        let decision = if mode.supports_session_grants() && grant_allowed {
            PermissionDecision::Allow
        } else {
            decision
        };
        decision == PermissionDecision::Allow
    }

    /// Executes model-issued calls in order. Contiguous ordinary tools that
    /// explicitly support parallel execution are polled together after a
    /// no-prompt permission preflight; results are returned in model order.
    /// Subagents retain separate batching rules when model parallel calls are enabled.
    async fn execute_tool_calls_and_record<E, A, Efut, Afut>(
        &mut self,
        calls: &[HistoryToolCall],
        on_event: &mut E,
        approve: &mut A,
    ) -> Result<()>
    where
        E: FnMut(AgentEvent) -> Efut,
        A: FnMut(PermissionRequest) -> Afut,
        Efut: Future<Output = Result<()>>,
        Afut: Future<Output = Result<PermissionApproval>>,
    {
        let mut index = 0;
        while index < calls.len() {
            if !is_subagent_tool_name(&calls[index].name) {
                if self.active_model_metadata().parallel_tool_calls
                    && self.parallel_tool_call_is_ready(&calls[index])
                {
                    let mut end = index + 1;
                    while calls
                        .get(end)
                        .is_some_and(|call| self.parallel_tool_call_is_ready(call))
                    {
                        end += 1;
                    }
                    if end - index > 1 {
                        let records = tool_execution::execute_parallel_tool_call_batch(
                            self,
                            &calls[index..end],
                            on_event,
                        )
                        .await?;
                        let mut records = records.into_iter().enumerate();
                        while let Some((offset, batch_record)) = records.next() {
                            let call = &calls[index + offset];
                            let result = async {
                                tool_execution::finalize_parallel_tool_call(
                                    self,
                                    &batch_record.record,
                                    on_event,
                                )
                                .await?;
                                self.record_tool_call_result(
                                    call,
                                    batch_record.record.clone(),
                                    on_event,
                                )
                                .await
                            }
                            .instrument(batch_record.span())
                            .await;
                            match result {
                                Ok(ToolCallRecordOutcome::Completed) => {
                                    batch_record.finish(&Ok(()))
                                }
                                Ok(ToolCallRecordOutcome::Cancelled) => {
                                    batch_record.finish(&Ok(()));
                                    let remaining = records
                                        .map(|(remaining_offset, remaining)| {
                                            (&calls[index + remaining_offset], remaining)
                                        })
                                        .collect::<Vec<_>>();
                                    tool_execution::cancel_parallel_calls_best_effort(
                                        remaining
                                            .iter()
                                            .map(|(call, _)| {
                                                (call.call_id.clone(), call.name.clone())
                                            })
                                            .collect(),
                                        on_event,
                                    )
                                    .await;
                                    for (_, remaining) in remaining {
                                        remaining.finish(&Err(anyhow!(
                                            "parallel tool batch reconciliation aborted"
                                        )));
                                    }
                                    return Err(anyhow!("{} cancelled", call.name));
                                }
                                Err(error) => {
                                    batch_record.finish(&Err(anyhow!(
                                        "parallel tool batch reconciliation failed"
                                    )));
                                    let remaining = records
                                        .map(|(remaining_offset, remaining)| {
                                            (&calls[index + remaining_offset], remaining)
                                        })
                                        .collect::<Vec<_>>();
                                    tool_execution::cancel_parallel_calls_best_effort(
                                        remaining
                                            .iter()
                                            .map(|(call, _)| {
                                                (call.call_id.clone(), call.name.clone())
                                            })
                                            .collect(),
                                        on_event,
                                    )
                                    .await;
                                    for (_, remaining) in remaining {
                                        remaining.finish(&Err(anyhow!(
                                            "parallel tool batch reconciliation aborted"
                                        )));
                                    }
                                    return Err(error);
                                }
                            }
                        }
                        index = end;
                        continue;
                    }
                }
                self.execute_tool_call_and_record(&calls[index], on_event, approve)
                    .await?;
                index += 1;
                continue;
            }

            let mut end = index + 1;
            if self.active_model_metadata().parallel_tool_calls {
                while calls
                    .get(end)
                    .is_some_and(|call| is_subagent_tool_name(&call.name))
                {
                    end += 1;
                }
            }
            let records = tool_execution::execute_subagent_tool_call_batch(
                self,
                &calls[index..end],
                on_event,
                approve,
            )
            .await?;
            let mut cancellation = None;
            let mut records = records.into_iter().enumerate();
            while let Some((offset, batch_record)) = records.next() {
                let call = &calls[index + offset];
                let result = async {
                    tool_execution::finalize_subagent_tool_call(
                        self,
                        call,
                        &batch_record.record,
                        on_event,
                    )
                    .await?;
                    self.record_tool_call_result(call, batch_record.record.clone(), on_event)
                        .await
                }
                .instrument(batch_record.span())
                .await;
                match result {
                    Ok(ToolCallRecordOutcome::Completed) => batch_record.finish(&Ok(())),
                    Ok(ToolCallRecordOutcome::Cancelled) => {
                        batch_record.finish(&Ok(()));
                        if cancellation.is_none() {
                            cancellation = Some(anyhow!("{} cancelled", call.name));
                        }
                    }
                    Err(error) => {
                        batch_record.finish(&Err(anyhow!("subagent batch reconciliation failed")));
                        for (_, remaining) in records {
                            remaining
                                .finish(&Err(anyhow!("subagent batch reconciliation aborted")));
                        }
                        return Err(error);
                    }
                }
            }
            if let Some(error) = cancellation {
                return Err(error);
            }
            index = end;
        }
        Ok(())
    }

    async fn record_tool_call_result<E, Efut>(
        &mut self,
        call: &HistoryToolCall,
        record: ToolExecutionRecord,
        on_event: &mut E,
    ) -> Result<ToolCallRecordOutcome>
    where
        E: FnMut(AgentEvent) -> Efut,
        Efut: Future<Output = Result<()>>,
    {
        debug!(
            tool_name = %call.name,
            call_id = %call.call_id,
            output = ?record.output,
            effects = ?record.effects,
            "tool call completed"
        );

        let output_json = serde_json::to_string(&record.output.for_text_history())?;
        self.append_history_item(HistoryItem::ToolOutput {
            call_id: call.call_id.clone(),
            output_json,
            images: record.output.images.clone(),
        })?;
        self.reconcile_loaded_skill_material()?;
        if let Some(usage) = self.projected_token_usage() {
            on_event(AgentEvent::TokenUsageUpdated {
                used_tokens: usage.used_tokens,
                context_window_tokens: usage.context_window_tokens,
                input_tokens: usage.input_tokens,
                // This is a projection refresh, not a new provider response.
                // The TUI already accumulates the response's output tokens.
                output_tokens: 0,
                cached_tokens: usage.cached_tokens,
                cache_report: None,
            })
            .await?;
        }

        debug!(
            history_len = self.active_history_items().len(),
            "tool output appended to history"
        );

        if !self.subagent_effects_are_already_recorded(&record) {
            let evidence = self.remember_tool_evidence(&record)?;
            on_event(AgentEvent::EvidenceRecorded(evidence)).await?;
        }

        if is_cancelled_subagent_record(&record) {
            return Ok(ToolCallRecordOutcome::Cancelled);
        }

        Ok(ToolCallRecordOutcome::Completed)
    }

    fn subagent_effects_are_already_recorded(&self, record: &ToolExecutionRecord) -> bool {
        if !is_subagent_tool_name(&record.tool_name)
            && record.tool_name != tool_names::TOOL_AGENT_WAIT
        {
            return false;
        }
        let Some(run_id) = record
            .output
            .data
            .as_ref()
            .and_then(|data| data.get("run_id"))
            .and_then(Value::as_str)
        else {
            return false;
        };
        self.runtime_snapshot.evidence.iter().any(|evidence| {
            matches!(
                &evidence.source,
                crate::evidence::EvidenceSource::Subagent {
                    run_id: recorded,
                    ..
                } if recorded == run_id
            )
        })
    }

    fn remember_tool_evidence(&mut self, record: &ToolExecutionRecord) -> Result<EvidenceRecord> {
        evidence_memory::remember_tool_evidence(self, record)
    }

    pub async fn run_stream<F, E, A>(
        &mut self,
        user_input: &str,
        mut on_delta: F,
        mut on_event: E,
        mut approve: A,
    ) -> Result<String>
    where
        F: FnMut(&str) -> Result<()> + Send,
        E: FnMut(AgentEvent) -> Result<()> + Send,
        A: FnMut(PermissionRequest) -> Result<PermissionApproval> + Send,
    {
        self.run_stream_async(
            user_input,
            |delta| std::future::ready(on_delta(delta)),
            |event| std::future::ready(on_event(event)),
            |request| std::future::ready(approve(request)),
        )
        .await
    }

    #[cfg(test)]
    pub async fn compact_session_async<E, Efut>(
        &mut self,
        on_event: E,
    ) -> Result<ManualCompactionOutcome>
    where
        E: FnMut(AgentEvent) -> Efut + Send,
        Efut: Future<Output = Result<()>> + Send,
    {
        compaction::compact_session_stream_async(self, on_event, || Ok(())).await
    }

    pub async fn compact_session_stream_async<E, Efut, S>(
        &mut self,
        on_event: E,
        on_start: S,
    ) -> Result<ManualCompactionOutcome>
    where
        E: FnMut(AgentEvent) -> Efut + Send,
        Efut: Future<Output = Result<()>> + Send,
        S: FnMut() -> Result<()> + Send,
    {
        compaction::compact_session_stream_async(self, on_event, on_start).await
    }

    #[cfg(test)]
    fn prepare_turn_prelude(&mut self, user_input: &str) -> Vec<PromptMessage> {
        self.try_prepare_turn_prelude(user_input)
            .expect("test/internal turn prelude should resolve selected skills")
    }

    #[cfg(test)]
    fn try_prepare_turn_prelude(&mut self, user_input: &str) -> Result<Vec<PromptMessage>> {
        self.try_prepare_turn_prelude_with_skills(user_input, &[])
    }

    fn try_prepare_turn_prelude_with_skills(
        &mut self,
        user_input: &str,
        selected_skills: &[String],
    ) -> Result<Vec<PromptMessage>> {
        let manual_skill_material = self.manual_skill_material_messages(selected_skills)?;
        self.invalidate_request_projection();
        let turn = WorkflowTurnState::from_user_input(user_input);
        self.next_turn_id = self.next_turn_id.saturating_add(1);
        self.turn = TurnRuntimeState::new(self.next_turn_id, turn.clone());
        self.runtime_snapshot.workflow.auto_continue = AutoContinueState::default();
        if self.pressure_compaction_suppressed {
            self.turn.pressure_compaction.suppress();
        }
        self.runtime_snapshot.current_turn_id = Some(self.next_turn_id);

        // Anchored bootstrap hook: runs BEFORE the current user message is
        // appended to history (protocol_stream calls this first), so the first
        // request sees the Bootstrap phase with an empty history. The phase is
        // bound once per turn here; the tool catalog and alias resolution read
        // the same value so one request never mixes phases.
        if let Some(anchored) = &self.anchored
            && anchored.enabled_for(&self.model)
            && self.anchored_override
        {
            let phase = anchored.phase(&self.active_history_items());
            self.anchored_request_phase = Some(phase);
            return Ok(anchored.prelude(
                &phase,
                &self.prelude,
                Some(runtime_context_message()),
                self.skill_prelude_message(),
                turn.developer_context_message(),
                &manual_skill_material,
            ));
        }

        let mut turn_prelude = self.prelude.clone();
        turn_prelude.push(runtime_context_message());
        if let Some(message) = self.skill_prelude_message() {
            turn_prelude.push(message);
        }
        turn_prelude.extend(manual_skill_material);
        if let Some(message) = turn.developer_context_message() {
            turn_prelude.push(message);
        }
        Ok(turn_prelude)
    }

    fn manual_skill_material_messages(
        &self,
        selected_skills: &[String],
    ) -> Result<Vec<PromptMessage>> {
        if selected_skills.is_empty() {
            return Ok(Vec::new());
        }
        let registry = self
            .skill_registry
            .as_ref()
            .ok_or_else(|| anyhow!("unknown selected skill: {}", selected_skills[0]))?;
        let mut seen = HashSet::new();
        let selected_skills = selected_skills
            .iter()
            .filter(|name| seen.insert(name.as_str()))
            .cloned()
            .collect::<Vec<_>>();
        registry
            .selected_entries(&selected_skills)?
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
            "可用的本地 skills：\n需要时用 `skill` 工具加载相关 skills。不要投机性加载 skills。Skills 不会改变权限或扩大工具范围。",
        );
        for card in &self.skill_cards {
            text.push_str(&format!(
                "\n- {} — {}（来源：{}）",
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
                self.runtime_snapshot.workflow.todos = payload.items;
                self.invalidate_request_projection();
            }
            "workflow__auto_continue" => {
                let payload: WorkflowAutoContinuePayload = serde_json::from_value(args.clone())?;
                let next_state = AutoContinueState {
                    enabled: payload.enabled,
                };
                on_event(AgentEvent::AutoContinueChanged {
                    state: next_state.clone(),
                })
                .await?;
                self.runtime_snapshot.workflow.auto_continue = next_state;
                self.invalidate_request_projection();
                if payload.enabled {
                    self.turn.auto_continue_active = true;
                }
            }
            _ => {}
        }

        Ok(())
    }

    fn finalize_turn_decision(&self) -> FinalizeDecision {
        if self.runtime_snapshot.workflow.auto_continue.enabled {
            FinalizeDecision::Continue
        } else {
            FinalizeDecision::Finish
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
        match self.finalize_turn_decision() {
            FinalizeDecision::Finish => Ok(false),
            FinalizeDecision::Continue => {
                *continuation_count += 1;
                self.turn.counters.continuations = *continuation_count;
                let text = "Continue the current task internally. Do not repeat finished work. Use the todo list as working context when present; decide yourself when to disable auto-continue.".to_string();
                self.append_history_item(HistoryItem::internal_continuation(text.clone()))?;
                on_event(AgentEvent::InternalContinuation {
                    text,
                    source: crate::transcript::InternalContinuationSource::AutoContinue,
                })
                .await?;
                on_event(AgentEvent::AutoContinuationScheduled {
                    continuation_count: *continuation_count,
                    remaining_unfinished: self.remaining_unfinished_todos().unwrap_or(0),
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
        self.turn.auto_continue_active = false;
        self.turn.pressure_compaction.reset_for_turn_end();
        self.turn.current_turn_start_index = None;
        self.runtime_snapshot.current_turn_id = None;
        self.runtime_snapshot = self.rebuilt_runtime_snapshot_from_protocol_frames(
            &self.active_protocol_frames(),
            self.active_protocol_frames().len(),
            &self.active_history_items(),
        )?;
        self.protocol_append_state = crate::protocol_frames::ProtocolAppendState::from_frames(
            &self.runtime_snapshot.active_protocol_frames(),
            None,
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
                "本回合产生了写变更（含委派的子工作），且验证已运行但失败。在依赖这些变更前请先审查失败的验证输出；至少有一项验证失败。"
            } else {
                "本回合产生了写变更（含委派的子工作），但未运行验证。如有需要，请审查并运行最相关的检查。"
            };

            ValidationAdvisory {
                write_effects,
                validation_effects,
                failed_validation_effects,
                message: message.into(),
            }
        })
    }

    fn remaining_unfinished_todos(&self) -> Option<usize> {
        if self
            .runtime_snapshot
            .workflow
            .todos
            .iter()
            .any(|todo| todo.status == TodoStatus::Blocked)
        {
            return None;
        }

        let unfinished = self
            .runtime_snapshot
            .workflow
            .todos
            .iter()
            .filter(|todo| todo.status.is_unfinished())
            .count();
        (unfinished > 0).then_some(unfinished)
    }

    fn record_tool_effects(&mut self, record: &ToolExecutionRecord) {
        if is_subagent_tool_name(&record.tool_name)
            || record.tool_name == tool_names::TOOL_AGENT_WAIT
        {
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

    pub(crate) fn append_internal_continuation(&mut self, text: String) -> Result<()> {
        self.append_history_item(HistoryItem::internal_continuation(text))
    }

    pub(crate) fn begin_internal_continuation_turn(&mut self, text: &str) -> Result<()> {
        let _ = self.try_prepare_turn_prelude_with_skills(text, &[])?;
        Ok(())
    }

    pub(crate) fn install_background_subagent_result(
        &mut self,
        result: &crate::subagent::SubagentRunSummary,
    ) -> Result<()> {
        self.record_background_subagent_effects(result);
        Ok(())
    }

    fn record_background_subagent_effects(&mut self, result: &crate::subagent::SubagentRunSummary) {
        if self.runtime_snapshot.evidence.iter().any(|evidence| {
            matches!(
                &evidence.source,
                crate::evidence::EvidenceSource::Subagent {
                    run_id: recorded,
                    ..
                } if recorded == &result.run_id
            )
        }) {
            return;
        }
        let parent_tool = subagent_tool_name_for_agent_name(&result.agent_name)
            .map(ToString::to_string)
            .unwrap_or_else(|| format!("agent__{}", result.agent_name));
        let record = ToolExecutionRecord {
            call_id: format!("background-{}", result.run_id),
            tool_name: parent_tool.clone(),
            arguments: Some(serde_json::json!({ "background": true })),
            permission_class: crate::permission::ToolPermissionClass::Preview,
            directive: self.turn.policy.directive,
            status: ToolExecutionStatus::Executed,
            rejection: None,
            output: ToolResult::ok(
                parent_tool,
                serde_json::json!({
                    "run_id": result.run_id,
                    "child_session_id": result.child_session_id,
                    "agent_name": result.agent_name,
                    "status": result.status.as_str(),
                    "summary": result.summary,
                    "structured_result": result.structured_result,
                    "active": false,
                    "background": true,
                }),
            ),
            effects: ToolEffects {
                kind: ToolEffectKind::Read,
                primary_path: None,
                edited_paths: result.structured_result.files_changed.clone(),
                command: None,
            },
        };
        self.record_subagent_effects(&record);
    }

    #[cfg(test)]
    fn child_effect_counts_for_test(&self) -> (usize, usize, usize) {
        (
            self.turn.counters.child_write_effects,
            self.turn.counters.child_validation_effects,
            self.turn.counters.child_failed_validation_effects,
        )
    }

    fn record_subagent_effects(&mut self, record: &ToolExecutionRecord) {
        if self.subagent_effects_are_already_recorded(record) {
            return;
        }
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

pub(crate) struct TurnContinuationProviderGuard<'a> {
    agent: &'a mut Agent,
    previous: Option<TurnContinuationProvider>,
}

impl<'a> TurnContinuationProviderGuard<'a> {
    fn install(agent: &'a mut Agent, replacement: Option<TurnContinuationProvider>) -> Self {
        let previous = agent.turn_continuation_provider.take();
        agent.turn_continuation_provider = replacement;
        Self { agent, previous }
    }

    pub(crate) fn agent(&mut self) -> &mut Agent {
        self.agent
    }
}

impl Drop for TurnContinuationProviderGuard<'_> {
    fn drop(&mut self) {
        self.agent.turn_continuation_provider = self.previous.take();
    }
}

struct QuestionHandlerGuard<'a> {
    agent: &'a mut Agent,
    previous: Option<QuestionCallback>,
}

impl<'a> QuestionHandlerGuard<'a> {
    fn install(agent: &'a mut Agent, replacement: Option<QuestionCallback>) -> Self {
        let previous = agent.question_handler.take();
        agent.question_handler = replacement;
        Self { agent, previous }
    }

    fn agent(&mut self) -> &mut Agent {
        self.agent
    }
}

impl Drop for QuestionHandlerGuard<'_> {
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
fn is_executable_tool(agent: &Agent, tool_name: &str) -> bool {
    if is_subagent_control_tool_name(tool_name) {
        return agent.subagent_delegate.is_some();
    }
    match subagent_catalog_entry_by_tool_name(tool_name) {
        Some(_) => agent.subagent_delegate.is_some(),
        None => agent.tools.contains(tool_name),
    }
}

fn subagent_tool_permission_class(
    tool_name: &str,
) -> Option<crate::permission::ToolPermissionClass> {
    if is_subagent_control_tool_name(tool_name) {
        return Some(crate::permission::ToolPermissionClass::Preview);
    }
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

pub(crate) fn is_subagent_control_tool_name(name: &str) -> bool {
    matches!(
        name,
        tool_names::TOOL_AGENT_JOBS
            | tool_names::TOOL_AGENT_STATUS
            | tool_names::TOOL_AGENT_WAIT
            | tool_names::TOOL_AGENT_CANCEL
    )
}

fn subagent_control_tool_specs() -> Vec<crate::request_builder::ToolSpec> {
    vec![
        crate::request_builder::ToolSpec {
            name: tool_names::TOOL_AGENT_JOBS.into(),
            description: "List subagent jobs for the current parent session, including active and terminal runs.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": false
            }),
            strict: true,
        },
        crate::request_builder::ToolSpec {
            name: tool_names::TOOL_AGENT_STATUS.into(),
            description: "Get one existing subagent run by run_id without starting or taking over work.".into(),
            parameters: subagent_run_id_schema(),
            strict: true,
        },
        crate::request_builder::ToolSpec {
            name: tool_names::TOOL_AGENT_WAIT.into(),
            description: "Bring one active background subagent run to the foreground and block until it reaches a terminal state.".into(),
            parameters: subagent_run_id_schema(),
            strict: true,
        },
        crate::request_builder::ToolSpec {
            name: tool_names::TOOL_AGENT_CANCEL.into(),
            description: "Request cancellation of one active subagent run by run_id.".into(),
            parameters: subagent_run_id_schema(),
            strict: true,
        },
    ]
}

fn subagent_run_id_schema() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "run_id": {"type": "string", "minLength": 1}
        },
        "required": ["run_id"],
        "additionalProperties": false
    })
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
            if is_read_only_subagent_tool_name(tool_name)
                || is_subagent_control_tool_name(tool_name)
            {
                ToolEffectKind::Read
            } else {
                match tool_name {
                    "fs__read"
                    | "fs__list"
                    | "skill"
                    | "search__rg"
                    | "web__fetch"
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
                    "workflow__todos" | "workflow__auto_continue" => {
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

fn shell_command_succeeded(output: &ToolResult) -> bool {
    if !output.ok {
        return false;
    }

    let Some(data) = output.data.as_ref() else {
        return true;
    };

    if let Some(status) = data.get("status").and_then(Value::as_i64)
        && status != 0
    {
        return false;
    }

    if let Some(success) = data.get("success").and_then(Value::as_bool)
        && !success
    {
        return false;
    }

    data.get("error").is_none()
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
        HistoryItem::AssistantTurn {
            text,
            reasoning_content,
            replay,
            calls,
        } => crate::protocol_frames::ProtocolFrameItem::AssistantTurn {
            text: text.clone(),
            reasoning_content: reasoning_content.clone(),
            replay: replay.clone(),
            calls: calls.clone(),
        },
        HistoryItem::ToolOutput {
            call_id,
            output_json,
            images,
        } => crate::protocol_frames::ProtocolFrameItem::ToolOutput {
            call_id: call_id.clone(),
            output_json: output_json.clone(),
            images: images.clone(),
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
        crate::protocol_frames::ProtocolFrameItem::AssistantTurn { text, calls, .. } => (
            if calls.is_empty() {
                RuntimeFrameKind::Assistant
            } else {
                RuntimeFrameKind::ToolCall
            },
            format!(
                "assistant:{}:{}",
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
            ..
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

fn next_protocol_source_sequence(agent: &Agent) -> u64 {
    let from_frames = agent
        .runtime_snapshot
        .frames
        .iter()
        .filter_map(|frame| frame.provenance.source_span.map(|span| span.end_sequence))
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
    if snapshot
        .leaf_sequence
        .map(|leaf| leaf < high)
        .unwrap_or(true)
    {
        snapshot.leaf_sequence = Some(high);
    }
}

// Rebuild helpers treat history as the sole protocol authority for the open
// session; RuntimeSnapshot mirrors the active session state.

/// Copy non-protocol runtime surfaces from `source` onto `target` without
/// touching active protocol payloads. Used when rebuilding protocol shells.
fn merge_non_protocol_runtime_metadata(target: &mut RuntimeSnapshot, source: &RuntimeSnapshot) {
    target.child_sessions = source.child_sessions.clone();
    target.prompt_contributors = source.prompt_contributors.clone();
    target.evidence = source.evidence.clone();
    target.context_view = source.context_view.clone();
    target.context_tree = source.context_tree.clone();
    target.active_context = source.active_context.clone();
    target.workflow = source.workflow.clone();
    target.compaction.compacted_frame_ids = source.compaction.compacted_frame_ids.clone();
    target.compaction.retired_source_spans = source.compaction.retired_source_spans.clone();
    // Keep any non-protocol frames (no protocol payload) that the rebuild dropped.
    let target_ids = target
        .frames
        .iter()
        .map(|frame| frame.id)
        .collect::<HashSet<_>>();
    target.frames.extend(
        source
            .frames
            .iter()
            .filter(|frame| frame.protocol.is_none() && !target_ids.contains(&frame.id))
            .cloned(),
    );
    target.recompute_protected_frame_ids();
}

pub(super) fn rebind_active_protocol_from_history(
    snapshot: &mut RuntimeSnapshot,
    history: &[HistoryItem],
) -> Result<()> {
    let active_ids = snapshot
        .frames
        .iter()
        .filter(|frame| {
            frame.visibility == crate::runtime_context::FrameVisibility::Active
                && frame.protocol.is_some()
        })
        .map(|frame| frame.id)
        .collect::<Vec<_>>();
    ensure!(
        active_ids.len() == history.len(),
        "cannot rebind history onto runtime snapshot: active protocol frames {} vs history {}",
        active_ids.len(),
        history.len()
    );
    let payload_by_id = active_ids
        .into_iter()
        .zip(history.iter())
        .map(|(id, item)| (id, protocol_frame_item_from_history_item(item)))
        .collect::<HashMap<_, _>>();
    for frame in &mut snapshot.frames {
        if let Some(item) = payload_by_id.get(&frame.id) {
            if let crate::protocol_frames::ProtocolFrameItem::ToolOutput { output_json, .. } = item
            {
                frame.summary = Some(output_json.clone());
            }
            frame.protocol = Some(item.clone());
        }
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TurnRuntimeState {
    turn_id: u64,
    current_turn_start_index: Option<usize>,
    policy: WorkflowTurnState,
    counters: TurnCounters,
    // Once enabled, auto-continue owns the rest of this turn so the LLM can
    // explicitly disable it and still receive one final response. This is
    // separate from the persisted setting, which records the LLM's last choice.
    auto_continue_active: bool,
    frozen_evidence: Option<FrozenTurnEvidence>,
    pressure_compaction: PressureCompactionState,
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
            counters: TurnCounters::default(),
            auto_continue_active: false,
            frozen_evidence: None,
            pressure_compaction: PressureCompactionState::default(),
        }
    }
}

/// Ephemeral pressure-compaction state. It is absent from transcript and
/// snapshot projections, so a restored turn never inherits an attempted prompt.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct PressureCompactionState {
    last_attempted_frontier: Option<PressureCompactionFrontier>,
    /// Set after a successful pressure compact in this turn. Soft-watermark
    /// re-entry is suppressed so partial reclaim cannot spin compact→still
    /// above watermark→compact again across agent iterations; hard overflow
    /// still retries when the protocol frontier advances.
    compacted_this_turn: bool,
    suppressed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PressureCompactionFrontier {
    frame_count: usize,
    protocol_prefix_digest: String,
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

impl Default for TurnRuntimeState {
    fn default() -> Self {
        Self::new(0, WorkflowTurnState::default())
    }
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
enum FinalizeDecision {
    Finish,
    Continue,
}

#[derive(Debug, Clone, Deserialize)]
struct WorkflowTodosPayload {
    items: Vec<TodoItem>,
}

#[derive(Debug, Clone, Deserialize)]
struct WorkflowAutoContinuePayload {
    enabled: bool,
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
        let text = match self.directive {
            ExecutionDirective::None => return None,
            ExecutionDirective::ReadOnly => {
                "本回合为只读。不要修改文件，也不要运行非只读命令。".to_string()
            }
            ExecutionDirective::PlanOnly => {
                "本回合仅做规划。只产出分析与计划。不要修改文件，也不要运行非只读命令。".to_string()
            }
            ExecutionDirective::AnalyzeOnly => {
                "本回合仅做分析。只检查与解释。不要修改文件，也不要运行非只读命令。".to_string()
            }
            ExecutionDirective::DoNotEdit => {
                "本回合有明确的禁止编辑指令。不要修改文件，也不要运行非只读命令。".to_string()
            }
        };

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

fn estimate_trailing_history_item_tokens(item: &HistoryItem) -> u64 {
    if let HistoryItem::UserMessage { content } = item
        && !content.attachments.is_empty()
    {
        let compact_item = HistoryItem::user(content.prompt_plan_text());
        let json_len = serde_json::to_string(&compact_item)
            .map(|serialized| serialized.len())
            .unwrap_or(0);
        let text_tokens = (json_len as u64).div_ceil(4);
        let visual_tokens = content
            .attachments
            .iter()
            .map(crate::user_content::UserImageAttachment::visual_token_charge)
            .sum::<u64>();
        return text_tokens.saturating_add(visual_tokens);
    }

    serde_json::to_string(item)
        .map(|serialized| (serialized.len() as u64).div_ceil(4))
        .unwrap_or(0)
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
            Self::DelegationScopeDenied => "delegation_scope_denied",
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
mod incremental_append_tests {
    use super::*;

    fn evidence(id: &str) -> EvidenceRecord {
        EvidenceRecord {
            id: id.into(),
            sequence: 1,
            timestamp_ms: 0,
            evidence_kind: crate::evidence::EvidenceKind::Decision,
            title: "decision".into(),
            summary: "preserve evidence".into(),
            detail: Some("metadata".into()),
            source: EvidenceSource::Transcript { sequence: 1 },
            tags: vec!["phase1".into()],
        }
    }

    #[test]
    fn assistant_turn_conversion_preserves_reasoning_and_replay_without_tool_calls() {
        let item = HistoryItem::AssistantTurn {
            text: Some("answer".into()),
            reasoning_content: Some("reasoning".into()),
            replay: crate::model_runtime::OpaqueReplayState::from_anthropic_thinking_blocks_json(
                r#"[{"type":"thinking","signature":"signed"}]"#,
            ),
            calls: Vec::new(),
        };

        assert_eq!(protocol_frame_item_from_history_item(&item), item);
    }

    #[test]
    fn incremental_agent_append_matches_cold_rebuild_with_metadata_and_evidence() {
        let base = vec![
            HistoryItem::user("older request"),
            HistoryItem::assistant("older answer"),
        ];
        let full_history = vec![
            base[0].clone(),
            base[1].clone(),
            HistoryItem::user("current request"),
            HistoryItem::AssistantTurn {
                text: Some("working".into()),
                reasoning_content: Some("reasoning".into()),
                replay:
                    crate::model_runtime::OpaqueReplayState::from_anthropic_thinking_blocks_json(
                        r#"{"signature":"wire"}"#,
                    ),
                calls: vec![
                    HistoryToolCall {
                        call_id: "call-1".into(),
                        name: "fs__read".into(),
                        arguments_json: r#"{"path":"a"}"#.into(),
                    },
                    HistoryToolCall {
                        call_id: "call-2".into(),
                        name: "fs__read".into(),
                        arguments_json: r#"{"path":"b"}"#.into(),
                    },
                ],
            },
            HistoryItem::ToolOutput {
                call_id: "call-2".into(),
                output_json: r#"{"result":"b"}"#.into(),
                images: Vec::new(),
            },
            HistoryItem::ToolOutput {
                call_id: "call-1".into(),
                output_json: r#"{"result":"a"}"#.into(),
                images: Vec::new(),
            },
            HistoryItem::assistant("done"),
        ];

        let mut incremental = Agent::new("m1", 4, 4);
        incremental.set_history_for_test(base.clone());
        incremental.runtime_snapshot.context_scope_revision = 7;
        incremental.runtime_snapshot.current_segment_id = Some(11);
        incremental.runtime_snapshot.current_turn_id = Some(3);
        incremental.turn.current_turn_start_index = Some(base.len());
        for item in full_history.iter().skip(base.len()) {
            incremental
                .append_history_item(item.clone())
                .expect("incremental append should succeed");
        }
        incremental
            .add_evidence(evidence("ev-1"))
            .expect("evidence append should succeed");

        let mut cold = Agent::new("m1", 4, 4);
        cold.runtime_snapshot.context_scope_revision = 7;
        cold.runtime_snapshot.current_segment_id = Some(11);
        cold.runtime_snapshot.current_turn_id = Some(3);
        cold.turn.current_turn_start_index = Some(base.len());
        cold.set_history_for_test(full_history);
        cold.add_evidence(evidence("ev-1"))
            .expect("cold evidence append should succeed");

        assert_eq!(incremental.runtime_snapshot, cold.runtime_snapshot);
        assert_eq!(
            incremental.active_history_items(),
            cold.active_history_items()
        );
        assert_eq!(incremental.runtime_snapshot.leaf_sequence, Some(7));
        assert_eq!(
            incremental
                .runtime_snapshot
                .compaction
                .turn_protected_frame_ids,
            cold.runtime_snapshot.compaction.turn_protected_frame_ids
        );
        assert_eq!(incremental.evidence(), cold.evidence());
    }
}

#[cfg(test)]
mod tests;

fn default_agent_prelude() -> Vec<PromptMessage> {
    vec![PromptMessage::system(DEFAULT_AGENT_PRELUDE)]
}

fn runtime_context_message() -> PromptMessage {
    runtime_context_message_from_parts(&current_date_label(), &timezone_label())
}

fn runtime_context_message_from_parts(date: &str, timezone: &str) -> PromptMessage {
    PromptMessage::developer_with_origin(
        format!("运行时上下文：\n- 当前日期：{date}\n- 时区：{timezone}"),
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
