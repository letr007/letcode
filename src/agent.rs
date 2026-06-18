use anyhow::{Context, Result, anyhow, bail};
use async_openai::Client;
use async_openai::config::Config;
use async_openai::error::OpenAIError;
use async_openai::types::chat::{
    ChatCompletionMessageToolCall, ChatCompletionMessageToolCallChunk, FinishReason,
};
use async_openai::types::responses::{OutputItem, Response, ResponseStreamEvent};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::future::Future;
use std::pin::Pin;
use std::process::Command;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::{debug, error, info, trace, warn};

use crate::config::ApiProtocol;
use crate::evidence::{EvidenceDraft, EvidenceRecord, require_unique_evidence_id};
use crate::permission::{
    ExecutionDirective, PermissionDecision, PermissionMode, PermissionPolicy, PermissionRequest,
    ToolScope, restricted_by_directive_with_class,
};
use crate::request_builder::{
    BuiltRequest, HistoryItem, HistoryToolCall, ModelReasoningEffort, ModelRequestMetadata,
    PromptMessage, RequestBuilderInput, build_request,
};
use crate::skills::{SkillCard, SkillRegistry, SkillTool};
use crate::tool::{ToolHandler, ToolRegistry, ToolResult};
use crate::tool_format::format_tool_call;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ToolExecutionStatus {
    Executed,
    Rejected,
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

#[derive(Debug, Clone)]
pub enum AgentEvent {
    TurnStarted(TurnStartedEvent),
    TokenUsageUpdated {
        used_tokens: u64,
        context_window_tokens: u64,
    },
    ReasoningDelta {
        item_id: String,
        delta: String,
    },
    ReasoningDone {
        item_id: String,
        text: String,
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
    ToolCallFinished {
        call_id: String,
        name: String,
        ok: bool,
        output: ToolResult,
    },
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
    TurnFinalized(TurnFinalizedEvent),
    EvidenceRecorded(EvidenceRecord),
}

pub trait SubagentDelegate<C: Config>: Send + Sync {
    fn run_explorer<'a>(
        &'a self,
        parent: &'a Agent<C>,
        task: String,
    ) -> Pin<Box<dyn Future<Output = Result<ToolResult>> + Send + 'a>>;
    fn run_fixer<'a>(
        &'a self,
        parent: &'a Agent<C>,
        task: String,
    ) -> Pin<Box<dyn Future<Output = Result<ToolResult>> + Send + 'a>>;
}

#[derive(Debug, Clone)]
pub enum ConversationRole {
    User,
    Assistant,
}

#[derive(Debug, Clone)]
pub struct ConversationMessage {
    pub role: ConversationRole,
    pub content: String,
}

const DEFAULT_AGENT_PRELUDE: &str = r#"You are a coding agent operating inside a local repository.
Work from the actual project state. Inspect relevant files before changing code. Prefer the smallest correct change that follows existing patterns.
Use tools deliberately: read/search before editing, edit only intended files, and run the validation that fits the task after changes when it is relevant.
Stay within scope. Do not refactor, reformat, rename, or modify unrelated code unless necessary; if broader changes are needed, explain why.
When tools, edits, or validation fail, inspect the error before retrying. Do not hide failures with broad fallbacks or skipped validation; fail fast and explain the actionable cause.
Use context efficiently: search before reading large files, read only relevant sections, avoid dumping long outputs, and summarize state for long tasks.
When requirements are ambiguous or risky, ask a concise clarifying question.
Keep responses concise. Summarize changed files and validation results when code was modified."#;

const ENGINEERING_WORKFLOW_PRELUDE: &str = r#"This turn is an engineering workflow task.
Delegate bounded work when it improves quality, speed, or context hygiene, especially for low-level or read-heavy tasks that would otherwise pollute the main agent context.
Keep delegation controlled: avoid recursive delegation, avoid unnecessary multi-agent orchestration, and preserve a clear parent agent narrative.
For non-trivial work, keep a short working plan, track the steps you complete, and surface any remaining work or blockers before you stop."#;
const SESSION_TITLE_PRELUDE: &str = r#"Generate a concise session title for the user's first message.
Return only the title text.
Do not use quotes, bullets, markdown, prefixes, or explanations.
Keep it specific and under 80 characters."#;
const MAX_SKILL_CARDS_IN_PRELUDE: usize = 64;

pub struct Agent<C: Config> {
    pub client: Client<C>,
    model: String,
    subagent_model_overrides: HashMap<String, String>,
    default_protocol: ApiProtocol,
    model_protocols: HashMap<String, ApiProtocol>,
    model_catalog: HashMap<String, ModelRequestMetadata>,
    prelude: Vec<PromptMessage>,
    history: Vec<HistoryItem>,
    evidence: Vec<EvidenceRecord>,
    tools: ToolRegistry,
    skill_registry: Option<Arc<SkillRegistry>>,
    skill_cards: Vec<SkillCard>,
    subagent_delegate: Option<Arc<dyn SubagentDelegate<C>>>,
    permission_policy: PermissionPolicy,
    turn: TurnRuntimeState,
    next_turn_id: u64,
    max_iterations: usize,
    max_tool_calls: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentTemplate {
    pub name: String,
    pub purpose: String,
    pub system_prompt: String,
    pub tool_scope: ToolScope,
    pub permission_mode: PermissionMode,
    pub timeout_secs: Option<u64>,
}

impl AgentTemplate {
    pub fn explorer() -> Self {
        Self {
            name: "explorer".into(),
            purpose: "只读仓库探索".into(),
            system_prompt: concat!(
                "你是一个只读的 explorer 子代理。请围绕分配给你的任务调查本地项目，仓库，文件夹等、给出结论，",
                "并且只能使用只读工具。不要编辑文件，不要运行具备写能力的命令，也不要继续委派。"
            )
            .into(),
            tool_scope: ToolScope::ReadOnlyExplorer,
            permission_mode: PermissionMode::Default,
            timeout_secs: None,
        }
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
            timeout_secs: None,
        }
    }
}

pub struct AgentFactory;

impl AgentFactory {
    pub fn create_child<C: Config + Clone>(
        parent: &Agent<C>,
        template: &AgentTemplate,
    ) -> Agent<C> {
        let mut prelude = parent.prelude.clone();
        prelude.push(PromptMessage::developer(template.system_prompt.clone()));
        let mut permission_policy = PermissionPolicy::default();
        permission_policy.set_mode(template.permission_mode);

        Agent {
            client: parent.client.clone(),
            model: parent
                .subagent_model_override(&template.name)
                .unwrap_or(parent.model())
                .to_string(),
            subagent_model_overrides: parent.subagent_model_overrides.clone(),
            default_protocol: parent.default_protocol,
            model_protocols: parent.model_protocols.clone(),
            model_catalog: parent.model_catalog.clone(),
            prelude,
            history: Vec::new(),
            evidence: Vec::new(),
            tools: parent.tools.scoped(template.tool_scope),
            skill_registry: parent.skill_registry.clone(),
            skill_cards: parent.skill_cards.clone(),
            subagent_delegate: None,
            permission_policy,
            turn: TurnRuntimeState::default(),
            next_turn_id: 0,
            max_iterations: parent.max_iterations,
            max_tool_calls: parent.max_tool_calls,
        }
    }
}

impl<C: Config> Agent<C> {
    pub fn new(
        client: Client<C>,
        model: impl Into<String>,
        max_iterations: usize,
        max_tool_calls: usize,
    ) -> Self {
        Self {
            client,
            model: model.into(),
            subagent_model_overrides: HashMap::new(),
            default_protocol: ApiProtocol::Responses,
            model_protocols: HashMap::new(),
            model_catalog: HashMap::new(),
            prelude: default_agent_prelude(),
            history: vec![],
            evidence: vec![],
            tools: ToolRegistry::default_tools(),
            skill_registry: None,
            skill_cards: Vec::new(),
            subagent_delegate: None,
            permission_policy: PermissionPolicy::default(),
            turn: TurnRuntimeState::default(),
            next_turn_id: 0,
            max_iterations: max_iterations,
            max_tool_calls,
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

    fn active_protocol(&self) -> ApiProtocol {
        self.model_protocols
            .get(&self.model)
            .copied()
            .unwrap_or(self.default_protocol)
    }

    fn active_model_metadata(&self) -> ModelRequestMetadata {
        self.model_catalog
            .get(&self.model)
            .copied()
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

    pub fn permission_mode(&self) -> PermissionMode {
        self.permission_policy.mode()
    }

    pub fn set_permission_mode(&mut self, mode: PermissionMode) {
        self.permission_policy.set_mode(mode);
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn reasoning_effort(&self) -> Option<ModelReasoningEffort> {
        self.active_model_metadata().reasoning_effort
    }

    pub fn tool_scope(&self) -> ToolScope {
        self.tools.scope()
    }

    pub fn subagent_model_override(&self, agent_name: &str) -> Option<&str> {
        self.subagent_model_overrides
            .get(agent_name)
            .map(String::as_str)
    }

    pub fn set_model(&mut self, model: impl Into<String>) {
        self.model = model.into();
    }

    pub fn set_subagent_model_override(
        &mut self,
        agent_name: impl Into<String>,
        model: impl Into<String>,
    ) {
        self.subagent_model_overrides
            .insert(agent_name.into(), model.into());
    }

    pub fn set_reasoning_effort(&mut self, effort: ModelReasoningEffort) {
        let mut metadata = self.active_model_metadata();
        metadata.supports_reasoning = true;
        metadata.reasoning_effort = Some(effort);
        self.model_catalog.insert(self.model.clone(), metadata);
    }

    #[allow(dead_code)]
    pub fn restore_transcript_messages(&mut self, messages: Vec<ConversationMessage>) {
        self.history = messages
            .into_iter()
            .map(|message| match message.role {
                ConversationRole::User => HistoryItem::user(message.content),
                ConversationRole::Assistant => HistoryItem::assistant(message.content),
            })
            .collect();
    }

    #[allow(dead_code)]
    pub fn restore_evidence(&mut self, evidence: Vec<EvidenceRecord>) -> Result<()> {
        let mut restored = Vec::with_capacity(evidence.len());
        for record in evidence {
            require_unique_evidence_id(&restored, &record.id)?;
            restored.push(record);
        }
        self.evidence = restored;
        Ok(())
    }

    pub fn restore_session_context(
        &mut self,
        messages: Vec<ConversationMessage>,
        evidence: Vec<EvidenceRecord>,
        max_turn_id: u64,
    ) -> Result<()> {
        let mut restored_evidence = Vec::with_capacity(evidence.len());
        for record in evidence {
            require_unique_evidence_id(&restored_evidence, &record.id)?;
            restored_evidence.push(record);
        }
        let restored_history = messages
            .into_iter()
            .map(|message| match message.role {
                ConversationRole::User => HistoryItem::user(message.content),
                ConversationRole::Assistant => HistoryItem::assistant(message.content),
            })
            .collect();

        self.history = restored_history;
        self.evidence = restored_evidence;
        self.next_turn_id = max_turn_id;
        self.turn = TurnRuntimeState::default();
        Ok(())
    }

    pub fn add_evidence(&mut self, evidence: EvidenceRecord) -> Result<()> {
        require_unique_evidence_id(&self.evidence, &evidence.id)?;
        self.evidence.push(evidence);
        Ok(())
    }

    #[allow(dead_code)]
    pub fn evidence(&self) -> &[EvidenceRecord] {
        &self.evidence
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
            self.try_register_tool(SkillTool::new(registry))
        }
    }

    pub fn set_subagent_delegate(&mut self, delegate: Arc<dyn SubagentDelegate<C>>) {
        self.subagent_delegate = Some(delegate);
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
            history: Vec::new(),
            evidence: Vec::new(),
            tools: ToolRegistry::new(),
            skill_registry: None,
            skill_cards: Vec::new(),
            subagent_delegate: None,
            permission_policy: PermissionPolicy::default(),
            turn: TurnRuntimeState::default(),
            next_turn_id: 0,
            max_iterations: 1,
            max_tool_calls: 0,
        }
    }

    pub async fn generate_session_title(&mut self, user_input: &str) -> Result<String> {
        let raw = self
            .run_stream(user_input, |_| Ok(()), |_| Ok(()), |_| Ok(false))
            .await?;
        normalize_session_title(&raw)
    }

    #[allow(dead_code)]
    pub async fn run(&mut self, user_input: &str) -> Result<String> {
        self.run_stream(user_input, |_| Ok(()), |_| Ok(()), |_| Ok(true))
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
        Afut: Future<Output = Result<bool>>,
    {
        match self.active_protocol() {
            ApiProtocol::Responses => {
                self.run_responses_stream_async(user_input, on_delta, on_event, approve)
                    .await
            }
            ApiProtocol::Completions => {
                self.run_oai_comp_stream_async(user_input, on_delta, on_event, approve)
                    .await
            }
        }
    }

    async fn run_responses_stream_async<F, E, A, Dfut, Efut, Afut>(
        &mut self,
        user_input: &str,
        mut on_delta: F,
        mut on_event: E,
        mut approve: A,
    ) -> Result<String>
    where
        F: FnMut(&str) -> Dfut,
        E: FnMut(AgentEvent) -> Efut,
        A: FnMut(PermissionRequest) -> Afut,
        Dfut: Future<Output = Result<()>>,
        Efut: Future<Output = Result<()>>,
        Afut: Future<Output = Result<bool>>,
    {
        let turn_prelude = self.prepare_turn_prelude(user_input);
        let protected_start_index = self.history.len();
        self.history.push(HistoryItem::user(user_input));
        Self::emit_audit_event(
            &mut on_event,
            AgentEvent::TurnStarted(self.turn_started_event()),
            "turn_started",
        )
        .await;
        debug!(
            user_input_len = user_input.len(),
            history_len = self.history.len(),
            "user message added to history"
        );

        let mut final_text = String::new();
        let mut tool_call_count = 0;
        let mut continuation_count = 0;

        for iteration in 0..self.max_iterations {
            let mut completed_reasoning_ids = HashSet::new();
            let mut turn_text = String::new();
            debug!(
                iteration,
                model = %self.model,
                history_len = self.history.len(),
                tool_call_count,
                max_tool_calls = self.max_tool_calls,
                "creating streamed response"
            );

            let tool_definitions = self.tool_definitions();
            let build = build_request(RequestBuilderInput {
                protocol: ApiProtocol::Responses,
                model_id: &self.model,
                model: self.active_model_metadata(),
                prelude: &turn_prelude,
                history: &self.history,
                protected_start_index,
                tools: &tool_definitions,
                evidence: &self.evidence,
            })?;
            on_event(AgentEvent::TokenUsageUpdated {
                used_tokens: build.budget.estimated_request_tokens,
                context_window_tokens: build.budget.context_window_tokens,
            })
            .await?;
            if build.budget.truncated {
                debug!(
                    model = %self.model,
                    original_history_items = build.budget.original_history_items,
                    retained_history_items = build.budget.retained_history_items,
                    dropped_history_items = build.budget.dropped_history_items,
                    context_window_tokens = build.budget.context_window_tokens,
                    input_budget_tokens = build.budget.input_budget_tokens,
                    estimated_request_tokens = build.budget.estimated_request_tokens,
                    "request history truncated to fit budget"
                );
            }

            let BuiltRequest::Responses(request) = build.request else {
                return Err(anyhow!("request builder returned non-responses request"));
            };

            let mut stream = self.client.responses().create_stream(request).await?;
            let mut completed_response: Option<Response> = None;
            let mut pending_tool_calls = HashSet::new();

            while let Some(event) = stream.next().await {
                let event = match event {
                    Ok(event) => event,
                    Err(error) if is_ignorable_response_lifecycle_deserialize_error(&error) => {
                        warn!(error = %error, "ignored malformed response lifecycle stream event");
                        continue;
                    }
                    Err(error) => return Err(error.into()),
                };

                match event {
                    ResponseStreamEvent::ResponseOutputTextDelta(event) => {
                        trace!(delta_len = event.delta.len(), "received text delta");
                        on_delta(&event.delta).await?;
                        turn_text.push_str(&event.delta);
                        final_text.push_str(&event.delta);
                    }
                    ResponseStreamEvent::ResponseReasoningSummaryTextDelta(event) => {
                        on_event(AgentEvent::ReasoningDelta {
                            item_id: event.item_id,
                            delta: event.delta,
                        })
                        .await?;
                    }
                    ResponseStreamEvent::ResponseReasoningSummaryTextDone(event) => {
                        completed_reasoning_ids.insert(event.item_id.clone());
                        on_event(AgentEvent::ReasoningDone {
                            item_id: event.item_id,
                            text: event.text,
                        })
                        .await?;
                    }
                    ResponseStreamEvent::ResponseOutputItemAdded(event) => {
                        if let OutputItem::FunctionCall(call) = event.item {
                            emit_tool_call_pending_if_ready(
                                &mut pending_tool_calls,
                                &call.call_id,
                                &call.name,
                                &mut on_event,
                            )
                            .await?;
                        }
                    }
                    ResponseStreamEvent::ResponseCompleted(event) => {
                        debug!(
                            response_id = %event.response.id,
                            output_items = event.response.output.len(),
                            "streamed response completed"
                        );
                        completed_response = Some(event.response);
                    }
                    ResponseStreamEvent::ResponseFailed(event) => {
                        error!(response = ?event.response, "response failed");
                        return Err(anyhow!("response failed: {:#?}", event.response));
                    }
                    ResponseStreamEvent::ResponseIncomplete(event) => {
                        warn!(response = ?event.response, "response incomplete");
                        return Err(anyhow!("response incomplete: {:#?}", event.response));
                    }
                    _ => {}
                }
            }

            let response = completed_response
                .ok_or_else(|| anyhow!("stream ended without response.completed"))?;

            for (index, item) in response.output.iter().enumerate() {
                if let OutputItem::Reasoning(reasoning) = item {
                    let item_id = reasoning
                        .id
                        .clone()
                        .unwrap_or_else(|| format!("reasoning-{iteration}-{index}"));
                    if completed_reasoning_ids.contains(&item_id) {
                        continue;
                    }

                    let text = reasoning_summary_text(item);
                    if !text.is_empty() {
                        on_event(AgentEvent::ReasoningDone { item_id, text }).await?;
                    }
                }
            }

            let tool_calls = response
                .output
                .iter()
                .filter_map(|item| match item {
                    OutputItem::FunctionCall(call) => Some(HistoryToolCall {
                        call_id: call.call_id.clone(),
                        name: call.name.clone(),
                        arguments_json: call.arguments.clone(),
                    }),
                    _ => None,
                })
                .collect::<Vec<_>>();

            self.ensure_tool_call_budget(tool_call_count, tool_calls.len())?;

            tool_call_count += tool_calls.len();

            if tool_calls.is_empty() {
                if turn_text.is_empty() {
                    turn_text = response
                        .output_text()
                        .unwrap_or_else(|| "No response content".to_string());
                    final_text.push_str(&turn_text);
                }

                self.history.push(HistoryItem::assistant(turn_text.clone()));

                if self
                    .continue_or_finalize_no_tool_reply(
                        &mut on_event,
                        tool_call_count,
                        &mut continuation_count,
                    )
                    .await?
                {
                    continue;
                }

                info!(
                    output_chars = final_text.chars().count(),
                    history_len = self.history.len(),
                    "final answer completed"
                );

                return Ok(final_text);
            }

            self.append_assistant_tool_calls(&turn_text, &tool_calls);

            debug!(
                iteration,
                tool_calls = tool_calls.len(),
                tool_call_count,
                history_len = self.history.len(),
                "response tool calls appended to history"
            );

            for call in tool_calls {
                info!(
                    tool_name = %call.name,
                    call_id = %call.call_id,
                    "tool call requested"
                );
                debug!(
                    tool_name = %call.name,
                    call_id = %call.call_id,
                    arguments = %call.arguments_json,
                    "tool call arguments"
                );

                self.execute_tool_call_and_record(&call, &mut on_event, &mut approve)
                    .await?;
            }
        }

        Err(anyhow!(
            "stopped: too many agent iterations (max {})",
            self.max_iterations
        ))
    }

    async fn run_oai_comp_stream_async<F, E, A, Dfut, Efut, Afut>(
        &mut self,
        user_input: &str,
        mut on_delta: F,
        mut on_event: E,
        mut approve: A,
    ) -> Result<String>
    where
        F: FnMut(&str) -> Dfut,
        E: FnMut(AgentEvent) -> Efut,
        A: FnMut(PermissionRequest) -> Afut,
        Dfut: Future<Output = Result<()>>,
        Efut: Future<Output = Result<()>>,
        Afut: Future<Output = Result<bool>>,
    {
        let turn_prelude = self.prepare_turn_prelude(user_input);
        let protected_start_index = self.history.len();
        self.history.push(HistoryItem::user(user_input));
        Self::emit_audit_event(
            &mut on_event,
            AgentEvent::TurnStarted(self.turn_started_event()),
            "turn_started",
        )
        .await;
        debug!(
            user_input_len = user_input.len(),
            history_len = self.history.len(),
            "user message added to history"
        );

        let mut final_text = String::new();
        let mut tool_call_count = 0;
        let mut continuation_count = 0;

        for iteration in 0..self.max_iterations {
            debug!(
                iteration,
                model = %self.model,
                history_len = self.history.len(),
                tool_call_count,
                max_tool_calls = self.max_tool_calls,
                "creating streamed chat completion"
            );

            let tool_definitions = self.tool_definitions();
            let build = build_request(RequestBuilderInput {
                protocol: ApiProtocol::Completions,
                model_id: &self.model,
                model: self.active_model_metadata(),
                prelude: &turn_prelude,
                history: &self.history,
                protected_start_index,
                tools: &tool_definitions,
                evidence: &self.evidence,
            })?;
            on_event(AgentEvent::TokenUsageUpdated {
                used_tokens: build.budget.estimated_request_tokens,
                context_window_tokens: build.budget.context_window_tokens,
            })
            .await?;
            if build.budget.truncated {
                debug!(
                    model = %self.model,
                    original_history_items = build.budget.original_history_items,
                    retained_history_items = build.budget.retained_history_items,
                    dropped_history_items = build.budget.dropped_history_items,
                    context_window_tokens = build.budget.context_window_tokens,
                    input_budget_tokens = build.budget.input_budget_tokens,
                    estimated_request_tokens = build.budget.estimated_request_tokens,
                    "request history truncated to fit budget"
                );
            }
            let BuiltRequest::Completions(request) = build.request else {
                return Err(anyhow!("request builder returned non-completions request"));
            };

            let response = send_compatible_chat_completion_stream(&self.client, &request).await?;
            let mut byte_stream = response.bytes_stream();
            let mut sse_buffer = String::new();
            let mut turn_text = String::new();
            let mut tool_calls: BTreeMap<usize, ChatCompletionMessageToolCall> = BTreeMap::new();
            let mut pending_tool_calls = HashSet::new();
            let mut finish_reasons: Vec<FinishReason> = Vec::new();
            let mut reasoning =
                InlineReasoningExtractor::new(format!("chat-reasoning-{iteration}"));
            let mut native_reasoning =
                NativeReasoningAccumulator::new(format!("chat-native-reasoning-{iteration}"));

            while let Some(chunk) = byte_stream.next().await {
                let chunk = chunk?;
                append_sse_chunk(&mut sse_buffer, &chunk);
                let events = drain_sse_data_events(&mut sse_buffer);
                for event in events {
                    let Some(data) = event else {
                        continue;
                    };
                    let response: CompatibleChatCompletionStreamResponse =
                        serde_json::from_str(&data).with_context(|| {
                            format!("failed to parse chat completions stream event: {data}")
                        })?;
                    for choice in response.choices {
                        if choice.index != 0 {
                            return Err(anyhow!(
                                "completions returned unexpected choice index {}; only n=1/index 0 is supported",
                                choice.index
                            ));
                        }

                        if let Some(delta) = choice.delta {
                            if let Some(reasoning_delta) = delta.reasoning_delta() {
                                if let Some(event) = native_reasoning.push(reasoning_delta) {
                                    on_event(event).await?;
                                }
                            }

                            if let Some(content_delta) = delta.content {
                                trace!(delta_len = content_delta.len(), "received chat text delta");
                                for part in reasoning.push(&content_delta) {
                                    match part {
                                        StreamTextPart::Visible(text) => {
                                            on_delta(&text).await?;
                                            turn_text.push_str(&text);
                                            final_text.push_str(&text);
                                        }
                                        StreamTextPart::ReasoningDelta { item_id, delta } => {
                                            on_event(AgentEvent::ReasoningDelta { item_id, delta })
                                                .await?;
                                        }
                                        StreamTextPart::ReasoningDone { item_id, text } => {
                                            on_event(AgentEvent::ReasoningDone { item_id, text })
                                                .await?;
                                        }
                                    }
                                }
                            }

                            if let Some(chunks) = delta.tool_calls {
                                for chunk in chunks {
                                    let index = chunk.index as usize;
                                    merge_chat_tool_call_chunk(&mut tool_calls, chunk);
                                    if let Some(call) = tool_calls.get(&index) {
                                        emit_tool_call_pending_if_ready(
                                            &mut pending_tool_calls,
                                            &call.id,
                                            &call.function.name,
                                            &mut on_event,
                                        )
                                        .await?;
                                    }
                                }
                            }
                        }

                        if let Some(reason) = choice.finish_reason {
                            finish_reasons.push(reason);
                        }
                    }
                }
            }

            let events = finish_sse_data_events(&mut sse_buffer);
            for event in events {
                let Some(data) = event else {
                    continue;
                };
                let response: CompatibleChatCompletionStreamResponse = serde_json::from_str(&data)
                    .with_context(|| {
                        format!("failed to parse chat completions stream event: {data}")
                    })?;
                for choice in response.choices {
                    if choice.index != 0 {
                        return Err(anyhow!(
                            "completions returned unexpected choice index {}; only n=1/index 0 is supported",
                            choice.index
                        ));
                    }

                    if let Some(delta) = choice.delta {
                        if let Some(reasoning_delta) = delta.reasoning_delta() {
                            if let Some(event) = native_reasoning.push(reasoning_delta) {
                                on_event(event).await?;
                            }
                        }

                        if let Some(content_delta) = delta.content {
                            trace!(delta_len = content_delta.len(), "received chat text delta");
                            for part in reasoning.push(&content_delta) {
                                match part {
                                    StreamTextPart::Visible(text) => {
                                        on_delta(&text).await?;
                                        turn_text.push_str(&text);
                                        final_text.push_str(&text);
                                    }
                                    StreamTextPart::ReasoningDelta { item_id, delta } => {
                                        on_event(AgentEvent::ReasoningDelta { item_id, delta })
                                            .await?;
                                    }
                                    StreamTextPart::ReasoningDone { item_id, text } => {
                                        on_event(AgentEvent::ReasoningDone { item_id, text })
                                            .await?;
                                    }
                                }
                            }
                        }

                        if let Some(chunks) = delta.tool_calls {
                            for chunk in chunks {
                                let index = chunk.index as usize;
                                merge_chat_tool_call_chunk(&mut tool_calls, chunk);
                                if let Some(call) = tool_calls.get(&index) {
                                    emit_tool_call_pending_if_ready(
                                        &mut pending_tool_calls,
                                        &call.id,
                                        &call.function.name,
                                        &mut on_event,
                                    )
                                    .await?;
                                }
                            }
                        }
                    }

                    if let Some(reason) = choice.finish_reason {
                        finish_reasons.push(reason);
                    }
                }
            }

            for part in reasoning.finish() {
                match part {
                    StreamTextPart::Visible(text) => {
                        on_delta(&text).await?;
                        turn_text.push_str(&text);
                        final_text.push_str(&text);
                    }
                    StreamTextPart::ReasoningDelta { item_id, delta } => {
                        on_event(AgentEvent::ReasoningDelta { item_id, delta }).await?;
                    }
                    StreamTextPart::ReasoningDone { item_id, text } => {
                        on_event(AgentEvent::ReasoningDone { item_id, text }).await?;
                    }
                }
            }
            if let Some(event) = native_reasoning.finish() {
                on_event(event).await?;
            }

            let has_tool_calls = !tool_calls.is_empty();
            validate_chat_finish_reasons(&finish_reasons, has_tool_calls)?;

            if !has_tool_calls {
                if final_text.is_empty() {
                    final_text = "No response content".to_string();
                }

                self.history.push(HistoryItem::assistant(turn_text.clone()));

                if self
                    .continue_or_finalize_no_tool_reply(
                        &mut on_event,
                        tool_call_count,
                        &mut continuation_count,
                    )
                    .await?
                {
                    continue;
                }

                info!(
                    output_chars = final_text.chars().count(),
                    history_len = self.history.len(),
                    "final chat completion answer completed"
                );

                return Ok(final_text);
            }

            let tool_calls = compact_indexed_chat_tool_calls(tool_calls);
            validate_chat_tool_calls(&tool_calls)?;
            let tool_calls = tool_calls
                .into_iter()
                .map(|call| HistoryToolCall {
                    call_id: call.id,
                    name: call.function.name,
                    arguments_json: call.function.arguments,
                })
                .collect::<Vec<_>>();

            self.ensure_tool_call_budget(tool_call_count, tool_calls.len())?;

            tool_call_count += tool_calls.len();
            self.append_assistant_tool_calls(&turn_text, &tool_calls);

            for call in tool_calls {
                info!(
                    tool_name = %call.name,
                    call_id = %call.call_id,
                    "chat tool call requested"
                );
                debug!(
                    tool_name = %call.name,
                    call_id = %call.call_id,
                    arguments = %call.arguments_json,
                    "chat tool call arguments"
                );

                self.execute_tool_call_and_record(&call, &mut on_event, &mut approve)
                    .await?;
            }
        }

        Err(anyhow!(
            "stopped: too many agent iterations (max {})",
            self.max_iterations
        ))
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
        Afut: Future<Output = Result<bool>>,
    {
        let record = match serde_json::from_str::<Value>(&call.arguments_json) {
            Ok(args) => {
                let directive = self.turn.policy.directive;
                let permission_class = self.tools.permission_class(&call.name);

                if !self.tools.scope().allows_tool(&call.name) {
                    let output = ToolResult::err(
                        &call.name,
                        self.tools.scope().rejection_message(&call.name),
                    );
                    let record = ToolExecutionRecord::new(
                        call,
                        Some(args),
                        permission_class,
                        directive,
                        ToolExecutionStatus::Rejected,
                        Some(ToolExecutionRejection::ToolScopeDenied),
                        output,
                    );
                    on_event(AgentEvent::ToolCallFinished {
                        call_id: call.call_id.clone(),
                        name: call.name.clone(),
                        ok: record.output.ok,
                        output: record.output.clone(),
                    })
                    .await?;
                    self.record_tool_effects(&record);
                    Self::emit_audit_event(
                        on_event,
                        AgentEvent::ToolExecutionSummary(
                            self.tool_execution_summary_event(&record),
                        ),
                        "tool_execution_summary",
                    )
                    .await;
                    return Ok(record);
                }

                if let Some(message) = restricted_by_directive_with_class(
                    &call.name,
                    &args,
                    permission_class,
                    directive,
                ) {
                    let output = ToolResult::err(&call.name, message);
                    let record = ToolExecutionRecord::new(
                        call,
                        Some(args),
                        permission_class,
                        directive,
                        ToolExecutionStatus::Rejected,
                        Some(ToolExecutionRejection::DirectiveBlocked),
                        output,
                    );
                    on_event(AgentEvent::ToolCallFinished {
                        call_id: call.call_id.clone(),
                        name: call.name.clone(),
                        ok: record.output.ok,
                        output: record.output.clone(),
                    })
                    .await?;
                    self.record_tool_effects(&record);
                    Self::emit_audit_event(
                        on_event,
                        AgentEvent::ToolExecutionSummary(
                            self.tool_execution_summary_event(&record),
                        ),
                        "tool_execution_summary",
                    )
                    .await;
                    return Ok(record);
                }

                let permission_decision = self.permission_policy.check_class_with_directive(
                    &call.name,
                    &args,
                    permission_class,
                    directive,
                );
                let should_execute = if is_workflow_control_tool(&call.name) {
                    true
                } else {
                    match permission_decision {
                        PermissionDecision::Allow => true,
                        PermissionDecision::Ask => {
                            approve(PermissionRequest {
                                call_id: Some(call.call_id.clone()),
                                tool: call.name.clone(),
                                args: args.clone(),
                                class: permission_class,
                                summary: format_tool_call(&call.name, &args),
                                preview: None,
                            })
                            .await?
                        }
                        PermissionDecision::Deny => false,
                    }
                };

                if should_execute {
                    on_event(AgentEvent::ToolCallStarted {
                        call_id: call.call_id.clone(),
                        name: call.name.clone(),
                        args: args.clone(),
                    })
                    .await?;

                    let output = if call.name == "agent__explore" || call.name == "agent__fixer" {
                        self.execute_subagent_tool(&call.name, &args).await
                    } else {
                        self.tools.call(&call.name, args.clone()).await
                    };

                    if output.ok {
                        self.apply_control_tool_state(&call.name, &args, on_event)
                            .await?;
                    }

                    on_event(AgentEvent::ToolCallFinished {
                        call_id: call.call_id.clone(),
                        name: call.name.clone(),
                        ok: output.ok,
                        output: output.clone(),
                    })
                    .await?;

                    ToolExecutionRecord::new(
                        call,
                        Some(args),
                        permission_class,
                        directive,
                        ToolExecutionStatus::Executed,
                        None,
                        output,
                    )
                } else {
                    let output = if matches!(permission_decision, PermissionDecision::Deny) {
                        ToolResult::err(&call.name, "permission denied by current mode")
                    } else {
                        ToolResult::err(&call.name, "user denied permission")
                    };
                    let rejection = if matches!(permission_decision, PermissionDecision::Deny) {
                        ToolExecutionRejection::PermissionDeniedByPolicy
                    } else {
                        ToolExecutionRejection::PermissionDeniedByUser
                    };
                    let record = ToolExecutionRecord::new(
                        call,
                        Some(args),
                        permission_class,
                        directive,
                        ToolExecutionStatus::Rejected,
                        Some(rejection),
                        output,
                    );
                    on_event(AgentEvent::ToolCallFinished {
                        call_id: call.call_id.clone(),
                        name: call.name.clone(),
                        ok: record.output.ok,
                        output: record.output.clone(),
                    })
                    .await?;
                    record
                }
            }
            Err(err) => {
                warn!(
                    tool_name = %call.name,
                    call_id = %call.call_id,
                    error = %err,
                    raw_arguments = %call.arguments_json,
                    "invalid tool call JSON arguments"
                );
                let output = ToolResult::err(
                    &call.name,
                    format!(
                        "invalid JSON arguments: {err}; raw: {}",
                        call.arguments_json
                    ),
                );
                on_event(AgentEvent::ToolCallFinished {
                    call_id: call.call_id.clone(),
                    name: call.name.clone(),
                    ok: false,
                    output: output.clone(),
                })
                .await?;
                ToolExecutionRecord::new(
                    call,
                    None,
                    self.tools.permission_class(&call.name),
                    self.turn.policy.directive,
                    ToolExecutionStatus::Rejected,
                    Some(ToolExecutionRejection::InvalidJsonArguments),
                    output,
                )
            }
        };

        self.record_tool_effects(&record);
        Self::emit_audit_event(
            on_event,
            AgentEvent::ToolExecutionSummary(self.tool_execution_summary_event(&record)),
            "tool_execution_summary",
        )
        .await;
        Ok(record)
    }

    async fn execute_subagent_tool(&self, tool_name: &str, args: &Value) -> ToolResult {
        let Some(task) = args.get("task").and_then(Value::as_str).map(str::trim) else {
            return ToolResult::err(
                tool_name,
                format!("{tool_name} requires string field 'task'"),
            );
        };
        if task.is_empty() {
            return ToolResult::err(tool_name, format!("{tool_name} task must not be empty"));
        }

        let Some(delegate) = self.subagent_delegate.clone() else {
            return ToolResult::err(
                tool_name,
                format!("{tool_name} is unavailable outside a subagent-capable runtime"),
            );
        };

        let result = match tool_name {
            "agent__explore" => delegate.run_explorer(self, task.to_string()).await,
            "agent__fixer" => delegate.run_fixer(self, task.to_string()).await,
            _ => Err(anyhow!("unknown subagent tool: {tool_name}")),
        };

        match result {
            Ok(result) => result,
            Err(error) => ToolResult::err(tool_name, error.to_string()),
        }
    }

    fn tool_definitions(&self) -> Vec<crate::request_builder::ToolSpec> {
        let mut specs = self.tools.specs();
        if self.subagent_delegate.is_none() {
            specs.retain(|spec| spec.name != "agent__explore" && spec.name != "agent__fixer");
        }
        specs
    }

    fn ensure_tool_call_budget(&self, current_count: usize, requested_count: usize) -> Result<()> {
        let total_count = current_count + requested_count;
        if total_count > self.max_tool_calls {
            return Err(anyhow!(
                "stopped: too many tool calls ({} requested, max {})",
                total_count,
                self.max_tool_calls
            ));
        }

        Ok(())
    }

    fn append_assistant_tool_calls(&mut self, turn_text: &str, tool_calls: &[HistoryToolCall]) {
        self.history.push(HistoryItem::AssistantToolCalls {
            text: if turn_text.is_empty() {
                None
            } else {
                Some(turn_text.to_string())
            },
            calls: tool_calls.to_vec(),
        });
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
        Afut: Future<Output = Result<bool>>,
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
        self.history.push(HistoryItem::ToolOutput {
            call_id: call.call_id.clone(),
            output_json,
        });

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
        let draft = EvidenceDraft::from_tool_execution_record(record);
        let sequence = self.next_evidence_sequence();
        let id = draft
            .id
            .clone()
            .unwrap_or_else(|| format!("ev-agent-{sequence:06}"));
        let record = draft.into_record(id, sequence, 0)?;
        self.add_evidence(record.clone())?;
        Ok(record)
    }

    fn next_evidence_sequence(&self) -> u64 {
        self.evidence
            .iter()
            .map(|record| record.sequence)
            .max()
            .unwrap_or(0)
            .saturating_add(1)
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
        A: FnMut(PermissionRequest) -> Result<bool>,
    {
        self.run_stream_async(
            user_input,
            |delta| std::future::ready(on_delta(delta)),
            |event| std::future::ready(on_event(event)),
            |request| std::future::ready(approve(request)),
        )
        .await
    }

    fn prepare_turn_prelude(&mut self, user_input: &str) -> Vec<PromptMessage> {
        let turn = WorkflowTurnState::from_user_input(user_input);
        self.next_turn_id = self.next_turn_id.saturating_add(1);
        self.turn = TurnRuntimeState::new(self.next_turn_id, turn.clone());

        let mut turn_prelude = self.prelude.clone();
        turn_prelude.push(runtime_context_message());
        if let Some(message) = self.skill_prelude_message() {
            turn_prelude.push(message);
        }
        if let Some(message) = turn.developer_context_message() {
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

        Some(PromptMessage::developer(text))
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
                self.turn.workflow.todos = payload.items;
                on_event(AgentEvent::TodoSnapshotUpdated {
                    items: self.turn.workflow.todos.clone(),
                })
                .await?;
            }
            "workflow__auto_continue" => {
                let payload: WorkflowAutoContinuePayload = serde_json::from_value(args.clone())?;
                self.turn.workflow.auto_continue.enabled = payload.enabled;
                if let Some(max_continuations) = payload.max_continuations {
                    if max_continuations > AutoContinueState::ABSOLUTE_MAX_CONTINUATIONS {
                        return Err(anyhow!(
                            "max_continuations {max_continuations} exceeds maximum {}",
                            AutoContinueState::ABSOLUTE_MAX_CONTINUATIONS
                        ));
                    }
                    self.turn.workflow.auto_continue.max_continuations = max_continuations;
                }
                on_event(AgentEvent::AutoContinueChanged {
                    state: self.turn.workflow.auto_continue.clone(),
                })
                .await?;
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
                self.history.push(HistoryItem::internal_continuation(
                    "Continue the current task internally. Do not repeat finished work. Focus on unfinished todo items and stop when they are complete or blocked.",
                ));
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
        (self.turn.counters.write_effects > 0 && self.turn.counters.validation_effects == 0).then(|| {
            let failed_validation_effects = self.turn.counters.failed_validation_effects;
            let message = if failed_validation_effects > 0 {
                "This turn made write changes and validation ran but failed. Review the failed validation output before relying on the changes."
            } else {
                "This turn made write changes without running validation. Review and run the most relevant checks if needed."
            };

            ValidationAdvisory {
                write_effects: self.turn.counters.write_effects,
                validation_effects: self.turn.counters.validation_effects,
                failed_validation_effects,
                message: message.into(),
            }
        })
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
            match tool_name {
                "fs__read"
                | "fs__list"
                | "skill"
                | "search__rg"
                | "agent__explore"
                | "code__ast_search"
                | "git__status"
                | "git__diff"
                | "git__log"
                | "code__ast_replace_preview" => ToolEffectKind::Read,
                "agent__fixer" | "fs__write" | "fs__append" | "fs__mkdir" | "edit__apply_patch" => {
                    ToolEffectKind::Write
                }
                "shell__exec" if command.as_deref().is_some_and(is_validation_command_text) => {
                    if shell_command_succeeded(output) {
                        ToolEffectKind::Validation
                    } else {
                        ToolEffectKind::Diagnostic
                    }
                }
                "shell__exec" => ToolEffectKind::Command,
                "workflow__todos" | "workflow__auto_continue" => ToolEffectKind::WorkflowControl,
                _ => ToolEffectKind::Unknown,
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
    const ABSOLUTE_MAX_CONTINUATIONS: usize = 8;
}

impl Default for AutoContinueState {
    fn default() -> Self {
        Self {
            enabled: false,
            max_continuations: Self::DEFAULT_MAX_CONTINUATIONS,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TurnRuntimeState {
    turn_id: u64,
    policy: WorkflowTurnState,
    workflow: WorkflowState,
    counters: TurnCounters,
    last_continuation_todos: Option<Vec<TodoItem>>,
}

impl TurnRuntimeState {
    fn new(turn_id: u64, policy: WorkflowTurnState) -> Self {
        Self {
            turn_id,
            policy,
            workflow: WorkflowState::default(),
            counters: TurnCounters::default(),
            last_continuation_todos: None,
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

        Some(PromptMessage::developer(text))
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

fn is_workflow_control_tool(tool_name: &str) -> bool {
    matches!(tool_name, "workflow__todos" | "workflow__auto_continue")
}

fn is_cancelled_subagent_record(record: &ToolExecutionRecord) -> bool {
    matches!(record.tool_name.as_str(), "agent__explore" | "agent__fixer")
        && record
            .output
            .data
            .as_ref()
            .and_then(|data| data.get("status"))
            .and_then(Value::as_str)
            == Some("cancelled")
}

impl ToolExecutionStatus {
    fn as_str(self) -> &'static str {
        match self {
            Self::Executed => "executed",
            Self::Rejected => "rejected",
        }
    }
}

impl ToolExecutionRejection {
    fn as_str(self) -> &'static str {
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

fn is_ignorable_response_lifecycle_deserialize_error(error: &OpenAIError) -> bool {
    let OpenAIError::JSONDeserialize(source, content) = error else {
        return false;
    };

    source.to_string().contains("missing field `model`")
        && serde_json::from_str::<Value>(content)
            .ok()
            .and_then(|value| value.get("type").and_then(Value::as_str).map(str::to_owned))
            .as_deref()
            .is_some_and(|event_type| {
                matches!(event_type, "response.created" | "response.in_progress")
            })
}

async fn send_compatible_chat_completion_stream<C: Config>(
    client: &Client<C>,
    request: &impl Serialize,
) -> Result<reqwest::Response> {
    let config = client.config();
    let response = reqwest::Client::new()
        .post(config.url("/chat/completions"))
        .query(&config.query())
        .headers(config.headers())
        .json(request)
        .send()
        .await
        .context("failed to create streamed chat completion")?;

    let status = response.status();
    if !status.is_success() {
        let message = response
            .text()
            .await
            .unwrap_or_else(|error| format!("failed to read error body: {error}"));
        bail!("chat completions request failed with status {status}: {message}");
    }

    Ok(response)
}

#[derive(Debug, Deserialize)]
struct CompatibleChatCompletionStreamResponse {
    choices: Vec<CompatibleChatChoiceStream>,
}

#[derive(Debug, Deserialize)]
struct CompatibleChatChoiceStream {
    index: u32,
    delta: Option<CompatibleChatCompletionStreamResponseDelta>,
    finish_reason: Option<FinishReason>,
}

#[derive(Debug, Deserialize)]
struct CompatibleChatCompletionStreamResponseDelta {
    content: Option<String>,
    tool_calls: Option<Vec<ChatCompletionMessageToolCallChunk>>,
    reasoning_content: Option<CompatibleReasoningDelta>,
    reasoning: Option<CompatibleReasoningDelta>,
    thinking: Option<CompatibleReasoningDelta>,
}

impl CompatibleChatCompletionStreamResponseDelta {
    fn reasoning_delta(&self) -> Option<String> {
        [
            self.reasoning_content.as_ref(),
            self.reasoning.as_ref(),
            self.thinking.as_ref(),
        ]
        .into_iter()
        .flatten()
        .find_map(|reasoning| reasoning.to_text().filter(|text| !text.is_empty()))
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum CompatibleReasoningDelta {
    Text(String),
    Object {
        content: Option<String>,
        text: Option<String>,
        summary: Option<String>,
    },
    Array(Vec<CompatibleReasoningDelta>),
}

impl CompatibleReasoningDelta {
    fn to_text(&self) -> Option<String> {
        match self {
            Self::Text(text) => Some(text.clone()),
            Self::Object {
                content,
                text,
                summary,
            } => content
                .as_ref()
                .or(text.as_ref())
                .or(summary.as_ref())
                .cloned(),
            Self::Array(parts) => {
                let text = parts.iter().filter_map(Self::to_text).collect::<String>();
                (!text.is_empty()).then_some(text)
            }
        }
    }
}

#[derive(Debug, Clone)]
struct NativeReasoningAccumulator {
    item_id: String,
    text: String,
}

impl NativeReasoningAccumulator {
    fn new(item_id: impl Into<String>) -> Self {
        Self {
            item_id: item_id.into(),
            text: String::new(),
        }
    }

    fn push(&mut self, delta: String) -> Option<AgentEvent> {
        if delta.is_empty() {
            return None;
        }
        self.text.push_str(&delta);
        Some(AgentEvent::ReasoningDelta {
            item_id: self.item_id.clone(),
            delta,
        })
    }

    fn finish(self) -> Option<AgentEvent> {
        (!self.text.is_empty()).then_some(AgentEvent::ReasoningDone {
            item_id: self.item_id,
            text: self.text,
        })
    }
}

fn append_sse_chunk(buffer: &mut String, chunk: &[u8]) {
    buffer.push_str(&String::from_utf8_lossy(chunk));
}

fn drain_sse_data_events(buffer: &mut String) -> Vec<Option<String>> {
    let mut events = Vec::new();
    while let Some((index, len)) = find_sse_event_boundary(buffer) {
        let raw = buffer[..index].to_string();
        buffer.drain(..index + len);
        if let Some(event) = parse_sse_data_event(&raw) {
            events.push(event);
        }
    }
    events
}

fn finish_sse_data_events(buffer: &mut String) -> Vec<Option<String>> {
    let mut events = drain_sse_data_events(buffer);
    if !buffer.trim().is_empty() {
        let raw = std::mem::take(buffer);
        if let Some(event) = parse_sse_data_event(&raw) {
            events.push(event);
        }
    }
    events
}

fn find_sse_event_boundary(buffer: &str) -> Option<(usize, usize)> {
    match (buffer.find("\n\n"), buffer.find("\r\n\r\n")) {
        (Some(lf), Some(crlf)) if crlf < lf => Some((crlf, 4)),
        (Some(lf), _) => Some((lf, 2)),
        (None, Some(crlf)) => Some((crlf, 4)),
        (None, None) => None,
    }
}

fn parse_sse_data_event(raw: &str) -> Option<Option<String>> {
    let data = raw
        .lines()
        .filter_map(|line| line.strip_prefix("data:"))
        .map(str::trim_start)
        .collect::<Vec<_>>()
        .join("\n");

    if data.is_empty() {
        return None;
    }
    if data.trim() == "[DONE]" {
        return Some(None);
    }
    Some(Some(data))
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_openai::config::OpenAIConfig;
    use serde_json::json;
    use std::path::PathBuf;

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

    fn test_tool_call(name: &str, arguments_json: &str) -> HistoryToolCall {
        HistoryToolCall {
            call_id: format!("call-{name}"),
            name: name.into(),
            arguments_json: arguments_json.into(),
        }
    }

    struct StaticSubagentDelegate {
        result: ToolResult,
    }

    impl SubagentDelegate<OpenAIConfig> for StaticSubagentDelegate {
        fn run_explorer<'a>(
            &'a self,
            _parent: &'a Agent<OpenAIConfig>,
            _task: String,
        ) -> Pin<Box<dyn Future<Output = Result<ToolResult>> + Send + 'a>> {
            let result = self.result.clone();
            Box::pin(async move { Ok(result) })
        }

        fn run_fixer<'a>(
            &'a self,
            _parent: &'a Agent<OpenAIConfig>,
            _task: String,
        ) -> Pin<Box<dyn Future<Output = Result<ToolResult>> + Send + 'a>> {
            let result = self.result.clone();
            Box::pin(async move { Ok(result) })
        }
    }

    fn static_delegate(result: ToolResult) -> Arc<dyn SubagentDelegate<OpenAIConfig>> {
        Arc::new(StaticSubagentDelegate { result })
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
        assert!(!specs.iter().any(|spec| spec.name == "agent__explore"));
        assert!(!specs.iter().any(|spec| spec.name == "agent__fixer"));

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
        assert!(specs.iter().any(|spec| spec.name == "agent__explore"));
        assert!(specs.iter().any(|spec| spec.name == "agent__fixer"));
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
        let mut events = Vec::new();

        let error = agent
            .execute_tool_call_and_record(
                &call,
                &mut |event| {
                    events.push(event);
                    std::future::ready(Ok(()))
                },
                &mut |_| std::future::ready(Ok(true)),
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
        let mut events = Vec::new();

        let error = agent
            .execute_tool_call_and_record(
                &call,
                &mut |event| {
                    events.push(event);
                    std::future::ready(Ok(()))
                },
                &mut |_| std::future::ready(Ok(true)),
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
        agent.history.push(HistoryItem::user("hello"));
        let b1 = build_request(RequestBuilderInput {
            protocol: ApiProtocol::Responses,
            model_id: agent.model(),
            model: agent.active_model_metadata(),
            prelude: &agent.prelude,
            history: &agent.history,
            protected_start_index: agent.history.len().saturating_sub(1),
            tools: &[],
            evidence: &[],
        })
        .expect("request builds");
        assert_eq!(b1.budget.context_window_tokens, 2048.max(1024));

        // Switch model and build again.
        agent.set_model("m2");
        let b2 = build_request(RequestBuilderInput {
            protocol: ApiProtocol::Responses,
            model_id: agent.model(),
            model: agent.active_model_metadata(),
            prelude: &agent.prelude,
            history: &agent.history,
            protected_start_index: agent.history.len().saturating_sub(1),
            tools: &[],
            evidence: &[],
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
    fn ignores_non_terminal_lifecycle_events_missing_model_deserialize_error() {
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
            let error = serde_json::from_value::<ResponseStreamEvent>(raw.clone())
                .expect_err("lifecycle event without model should not deserialize");
            let error = OpenAIError::JSONDeserialize(error, raw.to_string());

            assert!(
                is_ignorable_response_lifecycle_deserialize_error(&error),
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
        let error = serde_json::from_value::<ResponseStreamEvent>(raw.clone())
            .expect_err("response.completed without model should not deserialize");
        let error = OpenAIError::JSONDeserialize(error, raw.to_string());

        assert!(!is_ignorable_response_lifecycle_deserialize_error(&error));
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
                std::future::ready(Ok(true))
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
                &mut |_| std::future::ready(Ok(true)),
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
                std::future::ready(Ok(true))
            })
            .await
            .expect("todo control tool should succeed");

        assert_eq!(agent.todos().len(), 2);
        assert_eq!(agent.todos()[0].status, TodoStatus::Pending);
        assert_eq!(agent.todos()[1].status, TodoStatus::Completed);
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
            [AgentEvent::AutoContinuationScheduled {
                continuation_count: 1,
                remaining_unfinished: 1,
            }]
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
        assert!(title_agent.evidence.is_empty());
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
    fn empty_skill_registry_does_not_register_skill_tool_or_prelude() {
        let mut agent = test_agent();
        agent
            .register_skill_registry(Arc::new(SkillRegistry::default()))
            .expect("register empty skill registry");

        assert!(
            !agent
                .tool_definitions()
                .iter()
                .any(|spec| spec.name == "skill")
        );
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
                &mut |_| std::future::ready(Ok(true)),
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
                std::future::ready(Ok(true))
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
                &mut |_| std::future::ready(Ok(true)),
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
                &mut |_| std::future::ready(Ok(true)),
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
                &mut |_| std::future::ready(Ok(true)),
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
        assert!(agent.pending_validation_advisory().is_none());
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
    PromptMessage::developer(format!(
        "Runtime context:\n- Current date: {date}\n- Timezone: {timezone}"
    ))
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

async fn emit_tool_call_pending_if_ready<E, Efut>(
    emitted_pending_tool_calls: &mut HashSet<String>,
    call_id: &str,
    name: &str,
    on_event: &mut E,
) -> Result<()>
where
    E: FnMut(AgentEvent) -> Efut,
    Efut: Future<Output = Result<()>>,
{
    if call_id.trim().is_empty() || name.trim().is_empty() {
        return Ok(());
    }

    if emitted_pending_tool_calls.insert(call_id.to_string()) {
        on_event(AgentEvent::ToolCallPending {
            call_id: call_id.to_string(),
            name: name.to_string(),
        })
        .await?;
    }

    Ok(())
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
