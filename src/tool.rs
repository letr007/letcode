use anyhow::{Context, Result, anyhow, bail};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Component;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use tokio::fs;
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::Command;
use tokio::sync::mpsc;
use tokio::time::{Duration, sleep, timeout};
use tracing::{debug, warn};

use crate::code_analysis::{AstReplacePreviewRequest, AstSearchRequest, CodeAnalysisRegistry};
use crate::context_tools;
use crate::context_tree::ContextTreeState;
use crate::context_view::ContextViewProjection;
use crate::memory;
use crate::permission::{ToolPermissionClass, ToolScope, classify_tool};
use crate::request_builder::ToolSpec;
use crate::tool_names;

const DEFAULT_READ_LINE_LIMIT: usize = 200;
const MAX_READ_LINE_LIMIT: usize = 5_000;
const MAX_READ_BYTES: usize = 4 * 1024 * 1024;
const MAX_COMMAND_OUTPUT_BYTES: usize = 256 * 1024;
const COMMAND_TIMEOUT_SECS: u64 = 300;
const MAX_WORKFLOW_TODOS: usize = 100;
const MAX_WORKFLOW_TODO_FIELD_CHARS: usize = 1_000;
const MAX_WORKFLOW_AUTO_CONTINUATIONS: u64 = 16;
const MAX_CONTEXT_CHECKPOINT_REASON_CHARS: usize = 2_000;
const MAX_CONTEXT_CHECKPOINT_LABEL_CHARS: usize = 120;
const MAX_CONTEXT_RETURN_SUMMARY_CHARS: usize = 2_000;
const MAX_CONTEXT_RETURN_NEXT_ACTION_CHARS: usize = 1_000;
const MAX_SUBAGENT_TEXT_FIELD_CHARS: usize = 16_000;
const MAX_SUBAGENT_LIST_ITEMS: usize = 128;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NormalizedSubagentInput {
    pub objective: String,
    pub success_criteria: Vec<String>,
    pub allowed_paths: Vec<String>,
    pub forbidden_paths: Vec<String>,
    pub owned_paths: Vec<String>,
    pub timeout_secs: Option<u64>,
    pub max_tool_calls: Option<usize>,
}

impl NormalizedSubagentInput {
    pub fn render_for_delegate(&self, tool_name: &str) -> String {
        let mut lines = vec![format!("Objective: {}", self.objective)];

        if !self.success_criteria.is_empty() {
            lines.push("Success criteria:".into());
            lines.extend(self.success_criteria.iter().map(|item| format!("- {item}")));
        }

        if !self.allowed_paths.is_empty() {
            lines.push(format!("Allowed paths: {}", self.allowed_paths.join(", ")));
        }
        if !self.forbidden_paths.is_empty() {
            lines.push(format!(
                "Forbidden paths: {}",
                self.forbidden_paths.join(", ")
            ));
        }
        if !self.owned_paths.is_empty() {
            lines.push(format!("Owned paths: {}", self.owned_paths.join(", ")));
        }

        if self.timeout_secs.is_some() || self.max_tool_calls.is_some() {
            lines.push(format!(
                "Execution bounds: timeout_secs={}, max_tool_calls={}",
                self.timeout_secs
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "inherit".into()),
                self.max_tool_calls
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "inherit".into())
            ));
        }

        lines.push("Delegation contract: do not recursively delegate; stay within the provided scope and report findings or implementation outcome succinctly.".into());

        if tool_name == tool_names::TOOL_AGENT_EXPLORE {
            lines.push("Mode: read-only exploration only.".into());
        }

        lines.join("\n")
    }

    pub fn effective_timeout_secs(&self, default: Option<u64>) -> Option<u64> {
        self.timeout_secs.or(default)
    }

    pub fn effective_max_tool_calls(&self, default: Option<usize>) -> Option<usize> {
        self.max_tool_calls.or(default)
    }

    pub fn has_write_scope(&self) -> bool {
        !(self.allowed_paths.is_empty()
            && self.owned_paths.is_empty()
            && self.forbidden_paths.is_empty())
    }

    pub fn permits_write_path(&self, path: &str) -> bool {
        let normalized = normalize_delegation_path(path);
        if normalized.is_empty() {
            return false;
        }
        if self
            .forbidden_paths
            .iter()
            .any(|prefix| path_matches_scope(&normalized, prefix))
        {
            return false;
        }
        let allowed = self
            .allowed_paths
            .iter()
            .any(|prefix| path_matches_scope(&normalized, prefix));
        let owned = self
            .owned_paths
            .iter()
            .any(|prefix| path_matches_scope(&normalized, prefix));

        if self.allowed_paths.is_empty() && self.owned_paths.is_empty() {
            return true;
        }

        allowed || owned
    }
}

fn normalize_delegation_path(path: &str) -> String {
    path.trim()
        .trim_start_matches("./")
        .trim_matches('/')
        .to_string()
}

fn path_matches_scope(path: &str, scope: &str) -> bool {
    let scope = normalize_delegation_path(scope);
    !scope.is_empty() && (path == scope || path.starts_with(&format!("{scope}/")))
}

pub fn normalize_subagent_input(tool_name: &str, args: &Value) -> Result<NormalizedSubagentInput> {
    let task = optional_trimmed_string(args, "task")?;
    let objective = optional_trimmed_string(args, "objective")?;
    let objective = objective.or(task).ok_or_else(|| {
        anyhow!(
            "{tool_name} requires a non-empty 'task' or 'objective' field to describe the delegated work"
        )
    })?;

    Ok(NormalizedSubagentInput {
        objective,
        success_criteria: optional_trimmed_string_list(args, "success_criteria")?,
        allowed_paths: optional_trimmed_string_list(args, "allowed_paths")?,
        forbidden_paths: optional_trimmed_string_list(args, "forbidden_paths")?,
        owned_paths: optional_trimmed_string_list(args, "owned_paths")?,
        timeout_secs: optional_u64(args, "timeout_secs")?,
        max_tool_calls: optional_u64(args, "max_tool_calls")?.map(|value| value as usize),
    })
}

fn optional_trimmed_string(args: &Value, field: &str) -> Result<Option<String>> {
    let Some(value) = args.get(field) else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    let Some(text) = value.as_str() else {
        bail!("field '{field}' must be a string or null");
    };
    let trimmed = text.trim();
    if trimmed.is_empty() {
        bail!("field '{field}' must not be empty or whitespace when provided");
    }
    if trimmed.chars().count() > MAX_SUBAGENT_TEXT_FIELD_CHARS {
        bail!("field '{field}' exceeds {MAX_SUBAGENT_TEXT_FIELD_CHARS} characters");
    }
    Ok(Some(trimmed.to_string()))
}

fn optional_trimmed_string_list(args: &Value, field: &str) -> Result<Vec<String>> {
    let Some(value) = args.get(field) else {
        return Ok(Vec::new());
    };
    if value.is_null() {
        return Ok(Vec::new());
    }
    let Some(items) = value.as_array() else {
        bail!("field '{field}' must be an array of strings or null");
    };
    if items.len() > MAX_SUBAGENT_LIST_ITEMS {
        bail!("field '{field}' accepts at most {MAX_SUBAGENT_LIST_ITEMS} items");
    }

    items.iter()
        .enumerate()
        .map(|(index, item)| {
            let Some(text) = item.as_str() else {
                bail!("field '{field}' item {index} must be a string");
            };
            let trimmed = text.trim();
            if trimmed.is_empty() {
                bail!("field '{field}' item {index} must not be empty or whitespace");
            }
            if trimmed.chars().count() > MAX_SUBAGENT_TEXT_FIELD_CHARS {
                bail!(
                    "field '{field}' item {index} exceeds {MAX_SUBAGENT_TEXT_FIELD_CHARS} characters"
                );
            }
            Ok(trimmed.to_string())
        })
        .collect()
}

fn optional_u64(args: &Value, field: &str) -> Result<Option<u64>> {
    let Some(value) = args.get(field) else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    let Some(number) = value.as_u64() else {
        bail!("field '{field}' must be an integer or null");
    };
    if number == 0 {
        bail!("field '{field}' must be greater than 0 when provided");
    }
    Ok(Some(number))
}

pub(crate) fn subagent_parameters_schema(task_description: &str) -> Value {
    json!({
        "type": "object",
        "properties": {
            "task": {
                "type": ["string", "null"],
                "description": task_description
            },
            "objective": {
                "type": ["string", "null"],
                "description": "更明确的委派目标；未提供时兼容旧版 task 输入"
            },
            "success_criteria": {
                "type": ["array", "null"],
                "items": {"type": "string"},
                "description": "完成委派任务所需满足的验收条件"
            },
            "allowed_paths": {
                "type": ["array", "null"],
                "items": {"type": "string"},
                "description": "允许子代理读取或修改的路径边界"
            },
            "forbidden_paths": {
                "type": ["array", "null"],
                "items": {"type": "string"},
                "description": "明确禁止触碰的路径边界"
            },
            "owned_paths": {
                "type": ["array", "null"],
                "items": {"type": "string"},
                "description": "当前委派拥有编辑权的路径集合"
            }
        },
        "required": [
            "task",
            "objective",
            "success_criteria",
            "allowed_paths",
            "forbidden_paths",
            "owned_paths"
        ],
        "additionalProperties": false
    })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub ok: bool,
    pub tool: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<ToolError>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolError {
    pub message: String,
    pub recoverable: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolOutputStream {
    Stdout,
    Stderr,
}

impl ToolOutputStream {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Stdout => "stdout",
            Self::Stderr => "stderr",
        }
    }

    pub fn truncated_key(self) -> &'static str {
        match self {
            Self::Stdout => "stdout_truncated",
            Self::Stderr => "stderr_truncated",
        }
    }
}

impl ToolResult {
    pub fn ok(tool: impl Into<String>, data: Value) -> Self {
        Self {
            ok: true,
            tool: tool.into(),
            data: Some(data),
            error: None,
        }
    }

    pub fn err(tool: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            ok: false,
            tool: tool.into(),
            data: None,
            error: Some(ToolError {
                message: message.into(),
                recoverable: true,
            }),
        }
    }

    pub fn err_with_data(tool: impl Into<String>, message: impl Into<String>, data: Value) -> Self {
        Self {
            ok: false,
            tool: tool.into(),
            data: Some(data),
            error: Some(ToolError {
                message: message.into(),
                recoverable: true,
            }),
        }
    }
}

#[async_trait]
pub trait ToolHandler: Send + Sync {
    fn name(&self) -> &str;

    fn description(&self) -> &str;

    fn parameters(&self) -> Value;

    fn strict(&self) -> bool {
        true
    }

    fn permission_class(&self) -> ToolPermissionClass {
        classify_tool(self.name())
    }

    async fn execute(&self, args: Value) -> Result<Value>;

    async fn execute_with_context(
        &self,
        args: Value,
        _context: ToolExecutionContext,
    ) -> Result<Value> {
        self.execute(args).await
    }

    async fn execute_streaming(
        &self,
        args: Value,
        context: ToolExecutionContext,
        _emit: ToolOutputEmitter<'_>,
    ) -> Result<Value> {
        self.execute_with_context(args, context).await
    }

    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: self.name().to_string(),
            description: self.description().to_string(),
            parameters: self.parameters(),
            strict: self.strict(),
        }
    }
}

pub type ToolOutputEmitter<'a> = &'a mut (dyn FnMut(ToolOutputStream, String) -> Result<()> + Send);

#[derive(Debug, Clone, Default)]
pub struct ToolExecutionContext {
    pub allow_outside_workspace: bool,
    pub context_view: Option<Arc<ContextViewProjection>>,
    pub context_tree: Option<Arc<ContextTreeState>>,
}

impl ToolExecutionContext {
    pub fn outside_workspace_granted() -> Self {
        Self {
            allow_outside_workspace: true,
            context_view: None,
            context_tree: None,
        }
    }

    pub fn with_context_view(context_view: Arc<ContextViewProjection>) -> Self {
        Self {
            allow_outside_workspace: false,
            context_view: Some(context_view),
            context_tree: None,
        }
    }

    pub fn with_context_snapshots(
        context_view: Arc<ContextViewProjection>,
        context_tree: Arc<ContextTreeState>,
    ) -> Self {
        Self {
            allow_outside_workspace: false,
            context_view: Some(context_view),
            context_tree: Some(context_tree),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExternalWorkspaceAccess {
    pub paths: Vec<String>,
}

impl ExternalWorkspaceAccess {
    pub fn preview(&self) -> String {
        let mut text = String::from("Outside-workspace access requested:");
        for path in &self.paths {
            text.push_str("\n- ");
            text.push_str(path);
        }
        text
    }
}

#[derive(Clone, Default)]
pub struct ToolRegistry {
    tools: BTreeMap<String, Arc<dyn ToolHandler>>,
    scope: ToolScope,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn default_tools() -> Self {
        let mut registry = Self::new();
        registry.register(EchoTool);
        registry.register(WorkflowTodosTool);
        registry.register(WorkflowAutoContinueTool);
        registry.register(MemoryRecallTool);
        registry.register(ContextCheckpointTool);
        registry.register(ContextReturnTool);
        registry.register(AgentExploreTool);
        registry.register(AgentFixerTool);
        registry.register(ListDirTool);
        registry.register(ReadFileTool);
        registry.register(WriteFileTool);
        registry.register(AppendFileTool);
        registry.register(RunCommandTool);
        registry.register(MkdirTool);
        registry.register(RgTool);
        registry.register(GitStatusTool);
        registry.register(GitDiffTool);
        registry.register(GitLogTool);
        registry.register(ApplyPatchTool);
        registry.register(AstSearchTool);
        registry.register(AstReplacePreviewTool);
        context_tools::register_context_tools(&mut registry);
        registry
    }

    pub fn register<T>(&mut self, tool: T)
    where
        T: ToolHandler + 'static,
    {
        let name = tool.name().to_string();
        self.tools.insert(name, Arc::new(tool));
    }

    pub fn scoped(&self, scope: ToolScope) -> Self {
        Self {
            tools: self.tools.clone(),
            scope,
        }
    }

    pub fn without_tools(mut self, names: &[&str]) -> Self {
        for name in names {
            self.tools.remove(*name);
        }
        self
    }

    pub fn scope(&self) -> ToolScope {
        self.scope
    }

    pub fn try_register<T>(&mut self, tool: T) -> Result<()>
    where
        T: ToolHandler + 'static,
    {
        let name = tool.name().to_string();
        if self.tools.contains_key(&name) {
            bail!("tool '{name}' is already registered");
        }
        self.tools.insert(name, Arc::new(tool));
        Ok(())
    }

    pub fn specs(&self) -> Vec<ToolSpec> {
        self.tools
            .values()
            .filter(|tool| self.scope.allows_tool(tool.name()))
            .map(|tool| tool.spec())
            .collect()
    }

    pub fn permission_class(&self, name: &str) -> ToolPermissionClass {
        self.tools
            .get(name)
            .map(|tool| tool.permission_class())
            .unwrap_or_else(|| classify_tool(name))
    }

    pub async fn call(&self, name: &str, args: Value) -> ToolResult {
        self.call_with_context(name, args, ToolExecutionContext::default())
            .await
    }

    pub async fn call_with_context(
        &self,
        name: &str,
        args: Value,
        context: ToolExecutionContext,
    ) -> ToolResult {
        let mut emit = |_stream: ToolOutputStream, _chunk: String| Ok(());
        self.call_streaming(name, args, context, &mut emit).await
    }

    pub async fn call_streaming(
        &self,
        name: &str,
        args: Value,
        context: ToolExecutionContext,
        emit: ToolOutputEmitter<'_>,
    ) -> ToolResult {
        debug!(tool_name = %name, args = %args, "calling tool");

        if !self.scope.allows_tool(name) {
            warn!(tool_name = %name, scope = %self.scope, "tool rejected by scope");
            return ToolResult::err(name, self.scope.rejection_message(name));
        }

        let Some(tool) = self.tools.get(name) else {
            warn!(tool_name = %name, "unknown tool requested");
            return ToolResult::err(name, format!("unknown tool: {name}"));
        };

        match tool.execute_streaming(args, context, emit).await {
            Ok(data) => ToolResult::ok(name, data),
            Err(err) => {
                warn!(tool_name = %name, error = %err, "tool execution failed");
                ToolResult::err(name, err.to_string())
            }
        }
    }
}

#[allow(dead_code)]
pub fn tool_definitions() -> Vec<ToolSpec> {
    ToolRegistry::default_tools().specs()
}

#[allow(dead_code)]
pub async fn call_tool(name: &str, args: Value) -> ToolResult {
    ToolRegistry::default_tools().call(name, args).await
}

pub fn external_workspace_access_for_tool(
    name: &str,
    args: &Value,
) -> Option<ExternalWorkspaceAccess> {
    let mut paths = BTreeSet::new();

    match name {
        "fs__list" | "fs__read" => {
            if let Some(path) = args.get("path").and_then(Value::as_str)
                && let Some(path) = outside_existing_workspace_path(path)
            {
                paths.insert(path);
            }
        }
        "search__rg" => {
            let path = args.get("path").and_then(Value::as_str).unwrap_or(".");
            if let Some(path) = outside_existing_workspace_path(path) {
                paths.insert(path);
            }
        }
        "fs__write" | "fs__append" => {
            if let Some(path) = args.get("path").and_then(Value::as_str)
                && let Some(path) = outside_writable_workspace_path(path)
            {
                paths.insert(path);
            }
        }
        "fs__mkdir" => {
            if let Some(path) = args.get("path").and_then(Value::as_str)
                && let Some(path) = outside_new_workspace_path(path)
            {
                paths.insert(path);
            }
        }
        "edit__apply_patch" => {
            if let Some(edits) = args.get("edits").and_then(Value::as_array) {
                for edit in edits {
                    if let Some(path) = edit.get("path").and_then(Value::as_str)
                        && let Some(path) = outside_existing_workspace_path(path)
                    {
                        paths.insert(path);
                    }
                }
            }
        }
        "code__ast_search" | "code__ast_replace_preview" => {
            if let Some(path) = args.get("path").and_then(Value::as_str)
                && let Some(path) = outside_existing_workspace_path(path)
            {
                paths.insert(path);
            }
        }
        _ => {}
    }

    if paths.is_empty() {
        None
    } else {
        Some(ExternalWorkspaceAccess {
            paths: paths.into_iter().collect(),
        })
    }
}

struct EchoTool;

struct WorkflowTodosTool;

struct WorkflowAutoContinueTool;

struct ContextCheckpointTool;

struct ContextReturnTool;

struct MemoryRecallTool;

struct AgentExploreTool;

struct AgentFixerTool;

#[async_trait]
impl ToolHandler for EchoTool {
    fn name(&self) -> &'static str {
        "util__echo"
    }

    fn description(&self) -> &'static str {
        "Echo back the provided text."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text to echo back"
                }
            },
            "required": ["text"],
            "additionalProperties": false
        })
    }

    async fn execute(&self, args: Value) -> Result<Value> {
        Ok(json!({
            "result": args.get("text").cloned().unwrap_or(json!(""))
        }))
    }
}

#[async_trait]
impl ToolHandler for WorkflowTodosTool {
    fn name(&self) -> &'static str {
        "workflow__todos"
    }

    fn description(&self) -> &'static str {
        "Update the agent's current todo list for this turn."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "maxItems": MAX_WORKFLOW_TODOS,
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "string",
                                "maxLength": MAX_WORKFLOW_TODO_FIELD_CHARS,
                                "description": "Stable todo item id"
                            },
                            "content": {
                                "type": "string",
                                "maxLength": MAX_WORKFLOW_TODO_FIELD_CHARS,
                                "description": "Short todo description"
                            },
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "blocked", "completed", "cancelled"],
                                "description": "Todo status"
                            }
                        },
                        "required": ["id", "content", "status"],
                        "additionalProperties": false
                    },
                    "description": "Current turn todo list snapshot"
                }
            },
            "required": ["items"],
            "additionalProperties": false
        })
    }

    async fn execute(&self, args: Value) -> Result<Value> {
        validate_workflow_todos(&args)?;
        Ok(args)
    }
}

#[async_trait]
impl ToolHandler for WorkflowAutoContinueTool {
    fn name(&self) -> &'static str {
        "workflow__auto_continue"
    }

    fn description(&self) -> &'static str {
        "Enable or disable bounded internal auto-continuation for unfinished todos."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "enabled": {
                    "type": "boolean",
                    "description": "Whether bounded auto-continuation is enabled"
                },
                "max_continuations": {
                    "type": ["integer", "null"],
                    "minimum": 0,
                    "maximum": MAX_WORKFLOW_AUTO_CONTINUATIONS,
                    "description": "Optional per-turn continuation limit. Use null to keep the default."
                }
            },
            "required": ["enabled", "max_continuations"],
            "additionalProperties": false
        })
    }

    async fn execute(&self, args: Value) -> Result<Value> {
        validate_workflow_auto_continue(&args)?;
        Ok(args)
    }
}

#[async_trait]
impl ToolHandler for MemoryRecallTool {
    fn name(&self) -> &'static str {
        tool_names::TOOL_MEMORY_RECALL
    }

    fn description(&self) -> &'static str {
        "Recall useful experiment results, decisions, validations, or diagnostics from recent top-level sessions before repeating investigation or retrying a failed approach."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "query": {"type": ["string", "null"]},
                "paths": {"type": ["array", "null"], "items": {"type": "string"}},
                "kinds": {
                    "type": ["array", "null"],
                    "items": {"type": "string", "enum": ["experiment_result", "decision", "validation", "diagnostic"]}
                },
                "statuses": {
                    "type": ["array", "null"],
                    "items": {"type": "string", "enum": ["active", "useful", "dead_end", "blocked"]}
                },
                "limit": {"type": ["integer", "null"], "minimum": 1, "maximum": 20}
            },
            "required": ["query", "paths", "kinds", "statuses", "limit"],
            "additionalProperties": false
        })
    }

    fn permission_class(&self) -> ToolPermissionClass {
        ToolPermissionClass::Read
    }

    async fn execute(&self, args: Value) -> Result<Value> {
        let query = memory::validate_memory_recall_query(&args)?;
        let memories = memory::recall_recent_memories(&query)?;
        Ok(json!({"memories": memories}))
    }
}

#[async_trait]
impl ToolHandler for ContextCheckpointTool {
    fn name(&self) -> &'static str {
        tool_names::TOOL_CONTEXT_CHECKPOINT
    }

    fn description(&self) -> &'static str {
        "Create a context-only checkpoint before risky exploration or alternative approaches so later work continues on a new branch. This does not revert, isolate, or roll back files in the workspace."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "label": {
                    "type": ["string", "null"],
                    "maxLength": MAX_CONTEXT_CHECKPOINT_LABEL_CHARS,
                    "description": "Optional short branch label, such as 'try parser fix'"
                },
                "reason": {
                    "type": "string",
                    "maxLength": MAX_CONTEXT_CHECKPOINT_REASON_CHARS,
                    "description": "Why a new context branch is needed"
                }
            },
            "required": ["label", "reason"],
            "additionalProperties": false
        })
    }

    async fn execute(&self, args: Value) -> Result<Value> {
        let payload = validate_context_checkpoint(&args)?;
        Ok(json!({
            "label": payload.label,
            "reason": payload.reason,
            "context_only": true,
            "filesystem_rolled_back": false,
            "message": "Created a context checkpoint request. After this tool call is recorded, the agent will continue on a new context branch. This only affects agent context; files were not reverted."
        }))
    }
}

#[async_trait]
impl ToolHandler for ContextReturnTool {
    fn name(&self) -> &'static str {
        tool_names::TOOL_CONTEXT_RETURN
    }

    fn description(&self) -> &'static str {
        "Return from the current context experiment to the parent context and carry back a concise conclusion. This restores agent context only and does not revert files in the workspace."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "outcome": {
                    "type": "string",
                    "enum": ["useful", "dead_end", "blocked"],
                    "description": "How the current context experiment ended"
                },
                "summary": {
                    "type": "string",
                    "maxLength": MAX_CONTEXT_RETURN_SUMMARY_CHARS,
                    "description": "Concise conclusion to carry back into the parent context"
                },
                "next_action": {
                    "type": ["string", "null"],
                    "maxLength": MAX_CONTEXT_RETURN_NEXT_ACTION_CHARS,
                    "description": "Optional recommended next action after returning to the parent context"
                }
            },
            "required": ["outcome", "summary", "next_action"],
            "additionalProperties": false
        })
    }

    async fn execute(&self, args: Value) -> Result<Value> {
        let payload = validate_context_return(&args)?;
        Ok(json!({
            "outcome": payload.outcome,
            "summary": payload.summary,
            "next_action": payload.next_action,
            "context_restored": true,
            "filesystem_rolled_back": false,
            "message": "Returned from the current context experiment to the parent context. Files were not reverted."
        }))
    }
}

#[async_trait]
impl ToolHandler for AgentExploreTool {
    fn name(&self) -> &'static str {
        "agent__explore"
    }

    fn description(&self) -> &'static str {
        "将限定范围的只读仓库调研任务委派给 explorer 子代理，并返回摘要。"
    }

    fn parameters(&self) -> Value {
        subagent_parameters_schema("交给 explorer 子代理执行的聚焦只读调研任务")
    }

    async fn execute(&self, args: Value) -> Result<Value> {
        validate_agent_explore(&args)?;
        Ok(args)
    }
}

fn validate_agent_explore(args: &Value) -> Result<()> {
    normalize_subagent_input("agent__explore", args)?;
    Ok(())
}

#[async_trait]
impl ToolHandler for AgentFixerTool {
    fn name(&self) -> &'static str {
        "agent__fixer"
    }

    fn description(&self) -> &'static str {
        "将限定范围的实现或修复任务委派给 fixer 子代理，并返回摘要。"
    }

    fn parameters(&self) -> Value {
        subagent_parameters_schema("交给 fixer 子代理执行的聚焦实现或修复任务")
    }

    async fn execute(&self, args: Value) -> Result<Value> {
        validate_agent_fixer(&args)?;
        Ok(args)
    }
}

fn validate_agent_fixer(args: &Value) -> Result<()> {
    normalize_subagent_input("agent__fixer", args)?;
    Ok(())
}

fn validate_workflow_todos(args: &Value) -> Result<()> {
    let Some(items) = args.get("items").and_then(Value::as_array) else {
        bail!("workflow__todos requires an items array");
    };

    if items.len() > MAX_WORKFLOW_TODOS {
        bail!(
            "workflow__todos accepts at most {MAX_WORKFLOW_TODOS} items, got {}",
            items.len()
        );
    }

    let mut seen_ids = BTreeSet::new();
    let mut in_progress_count = 0;

    for (index, item) in items.iter().enumerate() {
        let mut id_value = None;

        for field in ["id", "content"] {
            let Some(value) = item.get(field).and_then(Value::as_str) else {
                bail!("workflow__todos item {index} requires string field '{field}'");
            };
            let length = value.chars().count();
            if length > MAX_WORKFLOW_TODO_FIELD_CHARS {
                bail!(
                    "workflow__todos item {index} field '{field}' exceeds {MAX_WORKFLOW_TODO_FIELD_CHARS} characters"
                );
            }

            if value.trim().is_empty() {
                bail!(
                    "workflow__todos item {index} field '{field}' must not be empty or whitespace"
                );
            }

            if field == "id" {
                id_value = Some(value);
            }
        }

        let id = id_value.expect("id must be captured after validation");
        if !seen_ids.insert(id) {
            bail!("workflow__todos item {index} has duplicate id '{id}'");
        }

        if item.get("status").and_then(Value::as_str) == Some("in_progress") {
            in_progress_count += 1;
            if in_progress_count > 1 {
                bail!("workflow__todos allows at most one item with status 'in_progress'");
            }
        }

        match item.get("status").and_then(Value::as_str) {
            Some("pending" | "in_progress" | "blocked" | "completed" | "cancelled") => {}
            Some(status) => {
                bail!("workflow__todos item {index} has invalid status '{status}'");
            }
            None => bail!("workflow__todos item {index} requires string field 'status'"),
        }
    }

    Ok(())
}

fn validate_workflow_auto_continue(args: &Value) -> Result<()> {
    if args.get("enabled").and_then(Value::as_bool).is_none() {
        bail!("workflow__auto_continue requires boolean field 'enabled'");
    }

    let Some(max_continuations) = args.get("max_continuations") else {
        bail!("workflow__auto_continue requires field 'max_continuations' as integer or null");
    };

    if max_continuations.is_null() {
        return Ok(());
    }

    let Some(max_continuations) = max_continuations.as_u64() else {
        bail!("workflow__auto_continue field 'max_continuations' must be integer or null");
    };
    if max_continuations > MAX_WORKFLOW_AUTO_CONTINUATIONS {
        bail!(
            "workflow__auto_continue max_continuations must be <= {MAX_WORKFLOW_AUTO_CONTINUATIONS}, got {max_continuations}"
        );
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
struct ContextCheckpointPayload {
    label: Option<String>,
    reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
struct ContextReturnPayload {
    outcome: String,
    summary: String,
    next_action: Option<String>,
}

fn validate_context_checkpoint(args: &Value) -> Result<ContextCheckpointPayload> {
    let label = optional_trimmed_string(args, "label")?;
    if let Some(label) = &label
        && label.chars().count() > MAX_CONTEXT_CHECKPOINT_LABEL_CHARS
    {
        bail!(
            "context__checkpoint field 'label' exceeds {MAX_CONTEXT_CHECKPOINT_LABEL_CHARS} characters"
        );
    }

    let Some(reason) = args.get("reason") else {
        bail!("context__checkpoint requires string field 'reason'");
    };
    let Some(reason) = reason.as_str() else {
        bail!("context__checkpoint requires string field 'reason'");
    };
    let reason = reason.trim();
    if reason.is_empty() {
        bail!("context__checkpoint field 'reason' must not be empty or whitespace");
    }
    if reason.chars().count() > MAX_CONTEXT_CHECKPOINT_REASON_CHARS {
        bail!(
            "context__checkpoint field 'reason' exceeds {MAX_CONTEXT_CHECKPOINT_REASON_CHARS} characters"
        );
    }

    Ok(ContextCheckpointPayload {
        label,
        reason: reason.to_string(),
    })
}

fn validate_context_return(args: &Value) -> Result<ContextReturnPayload> {
    let Some(outcome) = args.get("outcome").and_then(Value::as_str) else {
        bail!("context__return requires string field 'outcome'");
    };
    if !matches!(outcome, "useful" | "dead_end" | "blocked") {
        bail!("context__return field 'outcome' must be one of: useful, dead_end, blocked");
    }

    let Some(summary) = args.get("summary").and_then(Value::as_str) else {
        bail!("context__return requires string field 'summary'");
    };
    let summary = summary.trim();
    if summary.is_empty() {
        bail!("context__return field 'summary' must not be empty or whitespace");
    }
    if summary.chars().count() > MAX_CONTEXT_RETURN_SUMMARY_CHARS {
        bail!(
            "context__return field 'summary' exceeds {MAX_CONTEXT_RETURN_SUMMARY_CHARS} characters"
        );
    }

    let next_action = optional_trimmed_string(args, "next_action")?;
    if let Some(next_action) = &next_action
        && next_action.chars().count() > MAX_CONTEXT_RETURN_NEXT_ACTION_CHARS
    {
        bail!(
            "context__return field 'next_action' exceeds {MAX_CONTEXT_RETURN_NEXT_ACTION_CHARS} characters"
        );
    }

    Ok(ContextReturnPayload {
        outcome: outcome.to_string(),
        summary: summary.to_string(),
        next_action,
    })
}

struct ListDirTool;

#[async_trait]
impl ToolHandler for ListDirTool {
    fn name(&self) -> &'static str {
        "fs__list"
    }

    fn description(&self) -> &'static str {
        "List files and directories under the current workspace."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path relative to the current workspace, e.g. '.' or 'src'"
                }
            },
            "required": ["path"],
            "additionalProperties": false
        })
    }

    async fn execute(&self, args: Value) -> Result<Value> {
        list_dir(args, ToolExecutionContext::default()).await
    }

    async fn execute_with_context(
        &self,
        args: Value,
        context: ToolExecutionContext,
    ) -> Result<Value> {
        list_dir(args, context).await
    }
}

struct ReadFileTool;

#[async_trait]
impl ToolHandler for ReadFileTool {
    fn name(&self) -> &'static str {
        "fs__read"
    }

    fn description(&self) -> &'static str {
        "Read a UTF-8 text file under the current workspace by 1-based line offset and line limit."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to the current workspace"
                },
                "offset": {
                    "type": "integer",
                    "description": "1-based line number to start reading from. Use 1 to read from the beginning."
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to read. Use 200 for a typical first read. Capped by the application."
                }
            },
            "required": ["path", "offset", "limit"],
            "additionalProperties": false
        })
    }

    async fn execute(&self, args: Value) -> Result<Value> {
        read_file(args, ToolExecutionContext::default()).await
    }

    async fn execute_with_context(
        &self,
        args: Value,
        context: ToolExecutionContext,
    ) -> Result<Value> {
        read_file(args, context).await
    }
}

struct WriteFileTool;

#[async_trait]
impl ToolHandler for WriteFileTool {
    fn name(&self) -> &'static str {
        "fs__write"
    }

    fn description(&self) -> &'static str {
        "Create or overwrite a UTF-8 text file under the current workspace."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to the current workspace"
                },
                "content": {
                    "type": "string",
                    "description": "Full file content to write"
                }
            },
            "required": ["path", "content"],
            "additionalProperties": false
        })
    }

    async fn execute(&self, args: Value) -> Result<Value> {
        write_file(args, ToolExecutionContext::default()).await
    }

    async fn execute_with_context(
        &self,
        args: Value,
        context: ToolExecutionContext,
    ) -> Result<Value> {
        write_file(args, context).await
    }
}

struct AppendFileTool;

#[async_trait]
impl ToolHandler for AppendFileTool {
    fn name(&self) -> &'static str {
        "fs__append"
    }

    fn description(&self) -> &'static str {
        "Append UTF-8 text to a file under the current workspace."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to the current workspace"
                },
                "content": {
                    "type": "string",
                    "description": "Text to append"
                }
            },
            "required": ["path", "content"],
            "additionalProperties": false
        })
    }

    async fn execute(&self, args: Value) -> Result<Value> {
        append_file(args, ToolExecutionContext::default()).await
    }

    async fn execute_with_context(
        &self,
        args: Value,
        context: ToolExecutionContext,
    ) -> Result<Value> {
        append_file(args, context).await
    }
}

struct RunCommandTool;

#[async_trait]
impl ToolHandler for RunCommandTool {
    fn name(&self) -> &'static str {
        "shell__exec"
    }

    fn description(&self) -> &'static str {
        "Run a shell command in the current workspace. Authorization is handled by the tool-level permission policy."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to run, e.g. cargo check or ls -la"
                }
            },
            "required": ["command"],
            "additionalProperties": false
        })
    }

    async fn execute(&self, args: Value) -> Result<Value> {
        run_command(args).await
    }

    async fn execute_streaming(
        &self,
        args: Value,
        _context: ToolExecutionContext,
        emit: ToolOutputEmitter<'_>,
    ) -> Result<Value> {
        let command = required_string(&args, "command")?;
        run_workspace_shell_command_streaming(command, COMMAND_TIMEOUT_SECS, emit).await
    }
}

struct MkdirTool;

#[async_trait]
impl ToolHandler for MkdirTool {
    fn name(&self) -> &'static str {
        "fs__mkdir"
    }

    fn description(&self) -> &'static str {
        "Create a directory under the current workspace, including missing parent directories."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path relative to the current workspace"
                }
            },
            "required": ["path"],
            "additionalProperties": false
        })
    }

    async fn execute(&self, args: Value) -> Result<Value> {
        mkdir(args, ToolExecutionContext::default()).await
    }

    async fn execute_with_context(
        &self,
        args: Value,
        context: ToolExecutionContext,
    ) -> Result<Value> {
        mkdir(args, context).await
    }
}

struct RgTool;

#[async_trait]
impl ToolHandler for RgTool {
    fn name(&self) -> &'static str {
        "search__rg"
    }

    fn description(&self) -> &'static str {
        "Search text in the current workspace using ripgrep. Returns structured matches."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Ripgrep search pattern"
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file path relative to the workspace. Defaults to ."
                },
                "include": {
                    "type": "string",
                    "description": "Optional glob include pattern, e.g. *.rs"
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether the search is case-sensitive. Defaults to false"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum matches to return. Defaults to 100, capped at 1000"
                }
            },
            "required": ["pattern", "path", "include", "case_sensitive", "max_results"],
            "additionalProperties": false
        })
    }

    async fn execute(&self, args: Value) -> Result<Value> {
        rg(args, ToolExecutionContext::default()).await
    }

    async fn execute_with_context(
        &self,
        args: Value,
        context: ToolExecutionContext,
    ) -> Result<Value> {
        rg(args, context).await
    }
}

struct GitStatusTool;

#[async_trait]
impl ToolHandler for GitStatusTool {
    fn name(&self) -> &'static str {
        "git__status"
    }

    fn description(&self) -> &'static str {
        "Show git working tree status for the current workspace."
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
        git_status().await
    }
}

struct GitDiffTool;

#[async_trait]
impl ToolHandler for GitDiffTool {
    fn name(&self) -> &'static str {
        "git__diff"
    }

    fn description(&self) -> &'static str {
        "Show git diff for the current workspace. Supports unstaged or staged diff and optional path."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "staged": {
                    "type": "boolean",
                    "description": "Show staged diff instead of unstaged diff. Defaults to false"
                },
                "path": {
                    "type": ["string", "null"],
                    "description": "Optional workspace-relative path to limit the diff"
                }
            },
            "required": ["staged", "path"],
            "additionalProperties": false
        })
    }

    async fn execute(&self, args: Value) -> Result<Value> {
        git_diff(args).await
    }
}

struct GitLogTool;

#[async_trait]
impl ToolHandler for GitLogTool {
    fn name(&self) -> &'static str {
        "git__log"
    }

    fn description(&self) -> &'static str {
        "Show recent git commits for the current workspace."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "max_count": {
                    "type": "integer",
                    "description": "Maximum number of commits. Defaults to 10, capped at 50"
                }
            },
            "required": ["max_count"],
            "additionalProperties": false
        })
    }

    async fn execute(&self, args: Value) -> Result<Value> {
        git_log(args).await
    }
}

struct ApplyPatchTool;

#[async_trait]
impl ToolHandler for ApplyPatchTool {
    fn name(&self) -> &'static str {
        "edit__apply_patch"
    }

    fn description(&self) -> &'static str {
        "Apply exact-match text replacements to existing UTF-8 files under the workspace. Each edit must provide the exact old text in `find` and replacement text in `replace`. By default use replace_all=false so the tool fails unless `find` matches exactly once. This is intended for precise, low-ambiguity code edits."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "edits": {
                    "type": "array",
                    "description": "Exact-match replacement edits to apply atomically",
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Existing UTF-8 file path relative to the current workspace"
                            },
                            "find": {
                                "type": "string",
                                "description": "Exact old text to replace. Include enough surrounding context to make it unique. Must not be empty"
                            },
                            "replace": {
                                "type": "string",
                                "description": "New text that replaces `find`"
                            },
                            "replace_all": {
                                "type": "boolean",
                                "description": "If false, fail unless `find` occurs exactly once. If true, replace every occurrence but still fail if there are zero matches"
                            }
                        },
                        "required": ["path", "find", "replace", "replace_all"],
                        "additionalProperties": false
                    }
                }
            },
            "required": ["edits"],
            "additionalProperties": false
        })
    }

    async fn execute(&self, args: Value) -> Result<Value> {
        apply_patch(args, ToolExecutionContext::default()).await
    }

    async fn execute_with_context(
        &self,
        args: Value,
        context: ToolExecutionContext,
    ) -> Result<Value> {
        apply_patch(args, context).await
    }
}

struct AstSearchTool;

#[async_trait]
impl ToolHandler for AstSearchTool {
    fn name(&self) -> &'static str {
        "code__ast_search"
    }

    fn description(&self) -> &'static str {
        "Search code with a language-agnostic AST-aware pattern using the configured code analysis backend. Currently uses ast-grep CLI when available. Patterns are code, not regex, and can use metavariables like $A or $$$ARGS. This tool does not modify files."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Workspace-relative file or directory path to search, e.g. src or ."
                },
                "language": {
                    "type": "string",
                    "description": "Language name/alias for ast-grep, e.g. rust, typescript, python, go; use auto to infer from file extensions"
                },
                "pattern": {
                    "type": "string",
                    "description": "AST pattern written as valid code, e.g. self.tools.call($NAME, $ARGS).await"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum matches to return, capped at 1000"
                }
            },
            "required": ["path", "language", "pattern", "max_results"],
            "additionalProperties": false
        })
    }

    async fn execute(&self, args: Value) -> Result<Value> {
        self.execute_with_context(args, ToolExecutionContext::default())
            .await
    }

    async fn execute_with_context(
        &self,
        args: Value,
        context: ToolExecutionContext,
    ) -> Result<Value> {
        CodeAnalysisRegistry::default_backends()
            .ast_search(AstSearchRequest {
                path: required_string(&args, "path")?.to_string(),
                language: Some(required_string(&args, "language")?.to_string()),
                pattern: required_string(&args, "pattern")?.to_string(),
                max_results: optional_usize(&args, "max_results").unwrap_or(100),
                allow_outside_workspace: context.allow_outside_workspace,
            })
            .await
    }
}

struct AstReplacePreviewTool;

#[async_trait]
impl ToolHandler for AstReplacePreviewTool {
    fn name(&self) -> &'static str {
        "code__ast_replace_preview"
    }

    fn description(&self) -> &'static str {
        "Preview an AST-aware rewrite with the configured code analysis backend. This returns a diff preview only and does not write files. Use edit__apply_patch for audited edits."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Workspace-relative file or directory path to preview rewrites in"
                },
                "language": {
                    "type": "string",
                    "description": "Language name/alias for ast-grep, or auto to infer from file extensions"
                },
                "pattern": {
                    "type": "string",
                    "description": "AST pattern written as valid code, e.g. console.log($MSG)"
                },
                "rewrite": {
                    "type": "string",
                    "description": "Rewrite pattern, e.g. logger.info($MSG)"
                }
            },
            "required": ["path", "language", "pattern", "rewrite"],
            "additionalProperties": false
        })
    }

    async fn execute(&self, args: Value) -> Result<Value> {
        self.execute_with_context(args, ToolExecutionContext::default())
            .await
    }

    async fn execute_with_context(
        &self,
        args: Value,
        context: ToolExecutionContext,
    ) -> Result<Value> {
        CodeAnalysisRegistry::default_backends()
            .ast_replace_preview(AstReplacePreviewRequest {
                path: required_string(&args, "path")?.to_string(),
                language: Some(required_string(&args, "language")?.to_string()),
                pattern: required_string(&args, "pattern")?.to_string(),
                rewrite: required_string(&args, "rewrite")?.to_string(),
                allow_outside_workspace: context.allow_outside_workspace,
            })
            .await
    }
}

async fn list_dir(args: Value, context: ToolExecutionContext) -> Result<Value> {
    let path = existing_workspace_path(required_string(&args, "path")?, &context)?;
    let mut entries = fs::read_dir(&path)
        .await
        .with_context(|| format!("failed to read directory {}", path.display()))?;

    let mut result = Vec::new();
    while let Some(entry) = entries.next_entry().await? {
        let metadata = entry.metadata().await?;
        let kind = if metadata.is_dir() {
            "dir"
        } else if metadata.is_file() {
            "file"
        } else if metadata.is_symlink() {
            "symlink"
        } else {
            "other"
        };

        result.push(json!({
            "name": entry.file_name().to_string_lossy(),
            "path": display_workspace_relative(&entry.path())?,
            "kind": kind,
            "bytes": metadata.len(),
        }));
    }

    Ok(json!({ "entries": result }))
}

async fn read_file(args: Value, context: ToolExecutionContext) -> Result<Value> {
    let path = existing_workspace_path(required_string(&args, "path")?, &context)?;
    let metadata = fs::metadata(&path).await?;
    if !metadata.is_file() {
        bail!("path is not a file: {}", path.display());
    }

    let offset = optional_usize(&args, "offset").unwrap_or(1);
    if offset == 0 {
        bail!("offset must be >= 1");
    }

    let limit = optional_usize(&args, "limit")
        .unwrap_or(DEFAULT_READ_LINE_LIMIT)
        .clamp(1, MAX_READ_LINE_LIMIT);

    let file = fs::File::open(&path)
        .await
        .with_context(|| format!("failed to open file {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    let mut line_number = 0usize;
    let mut lines_read = 0usize;
    let mut content = String::new();
    let mut content_bytes = 0usize;
    let mut has_more = false;
    let mut byte_truncated = false;

    while let Some(line) = lines
        .next_line()
        .await
        .with_context(|| format!("failed to read UTF-8 line from {}", path.display()))?
    {
        line_number += 1;

        if line_number < offset {
            continue;
        }

        if lines_read >= limit {
            has_more = true;
            break;
        }

        let line_with_newline = format!("{line}\n");
        let line_bytes = line_with_newline.len();
        if content_bytes + line_bytes > MAX_READ_BYTES {
            if lines_read == 0 {
                bail!(
                    "line {} exceeds max read bytes ({}) in {}",
                    line_number,
                    MAX_READ_BYTES,
                    path.display()
                );
            }
            byte_truncated = true;
            has_more = true;
            break;
        }

        content.push_str(&line_with_newline);
        content_bytes += line_bytes;
        lines_read += 1;
    }

    let end_line = if lines_read == 0 {
        Value::Null
    } else {
        json!(offset + lines_read - 1)
    };
    let next_offset = if has_more {
        json!(offset + lines_read)
    } else {
        Value::Null
    };

    Ok(json!({
        "path": display_workspace_relative(&path)?,
        "content": content,
        "offset": offset,
        "limit": limit,
        "start_line": offset,
        "end_line": end_line,
        "lines_read": lines_read,
        "next_offset": next_offset,
        "has_more": has_more,
        "truncated": has_more || byte_truncated,
        "content_bytes": content_bytes,
        "total_bytes": metadata.len(),
    }))
}

async fn write_file(args: Value, context: ToolExecutionContext) -> Result<Value> {
    let raw_path = required_string(&args, "path")?;
    let content = required_string(&args, "content")?;
    let path = writable_workspace_path(raw_path, &context)?;

    fs::write(&path, content)
        .await
        .with_context(|| format!("failed to write file {}", path.display()))?;

    Ok(json!({
        "path": display_workspace_relative(&path)?,
        "bytes_written": content.len(),
    }))
}

async fn append_file(args: Value, context: ToolExecutionContext) -> Result<Value> {
    let raw_path = required_string(&args, "path")?;
    let content = required_string(&args, "content")?;
    let path = writable_workspace_path(raw_path, &context)?;

    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .await
        .with_context(|| format!("failed to open file {}", path.display()))?;
    file.write_all(content.as_bytes()).await?;

    Ok(json!({
        "path": display_workspace_relative(&path)?,
        "bytes_appended": content.len(),
    }))
}

async fn run_command(args: Value) -> Result<Value> {
    let command = required_string(&args, "command")?;
    run_workspace_shell_command(command, COMMAND_TIMEOUT_SECS).await
}

async fn mkdir(args: Value, context: ToolExecutionContext) -> Result<Value> {
    let raw_path = required_string(&args, "path")?;
    let path = new_workspace_path(raw_path, &context)?;

    fs::create_dir_all(&path)
        .await
        .with_context(|| format!("failed to create directory {}", path.display()))?;

    Ok(json!({
        "path": display_workspace_relative(&path)?,
        "created": true,
    }))
}

async fn rg(args: Value, context: ToolExecutionContext) -> Result<Value> {
    let pattern = required_string(&args, "pattern")?;
    let raw_path = optional_string(&args, "path").unwrap_or(".");
    let path = existing_workspace_path(raw_path, &context)?;
    let include = optional_string(&args, "include");
    let case_sensitive = optional_bool(&args, "case_sensitive").unwrap_or(false);
    let max_results = optional_usize(&args, "max_results")
        .unwrap_or(100)
        .clamp(1, 1000);

    let mut command_args = vec![
        "--line-number".to_string(),
        "--column".to_string(),
        "--no-heading".to_string(),
        "--color".to_string(),
        "never".to_string(),
    ];

    if !case_sensitive {
        command_args.push("--ignore-case".to_string());
    }

    if let Some(include) = include {
        command_args.push("--glob".to_string());
        command_args.push(include.to_string());
    }

    let mut target_path = display_workspace_relative(&path)?;
    if target_path.is_empty() {
        target_path = ".".to_string();
    }

    command_args.push(pattern.to_string());
    command_args.push(target_path);

    let output = run_workspace_command("rg", &command_args, COMMAND_TIMEOUT_SECS).await?;
    let stdout = output
        .get("stdout")
        .and_then(Value::as_str)
        .unwrap_or_default();

    let mut matches = Vec::new();
    let mut truncated = output
        .get("stdout_truncated")
        .and_then(Value::as_bool)
        .unwrap_or(false);

    for line in stdout.lines() {
        if matches.len() >= max_results {
            truncated = true;
            break;
        }

        let mut parts = line.splitn(4, ':');
        let Some(path) = parts.next() else { continue };
        let Some(line_number) = parts.next().and_then(|value| value.parse::<u64>().ok()) else {
            continue;
        };
        let Some(column) = parts.next().and_then(|value| value.parse::<u64>().ok()) else {
            continue;
        };
        let Some(text) = parts.next() else { continue };

        matches.push(json!({
            "path": path,
            "line": line_number,
            "column": column,
            "text": text,
        }));
    }

    Ok(json!({
        "pattern": pattern,
        "path": display_workspace_relative(&path)?,
        "matches": matches,
        "truncated": truncated,
        "status": output.get("status").cloned().unwrap_or(Value::Null),
        "success": output.get("success").cloned().unwrap_or(Value::Bool(false)),
        "stderr": output.get("stderr").cloned().unwrap_or(Value::String(String::new())),
    }))
}

async fn git_status() -> Result<Value> {
    let args = vec![
        "status".to_string(),
        "--short".to_string(),
        "--branch".to_string(),
    ];
    run_workspace_command("git", &args, COMMAND_TIMEOUT_SECS).await
}

async fn git_diff(args: Value) -> Result<Value> {
    let staged = optional_bool(&args, "staged").unwrap_or(false);
    let mut command_args = vec!["diff".to_string()];

    if staged {
        command_args.push("--cached".to_string());
    }

    if let Some(path) = optional_string(&args, "path").filter(|path| !path.trim().is_empty()) {
        command_args.push("--".to_string());
        command_args.push(safe_relative_path_arg(path)?);
    }

    run_workspace_command("git", &command_args, COMMAND_TIMEOUT_SECS).await
}

async fn git_log(args: Value) -> Result<Value> {
    let max_count = optional_usize(&args, "max_count")
        .unwrap_or(10)
        .clamp(1, 50);
    let command_args = vec![
        "log".to_string(),
        format!("--max-count={max_count}"),
        "--oneline".to_string(),
        "--decorate".to_string(),
    ];

    run_workspace_command("git", &command_args, COMMAND_TIMEOUT_SECS).await
}

async fn apply_patch(args: Value, context: ToolExecutionContext) -> Result<Value> {
    let edits = args
        .get("edits")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("missing or invalid array argument: edits"))?;

    if edits.is_empty() {
        bail!("edits cannot be empty");
    }

    let mut contents: BTreeMap<PathBuf, String> = BTreeMap::new();
    let mut results = Vec::with_capacity(edits.len());

    for (index, edit) in edits.iter().enumerate() {
        let path = existing_workspace_path(required_string(edit, "path")?, &context)?;
        let find = required_string(edit, "find")?;
        let replace = required_string(edit, "replace")?;
        let replace_all = edit
            .get("replace_all")
            .and_then(Value::as_bool)
            .ok_or_else(|| {
                anyhow!("missing or invalid boolean argument: edits[{index}].replace_all")
            })?;

        if find.is_empty() {
            bail!("edits[{index}].find cannot be empty");
        }

        let metadata = fs::metadata(&path).await?;
        if !metadata.is_file() {
            bail!("edits[{index}].path is not a file: {}", path.display());
        }

        if !contents.contains_key(&path) {
            let content = fs::read_to_string(&path)
                .await
                .with_context(|| format!("failed to read UTF-8 file {}", path.display()))?;
            contents.insert(path.clone(), content);
        }

        let content = contents
            .get_mut(&path)
            .ok_or_else(|| anyhow!("internal error: missing staged content"))?;
        let matches = content.matches(find).count();

        if matches == 0 {
            bail!(
                "edits[{index}] did not match any text in {}",
                display_workspace_relative(&path)?
            );
        }

        if !replace_all && matches != 1 {
            bail!(
                "edits[{index}] matched {matches} occurrences in {}; provide more context or set replace_all=true",
                display_workspace_relative(&path)?
            );
        }

        if replace_all {
            *content = content.replace(find, replace);
        } else {
            *content = content.replacen(find, replace, 1);
        }

        results.push(json!({
            "path": display_workspace_relative(&path)?,
            "replacements": matches,
            "replace_all": replace_all,
        }));
    }

    for (path, content) in &contents {
        fs::write(path, content)
            .await
            .with_context(|| format!("failed to write patched file {}", path.display()))?;
    }

    Ok(json!({
        "files_changed": contents.len(),
        "edits_applied": edits.len(),
        "edits": results,
    }))
}

async fn run_workspace_command(command: &str, args: &[String], timeout_secs: u64) -> Result<Value> {
    let root = workspace_root()?;
    debug!(command = %command, args = ?args, "running workspace command");

    let output = match timeout(
        Duration::from_secs(timeout_secs),
        Command::new(command)
            .args(args)
            .current_dir(root)
            .stdin(Stdio::null())
            .output(),
    )
    .await
    {
        Ok(output) => output?,
        Err(_) => {
            return Ok(json!({
                "error": format!("command timed out after {timeout_secs}s")
            }));
        }
    };

    let stdout = truncate_utf8(
        &String::from_utf8_lossy(&output.stdout),
        MAX_COMMAND_OUTPUT_BYTES,
    );
    let stderr = truncate_utf8(
        &String::from_utf8_lossy(&output.stderr),
        MAX_COMMAND_OUTPUT_BYTES,
    );

    Ok(json!({
        "status": output.status.code(),
        "success": output.status.success(),
        "stdout": stdout.text,
        "stdout_truncated": stdout.truncated,
        "stderr": stderr.text,
        "stderr_truncated": stderr.truncated,
    }))
}

async fn run_workspace_shell_command(command: &str, timeout_secs: u64) -> Result<Value> {
    let root = workspace_root()?;
    let (shell, shell_flag) = shell_invocation();
    debug!(command = %command, shell = %shell, "running workspace shell command");

    let output = match timeout(
        Duration::from_secs(timeout_secs),
        Command::new(shell)
            .arg(shell_flag)
            .arg(command)
            .current_dir(root)
            .stdin(Stdio::null())
            .output(),
    )
    .await
    {
        Ok(output) => output?,
        Err(_) => {
            return Ok(json!({
                "command": command,
                "error": format!("command timed out after {timeout_secs}s")
            }));
        }
    };

    let stdout = truncate_utf8(
        &String::from_utf8_lossy(&output.stdout),
        MAX_COMMAND_OUTPUT_BYTES,
    );
    let stderr = truncate_utf8(
        &String::from_utf8_lossy(&output.stderr),
        MAX_COMMAND_OUTPUT_BYTES,
    );

    Ok(json!({
        "command": command,
        "shell": shell,
        "status": output.status.code(),
        "success": output.status.success(),
        "stdout": stdout.text,
        "stdout_truncated": stdout.truncated,
        "stderr": stderr.text,
        "stderr_truncated": stderr.truncated,
    }))
}

async fn run_workspace_shell_command_streaming(
    command: &str,
    timeout_secs: u64,
    emit: ToolOutputEmitter<'_>,
) -> Result<Value> {
    let root = workspace_root()?;
    let (shell, shell_flag) = shell_invocation();
    debug!(command = %command, shell = %shell, "running streaming workspace shell command");

    let mut child = Command::new(shell)
        .arg(shell_flag)
        .arg(command)
        .current_dir(root)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("failed to spawn shell command: {command}"))?;

    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| anyhow!("failed to capture command stdout"))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| anyhow!("failed to capture command stderr"))?;

    let (tx, mut rx) = mpsc::unbounded_channel();
    tokio::spawn(read_command_stream(
        ToolOutputStream::Stdout,
        stdout,
        tx.clone(),
    ));
    tokio::spawn(read_command_stream(ToolOutputStream::Stderr, stderr, tx));

    let mut stdout = StreamAccumulator::new();
    let mut stderr = StreamAccumulator::new();
    let mut wait = Box::pin(child.wait());
    let timeout_sleep = sleep(Duration::from_secs(timeout_secs));
    tokio::pin!(timeout_sleep);
    let mut timed_out = false;
    let mut status = None;

    loop {
        tokio::select! {
            Some((stream, chunk)) = rx.recv() => {
                match stream {
                    ToolOutputStream::Stdout => stdout.push(&chunk),
                    ToolOutputStream::Stderr => stderr.push(&chunk),
                }
                emit(stream, chunk)?;
            }
            result = &mut wait => {
                status = Some(result?);
                break;
            }
            _ = &mut timeout_sleep => {
                timed_out = true;
                break;
            }
        }
    }
    drop(wait);

    let status = if timed_out {
        let _ = child.kill().await;
        child.wait().await?
    } else {
        status.ok_or_else(|| anyhow!("command exited without status"))?
    };

    while let Some((stream, chunk)) = rx.recv().await {
        match stream {
            ToolOutputStream::Stdout => stdout.push(&chunk),
            ToolOutputStream::Stderr => stderr.push(&chunk),
        }
        emit(stream, chunk)?;
    }

    let mut data = json!({
        "command": command,
        "shell": shell,
        "status": status.code(),
        "success": status.success() && !timed_out,
        "stdout": stdout.text,
        "stdout_truncated": stdout.truncated,
        "stderr": stderr.text,
        "stderr_truncated": stderr.truncated,
    });

    if timed_out {
        data["error"] = Value::String(format!("command timed out after {timeout_secs}s"));
    }

    Ok(data)
}

async fn read_command_stream<R>(
    stream: ToolOutputStream,
    mut reader: R,
    tx: mpsc::UnboundedSender<(ToolOutputStream, String)>,
) where
    R: AsyncRead + Unpin + Send + 'static,
{
    let mut buffer = [0u8; 8192];
    loop {
        match reader.read(&mut buffer).await {
            Ok(0) => break,
            Ok(read) => {
                let chunk = String::from_utf8_lossy(&buffer[..read]).to_string();
                if tx.send((stream, chunk)).is_err() {
                    break;
                }
            }
            Err(error) => {
                warn!(stream = stream.as_str(), error = %error, "failed to read command output stream");
                break;
            }
        }
    }
}

struct StreamAccumulator {
    text: String,
    truncated: bool,
}

impl StreamAccumulator {
    fn new() -> Self {
        Self {
            text: String::new(),
            truncated: false,
        }
    }

    fn push(&mut self, chunk: &str) {
        if self.text.len() >= MAX_COMMAND_OUTPUT_BYTES {
            self.truncated = true;
            return;
        }
        self.text.push_str(chunk);
        if self.text.len() > MAX_COMMAND_OUTPUT_BYTES {
            self.truncated = true;
            self.text.truncate(MAX_COMMAND_OUTPUT_BYTES);
            while !self.text.is_char_boundary(self.text.len()) {
                self.text.pop();
            }
        }
    }
}

fn shell_invocation() -> (&'static str, &'static str) {
    if cfg!(windows) {
        ("cmd", "/C")
    } else {
        ("/bin/sh", "-c")
    }
}

fn required_string<'a>(args: &'a Value, key: &str) -> Result<&'a str> {
    args.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("missing or invalid string argument: {key}"))
}

fn optional_string<'a>(args: &'a Value, key: &str) -> Option<&'a str> {
    args.get(key).and_then(Value::as_str)
}

fn optional_bool(args: &Value, key: &str) -> Option<bool> {
    args.get(key).and_then(Value::as_bool)
}

fn optional_usize(args: &Value, key: &str) -> Option<usize> {
    args.get(key)
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
}

fn workspace_root() -> Result<PathBuf> {
    std::env::current_dir()?
        .canonicalize()
        .context("failed to canonicalize current workspace")
}

fn outside_existing_workspace_path(path: &str) -> Option<String> {
    let root = workspace_root().ok()?;
    let candidate = join_workspace_path(&root, path);

    if let Ok(canonical) = candidate.canonicalize() {
        return outside_workspace_label(&root, &canonical);
    }

    syntactic_outside_workspace_label(&root, path, &candidate)
}

fn outside_writable_workspace_path(path: &str) -> Option<String> {
    let root = workspace_root().ok()?;
    let candidate = join_workspace_path(&root, path);

    if let Some(parent) = candidate.parent()
        && let Ok(canonical_parent) = parent.canonicalize()
        && let Some(label) = outside_workspace_label(&root, &canonical_parent)
    {
        return Some(label);
    }

    syntactic_outside_workspace_label(&root, path, &candidate)
}

fn outside_new_workspace_path(path: &str) -> Option<String> {
    let root = workspace_root().ok()?;
    let candidate = join_workspace_path(&root, path);

    if let Some(canonical_ancestor) = canonical_existing_ancestor(&candidate)
        && let Some(label) = outside_workspace_label(&root, &canonical_ancestor)
    {
        return Some(label);
    }

    syntactic_outside_workspace_label(&root, path, &candidate)
}

fn canonical_existing_ancestor(path: &Path) -> Option<PathBuf> {
    let mut current = Some(path);
    while let Some(candidate) = current {
        if let Ok(canonical) = candidate.canonicalize() {
            return Some(canonical);
        }
        current = candidate.parent();
    }
    None
}

fn outside_workspace_label(root: &Path, path: &Path) -> Option<String> {
    (!path.starts_with(root)).then(|| path.display().to_string())
}

fn syntactic_outside_workspace_label(
    root: &Path,
    raw_path: &str,
    candidate: &Path,
) -> Option<String> {
    let raw_path = Path::new(raw_path);
    if raw_path.is_absolute() {
        return outside_workspace_label(root, raw_path);
    }

    if relative_path_escapes_workspace(raw_path) {
        return Some(candidate.display().to_string());
    }

    None
}

fn relative_path_escapes_workspace(path: &Path) -> bool {
    let mut depth = 0usize;
    for component in path.components() {
        match component {
            Component::Normal(_) => depth = depth.saturating_add(1),
            Component::CurDir => {}
            Component::ParentDir => {
                if depth == 0 {
                    return true;
                }
                depth -= 1;
            }
            Component::RootDir | Component::Prefix(_) => return true,
        }
    }
    false
}

fn existing_workspace_path(path: &str, context: &ToolExecutionContext) -> Result<PathBuf> {
    let root = workspace_root()?;
    let candidate = join_workspace_path(&root, path);
    let canonical = candidate
        .canonicalize()
        .with_context(|| format!("path does not exist: {}", candidate.display()))?;
    if !context.allow_outside_workspace {
        ensure_inside_workspace(&root, &canonical)?;
    }
    Ok(canonical)
}

fn writable_workspace_path(path: &str, context: &ToolExecutionContext) -> Result<PathBuf> {
    let root = workspace_root()?;
    let candidate = join_workspace_path(&root, path);
    let parent = candidate
        .parent()
        .ok_or_else(|| anyhow!("path has no parent: {}", candidate.display()))?
        .canonicalize()
        .with_context(|| format!("parent directory does not exist: {}", candidate.display()))?;

    if !context.allow_outside_workspace {
        ensure_inside_workspace(&root, &parent)?;
    }
    Ok(parent.join(
        candidate
            .file_name()
            .ok_or_else(|| anyhow!("path has no file name: {}", candidate.display()))?,
    ))
}

fn new_workspace_path(path: &str, context: &ToolExecutionContext) -> Result<PathBuf> {
    let root = workspace_root()?;
    if context.allow_outside_workspace {
        if Path::new(path).as_os_str().is_empty() {
            bail!("path cannot be empty");
        }
        let candidate = join_workspace_path(&root, path);
        return Ok(candidate);
    }
    let relative = safe_relative_path_arg(path)?;
    let candidate = root.join(relative);
    if let Some(canonical_ancestor) = canonical_existing_ancestor(&candidate) {
        ensure_inside_workspace(&root, &canonical_ancestor)?;
    }
    Ok(candidate)
}

fn safe_relative_path_arg(path: &str) -> Result<String> {
    let path = Path::new(path);
    if path.is_absolute() {
        bail!("absolute paths are not allowed here: {}", path.display());
    }

    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Normal(part) => normalized.push(part),
            Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                bail!("parent traversal is not allowed: {}", path.display());
            }
        }
    }

    if normalized.as_os_str().is_empty() {
        bail!("path cannot be empty");
    }

    Ok(normalized.to_string_lossy().to_string())
}

fn join_workspace_path(root: &Path, path: &str) -> PathBuf {
    let path = Path::new(path);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.join(path)
    }
}

fn ensure_inside_workspace(root: &Path, path: &Path) -> Result<()> {
    if !path.starts_with(root) {
        bail!(
            "path is outside workspace: {} (workspace: {})",
            path.display(),
            root.display()
        );
    }
    Ok(())
}

fn display_workspace_relative(path: &Path) -> Result<String> {
    let root = workspace_root()?;
    let absolute = if path.exists() {
        path.canonicalize()?
    } else {
        path.to_path_buf()
    };

    Ok(absolute
        .strip_prefix(root)
        .unwrap_or(&absolute)
        .to_string_lossy()
        .to_string())
}

struct TruncatedText {
    text: String,
    truncated: bool,
}

fn truncate_utf8(text: &str, max_bytes: usize) -> TruncatedText {
    if text.len() <= max_bytes {
        return TruncatedText {
            text: text.to_string(),
            truncated: false,
        };
    }

    let mut end = max_bytes;
    while !text.is_char_boundary(end) {
        end -= 1;
    }

    TruncatedText {
        text: text[..end].to_string(),
        truncated: true,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        MAX_WORKFLOW_AUTO_CONTINUATIONS, MAX_WORKFLOW_TODOS, ToolExecutionContext, ToolRegistry,
        external_workspace_access_for_tool, normalize_subagent_input,
    };
    use crate::permission::ToolScope;
    use crate::skills::{SkillEntry, SkillRegistry, SkillTool};
    use crate::tool::ToolOutputStream;
    use serde_json::json;
    use std::path::PathBuf;
    use std::sync::{Arc, Mutex};

    async fn call_workflow_todos(items: serde_json::Value) -> crate::tool::ToolResult {
        ToolRegistry::default_tools()
            .call("workflow__todos", json!({"items": items}))
            .await
    }

    #[tokio::test]
    async fn shell_exec_streams_output_deltas_and_returns_final_output() {
        let registry = ToolRegistry::default_tools();
        let chunks = Arc::new(Mutex::new(Vec::<(ToolOutputStream, String)>::new()));
        let captured = Arc::clone(&chunks);
        let mut emit = move |stream, chunk| {
            captured.lock().expect("capture lock").push((stream, chunk));
            Ok(())
        };

        let output = registry
            .call_streaming(
                "shell__exec",
                json!({"command":"printf 'out\\n'; printf 'err\\n' >&2"}),
                ToolExecutionContext::default(),
                &mut emit,
            )
            .await;

        assert!(output.ok, "{:?}", output.error);
        let data = output.data.expect("shell output data");
        assert_eq!(data["stdout"], json!("out\n"));
        assert_eq!(data["stderr"], json!("err\n"));
        let chunks = chunks.lock().expect("capture lock");
        assert!(
            chunks
                .iter()
                .any(|(stream, chunk)| *stream == ToolOutputStream::Stdout && chunk.contains("out")),
            "{chunks:?}"
        );
        assert!(
            chunks
                .iter()
                .any(|(stream, chunk)| *stream == ToolOutputStream::Stderr && chunk.contains("err")),
            "{chunks:?}"
        );
    }

    #[tokio::test]
    async fn external_workspace_read_requires_explicit_grant() {
        let outside_path = std::env::temp_dir().join(format!(
            "letcode-outside-tool-read-{}.txt",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system time")
                .as_nanos()
        ));
        std::fs::write(&outside_path, "outside\n").expect("write outside fixture");
        let args = json!({
            "path": outside_path.to_string_lossy(),
            "offset": 1,
            "limit": 10,
        });

        let access = external_workspace_access_for_tool("fs__read", &args)
            .expect("outside path should require approval");
        assert_eq!(access.paths.len(), 1);

        let registry = ToolRegistry::default_tools();
        let denied = registry.call("fs__read", args.clone()).await;
        assert!(!denied.ok);
        assert!(
            denied
                .error
                .as_ref()
                .is_some_and(|error| error.message.contains("outside workspace")),
            "{:?}",
            denied.error
        );

        let allowed = registry
            .call_with_context(
                "fs__read",
                args,
                ToolExecutionContext::outside_workspace_granted(),
            )
            .await;
        assert!(allowed.ok, "{:?}", allowed.error);
        assert!(
            allowed
                .data
                .as_ref()
                .and_then(|data| data.get("content"))
                .and_then(serde_json::Value::as_str)
                .is_some_and(|content| content.contains("outside"))
        );

        let _ = std::fs::remove_file(outside_path);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn external_workspace_mkdir_via_symlink_requires_explicit_grant() {
        use std::os::unix::fs::symlink;

        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time")
            .as_nanos();
        let outside_dir = std::env::temp_dir().join(format!("letcode-outside-mkdir-{unique}"));
        std::fs::create_dir_all(&outside_dir).expect("create outside fixture");
        std::fs::create_dir_all("target").expect("create target dir");
        let workspace_link = PathBuf::from("target").join(format!("letcode-outside-link-{unique}"));
        symlink(&outside_dir, &workspace_link).expect("create workspace symlink");
        let escaped_child = workspace_link.join("child");
        let args = json!({"path": escaped_child.to_string_lossy()});

        let access = external_workspace_access_for_tool("fs__mkdir", &args)
            .expect("symlink escape should require approval");
        assert_eq!(access.paths.len(), 1);

        let registry = ToolRegistry::default_tools();
        let denied = registry.call("fs__mkdir", args.clone()).await;
        assert!(!denied.ok);
        assert!(!outside_dir.join("child").exists());

        let allowed = registry
            .call_with_context(
                "fs__mkdir",
                args,
                ToolExecutionContext::outside_workspace_granted(),
            )
            .await;
        assert!(allowed.ok, "{:?}", allowed.error);
        assert!(outside_dir.join("child").is_dir());

        let _ = std::fs::remove_file(workspace_link);
        let _ = std::fs::remove_dir_all(outside_dir);
    }

    #[test]
    fn default_tools_use_namespaced_names_without_legacy_aliases() {
        let specs = ToolRegistry::default_tools().specs();
        let names = specs
            .iter()
            .map(|spec| spec.name.as_str())
            .collect::<Vec<_>>();

        for expected in [
            "util__echo",
            "workflow__todos",
            "workflow__auto_continue",
            "memory__recall",
            "context__checkpoint",
            "context__return",
            "context__list",
            "context__search",
            "context__open",
            "context__summarize",
            "context__pin",
            "context__archive",
            "context__remove",
            "agent__explore",
            "agent__fixer",
            "fs__list",
            "fs__read",
            "fs__write",
            "fs__append",
            "fs__mkdir",
            "shell__exec",
            "search__rg",
            "git__status",
            "git__diff",
            "git__log",
            "edit__apply_patch",
            "code__ast_search",
            "code__ast_replace_preview",
        ] {
            assert!(
                names.contains(&expected),
                "missing tool {expected}: {names:?}"
            );
        }

        for legacy in [
            "echo",
            "todos",
            "auto_continue",
            "explore",
            "list_dir",
            "read_file",
            "write_file",
            "append_file",
            "mkdir",
            "run_command",
            "rg",
            "git_status",
            "git_diff",
            "git_log",
            "apply_patch",
            "ast_search",
            "ast_replace_preview",
        ] {
            assert!(
                !names.contains(&legacy),
                "legacy alias is exposed: {legacy}"
            );
        }
    }

    #[test]
    fn git_diff_schema_allows_null_path_for_workspace_diff() {
        let specs = ToolRegistry::default_tools().specs();
        let git_diff = specs
            .iter()
            .find(|spec| spec.name == "git__diff")
            .expect("git diff tool is registered");

        assert_eq!(
            git_diff.parameters["properties"]["path"]["type"],
            serde_json::json!(["string", "null"])
        );
        assert_eq!(
            git_diff.parameters["required"],
            serde_json::json!(["staged", "path"])
        );
    }

    #[test]
    fn workflow_auto_continue_schema_is_strict_compatible() {
        let specs = ToolRegistry::default_tools().specs();
        let auto_continue = specs
            .iter()
            .find(|spec| spec.name == "workflow__auto_continue")
            .expect("workflow auto-continue tool is registered");

        assert_eq!(
            auto_continue.parameters["properties"]["max_continuations"]["type"],
            serde_json::json!(["integer", "null"])
        );
        assert_eq!(
            auto_continue.parameters["required"],
            serde_json::json!(["enabled", "max_continuations"])
        );
    }

    #[tokio::test]
    async fn context_checkpoint_accepts_valid_payload_and_marks_context_boundary() {
        let output = ToolRegistry::default_tools()
            .call(
                "context__checkpoint",
                json!({
                    "label": " try parser fix ",
                    "reason": " Need risky exploration without polluting current context "
                }),
            )
            .await;

        assert!(output.ok, "{output:?}");
        let data = output.data.expect("checkpoint data");
        assert_eq!(data["label"], json!("try parser fix"));
        assert_eq!(
            data["reason"],
            json!("Need risky exploration without polluting current context")
        );
        assert_eq!(data["context_only"], json!(true));
        assert_eq!(data["filesystem_rolled_back"], json!(false));
    }

    #[tokio::test]
    async fn context_checkpoint_rejects_empty_reason() {
        let output = ToolRegistry::default_tools()
            .call(
                "context__checkpoint",
                json!({
                    "label": "try parser fix",
                    "reason": "  \n\t  "
                }),
            )
            .await;

        assert!(!output.ok);
        assert!(
            output
                .error
                .as_ref()
                .expect("checkpoint error")
                .message
                .contains("must not be empty or whitespace")
        );
    }

    #[tokio::test]
    async fn context_return_accepts_valid_payload_and_rejects_blank_summary() {
        let output = ToolRegistry::default_tools()
            .call(
                "context__return",
                json!({
                    "outcome": "useful",
                    "summary": "  parser approach isolated the real issue  ",
                    "next_action": "apply the tokenizer fix on main"
                }),
            )
            .await;

        assert!(output.ok, "{output:?}");
        let data = output.data.expect("return data");
        assert_eq!(data["outcome"], json!("useful"));
        assert_eq!(
            data["summary"],
            json!("parser approach isolated the real issue")
        );
        assert_eq!(
            data["next_action"],
            json!("apply the tokenizer fix on main")
        );
        assert_eq!(data["filesystem_rolled_back"], json!(false));

        let invalid = ToolRegistry::default_tools()
            .call(
                "context__return",
                json!({
                    "outcome": "blocked",
                    "summary": "   ",
                    "next_action": null
                }),
            )
            .await;

        assert!(!invalid.ok);
        assert!(
            invalid
                .error
                .as_ref()
                .expect("return error")
                .message
                .contains("must not be empty or whitespace")
        );
    }

    #[tokio::test]
    async fn memory_recall_returns_empty_array_and_validates_limit() {
        let base_dir = std::env::temp_dir().join(format!(
            "letcode-memory-tool-empty-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("time ok")
                .as_nanos()
        ));
        std::fs::create_dir_all(&base_dir).expect("create empty memory dir");
        crate::memory::set_memory_sessions_dir(base_dir);
        let output = ToolRegistry::default_tools()
            .call(
                "memory__recall",
                json!({
                    "query": null,
                    "paths": null,
                    "kinds": null,
                    "statuses": null,
                    "limit": 1
                }),
            )
            .await;

        assert!(output.ok, "{output:?}");
        assert_eq!(output.data.expect("memory data")["memories"], json!([]));

        let invalid = ToolRegistry::default_tools()
            .call(
                "memory__recall",
                json!({
                    "query": null,
                    "paths": null,
                    "kinds": null,
                    "statuses": null,
                    "limit": 99
                }),
            )
            .await;
        assert!(!invalid.ok);
        assert!(
            invalid
                .error
                .as_ref()
                .expect("memory recall error")
                .message
                .contains("must be between 1 and 20")
        );
    }

    #[test]
    fn subagent_tool_schemas_expose_delegation_fields_without_runtime_budget_knobs() {
        let specs = ToolRegistry::default_tools().specs();
        let explore = specs
            .iter()
            .find(|spec| spec.name == "agent__explore")
            .expect("agent explore tool is registered");

        for field in [
            "task",
            "objective",
            "success_criteria",
            "allowed_paths",
            "forbidden_paths",
            "owned_paths",
        ] {
            assert!(
                explore.parameters["properties"].get(field).is_some(),
                "missing field {field} in subagent schema"
            );
        }

        for field in ["timeout_secs", "max_tool_calls"] {
            assert!(
                explore.parameters["properties"].get(field).is_none(),
                "runtime budget field {field} should not be exposed in normal subagent schema"
            );
        }

        assert_eq!(
            explore.parameters["properties"]["objective"]["type"],
            json!(["string", "null"])
        );
        assert_eq!(
            explore.parameters["properties"]["allowed_paths"]["type"],
            json!(["array", "null"])
        );
        assert_eq!(
            explore.parameters["required"],
            json!([
                "task",
                "objective",
                "success_criteria",
                "allowed_paths",
                "forbidden_paths",
                "owned_paths"
            ])
        );
    }

    #[test]
    fn normalize_subagent_input_accepts_legacy_task_only_payload() {
        let input =
            normalize_subagent_input("agent__explore", &json!({"task": " inspect src/agent.rs "}))
                .expect("legacy task-only payload should normalize");

        assert_eq!(input.objective, "inspect src/agent.rs");
        assert!(input.success_criteria.is_empty());
        assert!(input.allowed_paths.is_empty());
        assert_eq!(
            input.render_for_delegate("agent__explore"),
            "Objective: inspect src/agent.rs\nDelegation contract: do not recursively delegate; stay within the provided scope and report findings or implementation outcome succinctly.\nMode: read-only exploration only."
        );
    }

    #[test]
    fn normalize_subagent_input_supports_bounded_payload() {
        let input = normalize_subagent_input(
            "agent__fixer",
            &json!({
                "objective": "Implement bounded delegation",
                "success_criteria": ["tests pass", "contract exposed"],
                "allowed_paths": ["src/agent.rs", "src/tool.rs"],
                "forbidden_paths": ["Cargo.toml"],
                "owned_paths": ["src/tool.rs"],
                "timeout_secs": 45,
                "max_tool_calls": 7
            }),
        )
        .expect("bounded payload should normalize");

        assert_eq!(input.objective, "Implement bounded delegation");
        assert_eq!(
            input.success_criteria,
            vec!["tests pass", "contract exposed"]
        );
        assert_eq!(input.allowed_paths, vec!["src/agent.rs", "src/tool.rs"]);
        assert_eq!(input.forbidden_paths, vec!["Cargo.toml"]);
        assert_eq!(input.owned_paths, vec!["src/tool.rs"]);
        assert_eq!(input.timeout_secs, Some(45));
        assert_eq!(input.max_tool_calls, Some(7));
    }

    #[test]
    fn normalize_subagent_input_rejects_missing_or_blank_objective() {
        let missing = normalize_subagent_input("agent__explore", &json!({}))
            .expect_err("missing objective should fail");
        assert!(
            missing
                .to_string()
                .contains("requires a non-empty 'task' or 'objective'")
        );

        let blank =
            normalize_subagent_input("agent__fixer", &json!({"task": "   ", "objective": null}))
                .expect_err("blank task should fail");
        assert!(
            blank
                .to_string()
                .contains("field 'task' must not be empty or whitespace")
        );
    }

    #[test]
    fn normalize_subagent_input_rejects_invalid_bounds_and_path_entries() {
        let invalid_path = normalize_subagent_input(
            "agent__fixer",
            &json!({"objective": "x", "allowed_paths": ["src", " "]}),
        )
        .expect_err("blank path should fail");
        assert!(invalid_path.to_string().contains("allowed_paths' item 1"));

        let invalid_budget = normalize_subagent_input(
            "agent__fixer",
            &json!({"objective": "x", "max_tool_calls": 0}),
        )
        .expect_err("zero max_tool_calls should fail");
        assert!(
            invalid_budget
                .to_string()
                .contains("field 'max_tool_calls' must be greater than 0")
        );
    }

    #[test]
    fn normalized_subagent_input_enforces_write_scope_paths() {
        let input = normalize_subagent_input(
            "agent__fixer",
            &json!({
                "objective": "fix",
                "allowed_paths": ["src"],
                "owned_paths": ["src/fixes"],
                "forbidden_paths": ["src/secrets"]
            }),
        )
        .expect("scope should normalize");

        assert!(input.has_write_scope());
        assert!(input.permits_write_path("src/fixes/mod.rs"));
        assert!(input.permits_write_path("src/lib.rs"));
        assert!(!input.permits_write_path("src/secrets/token.rs"));
        assert!(!input.permits_write_path("tests/outside.rs"));
    }

    #[test]
    fn default_tool_schemas_require_every_declared_property_for_strict_mode() {
        let specs = ToolRegistry::default_tools().specs();

        for spec in specs {
            let properties = spec.parameters["properties"]
                .as_object()
                .expect("tool properties must be an object")
                .keys()
                .cloned()
                .collect::<std::collections::BTreeSet<_>>();
            let required = spec.parameters["required"]
                .as_array()
                .expect("tool required must be an array")
                .iter()
                .map(|value| {
                    value
                        .as_str()
                        .expect("required field names must be strings")
                        .to_string()
                })
                .collect::<std::collections::BTreeSet<_>>();

            assert_eq!(
                required, properties,
                "strict schema for tool {} must require every declared property",
                spec.name
            );
        }
    }

    #[tokio::test]
    async fn workflow_control_tools_reject_unbounded_payloads() {
        let tools = ToolRegistry::default_tools();
        let too_many_items = (0..=MAX_WORKFLOW_TODOS)
            .map(|index| json!({"id": format!("t{index}"), "content": "x", "status": "pending"}))
            .collect::<Vec<_>>();

        let todo_output = tools
            .call("workflow__todos", json!({"items": too_many_items}))
            .await;
        assert!(!todo_output.ok);
        assert!(
            todo_output
                .error
                .as_ref()
                .expect("todo error")
                .message
                .contains(&format!("at most {MAX_WORKFLOW_TODOS} items"))
        );

        let auto_output = tools
            .call(
                "workflow__auto_continue",
                json!({"enabled": true, "max_continuations": MAX_WORKFLOW_AUTO_CONTINUATIONS + 1}),
            )
            .await;
        assert!(!auto_output.ok);
        assert!(
            auto_output
                .error
                .as_ref()
                .expect("auto-continue error")
                .message
                .contains(&format!("<= {MAX_WORKFLOW_AUTO_CONTINUATIONS}"))
        );
    }

    #[tokio::test]
    async fn workflow_todos_rejects_blank_id() {
        let output = call_workflow_todos(json!([
            {"id": "   ", "content": "valid", "status": "pending"}
        ]))
        .await;

        assert!(!output.ok);
        assert!(
            output
                .error
                .as_ref()
                .expect("todo error")
                .message
                .contains("field 'id' must not be empty or whitespace")
        );
    }

    #[tokio::test]
    async fn workflow_todos_rejects_blank_content() {
        let output = call_workflow_todos(json!([
            {"id": "todo-1", "content": " \n\t ", "status": "pending"}
        ]))
        .await;

        assert!(!output.ok);
        assert!(
            output
                .error
                .as_ref()
                .expect("todo error")
                .message
                .contains("field 'content' must not be empty or whitespace")
        );
    }

    #[tokio::test]
    async fn workflow_todos_rejects_duplicate_ids() {
        let output = call_workflow_todos(json!([
            {"id": "todo-1", "content": "first", "status": "pending"},
            {"id": "todo-1", "content": "second", "status": "completed"}
        ]))
        .await;

        assert!(!output.ok);
        assert!(
            output
                .error
                .as_ref()
                .expect("todo error")
                .message
                .contains("duplicate id 'todo-1'")
        );
    }

    #[tokio::test]
    async fn workflow_todos_rejects_multiple_in_progress_items() {
        let output = call_workflow_todos(json!([
            {"id": "todo-1", "content": "first", "status": "in_progress"},
            {"id": "todo-2", "content": "second", "status": "in_progress"}
        ]))
        .await;

        assert!(!output.ok);
        assert!(
            output
                .error
                .as_ref()
                .expect("todo error")
                .message
                .contains("at most one item with status 'in_progress'")
        );
    }

    #[tokio::test]
    async fn workflow_todos_rejects_invalid_status() {
        let output = call_workflow_todos(json!([
            {"id": "todo-1", "content": "first", "status": "doing"}
        ]))
        .await;

        assert!(!output.ok);
        assert!(
            output
                .error
                .as_ref()
                .expect("todo error")
                .message
                .contains("invalid status 'doing'")
        );
    }

    #[tokio::test]
    async fn workflow_auto_continue_rejects_invalid_types() {
        let tools = ToolRegistry::default_tools();
        let output = tools
            .call(
                "workflow__auto_continue",
                json!({"enabled": "yes", "max_continuations": null}),
            )
            .await;
        assert!(!output.ok);
        assert!(
            output
                .error
                .as_ref()
                .expect("auto-continue error")
                .message
                .contains("requires boolean field 'enabled'")
        );

        let output = tools
            .call(
                "workflow__auto_continue",
                json!({"enabled": true, "max_continuations": "many"}),
            )
            .await;
        assert!(!output.ok);
        assert!(
            output
                .error
                .as_ref()
                .expect("auto-continue error")
                .message
                .contains("must be integer or null")
        );
    }

    #[tokio::test]
    async fn read_only_explorer_scope_filters_specs_and_rejects_calls() {
        let registry = SkillRegistry::from_entries(vec![SkillEntry {
            name: "rust-audit".into(),
            description: "Inspect Rust code".into(),
            body: "# Body".into(),
            content: "---\nname: rust-audit\ndescription: Inspect Rust code\n---\n# Body\n".into(),
            location: "config".into(),
            path: PathBuf::from("/tmp/rust-audit/SKILL.md"),
            base_dir: PathBuf::from("/tmp/rust-audit"),
        }])
        .expect("skill registry");
        let mut tools = ToolRegistry::default_tools();
        tools
            .try_register(SkillTool::new(Arc::new(registry)))
            .expect("register skill tool");
        let tools = tools.scoped(ToolScope::ReadOnlyExplorer);
        let specs = tools.specs();

        assert!(specs.iter().any(|spec| spec.name == "fs__read"));
        assert!(specs.iter().any(|spec| spec.name == "skill"));
        assert!(!specs.iter().any(|spec| spec.name == "agent__explore"));
        assert!(!specs.iter().any(|spec| spec.name == "agent__fixer"));
        assert!(!specs.iter().any(|spec| spec.name == "fs__write"));
        assert!(!specs.iter().any(|spec| spec.name == "workflow__todos"));
        assert!(!specs.iter().any(|spec| spec.name == "memory__recall"));
        assert!(!specs.iter().any(|spec| spec.name == "context__checkpoint"));
        assert!(!specs.iter().any(|spec| spec.name == "context__return"));

        let output = tools
            .call("fs__write", json!({"path": "src/lib.rs", "content": "x"}))
            .await;
        assert!(!output.ok);
        assert_eq!(
            output.error.as_ref().expect("scope error").message,
            "tool 'fs__write' is not allowed in read_only_explorer scope"
        );

        let output = tools
            .call("agent__explore", json!({"task": "inspect"}))
            .await;
        assert!(!output.ok);
        assert_eq!(
            output.error.as_ref().expect("scope error").message,
            "tool 'agent__explore' is not allowed in read_only_explorer scope"
        );

        let output = tools
            .call("agent__fixer", json!({"task": "implement"}))
            .await;
        assert!(!output.ok);
        assert_eq!(
            output.error.as_ref().expect("scope error").message,
            "tool 'agent__fixer' is not allowed in read_only_explorer scope"
        );
    }
}
