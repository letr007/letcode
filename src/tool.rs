use anyhow::{Context, Result, anyhow, bail};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};
use std::future::Future;
use std::path::Component;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;
#[cfg(unix)]
use std::{
    fs::File,
    os::fd::{AsRawFd, FromRawFd, OwnedFd},
};
use tokio::fs;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

use crate::context_tree::ContextTreeState;
use crate::context_view::ContextViewProjection;
use crate::permission::{PermissionResource, ToolPermissionClass, classify_tool, path_preview};
use crate::request_builder::ToolSpec;
use crate::runtime_context::RuntimeSnapshot;
use crate::tool_names;

mod code_analysis;
mod command;
mod git;
mod memory;
mod question;
mod registry;
mod search;
mod workflow;

pub use registry::ToolRegistry;

const DEFAULT_READ_LINE_LIMIT: usize = 200;
const MAX_READ_LINE_LIMIT: usize = 5_000;
const MAX_READ_BYTES: usize = 4 * 1024 * 1024;
const COMMAND_TIMEOUT_SECS: u64 = 300;
const MAX_SUBAGENT_RECONCILIATION_SUMMARY_CHARS: usize = 2_000;
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuestionOption {
    pub label: String,
    pub description: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuestionSpec {
    pub question: String,
    pub header: String,
    #[serde(default)]
    pub options: Vec<QuestionOption>,
    #[serde(default)]
    pub multiple: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuestionRequest {
    pub questions: Vec<QuestionSpec>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuestionResponse {
    pub answers: Vec<Vec<String>>,
}

pub type QuestionCallbackFuture = Pin<Box<dyn Future<Output = Result<QuestionResponse>> + Send>>;
pub type QuestionCallback = Arc<dyn Fn(QuestionRequest) -> QuestionCallbackFuture + Send + Sync>;

#[derive(Clone, Default)]
pub struct ToolExecutionContext {
    pub allow_outside_workspace: bool,
    /// Authoritative branch/leaf-scoped context for context tools.
    pub runtime_snapshot: Option<Arc<RuntimeSnapshot>>,
    /// Dormant compatibility inputs for non-context-tool callers.
    pub context_view: Option<Arc<ContextViewProjection>>,
    pub context_tree: Option<Arc<ContextTreeState>>,
    pub question_handler: Option<QuestionCallback>,
    prepared_writable_leaf: Option<PreparedWritableLeaf>,
    prepared_apply_patch: Option<PreparedApplyPatch>,
}

impl std::fmt::Debug for ToolExecutionContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolExecutionContext")
            .field("allow_outside_workspace", &self.allow_outside_workspace)
            .field(
                "runtime_snapshot",
                &self.runtime_snapshot.as_ref().map(|_| "<runtime_snapshot>"),
            )
            .field(
                "context_view",
                &self.context_view.as_ref().map(|_| "<context_view>"),
            )
            .field(
                "context_tree",
                &self.context_tree.as_ref().map(|_| "<context_tree>"),
            )
            .field(
                "question_handler",
                &self.question_handler.as_ref().map(|_| "<question_handler>"),
            )
            .field(
                "prepared_writable_leaf",
                &self
                    .prepared_writable_leaf
                    .as_ref()
                    .map(|_| "<prepared_writable_leaf>"),
            )
            .field(
                "prepared_apply_patch",
                &self
                    .prepared_apply_patch
                    .as_ref()
                    .map(|_| "<prepared_apply_patch>"),
            )
            .finish()
    }
}

impl ToolExecutionContext {
    pub fn outside_workspace_granted() -> Self {
        Self {
            allow_outside_workspace: true,
            runtime_snapshot: None,
            context_view: None,
            context_tree: None,
            question_handler: None,
            prepared_writable_leaf: None,
            prepared_apply_patch: None,
        }
    }

    pub fn with_context_view(context_view: Arc<ContextViewProjection>) -> Self {
        let mut runtime_snapshot = RuntimeSnapshot::new("compatibility");
        runtime_snapshot.set_context_view((*context_view).clone());
        Self {
            allow_outside_workspace: false,
            runtime_snapshot: Some(Arc::new(runtime_snapshot)),
            context_view: Some(context_view),
            context_tree: None,
            question_handler: None,
            prepared_writable_leaf: None,
            prepared_apply_patch: None,
        }
    }

    pub fn with_context_snapshots(
        context_view: Arc<ContextViewProjection>,
        context_tree: Arc<ContextTreeState>,
    ) -> Self {
        let mut runtime_snapshot = RuntimeSnapshot::new("compatibility");
        runtime_snapshot.set_context_view((*context_view).clone());
        runtime_snapshot.set_context_tree((*context_tree).clone());
        Self {
            allow_outside_workspace: false,
            runtime_snapshot: Some(Arc::new(runtime_snapshot)),
            context_view: Some(context_view),
            context_tree: Some(context_tree),
            question_handler: None,
            prepared_writable_leaf: None,
            prepared_apply_patch: None,
        }
    }

    pub fn with_runtime_snapshot(runtime_snapshot: Arc<RuntimeSnapshot>) -> Self {
        Self {
            allow_outside_workspace: false,
            runtime_snapshot: Some(runtime_snapshot),
            context_view: None,
            context_tree: None,
            question_handler: None,
            prepared_writable_leaf: None,
            prepared_apply_patch: None,
        }
    }

    pub(crate) fn attach_prepared_writable_leaf(&mut self, prepared: PreparedWritableLeaf) {
        self.prepared_writable_leaf = Some(prepared);
    }

    pub(crate) fn attach_prepared_apply_patch(&mut self, prepared: PreparedApplyPatch) {
        self.prepared_apply_patch = Some(prepared);
    }
}

/// An opaque, authorization-time binding for an ApplyPatch batch.
#[derive(Clone)]
pub(crate) struct PreparedApplyPatch {
    inner: Arc<PreparedApplyPatchInner>,
}

impl std::fmt::Debug for PreparedApplyPatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedApplyPatch")
            .field("binding", &"<prepared_apply_patch>")
            .finish()
    }
}

const WRITABLE_DESTINATION_CHANGED: &str = "writable destination changed after authorization";

/// A canonical, authorization-time binding for one writable filesystem leaf.
#[derive(Clone)]
pub(crate) struct PreparedWritableLeaf {
    workspace_root: PathBuf,
    destination: PathBuf,
    parent: PathBuf,
    leaf: std::ffi::OsString,
    #[cfg(unix)]
    parent_dev: u64,
    #[cfg(unix)]
    parent_ino: u64,
}

impl PreparedWritableLeaf {
    pub(crate) fn external_workspace_access(&self) -> Option<ExternalWorkspaceAccess> {
        (!self.destination.starts_with(&self.workspace_root)).then(|| ExternalWorkspaceAccess {
            paths: vec![path_preview(&self.destination)],
        })
    }

    pub(crate) fn permission_resource(&self, tool: &str) -> PermissionResource {
        PermissionResource::ExactPath {
            tool: tool.into(),
            path: self.destination.clone(),
        }
    }

    fn validate_current_path(&self, raw_path: &str) -> Result<()> {
        let current = prepare_writable_leaf(raw_path)?;
        if current.destination != self.destination
            || current.parent != self.parent
            || !current.same_parent_instance(self)
        {
            bail!(WRITABLE_DESTINATION_CHANGED);
        }
        Ok(())
    }

    #[cfg(unix)]
    fn same_parent_instance(&self, other: &Self) -> bool {
        self.parent_dev == other.parent_dev && self.parent_ino == other.parent_ino
    }

    #[cfg(not(unix))]
    fn same_parent_instance(&self, _other: &Self) -> bool {
        true
    }
}

impl std::fmt::Debug for PreparedWritableLeaf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedWritableLeaf")
            .finish_non_exhaustive()
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

impl ToolRegistry {
    pub fn default_tools() -> Self {
        let mut registry = Self::new();
        registry.register(EchoTool);
        question::register(&mut registry);
        workflow::register(&mut registry);
        memory::register(&mut registry);
        registry.register(AgentExploreTool);
        registry.register(AgentFixerTool);
        registry.register(AgentReconcileTool);
        registry.register(ListDirTool);
        registry.register(ReadFileTool);
        registry.register(WriteFileTool);
        registry.register(AppendFileTool);
        command::register(&mut registry);
        registry.register(MkdirTool);
        search::register(&mut registry);
        git::register(&mut registry);
        registry.register(ApplyPatchTool);
        code_analysis::register(&mut registry);
        // Context tools are intentionally not registered while the prompt path
        // is history-only. Keep the module for later reintroduction.
        registry
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
                && let Ok(prepared) = prepare_writable_leaf(path)
                && let Some(access) = prepared.external_workspace_access()
            {
                paths.extend(access.paths);
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

/// Build the narrowest stable resource identity used by session Allow Always grants.
/// Paths are canonicalized before comparison so spelling tricks and symlinks cannot widen a grant.
pub fn permission_resource_for_tool(name: &str, args: &Value) -> Option<PermissionResource> {
    let exact_path = |path: &str| canonical_destination_path(path);
    let directory_path = |path: &str| canonical_existing_path(path);
    match name {
        tool_names::TOOL_FS_READ => {
            let path = args.get("path")?.as_str()?;
            let canonical = canonical_existing_path(path)?;
            if canonical.is_dir() {
                Some(PermissionResource::Directory {
                    tool: name.into(),
                    path: canonical,
                })
            } else {
                Some(PermissionResource::ExactPath {
                    tool: name.into(),
                    path: canonical,
                })
            }
        }
        tool_names::TOOL_FS_LIST
        | tool_names::TOOL_SEARCH_RG
        | tool_names::TOOL_CODE_AST_SEARCH
        | tool_names::TOOL_CODE_AST_REPLACE_PREVIEW => {
            let path = args.get("path").and_then(Value::as_str).unwrap_or(".");
            Some(PermissionResource::Directory {
                tool: name.into(),
                path: directory_path(path)?,
            })
        }
        tool_names::TOOL_FS_WRITE | tool_names::TOOL_FS_APPEND => {
            let path = args.get("path")?.as_str()?;
            Some(prepare_writable_leaf(path).ok()?.permission_resource(name))
        }
        tool_names::TOOL_FS_MKDIR => {
            let path = args.get("path")?.as_str()?;
            Some(PermissionResource::ExactPath {
                tool: name.into(),
                path: exact_path(path)?,
            })
        }
        tool_names::TOOL_EDIT_APPLY_PATCH => {
            let paths = args
                .get("edits")?
                .as_array()?
                .iter()
                .map(|edit| {
                    edit.get("path")
                        .and_then(Value::as_str)
                        .and_then(exact_path)
                })
                .collect::<Option<BTreeSet<_>>>()?;
            Some(PermissionResource::PatchTargets {
                tool: name.into(),
                paths,
            })
        }
        tool_names::TOOL_SHELL_EXEC => Some(PermissionResource::Exact {
            tool: name.into(),
            value: args.get("command")?.as_str()?.trim().to_string(),
        }),
        _ => Some(PermissionResource::Exact {
            tool: name.into(),
            value: canonical_json(args),
        }),
    }
}

fn canonical_existing_path(path: &str) -> Option<PathBuf> {
    join_workspace_path(&workspace_root().ok()?, path)
        .canonicalize()
        .ok()
}

fn canonical_destination_path(path: &str) -> Option<PathBuf> {
    let root = workspace_root().ok()?;
    let candidate = join_workspace_path(&root, path);
    if let Ok(path) = candidate.canonicalize() {
        return Some(path);
    }
    let parent = candidate.parent()?.canonicalize().ok()?;
    Some(parent.join(candidate.file_name()?))
}

fn canonical_json(value: &Value) -> String {
    match value {
        Value::Object(map) => {
            let mut keys: Vec<_> = map.keys().collect();
            keys.sort();
            format!(
                "{{{}}}",
                keys.into_iter()
                    .map(|key| format!(
                        "{}:{}",
                        serde_json::to_string(key).expect("string"),
                        canonical_json(&map[key])
                    ))
                    .collect::<Vec<_>>()
                    .join(",")
            )
        }
        Value::Array(values) => format!(
            "[{}]",
            values
                .iter()
                .map(canonical_json)
                .collect::<Vec<_>>()
                .join(",")
        ),
        _ => serde_json::to_string(value).expect("serializable JSON"),
    }
}

struct EchoTool;

struct AgentExploreTool;

struct AgentFixerTool;

struct AgentReconcileTool;

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

#[async_trait]
impl ToolHandler for AgentReconcileTool {
    fn name(&self) -> &'static str {
        tool_names::TOOL_AGENT_RECONCILE
    }

    fn description(&self) -> &'static str {
        "显式记录父代理已采纳、驳回或标记冲突的子代理结果。"
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "run_id": {
                    "type": "string",
                    "maxLength": MAX_SUBAGENT_TEXT_FIELD_CHARS
                },
                "child_session_id": {
                    "type": "string",
                    "maxLength": MAX_SUBAGENT_TEXT_FIELD_CHARS
                },
                "agent_name": {
                    "type": "string",
                    "enum": ["fixer", "explorer", "oracle", "designer", "librarian", "general"]
                },
                "decision": {
                    "type": "string",
                    "enum": ["accepted", "rejected", "conflict"]
                },
                "summary": {
                    "type": "string",
                    "maxLength": MAX_SUBAGENT_RECONCILIATION_SUMMARY_CHARS
                }
            },
            "required": ["run_id", "child_session_id", "agent_name", "decision", "summary"],
            "additionalProperties": false
        })
    }

    async fn execute(&self, args: Value) -> Result<Value> {
        let payload = validate_agent_reconcile(&args)?;
        Ok(json!({
            "run_id": payload.run_id,
            "child_session_id": payload.child_session_id,
            "agent_name": payload.agent_name,
            "decision": payload.decision,
            "summary": payload.summary,
            "reconciled": true,
            "pending_recording": true
        }))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct AgentReconcilePayload {
    run_id: String,
    child_session_id: String,
    agent_name: String,
    decision: String,
    summary: String,
}

fn validate_agent_reconcile(args: &Value) -> Result<AgentReconcilePayload> {
    let run_id = required_trimmed_string_field(args, "run_id", MAX_SUBAGENT_TEXT_FIELD_CHARS)?;
    let child_session_id =
        required_trimmed_string_field(args, "child_session_id", MAX_SUBAGENT_TEXT_FIELD_CHARS)?;
    let agent_name = required_trimmed_string_field(args, "agent_name", 64)?;
    if crate::agent::subagent_tool_name_for_agent_name(&agent_name).is_none() {
        bail!(
            "field 'agent_name' must be one of fixer, explorer, oracle, designer, librarian, general"
        );
    }
    let decision = required_trimmed_string_field(args, "decision", 32)?;
    if !matches!(decision.as_str(), "accepted" | "rejected" | "conflict") {
        bail!("field 'decision' must be one of accepted, rejected, conflict");
    }
    let summary =
        required_trimmed_string_field(args, "summary", MAX_SUBAGENT_RECONCILIATION_SUMMARY_CHARS)?;
    Ok(AgentReconcilePayload {
        run_id,
        child_session_id,
        agent_name,
        decision,
        summary,
    })
}

fn required_trimmed_string_field(args: &Value, field: &str, max_chars: usize) -> Result<String> {
    let Some(value) = args.get(field) else {
        bail!(
            "{} requires string field '{field}'",
            tool_names::TOOL_AGENT_RECONCILE
        );
    };
    let Some(text) = value.as_str() else {
        bail!("field '{field}' must be a string");
    };
    let trimmed = text.trim();
    if trimmed.is_empty() {
        bail!("field '{field}' must not be empty or whitespace");
    }
    if trimmed.chars().count() > max_chars {
        bail!("field '{field}' exceeds {max_chars} characters");
    }
    Ok(trimmed.to_string())
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

struct ApplyPatchTool;

#[async_trait]
impl ToolHandler for ApplyPatchTool {
    fn name(&self) -> &'static str {
        "edit__apply_patch"
    }

    fn description(&self) -> &'static str {
        "Apply exact-match text replacements to existing UTF-8 files under the workspace. Each edit must provide the exact old text in `find` and replacement text in `replace`. By default use replace_all=false so the tool fails unless `find` matches exactly once. All edits are first validated against staged in-memory content before any file is written. After validation, files are written individually and non-transactionally, so I/O, timeout, cancellation, or process failure can leave previously written files changed. This is intended for precise, low-ambiguity code edits."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "edits": {
                    "type": "array",
                    "description": "Exact-match replacement edits. All edits are validated against staged in-memory content before any file is written. After validation, files are written individually and non-transactionally, so I/O, timeout, cancellation, or process failure can leave previously written files changed",
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
    let prepared = writable_leaf_for_execution(raw_path, &context)?;
    let path = prepared.destination.clone();
    secure_write_writable_leaf(&prepared, content.as_bytes(), false).await?;

    Ok(json!({
        "path": display_workspace_relative(&path)?,
        "bytes_written": content.len(),
    }))
}

async fn append_file(args: Value, context: ToolExecutionContext) -> Result<Value> {
    let raw_path = required_string(&args, "path")?;
    let content = required_string(&args, "content")?;
    let prepared = writable_leaf_for_execution(raw_path, &context)?;
    let path = prepared.destination.clone();
    secure_write_writable_leaf(&prepared, content.as_bytes(), true).await?;

    Ok(json!({
        "path": display_workspace_relative(&path)?,
        "bytes_appended": content.len(),
    }))
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

const APPLY_PATCH_CHANGED: &str = "apply patch target changed after authorization";
const APPLY_PATCH_MODIFIED: &str = "apply patch target was concurrently modified";

struct ParsedApplyPatchEdit {
    path: String,
    find: String,
    replace: String,
    replace_all: bool,
}

struct PreparedApplyPatchInner {
    workspace_root: PathBuf,
    edit_paths: Vec<String>,
    edit_targets: Vec<PathBuf>,
    targets: BTreeMap<PathBuf, PreparedApplyPatchTarget>,
    #[cfg(unix)]
    parents: BTreeMap<PathBuf, PreparedApplyPatchParent>,
    #[cfg(test)]
    hook: std::sync::Mutex<Option<(ApplyPatchWorkerPoint, Box<dyn FnOnce() + Send>)>>,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
enum ApplyPatchWorkerPoint {
    BeforeStagingValidation,
    BeforeBatchPrecommit,
    BeforeCommitOpen { path: PathBuf, ordinal: usize },
}

#[cfg(unix)]
struct PreparedApplyPatchParent {
    fd: OwnedFd,
    dev: u64,
    ino: u64,
}

#[cfg(unix)]
struct PreparedApplyPatchTarget {
    leaf: std::ffi::OsString,
    parent: PathBuf,
    fd: OwnedFd,
    dev: u64,
    ino: u64,
    size: i64,
    mtime: (i64, i64),
    ctime: (i64, i64),
    nlink: u64,
}

#[cfg(not(unix))]
struct PreparedApplyPatchTarget;

fn parse_apply_patch(args: &Value) -> Result<Vec<ParsedApplyPatchEdit>> {
    let edits = args
        .get("edits")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("missing or invalid array argument: edits"))?;
    if edits.is_empty() {
        bail!("edits cannot be empty");
    }
    edits
        .iter()
        .enumerate()
        .map(|(index, edit)| {
            let path = required_string(edit, "path")?.to_owned();
            let find = required_string(edit, "find")?.to_owned();
            let replace = required_string(edit, "replace")?.to_owned();
            let replace_all = edit
                .get("replace_all")
                .and_then(Value::as_bool)
                .ok_or_else(|| {
                    anyhow!("missing or invalid boolean argument: edits[{index}].replace_all")
                })?;
            if find.is_empty() {
                bail!("edits[{index}].find cannot be empty");
            }
            Ok(ParsedApplyPatchEdit {
                path,
                find,
                replace,
                replace_all,
            })
        })
        .collect()
}

pub(crate) fn prepare_apply_patch_targets(args: &Value) -> Result<PreparedApplyPatch> {
    let edits = parse_apply_patch(args)?;
    prepare_apply_patch_edits(&edits)
}

#[cfg(unix)]
fn prepare_apply_patch_edits(edits: &[ParsedApplyPatchEdit]) -> Result<PreparedApplyPatch> {
    use std::os::unix::{ffi::OsStrExt, fs::MetadataExt};

    let workspace_root = workspace_root()?;
    let edit_paths = edits.iter().map(|edit| edit.path.clone()).collect();
    let mut edit_targets = Vec::with_capacity(edits.len());
    let mut canonical = BTreeSet::new();
    for edit in edits {
        let candidate = join_workspace_path(&workspace_root, &edit.path);
        let target = candidate
            .canonicalize()
            .with_context(|| format!("path does not exist: {}", candidate.display()))?;
        edit_targets.push(target.clone());
        canonical.insert(target);
    }
    if canonical.len() > 64 {
        bail!("apply patch accepts at most 64 unique target files");
    }

    let mut parents = BTreeMap::new();
    let mut targets = BTreeMap::new();
    let mut identities = BTreeSet::new();
    for destination in canonical {
        let parent = destination
            .parent()
            .ok_or_else(|| anyhow!("path has no parent: {}", destination.display()))?
            .to_path_buf();
        let leaf = destination
            .file_name()
            .ok_or_else(|| anyhow!("path has no file name: {}", destination.display()))?
            .to_os_string();
        if !parents.contains_key(&parent) {
            let parent_c = std::ffi::CString::new(parent.as_os_str().as_bytes())?;
            let fd = unsafe {
                libc::open(
                    parent_c.as_ptr(),
                    libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW,
                )
            };
            if fd < 0 {
                return Err(std::io::Error::last_os_error()).with_context(|| {
                    format!("failed to open parent directory {}", parent.display())
                });
            }
            let fd = unsafe { OwnedFd::from_raw_fd(fd) };
            let metadata = File::from(fd.try_clone().map_err(|error| {
                apply_patch_io_error(&destination, "clone parent directory anchor", error)
            })?)
            .metadata()
            .map_err(|error| {
                apply_patch_io_error(&destination, "inspect parent directory", error)
            })?;
            parents.insert(
                parent.clone(),
                PreparedApplyPatchParent {
                    fd,
                    dev: metadata.dev(),
                    ino: metadata.ino(),
                },
            );
        }
        let anchor = parents.get(&parent).expect("inserted parent");
        let leaf_c = std::ffi::CString::new(leaf.as_bytes())?;
        let fd = unsafe {
            libc::openat(
                anchor.fd.as_raw_fd(),
                leaf_c.as_ptr(),
                libc::O_RDONLY | libc::O_NONBLOCK | libc::O_CLOEXEC | libc::O_NOFOLLOW,
            )
        };
        if fd < 0 {
            return Err(std::io::Error::last_os_error())
                .with_context(|| format!("failed to open file {}", destination.display()));
        }
        let fd = unsafe { OwnedFd::from_raw_fd(fd) };
        let metadata = File::from(
            fd.try_clone()
                .map_err(|error| apply_patch_io_error(&destination, "clone file anchor", error))?,
        )
        .metadata()
        .map_err(|error| apply_patch_io_error(&destination, "inspect file", error))?;
        if !metadata.is_file() {
            bail!("edits path is not a file: {}", destination.display());
        }
        if !identities.insert((metadata.dev(), metadata.ino())) {
            bail!("apply patch targets alias the same existing file");
        }
        targets.insert(
            destination.clone(),
            PreparedApplyPatchTarget {
                leaf,
                parent,
                fd,
                dev: metadata.dev(),
                ino: metadata.ino(),
                size: metadata.size() as i64,
                mtime: (metadata.mtime(), metadata.mtime_nsec()),
                ctime: (metadata.ctime(), metadata.ctime_nsec()),
                nlink: metadata.nlink(),
            },
        );
    }
    Ok(PreparedApplyPatch {
        inner: Arc::new(PreparedApplyPatchInner {
            workspace_root,
            edit_paths,
            edit_targets,
            targets,
            parents,
            #[cfg(test)]
            hook: std::sync::Mutex::new(None),
        }),
    })
}

#[cfg(not(unix))]
fn prepare_apply_patch_edits(_edits: &[ParsedApplyPatchEdit]) -> Result<PreparedApplyPatch> {
    bail!("secure apply patch authorization is unsupported on this platform")
}

impl PreparedApplyPatch {
    pub(crate) fn external_workspace_access(&self) -> Option<ExternalWorkspaceAccess> {
        let paths: Vec<_> = self
            .inner
            .targets
            .keys()
            .filter(|path| !path.starts_with(&self.inner.workspace_root))
            .map(|path| path_preview(path))
            .collect();
        (!paths.is_empty()).then_some(ExternalWorkspaceAccess { paths })
    }

    pub(crate) fn permission_resource(&self, tool: &str) -> PermissionResource {
        PermissionResource::PatchTargets {
            tool: tool.into(),
            paths: self.inner.targets.keys().cloned().collect(),
        }
    }

    #[cfg(test)]
    fn set_worker_hook(&self, point: ApplyPatchWorkerPoint, hook: impl FnOnce() + Send + 'static) {
        *self.inner.hook.lock().expect("apply patch hook poisoned") = Some((point, Box::new(hook)));
    }

    #[cfg(test)]
    fn run_worker_hook(&self, point: ApplyPatchWorkerPoint) {
        let hook = {
            let mut hook = self.inner.hook.lock().expect("apply patch hook poisoned");
            hook.as_ref()
                .is_some_and(|(expected, _)| expected == &point)
                .then(|| hook.take().expect("hook checked").1)
        };
        if let Some(hook) = hook {
            hook();
        }
    }
}

fn prepared_apply_patch_for_execution(
    args: &Value,
    context: &ToolExecutionContext,
) -> Result<(Vec<ParsedApplyPatchEdit>, PreparedApplyPatch)> {
    let edits = parse_apply_patch(args)?;
    let prepared = match &context.prepared_apply_patch {
        Some(prepared) => prepared.clone(),
        None => prepare_apply_patch_edits(&edits)?,
    };
    if !context.allow_outside_workspace && prepared.external_workspace_access().is_some() {
        bail!("path is outside workspace");
    }
    validate_apply_patch_mapping(&prepared, &edits)?;
    Ok((edits, prepared))
}

fn validate_apply_patch_mapping(
    prepared: &PreparedApplyPatch,
    edits: &[ParsedApplyPatchEdit],
) -> Result<()> {
    if edits.len() != prepared.inner.edit_targets.len()
        || edits
            .iter()
            .map(|edit| &edit.path)
            .ne(prepared.inner.edit_paths.iter())
    {
        bail!(APPLY_PATCH_CHANGED);
    }
    for expected in &prepared.inner.edit_targets {
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            let target = prepared
                .inner
                .targets
                .get(expected)
                .expect("prepared target");
            let parent = prepared
                .inner
                .parents
                .get(&target.parent)
                .expect("prepared parent");
            let metadata = std::fs::metadata(&target.parent).map_err(|error| {
                apply_patch_open_error(expected, "inspect parent directory", error)
            })?;
            if !metadata.is_dir() || metadata.dev() != parent.dev || metadata.ino() != parent.ino {
                bail!(APPLY_PATCH_CHANGED);
            }
            let leaf_metadata = std::fs::metadata(expected)
                .map_err(|error| apply_patch_open_error(expected, "inspect file", error))?;
            if !apply_patch_metadata_matches(target, &leaf_metadata) {
                bail!(APPLY_PATCH_CHANGED);
            }
        }
    }
    Ok(())
}

#[cfg(unix)]
fn apply_patch_open_error(path: &Path, action: &str, error: std::io::Error) -> anyhow::Error {
    match error.raw_os_error() {
        Some(libc::ENOENT | libc::ENOTDIR | libc::ELOOP) => anyhow!(APPLY_PATCH_CHANGED),
        _ => anyhow!(error).context(format!("failed to {action} {}", path.display())),
    }
}

#[cfg(unix)]
fn apply_patch_io_error(path: &Path, action: &str, error: std::io::Error) -> anyhow::Error {
    anyhow!(error).context(format!("failed to {action} {}", path.display()))
}

#[cfg(unix)]
fn apply_patch_metadata_matches(
    target: &PreparedApplyPatchTarget,
    metadata: &std::fs::Metadata,
) -> bool {
    use std::os::unix::fs::MetadataExt;
    metadata.is_file() && metadata.dev() == target.dev && metadata.ino() == target.ino
}

#[cfg(unix)]
fn apply_patch_version_matches(
    target: &PreparedApplyPatchTarget,
    metadata: &std::fs::Metadata,
) -> bool {
    use std::os::unix::fs::MetadataExt;
    metadata.size() as i64 == target.size
        && (metadata.mtime(), metadata.mtime_nsec()) == target.mtime
        && (metadata.ctime(), metadata.ctime_nsec()) == target.ctime
        && metadata.nlink() == target.nlink
}

#[cfg(unix)]
fn apply_patch_open(prepared: &PreparedApplyPatch, path: &Path, writable: bool) -> Result<File> {
    use std::os::unix::ffi::OsStrExt;
    let target = prepared.inner.targets.get(path).expect("prepared target");
    let parent = prepared
        .inner
        .parents
        .get(&target.parent)
        .expect("prepared parent");
    let parent_fd = parent
        .fd
        .try_clone()
        .map_err(|error| apply_patch_io_error(path, "clone parent directory anchor", error))?;
    let parent_metadata = File::from(parent_fd)
        .metadata()
        .map_err(|error| apply_patch_io_error(path, "inspect parent directory", error))?;
    use std::os::unix::fs::MetadataExt;
    if parent_metadata.dev() != parent.dev || parent_metadata.ino() != parent.ino {
        bail!(APPLY_PATCH_CHANGED);
    }
    let leaf = std::ffi::CString::new(target.leaf.as_bytes())?;
    let flags = if writable {
        libc::O_RDWR
    } else {
        libc::O_RDONLY
    } | libc::O_NONBLOCK
        | libc::O_CLOEXEC
        | libc::O_NOFOLLOW;
    let fd = unsafe { libc::openat(parent.fd.as_raw_fd(), leaf.as_ptr(), flags) };
    if fd < 0 {
        let open_error = std::io::Error::last_os_error();
        return Err(apply_patch_open_failure(
            parent,
            target,
            path,
            if writable {
                "open for writing"
            } else {
                "open for reading"
            },
            open_error,
        ));
    }
    let file = unsafe { File::from_raw_fd(fd) };
    let metadata = file
        .metadata()
        .map_err(|error| apply_patch_io_error(path, "inspect file", error))?;
    if !apply_patch_metadata_matches(target, &metadata) {
        bail!(APPLY_PATCH_CHANGED);
    }
    if !apply_patch_version_matches(target, &metadata) {
        bail!(APPLY_PATCH_MODIFIED);
    }
    Ok(file)
}

#[cfg(unix)]
fn apply_patch_open_failure(
    parent: &PreparedApplyPatchParent,
    target: &PreparedApplyPatchTarget,
    path: &Path,
    action: &str,
    open_error: std::io::Error,
) -> anyhow::Error {
    use std::os::unix::ffi::OsStrExt;

    let leaf = match std::ffi::CString::new(target.leaf.as_bytes()) {
        Ok(leaf) => leaf,
        Err(_) => return apply_patch_io_error(path, action, open_error),
    };
    let mut stat = std::mem::MaybeUninit::<libc::stat>::uninit();
    let result = unsafe {
        libc::fstatat(
            parent.fd.as_raw_fd(),
            leaf.as_ptr(),
            stat.as_mut_ptr(),
            libc::AT_SYMLINK_NOFOLLOW,
        )
    };
    if result < 0 {
        return match std::io::Error::last_os_error().raw_os_error() {
            Some(libc::ENOENT | libc::ENOTDIR | libc::ELOOP) => anyhow!(APPLY_PATCH_CHANGED),
            _ => apply_patch_io_error(path, action, open_error),
        };
    }
    let stat = unsafe { stat.assume_init() };
    let regular = (stat.st_mode & libc::S_IFMT) == libc::S_IFREG;
    if !regular || stat.st_dev as u64 != target.dev || stat.st_ino as u64 != target.ino {
        anyhow!(APPLY_PATCH_CHANGED)
    } else {
        apply_patch_io_error(path, action, open_error)
    }
}

#[cfg(unix)]
fn apply_patch_read(
    file: &mut File,
    target: &PreparedApplyPatchTarget,
    path: &Path,
) -> Result<Vec<u8>> {
    use std::io::{Read, Seek, SeekFrom};
    let before = file
        .metadata()
        .map_err(|error| apply_patch_io_error(path, "inspect file", error))?;
    if !apply_patch_metadata_matches(target, &before) {
        bail!(APPLY_PATCH_CHANGED);
    }
    if !apply_patch_version_matches(target, &before) {
        bail!(APPLY_PATCH_MODIFIED);
    }
    file.seek(SeekFrom::Start(0))
        .map_err(|error| apply_patch_io_error(path, "seek file", error))?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)
        .map_err(|error| apply_patch_io_error(path, "read file", error))?;
    let after = file
        .metadata()
        .map_err(|error| apply_patch_io_error(path, "inspect file", error))?;
    if !apply_patch_metadata_matches(target, &after) {
        bail!(APPLY_PATCH_CHANGED);
    }
    if !apply_patch_version_matches(target, &after) {
        bail!(APPLY_PATCH_MODIFIED);
    }
    Ok(bytes)
}

async fn apply_patch(args: Value, context: ToolExecutionContext) -> Result<Value> {
    tokio::task::spawn_blocking(move || apply_patch_worker(args, context)).await?
}

#[cfg(unix)]
fn apply_patch_worker(args: Value, context: ToolExecutionContext) -> Result<Value> {
    use std::io::{Seek, SeekFrom, Write};
    let (edits, prepared) = prepared_apply_patch_for_execution(&args, &context)?;
    let mut contents = BTreeMap::new();
    let mut originals = BTreeMap::new();
    for path in prepared.inner.targets.keys() {
        let target = prepared.inner.targets.get(path).expect("prepared target");
        let mut file = File::from(
            target
                .fd
                .try_clone()
                .map_err(|error| apply_patch_io_error(path, "open approved file", error))?,
        );
        let bytes = apply_patch_read(&mut file, target, path)?;
        let content = String::from_utf8(bytes)
            .with_context(|| format!("failed to read UTF-8 file {}", path.display()))?;
        originals.insert(path.clone(), content.clone());
        contents.insert(path.clone(), content);
    }
    #[cfg(test)]
    prepared.run_worker_hook(ApplyPatchWorkerPoint::BeforeStagingValidation);
    let mut results = Vec::with_capacity(edits.len());
    for (index, edit) in edits.iter().enumerate() {
        let path = &prepared.inner.edit_targets[index];
        let content = contents.get_mut(path).expect("staged content");
        let matches = content.matches(&edit.find).count();
        let label = bound_display(&prepared.inner.workspace_root, path);
        if matches == 0 {
            bail!("edits[{index}] did not match any text in {label}");
        }
        if !edit.replace_all && matches != 1 {
            bail!(
                "edits[{index}] matched {matches} occurrences in {label}; provide more context or set replace_all=true"
            );
        }
        *content = if edit.replace_all {
            content.replace(&edit.find, &edit.replace)
        } else {
            content.replacen(&edit.find, &edit.replace, 1)
        };
        results
            .push(json!({"path": label, "replacements": matches, "replace_all": edit.replace_all}));
    }
    validate_apply_patch_mapping(&prepared, &edits)?;
    #[cfg(test)]
    prepared.run_worker_hook(ApplyPatchWorkerPoint::BeforeBatchPrecommit);
    for path in prepared.inner.targets.keys() {
        let target = prepared.inner.targets.get(path).expect("prepared target");
        let mut file = apply_patch_open(&prepared, path, false)?;
        let bytes = apply_patch_read(&mut file, target, path)?;
        if bytes != originals.get(path).expect("staged original").as_bytes() {
            bail!(APPLY_PATCH_MODIFIED);
        }
    }
    for (ordinal, (path, content)) in contents.iter().enumerate() {
        #[cfg(test)]
        prepared.run_worker_hook(ApplyPatchWorkerPoint::BeforeCommitOpen {
            path: path.clone(),
            ordinal,
        });
        #[cfg(not(test))]
        let _ = ordinal;
        let target = prepared.inner.targets.get(path).expect("prepared target");
        let mut file = apply_patch_open(&prepared, path, true)?;
        let current = apply_patch_read(&mut file, target, path)?;
        if current != originals.get(path).expect("staged original").as_bytes() {
            bail!(APPLY_PATCH_MODIFIED);
        }
        file.set_len(0)
            .map_err(|error| apply_patch_io_error(path, "write patched file", error))?;
        file.seek(SeekFrom::Start(0))
            .map_err(|error| apply_patch_io_error(path, "seek file", error))?;
        file.write_all(content.as_bytes())
            .map_err(|error| apply_patch_io_error(path, "write patched file", error))?;
        file.flush()
            .map_err(|error| apply_patch_io_error(path, "flush file", error))?;
        file.seek(SeekFrom::Start(0))
            .map_err(|error| apply_patch_io_error(path, "seek file", error))?;
        let mut verified = Vec::new();
        std::io::Read::read_to_end(&mut file, &mut verified)
            .map_err(|error| apply_patch_io_error(path, "verify file", error))?;
        if verified != content.as_bytes() {
            bail!("failed to write patched file {}", path.display());
        }
    }
    Ok(json!({"files_changed": contents.len(), "edits_applied": edits.len(), "edits": results}))
}

#[cfg(not(unix))]
fn apply_patch_worker(args: Value, context: ToolExecutionContext) -> Result<Value> {
    let _ = prepared_apply_patch_for_execution(&args, &context)?;
    bail!("secure apply patch authorization is unsupported on this platform")
}

fn bound_display(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .to_string()
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

pub(crate) fn prepare_writable_leaf(path: &str) -> Result<PreparedWritableLeaf> {
    prepare_writable_leaf_platform(path)
}

#[cfg(unix)]
fn prepare_writable_leaf_platform(path: &str) -> Result<PreparedWritableLeaf> {
    use std::os::unix::fs::MetadataExt;

    let workspace_root = workspace_root()?;
    let candidate = join_workspace_path(&workspace_root, path);
    let destination = resolve_writable_leaf_destination(&candidate)?;
    let parent = destination
        .parent()
        .ok_or_else(|| anyhow!("path has no parent: {}", destination.display()))?
        .to_path_buf();
    let leaf = destination
        .file_name()
        .ok_or_else(|| anyhow!("path has no file name: {}", destination.display()))?
        .to_os_string();
    let metadata = std::fs::metadata(&parent)
        .with_context(|| format!("parent directory does not exist: {}", parent.display()))?;
    if !metadata.is_dir() {
        bail!("parent is not a directory: {}", parent.display());
    }
    Ok(PreparedWritableLeaf {
        workspace_root,
        destination,
        parent,
        leaf,
        parent_dev: metadata.dev(),
        parent_ino: metadata.ino(),
    })
}

#[cfg(not(unix))]
fn prepare_writable_leaf_platform(_path: &str) -> Result<PreparedWritableLeaf> {
    bail!("secure writable leaf authorization is unsupported on this platform")
}

#[cfg(unix)]
fn resolve_writable_leaf_destination(candidate: &Path) -> Result<PathBuf> {
    let mut current = candidate.to_path_buf();
    for _ in 0..40 {
        let parent = current
            .parent()
            .ok_or_else(|| anyhow!("path has no parent: {}", current.display()))?
            .canonicalize()
            .with_context(|| format!("parent directory does not exist: {}", current.display()))?;
        let leaf = current
            .file_name()
            .ok_or_else(|| anyhow!("path has no file name: {}", current.display()))?;
        let resolved = parent.join(leaf);
        match std::fs::symlink_metadata(&resolved) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                let target = std::fs::read_link(&resolved)
                    .with_context(|| format!("failed to read symlink: {}", resolved.display()))?;
                current = if target.is_absolute() {
                    target
                } else {
                    parent.join(target)
                };
            }
            Ok(_) => {
                return resolved.canonicalize().with_context(|| {
                    format!(
                        "failed to canonicalize writable destination: {}",
                        resolved.display()
                    )
                });
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(resolved),
            Err(error) => {
                return Err(error).with_context(|| {
                    format!(
                        "failed to inspect writable destination: {}",
                        resolved.display()
                    )
                });
            }
        }
    }
    bail!(
        "too many symlink levels while resolving writable destination: {}",
        candidate.display()
    )
}

fn writable_leaf_for_execution(
    raw_path: &str,
    context: &ToolExecutionContext,
) -> Result<PreparedWritableLeaf> {
    let prepared = if let Some(prepared) = context.prepared_writable_leaf.clone() {
        prepared
            .validate_current_path(raw_path)
            .map_err(|_| anyhow!(WRITABLE_DESTINATION_CHANGED))?;
        prepared
    } else {
        prepare_writable_leaf(raw_path)?
    };
    if !context.allow_outside_workspace {
        ensure_inside_workspace(&prepared.workspace_root, &prepared.destination)?;
    }
    Ok(prepared)
}

#[cfg(unix)]
async fn secure_write_writable_leaf(
    prepared: &PreparedWritableLeaf,
    content: &[u8],
    append: bool,
) -> Result<()> {
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::fs::MetadataExt;

    let parent_c = std::ffi::CString::new(prepared.parent.as_os_str().as_bytes())?;
    let parent_fd = unsafe {
        libc::open(
            parent_c.as_ptr(),
            libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW,
        )
    };
    if parent_fd < 0 {
        bail!(WRITABLE_DESTINATION_CHANGED);
    }
    let parent_file = unsafe { File::from_raw_fd(parent_fd) };
    let metadata = parent_file.metadata()?;
    if metadata.dev() != prepared.parent_dev || metadata.ino() != prepared.parent_ino {
        bail!(WRITABLE_DESTINATION_CHANGED);
    }
    let leaf_c = std::ffi::CString::new(prepared.leaf.as_bytes())?;
    let flags = libc::O_WRONLY
        | libc::O_CREAT
        | libc::O_CLOEXEC
        | libc::O_NOFOLLOW
        | if append {
            libc::O_APPEND
        } else {
            libc::O_TRUNC
        };
    let leaf_fd = unsafe { libc::openat(parent_file.as_raw_fd(), leaf_c.as_ptr(), flags, 0o666) };
    if leaf_fd < 0 {
        let error = std::io::Error::last_os_error();
        if matches!(error.raw_os_error(), Some(libc::ELOOP)) {
            bail!(WRITABLE_DESTINATION_CHANGED);
        }
        return Err(error)
            .with_context(|| format!("failed to open file {}", prepared.destination.display()));
    }
    let owned = unsafe { OwnedFd::from_raw_fd(leaf_fd) };
    let mut file = tokio::fs::File::from_std(File::from(owned));
    file.write_all(content).await?;
    file.flush().await?;
    Ok(())
}

#[cfg(not(unix))]
async fn secure_write_writable_leaf(
    _prepared: &PreparedWritableLeaf,
    _content: &[u8],
    _append: bool,
) -> Result<()> {
    bail!("secure writable leaf authorization is unsupported on this platform")
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

#[cfg(test)]
mod tests {
    use super::{
        ApplyPatchWorkerPoint, ToolExecutionContext, ToolRegistry,
        external_workspace_access_for_tool, normalize_subagent_input, permission_resource_for_tool,
        prepare_apply_patch_targets, prepare_writable_leaf, secure_write_writable_leaf,
    };
    use crate::permission::{PermissionResource, ToolScope};
    use crate::skills::{SkillEntry, SkillRegistry, SkillTool};
    use crate::tool::ToolOutputStream;
    use crate::tool_names;
    use serde_json::json;
    use std::path::PathBuf;
    use std::sync::{Arc, Mutex};

    async fn call_workflow_todos(items: serde_json::Value) -> crate::tool::ToolResult {
        ToolRegistry::default_tools()
            .call("workflow__todos", json!({"items": items}))
            .await
    }

    #[test]
    fn permission_resources_are_narrow_and_canonical() {
        let directory = permission_resource_for_tool("fs__read", &json!({"path": "src"}))
            .expect("src directory resource");
        let descendant =
            permission_resource_for_tool("fs__read", &json!({"path": "src/permission.rs"}))
                .expect("src file resource");
        let sibling = PermissionResource::ExactPath {
            tool: "fs__read".into(),
            path: std::env::current_dir()
                .expect("cwd")
                .join("srcx/permission.rs"),
        };
        assert!(directory.matches(&PermissionResource::Directory {
            tool: "fs__read".into(),
            path: std::env::current_dir().expect("cwd").join("src/nested")
        }));
        assert!(directory.matches(&descendant));
        assert!(!directory.matches(&sibling));
        assert!(!descendant.matches(&directory));

        for (tool, args) in [
            ("fs__read", json!({"path": "src/permission.rs"})),
            (
                "fs__write",
                json!({"path": "src/permission.rs", "content": "ignored"}),
            ),
            (
                "fs__append",
                json!({"path": "src/permission.rs", "content": "ignored"}),
            ),
            ("fs__mkdir", json!({"path": "target"})),
        ] {
            assert!(
                matches!(permission_resource_for_tool(tool, &args), Some(PermissionResource::ExactPath { tool: resource_tool, .. }) if resource_tool == tool),
                "{tool} must use an exact path resource"
            );
        }
        assert_eq!(
            permission_resource_for_tool("shell__exec", &json!({"command": " cargo test "})),
            permission_resource_for_tool("shell__exec", &json!({"command": "cargo test"}))
        );
        assert_eq!(
            permission_resource_for_tool("custom", &json!({"b": 1, "a": {"z": 2, "y": 3}})),
            permission_resource_for_tool("custom", &json!({"a": {"y": 3, "z": 2}, "b": 1}))
        );
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

    #[cfg(unix)]
    #[tokio::test]
    async fn writable_leaf_symlink_authorization_uses_canonical_destination() {
        use std::os::unix::fs::symlink;

        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time")
            .as_nanos();
        let outside = std::env::temp_dir().join(format!("letcode-writable-leaf-{unique}"));
        let link = PathBuf::from("target").join(format!("letcode-writable-link-{unique}"));
        std::fs::create_dir_all(&outside).expect("create outside fixture");
        std::fs::create_dir_all("target").expect("create target fixture");
        symlink(&outside, &link).expect("create leaf parent symlink");
        let raw = link.join("written.txt").to_string_lossy().to_string();
        let args = json!({"path": raw, "content": "one"});
        let prepared = prepare_writable_leaf(args["path"].as_str().expect("path"))
            .expect("prepare writable leaf");
        let access = prepared
            .external_workspace_access()
            .expect("external writable access");
        let resource = prepared.permission_resource("fs__write");
        assert_eq!(
            access.paths,
            vec![
                outside
                    .canonicalize()
                    .expect("canonical outside fixture")
                    .join("written.txt")
                    .display()
                    .to_string()
            ]
        );
        assert_eq!(
            resource,
            permission_resource_for_tool("fs__write", &args).expect("write resource")
        );
        assert_eq!(
            external_workspace_access_for_tool("fs__write", &args),
            Some(access)
        );

        let registry = ToolRegistry::default_tools();
        let denied = registry.call("fs__write", args.clone()).await;
        assert!(!denied.ok, "{denied:?}");
        assert!(!outside.join("written.txt").exists());

        let written = registry
            .call_with_context(
                "fs__write",
                args,
                ToolExecutionContext::outside_workspace_granted(),
            )
            .await;
        assert!(written.ok, "{:?}", written.error);
        let appended = registry
            .call_with_context(
                "fs__append",
                json!({"path": link.join("written.txt").to_string_lossy(), "content": " two"}),
                ToolExecutionContext::outside_workspace_granted(),
            )
            .await;
        assert!(appended.ok, "{:?}", appended.error);
        assert_eq!(
            std::fs::read_to_string(outside.join("written.txt")).unwrap(),
            "one two"
        );

        let _ = std::fs::remove_file(link);
        let _ = std::fs::remove_dir_all(outside);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn writable_leaf_symlinks_require_grants_and_preserve_the_resolved_leaf_identity() {
        use std::os::unix::fs::symlink;

        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time")
            .as_nanos();
        let outside = std::env::temp_dir().join(format!("letcode-writable-leaf-target-{unique}"));
        let links = PathBuf::from("target").join(format!("letcode-writable-leaf-links-{unique}"));
        std::fs::create_dir_all(&outside).expect("create outside fixture");
        std::fs::create_dir_all(&links).expect("create links fixture");

        for (name, target, expected) in [
            ("existing", outside.join("existing.txt"), "written appended"),
            ("dangling", outside.join("dangling.txt"), "written appended"),
        ] {
            let target = outside
                .canonicalize()
                .expect("canonical outside fixture")
                .join(target.file_name().expect("target leaf"));
            if name == "existing" {
                std::fs::write(&target, "old").expect("create existing target");
            }
            let link = links.join(name);
            symlink(&target, &link).expect("create leaf symlink");
            let args = json!({"path": link.to_string_lossy(), "content": "written"});
            let prepared = prepare_writable_leaf(args["path"].as_str().expect("path"))
                .expect("prepare resolved leaf");
            assert_eq!(prepared.destination, target);
            assert_eq!(
                prepared.permission_resource("fs__write"),
                permission_resource_for_tool("fs__write", &args).expect("resource")
            );
            assert!(
                prepared
                    .external_workspace_access()
                    .expect("external access")
                    .preview()
                    .contains(outside.to_string_lossy().as_ref())
            );

            let registry = ToolRegistry::default_tools();
            let denied = registry.call("fs__write", args.clone()).await;
            assert!(!denied.ok);
            if name == "existing" {
                assert_eq!(
                    std::fs::read_to_string(&target).expect("read unchanged target"),
                    "old"
                );
            } else {
                assert!(!target.exists());
            }

            let written = registry
                .call_with_context(
                    "fs__write",
                    args,
                    ToolExecutionContext::outside_workspace_granted(),
                )
                .await;
            assert!(written.ok, "{:?}", written.error);
            let appended = registry
                .call_with_context(
                    "fs__append",
                    json!({"path": link.to_string_lossy(), "content": " appended"}),
                    ToolExecutionContext::outside_workspace_granted(),
                )
                .await;
            assert!(appended.ok, "{:?}", appended.error);
            assert_eq!(
                std::fs::read_to_string(&target).expect("read target"),
                expected
            );
            std::fs::remove_file(link).expect("remove link");
        }

        let _ = std::fs::remove_dir_all(links);
        let _ = std::fs::remove_dir_all(outside);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn direct_writable_leaf_matrix_covers_external_and_internal_leaf_symlinks() {
        use std::os::unix::fs::symlink;

        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time")
            .as_nanos();
        let outside = std::env::temp_dir().join(format!("letcode-direct-leaf-{unique}"));
        let links = PathBuf::from("target").join(format!("letcode-direct-links-{unique}"));
        std::fs::create_dir_all(&outside).expect("create outside fixture");
        std::fs::create_dir_all(&links).expect("create links fixture");
        let registry = ToolRegistry::default_tools();

        for (tool, initial, expected) in [
            ("fs__write", "old", "written"),
            ("fs__append", "old", "oldwritten"),
        ] {
            let target = outside.join(format!("existing-{tool}"));
            std::fs::write(&target, initial).expect("create external target");
            let link = links.join(format!("existing-{tool}"));
            symlink(&target, &link).expect("create external link");
            let args = json!({"path": link.to_string_lossy(), "content": "written"});
            assert!(
                !registry.call(tool, args.clone()).await.ok,
                "{tool} without grant"
            );
            assert_eq!(std::fs::read_to_string(&target).unwrap(), initial);
            assert!(
                registry
                    .call_with_context(
                        tool,
                        args,
                        ToolExecutionContext::outside_workspace_granted()
                    )
                    .await
                    .ok
            );
            assert_eq!(std::fs::read_to_string(&target).unwrap(), expected);
            std::fs::remove_file(link).unwrap();

            let dangling_target = outside.join(format!("dangling-{tool}"));
            let dangling = links.join(format!("dangling-{tool}"));
            symlink(&dangling_target, &dangling).expect("create dangling external link");
            let args = json!({"path": dangling.to_string_lossy(), "content": "written"});
            assert!(
                !registry.call(tool, args.clone()).await.ok,
                "{tool} dangling without grant"
            );
            assert!(!dangling_target.exists());
            assert!(
                registry
                    .call_with_context(
                        tool,
                        args,
                        ToolExecutionContext::outside_workspace_granted()
                    )
                    .await
                    .ok
            );
            assert_eq!(
                std::fs::read_to_string(&dangling_target).unwrap(),
                "written"
            );
            std::fs::remove_file(dangling).unwrap();
        }

        let internal_target = links.join("internal-target");
        std::fs::write(&internal_target, "old").unwrap();
        let relative = links.join("relative");
        let hop = links.join("hop");
        symlink("internal-target", &relative).unwrap();
        symlink("relative", &hop).unwrap();
        for (tool, expected) in [("fs__write", "written"), ("fs__append", "writtenwritten")] {
            assert!(
                registry
                    .call(
                        tool,
                        json!({"path": hop.to_string_lossy(), "content": "written"})
                    )
                    .await
                    .ok
            );
            assert_eq!(std::fs::read_to_string(&internal_target).unwrap(), expected);
        }
        let malformed = registry
            .call(
                "fs__write",
                json!({"path": format!("target/missing-{unique}/leaf"), "content": "x"}),
            )
            .await;
        assert!(!malformed.ok);
        assert!(
            malformed
                .error
                .unwrap()
                .message
                .contains("parent directory does not exist")
        );
        let _ = std::fs::remove_dir_all(&links);
        let _ = std::fs::remove_dir_all(&outside);
    }

    #[cfg(unix)]
    #[test]
    fn writable_leaf_loop_and_missing_parent_fail_before_execution() {
        use std::os::unix::fs::symlink;

        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time")
            .as_nanos();
        std::fs::create_dir_all("target").expect("create target fixture");
        let first = PathBuf::from("target").join(format!("letcode-loop-a-{unique}"));
        let second = PathBuf::from("target").join(format!("letcode-loop-b-{unique}"));
        symlink(second.file_name().expect("second sibling"), &first)
            .expect("create first loop link");
        symlink(first.file_name().expect("first sibling"), &second)
            .expect("create second loop link");
        let loop_error =
            prepare_writable_leaf(&first.to_string_lossy()).expect_err("a -> b -> a must loop");
        assert!(loop_error.to_string().contains("too many symlink levels"));
        let missing_error =
            prepare_writable_leaf(&format!("target/letcode-missing-{unique}/child"))
                .expect_err("missing ultimate parent must fail independently");
        assert!(
            missing_error
                .to_string()
                .contains("parent directory does not exist")
        );
        let _ = std::fs::remove_file(first);
        let _ = std::fs::remove_file(second);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn secure_writable_leaf_openat_nofollow_rejects_leaf_swapped_after_preparation() {
        use std::os::unix::fs::symlink;

        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let parent = PathBuf::from("target").join(format!("letcode-nofollow-{unique}"));
        let leaf = parent.join("leaf");
        let target = parent.join("target");
        std::fs::create_dir_all(&parent).unwrap();
        std::fs::write(&leaf, "authorized regular leaf").unwrap();
        std::fs::write(&target, "must stay unchanged").unwrap();
        let prepared =
            prepare_writable_leaf(leaf.to_str().unwrap()).expect("authorize regular leaf");
        std::fs::remove_file(&leaf).unwrap();
        symlink(&target, &leaf).unwrap();

        let error = secure_write_writable_leaf(&prepared, b"must not write", false)
            .await
            .expect_err("openat O_NOFOLLOW must reject the replacement symlink");
        assert_eq!(
            error.to_string(),
            "writable destination changed after authorization"
        );
        assert_eq!(
            std::fs::read_to_string(&target).unwrap(),
            "must stay unchanged"
        );
        let _ = std::fs::remove_dir_all(parent);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn prepared_writable_leaf_rejects_parent_replacement_before_writing() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time")
            .as_nanos();
        let parent = PathBuf::from("target").join(format!("letcode-parent-replace-{unique}"));
        let retired = PathBuf::from("target").join(format!("letcode-parent-retired-{unique}"));
        let raw = parent.join("leaf.txt");
        std::fs::create_dir_all(&parent).expect("create initial parent");
        let prepared = prepare_writable_leaf(raw.to_str().expect("UTF-8 fixture path"))
            .expect("prepare writable leaf");
        std::fs::rename(&parent, &retired).expect("replace authorized parent");
        std::fs::create_dir(&parent).expect("create replacement parent");
        let mut context = ToolExecutionContext::default();
        context.attach_prepared_writable_leaf(prepared);

        let result = ToolRegistry::default_tools()
            .call_with_context(
                "fs__write",
                json!({"path": raw, "content": "must not write"}),
                context,
            )
            .await;
        assert!(!result.ok);
        assert_eq!(
            result.error.as_ref().map(|error| error.message.as_str()),
            Some("writable destination changed after authorization")
        );
        assert!(!parent.join("leaf.txt").exists());
        assert!(!retired.join("leaf.txt").exists());
        let _ = std::fs::remove_dir_all(parent);
        let _ = std::fs::remove_dir_all(retired);
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
            "agent__explore",
            "agent__fixer",
            "agent__reconcile",
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

        for retired in [
            "context__checkpoint",
            "context__return",
            "context__list",
            "context__search",
            "context__grep",
            "context__open",
            "context__summarize",
            "context__pin",
            "context__archive",
            "context__remove",
            "context__resolve",
        ] {
            assert!(
                !names.contains(&retired),
                "context tool is exposed while history-only: {retired}"
            );
        }
    }

    #[tokio::test]
    async fn unknown_tool_scope_rejection_precedes_lookup() {
        let name = "missing__tool";
        let read_only = ToolRegistry::default_tools().scoped(ToolScope::ReadOnlyExplorer);
        let rejected = read_only.call(name, json!({})).await;
        assert_eq!(
            rejected.error.as_ref().expect("scope error").message,
            "tool 'missing__tool' is not allowed in read_only_explorer scope"
        );

        let unknown = ToolRegistry::default_tools()
            .scoped(ToolScope::FullAccess)
            .call(name, json!({}))
            .await;
        assert_eq!(
            unknown.error.as_ref().expect("unknown tool error").message,
            "unknown tool: missing__tool"
        );
    }

    #[test]
    fn default_tool_specs_are_in_btree_map_lexical_order() {
        let names = ToolRegistry::default_tools()
            .specs()
            .into_iter()
            .map(|spec| spec.name)
            .collect::<Vec<_>>();
        let expected = vec![
            "agent__explore".to_string(),
            "agent__fixer".to_string(),
            "agent__reconcile".to_string(),
            "code__ast_replace_preview".to_string(),
            "code__ast_search".to_string(),
            "edit__apply_patch".to_string(),
            "fs__append".to_string(),
            "fs__list".to_string(),
            "fs__mkdir".to_string(),
            "fs__read".to_string(),
            "fs__write".to_string(),
            "git__diff".to_string(),
            "git__log".to_string(),
            "git__status".to_string(),
            "memory__recall".to_string(),
            "question".to_string(),
            "search__rg".to_string(),
            "shell__exec".to_string(),
            "util__echo".to_string(),
            "workflow__auto_continue".to_string(),
            "workflow__todos".to_string(),
        ];

        assert_eq!(names, expected);
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
        let too_many_items = (0..=100)
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
                .contains("at most 100 items")
        );

        let auto_output = tools
            .call(
                "workflow__auto_continue",
                json!({"enabled": true, "max_continuations": 17}),
            )
            .await;
        assert!(!auto_output.ok);
        assert!(
            auto_output
                .error
                .as_ref()
                .expect("auto-continue error")
                .message
                .contains("<= 16")
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

        let output = tools
            .call(
                "agent__reconcile",
                json!({
                    "run_id": "run-1",
                    "child_session_id": "child-1",
                    "agent_name": "explorer",
                    "decision": "accepted",
                    "summary": "accepted child result"
                }),
            )
            .await;
        assert!(!output.ok);
        assert_eq!(
            output.error.as_ref().expect("scope error").message,
            "tool 'agent__reconcile' is not allowed in read_only_explorer scope"
        );
    }

    #[tokio::test]
    async fn agent_reconcile_accepts_valid_payload() {
        let output = ToolRegistry::default_tools()
            .call(
                tool_names::TOOL_AGENT_RECONCILE,
                json!({
                    "run_id": " run-1 ",
                    "child_session_id": " child-1 ",
                    "agent_name": "explorer",
                    "decision": "accepted",
                    "summary": " absorbed the child findings "
                }),
            )
            .await;

        assert!(output.ok, "{output:?}");
        let data = output.data.expect("reconcile data");
        assert_eq!(data["run_id"], json!("run-1"));
        assert_eq!(data["child_session_id"], json!("child-1"));
        assert_eq!(data["agent_name"], json!("explorer"));
        assert_eq!(data["decision"], json!("accepted"));
        assert_eq!(data["summary"], json!("absorbed the child findings"));
        assert_eq!(data["reconciled"], json!(true));
        assert_eq!(data["pending_recording"], json!(true));
    }

    #[tokio::test]
    async fn agent_reconcile_rejects_invalid_payload() {
        let output = ToolRegistry::default_tools()
            .call(
                tool_names::TOOL_AGENT_RECONCILE,
                json!({
                    "run_id": "run-1",
                    "child_session_id": "child-1",
                    "agent_name": "explorer",
                    "decision": "merge",
                    "summary": "ok"
                }),
            )
            .await;
        assert!(!output.ok);
        assert!(
            output
                .error
                .as_ref()
                .expect("reconcile error")
                .message
                .contains("field 'decision' must be one of accepted, rejected, conflict")
        );

        let output = ToolRegistry::default_tools()
            .call(
                tool_names::TOOL_AGENT_RECONCILE,
                json!({
                    "run_id": "run-1",
                    "child_session_id": "child-1",
                    "agent_name": "unknown",
                    "decision": "accepted",
                    "summary": "ok"
                }),
            )
            .await;
        assert!(!output.ok);
        assert!(
            output
                .error
                .as_ref()
                .expect("reconcile error")
                .message
                .contains("field 'agent_name' must be one of")
        );
    }

    #[test]
    fn apply_patch_tool_spec_describes_batch_prevalidation_and_non_transactional_writes() {
        let apply_patch = ToolRegistry::default_tools()
            .specs()
            .into_iter()
            .find(|spec| spec.name == "edit__apply_patch")
            .expect("apply patch tool is registered");

        assert_eq!(
            apply_patch.description,
            "Apply exact-match text replacements to existing UTF-8 files under the workspace. Each edit must provide the exact old text in `find` and replacement text in `replace`. By default use replace_all=false so the tool fails unless `find` matches exactly once. All edits are first validated against staged in-memory content before any file is written. After validation, files are written individually and non-transactionally, so I/O, timeout, cancellation, or process failure can leave previously written files changed. This is intended for precise, low-ambiguity code edits."
        );
        assert_eq!(
            apply_patch.parameters["properties"]["edits"]["description"],
            json!(
                "Exact-match replacement edits. All edits are validated against staged in-memory content before any file is written. After validation, files are written individually and non-transactionally, so I/O, timeout, cancellation, or process failure can leave previously written files changed"
            )
        );
        assert!(
            !apply_patch
                .description
                .to_ascii_lowercase()
                .contains("atomic")
        );
        assert!(
            !apply_patch.parameters["properties"]["edits"]["description"]
                .as_str()
                .expect("edits description is a string")
                .to_ascii_lowercase()
                .contains("atomic")
        );
    }

    #[tokio::test]
    async fn apply_patch_validation_failure_leaves_all_files_unchanged() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time")
            .as_nanos();
        let fixture =
            PathBuf::from("target").join(format!("letcode-apply-patch-validation-{unique}"));
        let first = fixture.join("first.txt");
        let second = fixture.join("second.txt");
        std::fs::create_dir_all(&fixture).expect("create fixture directory");
        std::fs::write(&first, "first original").expect("write first fixture");
        std::fs::write(&second, "second original").expect("write second fixture");

        let result = ToolRegistry::default_tools()
            .call(
                "edit__apply_patch",
                json!({
                    "edits": [
                        {
                            "path": first,
                            "find": "first original",
                            "replace": "first changed",
                            "replace_all": false
                        },
                        {
                            "path": second,
                            "find": "missing text",
                            "replace": "second changed",
                            "replace_all": false
                        }
                    ]
                }),
            )
            .await;

        assert!(!result.ok, "{result:?}");
        assert_eq!(std::fs::read_to_string(&first).unwrap(), "first original");
        assert_eq!(std::fs::read_to_string(&second).unwrap(), "second original");
        let _ = std::fs::remove_dir_all(fixture);
    }

    #[tokio::test]
    async fn apply_patch_applies_same_path_edits_to_staged_content_in_input_order() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time")
            .as_nanos();
        let fixture = PathBuf::from("target").join(format!("letcode-apply-patch-staged-{unique}"));
        let file = fixture.join("file.txt");
        std::fs::create_dir_all(&fixture).expect("create fixture directory");
        std::fs::write(&file, "first").expect("write fixture");

        let result = ToolRegistry::default_tools()
            .call(
                "edit__apply_patch",
                json!({
                    "edits": [
                        {
                            "path": file,
                            "find": "first",
                            "replace": "second",
                            "replace_all": false
                        },
                        {
                            "path": file,
                            "find": "second",
                            "replace": "third",
                            "replace_all": false
                        }
                    ]
                }),
            )
            .await;

        assert!(result.ok, "{result:?}");
        assert_eq!(
            result.data,
            Some(json!({
                "files_changed": 1,
                "edits_applied": 2,
                "edits": [
                    {"path": file, "replacements": 1, "replace_all": false},
                    {"path": file, "replacements": 1, "replace_all": false}
                ]
            }))
        );
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "third");
        let _ = std::fs::remove_dir_all(fixture);
    }

    #[cfg(unix)]
    #[test]
    fn apply_patch_preparation_rejects_more_than_64_canonical_targets_before_anchors() {
        let fixture =
            std::env::temp_dir().join(format!("letcode-apply-patch-cap-{}", std::process::id()));
        std::fs::create_dir_all(&fixture).expect("fixture directory");
        let edits: Vec<_> = (0..65)
            .map(|index| {
                let path = fixture.join(format!("{index}.txt"));
                std::fs::write(&path, "old").expect("fixture file");
                json!({"path": path, "find": "old", "replace": "new", "replace_all": false})
            })
            .collect();
        let error =
            prepare_apply_patch_targets(&json!({"edits": edits})).expect_err("cap must reject");
        assert_eq!(
            error.to_string(),
            "apply patch accepts at most 64 unique target files"
        );
        let _ = std::fs::remove_dir_all(fixture);
    }

    #[cfg(unix)]
    #[test]
    fn apply_patch_preparation_rejects_batch_hardlink_aliases() {
        let fixture = std::env::temp_dir().join(format!(
            "letcode-apply-patch-hardlink-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&fixture).expect("fixture directory");
        let first = fixture.join("first.txt");
        let second = fixture.join("second.txt");
        std::fs::write(&first, "old").expect("fixture file");
        std::fs::hard_link(&first, &second).expect("hardlink");
        let error = prepare_apply_patch_targets(&json!({"edits": [
            {"path": first, "find": "old", "replace": "new", "replace_all": false},
            {"path": second, "find": "old", "replace": "new", "replace_all": false}
        ]}))
        .expect_err("hardlink aliases must reject");
        assert_eq!(
            error.to_string(),
            "apply patch targets alias the same existing file"
        );
        let _ = std::fs::remove_dir_all(fixture);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn apply_patch_bound_execution_rejects_replaced_leaf_without_writing() {
        let fixture = std::env::temp_dir().join(format!(
            "letcode-apply-patch-replacement-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&fixture).expect("fixture directory");
        let file = fixture.join("file.txt");
        let replacement = fixture.join("replacement.txt");
        std::fs::write(&file, "old").expect("fixture file");
        let args = json!({"edits": [{"path": file, "find": "old", "replace": "new", "replace_all": false}]});
        let prepared = prepare_apply_patch_targets(&args).expect("prepare binding");
        std::fs::write(&replacement, "replacement").expect("replacement file");
        std::fs::rename(&replacement, &file).expect("replace leaf");
        let mut context = ToolExecutionContext::outside_workspace_granted();
        context.attach_prepared_apply_patch(prepared);
        let result = ToolRegistry::default_tools()
            .call_with_context("edit__apply_patch", args, context)
            .await;
        assert_eq!(
            result.error.expect("error").message,
            "apply patch target changed after authorization"
        );
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "replacement");
        let _ = std::fs::remove_dir_all(fixture);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn apply_patch_attached_binding_rejects_unresolvable_rebind_without_preparing_it() {
        let fixture = std::env::temp_dir().join(format!(
            "letcode-apply-patch-lazy-binding-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&fixture).expect("fixture directory");
        let file = fixture.join("file.txt");
        std::fs::write(&file, "old").expect("fixture file");
        let authorized = json!({"edits": [{"path": file, "find": "old", "replace": "new", "replace_all": false}]});
        let prepared = prepare_apply_patch_targets(&authorized).expect("prepare binding");
        let mut context = ToolExecutionContext::outside_workspace_granted();
        context.attach_prepared_apply_patch(prepared);
        let result = ToolRegistry::default_tools()
            .call_with_context(
                "edit__apply_patch",
                json!({"edits": [{
                    "path": fixture.join("missing").join("target.txt"),
                    "find": "old",
                    "replace": "new",
                    "replace_all": false
                }]}),
                context,
            )
            .await;
        assert_eq!(
            result.error.expect("error").message,
            "apply patch target changed after authorization"
        );
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "old");
        let _ = std::fs::remove_dir_all(fixture);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn apply_patch_bound_execution_rejects_fifo_replacement_without_blocking_or_writing() {
        use std::ffi::CString;
        use std::os::unix::ffi::OsStrExt;
        use std::os::unix::fs::FileTypeExt;

        let fixture = std::env::temp_dir().join(format!(
            "letcode-apply-patch-fifo-replacement-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&fixture).expect("fixture directory");
        let file = fixture.join("file.txt");
        std::fs::write(&file, "old").expect("fixture file");
        let args = json!({"edits": [{"path": file, "find": "old", "replace": "new", "replace_all": false}]});
        let prepared = prepare_apply_patch_targets(&args).expect("prepare binding");
        std::fs::remove_file(&file).expect("remove approved leaf");
        let fifo = CString::new(file.as_os_str().as_bytes()).expect("fifo path");
        assert_eq!(
            unsafe { libc::mkfifo(fifo.as_ptr(), 0o600) },
            0,
            "create fifo"
        );
        let mut context = ToolExecutionContext::outside_workspace_granted();
        context.attach_prepared_apply_patch(prepared);
        let result = ToolRegistry::default_tools()
            .call_with_context("edit__apply_patch", args, context)
            .await;
        assert_eq!(
            result.error.expect("error").message,
            "apply patch target changed after authorization"
        );
        assert!(
            std::fs::metadata(&file)
                .expect("fifo metadata")
                .file_type()
                .is_fifo()
        );
        let _ = std::fs::remove_dir_all(fixture);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn apply_patch_bound_execution_classifies_same_size_mutation_as_concurrent() {
        let fixture = std::env::temp_dir().join(format!(
            "letcode-apply-patch-version-mutation-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&fixture).expect("fixture directory");
        let file = fixture.join("file.txt");
        std::fs::write(&file, "old").expect("fixture file");
        let args = json!({"edits": [{"path": file, "find": "old", "replace": "new", "replace_all": false}]});
        let prepared = prepare_apply_patch_targets(&args).expect("prepare binding");
        std::fs::write(&file, "new").expect("same-size mutation");
        let mut context = ToolExecutionContext::outside_workspace_granted();
        context.attach_prepared_apply_patch(prepared);
        let result = ToolRegistry::default_tools()
            .call_with_context("edit__apply_patch", args, context)
            .await;
        assert_eq!(
            result.error.expect("error").message,
            "apply patch target was concurrently modified"
        );
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "new");
        let _ = std::fs::remove_dir_all(fixture);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn apply_patch_direct_context_requires_external_grant_and_supports_mixed_targets() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time")
            .as_nanos();
        let internal =
            PathBuf::from("target").join(format!("letcode-apply-patch-internal-{unique}"));
        let external = std::env::temp_dir().join(format!("letcode-apply-patch-external-{unique}"));
        std::fs::write(&internal, "old").expect("internal fixture");
        std::fs::write(&external, "old").expect("external fixture");
        let args = json!({"edits": [
            {"path": internal, "find": "old", "replace": "internal", "replace_all": false},
            {"path": external, "find": "old", "replace": "external", "replace_all": false}
        ]});

        let denied = ToolRegistry::default_tools()
            .call("edit__apply_patch", args.clone())
            .await;
        assert!(!denied.ok);
        assert_eq!(std::fs::read_to_string(&internal).unwrap(), "old");
        assert_eq!(std::fs::read_to_string(&external).unwrap(), "old");

        let granted = ToolRegistry::default_tools()
            .call_with_context(
                "edit__apply_patch",
                args,
                ToolExecutionContext::outside_workspace_granted(),
            )
            .await;
        assert!(granted.ok, "{granted:?}");
        assert_eq!(std::fs::read_to_string(&internal).unwrap(), "internal");
        assert_eq!(std::fs::read_to_string(&external).unwrap(), "external");
        let _ = std::fs::remove_file(internal);
        let _ = std::fs::remove_file(external);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn apply_patch_symlink_aliases_share_one_canonical_staged_target() {
        use std::os::unix::fs::symlink;

        let fixture =
            std::env::temp_dir().join(format!("letcode-apply-patch-alias-{}", std::process::id()));
        std::fs::create_dir_all(&fixture).unwrap();
        let target = fixture.join("target.txt");
        let first = fixture.join("first");
        let second = fixture.join("second");
        std::fs::write(&target, "old").unwrap();
        symlink(&target, &first).unwrap();
        symlink(&target, &second).unwrap();
        let result = ToolRegistry::default_tools()
            .call_with_context(
                "edit__apply_patch",
                json!({"edits": [
                    {"path": first, "find": "old", "replace": "middle", "replace_all": false},
                    {"path": second, "find": "middle", "replace": "new", "replace_all": false}
                ]}),
                ToolExecutionContext::outside_workspace_granted(),
            )
            .await;
        assert!(result.ok, "{result:?}");
        assert_eq!(result.data.as_ref().unwrap()["files_changed"], 1);
        assert_eq!(std::fs::read_to_string(&target).unwrap(), "new");
        let _ = std::fs::remove_dir_all(fixture);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn apply_patch_staging_and_precommit_mutations_leave_every_target_unchanged_by_patch() {
        for point in [
            ApplyPatchWorkerPoint::BeforeStagingValidation,
            ApplyPatchWorkerPoint::BeforeBatchPrecommit,
        ] {
            let fixture = std::env::temp_dir().join(format!(
                "letcode-apply-patch-barrier-{}-{:?}",
                std::process::id(),
                point
            ));
            std::fs::create_dir_all(&fixture).unwrap();
            let first = fixture.join("first.txt");
            let second = fixture.join("second.txt");
            std::fs::write(&first, "first old").unwrap();
            std::fs::write(&second, "second old").unwrap();
            let args = json!({"edits": [
                {"path": first, "find": "old", "replace": "new", "replace_all": false},
                {"path": second, "find": "old", "replace": "new", "replace_all": false}
            ]});
            let prepared = prepare_apply_patch_targets(&args).unwrap();
            let mutate = second.clone();
            prepared.set_worker_hook(point, move || {
                std::fs::write(mutate, "second mutated").unwrap();
            });
            let mut context = ToolExecutionContext::outside_workspace_granted();
            context.attach_prepared_apply_patch(prepared);
            let result = ToolRegistry::default_tools()
                .call_with_context("edit__apply_patch", args, context)
                .await;
            assert!(!result.ok);
            assert_eq!(std::fs::read_to_string(&first).unwrap(), "first old");
            assert_eq!(std::fs::read_to_string(&second).unwrap(), "second mutated");
            let _ = std::fs::remove_dir_all(fixture);
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn apply_patch_commit_rechecks_nonregular_replacements_and_preserves_prior_commit_on_io_error()
     {
        use std::ffi::CString;
        use std::os::unix::{ffi::OsStrExt, fs::PermissionsExt};

        let fixture =
            std::env::temp_dir().join(format!("letcode-apply-patch-commit-{}", std::process::id()));
        std::fs::create_dir_all(&fixture).unwrap();
        let fifo = fixture.join("fifo.txt");
        std::fs::write(&fifo, "old").unwrap();
        let fifo_args = json!({"edits": [{"path": fifo, "find": "old", "replace": "new", "replace_all": false}]});
        let prepared = prepare_apply_patch_targets(&fifo_args).unwrap();
        let fifo_for_hook = fifo.clone();
        prepared.set_worker_hook(
            ApplyPatchWorkerPoint::BeforeCommitOpen {
                path: fifo.canonicalize().unwrap(),
                ordinal: 0,
            },
            move || {
                std::fs::remove_file(&fifo_for_hook).unwrap();
                let path = CString::new(fifo_for_hook.as_os_str().as_bytes()).unwrap();
                assert_eq!(unsafe { libc::mkfifo(path.as_ptr(), 0o600) }, 0);
            },
        );
        let mut context = ToolExecutionContext::outside_workspace_granted();
        context.attach_prepared_apply_patch(prepared);
        let result = ToolRegistry::default_tools()
            .call_with_context("edit__apply_patch", fifo_args, context)
            .await;
        assert_eq!(
            result.error.unwrap().message,
            "apply patch target changed after authorization"
        );

        let directory = fixture.join("directory.txt");
        std::fs::write(&directory, "old").unwrap();
        let directory_args = json!({"edits": [{"path": directory, "find": "old", "replace": "new", "replace_all": false}]});
        let prepared = prepare_apply_patch_targets(&directory_args).unwrap();
        let directory_for_hook = directory.clone();
        prepared.set_worker_hook(
            ApplyPatchWorkerPoint::BeforeCommitOpen {
                path: directory.canonicalize().unwrap(),
                ordinal: 0,
            },
            move || {
                std::fs::remove_file(&directory_for_hook).unwrap();
                std::fs::create_dir(&directory_for_hook).unwrap();
            },
        );
        let mut context = ToolExecutionContext::outside_workspace_granted();
        context.attach_prepared_apply_patch(prepared);
        let result = ToolRegistry::default_tools()
            .call_with_context("edit__apply_patch", directory_args, context)
            .await;
        assert_eq!(
            result.error.unwrap().message,
            "apply patch target changed after authorization"
        );

        let first = fixture.join("a-first.txt");
        let later = fixture.join("b-later.txt");
        std::fs::write(&first, "old").unwrap();
        std::fs::write(&later, "old").unwrap();
        let args = json!({"edits": [
            {"path": first, "find": "old", "replace": "new", "replace_all": false},
            {"path": later, "find": "old", "replace": "new", "replace_all": false}
        ]});
        let prepared = prepare_apply_patch_targets(&args).unwrap();
        let later_for_hook = later.clone();
        prepared.set_worker_hook(
            ApplyPatchWorkerPoint::BeforeCommitOpen {
                path: later.canonicalize().unwrap(),
                ordinal: 1,
            },
            move || {
                std::fs::set_permissions(&later_for_hook, std::fs::Permissions::from_mode(0o000))
                    .unwrap();
            },
        );
        let mut context = ToolExecutionContext::outside_workspace_granted();
        context.attach_prepared_apply_patch(prepared);
        let result = ToolRegistry::default_tools()
            .call_with_context("edit__apply_patch", args, context)
            .await;
        let error = result.error.expect("later I/O error").message;
        assert!(error.contains(later.to_string_lossy().as_ref()), "{error}");
        assert_ne!(error, "apply patch target changed after authorization");
        assert_eq!(std::fs::read_to_string(&first).unwrap(), "new");
        std::fs::set_permissions(&later, std::fs::Permissions::from_mode(0o600)).unwrap();
        assert_eq!(std::fs::read_to_string(&later).unwrap(), "old");
        let _ = std::fs::remove_dir_all(fixture);
    }
}
