use anyhow::{Context, Result, anyhow, bail};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::BTreeSet;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;
#[cfg(unix)]
use std::{
    fs::File,
    os::fd::{AsRawFd, FromRawFd, OwnedFd},
};
#[cfg(unix)]
use tokio::io::AsyncWriteExt;

use crate::permission::{PermissionResource, ToolPermissionClass, classify_tool, path_preview};
use crate::request_builder::ToolSpec;
use crate::tool_names;
use paths::{
    canonical_destination_path, canonical_existing_path, ensure_inside_workspace,
    join_workspace_path, outside_existing_workspace_path, outside_new_workspace_path,
    workspace_root,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolParallelism {
    Parallel,
    Exclusive,
}

mod apply_patch;
mod args;
mod code_analysis;
mod command;
mod config_validate;
mod delegation;
mod fold_artifact;
pub(crate) use fold_artifact::is_trusted_artifact_path;
mod fs;
mod git;
mod memory;
mod paths;
mod question;
mod registry;
mod search;
mod web_fetch;
mod workflow;

pub(crate) use apply_patch::{PreparedApplyPatch, prepare_apply_patch_targets};
pub use delegation::NormalizedSubagentInput;
pub(crate) use delegation::{
    SubagentPathScope, delegation_scope_denial, is_delegation_path_scoped_tool,
    normalize_subagent_input, subagent_parameters_schema,
};
pub use registry::ToolRegistry;

const DEFAULT_READ_LINE_LIMIT: usize = 200;
const MAX_READ_LINE_LIMIT: usize = 5_000;
const MAX_READ_BYTES: usize = 4 * 1024 * 1024;
const MAX_READ_IMAGE_BYTES: u64 = 20 * 1024 * 1024;
const COMMAND_TIMEOUT_SECS: u64 = 300;
const MAX_SUBAGENT_TEXT_FIELD_CHARS: usize = 16_000;
const MAX_SUBAGENT_LIST_ITEMS: usize = 128;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub ok: bool,
    pub tool: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub images: Vec<crate::user_content::UserImageAttachment>,

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
            images: Vec::new(),
            error: None,
        }
    }

    pub fn err(tool: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            ok: false,
            tool: tool.into(),
            data: None,
            images: Vec::new(),
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
            images: Vec::new(),
            error: Some(ToolError {
                message: message.into(),
                recoverable: true,
            }),
        }
    }

    pub fn with_images(mut self, images: Vec<crate::user_content::UserImageAttachment>) -> Self {
        self.images = images;
        self
    }

    pub fn for_text_history(&self) -> Self {
        let mut output = self.clone();
        output.images.clear();
        output
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

    /// Execution is exclusive unless this concrete handler has been reviewed
    /// and explicitly opts in to overlapping calls.
    fn parallelism(&self) -> ToolParallelism {
        ToolParallelism::Exclusive
    }

    async fn execute(&self, args: Value) -> Result<Value>;

    async fn execute_tool_result(
        &self,
        args: Value,
        context: ToolExecutionContext,
    ) -> Result<ToolResult> {
        self.execute_with_context(args, context)
            .await
            .map(|data| ToolResult::ok(self.name(), data))
    }

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
    ) -> Result<ToolResult> {
        self.execute_tool_result(args, context).await
    }

    fn spec(&self) -> ToolSpec {
        let parameters = self.parameters();
        let strict = self.strict();
        if strict {
            let missing = strict_parameter_violations(&parameters).unwrap_or_default();
            debug_assert!(
                missing.is_empty(),
                "strict tool schema for '{}' is missing required entries for properties: {missing:?}",
                self.name()
            );
        }
        ToolSpec {
            name: self.name().to_string(),
            description: self.description().to_string(),
            parameters,
            strict,
        }
    }
}

/// OpenAI structured/strict function tools reject schemas where `required` omits
/// any key from `properties` (optional fields must still be listed and allow null).
/// Returns `None` when the schema has no `properties` object (nothing to verify).
fn strict_parameter_violations(parameters: &Value) -> Option<Vec<String>> {
    let properties = parameters.get("properties").and_then(Value::as_object)?;
    let required = parameters
        .get("required")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .collect::<BTreeSet<_>>()
        })
        .unwrap_or_default();
    Some(
        properties
            .keys()
            .filter(|key| !required.contains(key.as_str()))
            .cloned()
            .collect(),
    )
}

#[cfg(test)]
fn assert_strict_tool_parameters(tool_name: &str, parameters: &Value) {
    let missing = strict_parameter_violations(parameters)
        .unwrap_or_else(|| panic!("{tool_name}: parameters.properties must be an object"));
    assert!(
        missing.is_empty(),
        "{tool_name}: strict schema required must include every properties key; missing {missing:?}"
    );
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
    pub question_handler: Option<QuestionCallback>,
    prepared_writable_leaf: Option<PreparedWritableLeaf>,
    prepared_apply_patch: Option<PreparedApplyPatch>,
}

impl std::fmt::Debug for ToolExecutionContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolExecutionContext")
            .field("allow_outside_workspace", &self.allow_outside_workspace)
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

    pub(crate) fn prepared_apply_patch(&self) -> Option<&PreparedApplyPatch> {
        self.prepared_apply_patch.as_ref()
    }
}

const WRITABLE_DESTINATION_CHANGED: &str = "writable destination changed after authorization";

/// A canonical, authorization-time binding for one writable filesystem leaf.
#[derive(Clone)]
pub(crate) struct PreparedWritableLeaf {
    workspace_root: PathBuf,
    destination: PathBuf,
    parent: PathBuf,
    #[cfg(unix)]
    leaf: std::ffi::OsString,
    #[cfg(unix)]
    parent_dev: u64,
    #[cfg(unix)]
    parent_ino: u64,
    #[cfg(windows)]
    parent_dir: Arc<cap_std::fs::Dir>,
    #[cfg(windows)]
    existing_file: Option<Arc<cap_std::fs::File>>,
}

impl PreparedWritableLeaf {
    pub(crate) fn destination(&self) -> &Path {
        &self.destination
    }

    #[cfg(windows)]
    fn leaf_name(&self) -> std::ffi::OsString {
        self.destination
            .file_name()
            .expect("prepared writable leaf has file name")
            .to_os_string()
    }

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
        let workspace_root = workspace_root()?;
        let candidate = join_workspace_path(&workspace_root, raw_path);
        let destination = current_writable_destination(&candidate)?;
        if workspace_root != self.workspace_root
            || destination != self.destination
            || destination.parent() != Some(self.parent.as_path())
            || !self.parent_instance_is_current()
        {
            bail!(WRITABLE_DESTINATION_CHANGED);
        }
        Ok(())
    }

    #[cfg(unix)]
    fn parent_instance_is_current(&self) -> bool {
        use std::os::unix::fs::MetadataExt;
        std::fs::metadata(&self.parent).is_ok_and(|metadata| {
            metadata.is_dir()
                && metadata.dev() == self.parent_dev
                && metadata.ino() == self.parent_ino
        })
    }

    #[cfg(windows)]
    fn parent_instance_is_current(&self) -> bool {
        // cap-std keeps the directory capability open without FILE_SHARE_DELETE,
        // so Windows cannot replace or rename this parent while authorization is live.
        self.parent_dir.dir_metadata().is_ok()
    }

    #[cfg(not(any(unix, windows)))]
    fn parent_instance_is_current(&self) -> bool {
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
        config_validate::register(&mut registry);
        registry.register(AgentExploreTool);
        registry.register(AgentFixerTool);
        fs::register(&mut registry);
        command::register(&mut registry);
        search::register(&mut registry);
        web_fetch::register(&mut registry);
        git::register(&mut registry);
        apply_patch::register(&mut registry);
        code_analysis::register(&mut registry);
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

    fn parallelism(&self) -> ToolParallelism {
        ToolParallelism::Parallel
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

#[cfg(windows)]
fn prepare_writable_leaf_platform(path: &str) -> Result<PreparedWritableLeaf> {
    let workspace_root = workspace_root()?;
    let candidate = join_workspace_path(&workspace_root, path);
    let destination = resolve_writable_leaf_destination_windows(&candidate)?;
    let parent = destination
        .parent()
        .ok_or_else(|| anyhow!("path has no parent: {}", destination.display()))?
        .to_path_buf();
    if destination.file_name().is_none() {
        bail!("path has no file name: {}", destination.display());
    }
    let parent_dir = Arc::new(
        cap_std::fs::Dir::open_ambient_dir(&parent, cap_std::ambient_authority())
            .with_context(|| format!("failed to open parent directory {}", parent.display()))?,
    );
    let existing_file = match parent_dir.open(destination.file_name().expect("validated file name"))
    {
        Ok(file) => Some(Arc::new(file)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => return Err(error).context("failed to bind writable destination"),
    };
    Ok(PreparedWritableLeaf {
        workspace_root,
        destination,
        parent,
        parent_dir,
        existing_file,
    })
}

#[cfg(not(any(unix, windows)))]
fn prepare_writable_leaf_platform(_path: &str) -> Result<PreparedWritableLeaf> {
    bail!("secure writable leaf authorization is unsupported on this platform")
}

#[cfg(windows)]
fn resolve_writable_leaf_destination_windows(candidate: &Path) -> Result<PathBuf> {
    let parent = candidate
        .parent()
        .ok_or_else(|| anyhow!("path has no parent: {}", candidate.display()))?
        .canonicalize()
        .with_context(|| format!("parent directory does not exist: {}", candidate.display()))?;
    let leaf = candidate
        .file_name()
        .ok_or_else(|| anyhow!("path has no file name: {}", candidate.display()))?;
    let resolved = parent.join(leaf);
    match std::fs::symlink_metadata(&resolved) {
        Ok(metadata) if metadata.file_type().is_symlink() => {
            bail!(
                "writable destination cannot be a link: {}",
                resolved.display()
            )
        }
        Ok(metadata) if windows_metadata_is_reparse_point(&metadata) => {
            bail!(
                "writable destination cannot be a reparse point: {}",
                resolved.display()
            )
        }
        Ok(_) => resolved.canonicalize().with_context(|| {
            format!(
                "failed to canonicalize writable destination: {}",
                resolved.display()
            )
        }),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(resolved),
        Err(error) => Err(error).with_context(|| {
            format!(
                "failed to inspect writable destination: {}",
                resolved.display()
            )
        }),
    }
}

fn current_writable_destination(candidate: &Path) -> Result<PathBuf> {
    #[cfg(unix)]
    {
        resolve_writable_leaf_destination(candidate)
    }
    #[cfg(windows)]
    {
        resolve_writable_leaf_destination_windows(candidate)
    }
    #[cfg(not(any(unix, windows)))]
    {
        Ok(candidate.to_path_buf())
    }
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

#[cfg(windows)]
async fn secure_write_writable_leaf(
    prepared: &PreparedWritableLeaf,
    content: &[u8],
    append: bool,
) -> Result<()> {
    use std::io::Write;

    if let Some(existing_file) = &prepared.existing_file {
        let mut file = existing_file
            .try_clone()
            .map_err(|_| anyhow!(WRITABLE_DESTINATION_CHANGED))?;
        if append {
            use std::io::{Seek, SeekFrom};
            file.seek(SeekFrom::End(0))?;
        } else {
            file.set_len(0)?;
        }
        file.write_all(content)?;
        file.flush()?;
    } else {
        let mut options = cap_std::fs::OpenOptions::new();
        options.write(true).create_new(true);
        let mut file = prepared
            .parent_dir
            .open_with(&prepared.leaf_name(), &options)
            .map_err(|_| anyhow!(WRITABLE_DESTINATION_CHANGED))?;
        file.write_all(content)?;
        file.flush()?;
    }
    Ok(())
}

#[cfg(not(any(unix, windows)))]
async fn secure_write_writable_leaf(
    _prepared: &PreparedWritableLeaf,
    _content: &[u8],
    _append: bool,
) -> Result<()> {
    bail!("secure writable leaf authorization is unsupported on this platform")
}

#[cfg(windows)]
fn windows_metadata_is_reparse_point(metadata: &std::fs::Metadata) -> bool {
    use std::os::windows::fs::MetadataExt;
    use windows_sys::Win32::Storage::FileSystem::FILE_ATTRIBUTE_REPARSE_POINT;
    metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0
}

#[cfg(test)]
mod tests {
    use super::apply_patch::ApplyPatchWorkerPoint;
    use super::{
        NormalizedSubagentInput, SubagentPathScope, ToolExecutionContext, ToolRegistry,
        assert_strict_tool_parameters, external_workspace_access_for_tool,
        normalize_subagent_input, permission_resource_for_tool, prepare_apply_patch_targets,
        prepare_writable_leaf, secure_write_writable_leaf, subagent_parameters_schema,
    };
    use crate::permission::{PermissionResource, ToolScope};
    use crate::skills::{SkillEntry, SkillRegistry, SkillTool};
    use crate::tool::ToolOutputStream;
    use crate::tool_names;
    use base64::Engine as _;
    use serde_json::{Value, json};
    use std::path::PathBuf;
    use std::sync::{Arc, Mutex};

    async fn call_workflow_todos(items: serde_json::Value) -> crate::tool::ToolResult {
        ToolRegistry::default_tools()
            .call("workflow__todos", json!({"items": items}))
            .await
    }

    #[test]
    fn subagent_path_scope_from_input_empty_or_unresolvable() {
        let empty = NormalizedSubagentInput {
            objective: "x".into(),
            success_criteria: Vec::new(),
            allowed_paths: Vec::new(),
            forbidden_paths: Vec::new(),
            owned_paths: Vec::new(),
            timeout_secs: None,
            max_tool_calls: None,
            model: None,
            target_child_session_id: None,
        };
        assert!(
            SubagentPathScope::from_input(&empty)
                .expect("empty scope")
                .is_none()
        );

        let bad = NormalizedSubagentInput {
            owned_paths: vec!["/definitely/missing/letcode-scope-root".into()],
            ..empty
        };
        assert!(SubagentPathScope::from_input(&bad).is_err());
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

    #[tokio::test]
    async fn fs_read_text_output_remains_compatible_and_has_no_images() {
        let path = std::env::temp_dir().join(format!(
            "letcode-read-text-{}.txt",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system time")
                .as_nanos()
        ));
        std::fs::write(&path, "alpha\nbeta\n").expect("write text fixture");

        let output = ToolRegistry::default_tools()
            .call_with_context(
                "fs__read",
                json!({
                    "path": path.to_string_lossy(),
                    "offset": 2,
                    "limit": 1,
                }),
                ToolExecutionContext::outside_workspace_granted(),
            )
            .await;

        assert!(output.ok, "{:?}", output.error);
        assert!(output.images.is_empty());
        assert_eq!(
            output
                .data
                .as_ref()
                .and_then(|data| data.get("content"))
                .and_then(Value::as_str),
            Some("beta\n")
        );
        assert_eq!(
            output
                .data
                .as_ref()
                .and_then(|data| data.get("start_line"))
                .and_then(Value::as_u64),
            Some(2)
        );

        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn fs_read_returns_supported_images_as_multimodal_content() {
        let path = std::env::temp_dir().join(format!(
            "letcode-read-image-{}.png",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system time")
                .as_nanos()
        ));
        let png = base64::engine::general_purpose::STANDARD
            .decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl2n3sAAAAASUVORK5CYII=")
            .expect("decode png fixture");
        std::fs::write(&path, &png).expect("write png fixture");

        let output = ToolRegistry::default_tools()
            .call_with_context(
                "fs__read",
                json!({
                    "path": path.to_string_lossy(),
                    "offset": 1,
                    "limit": 10,
                }),
                ToolExecutionContext::outside_workspace_granted(),
            )
            .await;

        assert!(output.ok, "{:?}", output.error);
        assert_eq!(output.images.len(), 1);
        assert_eq!(output.images[0].mime, "image/png");
        assert!(
            output.images[0]
                .data_url
                .starts_with("data:image/png;base64,")
        );
        assert_eq!(
            output
                .data
                .as_ref()
                .and_then(|data| data.get("kind"))
                .and_then(Value::as_str),
            Some("image")
        );
        assert!(
            output
                .data
                .as_ref()
                .is_some_and(|data| !data.to_string().contains("iVBOR"))
        );

        let _ = std::fs::remove_file(path);
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

    #[cfg(windows)]
    #[tokio::test]
    async fn windows_write_and_append_tools_modify_regular_files() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time")
            .as_nanos();
        let path = PathBuf::from("target").join(format!("letcode-windows-write-{unique}.txt"));
        std::fs::create_dir_all("target").expect("create target directory");
        let registry = ToolRegistry::default_tools();

        let written = registry
            .call(
                "fs__write",
                json!({"path": path.to_string_lossy(), "content": "first"}),
            )
            .await;
        assert!(written.ok, "{:?}", written.error);
        let appended = registry
            .call(
                "fs__append",
                json!({"path": path.to_string_lossy(), "content": " second"}),
            )
            .await;
        assert!(appended.ok, "{:?}", appended.error);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "first second");
        let _ = std::fs::remove_file(path);
    }

    #[cfg(windows)]
    #[tokio::test]
    async fn windows_writable_leaf_rejects_symlink_replacement() {
        use std::os::windows::fs::symlink_file;

        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time")
            .as_nanos();
        let parent = PathBuf::from("target").join(format!("letcode-windows-link-{unique}"));
        let leaf = parent.join("leaf.txt");
        let target = parent.join("target.txt");
        std::fs::create_dir_all(&parent).unwrap();
        std::fs::write(&leaf, "original").unwrap();
        std::fs::write(&target, "unchanged").unwrap();
        let prepared = prepare_writable_leaf(leaf.to_str().unwrap()).unwrap();
        std::fs::remove_file(&leaf).unwrap();
        symlink_file(&target, &leaf).expect("create replacement symlink");

        let error = secure_write_writable_leaf(&prepared, b"must not write", false)
            .await
            .expect_err("replacement link must be rejected");
        assert_eq!(
            error.to_string(),
            "writable destination changed after authorization"
        );
        assert_eq!(std::fs::read_to_string(&target).unwrap(), "unchanged");
        let _ = std::fs::remove_dir_all(parent);
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
    fn normalize_subagent_input_accepts_model_and_rejects_takeover_conflict() {
        let input = normalize_subagent_input(
            "agent__explore",
            &json!({"objective": "inspect", "model": "expert/special"}),
        )
        .expect("model override normalizes");
        assert_eq!(input.model.as_deref(), Some("expert/special"));

        let error = normalize_subagent_input(
            "agent__explore",
            &json!({
                "objective": "inspect",
                "model": "expert/special",
                "target_child_session_id": "child-1"
            }),
        )
        .expect_err("override and takeover conflict");
        assert!(
            error
                .to_string()
                .contains("field 'model' cannot be combined with 'target_child_session_id'")
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
    async fn workflow_todos_allows_multiple_in_progress_items() {
        let output = call_workflow_todos(json!([
            {"id": "todo-1", "content": "first", "status": "in_progress"},
            {"id": "todo-2", "content": "second", "status": "in_progress"}
        ]))
        .await;

        assert!(output.ok);
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

    #[cfg(windows)]
    #[tokio::test]
    async fn windows_apply_patch_updates_existing_file() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time")
            .as_nanos();
        let path = PathBuf::from("target").join(format!("letcode-windows-patch-{unique}.txt"));
        std::fs::create_dir_all("target").unwrap();
        std::fs::write(&path, "before\n").unwrap();

        let output = ToolRegistry::default_tools()
            .call(
                "edit__apply_patch",
                json!({"edits": [{
                    "path": path.to_string_lossy(),
                    "find": "before",
                    "replace": "after",
                    "replace_all": false
                }]}),
            )
            .await;

        assert!(output.ok, "{:?}", output.error);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "after\n");
        let _ = std::fs::remove_file(path);
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
