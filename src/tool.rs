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

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
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
mod context_history;
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

pub(crate) fn workspace_root_for_subagent_lock() -> Result<PathBuf> {
    workspace_root()
}

pub(crate) fn canonical_subagent_observed_path(path: &str) -> Result<PathBuf> {
    let root = workspace_root()?;
    let candidate = join_workspace_path(&root, path);
    if let Ok(canonical) = candidate.canonicalize() {
        return Ok(canonical);
    }
    let mut current = candidate.clone();
    let mut suffix = Vec::new();
    loop {
        if current.exists() {
            let mut canonical = current.canonicalize()?;
            for component in suffix.iter().rev() {
                canonical.push(component);
            }
            return Ok(canonical);
        }
        let name = current
            .file_name()
            .ok_or_else(|| anyhow!("cannot resolve subagent changed path: {path}"))?
            .to_os_string();
        suffix.push(name);
        current = current
            .parent()
            .ok_or_else(|| anyhow!("cannot resolve subagent changed path: {path}"))?
            .to_path_buf();
    }
}
pub(crate) use fs::supported_image_mime;
pub use registry::ToolRegistry;

const DEFAULT_READ_LINE_LIMIT: usize = 200;
const MAX_READ_LINE_LIMIT: usize = 5_000;
const MAX_READ_BYTES: usize = 4 * 1024 * 1024;
pub(crate) const MAX_READ_IMAGE_BYTES: u64 = 20 * 1024 * 1024;
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
    pub(crate) history_cursor: Option<crate::context_history::HistoryCursor>,
    pub allow_outside_workspace: bool,
    pub(crate) allow_tool_result_images: bool,
    pub question_handler: Option<QuestionCallback>,
    prepared_writable_leaf: Option<PreparedWritableLeaf>,
    prepared_apply_patch: Option<PreparedApplyPatch>,
}

impl std::fmt::Debug for ToolExecutionContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolExecutionContext")
            .field("allow_outside_workspace", &self.allow_outside_workspace)
            .field("allow_tool_result_images", &self.allow_tool_result_images)
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
            allow_tool_result_images: false,
            history_cursor: None,
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
        context_history::register(&mut registry);
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
    use cap_fs_ext::{FollowSymlinks, OpenOptionsFollowExt};
    use cap_std::fs::OpenOptionsExt;
    use windows_sys::Win32::Storage::FileSystem::{FILE_SHARE_READ, FILE_SHARE_WRITE};

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
    let existing_file = match std::fs::symlink_metadata(&destination) {
        Ok(metadata) => {
            if metadata.file_type().is_symlink() || windows_metadata_is_reparse_point(&metadata) {
                bail!(
                    "writable destination cannot be a link or reparse point: {}",
                    destination.display()
                );
            }
            let mut options = cap_std::fs::OpenOptions::new();
            options
                .read(true)
                .write(true)
                .follow(FollowSymlinks::No)
                .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE);
            let file = parent_dir
                .open_with(
                    destination.file_name().expect("validated file name"),
                    &options,
                )
                .with_context(|| {
                    format!(
                        "failed to bind writable destination {}",
                        destination.display()
                    )
                })?;
            if windows_file_is_reparse_point(
                &file
                    .try_clone()
                    .context("failed to clone writable destination anchor")?
                    .into_std(),
            )? {
                bail!(
                    "writable destination cannot be a link or reparse point: {}",
                    destination.display()
                );
            }
            Some(Arc::new(file))
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => {
            return Err(error).with_context(|| {
                format!(
                    "failed to inspect writable destination: {}",
                    destination.display()
                )
            });
        }
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
    use cap_fs_ext::{FollowSymlinks, OpenOptionsFollowExt};
    use std::io::{Seek, SeekFrom, Write};

    if !prepared.parent_instance_is_current() {
        bail!(WRITABLE_DESTINATION_CHANGED);
    }

    let mut file = if let Some(existing_file) = &prepared.existing_file {
        existing_file
            .try_clone()
            .map(cap_std::fs::File::into_std)
            .map_err(|_| anyhow!(WRITABLE_DESTINATION_CHANGED))?
    } else {
        let mut options = cap_std::fs::OpenOptions::new();
        options
            .write(true)
            .create_new(true)
            .follow(FollowSymlinks::No);
        prepared
            .parent_dir
            .open_with(prepared.leaf_name(), &options)
            .map(cap_std::fs::File::into_std)
            .map_err(|_| anyhow!(WRITABLE_DESTINATION_CHANGED))?
    };
    if append {
        file.seek(SeekFrom::End(0))?;
    } else {
        file.set_len(0)?;
        file.seek(SeekFrom::Start(0))?;
    }
    file.write_all(content)?;
    file.flush()?;
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

#[cfg(windows)]
fn windows_file_is_reparse_point(file: &std::fs::File) -> Result<bool> {
    use std::mem::size_of;
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Storage::FileSystem::{
        FILE_ATTRIBUTE_REPARSE_POINT, FILE_ATTRIBUTE_TAG_INFO, FileAttributeTagInfo,
        GetFileInformationByHandleEx,
    };
    let mut information = FILE_ATTRIBUTE_TAG_INFO::default();
    let ok = unsafe {
        GetFileInformationByHandleEx(
            file.as_raw_handle().cast(),
            FileAttributeTagInfo,
            (&mut information as *mut FILE_ATTRIBUTE_TAG_INFO).cast(),
            size_of::<FILE_ATTRIBUTE_TAG_INFO>() as u32,
        )
    };
    if ok == 0 {
        return Err(std::io::Error::last_os_error())
            .context("failed to inspect Windows file attributes");
    }
    Ok(information.FileAttributes & FILE_ATTRIBUTE_REPARSE_POINT != 0)
}

#[cfg(test)]
#[path = "../tests/internal/tool/io_tests.rs"]
mod io_tests;

#[cfg(test)]
#[path = "../tests/internal/tool/tests.rs"]
mod tests;
