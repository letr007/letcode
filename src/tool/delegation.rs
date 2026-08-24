//! Subagent (delegated child) input normalization and path-scope checks.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::permission::path_preview;
use crate::tool_names;

use super::paths::{canonical_existing_path, join_workspace_path, workspace_root};
use super::{
    MAX_SUBAGENT_LIST_ITEMS, MAX_SUBAGENT_TEXT_FIELD_CHARS, PreparedApplyPatch,
    PreparedWritableLeaf,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NormalizedSubagentInput {
    pub objective: String,
    pub success_criteria: Vec<String>,
    pub allowed_paths: Vec<String>,
    pub forbidden_paths: Vec<String>,
    pub owned_paths: Vec<String>,
    pub timeout_secs: Option<u64>,
    pub max_tool_calls: Option<usize>,
    pub model: Option<String>,
    pub target_child_session_id: Option<String>,
    #[serde(default)]
    pub background: bool,
}

impl NormalizedSubagentInput {
    pub fn render_for_delegate(&self, tool_name: &str) -> String {
        let mut lines = vec![format!("目标：{}", self.objective)];

        if !self.success_criteria.is_empty() {
            lines.push("成功标准：".into());
            lines.extend(self.success_criteria.iter().map(|item| format!("- {item}")));
        }

        if !self.allowed_paths.is_empty() {
            lines.push(format!("允许路径：{}", self.allowed_paths.join(", ")));
        }
        if !self.forbidden_paths.is_empty() {
            lines.push(format!("禁止路径：{}", self.forbidden_paths.join(", ")));
        }
        if !self.owned_paths.is_empty() {
            lines.push(format!("负责路径：{}", self.owned_paths.join(", ")));
        }
        if let Some(model) = &self.model {
            lines.push(format!("单次模型路由：{model}"));
        }
        if let Some(target) = &self.target_child_session_id {
            lines.push(format!("接管子会话：{target}"));
        }

        if self.timeout_secs.is_some() || self.max_tool_calls.is_some() {
            lines.push(format!(
                "执行边界：timeout_secs={}，max_tool_calls={}",
                self.timeout_secs
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "继承".into()),
                self.max_tool_calls
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "继承".into())
            ));
        }

        lines.push("委派约定：不要递归委派；保持在给定范围内，并简洁报告发现或实现结果。".into());

        if tool_name == tool_names::TOOL_AGENT_EXPLORE {
            lines.push("模式：仅只读探索。".into());
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

/// Run-local directory authorization for a delegated child agent.
/// Roots are canonicalized at install time; comparisons use canonical targets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SubagentPathScope {
    allowed_roots: Vec<PathBuf>,
    owned_roots: Vec<PathBuf>,
    forbidden_roots: Vec<PathBuf>,
}

impl SubagentPathScope {
    pub(crate) fn from_input(input: &NormalizedSubagentInput) -> Result<Option<Self>> {
        if input.allowed_paths.is_empty()
            && input.owned_paths.is_empty()
            && input.forbidden_paths.is_empty()
        {
            return Ok(None);
        }
        Ok(Some(Self {
            allowed_roots: canonicalize_scope_roots(&input.allowed_paths)?,
            owned_roots: canonicalize_scope_roots(&input.owned_paths)?,
            forbidden_roots: canonicalize_scope_roots(&input.forbidden_paths)?,
        }))
    }

    pub(crate) fn permits_read<'a>(&self, paths: impl IntoIterator<Item = &'a Path>) -> bool {
        for path in paths {
            if self.is_forbidden(path) {
                return false;
            }
            if self.allowed_roots.is_empty() && self.owned_roots.is_empty() {
                continue;
            }
            if !self.in_allowed_or_owned(path) {
                return false;
            }
        }
        true
    }

    pub(crate) fn owned_roots(&self) -> &[PathBuf] {
        &self.owned_roots
    }

    pub(crate) fn permits_write<'a>(&self, paths: impl IntoIterator<Item = &'a Path>) -> bool {
        for path in paths {
            if self.is_forbidden(path) {
                return false;
            }
            if !self.in_owned(path) {
                return false;
            }
        }
        true
    }

    fn is_forbidden(&self, path: &Path) -> bool {
        self.forbidden_roots
            .iter()
            .any(|root| path_under_root(path, root))
    }

    fn in_owned(&self, path: &Path) -> bool {
        self.owned_roots
            .iter()
            .any(|root| path_under_root(path, root))
    }

    fn in_allowed_or_owned(&self, path: &Path) -> bool {
        self.allowed_roots
            .iter()
            .any(|root| path_under_root(path, root))
            || self.in_owned(path)
    }
}

fn canonicalize_scope_roots(paths: &[String]) -> Result<Vec<PathBuf>> {
    let mut roots = Vec::with_capacity(paths.len());
    for path in paths {
        let canonical = canonical_existing_path(path)
            .ok_or_else(|| anyhow!("delegated path scope root cannot be resolved: {path}"))?;
        roots.push(canonical);
    }
    Ok(roots)
}

fn path_under_root(path: &Path, root: &Path) -> bool {
    path == root || path.starts_with(root)
}

pub(crate) fn is_delegation_path_scoped_tool(name: &str) -> bool {
    matches!(
        name,
        tool_names::TOOL_FS_READ
            | tool_names::TOOL_FS_LIST
            | tool_names::TOOL_SEARCH_RG
            | tool_names::TOOL_CODE_AST_SEARCH
            | tool_names::TOOL_CODE_AST_REPLACE_PREVIEW
            | tool_names::TOOL_FS_WRITE
            | tool_names::TOOL_FS_APPEND
            | tool_names::TOOL_FS_MKDIR
            | tool_names::TOOL_EDIT_APPLY_PATCH
    )
}

/// Returns an error message when a structured tool target is outside the
/// delegated path scope. `None` means either no check applied or permitted.
pub(crate) fn delegation_scope_denial(
    scope: &SubagentPathScope,
    tool: &str,
    args: &Value,
    prepared_writable: Option<&PreparedWritableLeaf>,
    prepared_patch: Option<&PreparedApplyPatch>,
) -> Option<String> {
    if !is_delegation_path_scoped_tool(tool) {
        return None;
    }

    if tool == tool_names::TOOL_EDIT_APPLY_PATCH {
        let Some(prepared) = prepared_patch else {
            return Some(
                "delegated path scope denied: edit__apply_patch missing prepared targets".into(),
            );
        };
        let targets: Vec<&Path> = prepared.target_paths().collect();
        if targets.is_empty() {
            return Some(
                "delegated path scope denied: edit__apply_patch has no bound targets".into(),
            );
        }
        for target in &targets {
            if scope.is_forbidden(target) {
                return Some(format!(
                    "delegated path scope denied: path {} is forbidden",
                    path_preview(target)
                ));
            }
        }
        if !scope.permits_write(targets.iter().copied()) {
            return Some(format!(
                "delegated path scope denied: edit__apply_patch targets {}, which is outside owned_paths",
                targets
                    .iter()
                    .map(|path| path_preview(path))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        return None;
    }

    if matches!(tool, tool_names::TOOL_FS_WRITE | tool_names::TOOL_FS_APPEND) {
        let Some(prepared) = prepared_writable else {
            return Some(format!(
                "delegated path scope denied: {tool} missing prepared writable leaf"
            ));
        };
        let destination = prepared.destination();
        if scope.is_forbidden(destination) {
            return Some(format!(
                "delegated path scope denied: path {} is forbidden",
                path_preview(destination)
            ));
        }
        if !scope.permits_write(std::iter::once(destination)) {
            return Some(format!(
                "delegated path scope denied: {tool} target {}, which is outside owned_paths",
                path_preview(destination)
            ));
        }
        return None;
    }

    if tool == tool_names::TOOL_FS_MKDIR {
        let Some(path) = args.get("path").and_then(Value::as_str) else {
            return Some("delegated path scope denied: fs__mkdir missing path".into());
        };
        let destination = match resolve_scope_destination(path) {
            Ok(path) => path,
            Err(error) => {
                return Some(format!(
                    "delegated path scope denied: cannot resolve mkdir path: {error}"
                ));
            }
        };
        if scope.is_forbidden(&destination) {
            return Some(format!(
                "delegated path scope denied: path {} is forbidden",
                path_preview(&destination)
            ));
        }
        if !mkdir_creates_only_within_owned(&destination, scope) {
            return Some(format!(
                "delegated path scope denied: fs__mkdir target {}, which is outside owned_paths or would create parents outside owned_paths",
                path_preview(&destination)
            ));
        }
        return None;
    }

    // Read / preview tools.
    let path = args.get("path").and_then(Value::as_str).unwrap_or(".");
    let canonical = match canonical_existing_path(path) {
        Some(path) => path,
        None => {
            return Some(format!(
                "delegated path scope denied: cannot resolve path `{path}` for scope check"
            ));
        }
    };
    if scope.is_forbidden(&canonical) {
        return Some(format!(
            "delegated path scope denied: path {} is forbidden",
            path_preview(&canonical)
        ));
    }
    if !scope.permits_read(std::iter::once(canonical.as_path())) {
        return Some(format!(
            "delegated path scope denied: {tool} path {}, which is outside allowed_paths/owned_paths",
            path_preview(&canonical)
        ));
    }
    None
}

fn resolve_scope_destination(path: &str) -> Result<PathBuf> {
    let root = workspace_root()?;
    let candidate = join_workspace_path(&root, path);
    if candidate.as_os_str().is_empty() {
        bail!("path cannot be empty");
    }
    let mut existing = candidate.clone();
    let mut suffix = Vec::new();
    loop {
        if existing.exists() {
            let mut resolved = existing
                .canonicalize()
                .with_context(|| format!("failed to canonicalize {}", existing.display()))?;
            for part in suffix.iter().rev() {
                resolved.push(part);
            }
            return Ok(resolved);
        }
        let file_name = existing
            .file_name()
            .ok_or_else(|| anyhow!("cannot resolve path: {}", candidate.display()))?
            .to_os_string();
        suffix.push(file_name);
        let Some(parent) = existing.parent() else {
            bail!("cannot resolve path: {}", candidate.display());
        };
        if parent.as_os_str().is_empty() || parent == existing.as_path() {
            bail!("cannot resolve path: {}", candidate.display());
        }
        existing = parent.to_path_buf();
    }
}

fn mkdir_creates_only_within_owned(destination: &Path, scope: &SubagentPathScope) -> bool {
    if !scope.permits_write(std::iter::once(destination)) {
        return false;
    }
    let mut current = destination.to_path_buf();
    loop {
        if current.exists() {
            return true;
        }
        if !scope
            .owned_roots
            .iter()
            .any(|root| path_under_root(&current, root))
        {
            return false;
        }
        match current.parent() {
            Some(parent) if parent != current.as_path() => current = parent.to_path_buf(),
            _ => return false,
        }
    }
}

pub fn normalize_subagent_input(tool_name: &str, args: &Value) -> Result<NormalizedSubagentInput> {
    let task = optional_trimmed_string(args, "task")?;
    let objective = optional_trimmed_string(args, "objective")?;
    let objective = objective.or(task).ok_or_else(|| {
        anyhow!(
            "{tool_name} requires a non-empty 'task' or 'objective' field to describe the delegated work"
        )
    })?;

    let model = optional_trimmed_string(args, "model")?;
    let target_child_session_id = optional_trimmed_string(args, "target_child_session_id")?;
    if model.is_some() && target_child_session_id.is_some() {
        bail!("field 'model' cannot be combined with 'target_child_session_id'");
    }

    let owned_paths = optional_trimmed_string_list(args, "owned_paths")?;
    if tool_name == tool_names::TOOL_AGENT_FIXER && owned_paths.is_empty() {
        bail!("agent__fixer requires non-empty owned_paths for file-level write locking");
    }

    Ok(NormalizedSubagentInput {
        objective,
        success_criteria: optional_trimmed_string_list(args, "success_criteria")?,
        allowed_paths: optional_trimmed_string_list(args, "allowed_paths")?,
        forbidden_paths: optional_trimmed_string_list(args, "forbidden_paths")?,
        owned_paths,
        timeout_secs: optional_u64(args, "timeout_secs")?,
        max_tool_calls: optional_u64(args, "max_tool_calls")?.map(|value| value as usize),
        model,
        target_child_session_id,
        background: optional_bool(args, "background")?.unwrap_or(false),
    })
}

fn optional_bool(args: &Value, field: &str) -> Result<Option<bool>> {
    let Some(value) = args.get(field) else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    value
        .as_bool()
        .map(Some)
        .ok_or_else(|| anyhow!("field '{field}' must be a boolean or null"))
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

pub(crate) fn optional_u64(args: &Value, field: &str) -> Result<Option<u64>> {
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
                "description": "当前委派拥有编辑权的文件或目录子树；fixer 必填，路径重叠的读写/写写任务会被拒绝"
            },
            "model": {
                "type": ["string", "null"],
                "description": "仅用于新建子代理的单次 provider/model 路由覆盖"
            },
            "target_child_session_id": {
                "type": ["string", "null"],
                "description": "接替已有终态子会话并复用其上下文；省略则新建子代理会话"
            },
            "background": {
                "type": ["boolean", "null"],
                "description": "true=在后台运行并立即返回；完成后自动通知父会话。默认 false，等待结果后再继续"
            }
        },
        // OpenAI strict tool schemas require every properties key to appear in
        // required (optional fields use type unions that include null).
        "required": [
            "task",
            "objective",
            "success_criteria",
            "allowed_paths",
            "forbidden_paths",
            "owned_paths",
            "model",
            "target_child_session_id",
            "background"
        ],
        "additionalProperties": false
    })
}
