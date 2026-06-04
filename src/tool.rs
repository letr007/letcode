use anyhow::{Context, Result, anyhow, bail};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::path::Component;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use tokio::fs;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::Command;
use tokio::time::{Duration, timeout};
use tracing::{debug, warn};

use crate::code_analysis::{AstReplacePreviewRequest, AstSearchRequest, CodeAnalysisRegistry};
use crate::request_builder::ToolSpec;

const DEFAULT_READ_LINE_LIMIT: usize = 200;
const MAX_READ_LINE_LIMIT: usize = 1000;
const MAX_READ_BYTES: usize = 1024 * 1024;
const MAX_COMMAND_OUTPUT_BYTES: usize = 32 * 1024;
const COMMAND_TIMEOUT_SECS: u64 = 30;

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
}

#[async_trait]
pub trait ToolHandler: Send + Sync {
    fn name(&self) -> &str;

    fn description(&self) -> &str;

    fn parameters(&self) -> Value;

    fn strict(&self) -> bool {
        true
    }

    async fn execute(&self, args: Value) -> Result<Value>;

    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: self.name().to_string(),
            description: self.description().to_string(),
            parameters: self.parameters(),
            strict: self.strict(),
        }
    }
}

#[derive(Clone, Default)]
pub struct ToolRegistry {
    tools: BTreeMap<String, Arc<dyn ToolHandler>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn default_tools() -> Self {
        let mut registry = Self::new();
        registry.register(EchoTool);
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
        registry
    }

    pub fn register<T>(&mut self, tool: T)
    where
        T: ToolHandler + 'static,
    {
        let name = tool.name().to_string();
        self.tools.insert(name, Arc::new(tool));
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
        self.tools.values().map(|tool| tool.spec()).collect()
    }

    pub async fn call(&self, name: &str, args: Value) -> ToolResult {
        debug!(tool_name = %name, args = %args, "calling tool");

        let Some(tool) = self.tools.get(name) else {
            warn!(tool_name = %name, "unknown tool requested");
            return ToolResult::err(name, format!("unknown tool: {name}"));
        };

        match tool.execute(args).await {
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

struct EchoTool;

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
        list_dir(args).await
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
        read_file(args).await
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
        write_file(args).await
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
        append_file(args).await
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
        mkdir(args).await
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
        rg(args).await
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
        apply_patch(args).await
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
        CodeAnalysisRegistry::default_backends()
            .ast_search(AstSearchRequest {
                path: required_string(&args, "path")?.to_string(),
                language: Some(required_string(&args, "language")?.to_string()),
                pattern: required_string(&args, "pattern")?.to_string(),
                max_results: optional_usize(&args, "max_results").unwrap_or(100),
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
        CodeAnalysisRegistry::default_backends()
            .ast_replace_preview(AstReplacePreviewRequest {
                path: required_string(&args, "path")?.to_string(),
                language: Some(required_string(&args, "language")?.to_string()),
                pattern: required_string(&args, "pattern")?.to_string(),
                rewrite: required_string(&args, "rewrite")?.to_string(),
            })
            .await
    }
}

async fn list_dir(args: Value) -> Result<Value> {
    let path = existing_workspace_path(required_string(&args, "path")?)?;
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

async fn read_file(args: Value) -> Result<Value> {
    let path = existing_workspace_path(required_string(&args, "path")?)?;
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

async fn write_file(args: Value) -> Result<Value> {
    let raw_path = required_string(&args, "path")?;
    let content = required_string(&args, "content")?;
    let path = writable_workspace_path(raw_path)?;

    fs::write(&path, content)
        .await
        .with_context(|| format!("failed to write file {}", path.display()))?;

    Ok(json!({
        "path": display_workspace_relative(&path)?,
        "bytes_written": content.len(),
    }))
}

async fn append_file(args: Value) -> Result<Value> {
    let raw_path = required_string(&args, "path")?;
    let content = required_string(&args, "content")?;
    let path = writable_workspace_path(raw_path)?;

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

async fn mkdir(args: Value) -> Result<Value> {
    let raw_path = required_string(&args, "path")?;
    let path = new_workspace_path(raw_path)?;

    fs::create_dir_all(&path)
        .await
        .with_context(|| format!("failed to create directory {}", path.display()))?;

    Ok(json!({
        "path": display_workspace_relative(&path)?,
        "created": true,
    }))
}

async fn rg(args: Value) -> Result<Value> {
    let pattern = required_string(&args, "pattern")?;
    let raw_path = optional_string(&args, "path").unwrap_or(".");
    let path = existing_workspace_path(raw_path)?;
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

async fn apply_patch(args: Value) -> Result<Value> {
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
        let path = existing_workspace_path(required_string(edit, "path")?)?;
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

fn existing_workspace_path(path: &str) -> Result<PathBuf> {
    let root = workspace_root()?;
    let candidate = join_workspace_path(&root, path);
    let canonical = candidate
        .canonicalize()
        .with_context(|| format!("path does not exist: {}", candidate.display()))?;
    ensure_inside_workspace(&root, &canonical)?;
    Ok(canonical)
}

fn writable_workspace_path(path: &str) -> Result<PathBuf> {
    let root = workspace_root()?;
    let candidate = join_workspace_path(&root, path);
    let parent = candidate
        .parent()
        .ok_or_else(|| anyhow!("path has no parent: {}", candidate.display()))?
        .canonicalize()
        .with_context(|| format!("parent directory does not exist: {}", candidate.display()))?;

    ensure_inside_workspace(&root, &parent)?;
    Ok(parent.join(
        candidate
            .file_name()
            .ok_or_else(|| anyhow!("path has no file name: {}", candidate.display()))?,
    ))
}

fn new_workspace_path(path: &str) -> Result<PathBuf> {
    let root = workspace_root()?;
    let relative = safe_relative_path_arg(path)?;
    Ok(root.join(relative))
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
    use super::ToolRegistry;

    #[test]
    fn default_tools_use_namespaced_names_without_legacy_aliases() {
        let specs = ToolRegistry::default_tools().specs();
        let names = specs
            .iter()
            .map(|spec| spec.name.as_str())
            .collect::<Vec<_>>();

        for expected in [
            "util__echo",
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
}
