//! File-system tool handlers (list/read/write/append/mkdir).

use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use async_trait::async_trait;
use serde_json::{Value, json};
use tokio::fs;
use tokio::io::{AsyncBufReadExt, BufReader};

use super::args::{optional_usize, required_string};
use super::paths::{display_workspace_relative, existing_workspace_path, new_workspace_path};
use super::{
    DEFAULT_READ_LINE_LIMIT, MAX_READ_BYTES, MAX_READ_IMAGE_BYTES, MAX_READ_LINE_LIMIT,
    ToolExecutionContext, ToolHandler, ToolParallelism, ToolRegistry, ToolResult,
    secure_write_writable_leaf, writable_leaf_for_execution,
};

struct ListDirTool;

#[async_trait]
impl ToolHandler for ListDirTool {
    fn name(&self) -> &'static str {
        "fs__list"
    }

    fn description(&self) -> &'static str {
        "List files and directories. Use workspace-relative paths by default; external paths require explicit authorization."
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

    fn parallelism(&self) -> ToolParallelism {
        ToolParallelism::Parallel
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
        "Read a UTF-8 text file or supported image. Text reads use 1-based line offset and line limits; image reads return the image as multimodal content. Use workspace-relative paths by default; external paths require explicit authorization."
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

    fn parallelism(&self) -> ToolParallelism {
        ToolParallelism::Parallel
    }

    async fn execute(&self, args: Value) -> Result<Value> {
        read_file(args, ToolExecutionContext::default())
            .await
            .and_then(|output| {
                output
                    .data
                    .ok_or_else(|| anyhow!("fs__read returned no data"))
            })
    }

    async fn execute_tool_result(
        &self,
        args: Value,
        context: ToolExecutionContext,
    ) -> Result<ToolResult> {
        read_file(args, context).await
    }

    async fn execute_with_context(
        &self,
        args: Value,
        context: ToolExecutionContext,
    ) -> Result<Value> {
        self.execute_tool_result(args, context)
            .await
            .and_then(|output| {
                output
                    .data
                    .ok_or_else(|| anyhow!("fs__read returned no data"))
            })
    }
}

struct WriteFileTool;

#[async_trait]
impl ToolHandler for WriteFileTool {
    fn name(&self) -> &'static str {
        "fs__write"
    }

    fn description(&self) -> &'static str {
        "Create or overwrite a UTF-8 text file. Use workspace-relative paths by default; external paths require explicit authorization."
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
        "Append UTF-8 text to a file. Use workspace-relative paths by default; external paths require explicit authorization."
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
        "Create a directory, including missing parent directories. Use workspace-relative paths by default; external paths require explicit authorization."
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

async fn read_file(args: Value, context: ToolExecutionContext) -> Result<ToolResult> {
    let path = existing_workspace_path(required_string(&args, "path")?, &context)?;
    let metadata = fs::metadata(&path).await?;
    if !metadata.is_file() {
        bail!("path is not a file: {}", path.display());
    }

    if let Some(mime) = supported_image_mime(&path) {
        return read_image_file(&path, &metadata, mime).await;
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

    Ok(ToolResult::ok(
        "fs__read",
        json!({
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
        }),
    ))
}

fn supported_image_mime(path: &Path) -> Option<&'static str> {
    match path
        .extension()
        .and_then(|extension| extension.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("png") => Some("image/png"),
        Some("jpg" | "jpeg") => Some("image/jpeg"),
        Some("webp") => Some("image/webp"),
        Some("gif") => Some("image/gif"),
        _ => None,
    }
}

async fn read_image_file(
    path: &Path,
    metadata: &std::fs::Metadata,
    mime: &str,
) -> Result<ToolResult> {
    if metadata.len() > MAX_READ_IMAGE_BYTES {
        bail!(
            "image exceeds max read bytes ({MAX_READ_IMAGE_BYTES}) in {}",
            path.display()
        );
    }
    let bytes = fs::read(path)
        .await
        .with_context(|| format!("failed to read image {}", path.display()))?;
    let display_path = display_workspace_relative(path)?;
    let label = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("image")
        .to_string();
    let image = crate::user_content::UserImageAttachment::from_bytes(label, mime, &bytes);
    Ok(ToolResult::ok(
        "fs__read",
        json!({
            "path": display_path,
            "kind": "image",
            "mime": mime,
            "bytes": bytes.len(),
        }),
    )
    .with_images(vec![image]))
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

pub(super) fn register(registry: &mut ToolRegistry) {
    registry.register(ListDirTool);
    registry.register(ReadFileTool);
    registry.register(WriteFileTool);
    registry.register(AppendFileTool);
    registry.register(MkdirTool);
}
