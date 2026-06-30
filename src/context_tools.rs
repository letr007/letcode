use anyhow::{Result, anyhow, bail, ensure};
use async_trait::async_trait;
use serde_json::{Value, json};
use std::convert::TryFrom;
use std::sync::Arc;

use crate::context_tree::{ContextNodeId, ContextNodeRecord, ContextNodeStatus, ContextTreeState};
use crate::context_view::{
    ContextBlock, ContextBlockKind, ContextBlockSource, ContextViewProjection, ContextViewStatus,
    FoldedOutputMetadata, project_context_view,
};
use crate::permission::ToolPermissionClass;
use crate::tool::{ToolExecutionContext, ToolHandler, ToolRegistry};
use crate::tool_names;
use crate::transcript::transcript_projection::project_context_tree;
use crate::transcript::{TranscriptEvent, TranscriptRecord};
use crate::user_content::UserMessageContent;

const DEFAULT_LIST_LIMIT: usize = 20;
const MAX_LIST_LIMIT: usize = 100;
const DEFAULT_OPEN_MAX_BYTES: usize = 2048;
const MAX_OPEN_MAX_BYTES: usize = 16 * 1024;
const MAX_QUERY_CHARS: usize = 256;
const MAX_ID_CHARS: usize = 256;
const MAX_SUMMARY_CHARS: usize = 4000;
const MAX_ARTIFACT_KIND_CHARS: usize = 64;

pub(crate) fn register_context_tools(registry: &mut ToolRegistry) {
    registry.register(ContextListTool);
    registry.register(ContextSearchTool);
    registry.register(ContextOpenTool);
    registry.register(ContextSummarizeTool);
    registry.register(ContextPinTool);
    registry.register(ContextArchiveTool);
    registry.register(ContextRemoveTool);
}

struct ContextListTool;
struct ContextSearchTool;
struct ContextOpenTool;
struct ContextSummarizeTool;
struct ContextPinTool;
struct ContextArchiveTool;
struct ContextRemoveTool;

#[async_trait]
impl ToolHandler for ContextListTool {
    fn name(&self) -> &str {
        tool_names::TOOL_CONTEXT_LIST
    }
    fn description(&self) -> &str {
        "List available context nodes, blocks, summaries, and folded outputs."
    }
    fn permission_class(&self) -> ToolPermissionClass {
        ToolPermissionClass::Read
    }
    fn parameters(&self) -> Value {
        json!({
            "type":"object",
            "properties":{
                "include_archived":{"type":"boolean"},
                "include_removed":{"type":"boolean"},
                "limit":{"type":["integer","null"],"minimum":1,"maximum":MAX_LIST_LIMIT}
            },
            "required":["include_archived","include_removed","limit"],
            "additionalProperties":false
        })
    }
    async fn execute(&self, _args: Value) -> Result<Value> {
        bail!("context view projection unavailable")
    }
    async fn execute_with_context(
        &self,
        args: Value,
        context: ToolExecutionContext,
    ) -> Result<Value> {
        let projection = require_projection(&context)?;
        let tree = require_context_tree(&context)?;
        let include_archived = args
            .get("include_archived")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let include_removed = args
            .get("include_removed")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let limit = parse_limit(&args, DEFAULT_LIST_LIMIT)?;

        let nodes = tree
            .nodes()
            .take(limit)
            .map(|node| node_ref_json(&tree, node))
            .collect::<Vec<_>>();

        let blocks = sorted_blocks(&projection)
            .into_iter()
            .filter(|(id, block)| {
                block_visible_for_listing(
                    &projection,
                    id.as_str(),
                    block,
                    include_archived,
                    include_removed,
                )
            })
            .take(limit)
            .map(|(id, block)| block_ref_json(&projection, id.as_str(), block))
            .collect::<Vec<_>>();

        let summaries = projection
            .summary_artifacts
            .iter()
            .take(limit)
            .map(|artifact| {
                json!({
                    "ref_type":"summary",
                    "ref_id":artifact.artifact_id,
                    "node_id":artifact.node_id,
                    "version":artifact.version,
                    "artifact_kind":artifact.artifact_kind,
                    "summary":truncate(&artifact.summary, 160)
                })
            })
            .collect::<Vec<_>>();

        let folded_outputs = sorted_blocks(&projection)
            .into_iter()
            .filter(|(id, block)| {
                block.folded_output_id.is_some()
                    && block_visible_for_listing(
                        &projection,
                        id.as_str(),
                        block,
                        include_archived,
                        include_removed,
                    )
            })
            .filter_map(|(_, block)| block.folded_output_id.as_deref())
            .filter_map(|output_id| projection.folded_outputs.get(output_id))
            .take(limit)
            .map(folded_output_ref_json)
            .collect::<Vec<_>>();

        Ok(
            json!({"ok":true,"nodes":nodes,"blocks":blocks,"summaries":summaries,"folded_outputs":folded_outputs}),
        )
    }
}

#[async_trait]
impl ToolHandler for ContextSearchTool {
    fn name(&self) -> &str {
        tool_names::TOOL_CONTEXT_SEARCH
    }
    fn description(&self) -> &str {
        "Search context nodes, blocks, summaries, and folded outputs."
    }
    fn permission_class(&self) -> ToolPermissionClass {
        ToolPermissionClass::Read
    }
    fn parameters(&self) -> Value {
        json!({
            "type":"object",
            "properties":{
                "query":{"type":"string","maxLength":MAX_QUERY_CHARS},
                "include_archived":{"type":"boolean"},
                "include_removed":{"type":"boolean"},
                "limit":{"type":["integer","null"],"minimum":1,"maximum":MAX_LIST_LIMIT}
            },
            "required":["query","include_archived","include_removed","limit"],
            "additionalProperties":false
        })
    }
    async fn execute(&self, _args: Value) -> Result<Value> {
        bail!("context view projection unavailable")
    }
    async fn execute_with_context(
        &self,
        args: Value,
        context: ToolExecutionContext,
    ) -> Result<Value> {
        let projection = require_projection(&context)?;
        let tree = require_context_tree(&context)?;
        let query = required_trimmed_string(&args, "query", MAX_QUERY_CHARS)?;
        let include_archived = args
            .get("include_archived")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let include_removed = args
            .get("include_removed")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let limit = parse_limit(&args, DEFAULT_LIST_LIMIT)?;
        let query_lower = query.to_ascii_lowercase();

        let mut matches = Vec::new();
        for node in tree.nodes() {
            let haystack = format!(
                "{} {} {} {} {} {}",
                node.node_id.as_str(),
                node.label.as_deref().unwrap_or(""),
                node.purpose.as_deref().unwrap_or(""),
                node_status_label(&node.status),
                node.block_ref
                    .as_ref()
                    .map(|block_ref| block_ref.block_id.as_str())
                    .unwrap_or(""),
                node.source_ref
                    .as_ref()
                    .map(format_source_ref)
                    .unwrap_or_default()
            );
            if haystack.to_ascii_lowercase().contains(&query_lower) {
                matches.push(node_ref_json(&tree, node));
            }
            if matches.len() >= limit {
                break;
            }
        }
        for (id, block) in sorted_blocks(&projection) {
            if !block_visible_for_listing(
                &projection,
                id.as_str(),
                block,
                include_archived,
                include_removed,
            ) {
                continue;
            }
            let haystack = format!(
                "{} {} {} {} {} {}",
                id.as_str(),
                block.title,
                block.detail,
                context_block_kind_label(block.kind),
                format_block_source(&block.source),
                block_status_string(&projection, id.as_str())
            );
            if haystack.to_ascii_lowercase().contains(&query_lower) {
                matches.push(block_ref_json(&projection, id.as_str(), block));
            }
            if matches.len() >= limit {
                break;
            }
        }
        for artifact in &projection.summary_artifacts {
            if matches.len() >= limit {
                break;
            }
            let haystack = format!(
                "{} {} {} {} {} {} {} {} {}",
                artifact.artifact_id,
                artifact.node_id,
                artifact.artifact_kind,
                artifact.version,
                artifact.summary,
                artifact.source_block_id.as_deref().unwrap_or(""),
                artifact.source_node_id.as_deref().unwrap_or(""),
                artifact
                    .source_start_sequence
                    .map(|value| value.to_string())
                    .unwrap_or_default(),
                artifact
                    .source_end_sequence
                    .map(|value| value.to_string())
                    .unwrap_or_default()
            );
            if haystack.to_ascii_lowercase().contains(&query_lower) {
                matches.push(summary_ref_json(artifact));
            }
        }
        for metadata in projection.folded_outputs.values() {
            if matches.len() >= limit {
                break;
            }
            let Some((block_id, block)) = folded_block_for_output(&projection, &metadata.output_id)
            else {
                continue;
            };
            if !block_visible_for_listing(
                &projection,
                block_id,
                block,
                include_archived,
                include_removed,
            ) {
                continue;
            }
            let haystack = format!(
                "{} {} {} {} {} {} {} {} {}",
                metadata.output_id,
                metadata.tool_name.as_deref().unwrap_or(""),
                metadata.shell_command.as_deref().unwrap_or(""),
                metadata.content,
                folded_status(metadata),
                metadata.byte_count,
                metadata.line_count,
                metadata
                    .source_start_sequence
                    .map(|value| value.to_string())
                    .unwrap_or_default(),
                metadata
                    .source_end_sequence
                    .map(|value| value.to_string())
                    .unwrap_or_default()
            );
            if haystack.to_ascii_lowercase().contains(&query_lower) {
                matches.push(folded_output_match_json(
                    &projection,
                    block_id,
                    block,
                    metadata,
                ));
            }
        }

        Ok(json!({"ok":true,"matches":matches}))
    }
}

#[async_trait]
impl ToolHandler for ContextOpenTool {
    fn name(&self) -> &str {
        tool_names::TOOL_CONTEXT_OPEN
    }
    fn description(&self) -> &str {
        "Open a context node, block, summary, or folded output by stable reference."
    }
    fn permission_class(&self) -> ToolPermissionClass {
        ToolPermissionClass::Read
    }
    fn parameters(&self) -> Value {
        json!({
            "type":"object",
            "properties":{
                "ref_type":{"type":"string","enum":["node","block","summary","folded_output"]},
                "ref_id":{"type":"string","maxLength":MAX_ID_CHARS},
                "max_bytes":{"type":["integer","null"],"minimum":1,"maximum":MAX_OPEN_MAX_BYTES}
            },
            "required":["ref_type","ref_id","max_bytes"],
            "additionalProperties":false
        })
    }
    async fn execute(&self, _args: Value) -> Result<Value> {
        bail!("context view projection unavailable")
    }
    async fn execute_with_context(
        &self,
        args: Value,
        context: ToolExecutionContext,
    ) -> Result<Value> {
        let projection = require_projection(&context)?;
        let tree = require_context_tree(&context)?;
        let ref_type = required_trimmed_string(&args, "ref_type", 32)?;
        let ref_id = required_trimmed_string(&args, "ref_id", MAX_ID_CHARS)?;
        let max_bytes = parse_max_bytes(&args)?;
        match ref_type.as_str() {
            "node" => {
                let node_id = ContextNodeId::new(ref_id.clone())?;
                let node = tree
                    .node(&node_id)
                    .ok_or_else(|| anyhow!("unknown context node '{ref_id}'"))?;
                Ok(open_node_json(&projection, &tree, node, max_bytes))
            }
            "block" => {
                let block = projection
                    .blocks
                    .iter()
                    .find(|(id, _)| id.as_str() == ref_id)
                    .map(|(_, block)| block)
                    .ok_or_else(|| anyhow!("unknown context block '{ref_id}'"))?;
                let status = ensure_block_openable(&projection, &ref_id)?;
                let detail = truncate(&block.detail, max_bytes);
                Ok(json!({
                    "ok":true,
                    "ref_type":"block",
                    "ref_id":ref_id,
                    "status":status,
                    "source":format_block_source(&block.source),
                    "detail":detail,
                    "operation_metadata":{"operation":"open_detail","block_id":ref_id},
                    "pending_recording":true
                }))
            }
            "summary" => {
                let artifact = projection
                    .open_summary_artifact(&ref_id)
                    .ok_or_else(|| anyhow!("unknown context summary '{ref_id}'"))?;
                Ok(json!({
                    "ok":true,
                    "ref_type":"summary",
                    "ref_id":ref_id,
                    "node_id":artifact.node_id,
                    "version":artifact.version,
                    "summary":truncate(&artifact.summary, max_bytes),
                    "source_block_id":artifact.source_block_id,
                    "source_node_id":artifact.source_node_id,
                    "source_start_sequence":artifact.source_start_sequence,
                    "source_end_sequence":artifact.source_end_sequence
                }))
            }
            "folded_output" => {
                let metadata = projection
                    .folded_outputs
                    .get(&ref_id)
                    .ok_or_else(|| anyhow!("unknown folded output '{ref_id}'"))?;
                let (block_id, _) = folded_block_for_output(&projection, &ref_id)
                    .ok_or_else(|| anyhow!("orphan folded output '{ref_id}'"))?;
                ensure_block_openable(&projection, block_id)?;
                let open = projection
                    .open_folded_output(&ref_id, max_bytes)
                    .ok_or_else(|| anyhow!("unknown folded output '{ref_id}'"))?;
                Ok(json!({
                    "ok":true,
                    "ref_type":"folded_output",
                    "ref_id":ref_id,
                    "tool":metadata.tool_name,
                    "stream":metadata.stream,
                    "status":folded_status(metadata),
                    "command":metadata.shell_command,
                    "content":open.content,
                    "returned_bytes":open.returned_bytes,
                    "total_bytes":open.total_bytes,
                    "truncated":open.truncated
                }))
            }
            _ => bail!("unknown ref_type '{ref_type}'"),
        }
    }
}

#[async_trait]
impl ToolHandler for ContextPinTool {
    fn name(&self) -> &str {
        tool_names::TOOL_CONTEXT_PIN
    }
    fn description(&self) -> &str {
        "Pin a context block for future prompt inclusion."
    }
    fn permission_class(&self) -> ToolPermissionClass {
        ToolPermissionClass::Preview
    }
    fn parameters(&self) -> Value {
        block_mutation_schema()
    }
    async fn execute(&self, _args: Value) -> Result<Value> {
        bail!("context view projection unavailable")
    }
    async fn execute_with_context(
        &self,
        args: Value,
        context: ToolExecutionContext,
    ) -> Result<Value> {
        validate_block_operation(
            &context,
            &required_trimmed_string(&args, "block_id", MAX_ID_CHARS)?,
            "pin",
        )
    }
}

#[async_trait]
impl ToolHandler for ContextArchiveTool {
    fn name(&self) -> &str {
        tool_names::TOOL_CONTEXT_ARCHIVE
    }
    fn description(&self) -> &str {
        "Archive a context block from the default view."
    }
    fn permission_class(&self) -> ToolPermissionClass {
        ToolPermissionClass::Preview
    }
    fn parameters(&self) -> Value {
        block_mutation_schema()
    }
    async fn execute(&self, _args: Value) -> Result<Value> {
        bail!("context view projection unavailable")
    }
    async fn execute_with_context(
        &self,
        args: Value,
        context: ToolExecutionContext,
    ) -> Result<Value> {
        validate_block_operation(
            &context,
            &required_trimmed_string(&args, "block_id", MAX_ID_CHARS)?,
            "archive",
        )
    }
}

#[async_trait]
impl ToolHandler for ContextRemoveTool {
    fn name(&self) -> &str {
        tool_names::TOOL_CONTEXT_REMOVE
    }
    fn description(&self) -> &str {
        "Remove a context block from the default view."
    }
    fn permission_class(&self) -> ToolPermissionClass {
        ToolPermissionClass::Preview
    }
    fn parameters(&self) -> Value {
        block_mutation_schema()
    }
    async fn execute(&self, _args: Value) -> Result<Value> {
        bail!("context view projection unavailable")
    }
    async fn execute_with_context(
        &self,
        args: Value,
        context: ToolExecutionContext,
    ) -> Result<Value> {
        validate_block_operation(
            &context,
            &required_trimmed_string(&args, "block_id", MAX_ID_CHARS)?,
            "remove_from_view",
        )
    }
}

#[async_trait]
impl ToolHandler for ContextSummarizeTool {
    fn name(&self) -> &str {
        tool_names::TOOL_CONTEXT_SUMMARIZE
    }
    fn description(&self) -> &str {
        "Validate a proposed context summary artifact and return metadata to record."
    }
    fn permission_class(&self) -> ToolPermissionClass {
        ToolPermissionClass::Preview
    }
    fn parameters(&self) -> Value {
        json!({
            "type":"object",
            "properties":{
                "node_id":{"type":"string","maxLength":MAX_ID_CHARS},
                "artifact_id":{"type":"string","maxLength":MAX_ID_CHARS},
                "summary":{"type":"string","maxLength":MAX_SUMMARY_CHARS},
                "source_block_id":{"type":["string","null"],"maxLength":MAX_ID_CHARS},
                "source_node_id":{"type":["string","null"],"maxLength":MAX_ID_CHARS},
                "source_start_sequence":{"type":["integer","null"],"minimum":0},
                "source_end_sequence":{"type":["integer","null"],"minimum":0},
                "artifact_kind":{"type":["string","null"],"maxLength":MAX_ARTIFACT_KIND_CHARS},
                "version":{"type":["integer","null"],"minimum":1,"maximum":u32::MAX}
            },
            "required":["node_id","artifact_id","summary","source_block_id","source_node_id","source_start_sequence","source_end_sequence","artifact_kind","version"],
            "additionalProperties":false
        })
    }
    async fn execute(&self, _args: Value) -> Result<Value> {
        bail!("context view projection unavailable")
    }
    async fn execute_with_context(
        &self,
        args: Value,
        context: ToolExecutionContext,
    ) -> Result<Value> {
        let projection = require_projection(&context)?;
        let tree = require_context_tree(&context)?;
        let node_id = required_trimmed_string(&args, "node_id", MAX_ID_CHARS)?;
        let node_ref = ContextNodeId::new(node_id.clone())?;
        ensure!(
            tree.node(&node_ref).is_some(),
            "unknown context node '{node_id}'"
        );
        let artifact_id = required_trimmed_string(&args, "artifact_id", MAX_ID_CHARS)?;
        ensure!(
            projection
                .summary_artifacts
                .iter()
                .all(|artifact| artifact.artifact_id != artifact_id),
            "duplicate artifact_id '{artifact_id}'"
        );
        let summary = required_trimmed_string(&args, "summary", MAX_SUMMARY_CHARS)?;
        let source_block_id = optional_trimmed_string(&args, "source_block_id", MAX_ID_CHARS)?;
        if let Some(block_id) = &source_block_id {
            ensure!(
                projection.blocks.keys().any(|id| id.as_str() == block_id),
                "unknown context block '{block_id}'"
            );
        }
        let source_node_id = optional_trimmed_string(&args, "source_node_id", MAX_ID_CHARS)?;
        if let Some(source_node_id) = &source_node_id {
            let source_node_ref = ContextNodeId::new(source_node_id.clone())?;
            ensure!(
                tree.node(&source_node_ref).is_some(),
                "unknown context node '{source_node_id}'"
            );
        }
        let artifact_kind =
            optional_trimmed_string(&args, "artifact_kind", MAX_ARTIFACT_KIND_CHARS)?
                .unwrap_or_else(|| "summary".into());
        let version = parse_summary_version(&args, &projection, &node_id, &artifact_kind)?;
        let source_start_sequence = args.get("source_start_sequence").and_then(Value::as_u64);
        let source_end_sequence = args.get("source_end_sequence").and_then(Value::as_u64);
        ensure!(
            source_start_sequence.is_some() == source_end_sequence.is_some(),
            "source_start_sequence and source_end_sequence must both be provided"
        );
        ensure!(
            source_block_id.is_some()
                || source_node_id.is_some()
                || source_start_sequence.is_some(),
            "summary_metadata requires at least one traceable source"
        );
        if let (Some(start), Some(end)) = (source_start_sequence, source_end_sequence) {
            ensure!(
                start <= end,
                "source_start_sequence must be <= source_end_sequence"
            );
        }
        Ok(json!({
            "ok":true,
            "summary_metadata":{
                "node_id":node_id,
                "artifact_id":artifact_id,
                "artifact_kind":artifact_kind,
                "version":version,
                "summary":summary,
                "source_block_id":source_block_id,
                "source_node_id":source_node_id,
                "source_start_sequence":source_start_sequence,
                "source_end_sequence":source_end_sequence
            },
            "pending_recording":true
        }))
    }
}

fn block_mutation_schema() -> Value {
    json!({
        "type":"object",
        "properties":{"block_id":{"type":"string","maxLength":MAX_ID_CHARS}},
        "required":["block_id"],
        "additionalProperties":false
    })
}

fn validate_block_operation(
    context: &ToolExecutionContext,
    block_id: &str,
    operation: &str,
) -> Result<Value> {
    let projection = require_projection(context)?;
    let mut state = projection.view_state.clone();
    let op = match operation {
        "pin" => crate::context_view::ContextViewOperation::Pin {
            block_id: crate::context_view::ContextBlockId::new(block_id.to_string())?,
        },
        "archive" => crate::context_view::ContextViewOperation::Archive {
            block_id: crate::context_view::ContextBlockId::new(block_id.to_string())?,
        },
        "remove_from_view" => crate::context_view::ContextViewOperation::RemoveFromView {
            block_id: crate::context_view::ContextBlockId::new(block_id.to_string())?,
        },
        other => bail!("unsupported context operation '{other}'"),
    };
    state
        .apply(&projection.blocks, &op)
        .map_err(|error| anyhow!(error.to_string()))?;
    Ok(json!({
        "ok":true,
        "operation_metadata":{"operation":operation,"block_id":block_id},
        "pending_recording":true
    }))
}

fn require_projection(context: &ToolExecutionContext) -> Result<Arc<ContextViewProjection>> {
    context
        .context_view
        .clone()
        .ok_or_else(|| anyhow!("context view projection unavailable"))
}

fn require_context_tree(context: &ToolExecutionContext) -> Result<Arc<ContextTreeState>> {
    context
        .context_tree
        .clone()
        .ok_or_else(|| anyhow!("context tree snapshot unavailable"))
}

fn parse_limit(args: &Value, default: usize) -> Result<usize> {
    let limit = match args.get("limit") {
        Some(Value::Null) | None => default,
        Some(value) => usize::try_from(
            value
                .as_u64()
                .ok_or_else(|| anyhow!("limit must be integer or null"))?,
        )
        .map_err(|_| anyhow!("limit is too large"))?,
    };
    ensure!(
        limit > 0 && limit <= MAX_LIST_LIMIT,
        "limit must be between 1 and {MAX_LIST_LIMIT}"
    );
    Ok(limit)
}

fn parse_max_bytes(args: &Value) -> Result<usize> {
    let max_bytes = match args.get("max_bytes") {
        Some(Value::Null) | None => DEFAULT_OPEN_MAX_BYTES,
        Some(value) => usize::try_from(
            value
                .as_u64()
                .ok_or_else(|| anyhow!("max_bytes must be integer or null"))?,
        )
        .map_err(|_| anyhow!("max_bytes is too large"))?,
    };
    ensure!(
        max_bytes > 0 && max_bytes <= MAX_OPEN_MAX_BYTES,
        "max_bytes must be between 1 and {MAX_OPEN_MAX_BYTES}"
    );
    Ok(max_bytes)
}

fn required_trimmed_string(args: &Value, field: &str, max_chars: usize) -> Result<String> {
    let value = args
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("field '{field}' must be a string"))?;
    let trimmed = value.trim();
    ensure!(!trimmed.is_empty(), "field '{field}' must not be empty");
    ensure!(
        trimmed.chars().count() <= max_chars,
        "field '{field}' exceeds {max_chars} characters"
    );
    Ok(trimmed.to_string())
}

fn optional_trimmed_string(args: &Value, field: &str, max_chars: usize) -> Result<Option<String>> {
    match args.get(field) {
        Some(Value::Null) | None => Ok(None),
        Some(Value::String(value)) => {
            let trimmed = value.trim();
            ensure!(
                !trimmed.is_empty(),
                "field '{field}' must not be empty when provided"
            );
            ensure!(
                trimmed.chars().count() <= max_chars,
                "field '{field}' exceeds {max_chars} characters"
            );
            Ok(Some(trimmed.to_string()))
        }
        Some(_) => bail!("field '{field}' must be a string or null"),
    }
}

fn sorted_blocks(
    projection: &ContextViewProjection,
) -> Vec<(&crate::context_view::ContextBlockId, &ContextBlock)> {
    let mut blocks = projection.blocks.iter().collect::<Vec<_>>();
    blocks.sort_by(|(left_id, left), (right_id, right)| {
        left.source_start_sequence
            .or(left.available_sequence)
            .unwrap_or(u64::MAX)
            .cmp(
                &right
                    .source_start_sequence
                    .or(right.available_sequence)
                    .unwrap_or(u64::MAX),
            )
            .then_with(|| left_id.as_str().cmp(right_id.as_str()))
    });
    blocks
}

fn block_visible_for_listing(
    projection: &ContextViewProjection,
    block_id: &str,
    block: &ContextBlock,
    include_archived: bool,
    include_removed: bool,
) -> bool {
    if block.is_protected() {
        return true;
    }
    match block_status_string(projection, block_id).as_str() {
        "visible" | "pinned" => true,
        "archived" => include_archived,
        "removed_from_view" => include_removed,
        _ => true,
    }
}

fn block_status_string(projection: &ContextViewProjection, block_id: &str) -> String {
    match projection
        .blocks
        .keys()
        .find(|id| id.as_str() == block_id)
        .and_then(|id| projection.view_state.status(id))
        .unwrap_or(ContextViewStatus::Visible)
    {
        ContextViewStatus::Visible => "visible".to_string(),
        ContextViewStatus::Pinned => "pinned".to_string(),
        ContextViewStatus::Archived => "archived".to_string(),
        ContextViewStatus::RemovedFromView => "removed_from_view".to_string(),
    }
}

fn block_ref_json(
    projection: &ContextViewProjection,
    block_id: &str,
    block: &ContextBlock,
) -> Value {
    json!({
        "ref_type":"block",
        "ref_id":block_id,
        "title":block.title,
        "kind":context_block_kind_label(block.kind),
        "status":block_status_string(projection, block_id),
        "source":format_block_source(&block.source),
        "detail":truncate(&block.detail, 160)
    })
}

fn node_ref_json(tree: &ContextTreeState, node: &ContextNodeRecord) -> Value {
    json!({
        "ref_type":"node",
        "ref_id":node.node_id.as_str(),
        "node_id":node.node_id.as_str(),
        "parent_node_id":node.parent_node_id.as_ref().map(|id| id.as_str()),
        "label":node.label.clone(),
        "purpose":node.purpose.clone(),
        "status":node_status_label(&node.status),
        "active":tree.active_node_id() == Some(&node.node_id),
        "block_ref":node.block_ref.clone(),
        "source_ref":node.source_ref.clone(),
    })
}

fn open_node_json(
    projection: &ContextViewProjection,
    tree: &ContextTreeState,
    node: &ContextNodeRecord,
    max_bytes: usize,
) -> Value {
    let child_node_ids = tree
        .nodes()
        .filter(|candidate| candidate.parent_node_id.as_ref() == Some(&node.node_id))
        .map(|candidate| candidate.node_id.as_str().to_string())
        .collect::<Vec<_>>();
    let summaries = projection
        .summary_artifacts
        .iter()
        .filter(|artifact| {
            artifact.node_id == node.node_id.as_str()
                || artifact.source_node_id.as_deref() == Some(node.node_id.as_str())
        })
        .map(|artifact| {
            json!({
                "artifact_id":artifact.artifact_id,
                "node_id":artifact.node_id,
                "artifact_kind":artifact.artifact_kind,
                "version":artifact.version,
                "summary":truncate(&artifact.summary, max_bytes),
                "source_node_id":artifact.source_node_id,
                "source_block_id":artifact.source_block_id,
            })
        })
        .collect::<Vec<_>>();
    let referenced_block = node.block_ref.as_ref().and_then(|block_ref| {
        open_node_referenced_block_json(projection, &block_ref.block_id, max_bytes)
    });

    json!({
        "ok":true,
        "ref_type":"node",
        "ref_id":node.node_id.as_str(),
        "node_id":node.node_id.as_str(),
        "parent_node_id":node.parent_node_id.as_ref().map(|id| id.as_str()),
        "label":node.label.clone(),
        "purpose":node.purpose.clone(),
        "status":node_status_label(&node.status),
        "active":tree.active_node_id() == Some(&node.node_id),
        "block_ref":node.block_ref.clone(),
        "source_ref":node.source_ref.clone(),
        "child_node_ids":child_node_ids,
        "related_summaries":summaries,
        "referenced_block":referenced_block,
    })
}

fn open_node_referenced_block_json(
    projection: &ContextViewProjection,
    block_id: &str,
    max_bytes: usize,
) -> Option<Value> {
    let block = projection
        .blocks
        .iter()
        .find(|(id, _)| id.as_str() == block_id)
        .map(|(_, block)| block)?;
    let status = block_status_string(projection, block_id);
    let mut value = json!({
        "ref_id":block_id,
        "status":status.clone(),
        "title":block.title,
        "kind":context_block_kind_label(block.kind),
    });
    if status != "removed_from_view" {
        value["detail"] = json!(truncate(&block.detail, max_bytes));
    }
    Some(value)
}

fn ensure_block_openable(projection: &ContextViewProjection, block_id: &str) -> Result<String> {
    let status = block_status_string(projection, block_id);
    ensure!(
        status != "removed_from_view",
        "cannot open removed context block '{block_id}'"
    );
    Ok(status)
}

fn folded_output_ref_json(metadata: &FoldedOutputMetadata) -> Value {
    json!({
        "ref_type":"folded_output",
        "ref_id":metadata.output_id,
        "tool":metadata.tool_name,
        "stream":metadata.stream,
        "status":folded_status(metadata),
        "command":metadata.shell_command,
        "size_bytes":metadata.byte_count
    })
}

fn summary_ref_json(artifact: &crate::context_view::SummaryArtifact) -> Value {
    json!({
        "ref_type":"summary",
        "ref_id":artifact.artifact_id,
        "node_id":artifact.node_id,
        "artifact_kind":artifact.artifact_kind,
        "version":artifact.version,
        "summary":truncate(&artifact.summary,160),
        "source_block_id":artifact.source_block_id,
        "source_node_id":artifact.source_node_id,
        "source_start_sequence":artifact.source_start_sequence,
        "source_end_sequence":artifact.source_end_sequence
    })
}

fn folded_output_match_json(
    projection: &ContextViewProjection,
    block_id: &str,
    block: &ContextBlock,
    metadata: &FoldedOutputMetadata,
) -> Value {
    json!({
        "ref_type":"folded_output",
        "ref_id":metadata.output_id,
        "tool":metadata.tool_name,
        "command":metadata.shell_command,
        "stream":metadata.stream,
        "output_kind":metadata.output_kind,
        "status":folded_status(metadata),
        "source_start_sequence":metadata.source_start_sequence,
        "source_end_sequence":metadata.source_end_sequence,
        "byte_count":metadata.byte_count,
        "line_count":metadata.line_count,
        "block_status":block_status_string(projection, block_id),
        "block_source":format_block_source(&block.source),
    })
}

fn folded_block_for_output<'a>(
    projection: &'a ContextViewProjection,
    output_id: &str,
) -> Option<(&'a str, &'a ContextBlock)> {
    projection.blocks.iter().find_map(|(id, block)| {
        (block.folded_output_id.as_deref() == Some(output_id)).then_some((id.as_str(), block))
    })
}

fn context_block_kind_label(kind: ContextBlockKind) -> &'static str {
    match kind {
        ContextBlockKind::HardConstraint => "hard_constraint",
        ContextBlockKind::CurrentUserRequirement => "current_user_requirement",
        ContextBlockKind::UnresolvedError => "unresolved_error",
        ContextBlockKind::Permission => "permission",
        ContextBlockKind::FileWriteFact => "file_write_fact",
        ContextBlockKind::TestResult => "test_result",
        ContextBlockKind::CommitHash => "commit_hash",
        ContextBlockKind::ToolOutput => "tool_output",
        ContextBlockKind::Note => "note",
    }
}

fn node_status_label(status: &ContextNodeStatus) -> &'static str {
    match status {
        ContextNodeStatus::Active => "active",
        ContextNodeStatus::Inactive => "inactive",
        ContextNodeStatus::Archived => "archived",
    }
}

fn format_source_ref(source_ref: &crate::context_tree::ContextSourceRef) -> String {
    match source_ref.source_id.as_deref() {
        Some(source_id) => format!("{}:{}", source_ref.source_kind, source_id),
        None => source_ref.source_kind.clone(),
    }
}

fn format_block_source(source: &ContextBlockSource) -> String {
    match source {
        ContextBlockSource::TranscriptSpan {
            start_sequence,
            end_sequence,
        } => format!("transcript:{start_sequence}..{end_sequence}"),
        ContextBlockSource::SummaryArtifact { artifact_id } => format!("summary:{artifact_id}"),
        ContextBlockSource::FoldedOutput { output_id } => format!("folded:{output_id}"),
    }
}

fn folded_status(metadata: &FoldedOutputMetadata) -> String {
    match (metadata.exit_status, metadata.tool_ok) {
        (Some(status), Some(ok)) => format!("status={status},ok={ok}"),
        (Some(status), None) => format!("status={status}"),
        (None, Some(ok)) => format!("ok={ok}"),
        (None, None) => "unknown".into(),
    }
}

fn next_summary_version(
    projection: &ContextViewProjection,
    node_id: &str,
    artifact_kind: &str,
) -> u32 {
    projection
        .summary_artifacts
        .iter()
        .filter(|artifact| artifact.node_id == node_id && artifact.artifact_kind == artifact_kind)
        .map(|artifact| artifact.version)
        .max()
        .unwrap_or(0)
        .saturating_add(1)
}

fn parse_summary_version(
    args: &Value,
    projection: &ContextViewProjection,
    node_id: &str,
    artifact_kind: &str,
) -> Result<u32> {
    match args.get("version") {
        Some(Value::Null) | None => Ok(next_summary_version(projection, node_id, artifact_kind)),
        Some(value) => {
            let raw = value
                .as_u64()
                .ok_or_else(|| anyhow!("version must be integer or null"))?;
            let version =
                u32::try_from(raw).map_err(|_| anyhow!("version must be <= {}", u32::MAX))?;
            ensure!(version > 0, "version must be >= 1");
            Ok(version)
        }
    }
}

fn truncate(text: &str, max_chars: usize) -> String {
    let mut truncated = text.chars().take(max_chars).collect::<String>();
    if text.chars().count() > max_chars {
        truncated.push('…');
    }
    truncated
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(sequence: u64, event: TranscriptEvent) -> TranscriptRecord {
        TranscriptRecord {
            session_id: "s".into(),
            sequence,
            timestamp_ms: 0,
            context_branch_id: None,
            event,
        }
    }

    fn base_records() -> Vec<TranscriptRecord> {
        vec![
            record(
                1,
                TranscriptEvent::UserMessage {
                    content: UserMessageContent::from("Do not drop this requirement"),
                },
            ),
            record(
                2,
                TranscriptEvent::AssistantMessage {
                    content: "visible note".into(),
                },
            ),
            record(
                3,
                TranscriptEvent::ContextViewOperationMetadata {
                    operation: "pin".into(),
                    node_id: None,
                    block_id: Some("block-seq-2-note".into()),
                    detail: None,
                },
            ),
            record(
                4,
                TranscriptEvent::ToolCallStarted {
                    call_id: "call-1".into(),
                    name: "shell__exec".into(),
                    args: json!({"command":"cargo test"}),
                },
            ),
            record(
                5,
                TranscriptEvent::ToolCallFinished {
                    call_id: "call-1".into(),
                    name: "shell__exec".into(),
                    ok: true,
                    output: crate::tool::ToolResult::ok(
                        "shell__exec",
                        json!({"status":0,"stdout":"x".repeat(5000),"stdout_truncated":false,"stderr":"","stderr_truncated":false}),
                    ),
                },
            ),
        ]
    }

    fn tree_records() -> Vec<TranscriptRecord> {
        vec![
            record(
                6,
                TranscriptEvent::ContextNodeCreated {
                    node_id: "node-a".into(),
                    parent_node_id: Some("root".into()),
                    label: Some("Build node".into()),
                    purpose: Some("Track build context".into()),
                    block_ref: Some(crate::context_tree::ContextBlockRef {
                        block_id: "block-seq-2-note".into(),
                    }),
                    source_ref: Some(crate::context_tree::ContextSourceRef {
                        source_kind: "summary".into(),
                        source_id: Some("sum-node-a".into()),
                    }),
                },
            ),
            record(
                7,
                TranscriptEvent::ContextNodeLifecycle {
                    node_id: "node-a".into(),
                    status: crate::context_tree::ContextNodeStatus::Inactive,
                },
            ),
            record(
                8,
                TranscriptEvent::ContextSummaryArtifactMetadata {
                    node_id: "node-a".into(),
                    artifact_id: "sum-node-a".into(),
                    artifact_kind: "summary".into(),
                    version: Some(1),
                    summary: Some("Node summary".into()),
                    source_node_id: Some("node-a".into()),
                    source_block_id: Some("block-seq-2-note".into()),
                    source_start_sequence: Some(2),
                    source_end_sequence: Some(2),
                },
            ),
        ]
    }

    fn projection_with_tree_data() -> Arc<ContextViewProjection> {
        let mut records = base_records();
        records.extend(tree_records());
        Arc::new(project_context_view(&records).expect("projection with tree data"))
    }

    fn tree() -> Arc<ContextTreeState> {
        let mut records = base_records();
        records.extend(tree_records());
        Arc::new(project_context_tree(&records).expect("context tree"))
    }

    fn projection_with_removed_folded_output() -> Arc<ContextViewProjection> {
        Arc::new(project_context_view(&[
            record(1, TranscriptEvent::ToolCallStarted { call_id: "call-1".into(), name: "shell__exec".into(), args: json!({"command":"cargo test"}) }),
            record(2, TranscriptEvent::ToolCallFinished { call_id: "call-1".into(), name: "shell__exec".into(), ok: true, output: crate::tool::ToolResult::ok("shell__exec", json!({"status":0,"stdout":"x".repeat(5000),"stdout_truncated":false,"stderr":"","stderr_truncated":false})) }),
            record(3, TranscriptEvent::ContextViewOperationMetadata { operation: "remove_from_view".into(), node_id: None, block_id: Some("block-seq-2-folded-output-folded-output-seq-2-stdout".into()), detail: None }),
        ]).expect("projection with removed folded output"))
    }

    fn projection_with_summary_artifact() -> Arc<ContextViewProjection> {
        Arc::new(
            project_context_view(&[
                record(
                    1,
                    TranscriptEvent::AssistantMessage {
                        content: "visible note".into(),
                    },
                ),
                record(
                    2,
                    TranscriptEvent::ContextSummaryArtifactMetadata {
                        node_id: "node-a".into(),
                        artifact_id: "sum-1".into(),
                        artifact_kind: "summary".into(),
                        version: Some(1),
                        summary: Some("Existing summary".into()),
                        source_node_id: None,
                        source_block_id: Some("block-seq-1-note".into()),
                        source_start_sequence: Some(1),
                        source_end_sequence: Some(1),
                    },
                ),
            ])
            .expect("projection with summary artifact"),
        )
    }

    fn context(
        projection: Option<Arc<ContextViewProjection>>,
        tree: Option<Arc<ContextTreeState>>,
    ) -> ToolExecutionContext {
        ToolExecutionContext {
            allow_outside_workspace: false,
            context_view: projection,
            context_tree: tree,
        }
    }

    #[tokio::test]
    async fn context_tools_fail_fast_when_projection_is_missing() {
        let registry = ToolRegistry::default_tools();
        let output = registry
            .call_with_context(
                tool_names::TOOL_CONTEXT_LIST,
                json!({"include_archived":false,"include_removed":false,"limit":null}),
                context(None, None),
            )
            .await;
        assert!(!output.ok);
        assert!(
            output
                .error
                .as_ref()
                .is_some_and(|error| error.message.contains("projection unavailable"))
        );
    }

    #[tokio::test]
    async fn context_list_search_and_open_work_from_snapshot() {
        let registry = ToolRegistry::default_tools();
        let projection = projection_with_tree_data();
        let tree = tree();
        let listed = registry
            .call_with_context(
                tool_names::TOOL_CONTEXT_LIST,
                json!({"include_archived":false,"include_removed":false,"limit":null}),
                context(Some(projection.clone()), Some(tree.clone())),
            )
            .await;
        assert!(listed.ok, "{listed:?}");
        let nodes = listed
            .data
            .as_ref()
            .and_then(|d| d.get("nodes"))
            .and_then(Value::as_array)
            .expect("nodes");
        assert!(nodes.iter().any(|node| node["ref_id"] == "node-a"));
        let blocks = listed
            .data
            .as_ref()
            .and_then(|d| d.get("blocks"))
            .and_then(Value::as_array)
            .expect("blocks");
        assert!(
            blocks
                .iter()
                .any(|block| block["ref_id"] == "block-seq-2-note")
        );

        let searched = registry.call_with_context(tool_names::TOOL_CONTEXT_SEARCH, json!({"query":"cargo test","include_archived":false,"include_removed":false,"limit":null}), context(Some(projection.clone()), Some(tree.clone()))).await;
        assert!(searched.ok, "{searched:?}");
        assert!(
            searched
                .data
                .as_ref()
                .and_then(|d| d.get("matches"))
                .and_then(Value::as_array)
                .is_some_and(|items| !items.is_empty())
        );

        let opened = registry
            .call_with_context(
                tool_names::TOOL_CONTEXT_OPEN,
                json!({"ref_type":"block","ref_id":"block-seq-2-note","max_bytes":null}),
                context(Some(projection), Some(tree)),
            )
            .await;
        assert!(opened.ok, "{opened:?}");
        assert_eq!(
            opened
                .data
                .as_ref()
                .and_then(|d| d.get("operation_metadata"))
                .and_then(|d| d.get("operation")),
            Some(&json!("open_detail"))
        );
    }

    #[tokio::test]
    async fn context_mutations_validate_and_fail_atomically() {
        let registry = ToolRegistry::default_tools();
        let projection = projection_with_tree_data();
        let pinned = registry
            .call_with_context(
                tool_names::TOOL_CONTEXT_PIN,
                json!({"block_id":"block-seq-2-note"}),
                context(Some(projection.clone()), None),
            )
            .await;
        assert!(pinned.ok, "{pinned:?}");
        assert_eq!(
            pinned
                .data
                .as_ref()
                .and_then(|d| d.get("pending_recording")),
            Some(&json!(true))
        );

        let protected_archive = registry
            .call_with_context(
                tool_names::TOOL_CONTEXT_ARCHIVE,
                json!({"block_id":"block-seq-1-user-requirement"}),
                context(Some(projection.clone()), None),
            )
            .await;
        assert!(!protected_archive.ok);
        assert!(
            protected_archive
                .error
                .as_ref()
                .is_some_and(|error| error.message.contains("protected context block"))
        );

        let still_listed = registry
            .call_with_context(
                tool_names::TOOL_CONTEXT_LIST,
                json!({"include_archived":false,"include_removed":false,"limit":null}),
                context(Some(projection), Some(tree())),
            )
            .await;
        assert!(still_listed.ok);
        assert!(
            still_listed
                .data
                .as_ref()
                .and_then(|d| d.get("blocks"))
                .and_then(Value::as_array)
                .is_some_and(|blocks| blocks
                    .iter()
                    .any(|block| block["ref_id"] == "block-seq-1-user-requirement"))
        );
    }

    #[tokio::test]
    async fn context_summarize_validates_and_returns_metadata() {
        let registry = ToolRegistry::default_tools();
        let projection = projection_with_tree_data();
        let output = registry
            .call_with_context(
                tool_names::TOOL_CONTEXT_SUMMARIZE,
                json!({
                    "node_id":"node-a",
                    "artifact_id":"sum-1",
                    "summary":"Valid summary",
                    "source_block_id":"block-seq-2-note",
                    "source_node_id":null,
                    "source_start_sequence":2,
                    "source_end_sequence":2,
                    "artifact_kind":null,
                    "version":null
                }),
                context(Some(projection), Some(tree())),
            )
            .await;
        assert!(output.ok, "{output:?}");
        assert_eq!(
            output
                .data
                .as_ref()
                .and_then(|d| d.get("summary_metadata"))
                .and_then(|d| d.get("version")),
            Some(&json!(2))
        );
    }

    #[tokio::test]
    async fn context_open_rejects_removed_folded_output() {
        let registry = ToolRegistry::default_tools();
        let projection = projection_with_removed_folded_output();
        let output = registry
            .call_with_context(
                tool_names::TOOL_CONTEXT_OPEN,
                json!({
                    "ref_type":"folded_output",
                    "ref_id":"folded-output-seq-2-stdout",
                    "max_bytes":null
                }),
                context(Some(projection), Some(tree())),
            )
            .await;

        assert!(!output.ok);
        assert!(output.error.as_ref().is_some_and(|error| error.message.contains("cannot open removed context block 'block-seq-2-folded-output-folded-output-seq-2-stdout'")));
    }

    #[tokio::test]
    async fn context_summarize_rejects_duplicate_artifact_id() {
        let registry = ToolRegistry::default_tools();
        let projection = projection_with_summary_artifact();
        let output = registry
            .call_with_context(
                tool_names::TOOL_CONTEXT_SUMMARIZE,
                json!({
                    "node_id":"node-a",
                    "artifact_id":"sum-1",
                    "summary":"Duplicate summary",
                    "source_block_id":"block-seq-1-note",
                    "source_node_id":null,
                    "source_start_sequence":1,
                    "source_end_sequence":1,
                    "artifact_kind":null,
                    "version":null
                }),
                context(Some(projection), Some(tree())),
            )
            .await;

        assert!(!output.ok);
        assert!(
            output
                .error
                .as_ref()
                .is_some_and(|error| error.message.contains("duplicate artifact_id 'sum-1'"))
        );
    }

    #[tokio::test]
    async fn context_summarize_rejects_oversized_version() {
        let registry = ToolRegistry::default_tools();
        let projection = projection_with_tree_data();
        let output = registry
            .call_with_context(
                tool_names::TOOL_CONTEXT_SUMMARIZE,
                json!({
                    "node_id":"node-a",
                    "artifact_id":"sum-oversized",
                    "summary":"Valid summary",
                    "source_block_id":"block-seq-2-note",
                    "source_node_id":null,
                    "source_start_sequence":2,
                    "source_end_sequence":2,
                    "artifact_kind":null,
                    "version":u64::from(u32::MAX) + 1
                }),
                context(Some(projection), Some(tree())),
            )
            .await;

        assert!(!output.ok);
        assert!(output.error.as_ref().is_some_and(|error| {
            error
                .message
                .contains(&format!("version must be <= {}", u32::MAX))
        }));
    }

    #[tokio::test]
    async fn context_summarize_rejects_missing_traceable_source() {
        let registry = ToolRegistry::default_tools();
        let projection = projection_with_tree_data();
        let output = registry
            .call_with_context(
                tool_names::TOOL_CONTEXT_SUMMARIZE,
                json!({
                    "node_id":"node-a",
                    "artifact_id":"sum-missing-source",
                    "summary":"Valid summary",
                    "source_block_id":null,
                    "source_node_id":null,
                    "source_start_sequence":null,
                    "source_end_sequence":null,
                    "artifact_kind":null,
                    "version":null
                }),
                context(Some(projection), Some(tree())),
            )
            .await;

        assert!(!output.ok);
        assert!(output.error.as_ref().is_some_and(|error| {
            error
                .message
                .contains("requires at least one traceable source")
        }));
    }

    #[tokio::test]
    async fn context_summarize_rejects_unknown_source_node_id() {
        let registry = ToolRegistry::default_tools();
        let projection = projection_with_tree_data();
        let output = registry
            .call_with_context(
                tool_names::TOOL_CONTEXT_SUMMARIZE,
                json!({
                    "node_id":"node-a",
                    "artifact_id":"sum-unknown-source-node",
                    "summary":"Valid summary",
                    "source_block_id":null,
                    "source_node_id":"node-missing",
                    "source_start_sequence":null,
                    "source_end_sequence":null,
                    "artifact_kind":null,
                    "version":null
                }),
                context(Some(projection), Some(tree())),
            )
            .await;

        assert!(!output.ok);
        assert!(output.error.as_ref().is_some_and(|error| {
            error
                .message
                .contains("unknown context node 'node-missing'")
        }));
    }

    #[tokio::test]
    async fn context_summarize_rejects_unknown_node_id() {
        let registry = ToolRegistry::default_tools();
        let projection = projection_with_tree_data();
        let output = registry
            .call_with_context(
                tool_names::TOOL_CONTEXT_SUMMARIZE,
                json!({
                    "node_id":"node-missing",
                    "artifact_id":"sum-unknown-node",
                    "summary":"Valid summary",
                    "source_block_id":"block-seq-2-note",
                    "source_node_id":null,
                    "source_start_sequence":2,
                    "source_end_sequence":2,
                    "artifact_kind":null,
                    "version":null
                }),
                context(Some(projection), Some(tree())),
            )
            .await;

        assert!(!output.ok);
        assert!(output.error.as_ref().is_some_and(|error| {
            error
                .message
                .contains("unknown context node 'node-missing'")
        }));
    }

    #[tokio::test]
    async fn context_search_finds_node() {
        let registry = ToolRegistry::default_tools();
        let projection = projection_with_tree_data();
        let output = registry
            .call_with_context(
                tool_names::TOOL_CONTEXT_SEARCH,
                json!({
                    "query":"Build node",
                    "include_archived":false,
                    "include_removed":false,
                    "limit":null
                }),
                context(Some(projection), Some(tree())),
            )
            .await;

        assert!(output.ok, "{output:?}");
        assert!(
            output
                .data
                .as_ref()
                .and_then(|d| d.get("matches"))
                .and_then(Value::as_array)
                .is_some_and(|items| items
                    .iter()
                    .any(|item| item["ref_type"] == "node" && item["ref_id"] == "node-a"))
        );
    }

    #[tokio::test]
    async fn context_search_returns_rich_block_and_folded_payloads() {
        let registry = ToolRegistry::default_tools();
        let projection = projection_with_tree_data();
        let block_output = registry
            .call_with_context(
                tool_names::TOOL_CONTEXT_SEARCH,
                json!({
                    "query":"visible note",
                    "include_archived":false,
                    "include_removed":false,
                    "limit":null
                }),
                context(Some(projection), Some(tree())),
            )
            .await;

        assert!(block_output.ok, "{block_output:?}");
        let block_matches = block_output
            .data
            .as_ref()
            .and_then(|d| d.get("matches"))
            .and_then(Value::as_array)
            .expect("block matches");
        let block = block_matches
            .iter()
            .find(|item| item["ref_type"] == "block" && item["ref_id"] == "block-seq-2-note")
            .expect("note block match");
        assert_eq!(block.get("kind"), Some(&json!("note")));
        assert_eq!(block.get("status"), Some(&json!("pinned")));
        assert_eq!(block.get("source"), Some(&json!("transcript:2..2")));

        let folded_output = registry
            .call_with_context(
                tool_names::TOOL_CONTEXT_SEARCH,
                json!({
                    "query":"cargo test",
                    "include_archived":false,
                    "include_removed":false,
                    "limit":null
                }),
                context(Some(projection_with_tree_data()), Some(tree())),
            )
            .await;

        assert!(folded_output.ok, "{folded_output:?}");
        let folded_matches = folded_output
            .data
            .as_ref()
            .and_then(|d| d.get("matches"))
            .and_then(Value::as_array)
            .expect("folded matches");
        let folded = folded_matches
            .iter()
            .find(|item| item["ref_type"] == "folded_output")
            .expect("folded match");
        assert_eq!(folded.get("stream"), Some(&json!("stdout")));
        assert_eq!(folded.get("output_kind"), Some(&json!("shell_output")));
        assert_eq!(folded.get("block_status"), Some(&json!("visible")));
        assert!(folded.get("block_source").is_some());
        assert!(folded.get("byte_count").is_some());
        assert!(folded.get("line_count").is_some());
    }

    #[tokio::test]
    async fn context_search_finds_summary_source_metadata() {
        let registry = ToolRegistry::default_tools();
        let projection = projection_with_tree_data();
        let output = registry
            .call_with_context(
                tool_names::TOOL_CONTEXT_SEARCH,
                json!({
                    "query":"sum-node-a",
                    "include_archived":false,
                    "include_removed":false,
                    "limit":null
                }),
                context(Some(projection), Some(tree())),
            )
            .await;

        assert!(output.ok, "{output:?}");
        assert!(
            output
                .data
                .as_ref()
                .and_then(|d| d.get("matches"))
                .and_then(Value::as_array)
                .is_some_and(|items| items.iter().any(|item| {
                    item["ref_type"] == "summary"
                        && item["ref_id"] == "sum-node-a"
                        && item["node_id"] == "node-a"
                        && item["artifact_kind"] == "summary"
                        && item["version"] == 1
                        && item["source_node_id"] == "node-a"
                        && item["source_block_id"] == "block-seq-2-note"
                        && item["source_start_sequence"] == 2
                        && item["source_end_sequence"] == 2
                }))
        );
    }

    #[tokio::test]
    async fn context_open_node_by_id_returns_snapshot_metadata() {
        let registry = ToolRegistry::default_tools();
        let projection = projection_with_tree_data();
        let output = registry
            .call_with_context(
                tool_names::TOOL_CONTEXT_OPEN,
                json!({
                    "ref_type":"node",
                    "ref_id":"node-a",
                    "max_bytes":32
                }),
                context(Some(projection), Some(tree())),
            )
            .await;

        assert!(output.ok, "{output:?}");
        let data = output.data.expect("open node data");
        assert_eq!(data.get("node_id"), Some(&json!("node-a")));
        assert_eq!(data.get("parent_node_id"), Some(&json!("root")));
        assert_eq!(data.get("status"), Some(&json!("inactive")));
        assert!(
            data.get("related_summaries")
                .and_then(Value::as_array)
                .is_some_and(|items| items.iter().any(|item| item["artifact_id"] == "sum-node-a"))
        );
        assert_eq!(
            data.get("referenced_block")
                .and_then(|block| block.get("ref_id")),
            Some(&json!("block-seq-2-note"))
        );
    }
}
