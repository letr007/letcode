use anyhow::{Result, anyhow, bail, ensure};
use async_trait::async_trait;
use serde_json::{Value, json};
use std::convert::TryFrom;
use std::sync::Arc;

use crate::context_tree::{ContextNodeId, ContextNodeRecord, ContextNodeStatus, ContextTreeState};
#[cfg(test)]
use crate::context_view::project_context_view;
use crate::context_view::{
    ContextBlock, ContextBlockKind, ContextBlockRetention, ContextBlockSource,
    ContextViewProjection, ContextViewStatus, FoldedOutputMetadata,
};
use crate::permission::ToolPermissionClass;
use crate::runtime_context::{RuntimeSnapshot, SourceSpan};
use crate::tool::{ToolExecutionContext, ToolHandler, ToolRegistry};
use crate::tool_names;
#[cfg(test)]
use crate::transcript::transcript_projection::project_context_tree;
#[cfg(test)]
use crate::transcript::{TranscriptEvent, TranscriptRecord};
#[cfg(test)]
use crate::user_content::UserMessageContent;

const DEFAULT_LIST_LIMIT: usize = 20;
const MAX_LIST_LIMIT: usize = 100;
const DEFAULT_OPEN_MAX_BYTES: usize = 2048;
const MAX_OPEN_MAX_BYTES: usize = 16 * 1024;
const DEFAULT_GREP_CONTEXT_LINES: usize = 2;
const MAX_GREP_CONTEXT_LINES: usize = 5;
const DEFAULT_GREP_MAX_MATCHES: usize = 10;
const MAX_GREP_MAX_MATCHES: usize = 50;
const MAX_GREP_LINE_CHARS: usize = 2048;
const MAX_QUERY_CHARS: usize = 256;
const MAX_ID_CHARS: usize = 256;
const MAX_SUMMARY_CHARS: usize = 4000;
const MAX_ARTIFACT_KIND_CHARS: usize = 64;

pub(crate) fn register_context_tools(registry: &mut ToolRegistry) {
    registry.register(ContextListTool);
    registry.register(ContextSearchTool);
    registry.register(ContextGrepTool);
    registry.register(ContextOpenTool);
    registry.register(ContextSummarizeTool);
    registry.register(ContextPinTool);
    registry.register(ContextArchiveTool);
    registry.register(ContextRemoveTool);
    registry.register(ContextResolveTool);
}

struct ContextListTool;
struct ContextSearchTool;
struct ContextGrepTool;
struct ContextOpenTool;
struct ContextSummarizeTool;
struct ContextPinTool;
struct ContextArchiveTool;
struct ContextRemoveTool;
struct ContextResolveTool;

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
            .filter(|node| {
                node_visible_for_listing(&projection, node, include_archived, include_removed)
            })
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
        for node in tree.nodes().filter(|node| {
            node_visible_for_listing(&projection, node, include_archived, include_removed)
        }) {
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
            ) || projection.status_for(id) == ContextViewStatus::RemovedFromView
            {
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
        for (id, block) in sorted_blocks(&projection) {
            if matches.len() >= limit {
                break;
            }
            let Some(output_id) = block.folded_output_id.as_deref() else {
                continue;
            };
            let Some(metadata) = projection.folded_outputs.get(output_id) else {
                continue;
            };
            let block_id = id.as_str();
            if !block_visible_for_listing(
                &projection,
                block_id,
                block,
                include_archived,
                include_removed,
            ) || projection.status_for(id) == ContextViewStatus::RemovedFromView
            {
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
        let snapshot = require_runtime_snapshot(&context)?;
        let projection = &snapshot.context_view;
        let tree = &snapshot.context_tree;
        let ref_type = required_trimmed_string(&args, "ref_type", 32)?;
        let ref_id = required_trimmed_string(&args, "ref_id", MAX_ID_CHARS)?;
        let max_bytes = parse_max_bytes(&args)?;
        match ref_type.as_str() {
            "node" => {
                let node_id = ContextNodeId::new(ref_id.clone())?;
                let node = tree
                    .node(&node_id)
                    .ok_or_else(|| anyhow!("unknown context node '{ref_id}'"))?;
                validate_node_source(snapshot.as_ref(), node)?;
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
impl ToolHandler for ContextGrepTool {
    fn name(&self) -> &str {
        tool_names::TOOL_CONTEXT_GREP
    }
    fn description(&self) -> &str {
        "Search a folded output and return bounded line snippets around matches."
    }
    fn permission_class(&self) -> ToolPermissionClass {
        ToolPermissionClass::Read
    }
    fn parameters(&self) -> Value {
        json!({
            "type":"object",
            "properties":{
                "ref_id":{"type":"string","maxLength":MAX_ID_CHARS},
                "query":{"type":"string","maxLength":MAX_QUERY_CHARS},
                "case_sensitive":{"type":"boolean"},
                "context_lines":{"type":["integer","null"],"minimum":0,"maximum":MAX_GREP_CONTEXT_LINES},
                "max_matches":{"type":["integer","null"],"minimum":1,"maximum":MAX_GREP_MAX_MATCHES}
            },
            "required":["ref_id","query","case_sensitive","context_lines","max_matches"],
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
        let ref_id = required_trimmed_string(&args, "ref_id", MAX_ID_CHARS)?;
        let query = required_trimmed_string(&args, "query", MAX_QUERY_CHARS)?;
        let case_sensitive = args
            .get("case_sensitive")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let context_lines = parse_nullable_usize(
            &args,
            "context_lines",
            DEFAULT_GREP_CONTEXT_LINES,
            0,
            MAX_GREP_CONTEXT_LINES,
        )?;
        let max_matches = parse_nullable_usize(
            &args,
            "max_matches",
            DEFAULT_GREP_MAX_MATCHES,
            1,
            MAX_GREP_MAX_MATCHES,
        )?;
        let metadata = projection
            .folded_outputs
            .get(&ref_id)
            .ok_or_else(|| anyhow!("unknown folded output '{ref_id}'"))?;
        let (block_id, _) = folded_block_for_output(&projection, &ref_id)
            .ok_or_else(|| anyhow!("orphan folded output '{ref_id}'"))?;
        ensure_block_openable(&projection, block_id)?;

        let lines = metadata.content.lines().collect::<Vec<_>>();
        let query_cmp = if case_sensitive {
            query.clone()
        } else {
            query.to_ascii_lowercase()
        };
        let matching_lines = lines
            .iter()
            .enumerate()
            .filter_map(|(index, line)| {
                line_match_bounds(line, &query_cmp, case_sensitive).map(|_| index)
            })
            .collect::<Vec<_>>();
        let total_matching_lines = matching_lines.len();
        let returned_matching_lines = total_matching_lines.min(max_matches);
        let selected_matching_lines = matching_lines
            .iter()
            .copied()
            .take(max_matches)
            .collect::<Vec<_>>();
        let matches = build_grep_match_groups(
            &lines,
            &selected_matching_lines,
            &query_cmp,
            case_sensitive,
            context_lines,
        );
        let truncated = total_matching_lines > returned_matching_lines;

        Ok(json!({
            "ok":true,
            "ref_type":"folded_output",
            "ref_id":ref_id,
            "query":query,
            "case_sensitive":case_sensitive,
            "context_lines":context_lines,
            "max_matches":max_matches,
            "match_count_returned":returned_matching_lines,
            "group_count_returned":matches.len(),
            "total_matching_lines":total_matching_lines,
            "truncated":truncated,
            "matches":matches,
            "total_bytes":metadata.byte_count,
            "total_lines":metadata.line_count,
            "stream":metadata.stream,
            "tool":metadata.tool_name,
            "command":metadata.shell_command
        }))
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
impl ToolHandler for ContextResolveTool {
    fn name(&self) -> &str {
        tool_names::TOOL_CONTEXT_RESOLVE
    }
    fn description(&self) -> &str {
        "Mark an unresolved error context block as resolved."
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
            "resolve",
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
        let snapshot = require_runtime_snapshot(&context)?;
        let projection = Arc::new(snapshot.context_view.clone());
        let tree = Arc::new(snapshot.context_tree.clone());
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
            ensure!(
                projection
                    .blocks
                    .iter()
                    .find(|(id, _)| id.as_str() == block_id)
                    .is_some_and(|(id, _)| projection.is_addressable(id)),
                "cannot summarize non-addressable context block '{block_id}'"
            );
        }
        let source_node_id = optional_trimmed_string(&args, "source_node_id", MAX_ID_CHARS)?;
        if let Some(source_node_id) = &source_node_id {
            let source_node_ref = ContextNodeId::new(source_node_id.clone())?;
            let source_node = tree.node(&source_node_ref);
            ensure!(
                source_node.is_some(),
                "unknown context node '{source_node_id}'"
            );
            validate_node_source(
                snapshot.as_ref(),
                source_node.expect("validated context node"),
            )?;
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
            let span = SourceSpan::new(start, end)?;
            ensure!(
                !snapshot.overlaps_retired_source_span(span),
                "cannot summarize a range containing compacted context source"
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
    let target_id = crate::context_view::ContextBlockId::new(block_id.to_string())?;
    ensure!(
        !projection.is_compacted(&target_id),
        "cannot {operation} compacted context block '{block_id}'"
    );
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
        "resolve" => crate::context_view::ContextViewOperation::Resolve {
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
    Ok(Arc::new(
        require_runtime_snapshot(context)?.context_view.clone(),
    ))
}

fn require_context_tree(context: &ToolExecutionContext) -> Result<Arc<ContextTreeState>> {
    Ok(Arc::new(
        require_runtime_snapshot(context)?.context_tree.clone(),
    ))
}

fn require_runtime_snapshot(context: &ToolExecutionContext) -> Result<Arc<RuntimeSnapshot>> {
    context
        .runtime_snapshot
        .clone()
        .ok_or_else(|| anyhow!("runtime context snapshot unavailable"))
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

fn parse_nullable_usize(
    args: &Value,
    field: &str,
    default: usize,
    min: usize,
    max: usize,
) -> Result<usize> {
    let value = match args.get(field) {
        Some(Value::Null) | None => default,
        Some(value) => usize::try_from(
            value
                .as_u64()
                .ok_or_else(|| anyhow!("{field} must be integer or null"))?,
        )
        .map_err(|_| anyhow!("{field} is too large"))?,
    };
    ensure!(
        value >= min && value <= max,
        "{field} must be between {min} and {max}"
    );
    Ok(value)
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
    let Some(id) = projection.blocks.keys().find(|id| id.as_str() == block_id) else {
        return false;
    };
    if projection.is_compacted(id) || projection.is_resolved(id) {
        return false;
    }
    projection.is_provider_active_block(id, block)
        || (include_archived && projection.status_for(id) == ContextViewStatus::Archived)
        || (include_removed && projection.status_for(id) == ContextViewStatus::RemovedFromView)
}

fn node_visible_for_listing(
    projection: &ContextViewProjection,
    node: &ContextNodeRecord,
    include_archived: bool,
    include_removed: bool,
) -> bool {
    if node.status == ContextNodeStatus::Archived && !include_archived {
        return false;
    }
    node.block_ref.as_ref().is_none_or(|block_ref| {
        projection
            .blocks
            .iter()
            .find(|(id, _)| id.as_str() == block_ref.block_id)
            .is_some_and(|(id, block)| {
                block_visible_for_listing(
                    projection,
                    id.as_str(),
                    block,
                    include_archived,
                    include_removed,
                )
            })
    })
}

fn validate_node_source(snapshot: &RuntimeSnapshot, node: &ContextNodeRecord) -> Result<()> {
    let projection = &snapshot.context_view;
    if let Some(block_ref) = &node.block_ref {
        let addressable = projection
            .blocks
            .iter()
            .find(|(id, _)| id.as_str() == block_ref.block_id)
            .is_some_and(|(id, _)| projection.is_addressable(id));
        ensure!(
            addressable,
            "context node '{}' references non-addressable context",
            node.node_id.as_str()
        );
    }
    let Some(source_ref) = &node.source_ref else {
        return Ok(());
    };
    let source_id = source_ref
        .source_id
        .as_deref()
        .filter(|id| !id.trim().is_empty())
        .ok_or_else(|| {
            anyhow!(
                "context node '{}' has malformed source reference",
                node.node_id.as_str()
            )
        })?;
    match source_ref.source_kind.as_str() {
        "summary" => ensure!(
            projection.open_summary_artifact(source_id).is_some(),
            "context node '{}' references unavailable summary",
            node.node_id.as_str()
        ),
        "context_branch" => ensure!(
            source_id == snapshot.active_context.branch_id,
            "context node '{}' references inactive context branch",
            node.node_id.as_str()
        ),
        _ => bail!(
            "context node '{}' has unsupported source reference",
            node.node_id.as_str()
        ),
    }
    Ok(())
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
        ContextViewStatus::Resolved => "resolved".to_string(),
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
        "retention":context_block_retention_label(block.retention_class()),
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
    let id = projection
        .blocks
        .keys()
        .find(|id| id.as_str() == block_id)?;
    if !projection.is_addressable(id) {
        return None;
    }
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
        "retention":context_block_retention_label(block.retention_class()),
    });
    if status != "removed_from_view" {
        value["detail"] = json!(truncate(&block.detail, max_bytes));
    }
    Some(value)
}

fn ensure_block_openable(projection: &ContextViewProjection, block_id: &str) -> Result<String> {
    if projection
        .blocks
        .keys()
        .find(|id| id.as_str() == block_id)
        .is_some_and(|id| projection.is_compacted(id))
    {
        bail!("cannot open compacted context block '{block_id}'");
    }
    let status = block_status_string(projection, block_id);
    if status == "removed_from_view" {
        bail!("cannot open removed context block '{block_id}'");
    }
    ensure!(
        status != "resolved",
        "cannot open resolved context block '{block_id}'"
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

fn line_match_bounds(line: &str, query: &str, case_sensitive: bool) -> Option<(usize, usize)> {
    if case_sensitive {
        line.find(query).map(|start| (start, start + query.len()))
    } else {
        let lower = line.to_ascii_lowercase();
        lower.find(query).map(|start| (start, start + query.len()))
    }
}

fn grep_line_json(index: usize, text: &str, bounds: Option<(usize, usize)>) -> Value {
    let (display_text, text_truncated) = bounded_grep_line_text(text, bounds);
    let display_bounds = bounds.filter(|_| !text_truncated);
    json!({
        "line_number":index + 1,
        "text":display_text,
        "matched":bounds.is_some(),
        "match_start":display_bounds.map(|(value, _)| value),
        "match_end":display_bounds.map(|(_, value)| value),
        "text_truncated":text_truncated
    })
}

fn bounded_grep_line_text(text: &str, bounds: Option<(usize, usize)>) -> (String, bool) {
    let char_count = text.chars().count();
    if char_count <= MAX_GREP_LINE_CHARS {
        return (text.to_string(), false);
    }

    let Some((start, end)) = bounds else {
        return (truncate(text, MAX_GREP_LINE_CHARS), true);
    };

    let start_char = text[..start].chars().count();
    let end_char = text[..end].chars().count();
    let match_chars = end_char.saturating_sub(start_char).max(1);
    let window_chars = MAX_GREP_LINE_CHARS.saturating_sub(2).max(match_chars);
    let leading_chars = (window_chars.saturating_sub(match_chars)) / 2;
    let mut window_start = start_char.saturating_sub(leading_chars);
    if end_char > window_start.saturating_add(window_chars) {
        window_start = end_char.saturating_sub(window_chars);
    }
    let window_start = window_start.min(char_count);
    let window_end = window_start.saturating_add(window_chars).min(char_count);

    let mut display = String::new();
    if window_start > 0 {
        display.push('…');
    }
    display.extend(
        text.chars()
            .skip(window_start)
            .take(window_end - window_start),
    );
    if window_end < char_count {
        display.push('…');
    }
    (display, true)
}

fn build_grep_match_groups(
    lines: &[&str],
    matching_lines: &[usize],
    query: &str,
    case_sensitive: bool,
    context_lines: usize,
) -> Vec<Value> {
    let mut groups = Vec::new();
    let mut cursor = 0;

    while cursor < matching_lines.len() {
        let mut start = matching_lines[cursor].saturating_sub(context_lines);
        let mut end = matching_lines[cursor]
            .saturating_add(context_lines)
            .min(lines.len().saturating_sub(1));
        cursor += 1;

        while cursor < matching_lines.len() {
            let next_start = matching_lines[cursor].saturating_sub(context_lines);
            let next_end = matching_lines[cursor]
                .saturating_add(context_lines)
                .min(lines.len().saturating_sub(1));
            if next_start > end.saturating_add(1) {
                break;
            }
            start = start.min(next_start);
            end = end.max(next_end);
            cursor += 1;
        }

        let snippet_lines = (start..=end)
            .map(|index| {
                let text = lines[index];
                let bounds = line_match_bounds(text, query, case_sensitive);
                grep_line_json(index, text, bounds)
            })
            .collect::<Vec<_>>();
        let matched_line_numbers = snippet_lines
            .iter()
            .filter_map(|line| {
                line.get("matched")
                    .and_then(Value::as_bool)
                    .filter(|matched| *matched)
                    .and_then(|_| line.get("line_number"))
                    .and_then(Value::as_u64)
            })
            .collect::<Vec<_>>();
        groups.push(json!({
            "start_line_number":start + 1,
            "end_line_number":end + 1,
            "matched_line_numbers":matched_line_numbers,
            "lines":snippet_lines
        }));
    }

    groups
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
        ContextBlockKind::ReasoningNote => "reasoning_note",
    }
}

fn context_block_retention_label(retention: ContextBlockRetention) -> &'static str {
    match retention {
        ContextBlockRetention::Critical => "critical",
        ContextBlockRetention::Protected => "protected",
        ContextBlockRetention::Working => "working",
        ContextBlockRetention::Debug => "debug",
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
pub(crate) fn group_16_runtime_snapshot() -> RuntimeSnapshot {
    use crate::context_view::{
        ContextBlockId, ContextViewOperation, ContextViewState, FoldedOutputMetadata,
        SummaryArtifact,
    };
    use crate::protocol_frames::ProtocolFrameItem;
    use crate::request_builder::HistoryToolCall;
    use crate::runtime_context::{
        FrameVisibility, RuntimeFrame, RuntimeFrameIdSeed, RuntimeFrameKind,
        RuntimeFrameProvenance, RuntimeSource,
    };
    use crate::user_content::UserMessageContent;
    use std::collections::BTreeMap;

    fn block(
        id: &str,
        title: &str,
        detail: &str,
        sequence: u64,
        folded_output_id: Option<&str>,
    ) -> ContextBlock {
        ContextBlock {
            block_id: ContextBlockId::new(id).expect("valid fixture block id"),
            node_id: None,
            kind: ContextBlockKind::Note,
            title: title.into(),
            detail: detail.into(),
            source: ContextBlockSource::TranscriptSpan {
                start_sequence: sequence,
                end_sequence: sequence,
            },
            source_start_sequence: Some(sequence),
            available_sequence: Some(sequence),
            protected_reasons: Vec::new(),
            folded_output_id: folded_output_id.map(str::to_string),
        }
    }

    let mut blocks = BTreeMap::new();
    for block in [
        block(
            "active-block",
            "CANONICAL ACTIVE TITLE",
            "CANONICAL ACTIVE CONTENT CURRENT-TAIL-SENTINEL",
            20,
            None,
        ),
        block(
            "pinned-block",
            "PINNED ACTIVE TITLE",
            "PINNED ACTIVE CONTENT",
            21,
            None,
        ),
        block(
            "archived-block",
            "ARCHIVED TITLE",
            "ARCHIVED CONTENT",
            22,
            None,
        ),
        block(
            "removed-block",
            "REMOVED TITLE",
            "REMOVED SENTINEL",
            23,
            None,
        ),
        block(
            "retired-raw-block",
            "RETIRED RAW TITLE",
            "RETIRED-RAW-SENTINEL",
            10,
            None,
        ),
        block(
            "active-folded-block",
            "ACTIVE FOLDED TITLE",
            "ACTIVE FOLDED DETAIL",
            24,
            Some("active-folded-output"),
        ),
        block(
            "compacted-folded-block",
            "COMPACTED FOLDED TITLE",
            "COMPACTED FOLDED DETAIL",
            11,
            Some("compacted-folded-output"),
        ),
    ] {
        blocks.insert(block.block_id.clone(), block);
    }
    let id = |id| ContextBlockId::new(id).expect("fixture block id");
    let operations = vec![
        ContextViewOperation::Pin {
            block_id: id("pinned-block"),
        },
        ContextViewOperation::Archive {
            block_id: id("archived-block"),
        },
        ContextViewOperation::RemoveFromView {
            block_id: id("removed-block"),
        },
        ContextViewOperation::OpenDetail {
            block_id: id("active-block"),
        },
    ];
    let view_state =
        ContextViewState::replay(&blocks, &operations).expect("fixture view operations");
    let mut view = ContextViewProjection {
        blocks,
        view_state,
        summary_artifacts: vec![SummaryArtifact {
            artifact_id: "current-tail-summary".into(),
            node_id: "root".into(),
            artifact_kind: "summary".into(),
            version: 1,
            summary: "CURRENT-TAIL-SENTINEL".into(),
            source_node_id: None,
            source_block_id: Some("active-block".into()),
            source_start_sequence: Some(20),
            source_end_sequence: Some(20),
            created_sequence: 30,
        }],
        folded_outputs: BTreeMap::from([
            (
                "active-folded-output".into(),
                FoldedOutputMetadata {
                    output_id: "active-folded-output".into(),
                    node_id: None,
                    output_kind: "shell_output".into(),
                    call_id: Some("current-call".into()),
                    tool_name: Some("shell__exec".into()),
                    stream: Some("stdout".into()),
                    content: "ACTIVE-FOLDED-SENTINEL".into(),
                    byte_count: 22,
                    line_count: 1,
                    truncated: false,
                    shell_command: Some("cargo test".into()),
                    source_start_sequence: Some(24),
                    source_end_sequence: Some(24),
                    available_sequence: Some(24),
                    tool_ok: Some(true),
                    exit_status: Some(0),
                },
            ),
            (
                "compacted-folded-output".into(),
                FoldedOutputMetadata {
                    output_id: "compacted-folded-output".into(),
                    node_id: None,
                    output_kind: "shell_output".into(),
                    call_id: Some("retired-call".into()),
                    tool_name: Some("shell__exec".into()),
                    stream: Some("stdout".into()),
                    content: "RETIRED-FOLDED-SENTINEL".into(),
                    byte_count: 23,
                    line_count: 1,
                    truncated: false,
                    shell_command: Some("retired command".into()),
                    source_start_sequence: Some(11),
                    source_end_sequence: Some(11),
                    available_sequence: Some(11),
                    tool_ok: Some(true),
                    exit_status: Some(0),
                },
            ),
        ]),
        compacted_block_ids: Default::default(),
    };
    view.apply_retired_spans(&[SourceSpan::new(10, 11).expect("fixture retired span")]);

    let mut snapshot = RuntimeSnapshot::new("group-16")
        .with_session_id("group-16-session")
        .with_leaf_sequence(30);
    snapshot.set_context_view(view);
    snapshot.active_context.active_node_id = Some("root".into());
    snapshot.active_context.open_detail_block_id = Some("active-block".into());
    snapshot.active_context.visible_block_ids = snapshot.context_view.provider_visible_block_ids();
    snapshot.active_context.pinned_block_ids = snapshot.context_view.provider_pinned_block_ids();
    snapshot.push_folded_output(crate::runtime_context::FoldedOutputReference {
        output_id: "active-folded-output".into(),
        node_id: None,
        call_id: Some("current-call".into()),
        tool_name: Some("shell__exec".into()),
        source_span: Some(SourceSpan::new(24, 24).expect("fixture folded span")),
    });
    for (ordinal, kind, visibility, span, item) in [
        (
            0,
            RuntimeFrameKind::User,
            FrameVisibility::Retired,
            Some(SourceSpan::new(10, 10).expect("fixture span")),
            ProtocolFrameItem::UserMessage {
                content: UserMessageContent::from("RETIRED-RAW-SENTINEL"),
            },
        ),
        (
            1,
            RuntimeFrameKind::Summary,
            FrameVisibility::Active,
            Some(SourceSpan::new(30, 30).expect("fixture span")),
            ProtocolFrameItem::ContextSummary {
                text: "CURRENT-TAIL-SENTINEL".into(),
            },
        ),
        (
            2,
            RuntimeFrameKind::ToolCall,
            FrameVisibility::Active,
            Some(SourceSpan::new(24, 24).expect("fixture span")),
            ProtocolFrameItem::AssistantToolCalls {
                text: None,
                calls: vec![HistoryToolCall {
                    call_id: "current-call".into(),
                    name: "shell__exec".into(),
                    arguments_json: "{}".into(),
                }],
            },
        ),
        (
            3,
            RuntimeFrameKind::ToolOutput,
            FrameVisibility::Active,
            Some(SourceSpan::new(24, 24).expect("fixture span")),
            ProtocolFrameItem::ToolOutput {
                call_id: "current-call".into(),
                output_json:
                    r#"{"status":0,"body":"SURVIVING-PROTOCOL-SENTINEL ACTIVE-FOLDED-SENTINEL"}"#
                        .into(),
            },
        ),
        (
            4,
            RuntimeFrameKind::User,
            FrameVisibility::Active,
            Some(SourceSpan::new(25, 25).expect("fixture span")),
            ProtocolFrameItem::UserMessage {
                content: UserMessageContent::from("SURVIVING USER SENTINEL"),
            },
        ),
    ] {
        snapshot.push_frame(
            RuntimeFrame::new(
                kind,
                visibility,
                RuntimeFrameProvenance::new(RuntimeSource::Transcript)
                    .with_span(span.expect("span")),
                RuntimeFrameIdSeed {
                    frame_kind: kind,
                    source: RuntimeSource::Transcript,
                    ordinal,
                    stable_key: "group-16",
                    source_span: span,
                },
            )
            .with_protocol(item),
        );
    }
    snapshot
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

    fn compacted_tree_records() -> Vec<TranscriptRecord> {
        let mut records = base_records();
        records.extend(tree_records());
        records.push(record(
            9,
            TranscriptEvent::ContextCompaction(crate::agent::ContextCompactionEvent {
                outcome: "succeeded".into(),
                summary: "summary".into(),
                tail_start_index: 4,
                original_history_items: 4,
                retained_history_items: 1,
                retired_source_spans: vec![crate::agent::ContextCompactionSourceSpan {
                    start_sequence: 1,
                    end_sequence: 5,
                }],
                frame_identity_bindings: Vec::new(),
                detail: None,
            }),
        ));
        records
    }

    fn projection_with_compacted_tree_data() -> Arc<ContextViewProjection> {
        Arc::new(
            project_context_view(&compacted_tree_records())
                .expect("projection with compacted tree data"),
        )
    }

    fn compacted_tree() -> Arc<ContextTreeState> {
        Arc::new(project_context_tree(&compacted_tree_records()).expect("compacted context tree"))
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

    fn projection_with_compacted_folded_output() -> Arc<ContextViewProjection> {
        Arc::new(project_context_view(&[
            record(1, TranscriptEvent::ToolCallStarted { call_id: "call-1".into(), name: "shell__exec".into(), args: json!({"command":"cargo test"}) }),
            record(2, TranscriptEvent::ToolCallFinished { call_id: "call-1".into(), name: "shell__exec".into(), ok: true, output: crate::tool::ToolResult::ok("shell__exec", json!({"status":0,"stdout":"Needle in compacted output\nsecond line".repeat(400),"stdout_truncated":false,"stderr":"","stderr_truncated":false})) }),
            record(3, TranscriptEvent::ContextCompaction(crate::agent::ContextCompactionEvent { outcome: "succeeded".into(), summary: "summary".into(), tail_start_index: 2, original_history_items: 2, retained_history_items: 1, retired_source_spans: vec![crate::agent::ContextCompactionSourceSpan { start_sequence: 1, end_sequence: 2 }], frame_identity_bindings: Vec::new(), detail: None })),
        ]).expect("projection with compacted folded output"))
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

    fn projection_with_large_multiline_folded_output() -> Arc<ContextViewProjection> {
        let stdout = (1..=140)
            .map(|line| match line {
                4 => "CaseSensitiveNeedle".to_string(),
                138 => "before target".to_string(),
                139 => "Needle near end".to_string(),
                140 => "after target".to_string(),
                _ => format!("line {line}: {}", "x".repeat(40)),
            })
            .collect::<Vec<_>>()
            .join("\n");
        Arc::new(
            project_context_view(&[
                record(
                    1,
                    TranscriptEvent::ToolCallStarted {
                        call_id: "call-grep".into(),
                        name: "shell__exec".into(),
                        args: json!({"command":"long command"}),
                    },
                ),
                record(
                    2,
                    TranscriptEvent::ToolCallFinished {
                        call_id: "call-grep".into(),
                        name: "shell__exec".into(),
                        ok: true,
                        output: crate::tool::ToolResult::ok(
                            "shell__exec",
                            json!({
                                "status":0,
                                "stdout":stdout,
                                "stdout_truncated":false,
                                "stderr":"",
                                "stderr_truncated":false
                            }),
                        ),
                    },
                ),
            ])
            .expect("projection with large multiline folded output"),
        )
    }

    fn projection_with_folded_output_content(stdout: &str) -> Arc<ContextViewProjection> {
        Arc::new(
            project_context_view(&[
                record(
                    1,
                    TranscriptEvent::ToolCallStarted {
                        call_id: "call-grep".into(),
                        name: "shell__exec".into(),
                        args: json!({"command":"long command"}),
                    },
                ),
                record(
                    2,
                    TranscriptEvent::ToolCallFinished {
                        call_id: "call-grep".into(),
                        name: "shell__exec".into(),
                        ok: true,
                        output: crate::tool::ToolResult::ok(
                            "shell__exec",
                            json!({
                                "status":0,
                                "stdout":stdout,
                                "stdout_truncated":false,
                                "stderr":"",
                                "stderr_truncated":false
                            }),
                        ),
                    },
                ),
            ])
            .expect("projection with folded output content"),
        )
    }

    fn context(
        projection: Option<Arc<ContextViewProjection>>,
        tree: Option<Arc<ContextTreeState>>,
    ) -> ToolExecutionContext {
        let runtime_snapshot = projection.as_ref().map(|projection| {
            let mut snapshot = RuntimeSnapshot::new("test");
            snapshot.set_context_view((**projection).clone());
            if let Some(tree) = &tree {
                snapshot.set_context_tree((**tree).clone());
            }
            Arc::new(snapshot)
        });
        ToolExecutionContext {
            allow_outside_workspace: false,
            runtime_snapshot,
            context_view: projection,
            context_tree: tree,
            question_handler: None,
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
        assert!(output.error.as_ref().is_some_and(|error| {
            error
                .message
                .contains("runtime context snapshot unavailable")
        }));
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
    async fn group_16_context_tools_follow_the_canonical_snapshot() {
        let snapshot = Arc::new(group_16_runtime_snapshot());
        let registry = ToolRegistry::default_tools();
        let context = || ToolExecutionContext::with_runtime_snapshot(snapshot.clone());
        let list = registry
            .call_with_context(
                tool_names::TOOL_CONTEXT_LIST,
                json!({"include_archived":false,"include_removed":false,"limit":null}),
                context(),
            )
            .await;
        assert!(list.ok, "{list:?}");
        let listed_block_ids = list.data.as_ref().expect("list data")["blocks"]
            .as_array()
            .expect("blocks")
            .iter()
            .filter_map(|item| item["ref_id"].as_str())
            .collect::<Vec<_>>();
        let listed_folded_ids = list.data.as_ref().expect("list data")["folded_outputs"]
            .as_array()
            .expect("folded outputs")
            .iter()
            .filter_map(|item| item["ref_id"].as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            listed_block_ids,
            snapshot
                .context_view
                .provider_active_blocks()
                .iter()
                .map(|(id, _)| id.as_str())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            listed_folded_ids,
            snapshot
                .context_view
                .provider_folded_outputs()
                .iter()
                .map(|output| output.output_id.as_str())
                .collect::<Vec<_>>()
        );
        assert!(!listed_block_ids.contains(&"archived-block"));
        assert!(!listed_block_ids.contains(&"removed-block"));
        assert!(!listed_block_ids.contains(&"retired-raw-block"));
        assert_eq!(listed_folded_ids, vec!["active-folded-output"]);

        let archived = registry
            .call_with_context(
                tool_names::TOOL_CONTEXT_LIST,
                json!({"include_archived":true,"include_removed":false,"limit":null}),
                context(),
            )
            .await;
        let archived_data = archived.data.expect("archived list data");
        let archived_ids = archived_data["blocks"]
            .as_array()
            .expect("archived blocks")
            .iter()
            .filter_map(|item| item["ref_id"].as_str())
            .collect::<Vec<_>>();
        assert!(archived_ids.contains(&"archived-block"));
        assert!(!archived_ids.contains(&"removed-block"));
        assert!(!archived_ids.contains(&"retired-raw-block"));

        for (ref_type, ref_id, sentinel) in [
            ("block", "retired-raw-block", "RETIRED-RAW-SENTINEL"),
            (
                "folded_output",
                "compacted-folded-output",
                "RETIRED-FOLDED-SENTINEL",
            ),
            ("block", "removed-block", "REMOVED SENTINEL"),
        ] {
            let search = registry
                .call_with_context(
                    tool_names::TOOL_CONTEXT_SEARCH,
                    json!({"query":sentinel,"include_archived":true,"include_removed":true,"limit":null}),
                    context(),
                )
                .await;
            assert!(search.ok, "{search:?}");
            assert!(
                search.data.expect("search data")["matches"]
                    .as_array()
                    .expect("matches")
                    .iter()
                    .all(|item| item["ref_id"] != ref_id)
            );
            let open = registry
                .call_with_context(
                    tool_names::TOOL_CONTEXT_OPEN,
                    json!({"ref_type":ref_type,"ref_id":ref_id,"max_bytes":null}),
                    context(),
                )
                .await;
            assert!(!open.ok, "{open:?}");
        }
    }

    #[tokio::test]
    async fn context_resolve_marks_unresolved_errors_and_hides_them_by_default() {
        let registry = ToolRegistry::default_tools();
        let projection = Arc::new(
            project_context_view(&[record(
                1,
                TranscriptEvent::Error {
                    message: "context view projection unavailable".into(),
                },
            )])
            .expect("projection with unresolved error"),
        );

        let resolved = registry
            .call_with_context(
                tool_names::TOOL_CONTEXT_RESOLVE,
                json!({"block_id":"block-seq-1-error"}),
                context(Some(projection.clone()), None),
            )
            .await;
        assert!(resolved.ok, "{resolved:?}");
        assert_eq!(
            resolved
                .data
                .as_ref()
                .and_then(|d| d.get("operation_metadata"))
                .and_then(|d| d.get("operation")),
            Some(&json!("resolve"))
        );

        let resolved_projection = Arc::new(
            project_context_view(&[
                record(
                    1,
                    TranscriptEvent::Error {
                        message: "context view projection unavailable".into(),
                    },
                ),
                record(
                    2,
                    TranscriptEvent::ContextViewOperationMetadata {
                        operation: "resolve".into(),
                        node_id: None,
                        block_id: Some("block-seq-1-error".into()),
                        detail: None,
                    },
                ),
            ])
            .expect("resolved projection"),
        );
        let listed = registry
            .call_with_context(
                tool_names::TOOL_CONTEXT_LIST,
                json!({"include_archived":false,"include_removed":false,"limit":null}),
                context(Some(resolved_projection.clone()), Some(tree())),
            )
            .await;
        assert!(listed.ok, "{listed:?}");
        assert!(
            listed
                .data
                .as_ref()
                .and_then(|d| d.get("blocks"))
                .and_then(Value::as_array)
                .is_some_and(|blocks| blocks
                    .iter()
                    .all(|block| block["ref_id"] != "block-seq-1-error"))
        );

        let listed_archived = registry
            .call_with_context(
                tool_names::TOOL_CONTEXT_LIST,
                json!({"include_archived":true,"include_removed":false,"limit":null}),
                context(Some(resolved_projection), Some(tree())),
            )
            .await;
        assert!(listed_archived.ok, "{listed_archived:?}");
        assert!(
            listed_archived
                .data
                .as_ref()
                .and_then(|d| d.get("blocks"))
                .and_then(Value::as_array)
                .is_some_and(|blocks| blocks
                    .iter()
                    .all(|block| block["ref_id"] != "block-seq-1-error"))
        );
    }

    #[tokio::test]
    async fn context_resolve_rejects_non_error_blocks() {
        let registry = ToolRegistry::default_tools();
        let output = registry
            .call_with_context(
                tool_names::TOOL_CONTEXT_RESOLVE,
                json!({"block_id":"block-seq-2-note"}),
                context(Some(projection_with_tree_data()), None),
            )
            .await;

        assert!(!output.ok);
        assert!(output.error.as_ref().is_some_and(|error| {
            error
                .message
                .contains("cannot resolve context block 'block-seq-2-note' with kind note")
        }));
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
    async fn context_search_skips_compacted_folded_output() {
        let registry = ToolRegistry::default_tools();
        let projection = projection_with_compacted_folded_output();
        let output = registry
            .call_with_context(
                tool_names::TOOL_CONTEXT_SEARCH,
                json!({
                    "query":"Needle in compacted output",
                    "include_archived":true,
                    "include_removed":false,
                    "limit":null
                }),
                context(Some(projection), Some(tree())),
            )
            .await;

        assert!(output.ok, "{output:?}");
        assert_eq!(
            output
                .data
                .as_ref()
                .and_then(|d| d.get("matches"))
                .and_then(Value::as_array)
                .map(Vec::len),
            Some(0)
        );
    }

    #[tokio::test]
    async fn context_grep_rejects_compacted_folded_output() {
        let registry = ToolRegistry::default_tools();
        let projection = projection_with_compacted_folded_output();
        let output = registry
            .call_with_context(
                tool_names::TOOL_CONTEXT_GREP,
                json!({
                    "ref_id":"folded-output-seq-2-stdout",
                    "query":"Needle",
                    "case_sensitive":true,
                    "context_lines":0,
                    "max_matches":5
                }),
                context(Some(projection), None),
            )
            .await;

        assert!(!output.ok);
        assert!(output.error.as_ref().is_some_and(|error| {
            error
                .message
                .contains("cannot open compacted context block 'block-seq-2-folded-output-folded-output-seq-2-stdout'")
        }));
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

    #[tokio::test]
    async fn context_open_node_rejects_compacted_referenced_block() {
        let registry = ToolRegistry::default_tools();
        let projection = projection_with_compacted_tree_data();

        let node_output = registry
            .call_with_context(
                tool_names::TOOL_CONTEXT_OPEN,
                json!({
                    "ref_type":"node",
                    "ref_id":"node-a",
                    "max_bytes":32
                }),
                context(Some(projection.clone()), Some(compacted_tree())),
            )
            .await;

        assert!(!node_output.ok);
        assert!(node_output.error.as_ref().is_some_and(|error| {
            error
                .message
                .contains("context node 'node-a' references non-addressable context")
        }));

        let block_output = registry
            .call_with_context(
                tool_names::TOOL_CONTEXT_OPEN,
                json!({
                    "ref_type":"block",
                    "ref_id":"block-seq-2-note",
                    "max_bytes":32
                }),
                context(Some(projection), Some(compacted_tree())),
            )
            .await;

        assert!(!block_output.ok);
        assert!(block_output.error.as_ref().is_some_and(|error| {
            error
                .message
                .contains("cannot open compacted context block 'block-seq-2-note'")
        }));
    }

    #[test]
    fn node_source_validation_rejects_invalid_sources_and_allows_addressable_archived_nodes() {
        let projection = projection_with_tree_data();
        let mut snapshot = RuntimeSnapshot::new("main");
        snapshot.set_context_view((*projection).clone());
        let archived_summary_node = ContextNodeRecord {
            node_id: ContextNodeId::new("archived-summary").expect("node id"),
            parent_node_id: Some(ContextNodeId::root()),
            label: None,
            purpose: None,
            block_ref: Some(crate::context_tree::ContextBlockRef {
                block_id: "block-seq-2-note".into(),
            }),
            source_ref: Some(crate::context_tree::ContextSourceRef {
                source_kind: "summary".into(),
                source_id: Some("sum-node-a".into()),
            }),
            status: ContextNodeStatus::Archived,
        };
        validate_node_source(&snapshot, &archived_summary_node).expect("archived addressable node");

        for source_ref in [
            crate::context_tree::ContextSourceRef {
                source_kind: "context_branch".into(),
                source_id: Some("other".into()),
            },
            crate::context_tree::ContextSourceRef {
                source_kind: "summary".into(),
                source_id: Some("missing".into()),
            },
            crate::context_tree::ContextSourceRef {
                source_kind: "transcript".into(),
                source_id: Some("2".into()),
            },
            crate::context_tree::ContextSourceRef {
                source_kind: "summary".into(),
                source_id: None,
            },
        ] {
            let mut node = archived_summary_node.clone();
            node.source_ref = Some(source_ref);
            assert!(validate_node_source(&snapshot, &node).is_err());
        }

        let mut branch_node = archived_summary_node.clone();
        branch_node.source_ref = Some(crate::context_tree::ContextSourceRef {
            source_kind: "context_branch".into(),
            source_id: Some("main".into()),
        });
        validate_node_source(&snapshot, &branch_node).expect("active branch source");
    }

    #[test]
    fn archived_nodes_require_include_archived_even_when_removed_blocks_are_included() {
        let projection = projection_with_tree_data();
        let mut node = ContextNodeRecord {
            node_id: ContextNodeId::new("archived").expect("node id"),
            parent_node_id: Some(ContextNodeId::root()),
            label: None,
            purpose: None,
            block_ref: Some(crate::context_tree::ContextBlockRef {
                block_id: "block-seq-2-note".into(),
            }),
            source_ref: None,
            status: ContextNodeStatus::Archived,
        };
        assert!(!node_visible_for_listing(&projection, &node, false, true));
        assert!(node_visible_for_listing(&projection, &node, true, false));

        node.status = ContextNodeStatus::Inactive;
        assert!(node_visible_for_listing(&projection, &node, false, false));
    }

    #[tokio::test]
    async fn context_grep_returns_bounded_snippets_near_end_of_folded_output() {
        let registry = ToolRegistry::default_tools();
        let projection = projection_with_large_multiline_folded_output();
        let output = registry
            .call_with_context(
                tool_names::TOOL_CONTEXT_GREP,
                json!({
                    "ref_id":"folded-output-seq-2-stdout",
                    "query":"Needle near end",
                    "case_sensitive":true,
                    "context_lines":1,
                    "max_matches":5
                }),
                context(Some(projection), None),
            )
            .await;

        assert!(output.ok, "{output:?}");
        let data = output.data.expect("grep data");
        assert_eq!(data.get("total_matching_lines"), Some(&json!(1)));
        assert_eq!(data.get("match_count_returned"), Some(&json!(1)));
        assert_eq!(data.get("truncated"), Some(&json!(false)));

        let matches = data
            .get("matches")
            .and_then(Value::as_array)
            .expect("matches");
        let lines = matches[0]
            .get("lines")
            .and_then(Value::as_array)
            .expect("snippet lines");
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[0].get("line_number"), Some(&json!(138)));
        assert_eq!(lines[1].get("line_number"), Some(&json!(139)));
        assert_eq!(lines[1].get("matched"), Some(&json!(true)));
        assert_eq!(lines[1].get("text"), Some(&json!("Needle near end")));
        assert_eq!(lines[2].get("line_number"), Some(&json!(140)));
        assert!(lines.iter().all(|line| {
            line.get("text")
                .and_then(Value::as_str)
                .is_some_and(|text| !text.contains("line 1:"))
        }));
    }

    #[tokio::test]
    async fn context_grep_respects_case_sensitivity_and_context_lines() {
        let registry = ToolRegistry::default_tools();
        let projection = projection_with_large_multiline_folded_output();
        let insensitive = registry
            .call_with_context(
                tool_names::TOOL_CONTEXT_GREP,
                json!({
                    "ref_id":"folded-output-seq-2-stdout",
                    "query":"casesensitiveneedle",
                    "case_sensitive":false,
                    "context_lines":0,
                    "max_matches":5
                }),
                context(Some(projection.clone()), None),
            )
            .await;
        assert!(insensitive.ok, "{insensitive:?}");
        assert_eq!(
            insensitive
                .data
                .as_ref()
                .and_then(|data| data.get("total_matching_lines")),
            Some(&json!(1))
        );
        assert_eq!(
            insensitive
                .data
                .as_ref()
                .and_then(|data| data.get("matches"))
                .and_then(Value::as_array)
                .and_then(|items| items.first())
                .and_then(|item| item.get("lines"))
                .and_then(Value::as_array)
                .map(Vec::len),
            Some(1)
        );

        let sensitive = registry
            .call_with_context(
                tool_names::TOOL_CONTEXT_GREP,
                json!({
                    "ref_id":"folded-output-seq-2-stdout",
                    "query":"casesensitiveneedle",
                    "case_sensitive":true,
                    "context_lines":0,
                    "max_matches":5
                }),
                context(Some(projection), None),
            )
            .await;
        assert!(sensitive.ok, "{sensitive:?}");
        assert_eq!(
            sensitive
                .data
                .as_ref()
                .and_then(|data| data.get("total_matching_lines")),
            Some(&json!(0))
        );
        assert_eq!(
            sensitive
                .data
                .as_ref()
                .and_then(|data| data.get("matches"))
                .and_then(Value::as_array)
                .map(Vec::len),
            Some(0)
        );
    }

    #[tokio::test]
    async fn context_grep_limits_matching_lines_and_truncates_long_lines() {
        let registry = ToolRegistry::default_tools();
        let long_prefix = "x".repeat(MAX_GREP_LINE_CHARS + 128);
        let long_suffix = "y".repeat(MAX_GREP_LINE_CHARS + 128);
        let content =
            format!("{long_prefix}first-needle{long_suffix}\nsecond needle\nthird needle");
        let projection = projection_with_folded_output_content(&content);
        let output = registry
            .call_with_context(
                tool_names::TOOL_CONTEXT_GREP,
                json!({
                    "ref_id":"folded-output-seq-2-stdout",
                    "query":"needle",
                    "case_sensitive":true,
                    "context_lines":0,
                    "max_matches":2
                }),
                context(Some(projection), None),
            )
            .await;

        assert!(output.ok, "{output:?}");
        let data = output.data.expect("grep data");
        assert_eq!(data.get("total_matching_lines"), Some(&json!(3)));
        assert_eq!(data.get("match_count_returned"), Some(&json!(2)));
        assert_eq!(data.get("truncated"), Some(&json!(true)));

        let groups = data
            .get("matches")
            .and_then(Value::as_array)
            .expect("matches");
        let returned_lines = groups
            .iter()
            .flat_map(|group| {
                group
                    .get("lines")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
            })
            .collect::<Vec<_>>();
        assert_eq!(returned_lines.len(), 2);
        assert_eq!(returned_lines[0].get("text_truncated"), Some(&json!(true)));
        assert!(
            returned_lines[0]
                .get("text")
                .and_then(Value::as_str)
                .is_some_and(|text| text.contains("first-needle") && text.len() < content.len())
        );
        assert_eq!(returned_lines[1].get("text"), Some(&json!("second needle")));
    }
}
