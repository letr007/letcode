use anyhow::{Result, anyhow, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use crate::tool_names;
use crate::transcript::{TranscriptEvent, TranscriptRecord};

pub(crate) const DEFAULT_FOLDED_OUTPUT_THRESHOLD_BYTES: usize = 4 * 1024;
pub(crate) const DEFAULT_OPEN_CONTENT_MAX_BYTES: usize = 2 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub(crate) struct ContextBlockId(String);

impl ContextBlockId {
    pub(crate) fn new(value: impl Into<String>) -> Result<Self> {
        let value = value.into();
        let trimmed = value.trim();
        ensure!(!trimmed.is_empty(), "context block_id must not be empty");
        Ok(Self(trimmed.to_string()))
    }

    pub(crate) fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ContextBlockKind {
    HardConstraint,
    CurrentUserRequirement,
    UnresolvedError,
    Permission,
    FileWriteFact,
    TestResult,
    CommitHash,
    ToolOutput,
    Note,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ProtectedReason {
    HardConstraint,
    CurrentUserRequirement,
    UnresolvedError,
    Permission,
    FileWriteFact,
    TestResult,
    CommitHash,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum ContextBlockSource {
    TranscriptSpan {
        start_sequence: u64,
        end_sequence: u64,
    },
    SummaryArtifact {
        artifact_id: String,
    },
    FoldedOutput {
        output_id: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct ContextBlock {
    pub block_id: ContextBlockId,
    pub kind: ContextBlockKind,
    pub title: String,
    pub detail: String,
    pub source: ContextBlockSource,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_start_sequence: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub available_sequence: Option<u64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub protected_reasons: Vec<ProtectedReason>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub folded_output_id: Option<String>,
}

impl ContextBlock {
    pub(crate) fn is_protected(&self) -> bool {
        !self.protected_reasons.is_empty()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ContextViewStatus {
    Visible,
    Pinned,
    Archived,
    RemovedFromView,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct ContextViewState {
    block_statuses: BTreeMap<ContextBlockId, ContextViewStatus>,
    open_detail_block_id: Option<ContextBlockId>,
}

impl Default for ContextViewState {
    fn default() -> Self {
        Self {
            block_statuses: BTreeMap::new(),
            open_detail_block_id: None,
        }
    }
}

impl ContextViewState {
    pub(crate) fn replay(
        blocks: &BTreeMap<ContextBlockId, ContextBlock>,
        operations: &[ContextViewOperation],
    ) -> std::result::Result<Self, ContextViewError> {
        let mut state = Self::default();
        for block_id in blocks.keys() {
            state
                .block_statuses
                .insert(block_id.clone(), ContextViewStatus::Visible);
        }
        for operation in operations {
            state.apply(blocks, operation)?;
        }
        Ok(state)
    }

    pub(crate) fn apply(
        &mut self,
        blocks: &BTreeMap<ContextBlockId, ContextBlock>,
        operation: &ContextViewOperation,
    ) -> std::result::Result<(), ContextViewError> {
        let block = blocks
            .get(operation.block_id())
            .ok_or_else(|| ContextViewError::UnknownBlock(operation.block_id().as_str().to_string()))?;
        let current = self
            .block_statuses
            .get(operation.block_id())
            .copied()
            .unwrap_or(ContextViewStatus::Visible);

        match operation {
            ContextViewOperation::Pin { block_id } => {
                self.block_statuses
                    .insert(block_id.clone(), ContextViewStatus::Pinned);
            }
            ContextViewOperation::Archive { block_id } => {
                ensure_block_mutable(block, "archive")?;
                self.block_statuses
                    .insert(block_id.clone(), ContextViewStatus::Archived);
                if self.open_detail_block_id.as_ref() == Some(block_id) {
                    self.open_detail_block_id = None;
                }
            }
            ContextViewOperation::RemoveFromView { block_id } => {
                ensure_block_mutable(block, "remove_from_view")?;
                self.block_statuses
                    .insert(block_id.clone(), ContextViewStatus::RemovedFromView);
                if self.open_detail_block_id.as_ref() == Some(block_id) {
                    self.open_detail_block_id = None;
                }
            }
            ContextViewOperation::OpenDetail { block_id } => {
                if current == ContextViewStatus::RemovedFromView {
                    return Err(ContextViewError::RemovedBlockCannotOpen(block_id.as_str().into()));
                }
                self.open_detail_block_id = Some(block_id.clone());
            }
        }

        Ok(())
    }

    pub(crate) fn status(&self, block_id: &ContextBlockId) -> Option<ContextViewStatus> {
        self.block_statuses.get(block_id).copied()
    }

    pub(crate) fn open_detail_block_id(&self) -> Option<&ContextBlockId> {
        self.open_detail_block_id.as_ref()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ContextViewOperation {
    Pin { block_id: ContextBlockId },
    Archive { block_id: ContextBlockId },
    RemoveFromView { block_id: ContextBlockId },
    OpenDetail { block_id: ContextBlockId },
}

impl ContextViewOperation {
    pub(crate) fn block_id(&self) -> &ContextBlockId {
        match self {
            Self::Pin { block_id }
            | Self::Archive { block_id }
            | Self::RemoveFromView { block_id }
            | Self::OpenDetail { block_id } => block_id,
        }
    }

    pub(crate) fn parse(operation: &str, block_id: ContextBlockId) -> Result<Self> {
        match operation.trim() {
            "pin" => Ok(Self::Pin { block_id }),
            "archive" => Ok(Self::Archive { block_id }),
            "remove_from_view" => Ok(Self::RemoveFromView { block_id }),
            "open_detail" => Ok(Self::OpenDetail { block_id }),
            other => Err(anyhow!("unknown context view operation '{other}'")),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ContextViewError {
    UnknownBlock(String),
    ProtectedBlockMutation {
        block_id: String,
        operation: &'static str,
        reason: ProtectedReason,
    },
    RemovedBlockCannotOpen(String),
    OperationTargetsFutureBlock {
        block_id: String,
        operation_sequence: u64,
        block_sequence: u64,
    },
}

impl fmt::Display for ContextViewError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownBlock(block_id) => write!(f, "unknown context block '{block_id}'"),
            Self::ProtectedBlockMutation {
                block_id,
                operation,
                reason,
            } => write!(
                f,
                "cannot {operation} protected context block '{block_id}' ({})",
                protected_reason_label(*reason)
            ),
            Self::RemovedBlockCannotOpen(block_id) => {
                write!(f, "cannot open removed context block '{block_id}'")
            }
            Self::OperationTargetsFutureBlock {
                block_id,
                operation_sequence,
                block_sequence,
            } => write!(
                f,
                "context view operation at sequence {operation_sequence} targets future block '{block_id}' created at sequence {block_sequence}"
            ),
        }
    }
}

impl std::error::Error for ContextViewError {}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct SummaryArtifact {
    pub artifact_id: String,
    pub node_id: String,
    pub artifact_kind: String,
    pub version: u32,
    pub summary: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_node_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_block_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_start_sequence: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_end_sequence: Option<u64>,
    pub created_sequence: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct FoldedOutputMetadata {
    pub output_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub node_id: Option<String>,
    pub output_kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<String>,
    pub content: String,
    pub byte_count: usize,
    pub line_count: usize,
    #[serde(default)]
    pub truncated: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shell_command: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_start_sequence: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_end_sequence: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub available_sequence: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_ok: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exit_status: Option<i32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BoundedOpenResult {
    pub content: String,
    pub returned_bytes: usize,
    pub total_bytes: usize,
    pub truncated: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct ContextViewProjection {
    pub blocks: BTreeMap<ContextBlockId, ContextBlock>,
    pub view_state: ContextViewState,
    pub summary_artifacts: Vec<SummaryArtifact>,
    pub folded_outputs: BTreeMap<String, FoldedOutputMetadata>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RecordedContextViewOperation {
    pub sequence: u64,
    pub operation: ContextViewOperation,
}

impl ContextViewProjection {
    pub(crate) fn open_summary_artifact(&self, artifact_id: &str) -> Option<&SummaryArtifact> {
        self.summary_artifacts
            .iter()
            .find(|artifact| artifact.artifact_id == artifact_id)
    }

    pub(crate) fn list_summary_artifacts_for_node(&self, node_id: &str) -> Vec<&SummaryArtifact> {
        self.summary_artifacts
            .iter()
            .filter(|artifact| artifact.node_id == node_id)
            .collect()
    }

    pub(crate) fn open_folded_output(
        &self,
        output_id: &str,
        max_bytes: usize,
    ) -> Option<BoundedOpenResult> {
        self.folded_outputs
            .get(output_id)
            .map(|metadata| open_folded_output(metadata, max_bytes))
    }
}

pub(crate) fn project_context_view(records: &[TranscriptRecord]) -> Result<ContextViewProjection> {
    let folded_outputs = restore_folded_outputs(records, DEFAULT_FOLDED_OUTPUT_THRESHOLD_BYTES)?;
    let blocks = index_context_blocks(records, &folded_outputs);
    let operations = restore_context_view_operations(records)?;
    let view_state = replay_recorded_context_view_state(&blocks, &operations)
        .map_err(|error| anyhow!(error.to_string()))?;
    let summary_artifacts = restore_summary_artifacts(records)?;
    Ok(ContextViewProjection {
        blocks,
        view_state,
        summary_artifacts,
        folded_outputs,
    })
}

pub(crate) fn index_context_blocks(
    records: &[TranscriptRecord],
    folded_outputs: &BTreeMap<String, FoldedOutputMetadata>,
) -> BTreeMap<ContextBlockId, ContextBlock> {
    let mut blocks = BTreeMap::new();
    let folded_by_sequence = folded_outputs
        .values()
        .filter_map(|metadata| metadata.source_start_sequence.map(|sequence| (sequence, metadata)))
        .fold(BTreeMap::<u64, Vec<&FoldedOutputMetadata>>::new(), |mut acc, (sequence, metadata)| {
            acc.entry(sequence).or_default().push(metadata);
            acc
        });

    for record in records {
        match &record.event {
            TranscriptEvent::UserMessage { content } => {
                let text = content.display_text();
                if !text.trim().is_empty() {
                    insert_block(
                        &mut blocks,
                        block_id(record.sequence, "user-requirement"),
                        ContextBlockKind::CurrentUserRequirement,
                        "User requirement".into(),
                        text.clone(),
                        transcript_source(record.sequence),
                        Some(record.sequence),
                        vec![ProtectedReason::CurrentUserRequirement],
                        None,
                    );
                    if is_hard_constraint_text(&text) {
                        insert_block(
                            &mut blocks,
                            block_id(record.sequence, "hard-constraint"),
                            ContextBlockKind::HardConstraint,
                            "Hard constraint".into(),
                            text,
                            transcript_source(record.sequence),
                            Some(record.sequence),
                            vec![ProtectedReason::HardConstraint],
                            None,
                        );
                    }
                }
            }
            TranscriptEvent::AssistantMessage { content }
            | TranscriptEvent::ReasoningMessage { content } => {
                if !content.trim().is_empty() {
                    insert_block(
                        &mut blocks,
                        block_id(record.sequence, "note"),
                        ContextBlockKind::Note,
                        "Session note".into(),
                        content.clone(),
                        transcript_source(record.sequence),
                        Some(record.sequence),
                        Vec::new(),
                        None,
                    );
                    for (index, hash) in extract_commit_hashes(content).into_iter().enumerate() {
                        insert_block(
                            &mut blocks,
                            block_id(record.sequence, &format!("commit-{index}")),
                            ContextBlockKind::CommitHash,
                            "Commit hash".into(),
                            hash,
                            transcript_source(record.sequence),
                            Some(record.sequence),
                            vec![ProtectedReason::CommitHash],
                            None,
                        );
                    }
                }
            }
            TranscriptEvent::Error { message } => {
                insert_block(
                    &mut blocks,
                    block_id(record.sequence, "error"),
                    ContextBlockKind::UnresolvedError,
                    "Unresolved error".into(),
                    message.clone(),
                    transcript_source(record.sequence),
                    Some(record.sequence),
                    vec![ProtectedReason::UnresolvedError],
                    None,
                );
            }
            TranscriptEvent::PermissionDecision {
                tool,
                allowed,
                reason,
                ..
            } => {
                let detail = if *allowed {
                    format!("Permission allowed for {tool}")
                } else {
                    format!(
                        "Permission denied for {tool}{}",
                        reason
                            .as_deref()
                            .map(|text| format!(": {text}"))
                            .unwrap_or_default()
                    )
                };
                insert_block(
                    &mut blocks,
                    block_id(record.sequence, "permission"),
                    ContextBlockKind::Permission,
                    "Permission decision".into(),
                    detail,
                    transcript_source(record.sequence),
                    Some(record.sequence),
                    vec![ProtectedReason::Permission],
                    None,
                );
            }
            TranscriptEvent::ValidationAdvisory(advisory) => {
                insert_block(
                    &mut blocks,
                    block_id(record.sequence, "validation"),
                    ContextBlockKind::TestResult,
                    "Validation result".into(),
                    advisory.message.clone(),
                    transcript_source(record.sequence),
                    Some(record.sequence),
                    vec![ProtectedReason::TestResult],
                    None,
                );
            }
            TranscriptEvent::ToolExecutionSummary(event) => {
                match event.effect_kind.as_str() {
                    "write" => insert_block(
                        &mut blocks,
                        block_id(record.sequence, "write"),
                        ContextBlockKind::FileWriteFact,
                        "File write fact".into(),
                        event
                            .primary_path
                            .clone()
                            .unwrap_or_else(|| event.name.clone()),
                        transcript_source(record.sequence),
                        Some(record.sequence),
                        vec![ProtectedReason::FileWriteFact],
                        None,
                    ),
                    "validation" => insert_block(
                        &mut blocks,
                        block_id(record.sequence, "test"),
                        ContextBlockKind::TestResult,
                        "Test result".into(),
                        event
                            .command
                            .clone()
                            .unwrap_or_else(|| event.name.clone()),
                        transcript_source(record.sequence),
                        Some(record.sequence),
                        vec![ProtectedReason::TestResult],
                        None,
                    ),
                    _ => {}
                }
            }
            TranscriptEvent::ToolCallFinished {
                name,
                ok,
                output,
                ..
            } => {
                if !ok {
                    let detail = output
                        .error
                        .as_ref()
                        .map(|error| error.message.clone())
                        .unwrap_or_else(|| format!("{name} failed"));
                    insert_block(
                        &mut blocks,
                        block_id(record.sequence, "tool-error"),
                        ContextBlockKind::UnresolvedError,
                        "Tool error".into(),
                        detail,
                        transcript_source(record.sequence),
                        Some(record.sequence),
                        vec![ProtectedReason::UnresolvedError],
                        None,
                    );
                }

                if let Some(metadata_items) = folded_by_sequence.get(&record.sequence) {
                    for metadata in metadata_items {
                        insert_block_with_availability(
                            &mut blocks,
                            folded_block_id(record.sequence, &metadata.output_id),
                            ContextBlockKind::ToolOutput,
                            folded_title(metadata),
                            folded_detail(metadata),
                            ContextBlockSource::FoldedOutput {
                                output_id: metadata.output_id.clone(),
                            },
                            metadata.source_start_sequence.or(Some(record.sequence)),
                            metadata.available_sequence.or(Some(record.sequence)),
                            Vec::new(),
                            Some(metadata.output_id.clone()),
                        );
                    }
                }

                if let Some(data) = &output.data {
                    for (index, hash) in extract_commit_hashes(&value_text(data)).into_iter().enumerate()
                    {
                        insert_block(
                            &mut blocks,
                            block_id(record.sequence, &format!("commit-{index}")),
                            ContextBlockKind::CommitHash,
                            "Commit hash".into(),
                            hash,
                            transcript_source(record.sequence),
                            Some(record.sequence),
                            vec![ProtectedReason::CommitHash],
                            None,
                        );
                    }
                }
            }
            _ => {}
        }
    }

    blocks
}

pub(crate) fn restore_context_view_operations(
    records: &[TranscriptRecord],
) -> Result<Vec<RecordedContextViewOperation>> {
    let mut operations = Vec::new();
    for record in records {
        let TranscriptEvent::ContextViewOperationMetadata {
            operation,
            block_id,
            ..
        } = &record.event
        else {
            continue;
        };
        let block_id = block_id.as_ref().ok_or_else(|| {
            anyhow!(
                "context view operation at sequence {} is missing block_id",
                record.sequence
            )
        })?;
        operations.push(RecordedContextViewOperation {
            sequence: record.sequence,
            operation: ContextViewOperation::parse(operation, ContextBlockId::new(block_id.clone())?)?,
        });
    }
    Ok(operations)
}

pub(crate) fn replay_recorded_context_view_state(
    blocks: &BTreeMap<ContextBlockId, ContextBlock>,
    operations: &[RecordedContextViewOperation],
) -> std::result::Result<ContextViewState, ContextViewError> {
    let mut state = ContextViewState::default();
    for block_id in blocks.keys() {
        state
            .block_statuses
            .insert(block_id.clone(), ContextViewStatus::Visible);
    }
    for recorded in operations {
        let block = blocks.get(recorded.operation.block_id()).ok_or_else(|| {
            ContextViewError::UnknownBlock(recorded.operation.block_id().as_str().to_string())
        })?;
        let block_sequence = block_available_sequence(block);
        if block_sequence > recorded.sequence {
            return Err(ContextViewError::OperationTargetsFutureBlock {
                block_id: recorded.operation.block_id().as_str().to_string(),
                operation_sequence: recorded.sequence,
                block_sequence,
            });
        }
        state.apply(blocks, &recorded.operation)?;
    }
    Ok(state)
}

pub(crate) fn restore_summary_artifacts(records: &[TranscriptRecord]) -> Result<Vec<SummaryArtifact>> {
    let mut artifacts = Vec::new();
    let mut seen_ids = BTreeSet::new();
    for record in records {
        let TranscriptEvent::ContextSummaryArtifactMetadata {
            node_id,
            artifact_id,
            artifact_kind,
            version,
            summary,
            source_node_id,
            source_block_id,
            source_start_sequence,
            source_end_sequence,
        } = &record.event
        else {
            continue;
        };
        ensure!(
            seen_ids.insert(artifact_id.clone()),
            "duplicate context summary artifact_id '{}'",
            artifact_id
        );
        artifacts.push(SummaryArtifact {
            artifact_id: artifact_id.clone(),
            node_id: node_id.clone(),
            artifact_kind: artifact_kind.clone(),
            version: version.unwrap_or(1),
            summary: summary.clone().unwrap_or_default(),
            source_node_id: source_node_id.clone(),
            source_block_id: source_block_id.clone(),
            source_start_sequence: *source_start_sequence,
            source_end_sequence: *source_end_sequence,
            created_sequence: record.sequence,
        });
    }
    artifacts.sort_by(|left, right| {
        left.node_id
            .cmp(&right.node_id)
            .then_with(|| left.version.cmp(&right.version))
            .then_with(|| left.created_sequence.cmp(&right.created_sequence))
    });
    Ok(artifacts)
}

pub(crate) fn restore_folded_outputs(
    records: &[TranscriptRecord],
    threshold_bytes: usize,
) -> Result<BTreeMap<String, FoldedOutputMetadata>> {
    let mut outputs = BTreeMap::new();
    let mut started_calls = BTreeMap::<String, Value>::new();

    for record in records {
        match &record.event {
            TranscriptEvent::ToolCallStarted { call_id, args, .. } => {
                started_calls.insert(call_id.clone(), args.clone());
            }
            TranscriptEvent::FoldedOutputMetadata {
                node_id,
                output_id,
                output_kind,
                call_id,
                tool_name,
                stream,
                content,
                byte_count,
                line_count,
                truncated,
                shell_command,
                source_start_sequence,
                source_end_sequence,
                tool_ok,
                exit_status,
            } => {
                ensure!(
                    !outputs.contains_key(output_id),
                    "duplicate folded output_id '{}'",
                    output_id
                );
                let content = content.clone().unwrap_or_default();
                outputs.insert(
                    output_id.clone(),
                    FoldedOutputMetadata {
                        output_id: output_id.clone(),
                        node_id: node_id.clone(),
                        output_kind: output_kind.clone(),
                        call_id: call_id.clone(),
                        tool_name: tool_name.clone(),
                        stream: stream.clone(),
                        byte_count: byte_count.unwrap_or(content.len()),
                        line_count: line_count.unwrap_or(count_lines(&content)),
                        truncated: truncated.unwrap_or(false),
                        shell_command: shell_command.clone(),
                        source_start_sequence: *source_start_sequence,
                        source_end_sequence: *source_end_sequence,
                        available_sequence: Some(record.sequence),
                        tool_ok: *tool_ok,
                        exit_status: *exit_status,
                        content,
                    },
                );
            }
            TranscriptEvent::ToolCallFinished {
                call_id,
                name,
                ok,
                output,
                ..
            } => {
                let Some(data) = output.data.as_ref() else {
                    continue;
                };
                let args = started_calls.get(call_id);
                let shell_command = args
                    .and_then(|value| value.get("command"))
                    .and_then(Value::as_str)
                    .map(str::to_string);
                let exit_status = data
                    .get("status")
                    .and_then(Value::as_i64)
                    .and_then(|value| i32::try_from(value).ok());
                for (stream, content, truncated_flag) in extract_output_streams(data) {
                    let is_shell = is_shell_tool_name(name);
                    let should_fold = if is_shell {
                        content.len() > threshold_bytes || truncated_flag
                    } else {
                        content.len() > threshold_bytes
                    };
                    if !should_fold {
                        continue;
                    }
                    let output_id = format!("folded-output-seq-{}-{stream}", record.sequence);
                    outputs.entry(output_id.clone()).or_insert_with(|| FoldedOutputMetadata {
                        output_id,
                        node_id: None,
                        output_kind: if is_shell {
                            "shell_output".into()
                        } else {
                            "tool_output".into()
                        },
                        call_id: Some(call_id.clone()),
                        tool_name: Some(name.clone()),
                        stream: Some(stream),
                        byte_count: content.len(),
                        line_count: count_lines(&content),
                        truncated: truncated_flag,
                        shell_command: shell_command.clone(),
                        source_start_sequence: Some(record.sequence),
                        source_end_sequence: Some(record.sequence),
                        available_sequence: Some(record.sequence),
                        tool_ok: Some(*ok),
                        exit_status,
                        content,
                    });
                }
            }
            _ => {}
        }
    }

    Ok(outputs)
}

pub(crate) fn open_folded_output(
    metadata: &FoldedOutputMetadata,
    max_bytes: usize,
) -> BoundedOpenResult {
    let (content, returned_bytes, truncated) = truncate_to_bytes(&metadata.content, max_bytes);
    BoundedOpenResult {
        content,
        returned_bytes,
        total_bytes: metadata.byte_count,
        truncated: truncated || metadata.byte_count > returned_bytes,
    }
}

fn block_available_sequence(block: &ContextBlock) -> u64 {
    block.available_sequence.or(block.source_start_sequence).unwrap_or(0)
}

fn ensure_block_mutable(
    block: &ContextBlock,
    operation: &'static str,
) -> std::result::Result<(), ContextViewError> {
    if let Some(reason) = block.protected_reasons.first().copied() {
        return Err(ContextViewError::ProtectedBlockMutation {
            block_id: block.block_id.as_str().to_string(),
            operation,
            reason,
        });
    }
    Ok(())
}

fn protected_reason_label(reason: ProtectedReason) -> &'static str {
    match reason {
        ProtectedReason::HardConstraint => "hard_constraint",
        ProtectedReason::CurrentUserRequirement => "current_user_requirement",
        ProtectedReason::UnresolvedError => "unresolved_error",
        ProtectedReason::Permission => "permission",
        ProtectedReason::FileWriteFact => "file_write_fact",
        ProtectedReason::TestResult => "test_result",
        ProtectedReason::CommitHash => "commit_hash",
    }
}

fn insert_block(
    blocks: &mut BTreeMap<ContextBlockId, ContextBlock>,
    block_id: ContextBlockId,
    kind: ContextBlockKind,
    title: String,
    detail: String,
    source: ContextBlockSource,
    source_start_sequence: Option<u64>,
    protected_reasons: Vec<ProtectedReason>,
    folded_output_id: Option<String>,
) {
    insert_block_with_availability(
        blocks,
        block_id,
        kind,
        title,
        detail,
        source,
        source_start_sequence,
        source_start_sequence,
        protected_reasons,
        folded_output_id,
    );
}

fn insert_block_with_availability(
    blocks: &mut BTreeMap<ContextBlockId, ContextBlock>,
    block_id: ContextBlockId,
    kind: ContextBlockKind,
    title: String,
    detail: String,
    source: ContextBlockSource,
    source_start_sequence: Option<u64>,
    available_sequence: Option<u64>,
    protected_reasons: Vec<ProtectedReason>,
    folded_output_id: Option<String>,
) {
    blocks.insert(
        block_id.clone(),
        ContextBlock {
            block_id,
            kind,
            title,
            detail,
            source,
            source_start_sequence,
            available_sequence,
            protected_reasons,
            folded_output_id,
        },
    );
}

fn transcript_source(sequence: u64) -> ContextBlockSource {
    ContextBlockSource::TranscriptSpan {
        start_sequence: sequence,
        end_sequence: sequence,
    }
}

fn block_id(sequence: u64, suffix: &str) -> ContextBlockId {
    ContextBlockId(format!("block-seq-{sequence}-{suffix}"))
}

fn folded_block_id(sequence: u64, output_id: &str) -> ContextBlockId {
    ContextBlockId(format!("block-seq-{sequence}-folded-output-{output_id}"))
}

fn is_hard_constraint_text(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    ["do not", "must", "never", "only", "required", "forbid"]
        .iter()
        .any(|needle| lower.contains(needle))
}

fn extract_commit_hashes(text: &str) -> Vec<String> {
    let lower = text.to_ascii_lowercase();
    if !(lower.contains("commit") || lower.contains("head")) {
        return Vec::new();
    }
    let mut hashes = Vec::new();
    for token in text.split(|ch: char| !ch.is_ascii_alphanumeric()) {
        if is_commit_hash_token(token) {
            hashes.push(token.to_string());
        }
    }
    hashes.sort();
    hashes.dedup();
    hashes
}

fn is_commit_hash_token(token: &str) -> bool {
    let len = token.len();
    (7..=40).contains(&len) && token.chars().all(|ch| ch.is_ascii_hexdigit())
}

fn value_text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}

fn extract_output_streams(data: &Value) -> Vec<(String, String, bool)> {
    let mut streams = Vec::new();
    for key in ["stdout", "stderr"] {
        if let Some(text) = data.get(key).and_then(Value::as_str) {
            let truncated = data
                .get(format!("{key}_truncated"))
                .and_then(Value::as_bool)
                .unwrap_or(false);
            streams.push((key.to_string(), text.to_string(), truncated));
        }
    }
    if streams.is_empty() && let Some(text) = data.get("output").and_then(Value::as_str) {
        streams.push(("output".into(), text.to_string(), false));
    }
    streams
}

fn folded_title(metadata: &FoldedOutputMetadata) -> String {
    match metadata.stream.as_deref() {
        Some(stream) if metadata.output_kind == "shell_output" => {
            format!("Folded shell {stream} output")
        }
        Some(stream) => format!("Folded {stream} output"),
        None => "Folded output".into(),
    }
}

fn folded_detail(metadata: &FoldedOutputMetadata) -> String {
    let status = folded_status_label(metadata);
    match &metadata.shell_command {
        Some(command) => format!("{} bytes from command: {command} ({status})", metadata.byte_count),
        None => format!("{} bytes retained by reference ({status})", metadata.byte_count),
    }
}

fn folded_status_label(metadata: &FoldedOutputMetadata) -> String {
    match (metadata.exit_status, metadata.tool_ok) {
        (Some(status), Some(ok)) => format!("status={status}, ok={ok}"),
        (Some(status), None) => format!("status={status}"),
        (None, Some(ok)) => format!("ok={ok}"),
        (None, None) => "status=unknown".into(),
    }
}

fn is_shell_tool_name(name: &str) -> bool {
    matches!(name, tool_names::TOOL_SHELL_EXEC | "bash")
}

fn count_lines(text: &str) -> usize {
    if text.is_empty() {
        0
    } else {
        text.lines().count().max(1)
    }
}

fn truncate_to_bytes(text: &str, max_bytes: usize) -> (String, usize, bool) {
    if text.len() <= max_bytes {
        return (text.to_string(), text.len(), false);
    }
    let mut end = max_bytes.min(text.len());
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    (text[..end].to_string(), end, true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{ToolExecutionSummaryEvent, ValidationAdvisory};
    use crate::tool::ToolResult;
    use crate::transcript::{TranscriptEvent, TranscriptRecord};
    use crate::user_content::UserMessageContent;
    use serde_json::json;

    fn record_at(sequence: u64, event: TranscriptEvent) -> TranscriptRecord {
        TranscriptRecord {
            session_id: "s".into(),
            sequence,
            timestamp_ms: 0,
            context_branch_id: None,
            event,
        }
    }

    #[test]
    fn protected_context_cannot_be_archived_or_removed() {
        let projection = project_context_view(&[record_at(
            1,
            TranscriptEvent::UserMessage {
                content: UserMessageContent::from("Do not commit; must keep tests passing"),
            },
        )])
        .expect("project context view");
        let block_id = ContextBlockId::new("block-seq-1-user-requirement").expect("block id");
        let block = projection.blocks.get(&block_id).expect("protected block exists");
        assert!(block.is_protected());

        let archive_error = ContextViewState::replay(
            &projection.blocks,
            &[ContextViewOperation::Archive {
                block_id: block_id.clone(),
            }],
        )
        .expect_err("archive should fail");
        assert!(archive_error.to_string().contains("cannot archive protected context block"));

        let remove_error = ContextViewState::replay(
            &projection.blocks,
            &[ContextViewOperation::RemoveFromView { block_id }],
        )
        .expect_err("remove should fail");
        assert!(remove_error
            .to_string()
            .contains("cannot remove_from_view protected context block"));
    }

    #[test]
    fn replay_supports_pin_archive_remove_and_open_detail() {
        let projection = project_context_view(&[
            record_at(
                1,
                TranscriptEvent::AssistantMessage {
                    content: "note one".into(),
                },
            ),
            record_at(
                2,
                TranscriptEvent::ContextViewOperationMetadata {
                    operation: "pin".into(),
                    node_id: None,
                    block_id: Some("block-seq-1-note".into()),
                    detail: None,
                },
            ),
            record_at(
                3,
                TranscriptEvent::ContextViewOperationMetadata {
                    operation: "open_detail".into(),
                    node_id: None,
                    block_id: Some("block-seq-1-note".into()),
                    detail: None,
                },
            ),
            record_at(
                4,
                TranscriptEvent::AssistantMessage {
                    content: "note two".into(),
                },
            ),
            record_at(
                5,
                TranscriptEvent::ContextViewOperationMetadata {
                    operation: "archive".into(),
                    node_id: None,
                    block_id: Some("block-seq-4-note".into()),
                    detail: None,
                },
            ),
            record_at(
                6,
                TranscriptEvent::AssistantMessage {
                    content: "note three".into(),
                },
            ),
            record_at(
                7,
                TranscriptEvent::ContextViewOperationMetadata {
                    operation: "remove_from_view".into(),
                    node_id: None,
                    block_id: Some("block-seq-6-note".into()),
                    detail: None,
                },
            ),
        ])
        .expect("project view state");

        assert_eq!(
            projection
                .view_state
                .status(&ContextBlockId::new("block-seq-1-note").expect("id")),
            Some(ContextViewStatus::Pinned)
        );
        assert_eq!(
            projection
                .view_state
                .status(&ContextBlockId::new("block-seq-4-note").expect("id")),
            Some(ContextViewStatus::Archived)
        );
        assert_eq!(
            projection
                .view_state
                .status(&ContextBlockId::new("block-seq-6-note").expect("id")),
            Some(ContextViewStatus::RemovedFromView)
        );
        assert_eq!(
            projection.view_state.open_detail_block_id().map(ContextBlockId::as_str),
            Some("block-seq-1-note")
        );
    }

    #[test]
    fn summary_artifacts_are_openable_and_versioned() {
        let projection = project_context_view(&[
            record_at(
                1,
                TranscriptEvent::ContextSummaryArtifactMetadata {
                    node_id: "node-a".into(),
                    artifact_id: "sum-v1".into(),
                    artifact_kind: "summary".into(),
                    version: Some(1),
                    summary: Some("first summary".into()),
                    source_node_id: Some("node-a".into()),
                    source_block_id: Some("block-seq-10-note".into()),
                    source_start_sequence: Some(10),
                    source_end_sequence: Some(12),
                },
            ),
            record_at(
                2,
                TranscriptEvent::ContextSummaryArtifactMetadata {
                    node_id: "node-a".into(),
                    artifact_id: "sum-v2".into(),
                    artifact_kind: "summary".into(),
                    version: Some(2),
                    summary: Some("second summary".into()),
                    source_node_id: Some("node-a".into()),
                    source_block_id: Some("block-seq-10-note".into()),
                    source_start_sequence: Some(10),
                    source_end_sequence: Some(14),
                },
            ),
        ])
        .expect("project summaries");

        let node_artifacts = projection.list_summary_artifacts_for_node("node-a");
        assert_eq!(node_artifacts.len(), 2);
        assert_eq!(node_artifacts[0].version, 1);
        assert_eq!(node_artifacts[1].version, 2);
        let latest = projection.open_summary_artifact("sum-v2").expect("artifact exists");
        assert_eq!(latest.summary, "second summary");
        assert_eq!(latest.source_end_sequence, Some(14));
        assert_eq!(latest.source_block_id.as_deref(), Some("block-seq-10-note"));
    }

    #[test]
    fn folded_output_metadata_is_openable_with_bounded_reads() {
        let projection = project_context_view(&[record_at(
            1,
            TranscriptEvent::FoldedOutputMetadata {
                node_id: Some("node-a".into()),
                output_id: "fold-1".into(),
                output_kind: "shell_output".into(),
                call_id: Some("call-1".into()),
                tool_name: Some("shell__exec".into()),
                stream: Some("stdout".into()),
                content: Some("abcdefghi".into()),
                byte_count: Some(9),
                line_count: Some(1),
                truncated: Some(false),
                shell_command: Some("git status".into()),
                source_start_sequence: Some(1),
                source_end_sequence: Some(1),
                tool_ok: Some(true),
                exit_status: Some(0),
            },
        )])
        .expect("project folded outputs");

        let opened = projection
            .open_folded_output("fold-1", 4)
            .expect("folded output exists");
        assert_eq!(opened.content, "abcd");
        assert!(opened.truncated);
        assert_eq!(opened.total_bytes, 9);
    }

    #[test]
    fn shell_outputs_are_folded_conservatively_without_summaries() {
        let large_stdout = "x".repeat(DEFAULT_FOLDED_OUTPUT_THRESHOLD_BYTES + 128);
        let projection = project_context_view(&[
            record_at(
                1,
                TranscriptEvent::ToolCallStarted {
                    call_id: "call-1".into(),
                    name: "shell__exec".into(),
                    args: json!({"command": "cargo test --bin letcode"}),
                },
            ),
            record_at(
                2,
                TranscriptEvent::ToolCallFinished {
                    call_id: "call-1".into(),
                    name: "shell__exec".into(),
                    ok: true,
                    output: ToolResult::ok(
                        "shell__exec",
                        json!({
                            "stdout": large_stdout,
                            "stdout_truncated": false,
                            "stderr": "",
                            "stderr_truncated": false
                        }),
                    ),
                },
            ),
        ])
        .expect("project shell folding");

        let folded = projection
            .folded_outputs
            .get("folded-output-seq-2-stdout")
            .expect("auto folded output exists");
        assert_eq!(folded.output_kind, "shell_output");
        assert_eq!(folded.shell_command.as_deref(), Some("cargo test --bin letcode"));
        let opened = projection
            .open_folded_output("folded-output-seq-2-stdout", DEFAULT_OPEN_CONTENT_MAX_BYTES)
            .expect("open folded output");
        assert!(opened.truncated);
        assert!(opened.content.chars().all(|ch| ch == 'x'));
        let block = projection
            .blocks
            .get(
                &ContextBlockId::new("block-seq-2-folded-output-folded-output-seq-2-stdout")
                    .expect("id"),
            )
            .expect("folded output block exists");
        assert_eq!(block.kind, ContextBlockKind::ToolOutput);
        assert_eq!(block.folded_output_id.as_deref(), Some("folded-output-seq-2-stdout"));
        assert!(block.detail.contains("cargo test --bin letcode"));
        assert!(block.detail.contains("ok=true"));
    }

    #[test]
    fn context_view_rejects_operation_targeting_future_block() {
        let error = project_context_view(&[
            record_at(
                1,
                TranscriptEvent::ContextViewOperationMetadata {
                    operation: "pin".into(),
                    node_id: None,
                    block_id: Some("block-seq-2-note".into()),
                    detail: None,
                },
            ),
            record_at(
                2,
                TranscriptEvent::AssistantMessage {
                    content: "future note".into(),
                },
            ),
        ])
        .expect_err("future block operation should fail");

        assert!(error
            .to_string()
            .contains("targets future block 'block-seq-2-note' created at sequence 2"));
    }

    #[test]
    fn context_view_rejects_operation_targeting_future_explicit_folded_metadata() {
        let error = project_context_view(&[
            record_at(
                1,
                TranscriptEvent::ToolCallFinished {
                    call_id: "call-explicit".into(),
                    name: "shell__exec".into(),
                    ok: true,
                    output: ToolResult::ok("shell__exec", json!({"stdout": "short"})),
                },
            ),
            record_at(
                2,
                TranscriptEvent::ContextViewOperationMetadata {
                    operation: "pin".into(),
                    node_id: None,
                    block_id: Some("block-seq-1-folded-output-fold-explicit".into()),
                    detail: None,
                },
            ),
            record_at(
                3,
                TranscriptEvent::FoldedOutputMetadata {
                    node_id: Some("node-a".into()),
                    output_id: "fold-explicit".into(),
                    output_kind: "shell_output".into(),
                    call_id: Some("call-explicit".into()),
                    tool_name: Some("shell__exec".into()),
                    stream: Some("stdout".into()),
                    content: Some("abcdefghi".into()),
                    byte_count: Some(9),
                    line_count: Some(1),
                    truncated: Some(false),
                    shell_command: Some("echo short".into()),
                    source_start_sequence: Some(1),
                    source_end_sequence: Some(1),
                    tool_ok: Some(true),
                    exit_status: Some(0),
                },
            ),
        ])
        .expect_err("future explicit folded metadata operation should fail");

        assert!(error.to_string().contains(
            "targets future block 'block-seq-1-folded-output-fold-explicit' created at sequence 3"
        ));
    }

    #[test]
    fn shell_call_with_large_stdout_and_stderr_creates_two_folded_blocks() {
        let large_stdout = "o".repeat(DEFAULT_FOLDED_OUTPUT_THRESHOLD_BYTES + 32);
        let large_stderr = "e".repeat(DEFAULT_FOLDED_OUTPUT_THRESHOLD_BYTES + 64);
        let projection = project_context_view(&[
            record_at(
                1,
                TranscriptEvent::ToolCallStarted {
                    call_id: "call-2".into(),
                    name: "shell__exec".into(),
                    args: json!({"command": "cargo test"}),
                },
            ),
            record_at(
                2,
                TranscriptEvent::ToolCallFinished {
                    call_id: "call-2".into(),
                    name: "shell__exec".into(),
                    ok: true,
                    output: ToolResult::ok(
                        "shell__exec",
                        json!({
                            "status": 0,
                            "stdout": large_stdout,
                            "stdout_truncated": false,
                            "stderr": large_stderr,
                            "stderr_truncated": false
                        }),
                    ),
                },
            ),
        ])
        .expect("project both folded streams");

        let stdout_block_id =
            ContextBlockId::new("block-seq-2-folded-output-folded-output-seq-2-stdout")
                .expect("stdout block id");
        let stderr_block_id =
            ContextBlockId::new("block-seq-2-folded-output-folded-output-seq-2-stderr")
                .expect("stderr block id");
        assert!(projection.blocks.contains_key(&stdout_block_id));
        assert!(projection.blocks.contains_key(&stderr_block_id));
        assert!(projection.open_folded_output("folded-output-seq-2-stdout", 64).is_some());
        assert!(projection.open_folded_output("folded-output-seq-2-stderr", 64).is_some());
    }

    #[test]
    fn failed_large_shell_output_preserves_status_and_command() {
        let large_stderr = "f".repeat(DEFAULT_FOLDED_OUTPUT_THRESHOLD_BYTES + 64);
        let projection = project_context_view(&[
            record_at(
                1,
                TranscriptEvent::ToolCallStarted {
                    call_id: "call-3".into(),
                    name: "shell__exec".into(),
                    args: json!({"command": "cargo test --quiet"}),
                },
            ),
            record_at(
                2,
                TranscriptEvent::ToolCallFinished {
                    call_id: "call-3".into(),
                    name: "shell__exec".into(),
                    ok: false,
                    output: ToolResult::err_with_data(
                        "shell__exec",
                        "command failed",
                        json!({
                            "status": 101,
                            "stdout": "",
                            "stdout_truncated": false,
                            "stderr": large_stderr,
                            "stderr_truncated": false
                        }),
                    ),
                },
            ),
        ])
        .expect("project failed folded shell output");

        let folded = projection
            .folded_outputs
            .get("folded-output-seq-2-stderr")
            .expect("failed folded stderr exists");
        assert_eq!(folded.tool_ok, Some(false));
        assert_eq!(folded.exit_status, Some(101));
        assert_eq!(folded.shell_command.as_deref(), Some("cargo test --quiet"));
        let block = projection
            .blocks
            .get(
                &ContextBlockId::new("block-seq-2-folded-output-folded-output-seq-2-stderr")
                    .expect("block id"),
            )
            .expect("failed folded block exists");
        assert!(block.detail.contains("cargo test --quiet"));
        assert!(block.detail.contains("status=101"));
        assert!(block.detail.contains("ok=false"));
    }

    #[test]
    fn projection_is_derived_without_mutating_raw_records() {
        let records = vec![
            record_at(
                1,
                TranscriptEvent::AssistantMessage {
                    content: "note".into(),
                },
            ),
            record_at(
                2,
                TranscriptEvent::ContextViewOperationMetadata {
                    operation: "archive".into(),
                    node_id: None,
                    block_id: Some("block-seq-1-note".into()),
                    detail: None,
                },
            ),
            record_at(
                3,
                TranscriptEvent::ContextSummaryArtifactMetadata {
                    node_id: "node-a".into(),
                    artifact_id: "sum-v1".into(),
                    artifact_kind: "summary".into(),
                    version: Some(1),
                    summary: Some("summary".into()),
                    source_node_id: Some("node-a".into()),
                    source_block_id: Some("block-seq-1-note".into()),
                    source_start_sequence: Some(1),
                    source_end_sequence: Some(1),
                },
            ),
            record_at(
                4,
                TranscriptEvent::FoldedOutputMetadata {
                    node_id: None,
                    output_id: "fold-raw".into(),
                    output_kind: "tool_output".into(),
                    call_id: None,
                    tool_name: Some("shell__exec".into()),
                    stream: Some("stdout".into()),
                    content: Some("abcdef".into()),
                    byte_count: Some(6),
                    line_count: Some(1),
                    truncated: Some(false),
                    shell_command: Some("git status".into()),
                    source_start_sequence: Some(4),
                    source_end_sequence: Some(4),
                    tool_ok: Some(true),
                    exit_status: Some(0),
                },
            ),
        ];
        let original_len = records.len();
        let projection = project_context_view(&records).expect("project derived view");

        assert_eq!(records.len(), original_len);
        assert_eq!(projection.summary_artifacts.len(), 1);
        assert!(projection.folded_outputs.contains_key("fold-raw"));
        assert_eq!(
            projection
                .view_state
                .status(&ContextBlockId::new("block-seq-1-note").expect("id")),
            Some(ContextViewStatus::Archived)
        );
    }

    #[test]
    fn protected_blocks_include_write_test_permission_and_commit_categories() {
        let projection = project_context_view(&[
            record_at(
                1,
                TranscriptEvent::PermissionDecision {
                    call_id: Some("call-1".into()),
                    tool: "shell__exec".into(),
                    args: json!({"command": "cargo test"}),
                    allowed: false,
                    reason: Some("Denied".into()),
                },
            ),
            record_at(
                2,
                TranscriptEvent::ToolExecutionSummary(ToolExecutionSummaryEvent {
                    turn_id: 1,
                    call_id: "call-2".into(),
                    name: "write".into(),
                    status: "completed".into(),
                    rejection: None,
                    effect_kind: "write".into(),
                    primary_path: Some("src/main.rs".into()),
                    command: None,
                }),
            ),
            record_at(
                3,
                TranscriptEvent::ValidationAdvisory(ValidationAdvisory {
                    write_effects: 1,
                    validation_effects: 1,
                    failed_validation_effects: 1,
                    message: "cargo test failed".into(),
                }),
            ),
            record_at(
                4,
                TranscriptEvent::AssistantMessage {
                    content: "commit abcdef1 is the current base".into(),
                },
            ),
        ])
        .expect("project protected blocks");

        assert!(projection
            .blocks
            .values()
            .any(|block| block.protected_reasons.contains(&ProtectedReason::Permission)));
        assert!(projection
            .blocks
            .values()
            .any(|block| block.protected_reasons.contains(&ProtectedReason::FileWriteFact)));
        assert!(projection
            .blocks
            .values()
            .any(|block| block.protected_reasons.contains(&ProtectedReason::TestResult)));
        assert!(projection
            .blocks
            .values()
            .any(|block| block.protected_reasons.contains(&ProtectedReason::CommitHash)));
    }
}
