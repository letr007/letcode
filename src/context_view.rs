use anyhow::{Result, anyhow, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use crate::context_tree::{ContextNodeId, ContextTreeOp, ContextTreeState};
use crate::runtime_context::SourceSpan;
use crate::tool_names;
use crate::transcript::transcript_projection::restore_retired_source_spans_projection;
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
    ReasoningNote,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ContextBlockRetention {
    Critical,
    Protected,
    Working,
    Debug,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub node_id: Option<String>,
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

    pub(crate) fn retention_class(&self) -> ContextBlockRetention {
        match self.kind {
            ContextBlockKind::HardConstraint | ContextBlockKind::CurrentUserRequirement => {
                ContextBlockRetention::Critical
            }
            ContextBlockKind::UnresolvedError
            | ContextBlockKind::Permission
            | ContextBlockKind::FileWriteFact
            | ContextBlockKind::TestResult
            | ContextBlockKind::CommitHash => ContextBlockRetention::Protected,
            ContextBlockKind::ToolOutput | ContextBlockKind::Note => ContextBlockRetention::Working,
            ContextBlockKind::ReasoningNote => ContextBlockRetention::Debug,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ContextViewStatus {
    Visible,
    Pinned,
    Archived,
    Resolved,
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
        let block = blocks.get(operation.block_id()).ok_or_else(|| {
            ContextViewError::UnknownBlock(operation.block_id().as_str().to_string())
        })?;
        let current = self
            .block_statuses
            .get(operation.block_id())
            .copied()
            .unwrap_or(ContextViewStatus::Visible);

        match operation {
            ContextViewOperation::Pin { block_id } => {
                if current == ContextViewStatus::Resolved {
                    return Err(ContextViewError::ResolvedBlockCannotPin(
                        block_id.as_str().into(),
                    ));
                }
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
            ContextViewOperation::Resolve { block_id } => {
                ensure_block_resolvable(block)?;
                self.block_statuses
                    .insert(block_id.clone(), ContextViewStatus::Resolved);
                if self.open_detail_block_id.as_ref() == Some(block_id) {
                    self.open_detail_block_id = None;
                }
            }
            ContextViewOperation::OpenDetail { block_id } => {
                if current == ContextViewStatus::RemovedFromView {
                    return Err(ContextViewError::RemovedBlockCannotOpen(
                        block_id.as_str().into(),
                    ));
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
    Resolve { block_id: ContextBlockId },
    OpenDetail { block_id: ContextBlockId },
}

impl ContextViewOperation {
    pub(crate) fn block_id(&self) -> &ContextBlockId {
        match self {
            Self::Pin { block_id }
            | Self::Archive { block_id }
            | Self::RemoveFromView { block_id }
            | Self::Resolve { block_id }
            | Self::OpenDetail { block_id } => block_id,
        }
    }

    pub(crate) fn parse(operation: &str, block_id: ContextBlockId) -> Result<Self> {
        match operation.trim() {
            "pin" => Ok(Self::Pin { block_id }),
            "archive" => Ok(Self::Archive { block_id }),
            "remove_from_view" => Ok(Self::RemoveFromView { block_id }),
            "resolve" => Ok(Self::Resolve { block_id }),
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
    ResolvedBlockCannotPin(String),
    NonResolvableBlock {
        block_id: String,
        kind: ContextBlockKind,
    },
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
            Self::ResolvedBlockCannotPin(block_id) => {
                write!(f, "cannot pin resolved context block '{block_id}'")
            }
            Self::NonResolvableBlock { block_id, kind } => write!(
                f,
                "cannot resolve context block '{block_id}' with kind {}",
                context_block_kind_label(*kind)
            ),
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct ContextViewProjection {
    pub blocks: BTreeMap<ContextBlockId, ContextBlock>,
    pub view_state: ContextViewState,
    pub summary_artifacts: Vec<SummaryArtifact>,
    pub folded_outputs: BTreeMap<String, FoldedOutputMetadata>,
    pub compacted_block_ids: BTreeSet<ContextBlockId>,
}

impl Default for ContextViewProjection {
    fn default() -> Self {
        Self {
            blocks: BTreeMap::new(),
            view_state: ContextViewState::default(),
            summary_artifacts: Vec::new(),
            folded_outputs: BTreeMap::new(),
            compacted_block_ids: BTreeSet::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RecordedContextViewOperation {
    pub sequence: u64,
    pub operation: ContextViewOperation,
}

impl ContextViewProjection {
    pub(crate) fn apply_retired_spans(&mut self, retired_spans: &[SourceSpan]) {
        self.compacted_block_ids = collect_compacted_block_ids_for_runtime(
            &self.blocks,
            &self.folded_outputs,
            retired_spans,
        );
        self.view_state
            .force_compacted_archived(&self.compacted_block_ids);
    }

    pub(crate) fn is_default_active(&self, block_id: &ContextBlockId) -> bool {
        !self.is_compacted(block_id)
            && matches!(
                self.status_for(block_id),
                ContextViewStatus::Visible | ContextViewStatus::Pinned
            )
    }

    /// Folded output compaction is owned by its source block. A folded output
    /// may have a multi-record source range, so the block compaction index is
    /// deliberately computed from that full range rather than its first record.
    pub(crate) fn is_compacted_folded_output(&self, output_id: &str) -> bool {
        self.blocks.iter().any(|(block_id, block)| {
            block.folded_output_id.as_deref() == Some(output_id) && self.is_compacted(block_id)
        })
    }

    pub(crate) fn is_addressable(&self, block_id: &ContextBlockId) -> bool {
        !self.is_compacted(block_id)
            && !matches!(
                self.status_for(block_id),
                ContextViewStatus::RemovedFromView | ContextViewStatus::Resolved
            )
    }
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

    pub(crate) fn is_compacted(&self, block_id: &ContextBlockId) -> bool {
        self.compacted_block_ids.contains(block_id)
    }

    pub(crate) fn status_for(&self, block_id: &ContextBlockId) -> ContextViewStatus {
        self.view_state
            .status(block_id)
            .unwrap_or(ContextViewStatus::Visible)
    }

    pub(crate) fn is_opened(&self, block_id: &ContextBlockId) -> bool {
        self.view_state.open_detail_block_id() == Some(block_id)
    }

    pub(crate) fn is_resolved(&self, block_id: &ContextBlockId) -> bool {
        self.status_for(block_id) == ContextViewStatus::Resolved
    }

    pub(crate) fn is_pinned_visible(&self, block_id: &ContextBlockId) -> bool {
        self.status_for(block_id) == ContextViewStatus::Pinned
    }

    pub(crate) fn is_normally_visible(&self, block_id: &ContextBlockId) -> bool {
        matches!(
            self.status_for(block_id),
            ContextViewStatus::Visible | ContextViewStatus::Pinned
        )
    }

    pub(crate) fn include_in_context_index(
        &self,
        block_id: &ContextBlockId,
        block: &ContextBlock,
    ) -> bool {
        if self.is_resolved(block_id) || self.is_compacted(block_id) {
            return false;
        }
        if block.retention_class() == ContextBlockRetention::Debug {
            return self.is_pinned_visible(block_id) || self.is_opened(block_id);
        }
        block.is_protected() || self.is_normally_visible(block_id)
    }

    pub(crate) fn provider_visible_block_ids(&self) -> Vec<String> {
        sorted_context_blocks(self)
            .into_iter()
            .filter(|(block_id, block)| self.include_in_context_index(block_id, block))
            .map(|(block_id, _)| block_id.as_str().to_string())
            .collect()
    }

    /// The complete block set exposed to a provider: index material plus the
    /// one valid detail block. This intentionally excludes all non-addressable
    /// material even when it would otherwise qualify for the index.
    pub(crate) fn is_provider_active_block(
        &self,
        block_id: &ContextBlockId,
        block: &ContextBlock,
    ) -> bool {
        if self.is_compacted(block_id)
            || matches!(
                self.status_for(block_id),
                ContextViewStatus::RemovedFromView | ContextViewStatus::Resolved
            )
        {
            return false;
        }
        self.include_in_context_index(block_id, block)
            || self.provider_open_detail_block_id().as_deref() == Some(block_id.as_str())
    }

    pub(crate) fn provider_active_blocks(&self) -> Vec<(&ContextBlockId, &ContextBlock)> {
        sorted_context_blocks(self)
            .into_iter()
            .filter(|(block_id, block)| self.is_provider_active_block(block_id, block))
            .collect()
    }

    /// Complete canonical source projection, independent of prompt visibility.
    pub(crate) fn all_context_blocks(&self) -> Vec<(&ContextBlockId, &ContextBlock)> {
        sorted_context_blocks(self)
    }

    /// Compacted blocks remain runtime provenance frames on restore even though
    /// they are no longer prompt-visible.
    pub(crate) fn provider_compacted_block_ids(&self) -> Vec<String> {
        sorted_context_blocks(self)
            .into_iter()
            .filter(|(block_id, _)| self.is_compacted(block_id))
            .map(|(block_id, _)| block_id.as_str().to_string())
            .collect()
    }

    pub(crate) fn provider_pinned_block_ids(&self) -> Vec<String> {
        sorted_context_blocks(self)
            .into_iter()
            .filter(|(block_id, _)| {
                !self.is_compacted(block_id) && self.is_pinned_visible(block_id)
            })
            .map(|(block_id, _)| block_id.as_str().to_string())
            .collect()
    }

    pub(crate) fn provider_open_detail_block_id(&self) -> Option<String> {
        let block_id = self.view_state.open_detail_block_id()?;
        if self.is_compacted(block_id)
            || self.status_for(block_id) == ContextViewStatus::RemovedFromView
            || self.is_resolved(block_id)
        {
            return None;
        }
        Some(block_id.as_str().to_string())
    }

    pub(crate) fn provider_folded_outputs(&self) -> Vec<&FoldedOutputMetadata> {
        sorted_context_blocks(self)
            .into_iter()
            .filter(|(block_id, block)| {
                !self.is_compacted(block_id)
                    && block.folded_output_id.is_some()
                    && (self.is_normally_visible(block_id) || self.is_opened(block_id))
            })
            .filter_map(|(_, block)| {
                block
                    .folded_output_id
                    .as_deref()
                    .and_then(|output_id| self.folded_outputs.get(output_id))
            })
            .collect()
    }

    /// Complete canonical output projection, independent of prompt visibility.
    pub(crate) fn all_folded_outputs(&self) -> Vec<&FoldedOutputMetadata> {
        let mut outputs = self.folded_outputs.values().collect::<Vec<_>>();
        outputs.sort_by(|left, right| {
            left.source_start_sequence
                .or(left.available_sequence)
                .unwrap_or(u64::MAX)
                .cmp(
                    &right
                        .source_start_sequence
                        .or(right.available_sequence)
                        .unwrap_or(u64::MAX),
                )
                .then_with(|| left.output_id.cmp(&right.output_id))
        });
        outputs
    }

    pub(crate) fn provider_compacted_folded_outputs(&self) -> Vec<&FoldedOutputMetadata> {
        sorted_context_blocks(self)
            .into_iter()
            .filter(|(block_id, block)| {
                self.is_compacted(block_id) && block.folded_output_id.is_some()
            })
            .filter_map(|(_, block)| {
                block
                    .folded_output_id
                    .as_deref()
                    .and_then(|output_id| self.folded_outputs.get(output_id))
            })
            .collect()
    }

    pub(crate) fn is_active_folded_output(&self, output_id: &str) -> bool {
        self.provider_folded_outputs()
            .iter()
            .any(|output| output.output_id == output_id)
    }
}

fn sorted_context_blocks(
    context_view: &ContextViewProjection,
) -> Vec<(&ContextBlockId, &ContextBlock)> {
    let mut blocks = context_view.blocks.iter().collect::<Vec<_>>();
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

pub(crate) fn project_context_view(records: &[TranscriptRecord]) -> Result<ContextViewProjection> {
    crate::transcript::transcript_projection::validate_successful_compactions(records)?;
    let folded_outputs = restore_folded_outputs(records, DEFAULT_FOLDED_OUTPUT_THRESHOLD_BYTES)?;
    let blocks = index_context_blocks(records, &folded_outputs)?;
    let operations = restore_context_view_operations(records)?;
    let retired_source_spans = restore_retired_source_spans_projection(records);
    let compacted_block_ids =
        collect_compacted_block_ids(&blocks, &folded_outputs, &retired_source_spans);
    let view_state = replay_recorded_context_view_state(&blocks, &operations, &compacted_block_ids)
        .map_err(|error| anyhow!(error.to_string()))?;
    let summary_artifacts = restore_summary_artifacts(records)?;
    Ok(ContextViewProjection {
        blocks,
        view_state,
        summary_artifacts,
        folded_outputs,
        compacted_block_ids,
    })
}

pub(crate) fn index_context_blocks(
    records: &[TranscriptRecord],
    folded_outputs: &BTreeMap<String, FoldedOutputMetadata>,
) -> Result<BTreeMap<ContextBlockId, ContextBlock>> {
    let mut blocks = BTreeMap::new();
    let mut context_tree = ContextTreeState::with_default_root();
    let folded_by_sequence = folded_outputs
        .values()
        .filter_map(|metadata| {
            metadata
                .source_start_sequence
                .map(|sequence| (sequence, metadata))
        })
        .fold(
            BTreeMap::<u64, Vec<&FoldedOutputMetadata>>::new(),
            |mut acc, (sequence, metadata)| {
                acc.entry(sequence).or_default().push(metadata);
                acc
            },
        );

    for record in records {
        apply_context_tree_record(&mut context_tree, record)?;
        let active_node_id = context_tree
            .active_node_id()
            .map(|node_id| node_id.as_str().to_string());
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
                            None,
                        );
                    }
                }
            }
            TranscriptEvent::AssistantMessage { content } => {
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
                            None,
                        );
                    }
                }
            }
            TranscriptEvent::ReasoningMessage { content } => {
                if !content.trim().is_empty() {
                    insert_block(
                        &mut blocks,
                        block_id(record.sequence, "reasoning-note"),
                        ContextBlockKind::ReasoningNote,
                        "Reasoning note".into(),
                        content.clone(),
                        transcript_source(record.sequence),
                        Some(record.sequence),
                        Vec::new(),
                        None,
                        None,
                    );
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
                    None,
                );
            }
            TranscriptEvent::ToolExecutionSummary(event) => match event.effect_kind.as_str() {
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
                    Some(active_node_id.clone().ok_or_else(|| {
                        anyhow!(
                            "file write fact at transcript sequence {} has no active context node",
                            record.sequence
                        )
                    })?),
                    None,
                ),
                "validation" => insert_block(
                    &mut blocks,
                    block_id(record.sequence, "test"),
                    ContextBlockKind::TestResult,
                    "Test result".into(),
                    event.command.clone().unwrap_or_else(|| event.name.clone()),
                    transcript_source(record.sequence),
                    Some(record.sequence),
                    vec![ProtectedReason::TestResult],
                    None,
                    None,
                ),
                _ => {}
            },
            TranscriptEvent::ToolCallFinished {
                name, ok, output, ..
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
                            metadata.node_id.clone(),
                            Some(metadata.output_id.clone()),
                        );
                    }
                }

                if let Some(data) = &output.data {
                    for (index, hash) in extract_commit_hashes(&value_text(data))
                        .into_iter()
                        .enumerate()
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
                            None,
                        );
                    }
                }
            }
            _ => {}
        }
    }

    Ok(blocks)
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
            operation: ContextViewOperation::parse(
                operation,
                ContextBlockId::new(block_id.clone())?,
            )?,
        });
    }
    Ok(operations)
}

pub(crate) fn replay_recorded_context_view_state(
    blocks: &BTreeMap<ContextBlockId, ContextBlock>,
    operations: &[RecordedContextViewOperation],
    compacted_block_ids: &BTreeSet<ContextBlockId>,
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
    state.force_compacted_archived(compacted_block_ids);
    Ok(state)
}

fn collect_compacted_block_ids(
    blocks: &BTreeMap<ContextBlockId, ContextBlock>,
    folded_outputs: &BTreeMap<String, FoldedOutputMetadata>,
    retired_source_spans: &[crate::agent::ContextCompactionSourceSpan],
) -> BTreeSet<ContextBlockId> {
    blocks
        .iter()
        .filter_map(|(block_id, block)| {
            let start = block.source_start_sequence?;
            let end = match &block.source {
                ContextBlockSource::TranscriptSpan { end_sequence, .. } => *end_sequence,
                ContextBlockSource::FoldedOutput { output_id } => folded_outputs
                    .get(output_id)
                    .and_then(|output| output.source_end_sequence)
                    .unwrap_or(start),
                _ => start,
            };
            // Context blocks are atomic retrieval units. Retiring any portion
            // makes the entire block non-addressable.
            retired_source_spans
                .iter()
                .any(|span| span.start_sequence <= end && start <= span.end_sequence)
                .then_some(block_id.clone())
        })
        .collect()
}

fn collect_compacted_block_ids_for_runtime(
    blocks: &BTreeMap<ContextBlockId, ContextBlock>,
    folded_outputs: &BTreeMap<String, FoldedOutputMetadata>,
    retired_spans: &[SourceSpan],
) -> BTreeSet<ContextBlockId> {
    blocks
        .iter()
        .filter_map(|(block_id, block)| {
            let start = block.source_start_sequence?;
            let end = match &block.source {
                ContextBlockSource::TranscriptSpan { end_sequence, .. } => *end_sequence,
                ContextBlockSource::FoldedOutput { output_id } => folded_outputs
                    .get(output_id)
                    .and_then(|output| output.source_end_sequence)
                    .unwrap_or(start),
                _ => start,
            };
            let span = SourceSpan::new(start, end).ok()?;
            retired_spans
                .iter()
                .copied()
                .any(|retired| retired.overlaps(span))
                .then_some(block_id.clone())
        })
        .collect()
}

impl ContextViewState {
    fn force_compacted_archived(&mut self, compacted_block_ids: &BTreeSet<ContextBlockId>) {
        for block_id in compacted_block_ids {
            self.block_statuses
                .insert(block_id.clone(), ContextViewStatus::Archived);
            if self.open_detail_block_id.as_ref() == Some(block_id) {
                self.open_detail_block_id = None;
            }
        }
    }
}

pub(crate) fn restore_summary_artifacts(
    records: &[TranscriptRecord],
) -> Result<Vec<SummaryArtifact>> {
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
                    outputs
                        .entry(output_id.clone())
                        .or_insert_with(|| FoldedOutputMetadata {
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
    block
        .available_sequence
        .or(block.source_start_sequence)
        .unwrap_or(0)
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

fn ensure_block_resolvable(block: &ContextBlock) -> std::result::Result<(), ContextViewError> {
    if block.kind != ContextBlockKind::UnresolvedError {
        return Err(ContextViewError::NonResolvableBlock {
            block_id: block.block_id.as_str().to_string(),
            kind: block.kind,
        });
    }
    Ok(())
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
    node_id: Option<String>,
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
        node_id,
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
    node_id: Option<String>,
    folded_output_id: Option<String>,
) {
    blocks.insert(
        block_id.clone(),
        ContextBlock {
            block_id,
            node_id,
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

fn apply_context_tree_record(
    context_tree: &mut ContextTreeState,
    record: &TranscriptRecord,
) -> Result<()> {
    let op = match &record.event {
        TranscriptEvent::ContextNodeCreated {
            node_id,
            parent_node_id,
            label,
            purpose,
            block_ref,
            source_ref,
        } => Some(ContextTreeOp::CreateNode {
            node_id: ContextNodeId::new(node_id.clone())?,
            parent_node_id: parent_node_id.clone().map(ContextNodeId::new).transpose()?,
            label: label.clone(),
            purpose: purpose.clone(),
            block_ref: block_ref.clone(),
            source_ref: source_ref.clone(),
        }),
        TranscriptEvent::ContextNodeLifecycle { node_id, status } => {
            Some(ContextTreeOp::SetNodeStatus {
                node_id: ContextNodeId::new(node_id.clone())?,
                status: status.clone(),
            })
        }
        _ => None,
    };

    if let Some(op) = op {
        context_tree.apply(&op)?;
    }

    Ok(())
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
    if streams.is_empty()
        && let Some(text) = data.get("output").and_then(Value::as_str)
    {
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
        Some(command) => format!(
            "{} bytes from command: {command} ({status})",
            metadata.byte_count
        ),
        None => format!(
            "{} bytes retained by reference ({status})",
            metadata.byte_count
        ),
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
    use crate::agent::{
        ContextCompactionSourceSpan, ToolExecutionSummaryEvent, ValidationAdvisory,
    };
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
        let block = projection
            .blocks
            .get(&block_id)
            .expect("protected block exists");
        assert!(block.is_protected());

        let archive_error = ContextViewState::replay(
            &projection.blocks,
            &[ContextViewOperation::Archive {
                block_id: block_id.clone(),
            }],
        )
        .expect_err("archive should fail");
        assert!(
            archive_error
                .to_string()
                .contains("cannot archive protected context block")
        );

        let remove_error = ContextViewState::replay(
            &projection.blocks,
            &[ContextViewOperation::RemoveFromView { block_id }],
        )
        .expect_err("remove should fail");
        assert!(
            remove_error
                .to_string()
                .contains("cannot remove_from_view protected context block")
        );
    }

    #[test]
    fn resolve_marks_unresolved_error_without_mutating_raw_block() {
        let projection = project_context_view(&[
            record_at(
                1,
                TranscriptEvent::Error {
                    message: "context view projection unavailable".into(),
                },
            ),
            record_at(
                2,
                TranscriptEvent::ContextViewOperationMetadata {
                    operation: "resolve".into(),
                    node_id: None,
                    block_id: Some("block-seq-1-error".into()),
                    detail: None,
                },
            ),
        ])
        .expect("resolved error projects");

        let block_id = ContextBlockId::new("block-seq-1-error").expect("id");
        let block = projection
            .blocks
            .get(&block_id)
            .expect("error block exists");
        assert_eq!(block.kind, ContextBlockKind::UnresolvedError);
        assert!(block.is_protected());
        assert_eq!(
            projection.view_state.status(&block_id),
            Some(ContextViewStatus::Resolved)
        );
    }

    #[test]
    fn resolve_rejects_non_error_blocks() {
        let error = project_context_view(&[
            record_at(
                1,
                TranscriptEvent::AssistantMessage {
                    content: "plain note".into(),
                },
            ),
            record_at(
                2,
                TranscriptEvent::ContextViewOperationMetadata {
                    operation: "resolve".into(),
                    node_id: None,
                    block_id: Some("block-seq-1-note".into()),
                    detail: None,
                },
            ),
        ])
        .expect_err("non-error resolve should fail");

        assert!(
            error
                .to_string()
                .contains("cannot resolve context block 'block-seq-1-note' with kind note")
        );
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
            projection
                .view_state
                .open_detail_block_id()
                .map(ContextBlockId::as_str),
            Some("block-seq-1-note")
        );
    }

    #[test]
    fn reasoning_messages_are_debug_retention_blocks() {
        let projection = project_context_view(&[
            record_at(
                1,
                TranscriptEvent::ReasoningMessage {
                    content: "scratch reasoning trace".into(),
                },
            ),
            record_at(
                2,
                TranscriptEvent::AssistantMessage {
                    content: "durable assistant note".into(),
                },
            ),
        ])
        .expect("project context view");

        let reasoning_id = ContextBlockId::new("block-seq-1-reasoning-note").expect("id");
        let reasoning = projection
            .blocks
            .get(&reasoning_id)
            .expect("reasoning block exists");
        assert_eq!(reasoning.kind, ContextBlockKind::ReasoningNote);
        assert_eq!(reasoning.retention_class(), ContextBlockRetention::Debug);
        assert!(!reasoning.is_protected());
        assert!(
            !projection
                .blocks
                .contains_key(&ContextBlockId::new("block-seq-1-note").expect("id"))
        );

        let note = projection
            .blocks
            .get(&ContextBlockId::new("block-seq-2-note").expect("id"))
            .expect("assistant note exists");
        assert_eq!(note.kind, ContextBlockKind::Note);
        assert_eq!(note.retention_class(), ContextBlockRetention::Working);
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
        let latest = projection
            .open_summary_artifact("sum-v2")
            .expect("artifact exists");
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
        assert_eq!(
            folded.shell_command.as_deref(),
            Some("cargo test --bin letcode")
        );
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
        assert_eq!(
            block.folded_output_id.as_deref(),
            Some("folded-output-seq-2-stdout")
        );
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

        assert!(
            error
                .to_string()
                .contains("targets future block 'block-seq-2-note' created at sequence 2")
        );
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
    fn compaction_archives_retired_old_blocks_but_leaves_tail_visible() {
        let projection = project_context_view(&[
            record_at(
                1,
                TranscriptEvent::UserMessage {
                    content: UserMessageContent::from("old requirement"),
                },
            ),
            record_at(
                2,
                TranscriptEvent::AssistantMessage {
                    content: "old note".into(),
                },
            ),
            record_at(
                3,
                TranscriptEvent::ToolCallStarted {
                    call_id: "call-1".into(),
                    name: "shell__exec".into(),
                    args: json!({"command": "cargo test"}),
                },
            ),
            record_at(
                4,
                TranscriptEvent::ToolCallFinished {
                    call_id: "call-1".into(),
                    name: "shell__exec".into(),
                    ok: true,
                    output: ToolResult::ok(
                        "shell__exec",
                        json!({"stdout": "x".repeat(5000), "stdout_truncated": false, "stderr": "", "stderr_truncated": false}),
                    ),
                },
            ),
            record_at(
                5,
                TranscriptEvent::ContextViewOperationMetadata {
                    operation: "open_detail".into(),
                    node_id: None,
                    block_id: Some("block-seq-2-note".into()),
                    detail: None,
                },
            ),
            record_at(
                6,
                TranscriptEvent::ContextCompaction(crate::agent::ContextCompactionEvent {
                    outcome: "succeeded".into(),
                    summary: "summary".into(),
                    tail_start_index: 4,
                    original_history_items: 4,
                    retained_history_items: 1,
                    retired_source_spans: vec![ContextCompactionSourceSpan {
                        start_sequence: 1,
                        end_sequence: 4,
                    }],
                    frame_identity_bindings: Vec::new(),
                    detail: None,
                }),
            ),
            record_at(
                7,
                TranscriptEvent::UserMessage {
                    content: UserMessageContent::from("current requirement"),
                },
            ),
        ])
        .expect("project compacted context view");

        let old_user = ContextBlockId::new("block-seq-1-user-requirement").expect("id");
        let old_note = ContextBlockId::new("block-seq-2-note").expect("id");
        let old_folded =
            ContextBlockId::new("block-seq-4-folded-output-folded-output-seq-4-stdout")
                .expect("id");
        let tail_user = ContextBlockId::new("block-seq-7-user-requirement").expect("id");

        assert!(projection.is_compacted(&old_user));
        assert!(projection.is_compacted(&old_note));
        assert!(projection.is_compacted(&old_folded));
        assert!(!projection.is_compacted(&tail_user));
        assert_eq!(
            projection.view_state.status(&old_note),
            Some(ContextViewStatus::Archived)
        );
        assert_eq!(projection.view_state.open_detail_block_id(), None);
        assert_eq!(
            projection.view_state.status(&tail_user),
            Some(ContextViewStatus::Visible)
        );
    }

    #[test]
    fn partial_retired_span_compacts_transcript_and_folded_blocks() {
        let transcript = ContextBlockId::new("transcript").expect("id");
        let folded = ContextBlockId::new("folded").expect("id");
        let mut blocks = BTreeMap::new();
        for (id, source, start) in [
            (
                transcript.clone(),
                ContextBlockSource::TranscriptSpan {
                    start_sequence: 10,
                    end_sequence: 12,
                },
                10,
            ),
            (
                folded.clone(),
                ContextBlockSource::FoldedOutput {
                    output_id: "output".into(),
                },
                10,
            ),
        ] {
            blocks.insert(
                id.clone(),
                ContextBlock {
                    block_id: id,
                    node_id: None,
                    kind: ContextBlockKind::ToolOutput,
                    title: "block".into(),
                    detail: "detail".into(),
                    source,
                    source_start_sequence: Some(start),
                    available_sequence: Some(start),
                    protected_reasons: Vec::new(),
                    folded_output_id: None,
                },
            );
        }
        let mut folded_outputs = BTreeMap::new();
        folded_outputs.insert(
            "output".into(),
            FoldedOutputMetadata {
                output_id: "output".into(),
                node_id: None,
                output_kind: "shell_output".into(),
                call_id: None,
                tool_name: None,
                stream: None,
                content: "detail".into(),
                byte_count: 6,
                line_count: 1,
                truncated: false,
                shell_command: None,
                source_start_sequence: Some(10),
                source_end_sequence: Some(12),
                available_sequence: Some(10),
                tool_ok: None,
                exit_status: None,
            },
        );

        let runtime = collect_compacted_block_ids_for_runtime(
            &blocks,
            &folded_outputs,
            &[SourceSpan::new(12, 13).expect("span")],
        );
        let durable = collect_compacted_block_ids(
            &blocks,
            &folded_outputs,
            &[ContextCompactionSourceSpan {
                start_sequence: 9,
                end_sequence: 10,
            }],
        );
        assert_eq!(
            runtime,
            BTreeSet::from([transcript.clone(), folded.clone()])
        );
        assert_eq!(durable, BTreeSet::from([transcript, folded]));
    }

    #[test]
    fn derived_retired_region_compacts_non_history_blocks_between_retired_history_sources() {
        let projection = project_context_view(&[
            record_at(
                1,
                TranscriptEvent::UserMessage {
                    content: UserMessageContent::from("old user"),
                },
            ),
            record_at(
                2,
                TranscriptEvent::ReasoningMessage {
                    content: "old reasoning".into(),
                },
            ),
            record_at(
                3,
                TranscriptEvent::AssistantMessage {
                    content: "old assistant".into(),
                },
            ),
            record_at(
                4,
                TranscriptEvent::ContextCompaction(crate::agent::ContextCompactionEvent {
                    outcome: "succeeded".into(),
                    summary: "summary".into(),
                    tail_start_index: 2,
                    original_history_items: 2,
                    retained_history_items: 1,
                    retired_source_spans: Vec::new(),
                    frame_identity_bindings: Vec::new(),
                    detail: None,
                }),
            ),
            record_at(
                5,
                TranscriptEvent::AssistantMessage {
                    content: "retained tail".into(),
                },
            ),
        ])
        .expect("project compacted context view with reasoning block");

        let reasoning = ContextBlockId::new("block-seq-2-reasoning-note").expect("id");
        let retained_tail = ContextBlockId::new("block-seq-5-note").expect("id");

        assert!(projection.is_compacted(&reasoning));
        assert_eq!(
            projection.view_state.status(&reasoning),
            Some(ContextViewStatus::Archived)
        );
        assert!(!projection.is_compacted(&retained_tail));
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
        assert!(
            projection
                .open_folded_output("folded-output-seq-2-stdout", 64)
                .is_some()
        );
        assert!(
            projection
                .open_folded_output("folded-output-seq-2-stderr", 64)
                .is_some()
        );
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

        assert!(projection.blocks.values().any(|block| {
            block
                .protected_reasons
                .contains(&ProtectedReason::Permission)
        }));
        assert!(projection.blocks.values().any(|block| {
            block
                .protected_reasons
                .contains(&ProtectedReason::FileWriteFact)
        }));
        assert!(projection.blocks.values().any(|block| {
            block
                .protected_reasons
                .contains(&ProtectedReason::TestResult)
        }));
        assert!(projection.blocks.values().any(|block| {
            block
                .protected_reasons
                .contains(&ProtectedReason::CommitHash)
        }));
    }

    #[test]
    fn file_write_fact_stays_protected_and_tracks_active_context_node() {
        let projection = project_context_view(&[
            record_at(
                1,
                TranscriptEvent::ContextNodeCreated {
                    node_id: "branch/parser-fix".into(),
                    parent_node_id: Some("root".into()),
                    label: Some("Parser fix".into()),
                    purpose: None,
                    block_ref: None,
                    source_ref: None,
                },
            ),
            record_at(
                2,
                TranscriptEvent::ContextNodeLifecycle {
                    node_id: "root".into(),
                    status: crate::context_tree::ContextNodeStatus::Inactive,
                },
            ),
            record_at(
                3,
                TranscriptEvent::ContextNodeLifecycle {
                    node_id: "branch/parser-fix".into(),
                    status: crate::context_tree::ContextNodeStatus::Active,
                },
            ),
            record_at(
                4,
                TranscriptEvent::ToolExecutionSummary(ToolExecutionSummaryEvent {
                    turn_id: 1,
                    call_id: "call-write".into(),
                    name: "fs__write".into(),
                    status: "completed".into(),
                    rejection: None,
                    effect_kind: "write".into(),
                    primary_path: Some("src/lib.rs".into()),
                    command: None,
                }),
            ),
            record_at(
                5,
                TranscriptEvent::ToolCallFinished {
                    call_id: "call-return".into(),
                    name: tool_names::TOOL_CONTEXT_RETURN.into(),
                    ok: true,
                    output: ToolResult::ok(
                        tool_names::TOOL_CONTEXT_RETURN,
                        json!({
                            "message": "Returned from the current context experiment to the parent context. Files were not reverted.",
                            "warning": "Context restored, files were NOT reverted"
                        }),
                    ),
                },
            ),
        ])
        .expect("project write fact with context node");

        let block = projection
            .blocks
            .get(&ContextBlockId::new("block-seq-4-write").expect("block id"))
            .expect("write fact block exists");
        assert!(block.is_protected());
        assert_eq!(block.node_id.as_deref(), Some("branch/parser-fix"));
        assert_eq!(block.detail, "src/lib.rs");
        assert!(!block.detail.contains("reverted"));

        let error = ContextViewState::replay(
            &projection.blocks,
            &[ContextViewOperation::Archive {
                block_id: ContextBlockId::new("block-seq-4-write").expect("block id"),
            }],
        )
        .expect_err("write fact should remain protected");
        assert!(
            error
                .to_string()
                .contains("cannot archive protected context block")
        );
    }

    #[test]
    fn context_view_projection_reports_invalid_context_tree_metadata() {
        let error = project_context_view(&[record_at(
            1,
            TranscriptEvent::ContextNodeCreated {
                node_id: "child".into(),
                parent_node_id: Some("missing".into()),
                label: None,
                purpose: None,
                block_ref: None,
                source_ref: None,
            },
        )])
        .expect_err("invalid context tree metadata should fail");

        assert!(
            error
                .to_string()
                .contains("unknown parent context node 'missing'")
        );
    }

    #[test]
    fn file_write_fact_requires_active_context_node() {
        let error = project_context_view(&[
            record_at(
                1,
                TranscriptEvent::ContextNodeLifecycle {
                    node_id: "root".into(),
                    status: crate::context_tree::ContextNodeStatus::Inactive,
                },
            ),
            record_at(
                2,
                TranscriptEvent::ToolExecutionSummary(ToolExecutionSummaryEvent {
                    turn_id: 1,
                    call_id: "call-write".into(),
                    name: "fs__write".into(),
                    status: "completed".into(),
                    rejection: None,
                    effect_kind: "write".into(),
                    primary_path: Some("src/lib.rs".into()),
                    command: None,
                }),
            ),
        ])
        .expect_err("write fact without active context node should fail");

        assert!(
            error
                .to_string()
                .contains("file write fact at transcript sequence 2 has no active context node")
        );
    }
}
