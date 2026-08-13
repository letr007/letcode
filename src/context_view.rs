use anyhow::{Result, anyhow, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use crate::context_tree::{ContextNodeId, ContextTreeOp, ContextTreeState};
use crate::runtime_context::SourceSpan;
use crate::transcript::{TranscriptEvent, TranscriptRecord};

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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub(crate) struct ContextViewState {
    block_statuses: BTreeMap<ContextBlockId, ContextViewStatus>,
    open_detail_block_id: Option<ContextBlockId>,
}

impl ContextViewState {
    #[cfg(test)]
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub(crate) struct ContextViewProjection {
    pub blocks: BTreeMap<ContextBlockId, ContextBlock>,
    pub view_state: ContextViewState,
    pub summary_artifacts: Vec<SummaryArtifact>,
    pub compacted_block_ids: BTreeSet<ContextBlockId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RecordedContextViewOperation {
    pub sequence: u64,
    pub operation: ContextViewOperation,
}

impl ContextViewProjection {
    pub(crate) fn apply_retired_spans(&mut self, retired_spans: &[SourceSpan]) {
        self.compacted_block_ids =
            collect_compacted_block_ids_for_runtime(&self.blocks, retired_spans);
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

    pub(crate) fn open_summary_artifact(&self, artifact_id: &str) -> Option<&SummaryArtifact> {
        self.summary_artifacts
            .iter()
            .find(|artifact| artifact.artifact_id == artifact_id)
    }

    #[cfg(test)]
    pub(crate) fn list_summary_artifacts_for_node(&self, node_id: &str) -> Vec<&SummaryArtifact> {
        self.summary_artifacts
            .iter()
            .filter(|artifact| artifact.node_id == node_id)
            .collect()
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
    pub(crate) fn provider_pinned_block_ids(&self) -> Vec<String> {
        sorted_context_blocks(self)
            .into_iter()
            .filter(|(block_id, _block)| {
                !self.is_compacted(block_id) && self.is_pinned_visible(block_id)
            })
            .map(|(block_id, _)| block_id.as_str().to_string())
            .collect()
    }

    pub(crate) fn provider_open_detail_block_id(&self) -> Option<String> {
        let block_id = self.view_state.open_detail_block_id()?;
        let _block = self.blocks.get(block_id)?;
        if self.is_compacted(block_id)
            || self.status_for(block_id) == ContextViewStatus::RemovedFromView
            || self.is_resolved(block_id)
        {
            return None;
        }
        Some(block_id.as_str().to_string())
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

#[allow(dead_code)] // Kept as the direct transcript projection API for internal callers and focused tests.
pub(crate) fn project_context_view(records: &[TranscriptRecord]) -> Result<ContextViewProjection> {
    crate::transcript::transcript_projection::validate_context_projection_events(records)?;
    project_context_view_unvalidated(records)
}

/// Builds the canonical Phase 2 artifact view once replacement events have
/// already been validated. Checkpoint validation uses this non-recursive form.
pub(crate) fn project_context_view_unvalidated(
    records: &[TranscriptRecord],
) -> Result<ContextViewProjection> {
    let blocks = index_context_blocks(records)?;
    let operations = restore_context_view_operations(records)?;
    let compacted_block_ids = BTreeSet::new();
    let view_state = replay_recorded_context_view_state(&blocks, &operations, &compacted_block_ids)
        .map_err(|error| anyhow!(error.to_string()))?;
    let summary_artifacts = restore_summary_artifacts(records)?;
    Ok(ContextViewProjection {
        blocks,
        view_state,
        summary_artifacts,
        compacted_block_ids,
    })
}

pub(crate) fn index_context_blocks(
    records: &[TranscriptRecord],
) -> Result<BTreeMap<ContextBlockId, ContextBlock>> {
    let mut blocks = BTreeMap::new();
    let mut context_tree = ContextTreeState::with_default_root();

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
            TranscriptEvent::AssistantMessage { content } if !content.trim().is_empty() => {
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
            TranscriptEvent::ReasoningMessage { content, .. } if !content.trim().is_empty() => {
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
                );
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
                    );
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

fn collect_compacted_block_ids_for_runtime(
    blocks: &BTreeMap<ContextBlockId, ContextBlock>,
    retired_spans: &[SourceSpan],
) -> BTreeSet<ContextBlockId> {
    blocks
        .iter()
        .filter_map(|(block_id, block)| {
            let start = block.source_start_sequence?;
            let end = match &block.source {
                ContextBlockSource::TranscriptSpan { end_sequence, .. } => *end_sequence,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{ContextCompactionEvent, ToolExecutionSummaryEvent, ValidationAdvisory};
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
