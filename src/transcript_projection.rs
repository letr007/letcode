use crate::agent::{AutoContinueState, ContextCompactionFrameBinding, ContextCompactionSourceSpan};
use crate::context_tree::{ContextNodeId, ContextTreeOp, ContextTreeState};
use crate::context_view::{self, ContextViewProjection};
use crate::evidence::EvidenceRecord;
use crate::protocol_frames::{ProtocolFrame, analyze_history_items, history_items_to_frames};
use crate::request_builder::HistoryItem;
use crate::runtime_context::{
    FoldedOutputReference, FrameVisibility, PromptContributorKind, PromptContributorPlaceholder,
    RuntimeChildSession, RuntimeFrame, RuntimeFrameId, RuntimeFrameIdSeed, RuntimeFrameKind,
    RuntimeFrameProvenance, RuntimeSnapshot, RuntimeSource, SourceSpan,
};
use crate::tool_format::format_tool_call;
use crate::transcript::{
    ChildSessionSummary, JobBoardEntry, ROOT_CONTEXT_BRANCH_ID, TranscriptEvent, TranscriptRecord,
};
use crate::tui::events::{
    AutoContinueChangedEvent, ErrorEvent, TodoSnapshotEvent, TokenUsageEvent, ToolFinishedEvent,
    ToolOutcome, ToolStartedEvent, UserMessageEvent,
};
use crate::tui::timeline::{
    COMPACTION_SEPARATOR_LABEL, MessageRole, PermissionPromptStatus, Timeline,
    restored_tool_summary,
};
use crate::user_content::UserMessageSubmission;
use crate::{agent::ConversationMessage, subagent::StructuredSubagentResult};
use anyhow::{Context, anyhow, ensure};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

/// Transcript restore intentionally differs from live projection in a few places:
/// - Permission decisions restore as terminal approved/denied permission items, not as pending prompts.
/// - Context compaction restores as separator + assistant summary + separator.
/// - Subagent lifecycle/result records are ignored in restored timelines.
/// - Turn audit and unknown transcript events are ignored during timeline restore.
pub(crate) fn timeline_from_transcript_records(records: &[TranscriptRecord]) -> Timeline {
    let mut projection = TranscriptTimelineProjection::default();
    for record in records {
        projection.apply_record(record);
    }
    projection.timeline
}

#[derive(Debug, Clone)]
pub(crate) struct SessionContextCursor {
    pub branch_id: Option<String>,
    pub leaf_sequence: Option<u64>,
}

#[derive(Debug, Clone)]
pub(crate) struct SessionRestoreSnapshot {
    pub session_id: String,
    pub branch_id: String,
    pub leaf_sequence: u64,
    pub records: Vec<TranscriptRecord>,
    pub messages: Vec<ConversationMessage>,
    pub history: Vec<HistoryItem>,
    pub evidence: Vec<EvidenceRecord>,
    pub latest_model: Option<String>,
    pub max_turn_id: u64,
    pub token_usage: Option<TokenUsageEvent>,
}

#[derive(Debug, Clone)]
pub(crate) struct RuntimeRestoreSnapshot {
    pub session_id: String,
    pub branch_id: String,
    pub leaf_sequence: u64,
    pub records: Vec<TranscriptRecord>,
    pub protocol_frames: Vec<ProtocolFrame>,
    pub snapshot: RuntimeSnapshot,
    pub latest_model: Option<String>,
    pub max_turn_id: u64,
    pub token_usage: Option<TokenUsageEvent>,
}

impl SessionRestoreSnapshot {
    pub(crate) fn evidence_count(&self) -> usize {
        self.evidence.len()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ContextBranchInfo {
    pub branch_id: String,
    pub parent_branch_id: Option<String>,
    pub label: Option<String>,
    pub tip_sequence: u64,
    pub is_current: bool,
}

pub(crate) fn project_session_restore_snapshot(
    session_id: String,
    records: Vec<TranscriptRecord>,
    token_usage: Option<TokenUsageEvent>,
) -> anyhow::Result<SessionRestoreSnapshot> {
    build_session_context_snapshot(
        session_id,
        records,
        token_usage,
        SessionContextCursor {
            branch_id: None,
            leaf_sequence: None,
        },
    )
}

pub(crate) fn project_context_tree(
    records: &[TranscriptRecord],
) -> anyhow::Result<ContextTreeState> {
    replay_context_tree(records)
}

pub(crate) fn project_context_view(
    records: &[TranscriptRecord],
) -> anyhow::Result<ContextViewProjection> {
    context_view::project_context_view(records)
}

pub(crate) fn replay_context_tree(
    records: &[TranscriptRecord],
) -> anyhow::Result<ContextTreeState> {
    let mut ops = Vec::new();
    let mut saw_context_tree_metadata = false;

    for record in records {
        match &record.event {
            TranscriptEvent::ContextNodeCreated {
                node_id,
                parent_node_id,
                label,
                purpose,
                block_ref,
                source_ref,
            } => {
                saw_context_tree_metadata = true;
                let node_id = ContextNodeId::new(node_id.clone()).with_context(|| {
                    format!(
                        "invalid context node_id at transcript sequence {}",
                        record.sequence
                    )
                })?;
                let parent_node_id = parent_node_id
                    .as_ref()
                    .map(|value| ContextNodeId::new(value.clone()))
                    .transpose()
                    .with_context(|| {
                        format!(
                            "invalid parent context node_id at transcript sequence {}",
                            record.sequence
                        )
                    })?;
                ops.push(ContextTreeOp::CreateNode {
                    node_id,
                    parent_node_id,
                    label: label.clone(),
                    purpose: purpose.clone(),
                    block_ref: block_ref.clone(),
                    source_ref: source_ref.clone(),
                });
            }
            TranscriptEvent::ContextNodeLifecycle { node_id, status } => {
                saw_context_tree_metadata = true;
                ops.push(ContextTreeOp::SetNodeStatus {
                    node_id: ContextNodeId::new(node_id.clone()).with_context(|| {
                        format!(
                            "invalid context node_id at transcript sequence {}",
                            record.sequence
                        )
                    })?,
                    status: status.clone(),
                });
            }
            _ => {}
        }
    }

    if !saw_context_tree_metadata {
        return Ok(ContextTreeState::with_default_root());
    }

    ContextTreeState::replay(&ops)
}

pub(crate) fn list_context_branches(
    records: &[TranscriptRecord],
    current_branch_id: Option<&str>,
) -> anyhow::Result<Vec<ContextBranchInfo>> {
    let index = build_branch_index(records)?;
    let active_branch_id = resolve_active_branch_id(&index, current_branch_id);
    let mut branches = index
        .definitions
        .iter()
        .map(|(branch_id, definition)| {
            Ok(ContextBranchInfo {
                branch_id: branch_id.clone(),
                parent_branch_id: definition.parent_branch_id.clone(),
                label: definition.label.clone(),
                tip_sequence: index.branch_tip(branch_id)?,
                is_current: branch_id == &active_branch_id,
            })
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    branches.sort_by(|left, right| {
        (left.branch_id != ROOT_CONTEXT_BRANCH_ID)
            .cmp(&(right.branch_id != ROOT_CONTEXT_BRANCH_ID))
            .then_with(|| left.branch_id.cmp(&right.branch_id))
    });
    Ok(branches)
}

#[derive(Debug, Clone)]
struct BranchDefinition {
    parent_branch_id: Option<String>,
    base_sequence: u64,
    label: Option<String>,
}

#[derive(Debug, Clone)]
struct CheckoutState {
    branch_id: String,
    leaf_sequence: u64,
}

#[derive(Debug, Default)]
struct BranchIndex {
    definitions: BTreeMap<String, BranchDefinition>,
    latest_checkout: Option<CheckoutState>,
    branch_tips: BTreeMap<String, u64>,
}

#[derive(Debug)]
struct ResolvedBranchContext {
    branch_id: String,
    leaf_sequence: u64,
    scope_checkout_sequence: Option<u64>,
    records: Vec<TranscriptRecord>,
}

pub(crate) fn build_session_context_snapshot(
    session_id: String,
    records: Vec<TranscriptRecord>,
    token_usage: Option<TokenUsageEvent>,
    cursor: SessionContextCursor,
) -> anyhow::Result<SessionRestoreSnapshot> {
    let resolved = resolve_branch_context(records.clone(), cursor)?;
    validate_successful_compactions(&resolved.records)?;
    let history = restore_session_history_projection(&resolved.records);
    let messages = history
        .clone()
        .into_iter()
        .filter_map(super::history_item_to_conversation_message)
        .collect();
    let evidence = crate::evidence::restore_evidence_records(&resolved.records)?;
    let latest_model = restore_latest_model_projection(&resolved.records);
    // Turn IDs are allocated across the append-only session, independently of
    // the branch/leaf used for lifecycle and active-turn restoration.
    let max_turn_id = restore_max_turn_id_projection(&records);

    Ok(SessionRestoreSnapshot {
        session_id,
        branch_id: resolved.branch_id,
        leaf_sequence: resolved.leaf_sequence,
        records: resolved.records,
        messages,
        history,
        evidence,
        latest_model,
        max_turn_id,
        token_usage,
    })
}

pub(crate) fn project_runtime_restore_snapshot(
    session_id: String,
    records: Vec<TranscriptRecord>,
    token_usage: Option<TokenUsageEvent>,
    cursor: SessionContextCursor,
    child_sessions: &[ChildSessionSummary],
) -> anyhow::Result<RuntimeRestoreSnapshot> {
    let resolved = resolve_branch_context(records.clone(), cursor)?;
    validate_successful_compactions(&resolved.records)?;
    let latest_model = restore_latest_model_projection(&resolved.records);
    // Keep allocation global to this session while all active state below stays
    // scoped to the resolved branch and leaf.
    let max_turn_id = restore_max_turn_id_projection(&records);
    let snapshot = runtime_snapshot_from_resolved_context(
        &session_id,
        &records,
        &resolved,
        latest_model.as_deref(),
        child_sessions,
    )?;
    // The runtime projection is the authority for restored protocol identity.
    // Do not rebuild these from compatibility history, which deliberately has
    // no runtime IDs or transcript provenance.
    let protocol_frames = snapshot.active_protocol_frames();

    Ok(RuntimeRestoreSnapshot {
        session_id,
        branch_id: resolved.branch_id,
        leaf_sequence: resolved.leaf_sequence,
        records: resolved.records,
        protocol_frames,
        snapshot,
        latest_model,
        max_turn_id,
        token_usage,
    })
}

pub(crate) fn restore_session_protocol_frames_projection(
    records: &[TranscriptRecord],
) -> Vec<ProtocolFrame> {
    history_items_to_frames(&restore_session_history_projection(records))
}

fn runtime_snapshot_from_resolved_context(
    session_id: &str,
    all_records: &[TranscriptRecord],
    resolved: &ResolvedBranchContext,
    latest_model: Option<&str>,
    child_sessions: &[ChildSessionSummary],
) -> anyhow::Result<RuntimeSnapshot> {
    validate_compaction_binding_checkpoints(session_id, all_records, resolved, child_sessions)?;
    let mut snapshot = runtime_snapshot_from_resolved_context_unbound(
        session_id,
        all_records,
        resolved,
        latest_model,
        child_sessions,
    )?;
    apply_latest_compaction_frame_bindings(&mut snapshot, &resolved.records)?;
    Ok(snapshot)
}

/// Builds the canonical runtime frame projection without durable ID bindings.
/// Checkpoint validation uses this deliberately non-recursive projection.
fn runtime_snapshot_from_resolved_context_unbound(
    session_id: &str,
    all_records: &[TranscriptRecord],
    resolved: &ResolvedBranchContext,
    latest_model: Option<&str>,
    child_sessions: &[ChildSessionSummary],
) -> anyhow::Result<RuntimeSnapshot> {
    let mut snapshot = RuntimeSnapshot::new(resolved.branch_id.clone())
        .with_session_id(session_id.to_string())
        .with_leaf_sequence(resolved.leaf_sequence)
        .with_context_scope_revision(context_scope_revision(all_records, resolved));
    if let Some(latest_model) = latest_model {
        snapshot = snapshot.with_latest_model(latest_model.to_string());
    }
    if let Some(turn_id) = active_turn_id_from_lifecycle_records(&resolved.records) {
        snapshot = snapshot.with_current_turn_id(turn_id);
    }

    let projection_records = runtime_projection_records(all_records, resolved);
    let context_tree = replay_context_tree(&projection_records)?;
    let context_view = project_context_view(&projection_records)?;
    let evidence = crate::evidence::restore_evidence_records(&resolved.records)?;
    let retired_source_spans = restore_retired_source_spans_projection(&resolved.records);

    snapshot.active_context.parent_branch_id = branch_parent_id(all_records, &resolved.branch_id)?;
    snapshot.active_context.active_node_id = context_tree
        .active_node_id()
        .map(|node_id| node_id.as_str().to_string());
    snapshot.active_context.open_detail_block_id = context_view.provider_open_detail_block_id();
    snapshot.active_context.visible_block_ids = context_view.provider_visible_block_ids();
    snapshot.active_context.pinned_block_ids = context_view.provider_pinned_block_ids();
    snapshot.compaction.retired_source_spans = runtime_source_spans(&retired_source_spans)?;
    snapshot.set_context_tree(context_tree.clone());
    snapshot.set_context_view(context_view.clone());
    snapshot.set_evidence(evidence.clone());

    let history_entries = restore_history_projection(&resolved.records);
    let history_frame_ids = append_history_frames(&mut snapshot, &history_entries);
    append_retired_history_frames(&mut snapshot, &resolved.records, &retired_source_spans);
    append_context_frames(&mut snapshot, &context_view)?;
    append_evidence_frames(&mut snapshot, &evidence)?;
    append_summary_artifact_frames(&mut snapshot, &context_view)?;
    append_folded_output_refs(&mut snapshot, &context_view)?;
    append_child_sessions(&mut snapshot, child_sessions)?;
    append_prompt_contributors(&mut snapshot, &context_view, &evidence, child_sessions)?;
    snapshot.compaction.compacted_frame_ids = snapshot
        .frames
        .iter()
        .filter(|frame| frame.visibility == FrameVisibility::Retired)
        .map(|frame| frame.id)
        .collect();
    snapshot.compaction.compacted_frame_ids.sort();
    snapshot.compaction.compacted_frame_ids.dedup();
    snapshot.set_turn_protected_frame_ids(protected_history_frame_ids(
        &history_entries,
        &history_frame_ids,
        snapshot.current_turn_id,
    )?);

    Ok(snapshot)
}

/// A checkout establishes a durable cursor scope even when its branch later
/// advances. Sequence zero keeps old transcripts deterministic.
fn context_scope_revision(_records: &[TranscriptRecord], resolved: &ResolvedBranchContext) -> u64 {
    resolved.scope_checkout_sequence.unwrap_or(0)
}

fn runtime_projection_records(
    all_records: &[TranscriptRecord],
    resolved: &ResolvedBranchContext,
) -> Vec<TranscriptRecord> {
    let allowed_sequences = resolved
        .records
        .iter()
        .map(|record| record.sequence)
        .collect::<BTreeSet<_>>();
    let metadata_frontier = selected_scope_metadata_frontier(all_records, resolved);
    let mut known_nodes = BTreeSet::from(["root".to_string()]);
    all_records
        .iter()
        .filter(|record| {
            if allowed_sequences.contains(&record.sequence) {
                if let TranscriptEvent::ContextNodeCreated { node_id, .. } = &record.event {
                    known_nodes.insert(node_id.clone());
                }
                return true;
            }
            // Historical node lifecycle metadata predates the cursor cut and
            // remains part of the resolved path; scope-frontier filtering is
            // only needed for metadata appended beyond that content leaf.
            if record.sequence <= resolved.leaf_sequence
                && record_belongs_to_selected_scope(all_records, record, resolved)
                && matches!(
                    record.event,
                    TranscriptEvent::ContextNodeCreated { .. }
                        | TranscriptEvent::ContextNodeLifecycle { .. }
                )
            {
                if let TranscriptEvent::ContextNodeCreated { node_id, .. } = &record.event {
                    known_nodes.insert(node_id.clone());
                }
                return true;
            }
            if record.sequence > metadata_frontier
                || !record_belongs_to_selected_scope(all_records, record, resolved)
            {
                return false;
            }
            match &record.event {
                TranscriptEvent::ContextNodeCreated {
                    node_id, block_ref, ..
                } => {
                    let associated = block_ref.as_ref().is_none_or(|block| {
                        block_sequence_from_id(&block.block_id)
                            .is_some_and(|sequence| allowed_sequences.contains(&sequence))
                    });
                    if associated {
                        known_nodes.insert(node_id.clone());
                    }
                    associated
                }
                TranscriptEvent::ContextNodeLifecycle { node_id, .. } => {
                    known_nodes.contains(node_id)
                }
                TranscriptEvent::ContextViewOperationMetadata { block_id, .. } => block_id
                    .as_deref()
                    .and_then(block_sequence_from_id)
                    .is_some_and(|sequence| allowed_sequences.contains(&sequence)),
                TranscriptEvent::ContextSummaryArtifactMetadata {
                    source_start_sequence,
                    source_end_sequence,
                    ..
                }
                | TranscriptEvent::FoldedOutputMetadata {
                    source_start_sequence,
                    source_end_sequence,
                    ..
                } => span_intersects_allowed_sequences(
                    *source_start_sequence,
                    *source_end_sequence,
                    &allowed_sequences,
                ),
                _ => false,
            }
        })
        .cloned()
        .collect()
}

/// Metadata is append-only and normally has no branch id. Its ownership is
/// therefore bounded by the checkout that selected the active scope and the
/// next checkout that replaces it.
fn selected_scope_metadata_frontier(
    records: &[TranscriptRecord],
    resolved: &ResolvedBranchContext,
) -> u64 {
    let Some(scope_start) = resolved.scope_checkout_sequence else {
        return resolved.leaf_sequence;
    };
    records
        .iter()
        .filter(|record| record.sequence > scope_start)
        .find_map(|record| {
            matches!(record.event, TranscriptEvent::ContextCheckout { .. })
                .then_some(record.sequence - 1)
        })
        .unwrap_or_else(|| records.last().map(|record| record.sequence).unwrap_or(0))
}

fn record_belongs_to_selected_scope(
    all_records: &[TranscriptRecord],
    record: &TranscriptRecord,
    resolved: &ResolvedBranchContext,
) -> bool {
    if let Some(branch_id) = record.context_branch_id.as_deref() {
        return branch_id == resolved.branch_id;
    }

    // Context metadata has no branch id. Its owner is the branch selected by
    // the preceding checkout; records before any checkout remain shared legacy
    // metadata. This prevents a sibling's post-fork node/view updates from
    // leaking into the selected cursor.
    all_records
        .iter()
        .rev()
        .filter(|candidate| candidate.sequence <= record.sequence)
        .find_map(|candidate| match &candidate.event {
            TranscriptEvent::ContextCheckout { branch_id, .. } => Some(branch_id.as_str()),
            _ => None,
        })
        .is_none_or(|branch_id| branch_id == resolved.branch_id)
}

fn block_sequence_from_id(block_id: &str) -> Option<u64> {
    block_id
        .strip_prefix("block-seq-")?
        .split_once('-')?
        .0
        .parse()
        .ok()
}

fn span_intersects_allowed_sequences(
    start_sequence: Option<u64>,
    end_sequence: Option<u64>,
    allowed_sequences: &BTreeSet<u64>,
) -> bool {
    match (start_sequence, end_sequence) {
        (Some(start), Some(end)) => {
            (start..=end).any(|sequence| allowed_sequences.contains(&sequence))
        }
        (Some(start), None) => allowed_sequences.contains(&start),
        (None, Some(end)) => allowed_sequences.contains(&end),
        (None, None) => false,
    }
}

fn branch_parent_id(
    records: &[TranscriptRecord],
    branch_id: &str,
) -> anyhow::Result<Option<String>> {
    let index = build_branch_index(records)?;
    Ok(index
        .definitions
        .get(branch_id)
        .and_then(|definition| definition.parent_branch_id.clone()))
}

fn runtime_source_spans(spans: &[ContextCompactionSourceSpan]) -> anyhow::Result<Vec<SourceSpan>> {
    spans
        .iter()
        .map(|span| SourceSpan::new(span.start_sequence, span.end_sequence))
        .collect()
}

fn append_history_frames(
    snapshot: &mut RuntimeSnapshot,
    entries: &[HistoryProjectionEntry],
) -> Vec<RuntimeFrameId> {
    let mut frame_ids = Vec::with_capacity(entries.len());
    for (ordinal, entry) in entries.iter().enumerate() {
        let Some((kind, stable_key, summary)) = history_entry_frame_parts(&entry.item) else {
            continue;
        };
        // A compaction summary is a newly-created artifact, not a replayed raw
        // transcript frame. Its retired raw sources stay on the compaction
        // state and retired frames only, matching live compaction provenance.
        let (source, source_span) = if matches!(entry.item, HistoryItem::ContextSummary { .. }) {
            (RuntimeSource::SummaryArtifact, None)
        } else {
            (
                RuntimeSource::Transcript,
                merged_runtime_source_span(&entry.source_spans),
            )
        };
        let provenance = runtime_provenance(source, source_span, None);
        let frame = RuntimeFrame::new(
            kind,
            FrameVisibility::Active,
            provenance.clone(),
            RuntimeFrameIdSeed {
                frame_kind: kind,
                source,
                ordinal: ordinal as u32,
                stable_key: &stable_key,
                source_span,
            },
        )
        .with_summary(summary)
        .with_protocol(crate::agent::protocol_frame_item_from_history_item(
            &entry.item,
        ));
        frame_ids.push(frame.id);
        snapshot.push_frame(frame);
    }
    frame_ids
}

/// Reconstruct the pre-compaction protocol frames from the append-only journal.
/// The active history projection intentionally replaces these frames with a
/// summary; RuntimeSnapshot retains them as retired identity/provenance records.
fn append_retired_history_frames(
    snapshot: &mut RuntimeSnapshot,
    records: &[TranscriptRecord],
    retired_spans: &[ContextCompactionSourceSpan],
) {
    let raw_records = records
        .iter()
        .filter(|record| !matches!(record.event, TranscriptEvent::ContextCompaction(_)))
        .cloned()
        .collect::<Vec<_>>();
    for (ordinal, entry) in restore_history_projection(&raw_records).iter().enumerate() {
        if entry.source_spans.is_empty()
            || !entry.source_spans.iter().all(|source| {
                retired_spans.iter().any(|retired| {
                    retired.start_sequence <= source.start_sequence
                        && source.end_sequence <= retired.end_sequence
                })
            })
        {
            continue;
        }
        let Some((kind, stable_key, summary)) = history_entry_frame_parts(&entry.item) else {
            continue;
        };
        let source_span = merged_runtime_source_span(&entry.source_spans);
        let provenance = runtime_provenance(RuntimeSource::Transcript, source_span, None);
        let frame = RuntimeFrame::new(
            kind,
            FrameVisibility::Retired,
            provenance,
            RuntimeFrameIdSeed {
                frame_kind: kind,
                source: RuntimeSource::Transcript,
                ordinal: ordinal as u32,
                stable_key: &stable_key,
                source_span,
            },
        )
        .with_summary(summary)
        .with_protocol(crate::agent::protocol_frame_item_from_history_item(
            &entry.item,
        ));
        if !snapshot
            .frames
            .iter()
            .any(|existing| existing.id == frame.id)
        {
            snapshot.push_frame(frame);
        }
    }
}

fn append_context_frames(
    snapshot: &mut RuntimeSnapshot,
    context_view: &ContextViewProjection,
) -> anyhow::Result<()> {
    // Allocate IDs in the context view's canonical projection order. Visibility
    // is state, not identity: partitioning active and retired blocks before this
    // point would renumber an active block when an earlier block is compacted.
    for (ordinal, (block_id, block)) in context_view.all_context_blocks().into_iter().enumerate() {
        let source_span = context_block_source_span(block)?;
        let provenance = runtime_provenance(
            RuntimeSource::ContextView,
            source_span,
            Some(block.block_id.as_str().to_string()),
        );
        snapshot.push_frame(
            RuntimeFrame::new(
                RuntimeFrameKind::ContextBlock,
                if context_view.is_compacted(block_id) {
                    FrameVisibility::Retired
                } else if context_view.is_default_active(block_id) {
                    FrameVisibility::Active
                } else {
                    FrameVisibility::Folded
                },
                provenance,
                RuntimeFrameIdSeed {
                    frame_kind: RuntimeFrameKind::ContextBlock,
                    source: RuntimeSource::ContextView,
                    ordinal: ordinal as u32,
                    stable_key: block.block_id.as_str(),
                    source_span,
                },
            )
            .with_summary(format!("{}: {}", block.title, block.detail)),
        );
    }
    Ok(())
}

/// Applies the newest durable identity map after canonical frame construction
/// and before any references are derived from those IDs.
fn apply_latest_compaction_frame_bindings(
    snapshot: &mut RuntimeSnapshot,
    records: &[TranscriptRecord],
) -> anyhow::Result<()> {
    let events = records
        .iter()
        .filter_map(|record| match &record.event {
            TranscriptEvent::ContextCompaction(event)
                if event.outcome == "succeeded" && !event.frame_identity_bindings.is_empty() =>
            {
                Some(event)
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    if events.is_empty() {
        return Ok(());
    }

    let bindings = &events
        .last()
        .expect("nonempty events")
        .frame_identity_bindings;
    apply_frame_identity_bindings(snapshot, bindings)?;
    snapshot.validate_references()
}

/// Validate each modern cumulative map against the exact runtime frame set at
/// that event.  This must happen before the final projection is bound: a later
/// map can legitimately rebind a retained frame, but cannot repair an earlier
/// collision with that frame's then-current canonical ID.
fn validate_compaction_binding_checkpoints(
    session_id: &str,
    all_records: &[TranscriptRecord],
    resolved: &ResolvedBranchContext,
    child_sessions: &[ChildSessionSummary],
) -> anyhow::Result<()> {
    let mut bound_ids = BTreeMap::new();
    let mut bound_keys = BTreeMap::new();

    for (index, record) in resolved.records.iter().enumerate() {
        let TranscriptEvent::ContextCompaction(event) = &record.event else {
            continue;
        };
        if event.outcome != "succeeded" || event.frame_identity_bindings.is_empty() {
            continue;
        }

        let checkpoint_records = resolved.records[..=index].to_vec();
        let checkpoint = ResolvedBranchContext {
            branch_id: resolved.branch_id.clone(),
            leaf_sequence: record.sequence,
            scope_checkout_sequence: resolved.scope_checkout_sequence,
            records: checkpoint_records,
        };
        let checkpoint_all_records = all_records
            .iter()
            .filter(|candidate| candidate.sequence <= record.sequence)
            .cloned()
            .collect::<Vec<_>>();
        let latest_model = restore_latest_model_projection(&checkpoint.records);
        let mut snapshot = runtime_snapshot_from_resolved_context_unbound(
            session_id,
            &checkpoint_all_records,
            &checkpoint,
            latest_model.as_deref(),
            child_sessions,
        )?;

        apply_cumulative_frame_identity_bindings(&mut snapshot, &bound_keys)?;
        snapshot.validate_references()?;
        validate_frame_identity_bindings(&snapshot, event, &bound_ids, &bound_keys)?;

        let mut candidate = snapshot.clone();
        apply_frame_identity_bindings(&mut candidate, &event.frame_identity_bindings)?;
        candidate.validate_references()?;

        for binding in &event.frame_identity_bindings {
            bound_ids.insert(binding.frame_id, binding.key.clone());
            bound_keys.insert(binding.key.clone(), binding.frame_id);
        }
    }
    Ok(())
}

fn apply_cumulative_frame_identity_bindings(
    snapshot: &mut RuntimeSnapshot,
    bindings: &BTreeMap<String, u64>,
) -> anyhow::Result<()> {
    let bindings = bindings
        .iter()
        .map(|(key, frame_id)| ContextCompactionFrameBinding {
            key: key.clone(),
            frame_id: *frame_id,
        })
        .collect::<Vec<_>>();
    apply_frame_identity_bindings(snapshot, &bindings)
}

fn apply_frame_identity_bindings(
    snapshot: &mut RuntimeSnapshot,
    bindings: &[ContextCompactionFrameBinding],
) -> anyhow::Result<()> {
    let mut remapped_ids = BTreeMap::new();
    for binding in bindings {
        let mut matches = snapshot
            .frames
            .iter_mut()
            .filter(|frame| frame.durable_identity_key() == binding.key)
            .collect::<Vec<_>>();
        ensure!(
            matches.len() == 1,
            "context compaction frame identity binding key must resolve exactly once"
        );
        let frame = matches.pop().expect("checked exactly one matching frame");
        let frame_id = RuntimeFrameId::from_persisted(binding.frame_id);
        remapped_ids.insert(frame.id, frame_id);
        frame.id = frame_id;
    }
    remap_runtime_frame_references(snapshot, &remapped_ids);
    Ok(())
}

fn remap_runtime_frame_references(
    snapshot: &mut RuntimeSnapshot,
    remapped_ids: &BTreeMap<RuntimeFrameId, RuntimeFrameId>,
) {
    let remap = |id: &mut RuntimeFrameId| {
        if let Some(mapped) = remapped_ids.get(id) {
            *id = *mapped;
        }
    };
    for id in snapshot
        .compaction
        .protected_frame_ids
        .iter_mut()
        .chain(snapshot.compaction.explicit_protected_frame_ids.iter_mut())
        .chain(snapshot.compaction.turn_protected_frame_ids.iter_mut())
        .chain(snapshot.compaction.compacted_frame_ids.iter_mut())
        .chain(
            snapshot
                .prompt_contributors
                .iter_mut()
                .flat_map(|contributor| {
                    contributor
                        .frame_ids
                        .iter_mut()
                        .chain(contributor.source_frame_ids.iter_mut())
                }),
        )
    {
        remap(id);
    }
}

fn validate_frame_identity_bindings(
    snapshot: &RuntimeSnapshot,
    event: &crate::agent::ContextCompactionEvent,
    bound_ids: &BTreeMap<u64, String>,
    bound_keys: &BTreeMap<String, u64>,
) -> anyhow::Result<()> {
    if event.frame_identity_bindings.is_empty() {
        return Ok(());
    }
    let mut keys = BTreeSet::new();
    let mut ids = BTreeSet::new();
    for binding in &event.frame_identity_bindings {
        ensure!(
            !binding.key.is_empty() && keys.insert(binding.key.clone()),
            "context compaction frame identity bindings contain a duplicate or empty key"
        );
        ensure!(
            ids.insert(binding.frame_id),
            "context compaction frame identity bindings contain a duplicate frame id"
        );
        if let Some(existing_id) = bound_keys.get(&binding.key) {
            ensure!(
                *existing_id == binding.frame_id,
                "context compaction frame identity binding key has a conflicting frame id"
            );
        }
        if let Some(existing_key) = bound_ids.get(&binding.frame_id) {
            ensure!(
                existing_key == &binding.key,
                "context compaction frame identity binding frame id has a conflicting key"
            );
        }
        let matches = snapshot
            .frames
            .iter()
            .filter(|frame| frame.durable_identity_key() == binding.key)
            .collect::<Vec<_>>();
        ensure!(
            matches.len() == 1,
            "context compaction frame identity binding key must resolve exactly once"
        );
        let frame = matches[0];
        if frame.visibility == FrameVisibility::Retired {
            ensure!(
                frame.protocol.is_some(),
                "retired identity binding must target a protocol frame"
            );
            let span = frame
                .provenance
                .source_span
                .ok_or_else(|| anyhow!("retired identity binding must target a spanned frame"))?;
            ensure!(
                event.retired_source_spans.iter().any(|retired| {
                    retired.start_sequence <= span.start_sequence
                        && span.end_sequence <= retired.end_sequence
                }),
                "retired identity binding is outside this event's cumulative retired source spans"
            );
        } else {
            ensure!(
                frame.visibility == FrameVisibility::Active
                    && frame.kind == RuntimeFrameKind::Summary
                    && frame.provenance.source == RuntimeSource::SummaryArtifact
                    && frame.provenance.source_span.is_none()
                    && matches!(
                        frame.protocol,
                        Some(crate::protocol_frames::ProtocolFrameItem::ContextSummary { .. })
                    ),
                "summary identity binding must target the active span-less context summary artifact"
            );
        }
    }
    let expected = snapshot
        .frames
        .iter()
        .filter(|frame| {
            (frame.visibility == FrameVisibility::Retired
                && frame.protocol.is_some()
                && frame.provenance.source_span.is_some_and(|span| {
                    event.retired_source_spans.iter().any(|retired| {
                        retired.start_sequence <= span.start_sequence
                            && span.end_sequence <= retired.end_sequence
                    })
                }))
                || (frame.visibility == FrameVisibility::Active
                    && frame.kind == RuntimeFrameKind::Summary
                    && frame.provenance.source == RuntimeSource::SummaryArtifact
                    && frame.provenance.source_span.is_none()
                    && matches!(
                        frame.protocol,
                        Some(crate::protocol_frames::ProtocolFrameItem::ContextSummary { .. })
                    ))
        })
        .map(RuntimeFrame::durable_identity_key)
        .collect::<Vec<_>>();
    let active_summaries = snapshot
        .frames
        .iter()
        .filter(|frame| {
            frame.visibility == FrameVisibility::Active
                && frame.kind == RuntimeFrameKind::Summary
                && frame.provenance.source == RuntimeSource::SummaryArtifact
                && frame.provenance.source_span.is_none()
                && matches!(
                    frame.protocol,
                    Some(crate::protocol_frames::ProtocolFrameItem::ContextSummary { .. })
                )
        })
        .count();
    ensure!(
        active_summaries == 1,
        "context compaction frame identity bindings require exactly one active span-less context summary"
    );
    let expected_keys = expected.iter().cloned().collect::<BTreeSet<_>>();
    ensure!(
        expected.len() == expected_keys.len(),
        "context compaction frame identity bindings have colliding durable keys"
    );
    ensure!(
        keys == expected_keys,
        "context compaction frame identity bindings must cover all retired protocol frames and the active summary"
    );
    Ok(())
}

pub(crate) fn compaction_frame_identity_bindings(
    snapshot: &RuntimeSnapshot,
) -> Vec<ContextCompactionFrameBinding> {
    snapshot
        .frames
        .iter()
        .filter(|frame| {
            (frame.visibility == FrameVisibility::Retired && frame.protocol.is_some())
                || (frame.visibility == FrameVisibility::Active
                    && frame.kind == RuntimeFrameKind::Summary
                    && frame.provenance.source == RuntimeSource::SummaryArtifact
                    && frame.provenance.source_span.is_none()
                    && matches!(
                        frame.protocol,
                        Some(crate::protocol_frames::ProtocolFrameItem::ContextSummary { .. })
                    ))
        })
        .map(|frame| ContextCompactionFrameBinding {
            key: frame.durable_identity_key(),
            frame_id: frame.id.as_u64(),
        })
        .collect()
}

pub(crate) fn validate_successful_compactions(records: &[TranscriptRecord]) -> anyhow::Result<()> {
    for (index, record) in records.iter().enumerate() {
        if let TranscriptEvent::ContextCompaction(event) = &record.event
            && event.outcome == "succeeded"
        {
            validate_context_compaction_event(&records[..index], event)?;
        }
    }
    Ok(())
}

fn append_evidence_frames(
    snapshot: &mut RuntimeSnapshot,
    evidence: &[EvidenceRecord],
) -> anyhow::Result<()> {
    for (ordinal, record) in evidence.iter().enumerate() {
        let source_span = Some(SourceSpan::new(record.sequence, record.sequence)?);
        let provenance = runtime_provenance(
            RuntimeSource::Transcript,
            source_span,
            Some(record.id.clone()),
        );
        snapshot.push_frame(
            RuntimeFrame::new(
                RuntimeFrameKind::Metadata,
                FrameVisibility::Active,
                provenance,
                RuntimeFrameIdSeed {
                    frame_kind: RuntimeFrameKind::Metadata,
                    source: RuntimeSource::Transcript,
                    ordinal: ordinal as u32,
                    stable_key: &record.id,
                    source_span,
                },
            )
            .with_summary(format!("evidence {}: {}", record.id, record.summary)),
        );
    }
    Ok(())
}

fn append_summary_artifact_frames(
    snapshot: &mut RuntimeSnapshot,
    context_view: &ContextViewProjection,
) -> anyhow::Result<()> {
    for (ordinal, artifact) in context_view.summary_artifacts.iter().enumerate() {
        let source_span = match (artifact.source_start_sequence, artifact.source_end_sequence) {
            (Some(start), Some(end)) => Some(SourceSpan::new(start, end)?),
            (Some(start), None) => Some(SourceSpan::new(start, start)?),
            _ => None,
        };
        let provenance = runtime_provenance(
            RuntimeSource::SummaryArtifact,
            source_span,
            Some(artifact.artifact_id.clone()),
        );
        snapshot.push_frame(
            RuntimeFrame::new(
                RuntimeFrameKind::Summary,
                FrameVisibility::Active,
                provenance,
                RuntimeFrameIdSeed {
                    frame_kind: RuntimeFrameKind::Summary,
                    source: RuntimeSource::SummaryArtifact,
                    ordinal: ordinal as u32,
                    stable_key: &artifact.artifact_id,
                    source_span,
                },
            )
            .with_summary(artifact.summary.clone()),
        );
    }
    Ok(())
}

fn append_folded_output_refs(
    snapshot: &mut RuntimeSnapshot,
    context_view: &ContextViewProjection,
) -> anyhow::Result<()> {
    // Folded output IDs use the same stable source order as their owning
    // context blocks. Do not move compacted outputs behind active ones.
    for (ordinal, metadata) in context_view.all_folded_outputs().into_iter().enumerate() {
        let visibility = if context_view.is_compacted_folded_output(&metadata.output_id) {
            FrameVisibility::Retired
        } else {
            FrameVisibility::Folded
        };
        let source_span = match (metadata.source_start_sequence, metadata.source_end_sequence) {
            (Some(start), Some(end)) => Some(SourceSpan::new(start, end)?),
            (Some(start), None) => Some(SourceSpan::new(start, start)?),
            _ => None,
        };
        snapshot.push_folded_output(FoldedOutputReference {
            output_id: metadata.output_id.clone(),
            node_id: metadata.node_id.clone(),
            call_id: metadata.call_id.clone(),
            tool_name: metadata.tool_name.clone(),
            source_span,
        });
        snapshot.push_frame(
            RuntimeFrame::new(
                RuntimeFrameKind::FoldedOutput,
                visibility,
                runtime_provenance(
                    RuntimeSource::FoldedOutput,
                    source_span,
                    Some(metadata.output_id.clone()),
                ),
                RuntimeFrameIdSeed {
                    frame_kind: RuntimeFrameKind::FoldedOutput,
                    source: RuntimeSource::FoldedOutput,
                    ordinal: ordinal as u32,
                    stable_key: &metadata.output_id,
                    source_span,
                },
            )
            .with_summary(format!("folded output {}", metadata.output_id)),
        );
    }
    Ok(())
}

fn append_child_sessions(
    snapshot: &mut RuntimeSnapshot,
    child_sessions: &[ChildSessionSummary],
) -> anyhow::Result<()> {
    for (ordinal, child) in child_sessions.iter().enumerate() {
        snapshot.push_child_session(RuntimeChildSession {
            parent_run_id: child.parent_run_id.clone(),
            child_session_id: child.child_session_id.clone(),
            agent_name: child.agent_name.clone(),
            status: child.status.clone(),
            summary: child.summary.clone(),
            timestamp_ms: child.timestamp_ms,
        });
        snapshot.push_frame(
            RuntimeFrame::new(
                RuntimeFrameKind::Metadata,
                FrameVisibility::Active,
                runtime_provenance(
                    RuntimeSource::SessionState,
                    None,
                    Some(child.child_session_id.clone()),
                ),
                RuntimeFrameIdSeed {
                    frame_kind: RuntimeFrameKind::Metadata,
                    source: RuntimeSource::SessionState,
                    ordinal: ordinal as u32,
                    stable_key: &child.child_session_id,
                    source_span: None,
                },
            )
            .with_summary(format!(
                "child session {} ({}) — {}",
                child.agent_name, child.status, child.summary
            )),
        );
    }
    Ok(())
}

fn append_prompt_contributors(
    snapshot: &mut RuntimeSnapshot,
    context_view: &ContextViewProjection,
    evidence: &[EvidenceRecord],
    child_sessions: &[ChildSessionSummary],
) -> anyhow::Result<()> {
    if !snapshot.active_context.visible_block_ids.is_empty() {
        snapshot.push_prompt_contributor(PromptContributorPlaceholder {
            contributor_id: "context-view-active".into(),
            kind: PromptContributorKind::ContextMaterial,
            label: Some("Active context view".into()),
            provenance: RuntimeFrameProvenance::new(RuntimeSource::ContextView),
            frame_ids: snapshot
                .frames
                .iter()
                .filter(|frame| frame.kind == RuntimeFrameKind::ContextBlock)
                .map(|frame| frame.id)
                .collect(),
            source_frame_ids: Vec::new(),
        });
    }
    if !evidence.is_empty() {
        snapshot.push_prompt_contributor(PromptContributorPlaceholder {
            contributor_id: "evidence".into(),
            kind: PromptContributorKind::Evidence,
            label: Some("Evidence".into()),
            provenance: RuntimeFrameProvenance::new(RuntimeSource::Transcript),
            frame_ids: snapshot
                .frames
                .iter()
                .filter(|frame| {
                    frame
                        .summary
                        .as_deref()
                        .is_some_and(|summary| summary.starts_with("evidence "))
                })
                .map(|frame| frame.id)
                .collect(),
            source_frame_ids: Vec::new(),
        });
    }
    if !context_view.summary_artifacts.is_empty() {
        snapshot.push_prompt_contributor(PromptContributorPlaceholder {
            contributor_id: "summary-artifacts".into(),
            kind: PromptContributorKind::ContextMaterial,
            label: Some("Summary artifacts".into()),
            provenance: RuntimeFrameProvenance::new(RuntimeSource::SummaryArtifact),
            frame_ids: snapshot
                .frames
                .iter()
                .filter(|frame| frame.kind == RuntimeFrameKind::Summary)
                .map(|frame| frame.id)
                .collect(),
            source_frame_ids: Vec::new(),
        });
    }
    if !snapshot.folded_outputs.is_empty() {
        snapshot.push_prompt_contributor(PromptContributorPlaceholder {
            contributor_id: "folded-outputs".into(),
            kind: PromptContributorKind::FoldedOutputSummary,
            label: Some("Folded outputs".into()),
            provenance: RuntimeFrameProvenance::new(RuntimeSource::FoldedOutput),
            frame_ids: snapshot
                .frames
                .iter()
                .filter(|frame| frame.kind == RuntimeFrameKind::FoldedOutput)
                .map(|frame| frame.id)
                .collect(),
            source_frame_ids: Vec::new(),
        });
    }
    if !child_sessions.is_empty() {
        snapshot.push_prompt_contributor(PromptContributorPlaceholder {
            contributor_id: "child-sessions".into(),
            kind: PromptContributorKind::Other,
            label: Some("Child sessions".into()),
            provenance: RuntimeFrameProvenance::new(RuntimeSource::SessionState),
            frame_ids: snapshot
                .frames
                .iter()
                .filter(|frame| {
                    frame.provenance.source == RuntimeSource::SessionState
                        && frame.kind == RuntimeFrameKind::Metadata
                })
                .map(|frame| frame.id)
                .collect(),
            source_frame_ids: Vec::new(),
        });
    }
    Ok(())
}

fn history_entry_frame_parts(item: &HistoryItem) -> Option<(RuntimeFrameKind, String, String)> {
    match item {
        HistoryItem::ContextSummary { text } => Some((
            RuntimeFrameKind::Summary,
            format!("context-summary:{text}"),
            text.clone(),
        )),
        HistoryItem::UserMessage { content } => Some((
            RuntimeFrameKind::User,
            format!("user:{}", content.display_text()),
            content.display_text(),
        )),
        HistoryItem::InternalContinuation { text } => Some((
            RuntimeFrameKind::Metadata,
            format!("internal-continuation:{text}"),
            text.clone(),
        )),
        HistoryItem::AssistantText { text } => Some((
            RuntimeFrameKind::Assistant,
            format!("assistant:{text}"),
            text.clone(),
        )),
        HistoryItem::AssistantToolCalls { text, calls } => {
            let stable_key = calls
                .iter()
                .map(|call| format!("{}:{}:{}", call.call_id, call.name, call.arguments_json))
                .collect::<Vec<_>>()
                .join("|");
            let summary = text.clone().unwrap_or_else(|| {
                calls
                    .iter()
                    .map(|call| format!("{}({})", call.name, call.call_id))
                    .collect::<Vec<_>>()
                    .join(", ")
            });
            Some((RuntimeFrameKind::ToolCall, stable_key, summary))
        }
        HistoryItem::ToolOutput {
            call_id,
            output_json,
        } => Some((
            RuntimeFrameKind::ToolOutput,
            format!("tool-output:{call_id}:{output_json}"),
            format!("tool output {call_id}"),
        )),
    }
}

fn merged_runtime_source_span(spans: &[ContextCompactionSourceSpan]) -> Option<SourceSpan> {
    let first = spans.first()?;
    let end_sequence = spans
        .iter()
        .map(|span| span.end_sequence)
        .max()
        .unwrap_or(first.end_sequence);
    SourceSpan::new(first.start_sequence, end_sequence).ok()
}

fn runtime_provenance(
    source: RuntimeSource,
    source_span: Option<SourceSpan>,
    source_id: Option<String>,
) -> RuntimeFrameProvenance {
    let mut provenance = RuntimeFrameProvenance::new(source);
    if let Some(source_span) = source_span {
        provenance = provenance.with_span(source_span);
    }
    if let Some(source_id) = source_id {
        provenance = provenance.with_source_id(source_id);
    }
    provenance
}

fn context_block_source_span(
    block: &crate::context_view::ContextBlock,
) -> anyhow::Result<Option<SourceSpan>> {
    match &block.source {
        crate::context_view::ContextBlockSource::TranscriptSpan {
            start_sequence,
            end_sequence,
        } => Ok(Some(SourceSpan::new(*start_sequence, *end_sequence)?)),
        _ => Ok(block
            .source_start_sequence
            .map(|sequence| SourceSpan::new(sequence, sequence))
            .transpose()?),
    }
}

fn resolve_branch_context(
    records: Vec<TranscriptRecord>,
    cursor: SessionContextCursor,
) -> anyhow::Result<ResolvedBranchContext> {
    let index = build_branch_index(&records)?;
    let branch_id = cursor
        .branch_id
        .unwrap_or_else(|| resolve_active_branch_id(&index, None));
    let leaf_sequence = match cursor.leaf_sequence {
        Some(leaf_sequence) => leaf_sequence,
        // A checkout chooses the active branch/scope, not a permanently frozen
        // content leaf. Explicit cursors are the sole way to request a cut.
        None => index.branch_tip(&branch_id)?,
    };

    let max_sequence = records
        .iter()
        .map(|record| record.sequence)
        .max()
        .unwrap_or(0);
    ensure!(
        leaf_sequence <= max_sequence || (leaf_sequence == 0 && max_sequence == 0),
        "session context leaf_sequence {leaf_sequence} exceeds max transcript sequence {max_sequence}"
    );

    // An explicit cursor retains the latest scope interval for its branch even
    // when a later checkout selected a different branch.
    let scope_checkout_sequence = records.iter().rev().find_map(|record| {
        matches!(
            &record.event,
            TranscriptEvent::ContextCheckout {
                branch_id: checkout_branch_id,
                ..
            } if checkout_branch_id == &branch_id
        )
        .then_some(record.sequence)
    });
    let records = collect_branch_path_records(&records, &index, &branch_id, leaf_sequence)?;
    Ok(ResolvedBranchContext {
        branch_id,
        leaf_sequence,
        scope_checkout_sequence,
        records,
    })
}

fn build_branch_index(records: &[TranscriptRecord]) -> anyhow::Result<BranchIndex> {
    let mut index = BranchIndex::default();
    index.definitions.insert(
        ROOT_CONTEXT_BRANCH_ID.to_string(),
        BranchDefinition {
            parent_branch_id: None,
            base_sequence: 0,
            label: None,
        },
    );
    index
        .branch_tips
        .insert(ROOT_CONTEXT_BRANCH_ID.to_string(), 0);

    for (position, record) in records.iter().enumerate() {
        match &record.event {
            TranscriptEvent::ContextBranchCreated {
                branch_id,
                parent_branch_id,
                base_sequence,
                label,
            } => {
                ensure!(
                    !index.definitions.contains_key(branch_id),
                    "duplicate context branch_id '{branch_id}'"
                );
                ensure!(
                    index.definitions.contains_key(parent_branch_id),
                    "missing parent context branch '{parent_branch_id}' for branch '{branch_id}'"
                );
                ensure!(
                    base_sequence_resolves_on_parent_path(
                        &records[..position],
                        &index,
                        parent_branch_id,
                        *base_sequence,
                    )?,
                    "context branch '{branch_id}' base_sequence {base_sequence} is not resolvable on parent branch '{parent_branch_id}'"
                );
                index.definitions.insert(
                    branch_id.clone(),
                    BranchDefinition {
                        parent_branch_id: Some(parent_branch_id.clone()),
                        base_sequence: *base_sequence,
                        label: label.clone(),
                    },
                );
                index.branch_tips.insert(branch_id.clone(), *base_sequence);
            }
            TranscriptEvent::ContextCheckout {
                branch_id,
                leaf_sequence,
            } => {
                ensure!(
                    index.definitions.contains_key(branch_id),
                    "unknown context branch '{branch_id}' in checkout metadata"
                );
                index.latest_checkout = Some(CheckoutState {
                    branch_id: branch_id.clone(),
                    leaf_sequence: *leaf_sequence,
                });
            }
            TranscriptEvent::ContextBranchSummary {
                branch_id,
                leaf_sequence,
                ..
            } => {
                ensure!(
                    index.definitions.contains_key(branch_id),
                    "unknown context branch '{branch_id}' in branch summary metadata"
                );
                let branch_tip = branch_tip_for_records(records, &index, branch_id)?;
                ensure!(
                    *leaf_sequence <= branch_tip,
                    "context branch summary leaf_sequence {leaf_sequence} exceeds tip {branch_tip} for branch '{branch_id}'"
                );
            }
            _ => {
                if record.event.is_context_branch_metadata() {
                    continue;
                }
                let effective_branch_id = effective_branch_id(record);
                ensure!(
                    index.definitions.contains_key(effective_branch_id),
                    "unknown context branch '{effective_branch_id}' in record scope at sequence {}",
                    record.sequence
                );
                index.branch_tips.insert(
                    effective_branch_id.to_string(),
                    branch_tip_for_records(records, &index, effective_branch_id)?,
                );
            }
        }
    }

    if let Some(checkout) = &index.latest_checkout {
        let branch_tip = index.branch_tip(&checkout.branch_id)?;
        ensure!(
            checkout.leaf_sequence <= branch_tip,
            "context checkout leaf_sequence {} exceeds tip {} for branch '{}'",
            checkout.leaf_sequence,
            branch_tip,
            checkout.branch_id
        );
    }

    Ok(index)
}

fn base_sequence_resolves_on_parent_path(
    records: &[TranscriptRecord],
    index: &BranchIndex,
    parent_branch_id: &str,
    base_sequence: u64,
) -> anyhow::Result<bool> {
    if base_sequence == 0 {
        return Ok(true);
    }

    if base_sequence > branch_tip_for_records(records, index, parent_branch_id)? {
        return Ok(false);
    }

    let path = collect_branch_path_records(records, index, parent_branch_id, base_sequence)?;
    Ok(path.iter().any(|record| record.sequence == base_sequence))
}

fn branch_tip_for_records(
    records: &[TranscriptRecord],
    index: &BranchIndex,
    branch_id: &str,
) -> anyhow::Result<u64> {
    let definition = index
        .definitions
        .get(branch_id)
        .ok_or_else(|| anyhow!("unknown context branch '{branch_id}'"))?;
    let local_tip = records
        .iter()
        .filter(|record| !record.event.is_context_branch_metadata())
        .filter(|record| effective_branch_id(record) == branch_id)
        .map(|record| record.sequence)
        .max()
        .unwrap_or(definition.base_sequence);
    Ok(local_tip.max(definition.base_sequence))
}

fn collect_branch_path_records(
    records: &[TranscriptRecord],
    index: &BranchIndex,
    branch_id: &str,
    leaf_sequence: u64,
) -> anyhow::Result<Vec<TranscriptRecord>> {
    let branch_tip = index.branch_tip(branch_id)?;
    ensure!(
        leaf_sequence <= branch_tip,
        "requested leaf_sequence {leaf_sequence} exceeds tip {branch_tip} for branch '{branch_id}'"
    );

    if branch_id == ROOT_CONTEXT_BRANCH_ID {
        return Ok(records
            .iter()
            .filter(|record| !record.event.is_context_branch_metadata())
            .filter(|record| effective_branch_id(record) == ROOT_CONTEXT_BRANCH_ID)
            .filter(|record| record.sequence <= leaf_sequence)
            .cloned()
            .collect());
    }

    let definition = index
        .definitions
        .get(branch_id)
        .ok_or_else(|| anyhow!("unknown context branch '{branch_id}'"))?;
    let parent_branch_id = definition
        .parent_branch_id
        .as_deref()
        .ok_or_else(|| anyhow!("context branch '{branch_id}' is missing a parent"))?;
    ensure!(
        leaf_sequence >= definition.base_sequence,
        "requested leaf_sequence {leaf_sequence} precedes base_sequence {} for branch '{branch_id}'",
        definition.base_sequence
    );

    let mut path =
        collect_branch_path_records(records, index, parent_branch_id, definition.base_sequence)?;
    path.extend(
        records
            .iter()
            .filter(|record| !record.event.is_context_branch_metadata())
            .filter(|record| effective_branch_id(record) == branch_id)
            .filter(|record| record.sequence <= leaf_sequence)
            .cloned(),
    );
    Ok(path)
}

fn effective_branch_id(record: &TranscriptRecord) -> &str {
    record
        .context_branch_id
        .as_deref()
        .unwrap_or(ROOT_CONTEXT_BRANCH_ID)
}

fn resolve_active_branch_id(index: &BranchIndex, current_branch_id: Option<&str>) -> String {
    current_branch_id
        .map(str::to_string)
        .or_else(|| {
            index
                .latest_checkout
                .as_ref()
                .map(|checkout| checkout.branch_id.clone())
        })
        .unwrap_or_else(|| ROOT_CONTEXT_BRANCH_ID.to_string())
}

impl BranchIndex {
    fn branch_tip(&self, branch_id: &str) -> anyhow::Result<u64> {
        self.branch_tips
            .get(branch_id)
            .copied()
            .ok_or_else(|| anyhow!("unknown context branch '{branch_id}'"))
    }
}

pub(crate) fn restore_session_history_projection(records: &[TranscriptRecord]) -> Vec<HistoryItem> {
    restore_history_projection(records)
        .into_iter()
        .map(|entry| entry.item)
        .collect()
}

pub(crate) fn derive_retired_source_spans(
    records: &[TranscriptRecord],
    tail_start_index: usize,
) -> Vec<ContextCompactionSourceSpan> {
    let history = restore_history_projection(records);
    canonical_retired_source_spans(merge_source_spans(
        history
            .iter()
            .take(tail_start_index.min(history.len()))
            .flat_map(|entry| entry.source_spans.iter().cloned()),
    ))
}

/// Canonical persisted retirement spans cover the deterministic raw source
/// closure of a retired history prefix. One inclusive interval deliberately
/// includes non-history records between those sources, because their derived
/// context/folded projections retire with that raw prefix.
pub(crate) fn canonical_retired_source_spans(
    spans: Vec<ContextCompactionSourceSpan>,
) -> Vec<ContextCompactionSourceSpan> {
    let spans = merge_source_spans(spans);
    let Some(first) = spans.first() else {
        return Vec::new();
    };
    let end_sequence = spans
        .iter()
        .map(|span| span.end_sequence)
        .max()
        .unwrap_or(first.end_sequence);
    vec![ContextCompactionSourceSpan {
        start_sequence: first.start_sequence,
        end_sequence,
    }]
}

/// Canonical cumulative retirement state. Each newly retired prefix is first
/// closed with `canonical_retired_source_spans`; prior closures remain separate
/// when retained raw material lies between them.
pub(crate) fn canonical_cumulative_retired_source_spans(
    prior: impl IntoIterator<Item = ContextCompactionSourceSpan>,
    new_closure: impl IntoIterator<Item = ContextCompactionSourceSpan>,
) -> Vec<ContextCompactionSourceSpan> {
    merge_source_spans(prior.into_iter().chain(new_closure))
}

pub(crate) fn restore_retired_source_spans_projection(
    records: &[TranscriptRecord],
) -> Vec<ContextCompactionSourceSpan> {
    let mut retired = Vec::new();
    for record in records {
        if let TranscriptEvent::ContextCompaction(event) = &record.event
            && event.outcome == "succeeded"
        {
            retired.extend(if event.retired_source_spans.is_empty() {
                derive_retired_source_spans(
                    &records[..records
                        .iter()
                        .position(|candidate| candidate.sequence == record.sequence)
                        .unwrap_or(records.len())],
                    event.tail_start_index,
                )
            } else {
                event.retired_source_spans.clone()
            });
        }
    }
    canonical_cumulative_retired_source_spans(Vec::new(), retired)
}

pub(crate) fn validate_context_compaction_event(
    records: &[TranscriptRecord],
    event: &crate::agent::ContextCompactionEvent,
) -> anyhow::Result<()> {
    ensure!(
        event.outcome == "succeeded",
        "only successful compaction events are projectable"
    );
    if !event.frame_identity_bindings.is_empty() {
        let mut keys = BTreeSet::new();
        let mut ids = BTreeSet::new();
        for binding in &event.frame_identity_bindings {
            ensure!(
                !binding.key.is_empty() && keys.insert(&binding.key),
                "context compaction frame identity bindings contain a duplicate or empty key"
            );
            ensure!(
                ids.insert(binding.frame_id),
                "context compaction frame identity bindings contain a duplicate frame id"
            );
        }
    }
    ensure!(
        !event.summary.trim().is_empty(),
        "context compaction summary must not be empty"
    );
    let history = restore_history_projection(records);
    ensure!(
        event.original_history_items == history.len(),
        "context compaction original_history_items is inconsistent with visible history"
    );
    ensure!(
        event.tail_start_index <= history.len(),
        "context compaction tail_start_index exceeds original history"
    );
    // Pre-GROUP-11 journals did not persist retirement spans. Their retained
    // count was either the final count or the pre-summary tail count. Treat
    // only that complete historical shape as legacy, then derive its spans
    // from the same visible branch prefix used by the original recorder.
    let canonical_retained_history_items = 1 + history.len() - event.tail_start_index;
    let legacy_event = event.retired_source_spans.is_empty()
        && matches!(
            event.retained_history_items,
            count if count == canonical_retained_history_items
                || count == history.len() - event.tail_start_index
        );
    ensure!(
        legacy_event || event.retained_history_items == canonical_retained_history_items,
        "context compaction retained_history_items is inconsistent with summary and tail"
    );
    let prior_retired_source_spans = restore_retired_source_spans_projection(records);
    let newly_retired_source_spans =
        derive_new_retired_source_spans(records, event.tail_start_index);
    let retired_source_spans = if legacy_event {
        derive_retired_source_spans(records, event.tail_start_index)
    } else {
        event.retired_source_spans.clone()
    };
    if !legacy_event {
        let expected = canonical_cumulative_retired_source_spans(
            prior_retired_source_spans,
            newly_retired_source_spans,
        );
        ensure!(
            retired_source_spans == expected,
            "context compaction retired source spans must exactly match the retired history source closure"
        );
    }
    let mut previous = None;
    for span in &retired_source_spans {
        ensure!(
            span.start_sequence <= span.end_sequence,
            "context compaction has inverted retired source span"
        );
        if let Some(previous) = previous {
            ensure!(
                previous < span.start_sequence,
                "context compaction retired source spans must be ordered and disjoint"
            );
        }
        previous = Some(span.end_sequence);
    }
    let covered = |source: &ContextCompactionSourceSpan| {
        retired_source_spans.iter().any(|retired| {
            retired.start_sequence <= source.start_sequence
                && source.end_sequence <= retired.end_sequence
        })
    };
    for entry in history.iter().take(event.tail_start_index) {
        if !matches!(entry.item, HistoryItem::ContextSummary { .. }) {
            ensure!(
                entry.source_spans.iter().all(covered),
                "retired source spans do not cover retired history source"
            );
        }
    }
    for entry in history.iter().skip(event.tail_start_index) {
        ensure!(
            entry.source_spans.iter().all(|source| !retired_source_spans
                .iter()
                .any(|retired| retired.start_sequence <= source.end_sequence
                    && source.start_sequence <= retired.end_sequence)),
            "retired source spans overlap retained protocol source"
        );
    }
    Ok(())
}

fn derive_new_retired_source_spans(
    records: &[TranscriptRecord],
    tail_start_index: usize,
) -> Vec<ContextCompactionSourceSpan> {
    let history = restore_history_projection(records);
    canonical_retired_source_spans(merge_source_spans(
        history
            .iter()
            .take(tail_start_index.min(history.len()))
            .filter(|entry| !matches!(entry.item, HistoryItem::ContextSummary { .. }))
            .flat_map(|entry| entry.source_spans.iter().cloned()),
    ))
}

#[derive(Debug, Clone)]
struct HistoryProjectionEntry {
    item: HistoryItem,
    source_spans: Vec<ContextCompactionSourceSpan>,
    turn_id: Option<u64>,
}

fn restore_history_projection(records: &[TranscriptRecord]) -> Vec<HistoryProjectionEntry> {
    let mut history: Vec<HistoryProjectionEntry> = Vec::new();
    let mut active_turn_id = None;
    for (index, record) in records.iter().enumerate() {
        match &record.event {
            TranscriptEvent::TurnStarted(event) => {
                // Recorders historically append the user frame before the start
                // audit record. Associate that adjacent frame with this turn.
                if let Some(previous) = history.last_mut()
                    && matches!(previous.item, HistoryItem::UserMessage { .. })
                {
                    previous.turn_id = Some(event.turn_id);
                }
                active_turn_id = Some(event.turn_id);
            }
            TranscriptEvent::ContextCompaction(event) => {
                if event.outcome != "succeeded" {
                    continue;
                }
                // Successful events have already been validated by every
                // fallible projection entry point. Never reinterpret malformed
                // persisted indexes by clamping them into a different request.
                let tail_start = event.tail_start_index;
                let retired_spans = if event.retired_source_spans.is_empty() {
                    derive_retired_source_spans(&records[..index], tail_start)
                } else {
                    merge_source_spans(event.retired_source_spans.iter().cloned())
                };
                let mut compacted =
                    Vec::with_capacity(1 + history.len().saturating_sub(tail_start));
                compacted.push(HistoryProjectionEntry {
                    item: HistoryItem::context_summary(event.summary.clone()),
                    source_spans: retired_spans,
                    turn_id: None,
                });
                compacted.extend(history.drain(tail_start..));
                history = compacted;
            }
            TranscriptEvent::TurnInterrupted { turn_id } => {
                if active_turn_id.is_none() || turn_id.is_none() || *turn_id == active_turn_id {
                    close_interrupted_turn(&mut history);
                    active_turn_id = None;
                }
            }
            TranscriptEvent::TurnFinalized(event) if event.outcome == "interrupted" => {
                if Some(event.turn_id) == active_turn_id {
                    close_interrupted_turn(&mut history);
                    active_turn_id = None;
                }
            }
            TranscriptEvent::TurnFinalized(event) if Some(event.turn_id) == active_turn_id => {
                active_turn_id = None;
            }
            _ => append_history_projection_entry_from_transcript_record(
                &mut history,
                record,
                active_turn_id,
            ),
        }
    }
    let cancelled_call_ids = records
        .iter()
        .filter_map(|record| match &record.event {
            TranscriptEvent::ToolCallCancelled { call_id, .. } => Some(call_id.as_str()),
            _ => None,
        })
        .collect::<std::collections::BTreeSet<_>>();
    normalize_incomplete_tool_call_groups(&mut history, active_turn_id, &cancelled_call_ids);
    history
}

/// Returns the unmatched lifecycle start visible at the selected branch leaf.
/// Historical turn IDs are an allocation counter, not evidence of a live turn.
fn active_turn_id_from_lifecycle_records(records: &[TranscriptRecord]) -> Option<u64> {
    let mut active_turn_id = None;
    for record in records {
        match &record.event {
            TranscriptEvent::TurnStarted(event) => active_turn_id = Some(event.turn_id),
            TranscriptEvent::TurnInterrupted { turn_id }
                if active_turn_id.is_none() || turn_id.is_none() || *turn_id == active_turn_id =>
            {
                active_turn_id = None;
            }
            TranscriptEvent::TurnFinalized(event) if Some(event.turn_id) == active_turn_id => {
                active_turn_id = None;
            }
            _ => {}
        }
    }
    active_turn_id
}

/// Historical tool calls cannot be resumed after a process restart. Remove an
/// incomplete group from the *projection* (never from the append-only
/// transcript), retaining any assistant text as a normal assistant message.
/// This leaves subsequent user turns protocol-legal without reordering records.
fn normalize_incomplete_tool_call_groups(
    history: &mut Vec<HistoryProjectionEntry>,
    active_turn_id: Option<u64>,
    cancelled_call_ids: &std::collections::BTreeSet<&str>,
) {
    let items = history
        .iter()
        .map(|entry| entry.item.clone())
        .collect::<Vec<_>>();
    let Ok(protocol) = analyze_history_items(&items, None) else {
        return;
    };
    let incomplete_indexes = protocol
        .tool_call_groups
        .iter()
        .filter(|group| {
            let retains_active_group = active_turn_id.is_some()
                && history[group.assistant_index].turn_id == active_turn_id;
            group.status == crate::protocol_frames::ToolCallGroupStatus::Incomplete
                && !retains_active_group
                && !group
                    .call_ids
                    .iter()
                    .any(|call_id| cancelled_call_ids.contains(call_id.as_str()))
        })
        .flat_map(|group| {
            std::iter::once(group.assistant_index).chain(group.tool_output_indexes.iter().copied())
        })
        .collect::<std::collections::BTreeSet<_>>();
    let cancelled_group_outputs = protocol
        .tool_call_groups
        .iter()
        .filter(|group| {
            group.status == crate::protocol_frames::ToolCallGroupStatus::Incomplete
                && group
                    .call_ids
                    .iter()
                    .any(|call_id| cancelled_call_ids.contains(call_id.as_str()))
        })
        .map(|group| {
            let output_call_ids = group
                .tool_output_indexes
                .iter()
                .filter_map(|index| match &history[*index].item {
                    HistoryItem::ToolOutput { call_id, .. } => Some(call_id.as_str()),
                    _ => None,
                })
                .collect::<std::collections::BTreeSet<_>>();
            (
                group.assistant_index,
                group
                    .call_ids
                    .iter()
                    .filter(|call_id| !output_call_ids.contains(call_id.as_str()))
                    .cloned()
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<std::collections::BTreeMap<_, _>>();
    if incomplete_indexes.is_empty() && cancelled_group_outputs.is_empty() {
        return;
    }

    let mut normalized = Vec::with_capacity(history.len());
    for (index, entry) in history.drain(..).enumerate() {
        if !incomplete_indexes.contains(&index) {
            normalized.push(entry);
        } else if let HistoryItem::AssistantToolCalls {
            text: Some(text), ..
        } = entry.item
        {
            normalized.push(HistoryProjectionEntry {
                item: HistoryItem::assistant(text),
                source_spans: entry.source_spans,
                turn_id: entry.turn_id,
            });
        }
        if let Some(call_ids) = cancelled_group_outputs.get(&index) {
            for call_id in call_ids {
                normalized.push(HistoryProjectionEntry {
                    item: HistoryItem::ToolOutput {
                        call_id: call_id.clone(),
                        output_json: r#"{"status":"cancelled","summary":"user cancelled"}"#.into(),
                    },
                    source_spans: Vec::new(),
                    turn_id: None,
                });
            }
        }
    }
    *history = normalized;
}

fn close_interrupted_turn(history: &mut Vec<HistoryProjectionEntry>) {
    let Some(last_conversation_item) = history.iter().rfind(|item| {
        matches!(
            item.item,
            HistoryItem::UserMessage { .. }
                | HistoryItem::InternalContinuation { .. }
                | HistoryItem::AssistantText { .. }
                | HistoryItem::ContextSummary { .. }
        )
    }) else {
        return;
    };

    if matches!(
        last_conversation_item.item,
        HistoryItem::UserMessage { .. } | HistoryItem::InternalContinuation { .. }
    ) {
        history.push(HistoryProjectionEntry {
            item: HistoryItem::assistant(String::new()),
            source_spans: Vec::new(),
            turn_id: None,
        });
    }
}

fn append_history_projection_entry_from_transcript_record(
    history: &mut Vec<HistoryProjectionEntry>,
    record: &TranscriptRecord,
    active_turn_id: Option<u64>,
) {
    // Committed batches are the protocol authority. Their subsequent lifecycle
    // starts are audit records only and must not create a second group.
    if let TranscriptEvent::ToolCallStarted { call_id, .. } = &record.event
        && tool_call_is_declared_by_incomplete_group(history, call_id)
    {
        return;
    }
    if let Some(item) = super::append_history_item_from_transcript_record(record) {
        // Legacy transcripts have one start record per call. Consecutive starts
        // are one assistant response until a result is appended.
        if matches!(record.event, TranscriptEvent::ToolCallStarted { .. })
            && let HistoryItem::AssistantToolCalls { calls, .. } = &item
            && let Some(previous) = history.last_mut()
            && let HistoryItem::AssistantToolCalls {
                calls: previous_calls,
                ..
            } = &mut previous.item
        {
            previous_calls.extend(calls.clone());
            previous
                .source_spans
                .extend(source_spans_for_history_record(record));
            return;
        }
        history.push(HistoryProjectionEntry {
            item,
            source_spans: source_spans_for_history_record(record),
            turn_id: active_turn_id,
        });
    }
}

fn tool_call_is_declared_by_incomplete_group(
    history: &[HistoryProjectionEntry],
    call_id: &str,
) -> bool {
    let Some(assistant_index) = history
        .iter()
        .rposition(|entry| matches!(entry.item, HistoryItem::AssistantToolCalls { .. }))
    else {
        return false;
    };
    let HistoryItem::AssistantToolCalls { calls, .. } = &history[assistant_index].item else {
        return false;
    };
    if !calls.iter().any(|call| call.call_id == call_id) {
        return false;
    }
    !history[assistant_index + 1..].iter().any(|entry| {
        matches!(&entry.item, HistoryItem::ToolOutput { call_id: output_call_id, .. } if output_call_id == call_id)
    })
}

fn protected_history_frame_ids(
    entries: &[HistoryProjectionEntry],
    frame_ids: &[RuntimeFrameId],
    current_turn_id: Option<u64>,
) -> anyhow::Result<Vec<RuntimeFrameId>> {
    let current_turn_start_index = current_turn_id.and_then(|turn_id| {
        entries
            .iter()
            .position(|entry| entry.turn_id == Some(turn_id))
    });
    let history = entries
        .iter()
        .map(|entry| entry.item.clone())
        .collect::<Vec<_>>();
    let protocol = analyze_history_items(&history, current_turn_start_index)?;
    let mut protected = protocol.protected_history_indexes();
    // The active turn is an atomic protocol boundary, including ordinary user
    // and assistant messages. Historical completed turns are intentionally not.
    if let Some(start) = current_turn_start_index {
        protected.extend(start..history.len());
    }
    Ok(protected
        .into_iter()
        .filter_map(|index| frame_ids.get(index).copied())
        .collect())
}

fn source_spans_for_history_record(record: &TranscriptRecord) -> Vec<ContextCompactionSourceSpan> {
    match &record.event {
        TranscriptEvent::UserMessage { .. }
        | TranscriptEvent::AssistantMessage { .. }
        | TranscriptEvent::AssistantToolCallBatch { .. }
        | TranscriptEvent::InternalContinuation { .. }
        | TranscriptEvent::ToolCallStarted { .. }
        | TranscriptEvent::ToolCallFinished { .. }
        | TranscriptEvent::ContextExperimentReturned { .. } => {
            vec![ContextCompactionSourceSpan {
                start_sequence: record.sequence,
                end_sequence: record.sequence,
            }]
        }
        _ => Vec::new(),
    }
}

fn merge_source_spans(
    spans: impl IntoIterator<Item = ContextCompactionSourceSpan>,
) -> Vec<ContextCompactionSourceSpan> {
    let mut spans = spans.into_iter().collect::<Vec<_>>();
    spans.sort_by_key(|span| (span.start_sequence, span.end_sequence));
    let mut merged: Vec<ContextCompactionSourceSpan> = Vec::new();
    for span in spans {
        if let Some(last) = merged.last_mut()
            && span.start_sequence <= last.end_sequence.saturating_add(1)
        {
            last.end_sequence = last.end_sequence.max(span.end_sequence);
        } else {
            merged.push(span);
        }
    }
    merged
}

pub(crate) fn restore_latest_model_projection(records: &[TranscriptRecord]) -> Option<String> {
    let mut model = None;
    for record in records {
        match &record.event {
            TranscriptEvent::SessionStarted { model: started } => model = Some(started.clone()),
            TranscriptEvent::ModelChanged { new_model, .. } => model = Some(new_model.clone()),
            _ => {}
        }
    }
    model
}

pub(crate) fn restore_max_turn_id_projection(records: &[TranscriptRecord]) -> u64 {
    records
        .iter()
        .filter_map(|record| match &record.event {
            TranscriptEvent::TurnStarted(event) => Some(event.turn_id),
            TranscriptEvent::ToolExecutionSummary(event) => Some(event.turn_id),
            TranscriptEvent::TurnFinalized(event) => Some(event.turn_id),
            TranscriptEvent::TurnInterrupted { turn_id } => *turn_id,
            _ => None,
        })
        .max()
        .unwrap_or(0)
}

pub(crate) fn project_child_session_summaries(
    child_dir: &Path,
    parent_records: &[TranscriptRecord],
) -> Vec<ChildSessionSummary> {
    let mut children = BTreeMap::new();

    for record in parent_records {
        if let TranscriptEvent::SubagentResult {
            parent_session_id,
            parent_run_id,
            child_session_id,
            agent_name,
            status,
            summary,
            ..
        } = &record.event
            && child_dir.join(format!("{child_session_id}.jsonl")).exists()
        {
            children.insert(
                child_session_id.clone(),
                ChildSessionSummary {
                    parent_session_id: parent_session_id.clone(),
                    parent_run_id: parent_run_id.clone(),
                    child_session_id: child_session_id.clone(),
                    agent_name: agent_name.clone(),
                    status: status.clone(),
                    summary: summary.clone(),
                    timestamp_ms: record.timestamp_ms,
                },
            );
        }
    }

    let mut children = children.into_values().collect::<Vec<_>>();
    children.sort_by(|left, right| {
        left.timestamp_ms
            .cmp(&right.timestamp_ms)
            .then_with(|| left.child_session_id.cmp(&right.child_session_id))
    });
    children
}

pub(crate) fn project_job_board(
    child_dir: &Path,
    parent_records: &[TranscriptRecord],
) -> anyhow::Result<Vec<JobBoardEntry>> {
    let mut jobs = BTreeMap::<String, JobBoardAccumulator>::new();

    for record in parent_records {
        match &record.event {
            TranscriptEvent::SubagentResult {
                run_id,
                child_session_id,
                agent_name,
                status,
                summary,
                ..
            } => {
                let entry = jobs.entry(run_id.clone()).or_default();
                entry.run_id = run_id.clone();
                entry.child_session_id = child_session_id.clone();
                entry.agent_name = agent_name.clone();
                entry.status = status.clone();
                entry.summary = summary.clone();
                entry.terminal = true;
                entry.active = false;
            }
            TranscriptEvent::Evidence {
                source:
                    crate::evidence::EvidenceSource::Subagent {
                        run_id,
                        child_session_id,
                        parent_tool,
                        ..
                    },
                summary,
                detail,
                tags,
                ..
            } => {
                let entry = jobs.entry(run_id.clone()).or_default();
                entry.run_id = run_id.clone();
                if entry.child_session_id.is_empty() {
                    entry.child_session_id = child_session_id.clone();
                }
                if entry.agent_name.is_empty() {
                    entry.agent_name = parent_tool.trim_start_matches("agent__").to_string();
                }
                if tags.iter().any(|tag| tag == "subagent_result") {
                    entry.summary = summary.clone();
                    if let Some(detail) = detail
                        && let Ok(structured) =
                            serde_json::from_str::<StructuredSubagentResult>(detail)
                    {
                        entry.malformed = structured.malformed;
                        entry.structured_status = Some(structured.status.clone());
                        if entry.status.is_empty() {
                            entry.status = structured.status;
                        }
                    }
                }
                if tags
                    .iter()
                    .any(|tag| tag == "subagent_reconciliation" || tag == "reconciled")
                {
                    entry.reconciled = true;
                }
            }
            _ => {}
        }
    }

    if child_dir.exists() {
        for entry in std::fs::read_dir(child_dir)? {
            let path = entry?.path();
            if path.extension().and_then(|ext| ext.to_str()) != Some("jsonl") {
                continue;
            }
            let child_session_id = match path.file_stem().and_then(|stem| stem.to_str()) {
                Some(value) => value.to_string(),
                None => continue,
            };
            let child_records = crate::transcript::read_records_allow_partial_tail(&path)?;
            let latest = child_records
                .iter()
                .rev()
                .find_map(|record| match &record.event {
                    TranscriptEvent::SubagentLifecycle {
                        run_id,
                        agent_name,
                        status,
                        detail,
                        ..
                    } => Some((
                        run_id.clone(),
                        agent_name.clone(),
                        status.clone(),
                        detail.clone(),
                    )),
                    _ => None,
                });
            let Some((run_id, agent_name, status, detail)) = latest else {
                continue;
            };
            if status != "running" {
                continue;
            }
            let job = jobs.entry(run_id.clone()).or_default();
            if job.terminal {
                continue;
            }
            job.run_id = run_id;
            job.child_session_id = child_session_id;
            job.agent_name = agent_name;
            job.status = status;
            job.summary = detail.unwrap_or_else(|| "subagent running".into());
            job.active = true;
        }
    }

    let mut entries = jobs
        .into_values()
        .filter(|entry| !entry.run_id.is_empty())
        .map(|entry| {
            let reconciled = entry.terminal && entry.reconciled;
            let unreconciled = entry.terminal && !entry.reconciled;
            let reusable_eligible = reconciled
                && entry.status == "completed"
                && entry.structured_status.as_deref() == Some("completed")
                && !entry.malformed;
            JobBoardEntry {
                active: entry.active,
                unreconciled,
                reconciled,
                reusable_eligible,
                run_id: entry.run_id,
                child_session_id: entry.child_session_id,
                agent_name: entry.agent_name,
                status: entry.status,
                summary: entry.summary,
            }
        })
        .collect::<Vec<_>>();
    entries.sort_by(|a, b| a.run_id.cmp(&b.run_id));
    Ok(entries)
}

#[derive(Debug, Clone, Default)]
struct JobBoardAccumulator {
    run_id: String,
    child_session_id: String,
    agent_name: String,
    status: String,
    summary: String,
    active: bool,
    terminal: bool,
    reconciled: bool,
    malformed: bool,
    structured_status: Option<String>,
}

#[derive(Debug, Default)]
struct TranscriptTimelineProjection {
    timeline: Timeline,
    current_auto_continue: AutoContinueState,
}

impl TranscriptTimelineProjection {
    fn apply_record(&mut self, record: &TranscriptRecord) {
        match &record.event {
            TranscriptEvent::UserMessage { content } => {
                self.timeline
                    .push_user_message(UserMessageEvent::from_submission(
                        UserMessageSubmission::new(
                            format!("restored-user-message-{}", record.sequence),
                            content.clone(),
                        ),
                    ))
            }
            TranscriptEvent::AssistantMessage { content } => self
                .timeline
                .push_restored_message(MessageRole::Assistant, content.clone()),
            TranscriptEvent::ContextCompaction(event) => {
                self.timeline
                    .push_compaction_separator(COMPACTION_SEPARATOR_LABEL);
                self.timeline
                    .push_restored_message(MessageRole::Assistant, event.summary.clone());
                self.timeline
                    .push_compaction_separator(COMPACTION_SEPARATOR_LABEL);
            }
            TranscriptEvent::ReasoningMessage { content } => {
                self.timeline.push_restored_reasoning(
                    format!("restored-reasoning-{}", record.sequence),
                    content.clone(),
                );
            }
            TranscriptEvent::ContextExperimentReturned {
                branch_id,
                outcome,
                summary,
                next_action,
                had_writes,
                ..
            } => self.timeline.push_restored_message(
                MessageRole::Assistant,
                crate::transcript::format_context_experiment_return(
                    branch_id,
                    outcome,
                    summary,
                    next_action.as_deref(),
                    *had_writes,
                ),
            ),
            TranscriptEvent::ToolCallStarted {
                call_id,
                name,
                args,
            } => {
                self.timeline.push_tool_started(ToolStartedEvent {
                    call_id: call_id.clone(),
                    name: name.clone(),
                    summary: format_tool_call(name, args),
                    arguments: Some(args.to_string()),
                });
            }
            TranscriptEvent::ToolCallFinished {
                call_id,
                name,
                ok,
                output,
            } => {
                self.timeline.push_tool_finished(ToolFinishedEvent {
                    call_id: call_id.clone(),
                    name: name.clone(),
                    summary: restored_tool_summary(name, *ok),
                    outcome: if *ok {
                        ToolOutcome::Success
                    } else {
                        ToolOutcome::Failure
                    },
                    output: serde_json::to_value(output)
                        .ok()
                        .map(|value| value.to_string()),
                });
            }
            TranscriptEvent::ToolCallCancelled { call_id, name } => {
                self.timeline.cancel_tool(call_id, name);
            }
            TranscriptEvent::TodoSnapshot { items } => {
                self.timeline
                    .push_todo_snapshot(TodoSnapshotEvent::new(items.clone()));
                self.timeline
                    .apply_auto_continue_changed(AutoContinueChangedEvent::new(
                        self.current_auto_continue.clone(),
                    ));
            }
            TranscriptEvent::AutoContinueChanged { state } => {
                self.current_auto_continue = state.clone();
                self.timeline
                    .apply_auto_continue_changed(AutoContinueChangedEvent::new(state.clone()));
            }
            TranscriptEvent::PermissionDecision {
                call_id,
                tool,
                args,
                allowed,
                reason,
            } => {
                self.timeline.push_restored_permission_decision(
                    call_id.clone().unwrap_or_else(|| tool.clone()),
                    tool.clone(),
                    format_tool_call(tool, args),
                    Some(args.to_string()),
                    if *allowed {
                        PermissionPromptStatus::Approved
                    } else {
                        PermissionPromptStatus::Denied
                    },
                    reason.clone(),
                );
            }
            TranscriptEvent::Error { message } => {
                self.timeline.push_error(ErrorEvent::new(message.clone()));
            }
            TranscriptEvent::TurnInterrupted { .. } => {
                self.timeline.cancel_active_tools();
                self.timeline.push_notice("Interrupted by user");
            }
            TranscriptEvent::TurnFinalized(event) => {
                if event.outcome == "interrupted" {
                    self.timeline.cancel_active_tools();
                    self.timeline.push_notice("Interrupted by user");
                }
            }
            TranscriptEvent::SubagentResult { .. }
            | TranscriptEvent::SubagentLifecycle { .. }
            | TranscriptEvent::SessionStarted { .. }
            | TranscriptEvent::SessionTitle { .. }
            | TranscriptEvent::ContextBranchCreated { .. }
            | TranscriptEvent::ContextBranchSummary { .. }
            | TranscriptEvent::ContextCheckout { .. }
            | TranscriptEvent::ContextExperimentStarted { .. }
            | TranscriptEvent::ContextNodeCreated { .. }
            | TranscriptEvent::ContextNodeLifecycle { .. }
            | TranscriptEvent::ContextViewOperationMetadata { .. }
            | TranscriptEvent::ContextSummaryArtifactMetadata { .. }
            | TranscriptEvent::FoldedOutputMetadata { .. }
            | TranscriptEvent::TurnStarted(_)
            | TranscriptEvent::ModelChanged { .. }
            | TranscriptEvent::PermissionModeChanged { .. }
            | TranscriptEvent::AutoContinuationScheduled { .. }
            | TranscriptEvent::AssistantToolCallBatch { .. }
            | TranscriptEvent::InternalContinuation { .. }
            | TranscriptEvent::ValidationAdvisory(_)
            | TranscriptEvent::ToolExecutionSummary(_)
            | TranscriptEvent::Evidence { .. }
            | TranscriptEvent::Unknown => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::ContextCompactionEvent;
    use crate::agent::{ToolExecutionSummaryEvent, TurnStartedEvent};
    use crate::context_tree::ContextNodeStatus;
    use crate::context_view::{
        ContextBlock, ContextBlockId, ContextBlockKind, ContextBlockSource, ContextViewProjection,
        FoldedOutputMetadata,
    };
    use crate::evidence::{EvidenceKind, EvidenceSource};
    use crate::protocol_frames::history_items_from_frames;
    use crate::request_builder::HistoryToolCall;
    use crate::runtime_context::RuntimeFrameKind;
    use crate::tool::ToolResult;
    use crate::tui::timeline::{TimelineItem, ToolExecutionStatus};
    use crate::user_content::{UserImageAttachment, UserMessageContent};
    use serde_json::json;

    fn record(event: TranscriptEvent) -> TranscriptRecord {
        record_at(1, event)
    }

    fn record_at(sequence: u64, event: TranscriptEvent) -> TranscriptRecord {
        TranscriptRecord {
            session_id: "s".into(),
            sequence,
            timestamp_ms: 0,
            context_branch_id: None,
            event,
        }
    }

    fn branch_record_at(
        sequence: u64,
        branch_id: &str,
        event: TranscriptEvent,
    ) -> TranscriptRecord {
        TranscriptRecord {
            session_id: "s".into(),
            sequence,
            timestamp_ms: 0,
            context_branch_id: Some(branch_id.into()),
            event,
        }
    }

    fn metadata_record_at(sequence: u64, event: TranscriptEvent) -> TranscriptRecord {
        TranscriptRecord {
            session_id: "s".into(),
            sequence,
            timestamp_ms: 0,
            context_branch_id: None,
            event,
        }
    }

    fn mixed_context_view(compacted: &[&str]) -> ContextViewProjection {
        let mut projection = ContextViewProjection::default();
        for (sequence, output_id) in [(1, "fold-1"), (2, "fold-2"), (3, "fold-3")] {
            let block_id = ContextBlockId::new(format!("block-{sequence}")).expect("valid block");
            projection.blocks.insert(
                block_id.clone(),
                ContextBlock {
                    block_id,
                    node_id: None,
                    kind: ContextBlockKind::ToolOutput,
                    title: format!("block {sequence}"),
                    detail: format!("detail {sequence}"),
                    source: ContextBlockSource::FoldedOutput {
                        output_id: output_id.into(),
                    },
                    source_start_sequence: Some(sequence),
                    available_sequence: Some(sequence),
                    protected_reasons: Vec::new(),
                    folded_output_id: Some(output_id.into()),
                },
            );
            projection.folded_outputs.insert(
                output_id.into(),
                FoldedOutputMetadata {
                    output_id: output_id.into(),
                    node_id: None,
                    output_kind: "tool_output".into(),
                    call_id: None,
                    tool_name: None,
                    stream: None,
                    content: String::new(),
                    byte_count: 0,
                    line_count: 0,
                    truncated: false,
                    shell_command: None,
                    source_start_sequence: Some(sequence),
                    source_end_sequence: Some(sequence),
                    available_sequence: Some(sequence),
                    tool_ok: None,
                    exit_status: None,
                },
            );
        }
        projection.compacted_block_ids = compacted
            .iter()
            .map(|id| ContextBlockId::new(*id).expect("valid block"))
            .collect();
        projection
    }

    fn snapshot_for_context_view(context_view: &ContextViewProjection) -> RuntimeSnapshot {
        let mut snapshot = RuntimeSnapshot::new("main");
        snapshot.active_context.visible_block_ids = context_view.provider_visible_block_ids();
        append_context_frames(&mut snapshot, context_view).expect("context frames");
        append_folded_output_refs(&mut snapshot, context_view).expect("folded frames");
        append_prompt_contributors(&mut snapshot, context_view, &[], &[])
            .expect("prompt contributors");
        snapshot
    }

    fn frame_ids_by_source(
        snapshot: &RuntimeSnapshot,
        kind: RuntimeFrameKind,
    ) -> BTreeMap<String, RuntimeFrameId> {
        snapshot
            .frames
            .iter()
            .filter(|frame| frame.kind == kind)
            .map(|frame| {
                (
                    frame.provenance.source_id.clone().expect("source id"),
                    frame.id,
                )
            })
            .collect()
    }

    #[test]
    fn context_and_folded_frame_ids_ignore_mixed_retirement_visibility() {
        let live = snapshot_for_context_view(&mixed_context_view(&[]));
        let mixed = snapshot_for_context_view(&mixed_context_view(&["block-2"]));
        let repeated = snapshot_for_context_view(&mixed_context_view(&["block-1", "block-2"]));

        for kind in [
            RuntimeFrameKind::ContextBlock,
            RuntimeFrameKind::FoldedOutput,
        ] {
            let live_ids = frame_ids_by_source(&live, kind);
            assert_eq!(frame_ids_by_source(&mixed, kind), live_ids);
            assert_eq!(frame_ids_by_source(&repeated, kind), live_ids);
        }
        assert!(
            mixed
                .context_view
                .provider_visible_block_ids()
                .iter()
                .all(|id| id != "block-2")
        );
        assert!(mixed.frames.iter().any(|frame| {
            frame.provenance.source_id.as_deref() == Some("block-2")
                && frame.visibility == FrameVisibility::Retired
        }));
        assert!(mixed.frames.iter().any(|frame| {
            frame.provenance.source_id.as_deref() == Some("fold-2")
                && frame.visibility == FrameVisibility::Retired
        }));

        let contributor_ids = |snapshot: &RuntimeSnapshot, contributor_id| {
            snapshot
                .prompt_contributors
                .iter()
                .find(|contributor| contributor.contributor_id == contributor_id)
                .expect("contributor")
                .frame_ids
                .clone()
        };
        assert_eq!(
            contributor_ids(&mixed, "context-view-active"),
            contributor_ids(&live, "context-view-active")
        );
        assert_eq!(
            contributor_ids(&mixed, "folded-outputs"),
            contributor_ids(&live, "folded-outputs")
        );
        mixed
            .validate_references()
            .expect("contributor references resolve");

        let restored: RuntimeSnapshot = serde_json::from_str(
            &serde_json::to_string(&mixed).expect("persist compacted snapshot"),
        )
        .expect("restore compacted snapshot");
        assert_eq!(
            frame_ids_by_source(&restored, RuntimeFrameKind::ContextBlock),
            frame_ids_by_source(&mixed, RuntimeFrameKind::ContextBlock)
        );
        assert_eq!(
            frame_ids_by_source(&restored, RuntimeFrameKind::FoldedOutput),
            frame_ids_by_source(&mixed, RuntimeFrameKind::FoldedOutput)
        );
        assert_eq!(
            contributor_ids(&restored, "context-view-active"),
            contributor_ids(&mixed, "context-view-active")
        );
        assert_eq!(
            contributor_ids(&restored, "folded-outputs"),
            contributor_ids(&mixed, "folded-outputs")
        );
        restored
            .validate_references()
            .expect("restored contributor references resolve");
    }

    #[test]
    fn restored_permissions_are_terminal_not_pending_prompts() {
        let timeline =
            timeline_from_transcript_records(&[record(TranscriptEvent::PermissionDecision {
                call_id: Some("call-1".into()),
                tool: "shell__exec".into(),
                args: json!({"command": "cargo test"}),
                allowed: false,
                reason: Some("Denied by user from TUI permission prompt".into()),
            })]);

        assert!(matches!(
            timeline.items().first(),
            Some(TimelineItem::Permission(permission))
                if permission.status == PermissionPromptStatus::Denied
                    && permission.resolution_reason.as_deref() == Some("Denied by user from TUI permission prompt")
        ));
    }

    #[test]
    fn restored_compaction_uses_separator_summary_separator_shape() {
        let timeline = timeline_from_transcript_records(&[record(
            TranscriptEvent::ContextCompaction(ContextCompactionEvent {
                outcome: "succeeded".into(),
                summary: "Earlier context summary".into(),
                tail_start_index: 5,
                original_history_items: 11,
                retained_history_items: 3,
                retired_source_spans: Vec::new(),
                frame_identity_bindings: Vec::new(),
                detail: None,
            }),
        )]);

        assert_eq!(timeline.items().len(), 3);
        assert!(matches!(timeline.items()[0], TimelineItem::Notice(_)));
        assert!(matches!(timeline.items()[1], TimelineItem::Assistant(_)));
        assert!(matches!(timeline.items()[2], TimelineItem::Notice(_)));
    }

    #[test]
    fn restored_subagent_records_are_ignored_in_timeline_projection() {
        let timeline = timeline_from_transcript_records(&[
            record(TranscriptEvent::SubagentLifecycle {
                run_id: "run-1".into(),
                parent_session_id: "parent".into(),
                parent_run_id: "turn-1".into(),
                agent_name: "explorer".into(),
                status: "running".into(),
                detail: None,
            }),
            TranscriptRecord {
                session_id: "s".into(),
                sequence: 2,
                timestamp_ms: 1,
                context_branch_id: None,
                event: TranscriptEvent::SubagentResult {
                    run_id: "run-1".into(),
                    parent_session_id: "parent".into(),
                    parent_run_id: "turn-1".into(),
                    child_session_id: "child".into(),
                    agent_name: "explorer".into(),
                    status: "completed".into(),
                    summary: "done".into(),
                },
            },
        ]);

        assert!(timeline.items().is_empty());
    }

    #[test]
    fn restored_tool_events_keep_terminal_outcomes_without_live_pending_path() {
        let timeline = timeline_from_transcript_records(&[
            record(TranscriptEvent::ToolCallStarted {
                call_id: "call-1".into(),
                name: "shell__exec".into(),
                args: json!({"command": "sleep 10"}),
            }),
            TranscriptRecord {
                session_id: "s".into(),
                sequence: 2,
                timestamp_ms: 1,
                context_branch_id: None,
                event: TranscriptEvent::ToolCallCancelled {
                    call_id: "call-1".into(),
                    name: "shell__exec".into(),
                },
            },
        ]);

        assert!(matches!(
            timeline.items().first(),
            Some(TimelineItem::Tool(tool)) if tool.status == ToolExecutionStatus::Cancelled
        ));
        assert!(timeline.active_tool().is_none());
    }

    #[test]
    fn restored_user_messages_keep_image_attachments() {
        let timeline = timeline_from_transcript_records(&[record(TranscriptEvent::UserMessage {
            content: UserMessageContent::new(
                "inspect this",
                vec![UserImageAttachment {
                    id: "img-1".into(),
                    label: "screen.png".into(),
                    mime: "image/png".into(),
                    data_url: "data:image/png;base64,AAAA".into(),
                }],
            ),
        })]);

        assert!(matches!(
            timeline.items().first(),
            Some(TimelineItem::User(message))
                if message.text == "inspect this"
                    && message.attachments.len() == 1
                    && message.attachments[0].label == "screen.png"
        ));

        let history = restore_session_history_projection(&[record(TranscriptEvent::UserMessage {
            content: UserMessageContent::new(
                "inspect this",
                vec![UserImageAttachment {
                    id: "img-1".into(),
                    label: "screen.png".into(),
                    mime: "image/png".into(),
                    data_url: "data:image/png;base64,AAAA".into(),
                }],
            ),
        })]);
        assert!(matches!(
            history.first(),
            Some(HistoryItem::UserMessage { content })
                if content.attachments.len() == 1 && content.attachments[0].label == "screen.png"
        ));
    }

    #[test]
    fn replay_context_tree_uses_default_root_for_legacy_transcripts() {
        let tree = replay_context_tree(&[
            record_at(
                1,
                TranscriptEvent::SessionStarted {
                    model: "gpt-5".into(),
                },
            ),
            record_at(
                2,
                TranscriptEvent::UserMessage {
                    content: UserMessageContent::from("hello"),
                },
            ),
        ])
        .expect("replay legacy context tree");

        assert_eq!(tree.root_node_id().as_str(), "root");
        assert_eq!(tree.active_node_id().map(|id| id.as_str()), Some("root"));
        assert_eq!(tree.node_count(), 1);
    }

    #[test]
    fn replay_context_tree_reconstructs_valid_tree() {
        let tree = project_context_tree(&[
            metadata_record_at(
                1,
                TranscriptEvent::ContextNodeCreated {
                    node_id: "child".into(),
                    parent_node_id: Some("root".into()),
                    label: Some("Task branch".into()),
                    purpose: Some("Investigate session-level replay".into()),
                    block_ref: None,
                    source_ref: None,
                },
            ),
            metadata_record_at(
                2,
                TranscriptEvent::ContextNodeLifecycle {
                    node_id: "root".into(),
                    status: ContextNodeStatus::Inactive,
                },
            ),
            metadata_record_at(
                3,
                TranscriptEvent::ContextNodeLifecycle {
                    node_id: "child".into(),
                    status: ContextNodeStatus::Active,
                },
            ),
        ])
        .expect("replay valid context tree");

        let child = tree
            .node(&ContextNodeId::new("child").expect("node id"))
            .expect("child node exists");
        assert_eq!(
            child.parent_node_id.as_ref().map(|id| id.as_str()),
            Some("root")
        );
        assert_eq!(child.label.as_deref(), Some("Task branch"));
        assert_eq!(
            child.purpose.as_deref(),
            Some("Investigate session-level replay")
        );
        assert_eq!(tree.active_node_id().map(|id| id.as_str()), Some("child"));
        assert_eq!(tree.node_count(), 2);
    }

    #[test]
    fn replay_context_tree_rejects_unknown_parent() {
        let error = replay_context_tree(&[metadata_record_at(
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
        .expect_err("unknown parent should fail");

        assert!(
            error
                .to_string()
                .contains("unknown parent context node 'missing'")
        );
    }

    #[test]
    fn replay_context_tree_rejects_duplicate_active_node() {
        let error = replay_context_tree(&[
            metadata_record_at(
                1,
                TranscriptEvent::ContextNodeCreated {
                    node_id: "child-a".into(),
                    parent_node_id: Some("root".into()),
                    label: None,
                    purpose: None,
                    block_ref: None,
                    source_ref: None,
                },
            ),
            metadata_record_at(
                2,
                TranscriptEvent::ContextNodeCreated {
                    node_id: "child-b".into(),
                    parent_node_id: Some("root".into()),
                    label: None,
                    purpose: None,
                    block_ref: None,
                    source_ref: None,
                },
            ),
            metadata_record_at(
                3,
                TranscriptEvent::ContextNodeLifecycle {
                    node_id: "root".into(),
                    status: ContextNodeStatus::Inactive,
                },
            ),
            metadata_record_at(
                4,
                TranscriptEvent::ContextNodeLifecycle {
                    node_id: "child-a".into(),
                    status: ContextNodeStatus::Active,
                },
            ),
            metadata_record_at(
                5,
                TranscriptEvent::ContextNodeLifecycle {
                    node_id: "child-b".into(),
                    status: ContextNodeStatus::Active,
                },
            ),
        ])
        .expect_err("duplicate active node should fail");

        assert!(
            error
                .to_string()
                .contains("cannot activate context node 'child-b' while 'child-a' is active")
        );
    }

    #[test]
    fn replay_context_tree_rejects_duplicate_node_with_second_parent() {
        let error = replay_context_tree(&[
            metadata_record_at(
                1,
                TranscriptEvent::ContextNodeCreated {
                    node_id: "parent-b".into(),
                    parent_node_id: Some("root".into()),
                    label: None,
                    purpose: None,
                    block_ref: None,
                    source_ref: None,
                },
            ),
            metadata_record_at(
                2,
                TranscriptEvent::ContextNodeCreated {
                    node_id: "child".into(),
                    parent_node_id: Some("root".into()),
                    label: None,
                    purpose: None,
                    block_ref: None,
                    source_ref: None,
                },
            ),
            metadata_record_at(
                3,
                TranscriptEvent::ContextNodeCreated {
                    node_id: "child".into(),
                    parent_node_id: Some("parent-b".into()),
                    label: None,
                    purpose: None,
                    block_ref: None,
                    source_ref: None,
                },
            ),
        ])
        .expect_err("duplicate node should fail");

        assert!(
            error
                .to_string()
                .contains("duplicate context node_id 'child'")
        );
    }

    #[test]
    fn replay_context_tree_rejects_self_parent() {
        let error = replay_context_tree(&[metadata_record_at(
            1,
            TranscriptEvent::ContextNodeCreated {
                node_id: "self".into(),
                parent_node_id: Some("self".into()),
                label: None,
                purpose: None,
                block_ref: None,
                source_ref: None,
            },
        )])
        .expect_err("self parent should fail");

        assert!(
            error
                .to_string()
                .contains("context node 'self' cannot be its own parent")
        );
    }

    #[test]
    fn default_cursor_preserves_current_behavior() {
        let records = vec![
            record_at(
                1,
                TranscriptEvent::SessionStarted {
                    model: "gpt-5".into(),
                },
            ),
            record_at(
                2,
                TranscriptEvent::UserMessage {
                    content: UserMessageContent::from("hello"),
                },
            ),
            record_at(
                3,
                TranscriptEvent::AssistantMessage {
                    content: "hi".into(),
                },
            ),
        ];

        let expected = project_session_restore_snapshot("s".into(), records.clone(), None)
            .expect("default snapshot");
        let actual = build_session_context_snapshot(
            "s".into(),
            records.clone(),
            None,
            SessionContextCursor {
                branch_id: None,
                leaf_sequence: None,
            },
        )
        .expect("cursor snapshot");

        assert_eq!(actual.branch_id, ROOT_CONTEXT_BRANCH_ID);
        assert_eq!(actual.leaf_sequence, 3);
        assert_eq!(actual.records.len(), expected.records.len());
        assert_eq!(
            format!("{:?}", actual.messages),
            format!("{:?}", expected.messages)
        );
        assert_eq!(actual.history, expected.history);
        assert_eq!(actual.evidence, expected.evidence);
        assert_eq!(actual.latest_model, expected.latest_model);
        assert_eq!(actual.max_turn_id, expected.max_turn_id);
    }

    #[test]
    fn explicit_leaf_truncates_future_records() {
        let records = vec![
            record_at(
                1,
                TranscriptEvent::UserMessage {
                    content: UserMessageContent::from("before"),
                },
            ),
            record_at(
                2,
                TranscriptEvent::AssistantMessage {
                    content: "visible".into(),
                },
            ),
            record_at(
                3,
                TranscriptEvent::AssistantMessage {
                    content: "hidden".into(),
                },
            ),
        ];

        let snapshot = build_session_context_snapshot(
            "s".into(),
            records.clone(),
            None,
            SessionContextCursor {
                branch_id: None,
                leaf_sequence: Some(2),
            },
        )
        .expect("snapshot truncated at leaf");

        assert_eq!(snapshot.leaf_sequence, 2);
        assert_eq!(snapshot.records.len(), 2);
        assert!(
            snapshot
                .messages
                .iter()
                .all(|message| message.content != "hidden")
        );
        assert!(matches!(
            snapshot.history.as_slice(),
            [HistoryItem::UserMessage { .. }, HistoryItem::AssistantText { text }] if text == "visible"
        ));
    }

    #[test]
    fn compaction_before_leaf_still_restores_context_summary() {
        let records = vec![
            record_at(
                1,
                TranscriptEvent::UserMessage {
                    content: UserMessageContent::from("before"),
                },
            ),
            record_at(
                2,
                TranscriptEvent::AssistantMessage {
                    content: "reply".into(),
                },
            ),
            record_at(
                3,
                TranscriptEvent::ContextCompaction(ContextCompactionEvent {
                    outcome: "succeeded".into(),
                    summary: "summary".into(),
                    tail_start_index: 1,
                    original_history_items: 2,
                    retained_history_items: 1,
                    retired_source_spans: Vec::new(),
                    frame_identity_bindings: Vec::new(),
                    detail: None,
                }),
            ),
            record_at(
                4,
                TranscriptEvent::AssistantMessage {
                    content: "after".into(),
                },
            ),
        ];

        let snapshot = build_session_context_snapshot(
            "s".into(),
            records.clone(),
            None,
            SessionContextCursor {
                branch_id: None,
                leaf_sequence: Some(4),
            },
        )
        .expect("snapshot with compaction");

        assert!(matches!(
            snapshot.history.first(),
            Some(HistoryItem::ContextSummary { text }) if text == "summary"
        ));
    }

    #[test]
    fn compaction_projection_rejects_malformed_modern_fields_but_reads_legacy_shape() {
        let prefix = || {
            vec![
                record_at(
                    1,
                    TranscriptEvent::UserMessage {
                        content: UserMessageContent::from("old"),
                    },
                ),
                record_at(
                    2,
                    TranscriptEvent::AssistantMessage {
                        content: "tail".into(),
                    },
                ),
            ]
        };
        let project = |event| {
            let mut records = prefix();
            records.push(record_at(3, TranscriptEvent::ContextCompaction(event)));
            build_session_context_snapshot(
                "s".into(),
                records,
                None,
                SessionContextCursor {
                    branch_id: None,
                    leaf_sequence: None,
                },
            )
        };

        let legacy = project(ContextCompactionEvent {
            outcome: "succeeded".into(),
            summary: "legacy summary".into(),
            tail_start_index: 1,
            original_history_items: 2,
            retained_history_items: 1,
            retired_source_spans: Vec::new(),
            frame_identity_bindings: Vec::new(),
            detail: None,
        })
        .expect("legacy span-less compaction remains readable");
        assert!(matches!(
            legacy.history.as_slice(),
            [HistoryItem::ContextSummary { text }, HistoryItem::AssistantText { text: tail }]
                if text == "legacy summary" && tail == "tail"
        ));

        let modern_spans = vec![ContextCompactionSourceSpan {
            start_sequence: 1,
            end_sequence: 1,
        }];
        let retained_error = project(ContextCompactionEvent {
            outcome: "succeeded".into(),
            summary: "summary".into(),
            tail_start_index: 1,
            original_history_items: 2,
            retained_history_items: 1,
            retired_source_spans: modern_spans.clone(),
            frame_identity_bindings: Vec::new(),
            detail: None,
        })
        .expect_err("modern retained count must include the summary");
        assert!(
            retained_error
                .to_string()
                .contains("retained_history_items is inconsistent with summary and tail")
        );

        let original_count_error = project(ContextCompactionEvent {
            outcome: "succeeded".into(),
            summary: "summary".into(),
            tail_start_index: 1,
            original_history_items: 3,
            retained_history_items: 2,
            retired_source_spans: modern_spans.clone(),
            frame_identity_bindings: Vec::new(),
            detail: None,
        })
        .expect_err("modern original count must match the visible branch history");
        assert!(
            original_count_error
                .to_string()
                .contains("original_history_items is inconsistent with visible history")
        );

        let tail_error = project(ContextCompactionEvent {
            outcome: "succeeded".into(),
            summary: "summary".into(),
            tail_start_index: 3,
            original_history_items: 2,
            retained_history_items: 0,
            retired_source_spans: modern_spans,
            frame_identity_bindings: Vec::new(),
            detail: None,
        })
        .expect_err("modern tail index must be in the visible branch history");
        assert!(
            tail_error
                .to_string()
                .contains("tail_start_index exceeds original history")
        );
    }

    #[test]
    fn modern_compaction_spans_must_equal_the_canonical_retired_source_closure() {
        let records = vec![
            record_at(
                1,
                TranscriptEvent::UserMessage {
                    content: UserMessageContent::from("old user"),
                },
            ),
            record_at(
                2,
                TranscriptEvent::ReasoningMessage {
                    content: "dependent raw record".into(),
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
                TranscriptEvent::AssistantMessage {
                    content: "retained tail".into(),
                },
            ),
        ];
        let event = |retired_source_spans| ContextCompactionEvent {
            outcome: "succeeded".into(),
            summary: "summary".into(),
            tail_start_index: 2,
            original_history_items: 3,
            retained_history_items: 2,
            retired_source_spans,
            frame_identity_bindings: Vec::new(),
            detail: None,
        };
        let canonical = vec![ContextCompactionSourceSpan {
            start_sequence: 1,
            end_sequence: 3,
        }];

        validate_context_compaction_event(&records, &event(canonical.clone()))
            .expect("the deterministic closure includes the dependent raw record");
        for invalid in [
            vec![ContextCompactionSourceSpan {
                start_sequence: 1,
                end_sequence: 4,
            }],
            vec![ContextCompactionSourceSpan {
                start_sequence: 1,
                end_sequence: 2,
            }],
            vec![
                ContextCompactionSourceSpan {
                    start_sequence: 1,
                    end_sequence: 1,
                },
                ContextCompactionSourceSpan {
                    start_sequence: 2,
                    end_sequence: 3,
                },
            ],
        ] {
            assert!(validate_context_compaction_event(&records, &event(invalid)).is_err());
        }
    }

    #[test]
    fn compaction_binding_replay_rejects_prior_id_collision_before_current_remap() {
        let first_event = ContextCompactionEvent {
            outcome: "succeeded".into(),
            summary: "first summary".into(),
            tail_start_index: 1,
            original_history_items: 2,
            retained_history_items: 2,
            retired_source_spans: vec![ContextCompactionSourceSpan {
                start_sequence: 1,
                end_sequence: 1,
            }],
            frame_identity_bindings: Vec::new(),
            detail: None,
        };
        let mut records = vec![
            record_at(
                1,
                TranscriptEvent::UserMessage {
                    content: UserMessageContent::from("old user"),
                },
            ),
            record_at(
                2,
                TranscriptEvent::AssistantMessage {
                    content: "retained reply".into(),
                },
            ),
            record_at(3, TranscriptEvent::ContextCompaction(first_event)),
            record_at(
                4,
                TranscriptEvent::UserMessage {
                    content: UserMessageContent::from("new user"),
                },
            ),
        ];
        let unbound_snapshot = |records: &[TranscriptRecord]| {
            let resolved = resolve_branch_context(
                records.to_vec(),
                SessionContextCursor {
                    branch_id: None,
                    leaf_sequence: None,
                },
            )
            .expect("resolved root branch");
            let latest_model = restore_latest_model_projection(&resolved.records);
            runtime_snapshot_from_resolved_context_unbound(
                "s",
                records,
                &resolved,
                latest_model.as_deref(),
                &[],
            )
            .expect("unbound runtime snapshot")
        };

        let first_snapshot = unbound_snapshot(&records[..3]);
        let mut first_bindings = compaction_frame_identity_bindings(&first_snapshot);
        let second_retired_source_spans = canonical_cumulative_retired_source_spans(
            vec![ContextCompactionSourceSpan {
                start_sequence: 1,
                end_sequence: 1,
            }],
            derive_new_retired_source_spans(&records, 3),
        );
        records.push(record_at(
            5,
            TranscriptEvent::ContextCompaction(ContextCompactionEvent {
                outcome: "succeeded".into(),
                summary: "second summary".into(),
                tail_start_index: 3,
                original_history_items: 3,
                retained_history_items: 1,
                retired_source_spans: second_retired_source_spans,
                frame_identity_bindings: Vec::new(),
                detail: None,
            }),
        ));
        let second_snapshot = unbound_snapshot(&records);
        let new_user = second_snapshot
            .frames
            .iter()
            .find(|frame| {
                frame.provenance.source_span
                    == Some(SourceSpan {
                        start_sequence: 4,
                        end_sequence: 4,
                    })
            })
            .expect("new user frame");
        let summary_key = first_snapshot
            .frames
            .iter()
            .find(|frame| frame.provenance.source == RuntimeSource::SummaryArtifact)
            .expect("first summary frame")
            .durable_identity_key();
        let new_user_key = new_user.durable_identity_key();
        let colliding_id = new_user.id.as_u64();
        first_bindings
            .iter_mut()
            .find(|binding| binding.key == summary_key)
            .expect("first summary binding")
            .frame_id = colliding_id;

        let mut second_bindings = compaction_frame_identity_bindings(&second_snapshot);
        let remapped_new_user_id = u64::MAX;
        assert!(
            second_snapshot
                .frames
                .iter()
                .all(|frame| frame.id.as_u64() != remapped_new_user_id)
        );
        second_bindings
            .iter_mut()
            .find(|binding| binding.key == summary_key)
            .expect("second summary binding")
            .frame_id = colliding_id;
        second_bindings
            .iter_mut()
            .find(|binding| binding.key == new_user_key)
            .expect("new user binding")
            .frame_id = remapped_new_user_id;
        match &mut records[2].event {
            TranscriptEvent::ContextCompaction(event) => {
                event.frame_identity_bindings = first_bindings;
            }
            _ => unreachable!("first compaction record"),
        }
        match &mut records[4].event {
            TranscriptEvent::ContextCompaction(event) => {
                event.frame_identity_bindings = second_bindings;
            }
            _ => unreachable!("second compaction record"),
        }

        let error = project_runtime_restore_snapshot(
            "s".into(),
            records,
            None,
            SessionContextCursor {
                branch_id: None,
                leaf_sequence: None,
            },
            &[],
        )
        .expect_err("prior binding collision must fail before the current remap");
        assert!(
            error
                .to_string()
                .contains("runtime snapshot contains duplicate frame id")
        );
    }

    #[test]
    fn derive_retired_source_spans_tracks_history_sources_before_tail() {
        let records = vec![
            record_at(
                1,
                TranscriptEvent::UserMessage {
                    content: UserMessageContent::from("old user"),
                },
            ),
            record_at(
                2,
                TranscriptEvent::AssistantMessage {
                    content: "old assistant".into(),
                },
            ),
            record_at(
                3,
                TranscriptEvent::ToolCallStarted {
                    call_id: "call-1".into(),
                    name: "shell__exec".into(),
                    args: serde_json::json!({"command": "cargo test"}),
                },
            ),
            record_at(
                4,
                TranscriptEvent::ToolCallFinished {
                    call_id: "call-1".into(),
                    name: "shell__exec".into(),
                    ok: true,
                    output: crate::tool::ToolResult::ok("shell__exec", serde_json::json!({})),
                },
            ),
        ];

        let spans = derive_retired_source_spans(&records, 3);
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].start_sequence, 1);
        assert_eq!(spans[0].end_sequence, 3);
    }

    #[test]
    fn derive_retired_source_spans_covers_non_history_records_between_retired_sources() {
        let records = vec![
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
                TranscriptEvent::AssistantMessage {
                    content: "retained tail".into(),
                },
            ),
        ];

        let spans = derive_retired_source_spans(&records, 2);
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].start_sequence, 1);
        assert_eq!(spans[0].end_sequence, 3);
    }

    #[test]
    fn leaf_beyond_max_sequence_returns_error() {
        let records = vec![record_at(
            2,
            TranscriptEvent::UserMessage {
                content: UserMessageContent::from("hello"),
            },
        )];

        let error = build_session_context_snapshot(
            "s".into(),
            records,
            None,
            SessionContextCursor {
                branch_id: None,
                leaf_sequence: Some(3),
            },
        )
        .expect_err("leaf past max sequence should fail");

        assert!(
            error
                .to_string()
                .contains("leaf_sequence 3 exceeds max transcript sequence 2")
        );
    }

    #[test]
    fn max_turn_id_is_global_across_restored_leaf_cuts() {
        let records = vec![
            record_at(
                1,
                TranscriptEvent::TurnStarted(TurnStartedEvent {
                    turn_id: 1,
                    intent: "task".into(),
                    directive: "do it".into(),
                    validation_reminder: String::new(),
                }),
            ),
            record_at(
                2,
                TranscriptEvent::ToolExecutionSummary(ToolExecutionSummaryEvent {
                    turn_id: 1,
                    call_id: "call-1".into(),
                    name: "shell__exec".into(),
                    status: "completed".into(),
                    rejection: None,
                    effect_kind: "read".into(),
                    primary_path: None,
                    command: None,
                }),
            ),
            record_at(
                3,
                TranscriptEvent::ContextBranchCreated {
                    branch_id: "later".into(),
                    parent_branch_id: ROOT_CONTEXT_BRANCH_ID.into(),
                    base_sequence: 2,
                    label: None,
                },
            ),
            branch_record_at(
                4,
                "later",
                TranscriptEvent::TurnStarted(TurnStartedEvent {
                    turn_id: 7,
                    intent: "future".into(),
                    directive: "later".into(),
                    validation_reminder: String::new(),
                }),
            ),
            branch_record_at(
                5,
                "later",
                TranscriptEvent::TurnInterrupted { turn_id: Some(7) },
            ),
        ];
        let records_for_snapshot = records.clone();

        let snapshot = build_session_context_snapshot(
            "s".into(),
            records_for_snapshot,
            None,
            SessionContextCursor {
                branch_id: None,
                leaf_sequence: Some(2),
            },
        )
        .expect("snapshot before future turn");

        assert_eq!(snapshot.max_turn_id, 7);

        let runtime = project_runtime_restore_snapshot(
            "s".into(),
            records,
            None,
            SessionContextCursor {
                branch_id: Some(ROOT_CONTEXT_BRANCH_ID.into()),
                leaf_sequence: Some(2),
            },
            &[],
        )
        .expect("root runtime snapshot before sibling turn");
        assert_eq!(runtime.max_turn_id, 7);
        assert_eq!(runtime.snapshot.current_turn_id, Some(1));
    }

    #[test]
    fn evidence_respects_leaf_cut() {
        let records = vec![
            record_at(
                1,
                TranscriptEvent::Evidence {
                    id: "ev-1".into(),
                    evidence_kind: EvidenceKind::Decision,
                    title: "one".into(),
                    summary: "visible".into(),
                    detail: None,
                    source: EvidenceSource::Transcript { sequence: 1 },
                    tags: vec![],
                },
            ),
            record_at(
                2,
                TranscriptEvent::Evidence {
                    id: "ev-2".into(),
                    evidence_kind: EvidenceKind::Decision,
                    title: "two".into(),
                    summary: "hidden".into(),
                    detail: None,
                    source: EvidenceSource::Transcript { sequence: 2 },
                    tags: vec![],
                },
            ),
        ];

        let snapshot = build_session_context_snapshot(
            "s".into(),
            records,
            None,
            SessionContextCursor {
                branch_id: None,
                leaf_sequence: Some(1),
            },
        )
        .expect("snapshot with evidence leaf");

        assert_eq!(snapshot.evidence.len(), 1);
        assert_eq!(snapshot.evidence[0].summary, "visible");
    }

    #[test]
    fn old_transcript_default_restore_still_matches_linear_behavior() {
        let records = vec![
            record_at(
                1,
                TranscriptEvent::SessionStarted {
                    model: "gpt-5".into(),
                },
            ),
            record_at(
                2,
                TranscriptEvent::UserMessage {
                    content: UserMessageContent::from("hello"),
                },
            ),
            record_at(
                3,
                TranscriptEvent::AssistantMessage {
                    content: "hi".into(),
                },
            ),
        ];

        let snapshot =
            project_session_restore_snapshot("s".into(), records, None).expect("linear snapshot");

        assert_eq!(snapshot.branch_id, ROOT_CONTEXT_BRANCH_ID);
        assert_eq!(snapshot.leaf_sequence, 3);
        assert!(matches!(
            snapshot.history.as_slice(),
            [HistoryItem::UserMessage { .. }, HistoryItem::AssistantText { text }] if text == "hi"
        ));
    }

    #[test]
    fn explicit_branch_inherits_parent_prefix_and_excludes_parent_after_fork_base() {
        let records = vec![
            record_at(
                1,
                TranscriptEvent::UserMessage {
                    content: UserMessageContent::from("root-before"),
                },
            ),
            record_at(
                2,
                TranscriptEvent::AssistantMessage {
                    content: "root-at-fork".into(),
                },
            ),
            record_at(
                3,
                TranscriptEvent::ContextBranchCreated {
                    branch_id: "feature".into(),
                    parent_branch_id: ROOT_CONTEXT_BRANCH_ID.into(),
                    base_sequence: 2,
                    label: Some("feature".into()),
                },
            ),
            record_at(
                4,
                TranscriptEvent::AssistantMessage {
                    content: "root-after-fork".into(),
                },
            ),
            branch_record_at(
                5,
                "feature",
                TranscriptEvent::UserMessage {
                    content: UserMessageContent::from("child-only"),
                },
            ),
        ];

        let snapshot = build_session_context_snapshot(
            "s".into(),
            records,
            None,
            SessionContextCursor {
                branch_id: Some("feature".into()),
                leaf_sequence: None,
            },
        )
        .expect("branch snapshot");

        assert_eq!(snapshot.branch_id, "feature");
        assert_eq!(
            snapshot
                .records
                .iter()
                .map(|record| record.sequence)
                .collect::<Vec<_>>(),
            vec![1, 2, 5]
        );
        assert!(
            snapshot
                .messages
                .iter()
                .all(|message| message.content != "root-after-fork")
        );
    }

    #[test]
    fn latest_context_checkout_affects_default_branch_selection() {
        let records = vec![
            record_at(
                1,
                TranscriptEvent::UserMessage {
                    content: UserMessageContent::from("root-before"),
                },
            ),
            record_at(
                2,
                TranscriptEvent::ContextBranchCreated {
                    branch_id: "feature".into(),
                    parent_branch_id: ROOT_CONTEXT_BRANCH_ID.into(),
                    base_sequence: 1,
                    label: None,
                },
            ),
            branch_record_at(
                3,
                "feature",
                TranscriptEvent::AssistantMessage {
                    content: "branch-visible".into(),
                },
            ),
            record_at(
                4,
                TranscriptEvent::ContextCheckout {
                    branch_id: "feature".into(),
                    leaf_sequence: 3,
                },
            ),
            record_at(
                5,
                TranscriptEvent::AssistantMessage {
                    content: "root-later".into(),
                },
            ),
        ];

        let snapshot = project_session_restore_snapshot("s".into(), records, None)
            .expect("default restore uses latest checkout");

        assert_eq!(snapshot.branch_id, "feature");
        assert_eq!(snapshot.leaf_sequence, 3);
        assert_eq!(
            snapshot
                .records
                .iter()
                .map(|record| record.sequence)
                .collect::<Vec<_>>(),
            vec![1, 3]
        );
    }

    #[test]
    fn invalid_branch_resolution_errors_fail_fast() {
        let unknown_branch_error = build_session_context_snapshot(
            "s".into(),
            vec![record_at(
                1,
                TranscriptEvent::UserMessage {
                    content: UserMessageContent::from("hello"),
                },
            )],
            None,
            SessionContextCursor {
                branch_id: Some("missing".into()),
                leaf_sequence: None,
            },
        )
        .expect_err("unknown branch should fail");
        assert!(
            unknown_branch_error
                .to_string()
                .contains("unknown context branch 'missing'")
        );

        let invalid_base_error = build_session_context_snapshot(
            "s".into(),
            vec![
                record_at(
                    1,
                    TranscriptEvent::UserMessage {
                        content: UserMessageContent::from("root"),
                    },
                ),
                record_at(
                    2,
                    TranscriptEvent::ContextBranchCreated {
                        branch_id: "feature".into(),
                        parent_branch_id: ROOT_CONTEXT_BRANCH_ID.into(),
                        base_sequence: 9,
                        label: None,
                    },
                ),
            ],
            None,
            SessionContextCursor {
                branch_id: None,
                leaf_sequence: None,
            },
        )
        .expect_err("invalid base should fail");
        assert!(
            invalid_base_error
                .to_string()
                .contains("base_sequence 9 is not resolvable")
        );

        let leaf_beyond_tip_error = build_session_context_snapshot(
            "s".into(),
            vec![
                record_at(
                    1,
                    TranscriptEvent::UserMessage {
                        content: UserMessageContent::from("root"),
                    },
                ),
                record_at(
                    2,
                    TranscriptEvent::ContextBranchCreated {
                        branch_id: "feature".into(),
                        parent_branch_id: ROOT_CONTEXT_BRANCH_ID.into(),
                        base_sequence: 1,
                        label: None,
                    },
                ),
                branch_record_at(
                    3,
                    "feature",
                    TranscriptEvent::AssistantMessage {
                        content: "child".into(),
                    },
                ),
                record_at(
                    5,
                    TranscriptEvent::AssistantMessage {
                        content: "root-later".into(),
                    },
                ),
            ],
            None,
            SessionContextCursor {
                branch_id: Some("feature".into()),
                leaf_sequence: Some(4),
            },
        )
        .expect_err("leaf beyond tip should fail");
        assert!(
            leaf_beyond_tip_error
                .to_string()
                .contains("requested leaf_sequence 4 exceeds tip 3 for branch 'feature'")
        );
    }

    #[test]
    fn branch_local_compaction_replay_stays_branch_local() {
        let records = vec![
            record_at(
                1,
                TranscriptEvent::UserMessage {
                    content: UserMessageContent::from("root-a"),
                },
            ),
            record_at(
                2,
                TranscriptEvent::AssistantMessage {
                    content: "root-b".into(),
                },
            ),
            record_at(
                3,
                TranscriptEvent::ContextBranchCreated {
                    branch_id: "feature".into(),
                    parent_branch_id: ROOT_CONTEXT_BRANCH_ID.into(),
                    base_sequence: 2,
                    label: None,
                },
            ),
            branch_record_at(
                4,
                "feature",
                TranscriptEvent::UserMessage {
                    content: UserMessageContent::from("child-a"),
                },
            ),
            branch_record_at(
                5,
                "feature",
                TranscriptEvent::AssistantMessage {
                    content: "child-b".into(),
                },
            ),
            branch_record_at(
                6,
                "feature",
                TranscriptEvent::ContextCompaction(ContextCompactionEvent {
                    outcome: "succeeded".into(),
                    summary: "child-summary".into(),
                    tail_start_index: 2,
                    original_history_items: 4,
                    retained_history_items: 2,
                    retired_source_spans: Vec::new(),
                    frame_identity_bindings: Vec::new(),
                    detail: None,
                }),
            ),
            record_at(
                7,
                TranscriptEvent::AssistantMessage {
                    content: "root-after-fork".into(),
                },
            ),
        ];

        let snapshot = build_session_context_snapshot(
            "s".into(),
            records,
            None,
            SessionContextCursor {
                branch_id: Some("feature".into()),
                leaf_sequence: None,
            },
        )
        .expect("branch compaction snapshot");

        assert!(matches!(
            snapshot.history.as_slice(),
            [HistoryItem::ContextSummary { text }, HistoryItem::UserMessage { content }, HistoryItem::AssistantText { text: child_text }]
                if text == "child-summary" && content.display_text() == "child-a" && child_text == "child-b"
        ));
    }

    #[test]
    fn list_context_branches_marks_current_branch_and_labels() {
        let records = vec![
            record_at(
                1,
                TranscriptEvent::UserMessage {
                    content: UserMessageContent::from("root"),
                },
            ),
            record_at(
                2,
                TranscriptEvent::ContextBranchCreated {
                    branch_id: "feature-a".into(),
                    parent_branch_id: ROOT_CONTEXT_BRANCH_ID.into(),
                    base_sequence: 1,
                    label: Some("Feature A".into()),
                },
            ),
            branch_record_at(
                3,
                "feature-a",
                TranscriptEvent::AssistantMessage {
                    content: "child".into(),
                },
            ),
        ];

        let branches = list_context_branches(&records, Some("feature-a")).expect("branches");

        assert_eq!(branches.len(), 2);
        assert_eq!(branches[0].branch_id, ROOT_CONTEXT_BRANCH_ID);
        assert_eq!(branches[1].branch_id, "feature-a");
        assert_eq!(branches[1].label.as_deref(), Some("Feature A"));
        assert_eq!(branches[1].tip_sequence, 3);
        assert!(branches[1].is_current);
        assert!(!branches[0].is_current);
    }

    #[test]
    fn runtime_snapshot_projection_collects_context_view_tree_evidence_and_compaction() {
        let records = vec![
            record_at(
                1,
                TranscriptEvent::SessionStarted {
                    model: "gpt-5".into(),
                },
            ),
            metadata_record_at(
                2,
                TranscriptEvent::ContextNodeCreated {
                    node_id: "child".into(),
                    parent_node_id: Some("root".into()),
                    label: Some("Task".into()),
                    purpose: Some("Projection test".into()),
                    block_ref: None,
                    source_ref: None,
                },
            ),
            metadata_record_at(
                3,
                TranscriptEvent::ContextNodeLifecycle {
                    node_id: "root".into(),
                    status: ContextNodeStatus::Inactive,
                },
            ),
            metadata_record_at(
                4,
                TranscriptEvent::ContextNodeLifecycle {
                    node_id: "child".into(),
                    status: ContextNodeStatus::Active,
                },
            ),
            record_at(
                5,
                TranscriptEvent::UserMessage {
                    content: UserMessageContent::from("Do not skip tests"),
                },
            ),
            record_at(
                6,
                TranscriptEvent::ToolCallStarted {
                    call_id: "call-1".into(),
                    name: "shell__exec".into(),
                    args: json!({"command":"cargo check"}),
                },
            ),
            record_at(
                7,
                TranscriptEvent::ToolCallFinished {
                    call_id: "call-1".into(),
                    name: "shell__exec".into(),
                    ok: true,
                    output: ToolResult::ok("shell__exec", json!({"stdout": "done", "status": 0})),
                },
            ),
            metadata_record_at(
                8,
                TranscriptEvent::FoldedOutputMetadata {
                    node_id: Some("child".into()),
                    output_id: "fold-1".into(),
                    output_kind: "shell_output".into(),
                    call_id: Some("call-1".into()),
                    tool_name: Some("shell__exec".into()),
                    stream: Some("stdout".into()),
                    content: Some("done".into()),
                    byte_count: Some(4),
                    line_count: Some(1),
                    truncated: Some(false),
                    shell_command: Some("cargo check".into()),
                    source_start_sequence: Some(7),
                    source_end_sequence: Some(7),
                    tool_ok: Some(true),
                    exit_status: Some(0),
                },
            ),
            metadata_record_at(
                9,
                TranscriptEvent::ContextViewOperationMetadata {
                    operation: "pin".into(),
                    node_id: Some("child".into()),
                    block_id: Some("block-seq-7-folded-output-fold-1".into()),
                    detail: None,
                },
            ),
            metadata_record_at(
                10,
                TranscriptEvent::ContextViewOperationMetadata {
                    operation: "open_detail".into(),
                    node_id: Some("child".into()),
                    block_id: Some("block-seq-7-folded-output-fold-1".into()),
                    detail: None,
                },
            ),
            record_at(
                11,
                TranscriptEvent::Evidence {
                    id: "ev-1".into(),
                    evidence_kind: EvidenceKind::Validation,
                    title: "cargo check".into(),
                    summary: "cargo check passed".into(),
                    detail: None,
                    source: EvidenceSource::Command {
                        command: "cargo check".into(),
                        status: Some(0),
                    },
                    tags: vec!["validation".into()],
                },
            ),
            metadata_record_at(
                12,
                TranscriptEvent::ContextSummaryArtifactMetadata {
                    node_id: "child".into(),
                    artifact_id: "sum-1".into(),
                    artifact_kind: "summary".into(),
                    version: Some(1),
                    summary: Some("Condensed context".into()),
                    source_node_id: Some("child".into()),
                    source_block_id: None,
                    source_start_sequence: Some(5),
                    source_end_sequence: Some(7),
                },
            ),
            record_at(
                13,
                TranscriptEvent::ContextCompaction(ContextCompactionEvent {
                    outcome: "succeeded".into(),
                    summary: "Earlier context retired".into(),
                    tail_start_index: 1,
                    original_history_items: 3,
                    retained_history_items: 3,
                    retired_source_spans: vec![ContextCompactionSourceSpan {
                        start_sequence: 5,
                        end_sequence: 5,
                    }],
                    frame_identity_bindings: Vec::new(),
                    detail: None,
                }),
            ),
        ];

        let child_sessions = vec![ChildSessionSummary {
            parent_session_id: "s".into(),
            parent_run_id: "run-1".into(),
            child_session_id: "child-session-1".into(),
            agent_name: "explorer".into(),
            status: "completed".into(),
            summary: "Looked up compile state".into(),
            timestamp_ms: 42,
        }];

        let projected = project_runtime_restore_snapshot(
            "s".into(),
            records,
            None,
            SessionContextCursor {
                branch_id: None,
                leaf_sequence: None,
            },
            &child_sessions,
        )
        .expect("runtime snapshot projection");

        assert_eq!(projected.snapshot.latest_model.as_deref(), Some("gpt-5"));
        assert_eq!(projected.snapshot.leaf_sequence, Some(13));
        assert_eq!(
            projected
                .snapshot
                .context_tree
                .active_node_id()
                .map(|id| id.as_str()),
            Some("child")
        );
        assert_eq!(projected.snapshot.evidence.len(), 1);
        assert_eq!(projected.snapshot.child_sessions.len(), 1);
        assert_eq!(projected.snapshot.folded_outputs.len(), 1);
        assert_eq!(projected.snapshot.compaction.retired_source_spans.len(), 1);
        assert_eq!(
            history_items_from_frames(&projected.protocol_frames),
            restore_session_history_projection(&projected.records)
        );
        assert_eq!(
            projected.protocol_frames,
            projected.snapshot.active_protocol_frames(),
            "runtime restore protocol frames must retain transcript frame identity"
        );
        let compaction_summary = projected
            .snapshot
            .frames
            .iter()
            .find(|frame| {
                matches!(
                    frame.protocol,
                    Some(crate::protocol_frames::ProtocolFrameItem::ContextSummary { .. })
                )
            })
            .expect("restored active compaction summary frame");
        assert_eq!(
            compaction_summary.provenance.source,
            RuntimeSource::SummaryArtifact
        );
        assert_eq!(compaction_summary.provenance.source_span, None);
        assert_eq!(
            projected
                .snapshot
                .active_context
                .open_detail_block_id
                .as_deref(),
            Some("block-seq-7-folded-output-fold-1")
        );
        assert_eq!(
            projected.snapshot.active_context.pinned_block_ids,
            vec!["block-seq-7-folded-output-fold-1".to_string()]
        );
        assert!(
            projected
                .snapshot
                .frames
                .iter()
                .any(|frame| frame.kind == RuntimeFrameKind::Summary
                    && frame.summary.as_deref() == Some("Condensed context"))
        );
        assert!(
            projected
                .snapshot
                .prompt_contributors
                .iter()
                .any(|contributor| contributor.contributor_id == "folded-outputs")
        );
    }

    #[test]
    fn session_protocol_frames_restore_history_compatibly() {
        let records = vec![
            record_at(
                1,
                TranscriptEvent::UserMessage {
                    content: UserMessageContent::from("question"),
                },
            ),
            record_at(
                2,
                TranscriptEvent::AssistantMessage {
                    content: "answer".into(),
                },
            ),
            record_at(
                3,
                TranscriptEvent::ToolCallStarted {
                    call_id: "call-1".into(),
                    name: "shell__exec".into(),
                    args: json!({"command":"cargo check"}),
                },
            ),
            record_at(
                4,
                TranscriptEvent::ToolCallFinished {
                    call_id: "call-1".into(),
                    name: "shell__exec".into(),
                    ok: true,
                    output: ToolResult::ok("shell__exec", json!({"status": 0})),
                },
            ),
        ];

        let history = restore_session_history_projection(&records);
        let frames = restore_session_protocol_frames_projection(&records);

        assert_eq!(history_items_from_frames(&frames), history);
    }

    #[test]
    fn legacy_incomplete_tool_group_is_removed_before_a_new_turn_is_appended() {
        let records = vec![
            record_at(
                1,
                TranscriptEvent::UserMessage {
                    content: UserMessageContent::from("interrupted prompt"),
                },
            ),
            record_at(
                2,
                TranscriptEvent::ToolCallStarted {
                    call_id: "call-1".into(),
                    name: "fs__read".into(),
                    args: json!({"path": "src/main.rs"}),
                },
            ),
        ];

        let mut restored = restore_session_history_projection(&records);
        assert_eq!(restored, vec![HistoryItem::user("interrupted prompt")]);

        restored.push(HistoryItem::user("new prompt"));
        let protocol = analyze_history_items(&restored, None).expect("new turn can be appended");
        assert!(!protocol.has_incomplete_tool_call_groups());
    }

    #[test]
    fn legacy_cancelled_tool_call_restores_as_terminal_output() {
        let records = vec![
            record_at(
                1,
                TranscriptEvent::UserMessage {
                    content: UserMessageContent::from("interrupted prompt"),
                },
            ),
            record_at(
                2,
                TranscriptEvent::ToolCallStarted {
                    call_id: "call-1".into(),
                    name: "fs__read".into(),
                    args: json!({"path": "src/main.rs"}),
                },
            ),
            record_at(
                3,
                TranscriptEvent::ToolCallCancelled {
                    call_id: "call-1".into(),
                    name: "fs__read".into(),
                },
            ),
        ];

        let mut restored = restore_session_history_projection(&records);
        assert!(
            matches!(restored.last(), Some(HistoryItem::ToolOutput { call_id, output_json })
            if call_id == "call-1" && output_json == r#"{"status":"cancelled","summary":"user cancelled"}"#)
        );
        restored.push(HistoryItem::user("new prompt"));
        crate::protocol_frames::validate_history_items_complete(&restored, None)
            .expect("cancelled legacy call is terminal");
    }

    #[test]
    fn cancelled_durable_multi_call_batch_restores_every_terminal_output() {
        let calls = vec![
            HistoryToolCall {
                call_id: "call-1".into(),
                name: "fs__read".into(),
                arguments_json: "{}".into(),
            },
            HistoryToolCall {
                call_id: "call-2".into(),
                name: "fs__read".into(),
                arguments_json: "{}".into(),
            },
        ];
        let records = vec![
            record_at(
                1,
                TranscriptEvent::AssistantToolCallBatch { text: None, calls },
            ),
            record_at(
                2,
                TranscriptEvent::ToolCallCancelled {
                    call_id: "call-1".into(),
                    name: "fs__read".into(),
                },
            ),
        ];

        let restored = restore_session_history_projection(&records);
        assert!(
            matches!(restored.first(), Some(HistoryItem::AssistantToolCalls { calls, .. }) if calls.len() == 2)
        );
        assert_eq!(
            restored
                .iter()
                .filter(|item| matches!(item, HistoryItem::ToolOutput { .. }))
                .count(),
            2
        );
        crate::protocol_frames::validate_history_items_complete(&restored, None)
            .expect("all cancelled batch calls have terminal outputs");
    }

    #[test]
    fn runtime_snapshot_projection_respects_branch_cursor() {
        let records = vec![
            record_at(
                1,
                TranscriptEvent::UserMessage {
                    content: UserMessageContent::from("root-before"),
                },
            ),
            record_at(
                2,
                TranscriptEvent::ContextBranchCreated {
                    branch_id: "feature".into(),
                    parent_branch_id: ROOT_CONTEXT_BRANCH_ID.into(),
                    base_sequence: 1,
                    label: Some("feature".into()),
                },
            ),
            branch_record_at(
                3,
                "feature",
                TranscriptEvent::AssistantMessage {
                    content: "feature-only".into(),
                },
            ),
            record_at(
                4,
                TranscriptEvent::AssistantMessage {
                    content: "root-after".into(),
                },
            ),
        ];

        let projected = project_runtime_restore_snapshot(
            "s".into(),
            records,
            None,
            SessionContextCursor {
                branch_id: Some("feature".into()),
                leaf_sequence: Some(3),
            },
            &[],
        )
        .expect("branch runtime snapshot");

        assert_eq!(projected.branch_id, "feature");
        assert_eq!(projected.leaf_sequence, 3);
        assert_eq!(projected.snapshot.active_context.branch_id, "feature");
        assert!(
            projected
                .snapshot
                .frames
                .iter()
                .any(|frame| frame.summary.as_deref() == Some("feature-only"))
        );
        assert!(
            projected
                .snapshot
                .frames
                .iter()
                .all(|frame| frame.summary.as_deref() != Some("root-after"))
        );
    }

    #[test]
    fn runtime_snapshot_marks_incomplete_current_turn_tool_group_as_protected() {
        let records = vec![
            record_at(
                1,
                TranscriptEvent::TurnStarted(TurnStartedEvent {
                    turn_id: 7,
                    intent: "chat".into(),
                    directive: "answer".into(),
                    validation_reminder: String::new(),
                }),
            ),
            record_at(
                2,
                TranscriptEvent::UserMessage {
                    content: UserMessageContent::from("question"),
                },
            ),
            record_at(
                3,
                TranscriptEvent::ToolCallStarted {
                    call_id: "call-1".into(),
                    name: "fs__read".into(),
                    args: json!({"path":"src/main.rs"}),
                },
            ),
        ];

        let projected = project_runtime_restore_snapshot(
            "s".into(),
            records,
            None,
            SessionContextCursor {
                branch_id: None,
                leaf_sequence: None,
            },
            &[],
        )
        .expect("runtime snapshot projection");

        let protected = &projected.snapshot.compaction.protected_frame_ids;
        assert_eq!(protected.len(), 3);
        assert!(projected.snapshot.frames.iter().any(|frame| {
            protected.contains(&frame.id) && frame.kind == RuntimeFrameKind::User
        }));
        assert!(projected.snapshot.frames.iter().any(|frame| {
            protected.contains(&frame.id) && frame.kind == RuntimeFrameKind::ToolCall
        }));
    }

    #[test]
    fn append_only_checkout_restore_scopes_branch_content_and_runtime_metadata() {
        let records = vec![
            record_at(
                1,
                TranscriptEvent::UserMessage {
                    content: UserMessageContent::from("root-prefix"),
                },
            ),
            record_at(
                2,
                TranscriptEvent::AssistantMessage {
                    content: "root-fork-base".into(),
                },
            ),
            metadata_record_at(
                3,
                TranscriptEvent::ContextBranchCreated {
                    branch_id: "child".into(),
                    parent_branch_id: ROOT_CONTEXT_BRANCH_ID.into(),
                    base_sequence: 2,
                    label: Some("child lane".into()),
                },
            ),
            metadata_record_at(
                4,
                TranscriptEvent::ContextCheckout {
                    branch_id: "child".into(),
                    leaf_sequence: 2,
                },
            ),
            branch_record_at(
                5,
                "child",
                TranscriptEvent::UserMessage {
                    content: UserMessageContent::from("child-only"),
                },
            ),
            branch_record_at(
                6,
                "child",
                TranscriptEvent::AssistantMessage {
                    content: "child-reply".into(),
                },
            ),
            metadata_record_at(
                7,
                TranscriptEvent::ContextNodeCreated {
                    node_id: "child-node".into(),
                    parent_node_id: Some("root".into()),
                    label: Some("Child".into()),
                    purpose: None,
                    block_ref: None,
                    source_ref: None,
                },
            ),
            metadata_record_at(
                8,
                TranscriptEvent::ContextNodeLifecycle {
                    node_id: "root".into(),
                    status: ContextNodeStatus::Inactive,
                },
            ),
            metadata_record_at(
                9,
                TranscriptEvent::ContextNodeLifecycle {
                    node_id: "child-node".into(),
                    status: ContextNodeStatus::Active,
                },
            ),
            metadata_record_at(
                10,
                TranscriptEvent::ContextViewOperationMetadata {
                    operation: "pin".into(),
                    node_id: Some("child-node".into()),
                    block_id: Some("block-seq-6-note".into()),
                    detail: None,
                },
            ),
            metadata_record_at(
                11,
                TranscriptEvent::ContextCheckout {
                    branch_id: ROOT_CONTEXT_BRANCH_ID.into(),
                    leaf_sequence: 2,
                },
            ),
            record_at(
                12,
                TranscriptEvent::AssistantMessage {
                    content: "root-after-fork".into(),
                },
            ),
            metadata_record_at(
                13,
                TranscriptEvent::ContextNodeCreated {
                    node_id: "root-after-node".into(),
                    parent_node_id: Some("root".into()),
                    label: Some("Root after fork".into()),
                    purpose: None,
                    block_ref: None,
                    source_ref: None,
                },
            ),
            metadata_record_at(
                14,
                TranscriptEvent::ContextNodeLifecycle {
                    node_id: "root".into(),
                    status: ContextNodeStatus::Inactive,
                },
            ),
            metadata_record_at(
                15,
                TranscriptEvent::ContextNodeLifecycle {
                    node_id: "root-after-node".into(),
                    status: ContextNodeStatus::Active,
                },
            ),
            metadata_record_at(
                16,
                TranscriptEvent::ContextViewOperationMetadata {
                    operation: "pin".into(),
                    node_id: Some("root-after-node".into()),
                    block_id: Some("block-seq-12-note".into()),
                    detail: None,
                },
            ),
            metadata_record_at(
                17,
                TranscriptEvent::ContextCheckout {
                    branch_id: "child".into(),
                    leaf_sequence: 6,
                },
            ),
            metadata_record_at(
                18,
                TranscriptEvent::ContextNodeLifecycle {
                    node_id: "root-after-node".into(),
                    status: ContextNodeStatus::Inactive,
                },
            ),
            metadata_record_at(
                19,
                TranscriptEvent::ContextNodeLifecycle {
                    node_id: "child-node".into(),
                    status: ContextNodeStatus::Active,
                },
            ),
            metadata_record_at(
                20,
                TranscriptEvent::ContextViewOperationMetadata {
                    operation: "open_detail".into(),
                    node_id: Some("child-node".into()),
                    block_id: Some("block-seq-6-note".into()),
                    detail: None,
                },
            ),
        ];
        let original_input = serde_json::to_value(&records).expect("serialize input records");
        let project = |branch_id| {
            project_runtime_restore_snapshot(
                "s".into(),
                records.clone(),
                None,
                SessionContextCursor {
                    branch_id,
                    leaf_sequence: None,
                },
                &[],
            )
            .expect("project branch cursor")
        };
        let root = project(Some(ROOT_CONTEXT_BRANCH_ID.into()));
        let child = project(Some("child".into()));
        let latest = project(None);
        let contents = |projected: &RuntimeRestoreSnapshot| {
            projected
                .records
                .iter()
                .filter_map(|record| match &record.event {
                    TranscriptEvent::UserMessage { content } => Some(content.display_text()),
                    TranscriptEvent::AssistantMessage { content } => Some(content.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>()
        };

        assert_eq!(
            contents(&root),
            ["root-prefix", "root-fork-base", "root-after-fork"]
        );
        assert_eq!(
            contents(&child),
            ["root-prefix", "root-fork-base", "child-only", "child-reply"]
        );
        assert_eq!(latest.branch_id, "child");
        assert_eq!(contents(&latest), contents(&child));
        assert_eq!(
            child.snapshot.active_context.parent_branch_id.as_deref(),
            Some(ROOT_CONTEXT_BRANCH_ID)
        );
        assert_eq!(root.snapshot.context_scope_revision, 11);
        assert_eq!(child.snapshot.context_scope_revision, 17);
        assert_eq!(latest.snapshot.context_scope_revision, 17);
        assert_eq!(
            root.snapshot
                .context_tree
                .active_node_id()
                .map(|id| id.as_str()),
            Some("root-after-node")
        );
        assert_eq!(
            child
                .snapshot
                .context_tree
                .active_node_id()
                .map(|id| id.as_str()),
            Some("child-node")
        );
        assert_eq!(
            root.snapshot.active_context.pinned_block_ids,
            ["block-seq-12-note"]
        );
        assert_eq!(
            child
                .snapshot
                .active_context
                .open_detail_block_id
                .as_deref(),
            Some("block-seq-6-note")
        );
        assert_eq!(
            latest.snapshot.active_context.open_detail_block_id,
            child.snapshot.active_context.open_detail_block_id
        );
        assert_eq!(
            serde_json::to_value(&records).expect("serialize records after projection"),
            original_input
        );
    }
}
