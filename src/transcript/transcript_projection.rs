use crate::agent::ConversationMessage;
use crate::context_view::{self, ContextViewProjection};
use crate::evidence::EvidenceRecord;
use crate::protocol_frames::{ProtocolFrame, history_items_to_frames};
use crate::request_builder::HistoryItem;
use crate::runtime_context::{
    FrameVisibility, RuntimeFrame, RuntimeFrameKind, RuntimeSnapshot, RuntimeSource,
};
use crate::transcript::{ChildSessionSummary, TranscriptEvent, TranscriptRecord};
use anyhow::{anyhow, ensure};
use std::collections::BTreeSet;

#[cfg(test)]
use crate::context_tree::ContextNodeId;
#[cfg(test)]
use crate::protocol_frames::analyze_history_items;
use crate::transcript::ROOT_CONTEXT_BRANCH_ID;
#[cfg(test)]
use crate::transcript::{
    LogicalCheckpointAuditSourceV1, LogicalCheckpointEventV1, LogicalCheckpointSourceSpanV1,
};

#[path = "transcript_projection/context_tree.rs"]
mod context_tree;

pub(crate) use context_tree::{project_context_tree, replay_context_tree};

#[path = "transcript_projection/job_board.rs"]
mod job_board;

pub(crate) use job_board::{project_child_session_summaries, project_job_board};

#[path = "transcript_projection/branch.rs"]
mod branch;

pub(crate) use branch::{effective_branch_id_at_frontier, list_context_branches};

use branch::{
    branch_parent_id, branch_tip_for_records, build_branch_index, collect_branch_path_records,
    resolve_active_branch_id, resolve_branch_context,
};

#[path = "transcript_projection/history.rs"]
mod history;

pub(crate) use history::{
    restore_latest_model_projection, restore_max_turn_id_projection,
    restore_session_history_projection,
};

use history::{
    active_turn_id_from_lifecycle_records, active_turn_segment_from_lifecycle_records,
    checkpoint_spans_from_history, checkpoint_spans_to_compaction, restore_history_projection,
};

#[path = "transcript_projection/checkpoint.rs"]
mod checkpoint;

#[cfg(test)]
pub(crate) use checkpoint::prepare_logical_checkpoint_candidate;
pub(crate) use checkpoint::validate_logical_checkpoint_candidate;

use checkpoint::validate_logical_checkpoint_record;

#[path = "transcript_projection/runtime.rs"]
mod runtime;

use runtime::{runtime_projection_records, runtime_snapshot_from_resolved_context_unbound};

#[cfg(test)]
use runtime::snapshot_for_context_view_for_test as snapshot_for_context_view;

#[cfg(test)]
use checkpoint::{CoveredCallGroup, validate_current_fact_provenance};

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
}

impl SessionRestoreSnapshot {
    pub(crate) fn evidence_count(&self) -> usize {
        self.evidence.len()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextBranchInfo {
    pub branch_id: String,
    pub parent_branch_id: Option<String>,
    pub label: Option<String>,
    pub tip_sequence: u64,
    pub is_current: bool,
}

pub(crate) fn project_session_restore_snapshot(
    session_id: String,
    records: Vec<TranscriptRecord>,
) -> anyhow::Result<SessionRestoreSnapshot> {
    build_session_context_snapshot(
        session_id,
        records,
        SessionContextCursor {
            branch_id: None,
            leaf_sequence: None,
        },
    )
}

pub(crate) fn project_context_view(
    records: &[TranscriptRecord],
) -> anyhow::Result<ContextViewProjection> {
    context_view::project_context_view(records)
}

#[derive(Debug)]
struct ResolvedBranchContext {
    branch_id: String,
    leaf_sequence: u64,
    scope_checkout_sequence: Option<u64>,
    records: Vec<TranscriptRecord>,
}

/// Immutable, branch-aware input for validating one pre-append compaction
/// candidate.  The selected history deliberately remains separate from the
/// complete journal and its metadata projection.
pub(crate) struct ContextCompactionValidationScope {
    journal_records: Vec<TranscriptRecord>,
    pub(crate) expected_frontier: u64,
    resolved: ResolvedBranchContext,
    actual_append_branch_id: Option<String>,
}

impl ContextCompactionValidationScope {
    pub(crate) fn selected_history_records(&self) -> &[TranscriptRecord] {
        &self.resolved.records
    }

    /// The sole branch scope for both candidate replay and the durable record.
    /// Root content retains the journal's canonical global (`None`) scope.
    pub(crate) fn actual_append_branch_id(&self) -> &Option<String> {
        &self.actual_append_branch_id
    }
}

pub(crate) fn context_compaction_validation_scope(
    records: &[TranscriptRecord],
    expected_frontier: u64,
    cursor: SessionContextCursor,
) -> anyhow::Result<ContextCompactionValidationScope> {
    let journal_records = records
        .iter()
        .filter(|record| record.sequence <= expected_frontier)
        .cloned()
        .collect::<Vec<_>>();
    ensure!(
        journal_records
            .iter()
            .map(|record| record.sequence)
            .max()
            .unwrap_or(0)
            == expected_frontier,
        "context compaction journal frontier does not match committed transcript"
    );
    let resolved = resolve_branch_context(journal_records.clone(), cursor.clone())?;
    let actual_append_branch_id =
        (resolved.branch_id != ROOT_CONTEXT_BRANCH_ID).then(|| resolved.branch_id.clone());
    Ok(ContextCompactionValidationScope {
        journal_records,
        expected_frontier,
        resolved,
        actual_append_branch_id,
    })
}

pub(crate) fn active_turn_id_at_context_cursor(
    records: Vec<TranscriptRecord>,
    cursor: SessionContextCursor,
) -> anyhow::Result<Option<u64>> {
    let resolved = resolve_branch_context(records, cursor)?;
    Ok(active_turn_id_from_lifecycle_records(&resolved.records))
}

pub(crate) fn build_session_context_snapshot(
    session_id: String,
    records: Vec<TranscriptRecord>,
    cursor: SessionContextCursor,
) -> anyhow::Result<SessionRestoreSnapshot> {
    let resolved = resolve_branch_context(records.clone(), cursor)?;
    validate_projection_events(
        &session_id,
        &records,
        &runtime_projection_records(&records, &resolved),
        &resolved.branch_id,
    )?;
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
    })
}

pub(crate) fn project_runtime_restore_snapshot(
    session_id: String,
    records: Vec<TranscriptRecord>,
    cursor: SessionContextCursor,
    child_sessions: &[ChildSessionSummary],
) -> anyhow::Result<RuntimeRestoreSnapshot> {
    let resolved = resolve_branch_context(records.clone(), cursor)?;
    validate_projection_events(
        &session_id,
        &records,
        &runtime_projection_records(&records, &resolved),
        &resolved.branch_id,
    )?;
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
    })
}

pub(crate) fn restore_session_protocol_frames_projection(
    records: &[TranscriptRecord],
) -> anyhow::Result<Vec<ProtocolFrame>> {
    validate_context_projection_events(records)?;
    Ok(history_items_to_frames(
        &restore_session_history_projection(records),
    ))
}

fn runtime_snapshot_from_resolved_context(
    session_id: &str,
    all_records: &[TranscriptRecord],
    resolved: &ResolvedBranchContext,
    latest_model: Option<&str>,
    child_sessions: &[ChildSessionSummary],
) -> anyhow::Result<RuntimeSnapshot> {
    runtime_snapshot_from_resolved_context_unbound(
        session_id,
        all_records,
        resolved,
        latest_model,
        child_sessions,
    )
}

/// Builds the canonical runtime frame projection without durable ID bindings.
/// A checkout establishes a durable cursor scope even when its branch later
/// advances. Sequence zero keeps old transcripts deterministic.
fn context_scope_revision(_records: &[TranscriptRecord], resolved: &ResolvedBranchContext) -> u64 {
    resolved.scope_checkout_sequence.unwrap_or(0)
}
/// Metadata is append-only and normally has no branch id. Its ownership is
/// therefore bounded by the checkout that selected the active scope and the
/// next checkout that replaces it.
/// Reconstruct the pre-compaction protocol frames from the append-only journal.
///
/// We deliberately derive this from the pre-event record slice rather than from
/// frame indexes stored in the event. That keeps provenance stable across restarts
/// and lets validation reject stale or forged event boundaries.
pub(crate) fn validate_successful_compactions(records: &[TranscriptRecord]) -> anyhow::Result<()> {
    for (index, record) in records.iter().enumerate() {
        if let TranscriptEvent::ContextCompaction(event) = &record.event {
            let scope = context_compaction_validation_scope(
                &records[..index],
                record.sequence.saturating_sub(1),
                SessionContextCursor {
                    branch_id: Some(
                        record
                            .context_branch_id
                            .clone()
                            .unwrap_or_else(|| ROOT_CONTEXT_BRANCH_ID.to_string()),
                    ),
                    leaf_sequence: None,
                },
            )?;
            validate_context_compaction_event_in_scope(&scope, event)?;
        }
    }
    Ok(())
}

/// Context-only projections validate their complete journal before applying
/// replacement events. Selected branch content must use the explicit
/// unvalidated projection path after its complete journal has been validated.
pub(crate) fn validate_context_projection_events(
    records: &[TranscriptRecord],
) -> anyhow::Result<()> {
    let branch_id = resolve_branch_context(
        records.to_vec(),
        SessionContextCursor {
            branch_id: None,
            leaf_sequence: None,
        },
    )?
    .branch_id;
    validate_projection_events("", records, records, &branch_id)
}

/// Validate replacement events in transcript order.  A later replacement may
/// depend on an earlier one, but must never make an earlier malformed event
/// appear valid.
fn validate_projection_events(
    session_id: &str,
    all_records: &[TranscriptRecord],
    visible: &[TranscriptRecord],
    selected_branch_id: &str,
) -> anyhow::Result<()> {
    for record in visible {
        match &record.event {
            TranscriptEvent::ContextCompaction(event) => {
                let scope = context_compaction_validation_scope(
                    all_records,
                    record.sequence.saturating_sub(1),
                    SessionContextCursor {
                        branch_id: record
                            .context_branch_id
                            .clone()
                            .or_else(|| Some(selected_branch_id.to_string())),
                        leaf_sequence: None,
                    },
                )?;
                validate_context_compaction_event_in_scope(&scope, event)?;
            }
            TranscriptEvent::LogicalCheckpoint(event) => {
                validate_logical_checkpoint_record(session_id, all_records, record, event)?;
            }
            _ => {}
        }
    }
    Ok(())
}
pub(crate) fn canonical_compaction_tail_start(
    records: &[TranscriptRecord],
    requested: usize,
) -> anyhow::Result<usize> {
    // Retained for tolerant legacy callers. New compaction events compare the
    // requested index against this exact normalization instead of adopting it.
    Ok(history::normalize_compaction_tail_start(
        &restore_history_projection(records),
        requested,
    ))
}

pub(crate) fn validate_context_compaction_event(
    records: &[TranscriptRecord],
    event: &crate::agent::ContextCompactionEvent,
) -> anyhow::Result<()> {
    let frontier = records
        .iter()
        .map(|record| record.sequence)
        .max()
        .unwrap_or(0);
    let scope = context_compaction_validation_scope(
        records,
        frontier,
        SessionContextCursor {
            branch_id: None,
            leaf_sequence: None,
        },
    )?;
    validate_context_compaction_event_in_scope(&scope, event)
}

pub(crate) fn validate_context_compaction_event_in_scope(
    scope: &ContextCompactionValidationScope,
    event: &crate::agent::ContextCompactionEvent,
) -> anyhow::Result<()> {
    ensure!(
        !event.summary.trim().is_empty(),
        "context compaction summary must not be empty"
    );

    let history = restore_history_projection(scope.selected_history_records());
    let normalized = history::normalize_compaction_tail_start(&history, event.tail_start_index);

    if let Some(checkpoint) = &event.checkpoint {
        crate::agent::compaction::validate_checkpoint_sections(&event.summary)?;
        let next_action = crate::agent::compaction::checkpoint_first_next_step(&event.summary)?;
        ensure!(
            checkpoint.next_action == next_action,
            "context compaction checkpoint next_action must equal the exact first Next Steps item"
        );
        let expected_file_operations =
            crate::agent::compaction::checkpoint_file_operations(&event.summary)?;
        ensure!(
            checkpoint.file_operations == expected_file_operations,
            "context compaction checkpoint file operations must equal the summary metadata (checkpoint {:?}, summary {:?})",
            checkpoint.file_operations,
            expected_file_operations,
        );

        let active_turn_id =
            history::active_turn_id_from_lifecycle_records(scope.selected_history_records());
        let preserved_user_index = active_turn_id
            .and_then(|turn_id| {
                history.iter().position(|entry| {
                    entry.turn_id == Some(turn_id)
                        && matches!(
                            entry.item,
                            crate::request_builder::HistoryItem::UserMessage { .. }
                        )
                })
            })
            .filter(|index| *index < normalized);
        let compacted_prefix = history
            .iter()
            .enumerate()
            .take(normalized)
            .filter(|(index, _)| Some(*index) != preserved_user_index)
            .map(|(_, entry)| entry.item.clone())
            .collect::<Vec<_>>();
        let expected_handoff = preserved_user_index
            .map(|_| {
                crate::agent::compaction::render_split_turn_handoff(
                    &event.summary,
                    &compacted_prefix,
                    &next_action,
                )
            })
            .transpose()?;
        ensure!(
            checkpoint.split_turn_handoff == expected_handoff,
            "context compaction checkpoint split-turn handoff does not match the compacted prefix"
        );
        let expected_continuation = crate::agent::compaction::render_internal_continuation(
            &next_action,
            expected_handoff.as_deref(),
        );
        ensure!(
            checkpoint.continuation == expected_continuation,
            "context compaction checkpoint continuation does not match its durable next action and handoff"
        );
    }

    // New records have one authoritative boundary. It must exactly equal the
    // replay normalization: clamp, canonicalize tool groups, then retain every
    // incomplete group. Legacy records bypass this strict validator and use the
    // same normalization from history projection.
    ensure!(
        event.tail_start_index == normalized,
        "context compaction tail_start_index must equal the canonical incomplete-safe boundary (requested {}, canonical {})",
        event.tail_start_index,
        normalized,
    );
    Ok(())
}

pub(crate) fn validate_context_compaction_candidate_replay(
    session_id: &str,
    scope: &ContextCompactionValidationScope,
    event: &crate::agent::ContextCompactionEvent,
) -> anyhow::Result<()> {
    let sequence = scope
        .expected_frontier
        .checked_add(1)
        .ok_or_else(|| anyhow!("context compaction journal frontier overflow"))?;
    let mut candidate_records = scope.journal_records.clone();
    candidate_records.push(TranscriptRecord {
        session_id: session_id.to_string(),
        sequence,
        timestamp_ms: 0,
        context_branch_id: scope.actual_append_branch_id.clone(),
        event: TranscriptEvent::ContextCompaction(event.clone()),
    });
    project_runtime_restore_snapshot(
        session_id.to_string(),
        candidate_records,
        SessionContextCursor {
            branch_id: Some(scope.resolved.branch_id.clone()),
            leaf_sequence: Some(sequence),
        },
        &[],
    )?;
    Ok(())
}

pub(crate) fn sanitize_compaction_summary_body(summary: &str) -> String {
    let trimmed = summary.trim();
    if trimmed.is_empty() {
        "[empty compaction summary]".to_string()
    } else {
        trimmed.to_string()
    }
}

#[cfg(test)]
#[path = "transcript_projection/tests.rs"]
mod tests;
