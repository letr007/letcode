use crate::agent::ConversationMessage;
#[cfg(test)]
use crate::context_view::{self, ContextViewProjection};
use crate::evidence::EvidenceRecord;
use crate::protocol_frames::ProtocolFrame;
#[cfg(test)]
use crate::protocol_frames::history_items_to_frames;
use crate::request_builder::HistoryItem;
use crate::runtime_context::RuntimeSnapshot;
use crate::transcript::{ChildSessionSummary, TranscriptEvent, TranscriptRecord};
use crate::workflow_state::WorkflowState;
use anyhow::ensure;

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

#[cfg(test)]
pub(crate) use context_tree::project_context_tree;
pub(crate) use context_tree::replay_context_tree;

#[path = "transcript_projection/job_board.rs"]
mod job_board;

pub(crate) use job_board::project_child_session_summaries;
pub(crate) use job_board::project_job_board;

#[path = "transcript_projection/branch.rs"]
mod branch;

pub(crate) use branch::effective_branch_id_at_frontier;

#[cfg(test)]
pub(crate) use branch::list_context_branches;

#[path = "transcript_projection/session_tree.rs"]
mod session_tree;

#[cfg(test)]
pub(crate) use session_tree::HistoryNavigationState;
pub(crate) use session_tree::{
    SessionHistoryEntry, SessionHistoryEntryKind, history_navigation_state,
    project_session_history_tree,
};

use branch::{
    branch_parent_id, branch_tip_for_records, build_branch_index, collect_branch_path_records,
    resolve_active_branch_id, resolve_branch_context,
};

#[path = "transcript_projection/history.rs"]
mod history;

pub(crate) use history::{
    restore_latest_model_projection, restore_latest_permission_mode_projection,
    restore_latest_reasoning_effort_projection, restore_max_turn_id_projection,
    restore_session_history_projection,
};

use history::{
    active_turn_id_from_lifecycle_records, active_turn_segment_from_lifecycle_records,
    checkpoint_spans_from_history, restore_history_projection,
};

#[path = "transcript_projection/checkpoint.rs"]
mod checkpoint;

#[cfg(test)]
pub(crate) use checkpoint::prepare_logical_checkpoint_candidate;

#[cfg(test)]
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
    #[allow(dead_code)] // Retained for serde-compatible session restore snapshots.
    pub session_id: String,
    #[allow(dead_code)] // Retained for branch-aware restore diagnostics.
    pub branch_id: String,
    #[allow(dead_code)] // Retained as part of the durable navigation cursor.
    pub leaf_sequence: u64,
    pub records: Vec<TranscriptRecord>,
    #[allow(dead_code)] // Retained for the complete session restore projection.
    pub messages: Vec<ConversationMessage>,
    #[allow(dead_code)] // Retained for the complete session restore projection.
    pub history: Vec<HistoryItem>,
    #[allow(dead_code)] // Retained for evidence-aware session restoration.
    pub evidence: Vec<EvidenceRecord>,
    #[allow(dead_code)] // Retained for model restoration compatibility.
    pub latest_model: Option<String>,
    #[allow(dead_code)] // Retained for turn-sequence restoration compatibility.
    pub max_turn_id: u64,
}

#[derive(Debug, Clone)]
pub(crate) struct RuntimeRestoreSnapshot {
    #[allow(dead_code)] // Retained for serde-compatible runtime restore snapshots.
    pub session_id: String,
    pub branch_id: String,
    #[allow(dead_code)] // Retained as part of the durable navigation cursor.
    pub leaf_sequence: u64,
    pub records: Vec<TranscriptRecord>,
    #[allow(dead_code)] // Retained for serde-compatible runtime restore snapshots.
    pub protocol_frames: Vec<ProtocolFrame>,
    pub snapshot: RuntimeSnapshot,
    pub latest_model: Option<String>,
    pub latest_permission_mode: Option<String>,
    pub max_turn_id: u64,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextBranchInfo {
    pub branch_id: String,
    pub parent_branch_id: Option<String>,
    pub label: Option<String>,
    pub tip_sequence: u64,
    pub is_current: bool,
}

#[cfg(test)]
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

#[cfg(test)]
pub(crate) fn project_context_view(
    records: &[TranscriptRecord],
) -> anyhow::Result<ContextViewProjection> {
    context_view::project_context_view(records)
}

#[allow(dead_code)] // Production transcript validation uses this branch-aware projection path.
pub(crate) fn project_context_tree_for_active_branch(
    records: &[TranscriptRecord],
    current_branch_id: Option<&str>,
) -> anyhow::Result<crate::context_tree::ContextTreeState> {
    let resolved = resolve_branch_context(
        records.to_vec(),
        SessionContextCursor {
            branch_id: current_branch_id.map(str::to_owned),
            leaf_sequence: None,
        },
    )?;
    replay_context_tree(&runtime_projection_records(records, &resolved))
}

#[derive(Debug)]
struct ResolvedBranchContext {
    branch_id: String,
    leaf_sequence: u64,
    scope_checkout_sequence: Option<u64>,
    records: Vec<TranscriptRecord>,
}

/// Immutable, branch-aware input for validating one pre-append compaction.
pub(crate) struct ContextCompactionValidationScope {
    resolved: ResolvedBranchContext,
    actual_append_branch_id: Option<String>,
}

impl ContextCompactionValidationScope {
    pub(crate) fn selected_history_records(&self) -> &[TranscriptRecord] {
        &self.resolved.records
    }

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
    )?;
    let latest_model = restore_latest_model_projection(&resolved.records);
    let latest_permission_mode = restore_latest_permission_mode_projection(&resolved.records);
    // Keep allocation global to this session while all active state below stays
    // scoped to the resolved branch and leaf.
    let max_turn_id = restore_max_turn_id_projection(&records);
    let mut snapshot = runtime_snapshot_from_resolved_context(
        &session_id,
        &records,
        &resolved,
        latest_model.as_deref(),
        child_sessions,
    )?;
    snapshot.workflow = project_workflow_state(&resolved.records).state;
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
        latest_permission_mode,
        max_turn_id,
    })
}

pub(crate) struct WorkflowProjection {
    pub state: WorkflowState,
    pub has_todos: bool,
    pub has_auto_continue: bool,
}

pub(crate) fn project_workflow_state(records: &[TranscriptRecord]) -> WorkflowProjection {
    let mut projection = WorkflowProjection {
        state: WorkflowState::default(),
        has_todos: false,
        has_auto_continue: false,
    };
    for record in records {
        match &record.event {
            TranscriptEvent::UserMessage { .. }
            | TranscriptEvent::TurnStarted(_)
            | TranscriptEvent::TurnInterrupted { .. }
            | TranscriptEvent::Error { .. } => {
                projection.state = WorkflowState::default();
                projection.has_todos = false;
                projection.has_auto_continue = false;
            }
            TranscriptEvent::TodoSnapshot { items } => {
                projection.state.todos = items.clone();
                projection.has_todos = true;
            }
            TranscriptEvent::AutoContinueChanged { state } => {
                projection.state.auto_continue = state.clone();
                projection.has_auto_continue = true;
            }
            _ => {}
        }
    }
    projection
}

#[cfg(test)]
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
/// Context-only projections validate their complete journal before applying
/// replacement events. Selected branch content must use the explicit
/// unvalidated projection path after its complete journal has been validated.
pub(crate) fn validate_context_projection_events(
    records: &[TranscriptRecord],
) -> anyhow::Result<()> {
    resolve_branch_context(
        records.to_vec(),
        SessionContextCursor {
            branch_id: None,
            leaf_sequence: None,
        },
    )?;
    validate_projection_events("", records, records)
}

/// Validate replacement events in transcript order.  A later replacement may
/// depend on an earlier one, but must never make an earlier malformed event
/// appear valid.
fn validate_projection_events(
    session_id: &str,
    all_records: &[TranscriptRecord],
    visible: &[TranscriptRecord],
) -> anyhow::Result<()> {
    for record in visible {
        match &record.event {
            TranscriptEvent::ContextCompaction(event) => {
                // A compaction is validated against the branch it was committed
                // on. Root compactions carry `context_branch_id: None` even when
                // a later restore selects another branch, so they re-validate
                // against the root scope rather than the selected branch.
                let scope = context_compaction_validation_scope(
                    all_records,
                    record.sequence.saturating_sub(1),
                    SessionContextCursor {
                        branch_id: record
                            .context_branch_id
                            .clone()
                            .or_else(|| Some(ROOT_CONTEXT_BRANCH_ID.to_string())),
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
pub(crate) fn validate_context_compaction_event_in_scope(
    scope: &ContextCompactionValidationScope,
    event: &crate::agent::ContextCompactionEvent,
) -> anyhow::Result<()> {
    ensure!(
        !event.summary.trim().is_empty(),
        "context compaction summary must not be empty"
    );

    // Legacy records have an index boundary or checkpoint and replay as
    // recorded. A record with neither is modern, including full compactions
    // whose `first_kept_entry_id` is `None`.
    let is_legacy = event.tail_start_index.is_some() || event.checkpoint.is_some();
    if !is_legacy && let Some(first_kept_entry_id) = event.first_kept_entry_id.as_deref() {
        ensure!(
            !first_kept_entry_id.trim().is_empty(),
            "modern context compaction first_kept_entry_id must not be empty"
        );
        let history = restore_history_projection(scope.selected_history_records());
        ensure!(
            history
                .iter()
                .any(|entry| entry.stable_key == first_kept_entry_id),
            "context compaction first_kept_entry_id '{}' is absent from the pre-compaction projection",
            first_kept_entry_id
        );
    }

    Ok(())
}

pub(crate) fn sanitize_compaction_summary_body(summary: &str) -> anyhow::Result<String> {
    let trimmed = summary.trim();
    ensure!(
        !trimmed.is_empty(),
        "context compaction summary must not be empty"
    );
    Ok(trimmed.to_string())
}

#[cfg(test)]
#[path = "transcript_projection/tests.rs"]
mod tests;
